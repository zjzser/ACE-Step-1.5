"""Tests for the developer inference CLI helpers."""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

from dev_infer.cli import (
    _apply_description,
    _build_baseline,
    _build_resolved_example_payload,
    _parser,
)
from dev_infer.runtime import (
    ORIGINAL_EXAMPLE_FILENAME,
    RESOLVED_EXAMPLE_FILENAME,
    _load_full_sft_weights,
    _normalize_full_sft_state_dict,
    rename_output_audios,
    save_run_artifacts,
    supported_tasks_for_model,
    validate_baseline_task,
)
from dev_infer.schemas import BaselineConfig, RunConfig
from dev_infer.tasks import TASK_BUILDERS, validate_run


class TaskValidationTests(unittest.TestCase):
    """Task validation tests."""

    def test_text2music_needs_caption_or_lyrics(self):
        run = RunConfig(task="text2music")
        with self.assertRaises(ValueError):
            validate_run(run)

    def test_cover_needs_src_audio(self):
        run = RunConfig(task="cover", caption="jazz cover")
        with self.assertRaises(ValueError):
            validate_run(run)

    def test_repaint_checks_range(self):
        run = RunConfig(
            task="repaint",
            caption="replace with piano",
            src_audio="/tmp/a.wav",
            repainting_start=12.0,
            repainting_end=10.0,
        )
        with self.assertRaises(ValueError):
            validate_run(run)

    def test_supported_task_builders_are_registered(self):
        self.assertEqual(set(TASK_BUILDERS), {"text2music", "cover", "repaint"})


class BaselineSupportTests(unittest.TestCase):
    """Baseline capability tests."""

    def test_turbo_supports_three_tasks(self):
        self.assertEqual(
            supported_tasks_for_model("acestep-v15-turbo"),
            {"text2music", "cover", "repaint"},
        )

    def test_base_supports_extract_family(self):
        supported = supported_tasks_for_model("acestep-v15-base")
        self.assertIn("extract", supported)
        self.assertIn("lego", supported)
        self.assertIn("complete", supported)

    def test_validate_baseline_task_rejects_unsupported_pair(self):
        baseline = BaselineConfig(
            kind="base",
            project_root=Path("."),
            checkpoint_dir=Path("./checkpoints"),
            config_path="acestep-v15-sft",
            save_dir=Path("./outputs"),
        )
        with self.assertRaises(ValueError):
            validate_baseline_task(baseline, "extract")


class DescriptionMergeTests(unittest.TestCase):
    """Description loading tests."""

    def test_description_fills_missing_fields(self):
        parser = _parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            description_path = Path(tmpdir) / "description.json"
            args = parser.parse_args(["run", "--description", str(description_path)])
            description_path.write_text(
                """
                {
                  "baseline": "base",
                  "task": "repaint",
                  "checkpoint_dir": "/tmp/checkpoints",
                  "save_dir": "./outputs/example",
                  "caption": "replace with piano bridge",
                  "src_audio": "/tmp/input.wav"
                }
                """,
                encoding="utf-8",
            )
            merged = _apply_description(args)

        self.assertEqual(merged.baseline, "base")
        self.assertEqual(merged.task, "repaint")
        self.assertEqual(merged.checkpoint_dir, "/tmp/checkpoints")
        self.assertEqual(merged.caption, "replace with piano bridge")
        self.assertEqual(merged.src_audio, "/tmp/input.wav")

    def test_cli_value_overrides_description(self):
        parser = _parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            description_path = Path(tmpdir) / "description.json"
            description_path.write_text(
                """
                {
                  "task": "repaint",
                  "caption": "from description"
                }
                """,
                encoding="utf-8",
            )
            args = parser.parse_args(
                [
                    "run",
                    "--description",
                    str(description_path),
                    "--caption",
                    "from cli",
                ]
            )
            merged = _apply_description(args)

        self.assertEqual(merged.caption, "from cli")

    def test_legacy_example_flag_maps_to_description(self):
        parser = _parser()
        args = parser.parse_args(["run", "--example", "sample.json"])
        self.assertEqual("sample.json", args.description)

    def test_checkpoint_dir_is_required(self):
        parser = _parser()
        args = parser.parse_args(
            [
                "run",
                "--baseline",
                "base",
                "--task",
                "text2music",
                "--save-dir",
                "./outputs/test",
                "--caption",
                "hello",
            ]
        )
        with self.assertRaises(ValueError):
            _build_baseline(args)


class ArtifactSavingTests(unittest.TestCase):
    """Run artifact persistence tests."""

    def test_save_run_artifacts_writes_clear_example_files(self):
        baseline = BaselineConfig(
            kind="base",
            project_root=Path("."),
            checkpoint_dir=Path("/tmp/checkpoints"),
            config_path="acestep-v15-turbo",
            save_dir=Path("/tmp/unused"),
        )
        run = RunConfig(
            task="text2music",
            caption="warm piano",
            lyrics="[Instrumental]",
            duration=20,
        )
        config = SimpleNamespace(to_dict=lambda: {"audio_format": "flac", "batch_size": 1})
        result = SimpleNamespace(
            success=True,
            error=None,
            status_message="ok",
            audios=[
                {
                    "path": "/tmp/out.flac",
                    "key": "abc",
                    "sample_rate": 44100,
                    "tensor": "skip me",
                    "params": {"seed": 42},
                }
            ],
            extra_outputs={
                "time_costs": {"pipeline_total_time": 5.0},
                "lm_metadata": {"caption": "warm piano"},
                "latents": "not persisted in full",
            },
        )
        resolved_example = _build_resolved_example_payload(baseline, run, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)
            source_dir = save_dir / "source"
            source_dir.mkdir()
            example_path = source_dir / "description.json"
            example_path.write_text('{"caption": "warm piano"}', encoding="utf-8")
            metadata_path = save_run_artifacts(
                save_dir,
                baseline,
                run,
                config,
                result,
                6.0,
                example_path=str(example_path),
                resolved_example=resolved_example,
                load_summary={
                    "lora": {
                        "path": "/tmp/adapter",
                        "scale": 0.8,
                        "load_message": "✅ LoRA loaded",
                    }
                },
            )

            self.assertEqual(save_dir / "metadata.json", metadata_path)
            self.assertTrue((save_dir / ORIGINAL_EXAMPLE_FILENAME).is_file())
            self.assertTrue((save_dir / RESOLVED_EXAMPLE_FILENAME).is_file())
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            resolved_copy = json.loads(
                (save_dir / RESOLVED_EXAMPLE_FILENAME).read_text(encoding="utf-8")
            )

        self.assertEqual(ORIGINAL_EXAMPLE_FILENAME, payload["example"]["original_copy"])
        self.assertEqual(RESOLVED_EXAMPLE_FILENAME, payload["example"]["resolved_copy"])
        self.assertEqual("base", resolved_copy["baseline"])
        self.assertEqual("text2music", resolved_copy["task"])
        self.assertEqual("/tmp/adapter", payload["load_summary"]["lora"]["path"])
        self.assertEqual(0.8, payload["load_summary"]["lora"]["scale"])



    def test_load_full_sft_weights_returns_summary(self):
        class FakeDecoder:
            def __init__(self):
                self.loaded = None

            def load_state_dict(self, state_dict, strict=False):
                self.loaded = (state_dict, strict)
                return SimpleNamespace(
                    missing_keys=["decoder.extra"],
                    unexpected_keys=["decoder.odd"],
                )

            def eval(self):
                return None

        handler = SimpleNamespace(model=SimpleNamespace(decoder=FakeDecoder()))

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "decoder_state_dict.pt"
            import torch

            torch.save(OrderedDict({"module.decoder.weight": 1}), weights_path)
            summary = _load_full_sft_weights(handler, str(weights_path))

        self.assertEqual(str(weights_path), summary["weights_path"])
        self.assertEqual(1, summary["loaded_key_count"])
        self.assertEqual(1, summary["missing_keys_count"])
        self.assertEqual(1, summary["unexpected_keys_count"])
        self.assertIn("Loaded full-SFT decoder weights", summary["message"])
        self.assertEqual(({"decoder.weight": 1}, False), handler.model.decoder.loaded)

class OutputNamingTests(unittest.TestCase):
    """Output filename normalization tests."""

    def test_single_audio_is_renamed_to_directory_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "turbo_example_04"
            save_dir.mkdir()
            source = save_dir / "random-id.flac"
            source.write_text("audio", encoding="utf-8")
            result = SimpleNamespace(audios=[{"path": str(source)}])

            renamed = rename_output_audios(save_dir, result)

            target = save_dir / "turbo_example_04.flac"
            self.assertEqual([str(target)], renamed)
            self.assertTrue(target.is_file())
            self.assertEqual(str(target), result.audios[0]["path"])

    def test_multiple_audios_get_index_suffixes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "batch_case"
            save_dir.mkdir()
            first = save_dir / "a.flac"
            second = save_dir / "b.flac"
            first.write_text("a", encoding="utf-8")
            second.write_text("b", encoding="utf-8")
            result = SimpleNamespace(audios=[{"path": str(first)}, {"path": str(second)}])

            renamed = rename_output_audios(save_dir, result)

            expected = [
                str(save_dir / "batch_case_01.flac"),
                str(save_dir / "batch_case_02.flac"),
            ]
            self.assertEqual(expected, renamed)
            self.assertEqual(expected[0], result.audios[0]["path"])
            self.assertEqual(expected[1], result.audios[1]["path"])


class FullSftPayloadTests(unittest.TestCase):
    """Full-SFT payload compatibility tests."""

    def test_normalize_full_sft_state_dict_strips_module_prefix(self):
        payload = OrderedDict(
            {
                "module.layers.0.weight": "w0",
                "module.layers.0.bias": "b0",
            }
        )

        result = _normalize_full_sft_state_dict(payload)

        self.assertEqual(
            {
                "layers.0.weight": "w0",
                "layers.0.bias": "b0",
            },
            result,
        )

    def test_normalize_full_sft_state_dict_supports_nested_state_dict(self):
        payload = {"state_dict": {"module.decoder.weight": "w"}}

        result = _normalize_full_sft_state_dict(payload)

        self.assertEqual({"decoder.weight": "w"}, result)


if __name__ == "__main__":
    unittest.main()
