"""Tests for TensorBoard sample-preview helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
import torchaudio
from torch.utils.data import Subset

from acestep.training_v2.tensorboard_preview import (
    build_preview_batch_from_sample,
    build_sample_preview,
    build_spectrogram_image,
    collect_preview_samples,
    extract_first_audio_path,
    load_audio_preview,
    normalize_waveform_for_tensorboard,
    select_preview_dataset,
    should_log_sample_preview,
)


class TestShouldLogSamplePreview(unittest.TestCase):
    """Validate preview cadence decisions."""

    def test_returns_false_when_disabled(self) -> None:
        self.assertFalse(should_log_sample_preview(0, 0))

    def test_returns_true_on_matching_epoch(self) -> None:
        self.assertTrue(should_log_sample_preview(2, 1))

    def test_returns_false_on_non_matching_epoch(self) -> None:
        self.assertFalse(should_log_sample_preview(3, 1))


class TestBuildSamplePreview(unittest.TestCase):
    """Validate TensorBoard preview formatting."""

    def test_formats_metadata_and_shapes(self) -> None:
        batch = {
            "target_latents": torch.zeros(2, 8, 64),
            "encoder_hidden_states": torch.zeros(2, 12, 1024),
            "metadata": [
                {
                    "filename": "demo.wav",
                    "caption": "lofi piano",
                    "genre": "jazz",
                    "lyrics": "line one\nline two",
                    "audio_path": "/tmp/demo.wav",
                },
            ],
        }

        preview = build_sample_preview(batch, epoch=0)

        self.assertIn("## Ground Truth Preview", preview)
        self.assertIn("- epoch: `1`", preview)
        self.assertIn("- target_latents_shape: `(2, 8, 64)`", preview)
        self.assertIn("- encoder_hidden_states_shape: `(2, 12, 1024)`", preview)
        self.assertIn("- filename: `demo.wav`", preview)
        self.assertIn("- caption: `lofi piano`", preview)
        self.assertIn("- lyrics: line one line two", preview)

    def test_handles_missing_metadata(self) -> None:
        preview = build_sample_preview({"metadata": None}, epoch=1)

        self.assertIn("- epoch: `2`", preview)
        self.assertIn("_metadata unavailable_", preview)


class TestPreviewSamplePool(unittest.TestCase):
    """Validate deterministic preview-pool selection helpers."""

    def test_collect_preview_samples_reads_first_items_from_subset(self) -> None:
        base_dataset = [
            {"metadata": {"filename": "sample_0.wav"}},
            {"metadata": {"filename": "sample_1.wav"}},
            {"metadata": {"filename": "sample_2.wav"}},
        ]
        subset = Subset(base_dataset, [2, 0])

        preview_samples = collect_preview_samples(subset, max_items=2)

        self.assertEqual(2, len(preview_samples))
        self.assertEqual("sample_2.wav", preview_samples[0]["metadata"]["filename"])
        self.assertEqual("sample_0.wav", preview_samples[1]["metadata"]["filename"])

    def test_build_preview_batch_from_sample_wraps_tensors_as_single_item_batch(self) -> None:
        sample = {
            "target_latents": torch.zeros(8, 64),
            "encoder_hidden_states": torch.zeros(12, 1024),
            "metadata": {"filename": "demo.wav"},
        }

        batch = build_preview_batch_from_sample(sample)

        self.assertEqual((1, 8, 64), tuple(batch["target_latents"].shape))
        self.assertEqual((1, 12, 1024), tuple(batch["encoder_hidden_states"].shape))
        self.assertEqual("demo.wav", batch["metadata"][0]["filename"])

    def test_select_preview_dataset_prefers_validation_split(self) -> None:
        data_module = type("DM", (), {
            "train_dataset": [{"metadata": {"filename": "train.wav"}}],
            "val_dataset": [{"metadata": {"filename": "val.wav"}}],
        })()

        dataset, source = select_preview_dataset(data_module)

        self.assertEqual("validation", source)
        self.assertEqual("val.wav", dataset[0]["metadata"]["filename"])


class TestAudioPreviewHelpers(unittest.TestCase):
    """Validate audio preview extraction and spectrogram generation."""

    def test_extract_first_audio_path(self) -> None:
        batch = {"metadata": [{"audio_path": "/tmp/demo.wav"}]}
        self.assertEqual("/tmp/demo.wav", extract_first_audio_path(batch))

    def test_load_audio_preview_resamples_and_mixes_down(self) -> None:
        with TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "demo.wav"
            waveform = torch.stack([
                torch.linspace(-0.5, 0.5, 8000),
                torch.linspace(0.5, -0.5, 8000),
            ])
            torchaudio.save(str(audio_path), waveform, 8000)

            preview, sample_rate = load_audio_preview(str(audio_path), target_sample_rate=16000)

            self.assertEqual(16000, sample_rate)
            self.assertEqual(1, preview.shape[0])
            self.assertGreater(preview.shape[1], 8000)

    def test_build_spectrogram_image_returns_single_channel_image(self) -> None:
        waveform = torch.sin(torch.linspace(0, 100, 16000)).unsqueeze(0)

        image = build_spectrogram_image(waveform)

        self.assertEqual(3, image.ndim)
        self.assertEqual(1, image.shape[0])
        self.assertGreater(image.shape[1], 0)
        self.assertGreater(image.shape[2], 0)

    def test_normalize_waveform_for_tensorboard_mixes_multichannel_audio(self) -> None:
        waveform = torch.stack([
            torch.linspace(-1.0, 1.0, 8),
            torch.linspace(1.0, -1.0, 8),
        ])

        normalized = normalize_waveform_for_tensorboard(waveform)

        self.assertEqual((1, 8), tuple(normalized.shape))
        self.assertTrue(torch.all(normalized <= 1.0))
        self.assertTrue(torch.all(normalized >= -1.0))
