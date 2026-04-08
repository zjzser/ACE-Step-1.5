"""Simple developer-facing CLI for unified inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .runtime import (
    initialize_handlers,
    rename_output_audios,
    run_generation,
    save_run_artifacts,
    validate_baseline_task,
)
from .schemas import BaselineConfig, RunConfig
from .tasks import build_generation_config, build_generation_params

RUN_FIELDS = {
    "task",
    "caption",
    "lyrics",
    "instrumental",
    "reference_audio",
    "src_audio",
    "audio_codes",
    "vocal_language",
    "bpm",
    "keyscale",
    "timesignature",
    "duration",
    "inference_steps",
    "seed",
    "guidance_scale",
    "use_adg",
    "shift",
    "infer_method",
    "repainting_start",
    "repainting_end",
    "audio_cover_strength",
    "thinking",
    "batch_size",
    "use_random_seed",
    "seeds",
    "audio_format",
}

BASELINE_FIELDS = {
    "baseline",
    "project_root",
    "checkpoint_dir",
    "save_dir",
    "config_path",
    "device",
    "lm_model_path",
    "backend",
    "no_init_llm",
    "lora_path",
    "lora_scale",
    "full_sft_path",
}



def _parser() -> argparse.ArgumentParser:
    """Build the root parser."""

    parser = argparse.ArgumentParser(
        prog="python -m dev_infer",
        description="Developer inference CLI for ACE-Step",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one inference job")
    run.add_argument("--description", default=None)
    run.add_argument("--example", dest="description", default=None, help=argparse.SUPPRESS)
    run.add_argument("--baseline", choices=["base", "lora", "full-sft"], default=None)
    run.add_argument("--task", choices=["text2music", "cover", "repaint"], default=None)
    run.add_argument("--project-root", default=".")
    run.add_argument("--checkpoint-dir", default=None)
    run.add_argument("--save-dir", default=None)
    run.add_argument("--config-path", default="acestep-v15-sft")
    run.add_argument("--device", default="cuda")
    run.add_argument("--lm-model-path", default="acestep-5Hz-lm-1.7B")
    run.add_argument("--backend", default="vllm")
    run.add_argument("--no-init-llm", action="store_true")

    run.add_argument("--lora-path", default=None)
    run.add_argument("--lora-scale", type=float, default=1.0)
    run.add_argument("--full-sft-path", default=None)

    run.add_argument("--caption", default="")
    run.add_argument("--lyrics", default="")
    run.add_argument("--instrumental", action="store_true")
    run.add_argument("--reference-audio", default=None)
    run.add_argument("--src-audio", default=None)
    run.add_argument("--audio-codes", default="")
    run.add_argument("--vocal-language", default="unknown")
    run.add_argument("--bpm", type=int, default=None)
    run.add_argument("--keyscale", default="")
    run.add_argument("--timesignature", default="")
    run.add_argument("--duration", type=float, default=-1.0)
    run.add_argument("--inference-steps", type=int, default=8)
    run.add_argument("--seed", type=int, default=-1)
    run.add_argument("--guidance-scale", type=float, default=7.0)
    run.add_argument("--use-adg", action="store_true")
    run.add_argument("--shift", type=float, default=1.0)
    run.add_argument("--infer-method", default="ode")
    run.add_argument("--repainting-start", type=float, default=0.0)
    run.add_argument("--repainting-end", type=float, default=-1.0)
    run.add_argument("--audio-cover-strength", type=float, default=1.0)
    run.add_argument("--no-thinking", action="store_true")

    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--use-random-seed", action="store_true")
    run.add_argument("--seeds", type=int, nargs="*", default=None)
    run.add_argument("--audio-format", default="flac")
    return parser



def _load_description(path_str: str | None) -> dict:
    """Load one description JSON file."""

    if not path_str:
        return {}
    path = Path(path_str).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Description file must contain a JSON object")
    return data



def _apply_description(args: argparse.Namespace) -> argparse.Namespace:
    """Fill missing CLI fields from a description file."""

    description = _load_description(args.description)
    for key in RUN_FIELDS | BASELINE_FIELDS:
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        if current is not None and current != "":
            continue
        if key not in description:
            continue
        setattr(args, key, description[key])

    if args.task is None and "task" in description:
        args.task = description["task"]
    if args.baseline is None and "baseline" in description:
        args.baseline = description["baseline"]
    if args.save_dir is None and "save_dir" in description:
        args.save_dir = description["save_dir"]
    return args



def _require_arg(value: str | None, name: str) -> str:
    """Require one merged CLI argument."""

    if value:
        return value
    raise ValueError(f"Missing required argument: {name}")



def _build_baseline(args: argparse.Namespace) -> BaselineConfig:
    """Create BaselineConfig from argparse args."""

    project_root = Path(args.project_root).expanduser().resolve()
    checkpoint_dir = Path(
        _require_arg(args.checkpoint_dir, "--checkpoint-dir")
    ).expanduser().resolve()
    return BaselineConfig(
        kind=_require_arg(args.baseline, "--baseline"),
        project_root=project_root,
        checkpoint_dir=checkpoint_dir,
        config_path=args.config_path,
        save_dir=Path(_require_arg(args.save_dir, "--save-dir")).expanduser().resolve(),
        device=args.device,
        lm_model_path=args.lm_model_path,
        backend=args.backend,
        init_llm=not args.no_init_llm,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        full_sft_path=args.full_sft_path,
    )



def _build_run(args: argparse.Namespace) -> RunConfig:
    """Create RunConfig from argparse args."""

    return RunConfig(
        task=_require_arg(args.task, "--task"),
        caption=args.caption,
        lyrics=args.lyrics,
        instrumental=args.instrumental,
        reference_audio=args.reference_audio,
        src_audio=args.src_audio,
        audio_codes=args.audio_codes,
        vocal_language=args.vocal_language,
        bpm=args.bpm,
        keyscale=args.keyscale,
        timesignature=args.timesignature,
        duration=args.duration,
        inference_steps=args.inference_steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        use_adg=args.use_adg,
        shift=args.shift,
        infer_method=args.infer_method,
        repainting_start=args.repainting_start,
        repainting_end=args.repainting_end,
        audio_cover_strength=args.audio_cover_strength,
        thinking=not args.no_thinking,
        batch_size=args.batch_size,
        use_random_seed=args.use_random_seed,
        seeds=args.seeds,
        audio_format=args.audio_format,
    )



def _build_resolved_example_payload(
    baseline: BaselineConfig,
    run: RunConfig,
    config,
) -> dict:
    """Build a readable resolved config payload for output inspection."""

    return {
        "baseline": baseline.kind,
        "task": run.task,
        "baseline_config": {
            "checkpoint_dir": str(baseline.checkpoint_dir),
            "config_path": baseline.config_path,
            "save_dir": str(baseline.save_dir),
            "device": baseline.device,
            "lm_model_path": baseline.lm_model_path,
            "backend": baseline.backend,
            "lora_path": baseline.lora_path,
            "lora_scale": baseline.lora_scale,
            "full_sft_path": baseline.full_sft_path,
        },
        "run": run.__dict__,
        "generation_config": config.to_dict(),
    }



def run_command(args: argparse.Namespace) -> int:
    """Run the developer inference command."""

    args = _apply_description(args)
    baseline = _build_baseline(args)
    run = _build_run(args)
    validate_baseline_task(baseline, run.task)
    params = build_generation_params(run)
    config = build_generation_config(run)
    dit_handler, llm_handler, load_summary = initialize_handlers(baseline)
    started_at = perf_counter()
    result = run_generation(
        dit_handler,
        llm_handler,
        params,
        config,
        baseline.save_dir,
    )
    elapsed_seconds = perf_counter() - started_at
    renamed_audio_paths = rename_output_audios(baseline.save_dir, result)
    metadata_path = save_run_artifacts(
        baseline.save_dir,
        baseline,
        run,
        config,
        result,
        elapsed_seconds,
        example_path=args.description,
        resolved_example=_build_resolved_example_payload(baseline, run, config),
        load_summary=load_summary,
    )
    if not result.success:
        print(f"Generation failed: {result.error or result.status_message}")
        print(f"Saved run metadata to {metadata_path}")
        return 1

    print(f"Saved {len(result.audios)} audio file(s) to {baseline.save_dir}")
    for audio_path in renamed_audio_paths or [audio["path"] for audio in result.audios]:
        print(audio_path)
    print(f"Saved run metadata to {metadata_path}")
    return 0



def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run_command(args)
    except ValueError as exc:
        parser.error(str(exc))
    parser.error(f"Unknown command: {args.command}")
    return 2
