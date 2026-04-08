"""Task parsing and GenerationParams builders for developer inference."""

from __future__ import annotations

from typing import Callable

from acestep.constants import TASK_INSTRUCTIONS

from .schemas import RunConfig

TaskBuilder = Callable[[RunConfig], object]


def _common_params(run: RunConfig, instruction: str) -> dict:
    """Build task-agnostic GenerationParams fields."""

    return {
        "task_type": run.task,
        "instruction": instruction,
        "reference_audio": run.reference_audio,
        "src_audio": run.src_audio,
        "audio_codes": run.audio_codes,
        "caption": run.caption,
        "lyrics": run.lyrics,
        "instrumental": run.instrumental,
        "vocal_language": run.vocal_language,
        "bpm": run.bpm,
        "keyscale": run.keyscale,
        "timesignature": run.timesignature,
        "duration": run.duration,
        "inference_steps": run.inference_steps,
        "seed": run.seed,
        "guidance_scale": run.guidance_scale,
        "use_adg": run.use_adg,
        "shift": run.shift,
        "infer_method": run.infer_method,
        "repainting_start": run.repainting_start,
        "repainting_end": run.repainting_end,
        "audio_cover_strength": run.audio_cover_strength,
        "thinking": run.thinking,
    }


def build_text2music(run: RunConfig):
    """Build params for ``text2music``."""

    from acestep.inference import GenerationParams

    return GenerationParams(**_common_params(run, TASK_INSTRUCTIONS["text2music"]))


def build_cover(run: RunConfig):
    """Build params for ``cover``."""

    from acestep.inference import GenerationParams

    return GenerationParams(**_common_params(run, TASK_INSTRUCTIONS["cover"]))


def build_repaint(run: RunConfig):
    """Build params for ``repaint``."""

    from acestep.inference import GenerationParams

    return GenerationParams(**_common_params(run, TASK_INSTRUCTIONS["repaint"]))


TASK_BUILDERS: dict[str, TaskBuilder] = {
    "text2music": build_text2music,
    "cover": build_cover,
    "repaint": build_repaint,
}


def validate_run(run: RunConfig) -> None:
    """Validate developer CLI inputs before model loading."""

    if run.task not in TASK_BUILDERS:
        raise ValueError(f"Unsupported task: {run.task}")

    if run.task == "text2music":
        if not run.caption and not run.lyrics:
            raise ValueError("text2music requires --caption or --lyrics")
        return

    if not run.src_audio:
        raise ValueError(f"{run.task} requires --src-audio")

    if not run.caption:
        raise ValueError(f"{run.task} requires --caption")

    if run.task == "repaint":
        if run.repainting_end != -1 and run.repainting_end <= run.repainting_start:
            raise ValueError("--repainting-end must be greater than --repainting-start")


def build_generation_params(run: RunConfig):
    """Convert RunConfig into GenerationParams."""

    validate_run(run)
    return TASK_BUILDERS[run.task](run)


def build_generation_config(run: RunConfig):
    """Convert RunConfig into GenerationConfig."""

    from acestep.inference import GenerationConfig

    return GenerationConfig(
        batch_size=run.batch_size,
        use_random_seed=run.use_random_seed,
        seeds=run.seeds,
        audio_format=run.audio_format,
    )
