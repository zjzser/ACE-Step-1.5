"""Shared dataclasses for the developer inference CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaselineConfig:
    """Model-loading settings for one inference run."""

    kind: str
    project_root: Path
    checkpoint_dir: Path
    config_path: str
    save_dir: Path
    device: str = "cuda"
    lm_model_path: str = "acestep-5Hz-lm-1.7B"
    backend: str = "vllm"
    init_llm: bool = True
    lora_path: str | None = None
    lora_scale: float = 1.0
    full_sft_path: str | None = None


@dataclass
class RunConfig:
    """User-facing run settings for one task execution."""

    task: str
    caption: str = ""
    lyrics: str = ""
    instrumental: bool = False
    reference_audio: str | None = None
    src_audio: str | None = None
    audio_codes: str = ""
    vocal_language: str = "unknown"
    bpm: int | None = None
    keyscale: str = ""
    timesignature: str = ""
    duration: float = -1.0
    inference_steps: int = 8
    seed: int = -1
    guidance_scale: float = 7.0
    use_adg: bool = False
    shift: float = 1.0
    infer_method: str = "ode"
    repainting_start: float = 0.0
    repainting_end: float = -1.0
    audio_cover_strength: float = 1.0
    thinking: bool = True
    batch_size: int = 1
    use_random_seed: bool = True
    seeds: list[int] | None = None
    audio_format: str = "flac"
