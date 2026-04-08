"""Runtime helpers for the developer inference CLI."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from acestep.constants import TASK_TYPES_BASE, TASK_TYPES_TURBO

from .schemas import BaselineConfig, RunConfig

ORIGINAL_EXAMPLE_FILENAME = "example.original.json"
RESOLVED_EXAMPLE_FILENAME = "example.resolved.json"


def supported_tasks_for_model(config_path: str) -> set[str]:
    """Return supported tasks for a given DiT model name."""

    normalized = (config_path or "").strip().lower()
    if normalized.startswith("acestep-v15-base") and "sft" not in normalized:
        return set(TASK_TYPES_BASE)
    return set(TASK_TYPES_TURBO)


def validate_baseline_task(baseline: BaselineConfig, task: str) -> None:
    """Raise when the selected model should not run the task."""

    supported = supported_tasks_for_model(baseline.config_path)
    if task not in supported:
        raise ValueError(
            f"Task '{task}' is not supported by model '{baseline.config_path}'. "
            f"Supported tasks: {sorted(supported)}"
        )


def _resolve_full_sft_weights_path(full_sft_path: str) -> Path:
    """Resolve a full-SFT weights file path."""

    path = Path(full_sft_path).expanduser()
    if path.is_file():
        return path
    candidate = path / "decoder_state_dict.pt"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "full-sft baseline requires a file path or a directory containing "
        "'decoder_state_dict.pt'"
    )


def _normalize_full_sft_state_dict(payload) -> dict:
    """Normalize saved full-SFT payloads into decoder state dicts."""

    if not isinstance(payload, dict):
        raise ValueError("Full-SFT payload must be a mapping of tensor weights")

    candidate = payload.get("state_dict") if "state_dict" in payload else payload
    if not isinstance(candidate, dict):
        raise ValueError("Unsupported nested full-SFT payload structure")

    normalized = {}
    for key, value in candidate.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        normalized[normalized_key] = value
    return normalized


def _load_full_sft_weights(dit_handler, full_sft_path: str) -> dict:
    """Load decoder weights for a full-SFT baseline."""

    import torch

    weights_path = _resolve_full_sft_weights_path(full_sft_path)
    payload = torch.load(weights_path, map_location="cpu")
    try:
        state_dict = _normalize_full_sft_state_dict(payload)
    except ValueError as exc:
        raise ValueError(f"Unsupported full-SFT payload in {weights_path}") from exc

    result = dit_handler.model.decoder.load_state_dict(state_dict, strict=False)
    dit_handler.model.decoder.eval()
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)
    return {
        "message": (
            f"Loaded full-SFT decoder weights from {weights_path} "
            f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
        ),
        "weights_path": str(weights_path),
        "loaded_key_count": len(state_dict),
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def initialize_handlers(baseline: BaselineConfig):
    """Initialize DiT and LM handlers for one run."""

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()
    load_summary = {"baseline_kind": baseline.kind}

    dit_handler.initialize_service(
        project_root=str(baseline.project_root),
        config_path=baseline.config_path,
        device=baseline.device,
    )

    if baseline.kind == "lora":
        if not baseline.lora_path:
            raise ValueError("lora baseline requires --lora-path")
        message = dit_handler.load_lora(baseline.lora_path)
        if not message.startswith("✅"):
            raise RuntimeError(message)
        dit_handler.set_lora_scale(baseline.lora_scale)
        load_summary["lora"] = {
            "path": baseline.lora_path,
            "scale": baseline.lora_scale,
            "load_message": message,
        }
    elif baseline.kind == "full-sft":
        if not baseline.full_sft_path:
            raise ValueError("full-sft baseline requires --full-sft-path")
        full_sft_summary = _load_full_sft_weights(dit_handler, baseline.full_sft_path)
        load_summary["full_sft"] = full_sft_summary
        print(full_sft_summary["message"])

    if baseline.init_llm:
        llm_handler.initialize(
            checkpoint_dir=str(baseline.checkpoint_dir),
            lm_model_path=baseline.lm_model_path,
            backend=baseline.backend,
            device=baseline.device,
        )

    return dit_handler, llm_handler, load_summary


def _json_safe(value):
    """Convert nested values into JSON-safe metadata."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return str(value)


def _serialize_result(result) -> dict:
    """Build a compact JSON-safe result payload."""

    audios = []
    for audio in getattr(result, "audios", []):
        item = {key: _json_safe(value) for key, value in audio.items() if key != "tensor"}
        audios.append(item)

    extra_outputs = getattr(result, "extra_outputs", {}) or {}
    return {
        "success": getattr(result, "success", False),
        "error": getattr(result, "error", None),
        "status_message": getattr(result, "status_message", ""),
        "audios": audios,
        "extra_outputs": {
            "keys": sorted(extra_outputs.keys()),
            "lm_metadata": _json_safe(extra_outputs.get("lm_metadata")),
            "time_costs": _json_safe(extra_outputs.get("time_costs", {})),
        },
    }


def _build_timing_payload(run: RunConfig, result, elapsed_seconds: float) -> dict:
    """Build timing and RTF metadata for one run."""

    extra_outputs = getattr(result, "extra_outputs", {}) or {}
    time_costs = extra_outputs.get("time_costs", {}) or {}
    pipeline_seconds = time_costs.get("pipeline_total_time")
    target_duration = run.duration if run.duration and run.duration > 0 else None
    rtf_source_seconds = pipeline_seconds if pipeline_seconds is not None else elapsed_seconds
    rtf = None
    if target_duration:
        rtf = rtf_source_seconds / target_duration

    return {
        "wall_clock_seconds": elapsed_seconds,
        "pipeline_total_time_seconds": pipeline_seconds,
        "requested_duration_seconds": target_duration,
        "rtf": rtf,
        "rtf_source": (
            "pipeline_total_time_seconds"
            if pipeline_seconds is not None
            else "wall_clock_seconds"
        ),
    }




def rename_output_audios(save_dir: Path, result) -> list[str]:
    """Rename generated audio files to match the output directory name."""

    renamed_paths = []
    audios = getattr(result, "audios", []) or []
    total = len(audios)
    for index, audio in enumerate(audios, start=1):
        raw_path = audio.get("path")
        if not raw_path:
            continue
        source_path = Path(raw_path)
        if not source_path.is_file():
            continue
        suffix = source_path.suffix
        target_name = save_dir.name if total == 1 else f"{save_dir.name}_{index:02d}"
        target_path = save_dir / f"{target_name}{suffix}"
        if source_path != target_path:
            if target_path.exists():
                target_path.unlink()
            source_path.replace(target_path)
            audio["path"] = str(target_path)
        renamed_paths.append(str(target_path))
    return renamed_paths

def save_run_artifacts(
    save_dir: Path,
    baseline: BaselineConfig,
    run: RunConfig,
    config,
    result,
    elapsed_seconds: float,
    example_path: str | None = None,
    resolved_example: dict | None = None,
    load_summary: dict | None = None,
) -> Path:
    """Write metadata and prompt files next to generated audio."""

    save_dir.mkdir(parents=True, exist_ok=True)
    resolved_example_path = None
    original_example_path = None
    if example_path:
        original_example_path = Path(example_path).expanduser().resolve()
        shutil.copy2(original_example_path, save_dir / ORIGINAL_EXAMPLE_FILENAME)
    if resolved_example is not None:
        resolved_example_path = save_dir / RESOLVED_EXAMPLE_FILENAME
        resolved_example_path.write_text(
            json.dumps(_json_safe(resolved_example), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    metadata = {
        "baseline": _json_safe(asdict(baseline)),
        "run": _json_safe(asdict(run)),
        "example": {
            "source_path": str(original_example_path) if original_example_path else None,
            "original_copy": ORIGINAL_EXAMPLE_FILENAME if original_example_path else None,
            "resolved_copy": RESOLVED_EXAMPLE_FILENAME if resolved_example_path else None,
        },
        "prompt": run.caption,
        "lyrics": run.lyrics,
        "generation_config": _json_safe(config.to_dict()),
        "load_summary": _json_safe(load_summary or {}),
        "timing": _build_timing_payload(run, result, elapsed_seconds),
        "result": _serialize_result(result),
    }
    metadata_path = save_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if run.caption:
        (save_dir / "prompt.txt").write_text(run.caption, encoding="utf-8")
    if run.lyrics:
        (save_dir / "lyrics.txt").write_text(run.lyrics, encoding="utf-8")
    status_message = getattr(result, "status_message", "")
    if status_message:
        (save_dir / "status.txt").write_text(status_message, encoding="utf-8")
    return metadata_path


def run_generation(dit_handler, llm_handler, params, config, save_dir: Path):
    """Execute one generation call."""

    from acestep.inference import generate_music

    save_dir.mkdir(parents=True, exist_ok=True)
    return generate_music(
        dit_handler,
        llm_handler,
        params,
        config,
        save_dir=str(save_dir),
    )
