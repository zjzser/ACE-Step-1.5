"""TensorBoard preview helpers for readable training-time case inspection."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from torch.utils.data import Subset

import torch
import torchaudio


_DISPLAY_FIELDS = (
    "filename",
    "caption",
    "genre",
    "duration",
    "bpm",
    "keyscale",
    "timesignature",
    "custom_tag",
    "prompt_override",
    "audio_path",
)


def should_log_sample_preview(sample_every_n_epochs: int, epoch: int) -> bool:
    """Return whether the current epoch should emit a recurring sample preview."""
    if sample_every_n_epochs <= 0:
        return False
    return (epoch + 1) % sample_every_n_epochs == 0


def _format_value(value: Any) -> str:
    """Format a metadata value into a compact single-line string."""
    return str(value).strip().replace("\n", " ")


def _clip_text(text: Any, *, limit: int = 180) -> str:
    """Clip free-form text for readable TensorBoard summaries."""
    clipped = _format_value(text)
    if len(clipped) > limit:
        return f"{clipped[:limit - 3]}..."
    return clipped


def build_sample_preview(
    batch: Mapping[str, Any],
    *,
    epoch: int,
    max_items: int = 2,
) -> str:
    """Build a readable Markdown preview for TensorBoard from batch metadata."""
    lines = [f"## Ground Truth Preview", "", f"- epoch: `{epoch + 1}`"]

    target_latents = batch.get("target_latents")
    if hasattr(target_latents, "shape"):
        lines.append(
            f"- target_latents_shape: `{tuple(int(dim) for dim in target_latents.shape)}`"
        )

    encoder_hidden_states = batch.get("encoder_hidden_states")
    if hasattr(encoder_hidden_states, "shape"):
        lines.append(
            "- encoder_hidden_states_shape: "
            f"`{tuple(int(dim) for dim in encoder_hidden_states.shape)}`"
        )

    metadata_items = batch.get("metadata")
    if not isinstance(metadata_items, Sequence) or isinstance(metadata_items, (str, bytes)):
        lines.extend(["", "_metadata unavailable_"])
        return "\n".join(lines)

    for idx, metadata in enumerate(metadata_items[:max_items]):
        lines.extend(["", f"### Sample {idx}"])
        if not isinstance(metadata, Mapping):
            lines.append("_metadata unavailable_")
            continue

        for field in _DISPLAY_FIELDS:
            value = metadata.get(field)
            if value in (None, "", []):
                continue
            lines.append(f"- {field}: `{_format_value(value)}`")

        lyrics = metadata.get("lyrics")
        if lyrics not in (None, "", []):
            lines.append(f"- lyrics: {_clip_text(lyrics)}")

    return "\n".join(lines)


def extract_first_audio_path(batch: Mapping[str, Any]) -> Optional[str]:
    """Return the first sample's source audio path when available."""
    metadata_items = batch.get("metadata")
    if not isinstance(metadata_items, Sequence) or isinstance(metadata_items, (str, bytes)):
        return None
    if not metadata_items:
        return None
    first = metadata_items[0]
    if not isinstance(first, Mapping):
        return None
    audio_path = first.get("audio_path")
    return str(audio_path) if audio_path else None


def load_audio_preview(
    audio_path: str,
    *,
    target_sample_rate: int = 24000,
    max_duration_seconds: float = 30.0,
) -> tuple[torch.Tensor, int]:
    """Load a short mono waveform preview from an on-disk training sample."""
    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    max_samples = int(sample_rate * max_duration_seconds)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[:, :max_samples]

    waveform = waveform.detach().cpu().float().clamp(-1.0, 1.0)
    return waveform, sample_rate


def normalize_waveform_for_tensorboard(waveform: torch.Tensor) -> torch.Tensor:
    """Normalize arbitrary waveform shapes into mono ``[1, num_samples]``."""
    if waveform.ndim == 1:
        normalized = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        normalized = waveform
    elif waveform.ndim == 3 and waveform.shape[0] == 1:
        normalized = waveform[0]
    else:
        raise ValueError(
            f"Unsupported waveform shape for TensorBoard audio: {tuple(waveform.shape)}"
        )

    if normalized.ndim != 2:
        raise ValueError(f"Waveform normalization failed for shape: {tuple(waveform.shape)}")

    if normalized.shape[0] > 1:
        normalized = normalized.mean(dim=0, keepdim=True)

    return normalized.detach().cpu().float().clamp(-1.0, 1.0)


def collect_preview_samples(dataset: Any, *, max_items: int = 2) -> list[Mapping[str, Any]]:
    """Collect a small deterministic preview pool from the selected dataset."""
    if dataset is None or max_items <= 0:
        return []

    if isinstance(dataset, Subset):
        indices = list(dataset.indices)[:max_items]
        return [dataset.dataset[int(index)] for index in indices]

    dataset_len = len(dataset)
    return [dataset[index] for index in range(min(max_items, dataset_len))]


def select_preview_dataset(data_module: Any) -> tuple[Any, str]:
    """Prefer validation data for previews and fall back to training data."""
    val_dataset = getattr(data_module, "val_dataset", None)
    if val_dataset is not None:
        return val_dataset, "validation"
    return getattr(data_module, "train_dataset", None), "training"


def build_preview_batch_from_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a dataset sample into a single-item batch for preview rendering."""
    batch: dict[str, Any] = {"metadata": [sample.get("metadata", {})]}

    target_latents = sample.get("target_latents")
    if isinstance(target_latents, torch.Tensor):
        batch["target_latents"] = target_latents.unsqueeze(0)

    encoder_hidden_states = sample.get("encoder_hidden_states")
    if isinstance(encoder_hidden_states, torch.Tensor):
        batch["encoder_hidden_states"] = encoder_hidden_states.unsqueeze(0)

    return batch


def build_spectrogram_image(
    waveform: torch.Tensor,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> torch.Tensor:
    """Convert a mono waveform into a normalized spectrogram image tensor."""
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError("waveform must have shape [1, num_samples]")

    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(
        waveform[0],
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    magnitude = stft.abs()
    spectrogram = torch.log1p(magnitude)
    spectrogram = spectrogram.flip(0)

    max_val = float(spectrogram.max())
    if max_val > 0:
        spectrogram = spectrogram / max_val

    return spectrogram.unsqueeze(0)
