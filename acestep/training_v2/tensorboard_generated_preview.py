"""Training-time generated-audio preview helpers for TensorBoard."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from acestep.training_v2.tensorboard_preview import normalize_waveform_for_tensorboard


def _coerce_int(value: Any) -> Optional[int]:
    """Convert supported scalar values to ``int`` when possible."""
    if value in (None, "", "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_duration(value: Any, *, default: float = 30.0, maximum: float = 30.0) -> float:
    """Resolve preview duration with a safety cap for training-time generation."""
    try:
        duration = float(value)
    except (TypeError, ValueError):
        duration = default
    if duration <= 0:
        duration = default
    return max(5.0, min(duration, maximum))


def build_generation_preview_request(metadata: Mapping[str, Any], training_config: Any) -> dict[str, Any]:
    """Build a same-condition preview-generation request from sample metadata."""
    caption = str(metadata.get("caption") or metadata.get("filename") or "training preview").strip()
    lyrics = str(metadata.get("lyrics") or "[Instrumental]").strip()
    return {
        "caption": caption,
        "lyrics": lyrics,
        "bpm": _coerce_int(metadata.get("bpm")),
        "key_scale": str(metadata.get("keyscale") or "").strip(),
        "time_signature": str(metadata.get("timesignature") or "").strip(),
        "vocal_language": str(metadata.get("language") or "en").strip() or "en",
        "audio_duration": _coerce_duration(metadata.get("duration")),
        "inference_steps": int(getattr(training_config, "num_inference_steps", 8)),
        "shift": float(getattr(training_config, "shift", 1.0)),
        "seed": int(getattr(training_config, "seed", 42)),
    }


def build_generated_preview_text(request: Mapping[str, Any]) -> str:
    """Build a readable Markdown TensorBoard summary for generated previews."""
    lyrics = str(request["lyrics"]).strip().replace("\n", " ")
    if len(lyrics) > 180:
        lyrics = f"{lyrics[:177]}..."
    return "\n".join([
        "## Generated Preview",
        "",
        "- generated_from_current_model: `true`",
        f"- caption: `{request['caption']}`",
        f"- lyrics: {lyrics}",
        f"- duration: `{request['audio_duration']}`",
        f"- inference_steps: `{request['inference_steps']}`",
        f"- shift: `{request['shift']}`",
        f"- seed: `{request['seed']}`",
        f"- bpm: `{request['bpm'] if request['bpm'] is not None else 'N/A'}`",
        f"- key_scale: `{request['key_scale'] or 'N/A'}`",
        f"- time_signature: `{request['time_signature'] or 'N/A'}`",
        f"- vocal_language: `{request['vocal_language']}`",
    ])


def generate_preview_audio(
    *,
    model: Any,
    training_config: Any,
    metadata: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int, str]:
    """Generate one same-condition audio preview from the current training model."""
    from acestep.handler import AceStepHandler
    from acestep.training_v2.model_loader import (
        load_silence_latent,
        load_text_encoder,
        load_vae,
        unload_models,
    )

    request = build_generation_preview_request(metadata, training_config)
    handler = AceStepHandler()
    handler.model = model
    handler.config = getattr(model, "config", None)
    handler.config_path = f"acestep-v15-{training_config.model_variant}"
    handler.device = str(device)
    handler.dtype = dtype
    handler.batch_size = 1
    handler.offload_to_cpu = True
    handler.offload_dit_to_cpu = False

    vae = None
    text_encoder = None
    tokenizer = None
    silence_latent = None

    was_training = bool(getattr(model, 'training', False))
    decoder = getattr(model, 'decoder', None)
    decoder_was_training = bool(getattr(decoder, 'training', False)) if decoder is not None else False

    try:
        model.eval()
        if decoder is not None:
            decoder.eval()

        vae = load_vae(training_config.checkpoint_dir, device='cpu', precision='fp32')
        tokenizer, text_encoder = load_text_encoder(
            training_config.checkpoint_dir,
            device='cpu',
            precision='fp32',
        )
        silence_latent = load_silence_latent(
            training_config.checkpoint_dir,
            device='cpu',
            precision='fp32',
            variant=training_config.model_variant,
        )

        handler.vae = vae
        handler.text_tokenizer = tokenizer
        handler.text_encoder = text_encoder
        handler.silence_latent = silence_latent

        result = handler.generate_music(
            captions=request['caption'],
            lyrics=request['lyrics'],
            bpm=request['bpm'],
            key_scale=request['key_scale'],
            time_signature=request['time_signature'],
            vocal_language=request['vocal_language'],
            inference_steps=request['inference_steps'],
            guidance_scale=7.0,
            use_random_seed=False,
            seed=request['seed'],
            audio_duration=request['audio_duration'],
            batch_size=1,
            task_type='text2music',
            shift=request['shift'],
            infer_method='ode',
            use_tiled_decode=True,
        )
        if not result.get('success', False):
            raise RuntimeError(result.get('error') or result.get('status_message') or 'preview generation failed')

        audios = result.get('audios') or []
        if not audios:
            raise RuntimeError('preview generation returned no audio')

        first = audios[0]
        waveform = normalize_waveform_for_tensorboard(first['tensor'])
        sample_rate = int(first.get('sample_rate', 48000))
        return waveform, sample_rate, build_generated_preview_text(request)
    finally:
        if decoder is not None and decoder_was_training:
            decoder.train()
        if was_training:
            model.train()
        unload_models(vae, text_encoder, tokenizer, silence_latent)
