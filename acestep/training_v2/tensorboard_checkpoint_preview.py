"""Background TensorBoard generated-preview launcher for saved checkpoints."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from loguru import logger

from acestep.training.data_module import PreprocessedDataModule
from acestep.training_v2.model_loader import load_decoder_for_training
from acestep.training_v2.tensorboard_generated_preview import generate_preview_audio
from acestep.training_v2.tensorboard_preview import (
    build_spectrogram_image,
    collect_preview_samples,
    normalize_waveform_for_tensorboard,
    select_preview_dataset,
)
from acestep.training_v2.tensorboard_utils import TrainingLogger


def _parse_args() -> argparse.Namespace:
    """Parse CLI args for one-off checkpoint preview generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-path', required=True, help='Checkpoint directory to preview')
    parser.add_argument('--output-dir', required=True, help='Training output dir containing runs/')
    parser.add_argument('--dataset-dir', required=True, help='Directory containing preprocessed .pt tensors')
    parser.add_argument('--checkpoint-dir', required=True, help='Base ACE-Step checkpoints root')
    parser.add_argument('--model-variant', default='turbo', help='Model variant (default: turbo)')
    parser.add_argument('--full-sft', action='store_true', default=False, help='Load full-SFT decoder checkpoints')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Inference device')
    parser.add_argument('--precision', default='bf16', choices=['bf16', 'fp16', 'fp32'], help='Inference precision')
    parser.add_argument('--val-split', type=float, default=0.0, help='Optional validation split; previews prefer validation samples')
    parser.add_argument('--max-samples', type=int, default=2, help='Number of fixed preview samples to use')
    parser.add_argument('--log-dir', default=None, help='TensorBoard log dir (default: {output-dir}/runs)')
    return parser.parse_args()


def _build_training_config(args: argparse.Namespace) -> Any:
    """Create the minimal config object expected by preview helpers."""
    return SimpleNamespace(
        checkpoint_dir=args.checkpoint_dir,
        model_variant=args.model_variant,
        num_inference_steps=8,
        shift=3.0 if args.model_variant == 'turbo' else 1.0,
        seed=42,
    )


def _load_progress(ckpt_dir: Path) -> int:
    """Read global_step from training_state when available."""
    state_path = ckpt_dir / 'training_state.pt'
    if not state_path.is_file():
        return 0
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    return int(state.get('global_step', 0))


def _resolve_lora_adapter_path(ckpt_dir: Path) -> Path:
    """Return the filesystem path that contains LoRA adapter weights."""
    adapter_dir = ckpt_dir / 'adapter'
    if adapter_dir.is_dir():
        return adapter_dir
    return ckpt_dir


def _load_model_for_checkpoint(args: argparse.Namespace, ckpt_dir: Path) -> Any:
    """Load a model and attach checkpoint weights for preview generation."""
    model = load_decoder_for_training(
        checkpoint_dir=args.checkpoint_dir,
        variant=args.model_variant,
        device=args.device,
        precision=args.precision,
    )
    if args.full_sft:
        state_path = ckpt_dir / 'decoder_state_dict.pt'
        state = torch.load(state_path, map_location=args.device, weights_only=False)
        model.decoder.load_state_dict(state, strict=False)
    else:
        from peft import PeftModel
        adapter_path = _resolve_lora_adapter_path(ckpt_dir)
        model.decoder = PeftModel.from_pretrained(model.decoder, str(adapter_path))
    model.eval()
    decoder = getattr(model, 'decoder', None)
    if decoder is not None:
        decoder.eval()
    return model


def _load_preview_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load the fixed preview sample pool from validation or training data."""
    data_module = PreprocessedDataModule(
        tensor_dir=args.dataset_dir,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=False,
        val_split=args.val_split,
    )
    data_module.setup('fit')
    dataset, source = select_preview_dataset(data_module)
    samples = collect_preview_samples(dataset, max_items=args.max_samples)
    logger.info('[INFO] Background generated previews use {} samples', source)
    return samples


def _write_generated_previews(args: argparse.Namespace) -> None:
    """Generate TensorBoard previews for a single checkpoint."""
    ckpt_dir = Path(args.checkpoint_path)
    log_dir = Path(args.log_dir) if args.log_dir else Path(args.output_dir) / 'runs'
    tb = TrainingLogger(log_dir)
    cfg = _build_training_config(args)
    samples = _load_preview_samples(args)
    step = _load_progress(ckpt_dir)
    model = _load_model_for_checkpoint(args, ckpt_dir)
    try:
        first_param = next(model.parameters(), None)
        model_device = first_param.device if first_param is not None else torch.device('cpu')
        dtype = first_param.dtype if first_param is not None else torch.float32
        for sample_idx, preview_sample in enumerate(samples):
            metadata = preview_sample.get('metadata', {})
            if not isinstance(metadata, dict):
                continue
            tag_prefix = f'train_preview/sample_{sample_idx}'
            gen_waveform, gen_sample_rate, gen_text = generate_preview_audio(
                model=model,
                training_config=cfg,
                metadata=metadata,
                device=model_device,
                dtype=dtype,
            )
            gen_waveform = normalize_waveform_for_tensorboard(gen_waveform)
            tb.log_text(f'{tag_prefix}/generated_case', gen_text, step)
            tb.log_audio(f'{tag_prefix}/generated_audio', gen_waveform, step, gen_sample_rate)
            tb.log_image(
                f'{tag_prefix}/generated_spectrogram',
                build_spectrogram_image(gen_waveform),
                step,
            )
        tb.flush()
        logger.info('[OK] Logged generated previews for {}', ckpt_dir.name)
    finally:
        tb.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_preview_command(*, python_executable: str, checkpoint_path: str, output_dir: str, dataset_dir: str, checkpoint_dir: str, model_variant: str, device: str, precision: str, val_split: float, full_sft: bool, log_dir: str | None = None) -> list[str]:
    """Build the detached one-off preview command for a saved checkpoint."""
    cmd = [
        python_executable,
        '-m',
        'acestep.training_v2.tensorboard_checkpoint_preview',
        '--checkpoint-path', checkpoint_path,
        '--output-dir', output_dir,
        '--dataset-dir', dataset_dir,
        '--checkpoint-dir', checkpoint_dir,
        '--model-variant', model_variant,
        '--device', device,
        '--precision', precision,
        '--val-split', str(val_split),
    ]
    if full_sft:
        cmd.append('--full-sft')
    if log_dir:
        cmd.extend(['--log-dir', log_dir])
    return cmd


def launch_generated_preview_job(*, python_executable: str, checkpoint_path: str, output_dir: str, dataset_dir: str, checkpoint_dir: str, model_variant: str, device: str, precision: str, val_split: float, full_sft: bool, log_dir: str | None = None) -> None:
    """Launch a detached one-off preview generation process."""
    cmd = build_preview_command(
        python_executable=python_executable,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        checkpoint_dir=checkpoint_dir,
        model_variant=model_variant,
        device=device,
        precision=precision,
        val_split=val_split,
        full_sft=full_sft,
        log_dir=log_dir,
    )
    kwargs: dict[str, Any] = {
        'stdout': subprocess.DEVNULL,
        'stderr': subprocess.DEVNULL,
        'stdin': subprocess.DEVNULL,
        'cwd': str(Path(output_dir).resolve().parents[2]) if len(Path(output_dir).resolve().parents) >= 3 else None,
        'env': os.environ.copy(),
    }
    if os.name == 'nt':
        kwargs['creationflags'] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs['start_new_session'] = True
    subprocess.Popen(cmd, **kwargs)
    logger.info('[INFO] Spawned background TensorBoard preview job for {}', checkpoint_path)


def main() -> int:
    """CLI entrypoint for one-off checkpoint preview generation."""
    args = _parse_args()
    _write_generated_previews(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
