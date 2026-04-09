"""Tests for background TensorBoard checkpoint preview helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from acestep.training_v2.tensorboard_checkpoint_preview import (
    _resolve_lora_adapter_path,
    build_preview_command,
)
from acestep.training_v2.tensorboard_preview import select_preview_dataset


class TestCheckpointPreviewCommand(unittest.TestCase):
    """Validate detached preview command construction."""

    def test_build_preview_command_for_lora(self) -> None:
        cmd = build_preview_command(
            python_executable='python',
            checkpoint_path='/tmp/epoch_10',
            output_dir='/tmp/out',
            dataset_dir='/tmp/data',
            checkpoint_dir='/tmp/base',
            model_variant='turbo',
            device='cuda:0',
            precision='bf16',
            val_split=0.1,
            full_sft=False,
        )
        self.assertIn('acestep.training_v2.tensorboard_checkpoint_preview', cmd)
        self.assertIn('--checkpoint-path', cmd)
        self.assertIn('/tmp/epoch_10', cmd)
        self.assertNotIn('--full-sft', cmd)

    def test_build_preview_command_for_full_sft(self) -> None:
        cmd = build_preview_command(
            python_executable='python',
            checkpoint_path='/tmp/epoch_20',
            output_dir='/tmp/out',
            dataset_dir='/tmp/data',
            checkpoint_dir='/tmp/base',
            model_variant='turbo',
            device='cuda:2',
            precision='bf16',
            val_split=0.2,
            full_sft=True,
        )
        self.assertIn('--full-sft', cmd)


class TestPreviewSelection(unittest.TestCase):
    """Validation previews should prefer held-out data when present."""

    def test_select_preview_dataset_prefers_validation(self) -> None:
        data_module = SimpleNamespace(
            train_dataset=[{'metadata': {'filename': 'train.wav'}}],
            val_dataset=[{'metadata': {'filename': 'val.wav'}}],
        )
        dataset, source = select_preview_dataset(data_module)
        self.assertEqual('validation', source)
        self.assertEqual('val.wav', dataset[0]['metadata']['filename'])


class TestLoraAdapterPath(unittest.TestCase):
    """Validate LoRA adapter directory resolution for checkpoint previews."""

    def test_prefers_adapter_subdir_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / 'epoch_10'
            adapter_dir = ckpt_dir / 'adapter'
            adapter_dir.mkdir(parents=True)
            resolved = _resolve_lora_adapter_path(ckpt_dir)
            self.assertEqual(adapter_dir, resolved)

    def test_falls_back_to_checkpoint_dir_without_adapter_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / 'epoch_10'
            ckpt_dir.mkdir(parents=True)
            resolved = _resolve_lora_adapter_path(ckpt_dir)
            self.assertEqual(ckpt_dir, resolved)


if __name__ == '__main__':
    unittest.main()
