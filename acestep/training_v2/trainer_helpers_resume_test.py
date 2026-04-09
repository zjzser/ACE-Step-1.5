"""Tests for full SFT checkpoint resume fallback behaviour."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from acestep.training_v2.trainer_helpers import resume_checkpoint


class _DummyDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _DummyDecoder()


class _DummyOptimizer:
    def __init__(self) -> None:
        self.loaded = False

    def load_state_dict(self, _state):
        self.loaded = True


class _DummyScheduler:
    def __init__(self) -> None:
        self.loaded = False
        self.stepped_to = None

    def load_state_dict(self, _state):
        self.loaded = True

    def step(self, value=None):
        self.stepped_to = value


class TestResumeCheckpointFullSFT(unittest.TestCase):
    """Validate full SFT resume restores decoder and progress only."""

    def test_full_sft_resume_restores_progress_without_optimizer_state(self) -> None:
        with TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            decoder = _DummyDecoder()
            torch.save(decoder.state_dict(), ckpt_dir / 'decoder_state_dict.pt')
            torch.save(
                {
                    'epoch': 20,
                    'global_step': 123,
                    'optimizer_state_dict': {'bad': True},
                    'scheduler_state_dict': {'ok': True},
                },
                ckpt_dir / 'training_state.pt',
            )

            trainer = SimpleNamespace(
                module=SimpleNamespace(model=_DummyModel(), device=torch.device('cpu')),
                training_config=SimpleNamespace(full_sft=True),
            )
            optimizer = _DummyOptimizer()
            scheduler = _DummyScheduler()

            updates = []
            gen = resume_checkpoint(trainer, str(ckpt_dir), optimizer, scheduler)
            try:
                while True:
                    updates.append(next(gen))
            except StopIteration as stop:
                result = stop.value

            self.assertEqual((20, 123), result)
            self.assertEqual('info', updates[0].kind)
            self.assertIn('training progress only', updates[0].msg)
            self.assertEqual('info', updates[1].kind)
            self.assertIn('epoch 20, step 123', updates[1].msg)
            self.assertFalse(optimizer.loaded)
            self.assertFalse(scheduler.loaded)
            self.assertEqual(123, scheduler.stepped_to)
