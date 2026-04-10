"""Tests for full SFT checkpoint resume behaviour."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    def __init__(self, should_fail: bool = False) -> None:
        self.loaded = False
        self.should_fail = should_fail
        self.loaded_state = None

    def load_state_dict(self, state):
        if self.should_fail:
            raise RuntimeError("optimizer mismatch")
        self.loaded = True
        self.loaded_state = state


class _DummyScheduler:
    def __init__(self, should_fail: bool = False) -> None:
        self.loaded = False
        self.should_fail = should_fail
        self.loaded_state = None
        self.stepped_to = None

    def load_state_dict(self, state):
        if self.should_fail:
            raise RuntimeError("scheduler mismatch")
        self.loaded = True
        self.loaded_state = state

    def step(self, value=None):
        self.stepped_to = value


DeepSpeedStrategy = type("DeepSpeedStrategy", (), {})


class TestResumeCheckpointFullSFT(unittest.TestCase):
    """Validate full SFT resume restores decoder, progress, and training state."""

    def _build_checkpoint_dir(self) -> Path:
        tmpdir = TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        ckpt_dir = Path(tmpdir.name)
        decoder = _DummyDecoder()
        torch.save(decoder.state_dict(), ckpt_dir / "decoder_state_dict.pt")
        torch.save(
            {
                "epoch": 20,
                "global_step": 123,
                "optimizer_state_dict": {
                    "state": {0: {"exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1)}},
                    "param_groups": [{"lr": 1e-4}],
                },
                "scheduler_state_dict": {"last_epoch": 123},
            },
            ckpt_dir / "training_state.pt",
        )
        return ckpt_dir

    def test_full_sft_resume_prefers_distributed_checkpoint_when_available(self) -> None:
        ckpt_dir = self._build_checkpoint_dir()
        distributed_dir = ckpt_dir / "distributed"
        distributed_dir.mkdir()
        fabric = MagicMock()
        fabric.world_size = 3
        fabric.strategy = DeepSpeedStrategy()
        fabric.load.return_value = {"epoch": 20, "global_step": 123}
        trainer = SimpleNamespace(
            module=SimpleNamespace(model=_DummyModel(), device=torch.device("cpu")),
            training_config=SimpleNamespace(full_sft=True, resume_mode="strict"),
            fabric=fabric,
        )
        optimizer = _DummyOptimizer()
        scheduler = _DummyScheduler()

        updates = []
        with patch("acestep.training_v2.trainer_helpers._uses_deepspeed_full_sft", return_value=True):
            gen = resume_checkpoint(trainer, str(ckpt_dir), optimizer, scheduler)
            try:
                while True:
                    updates.append(next(gen))
            except StopIteration as stop:
                result = stop.value

        self.assertEqual((20, 123), result)
        fabric.load.assert_called_once()
        load_args, load_kwargs = fabric.load.call_args
        self.assertEqual(distributed_dir, Path(load_args[0]))
        self.assertFalse(load_kwargs["strict"])
        self.assertEqual("info", updates[0].kind)
        self.assertIn("optimizer OK", updates[0].msg)
        self.assertIn("scheduler OK", updates[0].msg)

    def test_full_sft_resume_strict_mode_requires_distributed_checkpoint(self) -> None:
        ckpt_dir = self._build_checkpoint_dir()
        fabric = MagicMock()
        fabric.world_size = 3
        fabric.strategy = DeepSpeedStrategy()
        trainer = SimpleNamespace(
            module=SimpleNamespace(model=_DummyModel(), device=torch.device("cpu")),
            training_config=SimpleNamespace(full_sft=True, resume_mode="strict"),
            fabric=fabric,
        )

        with patch("acestep.training_v2.trainer_helpers._uses_deepspeed_full_sft", return_value=True):
            gen = resume_checkpoint(trainer, str(ckpt_dir), _DummyOptimizer(), _DummyScheduler())
            with self.assertRaisesRegex(RuntimeError, "Strict full SFT resume requires a distributed checkpoint"):
                next(gen)

    def test_full_sft_resume_portable_mode_ignores_distributed_checkpoint(self) -> None:
        ckpt_dir = self._build_checkpoint_dir()
        (ckpt_dir / "distributed").mkdir()
        fabric = MagicMock()
        fabric.world_size = 3
        fabric.strategy = DeepSpeedStrategy()
        trainer = SimpleNamespace(
            module=SimpleNamespace(model=_DummyModel(), device=torch.device("cpu")),
            training_config=SimpleNamespace(full_sft=True, resume_mode="portable"),
            fabric=fabric,
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
        fabric.load.assert_not_called()
        self.assertTrue(optimizer.loaded)
        self.assertTrue(scheduler.loaded)
        self.assertIn("optimizer OK", updates[0].msg)
        self.assertIn("scheduler OK", updates[0].msg)

    def test_full_sft_resume_restores_optimizer_and_scheduler_when_available(self) -> None:
        ckpt_dir = self._build_checkpoint_dir()
        trainer = SimpleNamespace(
            module=SimpleNamespace(model=_DummyModel(), device=torch.device("cpu")),
            training_config=SimpleNamespace(full_sft=True, resume_mode="strict"),
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
        self.assertEqual("info", updates[0].kind)
        self.assertIn("optimizer OK", updates[0].msg)
        self.assertIn("scheduler OK", updates[0].msg)
        self.assertEqual("info", updates[1].kind)
        self.assertIn("epoch 20, step 123", updates[1].msg)
        self.assertTrue(optimizer.loaded)
        self.assertTrue(scheduler.loaded)
        self.assertIsNone(scheduler.stepped_to)
        self.assertEqual(1, len(optimizer.loaded_state["state"]))

    def test_full_sft_resume_falls_back_when_optimizer_and_scheduler_restore_fail(self) -> None:
        ckpt_dir = self._build_checkpoint_dir()
        trainer = SimpleNamespace(
            module=SimpleNamespace(model=_DummyModel(), device=torch.device("cpu")),
            training_config=SimpleNamespace(full_sft=True, resume_mode="strict"),
        )
        optimizer = _DummyOptimizer(should_fail=True)
        scheduler = _DummyScheduler(should_fail=True)

        updates = []
        gen = resume_checkpoint(trainer, str(ckpt_dir), optimizer, scheduler)
        try:
            while True:
                updates.append(next(gen))
        except StopIteration as stop:
            result = stop.value

        self.assertEqual((20, 123), result)
        self.assertEqual("warn", updates[0].kind)
        self.assertIn("fresh optimizer", updates[0].msg)
        self.assertEqual("warn", updates[1].kind)
        self.assertIn("step alignment", updates[1].msg)
        self.assertEqual("info", updates[2].kind)
        self.assertIn("fallback mode", updates[2].msg)
        self.assertEqual("info", updates[3].kind)
        self.assertIn("epoch 20, step 123", updates[3].msg)
        self.assertFalse(optimizer.loaded)
        self.assertFalse(scheduler.loaded)
        self.assertEqual(123, scheduler.stepped_to)
