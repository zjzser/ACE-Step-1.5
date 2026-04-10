"""Regression tests for resume-mode behavior in the basic training loop."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

training_pkg = types.ModuleType("acestep.training")
training_pkg.__path__ = []
sys.modules.setdefault("acestep.training", training_pkg)

data_module = types.ModuleType("acestep.training.data_module")
data_module.PreprocessedDataModule = object
sys.modules.setdefault("acestep.training.data_module", data_module)
training_pkg.data_module = data_module

from acestep.training_v2.trainer_basic_loop import run_basic_training_loop


class _DummyOptimizer:
    """Minimal optimizer stub for early-loop resume tests."""

    def zero_grad(self, set_to_none: bool = True) -> None:
        return None


class _DummyScheduler:
    """Minimal scheduler stub for early-loop resume tests."""

    def step(self, value=None) -> None:
        return None

    def get_last_lr(self):
        return [1e-5]


class _DummyTensorBoard:
    """Minimal TensorBoard logger stub."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        return None


class TestBasicLoopResumeModes(unittest.TestCase):
    """Resume-mode behavior should differ cleanly between portable and strict."""

    def _make_trainer(self, output_dir: str, resume_from: str, resume_mode: str):
        param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        model = MagicMock()
        model.parameters.return_value = [param]
        model.decoder = MagicMock()
        module = SimpleNamespace(
            model=model,
            device=torch.device("cpu"),
            device_type="cpu",
        )
        cfg = SimpleNamespace(
            output_dir=output_dir,
            effective_log_dir=output_dir,
            learning_rate=1e-5,
            weight_decay=0.01,
            gradient_accumulation_steps=1,
            max_epochs=1,
            warmup_steps=0,
            optimizer_type="adamw",
            scheduler_type="constant",
            gradient_checkpointing=False,
            max_grad_norm=1.0,
            log_every=1,
            log_heavy_every=100,
            save_every_n_epochs=10,
            resume_from=resume_from,
            resume_mode=resume_mode,
            sample_every_n_epochs=0,
        )
        return SimpleNamespace(training_config=cfg, module=module)

    def test_portable_missing_resume_path_warns_then_raises(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(
                output_dir=tmpdir,
                resume_from=str(Path(tmpdir) / "missing_ckpt"),
                resume_mode="portable",
            )
            data_module = SimpleNamespace(train_dataloader=lambda: [object()])

            with (
                patch("acestep.training_v2.trainer_basic_loop.TrainingLogger", _DummyTensorBoard),
                patch("acestep.training_v2.trainer_basic_loop.build_optimizer", return_value=_DummyOptimizer()),
                patch("acestep.training_v2.trainer_basic_loop.build_scheduler", return_value=_DummyScheduler()),
                patch("acestep.training_v2.trainer_basic_loop.select_preview_dataset", return_value=([], None)),
                patch("acestep.training_v2.trainer_basic_loop.collect_preview_samples", return_value=[]),
            ):
                gen = run_basic_training_loop(trainer, data_module, {"should_stop": True})
                next(gen)
                warn = next(gen)
                self.assertEqual("warn", warn.kind)
                self.assertIn("Requested resume checkpoint not found", warn.msg)
                with self.assertRaisesRegex(FileNotFoundError, "Requested resume checkpoint not found"):
                    next(gen)

    def test_strict_missing_resume_path_raises(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(
                output_dir=tmpdir,
                resume_from=str(Path(tmpdir) / "missing_ckpt"),
                resume_mode="strict",
            )
            data_module = SimpleNamespace(train_dataloader=lambda: [object()])

            with (
                patch("acestep.training_v2.trainer_basic_loop.TrainingLogger", _DummyTensorBoard),
                patch("acestep.training_v2.trainer_basic_loop.build_optimizer", return_value=_DummyOptimizer()),
                patch("acestep.training_v2.trainer_basic_loop.build_scheduler", return_value=_DummyScheduler()),
                patch("acestep.training_v2.trainer_basic_loop.select_preview_dataset", return_value=([], None)),
                patch("acestep.training_v2.trainer_basic_loop.collect_preview_samples", return_value=[]),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "Requested resume checkpoint not found"):
                    list(run_basic_training_loop(trainer, data_module, {"should_stop": True}))
