"""Unit tests for full SFT setup in FixedLoRAModule."""

import sys
import types
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

training_pkg = types.ModuleType("acestep.training")
training_pkg.__path__ = []
sys.modules.setdefault("acestep.training", training_pkg)

configs_mod = types.ModuleType("acestep.training.configs")


class _TrainingConfig:
    def to_dict(self):
        return self.__dict__.copy()


class _LoRAConfig:
    def to_dict(self):
        return self.__dict__.copy()


class _LoKRConfig:
    def to_dict(self):
        return self.__dict__.copy()


configs_mod.LoRAConfig = _LoRAConfig
configs_mod.LoKRConfig = _LoKRConfig
configs_mod.TrainingConfig = _TrainingConfig
sys.modules["acestep.training.configs"] = configs_mod
training_pkg.configs = configs_mod

lora_injection = types.ModuleType("acestep.training.lora_injection")
lora_injection.inject_lora_into_dit = lambda model, cfg: (model, {})
sys.modules["acestep.training.lora_injection"] = lora_injection
training_pkg.lora_injection = lora_injection

lora_utils = types.ModuleType("acestep.training.lora_utils")
lora_utils.check_peft_available = lambda: True
sys.modules["acestep.training.lora_utils"] = lora_utils
training_pkg.lora_utils = lora_utils

lokr_utils = types.ModuleType("acestep.training.lokr_utils")
lokr_utils.check_lycoris_available = lambda: True
lokr_utils.inject_lokr_into_dit = lambda model, cfg: (model, None, {})
lokr_utils.load_lokr_weights = lambda *args, **kwargs: None
lokr_utils.save_lokr_weights = lambda *args, **kwargs: None
sys.modules["acestep.training.lokr_utils"] = lokr_utils
training_pkg.lokr_utils = lokr_utils

from acestep.training_v2.fixed_lora_module import FixedLoRAModule


class _DummyModel(nn.Module):
    """Minimal model with decoder and non-decoder parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Linear(4, 4)
        self.music_encoder = nn.Linear(4, 4)
        self.config = SimpleNamespace()
        self.null_condition_emb = nn.Parameter(torch.zeros(1, 4))


class TestFixedLoRAModuleFullSFT(unittest.TestCase):
    """Full SFT mode should skip adapter injection and unfreeze only decoder params."""

    def test_full_sft_keeps_only_decoder_trainable(self):
        """Decoder params stay trainable while non-decoder params remain frozen."""
        model = _DummyModel()
        training_cfg = SimpleNamespace(
            adapter_type="lora",
            full_sft=True,
            timestep_mu=-0.4,
            timestep_sigma=1.0,
            data_proportion=0.5,
            cfg_ratio=0.15,
        )

        module = FixedLoRAModule(
            model=model,
            adapter_config=SimpleNamespace(),
            training_config=training_cfg,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertIs(module.model, model)
        self.assertTrue(all(p.requires_grad for p in model.decoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.music_encoder.parameters()))
        self.assertEqual(module.adapter_info, {})


if __name__ == "__main__":
    unittest.main()
