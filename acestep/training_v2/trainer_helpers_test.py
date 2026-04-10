"""Unit tests for acestep.training_v2.trainer_helpers.save_adapter_flat.

These tests use only stdlib and unittest.mock so that torch is not required.
The save_adapter_flat function only needs attribute-level duck-typing, so
plain MagicMock objects work as stand-ins for nn.Module subclasses.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


training_pkg = types.ModuleType("acestep.training")
training_pkg.__path__ = []
sys.modules.setdefault("acestep.training", training_pkg)

lora_checkpoint = types.ModuleType("acestep.training.lora_checkpoint")
def _load_training_checkpoint(*args, **kwargs):
    return {}

def _save_lora_weights(*args, **kwargs):
    return None

lora_checkpoint.load_training_checkpoint = _load_training_checkpoint
lora_checkpoint.save_lora_weights = _save_lora_weights
sys.modules["acestep.training.lora_checkpoint"] = lora_checkpoint
training_pkg.lora_checkpoint = lora_checkpoint

lokr_utils = types.ModuleType("acestep.training.lokr_utils")
def _load_lokr_weights(*args, **kwargs):
    return None

def _save_lokr_weights(*args, **kwargs):
    return None

lokr_utils.load_lokr_weights = _load_lokr_weights
lokr_utils.save_lokr_weights = _save_lokr_weights
sys.modules["acestep.training.lokr_utils"] = lokr_utils
training_pkg.lokr_utils = lokr_utils


def _make_fabric_wrapper(inner: MagicMock) -> MagicMock:
    """Return a mock that looks like a Fabric _FabricModule wrapping *inner*."""
    wrapper = MagicMock(spec_set=["_forward_module"])
    wrapper._forward_module = inner
    return wrapper


def _make_peft_decoder() -> MagicMock:
    """Return a mock that looks like a PEFT PeftModel-wrapped decoder."""
    decoder = MagicMock(spec=["save_pretrained"])
    return decoder


def _make_base_decoder() -> MagicMock:
    """Return a mock with no save_pretrained (raw non-PEFT decoder fallback)."""
    return MagicMock(spec=[])  # no save_pretrained attribute


def _make_full_model(decoder: MagicMock) -> MagicMock:
    """Return a mock that looks like AceStepConditionGenerationModel.

    The full model also has save_pretrained (inherits from HF PreTrainedModel)
    -- that method must NOT be invoked; only the decoder's save_pretrained
    should be called.
    """
    model = MagicMock(spec=["decoder", "save_pretrained"])
    model.decoder = decoder
    model.save_pretrained = MagicMock()
    return model


def _make_trainer(
    decoder: MagicMock,
    adapter_type: str = "lora",
    *,
    full_sft: bool = False,
) -> MagicMock:
    """Build a minimal trainer mock with the given decoder embedded."""
    full_model = _make_full_model(decoder)
    module = MagicMock()
    module.model = full_model
    trainer = MagicMock()
    trainer.adapter_type = adapter_type
    trainer.module = module
    trainer.training_config = MagicMock(full_sft=full_sft)
    return trainer


class TestSaveAdapterFlatLora(unittest.TestCase):
    """save_adapter_flat must write adapter files (not the full model) for LoRA."""

    def test_calls_save_pretrained_on_peft_decoder_not_full_model(self):
        """save_pretrained() must be called on the PeftModel decoder, not the full model."""
        peft_decoder = _make_peft_decoder()
        trainer = _make_trainer(peft_decoder)

        from acestep.training_v2.trainer_helpers import save_adapter_flat

        with patch("os.makedirs"):
            save_adapter_flat(trainer, "/tmp/out")

        # The PeftModel's save_pretrained must have been invoked.
        peft_decoder.save_pretrained.assert_called_once_with("/tmp/out")
        # The full model's save_pretrained must NOT have been called.
        trainer.module.model.save_pretrained.assert_not_called()

    def test_unwraps_fabric_wrapper_before_saving(self):
        """When Fabric wraps the decoder, save_pretrained is still called on the PeftModel."""
        peft_decoder = _make_peft_decoder()
        fabric_wrapped = _make_fabric_wrapper(peft_decoder)
        trainer = _make_trainer(fabric_wrapped)

        from acestep.training_v2.trainer_helpers import save_adapter_flat

        with patch("os.makedirs"):
            save_adapter_flat(trainer, "/tmp/out")

        peft_decoder.save_pretrained.assert_called_once_with("/tmp/out")
        trainer.module.model.save_pretrained.assert_not_called()

    def test_doubly_wrapped_fabric_still_reaches_peft_decoder(self):
        """Handles two layers of Fabric _FabricModule wrapping."""
        peft_decoder = _make_peft_decoder()
        fabric_inner = _make_fabric_wrapper(peft_decoder)
        fabric_outer = _make_fabric_wrapper(fabric_inner)
        trainer = _make_trainer(fabric_outer)

        from acestep.training_v2.trainer_helpers import save_adapter_flat

        with patch("os.makedirs"):
            save_adapter_flat(trainer, "/tmp/out")

        peft_decoder.save_pretrained.assert_called_once_with("/tmp/out")

    def test_fallback_to_save_lora_weights_when_no_save_pretrained(self):
        """If the decoder has no save_pretrained, save_lora_weights is used as fallback."""
        raw_decoder = _make_base_decoder()
        trainer = _make_trainer(raw_decoder)

        from acestep.training_v2.trainer_helpers import save_adapter_flat

        with (
            patch("os.makedirs"),
            patch(
                "acestep.training_v2.trainer_helpers.save_lora_weights"
            ) as mock_slw,
        ):
            save_adapter_flat(trainer, "/tmp/out")

        mock_slw.assert_called_once_with(trainer.module.model, "/tmp/out")

    def test_regression_full_model_save_pretrained_not_called_for_lora(self):
        """Regression: the full AceStep model's save_pretrained must never be called.

        Previously, _unwrap_decoder(module.model) returned the full
        AceStepConditionGenerationModel (which also has save_pretrained), causing
        model.safetensors + config.json to be written instead of adapter files.
        """
        peft_decoder = _make_peft_decoder()
        trainer = _make_trainer(peft_decoder)

        from acestep.training_v2.trainer_helpers import save_adapter_flat

        with patch("os.makedirs"):
            save_adapter_flat(trainer, "/tmp/out")

        # Critically, the full model's save_pretrained must not have been called.
        trainer.module.model.save_pretrained.assert_not_called()


DeepSpeedStrategy = type("DeepSpeedStrategy", (), {})


class TestSaveCheckpointFullSFTDistributed(unittest.TestCase):
    """Distributed full SFT checkpoints should use Fabric save on all ranks."""

    def test_save_checkpoint_uses_fabric_save_for_deepspeed_full_sft(self):
        decoder = MagicMock(spec=["state_dict"])
        decoder.state_dict.return_value = {"w": MagicMock()}
        fabric = MagicMock()
        fabric.world_size = 3
        fabric.global_rank = 1
        fabric.strategy = DeepSpeedStrategy()
        trainer = _make_trainer(decoder, full_sft=True)
        trainer.fabric = fabric

        from acestep.training_v2.trainer_helpers import save_checkpoint

        optimizer = MagicMock()
        optimizer.state_dict.return_value = {"ds": True}
        scheduler = MagicMock()
        scheduler.state_dict.return_value = {"sched": True}

        with (
            patch("torch.save") as mock_torch_save,
            patch("acestep.training_v2.trainer_helpers.save_adapter_flat") as mock_save_adapter,
            patch("acestep.training_v2.trainer_helpers._uses_deepspeed_full_sft", return_value=True),
        ):
            save_checkpoint(trainer, optimizer, scheduler, 5, 12, "/tmp/out")

        fabric.save.assert_called_once()
        save_args, _ = fabric.save.call_args
        self.assertEqual("/tmp/out/distributed", save_args[0])
        self.assertEqual(5, save_args[1]["epoch"])
        self.assertEqual(12, save_args[1]["global_step"])
        mock_save_adapter.assert_not_called()
        mock_torch_save.assert_not_called()


class TestSaveAdapterFlatFullSFT(unittest.TestCase):
    """Full SFT saves decoder weights instead of adapter-only artifacts."""

    def test_saves_decoder_state_and_optional_pretrained_layout(self):
        """Full SFT mode should persist the decoder state dict and model layout."""
        decoder = MagicMock(spec=["state_dict"])
        decoder.state_dict.return_value = {"w": MagicMock()}
        trainer = _make_trainer(decoder, full_sft=True)

        from acestep.training_v2.trainer_helpers import save_adapter_flat

        with (
            patch("os.makedirs"),
            patch("torch.save") as mock_torch_save,
        ):
            save_adapter_flat(trainer, "/tmp/out")

        mock_torch_save.assert_called_once()
        trainer.module.model.save_pretrained.assert_called_once_with("/tmp/out")

    def test_save_final_skips_adapter_verification_for_full_sft(self):
        """Full SFT final save should not run adapter-only verification."""
        decoder = MagicMock(spec=["state_dict"])
        decoder.state_dict.return_value = {}
        trainer = _make_trainer(decoder, full_sft=True)

        from acestep.training_v2.trainer_helpers import save_final

        with (
            patch("os.makedirs"),
            patch("torch.save"),
            patch("acestep.training_v2.trainer_helpers.verify_saved_adapter") as mock_verify,
        ):
            save_final(trainer, "/tmp/out")

        mock_verify.assert_not_called()


