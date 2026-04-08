# Resume Training Commands

Below are the continue-training commands for the three existing output folders under `muse-sft-test/output`.

Assumptions:
- Repo root is `/data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft`
- Dataset tensors are in `./muse-sft-test/preprocessed_tensors`
- Model variant is `turbo`

## 1. `full_sft_muse_3gpu`

This one is full SFT. The checkpoint contains `decoder_state_dict.pt` and `training_state.pt`, so it can resume directly.

```bash
cd /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft

uv run python train.py --yes fixed \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/full_sft_muse_3gpu \
  --full-sft \
  --num-devices 3 \
  --strategy ddp \
  --resume-from ./muse-sft-test/output/full_sft_muse_3gpu/checkpoints/epoch_20_loss_1.2886
```

## 2. `lora_8samples_ddp3_test`

This one is LoRA DDP training. The checkpoint directory contains `adapter/adapter_model.safetensors` and `training_state.pt`, so it can resume directly.

```bash
cd /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft

uv run python train.py --yes fixed \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/lora_8samples_ddp3_test \
  --num-devices 3 \
  --strategy ddp \
  --resume-from ./muse-sft-test/output/lora_8samples_ddp3_test/checkpoints/epoch_50_loss_1.1032
```

## 3. `lora_8samples_single_test`

This one is single-GPU LoRA training. The checkpoint contains `adapter_model.safetensors`, `adapter_config.json`, and `training_state.pt`, so it can resume directly.

```bash
cd /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft

uv run python train.py --yes fixed \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/lora_8samples_single_test \
  --resume-from ./muse-sft-test/output/lora_8samples_single_test/checkpoints/epoch_50_loss_1.0620
```

## Notes

- If you want to continue from an earlier checkpoint, replace only the `--resume-from` path.
- If the original run used different hyperparameters such as `--lr`, `--epochs`, `--batch-size`, or `--gradient-accumulation`, add the same flags back to keep behavior consistent.
- `--epochs` in this trainer means total target epochs, not "add N more epochs". For example, resuming from epoch 50 and training to epoch 100 means setting `--epochs 100`.
