#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft"
CHECKPOINT_DIR="$ROOT/checkpoints"
DESCRIPTION="$ROOT/dev_infer/examples/text2music/example_04.json"
OUTPUT_DIR="$ROOT/dev_infer/output/text2music"
LORA_PATH="$ROOT/muse-sft-test/output/lora_8samples_ddp3_test/final/adapter"
FULL_SFT_PATH="$ROOT/muse-sft-test/output/full_sft_muse_3gpu/final/decoder_state_dict.pt"
MODE="${1:-all}"

run_base() {
  python -m dev_infer run \
    --baseline base \
    --task text2music \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config-path acestep-v15-turbo \
    --save-dir "$OUTPUT_DIR/turbo_example_04" \
    --description "$DESCRIPTION" \
    --audio-format wav
}

run_lora() {
  python -m dev_infer run \
    --baseline lora \
    --task text2music \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config-path acestep-v15-turbo \
    --lora-path "$LORA_PATH" \
    --lora-scale 1.0 \
    --save-dir "$OUTPUT_DIR/lora_turbo_example_04" \
    --description "$DESCRIPTION" \
    --audio-format wav
}

run_full_sft() {
  python -m dev_infer run \
    --baseline full-sft \
    --task text2music \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config-path acestep-v15-turbo \
    --full-sft-path "$FULL_SFT_PATH" \
    --save-dir "$OUTPUT_DIR/full_sft_turbo_example_04" \
    --description "$DESCRIPTION" \
    --audio-format wav
}

case "$MODE" in
  base)
    run_base
    ;;
  lora)
    run_lora
    ;;
  full-sft)
    run_full_sft
    ;;
  all)
    run_base
    run_lora
    run_full_sft
    ;;
  *)
    echo "Usage: $0 [all|base|lora|full-sft]" >&2
    exit 1
    ;;
esac
