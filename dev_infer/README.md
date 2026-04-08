# dev_infer

Developer inference CLI for ACE-Step.

## Verified text2music

Conditions:

- run from repo root: `/data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft`
- Python environment is available
- checkpoints are under `/data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/checkpoints`
- `acestep-v15-turbo` and `acestep-5Hz-lm-1.7B` already exist in that directory
- `uv run ...` is optional and only works if `uv` is installed
- `dev_infer/run_verified_text2music.sh` runs the verified base, LoRA, and full-SFT text2music commands


Run the verified base, LoRA, and full-SFT text2music commands:

```bash
bash dev_infer/run_verified_text2music.sh
```

Run only one variant:

```bash
bash dev_infer/run_verified_text2music.sh base
bash dev_infer/run_verified_text2music.sh lora
bash dev_infer/run_verified_text2music.sh full-sft
```

Use a description file with `python`:

```bash
python -m dev_infer run \
  --baseline base \
  --task text2music \
  --checkpoint-dir /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/checkpoints \
  --description /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/dev_infer/examples/text2music/example_01.json \
  --config-path acestep-v15-turbo \
  --save-dir /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/dev_infer/output/text2music/turbo_example_01 \
  --audio-format wav
```

Use a description file with `uv run`:

```bash
uv run python -m dev_infer run \
  --baseline base \
  --task text2music \
  --checkpoint-dir /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/checkpoints \
  --config-path acestep-v15-turbo \
  --save-dir /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/dev_infer/output/text2music/turbo_example_01 \
  --description /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/dev_infer/examples/text2music/example_01.json \
  --audio-format wav
```

Use direct CLI arguments:

```bash
python -m dev_infer run \
  --baseline base \
  --task text2music \
  --checkpoint-dir /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/checkpoints \
  --config-path acestep-v15-turbo \
  --save-dir /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/dev_infer/output/text2music/turbo_manual_01 \
  --caption "Warm indie folk with soft female vocal, acoustic guitar, light cello, and a calm evening mood." \
  --lyrics "[Verse]\nHold the dusk inside your hands\nLet the city fade to gold\n[Chorus]\nStay with me through quiet light\nKeep this little fire warm" \
  --bpm 92 \
  --duration 45 \
  --keyscale "C major" \
  --vocal-language en \
  --audio-format wav
```

Use LoRA with a description file:

```bash
python -m dev_infer run \
  --baseline lora \
  --task text2music \
  --checkpoint-dir /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/checkpoints \
  --config-path acestep-v15-turbo \
  --lora-path /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/muse-sft-test/output/lora_8samples_ddp3_test/final/adapter \
  --lora-scale 1.0 \
  --save-dir /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/dev_infer/output/text2music/lora_turbo_example_04 \
  --description /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/dev_infer/examples/text2music/example_04.json \
  --audio-format wav
```

Use full-SFT with a description file:

```bash
python -m dev_infer run \
  --baseline full-sft \
  --task text2music \
  --checkpoint-dir /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/checkpoints \
  --config-path acestep-v15-turbo \
  --full-sft-path /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/muse-sft-test/output/full_sft_muse_3gpu/final/decoder_state_dict.pt \
  --save-dir /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/dev_infer/output/text2music/full_sft_turbo_example_04 \
  --description /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/dev_infer/examples/text2music/example_04.json \
  --audio-format wav
```

## Outputs

Each run writes:

- audio files
- `metadata.json`
- `example.original.json` and `example.resolved.json` when `--description` is used
- `prompt.txt`
- `lyrics.txt` when lyrics are present
- `status.txt`
- `metadata.json.load_summary` records LoRA or full-SFT load details when used
- audio files are renamed to match the output directory name, for example `turbo_example_04.wav`
- examples here use `--audio-format wav`
- `--audio-format` supports `flac`, `wav`, `mp3`, `wav32`, `opus`, and `aac`
