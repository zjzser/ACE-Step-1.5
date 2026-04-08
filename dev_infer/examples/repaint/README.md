# repaint examples

This folder contains small developer examples for `dev_infer` repaint runs.

## Run one example

```bash
python -m dev_infer run \
  --example dev_infer/examples/repaint/example_01.json \
  --checkpoint-dir /abs/path/to/checkpoints \
  --save-dir ./outputs/dev_infer/repaint/example_01 \
  --src-audio /abs/path/to/your/source_audio.opus
```

The example file provides the task shape and repaint parameters. In practice,
you usually override:

- `--checkpoint-dir`
- `--save-dir`
- `--src-audio`

## Notes

- The bundled examples use placeholder `src_audio` values so they are safe to
  commit.
- `example_04.json` and `example_05.json` are based on the official public
  repaint demo metadata.
- The public source links for those official repaint demos are listed inside the
  JSON under `official_reference`.
