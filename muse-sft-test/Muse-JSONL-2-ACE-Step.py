#!/usr/bin/env python3
"""convert_muse_to_acestep.py -- Convert Muse JSONL to ACE-Step dataset format."""
# https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Large_Scale_SFT_Training_Guide.md#prerequisites  step 1.3

import json
import os
import shutil
from pathlib import Path
from typing import Optional


def convert_muse_jsonl(
    jsonl_path: str,
    audio_root: str,
    output_dir: str,
    max_samples: Optional[int] = None,
):
    """Convert a Muse JSONL file into ACE-Step's per-song directory layout.

    ACE-Step expects:
        output_dir/
        ├── song_0001.mp3
        ├── song_0001.lyrics.txt
        ├── song_0001.json
        ├── song_0002.mp3
        ...

    Args:
        jsonl_path: Path to train_cn.jsonl or train_en.jsonl
        audio_root: Root directory containing the extracted audio files
        output_dir: Where to write the ACE-Step compatible dataset
        max_samples: Limit number of samples (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and count >= max_samples:
                break

            record = json.loads(line.strip())
            song_id = record["song_id"]
            audio_path = os.path.join(audio_root, record["audio_path"])

            # Skip if audio file doesn't exist
            if not os.path.exists(audio_path):
                print(f"[SKIP] Audio not found: {audio_path}")
                continue

            # Build caption from style field
            style = record.get("style", "")
            caption = style  # e.g. "Pop, Rock, Electronic, Male Vocal, Energetic"

            # Extract lyrics from sections
            lyrics_parts = []
            for section in record.get("sections", []):
                section_type = section.get("section", "")
                text = section.get("text", "").strip()
                if text:
                    lyrics_parts.append(f"[{section_type}]")
                    lyrics_parts.append(text)
                elif section_type == "Intro":
                    lyrics_parts.append("[Intro]")
                elif section_type == "Outro":
                    lyrics_parts.append("[Outro]")
                elif section_type in ("Interlude", "Break"):
                    lyrics_parts.append(f"[{section_type}]")

            lyrics = "\n".join(lyrics_parts)
            if not lyrics.strip():
                lyrics = "[Instrumental]"

            # Detect if instrumental (no lyrics text in any section)
            is_instrumental = all(
                not s.get("text", "").strip()
                for s in record.get("sections", [])
            )

            # Determine BPM from style (not directly in Muse metadata,
            # so we leave it for auto-detection or set a reasonable default)
            # You can use a BPM detection library like librosa if needed.

            # Copy audio file
            ext = Path(audio_path).suffix
            base_name = f"{song_id}_{record.get('track_index', 0)}"
            dest_audio = os.path.join(output_dir, f"{base_name}{ext}")
            if not os.path.exists(dest_audio):
                shutil.copy2(audio_path, dest_audio)

            # Write lyrics file
            lyrics_file = os.path.join(output_dir, f"{base_name}.lyrics.txt")
            with open(lyrics_file, "w", encoding="utf-8") as lf:
                lf.write(lyrics)

            # Write metadata JSON
            meta = {
                "caption": caption,
                "is_instrumental": is_instrumental,
            }

            # Add section-level descriptions as custom_tag for extra context
            descs = [
                s.get("desc", "")
                for s in record.get("sections", [])
                if s.get("desc", "").strip()
            ]
            if descs:
                # Use the first section description as additional context
                meta["custom_tag"] = descs[0][:200]

            meta_file = os.path.join(output_dir, f"{base_name}.json")
            with open(meta_file, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)

            count += 1
            if count % 1000 == 0:
                print(f"[INFO] Converted {count} samples...")

    print(f"[DONE] Converted {count} samples to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to Muse JSONL file")
    parser.add_argument("--audio-root", required=True, help="Root dir with extracted audio")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    convert_muse_jsonl(args.jsonl, args.audio_root, args.output, args.max_samples)
    
