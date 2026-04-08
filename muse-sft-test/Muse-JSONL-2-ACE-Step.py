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
    
    
'''
python Muse-JSONL-2-ACE-Step.py \
        --jsonl /data1/zhoujunzuo/Task/2026April/Music-ACE-series/Music_data/submuse_en_2000/train_en.jsonl \
        --audio-root /data1/zhoujunzuo/Task/2026April/Music-ACE-series/Music_data/submuse_en_2000/ \
        --output /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/muse-sft-test/ACE-Step-style/ \
        --max-samples 8
'''


'''

CUDA_VISIBLE_DEVICES=3 python -m acestep.training_v2.cli.train_fixed \
    --preprocess \
    --audio-dir ./muse-sft-test/ACE-Step-style \
    --tensor-output ./muse-sft-test/preprocessed_tensors \
    --checkpoint-dir ../ACE-Step-1.5-lora/checkpoints \
    --model-variant turbo \
    --max-duration 240 \
    --device cuda:0

'''
# 上面这是预处理，来自 https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Large_Scale_SFT_Training_Guide.md#prerequisites 里
# 接上，1.4 写的 但似乎并找不到真正的train代码，而直接运行 python train.py 就直接进入了side step了，所以 走 python train.py 加其他命令就相当于直接启动


# 按照 GPT 的说法，改成下面 168行 这样就可以，果然跑通

# 其中__dummy_output_dir__ 本质上什么都不是, 这里随便填一个路径，只是为了满足参数解析器要求，这次预处理实际不会用到它

# 第一个 export 也可以不用，其实已经每次都导入了  
'''

LD_LIBRARY_PATH="$NPP_LIB:$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
CUDA_VISIBLE_DEVICES=3 python train.py fixed \
                            --preprocess \
                            --audio-dir ./muse-sft-test/ACE-Step-style \
                            --tensor-output ./muse-sft-test/preprocessed_tensors \
                            --checkpoint-dir ../ACE-Step-1.5-lora/checkpoints \
                            --model-variant turbo \
                            --max-duration 240 \
                            --device cuda:0 \
                            --dataset-dir ./__dummy_dataset_dir__ \
                            --output-dir ./__dummy_output_dir__ 
'''


# 多卡 3 卡的一种尝试方案, 每卡大约 6000 MiB的大进程 , 400 MiB 的小进程 对应文档的 2.2

'''
CUDA_VISIBLE_DEVICES=2,3,4 python train.py --yes fixed \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/lora_8samples_ddp3_test \
  --checkpoint-dir ../ACE-Step-1.5-lora/checkpoints \
  --model-variant turbo \
  --device cuda:0 \
  --adapter-type lora \
  --rank 8 \
  --alpha 16 \
  --dropout 0.1 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --epochs 50 \
  --learning-rate 1e-4 \
  --warmup-steps 50 \
  --num-devices 3 \
  --strategy ddp \
  --num-workers 0 \
  --offload-encoder
'''

# 退化 1 卡 auto 的一种尝试方案, 大约 5000 MiB 对应文档的 2.1

'''
CUDA_VISIBLE_DEVICES=3 python train.py --yes fixed \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/lora_8samples_single_test \
  --checkpoint-dir ../ACE-Step-1.5-lora/checkpoints \
  --model-variant turbo \
  --device cuda:0 \
  --adapter-type lora \
  --rank 8 \
  --alpha 16 \
  --dropout 0.1 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --epochs 50 \
  --learning-rate 1e-4 \
  --warmup-steps 50 \
  --num-devices 1 \
  --strategy ddp \
  --offload-encoder
'''

# sft 的训练方式

'''
LD_LIBRARY_PATH="$NPP_LIB:$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
CUDA_VISIBLE_DEVICES=2,3,4 python train.py --yes fixed \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/full_sft_muse_3gpu \
  --checkpoint-dir ../ACE-Step-1.5-lora/checkpoints \
  --model-variant turbo \
  --device cuda:0 \
  --full-sft \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --epochs 20 \
  --learning-rate 1e-5 \
  --warmup-steps 2000 \
  --weight-decay 0.01 \
  --optimizer-type adamw \
  --scheduler-type cosine \
  --gradient-checkpointing \
  --num-devices 3 \
  --strategy ddp \
  --save-every 4
'''
# 这里为了方便监视，给出了另外一版，这个也是在预加载上先集中在指定 3 号卡上面，原因如下：
# 之前的第一次 OOM，发生在进入 Lightning Fabric / DeepSpeed 多卡初始化之前的“单卡预加载瞬间峰值”阶段。具体是 train_fixed.py 调用
# model_loader.load_decoder_for_full_sft() 后，在 loader 里执行 model.to(device) 的那一刻，完整模型会先整体压到一张卡上，这个瞬时峰值曾
# 经超过单卡可用显存。把初始 --device 从映射到物理 GPU 2 的 cuda:0 改为映射到物理 GPU 3 的 cuda:1 后，这个“分布式初
# 始化前的单卡预加载”阶段已能通过，说明之前的第一次 OOM 主要发生在这里。

# 最终也是这一次跑通了
'''
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --yes fixed \
    --dataset-dir ./muse-sft-test/preprocessed_tensors \
    --output-dir ./muse-sft-test/output/full_sft_muse_3gpu \
    --checkpoint-dir ../ACE-Step-1.5-lora/checkpoints \
    --model-variant turbo \
    --device cuda:2 \
    --full-sft \
    --batch-size 1 \
    --gradient-accumulation 4 \
    --epochs 20 --learning-rate 1e-5 \
    --warmup-steps 2000 \
    --weight-decay 0.01 \
    --optimizer-type adamw \
    --scheduler-type cosine \
    --gradient-checkpointing \
    --num-devices 3 \
    --strategy ddp \
    --save-every 4 
'''

# 在这里有一个改动：/data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/acestep/training_v2/trainer_fixed.py line 269 - 270
'''
  针对 DeepSpeed optimizer 做了最小修复。原因是按现有 full SFT 多卡路径已经成功进入 DeepSpeed，但由于配置里启用了 ZeRO optimizer CPU offload，
  而训练代码仍然使用外部构建的普通 AdamW，DeepSpeed 直接报错拒绝启动。GitHub 手册没有覆盖这个分支下当前主干的 optimizer 兼容问题，所以不能只照手册原样
  抄。

  我没有改成另一条更重的路线 DeepSpeedCPUAdam，因为那会扩大改动范围并触碰现有通用 optimizer 构建逻辑。现在采用的是更保守的方案：仅在 DeepSpeed 的
  ds_config 中加入 zero_force_ds_cpu_optimizer: False，允许继续使用现有 AdamW。这样只影响“多卡 full SFT + DeepSpeed”分支，不影响 LoRA、LoKR、单卡训练
  或非 DeepSpeed 路径

'''

#   一处改动
#   - 在 acestep/training_v2/trainer_fixed.py 两处 self.fabric.clip_gradients(...) 外面加了判断
#   - 当 cfg.full_sft and num_devices > 1 时，不再手动裁剪

'''
DeepSpeed 已经在 ds_config 中接管了梯度裁剪，
代码里仍沿用普通 Fabric 路径手动调用 clip_gradients()，两者冲突导致失败。
现已修复为：仅在多卡 full SFT 的 DeepSpeed 分支下跳过手动裁剪，其余训练路径不变。

'''
