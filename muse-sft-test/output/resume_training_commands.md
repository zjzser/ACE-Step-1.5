# 断点继续训练说明

本文档整理 `muse-sft-test/output` 目录下当前实验相关的断点继续训练方式，并结合本仓库实际实现，说明 `LoRA` 与 `full SFT` 两条路径的恢复语义、限制与建议用法。

## 一、统一前提

默认仓库根目录：

```bash
/data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft
```

默认数据目录：

```bash
./muse-sft-test/preprocessed_tensors
```

默认模型变体：

```bash
turbo
```

## 二、当前仓库里两类“继续训练”的真实含义

### 1. LoRA 的 `--resume-from`

LoRA 路径当前实现的是“相对完整的断点续训”。

恢复时会尝试加载：

- LoRA adapter 权重
- `optimizer_state_dict`
- `scheduler_state_dict`
- `epoch`
- `global_step`

也就是说，LoRA 的 `--resume-from` 语义更接近“恢复原训练现场”。

但是需要注意：

- 它并不会恢复 DataLoader / sampler / RNG 的完整状态。
- 如果恢复时把 `--epochs` 改得很大，尤其在使用 `cosine` scheduler 时，训练曲线未必表现为严格连续。
- 因此，LoRA 的 `resume` 更适合“训练中断后按原计划继续跑”，不太适合直接拿来做“第二阶段长期延长训练”。

### 2. full SFT 的 `--resume-from`

full SFT 路径当前实现的是“弱恢复”。

恢复时只会加载：

- `decoder_state_dict.pt`
- `epoch`
- `global_step`

然后只会尝试按 `global_step` 对齐 scheduler 的当前位置。

它**不会恢复**：

- optimizer state
- scheduler state dict 本体

这是当前仓库的有意设计，不是偶然遗漏。源码中的说明是：full SFT 为了 portability，故意跳过 DeepSpeed optimizer/scheduler state 的恢复。

因此，full SFT 的 `--resume-from` 更准确的语义是：

- 从上一个 decoder 权重继续训练
- 不是严格意义上的无缝断点续训

恢复点附近出现明显 loss 跳变是预期现象。

## 三、已经验证过的实验结论

### 1. full SFT

实验现象：

- 可以从 checkpoint 成功继续训练。
- 恢复点附近存在明显 loss 抬升。
- 后续仍能重新下降并继续收敛。

结论：

- full SFT 已验证“可以继续训练”。
- 但它不是严格连续训练，更像“从已有 decoder 权重继续 finetune”。

### 2. LoRA

实验现象：

- 可以从 checkpoint 成功继续训练。
- `training_state.pt` 中确实保存了 optimizer 与 scheduler 状态。
- 恢复后仍可能出现较明显波动，尤其在修改总 `epochs` 的情况下更明显。

结论：

- LoRA 已验证“checkpoint 的保存与加载逻辑确实生效”。
- 但如果把原本较短的训练计划直接延长很多，resume 后的曲线不一定具备理想的连续性。

## 四、为什么改 `--epochs` 会影响 LoRA 恢复表现

本仓库的 scheduler 是按下面的公式构造的：

```text
total_steps = steps_per_epoch * max_epochs
```

因此即使下面这些参数都不变：

- learning rate
- warmup steps
- batch size
- gradient accumulation
- 数据集
- GPU 数量
- optimizer type
- scheduler type

只要 `--epochs` 变了，训练计划本身就已经不是同一条学习率曲线。

对于 `cosine` scheduler，这种影响尤其明显。

所以：

- 如果目的是验证“断点续训是否真的接上”，恢复时应尽量保持原训练计划不变，只做小幅延续。
- 如果目的是“从 40 继续拉长到 120”，更合理的工程语义应该是“第二阶段训练”，而不是简单复用原来的 `--resume-from`。

## 五、当前已有 checkpoint 对应命令

### 1. full SFT：从 `epoch_30_loss_1.3470` 继续到总 `60` epoch

使用 1、3、4 号卡：

```bash
cd /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft

CUDA_VISIBLE_DEVICES=1,3,4 uv run python train.py --yes fixed \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/full_sft_muse_3gpu \
  --full-sft \
  --num-devices 3 \
  --strategy ddp \
  --epochs 60 \
  --resume-from ./muse-sft-test/output/full_sft_muse_3gpu/checkpoints/epoch_30_loss_1.3470
```

说明：

- `--epochs 60` 表示总目标 epoch 为 60，不是“再跑 60 个”。
- full SFT 即使恢复成功，也不代表 optimizer 历史被接回。

### 2. LoRA DDP：从 `epoch_40_loss_1.0839` 继续到总 `120` epoch，每 20 个 epoch 保存一次

```bash
cd /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft

CUDA_VISIBLE_DEVICES=1,3,4 python train.py --yes fixed \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/lora_8samples_ddp3_test \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --device cuda:0 \
  --adapter-type lora \
  --rank 8 \
  --alpha 16 \
  --dropout 0.1 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --epochs 120 \
  --learning-rate 1e-4 \
  --warmup-steps 50 \
  --num-devices 3 \
  --strategy ddp \
  --num-workers 0 \
  --offload-encoder \
  --save-every 20 \
  --val-split 0.1 \
  --resume-from ./muse-sft-test/output/lora_8samples_ddp3_test/checkpoints/epoch_40_loss_1.0839
```

说明：

- 这条命令可以跑，也会恢复 LoRA checkpoint。
- 但如果用于“从 40 大幅拉长到 120”，应把它理解为“实验性延长训练”，不要把它当成严格连续的 resume 基准。
- 如果后续要做更合理的第二阶段训练，建议增加“只加载 adapter 权重、不恢复 optimizer/scheduler”的单独入口。

## 六、如何判断恢复是否真的成功

### 1. LoRA

如果 LoRA 恢复成功，日志里通常应出现类似信息：

```text
[OK] Resumed from epoch 40, step ..., optimizer OK, scheduler OK
```

这意味着：

- adapter 权重已加载
- optimizer state 已加载
- scheduler state 已加载

### 2. full SFT

如果 full SFT 恢复成功，日志里通常应出现类似信息：

```text
[OK] Resumed full SFT decoder from epoch 30, step ...
```

这只表示：

- decoder 权重已加载
- epoch / step 已恢复

不代表 optimizer/scheduler state 被完整恢复。

## 七、当前建议

### 1. 如果要验证 resume 机制本身

建议：

- 不改变训练配置
- 只做短距离续跑
- 例如从 40 接到 50，而不是直接接到 120

### 2. 如果要做第二阶段长期训练

建议：

- 不要简单复用原来的 `--resume-from` 语义
- 更合理的是：只加载已有权重，重新建立 optimizer 与 scheduler，作为新的训练阶段继续

对于本仓库而言：

- LoRA 当前缺少“仅加载 adapter 权重继续训练”的单独 CLI 入口。
- full SFT 当前缺少“严格恢复 optimizer/scheduler state”的实现。

这两点都可以作为后续改进方向。
