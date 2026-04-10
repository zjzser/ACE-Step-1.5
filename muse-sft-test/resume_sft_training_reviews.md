# SFT 断点续训说明

## 1. 这份文档的目的

这份文档专门说明 `muse-sft-test` 里的 full SFT 训练在中断后如何继续训练，以及当前仓库里 `portable` / `strict` 两种 resume 语义分别代表什么。

这里讨论的对象是：
- `--full-sft`
- 多卡 DDP / DeepSpeed 路径
- `muse-sft-test/output/full_sft_*` 目录下的 checkpoint

---

## 2. 之前的问题是什么

这一节说的是历史行为，不是当前最终行为。

最早的 full SFT resume 只能恢复：
- `decoder_state_dict.pt`
- `epoch`
- `global_step`

但不能完整恢复 DeepSpeed ZeRO 的 optimizer / scheduler 分布式状态。

因此表现为：
- 训练虽然能从某个 checkpoint 继续跑
- 但 loss 在恢复点附近会明显跳变
- 更像“从已有权重继续训练”
- 而不是“严格恢复上一次训练现场”

这也是为什么早期 full SFT 的 resume 曲线常常在恢复点处不连续。

当前仓库已经在这个基础上继续演进为两套模式：
- `portable`：保留轻量恢复能力
- `strict`：引入 `distributed/`，支持更严格的 distributed resume

---

## 3. 这两个新参数到底是干什么的

这次新增了两个可选参数，专门用来把 full SFT 的断点续训语义说清楚：
- `--save-mode portable|strict`
- `--resume-mode portable|strict`

如果你不写，默认都是：
- `portable`

理解这两个参数时，可以先记住一句最重要的话：
- `save-mode` 决定“保存什么”
- `resume-mode` 决定“恢复时优先按哪种语义加载”

### 3.1 `--save-mode portable`

这是默认保存模式。

它的精神是：
- 先把训练保存下来
- 但不要背上太重的 distributed 包袱
- 节省磁盘空间
- 给以后更灵活的继续训练留空间

具体行为是：
- 不保存 `distributed/`
- 只保存平铺 checkpoint
- 例如：
  - `decoder_state_dict.pt`
  - `training_state.pt`
  - 其他普通保存文件

适合：
- 日常实验
- 磁盘比较紧张
- 后续可能换卡数、换环境继续训练

代价是：
- 以后不能做真正 strict distributed resume
- 更像“基于已有权重和普通状态继续训练”

### 3.2 `--save-mode strict`

这是严格保存模式。

它的精神是：
- 既保存普通 checkpoint
- 也额外保存 strict distributed resume 所需的完整现场

具体行为是：
- 会额外保存 `distributed/`

这个目录里保存的是：
- DeepSpeed 分布式模型状态
- optimizer shard
- scheduler 相关状态
- strict 恢复需要的拓扑信息

适合：
- 想严格恢复训练现场
- 想尽量保证恢复点前后连续性
- 愿意接受更大的磁盘占用

代价是：
- checkpoint 明显更大
- 后面 strict resume 时，对环境一致性的要求更高

### 3.3 `--resume-mode portable`

这是默认恢复模式。

它的精神是：
- 恢复时尽量灵活
- 不强依赖 `distributed/`
- 优先把训练继续下去

具体行为是：
- 无视保留的 `distributed/`
- 只用 checkpoint 外层的平铺文件恢复

也就是说它更像：
- 土法把主要参数和普通训练状态接回来
- 然后继续训练

适合：
- 前后显卡数目不一致
- world size 变了
- 只是想把训练接着跑

但代价也要接受：
- 这时不考虑 strict distributed optimizer 状态
- 恢复点附近更容易出现明显损失函数波动
- 它更像“继续训练”，不是“原样恢复训练现场”

### 3.4 `--resume-mode strict`

这是严格恢复模式。

它的精神是：
- 恢复时尽量按上一次 distributed 训练现场原样接回
- 不是只接权重，而是尽量接 DeepSpeed / ZeRO 的完整状态

具体行为是：
- 优先使用 checkpoint 目录里的 `distributed/`
- 尝试恢复 DeepSpeed / ZeRO 的完整分布式训练现场

适合：
- 前后显卡数、world size、策略都能保持一致
- 想验证训练连续性
- 想做真正意义上的断点续训

代价是：
- 对环境一致性要求高
- 显卡数不对、rank 对不上、显卡状态不足，都可能直接失败

---

## 4. 后来人怎么选这两个参数

可以直接按下面这套经验来选。

### 4.1 如果你只是正常做实验

建议：
- `--save-mode portable`
- `--resume-mode portable`

原因：
- 默认就够用
- 更省空间
- 更灵活
- 不会强迫你维持完全一样的分布式环境

### 4.2 如果你想做严格断点续训

建议：
- `--save-mode strict`
- `--resume-mode strict`

但必须满足：
- 前后显卡数目一致
- world size 一致
- 分布式策略一致
- `distributed/` 目录必须保留
- `resume-from` 路径必须准确

否则就应该预期：
- 直接报错
- 而不是偷偷从第 0 步开始训练

### 4.3 如果前后显卡数不一致怎么办

这时不要硬上 strict。

更合理的选择是：
- `--resume-mode portable`

这样即使前后显卡数目不一致，由于此时只加载参数本身，不严格考虑 distributed 优化状态，通常也还能继续训练。

但要清楚：
- 这种恢复更灵活
- 但损失函数在衔接处往往会波动更大

### 4.4 有一件事现在两种模式完全一样

那就是：
- 只要你显式传了 `--resume-from`
- 这个路径就必须正确

如果路径失效：
- 无论 `--resume-mode` 是什么
- 都会先告警
- 然后直接报错退出停止

不会再重新开始训练。

---

## 5. `distributed/` 为什么重要

对于 full SFT 多卡训练，strict resume 真正依赖的是：
- `distributed/`

外层那些平铺文件只能支持较轻量的恢复；
而 `distributed/` 里保存的是 DeepSpeed ZeRO 的真实分片状态。

也就是说：
- 没有 `distributed/`，仍然可以做 portable resume
- 但没有 `distributed/`，就不能做严格的 strict distributed resume

这也是 strict / portable 的本质区别。

---

## 6. 什么时候会报错停止

### 6.1 `--resume-from` 路径不存在

现在只要显式传了：
- `--resume-from ...`

但路径不存在：
- 不管是 `portable` 还是 `strict`
- 都会先给告警
- 然后直接报错停止

不会再默默从第 0 步开始训练。

### 6.2 strict 下常见失败条件

如果使用：
- `--resume-mode strict`

那么以下情况都可能直接失败：
- checkpoint 目录中没有 `distributed/`
- 当前 GPU 数量和保存时不一致
- world size 不一致
- `CUDA_VISIBLE_DEVICES` 顺序变化太大
- distributed strategy 不一致
- 显存状态不足，导致 DeepSpeed 初始化失败
- 某些 rank 无法正确恢复，进而引发 `Broken pipe` 或 `cudaErrorInvalidValue`

strict 的原则是：
- 成功就严格恢复
- 失败就明确失败
- 不偷偷回退到 portable
- 不偷偷从头训练

---

## 7. portable 和 strict 的实际区别

可以用一句话概括：

- `portable`：更像“只加载主要权重和普通训练状态，然后继续训练”
- `strict`：更像“把上一次多卡 DeepSpeed 训练现场原样接回来”

因此：
- 如果你只是想继续训练，并且环境已经变了，`portable` 更合适
- 如果你想验证训练曲线连续性，并且环境能保持一致，`strict` 更合适

---

## 8. strict resume 时建议先看 `resume_manifest.json`

如果 checkpoint 是用：
- `--save-mode strict`

保存出来的，那么 `distributed/` 下面会有：
- `resume_manifest.json`

这个文件现在会集中记录：

### 8.1 `checkpoint_metadata`
- checkpoint 路径
- distributed checkpoint 路径
- `epoch`
- `global_step`
- `global_rank`

### 8.2 `strict_resume_requirements`
- `save_mode`
- `resume_mode`
- `num_devices`
- `world_size`
- `strategy`
- `device`
- `precision`
- `cuda_visible_devices`
- `must_keep_distributed_directory`
- `how_to_resume_strict`

### 8.3 `training_config`
- `model_variant`
- `dataset_dir`
- `checkpoint_dir`
- `output_dir`
- `optimizer_type`
- `scheduler_type`
- `batch_size`
- `gradient_accumulation_steps`
- `learning_rate`
- `warmup_steps`
- `weight_decay`
- `gradient_checkpointing`
- `offload_encoder`

因此后续用户如果要做 strict resume，最好的办法就是：
- 先看 `resume_manifest.json`
- 再按里面的配置去复现设备数、策略和命令

---

## 9. 推荐命令示例

### 9.1 portable 保存

```bash
CUDA_VISIBLE_DEVICES=0,2 python train.py --yes fixed \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/full_sft_muse_portable \
  --checkpoint-dir ./checkpoints \
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
  --num-devices 2 \
  --strategy ddp \
  --save-every 5 \
  --save-mode portable
```

### 9.2 portable 继续训练

```bash
CUDA_VISIBLE_DEVICES=0,2 python train.py --yes fixed \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/full_sft_muse_portable \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --device cuda:0 \
  --full-sft \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --epochs 30 \
  --learning-rate 1e-5 \
  --warmup-steps 2000 \
  --weight-decay 0.01 \
  --optimizer-type adamw \
  --scheduler-type cosine \
  --gradient-checkpointing \
  --num-devices 2 \
  --strategy ddp \
  --save-every 5 \
  --save-mode portable \
  --resume-mode portable \
  --resume-from ./muse-sft-test/output/full_sft_muse_portable/checkpoints/epoch_20_loss_xxxx
```

### 9.3 strict 保存

```bash
CUDA_VISIBLE_DEVICES=0,2 python train.py --yes fixed \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/full_sft_muse_strict \
  --checkpoint-dir ./checkpoints \
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
  --num-devices 2 \
  --strategy ddp \
  --save-every 5 \
  --save-mode strict
```

### 9.4 strict 继续训练

```bash
CUDA_VISIBLE_DEVICES=0,2 python train.py --yes fixed \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/full_sft_muse_strict \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --device cuda:0 \
  --full-sft \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --epochs 30 \
  --learning-rate 1e-5 \
  --warmup-steps 2000 \
  --weight-decay 0.01 \
  --optimizer-type adamw \
  --scheduler-type cosine \
  --gradient-checkpointing \
  --num-devices 2 \
  --strategy ddp \
  --save-every 5 \
  --save-mode strict \
  --resume-mode strict \
  --resume-from ./muse-sft-test/output/full_sft_muse_strict/checkpoints/epoch_20_loss_xxxx
```

注意：
- strict 续训时，GPU 数量、world size、策略最好和保存时一致
- strict 续训前应先检查 `distributed/resume_manifest.json`

---

## 10. 当前建议

如果目标是：

### 10.1 日常实验、磁盘有限、环境可能变化
建议：
- `--save-mode portable`
- `--resume-mode portable`

### 10.2 严格验证断点续训连续性
建议：
- `--save-mode strict`
- `--resume-mode strict`
- 保留 `distributed/`
- 保持相同 GPU 数、相同 world size、相同策略

### 10.3 中断后只是想把训练接下去
如果环境和卡数已经变了：
- 优先用 `portable`
- 不要硬上 strict

---

## 11. 总结

当前 full SFT resume 已经分成两条明确路线：

- `portable`
  更灵活，更省空间，但恢复不一定严格连续

- `strict`
  更完整，更严格，但依赖 `distributed/` 和一致的分布式训练环境

如果要严格续训：
- 看 `resume_manifest.json`
- 对齐设备和分布式条件
- 保留 `distributed/`

如果只是继续训练：
- 用 `portable`
- 接受一定程度的 loss 波动