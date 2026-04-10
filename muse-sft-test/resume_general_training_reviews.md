# Resume 通用说明

## 1. 这份文档的范围

这份文档记录跨 `LoRA` 和 `full SFT` 都有参考价值的 resume 经验，重点是：
- `LoRA` 的恢复语义
- 为什么改 `--epochs` 会改变 scheduler 行为
- 如何从日志判断恢复是否真的生效
- 哪些旧结论已经过时，哪些仍然值得保留

如果只关心 full SFT，请优先看：
- [resume_sft_training_reviews.md](./resume_sft_training_reviews.md)

---

## 2. LoRA 的 `--resume-from` 语义

`LoRA` 路径当前实现的是相对完整的断点续训。

恢复时会尝试加载：
- LoRA adapter 权重
- `optimizer_state_dict`
- `scheduler_state_dict`
- `epoch`
- `global_step`

因此它比早期 full SFT 的“只恢复权重和进度”更接近真正的 resume。

但仍要注意：
- 它不会恢复 DataLoader / sampler / RNG 的完整状态
- 如果恢复后把训练总长度改大很多，曲线不一定严格连续

---

## 3. 为什么改 `--epochs` 会影响恢复表现

本仓库的 scheduler 是按下面的思路构造的：

```text
total_steps = steps_per_epoch * max_epochs
```

这意味着即使下面这些参数都不变：
- `learning_rate`
- `warmup_steps`
- `batch_size`
- `gradient_accumulation`
- 数据集
- GPU 数量
- `optimizer_type`
- `scheduler_type`

只要 `--epochs` 变了，训练计划本身就已经不是同一条学习率曲线。

对 `cosine` scheduler 来说，这个影响尤其明显。

因此：
- 如果目的是验证 resume 是否真的连续，恢复时应尽量保持原训练计划不变
- 如果目的是“在已有权重上延长训练很多”，更合理的语义往往是第二阶段训练，而不是原地 resume

---

## 4. 如何判断恢复是否真的成功

### 4.1 LoRA

如果 LoRA 恢复成功，日志里通常应出现类似：

```text
[OK] Resumed from epoch 40, step ..., optimizer OK, scheduler OK
```

这表示：
- adapter 权重已加载
- optimizer state 已加载
- scheduler state 已加载

### 4.2 full SFT

如果 full SFT 使用的是当前 `portable` / `strict` 机制，那么判断方式应结合模式来看：

- `portable`
  看是否成功加载平铺文件，并进入非零的 `epoch` / `step`
- `strict`
  看是否成功使用 `distributed/`，并且没有触发 strict fail-fast 退出

另外，strict checkpoint 现在会带：
- `distributed/resume_manifest.json`

后续使用者应先看这个文件再做恢复。

---

## 5. 旧文档里哪些内容仍然有参考价值

下面这些内容仍然值得保留：
- `LoRA` 的恢复语义说明
- 改 `--epochs` 会改变 scheduler 行为的解释
- 用日志判断 resume 是否成功的经验
- 一些历史 checkpoint 路径和命令示例

例如历史上用过的路径：
- `muse-sft-test/output/lora_8samples_ddp3_test/checkpoints/epoch_40_loss_1.0839`
- `muse-sft-test/output/full_sft_muse_3gpu/checkpoints/epoch_30_loss_1.3470`

这些路径仍然可以作为实验记录的一部分。

---

## 6. 旧文档里哪些内容已经过时

下面这些说法现在需要视为历史状态，而不是当前结论：
- full SFT 只能做“弱恢复”
- full SFT 不恢复 DeepSpeed distributed optimizer / scheduler 状态
- full SFT 当前缺少严格恢复能力

这些描述在最初阶段是正确的，但在引入下面两套模式后已经不再完整：
- `--save-mode portable|strict`
- `--resume-mode portable|strict`

现在的准确说法应该是：
- full SFT 仍然支持 `portable` 轻量恢复
- full SFT 也已经支持依赖 `distributed/` 的 `strict` 恢复

---

## 7. 当前建议

如果你只是想继续训练，并且环境可能变化：
- 优先用 `portable`

如果你想尽量恢复训练连续性，并且能保证环境一致：
- 用 `strict`
- 保留 `distributed/`
- 先检查 `resume_manifest.json`

