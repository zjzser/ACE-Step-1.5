# Muse SFT Test Log

## 1. 目标

这份文档记录 `muse-sft-test` 目录下的实验流程、训练命令、断点续训测试、失败原因分析，以及为解决 full SFT 多卡断点续训所做的代码改动。

重点包括：
- Muse JSONL 数据转为 ACE-Step 风格数据。
- 预处理为 `.pt` 张量。
- LoRA 单卡、多卡训练与断点续训。
- full SFT 多卡训练与断点续训。
- `portable` / `strict` 两套 checkpoint 保存与恢复模式。

---

## 2. 数据准备

### 2.1 Muse JSONL 转 ACE-Step 风格目录

```bash
python Muse-JSONL-2-ACE-Step.py \
  --jsonl /data1/zhoujunzuo/Task/2026April/Music-ACE-series/Music_data/submuse_en_2000/train_en.jsonl \
  --audio-root /data1/zhoujunzuo/Task/2026April/Music-ACE-series/Music_data/submuse_en_2000/ \
  --output /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/muse-sft-test/ACE-Step-style/ \
  --max-samples 8
```

### 2.2 预处理为 `.pt` 张量

预处理输出目录：`./muse-sft-test/preprocessed_tensors`

输出张量中包含：
- `target_latents`
- `context_latents`
- 文本条件相关张量
- `metadata`，其中保留原始 `audio_path`，方便后续回读

一个可用命令如下：

```bash
LD_LIBRARY_PATH="$NPP_LIB:$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
CUDA_VISIBLE_DEVICES=3 python train.py fixed \
  --preprocess \
  --audio-dir ./muse-sft-test/ACE-Step-style \
  --tensor-output ./muse-sft-test/preprocessed_tensors \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --max-duration 240 \
  --device cuda:0 \
  --dataset-dir ./__dummy_dataset_dir__ \
  --output-dir ./__dummy_output_dir__
```

说明：
- `train.py fixed --preprocess` 是当前仓库里真正可用的预处理入口。
- `--dataset-dir` 和 `--output-dir` 在这个模式下只是为了满足参数解析，实际预处理不会使用这两个路径。

---

## 3. LoRA 训练与断点续训

### 3.1 单卡 LoRA

```bash
CUDA_VISIBLE_DEVICES=3 python train.py --yes fixed \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/lora_8samples_single_test \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --device cuda:0 \
  --adapter-type lora \
  --rank 8 \
  --alpha 16 \
  --dropout 0.1 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --epochs 40 \
  --learning-rate 1e-4 \
  --warmup-steps 50 \
  --num-devices 1 \
  --strategy ddp \
  --offload-encoder
```

### 3.2 三卡 LoRA

```bash
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
  --epochs 40 \
  --learning-rate 1e-4 \
  --warmup-steps 50 \
  --num-devices 3 \
  --strategy ddp \
  --num-workers 0 \
  --offload-encoder \
  --save-every 10 \
  --val-split 0.1
```

### 3.3 LoRA 断点续训示例

```bash
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

### 3.4 LoRA 断点续训结论

LoRA 这条路径本身就会保存：
- adapter 权重
- optimizer state
- scheduler state
- `epoch` / `global_step`

因此它属于较完整的 resume 路径。

但实验上也观察到一个问题：
- 即使 resume 成功，loss 在恢复点附近仍可能出现明显波动。
- 尤其当恢复后又改变总训练长度，例如从 `40 epoch` 直接延长到 `120 epoch`，cosine scheduler 的整体轨迹会发生变化。

因此：
- 如果目的是严格验证 resume 连续性，恢复时应尽量保持原训练计划不变。
- 如果目的是“在已有权重基础上继续多训一段”，更合理的语义其实是第二阶段 finetune，而不只是 resume。

---

## 4. full SFT 训练与断点续训

### 4.1 一个可用的两卡 full SFT 命令

```bash
LD_LIBRARY_PATH="$NPP_LIB:$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
CUDA_VISIBLE_DEVICES=0,2 python train.py --yes fixed \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/full_sft_muse_3gpu \
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
  --val-split 0.1
```

### 4.2 早期 full SFT resume 的问题

一开始 full SFT 的多卡 resume 只能恢复：
- decoder 权重
- `epoch` / `global_step`

它不会完整恢复 DeepSpeed ZeRO 的 optimizer/scheduler 状态，因此会导致：
- 恢复点附近 loss 明显跳变
- 优化轨迹不连续
- 更像“从已有权重继续微调”，而不是严格意义上的断点续训

### 4.3 预加载显存峰值问题

一个关键经验是，full SFT 多卡启动前仍然会先发生一次单卡预加载峰值。

之前第一次 OOM 不是出现在真正的多卡训练阶段，而是出现在：
- `train_fixed.py`
- 调用 `model_loader.load_decoder_for_full_sft()` 后
- 在 `model.to(device)` 这一瞬间

也就是说，完整模型会先整体压到某一张卡上，再进入后续分布式初始化。

因此需要通过调整 `CUDA_VISIBLE_DEVICES` 和 `--device` 的映射，尽量把这次预加载落到显存状态更宽松的卡上。

### 4.4 一个最终跑通的 full SFT 例子

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --yes fixed \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/full_sft_muse_3gpu \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --device cuda:2 \
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
```

---

## 5. 为 full SFT resume 做的关键代码改动

### 5.1 DeepSpeed optimizer 兼容修复

在多卡 full SFT + DeepSpeed 路径下，原实现会因为 ZeRO optimizer CPU offload 与外部普通 AdamW 的组合而直接拒绝启动。

当时采用的是最小修复：
- 在 DeepSpeed 配置中加入 `zero_force_ds_cpu_optimizer: False`
- 保留现有 AdamW 路径
- 不去大范围改成 `DeepSpeedCPUAdam`

这样影响面更小，只触及：
- 多卡 full SFT + DeepSpeed

不会影响：
- LoRA
- LoKR
- 单卡训练
- 非 DeepSpeed 路径

### 5.2 DeepSpeed 梯度裁剪冲突修复

DeepSpeed 已经在 `ds_config` 中接管了梯度裁剪。

原代码仍沿用普通 Fabric 路径手动调用 `clip_gradients()`，导致两套裁剪逻辑冲突，训练直接失败。

修复策略是：
- 在多卡 full SFT + DeepSpeed 分支下，不再手动裁剪
- 其余训练路径保持原样

### 5.3 distributed checkpoint 支持

后来真正解决 full SFT 多卡 strict resume 的关键改动，是在 checkpoint 中加入：
- `distributed/`

这个目录内部保存的是 DeepSpeed ZeRO 分片后的完整训练现场，包括：
- 模型状态
- optimizer shard
- scheduler 相关状态
- DeepSpeed 自身元数据

这一步之后，full SFT 才真正具备严格断点续训能力。

---

## 6. strict / portable 两套 checkpoint 语义

### 6.1 新增参数

这里保留一段更接近原始实验笔记的说明：

> 新加了两个可选的命令来灵活调节：`--save-mode portable` 或者 `strict`，`--resume-mode portable` 或者 `strict`。
> 如果没选，默认都是 `portable` 选项。
> 区别在于：`--save-mode portable` 不保存 `distributed` 文件夹，节省空间；`--save-mode strict` 必须要有 `distributed` 文件夹。
> `--resume-mode portable` 则无视保留的 `distributed` 文件夹，土法吸收参数；`--resume-mode strict` 则是考虑从 `distributed` 加载 DeepSpeed 文件。
> 如果 `--resume-from` 路径失效的时候，无论 `--resume-mode` 是什么，都是自动报错退出停止，而不是重新开始训练。
> 如果 `--resume-mode` 是 `strict`，但是显卡数目不对，导致 rank 数对不起，或者显卡状态不足，这些都是常见情况，那么也会自动报错退出停止，而不是重新从第 0 步开始训练。

现在增加了两个可选参数：
- `--save-mode portable|strict`
- `--resume-mode portable|strict`

默认值都是：
- `portable`

### 6.2 `save-mode`

`--save-mode portable`
- 默认值
- 不保存 `distributed/`
- 更省空间
- 更灵活

`--save-mode strict`
- 会额外保存 `distributed/`
- 更适合同卡数、同拓扑的严格续训
- 占用磁盘明显更大

### 6.3 `resume-mode`

`--resume-mode portable`
- 忽略 `distributed/`
- 只使用平铺权重和训练状态文件
- 更适合换卡数、换环境继续训练
- 训练连续性不如 strict

`--resume-mode strict`
- 会使用 `distributed/`
- 适合同样的卡数、world size、策略、设备拓扑下严格恢复
- 连续性最好
- 对配置一致性要求更高

### 6.4 `resume-from` 缺失路径的当前语义

现在只要你显式传了 `--resume-from`，但路径不存在：
- 无论 `resume-mode` 是 `portable` 还是 `strict`
- 都会先给清楚的告警
- 然后直接报错停止

也就是说，不再允许“路径写错后静默从 0 开始训练”。

### 6.5 strict 失败的语义

如果 `--resume-mode strict` 时：
- 缺少 `distributed/`
- world size 不匹配
- GPU 数量改变
- `CUDA_VISIBLE_DEVICES` 映射和原来不一致
- DeepSpeed distributed load 失败

那么训练会直接报错停止，而不是偷偷回退到 portable 或从头训练。

---

## 7. strict distributed checkpoint 里的 `resume_manifest.json`

为了让后续 strict resume 的使用者“按图索骥”，现在在：
- `.../distributed/resume_manifest.json`

里会额外保存一份可读元信息。

这个文件当前按 3 组信息组织：

### 7.1 `checkpoint_metadata`

包括：
- checkpoint 路径
- distributed checkpoint 路径
- `epoch`
- `global_step`
- 是否 full SFT
- 当前 `global_rank`

### 7.2 `strict_resume_requirements`

这是最重要的一块，专门告诉后续 strict resume 需要满足什么条件，包括：
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

也就是说，后续用户只要打开这个文件，就能直接知道：
- 应该用 strict 还是 portable
- 需要几张卡
- 需要什么分布式策略
- 是否需要保持 `CUDA_VISIBLE_DEVICES` 顺序一致

### 7.3 `training_config`

这里是普通训练超参，方便复现实验：
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

---

## 8. strict full SFT resume 示例

这里也保留一段原始经验说明：

> 下面就是 `strict` 严格继续训练和严格保存的例子，这必须要前后用严格一致的显卡数目，否则就会对不上报错，并且要准确的恢复路径，否则也会报错。
> 如果是 `--resume-mode portable` 那就灵活很多了，即使前后显卡数目不一致，但由于此时只加载参数本身，不考虑优化状态，也依旧可以接下参数继续训练。
> 只是这时的优化器设置会使得相接之时出现大量的损失函数波动。
> 当然，`portable` 也是需要正确的续点恢复路径的，否则同样会报错。

下面是一个严格保存、严格恢复的例子：

```bash
CUDA_VISIBLE_DEVICES=0,2 python train.py \
  --yes fixed \
  --dataset-dir ./muse-sft-test/preprocessed_tensors \
  --output-dir ./muse-sft-test/output/full_sft_muse_3gpu_resume_3 \
  --checkpoint-dir ./checkpoints \
  --model-variant turbo \
  --device cuda:0 \
  --full-sft \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --learning-rate 1e-5 \
  --warmup-steps 2000 \
  --weight-decay 0.01 \
  --optimizer-type adamw \
  --scheduler-type cosine \
  --gradient-checkpointing \
  --num-devices 2 \
  --strategy ddp \
  --val-split 0.1 \
  --epochs 20 \
  --save-every 5 \
  --save-mode strict \
  --resume-from muse-sft-test/output/full_sft_muse_3gpu_resume_3/checkpoints/epoch_10_loss_1.6547 \
  --resume-mode strict
```

这条命令成立的前提是：
- `resume-from` 路径正确
- `distributed/` 存在
- 前后 GPU 数量一致
- world size 一致
- 分布式策略一致
- `CUDA_VISIBLE_DEVICES` 映射尽量保持一致

否则 strict resume 会直接报错。

---

## 9. portable full SFT resume 的意义

如果使用：
- `--save-mode portable`
- `--resume-mode portable`

那么训练不会依赖 `distributed/`，而是只恢复较轻量的平铺文件。

优点：
- 省空间
- 更灵活
- 换卡数时更容易继续训练

缺点：
- 不能保证严格连续恢复
- loss 在恢复点附近可能出现更明显波动
- 更像“在已有权重上继续训练”而不是原样恢复训练现场

---

## 10. TensorBoard

常用查看命令：

```bash
tensorboard --logdir muse-sft-test/output/lora_8samples_ddp3_test/runs --port 6006
```

```bash
tensorboard --logdir muse-sft-test/output/lora_8samples_single_test/runs --port 6006
```

```bash
tensorboard --logdir muse-sft-test/output/full_sft_muse_3gpu/runs --port 6006
```

```bash
tensorboard --logdir muse-sft-test/output/full_sft_muse_3gpu_resume/runs --port 6007
```

---

## 11. 本次实验结论

### 11.1 LoRA

- LoRA 的 resume 机制本来就比较完整。
- 能成功保存和恢复 adapter、optimizer、scheduler、epoch、step。
- 但如果恢复后重新定义了训练总长度，尤其搭配 cosine scheduler，loss 仍可能出现明显波动。

### 11.2 full SFT

- 最初只能恢复权重和进度，不能严格恢复 DeepSpeed ZeRO 优化状态。
- 后来通过引入 `distributed/`，full SFT 多卡 strict resume 才真正完成。
- `distributed/` 很大，但这是 strict distributed resume 的核心代价。

### 11.3 当前最佳理解

- 想省空间、想灵活迁移：用 `portable`
- 想严格恢复、验证连续性：用 `strict`
- 只要写了 `--resume-from`，路径就必须正确，否则直接报错停止

### 总之，设计者认为：

新加了两个（可选的）命令来灵活调节
--save-mode portable 或者 strict
--resume-mode portable 或者 strict

如果没选，默认都是 portable 选项。

区别在于：
--save-mode portable 不保存 distributed 文件夹，节省空间
--save-mode strict 必须要有 distributed 文件夹

--resume-mode portable 会无视保留的 distributed 文件夹，采用“土法吸收参数”的方式继续训练
--resume-mode strict 则会考虑从 distributed 加载 deepspeed 文件

如果 --resume-from 路径失效，无论 --resume-mode 是什么，都会自动报错退出停止，而不是重新开始训练。

如果 --resume-mode 是 strict，但是显卡数目不对（比之前多一张或少一张），导致 rank 对不上，或者显卡状态不足，这些常见情况也都会自动报错退出停止，而不是重新从第 0 步开始训练。

例如下面就是 strict 严格继续训练和严格保存的例子。
这要求前后使用严格一致的显卡数目，否则就会对不上并报错；同时恢复路径也必须准确，否则也会报错。

如果是 --resume-mode portable，就灵活很多。即使前后显卡数目不一致，由于此时只加载参数本身，不考虑优化器状态，也依旧可以接上参数继续训练。
只是这时优化器状态无法完整衔接，通常会导致续训时出现较大的 loss 波动。
当然，portable 也仍然需要正确的续点恢复路径，否则一样会报错。

'''
CUDA_VISIBLE_DEVICES=0,2 python train.py \
  --yes fixed  \
  --dataset-dir ./muse-sft-test/preprocessed_tensors   \
  --output-dir ./muse-sft-test/output/full_sft_muse_3gpu_resume_3   \
  --checkpoint-dir ./checkpoints   \
  --model-variant turbo   \
  --device cuda:0   \
  --full-sft   \
  --batch-size 1  \
  --gradient-accumulation 4   \
  --learning-rate 1e-5   \
  --warmup-steps 2000   \
  --weight-decay 0.01   \
  --optimizer-type adamw  \
  --scheduler-type cosine   \
  --gradient-checkpointing   \
  --num-devices 2   \
  --strategy ddp   \
  --val-split 0.1  \
  --epochs 20   \
  --save-every 5   \
  --save-mode strict \
  --resume-from muse-sft-test/output/full_sft_muse_3gpu_resume_3/checkpoints/epoch_10_loss_1.6547 \
  --resume-mode strict 

tensorboard --logdir muse-sft-test/output/full_sft_muse_3gpu_resume_3/runs --port 6006
'''