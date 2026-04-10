#  muse-sft-test/Muse-JSONL-2-ACE-Step.py 转换 muse 格式为 sce-step 格式
'''
python Muse-JSONL-2-ACE-Step.py \
        --jsonl /data1/zhoujunzuo/Task/2026April/Music-ACE-series/Music_data/submuse_en_2000/train_en.jsonl \
        --audio-root /data1/zhoujunzuo/Task/2026April/Music-ACE-series/Music_data/submuse_en_2000/ \
        --output /data1/zhoujunzuo/Task/2026April/Music-ACE-series/ACE-Step-1.5-sft/muse-sft-test/ACE-Step-style/ \
        --max-samples 8
'''

#  将正规的 sce-step 格式 预处理为.pt格式，里面包括各种 
#  - target_latents: (5285, 64)，context_latents: (5285, 128)
#  - metadata(里面有原来的 audio_path 路径，所以可以回读)
#  - 还有文本条件相关张量

'''

CUDA_VISIBLE_DEVICES=3 python -m acestep.training_v2.cli.train_fixed \
    --preprocess \
    --audio-dir ./muse-sft-test/ACE-Step-style \
    --tensor-output ./muse-sft-test/preprocessed_tensors \
    --checkpoint-dir ./checkpoints \
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
                            --checkpoint-dir ./checkpoints \
                            --model-variant turbo \
                            --max-duration 240 \
                            --device cuda:0 \
                            --dataset-dir ./__dummy_dataset_dir__ \
                            --output-dir ./__dummy_output_dir__ 
'''
#  1 卡 auto 的一种尝试方案, 大约 5000 MiB 对应文档的 2.1

'''
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
'''

# 多卡 3 卡的一种尝试方案, 每卡大约 6000 MiB的大进程 , 400 MiB 的小进程 对应文档的 2.2

'''
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
'''
# 断点续传lora的案例

'''
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

'''


# sft 的训练方式  
# LD_LIBRARY_PATH="$NPP_LIB:$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" 这句话是需要的，有时不加就会报错
# 解释一下 --val-split 0.1 可选参数，抽比例当验证集

'''
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
'''

# 断点继续训练 sft ，继续10个epoch，达到30个，
# 其中优化器状态涉及deepspeed，索性让代码没有保存(保存的话继续训练得接上完全一样)，因此二阶优化状态无法继承，在初期损失函数大小会有反扑

'''
CUDA_VISIBLE_DEVICES=1,3,4 python train.py --yes fixed \
    --dataset-dir ./muse-sft-test/preprocessed_tensors \
    --output-dir ./muse-sft-test/output/full_sft_muse_3gpu \
    --checkpoint-dir ./checkpoints \
    --model-variant turbo \
    --device cuda:2 \
    --full-sft \
    --batch-size 1 \
    --gradient-accumulation 4 \
    --epochs 120 \
    --warmup-steps 2000 \
    --weight-decay 0.01 \
    --optimizer-type adamw \
    --scheduler-type cosine \
    --gradient-checkpointing \
    --num-devices 3 \
    --strategy ddp \
    --save-every 10 \
    --val-split 0.1 \
    --resume-from ./muse-sft-test/output/full_sft_muse_3gpu/checkpoints/epoch_30_loss_1.3470

    或者

CUDA_VISIBLE_DEVICES=0,2 python train.py --yes fixed \
    --checkpoint-dir ./checkpoints             \
    --model-variant turbo      \
    --dataset-dir ./muse-sft-test/preprocessed_tensors     \
    --output-dir ./muse-sft-test/output/full_sft_muse_3gpu        \
    --full-sft     \
    --num-devices 2     \
    --strategy ddp     \
    --epochs 80     \
    --save-every 10 \
    --resume-from muse-sft-test/output/full_sft_muse_3gpu/checkpoints/epoch_30_loss_1.1466

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
    --checkpoint-dir ./checkpoints \
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


# tensorboard
'''
tensorboard --logdir ./output/lora_muse_full/runs --port 6006

tensorboard --logdir muse-sft-test/output/lora_8samples_ddp3_test/runs --port 6006

tensorboard --logdir muse-sft-test/output/full_sft_muse_3gpu/runs --port 6006

tensorboard --logdir muse-sft-test/output/lora_8samples_single_test/runs --port 6006

'''

'''
tensorboard --logdir muse-sft-test/output/full_sft_muse_3gpu_resume/runs --port 6007

经过了GPT的修改，我终于解决了断点续传的问题，除了muse-sft-test/output/full_sft_muse_3gpu_resume/checkpoints/epoch_10_loss_1.4058/distributed
这个路径里保存了几个巨大的文件，和显卡数量有关，显然是保存了deepspeed下每个对应子进程的全部内存

但是

LD_LIBRARY_PATH="$NPP_LIB:$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" CUDA_VISIBLE_DEVICES=0,2 python train.py --yes fixed   --dataset-dir ./muse-sft-test/preprocessed_tensors   --output-dir ./muse-sft-test/output/full_sft_muse_3gpu_resume   --checkpoint-dir ./checkpoints   --model-variant turbo   --device cuda:0   --full-sft   --batch-size 1   --gradient-accumulation 4   --epochs 30   --learning-rate 1e-5   --warmup-steps 2000   --weight-decay 0.01   --optimizer-type adamw   --scheduler-type cosine   --gradient-checkpointing   --num-devices 2   --strategy ddp   --save-every 10   --val-split 0.1

LD_LIBRARY_PATH="$NPP_LIB:$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" CUDA_VISIBLE_DEVICES=0,2 python train.py --yes fixed   --dataset-dir ./muse-sft-test/preprocessed_tensors   --output-dir ./muse-sft-test/output/full_sft_muse_3gpu_resume   --checkpoint-dir ./checkpoints   --model-variant turbo   --device cuda:0   --full-sft   --batch-size 1   --gradient-accumulation 4   --epochs 50   --learning-rate 1e-5   --warmup-steps 2000   --weight-decay 0.01   --optimizer-type adamw   --scheduler-type cosine   --gradient-checkpointing   --num-devices 2   --strategy ddp   --save-every 10   --val-split 0.1 --resume-from muse-sft-test/output/full_sft_muse_3gpu_resume/checkpoints/epoch_30_loss_1.1458

LD_LIBRARY_PATH="$NPP_LIB:$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" CUDA_VISIBLE_DEVICES=0,2 python train.py --yes fixed   --dataset-dir ./muse-sft-test/preprocessed_tensors   --output-dir ./muse-sft-test/output/full_sft_muse_3gpu_resume   --checkpoint-dir ./checkpoints   --model-variant turbo   --device cuda:0   --full-sft   --batch-size 1   --gradient-accumulation 4   --epochs 75   --learning-rate 1e-5   --warmup-steps 2000   --weight-decay 0.01   --optimizer-type adamw   --scheduler-type cosine   --gradient-checkpointing   --num-devices 2   --strategy ddp   --save-every 10   --val-split 0.1 --resume-from muse-sft-test/output/full_sft_muse_3gpu_resume/checkpoints/epoch_50_loss_0.9939

足够复刻出muse-sft-test/output/full_sft_muse_3gpu_resume/checkpoints/sft-retrain-distributed.png了

'''
# 新加了两个(可选的)命令来灵活调节
# --save-mode portable 或者 strict
# --resume-mode portable 或者 strict
# 如果没选，默认都是portable选项
# 区别在于 --save-mode portable 不保存 distributed 文件夹节省空间 --save-mode strict 必须要有distributed文件夹
# --resume-mode portable 则无视保留的 distributed 文件夹，土法吸收参数，--resume-mode strict 则是考虑从 distributed 加载deepspeed文件
# 如果 -resume-from 路径失效的时候，无论--resume-mode 是什么，都是自动报错退出停止，而不是重新开始训练
# 如果 --resume-mode 是 strict，但是 显卡数目不对(比之前多一张少一张)，导致rank数对不起，或者显卡状态不足，这些都是常见情况，那么也会自动报错退出停止，而不是重新从第0步开始训练

# 例如下面就是 strict 严格的继续训练和严格地保存的例子，这必须要前后用严格地显卡数目，否则就会对不上报错，并且要准确的恢复路径，否则报错
# 如果是 -resume-mode portable 那就随意很多了，即使前后显卡数目不一致，但由于此时只加载参数本身，不考虑优化状态，也依旧可以接下参数继续训练，
# 只是这时的优化器设置会使得相接之时出现大量的损失函数波动。当然，portable也是需要正确的续点恢复路径的，否则报错

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