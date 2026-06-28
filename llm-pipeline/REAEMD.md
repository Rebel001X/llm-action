# LLM 训练流水线（Pipeline）：从微调脚本读懂全链路

> 一句话定位：把"一条 `torchrun` 命令"拆开，看清数据→模型→优化器→分布式→显存→保存的完整训练流水线。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]

## 阅读地图

| 小节 | 你将搞懂 | 关键真料（来自原文脚本） |
|------|----------|--------------------------|
| 0 锚点 | 一条训练命令到底在跑什么 | `torchrun --nproc_per_node=N finetune.py ...` |
| 1 地基 | 全参 vs PEFT vs P-Tuning 的位置 | ChatGLM3 P-Tuning / BELLE 全参/LoRA 三段脚本 |
| 2 数据 | jsonl、multi-turn、cutoff_len | `--train_format multi-turn`、`cutoff_len=1024` |
| 3 批大小 | 等效 batch 的乘法公式 | `per_device × grad_accum × NUM_GPUS` |
| 4 优化器 | LR / warmup / cosine / weight_decay | `3e-4`、`8e-6`、`warmup_ratio 0.01` |
| 5 精度 | fp16 vs bf16 怎么选 | `--fp16`（ChatGLM3）vs `--bf16`（BELLE） |
| 6 分布式 | torchrun + DeepSpeed ZeRO | `--deepspeed configs/deepspeed_config_stage3.json` |
| 7 显存 | gradient_checkpointing / 8bit / ZeRO-3 | `--gradient_checkpointing`、`--use_int8_training` |
| 8 保存与续训 | save_steps、resume | `--save_strategy steps`、`--resume_from_checkpoint` |
| 实操 | 三份完整脚本原样保留 | 见「实操：完整脚本」 |
| 坑 | fp16 溢出 / OOM / 续训对不上 | 见「常见坑」 |

## 0. 一句话锚点

> **训练流水线 = 把"原始语料 + 基座权重"喂进"前向→反向→优化器更新"的循环，再用分布式与显存技巧让它在有限的 GPU 上跑得起来。** 原文给出的三份脚本，分别是这条流水线的三个真实切面：ChatGLM3 的 **P-Tuning v2**、BELLE 的 **全参微调**、BELLE 的 **LoRA 微调**。

一条命令的解剖：

```
torchrun --standalone --nnodes=1 --nproc_per_node=4   finetune.py  --train_file ...  --deepspeed configs/deepspeed.json
└─────────── 启动器 ───────────┘ └─ 多机 ─┘ └─ 每机几卡 ─┘ └训练脚本┘ └── 数据 ──┘ └──── 分布式后端 ────┘
```

- `torchrun`：PyTorch 官方分布式启动器，负责拉起 `nproc_per_node` 个进程、设好 `RANK/WORLD_SIZE/LOCAL_RANK` 环境变量。
- `finetune.py` / `sft_train.py`：真正的训练逻辑（HF `Trainer`/自定义循环）。
- `--deepspeed xxx.json`：把优化器/梯度/参数切分（ZeRO）的策略交给 DeepSpeed。

## 1. 地基：三种微调在流水线上的位置

```
                       基座权重 W (e.g. chatglm3-6b / llama-7b)
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   全参微调 FT            LoRA (PEFT)           P-Tuning v2
   更新整个 W          冻结 W，学低秩 ΔW       冻结 W，学每层前缀向量
   显存最大            显存小，存 A,B 矩阵      显存最小，存 prefix
   BELLE run_sft(FT)   BELLE run_sft(LoRA)     ChatGLM3 finetune_pt
```

- **全参微调（Full Fine-Tuning）**：所有参数都算梯度、都更新。效果上限高，但 7B 模型在 fp16 下需要约 `参数×(2权重+2梯度+12 Adam态)=16 字节/参 ≈ 112GB`，必须靠 ZeRO-3 切分到多卡。
- **LoRA**：冻结 $W$，只学低秩增量 $\Delta W = BA$（$B\in\mathbb{R}^{d\times r}, A\in\mathbb{R}^{r\times k}$，$r\ll d$）。前向变成 $h=Wx + BAx$。可训练参数常常只占 0.1%~1%，优化器状态随之骤减。
- **P-Tuning v2**：在每层注入可学习的前缀（prefix）向量，等价于给 KV 加一段"软提示"，基座完全冻结，是显存最省的一档。原文 ChatGLM3 脚本用的就是 `--train_format multi-turn` 的 P-Tuning 微调入口。

> 对照三档：显存 FT ≫ LoRA > P-Tuning；效果上限通常 FT ≥ LoRA ≥ P-Tuning（任务相关）。详见 [[llm-train/peft/PEFT-API]]。

## 2. 数据：jsonl、multi-turn 与 cutoff_len

原文真料：
- ChatGLM3：`DATASET_PATH=formatted_data/tool_alpaca.jsonl`，`--train_format multi-turn`，`--max_seq_length 2048`。
- BELLE：`train_file=belleMath.json` / `validation_file=belleMath-dev1K.json`，`cutoff_len=1024`，传入 `--model_max_length ${cutoff_len}`。

```
一条多轮样本 (multi-turn)            tokenizer + 拼模板
┌───────────────────────────┐       ┌──────────────────────────────┐
│ user:  解这道数学题 ...     │  ──▶  │ [gMASK]<sop> user ... assistant│
│ asst:  先设未知数 ...       │       │ ... </s> (截断到 cutoff_len)   │
│ user:  那如果改成 ...       │       └──────────────────────────────┘
│ asst:  ...                 │              │
└───────────────────────────┘              ▼  loss mask：只在 assistant 段算 loss
```

- **为什么要 `train_format multi-turn`**：多轮对话里一条样本含多次 assistant 回复，框架需要正确地把"用户问/系统提示"位置的 label 设为 `-100`（忽略），只对模型该学的回答段计算 loss。格式错会让模型学着复述用户输入。
- **为什么有 `cutoff_len` / `max_seq_length`**：注意力计算量随序列长度近似 $O(L^2)$，显存也随 $L$ 线性增长（KV、激活）。截断长度是吞吐与显存的总闸门。BELLE 取 1024、ChatGLM3 取 2048，是常见的"够用且省显存"折衷。

## 3. 批大小：三个数相乘才是"等效 batch"

这是脚本里最容易被忽略的乘法。等效（全局）batch：

$$B_{\text{eff}} = \text{per\_device\_train\_batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{NUM\_GPUS}$$

用原文数值手算：

| 脚本 | per_device | grad_accum | GPUs | 等效 batch |
|------|-----------:|-----------:|-----:|-----------:|
| ChatGLM3 P-Tuning | 16 | 1 | 4 | **64** |
| BELLE 全参（注释样例） | 2 | 4 | 8 | **64** |
| BELLE LoRA 8bit（注释） | 1 | 8 | 8 | **64** |
| BELLE LoRA 非 8bit（启用） | 1 | 1 | 8 | **8** |

```
梯度累积示意（per_device=1, grad_accum=8）
micro-batch:  1   2   3  ...  8        然后才 optimizer.step()
   loss/8 ─┐ loss/8 ─┐  ...  loss/8 ─┐
           └──── 梯度逐步累加 ────────┘──▶ 一次参数更新（等效 batch=8）
```

- **为什么要梯度累积**：显存放不下大 batch 时，用"多次小前向反向、攒够梯度再 step"换取大 batch 的统计稳定性。代价是更多前向/反向次数（更慢），但显存不变。
- **坑**：改了 GPU 数或 grad_accum，却没同步调 LR，等效 batch 变了学习动态就变了（见第 4 节）。

## 4. 优化器：LR / warmup / cosine / weight_decay

原文三套真实超参对照：

| 参数 | ChatGLM3 P-Tuning | BELLE 全参 | BELLE LoRA |
|------|-------------------|-----------|-----------|
| learning_rate | `LR=1e-4` | `8e-6` | `3e-4` |
| lr_scheduler | （脚本未显式，框架默认） | `cosine` | `cosine` |
| warmup_ratio | — | `0.05` | `0.01` |
| weight_decay | — | `0.00001` | `0.00001` |
| num_train_epochs / max_steps | `MAX_STEP=200` | `2` | `10` |

为什么 LR 差这么多？

```
更新的"目标"不同 ⇒ 合适 LR 不同
全参微调:   直接改原权重 → 学习率必须小(8e-6)，否则灾难性遗忘/震荡
LoRA:       只学小的 A,B  → 学习率可大(3e-4)，新参数从零起步需要更快收敛
P-Tuning:   学前缀向量    → 介于两者之间(1e-4)
```

warmup + cosine 退火的 LR 曲线：

```
LR
 │        ____
 │      /      \____
 │    /              \____
 │  /  warmup            \___ cosine 衰减到 ~0
 │/                            
 └───────────────────────────► step
   ↑warmup_ratio*total          ↑训练末期
```

- **warmup（线性升温）**：训练初期权重还很"乱"，直接用大 LR 容易让 loss 爆炸。先从 0 线性升到峰值，让 Adam 的二阶矩估计先稳下来。`warmup_ratio=0.05` 即前 5% 步用于升温。
- **cosine 退火**：峰值后按余弦曲线平滑降到接近 0，末期小步精修，利于收敛到更平坦的极小值。
- **weight_decay=1e-5**：L2 正则，抑制权重膨胀；微调里通常给很小值，避免把基座的好特征"正则掉"。
- **数值示例**：BELLE LoRA 跑 10 epoch、warmup_ratio=0.01。若一个 epoch 1000 步，则前 `10000×0.01=100` 步线性升温到 `3e-4`，其后余弦衰减。

## 5. 精度：fp16 还是 bf16

原文真料：ChatGLM3 用 `--fp16`；BELLE 全参/LoRA 都用 `--torch_dtype "bfloat16" --bf16`。

| | fp16 | bf16 |
|---|---|---|
| 指数位 | 5 | **8（同 fp32）** |
| 尾数位 | 10 | 7 |
| 动态范围 | 窄，易上溢/下溢 | 宽，几乎不溢出 |
| 精度 | 高一点 | 低一点 |
| 是否需 loss scaling | 需要 | 通常不需 |
| 硬件 | 老卡(V100)也支持 | Ampere+(A100/H800/4090)更佳 |

```
位布局对照
fp16: [S][EEEEE][MMMMMMMMMM]      指数5位 → 表示范围 ~6e-5 ~ 65504
bf16: [S][EEEEEEEE][MMMMMMM]      指数8位 → 范围同 fp32，溢出风险极低
```

- **为什么 BELLE 选 bf16**：大模型训练里激活/梯度的数值范围跨度大，fp16 易出 `NaN`（梯度上溢或下溢为 0）。bf16 牺牲尾数精度换动态范围，省去 loss scaling 调参，训练更稳，是 A100/H800 时代的默认。
- **为什么 ChatGLM3 给 fp16**：兼容更老的 GPU（如 V100 无 bf16），但需框架自带的动态 loss scaling 兜底。

## 6. 分布式：torchrun 启动器 + DeepSpeed ZeRO

原文每份脚本都走 `torchrun --nproc_per_node=N ... --deepspeed configs/xxx.json`。两者分工：

```
torchrun ─── 进程编排（拉起 N 个进程、分配 RANK/LOCAL_RANK、建 NCCL 通信组）
   │
   └─▶ 每个进程内的 DeepSpeed ─── 状态切分（ZeRO）：把优化器态/梯度/参数分摊到各卡
```

ZeRO 三级（BELLE 全参/LoRA 用的是 `deepspeed_config_stage3.json` → **ZeRO-3**）：

```
              单卡需保存                ZeRO-1   ZeRO-2   ZeRO-3
优化器状态(Adam m,v)  ✔ 切分              ✔        ✔        ✔
梯度 grad                                          ✔        ✔
模型参数 param                                              ✔
显存占用              全量 ──────────────────────────────▶ 最省
通信量               少 ─────────────────────────────────▶ 最多
```

- **为什么全参微调几乎必须 ZeRO-3**：7B 在 Adam+fp16 下单卡装不下（约 112GB，见第 1 节）。ZeRO-3 把参数也切到 8 卡，每卡只持有 1/8，前向/反向时按需 all-gather，用通信换显存。
- **`--ddp_timeout 36000`**：把分布式通信超时拉到 36000 秒（10 小时），防止某卡数据加载/编译慢导致 NCCL 误判超时而整组挂掉。这是大数据集/慢 IO 场景的实战防坑参数。
- 详解见 [[ai-framework/deepspeed/README]]；张量/流水并行另见 [[ai-framework/megatron-lm/README]]、底层框架 [[ai-framework/pytorch/README]]。

## 7. 显存：四把省显存的"真料开关"

把脚本里所有降显存手段排成一条优先级：

| 开关（原文出现处） | 机制 | 代价 |
|--------------------|------|------|
| `--gradient_checkpointing` | 前向只存部分激活，反向时重算 | 算力 +~30%，换激活显存大降 |
| `--deepspeed ..._stage3.json` | ZeRO-3 切分参数/梯度/优化器态 | 通信增加 |
| `--use_lora` | 冻结基座，只训低秩矩阵 | 效果上限略降 |
| `--use_int8_training`（LoRA 8bit） | 基座按 int8 量化加载（QLoRA 风格） | 推理/训练有量化误差 |

```
梯度检查点（gradient checkpointing）原理
普通:  前向把每层激活都存下来 → 反向直接用      （省时间，费显存）
检查点:前向只存少量"锚点"激活 → 反向重新前向算  （费时间，省显存）
        激活显存 O(L) ──▶ 近似 O(√L)
```

- **三档 LoRA 脚本对照**（原文同一文件三段）：
  - 全参 FT：`per_device=2, grad_accum=4`，无 LoRA，ZeRO 由 `deepspeed_config.json` 定。
  - LoRA + 8bit：`--use_lora --use_int8_training --lora_config configs/lora_config_llama.json`，`per_device=1, grad_accum=8`。显存最省，单卡可跑更大模型。
  - LoRA 非 8bit（脚本实际启用项）：`--use_lora --deepspeed ..._stage3.json`，`per_device=1, grad_accum=1`，`LR=3e-4`，`num_train_epochs=10`。
- 量化原理延伸：[[llm-compression/quantization/量化基础]] · [[llm-compression/README]]。

## 8. 保存与续训

原文真料：ChatGLM3 `SAVE_INTERVAL=50 → --save_steps 50`；BELLE `--save_strategy "steps" --save_total_limit 3`，并注释了 `--resume_from_checkpoint ...`。

```
step:  0 ──50──100──150──200
         │   │    │    │
       save save save save   ← --save_steps 50
         保留最近 3 个(--save_total_limit 3)，老的自动删
```

- **`save_total_limit 3`**：只留最近 3 个 checkpoint，防止磁盘被巨大权重塞满（每个 7B fp16 约 14GB）。
- **`resume_from_checkpoint`**：断点续训会同时恢复"权重 + 优化器状态 + scheduler 步数 + RNG"，所以续训前**不能改 LR/warmup/batch 等会改变训练动态的超参**（见坑表）。
- **`--seed 1234`**：固定随机种子，让数据 shuffle、dropout、初始化可复现。

## 实操：完整脚本（原文真料原样保留）

### A. ChatGLM3 多轮 P-Tuning v2
来源：`https://github.com/THUDM/ChatGLM3/blob/main/finetune_chatmodel_demo/scripts/finetune_pt_multiturn.sh`

```bash
#! /usr/bin/env bash
set -ex

LR=1e-4
NUM_GPUS=4
MAX_SEQ_LEN=2048
DEV_BATCH_SIZE=16
GRAD_ACCUMULARION_STEPS=1
MAX_STEP=200
SAVE_INTERVAL=50

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=tool_alpaca_ft
DATASET_PATH=formatted_data/tool_alpaca.jsonl

BASE_MODEL_PATH=THUDM/chatglm3-6b
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format multi-turn \
    --train_file $DATASET_PATH \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --fp16 \
    --deepspeed configs/deepspeed.json 2>&1 | tee ${OUTPUT_DIR}/train.log
```

### B. BELLE 全参 / LoRA 微调
来源：
- `https://github.com/baichuan-inc/Baichuan-7B/blob/main/scripts/train.sh`
- `https://github.com/LianjiaTech/BELLE/tree/main/train/scripts`
- `https://github.com/LianjiaTech/BELLE/blob/main/train/scripts/run_sft.sh`

```bash
#! /bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export WANDB_PROJECT=...
export WANDB_RUN_ID=...
export WANDB_RESUME=allow
export ABS_PATH=...
export PYTHONPATH="$ABS_PATH/BELLE/train"
model_name_or_path=/path_to_llm/hf_llama_7b/ # or bloomz-7b1-mt

train_file=belleMath.json
validation_file=belleMath-dev1K.json
output_dir="$ABS_PATH/BELLE/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=1024

#FT
# torchrun --nproc_per_node 8 src/entry_point/sft_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --deepspeed configs/deepspeed_config.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 2 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 8e-6 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --torch_dtype "bfloat16" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#    # --use_flash_attention
#    # --resume_from_checkpoint ...


#LoRA with 8bit
# torchrun --nproc_per_node 8 src/entry_point/sft_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --use_lora \
#     --use_int8_training \
#     --lora_config configs/lora_config_llama.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --num_train_epochs 2 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 8e-6 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --torch_dtype "bfloat16" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#    # --use_flash_attention
#    # --resume_from_checkpoint ...

# LoRA without 8bit
torchrun --nproc_per_node 8 src/entry_point/sft_train.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --use_lora \
    --deepspeed configs/deepspeed_config_stage3.json \
    --lora_config configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 3e-4 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
   # --use_flash_attention
   # --resume_from_checkpoint ...
```

### C. 其他可参考的微调框架
- Firefly：`https://github.com/yangjianxin1/Firefly`（统一多种国产模型 SFT/QLoRA 的轻量训练框架）。

> 注：脚本里 `--use_flash_attention` 被注释。开启可大幅省注意力显存、提吞吐，原理见 [[llm-optimizer/FlashAttention]]；推理侧的 KV 复用见 [[llm-optimizer/kv-cache]]。

## 常见问题 / 坑

| 现象 / 操作 | 原因 | 处理 |
|-------------|------|------|
| fp16 训练 loss 变 `NaN` | 梯度上溢/下溢，动态范围太窄 | 换 `--bf16`（如 BELLE）；或调大/动态 loss scale |
| 改了 GPU 数量但收敛变差 | 等效 batch = per_device×grad_accum×GPUs 跟着变，LR 没同步调 | 按线性/平方根法则同步缩放 LR |
| 续训后 loss 跳变 | `resume_from_checkpoint` 恢复了优化器/scheduler，但你改了 LR/warmup/batch | 续训保持训练动态超参不变 |
| 全参微调直接 OOM | 7B+Adam+fp16 单卡 ~112GB 装不下 | 用 `deepspeed_config_stage3.json`（ZeRO-3）切分；加 `--gradient_checkpointing` |
| `--use_int8_training` 后效果掉点 | int8 量化误差 + 仅训 LoRA | 任务允许时接受；或回退非 8bit / 全参 |
| 多轮数据模型学会复述用户 | 没设 `--train_format multi-turn`，label mask 错 | 用正确格式，仅对 assistant 段算 loss |
| NCCL 超时整组挂掉 | 某卡数据加载/编译慢，默认超时太短 | 加 `--ddp_timeout 36000` |
| 磁盘被 checkpoint 撑爆 | 每个 7B ckpt ~14GB，存太多 | `--save_total_limit 3` 只留最近几个 |
| `--seed` 没固定，结果不可复现 | shuffle/dropout/init 随机 | 显式 `--seed 1234` |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 训练框架：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 模型与算子：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]] · [[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 对齐：[[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 压缩/量化：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 推理：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 硬件/网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 评测与估算：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
