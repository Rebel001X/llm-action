# QLoRA：用 4-bit 量化把 65B 模型微调塞进单张 48GB 显卡

> 一句话定位：QLoRA = 4-bit NF4 量化冻结的基座权重 + 在其之上训练 LoRA 适配器，让"全参数级别"的微调质量在单卡上跑起来。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/peft/PEFT-API]] · [[ai-framework/huggingface-peft/README]] · [[llm-compression/quantization/量化基础]] · [[llm-train/peft/Prefix-Tuning]]

---

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|-------------|--------|
| 0 | 一句话锚点：QLoRA 到底省了什么 | 4-bit + LoRA |
| 1 | 地基：先回顾 LoRA、量化、显存账本 | 前置知识 |
| 2 | 三大创新：NF4 / 双重量化 / 分页优化器 | 论文核心 |
| 3 | 前向/反向数据流（ASCII 图）| 反量化即时计算 |
| 4 | 显存对照表 + 65B 手算 | 数值直觉 |
| 5 | 关键参数逐行拆解（对照原文命令）| lora_r/quant_type/bits... |
| 实操 | LLaMA-65B 单卡 / 多卡完整命令（原文真料）| qlora.py |
| 坑 | 常见问题与排错表 | 踩坑 |
| 🔗 | 跳转链接 | 双链 |

---

## 0. 一句话锚点

- **源码地址**：https://github.com/artidoro/qlora
- **commit id**：`cc488110b5ea23594a418daca7085000a9420625`

QLoRA 解决的核心矛盾：**全参数微调 65B 模型需要 780GB+ 显存（远超任何单卡），而 QLoRA 只需要约 48GB（单张 A6000 / A100-80G 绰绰有余）。**

它的做法可以浓缩成一行：

```
冻结的基座权重 → 压成 4-bit 存着（省显存）
                  ↑
            只训练旁边挂的一对小矩阵 A、B（LoRA）
```

关键洞察：**被压成 4-bit 的是"不训练的"基座权重；真正参与梯度更新的 LoRA 适配器仍是 16-bit 高精度。** 所以模型容量没丢，省下来的全是"冻结部分"的存储成本。

---

## 1. 地基 / 前置

要看懂 QLoRA，先把三块拼图摆好。

### 1.1 LoRA 回顾（旁路低秩矩阵）

LoRA 不去改原始权重 $W \in \mathbb{R}^{d\times k}$，而是给它并联一个低秩增量：

$$
h = Wx + \Delta W x = Wx + \frac{\alpha}{r} BA\,x
$$

- $A \in \mathbb{R}^{r\times k}$，$B \in \mathbb{R}^{d\times r}$，秩 $r \ll \min(d,k)$
- 训练时 **只更新 $A, B$**，$W$ 冻结
- $\alpha$ 是缩放因子，对应命令里的 `--lora_alpha`，缩放系数即 $\alpha/r$

直觉：原本要训 $d\times k$ 个参数，现在只训 $r\times(d+k)$ 个，参数量降几个数量级。详见 [[llm-train/peft/PEFT-API]]。

### 1.2 量化回顾（把 16-bit 压到 4-bit）

量化 = 用更少的比特表示数值。FP16 一个权重占 2 字节；INT4 / NF4 只占 0.5 字节，**直接省 4 倍存储**。代价是精度损失。详见 [[llm-compression/quantization/量化基础]]。

### 1.3 显存账本：训练显存花在哪？

训练一个 $P$ 参数的模型，显存四大块（以 FP16 + Adam 为例）：

| 项目 | 占用 | 65B 模型 |
|------|------|---------|
| 模型权重 (FP16) | $2P$ | 130 GB |
| 梯度 (FP16) | $2P$ | 130 GB |
| Adam 一阶动量 $m$ (FP32) | $4P$ | 260 GB |
| Adam 二阶动量 $v$ (FP32) | $4P$ | 260 GB |
| **合计** | $\sim 12P$ | **~780 GB** |

**关键观察**：优化器状态 $m,v$ 占了大头（8P）。LoRA 之所以省，是因为**梯度和优化器状态只对那一小撮 LoRA 参数计算**——780GB 里的绝大部分瞬间归零。QLoRA 进一步把剩下的"模型权重 130GB"也压成 4-bit → ~33GB。这就是 65B 单卡的来由。

---

## 2. QLoRA 的三大创新

普通 LoRA 之上，QLoRA 论文（Dettmers et al., 2023）叠了三个技术，缺一不可。

### 2.1 创新一：NF4（4-bit NormalFloat）

普通 INT4 把数值区间**均匀**切 16 份。但神经网络权重近似服从**零均值正态分布**——中间密、两头疏。均匀切分会浪费很多"格子"在几乎没有权重的尾部。

NF4 的思路：让 16 个量化点**按正态分布的分位数（quantile）来摆**，使每个格子里落进的权重数量大致相等（信息论上最优）。

```
普通 INT4（均匀格子）：     NF4（分位数格子）：
  权重分布(正态)              权重分布(正态)
        ▁▃▅█▅▃▁                    ▁▃▅█▅▃▁
  |  |  |  |  |  |        |   | | ||||| |   |
  格子均匀 → 中间挤,两头空    格子按密度 → 信息利用满
```

对应命令参数：`--quant_type nf4`（另一选项是 `fp4`，论文实测 nf4 更优）。

### 2.2 创新二：双重量化（Double Quantization）

量化时每一小块（block）权重要存一个 FP32 的**缩放常数（quantization constant / absmax）**。块小了常数就多，这些常数本身也吃显存。

双重量化 = **对这些量化常数再量化一次**。论文里：以 block size 64 量化权重得到的 FP32 常数，再以 block size 256 量化成 8-bit。

数值直觉：每个权重原本要分摊 $32/64 = 0.5$ bit 的常数开销；双重量化后降到约 $8/64 + 32/(64\times256) \approx 0.127$ bit。对 65B 模型相当于省下约 3GB。

对应命令参数：`--double_quant`。

### 2.3 创新三：分页优化器（Paged Optimizers）

长序列或梯度检查点重算时，会出现瞬时显存尖峰，导致 OOM。分页优化器借助 NVIDIA 统一内存（Unified Memory），在显存吃紧时把优化器状态**自动换页到 CPU 内存**，尖峰过去再换回——像操作系统的内存分页一样，避免训练直接崩掉。

> 注：原文命令未显式开 paged 优化器开关，这是 qlora.py 内部默认机制；理解原理即可，遇到偶发 OOM 时它是你的安全垫。

---

## 3. 前向 / 反向的数据流（核心机制图）

QLoRA 最容易误解的一点：**4-bit 权重并不直接参与矩阵乘法**。计算时会临时反量化（dequantize）回 16-bit，算完即丢，4-bit 形态只用于"存储"。

```
                      ┌──────────────────────────────────────┐
   输入 x (BF16) ───► │            QLoRA Linear 层            │
                      │                                      │
                      │   存储:  W_4bit (NF4, 冻结)          │
                      │            │ 计算时即时反量化          │
                      │            ▼                          │
                      │       W_16bit ──► W·x  ──┐            │
                      │                          ├──► h = Wx + (α/r)BAx ──► 输出
                      │   LoRA:  A,B (BF16,可训) │            │
                      │            └──► (α/r)BAx ┘            │
                      └──────────────────────────────────────┘
   反向传播:
     ✗ 不更新 W_4bit（冻结，无梯度）
     ✓ 只对 A、B 算梯度 → 优化器只维护 A、B 的 m、v
```

**为什么省显存又不掉质量**：
- 存储省：130GB 的 W 压成 ~33GB 的 W_4bit（省 ~97GB）
- 优化器省：梯度/动量只算 LoRA（几百 MB，而非 780GB 里的 650GB）
- 质量不掉：参与训练的 A、B 全程 BF16 高精度（命令里 `--bf16`），且反量化保证前向计算精度

---

## 4. 显存对照表 + 65B 手算

| 方案 | 65B 模型权重 | 梯度 | 优化器状态 | 总显存(粗略) | 单卡可行? |
|------|------------|------|-----------|------------|----------|
| 全参数 FP16 + Adam | 130 GB | 130 GB | 520 GB | ~780 GB | ❌ 需上百卡 |
| LoRA (基座 FP16) | 130 GB | ~0 | ~0 | ~140 GB | ❌ 单卡放不下 |
| **QLoRA (基座 NF4)** | **~33 GB** | ~0 | ~0 + LoRA | **~48 GB** | ✅ A6000/A100-80G |

**手算 65B 基座的 4-bit 存储**：
$$
65 \times 10^9 \text{ 参数} \times 0.5 \text{ Byte/参数} = 32.5 \text{ GB}
$$
加上 LoRA 参数、激活值、量化常数、临时反量化缓冲，论文报告 65B QLoRA 实测约 **48GB**——刚好一张卡。这就是 QLoRA 论文标题"可在单张 48GB GPU 上微调 65B"的算术来源。

---

## 5. 关键参数逐行拆解

下表对照原文命令里的真实参数，逐个讲清"是什么、为什么这么设"。

| 参数 | 取值(原文) | 含义 / 为什么 |
|------|-----------|--------------|
| `--bits 4` | 4 | 基座量化到 4-bit，QLoRA 的灵魂 |
| `--quant_type nf4` | nf4 | 用 NormalFloat4 而非均匀 INT4，见 §2.1 |
| `--double_quant` | 开启 | 双重量化，对量化常数再压一遍，见 §2.2 |
| `--lora_r 64` | 64 | LoRA 秩 $r$，越大容量越强、参数越多 |
| `--lora_alpha 16` | 16 | 缩放 $\alpha$，实际缩放系数 $\alpha/r = 16/64 = 0.25$ |
| `--lora_modules all` | all | 给**所有**线性层都加 LoRA（论文发现这样比只加 q,v 更接近全参微调）|
| `--lora_dropout 0.05` | 0.05 | LoRA 上的 dropout，防过拟合 |
| `--bf16` | 开启 | 计算精度用 BF16（反量化后、LoRA 参数都是 BF16）|
| `--gradient_checkpointing` | 开启 | 用重算换显存，配合分页优化器扛住显存尖峰 |
| `--gradient_accumulation_steps 16` | 16 | 梯度累积，等效 batch = 1×16，单卡也能要大 batch |
| `--per_device_train_batch_size 1` | 1 | 单卡每步只放 1 条样本（65B 太大）|
| `--source_max_len 16` / `--target_max_len 512` | 16 / 512 | 输入/输出截断长度，控激活显存 |
| `--learning_rate 0.0001` | 1e-4 | LoRA 学习率，比全参微调略大 |
| `--lr_scheduler_type constant` | constant | 恒定学习率（配 `--warmup_ratio 0.03` 预热）|
| `--max_grad_norm 0.3` | 0.3 | 梯度裁剪阈值，稳训练 |
| `--max_steps 200` | 200 | 演示用步数（真实微调通常更多）|
| `--do_mmlu_eval` | 开启 | 训练中跑 MMLU 基准评测 |
| `--group_by_length` | 开启 | 按长度分组样本，减少 padding 浪费 |

**单卡 vs 多卡的唯一区别**：单卡命令前面加了 `CUDA_VISIBLE_DEVICES=0` 限定只用 0 号卡；多卡命令去掉该限定，qlora.py / HuggingFace Trainer 会自动用所有可见 GPU 做数据并行。两份命令的**超参完全一致**。

---

## 实操：LLaMA 65B 微调（原文真料，完整保留）

### 单 GPU

```
CUDA_VISIBLE_DEVICES=0 python qlora.py \
    --model_name_or_path /data/nfs/guodong.li/pretrain/hf-llama-model/llama-65b \
    --dataset /data/nfs/guodong.li/data/alpaca_data_cleaned.json \
    --output_dir /home/guodong.li/output/llama-65b-qlora \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 100 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --eval_dataset_size 128 \
    --max_eval_samples 200 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 200 \
    --eval_steps 50 \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to tensorboard
```

### 多 GPU

```
python qlora.py \
    --model_name_or_path /data/nfs/guodong.li/pretrain/hf-llama-model/llama-65b \
    --dataset /data/nfs/guodong.li/data/alpaca_data_cleaned.json \
    --output_dir /home/guodong.li/output/llama-65b-qlora \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 100 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --eval_dataset_size 128 \
    --max_eval_samples 200 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 200 \
    --eval_steps 50 \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 0 \
    --report_to tensorboard
```

---

## 常见问题 / 坑

| 现象 / 问题 | 原因 | 排查 / 解决 |
|------------|------|------------|
| 训练偶发 OOM（尤其长序列）| 梯度检查点重算时显存尖峰 | 依赖分页优化器换页；或调小 `target_max_len`、`per_device_train_batch_size` |
| 4-bit 模型推理也很省，但变慢 | 计算时要即时反量化回 16-bit | 这是 QLoRA 的固有取舍：省显存、牺牲一点速度 |
| 误以为 4-bit 权重直接参与矩阵乘 | 概念混淆 | 4-bit 只用于**存储**；计算前反量化，见 §3 |
| `lora_modules` 只加 q,v 效果不如全参 | LoRA 覆盖面不够 | 用 `--lora_modules all` 给所有线性层加 LoRA（论文结论）|
| 量化常数本身吃显存 | block 划分多 → 常数多 | 开 `--double_quant`，对常数再量化，见 §2.2 |
| nf4 与 fp4 选哪个 | 都是 4-bit 但分布不同 | 论文实测 `nf4` 通常更优，原文也用 nf4 |
| 效果不及全参微调 | 秩太小 / 学习率不当 | 增大 `lora_r`（如 64），调 `learning_rate`，加足训练步数（原文 200 步仅演示）|
| 多卡没生效 | 还带着 `CUDA_VISIBLE_DEVICES=0` | 多卡时去掉该限定，让 Trainer 自动数据并行 |

---

## 🔗 跳转链接

**枢纽**
- [[00-知识地图]]
- [[llm-train/README]]
- [[llm-train/pytorch/distribution/README]]
- [[llm-train/megatron/README]]
- [[llm-train/megatron-deepspeed/README]]

**框架 / 库**
- [[ai-framework/megatron-lm/README]]
- [[ai-framework/deepspeed/README]]
- [[ai-framework/pytorch/README]]
- [[ai-framework/huggingface-peft/README]]

**PEFT 同族（强相关）**
- [[llm-train/peft/PEFT-API]]
- [[llm-train/peft/Prompt-Tuning]]
- [[llm-train/peft/Prefix-Tuning]]

**底层 / 上下游**
- [[ai-infra/网络/集合通信原语]]
- [[ai-infra/网络/NCCL]]
- [[llm-compression/quantization/量化基础]]
- [[llm-algo/transformer/模型架构]]
- [[llm-alignment/RLHF]]
- [[B07:llm-inference/大模型推理张量并行]]
