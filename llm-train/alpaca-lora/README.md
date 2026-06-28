# Alpaca-LoRA：用 LoRA 在多卡上微调 LLaMA 全家桶（7B/13B/30B/65B）

> 一句话定位：Alpaca-LoRA = **斯坦福 Alpaca 指令数据集 + LoRA 低秩适配 + LLaMA 基座**，把"全量微调一个 7B~65B 模型需要几百 GB 显存"的事，压到"单机 8 卡、每卡几十 GB、只训 0.15% 的参数"就能跑通。
> 📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/huggingface-peft/README]] · [[llm-train/peft/PEFT-API]] · [[llm-train/peft/Prefix-Tuning]] · [[llm-train/pytorch/distribution/README]]

---

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
| --- | --- | --- |
| 0. 一句话锚点 | 整件事的最小心智模型 | Alpaca + LoRA + LLaMA |
| 1. 地基/前置 | LLaMA 基座、指令数据、为什么要 PEFT | 全量微调 vs PEFT |
| 2. LoRA 原理 | 低秩分解到底省了什么 | $W+BA$、秩 r、$\alpha$ |
| 3. lora_target_modules | 为什么只挂在 q/k/v/o | 注意力投影矩阵 |
| 4. 数据并行(DP) 跑法 | torchrun 8 卡到底在干嘛 | DDP、AllReduce |
| 5. batch / micro_batch | 梯度累积怎么算 | 全局批 = 微批 × 卡数 × 累积 |
| 6. 各规模实操命令 | 7B/13B/30B/65B 原样命令 + 解读 | 真料区 |
| 7. 监控与测试 | tensorboard + 测试用例 | 收敛/评测 |
| 常见坑 | 别人踩过的雷 | OOM/拼写/数据格式 |
| 🔗 跳转 | 回枢纽 | 知识地图 |

- 源码：https://github.com/tloen/alpaca-lora
- commit id：`9de612e582ab86013b5d1c3be6b0ed9f5ab2065a`

---

## 0. 一句话锚点

> **冻结整个 LLaMA，只在注意力的几个投影矩阵旁边挂一对小矩阵 $B,A$ 来学习指令对话能力；8 张卡用数据并行各自算一份梯度再 AllReduce 求平均，于是一台机器就能微调 65B。**

把这句话拆成三个原子：

1. **冻结 + 旁挂小矩阵** → 这是 LoRA（省显存、省存储）。
2. **学指令对话能力** → 这是 Alpaca 数据（`alpaca_gpt4_data_zh.json` 等指令-回答对）。
3. **8 卡 AllReduce** → 这是数据并行（`torchrun --nproc_per_node=8`）。

---

## 1. 地基 / 前置

### 1.1 三个组成部分

```
          ┌──────────────────────────────────────────────┐
          │                Alpaca-LoRA                     │
          ├───────────────┬───────────────┬───────────────┤
          │   基座 (冻结)  │   适配 (可训)  │   数据 (指令)  │
          │   LLaMA 7~65B │   LoRA 旁路    │   Alpaca/GPT4 │
          │   ~32B 参数   │   ~5100万参数  │   46818 条    │
          └───────────────┴───────────────┴───────────────┘
                  ↑              ↑              ↑
            不更新梯度       只更新这部分     instruction→output
```

### 1.2 为什么不直接全量微调？

| 维度 | 全量微调 (Full FT) | LoRA |
| --- | --- | --- |
| 更新的参数量 | 100% | 约 0.15%（见下文实测） |
| 优化器状态显存 | 每个参数要存 Adam 的 m、v（×2）+ fp32 主权重 | 只为 LoRA 那一小撮参数存 |
| 产物大小 | 一份完整模型（几十 GB） | 一份 adapter（几十 MB） |
| 切换任务 | 各存一个完整模型 | 各存一个小 adapter，共享基座 |

**核心直觉**：Adam 优化器对每个**可训练**参数都要额外存动量 m 和方差 v。LoRA 把可训练参数砍到 0.15%，优化器状态显存几乎归零，省的不是前向激活，而是**优化器 + 梯度**这块大头。

> 30B 实测日志里这一行就是证据：
> ```
> trainable params: 51118080 || all params: 32580061696 || trainable%: 0.15689988704433913
> ```
> 即 5111 万 / 326 亿 ≈ **0.157%** 的参数参与训练。

---

## 2. LoRA 原理：低秩分解到底省了什么

### 2.1 数学形式

原始线性层做的是 $h = Wx$，其中 $W \in \mathbb{R}^{d\times d}$。LoRA 不动 $W$，而在旁边加一条低秩旁路：

$$h = Wx + \Delta W x = Wx + \frac{\alpha}{r}\,B A\,x$$

- $A \in \mathbb{R}^{r \times d}$，$B \in \mathbb{R}^{d \times r}$，秩 $r \ll d$。
- $A$ 用高斯初始化，$B$ 初始化为 0 → 训练开始时 $\Delta W = BA = 0$，模型行为与基座完全一致，**不破坏预训练能力**。
- $\alpha$ 是缩放系数，$\frac{\alpha}{r}$ 控制旁路对主干的影响强度。

### 2.2 参数量手算（为什么省）

设隐藏维 $d=4096$（7B 的注意力投影维度），秩 $r=16$（本仓库 `--lora_r=16`）：

```
全量一个投影矩阵 W :  d × d   = 4096 × 4096 = 16,777,216 参数
LoRA 旁路 B + A     :  d×r + r×d = 4096×16 + 16×4096 = 131,072 参数
压缩比             :  131072 / 16777216 ≈ 0.78%  → 省 99.2%
```

```
        x (d 维)
          │
   ┌──────┴───────┐
   ▼              ▼
┌──────┐      ┌────────┐  A: d→r  压扁
│  W   │      │   A    │  (冻结的 W 还在原位)
│冻结  │      └───┬────┘
│16.7M │          ▼
└──┬───┘      ┌────────┐  B: r→d  还原
   │          │   B    │
   │          └───┬────┘
   │      (×α/r)  │
   ▼              ▼
   └────►(+)◄─────┘
          │
          ▼  h = Wx + (α/r)BAx
```

### 2.3 `--lora_r=16` 该怎么选

| r | 表达能力 | 显存/产物 | 适用 |
| --- | --- | --- | --- |
| 4~8 | 较弱 | 最省 | 简单领域适配 |
| 16 | 平衡（本仓库选择） | 适中 | 通用指令微调 |
| 32~64 | 更强 | 更大 | 难任务、需更接近全量 |

r 越大越接近全量微调的拟合能力，但收益边际递减；本仓库统一用 **r=16** 是工程上的稳妥默认。

---

## 3. `--lora_target_modules='[q_proj,k_proj,v_proj,o_proj]'`：为什么只挂注意力

### 3.1 这四个是谁

Transformer 一层里，自注意力把输入投影成 Query/Key/Value，再把输出投影回去：

```
        x
        │
  ┌─────┼─────┬─────┐
  ▼     ▼     ▼     │
q_proj k_proj v_proj│   ← LoRA 挂这里
  │     │     │     │
  └──Attention──┐   │
                ▼   │
             o_proj │   ← 也挂这里
                │   │
                ▼   │
              (+残差)◄┘
```

- `q_proj / k_proj / v_proj`：生成 Q、K、V 的三个投影矩阵。
- `o_proj`：注意力输出再投影回隐藏维。
- **没挂 MLP（gate/up/down）**：这是经典 LoRA 论文的做法，注意力投影对"学习新行为/风格"最敏感，性价比最高。想更接近全量效果时，可以把 MLP 也加进 target_modules（但本仓库未这么做，遵循原文不擅自更改）。

### 3.2 对照：挂哪些模块

| target_modules | 可训练参数 | 效果 | 本仓库 |
| --- | --- | --- | --- |
| `[q_proj,v_proj]` | 最少 | LoRA 论文最小配置 | 否 |
| `[q_proj,k_proj,v_proj,o_proj]` | 中等 | 注意力全投影 | **是** ✅ |
| 上面 + MLP | 最多 | 最接近全量 | 否 |

---

## 4. 数据并行（DP）：`torchrun --nproc_per_node=8` 在干嘛

### 4.1 一句话

8 张卡每张都放**一份完整模型**（LoRA 模式下放得下），各自吃不同的一批数据算梯度，再用 **AllReduce** 把 8 份梯度求平均，保证 8 张卡参数始终一致。这就是 PyTorch DDP（DistributedDataParallel）。

```
   data shard 0   shard 1   ...   shard 7
        │           │              │
     ┌──▼──┐     ┌──▼──┐        ┌──▼──┐
     │GPU0 │     │GPU1 │  ...   │GPU7 │   每卡一份完整模型副本
     │ ∇0  │     │ ∇1  │        │ ∇7  │   各算各的梯度
     └──┬──┘     └──┬──┘        └──┬──┘
        └───────────┴──── AllReduce ───┘
                      ▼
            平均梯度 ḡ = (∇0+...+∇7)/8
                      ▼
            每卡用同一个 ḡ 更新 → 参数永远同步
```

> AllReduce 的原语细节见 [[ai-infra/网络/集合通信原语]]、底层库见 [[ai-infra/网络/NCCL]]。
> DP 与张量并行(TP)/流水并行(PP)的区别见 [[B07:llm-inference/大模型推理张量并行]]。

### 4.2 关键参数

| 参数 | 含义 | 本仓库取值 |
| --- | --- | --- |
| `--nproc_per_node=8` | 单机起 8 个进程（8 卡） | 8 |
| `--master_port=29005` | 进程组通信端口 | 29005 |
| `finetune_metrics_epoch.py` | 带逐 epoch 指标记录的训练脚本 | — |
| `finetune.py` | 原始训练脚本 | 30B/65B 部分命令用 |

---

## 5. batch_size / micro_batch_size：梯度累积怎么算

### 5.1 三个量的关系

```
全局批大小 batch_size
   = micro_batch_size × 卡数(8) × 梯度累积步数(gradient_accumulation_steps)

           ┌─────────── 全局批 batch_size ───────────┐
GPU0: [micro][micro]...   ← 累积 N 次才更新一次
GPU1: [micro][micro]...
 ...   (8 卡并行)
```

脚本内部一般这样推：`gradient_accumulation_steps = batch_size // micro_batch_size // world_size`。

### 5.2 用真料手算

以 **7B** 命令为例（`--batch_size 80 --micro_batch_size 10`，8 卡）：

```
梯度累积步数 = 80 / 10 / 8 = 1   → 不累积，每个 micro-batch 就更新
每张卡每步处理 = 10 条
全局每步处理   = 10 × 8 = 80 条 = batch_size ✅
```

以 **30B** 命令为例（`--batch_size 16 --micro_batch_size 2`，8 卡）：

```
梯度累积步数 = 16 / 2 / 8 = 1
每卡 micro = 2（30B 太大，micro 必须调小防 OOM）
```

> **规律**：模型越大，`micro_batch_size` 越要往下压（7B=10 → 13B=6 → 30B=2 → 65B=1），靠 `batch_size` 保持有效批量大小。这是显存约束下的标准操作。

### 5.3 其它通用参数

| 参数 | 作用 | 为什么 |
| --- | --- | --- |
| `--num_epochs 10`（zh 数据） | 训 10 轮 | 中文指令数据量小，多轮更充分 |
| `--num_epochs 3`（cleaned/65B） | 训 3 轮 | 数据更大或更省算力时少轮 |
| `--cutoff_len=512` | 截断到 512 token | 控制序列长度→控显存，超出部分丢弃 |
| `--group_by_length` | 按长度分组成 batch | 同批长度相近 → 少 padding → 更快更省 |

> ⚠️ `--group_by_length` 是吞吐优化：把长度接近的样本放一个 batch，减少 pad token 的浪费。代价是打乱了纯随机采样，一般影响可忽略。

---

## 6. 实操命令（原文真料，原样保留）

> 以下命令、路径、参数均来自原仓库实测，未做改动。环境为单机 8 卡（A100/A800 级，每卡约 74~76G 显存占用）。

### 6.1 7B

```bash
torchrun --nproc_per_node=8 --master_port=29005 finetune_metrics_epoch.py \
--base_model '/data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b' \
--data_path '/home/guodong.li/llama-mp/GPT-4-LLM/data/alpaca_gpt4_data_zh.json' \
--output_dir '/home/guodong.li/output/alpaca-lora-7b-dp-zh' \
--batch_size 80 \
--micro_batch_size 10 \
--num_epochs 10 \
--cutoff_len=512 \
--group_by_length \
--lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
--lora_r=16
```

| 模型 | 显存 | 耗时 | 数据量 |
| --- | --- | --- | --- |
| 7B | 8 * 74G | 2 小时 5 分钟 | 46818 |

![image](https://github.com/liguodongiot/llm-action/assets/13220186/238d86da-bbda-4944-94e4-49a87284e026)

### 6.2 13B

```bash
torchrun --nproc_per_node=8 --master_port=29005 finetune_metrics_epoch.py \
--base_model '/data/nfs/guodong.li/pretrain/hf-llama-model/llama-13b' \
--data_path '/home/guodong.li/llama-mp/GPT-4-LLM/data/alpaca_gpt4_data_zh.json' \
--output_dir '/home/guodong.li/output/alpaca-lora-13b-dp-zh' \
--batch_size 48 \
--micro_batch_size 6 \
--num_epochs 10 \
--cutoff_len=512 \
--group_by_length \
--lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
--lora_r=16
```

| 模型 | 显存 | 耗时 | 数据量 |
| --- | --- | --- | --- |
| 13B | 8 * 76G | 2 小时 10 分钟 | 46818 |

![image](https://github.com/liguodongiot/llm-action/assets/13220186/a66ea4a1-79fb-40d9-8a10-7a9132fde882)

### 6.3 30B

```bash
torchrun --nproc_per_node=8 --master_port=29005 finetune_metrics_epoch.py \
--base_model '/data/nfs/guodong.li/pretrain/hf-llama-model/llama-30b' \
--data_path '/home/guodong.li/llama-mp/GPT-4-LLM/data/alpaca_gpt4_data_zh.json' \
--output_dir '/home/guodong.li/output/alpaca-lora-30b-dp-zh-1' \
--batch_size 16 \
--micro_batch_size 2 \
--num_epochs 10 \
--cutoff_len=512 \
--group_by_length \
--lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
--lora_r=16
```

训练过程（真实日志）：

```
trainable params: 51118080 || all params: 32580061696 || trainable%: 0.15689988704433913

{'train_runtime': 55949.6417, 'train_samples_per_second': 8.368, 'train_steps_per_second': 0.523, 'train_loss': 0.4879480355503537, 'epoch': 10.0}

100%|████████████████████████████████████████████████████████████████████████████████████████████| 29270/29270 [15:32:27<00:00,  1.91s/it]
```

**日志读法**：

| 字段 | 值 | 含义 |
| --- | --- | --- |
| `trainable params` | 51,118,080 | LoRA 实际训练的参数（5111 万） |
| `all params` | 32,580,061,696 | 模型总参数（326 亿，即 30B） |
| `trainable%` | 0.1569% | 只训了千分之一点五的参数 |
| `train_runtime` | 55949.6 s | ≈ 15.5 小时，对得上下表 |
| `train_loss` | 0.488 | 10 轮后的训练损失 |
| `29270` steps | — | 总步数 = 数据量/有效批 × epochs |

| 模型 | 显存 | 耗时 | 数据量 |
| --- | --- | --- | --- |
| 30B | 8 * 75G | 15 小时 30 分钟 | 46818 |

![image](https://github.com/liguodongiot/llm-action/assets/13220186/303b850c-3332-45aa-968d-bb0f52fa44a6)

**另一组 30B 命令**（换用 `finetune.py` + 英文 cleaned 数据 + 更大批 + 3 轮）：

```bash
torchrun --nproc_per_node=8 --master_port=29005 finetune.py \
--base_model '/data/nfs/guodong.li/pretrain/hf-llama-model/llama-30b' \
--data_path '/data/nfs/guodong.li/data/alpaca_data_cleaned.json' \
--output_dir '/home/guodong.li/output/alpaca-lora-30b-dp' \
--batch_size 96 \
--micro_batch_size 6 \
--num_epochs 3
```

> 对比：这组用 `batch_size 96 / micro 6`（更大批），`num_epochs 3`（更少轮），且换成英文清洗数据，说明**批大小/轮数要随数据集和目标调整**，不是固定的。

### 6.4 65B

```bash
torchrun --nproc_per_node=8 --master_port=29005 finetune.py \
--base_model '/data/nfs/guodong.li/pretrain/hf-llama-model/llama-65b' \
--data_path '/home/guodong.li/llama-mp/GPT-4-LLM/data/alpaca_gpt4_data_zh.json' \
--output_dir '/home/guodong.li/output/alpaca-lora-65b-dp-zh' \
--batch_size 8 \
--micro_batch_size 1 \
--num_epochs 3
```

> 65B 的 `micro_batch_size 1`（极限压到 1）+ `batch_size 8`（梯度累积 = 8/1/8 = 1）是显存约束下的下限配置。模型越大，单卡能塞的 micro 越少。

### 6.5 规模 → 配置一览（横向对照）

| 模型 | 脚本 | batch_size | micro_batch | epochs | 数据 | 显存/卡 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7B | finetune_metrics_epoch | 80 | 10 | 10 | gpt4_zh | 74G | 2h05 |
| 13B | finetune_metrics_epoch | 48 | 6 | 10 | gpt4_zh | 76G | 2h10 |
| 30B | finetune_metrics_epoch | 16 | 2 | 10 | gpt4_zh | 75G | 15h30 |
| 30B | finetune | 96 | 6 | 3 | cleaned(en) | — | — |
| 65B | finetune | 8 | 1 | 3 | gpt4_zh | — | — |

**趋势看图**：`micro_batch_size` 单调下降 `10→6→2→1`，正是"模型越大、单卡显存越紧、单步能放的样本越少"的直接体现。

---

## 7. 监控与测试

### 7.1 测试用例（验证微调后是否会"听指令"）

```
请给我讲一个温馨的睡前故事
如何快速提升自己的写作能力？
计算以下表达式：(6+2)*(2-2)。
What are the five characteristics of a good argument?
```

> 这四条覆盖了：创作（讲故事）、建议（写作能力）、推理计算（表达式 = 0）、英文论证——用来快速肉眼检查模型的**指令遵循、推理、中英双语**能力。

### 7.2 tensorboard 看训练曲线

```bash
source /home/guodong.li/virtual-venv/alpara-lora-venv-py310-cu117/bin/activate

tensorboard --logdir /home/guodong.li/output/alpaca-lora-7b-dp-zh --port=16007 --host=0.0.0.0
tensorboard --logdir /home/guodong.li/output/alpaca-lora-13b-dp-zh --port=16008 --host=0.0.0.0
tensorboard --logdir /home/guodong.li/output/alpaca-lora-30b-dp-zh-1 --port=16009 --host=0.0.0.0
```

- 虚拟环境名 `alpara-lora-venv-py310-cu117` 透露环境：**Python 3.10 + CUDA 11.7**。
- `--host=0.0.0.0` 让远程机器也能访问；每个模型用不同端口（16007/16008/16009）便于同时对比。
- 主要看 **train_loss 是否单调下降并收敛**（30B 最终降到 0.488）。

---

## 常见问题 / 坑

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| CUDA OOM | `micro_batch_size` 相对模型太大 | 调小 micro（30B→2，65B→1），同时降 `cutoff_len` |
| 全局批不匹配预期 | 忘了乘卡数 | 牢记 `batch = micro × 卡数 × 累积步` |
| `lora_target_modules` 不生效 | 写法/引号问题 | 必须 `'[q_proj,k_proj,v_proj,o_proj]'` 整串带引号，模块名要和模型权重命名一致 |
| LoRA 训完效果约等于没训 | $B$ 没正确初始化为 0 / 学习率太小 / r 太小 | 检查初始化、适当增大 r 或 lr |
| 多机端口冲突 | 多任务用了同一 `master_port` | 每个任务换不同端口（如 29005/29006...） |
| 数据加载报错 | json 不是 `[{instruction, input, output}]` 格式 | 对齐 Alpaca 数据 schema |
| `group_by_length` 后 loss 抖动 | 同长度成批改变了采样分布 | 通常可接受；要纯随机就去掉该 flag |
| tensorboard 打不开 | 没加 `--host=0.0.0.0` 或端口被占 | 加 host、换端口 |
| 中文数据训太久 | epochs=10 + 序列长 | 数据小可保持多轮；算力紧可降到 3 轮 |

---

## 🔗 跳转链接

**回枢纽**：[[00-知识地图]]

**训练框架与并行**：
- [[llm-train/README]] · [[llm-train/pytorch/distribution/README]]（DDP/数据并行）
- [[llm-train/megatron/README]] · [[llm-train/megatron-deepspeed/README]]
- [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]]

**PEFT / LoRA 家族**：
- [[ai-framework/huggingface-peft/README]] · [[llm-train/peft/PEFT-API]]
- [[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]]

**底层通信**：
- [[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/NCCL]]

**上下游**：
- [[llm-algo/transformer/模型架构]]（注意力 q/k/v/o 投影从哪来）
- [[llm-alignment/RLHF]]（指令微调之后的对齐步骤）
- [[llm-compression/quantization/量化基础]]（配合 QLoRA 进一步省显存）
- [[B07:llm-inference/大模型推理张量并行]]（DP vs TP vs PP）
