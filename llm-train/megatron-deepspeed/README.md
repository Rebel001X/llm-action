# Megatron-DeepSpeed 实战

> Megatron-DeepSpeed = 把 Megatron-LM 的「模型并行(TP/PP)」和 DeepSpeed 的「数据并行 + ZeRO 显存优化」缝合在一起，组成可训练百亿~千亿参数稠密大模型的 **3D 并行** 训练框架(BLOOM-176B 就是用它训出来的)。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/megatron-lm/README]] [[ai-framework/deepspeed/README]] [[llm-train/README]]

---

## 阅读地图

| 你想搞清楚的问题 | 跳到哪一节 |
| --- | --- |
| 为什么单靠 Megatron 或单靠 DeepSpeed 不够? | [§1 地基](#1-地基为什么需要把两者缝起来) |
| 两个流派的代码仓库(微软 vs bigscience)有啥区别? | [§2 两个分支](#2-两个分支微软-vs-bigscience) |
| TP / PP / DP 三个维度到底各切什么? | [§3 3D 并行三个维度](#3-3d-并行三个维度tppp--dpzero) |
| ZeRO 三个 stage 分别省什么显存? | [§4 ZeRO 在其中的角色](#4-zero-在其中的角色三个-stage) |
| 一次完整训练要走哪些步骤? | [§5 训练全流程](#5-端到端训练全流程) |
| 数据怎么预处理成二进制? | [§6 数据流水线](#6-数据流水线mmapindexed-dataset) |
| checkpoint 结构 / 改并行度后怎么转? | [§7 Checkpoint 与转换](#7-checkpoint-与转换) |
| 显存/通信量怎么估?给我数字 | [数值例子](#数值例子bloom-176b-级别的-3d-并行配置) |
| 和纯 Megatron-LM / 纯 DeepSpeed 怎么选? | [对照表](#对照表与同类对比) |
| 我踩坑了 | [常见问题](#常见问题典型坑) |

---

## 0. 一句话锚点

**一句话：** Megatron 负责「把一个大到放不下的模型沿权重矩阵切开(张量并行)、沿层切开(流水线并行)」，DeepSpeed 负责「在剩下的数据并行维度上，把优化器状态/梯度/参数再分片(ZeRO)，并管掉混合精度、梯度累积、通信重叠等工程脏活」。两者乘在一起 → **TP × PP × DP = 总 GPU 数**。

---

## 1. 地基：为什么需要把两者缝起来

### 1.1 单卡放不下：先算一笔显存账

训练时显存的大头不是激活值，而是 **模型状态(model states)**。对一个 $\Phi$ 参数的模型，用 Adam + fp16 混合精度训练：

| 项目 | 每参数字节 | 说明 |
| --- | --- | --- |
| fp16 参数 | 2 | 前向/反向用 |
| fp16 梯度 | 2 | 反向产出 |
| fp32 参数副本(master) | 4 | 优化器更新用 |
| fp32 Adam 动量 $m$ | 4 | |
| fp32 Adam 方差 $v$ | 4 | |
| **合计** | **16** | 即 $16\Phi$ 字节 |

所以 **1B(十亿)参数 ≈ 16 GB 模型状态**(还没算激活值)。

- 7B 模型 → $16 \times 7 = 112$ GB，单张 80GB A100 **直接 OOM**。
- 175B 模型 → $16 \times 175 = 2800$ GB，需要至少 35 张 80GB 卡 **只装模型状态**。

> 结论：百亿以上参数，**必须把模型本身切开**，单纯堆数据并行(每卡一份完整模型)无解。

### 1.2 Megatron 解决「模型切不开」，DeepSpeed 解决「副本太重」

```
            问题                         谁来解决
 ┌───────────────────────────┐   ┌──────────────────────────┐
 │ 一层 Linear 的权重矩阵     │   │ Megatron 张量并行 TP      │
 │ 太大，单卡放不下           │──▶│ 把矩阵按行/列切到多卡      │
 ├───────────────────────────┤   ├──────────────────────────┤
 │ 模型有几十上百层，         │   │ Megatron 流水线并行 PP    │
 │ 累加起来单卡放不下         │──▶│ 把层分段放到不同卡        │
 ├───────────────────────────┤   ├──────────────────────────┤
 │ 数据并行每卡一份完整       │   │ DeepSpeed ZeRO            │
 │ 优化器状态，冗余巨大       │──▶│ 把优化器状态/梯度/参数分片 │
 ├───────────────────────────┤   ├──────────────────────────┤
 │ fp16/梯度累积/通信重叠/    │   │ DeepSpeed Engine          │
 │ checkpoint 等工程脏活      │──▶│ 统一封装                  │
 └───────────────────────────┘   └──────────────────────────┘
```

**一句话分工**：Megatron 切「模型结构」(沿矩阵、沿层)，DeepSpeed 切「训练副本」(沿优化器状态)，正交叠加。

---

## 2. 两个分支：微软 vs bigscience

历史上有两个主要的 `Megatron-DeepSpeed` 仓库，本目录两份子 README 分别对应它们：

| 维度 | 微软版 [microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) | bigscience 版 [bigscience-workshop/Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed) |
| --- | --- | --- |
| 定位 | 微软官方维护，持续跟进新特性 | 为训练 BLOOM-176B 而 fork，偏「打过的仗」 |
| 代表模型 | 支持 GPT / LLaMA、LLaMA2 等 | BLOOM(176B 多语言) |
| 序列并行 | 较新版本支持 | 当年训 BLOOM 时**未支持序列并行** |
| 是否仍活跃 | 是，作为长期分支 | 训练完成后基本冻结，作为复现 BLOOM 的「考古」参考 |
| 子目录 | `./microsoft/` | 本节链接 |

```
 NVIDIA Megatron-LM (上游)
        │ fork + 接 DeepSpeed
        ├──────────────► microsoft/Megatron-DeepSpeed   ← 持续演进，支持 llama/llama2
        │
        └──────────────► bigscience/Megatron-DeepSpeed  ← 训 BLOOM-176B，已冻结
```

> 选型建议：**新项目用微软版**(特性新、维护活跃)；想**复现 BLOOM 训练细节**就读 bigscience 版及其 `bigscience` 训练日志仓库。
> 参考资料：HuggingFace 博客《[使用 DeepSpeed 和 Megatron 训练 BLOOM](https://huggingface.co/blog/zh/bloom-megatron-deepspeed)》。

---

## 3. 3D 并行：三个维度(TP/PP + DP/ZeRO)

3D = 三个**正交**的切分轴。GPU 总数 = $TP \times PP \times DP$。

```
        三个正交维度(以 32 卡, TP=4, PP=2, DP=4 为例)
        ───────────────────────────────────────────────

  DP(数据并行)：不同数据样本走不同副本，副本间 all-reduce 梯度
      副本0        副本1        副本2        副本3
        │            │            │            │
  ┌─────┴─────┐                                       每个"副本"内部
  │  PP(流水线)│  ← 把模型 L 层切成 2 段(stage0/stage1)  再被 PP×TP 切
  │  stage0   │
  │  ┌──┬──┬──┬──┐                                      stage 内部一层
  │  │g0│g1│g2│g3│  ← TP(张量并行)：一层的权重矩阵切到 4 卡
  │  └──┴──┴──┴──┘
  │  stage1   │
  │  ┌──┬──┬──┬──┐
  │  │g4│g5│g6│g7│
  │  └──┴──┴──┴──┘
  └───────────┘
```

### 3.1 TP 张量并行(切一层权重矩阵) —— 来自 Megatron

把一层内部的大矩阵乘法切到多卡。以 Transformer FFN 的两个 Linear 为例：第一个 Linear 权重 $A$ **按列切**($A=[A_1,A_2]$)，第二个 Linear 权重 $B$ **按行切**：

$$
Y = \text{GeLU}(XA)B,\quad A=[A_1, A_2],\ B=\begin{bmatrix}B_1\\B_2\end{bmatrix}
$$

每卡算 $\text{GeLU}(XA_i)B_i$，最后一次 **all-reduce** 求和。MHA 注意力同理(按注意力头切)。

```
  X ──┬──► [A1] ─► GeLU ─► [B1] ──┐
      │   (GPU0)                  ├─ all-reduce ─► Y
      └──► [A2] ─► GeLU ─► [B2] ──┘
          (GPU1)
```

- **特点**：通信极频繁(每层 2 次 all-reduce)，必须放在 **NVLink/NVSwitch 单机内**，TP 一般 $\le 8$(单机卡数)。

### 3.2 PP 流水线并行(切层) —— 来自 Megatron

把模型 $L$ 层切成几段(stage)放不同卡，一个 micro-batch 像流水线一样在 stage 间传递。朴素流水线有「气泡(bubble)」(部分卡空转)，用 **1F1B / interleaved** 调度把气泡压小。

```
 时间 ─────────────────────────────────►
 GPU0(stage0): F1 F2 F3 F4 ......  B4 B3 B2 B1
 GPU1(stage1):    F1 F2 F3 F4 ... B4 B3 B2 B1
                  └气泡┘                  └气泡┘
 把 batch 切成 m 个 micro-batch → 气泡占比 ≈ (PP-1)/(m+PP-1)
```

- **通信量小**(只在 stage 边界传激活)，可跨机；但 micro-batch 数要远大于 PP 段数才能把气泡压低。

### 3.3 DP 数据并行 + ZeRO —— 来自 DeepSpeed

剩下的卡做数据并行：每个**模型副本**吃不同 batch，反向后对梯度 all-reduce。DeepSpeed 在这一维度上叠 **ZeRO** 分片(见 §4)，把「每副本一份完整优化器状态」的冗余消掉。

> 关键点：**DP 的并行宽度 = 总卡数 ÷ (TP×PP)**，ZeRO 在这个 DP 组内分片。

---

## 4. ZeRO 在其中的角色(三个 stage)

ZeRO(Zero Redundancy Optimizer)逐级把「数据并行下冗余的模型状态」沿 DP 维度切开。设 DP 路数为 $N_d$：

| Stage | 分片什么 | 单卡模型状态(原 $16\Phi$) | 额外通信 |
| --- | --- | --- | --- |
| ZeRO-1 | 优化器状态(fp32 master+m+v) | $4\Phi + \tfrac{12\Phi}{N_d}$ | 几乎不变 |
| ZeRO-2 | + 梯度 | $2\Phi + \tfrac{14\Phi}{N_d}$ | 几乎不变 |
| ZeRO-3 | + 参数 | $\tfrac{16\Phi}{N_d}$ | 多一轮参数 all-gather |

```
 普通 DP(每卡一份):  [P][G][Os]  [P][G][Os]  [P][G][Os]  ← 完全冗余
 ZeRO-1         :  [P][G][Os0] [P][G][Os1] [P][G][Os2] ← Os 分片
 ZeRO-2         :  [P][G0][Os0][P][G1][Os1][P][G2][Os2]← G 也分片
 ZeRO-3         :  [P0][G0][Os0][P1][G1][Os1][P2][G2][Os2] ← P 也分片(用时 all-gather)
```

**和 Megatron 的配合惯例**：
- 当 TP/PP 已经把模型切得很小时，通常用 **ZeRO-1**(只分优化器状态，通信开销最低、最稳)。
- ZeRO-3 与 PP **不建议同时开**(两者都在切参数/传参数，语义冲突、易出错)。常见组合是 **TP + PP + ZeRO-1**，或 **TP + ZeRO-2/3(不开 PP)**。
- 还可叠加 **ZeRO-Offload**(把优化器状态/参数卸载到 CPU 内存)或 **ZeRO-Infinity**(卸载到 NVMe)换显存，代价是吞吐下降。

---

## 5. 端到端训练全流程

```
 ┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
 │ 1.原始语料  │──▶│ 2.分词+二进制 │──▶│ 3.配置3D并行 │──▶│ 4.启动训练   │
 │ jsonl/txt   │   │ preprocess   │   │ TP/PP/ZeRO   │   │ launcher+ds  │
 └─────────────┘   └──────────────┘   └─────────────┘   └──────┬───────┘
                                                                │
       ┌────────────────────────────────────────────────────────┘
       ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │ 5.周期保存   │──▶│ 6.断点续训   │──▶│ 7.转换并行度 │──▶│ 8.转 HF 格式 │
 │ checkpoint   │   │ --load 恢复  │   │ convert ckpt │   │ 推理/上传    │
 └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

1. **准备环境**：CUDA + PyTorch + apex(fused kernel) + DeepSpeed + Megatron-DeepSpeed 代码。多机要配好 NCCL、`pdsh`/SSH 互信、`hostfile`。
2. **数据预处理**(§6)：把 jsonl 语料转成 `*.bin` + `*.idx`。
3. **写并行配置**：通过命令行传 `--tensor-model-parallel-size`、`--pipeline-model-parallel-size`，并配 `--deepspeed` + 一个 `ds_config.json`(里面写 ZeRO stage、fp16/bf16、梯度累积等)。
4. **启动器**：单机用 `deepspeed`/`torchrun`，多机用 DeepSpeed 的 `--hostfile` 或 Slurm(本目录 `microsoft/slurm/` 有示例)。
5. **训练循环**：global batch = micro-batch × 梯度累积 × DP，框架自动跑混合精度 + 通信重叠。
6. **保存/续训**：按 `--save-interval` 周期写 checkpoint，`--load` 自动找最新续训。
7/8. **转换**：改并行度或导出 HF(§7)。

> 具体 CLI 标志/脚本名随版本变化，**以各分支官方脚本(`pretrain_gpt.py`、`examples/` 下的 `.sh`)为准**。

---

## 6. 数据流水线(mmap/indexed dataset)

Megatron 不直接读文本，而是预先把语料 token 化并存成 **二进制 + 索引** 的列式格式，训练时用 **内存映射(mmap)** 零拷贝随机访问。

```
 corpus.jsonl ──(preprocess_data.py)──►  my-gpt2_text_document.bin   (拼接后的 token id 流, 紧凑二进制)
   {"text":"..."}                        my-gpt2_text_document.idx   (每条文档在 bin 中的偏移/长度)
                                                │
                                   训练时 mmap，按 sample 切 seq_len 窗口
```

- `.bin`：所有文档分词后的 token id 顺序拼接，省去运行时分词开销。
- `.idx`：记录每篇文档的起止位置，让 dataloader 能 O(1) 定位 + 跨文档切固定 `seq_len` 的训练样本。
- 大语料常拆成多个 shard 并行预处理；混合多数据源用「权重采样」配置每个源的占比。

> 为什么这么设计：千亿训练要喂 TB 级 token，**运行时分词会成为瓶颈**；mmap 让多个 dataloader 进程共享同一份只读内存，不重复占内存。

---

## 7. Checkpoint 与转换

### 7.1 checkpoint 长什么样

3D 并行下，参数被 TP×PP 切碎，**每个 (TP rank, PP rank) 保存自己那一片**，外加 DeepSpeed 的优化器状态分片：

```
 checkpoints/iter_0001000/
   ├── mp_rank_00_000/      ← (tp=0, pp=0) 这一片的模型权重
   ├── mp_rank_01_000/      ← (tp=1, pp=0)
   ├── mp_rank_00_001/      ← (tp=0, pp=1)
   ├── ...
   ├── zero_pp_rank_*/      ← DeepSpeed ZeRO 优化器状态分片
   └── latest               ← 指向最新 iter 的指针
```

> 含义：**checkpoint 的目录结构和你训练时的 TP/PP/DP 强绑定**。这也是为什么换机器、换卡数前要先转换。

### 7.2 三种转换需求

| 需求 | 工具/方向 | 说明 |
| --- | --- | --- |
| 改并行度(如 TP4→TP2 续训) | `tools/convert_checkpoint`(微软分支) | 重新切分 mp 分片 |
| 导出给推理/HF | Megatron ckpt → HuggingFace | 见 [transformers/models/megatron_gpt2](https://github.com/huggingface/transformers/tree/main/src/transformers/models/megatron_gpt2) |
| ZeRO 分片合并 | `zero_to_fp32.py`(DeepSpeed 自带) | 把 ZeRO 分片合成单个 fp32 权重 |

- **转换原则**：先把 ZeRO 分片合并 → 再调整 TP/PP 切分 → 再(可选)转 HF 命名。
- 改并行度只能换「切法」，**不能凭空改模型结构**(隐藏维度/层数不变)。
- 微软分支转换工具：<https://github.com/microsoft/Megatron-DeepSpeed/tree/main/tools/convert_checkpoint>

---

## 数值例子：BLOOM-176B 级别的 3D 并行配置

**目标**：训 $\Phi = 176\text{B}$ 稠密模型，单卡 80GB A100。给一组**有代表性**的配置(数量级估算，非官方精确值)。

**Step 1 — 纯模型状态需要多少卡？**
$16\Phi = 16 \times 176 = 2816$ GB 模型状态。$2816 / 80 \approx 35$ 卡**只装模型状态**，加激活/碎片/通信缓冲，实际远不止。BLOOM 用了约 **384 张 A100**。

**Step 2 — 分配三个维度。** 一种典型切法：
$$
TP = 4,\quad PP = 12,\quad DP = 8 \ \Rightarrow\ 4 \times 12 \times 8 = 384 \text{ 卡}
$$
- TP=4：放在单机内(NVLink)，每卡只存 $1/4$ 的层内矩阵。
- PP=12：176B 的层切 12 段跨机。
- DP=8：8 路数据并行，ZeRO-1 把优化器状态再分到这 8 路。

**Step 3 — 估单卡模型状态。** 模型被 TP×PP = 48 份切碎，再叠 ZeRO-1(优化器状态 /8)：
$$
\underbrace{\frac{4\Phi}{48}}_{\text{fp16参数+梯度}} + \underbrace{\frac{12\Phi}{48 \times 8}}_{\text{ZeRO-1 优化器状态}} = \frac{4\times176}{48} + \frac{12\times176}{384} \approx 14.7 + 5.5 = 20.2 \text{ GB}
$$
留出约 60GB 给激活值、通信 buffer、碎片 —— 刚好落在 80GB 内。

**Step 4 — 流水线气泡。** 取 global batch = 2048 tokens-seq，micro-batch=1，梯度累积 $m$ 个 micro-batch。气泡占比 $\approx \frac{PP-1}{m+PP-1}=\frac{11}{m+11}$：
- $m=32$ → 气泡 $\approx 25\%$(偏高)
- $m=176$ → 气泡 $\approx 6\%$(好)

> **教训**：PP 越深，越要把 micro-batch 数堆大才能填满流水线，否则一堆卡空转。

**Step 5 — TP 通信量直觉。** TP 每层 2 次 all-reduce，每次约传 $b\cdot s\cdot h$ 个激活元素($b$=micro-batch, $s$=seq, $h$=hidden)。这是为什么 **TP 必须锁在 NVLink 域内**——跨机 TP 会被网络带宽拖死。

---

## 对照表：与同类对比

| 方案 | 模型切分(TP/PP) | 优化器分片(ZeRO) | 适用规模 | 何时选 | 主要权衡 |
| --- | --- | --- | --- | --- | --- |
| **纯 Megatron-LM** | ✅ TP+PP+SP | ❌(只有普通 DP) | 数十亿~千亿 | NVIDIA 全家桶、追极致吞吐 | 没有 ZeRO，DP 维度优化器冗余大 |
| **纯 DeepSpeed(ZeRO-3)** | ❌(不切单层) | ✅ ZeRO-1/2/3+Offload | 数十亿~百亿 | 想少改代码、靠 ZeRO 撑显存 | 单层超大矩阵切不开；超大模型 ZeRO-3 通信重 |
| **Megatron-DeepSpeed** | ✅ TP+PP | ✅ ZeRO-1/2(+Offload) | **百亿~千亿** | 单层放不下 **且** 要省优化器显存 | 配置复杂、ZeRO-3 与 PP 冲突 |
| **FSDP(PyTorch 原生)** | ❌(参数分片≈ZeRO-3) | ✅(等价 ZeRO-3) | 数十亿~百亿 | 想用原生、少依赖 | 缺成熟 TP/PP，超大模型不如 3D 并行 |

**选型口诀**：
- 单层矩阵单卡放得下 + 主要愁优化器显存 → **纯 DeepSpeed / FSDP**。
- 单层矩阵都放不下(百亿+) → 必须上 **TP**(Megatron) → 用 **Megatron-DeepSpeed**。
- 全 NVIDIA 卡 + 追吞吐极致 + 团队能 hold 住 → **纯 Megatron-LM** 也够。

---

## 常见问题(典型坑)

| 现象 / 坑 | 根因 | 处理 |
| --- | --- | --- |
| 启动直接 OOM | TP/PP 太小或 ZeRO stage 太低 | 加大 TP/PP 或提到 ZeRO-2；开 activation checkpointing |
| ZeRO-3 + PP 报错/结果错乱 | 两者都在切/传参数，语义冲突 | **不要同时开**；用 TP+PP+ZeRO-1 |
| 很多卡利用率低、吞吐差 | PP 深但 micro-batch 数太少 → 气泡大 | 增大梯度累积步数 $m$,让 $m \gg PP$ |
| TP 一开就慢/卡死 | TP 跨机走了以太网/IB | 把 **TP 限制在单机 NVLink 域内**($TP\le$ 单机卡数) |
| 换卡数后 checkpoint 加载失败 | mp 分片数与新 TP×PP 不匹配 | 先用 `convert_checkpoint` 转并行度再 `--load` |
| loss 变 NaN | fp16 动态范围溢出 | 改 **bf16**;或调 loss scale / 学习率 warmup |
| NCCL timeout / 多机起不来 | hostfile、SSH 互信、NCCL 环境变量没配好 | 检查 `pdsh`、`NCCL_SOCKET_IFNAME`、`NCCL_IB_*` |
| 全局 batch 对不上预期 | 没算清 micro×累积×DP 三者乘积 | $\text{global} = \text{micro}\times\text{accum}\times DP$ |
| 转 HF 后权重对不齐 | TP 切分维度与 HF 命名/拼接顺序不一致 | 用官方转换脚本,别手写拼接 |
| BLOOM 复现想用序列并行 | bigscience 分支当年**未支持序列并行** | 用微软分支或上游 Megatron-LM 的 SP |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局知识地图，先看这张
- [[ai-framework/megatron-lm/README]] — Megatron-LM 本体:TP/PP/SP 原理与 kernel
- [[ai-framework/deepspeed/README]] — DeepSpeed 本体:ZeRO/Offload/Engine
- [[llm-train/README]] — 大模型训练总览(本目录所在分区)
- 子目录:[微软版实战](./microsoft/README.md) · [Slurm 多机示例](./microsoft/slurm/README.md)
- 外部:[microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) · [bigscience/Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed) · [BLOOM 训练博客(中文)](https://huggingface.co/blog/zh/bloom-megatron-deepspeed)
