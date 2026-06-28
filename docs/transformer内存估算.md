# Transformer 训练/推理内存估算

> 把"一个 Transformer 模型到底吃掉多少显存"从字节级讲清：参数、梯度、优化器状态、激活值、KV Cache 逐项手算。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/transformer/模型架构]] · [[llm-algo/FLOPs]] · [[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]

参考：transformer-math (EleutherAI) / transformer-inference-arithmetic (kipply)。

## 阅读地图

| 节 | 内容 | 你将能手算 |
|----|------|-----------|
| 0 | 一句话锚点 | 显存四大块 |
| 1 | 地基：数据类型与字节 | fp32/fp16/bf16/fp8 占几字节 |
| 2 | 参数量公式 | 给定 $h,L,V$ 算出 $P$ |
| 3 | 训练显存：参数+梯度+优化器 | Adam 为什么要 16 bytes/参数 |
| 4 | 训练显存：激活值 | 为什么激活值常是大头 |
| 5 | 重计算/梯度检查点 | 激活值如何从 $O(L)$ 降到 $O(\sqrt{L})$ |
| 6 | 并行如何分摊显存 | TP/PP/ZeRO 各切哪一块 |
| 7 | 推理显存：参数+KV Cache | KV Cache 随 batch×seq 线性涨 |
| 8 | 数值手算汇总 | 7B/13B/70B 全表 |
| — | 常见问题 + 跳转 | |

## 0. 一句话锚点

显存占用 = **静态部分**（随模型大小固定）+ **动态部分**（随 batch、序列长度变化）。

- 训练静态四大块：**参数 P、梯度 G、优化器状态 O、激活值 A**。前三者合称"模型状态"。
- 推理两大块：**参数 P、KV Cache**。没有梯度、优化器、反向激活。

```
                  ┌─────────── 训练显存 ───────────┐
                  │ 模型状态(静态)        激活(动态) │
   总显存 ≈  ┌────┴────┬────────┬──────────┬────────┐
            │ 参数 P  │ 梯度 G │ 优化器 O │ 激活 A │ + 碎片/通信缓冲
            └─────────┴────────┴──────────┴────────┘
                  ┌─────────── 推理显存 ───────────┐
   总显存 ≈  ┌────┴────┬───────────────────────────┐
            │ 参数 P  │ KV Cache (随 batch×seq 涨) │ + 临时激活
            └─────────┴───────────────────────────┘
```

## 1. 地基：数据类型与字节数

一切显存最终都是"**元素个数 × 每元素字节数**"。先把单位钉死。

| 类型 | 字节/元素 | 用途 |
|------|----------|------|
| fp32 (单精度) | 4 | 优化器主副本、累加 |
| fp16 / bf16 (半精度) | 2 | 混合精度的参数/梯度/激活 |
| fp8 (E4M3/E5M2) | 1 | 前沿训练/推理，需缩放因子 |
| int8 | 1 | 推理量化 |
| int4 | 0.5 | 极致推理量化 |

换算锚点（牢记）：

$$1\ \text{GiB} = 2^{30}\ \text{bytes} = 1024^3 \approx 1.074\times10^9\ \text{bytes}$$

工程上常用 $1\ \text{GB}=10^9$ 字节做粗估，本文除非标注，"GB"按 $10^9$ 估、对照真实卡（如 80GB H100）时用 GiB。

```
   1 个 fp16 参数 = 2 bytes
   1 个 fp32 参数 = 4 bytes
   10 亿(1e9) 个 fp16 参数 = 2e9 bytes ≈ 2 GB   ← "1B 参数 ≈ 2GB(fp16)" 的来历
```

> 记忆口诀：**fp16 下，每 1B(十亿)参数 ≈ 2 GB**。fp32 翻倍到 4 GB，fp8/int8 减半到 1 GB。

## 2. 参数量公式（从结构推 P）

Transformer 一层(decoder block) 的参数主要在两处：**Attention** 和 **MLP/FFN**。设隐藏维 $h$，层数 $L$，词表 $V$，FFN 中间维 $d_{ff}$（多数模型 $d_{ff}=4h$）。

**Attention 部分**（Q/K/V/O 四个投影，每个 $h\times h$）：

$$P_{attn} = 4h^2 \quad(\text{每层})$$

**MLP 部分**（两个线性层 $h\to d_{ff}\to h$）：

$$P_{mlp} = 2 \cdot h \cdot d_{ff} = 2 \cdot h \cdot 4h = 8h^2 \quad(\text{每层, 当 } d_{ff}=4h)$$

每层合计（忽略 LayerNorm 的 $O(h)$ 小项）：

$$P_{layer} \approx 4h^2 + 8h^2 = 12h^2$$

加上 **嵌入层**（输入 embedding 与输出 lm_head，常各 $Vh$，可能共享）：

$$\boxed{P \approx 12 L h^2 + V h \;(\text{+ tie 时只算一次})}$$

```
  一个 Transformer Block 的参数分布 (h=4096)
  ┌────────────────────────────────────────────┐
  │ Attention  4h² = 4·4096²  ≈  67.1 M         │ ███████ 33%
  │ MLP        8h² = 8·4096²  ≈ 134.2 M         │ ██████████████ 67%
  └────────────────────────────────────────────┘
   每层 ≈ 201.3 M ；MLP 约占 2/3，是参数大头
```

**手算示例（GPT-3 13B 量级）**：$h=5120,\ L=40,\ V=50257$。

- 主干：$12 L h^2 = 12 \times 40 \times 5120^2 = 12 \times 40 \times 2.621\times10^7 \approx 1.258\times10^{10}$
- 嵌入：$Vh = 50257 \times 5120 \approx 2.57\times10^8$
- 合计 $\approx 1.28\times10^{10} \approx 12.8\text{B}$ ✓ 与官方 13B 吻合。

## 3. 训练显存（一）：模型状态 = 参数 + 梯度 + 优化器

这是**混合精度 + Adam** 训练的经典账本（ZeRO 论文的核心起点）。设参数量 $P$。

| 项 | 精度 | 字节/参数 | 说明 |
|----|------|----------|------|
| 参数 (fp16 副本) | fp16 | 2 | 前向/反向用 |
| 梯度 (fp16) | fp16 | 2 | 反向产出 |
| 优化器：fp32 参数主副本 | fp32 | 4 | master weights |
| 优化器：Adam 一阶动量 $m$ | fp32 | 4 | momentum |
| 优化器：Adam 二阶动量 $v$ | fp32 | 4 | variance |
| **合计** | | **16** | 即 $16P$ 字节 |

$$\boxed{M_{state} = (2+2+12)\,P = 16P \ \text{bytes}}$$

```
   每个参数在混合精度+Adam下的 16 字节
   ┌────┬────┬──────────┬──────────┬──────────┐
   │fp16│fp16│ fp32 主  │ Adam m   │ Adam v   │
   │参数│梯度│ 副本     │ (动量)   │ (方差)   │
   │ 2B │ 2B │   4B     │   4B     │   4B     │  = 16 bytes
   └────┴────┴──────────┴──────────┴──────────┘
        ▲ 优化器状态共 12B，占 3/4，是"看不见的"大头
```

**为什么要 fp32 主副本？** fp16 的最小可表示间隔（machine epsilon）较大，`权重 += 很小的更新量` 会因舍入丢失（下溢）。保留一份 fp32 master weights 累加更新，再 cast 回 fp16 用于计算，兼顾精度与速度。

**手算（7B 模型，纯模型状态）**：$P=7\times10^9$。
$$16 \times 7\times10^9 = 1.12\times10^{11}\ \text{bytes} \approx 112\ \text{GB}$$
单卡 80GB 装不下 → 必须并行/ZeRO（见第 6 节）。

> 若用 8-bit Adam（动量存 int8），优化器从 8B 降到约 2B，合计从 16P 降到约 6P，省一半多。

## 4. 训练显存（二）：激活值（常被低估的大头）

**激活值**是前向过程中需要保存、供反向计算梯度的中间张量。它**不随参数缩放，而随 batch、序列长度、层数线性增长**，常在长序列/大 batch 时超过模型状态。

EleutherAI 给出的单层激活近似（无重计算、含 attention 中间量），设 batch $b$、序列 $s$、隐藏维 $h$、注意力头数 $a$、半精度 2 字节：

$$A_{layer} \approx s\,b\,h\,(34 + 5\cdot \tfrac{a\,s}{h})\ \text{bytes}$$

总激活 $A \approx L \cdot A_{layer}$。其中 $34$ 来自各线性/LayerNorm/GeLU 的中间结果，$5\cdot\frac{as}{h}$ 项来自注意力得分矩阵 $QK^\top$（形状 $b\times a\times s\times s$，**随 $s^2$ 增长**，这是长序列爆显存的根因）。

```
  激活值随序列长度 s 的增长（示意）
  显存
   │                                   ╱  ← O(s²) 注意力得分主导
   │                              ╱
   │                        ╱
   │                  ╱
   │           ╱            ← O(s) 线性部分主导(短序列)
   │     ╱
   └──────────────────────────────────→ 序列长度 s
         FlashAttention 把 O(s²) 的 score 矩阵从显存中消除！
```

**FlashAttention 的关键作用**：它不在 HBM 中物化整个 $s\times s$ 得分矩阵，而是分块在片上 SRAM 计算，把注意力激活从 $O(s^2)$ 降到 $O(s)$。所以现代训练里 $5\cdot\frac{as}{h}$ 这一项几乎消失，激活主要由那 $34$ 项决定。详见 [[llm-optimizer/FlashAttention]]。

**手算（h=4096, L=32, a=32, b=1, s=2048, 无重计算）**：
- 线性项：$34 sbh = 34 \times 2048 \times 1 \times 4096 \approx 2.85\times10^8$
- 注意力项：$5\frac{as}{h}sbh = 5\times\frac{32\times2048}{4096}\times2048\times1\times4096 \approx 5\times16\times2048\times4096 \approx 6.71\times10^8$
- 单层 $\approx 9.56\times10^8$ bytes ≈ 0.96 GB，×32 层 ≈ **30.6 GB**（仅 batch=1！batch=8 就 ~245 GB，必须重计算或 Flash）。

## 5. 重计算 / 梯度检查点（Gradient Checkpointing）

核心权衡：**用算力换显存**。反向需要激活，但我们可以**只保存每层的输入**，反向时再重新前向一遍该层恢复中间激活。

- 不重计算：存全部激活，显存 $O(L)$，反向无额外计算。
- 全重计算：每层只存边界，显存 $\approx \sqrt{L}$ 级（选择性分段），代价多一次前向 ≈ **+33% 计算量**（$\frac{1\text{fwd}+1\text{recompute fwd}+2\text{bwd}}{1\text{fwd}+2\text{bwd}}\approx\frac{4}{3}$）。

```
  普通：存 a1 a2 a3 ... aL   (显存正比 L)
   ┌──┐┌──┐┌──┐      ┌──┐
   │a1││a2││a3│ ...  │aL│   全部驻留
   └──┘└──┘└──┘      └──┘

  检查点：只存少数"锚点"，反向时重算其间
   ┌──┐            ┌──┐
   │a1│  ........  │a√L│    反向到这段时重跑前向恢复
   └──┘            └──┘     显存 ~O(√L)，多花 ~33% FLOPs
```

EleutherAI 的近似：用激活检查点后，激活约可降到 $\approx 2\,s\,b\,h\,L$ 量级（只存层间张量）。这把第 4 节那个 30.6 GB 的例子压到几 GB 级。

## 6. 并行与 ZeRO：哪一块被切分

单卡装不下时，靠并行把"模型状态/激活"分摊到多卡。

| 技术 | 切分对象 | 通信代价 |
|------|---------|---------|
| 数据并行 DP | 不切（每卡全副本） | 梯度 AllReduce |
| ZeRO-1 | 切优化器状态 O | 同 DP + 略增 |
| ZeRO-2 | 切 O + 梯度 G | AllReduce→Reduce-Scatter+AllGather |
| ZeRO-3 | 切 O + G + 参数 P | 前向/反向额外 AllGather 参数 |
| 张量并行 TP | 切层内权重(行/列) | 每层 AllReduce(激活) |
| 流水并行 PP | 按层切到不同卡 | 仅边界激活 P2P |

```
   ZeRO 三级对 16P 的切分 (N 卡)
   原始(DP)  ┌ P(2) ┬ G(2) ┬ O(12) ┐  每卡都 16P
   ZeRO-1    ┌ P(2) ┬ G(2) ┬ O/N   ┐  每卡 4P + 12P/N
   ZeRO-2    ┌ P(2) ┬ G/N  ┬ O/N   ┐  每卡 2P + 14P/N
   ZeRO-3    ┌ P/N  ┬ G/N  ┬ O/N   ┐  每卡 16P/N  ← 近线性下降
```

**手算（7B，ZeRO-3，N=8 卡）**：$16P/N = 112\text{GB}/8 = 14\text{GB}$/卡，单张 24GB 卡可行。详见 [[ai-framework/deepspeed/README]]。TP 切激活与权重，见 [[ai-framework/megatron-lm/README]] 与 [[llm-inference/大模型推理张量并行]]；通信原语见 [[ai-infra/网络/集合通信原语]]。

## 7. 推理显存：参数 + KV Cache

推理没有梯度/优化器/反向激活，显存简单很多，但多了一个随上下文增长的 **KV Cache**。

**(a) 参数**：$M_{param} = P \times \text{字节}$。fp16 即 $2P$；int8 即 $1P$；int4 即 $0.5P$。

**(b) KV Cache**：自回归解码时，每个已生成 token 的每层 K、V 都要缓存以避免重算。每 token 每层缓存 K 和 V 各一份，每份 $h$ 个元素（更精确地与注意力实现/GQA 有关）：

$$\boxed{M_{kv} = 2 \times b \times s \times L \times h \times \text{bytes}}$$

其中 $2$ 是 K 和 V 两份。若用 **GQA/MQA**（KV 头数 $a_{kv} <$ 查询头 $a$），则 $h$ 换成 $h\cdot\frac{a_{kv}}{a}$，KV Cache 大幅缩小（这是 Llama-2-70B 用 GQA 的主因）。

```
   KV Cache 随生成长度线性膨胀
   每解码一个新 token：每层追加 1 个 K 向量 + 1 个 V 向量
   layer L ┆ K K K K K K ...→  (长度 = 已生成 token 数 s)
           ┆ V V V V V V ...→
     ...   ┆
   layer 1 ┆ K K K K K K ...→
           ┆ V V V V V V ...→
            └ 显存 = 2·b·s·L·h·bytes，与 s 成正比
```

**手算（Llama-13B 推理，fp16，b=1, s=2048）**：$h=5120, L=40$。
$$M_{kv}=2\times1\times2048\times40\times5120\times2\ \text{B} = 1.68\times10^9\ \text{B}\approx 1.68\ \text{GB}$$
参数 $2\times13\text{B}=26$ GB。合计 ~27.7 GB（单 token 路径）。**若 batch=32**，KV Cache → ~53.7 GB，反超参数成为大头 → KV Cache 优化（PagedAttention/量化）是推理服务核心。详见 [[llm-optimizer/kv-cache]] 与 [[llm-inference/KV-Cache优化]]。

## 数值手算汇总表

模型状态按 fp16+Adam = $16P$；推理参数按 fp16 = $2P$。（$P$ 取近似整值）

| 模型 | $P$ | 训练模型状态 $16P$ | 推理参数 $2P$ | KV/token/层(fp16) | KV@b1,s4096 |
|------|-----|-------------------|---------------|-------------------|-------------|
| 7B  ($h$=4096,$L$=32) | 7e9  | 112 GB | 14 GB | $2hL$=256 KB → 全模 8.4 MB/tok·? | ~2.1 GB |
| 13B ($h$=5120,$L$=40) | 13e9 | 208 GB | 26 GB | — | ~3.4 GB |
| 70B ($h$=8192,$L$=80) | 70e9 | 1120 GB | 140 GB | — | ~21.5 GB* |

\* 70B 用 GQA（8 KV 头 vs 64 查询头），实际 KV Cache 约为上表的 $8/64=1/8$，即 ~2.7 GB。

KV@b,s 通式：$2\cdot b\cdot s\cdot L\cdot h\cdot 2\text{B}$。如 7B,b=1,s=4096：$2\times1\times4096\times32\times4096\times2=2.15\times10^9$ B ≈ 2.1 GB ✓。

**训练总显存粗公式（单卡，无并行）**：

$$M_{train} \approx \underbrace{16P}_{\text{模型状态}} + \underbrace{A(b,s,h,L)}_{\text{激活, 有无重计算差别巨大}} + \text{缓冲/碎片}$$

**推理总显存粗公式**：

$$M_{infer} \approx \underbrace{2P}_{\text{参数}} + \underbrace{2bsLh\cdot2}_{\text{KV Cache}} + \text{临时激活(小)}$$

## 常见问题

| 问题 | 答案 |
|------|------|
| 为什么训练显存远大于推理？ | 训练有梯度(2P)+优化器(12P)+反向激活；推理只有参数+KV，约 $2P$ vs $16P$+激活。 |
| "1B 参数要多少显存"？ | 推理 fp16 约 2GB；训练(混合精度+Adam)约 16GB。 |
| 激活值和参数谁大？ | 短序列/小 batch 参数大；长序列/大 batch 激活(尤其 $O(s^2)$ 注意力)可反超 → 用重计算/FlashAttention。 |
| FlashAttention 省的是哪块？ | 省激活值里的 $O(s^2)$ 注意力得分矩阵，不省参数。 |
| ZeRO-3 和张量并行区别？ | ZeRO-3 沿数据并行维切模型状态、按需 AllGather 参数；TP 切层内权重、每层 AllReduce 激活。 |
| KV Cache 怎么省？ | GQA/MQA 减少 KV 头、KV 量化(int8/fp8)、PagedAttention 减碎片、滑动窗口截断。 |
| 8-bit Adam 省多少？ | 优化器从 12P 降到约 4P，模型状态从 16P 降到约 8P。 |
| 为何留 fp32 主副本？ | 防 fp16 下"小更新量"被舍入下溢，保证收敛精度。 |
| 估算时还要留多少余量？ | 实际另需 ~10-20% 给通信缓冲、显存碎片、临时张量、CUDA context。 |
| fp8 训练能省多少？ | 参数/梯度/部分激活降到 1B，但需缩放因子且精度敏感，见 [[llm-compression/quantization/fp8]]。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全仓导航枢纽
- [[llm-algo/transformer/模型架构]] — 参数量从结构推导的源头
- [[llm-algo/FLOPs]] — 与内存对偶的计算量估算（$6PD$ 法则）
- [[llm-algo/mlp]] / [[llm-algo/moe/README]] — MLP/MoE 参数与显存差异
- [[llm-algo/旋转编码RoPE]] — 位置编码对 KV Cache 的影响
- [[llm-optimizer/FlashAttention]] — 消除 $O(s^2)$ 激活的关键
- [[llm-optimizer/kv-cache]] / [[llm-inference/KV-Cache优化]] — KV Cache 估算与优化
- [[llm-inference/解码策略]] — 解码方式如何影响 KV 增长
- [[llm-inference/大模型推理张量并行]] / [[llm-inference/README]] — 推理并行分摊显存
- [[llm-compression/quantization/量化基础]] / [[llm-compression/quantization/fp8]] — 量化降参数/KV显存
- [[llm-train/README]] — 训练全流程显存视角
- [[ai-framework/deepspeed/README]] — ZeRO-1/2/3 切分显存
- [[ai-framework/megatron-lm/README]] — TP/PP 切分显存
- [[ai-infra/网络/集合通信原语]] — AllReduce/AllGather 通信代价
- [[ai-infra/算力/GPU工作原理]] / [[ai-infra/ai-hardware/CUDA]] — HBM/SRAM 与显存层级
- [[llm-optimizer/计算通信重叠]] — 并行时通信缓冲对显存的占用
