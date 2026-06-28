# Reducing Activation Recomputation in Large Transformer Models（减少大型 Transformer 训练中的激活重计算）

> 一句话定位：用「序列并行 + 选择性激活重计算」把训练显存里最贵的那块——激活（activation）——砍掉约 5×，同时把传统全量重计算带来的 30%+ 速度损失压到 <3%。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] · [[docs/transformer内存估算]] · [[B07:llm-inference/大模型推理张量并行]] · [[llm-optimizer/FlashAttention]]

> 论文：Korthikanti et al., NVIDIA, 2022。arXiv: https://arxiv.org/pdf/2205.05198 （Megatron-LM 团队，配套实现进了 Megatron-LM）

## 阅读地图

| 你想知道 | 跳到 |
|---|---|
| 这篇到底解决什么问题 | §0 锚点 / §1 地基 |
| 训练显存都被谁吃了 | §2 显存四大块 |
| 一层 Transformer 的激活到底有多少字节 | §3 激活显存公式（核心） |
| 张量并行为什么没省到激活 | §4 TP 的盲区 |
| 序列并行怎么补刀 | §5 Sequence Parallelism |
| 为什么不全量重计算、怎么"选择性" | §6 选择性重计算（核心创新） |
| 公式 + 手算一遍 22B/175B | §7 数值示例 |
| 效果、局限、和谁对比 | §8 评价表 |

## 0. 一句话锚点

**问题**：模型越大、序列越长，训练时要缓存给反向传播用的「激活」就越多，显存先于参数爆炸。
**老办法**：activation checkpointing（梯度检查点）——前向只存每层入口，反向时把整层重算一遍。省显存，但**多算了一整遍前向**，约 +30~40% 训练时间。
**这篇的两招**：
1. **序列并行（Sequence Parallelism, SP）**：把张量并行（TP）切不到的那些「沿序列维逐元素」的算子（LayerNorm、Dropout）也切开，分摊到各 GPU，**不增加通信量**就把激活显存再分掉。
2. **选择性激活重计算（Selective Activation Recomputation）**：只重算「占显存大、但 FLOPs 便宜」的那部分（注意力里的 $QK^T$、softmax、dropout、$\cdot V$），其余直接存。
**合起来**：激活显存 ↓ 约 5×，重计算时间开销从 30%+ 降到 ↓90%（即剩 ~2-3%）。具体数字见原文。

## 1. 地基：训练为什么要存激活，存了多少

反向传播链式法则：算 $\frac{\partial L}{\partial W}$ 通常要用到该层**前向的输入/中间结果**。例如 $Y = XW$，则 $\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y}$——必须留着 $X$。这些为反向保留的前向中间值就是**激活（activations）**。

```
前向: x ──L1──► a1 ──L2──► a2 ──L3──► loss
            存a1    存a2          （都得留到反向用）
反向:        ◄──── 用a2 ──── 用a1 ────
```

关键直觉：**激活显存 ∝ batch × 序列长度 × 隐藏维 × 层数**，而参数显存只 ∝ 参数量（与 batch/序列无关）。所以一旦上长序列、大 batch，激活会**线性甚至更快**膨胀，成为训练 OOM 的头号元凶。

## 2. 训练显存的四大块（先把账分清）

```
┌─────────────────────────────────────────────┐
│  训练一步的 GPU 显存                            │
├───────────────┬─────────────────────────────┤
│ 1 参数 W       │ 2 bytes/param (fp16/bf16)    │  与序列无关
│ 2 梯度 g       │ 2 bytes/param               │  与序列无关
│ 3 优化器状态    │ Adam: fp32 W/m/v ≈12 b/param│  与序列无关 (→ZeRO/Optimizer切分)
│ 4 激活 a       │ ∝ b·s·h·L  ← 本文主角        │  随 batch/序列暴涨
└───────────────┴─────────────────────────────┘
```

1-3 已有成熟解法：混合精度、ZeRO/优化器并行、TP/PP 切分参数。**唯独第 4 块激活，本文之前缺乏高效手段**——要么硬存（OOM），要么全量重算（慢）。本文专攻第 4 块。

## 3. 激活显存公式（论文核心账本）

论文给出**单层 Transformer**（不含重计算）每个样本所需缓存激活的字节数。记号：$s$=序列长，$b$=micro-batch，$h$=隐藏维，$a$=注意力头数，$t$=张量并行度。fp16 下每元素 2 字节（dropout mask 例外，存为 1 字节）。

**单层、无并行**的激活近似为：

$$
\text{Act/层} = s\,b\,h\left(34 + 5\,\frac{a\,s}{h}\right)\ \text{bytes}
$$

- 前半 $34\,sbh$：来自两个 LayerNorm、QKV 投影、注意力输出投影、MLP 两层、各 dropout 等**线性于 $h$** 的部分。
- 后半 $5\,sbh\cdot\frac{as}{h} = 5\,a\,s^2 b$：来自注意力内部 $QK^\top$（得到 $s\times s$ 矩阵）、softmax、softmax-dropout、对 $V$ 加权——这块**随 $s^2$ 增长**，长序列时主导显存，正是后面要"选择性重算"的目标。

> ASCII：一层激活按"贡献"拆开
> ```
> [LN1][QKV投影][注意力核心 QK^T/softmax/drop/·V][输出投影][LN2][MLP-h→4h][GeLU][MLP-4h→h][dropout]
>   2h    ...        ← 这段 ∝ a·s²  →               ...   2h   ...    8h    ...      s/8(mask)
>        线性于h的"34sbh"   |    平方于s的"5as²b"            线性于h的"34sbh"
> ```

整网 = 上式 × 层数 $L$。这就是"为什么训大模型/长上下文激活会先爆"的定量解释。

## 4. 张量并行（TP）的盲区：它没切到全部激活

回顾 Megatron 张量并行（[[B07:llm-inference/大模型推理张量并行]]）：把注意力的 QKV/输出投影、MLP 的两个权重矩阵按列/行切到 $t$ 张卡，配 1 次 all-reduce（前向）+ 1 次（反向）。

```
TP 切到的部分（÷t）:  [QKV][注意力核心][投影] [MLP1][GeLU][MLP2]
TP 没切的部分（×1）:  [LayerNorm] [输入/输出处的 Dropout]
                      ↑ 这些"逐 token、逐元素"的算子在每张卡上是完整复制的
```

把 §3 公式按 TP 重新记账：能被 TP 分摊的项变成 $/t$，但 **两个 LayerNorm 与首尾 Dropout 仍是全量 $\times 1$**：

$$
\text{Act/层(TP)} = s\,b\,h\left[10 + \frac{24}{t} + 5\,\frac{a\,s}{h\,t}\right]
$$

那个常数 **10**（≈ 2×LN + dropout 等）不随 $t$ 缩小——TP 卡再多，这部分激活也不降。这就是 TP 的盲区，序列并行专门来补它。

## 5. 序列并行（Sequence Parallelism, SP）：补上 TP 的盲区

**核心观察**：LayerNorm、Dropout 这些算子在 token 之间是**独立的**（逐元素/逐 token），所以可以沿**序列维 $s$** 把张量切成 $t$ 段，分给同一 TP 组的 $t$ 张卡，各算各的 token，激活就也 $/t$ 了。

难点：TP 区按 $h$ 维切，SP 区按 $s$ 维切，两者交界要换布局。Megatron 的妙处是**不增加通信总量**——把原来 TP 里的一对 all-reduce **拆成 all-gather + reduce-scatter**：

```
          序列并行区(按 s 切)          张量并行区(按 h 切)
 ┌───────────────┐   g (all-gather)  ┌───────────────────────┐
 │ LayerNorm     │ ───────────────►  │ QKV / Attention / MLP │
 │ (s/t 段)      │                   │ (h/t 切)              │
 └───────────────┘   ḡ(reduce-scatter)└──────────┬───────────┘
          ▲ ◄──────────────────────────────────┘
          │  Dropout / 残差 又回到按 s 切
```

通信账（关键结论）：
- 原版 TP：每层前向 **2 次 all-reduce**（注意力后 1、MLP 后 1），反向同理。
- SP+TP：每层前向 **2 次 all-gather + 2 次 reduce-scatter**。
- 而 1 次 all-reduce 的环形实现 = 1 次 reduce-scatter + 1 次 all-gather，**通信量相等**。所以 SP **白嫖**了 LayerNorm/Dropout 激活的 $/t$ 分摊，通信量不变。

加上 SP 后，§4 那个不缩的"10"也变成 $/t$，单层激活近似为：

$$
\boxed{\ \text{Act/层(TP+SP)} = \frac{s\,b\,h}{t}\left(34 + 5\,\frac{a\,s}{h}\right)\ }
$$

即**整层激活都被 $t$ 均摊**了。漂亮，但当 $s$ 很大时 $5\frac{as}{h}$ 那项（$\propto s^2$）依旧是大头——交给下一招。

## 6. 选择性激活重计算（本文最核心的创新）

**全量重计算**（传统 checkpointing）：每层只存入口 $sbh$，反向把整层重算，省到极致但**计算翻倍**。

**本文洞察**：层内不同算子的"性价比"差别巨大——
- 注意力核心（$QK^\top$、softmax、softmax-dropout、$\cdot V$）：**显存占 $5as^2b$（随 $s^2$，巨大），但 FLOPs 很便宜**（无大权重矩阵乘，逐元素 + 小矩阵乘）。→ **不存，反向重算**，省下大头显存、几乎不增计算。
- 其余（投影、MLP 的 $h\to4h\to h$）：**FLOPs 贵（大矩阵乘），显存相对小**。→ **直接存**，避免昂贵重算。

```
            显存占用                重算代价(FLOPs)        策略
QK^T/softmax/drop/·V   ████████(∝s²)   ▌(便宜)          → 丢弃，反向重算 ✅
QKV/输出投影/MLP        ██              ████████(贵)      → 存住，不重算  ✅
LayerNorm/Dropout      ▌               ▌                → 存(已被SP分摊)
```

只重算注意力核心后，单层"需缓存"的激活变为（去掉那个 $5\frac{as}{h}$ 项）：

$$
\text{Act/层(TP+SP+选择性)} = \frac{34\,s\,b\,h}{t}\ \text{bytes}
$$

**整层激活只剩与 $s$ 线性、且被 $t$ 均摊的 $34\,sbh/t$**，彻底摆脱 $s^2$ 项。重算的只是几个便宜算子，额外 FLOPs 约只占整层前向的很小一部分——这正是"省 5× 显存、却几乎不掉速"的来源。

> 对照 FlashAttention：[[llm-optimizer/FlashAttention]] 用分块在 SRAM 里在线 softmax，根本不把 $s\times s$ 落到 HBM；本文是"先生成、反向时重算"。二者目标一致（杀掉 $s^2$ 激活），路径不同，可叠加。

## 7. 关键公式汇总 + 数值手算

把演进串起来（单层，bytes）：

| 配置 | 单层激活公式 | $s^2$ 项是否还在 | 是否随 $t$ 均摊 |
|---|---|---|---|
| 无并行 | $sbh\,(34 + 5\tfrac{as}{h})$ | 在 | 否 |
| 仅 TP | $sbh\,[10 + \tfrac{24}{t} + 5\tfrac{as}{ht}]$ | 在($/t$) | 部分(留"10") |
| TP+SP | $\tfrac{sbh}{t}(34 + 5\tfrac{as}{h})$ | 在($/t$) | 全部 |
| **TP+SP+选择性重算** | $\tfrac{34\,sbh}{t}$ | **消除** | 全部 |

**手算示例 A：GPT-3 175B 类配置**（取常见设定，仅示意，精确以原文为准）
$h=12288,\ a=96,\ s=2048,\ b=1,\ L=96,\ t=8$。

- 无并行、单层、含 $s^2$ 项：
  $\frac{as}{h}=\frac{96\cdot2048}{12288}=16$，故 $34+5\cdot16=114$。
  单层 $=2048\cdot1\cdot12288\cdot114 \approx 2.87\times10^{9}$ bytes ≈ **约 2.7 GiB/层**。整网 ×96 ≈ 不可能放下。
- TP+SP（$/t=/8$）：单层 ≈ 2.7GiB$/8$ ≈ **约 0.34 GiB/层**。
- 再加选择性重算（只剩 $34sbh/t$）：
  $34\cdot2048\cdot12288\cdot1 /8 \approx 1.07\times10^{8}$ bytes ≈ **约 0.10 GiB/层**。
- 从 ~2.7GiB → ~0.10GiB，**约 26×**（含并行 8×）；若只看 SP→选择性这一步，$114/34\approx3.4×$，叠加 LayerNorm 重新分摊后整体约 5×，与论文"约 5×"口径一致（精确数字见原文）。

**手算示例 B：长序列敏感性**——把 $s$ 翻倍到 4096，$\frac{as}{h}=32$，$34+5\cdot32=194$。
- 含 $s^2$：单层激活 ∝194，**比短序列翻倍还多**（非线性）。
- 选择性重算后：仍是 $34sbh$，只**随 $s$ 线性**翻倍。
→ 直观说明：**序列越长，本文方法收益越大**（把 $s^2$ 拍成 $s$）。

**通信量校验**（§5 结论的数值版）：环形 all-reduce 通信量 $=2\frac{t-1}{t}M$，等于 reduce-scatter $\frac{t-1}{t}M$ + all-gather $\frac{t-1}{t}M$。SP 把 2 次 all-reduce 换成 2 组(scatter+gather)，总量不变 → **SP 显存收益是"免费"的**。

## 8. 评价 / 对照 / 局限

| 维度 | 结论 |
|---|---|
| 激活显存 | 约 ↓5×（SP 分摊 + 消除 $s^2$ 重算项），见原文表 |
| 速度开销 | 重计算开销从 ~30-40% 降到 ~2-3%（↓>90%），因为只重算便宜算子 |
| 端到端吞吐 | 在 22B~1T 规模上 MFU 显著提升（具体百分比见原文） |
| 通信 | SP 不增加通信总量（all-reduce ↔ scatter+gather 等价拆分） |
| 与全量 checkpoint 对比 | 同等或更省显存，但几乎不掉速——"既要又要" |
| 与 ZeRO/Optimizer 切分对比 | 正交：那是切参数/优化器(块1-3)，本文切激活(块4)，可叠加 |
| 与 FlashAttention 对比 | 目标重叠(杀 $s^2$ 激活)，机制不同，可叠加；FA 还省 HBM 读写 |
| 局限 1 | SP 依赖 TP 组内 NVLink 级带宽；跨节点慢链路收益打折 |
| 局限 2 | 选择性重算的"该重算哪些"是针对标准 Transformer 结构手工裁定，新结构需重新分析 |
| 局限 3 | 公式按标准 MHA 推导；MQA/GQA、MoE（[[llm-algo/moe/README]]）需重算账本 |
| 后续影响 | 已并入 Megatron-LM，成为大模型训练的事实标准之一；长上下文训练尤其受益 |

**对工程的启示**：
1. 训大模型先把显存"四大块"分清，别把切参数(ZeRO)和切激活(本文)混为一谈。
2. 开了 TP 一定顺手开 SP——几乎零成本再分摊 LayerNorm/Dropout 激活。
3. 不要无脑全量 checkpointing；优先"选择性"——按"显存大 / FLOPs 便宜"原则挑算子重算。
4. 序列越长越该用本文方法；与 FlashAttention 叠加效果最佳。

## 🔗 跳转链接

- [[00-知识地图]]
- [[llm-train/README]] · [[llm-train/pytorch/distribution/README]]
- [[B07:llm-inference/大模型推理张量并行]]
- [[docs/transformer内存估算]]
- [[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- [[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]]
- [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/GPU工作原理]]
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
