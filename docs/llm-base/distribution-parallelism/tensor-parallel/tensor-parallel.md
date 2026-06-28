# 张量并行（Tensor Parallelism, TP）

> 把单个算子（矩阵乘）按行/列切到多张 GPU 上**同时算**，再用集合通信把结果拼/加回来——让一层放不下的大模型也能训练与推理。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]] · [[ai-framework/megatron-lm/README]] · [[llm-algo/transformer/模型架构]] · [[llm-optimizer/计算通信重叠]]

## 阅读地图

| 节 | 内容 | 关键产出 |
|----|------|----------|
| 0 | 一句话锚点 | TP 是"切算子"不是"切数据" |
| 1 | 地基：为什么要 TP、和 DP/PP 区别 | 三种并行的分工 |
| 2 | 列并行（Column Parallel）| $Y=XA$ 按列切，无需通信 |
| 3 | 行并行（Row Parallel）| $Z=YB$ 按行切，AllReduce |
| 4 | 列+行黄金组合：MLP | 前向 1 次、反向 1 次 AllReduce |
| 5 | 多头注意力的 TP | 按 head 天然切分 |
| 6 | $f$ 与 $g$ 共轭算子 | 前向/反向通信对偶 |
| 7 | 词嵌入与输出层并行 | 词表维度切分 + AllReduce |
| 8 | 通信量与带宽：为什么 TP 不出机 | NVLink vs PCIe |
| 9 | 显存怎么省的 | 参数/激活/优化器逐项手算 |
| 10 | 序列并行（SP）补刀 | 省 LayerNorm/Dropout 激活 |
| - | 数值手算（13B 实例） | 通信量 / 显存 / 带宽 |
| - | 常见问题 + 跳转链接 | - |

## 0. 一句话锚点

数据并行（DP）是"**每张卡放整模型、各算各的数据**"；张量并行是"**一份数据、把每个矩阵乘法横着或竖着锯成 N 段，N 张卡各算一段**"。锯开之后结果不完整，所以必须插入**集合通信**（AllReduce / AllGather）把碎片拼回数学上等价的结果。TP 的核心就两件事：**怎么切矩阵**、**在哪里通信**。

## 1. 地基：三种并行的分工

Transformer 训练有三个正交的切分维度，缺一不可：

```
                ┌─────────── 一个 Transformer 模型 ───────────┐
   数据并行 DP   │  GPU0:整模型   GPU1:整模型   GPU2:整模型     │  切 batch
                │   batch[0:4]    batch[4:8]    batch[8:12]   │  梯度 AllReduce
                └────────────────────────────────────────────┘
   流水并行 PP   层0-7 → GPU0 | 层8-15 → GPU1 | 层16-23 → GPU2   切"层"(深度)
                                                                激活做 P2P 传递
   张量并行 TP   一层内部:  [GPU0 半个矩阵][GPU1 半个矩阵]        切"算子"(宽度)
                                                                层内 AllReduce
```

| 维度 | 切什么 | 通信内容 | 通信频率 | 适合放在哪 |
|------|--------|----------|----------|-----------|
| DP（数据并行）| batch 样本 | 梯度（AllReduce）| 每 step 1 次 | 跨机也行 |
| PP（流水并行）| 模型的层 | 层间激活（P2P）| 每 micro-batch | 跨机/机内 |
| **TP（张量并行）** | 单层内的权重矩阵 | 层内激活（AllReduce）| **每层 2 次** | **必须机内 NVLink** |

> 关键直觉：TP 通信**频率最高、数据量最大**（每一层都要通信），所以它对带宽极度敏感，**必须放在同一台机器的 NVLink 域内**。DP/PP 才允许跨机。详见 [[ai-infra/网络/集合通信原语]]。

延伸阅读：流水并行见 `pipeline-parallelism/README`，数据并行见 `data-parallelism/README`，混合并行见 [[ai-framework/megatron-lm/README]]。

## 2. 列并行：$Y = XA$ 按列切（Column Parallel）

以一个线性层为例，它是通用矩阵乘法（GEMM）：$Y = XA$。设 $X \in \mathbb{R}^{b \times h}$（batch×hidden），$A \in \mathbb{R}^{h \times h'}$。

给定 2 个处理器，把权重 $A$ **按列**切成 $A = [A_1\ A_2]$，每个 $A_i \in \mathbb{R}^{h \times h'/2}$。在每张卡上独立计算：

$$Y_1 = X A_1, \quad Y_2 = X A_2, \quad Y = [Y_1\ Y_2]$$

```
列并行  Y = X·[A1 A2]      X 在两张卡上是完整复制的
        ┌──────── h' ────────┐
   X    │   A1     │    A2    │       GPU0 算 Y1 = X·A1  (b × h'/2)
 (b×h)  │ (h×h'/2) │ (h×h'/2) │       GPU1 算 Y2 = X·A2  (b × h'/2)
        └─GPU0─────┴───GPU1───┘
   前向：各算各的，输出天然按列切，★★无需通信★★
   结果 Y = [Y1 | Y2] 仍是切开状态（留给下一层用）
```

**为什么列切前向不用通信？** 因为输出矩阵 $Y$ 的第 $j$ 列只依赖 $A$ 的第 $j$ 列，列与列之间不耦合。每张卡算出的 $Y_i$ 是最终结果的一部分，直接保留即可，无需求和。

**反向（重点）**：每张卡只能算出输入梯度的局部贡献 $\dot{X}_i = \dot{Y}_i A_i^{T}$。完整输入梯度是

$$\dot{X} = \dot{Y} A^{T} = \dot{Y}_1 A_1^{T} + \dot{Y}_2 A_2^{T}$$

这是一个**求和**，所以**反向需要一次 AllReduce** 把各卡的 $\dot{X}_i$ 加起来。前向不通信、反向通信——这就是后面要讲的 $f$ 算子。

## 3. 行并行：$Z = YB$ 按行切（Row Parallel）

当第二个线性层 $Z = YB$ 跟在列并行层后面时，输入 $Y$ 已经是按列切好的 $[Y_1\ Y_2]$。此时把 $B$ **按行**切：

$$B = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}, \qquad Z = [Y_1\ Y_2]\begin{bmatrix} B_1 \\ B_2 \end{bmatrix} = Y_1 B_1 + Y_2 B_2$$

```
行并行  Z = [Y1 Y2]·[B1; B2]      Y 已经是切开的（来自上一层列并行）
        ┌── h'' ──┐
   Y1   │   B1    │   GPU0 算 Z_part0 = Y1·B1   (b × h'')
 (b×h'/2)│(h'/2×h'')│
   Y2   │   B2    │   GPU1 算 Z_part1 = Y2·B2   (b × h'')
 (b×h'/2)│(h'/2×h'')│
        └─────────┘
   两卡各得 (b × h'') 的"部分和"，维度对，但数值都不完整！
   ★★前向需要 AllReduce★★  Z = Z_part0 + Z_part1
```

**为什么行切前向必须通信？** 因为矩阵乘 $Z_{ij} = \sum_k Y_{ik}B_{kj}$，求和指标 $k$ 横跨被切开的维度。GPU0 只算了 $k$ 落在前半的项、GPU1 只算了后半的项，每张卡拿到的都是**部分和**，必须 AllReduce 相加才得到正确的 $Z$。

**反向**：$\dot{Y}_i = \dot{Z} B_i^{T}$。注意 $\dot{Z}$ 在 AllReduce 后两卡是相同的，各卡乘自己的 $B_i^{T}$ 直接得到对应的 $\dot{Y}_i$，**反向不需要通信**。前向通信、反向不通信——这就是 $g$ 算子，与列并行恰好对偶。

## 4. 黄金组合：MLP 用"列→行"省掉中间通信

Megatron 的精髓：MLP 是两个连续线性层 $\text{Dropout}(\text{GeLU}(XA)B)$。把第一层 $A$ 设为**列并行**、第二层 $B$ 设为**行并行**，则中间结果 $Y=\text{GeLU}(XA)$ **天生就是切开的**，可以直接喂给行并行层，**整个 MLP 前向只需要末尾一次 AllReduce**。

```
      X(完整)        列并行A        行并行B          Z(完整)
   ┌────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐
   │  X 复制 │──▶│ Y1=GeLU  │──▶│ Z0=Y1·B1 │─┐ │        │
   │ 到两卡  │   │ (X·A1)   │   │          │ ├AllReduce▶│ Z │
   │        │──▶│ Y2=GeLU  │──▶│ Z1=Y2·B2 │─┘ │        │
   └────────┘   │ (X·A2)   │   └──────────┘   └────────┘
        ↑         └──────────┘                     ↑
     f 算子                                      g 算子
  前向:直通  GeLU 逐元素,切开算等价         前向:AllReduce
  反向:AllReduce 不破坏切分性               反向:直通
```

为什么 GeLU 能放在切开状态算？因为 GeLU 是**逐元素（element-wise）**函数，$\text{GeLU}([Y_1\ Y_2]) = [\text{GeLU}(Y_1)\ \text{GeLU}(Y_2)]$，对列切不敏感。**关键设计：第一层故意用列并行，就是为了让非线性激活能在切开状态下正确计算**——如果第一层用行并行，GeLU 前就要先 AllReduce，等于白切。

**每个 MLP 块通信账**：前向 1 次 AllReduce（$g$）+ 反向 1 次 AllReduce（$f$）= **2 次**。

## 5. 多头注意力（MHA）的张量并行

注意力天生适合 TP：**多个 head 本来就互相独立**，按 head 分组到各卡即可。设 $a$ 个头、TP 度为 $t$，则每卡负责 $a/t$ 个头。

```
   QKV 投影(列并行)         注意力(各算各的)        输出投影(行并行)
   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
X─▶│ GPU0: head 0-7│─▶ Attn(Q0,K0,V0) ─▶│ O0=·Wo_part0 ─┐    │
   │  Wq/Wk/Wv 切列│      │              │   │           ├AllReduce
X─▶│ GPU1:head 8-15│─▶ Attn(Q1,K1,V1) ─▶│ O1=·Wo_part1 ─┘    │
   │  Wq/Wk/Wv 切列│      └──────────────┘      └──────────────┘
   └──────────────┘
    每卡独立完成自己几个头的 softmax(QKᵀ/√d)V，head 间零通信
```

- **QKV 投影**：$W_Q, W_K, W_V$ 按列并行（等价于按 head 切），每卡得到自己负责头的 Q/K/V。
- **注意力计算**：$\text{softmax}(Q_iK_i^{T}/\sqrt{d})V_i$ 在每张卡内部完整跑，head 之间不耦合，**零通信**。这也是 FlashAttention 能在每卡内独立运行的前提，见 [[llm-optimizer/FlashAttention]]。
- **输出投影**：$W_O$ 按行并行，末尾一次 AllReduce 合并。

每个注意力块通信账：同样是前向 1 次 + 反向 1 次 = **2 次 AllReduce**。

> 注意 GQA/MQA：KV 头数 < Q 头数时，KV 头要在 TP 组内复制或按比例切，细节见 [[llm-inference/大模型推理张量并行]]。RoPE 位置编码逐头施加，与 TP 切分正交，见 [[llm-algo/旋转编码RoPE]]。

## 6. $f$ 与 $g$：一对共轭通信算子

Megatron 用两个对偶算子封装了全部 TP 通信，这是理解 TP 的"代数骨架"：

| 算子 | 位置 | 前向 forward | 反向 backward |
|------|------|--------------|---------------|
| $f$ | 进入并行区（列并行前）| **恒等**（直通，X 复制）| **AllReduce**（聚合 $\dot X$）|
| $g$ | 离开并行区（行并行后）| **AllReduce**（聚合 Z）| **恒等**（直通）|

```
   ─── f ───[列并行]───[行并行]─── g ───
   前向: 直通    无通信    无通信   AllReduce
   反向:AllReduce 无通信    无通信    直通
        ↑___________________________↑
        前向 g 通信 ⇄ 反向 f 通信  （对偶）
```

一个 Transformer 层 = 1 个注意力块 + 1 个 MLP 块，每块各有一对 $(f, g)$。所以**每层前向 2 次 AllReduce、反向 2 次 AllReduce，合计每层 4 次**。这是 TP 通信量估算的基本单位。

## 7. 词嵌入与输出层的并行

词表 $V$ 很大（如 5 万~15 万），嵌入矩阵 $E \in \mathbb{R}^{V \times h}$ 也要切：

- **输入嵌入**：按**词表维度** $V$ 切。每卡只持有部分词的向量；某个 token 若不属于本卡负责的词区间，该卡输出 0，最后 **AllReduce** 求和得到完整嵌入。
- **输出层（LM Head）**：与输入嵌入**权重共享**，按词表切后各卡算自己词区间的 logits，再配合并行交叉熵：只对本卡词区间算分子，分母 $\sum_v e^{z_v}$ 用 **AllReduce** 跨卡求和。这样避免把整个 $b\times s\times V$ 的巨型 logits 张量在单卡上物化。

```
   token id ──▶ ┌ GPU0: 词 0~V/2-1 的 emb，否则置 0 ┐
                │                                  ├AllReduce─▶ 完整 emb
                └ GPU1: 词 V/2~V-1 的 emb，否则置 0 ┘
   输出端反着来：各卡算局部 logits → 并行 softmax 分母 AllReduce
```

## 8. 通信量与带宽：为什么 TP 不能出机

**AllReduce 的通信量**（Ring 实现，每卡收发量）：传输一个大小为 $M$ 字节的张量，每张卡的总通信量约为 $2M(t-1)/t \approx 2M$（与卡数几乎无关，这是 Ring AllReduce 的优良性质）。

单层前向需要传输的激活张量大小 $M = b \cdot s \cdot h \cdot 2$ 字节（fp16/bf16，$b$=batch、$s$=seq、$h$=hidden）。**每层 4 次 AllReduce**（前2反2），所以单层每卡通信量约 $4 \times 2M = 8M$。

```
   带宽分级（数量级，以官方为准）
   NVLink/NVSwitch  ~数百 GB/s ~ TB/s   ← TP 必须在这层
   PCIe 4.0/5.0     ~数十 GB/s          ← 勉强 DP，TP 会卡死
   IB/RoCE 跨机     ~数十~百 GB/s        ← DP/PP 用，TP 禁止
```

直觉结论：TP 每层都通信、且传的是大激活张量，**只有 NVLink 级带宽才扛得住**。一旦 TP 跨 PCIe 或跨机，通信时间会远超计算时间，吞吐崩溃。所以实践铁律：**TP 度 ≤ 单机 GPU 数（通常 8）**，跨机用 PP/DP。通信与计算重叠技巧见 [[llm-optimizer/计算通信重叠]]。

## 9. 显存怎么省的（逐项拆解）

TP 度为 $t$，理想情况下**权重、梯度、优化器状态、以及大部分激活都切成 $1/t$**：

| 显存项 | 单卡（无 TP）| TP 切分后 | 说明 |
|--------|-------------|-----------|------|
| 模型参数 | $P$ | $P/t$ | 权重矩阵被切 |
| 梯度 | $P$ | $P/t$ | 与参数同切 |
| 优化器状态（Adam）| $\sim 6P$（fp32 动量+方差+主权重）| $6P/t$ | 跟着参数切 |
| 激活（线性/注意力）| $A$ | $A/t$ | 被切维度的激活也是 $1/t$ |
| LayerNorm/Dropout 激活 | $A_0$ | $A_0$（不切！）| 需 **序列并行** 才能切，见第 10 节 |

> 对比：DP 不省任何单卡显存（每卡整模型），只是分摊数据；TP 是**真正把单层放不下的模型摊薄**的手段。参数/激活内存的逐项估算见 [[transformer内存估算]]，FLOPs 估算见 [[llm-algo/FLOPs]]。

## 10. 序列并行（Sequence Parallelism, SP）补刀

TP 切不了 LayerNorm 和 Dropout（它们不是矩阵乘，是逐元素/逐 token 操作）。这些激活在每张卡上**完整复制**，成为显存死角。**序列并行**把这些区域按**序列维度 $s$** 切到各卡：

```
   [AllReduce 区]      改成      [AllGather + ReduceScatter 区]
   LayerNorm 复制               LayerNorm 按 s 切 → 激活也省 1/t
   通信总量不变（g=AllReduce 拆成 AllGather+ReduceScatter）
```

SP 与 TP 共用同一组 GPU，**通信总量基本不变**（一次 AllReduce ≈ 一次 AllGather + 一次 ReduceScatter），却额外把 LayerNorm/Dropout 激活也降到 $1/t$。这是长序列训练的关键省显存手段，几乎总是和 TP 一起开。

---

## 数值手算：13B 模型，TP=8，机内 8×A100

设定：hidden $h=5120$，层数 $L=40$，序列 $s=2048$，micro-batch $b=1$，参数 $P=13\times10^9$，精度 bf16（2 字节），TP 度 $t=8$。

**(1) 单卡参数显存**

$$\frac{P \times 2\text{B}}{t} = \frac{13\times10^9 \times 2}{8} = \frac{26\text{ GB}}{8} \approx 3.25\text{ GB（仅权重）}$$

加上梯度（bf16，3.25 GB）和 Adam 优化器状态（fp32 主权重+动量+方差 ≈ $6P/t$）：

$$\frac{6 \times 13\times10^9 \times 4\text{B}}{8} \approx \frac{312\text{ GB}}{8} = 39\text{ GB（优化器）}$$

可见优化器状态是大头，TP 把它从 312 GB 摊到每卡 39 GB——这正是"单层放不下"问题被 TP 解决的关键。

**(2) 单层单次 AllReduce 的激活张量**

$$M = b \times s \times h \times 2\text{B} = 1 \times 2048 \times 5120 \times 2 = 2.1\times10^7\text{ B} \approx 20\text{ MB}$$

**(3) Ring AllReduce 每卡通信量**（系数约 $2(t-1)/t = 2\times7/8 = 1.75$）

$$1.75 \times 20\text{ MB} = 35\text{ MB（单次单卡收发）}$$

**(4) 整个前向（40 层 × 每层 2 次 AllReduce）**

$$40 \times 2 \times 35\text{ MB} = 2800\text{ MB} \approx 2.8\text{ GB（每卡每次前向通信）}$$

**(5) 通信耗时**（设 NVLink 有效带宽 ~200 GB/s，以官方为准）

$$\frac{2.8\text{ GB}}{200\text{ GB/s}} \approx 14\text{ ms（前向通信）}$$

**反思**：14 ms 看着小，但若把这套 AllReduce 放到 PCIe（~20 GB/s）就变成 140 ms，再放到跨机 IB 还要叠加延迟——计算可能只要几十 ms，通信反而成瓶颈。**这从数字上证明了"TP 必须待在 NVLink 域内"的铁律。** 增大 batch（$b$↑）会线性放大通信量 $M$，但同时摊薄启动开销，所以实战要在通信与算力利用间找平衡点。

## 常见问题

| 问题 | 解答 |
|------|------|
| TP 和 DP 能同时用吗？ | 能且常配合。典型 3D 并行 = DP×PP×TP，TP 在最内层（机内 NVLink），见 [[ai-framework/megatron-lm/README]] |
| 为什么列并行接行并行能省通信？ | 列并行输出天然切开，正好做行并行输入，中间 GeLU 逐元素算，整个 MLP 前向只末尾 1 次 AllReduce |
| 为什么 TP 不能跨机？ | 每层都通信、传大激活，只有 NVLink 扛得住；跨 PCIe/IB 通信会数倍于计算，吞吐崩溃 |
| TP 度最大设多少？ | 一般 ≤ 单机 GPU 数（如 8）；再大就被迫跨机，得不偿失 |
| 推理时 TP 还需要吗？ | 需要。大模型权重单卡放不下、或要降延迟时用 TP，KV-Cache 也随之切，见 [[llm-inference/KV-Cache优化]] |
| SP 会增加通信吗？ | 几乎不增。AllReduce 拆成 AllGather+ReduceScatter，总量持平，却多省 LayerNorm/Dropout 激活 |
| TP 影响精度/数值吗？ | 数学上严格等价（AllReduce 求和），bf16 下顺序相关的浮点误差可忽略 |
| MoE 怎么和 TP 结合？ | 专家维度用专家并行（EP），专家内部矩阵乘仍可叠 TP，见 [[llm-algo/moe/README]] 与 `moe-parallel/README` |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全仓库导航总入口
- [[llm-inference/大模型推理张量并行]] — 推理侧 TP（B07 重点，与本文训练侧互补）
- [[ai-infra/网络/集合通信原语]] — AllReduce / AllGather / ReduceScatter 底层机制
- [[ai-framework/megatron-lm/README]] — Megatron-LM 实现：$f/g$ 算子、3D 并行
- [[llm-algo/transformer/模型架构]] — TP 切的就是这里的 MLP / 注意力
- [[llm-algo/mlp]] — MLP 两层结构（列并行→行并行的对象）
- [[llm-optimizer/FlashAttention]] — 每卡内注意力高效计算
- [[llm-optimizer/计算通信重叠]] — 把 TP 的 AllReduce 藏到计算后面
- [[transformer内存估算]] — 参数/激活/优化器显存逐项估算
- [[llm-algo/FLOPs]] — 计算量估算，配合通信量看 roofline
- [[llm-inference/KV-Cache优化]] — 推理 TP 下 KV-Cache 切分
- [[ai-framework/deepspeed/README]] — ZeRO（DP 省显存）与 TP 的取舍
- [[ai-infra/算力/GPU工作原理]] — NVLink/带宽为何决定 TP 边界

参考：
- 图解大模型训练之：张量模型并行 Megatron-LM：https://zhuanlan.zhihu.com/p/622212228
- Megatron 论文和代码详细分析：https://zhuanlan.zhihu.com/p/366906920
- [源码解析] 模型并行分布式训练 Megatron：https://juejin.cn/post/7057837676430360584
- 张量模型并行详解 | 深度学习分布式训练专题：https://www.paddlepaddle.org.cn/support/news?action=detail&id=2913
- Megatron-LM 论文：*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*（Shoeybi et al., 2019）
