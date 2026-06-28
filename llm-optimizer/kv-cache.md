# KV Cache 原理与优化
> 自回归推理把已算过的 Key/Value 缓存下来,用空间换时间——但这块缓存会随序列线性膨胀,成为长上下文推理的头号显存与带宽瓶颈。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/KV-Cache优化]] [[llm-optimizer/FlashAttention]] [[llm-algo/FLOPs]]

## 阅读地图

| 节 | 你会得到什么 | 关键产物 |
|----|------------|---------|
| 0 | 一句话锚点 | KV Cache 的本质 |
| 1 | 地基:注意力为什么要缓存 | Q/K/V 是什么、自回归是什么 |
| 2 | 没有缓存会怎样(重算的浪费) | $O(n^2)$ vs $O(n)$ |
| 3 | KV Cache 是什么、怎么工作 | prefill / decode 两阶段 |
| 4 | 大小公式 $2\cdot L\cdot h\cdot d_h\cdot n$ | 逐项拆解每个字母 |
| 5 | 手算显存 | LLaMA-7B / 13B 实测数字 |
| 6 | 为什么 decode 是带宽瓶颈 | 算术强度 < 1 |
| 7 | 优化总览 | MQA/GQA/MLA/量化/Paged |
| 8 | 数值大对照 | 各方案省多少 |
| — | 常见问题 + 跳转 | 排坑表 |

## 0. 一句话锚点

> **KV Cache = 把每个已生成 token 在每一层算出的 Key 向量和 Value 向量存下来,这样生成下一个 token 时不必把前面所有 token 重新过一遍模型。**

它把 decode 阶段每步的注意力计算从"重算整段历史"的 $O(n^2)$ 降到"只算新 token"的 $O(n)$,代价是一块随序列长度 **线性增长** 的显存。理解 KV Cache,等于理解 LLM 推理的成本结构。

## 1. 地基:自注意力与自回归(从最原子讲起)

不假设你记得任何公式。先把三件事讲死。

### 1.1 token 与隐藏向量

一段文字被切成 token(词片)。每个 token 进入模型后,在每一层都被表示成一个长度为 $d_{model}$ 的向量(隐藏态)。比如 LLaMA-7B 的 $d_{model}=4096$。

### 1.2 Q / K / V 是怎么来的

在每个注意力层,当前层的隐藏向量 $x$ 通过三个不同的权重矩阵投影出三样东西:

$$Q = x W_Q,\quad K = x W_K,\quad V = x W_V$$

直觉(三个角色):

```
Query (Q)  = "我想找什么"     —— 当前 token 的提问
Key   (K)  = "我能被找到的标签" —— 历史每个 token 的索引
Value (V)  = "我携带的内容"     —— 历史每个 token 的信息
```

注意力做的事:用我的 Q 去和所有历史的 K 做点积,得到"我该关注谁"的权重,再用这权重对所有历史的 V 加权求和,得到输出。

$$\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V$$

### 1.3 自回归(autoregressive):一次只生成一个 token

LLM 解码是 **逐 token** 的:已知前 $t$ 个 token,预测第 $t{+}1$ 个;把它接到序列尾部,再预测第 $t{+}2$ 个……

```
步骤1: [今天]            → 预测 → 天
步骤2: [今天 天]         → 预测 → 气
步骤3: [今天 天 气]      → 预测 → 真
步骤4: [今天 天 气 真]   → 预测 → 好
        ^^^^^^^^^^^^^ 每一步都要对"全部历史"做注意力
```

**关键观察**:第 $t{+}1$ 步预测,只有最后那个新 token 的 Q 是新的;而历史 token 的 K、V 在前面的步骤里 **早就算过了,且永远不变**(因为它们的输入不变)。这就是缓存的根据。

## 2. 没有缓存会怎样:每步都重算整段历史

如果不缓存,第 $t$ 步要把长度为 $t$ 的整段序列重新前向一遍,算出全部 $t$ 个 K、V。生成一段长 $n$ 的文本,总计算量是:

$$\sum_{t=1}^{n} t = \frac{n(n+1)}{2}=O(n^2)$$

```
无缓存(每步重算)            有缓存(每步只算新token)
step1: 算1个               step1: 算1个 + 存1个KV
step2: 算2个(重算1)        step2: 算1个(读历史KV)
step3: 算3个(重算1,2)      step3: 算1个(读历史KV)
...                        ...
总计: 1+2+...+n = O(n²)     总计: n 次, O(n)
```

99% 的重复计算被白白浪费——这正是 KV Cache 要消灭的。

## 3. KV Cache 是什么、两阶段怎么工作

**KV Cache 就是一块显存缓冲区**,存下每一层、每个历史 token 的 K 向量和 V 向量。推理分两个阶段:

### 3.1 Prefill(预填充)阶段

把整段 prompt 一次性并行喂进模型,算出 prompt 里 **每个 token 在每一层** 的 K、V,**全部写入缓存**。这一步是计算密集(大矩阵乘,GPU 算力吃满)。

### 3.2 Decode(解码)阶段

每次只输入 **上一步生成的那 1 个 token**:
1. 算它的 Q、K、V(只是一个向量,不是矩阵);
2. 把新的 K、V **追加(append)** 到缓存末尾;
3. 用新 Q 和缓存里 **全部** K 做注意力,再对全部 V 加权;
4. 输出下一个 token。

```
                      KV Cache (每层一份, 这里画某一层)
                  ┌──────┬──────┬──────┬──────┬───────┐
 已生成历史的K →  │ K1   │ K2   │ K3   │ K4   │ [K5]← │ append
 已生成历史的V →  │ V1   │ V2   │ V3   │ V4   │ [V5]← │
                  └──────┴──────┴──────┴──────┴───────┘
                     ↑读取全部K做点积       ↑本步新写入
 新token的Q5 ─────────┘
   score = softmax(Q5·[K1..K5]/√dh)
   out   = score · [V1..V5]
```

缓存只增不改:历史 K/V 永不重算,新 token 只 append。这是 decode 高效的根本。

## 4. 大小公式:$2\cdot L\cdot h\cdot d_h\cdot n$(逐字母拆)

一个序列、一个样本,KV Cache 的元素个数:

$$N_{elem}=2\cdot L\cdot h\cdot d_h\cdot n$$

逐项解释"为什么是它":

| 符号 | 含义 | 为什么出现 |
|------|------|-----------|
| $2$ | K 和 V 两份 | 每个位置既存 Key 又存 Value |
| $L$ | 层数 (num layers) | 每一层都有独立的注意力,各存一份 |
| $h$ | 注意力头数 (num heads) | 多头注意力,每个头独立的 K/V |
| $d_h$ | 每个头的维度 (head dim) | $d_h = d_{model}/h$,单头向量长度 |
| $n$ | 序列长度 (已缓存 token 数) | 每个历史 token 占一格 |

注意 $h\cdot d_h = d_{model}$,所以也常写成:

$$N_{elem}=2\cdot L\cdot d_{model}\cdot n$$

再乘上 **每个元素的字节数** $b$(FP16=2,FP8/INT8=1,FP32=4)和 **batch 大小** $B$,得到总字节:

$$\boxed{\;\text{Bytes}=2\cdot B\cdot L\cdot h\cdot d_h\cdot n\cdot b\;}$$

**核心结论:在模型确定后,$L,h,d_h$ 都是常数,显存正比于 $B\cdot n$——即批大小 × 序列长度。** 序列翻倍,KV Cache 翻倍;这就是长上下文的成本来源。

## 5. 逐数手算:KV Cache 到底吃多少显存

### 5.1 LLaMA-7B,单序列,FP16

已知:$L=32$,$h=32$,$d_h=128$(故 $d_{model}=4096$),$b=2$ 字节,$B=1$。

**单个 token 的 KV(所有层合计):**
$$2\times 32 \times 32 \times 128 \times 2 = 2\times 32\times 4096\times 2 = 524{,}288 \text{ 字节} \approx 0.5\text{ MB}$$

逐步拆:
```
2(K+V) × 32层 × 32头 × 128维 = 262,144 个元素
262,144 × 2 字节(FP16)       = 524,288 字节
524,288 / 1024 / 1024         ≈ 0.5 MB  ← 每个token每序列约0.5MB
```

**序列长 $n=2048$:**
$$0.5\text{ MB} \times 2048 = 1024\text{ MB} = 1\text{ GB}$$

**序列长 $n=8192$:**
$$0.5\text{ MB}\times 8192 = 4096\text{ MB} = 4\text{ GB}$$

一个 7B 模型权重本身 FP16 约 14 GB;序列拉到 8K,光 KV Cache 就再吃 4 GB。**batch=16、8K 上下文** 时:$4\text{GB}\times 16 = 64\text{ GB}$——KV Cache 已远超模型权重,成为显存主角。

### 5.2 LLaMA-13B 对照

$L=40$,$h=40$,$d_h=128$($d_{model}=5120$),FP16:

```
每token = 2 × 40 × 40 × 128 × 2 = 819,200 字节 ≈ 0.78 MB
n=2048  : 0.78 × 2048 ≈ 1.6 GB
n=4096  : 0.78 × 4096 ≈ 3.1 GB
```

**记忆法**:KV Cache(GB) ≈ 每token的MB × 序列长 × batch ÷ 1024。

## 6. 为什么 decode 是"带宽瓶颈"而非"算力瓶颈"

这是 KV Cache 优化的 **物理动机**,务必算清。

### 6.1 算术强度(Arithmetic Intensity)

定义:每从显存搬运 1 字节,能做多少次浮点运算。

$$\text{AI}=\frac{\text{FLOPs}}{\text{Bytes moved}}$$

GPU 有个临界点(roofline 的拐点):以 A100 为例,算力约 $312$ TFLOPS(FP16),显存带宽约 $2$ TB/s,临界算术强度:

$$\text{AI}_{crit}=\frac{312\times10^{12}}{2\times10^{12}}\approx 156 \text{ FLOPs/Byte}$$

低于这个值 → **memory-bound(带宽受限)**;高于 → compute-bound(算力受限)。

### 6.2 decode 一步的算术强度极低

decode 时 batch=1,只处理 1 个 token。注意力那步:
- 要 **读取整块 KV Cache**(随 $n$ 线性增大,几个 GB)。
- 只对这 1 个新 token 做计算(很少的 FLOPs)。

```
prefill: 大矩阵 × 大矩阵  → 算力吃满 (compute-bound)
decode : 向量  × 大矩阵  → 搬一堆KV只算一点 (memory-bound)

         ┌─────────────────────────────┐
   读取→ │   整块 KV Cache (n×d, GB级)  │ ← 每生成1个token都全读一遍
         └─────────────────────────────┘
   计算→   只有 1 个 Q 向量参与 → FLOPs 极少
   ⇒ AI ≪ 156 ⇒ 时间全花在"等显存搬KV",GPU算力闲置
```

**结论**:decode 每步的延迟 ≈ 把 KV Cache 从显存读一遍的时间。所以
$$\text{decode 速度} \propto \frac{\text{显存带宽}}{\text{KV Cache 大小}}$$

**这给出两条优化主线**:① 把 KV Cache 做小(省带宽、省显存);② 把读写组织得更高效(减少搬运/碎片)。下一节的所有方法都落在这两条线上。

## 7. 优化总览:六大方向

### 7.1 MQA(Multi-Query Attention)— 共享 K/V 头

**原理**:Query 仍是 $h$ 个头,但所有头 **共享同一份 K 和 V**(K/V 头数 = 1)。
**为什么省**:公式里 $h$ 这一项(对 K/V)从 $h$ 降到 $1$,KV Cache 缩小 $h$ 倍。

```
MHA:  Q1 Q2 ... Qh     MQA:  Q1 Q2 ... Qh
      K1 K2 ... Kh            └──┬──┘
      V1 V2 ... Vh             共享 K, V (1份)
      h份K/V                   1份K/V → 省 h 倍
```
**权衡**:K/V 表达能力下降,质量略有损失。

### 7.2 GQA(Grouped-Query Attention)— 分组共享

**原理**:MHA 与 MQA 的折中。把 $h$ 个 Query 头分成 $g$ 组,每组共享一份 K/V。K/V 头数 = $g$。
**为什么省**:KV Cache 缩小 $h/g$ 倍。LLaMA-2-70B 用 $h=64,g=8$,省 **8 倍**。

```
GQA (g=2 组示意, h=4):
  组A: Q1 Q2 → 共享 Kα Vα
  组B: Q3 Q4 → 共享 Kβ Vβ
  K/V 头数 = 2 (而非4)  ⇒ 省 4/2 = 2 倍
```
**权衡**:$g$ 越小越省但质量越降;GQA 是当下主流默认(质量几乎无损,显存大省)。

### 7.3 MLA(Multi-head Latent Attention)— 低秩压缩(DeepSeek)

**原理**:不直接缓存完整的 K、V,而是把它们 **压缩成一个低维潜在向量** $c$ 缓存下来,用时再上投影还原。缓存的是 $c$(维度远小于 $h\cdot d_h$)。
**为什么省**:缓存量从 $\propto d_{model}$ 降到 $\propto d_c$(潜在维),DeepSeek-V2 报告把 KV Cache 压到 GQA 的几分之一,同时质量优于 GQA。

```
MHA:  缓存 [K完整 | V完整]    (大)
MLA:  缓存 [ c (低秩潜向量) ]  (小)
            │ 用时上投影还原 ↑
       W_up·c → K,V  (计算换显存)
```
**权衡**:多了上投影计算;实现复杂(还要兼容 RoPE 位置编码)。

### 7.4 KV Cache 量化 — 降低每元素字节数

**原理**:把缓存里的 K/V 从 FP16(2 字节)量化到 INT8(1 字节)甚至 INT4(0.5 字节)。
**为什么省**:公式里 $b$ 直接减半/再减半,显存 ÷2 或 ÷4。

**手算量化 scale**(对称 INT8,把一组数映射到 $[-127,127]$):
$$\text{scale}=\frac{\max|x|}{127}$$
设某组 K 值的最大绝对值是 $|x|_{max}=2.54$:
$$\text{scale}=\frac{2.54}{127}=0.02$$
量化:$q=\text{round}(x/0.02)$;反量化:$\hat x = q\times 0.02$。例如 $x=1.0 \Rightarrow q=50 \Rightarrow \hat x=1.0$;$x=0.531\Rightarrow q=27\Rightarrow \hat x=0.54$(误差 0.009)。
**权衡**:精度损失;K 通常比 V 更敏感,常做 per-channel/分组量化、或只量化 V 来保质量。

### 7.5 PagedAttention(vLLM)— 像操作系统管内存一样管 KV

**痛点**:传统做法给每个序列 **预留一整段连续显存**(按最大长度),实际没用满就 **内部碎片浪费**;不同长度的序列也无法共享。
**原理**:借鉴 OS 虚拟内存分页。把 KV Cache 切成固定大小的 **块(block/page)**,序列的 KV 按需分配到不连续的物理块,用一张 **块表(block table)** 记录逻辑→物理映射。

```
逻辑视图(序列看到连续)      物理显存(实际离散块)
seq A: [blk0][blk1][blk2] ─┐   ┌─→ ▣ phys#7
                           ├──→│   ▣ phys#3
块表 A: 0→7,1→3,2→9 ───────┘   └─→ ▣ phys#9
                                   ▣ phys#? (空闲池, 按需取)
```
**收益**:① 显存碎片几乎归零(浪费 < 4%);② **前缀共享**——多个请求若有相同 prompt 前缀,可让块表指向同一物理块(copy-on-write),省显存 + 省 prefill。
**权衡**:多一层间址,注意力 kernel 需改写以支持分块(配合自定义算子)。

### 7.6 注意力稀疏/淘汰 — 不缓存所有历史

**原理**:并非所有历史 token 都重要。**H2O(Heavy-Hitter Oracle)** 观察到少数"重权 token"贡献了大部分注意力,可只保留这些 + 最近窗口,淘汰其余,**KV Cache 容量封顶**。StreamingLLM 类似:保留"注意力汇聚点(sink)"+ 滑动窗口。
**为什么省**:缓存大小从 $\propto n$ 变为 **常数上界**,支持近乎无限长流式生成。
**权衡**:被淘汰的 token 信息永久丢失,对需要远距离精确回看的任务有损。

> 论文:[H2O: Heavy-Hitter Oracle](https://arxiv.org/pdf/2306.14048) · 代码 https://github.com/FMInference/H2O ;以上各方案的具体超参/实现以官方论文与框架文档为准。

## 8. 数值大对照:同样 LLaMA-7B、8K 上下文、batch=1

以 §5.1 的 FP16 基线 4 GB 为锚,看各方案把 KV Cache 压到多少:

| 方案 | 机制 | 相对基线 | 8K/单序列估算 | 主要代价 |
|------|------|---------|--------------|---------|
| MHA(基线) | 每头独立 K/V | 1× | 4 GB | — |
| MQA | 全头共享 1 份 K/V | ÷h | ≈ 0.13 GB | 质量损失明显 |
| GQA(g=8) | 分 8 组共享 | ÷(h/8) | ≈ 0.5 GB | 质量几乎无损 |
| MLA | 低秩潜向量缓存 | 约 ÷4~÷8 | ≈ 0.5~1 GB | 上投影计算+实现复杂 |
| INT8 量化 | 字节数 ÷2 | ÷2 | 2 GB | 精度损失(小) |
| INT4 量化 | 字节数 ÷4 | ÷4 | 1 GB | 精度损失(较大) |
| GQA+INT8 | 叠加 | ÷16 | ≈ 0.25 GB | 组合调优 |
| H2O/Streaming | 容量封顶 | →常数 | 与 $n$ 解耦 | 远距信息丢失 |
| PagedAttention | 不缩小,去碎片 | ~1×(利用率↑) | 浪费<4% | 实现复杂度 |

注:MQA/GQA 行的"÷h"以 $h=32$ 估;各数为量级估算,实际随实现而异。**正交可叠加**:GQA(架构)+ 量化(精度)+ PagedAttention(内存管理)+ H2O(淘汰)可同时使用。

## 复杂度小结

| 维度 | prefill | decode(每步) |
|------|---------|-------------|
| 计算量 | $O(n^2)$ 一次性 | $O(n)$ 读历史 |
| 瓶颈 | compute-bound | **memory-bound** |
| KV Cache | 写入全部 | 读全部 + append 1 |
| 优化重点 | 算子(FlashAttention) | **缩小/高效搬运 KV** |

## 常见问题

| 疑问 | 真相 |
|------|------|
| KV Cache 存的是注意力分数吗? | 不是。存的是每层每 token 的 **K、V 向量**;分数每步现算。 |
| 为什么不缓存 Q? | Q 只有"当前 token"这一个有用且每步都变,缓存它没意义。 |
| KV Cache 会随生成不断变大吗? | 会,严格正比于已生成长度 $n$,只增不减(除非用 H2O 类淘汰)。 |
| FlashAttention 能减小 KV Cache 吗? | 不能。它优化的是注意力 **计算的显存读写**(中间矩阵不落地),不改变 KV Cache 本身大小。见 [[llm-optimizer/FlashAttention]]。 |
| GQA 和 MQA 谁好? | GQA 是更优折中:质量接近 MHA、显存接近 MQA,故成主流。 |
| 量化 K 和 V 一样安全吗? | 通常 K 更敏感(直接进 softmax);常对 K 用更细粒度量化或保留更高精度。 |
| 为什么 batch 一大显存就爆? | KV Cache ∝ $B\cdot n$,batch 和序列都是线性放大因子,叠乘很可怕。 |
| prefill 慢还是 decode 慢? | prefill 算力受限(长 prompt 慢);decode 带宽受限(长输出慢)。两者瓶颈不同。 |
| PagedAttention 让模型更快了吗? | 不直接加速单步,但去碎片→能塞更大 batch→吞吐(throughput)大涨。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引,KV Cache 在推理优化中的位置
- [[llm-inference/KV-Cache优化]] — 推理侧落地:框架实现、调度与吞吐
- [[llm-optimizer/FlashAttention]] — 注意力计算的显存优化(与 KV Cache 互补,各管一摊)
- [[llm-algo/FLOPs]] — FLOPs/算术强度/roofline,理解 decode 带宽瓶颈的算账基础
