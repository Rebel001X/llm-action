# FlashAttention

> 用**分块(Tiling) + 在线 Softmax(Online Softmax) + 重计算(Recompute)** 把注意力从"显存带宽瓶颈(memory-bound)"改造成"片上计算(compute-bound)"的**精确(exact)**注意力算法——不近似，只是少搬数据。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]] · [[llm-algo/transformer/模型架构]] · [[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]]

---

## 阅读地图

| 节 | 内容 | 你将能手算/画出 |
|----|------|----------------|
| 0 | 一句话锚点 | FA 到底改了什么 |
| 1 | 地基：标准注意力 & GPU 存储层级 | 为什么标准注意力是 memory-bound |
| 2 | 屋顶线(Roofline)：算术强度 | 估出 HBM 读写字节数 |
| 3 | Softmax 的数值稳定 & "全量依赖"难题 | 为什么不能简单分块 |
| 4 | Online Softmax：边走边修正 | 两块合并的递推公式 |
| 5 | Tiling：把 Q/K/V 切块流式扫 | 内外循环 + ASCII 数据流 |
| 6 | Recompute：反向不存 $N^2$ 矩阵 | 省下多少显存 |
| 7 | FlashAttention-2：循环换序 & 并行 | 为什么更快 |
| 8 | Flash-Decoding：解码期沿 KV 切分 | 长序列 8x 提速来源 |
| 9 | 数值手算：完整一遍 online softmax | 逐数对账 |
| 10 | 复杂度/显存账本 | $O(N^2)\to O(N)$ 显存 |
| - | 常见问题 + 跳转链接 | |

---

## 0. 一句话锚点

注意力的输出是
$$O = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

标准实现会**显式地把 $N\times N$ 的注意力矩阵 $S=QK^\top$ 和 $P=\mathrm{softmax}(S)$ 写进 HBM(显存) 再读回来**。当 $N$(序列长)很大时，这块 $N^2$ 的中间矩阵的**搬运**(而不是计算)成了瓶颈。

FlashAttention 的核心主张：**结果一字不差**，但是

- **永不把完整的 $S$/$P$ 落到 HBM**——它们只在 SRAM(片上高速缓存) 里以小块形式短暂存在；
- 用 **online softmax** 让 softmax 可以"分段累加"，不需要一次看到整行；
- 反向传播时**不存** $N^2$ 的 $P$，而是用保存的 softmax 统计量(每行的 $m$ 和 $\ell$)**重算**。

口诀：**Online Softmax + Tiling + Recompute**。

```
            标准注意力                          FlashAttention
   ┌──────────────────────────┐      ┌──────────────────────────┐
   │ S = QKᵀ   ──写HBM──▶ N×N │      │  分块循环，S 块只在 SRAM │
   │ P = softmax(S) 写HBM N×N │  vs  │  online softmax 累加 O   │
   │ O = P·V   读HBM 两个 N×N │      │  HBM 只读写 Q,K,V,O (O(N))│
   └──────────────────────────┘      └──────────────────────────┘
     HBM 流量 ~ O(N²)                   HBM 流量 ~ O(N²/M·d) 但常数极小
                                        实测 HBM 访存降 ~5–20×
```

---

## 1. 地基：标准注意力 + GPU 存储层级

### 1.1 标准注意力的逐步算子

设 $Q,K,V \in \mathbb{R}^{N\times d}$($N$ 是序列长，$d$ 是 head 维度)：

```
 step 1   S = Q Kᵀ / √d        形状 N×N        ← 写回 HBM
 step 2   P = softmax(S, 行方向) 形状 N×N        ← 读 S, 写 P 到 HBM
 step 3   O = P V               形状 N×d         ← 读 P, 读 V, 写 O
```

注意力本身的算力 (FLOPs) 是 $O(N^2 d)$，但是它要在 HBM 上**来回搬运** $O(N^2)$ 个元素。问题就出在这里。

### 1.2 GPU 存储层级(为什么搬运比计算贵)

```
        ┌────────────────────────────────────────────┐
        │  寄存器 / SRAM(片上 shared memory)          │
        │  容量: ~20MB(整卡), 单 SM ~100-228KB        │
        │  带宽: ~19 TB/s  ← 极快                      │  快但小
        ├────────────────────────────────────────────┤
        │  HBM(显存, global memory)                   │
        │  容量: 40–80 GB(A100/H100)                  │
        │  带宽: ~1.5–3.3 TB/s ← 慢一个数量级          │  大但慢
        ├────────────────────────────────────────────┤
        │  CPU DRAM                                    │
        └────────────────────────────────────────────┘
```

关键事实：**SRAM 比 HBM 快约 10 倍，但只有几十 KB/SM**。一个 $N=4096$ 的 $S$ 矩阵就是 $4096^2\times2\text{B}=32\text{MB}$(fp16)，**根本塞不进 SRAM**，所以标准实现被迫往 HBM 倒。FlashAttention 的全部技巧，就是**永远只在 SRAM 里处理小块**，避免那次 $32\text{MB}\times$多次的 HBM 往返。

> 详见 [[ai-infra/算力/GPU工作原理]]、[[ai-infra/ai-hardware/CUDA]]。

---

## 2. 屋顶线 / 算术强度：先把账算明白

判断一个算子是 **compute-bound(算力受限)** 还是 **memory-bound(带宽受限)**，看**算术强度** = FLOPs / 访存字节(Arithmetic Intensity, AI)。

对标准注意力(忽略常数)，每个元素只参与极少的乘加，但要被读写多次。我们粗算 HBM 访存字节(fp16，2 字节/元素，$N=4096$, $d=64$)：

```
 写 S:  N² 元素 = 4096² × 2B = 32 MB
 读 S + 写 P(softmax): 2 × 32 MB = 64 MB
 读 P + 写 O: ~32 MB + 0.5 MB
 ──────────────────────────────────
 仅中间 N² 矩阵的 HBM 流量 ≈ 128 MB / 每个 head
```

而真正"有用"的计算只有 $2\cdot 2N^2 d \approx 2\times2\times4096^2\times64 \approx 8.6$ GFLOPs。

- A100 算力 ~312 TFLOPS(fp16)，做 8.6 GFLOPs 需 $\approx 27\,\mu s$。
- A100 HBM ~2 TB/s，搬 128 MB 需 $\approx 64\,\mu s$。

**搬运时间 > 计算时间 → memory-bound**。FlashAttention 把那 128MB 的 $N^2$ 往返**几乎全部消除**(只剩 $Q,K,V,O$ 各 $N\times d$ 的一次读写)，于是端到端被带宽卡的部分被打掉。

```
       FLOPS ↑
   312T ┤ ░░░░░░░░░░░░░ 屋顶(算力上限)
        │ ░░░░░╱
        │ ░░░╱  ← FlashAttention 把工作点
        │ ░╱        往右推(提高算术强度)
        │╱  ● 标准注意力(贴在带宽斜坡上)
        └────────────────────────▶ 算术强度 (FLOP/Byte)
```

---

## 3. Softmax 的数值稳定与"全量依赖"难题

Softmax 对一行 $x=(x_1,\dots,x_N)$：
$$\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j}e^{x_j}}$$

直接算 $e^{x_i}$ 会溢出，所以工程上一律减去**行最大值** $m=\max_j x_j$(safe softmax)：
$$\mathrm{softmax}(x)_i = \frac{e^{x_i-m}}{\sum_j e^{x_j-m}}$$

这等价、但数值安全。**难点来了**：$m$ 和分母 $\ell=\sum_j e^{x_j-m}$ 都需要**看到整行 $N$ 个数**。如果我们想把 $K,V$ 切成块、流式地一块块算，第一块算的时候根本不知道后面会不会冒出更大的 $x$——一旦后面出现更大的值，前面用旧 $m$ 算出的指数就全错了。

```
   一行 logits(被切成 3 块):
   ┌────────┬────────┬────────┐
   │ 块1 m1 │ 块2 m2 │ 块3 m3 │
   └────────┴────────┴────────┘
   想要的全局 m = max(m1,m2,m3)，但流式时还没见到块2/块3！
```

这正是 **online softmax** 要解决的：**边扫边修正**，每来一个新块，就把之前累计的结果按新的全局最大值"重新缩放"。

---

## 4. Online Softmax：两块合并的递推

参考：*Online normalizer calculation for softmax*（arXiv:1805.02867）。

我们维护三个**运行量(running state)**：
- $m$：到目前为止见过的**最大 logit**；
- $\ell$：到目前为止的**指数和**(以当前 $m$ 为基准)；
- $O$：到目前为止的**未归一化输出累加**(即 $\sum e^{x_j-m}\,v_j$)。

来了一个新块，块内最大值 $m^{\text{new}}$、块内指数和 $\ell^{\text{blk}}=\sum_{j\in\text{blk}}e^{x_j-m^{\text{new}}}$、块内加权值 $O^{\text{blk}}=\sum_{j\in\text{blk}}e^{x_j-m^{\text{new}}}v_j$。

**合并(merge)公式**——把旧状态和新块对齐到统一的全局最大值 $m'=\max(m,m^{\text{new}})$：

$$m' = \max(m,\ m^{\text{new}})$$
$$\ell' = e^{m-m'}\,\ell \;+\; e^{m^{\text{new}}-m'}\,\ell^{\text{blk}}$$
$$O' = e^{m-m'}\,O \;+\; e^{m^{\text{new}}-m'}\,O^{\text{blk}}$$

直觉：旧累加是以旧 $m$ 为基准的，新基准变了，就乘一个**修正因子** $e^{m-m'}$ 把它"降档"到新基准；新块同理乘 $e^{m^{\text{new}}-m'}$。两边对齐后才能相加。

最后**全部块扫完**，再做一次归一化：
$$O_{\text{final}} = O' / \ell'$$

```
   running:  m=-∞, ℓ=0, O=0
   ┌───── 块到来 ─────┐
   │ 算块内 m_new, ℓ_blk, O_blk
   │ m'  = max(m, m_new)
   │ α   = e^(m - m')      (旧的修正因子, ≤1)
   │ β   = e^(m_new - m')  (新的修正因子, ≤1)
   │ ℓ  ← α·ℓ + β·ℓ_blk
   │ O  ← α·O + β·O_blk        ← 关键: O 也跟着 rescale!
   │ m  ← m'
   └──────────────────┘  ... 下一块
   结束:  O ← O / ℓ
```

**没有任何近似**：合并完全等价于一次性对整行做 safe softmax。这是 FA 能号称 *exact attention* 的数学根基。

---

## 5. Tiling：把 Q/K/V 切块，双层循环流式扫

把 $Q$ 切成 $T_r$ 个行块(每块 $B_r$ 行)，$K,V$ 切成 $T_c$ 个列块(每块 $B_c$ 行)。块大小选得让 $Q,K,V,O$ 的一个块都能塞进 **SRAM**。

FlashAttention-1 的循环结构(**外层遍历 K/V 块，内层遍历 Q 块**)：

```
 for j in K/V 列块 (j = 1..Tc):          ← 外循环
     从 HBM 载入 Kj, Vj 到 SRAM
     for i in Q 行块 (i = 1..Tr):        ← 内循环
         从 HBM 载入 Qi, 以及该行块的运行量 mi, ℓi, Oi
         Sij = Qi · Kjᵀ / √d             (Br×Bc, 只在 SRAM)
         m_new = rowmax(Sij)
         P̃ij  = exp(Sij - m_new)        (块内未归一化概率)
         ℓ_blk = rowsum(P̃ij)
         O_blk = P̃ij · Vj
         —— online softmax 合并(见第4节)——
         mi' = max(mi, m_new)
         ℓi  = e^(mi-mi')·ℓi + e^(m_new-mi')·ℓ_blk
         Oi  = e^(mi-mi')·Oi + e^(m_new-mi')·O_blk
         mi  = mi'
         写回 mi, ℓi, Oi 到 HBM
 最后: Oi ← Oi / ℓi   写回 HBM
```

数据流 ASCII：

```
        HBM                          SRAM(片上)                  HBM
   ┌──────────┐   载入块    ┌─────────────────────────┐   写回   ┌──────┐
   │ Q (N×d)  │──Qi────────▶│  Sij=Qi·Kjᵀ  (Br×Bc)    │         │      │
   │ K (N×d)  │──Kj────────▶│  P̃ij=exp(Sij-m)         │──Oi────▶│  O   │
   │ V (N×d)  │──Vj────────▶│  online softmax 累加 Oi  │         │(N×d) │
   └──────────┘             └─────────────────────────┘         └──────┘
       └─ N² 矩阵从不整体出现在 HBM；只有 Br×Bc 小块在 SRAM 闪现 ─┘
```

**为什么省 HBM**：$S/P$ 这两个 $N^2$ 矩阵从来不写 HBM；HBM 只承担 $Q,K,V,O$(各 $N\times d$)的若干次读写。HBM 流量从 $O(N^2)$ 降到约 $O(N^2 d / M)$($M$=SRAM 容量)，但**关键是去掉了那条 $N^2$ 的硬往返**，实测访存量降 5–20 倍。

> 因果掩码(causal mask)可以让位于对角线右上方(未来 token)的整个 K/V 块**直接跳过不算**，进一步省一半计算。

---

## 6. Recompute：反向传播不存 $N^2$ 矩阵

训练需要反向。标准反向需要前向的注意力概率 $P$(用于求 $dQ,dK,dV$)，但 $P$ 是 $N\times N$——存它就把前向省下的显存又吐回去了。

**FlashAttention 的做法**：前向**只保存**每行的 $m_i$ 和 $\ell_i$(两个长度 $N$ 的向量，$O(N)$ 而非 $O(N^2)$)。反向时**用 $Q,K,V$ 和保存的 $(m,\ell)$ 现场重算** $S_{ij}=Q_iK_j^\top$、$P_{ij}=\exp(S_{ij}-m_i)/\ell_i$，算完梯度就丢。

```
   前向只额外存:  m (N,)  +  ℓ (N,)   →  O(N) 显存
   反向需要 P 时:  现场 重算 Sij、Pij(块级)  →  多花算力, 省显存
   ─────────────────────────────────────────────
   这是经典的 "recompute / gradient checkpointing" 思想:
        用 FLOPs 换 HBM 带宽和显存
```

由于反向本来就是 memory-bound，多算的这点 FLOPs 几乎"免费"，而省下的 $N^2$ 显存和访存让整体**更快又更省**。

---

## 7. FlashAttention-2：循环换序 + 更好的并行/工作划分

FA-1 已经很好，但还有三个低效点，FA-2 逐一修掉：

1. **循环顺序对调**：FA-2 把 **Q 行块放外层、K/V 块放内层**。这样每个 Q 行块的运行量 $(m,\ell,O)$ 一直**驻留在寄存器/SRAM**，整行扫完才写一次 HBM，避免 FA-1 中反复读写运行量。

```
   FA-1:  for KV块:  for Q块:  ...   (运行量随 Q 块反复进出)
   FA-2:  for Q块:   for KV块: ...   (运行量常驻, 内层累加, 末尾一次写回)
```

2. **减少非矩阵乘(non-matmul)运算**：GPU 的 Tensor Core 做矩阵乘极快，但 `exp`、逐元素 rescale 这类**非 matmul** 操作吞吐低得多。FA-2 把每个内层迭代里的 rescale 推迟，**只在行块结束时除以一次 $\ell$**，省掉大量逐块除法。

3. **更好的并行维度**：FA-2 额外在 **序列长度维度** 上切分并行(不仅 batch×head)，让长序列、小 batch 时 GPU 的 SM 也能吃满；并优化 warp 间的工作划分，减少 shared memory 通信。

结果：FA-2 相对 FA-1 大约**再快 ~2×**，在 A100 上达到接近理论峰值的利用率。

```
   利用率(占峰值)   FA-1 ████████░░░░░░  ~25-40%
                    FA-2 ███████████████ ~50-73%
```

---

## 8. Flash-Decoding：推理解码期沿 KV 维度并行

参考：https://crfm.stanford.edu/2023/10/12/flashdecoding.html

**场景不同**：训练/prefill 时 $Q$ 有很多行(长序列)，并行度天然够。但**自回归解码(decoding)** 每步只生成 1 个 token → $Q$ 只有 **1 行**，而 KV cache 可能有几万行。此时按 Q 切块只有 1 个块 → **GPU 大量 SM 闲置**，并行度严重不足。

Flash-Decoding 的招法：**沿 KV 序列维度把 K/V 切成若干 split，分给不同 SM 并行算**，每个 split 各自算出一个**局部** online-softmax 部分结果 $(O_s, m_s, \ell_s)$，最后用第 4 节的合并公式把所有 split **归约(reduce)** 成最终输出。

```
   解码: Q 只有 1 行, KV cache 很长 ─┐
                                       ▼
   KV cache  [════════════════════════════════]  长度 N
   split:    [ s1 ][ s2 ][ s3 ][ s4 ] ...  并行分给多个 SM
              │     │     │     │
              ▼     ▼     ▼     ▼
            (O1,m1,ℓ1)(O2,..)(O3,..)(O4,..)   ← 各 split 局部结果
                       │
                  online-softmax 合并(rescale + 相加)
                       ▼
                  最终 O (1×d)
```

效果：长序列生成时把闲置 SM 利用起来，注意力部分**最高 ~8× 加速**。它是 vLLM/TensorRT-LLM 等推理引擎长上下文解码的关键。
> 配套阅读 [[llm-inference/KV-Cache优化]]、[[llm-optimizer/kv-cache]]。

---

## 9. 数值手算：完整跑一遍 Online Softmax

取一行 logits（已除 $\sqrt d$）：$x=(1,\ 3,\ 2,\ 5)$，对应 value 标量 $v=(10,\ 20,\ 30,\ 40)$（为了手算，把 $V$ 设成标量；真实是 $d$ 维向量，逐分量同理）。
**目标**：$O=\sum_i \mathrm{softmax}(x)_i\, v_i$。把它切成两块 `块A=(1,3)`、`块B=(2,5)`，用 online softmax 算，并与一次性 safe-softmax 对账。

**块 A** $(x=1,3;\ v=10,20)$：
- $m_A=\max(1,3)=3$
- $e^{1-3}=e^{-2}=0.1353,\quad e^{3-3}=e^{0}=1$
- $\ell_A = 0.1353+1 = 1.1353$
- $O_A = 0.1353\cdot10 + 1\cdot20 = 1.353 + 20 = 21.353$

初始化 running：$m=3,\ \ell=1.1353,\ O=21.353$。

**块 B** $(x=2,5;\ v=30,40)$：
- $m_B=\max(2,5)=5$
- $e^{2-5}=e^{-3}=0.0498,\quad e^{5-5}=1$
- $\ell_B = 0.0498+1 = 1.0498$
- $O_B = 0.0498\cdot30 + 1\cdot40 = 1.494 + 40 = 41.494$

**合并**：$m'=\max(3,5)=5$
- 旧修正因子 $\alpha=e^{m-m'}=e^{3-5}=e^{-2}=0.1353$
- 新修正因子 $\beta=e^{m_B-m'}=e^{5-5}=1$
- $\ell' = 0.1353\cdot1.1353 + 1\cdot1.0498 = 0.1536 + 1.0498 = 1.2034$
- $O' = 0.1353\cdot21.353 + 1\cdot41.494 = 2.889 + 41.494 = 44.383$

**归一化**：$O_{\text{final}} = O'/\ell' = 44.383 / 1.2034 = 36.88$

**对账(一次性 safe softmax)**：全局 $m=5$，
$e^{1-5}=0.0183,\ e^{3-5}=0.1353,\ e^{2-5}=0.0498,\ e^{5-5}=1$，和 $=1.2034$。
权重 $=(0.0152,0.1124,0.0414,0.8310)$，
$O=0.0152\cdot10+0.1124\cdot20+0.0414\cdot30+0.8310\cdot40 = 0.152+2.248+1.242+33.24 = 36.88$ ✅

**完全一致**——这就是 "exact"：online softmax 不是近似，而是**严格等价**的重排。

---

## 10. 复杂度 / 显存账本

| 量 | 标准注意力 | FlashAttention |
|----|-----------|----------------|
| 计算 FLOPs | $O(N^2 d)$ | $O(N^2 d)$（**相同**，不省算力） |
| **HBM 访存** | $O(N^2)$ | $\approx O(N^2 d / M)$，常数极小 |
| **激活显存** | $O(N^2)$（存 $S,P$） | $O(N)$（只存 $m,\ell$） |
| 反向所需 | 存 $P$（$N^2$） | 重算（存 $m,\ell$，$O(N)$） |
| 精确性 | 精确 | **精确**（非近似） |

**显存手算**（$N=8192$，单 head，fp16）：
- 标准存 $S$+$P$：$2\times N^2\times 2\text{B}=2\times8192^2\times2 = 256\text{MB}$（仅一个 head！多 head/batch 直接爆）
- FA 存 $m,\ell$：$2\times N\times 4\text{B}=2\times8192\times4 = 64\text{KB}$

**省了约 4000 倍的中间激活显存**。这就是为什么超长上下文（32k/128k）训练几乎必须用 FlashAttention。

```
   激活显存随 N 增长:
   标准  ▁▂▃▅█  ← N² 抛物线, 很快撑爆 80GB
   FA    ▁▁▁▁▁  ← N 线性, 几乎贴地
                 N→
```

---

## 常见问题

| 问题 | 回答 |
|------|------|
| FA 是近似算法吗？ | **不是**。output 与标准注意力逐位等价（浮点误差级别），所以叫 *exact attention*。 |
| FA 减少了 FLOPs 吗？ | **没有**。计算量仍是 $O(N^2d)$。它减少的是 **HBM 访存** 和 **激活显存**，把 memory-bound 变 compute-bound。 |
| 为什么 online softmax 不会数值溢出？ | 每块都减去当前块最大值再取 exp，合并时按全局最大值 rescale，指数恒 $\le 1$。 |
| 为什么反向要重算？ | 不存 $N^2$ 的 $P$ 才能保持 $O(N)$ 显存；反向本就 memory-bound，重算的 FLOPs 几乎免费。 |
| FA-2 比 FA-1 快在哪？ | 循环换序使运行量常驻、减少非 matmul 运算、增加序列维并行。约再快 2×。 |
| Flash-Decoding 解决什么？ | 解码每步 $Q$ 只有 1 行 → 并行度低；沿 KV 维切 split 并行再归约，长序列最高 ~8×。 |
| 块大小怎么定？ | 让 $Q_i,K_j,V_j,O_i$ 块同时塞进 SRAM；受 SRAM 容量 $M$ 约束，是访存与并行的折中。 |
| 和 KV-Cache 量化/Paged 冲突吗？ | 不冲突，正交。FA 管"怎么算注意力"，KV-Cache 管"怎么存 K/V"，常组合使用。 |

---

## 🔗 跳转链接

- 总图：[[00-知识地图]]
- 注意力机制本体：[[llm-algo/transformer/模型架构]]
- 富笔记(优化视角)：[[llm-optimizer/FlashAttention]]
- KV 缓存与解码：[[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/解码策略]]
- 计算量基础：[[llm-algo/FLOPs]]
- 计算/通信重叠：[[llm-optimizer/计算通信重叠]]
- GPU 与 CUDA 底层：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]]
- 数值精度(fp8/量化)：[[llm-compression/quantization/fp8]] · [[llm-compression/quantization/量化基础]]
- 训练框架中的注意力：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]
- 推理总览：[[llm-inference/README]] · [[llm-inference/大模型推理张量并行]]

> 参考文献：
> - FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
> - FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
> - Online normalizer calculation for softmax (arXiv:1805.02867)
> - Flash-Decoding (Stanford CRFM, 2023)
> - 代码：https://github.com/Dao-AILab/flash-attention （版本/接口以官方为准）
