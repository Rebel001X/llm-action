# FLOPs / 参数量 / 计算量估算

> 给定一个 Transformer 配置，用纸笔在 60 秒内估出它的「参数量、训练算力、推理显存」三件套。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/transformer/模型架构]] [[llm-optimizer/kv-cache]] [[ai-infra/算力/GPU工作原理]]

## 阅读地图

| 节 | 你会得到什么 | 一句话结论 |
|---|---|---|
| 0 | 三个魔法数 | 参数 `12·L·d²`，前向 `2N`，训练 `6N` |
| 1 | 地基：FLOP / MAC / 矩阵乘 | 一次乘加 = 2 FLOPs，是一切的原子 |
| 2 | 参数量逐项推导 | Attention `4d²` + FFN `8d²` = `12d²` 每层 |
| 3 | 为什么前向 ≈ `2N` | 每个权重参与一次乘 + 一次加 |
| 4 | 为什么训练 ≈ `6N` | 前向 `2N` + 反向 `4N` |
| 5 | 注意力的额外项 | `seq²·d` 项，长序列才显著 |
| 6 | KV Cache 显存公式 | `2·L·b·s·d·dtype` |
| 7 | 一次完整手算 | 用 GPT-3 13B 全程算到底 |
| 8 | MFU / HFU | 实测算力 ÷ 理论峰值 |
| 表 | 复杂度对照 / 常见问题 | 速查 |

## 0. 一句话锚点

把一个 Transformer 的规模浓缩成三个数，记住它们，面试和容量规划都够用：

```
参数量   N ≈ 12 · L · d²          (L=层数, d=hidden_size)
前向FLOPs    ≈ 2 · N · tokens     (每参数 1 乘 1 加)
训练FLOPs    ≈ 6 · N · tokens     (前向2 + 反向4)
KV显存   = 2 · L · b · s · d · bytes  (K 和 V 各一份)
```

下面把每一个数字「为什么是这个系数」从最底层拆开。

## 1. 地基：FLOP 到底是什么

**FLOP** = Floating Point Operation = 一次浮点运算。一次浮点**加法**算 1 FLOP，一次浮点**乘法**也算 1 FLOP。

注意区分两个易混词：
- **FLOPs**(小写 s，复数)：运算的**总次数**，是个标量数量。本文讲的都是这个。
- **FLOPS / FLOP/s**(带 /s)：每秒运算次数，是**速率**，是 GPU 的性能指标（如 A100 BF16 ≈ 312 TFLOP/s）。

**为什么深度学习里「乘加」成对出现**：神经网络的核心是矩阵乘法，矩阵乘法的内层永远是「乘了再累加」(`acc += a*b`)。硬件甚至专门做了一条 **FMA**(Fused Multiply-Add) 指令把它俩合一。但计 FLOPs 时我们仍按 **2 FLOPs** 计（1 乘 + 1 加）。这就是后面所有「2」的来源。

**一次矩阵乘的 FLOPs**：设 $A$ 是 $m\times k$，$B$ 是 $k\times n$，输出 $C$ 是 $m\times n$。

```
        k                 n                    n
     ┌──────┐         ┌──────┐            ┌──────┐
  m  │  A   │   ×   k │  B   │    =     m │  C   │
     └──────┘         └──────┘            └──────┘

 C 有 m·n 个元素，每个元素 = k 次乘 + (k-1) 次加 ≈ k 次乘加
 总乘加 = m·n·k  →  总 FLOPs = 2·m·n·k
```

$$\text{FLOPs}_{\text{matmul}} = 2 \cdot m \cdot n \cdot k$$

**逐数验证**：$A$ 为 $2\times3$，$B$ 为 $3\times4$。输出 $2\times4=8$ 个元素，每个元素点积长度 3（3 乘 2 加）。乘加次数 $=2\cdot4\cdot3=24$，FLOPs $=48$。手数一遍：$C_{11}=a_{11}b_{11}+a_{12}b_{21}+a_{13}b_{31}$，3 乘 2 加 = 5 次运算，8 个元素 ≈ 40，加上凑整的口径差异，工程上一律用 $2mnk=48$。

这个 `2mnk` 是本文唯一需要背的「原子公式」。

## 2. 参数量：逐项数清每一个权重

先约定符号（这套符号贯穿全文）：

| 符号 | 含义 | GPT-3 13B 取值 |
|---|---|---|
| $L$ | Transformer 层数 | 40 |
| $d$ | 隐藏维度 hidden_size | 5120 |
| $h$ | 注意力头数 | 40 |
| $d_{ff}$ | FFN 中间维度，通常 $=4d$ | 20480 |
| $V$ | 词表大小 vocab | 50257 |
| $s$ | 序列长度 | 2048 |
| $b$ | batch size | — |

一层 Transformer 的结构与参数分布：

```
            输入 x  (维度 d)
              │
   ┌──────────┴───────────┐
   │   Multi-Head Attn     │
   │  Wq Wk Wv Wo 各 d×d   │  ← 4·d² 参数
   └──────────┬───────────┘
              │ + 残差
   ┌──────────┴───────────┐
   │        FFN            │
   │  W1: d×4d  W2: 4d×d   │  ← 4d² + 4d² = 8·d² 参数
   └──────────┬───────────┘
              │ + 残差
            输出 (维度 d)

   每层合计 ≈ 4d² + 8d² = 12·d²
```

**(a) 注意力部分**。Multi-Head Attention 有 4 个投影矩阵：$W_Q, W_K, W_V$ 把输入投到 Q/K/V，$W_O$ 把多头拼接后投回。**关键洞察**：多头只是把维度 $d$ 切成 $h$ 份各算，总维度不变，所以每个矩阵都是 $d\times d$。

$$P_{\text{attn}} = \underbrace{d^2}_{W_Q} + \underbrace{d^2}_{W_K} + \underbrace{d^2}_{W_V} + \underbrace{d^2}_{W_O} = 4d^2$$

**(b) FFN 部分**。两层全连接：先升维 $d\to 4d$，再降维 $4d\to d$。

$$P_{\text{ffn}} = \underbrace{d\cdot 4d}_{W_1} + \underbrace{4d\cdot d}_{W_2} = 8d^2$$

**(c) 每层合计**：$4d^2 + 8d^2 = 12d^2$。（偏置、LayerNorm 的 $\gamma/\beta$ 都是 $O(d)$ 量级，相对 $d^2$ 可忽略。）

**(d) 全部 L 层 + Embedding**：

$$\boxed{N \approx 12 \cdot L \cdot d^2 \;+\; V\cdot d}$$

其中 $V\cdot d$ 是词嵌入表（输入 embedding 与输出 LM head 常常**共享权重 / tied**，故只算一份）。当模型很大时 $12Ld^2 \gg Vd$，可直接用 $N\approx 12Ld^2$。

**逐数手算（GPT-3 13B）**：
```
每层  = 12 · d²        = 12 · 5120²      = 12 · 26,214,400 ≈ 3.146 亿
40 层 = 40 · 3.146 亿  ≈ 125.8 亿
Embed = V · d = 50257 · 5120 ≈ 2.57 亿
合计  ≈ 125.8 + 2.6 ≈ 128.4 亿 ≈ 12.8 B  ✓ 与官方 13B 吻合
```
偏差几亿来自我们忽略的 bias/LN 和实际 $d_{ff}$ 微调，量级完全正确。

## 3. 为什么前向 ≈ 2N：每个权重「来访一次」

设处理 `T` 个 token。前向传播时，每个 token 流过网络，会和**每一个权重**做一次乘加。

**直觉推导**：一个权重矩阵 $W$ 形状 $a\times b$（含 $ab$ 个参数），输入 $T$ 个 token（形状 $T\times a$），输出 $T\times b$。这是矩阵乘 $[T,a]\times[a,b]$，FLOPs $= 2\cdot T\cdot a\cdot b = 2T\cdot(\text{该矩阵参数量})$。

把所有权重矩阵加起来：

$$\text{前向 FLOPs} = \sum_W 2T\cdot P_W = 2T\sum_W P_W = 2\cdot T\cdot N$$

```
 含义图：
   每个权重 w  ──被一个token访问──>  1次乘 + 1次加 = 2 FLOPs
   T 个token × N 个权重 × 2     =   2·T·N
```

$$\boxed{C_{\text{forward}} \approx 2 N T}$$

> 注意：这里 $T$ 是 token 总数（= batch × seq）。这个 `2` 和第 1 节矩阵乘的 `2mnk` 是**同一个 2**——乘加配对。

## 4. 为什么训练 ≈ 6N：前向 2 + 反向 4

训练一步 = 前向 + 反向 + 更新。反向传播为什么是前向的 **2 倍**？

对每个权重矩阵，反向要算**两个**梯度：
1. **对输入的梯度** $\partial L/\partial x$（要往前一层传，链式法则继续）——一次 matmul。
2. **对权重的梯度** $\partial L/\partial W$（用来更新这个权重）——一次 matmul。

```
   前向:   x ──W──> y                       1 次 matmul  (2NT)
   反向:   ┌ dL/dx  (传给上一层)            1 次 matmul
           └ dL/dW  (更新本权重)            1 次 matmul   ← 反向 2 次
   ──────────────────────────────────────
   前向:反向 = 1 : 2  →  总 = 3 倍前向 = 6NT
```

$$\boxed{C_{\text{train}} \approx C_{fwd} + 2C_{fwd} = 3\cdot 2NT = 6 N T}$$

这就是 OpenAI/DeepMind 论文里反复出现的 **`C ≈ 6ND`**（$D$ = 训练 token 数）的由来。

**激活重计算（gradient checkpointing）的修正**：为省显存，反向时常**重新前向**一遍以重建激活值，于是多出 1 份前向：

$$C_{\text{train+recompute}} \approx (1_{fwd} + 2_{bwd} + 1_{recompute})\cdot 2NT = 8NT$$

即每 token 约 **8N**（常写成「前向1+反向2+重算1=4 倍前向」）。是否开启是「显存 ↔ 算力」的权衡，详见第 8 节 HFU。

**逐数手算（13B 模型，训练 300B token）**：
$$C = 6 \cdot 12.8\times10^9 \cdot 300\times10^9 = 2.3\times10^{22}\ \text{FLOPs}$$
一张 A100 实测 ~150 TFLOP/s（MFU≈50%），则
$$\frac{2.3\times10^{22}}{1.5\times10^{14}} \approx 1.5\times10^8\ \text{s} \approx 1780\ \text{GPU·天} \approx 64\ \text{张卡跑 28 天}.$$

## 5. 注意力的「隐藏项」：与序列长度平方相关

第 2~4 节把模型当成「一堆权重矩阵」，但注意力里还有**两步没有权重的矩阵乘**：

```
   QKᵀ:  [s, d] × [d, s] → [s, s]      FLOPs = 2·s·s·d
   分数·V: [s, s] × [s, d] → [s, d]    FLOPs = 2·s·s·d
   ───────────────────────────────────
   每层每序列 ≈ 4·s²·d   (含所有头, 合并后 d 不变)
```

每层每个序列注意力额外 FLOPs $\approx 4 s^2 d$（前向）。把它和权重项放一起比较（单序列、单层、前向）：

$$\frac{\text{注意力 } 4s^2 d}{\text{权重 } 2\cdot s\cdot 12d^2} = \frac{4s^2d}{24sd^2} = \frac{s}{6d}$$

**结论**：当 $s \ll 6d$ 时（如 $s=2048, d=5120 \Rightarrow 6d=30720$），注意力的平方项**可忽略**，`6N` 公式成立。但当**长上下文** $s \to 32k, 128k$ 且 $d$ 不大时，$s^2$ 项主导，必须单独加上——这也是 FlashAttention、稀疏注意力要优化的部分。

## 6. KV Cache 显存：推理为什么越聊越占显存

**为什么需要 KV Cache**：自回归生成时，每生成 1 个新 token 都要对**之前所有 token**做注意力。若不缓存，每步都要重算全部 K、V，复杂度 $O(s^2)$；缓存后每步只算新 token 的 K、V，把历史的 K、V 存下来复用，把每步降到 $O(s)$。代价是**显存**。

**显存逐项推导**：每一层都要存 K 和 V 两份张量，每份形状 $[b, s, d]$：

```
   一层:  K 张量 [b,s,d]  +  V 张量 [b,s,d]   = 2·b·s·d 个元素
   L 层:  乘 L                                = 2·L·b·s·d 个元素
   字节:  乘 dtype_bytes (fp16=2, fp8=1)
```

$$\boxed{M_{\text{KV}} = 2 \cdot L \cdot b \cdot s \cdot d \cdot \text{bytes}}$$

（用 GQA/MQA 时 K、V 头数从 $h$ 降到 $g$，公式中的 $d$ 要乘 $g/h$，显存按比例下降——这正是 GQA 省显存的根本原因，见 [[llm-optimizer/kv-cache]]。）

**逐数手算（13B，fp16，b=1，s=2048）**：
```
M = 2 · 40 · 1 · 2048 · 5120 · 2 bytes
  = 2 · 40 · 2048 · 5120 · 2
  = 1,677,721,600 bytes
  ≈ 1.68 GB        (单条序列就 1.68GB!)
```
再看长上下文 `s=32768`：线性放大 16 倍 ≈ **26.8 GB**，已逼近一张卡的容量——这就是长上下文推理的显存墙，也是各种 KV 量化 / 驱逐策略的战场。

**和参数显存对比**：13B 权重本身 fp16 占 $12.8\times10^9\times2 \approx 25.6$ GB。所以「权重 25.6GB + KV 1.68GB×并发数」，并发一上来 KV 就成为主要变量。

## 7. 完整手算示例：从配置到三件套

把前面所有公式串成一条龙。**配置**：$L=40, d=5120, V=50257, s=2048$，fp16。

**① 参数量**
```
12·L·d² = 12·40·5120² = 480·26,214,400 ≈ 1.258×10¹⁰
+ V·d   = 50257·5120  ≈ 2.57×10⁸
N ≈ 1.284×10¹⁰  ≈ 12.8 B
```

**② 单次前向 FLOPs（处理一个 batch，T = b·s，设 b=1）**
```
C_fwd = 2·N·T = 2·1.284×10¹⁰·2048 ≈ 5.26×10¹³ FLOPs ≈ 52.6 TFLOPs
（再加注意力项 4·L·s²·d = 4·40·2048²·5120 ≈ 3.4×10¹² ≈ 3.4 TFLOPs，约 6%，长序列才显著）
```

**③ 训练一步 FLOPs（同 batch）**
```
C_train = 6·N·T = 3 × 52.6 T ≈ 157.9 TFLOPs / step
A100 312TFLOP/s 峰值, MFU 50% → 156TFLOP/s 实测
单步耗时 ≈ 157.9 / 156 ≈ 1.0 s/step  (b=1; 实际 b 越大越摊薄开销)
```

**④ 推理 KV 显存（b=1, s=2048, fp16）**
```
M_KV = 2·40·1·2048·5120·2 ≈ 1.68 GB
权重显存 = N·2 bytes ≈ 25.6 GB
总显存 ≈ 25.6 + 1.68 + 激活/碎片 ≈ 28~30 GB  → 单张 A100 40G 可推
```

一张配置卡，60 秒算完三件套。

## 8. MFU：你的算力真的用上了吗

理论 FLOPs 和实测性能之间有巨大鸿沟，用 **MFU**(Model FLOPs Utilization) 衡量：

$$\text{MFU} = \frac{\text{模型有效算力(6ND/实际秒数)}}{\text{硬件峰值算力(GPU数 × 单卡峰值)}}$$

- **MFU**：分子只算「模型本身需要的」`6ND`，**不含**重计算等额外开销。是衡量「端到端训练效率」的金标准。
- **HFU**(Hardware FLOPs Utilization)：分子把重计算等也算进去（用 `8ND`）。HFU 总是 ≥ MFU，反映「硬件被喂了多少活」，但不代表都是有用功。

```
   理论峰值 ████████████████████  100%   (A100: 312 TFLOP/s)
   HFU      ████████████░░░░░░░░   ~60%   (含重计算的硬件利用)
   MFU      ██████████░░░░░░░░░░   ~50%   (纯模型有效算力)
            └─ 差距来自: 访存带宽墙 / 通信 / kernel启动 / pipeline气泡
```

**逐数算 MFU**：上例单步 `6ND = 157.9 TFLOPs`，实测 1.0 s/step（单卡）：
$$\text{MFU} = \frac{157.9\times10^{12}}{1.0\times 312\times10^{12}} \approx 50.6\%$$
**经验值**：大模型训练 MFU 通常 35%~55% 算健康；低于 30% 说明被通信或访存卡住（参见 [[ai-infra/算力/GPU工作原理]] 的 roofline 模型）。提升手段：更大 batch、张量并行通信重叠、FlashAttention 降访存、用 BF16/FP8。

## 复杂度与显存对照表

| 量 | 公式 | 主导项 | 备注 |
|---|---|---|---|
| 参数量 | $12Ld^2 + Vd$ | $12Ld^2$ | 大模型下 embedding 可忽略 |
| 前向 FLOPs | $2NT$ | 线性于 N、token | 每权重 1 乘 1 加 |
| 训练 FLOPs | $6NT$ | 前向×3 | 反向 = 前向×2 |
| 训练+重算 | $8NT$ | 前向×4 | gradient checkpointing |
| 注意力 FLOPs | $4Ls^2d$/前向 | $s^2$ | 长序列才显著 |
| 权重显存 | $N\cdot \text{bytes}$ | — | fp16=2, fp32=4 |
| KV 显存 | $2Lbsd\cdot\text{bytes}$ | 线性于 s、b | GQA 按头数比例降 |
| 优化器状态(Adam) | $\sim 2N$~$6N$·bytes | — | 一阶+二阶矩+主参数 |

## 常见问题

| 疑问 | 真相 |
|---|---|
| 为什么是 `2N` 不是 `N`？ | 矩阵乘内层是「乘 + 加」成对，1 次乘加 = 2 FLOPs |
| 为什么训练是前向的 3 倍？ | 反向要算「对输入的梯度」和「对权重的梯度」两次 matmul，2 倍前向，+前向本身=3 倍 |
| `6N` 里要不要算 embedding？ | embedding 查表是访存不是计算，几乎不产生 FLOPs，但参数量要算 |
| 为什么忽略 softmax/LayerNorm？ | 它们是 $O(d)$ 或 $O(s\cdot d)$，相对矩阵乘 $O(d^2)$/$O(s^2 d)$ 量级低，工程估算可忽略 |
| KV Cache 为啥不含 Q？ | Q 只用于当前 token 的一次性计算，无需跨步缓存；K、V 要被未来所有 token 复用 |
| MFU 为啥到不了 100%？ | 受访存带宽、通信、kernel 启动、流水线气泡限制，是显存墙/通信墙的体现 |
| FLOPs 和 FLOPS 区别？ | 小写复数 FLOPs=运算总次数；带 /s 的 FLOP/s=每秒速率（硬件性能） |
| `d_ff` 不是 4d 怎么办？ | FFN 项改为 $2\cdot d\cdot d_{ff}$，每层参数 = $4d^2 + 2d\cdot d_{ff}$；如 SwiGLU 有 3 个矩阵要单独数 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，回到主干
- [[llm-algo/transformer/模型架构]] — Attention/FFN 的结构细节，本文参数推导的依据
- [[llm-optimizer/kv-cache]] — KV Cache 的工程实现、GQA/MQA、量化与驱逐
- [[ai-infra/算力/GPU工作原理]] — 峰值算力、roofline、访存墙，理解 MFU 上不去的根因

---
> 参考：backward/forward FLOP ratio (epochai.org/blog/backward-forward-FLOP-ratio)；激活重计算 (arxiv.org/pdf/2205.05198)；最简 LLM FLOPs 估算 (zhuanlan.zhihu.com/p/652697200)。
