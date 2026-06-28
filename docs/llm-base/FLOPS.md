# FLOPs 与大模型计算量估算

> 一句话定位：把"一次前向/反向/训练要做多少次浮点乘加"算清楚，是估训练时长、选硬件、定 batch、做 MFU 评估的地基。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/FLOPs]] · [[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[ai-infra/算力/GPU工作原理]] · [[ai-framework/megatron-lm/README]]

## 阅读地图

| 节 | 你将搞清楚的问题 | 关键产物 |
| --- | --- | --- |
| 0 | FLOPs / FLOPS 到底差在哪 | 锚点定义 |
| 1 | 一次乘加算几个 FLOP？矩阵乘多少 FLOP | $2mnk$ 公式 |
| 2 | 算力的"理论峰值"怎么读 | TFLOPS 表 |
| 3 | 一层 Transformer 拆成几个矩阵乘 | 逐算子 FLOP |
| 4 | 训练总量经验公式 $C\approx 6ND$ 怎么来 | 6N×token |
| 5 | 前向/反向为何是 1:2 | 反向链式法则 |
| 6 | Attention 的 $O(s^2)$ 项何时不可忽略 | 长序列修正 |
| 7 | MFU / HFU：算力利用率怎么算 | 利用率公式 |
| 8 | MoE / 推理 decode 的 FLOP 区别 | 激活参数、KV |
| 手算 | GPT-3 175B 端到端逐数算一遍 | 训练天数 |
| FAQ | 易错点速查 | 表格 |

---

## 0. 一句话锚点

- **FLOP**（小写 s 是复数）= **Floating Point Operation**，一次浮点运算（一次加法或一次乘法各算 1 个 FLOP）。它衡量**工作量**（要做多少次运算）。
- **FLOPS / FLOP/s**（大写 S 或带 /s）= **Floating Point Operations per Second**，每秒浮点运算次数。它衡量**速度**（硬件多快）。
- 二者关系，像"路程"与"速度"：

$$\text{运行时间(s)} = \frac{\text{任务总 FLOPs}}{\text{硬件有效 FLOP/s}}$$

```
   工作量 (FLOPs)            速度 (FLOP/s)              时间 (s)
 ┌──────────────┐         ┌──────────────┐        ┌──────────┐
 │ 模型要算多少 │   ÷     │ GPU 每秒能算 │   =    │  跑多久  │
 │  次浮点运算  │ ──────► │  多少次运算  │ ─────► │          │
 └──────────────┘         └──────────────┘        └──────────┘
        ▲                        ▲
   本文 §1~§6               本文 §2（峰值）
                            ×MFU（§7，实际打折）
```

> 记忆口诀：**FLOPs 是名词（多少活），FLOPS 是速率（多快）**。本文除明确写"算力/峰值"外，谈"模型多大"一律指 FLOPs（工作量）。

---

## 1. 地基：一次乘加 = 2 FLOP，矩阵乘 = $2mnk$

神经网络几乎全部计算来自**矩阵乘法**（GEMM）。先把最小单元拆原子。

**乘加（MAC, Multiply-Accumulate）**：神经网络里最常见的动作是 $a \leftarrow a + w \times x$，即"乘一次、加一次"。所以：

$$1\ \text{MAC} = 1\ \text{乘} + 1\ \text{加} = 2\ \text{FLOP}$$

**矩阵乘的 FLOP**：设 $A \in \mathbb{R}^{m\times k}$，$B \in \mathbb{R}^{k\times n}$，输出 $C = AB \in \mathbb{R}^{m\times n}$。

- $C$ 有 $m\times n$ 个元素；
- 每个元素是两个长度为 $k$ 的向量做点积 → $k$ 次乘 + $(k{-}1)$ 次加 $\approx 2k$ FLOP；

$$\boxed{\text{FLOPs}(AB) = 2\,m\,n\,k}$$

这是全文最重要的一个公式。记住"**输出元素个数 × 收缩维度 × 2**"。

```
       k (收缩维度)                n
   ┌───────────────┐         ┌───────────┐
 m │       A       │   ×   k │     B     │   =   m │ C │ n
   └───────────────┘         └───────────┘         └───┘
                                                  每个 C[i][j]:
                                                  k 次乘 + k 次加 ≈ 2k FLOP
                              C 共 m·n 个元素  →  总 2·m·n·k FLOP
```

**带 batch**：若有 $b$ 个样本（batch 维度独立），总量再乘 $b$：$\text{FLOPs} = 2\,b\,m\,n\,k$。

**为什么忽略激活/LayerNorm/偏置？** 它们是逐元素（element-wise）操作，FLOP 量级为 $O(\text{元素数})$，相对 GEMM 的 $O(\text{元素数}\times k)$ 小 1~2 个数量级，估算时常并入误差。但**访存**上它们可能很贵（见 §7 与 [[llm-optimizer/FlashAttention]]）。

---

## 2. 算力峰值：读懂 TFLOPS 这张表

硬件标称算力 = **每秒能做多少 FLOP**，单位常用 **TFLOPS（$10^{12}$）/ PFLOPS（$10^{15}$）**。注意三件事：

1. **精度不同，峰值不同**：低精度（FP16/BF16/FP8）有专用 Tensor Core，峰值是 FP32 的数倍～数十倍。
2. **稀疏 vs 稠密**：厂商常标"含 2:4 结构化稀疏"的翻倍值，真实稠密训练要看**稠密（dense）**那一栏，别被翻倍数字骗。
3. **峰值≠实际**：实际只能拿到峰值的一部分（MFU，§7），通常训练 30%~55%。

```
 精度阶梯（越往下越快、越省显存，但数值范围/精度递减）
 ┌──────────────────────────────────────────────┐
 │ FP32  ── 基准，慢，做主权重/优化器状态        │
 │ TF32  ── Ampere 引入，范围同FP32、尾数同FP16  │
 │ FP16  ── 2× Tensor Core，需 loss scaling      │
 │ BF16  ── 范围同FP32、精度低，训练首选         │
 │ FP8   ── Hopper 起，再 2×，推理/部分训练      │
 │ INT8  ── 量化推理                              │
 └──────────────────────────────────────────────┘
```

> 典型量级（**以官方 datasheet 为准**，此处仅给数量级帮助估算）：A100 BF16 稠密约 **312 TFLOPS**；H100（SXM）BF16 稠密约 **~990 TFLOPS** 量级、FP8 再约翻倍。具体型号/功耗墙/散热都会影响，落地请查官方规格书。

精度细节见 [[机器学习中常用的数据类型]] 与 [[llm-compression/quantization/fp8]]。

---

## 3. 一层 Transformer：拆成几个矩阵乘

要数整模型 FLOP，先把**一个 Decoder Block** 拆到算子级。记号：

| 符号 | 含义 | 典型 |
| --- | --- | --- |
| $b$ | batch（句子数） | 1~ |
| $s$ | 序列长度 seq_len | 2048~ |
| $h$ | 隐藏维 hidden | 4096~ |
| $n_h$ | 注意力头数 | 32~ |
| $h_{ff}$ | FFN 中间维 | 通常 $4h$ |
| $L$ | 层数 | 32~ |
| $V$ | 词表大小 | 32k~ |

一个 Block 的主要 GEMM（按 token 总数 $T=b\cdot s$ 计）：

```
  输入 X [b,s,h]
     │
     ▼
 ┌──────────────── Self-Attention ────────────────┐
 │ ① QKV 投影:  X·W_qkv   [h → 3h]                 │   2·T·h·(3h) = 6 T h²
 │ ② 打分:      Q·Kᵀ      [s,h]×[h,s]              │   2·b·n_h·s²·(h/n_h)=2 T s h
 │ ③ 加权和:    A·V       [s,s]×[s,h]              │   2 T s h
 │ ④ 输出投影:  ·W_o      [h → h]                  │   2·T·h·h = 2 T h²
 └─────────────────────────────────────────────────┘
     │
     ▼
 ┌──────────────────── FFN/MLP ────────────────────┐
 │ ⑤ 升维:  ·W1  [h → 4h]                          │   2·T·h·(4h) = 8 T h²
 │ ⑥ 降维:  ·W2  [4h → h]                          │   2·T·(4h)·h = 8 T h²
 └─────────────────────────────────────────────────┘
```

把与参数相关的 GEMM（①④⑤⑥）加起来，每层前向：

$$\text{FFN+Proj} = (6+2+8+8)\,T h^2 = 24\,T h^2 \quad(\text{令 } h_{ff}=4h)$$

注意②③两项含 $s^2$，记作 attention 的"score/context"项，前向每层 $= 4\,T s h = 4\,b s^2 h$。

> 关键观察：**①④⑤⑥ 与序列长度 $s$ 线性**（藏在 $T=bs$ 里），**②③ 与 $s$ 平方**。短序列时 $h \gg s$，平方项可忽略；长序列时它会反超（§6）。

MLP 结构细节见 [[llm-algo/mlp]]，整体架构见 [[llm-algo/transformer/模型架构]]，位置编码不参与大 GEMM（见 [[llm-algo/旋转编码RoPE]]）。

---

## 4. 训练总量经验公式 $C \approx 6ND$

这是估训练成本最常用的"一行公式"，来自 OpenAI / Kaplan 缩放律与 Chinchilla 等工作：

$$\boxed{C \approx 6 \, N \, D}$$

- $C$：训练总 FLOPs；
- $N$：模型**非嵌入参数量**（参数个数）；
- $D$：训练 **token 总数**。

**为什么是 6？** 把"每个参数、每个 token"的代价拆开：

```
 每参数·每 token 的浮点运算 ≈ 6
 ┌────────────┬───────────────────────────┐
 │ 前向 forward      每参数 1 次 MAC = 2 FLOP │  ← §1
 │ 反向 backward     约前向的 2 倍   = 4 FLOP │  ← §5
 └────────────┴───────────────────────────┘
                                 合计 ≈ 6 FLOP / (参数·token)
```

直觉：§3 里每层"参数相关 GEMM"前向 $\approx 24Th^2$。而每层参数量约 $12h^2$（QKV$=3h^2$、O$=h^2$、FFN$=8h^2$）。于是"前向 FLOP / 参数" $= 24Th^2 /(12h^2)=2T$，即**每参数每 token 2 FLOP**；加上反向 ×3，得 **6**。

$$C \approx \underbrace{2ND}_{\text{前向}} + \underbrace{4ND}_{\text{反向}} = 6ND$$

> 适用范围：稠密 Transformer、$s \ll h$（平方项可略）、不含激活重计算的额外前向。开了**激活重计算/梯度检查点**（[[llm-train/README]]），反向要多一遍前向，常用 $C\approx 8ND$（即 1 前向 + 1 重算前向 + 2 反向 = 8）。

---

## 5. 前向 : 反向 = 1 : 2 为什么

反向传播对每个权重矩阵 $W$ 要算**两个**梯度，各是一次与前向同形状的 GEMM：

```
 前向:  Y = X · W                            ── 1 个 GEMM
 反向:  ∂L/∂X = ∂L/∂Y · Wᵀ   (传给上一层)    ── 1 个 GEMM
        ∂L/∂W = Xᵀ · ∂L/∂Y   (更新本层权重)  ── 1 个 GEMM
                                              ─────────────
                              反向共 2 个 ≈ 前向的 2 倍
```

每个 GEMM 的 FLOP 与前向同阶（$2mnk$ 量级），所以"反向 ≈ 2× 前向"。三者相加 → **训练 ≈ 3× 前向 FLOP**，正对应 $6ND = 3\times 2ND$。

> 链式法则细节、为何 $\partial L/\partial X$ 与 $\partial L/\partial W$ 都各一遍 GEMM，见 [[llm-algo/mlp]] 的反向推导。

---

## 6. Attention 的 $O(s^2)$ 项：长序列必须算

§3 的②③两项前向每层 $= 4bs^2h$。把全模型（$L$ 层）的两类前向 FLOP 写在一起：

$$\text{前向} \approx \underbrace{2ND}_{\text{参数项, } \propto s}\; +\; \underbrace{2\cdot L \cdot 2 b s^2 h \cdot (\text{batch 数})}_{\text{注意力项, } \propto s^2}$$

更实用的判据——**注意力项 / 参数项**的比值：

$$\frac{\text{attn 项}}{\text{参数项}} \;\approx\; \frac{s}{6h}\ (\text{量级})$$

```
   FLOP 占比随序列长度
   │
   │  参数项(线性)  ████████████████████  恒定主导
   │  注意力项(平方) ▁▂▃▅▇  随 s 快速上升
   └────────────────────────────────►  s
        s≪h: 可忽略     s~6h: 不可忽略    s≫h: 反客为主
```

- 例：$h=4096$ 时，$6h\approx 24576$。$s=2048$ → attn 仅约占 8%，常被并入 $6ND$ 误差；$s=32768$（长上下文）→ attn 与参数项同量级，**必须显式加上**。
- 这也是 [[llm-optimizer/FlashAttention]] 与 KV-Cache（[[llm-inference/KV-Cache优化]]、[[llm-optimizer/kv-cache]]）的动机：FlashAttention 不改变 FLOP 量级但大幅降访存；KV-Cache 在**推理 decode** 阶段把重复的 $s^2$ 摊销掉。

---

## 7. MFU / HFU：把"工作量"换成"真实时间"

有了任务 FLOPs 和硬件峰值，还差一个"打折系数"。

**MFU（Model FLOPs Utilization，模型算力利用率）**：

$$\text{MFU} = \frac{\text{模型理论 FLOPs/s（仅必要计算，用 }6ND\text{口径）}}{\text{硬件峰值 FLOPs/s}}$$

**HFU（Hardware FLOPs Utilization）**：分子改为**硬件实际执行**的 FLOP（含重计算的额外前向等"浪费"）。所以恒有 $\text{HFU} \ge \text{MFU}$。

```
   ┌─────────────────── 硬件峰值 100% ───────────────────┐
   │ 访存等待 │ 通信  │ kernel启动 │  有效计算(HFU)        │
   │  memory  │ comm  │  overhead  │ ┌──重计算浪费──┐MFU │
   │  bound   │       │            │ │              │     │
   └──────────┴───────┴────────────┴─┴──────────────┴─────┘
       拖低利用率的三大元凶          实际"有用"的部分
```

- 训练大模型常见 **MFU 30%~55%**；超过 50% 已属优化良好。
- 拉低 MFU 的原因：通信（TP/PP/DP，见 [[ai-infra/网络/集合通信原语]]、[[llm-inference/大模型推理张量并行]]）、访存瓶颈（小算子/LayerNorm/softmax 是 memory-bound）、流水线气泡、kernel launch 开销。重叠通信与计算可缓解（[[llm-optimizer/计算通信重叠]]）。

**用 MFU 反推时间**：

$$\text{训练时间} = \frac{C}{\text{GPU 数}\times \text{单卡峰值}\times \text{MFU}}$$

---

## 8. 变体：MoE 与推理 decode 的 FLOP 差异

**MoE（混合专家）**：FLOP 只与**被激活**的专家有关，不是全部参数。设总参数 $N$、每 token 激活 $N_{act}$（如 8 选 2），则训练用 $C\approx 6 N_{act} D$，**计算量按激活参数算，显存按总参数算**。这正是 MoE"参数大、算力省"的本质（见 [[llm-algo/moe/README]]、[[llm-base/distribution-parallelism/moe-parallel/README]]）。

**推理两阶段**（见 [[llm-inference/README]]、[[llm-inference/解码策略]]）：

```
 ① Prefill（处理 prompt，长度 s）   ── 算力受限 compute-bound
    一次性前向所有 token：FLOP ≈ 2·N·s   （只有前向，无 ×3）
 ② Decode（逐 token 生成）          ── 访存受限 memory-bound
    每生成 1 token：FLOP ≈ 2·N （借 KV-Cache，不重算历史）
    但要把全部权重从显存读一遍 → 瓶颈在带宽不在算力
```

- decode 阶段每步 FLOP 很小（$2N$），却要搬运整套权重，所以**推理 decode 几乎总是 memory-bound**，MFU 很低，优化重点转向 KV-Cache、量化、批处理（continuous batching）。
- 前向系数是 **2**（不是训练的 6），因为推理无反向。

---

## 数值手算：GPT-3 175B 端到端

**已知**：$N = 175\times10^9$ 参数；训练 $D = 300\times10^9$ token。集群：1024 张 A100，单卡 BF16 稠密峰值取 $312$ TFLOPS，假设 $\text{MFU}=45\%$。

**Step 1 — 总 FLOPs（用 $6ND$）**

$$C = 6 \times (175\times10^9) \times (300\times10^9) = 6 \times 5.25\times10^{22} = 3.15\times10^{23}\ \text{FLOPs}$$

**Step 2 — 集群有效算力**

$$P_{eff} = 1024 \times 312\times10^{12} \times 0.45 \approx 1.437\times10^{17}\ \text{FLOP/s}$$

（$1024\times312\text{T}=319.5\text{ PFLOPS}$ 峰值，×0.45 ≈ 143.8 PFLOPS 有效）

**Step 3 — 训练时间**

$$t = \frac{C}{P_{eff}} = \frac{3.15\times10^{23}}{1.437\times10^{17}} \approx 2.19\times10^{6}\ \text{s} \approx 25.4\ \text{天}$$

```
   3.15e23 FLOPs
  ───────────────────────────────  ≈ 2.19e6 s ≈ 25 天
   1024卡 × 312T × 0.45 MFU
```

> 量级对得上公开报道（GPT-3 约数千 V100·年量级的算力）。若 MFU 掉到 30%，时间 → 约 38 天；若换 H100（BF16 峰值约 3×），同 MFU 下 → 约 8.5 天。**这就是估算的全部价值：换硬件/换 MFU，秒出工期。**

**附：单层 FLOP 校验（$h{=}12288, s{=}2048, b{=}1$，GPT-3 维度）**
每层参数项前向 $24Th^2 = 24\times2048\times12288^2 \approx 7.4\times10^{12}$ FLOP；注意力项 $4bs^2h = 4\times2048^2\times12288 \approx 2.06\times10^{11}$，占比约 $2.8\%$ —— 印证 §6"$s\ll h$ 时平方项可忽略"。

---

## 常见问题

| 问题 | 答案 |
| --- | --- |
| FLOPs 和 FLOPS 区别？ | FLOPs=工作量（多少次运算）；FLOPS=速度（每秒几次）。小写 s 复数 vs /s 速率。 |
| 矩阵乘 FLOP 公式？ | $2mnk$（输出元素 $m{\times}n$ × 收缩维 $k$ × 2）。 |
| 为什么乘加是 2 FLOP？ | 1 乘 + 1 加，各 1 FLOP。 |
| $6ND$ 的 6 哪来的？ | 前向 2 + 反向 4 = 6（每参数每 token）。 |
| 开了梯度检查点用几？ | 约 8（多一遍重算前向）。 |
| 注意力 $s^2$ 项何时要加？ | $s$ 接近或超过 $6h$ 量级（长上下文）时；短序列可并入误差。 |
| 推理前向系数是几？ | 2（无反向）。decode 每步约 $2N$。 |
| MoE 怎么算？ | FLOP 用激活参数 $N_{act}$；显存用总参数 $N$。 |
| MFU 多少算好？ | 训练 30%~55%；>50% 即优化良好。 |
| 为啥实际远低于峰值？ | 通信、访存（memory-bound 算子）、流水线气泡、kernel 开销。 |
| 偏置/LayerNorm/激活算不算？ | FLOP 上可略（$O(N)$ vs GEMM $O(Nk)$）；但访存上可能很贵。 |
| decode 为何 memory-bound？ | 每步 FLOP 小但要搬整套权重，瓶颈在带宽。 |

---

## 🔗 跳转链接

- 计算量同主题富笔记：[[llm-algo/FLOPs]]
- 结构来源：[[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 访存优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]] · [[llm-optimizer/计算通信重叠]]
- 推理侧：[[llm-inference/README]] · [[llm-inference/解码策略]] · [[llm-inference/大模型推理张量并行]]
- 算力与硬件：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]] · [[ai-infra/网络/集合通信原语]]
- 精度/量化：[[机器学习中常用的数据类型]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 训练框架与流程：[[llm-train/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]
- 知识地图：[[00-知识地图]]
