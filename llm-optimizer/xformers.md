# xFormers
> Meta(FAIL)开源的 Transformer 组件库：把注意力、激活、归一化等做成一组**可组合的高效模块（building blocks）**，核心卖点是 **memory-efficient attention**（省显存的精确注意力）——它是 FlashAttention 思想最早的工程载体之一，也是 PyTorch SDPA 后端、Diffusers/SD 加速的幕后功臣。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-optimizer/FlashAttention]] [[llm-inference/FlashInfer]]

## 阅读地图
| 节 | 内容 | 一句话收获 |
|---|---|---|
| 0 | 一句话锚点 | xFormers = 「乐高式」高效 Transformer 算子库，主打省显存注意力 |
| 1 | 地基：注意力为什么吃显存 / 库 vs 算法 | 先分清「xFormers 是库，FlashAttention 是算法」 |
| 2 | memory-efficient attention 是什么 | 不物化 n×n 分数矩阵，显存 $O(n^2)\to O(n)$ |
| 3 | 它和 FlashAttention 到底什么关系 | 同源思想、互相吸收：dispatcher 会自动选最快后端 |
| 4 | 可组合块（building blocks） | Attention/FFN/位置编码都能拆装组合 |
| 5 | 偏置与稀疏：`attn_bias` 体系 | causal / padding / ALiBi / block-sparse 一套接口 |
| 6 | 调用方式与 dispatcher | 一个 `memory_efficient_attention` 入口，底层多 kernel |
| 7 | 何时用 / 何时不用 | 选型决策树 |
| 8 | 在生态中的位置 | SDPA / SD / Diffusers / 训练框架怎么用它 |
| 9 | 数值例子：显存到底省多少 | 用具体形状手算物化矩阵的字节数 |
| 10 | 常见问题 | 一眼看清坑与取舍 |

## 0. 一句话锚点
- **xFormers 是一个「库」，不是单一算法**。它打包了一堆为 Transformer 量身定做的高效、省显存、可组合的 PyTorch 算子，最有名的是 `memory_efficient_attention`。
- **它的注意力核心思想 = FlashAttention 思想**：分块计算 + online softmax，**永不把完整的 $n\times n$ 注意力分数矩阵写回显存**。所以显存从 $O(n^2)$ 降到 $O(n)$，**结果是精确的（不是近似）**。
- **它的工程价值在「dispatcher（调度器）」**：你只调一个统一 API，库会根据 GPU 架构、数据类型、张量形状、是否带偏置，**自动挑一个当下最快的底层 kernel**（可能是 xFormers 自己的 CUTLASS kernel，也可能直接转去调 FlashAttention）。
- **一句区分**：FlashAttention 是「一个把注意力做快做省的精确算法 + 它的 CUDA 实现」；xFormers 是「一个把这类算法 + 一堆其它 Transformer 模块装进 PyTorch、且帮你自动选最优实现的工具箱」。两者**不是竞争，是包含与协作**。

## 1. 地基：先把两件事讲清

### 1.1 注意力为什么吃显存（一分钟复习）
长度为 $n$、维度为 $d$ 的序列，$Q,K,V\in\mathbb{R}^{n\times d}$，注意力为：

$$O=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V$$

朴素实现分三步：
1. **打分**：$S=QK^\top/\sqrt d$，$S\in\mathbb{R}^{n\times n}$ —— 这是一张**边长 = 序列长度**的方阵。
2. **归一化**：$P=\mathrm{softmax}(S)$，按行做。
3. **加权**：$O=PV$。

痛点全在第 1 步那张 $n\times n$ 的 $S$（和 $P$）：它要**写进 HBM 显存、再读回来**。$n=8192$ 时，单头单层 $S$ 就是 $8192^2$ 个数，fp16 下 = 128 MB；多头多 batch 一乘就爆。而且它带来 $O(n^2)$ 的**显存**和大量 HBM 读写（注意力是**访存受限 memory-bound**的，见 [[ai-infra/算力/GPU工作原理]]）。

> **关键认知**：朴素注意力的「贵」不在算力，在那张 $n\times n$ 矩阵的反复读写显存。省掉它 = 既省显存又省时间。这正是 memory-efficient attention 要解决的。

### 1.2「库」和「算法」是两个层面
很多人把 xFormers 和 FlashAttention 当成「二选一」，这是误解。摊开看：

```
   层次          代表                做的事
 ───────────────────────────────────────────────
  应用层    HF Transformers / SD     搭模型、训练、推理
  库  层    xFormers ◄────────────  提供「可组合算子」+「自动选 kernel」的 dispatcher
            │                        ├─ memory_efficient_attention(...)
            │                        ├─ SwiGLU / fused MLP
            │                        ├─ 各种 attn_bias / 稀疏掩码
            v
  算法/kernel 层
            FlashAttention(fwd/bwd)  ◄─ 一个具体的高效精确注意力实现
            xFormers 自家 CUTLASS kernel
            PyTorch 原生 / Triton kernel
```

xFormers 站在「库」这一层：上接框架，下面**挂着多个 kernel 实现**，包括 FlashAttention。

## 2. memory-efficient attention 是什么

### 2.1 一句话：不物化 $S$，分块在片上算
`xformers.ops.memory_efficient_attention(q, k, v, attn_bias=...)` 的内核做的就是 FlashAttention 那套：把 $Q,K,V$ 切成能放进 GPU 片上 **SRAM** 的小块，逐块计算局部注意力，用 **online softmax** 边算边把结果累加进输出 $O$，**全程不把完整 $n\times n$ 的 $S$ 写回 HBM**。

```
  朴素注意力（物化 n×n）              memory-efficient（分块，不物化）
  ┌──────────────┐                   for j in K/V 块:
  │ S = QK^T  (n×n) │ ── 写 HBM        for i in Q 块:
  │   ↓ 读回 HBM    │                     片上算 S_ij(小块)
  │ P = softmax(S) │ ── 写 HBM            online 更新 (m_i, l_i, O_i)
  │   ↓ 读回 HBM    │                  ── 整张 S 永不落 HBM ──
  │ O = P V         │                  显存 O(n) ，结果精确
  └──────────────┘
  显存 O(n^2)，HBM 读写多
```

### 2.2 online softmax：为什么分块还能算对（直觉版）
softmax 要除以「整行的指数和」，可分块似乎看不到整行。online softmax 的技巧：每处理一个新块，维护两个**累计量**——当前见过的**行最大值** $m$ 和**指数和** $\ell$——发现更大的最大值就把已有结果**按比例缩放**修正。等所有块过完，结果**和一次性看整行完全相等**。所以它是**精确**，不是近似（详细推导见 [[llm-optimizer/FlashAttention]] 第 4 节）。

伪代码骨架：

```
m = -inf ; l = 0 ; O = 0
for 每个 K/V 块 (k_j, v_j):
    s   = q · k_j^T / sqrt(d)          # 小块分数
    m_new = max(m, rowmax(s))
    p   = exp(s - m_new)               # 数值稳定
    l   = l * exp(m - m_new) + rowsum(p)   # 旧和按比例缩放后并入
    O   = O * exp(m - m_new) + p @ v_j     # 旧输出同样缩放
    m   = m_new
O = O / l                              # 最后统一归一化
```

> 记忆点：**`exp(m - m_new)` 这个缩放因子**就是「发现更大最大值后，把过去的账本折算到新基准」的修正系数。它让分块结果严丝合缝等于全局结果。

## 3. 它和 FlashAttention 到底什么关系

这是本主题最容易混的点，单独拆清楚。

### 3.1 时间线与同源
- **2021/2022**：xFormers 先发布，里面就带了「memory-efficient attention」kernel（基于 Rabe & Staats 2021 的「self-attention does not need $O(n^2)$ memory」+ 自家 CUTLASS 实现）。
- **2022**：FlashAttention（Tri Dao 等）发表，把同类思想做成更优的 IO-aware kernel，速度更快。
- **之后**：两边**互相吸收**。xFormers 把 FlashAttention 收为**可选后端之一**；FlashAttention 也持续迭代（v2/v3）。今天你调 xFormers 的注意力，**底层很可能就是 FlashAttention 的 kernel 在跑**。

### 3.2 三句话区分
| | FlashAttention | xFormers |
|---|---|---|
| 本质 | 一个**精确注意力算法**及其高度优化的 CUDA kernel | 一个**Transformer 算子库 + dispatcher** |
| 范围 | 专注「注意力」一件事 | 注意力 + FFN/SwiGLU + 位置编码 + 稀疏 + 偏置… |
| 关系 | 被 xFormers **当作后端调用** | **包含/调度**包括 FlashAttention 在内的多个 kernel |

### 3.3 dispatcher：自动选后端
xFormers 注意力入口背后挂着一个**算子分发器**。你传进 $q,k,v,$ `attn_bias`，它根据【GPU 架构(SM 版本)、dtype(fp16/bf16)、head_dim、是否带特定偏置、是否变长】等条件，**遍历候选实现、选一个支持且最快的**：

```
 memory_efficient_attention(q,k,v,bias)
        │
        ▼  dispatcher：按 (arch,dtype,head_dim,bias 类型,变长?) 打分
   ┌────────────┬────────────┬───────────────┬──────────────┐
   │ Flash 后端  │ CUTLASS 后端 │ Triton/decoder │ 朴素 fallback │
   │(FlashAttn) │(xFormers 自家)│ (特定场景特化)  │ (兜底,慢)     │
   └────────────┴────────────┴───────────────┴──────────────┘
        └── 选中其一执行，结果对调用者透明 ──┘
```

> 实践含义：**你不需要手动判断该用哪个 kernel**——这正是「库」相对「单算法」的价值。某些偏置/形状下 FlashAttention 当时不支持，xFormers 的 CUTLASS 后端能顶上，反之亦然。

## 4. 可组合块（building blocks）

xFormers 的另一半价值：除了注意力，它把 Transformer 各部件做成**可独立调用、可拼装**的高效模块。理念是「一个 Transformer = 一堆块的组合，每块都可替换」。

```
              一个 Transformer Block 的拆解
  ┌───────────────────────────────────────────────┐
  │  x ─► [LayerNorm] ─► [Attention 块] ─► (+x 残差) │
  │         │                │                       │
  │         │          可换：MHA / MQA / GQA /       │
  │         │                memory-efficient / 稀疏  │
  │  ─► [LayerNorm] ─► [FeedForward 块] ─► (+ 残差)   │
  │                          │                       │
  │                   可换：MLP / SwiGLU(fused) /     │
  │                         GLU 变体 / Mixture        │
  └───────────────────────────────────────────────┘
   位置编码块：Sinusoidal / Rotary(RoPE) / ALiBi(走 bias)
```

常见可组合组件（具体名称以官方 API 为准）：
- **注意力组件**：标准 MHA、memory-efficient attention、稀疏/块稀疏注意力等。
- **前馈组件**：融合的 **SwiGLU/GLU** 前馈（把两次线性 + 门控激活融成更省访存的实现）。
- **位置编码**：旋转位置编码 RoPE、ALiBi（后者通过 `attn_bias` 注入线性偏置）。
- **归一化/激活/dropout** 等小算子的高效版本。

> 价值：研究者搭新结构时，可以「换一块、测一块」，而不用从零写 CUDA；工程上则能挑选**已被融合优化过**的块直接提速。注意：早期 xFormers 提供过一个组装整模型的高层 API，社区重心后来更偏向「直接用 `memory_efficient_attention` + 少数融合算子」，整模型组装 API 的活跃度随版本变化——**以官方当前文档为准**。

## 5. 偏置与稀疏：`attn_bias` 体系

注意力常需要「掩码 / 偏置」：因果遮挡、padding 屏蔽、ALiBi 斜率、块稀疏等。xFormers 把它们统一成 **`attn_bias` 对象**，传给同一个注意力入口。好处是这些偏置能被 kernel **原生、融合地**处理，而不是先物化一张 $n\times n$ 的 mask 矩阵再相加（那又回到了 $O(n^2)$ 显存）。

```
  attn_bias 家族（示意）
  ─ LowerTriangularMask        因果：位置 i 只能看 ≤ i（decoder 自回归）
  ─ Block/Padding 掩码          变长 batch：屏蔽 padding，支持「无填充」打包
  ─ ALiBi 斜率                  按相对距离加线性惩罚（外推长度友好）
  ─ BlockDiagonal / 块稀疏       只算选定块，跳过其余 → 省算省存
```

```
  因果掩码（n=5），✓=可见 ✗=屏蔽
        k0 k1 k2 k3 k4
   q0 [ ✓  ✗  ✗  ✗  ✗ ]
   q1 [ ✓  ✓  ✗  ✗  ✗ ]
   q2 [ ✓  ✓  ✓  ✗  ✗ ]   ← kernel 内部直接「不算右上三角」，
   q3 [ ✓  ✓  ✓  ✓  ✗ ]      省一半算力，且不物化 mask
   q4 [ ✓  ✓  ✓  ✓  ✓ ]
```

> 变长/打包（**BlockDiagonal**）很关键：把一个 batch 里长度不一的多条序列**首尾拼成一条**、用块对角偏置标出边界，避免 padding 浪费——这与 [[llm-inference/FlashInfer]] 在推理侧做变长/分页注意力的动机一致。

## 6. 调用方式与 dispatcher（实践骨架）

最常用就一个函数，形状约定通常是 `(batch, seqlen, num_heads, head_dim)`（**注意头维在 seq 之后**，与部分 PyTorch API 顺序不同）：

```python
import torch
from xformers.ops import memory_efficient_attention, LowerTriangularMask

B, S, H, D = 2, 4096, 16, 64
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# 不带掩码（双向/编码器场景）
out = memory_efficient_attention(q, k, v)

# 因果（自回归解码器）：传一个 bias 对象，kernel 内部高效处理
out = memory_efficient_attention(q, k, v, attn_bias=LowerTriangularMask())
# 反向也支持：out 可直接接 loss.backward()，省显存优势在训练时同样成立
```

要点：
- **dtype 用 fp16/bf16** 才能命中最快 kernel；fp32 往往退化到慢路径。
- **head_dim 有上限/对齐要求**（随后端/版本不同，常见到 128/256 等），不满足会回退——以官方为准。
- 想知道实际选了哪个后端，可用库提供的 dispatch 检查工具打印候选与选中项（API 名以官方为准）。

## 7. 何时用 / 何时不用

```
                  我要不要直接上 xFormers？
                          │
        ┌─────────────────┴──────────────────┐
   训练/微调长序列                         纯推理服务（高并发、KV cache）
   或显存吃紧？                                  │
        │ 是                          首选推理专用栈：FlashInfer /
        ▼                             vLLM(PagedAttention) 等
   用 memory_efficient_attention      （它们针对解码、分页 KV、
   （或 PyTorch SDPA，见下）            变长批做了更专门优化）
        │
   还需要 SwiGLU / 稀疏 / 特殊 bias？
        │ 是 ── xFormers 的可组合块直接省事
        │ 否 ── 也可只用 PyTorch SDPA（底层同样会走 Flash/mem-eff）
```

- **该用**：① 训练/微调中**注意力显存爆**、想把序列拉长；② 需要**省显存的精确注意力**且不想自己写 kernel；③ 需要 **SwiGLU、块稀疏、ALiBi、变长打包**等现成高效块；④ 跑 **Stable Diffusion / Diffusers**，xFormers 是经典提速开关。
- **可不用 / 改用别的**：① 你已在用 **PyTorch SDPA**——它底层会自动选 Flash/memory-efficient 后端，多数训练场景够用，少一个依赖；② **大规模推理服务**（在线解码、长 KV cache、连续批处理）——优先 [[llm-inference/FlashInfer]]、vLLM 等**推理专用**注意力/KV 方案，它们对 decode 阶段、分页 KV、变长批的优化更深；③ 形状/dtype 不命中其快路径时，提速有限。

## 8. 在生态中的位置

```
  ┌── PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention)
  │      └─ 后端候选：flash / memory-efficient(同 xFormers 思想) / math
  │
  ├── xFormers ── memory_efficient_attention + 可组合块
  │      └─ 后端候选：FlashAttention / 自家 CUTLASS / Triton / fallback
  │
  ├── Stable Diffusion / Diffusers ── 历史上「开启 xformers」= 一键省显存提速
  │
  └── 训练框架(如部分 LLM 训练栈) ── 把 mem-eff attention 当注意力实现
```

一句话定位：xFormers 既是**直接可用的加速库**，又是**算法思想的传播枢纽**——它把「不物化 $n\times n$」这套省显存注意力，标准化、可组合化、自动调度化地交到了 PyTorch 用户手里；同样的思想后来沉淀进了 PyTorch SDPA，也启发了推理侧的 [[llm-inference/FlashInfer]]。

## 9. 数值例子：显存到底省多少

设 **batch $B=8$，序列 $n=8192$，头数 $H=32$，头维 $d=128$，fp16（2 字节/数）**，看「物化 $n\times n$ 分数矩阵」要多少显存。

**朴素注意力**：要为每个 (batch, head) 存一张 $n\times n$ 的 $S$（前向算 $P$、反向还要它）。
- 单张 $S$ 元素数：$n^2 = 8192^2 = 67{,}108{,}864 \approx 6.7\times10^7$。
- 单张字节：$6.7\times10^7 \times 2\text{B} \approx 134\,\text{MB}$。
- 乘以 $B\times H = 8\times32 = 256$：

$$134\,\text{MB}\times 256 \approx 34\,\text{GB}$$

**仅这一张中间矩阵就要约 34 GB**——足以让一张 40 GB 卡在训练时直接 OOM（还没算权重、激活、优化器状态）。

**memory-efficient attention**：不物化 $S$，注意力中间显存随 $n$ 线性增长。粗估每个 (batch,head) 只需保存与 $O$、行统计量 $(m,\ell)$ 同量级的东西，约 $O(n\cdot d)$：
- $n\cdot d = 8192\times128 \approx 1.05\times10^6$ 个数 ≈ 2 MB/头。
- $\times 256 \approx 0.5\,\text{GB}$ 量级（外加 $Q,K,V,O$ 本身的常规存储）。

| 量 | 朴素（物化 $S$） | memory-efficient | 倍数 |
|---|---|---|---|
| 注意力中间显存 | $O(n^2)$ ≈ **34 GB** | $O(n\cdot d)$ ≈ **亚 GB 级** | 省 **几十倍** |
| HBM 读写 | 多（反复读写 $S$） | 少（$S$ 不落显存） | 大降 |
| 数值结果 | 基准 | **逐位等价（精确）** | 不变 |
| FLOPs | 基准 | 几乎相同 | ≈ 不变 |

> 结论与 FlashAttention 完全一致：**省的是显存与访存，不是计算量**；正因访存是瓶颈，省访存顺带也提了速。把 $n$ 翻倍，朴素的 $S$ 显存 ×4（$O(n^2)$），mem-eff 只 ×2（$O(n)$）——序列越长，优势越大。

## 10. 常见问题
| 问题 | 回答 |
|---|---|
| xFormers 和 FlashAttention 二选一吗？ | 不是。xFormers 是「库 + 调度器」，FlashAttention 是它**可调用的后端之一**；很多时候你用 xFormers，底层跑的就是 FlashAttention。 |
| memory-efficient attention 是近似吗？ | **否，精确**。靠 online softmax，分块结果逐位等于一次性算全行。 |
| 它省的是算力还是显存？ | 主要省**显存 + HBM 访存**；FLOPs 几乎不变。因注意力访存受限，省访存也顺带提速。 |
| 我已经用 PyTorch SDPA，还需要 xFormers 吗？ | SDPA 多数训练场景够用（底层也走同类后端）。若要 **SwiGLU/块稀疏/特殊 bias/变长打包** 等现成块，或要更细的后端控制，再上 xFormers。 |
| 推理服务该用它吗？ | 训练/离线可以；**在线高并发解码**优先 [[llm-inference/FlashInfer]]、vLLM 等推理专用栈，它们对 decode、分页 KV、连续批处理优化更深。 |
| 为什么没命中快 kernel？ | 常见原因：dtype 是 fp32、head_dim 不满足对齐/上限、bias 类型不被某后端支持、GPU 架构太老。换 fp16/bf16、对齐形状，或查 dispatch 日志。 |
| 张量形状怎么排？ | 通常 `(B, S, H, D)`，头维在 seq **之后**，与部分 PyTorch 习惯不同，传错会报形状错误。 |
| Stable Diffusion 里的「xformers 开关」是什么？ | 就是把交叉/自注意力换成 memory-efficient 实现，**省显存 + 提速**，是社区经典加速手段。 |
| 这些数字（head_dim 上限、显存）准吗？ | 显存量级按本文形状手算可复现；**具体 head_dim 上限/支持矩阵随版本变化，以官方 README/文档为准**。 |

## 🔗 跳转链接
- [[00-知识地图]] — 全局导航总入口
- [[llm-optimizer/FlashAttention]] — 同源的精确省显存注意力算法（online softmax 推导、v1/v2/v3 演进），xFormers 的核心后端
- [[llm-inference/FlashInfer]] — 推理侧的注意力/KV 优化（分页、变长、decode 专门化），与 xFormers 形成「训练 ↔ 推理」分工
- [[llm-optimizer/kv-cache]] — KV cache 是推理省显存的另一条主线，与注意力优化互补
- 官方仓库：https://github.com/facebookresearch/xformers
