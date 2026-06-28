# DeepSpeed-Inference：大模型推理引擎的算子融合与张量并行

> 一句话定位：DeepSpeed-Inference 是微软 DeepSpeed 团队为 Transformer 推理打造的高性能内核库，靠「手工算子融合 + 推理张量并行 + 权重量化/Offload」把延迟和显存压到极致。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/deepspeed/README]] · [[llm-optimizer/FlashAttention]] · [[llm-inference/大模型推理张量并行]] · [[llm-optimizer/kv-cache]] · [[llm-optimizer/计算通信重叠]]

## 阅读地图

| 节 | 你会学到 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | 融合内核 + TP + 量化 |
| 1 | 地基：推理为什么慢 | 访存墙 / kernel launch / Transformer 层结构 |
| 2 | 算子融合总览（四类） | QKV 融合 / Attn 融合 / MLP 融合 / Bias-Residual 融合 |
| 3 | 融合一：LayerNorm + QKV 横向融合 | 减少 kernel / 减少访存 |
| 4 | 融合二：自注意力融合（FlashAttention 思路） | 不落 S 矩阵 / online-softmax |
| 5 | 融合三：MLP 残差-Norm-FC-激活融合 | GeLU 融合 / 纵向融合 |
| 6 | 融合四：Bias + Residual 融合 | element-wise 合并 |
| 7 | 推理张量并行（Inference-TP） | 列切/行切 / all-reduce |
| 8 | 权重量化与 ZeRO-Inference Offload | INT8 / NVMe offload |
| 9 | 数值手算 | 访存量 / 通信量 / 显存 |
| 10 | 常见问题 | FAQ 表 |

## 0. 一句话锚点

Transformer 推理是 **访存受限（memory-bound）** 的：算得快、搬得慢。DeepSpeed-Inference 的核心招式只有三板斧：

1. **算子融合（kernel fusion）**：把一层里几十个零碎的小算子，按数据相关性合并成 4~5 个大融合核，减少「kernel 启动次数」和「中间结果反复进出显存」。
2. **推理张量并行（Inference-adapted Tensor Parallelism）**：把单层权重切到多张卡，单卡放不下的大模型也能跑，且降低单卡访存。
3. **量化 + Offload（ZeRO-Inference）**：INT8 权重量化省一半显存带宽；权重 Offload 到 CPU/NVMe，让单卡也能跑超大模型。

本文重点在第 1 点（最有「从底层讲清」价值），并把 2、3 讲透。

## 1. 地基：推理为什么慢，融合为什么有用

### 1.1 GPU 执行一个算子的真实成本

每个 CUDA kernel 的生命周期：

```
CPU 端                         GPU 端
  │  launch kernel (≈5~10us)
  ├──────────────────────────►  从 HBM 读输入张量  (访存)
  │                              在 SM 上计算       (算力)
  │                              把结果写回 HBM     (访存)
  │  ◄───────────────────────   完成
```

两个隐藏开销：
- **Kernel launch 开销**：每次启动约 $5\sim10\,\mu s$ 的固定 CPU→GPU 调度成本。一层有 N 个算子就是 N 次。
- **中间结果落显存**：算子 A 的输出写回 HBM，算子 B 再从 HBM 读回来。这一写一读纯属浪费——如果 A、B 融合，中间结果留在寄存器/共享内存里就不用走 HBM。

### 1.2 Roofline：推理卡在「访存墙」

定义**算术强度** $I = \dfrac{\text{FLOPs}}{\text{Bytes 访存}}$（FLOPs/Byte）。GPU 的「平衡点」：

$$I_{\text{平衡}} = \frac{\text{峰值算力}}{\text{显存带宽}}$$

以 A100 为例：FP16 峰值算力 $\approx 312\,\text{TFLOPS}$，HBM 带宽 $\approx 2\,\text{TB/s}$，于是

$$I_{\text{平衡}} = \frac{312\times10^{12}}{2\times10^{12}} = 156\ \text{FLOP/Byte}$$

```
性能(FLOPS)
   │            ┌──────── 算力屋顶 312T
   │           ╱
   │          ╱  ← 斜线 = 带宽×I
   │         ╱
   │   ●LN  ╱   ●Bias  ●激活   ← I≈1~2，远在斜坡左侧 = 访存受限
   └────┴─────────────────────► 算术强度 I
       1    ...    156(平衡点)
```

LayerNorm、Bias 加法、激活、softmax 这些 element-wise 算子的 $I$ 只有 1~2 FLOP/Byte，**远低于 156**，所以它们的耗时几乎完全由「搬数据」决定，算力闲着。**融合的本质就是消灭这些算子的重复访存**。

### 1.3 一个 Transformer Decoder 层的算子清单

```
       x ──► LayerNorm1 ──► Linear_Q ─┐
                          ├ Linear_K ─┼─► Attention(QKᵀ→softmax→·V) ─► Linear_O ─► +x ─┐
                          └ Linear_V ─┘                                                │
        ┌─────────────────────────────────────────────────────────────────────────────┘
        └► LayerNorm2 ──► Linear_FC1 ──► GeLU ──► Linear_FC2 ──► +residual ──► out
```

朴素实现里，光是上面就有 LN×2、Linear×6、softmax、GeLU、若干 Bias 加法和 Residual 加法，**十几个 kernel**。DeepSpeed 把它们压成 **4 类融合核**。

## 2. 算子融合总览：四类融合

DeepSpeed-Inference 针对 Transformer 层结构手工实现了四类融合（这正是本文原始要点）：

| # | 融合名 | 合并了什么 | 类型 |
|---|---|---|---|
| 一 | **归一化层 + QKV 横向融合** | LayerNorm 与三次 Q/K/V 投影合并 | 横向(siblings) + 纵向 |
| 二 | **自注意力计算融合** | $QK^\top$→scale→softmax→$\cdot V$ 合并（即 FlashAttention 思路） | 纵向(chain) |
| 三 | **残差 + 归一化 + 全连接 + 激活融合** | MLP 第一个 FC 上下相关算子合并 | 纵向 |
| 四 | **偏置加法 + 残差连接融合** | Bias 加法与 Residual 加法合并 | element-wise |

两个正交维度：
- **横向融合（horizontal）**：多个**互相独立、输入相同**的算子合并（如 Q/K/V 三个投影共享同一个输入 LN 输出）。
- **纵向融合（vertical）**：数据**前后依赖**的算子链合并（如 FC1→GeLU），让中间结果不落 HBM。

```
横向融合（共享输入）           纵向融合（链式依赖）
        x                          a ─► op1 ─► op2 ─► op3 ─► out
      ┌─┼─┐                              └─融成一个核，中间不落HBM─┘
     Q  K  V   ← 一个核算三个
```

## 3. 融合一：LayerNorm + QKV 横向融合

### 3.1 朴素 vs 融合

```
朴素：x ─►[LN]─►h─►[Wq]─►Q   3次LN读x? 不，LN算一次h
                 ├►[Wk]─►K   但 h 写回HBM，再被Wq/Wk/Wv 各读一次 = h读3次
                 └►[Wv]─►V   且 3个 GEMM 是 3次 kernel launch

融合：x ─►[ LN + Wqkv 融合核 ]─► [Q|K|V]  （一个核，h 留在片上，1次launch）
                                   ↑ 把 Wq|Wk|Wv 在列方向拼成一个大矩阵 Wqkv
```

### 3.2 为什么能合并 Q/K/V

三个投影是「兄弟算子」：输入都是 LN 的输出 $h$，互相独立。把三个权重矩阵在输出维度（列）方向拼接：

$$W_{qkv} = [\,W_q \mid W_k \mid W_v\,] \in \mathbb{R}^{d\times 3d}$$

一次 GEMM $h W_{qkv}$ 就同时得到 Q、K、V。**好处**：
- 3 次 kernel launch → 1 次；
- $h$ 只从片上读一次（LN 输出直接喂给 GEMM，不落 HBM）；
- 大 GEMM 比三个小 GEMM 的算力利用率（占用率）更高。

### 3.3 注意：解码阶段 GEMM 退化为 GEMV

推理「逐 token 生成」时 batch 内每个序列只算 1 个新 token，矩阵乘 $1\times d \cdot d\times 3d$ 退化成 **GEMV（矩阵-向量乘）**，算术强度极低、纯访存受限——这正是融合最该省访存的地方。

## 4. 融合二：自注意力计算融合（FlashAttention 思路）

### 4.1 朴素注意力的访存灾难

$$\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

朴素实现会**物化**中间分数矩阵 $S=QK^\top \in \mathbb{R}^{n\times n}$（n=序列长度）：

```
Q,K ─►[GEMM]─► S(n×n) ──写HBM──► [scale] ──► [softmax] ──► P(n×n) ──► [GEMM ·V] ──► O
                  ↑ n=2048 时 S 占 2048² ×2B ≈ 8MB/头，反复进出 HBM = O(n²) 访存
```

### 4.2 融合后：不落 S，online-softmax 分块

FlashAttention 把整条链融成一个核，分块（tiling）遍历 K/V，**S 永不写回 HBM**，softmax 用「在线更新」的方式增量算：

```
for 每个 K/V 块 j:
    S_ij = Q_i Kⱼᵀ        ← 留在共享内存
    用 running max m、running sum l 在线更新 softmax
    O_i  += P_ij · Vⱼ      ← 累加到寄存器
```

访存从 $O(n^2)$ 降到 $O(n)$（只读 Q/K/V、写 O）。原文所说「自注意力计算融合，业界熟知的 FlashAttention 即成熟方案」就是指这一类。细节见 [[llm-optimizer/FlashAttention]]。

### 4.3 推理特有：KV-Cache 让 Attn 变成增量

解码阶段，历史 K/V 已缓存，新 token 只需算 1 行 $S$（$1\times n$），即「增量注意力」。融合核要直接读 KV-Cache、避免重复计算历史。见 [[llm-inference/KV-Cache优化]] / [[llm-optimizer/kv-cache]]。

## 5. 融合三：MLP 的残差-Norm-FC-激活融合

MLP（FFN）块：

$$\text{out} = x + W_2\cdot\text{GeLU}(W_1\cdot\text{LN}(x) + b_1) + b_2$$

```
朴素：x─►[LN]─►[FC1]─►h─►[+b1]─►[GeLU]─►g─►[FC2]─►[+b2]─►[+x]─►out
            每个 ─► 都是一次 launch + 中间张量进出HBM

融合：x─►[ LN+FC1 融合核 ]─►[ Bias+GeLU 融合核 ]─►[ FC2 ]─►[ Bias+Residual 融合核 ]─►out
       纵向把 LN 喂进 FC1；GeLU 把 +b1 和激活合并（element-wise 链）
```

**关键点**：`Bias + GeLU` 是典型纵向融合——两个 element-wise 算子前后依赖，合并后中间张量 $h$（形状 $n\times 4d$，巨大）不再写 HBM。FFN 的隐藏维通常是 $4d$，这块中间张量是全层最大的之一，省下它的访存收益最高。

## 6. 融合四：偏置加法 + 残差连接融合

Transformer 里大量出现「线性层输出 + bias + 残差」的模式：

$$y = (\,\text{Linear}(x) + b\,) + \text{residual}$$

```
朴素：z─►[+ bias]─►t─►[+ residual]─►y    两次 element-wise，t 落HBM
融合：z─►[ +bias +residual 一个核 ]─►y    一次读 z/bias/res，一次写 y
```

bias 加法和残差加法都是 element-wise，算术强度 ≈1，**纯访存受限**。合并后访存次数减半。DeepSpeed 还常把它和后续的下一层 LayerNorm 一起做（pre-LN 结构里 residual 输出紧接着是下一层的 LN）。

## 7. 推理张量并行（Inference-adapted TP）

单卡放不下 → 把每层权重切到 $p$ 张卡。DeepSpeed-Inference 用经典的 Megatron 式切法（详见 [[llm-inference/大模型推理张量并行]] / [[ai-framework/megatron-lm/README]]）：

### 7.1 切法：QKV 列切、O 行切，一次 all-reduce

```
       x (每卡都有完整 x)
        │  广播
   ┌────┴────┐
 GPU0       GPU1
 Wqkv列切0  Wqkv列切1   ← 列并行：各算各的头，无需通信
 Attn(头0)  Attn(头1)
 Wo行切0    Wo行切1     ← 行并行：各得部分和
   └──► all-reduce ◄──┘  ← 把部分和加起来，得到完整 O
```

- **Attention**：QKV 按列切（按注意力头分组），每卡独立算自己的头；输出投影 $W_o$ 按行切，最后一次 **all-reduce** 求和。
- **MLP**：$W_1$ 列切、$W_2$ 行切，同样一次 all-reduce。
- 一个 Transformer 层 = **2 次 all-reduce**（Attn 一次、MLP 一次）。

### 7.2 推理 TP 与训练 TP 的差异

| 维度 | 训练 TP | 推理 TP |
|---|---|---|
| 通信 | 前向+反向各 all-reduce | 只有前向 |
| batch | 大 | 常为 1，对延迟敏感 |
| 目标 | 吞吐 | **首 token 延迟 + 单 token 延迟** |
| 通信占比 | 可被计算掩盖 | batch 小时通信占比高，是瓶颈 |

batch=1 解码时，GEMM 退化 GEMV，计算极快，**all-reduce 通信反而成了主导**。可结合 [[llm-optimizer/计算通信重叠]] 思路尽量掩盖。

## 8. 权重量化与 ZeRO-Inference Offload

- **INT8 权重量化**：DeepSpeed 提供 MoQ（Mixture-of-Quantization），权重 FP16→INT8，**显存与访存带宽减半**，对访存受限的推理直接近 2× 加速。激活仍可保 FP16。原理见 [[llm-compression/quantization/量化基础]]。
- **ZeRO-Inference（Offload）**：把权重放到 **CPU 内存甚至 NVMe**，推理时按层 prefetch 到 GPU，算完即释放。让单张消费级 GPU 跑下百亿~千亿模型，代价是 PCIe/NVMe 带宽成新瓶颈。架构见 [[ai-framework/deepspeed/README]]。
- **MoE 推理**：稀疏专家结构需专家并行，参考 [[llm-algo/moe/README]]。

## 9. 数值手算

### 9.1 融合省了多少访存？以 Bias+Residual 为例

设隐藏维 $d=4096$，序列 $n=2048$，FP16（2 字节）。中间张量 $t$ 大小：

$$n\times d\times 2 = 2048\times4096\times2 = 16\,\text{MB}$$

- 朴素：写 $t$（16MB）+ 读 $t$（16MB）= **32MB 多余访存**。
- 融合：省掉这 32MB。在 2TB/s 带宽下省 $\dfrac{32\times10^6}{2\times10^{12}}=16\,\mu s$/层。
- 一层省的 kernel launch：4 类融合把 ~12 个 kernel 压到 ~5 个，省 7 次 ×$8\mu s = 56\,\mu s$/层。
- **70 层模型**：$(16+56)\times70 \approx 5\,\text{ms}$/token，对解码每 token 几十毫秒级是可观占比。

### 9.2 QKV 横向融合省的 h 访存

LN 输出 $h$ 形状 $n\times d$ = 16MB（同上）。朴素被 Q/K/V 三个 GEMM 各读一次 = 读 3 次（48MB）；融合后 LN 输出直接进大 GEMM，读 1 次（16MB）。**省 32MB/层**。

### 9.3 张量并行通信量（一层）

TP=8、$d=4096$、$n=2048$、FP16。每次 all-reduce 的张量是激活 $n\times d$ = 16MB。Ring all-reduce 单卡收发约 $2\times\frac{p-1}{p}\times \text{size}$：

$$2\times\frac{7}{8}\times16\,\text{MB} = 28\,\text{MB}/\text{卡}/\text{次}$$

一层 2 次 = 56MB/卡/层。NVLink 300GB/s 下：$\dfrac{56\times10^6}{300\times10^9}\approx0.19\,\text{ms}$/层。70 层 ≈ **13ms**（prefill 阶段）。原语机制见 [[ai-infra/网络/集合通信原语]]。

### 9.4 INT8 量化省显存

13B 模型权重 FP16 = $13\times10^9\times2 = 26\,\text{GB}$；INT8 = $13\times10^9\times1 = 13\,\text{GB}$。从「单卡 24G 放不下」变成「放得下」，且访存带宽需求减半 → 访存受限推理近 2× 提速。

## 10. 常见问题

| 问题 | 答 |
|---|---|
| 算子融合到底快在哪？ | 减少 kernel launch 次数 + 减少中间结果进出 HBM（推理是访存受限，省访存=省时间） |
| 横向和纵向融合区别？ | 横向=独立兄弟算子共享输入合并（QKV）；纵向=链式依赖合并、中间不落 HBM（FC1+GeLU） |
| 为什么 Q/K/V 能拼成一个 GEMM？ | 三者输入相同、互相独立，权重在列方向拼成 $W_{qkv}$，一次大 GEMM 算完 |
| 自注意力融合就是 FlashAttention？ | 是同一思路：不物化 $n\times n$ 分数矩阵、online-softmax 分块，访存从 $O(n^2)$ 降到 $O(n)$ |
| 推理 TP 和训练 TP 差别？ | 推理只前向、batch 常为 1、延迟敏感；batch 小时 all-reduce 通信占比高成瓶颈 |
| 解码阶段 GEMM 为何退化？ | 每步只算 1 个新 token，$1\times d$ 矩阵乘 → GEMV，算术强度极低、纯访存受限 |
| ZeRO-Inference 怎么跑大模型？ | 权重 Offload 到 CPU/NVMe，按层 prefetch；瓶颈转移到 PCIe/NVMe 带宽 |
| 与 vLLM 的定位差异？ | DeepSpeed-Inference 偏「融合内核+TP+量化」；vLLM 偏「PagedAttention 显存管理+连续批处理」，见 vllm.md |
| 具体 API/版本/默认值？ | 随版本演进较快，以官方文档为准（本文只讲稳定原理与机制） |

## 🔗 跳转链接

- 框架总览：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]
- 注意力与 KV：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]]
- 并行与通信：[[llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]] · [[llm-optimizer/计算通信重叠]]
- 模型结构与算量：[[llm-algo/transformer/模型架构]] · [[llm-algo/FLOPs]] · [[llm-algo/mlp]] · [[llm-algo/moe/README]]
- 量化：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 硬件地基：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]]
- 推理总览与解码：[[llm-inference/README]] · [[llm-inference/解码策略]]
- 训练侧：[[llm-train/README]]
- 全局：[[00-知识地图]]
