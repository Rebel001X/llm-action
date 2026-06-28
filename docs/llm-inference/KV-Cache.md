# KV Cache（键值缓存）

> 自回归推理时把每个 token 已算出的 Key/Value 缓存下来、避免重复计算的核心加速技术；它把"每步重算整段"的 $O(n^2)$ 退化成"每步只算一行"的 $O(n)$，但代价是显存被它吃成内存刺客。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-optimizer/kv-cache]] · [[llm-inference/KV-Cache优化]] · [[llm-optimizer/FlashAttention]] · [[llm-algo/transformer/模型架构]] · [[llm-inference/解码策略]] · [[llm-inference/大模型推理张量并行]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|--------|
| 0 | 一句话锚点 | 缓存 K/V，换计算为显存 |
| 1 | 地基：自回归 + 注意力为什么能缓存 | causal mask / 增量解码 |
| 2 | 没有 cache 时的重复计算量 | $O(n^2 d)$ 浪费 |
| 3 | KV Cache 机制：Prefill vs Decode 两阶段 | 预填充 / 解码 |
| 4 | 缓存里到底存了什么、形状多大 | $[B,L,H,S,d]$ |
| 5 | 显存公式与逐数手算 | 2·B·L·H·S·d·dtype |
| 6 | 为什么不缓存 Q、不缓存 attention 权重 | 只有 K/V 被复用 |
| 7 | 带宽墙：KV Cache 是访存瓶颈 | memory-bound |
| 8 | 省显存的家族：MQA/GQA/MLA/量化/PagedAttention | 压缩与管理 |
| 数值手算 | 7B/13B 多场景显存账本 | GB 级估算 |
| FAQ | 高频疑问表 | — |

## 0. 一句话锚点

**自回归生成第 $t$ 个 token 时，注意力需要用到前面所有 token 的 Key 和 Value。这些 K/V 在它们各自被生成的那一步就已经算过一次了，且永远不变。把它们存起来（KV Cache），后续每步就只需为"当前这一个新 token"算 K/V，而不必把整段历史重算一遍。**

一句话权衡：**KV Cache 用"显存空间"换"计算时间"**，把生成阶段从计算密集变成访存密集。

## 1. 地基：自回归 + 注意力，为什么 K/V 可以缓存

大语言模型生成是**自回归（autoregressive）**的：一次吐一个 token，把它接到序列末尾，再喂回去预测下一个。

```
prompt: "今天 天气"
step1: [今天, 天气] -> 预测 "很"
step2: [今天, 天气, 很] -> 预测 "好"
step3: [今天, 天气, 很, 好] -> 预测 <eos>
        ^^^^^^^^^^^^^^^^ 序列每步加长 1
```

单层注意力（缩放点积）：

$$\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}+M\right)V$$

其中 $M$ 是**因果掩码（causal mask）**：第 $i$ 个 token 只能看 $\le i$ 的位置（上三角置 $-\infty$）。

```
       k1   k2   k3   k4   (被关注的 key)
 q1 [  .    -inf -inf -inf ]
 q2 [  .    .    -inf -inf ]   . = 允许
 q3 [  .    .    .    -inf ]   每行 softmax 后求和=1
 q4 [  .    .    .    .    ]
```

**关键观察**：因为有因果掩码，token $i$ 的 $K_i,V_i$ 一旦算出来，在后续任何步里都**完全不变**（它不依赖未来 token）。这就是"可缓存"的根本前提——被复用的量是**只依赖历史、不依赖未来**的常量。

> 链路：注意力机制本身见 [[llm-algo/transformer/模型架构]]；位置编码若是 RoPE，K 在写入 cache 前/后旋转的细节见 [[llm-algo/旋转编码RoPE]]。

## 2. 没有 cache 时，浪费了多少计算

设隐藏维 $d$、序列已长 $n$。生成第 $n{+}1$ 个 token，**朴素做法**把 $n{+}1$ 个 token 全部过一遍模型：

- 算 $Q,K,V$：每个 token 做 $d\times d$ 投影，$n{+}1$ 个 token 共 $\approx 3(n{+}1)d^2$ 次乘加。
- 但其中 $K_1..K_n, V_1..V_n$ 上一步就算过了！只有第 $n{+}1$ 个是新的。

把整个长度 $n$ 的句子逐字生成完，朴素法总计算 $\propto \sum_{t=1}^{n} t \cdot d^2 = O(n^2 d^2)$。

```
朴素(无cache)        有 KV Cache
step t 重算 t 个token   step t 只算 1 个token
 t=1 |#                  t=1 |#
 t=2 |##                 t=2 |#
 t=3 |###                t=3 |#
 t=4 |####               t=4 |#
 累计 ~ n^2/2            累计 ~ n   (线性!)
```

**结论**：KV Cache 把生成阶段每步的注意力前置投影从 $O(n)$ 降到 $O(1)$，整句从 $O(n^2)$ 降到 $O(n)$。这是它存在的全部理由。

## 3. KV Cache 机制：两个阶段

推理被切成两个性质完全不同的阶段：

### 3.1 Prefill（预填充）

把整个 prompt（长 $S_p$）**一次性并行**喂进去，算出所有层每个 token 的 K/V，**写入 cache**，同时得到第一个输出 token。这一步是**计算密集（compute-bound）**的，因为有大矩阵乘 $S_p \times S_p$。

```
Prefill: prompt 5 个 token 并行进
   t1 t2 t3 t4 t5
   |  |  |  |  |
   v  v  v  v  v
  [ 计算 K/V, 写满 cache 前 5 格 ]
   K: [k1 k2 k3 k4 k5 . . . ]
   V: [v1 v2 v3 v4 v5 . . . ]
                       └ 预留增长空间
   -> 输出 token6
```

### 3.2 Decode（解码 / 增量生成）

每步只进**一个**新 token，算它的 $q,k,v$；把新 $k,v$ **追加（append）**到 cache，然后 $q$ 与 cache 里**全部** K 做注意力。这一步是**访存密集（memory-bound）**的——见 §7。

```
Decode step (生成 token7):
   只有 token6 进来
        |
        v
   算 q6,k6,v6 -> append
   K: [k1 k2 k3 k4 k5 k6 . ]   <- 新增 k6
   V: [v1 v2 v3 v4 v5 v6 . ]   <- 新增 v6
   注意力: q6 · [k1..k6]^T -> softmax -> 加权 [v1..v6]
   -> 输出 token7
```

> 两阶段的延迟指标：Prefill 决定 **TTFT**（首 token 时延），Decode 决定 **TPOT/ITL**（每 token 时延）。分离两阶段单独优化（PD 分离）是当下推理系统热点，见 [[llm-inference/README]]。

## 4. 缓存里存了什么、形状多大

每一层、每个注意力头，都要为**每个历史 token**存一份 K 向量和一份 V 向量。完整 KV Cache 张量形状：

$$\underbrace{2}_{K,V}\times \underbrace{B}_{batch}\times \underbrace{L}_{layers}\times \underbrace{H}_{heads}\times \underbrace{S}_{seq\ len}\times \underbrace{d}_{head\ dim}$$

注意 $H\times d = d_{model}$（隐藏维），所以也常写成 $2\cdot B\cdot L\cdot S\cdot d_{model}$。

```
一个 token 在一层里占的 cache:
  K 向量: [== d_model ==]   (H 个头拼起来)
  V 向量: [== d_model ==]
  共 2 * d_model 个数

整模型一个 token: 2 * L * d_model 个数
整 batch S 步:    2 * B * L * S * d_model 个数
```

**不缓存的东西**：Q（每步只用当前 token 的 q，用完即弃）、attention 分数矩阵、FFN 中间激活——它们要么不复用、要么 FlashAttention 直接在片上算掉不落地（见 [[llm-optimizer/FlashAttention]]）。

## 5. 显存公式与逐数手算

显存（字节）：

$$\text{Mem}_{KV}=2\cdot B\cdot L\cdot H\cdot S\cdot d\cdot \text{bytes}=2\cdot B\cdot L\cdot S\cdot d_{model}\cdot \text{bytes}$$

dtype 字节数：FP32=4，FP16/BF16=2，FP8/INT8=1。

### 复刻原文那笔 64GB 账（FP32）

参数：$B=32,\ H=32,\ L=32,\ d_{model}=4096,\ S=2048$，FP32（4B）。

$$2\times 32\times 4096\times 2048\times 32\times 4 = 68{,}719{,}476{,}736\ \text{字节}$$

$$\div 1024^3 = \boxed{64\ \text{GB}}$$

> 注意原式里 $H{=}32$ 和 $d_{model}{=}4096$ 同时出现，是把 $d_{model}{=}H\cdot d$ 的 $d=128$ 折进 4096 的写法（即 $4096=32\times128$，公式用的是 $d_{model}=4096$，那个 32 是 layer 不是 head 的重复，按 $2\cdot B\cdot L\cdot S\cdot d_{model}$ 读最稳）。**记忆口诀：每 token 每层 = $2\,d_{model}$ 个数。**

### 换 FP16（推理常用）

同样配置改 BF16（2B）：$64\text{GB}\times \frac{2}{4}=\textbf{32 GB}$。仅 KV Cache 就吃掉一张 A100-40G 的大半。

### 单条序列要多少（B=1，LLaMA-7B 量级）

$L=32,\ d_{model}=4096,\ S=2048$，FP16：

$$2\times 1\times 32\times 2048\times 4096\times 2 = 1{,}073{,}741{,}824\ \text{字节}=\textbf{1 GB / 条}$$

```
每条 1GB 的含义:
  batch=1, 2K上下文 -> 1 GB
  并发 24 条        -> 24 GB  (还没算 14GB 权重!)
  -> 显存很快爆,所以要 PagedAttention 精打细算
```

**口算模板**：$\text{GB}\approx \dfrac{2\cdot B\cdot L\cdot S\cdot d_{model}\cdot \text{bytes}}{10^9}$。

## 6. 为什么只缓存 K 和 V

| 张量 | 是否缓存 | 原因 |
|------|---------|------|
| $K_i,V_i$ | ✅ 缓存 | 只依赖 token $i$，未来步反复被注意力读取 |
| $Q_i$ | ❌ 不缓存 | 每步只需当前 token 的 $q$，下一步的 $q$ 是新 token 的，旧 $q$ 不再用 |
| attention 权重 | ❌ | 每步重算（且与新 $q$ 有关），FlashAttention 不落地 |
| FFN 激活 | ❌ | 不跨 token 复用 |

```
谁会被"未来"读到?
  q_t -> 只参与 step t 这一次
  k_t,v_t -> step t, t+1, t+2, ... 全被读  <-- 所以缓存它俩
```

## 7. 带宽墙：KV Cache 是访存瓶颈

Decode 阶段每生成 1 个 token，要把**整个 KV Cache 从 HBM 读一遍**做注意力，但计算量极小（一个 $q$ 对所有 $k$）。这是典型 **memory-bound**：

- 算术强度（FLOPs / 字节）低，GPU 算力闲置，瓶颈在显存带宽。
- 所以 batch 越大越好（一次读权重摊给更多请求），这正是 **continuous batching** 的动机。

```
Decode 的钱花在哪:
   计算  | ▁           (小)
   读KV  | ████████    (大, 占满带宽)
   => 提速靠"减少要读的字节": MQA/GQA/MLA/KV量化/淘汰旧token
```

> 与 FlashAttention 的关系：FlashAttention 优化的是**注意力本身的访存（不落地 $S\times S$ 矩阵）**，KV Cache 优化的是**跨步的重复计算**；二者正交、常叠加使用。见 [[llm-optimizer/FlashAttention]]。

## 8. 省显存 / 省带宽的家族

| 方法 | 思路 | KV 内存倍数 | 备注 |
|------|------|-----------|------|
| MHA（基线） | 每个头独立 K/V | ×1 | 标准多头 |
| **MQA** | 所有头共享 1 份 K/V | ÷H | Multi-Query，质量略降 |
| **GQA** | 分 $g$ 组，每组共享 K/V | ÷(H/g) | LLaMA-2/3 用，质量与速度折中 |
| **MLA** | K/V 低秩压缩到 latent 再还原 | 大幅↓ | DeepSeek，见 [[llm-algo/moe/README]] |
| **KV 量化** | K/V 存 INT8/FP8 | ÷2 / ÷4 | 见 [[llm-compression/quantization/fp8]] |
| **PagedAttention** | 分页管理、按需分配、共享前缀 | 碎片≈0 | vLLM 核心，见 [[llm-inference/vllm]] |
| **窗口/淘汰** | 只留近窗或重要 token | ↓S | StreamingLLM / H2O |

```
GQA 直觉 (H=8, g=2):
  MHA:  每头一份K/V  -> 8 份
  GQA:  头分2组,组内共享 -> 2 份   (省 4 倍)
  MQA:  全部共享     -> 1 份   (省 8 倍, 质量风险最高)
```

数值感受（7B，$L{=}32,d_{model}{=}4096,S{=}2048$, FP16, B=1）：
- MHA：1 GB / 条
- GQA（8 KV 头 vs 32 Q 头，÷4）：**0.25 GB / 条**
- INT8 量化再叠加：**0.125 GB / 条** → 同显存能多塞 8 倍并发。

> 这些优化的工程组织详见 [[llm-inference/KV-Cache优化]] 与 [[llm-optimizer/kv-cache]]。

## 数值手算：三张账本

**账本 A — "为什么会 OOM"**（LLaMA-13B，$L{=}40,d_{model}{=}5120$，FP16）

单条 4K 上下文：
$$2\times 1\times 40\times 4096\times 5120\times 2 = 3{,}355{,}443{,}200\ \text{B}\approx \textbf{3.13 GB / 条}$$
A100-40G 装下 13B 权重（FP16≈26GB）后只剩 ~14GB，**最多 ~4 条并发** 就满了。

**账本 B — 上下文翻倍**：$S$ 是线性项，2K→8K，单条 KV 从 1GB→**4GB**。长上下文场景 KV Cache 远超权重，是显存主矛盾。

**账本 C — GQA 的回报**（同账本 A，但 KV 头从 40 降到 8，÷5）：
$$3.13\ \text{GB}\div 5 \approx \textbf{0.63 GB / 条}\Rightarrow 14\text{GB} 可装 \approx 22\ 条$$
并发能力提升约 5 倍，这正是现代模型普遍采用 GQA 的原因。

## 常见问题

| 问题 | 答案 |
|------|------|
| KV Cache 会影响输出结果吗？ | 不会，数学等价；只省重复计算，logits 不变（数值上有微小浮点差异） |
| 为什么 batch 大反而更划算？ | Decode 是访存密集，读一次权重摊给更多请求，吞吐↑ |
| Prefill 和 Decode 谁慢？ | 单步 Prefill 计算重（决定 TTFT）；Decode 单步轻但步数多（决定总时延/吞吐） |
| KV Cache 和 FlashAttention 冲突吗？ | 不冲突，正交叠加：一个省跨步重算，一个省注意力中间矩阵 |
| 显存里 KV 比权重还大正常吗？ | 长上下文 + 大并发时很正常，此时 KV 是主矛盾 |
| 能边解码边丢旧 KV 吗？ | 可以（窗口注意力/StreamingLLM/H2O），但可能损失远距依赖 |
| 量化 KV 会掉点吗？ | INT8 通常几乎无损，FP8 更稳；K 比 V 更敏感，常对 K 更保守 |
| 张量并行下 KV 怎么切？ | 按 head 维切到各卡，每卡只存自己负责的头，见 [[llm-inference/大模型推理张量并行]] |

## 🔗 跳转链接

- 总图：[[00-知识地图]]
- KV 优化工程与系统：[[llm-inference/KV-Cache优化]] · [[llm-optimizer/kv-cache]]
- 注意力中间矩阵优化：[[llm-optimizer/FlashAttention]] · [[flash-attention/FlashAttention]]
- 注意力/架构地基：[[llm-algo/transformer/模型架构]] · [[llm-algo/旋转编码RoPE]] · [[llm-algo/moe/README]]
- 计算量与内存估算：[[llm-algo/FLOPs]] · [[transformer内存估算]]
- 解码与服务：[[llm-inference/解码策略]] · [[llm-inference/vllm]] · [[llm-inference/README]]
- 并行与通信：[[llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]]
- 低比特存储：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 硬件背景（带宽墙）：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]]
