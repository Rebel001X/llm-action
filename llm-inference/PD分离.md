# Prefill-Decode 分离

> 把 LLM 推理的"读题"阶段(Prefill)和"逐字写答案"阶段(Decode)拆到**不同的 GPU 池**上分别部署，各自用最合适的并行/批处理策略，让计算密集的 Prefill 不再卡住访存密集的 Decode。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/分离式推理架构]] [[llm-inference/Mooncake]] [[llm-optimizer/kv-cache]]

## 阅读地图

| 你想知道 | 看哪节 |
|---|---|
| Prefill / Decode 到底是什么 | §1 地基 |
| 为什么一个 compute-bound 一个 memory-bound | §2 Roofline |
| 合在一起跑有什么坏处 | §3 干扰 |
| 分离后各自怎么优化 | §4 分离收益 |
| KV Cache 怎么从 P 传到 D | §5 KV 传输 |
| 调度器怎么排队、怎么算 SLO | §6 调度与 SLO |
| Mooncake / DistServe 架构长啥样 | §7 系统架构 |
| 用具体数字算一遍 | §8 手算 |
| 什么时候**不该**分离 | 常见问题 |

## 0. 一句话锚点

**一次 LLM 请求 = 一次 Prefill + 很多次 Decode。** Prefill 把整段 prompt 并行喂进去算出第一个 token（吃**算力**），Decode 之后每次只算一个 token（吃**显存带宽**）。两者性格相反，硬塞进同一张卡同一个 batch 会互相拖累 —— 分离部署就是把它们**物理隔开**。

## 1. 地基：一次推理的两个阶段

### 1.1 自回归生成的两步走

给定 prompt（长度 $S$ 个 token），模型要生成 $T$ 个新 token：

```
prompt: "请用一句话解释相对论"   (S = 10 tokens)
            │
            ▼
   ┌──────────────────┐
   │  Prefill 阶段     │  一次前向，并行处理全部 S 个 token
   │  (prompt 编码)    │  产出: 第 1 个输出 token + S 个位置的 KV Cache
   └──────────────────┘
            │  生成 token #1 "相"
            ▼
   ┌──────────────────┐
   │  Decode 阶段      │  每步只输入上一个 token，算 1 个新 token
   │  (逐 token 生成)  │  每步读取并追加 KV Cache，循环 T-1 次
   └──────────────────┘
       │ "相" → "对" → "论" → ... → <eos>
       ▼
   完整回答
```

**关键非对称**：Prefill 一次处理 $S$ 个 token；Decode 每步只处理 **1** 个 token，但要重复 $T$ 次。

### 1.2 为什么 Decode 只算 1 个 token 却不慢崩？—— KV Cache

注意力需要当前 token 的 Query 去和**前面所有** token 的 Key/Value 做点积。如果不缓存，第 $t$ 步要重算前 $t-1$ 个位置的 K、V，复杂度 $O(t)$，总成本 $O(T^2)$。

**KV Cache**：Prefill 时把每个位置每层的 $K,V$ 存下来；Decode 第 $t$ 步只算**新 token** 的 $q_t,k_t,v_t$，把 $k_t,v_t$ 追加进缓存，然后 $q_t$ 和缓存里全部 $K,V$ 做注意力。于是每步只算 1 个 token 的 QKV，但要**读全量 KV Cache**。详见 [[llm-optimizer/kv-cache]]。

```
Decode 第 t 步：
  新输入 token ──► 算 q_t,k_t,v_t (只 1 个 token 的矩阵乘，小)
                       │
                       ▼
   KV Cache: [k_1..k_{t-1}]  ◄── 追加 k_t  (要把整条缓存读进来！)
             [v_1..v_{t-1}]  ◄── 追加 v_t
                       │
                       ▼
   attn = softmax(q_t · K^T) · V   ──► 输出 token #t
```

读全量 KV Cache 这件事，正是 Decode 变成"访存密集"的根源（§2）。

## 2. 核心：Prefill 是 compute-bound，Decode 是 memory-bound

### 2.1 算术强度（Arithmetic Intensity）

判断一个 kernel 是被算力卡（compute-bound）还是被带宽卡（memory-bound），看**算术强度**：

$$I = \frac{\text{FLOPs（要做的浮点运算）}}{\text{Bytes（要搬的字节数）}}\quad(\text{单位：FLOP/Byte})$$

把它和硬件的"脊点"比。以 A100 为例：算力 $\approx 312$ TFLOP/s（FP16），HBM 带宽 $\approx 2$ TB/s，脊点

$$I_{\text{ridge}}=\frac{312\times10^{12}}{2\times10^{12}}=156\ \text{FLOP/Byte}.$$

- $I > 156$ → 搬一字节能干很多活 → **compute-bound**（卡算力）。
- $I < 156$ → 搬数据搬不过来，算力闲着 → **memory-bound**（卡带宽）。

### 2.2 为什么 Prefill 是 compute-bound

Prefill 一次过 $S$ 个 token，权重矩阵 $W$（一次加载）要乘以一个 $S$ 行的激活矩阵。**权重读一次，被 $S$ 个 token 复用**，所以矩阵乘是大 GEMM（$M=S$ 维度大），算术强度高、塞满 Tensor Core。$S$ 越大越 compute-bound。

### 2.3 为什么 Decode 是 memory-bound

Decode 每步只有 **1** 个 token（batch=1 时）。权重 $W$ 还是得整块从 HBM 读进来，却只乘 1 行激活 —— 退化成 **GEMV**（矩阵×向量）。权重读一遍只服务 1 个 token，算术强度极低。

```
        FLOP/Byte
          ▲
   312T ──┤        ╱──────────  ← compute roof（算力上限）
   算力   │      ╱
          │    ╱ ← Prefill 落在这边 (I 高, 顶到算力)
          │  ╱
          │╱      ← Decode 落在这边 (I 低, 被带宽压住)
          └────────────────────►  算术强度 I
               156 (ridge)
```

**一句话**：Prefill 把 GPU 算力榨干；Decode 把 GPU 显存带宽榨干，算力大量空转。两种瓶颈完全不同 → 优化手段也完全不同 → 这是分离的**物理根因**。

## 3. 不分离的痛：Prefill 干扰 Decode

传统做法（vLLM 默认的 continuous batching）把 Prefill 和 Decode 请求**混在同一个 batch、同一张卡**上跑。问题：

### 3.1 Prefill 会"抢跑"，撑爆延迟

一个长 prompt（比如 4000 token）的 Prefill 是个大 GEMM，占满一次 forward 几十毫秒。期间所有正在 Decode 的请求都得**等这一拍**结束才能出下一个 token。

```
时间轴 (同一张卡, 混合 batch):
Decode-A: ·│  停  停  停  │· · ·   ← 卡住，用户看到字"卡顿"
Decode-B: ·│  停  停  停  │· · ·
Prefill-X:  │██████████████│        ← 一个长 prompt 进来，独占了这几拍
            ▲                ▲
         X 到达           X 算完
         A,B 的 TPOT 被拉长（卡顿）
```

- **TTFT**（Time To First Token，首 token 延迟）：被前面排队的 Decode 拖慢。
- **TPOT**（Time Per Output Token，每 token 延迟）：被插进来的 Prefill 拖慢，输出**忽快忽慢**。

### 3.2 一张卡没法同时讨好两边

- 想让 Prefill 快 → 用大并行度（如 TP=8）摊薄长 prompt 的算力。
- 想让 Decode 省 → Decode 是 memory-bound，TP 太大反而因通信开销和 KV 切分而低效，更想**大 batch** 把权重读取成本摊给更多请求。

同一张卡的并行配置只能取一个，必然两头不讨好。

## 4. 分离后的收益：各自最优

把集群分成两个池子：**Prefill 池**（P 节点）和 **Decode 池**（D 节点）。

```
            请求
             │
        ┌────▼─────┐    KV Cache (一次性大块)
        │  P 池     │ ═════════════════════►  ┌──────────┐
        │ Prefill   │   高带宽传输 (RDMA)      │  D 池    │
        │ compute   │                          │ Decode   │
        │ -bound    │                          │ memory   │
        │ TP 大/批小 │                          │ -bound   │
        └──────────┘                          │ 大 batch │
                                              └────┬─────┘
                                                   │ 逐 token 流式返回
                                                   ▼
                                                 用户
```

各自独立优化（互不干扰）：

| 维度 | Prefill 池 | Decode 池 |
|---|---|---|
| 瓶颈 | 算力 | 显存带宽 + 容量 |
| 并行策略 | 偏大 TP，摊长 prompt 算力 | 偏大 batch / 适度 TP，摊权重读取 |
| 批处理 | chunked-prefill，控制单批长度 | continuous batching，越多并发越省 |
| 硬件取向 | 算力强的卡（高 TFLOPS） | 带宽/显存大的卡 |
| 扩缩容 | 随 prompt 长度/QPS 独立扩 | 随生成长度/并发独立扩 |

**互不干扰**是最大红利：Prefill 排队不再卡 Decode 的 TPOT，Decode 大 batch 不再拖 Prefill 的 TTFT。两类 SLO 可以**分别满足、分别扩容**（一个被打满只扩那一池，不用整体加卡）。

## 5. KV Cache 如何从 P 传到 D

分离的代价：Prefill 在 P 节点算出的 KV Cache，必须搬到 D 节点才能继续生成。这是分离架构的**核心工程难点**。

### 5.1 要搬多大？

每个 token、每层、K 和 V 各一份。单请求 KV Cache 大小：

$$\text{KV bytes} = 2\,(\text{K,V}) \times L\,(\text{层}) \times S\,(\text{token}) \times H_{kv}\times d_{head}\times b$$

其中 $L$ 层数，$H_{kv}$ 是 **KV 头数**（GQA 下远小于 Q 头数），$d_{head}$ 头维度，$b$ 每元素字节（FP16=2）。详细推导见 [[llm-optimizer/kv-cache]]，§8 会代入数字。

### 5.2 传输路径与机制

```
P 节点 GPU 显存 (KV Cache)
        │  ① 不经 CPU，GPU↔GPU 直传
        ▼
   ┌─────────────────────────────┐
   │ 节点内: NVLink (~数百 GB/s)   │
   │ 节点间: RDMA/RoCE over IB     │  GPUDirect, 绕开 CPU 拷贝
   └─────────────────────────────┘
        │
        ▼
D 节点 GPU 显存 (写入对应 paged KV 槽位)
```

关键技术点：
- **GPUDirect RDMA**：网卡直接读写 GPU 显存，**绕开 CPU 内存中转**，避免两次拷贝。
- **分层流水（layer-wise pipelining）**：不必等全部 $L$ 层算完才传。第 $\ell$ 层 Prefill 一算完就开始传第 $\ell$ 层 KV，**计算与传输重叠**，把传输延迟藏进 Prefill 时间里。
- **Paged 布局对齐**：P 和 D 两侧都用分页 KV（vLLM 风格），按块（block）传，方便寻址和复用。

### 5.3 传输 vs 重算的权衡

为什么不在 D 节点**重新 Prefill** 一遍省得传？因为重算要再花一次完整 Prefill 算力（compute-bound，贵），而传输只是搬数据（可被网络带宽和流水重叠摊掉）。**长 prompt 时传输划算；prompt 极短时重算可能更省**（搬运 overhead 反而占比高）。系统会按长度选择策略。

## 6. 调度与 SLO

### 6.1 两个核心 SLO 指标

| 指标 | 含义 | 主要由谁决定 | 用户体感 |
|---|---|---|---|
| **TTFT** | 从请求到达到吐出**第一个** token 的时间 | Prefill 时延 + 排队 | "等了多久才开始回答" |
| **TPOT** | 生成阶段**平均每个** token 的间隔（也叫 ITL） | Decode 每步时延 | "回答出字流不流畅" |

端到端延迟近似：

$$\text{Latency} \approx \text{TTFT} + (T-1)\times \text{TPOT}.$$

吞吐则常用每秒输出 token 数（throughput）衡量。**SLO 通常写成 P99 形式**，如"TTFT P99 < 500 ms 且 TPOT P99 < 50 ms"。

### 6.2 分离让 SLO 可"分别治理"

合部时 TTFT 和 TPOT 互相打架（§3）。分离后：
- **P 池**只对 TTFT 负责 → 想压 TTFT 就加 P 节点、或限制单批 prefill 长度（chunked prefill）。
- **D 池**只对 TPOT 负责 → 想压 TPOT 就控制 D 的 batch 大小（batch 越大单步越慢但吞吐越高，需在 TPOT 约束下找最大 batch）。

### 6.3 调度器要做的事

```
        Router / Scheduler
   ┌──────────────────────────────────┐
   │ 1. 按 TTFT SLO 把新请求派给空闲 P  │
   │ 2. P 算完 → 触发 KV 传输 → 入 D 队 │
   │ 3. D 在 TPOT 约束下凑最大 batch     │
   │ 4. 负载均衡: P/D 各自独立扩缩容     │
   │ 5. Prefix 命中: 复用已有 KV，跳过 P │
   └──────────────────────────────────┘
```

- **Chunked Prefill**：把超长 prompt 切成小块分多拍算，避免一个长 prompt 独占太久把 TTFT 拖爆，也便于和别的请求交错。
- **Prefix Caching**：相同前缀（如同一 system prompt）的 KV 可跨请求复用，命中就**整段跳过 Prefill**，TTFT 骤降。
- **负载感知路由**：哪边热扩哪边。Prefill 重的负载（长 prompt、RAG）多加 P；长输出、高并发多加 D。

## 7. 系统架构：DistServe / Mooncake

### 7.1 思想脉络

- **DistServe**（学术界代表）：首次系统化论证"P/D 分离 + 各自独立选并行配置"能在满足 TTFT/TPOT SLO 下显著提升单卡有效吞吐。核心：**goodput**（满足 SLO 前提下的吞吐）而非裸吞吐。
- **Mooncake**（Kimi 的生产系统）：**以 KVCache 为中心**的分离架构，把分散在各节点的 CPU/DRAM/SSD 聚成一个**全局 KV Cache 池**，最大化 prefix 复用，并用调度在过载时做取舍。详见 [[llm-inference/Mooncake]]。

### 7.2 Mooncake 式架构（KVCache 为中心）

```
 ┌─────────────────────────────────────────────┐
 │            Conductor / 全局调度               │
 │   (按 SLO 派单, 决定复用/重算, 过载丢弃)       │
 └───────┬───────────────────────────┬──────────┘
         │                           │
   ┌─────▼──────┐              ┌──────▼─────┐
   │ Prefill 集群 │  KV (RDMA)  │ Decode 集群 │
   │  (P 节点)   │ ═══════════►│  (D 节点)   │
   └─────┬──────┘              └──────┬─────┘
         │   读/写                     │  读/写
         ▼                            ▼
   ╔═══════════════════════════════════════════╗
   ║   全局 KVCache Pool                         ║
   ║   GPU-HBM  ◄►  CPU-DRAM  ◄►  SSD            ║
   ║   (分级存储, prefix 跨请求/跨节点复用)        ║
   ╚═══════════════════════════════════════════╝
```

要点（稳定思想，具体参数以官方文档为准）：
1. **KVCache 池化分级**：热的留 HBM，温的进 DRAM，冷的落 SSD；prefix 命中率越高，省下的 Prefill 越多。
2. **调度即取舍**：过载时宁可**早拒**（reject）也别让所有请求都破 SLO —— 保住已接收请求的 TTFT/TPOT。
3. **传输与计算重叠**：layer-wise 流水（§5.2），把 KV 搬运藏进 Prefill。

> 国内方案与生态可对照 [[llm-inference/分离式推理架构]]；大规模 EP/部署的工程实践参考 https://lmsys.org/blog/2025-05-05-large-scale-ep/ 。

## 8. 数值示例（逐数手算）

设定一个 7B 级模型，便于手算：

| 符号 | 含义 | 取值 |
|---|---|---|
| $L$ | 层数 | 32 |
| $d$ | 隐藏维 | 4096 |
| $P$ | 参数量 | 7e9 |
| $H_{kv}$ | KV 头数（GQA） | 8 |
| $d_{head}$ | 头维度 | 128 |
| $b$ | 字节/元素（FP16） | 2 |
| $S$ | prompt 长度 | 2048 |
| 硬件 | A100 | 312 TFLOP/s，2 TB/s |

### 8.1 Prefill 的算术强度（验证 compute-bound）

FLOPs 经验公式：一次前向 $\approx 2 P S$（每参数每 token 约 2 次 MAC 浮点运算）。

$$\text{FLOPs}_{pf}=2\times7\text{e}9\times2048\approx2.87\text{e}13.$$

需搬的字节主体是权重读取（FP16）：$\text{Bytes}\approx P\times2=1.4\text{e}10$。

$$I_{pf}=\frac{2.87\text{e}13}{1.4\text{e}10}\approx 2048\ \text{FLOP/Byte}\;\gg\;156.$$

> 远超脊点 156 → **强 compute-bound**。直觉：权重读一次，被 2048 个 token 复用，所以 $I$ 大致正比于 $S$。

理想算时间 $\approx 2.87\text{e}13 / 3.12\text{e}14 \approx 92\ \text{ms}$ → 这就是 **TTFT** 量级（再加排队）。

### 8.2 Decode 单步的算术强度（验证 memory-bound）

batch=1，每步 1 个 token：FLOPs $\approx 2P\times1=1.4\text{e}10$。

搬的字节：要把全部权重读一遍（$1.4\text{e}10$）+ 读 KV Cache（相对小）。取主体 $\approx1.4\text{e}10$。

$$I_{dec}=\frac{1.4\text{e}10}{1.4\text{e}10}\approx 2\ \text{FLOP/Byte}\;\ll\;156.$$

> 远低于脊点 → **强 memory-bound**。算力 312T 几乎全闲。

单步理想时间被**带宽**决定：$1.4\text{e}10\ \text{B} / 2\text{e}12\ \text{B/s}\approx 7\ \text{ms}$ → 这就是 **TPOT** 量级。

**结论**：同一硬件，Prefill 顶算力（92 ms 受算力限），Decode 顶带宽（7 ms 受带宽限），两者瓶颈差一个数量级的"维度"，正是分离的依据。

### 8.3 大 batch 为何救 Decode

Decode 把 $N$ 个请求凑成一批：权重还是**只读一遍**，却服务 $N$ 个 token。算术强度变成

$$I_{dec}(N)\approx\frac{2P\cdot N}{2P}=N.$$

取 $N=64$ → $I\approx64$，逼近脊点；继续增大就 compute-bound 化、把带宽利用率打满。**这就是 D 池要大 batch 的数学原因**（受 TPOT 上限和显存约束封顶）。

### 8.4 要传多大的 KV Cache（P→D）

代入 §5.1 公式（$S=2048$）：

$$\text{KV}=2\times L\times S\times H_{kv}\times d_{head}\times b$$
$$=2\times32\times2048\times8\times128\times2$$
$$=2\times32\times2048\times2048\;\text{B}\quad(8\times128=1024,\ \times2b=2048)$$
$$=2.68\text{e}8\ \text{B}\approx 256\ \text{MiB}.$$

用 RDMA 100 GB/s 传：$2.68\text{e}8 / 1\text{e}11 \approx 2.7\ \text{ms}$。

> 对照 §8.1 的 Prefill 92 ms：传输 2.7 ms 只占 ~3%，且可被 layer-wise 流水进一步重叠隐藏 → **传 KV 远比在 D 重算一遍 Prefill（又 92 ms）划算**。这就是分离能成立的成本账。
>
> 注：若是 MHA（$H_{kv}=32$ 而非 GQA 的 8），KV 会大 4 倍 ≈ 1 GiB，传输也涨到 ~10 ms —— GQA 直接让分离的传输代价降一个量级。

## 对照/复杂度表

| 维度 | Prefill | Decode |
|---|---|---|
| 一次处理 token 数 | $S$（整段 prompt） | 1（×$T$ 步，或 batch=$N$） |
| 计算形态 | 大 GEMM | GEMV / 小 GEMM |
| 算术强度 $I$（例） | ~2048 | ~2（batch=1） |
| 瓶颈 | **算力**（compute-bound） | **带宽**（memory-bound） |
| 主导 SLO | TTFT | TPOT |
| 偏好并行 | 大 TP | 大 batch + 适度 TP |
| 喜欢的硬件 | 高 TFLOPS | 高带宽 / 大显存 |
| 时延量级（7B@A100） | ~92 ms/次 | ~7 ms/步 |

## 常见问题

| 疑问 | 真相 |
|---|---|
| 分离一定更快吗？ | 不一定。短输出、低并发、prompt 短时，KV 传输 overhead 可能盖过收益，**合部反而更简单高效**。分离在"高并发 + 长上下文 + 严格 SLO"下才显著赢。 |
| KV 传输会不会成为新瓶颈？ | 有这个风险。靠 GQA/MQA 缩小 KV、RDMA/NVLink 高带宽、layer-wise 流水重叠来压住（§8.4 算过 ~3%）。 |
| chunked prefill 和分离冲突吗？ | 不冲突，常一起用。chunked 在 P 池内把长 prompt 切块控制单批时延，分离是把 P/D 跨池隔开，两者正交。 |
| Decode 大 batch 会不会拖慢 TPOT？ | 会。batch 越大单步越慢，所以要在 **TPOT 上限内**取最大 batch，是个带约束的优化。 |
| 为什么不直接在 D 上重算 Prefill？ | 重算要再花一次完整 compute-bound Prefill（贵）；传 KV 只是搬数据且能流水重叠（便宜）。长 prompt 必传，极短 prompt 才考虑重算。 |
| Prefix Caching 和分离什么关系？ | 互补。命中前缀就整段**跳过 Prefill**，连 KV 都不用重传（直接从全局池取），是 Mooncake 池化的最大价值之一。 |
| 分离需要多机吗？ | 不必须。单机内也能"角色分离"（不同卡当 P/当 D），但跨机 + 全局 KV 池才能发挥扩缩容和复用的全部威力。 |

## 🔗 跳转链接

- [[00-知识地图]] — 总览与导航入口
- [[llm-inference/分离式推理架构]] — 分离式推理的总体范式与国内外方案
- [[llm-inference/Mooncake]] — 以 KVCache 为中心的生产级分离系统细节
- [[llm-optimizer/kv-cache]] — KV Cache 的原理、显存占用与分页管理
