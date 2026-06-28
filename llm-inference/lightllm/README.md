# LightLLM

> 一句话定位：LightLLM 是一个 **纯 Python、超轻量** 的 LLM 推理与服务框架——靠 **三进程异步协作 + token 级动态调度 + TokenAttention（无碎片 KV 管理）** 把高性能做进一个易读、易改、易扩展的代码库里。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/vllm/README]] [[llm-inference/README]]

## 阅读地图

| 节 | 你会搞懂 | 一句话 |
|----|---------|--------|
| 0 | 一句话锚点 | LightLLM = 三进程 + token 调度 + TokenAttention |
| 1 | 它解决什么问题 | 为什么需要"又轻又快"的纯 Python 框架 |
| 2 | KV-Cache 复习 | 自回归为何必须缓存、为何浪费显存 |
| 3 | 三进程异步架构 | HttpServer / Router / Model 各管什么 |
| 4 | 请求生命周期 | 一条请求从进到出走哪条路 |
| 5 | token-level 调度 | 不按"整批"走，按"逐 token"决策 |
| 6 | TokenAttention | 以 token 为粒度管 KV，零碎片 |
| 7 | Efficient Router | 调度器如何防 OOM、最大化并发 |
| 8 | 整体架构图 | 各组件如何串起来 |
| 9 | 数值例子 | 显存 / 碎片 / 并发手算 |
| 10 | 配置示例 | 启动参数含义与权衡 |
| 11 | 对照表 | vs vLLM / TGI / HF |
| 12 | 何时选 LightLLM | 决策清单 |

---

## 0. 一句话锚点

> **LightLLM 用"纯 Python + 三个解耦进程 + 以单个 token 为最小调度/存储单位"这套组合，做到了既轻量又高吞吐。**

把三个支柱刻进脑子：

```
              ┌──────────────────────────────────────┐
              │  ① 三进程异步   (解耦, 不互相阻塞)      │
   LightLLM = │  ② token-level 调度 (逐 token 决策)    │
              │  ③ TokenAttention (以 token 管 KV)    │
              └──────────────────────────────────────┘
                  目标：纯 Python 也能跑出高吞吐
```

- **轻量**：核心调度/服务逻辑是纯 Python，CUDA 重活交给 Triton/算子库，读代码、加功能、做实验门槛低。
- **高性能**：靠 token 级调度 + 无碎片显存把并发塞满。
- **可扩展**：模型实现模块化，加新模型/新算子相对容易。

> ⚠️ 本文讲**机制与思路**。精确的 CLI 默认值、版本号、API 签名以**官方文档 / 源码为准**（见底部链接）。

---

## 1. 地基：它解决什么问题

LLM 推理服务有一组绕不开的矛盾：

1. **吞吐 vs 延迟**：想吞吐高就要大 batch，但大 batch 会拖慢单条请求的首 token 延迟。
2. **显存利用 vs 安全**：想塞更多并发就要把显存用满，但用满又容易在某条请求突然变长时 OOM。
3. **性能 vs 可维护性**：很多高性能框架为了快，把核心逻辑写进 C++/CUDA，结果代码难读、难改、难做研究实验。

LightLLM 的取舍是：**用纯 Python 写控制面（服务、调度、显存账本），把计算面（attention/matmul）下沉到 Triton 等高性能 kernel**。这样既保住可读性，又不牺牲关键路径性能。

```
   传统高性能框架                      LightLLM
   ┌───────────────┐                  ┌───────────────┐
   │ 控制面 C++/CUDA│  难读难改         │ 控制面  Python │  易读易改
   ├───────────────┤                  ├───────────────┤
   │ 计算面 CUDA    │  快              │ 计算面 Triton  │  也快
   └───────────────┘                  └───────────────┘
```

**核心洞察**：推理服务的瓶颈往往不在"单个 kernel 多快"，而在**调度与显存怎么管**。把这两件事做对，纯 Python 一样能打。

---

## 2. 前置复习：KV-Cache 为什么存在、为什么浪费

自回归生成：第 $t$ 步要对前面所有 token 做注意力。每个 token 的 Key/Value 只依赖它自己，**算一次就能复用**，于是缓存起来——这就是 KV-Cache。

第 $t$ 步的注意力（单头，省略缩放）：

$$\text{Attn}_t = \text{softmax}\!\left(\frac{q_t \, K_{1:t}^\top}{\sqrt{d}}\right) V_{1:t}$$

其中 $K_{1:t}, V_{1:t}$ 就是缓存。**没有缓存**，每步都要重算前面所有 token 的 K/V，复杂度从 $O(t)$ 退化到 $O(t^2)$。

**单 token 的 KV 显存**（理解量级用）：

$$\text{bytes/token} = 2 \times L \times H_{kv} \times d_{head} \times \text{dtype}$$

- $2$：K 和 V 各一份
- $L$：层数，$H_{kv}$：KV 头数，$d_{head}$：每头维度
- dtype：fp16 = 2 字节

**浪费从哪来？** 很多框架按"请求 × 最大长度"预留**整块连续显存**。一条请求设 max_len=2048 但实际只生成 100 token，那 1948 个 token 的显存就被占着空转。并发一多，这种**内部碎片**能吃掉一大半显存——这正是 TokenAttention 要消灭的目标。

---

## 3. 三进程异步架构（核心）

LightLLM 把服务拆成**多个独立进程**，用异步消息互通，互不阻塞。概念上的三类角色：

```
   ┌──────────────┐  请求   ┌──────────────┐  调度好的批  ┌──────────────┐
   │ HttpServer   │ ──────▶ │   Router     │ ──────────▶ │ Model (推理)  │
   │ (接入/出口)   │ ◀────── │  (调度大脑)   │ ◀────────── │  GPU 上跑前向  │
   └──────────────┘  结果   └──────────────┘  生成的token └──────────────┘
        ▲                                                      │
        │  tokenize / detokenize 也常拆为独立进程               │
        └──────────────────────────────────────────────────────┘
```

三类角色各管一摊：

| 进程/角色 | 职责 | 为什么独立 |
|-----------|------|-----------|
| **HttpServer** | 收 HTTP 请求、做 tokenize、把请求塞进队列；流式把生成的 token detokenize 回吐给用户 | I/O 密集，独立后不拖累 GPU |
| **Router（调度器）** | 维护"在跑/等待"队列，做 **token 级调度**，决定下一步哪些请求进 batch，管显存账本 | 调度是大脑，必须独占一个事件循环 |
| **Model（推理后端）** | 在 GPU 上执行前向，跑 attention/matmul kernel，做张量并行 | 计算密集，独占 GPU 资源 |

**为什么要拆进程而不是多线程？**

- Python 有 GIL：多线程在 CPU 密集环节（tokenize/detokenize/调度逻辑）会互相抢锁，**真并行不了**。多进程绕开 GIL。
- **流水线重叠**：HttpServer 在 detokenize 第 $i$ 个请求结果的同时，Router 已经在调度第 $i+1$ 批，Model 在跑前向——三段流水并行，GPU 不空等。

```
   时间轴 →
   HttpServer:  [tok req3] [detok res1] [detok res2] ...
   Router:      [sched b1] [sched b2 ] [sched b3 ] ...
   Model(GPU):  [fwd b0  ] [fwd b1   ] [fwd b2   ] ...   ← GPU 几乎不空闲
```

进程间通过异步队列/RPC（如 ZMQ 之类的消息机制，具体以源码为准）传消息，谁也不卡谁。

---

## 4. 请求生命周期（一条请求走哪条路）

```
 用户
  │ POST /generate {"inputs":"...", "parameters":{...}}
  ▼
 ┌─────────────┐
 │ HttpServer  │ 1. 解析参数  2. tokenize → token ids
 └─────────────┘ 3. 封装为 Req，放入等待队列
  │
  ▼
 ┌─────────────┐  4. token 级调度：决定本步把哪些 Req 组成 batch
 │   Router    │  5. 为新进来的 Req 的 prompt token 申请 KV 槽位
 └─────────────┘  6. 把 batch 下发给 Model
  │
  ▼
 ┌─────────────┐  7. prefill / decode 前向，产出每条 Req 的下一个 token
 │   Model     │  8. 把新 token 写回各自的 KV 槽位
 └─────────────┘  9. 返回新 token 给 Router
  │
  ▼
 ┌─────────────┐ 10. 判断每条 Req 是否结束(EOS / 达 max_len)
 │   Router    │ 11. 未结束→留在运行队列下一步继续; 结束→释放其全部 KV 槽位
 └─────────────┘
  │  新 token (流式)
  ▼
 ┌─────────────┐ 12. detokenize → 文本片段，SSE/流式回吐用户
 │ HttpServer  │
 └─────────────┘
  │
  ▼  循环 7~12 直到该 Req 结束
 用户逐字看到输出
```

关键点：**步骤 4 每个 decode 步都重新决策**（谁进 batch、能否再放新请求），不是"一批跑到底"。这就是下一节的 token-level 调度。

---

## 5. token-level（逐 token）调度

对比两种调度粒度：

```
   静态批处理(Static Batching)：
   批内所有请求一起开始、一起结束 → 短的等长的，GPU 利用率低
   req短 ████░░░░░░░░░░  ← 早就生成完, 却卡着等
   req长 ██████████████
                       ↑ 整批才能释放

   token-level / 连续批处理(Continuous / iteration-level)：
   每生成一个 token 就重新组批；谁完成谁立刻退出, 空位马上补新请求
   step:  t0 t1 t2 t3 t4 t5 ...
   reqA   █  █  █  ✓
   reqB   █  █  █  █  █  ✓
   reqC          █  █  █  █   ← B/A 退出后立刻插进来
```

**token-level 调度做什么决策（每个迭代步）：**

1. 运行队列里的请求，各生成一个 token（decode 步）。
2. 检查谁触发 EOS / 达 max_new_tokens → **立刻结束并释放 KV**。
3. 看显存还剩多少 → 能否从等待队列**捞新请求进来**做 prefill。
4. 若显存吃紧 → 暂停/排队部分请求，避免 OOM。

**好处**：

- 短请求不被长请求"绑架"，**首 token 延迟低**。
- 一有请求结束，空出的算力和显存立刻被填满，**吞吐高**。
- 与 TokenAttention 配合：因为 KV 是按 token 管的，释放/补位都是 token 粒度，**精准无浪费**。

---

## 6. TokenAttention（以 token 为粒度管 KV）

这是 LightLLM 的招牌。核心思想一句话：**KV-Cache 的分配/回收单位是"一个 token"，不是"一整条请求的最大长度"，也不是"一个固定大小的页/块"。**

```
   朴素(按最大长度预留)：           TokenAttention(按 token 占用)：
   ┌──────────────────────┐        共享 KV 池 (一格 = 一个 token 的 KV)
   reqA │用██░░░░░░░░░░░░│ 浪费     ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
   reqB │用████░░░░░░░░░░│ 浪费     │A│A│B│B│B│C│ │ │ │ │ │ │ │
   reqC │用█░░░░░░░░░░░░░│ 浪费     └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
   └──────────────────────┘         用一个 Token Table 记 "每条请求的
   每条都占满 max_len 的连续块         token i → 池中第几格", 用多少占多少
```

**它怎么工作：**

- 维护一个**全局 KV 池**：把 GPU 显存切成大量"一个 token 的 KV"大小的槽位。
- 每条请求一张 **Token Table（索引表）**：记录"我的第 $i$ 个 token 的 KV 存在池中哪一格"。
- 请求每生成一个新 token → 从池里**申请一格**、填进去、在 Token Table 里记一笔。
- 请求结束 → **整条的格子一次性归还池子**，立刻可被别人用。

**和 vLLM 的 PagedAttention 区别在哪？**

| 维度 | PagedAttention(vLLM) | TokenAttention(LightLLM) |
|------|----------------------|--------------------------|
| 管理单位 | **block/page**（如 16 个 token 一块） | **单个 token** |
| 内部碎片 | 一块没填满会有**块内碎片**（如只用 3/16） | 单 token 粒度 → **几乎零碎片** |
| 索引结构 | block table（请求→块号） | token table（请求→token 槽位） |
| kernel | 自定义 paged attention kernel | 配套的 token-wise attention kernel |
| 取舍 | 块大 → 索引开销小但碎片大 | 粒度细 → 碎片几乎为零，索引更细 |

**为什么粒度到 token 能更省？** PagedAttention 把碎片从"请求级"降到"块内"，但块没填满仍浪费（一个 16-token 块只用 1 个，浪费 15）。TokenAttention 直接降到 token，**用多少占多少**，理论上把内部碎片压到接近 0，于是同样显存能塞下更多并发。

> attention 计算本身需要"非连续的 KV 按 token table 聚起来算"，因此 TokenAttention 配套了能直接读 token 池的 attention kernel（基于 Triton 实现，细节以源码为准）。

---

## 7. Efficient Router（调度器如何防 OOM、最大化并发）

Router 是把第 5 节（token 调度）和第 6 节（token 显存账本）拼起来的大脑。它最难的活：**在"尽量塞满显存"和"绝不 OOM"之间走钢丝**。

风险场景：现在显存还够，但运行中的请求都还要生成很多 token，再过几步可能集体变长 → 突然爆显存。

```
   现在:  KV池占用 ████████░░░░  (够)
   再跑5步, 每条都 +5 token:
          KV池占用 ████████████  (爆!) ← Router 必须提前预判
```

**Efficient Router 的做法（思路层面）：**

- 维护一个**显存占用的预测/上界估计**：不仅看"现在用了多少"，还估"按当前在跑的请求继续生成，未来若干步最多会用多少"。
- 只有当**预测的峰值仍在安全水位内**，才放新请求进来做 prefill。
- 显存吃紧时，可对部分请求做**排队/暂停**（必要时换出，细节以版本为准），优先保证已在跑的请求不被 OOM 打断。

这套预测式准入，让 LightLLM 能把显存用到很满又稳，是高吞吐的关键一环。

```
   等待队列 ──┐
             ▼
        ┌──────────────────────────┐
        │  Router 准入判断:          │
        │  预测未来峰值 ≤ 显存上限?   │
        │   是→放行做prefill          │
        │   否→留在等待队列           │
        └──────────────────────────┘
             │放行
             ▼
        运行队列(每步 token 调度) ──▶ Model
```

---

## 8. 整体架构图（把所有组件串起来）

```
                              用户 / 客户端
                                   │ HTTP (流式)
        ┌──────────────────────────┼───────────────────────────┐
        │                          ▼                            │
        │                 ┌──────────────────┐                  │
        │                 │   HttpServer     │  tokenize        │
        │                 │  (接入 + 出口)    │  detokenize      │
        │                 └────────┬─────────┘                  │
        │           入队 Req ▲     │  新token流式回吐 ▲           │
        │                   │      ▼                 │          │
        │                 ┌──────────────────────────┴───┐      │
        │                 │         Router (调度大脑)      │      │
        │                 │  ┌────────────────────────┐   │      │
        │                 │  │ token-level 调度循环    │   │      │
        │                 │  │ Efficient 准入(防OOM)   │   │      │
        │                 │  │ Token Table / KV 账本   │   │      │
        │                 │  └────────────────────────┘   │      │
        │                 └────────┬─────────────▲────────┘      │
        │            下发 batch     │             │ 新token        │
        │                          ▼             │               │
        │                 ┌──────────────────────┴───┐           │
        │                 │     Model Backend (GPU)   │          │
        │                 │  prefill / decode 前向     │          │
        │                 │  TokenAttention kernel     │          │
        │                 │  Triton 算子 / 张量并行(TP) │          │
        │                 │  ┌──────────────────────┐  │          │
        │                 │  │  全局 KV Token 池      │  │          │
        │                 │  │ ┌─┬─┬─┬─┬─┬─┬─┬─┬─┐  │  │          │
        │                 │  │ │ │ │ │ │ │ │ │ │ │  │  │          │
        │                 │  │ └─┴─┴─┴─┴─┴─┴─┴─┴─┘  │  │          │
        │                 │  └──────────────────────┘  │          │
        │                 └────────────────────────────┘          │
        │                   进程边界 = 异步消息(不阻塞)            │
        └─────────────────────────────────────────────────────────┘
```

要点回顾：**控制面（HttpServer/Router）纯 Python、多进程异步**；**计算面（Model）GPU + Triton kernel + TokenAttention 的 token 池**。

---

## 9. 数值例子（显存 / 碎片 / 并发手算）

设一个模型：层数 $L=32$，KV 头数 $H_{kv}=8$，每头维度 $d_{head}=128$，fp16（2 字节）。

**单 token KV 显存：**

$$2 \times 32 \times 8 \times 128 \times 2 = 131072 \text{ 字节} \approx 128 \text{ KB/token}$$

**一条 512-token 上下文的请求：** $512 \times 128\text{KB} = 64\text{ MB}$。

设可用于 KV 的显存为 $40$ GB：

**(a) 朴素按 max_len=2048 预留：** 每条预留 $2048 \times 128\text{KB} = 256\text{ MB}$。

$$\frac{40\text{ GB}}{256\text{ MB}} \approx 160 \text{ 条并发}$$

但若实际平均只生成 512 token，真正用到 $64$MB，**浪费 $(256-64)/256 = 75\%$**。

**(b) TokenAttention 按实际 token 占用：** 平均每条真用 $64$MB。

$$\frac{40\text{ GB}}{64\text{ MB}} \approx 640 \text{ 条并发}$$

**并发约 4 倍**，因为不再为"没用到的 max_len 尾巴"埋单。

**(c) 对比块粒度（如 16 token/块）的块内碎片：** 一条 512 token 正好 32 块，几乎无碎片；但若是 513 token，则要 33 块，最后一块只用 $1/16$，浪费 15 个 token 的 KV。请求越零碎、长度越不对齐，块内碎片越明显——TokenAttention 的 token 粒度直接消除这部分。

> 数字为帮助建立量级直觉，真实占用还含权重、激活、kernel 工作区等，以实测为准。

---

## 10. 配置示例（启动参数含义与权衡）

下面是**示意性**启动命令，参数名/默认值**以官方文档为准**，重点理解每项在调什么：

```bash
python -m lightllm.server.api_server \
    --model_dir /path/to/model \
    --tp 2 \
    --max_total_token_num 60000 \
    --max_req_input_len 4096 \
    --max_req_total_len 8192 \
    --running_max_req_size 256 \
    --host 0.0.0.0 --port 8000
```

| 参数(示意) | 含义 | 权衡 |
|-----------|------|------|
| `--model_dir` | 模型权重目录 | — |
| `--tp` | 张量并行度（用几张卡切模型） | 大模型放不下时调大；通信开销随之上升 |
| `--max_total_token_num` | **KV token 池总容量**（全局能缓存多少 token 的 KV） | 调大→并发更高但占显存更多；这是 TokenAttention 池的尺寸，最关键的吞吐旋钮 |
| `--max_req_input_len` | 单请求最大输入(prompt) token 数 | 控制超长 prompt，防单条吃爆 |
| `--max_req_total_len` | 单请求输入+输出总上限 | 限制最坏占用，配合 Router 防 OOM |
| `--running_max_req_size` | 同时在跑的最大请求数 | 限制并发上界，保护延迟与显存 |
| `--host/--port` | 服务地址 | — |

**调参直觉：**

- 想**更高吞吐** → 加大 `max_total_token_num`（在显存允许内把池子做大），并放宽 `running_max_req_size`。
- 怕 **OOM / 想稳延迟** → 收紧 `max_req_total_len` 与 `running_max_req_size`，让 Router 的准入更保守。
- **显存预算**粗算：池容量 × 单 token KV 字节 ≈ KV 占用（再留余量给权重/激活）。

> 其余如量化、并行细节、是否开某些 kernel 优化，请查当前版本文档；**不要照搬本文具体数字到生产**。

---

## 11. 对照表（vs vLLM / TGI / HF）

| 维度 | HF Transformers | TGI | vLLM | **LightLLM** |
|------|-----------------|-----|------|--------------|
| 定位 | 通用建模库 | 生产服务 | 高吞吐引擎 | **轻量高性能引擎** |
| 主语言/可读性 | Python | Rust+Python | Python+CUDA | **纯 Python（最易读改）** |
| KV 管理 | 连续预留 | 连续/分页 | **PagedAttention（块）** | **TokenAttention（token）** |
| 批处理 | 静态 | 连续批 | 连续批 | **token-level 连续批** |
| 内部碎片 | 大 | 中 | 小（块内仍有） | **近乎零** |
| 进程模型 | 单 | 服务化 | 引擎+服务 | **三进程异步解耦** |
| 适合谁 | 调研/小规模 | 工程部署 | 大规模高吞吐 | **研究/快速迭代 + 高吞吐** |

一句话区分：**vLLM 把 KV 当"分页虚拟内存"（块粒度）；LightLLM 把 KV 管到"单 token"粒度，且整个控制面纯 Python 易改。**

---

## 12. 何时选 LightLLM（决策清单）

✅ **适合：**

- 想要**高吞吐**，又希望框架**纯 Python、能读懂能改**（做推理研究、改调度策略、试新 attention 变体）。
- 关注**显存利用率到极致**：长度差异大、并发高的在线场景，TokenAttention 的零碎片优势明显。
- 想要**轻量依赖、快速上手**做实验或中小规模服务。
- 需要 token 级精细调度（短长请求混跑、低首 token 延迟）。

⚠️ **要权衡 / 可能另选：**

- 生态/社区规模、特定模型支持、某些前沿优化的成熟度，需对比当前版本（vLLM/SGLang 等也在快速演进）。
- 超大规模、对某些分布式/PD 分离特性有强需求时，按需对比各框架最新能力（见 [[llm-inference/PD分离]] 等）。
- 一切**精确性能数字**以你自己的硬件 + 当前版本**实测**为准。

**记忆口诀：** *轻（纯 Python）· 细（到 token）· 异（三进程）· 准（防 OOM 调度）。*

---

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- vLLM（PagedAttention 对照）：[[llm-inference/vllm/README]]
- 推理总览：[[llm-inference/README]]
- 相关主题：[[llm-inference/KV-Cache优化]] · [[llm-inference/PD分离]] · [[llm-inference/sglang/README]]

**官方与参考（外链）：**

- LightLLM 仓库：https://github.com/ModelTC/lightllm
- 官方文档（含 TokenAttention / Router 说明）：https://github.com/ModelTC/lightllm/blob/main/docs/LightLLM.md
- 介绍文：纯 Python 超轻量高性能 LLM 推理框架：https://mp.weixin.qq.com/s/-wMLMGAHkxeyDYkixqni9Q
- 代码解读 · 显存管理机制：https://zhuanlan.zhihu.com/p/667730434
- 代码解读 · 模型推理：https://zhuanlan.zhihu.com/p/666731524

> 本文聚焦稳定机制与设计思路；精确版本号、CLI 默认值、API 签名、源码行号请以**官方文档 / 源码**为准。
