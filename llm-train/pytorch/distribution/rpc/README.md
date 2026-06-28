# PyTorch RPC：分布式 RPC 框架与基于 RPC 的流水线并行

> 一句话定位：当训练从"每张卡都跑同一个模型、只切数据"（DDP）走向"模型本身被切到不同机器上、各节点角色不同"时，就需要 **RPC（远程过程调用）**——它让你像调用本地函数一样调用**别的进程上的函数**，并自动跨机器把梯度反向传播回来。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/pytorch/distribution/README]] [[llm-train/pytorch/distribution/pipeline-parallel/README]] [[ai-infra/网络/集合通信原语]] [[llm-train/README]]

## 阅读地图

| 节 | 标题 | 你将搞懂 | 难度 |
|----|------|----------|------|
| 0 | 一句话锚点 | RPC 到底解决什么、和 DDP 的根本区别 | ★ |
| 1 | 地基 / 前置 | 为什么集合通信不够，谁需要 RPC | ★★ |
| 2 | RPC 框架四大件总览 | RPC / RRef / 分布式 Autograd / 分布式优化器 | ★★ |
| 3 | RPC：三种远程调用 | rpc_sync / rpc_async / remote 的差异 | ★★ |
| 4 | RRef：远程引用 | 怎么"指向别人机器上的对象"而不拷贝 | ★★★ |
| 5 | 分布式 Autograd | 反向传播怎么跨机器自动串起来 | ★★★ |
| 6 | 分布式优化器 | 参数散在多机，optimizer 怎么统一更新 | ★★ |
| 7 | 用 RPC 搭流水线并行 | 把模型切成 stage 放不同机器 | ★★★ |
| 8 | 初始化与后端 | init_rpc / TensorPipe / 设备映射 | ★★ |
| 9 | 关键公式 / 通信账 / 数值示例 | 通信量、气泡、加速比手算 | ★★★ |
| 10 | 评价 / 对照 / 局限 | RPC vs 集合通信 vs DDP，何时用 | ★★ |

---

## 0. 一句话锚点

**PyTorch RPC = 一套让进程 A 能"远程"执行进程 B 上的函数、并把结果（或对结果的引用）拿回来的框架，且这条跨进程的调用链能被 Autograd 当作普通计算图的一部分自动求导。**

它和 DDP 是**两种不同的并行哲学**：

```
DDP（数据并行，靠集合通信）            RPC（模型/任务并行，靠点对点远程调用）
─────────────────────────            ──────────────────────────────────
Worker0: [完整模型] ←┐               Worker0: [模型前半 stage0]
Worker1: [完整模型] ←┤ AllReduce              │ rpc 调用
Worker2: [完整模型] ←┤ (梯度求平均)   Worker1: [模型后半 stage1]
Worker3: [完整模型] ←┘                        │ rpc 调用
每个 worker 角色对称、跑同一段代码     Worker2: [参数服务器 / 大 Embedding]
通信是"所有人一起"的集合操作            角色不对称，谁调用谁、点对点
```

一句话区分：**DDP 切数据、RPC 切模型/角色**。当模型小到能整个塞进单卡，你用 DDP；当模型必须被拆到多机、或不同节点要干不同的活（参数服务器、Embedding 服务器、RL 的 actor/learner 分离），你用 RPC。

---

## 1. 地基 / 前置（为什么集合通信不够）

[[llm-train/pytorch/distribution/README]] 讲的 DDP 建立在 **集合通信** 上（AllReduce/AllGather/Broadcast，见 [[ai-infra/网络/集合通信原语]]）。集合通信有一条隐含假设：

> **所有进程对称**——大家跑同一份代码、持有同一个模型、在同一时刻一起参与同一个通信操作。

但很多真实场景**打破了对称性**：

| 场景 | 为什么 DDP/集合通信不适合 |
|------|---------------------------|
| **模型太大，一张卡放不下** | 必须把不同层切到不同机器（流水线并行），机器之间是"前一段把激活值传给后一段"的**点对点、有方向**的传递，不是大家一起 AllReduce。|
| **参数服务器架构（PS）** | 有的进程只存参数、有的进程只算梯度，角色完全不同，无法用一段对称代码描述。|
| **超大 Embedding** | 几十亿行的 Embedding 表放在专门的"Embedding 服务器"上，worker 按需远程查表。|
| **强化学习 actor/learner 分离** | actor 在多机收集轨迹、learner 在中心训练，二者代码、节奏都不同。|

这些场景的共同需求是：**进程 A 想"指挥"进程 B 干一件具体的事（执行某个函数、持有某个张量），并且这件事要能参与反向传播。** 集合通信原语（只会"求和/广播"数据）表达不了"远程执行一段逻辑"。于是 PyTorch 提供了 `torch.distributed.rpc`。

**几个必须先建立的原子概念：**

| 概念 | 一句话 |
|------|--------|
| **RPC（远程过程调用）** | 让本进程像调本地函数一样，触发**远端进程**执行一个函数并（可选）取回返回值。|
| **worker / name** | RPC 世界里每个进程有一个**字符串名字**（如 `"worker1"`），调用时用名字寻址，而不是 rank 数字。|
| **RRef（Remote Reference，远程引用）** | 一个"指针"，指向**存活在别的 worker 上**的对象。可以传递、可以远程操作，但数据不搬到本地。|
| **分布式 Autograd** | 把跨越多个 worker 的前向调用链，自动织成一张分布式反向图，一条 `backward` 把所有机器上的梯度都算出来。|
| **分布式优化器** | 参数 RRef 散落多机，它在每个持有参数的 worker 上各起一个本地 optimizer，统一 `step()`。|

---

## 2. RPC 框架四大件总览

PyTorch 的 `torch.distributed.rpc` 由四块拼成，层层依赖：

```
        ┌────────────────────────────────────────────────┐
        │          应用：流水线并行 / 参数服务器 / RL        │
        └────────────────────────────────────────────────┘
                            ▲
   ┌────────────────┐  ┌────────────────────┐
   │ ④ 分布式优化器  │  │ ③ 分布式 Autograd   │   ← 跨机自动求导 + 统一更新
   │ DistributedOptim│  │ dist_autograd       │
   └────────────────┘  └────────────────────┘
                            ▲
        ┌────────────────────────────────────────────────┐
        │ ② RRef（远程引用）：跨 worker 指向对象/参数        │
        └────────────────────────────────────────────────┘
                            ▲
        ┌────────────────────────────────────────────────┐
        │ ① RPC（rpc_sync / rpc_async / remote）：远程执行  │
        └────────────────────────────────────────────────┘
                            ▲
        ┌────────────────────────────────────────────────┐
        │  传输后端 TensorPipe（点对点，自动选 IB/NVLink等）│
        └────────────────────────────────────────────────┘
```

- **① RPC**：地基。"在 worker B 上跑函数 f"。
- **② RRef**：让你能持有"远端对象的句柄"，从而把远端的子模块、参数当一等公民传来传去。
- **③ 分布式 Autograd**：当 RPC 调用里出现了 `requires_grad=True` 的张量，框架自动记录跨机依赖，`backward` 时反向穿过 RPC 边界。
- **④ 分布式优化器**：拿着一堆参数 RRef，自动在各 worker 本地建 optimizer 并 `step`。

下面逐件拆。

---

## 3. RPC：三种远程调用

核心是"在某个 worker 上执行某个函数"。三种调用语义，区别只在**何时阻塞、返回什么**：

| API | 阻塞？ | 返回值 | 用途 |
|-----|--------|--------|------|
| `rpc.rpc_sync(to, func, args)` | **同步阻塞**，等远端算完 | 真正的**结果值**（已拷回本地） | 立刻要用结果 |
| `rpc.rpc_async(to, func, args)` | **不阻塞**，立刻返回 | 一个 `Future`，`.wait()` 取结果 | 想并发发起多个调用 |
| `rpc.remote(to, func, args)` | **不阻塞** | 一个 **RRef**（结果留在远端，不拷回） | 结果要继续留在远端被操作 |

```
本地 worker0                      远端 worker1
────────────                      ────────────
rpc_sync("worker1", f, x) ──发请求──▶ 执行 f(x)
        │  (阻塞等待)                   │
        ◀──────结果拷回本地──────────── 返回
返回真实结果

remote("worker1", f, x) ───发请求──▶ 执行 f(x)
        │ 立刻返回一个 RRef            结果对象留在 worker1
        │ （像一张"远端取货单"）
继续干别的，需要时再 .to_here()
```

**关键直觉**：`remote` 是这套框架的"灵魂"——它返回 RRef 而不是数据本身。这意味着你可以**把一个大对象（比如一个子模型）建立在远端，永远不把它拷回本地**，只用 RRef 这张"取货单"去指挥它。流水线并行正是靠这个：每个 stage 的子模块用 `remote` 建在各自机器上，主进程只持有它们的 RRef。

**寻址用名字**：所有调用第一个参数 `to` 是 worker 的字符串名（`"worker1"`），由 `init_rpc(name=...)` 注册。这比 DDP 里裸用 rank 数字更可读，也支持角色化命名（`"ps"`、`"trainer0"`）。

---

## 4. RRef：远程引用（最难也最关键的一块）

**RRef = 一个分布式版的"智能指针"，指向某个存活在特定 worker 上的对象。** 它解决一个根本问题：*我怎么在 A 机器上"持有并操作"一个其实住在 B 机器上的东西，而不把它搬过来？*

```
worker0 (持有 RRef)            worker1 (RRef 的 "owner"，真正存对象)
──────────────────            ──────────────────────────────────
rref ──────────────────────────▶ [真实对象: 比如一个 nn.Linear 子模块]
 │  rref.to_here()  → 把对象一份拷贝拉回本地（慎用，会搬数据）
 │  rref.rpc_sync().forward(x) → 在 owner 上直接调它的方法（数据不动）
 │  rref.remote().forward(x)   → 同上，但返回新 RRef
```

几个要点：

- **owner vs user**：对象真正存在的那个 worker 叫 **owner**；持有 RRef 句柄的其它 worker 叫 **user**。owner 自己也能持有指向自己对象的 RRef（"本地 RRef"）。
- **不拷贝**：RRef 在 worker 间传递时，传的是"引用元信息"，不是底层张量。这正是处理超大 Embedding/大子模块的关键——**对象只此一份，存在 owner 上**。
- **远程操作三招**：
  - `rref.to_here()`：把对象**拉回本地**（真的搬数据，仅在你确实需要本地副本时用）。
  - `rref.rpc_sync()` / `rpc_async()` / `remote()`：在 **owner 上**调用对象的方法，数据留在 owner，只传参数和结果。
- **生命周期**：框架用**分布式引用计数**管理——只要还有任何 user 持有这个 RRef，owner 上的对象就不会被回收；所有 RRef 释放后才回收。这避免了"远端对象被提前删掉"的悬空引用。

**为什么流水线并行离不开 RRef**：把模型切成 `stage0@worker0`、`stage1@worker1`、`stage2@worker2`，主控进程用 `rpc.remote` 在各 worker 上构建子模块，拿到三个 RRef。前向时主控不需要任何子模块的真实权重，只用 RRef 依次"指挥"它们前向、并把上一段的输出 RRef 喂给下一段。**权重永远不跨机搬动，只有激活值在 stage 之间点对点流动。**

---

## 5. 分布式 Autograd（反向传播怎么跨机器）

普通 Autograd 在单进程内记录计算图：每个 `requires_grad` 张量记得自己"从哪来"，`backward()` 沿图反向传播。问题是：**当一次前向跨越了 RPC（worker0 的输出经 RPC 送到 worker1 继续算），这张图被 RPC 边界"切断"了**——worker1 不知道它的输入其实是 worker0 某个张量的函数。

**分布式 Autograd 的工作**：在 RPC 传输张量时，**自动在发送方和接收方各埋一个特殊节点**（`send` / `recv`），把两段本地图缝合成一张跨机的全局图。

```
worker0 本地图                RPC 边界               worker1 本地图
────────────                ─────────              ─────────────
x → linear0 → out0 ──[send]══════════[recv]──▶ in1 → linear1 → loss
                      ▲                  ▲
              发送时记录"我把梯度该回传给谁"  接收时记录"我的梯度从哪来"

反向：dist_autograd.backward([loss]) 在 worker1 发起
  loss → linear1 的梯度 → [recv 节点] ──回传到 worker0 的 [send 节点]
       ──▶ worker0 继续 linear0 反向 → x 的梯度
```

用法上有两个新东西：

- **`dist_autograd.context()`**：一个 `with` 上下文，给本次前向-反向分配一个全局唯一的 **context_id**。所有跨机梯度都按这个 id 归集，互不串扰（多个并发训练步可以各用各的 context）。
- **`dist_autograd.backward(context_id, [loss])`**：取代普通的 `loss.backward()`，触发跨机反向。梯度**不存在 `param.grad`**，而是存在该 context 里，用 `dist_autograd.get_gradients(context_id)` 取（通常你不直接取，交给分布式优化器）。

```python
with dist_autograd.context() as cid:
    out = forward_through_rpc(x)          # 前向可能跨多机
    loss = loss_fn(out, y)
    dist_autograd.backward(cid, [loss])   # 跨机反向，梯度入 context
    dist_optimizer.step(cid)              # 按 context 里的梯度更新各机参数
```

**记住**：`dist_autograd.backward` 必须传 **loss 列表**和 **context_id**，且梯度归在 context 而非 `.grad`——这是它和单机 `backward()` 的两个硬性区别。

---

## 6. 分布式优化器

参数现在散落多机（stage0 的参数在 worker0、stage1 的在 worker1……），还都是 RRef。`DistributedOptimizer` 接收**一组参数 RRef**，做两件事：

1. 在**每个持有参数的 owner worker 上**，本地实例化一份你指定的优化器（如 `optim.SGD`），只管它本地那部分参数。
2. 调用 `dist_optim.step(context_id)` 时，**并发地** RPC 通知每个 owner：用该 context 里属于你的梯度，跑一次本地 `step`。

```
DistributedOptimizer([rref_p0@w0, rref_p1@w1, rref_p2@w2], lr=..)
        │
   step(cid) ──┬── rpc → worker0: 本地 SGD.step(用 cid 里 p0 的梯度)
               ├── rpc → worker1: 本地 SGD.step(用 cid 里 p1 的梯度)
               └── rpc → worker2: 本地 SGD.step(用 cid 里 p2 的梯度)
       （三个 step 并发执行，互不依赖）
```

好处：你写起来仍像单机——构造一个 optimizer、一个 `step`——但底层自动把更新分发到各机，并精确地只用本 step（本 context）算出的梯度。

---

## 7. 用 RPC 搭流水线并行（本目录主题）

本目录原始指向 PyTorch 官方教程 *Distributed Pipeline Parallelism Using RPC*。把上面四件拼起来，就是基于 RPC 的流水线并行（[[llm-train/pytorch/distribution/pipeline-parallel/README]] 讲的是单机版 `Pipe`，这里是**跨机版**）。

**思路**：模型按层切成若干 stage，每个 stage 用 `rpc.remote` 建在一台机器上，得到 stage 的 RRef；主控进程串联这些 RRef 完成前向。

```
            主控进程 (持有各 stage 的 RRef，不持有真实权重)
                              │
   ┌──────────────┬──────────────┬──────────────┐
   ▼              ▼              ▼
[stage0@w0]   [stage1@w1]    [stage2@w2]
 conv/embed    transformer    head/loss
   ▲ 激活值      ▲ 激活值        ▲
   └──RPC点对点──┘──RPC点对点───┘
前向：x →(rpc) stage0.forward → out0 (RRef)
           →(rpc) stage1.forward(out0) → out1 (RRef)
           →(rpc) stage2.forward(out1) → loss
反向：dist_autograd.backward(cid,[loss]) 自动穿过三段
更新：dist_optimizer.step(cid)
```

**为什么要切微批（micro-batch）**：如果整批数据走完 stage0 才轮到 stage1，那么 stage1/stage2 在 stage0 算的时候全在**空转**（这段空转叫**气泡 bubble**）。流水线并行的精髓是把一个 batch 切成 $m$ 个微批，像工厂流水线一样让各 stage **同时**处理不同微批：

```
时间 →
        t0   t1   t2   t3   t4   t5   t6
stage0  μ0   μ1   μ2   μ3   ·    ·    ·
stage1  ·    μ0   μ1   μ2   μ3   ·    ·     ← 错峰流水
stage2  ·    ·    μ0   μ1   μ2   μ3   ·
        └填充┘    └─── 满负荷 ───┘ └排空┘
        (bubble)                  (bubble)
```

**最小示例思路（伪代码）**：

```python
# 1) 各 worker 上 init_rpc(name=..., rank=..., world_size=...)
# 2) 主控构建各 stage 的远程子模块
s0 = rpc.remote("w0", Stage0)            # RRef
s1 = rpc.remote("w1", Stage1)            # RRef
s2 = rpc.remote("w2", Stage2)            # RRef
# 3) 收集所有参数 RRef（远程取参数）
params = (s0.remote().parameter_rrefs().to_here()
          + s1.remote().parameter_rrefs().to_here()
          + s2.remote().parameter_rrefs().to_here())
dopt = DistributedOptimizer(optim.SGD, params, lr=0.05)
# 4) 训练步：切微批 + 跨机前向 + 分布式反向 + 分布式更新
for x, y in loader:
    with dist_autograd.context() as cid:
        out0 = s0.remote().forward(x)        # 返回 RRef，不拷回
        out1 = s1.remote().forward(out0)     # 把 RRef 直接喂下一段
        loss = s2.rpc_sync().forward(out1, y)
        dist_autograd.backward(cid, [loss])
        dopt.step(cid)
```

注意 `out0`、`out1` 全程是 **RRef**——激活值在远端流转，主控只传引用，避免把中间激活值反复拷回主控再拷出去。

---

## 8. 初始化与后端（TensorPipe / 设备映射）

**入口**：每个进程调用一次 `rpc.init_rpc(name, rank, world_size)`：

| 参数 | 含义 |
|------|------|
| `name` | 本 worker 的字符串名（如 `"worker1"`），后续被别人寻址用 |
| `rank` / `world_size` | 全局编号与总进程数，和 `init_process_group` 类似 |
| `rpc_backend_options` | 后端配置，最关键是**设备映射 device_maps** |

结束时所有进程调用 `rpc.shutdown()`，它会做一次**屏障（barrier）**，确保没有进程提前退出导致别人 RPC 失败。

**后端 = TensorPipe**：RPC 默认走 **TensorPipe** 后端（不是 NCCL！因为 RPC 是**点对点**而非集合通信）。TensorPipe 会**自动协商最快可用链路**：同机走共享内存/CUDA IPC，跨机走 InfiniBand/以太网。

**设备映射 device_maps（GPU 直传的关键）**：默认 RPC 传 GPU 张量时会先**搬到 CPU** 再传，跨机再搬回 GPU——慢。配置 `device_maps` 告诉 TensorPipe "我把张量发给 worker1 时，本地 cuda:0 对应它的 cuda:0"，就能走 **GPU 到 GPU 直传**（NVLink/GPUDirect），省掉 CPU 中转：

```python
opts = rpc.TensorPipeRpcBackendOptions()
opts.set_device_map("worker1", {0: 0})   # 本地cuda:0 → 远端cuda:0
rpc.init_rpc("worker0", rank=0, world_size=2, rpc_backend_options=opts)
```

> 具体 API 名、默认值以官方文档为准；不同 PyTorch 版本 `device_maps` 配置方式略有差异。

---

## 9. 关键公式 / 通信账 / 数值示例

### 9.1 流水线气泡占比

设 $p$ 个 stage、每个微批在单 stage 的前向+反向耗时记作 1 个"单位"，$m$ 个微批。理想满负荷需要 $m$ 个单位；但**填充 + 排空**额外引入 $(p-1)$ 个单位的气泡。

$$\text{气泡占比} = \frac{p-1}{m + p - 1}$$

**手算**：$p=4$ 个 stage、$m=1$（不切微批）时，气泡占比 $=\frac{3}{1+3}=75\%$——四张卡有效利用率仅 25%！把 $m$ 提到 $16$：$\frac{3}{16+3}=\frac{3}{19}\approx 15.8\%$，有效利用率约 84%。**这就是"必须切微批"的量化理由。**

理想加速比（忽略通信）：

$$\text{speedup} = \frac{p}{1 + \frac{p-1}{m}} = \frac{m\,p}{m+p-1}$$

$p=4, m=16$：$\frac{16\times4}{16+3}=\frac{64}{19}\approx 3.37\times$（相对理论上限 $4\times$，损失来自气泡）。

### 9.2 RPC 通信量 vs DDP 通信量

- **基于 RPC 的流水线并行**：stage 之间只传**激活值**。设 batch $B$、序列 $L$、隐藏维 $H$、fp16（2 字节），相邻 stage 间单次前向传输：

$$\text{激活量} = B \cdot L \cdot H \cdot 2 \text{ 字节}$$

例 $B=8, L=2048, H=4096$：$8\times2048\times4096\times2 \approx 1.34\times10^8$ 字节 $\approx 128\,\text{MB}$ 每个 stage 边界、每个方向。**和参数量无关**（只和这一层的输出张量大小有关）。

- **DDP**：每步 AllReduce **全部参数梯度**，通信量 $\approx 2\Phi$（$\Phi$ 为参数量字节数，AllReduce 约 $2\times$ 参数量）。70 亿参数 fp16：$\Phi\approx14\,\text{GB}$，单步约 $28\,\text{GB}$ 通信。

**对照结论**：当模型巨大（参数量 ≫ 单层激活），RPC 流水线的"传激活"远小于 DDP 的"传全部梯度"。但 RPC 是点对点串行依赖、易有气泡；DDP 通信可与反向计算**重叠**。所以二者常**组合**用（先流水线切模型，再每个 stage 内 DDP 切数据）。

### 9.3 显存账（流水线为什么省显存）

单卡放整个 $N$ 层模型需要 $N$ 层的参数+优化器状态+激活。切成 $p$ 个 stage 后，**每张卡只放 $N/p$ 层**：

$$\text{单卡参数显存} \approx \frac{\Phi}{p}$$

代价：为了反向，每个 stage 要**缓存在飞微批的激活**，缓存量随 $m$（在飞微批数）上升——这就是显存与气泡的权衡（$m$ 大气泡小，但激活缓存多）。

---

## 10. 评价 / 对照 / 局限

| 维度 | 集合通信 / DDP | RPC 框架 |
|------|----------------|----------|
| 并行哲学 | 切**数据**，进程对称 | 切**模型/角色**，进程不对称 |
| 通信模式 | 集合（AllReduce 等），所有人参与 | **点对点**远程调用，谁调谁 |
| 后端 | NCCL（GPU）/ Gloo | **TensorPipe**（自动选链路） |
| 寻址 | rank 数字 | **worker 名字**（可角色化） |
| 反向传播 | 本地 Autograd + 梯度 AllReduce | **分布式 Autograd**（跨机缝图） |
| 优化器 | 各自本地 optimizer | **DistributedOptimizer**（参数 RRef） |
| 典型用途 | 模型放得下、纯加速 | 流水线并行、参数服务器、大 Embedding、RL actor/learner |
| 编程复杂度 | 低（DDP 几乎透明） | **高**（RRef/context/手动编排 stage） |

**局限与注意**：

- **编排复杂**：RRef 生命周期、context_id、stage 划分都要手写，比 DDP 易出错。
- **气泡损失**：流水线天生有填充/排空气泡，必须靠切微批缓解（见 9.1）。
- **负载均衡**：stage 切得不均，最慢的 stage 卡住整条流水线（木桶效应）。
- **GPU 直传需配 device_maps**，否则张量经 CPU 中转，跨机带宽白白浪费。
- **现代趋势**：纯手写 RPC 流水线在生产中已较少；大模型训练更多用 **Megatron-LM / DeepSpeed** 封装好的流水线/张量并行，或 PyTorch 新的 `torch.distributed.pipelining`。但理解 RPC 这套底层语义，是看懂这些框架"模型并行如何跨机求导与更新"的基础。

> 版本相关 API（`init_rpc`、`TensorPipeRpcBackendOptions`、`dist_autograd`、`DistributedOptimizer` 的具体签名与默认值）**以官方文档为准**；本文公式中的数值为**手算估计**，工程实测受通信重叠、带宽、实现细节影响。

---

## 🔗 跳转链接

- 知识地图枢纽：[[00-知识地图]]
- 上级总览：[[llm-train/pytorch/distribution/README]]（DDP / 集合通信 / 进程组基础）
- 同主题：[[llm-train/pytorch/distribution/pipeline-parallel/README]]（单机 `Pipe` 流水线并行，本文是其跨机 RPC 版）
- 通信基础：[[ai-infra/网络/集合通信原语]]（理解 RPC 点对点 vs 集合通信的对照）
- 训练总览：[[llm-train/README]]

> 原始参考：DISTRIBUTED PIPELINE PARALLELISM USING RPC — https://pytorch.org/tutorials/intermediate/dist_pipeline_parallel_tutorial.html
