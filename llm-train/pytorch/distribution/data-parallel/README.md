# PyTorch 数据并行：DP / DDP / FSDP

> 一句话定位：当**单卡能放下整个模型**时，用「复制模型 + 切分数据 + 同步梯度」来加速训练；DP 是单进程多线程的玩具，DDP 是工业标准，FSDP 把参数/梯度/优化器状态也切片以训练放不下的大模型。📍 导航：[[00-知识地图]]
>
> 🔗 相关：[[llm-train/pytorch/distribution/README]] · [[ai-framework/pytorch/README]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/NCCL]]

---

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|--------|
| 0 | 一句话锚点 | 复制模型·切数据·同步梯度 |
| 1 | 地基：为什么数据并行在数学上成立 | 梯度线性·all-reduce |
| 2 | DP（DataParallel）：单进程多线程，为什么慢 | GIL·主卡瓶颈·scatter/gather |
| 3 | DDP（DistributedDataParallel）：多进程 + Reducer 分桶 | bucket·autograd hook·异步 all-reduce |
| 4 | DDP + 流水线并行（MP）混合 | 混合并行 |
| 5 | 启动方式：launch.py / torchrun / SLURM | rdzv·nnodes·node_rank |
| 6 | FSDP：参数分片，训练单卡放不下的模型 | shard·all-gather·reduce-scatter |
| 实操 | 所有真实命令原样保留并解释 | — |
| 坑 | 常见问题速查表 | — |
| 链接 | 官方文档 + 双链枢纽 | — |

---

## 0. 一句话锚点

> **数据并行（Data Parallel）= 每张卡放一份完整模型，把一个 batch 切成 N 份分给 N 张卡各算各的，再把 N 份梯度求平均同步回来。**

三个动作，缺一不可：

```
复制模型 (replicate)  →  切分数据 (scatter)  →  同步梯度 (all-reduce)
```

参考资料：https://zhuanlan.zhihu.com/p/343951042

---

## 1. 地基：为什么数据并行在数学上是对的

> 当一张 GPU 可以存储一个模型时，可以采用数据并行得到更准确的梯度或者加速训练，即每个 GPU 复制一份模型，将一批样本分为多份输入各个模型并行计算。**因为求导以及加和都是线性的，数据并行在数学上也有效。**

### 1.1 关键：梯度对样本求和是线性的

设全局 batch 有 $B$ 个样本，损失为各样本损失的平均：

$$L = \frac{1}{B}\sum_{i=1}^{B} \ell(x_i)$$

梯度同样是平均（梯度算子是线性的，求和号可以拿进拿出）：

$$\nabla_\theta L = \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta \ell(x_i)$$

现在把 $B$ 个样本平均分到 $N$ 张卡，每张卡 $b = B/N$ 个样本，第 $k$ 张卡本地算出：

$$g_k = \frac{1}{b}\sum_{i \in \text{卡}k} \nabla_\theta \ell(x_i)$$

把 $N$ 张卡的本地梯度**再平均一次**，正好还原全局梯度：

$$\frac{1}{N}\sum_{k=1}^{N} g_k = \frac{1}{N}\cdot\frac{1}{b}\sum_{k=1}^{N}\sum_{i\in k}\nabla\ell(x_i) = \frac{1}{B}\sum_{i=1}^{B}\nabla\ell(x_i) = \nabla_\theta L \quad\checkmark$$

**这就是为什么"各卡算各的，再 all-reduce 求平均"在数学上等价于单卡跑大 batch。** 这一步是所有数据并行（DP/DDP/FSDP）的理论基石。

### 1.2 数值手算示例

2 张卡，全局 batch=4，假设某参数的 4 个样本梯度分别是 $[2, 4, 6, 8]$：

```
单卡基准：  (2+4+6+8)/4 = 5.0

卡0 拿到 [2,4]：  g0 = (2+4)/2 = 3.0
卡1 拿到 [6,8]：  g1 = (6+8)/2 = 7.0
all-reduce 平均： (3.0+7.0)/2 = 5.0   ←  和单卡完全一致 ✅
```

> ⚠️ 易错点：本地必须是**平均**（除以本地 batch），最后 all-reduce 也是**平均**（除以卡数）。如果本地用 sum 不除，all-reduce 再平均，结果就会差一个 $b$ 倍。NCCL 的 `all_reduce` 默认做 SUM，所以 DDP 内部会把梯度先乘 $1/N$ 再 sum-reduce，等价于求平均。

### 1.3 数据并行 vs 其他并行（一句话区分）

| 并行方式 | 切什么 | 适用前提 |
|---------|--------|---------|
| **数据并行 DP/DDP** | 切 **数据**（batch），模型整份复制 | 单卡放得下模型 |
| **FSDP / ZeRO** | 切 **数据** + 切 **参数/梯度/优化器状态** | 单卡放不下完整状态 |
| 张量并行 TP | 切 **单层权重矩阵** | 单层都放不下 → 见 [[B07:llm-inference/大模型推理张量并行]] |
| 流水线并行 PP | 切 **层**（不同层放不同卡） | 模型深、按层切 |

---

## 2. DP（torch.nn.DataParallel）：能跑，但别用

### 2.1 用法（原文真料，原样保留）

```python
model = nn.DataParallel(model)
```

一行包起来就行，无需改训练循环——这是它唯一的优点。

### 2.2 它做了什么（单进程多线程）

```
                 ┌──────────── 主进程（单 Python 进程） ─────────────┐
                 │                                                   │
  输入 batch ───►│  GPU0(主卡): 持有完整模型 + 优化器                │
                 │     │ scatter 切 batch                            │
                 │     ├──► 复制模型到 GPU1 ──► 前向 ──► 算 loss      │
                 │     ├──► 复制模型到 GPU2 ──► 前向 ──► 算 loss      │
                 │     └──► 复制模型到 GPU3 ──► 前向 ──► 算 loss      │
                 │     gather 把各卡输出收回 GPU0 ──► 反向            │
                 │     梯度汇总到 GPU0 ──► GPU0 更新参数              │
                 └───────────────────────────────────────────────────┘
       每一步迭代：复制模型 + scatter + gather 全在主卡上做
```

### 2.3 为什么 DP 慢、官方不推荐

| 问题 | 原因 |
|------|------|
| **Python GIL** | 单进程多线程，多个 GPU 的调度被全局解释器锁串行化，CPU 端无法真正并行 |
| **主卡负载不均** | gather 把所有输出收到 GPU0，loss/反向都堆在主卡 → GPU0 显存爆、利用率瓶颈 |
| **每步重复复制模型** | 每个 iteration 都要把模型从主卡广播到其他卡，开销大 |
| **不支持多机** | 只能单机多卡 |

> 结论：**DP 已被官方标记为不推荐**，新代码一律用 DDP。DP 唯一价值是几行 demo 或快速验证。

---

## 3. DDP（DistributedDataParallel）：工业标准

> 核心思想：**每张卡一个独立进程**，每个进程持有完整模型副本，反向传播时通过 **all-reduce** 同步梯度。没有主卡瓶颈，没有 GIL 问题，支持多机。

### 3.1 DDP 的梯度同步：Reducer + 分桶（Bucket）

原文精华（原样保留并展开解释）：

> DDP 通过 **Reducer** 来管理梯度同步。为了提高通讯效率，Reducer 会将梯度归到不同的**桶（bucket）**里（按照模型参数的 **reverse order**，因为反向传播需要符合这样的顺序），一次归约一个桶。其中，桶的大小为参数 **`bucket_cap_mb` 默认为 25**（MB），可根据需要调整。
>
> DDP 通过在构建时注册 **autograd hook** 进行梯度同步。反向传播时，当一个梯度计算好后，相应的 hook 会告诉 DDP 可以用来归约。
>
> 当一个桶里的梯度都可以了，Reducer 就会启动**异步 all-reduce** 去计算所有进程的平均值。**all-reduce 异步启动使得 DDP 可以边计算边通信，提高效率（计算与通信重叠 overlap）。**
>
> 当所有桶都可以了，Reducer 会等所有 all-reduce 完成，然后将得到的梯度写到 `param.grad`。

### 3.2 为什么要"分桶" + "倒序" + "异步"？三个为什么

**为什么分桶（bucketing）？**
- all-reduce 有固定的启动开销（latency）。如果每个参数（成千上万个 tensor）单独发一次 all-reduce，启动开销会压垮带宽。
- 把多个小梯度**攒成一个桶**（默认 25MB）再一次性 all-reduce，把"很多次小通信"合并成"少数几次大通信"，更接近带宽上限。

**为什么倒序（reverse order）放桶？**
- 反向传播是**从最后一层往前**算的。最后一层的梯度**最先**算好。
- 把参数按 forward 的倒序（≈ backward 的顺序）放桶，就能让"先算好的梯度先凑满桶、先开始通信"，最大化计算/通信重叠。

**为什么异步（async all-reduce）？**
- 桶 A 凑满后立即异步发起 all-reduce，**不阻塞** GPU 继续算更靠前层的梯度（凑桶 B）。
- 于是 "算前面的层" 和 "传后面层的梯度" 在时间上重叠：

```
反向传播时间轴（→ 表示时间推进）
计算: [算桶3的梯度]→[算桶2的梯度]→[算桶1的梯度]→[算桶0的梯度]
通信:               [桶3 allreduce ]→[桶2 allreduce ]→[桶1 ar]→[桶0 ar]
                    ↑ 桶3一满就异步发，GPU不等，继续算桶2  →  计算与通信重叠！
```

### 3.3 DDP 整体数据流图

```
   进程0 / GPU0              进程1 / GPU1              进程2 / GPU2
 ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
 │ 完整模型副本 │         │ 完整模型副本 │         │ 完整模型副本 │
 │  (相同初值)  │         │  (相同初值)  │         │  (相同初值)  │
 └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
   本地batch分片            本地batch分片            本地batch分片
        │ forward                │ forward                │ forward
        │ backward(算梯度)       │ backward               │ backward
        ▼                        ▼                        ▼
   ┌─────────────────── all-reduce 各桶梯度求平均 ───────────────────┐
   │   g_avg = (g0 + g1 + g2) / 3      （NCCL ring all-reduce）       │
   └─────────────────────────────────────────────────────────────────┘
        │                        │                        │
   各自用 g_avg 更新        各自更新（结果相同）      各自更新
   → 三卡参数始终保持一致（因为初值相同、梯度相同、优化器相同）
```

> 🔑 关键不变量：**只要初始参数相同 + 每步梯度 all-reduce 后相同 + 优化器相同**，各卡模型在每一步后都**自动保持一致**，无需再广播参数。这是 DDP 比 DP 高效的根本原因——**只同步梯度，不同步整个模型**。

### 3.4 集合通信原语：all-reduce 是什么

DDP 的梯度同步依赖 **all-reduce** 原语（详见 [[ai-infra/网络/集合通信原语]] / [[ai-infra/网络/NCCL]]）。Ring All-Reduce = Reduce-Scatter + All-Gather，通信量 $\approx 2\frac{N-1}{N}\cdot S$（$S$=梯度总字节数，$N$=卡数），与卡数几乎无关，因此扩展性好。

```
all-reduce 语义： 每个进程都拿到「所有进程数据求和(或平均)」后的同一份结果
  in:  GPU0=[a]  GPU1=[b]  GPU2=[c]
  out: GPU0=[a+b+c]  GPU1=[a+b+c]  GPU2=[a+b+c]   （DDP里再 /N 得平均）
```

### 3.5 DDP + MP（流水线并行）混合

> **DDP 与流水线并行（Pipeline / Model Parallel）混合使用。**

当模型大到单卡放不下时，可以把模型按层切到几张卡（流水线并行 MP/PP），再用 DDP 把这个"跨卡的大模型"复制成多组做数据并行——即**混合并行**。

```
        ── 数据并行维度（DDP，复制 2 组）──►
   ┌─ 组0 ────────────────┐   ┌─ 组1 ────────────────┐
 ↑ │ GPU0(层0-3) GPU1(层4-7)│   │ GPU2(层0-3) GPU3(层4-7)│
 │ │   └── 流水线并行 ──┘   │   │   └── 流水线并行 ──┘   │
 模 └────────┬─────────────┘   └────────┬─────────────┘
 型          └──── DDP all-reduce 同步两组对应层的梯度 ────┘
 维
```

> 进一步的大规模混合并行（DP×TP×PP×ZeRO）由 Megatron / DeepSpeed 等框架实现，见 [[llm-train/megatron/README]] · [[llm-train/megatron-deepspeed/README]] · [[ai-framework/deepspeed/README]]。

---

## 实操：启动 DDP（原文命令全部保留并解释）

DDP 是多进程的，需要一个"启动器"在每张卡上拉起一个进程。历史上有两代：旧的 `launch.py` 和新的 `torchrun`（torch elastic）。

### A. launch.py（旧版启动器）

> 通过 `launch.py` 启动，在 8 个 GPU 节点上，每个 GPU 一个进程：

```bash
# 单机 8 卡：拉起 8 个进程
python /home/guodong.li/virtual-venv/megatron-ds-venv-py310-cu117/lib/python3.10/site-packages/torch/distributed/launch.py \
    --nnode=1 --node_rank=0 --nproc_per_node=8 \
    example.py --local_world_size=8

# 单机 1 卡：拉起 1 个进程（调试用）
python /home/guodong.li/virtual-venv/megatron-ds-venv-py310-cu117/lib/python3.10/site-packages/torch/distributed/launch.py \
    --nnode=1 --node_rank=0 --nproc_per_node=1 \
    example.py --local_world_size=1
```

参数含义对照表：

| 参数 | 含义 | 示例值 |
|------|------|--------|
| `--nnode` | 参与训练的**机器（节点）数** | 1 |
| `--node_rank` | **当前机器**在所有节点中的编号（从 0 开始） | 0 |
| `--nproc_per_node` | **每台机器**拉起的进程数（通常=该机 GPU 数） | 8 |
| `--local_world_size` | 传给脚本的本地进程总数（脚本自定义参数） | 8 |

> 总进程数 `world_size = nnode × nproc_per_node`。上例单机 8 卡 → world_size = 1×8 = 8。

### B. torchrun（torch elastic，推荐）

`torchrun` 是 `launch.py` 的继任者，增加了**弹性容错（elastic）**和 **rendezvous（rdzv，集合点）**机制，能处理节点动态加入/退出。

**多机示例（2 机 × 8 卡 = 16 个 GPU）：**

```bash
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 \
    elastic_ddp.py
```

> 我们在两台主机上运行 DDP 脚本，每台主机运行 8 个进程，也就是说，我们在 **16 个 GPU** 上运行它。
> ⚠️ **所有节点上的 `$MASTER_ADDR` 必须相同**（它是集合点 rendezvous 的地址）。
> `torchrun` 会在**启动它的节点**上拉起 8 个进程并各自调用 `elastic_ddp.py`，但要在 2 个节点上**同时**实际运行此命令，还需借助 SLURM 等集群管理工具。

rdzv 参数对照表：

| 参数 | 含义 |
|------|------|
| `--nnodes=2` | 2 台机器 |
| `--nproc_per_node=8` | 每台机器 8 个进程 |
| `--rdzv_id=100` | 本次作业的唯一 ID（同一作业所有节点必须相同） |
| `--rdzv_backend=c10d` | 集合点后端，`c10d` 是内置 TCP store，无需额外服务 |
| `--rdzv_endpoint=$MASTER_ADDR:29400` | 集合点地址:端口（所有节点指向同一个） |

**单机 8 卡（本地回环地址）：**

```bash
torchrun --nnodes=1 --nproc_per_node=8 \
    --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 \
    elastic_ddp.py
```

**显式指定 node_rank 的多机写法（2 机，各自手动起）：**

```bash
# 在节点0（rank0，通常也是 master）执行：
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=10.xx.2.46:29400 \
    multigpu_torchrun.py --batch_size 32  10 5

# 在节点1（rank1）执行：
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=10.xx.2.46:29400 \
    multigpu_torchrun.py --batch_size 32  10 5
```

> 注意两条命令只有 `--node_rank` 不同（0 和 1），`rdzv_endpoint` 都指向 `10.xx.2.46:29400`（节点0 的地址）。脚本尾部 `--batch_size 32 10 5` 是 `multigpu_torchrun.py` 自己的位置参数（如 batch_size / total_epochs / save_every）。

详细启动命令：https://pytorch.org/docs/stable/elastic/quickstart.html

### C. 配合 SLURM 集群调度

`torchrun` 只负责"在一台机器上拉进程"，**跨多节点的下发**要靠 SLURM。

在启用 SLURM 的集群上，先把 `MASTER_ADDR` 设为节点列表里的第一台主机：

```bash
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
```

然后用 SLURM 命令在 2 个节点上同时跑脚本：

```bash
srun --nodes=2 ./torchrun_script.sh
```

> 这只是一个例子；你可以选择自己的集群调度工具（SLURM / k8s / PAI 等）来启动 torchrun 作业。

启动方式总览对照表：

| 启动器 | 单机 | 多机 | 弹性容错 | 状态 |
|--------|:----:|:----:|:--------:|------|
| `python -m torch.distributed.launch` / `launch.py` | ✅ | ✅（需手动 node_rank） | ❌ | 旧版，已废弃 |
| `torchrun`（elastic） | ✅ | ✅（rdzv 集合点） | ✅ | **推荐** |
| `torchrun` + SLURM `srun` | — | ✅（自动下发） | ✅ | 大集群标准 |

---

## 6. FSDP（Fully Sharded Data Parallel）：参数也分片

### 6.1 为什么需要 FSDP

DDP 的前提是"单卡放得下整个模型"。但训练状态远不止参数本身。以混合精度 + Adam 为例，每个参数约需：

```
fp16 参数(2B) + fp16 梯度(2B) + fp32 参数副本(4B) + Adam 一阶动量(4B) + 二阶动量(4B)
≈ 16 字节 / 参数

  → 7B 模型：  7e9 × 16 ≈ 112 GB   ← 单张 80GB A100 都放不下！
```

DDP 在**每张卡上都存一份完整的这 16B/参数**，纯属浪费。**FSDP（对应 DeepSpeed ZeRO-3）的思路：把参数、梯度、优化器状态都切成 N 片，每张卡只常驻 1/N**，用到某层时再临时 all-gather 凑齐。

### 6.2 DDP vs FSDP 一图对比

```
DDP（每卡全量副本，冗余）
  GPU0: [完整参数][完整梯度][完整优化器状态]
  GPU1: [完整参数][完整梯度][完整优化器状态]   ← 完全重复，浪费显存
  GPU2: [完整参数][完整梯度][完整优化器状态]

FSDP（每卡只存 1/N，用时凑齐）
  GPU0: [参数分片0][梯度分片0][优化器分片0]
  GPU1: [参数分片1][梯度分片1][优化器分片1]   ← 各存一片，省 N 倍显存
  GPU2: [参数分片2][梯度分片2][优化器分片2]
  前向/反向到某层时： all-gather 临时拼出该层完整参数 → 算完即释放
  反向算梯度时：      reduce-scatter 把梯度散回各自分片
```

### 6.3 FSDP 的通信代价

- DDP 每步：1 次 all-reduce（梯度）。
- FSDP 每步：前向 all-gather（参数）+ 反向 all-gather（参数）+ 反向 reduce-scatter（梯度）≈ DDP 通信量的 **1.5 倍**。
- **用更多通信换更少显存**——这是 FSDP/ZeRO 的核心权衡。

> 概念上 FSDP ≈ DeepSpeed ZeRO Stage 3。更系统的 ZeRO 分级（Stage 1 切优化器 / Stage 2 加切梯度 / Stage 3 再切参数）见 [[ai-framework/deepspeed/README]] · [[llm-train/megatron-deepspeed/README]]。

---

## 常见问题 / 坑速查表

| 现象 / 问题 | 原因 | 解决 |
|-------------|------|------|
| 用 DP 发现 GPU0 显存爆、其他卡空闲 | DP 把输出 gather 到主卡、反向堆在主卡 | 改用 DDP（多进程无主卡瓶颈） |
| 多机 torchrun 卡在初始化不动 | 各节点 `MASTER_ADDR` / `rdzv_endpoint` 不一致，或端口被防火墙挡 | 确保所有节点指向同一 `addr:29400` 且端口连通 |
| 多机各节点 `rdzv_id` 不同 → 起不来 | 同一作业必须用相同 `rdzv_id` | 所有节点 `--rdzv_id` 设成同一个值 |
| 各卡 loss/精度不一致或发散 | 各进程初始权重不一致 / 随机种子未对齐 | DDP 构建时会广播 rank0 的参数；确保用 DDP 包装、固定种子 |
| 梯度大小不对（差 N 倍） | 本地未平均或 all-reduce 用 SUM 没除 N | DDP 内部已处理；自定义通信时记得 $1/N$ |
| `bucket_cap_mb` 调太小 → 通信变慢 | 桶太小 → all-reduce 次数多、启动开销大 | 默认 25MB 通常够用；带宽富余可调大 |
| `nproc_per_node` 设得比 GPU 数还多 | 多个进程抢同一张卡 | `nproc_per_node` 应等于该机可见 GPU 数 |
| 模型放不下单卡，DDP 直接 OOM | DDP 不切模型，只切数据 | 改用 FSDP / ZeRO-3 / 张量并行 / 流水线并行 |
| `--node_rank` 两节点都写 0 | 节点编号冲突 | 节点0 写 0，节点1 写 1，依次递增 |

---

## 🔗 官方文档

- DDP 教程：https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- DDP 设计笔记：https://pytorch.org/docs/master/notes/ddp.html
- DDP 示例：
  - https://github.com/pytorch/examples/tree/main/distributed/ddp （官方已不更新）
  - https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series
- torchrun / elastic 快速上手：https://pytorch.org/docs/stable/elastic/quickstart.html
- FSDP 入门：https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- FSDP 进阶：https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
- 中文综述：https://zhuanlan.zhihu.com/p/343951042

---

## 🔗 跳转链接

**枢纽导航：**
[[00-知识地图]] · [[llm-train/README]] · [[llm-train/pytorch/distribution/README]]

**并行训练框架：**
[[llm-train/megatron/README]] · [[llm-train/megatron-deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]]

**高效微调（PEFT）：**
[[ai-framework/huggingface-peft/README]] · [[llm-train/peft/PEFT-API]] · [[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]]

**通信与网络（DDP 底座）：**
[[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/NCCL]]

**延伸：**
[[llm-alignment/RLHF]] · [[llm-algo/transformer/模型架构]] · [[llm-compression/quantization/量化基础]] · [[B07:llm-inference/大模型推理张量并行]]
