# DeepSpeed 在 Slurm 集群上的多机训练

> 把 DeepSpeed 的"多机启动"问题交给 Slurm：用作业调度器分配节点、设置环境变量、拉起每张卡上的进程。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/deepspeed/README]] [[llm-train/pytorch/distribution/README]] [[ai-infra/ai-cluster/README]] [[ai-infra/网络/NCCL]]

## 阅读地图

| 你想知道的 | 看哪一节 |
|---|---|
| Slurm 是什么、为什么训练要用它 | §1 地基 |
| DeepSpeed 启动到底干了什么 | §2 launcher 机制 |
| 单机怎么变多机：rank/world 怎么算 | §3 分布式坐标系 |
| Slurm 怎么和 DeepSpeed 拼起来 | §4 两条集成路线 |
| `srun` vs `deepspeed --hostfile` 怎么选 | §4.2 对比 |
| 一份能跑的 sbatch 模板 | §5 配置示例 |
| 环境变量到底要设哪些 | §6 环境变量清单 |
| 卡住/连不上/rank 错乱怎么排 | 常见问题表 |

## 0. 一句话锚点

**Slurm 负责"在哪些机器上、给我几张卡"，DeepSpeed 负责"在这些卡上把训练进程拉起来并组成一个通信组"。** 二者的接缝就是一组分布式环境变量（`MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE / LOCAL_RANK`）——谁来填这组变量，就决定了你用哪种集成方式。

## 1. 地基：Slurm 解决什么问题

在单台 8 卡机器上，你直接 `deepspeed train.py` 就能跑。但训练大模型要几十上百张卡，分布在很多台物理机上，这时出现两类问题：

1. **资源争抢**：一个集群很多人用，谁能用哪几台机器、用多久，需要一个"管家"来排队和分配。
2. **多机拉起进程**：你不可能手动 ssh 到每台机器去敲命令。需要有人"代你"在每个节点上同时启动进程。

**Slurm（Simple Linux Utility for Resource Management）** 就是这个管家——一个 HPC 集群里最常见的**作业调度器**。它的核心概念：

- **Node（节点）**：一台物理机。
- **Partition（分区/队列）**：一组节点的逻辑分组（如 `gpu`、`debug`），提交作业时指定。
- **Job（作业）**：你提交的一次任务，Slurm 给它排队、分配节点、记账。
- **Task（任务）**：作业内部的并行单元。Slurm 通过 `srun` 在分配到的节点上把每个 task 当成一个进程拉起来。
- **GRES（Generic RESource）**：通用资源，GPU 就是通过 `--gres=gpu:8` 来申请的。

```
              ┌──────────────────────────────────────────────┐
   你 ─sbatch─▶│              Slurm 控制器 (slurmctld)          │
              │   排队 → 找到满足条件的节点 → 分配 → 记账        │
              └───────────────┬──────────────────────────────┘
                              │ 在每个被分配节点上启动 slurmd
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ node-01  │   │ node-02  │   │ node-03  │   ← 物理机
        │ GPU0..7  │   │ GPU0..7  │   │ GPU0..7  │
        └──────────┘   └──────────┘   └──────────┘
              └──── 高速互联 (IB / RoCE) ─────┘
```

> 一句话：Slurm 给你一批节点 + 一组描述"这批节点长什么样"的环境变量，剩下的训练逻辑由 DeepSpeed/PyTorch 接管。

## 2. DeepSpeed 的 launcher 机制：它到底干了什么

文件顶部那段就是 DeepSpeed 内部支持的几种**多机启动后端（launcher）**：

```
PDSH_LAUNCHER   = 'pdsh'      # 并行 ssh，默认；靠 ssh 在各节点拉起进程
PDSH_MAX_FAN_OUT = 1024       # pdsh 一次最多并发连多少节点

OPENMPI_LAUNCHER = 'openmpi'  # 走 MPI 的 mpirun
MPICH_LAUNCHER   = 'mpich'    # MPICH 实现
IMPI_LAUNCHER    = 'impi'     # Intel MPI
SLURM_LAUNCHER   = 'slurm'    # 走 srun，由 Slurm 来拉进程
MVAPICH_LAUNCHER = 'mvapich'  # MVAPICH (IB 优化的 MPI)
```

**关键理解：DeepSpeed 自己并不会"魔法地"出现在所有机器上。** 当你敲 `deepspeed train.py` 时，它做了三件事：

1. **读取节点清单（hostfile）**：知道总共有哪些机器、每台几张卡（GPU slots）。
2. **选一个 launcher**：默认用 **pdsh**（并行 ssh）远程登录到每台机器。
3. **在每个节点上启动一个 launcher 子进程**，再由它在本机用 `torch` 风格的方式拉起 `nproc_per_node` 个训练进程，并给每个进程注入 `RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT`。

```
deepspeed --hostfile=hosts train.py
        │
        ├─ 解析 hostfile：node-01 slots=8 / node-02 slots=8 ...
        │
        ├─ launcher=pdsh ──ssh──▶ node-01: 拉起 8 个进程 (local_rank 0..7)
        │                   └ssh─▶ node-02: 拉起 8 个进程 (local_rank 0..7)
        │
        └─ 给每个进程注入: RANK=全局序号, WORLD_SIZE=16, MASTER_ADDR=node-01 ...
```

**为什么需要 Slurm launcher？** 因为在 Slurm 管的集群里：

- 计算节点之间常常**禁用了 ssh 互连**（安全策略），pdsh 直接连不上 → 默认方式失效。
- Slurm 已经替你分配好了节点、做了进程隔离和记账，再用 pdsh 自己 ssh 进去会"绕过调度器"，资源对不上账。

所以在 Slurm 集群里，正确做法是**让 Slurm 来当 launcher**，而不是 pdsh。这就引出了下一节的坐标系问题——谁来填环境变量。

## 3. 分布式坐标系：rank / world_size 怎么算

任何 PyTorch/DeepSpeed 分布式作业，本质是 $N$ 个进程组成一个通信组，每个进程需要知道四个数：

| 变量 | 含义 | 例子（2 机 × 8 卡）|
|---|---|---|
| `WORLD_SIZE` | 总进程数 = 总卡数 | $2 \times 8 = 16$ |
| `RANK`（global rank）| 我在全局是第几个进程，$0 \le \text{RANK} < \text{WORLD\_SIZE}$ | $0 \sim 15$ |
| `LOCAL_RANK` | 我在本机是第几张卡，$0 \sim 7$ | 决定绑哪块 GPU |
| `MASTER_ADDR/PORT` | rendezvous（会合点），rank 0 进程的地址 | node-01:29500 |

全局 rank 的标准换算：

$$\text{RANK} = \text{node\_id} \times \text{gpus\_per\_node} + \text{LOCAL\_RANK}$$

例如 node-02（node_id=1）上 local_rank=3 的进程，global rank $= 1 \times 8 + 3 = 11$。

```
   WORLD_SIZE = 16
   ┌────────── node-01 (MASTER) ──────────┐  ┌────────── node-02 ──────────┐
   │ GPU0 GPU1 GPU2 ... GPU7              │  │ GPU0 GPU1 ... GPU7          │
   │ rank0 rank1 rank2 ... rank7          │  │ rank8 rank9 ... rank15      │
   │  ▲ MASTER_ADDR=node-01:29500 在这    │  │                             │
   └──────────────────────────────────────┘  └─────────────────────────────┘
              所有 16 个进程通过 MASTER 会合，建立 NCCL 通信组
```

**Slurm 集成的核心，就是用 Slurm 提供的变量去推导出上面这组变量。** Slurm 在每个 task 进程里会自动设好一批 `SLURM_*` 环境变量（具体名字以 Slurm 官方文档为准），常用的语义类别有：

- 本作业拿到的节点列表（用于推导 `MASTER_ADDR`，通常取列表第一个节点）；
- 本作业的总 task 数（可对应 `WORLD_SIZE`）；
- 当前 task 的全局编号（可对应 `RANK`）；
- 当前 task 在本节点内的编号（可对应 `LOCAL_RANK`）。

> 具体的 `SLURM_*` 变量名、是否包含某个量，不同 Slurm 版本/配置有差异，**以 `srun env | grep SLURM` 实测和官方文档为准**，不要硬背。

## 4. 两条集成路线

### 4.1 路线 A：srun 直接当 launcher（推荐用于纯 Slurm 集群）

思路：**不用 `deepspeed` 命令，改用 `srun python train.py`**，让 Slurm 在每个 GPU 上拉起一个 task，再在脚本里用 `SLURM_*` 变量翻译成 PyTorch 需要的变量。

```
sbatch 脚本
   └─ srun python train.py   (--ntasks=16 --ntasks-per-node=8 --gres=gpu:8)
            │  Slurm 在 16 个 task 上各跑一份 train.py
            ▼
      train.py 里:
        os.environ['RANK']       = os.environ['SLURM_PROCID']
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
        os.environ['MASTER_ADDR']= <节点列表第一个>
        deepspeed.init_distributed()  # 直接读这些标准变量建组
```

- 优点：完全走 Slurm，不依赖节点间 ssh；资源记账准确；和 `torchrun` 的 Slurm 用法一致。
- 注意：每个 task 只对应**一张卡**，所以训练脚本里要用 `LOCAL_RANK` 来 `torch.cuda.set_device()`，DeepSpeed 用 `init_distributed()` 从环境变量建组。

### 4.2 路线 B：deepspeed --launcher=slurm（用 DeepSpeed 自带的 Slurm 后端）

思路：仍然用 `deepspeed` 命令，但显式告诉它"别用 pdsh，用 srun"：

```
deepspeed --launcher slurm --hostfile=... train.py
          └ DeepSpeed 内部改用 srun 在各节点拉进程，而非 ssh
```

或者在 sbatch 里申请好节点后，用 DeepSpeed 提供的 Slurm 适配把节点信息喂给它。

> 路线 B 的确切命令行选项（如 `--launcher`、`--launcher_args`）以你安装的 DeepSpeed 版本 `deepspeed --help` 输出为准，**不要硬记**。

### 对比

| 维度 | 路线 A：srun python | 路线 B：deepspeed --launcher slurm |
|---|---|---|
| 谁拉进程 | Slurm（srun） | DeepSpeed 调 srun |
| 环境变量谁填 | 你在脚本里翻译 SLURM_* | DeepSpeed 帮你填 |
| 对节点 ssh 的依赖 | 无 | 无（用 srun）|
| 心智模型 | 和 torchrun 一致，透明 | 保留 deepspeed CLI 习惯 |
| 适合 | 纯 Slurm、想完全可控 | 已有大量 deepspeed 脚本想少改 |

两条路线**底层是一回事**：都要把那四个分布式变量正确设到每个进程里。区别只是"谁来填"。

## 5. 配置示例：一份可参考的 sbatch 模板（路线 A）

下面是结构示意，重在**说明每一行干什么**，具体参数值按你的集群调整：

```bash
#!/bin/bash
#SBATCH --job-name=ds-train          # 作业名，方便 squeue 里认
#SBATCH --partition=gpu              # 提交到哪个分区/队列
#SBATCH --nodes=2                    # 申请 2 台节点
#SBATCH --ntasks-per-node=8          # 每节点 8 个 task（=8 张卡，一卡一进程）
#SBATCH --gres=gpu:8                 # 每节点申请 8 块 GPU（GRES）
#SBATCH --cpus-per-task=8            # 每个 task 配几个 CPU 核（数据加载用）
#SBATCH --time=24:00:00              # 墙钟时间上限，超时会被杀
#SBATCH --output=logs/%x-%j.out      # 日志文件，%x=jobname %j=jobid

# 1) 选出 master 节点：通常取分配到的节点列表第一个
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500             # 任选一个空闲端口，做 rendezvous

# 2) NCCL 相关（按你的网卡/IB 调整，详见下一节）
export NCCL_DEBUG=INFO               # 出问题时看 NCCL 选了哪条路

# 3) 让 srun 在 16 个 task 上各拉一份 train.py
#    --ntasks 默认 = nodes * ntasks-per-node = 16
srun python train.py \
     --deepspeed --deepspeed_config ds_config.json
```

训练脚本 `train.py` 里只需把 Slurm 变量翻译成标准变量（概念示意）：

```python
import os, deepspeed
os.environ["RANK"]       = os.environ["SLURM_PROCID"]   # 全局 rank
os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]   # 总进程数
os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]  # 本机内卡号
# MASTER_ADDR / MASTER_PORT 已由 sbatch 脚本 export，子进程能继承

deepspeed.init_distributed()        # 读上面这些标准变量，建立通信组
# 之后 model_engine, _, _, _ = deepspeed.initialize(...) 照常
```

**为什么这样能跑通？** 因为 `srun` 启动的每个 task 进程都会**继承** sbatch 脚本里 `export` 的变量（`MASTER_ADDR/PORT/NCCL_*`），同时 Slurm 又给每个 task 注入了它自己的 `SLURM_PROCID/LOCALID/NTASKS`。两者合起来，刚好凑齐了 §3 那四个变量。

## 6. 环境变量清单（讲含义，不背默认值）

| 变量 | 谁设 | 作用 |
|---|---|---|
| `MASTER_ADDR` | 你（取节点列表第一个）| rendezvous 地址，所有进程来这里会合 |
| `MASTER_PORT` | 你（选空闲端口）| 会合端口，多作业同机要避免撞端口 |
| `WORLD_SIZE` | Slurm→你翻译 | 总进程数 = 总 GPU 数 |
| `RANK` | Slurm→你翻译 | 全局唯一序号，rank 0 是 master |
| `LOCAL_RANK` | Slurm→你翻译 | 绑定本机第几块 GPU |
| `NCCL_SOCKET_IFNAME` | 你（按网卡名）| 指定 NCCL 走哪块网卡（如 `ib0`/`eth0`），多网卡环境常踩 |
| `NCCL_IB_HCA` | 你（IB 环境）| 指定用哪些 InfiniBand HCA |
| `NCCL_DEBUG` | 你 | `INFO` 可看 NCCL 拓扑/路径选择，排障必开 |
| `CUDA_VISIBLE_DEVICES` | 慎用 | 一卡一进程时一般**别手动设**，交给 Slurm/LOCAL_RANK，否则容易和绑定冲突 |

> NCCL 相关变量是**多机训练最容易卡住的地方**：常见症状是"建组卡死、不报错"，多半是 NCCL 选错网卡或 IB 没走通。先 `NCCL_DEBUG=INFO` 看日志，再用 `NCCL_SOCKET_IFNAME` 显式指定网卡。详见 [[ai-infra/网络/NCCL]]。

## 常见问题 / 坑

| 现象 | 根因 | 处理方向 |
|---|---|---|
| 节点间连不上、pdsh 报 ssh 失败 | Slurm 集群禁用了计算节点互 ssh | 别用默认 pdsh，改 srun 当 launcher（路线 A/B）|
| 进程数不对、rank 重复或缺号 | `--ntasks-per-node` 与 `--gres=gpu` 数不一致 | 让"每节点 task 数 = 每节点 GPU 数"，一卡一进程 |
| 建组卡死，不报错也不退出 | MASTER_ADDR 不可达 / NCCL 选错网卡 | 确认 master 用的是节点列表第一个；设 `NCCL_SOCKET_IFNAME`；开 `NCCL_DEBUG=INFO` |
| `Address already in use` | 同机多作业撞了 MASTER_PORT | 给 MASTER_PORT 加随机/按 jobid 取值 |
| 所有进程都抢同一块 GPU | 没用 LOCAL_RANK 做 `set_device`，或手设了 `CUDA_VISIBLE_DEVICES` | 用 `LOCAL_RANK` 绑卡，去掉手动 `CUDA_VISIBLE_DEVICES` |
| 作业超时被杀但没存 checkpoint | `--time` 到点 Slurm 直接 kill | 设合理 `--time`，并用 Slurm 信号 + DeepSpeed 周期性存 ckpt |
| 多机比单机慢很多 | 没走 IB，退化到以太网 | 检查 `NCCL_IB_HCA`/网卡，确认走的是高速互联 |
| 环境变量翻译写错（如 RANK/LOCAL_RANK 弄反）| Slurm 变量语义记错 | 先 `srun env | grep SLURM` 实测每个变量的真实值再映射 |

**实践要点：**
- 一卡一进程是多机训练的标准模型：`ntasks-per-node == gpus_per_node`。
- master 永远取**分配到的节点列表的第一个**，用 `scontrol show hostnames` 解析，不要写死主机名。
- 任何"卡住但不报错"的多机问题，第一反应是 **NCCL + 网络**，先开 `NCCL_DEBUG=INFO`。
- 具体的 `SLURM_*` 变量名、`deepspeed` CLI 选项、NCCL 变量默认值，**一律以官方文档与实测为准**，本文讲的是机制不是定值。

## 🔗 跳转链接

- [[00-知识地图]]
- [[ai-framework/deepspeed/README]] — DeepSpeed 总览
- [[llm-train/pytorch/distribution/README]] — PyTorch 分布式与多机训练
- [[ai-infra/ai-cluster/README]] — AI 集群与调度
- [[ai-infra/网络/NCCL]] — NCCL 通信与排障
- 参考：Slurm MPI Guide（https://slurm.schedmd.com/mpi_guide.html）
