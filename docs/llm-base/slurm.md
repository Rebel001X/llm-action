# Slurm 集群作业调度系统

> 一句话定位：Slurm 是 HPC / 大模型训练集群的"操作系统调度内核"——把成百上千张 GPU 抽象成资源池，按队列和优先级把训练/推理作业分配到计算节点上。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]

---

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
|------|--------------|--------|
| 0. 一句话锚点 | Slurm 到底解决什么问题 | 资源仲裁、排队 |
| 1. 地基/前置 | 为什么 HPC 不能直接 ssh 跑 | 共享集群、争抢 |
| 2. 架构与守护进程 | slurmctld/slurmd/slurmdbd/slurmrestd | 中心管理器 |
| 3. 节点角色 | Head/Compute/Login/SlurmDBD | 物理拓扑 |
| 4. 用户与账户模型 | account/user/QOS/association | 计费与权限 |
| 5. 资源抽象 | GRES/TRES/Partition | GPU 调度 |
| 6. 三种提交方式 | srun/sbatch/salloc | 交互 vs 批处理 |
| 7. 作业生命周期 | 排队→分配→运行→完成 | 状态机 |
| 实操命令 | 全套 CLI + Python 并行例子 | 手册速查 |
| 常见坑 | 环境变量、资源不匹配 | 排错 |

---

## 0. 一句话锚点

> **Slurm = Simple Linux Utility for Resource Management。** 它是一个开源的集群作业调度器：用户把"我要 8 张 A100、运行 24 小时"这样的请求提交给它，Slurm 负责**排队、仲裁、分配节点、启动进程、回收资源**。

在大模型训练里，一次 Megatron/DeepSeed 训练动辄占用几十上百张卡，跨多机多卡。没有调度器时多人共用集群会互相踩踏；Slurm 把"谁先用、用多少、用多久"变成可治理的策略。

> 原文参考链接：Slurm 简介 — http://hmli.ustc.edu.cn/doc/linux/slurm-install/slurm-install.html

---

## 1. 地基 / 前置：为什么需要调度器

**场景对照**：

```
裸机 ssh 直接跑                     Slurm 调度
──────────────────                 ──────────────────
你 ssh 到 node5，nvidia-smi         你 sbatch 提交一个脚本，
看哪张卡空，手动 CUDA_VISIBLE         声明"我要 8 卡 / 2 节点 / 24h"
↓                                   ↓
3 个人同时挑中同一张卡 → OOM         Slurm 维护全局视图，
作业互相抢显存/CPU/带宽              保证分给你的卡别人拿不到
↓                                   ↓
没人知道谁在跑什么、跑了多久          sacct 记账：谁、用了多少机时
```

> **核心矛盾**：集群是**共享**的，GPU/CPU/内存/网络带宽都是**有限且独占性**的资源。调度器的价值就是把"无序争抢"变成"有序排队 + 强隔离"。

原文一句话总结了入口：

> 所有需运行的作业，无论是用于程序调试还是业务计算，都可以通过**交互式并行 srun**、**批处理式 sbatch** 或**分配式 salloc** 等命令提交，提交后可以利用相关命令查询作业状态。

---

## 2. 架构与守护进程（原文真料 + 原理）

Slurm 是典型的**中心管理器 + 计算代理**架构。原文列出了 4 个守护进程，逐一拆解：

| 守护进程 | 全称 | 跑在哪 | 职责 | 是否必需 |
|----------|------|--------|------|----------|
| `slurmctld` | Slurm Controller Daemon | Head Node | **中心大脑**：监测资源和作业、做调度决策 | ✅ 必需 |
| `slurmd` | Slurm Daemon | 每个 Compute Node | **远程 shell**：等待作业→执行→返回状态→再等待 | ✅ 必需 |
| `slurmdbd` | Slurm DataBase Daemon | SlurmDBD Node | 记账信息入库（可记多个集群） | ⭕ 非必需，建议采用 |
| `slurmrestd` | Slurm REST API Daemon | 通常 Head | 通过 REST API 与 Slurm 交互，功能均有对应 API | ⭕ 非必需 |

原文精确描述：
- `slurmctld` 作为中心管理器用于**监测资源和作业**，为提高可用性还可配置**另一个备份冗余管理器**。
- 各计算节点需启动 `slurmd`，以便作为**远程 shell** 使用：**等待作业、执行作业、返回状态、再等待更多作业**。
- `slurmdbd` 可以将**多个 slurm 管理的集群的记账信息记录在同一个数据库中**（也可记录到纯文本中）。
- `slurmrestd` 让所有功能都有**对应的 API**。

### 架构图

```
                       ┌──────────────────────────┐
                       │      Head / 控制节点        │
   提交作业            │   ┌────────────────────┐   │
 ┌─────────┐  srun    │   │     slurmctld      │   │  备份冗余
 │ Login   │ ───────► │   │  (中心管理器/大脑)   │◄──┼──► slurmctld
 │ 登录节点 │  sbatch  │   │  调度·资源仲裁·心跳  │   │   (HA 备机)
 └─────────┘          │   └─────────┬──────────┘   │
                       └─────────────┼──────────────┘
            记账查询 sacct            │ 下发任务 / 收心跳
                  │                  ▼
          ┌───────┴──────┐   ┌───────────────────────────────┐
          │  slurmdbd    │   │           计算节点群            │
          │ (数据库守护)  │   │  ┌────────┐ ┌────────┐ ┌─────┐ │
          │  MySQL/Maria │   │  │ slurmd │ │ slurmd │ │ ... │ │
          └──────────────┘   │  │ node1  │ │ node2  │ │     │ │
                             │  │ GPU×8  │ │ GPU×8  │ │     │ │
                             │  └────────┘ └────────┘ └─────┘ │
                             └───────────────────────────────┘
```

> **为什么 slurmctld 要做主备（HA）？** 它是单点：一旦挂掉，整个集群无法提交/调度作业（已在跑的作业不受影响）。配置 backup controller 后，主机宕机时备机接管状态文件（StateSaveLocation 共享存储），实现秒级切换。

> **slurmd 为什么叫"远程 shell"？** 它本质上是一个常驻代理，听 slurmctld 的命令，在本节点 fork 出用户进程、设置 cgroup 限制、收集退出码。对用户透明，但隔离全靠它。

---

## 3. 节点角色（原文真料）

原文给出了 4 类节点定义，这是物理拓扑的语言：

| 角色 | 原文定义 | 跑什么 |
|------|----------|--------|
| **Head Node** | 头节点 / 管理节点 / 控制节点，运行 `slurmctld` 管理服务的节点 | slurmctld |
| **Compute Node** | 计算节点，运行作业计算任务的节点 | slurmd（必需）|
| **Login Node** | 用户登录节点，用于用户登录的节点 | 用户 ssh 进来提交作业 |
| **SlurmDBD Node** | 存储调度策略、记账和作业等信息的节点 | slurmdbd（必需在该节点）|

> 原文补充：**客户节点 = 计算节点 + 用户登录节点**。

```
用户的视角（从外到内）：

  外网/办公网
      │  ssh
      ▼
 ┌──────────┐      ┌──────────┐      ┌────────────────┐
 │ Login    │────► │ Head     │────► │ Compute ×N      │
 │ 写脚本    │ 提交  │ slurmctld│ 调度  │ slurmd 跑你的活  │
 │ 编译代码  │      │ 排队仲裁  │      │ GPU/InfiniBand  │
 └──────────┘      └──────────┘      └────────────────┘
```

> **重要纪律**：登录节点 **不是** 用来跑训练的！直接在 Login Node 上 `python train.py` 会拖垮所有人的登录体验，且不受调度隔离。一切重活都要经 `srun`/`sbatch` 下发到 Compute Node。

---

## 4. 用户与账户模型（原文真料 + 计费原理）

Slurm 把"谁"和"花了多少钱"分得很细。原文术语对照：

| 术语 | 原文定义 | 通俗理解 |
|------|----------|----------|
| `account` | 账户，一个账户可含有多个用户 | 课题组 / 项目组 |
| `user` | 用户，多个用户可以共享一个账户 | 你的登录名 |
| `bank account` | 银行账户，对应机时费等 | 余额池，跑作业扣机时 |
| `association` | 关联。若用户的关联不在数据库中，将**阻止用户运行作业**，可阻止访问无效账户 | (user, account, partition, qos) 的绑定关系 |

```
account = "lab-nlp"  (实验室)
   ├── user = alice   ──┐
   ├── user = bob     ──┼── 共享同一个 bank account（机时余额）
   └── user = carol   ──┘
                          每跑一个作业 → 按 TRES×时长 扣机时

association 把它串起来：
   (alice, lab-nlp, gpu分区, normal-qos) ✅ 在库 → 允许提交
   (dave,  lab-nlp, ...) ❌ 不在库 → 被拒绝运行
```

> **为什么 association 能"阻止访问无效账户"？** Slurm 调度前查 slurmdbd：这个 (用户, 账户, 分区, QOS) 四元组是否注册过、余额是否够、是否超限额。任一不满足直接拒绝——这是集群计费和配额治理的根。

---

## 5. 资源抽象：GRES / TRES / QOS / Partition（原文真料）

这是 Slurm 调度 GPU 的核心。原文给出 5 个术语：

| 术语 | 全称 | 原文定义 | 在大模型场景的意义 |
|------|------|----------|---------------------|
| **GRES** | Generic Resource | 通用资源 | **GPU 就是典型 GRES**，如 `--gres=gpu:8` |
| **TRES** | Trackable RESources | 可追踪资源 | 计费维度：CPU、内存、GPU、能耗等都可计入 |
| **QOS** | Quality of Service | 服务质量，作业优先级 | 高优先级队列插队、限额、抢占 |
| **association** | 关联 | 见上节 | 权限与配额绑定 |
| **Partition** | 队列 / 分区 | 对节点、并行规模、作业时长、用户分组管理，合理分配资源 | 把 A100 节点和 V100 节点分到不同队列 |

### GRES：GPU 是怎么被调度的

```
节点 gpu-node1 在 slurm.conf / gres.conf 里声明：
    Gres=gpu:a100:8        ← 这台机器有 8 张 A100

你的作业声明需求：
    --gres=gpu:2           ← 我只要 2 张

slurmctld 仲裁后：
    把 gpu-node1 的 GPU 0,1 标记为你独占
    通过 cgroup 设置 CUDA_VISIBLE_DEVICES，别人看不到这 2 张
```

> **Partition 与 QOS 的分工**：
> - **Partition** 是"硬分组"——一批节点 + 默认时长/规模上限。例如 `gpu`（跑训练，48h 上限）、`debug`（调试，30 分钟上限）。
> - **QOS** 是"软策略"——叠在 partition 之上调优先级/抢占/限额。例如 `high` QOS 优先级更高、能抢占 `low` QOS 的作业。

### 数值示例：机时（TRES-时）怎么算

假设某集群按 GPU-小时计费，QOS 给 GPU 的权重系数 = 1.0：

$$
\text{机时} = \sum_i (\text{TRES}_i \times \text{weight}_i) \times \text{运行时长}
$$

跑一个作业：8 张 GPU × 24 小时，GPU 权重 1.0：

$$
\text{机时} = 8 \times 1.0 \times 24 = 192 \text{ GPU-小时}
$$

bank account 余额若只剩 100 GPU-小时，association 检查时就会**拒绝提交**（超额）。

---

## 6. 三种提交方式：srun / sbatch / salloc

原文一句话给了三种入口，这里展开它们的本质区别：

```
            交互式            批处理式           分配式
          ┌─────────┐      ┌─────────┐      ┌─────────┐
          │  srun   │      │ sbatch  │      │ salloc  │
          └────┬────┘      └────┬────┘      └────┬────┘
 阻塞前台?  是(看到输出)      否(后台跑)        是(给你一个 shell)
 适合       调试/单步         正式训练任务      手动多步交互
 输出       直接打到终端       写到 .out 文件    你自己 srun
```

| 命令 | 行为 | 典型用法 |
|------|------|----------|
| `srun` | **交互式并行**：阻塞终端，实时看输出，作业结束才返回 | 调试、跑一行命令 |
| `sbatch` | **批处理式**：提交一个脚本到队列，立即返回 jobid，后台排队执行 | 正式训练（首选）|
| `salloc` | **分配式**：先抢到一组节点的资源，给你一个交互 shell，你再在里面手动 `srun` | 多步交互式实验 |

```bash
# srun：交互跑一句
srun --gres=gpu:1 --pty bash      # 抢 1 卡，进入交互 shell

# sbatch：提交训练脚本（推荐）
sbatch train_llm.sbatch           # 返回 Submitted batch job 12345

# salloc：先要资源再操作
salloc -N 2 --gres=gpu:8          # 抢 2 节点 16 卡
srun python -m torch.distributed.run ...   # 在分配里启动
```

### 一个 sbatch 训练脚本骨架（大模型多机多卡）

```bash
#!/bin/bash
#SBATCH --job-name=llm-pretrain
#SBATCH --partition=gpu           # 用哪个队列（分区）
#SBATCH --nodes=2                 # 2 个节点
#SBATCH --ntasks-per-node=8       # 每节点 8 个任务（对应 8 卡）
#SBATCH --gres=gpu:8              # 每节点 8 张 GPU（GRES）
#SBATCH --cpus-per-task=8         # 每任务 8 个 CPU 核
#SBATCH --time=24:00:00           # 最长 24h（受 partition 上限约束）
#SBATCH --output=logs/%x-%j.out   # %x=作业名 %j=jobid

srun python -m torch.distributed.run \
     --nnodes=$SLURM_NNODES \
     --nproc_per_node=8 \
     train.py
```

> **为什么用 `srun` 启动而不是直接 `python`？** `srun` 会在 Slurm 分配的每个节点上拉起进程，并注入 `SLURM_*` 环境变量（节点列表、rank 等），让 PyTorch 分布式能自动拼出 `MASTER_ADDR`、`WORLD_SIZE`。

---

## 7. 作业生命周期：状态机

```
 提交           排队            分配           运行           结束
sbatch ──► PENDING(PD) ──► (调度命中) ──► RUNNING(R) ──► COMPLETED(CD)
              │  资源不够/优先级低 等待        │ 节点跑 slurmd        │
              │                              │ ↘ 失败 → FAILED(F)   │
              │  scancel ──────────────────► CANCELLED(CA)          │
              └────────────────────────────────────────────────────┘
                       超时 --time 到 ──► TIMEOUT(TO)
```

| 状态码 | 含义 | 常见原因 |
|--------|------|----------|
| `PD` (PENDING) | 排队中 | 资源被占满 / 优先级未到 / 余额不足 |
| `R` (RUNNING) | 运行中 | 正常 |
| `CD` (COMPLETED) | 正常结束（退出码 0）| — |
| `F` (FAILED) | 失败（非 0 退出）| 代码报错、OOM |
| `CA` (CANCELLED) | 被取消 | scancel / 管理员 |
| `TO` (TIMEOUT) | 超时杀死 | `--time` 设短了 |

---

## 实操：用户工具命令全集（原文真料）

原文列出的用户工具，逐条标注（保留原文，补全示例）：

| 命令 | 原文职责 | 速查示例 |
|------|----------|----------|
| `srun` | 运行作业 | `srun --gres=gpu:1 nvidia-smi` |
| `scancel` | 终止排队中或运行中的作业 | `scancel 12345`（杀单个）/ `scancel -u alice`（杀某用户全部）|
| `sinfo` | 查看系统状态 | `sinfo -N -l`（节点级详细）|
| `squeue` | 查看作业状态 | `squeue -u alice`（看我的作业）|
| `sacct` | 查看运行中或结束了的作业及作业步信息 | `sacct -j 12345 --format=JobID,State,Elapsed,MaxRSS` |
| `sview` | 图形化显示系统和作业状态（可含网络拓扑）| GUI |
| `scontrol` | 管理工具：监控、修改集群配置和状态信息 | `scontrol show job 12345` / `scontrol show node gpu1` |
| `sacctmgr` | 管理数据库：认证集群、有效用户、有效记账账户等 | `sacctmgr add user alice account=lab-nlp` |

```bash
# 看集群有哪些分区、节点空闲情况
sinfo
# PARTITION AVAIL TIMELIMIT NODES STATE NODELIST
# gpu*      up    2-00:00:00   2  idle  gpu[1-2]

# 看我自己的作业排队/运行
squeue -u $USER

# 看某作业为什么还在排队
scontrol show job 12345 | grep Reason

# 作业结束后查资源占用（峰值内存、用时）
sacct -j 12345 --format=JobID,JobName,State,Elapsed,MaxRSS,ReqGRES
```

### Python 多进程并行示例（原文代码，原样保留 + 注释原理）

原文给出了一个标准的"在 Slurm 分配的核数上做多进程并行"的例子。**关键点：用 `SLURM_CPUS_PER_TASK` 环境变量读取 Slurm 实际分给你的核数，而不是写死。**

```python
import os
from multiprocessing import Pool, cpu_count

# function you want to run in parallel:
def myfunction(a, b):
  return a + b

# list of tuples to serve as arguments to function:
args = [(1, 2), (9, 11), (6, 2)]

# number of cores you have allocated for your slurm task:
number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])

print("number_of_cores: ", number_of_cores)


# number_of_cores = cpu_count() # if not on the cluster you should do this instead

# multiprocssing pool to distribute tasks to:
with Pool(number_of_cores) as pool:
    # distribute computations and collect results:
    results = pool.starmap(myfunction, args)
```

> **为什么不用 `cpu_count()`？** `cpu_count()` 返回的是**物理机的总核数**（比如 128 核），但 Slurm 通过 cgroup **只分给你 8 核**（`--cpus-per-task=8`）。若开 128 个进程争抢 8 核，会严重过订（oversubscription）拖慢，还可能触发资源超限被杀。读 `SLURM_CPUS_PER_TASK` 才能恰好匹配分配。

> 原文已注释了**不在集群上时**的退路：`number_of_cores = cpu_count()`。

### 常用 SLURM_* 环境变量

| 变量 | 含义 | 用途 |
|------|------|------|
| `SLURM_CPUS_PER_TASK` | 每任务分到的 CPU 核数 | 设 Pool 大小 / OMP 线程数 |
| `SLURM_JOB_ID` | 作业号 | 日志命名、checkpoint 目录 |
| `SLURM_NNODES` | 节点数 | 拼分布式 world |
| `SLURM_NODELIST` | 分到的节点名列表 | 算 MASTER_ADDR |
| `SLURM_PROCID` | 全局进程 rank | 分布式 rank |
| `SLURM_LOCALID` | 节点内本地 rank | 绑 GPU |

---

## 常见问题 / 坑

| 现象 | 根因 | 解决 |
|------|------|------|
| 作业一直 `PD (PENDING)` 不动 | 资源不足 / 优先级低 / 余额不足 / association 不在库 | `scontrol show job <id>` 看 `Reason` 字段 |
| `Invalid account or account/partition combination` | association 未注册该 (user, account, partition, qos) | 找管理员 `sacctmgr add` |
| 多进程开了上百个但很慢 | 用了 `cpu_count()`（物理总核），过订严重 | 改用 `SLURM_CPUS_PER_TASK` |
| GPU 训练看不到卡 / `CUDA_VISIBLE_DEVICES` 为空 | 没声明 `--gres=gpu:N` | sbatch 里加 `#SBATCH --gres=gpu:8` |
| 作业被 `TIMEOUT` 杀掉 | `--time` 设太短，或超过 partition 上限 | 调大 `--time`，或换长时长分区 |
| 直接在登录节点跑训练拖垮集群 | Login Node 不是用来算的 | 一律 `srun`/`sbatch` 下发到 Compute Node |
| OOM 被杀但日志不明显 | cgroup 内存限制（`--mem`）小于实际占用 | `sacct ... MaxRSS` 看峰值，调大 `--mem` |
| slurmctld 宕机后无法提交 | 控制器单点 | 配置 backup controller（HA）|
| 多机训练 rank 拼不出来 | 没用 `srun` 启动，缺 SLURM_* 变量 | 用 `srun` 拉起，读 `SLURM_NODELIST` |

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 训练框架（Slurm 上最常跑的负载）：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]]
- 训练总览：[[llm-train/README]] · [[llm-train/peft/PEFT-API]]
- 硬件与算力（被 Slurm 调度的对象）：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/昇腾NPU]]
- 多机互联（多节点训练的网络底座）：[[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]
- 性能与评估：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
- 推理侧（集群也跑推理服务）：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/PD分离]]
