# Slurm 多机多卡训练：作业调度与分布式启动

> 一句话定位：Slurm 是 HPC/GPU 集群的「资源调度大脑」，负责把你的训练脚本调度到多台机器的多张卡上；本文讲透 Slurm + torchrun / DeepSpeed / Singularity 的多机多卡启动链路。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/pytorch/distribution/README]] · [[llm-train/megatron-deepspeed/README]] · [[ai-framework/deepspeed/README]] · [[ai-infra/网络/NCCL]]

---

## 阅读地图

| 小节 | 你将搞懂 | 关键词 |
| --- | --- | --- |
| 0. 一句话锚点 | Slurm 到底在整个训练栈里站哪一层 | 调度器 vs 通信库 |
| 1. 地基 | 集群、节点、作业、任务、GPU 的层级关系 | node / task / ntasks |
| 2. Slurm 架构 | slurmctld / slurmd / 三大命令怎么协作 | sbatch / srun / squeue |
| 3. MUNGE 免密 | 多机为什么能互相信任、不用 SSH 密码 | UID/GID 认证 |
| 4. sbatch 脚本 | `#SBATCH` 指令逐行解释 | nodes / ntasks / gpus-per-task |
| 5. Slurm + torchrun | rendezvous 怎么把 4 台机器组成一个进程组 | rdzv / head_node |
| 6. DeepSpeed 启动 | 单机/多机/docker/singularity 四种姿势 | --include / pmi2 |
| 7. MPI 与 PMI2 | Slurm 怎么帮 MPI 程序拉起进程 | --mpi=pmi2 |
| 实操命令 | 原文所有真命令一处汇总 | sbatch/squeue/scancel |
| 常见坑 | 多机训练最容易卡死的地方 | NCCL/IB/网卡 |

---

## 0. 一句话锚点

**Slurm 管「在哪台机器、用哪几张卡跑」；torchrun / DeepSpeed 管「这些进程怎么组成一个分布式组」；NCCL 管「这些进程之间怎么传梯度」。** 三者各司其职，不要混淆。

```
  你的角色            Slurm                 torchrun/DeepSpeed         NCCL
 ┌────────┐  sbatch  ┌──────────┐  拉起进程 ┌──────────────┐ 建通信 ┌─────────┐
 │ 提交作业├────────▶│ 分配资源 ├─────────▶│ 组建进程组   ├──────▶│ AllReduce│
 │ .slurm │          │ 4节点4卡 │          │ rank0..rank3 │       │ 同步梯度 │
 └────────┘          └──────────┘          └──────────────┘        └─────────┘
   调度层(本文)         资源层                 进程层                  通信层
```

---

## 1. 地基：节点 / 任务 / GPU 的层级

在动手前必须分清 Slurm 的四级概念，否则 `#SBATCH` 参数永远填不对：

| 术语 | 含义 | 类比 |
| --- | --- | --- |
| **Cluster** | 一整套被 Slurm 管理的机器 | 一栋楼 |
| **Node（节点）** | 一台物理服务器（含若干 GPU） | 一个房间 |
| **Job（作业）** | 你提交的一次资源请求 | 一次订房 |
| **Task（任务）** | 作业里被拉起的一个进程 | 房间里的一个人 |
| **GPU** | 分配给 task 的加速卡 | 人手里的工具 |

关键直觉：**1 个 GPU 对应 1 个 task 对应 1 个 rank（进程）** 是大模型训练最常见的布局。原文脚本里 `--nodes=4 --ntasks=4 --gpus-per-task=1` 就是「4 台机器，每台 1 个进程，每个进程 1 张卡」，总共 4 个 rank。

```
nodes=4, ntasks=4, gpus-per-task=1
 ┌── node0 ──┐ ┌── node1 ──┐ ┌── node2 ──┐ ┌── node3 ──┐
 │ task/rank0│ │ task/rank1│ │ task/rank2│ │ task/rank3│
 │   GPU0    │ │   GPU0    │ │   GPU0    │ │   GPU0    │
 └───────────┘ └───────────┘ └───────────┘ └───────────┘
       └────────── 1 个 torchrun 进程组(world_size=4) ──────────┘
```

> 若每台机器要用 8 张卡，常见写法是 `--nodes=4 --ntasks-per-node=8 --gpus-per-task=1`（world_size=32），或保持 `--ntasks=4` 而让 torchrun 的 `--nproc_per_node 8` 在节点内拉 8 个进程——两种风格对应不同的「谁负责拉进程」哲学（见第 5、6 节）。

---

## 2. Slurm 架构：三个守护进程 + 三个命令

Slurm 集群里跑着两类守护进程，你通过三个命令与它们打交道：

```
        ┌─────────────────── 控制节点 ───────────────────┐
        │   slurmctld  (中央调度器，决定作业去哪)         │
        │   slurmdbd   (记账数据库，可选)                 │
        └───────┬──────────────────────────────┬─────────┘
                │ 下发                          │ 下发
        ┌───────▼───────┐              ┌────────▼──────┐
        │ node0: slurmd │   ......     │ nodeN: slurmd │  ← 计算节点上的执行代理
        └───────────────┘              └───────────────┘

  你的命令：
   sbatch  脚本.slurm   → 提交一个批处理作业(异步，进队列)
   srun    命令         → 在已分配资源上启动任务(可交互/可嵌在 sbatch 内)
   squeue               → 查看作业队列状态
   scancel JOBID        → 取消作业
```

- **sbatch**：提交一个脚本作业，Slurm 排队、分配资源后异步执行。脚本顶部的 `#SBATCH` 行就是资源请求。
- **srun**：在已分配的资源上真正启动进程。它会在「每个 task」上各跑一份你给的命令——这是多机能并行起来的关键。
- **squeue / scancel**：运维三连里的「看」和「杀」。

> 官方文档：`srun` → https://slurm.schedmd.com/srun.html
> Web 管理面板（北大 PKUHPC 开源）：SCOW → https://github.com/PKUHPC/SCOW

---

## 3. MUNGE：多机免密认证的基石

> **MUNGE**（MUNGE Uid 'N' Gid Emporium）是一种用于创建和验证凭证的身份验证服务。它允许进程在一组具有公共用户和组的主机中验证另一个本地或远程进程的 UID 和 GID。

**为什么需要它？** 多机训练时，`slurmctld` 要给 `node3` 上的 `slurmd` 下命令、`srun` 要在远程节点拉进程——这些跨机调用必须确认「对面真的是我信任的那个用户」。如果靠 SSH 密码/密钥，调度延迟和运维成本都不可接受。MUNGE 用一把**全集群共享的密钥**（`/etc/munge/munge.key`）来签发和校验凭证：

```
  node0 进程                              node3 上的 slurmd
 ┌──────────┐  ① munge 用共享密钥签出凭证  ┌──────────────┐
 │  我是 UID │ ──────凭证(含UID/GID/时间戳)──▶│ unmunge 用同把 │
 │  1001     │                              │ 密钥校验签名   │
 └──────────┘                              │ → 确认 UID1001 │
   每台机器都有同一份 /etc/munge/munge.key  └──────────────┘
```

**常见坑**：所有节点的 `munge.key` 必须**完全一致**且权限为 `0400`、属主 `munge`；集群各节点**时钟必须同步**（凭证带时间戳，时钟漂移会导致 `Invalid credential`），所以生产环境普遍配 NTP/chrony。

---

## 4. sbatch 脚本：`#SBATCH` 指令逐行解释

下面是原文的 PyTorch 多机多卡 sbatch 脚本（参考 [pytorch/examples](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/sbatch_run.sh)）。先看头部资源声明：

```bash
#!/bin/bash

#SBATCH --job-name=multinode-example   # 作业名，squeue 里显示
#SBATCH --nodes=4                       # 申请 4 个节点
#SBATCH --ntasks=4                      # 总共 4 个任务(=4 个 srun 进程)
#SBATCH --gpus-per-task=1               # 每个任务 1 张 GPU
#SBATCH --cpus-per-task=4               # 每个任务 4 个 CPU 核(喂数据/dataloader)
```

| `#SBATCH` 指令 | 作用 | 填错的后果 |
| --- | --- | --- |
| `--job-name` | 作业可读名 | 仅影响可读性 |
| `--nodes=4` | 要几台机器 | 不够则排队等资源 |
| `--ntasks=4` | 拉几个进程（srun 副本数） | 设大于 GPU 数会争抢 |
| `--gpus-per-task=1` | 每进程几张卡 | 设错导致 OOM 或卡闲置 |
| `--cpus-per-task=4` | 每进程几个 CPU | 太少则 dataloader 成瓶颈 |

> 注意 `#SBATCH` 必须紧跟 `#!/bin/bash` 之后、任何普通命令之前，否则被当成普通注释忽略——这是新手最常踩的坑。

---

## 5. Slurm + torchrun：rendezvous 如何组队

脚本主体要解决一个核心问题：**4 台机器上的 4 个进程，怎么知道彼此的存在并组成一个 `world_size=4` 的进程组？** 答案是「选一个 head node 当集合点（rendezvous）」。原文脚本后半段：

```bash
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )  # 拿到本作业所有节点主机名
nodes_array=($nodes)
head_node=${nodes_array[0]}                                  # 第 0 个节点当"队长"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)  # 取队长 IP

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
--nnodes 4 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
/shared/examples/multinode_torchrun.py 50 10
```

**逐段为什么：**

1. `scontrol show hostnames $SLURM_JOB_NODELIST`：Slurm 把分配到的节点写在环境变量 `$SLURM_JOB_NODELIST`（形如 `node[0-3]`），这条命令把它**展开成逐行主机名**，便于脚本取第一个当 head。
2. `head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)`：只在 head 节点上跑一次 `hostname --ip-address`，拿到它的 IP 作为 rendezvous 端点。
3. **外层 `srun torchrun ...`**：`srun` 会在**每个节点各启动一份 `torchrun`**（共 4 份），每份再按 `--nproc_per_node 1` 拉 1 个训练进程。所有进程都连到 `--rdzv_endpoint $head_node_ip:29500` 集合，由 `c10d` 后端协商出各自的全局 rank。

```
        srun 在每个节点拉一份 torchrun
 node0(head)        node1          node2          node3
 torchrun           torchrun       torchrun       torchrun
   │  rank0           │ rank1         │ rank2        │ rank3
   └──────── 都连到 head_node_ip:29500 (c10d rendezvous) ────────┘
                       协商出 world_size=4 的进程组
```

| torchrun 参数 | 含义 | 本例值 |
| --- | --- | --- |
| `--nnodes` | 总节点数 | 4 |
| `--nproc_per_node` | 每节点进程数 | 1 |
| `--rdzv_id` | 本次会合的唯一 ID（`$RANDOM` 避免冲突） | 随机 |
| `--rdzv_backend` | 会合后端，`c10d` 是 PyTorch 内置、无需额外服务 | c10d |
| `--rdzv_endpoint` | 集合点 `IP:PORT` | head:29500 |

> **数值示例**：world_size = `nnodes × nproc_per_node` = 4 × 1 = 4。若改成每机 8 卡 `--nnodes 4 --nproc_per_node 8`，则 world_size=32，rank 编号 0~31。脚本结尾 `... 50 10` 是传给训练脚本 `multinode_torchrun.py` 的位置参数（如 total_epochs=50、save_every=10）。

---

## 6. DeepSpeed 启动的四种姿势（原文真命令）

DeepSpeed 自带 launcher（`deepspeed` 命令），与 torchrun 是两条并行路线。原文给了从简到繁的四级用法。

### 6.1 单机多卡

```bash
deepspeed --include localhost:0,1,2,3 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```

- `--include localhost:0,1,2,3`：本机用 0/1/2/3 四张卡（`localhost:GPU列表` 语法）。
- `-p 2`：流水线并行度（pipeline parallel）= 2（取决于 `train.py` 自定义参数）。
- 单机时 DeepSpeed 直接 fork 出 4 个进程，无需 rendezvous。

### 6.2 多机多卡（手动在每台机器各启一份）

```bash
# 节点0(rank0)
python -m torch.distributed.run --nproc_per_node=2 --nnode=2 --node_rank=0 --master_addr=10.99.2.xx \
--master_port=9901 train.py --deepspeed_config=ds_config.json -p 2 --steps=200

# 节点1(rank1)
python -m torch.distributed.run --nproc_per_node=2 --nnode=2 --node_rank=1 --master_addr=10.99.2.xx \
--master_port=9901 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```

**为什么要敲两遍？** 这里用的是 `torch.distributed.run`（等价于 torchrun）而非 Slurm，所以**没有 srun 帮你分发**——必须手动登录每台机器各跑一次，且 `--node_rank` 逐机不同（0、1），`--master_addr/--master_port` 必须全机一致指向同一个 master。这正凸显了第 5 节 Slurm `srun` 自动分发的价值。

```
 节点0: node_rank=0 ┐
                    ├─ 都指向 master_addr=10.99.2.xx:9901 → world_size=2×2=4
 节点1: node_rank=1 ┘
```

### 6.3 单机多卡 + Docker

```bash
sudo docker run -it --rm --gpus all \
--network=host \
--shm-size 4G \
-v /data/hpc/home/guodong.li/:/workspaces \
-v /data/hpc/home/guodong.li/.cache/:/root/.cache/ \
-w /workspaces/DeepSpeedExamples-20230430/training/pipeline_parallelism \
deepspeed/deepspeed:v072_torch112_cu117 /bin/bash

deepspeed --include localhost:4,5,6,7 --master_port 29001 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```

| docker 参数 | 为什么必须有 |
| --- | --- |
| `--gpus all` | 容器才能看到宿主 GPU |
| `--network=host` | 多机/NCCL 直连宿主网络，避免 NAT 端口问题 |
| `--shm-size 4G` | DataLoader/NCCL 用共享内存，默认 64MB 会报 `bus error` |
| `-v ...:/workspaces` | 挂代码与数据进容器 |
| `-v ...:/root/.cache/` | 复用模型/HF 缓存，避免重复下载 |
| `-w ...` | 进容器后的工作目录 |

> 注意此处 `--include localhost:4,5,6,7` 用的是后 4 张卡，并显式指定 `--master_port 29001` 避免与其他作业端口冲突。

### 6.4 单机多卡 + Singularity（HPC 友好的容器）

HPC 集群通常**不允许 root 跑 Docker**，于是把 Docker 镜像转成 Singularity 镜像（`.sif`，普通用户即可运行）：

```bash
# 1) 推到私有 Harbor 仓库
docker tag deepspeed/deepspeed:v072_torch112_cu117 harbor.aip.io/base/deepspeed:torch112_cu117
sudo docker push harbor.aip.io/base/deepspeed:torch112_cu117

# 2) 从私有仓库构建 .sif（NOHTTPS=1 跳过证书校验)
SINGULARITY_NOHTTPS=1 singularity build deepspeed.sif docker://harbor.aip.io/base/deepspeed:torch112_cu117

# 3) 运行(--nv 暴露 NVIDIA 驱动，-B 绑定目录)
singularity run --nv \
--pwd /workspaces/DeepSpeedExamples-20230430/training/pipeline_parallelism \
-B /data/hpc/home/guodong.li/:/workspaces:rw \
deepspeed.sif

# 4) 容器内启动训练(注意前置 NCCL 环境变量)
export NCCL_IB_DISABLE=1 && export NCCL_SOCKET_IFNAME=bond0 && export CC=/opt/hpcx/ompi/bin/mpicc && \
deepspeed --include localhost:4,5,6,7 --master_port 29001 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
```

**那三个 export 是这套流程的精华，逐个解释：**

| 环境变量 | 含义 | 为什么这里设 |
| --- | --- | --- |
| `NCCL_IB_DISABLE=1` | 禁用 InfiniBand，强制走以太网 | 容器内看不到 IB 设备或 IB 未配好时，避免 NCCL 启动卡死 |
| `NCCL_SOCKET_IFNAME=bond0` | 指定 NCCL 用 `bond0` 网卡通信 | 多网卡机器若选错网卡（如选了 docker0/lo）会连不通 |
| `CC=/opt/hpcx/ompi/bin/mpicc` | 指定 MPI 的 C 编译器 | 某些算子/扩展需即时编译，指向 HPC-X 自带 mpicc |

### 6.5 单机多卡 + Singularity + Slurm

把上面的 singularity 命令封进一个 `.slurm` 脚本，交给 Slurm 调度：

```bash
sbatch pp-standalone-singularity.slurm   # 提交

squeue                                    # 查看队列
scancel -v xx                             # 取消作业 xx(-v 输出详细信息)
```

### 6.6 多机多卡 + Singularity + Slurm

```bash
sbatch pp-multinode-singularity.slurm
```

多机场景的关键差异：脚本里需要用 `srun --mpi=pmi2`（见下一节）让 Slurm 跨节点把 MPI 进程正确拉起。原文要点：**`--mpi`：指定 mpi 类型为 pmi2**。

---

## 7. MPI 与 PMI2：Slurm 怎么帮 MPI 拉进程

DeepSpeed 多机后端可走 MPI；而 MPI 的进程在集群里到底怎么被拉起、怎么互相发现，靠的是 **PMI（Process Management Interface）**。Slurm 内置 PMI2 插件，能直接充当 MPI 的进程管理器，省去单独的 `mpirun`。

先查当前 Slurm 支持哪些 MPI 插件：

```bash
srun --mpi=list
```

然后在多机脚本里用对应类型（原文用 `pmi2`）：

```bash
srun --mpi=pmi2 ...   # 让 Slurm 用 PMI2 接口托管 MPI 进程的拉起与寻址
```

```
   没有 PMI 时：你要自己 mpirun -np N -hostfile ...  手动列机器
   有 Slurm+PMI2：srun --mpi=pmi2  →  Slurm 已知道节点列表，自动分发+编号
        ┌── slurmd@node0 ──┐  ┌── slurmd@node1 ──┐
        │  MPI rank0/1     │  │  MPI rank2/3     │   ← PMI2 负责互相发现
        └──────────────────┘  └──────────────────┘
```

> 直觉对照：torchrun 用 `c10d` rendezvous 组队（第 5 节）；MPI 程序用 `PMI2` 由 Slurm 托管组队。二者解决的是同一类「进程怎么找到彼此」的问题，只是协议不同。

---

## 实操命令速查（原文真料汇总）

| 目的 | 命令 |
| --- | --- |
| 提交批作业 | `sbatch xxx.slurm` |
| 查看队列 | `squeue` |
| 取消作业 | `scancel -v <JOBID>` |
| 列 MPI 插件 | `srun --mpi=list` |
| 多机 MPI 拉起 | `srun --mpi=pmi2 ...` |
| 展开节点列表 | `scontrol show hostnames $SLURM_JOB_NODELIST` |
| DeepSpeed 单机 | `deepspeed --include localhost:0,1,2,3 train.py --deepspeed_config=ds_config.json -p 2 --steps=200` |
| 构建 .sif | `SINGULARITY_NOHTTPS=1 singularity build deepspeed.sif docker://...` |
| 运行 .sif | `singularity run --nv --pwd <dir> -B <src>:<dst>:rw deepspeed.sif` |

---

## 常见问题 / 坑

| 现象 | 根因 | 解法 |
| --- | --- | --- |
| `Invalid credential` / 节点互信失败 | MUNGE key 不一致或时钟漂移 | 同步 `munge.key`（0400/munge属主）+ 配 NTP |
| `#SBATCH` 没生效 | 放在普通命令之后被当注释 | 必须紧跟 `#!/bin/bash`、在任何命令前 |
| 多机 NCCL 启动卡住不动 | 走了 IB 但 IB 未配好 | `export NCCL_IB_DISABLE=1` 退回以太网 |
| NCCL 连不通 / 选错网卡 | 多网卡时默认选了 lo/docker0 | `export NCCL_SOCKET_IFNAME=bond0` 指定网卡 |
| 容器内 `bus error` / DataLoader 崩 | 共享内存太小（默认 64MB） | docker 加 `--shm-size 4G` |
| 多机 torch.distributed.run 组不起来 | 各机 `--node_rank` 写重了或 master 不一致 | rank 逐机递增，`--master_addr/port` 全机一致 |
| `srun --mpi=pmi2` 报不支持 | Slurm 未编译对应 MPI 插件 | 先 `srun --mpi=list` 看支持哪些再选 |
| 端口冲突 `Address already in use` | 多作业共用同一 `master_port` | 显式指定不同 `--master_port`（如 29001） |
| HPC 上 Docker 跑不了 | 集群禁止普通用户 root 运行 docker | 转 Singularity（`.sif`，免 root） |

---

## 🔗 跳转链接

**枢纽**
- [[00-知识地图]]
- [[llm-train/README]]
- [[llm-train/pytorch/distribution/README]]
- [[llm-train/megatron/README]]
- [[llm-train/megatron-deepspeed/README]]

**框架**
- [[ai-framework/megatron-lm/README]]
- [[ai-framework/deepspeed/README]]
- [[ai-framework/pytorch/README]]
- [[ai-framework/huggingface-peft/README]]

**PEFT / 微调**
- [[llm-train/peft/PEFT-API]]
- [[llm-train/peft/Prompt-Tuning]]
- [[llm-train/peft/Prefix-Tuning]]

**网络 / 通信**
- [[ai-infra/网络/集合通信原语]]
- [[ai-infra/网络/NCCL]]

**对齐 / 算法 / 压缩 / 推理**
- [[llm-alignment/RLHF]]
- [[llm-algo/transformer/模型架构]]
- [[llm-compression/quantization/量化基础]]
- [[B07:llm-inference/大模型推理张量并行]]
