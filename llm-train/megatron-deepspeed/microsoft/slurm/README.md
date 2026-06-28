# Slurm 上跑 Megatron-DeepSpeed 多机多卡训练

> 用 Slurm 作业调度器 + `srun`/`sbatch` + PMIx + Singularity 容器，把 Megatron-DeepSpeed 的多机多卡 LLaMA 预训练「拉起来」的完整机制与脚本拆解。📍 导航：[[00-知识地图]]
> 🔗 相关：[[H800多机多卡训练坑点]] [[ai-infra/网络/集合通信原语]] [[llm-train/pytorch/distribution/README]] [[llm-train/README]]

## 阅读地图

| 节 | 内容 | 你会得到什么 |
|---|---|---|
| 0 | 一句话锚点 | Slurm 在多机训练里到底干啥 |
| 1 | 地基：为什么需要作业调度器 | 从「手动 ssh 拉起」到「一条命令调度全集群」 |
| 2 | Slurm 名词速通 | partition / node / gres / srun / sbatch / pmix |
| 3 | 拆解那条测试命令 | `srun -p ... --mpi=pmix_v3 -N 2 --gres=gpu:8 env` |
| 4 | 整条启动链路 | Slurm → srun → Singularity → torchrun → pretrain_gpt.py |
| 5 | rendezvous：master 地址/端口怎么来 | `scontrol`/`hostname --ip-address` 那几行 shell |
| 6 | sbatch 脚本逐行（30B/65B 实例） | `#SBATCH` 头 + NCCL 环境变量 + 3D 并行参数 |
| 7 | srun 与 torchrun 谁拉谁 | 两种进程模型的边界与坑 |
| - | 关键命令/数值示例 | 显存账 + 并行度手算 |
| - | 评价/对照/局限（表格） | Slurm vs 裸 torchrun vs k8s |

## 0. 一句话锚点

**Slurm 是 HPC 集群的「操作系统调度层」：你只描述「我要 2 台机、每台 8 张 H800」，它负责分配节点、设好 `SLURM_*` 环境变量、在每个节点上同时拉起进程；`--mpi=pmix_v3` 让这些跨节点进程能互相发现并建立通信。** 真正的训练逻辑（3D 并行、NCCL all-reduce）仍由容器里的 `torchrun + pretrain_gpt.py` 负责——**Slurm 只解决「在哪些机器上、同时、把进程跑起来」这件事**，不碰训练本身。

仓库里这个 `slurm/` 目录就是把前面单机版的 `pretrain_llama*.sh`「升级成多机版」的那层壳。那条单行命令：

```bash
srun -p h800-ib-1 --mpi=pmix_v3 -N 2 --gres=gpu:8 env
```

是**冒烟测试**——在 `h800-ib-1` 分区申请 2 台机、每台 8 卡，每个进程跑 `env` 打印环境变量，用来确认「调度器能把 2 机进程同时拉起、PMIx 正常、GPU 分到位」，**还没开始真训练**。

## 1. 地基：为什么需要 Slurm（作业调度器）

### 1.1 没有调度器时多机训练有多痛

裸跑多机 torchrun，你得手动：①登录每台机 → ②各自 `export MASTER_ADDR/NODE_RANK/...` → ③在每台机敲一遍 `torchrun --node_rank=k ...` → ④保证所有机同时启动、参数一致。**N 台机就要 N 个终端、N 次手敲**，还得抢机器（别人也在用同一批 GPU）。

调度器把这套自动化：

```
   裸手动多机                          Slurm 调度
   ┌──────┐ ssh ┌──────┐             你 ── sbatch job.slurm ──► Slurm 控制器
   │node0 │     │node1 │                              │
   │手敲  │     │手敲  │                              ├─► 找到空闲的 2×8卡 节点
   └──────┘     └──────┘                              ├─► 在 node0/node1 同时 srun
   N 台 = N 次手动, 易错, 要抢机          ├─► 注入 SLURM_NODELIST / SLURM_PROCID...
                                          └─► 跑完回收资源, 排队下一个作业
```

### 1.2 Slurm 解决的三个核心问题

1. **资源分配与排队**：集群被多人共享，Slurm 维护队列，按 partition/优先级把空闲节点+GPU 分给你的作业（避免两人抢同一张卡）。
2. **多节点同步拉起**：`srun` 一条命令在所有被分配节点上**同时** spawn 进程，并注入 `SLURM_*` 环境变量（节点列表、本进程全局编号等），让进程能彼此发现。
3. **进程间通信引导（PMIx）**：`--mpi=pmix_v3` 提供一个「会合/交换信息」的运行时，让分布在不同机器上的进程拿到彼此地址，完成 MPI/集合通信初始化。

> 一句话：**Slurm = 资源调度 + 多机同时启动 + 通信引导**。它是「拉起层」，不替代 NCCL（数据怎么通信）也不替代 Megatron（怎么切模型）。

## 2. Slurm 名词速通（看懂脚本必备）

| 名词 | 含义 | 在脚本里 |
|---|---|---|
| **partition（分区）** | 一组性质相同的节点（如全是 H800+IB） | `-p h800-ib-1` / `#SBATCH --partition=h800-ib-2` |
| **node（节点）** | 一台物理机（这里=8×H800） | `-N 2`（要 2 台）/ `#SBATCH -N 4` |
| **gres（通用资源）** | Generic RESource，最常用的是 GPU | `--gres=gpu:8`（每节点 8 卡） |
| **task（任务）** | srun 拉起的一个进程 | `--ntasks` / `-n` 控制；默认每节点 1 |
| **cpus（核数）** | 每任务分配的 CPU 核 | `-c 80`（给 dataloader/编译用） |
| **`srun`** | 在分配的节点上**立即并行启动**进程 | 真正拉进程的命令 |
| **`sbatch`** | 提交一个**批处理脚本**到队列，排队后执行 | 提交 `.slurm` 文件 |
| **`scontrol`** | 查询/控制作业与节点信息 | `scontrol show hostnames` 拿节点名 |
| **`--mpi=pmix_v3`** | 用 PMIx v3 作为 MPI 进程管理接口 | 跨节点进程会合的「引导器」 |
| **`SLURM_JOB_NODELIST`** | 本作业分到的节点列表（压缩格式） | shell 里解析出 master |
| **`SLURM_JOBID`** | 作业唯一 id | 用来生成 rdzv-id、端口 |

> `srun` vs `sbatch`：`sbatch job.slurm` 把脚本丢进队列、立刻返回（异步、可排队、可重定向日志）；脚本体内再用 `srun ...` 真正拉起多节点进程。**交互调试用 `srun`，正式作业用 `sbatch`**。

## 3. 拆解那条测试命令

```bash
srun -p h800-ib-1 --mpi=pmix_v3 -N 2 --gres=gpu:8 env
```

逐段读：

```
 srun            ── 在 Slurm 分配的节点上"立刻并行"启动进程
 -p h800-ib-1    ── partition: 在名为 h800-ib-1 的分区里挑节点(全是带IB的H800)
 --mpi=pmix_v3   ── 用 PMIx v3 做进程管理/会合(让跨节点进程能互相发现)
 -N 2            ── nodes: 申请 2 台机器
 --gres=gpu:8    ── 每台机器要 8 张 GPU(共 2×8=16 卡)
 env             ── 每个被拉起的进程执行的命令:打印自己的环境变量
```

**它在干嘛？** 这是一次**最小冒烟测试**：不跑训练，只让每个进程打印 `env`。你在输出里要确认：
- `SLURM_NODELIST` 是否真给了 2 台不同机器；
- `SLURM_PROCID` / `SLURM_NODEID` 是否正确编号（说明 PMIx 把多进程拉对了）；
- `CUDA_VISIBLE_DEVICES` / GPU 相关变量是否每节点 8 卡到位；
- IB 相关环境（如有）是否存在。

> 把它当作「多机训练前的体检」：**连 `env` 都拉不起 2 机，就别谈 16 卡训 30B**。先用它把「调度+PMIx+GPU 分配」这层确认 OK，再上真作业。这正是 [[H800多机多卡训练坑点]] 第 8 节「二分定位」的第一刀。

## 4. 整条启动链路：从 Slurm 到训练 step

把 `llama-multinode-ib.sh` / `*.slurm` 这类脚本展开，真实链路是**五层套娃**：

```
┌────────────────────────────────────────────────────────────────┐
│ ① sbatch job.slurm        提交作业, Slurm 排队 + 分配 N 台×8卡   │
│        │                                                        │
│ ② srun --mpi=pmix_v3 ...  在每个被分配节点上"同时"启动 1 个进程   │
│        │                  (PMIx 负责跨节点会合, 注入 SLURM_*)     │
│ ③ singularity run --nv    每节点进入同一个 .sif 容器(环境一致)    │
│        │                  -B 把宿主 /workspace 挂进容器          │
│ ④ torchrun --nnodes N \   容器内, torchrun 在本节点再 spawn      │
│        --nproc_per_node 8 8 个训练进程(每进程绑 1 张 H800)        │
│        --rdzv_endpoint=master  ← 各节点 torchrun 在此会合        │
│        │                                                        │
│ ⑤ pretrain_gpt.py ...     真正的训练: Megatron 切 TP/PP,         │
│           --tensor-model-parallel-size / --pipeline-... \       │
│           DeepSpeed 管 ZeRO, NCCL 跑 all-reduce                  │
└────────────────────────────────────────────────────────────────┘
        ↑ Slurm 管"在哪些机上拉进程"; torchrun 管"本机 8 进程编址";
          Megatron+DeepSpeed 管"怎么切模型"; NCCL 管"怎么通信"
```

**关键分工（务必记住）**：

- **Slurm `srun`**：决定「**哪些节点**、每节点启动**几个 srun 任务**」。这些脚本里 `srun` 默认每节点 1 个任务（不带 `--ntasks-per-node`），即每节点只起一个「torchrun 启动器」。
- **`torchrun --nproc_per_node 8`**：在**本节点内部**再 fork 出 8 个真正的训练进程，每进程绑一张卡。所以最终训练进程数 = `N × 8`。
- 两者**不要重复 spawn**：若让 `srun` 也按 GPU 数起任务，又让 torchrun 起 8 个，进程数会翻倍、world_size 对不上（见第 7 节坑点）。

## 5. rendezvous：master 地址与端口是怎么算出来的

脚本里这三行是多机训练的「会合地址生成器」，看懂它就懂了 torchrun 怎么知道去哪儿握手：

```bash
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_HOST" hostname --ip-address|awk '{print $1}')
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID|tail -c 4))
```

逐行解释**为什么这么写**：

1. `scontrol show hostnames "$SLURM_JOB_NODELIST"` 把 Slurm 的压缩节点列表（如 `node[01-04]`）**展开成一行一个主机名**，`head -1` 取第一个当 **master**。→ 所有节点必须选**同一个** master，取「列表第一个」是天然一致的约定。
2. `srun --nodes=1 --ntasks=1 -w "$MASTER_HOST" hostname --ip-address` 专门去**那台 master 机**上执行 `hostname --ip-address`，拿到它的 **IP**。→ 用 IP 而非主机名，规避 DNS 解析不稳（[[H800多机多卡训练坑点]] 5.2 的经验）。
3. `MASTER_PORT=$(expr 10000 + $(... tail -c 4))` 取 `SLURM_JOBID` 末 4 位 + 10000 当端口。→ **不同作业自动错开端口**，避免同机多作业撞端口导致 rendezvous 失败。

```
   node[01-04]  ─scontrol─►  node01            ← MASTER_HOST(取第一个)
                              node02
                              node03            每个节点的 torchrun 都
                              node04            连到 node01:PORT 会合
                                │
   torchrun(node01) ─┐         │ rdzv_endpoint = node01_ip : PORT
   torchrun(node02) ─┼─► 连 master:PORT ─► 凑齐 nnodes 个 →
   torchrun(node03) ─┤        协商 world_size, 给每进程分全局 RANK
   torchrun(node04) ─┘        然后才 init_process_group(nccl)
```

> 注意脚本同时算了 `MASTER_HOST`（主机名）和 `MASTER_ADDR`（IP）；`ENDPOINT_URL` 实际用的是 `MASTER_HOST:MASTER_PORT` 传给 `--rdzv_endpoint`。`MASTER_ADDR`（IP）多是为了 `echo` 排查/或某些场景显式喂 IP——**究竟传主机名还是 IP，以脚本实际拼接为准**，调试时打印出来核对最稳。

## 6. sbatch 脚本逐行（30B / 65B 实例拆解）

以仓库 `megatron-deepspeed-multinode-ib-part2-30b-fp16.slurm` 为骨架，分三块看：

### 6.1 `#SBATCH` 头：声明资源

```bash
#SBATCH --job-name=megatron-multinode-ib-30b-2   # 作业名(队列里好认)
#SBATCH --partition=h800-ib-2                     # 跑在带 IB 的 H800 分区
#SBATCH --output=log/%j.out                       # stdout 落到 log/<jobid>.out
#SBATCH --error=log/%j.out                        # stderr 也到同一文件(%j=jobid)
#SBATCH -N 4                                       # 4 台机器
#SBATCH -c 80                                      # 每任务 80 个 CPU 核
#SBATCH --gres=gpu:8                               # 每机 8 张 GPU(共 32 卡)
```

> `%j` 是 Slurm 占位符，运行时替换成作业 id，所以日志按作业自动分文件，便于事后查某次跑的日志。

### 6.2 NCCL 环境变量：决定通信走哪条路

```bash
export NCCL_DEBUG=info                  # 打开通信日志, 看走的是 IB 还是 Socket(必开)
export NCCL_IB_DISABLE=0                # 0=启用 InfiniBand(走 RDMA, 快)
export NCCL_IB_HCA=mlx5_0              # 指定用哪块 IB 网卡设备
export NCCL_PXN_DISABLE=1              # 关闭 PXN(跨网卡中转), 拓扑相关调优
export NCCL_IB_TIMEOUT=22             # IB 超时(放大以抗抖动)
export NCCL_IB_RETRY_CNT=13           # IB 重试次数
export NCCL_IB_PCI_RELAXED_ORDERING=1  # 放松 PCIe 排序, 某些平台提带宽
```

对照 65B 脚本里被注释切换的写法（**这是一个重要对照**）：

```bash
# 65B 脚本里改成走以太网而非 IB:
export NCCL_IB_DISABLE=1              # 1=禁用 IB
export NCCL_SOCKET_IFNAME=bond0       # 强制走 bond0 这张以太网卡
```

> 含义：**IB 路径**（`NCCL_IB_DISABLE=0`）是默认想要的高速 RDMA；当 IB 不稳/未配好时，可临时切到 **Socket 路径**（`=1` + 指定 `NCCL_SOCKET_IFNAME`）保证「能跑」，但**带宽和延迟会差一截**。各变量的具体取值/默认值以 NCCL 官方文档为准，调优原则见 [[H800多机多卡训练坑点]] 6.2：**每次只改一个变量并对比吞吐**。

### 6.3 srun + Singularity + torchrun + pretrain_gpt.py（核心一行）

```bash
srun --mpi=pmix_v3 singularity run --nv \
  --pwd /workspace/code/Megatron-DeepSpeed-llama-20230815 \
  -B /data/hpc/home/guodong.li/workspace:/workspace:rw \
  megatron-deepspeed-v1.sif \
  torchrun --nnodes 4 --nproc_per_node 8 \
    --rdzv_id=$SLURM_JOBID --rdzv_backend=c10d --rdzv_endpoint=$ENDPOINT_URL \
    pretrain_gpt.py \
    --tensor-model-parallel-size 4 --pipeline-model-parallel-size 8 \
    --num-layers 60 --hidden-size 6656 --ffn-hidden-size 17920 \
    --num-attention-heads 52 --micro-batch-size 2 --global-batch-size 8 \
    ...（数据/优化器/RoPE/SwiGLU/RMSNorm/DeepSpeed 等参数）
```

逐段：
- `--nv`：Singularity 把宿主 NVIDIA 驱动/设备透传进容器（不然容器内看不到 GPU）。
- `-B host:container:rw`：bind mount，把宿主 `/workspace`（代码、数据、tokenizer、checkpoint）挂进容器，**保证所有节点看到同一份代码/数据**（呼应坑点「逐字节一致」）。
- `--pwd ...`：容器内工作目录设到代码目录。
- `*.sif`：Singularity 镜像，**所有节点用同一个镜像**=环境天然一致。
- `torchrun --rdzv_*`：弹性会合三件套，`rdzv_id` 用 `SLURM_JOBID`（全作业唯一且各节点一致），`rdzv_endpoint` 就是第 5 节算出的 master。

## 关键命令/数值示例

### A. 三套实例的并行度与卡数（手算 DP）

`world_size = TP × PP × DP`，DP 由总卡数反推：

| 脚本 | 模型 | 节点×卡 = 总卡 | TP | PP | **DP = 卡 /(TP·PP)** | 校验 |
|---|---|---|---|---|---|---|
| `llama-multinode-ib.sh` | 7B | 2×8 = 16 | 2 | 2 | 16/(2·2)=**4** | TP=2≤8 走机内 ✓ |
| `...part2-30b-fp16.slurm` | 30B | 4×8 = 32 | 4 | 8 | 32/(4·8)=**1** | TP=4≤8 走机内 ✓ |
| `...part2-65b-fp16.slurm` | 65B | 8×8 = 64 | **16** | 4 | 64/(16·4)=**1** | ⚠ TP=16>8 跨机! |

> **65B 这个 `TP=16` 是一个值得警惕的配置**：单机只有 8 卡，TP=16 意味着张量并行组**跨了 2 台机**，而 TP 是最重的通信（每步 all-reduce）。这正解释了为什么 65B 脚本里把 `NCCL_IB_DISABLE` 切来切去——跨机 TP 对网络极敏感。一般「黄金法则」是 `TP ≤ 单机卡数`（见 [[H800多机多卡训练坑点]] 4.1），此处属于「显存放不下被迫跨机 TP」的折中，**以实际可跑/吞吐为准**。

### B. 整除约束自检（启动前必算）

- `world_size % (TP×PP) == 0`：30B 即 `32 % 32 = 0` ✓。
- `num_layers % PP == 0`：30B 即 `60 % 8 = 4 ≠ 0` ⚠ —— **理论上 60 不被 8 整除**，实际 Megatron/DeepSpeed 的 PP 切分需要层数可被 stage 数整除，这种配置要么有特殊处理要么需调整，**以实际运行报错为准**（这正是「启动即报并行度错」的高频来源）。
- `hidden_size % TP == 0`：30B `6656 % 4 = 0` ✓；`num_heads % TP`：`52 % 4 = 0` ✓。

### C. 显存账：为什么 30B 要 32 卡

混合精度 + Adam，参数态约 16 B/参数（fp16 参数 2 + fp16 梯度 2 + fp32 参数 4 + 一阶动量 4 + 二阶方差 4）：

$$
\text{显存}_{\text{状态}} \approx 16 \times N_{\text{params}}
$$

30B：$16 \times 30\text{e}9 \approx 480\text{ GB}$，再加激活，**远超单卡 80GB**。32 卡总显存 $32\times80=2560\text{ GB}$，靠 TP/PP 把状态切到各卡 + 激活重计算（`--deepspeed-activation-checkpointing`，算力换显存）才放得下。这就是「必须多机」的根因。

## 评价/对照/局限（表格）

| 维度 | Slurm + srun | 裸 torchrun（手动多机） | k8s + Operator |
|---|---|---|---|
| 资源排队/共享 | ★★★ 队列+优先级，HPC 标配 | ✗ 自己抢机 | ★★ 需额外调度组件 |
| 多机同时拉起 | ★★★ 一条 `srun`/`sbatch` | ★ 每机手敲 | ★★ 由 Operator 拉 |
| 进程会合引导 | PMIx（`--mpi=pmix_v3`）+ torchrun rdzv | 仅 torchrun rdzv | torchrun rdzv |
| 容器一致性 | Singularity `.sif`，HPC 友好(无 root) | 自行保证 | Docker 镜像 |
| 学习曲线 | 中（`#SBATCH`/`scontrol` 一套黑话） | 低但繁琐 | 高 |
| 弹性扩缩容 | 一般（作业粒度） | 弱 | 强 |

**局限/注意**：
- Slurm 只到「拉起进程」为止，**通信慢/hang/OOM 仍要回到 NCCL/Megatron 层排查**（→ [[H800多机多卡训练坑点]]）。
- `srun` 与 `torchrun` 的进程层级要分清，**别让两者都按 GPU 数 spawn** 导致进程翻倍。
- 脚本里的路径、镜像名、partition 名都是**该集群专属**，迁到别的集群必须改。
- 命令默认值/版本号以集群 `module`、镜像内 `nccl`/`pytorch` 实际版本为准，**不要照搬**。

## 7. srun 与 torchrun：谁拉谁（最易混的边界）

```
   ❌ 错误: srun 按 GPU 数起任务 + torchrun 又起 8 个 → 进程翻倍
   srun --ntasks-per-node=8 ... torchrun --nproc_per_node 8 ...
        └ 8 个 srun 任务            └ 每个又 spawn 8 = 8×8=64/节点 ✗✗

   ✅ 本仓库做法: srun 每节点 1 个任务(启动器) + torchrun 起 8 个
   srun (默认每节点1任务) ... torchrun --nnodes N --nproc_per_node 8
        └ N 个 torchrun 启动器     └ 每个 spawn 8 = 总 N×8 训练进程 ✓
```

> 记忆法：**srun 决定「几台机、每台几个启动器」，torchrun 决定「每台启动器再 fork 几个绑卡进程」**。本仓库选「srun 1 启动器/节点 + torchrun 8 进程/节点」，编址由 torchrun 的 rdzv 统一管，干净不打架。另一种纯 PMIx 模式（srun 直接起 N×8 任务、不用 torchrun）也可行，但要让 Megatron 从 `SLURM_PROCID` 读 rank，**两套别混用**。

## 🔗 跳转链接

- [[00-知识地图]]
- [[H800多机多卡训练坑点]] — 同集群多机训练的环境/互联/通信/稳定性踩坑总集
- [[ai-infra/网络/集合通信原语]] — all-reduce / all-gather，TP/DP 通信本质
- [[llm-train/pytorch/distribution/README]] — PyTorch 分布式与 torchrun rendezvous
- [[llm-train/README]] — 大模型训练总览（并行策略全景）
- [[ai-framework/deepspeed/README]] — DeepSpeed ZeRO 与 3D 并行
- [[ai-framework/megatron-lm/README]] — Megatron-LM 张量/流水线并行
- [[ai-infra/网络/集合通信原语]] — NCCL 通信原语与传输路径
