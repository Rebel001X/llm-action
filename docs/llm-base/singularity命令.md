# Singularity / Apptainer 命令详解（HPC 容器）

> 一句话定位：Singularity（现已更名 **Apptainer**）是为 **HPC / 多机大模型训练集群** 设计的容器运行时——**无 root 守护进程、镜像即单文件 `.sif`、原生支持 GPU/MPI/InfiniBand**，是在 Slurm 集群上跑 LLM 训练的"标准发行形式"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/网络/集合通信原语]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]

## 阅读地图

| 节 | 你会学到 | 关键词 |
|----|---------|--------|
| 0 | 一句话锚点：为什么 HPC 不用 Docker | 无 daemon / 单文件镜像 |
| 1 | 地基：容器 = namespace + 文件系统打包 | rootfs / OCI / SIF |
| 2 | Singularity vs Docker 的本质差异 | 权限模型 / 用户态 |
| 3 | SIF 镜像格式：一个文件装下整个根文件系统 | squashFS / 只读 |
| 4 | 命令全景：pull / build / run / exec / shell | 子命令地图 |
| 5 | 绑定挂载 `--bind`：把宿主目录塞进容器 | bind mount |
| 6 | `--nv` GPU 透传：如何把 8 张卡喂进容器 | NVIDIA driver |
| 7 | 多机：Singularity + MPI + Slurm + IB | srun / NCCL |
| 8 | 环境变量与 `--cleanenv` | 环境隔离 |
| 9 | 定义文件 `.def`：可复现地构建镜像 | %post / %environment |
| ★ | 数值手算：镜像体积 / 多机带宽 / 启动开销 | 手算 |

## 0. 一句话锚点

在你的笔记本上跑模型，`docker run` 就够了。但到了**几百上千张卡的共享 HPC 集群**，Docker 有三个致命问题：

1. **需要 root 守护进程**（`dockerd`）——集群管理员绝不会给普通用户 root。
2. **镜像是分层叠加**，存在共享 `/var/lib/docker`，多用户互相污染。
3. **和 Slurm/MPI 作业调度不协调**——一个作业进程树里再起一个 daemon 很别扭。

Singularity 的设计哲学正好相反：

```
Docker 模型：  用户 → docker CLI → [dockerd 守护进程(root)] → 容器
                                         ↑ 集群禁止

Singularity： 用户 → singularity 命令 → 直接 fork 出容器进程
                     (容器进程 = 调用者本人的 UID，没有 daemon)
```

**核心口诀**：*容器内的你，就是容器外的你（同一个 UID）*。容器进程是你 shell 的子进程，被 Slurm 当作普通进程管理，写文件就是你的权限——这就是 HPC 选它的根本原因。

> 工具版本/默认行为可能随发行版变化，本文讲**稳定机制**；具体标志位与默认值**以官方文档为准**。

## 1. 地基：容器到底是什么

容器不是虚拟机。它没有自己的内核，**共享宿主内核**，只是用 Linux 内核的两组能力把进程"圈"起来：

```
        虚拟机 (VM)                     容器 (Container)
   ┌──────────────────┐         ┌──────────────────┐
   │   App  App  App   │         │  App  App  App    │
   │  ┌────────────┐   │         │  (各自的 rootfs)  │
   │  │ Guest 内核 │   │         ├──────────────────┤
   │  └────────────┘   │         │  共享宿主内核     │← 关键区别
   │   Hypervisor      │         │  namespace+cgroup │
   ├──────────────────┤         ├──────────────────┤
   │     宿主内核      │         │     宿主内核      │
   └──────────────────┘         └──────────────────┘
     重(GB级)、慢启动            轻(共享内核)、秒级启动
```

两组内核能力：
- **namespace（命名空间）**：隔离"看见什么"——挂载点、PID、网络、用户。决定容器里 `ls /` 看到的是镜像里的根目录而非宿主的。
- **cgroup（控制组）**：限制"用多少"——CPU、内存。HPC 里这部分常交给 Slurm 管。

容器 = **打包好的根文件系统（rootfs：一整套 `/usr /lib /bin ...`）** + **运行时用 namespace 把进程塞进这个 rootfs**。Singularity 把这个 rootfs 压成**一个文件**。

## 2. Singularity vs Docker：本质差异

| 维度 | Docker | Singularity / Apptainer |
|------|--------|------------------------|
| 守护进程 | 需要 `dockerd`(root) | **无 daemon**，直接 exec |
| 镜像形态 | 多层 + 注册表缓存 | **单文件 `.sif`**（可 `scp`/`cp`） |
| 容器内身份 | 默认 root（UID 0） | **就是调用者 UID**（非 root） |
| 默认文件系统 | 容器隔离 | 默认**继承宿主** `$HOME`/`$PWD`/`/tmp` |
| GPU | `--gpus` + nvidia-runtime | `--nv`（自动找驱动） |
| MPI/IB | 需额外配置 | **原生**，进程模型天然契合 |
| 典型场景 | 微服务、CI、单机 | **HPC 多机训练、科研复现** |

最反直觉的一点：**Singularity 容器默认把你的宿主 `$HOME` 和当前目录挂进去**。所以容器里直接能看到你的数据集、代码、checkpoint——这对"跑训练"极其顺手，但也意味着隔离性比 Docker 弱（这是有意为之）。

```
docker run:                       singularity run:
┌─────────────┐                  ┌─────────────┐
│ 容器内 /home│ (空,隔离)        │ 容器内 /home│ = 宿主 /home (自动绑)
│ 容器内 /data│ (空)            │ 容器内 $PWD │ = 你启动时的目录
└─────────────┘                  └─────────────┘
要 -v 显式挂载                    数据"开箱即用"
```

## 3. SIF 镜像格式：一个文件装下整个根文件系统

`.sif`（Singularity Image Format）把整个 rootfs 用 **squashFS（只读压缩文件系统）** 打包进单个文件：

```
   my_train.sif  (单个文件，比如 6 GB)
   ┌────────────────────────────────────┐
   │ [header] 元数据/签名                │
   │ [definition] 构建用的 .def 文本     │
   │ [squashFS] ← 整个 / 根文件系统      │
   │    /bin /lib /usr/lib/python3 ...   │
   │    /opt/conda/envs/...              │
   │    CUDA运行时库 / PyTorch / Megatron │
   │ [可选: overlay 可写层]              │
   └────────────────────────────────────┘
       ↑ 只读、内容寻址、可加密签名
```

为什么这个设计对 LLM 集群好：
- **可复现**：一个文件 = 确定的软件栈（CUDA 版本、PyTorch 版本、所有依赖锁死），消除"我这能跑你那不行"。
- **可分发**：直接 `cp my_train.sif /shared/images/` 或 `scp` 到另一个集群，无需注册表。
- **只读 + 签名**：squashFS 只读，保证运行时软件栈不被偷改；可用 `singularity sign/verify` 做 PGP 校验。
- **省 inode**：squashFS 是一个文件，不像 Docker 解包成几十万个小文件压垮共享并行文件系统（Lustre/GPFS）的元数据服务。

## 4. 命令全景：五个核心子命令

```
                    singularity
        ┌──────────┬──────────┬─────────┬──────────┐
      pull        build      run       exec       shell
   下载现成镜像   构建镜像   跑默认入口  跑指定命令  进交互shell
        │           │          │          │          │
   从库/Docker拉   从.def或    %runscript  任意命令   bash提示符
                  docker://   定义的脚本
```

逐个看（机制层面）：

```bash
# (1) pull：把 Docker Hub / OCI 镜像转换成 .sif
singularity pull pytorch.sif docker://pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
#   ↑ 拉取各层 → 在本地"压平(flatten)"成单一 squashFS → 写出 .sif

# (2) build：从定义文件构建（需要 root 或 --fakeroot）
singularity build my_train.sif train.def

# (3) run：执行镜像里 %runscript 定义的默认行为
singularity run my_train.sif            # 等价于把镜像当一个"可执行文件"

# (4) exec：在容器内执行任意命令（最常用于训练）
singularity exec --nv my_train.sif python train.py --tp 8

# (5) shell：进入容器交互式 shell（调试用）
singularity shell --nv my_train.sif
```

`run` vs `exec` 的区别：`run` 跑镜像作者预设的入口脚本（`%runscript`）；`exec` 完全由你指定命令。训练脚本几乎都用 `exec`，因为你要传不同的超参。

## 5. 绑定挂载 `--bind`：把宿主目录塞进容器

容器的 rootfs 是只读的（squashFS）。你的数据集、输出目录在宿主上，需要**绑定挂载（bind mount）**把宿主路径映射进容器：

```
  宿主机                          容器内（运行时）
  /mnt/data/llama_corpus  ──bind──►  /data
  /mnt/ckpt/run42         ──bind──►  /output (可写)
  /home/you/code          ──(默认自动)──► /home/you/code

  语法： --bind 宿主路径:容器路径[:ro|rw]
```

```bash
singularity exec --nv \
  --bind /mnt/data/llama_corpus:/data:ro \
  --bind /mnt/ckpt/run42:/output:rw \
  my_train.sif \
  python pretrain.py --data /data --save /output
```

要点：
- `:ro` 只读（数据集），`:rw` 可写（checkpoint 输出，默认就是 rw）。
- 容器 rootfs 是只读的，**所有运行时产生的数据必须写到 bind 进来的可写目录**，否则写 `/`（除 `/tmp`、`$HOME` 外）会报只读错误。
- 默认自动绑定 `$HOME`、`$PWD`、`/tmp`、`/proc`、`/sys`、`/dev`——所以代码和家目录"自动可见"。

## 6. `--nv` GPU 透传：把 8 张卡喂进容器

GPU 计算需要**容器内的 CUDA 运行时库**（在镜像里）配合**宿主机的 NVIDIA 驱动**（在宿主上，容器里没有）。`--nv` 就是负责把宿主驱动的关键部分注入容器：

```
   容器镜像(.sif)          --nv 注入            宿主机
  ┌──────────────┐                        ┌──────────────┐
  │ CUDA runtime │                        │ NVIDIA 驱动   │
  │ (libcudart)  │◄── 自动绑定宿主 ───────│ libcuda.so   │
  │ PyTorch      │   驱动库 + nvidia-smi  │ /dev/nvidia0 │
  │ 应用代码     │◄── 设备节点透传 ──────│ ...nvidia7   │
  └──────────────┘                        └──────────────┘
   软件栈固定在镜像里        驱动跟随宿主(不进镜像)
```

为什么要分离：**驱动版本必须匹配宿主内核**，不能打进镜像（否则换台机器就挂）；而 CUDA 用户态库（`libcudart` 等）打进镜像，保证版本可复现。`--nv` 在运行时自动找到宿主 `libcuda.so`、设备节点 `/dev/nvidiaX`，绑进容器。

```bash
# 单机 8 卡
singularity exec --nv my_train.sif \
  torchrun --nproc_per_node=8 train.py
# 验证 GPU 可见：
singularity exec --nv my_train.sif nvidia-smi
```

> AMD GPU 对应标志是 `--rocm`。具体注入哪些库**以官方为准**。

## 7. 多机：Singularity + MPI + Slurm + InfiniBand

这是 HPC 跑 LLM 的真实姿势。Singularity 的"无 daemon、容器进程即普通进程"在这里大放异彩：

```
  Slurm 调度 (srun -N4 --ntasks-per-node=8)
        │  在 4 个节点各拉起 8 个进程(=每节点8卡)
        ▼
  node0: rank0..7   node1: rank8..15  ... node3: rank24..31
   │ 每个 rank = 一个 singularity exec 进程
   ▼
  容器内 PyTorch/NCCL  ──IB/RDMA──► 跨节点 AllReduce 梯度
   ↑ NCCL 直接用宿主 InfiniBand HCA（容器透传 /dev/infiniband）
```

```bash
# 在 Slurm 作业脚本里(每个 task 起一个容器进程)
srun --mpi=pmix \
  singularity exec --nv \
    --bind /mnt/data:/data \
    my_train.sif \
    python -m torch.distributed.run \
      --nproc_per_node=8 train.py
```

关键机制：
- **Slurm 负责进程编排**，每个 rank 是一个独立的 `singularity exec`，没有容器内 daemon 协调，干净。
- **NCCL/MPI 的跨节点通信走宿主 InfiniBand**，容器把 `/dev/infiniband` 透传，RDMA 直达，不走慢速 TCP（这对 [[ai-infra/网络/集合通信原语]] 里的 AllReduce 性能是决定性的）。
- 容器内外 MPI 版本最好兼容（"hybrid MPI"模型）。

集合通信为什么是瓶颈：见 [[llm-inference/大模型推理张量并行]]——张量并行每层前向都要 AllReduce，对带宽极敏感。

## 8. 环境变量与 `--cleanenv`

Singularity 默认**把宿主的环境变量带进容器**（这和 Docker 相反）。这有时方便（`$CUDA_VISIBLE_DEVICES` 自动继承），有时危险（宿主的 `PYTHONPATH`、`LD_LIBRARY_PATH` 污染容器内 Python）。

```
  默认:    宿主 env  ──全部带入──►  容器 env  (可能污染)
  --cleanenv: 宿主 env  ──切断──►  容器只用镜像内 %environment
```

```bash
# 你已有片段：禁用所有宿主环境变量，确保容器环境独立可复现
singularity run --cleanenv my_container.sif

# 单独传变量（以 SINGULARITYENV_ / APPTAINERENV_ 前缀）
SINGULARITYENV_MASTER_ADDR=node0 \
  singularity exec --nv my_train.sif python train.py
#   ↑ 容器内会看到 MASTER_ADDR=node0（前缀被剥掉）
```

经验法则：**追求可复现就上 `--cleanenv`**，再用前缀变量显式注入你真正需要的（`MASTER_ADDR`、`RANK`、`WORLD_SIZE`、`NCCL_*`）。这样训练环境 100% 由镜像 + 显式变量决定，不受宿主登录环境干扰。

## 9. 定义文件 `.def`：可复现地构建镜像

`.def` 是镜像的"源代码"，类比 Dockerfile。构建出的 `.sif` 把整个软件栈锁死：

```
Bootstrap: docker
From: nvidia/cuda:12.1.0-devel-ubuntu22.04   ← 基础镜像

%post                                         ← 构建时执行(装软件)
    apt-get update && apt-get install -y git
    pip install torch==2.3.0 megatron-core

%environment                                  ← 运行时的环境变量
    export PYTHONUNBUFFERED=1
    export NCCL_IB_DISABLE=0

%runscript                                    ← run 时的默认行为
    exec python /opt/train.py "$@"

%labels
    Author you
    Version 1.0
```

```bash
singularity build my_train.sif train.def      # 需 root 或 --fakeroot
```

各段语义：`%post` = 装东西（构建期一次性）；`%environment` = 每次运行注入的环境；`%runscript` = `singularity run` 的入口；`%files` = 把宿主文件拷进镜像。**`.def` 进版本控制 = 软件栈可审计、可复现**——这正是科研论文"代码可复现"要的东西。

## ★ 数值手算

**(1) SIF 镜像体积估算。** 一个典型 LLM 训练镜像装什么：

```
  Ubuntu base + CUDA devel   ≈ 4.5 GB
  PyTorch + 依赖              ≈ 2.5 GB
  Megatron/DeepSpeed/apex    ≈ 1.0 GB
  ─────────────────────────────────
  解包总和                   ≈ 8.0 GB
  squashFS 压缩(约 0.45x)    ≈ 8.0 × 0.45 ≈ 3.6 GB  → .sif
```
即 $V_{sif} \approx \alpha \cdot V_{rootfs}$，压缩比 $\alpha \approx 0.4 \sim 0.5$。一个文件 3.6 GB，`scp` 到新集群只需一次传输，对比 Docker 解包后几十万 inode 的元数据风暴，优势巨大。

**(2) 镜像分发时间。** 把 3.6 GB 的 `.sif` 推到 64 个计算节点的本地盘，走 10 Gbps 管理网：

$$t = \frac{V \cdot N}{B} = \frac{3.6\text{ GB} \times 64 \times 8\,(\text{bit/Byte})}{10\text{ Gbps}} = \frac{1843\text{ Gb}}{10\text{ Gb/s}} \approx 184\text{ s}$$

若放共享并行文件系统（Lustre）则只存一份，各节点首次访问按需读 squashFS 块，省去复制——这是大集群的常见做法。

**(3) 多机训练通信 vs 容器开销。** 容器化几乎不增加通信开销，因为 NCCL 走的是宿主 IB（RDMA 直透），不经容器网络栈。以 7B 模型、4 节点 × 8 卡 = 32 卡数据并行为例，每步 AllReduce 梯度量：

$$\text{梯度字节} = P_{param} \times 2\,(\text{fp16}) = 7\times10^9 \times 2 = 14\text{ GB}$$

Ring-AllReduce 实际传输 $\approx 2 \times \frac{N-1}{N} \times 14\text{ GB} = 2 \times \frac{31}{32} \times 14 \approx 27.1\text{ GB}$。在 200 Gbps（25 GB/s）IB 上：

$$t_{comm} \approx \frac{27.1\text{ GB}}{25\text{ GB/s}} \approx 1.08\text{ s/step}$$

**容器本身对这 1.08 s 的贡献 ≈ 0**——因为 `--nv` 注入的是设备节点直透，数据面绕过容器。容器只在**进程启动**时有约 0.1~0.5 s 的一次性 mount/namespace 开销，对动辄数千 step 的训练完全可忽略：

$$\frac{t_{启动}}{t_{总}} = \frac{0.3\text{ s}}{5000\text{ step} \times 1.08\text{ s}} \approx 5.6\times10^{-5} \ll 1\%$$

结论：**HPC 容器化是"近零开销"的工程收益**——拿到可复现性和易部署，几乎不付性能税。

## 常见问题

| 问题 | 答案 |
|------|------|
| Singularity 和 Apptainer 啥关系 | 同一软件，2021 年项目捐给 Linux 基金会后社区版改名 **Apptainer**；命令几乎一致，`singularity` 多为别名。SingularityCE 是 Sylabs 维护的另一分支。 |
| 为什么 HPC 不用 Docker | Docker 需 root daemon、镜像多层污染共享盘、与 Slurm 进程模型冲突；Singularity 无 daemon、单文件、容器进程=普通用户进程。 |
| 容器里默认是 root 吗 | **不是**。默认你在容器内就是你的宿主 UID（非特权），这是安全设计核心。 |
| 数据怎么进容器 | 默认自动绑 `$HOME`/`$PWD`/`/tmp`；其余用 `--bind 宿主:容器[:ro]`。 |
| GPU 不可见 | 忘了加 `--nv`；或宿主驱动与镜像 CUDA 版本不匹配。先 `singularity exec --nv img nvidia-smi` 验证。 |
| 写文件报 read-only | 在写只读的 squashFS rootfs；输出必须落到 bind 进来的可写目录或 `$HOME`/`/tmp`。 |
| 训练环境不可复现 | 用 `--cleanenv` 切断宿主环境 + 用 `.def` 锁死软件栈 + 前缀变量显式注入。 |
| `.sif` 能改吗 | squashFS 只读；要可写改动用 overlay（`--overlay`）或 sandbox 模式（`build --sandbox`）。 |
| 命令默认值记不清 | 各发行版/版本默认绑定目录、标志可能不同，**以 `singularity --help` 与官方文档为准**。 |

## 🔗 跳转链接

- 集群作业调度搭档：[[slurm]]（同目录 `slurm.md`，Singularity 几乎总和 Slurm 一起用）
- 硬件地基：[[ai-infra/算力/GPU工作原理]] — 理解 `--nv` 透传的驱动/运行时分层
- 网络地基：[[ai-infra/网络/集合通信原语]] — 多机 AllReduce 走 IB 的原理
- 训练框架：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] — 镜像里装的就是它们
- 并行原理：[[llm-inference/大模型推理张量并行]] — 容器内多卡为何对带宽敏感
- 计算量地基：[[llm-algo/FLOPs]] — 配合本文通信量手算理解训练瓶颈
- 知识总览：[[00-知识地图]] · 本仓库训练总览 [[llm-train/README]]
