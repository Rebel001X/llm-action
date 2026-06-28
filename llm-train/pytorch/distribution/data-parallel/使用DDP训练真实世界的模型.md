# 使用 DDP 训练真实世界的模型（minGPT 字符级 GPT）

> 用一个能跑通的 minGPT 字符级语言模型，把 PyTorch DDP 从「单机多卡」一路打到「多机多卡 + SLURM + Singularity」全流程跑实。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/pytorch/distribution/README]] · [[ai-framework/pytorch/README]] · [[ai-infra/网络/NCCL]] · [[ai-infra/网络/集合通信原语]]

参考教程：<https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html>

---

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
|------|--------------|--------|
| 0. 一句话锚点 | DDP「真实世界」例子到底在练什么 | minGPT / char-level GPT / 27.32M |
| 1. 地基/前置 | DDP 的复制-切分-AllReduce 心智模型 | rank / world_size / DistributedSampler |
| 2. minGPT-ddp 代码结构 | 5 个源文件各司其职 | main.py / trainer.py / model.py |
| 3. 单机多卡 | `torchrun --standalone` 启动 + 日志逐行解读 | nproc_per_node / store_based_barrier |
| 4. 多机多卡（手动） | 方案一：每节点手敲 torchrun | node_rank / master_addr / NCCL_IB_DISABLE |
| 5. 多机多卡（SLURM） | 方案二：sbatch 一键拉起多节点 | srun / SLURM_JOB_NODELIST / rdzv |
| 6. Singularity 容器 | 在 HPC 容器里跑同一套命令 | .sif / --nv / -B 挂载 |
| 实操速查 | 所有真实命令/脚本原样汇总 | torchrun / sbatch / singularity |
| 常见问题/坑 | 卡死、连不上、loss 不降的根因 | bond0 / IB / 端口 / Sampler |

---

## 0. 一句话锚点

把一个 **27.32M 参数的字符级 GPT**（在莎士比亚式语料上，55769 个字符、59 个唯一字符）用 **DDP** 在多张 GPU、多台机器上并行训练；每张卡跑一个进程、持有一份完整模型副本、吃不同的数据分片，靠 **AllReduce 同步梯度**保证所有副本始终一致。

> 「真实世界的模型」相对于前面的玩具线性层而言：它有真实的数据加载（`char_dataset.py`）、真实的模型结构（`model.py`）、断点续训快照（snapshot）、YAML 配置（`gpt2_train_cfg.yaml`）——这正是工程上 DDP 训练脚本该长的样子。

---

## 1. 地基/前置：DDP 的复制-切分-同步三件套

DDP（`DistributedDataParallel`）的核心只有三句话：

1. **复制（Replicate）**：每个进程在自己的 GPU 上放一份**完整**模型副本（不是切模型，模型必须放得下单卡）。
2. **切分（Shard data）**：`DistributedSampler` 把数据集切成 `world_size` 份，每个进程只读自己那份，互不重叠。
3. **同步（AllReduce gradients）**：每个进程独立前向+反向算出**局部梯度**，反向过程中 DDP 自动对所有进程的梯度做 **AllReduce（求和取平均）**，于是所有副本拿到**相同的平均梯度**，`optimizer.step()` 后参数依旧一致。

```
                world_size = 4 (单机 4 卡)
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ rank0   │ │ rank1   │ │ rank2   │ │ rank3   │   每卡一份完整模型
   │ GPU0    │ │ GPU1    │ │ GPU2    │ │ GPU3    │   = 27.32M 参数
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
   data shard0  shard1     shard2     shard3       DistributedSampler 切分
        │           │          │          │
     forward     forward    forward    forward     各自算 local loss
        │           │          │          │
     backward    backward   backward   backward    各自算 local grad
        └───────────┴────AllReduce─────┴──────┘    梯度求和/N → 平均梯度
                        ↓ 所有 rank 拿到同一份平均梯度
                  optimizer.step()  → 参数保持一致
```

几个反复出现的术语，先钉死：

| 术语 | 含义 | minGPT 例子里的值 |
|------|------|-------------------|
| `world_size` | 全局进程总数（≈ 总 GPU 数） | 单机=4，2 机×4 卡=8 |
| `rank` | 全局进程编号 0..world_size-1 | 8 卡时 0..7 |
| `local_rank` | 本机内进程编号 | 每机 0..3 |
| `nnodes` | 节点（机器）数 | 1 或 2 |
| `nproc_per_node` | 每节点进程数（每节点 GPU 数） | 4 |
| `node_rank` | 节点编号 | 0=主节点, 1=从节点 |
| `master_addr/port` | 集合点（rendezvous）地址 | `xx.99.2.xx:29500` |

公式：`world_size = nnodes × nproc_per_node`。8 卡 = 2 × 4。

> 🔗 AllReduce 是这一切的引擎，原理见 [[ai-infra/网络/集合通信原语]]；GPU 间实际跑 AllReduce 的库是 NCCL，见 [[ai-infra/网络/NCCL]]。

---

## 2. minGPT-ddp 代码结构：5 个文件各司其职

`minGPT-ddp/` 用于训练的文件分工（来自其 README）：

| 文件 | 职责 |
|------|------|
| `main.py` | **训练入口**。建立 DDP 进程组、读取所有配置、运行训练作业 |
| `trainer.py` | `Trainer` 类，用给定数据集在模型上跑分布式训练迭代（含快照保存/加载） |
| `model.py` | 定义 GPT 模型结构（Transformer block） |
| `char_dataset.py` | 字符级数据集的 `Dataset` 类（把文本切成 char token） |
| `gpt2_train_cfg.yaml` | 数据、模型、优化器、训练运行的 YAML 配置 |

> 这套「入口 + Trainer + model + dataset + yaml」的拆分，就是 [[llm-algo/transformer/模型架构]] 在工程落地时的标准骨架。模型本身（GPT）的并行只用到 DDP（数据并行）；当模型大到单卡放不下时，才需要叠加张量并行/流水线并行，见 [[B07:llm-inference/大模型推理张量并行]]、[[llm-train/megatron/README]]。

---

## 3. 单机多卡：`torchrun --standalone`

### 3.1 启动命令

```bash
torchrun --standalone --nproc_per_node=4 main.py
```

逐参数解释：

| 参数 | 作用 | 为什么 |
|------|------|--------|
| `torchrun` | PyTorch 弹性启动器（替代老的 `python -m torch.distributed.launch`） | 自动注入 `RANK/LOCAL_RANK/WORLD_SIZE` 等环境变量，脚本无需手写 |
| `--standalone` | 单机模式，自动选一个空闲端口做 rendezvous | 单机不必显式指定 master_addr/port |
| `--nproc_per_node=4` | 本机起 4 个进程 = 用 4 张卡 | 一进程绑一卡，world_size=4 |
| `main.py` | 训练入口脚本 | — |

### 3.2 启动日志逐行解读（真实输出）

```text
> torchrun --standalone --nproc_per_node=4 main.py
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
[2023-09-08 17:48:02,237][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 1
[2023-09-08 17:48:02,246][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 0
[2023-09-08 17:48:02,253][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 2
[2023-09-08 17:48:02,256][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 3
[2023-09-08 17:48:02,257][torch.distributed.distributed_c10d][INFO] - Rank 3: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
[2023-09-08 17:48:02,257][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
[2023-09-08 17:48:02,258][torch.distributed.distributed_c10d][INFO] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
[2023-09-08 17:48:02,264][torch.distributed.distributed_c10d][INFO] - Rank 2: Completed store-based barrier for key:store_based_barrier_key:1 with 4 nodes.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
Data has 55769 characters, 59 unique.
number of parameters: 27.32M
number of parameters: 27.32M
number of parameters: 27.32M
number of parameters: 27.32M
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
Snapshot not found. Training model from scratch
[GPU3] Epoch 1 | Iter 0 | Train Loss 4.18844
[GPU0] Epoch 1 | Iter 0 | Train Loss 4.18055
[2023-09-08 17:48:06,586][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[GPU2] Epoch 1 | Iter 0 | Train Loss 4.18575
[GPU1] Epoch 1 | Iter 0 | Train Loss 4.18100
[2023-09-08 17:48:06,590][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[2023-09-08 17:48:06,590][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[2023-09-08 17:48:06,590][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.
[GPU0] Epoch 1 | Iter 0 | Eval Loss 2.35044
[GPU2] Epoch 1 | Iter 0 | Eval Loss 2.33801
...
[GPU1] Epoch 3 | Iter 0 | Train Loss 2.21209
[GPU2] Epoch 3 | Iter 0 | Train Loss 2.21229
Snapshot saved at epoch 3
[GPU2] Epoch 3 | Iter 0 | Eval Loss 2.12978
[GPU1] Epoch 3 | Iter 0 | Eval Loss 2.12159
...
[GPU2] Epoch 6 | Iter 0 | Train Loss 1.96697
[GPU1] Epoch 6 | Iter 0 | Train Loss 1.97281
Snapshot saved at epoch 6
[GPU2] Epoch 6 | Iter 0 | Eval Loss 1.84860
[GPU3] Epoch 6 | Iter 0 | Eval Loss 1.85607
[GPU1] Epoch 6 | Iter 0 | Eval Loss 1.86408
[GPU0] Epoch 6 | Iter 0 | Eval Loss 1.87735
...
[GPU2] Epoch 9 | Iter 0 | Train Loss 1.43359
[GPU0] Epoch 9 | Iter 0 | Train Loss 1.45036
[GPU1] Epoch 9 | Iter 0 | Train Loss 1.44486
[GPU3] Epoch 9 | Iter 0 | Train Loss 1.41639
Snapshot saved at epoch 9
[GPU2] Epoch 9 | Iter 0 | Eval Loss 1.28527
[GPU3] Epoch 9 | Iter 0 | Eval Loss 1.29209
...
[GPU3] Epoch 10 | Iter 0 | Eval Loss 1.13304
[GPU1] Epoch 10 | Iter 0 | Eval Loss 1.13287
```

**这段日志的 5 个关键信号：**

| 日志片段 | 含义 | 为什么重要 |
|----------|------|-----------|
| `Setting OMP_NUM_THREADS ... to be 1` | torchrun 默认把每进程 OpenMP 线程数设为 1 | 防止 N 个进程各开满 CPU 线程互相抢核（4 进程 × 满线程会过载）；可手动调 `export OMP_NUM_THREADS=...` 优化 |
| `store_based_barrier ... with 4 nodes` ×4 | 4 个进程都加入了 rendezvous 屏障并完成同步 | **这是"建组成功"的标志**；如果卡在这里说明有进程连不上（网络/端口问题，见坑表） |
| `Data has 55769 characters, 59 unique` ×4 | 4 个进程各自加载了同一份数据集元信息 | 每进程独立 `Dataset`，再由 `DistributedSampler` 切片 |
| `number of parameters: 27.32M` ×4 | 4 份**完全相同**的模型副本 | 印证「复制」——DDP 不切模型，每卡都是 27.32M |
| `Reducer buckets have been rebuilt` | DDP 第一次迭代后按梯度实际就绪顺序重排了梯度桶（bucket） | DDP 把参数梯度分桶做 AllReduce，重建桶是为了**通信与反向计算重叠**，提升带宽利用 |

**loss 收敛印证 DDP 正确性**——训练 10 个 epoch，4 卡 train loss 同步下降：

```
Train Loss 轨迹 (取各 GPU 近似值)
4.18 ┤██  epoch1   ← 随机初始化，~ln(59)=4.08 附近
     │
2.21 ┤      ██  epoch3  (snapshot saved)
     │
1.97 ┤            ██  epoch6  (snapshot saved)
     │
1.44 ┤                  ██  epoch9 (snapshot saved)
1.13 ┤                       ██  epoch10 (eval)
     └────────────────────────────────────────►
```

**数值手算**：59 个唯一字符，模型未训练时近似均匀分布，初始交叉熵约 $\ln(59)\approx 4.08$。日志 epoch1 train loss ≈ 4.18，与理论值吻合——说明初始化正常、数据 pipeline 没接错。10 个 epoch 后 eval loss 降到 ~1.13，说明模型确实学到了字符级语言规律。

**快照（snapshot）机制**：`Snapshot not found. Training model from scratch` → 从零训练；之后 `Snapshot saved at epoch 3/6/9` → 周期性存盘。重跑时若检测到快照，会从断点续训（断点续训能力是「真实世界」脚本的标配）。

> ⚠️ 注意：日志里 `store-based barrier ... with 4 nodes` 的 "nodes" 指的是**进程数**，不是物理机器数。单机 4 卡也叫 "4 nodes"。

---

## 4. 多机多卡（方案一：手动逐节点启动）

当一台机器卡不够，就把 `world_size` 扩到多机。**方案一**是最直白的：在每台机器上手敲一条 `torchrun`，唯一区别是 `--node_rank` 不同。

```bash
# === 在主节点（node_rank=0）上执行 ===
export NCCL_IB_DISABLE=1 && export NCCL_SOCKET_IFNAME=bond0 && torchrun \
--nproc_per_node=2 --nnodes=2 --node_rank=0 \
--master_addr=xx.99.2.xx --master_port=29500 \
main.py

# === 在从节点（node_rank=1）上执行 ===
export NCCL_IB_DISABLE=1 && export NCCL_SOCKET_IFNAME=bond0 && torchrun \
--nproc_per_node=2 --nnodes=2 --node_rank=1 \
--master_addr=xx.99.2.xx --master_port=29500 \
main.py
```

```
            主节点 (node_rank=0, master_addr=xx.99.2.xx:29500)
         ┌────────────────────────────────────────────┐
         │  torchrun --nnodes=2 --node_rank=0          │
         │   ┌────────┐  ┌────────┐                    │
         │   │ rank0  │  │ rank1  │  (nproc=2)          │
         │   └────────┘  └────────┘                    │
         └───────────────▲────────────────────────────┘
                         │ TCP rendezvous @ 29500
                         │ NCCL 走 bond0 网卡 (IB 关闭)
         ┌───────────────▼────────────────────────────┐
         │  torchrun --nnodes=2 --node_rank=1          │
         │   ┌────────┐  ┌────────┐                    │
         │   │ rank2  │  │ rank3  │                    │
         │   └────────┘  └────────┘                    │
         └────────────────────────────────────────────┘
              从节点 (node_rank=1, 连同一个 master_addr)
                world_size = nnodes×nproc = 2×2 = 4
```

**逐项「为什么」：**

| 设置 | 作用 | 不设会怎样 |
|------|------|-----------|
| `--master_addr` 两节点填**同一个**主节点 IP | 所有进程到同一个 rendezvous 点集合 | 填错/不一致 → 永远建不成组，卡在 store_based_barrier |
| `--master_port=29500` 两节点**同一端口** | 同上 | 端口被占用/被防火墙挡 → 连接超时 |
| `--node_rank` 每节点**唯一** | 区分谁是 0 谁是 1，决定 global rank 分配 | 两节点都填 0 → rank 冲突，行为未定义 |
| `NCCL_SOCKET_IFNAME=bond0` | **强制** NCCL 走 `bond0` 这块网卡通信 | 多网卡机器上 NCCL 可能选错网卡（如 docker0/lo）→ 跨机通信失败或极慢 |
| `NCCL_IB_DISABLE=1` | 关闭 InfiniBand，强制走 Socket(TCP/以太网) | 没有 IB 硬件 / IB 没配好时，不关会报错或挂死 |

> 💡 `bond0`/`master_addr` 里的 `xx`/`10.xx.2.xxx` 是脱敏占位符，实际填你集群的真实网卡名和主节点内网 IP。

---

## 5. 多机多卡（方案二：SLURM 作业调度）

手动逐节点登录太累，生产环境用 **SLURM** 一条 `sbatch` 拉起所有节点。仓库提供了多个 sbatch 脚本，核心思路一致：**让 SLURM 分配节点 → 脚本里取出主节点 IP → 每节点跑 torchrun**。

### 5.1 提交

```bash
sbatch slurm/sbatch_run.sh
```

### 5.2 sbatch 脚本（真实，循环式逐节点 srun torchrun）

```bash
#!/bin/bash

#SBATCH --job-name=multinode-example
#SBATCH --partition=a800 #分区

#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=4

NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
# 对第一个节点赋值为主节点
MASTER_NODE=$(head -n 1 <<< "$NODELIST")
# 计数器
NODE_COUNT=0
# 一共的节点数
NODE_NUM=($(echo $NODELIST | tr " " "\n" | wc -l))

# 打印
echo $SLURM_NODEID
echo $NODELIST
echo $MASTER_NODE
echo $NODE_NUM

module load anaconda/3-2023.03
source activate
conda activate liguodong-310-multinode
module load cuda-cudnn8.9/11.7.1

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0

for NODE in $NODELIST; do
    if [ "$NODE" == "$MASTER_NODE" ]; then
        srun --nodes=1 --ntasks=1 -w $NODE torchrun --nproc_per_node=4 --nnodes=$NODE_NUM --node_rank=0 --master_addr=xx.99.2.xx --master_port=29500 main.py &
    else
        ((NODE_COUNT++))
        srun --nodes=1 --ntasks=1 -w $NODE torchrun --nproc_per_node=4 --nnodes=$NODE_NUM --node_rank=$NODE_COUNT --master_addr=xx.99.2.xx --master_port=29500 main.py &
    fi
done
wait
```

**脚本干了什么（自顶向下）：**

```
#SBATCH 头  ──► 向 SLURM 申请：2 节点、每 task 4 GPU、a800 分区
   │
scontrol show hostname $SLURM_JOB_NODELIST  ──► 拿到分到的节点名列表 NODELIST
   │
head -n 1  ──► 第一个节点当 MASTER_NODE
   │
module load / conda activate  ──► 装好 Python/CUDA 环境
   │
export NCCL_*  ──► 关 IB、锁定 bond0 网卡
   │
for NODE in NODELIST:
   ├─ 若是主节点 → srun ... torchrun --node_rank=0 ... &  (后台)
   └─ 否则       → srun ... torchrun --node_rank=$((++count)) ... &  (后台)
wait  ──► 等所有后台 srun 结束
```

关键点：`&` 把每个 `srun` 丢后台**并行**起，最后 `wait` 收尾；`node_rank` 由 `MASTER_NODE` 判断+计数器自增唯一分配，避免冲突。

### 5.3 另一种 srun 风格（rdzv 集合点，来自 `sbatch_run.sh` 早期版本）

PyTorch 官方推荐的弹性集合点（rendezvous）写法，用 `--rdzv_*` 替代手填 `node_rank`：

```bash
#!/bin/bash
#SBATCH --job-name=multinode-example
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

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

| 方式 | node_rank 怎么来 | 优点 |
|------|------------------|------|
| 显式 `--node_rank` + `--master_addr`（5.2） | 脚本里循环手算 | 直观，老 PyTorch 也支持 |
| `--rdzv_id/--rdzv_backend c10d/--rdzv_endpoint`（5.3） | torchrun 自动协商 | **弹性容错**，节点挂了能重 rendezvous，官方推荐 |

### 5.4 多机日志（8 卡，真实节选）

```text
> tail -100f slurm-949.out
0
ai-app-2-45 ai-app-2-46
ai-app-2-45
2
WARNING:torch.distributed.run:
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default ...
[2023-09-11 20:52:55,311][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 0
...（rank 0..7 依次加入）...
[2023-09-11 20:52:55,601][...] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
Data has 55769 characters, 59 unique.   ×8
number of parameters: 27.32M            ×8
Snapshot not found. Training model from scratch   ×8
[GPU5] Epoch 1 | Iter 0 | Train Loss 4.15149
[GPU4] Epoch 1 | Iter 0 | Train Loss 4.14975
...（GPU0..7 八张卡）...
[2023-09-11 20:53:06,174][torch.nn.parallel.distributed][INFO] - Reducer buckets have been rebuilt in this iteration.   ×8
[GPU0] Epoch 1 | Iter 0 | Eval Loss 2.41668
...
[GPU3] Epoch 10 | Iter 0 | Train Loss 1.97779
[GPU0] Epoch 10 | Iter 0 | Eval Loss 1.94594
[GPU2] Epoch 10 | Iter 0 | Eval Loss 1.94512
```

**对照单机 vs 多机**（同样 minGPT，同样 10 epoch）：

| 维度 | 单机 4 卡 | 多机 2×4=8 卡 |
|------|-----------|----------------|
| `store_based_barrier ... with N nodes` | 4 | 8 |
| 模型副本数 / 参数 | 4 × 27.32M | 8 × 27.32M（每卡仍是完整副本） |
| 节点列表 | 单机 | `ai-app-2-45 ai-app-2-46` |
| epoch1 train loss | ~4.18 | ~4.15 |
| epoch10 eval loss | ~1.13 | ~1.94 |

> ⚠️ 同样训 10 epoch，8 卡的 epoch10 eval loss（~1.94）反而比 4 卡（~1.13）高。原因：**卡数翻倍 → 等效全局 batch size 翻倍 → 同样 epoch 数下参数更新步数减半**，模型「见数据的次数」少了。这是数据并行扩展的经典坑——**扩卡通常要同步调大学习率（linear scaling rule）或增加 epoch/步数**，否则收敛变慢。见坑表。

---

## 6. Singularity 容器中运行（HPC 常见）

HPC 集群一般不让随便装环境，用 **Singularity（apptainer）** 容器跑。命令骨架与裸机一致，只是外面套一层 `singularity run`。

### 6.1 拉取镜像

```bash
singularity pull pytorch-multinode.sif docker://harbor.aip.io/base/pytorch-multinode:v1
```

### 6.2 单机多卡（容器内）

```bash
singularity run --nv \
--pwd /workspaces/examples-main/distributed/minGPT-ddp/mingpt \
-B /data/hpc/home/guodong.li/:/workspaces:rw \
pytorch-multinode.sif \
torchrun --standalone --nproc_per_node=4 main.py
```

### 6.3 多机多卡（容器内，逐节点）

```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0

# 主节点 node_rank=0
singularity run --nv \
--pwd /workspaces/examples-main/distributed/minGPT-ddp/mingpt \
-B /data/hpc/home/guodong.li/:/workspaces:rw \
pytorch-multinode.sif \
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
--master_addr=10.xx.2.xxx --master_port=29500 main.py

# 从节点 node_rank=1
singularity run --nv \
--pwd /workspaces/examples-main/distributed/minGPT-ddp/mingpt \
-B /data/hpc/home/guodong.li/:/workspaces:rw \
pytorch-multinode.sif \
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
--master_addr=10.xx.2.xxx --master_port=29500 main.py
```

### 6.4 配 SLURM（容器 + srun + singularity）

`minGPT-ddp/multinode.sh` 把 SLURM、Singularity、torchrun 三者拼起来（用 `--mpi=pmix_v3` 让 srun 启动容器内进程）：

```bash
#!/bin/bash
#SBATCH --job-name=multinode-example
#SBATCH --partition=a800 #分区
#SBATCH --output=log/%j.out #日志
#SBATCH --error=log/%j.err #日志
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:4

NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
MASTER_NODE=$(head -n 1 <<< "$NODELIST")
NODE_NUM=($(echo $NODELIST | tr " " "\n" | wc -l))
echo $SLURM_NODEID; echo $NODELIST; echo $MASTER_NODE; echo $NODE_NUM

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0

srun --mpi=pmix_v3 singularity run --nv \
--pwd /workspaces/examples-main/distributed/minGPT-ddp/mingpt \
-B /data/hpc/home/guodong.li/:/workspaces:rw \
pytorch-multinode.sif torchrun --nproc_per_node=4 main.py
```

或用现成脚本：`slurm/sbatch_run_sig.sh`。

**`singularity run` 参数解释：**

| 参数 | 作用 |
|------|------|
| `--nv` | 把宿主机 NVIDIA 驱动/库透传进容器（用 GPU 必加） |
| `--pwd <dir>` | 进入容器后的工作目录（指向 mingpt 代码所在路径） |
| `-B host:container:rw` | 绑定挂载：把宿主机目录映射进容器（数据/代码/输出读写） |
| `pytorch-multinode.sif` | 容器镜像文件 |
| `--mpi=pmix_v3` | srun 用 PMIx 启动容器内 MPI/分布式进程 |

---

## 实操速查（所有真实命令/脚本汇总）

| 场景 | 命令 |
|------|------|
| 单机 4 卡 | `torchrun --standalone --nproc_per_node=4 main.py` |
| 多机·手动·主节点 | `export NCCL_IB_DISABLE=1 && export NCCL_SOCKET_IFNAME=bond0 && torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=xx.99.2.xx --master_port=29500 main.py` |
| 多机·手动·从节点 | 同上，`--node_rank=1` |
| 多机·SLURM | `sbatch slurm/sbatch_run.sh`（循环式）或 `slurm/sbatch_run_sig.sh`（容器版） |
| 看日志 | `tail -100f slurm-949.out` |
| 拉容器 | `singularity pull pytorch-multinode.sif docker://harbor.aip.io/base/pytorch-multinode:v1` |
| 容器·单机 | `singularity run --nv --pwd <mingpt> -B <host>:<ctn>:rw pytorch-multinode.sif torchrun --standalone --nproc_per_node=4 main.py` |
| 关键环境变量 | `export NCCL_IB_DISABLE=1; export NCCL_SOCKET_IFNAME=bond0; export LOGLEVEL=INFO` |

启动方式选择树：

```
要跑 DDP 吗？
 ├─ 单机多卡 ────────────► torchrun --standalone --nproc_per_node=N
 └─ 多机多卡
     ├─ 手头没调度器，手动登录每台机 ─► 每节点 torchrun + --node_rank/--master_addr
     ├─ 有 SLURM
     │    ├─ 想容错/弹性 ─► srun torchrun + --rdzv_backend c10d --rdzv_endpoint
     │    └─ 简单稳定   ─► sbatch 脚本循环 srun ... torchrun --node_rank=...
     └─ HPC 容器环境 ──► 外套 singularity run --nv --pwd --B，里面命令不变
```

---

## 常见问题 / 坑

| 现象 | 根因 | 解法 |
|------|------|------|
| 卡在 `store_based_barrier ... with N nodes` 不动 | 有进程没连上 rendezvous（IP/端口/网卡错） | 检查所有节点 `--master_addr/--master_port` 是否一致且可达；端口没被占；防火墙放行 29500 |
| 跨机 NCCL 报错或挂死 | NCCL 选错网卡（如选了 `lo`/`docker0`） | `export NCCL_SOCKET_IFNAME=bond0` 锁定正确网卡；多网卡用真实网卡名 |
| 无 IB 硬件却报 IB 相关错 | NCCL 默认想走 InfiniBand | `export NCCL_IB_DISABLE=1` 退回 Socket 通信 |
| 各进程都填 `--node_rank=0` | rank 冲突，行为未定义/挂死 | 每节点 `node_rank` 必须唯一（0,1,2...） |
| CPU 被 N 个进程打满、整机卡顿 | 每进程默认开满 OpenMP 线程 | torchrun 已默认 `OMP_NUM_THREADS=1`；按需 `export OMP_NUM_THREADS=合理值` |
| 扩到更多卡后同 epoch 数 loss 更高（8 卡 eval 1.94 vs 4 卡 1.13） | 全局 batch 变大→更新步数变少；学习率没跟着调 | 用 linear scaling rule 调大 LR，或增大 epoch/步数；用 warmup |
| 各 rank 训练数据重复、相当于没并行 | 没用 `DistributedSampler` 或忘了每 epoch `sampler.set_epoch()` | DataLoader 传 `DistributedSampler`，每 epoch 调 `set_epoch(e)` 保证 shuffle 一致且分片不重叠 |
| 多机训练某节点 OOM/挂掉全军覆没 | 手动 `--node_rank` 模式无容错 | 改用 `--rdzv_backend c10d` 弹性集合点，节点掉了能重 rendezvous |
| 容器内看不到 GPU | 忘了 `--nv` | `singularity run --nv ...` 透传 NVIDIA 驱动 |
| 容器内找不到代码/数据 | 没挂载或 `--pwd` 路径不对 | `-B host:container:rw` 挂载，`--pwd` 指向容器内代码目录 |

> 关于「为什么扩卡反而要调学习率」：DDP 把梯度做 **AllReduce 取平均**，等效于把 N 个 micro-batch 拼成一个大 batch。大 batch 的梯度估计方差更小、更新更稳，但同样 epoch 下更新次数 = 样本数 / (per-gpu-batch × world_size) 减少。要么补步数，要么按 `lr_new = lr_base × world_size` 放大学习率（配 warmup）。这是数据并行扩展的第一性原理。

---

## 🔗 跳转链接

**枢纽 / 总览**
- [[00-知识地图]]
- [[llm-train/README]]
- [[llm-train/pytorch/distribution/README]]

**框架与并行**
- [[ai-framework/pytorch/README]]
- [[llm-train/megatron/README]] · [[llm-train/megatron-deepspeed/README]]
- [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]
- [[B07:llm-inference/大模型推理张量并行]]

**通信底座**
- [[ai-infra/网络/集合通信原语]]（AllReduce 是 DDP 梯度同步的引擎）
- [[ai-infra/网络/NCCL]]（`NCCL_IB_DISABLE` / `NCCL_SOCKET_IFNAME` 出处）

**模型与微调**
- [[llm-algo/transformer/模型架构]]（minGPT 就是 GPT/Transformer）
- [[ai-framework/huggingface-peft/README]] · [[llm-train/peft/PEFT-API]]
- [[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]]

**延伸**
- [[llm-alignment/RLHF]]
- [[llm-compression/quantization/量化基础]]
