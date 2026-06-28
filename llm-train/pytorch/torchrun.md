# torchrun 启动

> torchrun 是 PyTorch 官方的分布式作业启动器：它负责"拉起 N 个训练进程、给每个进程发一份身份证（RANK 等环境变量）、在崩溃时弹性重启"，让你只写单进程逻辑就能跑满整机/整集群。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-train/pytorch/distribution/多机多卡]] [[llm-train/pytorch/distribution/README]]

- 官方文档：https://pytorch.org/docs/stable/elastic/run.html

---

## 阅读地图

| 节 | 主题 | 一句话收获 |
|----|------|-----------|
| 0 | 一句话锚点 | torchrun = 进程拉起器 + 环境变量注入器 + 弹性容错调度器 |
| 1 | 它解决什么问题 | 手写 `mp.spawn` / 手设 RANK 的痛点，torchrun 全包了 |
| 2 | torchrun vs launch | `torch.distributed.launch` 的演进与差异 |
| 3 | 进程拓扑与环境变量 | RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR 的含义与来历 |
| 4 | Rendezvous 弹性会合 | 进程如何"认亲"、谁是 master、c10d 后端 |
| 5 | 单机多卡 | 一行命令拉满 8 卡的标准姿势 |
| 6 | 多机多卡 | 多节点 + `--rdzv-endpoint` 的组网方式 |
| 7 | 故障重启与弹性伸缩 | min/max nodes、重启语义、checkpoint 配合 |
| 8 | 训练脚本怎么写 | `init_process_group` + `set_device` 的最小骨架 |
| — | 配置示例 | 关键参数逐个解释 |
| — | 常见问题 | 端口冲突 / hang / NCCL 报错排查 |

---

## 0. 一句话锚点

> **torchrun 是一个"作业级"的命令行工具：你把训练脚本交给它，它在每张卡（或每个进程）上各启动一份脚本副本，并通过环境变量告诉每份副本"你是第几号、一共几个、去哪里集合"，同时在某个进程挂掉时按策略重启整组进程。**

记住三个职责，后面所有内容都是这三件事的展开：

```
            ┌──────────────────────────────────────────────┐
            │                  torchrun                     │
            ├──────────────────────────────────────────────┤
   职责①    │  拉起 N 个进程（每卡一个 / 每节点多个）          │
   职责②    │  给每个进程注入身份: RANK/LOCAL_RANK/WORLD_SIZE │
   职责③    │  会合(rendezvous) + 监控 + 故障弹性重启          │
            └──────────────────────────────────────────────┘
```

---

## 1. 地基/前置：它到底解决什么问题

要理解 torchrun，先看没有它时你得手动做什么。PyTorch 的数据并行（DDP）要求**每张 GPU 上跑一个独立的 OS 进程**，这些进程之间用集合通信（all-reduce 同步梯度）。于是必须有人回答四个问题：

1. **要起几个进程？** —— 单机 8 卡就是 8 个进程。
2. **每个进程的编号是什么？** —— DDP 需要 `rank` 来区分谁是谁、谁负责 rank0 的日志/保存。
3. **它们怎么找到彼此？** —— 需要一个约定的"集合点"（master 地址 + 端口）。
4. **挂了一个怎么办？** —— 训练几小时后某卡 OOM 或机器掉线，整组要不要重来？

**手动方案的痛点**（这正是 torchrun 的存在理由）：

```
手动 mp.spawn / 手设环境变量的世界：
  ├─ 你要自己 export RANK=0 / RANK=1 ... 在每台机器上敲
  ├─ 多机时要保证编号全局唯一（机器A是0-3，机器B是4-7），易错
  ├─ MASTER_ADDR/MASTER_PORT 要手动协调，端口撞了就 hang
  ├─ 进程崩了 → 整个作业死掉，没有自动恢复，几小时白跑
  └─ 想从 8 卡缩到 4 卡继续跑？没门，得改一堆参数重启
```

torchrun 把上面这些**全部自动化**：你只需告诉它"几台机器、每台几个进程、集合点在哪"，它负责编号、注入、监控、重启。**你的训练脚本因此可以完全不感知自己是单机还是多机**——它只管从环境变量读身份。这是 torchrun 设计上最重要的"解耦"。

---

## 2. torchrun vs `torch.distributed.launch`

torchrun 是旧启动器 `python -m torch.distributed.launch` 的**官方继任者**（底层基于 TorchElastic / `torch.distributed.run`）。两者关系：

```
   torch.distributed.launch  (旧, 仍可用但不再推荐)
            │  能力: 拉进程 + 设环境变量
            │  缺陷: 无弹性、无容错、靠 --use_env 才传环境变量
            ▼
   torch.distributed.run     (新模块, = TorchElastic 入口)
            │
            ▼
   torchrun                  (新, 上面那个模块的命令行别名)
            能力: 拉进程 + 环境变量 + rendezvous + 弹性容错
```

| 维度 | `torch.distributed.launch`（旧） | `torchrun`（新） |
|------|-------------------------------|-----------------|
| 本质 | 简单进程拉起器 | 基于 TorchElastic 的弹性调度器 |
| 传 LOCAL_RANK | 需 `--local_rank` 命令行参数或 `--use_env` | 直接走环境变量，脚本读 `os.environ["LOCAL_RANK"]` |
| 弹性伸缩 | 不支持 | 支持 `--nnodes=min:max` 动态增减节点 |
| 故障重启 | 不支持，一个进程死全死 | 支持 worker 级重启（按策略重拉整组） |
| 会合机制 | 固定 master，节点必须全到齐 | rendezvous，节点可动态加入/退出 |
| 推荐度 | 兼容保留 | **现在的标准做法** |

> 思路要点：能用 torchrun 就不要再用 launch。脚本里**不要**再写 `argparse` 解析 `--local_rank`，改为 `int(os.environ["LOCAL_RANK"])`，这是从 launch 迁移到 torchrun 最常见的改动点。（精确的命令行参数名以官方文档为准。）

---

## 3. 进程拓扑与四大环境变量

torchrun 注入的环境变量就是每个进程的"身份证"。先看拓扑，再看变量。

### 3.1 拓扑：节点 / 进程 / rank 的层级

```
        作业 (Job, WORLD_SIZE = 全局进程总数)
        ┌──────────────────────────────┬──────────────────────────────┐
        │          Node 0              │           Node 1              │
        │  (一台物理机, node_rank=0)    │   (一台物理机, node_rank=1)    │
        │ ┌────┬────┬────┬────┐         │ ┌────┬────┬────┬────┐          │
        │ │GPU0│GPU1│GPU2│GPU3│         │ │GPU0│GPU1│GPU2│GPU3│          │
        │ │ p  │ p  │ p  │ p  │         │ │ p  │ p  │ p  │ p  │          │
        │ └────┴────┴────┴────┘         │ └────┴────┴────┴────┘          │
        │ LOCAL_RANK 0  1  2  3         │ LOCAL_RANK 0  1  2  3          │
        │ RANK       0  1  2  3         │ RANK       4  5  6  7          │
        └──────────────────────────────┴──────────────────────────────┘
              WORLD_SIZE = 8   (2 节点 × 4 进程)
```

关键区分：**LOCAL_RANK 是"机内编号"，RANK 是"全局编号"**。绑定 GPU 用 LOCAL_RANK（机内 0~3 对应本机 GPU 0~3）；判断"是不是主进程做日志/存盘"用 RANK（全局 0 号）。

### 3.2 核心环境变量速查

| 变量 | 含义 | 单机 8 卡的取值 | 双机 ×4 的取值 |
|------|------|----------------|----------------|
| `RANK` | **全局**进程编号，唯一 | 0~7 | 0~7（跨机连续） |
| `LOCAL_RANK` | **机内**进程编号 | 0~7 | 各机 0~3 |
| `WORLD_SIZE` | 全局进程总数 | 8 | 8 |
| `LOCAL_WORLD_SIZE` | 本机进程数 | 8 | 4 |
| `MASTER_ADDR` | 会合主节点 IP/主机名 | 127.0.0.1 | node0 的 IP |
| `MASTER_PORT` | 会合端口（c10d 监听） | 如 29500 | 同一个端口 |
| `GROUP_RANK` / `NODE_RANK` | 节点编号 | 0 | 0 或 1 |

> **绝不混淆**：`RANK` 用于集合通信里的全局身份与"主进程"判断；`LOCAL_RANK` 用于 `torch.cuda.set_device(local_rank)` 绑卡。多机时若拿 `RANK` 去 `set_device` 会越界（机器只有 4 张卡却拿到 rank=5）。

### 3.3 数据流：环境变量从哪来、到哪去

```
   torchrun ──fork──► 算每个进程的 RANK / 设 MASTER_ADDR ──env注入──►
        worker 进程 os.environ[RANK/LOCAL_RANK/...] ──► dist.init_process_group() 读 env 组网
```

脚本里**不需要**手动 export 这些变量，torchrun 已经替你设好；脚本只负责**读**。

---

## 4. Rendezvous（会合）：进程如何"认亲"

弹性能力的核心是 **rendezvous（会合）机制**。它解决"一堆刚启动、互相不认识的进程，如何商定出 WORLD_SIZE、各自的 RANK、谁当 master"。

### 4.1 会合在做什么

```
   t0: 各 agent 启动, 谁也不知道总共几个
        Node0/1/2-agent ──► 都连向同一个 rendezvous 端点(rdzv-endpoint)
   t1: rendezvous 后端(如 c10d)收齐到达者
        - 等待 >= min_nodes 个 agent 到达
        - 形成本轮成员快照(membership) + 分配全局 RANK + 选出 master
   t2: 把 WORLD_SIZE / RANK / MASTER_ADDR 下发给各 agent
        ──► agent 据此拉起 worker 并注入环境变量
```

### 4.2 三个关键参数（思路，非精确默认值）

| 参数 | 作用 | 权衡 |
|------|------|------|
| `--rdzv-backend` | 会合后端，常用 `c10d`（内置，无需外部依赖） | `c10d` 自带；`etcd` 需额外部署但更适合大规模 |
| `--rdzv-endpoint` | 会合端点 `host:port`，所有节点填**同一个** | 必须可被所有节点访问；通常指向 node0 |
| `--rdzv-id` | 作业唯一标识（job id），同一作业各节点要一致 | 用于隔离不同作业，避免串台 |

> 对比：旧 launch 用固定的 `MASTER_ADDR/MASTER_PORT` 静态组网，**所有节点必须同时全部到齐**；torchrun 的 rendezvous 允许"先到 min_nodes 个就开跑，后到的下一轮加入"，这正是弹性的来源。c10d 后端会让 rendezvous endpoint 所在节点充当协调者并兼任 master。

### 4.3 为什么需要会合而不是手填 RANK

手填 RANK 在静态、小规模、永不挂的理想世界里能用；一旦要弹性（节点数会变）或容错（挂了重组），RANK 就不能写死——必须**每轮会合重新分配**。会合就是"动态、可重复的 RANK 分配协议"。

---

## 5. 单机多卡

最常见场景。思路：**一台机器，nproc-per-node 设成 GPU 数**，rendezvous 走本机回环，无需关心 IP。

```
   torchrun \
     --standalone \            # 单机模式: 自动用本机回环做会合, 免填 endpoint
     --nproc-per-node=8 \      # 本机起 8 个进程(对应 8 张 GPU)
     train.py --args ...
```

调用链：

```
   torchrun --standalone --nproc-per-node=8 train.py
        │
        ▼
   本机 agent: WORLD_SIZE=8, 自动选空闲端口做 rendezvous
        │
        ├─ worker0  RANK=0 LOCAL_RANK=0 ─► set_device(0) ─► GPU0
        ├─ worker1  RANK=1 LOCAL_RANK=1 ─► set_device(1) ─► GPU1
        ├─  ...
        └─ worker7  RANK=7 LOCAL_RANK=7 ─► set_device(7) ─► GPU7
                所有 worker 跑同一份 train.py
```

要点：
- `--standalone` 省去手填 `--rdzv-endpoint`，单机首选。
- `--nproc-per-node` 一般 = 可见 GPU 数；也可设 `gpu` 让其自动取（具体写法以官方文档为准）。
- 单机内 `MASTER_ADDR` 实际是 `127.0.0.1`，端口由 torchrun 挑选，基本不会撞。

---

## 6. 多机多卡

思路：**每台机器都跑一条 torchrun 命令**，靠相同的 `--rdzv-id` + 相同的 `--rdzv-endpoint` 把它们"会合"成一个作业。详见 [[llm-train/pytorch/distribution/多机多卡]]。

```
   ┌─────────────── Node 0 (作为 rendezvous endpoint, IP=10.0.0.1) ───────────────┐
   │  torchrun \                                                                    │
   │    --nnodes=2 \              # 总共 2 个节点                                    │
   │    --nproc-per-node=4 \      # 每节点 4 进程                                    │
   │    --rdzv-id=job123 \        # 作业 ID, 两节点必须一致                          │
   │    --rdzv-backend=c10d \                                                       │
   │    --rdzv-endpoint=10.0.0.1:29500 \   # 会合点, 两节点填同一个                  │
   │    train.py                                                                    │
   └────────────────────────────────────────────────────────────────────────────────┘
   ┌─────────────── Node 1 (IP=10.0.0.2) ──────────────────────────────────────────┐
   │  torchrun \                                                                    │
   │    --nnodes=2 \                                                                │
   │    --nproc-per-node=4 \                                                        │
   │    --rdzv-id=job123 \        # 与 node0 相同                                    │
   │    --rdzv-backend=c10d \                                                       │
   │    --rdzv-endpoint=10.0.0.1:29500 \   # 仍指向 node0                            │
   │    train.py                                                                    │
   └────────────────────────────────────────────────────────────────────────────────┘
```

组网后的全局视图（WORLD_SIZE = 2 × 4 = 8）：

```
   Node0: RANK 0,1,2,3  (LOCAL_RANK 0,1,2,3)
   Node1: RANK 4,5,6,7  (LOCAL_RANK 0,1,2,3)   ← 注意 LOCAL_RANK 各机重新从 0 数
                          │
                          ▼  NCCL all-reduce 跨机同步梯度(走 RDMA/IB 或 TCP)
```

多机要点（实践常踩坑）：
- **endpoint 一致**：所有节点 `--rdzv-endpoint` 填同一个地址（通常 node0），且该端口在 node0 上未被占、防火墙放行。
- **`--rdzv-id` 一致**：不一致会被当成两个不同作业，永远会合不上而 hang。
- **网络互通**：跨机靠 NCCL，需配置正确网卡。常用环境变量 `NCCL_SOCKET_IFNAME`（指定网卡）、`NCCL_IB_DISABLE`（是否禁 InfiniBand）；具体取值以集群环境/官方文档为准。
- **`set_device(LOCAL_RANK)`**：多机绝不能用 RANK 绑卡（会越界），必须用 LOCAL_RANK。

---

## 7. 故障重启与弹性伸缩

这是 torchrun 相对旧 launch 的"杀手锏"。

### 7.1 弹性节点范围

把 `--nnodes` 写成区间 `min:max`，作业就具备弹性：

```
   --nnodes=2:4     # 最少 2 节点即可开跑, 最多接纳 4 节点
   --nproc-per-node=8
```

- 凑齐 `min`（2）个节点就开始训练，不必死等全到。
- 后到的节点会在**下一轮 rendezvous** 时加入（worker 重启重分 RANK）。
- 某节点掉线后只要存活数 ≥ min，作业继续（重组后跑）。

### 7.2 重启语义与生命周期

```
   正常运行 ──[worker 崩溃/节点掉线]──► agent 检测到失败
        │
        ▼
   触发新一轮 rendezvous(重新统计成员、重分 RANK)
        │
        ├─► 未超 max_restarts: 重启整组 worker ──► 从 checkpoint 续训
        └─► 已达 max_restarts: 作业失败退出
```

| 参数 | 作用 | 权衡 |
|------|------|------|
| `--max-restarts` | 允许的最大重启次数 | 太小：偶发故障直接失败；太大：坏节点反复拖累 |
| `--nnodes=min:max` | 弹性节点范围 | min 越小越快开跑但初始算力少；max 给扩容上限 |
| `--monitor-interval` | agent 巡检 worker 的间隔 | 越短发现故障越快，开销略增 |

### 7.3 重启 ≠ 不丢进度：必须配 checkpoint

**关键认知**：torchrun 重启的是"进程"，**不是你的内存状态**。重启后整组 worker 从头执行 `train.py`，内存里的模型权重、优化器状态、当前 step 全部丢失。所以**弹性容错必须搭配 checkpoint**：

```
   训练循环里:
     每隔 N 步 / 每个 epoch:
        if RANK == 0:  保存 {model, optimizer, step} 到共享存储
   重启后 train.py 开头:
        if 存在 checkpoint:  加载并从该 step 续训
```

没有 checkpoint 的"弹性"是假弹性——崩了重启等于从 step 0 重来。**torchrun 负责把进程拉回来，把进度接上是你脚本的责任。**

---

## 8. 训练脚本最小骨架（读 env → 组网 → 绑卡）

torchrun 只管启动；脚本要做的就是**读它注入的环境变量并初始化通信组**。最小骨架（讲思路，API 以官方文档为准）：

```python
import os, torch
import torch.distributed as dist

def main():
    # 1) 读身份: torchrun 已注入这些 env, 不用自己 export
    local_rank = int(os.environ["LOCAL_RANK"])   # 机内编号, 用于绑卡
    rank       = int(os.environ["RANK"])         # 全局编号, 用于判主进程
    world_size = int(os.environ["WORLD_SIZE"])   # 全局进程数

    # 2) 绑卡: 必须用 LOCAL_RANK, 不能用 RANK
    torch.cuda.set_device(local_rank)

    # 3) 组网: 读 env 完成 rendezvous(MASTER_ADDR/PORT 由 torchrun 设好)
    dist.init_process_group(backend="nccl")      # GPU 用 nccl, CPU 用 gloo

    # 4) DDP 包模型
    model = MyModel().cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank])

    # 5) 只有主进程做日志/存盘, 避免 8 个进程抢着写
    if rank == 0:
        print(f"start training, world_size={world_size}")

    # ... 训练循环 + 周期性 checkpoint(为弹性重启续训) ...

    dist.destroy_process_group()                 # 收尾, 释放通信组

if __name__ == "__main__":
    main()
```

调用链全貌：`torchrun ──拉进程──► train.py(×N) ──读env──► init_process_group ──► NCCL 建组 ──► DDP all-reduce 同步梯度`，崩溃时 torchrun 监控并重启整组。

> 迁移提示：从 `torch.distributed.launch` 迁来时，删掉 `argparse` 里的 `--local_rank`，改成 `int(os.environ["LOCAL_RANK"])`，这是最高频的改动点。脚本骨架细节参见 [[llm-train/pytorch/distribution/README]]。

---

## 配置示例：关键参数逐个解释

把一条多机弹性命令拆开看每个参数为什么这么填：

```
   torchrun \
     --nnodes=2:4 \                  # 弹性: 2 起跑, 最多 4 节点
     --nproc-per-node=8 \            # 每机 8 进程 = 8 GPU
     --rdzv-id=run-2026-0621 \       # 作业唯一 ID, 各节点必须一致
     --rdzv-backend=c10d \           # 内置会合后端, 免外部依赖
     --rdzv-endpoint=10.0.0.1:29500 \# 会合点, 各节点填同一个(指向 node0)
     --max-restarts=3 \              # 最多重启 3 次, 超过则作业失败
     train.py \                      # 你的训练脚本
       --epochs 100 --lr 1e-4        # 你脚本自己的参数, 原样透传
```

| 参数 | 一句话 | 填错的后果 |
|------|--------|-----------|
| `--nnodes` | 节点数（或 min:max 区间） | 写死且与实际不符 → 永远会合不齐而 hang |
| `--nproc-per-node` | 每节点进程数 | 大于 GPU 数 → 多进程抢同卡或越界 |
| `--rdzv-id` | 作业 ID | 各节点不一致 → 串成两个作业，hang |
| `--rdzv-endpoint` | 会合点 host:port | 各节点不一致 / 端口被占 / 防火墙挡 → hang |
| `--max-restarts` | 最大重启次数 | 太小偶发故障即失败；太大坏节点反复拖 |
| `train.py 后的参数` | 透传给脚本 | 与 torchrun 参数顺序写反会被吞 |

> 注意：torchrun 自己的参数要写在脚本名**之前**；脚本名**之后**的参数原样传给 `train.py`。这条顺序规则最易踩坑。

---

## 常见问题

| 现象 | 可能原因 | 排查思路 |
|------|---------|---------|
| 多机启动后一直 hang，不报错 | `--rdzv-id` 或 `--rdzv-endpoint` 各节点不一致 / 没全到齐 | 核对所有节点这两个参数完全相同；确认到达节点数 ≥ min_nodes |
| `Address already in use` | rendezvous 端口被上次残留进程占用 | 换端口，或清理 `MASTER_PORT` 对应残留进程 |
| `CUDA error: invalid device ordinal` | 用了 `RANK` 而非 `LOCAL_RANK` 去 `set_device`，多机时越界 | 绑卡一律用 `LOCAL_RANK` |
| NCCL 初始化超时 / 跨机不通 | 网卡选错、IB 配置、防火墙挡了通信端口 | 设 `NCCL_SOCKET_IFNAME` 指定网卡；检查节点间 IP 互通与端口放行 |
| 进程崩溃后没有自动恢复 | 没设 `--max-restarts` 或脚本没接 checkpoint | 设 `--max-restarts`；训练循环里周期存盘、开头自动续训 |
| 重启后从 step 0 重来 | 误以为 torchrun 会保存内存状态 | torchrun 只重拉进程；进度恢复靠**你的** checkpoint 逻辑 |
| 日志被打印 8 遍 | 每个进程都在 print | 用 `if RANK == 0:` 包住日志/存盘 |
| 从 launch 迁移后报 `--local_rank` 相关错 | 脚本仍在 argparse 解析 `--local_rank` | 改为 `int(os.environ["LOCAL_RANK"])` |

> 一句话排障心法：**多机 hang 八成是"会合参数不一致或没到齐"，CUDA 越界八成是"拿 RANK 当 LOCAL_RANK 绑卡"。**

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[llm-train/pytorch/distribution/多机多卡]] — 多机多卡完整组网与示例
- [[llm-train/pytorch/distribution/README]] — 分布式训练总览与脚本骨架
- 官方文档：https://pytorch.org/docs/stable/elastic/run.html
