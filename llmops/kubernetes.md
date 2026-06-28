# Kubernetes for AI

> 用容器编排把"一堆 GPU 机器"变成"一个可声明、可自愈、可弹性的算力池"，统一承接 LLM 的训练与推理。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llmops/README]] [[llmops/模型推理平台方案]] [[ai-infra/ai-cluster/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | 声明式 / 调度 / 自愈 |
| 1 | 地基：容器、节点、Pod | container / kubelet / Pod |
| 2 | 为什么用 K8s（对比裸机 + Slurm） | 弹性 / 多租户 / 自愈 |
| 3 | 控制面 vs 数据面架构 | api-server / scheduler / etcd |
| 4 | Pod / Deployment / Service 三件套 | 副本 / 滚动 / 负载均衡 |
| 5 | GPU 怎么被调度（Device Plugin） | nvidia-device-plugin / Extended Resource |
| 6 | GPU 共享：Time-Slicing / MPS / MIG | 切分 / 隔离 / 利用率 |
| 7 | 训练：Volcano / Kubeflow | gang 调度 / PyTorchJob |
| 8 | 推理：KServe | Serverless / 自动扩缩 / scale-to-zero |
| 9 | 数值例子 + 实践要点 | shm / 拓扑 / 碎片 |
| QA | 常见坑表格 | OOM / Pending / NCCL |

## 0. 一句话锚点

Kubernetes（K8s）= **一个"集群操作系统"**。你不再说"在 node-7 上启动进程"，而是写一份 **声明式 YAML**："我要 8 个副本、每个要 1 张 A100、挂这个数据盘"。K8s 的**调度器**负责把它塞到合适的机器上，**控制器**负责让"实际状态"持续逼近你"声明的期望状态"——挂了就重启，机器宕了就迁移。对 AI 而言，最难的一块是让 K8s **认识 GPU、会切 GPU、会成组调度多机训练**。

## 1. 地基 / 前置（把概念拆到原子）

- **容器（Container）**：把"代码 + 依赖 + CUDA/cuDNN 运行时"打包成一个隔离进程，靠 Linux 的 namespace（隔离视图）+ cgroup（限额）实现。注意：容器**共享宿主机内核与 GPU 驱动**，所以镜像里只装 CUDA *运行时*，驱动在宿主机（这就是为什么换驱动要重启机器而非重建镜像）。
- **节点（Node）**：一台物理/虚拟机。上面跑 `kubelet`（节点代理，干活的）和容器运行时（containerd）。GPU 机器还要装 NVIDIA 驱动 + `nvidia-container-toolkit`（让容器能看到 `/dev/nvidia*`）。
- **Pod**：K8s **最小调度单位**，= 1 个或多个**共享网络与存储**的容器。一个 Pod 内的容器共享同一个 IP、同一块 `localhost`、可共享 Volume。**训练任务通常 1 Pod = 1 进程组（1 个或多张 GPU）**。
- **期望状态（desired state）vs 实际状态**：K8s 的灵魂。你只描述"想要什么"，**控制循环（reconcile loop）**不断对比并纠偏。这就是"自愈"的来源。

```
┌──────────────── Kubernetes 集群 ────────────────┐
│  控制面(Control Plane)        数据面(Worker Nodes) │
│  ┌───────────┐               ┌──────────────────┐ │
│  │ api-server│◄──kubectl     │ Node (GPU 机)     │ │
│  │ scheduler │               │  kubelet          │ │
│  │ etcd(状态)│               │  containerd       │ │
│  │ controllers│              │  ┌─Pod─┐ ┌─Pod─┐  │ │
│  └───────────┘               │  │GPUx1│ │GPUx1│  │ │
│        ▲                     │  └─────┘ └─────┘  │ │
│        └─────调度/汇报────────┤  nvidia-device-plugin│
└─────────────────────────────────────────────────┘
```

## 2. 为什么 AI 要用 K8s？

| 维度 | 裸机 / 手动 SSH | Slurm（HPC 传统） | Kubernetes |
|---|---|---|---|
| 部署形态 | 进程 | 批作业 | 容器（依赖自带，环境一致）|
| 自愈 | 无，挂了人工拉 | 有限 | 强（控制器自动重建/迁移）|
| 弹性扩缩 | 手动 | 弱 | 强（HPA / Cluster Autoscaler）|
| 推理服务 | 自己写 nginx | 不擅长 | 原生 Service/Ingress/扩缩 |
| 多租户隔离 | 难 | 有 | namespace + quota + RBAC |
| 训练成组调度 | 无 | **强（gang）** | 需 Volcano/Kueue 补强 |

**一句话权衡**：纯大规模训练，Slurm 的 gang/拓扑调度更成熟；但**"训练 + 推理 + CI + 数据处理"混合云原生平台**，K8s 是事实标准——所以业界用 **Volcano/Kueue 给 K8s 补上 gang 调度**，鱼和熊掌兼得。

## 3. 控制面 vs 数据面（一次请求的旅程）

```
你: kubectl apply -f deploy.yaml
      │
      ▼
[api-server] 校验 + 写入 etcd(期望状态)
      │
      ▼
[scheduler] 看哪些 Node 有空闲 GPU/内存 → 绑定 Pod→Node
      │
      ▼
[kubelet@Node] 收到绑定 → 调 containerd 拉镜像、起容器
      │
      ▼
[device-plugin] 把 GPU 注入容器(设置 NVIDIA_VISIBLE_DEVICES)
      │
      ▼
Pod Running，状态回写 etcd；controller 持续比对纠偏
```

- **api-server**：唯一入口，所有组件只跟它说话（不互相直连）。
- **etcd**：分布式 KV，**集群唯一事实来源**。挂了 = 失忆。
- **scheduler**：打分 + 过滤，决定 Pod 落哪台。GPU 调度的关键扩展点就在这里。
- **controller-manager**：跑各种 reconcile loop（Deployment、ReplicaSet…）。

## 4. Pod / Deployment / Service 三件套

**为什么不直接用 Pod？** Pod 是"一次性"的——挂了不会自己回来。你需要上层控制器管理"一组 Pod 的生命周期"。

- **Deployment**：管理**无状态**副本集。声明 `replicas: 3`，控制器保证永远有 3 个健康 Pod；改镜像版本 → **滚动更新（rolling update）**：起新的、健康后再杀旧的，零停机。适合**推理服务**。
- **StatefulSet**：有稳定网络名 + 持久存储 + 有序启停。适合需要固定 rank/主机名的场景。
- **Service**：给一组会漂移的 Pod 一个**稳定虚拟 IP + DNS 名**，并做负载均衡。Pod 重建后 IP 变了，Service 名不变——**解耦"谁调用"和"谁实现"**。

```
        ┌──────── Service: llm-infer (ClusterIP 10.96.0.7) ────────┐
请求 ──►│              (稳定入口 + 负载均衡)                          │
        └───┬───────────────┬───────────────┬──────────────────────┘
            ▼               ▼               ▼
        Pod(A100)       Pod(A100)       Pod(A100)   ← Deployment 管理 replicas=3
        某个挂了 ✗ ──► 控制器立刻拉起一个新的，Service 自动把它纳入后端
```

最小推理 Deployment（节选，重点看 GPU 资源声明）：

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 1          # ← 申请 1 张整卡（见第 5 节）
        volumeMounts:
        - { name: dshm, mountPath: /dev/shm }   # ← 多进程/张量并行靠 shm 通信
      volumes:
      - name: dshm
        emptyDir: { medium: Memory, sizeLimit: 16Gi }  # 默认 shm 仅 64MB，会 OOM
```

> 实践锚点：PyTorch DataLoader 多 worker、NCCL、TP 都用 `/dev/shm`。容器默认 `/dev/shm` 只有 **64MB**，跑大模型必报错。用上面的 `emptyDir{medium:Memory}` 把它撑大（参考阿里云 ECI 修改 shm 文档）。

## 5. GPU 怎么被调度——Device Plugin 机制

**核心问题**：K8s 原生只懂 CPU（核）和 Memory（字节），**根本不认识 GPU**。靠 **Device Plugin（设备插件）框架**把 GPU 注册成一种"扩展资源（Extended Resource）"。

机制（NVIDIA Device Plugin 以 DaemonSet 形式跑在每个 GPU 节点）：

1. 插件扫描本机 GPU，向 kubelet **注册资源名** `nvidia.com/gpu` 与数量（如 8）。
2. kubelet 把这个"可分配资源"上报给 scheduler。
3. 你在 Pod 写 `limits: nvidia.com/gpu: 2`。**注意：GPU 只能整数、不可超分（不像 CPU 能 `0.5`），且 request 必须等于 limit。**
4. scheduler 找到剩余 GPU≥2 的节点，绑定。
5. kubelet 起容器前回调插件 `Allocate`，插件设置 `NVIDIA_VISIBLE_DEVICES=GPU-uuid...`，把对应 `/dev/nvidia*` 注入容器。

```
┌ GPU Node (8×A100) ┐        scheduler 视角的"账本"
│ nvidia-device-    │  ──注册──►  nvidia.com/gpu: 8 (capacity)
│  plugin (DaemonSet)│            已分配 5, 可用 3
│  GPU0 GPU1 ... GPU7│            ← Pod 申请 2 → 落这里(够)
└───────────────────┘            ← Pod 申请 4 → 别的节点(此处不够)
```

**关键含义与权衡**：
- 默认是**整卡独占**——利用率天然偏低（一个只用 30% 显存的推理服务也占满一张卡）。这正是第 6 节"共享"要解决的痛点。
- **GPU 拓扑**：8 卡机内部 NVLink/NVSwitch 互联不均匀。默认调度不感知拓扑，可能把同一训练任务的 2 张卡分到 NVLink 不互通的位置 → AllReduce 变慢。需要 **NVIDIA GPU Operator** + 拓扑感知调度来优化（以官方文档为准）。
- **GPU Operator**：一键管理驱动、device-plugin、DCGM 监控、MIG 配置、容器 toolkit 的"全家桶"，省去手动在每台机器装驱动的痛苦。

## 6. GPU 共享：Time-Slicing / MPS / MIG

整卡独占太浪费。三种共享方式，**隔离强度与适用场景递增**：

| 方式 | 原理（拆到底层） | 显存隔离 | 算力隔离 | 故障隔离 | 适用 |
|---|---|---|---|---|---|
| **Time-Slicing** | 多个容器共享 1 张卡，GPU 时间片轮转（像 CPU 分时）| ❌ 不隔离，会互相 OOM | ❌ 抢占 | ❌ | 开发/测试、小推理、超卖 |
| **MPS** | CUDA Multi-Process Service，多进程**并发**共享 SM（非轮转）| 软隔离(可限百分比) | 部分 | ❌ 一个崩可能连累 | 多个小 kernel 并发、提升占用率 |
| **MIG** | A100/H100 硬件把 1 卡**物理切分**成最多 7 个实例(独立 SM+L2+显存通道) | ✅ 硬隔离 | ✅ 硬隔离 | ✅ 互不影响 | 生产多租户推理、稳定 SLA |

**MIG（Multi-Instance GPU）几何拆解**：A100-40GB 切成 `1g.5gb`（1 份计算 + 5GB 显存）等规格。`Ng.Mgb` 中 N=GPU 切片数、M=显存 GB。常见 profile：`1g.5gb / 2g.10gb / 3g.20gb / 7g.40gb`。

```
A100 整卡 (7 个 GPC, 40GB)
┌───────────────────────────────────────┐
│ ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐           │  ← 7× 1g.5gb（7 个独立小卡）
│ └──┘└──┘└──┘└──┘└──┘└──┘└──┘           │
└───────────────────────────────────────┘
        或混合切：
┌─────────────┬──────────┬──────┬──────┐
│   3g.20gb   │  2g.10gb │1g.5gb│1g.5gb│  ← 不同租户/不同模型各拿一块
└─────────────┴──────────┴──────┴──────┘
device-plugin 把每个实例当成独立资源:
  nvidia.com/mig-1g.5gb: 7   或   nvidia.com/mig-3g.20gb: 1 ...
```

**权衡口诀**：要利用率不要隔离 → Time-Slicing；要并发吞吐 → MPS；要生产级强隔离与稳定延迟 → MIG（但切分粒度固定、需 A100/H100、切完不能动态改）。

## 7. 训练编排：Volcano / Kubeflow

**核心痛点：成组（gang）调度**。一个 4 机 32 卡训练，**要么 32 张卡同时就位、要么一张别给**。原生 K8s 调度器是**逐 Pod**调度的——可能给你 28 张卡然后卡住，剩 4 个 Pod 永远 Pending，而那 28 张卡被占着空转 = **死锁 + 浪费**。

- **Volcano**：批调度器，补上 K8s 缺的能力：
  - **Gang Scheduling**：要么全部 Pod 一起调度成功，要么都不调度（all-or-nothing），避免资源死锁。
  - **队列 + 公平共享 + 优先级抢占**：多团队抢 GPU 时按配额公平分。
  - **拓扑/亲和**：尽量把同一 Job 的 Pod 放 NVLink 互通的位置。
- **Kubeflow Training Operator**：提供 `PyTorchJob`/`TFJob` 等**自定义资源（CRD）**。你声明"1 master + 3 worker"，Operator 自动注入 `MASTER_ADDR`、`WORLD_SIZE`、`RANK` 等分布式环境变量并管理生命周期，省去手写 rendezvous。**常与 Volcano 配合**（Kubeflow 负责"长什么样"，Volcano 负责"怎么成组调度上去"）。

```
PyTorchJob(Kubeflow CRD)              Volcano(gang)
  worker0 (RANK0,MASTER)  ┐
  worker1 (RANK1)         ├─► PodGroup ─► 调度器: 4 张卡全有? 
  worker2 (RANK2)         │              有→一起起；缺→全等待(不空占)
  worker3 (RANK3)         ┘
        └ Operator 自动注入 WORLD_SIZE=4, MASTER_ADDR=worker0
```

## 8. 推理服务：KServe

推理和训练需求相反：**训练是"批作业、跑完就退"；推理是"长期在线服务、要弹性、要灰度"**。KServe（前身 KFServing）是 K8s 上的**模型推理标准平台**：

- **InferenceService（CRD）**：你只写"模型在哪、用什么 runtime、要几张卡"，KServe 拉起带标准 `/v1/models/.../predict` 接口的服务。
- **自动扩缩 + Scale-to-Zero**：基于 QPS/并发自动加减副本；**没流量时缩到 0**（省 GPU 钱），来请求再冷启动（代价是首请求延迟）。
- **金丝雀发布（Canary）**：新模型先导 10% 流量验证，稳了再 100%——风险可控。
- **多框架**：内置 vLLM、TGI、Triton、sklearn 等 runtime，开箱即用。

```
请求 ─► Ingress ─► KServe Router
                     │  90% ──► Predictor v1 (Deployment, GPU×N)
                     │  10% ──► Predictor v2 (canary)   ← 灰度
                     └─ 无流量时 v1 副本=0(Scale-to-Zero)
   并发↑ ──► autoscaler 自动 +副本；并发↓ ──► 回收
```

> 关于平台选型（KServe vs Triton vs 自建 vLLM 网关）的横评，见 [[llmops/模型推理平台方案]]。

## 9. 数值例子 / 对照 / 实践要点

**例 1：GPU 利用率提升（为什么共享值得做）**
一个 7B 模型推理服务峰值只用约 14GB 显存、平均 GPU 利用率约 25%。
- 整卡独占 A100-40GB：1 服务 = 1 卡，**显存浪费 ≈ (40−14)/40 ≈ 65%**。
- MIG 切成 `2g.10gb`（10GB）刚好不够，用 `3g.20gb`（20GB）：1 张 A100 → 2 个实例 ≈ **同卡承载 2 个服务，利用率翻倍**。

**例 2：成组调度避免空占（Volcano 价值的量化）**
4 机 32 卡训练，逐 Pod 调度时拿到 28 卡卡住 30 分钟才放弃。
浪费 = $28 \text{ 卡} \times 0.5 \text{ h} = 14$ 卡·时；按 A100 约 \$2/卡·时（公开租赁价，约/以实际为准）≈ **\$28 白烧**，且这 30 分钟别人也用不了这 28 张卡。Gang 调度直接消灭这种死锁。

**例 3：shm 与碎片（落到 Pod 配置）**
- 多机 NCCL/TP 通信走 `/dev/shm`；默认 64MB → 张量并行直接 OOM。设 `emptyDir{medium:Memory, sizeLimit:16Gi}`。注意：**shm 用的是内存（tmpfs），不是磁盘 swap**——tmpfs 占内存空间，swap 占物理存储，二者别混。
- 训练显存碎片导致 `CUDA out of memory`（明明 reserved 还有空但分不出大块）时：`export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`，阻止分配器切分大于该值的块以减少碎片（仅对 native 后端有效，cudaMallocAsync 后端会忽略；性能代价视分配模式而定，万不得已再用）。以 [PyTorch CUDA 语义官方文档](https://pytorch.org/docs/stable/notes/cuda.html) 为准。

**例 4：拓扑感知的通信量直觉**
8 卡 AllReduce 一次梯度（设模型 14GB fp16 梯度）。NVLink 双向带宽约 600 GB/s、PCIe 约 64 GB/s（典型公开数字，约/以官方为准）。
- 同机 NVLink：$14 / 600 \approx 0.023$ s 量级。
- 误调度到 PCIe 路径：$14 / 64 \approx 0.22$ s，**慢约 10 倍**。所以"拓扑感知调度"对训练吞吐是实打实的钱。

**实践清单**：
- GPU Pod 必须 `request==limit`（GPU 不可超分），且为整数。
- 给 GPU 节点打 **taint**（污点），让普通 Pod 别误占 GPU 机；GPU Pod 加 **toleration** 才进得来。
- 生产推理用 **readiness probe**（模型加载完才接流量，避免冷的副本被打挂）。
- 监控用 **DCGM Exporter + Prometheus**，盯 GPU 利用率/显存/温度/ECC 错误。
- 多机训练优先 Volcano gang + 拓扑亲和，别裸 Deployment。

## 常见问题

| 现象 | 多半原因 | 处理方向 |
|---|---|---|
| Pod 一直 `Pending` | 没有节点满足 `nvidia.com/gpu` 申请；或 taint 没容忍 | `kubectl describe pod` 看 Events；加 toleration / 等扩容 |
| `0/N nodes available: insufficient nvidia.com/gpu` | GPU 已被占满或 device-plugin 没起 | 查 device-plugin DaemonSet；MIG 模式下资源名要对（`mig-3g.20gb`）|
| 容器看不到 GPU | 没装 nvidia-container-toolkit / `NVIDIA_VISIBLE_DEVICES` 没注入 | 装 GPU Operator；检查 runtimeClass |
| 多 worker DataLoader / NCCL 报 shm 错 | `/dev/shm` 默认仅 64MB | 挂 `emptyDir{medium:Memory}` 扩容 |
| `CUDA out of memory` 但 reserved 还有空 | 显存碎片 | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`（native 后端）|
| 多机训练 28/32 卡卡死 | 原生逐 Pod 调度，无 gang | 上 Volcano，开 gang scheduling |
| 共享卡互相 OOM/抢算力 | 用了 Time-Slicing（无隔离） | 换 MIG 做硬隔离 |
| 推理冷启动首请求很慢 | KServe Scale-to-Zero 冷启动 | 设最小副本=1，或接受延迟换成本 |
| 滚动更新瞬间 5xx | 新副本没就绪就接流量 | 配 readiness probe + `maxUnavailable` 调小 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局知识地图入口
- [[llmops/README]] — LLMOps 模块总览
- [[llmops/模型推理平台方案]] — KServe/Triton/自建网关横向对比
- [[ai-infra/ai-cluster/README]] — GPU 集群硬件/网络/拓扑底座
- PyTorch CUDA 语义（显存/碎片官方说明）：https://pytorch.org/docs/stable/notes/cuda.html
- 修改 Pod 共享内存（shm）：阿里云 ECI emptyDir Memory 文档
- swap 与 shm 区别：tmpfs 用内存、swap 用物理存储
