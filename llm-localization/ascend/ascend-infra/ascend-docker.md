# 昇腾 Docker 容器化部署(Ascend Docker)

> 让昇腾 NPU 像 GPU 那样"装进容器":把 NPU 设备、驱动用户态库、CANN 运行时一起挂进容器,实现可移植、可复现的训练/推理环境。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[llm-inference/README]] [[llm-train/README]]

## 阅读地图(表格)

| 节 | 你会得到什么 | 关键词 |
|----|-------------|--------|
| 0 | 一句话锚点 | NPU 入容器 |
| 1 | 在昇腾栈的定位 + 昇腾↔英伟达对照 | Ascend Docker ↔ nvidia-docker |
| 2 | 容器里到底要挂什么(设备/驱动/CANN 三件套) | device passthrough |
| 3 | 镜像分层:基础镜像 → CANN → 框架 → 套件 | 镜像金字塔 |
| 4 | 两条路:Ascend Docker Runtime vs 手动 --device | 自动注入 vs 手动挂载 |
| 5 | 部署全流程(步骤的含义,不是抄命令) | login/pull/run |
| 6 | 迁移要点:从 nvidia-docker 心智迁过来 | 迁移坑 |
| 7 | 常见坑 | 设备不可见/驱动版本错配 |
| 8 | 常见问题(表格) | FAQ |

> 关联文件:本目录另有 [[ascend-docker-runtime]](Runtime 注入机制细节)、[[昇腾镜像]]、[[docker环境升级cann]],本文是这些主题的"总览与心智图"。

## 0. 一句话锚点

**昇腾 Docker = 把 NPU 设备节点 + 驱动用户态库 + CANN 运行时,正确地"喂"进一个 Linux 容器。** 容器本身还是普通的 Docker;特殊性全在于"NPU 是一块带专用驱动的加速卡,容器默认看不见它",所以核心问题始终是:**怎样让容器内的进程访问到宿主机的 NPU**。这与英伟达世界里 nvidia-docker 解决的问题一一对应。

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

### 1.1 它在昇腾软件栈的哪一层

```
┌──────────────────────────────────────────────┐
│  应用 / 套件:MindFormers · MindIE · ModelLink │  ← 容器里跑的"业务"
├──────────────────────────────────────────────┤
│  框架:MindSpore / PyTorch(torch_npu)         │  ← 装进镜像
├──────────────────────────────────────────────┤
│  CANN(算子库 + 运行时 ACL + HCCL)            │  ← 装进镜像 / 或挂载
├══════════════════════════════════════════════┤
│  ★ 容器边界(Docker)★                         │  ← 昇腾 Docker 在这里
├──────────────────────────────────────────────┤
│  NPU 驱动(Driver)+ 固件(Firmware)          │  ← 留在宿主机,不进容器
├──────────────────────────────────────────────┤
│  硬件:昇腾 NPU(910/310 等,达芬奇架构)       │
└──────────────────────────────────────────────┘
```

关键分界线:**驱动/固件装在宿主机,CANN 及以上装在容器**。容器要做的,是把宿主机驱动暴露出来的设备节点和用户态 `.so` 库"借"进来,让容器内的 CANN 能调到真实硬件。这条分层原则和 nvidia-docker 完全同构(宿主机装 GPU driver,容器装 CUDA toolkit)。

### 1.2 昇腾 ↔ 英伟达生态对照表(迁移心智图)

| 维度 | 英伟达世界 | 昇腾世界 | 说明 |
|------|-----------|----------|------|
| 加速器 | GPU | NPU(达芬奇架构) | 见 [[达芬奇架构]] |
| 计算平台/工具链 | CUDA / CUDA Toolkit | CANN | 算子库 + 运行时 |
| 运行时 API | CUDA Runtime / Driver API | ACL(AscendCL) | 容器内调用入口 |
| 算子库 | cuDNN / cuBLAS | CANN 算子库(AOE/aclnn) | 见 [[ai-infra/ai-hardware/AI芯片软件生态]] |
| 集合通信 | NCCL | HCCL | 见 [[HCCL]]、[[ai-infra/网络/NCCL]] |
| 训练框架 | PyTorch / Megatron-LM | MindSpore / torch_npu / ModelLink | |
| 推理引擎 | TensorRT-LLM / vLLM | MindIE | |
| **容器运行时** | **nvidia-container-runtime(nvidia-docker)** | **Ascend Docker Runtime / Ascend-Docker** | 自动注入设备与库 |
| 设备发现/钩子 | nvidia-container-toolkit + libnvidia-container | ascend-docker-runtime 的 prestart hook | 见 [[ascend-docker-runtime]] |
| 设备节点 | `/dev/nvidia*` | `/dev/davinci*`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` | 多个节点协作 |
| 监控工具 | nvidia-smi | npu-smi(见 [[ascend-npu-smi]]) | |
| 镜像仓库 | NGC(nvcr.io) | Ascend Hub(ascendhub.huawei.com) | 见 [[昇腾镜像]] |
| 资源调度插件 | NVIDIA Device Plugin(k8s) | Ascend Device Plugin | k8s 上调度 NPU |

> 一句话记忆:**"nvidia-docker 之于 GPU,Ascend Docker Runtime 之于 NPU"**。你过去在 GPU 上用 `--gpus all`,在 NPU 上对应的是"由 Ascend Docker Runtime 自动注入"或"手动 `--device` 挂多个 davinci 节点"。

## 2. 容器里到底要挂什么:NPU 入容器的"三件套"

要让容器跑起 NPU 业务,本质上要把三类东西从宿主机送进容器:

```
宿主机                                 容器内
─────────────────────────────────     ─────────────────────────
① 设备节点 /dev/davinci0..N      ──►   进程能"看见"这块卡
   /dev/davinci_manager                 (设备管理)
   /dev/devmm_svm                        (共享虚拟内存)
   /dev/hisi_hdc                         (主机-设备通信)
② 驱动用户态库 (driver/lib64 下     ──►  CANN 运行时能调到驱动
   的 libdrv*.so、libascend* 等)         (ABI 必须与宿主驱动匹配)
③ 工具与配置 (npu-smi、              ──►  容器内可查卡、可做亲和
   /etc/ascend_install.info 等)          性绑定
─────────────────────────────────     ─────────────────────────
④ CANN / 框架 / 套件 通常直接打进镜像,不从宿主机挂
```

要点:
- **缺一不可**:只挂 `/dev/davinci0` 而漏了 `davinci_manager` / `devmm_svm` / `hisi_hdc`,会出现"卡能看到但初始化失败"的诡异错误。这是手动挂载方式最常见的坑。
- **②号"驱动用户态库"必须与宿主机驱动版本一致**:容器内的 CANN 调用的是宿主机的驱动 `.so`(通过挂载共享),不是容器自带的。所以**驱动留宿主机、库靠挂载**,从根上避免容器内外驱动错配。
- 三件套到底"怎么挂",有两条路(见第 4 节):自动注入(推荐)或手动 `--device`/`-v`。

## 3. 镜像分层:基础镜像 → CANN → 框架 → 套件

昇腾官方镜像(Ascend Hub)和你自建镜像,通常都是一座"金字塔":

```
        ┌────────────────────────┐
  顶层  │  套件镜像               │  MindFormers / MindIE / ModelLink 预装
        ├────────────────────────┤
        │  框架镜像               │  ascend-pytorch(含 torch_npu)
        │                        │  ascend-mindspore
        ├────────────────────────┤
        │  CANN 镜像              │  预装 toolkit + 算子库 + HCCL(无驱动)
        ├────────────────────────┤
  基座  │  OS 基础镜像            │  Ubuntu / openEuler / CentOS + 基础依赖
        └────────────────────────┘
   (驱动/固件不在镜像里 —— 它在宿主机)
```

- **越往上,镜像越"开箱即用",但也越大、越锁死版本**;越往下越灵活,但你要自己装框架/套件。
- **选镜像第一原则:CANN 版本要与宿主机驱动版本匹配**(配套关系以官方"驱动/固件 ↔ CANN 配套表"为准)。镜像里没有驱动,但镜像里 CANN 的版本必须落在宿主机驱动支持的区间内。
- 你看到的文件顶部那条 `docker pull ascendhub.huawei.com/...ascend-mindspore...`,就是从 Ascend Hub 拉一个"OS + CANN + MindSpore"的框架镜像。**具体镜像 tag / 版本号以华为昇腾官方文档(Ascend 社区)为准**,不要照抄某个固定 tag。
- 升级 CANN 时,既可以"换更高层的镜像",也可以"在现有容器/镜像里单独升 CANN"——后者见 [[docker环境升级cann]]。

## 4. 两条路:Ascend Docker Runtime(推荐) vs 手动挂载

### 4.1 自动注入:Ascend Docker Runtime

类比 nvidia-docker。安装 Ascend Docker Runtime 后,它向 Docker 注册一个 runtime,并挂上一个 **prestart hook**:容器启动前,hook 自动把第 2 节的"三件套"注入容器。

```
docker run ... (指定 ascend runtime / 或设为默认 runtime)
        │
        ▼  容器创建,prestart hook 触发
   ┌─────────────────────────────────────┐
   │  hook 读取:你声明要哪几张卡          │
   │   → 注入对应 /dev/davinci* 设备       │
   │   → 注入 davinci_manager/devmm_svm…   │
   │   → 挂载宿主机驱动用户态库            │
   │   → 注入 npu-smi 等工具               │
   └─────────────────────────────────────┘
        │
        ▼
   容器内 npu-smi 可见卡,CANN 可正常初始化
```

优点:**不用记一长串 `--device`,不会漏挂节点**;声明"要哪些卡"即可(常通过环境变量 `ASCEND_VISIBLE_DEVICES` 之类指定,语义类比 `NVIDIA_VISIBLE_DEVICES`)。注入细节见 [[ascend-docker-runtime]]。

> 具体安装步骤、runtime 名称、环境变量精确写法,以华为昇腾官方文档(Ascend 社区)为准。

### 4.2 手动挂载:`--device` + `-v`

不装 runtime 时,得自己把三件套用 `--device`(设备节点)和 `-v`(驱动库目录、工具、配置文件)逐项挂进去。

- 优点:无额外组件,环境最"纯";
- 缺点:**极易漏挂节点 / 漏挂某个 `.so` / 路径写错**,这是新手最高频的坑。能用 runtime 就别手动。
- 手动方式里还要注意:容器内进程能否访问设备,受 cgroup device 规则与容器权限影响,有时需放开设备访问权限——但"无脑 `--privileged`"是反模式(见第 6 节)。

## 5. 部署全流程(讲"每步为什么",不是抄命令)

> 命令仅示意"动作的含义",**精确命令/包名/版本号一律以华为昇腾官方文档(Ascend 社区)为准**。

```
[宿主机准备] 装 NPU 驱动+固件 ──► 装 Docker ──► (推荐)装 Ascend Docker Runtime
      │                              │                    │
      ▼                              ▼                    ▼
 npu-smi 能看到卡            docker info 正常        注册 ascend runtime
      │
      ▼
[拉镜像] 登录 Ascend Hub ──► pull 对应 CANN/框架镜像(版本要配宿主驱动)
      │
      ▼
[起容器] docker run:
      ├─ 用 ascend runtime,声明可见卡(自动注入三件套)
      ├─ 或手动 --device 挂 davinci* + -v 挂驱动库
      ├─ 挂业务数据/权重目录(-v),设 shm-size(大模型多进程要够大)
      └─ 网络:多机训练用 host 网络,保证 HCCL/RDMA 直连(见 network)
      │
      ▼
[容器内自检] npu-smi 看卡 ──► python 里 import 框架做一次 NPU 张量运算
      │
      ▼
[跑业务] 训练(MindFormers/ModelLink)或 推理(MindIE)
```

每步的"为什么":
1. **驱动装宿主机**:容器不应也不需要装驱动,避免内外错配;容器只借用户态库。
2. **登录 Ascend Hub 再 pull**:官方镜像在受控仓库,需登录鉴权(对应文件顶部的 `docker login`)。
3. **镜像版本配宿主驱动**:不配套会在容器内 CANN 初始化时报错。
4. **设 shm-size / host 网络**:大模型多卡多进程依赖共享内存与高速互联,默认值往往不够。
5. **容器内自检**:先 `npu-smi` 再跑一个最小张量运算,把"环境问题"和"代码问题"切开,定位更快。

## 6. 迁移要点:从 nvidia-docker 心智迁过来(注意事项与坑)

| 你在 GPU 上的习惯 | 在昇腾上对应做法 / 坑 |
|------------------|----------------------|
| `--gpus all` | 用 Ascend Docker Runtime 声明可见卡;手动时要挂**多个** davinci 节点 |
| 容器内装 CUDA toolkit | 容器内装 CANN;**驱动不进容器**,靠挂宿主机用户态库 |
| `nvidia-smi` | `npu-smi`(见 [[ascend-npu-smi]]);**容器内要能看到卡才算挂对** |
| 镜像随便拉随便用 | 镜像 CANN 版本**必须配宿主驱动版本**,否则容器内初始化失败 |
| NCCL 自动走 NVLink/IB | HCCL 走 HCCS/RoCE;多机要正确配 host 网络与 RDMA(见 [[HCCL]]、[[network]]) |
| `--privileged` 图省事 | **反模式**:应精确挂 `/dev/davinci*` 等,最小权限;privileged 会带来安全与可移植性问题 |

高频坑清单:
- **漏挂设备节点**:只挂 `davinci0`,漏 `davinci_manager`/`devmm_svm`/`hisi_hdc` → 卡可见但初始化失败。→ 用 runtime 自动注入。
- **驱动与 CANN 版本错配**:换了镜像没看配套表 → CANN 报版本不兼容。→ 先查官方配套表。
- **shm 太小**:多进程数据加载/通信 OOM 或 hang。→ 调大 `--shm-size`。
- **多机网络没用 host 模式 / RDMA 没配**:HCCL 建链失败或带宽极低。→ host 网络 + 正确网卡/IP 配置。
- **容器内时区/字符集/依赖缺失**:官方镜像基座可能精简,业务依赖要自己补。
- **k8s 上忘了装 Ascend Device Plugin**:Pod 申请 NPU 资源调度不到。→ 对应 GPU 的 NVIDIA Device Plugin。

## 7. 性能与调优(机制层面)

- **NUMA / 亲和性**:NPU 与 CPU/网卡有拓扑亲和关系,绑核绑卡能减少跨 NUMA 访存。容器要能拿到 `npu-smi` 与拓扑信息才能做亲和(见 [[npu监控]])。
- **通信路径**:单机多卡走 HCCS,多机走 RoCE/RDMA;容器网络模式直接影响 HCCL 能否走最优路径(host 网络优先)。
- **共享内存**:`shm-size` 影响多进程数据流水线,过小会成为瓶颈甚至 hang。
- **镜像瘦身**:把构建依赖与运行依赖分层,减小最终镜像,提升分发与冷启动速度——和 GPU 镜像优化思路一致。

## 8. 常见问题(表格)

| 问题 | 答案 |
|------|------|
| 昇腾 Docker 和 nvidia-docker 是一回事吗? | 解决的问题同构(让容器访问加速器),实现是各自生态;对应物是 **Ascend Docker Runtime ↔ nvidia-container-runtime**。 |
| 驱动要装进容器吗? | **不要**。驱动/固件留宿主机,容器只挂宿主机的驱动**用户态库**。 |
| 为什么挂了 `/dev/davinci0` 还是初始化失败? | 多半漏挂了 `davinci_manager`/`devmm_svm`/`hisi_hdc`。用 runtime 自动注入最稳。 |
| 镜像选 CANN 版本看什么? | 看宿主机**驱动版本**的配套表;CANN 必须在驱动支持区间内。 |
| 一定要用 Ascend Docker Runtime 吗? | 不强制,但**强烈推荐**;否则手动挂载极易漏项。 |
| 容器内怎么确认卡可用? | 跑 `npu-smi` 看卡,再用框架做一次 NPU 张量运算自检。 |
| k8s 上怎么调度 NPU? | 装 **Ascend Device Plugin**(对应 NVIDIA Device Plugin),Pod 按资源名申请。 |
| 该用 `--privileged` 吗? | 尽量不用;应最小权限精确挂设备,privileged 是反模式。 |
| 精确命令/版本去哪查? | **以华为昇腾官方文档(Ascend 社区)与镜像仓库说明为准**,本文不固化具体版本号。 |

## 🔗 跳转链接

- 枢纽:[[00-知识地图]]
- 同目录:[[ascend-docker-runtime]] · [[昇腾镜像]] · [[docker环境升级cann]] · [[ascend-npu-smi]] · [[npu监控]] · [[环境安装]] · [[network]] · [[达芬奇架构]] · [[HCCL]]
- 算力/硬件:[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/CUDA]]
- 网络/通信:[[ai-infra/网络/NCCL]] · [[ai-infra/网络/集合通信原语]]
- 框架/套件:[[ai-framework/megatron-lm/README]] · [[ai-framework/huggingface-transformers/README]]
- 训练/推理/压缩:[[llm-train/README]] · [[llm-inference/README]] · [[llm-compression/quantization/量化基础]]
- 模型:[[llm-algo/transformer/模型架构]]
