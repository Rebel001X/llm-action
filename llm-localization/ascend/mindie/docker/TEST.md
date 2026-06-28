# MindIE 推理引擎的 Docker 容器化部署

> 把昇腾 MindIE 推理引擎及其全部依赖(驱动联动、CANN、加速库、模型仓)装进一个可复现的容器里,一条命令拉起一套可推理的 NPU 环境。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[llm-inference/README]]

## 阅读地图

| 章节 | 你会得到什么 | 对标 CUDA 世界的类比 |
| --- | --- | --- |
| 0. 一句话锚点 | 这个目录到底在干嘛 | 一句话先定调 |
| 1. 地基:昇腾栈定位 + 对照表 | MindIE 容器在硬件→驱动→CANN→引擎中的层次 | `nvidia-docker` + TensorRT-LLM 镜像 |
| 2. 容器为什么这么"重" | NPU 设备直通 / 驱动卷挂载的原理 | `--gpus all` vs `--device` 手动直通 |
| 3. 两层镜像:env 与 all | env 基座镜像 + all 业务镜像的分层逻辑 | base CUDA 镜像 + 应用层镜像 |
| 4. 完整部署流程(7 步) | 从拿机器到发起一次推理 | 拉 nvcr.io 镜像→跑 trtllm-serve |
| 5. docker run 参数逐行拆解 | 每个 `--device` / `-v` 为什么不能少 | `--gpus` 一行搞定 vs 昇腾要手动列 |
| 6. 迁移要点与常见坑 | 从 GPU 容器迁过来要改什么 | 心智迁移图 |
| 常见问题 | 启动报错的高频原因速查 | FAQ |

## 0. 一句话锚点

**这个目录(`ascend/mindie/docker/`)讲的是:如何用 Docker 把 MindIE 推理引擎打包并在昇腾 NPU 上拉起来。** 核心难点不在"写 Dockerfile",而在于 **NPU 不像 GPU 有 `--gpus all` 这种一键直通**——你必须把 davinci 设备、驱动管理设备、以及宿主机上的驱动/工具目录,**一个个手动直通和挂载**进容器。本文件顶部保留的 `docker build` / `docker run` 命令就是这套"手动直通"的标准模板。

> 一句记忆:**镜像装的是"上层软件(CANN+MindIE)",驱动留在宿主机,靠 `--device` 直通 + `-v` 挂载把两者在容器里"接起来"。**

## 1. 地基:在昇腾栈的定位 + 对标英伟达生态

### 1.1 MindIE 容器在昇腾软件栈的哪一层

昇腾软件栈自底向上是:**硬件(NPU)→ 驱动/固件 → CANN → 框架/加速库 → 推理引擎(MindIE)**。容器化部署的关键认知是 **一条"分界线"**:

```
            ┌──────────────────────────── 容器内 (镜像里装的) ───────────────────────────┐
   应用层    │   你的服务脚本 / 模型权重 / 推理请求                                        │
   引擎层    │   MindIE (Service 服务化 + LLM 推理 + Benchmark)   ← 对标 TensorRT-LLM/vLLM │
   加速库    │   ATB 加速库 / 算子库 / Python+Torch_NPU 适配                               │
   异构计算   │   CANN (Runtime + 编译器 + 算子)                  ← 对标 CUDA Toolkit      │
            └─────────────────────────────────┬──────────────────────────────────────────┘
                                              │  ↑↓  靠 --device 直通 + -v 挂载 "穿透分界线"
            ┌─────────────────────────────────┴──────────────────────────────────────────┐
   驱动层    │   Ascend Driver / 固件 (npu-smi / dcmi)           ← 留在【宿主机】不进镜像   │
   硬件层    │   昇腾 NPU 芯片 (达芬奇架构 Cube/Vector 单元)      ← /dev/davinci*           │
            └───────────────────────────────────────────────────────────────────────────┘
```

> 黄金法则:**驱动(Driver/固件)装在宿主机,绝不打进镜像;CANN 及以上(加速库、MindIE)装进镜像。** 原因见第 2 节。这与 GPU 世界"驱动在宿主机、CUDA Toolkit 在镜像里"的分工是一致的。

### 1.2 昇腾 ↔ 英伟达 生态对照表(迁移心智图)

| 维度 | 昇腾(Ascend) | 英伟达(NVIDIA) | 一句话说明 |
| --- | --- | --- | --- |
| 加速硬件 | NPU(达芬奇架构) | GPU | 算力芯片本身 |
| 设备节点 | `/dev/davinci0..N` | `/dev/nvidia0..N` | 每张卡一个字符设备 |
| 设备管理节点 | `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` | `/dev/nvidiactl`、`/dev/nvidia-uvm` | 管理/内存/HDC 通道 |
| 异构计算平台 | CANN | CUDA Toolkit | 编译器 + Runtime + 算子 |
| 算子库 | CANN 算子库 / ATB 加速库 | cuDNN / cuBLAS | 高性能算子集合 |
| 设备状态工具 | `npu-smi` / `dcmi` | `nvidia-smi` | 查卡、看负载、看显存 |
| 容器直通方案 | 手动 `--device` + 挂载驱动目录 | `--gpus all`(nvidia-container-toolkit) | **最大差异点** |
| 训练框架 | MindSpore / PyTorch(torch_npu) | PyTorch / TensorFlow | 上层框架 |
| 大模型训练套件 | MindFormers / ModelLink | Megatron-LM / HF | 并行训练 |
| **推理引擎** | **MindIE** | **TensorRT-LLM / vLLM** | **本目录主角** |
| 集合通信 | HCCL | NCCL | 多卡通信库 |
| 量化工具 | msModelSlim | GPTQ / AWQ 工具链 | 压缩量化 |
| 容器镜像仓 | ascendhub.huawei.com | nvcr.io(NGC) | 官方镜像源 |

> 一句话:**MindIE ≈ 昇腾上的 TensorRT-LLM/vLLM;它的 Docker 部署 ≈ 从 NGC 拉镜像跑 trtllm-serve,但设备直通要手动而非 `--gpus all`。**

## 2. 容器为什么这么"重":NPU 直通与驱动挂载的原理

在 GPU 世界,`nvidia-container-toolkit` 帮你把 `--gpus all` 翻译成"设备节点直通 + 驱动库注入"。**昇腾目前的主流做法是手动完成这两件事**(也有 Ascend Docker Runtime 可简化,但理解手动版才能排错)。所以那一长串 `--device` 和 `-v` 不是冗余,而是缺一不可:

```
docker run 的三类必需参数,各自解决一个问题:
┌──────────────────────┬─────────────────────────────────────────────┐
│  --device=/dev/davinciX │  把"算力卡本体"直通进容器(你要用第几张卡就直通第几张) │
│  --device=/dev/davinci_manager │ 设备管理通道:进程要先经它注册才能用卡       │
│  --device=/dev/devmm_svm       │ 共享虚拟内存(SVM):Host/Device 统一寻址      │
│  --device=/dev/hisi_hdc        │ HDC 主机-设备通信通道                         │
├──────────────────────┼─────────────────────────────────────────────┤
│  -v .../Ascend/driver  │  把宿主机【驱动用户态库】挂进去——镜像里没装驱动!  │
│  -v .../dcmi           │  设备管理接口库,npu-smi 依赖它                   │
│  -v .../npu-smi        │  让容器内也能 npu-smi 看卡                        │
├──────────────────────┼─────────────────────────────────────────────┤
│  --shm-size=50g        │  大共享内存:多进程/多卡推理走 /dev/shm 传张量      │
│  --ipc=host / --net=host │ 共享 IPC 与网络命名空间,降低多卡/服务通信开销   │
└──────────────────────┴─────────────────────────────────────────────┘
```

**为什么驱动不打进镜像?** 因为驱动用户态库**必须和宿主机内核态驱动版本严格匹配**。若把驱动封进镜像,换一台驱动版本不同的机器镜像就跑不起来。把驱动留在宿主机、用 `-v` 挂载,镜像才能"一次构建、多机复用"。这正对应 GPU 世界"镜像里不装显卡驱动,只装 CUDA"的设计。

## 3. 两层镜像:`mindie-env` 与 `mindie-all`

本目录的两个 Dockerfile 体现了**分层构建**思想(对标 GPU 的 base 镜像 + 应用镜像):

```
┌─────────────────────────────────────────────────────────────┐
│  mindie-env  (基座镜像 / base image)                          │
│  ── OS 基础 + CANN 运行时 + Python/Torch_NPU + 系统依赖        │
│  ── 变动慢、体积大、可被多个业务镜像复用                       │
└───────────────────────────┬─────────────────────────────────┘
                            │  FROM mindie-env  (在其之上叠加)
┌───────────────────────────┴─────────────────────────────────┐
│  mindie-all  (业务镜像 / app image)                           │
│  ── MindIE 引擎 + ATB 加速库 + 模型适配脚本 + 启动入口         │
│  ── 变动快、迭代频繁;只重建这一层即可,base 层走缓存          │
└─────────────────────────────────────────────────────────────┘
```

**好处**:基座 `env` 镜像内容稳定,可被缓存复用;每次只迭代上层 `all` 镜像,构建快、传输小。这与"用 `nvcr.io/nvidia/pytorch` 当 base、再 `FROM` 它装业务代码"如出一辙。

> 命名里的 `1.0.RC1-800I-A2-aarch64` 表达三层信息:**软件版本 + 硬件型号(800I A2 推理卡)+ CPU 架构(aarch64,昇腾常配鲲鹏 ARM)**。具体版本号、镜像 tag、Dockerfile 内容以华为昇腾官方文档(Ascend 社区 / ascendhub)为准,切勿照抄本文中的示例 tag。

## 4. 完整部署流程(7 步,讲"为什么"而非背命令)

```
[1] 装驱动/固件(宿主机)──→[2] 装 Docker + (可选)Ascend Docker Runtime
        │                           │
        ▼                           ▼
[3] 拉/构建 mindie-env 基座 ──→[4] 构建 mindie-all 业务镜像
        │                           │
        ▼                           ▼
[5] docker run 直通 NPU 拉起容器 ──→[6] 容器内 npu-smi 自检(能看到卡=直通成功)
        │
        ▼
[7] 放权重 + 起 MindIE Service / 跑 benchmark 发起推理
```

1. **宿主机装驱动/固件**:这是地基,版本要和后续 CANN 匹配。装完用 `npu-smi` 能看到卡才算成功。
2. **装 Docker(可选 Ascend Docker Runtime)**:用了官方 Runtime 可省去手动列 `--device`,接近 `--gpus all` 体验;不用则照本文模板手动直通。
3. **准备 `mindie-env` 基座镜像**:可从 ascendhub 拉官方镜像,或用 `mindie-env-*.Dockerfile` 自建。
4. **构建 `mindie-all` 业务镜像**:`FROM` 基座,叠加 MindIE 引擎层。
5. **`docker run` 拉起容器**:照第 5 节逐行核对设备与挂载。
6. **容器内自检**:进容器跑 `npu-smi info`,看得到所有直通的卡 = 设备直通 + 驱动挂载都对了。**这是排错第一站。**
7. **发起推理**:挂载/拷入模型权重,启动 MindIE 的服务化(Service)或基准测试(Benchmark)。

> 第 6 步是黄金分割点:**容器内 `npu-smi` 看不到卡,99% 是直通/挂载没配对,而不是 MindIE 的问题。** 具体命令与版本以华为昇腾官方文档为准。

## 5. `docker run` 参数逐行拆解(对应文件顶部模板)

| 参数 | 作用 | 漏了会怎样 |
| --- | --- | --- |
| `--device=/dev/davinci4..7` | 直通第 4~7 号 NPU 卡(本例用 4 张) | 容器内看不到对应卡 |
| `--device=/dev/davinci_manager` | 设备管理通道,进程注册必经 | 初始化报错、无法 acquire 卡 |
| `--device=/dev/hisi_hdc` | 主机-设备 HDC 通信 | 部分驱动交互失败 |
| `--device=/dev/devmm_svm` | 共享虚拟内存 | 内存/寻址相关报错 |
| `-v .../Ascend/driver` | 挂宿主机驱动用户态库 | 找不到驱动库,直接起不来 |
| `-v .../dcmi`、`-v .../npu-smi` | 挂设备管理接口与查卡工具 | 容器内 npu-smi 不可用 |
| `--shm-size=50g` | 放大共享内存 | 多卡/大 batch 推理 OOM 或卡死 |
| `--ipc=host` | 共享 IPC 命名空间 | 多进程张量共享受限 |
| `--net=host` | 共享网络栈 | 服务端口/多机通信复杂化 |
| `--privileged=true` | 放开设备访问权限 | 权限不足无法访问设备节点 |
| `-v /usr/share/zoneinfo/...:/etc/localtime` | 对齐时区 | 日志时间错乱(非致命) |

> **关键:`--device=/dev/davinci4..7` 里的编号要和你实际想用的卡对应。** 想用 0~3 号卡,就改成 `davinci0..3`。这正是没有 `--gpus all` 时的代价——你得"点名"用哪几张卡。

## 6. 迁移要点与常见坑(从 GPU 容器迁到昇腾)

### 6.1 迁移要点对照

```
GPU 容器经验            →   昇腾容器对应做法
────────────────────────────────────────────────────────
docker run --gpus all   →   手动 --device=/dev/davinciX + 挂载驱动目录
                            (或装 Ascend Docker Runtime 简化)
nvidia-smi 自检          →   npu-smi info 自检
nvcr.io 拉镜像           →   ascendhub.huawei.com 拉镜像
FROM nvidia/cuda base    →   FROM mindie-env base
trtllm-serve / vllm serve→   MindIE Service 服务化
镜像里装 CUDA、驱动在宿主 →   镜像里装 CANN、驱动在宿主(理念一致)
```

### 6.2 高频坑

- **坑 1:容器里 `npu-smi` 看不到卡。** 多半是 `--device` 漏了某个管理节点(`davinci_manager`/`devmm_svm`/`hisi_hdc`),或驱动目录没 `-v` 挂上。逐项对照第 5 节。
- **坑 2:驱动版本与 CANN/镜像不匹配。** 宿主机驱动太旧或太新,与镜像内 CANN 版本不配套会初始化失败。**镜像 tag 与驱动版本的配套关系以官方文档为准。**
- **坑 3:`--shm-size` 太小。** 多卡推理走共享内存传张量,默认 64MB 远不够,需显式调大(本例 50g)。
- **坑 4:卡编号写死照抄。** 直接抄 `davinci4..7` 但你的机器只有 0~3 号空闲,会直通失败。**先 `npu-smi` 看哪些卡空闲再填编号。**
- **坑 5:架构不匹配。** 昇腾服务器常是 aarch64(鲲鹏 ARM),拉了 x86 镜像跑不起来。看准 tag 里的 `aarch64`/`x86_64`。
- **坑 6:权限不足。** 设备节点访问需要相应权限,本例用 `--privileged=true`;生产环境若收紧权限,需为容器用户配好设备访问组。

> 性能调优(机制层面):多卡推理瓶颈常在 **HCCL 通信**(对标 NCCL),确认走的是高速互联而非 PCIe 回退;`--shm-size` 与 `--ipc=host` 保证跨进程零拷贝;**绑定 NUMA / CPU 亲和**(昇腾 + 鲲鹏 ARM 多 NUMA 域)对吞吐影响显著。

## 常见问题

| 问题 | 答案 |
| --- | --- |
| MindIE 对标 GPU 上的什么? | TensorRT-LLM / vLLM,是昇腾的 LLM 推理引擎 |
| 为什么不用 `--gpus all`? | 昇腾默认需手动 `--device` 直通;装 Ascend Docker Runtime 可简化 |
| 驱动要不要打进镜像? | **不要**。驱动留宿主机,用 `-v` 挂载,保证镜像跨机复用 |
| `mindie-env` 和 `mindie-all` 区别? | env 是基座(OS+CANN),all 在其上叠加 MindIE 引擎层 |
| 容器自检第一步做什么? | 进容器跑 `npu-smi info`,看得到卡才算直通成功 |
| tag 里 `800I-A2`/`aarch64` 啥意思? | 硬件型号(800I A2 推理卡)+ CPU 架构(ARM) |
| 想换用第 0~3 号卡怎么改? | 把 `--device=/dev/davinci4..7` 改成 `davinci0..3` |
| 具体命令/版本去哪查? | **一律以华为昇腾官方文档(Ascend 社区 / ascendhub)为准** |

## 🔗 跳转链接

- [[00-知识地图]]
- [[ai-infra/算力/昇腾NPU]]
- [[ai-infra/ai-hardware/AI芯片软件生态]]
- [[ai-infra/ai-hardware/CUDA]]
- [[ai-infra/网络/NCCL]]
- [[ai-infra/网络/集合通信原语]]
- [[ai-framework/megatron-lm/README]]
- [[ai-framework/huggingface-transformers/README]]
- [[llm-compression/quantization/量化基础]]
- [[llm-inference/README]]
- [[llm-train/README]]
- [[llm-algo/transformer/模型架构]]
