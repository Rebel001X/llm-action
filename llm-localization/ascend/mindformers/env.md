# MindFormers 环境搭建（昇腾）

> MindFormers 是华为昇腾上的大模型训练/微调/推理套件，本篇讲它的环境怎么搭、各步骤为什么这么做、迁移自 GPU 时易踩的坑。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-infra/ai-hardware/CUDA]] [[ai-framework/megatron-lm/README]]

## 阅读地图

| 你想知道 | 看哪一节 |
| --- | --- |
| MindFormers 是什么、在栈里哪一层 | 0、1 |
| 昇腾环境和英伟达环境怎么对应 | 1（对照表） |
| 软件栈从硬件到套件怎么层层依赖 | 2（ASCII 图） |
| 搭环境的整体流程（驱动→CANN→框架→套件） | 3 |
| 为什么优先用官方 Docker 镜像 | 4 |
| 容器为什么要挂那么多 `/dev/davinci*` | 5 |
| 从 GPU/PyTorch 迁过来要改什么、坑在哪 | 迁移要点 |
| 装完怎么验证环境是好的 | 环境自检 |
| 常见报错速查 | 常见问题 |

## 0. 一句话锚点

**MindFormers ≈ 昇腾世界的 Megatron-LM + HuggingFace Transformers**：它基于 MindSpore，封装了主流大模型的并行训练、微调、推理全流程；而"搭环境"的本质，是把 **驱动 → CANN → MindSpore → MindFormers** 这条依赖链按版本配套关系一层层装好、且让容器能访问到 NPU 设备。

## 1. 地基：在昇腾栈的定位 + 对标英伟达生态

MindFormers 处于昇腾软件栈的**最上层（套件层）**。它不直接和硬件打交道，而是站在一长串依赖之上：底层是昇腾 NPU 硬件（达芬奇架构），往上是驱动/固件，再往上是 CANN（异构计算架构，相当于 CUDA + cuDNN + NCCL 的合体），再往上是 MindSpore 深度学习框架（相当于 PyTorch），最后才是 MindFormers 这层大模型套件。

理解这张"昇腾 ↔ 英伟达"对照表，是从 GPU 迁移过来的第一张心智地图：

| 层级 | 昇腾（Ascend） | 英伟达（NVIDIA） | 说明 |
| --- | --- | --- | --- |
| 加速硬件 | NPU（昇腾 910/310 系列） | GPU（A100/H100…） | 计算单元不同：达芬奇 Cube/Vector vs SM/Tensor Core |
| 驱动/固件 | Ascend Driver + Firmware | NVIDIA Driver | 让 OS 认得加速卡 |
| 计算底座 | **CANN** | **CUDA** | 编程模型 + 运行时 |
| 算子加速库 | CANN 内置算子库 | cuDNN / cuBLAS | 卷积/矩阵乘等高性能算子 |
| 集合通信 | **HCCL** | **NCCL** | 多卡/多机 AllReduce 等 |
| 设备查询工具 | `npu-smi` | `nvidia-smi` | 看卡、看占用 |
| 深度学习框架 | **MindSpore**（亦支持 PyTorch+torch_npu） | PyTorch / TensorFlow | 自动微分、图执行 |
| 大模型套件 | **MindFormers** | Megatron-LM / HF Transformers | 并行训练 + 模型库 |
| 推理引擎 | MindIE | TensorRT-LLM / vLLM | 部署推理 |
| 量化工具 | msModelSlim | GPTQ / AWQ 工具链 | 压缩与量化 |

> 一句话记忆：**搭 MindFormers 环境 = 搭 CUDA 那套，但每一层都换成昇腾的对应物，且层与层之间的"版本配套"卡得更严**。版本配套是昇腾环境最大的特点——驱动、CANN、MindSpore、MindFormers 必须是官方认证的配套组合，错配是头号事故源。

## 2. 软件栈：从硬件到套件的依赖链

```
         应用：你的训练 / 微调 / 推理脚本
   ┌───────────────────────────────────────────┐
   │  MindFormers  (大模型套件 ≈ Megatron + HF) │  ← 本篇的主角
   ├───────────────────────────────────────────┤
   │  MindSpore    (深度学习框架 ≈ PyTorch)     │  ← 自动微分 / 图模式
   ├───────────────────────────────────────────┤
   │  CANN         (异构计算架构 ≈ CUDA)        │  ← 算子库 / Runtime / HCCL
   │      ├── 算子库     (≈ cuDNN/cuBLAS)       │
   │      └── HCCL       (≈ NCCL，多卡通信)     │
   ├───────────────────────────────────────────┤
   │  Driver + Firmware  (≈ NVIDIA Driver)     │  ← OS 认卡
   ├───────────────────────────────────────────┤
   │  昇腾 NPU 硬件 (达芬奇架构 Cube/Vector)    │  ← 物理算力
   └───────────────────────────────────────────┘
```

**关键认知**：上层永远依赖下层，且依赖的是**特定版本**。这意味着：

- 你不能随便升级 MindSpore 而不管 CANN——它们必须配套。
- 容器里通常打包了 CANN + MindSpore + MindFormers（运行环境），但 **Driver/Firmware 留在宿主机**，靠挂载共享进容器。这就是为什么 `docker run` 要把宿主机的 `/usr/local/Ascend/driver` 挂进去。

## 3. 环境搭建的整体流程（讲含义，不背命令）

搭建分两条路线：**A. 直接用官方 Docker 镜像（强烈推荐）**；**B. 裸机从头装**。无论哪条，逻辑都是"先把下层地基铺好，再装上层套件"。

```
   宿主机（物理机/服务器）
   ┌────────────────────────────────────────┐
   │ ① 装 Driver + Firmware（一次性，宿主机）│
   │    → npu-smi 能看到卡 = 地基 OK         │
   └───────────────┬────────────────────────┘
                   │ docker run 挂载 driver + 设备
                   ▼
   容器内（开发/训练环境）
   ┌────────────────────────────────────────┐
   │ ② CANN     已在官方镜像里                │
   │ ③ MindSpore 已在官方镜像里               │
   │ ④ MindFormers：拉源码 → build.sh 编译装  │
   └────────────────────────────────────────┘
```

各步骤的**含义与注意点**（具体命令与版本以华为昇腾官方文档（Ascend 社区）为准）：

1. **装 Driver + Firmware（宿主机层）**：让操作系统识别 NPU。验证标准是 `npu-smi info` 能列出所有卡。这一步一旦装好，后续都靠挂载复用，不需要在容器里重装。
2. **准备 CANN（容器层，镜像已含）**：CANN 提供算子库、Runtime、HCCL，是"昇腾版 CUDA"。官方镜像通常已把配套版本的 CANN 装好，所以**优先用镜像能省掉最痛的版本对配**。
3. **准备 MindSpore（容器层，镜像已含）**：框架层，需与 CANN 配套。注意区分 CPU/Ascend 版本，必须装 **Ascend 版**才能调 NPU。
4. **安装 MindFormers（套件层，手动）**：通常从 Gitee 仓库拉对应分支源码，再用仓库内的构建脚本编译安装。这一步要选**与镜像里 MindSpore 版本匹配的 MindFormers 分支/Tag**，错配会在 import 时直接报错。

> 镜像名/版本号/分支名都会随时间变化，**不要硬记**；以官方文档给出的"版本配套表"为唯一依据。

## 4. 为什么优先用官方 Docker 镜像

裸机装的最大痛点是**版本配套**：Driver、CANN、MindSpore、MindFormers 四层必须严丝合缝。官方镜像把"CANN + MindSpore（+常含 MindFormers 依赖）"这一整套配套版本打包好了，你只需在宿主机装好 Driver，再把驱动和设备挂进容器即可，等于跳过了三层踩坑。

```
   裸机装（坑多）                  用官方镜像（坑少）
   ─────────────                  ────────────────
   手动装 Driver  ✔ 必须          手动装 Driver   ✔ 仍必须（宿主机）
   手动装 CANN    ✘ 易错配        CANN 已配好     ✔ 镜像自带
   手动装 MindSpore ✘ 易错配      MindSpore 已配好 ✔ 镜像自带
   手动装 MindFormers             MindFormers     仅这一步手动
```

注意昇腾镜像通常区分 **CPU 架构**（很多昇腾服务器是 **ARM64/aarch64**，鲲鹏 CPU），拉镜像时要选对架构，否则容器起不来。具体镜像仓库地址、平台标识、版本以官方文档为准。

## 5. 容器为什么要挂那么多 `/dev/davinci*`

`docker run` 启动训练容器时，要把宿主机的 NPU 设备文件透传进容器，否则容器内看不到卡。这些挂载各有含义：

```
   宿主机                          容器内（透传后可用）
   /dev/davinci0..7      ──────▶   8 张 NPU 卡的设备节点（每卡一个）
   /dev/davinci_manager  ──────▶   设备管理（卡的统一管理入口）
   /dev/devmm_svm        ──────▶   共享虚拟内存（SVM）
   /dev/hisi_hdc         ──────▶   主机-设备通信通道（HDC）
   /usr/local/Ascend/driver  ──▶   宿主机驱动（容器复用，不重装）
   /usr/local/bin/npu-smi    ──▶   设备查询工具（容器内也能 npu-smi）
   /var/log/npu              ──▶   NPU 日志（便于排障）
```

- **`/dev/davinci0..7`**：有几张卡就挂几个；只想用部分卡就只挂对应编号。这相当于 GPU 世界里 `--gpus` 指定可见 GPU。
- **`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`**：缺一不可的"基础设施"设备，少挂任意一个都可能在初始化时报错。
- **挂载 driver 而非容器内重装**：驱动是内核态的，必须和宿主机内核匹配，所以只能用宿主机的，靠挂载共享。
- **`--ipc=host` / `--network=host`**：多卡/多机训练时进程间通信、HCCL 通信常需要主机网络与共享内存，方便起见常开放（生产环境注意隔离性权衡）。

## 迁移要点 / 注意事项与坑

从 GPU + PyTorch + Megatron 迁到昇腾 + MindSpore + MindFormers，主要心智转换：

1. **版本配套是头号天条**：GPU 世界里 PyTorch 和 CUDA 容忍度较高；昇腾里 Driver/CANN/MindSpore/MindFormers 必须严格配套。**先查官方"版本配套表"，再动手**，能避开 80% 的坑。
2. **优先镜像、别裸装**：除非有特殊需求，直接用官方镜像；裸装意味着自己背配套责任。
3. **设备挂载别漏**：`davinci_manager`/`devmm_svm`/`hisi_hdc` 三个管理设备漏挂是新手最常见报错来源；驱动目录必须挂宿主机的。
4. **架构要对**：很多昇腾服务器是 ARM64，拉镜像/装包都要选 aarch64，x86 包在上面跑不了。
5. **框架范式差异**：MindSpore 默认强于**图模式（Graph Mode）**，与 PyTorch 的 eager 动态图心智不同；MindFormers 的并行配置（数据/张量/流水线并行）是用 **YAML 配置 + 脚本**驱动，对标 Megatron 的命令行参数，迁移时重点是把并行策略翻译成 MindFormers 的配置项。
6. **集合通信换成 HCCL**：多机训练要配 **HCCL 的 rank table / 组网信息**（对标 NCCL 的环境变量与 host 互通），网络不通是分布式训练第一杀手。
7. **算子覆盖度**：个别 GPU 上常见的自定义算子，昇腾不一定原生支持，可能要用等价算子替换或等官方适配——迁移自定义 kernel 时尤其注意。
8. **日志排障习惯**：昇腾报错常需看 NPU plog 日志（已挂的 `/var/log/npu`），和 GPU 看 CUDA error 的习惯不同。

## 环境自检（装完怎么确认是好的）

按依赖链**自下而上**逐层验证，哪层断了就知道问题在哪：

```
   ① npu-smi 能列出卡？        否 → Driver/设备挂载问题（最底层）
        │是
        ▼
   ② 容器内能 import MindSpore  否 → MindSpore/CANN 配套问题
      且 set_context 到 Ascend？
        │是
        ▼
   ③ 能 import mindformers？    否 → MindFormers 安装/分支版本问题
        │是
        ▼
   ④ 跑一个最小训练/推理样例？  否 → 配置 / 并行 / 数据问题
        │是
        ▼
        环境 OK ✔
```

具体的自检命令、API 名称以官方文档为准；核心思路是**逐层确认，不要跳层排查**。

## 常见问题

| 问题 | 可能原因 | 排查方向 |
| --- | --- | --- |
| `npu-smi` 看不到卡 | Driver 未装好 / 设备未挂载 | 先在宿主机确认，再查容器挂载 |
| 容器起不来或架构报错 | 镜像架构（x86 vs ARM64）不匹配 | 拉对应 CPU 架构的镜像 |
| import MindSpore 报错 | 装成 CPU 版 / 与 CANN 不配套 | 装 Ascend 版且版本配套 |
| import mindformers 失败 | 分支/Tag 与 MindSpore 不匹配 | 选配套分支重新构建 |
| 初始化时设备相关报错 | 漏挂 manager/svm/hdc 设备 | 补齐三个管理设备节点 |
| 多机训练卡住/不通 | HCCL 组网/rank table 配错 | 检查网络互通与 rank 配置 |
| 找不到驱动 | 没挂宿主机 driver 目录 | 挂载 `/usr/local/Ascend/driver` |
| 性能远低于预期 | 图模式/算子未优化/并行策略不当 | 用图模式、查算子、调并行 |

> 牢记护栏：以上不背任何具体命令/版本/镜像名/分支名——这些**全部以华为昇腾官方文档（Ascend 社区）的版本配套表为准**，因为它们随版本演进会变，硬记反而误导。

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
