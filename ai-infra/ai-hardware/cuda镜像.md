# CUDA 镜像（NGC CUDA Container Images）

> NVIDIA 官方维护的、把"特定版本 CUDA 工具栈"打包进容器的基础镜像，是绝大多数深度学习镜像（PyTorch/TensorRT/vLLM…）的"地基层"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/ai-hardware/README]] [[ai-infra/ai-hardware/CUDA]] [[llmops/kubernetes]] [[llm-inference/tensorrt/README]]

## 阅读地图

| 小节 | 你将搞清楚 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | base/runtime/devel |
| 1 | 它解决什么问题 | 驱动 vs 运行时解耦 |
| 2 | 镜像在整个栈里的位置 | 分层架构 ASCII |
| 3 | 三种变体 base/runtime/devel | 体积 vs 能力 |
| 4 | tag 命名规则怎么读 | CUDA/cuDNN/OS |
| 5 | GPU 怎么进容器（核心机制） | container-toolkit |
| 6 | 自己基于它构建镜像 | Dockerfile 分层 |
| 7 | 版本兼容如何对齐 | 驱动/CUDA/框架 |
| 示例 | 拉取 + 跑通 + 自建 | 命令骨架 |
| 坑 | 高频错误 | 表格 |

## 0. 一句话锚点

CUDA 镜像 = **一个已经把 `nvcc`、CUDA Runtime、（可选）cuDNN 等放好环境变量的 Linux 基础镜像**。它**不包含 GPU 驱动**——驱动由宿主机提供，运行时通过 NVIDIA Container Toolkit 注入。官方分发渠道是 NGC（`nvcr.io/nvidia/cuda`）和 Docker Hub（`nvidia/cuda`）。

> 顶部链接里的 `nvcr.io/nvidia/cuda:12.1.0-cudnn8-runtime-centos7` 就是一个典型 tag：CUDA 12.1.0 + cuDNN8 + runtime 变体 + CentOS7 基底。

## 1. 地基：它到底解决什么问题

裸机上做 CUDA 开发的痛点：

1. **CUDA Toolkit 安装繁琐**：不同项目要 CUDA 11.8 / 12.1 / 12.4，宿主机只能装一套，多版本切换靠 `update-alternatives` 或手动改 `PATH`/`LD_LIBRARY_PATH`，极易冲突。
2. **环境不可复现**："在我机器上能跑"——别人机器 CUDA 版本不同就崩。
3. **CI/集群多租户**：K8s 集群上不同 Pod 需要不同 CUDA 版本。

容器化把"用户态 CUDA 栈"打包，**和宿主机解耦**，每个容器自带它需要的 CUDA 版本。核心前提是把一个事实想清楚：

```
GPU 软件栈被切成两半：
  ┌─────────────── 内核态（宿主机，全局唯一）───────────────┐
  │  NVIDIA Kernel Driver（nvidia.ko）+ /dev/nvidia*        │  ← 不进容器
  └─────────────────────────────────────────────────────────┘
  ┌─────────────── 用户态（可被容器打包，可多版本并存）─────┐
  │  CUDA Runtime(libcudart) / cuDNN / NCCL / nvcc / ...     │  ← 进容器
  └─────────────────────────────────────────────────────────┘
```

- **内核驱动**：版本全局唯一，决定"最高能支持到哪个 CUDA 版本"。
- **用户态 CUDA**：可以每个容器一份、互不干扰。

CUDA 镜像打包的正是**用户态那一半**。

## 2. 镜像在整个栈里的位置（分层架构）

```
                          你的应用镜像（自己写的 Dockerfile）
                          ┌───────────────────────────────────┐
   越往上越"业务"          │  my-llm:latest                    │
        ▲                  │   ├─ pip install vllm / sft 代码  │
        │                  └───────────────────────────────────┘
        │                                ▲ FROM
        │                  ┌───────────────────────────────────┐
   框架镜像（可选）        │  nvcr.io/nvidia/pytorch:24.xx       │  ← NGC 调好的框架镜像
        │                  └───────────────────────────────────┘
        │                                ▲ FROM
        │                  ┌───────────────────────────────────┐
  ★ CUDA 镜像（本文）      │  nvcr.io/nvidia/cuda:12.1.0-...     │  base/runtime/devel
        │                  └───────────────────────────────────┘
        │                                ▲ FROM
        │                  ┌───────────────────────────────────┐
   OS 基础镜像             │  ubuntu:22.04 / centos7 / ubi8      │
        ▼                  └───────────────────────────────────┘
   ════════════════ 容器边界（以上都在镜像里）═════════════════
                          ┌───────────────────────────────────┐
   宿主机（不在镜像里）    │  NVIDIA Driver + container-toolkit │  ← 运行时注入
                          │  物理 GPU                          │
                          └───────────────────────────────────┘
```

要点：**CUDA 镜像几乎从不直接用来跑业务**，它是别人（PyTorch/TensorRT/vLLM 镜像）和你自己 `FROM` 的"地基"。

## 3. 三种变体：base / runtime / devel

官方 CUDA 镜像对**同一个 CUDA 版本**提供三档，能力依次增强、体积依次增大。这是最需要记住的概念。

```
体积小 ◄──────────────────────────────────────► 体积大
能力弱                                          能力强

 base            runtime              devel
  │                │                    │
  │ 只有最小 CUDA  │ base + CUDA 运行   │ runtime + 编译工具链
  │ 运行时库       │ 时库全集 (math 库  │ (nvcc 头文件/静态库)
  │ (libcudart 等) │  cuFFT/cuBLAS...)  │ 可在容器内编译 .cu
  ▼                ▼                    ▼
仅跑已编译且依赖   跑大多数预编译框架   需要从源码编译
极少的程序         (PyTorch 推理常用)   CUDA 算子/扩展时用
```

| 变体 | 大致包含 | 典型用途 | 取舍 |
|---|---|---|---|
| **base** | 最小 CUDA 运行时（如 `libcudart`） | 自己精确控制要装哪些库；做最小镜像 | 体积最小，但很多框架缺库跑不起来 |
| **runtime** | base + CUDA 运行时库全集（cuBLAS/cuFFT/cuRAND/cuSPARSE/NPP 等），常见还带 cuDNN/NCCL | **部署/推理**：跑预编译好的框架 | 平衡点，最常用于生产推理镜像 |
| **devel** | runtime + `nvcc`、头文件、静态库、CUDA 样例 | **构建阶段**：编译自定义 CUDA 扩展（如 apex、flash-attn、自写算子） | 体积最大，含编译器，不宜直接做生产镜像 |

> 选择口诀：**编译用 devel，运行用 runtime，极简自定义用 base。** cuDNN 是否内置取决于 tag（带 `cudnnX`/`-cudnn` 才有），需要 cuDNN 又选了不带的 tag，就得自己装。

**多阶段构建（multi-stage）的经典搭配**：`devel` 阶段编译出 `.so`/wheel → 拷进 `runtime` 阶段做最终镜像，既能编译又能瘦身（详见第 6 节）。

## 4. tag 命名规则怎么读

tag 是用 `-` 拼出来的"特征清单"。读懂它就知道这镜像有什么。**字段顺序/可选项以官方 NGC、Docker Hub 标签列表为准**，常见结构是：

```
  nvcr.io/nvidia/cuda : 12.1.0 - cudnn8 - runtime - ubuntu22.04
  └────────┬────────┘   └──┬──┘  └──┬──┘  └──┬───┘  └────┬─────┘
        仓库地址         CUDA版本  cuDNN   变体        OS 基底
                       (X.Y.Z)  (可选)  base/      ubuntu20.04
                                        runtime/   ubuntu22.04
                                        devel      centos7 / ubi8 ...
```

- **仓库地址**：`nvcr.io/nvidia/cuda`（NGC，建议）或 `nvidia/cuda`（Docker Hub）。NGC 上拉取热门 tag 通常需登录/接受条款，**具体以 NGC 页面为准**。
- **CUDA 版本 `X.Y.Z`**：如 `12.1.0`、`12.4.1`。是否带补丁号、有哪些版本，看官方标签列表。
- **cuDNN 段（可选）**：带 `cudnn8` / `-cudnn` 说明内置 cuDNN，否则要自己装。
- **变体**：第 3 节的 base/runtime/devel。
- **OS 基底**：`ubuntu20.04`/`ubuntu22.04`/`centos7`/`ubi8`（Red Hat UBI）等。CentOS7 已 EOL，新项目优先 Ubuntu/UBI。**不要凭记忆假设某个具体组合一定存在**——拉之前在标签列表里确认。

> 经验：选 tag 时，**CUDA 版本要 ≤ 宿主机驱动支持的最高版本**（见第 7 节），OS 选团队熟悉且仍在维护的，变体按"编译/运行"二选一。

## 5. GPU 怎么进容器（最核心的运行机制）

镜像里没有驱动，那容器怎么用上 GPU？答案是 **NVIDIA Container Toolkit**（旧称 nvidia-docker）。它的作用是在容器启动瞬间，把宿主机的驱动库和设备节点**挂载/注入**进容器。

```
   docker run --gpus all  nvcr.io/nvidia/cuda:...   nvidia-smi
        │
        ▼
   ┌──────────────── 容器运行时（containerd/docker）─────────────┐
   │  调用 OCI runtime（runc）启动容器                            │
   │            │                                                 │
   │            ▼  prestart hook / CDI                            │
   │  ┌─────────────────────────────────────────────────────┐    │
   │  │ NVIDIA Container Toolkit                             │    │
   │  │  把宿主机以下东西注入容器：                          │    │
   │  │   • /dev/nvidia0, /dev/nvidiactl ...（设备节点）     │    │
   │  │   • libcuda.so / libnvidia-ml.so（驱动用户态库）    │    │
   │  │   • nvidia-smi 等工具                                │    │
   │  └─────────────────────────────────────────────────────┘    │
   └──────────────────────────────────────────────────────────────┘
        │
        ▼
   容器内： 镜像自带的 libcudart(用户态CUDA) + 注入的 libcuda(驱动)
            = 完整可用的 CUDA 栈 → nvidia-smi 看到 GPU ✓
```

关键拼图：
- **镜像提供**：`libcudart` 等用户态 CUDA 运行时（CUDA Runtime API）。
- **toolkit 注入**：`libcuda.so`（CUDA Driver API，**版本必须匹配宿主机驱动**）。
- 二者在容器内拼成完整栈。这就是为什么**镜像不打包驱动**——驱动必须和宿主机内核模块同源。

> 没装 toolkit 或忘了 `--gpus`，容器里 `nvidia-smi` 会报找不到设备/库——这是头号新手坑。K8s 上则由 **NVIDIA device plugin** 把 GPU 暴露成可调度资源 `nvidia.com/gpu`（见 [[llmops/kubernetes]]）。

## 6. 基于 CUDA 镜像构建自己的镜像

最常见的两种写法。**下面是结构骨架，确切包名/版本以官方为准**。

**(a) 单阶段——直接装框架跑推理（用 runtime）**

```dockerfile
FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*          # 清理缓存，减小层体积
RUN pip install torch --index-url <对应CUDA版本的wheel源>
COPY app/ /app
WORKDIR /app
CMD ["python3", "serve.py"]
```

**(b) 多阶段——编译自定义算子后瘦身（devel 编译 → runtime 运行）**

```dockerfile
# ---- 阶段1：用 devel 编译（有 nvcc）----
FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder
RUN pip install ninja && \
    pip install flash-attn --no-build-isolation   # 触发 .cu 编译
# ... 产出 wheel / .so

# ---- 阶段2：用 runtime 做最终镜像（无 nvcc，体积小）----
FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
COPY --from=builder /opt/wheels /opt/wheels
RUN pip install /opt/wheels/*.whl
CMD ["python3", "serve.py"]
```

为什么要分阶段：`nvcc`、头文件、静态库只在**编译期**需要，进了最终镜像就是死重量（往往多出 GB 级）。多阶段把编译产物"搬"出来，最终镜像只留 runtime。

> 构建自定义算子时常需设 `TORCH_CUDA_ARCH_LIST`（目标 GPU 架构，如 `8.0;9.0`），否则只编默认架构、换卡可能 `no kernel image is available`。架构号与 GPU 的对应关系见 [[ai-infra/ai-hardware/CUDA]]。

## 7. 版本兼容：三层必须对齐

这是 CUDA 容器最容易翻车的地方。三个东西要相互兼容：

```
   宿主机 NVIDIA 驱动版本
        │  决定"最高能跑的 CUDA 版本"（向后兼容：新驱动能跑老CUDA）
        ▼
   镜像里的 CUDA Toolkit 版本（tag 决定）
        │  必须 ≤ 驱动支持上限
        ▼
   框架（PyTorch/TensorRT…）编译时绑定的 CUDA 版本
           应与镜像 CUDA 大版本一致（如都 12.x）
```

三条规则：
1. **驱动 ≥ CUDA**：驱动通常向后兼容，新驱动能跑老 CUDA 容器；反之，旧驱动跑新 CUDA 镜像会报"驱动太旧"。具体支持矩阵以 NVIDIA 官方 *CUDA Compatibility* 文档为准。
2. **CUDA Forward Compatibility（前向兼容包）**：特定企业场景下，可让较老驱动跑较新 CUDA，但有限制——别当默认手段。
3. **框架 ↔ CUDA 大版本对齐**：装 PyTorch 时选与镜像 CUDA 匹配的 wheel（如 cu121），混搭易出 `undefined symbol` / 加载失败。

> 排查口诀：容器内 `nvidia-smi` 顶部显示**驱动支持的 CUDA 上限**；`nvcc --version` 显示**镜像里的 CUDA 版本**。两者本就可以不同，前者≥后者即可。

## 典型流程 / 命令骨架（讲含义，不背默认值）

```bash
# 1) 拉镜像（具体可用 tag 看 NGC / Docker Hub 标签列表）
docker pull nvcr.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 2) 验证 GPU 能进容器（--gpus 由 container-toolkit 提供）
docker run --rm --gpus all \
       nvcr.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 \
       nvidia-smi
#   能列出 GPU 即说明：驱动注入成功 + 版本兼容

# 3) 只给容器特定卡（多卡机隔离）
docker run --rm --gpus '"device=0,1"' <镜像> nvidia-smi

# 4) 基于它构建自己的镜像
docker build -t my-llm:latest .
```

常用参数/字段的"作用"理解（不背默认值）：

| 项 | 作用 | 怎么权衡 |
|---|---|---|
| `--gpus all` / `device=...` | 声明容器可见哪些 GPU | 多租户/多任务时用 device 隔离，避免互相抢卡 |
| 变体 base/runtime/devel | 决定容器内有没有编译器/全套库 | 见第 3 节口诀 |
| tag 的 OS 段 | 决定包管理器（apt/yum/dnf）与 glibc | 选仍在维护的发行版 |
| `TORCH_CUDA_ARCH_LIST` | 编译期指定目标 GPU 架构 | 多卡型部署列全，单一型只列对应架构省时间 |
| 多阶段 `--from=builder` | 把编译产物搬进瘦镜像 | 生产镜像务必去掉 devel/编译器 |

## 常见问题 / 坑

| 现象 | 根因 | 处理方向 |
|---|---|---|
| 容器内 `nvidia-smi` 找不到设备 | 没装 NVIDIA Container Toolkit 或忘了 `--gpus` | 安装 toolkit；run 时加 `--gpus`；K8s 装 device plugin |
| `CUDA driver version is insufficient` | 宿主机驱动太旧，镜像 CUDA 太新 | 升驱动，或换更低 CUDA 版本的 tag（见第 7 节） |
| 用了 base 变体，框架报缺 cuBLAS/cuDNN | base 不含全套数学库/cuDNN | 换 runtime/devel，或在 base 上自行安装对应库 |
| 自编算子 `no kernel image is available` | 编译时没覆盖目标 GPU 架构 | 设 `TORCH_CUDA_ARCH_LIST` 包含部署卡架构 |
| 最终镜像几个 GB | 直接拿 devel 当生产镜像 | 改多阶段：devel 编译 → runtime 运行 |
| 拉 `nvcr.io` 镜像 401/找不到 | 未登录 NGC / tag 拼错或不存在 | 登录 NGC；到标签列表核对确切 tag |
| `pip install torch` 后加载报 `undefined symbol` | 框架 wheel 的 CUDA 版本与镜像不一致 | 选与镜像 CUDA 大版本匹配的 wheel（cu118/cu121…） |
| CentOS7 基底镜像装包失败 | 发行版 EOL，源失效 | 迁移到 Ubuntu/UBI 基底的 tag |

> 心法：**镜像 = 用户态 CUDA；驱动 = 宿主机注入；tag = 特征清单；变体 = 编译/运行二选一；版本 = 驱动≥CUDA≥框架对齐。** 任何精确版本号、确切 tag、包名都以官方 NGC / Docker Hub / CUDA 文档为准，不要凭记忆。

## 🔗 跳转链接

- [[00-知识地图]]
- [[ai-infra/ai-hardware/README]]
- [[ai-infra/ai-hardware/CUDA]]
- [[llmops/kubernetes]]
- [[llm-inference/tensorrt/README]]
