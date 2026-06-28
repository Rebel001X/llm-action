# PyTorch 安装

> PyTorch 安装的核心不是「pip install torch」一行命令，而是把「Python 版本 ↔ PyTorch 版本 ↔ CUDA 版本 ↔ GPU 驱动 ↔ GPU 算力（compute capability）」这条兼容链对齐——装错一环，要么导入失败、要么 `torch.cuda.is_available()` 返回 False、要么悄悄跑在 CPU 上。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/pytorch/README]] [[ai-framework/deepspeed/README]] [[ai-infra/ai-hardware/README]]

## 阅读地图

| 节 | 主题 | 你会得到什么 | 难度 |
|----|------|--------------|------|
| 0 | 一句话锚点 | 安装的本质是「对齐兼容链」 | ★ |
| 1 | 地基/前置 | 为什么不能直接 pip install torch | ★ |
| 2 | 兼容链五要素 | 驱动/CUDA Runtime/Toolkit/算力/Python | ★★★ |
| 3 | 三种安装路径 | pip wheel / conda / docker 怎么选 | ★★ |
| 4 | wheel URL 解剖 | 文件名每一段是什么意思 | ★★ |
| 5 | conda 安装 | `pytorch-cuda` 元包做了什么 | ★★ |
| 6 | NVIDIA NGC 镜像 | 为什么镜像最省心 | ★★ |
| 7 | 国内加速 | 镜像源/离线包 | ★ |
| 8 | 装完自检 | 怎么确认 GPU 真的能用 | ★★ |
| - | 配置示例 / FAQ / 跳转 | 实操与查漏 | ★ |

## 0. 一句话锚点

**安装 PyTorch = 选一个「PyTorch 版本 + 它自带的 CUDA Runtime 版本」组合，让它落在你机器「GPU 驱动 + GPU 算力」能支持的窗口里。**

最关键、也最反直觉的一点：

> **pip/conda 装的 PyTorch GPU 包，已经把 CUDA Runtime 库打包进去了**（那些 `cu118`、`cu121` 后缀就是它）。你**不需要**在系统里另外装 CUDA Toolkit，只需要一张**足够新的 GPU 驱动**。这是 90% 安装困惑的根源。

## 1. 地基/前置（它解决什么问题）

为什么不能无脑 `pip install torch` 就完事？因为 PyTorch 的 GPU 能力依赖一条**多层兼容链**，任何一层不匹配都会出问题：

```
   你的目标：python 里 torch.cuda.is_available() == True，且能用上 GPU

   可能的失败点：
   ① 装成了 CPU-only 版（pip 默认源可能给你纯 CPU 包）→ 永远 False
   ② PyTorch 自带的 CUDA Runtime 太新，但驱动太旧    → 报驱动版本不足
   ③ GPU 太新（如新架构），但 PyTorch 版本太旧不认识它 → "no kernel image"
   ④ Python 版本对不上 wheel（如 py3.12 找 cp310 包）  → 装不上/找不到
   ⑤ 在 CPU 上偷偷跑了（device 没切）                  → 慢得离谱但不报错
```

所以「安装」本质是一道**版本对齐题**。把这道题拆清楚，安装就不再玄学。

## 2. 兼容链五要素：把每一层讲清楚

这是全篇的地基。先把容易混的五个概念拆成原子：

```
   ┌─────────────────────────────────────────────────────────────┐
   │ ⑤ Python 版本     : wheel 文件名里的 cp310/cp311… 必须对上    │
   ├─────────────────────────────────────────────────────────────┤
   │ ④ PyTorch 版本    : 决定支持哪些 GPU 算力、哪个 CUDA Runtime  │
   ├─────────────────────────────────────────────────────────────┤
   │ ③ CUDA Runtime    : ★被打包进 pip/conda 的 torch 包里★        │
   │   (cu118/cu121…)    你装 torch 就附带了它，无需单独装          │
   ├─────────────────────────────────────────────────────────────┤
   │ ② GPU 驱动        : ★唯一必须由系统/宿主机提供的东西★         │
   │   (Driver)          驱动要足够新，才能支持上面的 CUDA Runtime  │
   ├─────────────────────────────────────────────────────────────┤
   │ ① GPU 硬件 + 算力  : compute capability(如 8.0/9.0)，物理固定 │
   └─────────────────────────────────────────────────────────────┘

   规则：上层依赖下层。装包时你选的是 ③④⑤，
        ①是硬件给定，②必须够新去「兜住」③。
```

### 2.1 区分「CUDA Toolkit」与「CUDA Runtime」（最大混淆点）

很多人以为「用 GPU 必须先装 CUDA Toolkit」，对 **pip/conda 安装的 PyTorch 来说这是错的**：

| 概念 | 是什么 | PyTorch 使用者需要单独装吗 |
|------|--------|---------------------------|
| **CUDA Toolkit** | 完整开发套件：`nvcc` 编译器、头文件、调试工具、库 | **通常不需要**（除非你要从源码编译扩展/算子） |
| **CUDA Runtime 库** | 运行 CUDA 程序所需的动态库（`libcudart` 等） | **不需要**——已随 torch 的 GPU wheel 打包 |
| **GPU 驱动 + 用户态 CUDA Driver** | 内核驱动 + `libcuda.so`，由 NVIDIA 驱动安装包提供 | **需要**——这是你系统里必须装的那一份 |

一句话：**驱动（②）由系统装，Runtime（③）随 torch 来，Toolkit（开发用）一般不碰。**

### 2.2 「驱动向后兼容」与 forward compatibility

NVIDIA 驱动对 CUDA Runtime 有**向后兼容**：一个较新的驱动能跑较旧的 CUDA Runtime。所以实践经验是——**驱动尽量装新**，就能兼容大多数 PyTorch 版本。

```
   驱动版本 ──支持──▶ 它能跑的最高 CUDA Runtime
   (越新越好)         (你装的 torch 的 cuXXX 必须 ≤ 这个上限)

   例：驱动较旧 → 只能跑到 cu118 → 那就别去装 cu124 的 torch
       驱动够新 → cu118/cu121/cu124 都能跑 → torch 版本随意挑
```

> 具体「哪个驱动版本对应哪个 CUDA 上限」以 NVIDIA 官方的 CUDA Toolkit Release Notes 兼容性表为准（见底部链接），不要凭记忆背数字。

### 2.3 算力（compute capability）：GPU 太新或太旧的坑

每块 NVIDIA GPU 有一个固定的**算力号**（compute capability，如 7.0、8.0、9.0），代表它的架构代次。PyTorch 的 GPU wheel 是为**一组算力**预编译好的（PTX/SASS）。

- **GPU 太新**：买了最新架构的卡，但用了发布在它之前的旧 PyTorch，里面没有对应算力的 kernel → 典型报错 `CUDA error: no kernel image is available for execution on the device`。**解法：升级 PyTorch 到支持该卡的版本。**
- **GPU 太旧**：很老的卡算力太低，新版 PyTorch 可能已经停止支持 → 只能用旧版 PyTorch。

```
   PyTorch 版本 ── 编译时锁定 ──▶ 支持的算力集合 {7.0, 7.5, 8.0, 8.6, 9.0, ...}
                                          │
   你的 GPU 算力 ──── 必须落在这个集合里 ──┘
```

## 3. 三种安装路径：怎么选

PyTorch 官方提供三条主要路径，定位不同：

```
   ┌──────────────┬───────────────────────────┬─────────────────────┐
   │ 方式         │ 适合谁 / 优点              │ 注意点              │
   ├──────────────┼───────────────────────────┼─────────────────────┤
   │ pip + wheel  │ 最常用、最灵活            │ 必须指定正确 index   │
   │ (index-url)  │ 精确控制版本+CUDA后缀     │ URL，否则装成 CPU 版 │
   ├──────────────┼───────────────────────────┼─────────────────────┤
   │ conda        │ 想让 conda 统一管 CUDA    │ 需 pytorch/nvidia    │
   │              │ 依赖、环境隔离干净        │ 渠道(channel)        │
   ├──────────────┼───────────────────────────┼─────────────────────┤
   │ docker(NGC)  │ 多机/生产/想省心          │ 镜像大；需 nvidia    │
   │              │ 驱动外其余全在镜像里      │ container toolkit    │
   └──────────────┴───────────────────────────┴─────────────────────┘
```

**选型直觉**：
- 单机做实验、要精确版本 → **pip wheel**（最主流）。
- 习惯 conda、想要干净隔离 → **conda**。
- 上集群/复现严格环境/CI → **NGC docker 镜像**（驱动以外一切都被冻结，复现性最好）。

> 获取「为你的版本组合定制的安装命令」最稳妥的来源是 PyTorch 官网首页的安装命令生成器（Get Started）和历史版本页：
> - 历史版本：https://pytorch.org/get-started/previous-versions/
> - wheel 文件索引：https://download.pytorch.org/whl/torch/

## 4. wheel URL 解剖：把文件名读懂

理解 wheel 文件名，就能自己判断「这个包合不合我的环境」。以仓库里出现的这条为例：

```
   torch-2.1.0+cu121-cp310-cp310-linux_x86_64.whl
   └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └────┬────┘
   包名  版本  CUDA后缀 Python ABI    平台
                       标签
```

逐段含义：

| 段 | 例 | 含义 / 你要核对什么 |
|----|----|--------------------|
| 包名 | `torch` | 还有 `torchvision`/`torchaudio`，三者版本要**配套** |
| 版本 | `2.1.0` | PyTorch 主版本，决定支持的算力与特性 |
| CUDA 后缀 | `+cu121` | **自带的 CUDA Runtime ≈ 12.1**；`+cpu` 表示纯 CPU 包 |
| Python 标签 | `cp310` | 给 **CPython 3.10** 的；和你的 `python --version` 必须一致 |
| ABI 标签 | `cp310` | 二进制接口标签，一般同上 |
| 平台 | `linux_x86_64` | 操作系统+架构；Windows 是 `win_amd64`，Mac 另有标签 |

> 关键认知：**`+cu121` 不是要你系统装 CUDA 12.1，而是「这个包内部带了 12.1 的 Runtime」**——你只要驱动够新能跑 12.1 即可。`torchvision`/`torchaudio` 必须选**和 torch 同一行**的配套版本，否则会出现符号/ABI 不匹配。

### 三件套版本配套

`torch`、`torchvision`、`torchaudio` 是绑定发布的，版本号有固定对应关系（不是随便组合）：

```
   torch 2.1.0  ──配套──  torchvision 0.16.0  +  torchaudio 2.1.0
   torch 2.2.0  ──配套──  torchvision 0.17.0  +  torchaudio 2.2.0
   （对应关系以官方历史版本页为准，不要跨行混搭）
```

## 5. conda 安装：`pytorch-cuda` 元包做了什么

conda 路径的命令长这样（仓库里已有示例）：

```
   conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
                 pytorch-cuda=11.8 -c pytorch -c nvidia
```

拆解每个部分在干什么：

```
   pytorch==2.2.0 …      → 指定三件套版本（和 pip 同理，要配套）
   pytorch-cuda=11.8     → ★元包(meta-package)★：声明「我要 CUDA 11.8 这套」
                           它会把对应的 CUDA Runtime 依赖一并拉进环境
   -c pytorch -c nvidia  → 渠道(channel)：去 pytorch 和 nvidia 官方频道找包
                           顺序有意义：靠前的渠道优先级更高
```

**conda vs pip 的差别**：conda 把 CUDA Runtime 当成**可被求解的依赖**（用 `pytorch-cuda` 元包声明），它会在你的 conda 环境里放一份运行库，环境隔离更彻底；pip 则是把 Runtime 直接打进 wheel。两者最终效果类似（都不需要系统级 CUDA Toolkit），区别在依赖管理方式。

> 注意：不同时期 conda 的渠道与元包名称可能调整（例如官方渠道策略变化）。具体当下应使用的命令，以 PyTorch 官网生成器给出的为准。

## 6. NVIDIA NGC 镜像：为什么最省心

NVIDIA 在 NGC（NVIDIA GPU Cloud）提供官方 PyTorch 容器镜像：

```
   - 镜像目录：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
   - 拉取示例：nvcr.io/nvidia/pytorch:24.05-py3
                                      └──┬──┘
                            标签 = 年.月（如 24.05 表示 2024 年 5 月那批）
                            每个标签里 PyTorch/CUDA/cuDNN/NCCL 版本都已固定且调优
```

**镜像把兼容链的 ③④⑤ 全冻结好了**，宿主机只需提供 ②驱动（且通过 nvidia-container-toolkit 把 GPU 透传进容器）。运行逻辑：

```
   ┌──────────────────────────── 宿主机 (Host) ────────────────────────────┐
   │  GPU 硬件①  +  NVIDIA 驱动②  +  nvidia-container-toolkit              │
   │                              │ 透传 /dev/nvidia* 与 libcuda           │
   │     ┌────────────────────────▼──────────────────────────────┐        │
   │     │  容器: nvcr.io/nvidia/pytorch:24.05-py3                │        │
   │     │  内含: PyTorch + CUDA Runtime③ + cuDNN + NCCL + Python │        │
   │     │  → 环境完全冻结，换机器只要驱动够新就能跑              │        │
   │     └───────────────────────────────────────────────────────┘        │
   └───────────────────────────────────────────────────────────────────────┘
```

**适用场景**：多机训练、生产部署、严格复现实验、CI。代价是镜像体积大（数 GB～十几 GB），且仍要求宿主机驱动足够新以兜住镜像里的 CUDA Runtime。

> 运行需要 `--gpus all`（或等价的 runtime 配置）把 GPU 暴露给容器。具体 docker run 参数以 NVIDIA 容器文档为准。

## 7. 国内加速

国内拉官方源常常慢或超时，几种常见加速手段（讲机制，不替你背具体地址）：

| 手段 | 机制 | 注意点 |
|------|------|--------|
| pip 镜像源 | 用国内 PyPI 镜像替换默认 index 拉**纯 Python 依赖** | 但 GPU 版 torch 本体往往不在 PyPI，需配合下面两条 |
| PyTorch 官方下载站 | 直接从 `download.pytorch.org/whl/cuXXX/` 找对应 wheel | URL 命名规则见第 4 节，可手动下载离线安装 |
| 离线 wheel 安装 | 先下好 `.whl` 文件，再 `pip install ./xxx.whl` | 适合无外网的内网机/集群；记得三件套一起下 |
| docker 镜像加速 | 配置容器镜像加速器拉 NGC/基础镜像 | 大镜像首拉一次，后续靠缓存 |

> 镜像源/加速器地址会变动，且各高校/云厂商不同，用前先确认当前可用地址；不要把过期地址写死进脚本。

## 8. 装完自检：确认 GPU 真的能用

安装完成后，**最重要的一步**是验证——不要假设它能用 GPU。一个最小自检清单：

```python
   import torch
   print(torch.__version__)            # 看版本，确认不是 +cpu 的纯 CPU 包
   print(torch.version.cuda)           # torch 自带的 CUDA Runtime 版本(如 12.1)
   print(torch.cuda.is_available())    # ★关键★：True 才说明 GPU 通路打通
   print(torch.cuda.device_count())    # 能看到几张卡
   print(torch.cuda.get_device_name(0))# 卡型号，确认认到了正确的卡

   # 真正跑一次 GPU 计算，确保不是「看得见但用不了」
   x = torch.rand(1000, 1000, device='cuda')
   y = x @ x                           # 不报错说明算力/kernel 匹配
```

排障决策树：

```
   torch.cuda.is_available() ?
        │
        ├─ False ──┬─ torch.__version__ 带 +cpu  → 装错成 CPU 版，重装 GPU wheel
        │          ├─ 驱动太旧/没装             → nvidia-smi 看驱动，升级驱动
        │          └─ 容器没透传 GPU            → 加 --gpus all / 配 runtime
        │
        └─ True ──── 跑 x@x 报 "no kernel image" → GPU 太新，升级 PyTorch 版本
```

> `nvidia-smi`（驱动自带工具）能看到**驱动支持的最高 CUDA 版本**和 GPU 状态——它显示的 CUDA 版本是「驱动能兜住的上限」，**不等于** PyTorch 实际用的 Runtime 版本（后者看 `torch.version.cuda`）。这俩经常被混为一谈。

## 配置示例 / 参数说明

把前面拼起来，一个「从零到能用」的思路流水线（具体命令以官网生成器为准）：

```
   ① nvidia-smi          → 确认驱动已装、看驱动支持的 CUDA 上限
        │ 驱动太旧？ → 先升级 GPU 驱动（系统级，唯一必须装的）
        ▼
   ② 确定 Python 版本     → python --version，决定 wheel 的 cpXXX 标签
        ▼
   ③ 去 PyTorch 官网生成器 → 选 OS / 包管理器 / CUDA 后缀，得到安装命令
        │   - CUDA 后缀(cuXXX)必须 ≤ 驱动能兜住的上限
        │   - GPU 很新时，PyTorch 版本要够新（覆盖该算力）
        ▼
   ④ pip / conda / docker → 三选一执行安装（三件套版本配套）
        ▼
   ⑤ 跑第 8 节自检脚本    → is_available()==True 且 x@x 不报错 → 完成
```

关键「参数/选项」在做什么（讲含义，不背默认值）：

| 选项/概念 | 作用 | 怎么权衡 |
|-----------|------|----------|
| pip `--index-url .../cuXXX` | 指定从带某 CUDA 后缀的官方源拉包 | **漏了它最容易装成 CPU 版**；cuXXX 选驱动能兜住的 |
| `+cpu` 后缀 | 纯 CPU 包，无 GPU 能力 | 只在无 GPU/调试时用；有卡千万别装它 |
| conda `pytorch-cuda=X.Y` | 元包，声明要哪套 CUDA Runtime | 值要 ≤ 驱动上限；走 conda 路径才用 |
| `-c pytorch -c nvidia` | conda 渠道及优先级 | 顺序影响求解，靠前优先 |
| NGC 镜像标签 `YY.MM-py3` | 冻结一整套环境 | 越新覆盖越新硬件，但更吃驱动新度 |
| `torch.utils.collect_env` | 一键打印环境兼容信息 | 提问/排障时贴它最高效 |

## 常见问题 / 坑

| 问题 / 现象 | 根因 | 处理方向 |
|-------------|------|----------|
| `cuda.is_available()` 一直 False | 装成了 `+cpu` 包，或没用对 index-url | 看 `torch.__version__` 有无 `+cpu`，重装 GPU wheel |
| `no kernel image is available` | GPU 算力太新，当前 PyTorch 不支持 | 升级 PyTorch 到覆盖该卡算力的版本 |
| 报驱动版本不足 | 驱动太旧，兜不住 torch 自带的 CUDA Runtime | 升级 GPU 驱动，或改装更低 cuXXX 后缀的 torch |
| `torchvision` 导入报符号错误 | 三件套版本没配套 | 按官方对应表选同一行的 vision/audio 版本 |
| pip 装不上 / 找不到包 | Python 版本与 wheel 的 `cpXXX` 不符 | 对齐 Python 版本，或换有对应包的 torch 版本 |
| 容器里 `is_available()` False | GPU 没透传进容器 | 加 `--gpus all`、装 nvidia-container-toolkit |
| `nvidia-smi` 的 CUDA ≠ `torch.version.cuda` | 前者是驱动上限，后者是 torch 实际 Runtime | 二者不必相等，只要 torch 的 ≤ 驱动上限即可 |
| 我必须先装 CUDA Toolkit 吗？ | 误解 | pip/conda 装 GPU torch **无需**单独装 Toolkit，只需驱动 |
| 离线机装不上依赖 | 无外网拉 Python 依赖 | 三件套 `.whl` 连同依赖一起下好，离线 `pip install` |
| 多卡只认到 1 张 | `CUDA_VISIBLE_DEVICES` 限制或驱动/拓扑问题 | 检查该环境变量、`nvidia-smi` 是否都在 |

> 反复强调的一条主线：**驱动够新（系统装）→ 选对 cuXXX 后缀（torch 自带 Runtime）→ Python 标签对上 → 三件套配套 → 自检确认**。把这条链走通，安装就稳了。具体版本号、CLI 参数、对应关系一律以官方文档/生成器为准，不要凭记忆硬写。

## 🔗 跳转链接

- [[00-知识地图]] — 全局知识地图，从这里找其他主题
- [[ai-framework/pytorch/README]] — PyTorch 核心机制：动态图/autograd/DDP/FSDP/torch.compile
- [[ai-framework/deepspeed/README]] — ZeRO 分片与并行的工程化（同样依赖正确的 PyTorch+CUDA 环境）
- [[ai-infra/ai-hardware/README]] — GPU 硬件、算力与互联，理解兼容链的硬件底座

### 官方参考（以这些为准，不要凭记忆）

- PyTorch 历史版本安装命令：https://pytorch.org/get-started/previous-versions/
- PyTorch wheel 文件索引：https://download.pytorch.org/whl/torch/
- NGC PyTorch 容器镜像：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- CUDA 与驱动兼容性（Release Notes）：https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
