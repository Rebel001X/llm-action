# Conda 环境与包管理（从底层机制讲清）

> 一句话定位：Conda 是一个**跨语言、跨平台的二进制包管理器 + 环境隔离器**，AI-Infra 里用它把"Python 版本 + CUDA 运行时 + PyTorch + C/C++ 依赖"打包成一个互不污染的沙盒。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/ai-hardware/CUDA]] · [[llm-train/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|-------------|--------|
| 0 | 一句话锚点：Conda 到底解决什么 | 隔离 / 二进制 / 跨语言 |
| 1 | 地基：为什么 `pip + venv` 在 AI 场景不够用 | ABI / CUDA / 共享库 |
| 2 | 三个核心对象：channel / package / environment | repodata / 沙盒 |
| 3 | 环境隔离的真相：只是改了 `PATH` | 软链接 / 前置目录 |
| 4 | SAT 依赖求解器：为什么 `conda install` 那么慢 | 布尔可满足 / libmamba |
| 5 | conda vs pip vs mamba vs uv 怎么选 | 混用陷阱 |
| 6 | AI-Infra 实战：装一套 PyTorch+CUDA 栈 | cudatoolkit / 驱动 |
| 7 | 数值手算：磁盘/求解规模估算 | 硬链接去重 |
| 8 | 工程化：environment.yml / lock / 复现 | 可复现构建 |
| — | 常见问题 + 跳转链接 | FAQ |

---

## 0. 一句话锚点

> **Conda = "给每个项目发一个独立的小型 Linux 用户空间"**，里面有自己的 `python`、自己的 `libstdc++`、自己的 `cudatoolkit`，彼此不打架。它管理的不是源码而是**预编译好的二进制**，所以装 PyTorch 不用本地编译。

记住一个区分：

- **`pip`**：只懂 Python 包（wheel/sdist），不管系统级的 C 库。
- **`conda`**：懂任何语言的二进制（python 解释器本身、`zlib`、`mkl`、`cudatoolkit`、`ffmpeg`……），它是**通用包管理器**，Python 只是它管理的其中一个包。

```
        管理范围对比
  pip ─────────────┐
  │ numpy  torch   │  ← 只是 Python 包
  │ (wheel)        │
  └────────────────┘
  conda ──────────────────────────────────┐
  │ python本身  libstdc++  mkl  cudatoolkit│  ← 连解释器和 C 库都管
  │ openssl  ffmpeg  numpy  torch          │
  └────────────────────────────────────────┘
```

---

## 1. 地基：为什么 AI 场景非要用 Conda

### 1.1 痛点一：Python 版本本身要隔离

系统自带 `python3.10`，但项目 A 要 3.9、项目 B 要 3.12。`venv` 只能复用**系统已装的**某个解释器，它不能"给你装一个 3.9"。Conda 把 **python 解释器当成一个普通包**，所以可以 `conda create -n a python=3.9`、`conda create -n b python=3.12`，互不影响。

### 1.2 痛点二：ABI / 共享库地狱

AI 包大量依赖 C/C++/Fortran 编译产物（BLAS、CUDA kernel）。这些 `.so` 之间存在 **ABI（二进制接口）兼容性**约束：

```
   torch.so ──需要──> libstdc++.so.6 (GLIBCXX_3.4.30)
        │
        └──需要──> libcudart.so.12 (CUDA 12.x runtime)
        │
        └──需要──> libgomp.so.1 (OpenMP)
```

如果系统的 `libstdc++` 太老（`GLIBCXX_3.4.30 not found` 这种经典报错），pip 装的 wheel 就跑不起来。Conda 把 **这些 C 库也打进环境**，并保证版本互相匹配——这是 pip 做不到的核心价值。

### 1.3 痛点三：CUDA 运行时与驱动解耦

这是 AI-Infra 最关键的一点，必须讲透（见 [[ai-infra/ai-hardware/CUDA]]）：

```
  ┌─────────────────────────────────────────────┐
  │ 用户态（可放进 conda 环境）                   │
  │   cudatoolkit / cuda-runtime  (libcudart)     │  ← conda 装这层
  │   cuDNN / NCCL / cuBLAS                        │
  ├─────────────────────────────────────────────┤
  │ 内核态（必须装在宿主机，conda 管不了）         │
  │   NVIDIA Driver  (libcuda.so, nvidia.ko)      │  ← 宿主机/容器外
  └─────────────────────────────────────────────┘
```

**规则**：驱动（`libcuda.so`）向后兼容旧的 runtime。所以**驱动只要够新**，你可以在不同 conda 环境里装 CUDA 11.8 / 12.1 / 12.4 的 runtime 共存。`nvidia-smi` 显示的是**驱动支持的最高 CUDA 版本**（CUDA Driver API），`nvcc --version` / `torch.version.cuda` 显示的是**当前环境里 runtime 的版本**——两者不一定相等，这是新手最大的困惑点。

---

## 2. 三个核心对象：channel / package / environment

### 2.1 Package（包）

一个 conda 包是一个 `.conda`（或老格式 `.tar.bz2`）压缩档，里面是**已经编译好**的文件树 + 元数据：

```
  pytorch-2.3.0-py3.10_cuda12.1.conda
  ├── info/
  │   ├── index.json      ← name, version, build, depends[]
  │   ├── paths.json      ← 每个文件的 sha256 + 是否需要前缀替换
  │   └── about.json
  └── lib/python3.10/site-packages/torch/...  ← 实际内容
```

包名结构：`name - version - build_string`，例如 `pytorch-2.3.0-py3.10_cuda12.1_cudnn8.9.2_0`。**build string 把编译时的关键变量（py 版本、cuda 版本）编码进去**，这就是同名包能区分 CUDA 变体的原因。

### 2.2 Channel（频道 / 软件源）

Channel 是包的托管仓库，本质是一个 HTTP 目录，核心文件是 **`repodata.json`**——它列出该频道某平台下所有包及其依赖关系（一个巨大的索引）。

```
  https://conda.anaconda.org/conda-forge/linux-64/repodata.json
                            └─ channel ─┘ └subdir┘
  subdir（平台）：linux-64 / linux-aarch64 / win-64 / osx-arm64 / noarch
```

常见频道：

| Channel | 说明 | 注意 |
|---------|------|------|
| `defaults` | Anaconda 官方（含 `pkgs/main`） | 商业使用可能涉及条款，以官方为准 |
| `conda-forge` | 社区维护，包最全最新 | AI-Infra 首选 |
| `pytorch` | PyTorch 官方频道 | 装 torch 常用 |
| `nvidia` | CUDA 相关包 | `cuda-toolkit` 等 |

> ⚠️ **频道混用陷阱**：从 `defaults` 和 `conda-forge` 各拿一半依赖，容易因 ABI 不一致冲突。推荐用 `channel_priority: strict`，让求解器优先在高优先级频道内闭合所有依赖。

### 2.3 Environment（环境）

一个环境就是磁盘上的一个目录（`envs/<name>/`），里面有完整的 `bin/`、`lib/`、`include/`。下一节讲清它的隔离机制其实非常朴素。

---

## 3. 环境隔离的真相：本质只是改了 PATH

很多人以为 conda 环境是虚拟机或容器——**不是**。它就是一个目录，激活 = 把这个目录的 `bin/` 插到 `PATH` 最前面。

```
  未激活 PATH:
    /usr/local/bin : /usr/bin : /bin
       which python → /usr/bin/python (系统的)

  conda activate myenv 之后:
    ~/miniconda3/envs/myenv/bin : /usr/local/bin : /usr/bin : /bin
    └──────── 被插到最前面 ────────┘
       which python → ~/miniconda3/envs/myenv/bin/python  ✅
```

`conda activate` 这个 shell 函数做的事：
1. 把 `envs/myenv/bin` 前置到 `$PATH`；
2. 设置 `CONDA_PREFIX=~/miniconda3/envs/myenv`（让 Python 知道去哪找包）；
3. 运行 `etc/conda/activate.d/` 下的脚本（例如设 `CUDA_HOME`、`LD_LIBRARY_PATH`）。

```
  环境目录结构（沙盒的全部）
  envs/myenv/
  ├── bin/        python, pip, nvcc, ...   ← 前置到 PATH
  ├── lib/        libstdc++.so, libcudart.so, python3.10/site-packages/
  ├── include/    头文件
  ├── conda-meta/ 已装包的清单（每包一个 .json）
  └── etc/conda/activate.d/   激活钩子脚本
```

> 关键洞察：**隔离来自"查找顺序"，不是来自命名空间或 cgroup**。这也解释了为什么 `LD_LIBRARY_PATH` 设错会"穿透"环境，把系统的 `.so` 链进来——因为动态链接器的查找顺序被你手动改了。

---

## 4. 依赖求解：为什么 `conda install` 会"卡很久"

### 4.1 这是一个 SAT（布尔可满足性）问题

装一个包，要在所有频道里挑出**一组互相兼容的版本**，使每个包的 `depends` 约束都被满足。这等价于求解一个布尔约束系统：

设每个"包-版本"是一个布尔变量 $x_{p,v}$（选/不选），约束形如：

- **唯一性**：同一个包最多选一个版本，$\sum_v x_{p,v} \le 1$。
- **依赖**：若选了 $A$ 的某版本且它要求 `B >=1.2`，则必须选满足该范围的某个 $B$ 版本。
- **目标**：在可行解里选"版本尽量新"的最优解。

这正是 NP 难的 SAT/优化问题。频道里有几万个包、每包几十个版本，搜索空间是天文数字——这就是经典 `conda` 慢的根源。

```
   求解器在做的事（示意）
   torch>=2.3 ─┐
               ├─ 必须同时满足 ──> cuda-runtime=12.1
   numpy ──────┤                   python=3.10
   torchvision─┘                   libstdc++(GLIBCXX>=3.4.30)
        ↑ 任一不满足就回溯，换一组版本重试
```

### 4.2 libmamba：把求解器换成更快的引擎

`mamba`（以及新版 conda 内置的 **libmamba solver**）用 C++ 实现的 SAT 求解器（基于 `libsolv`，源自 openSUSE 的 zypper），比老 Python 求解器快一个数量级。

```bash
# 新版 conda 可设为默认（机制：换求解器，不换包）
conda config --set solver libmamba   # 以官方文档为准
```

记忆口诀：**慢在"求解"，不慢在"下载"**。求解是 CPU 密集的约束搜索；libmamba 优化的就是这一步。

---

## 5. conda / pip / mamba / uv 各自定位

| 工具 | 管什么 | 求解器 | 何时用 |
|------|--------|--------|--------|
| `conda` | 任意二进制（含 python 本身、C 库、CUDA） | 内置（可切 libmamba） | 需要非 Python 依赖、CUDA 栈 |
| `mamba` | 同 conda（兼容命令） | libmamba（C++，快） | 想要更快的 conda 体验 |
| `pip` | Python 包（wheel/sdist） | 较弱的回溯求解 | 装纯 Python 库 / PyPI 独有包 |
| `uv` | Python 包（Rust 实现，极快） | 现代求解器 | 纯 Python 项目，替代 pip/venv |

### 5.1 conda 与 pip 混用的正确姿势

混用是常态（很多包只在 PyPI 有），但有**铁律**：

```
  正确顺序：先 conda，后 pip
  ┌────────────────────────────────────┐
  │ 1. conda 装好"地基"：               │
  │    python, cudatoolkit, numpy, mkl  │
  │ 2. 再 pip 装"上层"PyPI-only 的包    │
  │ 3. 之后不要再 conda install         │
  │    （否则 conda 可能覆盖/降级 pip包）│
  └────────────────────────────────────┘
```

原因：conda 的求解器**看不懂 pip 装的包**（它们不在 `conda-meta/` 里），所以后续 `conda install` 可能误判依赖、把 pip 装的 torch 覆盖掉。

---

## 6. AI-Infra 实战：装一套训练栈

目标：`python3.10 + PyTorch 2.x + CUDA 12.1`，用于跑 [[llm-train/README]] 里的训练或 [[ai-framework/deepspeed/README]]。

```bash
# 1) 建环境，python 当作包指定版本
conda create -n train python=3.10 -y
conda activate train

# 2) 装 PyTorch（用官方推荐频道，具体命令以 pytorch.org 为准）
conda install pytorch torchvision pytorch-cuda=12.1 \
      -c pytorch -c nvidia -y
```

验证三件套（理解它们各自的含义）：

```
  nvidia-smi        → 看【驱动】支持的最高 CUDA（Driver API）
  python -c "import torch; print(torch.version.cuda)"
                    → 看【环境内 runtime】编译的 CUDA 版本
  python -c "import torch; print(torch.cuda.is_available())"
                    → True 说明 runtime 能调通驱动
```

```
   能否跑起来的判定逻辑
   torch.version.cuda (12.1)  ≤  驱动支持的CUDA (如 12.4)?
                    │
            是 ─────┴───── 否
            │              │
       cuda.is_available   报错: driver too old
        = True  ✅          → 升级宿主机驱动
```

> 多卡通信（NCCL）相关见 [[ai-infra/网络/集合通信原语]]；conda 装的 torch 通常自带匹配的 NCCL，无需手动装。

---

## 7. 数值手算

### 7.1 硬链接去重：N 个环境为什么不是 N 倍磁盘

Conda 把包先解压到全局 **`pkgs/` 缓存**，再用**硬链接**（同一 inode）链进各环境，而不是复制。

```
  pkgs/numpy-1.26.../   ← 真实文件，占 30 MB（inode #1)
        │ hardlink        │ hardlink
        ▼                 ▼
  envs/A/.../numpy   envs/B/.../numpy   ← 共享同一 inode，几乎 0 额外磁盘
```

**手算**：假设 `numpy` 占 30 MB，10 个环境都用同一版本。

- 朴素复制：$30 \times 10 = 300$ MB。
- 硬链接：$30 \times 1 = 30$ MB（外加每个环境的元数据，忽略不计）。
- **节省**：$300 - 30 = 270$ MB，省下约 $\frac{270}{300}=90\%$。

> 前提：环境与 `pkgs/` 在**同一文件系统**（硬链接不能跨设备）。若把 envs 放到另一块盘，就退化成复制，磁盘暴涨——这是"conda 占满磁盘"的常见原因。

### 7.2 求解空间规模感受

设某频道有 $P=8000$ 个相关包，每包平均 $V=20$ 个版本。朴素枚举组合数：

$$
V^{P} = 20^{8000}
$$

是个天文数字。当然求解器靠约束传播 + 单元传播大幅剪枝，但这直观说明**为什么不能暴力枚举、必须用 SAT 求解器**，也说明 libmamba 的工程价值。

### 7.3 repodata 下载量估算

`conda-forge/linux-64/repodata.json` 体量约**数百 MB**（未压缩）。首次 `conda install` 需下载并解析它：

- 若解析 200 MB JSON，按 100 MB/s 磁盘 + 解析开销，单是"读懂频道有哪些包"就要数秒到十几秒。
- 这就是为什么**第一次**操作明显比后续慢（之后有缓存）。libmamba 还支持只下增量/分片索引来缓解。

---

## 8. 工程化：可复现环境

### 8.1 environment.yml（人写的"意图"）

```yaml
name: train
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.3.*
  - pytorch-cuda=12.1
  - pip
  - pip:
      - transformers==4.41.0   # PyPI-only 的放这里
```

```bash
conda env create -f environment.yml      # 创建
conda env update -f environment.yml --prune  # 更新并删去不再需要的包
```

### 8.2 三种"导出"的区别（复现的关键）

```
  conda env export                → 含 build string，最精确但只能同平台还原
  conda env export --no-builds    → 去掉 build，跨平台友好但不够确定
  conda list --explicit / lockfile→ 钉死每个包的 URL+哈希，真正可复现
```

```
   可复现性 ↑          可移植性 ↑
   lockfile  ───────────────  --no-builds
   (URL+sha256)              (只列 name=ver)
       │                          │
   同机器同平台              换平台也大概能装
   100% 复现                 但版本可能漂移
```

> 生产/CI 强烈建议用 **lock 文件**（如 `conda-lock` 工具，以官方为准）：它把求解结果固化成"每包精确 URL + 哈希"，杜绝"我这能跑你那不行"。理念同 [[llm-train/README]] 里强调的实验可复现。

### 8.3 清理与瘦身

```bash
conda clean --all       # 清 pkgs/ 缓存与索引，释放磁盘
conda env remove -n old # 删环境（只是删目录 + 解硬链接）
```

---

## 常见问题

| 问题 | 根因 | 处理思路 |
|------|------|---------|
| `nvidia-smi` 显示 CUDA 12.4，但 torch 装的是 12.1，能用吗？ | 二者一个是驱动 API、一个是 runtime | 能用，runtime ≤ 驱动支持版本即可（§6） |
| `GLIBCXX_3.4.30 not found` | 系统 `libstdc++` 太老 | conda 装 `libstdcxx-ng` 或整体在 conda 内闭环 |
| `conda install` 卡在 Solving environment | SAT 求解慢 | 切 libmamba 求解器（§4.2） |
| 装完 pip 包后 conda 又把它降级了 | conda 看不懂 pip 装的包 | 遵守"先 conda 后 pip、之后不再 conda install"（§5.1） |
| 磁盘被 conda 占满 | envs 与 pkgs 跨文件系统失去硬链接 / 缓存堆积 | 同盘放置 + `conda clean --all`（§7.1） |
| 频道混用报冲突 | defaults 与 conda-forge ABI 不一致 | `channel_priority: strict`（§2.2） |
| `cuda.is_available()` 为 False | 驱动太老 / runtime 与驱动不匹配 / 容器没透传 GPU | 查驱动版本、`--gpus all`（§6） |
| 想要更快的 conda | Python 求解器慢 | 用 `mamba` 或 libmamba（§4、§5） |

---

## 🔗 跳转链接

- [[00-知识地图]]
- [[ai-infra/ai-hardware/CUDA]] — CUDA runtime / 驱动 / nvcc 关系（本文 §1.3 §6 的硬件视角）
- [[ai-infra/算力/GPU工作原理]] — GPU 体系结构，理解为何要装 cudatoolkit
- [[ai-infra/网络/集合通信原语]] — 多卡 NCCL，与 conda 装的 torch 配套
- [[llm-train/README]] — 训练流程，可复现环境是前提
- [[ai-framework/deepspeed/README]] — DeepSpeed 安装常需 conda 管 C++ 依赖
- [[ai-framework/megatron-lm/README]] — Megatron-LM 环境搭建
- [[llm-inference/README]] — 推理部署同样需要隔离的运行环境
