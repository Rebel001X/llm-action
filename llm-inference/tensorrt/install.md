# TensorRT 安装

> NVIDIA TensorRT 的安装：从"装的到底是哪几块东西、为什么版本要严格对齐 CUDA/cuDNN、tar 包/pip/deb 三种方式怎么选"讲到逐行解释脚本与避坑。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/tensorrt/README]] [[llm-inference/README]] [[llm-compression/quantization/量化基础]] [[ai-infra/ai-hardware/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|--------------|--------|
| 0 | 一句话锚点 | 装什么 |
| 1 | TensorRT 是什么、安装为什么"麻烦" | 引擎/版本矩阵 |
| 2 | 安装包里到底有哪几块 | lib/python/工具 |
| 3 | 三种安装方式对比 | tar/deb/pip |
| 4 | 版本对齐：CUDA/cuDNN/驱动/Python | 兼容矩阵 |
| 5 | tar 包安装逐行拆解 | 解压/LD_LIBRARY_PATH/wheel |
| 6 | Python 三件套：full/lean/dispatch | runtime 形态 |
| 7 | 验证安装 | import/Builder |
| 8 | Torch-TensorRT 与生态 | 上层框架 |
| 9 | 典型流程图 + 配置示例 | 全景 |
| — | 常见坑表 | 排错 |

## 0. 一句话锚点

**TensorRT 安装 = 把"一组 C++ 推理优化库（`.so`）+ Python 绑定（wheel）+ 一堆解析/转换工具"放到机器上，并让操作系统和 Python 都能找到它们**。难点不在"装"，而在**版本对齐**：TensorRT 的某个版本，是为**特定的 CUDA、cuDNN、驱动、Python、GPU 架构**编译的，错一环就报 `undefined symbol` 或 `import error`。

> ⚠️ 本文用脚本里出现的 `8.6.x` / `CUDA 11.8` / `cp310` 等只作**示例占位**。真实安装时，**确切版本号、wheel 文件名、目录名一律以你下载到的安装包和 [NVIDIA 官方 Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/) 为准**，不要照抄本文数字。

## 1. 地基：TensorRT 是什么，安装为什么"挑环境"

### 1.1 它解决什么问题

TensorRT 是 NVIDIA 的**推理（inference）优化与运行时（runtime）SDK**。训练框架（PyTorch/TF）产出的模型直接跑推理，往往**慢且显存浪费**。TensorRT 做的事：

- **图优化**：算子融合（Conv+BN+ReLU 合一）、消除冗余、常量折叠；
- **精度降级**：FP32 → FP16 / INT8 / FP8，配合校准（calibration）；
- **kernel 自动选择（auto-tuning）**：在你这块具体的 GPU 上实测多种 kernel 实现，挑最快的；
- **生成 engine（plan 文件）**：一个**为这台机器/这块卡专门编译**的推理计划。

```
   训练框架模型                TensorRT                   部署
 ┌──────────────┐   解析    ┌────────────────────┐  序列化  ┌──────────┐
 │ ONNX / Torch │ ───────▶ │ 图优化 + 精度 + 选 │ ───────▶ │ engine   │
 │ / TF 模型    │          │ kernel(auto-tune)  │          │ (.plan)  │
 └──────────────┘          └────────────────────┘          └──────────┘
                                  ▲                              │
                                  │ 依赖 CUDA / cuDNN / cuBLAS    │ 加载执行
                                  └──────────────────────────────┘
```

### 1.2 为什么安装"挑环境"

因为 engine 是"为具体硬件编译"的，TensorRT 的二进制本身也**深度绑定底层栈**：

> **TensorRT 版本 ⟶ 需要特定 CUDA 版本 ⟶ 需要特定 cuDNN ⟶ 需要足够新的 NVIDIA 驱动 ⟶ 需要匹配的 Python ABI（cp310 等）**

任何一环不匹配，常见症状是 `ImportError: libnvinfer.so.X: cannot open shared object file` 或 `undefined symbol`。所以**安装的一半工作量是"对版本"**，详见第 4 节。

## 2. 安装包里到底有哪几块

把 TensorRT 想成"**一个核心 + 几圈外围**"。理解这点，第 5 节那串 `pip install` 才不再是"魔法咒语"。

```
                         TensorRT 安装目录 (TensorRT-<ver>/)
   ┌───────────────────────────────────────────────────────────────┐
   │  lib/          ← 核心 C++ 库 (.so)：libnvinfer / libnvonnxparser │
   │                  …这是"心脏"，必须能被动态链接器找到            │
   │  include/      ← C++ 头文件（写 C++ 程序才用）                   │
   │  bin/          ← 命令行工具：trtexec（最常用的基准/转换工具）    │
   │  python/       ← Python 绑定 wheel：tensorrt / lean / dispatch  │
   │  uff/          ← (旧) TF UFF 解析工具，新版基本弃用             │
   │  graphsurgeon/ ← (旧) 图编辑工具                                │
   │  onnx_graphsurgeon/ ← ONNX 图手术工具（裁剪/改图，仍常用）       │
   │  samples/      ← 官方示例                                        │
   │  data/         ← 示例数据                                        │
   └───────────────────────────────────────────────────────────────┘
```

三类东西、三种"装法"：

| 类别 | 内容 | "安装"意味着 |
|------|------|--------------|
| C++ 运行库 | `lib/*.so` | 让**动态链接器**找到 → 设 `LD_LIBRARY_PATH` 或装进系统库路径 |
| Python 绑定 | `python/*.whl` | `pip install` 进当前 venv |
| 辅助工具 | uff/graphsurgeon/onnx_graphsurgeon wheel | `pip install`，按需 |

**关键认知**：`pip install tensorrt-*.whl` 装的只是 **Python 调用层**；它最终还是要 `dlopen` 那些 `.so`。如果 `.so` 找不到，Python 端 `import tensorrt` 就崩——这就是为什么脚本里**既要 pip 又要 export LD_LIBRARY_PATH**。

## 3. 三种安装方式：tar / deb(rpm) / pip，怎么选

| 方式 | 形态 | 优点 | 缺点 | 适合 |
|------|------|------|------|------|
| **tar 包** | 解压即用，路径自己管 | 不污染系统、可多版本共存、最可控 | 要手动设 `LD_LIBRARY_PATH`、自己装 wheel | 服务器/共享机/要多版本（本脚本用的就是它） |
| **deb / rpm** | 系统包管理器装 | 自动进系统库路径、依赖自动拉 | 需 root、一台机一个版本、与系统 CUDA 绑死 | 独占机、跟系统 CUDA 配套 |
| **pip（`pip install tensorrt`）** | PyPI 直接装 | 一行命令、自动带依赖 wheel | 版本/CUDA 组合受 PyPI 索引限制、不含 trtexec 等完整工具链 | 纯 Python 推理、快速试验 |

> 选择心法：**要 `trtexec`、要多版本、不想动系统 → tar 包**；**只在 Python 里跑、图省事 → pip**；**整机就服务一个模型、有 root → deb/rpm**。本脚本走 **tar 包 + venv** 路线，是生产/共享机最常见的稳妥做法。

## 4. 版本对齐（安装真正的难点）

安装前**先查兼容矩阵**，确定这一组能彼此咬合：

```
   ┌────────────┐
   │ GPU 架构    │  (Ampere/Hopper… → 决定支持哪些精度，如 FP8 需 Hopper)
   └─────┬──────┘
         │
   ┌─────▼──────┐  驱动版本必须 ≥ CUDA 要求的最低驱动
   │ NVIDIA驱动  │  (nvidia-smi 看；CUDA Toolkit 不等于驱动)
   └─────┬──────┘
         │
   ┌─────▼──────┐  TensorRT 为某个 CUDA major.minor 编译
   │ CUDA        │  (脚本里 cuda-11.8；包名里就带 cuda 版本)
   └─────┬──────┘
         │
   ┌─────▼──────┐  cuDNN 版本也要落在 TensorRT 允许区间
   │ cuDNN       │
   └─────┬──────┘
         │
   ┌─────▼──────┐  wheel 文件名里的 cp310 = CPython 3.10 的 ABI
   │ Python ABI  │  venv 的 Python 必须与之一致，否则 pip 装不上
   └────────────┘
```

实操要点：

- `nvidia-smi` 看**驱动版本**（右上角 CUDA Version 是"驱动支持的最高 CUDA"，不是已装的 Toolkit）；
- 下载页/包名里会写明这个 TensorRT 包**对应的 CUDA 版本**（脚本里是 `cuda-11.8`）；
- wheel 名里的 `cp310` 必须 = 你 venv 的 Python 小版本（`cp310`↔3.10、`cp311`↔3.11）；
- `arch=$(uname -m)` 决定是 `x86_64` 还是 `aarch64`（ARM/Grace、Jetson 走 aarch64）。

> **不要凭记忆填版本号**。每个 TensorRT 版本支持哪些 CUDA/cuDNN，**以官方 Support Matrix 为准**。

## 5. tar 包安装逐行拆解

下面把脚本拆成"做什么 + 为什么"。命令里的版本号是占位。

### 5.1 建隔离环境（venv）

```bash
cd /home/guodong.li/virtual-venv
virtualenv -p /usr/bin/python3.10 model-inference-venv-py310-cu117
source .../model-inference-venv-py310-cu117/bin/activate
pip install torch        # 先有 torch，后面 Torch-TensorRT / 模型导出要用
```

**为什么**：TensorRT 的 wheel 绑定 Python ABI（`cp310`）。用 venv 把"这个 Python 版本 + 这套依赖"圈起来，避免污染系统 Python、也便于多版本共存。**命名带上 `py310`/`cu117` 是好习惯**——一眼知道这个环境的 Python 与 CUDA。

### 5.2 解压 tar 包

```bash
version="8.6.1.6"; arch=$(uname -m); cuda="cuda-11.8"
tar -xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.tar.gz
```

**为什么用变量**：包名把"版本/架构/CUDA"都编进文件名了，用变量拼接既防手抖又自带文档性。解压后得到 `TensorRT-<ver>/` 目录（即第 2 节那张图的结构）。

### 5.3 让链接器找到 `.so`（最关键的一步）

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/guodong.li/TensorRT-8.6.1.6/lib
```

**为什么**：`lib/` 里的 `libnvinfer.so` 等是核心。Linux 动态链接器（`ld.so`）默认只搜系统路径；tar 包不在系统路径里，必须把 `lib/` 加进 `LD_LIBRARY_PATH`，否则后面 `import tensorrt` 时 `dlopen` 失败。

```
   import tensorrt (Python)
        │  pip 装的 wheel 是个壳
        ▼
   dlopen("libnvinfer.so")  ──需要──▶  ld.so 去 LD_LIBRARY_PATH 里找
        │                                   │ 找到 → OK
        │                                   │ 找不到 → "cannot open shared object file"
```

> **进阶**：临时 `export` 只在当前 shell 有效。要永久生效，可写进 `~/.bashrc`，或更规范地用 `/etc/ld.so.conf.d/*.conf` + `ldconfig`（需 root）。脚本里两次 `export` 是因为前面用了 `<...>` 占位、后面才填真实绝对路径——**实际只需保留正确的那条**。

### 5.4 安装 Python wheel（核心三件套）

```bash
cd TensorRT-${version}/python
python3 -m pip install tensorrt-*-cp310-none-linux_x86_64.whl       # 完整版
python3 -m pip install tensorrt_lean-*-cp310-none-linux_x86_64.whl   # 精简版
python3 -m pip install tensorrt_dispatch-*-cp310-none-linux_x86_64.whl # 分发版
```

`*` 通配符让你不用手抄完整版本串；但 `cp310` 与 `linux_x86_64` 必须和你的环境匹配，否则 pip 直接报"not a supported wheel on this platform"。三者区别见第 6 节。

### 5.5 安装辅助工具 wheel（按需）

```bash
# UFF（旧 TF 路径，新项目基本不用）
python3 -m pip install uff-*.whl
which convert-to-uff           # 装好后会有这个命令

# graphsurgeon（旧图编辑工具）
cd ../graphsurgeon && python3 -m pip install graphsurgeon-*.whl

# onnx_graphsurgeon（ONNX 改图，仍常用：裁层、改输入shape、插自定义节点）
cd ../onnx_graphsurgeon && python3 -m pip install onnx_graphsurgeon-*.whl
```

**取舍**：`uff`/`graphsurgeon` 是 TF1/UFF 时代遗产，现代 ONNX 工作流**通常只需 `onnx_graphsurgeon`**（甚至都不需要）。不用就别装，少一层依赖少一个坑。

## 6. Python 三件套：full / lean / dispatch

很多人不懂为什么要装三个 wheel。它们对应**三种 runtime 形态**，是"构建能力"和"部署体积"的权衡：

| 包 | 含 Builder（能构建 engine）？ | 含完整 Runtime？ | 典型用途 |
|----|------|------|----------|
| `tensorrt`（full） | ✅ 能 build | ✅ 能 run | 开发机：又建又跑 |
| `tensorrt_lean`（精简 runtime） | ❌ 只跑不建 | ✅（仅运行已建好的 engine） | 部署端：只加载现成 engine，体积小 |
| `tensorrt_dispatch`（分发 runtime） | ❌ | ✅ + 版本兼容分发层 | 需跨 minor 版本兼容加载 engine 的场景 |

```
        ┌──────────── full ────────────┐   build + run，最大
   能力  │  ┌──── dispatch ────┐        │
   递减  │  │  ┌── lean ──┐    │        │   lean：只 run，最小
        │  │  │ run only │    │        │
        │  │  └──────────┘    │        │
        │  └──────────────────┘        │
        └──────────────────────────────┘
```

**心法**：**开发/构建 engine 用 full**；**纯部署、想瘦身只用 lean**；dispatch 用于需要"用较新 runtime 加载较旧 engine"这类版本分发场景。脚本三个都装，是为了"什么都能试"，生产环境可只留需要的那个。

## 7. 验证安装

```python
python3
>>> import tensorrt
>>> print(tensorrt.__version__)            # 打印版本，确认装对
>>> assert tensorrt.Builder(tensorrt.Logger())  # 能建 Builder = 库链接成功
```

**为什么这两步够用**：`import` 成功 → wheel 与 `.so` 都被找到（`LD_LIBRARY_PATH` 设对了）；`Builder(Logger())` 能构造 → 核心 C++ 对象可用、CUDA/cuDNN 链接无误。lean/dispatch 同理验证（它们没有 Builder 的完整能力时，按官方说明改成验证 `Runtime`）：

```python
>>> import tensorrt_lean as trt
>>> print(trt.__version__)
```

> 报错速查：`ImportError: libnvinfer.so` → 回看 5.3（`LD_LIBRARY_PATH`）；`not a supported wheel` → 5.4（cp 版本/架构不匹配）；`undefined symbol` → 第 4 节（CUDA/cuDNN 版本对不上）。

## 8. Torch-TensorRT 与上层生态

```bash
pip install torch_tensorrt
```

**Torch-TensorRT** 是 PyTorch 与 TensorRT 的"桥"：让你在 PyTorch 里**直接编译/调用** TensorRT，而不必手动走 ONNX 导出。它解决"我不想离开 PyTorch 工作流，但想要 TensorRT 的加速"。

生态全景（理解 install 在哪一层）：

```
   ┌──────────────────────────────────────────────┐
   │ 上层框架：Torch-TensorRT / TensorRT-LLM        │  ← LLM 推理常走 TRT-LLM
   ├──────────────────────────────────────────────┤
   │ 解析/改图：ONNX parser / onnx_graphsurgeon      │
   ├──────────────────────────────────────────────┤
   │ ★ TensorRT 核心：Builder + Runtime (本文装的)   │
   ├──────────────────────────────────────────────┤
   │ 底层：CUDA / cuDNN / cuBLAS / 驱动              │
   └──────────────────────────────────────────────┘
```

> 做 LLM 推理一般不会直接手搓 TensorRT，而是用 **TensorRT-LLM**（封装了 attention/KV-cache/量化等）。但 TRT-LLM 仍建立在本文这套核心之上，**版本对齐逻辑完全一样**。参见 [[llm-inference/tensorrt/README]]。

## 9. 全景流程与一份"干净"配置示例

```
 ① 查矩阵 → ② 建venv(py对) → ③ 下tar包(CUDA对) → ④ 解压
      └────────────────────────────────────────────┘
                          │
 ⑤ export LD_LIBRARY_PATH=.../lib   ←─ 让 .so 可见
                          │
 ⑥ pip install python/*.whl (full[+lean/dispatch])
                          │
 ⑦ import tensorrt; print(version); Builder(Logger())  ←─ 验证
                          │
 ⑧ (可选) pip install torch_tensorrt / 用 trtexec 跑基准
```

精简后的安装片段（去掉旧 UFF/占位，保留主干）：

```bash
# 0) 环境
source ~/venv/trt-py310/bin/activate
pip install torch

# 1) 解压（版本/CUDA以实际包名为准）
ver=8.6.x; arch=$(uname -m)
tar -xzf TensorRT-${ver}.Linux.${arch}-gnu.cuda-11.8.tar.gz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/TensorRT-${ver}/lib

# 2) 装 Python 绑定（cp310 与 venv 一致）
cd TensorRT-${ver}/python
pip install tensorrt-*-cp310-none-linux_x86_64.whl
# 部署瘦身可只装 lean；改图需要再装 onnx_graphsurgeon

# 3) 验证
python3 -c "import tensorrt as t; print(t.__version__); assert t.Builder(t.Logger())"
```

## 常见问题/坑

| 现象 / 坑 | 根因 | 处理 |
|-----------|------|------|
| `import tensorrt` 报 `libnvinfer.so: cannot open` | `LD_LIBRARY_PATH` 没含 `lib/` | 见 5.3，确认绝对路径正确并重开 shell |
| `xxx.whl is not a supported wheel` | wheel 的 `cp3xx`/架构与 venv 不符 | 换对应 Python 版本的 venv，或下对架构的包 |
| `undefined symbol` | TensorRT 与已装 CUDA/cuDNN 版本不匹配 | 回到第 4 节重新对矩阵 |
| `nvidia-smi` 的 CUDA 比包要求低 | 把"驱动支持的最高 CUDA"误当成已装 Toolkit | 关注驱动版本是否 ≥ 该 CUDA 的最低驱动要求 |
| 多版本 TensorRT 互相串扰 | 多个 `lib/` 同时在 `LD_LIBRARY_PATH` | 每个 venv 只指向一个版本的 `lib/` |
| pip 装的 `tensorrt` 没有 `trtexec` | pip 版不含完整工具链 | 用 tar/deb 包获取 `bin/trtexec` |
| `export` 重启后失效 | 只是当前 shell 变量 | 写入 `~/.bashrc` 或用 `ldconfig` 配置 |
| 装了 uff/graphsurgeon 仍报旧依赖错 | 旧 TF 路径已基本弃用 | 现代 ONNX 流程通常无需，不装即可 |
| FP8/某精度不可用 | GPU 架构不支持（如 FP8 需较新架构） | 见第 4 节，按 GPU 能力选精度 |

> 一句话总结：**装 TensorRT = 对齐版本 + 让 `.so` 可见 + 装对 Python wheel + 验证**。把这四件事做对，90% 的安装问题不会发生。具体版本号、文件名、API 一律以官方文档与你下载到的实际包为准。

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[llm-inference/tensorrt/README]] — TensorRT 专题总览
- [[llm-inference/README]] — 推理大类入口
- [[llm-compression/quantization/量化基础]] — TensorRT 的 INT8/FP8 量化原理
- [[ai-infra/ai-hardware/README]] — GPU 架构与精度支持的硬件背景
