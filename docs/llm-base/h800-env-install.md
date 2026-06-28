# H800 训练/推理环境安装（CUDA / NCCL / cuDNN / OpenMPI / PyTorch / DeepSpeed / Apex）

> 一句话定位：H800（Hopper, sm_90）是新一代数据中心 GPU，**计算能力 sm_90 要求 CUDA ≥ 11.8、PyTorch ≥ 2.0**，本文把"为什么"和一整套从 0 到能跑训练的安装命令编排清楚。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-framework/pytorch/README]] · [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 小节 | 你会得到什么 | 关键产物 |
| --- | --- | --- |
| 0. 一句话锚点 | sm_90 + CUDA 11.8 + PyTorch 2.0 的硬约束 | 一行记忆 |
| 1. 地基/前置 | SM 架构、计算能力、CUDA/驱动/runtime 三层关系 | 概念地图 |
| 2. 计算能力对照表 | H800 落在哪一格、为什么 sm_37 列表不含 90 | 架构表 |
| 3. CUDA 安装 | runfile 安装、软链布局 | `cuda_11.8.0_*.run` |
| 4. NCCL 安装 | 多卡集合通信库、为什么 copy 进 cuda 目录 | `nccl_2.16.5` |
| 5. cuDNN 安装 | 深度学习算子库 | `cudnn 8.9.2` |
| 6. OpenMPI 编译 | 多机启动器、`./configure` 参数 | `openmpi-5.0.1` |
| 7. 虚拟环境 + PyTorch | venv 隔离、cu118 轮子 | `torch-2.0.0+cu118` |
| 8. DeepSpeed / Apex | 训练框架与融合算子、为什么 `--no-build-isolation` | `apex 30a7ad3` |
| 9. CUDA 12.1 配套 | 升级到 12.1 的对应轮子 | `torch-2.1.2+cu121` |
| 常见坑 | 版本错配、软链、ABI、编译失败 | 排错表 |

---

## 0. 一句话锚点

> **H800 = Hopper 架构 = 计算能力 sm_90。要让 PyTorch 真正用上这块卡，必须 CUDA ≥ 11.8 且 PyTorch ≥ 2.0；否则 PyTorch 在 import/前向时直接报 "sm_90 is not compatible"。**

记忆口诀：**九零要十一点八，火炬要两点零（90 → 11.8 → 2.0）**。

---

## 1. 地基：SM、计算能力、CUDA 三层栈

### 1.1 什么是 SM 与"计算能力"

GPU 由许多 **SM（Streaming Multiprocessor，流式多处理器）** 组成。NVIDIA 用 **Compute Capability（计算能力）** 给每一代 SM 编号，写作 `sm_XY`：

- `X` = 主版本（架构代次，例如 9 = Hopper、8 = Ampere、7 = Volta/Turing）；
- `Y` = 次版本（同代内的小改，例如 sm_80 = A100、sm_86 = A10/3090、sm_90 = H100/H800）。

计算能力**不是性能分数**，而是"这代硬件支持哪些指令/特性"的标识。编译 CUDA 代码（`nvcc -gencode arch=compute_90,code=sm_90`）时要把目标 SM 编进去，运行时 GPU 的真实计算能力必须 **被编进二进制里**，否则跑不起来。这正是 PyTorch 报错的根因。

### 1.2 为什么"驱动 / CUDA Toolkit / runtime"要分清

```
  ┌─────────────────────────────────────────────────────────┐
  │  你的 Python: torch.cuda.is_available()                  │
  ├─────────────────────────────────────────────────────────┤
  │  ② CUDA Runtime (libcudart) ── PyTorch 轮子自带 (cu118)  │  ← pip 装的
  ├─────────────────────────────────────────────────────────┤
  │  ① CUDA Toolkit (nvcc/头文件/库) ── 编译 apex/自定义算子 │  ← runfile 装的
  ├─────────────────────────────────────────────────────────┤
  │  ⓪ NVIDIA Driver (内核态) ── 520.61.05 等               │  ← 决定能跑的最高 CUDA
  ├─────────────────────────────────────────────────────────┤
  │       H800 硬件 (sm_90)                                  │
  └─────────────────────────────────────────────────────────┘
```

- **驱动版本 ≥ CUDA 版本要求的最低值**：CUDA 11.8 配套驱动 `520.61.05`（见 runfile 文件名）。新驱动可向下兼容老 CUDA。
- **PyTorch 轮子里自带 runtime**：`torch-2.0.0+cu118` 表示它内置了 cu118 的 runtime，**不依赖你系统装的 Toolkit 也能前向/反向**。
- **系统 Toolkit（nvcc）只在你要"现场编译"时才用**：例如装 Apex、Megatron 的融合算子、DeepSpeed 的 op。所以本文既装轮子又装 Toolkit。

### 1.3 一句话因果链

> 卡是 sm_90 → 需要支持 sm_90 的 PyTorch 二进制 → 官方从 **cu118 + torch2.0** 起才编入 sm_90 → 所以"CUDA 11.8 / PyTorch 2.0"是**下限**而非随意选择。

---

## 2. 计算能力对照表：H800 在哪一格

下表是各代架构与 `sm_XX` 的映射（**原文真料保留**）：

| Fermi **†** | Kepler **†** | Maxwell **‡** | Pascal | Volta | Turing | Ampere | Ada (Lovelace) | [Hopper](https://www.nvidia.com/en-us/data-center/hopper-architecture/) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sm_20 | sm_30 | sm_50 | sm_60 | sm_70 | sm_75 | sm_80 | sm_89 | **sm_90** |
|     | sm_35 | sm_52 | sm_61 | sm_72<br>(Xavier) |     | sm_86 |     | sm_90a (Thor) |
|     | sm_37 | sm_53 | sm_62 |     |     | sm_87 (Orin) |     |     |

**†** Fermi 和 Kepler 从 CUDA 9 / 11 起被弃用。
**‡** Maxwell 从 CUDA 11.6 起被弃用。

参考：https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

**H800 = Hopper = `sm_90`**（与 H100 同架构，区别在 NVLink 带宽被限制，计算能力编号相同）。

### 2.1 真实报错（用 CUDA 11.7 + PyTorch 1.13.1 时）

> 如果使用 H800，CUDA 版本要在 **11.8 及以上**，同时 PyTorch 版本要在 **2.0.0 以上**。下面是使用 CUDA 11.7 + PyTorch 1.13.1 的报错：

```
NVIDIA H800 with CUDA capability sm_90 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.
```

**怎么读这条报错**：PyTorch 1.13.1 的二进制里编入的最高架构是 `sm_86`（Ampere 顶格），列表里**根本没有 sm_90**。GPU 真实能力 sm_90 找不到匹配的 cubin/PTX，于是拒绝运行。升级到 `cu118 + torch2.0` 后，编入列表会出现 `sm_90`，报错消失。

```
 torch1.13.1+cu117  支持: ... sm_80 sm_86            ┐  最高只到 86
 H800 实际能力:      sm_90  ───────────────────────  │  90 ∉ {…,86}  →  报错
 torch2.0.0+cu118   支持: ... sm_80 sm_86 sm_90      ┘  含 90  →  OK
```

---

## 3. CUDA Toolkit 11.8 安装（runfile 方式）

```bash
mkdir -p /home/local/cuda-11.8
sudo ln -s /home/local/cuda-11.7 /usr/local/cuda-11.8

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
chmod +x cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

**逐行解释**：

- `mkdir -p /home/local/cuda-11.8`：因为根分区 `/usr/local` 往往空间小，把真实文件放到大盘 `/home/local`，再用软链指过去，这是机房常见布局。
- `ln -s … /usr/local/cuda-11.8`：建立软链，下游所有脚本只认 `/usr/local/cuda-11.8` 这个标准路径。
- runfile 文件名里的 `520.61.05` 是**配套驱动版本**；如果机器已装更高版本驱动，安装时记得在交互菜单里 **取消勾选 Driver**，只装 Toolkit，避免覆盖好用的驱动。

> 🔧 安装后要让 shell 找到它（原文 OpenMPI 段也用到 PATH/LD_LIBRARY_PATH，统一在第 6 节给）：
> `export PATH=/usr/local/cuda-11.8/bin:$PATH`
> `export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH`
> 验证：`nvcc --version` 应显示 `release 11.8`。

---

## 4. NCCL 安装：多卡/多机集合通信的发动机

**NCCL（NVIDIA Collective Communications Library）** 实现 `all-reduce / all-gather / broadcast / reduce-scatter` 等[[ai-infra/网络/集合通信原语]]。数据并行训练每一步反向后都要 all-reduce 梯度，NCCL 直接决定多卡扩展效率。H800 走 NVLink/NVSwitch，NCCL 会自动选拓扑。

```bash
tar xvf nccl_2.16.5-1+cuda11.8_x86_64.txz
cd nccl_2.16.5-1+cuda11.8_x86_64/
sudo cp -r include/* /usr/local/cuda-11.8/include/
sudo cp -r lib/* /usr/local/cuda-11.8/lib64/

# export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
```

**为什么 copy 进 CUDA 目录**：NCCL 是头文件 + `.so` 的纯库分发。直接拷到 `cuda-11.8/include` 与 `cuda-11.8/lib64`，就能复用 CUDA 已配好的 `LD_LIBRARY_PATH`，不必再单独设环境变量。注意包名里的 `+cuda11.8` 必须与本机 CUDA 版本对齐——NCCL 与 CUDA 是**强绑定**的（ABI 依赖）。

```
   GPU0 ─┐                         ┌─ GPU0
   GPU1 ─┤  all-reduce 梯度        ├─ GPU1   每一步都做：
   GPU2 ─┼──► NCCL (NVLink) ──────►┼─ GPU2   ① reduce-scatter
   GPU3 ─┘   带宽决定扩展效率       └─ GPU3   ② all-gather
```

---

## 5. cuDNN 安装：卷积/RNN/注意力的高性能算子库

**cuDNN** 提供深度学习常见算子（卷积、归一化、注意力前后向等）的高度优化实现，PyTorch 通过它加速。

```bash
tar -xvf cudnn-linux-x86_64-8.9.2.26_cuda11-archive.tar.xz

cd cudnn-linux-x86_64-8.9.2.26_cuda11-archive
sudo cp include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp -P lib/libcudnn*  /usr/local/cuda-11.8/lib64/
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
```

**关键点**：

- `cp -P`：保留软链（`libcudnn.so → libcudnn.so.8 → libcudnn.so.8.9.2`）。漏掉 `-P` 会把软链复制成实体文件，破坏版本链。
- `chmod a+r`：保证所有用户可读，避免非 root 训练用户加载失败。
- 包名 `_cuda11` 必须与 CUDA 11.x 对应；`8.9.2.26` 是 cuDNN 版本号。

---

## 6. OpenMPI 5.0.1 编译安装：多机启动器

**OpenMPI** 在多机多卡训练里负责**进程启动与跨节点通信引导**（如 Megatron/DeepSpeed 用 `mpirun` 拉起每个 rank）。它常需从源码编译以匹配本机网络栈。

```bash
tar -xvzf openmpi-5.0.1.tar.gz

cd openmpi-5.0.1
./configure --prefix=/data/hpc/software/openmpi
make -j 80
sudo make install -j 80

export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:/usr/local/nccl_2.11.4-1+cuda11.0_x86_64/lib:/usr/local/openmpi/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/openmpi/bin:/usr/local/cuda-11.0/bin:$PATH
```

**逐项解释**：

- `./configure --prefix=…`：指定安装到独立目录，便于多版本共存与卸载。
- `make -j 80`：用 80 个并行任务编译（取决于机器核数），OpenMPI 编译较慢，并行能省大量时间。
- 末尾两行 `export`：把 OpenMPI 的 `bin/`、`lib/` 以及 CUDA、NCCL 一起加入搜索路径。
  > ⚠️ 注意原文这两行里写的是 `cuda-11.0` 与 `nccl_2.11.4 … cuda11.0`，这是从别的环境**复制粘贴遗留的旧路径**。在本机 11.8 环境下应改为 `/usr/local/cuda-11.8/lib64`、对应 `nccl_2.16.5 … cuda11.8` 的路径，否则会指向不存在/版本不符的目录（这是非常典型的踩坑点，详见末尾坑表）。

```
 mpirun -np 16 -hostfile hosts  python train.py
   │
   ├─ node0: rank0..7  ──┐
   └─ node1: rank8..15 ──┴─► 每个进程内再用 NCCL 走 NVLink/IB 通信
```

---

## 7. 虚拟环境 + PyTorch

### 7.1 为什么要 venv

不同项目对 `torch / cuda / transformers` 版本要求不同。用 `virtualenv` 把每个项目的依赖**物理隔离**，避免"装了 A 项目把 B 项目搞坏"。命名 `llama-venv-py310-cu118` 把**用途/Python 版本/CUDA 版本**写进名字，一眼可辨。

```bash
# mkdir -p /home/guodong.li/virtual-venv

cd /home/guodong.li/virtual-venv
virtualenv -p /usr/bin/python3.10 llama-venv-py310-cu118
source /home/guodong.li/virtual-venv/llama-venv-py310-cu118/bin/activate
```

- `-p /usr/bin/python3.10`：指定解释器为 Python 3.10（与下面轮子文件名里的 `cp310` 必须一致）。
- `source … /bin/activate`：激活后 `which python` 指向 venv 内部。

### 7.2 PyTorch（cu118 轮子）

下载地址：https://download.pytorch.org/whl/torch_stable.html

```bash
wget -c https://download.pytorch.org/whl/cu118/torch-2.0.0%2Bcu118-cp310-cp310-linux_x86_64.whl
wget -c https://download.pytorch.org/whl/cu118/torchvision-0.15.0%2Bcu118-cp310-cp310-linux_x86_64.whl

pip install torch-2.0.0+cu118-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.15.0+cu118-cp310-cp310-linux_x86_64.whl
```

**读懂轮子文件名**（这是装对版本的关键技能）：

```
   torch - 2.0.0 + cu118 - cp310 - cp310 - linux_x86_64 . whl
     │      │       │        │       │         │
   包名   torch版本  CUDA版本 Python  ABI tag   平台
                    (≥11.8) (3.10)  (3.10)    (linux x86-64)
```

四个维度（torch 版本、CUDA 版本、Python 版本、平台）**任何一个对不上都装不上或运行报错**。`%2B` 是 URL 里 `+` 的转义。`wget -c` 支持断点续传。

> 装完自检：
> ```python
> import torch
> print(torch.__version__)              # 2.0.0+cu118
> print(torch.cuda.is_available())      # True
> print(torch.cuda.get_device_name(0))  # NVIDIA H800
> print(torch.cuda.get_arch_list())     # 应包含 'sm_90'
> ```
> 只要 `get_arch_list()` 里有 `sm_90`，第 2 节那条报错就不会再出现。

---

## 8. 训练框架：DeepSpeed / Accelerate / Apex

### 8.1 DeepSpeed + Accelerate + tensorboardX

```bash
pip install deepspeed==0.9.5
pip install accelerate
pip install tensorboardX
```

- **DeepSpeed**：ZeRO 显存优化、流水/张量并行、offload，是大模型训练的主力，见 [[ai-framework/deepspeed/README]]。`0.9.5` 与 torch2.0 配套。
- **Accelerate**：HuggingFace 的轻量分布式封装，统一 `device/混合精度/多卡` 启动。
- **tensorboardX**：训练曲线可视化。

### 8.2 Apex（NVIDIA 融合算子，需源码编译）

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 30a7ad3
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
```

**为什么 Apex 这么"挑"**：Apex 提供 `FusedAdam`、融合 LayerNorm 等 CUDA 算子，必须**用本机 nvcc 现场编译**，因此它对 CUDA Toolkit/PyTorch 版本极敏感。逐个参数解释：

| 参数 | 作用 | 为什么需要 |
| --- | --- | --- |
| `git checkout 30a7ad3` | 锁定到该 commit | Apex 主分支频繁变动，固定 commit 保证可复现 |
| `--no-build-isolation` | 用当前 venv 的 torch 编译 | **关键**：否则 pip 会建临时隔离环境装别的 torch，导致编出的算子与你的 torch ABI 不符 |
| `--cpp_ext` / `--cuda_ext` | 同时编 C++ 与 CUDA 扩展 | 不带这两个就只装纯 Python 版，享受不到融合算子加速 |
| `--no-cache-dir` | 不用 pip 缓存 | 避免拿到旧的、与当前 torch 不匹配的构建产物 |
| `-v` | verbose | 编译易失败，详细日志方便定位 |

> ⚠️ 编 Apex 前**必须先激活 venv、装好 torch、`nvcc` 在 PATH 里且版本与 torch 的 cu 版本一致**（cu118 ↔ Toolkit 11.8）。这是新手最容易翻车的一步。

---

## 9. 配套：升级到 CUDA 12.1 的对应版本

如需更新栈到 CUDA 12.1（例如要用更高版本 PyTorch / 新特性），换用下面这套对应版本即可，思路与上文完全一致，只是版本号整体升一档：

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```

配套库与轮子：

```
NCCL：
https://developer.nvidia.com/downloads/compute/machine-learning/nccl/secure/2.18.3/agnostic/x64/nccl_2.18.3-1+cuda12.1_x86_64.txz/

CUDNN：
（按 12.x 对应版本下载，安装方式同第 5 节）

pytorch:
https://download.pytorch.org/whl/cu121/torch-2.1.2%2Bcu121-cp310-cp310-linux_x86_64.whl
https://download.pytorch.org/whl/cu121/torchaudio-2.1.2%2Bcu121-cp310-cp310-linux_x86_64.whl
https://download.pytorch.org/whl/cu121/torchvision-0.16.2%2Bcu121-cp310-cp310-linux_x86_64.whl
```

**版本对应一览（两套栈对照）**：

| 组件 | 11.8 栈 | 12.1 栈 |
| --- | --- | --- |
| CUDA Toolkit | 11.8.0（驱动 520.61.05） | 12.1.1（驱动 530.30.02） |
| NCCL | 2.16.5 +cuda11.8 | 2.18.3 +cuda12.1 |
| PyTorch | 2.0.0 +cu118 | 2.1.2 +cu121 |
| torchvision | 0.15.0 +cu118 | 0.16.2 +cu121 |
| torchaudio | —（原文未列） | 2.1.2 +cu121 |

> 规律：**CUDA 主版本 → 决定 NCCL/cuDNN/PyTorch 的 `+cuXXX` 后缀**，四者必须同档。H800 在两套栈下都可用（都 ≥ 11.8 / ≥ 2.0）。

---

## 常见问题/坑

| 现象 | 根因 | 解决 |
| --- | --- | --- |
| `sm_90 is not compatible` | torch 二进制最高只编到 sm_86（如 1.13.1+cu117） | 升到 `torch2.0+cu118` 及以上；`torch.cuda.get_arch_list()` 须含 `sm_90` |
| `torch.cuda.is_available()` 为 False | 驱动版本 < CUDA 要求，或装了 CPU 版轮子 | 检查 `nvidia-smi` 驱动 ≥ 520.61.05；确认轮子名带 `+cu118` 不是 `+cpu` |
| OpenMPI `export` 指向 `cuda-11.0` | 原文环境变量是旧路径残留 | 改成 `/usr/local/cuda-11.8/lib64` 和对应 `nccl_2.16.5…cuda11.8` 路径 |
| cuDNN 拷贝后报找不到 `.so` | 漏了 `cp -P`，软链被复制成实体文件 | 重新用 `cp -P` 保留 `libcudnn.so → .so.8 → .so.8.9.2` 链 |
| NCCL all-reduce 报版本错 | NCCL 的 `+cudaXX` 与本机 CUDA 不一致 | 下载与 CUDA 版本完全匹配的 NCCL 包 |
| Apex 编译失败/ABI 不符 | 没加 `--no-build-isolation`，或 nvcc 与 torch 的 cu 版本不一致 | 激活 venv→装好 torch→确认 `nvcc -V` 为 11.8→加 `--no-build-isolation` 重装 |
| `nvcc` 找不到 | 装了 Toolkit 但没进 PATH | `export PATH=/usr/local/cuda-11.8/bin:$PATH` |
| runfile 把好驱动覆盖了 | 安装时默认勾选了 Driver | 重跑 runfile，菜单里取消 Driver，只装 Toolkit |
| venv 里 python 不是 3.10 | `virtualenv -p` 指错解释器 | 用 `-p /usr/bin/python3.10`，与轮子 `cp310` 对齐 |

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 硬件原理：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 网络/通信：[[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]]
- 训练框架：[[ai-framework/pytorch/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]
- 训练/微调：[[llm-train/README]] · [[llm-train/peft/PEFT-API]]
- 对齐：[[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 推理：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/PD分离]] · [[llm-inference/解码策略]]
- 优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 压缩/量化：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 评测：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 内存估算：[[docs/transformer内存估算]]
- 模型/架构：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
