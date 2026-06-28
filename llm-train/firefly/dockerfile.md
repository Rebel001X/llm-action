# Firefly 训练镜像 Dockerfile 深度解析（两阶段构建：base-env + train-env）

> 用「基础环境镜像」+「训练环境镜像」两层 Dockerfile，把 CUDA / PyTorch / 多套 conda 训练环境（LLM / Baichuan2 / T5）一次性固化成可复现的训练容器。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] · [[ai-framework/pytorch/README]] · [[ai-framework/huggingface-peft/README]] · [[llm-compression/quantization/量化基础]]

---

## 阅读地图

| 小节 | 你会得到什么 | 适合谁 |
| --- | --- | --- |
| 0. 一句话锚点 | 这两个 Dockerfile 到底在干嘛 | 全员 |
| 1. 地基/前置 | 镜像分层、conda 多环境、`RUN/COPY/ENV` 语义 | 不熟 Docker 的人 |
| 2. 两阶段构建总览 | base-env → train-env 的依赖链与 ASCII 图 | 想看全貌 |
| 3. base-env.Dockerfile 逐行 | CentOS7 + devtoolset-9 + 双 PyTorch 环境 | 维护基础镜像 |
| 4. train-env.Dockerfile 逐行 | 克隆环境 + 装依赖 + 编译 bitsandbytes | 维护训练镜像 |
| 5. 为什么这样设计 | 缓存命中、瘦身、隔离冲突依赖 | 想学最佳实践 |
| 实操命令 | 原文的 build / run / 校验命令（原样保留） | 直接复制用 |
| 常见问题/坑 | 网络、CUDA 版本、conda clean、shm-size | 踩坑前先看 |

---

## 0. 一句话锚点

这份文件是 **两个串联的 Dockerfile**：

1. **`base-env.Dockerfile`** → 产出基础镜像 `tianqiong-base-env`：在 CentOS7 + CUDA 11.7 + Python 3.10 之上，预装两个 conda 环境（`torch1131-venv` 装 PyTorch 1.13.1，`torch201-venv` 装 PyTorch 2.0.1）。
2. **`train-env.Dockerfile`** → 以上一步镜像为 `FROM` 基底，**克隆**这两个环境，分别装 LLM 训练 / Baichuan2 / T5 三套依赖，并从源码编译 `bitsandbytes`（QLoRA 的 4/8-bit 量化内核），产出训练镜像 `tianqiong-train-env`。

> 关键思想：**不可变环境固化**。把"装环境"这件最易出错、最耗时、最依赖网络的事，一次性烤进镜像，之后每台机器 `docker run` 拉起来就能直接训，环境 100% 一致。

---

## 1. 地基/前置：四个必须先懂的原子概念

### 1.1 镜像分层（Layer）与缓存

Docker 镜像是**只读层的叠加**。Dockerfile 里**每一条 `RUN / COPY / ADD` 生成一层**。构建时 Docker 会逐层比对：如果某层的指令和上游层都没变，就**直接复用缓存**，跳过执行。

```
  Dockerfile 指令            生成的镜像层（自底向上叠加）
  ─────────────────         ─────────────────────────────
  FROM centos7+cuda   ──►   [L0 基础层 (只读)]
  RUN yum install     ──►   [L1 devtoolset-9 层]
  RUN conda create x2 ──►   [L2 双 PyTorch 环境层]   ← 改上面任一行，这层及以下全部失效重建
                            ═══════════════════════
                            容器运行时再叠一层可写层(RW)
```

> 推论：**变动最频繁的指令要放最后**，否则会击穿后面所有层的缓存。原文把"装系统工具"放最前、"装 Python 包"放后，正是这个道理。

### 1.2 `conda create --clone`：环境隔离的复制

```
conda create -n B --clone A   # 把已有环境 A 原样复制一份成 B
```

为什么 train-env 要 `--clone` 而不是重新 `create`？因为 base-env 里已经装好了正确版本的 PyTorch（这一步最慢、最依赖内网 whl）。克隆 = **直接拷贝已固化的 PyTorch**，省去重新下载编译，再在克隆体上叠加各模型的差异化依赖。

### 1.3 Dockerfile 指令速查（只列本文用到的）

| 指令 | 作用 | 本文示例 |
| --- | --- | --- |
| `FROM` | 指定基底镜像 | `FROM .../python:py310-11.7-cudnn8-devel-centos7` |
| `MAINTAINER` | 维护者（已弃用，建议用 `LABEL`） | `guodong.li ...` |
| `ENV` | 设置环境变量（构建+运行均生效） | `ENV APP_DIR=/workspace` |
| `RUN` | 构建期执行 shell，**生成新层** | `RUN yum -y install ...` |
| `COPY` | 把构建上下文文件拷进镜像 | `COPY train-env ${APP_DIR}/train-env` |
| `WORKDIR` | 设置后续默认工作目录 | `WORKDIR $APP_DIR` |

### 1.4 为什么每条 `RUN` 里都要 `source ~/.bashrc && conda activate`？

每条 `RUN` 都是**全新的、独立的 shell 子进程**。上一条 `RUN` 里 `conda activate` 的状态**不会**带到下一条。所以每条需要 conda 的 `RUN` 都得重新 `source ~/.bashrc`（让 `conda` 命令可用）再 `conda activate xxx`，否则会装错环境或报 "conda: command not found"。

---

## 2. 两阶段构建总览（依赖链 ASCII 图）

```
  ┌──────────────────────────────────────────────────────────────┐
  │ 官方/内网基础镜像                                              │
  │ python:py310-11.7-cudnn8-devel-centos7                         │
  │ (Python3.10 + CUDA11.7 + cuDNN8 + devel 工具链, CentOS7)       │
  └───────────────────────────┬──────────────────────────────────┘
                              │ FROM
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 阶段一  base-env.Dockerfile                                    │
  │  + devtoolset-9 (GCC 9, 编译 CUDA 扩展要新 gcc)               │
  │  + conda env: torch1131-venv  →  PyTorch 1.13.1 + cu117       │
  │  + conda env: torch201-venv   →  PyTorch 2.0.1  + cu117       │
  │  产物: tianqiong-base-env:v1-20240131                          │
  └───────────────────────────┬──────────────────────────────────┘
                              │ FROM
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 阶段二  train-env.Dockerfile                                   │
  │  COPY train-env/ (requirements + bitsandbytes 源码)           │
  │                                                                │
  │   llm-venv          = clone(torch1131-venv) + requirements-llm│
  │                        + 源码编译 bitsandbytes (cuda11x)       │
  │   llm-baichuan2-venv = clone(torch201-venv)  + req-baichuan2  │
  │                        + 源码编译 bitsandbytes                 │
  │   t5-venv           = clone(torch201-venv)  + requirements-t5 │
  │                                                                │
  │  RUN rm -rf train-env   (装完删源码, 瘦身)                     │
  │  WORKDIR /workspace                                            │
  │  产物: tianqiong-train-env:v1-20240131                         │
  └──────────────────────────────────────────────────────────────┘
```

**环境继承关系一眼看清：**

| 训练环境 | 克隆自 | PyTorch 版本 | 依赖文件 | 用途 |
| --- | --- | --- | --- | --- |
| `llm-venv` | `torch1131-venv` | 1.13.1+cu117 | `requirements-llm.txt` | 通用 LLM（含 Firefly）训练 |
| `llm-baichuan2-venv` | `torch201-venv` | 2.0.1+cu117 | `requirements-llm-baichuan2.txt` | Baichuan2 训练 |
| `t5-venv` | `torch201-venv` | 2.0.1+cu117 | `requirements-t5.txt` | T5 系列训练 |

> 为什么要分三个环境而不是一个大杂烩？因为不同模型对 `transformers`、`tokenizers`、PyTorch 版本的要求会**互相打架**（依赖地狱）。用独立 conda 环境隔离，谁也不影响谁——这是同一镜像内"多模型共存"的标准做法。

---

## 3. 阶段一：`base-env.Dockerfile` 逐行解析

> 以下为**原文真料，原样保留**：

```dockerfile
FROM aiharbor.xxxx.local/base/python:py310-11.7-cudnn8-devel-centos7

MAINTAINER guodong.li liguodongiot@163.com

RUN yum -y install devtoolset-9 which && yum clean all && rm -rf /var/cache/yum/* && rm -rf /tmp/* \
&& echo ""  >> /etc/profile && echo "source /opt/rh/devtoolset-9/enable" >> /etc/profile

RUN conda init \
&& conda create -n torch1131-venv python=3.10 -y && source ~/.bashrc && conda env list && conda activate torch1131-venv  && pip install --no-cache-dir http://10.xx.2.25:81/pypi/pytorch/torch-1.13.1%2Bcu117-cp310-cp310-linux_x86_64.whl \
&& conda create -n torch201-venv python=3.10 -y && source ~/.bashrc && conda env list &&  conda activate torch201-venv && pip install --no-cache-dir http://10.cc.2.46:8000/base-env/torch-2.0.1+cu117-cp310-cp310-linux_x86_64.whl \
&& rm -rf ~/.cache/pip/* && conda clean -all && rm -rf /tmp/*
```

**逐段说明：**

- **`FROM ...py310-11.7-cudnn8-devel-centos7`**
  - `devel`（不是 `runtime`）= 带 `nvcc` 编译器和 CUDA 头文件。**必须用 `devel`**，否则后面编译 `bitsandbytes` 时找不到 `nvcc` 会失败。
  - `cudnn8` = 预装 cuDNN8（卷积/注意力加速库）。
  - `centos7` = 老牌稳定的企业 Linux，但其自带 GCC 太老 → 下一步要补 devtoolset-9。

- **`RUN yum -y install devtoolset-9 which ...`**
  - `devtoolset-9` = Red Hat 的 GCC 9 工具集。CentOS7 自带的是 GCC 4.8，**编译现代 CUDA 扩展（如 bitsandbytes）需要 C++14/17，必须升级到 GCC 9+**。
  - `which` = 后续脚本常用的命令查找工具。
  - `&& echo "source /opt/rh/devtoolset-9/enable" >> /etc/profile`：把"启用 GCC9"写进登录脚本，让每个新 shell 默认用 GCC9。
  - **同一行内的 `yum clean all && rm -rf /var/cache/yum/* && rm -rf /tmp/*` 至关重要**：清理动作必须和安装**写在同一条 `RUN` 里**。如果分两条 `RUN`，缓存会先被存进上一层（镜像已变大），下一层再删也删不掉历史层 → 镜像白白膨胀。

- **`RUN conda init && conda create ... && pip install <torch whl>`**
  - 两个环境（`torch1131-venv`、`torch201-venv`）各装一个 PyTorch wheel。
  - **从内网 IP（`http://10.xx.2.25:81/...`）装 whl**：内网 PyPI 镜像，比公网快、不依赖外网、版本可控。`%2B` 是 URL 编码的 `+`（`torch-1.13.1+cu117` 中的 `+`）。
  - `--no-cache-dir`：不留 pip 缓存，省镜像空间。
  - 行尾 `&& rm -rf ~/.cache/pip/* && conda clean -all && rm -rf /tmp/*`：清缓存、清 conda 下载包、清临时文件——同样是为了瘦身且必须同 `RUN`。

> **版本对照**：CUDA 11.7 ↔ PyTorch 1.13.1+cu117 / 2.0.1+cu117。镜像 CUDA、PyTorch 编译标记 `cu117`、bitsandbytes 的 `CUDA_VERSION=117` **三者必须对齐**，错一个就会运行时报 CUDA 不匹配。

---

## 4. 阶段二：`train-env.Dockerfile` 逐行解析

> 以下为**原文真料，原样保留**：

```dockerfile
FROM harbor.xxxx.io/base/tianqiong-base-env:v1-20240131

MAINTAINER guodong.li liguodongiot@163.com

ENV APP_DIR=/workspace
RUN mkdir -p -m 777 $APP_DIR

COPY train-env ${APP_DIR}/train-env

# llm-安装依赖
RUN source ~/.bashrc && conda env list && conda create -n llm-venv --clone torch1131-venv \
&& conda activate llm-venv && pip install --no-cache-dir -r ${APP_DIR}/train-env/requirements-llm.txt \
-i http://nexus3.xxx.com/repository/pypi/simple --trusted-host nexus3.xxx.com && rm -rf ~/.cache/pip/* && conda clean -all

#RUN source ~/.bashrc && conda env list && conda activate llm-venv && pip install  --no-cache-dir  storageutils==0.1.2 -i http://nexus3.xxx.com/repository/xxx_py_release/simple --trusted-host nexus3.xxx.com && rm -rf ~/.cache/pip/* && conda clean -all

RUN source ~/.bashrc && conda env list && conda activate llm-venv && cd ${APP_DIR}/train-env/bitsandbytes && source /opt/rh/devtoolset-9/enable && CUDA_VERSION=117 make cuda11x && python setup.py install && python setup.py clean --all && rm -rf ~/.cache/pip/* && conda clean -all


# baichuan2-t5-安装依赖
RUN source ~/.bashrc && conda env list && conda create -n llm-baichuan2-venv --clone torch201-venv && conda activate llm-baichuan2-venv && pip install  --no-cache-dir -r \
${APP_DIR}/train-env/requirements-llm-baichuan2.txt \
-i http://nexus3.xxx.com/repository/pypi/simple --trusted-host nexus3.xxx.com && rm -rf ~/.cache/pip/* && conda clean -all

#RUN source ~/.bashrc && conda env list && conda activate llm-baichuan2-venv && pip install  --no-cache-dir storageutils==0.1.2 -i http://nexus3.xxx.com/repository/xxx_py_release/simple --trusted-host nexus3.xxx.com && rm -rf ~/.cache/pip/* && conda clean -all

RUN source ~/.bashrc && conda env list && conda activate llm-baichuan2-venv && cd ${APP_DIR}/train-env/bitsandbytes && source /opt/rh/devtoolset-9/enable && CUDA_VERSION=117 make cuda11x && python setup.py install  && python setup.py clean --all && rm -rf ~/.cache/pip/* && conda clean -all

# t5-安装依赖
RUN source ~/.bashrc && conda env list && conda create -n t5-venv --clone torch201-venv \
&& conda activate t5-venv &&  pip install --no-cache-dir -r ${APP_DIR}/train-env/requirements-t5.txt \
-i http://nexus3.xxd.com/repository/pypi/simple --trusted-host nexus3.xxx.com && rm -rf ~/.cache/pip/* && conda clean -all

#RUN source ~/.bashrc && conda env list && conda activate t5-venv && pip install --no-cache-dir storageutils==0.1.2 -i http://nexus3.mxxx.com/repository/xxx_py_release/simple --trusted-host nexus3.fss.com && rm -rf ~/.cache/pip/* && conda clean -all

RUN rm -rf ${APP_DIR}/train-env

#设置工作目录
WORKDIR $APP_DIR
```

**逐段说明：**

- **`ENV APP_DIR=/workspace` + `RUN mkdir -p -m 777 $APP_DIR`**
  - 统一工作目录变量。`-m 777` 给全权限，方便容器内任意用户读写（生产上其实偏松，但训练场景图省事常见）。

- **`COPY train-env ${APP_DIR}/train-env`**
  - 把构建上下文里的 `train-env/` 目录（含三个 `requirements-*.txt` 和 `bitsandbytes` 源码）拷进镜像。
  - ⚠️ 注意末尾有 `RUN rm -rf ${APP_DIR}/train-env`——**装完就删**。但因为分层机制，被删的文件仍残留在 `COPY` 那层里（镜像不会真变小，除非用多阶段构建 squash）。这里删主要是让最终文件系统"干净"，避免运行时误用源码。

- **三套环境的安装模式完全一致（克隆 → 装 requirements → 编译 bitsandbytes）：**

  | 步骤 | llm-venv | llm-baichuan2-venv | t5-venv |
  | --- | --- | --- | --- |
  | 克隆基底 | `--clone torch1131-venv` | `--clone torch201-venv` | `--clone torch201-venv` |
  | 装依赖 | `requirements-llm.txt` | `requirements-llm-baichuan2.txt` | `requirements-t5.txt` |
  | 编译 bnb | ✅ | ✅ | ❌（T5 不需 4bit 量化） |

- **`pip install -i http://nexus3.xxx.com/... --trusted-host nexus3.xxx.com`**
  - `-i` 指定内网 Nexus 私有 PyPI 源（公司自建包仓库，离线/合规/限速）。
  - `--trusted-host`：因为用的是 `http://`（非 https），pip 默认会拒绝，必须显式信任该主机，否则报 SSL/不受信任源错误。

- **编译 bitsandbytes 那行（最易翻车）：**
  ```
  cd .../bitsandbytes && source /opt/rh/devtoolset-9/enable && CUDA_VERSION=117 make cuda11x && python setup.py install
  ```
  - **`source /opt/rh/devtoolset-9/enable`**：临时启用 GCC9，否则用 CentOS7 默认 GCC4.8 编译 CUDA 内核会报 C++ 标准错误。
  - **`CUDA_VERSION=117 make cuda11x`**：bitsandbytes 需要为**具体 CUDA 版本**编译对应的 GPU 内核库（`libbitsandbytes_cuda117.so`）。`117` 必须与镜像 CUDA 11.7 一致。
  - `python setup.py install` → 装进当前激活的 conda 环境；`clean --all` 删编译中间产物。
  - 这也是为什么 base 镜像必须是 `devel`（带 nvcc）且要 GCC9——**bitsandbytes 是从源码现编译的，不是装 wheel**。

- **被注释掉的 `storageutils==0.1.2` 行（`#RUN ...`）**
  - 这是公司内部存储工具包，从另一个私有源（`xxx_py_release`）装。注释掉表示当前镜像版本**不需要**它（可能某些场景才开），保留注释是为了随时按需启用。

> **bitsandbytes 与本仓库的联系**：bitsandbytes 提供 8-bit 优化器与 4/8-bit 量化线性层，是 **QLoRA** 训练的底座。详见 [[llm-compression/quantization/量化基础]] 与 [[ai-framework/huggingface-peft/README]]。

---

## 5. 为什么这样设计（设计动机小结）

| 设计选择 | 为什么 | 不这么做的后果 |
| --- | --- | --- |
| 拆成 base + train 两个 Dockerfile | base 几乎不变（缓存稳定），train 频繁改 | 每次改 requirements 都重装 PyTorch，构建几十分钟 |
| 每个 `RUN` 行尾清缓存（同一条命令内） | 缓存不进层，镜像瘦 | 镜像膨胀几个 GB |
| 多 conda 环境 + `--clone` | 隔离依赖冲突 + 复用已固化 PyTorch | 依赖打架 / 重复下载 |
| 内网 whl + Nexus 私有源 | 快、可控、离线、合规 | 构建依赖外网，慢且不可复现 |
| 从源码编译 bitsandbytes | 必须匹配精确 CUDA 版本 | pip 装的预编译版常报 CUDA setup 失败 |
| 用 `devel` 基础镜像 + devtoolset-9 | 编 CUDA 扩展要 nvcc + 新 GCC | 编译 bitsandbytes 直接失败 |

---

## 实操命令（原文真料，原样保留）

### 6.1 构建并验证 base 镜像

```bash
sudo docker build --network=host -f base-env.Dockerfile -t harbor.xxxx.io/base/tianqiong-base-env:v1-20240131 .


sudo docker run -it --gpus '"device=4,5"' --network=host \
--shm-size 4G \
harbor.xxx.io/base/tianqiong-base-env:v1-20240131  \
/bin/bash 

source ~/.bashrc 
conda activate torch1131-venv && pip list | grep torch
conda activate torch201-venv && pip list | grep torch
```

### 6.2 构建并验证 train 镜像

```bash
sudo docker build --network=host -f train-env.Dockerfile -t harbor.xxx.io/base/tianqiong-train-env:v1-20240131 .



sudo docker run -it --gpus '"device=4,5"' --network=host \
--shm-size 4G \
harbor.xxx.io/base/tianqiong-train-env:v1-20240131  \
/bin/bash 


source ~/.bashrc 
conda activate llm-venv && pip list | grep torch
conda activate llm-baichuan2-venv && pip list | grep torch
conda activate t5-venv && pip list | grep torch
```

### 6.3 关键运行参数解释

| 参数 | 含义 | 为什么需要 |
| --- | --- | --- |
| `--network=host`（build & run） | 容器直接用宿主机网络栈 | 构建时能访问内网 whl / Nexus 源；运行时多机通信 |
| `-f xxx.Dockerfile` | 指定 Dockerfile 文件名 | 一个目录里有多个 Dockerfile，需显式指定 |
| `-t harbor.../name:tag` | 打镜像标签（仓库/名:版本） | 推到 Harbor 私有仓库 + 版本管理（`v1-20240131`） |
| `.`（末尾的点） | 构建上下文 = 当前目录 | `COPY` 的源相对它解析 |
| `--gpus '"device=4,5"'` | 只暴露第 4、5 号 GPU | 多卡机上按卡分配，避免占满 |
| `--shm-size 4G` | 共享内存设为 4G | **PyTorch DataLoader 多 worker 走 /dev/shm，默认 64M 会 OOM/卡死** |
| `-it` | 交互式 + 分配 TTY | 进容器开 bash 调试 |
| `pip list \| grep torch` | 校验 torch 版本是否正确 | 每个环境装对了才算镜像 OK |

> 校验逻辑：进容器后逐个 `conda activate` 各环境再 `grep torch`，确认 `llm-venv` 是 1.13.1、`baichuan2/t5` 是 2.0.1——**构建完必做这步冒烟测试**。

---

## 常见问题/坑

| 现象 | 根因 | 解决 |
| --- | --- | --- |
| 构建时下载 whl 超时 / 404 | 没加 `--network=host`，容器访问不到内网源 | build 时加 `--network=host`；确认内网 IP 可达 |
| `conda: command not found`（某 RUN 内） | 该 `RUN` 是新 shell，没 `source ~/.bashrc` | 每条需 conda 的 `RUN` 都先 `source ~/.bashrc` |
| 装到了错误的 conda 环境 | 跨 `RUN` 的 `conda activate` 不延续 | 同一条 `RUN` 内 activate 后立即装 |
| 编译 bitsandbytes 报 C++ 标准 / GCC 错 | 用了 CentOS7 默认 GCC4.8 | 编译前 `source /opt/rh/devtoolset-9/enable` |
| bitsandbytes 运行报 CUDA setup 失败 | `CUDA_VERSION` 与镜像 CUDA 不一致 | 保持 `CUDA_VERSION=117` ↔ 镜像 cu117 |
| `nvcc not found` | 基础镜像用了 `runtime` 而非 `devel` | 用 `...devel...` 基础镜像 |
| `pip install` 报源不受信任 | 用的是 `http://` 源没加白名单 | 加 `--trusted-host nexus3.xxx.com` |
| 镜像几个 GB 巨大 | 清缓存写成了独立 `RUN`（删不掉历史层） | 安装与 `rm/clean` 写在同一条 `RUN` |
| DataLoader 多 worker 卡死/报错 | `--shm-size` 太小（默认 64M） | run 时加 `--shm-size 4G`（或更大） |
| 改了 requirements 还得重装 PyTorch | 把 torch 和 requirements 放同一镜像 | 用本文的 base/train 两层分离 |

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]] · [[llm-train/README]]
- 分布式训练框架：[[llm-train/pytorch/distribution/README]] · [[llm-train/megatron/README]] · [[llm-train/megatron-deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]]
- 微调 / PEFT：[[ai-framework/huggingface-peft/README]] · [[llm-train/peft/PEFT-API]] · [[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]]
- 量化（bitsandbytes 关联）：[[llm-compression/quantization/量化基础]]
- 网络通信（多机训练 `--network=host` 关联）：[[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/NCCL]]
- 上下游：[[llm-algo/transformer/模型架构]] · [[llm-alignment/RLHF]] · [[B07:llm-inference/大模型推理张量并行]]
