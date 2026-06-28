# MindIE 1.0.RC2：昇腾 NPU 上的大模型推理服务部署实战

> 一句话定位：MindIE 是华为昇腾(Ascend)平台的端到端大模型推理引擎，本篇用真实命令把 1.0.RC2 镜像从拉取、起容器、装 CANN、配 config.json 到拉起 `mindieservice_daemon` 服务全流程跑通。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[llm-inference/README]] · [[llm-inference/vllm/README]]

## 阅读地图

| 小节 | 关键问题 | 你会得到 |
|------|---------|---------|
| 0. 一句话锚点 | MindIE 在昇腾栈里是哪一层？ | 一句话定位 |
| 1. 地基/前置 | 跑起来需要哪些硬件/软件层 | 驱动→CANN→ATB→MindIE 五层栈 |
| 2. 拿镜像 | 镜像从哪来、tag 怎么读 | AscendHub / SWR / rsync 三条路 |
| 3. 起容器 | 那一长串 `-v -e` 到底干嘛 | 逐参数拆解 + 设备挂载原理 |
| 4. 装 CANN + 环境变量 | 为什么要 source 一堆 set_env | 软件栈分层激活 |
| 5. config.json | 改哪几个字段才能拉起模型 | 模型路径/世界大小/显存 |
| 6. 起服务 | daemon 怎么读权重并对外暴露端口 | 启动链路 ASCII 图 |
| 7. 固化镜像 | 装好的环境怎么打包复用 | commit→save→rsync |
| 8. 一键脚本 | `llm-server3.sh` 四个参数 | 7B 单机双卡 / 72B 八卡 |
| 实操合集 | 把原文真料原样留存 | 可复制命令块 |
| 常见坑 | 起不来怎么排查 | 排查对照表 |

## 0. 一句话锚点

**MindIE = 昇腾版的 vLLM/TGI**：它吃 HuggingFace 权重，吐 OpenAI 风格的推理服务接口；底层不靠 CUDA，而是靠 **CANN + ATB** 把 Transformer 算子映射到昇腾 NPU（达芬奇架构 Cube/Vector 单元）。本篇的 RC2 是 2024 年的早期版本，部署形态是**容器镜像 + `mindieservice_daemon` 常驻进程**。

## 1. 地基/前置：昇腾推理软件栈五层

要理解后面那一堆 `source xxx/set_env.sh`，先建立软件栈的分层心智模型。NPU 不像 GPU 有统一的 CUDA Runtime，昇腾把能力切成了多层，每层一个 `set_env.sh`：

```
 应用层   ┌─────────────────────────────────────────────┐
          │  MindIE-Service (mindieservice_daemon)       │  ← 对外 HTTP/gRPC 服务、调度、续批
          │  llm_model (模型脚本: qwen-chat 等结构定义)    │  ← 把 HF 权重映射成昇腾图
          └─────────────────────────────────────────────┘
 加速库   ┌─────────────────────────────────────────────┐
          │  ATB (Ascend Transformer Boost)              │  ← 融合算子: FlashAttention/RoPE/RMSNorm
          │  nnal/atb/set_env.sh                          │
          └─────────────────────────────────────────────┘
 基础软件 ┌─────────────────────────────────────────────┐
          │  CANN ascend-toolkit (类比 CUDA Toolkit)      │  ← 编译器/算子库/Runtime/HCCL 集合通信
          │  ascend-toolkit/set_env.sh                    │
          └─────────────────────────────────────────────┘
 驱动+固件 ┌─────────────────────────────────────────────┐
          │  Ascend Driver + npu-smi                      │  ← 宿主机装, 容器里挂载进去, 不在镜像内
          └─────────────────────────────────────────────┘
 硬件     ┌─────────────────────────────────────────────┐
          │  800I A2 (aarch64) 8×NPU 推理卡               │
          └─────────────────────────────────────────────┘
```

**关键认知**：驱动(Driver)和 `npu-smi` **装在宿主机**，容器通过 `-v` 把它们挂载进去（见第 3 节）；而 CANN/ATB/MindIE **装在容器**内（镜像自带 + `install_and_enable_cann.sh`）。这就是为什么命令里既要 `source` 一堆环境变量，又要挂载宿主目录——**驱动在外、运行时在内，二者必须版本匹配**。

> 类比对照：GPU 栈 `NVIDIA Driver → CUDA → cuDNN/cuBLAS → TensorRT-LLM/vLLM`，昇腾栈 `Ascend Driver → CANN → ATB → MindIE`。`npu-smi` 对应 `nvidia-smi`，`ASCEND_VISIBLE_DEVICES` 对应 `CUDA_VISIBLE_DEVICES`/`NVIDIA_VISIBLE_DEVICES`。

| GPU 世界 | 昇腾世界 | 作用 |
|----------|---------|------|
| `nvidia-smi` | `npu-smi` | 查卡 / 看显存 / 看利用率 |
| CUDA Toolkit | CANN ascend-toolkit | 编译器+Runtime+算子库 |
| cuDNN / cuBLAS | ATB | Transformer 融合算子 |
| TensorRT-LLM / vLLM | MindIE-Service | 推理服务、续批、KV-Cache 调度 |
| `CUDA_VISIBLE_DEVICES` | `ASCEND_VISIBLE_DEVICES` | 选卡 |
| NCCL | HCCL | 多卡集合通信(AllReduce 等) |

> 📌 官方文档：
> - 概览：https://www.hiascend.com/document/detail/zh/mindie/10RC2/whatismindie/mindie_what_0001.html
> - 镜像：https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f

## 2. 拿镜像：三条路与 tag 解读

镜像 tag `mindie:1.0.RC2-800I-A2-aarch64` 把关键信息全编进去了，逐段拆：

```
 mindie : 1.0.RC2 - 800I-A2 - aarch64
   │        │          │         └── CPU 架构: ARM64 (鲲鹏), 不是 x86!
   │        │          └── 硬件型号: Atlas 800I A2 推理服务器
   │        └── 版本: 1.0 Release Candidate 2 (早期候选版)
   └── 产品: MindIE 推理引擎
```

> ⚠️ `aarch64` 意味着这是 ARM 镜像，宿主机必须是鲲鹏等 ARM CPU，x86 机器拉下来跑不了。

三种获取方式（原文真料）：

```bash
# 路1: 从华为云 SWR 镜像仓直接拉(需登录授权)
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.RC2-800I-A2-aarch64

# 路2: 从已有机器同步打包好的 tar (rsync 断点续传 -P, 走 ssh)
rsync -P --rsh=ssh -r root@192.168.16.xxx:/root/mindie-1.0.rc2.tar .
# 拿到 tar 后: docker load -i mindie-1.0.rc2.tar

# 路3: AscendHub 网页下载(见上方文档链接)
```

`rsync -P` 中 `-P` = `--partial --progress`：断了能续传 + 显示进度，**几十 GB 的镜像 tar 用它最稳**。

## 3. 起容器：逐参数拆解那串 `docker run`

这是全篇最容易抄错的地方。先看原文命令，再逐行解释**为什么**：

```bash
docker run -it -d --name mindie-rc2-45 --net=host \
  -e ASCEND_VISIBLE_DEVICES=4,5 \
  -p 1925:1025 \
  --shm-size=32g \
  -w /workspace \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /data/model_from_hf:/workspace/model \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.RC2-800I-A2-aarch64 \
  /bin/bash

docker exec -it mindie-rc2-45 bash
```

| 参数 | 作用 | 为什么必须有 |
|------|------|------------|
| `-it -d` | 交互式 + 后台常驻 | 服务要长跑，`-d` 不占终端 |
| `--net=host` | 容器直接用宿主网络栈 | 多卡 HCCL 通信走宿主网卡，避开 NAT 损耗 |
| `-e ASCEND_VISIBLE_DEVICES=4,5` | 只把 4、5 号 NPU 给容器 | 卡隔离，多容器分卡互不抢 |
| `-p 1925:1025` | 宿主 1925 → 容器 1025 | 容器内服务默认监听 1025，对外映射避免端口冲突 |
| `--shm-size=32g` | 共享内存 32G | 多卡张量并行靠 `/dev/shm` 传中间张量，太小会 OOM/挂死 |
| `-v .../driver` | 挂宿主驱动进容器 | **驱动不在镜像内**，必须挂载且与固件版本匹配 |
| `-v .../npu-smi` | 挂 `npu-smi` 二进制 | 容器内也能 `npu-smi info` 查卡 |
| `-v /data/model_from_hf:/workspace/model` | 挂模型权重 | 权重几十G，不打进镜像，外挂只读复用 |

设备可见性的隔离原理（多容器分卡的核心）：

```
   宿主机 8 张 NPU:  [0][1][2][3][4][5][6][7]
                              │   │       │  │
   容器A (-e ...=4,5) ────────┘   │       │  │   只能看到 2 张卡
   容器B (-e ...=6,7) ────────────────────┘  │   互不可见, 资源隔离
   容器C (-e ...=0..7) ─────────全部────────────  独占 8 卡跑 72B
```

> 💡 多个 `docker run` 端口各不相同（1925/1025/1825/1525）正是为了**同一台机器上并行起多个推理服务**，每个吃不同的卡和端口。

## 4. 装 CANN + 激活环境变量

进容器后第一件事：装 CANN，再逐层 `source`：

```bash
cd /opt/package
# 安装CANN包 (镜像自带安装脚本)
source ./install_and_enable_cann.sh

# 逐层激活软件栈(顺序即依赖顺序: 底层先于上层)
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # CANN 基础: 编译器/Runtime/HCCL
source /usr/local/Ascend/nnal/atb/set_env.sh         # ATB 融合算子库
source /usr/local/Ascend/mindie/set_env.sh           # MindIE 服务
source /usr/local/Ascend/llm_model/set_env.sh        # 模型脚本库(结构定义)
```

**为什么是 `source` 而不是执行？** 这些脚本只设置环境变量（`PATH`/`LD_LIBRARY_PATH`/`ASCEND_HOME` 等）。用 `source`（即 `.`）让变量留在**当前 shell**；若 `bash xxx.sh` 在子进程里设了，退出就丢了。

**顺序为什么重要？** 上层依赖下层的库路径。`ATB` 链接 CANN 的 `.so`，`MindIE` 又链接 ATB——先 source 底层，`LD_LIBRARY_PATH` 才能被上层正确追加。

```
LD_LIBRARY_PATH 累积:
  source toolkit  →  +CANN/lib64
  source atb      →  +CANN/lib64 +ATB/lib
  source mindie   →  +CANN/lib64 +ATB/lib +mindie/lib
  source llm_model→  + 模型脚本可被 import
```

## 5. config.json：让 daemon 知道加载哪个模型

服务读这个文件决定加载哪个模型、用几张卡、留多少显存：

```bash
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 关键: 把模型权重路径指到挂载进来的目录, 例如
/workspace/model/Qwen1.5-7B-Chat/
```

config.json 的心智模型（字段名以实际版本为准，这里讲**作用**）：

```
config.json
 ├─ modelWeightPath   → /workspace/model/Qwen1.5-7B-Chat/  指向 HF 权重目录
 ├─ worldSize         → 张量并行卡数 (TP), 与可见卡数一致
 ├─ npuMemSize        → 每卡留给 KV-Cache 的显存(GB)
 ├─ maxSeqLen / maxBatchSize → 续批上限
 └─ port              → 对外端口(默认 1025)
```

`worldSize` 与 KV-Cache 的关系（为什么 7B 用 2、72B 用 8）：模型权重按张量并行切到 N 张卡，**每张卡只放 1/N 的权重**，剩余显存给 KV-Cache。

数值手算（Qwen1.5-7B, FP16）：
- 权重 ≈ $7\text{B}\times 2\text{B} = 14\text{GB}$，2 卡 TP 后每卡 $\approx 7\text{GB}$。
- 800I A2 单卡约 64GB，则每卡 KV-Cache 可用 $\approx 64-7-\text{(激活/碎片)} \approx 50\text{GB}$ → 这就是 `--npu_mem_size=15` 之类参数想留的那块缓存。
- KV-Cache 单 token 大小 $= 2(\text{K,V})\times L_{\text{layer}}\times d_{\text{model}}\times 2\text{B}$。Qwen1.5-7B 有 $L=32, d=4096$：$2\times32\times4096\times2 = 0.5\text{MB/token}$，TP=2 后每卡 $0.25\text{MB/token}$ → 15GB 可缓约 $15\text{GB}/0.25\text{MB}=6$ 万 token。

## 6. 起服务：daemon 启动链路

```bash
export MIES_PYTHON_LOG_TO_FILE=1     # 日志落文件, 便于排查
export MIES_PYTHON_LOG_TO_STDOUT=1   # 同时打到 stdout
export PYTHONPATH=/usr/local/Ascend/llm_model:$PYTHONPATH  # 让 python 找到模型脚本
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

启动链路 ASCII：

```
./mindieservice_daemon
        │ 读 conf/config.json
        ▼
  解析 modelWeightPath / worldSize / npuMemSize
        │
        ▼  按 PYTHONPATH 找到 llm_model 里对应结构(qwen-chat...)
  加载 HF 权重 → 切张量到 worldSize 张 NPU (HCCL 建通信域)
        │
        ▼  ATB 编译融合算子(FlashAttention/RoPE/RMSNorm) → 编译图缓存
  预分配 KV-Cache (按 npuMemSize)
        │
        ▼
  监听端口(默认 1025) ── 接收请求 ── Continuous Batching 续批调度 ── 返回 token 流
```

`AIE_LLM_CONTINUOUS_BATCHING=1`（见第 8 节）：开启**连续批处理**——新请求随时插入正在跑的 batch，已完成的请求立刻让位，避免传统静态 batch「等最慢的那条」造成的 NPU 空转，是吞吐的关键开关（同 vLLM 的 continuous batching 思想）。

## 7. 固化镜像：把装好的环境打包复用

每次新机器都装 CANN 太慢——把调好的容器 `commit` 成新镜像，存 tar 分发：

```bash
# 把容器 365815a95f16 固化为带 CANN 的新镜像
docker commit -a "guodong" -m "mindie-1.0.RC2" 365815a95f16 harbor/ascend/mindie-base:1.0.RC2

# 导出为 tar
docker save -o mindie-base.tar harbor/ascend/mindie-base:1.0.RC2

# 同步到其它机器
rsync -P --rsh=ssh -r root@192.168.16.211:/home/workspace/mindie-base.tar .
```

之后用固化镜像起容器（`--rm` 退出即删，适合临时验证）：

```bash
docker run -it --rm \
  -e ASCEND_VISIBLE_DEVICES=2,3 \
  -p 1025:1025 \
  --shm-size=32g \
  -w /workspace \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /data/model_from_hf:/workspace/model \
  harbor/ascend/mindie-base:1.0.RC2 \
  /bin/bash
```

> `harbor/...` 是私有 Harbor 镜像仓的前缀，对应团队内网镜像；`-p 192.168.16.xx:1025:1025` 可绑定到具体宿主 IP（原文注释里的写法），只对外暴露指定网卡。

## 8. 一键脚本 `llm-server3.sh`：四参数拉起服务

把第 4~6 步封装成脚本，通过 `-v` 挂进容器后直接当 ENTRYPOINT 跑。两个真实规格：

**7B 单机双卡（每卡 15GB KV-Cache）：**

```bash
docker run -it --rm \
  -e ASCEND_VISIBLE_DEVICES=6,7 \
  -p 1525:1025 \
  --env AIE_LLM_CONTINUOUS_BATCHING=1 \
  --shm-size=32g \
  -w /workspace \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /data/model_from_hf/Qwen1.5-7B-Chat:/workspace/model \
  -v /home/workspace/llm-server3.sh:/workspace/llm-server.sh \
  -v /home/workspace/mindservice.log:/usr/local/Ascend/mindie/latest/mindie-service/logs/mindservice.log \
  harbor/ascend/mindie-base:1.0.RC2 \
  /workspace/llm-server.sh \
  --model_name=qwen-chat \
  --model_weight_path=/workspace/model \
  --world_size=2 \
  --npu_mem_size=15
```

**72B 单机八卡（每卡 8GB KV-Cache）：**

```bash
docker run -it --rm \
  -e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -p 1525:1025 \
  --env AIE_LLM_CONTINUOUS_BATCHING=1 \
  --shm-size=32g \
  -w /workspace \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /data/model_from_hf/Qwen2-72B-Instruct:/workspace/model \
  -v /home/workspace/llm-server3.sh:/workspace/llm-server.sh \
  -v /home/workspace/mindservice.log:/usr/local/Ascend/mindie/latest/mindie-service/logs/mindservice.log \
  harbor/ascend/mindie-base:1.0.RC2 \
  /workspace/llm-server.sh \
  --model_name=qwen-chat \
  --model_weight_path=/workspace/model \
  --world_size=8 \
  --npu_mem_size=8
```

四个参数对照：

| 参数 | 7B 取值 | 72B 取值 | 含义 |
|------|--------|---------|------|
| `--model_name` | qwen-chat | qwen-chat | 模型结构名(对应 llm_model 脚本) |
| `--model_weight_path` | /workspace/model | /workspace/model | 容器内权重路径(挂载点) |
| `--world_size` | 2 | 8 | 张量并行卡数(=可见卡数) |
| `--npu_mem_size` | 15 | 8 | 每卡 KV-Cache 显存(GB) |

为什么 72B 用 8 卡而 npu_mem 反而更小？72B FP16 权重 $\approx 144\text{GB}$，8 卡 TP 后每卡放 $\approx 18\text{GB}$ 权重，单卡剩余显存被权重吃掉更多，留给 KV-Cache 的自然更小（8GB）。这就是 `world_size↑` 与 `npu_mem_size↓` 的平衡。

`mindservice.log` 用 `-v` 挂出来到宿主 `/home/workspace/mindservice.log`：**服务一挂就看这个日志**，是排查首选。

## 实操命令合集（原文真料·可直接复制）

```bash
# ---- 1. 同步/拉镜像 ----
rsync -P --rsh=ssh -r root@192.168.16.xxx:/root/mindie-1.0.rc2.tar .
# tag: swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.RC2-800I-A2-aarch64

# ---- 2. 起容器 ----
docker run -it -d --name mindie-rc2-45 --net=host \
  -e ASCEND_VISIBLE_DEVICES=4,5 -p 1925:1025 --shm-size=32g -w /workspace \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /data/model_from_hf:/workspace/model \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.RC2-800I-A2-aarch64 /bin/bash
docker exec -it mindie-rc2-45 bash

# ---- 3. 装 CANN + 激活环境 ----
cd /opt/package
source ./install_and_enable_cann.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/llm_model/set_env.sh

# ---- 4. 配模型路径 ----
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 指向 /workspace/model/Qwen1.5-7B-Chat/

# ---- 5. 起服务 ----
export MIES_PYTHON_LOG_TO_FILE=1
export MIES_PYTHON_LOG_TO_STDOUT=1
export PYTHONPATH=/usr/local/Ascend/llm_model:$PYTHONPATH
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon

# ---- 6. 固化镜像 ----
docker commit -a "guodong" -m "mindie-1.0.RC2" 365815a95f16 harbor/ascend/mindie-base:1.0.RC2
docker save -o mindie-base.tar harbor/ascend/mindie-base:1.0.RC2
```

## 常见问题 / 坑

| 现象 | 根因 | 解决 |
|------|------|------|
| 镜像拉下来跑不了 / `exec format error` | tag 是 `aarch64`，宿主是 x86 | 必须在鲲鹏等 ARM 机器上跑 |
| 容器内 `npu-smi info` 找不到卡 | 没挂 driver / npu-smi，或驱动与固件版本不匹配 | 检查两个 `-v` 挂载；宿主驱动版本要 ≥ CANN 要求 |
| 多卡启动卡死 / shm 报错 | `--shm-size` 太小，TP 中间张量塞不下 | 提到 `32g`；`--net=host` 也别漏 |
| `import` 模型脚本失败 | 没设 `PYTHONPATH=/usr/local/Ascend/llm_model` | 起服务前 export |
| 服务起来但 OOM | `npu_mem_size` 设太大，权重+KV 超单卡显存 | 减小 `npu_mem_size`，或加大 `world_size` 多卡分摊 |
| `source set_env.sh` 后还是找不到库 | 顺序错(上层先于下层)或用了 `bash` 执行 | 严格按 toolkit→atb→mindie→llm_model 顺序 `source` |
| 端口冲突起不来 | 多容器都映射到同一宿主端口 | 改 `-p` 宿主侧端口(1925/1825/1525 各异) |
| 吞吐低、NPU 利用率忽高忽低 | 没开续批 | 加 `--env AIE_LLM_CONTINUOUS_BATCHING=1` |
| 服务挂了不知原因 | 没把日志挂出来 | `-v .../mindservice.log` 挂到宿主再看 |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 硬件与生态：[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/GPU工作原理]]
- 多卡通信：[[ai-infra/网络/集合通信原语]]（HCCL/AllReduce）· [[ai-infra/网络/InfiniBand]]
- 推理引擎对照：[[llm-inference/README]] · [[llm-inference/vllm/README]]（Continuous Batching 对照）· [[llm-inference/PD分离]] · [[llm-inference/解码策略]]
- 显存/缓存原理：[[llm-optimizer/kv-cache]] · [[llm-optimizer/FlashAttention]] · [[docs/transformer内存估算]]
- 模型结构：[[llm-algo/transformer/模型架构]] · [[llm-algo/旋转编码RoPE]]（ATB 融合的 RoPE）
- 量化(降显存):[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 性能评估：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
