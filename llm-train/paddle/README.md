# PaddlePaddle 大模型训练环境与分布式训练

> 飞桨（PaddlePaddle）是百度自研的深度学习框架，配合 PaddleNLP 套件提供国产软硬件栈下的大模型训练与并行能力。本文从环境搭建讲到 4D 并行原理。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/paddle/paddlenlp/README]] [[llm-train/README]] [[ai-framework/paddlepaddle/README]] [[llm-train/pytorch/distribution/README]]

## 阅读地图

| 小节 | 你将搞清楚 | 关键词 |
|------|-----------|--------|
| 0 锚点 | PaddlePaddle 在大模型训练里是什么角色 | 飞桨 / PaddleNLP |
| 1 地基 | 为什么需要专门的框架与镜像 | 动静统一 / 算子 / CUDA 匹配 |
| 2 整体架构 | 框架分层与训练栈全貌 | Python API / 执行器 / Kernel |
| 3 镜像与安装 | Docker 镜像、版本对齐、whl 选择 | CUDA/cuDNN/TRT/NCCL |
| 4 验证与排错 | 安装后怎么确认环境可用 | run_check / 版本后缀 |
| 5 develop 版 | 何时用每日构建版 | nightly / paddlenlp |
| 6 分布式并行 | 4D 并行原理与启动方式 | DP/TP/PP/Sharding |
| 7 配置示例 | docker run 与启动参数含义 | shm-size / launch |
| 坑 | 高频报错与规避 | 后缀错配 / NCCL / 共享内存 |

## 0. 一句话锚点

**PaddlePaddle = 一个和 PyTorch 同层级的深度学习框架；做大模型训练时你真正用的是它上面的 PaddleNLP 套件 + 它内置的分布式并行引擎（`paddle.distributed`）。** 本目录的核心工作量，其实是把"环境装对"——因为框架版本、CUDA、cuDNN、TensorRT、NCCL 必须严格对齐，错一个就跑不起来。

```
                 你要训练一个大模型
                         │
        ┌────────────────┴────────────────┐
        │                                  │
   模型与训练脚本                      底层执行引擎
   (PaddleNLP: Llama/Bloom/...)        (PaddlePaddle 框架)
        │                                  │
        └──────────── 都跑在 ─────────────┘
                         │
              GPU + CUDA + cuDNN + NCCL
                  (由 Docker 镜像锁定版本)
```

## 1. 地基：它解决什么问题

### 1.1 为什么不直接 `pip install` 就完事

深度学习框架不是纯 Python，它底层是一堆**编译好的 C++/CUDA 算子（Kernel）**。这些算子在编译时就**绑定了具体的 CUDA / cuDNN 版本**。所以：

- 你机器上的 **GPU 驱动** 决定了能支持的 **CUDA 上限**；
- 你装的 **paddlepaddle-gpu 包** 是针对**某个 CUDA 版本编译**的（whl 名字里的 `post117` 就是"为 CUDA 11.7 编译"的意思）；
- 三者错配 → 要么 import 报 `cudaErrorInsufficientDriver`，要么算子找不到符号。

> 这就是为什么官方推荐用 **Docker 镜像**：镜像里 CUDA / cuDNN / TensorRT / 框架已经一次性配好且互相匹配，省掉地狱级的版本对齐。

### 1.2 飞桨的两个差异化卖点

| 卖点 | 含义 | 对训练的意义 |
|------|------|-------------|
| 动静统一 | 同一份代码既能动态图调试，又能转静态图（`to_static`）加速/部署 | 调试用动态、上量产转静态 |
| 国产软硬件适配 | 对昇腾 NPU、昆仑芯等国产芯片有官方后端 | 信创/算力替代场景的主力框架之一 |

## 2. 整体架构：训练栈全貌

```
┌───────────────────────────────────────────────────────┐
│  上层：PaddleNLP（大模型套件）                          │
│   Trainer / AutoModel / 数据流水线 / 预置 Llama·Bloom   │
├───────────────────────────────────────────────────────┤
│  中层：PaddlePaddle Python API                          │
│   paddle.nn (组网) │ paddle.optimizer │ paddle.distributed│
├───────────────────────────────────────────────────────┤
│  执行层：动态图 Eager  /  静态图 Executor               │
│   自动微分(autograd) · 计算图 · 算子调度                 │
├───────────────────────────────────────────────────────┤
│  Kernel 层：C++/CUDA 算子库 (Phi)                       │
├───────────────────────────────────────────────────────┤
│  硬件后端：CUDA + cuDNN + NCCL  /  昇腾 CANN  /  XPU     │
└───────────────────────────────────────────────────────┘
```

理解这张图的关键：**你写的 Python 只是"描述计算"，真正干活的是底层算子；版本对齐对齐的就是最底下两层。**

## 3. 镜像与安装（本目录的核心）

### 3.1 拉取并启动官方 GPU 镜像

镜像源用的是百度云仓库 `registry.baidubce.com`（国内拉取快）。镜像标签自带了完整的版本签名：

```
registry.baidubce.com/paddlepaddle/paddle:<框架版本>-gpu-cuda<X>-cudnn<Y>-trt<Z>
```

例如 `paddle:2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4` 表示：飞桨 2.5.1 + CUDA 11.7 + cuDNN 8.4 + TensorRT 8.4，四件套已对齐。

```bash
nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4

# 常驻容器（开发用，自动重启、挂载工作目录）
docker run -dt \
  --name paddle \
  --restart=always \
  --gpus all \
  --network=host \
  --shm-size 4G \
  -v /home/guodong.li/workspace:/paddle \
  registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4 \
  /bin/bash

docker exec -it paddle bash
```

或者用临时容器（退出即删，适合一次性实验，`--rm`）：

```bash
sudo docker run -it --rm \
  --gpus all \
  --network=host \
  --shm-size 4G \
  -v /home/guodong.li/workspace:/workspace \
  registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4 \
  /bin/bash
```

### 3.2 版本匹配硬约束（来自镜像/官方说明）

- CUDA 工具包 **11.7** 配合 cuDNN **v8.4.1**；如需 Paddle-TensorRT 推理，需配合 **TensorRT 8.4.2.4**；
- 分布式多卡需 **NCCL ≥ 2.7**；
- 需要 GPU **算力（Compute Capability）≥ 3.5** 的硬件。

> 这些是该镜像对应的版本组合，换镜像标签则整套数字都会变。**具体版本以官方文档/镜像标签为准，不要照搬到别的镜像。**

### 3.3 用 conda 隔离 + pip 安装（非 Docker 路线）

如果不用镜像而是裸机/已有 CUDA 环境，先建独立 conda 环境避免污染：

```bash
conda create -n paddle python=3.8 -y
conda activate paddle

# 关键：whl 名字里的 post117 = 为 CUDA 11.7 编译，必须和本机 CUDA 对上
python -m pip install paddlepaddle-gpu==2.5.1.post117 \
  -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

`-f <index-url>` 的作用：告诉 pip 去飞桨官方 wheel 索引页找包（PyPI 上不一定有带 CUDA 后缀的版本）。索引 URL 里的 `mkl`（数学库）、`avx`（CPU 指令集）也是要和本机匹配的维度。

### 3.4 卸载

```bash
python3 -m pip uninstall paddlepaddle-gpu
```

> 重装/换版本前先卸载干净，避免 CPU 版与 GPU 版残留共存导致 import 时加载到错的那个。

## 4. 验证与排错

装完**第一件事就是自检**，别急着跑训练脚本：

```python
import paddle
paddle.utils.run_check()
```

它会做端到端体检：能否 import、能否找到 GPU、能否在 GPU 上跑一个最小算子、（多卡时）能否走通通信。输出大致是 "PaddlePaddle is installed successfully!"。

```
run_check() 内部逻辑（概念示意）
   ┌─────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────┐
   │ import  │→ │ 检测GPU  │→ │ 跑一个小算子 │→ │ 多卡通信  │
   │  成功?  │  │ 可见?    │  │ 数值正确?    │  │ (可选)    │
   └─────────┘  └──────────┘  └──────────────┘  └───────────┘
        任一步失败 → 报错信息直接指向哪一层没配好
```

## 5. develop（每日构建）版本

当稳定版还没合入你需要的新模型/新特性时，用 nightly：

```bash
# develop 版框架（注意版本号是占位符 0.0.0）
python -m pip install paddlepaddle==0.0.0 \
  -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
python -m pip install paddlepaddle==0.0.0 \
  -f https://www.paddlepaddle.org.cn/whl/mac/cpu/develop.html

# develop 版 PaddleNLP（--pre 允许预发布版）
pip install --pre --upgrade paddlenlp \
  -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

权衡：develop 版有最新模型/修复，但**稳定性无保证**；生产训练优先用 tagged 稳定版，验证新特性才用 develop。

## 6. 分布式并行（大模型训练的真正难点）

单卡放不下大模型，必须切分。飞桨 `paddle.distributed` 提供和主流框架同构的 **4D 混合并行**：

| 并行维度 | 切什么 | 解决什么 | 通信代价 |
|----------|--------|----------|----------|
| 数据并行 DP | 切 batch（每卡一份完整模型） | 提吞吐 | 梯度 AllReduce |
| 张量并行 TP | 切单层权重矩阵（行/列切） | 单层放不下 | 层内 AllReduce，最重，需 NVLink |
| 流水线并行 PP | 按层切成 stage | 层数太多放不下 | stage 间点对点（Send/Recv） |
| 分组切片 Sharding | 切优化器状态/梯度/参数 | 省显存（类 ZeRO） | AllGather/ReduceScatter |

```
4D 并行的卡布局（示例：DP=2, PP=2, TP=2，共 8 卡）

          流水线 stage0          流水线 stage1
        ┌──────────────┐       ┌──────────────┐
 DP组0  │ GPU0 ─TP─ GPU1│ ─PP→ │ GPU2 ─TP─ GPU3│
        └──────────────┘       └──────────────┘
        ┌──────────────┐       ┌──────────────┐
 DP组1  │ GPU4 ─TP─ GPU5│ ─PP→ │ GPU6 ─TP─ GPU7│
        └──────────────┘       └──────────────┘
        TP 在组内（机内 NVLink）·PP 跨 stage 传激活·DP 跨组同步梯度
```

放置原则（和 PyTorch/Megatron 一致）：**通信最重的 TP 放在机内同一节点（走 NVLink/NVSwitch），PP 和 DP 可跨机（走 IB/RoCE）。** 假设单卡显存 80GB、模型参数需 200GB，则至少需要 TP/PP/Sharding 把 200GB 摊到多卡上才放得下。

启动多卡训练用 launch 工具（概念）：

```bash
# 单机多卡：用 launch 自动拉起 N 个进程并设好 rank/通信环境变量
python -m paddle.distributed.launch --gpus "0,1,2,3" train.py
```

> 具体的并行度参数名（如 tensor_parallel_degree / pipeline_parallel_degree）、Trainer 配置项**以 PaddleNLP 官方文档/源码为准**，不同版本会有差异。

## 7. 典型配置：docker run 参数为什么这么写

| 参数 | 作用 | 为什么需要 / 怎么权衡 |
|------|------|----------------------|
| `--gpus all` | 把所有 GPU 透传进容器 | 不写则容器看不到 GPU，run_check 直接失败 |
| `--network=host` | 容器共用宿主机网络栈 | 多机分布式通信（NCCL/IB）省去端口映射麻烦 |
| `--shm-size 4G` | 扩大 `/dev/shm` 共享内存 | DataLoader 多进程靠共享内存传数据，默认 64MB 太小会 OOM/卡死 |
| `-v 宿主:容器` | 挂载工作目录 | 代码/数据/checkpoint 持久化，容器删了不丢 |
| `--restart=always` | 容器异常退出自动重启 | 长跑常驻开发机用；一次性实验用 `--rm` 反之 |
| `-dt` / `-it` | 后台守护 / 交互终端 | 守护用 `-dt` 再 exec 进入；交互用 `-it` |

显存换算例子：单卡 24GB，跑一个 7B 模型全参微调时，参数(fp16)≈14GB + 优化器状态(Adam, fp32)≈56GB 远超 24GB → 必须上 Sharding 或 LoRA，这也是为什么大模型几乎都走分布式/PEFT。

## 常见问题/坑

| 现象 | 根因 | 处理 |
|------|------|------|
| import 报 CUDA 驱动不足 | whl 的 `postXXX` 后缀 CUDA 版本高于本机驱动支持 | 选更低后缀的 whl 或升级驱动；优先用对齐好的 Docker 镜像 |
| `run_check` 找不到 GPU | docker 没加 `--gpus all` / 缺 nvidia-container-toolkit | 补 `--gpus all`，确认宿主已装 NVIDIA Container Toolkit |
| 多卡训练 NCCL 初始化失败 | NCCL < 2.7 或网络不通 | 升级 NCCL；多机加 `--network=host`，检查 IB/RoCE |
| DataLoader 卡死/共享内存报错 | `--shm-size` 太小 | 加大到 8G+ |
| pip 装到 CPU 版还误以为有 GPU | CPU/GPU 包残留共存 | 先 `uninstall` 干净再装 GPU 版 |
| 装了 `0.0.0` 版本很怪 | 那是 develop（nightly）占位版本号 | 正常现象，生产请换稳定版 |
| 国产芯片（昇腾/昆仑）跑不起来 | 用了 GPU 镜像而非对应后端包 | 换 CANN/XPU 对应的飞桨发行版 |

## 🔗 跳转链接

- [[00-知识地图]]
- [[llm-train/paddle/paddlenlp/README]] — PaddleNLP 大模型套件（Llama/Bloom/Baichuan 训练）
- [[llm-train/README]] — 大模型训练总览
- [[ai-framework/paddlepaddle/README]] — 飞桨框架本体
- [[llm-train/pytorch/distribution/README]] — PyTorch 分布式并行（对照理解 4D 并行）
- [[ai-framework/megatron-lm/README]] — Megatron 张量/流水线并行参考实现
- [[ai-infra/网络/NCCL]] — 多卡通信库
