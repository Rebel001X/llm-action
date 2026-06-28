# 基于 Megatron-LM 的 GPT-2 模型训练（从零到端到端）

> 用 NVIDIA Megatron-LM 把一个 345M 参数的 GPT-2 从「数据」一路训练到「权重 / 推理」，并借此把张量并行(TP)、流水并行(PP)、数据并行(DP) 三种切分方式讲透。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/megatron-lm/README]] [[llm-train/README]] [[llm-train/pytorch/distribution/README]] [[ai-framework/deepspeed/README]]

## 阅读地图

| 你想知道 | 跳到 |
|---|---|
| 这套东西到底在做什么、为什么用 Megatron | [0](#0-一句话锚点) · [1](#1-地基为什么需要-megatron-lm) |
| 一次完整训练的目录/调用链长什么样 | [2](#2-整体架构与调用链) |
| 数据从原始网页到 .bin/.idx 怎么变 | [3](#3-数据流水线原始文本--token-索引) |
| TP / PP / DP 三种并行到底切了什么 | [4](#4-三种并行tppp--dp一图看懂) |
| 单卡 / 4DP / 2TP+2DP 怎么跑、显存差多少 | [5](#5-三种运行模式与显存对比) |
| 关键启动参数是干嘛的、怎么权衡 | [6](#6-关键参数说明讲含义不背默认值) |
| checkpoint 的分片目录怎么读 | [7](#7-权重checkpoint目录结构) |
| 常见报错和坑 | [常见问题与坑](#常见问题与坑) |

## 0. 一句话锚点

**Megatron-LM 是 NVIDIA 出品的「超大 Transformer 训练框架」**：它在 PyTorch 之上，把一个大到单卡放不下的模型，沿着**张量维度、层维度、数据维度**三个方向切开，分散到很多张 GPU 上协同训练。本目录用经典的 **GPT-2 345M** 作为最小可跑通的例子，串起「装环境 → 拉代码 → 处理数据 → 启动训练 → 评估推理」的完整闭环。

## 1. 地基：为什么需要 Megatron-LM

普通的 `torch` 单卡训练，模型参数、激活值、优化器状态全都要塞进**一张 GPU 的显存**。GPT-2 345M 还能塞下（FP16 权重约 677 MB，加上 Adam 优化器的一二阶动量、梯度、激活，训练态会膨胀到几个 GB），但是到了几十亿、上千亿参数，单卡显存（哪怕 A100/A800 的 80 GB）根本装不下。这时有三个东西在抢显存：

```
单步训练的显存占用 ≈ 参数 + 梯度 + 优化器状态 + 激活值
                    │       │        │              │
   FP16: 2B/参数    │  2B/参数  Adam: 一阶+二阶动量   随 batch、序列长度、层数线性增长
                            (FP32 主副本通常 4+4+4=12B/参数)
```

> 经验法则：用 Adam + 混合精度训练，**每个参数大约要 16~20 字节**的常驻显存（权重、梯度、FP32 主权重、一阶/二阶动量），**激活值另算**。175B 模型光优化器状态就 ~2.8 TB —— 这就是必须做并行切分的根本原因。

Megatron-LM 解决的核心问题就是：**当「一张卡装不下」或「一张卡训练太慢」时，如何把工作正确地拆到多卡 / 多机，并把切分带来的通信开销压到最低。** 它给出的三板斧是张量并行(TP)、流水并行(PP)、数据并行(DP)，下文第 4 节详解。

## 2. 整体架构与调用链

一次 GPT-2 预训练，从启动脚本到反向传播的调用链大致如下（函数名以官方源码为准，这里给出「类别」而非逐行）：

```
examples/pretrain_gpt.sh                 ← 启动脚本：设环境变量 + 拼参数
   │  torchrun / python -m torch.distributed.run
   ▼
pretrain_gpt.py : pretrain(...)          ← 训练主入口
   │
   ├─ initialize_megatron()              ← ① 解析 arguments.py、初始化进程组(NCCL)
   │     └─ 建立 TP / PP / DP 三套通信 group（mpu，model parallel unit）
   │
   ├─ model_provider()                   ← ② 按 TP/PP 切分构建 GPTModel
   │
   ├─ train_valid_test_datasets_provider ← ③ 读 .bin/.idx，建 train/valid/test 数据集
   │
   └─ train()                            ← ④ 主训练循环
         每个 iteration：
           forward_step  → loss
           backward_step → grad           （TP：层内 all-reduce；PP：跨 stage 收发激活）
           optimizer.step()               （DP：跨副本 all-reduce 梯度）
           每 save_interval 存 checkpoint
```

关键直觉：**TP 与 PP 改变「模型怎么放」，DP 改变「数据怎么喂」**。三者可以叠加，总卡数 `world_size = TP × PP × DP`。

## 3. 数据流水线：原始文本 → token 索引

Megatron 不直接吃 `.txt`，它要的是预先 token 化、内存映射友好的二进制格式。整条流水线（详见同目录 [gpt-data-preprocess.md](./gpt-data-preprocess.md)）：

```
原始网页(OpenWebText)
   │  blacklist_urls.py          去黑名单/坏 URL
   ▼ merge_data.py               把一堆 .txt 合并成「一行一条 {"text": ...}」的 JSON
merged_output.json
   │  cleanup_dataset.py         ftfy 修复编码 + 英文检测 + 丢弃 <128 token 的短文档
   ▼ find/group/remove_duplicates  LSH 近似去重
merged_cleand.json  →  shuf  →  train_data.json
   │
   ▼ tools/preprocess_data.py    GPT2 BPE 分词 + 末尾追加 <eod>
my-gpt2_text_document.bin   ← 紧密排列的 token id（uint16/uint32）
my-gpt2_text_document.idx   ← 文档边界、偏移等索引，供 mmap 随机寻址
```

`preprocess_data.py` 的核心参数（讲用途，**具体名字/默认值以官方 `--help` 为准**）：

| 参数 | 作用 |
|---|---|
| `--input` | 输入的 JSON（一行一条 `{"text": ...}`） |
| `--output-prefix` | 输出前缀，会生成 `<prefix>_text_document.bin/.idx` |
| `--vocab-file` / `--merge-file` | GPT-2 BPE 的词表与合并规则 |
| `--tokenizer-type` | 这里用 `GPT2BPETokenizer` |
| `--append-eod` | 每篇文档末尾追加「文档结束」特殊 token，训练时用于切样本/重置注意力 |
| `--workers` / `--chunk-size` | 多进程分词的并行度与切块粒度，调大可加速 |

> 数值直觉：GPT-2 词表 50257，Megatron 会把它 **pad 到能被 `make-vocab-size-divisible-by`（如 128）整除**，得到 50304。这是为了让词嵌入矩阵在 TP 维度上能整齐切分、对 Tensor Core 友好。

**训练时怎么用**：`--data-path` 指向的是上面的前缀（带 `_text_document` 后缀、**不带** `.bin/.idx` 扩展名）。数据集会按 `--split 700,200,100` 之类比例切成 train/valid/test。

## 4. 三种并行：TP、PP + DP，一图看懂

这是理解 Megatron 的核心。假设一个 Transformer 有很多层，每层有「自注意力 + MLP」：

```
                    ┌──────── 一个完整模型（很多层）────────┐
                    │  Layer0  Layer1  ...  LayerN          │
                    └──────────────────────────────────────┘

① 张量并行 TP（切「层内」的大矩阵，列/行切分）
   Layer 内的 QKV、MLP 权重矩阵被横/竖切到多卡：
        GPU0: W[:, 0:h/2]      GPU1: W[:, h/2:h]
   每层前向/反向都要 all-reduce 把分片结果拼回来 → 通信频繁、必须高带宽(NVLink)
   ⇒ 适合「单层都放不下」时，通常只在「机内」开 TP

② 流水并行 PP（切「层与层之间」，按 stage 分段）
   Stage0(Layer0-1) → Stage1(Layer2-3) → ... 各占不同 GPU
   前向把激活往后传，反向把梯度往前传（P2P 通信）；
   需要把 batch 拆成 micro-batch 做「流水」以减少气泡(bubble)

③ 数据并行 DP（复制整模型，喂不同数据）
   GPU组A、GPU组B 各持有一份完整(或 TP/PP 切过的)模型副本
   各自跑不同 micro-batch，step 前对梯度做 all-reduce 求平均
```

三者的**通信强度**与**典型放置**：

```
通信量/频率   TP  ████████  最高（每层都通信）→ 放机内 NVLink
              PP  ███       中（仅 stage 边界传激活/梯度）→ 可跨机
              DP  ██        低（每 step 一次梯度 all-reduce）→ 最易扩展

放置直觉：world_size = TP × PP × DP
        优先在一台机器内开 TP（吃 NVLink 带宽），
        机器之间用 PP / DP（容忍较低带宽）。
```

> 一句话记忆：**TP 切宽（矩阵）、PP 切深（层）、DP 复制（副本）**。

## 5. 三种运行模式与显存对比

本目录提供三个示例脚本（来自 Megatron-LM 的 `examples/`），逐级展示并行：

| 脚本 | world_size | TP | PP | DP | 解决什么 |
|---|---|---|---|---|---|
| `pretrain_gpt.sh` | 1 | 1 | 1 | 1 | 单卡，**仅调试用**（代码是为分布式优化的） |
| `pretrain_gpt_distributed.sh` | 4 | 1 | 1 | 4 | 纯数据并行，吞吐 ×4，模型仍需单卡装下 |
| `pretrain_gpt_distributed_with_mp.sh` | 4 | 2 | 2 | 1 | 张量+流水并行，单卡装不下时把模型摊开 |

来自本仓库真实运行日志的显存观测（A800 80G，345M 模型）：

```
模式            每卡显存(约)     说明
单卡            ~9.7 GB         整模型 + 优化器 + 激活全在一张卡
4DP             ~9.7 GB/卡      每卡仍是整模型一份副本，只是各喂不同数据
2TP+2DP*        ~6.8~8.7 GB/卡  模型被切开，单卡负担下降
* 不同 rank 因 PP stage 不同，显存略有差异（embedding stage 偏高）
```

**怎么选**：模型单卡装得下 → 优先纯 DP（最简单、扩展最好）；单卡装不下 → 先在机内开 TP（≤ 单机 GPU 数，如 8），还不够再叠 PP 跨机，最后用 DP 把总吞吐拉满。

启动方式（示意，**确切参数以脚本/官方为准**）：

```bash
# 单卡调试
CUDA_VISIBLE_DEVICES=3 sh examples/pretrain_gpt.sh

# 4 卡数据并行（torchrun 拉起 4 个进程）
sh examples/pretrain_gpt_distributed.sh

# 2TP + 2PP（这里 DP=1，4 卡）
sh examples/pretrain_gpt_distributed_with_mp.sh
```

## 6. 关键参数说明（讲含义，不背默认值）

启动日志里 `arguments` 段会打印上百个参数。下面挑**最该理解**的几类，讲清「调它影响什么 / 怎么权衡」。具体默认值随版本变化，**一律以官方 `arguments.py` 与 `--help` 为准**。

**① 并行维度**

| 参数 | 含义 / 权衡 |
|---|---|
| `--tensor-model-parallel-size` | TP 大小。↑ 能放下更大单层，但每层通信增多，建议 ≤ 单机 GPU 数 |
| `--pipeline-model-parallel-size` | PP 大小。↑ 能放下更多层（跨机），但有流水气泡，需配合更多 micro-batch |
| 数据并行(DP) | 不直接给，等于 `world_size /(TP×PP)`，自动推导 |

**② Batch 与序列**

| 参数 | 含义 / 权衡 |
|---|---|
| `--micro-batch-size` | 单卡单次前向的样本数，受显存约束 |
| `--global-batch-size` | 全局有效 batch；= micro × DP × 梯度累积步数。↑ 训练更稳但需更多算力 |
| `--seq-length` / `--max-position-embeddings` | 序列长度；后者须 ≥ 前者。↑ 激活显存随之上涨 |

> 数值例子：4DP、`micro=1`、`global=8` ⇒ 梯度累积步数 = 8 /(1×4) = 2，即每卡攒 2 个 micro-batch 的梯度再 all-reduce。

**③ 优化与学习率**

| 参数 | 含义 |
|---|---|
| `--lr` / `--min-lr` | 峰值与最小学习率 |
| `--lr-decay-style` | 衰减曲线（GPT 常用 `cosine` 余弦衰减） |
| `--lr-warmup-fraction` | 预热占比，开头线性升温避免发散 |
| `--clip-grad` | 梯度裁剪阈值，防梯度爆炸 |
| `--weight-decay` | 权重衰减（正则） |

**④ 混合精度与显存优化**

| 参数 | 含义 / 权衡 |
|---|---|
| `--fp16` / `--bf16` | 半精度训练，省显存提速；bf16 动态范围更大、更稳 |
| `loss scale`（动态） | FP16 下放大 loss 防梯度下溢；日志里会看到它随 NaN 自动调整 |
| `--use-distributed-optimizer` | 把优化器状态按 DP 分片（类似 ZeRO-1），显存进一步下降 |
| `--recompute-*`（激活重计算） | 反向时重算激活而非缓存，**用时间换显存**，长序列/深模型常开 |
| `--sequence-parallel` | 与 TP 配合，把 LayerNorm/Dropout 的激活也沿序列切，再省一截激活显存 |
| `--use-flash-attn` | 用 FlashAttention 融合核，省激活显存、提速（见 [[llm-optimizer/FlashAttention]]） |

**⑤ 数据与词表**

| 参数 | 含义 |
|---|---|
| `--data-path` | 预处理产物前缀（带 `_text_document`，不带扩展名） |
| `--vocab-file` / `--merge-file` | BPE 词表与合并规则 |
| `--split` | train/valid/test 切分比例，如 `700,200,100` |
| `--make-vocab-size-divisible-by` | 把词表 pad 到该数倍数，利于 TP 切分与 Tensor Core |

**怎么读训练日志**（来自本仓库真实输出）：

```
 iteration 1000/5000 | consumed samples: 8000 | elapsed time per iteration (ms): 283.9 |
 learning rate: 4.6E-05 | global batch size: 8 | lm loss: 5.54 | loss scale: 65536.0 |
 grad norm: 2.475 | number of skipped iterations: 0 | number of nan iterations: 0 |
```
- `lm loss` 应稳步下降；`loss scale` 在 FP16 下自动伸缩；
- `skipped/nan iterations` 偶发 1~2 次正常（loss scale 回退），持续出现要查数据/学习率；
- `lm loss PPL`（困惑度）= $e^{\text{loss}}$，是语言模型常用指标。

## 7. 权重(checkpoint)目录结构

Megatron 的 checkpoint 是**按并行切分存的**，目录名编码了 rank。理解它才能正确加载/合并。

**纯 TP=1（单卡或纯 DP）**：只有一个模型分片
```
345m/
├── latest_checkpointed_iteration.txt   ← 内容是 "5000" 或 "release"，指向最新 ckpt
├── release/  或  iter_0005000/
│   └── mp_rank_00/
│       └── model_optim_rng.pt          ← 权重+优化器+RNG 状态
```

**TP=2、PP=2（2TP+2PP）**：`mp_rank_TT_PPP` 编码 (TP 序号, PP 序号)
```
345m-init-mp/
└── iter_0005000/
    ├── mp_rank_00_000   ┐
    ├── mp_rank_00_001   │ 4 个分片 = TP(2) × PP(2)
    ├── mp_rank_01_000   │ 每片只含本 rank 负责的那部分参数
    └── mp_rank_01_001   ┘
```

要在不同并行配置间迁移，或导出给推理引擎，需用 Megatron 自带的 **checkpoint 转换/合并工具**把分片合回完整权重（见同目录 [merge_ck_and_inference/](./merge_ck_and_inference/) 与 [model_merge_eval_inference.md](./model_merge_eval_inference.md)）。

## 环境准备（NGC PyTorch 容器）

Megatron 依赖特定 CUDA/cuDNN/apex/融合核，**强烈建议直接用 NVIDIA NGC 的 PyTorch 容器**，免去自己编译 apex 的折腾：

```bash
docker run -dt --name nvidia_pytorch_temp --restart=always --gpus all \
  --network=host --shm-size 4G \
  -v /home/gdong/workspace:/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:23.04-py3 /bin/bash

docker exec -it nvidia_pytorch_temp bash
```

> `--shm-size` 要给足（DataLoader 多进程走共享内存）；`--gpus all` 暴露所有卡；具体镜像 tag 按你机器的驱动/CUDA 版本选择。

## 拉取代码

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 992da75   # 本教程基于该提交；不同版本参数会有出入
```

本仓库实测时对两处做了小修（多为新旧版本兼容/编码问题，**以你当前版本实际情况为准**）：
- `megatron/tokenizer/file_utils.py`
- `tools/openwebtext/merge_data.py`

示例脚本：
- `examples/pretrain_gpt.sh` —— 单卡
- `examples/pretrain_gpt_distributed.sh` —— 数据并行
- `examples/pretrain_gpt_distributed_with_mp.sh` —— 张量并行 + 流水并行

## 常见问题与坑

| 现象 / 坑 | 原因 | 处理 |
|---|---|---|
| `could not find ... latest_checkpointed_iteration.txt`，从随机初始化开始 | 指定的 `--load` 目录里没有该元数据文件 | 确认路径；首次训练「从头开始」属正常 |
| 启动卡在「compiling and loading fused kernels」 | 第一次要 ninja 编译融合 CUDA 核 | 正常，等编译完；之后会缓存复用 |
| 大量 `skipped iterations` / `nan iterations` | FP16 下梯度溢出，loss scale 反复回退 | 用 `--bf16`、降学习率、查异常数据/裁剪梯度 |
| OOM 显存爆 | micro-batch/序列太大，或并行度不足 | 降 micro-batch、开激活重计算、加 TP/PP、用分布式优化器 |
| `--data-path` 报找不到文件 | 写了 `.bin/.idx` 扩展名或漏了 `_text_document` 后缀 | 只给「前缀」，如 `.../my-gpt2_text_document` |
| 多机 NCCL 超时/挂起 | 网络/IB 未配好或 `MASTER_ADDR`/端口错 | 查 NCCL 环境变量与网卡，见 [[ai-infra/网络/NCCL]] |
| checkpoint 在不同 TP/PP 下加载失败 | 分片数与当前并行配置不匹配 | 先用合并/转换工具转成目标并行布局 |
| 验证 loss 比训练 loss 高很多、且越训越高 | 小数据集严重过拟合（本例 train 仅 ~1700 文档，跑了几十 epoch） | 这是教学小数据的预期现象；真实训练需大语料 |

> 这套示例的目的在于**跑通流程、理解并行**，345M + 小语料会过拟合，loss/PPL 的绝对值不代表真实模型质量。

## 延伸阅读（本目录）

- [gpt-data-preprocess.md](./gpt-data-preprocess.md) —— 数据下载、清洗、去重、分词全过程
- [model_train.md](./model_train.md) —— 单卡 / 4DP / 2TP+2DP 的完整启动日志与显存对照
- [model_merge_eval_inference.md](./model_merge_eval_inference.md) —— 权重合并、评估与推理
- [merge_ck_and_inference/](./merge_ck_and_inference/) —— checkpoint 合并与推理脚本

## 🔗 跳转链接

- [[00-知识地图]]
- [[ai-framework/megatron-lm/README]] —— Megatron-LM 框架总览
- [[ai-framework/deepspeed/README]] —— DeepSpeed（ZeRO 系列，常与 Megatron 组合成 Megatron-DeepSpeed）
- [[llm-train/README]] —— 大模型训练总览
- [[llm-train/pytorch/distribution/README]] —— PyTorch 分布式（DDP / 多机）基础
- [[ai-infra/网络/NCCL]] —— 多卡/多机集合通信库
- [[llm-optimizer/FlashAttention]] —— 注意力加速与显存优化
