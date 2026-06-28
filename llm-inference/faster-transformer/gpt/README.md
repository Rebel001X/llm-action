# FasterTransformer GPT 推理

> NVIDIA FasterTransformer（FT）中针对 GPT 类自回归解码器（Decoder-only）的高度优化 C++/CUDA 推理实现：把整条 Transformer 前向流程「算子融合 + 显存复用 + 多 GPU 并行」打包成手写 kernel，追求极致延迟与吞吐。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/faster-transformer/README]] [[llm-inference/大模型推理张量并行]] [[llm-inference/tensorrt/README]] [[llm-inference/README]]

## 阅读地图

| 小节 | 讲什么 | 你将能回答 |
| --- | --- | --- |
| 0 | 一句话锚点 | FT GPT 到底是「什么东西」 |
| 1 | 地基/前置 | 它解决朴素 PyTorch 推理的哪些痛点 |
| 2 | GPT 自回归推理两阶段 | 为什么要分 context（prefill）与 decode |
| 3 | 整体架构与调用链 | 从权重到 token 经过哪些模块（ASCII 图） |
| 4 | 核心机制：算子融合 / KV Cache / Paged 思想 | FT「快」在哪 |
| 5 | 多 GPU 并行：张量并行 + 流水线并行 | 大模型怎么切到多卡多机 |
| 6 | 数据类型：FP16 / BF16 / INT8 / FP8 | 精度与速度怎么权衡 |
| 7 | 权重转换（c-model）与目录约定 | HF/Megatron 权重怎么喂给 FT |
| 配置 | 关键参数/配置项含义 | 每个旋钮调什么、怎么权衡 |
| 坑 | 常见问题 | 踩坑前先看 |

---

## 0. 一句话锚点

**FasterTransformer GPT = 把「GPT 前向推理」从 Python 算子图重写成一套手工优化的 CUDA kernel 集合**，对外暴露 C++ 库（`libth_transformer.so` 等）和 PyTorch / Triton 后端接口，核心目标是：**在固定的 GPU 上，让单个 GPT 模型推理得更快（低延迟）、更省（高吞吐/省显存）、能更大（多卡并行）。**

> ⚠️ 重要现状：FasterTransformer 已被 NVIDIA 官方标记为**进入维护/不再积极开发**，其能力被 **TensorRT-LLM** 继承与替代。学习 FT 仍然非常值钱——因为它把「LLM 推理为什么快」的底层机制讲得最透，是理解 vLLM / TensorRT-LLM 的「前置课」。新项目选型请优先看 [[llm-inference/tensorrt/README]] 与 TensorRT-LLM。具体支持矩阵以官方仓库 README 为准。

---

## 1. 地基：它解决什么问题

朴素地用 HuggingFace + PyTorch 跑 GPT 推理，会遇到三类系统性低效。FT 就是逐个把它们干掉：

| 痛点 | 朴素实现的问题 | FT 的应对 |
| --- | --- | --- |
| **算子太碎** | 一层 Transformer 拆成几十个小算子（LayerNorm、QKV 投影、Softmax、bias add、激活…），每个都要一次 kernel launch + 一次读写显存（HBM） | **kernel fusion**：把相邻算子手写融合成大 kernel，减少 launch 次数和 HBM 往返 |
| **重复计算历史** | 自回归每生成 1 个 token，朴素实现会重算前面所有 token 的 Key/Value | **KV Cache**：把历史 K/V 存下来，每步只算新 token |
| **单卡放不下/不够快** | 175B 这种模型单卡显存装不下；单卡算力也不够 | **张量并行（TP）+ 流水线并行（PP）**，配合 NCCL/MPI 跨卡跨机通信 |

一句话：**朴素推理是「算力没喂饱、显存被反复刷、模型切不开」；FT 把这三件事都做到接近硬件极限。**

要理解 FT，先要理解 GPT 推理的「两阶段」本质（见第 2 节），这是后面所有优化的出发点。

---

## 2. GPT 自回归推理：context 与 decode 两阶段

GPT 是 Decoder-only 自回归模型：给定 prompt，**一次只吐一个 token**，再把它接回输入，循环往复。这个过程天然分成两段，计算特性完全不同：

```
 输入 prompt: "中国的首都是"   (假设 5 个 token)
 ┌──────────────────────────────────────────────────────────┐
 │ 阶段 A：Context / Prefill（上下文编码）                      │
 │   - 一次性把 5 个 token 全部喂进网络                          │
 │   - 计算量大（5 个 token × 全部层），矩阵乘是“大 GEMM”          │
 │   - 受【算力(compute)】约束 → GPU 算得满，利用率高             │
 │   - 产物：每层每个 token 的 Key/Value → 写入 KV Cache         │
 └──────────────────────────────────────────────────────────┘
                         │  吐出第 1 个新 token "北"
                         ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 阶段 B：Decode / Generation（逐 token 生成）                 │
 │   step1: 输入 "北"  → 读 KV Cache(5) → 吐 "京", 追加 KV       │
 │   step2: 输入 "京"  → 读 KV Cache(6) → 吐 "。", 追加 KV       │
 │   ...                                                       │
 │   - 每步只处理 1 个 token，矩阵乘退化成“瘦长 GEMV”            │
 │   - 受【显存带宽(memory bandwidth)】约束 → 搬权重比算还慢      │
 └──────────────────────────────────────────────────────────┘
```

**为什么必须分开讲？** 因为两阶段瓶颈不同，优化手段也不同：

- Context 阶段是 **compute-bound**：希望大 GEMM、用上 Tensor Core、用上 FlashAttention 式的融合注意力。
- Decode 阶段是 **memory-bound**：每生成 1 个 token 都要把整个模型权重从 HBM 读一遍，于是「读得少 = 快」，这就是**量化（INT8/FP8）和 KV Cache 复用**在 decode 阶段收益最大的原因。

数值直觉：一个 token 的 decode，FLOPs 约等于 $2 \times P$（$P$=参数量），而需要搬运的权重字节约等于 $P \times \text{bytes}$。当 batch 很小时，算力远没用满，时间几乎全花在「把权重从显存搬到计算单元」上——这就是 decode 受带宽约束的本质。

---

## 3. 整体架构与调用链

FT GPT 的代码组织可以抽象为「四层」，从用户接口一路下沉到 CUDA kernel：

```
┌──────────────────────────────────────────────────────────────┐
│ ① 接口层 (Frontend)                                            │
│    examples/pytorch/gpt/*.py   ← Python 脚本/示例               │
│    Triton backend (fastertransformer_backend)  ← 部署服务化     │
│    PyTorch op：torch.classes.FasterTransformer.GptOp           │
│         │ 加载 .so：libth_transformer.so                       │
└─────────┼──────────────────────────────────────────────────────┘
          ▼
┌──────────────────────────────────────────────────────────────┐
│ ② 模型层 (C++ ParallelGpt)                                     │
│    ParallelGpt<T>  ── 管理：权重、并行通信、采样、循环调度        │
│      ├─ GptContextDecoder   （阶段 A：prefill，整段处理）        │
│      └─ GptDecoder          （阶段 B：decode，逐 step 处理）     │
│    DynamicDecodeLayer：温度/top-k/top-p/重复惩罚/beam search    │
└─────────┬──────────────────────────────────────────────────────┘
          ▼
┌──────────────────────────────────────────────────────────────┐
│ ③ 层/模块层 (per-layer building blocks)                        │
│    每个 Transformer Block 内部：                                │
│    LayerNorm → QKV GEMM → Self-Attention(+KV Cache) →          │
│    Out-Proj GEMM → 残差 → LayerNorm → FFN(GELU) → 残差          │
│    (TP 时这些 GEMM 被切分到多卡，靠 AllReduce 合并)              │
└─────────┬──────────────────────────────────────────────────────┘
          ▼
┌──────────────────────────────────────────────────────────────┐
│ ④ Kernel 层 (手写 CUDA / cuBLAS / cutlass)                     │
│    fused LayerNorm+bias、masked multi-head attention kernel、   │
│    add-bias-activation 融合、gemm via cuBLAS/cutlass、          │
│    采样 kernel、INT8/FP8 量化 kernel                            │
└──────────────────────────────────────────────────────────────┘
```

**调用链（一次 generate 请求）**：

```
Python 脚本
  → 构造输入 input_ids / 采样参数(top_k,top_p,temperature,beam_width...)
  → 调用 GptOp.forward(...)            [PyTorch custom op]
    → ParallelGpt<T>::forward()
       ├─ 一次 GptContextDecoder  (prefill 全部 prompt token → 填 KV Cache)
       ├─ 循环 max_new_tokens 次:
       │    GptDecoder (1 token) → logits
       │    DynamicDecodeLayer (采样/beam) → 选出下一个 token
       │    判断 EOS / 长度 → 决定是否停止
       └─ 收集输出 output_ids
  → 返回给 Python，detokenize 成文本
```

> 注意：以上类名（`ParallelGpt`、`GptContextDecoder`、`GptDecoder`、`DynamicDecodeLayer`）是 FT GPT 模块的**典型组织方式**，用于建立心智模型；**确切的类名/文件名/函数签名请以你 checkout 的 FasterTransformer 源码版本为准**（不同版本会重构）。

---

## 4. 核心机制：FT「快」在哪

### 4.1 算子融合（Kernel Fusion）

朴素图里「LayerNorm → +bias → GELU」是三个 kernel，三次读写 HBM。FT 把它们融合成**一个** kernel：数据进来一次，在寄存器/共享内存里把三件事都干完，再写出去一次。

```
朴素：  HBM→[LN]→HBM→[+bias]→HBM→[GELU]→HBM   (4 次 HBM 读写)
融合：  HBM→[ LN | +bias | GELU 一次搞定 ]→HBM   (2 次 HBM 读写)
```

收益不是省 FLOPs，而是**省 HBM 带宽 + 省 kernel launch 开销**。对 memory-bound 的 decode 阶段尤其关键。

### 4.2 Masked Multi-Head Attention + KV Cache

自注意力是 GPT 的核心。FT 为 decode 阶段写了专门的 **masked MHA kernel**，把「QK^T → scale → mask → softmax → ×V」融合进单 kernel，并直接读写 KV Cache：

```
decode step：只处理当前 1 个 token 的 Q
  Q(1×d)  ┐
          ├─ 与 KV Cache 里全部历史 K 做点积 → 注意力分数(1×L)
  K cache ┘     softmax
  V cache ── 加权求和 → 注意力输出(1×d)
  本步新算出的 (k,v) 追加进 KV Cache（L→L+1）
```

KV Cache 把「每步重算历史」从 $O(L^2)$ 降到「每步只算 1 个新 token」。代价是**显存换计算**：

$$\text{KV Cache 显存} = 2 \times L \times H \times d_{head} \times n_{layer} \times \text{batch} \times \text{bytes}$$

其中 $2$ 是 K 和 V，$L$ 是序列长度，$H$ 是注意力头数（GQA/MQA 下用 KV 头数），$d_{head}$ 是每头维度。**序列越长、batch 越大，KV Cache 越吃显存**，这往往是长上下文推理的显存瓶颈，所以会有 INT8 KV Cache 等进一步压缩手段。

### 4.3 显存预分配与复用

FT 在初始化时**一次性预分配**好工作缓冲区（中间激活、KV Cache 等），运行期不反复 malloc/free，避免显存碎片和分配开销。这是 C++ 推理引擎相对 Python 动态图的天然优势之一。

> 说明：FT 的 KV Cache 是「连续/预分配」式管理，与 vLLM 的 **PagedAttention（分页 KV Cache）** 思路不同。PagedAttention 在变长、多请求并发场景显存利用率更高；FT 更偏「单/少请求极致延迟」。选型时这是重要差异点。

---

## 5. 多 GPU 并行：把大模型切开

单卡装不下 175B 这种模型，FT 用两种正交的并行方式（可叠加）。要点：**TP 切「一层内部」，PP 切「层与层之间」**。

```
模型有 L 层，部署到 (PP=2) × (TP=4) = 8 张 GPU：

           TP 组 0 (4 卡，切一层内部)      TP 组 1
 PP stage0 ┌GPU0─GPU1─GPU2─GPU3┐  持有第 0..L/2-1 层
           └──── AllReduce ────┘
                  │ 激活经 P2P/网络传给下一 stage
 PP stage1 ┌GPU4─GPU5─GPU6─GPU7┐  持有第 L/2..L-1 层
           └──── AllReduce ────┘
```

### 5.1 张量并行（Tensor Parallelism, TP）

把**单个权重矩阵按列/行切片**分到多卡。例如 QKV 投影矩阵按列切成 4 份，每卡算自己那 1/4，再用一次 **AllReduce** 把结果合并。

- 优点：每卡显存 ≈ 1/TP，延迟低（一层内并行）。
- 代价：**每层都要做 AllReduce**，通信频繁 → **强依赖卡间高速互联（NVLink/NVSwitch）**，跨机走 IB 时通信会成为瓶颈。
- 因此 TP 通常**只在单机内**用（TP ≤ 单机 GPU 数）。详见 [[llm-inference/大模型推理张量并行]]。

### 5.2 流水线并行（Pipeline Parallelism, PP）

把**不同的层分到不同卡**（stage）。激活在 stage 间用点对点（P2P）传递。

- 优点：通信量小（只传层间激活），适合**跨机**扩展。
- 代价：有「流水线气泡」，且推理（尤其低 batch）下收益不如训练明显。

### 5.3 通信底座

FT 多 GPU 依赖 **NCCL**（卡间集合通信，AllReduce 等）+ **MPI**（多进程/多机启动与协调）。这也是为什么编译要带 `-DBUILD_MULTI_GPU=ON`，运行多卡示例要用 `mpirun`。集合通信原语（AllReduce/AllGather/Broadcast）的含义可参考仓库网络相关笔记。

---

## 6. 数据类型：精度与速度的权衡

| 类型 | 位宽 | 谁支持 | 特点 / 用在哪 |
| --- | --- | --- | --- |
| FP32 | 32 | 通用 | 基准精度，慢且费显存，推理基本不用 |
| **FP16** | 16 | Volta 及以后 | 推理默认，配合 Tensor Core，省一半显存与带宽 |
| **BF16** | 16 | Ampere(A100) 及以后 | 指数位与 FP32 同宽，数值范围大、更稳，越来越主流 |
| **INT8** | 8 | 需校准/量化 | 权重(+激活)量化，decode 带宽收益大；FT 含 INT8 GEMM 与 INT8 KV Cache 路径 |
| **FP8** | 8 | Hopper(H100) 及以后 | 8 位浮点，兼顾范围与精度；FT 编译需 `-DENABLE_FP8=ON`，对应 `gpt_fp8` 示例 |

**怎么选（直觉）**：

- 默认 **FP16/BF16**：兼容性好、几乎无损。新硬件优先 BF16。
- 追求极致 decode 吞吐/显存：上 **INT8 / FP8**，但要做量化校准、验证精度，且**只有对应架构 GPU（如 H100 才有 FP8）**才有硬件加速。
- 量化是「用精度换带宽/显存」，对 memory-bound 的 decode 收益最大。量化基础见 [[llm-compression/quantization]]。

> 具体哪些算子支持哪种精度、FP8 的 e4m3/e5m2 细节，以官方文档与源码为准。

---

## 7. 权重转换（c-model）与目录约定

FT **不直接吃** HuggingFace / Megatron 的原始 checkpoint，而是先转成它自己的二进制布局（俗称 **c-model**），原因有二：

1. **拆分到多卡**：转换脚本会按你指定的 `-i_g`（TP 数）把每个权重矩阵切好，落成 `*.bin` 文件，运行时按 rank 直接 mmap 加载，省去运行期切分。
2. **数据类型与内存布局对齐** FT kernel 期望的格式（连续、对齐、转置约定一致）。

```
原始权重 (HF / Megatron checkpoint)
        │  examples/.../*_ckpt_convert.py  (转换脚本)
        ▼
c-model 目录（按 GPU 数组织）
  c-model/345m/
    ├─ 1-gpu/    ← TP=1 的版本：一堆 model.layers.*.bin + config.ini
    ├─ 2-gpu/    ← TP=2：同一份权重被切成 2 份
    └─ 4-gpu/    ← TP=4
```

运行示例时，`--checkpoint-path` 指向某个 `N-gpu/` 目录，`N` 必须与你启动的 TP（进程数）匹配。父目录 [[llm-inference/faster-transformer/README]] 里给出了编译与 lambada 评测的完整命令；本目录聚焦 GPT 推理本身的概念。

---

## 关键参数 / 配置项说明（讲含义，不背默认值）

下面是 FT GPT 推理时**最常调的旋钮**，重点理解「调它影响什么」。**确切的参数名、默认值以你所用版本的脚本 `--help` 与 `config.ini` 为准。**

### 模型结构类（建模时固定，来自 config）

| 配置项 | 含义 | 怎么定 |
| --- | --- | --- |
| `head_num` / `size_per_head` | 注意力头数 / 每头维度 | 与权重一致，乘积≈hidden_size |
| `inter_size` | FFN 中间层维度 | 通常 4×hidden_size |
| `num_layer` | Transformer 层数 | 与权重一致 |
| `vocab_size` | 词表大小 | 与 tokenizer 一致 |
| `max_seq_len` | 支持的最大序列长度 | 影响 KV Cache 预分配大小 |

### 并行类

| 配置项 | 含义 | 权衡 |
| --- | --- | --- |
| `tensor_para_size` (TP) | 张量并行度 | 越大单卡越省显存，但 AllReduce 通信越多；建议 ≤ 单机卡数 |
| `pipeline_para_size` (PP) | 流水线并行度 | 跨机扩展用；有气泡，低 batch 收益有限 |
| (启动) `mpirun -n` 进程数 | = TP × PP | 必须与 c-model 切分一致 |

### 运行/采样类（每次请求可变）

| 配置项 | 含义 | 调它干嘛 |
| --- | --- | --- |
| `batch_size` | 同时处理的请求数 | 越大吞吐越高、延迟略升，吃显存（KV Cache×batch） |
| `request_output_len` / `max_new_tokens` | 最多生成多少 token | 控制生成长度与 KV Cache 上限 |
| `beam_width` | beam search 宽度 | >1 走 beam，质量可能更好但更慢更费显存；=1 为采样/贪心 |
| `temperature` | 采样温度 | 越高越随机，越低越确定 |
| `top_k` | 只在概率最高的 k 个里采样 | 控制多样性 |
| `top_p` (nucleus) | 累计概率到 p 的最小集合里采样 | 与 top_k 二选一或叠加 |
| `repetition_penalty` / `presence_penalty` | 重复惩罚 | 缓解复读机 |
| `len_penalty` | 长度惩罚（beam 时） | 调长短偏好 |
| `data_type` | FP16/BF16/INT8/FP8 | 见第 6 节，精度↔速度权衡 |

---

## 常见问题 / 坑

| 现象 / 问题 | 根因 | 对策 |
| --- | --- | --- |
| FT 不再更新、找不到新特性 | NVIDIA 已转向 **TensorRT-LLM** | 新项目用 TensorRT-LLM；FT 当学习/历史参考 |
| 多卡报维度/加载错误 | c-model 的 `N-gpu` 与启动 TP 数不一致 | 转换时的 `-i_g` 必须等于运行的 `tensor_para_size` |
| `cmake` 找不到 NCCL/MPI | 没开 `-DBUILD_MULTI_GPU=ON` 或环境缺库 | 用 NGC PyTorch 镜像（自带 NCCL/MPI），开多 GPU 编译开关 |
| 编译 FP8 失败 | 缺 `-DENABLE_FP8=ON` 或非 Hopper 架构 | FP8 需 H100 级 GPU + 对应分支/commit；`-DSM=90` |
| 长上下文 OOM | KV Cache 随 `L×batch` 线性增长 | 减小 batch/max_seq_len，或用 INT8 KV Cache |
| 编译指定架构 | `-DSM` 决定生成的 GPU 架构 | A100=80、H100/H800=90，按卡设对 |
| 精度变差（量化后） | INT8/FP8 未做好校准 | 做量化校准并对比基准；对精度敏感层保留高精度 |
| decode 慢但 GPU 利用率低 | decode 本就是 memory-bound | 增大 batch 提吞吐 / 上量化减带宽，单纯堆算力无用 |
| 权重对不上（HF 直接喂） | FT 要 c-model 二进制布局 | 必须先跑转换脚本，不能直接喂 HF checkpoint |
| 想做服务化/多请求调度 | FT 例子偏「跑批/单请求」 | 用 **Triton + fastertransformer_backend** 部署，或转 TensorRT-LLM |

---

## 🔗 跳转链接

- [[00-知识地图]] —— 全仓导航总入口
- [[llm-inference/faster-transformer/README]] —— FT 编译/镜像/运行（FP16 与 FP8 全流程）
- [[llm-inference/大模型推理张量并行]] —— TP 切分原理（本文第 5 节的深入版）
- [[llm-inference/tensorrt/README]] —— FT 的「继任者」生态，选型必看
- [[llm-inference/README]] —— 各推理引擎（vLLM/SGLang/TensorRT-LLM…）总览与对比
- [[llm-compression/quantization]] —— INT8/FP8 量化基础（第 6 节的延伸）

> 本文聚焦「稳定知识与机制」。涉及确切版本号、CLI 参数名/默认值、源码类名与函数签名、性能数字之处，请以你 checkout 的 FasterTransformer 官方仓库源码与文档为准。
