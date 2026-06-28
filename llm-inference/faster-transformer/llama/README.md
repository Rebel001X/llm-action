# FasterTransformer 部署 LLaMA

> 用 NVIDIA FasterTransformer（FT）这套"手写 CUDA 算子 + C++ 推理引擎"把 LLaMA 跑起来：先把 HuggingFace 权重转成 FT 的二进制格式，再用 FT 的 C++/PyTorch op 做张量并行 + 流水并行的高吞吐推理。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/faster-transformer/README]] [[llm-inference/tensorrt-llm/README]] [[llm-inference/triton/README]] [[ai-infra/网络/NCCL]]

## 阅读地图

| 节 | 讲什么 | 你能带走什么 |
|----|--------|--------------|
| 0 | 一句话锚点 | FT 在 LLaMA 推理里到底是个什么角色 |
| 1 | 地基：FT 解决什么问题 | 为什么不直接用 HF transformers 推理 |
| 2 | 整体架构 | 从 HF 权重到 token 输出的全链路 |
| 3 | LLaMA 适配点 | RoPE / RMSNorm / SwiGLU / GQA 在 FT 里怎么落地 |
| 4 | 权重转换 | `.bin` 格式、按 TP 切分的本质 |
| 5 | 并行机制 | TP 与 PP 怎么切、NCCL 怎么通信 |
| 6 | KV Cache 与解码 | 自回归推理为什么要缓存、显存怎么算 |
| 7 | 编译与运行流程 | Docker→cmake→转权重→跑 example |
| 8 | 关键参数 | 各配置项做什么用、怎么权衡 |
| 9 | 与同类对比 | FT vs TensorRT-LLM / vLLM |
| — | 常见坑 | 编译/转权重/并行/精度的高频问题 |

## 0. 一句话锚点

**FasterTransformer = NVIDIA 开源的、针对 Transformer 推理深度优化的 C++/CUDA 库。** 它不像 PyTorch 那样"一层一个 Python 算子"，而是把整个 Transformer 层的计算（LayerNorm、QKV GEMM、Attention、FFN）尽量**融合（fuse）成少数几个手写 CUDA kernel**，再配上**张量并行（TP）+ 流水线并行（PP）**，从而在 GPU 上跑出比原生框架更高的吞吐和更低的延迟。

LLaMA 不是 FT 官方主线最早支持的模型，社区（尤其 `void-main` 的 fork）通过新增 RoPE、RMSNorm、SwiGLU 等算子把 LLaMA 适配了进来——所以本文件顶部那几个链接指向的正是 FT issue #506、PR #575 和 void-main 的 fork。

> ⚠️ 时代背景：FT 现已进入**维护/归档状态**，NVIDIA 的主推方案是 **TensorRT-LLM**。本文讲 FT 是为了理解"手写算子级推理引擎"的底层机制，生产新项目应优先评估 TensorRT-LLM / vLLM / SGLang。具体版本与维护状态以官方仓库为准。

## 1. 地基：FT 解决什么问题

直接用 HuggingFace `transformers` 做推理，瓶颈在哪？

```
HF transformers 推理一层的样子（简化）：
  x ──RMSNorm──> ──Linear(Q)──> ──Linear(K)──> ──Linear(V)──>
       │            │              │              │
   每个箭头 = 一次独立的 CUDA kernel launch + 一次 HBM 读写
       │
   ──Attention──> ──Linear(O)──> ──RMSNorm──> ──FFN(gate/up/down)──>
```

问题有三类：

1. **Kernel launch 过多**：每个小算子都要 CPU→GPU 启动一次 kernel，launch 开销 + 中间结果反复进出显存（HBM），对**访存密集（memory-bound）**的解码阶段是致命的。
2. **缺少融合**：LayerNorm+bias+残差、激活+逐元素乘等本可以合并的操作没有合并。
3. **缺少推理专用并行**：HF 主要服务训练，对**推理时的张量并行**（把一个大矩阵切到多卡）支持不原生、不极致。

FT 的应对：
- **算子融合**：把一层里能合的都合进尽量少的 kernel（如融合 multi-head attention、融合 add-bias-residual-LayerNorm）。
- **手写/调优 GEMM**：用 cuBLAS / CUTLASS，并针对低精度（FP16/BF16/INT8/FP8）做特化。
- **内建 TP + PP**：用 NCCL + MPI 做跨卡/跨机通信，天然支持把超大模型切到多卡。

一句话：**FT 把"一堆 Python 小算子"压成"少数大 CUDA kernel + 多卡并行"，换取吞吐和延迟。**

## 2. 整体架构（全链路）

```
┌───────────────────────────────────────────────────────────────┐
│  阶段一：离线（一次性）                                         │
│                                                                 │
│   HF LLaMA 权重 (.safetensors / .bin, FP16)                    │
│        │  huggingface_llama_convert.py 之类转换脚本             │
│        ▼                                                         │
│   FT 权重目录  model/1-gpu 或 4-gpu/                            │
│     ├─ config.ini        (层数/头数/hidden/词表…)              │
│     ├─ model.layers.0.attention.query_key_value.weight.0.bin   │
│     ├─ ...               (按 TP rank 切分后的二进制张量)        │
│     └─ model.wte.bin / model.final_layernorm.weight.bin        │
└───────────────────────────────────────────────────────────────┘
                              │ 加载
                              ▼
┌───────────────────────────────────────────────────────────────┐
│  阶段二：在线推理（C++ Llama 类 / PyTorch op）                  │
│                                                                 │
│   输入 token ids                                                │
│        ▼                                                         │
│   ┌── Context/Prefill 阶段 ──┐  一次性处理整段 prompt           │
│   │  Embedding → N×LlamaLayer │  → 把每层 K,V 写入 KV Cache    │
│   └────────────────────────────┘                               │
│        ▼                                                         │
│   ┌── Decode/Generation 阶段 ─┐  逐 token 自回归               │
│   │  每步只算 1 个新 token     │  ← 读历史 KV Cache             │
│   │  → 采样下一个 token        │  → 新 K,V 追加进 Cache         │
│   └────────────────────────────┘                               │
│        ▼                                                         │
│   到达 EOS / max_len → 输出 token ids                          │
└───────────────────────────────────────────────────────────────┘
```

要点：
- **离线转换**只做一次，把"框架无关的权重"变成"FT 私有的、已按并行度切好的 `.bin`"。
- **在线推理**分两阶段：**Prefill**（计算密集，并行度高）和 **Decode**（访存密集，逐 token），KV Cache 是连接两者的桥梁（见第 6 节）。

## 3. LLaMA 相对标准 GPT 的适配点

FT 最早是为 BERT/GPT 写的。LLaMA 在结构上和 GPT 有几处关键差异，这正是 issue #506 / PR #575 / void-main fork 要补的算子：

| 组件 | 标准 GPT | LLaMA | FT 里要做什么 |
|------|----------|-------|----------------|
| 位置编码 | 学习式绝对位置 / 部分 RoPE | **RoPE 旋转位置编码** | 在 attention kernel 里对 Q、K 施加旋转 |
| 归一化 | LayerNorm（带均值+方差+bias） | **RMSNorm**（只用均方根，无均值无 bias） | 新增 RMSNorm kernel |
| FFN 激活 | GELU，单条 up→down | **SwiGLU**：gate 与 up 两条线 + SiLU 门控相乘 | 改 FFN 结构为 `down(SiLU(gate(x)) * up(x))` |
| 注意力（新版） | MHA | **GQA/MQA**（K/V 头数 < Q 头数，LLaMA-2 70B 等） | KV 投影维度变小，Cache 也变小 |
| 偏置 bias | 多处有 bias | LLaMA 各线性层**普遍无 bias** | 转换/算子里把 bias 置零或跳过 |

ASCII 对照 FFN（这是最直观的差异之一）：

```
GPT 风格 FFN：           LLaMA SwiGLU FFN：
   x                        x
   │                     ┌──┴──┐
 up(x)                 gate(x) up(x)
   │                     │      │
 GELU                  SiLU     │
   │                     └──×───┘   ← 逐元素相乘（门控）
 down(·)                    │
   │                      down(·)
   out                       out
```

$\text{SwiGLU}(x) = \big(\text{SiLU}(W_{gate}\,x)\big)\odot(W_{up}\,x),\quad \text{SiLU}(z)=z\cdot\sigma(z)$

RMSNorm（无均值、无偏置，比 LayerNorm 更省）：

$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}\odot g$

> 实践含义：如果你用的 FT 分支没合入这些算子，LLaMA 是跑不起来或结果乱码的。务必用支持 LLaMA 的分支（void-main fork 或已合入 PR #575 的版本）。具体支持矩阵以对应仓库 README 为准。

## 4. 权重转换：从 HF 到 FT `.bin`

FT 不直接吃 HuggingFace 格式，它要一组**裸二进制张量** + 一个 `config.ini`。转换脚本干两件事：

```
HF 权重张量 W  (形状 [out, in], FP16)
     │
     ├─(1) dtype 处理：转成目标精度（fp16/bf16）并保存为连续内存的 .bin
     │
     └─(2) 按张量并行度 TP 切分
              例：TP=4，把某个 [out, in] 的权重按列/行切成 4 份
              ┌──────────────┐        ┌────┬────┬────┬────┐
              │      W       │  ───►  │ W0 │ W1 │ W2 │ W3 │
              └──────────────┘        └────┴────┴────┴────┘
              生成 ...weight.0.bin / .1.bin / .2.bin / .3.bin
              GPU rank i 只加载 .i.bin
```

为什么要"离线切"而不是"运行时切"？
- 运行时加载的是**已经切好的小分片**，每张卡只读自己那份 → 加载快、显存峰值低。
- 切分方向（按行还是按列）要和 TP 的数学严格对应（见第 5 节），否则结果错误。

`config.ini` 里通常记录这些**结构超参**（名字以实际脚本为准）：层数、注意力头数、KV 头数（GQA 时 < 头数）、隐藏维度、FFN 中间维度、词表大小、`max_position`、RoPE 的 base/theta、RMSNorm 的 `eps`、权重数据类型、张量并行度等。引擎启动时读它来建图。

> 不要硬记脚本文件名和参数名——不同分支的转换脚本（`huggingface_llama_convert.py` / `llama_ckpt_convert.py` 等）参数不一样。**以你所用分支的 `examples/.../llama/` 下脚本和文档为准。**

## 5. 并行机制：张量并行（TP）与流水并行（PP）

### 5.1 张量并行 TP——把"一层"切到多卡

核心思想：把大矩阵乘法**切开分到多卡同时算**，再用一次通信合并。Megatron 风格的两种切法：

```
注意力 / FFN 第一层（按列切，Column Parallel）：
   Y = X · W ，把 W 按列切 [W0 | W1 | W2 | W3]
   每卡算 Yi = X · Wi  →  各卡得到 Y 的一部分列，无需立即通信

第二层（按行切，Row Parallel）：
   Z = Y · V ，把 V 按行切，每卡算部分和
   →  最后做一次 All-Reduce 把各卡部分和相加 = 完整 Z
```

一层 Transformer 在 TP 下的通信模式：

```
   各卡输入 X (复制)
        │ 列并行 GEMM（无通信）
        ▼
   注意力/FFN 各算各的分片
        │ 行并行 GEMM
        ▼
   ┌──── NCCL All-Reduce ────┐   ← 每层 attention 出口 + FFN 出口各一次
   └──────────┬──────────────┘
        合并成完整激活，进入下一层
```

- 通信量与 hidden size、序列长度成正比，**对带宽敏感** → 这就是为什么 TP 最好放在**单机内 NVLink 互联**的卡之间，跨机走 IB 会更慢。
- LLaMA-2 70B 这种量级，单卡装不下，TP=4 或 TP=8 是常见配置。

### 5.2 流水并行 PP——把"很多层"切到多卡/多机

```
   Stage0 (GPU0): Layer 0-19   ──激活──►
   Stage1 (GPU1): Layer 20-39  ──激活──►
   Stage2 (GPU2): Layer 40-59  ──激活──►
   Stage3 (GPU3): Layer 60-79  ──► 输出
```

- PP 按**层**切，stage 之间只传一次激活（点对点），通信量比 TP 小，**适合跨机**。
- 推理里 PP 主要用于"模型层数太多、显存不够"时进一步分摊，常与 TP 组合成 `world_size = TP × PP`。

总进程数：`mpirun -n (TP×PP)`，每个 rank 绑一张 GPU。FT 用 **MPI** 起多进程、用 **NCCL** 做集合通信。

## 6. KV Cache 与两阶段解码

自回归生成时，第 $t$ 步的注意力需要前面所有 token 的 Key/Value。如果每步重算，复杂度爆炸；**缓存它们**就只需为新 token 算一次。

```
Prefill（处理 prompt "我 爱 北 京"）：
   一次算 4 个 token 的 K,V  ──写入──►  KV Cache
   [K1 K2 K3 K4 | V1 V2 V3 V4]

Decode 第 5 步（生成"天"）：
   只算 token5 的 q5,k5,v5
   attention(q5, [K1..K5], [V1..V5])  ← 历史 K/V 直接读 Cache
   k5,v5 追加进 Cache
```

KV Cache 显存（单层、单卡，估算）：

$\text{KV bytes} \approx 2 \times \text{layers} \times \text{seq\_len} \times \text{n\_kv\_heads} \times \text{head\_dim} \times \text{batch} \times \text{dtype\_bytes}$

数值例子（LLaMA-2 13B 量级、FP16）：layers=40，n_kv_heads=40，head_dim=128，seq=2048，batch=1：

$2 \times 40 \times 2048 \times 40 \times 128 \times 1 \times 2 \approx 3.4\,\text{GB}$

所以：**长上下文 + 大 batch 时，KV Cache 往往比权重还吃显存。** GQA（减小 `n_kv_heads`）是 LLaMA-2 大模型省 Cache 的关键设计。FT 里 KV Cache 会按 TP 在头维度上一起切分，每卡只存自己负责的那些头。

## 7. 编译与运行流程（参考骨架）

整体流程（**命令以你所用分支文档为准，下面是骨架，不要照抄确切镜像 tag/参数**）：

```
[1] 起 NVIDIA PyTorch 镜像容器（含 CUDA/cuDNN/NCCL/MPI）
[2] git clone 支持 LLaMA 的 FT 分支 + submodule update
[3] cmake 配置：指定 SM 架构 / 开 BUILD_PYT / 开 BUILD_MULTI_GPU
[4] make -j 编译出 libth_transformer.so 等
[5] 跑权重转换脚本：HF LLaMA → FT .bin（指定 TP）
[6] 跑 example：单测 / 多卡 mpirun 推理
```

调用链（编译产物如何被用到）：

```
你的脚本 / example.cc
        │
        ▼
 PyTorch op 包装  (libth_transformer.so)
        │  或直接 C++ 调用
        ▼
 FT Llama 类（封装一层层 LlamaLayer）
        │
        ▼
 手写 CUDA kernels（RMSNorm / RoPE attention / SwiGLU FFN / GEMM）
        │
        ▼
 cuBLAS / CUTLASS GEMM  +  NCCL 集合通信
        │
        ▼
   GPU（多卡时跨卡 All-Reduce / P2P）
```

cmake 关键开关（讲含义，不背默认值）：

```bash
cmake -DSM=90 \                # 目标 GPU 架构，H100=90 / A100=80 / 必须匹配你的卡
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYT=ON \         # 编出 PyTorch op，方便用 Python 驱动
      -DBUILD_MULTI_GPU=ON ..  # 开多卡：编入 MPI+NCCL，TP/PP 才可用
make -j
```

> `-DSM` 写错（架构不匹配）会导致 kernel 无法运行或性能暴跌；多机/多卡一定要 `-DBUILD_MULTI_GPU=ON`，否则没有 NCCL 路径。具体 flag 与默认值以分支 `CMakeLists.txt` 为准。

运行（多卡示意）：

```bash
mpirun -n 4 \
  python examples/.../llama/llama_example.py \
    --ckpt_path  /workspace/model/llama-13b/4-gpu \
    --lib_path   /workspace/FasterTransformer/build/lib/libth_transformer.so \
    --tensor_para_size 4 \
    --pipeline_para_size 1
```

## 8. 关键参数说明（做什么用 / 怎么权衡）

| 参数（类别） | 作用 | 权衡 |
|--------------|------|------|
| `tensor_para_size` (TP) | 一层切到几张卡 | 越大单卡显存越省、通信越多；建议放 NVLink 内 |
| `pipeline_para_size` (PP) | 模型按层切到几个 stage | 跨机省带宽；但有 stage 间气泡/流水延迟 |
| `data_type` (fp16/bf16/int8/fp8) | 计算与权重精度 | 越低越省显存越快，但有精度损失风险；BF16 数值更稳 |
| `max_seq_len` / `max_input_len` | 上下文/输入上限 | 决定 KV Cache 预留显存，开太大白占显存 |
| `batch_size` | 同时处理的请求数 | 大 batch 提吞吐，但 KV Cache 线性增长 |
| `beam_width` | beam search 宽度 | >1 提质量，但显存/算力随之翻倍；贪心/采样设 1 |
| 采样：`top_k`/`top_p`/`temperature` | 控制随机性 | 偏确定→小 temp/小 top_p；偏多样→反之 |
| `repetition_penalty` | 抑制重复 | 太大可能伤流畅度 |
| RoPE `base`/`theta`、`rmsnorm eps` | 结构超参 | 必须与原模型一致，否则结果错乱（一般来自转换脚本） |

**精度路线直觉**：FP16/BF16 是基线；想再省显存/提速可上 INT8 weight-only 或 FP8（需 Hopper 架构支持），但要验证 LLaMA 在你任务上的精度可接受。量化更系统的内容见 [[llm-compression/quantization/量化基础]]。

## 9. 与同类方案对比

| 维度 | FasterTransformer | TensorRT-LLM | vLLM |
|------|-------------------|--------------|------|
| 形态 | C++/CUDA 手写算子库 | TensorRT 图编译 + 插件 | Python + CUDA，服务化友好 |
| LLaMA 支持 | 需社区分支补算子 | 官方一等支持 | 官方一等支持 |
| 并行 | TP+PP（NCCL/MPI） | TP+PP，更自动 | TP（+实验性 PP） |
| KV Cache | 连续预分配 | 优化的 paged 思路 | **PagedAttention**（显存利用率高） |
| 连续批处理 | 弱/需自己接 Triton | 支持 in-flight batching | **原生 continuous batching** |
| 现状 | 维护/归档，NVIDIA 转向 TRT-LLM | NVIDIA 主推 | 社区最活跃之一 |
| 适合 | 学习底层 / 已有 FT 栈 | 生产、NV 全家桶 | 在线服务、易用、吞吐高 |

一句话选型：**新项目优先 TensorRT-LLM / vLLM / SGLang；FT 更多是理解"手写算子级引擎"的教学价值，以及历史项目的维护。** 在线服务对比详见 [[llm-inference/vllm/README]]、[[llm-inference/sglang/README]]。

## 常见问题 / 坑

| 现象 | 根因 | 处理思路 |
|------|------|----------|
| LLaMA 跑出乱码/全 0 | 用了不支持 LLaMA 的 FT 分支（缺 RoPE/RMSNorm/SwiGLU） | 换支持 LLaMA 的分支（void-main fork / 合入 PR #575） |
| 编译报架构/ kernel 错 | `-DSM` 与实际 GPU 不符 | 按卡填：H100=90、A100=80、L40=89… |
| 多卡跑不起来 / 找不到 NCCL | 没开 `-DBUILD_MULTI_GPU=ON` 或 MPI 环境缺失 | 重新 cmake 开多卡，确认容器内有 MPI+NCCL |
| 转权重后维度对不上 | TP 切分方向错 / config 头数（含 KV 头数）填错 | 核对 GQA 的 `n_kv_heads`、列/行切法 |
| OOM（显存爆） | `max_seq_len`×`batch`×Cache 过大 / 权重精度太高 | 降 batch/seq、上 GQA、用 INT8/FP8 |
| 结果与 HF 略有差异 | 低精度/算子融合的数值误差 | 正常现象；要逐位对齐就拉高精度对比 |
| `mpirun -n` 与 TP×PP 不符 | 进程数 ≠ 并行总数 | 保证 `world_size = TP × PP`，一卡一 rank |
| 长上下文越跑越慢 | Decode 阶段访存密集 + Cache 增长 | 关注 KV Cache 优化，参考 [[llm-optimizer/kv-cache]] |

> 反复强调的护栏：本文里的脚本名、cmake flag、参数名是**类别与机制**层面的说明。**确切的镜像 tag、commit、脚本参数、默认值请以你所用 FT 分支的官方仓库 / README / 源码为准**，不同 fork 差异很大，切勿照抄硬记。

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- 上级（FT 总览 + 编译环境）：[[llm-inference/faster-transformer/README]]
- 推理总览：[[llm-inference/README]]
- 接班者（NVIDIA 主推）：[[llm-inference/tensorrt-llm/README]] [[llm-inference/tensorrt/README]]
- 服务化部署（FT 常配 Triton）：[[llm-inference/triton/README]]
- 在线服务对比：[[llm-inference/vllm/README]] [[llm-inference/sglang/README]]
- 通信底座（TP/PP 靠它）：[[ai-infra/网络/NCCL]] [[ai-infra/网络/集合通信原语]] [[ai-infra/网络/InfiniBand]]
- 显存与解码：[[llm-optimizer/kv-cache]] [[llm-optimizer/FlashAttention]]
- 低精度提速：[[llm-compression/quantization/量化基础]]
- 模型结构本身：[[llm-algo/transformer/模型架构]]

### 源头参考（务必以最新官方为准）

- FasterTransformer LLaMA 支持讨论：`https://github.com/NVIDIA/FasterTransformer/issues/506`
- LLaMA 适配 PR：`https://github.com/NVIDIA/FasterTransformer/pull/575`
- 社区 LLaMA fork（引擎）：`https://github.com/void-main/FasterTransformer`
- 社区 Triton 后端 fork：`https://github.com/void-main/fastertransformer_backend`
