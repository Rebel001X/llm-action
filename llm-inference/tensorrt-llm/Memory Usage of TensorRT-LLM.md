# TensorRT-LLM 显存占用(Memory Usage)

> 把 TensorRT-LLM 推理时一块 GPU 上的显存"花在哪里"讲透:权重、激活、I/O(KV Cache),以及引擎构建期与运行期各自的显存峰值。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/tensorrt-llm/TRT-LLM引擎构建参数]] [[llm-inference/KV-Cache优化]] [[llm-optimizer/kv-cache]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

> 一手依据(以官方为准):
> https://nvidia.github.io/TensorRT-LLM/reference/memory.html#understand-inference-time-gpu-memory-usage

## 阅读地图

| 小节 | 解决什么疑问 | 关键词 |
| --- | --- | --- |
| 0 | 一句话锚点 | 三大块:权重 / 激活 / I/O(KV) |
| 1 | 为什么要关心显存 | OOM、并发、最大序列长度 |
| 2 | 显存总账(全景 ASCII) | 构建期 vs 运行期 |
| 3 | 权重(Weights) | 量化、TP/PP 切分 |
| 4 | 内部激活(Activation) | 临时张量、显存池复用 |
| 5 | I/O 与 KV Cache(大头) | 计算公式、Paged KV |
| 6 | 运行期还有谁占显存 | CUDA Context、cuBLAS、碎片 |
| 7 | 构建期(build)峰值 | 优化器/profiling 显存 |
| 8 | 关键配置项怎么权衡 | free_gpu_memory_fraction 等 |
| 9 | 估算示例(数值) | 7B 模型一步步算 |
| 坑 | 常见问题与排错 | OOM、碎片、并发塌缩 |

## 0. 一句话锚点

TensorRT-LLM 推理时,一块 GPU 的显存主要被三类东西吃掉:

1. **权重(Weights)**:模型参数,静态、几乎不变。
2. **内部激活张量(Internal Activation Tensors)**:前向计算的临时中间结果,由 TensorRT 的显存池统一管理、尽量复用。
3. **I/O 张量(I/O Tensors)**:输入输出张量,**其中绝大部分是 KV Cache**,且随并发数 × 序列长度线性增长——这是推理显存里最"会膨胀"的一块。

> 记忆口诀:**权重是地基(固定),激活是脚手架(临时复用),KV Cache 是会长高的楼(随请求增长)。**

## 1. 地基:为什么必须搞懂显存账

推理服务最常见的事故就是 **OOM(Out Of Memory)**。而推理引擎的吞吐能力,本质上由"显存能塞下多少并发请求 × 多长的序列"决定。搞清楚显存花在哪,你才能回答这些工程问题:

- 这块卡(比如 24GB / 40GB / 80GB)能跑多大的模型?
- 同时能服务多少并发请求(batch)?每条最长能到多少 token?
- 提高 `max_batch_size` / `max_seq_len` 会让显存怎么涨,会不会 OOM?
- 为什么 build(构建引擎)的时候显存峰值比实际跑还高?

如果不理解这本账,你只能靠"试一下、OOM 了再调小"的方式碰运气,既慢又不可控。

## 2. 显存总账:全景图

TensorRT-LLM 的生命周期分两个阶段,显存画像完全不同:

- **构建期(Build / Engine 编译)**:把网络定义编译成优化后的 TensorRT engine,会做算子选择(kernel auto-tuning)、内存规划,期间可能出现**临时的高峰显存**。
- **运行期(Runtime / Inference)**:加载 engine 权重,分配激活池与 KV Cache,真正服务请求。

```
                ┌──────────────────────── GPU 显存(总容量) ────────────────────────┐
                │                                                                  │
  运行期        │  [CUDA Context/驱动开销]   ← 每进程固定几百 MB                     │
  Runtime       │  [模型权重 Weights]        ← 静态,量化后更小,TP/PP 切分后更小    │
                │  [内部激活池 Activation]   ← TensorRT 统一规划、复用,峰值受限     │
                │  [I/O 张量(主要是 KV Cache)] ← 随 batch × seq_len 线性膨胀(大头)│
                │  [cuBLAS/cuDNN workspace]  ← 库的工作区                           │
                │  [显存碎片 Fragmentation]  ← 分配/释放产生的空洞                   │
                │                                                                  │
                └──────────────────────────────────────────────────────────────────┘

  构建期        ┌──────────────────────── GPU 显存 ────────────────────────┐
  Build         │  [优化器/profiling 临时缓冲]  ← kernel auto-tuning、试跑   │
                │  [权重 + 中间编译产物]                                      │
                │  → 峰值可能 ≥ 运行期,需预留空间或单独在大卡上 build        │
                └──────────────────────────────────────────────────────────┘
```

下面逐块拆。

## 3. 权重(Weights):静态地基

权重就是模型参数 $W$。它的显存占用近似为:

$$\text{显存}_{\text{weights}} \approx N_{\text{params}} \times \text{bytes\_per\_param}$$

- **bytes\_per\_param** 由精度决定:FP16/BF16 = 2 字节,FP8 = 1 字节,INT4 ≈ 0.5 字节(再加少量缩放因子/zero-point 元数据)。
- 量化(quantization)直接削减这一块,是显存优化的第一抓手。详见 [[llm-compression/quantization/量化基础]] 与 [[llm-inference/tensorrt-llm/FP8]]。

**多卡切分**:在张量并行(TP)/流水并行(PP)下,权重被切到多张卡上,**每张卡只持有一份分片**:

$$\text{每卡}_{\text{weights}} \approx \frac{N_{\text{params}} \times \text{bytes}}{\text{TP} \times \text{PP}}$$

> 例:70B 模型 FP16 全量 ≈ 140GB,单卡放不下;TP=4 后每卡约 35GB(再叠加激活与 KV),才可能落在 40GB/80GB 卡上。多卡通信细节见 [[ai-infra/网络/集合通信原语]]。

权重特点:**一次加载、长期常驻、不随请求变化**——它是显存账里"最可预测"的一块。

## 4. 内部激活张量(Internal Activation Tensors)

前向过程中,每一层都会产生中间张量(attention 分数、MLP 中间维度、LayerNorm 输出等)。如果像训练那样把它们全留着,显存会爆。推理不需要保留它们用于反向传播,所以:

- TensorRT 在**构建期就规划好一个激活显存池(scratch / workspace)**,在推理时让生命周期不重叠的张量**复用同一块物理显存**。
- 因此激活的占用不是"所有层之和",而更接近"**任一时刻同时存活的最大那组张量**"——这是图着色式的内存分配优化。

```
  逻辑上每层都有激活张量:
  layer0_act ──┐
  layer1_act ──┼─►  生命周期不重叠时,复用同一块物理显存
  layer2_act ──┘
                   ┌────────── 激活池(固定大小) ──────────┐
   时刻 t1:        │ ████████  (layer0 在用)                │
   时刻 t2:        │ ████████  (layer0 释放, layer1 复用)   │
                   └──────────────────────────────────────┘
```

影响激活池大小的因素:`batch_size`、`max_seq_len`(尤其 prefill 阶段一次喂入很长 prompt 时)、hidden size、注意力实现(如启用 fused/flash 类 kernel 会改变临时张量形状,见 [[llm-optimizer/FlashAttention]])。

> 实践注意:激活池大小由构建期 profiling 的"最坏形状"决定。如果你 build 时声明的 `max_batch_size`/`max_input_len` 很大,激活池会按最坏情况预留,显存随之上涨——即使你实际跑的请求很小。

## 5. I/O 张量与 KV Cache(显存大头)

I/O 张量是引擎的输入(token ids、position 等)和输出(logits 等)。其中**最大的一块是 KV Cache**。

### 5.1 KV Cache 是什么、为什么需要

自回归解码每生成一个新 token,都要对之前所有 token 做注意力。如果每步都重算所有 Key/Value,复杂度是 $O(n^2)$ 且重复劳动巨大。**KV Cache 把每个 token 在每层算出的 K、V 缓存下来**,后续步骤直接复用,把每步注意力降到 $O(n)$。代价就是显存。机制详解见 [[llm-inference/KV-Cache优化]] 与 [[llm-optimizer/kv-cache]]。

### 5.2 KV Cache 显存公式

单条序列、所有层、K 和 V 两份的 KV Cache 字节数近似为:

$$\text{KV}_{\text{bytes}} = 2 \times L_{\text{layers}} \times S_{\text{seq}} \times H_{\text{kv}} \times d_{\text{head}} \times b_{\text{dtype}}$$

- $2$:K 与 V 各一份
- $L_{\text{layers}}$:层数
- $S_{\text{seq}}$:该序列已缓存的 token 数(= prompt + 已生成)
- $H_{\text{kv}} \times d_{\text{head}}$:KV 的总隐藏维度;注意 **GQA/MQA** 下 KV 头数 $H_{\text{kv}}$ 远小于注意力头数,这一块按比例缩小
- $b_{\text{dtype}}$:每元素字节数(FP16=2,FP8/INT8=1 —— 即 **KV Cache 量化**可再砍一半)

整机并发的 KV 总量约为对所有在跑请求求和:$\sum_{i} \text{KV}_{\text{bytes}}(S_i)$。**这就是为什么并发越高、序列越长,显存越紧张。**

### 5.3 Paged KV Cache(分页)

连续分配 KV 会因不同请求长度不一而产生大量碎片与浪费。TensorRT-LLM 采用**分页(paged)KV Cache**:把 KV 切成固定大小的 **block(页)**,按需从一个预分配的"KV 缓存池"里取页,逻辑上像操作系统的虚拟内存分页。

```
   预分配 KV Cache 池(运行期一次性吃下一大块显存)
   ┌──────────────────────────────────────────────────────────┐
   │ blk0  blk1  blk2  blk3  blk4  blk5  blk6  blk7 ...         │
   └──────────────────────────────────────────────────────────┘
        │      │            │      │
        ▼      ▼            ▼      ▼
   请求A: [blk0][blk1]   请求B: [blk3][blk2][blk4]   (按需取页,可不连续)
```

好处:**显存利用率高、碎片少、便于做前缀复用/驱逐**。代价:池大小是**预先一次性分配**的——这正是 `free_gpu_memory_fraction`(见第 8 节)控制的对象。

## 6. 运行期的"隐形"占用

除了上面三大块,运行期还有几项容易被忽略:

| 占用项 | 来源 | 量级感觉 |
| --- | --- | --- |
| CUDA Context / 驱动 | 每个进程初始化 CUDA 时的固定开销 | 几百 MB,跑不掉 |
| cuBLAS / cuDNN workspace | 矩阵乘/卷积库的临时工作区 | 与算子规模相关 |
| 通信缓冲(NCCL) | 多卡 TP/PP 的 all-reduce 等需要缓冲区 | 随并行规模上升,见 [[ai-infra/网络/NCCL]] |
| 显存碎片 | 反复分配/释放留下的空洞 | 长时间服务后可能可观 |

这些加起来不大,但在"卡得很紧"的部署里,正是它们决定你 OOM 还是不 OOM。

## 7. 构建期(Build)显存峰值

构建 engine 时,TensorRT 会做 **kernel auto-tuning**:针对当前形状试跑多种候选 kernel、测速、选最优。这个过程会分配额外临时缓冲,**峰值可能不低于、甚至高于运行期**。

实践注意:
- 如果 build 时 OOM,但你确信运行期能放下,通常是 build 峰值在作祟。可以在**更大显存的卡上 build**,产物 engine 再拿到目标卡上跑(注意:engine 与 GPU 架构/版本强绑定,跨架构一般不通用,具体约束以官方文档为准)。
- build 期声明的最大形状(max batch / max input / max output)越大,后续运行期预留越多——**按真实业务上限声明,别一上来就拉满。**

## 8. 关键配置项怎么权衡(讲含义,不背默认值)

> 以下是**类别与作用**的说明。具体参数名、默认值、单位请以你所用版本的官方文档 / `--help` 为准,不要照抄记忆中的数字。引擎构建侧参数另见 [[llm-inference/tensorrt-llm/TRT-LLM引擎构建参数]]。

| 配置类别(语义) | 它控制什么 | 调大 → | 调小 → |
| --- | --- | --- | --- |
| **KV 缓存可用显存比例**(如 `free_gpu_memory_fraction` / KV cache 占比) | 给 KV Cache 池预分配多少剩余显存 | 能容纳更多/更长并发,吞吐高 | 更保守、留更多余量防 OOM |
| **最大并发 / batch size** | 同时在跑的请求数上限 | 吞吐↑,但 KV 与激活↑ | 省显存,吞吐↓ |
| **最大序列长度 / 输入输出长度** | 单请求最长 token | 支持长上下文 | 省 KV 显存 |
| **权重量化精度**(FP16/FP8/INT8/INT4) | 权重每参数字节数 | —— | 省权重显存(可能影响精度) |
| **KV Cache 量化**(FP8/INT8 KV) | KV 每元素字节数 | —— | 直接腰斩 KV 显存 |
| **并行度 TP / PP** | 把权重/计算切到几张卡 | 单卡权重/激活↓,但加通信 | 卡少、单卡压力大 |
| **分页 block 大小** | KV 分页粒度 | 大块→管理简单、可能浪费 | 小块→利用率高、元数据多 |

**核心权衡三角**:`模型大小(权重)` ↔ `并发×序列长度(KV)` ↔ `卡的显存`。三者满足:

$$\text{权重} + \text{激活峰值} + \text{KV池} + \text{固定开销} \;\le\; \text{单卡显存}$$

工程上常见做法:先按权重定下能不能放、需不需要 TP;再用"剩余显存"去喂 KV 池,通过 `free_gpu_memory_fraction` 把剩余尽量给 KV 以拉满吞吐,同时留一点余量防碎片/隐形开销。

## 9. 估算示例(一步步算)

设一个 7B 类模型:层数 $L=32$,隐藏维 $H=4096$,注意力头 32 个 head、$d_{head}=128$,**无 GQA**(KV 头 = 32),权重 FP16,KV FP16,目标卡 24GB。

**第 1 步:权重**
$7\text{B} \times 2\text{ byte} = 14\text{ GB}$。

**第 2 步:固定开销 + 激活池(粗估)**
CUDA Context + 库 workspace + 激活池,工程上保守按 1.5~2.5GB 预留。取 2GB。

**第 3 步:单 token 的 KV(每条序列每 token)**
$2 \times L \times H_{kv} \times d_{head} \times b = 2 \times 32 \times (32 \times 128) \times 2 = 2 \times 32 \times 4096 \times 2$
$= 524{,}288\text{ byte} \approx 0.5\text{ MB / token}$。

**第 4 步:剩余显存能喂多少 token**
剩余 $\approx 24 - 14 - 2 = 8\text{ GB}$。
可缓存 token 数 $\approx 8\text{GB} / 0.5\text{MB} \approx 16{,}000\text{ token}$。

**第 5 步:换算成并发**
若每条序列约 2048 token,则约能并发 $16000 / 2048 \approx 8$ 条;若每条只有 512 token,则约 31 条。
**结论**:同样一张卡,短序列能跑更高并发;长上下文显著吃 KV。若把 KV 量化到 FP8,KV 减半,token 容量翻倍。

> 以上为数量级估算,真实数字受分页粒度、对齐、激活峰值、库开销影响,请以实测为准。

## 常见问题/坑

| 现象 | 根因 | 处理思路 |
| --- | --- | --- |
| 加载就 OOM | 权重 + 固定开销已超卡 | 量化权重 / 开 TP 多卡 / 换大卡 |
| 跑一会儿才 OOM | 并发/长序列把 KV 池撑爆 | 调小 KV 比例、限并发、限最大长度、KV 量化 |
| build 时 OOM,但应能跑 | auto-tuning 峰值高 | 在大显存卡上 build,产物挪到目标卡 |
| 显存够却吞吐上不去 | KV 池给得太保守 | 调大 KV 可用显存比例(留余量) |
| 长时间服务后偶发 OOM | 显存碎片累积 | 用分页 KV、定期观察、留余量 |
| 改了 batch/seq 显存暴涨 | 激活池/ KV 按最坏形状预留 | build 时按真实上限声明,别拉满 |
| 多卡显存不均 | TP/PP 切分或 KV 分布不均 | 检查并行配置与负载均衡 |
| 声明 max_seq_len 很大就慢/占多 | 激活池按最坏形状规划 | 仅声明业务真实需要的长度 |

> 监控建议:用 `nvidia-smi`、`pynvml` 或 TensorRT-LLM/Triton 自带的统计观测"已分配/可用/碎片",别只看一个总数。性能指标术语见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]。

## 🔗 跳转链接

- 返回总览：[[00-知识地图]]
- 同目录:[[llm-inference/tensorrt-llm/TRT-LLM引擎构建参数]]、[[llm-inference/tensorrt-llm/FP8]]
- KV 机制:[[llm-inference/KV-Cache优化]]、[[llm-optimizer/kv-cache]]
- 注意力优化:[[llm-optimizer/FlashAttention]]
- 量化省显存:[[llm-compression/quantization/量化基础]]
- 推理总览:[[llm-inference/README]]、[[llm-inference/vllm/README]]
- 多卡通信:[[ai-infra/网络/NCCL]]、[[ai-infra/网络/集合通信原语]]
- 指标术语:[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
