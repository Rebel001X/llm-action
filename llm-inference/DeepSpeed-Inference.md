# DeepSpeed-Inference

> 把"训练框架 DeepSpeed"里那套并行 + 显存优化能力，重新打磨成"推理引擎"：靠**张量并行 + kernel 注入**压低单 token 延迟，靠**ZeRO-Inference 把权重卸载到 CPU/NVMe**让小显存也能跑超大模型。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/deepspeed/README]] [[llm-inference/offload]] [[llm-inference/大模型推理张量并行]]

## 阅读地图

| 你想知道 | 看哪节 |
|---|---|
| DeepSpeed-Inference 是啥、解决什么问题 | §0 §1 |
| 训练 DeepSpeed 和推理 DeepSpeed 是不是一个东西 | §1.3 |
| 推理张量并行（TP）怎么切、为什么能并行 | §2 |
| kernel 注入（injection）是什么黑魔法 | §3 |
| `init_inference` 一行代码背后发生了什么 | §4 |
| 显存放不下大模型 → ZeRO-Inference 卸载 | §5 |
| 数值例子：175B 要多少卡 / 卸载后多慢 | §6 |
| 和 vLLM / TGI / TensorRT-LLM 比怎么选 | §对照表 |
| 什么时候**别**用它 | §权衡 §常见问题 |

## 0. 一句话锚点

**DeepSpeed-Inference = 推理张量并行（低延迟）+ kernel 注入（高算子效率）+ ZeRO-Inference 卸载（小显存跑大模型）三件套**。

- 你有**多张卡**、想让一个大模型**延迟更低** → 用它的**张量并行 + kernel 注入**。
- 你**显存不够**、宁可慢也要把模型跑起来 → 用它的 **ZeRO-Inference**（把权重 offload 到 CPU/NVMe）。
- 入口就一行：`deepspeed.init_inference(model, ...)`，把 HuggingFace 模型"套"成推理引擎。

记住一个对立面：它**不是** vLLM 那种"以高并发吞吐为第一目标"的服务引擎。DeepSpeed-Inference 更偏"**把单个大模型在给定硬件上跑起来 / 跑快一点**"。两者解决的问题不同（详见 §对照表）。

## 1. 地基：它到底解决什么问题

### 1.1 大模型推理的两个核心痛点

推理（inference）= 只做前向（forward），没有反向（backward），但依然有两座大山：

```
痛点 A：显存装不下           痛点 B：单次前向太慢
┌────────────────┐         ┌────────────────┐
│ 175B 模型       │         │ 1 张卡串行算     │
│ FP16 ≈ 350 GB   │         │ 96 层 Transformer│
│ 一张 80GB 卡    │         │ 用户等半天才出字  │
│ 根本放不下      │         │ 延迟高           │
└────────────────┘         └────────────────┘
       │                          │
       ▼                          ▼
  解法 = 卸载/切分            解法 = 多卡并行 + 快算子
  (ZeRO-Inference)          (张量并行 + kernel 注入)
```

- **痛点 A（容量）**：参数显存 ≈ 参数量 × 每参数字节数。175B × 2 字节（FP16）≈ **350 GB**，远超单卡 80GB。
- **痛点 B（延迟）**：自回归生成是**一个 token 接一个 token**，每个 token 都要把整个模型前向一遍。模型越大，单 token 越慢。

DeepSpeed-Inference 对这两座山各给一把锤子：**TP+kernel 注入**砍延迟，**ZeRO-Inference**砍显存压力。

### 1.2 为什么"推理"值得单独做一个引擎

训练时我们也并行，但训练并行（数据并行 + 优化器状态切分）是为"算梯度、更新权重"设计的。推理没有梯度、没有优化器状态（Adam 的一阶/二阶动量），显存结构完全不同：

$$
\text{训练显存} = \underbrace{W}_{\text{权重}} + \underbrace{G}_{\text{梯度}} + \underbrace{O}_{\text{优化器状态}} + \underbrace{A}_{\text{激活}}
$$
$$
\text{推理显存} = \underbrace{W}_{\text{权重}} + \underbrace{KV}_{\text{KV Cache}} + \underbrace{A_{\text{小}}}_{\text{激活(无需保存)}}
$$

推理把 $G$ 和 $O$ 全去掉了（训练里这俩往往是权重的 2~12 倍），所以推理引擎可以把省下来的算力/带宽，专门投到"让前向更快、让权重能放进更小的地方"。

### 1.3 与训练 DeepSpeed 的关系（关键！别搞混）

DeepSpeed 是一个**大库**，里面分两条线：

```
            DeepSpeed (同一个 pip 包: pip install deepspeed)
            ┌──────────────────────────┬──────────────────────────┐
            │      训练侧 (Training)     │     推理侧 (Inference)     │
            ├──────────────────────────┼──────────────────────────┤
 入口        │ deepspeed.initialize()   │ deepspeed.init_inference()│
 目标        │ 训得动、训得快、省显存     │ 跑得动、延迟低             │
 核心招式     │ ZeRO-1/2/3 (切梯度/优化器)│ 张量并行 + kernel 注入     │
            │ 流水线并行、混合精度训练   │ ZeRO-Inference (卸载权重)  │
 有无反向     │ 有 backward + 优化器更新  │ 只有 forward              │
            └──────────────────────────┴──────────────────────────┘
                         │                          │
                  共享底层: 通信原语(NCCL)、ZeRO 的"参数分片+按需聚合"思想
```

- **是同一个包**：`import deepspeed`，训练用 `initialize()`，推理用 `init_inference()`。
- **复用了思想**：ZeRO-Inference 的"权重不常驻、按需取回"直接脱胎于训练侧 ZeRO-3 的"参数分片 + 用前 all-gather"。
- **但目标相反**：训练 ZeRO 是为了"塞下优化器状态训得动"；推理 ZeRO 是为了"塞下权重跑得动"。
- 详细训练侧机制见 [[ai-framework/deepspeed/README]]。

## 2. 推理张量并行（Tensor Parallelism, TP）

### 2.1 张量并行是切什么

张量并行 = 把**单个大矩阵乘法**横着/竖着切成几块，分给多张卡同时算，再把结果拼/加回来。它切的是**层内的权重矩阵**，不是"把不同层放不同卡"（那是流水线并行）。

Transformer 里最吃显存/算力的就两块：**MLP（前馈）** 和 **多头注意力（MHA）**。TP 对这两块都有标准切法。

```
MLP 的张量并行（最经典）：  Y = GeLU(X · A) · B
                          X: [b, h]   A: [h, 4h]   B: [4h, h]

      第一个权重 A 按列切          第二个权重 B 按行切
   ┌──────────────────────┐    ┌──────────────────────┐
   │  A = [A1 | A2]        │    │  B = [ B1 ]          │
   │       列切            │    │      [ B2 ]  行切     │
   └──────────────────────┘    └──────────────────────┘

GPU0:  X·A1 → GeLU → ·B1 ─┐
GPU1:  X·A2 → GeLU → ·B2 ─┴─► all-reduce 求和 ► Y
       (每卡独立算一半)        (一次通信合并)
```

**为什么这样切就对了**：列切 A 后，GeLU 是逐元素的（element-wise），每一列独立，不需要通信；到了第二个矩阵 B 按行切，两半的部分积相加正好等于完整结果——所以每个 Transformer block 只需在末尾做**一次 all-reduce**。注意力部分同理：按"头（head）"维度切，每张卡算一部分头。

### 2.2 通信代价：一次 all-reduce 要传多少

一个 Transformer block 在 TP 下，MLP 和 Attention 各做一次 all-reduce。all-reduce 传输量近似：

$$
\text{每次 all-reduce 字节} \approx 2 \times \frac{N-1}{N} \times (\text{batch} \times \text{seq} \times h \times \text{bytes})
$$

其中 $N$ 是卡数，$h$ 是隐藏维。**关键结论**：通信量正比于"激活大小"（batch×seq×h），**与模型参数量无关**。所以 TP 在卡间带宽高（NVLink）时很划算，跨机（PCIe/网卡）时通信会变成瓶颈——**这就是 TP 通常只用在单机内多卡的根本原因**。

### 2.3 自动张量并行（Automatic TP）

老式做法要手写每层怎么切（Megatron 风格）。DeepSpeed-Inference 提供**自动张量并行**：你只给 `mp_size`（并行卡数），它自动识别 HuggingFace 模型结构（哪些是 attention/MLP 线性层），自动按上面规则切。文件顶部那段示例就是它：

```python
import deepspeed, torch, transformers, os
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
pipe = transformers.pipeline(task="text2text-generation",
                             model="google/t5-v1_1-small", device=local_rank)
pipe.model = deepspeed.init_inference(pipe.model,
                                      mp_size=world_size,  # ← 张量并行度
                                      dtype=torch.float)
output = pipe('Input String')
# 启动: deepspeed --num_gpus 2 your_script.py
```

> 自动 TP 的覆盖模型范围、注入策略随版本演进，**具体支持清单以官方文档为准**。

## 3. kernel 注入（Kernel Injection）

### 3.1 问题：朴素 PyTorch 算子太碎

一个 Transformer block 在原生 PyTorch 里被拆成几十个小算子：LayerNorm、QKV 投影、softmax、bias add、GeLU、残差……每个小算子都要：从显存读数据 → 算 → 写回显存。**大量时间花在"搬数据"而不是"算"**（memory-bound）。

```
朴素 PyTorch (碎):                kernel 注入 (融合):
LN  → 读/写                       ┌─────────────────────────┐
QKV → 读/写                       │ 一个大融合 kernel:        │
softmax → 读/写         ===>      │ LN+QKV+attn+bias+GeLU... │
GeLU → 读/写                      │ 中间结果留在寄存器/SRAM   │
bias → 读/写                      │ 只在首尾读/写一次显存     │
... 几十次显存往返                └─────────────────────────┘
   慢 (memory-bound)                快 (减少显存往返)
```

### 3.2 注入做了什么

**kernel 注入** = DeepSpeed 在加载模型时，**把 HuggingFace 模型里标准的 Transformer 层，替换成 DeepSpeed 手写优化的 CUDA 融合 kernel**。"注入"这个词的意思就是：它在运行时把你模型里的 `nn.Module`（比如 `BertLayer`、`GPT2Block`）偷偷换成等价但更快的实现。

- **算子融合**：把多个小算子合成一个大 kernel，减少显存往返。
- **针对推理优化**：例如对小 batch、自回归解码场景特调。
- 触发方式：`replace_with_kernel_inject=True`（部分模型自动识别；老 API 也用 `injection_policy` 手动指定哪几层替换）。

### 3.3 注意：注入与 TP 是两件事，可叠加

```
                  ┌──── 张量并行 (TP) ────┐   切矩阵 → 多卡分担 → 砍延迟/省单卡显存
init_inference ───┤
                  └──── kernel 注入 ──────┘   换快算子 → 单卡内更高效

二者正交、可同时开:  既切到 4 卡, 每张卡内又用融合 kernel
```

## 4. `init_inference` 背后的流水线

一行 `deepspeed.init_inference(model, mp_size=4, ...)` 实际做了这些事：

```
你给一个 HF model
      │
      ▼
1. 起 N 个进程 (deepspeed --num_gpus N)，建 NCCL 通信组
      │
      ▼
2. 解析模型结构，识别 attention / MLP 线性层
      │
      ▼
3. [TP] 把权重矩阵按 §2 规则切成 N 份，每张卡只留 1/N
      │
      ▼
4. [注入] 把标准 Transformer 层替换成融合 CUDA kernel
      │
      ▼
5. [可选 dtype 转换] 转 fp16/bf16/int8
      │
      ▼
6. 返回包好的 InferenceEngine，forward 时自动做 all-reduce
```

> 关键参数（概念层面，**精确签名以官方文档为准**）：`mp_size`（TP 度）、`dtype`（fp16/bf16/int8）、`replace_with_kernel_inject`（是否注入）、以及 ZeRO 配置（见 §5）。

## 5. ZeRO-Inference：用卸载换"跑得动"

### 5.1 核心思想：权重不必全程待在 GPU

ZeRO-Inference 的洞察：**推理时，权重在用到的那一刻才需要在 GPU 上**。算第 5 层时，第 50 层的权重完全可以躺在 CPU 内存甚至 NVMe SSD 里。于是：

```
传统: 所有权重常驻 GPU         ZeRO-Inference: 权重住 CPU/NVMe，按需取回
┌──────────────┐              ┌──────────────┐   计算第 i 层时:
│ GPU 80GB     │              │ GPU 只放      │   ┌─────────────┐
│ [全部 350GB] │  放不下!     │ 当前几层 +    │   │1.从CPU/NVMe │
│   ✗          │              │ KV Cache      │◄──│  取第 i 层权重│
└──────────────┘              └──────┬───────┘   │2.算          │
                                     │           │3.丢掉，取 i+1 │
                              ┌──────▼───────┐   └─────────────┘
                              │ CPU 内存 /    │   (prefetch 下一层
                              │ NVMe SSD      │    与计算重叠隐藏延迟)
                              │ [全部权重]    │
                              └──────────────┘
```

这正是训练侧 ZeRO-3 / ZeRO-Offload 思想在推理上的复用：**参数分层卸载 + 用前取回 + 预取重叠**。机制细节见 [[llm-inference/offload]]。

### 5.2 三级存储与带宽

```
            容量 小→大        带宽 快→慢
   GPU HBM  ───────────────────────────►  ~1-3 TB/s   (放当前层+KV)
   CPU DRAM ───────────────────────────►  ~10-50 GB/s (放全部权重，便宜)
   NVMe SSD ───────────────────────────►  ~2-7 GB/s   (装超大模型的兜底)
```

**代价非常直接**：权重从 CPU/NVMe 搬到 GPU 是有带宽上限的，这条搬运路径往往成为新瓶颈，吞吐会显著下降——**ZeRO-Inference 是"宁可慢，先跑起来"的方案**（典型场景：单卡跑超大模型、离线批处理）。

### 5.3 ZeRO-Inference vs 张量并行：解决不同问题

| 维度 | 张量并行 (TP) | ZeRO-Inference (卸载) |
|---|---|---|
| 解决 | 延迟高 / 单卡装不下 | 显存绝对不够（卡少/卡小） |
| 需要 | 多张卡 + 高速互联(NVLink) | 哪怕 1 张卡 + 大内存/SSD |
| 速度 | 快（多卡并行） | 慢（受卸载带宽限制） |
| 典型 | 在线低延迟服务 | 离线批处理 / 资源受限跑通 |
| 可否同时用 | 可以叠加：TP 跨卡 + 卸载省显存 | |

## 6. 数值例子 / 典型场景

### 例 1：175B 模型 FP16 要几张卡（纯 TP，不卸载）

参数显存：$175\text{e}9 \times 2\,\text{B} = 350\,\text{GB}$。
单张 A100 80GB，扣掉 KV Cache / 激活 / 框架开销，可用约 60GB 放权重：

$$
N = \left\lceil \frac{350}{60} \right\rceil = 6 \text{ 张（实务常取 } 8 \text{ 张做 TP=8)}
$$

TP=8 时每卡只放 $350/8 \approx 44\,\text{GB}$ 权重，剩余显存留给 KV Cache，舒适。

### 例 2：TP 的通信量（直观感受）

设 $h=12288$（175B 量级），batch=1，seq=1（解码单 token），FP16，TP=8：
每次 all-reduce ≈ $2 \times \frac{7}{8} \times (1\times1\times12288\times2) \approx 43\,\text{KB}$。
每层 2 次、96 层 → 单 token 约 $96\times2\times43\text{KB} \approx 8\,\text{MB}$ 通信。
在 NVLink（~600 GB/s）上微不足道；但若跨机走 PCIe/网卡，延迟立刻凸显——**故 TP 留在单机**。

### 例 3：ZeRO-Inference 卸载的吞吐直觉

假设单卡，模型权重 350GB 全在 NVMe（5 GB/s）。每生成 1 个 token 要把全部权重过一遍 GPU：

$$
t_{\text{搬运}} \approx \frac{350\,\text{GB}}{5\,\text{GB/s}} = 70\,\text{s / token}
$$

→ 单条几乎不可用。**但**：若用**大 batch**（比如一次 256 条），这 70s 搬运被 256 条样本**摊销**，每条样本均摊 $70/256 \approx 0.27\text{s}$，吞吐就回来了。**这就是为什么 ZeRO-Inference 偏离线大批量场景**——权重搬一次、喂尽量多样本。

> 以上为数量级估算，演示"为什么这么选"，非实测基准。

### 例 4：何时单机多卡用 DeepSpeed-Inference

- 1 台 8×A100/H100 机器，跑一个 30B~70B 模型做内部低延迟服务 → TP=4/8 + kernel 注入，单机吃满 NVLink，延迟低、不跨机。
- 只有 1~2 张卡，却要跑 70B/175B 做离线评测/数据生成 → ZeRO-Inference 卸载到 CPU/NVMe + 大 batch。

## 对照表（与同类推理方案对比）

| 方案 | 第一目标 | 招牌能力 | 弱项 | 何时选 |
|---|---|---|---|---|
| **DeepSpeed-Inference** | 跑得动 + 低延迟 | 自动 TP、kernel 注入、ZeRO 卸载 | 高并发吞吐/连续批处理非强项 | 单机多卡跑大模型；显存不够要卸载 |
| **vLLM** | 高吞吐在线服务 | PagedAttention、连续批处理(continuous batching) | 卸载/超大单模型非主打 | 高 QPS 在线 API 服务 |
| **TGI** (HF) | 易部署的服务 | 开箱即用、生态好 | 极致性能略逊专用引擎 | 想快速上线 HF 模型 |
| **TensorRT-LLM** | 极致单卡/多卡延迟 | NVIDIA 深度图优化、量化 | 编译/适配成本高、绑 N 卡 | 追求极限延迟、愿做编译优化 |

> 工程上常**组合**：DeepSpeed-Inference 的 TP/注入想法与 vLLM 的 PagedAttention 在不同项目里被混搭参考。选型看你的瓶颈是**延迟**、**吞吐**还是**显存装不下**。

## 常见问题

| 问题 | 答 |
|---|---|
| 训练用了 DeepSpeed，推理必须也用它吗？ | 不必。训练/推理可换引擎。推理选谁看瓶颈（见对照表）。 |
| 张量并行能跨机吗？ | 技术上能，但通信量正比于激活、且每层都要 all-reduce，跨机带宽低会拖垮延迟。**实务 TP 留单机**，跨机用流水线/数据并行。 |
| kernel 注入对所有模型都生效吗？ | 否，依赖 DeepSpeed 是否为该结构提供了融合 kernel/注入策略。**支持清单以官方文档为准**。 |
| ZeRO-Inference 为什么这么慢？ | 权重要从 CPU/NVMe 搬到 GPU，搬运带宽成瓶颈。用**大 batch 摊销**搬运、离线场景用，能把吞吐救回来（见例 3）。 |
| TP 和 ZeRO-Inference 冲突吗？ | 不冲突，可叠加：先 TP 把权重切到多卡，再对剩余部分卸载。 |
| 它能像 vLLM 那样做高并发连续批处理吗？ | 这不是它的主战场。要高 QPS 在线服务优先看 vLLM/TGI。 |
| `mp_size` 设几合适？ | 取能装下权重的最小卡数附近，过大反而增加通信开销；常用 2/4/8。 |
| FP16/BF16/INT8 怎么选？ | 显存紧、可接受精度损失 → 量化（INT8）；否则 BF16 更稳。`dtype` 控制。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航总入口
- [[ai-framework/deepspeed/README]] — 训练侧 DeepSpeed（ZeRO-1/2/3、流水线并行）总览，理解推理 ZeRO 的来源
- [[llm-inference/offload]] — 显存卸载机制（CPU/NVMe offload、预取重叠）细节，ZeRO-Inference 的底层
- [[llm-inference/大模型推理张量并行]] — 张量并行切法（MLP 列/行切、注意力按头切、all-reduce）深入
