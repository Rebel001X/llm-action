# FlexFlow Serve：基于 SpecInfer 的低延迟推理引擎

> 用「小模型猜 + 大模型并行验证」的投机推理，把 LLM 端到端延迟打下来的服务系统。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/README]] · [[llm-inference/解码策略]] · [[llm-inference/vllm/README]] · [[llm-optimizer/kv-cache]]

## 阅读地图

| 你想知道的 | 跳到 |
| --- | --- |
| FlexFlow Serve 是什么、一句话定位 | [§0](#0-一句话锚点) |
| 它跟 FlexFlow / vLLM 什么关系 | [§1](#1-地基前置编译型框架--推理分支) |
| 投机推理（SpecInfer）到底怎么加速 | [§2](#2-投机推理speculative--token-树并行验证) |
| token 树、并行解码机制 | [§2](#2-投机推理speculative--token-树并行验证) |
| 单卡跑大模型：CPU Offloading | [§3](#3-cpu-offloading单-gpu-跑大模型) |
| int4 / int8 量化怎么省显存 | [§4](#4-量化int4--int8) |
| 多 GPU 并行配置怎么填（4 个 degree） | [§5](#5-并行配置tensor--pipeline--gpu-内存) |
| 完整可跑代码 / Docker / 数据集 | [实操](#实操命令--配置--代码原文真料) |
| 容易踩的坑 | [常见坑](#常见问题--坑) |

## 0. 一句话锚点

**FlexFlow Serve = FlexFlow 框架的 `inference` 分支 + SpecInfer 投机推理算法。** 核心思想一句话：让一个或多个**小投机模型 (SSM, Small Speculative Model)** 先快速「猜」出未来若干个 token，组织成一棵 **token 树**；再让**大模型 (LLM) 一次前向并行验证**整棵树，命中的部分直接采纳。这样把「LLM 逐 token 自回归」变成「LLM 批量验证」，在**保持输出分布不变**的前提下大幅降低延迟。

```
传统自回归:  LLM → t1 → LLM → t2 → LLM → t3 → ...   (N 个 token = N 次 LLM 前向)
FlexFlow:    SSM 猜出 [t1 t2 t3 ...] 树 → LLM 一次前向并行验证 → 采纳命中前缀
             (N 个 token 可能只需 1~2 次 LLM 前向)
```

## 1. 地基/前置：编译型框架 + 推理分支

**FlexFlow 本体**是一个**编译型**分布式深度学习框架（区别于 PyTorch 的「即时执行」）：它把模型先 `compile` 成计算图，自动搜索张量/流水线并行的切分策略，再执行。FlexFlow Serve 是它专门用于**推理**的分支。

- 仓库：`https://github.com/flexflow/FlexFlow/tree/inference`（推理分支，**当前领先于 master**）
- 主仓：`https://github.com/flexflow/FlexFlow.git`

为什么要先理解「编译型」？因为下文代码里每个模型（LLM 和每个 SSM）都要显式 `.compile(...)`——这一步会把权重载入显存、固化并行策略、生成 token 树验证 kernel。这跟 vLLM「加载即用」的体验不同，是 FlexFlow 的基因。

```
        ┌──────────── FlexFlow (编译型 DL 框架) ────────────┐
        │  自动并行搜索  |  计算图编译  |  分布式执行         │
        └───────────────────────┬───────────────────────────┘
                                 │ inference 分支
                          ┌──────▼──────┐
                          │ FlexFlow    │  = SpecInfer 算法 + 服务接口
                          │   Serve     │
                          └─────────────┘
```

> 对照锚点：要理解为什么投机推理「能并行验证而不破坏分布」，需要先有 [[llm-inference/解码策略]]（贪心/采样/温度）和 [[llm-optimizer/kv-cache]] 的概念。

## 2. 投机推理（Speculative）—— token 树并行验证

> 原文原理（保留）：使 FlexFlow Serve 能够加速 LLM 服务的一项关键技术是 Speculative 推理，它结合了各种集体 boost-tuned 的小型投机模型 (SSM) 来共同预测 LLM 的输出。预测被组织为 **token 树**，每个节点代表一个候选 token 序列。使用一种新颖的**基于树的并行解码机制**，根据 LLM 的输出并行验证由 token 树表示的所有候选 token 序列的正确性。FlexFlow Serve 使用 **LLM 作为 token 树验证器而不是增量解码器**，这大大减少了端到端推理延迟和计算要求，同时**可证明保持模型质量**。

### 2.1 为什么慢？自回归的本质瓶颈

LLM 解码是 **memory-bound（访存受限）**：每生成一个 token，都要把**全部权重**从显存读一遍，但只算了 1 个 token 的矩阵乘。算力（FLOPs）大量闲置，瓶颈是把几十 GB 权重搬进片上的带宽。

**关键洞察**：如果一次前向能「顺手」算多个 token，权重只读一遍就摊薄了访存成本——这正是投机推理省时间的物理原因。

### 2.2 投机推理三步走

```
① 猜 (Draft)      多个 SSM 快速自回归，产出候选 token
② 组树 (Tree)     候选组织成 token 树，节点 = 候选 token 序列
                    ┌── t2a ── t3a
          t1 ───────┤
                    └── t2b ── t3b
③ 验 (Verify)     LLM 一次前向，对整棵树所有路径并行打分
                  逐节点比对：SSM 猜的 token == LLM argmax? 
                  命中则采纳，第一个不命中处「剪断」，用 LLM 的真值接上
```

**为什么用「树」而不是「一条链」？** 单个 SSM 猜一条链，一旦中间猜错，后面全废。多个 SSM 各猜一条、或一个 SSM 给出 top-k 分叉，组成树后**命中概率显著提高**：只要任一条路径前缀对，就能采纳更长的前缀。原文「集体 boost-tuned 的小型投机模型」即指多 SSM 互补。

### 2.3 为什么「可证明保持模型质量」

验证阶段用的判据本质是**拒绝采样 (rejection sampling)**：对 SSM 给出的每个候选 token，按 LLM 与 SSM 的概率比决定接受/拒绝，被拒绝处用修正分布重采样。数学上可证明：**最终输出的 token 分布与「纯用 LLM 自回归采样」完全一致**。所以投机推理是**无损加速**——它只省时间，不改结果分布。这也是它区别于「直接用小模型」的根本点。

### 2.4 数值直觉：到底快多少

设 LLM 单次前向延迟 $T_L$，SSM 单次 $T_S \approx 0.1\,T_L$，每轮验证平均接受 $k$ 个 token。

生成 $N$ 个 token：
- 自回归：$N \cdot T_L$
- 投机：约 $\dfrac{N}{k}\big(T_L + d\cdot T_S\big)$，其中 $d$ 为 SSM 草拟步数。

若 $k=3$、$d=4$、$T_S=0.1T_L$：

$$\text{投机} \approx \frac{N}{3}\big(T_L + 4\times0.1T_L\big) = \frac{N}{3}\times 1.4\,T_L \approx 0.47\,N\,T_L$$

即**约 2 倍加速**。接受率 $k$ 越高（SSM 越接近 LLM、树越宽）加速越大；但树太大则 LLM 验证一次的算力也涨，存在 sweet spot。

## 3. CPU Offloading（单 GPU 跑大模型）

> 原文（保留）：FlexFlow Serve 还提供基于 Offloading 的推理，用于在单个 GPU 上运行大型模型（例如 llama-7B）。CPU Offloading 是将张量保存在 **CPU 内存**中，并且在计算时仅将张量复制到 GPU。现在我们**有选择地** offload 最大的权重张量（线性层、注意力中的权重张量）。此外，由于小模型占用空间少得多，若不构成 GPU 内存瓶颈，offload 会带来更多运行空间和计算成本，因此**只对大模型进行 offload**。[TODO：更新说明] 您可以通过启用 `-offload` 和 `-offload-reserve-space-size` 标志来运行 offloading 示例。

**原理图**：

```
            GPU 显存 (小，~14 GB)              CPU 内存 (大，几十~上百 GB)
        ┌───────────────────────┐         ┌───────────────────────────┐
        │ 激活值 + 当前层权重     │◄────────┤ 全部线性/注意力大权重张量  │
        │ KV cache               │  按需   │ (常驻 CPU)                 │
        └───────────────────────┘  拷贝    └───────────────────────────┘
              每层计算前: cudaMemcpy(layer_w) → 算完释放 → 下一层
```

**为什么只 offload 大模型、只 offload 最大张量？** offloading 用「PCIe 拷贝时间」换「显存空间」。小模型/小张量本来就不挤显存，offload 它们等于白白付出拷贝开销（PCIe 带宽 ~16–32 GB/s，远低于显存的几百 GB/s），得不偿失。所以策略是**只把真正撑爆显存的大权重搬到 CPU**。

| 标志 | 作用 |
| --- | --- |
| `-offload` | 启用 CPU offloading |
| `-offload-reserve-space-size` | 预留用于暂存 offload 张量的显存大小 |

## 4. 量化（int4 / int8）

> 原文（保留）：FlexFlow Serve 支持 **int4 和 int8** 量化。压缩后的张量存储在 **CPU 端**。一旦复制到 GPU，这些张量就会进行**解压缩并转换回其原始精度**。

```
CPU: 存 int4/int8 压缩权重  ──拷贝──►  GPU: 解压回 fp16/bf16  ──►  参与计算
     (省 CPU↔GPU 传输带宽 + 省 CPU 内存)        (计算仍用原始精度)
```

**为什么压缩存 CPU、计算前再解压？** 这套量化和 offloading 是搭档：权重压成 int4/int8 后，CPU 内存占用和 PCIe 传输量都减小（int4 仅 fp16 的 1/4），缓解了 §3 中 offloading 的拷贝瓶颈。但计算时解回原始精度，避免低比特矩阵乘的精度损失——属于**「传输/存储量化、计算不量化」**路线。

| 精度 | 每权重比特 | 相对 fp16 体积 | 主要收益 | 代价 |
| --- | --- | --- | --- | --- |
| fp16/bf16 | 16 | 1× | 基准 | 显存/带宽最大 |
| int8 | 8 | 1/2 | 传输/CPU 内存减半 | 解压开销 |
| int4 | 4 | 1/4 | 最省 | 解压开销 + 精度风险更高 |

> 与权重-only 静态量化（[[llm-compression/quantization/GPTQ]]、[[llm-compression/quantization/fp8]]）的差异：FlexFlow 这里是**存储侧动态压缩**，定位是配合 offloading 省传输，而非追求极致 kernel 加速。

## 5. 并行配置（tensor / pipeline / GPU 内存）

`ff.init(...)` 是整个服务的资源编排入口，四个数字必须配套：

```python
ff.init(
    num_gpus=4,                      # 用几张 GPU
    memory_per_gpu=14000,            # 每卡给 FlexFlow 的显存(MB) ≈ 14 GB
    zero_copy_memory_per_node=30000, # 每节点 zero-copy(pinned CPU)内存(MB) ≈ 30 GB
    tensor_parallelism_degree=4,     # 张量并行度: 把一层切到 4 卡
    pipeline_parallelism_degree=1    # 流水线并行度: 不按层切分阶段
)
```

**约束关系（为什么这么填）**：

$$\text{tensor\_parallelism\_degree} \times \text{pipeline\_parallelism\_degree} = \text{num\_gpus}$$

上例 $4\times1=4$ ✅。两种并行的分工：

```
张量并行 (TP=4): 同一层的大矩阵横切到 4 卡, 每步需 AllReduce 通信
      Layer i:  [W 的 1/4][1/4][1/4][1/4]  →  4 卡协同算一层
                 GPU0      1    2    3

流水线并行 (PP): 不同层放不同卡, 像流水线一段段传 (此处=1, 未启用)
```

- `memory_per_gpu`：留给权重 + KV cache 的预算，太小会 OOM，太大可能挤掉系统/其它进程。
- `zero_copy_memory_per_node`：pinned（锁页）CPU 内存，给 §3 offloading 和 host↔device 高速拷贝用；这是 offloading 大模型的物理后盾，所以配得比单卡显存还大（30 GB）。

> TP 通信依赖 [[ai-infra/网络/集合通信原语]]（AllReduce）与 [[ai-infra/网络/InfiniBand]]；并行思想可对照 [[ai-framework/megatron-lm/README]]、[[ai-framework/deepspeed/README]]。

---

## 实操：命令 / 配置 / 代码（原文真料）

### 安装与 Docker（原文保留）

```bash
pip install flexflow

# 直接交互式跑预构建 CUDA 12.0 镜像
docker run --gpus all -it --rm --shm-size=8g ghcr.io/flexflow/flexflow-cuda-12.0:latest

# 挂载缓存目录(复用下载的权重)再跑
docker run -it --gpus all --shm-size=8g \
  -v ~/.cache/flexflow:/usr/FlexFlow/inference \
  ghcr.io/flexflow/flexflow-cuda-12.0:latest
```

> `--shm-size=8g`：加大共享内存，多 GPU/多进程数据交换需要；不加常报 shared memory 不足。镜像命名规则见下「预构建包」。

### 预构建包命名规则（原文保留）

运行 FlexFlow 最快的方式是用预构建容器，**每次提交到 inference 分支都会更新**（该分支领先 master）：

- **flexflow**：预构建的可运行版本。
  - AMD GPU：ROCm 5.3 / 5.4 / 5.5 / 5.6 四个版本。
  - NVIDIA GPU：CUDA 11.1 / 11.2 / 11.3 / 11.4 / 11.5 / 11.6 / 11.7 / 11.8 / 12.0 多版本。
  - 命名：`flexflow-<GPU 后端>-<GPU 软件版本>`，如 `flexflow-hip_rocm-5.6`、`flexflow-cuda-12.0`。
- **flexflow-environment**：基础层（CI / 内部使用），含构建/运行所需全部依赖；自己编译 FlexFlow 时有用。命名同理，如 CUDA 12.0 的 `flexflow-environment-cuda-12.0`。

### 小投机模型（SSM）权重（原文保留）

```
https://huggingface.co/JackFram/llama-68m/tree/main
https://huggingface.co/JackFram/llama-160m/tree/main

# llama-160m 单文件直链
https://huggingface.co/JackFram/llama-160m/resolve/main/pytorch_model.bin
https://huggingface.co/JackFram/llama-160m/resolve/main/config.json
https://huggingface.co/JackFram/llama-160m/resolve/main/generation_config.json
https://huggingface.co/JackFram/llama-160m/resolve/main/tokenizer.json
https://huggingface.co/JackFram/llama-160m/resolve/main/tokenizer.model
https://huggingface.co/JackFram/llama-160m/resolve/main/tokenizer_config.json
```

> `JackFram/llama-68m`、`llama-160m` 是社区常用的**极小 LLaMA**，专门当 SSM 草拟器：参数量小、推理飞快，与 LLaMA-7B 词表/结构对齐，适合做投机。

### 提示数据集（评测用，原文保留）

FlexFlow 提供五个评估 FlexFlow Serve 的提示数据集：

| 数据集 | 链接 |
| --- | --- |
| Chatbot 指令提示 | `https://specinfer.s3.us-east-2.amazonaws.com/prompts/chatbot.json` |
| ChatGPT 提示 | `https://specinfer.s3.us-east-2.amazonaws.com/prompts/chatgpt.json` |
| WebQA | `https://specinfer.s3.us-east-2.amazonaws.com/prompts/webqa.json` |
| Alpaca | `https://specinfer.s3.us-east-2.amazonaws.com/prompts/alpaca.json` |
| PIQA | `https://specinfer.s3.us-east-2.amazonaws.com/prompts/piqa.json` |

### 完整 Python 推理示例（原文保留 + 逐行注解）

```python
import flexflow.serve as ff

# ① 初始化资源/并行 (见 §5)
ff.init(
    num_gpus=4,
    memory_per_gpu=14000,
    zero_copy_memory_per_node=30000,
    tensor_parallelism_degree=4,
    pipeline_parallelism_degree=1
)

# ② 指定大模型 (验证器 LLM)
llm = ff.LLM("decapoda-research/llama-7b-hf")

# ③ 指定一组 SSM (此例仅一个草拟器)
ssms = []
ssm = ff.SSM("JackFram/llama-68m")
ssms.append(ssm)

# ④ 采样配置: do_sample=False + topk=1 ⇒ 贪心解码
generation_config = ff.GenerationConfig(
    do_sample=False, temperature=0.9, topp=0.8, topk=1
)

# ⑤ 编译 SSM (载入权重到显存)
for ssm in ssms:
    ssm.compile(generation_config)

# ⑥ 编译 LLM, 并把 SSM 挂上 ⇒ 启用投机推理 token 树验证
llm.compile(generation_config, ssms=ssms)

# ⑦ 生成 (内部走 SSM 草拟 → LLM 树验证)
result = llm.generate("Here are some travel tips for Tokyo:\n")
# 返回类型: GenerationResult
```

> 关键点：把 `ssms=ssms` 传给 `llm.compile` 是「开启投机推理」的开关；不传则退化为普通自回归。`do_sample=False, topk=1` 表示**贪心**，此时验证判据就是逐 token 比 argmax 是否相等（§2.3 的特例）。原文中 ⑤⑥ 重复出现一次，是示例的强调写法，实际编译一次即可。

---

## 常见问题 / 坑

| 现象 / 问题 | 原因 | 处理 |
| --- | --- | --- |
| `tp_degree × pp_degree ≠ num_gpus` 报错 | 并行度乘积必须等于 GPU 数 | 改成满足 $TP\times PP=\text{num\_gpus}$（例 4×1=4） |
| 多 GPU 跑容器报 shared memory 不足 | 默认 `/dev/shm` 太小 | docker 加 `--shm-size=8g` |
| 投机推理「没变快」甚至更慢 | SSM 与 LLM 差异大、接受率 $k$ 低；或 token 树过大反而拖累 LLM 验证 | 选更贴合 LLM 的 SSM（同系小模型）、调小树宽/草拟步数 |
| 输出和纯 LLM 不一致，担心质量 | 误解 | 投机推理是**无损**的（§2.3 拒绝采样），分布与纯 LLM 自回归一致 |
| 单卡放不下 7B | 显存不够 | 启用 `-offload` + `-offload-reserve-space-size`（§3），必要时叠加 int4/int8 量化（§4） |
| offload 小模型反而变慢 | PCIe 拷贝开销 > 省下的显存收益 | **只对大模型/最大张量** offload（原文明确策略） |
| int4 精度掉点 | 4-bit 量化误差较大 | 退回 int8，或仅对不敏感层量化 |
| 不想本地编译，环境难搞 | 依赖复杂 | 直接用预构建镜像 `flexflow-cuda-12.0`；要自己编可用 `flexflow-environment-*` 基础层 |
| inference 分支与 master 行为不一致 | inference 分支领先 master | 以 inference 分支为准，镜像随该分支每次提交更新 |

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 推理总览与对比：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 加速机制基础：[[llm-optimizer/kv-cache]] · [[llm-optimizer/FlashAttention]]
- 模型结构：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 量化对照：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 并行/框架：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]]
- 硬件与网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/ai-hardware/硬件对比]]
- 评测与估算：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
