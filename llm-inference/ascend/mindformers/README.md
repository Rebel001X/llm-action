# 昇腾 MindFormers 推理

> 一句话定位：MindFormers 是华为基于 MindSpore + 昇腾 CANN 软件栈打造的「大模型全流程套件」，把训练好的 Transformer 大模型在昇腾 NPU（Ascend 910/310 系列）上跑起推理，是国产化算力替代 NVIDIA CUDA 生态的核心一环。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]]

## 阅读地图

| 章节 | 你将搞懂 | 关键词 |
| --- | --- | --- |
| 0 一句话锚点 | MindFormers 在昇腾推理里到底是什么角色 | 套件 / 推理入口 |
| 1 地基 | 为什么需要它、它解决什么痛点 | 国产化 / 生态割裂 |
| 2 昇腾软硬件全景 | 从硅片到框架的整条软件栈 | NPU / CANN / MindSpore |
| 3 MindFormers 架构 | 套件由哪些模块构成 | 模型库 / Trainer / Pipeline |
| 4 推理请求生命周期 | 一条 prompt 进去发生了什么 | 分词 / KV Cache / 增量推理 |
| 5 关键推理优化 | NPU 上怎么跑得快 | 静态图 / 量化 / 并行 |
| 6 模型迁移 | CUDA 模型怎么搬到昇腾 | 权重转换 / 算子适配 |
| 7 与 CUDA 生态对比 | 概念一一对应 | vLLM / TensorRT 类比 |
| 8 配置示例 | 每个参数啥含义 | yaml / batch / seq_len |
| 9 常见问题 | 踩坑速查 | 精度 / OOM / 算子缺失 |

---

## 0. 一句话锚点

**MindFormers ≈ 昇腾世界里的「HuggingFace Transformers + 推理引擎」二合一**。
它向上提供「拿来即用」的主流大模型（LLaMA、ChatGLM、Baichuan、Qwen、盘古等）实现与权重加载，向下把这些模型编译成昇腾 NPU 能执行的算子图，由 CANN 驱动硬件完成计算。一句话：**你写 Python 调用 MindFormers，它负责让大模型在国产 NPU 上正确且高效地完成前向推理。**

---

## 1. 地基：它解决什么问题

### 1.1 痛点：生态割裂

全球 90% 以上的大模型代码是围绕 **NVIDIA CUDA** 写的：PyTorch → CUDA Kernel → cuDNN/cuBLAS → GPU。一旦换成华为昇腾 NPU，这条链路上**每一层都不通用**：

- PyTorch 的 `.cuda()` 在昇腾上无意义；
- CUDA Kernel（`__global__` 函数）昇腾不认识；
- TensorRT / vLLM / FlashAttention 这些推理利器都是 CUDA 专属。

```
         CUDA 世界                       昇腾世界
   ┌───────────────────┐         ┌───────────────────┐
   │  PyTorch / vLLM   │   ✗ 不通 │  MindSpore        │
   │  CUDA Kernel      │ ──────► │  CANN 算子(AscendC)│
   │  cuDNN / cuBLAS   │         │  CANN 加速库      │
   │  NVIDIA GPU       │         │  昇腾 NPU(达芬奇)  │
   └───────────────────┘         └───────────────────┘
```

> 核心矛盾：**算法资产在 CUDA，国产化要求在昇腾**。中间需要一座「桥」——这正是 MindFormers 的定位。

### 1.2 为什么要国产化推理

- **供应链安全**：高端 GPU 受出口管制，昇腾是国产可控替代。
- **政企合规**：信创/国产化采购要求软硬件自主可控。
- **成本与产能**：国内昇腾算力集群规模化部署，需配套软件栈把利用率拉满。

### 1.3 MindFormers 给出的答案

把「模型实现、权重格式、并行策略、推理优化」全部沉淀成一个套件，让算法工程师**不必懂底层 NPU 编程**，改几行 yaml 就能在昇腾上推理。

---

## 2. 昇腾软硬件全景（自底向上）

要理解 MindFormers，必须先看清它脚下踩的整条栈。详见 [[ai-infra/算力/昇腾NPU]]。

```
  应用层      你的推理脚本 / 服务 (Python)
 ─────────────────────────────────────────────
  套件层  ┌────────────── MindFormers ──────────────┐
          │ 模型库 │ Trainer │ Pipeline │ 权重转换 │
          └──────────────────┬──────────────────────┘
 ─────────────────────────────────────────────
  框架层      MindSpore   (计算图 / 自动微分 / 分布式)
 ─────────────────────────────────────────────
  异构计算   ┌────────────── CANN ──────────────────┐
  架构层     │ 图编译GE │ 算子库 │ AscendCL │ HCCL  │
             └──────────────────┬─────────────────┘
 ─────────────────────────────────────────────
  驱动/硬件   Ascend Driver  →  昇腾 NPU (达芬奇 Cube/Vector)
```

| 层 | 类比 CUDA 生态 | 作用 |
| --- | --- | --- |
| MindFormers | HF Transformers + 推理引擎 | 模型实现 + 推理流程编排 |
| MindSpore | PyTorch / TensorFlow | 计算图、张量、分布式 |
| CANN | CUDA Toolkit + cuDNN + cuBLAS + NCCL | 算子编译、加速库、通信 |
| AscendCL | CUDA Runtime/Driver API | 设备管理、内存、流（Stream） |
| HCCL | NCCL | 多卡集合通信（AllReduce 等） |
| 昇腾 NPU | NVIDIA GPU | 计算硬件（Cube 矩阵单元为核心） |

> **达芬奇架构关键点**：NPU 的算力主要来自 **Cube 单元**（专做矩阵乘 $A\times B$），辅以 Vector / Scalar 单元。大模型推理 99% 的算力花在 GEMM（线性层、注意力打分）上，正好喂饱 Cube。

---

## 3. MindFormers 套件架构

MindFormers 内部可拆成几个稳定的核心模块（具体目录以官方源码为准）：

```
            ┌──────────────────────────────────────────┐
  用户调用 → │  run_mindformer.py / Trainer / pipeline API │
            └───────────────┬──────────────────────────┘
        ┌───────────────────┼───────────────────────────┐
        ▼                   ▼                            ▼
 ┌────────────┐     ┌──────────────┐            ┌──────────────┐
 │  模型库     │     │  配置系统     │            │  Tokenizer   │
 │ (models/)  │     │  (yaml config)│            │  分词器       │
 │ LLaMA/GLM/ │     │ 模型超参 +    │            │ BPE/SentP.   │
 │ Qwen/盘古  │     │ 并行 + 推理参 │            └──────────────┘
 └─────┬──────┘     └──────┬───────┘
       │                   │
       ▼                   ▼
 ┌──────────────────────────────────────────────┐
 │            执行层 (基于 MindSpore)             │
 │  - 静态图编译 (GRAPH_MODE)                     │
 │  - 增量推理 / KV Cache 管理                    │
 │  - 分布式并行 (TP / PP) + HCCL 通信            │
 └──────────────────────────────────────────────┘
```

核心模块职责：

1. **模型库（model zoo）**：预置主流大模型的 MindSpore 实现 + 适配好的权重。这是「拿来即用」的来源（如本文末尾的 ChatGLM3、Baichuan2 卡片）。
2. **配置系统（yaml）**：一切行为由声明式 yaml 驱动——模型结构、并行切分、推理长度、量化开关。改 yaml 不改代码。
3. **Trainer / Pipeline API**：高层封装。推理常用 `pipeline("text_generation")` 或 `Trainer.predict()`，一行触发完整生成。
4. **权重转换工具**：把 HuggingFace `.bin/.safetensors` 转成 MindSpore `.ckpt`，并处理张量命名/切分（见第 6 节）。
5. **执行层**：交给 MindSpore 编译成静态图，再由 CANN 落到 NPU。

---

## 4. 推理请求生命周期：一条 prompt 的旅程

这是理解一切的主线。假设用户输入 `"请介绍昇腾"`，要生成 50 个 token。

```
[1] 文本 "请介绍昇腾"
        │  Tokenizer 分词
        ▼
[2] input_ids = [101, 3456, ...]   (shape: [1, L])
        │  搬到 NPU (host→device, AscendCL memcpy)
        ▼
[3] 首次前向 (Prefill / 全量阶段)
    ─ Embedding → N×TransformerBlock → LMHead
    ─ 每个 Block: Attention(Q,K,V) + FFN
    ─ 把所有层的 K,V 写入 KV Cache
        │
        ▼
[4] 采样得到第 1 个新 token (argmax / top-p)
        │
        ▼
[5] 增量前向 (Decode / 增量阶段) —— 循环
    ┌──────────────────────────────────────┐
    │ 只输入「上一个 token」(shape [1,1])     │
    │ 复用 KV Cache，只算新 token 的 Q       │
    │ Attention 用 新Q × 历史K,V             │
    │ 采样 → 追加到 KV Cache → 下一轮         │
    └──────────────────────────────────────┘
        │  重复 49 次，或遇到 EOS 停止
        ▼
[6] token_ids → Tokenizer 解码 → "昇腾是华为推出的..."
```

### 4.1 为什么要分 Prefill / Decode 两阶段

- **Prefill（全量）**：一次性处理整个 prompt（长度 $L$），算力打满，是 **compute-bound**。复杂度约 $O(L^2 d)$（注意力的 $QK^T$）。
- **Decode（增量）**：每次只生成 1 个 token，但要反复读取庞大的 KV Cache 和权重，是 **memory-bound**（受显存带宽限制）。

> **为什么要 KV Cache**：若不缓存，生成第 $t$ 个 token 时要重算前 $t-1$ 个 token 的 K、V，总复杂度退化到 $O(T^2)$。缓存后每步只需 $O(t)$，把生成 $T$ 个 token 从 $O(T^3)$ 级降到 $O(T^2)$。代价是显存：KV Cache 大小 $\approx 2 \times L_{seq} \times n_{layer} \times n_{head} \times d_{head} \times \text{batch} \times \text{bytes}$。

### 4.2 数值例子

LLaMA-7B：$n_{layer}=32$，hidden $=4096$，序列 $L=2048$，batch $=1$，FP16（2 字节）：

$$
\text{KV Cache} = 2 \times 2048 \times 32 \times 4096 \times 1 \times 2\ \text{B} \approx 1.07\ \text{GB}
$$

batch 调到 16，KV Cache 就涨到约 **17 GB**——这就是为什么长上下文 + 大 batch 极易在 NPU 上 OOM，也是配置 `seq_length` 和 `batch_size` 要权衡的根本原因。

---

## 5. 关键推理优化（NPU 上怎么跑得快）

### 5.1 静态图 vs 动态图

MindSpore 有两种模式，推理几乎都用静态图：

```
动态图 PYNATIVE_MODE         静态图 GRAPH_MODE
  逐算子下发，灵活好调试         先把整张图编译，再整体下发
  每步都有 host→device 开销     算子融合 + 一次编译多次执行
  ↓ 适合调试                   ↓ 适合部署，吞吐高
```

| 维度 | 动态图 | 静态图（推荐部署） |
| --- | --- | --- |
| 首次延迟 | 低 | 高（要编译，首 token 慢） |
| 稳态吞吐 | 低 | 高（算子融合、无 Python 开销） |
| 调试体验 | 好（可断点） | 差（图已固化） |

> 权衡：部署用静态图换吞吐，但首次推理有「编译预热」，所以服务要先跑一次 warmup。

### 5.2 量化（Quantization）

把权重从 FP16 压到 INT8/INT4，显存与带宽双降。在 memory-bound 的 Decode 阶段收益尤其大：

$$
\text{显存占用比} = \frac{\text{INT8: 1 字节}}{\text{FP16: 2 字节}} = \frac{1}{2}
$$

代价是精度损失，需做 per-channel scale 校准。MindFormers / CANN 提供 W8A16、W8A8 等方案（具体支持以官方文档为准）。

### 5.3 分布式并行

单卡放不下大模型时，按以下方式切分，跨卡通信走 **HCCL**（对标 NCCL）：

```
张量并行 TP（层内切）        流水并行 PP（层间切）
 ┌─────┐ ┌─────┐            卡0: Layer 0~7
 │卡0  │ │卡1  │            卡1: Layer 8~15
 │W左半│ │W右半│            卡2: Layer 16~23
 └──┬──┘ └──┬──┘            卡3: Layer 24~31
    └─AllReduce─┘           token 像流水线逐段流过
```

- **TP**：每层权重切成两半放不同卡，每步要 AllReduce 同步，通信频繁但延迟低，适合卡间高速互联（HCCS）。
- **PP**：不同层放不同卡，通信少但有「气泡」空转，适合跨节点。

---

## 6. 模型迁移：从 CUDA/HF 搬到昇腾

这是国产化落地最实际的一步。把一个 HuggingFace 模型迁到 MindFormers 大致三步：

```
┌─────────────────────────────────────────────────────────┐
│ 步骤1: 权重转换                                            │
│   HF (.bin/.safetensors, PyTorch 命名)                    │
│        │  convert_weight 脚本                              │
│        ▼  - 张量改名 (q_proj → wq ...)                     │
│   MindSpore (.ckpt)                                       │
│        - 维度/转置对齐 (PyTorch [out,in] vs MS 约定)       │
├─────────────────────────────────────────────────────────┤
│ 步骤2: 模型结构对齐                                        │
│   - 用 MindFormers 已实现的同名模型 (LLaMA/GLM...)         │
│   - 核对 RoPE / RMSNorm / 激活函数 与原版一致              │
│   - 若有自定义算子 → 用 AscendC 写或找等价算子              │
├─────────────────────────────────────────────────────────┤
│ 步骤3: 精度对齐验证                                        │
│   - 同一输入，比 logits 与原模型的相对误差                 │
│   - 阈值通常看 cosine 相似度 / 最大绝对误差                │
└─────────────────────────────────────────────────────────┘
```

迁移中最容易踩的坑：

| 问题 | 原因 | 处理思路 |
| --- | --- | --- |
| 权重名对不上 | HF 与 MS 命名不同 | 维护映射表逐一改名 |
| 结果全错/转置 | 线性层权重存储约定不同 | 转换时按需 `.T` |
| 算子不支持 | CANN 暂无该算子 | 用 AscendC 自定义或算子替换 |
| 精度有差异 | 累加顺序 / FP16 舍入 | 关键层用 FP32，逐层比对定位 |

> 提示：能否直接用「已迁好的模型卡片」（如文末 ChatGLM3、Baichuan2）是最省事的路径——优先复用，避免从零迁移。

---

## 7. 与 CUDA 生态的概念对比

详见 [[ai-infra/ai-hardware/AI芯片软件生态]]。把脑子里的 CUDA 知识一一映射过来即可：

| 任务 | CUDA 生态 | 昇腾生态 |
| --- | --- | --- |
| 深度学习框架 | PyTorch | MindSpore（或 PyTorch+torch_npu） |
| 大模型库 | HuggingFace Transformers | **MindFormers** |
| 推理引擎 | vLLM / TensorRT-LLM | MindFormers / MindIE |
| 底层工具包 | CUDA Toolkit | CANN |
| 算子加速库 | cuDNN / cuBLAS | CANN 算子库 + AscendC |
| 多卡通信 | NCCL | HCCL |
| 运行时 API | CUDA Runtime | AscendCL（ACL） |
| 自定义 Kernel | CUDA C++ (`__global__`) | AscendC |
| 显存概念 | GPU Memory | NPU Device Memory（HBM） |

> **认知锚点**：会 vLLM + HF Transformers，就能快速理解 MindFormers——把「写 PyTorch、调 vLLM」的心智，平移成「写 MindSpore、调 MindFormers」，主要差异在算子层和权重格式。

```
   你的已有知识              对应的昇腾技能
  HF from_pretrained  ──►  MindFormers AutoModel.from_pretrained
  vLLM LLM.generate   ──►  pipeline("text_generation")(prompt)
  CUDA OOM 调 batch    ──►  yaml 调 batch_size / seq_length
  NCCL AllReduce      ──►  HCCL AllReduce（写法相近）
```

---

## 8. 典型推理配置示例（讲含义，非可执行保证）

MindFormers 用 yaml 声明式驱动推理。下面是一个**示意性**配置，字段含义为重点，精确字段名以官方 model card 为准：

```yaml
# ===== 运行模式 =====
context:
  mode: 0                 # 0=GRAPH_MODE 静态图(部署); 1=PYNATIVE 动态图(调试)
  device_target: "Ascend" # 目标设备：昇腾 NPU
  device_id: 0            # 用哪张卡

# ===== 模型结构 =====
model:
  model_config:
    seq_length: 2048      # 最大序列长度 → 直接决定 KV Cache 大小与显存
    hidden_size: 4096     # 隐藏维度
    num_layers: 32        # Transformer 层数
    num_heads: 32         # 注意力头数
    vocab_size: 32000     # 词表大小
    use_past: True        # ★关键：开启 KV Cache 增量推理(必开，否则慢几十倍)
    compute_dtype: "float16"  # 计算精度，FP16 省显存

# ===== 推理生成参数 =====
generation:
  max_decode_length: 512  # 最多生成多少 token
  do_sample: True         # True=采样(更多样); False=贪心(更确定)
  top_k: 50               # 只从概率最高的 50 个里采样
  top_p: 0.9              # 核采样：累积概率 0.9 的最小集合
  temperature: 0.8        # <1 更保守，>1 更发散

# ===== 并行(单卡可省略) =====
parallel_config:
  model_parallel: 1       # 张量并行卡数；放不下时调大(需配 HCCL)
  pipeline_stage: 1       # 流水并行段数
```

### 关键参数权衡速记

| 参数 | 调大的好处 | 调大的代价 |
| --- | --- | --- |
| `seq_length` | 支持更长上下文 | KV Cache 平方/线性增长，易 OOM |
| `batch_size` | 吞吐高 | 显存线性上涨，首 token 延迟升 |
| `use_past` | 增量推理快几十倍 | 占 KV Cache 显存（几乎必开） |
| `model_parallel` | 能放下更大模型 | 引入 AllReduce 通信开销 |
| `compute_dtype=fp16` | 省一半显存与带宽 | 精度略降 |

> 调优心法：Decode 是 memory-bound，**先想办法降显存/带宽压力**（量化、降 batch、`use_past`），再考虑算力。

---

## 9. 常见问题

| 问题 | 现象 | 排查/解决思路 |
| --- | --- | --- |
| 首 token 特别慢 | 第一次推理卡几十秒 | 静态图编译预热，加 warmup 请求；之后稳态正常 |
| OOM（显存不足） | Device memory 报错 | 降 `batch_size`/`seq_length`、开量化、上 `model_parallel` |
| 精度对不上原模型 | 输出乱码/逻辑错 | 权重转换转置错或层未对齐，逐层比 logits 定位 |
| 算子不支持 | 报某 op 未注册 | CANN 版本太旧或缺算子，升级 CANN 或 AscendC 自定义 |
| 多卡跑不起来 | HCCL 初始化失败 | 检查 rank table / 卡间互联，确认 HCCL 环境变量 |
| CANN 与 MindSpore 版本冲突 | 加载即报错 | 严格按官方版本配套表对齐三者版本（以官方文档为准） |
| 生成不停/重复 | 不输出 EOS | 检查 `max_decode_length`、采样参数、EOS token id 配置 |

> 通用排错链路：**先确认软件栈版本配套（CANN↔MindSpore↔MindFormers）→ 再验证权重转换正确性 → 最后调推理性能参数**。版本不配套是国产化部署最常见的「玄学」根因。

---

## 🔗 跳转链接

- [[00-知识地图]] — 全站导航总入口
- [[ai-infra/算力/昇腾NPU]] — 昇腾 NPU 硬件与达芬奇架构细节
- [[ai-infra/ai-hardware/AI芯片软件生态]] — CUDA vs 昇腾 vs 其他国产芯片软件栈横评

## 参考文档

- chatglm3: https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/glm3.md
- baichuan2: https://gitee.com/mindspore/mindformers/blob/dev/research/baichuan2/baichuan2.md
