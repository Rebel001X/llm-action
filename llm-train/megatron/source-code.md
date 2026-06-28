# Megatron-LM 源码精读：从 pretrain_gpt 到张量并行算子

> 一句话定位：顺着 `pretrain_gpt.py` 的一次调用，把 Megatron-LM 训练主干（初始化 → 并行分组 → 建模 → 数据 → 训练循环）和最关键的张量并行算子（Column/Row Parallel、cross_entropy、4 个通信原语）拆成原子，看懂"3D 并行"在代码里到底长什么样。📍 导航：[[00-知识地图]]
>
> 🔗 相关：[[llm-train/megatron/README]] · [[ai-framework/megatron-lm/README]] · [[B07:llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]]

---

## 阅读地图

| 节 | 你会看懂 | Megatron 真实符号 |
|----|----------|-------------------|
| 0 | 一句话锚点：源码主干长什么样 | `pretrain()` |
| 1 | 前置：进程/rank/group/3D 并行术语 | `mpu` / `parallel_state` |
| 2 | 入口：`pretrain_gpt.py` 的三个 provider | `model_provider` / `train_valid_test_datasets_provider` |
| 3 | 训练主流程 `pretrain()` 的 4 步 | `megatron.training.pretrain` |
| 4 | 初始化 + 分布式环境 | `initialize_megatron` / `_initialize_distributed` |
| 5 | **并行分组**：16 GPU 切 TP×PP×DP | `initialize_model_parallel` |
| 6 | 模型搭建：GPTModel → Transformer 层 | `gpt_model` / `transformer` |
| 7 | **张量并行算子**：Column/Row Parallel | `tensor_parallel.layers` |
| 8 | 词表并行交叉熵 | `_VocabParallelCrossEntropy` |
| 9 | **4 个通信原语**：f / g / scatter / gather | `tensor_parallel.mappings` |
| 10 | 数据侧：dataset → dataloader → iterator | `data.gpt_dataset` / `data_samplers` |
| 实操 | 真实 import 路径与调用链速查 | —— |
| 坑 | 常见踩坑表 | —— |

---

## 0. 一句话锚点

Megatron-LM 的训练只有一条主干：

```
pretrain_gpt.py  →  megatron.training.pretrain(...)  →  初始化 → 建模 → 数据 → train()
```

`pretrain()` 的官方 docstring（原文真料，保留）说明了它干 4 件事：

```
Main training program.
This function will run the followings in the order provided:
    1) initialize Megatron.
    2) setup model, optimizer and lr schedule using the model_provider.
    3) call train_val_test_data_provider to get train/val/test datasets.
    4) train the model using the forward_step_func.
```

记住这 4 步，整份源码笔记就是把这 4 步逐层展开。

---

## 1. 地基 / 前置：4 个必须先分清的词

读 Megatron 源码，90% 的困惑来自不分清下面 4 个词。

| 术语 | 含义 | 代码里在哪 |
|------|------|-----------|
| **rank** | 一个进程的全局编号（0..world_size-1），通常 1 进程 = 1 GPU | `torch.distributed.get_rank()` |
| **world_size** | 总进程数 = 总 GPU 数 | `init_process_group` |
| **group（进程组）** | 一组 rank，集合通信（all-reduce/all-gather）只在组内发生 | `torch.distributed.new_group` |
| **mpu** | model-parallel-unit，即 `megatron.core.parallel_state`，管理所有并行子组 | `mpu.initialize_model_parallel` |

**3D 并行**就是把这堆 GPU 沿三个正交维度切：

```
        TP（张量并行，Tensor Parallel）
        把"一层"的权重矩阵切片，组内 all-reduce 拼回 → 通信最频繁，放同机内 NVLink
   ────────────────────────────────────────────────►
   │
PP │   PP（流水线并行，Pipeline Parallel）
（流│   把"不同层"分到不同 GPU，激活在 stage 间点对点传递 → 通信少，可跨机
水 │
线 │   DP（数据并行，Data Parallel）
）▼   每个完整模型副本吃不同 batch，梯度 all-reduce → 标准数据并行
```

为什么要分这么细？因为**集合通信的代价**完全不同：TP 每个 micro-step 都要 all-reduce（最贵），PP 只在层边界传激活（点对点，便宜），DP 只在反向结束 all-reduce 梯度（一次）。Megatron 的分组策略就是为了把"贵的通信"塞进同一台机器的 NVLink 内。

---

## 2. 入口：`pretrain_gpt.py` 的三个 provider

`pretrain_gpt.py` 本身很薄，它把"如何建模/如何取数据"以**回调函数**形式注入 `pretrain()`。这是 Megatron 的核心设计：**训练框架与模型解耦**。

```
pretrain_gpt.py
├── model_provider(pre_process, post_process)
│       └─► 返回一个 GPTModel 实例（见 §6）
├── train_valid_test_datasets_provider(train_val_test_num_samples)
│       └─► 调 build_train_valid_test_datasets 造数据集（见 §10）
└── data_post_process(data, ...)
        └─► 取完一个 batch 后的后处理钩子
```

| provider | 作用 | 对应 pretrain 的第几步 |
|----------|------|----------------------|
| `model_provider` | 怎么搭模型 | 步骤 2 |
| `train_valid_test_datasets_provider` | 怎么造数据集 | 步骤 3 |
| `data_post_process` | 数据后处理 | 训练循环内 |

**为什么是回调而不是直接写？** 因为 `pretrain()` 要被 GPT / BERT / T5 等多种模型复用。把"模型差异"封进 provider，主流程就完全通用了。

---

## 3. 训练主流程 `megatron.training.pretrain`

```python
from megatron.training import pretrain
from megatron.data.gpt_dataset import build_train_valid_test_datasets
```

`pretrain()` 内部按官方 docstring 顺序串起来：

```
pretrain()
 │
 ├─(1) initialize_megatron()                    # §4：初始化 Megatron + 分布式
 │
 ├─(2) setup_model_and_optimizer(model_provider)# 建模型/优化器/lr 调度
 │        └─ get_model()                         # 真正实例化 GPTModel
 │
 ├─(3) build_train_valid_test_data_iterators()  # §10：数据集 → 迭代器
 │
 └─(4) train(forward_step_func, model, ...)     # 训练循环
```

原文对应的真实符号（保留并解释）：

| 符号 | 原文注释 | 解释 |
|------|---------|------|
| `setup_model_and_optimizer` | "Model, optimizer, and learning rate. 设置模型及优化器" | 组装模型副本、包装 DDP/优化器、建 lr scheduler |
| `get_model` | "构建模型" | 实际 new 出 GPTModel；按 PP stage 决定本 rank 持有哪些层 |
| `build_train_valid_test_data_iterators` | "构建预训练数据迭代器" | 把 dataset 包成可 `next()` 的 iterator |
| `train` | "训练" | 主循环：forward → backward → optimizer.step → log |

**关键点**：`get_model` 不是每个 rank 都建完整模型。在 PP 下，rank 只建属于自己 stage 的那几层（`pre_process`/`post_process` 标志位决定它是否含 embedding / 输出头）。

---

## 4. 初始化：`initialize_megatron` 与 `_initialize_distributed`

```python
from megatron.initialize import initialize_megatron
```

`_initialize_distributed` 做三件事（原文真料，保留并补原理）：

```
1. 设置分布式环境：初始化进程，分配 GPU，并设置进程大组（group）
2. 制定 DP/TP/PP 分组策略，设置进程子组（subgroup）
3. 设置 DeepSpeed ZeRO-R，对 activation 进行优化
```

落到两个真实 API：

| 步骤 | 调用 | 干什么 |
|------|------|--------|
| 建"大组" | `torch.distributed.init_process_group` | 拉起 NCCL 后端，world_size 个 rank 全连通 |
| 建"子组" | `mpu.initialize_model_parallel` | 把大组切成 TP/PP/DP 三套子组（见 §5）|

```
init_process_group  ──►  一张大网（world_size 个 rank 全互通）
                              │
initialize_model_parallel ──► 切成三套正交子组：
                              ├─ tensor_model_parallel_group
                              ├─ pipeline_model_parallel_group
                              └─ data_parallel_group
```

**为什么先建大组再切子组？** NCCL 通信器（communicator）必须显式构造；子组是大组的子集，集合通信调用时传入对应 group，NCCL 就只在该组内发消息。没有子组，all-reduce 就会在全部 GPU 上做（错的，且慢）。

---

## 5. ⭐ 并行分组：`initialize_model_parallel`（全篇最重要）

`megatron.core.parallel_state.initialize_model_parallel` 是整份源码的"心脏"。原文给了一个**完整数值示例**，这里原样保留并配图详解。

> **原文示例**：假设我们总共有 16 个 GPU，用 g0 … g15 表示，使用 2 个 GPU 张量并行（TP=2），4 个 GPU 流水线并行（PP=4）。当前函数将创建 **8 个张量模型并行组、4 个流水线模型并行组和 8 个数据并行组**：
>
> - **8 个数据并行组**：[g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
> - **8 个张量模型并行组**：[g0,g1], [g2,g3], [g4,g5], [g6,g7], [g8,g9], [g10,g11], [g12,g13], [g14,g15]
> - **4 个流水线模型并行组**：[g0,g4,g8,g12], [g1,g5,g9,g13], [g2,g6,g10,g14], [g3,g7,g11,g15]

### 5.1 数值先对账

设 world_size $N=16$，TP=2，PP=4，则

$$
\text{DP} = \frac{N}{\text{TP}\times\text{PP}} = \frac{16}{2\times 4} = 2
$$

每种分组的"组数"= 总数 / 该并行度：

| 并行类型 | 并行度（每组大小） | 组数 = 16 / 大小 |
|----------|------------------|------------------|
| 张量并行 TP | 2 | 8 组 ✅ |
| 流水线并行 PP | 4 | 4 组 ✅ |
| 数据并行 DP | 2 | 8 组 ✅ |

和原文给的"8 / 4 / 8"完全对得上。

### 5.2 用一张 4×4 表看清切法

把 16 个 GPU 按 (PP stage 行, TP×DP 列) 摆成 4 行 × 4 列。**Megatron 的编号规则：TP 维变化最快，其次 DP，最后 PP**：

```
                 ── TP 相邻（最快变） ──►
              col0        col1        col2        col3
            ┌──────────┬──────────┬──────────┬──────────┐
 PP0  行0   │   g0     │   g1     │   g2     │   g3     │
            ├──────────┼──────────┼──────────┼──────────┤
 PP1  行1   │   g4     │   g5     │   g6     │   g7     │
            ├──────────┼──────────┼──────────┼──────────┤
 PP2  行2   │   g8     │   g9     │   g10    │   g11    │
            ├──────────┼──────────┼──────────┼──────────┤
 PP3  行3   │   g12    │   g13    │   g14    │   g15    │
            └──────────┴──────────┴──────────┴──────────┘

  TP 组 = 同一行内"相邻一对"：  [g0,g1] [g2,g3] ...   （横向、最近邻 → 放 NVLink）
  DP 组 = 同一行内"隔一个"：    [g0,g2] [g1,g3] ...   （同 stage，副本之间）
  PP 组 = 同一列、跨所有行：    [g0,g4,g8,g12] ...     （纵向、跨机也行）
```

逐条验证原文：
- TP 取相邻对：[g0,g1],[g2,g3],[g4,g5]… ✅（共 8 组）
- PP 取整列：col0→[g0,g4,g8,g12]，col1→[g1,g5,g9,g13]… ✅（共 4 组）
- DP 取同行隔位：行0 的 {g0,g2} 与 {g1,g3}，行1 的 {g4,g6} 与 {g5,g7}… ✅（共 8 组）

### 5.3 为什么这样编号？（设计意图）

| 维度 | 通信频率 | 放置策略 | 在表中的形态 |
|------|---------|---------|-------------|
| TP | 最高（每层 fwd/bwd 都 all-reduce） | **同机内**，吃 NVLink | 行内相邻 → rank 号最接近 |
| DP | 中（每 step 末梯度 all-reduce 一次） | 可同机可跨机 | 同行隔位 |
| PP | 最低（仅传 stage 间激活，P2P） | 可跨机 | 整列、rank 号跨度最大 |

**核心思想**：rank 号越接近 → 物理越靠近（同 NVSwitch）。所以让"最贵的 TP"占用相邻 rank，把昂贵的 all-reduce 关在一台机器里；"最便宜的 PP"用跨度最大的 rank（跨机也无所谓）。这就是 Megatron 分组顺序"TP 最快、PP 最慢变化"的全部理由。

---

## 6. 模型搭建：GPTModel → Transformer 层

```
megatron.model.gpt_model.GPTModel
        │
        └── TransformerLanguageModel  (megatron.model.language_model)
                ├── Embedding                       # 词表 + 位置嵌入
                └── ParallelTransformer             # N 个 transformer 层堆叠
                        └── ParallelTransformerLayer (× N)
                                ├── ParallelAttention   # 并行自注意力
                                └── ParallelMLP         # 并行 MLP
```

原文保留的真实注释：

| 类 | 原文 | 补充 |
|----|------|------|
| `MegatronModule` | "Megatron 针对 torch.nn.Module 的特定扩展以支持流水线" | 加了 PP 所需的权重共享/`shared_embedding` 逻辑 |
| `Float16Module` | —— | fp16/bf16 包装：输入转半精、输出转回 fp32 |
| `ParallelTransformerLayer` | "Transformer 层接受大小为 **[s, b, h]** 的输入并返回相同大小的输出" | s=seq_len, b=batch, h=hidden；TP 不改变这个张量形状（切的是内部权重） |
| `ParallelAttention` | "并行自注意力层抽象类" | QKV 用 Column-Parallel，输出投影用 Row-Parallel |
| `ParallelMLP` | "并行 MLP 层" | 第一个 Linear Column-Parallel，第二个 Row-Parallel |

**关键洞察**：`[s,b,h]` 进、`[s,b,h]` 出——TP 对外是"透明"的。一层内部把权重切成两半算，靠 §9 的通信原语在层边界把结果拼/规约回完整的 `[s,b,h]`，所以上层代码感知不到并行的存在。

---

## 7. ⭐ 张量并行算子：Column / Row Parallel Linear

`megatron.core.tensor_parallel.layers` 是 TP 的"肌肉"。一个 MLP `Y = GeLU(XA)`、`Z = YB` 怎么切？

### 7.1 ColumnParallelLinear（按列切权重 A）

把权重 $A$ 沿**列**切成 $[A_1, A_2]$，每个 TP rank 只存一半：

```
   X (完整, 每个 rank 都有)
    │  广播给两个 rank（前向 f = identity）
    ├──────────────┐
  rank0          rank1
  Y1 = X·A1     Y2 = X·A2        ← 各算一半，无需通信
    │              │
  （Y1,Y2 拼起来就是完整 Y，但此处不拼，直接喂给 Row 层）
```

数值例：hidden $h=4$，TP=2。$A$ 是 $4\times 8$，切成两个 $4\times 4$。rank0 算出 $Y$ 的前 4 列，rank1 算后 4 列。**前向无通信**——这是 Column 层的好处。

### 7.2 RowParallelLinear（按行切权重 B）

紧接 Column 层，把 $B$ 沿**行**切 $[B_1; B_2]$，输入正好是 Column 层切好的 $[Y_1, Y_2]$：

```
  rank0          rank1
  Y1             Y2
   │  Z1=Y1·B1    │  Z2=Y2·B2     ← 各算部分和
   └──────┬───────┘
       all-reduce (前向 g)         ← Z = Z1 + Z2，此处必须通信！
          │
      完整 Z (每个 rank 都有)
```

**Column→Row 配对**是 Megatron 的精髓：一个 MLP / 一个 Attention 块，**整个前向只需 1 次 all-reduce**（在 Row 层末尾），反向同样只 1 次。把通信压到最低。

| 层 | 切法 | 前向通信 | 反向通信 |
|----|------|---------|---------|
| ColumnParallelLinear | 权重按列切 | 无（f=identity） | all-reduce（g） |
| RowParallelLinear | 权重按行切 | all-reduce（g） | 无（f=identity） |

辅助：`_initialize_affine_weight_gpu`（原文："初始化 GPU 上模型并行的仿射权重"）——保证切片后各 rank 的权重初始化与"未切分的完整权重"在统计上一致（同一随机种子切片）。

---

## 8. 词表并行交叉熵：`_VocabParallelCrossEntropy`

`megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy`（原文："计算交叉熵"）。

为什么交叉熵要专门写并行版？因为词表 $V$ 很大（如 5 万+），输出 logits `[s, b, V]` 沿 $V$ 维被 TP 切了（见 `VocabParallelEmbedding`），每个 rank 只有部分词表的 logits。直接算 softmax 会错——分母（归一化项）需要全词表。

```
loss = -log softmax(logit_correct)
     = -logit_correct + log Σ_v exp(logit_v)
                                 ▲
                  这个 Σ 跨整个词表 → 需要 all-reduce
```

所以并行交叉熵流程：
1. 各 rank 本地求 `max`（数值稳定）→ all-reduce(max)
2. 各 rank 本地求 `Σexp` → all-reduce(sum) 得全词表分母
3. 正确类的 logit 在哪个 rank → all-reduce 取出
4. 拼出 loss

公式（数值稳定版，TP 下 $\max$ 与 $\sum$ 都要跨组归约）：

$$
\text{loss} = \log\!\Big(\textstyle\sum_{v=1}^{V} e^{z_v - z_{\max}}\Big) - (z_{y} - z_{\max})
$$

---

## 9. ⭐ 4 个通信原语：`tensor_parallel.mappings`

这是把 §7 的"切片计算"缝合起来的胶水。`megatron.core.tensor_parallel.mappings` 定义了 4 个 `autograd.Function`，每个都有**前向**和**反向**两种行为（互为对偶），对应 Megatron 论文里的 `f` 和 `g` 算子。

| 原语（类） | 原文注释 | 前向 | 反向 |
|-----------|---------|------|------|
| `_CopyToModelParallelRegion`（`f`） | "将输入传递到模型并行区域" | identity（拷贝） | all-reduce |
| `_ReduceFromModelParallelRegion`（`g`） | "All-reduce 来自模型并行区域的输入" | all-reduce | identity |
| `_ScatterToModelParallelRegion` | "分割输入并仅将相应的 chunk 保留到 rank 中" | split（切） | all-gather |
| `_GatherFromModelParallelRegion` | "从模型并行区域收集输入拼接在一起" | all-gather（拼） | split |

封装的友好接口：`copy_to_tensor_model_parallel_region` / `gather_from_tensor_model_parallel_region` 等。

```
   f (CopyToRegion)              g (ReduceFromRegion)
 前向: 复制 X 给各 rank        前向: all-reduce 求和
 反向: all-reduce 梯度         反向: 复制梯度
       ▲                              ▲
   Column 层入口用 f          Row 层出口用 g
   （前向不通信，反向规约）   （前向规约，反向不通信）
```

**为什么 f 和 g 前反向相反？** 因为前向 split/copy 的对偶（反向）天然是 gather/reduce。Megatron 把通信封进 `autograd.Function` 的 `backward`，于是写模型时只管 `f(x)`、`g(y)`，PyTorch 自动在反向插入对应通信——开发者不用手写一行 `all_reduce`。

`_ScatterToModelParallelRegion`（前向切、反向拼）/ `_GatherFromModelParallelRegion`（前向拼、反向切）用于"序列并行"等需要重切分张量的场景。

---

## 10. 数据侧：dataset → dataloader → iterator

```python
from megatron.data.gpt_dataset import build_train_valid_test_datasets
```

```
train_valid_test_datasets_provider          (pretrain_gpt.py)
        │  调用
        ▼
build_train_valid_test_datasets             (megatron.data.gpt_dataset)
        │  "构建预训练数据集"，返回 train/valid/test 三个 Dataset
        ▼
build_pretraining_data_loader(dataset, consumed_samples)   (megatron.data.data_samplers)
        │  "给定一个数据集构建数据加载器"
        │  consumed_samples → 断点续训：跳过已消费样本
        ▼
build_train_valid_test_data_iterators       (megatron.training)
        │  包成可 next() 的迭代器，喂给 train() 的 forward_step_func
        ▼
       train()
```

`consumed_samples` 是关键参数：它记录"已经吃掉多少样本"，让 dataloader 从断点处继续采样——这是大模型训练**断点续训**不重复数据的基础。

---

## 实操：真实 import 路径 + 调用链速查（原文真料汇总）

> 下表完整保留原文出现过的所有真实模块/类/函数与 import，按调用顺序排列，可当"跳转目录"用。

```python
# ── 入口 ───────────────────────────────────────────────
# pretrain_gpt.py
#   model_provider / train_valid_test_datasets_provider / data_post_process

# ── 训练主流程 ─────────────────────────────────────────
from megatron.training import pretrain
from megatron.data.gpt_dataset import build_train_valid_test_datasets
#   setup_model_and_optimizer / get_model
#   build_train_valid_test_data_iterators / train

# ── 初始化 ─────────────────────────────────────────────
from megatron.initialize import initialize_megatron
#   _initialize_distributed
#     → torch.distributed.init_process_group
#     → mpu.initialize_model_parallel

# ── 并行分组 ───────────────────────────────────────────
# megatron.core.parallel_state.initialize_model_parallel

# ── 模型 ───────────────────────────────────────────────
# megatron.model.gpt_model.GPTModel
# megatron.model.module.{MegatronModule, Float16Module}
# megatron.model.transformer.{ParallelTransformer, ParallelTransformerLayer,
#                             ParallelAttention, ParallelMLP}
# megatron.model.language_model.{TransformerLanguageModel, Embedding}

# ── 张量并行算子 ───────────────────────────────────────
# megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy
# megatron.core.tensor_parallel.layers.{VocabParallelEmbedding,
#     ColumnParallelLinear, RowParallelLinear, _initialize_affine_weight_gpu}
# megatron.core.tensor_parallel.mappings.{
#     copy_to_tensor_model_parallel_region,  _CopyToModelParallelRegion,
#     gather_from_tensor_model_parallel_region, _GatherFromModelParallelRegion,
#     _ScatterToModelParallelRegion, _ReduceFromModelParallelRegion}

# ── 配置 ───────────────────────────────────────────────
# megatron.core.model_parallel_config.ModelParallelConfig   # "Megatron Core 基础配置"

# ── 数据 ───────────────────────────────────────────────
# megatron.data.data_samplers.build_pretraining_data_loader(dataset, consumed_samples)
```

`pretrain()` 官方四步（再次保留原文 docstring，便于对照源码）：

```
1) initialize Megatron.
2) setup model, optimizer and lr schedule using the model_provider.
3) call train_val_test_data_provider to get train/val/test datasets.
4) train the model using the forward_step_func.
```

初始化 Megatron 的两件事（原文真料保留）：

```
1. 定义模型的切割框架
2. 在此框架上，初始化进程，分配 GPU，设置进程组（DP/TP/PP）
```

---

## 常见问题 / 坑

| 现象 / 疑问 | 原因 | 对策 |
|------------|------|------|
| all-reduce 卡死或极慢 | TP 组被分到了跨机 GPU（rank 号没相邻） | 确认 `initialize_model_parallel` 的 TP=同机相邻 rank；TP ≤ 单机 GPU 数 |
| world_size 与并行度对不上 | 必须满足 `N = TP × PP × DP` | 16=2×4×2，三者乘积必须整除/等于总卡数 |
| 改了 TP 但 loss 变了 | 词表/权重切片初始化不一致 | 看 `_initialize_affine_weight_gpu` 是否用同种子切片 |
| 直接对 TP-切分的 logits 算 CE 结果错 | softmax 分母需全词表，本地分母不全 | 必须用 `_VocabParallelCrossEntropy`（跨组 all-reduce 分母） |
| 自己写模型忘了加通信 | f/g 封在 `autograd.Function`，少一个反向梯度就错 | Column 层入口用 `f`，Row 层出口用 `g`，成对出现 |
| 断点续训数据重复 | dataloader 没传 `consumed_samples` | `build_pretraining_data_loader(dataset, consumed_samples)` 传入已消费数 |
| PP 下某 rank 没有 embedding | `get_model` 按 stage 只建本 rank 的层 | 正常现象；`pre_process/post_process` 控制是否含 embed/输出头 |
| 张量形状在 TP 后变了 | 误以为 TP 改变 `[s,b,h]` | TP 切的是**权重**，`[s,b,h]` 输入输出形状不变（§6） |

---

## 🔗 跳转链接

- 总枢纽：[[00-知识地图]]
- 训练总览：[[llm-train/README]] · [[llm-train/pytorch/distribution/README]]
- 本主题：[[llm-train/megatron/README]] · [[llm-train/megatron-deepspeed/README]]
- 框架源码：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]] · [[ai-framework/huggingface-peft/README]]
- 微调对照：[[llm-train/peft/PEFT-API]] · [[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]]
- 通信底座：[[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/NCCL]]
- 上下游：[[llm-algo/transformer/模型架构]] · [[llm-alignment/RLHF]] · [[llm-compression/quantization/量化基础]]
- 推理对照：[[B07:llm-inference/大模型推理张量并行]]
