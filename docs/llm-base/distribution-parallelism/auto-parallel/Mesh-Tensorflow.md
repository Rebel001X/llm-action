# Mesh-TensorFlow：把"数据并行"推广到任意维度的自动并行框架

> 一句话定位：Mesh-TensorFlow 用一套 DSL，把张量的每一个**命名维度**显式映射到一张**处理器网格（mesh）**的某个**物理维度**上，从而把"只在 batch 维拆分"的数据并行，泛化成"在任意维度（batch / hidden / vocab / heads…）都能拆分"的自动并行。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-base/distribution-parallelism/tensor-parallel/tensor-parallel]] · [[ai-infra/网络/集合通信原语]] · [[ai-framework/megatron-lm/README]] · [[llm-inference/大模型推理张量并行]] · [[llm-algo/FLOPs]]

## 阅读地图

| 小节 | 你会学到 | 关键词 |
|------|----------|--------|
| 0 | 一句话锚点：mesh × layout = 自动并行 | SPMD 泛化 |
| 1 | 地基：SPMD / 数据并行 / 张量并行回顾 | batch 维拆分 |
| 2 | 命名维度（Named Dimension）为什么是核心 | 语义化 shape |
| 3 | Mesh（处理器网格）= 多维设备阵列 | TPU pod 拓扑 |
| 4 | Layout（布局规则）= 维度→网格的映射 | 拆 or 复制 |
| 5 | 拆分的语义：split / replicate / 自动通信 | allreduce/allgather |
| 6 | 一个矩阵乘如何被自动并行 | einsum 视角 |
| 7 | 数据并行 / 模型并行 / 混合，只改 layout | 一行切换 |
| 8 | 通信量与显存的手算 | 字节级估算 |
| 9 | 局限、与 Megatron/GSPMD 的关系 | 历史脉络 |
| 数值手算 | Transformer FFN 层的通信/显存账本 | 65536 设备 |
| FAQ | 易混点澄清 | — |

## 0. 一句话锚点

把分布式训练想成两个独立的决策：

1. **逻辑模型**：张量有哪些维度（`batch, seq, d_model, vocab …`），算子是什么（基本都是 `einsum` + 逐元素）。
2. **物理摆放**：我有一张 $r_1 \times r_2 \times \dots$ 的**设备网格**；我决定"模型的某个维度，沿网格的哪个物理轴去切"。

```
逻辑张量(命名维度)        +      Layout(映射规则)      =     自动 SPMD 程序
[batch, d_model, vocab]          {batch->mesh_x,            (含必要的 allreduce/
                                  vocab ->mesh_y}            allgather/allslice)
```

**数据并行** = layout 只把 `batch` 维映射到网格；**张量/模型并行** = layout 把 `d_model`/`vocab` 等维映射到网格。框架本身不变，**只改 layout 这一个配置**。这就是 Mesh-TF 的全部精髓。

## 1. 地基：从 SPMD 到"任意维度并行"

### 1.1 SPMD 与数据并行

SPMD = Single Program, Multiple Data：每个设备跑**同一份程序**，喂**不同的数据切片**。最常见的实例就是**数据并行**——按 batch 维切：

```
全局 batch = 8,  4 张卡
设备0: 样本[0,1]   设备1: 样本[2,3]
设备2: 样本[4,5]   设备3: 样本[6,7]
         每张卡都有【完整的模型权重副本】
反向后:  对梯度做 AllReduce  →  权重保持一致
```

数据并行的本质：**沿 batch 维 split 激活，沿权重维 replicate 权重**。

### 1.2 它的天花板

- 模型大到**单卡放不下权重**时，数据并行失效（每卡仍要存整份权重）。
- 此时需要**模型并行**：把权重本身切开（沿 `d_model`、沿 `vocab`、沿 attention heads…）。

Mesh-TF 的洞见：数据并行不过是"在 batch 维切"的特例。**为什么不能在任意维度切？** 只要规定清楚"每个维度切到哪、切完算子需要补什么通信"，就能统一处理。这正是 [[llm-base/distribution-parallelism/tensor-parallel/tensor-parallel]] 与 [[ai-framework/megatron-lm/README]] 后来工程化的同一思想，Mesh-TF 是其更早、更抽象的源头之一。

## 2. 命名维度（Named Dimension）：一切的起点

普通框架里张量是 `shape=[8, 512, 1024]`，维度靠**位置**区分，机器不知道"哪一维是 batch、哪一维是 hidden"。Mesh-TF 要求给每个维度**起名字**并带长度：

```
mtf.Dimension("batch",   8)
mtf.Dimension("d_model", 1024)
mtf.Dimension("vocab",   50000)

张量 X : Shape([batch, d_model])   # 而不是 [8, 1024]
```

```
   位置式 shape                命名式 shape (Mesh-TF)
  ┌────┬─────┬──────┐        ┌──────┬─────────┬───────┐
  │ 8  │ 512 │ 1024 │   →    │batch │  seq    │d_model│
  └────┴─────┴──────┘        └──────┴─────────┴───────┘
   机器不懂语义                每一维有名字 ⇒ 可被 layout 引用
```

为什么必须命名？因为**并行决策是"按维度名"下达的**：layout 说"把名叫 `vocab` 的维度切到网格的 `y` 轴"。没有名字就无从指代。这也让算子（`einsum`）能按维度名自动对齐、自动判断需要什么通信。

## 3. Mesh：把设备摆成多维网格

**Mesh** = 处理器的**逻辑多维阵列**。它和"张量"无关，描述的是**物理设备拓扑**。

```
8 个 TPU core 摆成 2D mesh: rows=2, cols=4
                cols(x) →
        ┌────┬────┬────┬────┐
rows(y) │ d0 │ d1 │ d2 │ d3 │   ← y=0
   ↓    ├────┼────┼────┼────┤
        │ d4 │ d5 │ d6 │ d7 │   ← y=1
        └────┴────┴────┴────┘
mesh_shape = [("x", 4), ("y", 2)]   总设备 = 4*2 = 8
```

- mesh 是**几维**、每维多长，由你定（要匹配真实硬件互联，TPU pod 是 2D/3D torus）。
- mesh 的每个**轴**（`x`、`y`…）就是后面 layout 里"可以把某个张量维度切过去"的目标。

> 关键区分：**张量维度（batch/d_model/vocab）** 是逻辑的；**网格维度（x/y）** 是物理的。Layout 就是这两套维度之间的"连线"。

## 4. Layout：维度 → 网格的映射规则

Layout 是一组 `(张量维度名 → 网格轴名)` 的映射。**没被映射的张量维度 = 在所有设备上完整保留（不切）。**

```
layout = [("batch",   "x"),      # batch 维沿网格 x 轴切成 4 份
          ("d_model", "y")]      # d_model 维沿网格 y 轴切成 2 份
          # vocab 没出现 ⇒ vocab 维不切，每个设备都有完整 vocab
```

一个 `Shape([batch=8, d_model=1024, vocab=50000])` 的张量，在上面的 mesh+layout 下，每个设备实际持有的**本地分片**：

```
本地 batch   = 8 / 4 = 2       (沿 x 切)
本地 d_model = 1024 / 2 = 512  (沿 y 切)
本地 vocab   = 50000           (未切, 完整)

⇒ 每设备本地 shape = [2, 512, 50000]
   全局张量被切成 4*2 = 8 块, 每块互不重叠
```

```
       网格 x=0      x=1      x=2      x=3
y=0  [b0-1,        [b2-3,    ...      ...     ]  d_model[0:512]
      d0-511]
y=1  [b0-1,        ...                ...     ]  d_model[512:1024]
      d512-1023]
     每格 = 该 (batch切片, d_model切片) 的子张量
```

**约束**：一个张量维度只能映射到**一个**网格轴；一个网格轴可以承载**多个**不同张量的不同维度（不同算子用不同维度切到同一物理轴）。要求被切维度长度能被网格轴长度整除（否则需 padding）。

## 5. 拆分语义：split / replicate / 自动补通信

这是 Mesh-TF 最精彩处：**算子的通信是自动推导的**。规则来自"einsum 的维度怎么对齐"。

### 5.1 三种张量分布状态

```
SPLIT       : 某维被切到网格轴 → 各设备持有不重叠分片
REPLICATE   : 某维未映射     → 各设备持有完整副本
（中间态）   : 计算产生的 "部分和"，需要 reduce 才完整
```

### 5.2 矩阵乘 = einsum，按"求和维"判断通信

考虑 $Y = X W$，写成 einsum：

$$Y_{b,o} = \sum_{i} X_{b,i}\, W_{i,o}$$

- `b`(batch)、`o`(out)、`i`(in/contract，被求和)。
- **被求和的维 `i`** 是关键：如果 `i` 被 split 到某网格轴，那么每个设备只算出**部分和**，必须在该轴上做 **AllReduce** 才能得到完整 $Y$。

```
情形A:  i (contract维) 被 split 到 x 轴
   设备各自算  局部和 = Σ_{i in 本地分片} X·W
   ⇒ 沿 x 轴 AllReduce 求和  →  完整 Y      【需要通信】

情形B:  i 不切, batch b 被 split 到 x 轴
   每设备算自己那批样本的完整 Y
   ⇒ 无需通信 (这就是数据并行)             【零通信】

情形C:  o (输出维) 被 split 到 y 轴
   每设备算出 Y 的一部分输出通道, 各存各的
   ⇒ 无需通信, 但 Y 是 SPLIT 状态           【零通信, 延后】
```

**自动化的来源**：你只声明 layout，Mesh-TF 看每个 einsum 的求和维是否被切，自动插入 `AllReduce`（情形A）或保持分片（情形C）。下游算子若需要完整维度，再自动插入 `AllGather`。这与 [[ai-infra/网络/集合通信原语]] 里的原语一一对应。

## 6. 一个 FFN 层如何被自动并行（对照 Megatron）

Transformer 的 FFN：$Y = \text{GeLU}(X W_1) W_2$，其中 $X\in[b, d]$，$W_1\in[d, 4d]$，$W_2\in[4d, d]$。

经典的"列切-行切"张量并行（Megatron 风格）在 Mesh-TF 里就是一份 layout：

```
设网格轴 "mp"(model-parallel) 长度 = N

W1 沿 "4d"(隐藏扩张维) split 到 mp   →  列切
   H = GeLU(X·W1)  : H 在 4d 维是 SPLIT, 无需通信(求和维 d 未切)
W2 沿 "4d" split 到 mp              →  行切
   Y = H·W2        : 求和维 4d 被切 ⇒ 沿 mp 轴 AllReduce  ✅
```

```
   X[b,d]            W1[d,4d] 按4d列切        W2[4d,d] 按4d行切
全复制 ──► 各设备 ─┬─ W1_part ─► H_part(GeLU) ─┬─ W2_part ─► 局部Y
                  │ (4d/N 列)   (无通信)        │ (4d/N 行)
                  └────────────────────────────┘
                                         沿 mp 轴 AllReduce → 完整 Y
```

**一次 FFN 前向只需 1 次 AllReduce**——和 Megatron-LM 完全一致，因为它们是同一个数学结构。区别只是：Megatron 手写了通信，Mesh-TF 由 layout 自动推出。

## 7. 同一份模型，改 layout 即换并行策略

这是 Mesh-TF 的"卖点"：模型代码不动，换并行只换一行 layout。

```
纯数据并行:        layout = [("batch", "all")]            # 只切 batch
纯模型并行:        layout = [("d_model", "all")]          # 只切 hidden
混合(2D):          mesh=[("dp",4),("mp",2)]
                  layout=[("batch","dp"), ("d_model","mp")]  # batch×hidden 同切
专家并行(MoE):     layout=[("experts","mp")]              # 切 expert 维
词表并行:          layout=[("vocab","mp")]                # 切 vocab(省 logits 显存)
```

```
        ┌──────────────┐
        │  模型代码     │  ← 永远不变(命名维度 + einsum)
        └──────┬───────┘
               │  挂上不同 layout
   ┌───────────┼───────────┬───────────┐
   ▼           ▼           ▼           ▼
 数据并行    模型并行    2D混合      MoE专家并行
```

MoE 场景下 `experts` 维天然适合切分，参见 [[llm-algo/moe/README]]：每个设备只持有一部分专家，token 经 gating 路由（AllToAll）后到对应设备计算。

## 8. 通信量与显存的字节级估算

设：网格模型并行轴长 $N$，数据类型 4 字节（fp32）。

### 8.1 AllReduce 通信量（Ring 实现）

对大小为 $V$（元素数）的张量做 AllReduce，Ring-AllReduce 每设备收发约 $2\cdot\frac{N-1}{N}\cdot V$ 个元素（$\approx 2V$，与 $N$ 几乎无关）。FFN 前向那次 AllReduce 的 $Y$ 大小 $V = b\cdot d$。

$$\text{通信字节} \approx 2 \cdot b \cdot d \cdot 4\ \text{Byte（每设备收发）}$$

### 8.2 权重显存（被切后）

未切：$W_1+W_2 = 2\cdot d\cdot 4d \cdot 4 = 32 d^2$ 字节/设备。
沿 mp 轴切 $N$ 份后：每设备只存 $\dfrac{32 d^2}{N}$ 字节——**显存随 $N$ 线性下降**，这正是模型并行能放下大模型的原因。

```
N=1:   ████████████████  32 d² (放不下)
N=2:   ████████          16 d²
N=4:   ████               8 d²
N=8:   ██                 4 d²   ← 切得越多, 每卡越省
        代价: 多了 AllReduce 通信
```

## 数值手算：512 维 FFN 的账本

取 $b=8192$（全局 batch×seq 拉平），$d=1024$，FFN 中间维 $4d=4096$，fp32，模型并行 $N=8$。

**1) 权重显存（每设备）**

$$W_1: d\times 4d / N = 1024\times4096/8 = 524{,}288 \text{ 元素}$$
$$W_2: 4d\times d / N = 4096\times1024/8 = 524{,}288 \text{ 元素}$$
$$\text{合计} = 1{,}048{,}576 \times 4\,\text{B} = 4\,\text{MB/设备}$$

未切时为 $32 d^2 = 32\times1024^2 = 33{,}554{,}432\,\text{B} = 32\,\text{MB}$。切 8 份后正好 $32/8 = 4$ MB，✅ 与公式一致。

**2) 一次前向的 AllReduce 通信量**

$$V = b\times d = 8192\times1024 = 8{,}388{,}608 \text{ 元素}$$
$$\text{每设备收发} \approx 2V\times4\,\text{B} = 2\times8.39\text{e}6\times4 \approx 67\,\text{MB}$$

**3) FFN 计算量（FLOPs，参考 [[llm-algo/FLOPs]]）**

两次矩阵乘，前向 $\approx 2\times(b\cdot d\cdot 4d)\times2 = 16\,b\,d^2$：
$$16\times8192\times1024^2 \approx 1.37\times10^{11}\ \text{FLOPs}$$
切到 8 设备后每设备约 $1.72\times10^{10}$ FLOPs。

**4) 计算/通信比（粗判是否划算）**

每设备计算 $1.72\text{e}10$ FLOPs vs 通信 $67$ MB。在带宽 $100$ GB/s、算力 $50$ TFLOPS 的设想机器上：
- 计算时间 $\approx 1.72\text{e}10 / 5\text{e}13 \approx 0.34\ \text{ms}$
- 通信时间 $\approx 67\text{e}6 / 100\text{e}9 \approx 0.67\ \text{ms}$

通信 > 计算 ⇒ 该层模型并行**通信受限**，提示：模型并行轴宜放在**高带宽域内**（如单机 NVLink / TPU 同板），跨低带宽切会赔本。这就是为什么 layout 必须匹配真实拓扑。（数值为教学设想，实际以官方/实测为准。）

## 历史定位与演进

```
GPipe(流水线) ──┐
                ├─► Mesh-TF: 命名维度 + mesh + layout(更抽象/侵入性强)
数据并行(SPMD) ─┘            │
                            ▼
Megatron-LM(手写张量并行) ◄─ 同一数学, 工程化更轻
                            │
                            ▼
        GSPMD / XLA SPMD(编译器自动分片, "Mesh-TF 的继任者思想")
                            │
                            ▼
        JAX  pjit / shard_map / Mesh  (今天最常用的命名网格 API)
```

要点：Mesh-TF 的 **mesh + 命名维度 + layout** 三件套，几乎被现代 SPMD 体系（GSPMD、JAX `Mesh`/`PartitionSpec`）原样继承。理解它，就理解了"声明式自动并行"的母版。

## 常见问题

| 问题 | 答 |
|------|-----|
| Mesh-TF 和数据并行的关系？ | 数据并行 = layout 只切 `batch` 维的特例；Mesh-TF 是其向任意维度的泛化。|
| 为什么维度一定要命名？ | layout 按维度名下达并行指令，无名无从指代；也让 einsum 自动判断通信。|
| 通信是手写的吗？ | 不是。框架按每个 einsum 的"求和维是否被切"自动插 AllReduce/AllGather。|
| 一张量维度能切到多个网格轴吗？ | 一个张量维度只能映到一个网格轴；但一个网格轴可承载多个张量的不同维度。|
| 被切维度不能整除网格长度怎么办？ | 需 padding 到可整除；否则布局非法。|
| 它能并行卷积吗？ | 早期版本主要面向矩阵乘/Transformer，卷积支持弱，故偏向 Language Model。|
| 它和 Megatron 哪个好？ | 数学等价；Megatron 工程更轻、生态更广；Mesh-TF 抽象更统一、侵入性更强。|
| 现在还用 Mesh-TF 吗？ | 思想被 GSPMD / JAX `Mesh` 继承，直接用旧库较少，但概念是现代 SPMD 的根。|
| 模型并行轴该放哪？ | 放在高带宽域（NVLink/同板 TPU），跨低带宽切会因通信受限赔本（见手算第4步）。|

## 🔗 跳转链接

- 上游导航：[[00-知识地图]]
- 张量并行（同一数学、工程化版本）：[[llm-base/distribution-parallelism/tensor-parallel/tensor-parallel]] · [[ai-framework/megatron-lm/README]] · [[llm-inference/大模型推理张量并行]]
- 集合通信原语（AllReduce/AllGather/AllToAll 的代价模型）：[[ai-infra/网络/集合通信原语]]
- 计算量/显存账本：[[llm-algo/FLOPs]] · [[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]]
- MoE 专家并行（experts 维切分）：[[llm-algo/moe/README]]
- 训练框架与并行栈：[[ai-framework/deepspeed/README]] · [[llm-train/README]]
- 计算通信重叠（隐藏 AllReduce 延迟）：[[llm-optimizer/计算通信重叠]]
- 硬件拓扑（为什么 layout 要匹配互联）：[[ai-infra/算力/GPU工作原理]]
