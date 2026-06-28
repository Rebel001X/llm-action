# Prompt-Tuning / LoRA 对比

> 只训练一小撮"软提示"向量或一对低秩矩阵，冻结整个大模型本体——这就是参数高效微调（PEFT）。本篇从最底层把 Prompt-Tuning 与 LoRA 讲透，并逐数手算。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-train/peft/Prefix-Tuning]] [[llm-train/peft/PEFT-API]]

## 阅读地图

| 节 | 内容 | 你将带走 |
|----|------|---------|
| 0 | 一句话锚点 | 一句话记住 PEFT 的本质 |
| 1 | 地基：全量微调为什么贵 | 显存账、参数账 |
| 2 | Prompt-Tuning：软提示是什么 | embedding 拼接的原子级理解 |
| 3 | 软提示 vs 硬提示 vs Prefix | 三者边界 |
| 4 | LoRA：低秩更新原理 | $\Delta W = BA$ 从零推 |
| 5 | LoRA 手算 ΔW=BA | 一个 3×3 的数值例子 |
| 6 | 为什么参数高效 | 参数量/显存逐数对账 |
| 7 | 何时用哪个 | 决策树 |
| 8 | 数值示例：BLOOMZ-560m | 复现仓库里的 8192 |
| 9 | PEFT 代码骨架 | 跑起来 |
| 对照表 | 复杂度/适用对比 | 一图流 |
| 面试 | 高频问答+追问 | 可背诵 |

## 0. 一句话锚点

- **全量微调**：改动模型里**所有** $W$（几亿~几千亿个数）。
- **Prompt-Tuning**：模型一个字不改，只在输入**前面**插入 $k$ 个**可训练的"假词向量"**（soft prompt），让这几个向量去"操控"冻结的大模型。
- **LoRA**：模型权重 $W$ 一个字不改，但在它**旁边**并联一条 $\Delta W = BA$（两个瘦长矩阵相乘）的小路，只训练 $B,A$。

> 共同信条：**冻结主体，只训练极少新增参数**。差别只在"新增的东西插在哪、长什么样"。

## 1. 地基：全量微调为什么贵

设一个 Transformer 参数量为 $N$。全量微调（Full Fine-Tuning, FT）时，Adam 优化器要为**每个**参数额外存：

| 项目 | 大小（fp16/混合精度训练常见配置） |
|------|------|
| 权重 weights | $2N$ 字节（fp16）|
| 梯度 gradients | $2N$ 字节 |
| Adam 一阶动量 $m$ | $4N$ 字节（fp32）|
| Adam 二阶动量 $v$ | $4N$ 字节（fp32）|
| fp32 主权重副本 | $4N$ 字节 |

```
全量微调每参数显存账（不含激活）：
 weights 2 + grad 2 + m 4 + v 4 + master 4 = 16 字节 / 参数
```

**手算**：一个 7B 模型，$N=7\times10^9$。

$$16 \text{字节} \times 7\times10^9 = 1.12\times10^{11} \text{字节} \approx 112\ \text{GB}$$

> 仅优化器状态就 112 GB，再加激活值，单卡 80GB A100 装不下。这就是 PEFT 存在的根本理由：**我们能不能只让一小撮参数有梯度/动量？**

PEFT 的回答：冻结那 $N$ 个权重（它们只需 $2N$ 字节存权重、**无梯度无动量**），只给新增的 $N_{\text{train}} \ll N$ 个参数配齐 16 字节/参数的"全套装备"。

## 2. Prompt-Tuning：软提示（soft prompt）到底是什么

### 2.1 先看"硬提示"

你平时写的 prompt 是**文字**：`"判断这条推文是不是投诉："`。它被 tokenizer 切成 token id，再过 **embedding 表** $E \in \mathbb{R}^{V\times d}$（$V$ 词表大小，$d$ 隐藏维），变成一串向量。这串向量是**离散词表里实际存在的词**对应的行，所以叫**硬**——你只能从词表里挑词。

### 2.2 软提示：把"提示"从词表里解放出来

Prompt-Tuning 的洞见：既然输入到 Transformer 的本质是**一串 $d$ 维向量**，那为什么提示一定要对应"真实的词"？

我们直接新建 $k$ 个**可训练向量** $P = [p_1, p_2, \dots, p_k]$，每个 $p_i \in \mathbb{R}^d$。它们不对应任何真实 token，是连续空间里自由游走的"假词"——**soft prompt（软提示）**。

```
                 ┌──────────── 冻结的大模型 (Transformer) ────────────┐
                 │                                                     │
 [p1 p2 ... pk]  │  [e(x1) e(x2) ... e(xn)]                            │
  软提示(可训练)  →  真实输入词向量(由冻结embedding给出)  → 多层注意力 → 输出
  k 个 d 维向量     │           n 个 d 维向量                          │
   ▲只有这里有梯度   └─────────────────────────────────────────────────┘
```

前向时把软提示**拼在输入序列最前面**：

$$\text{输入序列} = [\,p_1,\dots,p_k,\ e(x_1),\dots,e(x_n)\,] \in \mathbb{R}^{(k+n)\times d}$$

整个 Transformer（注意力、FFN、最后的 LM head）**全部冻结**，loss 反传只更新 $P$ 这 $k\times d$ 个数。

### 2.3 为什么这能 work？——"为什么"层面

- 注意力机制里，后面的真实 token 会 **attend to** 前面的软提示。软提示相当于给每一层注入了一段"任务上下文/任务指令"的连续表示。
- 训练就是在搜索：**哪 $k$ 个向量能最好地"引导"冻结模型在这个任务上输出正确答案**。
- 模型越大，冻结主体本身能力越强，软提示越能"四两拨千斤"——Lester 等（2021）发现 Prompt-Tuning 在 10B+ 规模时几乎能追平全量微调。

### 2.4 初始化的讲究（仓库示例里就用了）

```
prompt_tuning_init = TEXT
prompt_tuning_init_text = "Classify if the tweet is a complaint or not:"
```

- `PromptTuningInit.RANDOM`：软提示随机初始化。简单但收敛慢。
- `PromptTuningInit.TEXT`：用一句**真实文字**的 embedding 来初始化软提示，相当于"站在一个好起点出发"，再让它自由漂移。仓库示例正是用这句英文初始化 8 个虚拟 token。

> 注意：`num_virtual_tokens=8` 决定了 $k=8$，与初始化文本的 token 数最好对齐（多截少补）。

## 3. 软提示 vs 硬提示 vs Prefix-Tuning

| 维度 | 硬提示(Prompt Engineering) | Prompt-Tuning(软提示) | Prefix-Tuning |
|------|------|------|------|
| 提示内容 | 真实词 token | 连续可训练向量 | 连续可训练向量 |
| 插在哪 | 仅输入层文本 | 仅**输入 embedding 层**最前 | **每一层**的 K/V 前都插 prefix |
| 训练吗 | 否 | 是(只训软提示) | 是(只训 prefix，常配 MLP 重参数化) |
| 新增参数 | 0 | $k\times d$ | $\approx L\times 2\times k\times d$（每层 K、V）|
| 表达力 | 弱 | 中（只动第一层） | 强（每层都注入） |
| 关系 | —— | Prefix 的简化版（只在第一层） | Prompt-Tuning 的"加深版" |

```
Prompt-Tuning：  软提示只在最底层注入一次
   层L ───────────────
   ...                  ← 中间各层没有任何新增向量
   层1  [P][真实词...]   ← 仅此处插

Prefix-Tuning：  每一层的注意力 K/V 前都插 prefix
   层L  [pre_L][...]
   ...   [pre_i][...]    ← 每层都插，所以参数随层数 L 线性增长
   层1  [pre_1][...]
```

详见 [[llm-train/peft/Prefix-Tuning]]。一句话记忆：**Prompt-Tuning = 只在输入层插的极简 Prefix-Tuning。**

## 4. LoRA：低秩更新原理（从零推 $\Delta W = BA$）

### 4.1 全量微调改的是什么

线性层做 $h = Wx$，$W \in \mathbb{R}^{d\times d}$。微调即把 $W$ 更新成 $W' = W + \Delta W$，其中 $\Delta W$ 是这次任务学到的"权重改变量"。全量微调里 $\Delta W$ 是个**满秩、$d\times d$** 的大矩阵——这正是贵的来源。

### 4.2 关键假设：$\Delta W$ 是"低秩"的

LoRA（Hu 等，2021）的核心假设：**微调带来的权重改变量 $\Delta W$ 内在秩很低**——也就是说 $\Delta W$ 虽然形状是 $d\times d$，但它真正"有用的方向"只有 $r$ 个（$r \ll d$）。

**线性代数事实**：任何秩不超过 $r$ 的矩阵 $\Delta W \in \mathbb{R}^{d\times d}$，都能分解成两个瘦矩阵相乘：

$$\Delta W = B A,\qquad B \in \mathbb{R}^{d\times r},\quad A \in \mathbb{R}^{r\times d}$$

```
   ΔW (d×d, 满)         =        B (d×r)    ·      A (r×d)
   ┌───────────┐                 ┌──┐              ┌───────────┐
   │           │                 │  │              │           │  (只有 r 行)
   │  d×d 个数  │       ≈         │d×r│       ·      └───────────┘
   │           │                 │  │
   └───────────┘                 └──┘
   参数 d²                       参数 d·r          参数 r·d
                                 ─────合计 2dr ≪ d²─────
```

### 4.3 前向公式与缩放

LoRA 不去改 $W$，而是**并联**一条旁路：

$$h = Wx + \Delta W\,x = Wx + \frac{\alpha}{r}\,B A\,x$$

- $W$：冻结，不训练。
- $A$：随机高斯初始化。$B$：初始化为**全 0**——保证训练第一步 $\Delta W = B A = 0$，模型从原模型平滑出发，不破坏预训练能力。
- $\frac{\alpha}{r}$：缩放系数。$\alpha$ 是超参，除以 $r$ 让"不同 $r$ 下学习率尺度可比"。常设 $\alpha = 2r$ 或 $\alpha = r$。

### 4.4 训练只更新 $A,B$；推理可合并

- **训练**：只有 $A,B$ 有梯度，参数从 $d^2$ 降到 $2dr$。
- **推理**：可以把 $W \leftarrow W + \frac{\alpha}{r}BA$ **预先合并**回原权重，于是推理时**零额外延迟**（不像 Prompt/Prefix-Tuning 会拉长序列）。这是 LoRA 相对软提示类方法的杀手锏。

## 5. 手算 $\Delta W = BA$（一个具体数值例子）

取 $d=3,\ r=1,\ \alpha=1$。令

$$B = \begin{bmatrix}2\\0\\1\end{bmatrix}\ (3\times1),\qquad A = \begin{bmatrix}1 & 0 & 3\end{bmatrix}\ (1\times3)$$

逐元素算外积 $\Delta W = BA$（$\Delta W_{ij} = B_i A_j$）：

$$\Delta W = \begin{bmatrix}2\cdot1 & 2\cdot0 & 2\cdot3\\ 0\cdot1 & 0\cdot0 & 0\cdot3\\ 1\cdot1 & 1\cdot0 & 1\cdot3\end{bmatrix} = \begin{bmatrix}2 & 0 & 6\\ 0 & 0 & 0\\ 1 & 0 & 3\end{bmatrix}$$

```
 B (3×1)   A (1×3)            ΔW = B·A (3×3)
 [2]                          [2*1  2*0  2*3]   [2 0 6]
 [0]   ·   [1 0 3]      =     [0*1  0*0  0*3] = [0 0 0]
 [1]                          [1*1  1*0  1*3]   [1 0 3]
```

**验证秩**：$\Delta W$ 的每一行都是 $A=[1,0,3]$ 的倍数（行 1 是 2 倍、行 3 是 1 倍、行 2 是 0 倍）→ 所有行共线 → $\text{rank}(\Delta W)=1=r$。✔ 这正是低秩的含义：$9$ 个输出数，全由 $B$ 的 3 个数 + $A$ 的 3 个数（共 6 个）生成。

**前向一遍**：输入 $x=[1,2,1]^\top$，旁路贡献

$$\Delta W\,x = \begin{bmatrix}2 & 0 & 6\\0&0&0\\1&0&3\end{bmatrix}\begin{bmatrix}1\\2\\1\end{bmatrix} = \begin{bmatrix}2+0+6\\0\\1+0+3\end{bmatrix} = \begin{bmatrix}8\\0\\4\end{bmatrix}$$

更省力的算法：先算 $Ax = 1\cdot1+0\cdot2+3\cdot1 = 4$（一个标量），再 $B\cdot(Ax) = [2,0,1]^\top\cdot4 = [8,0,4]^\top$。

> **为什么省**：直接 $\Delta W x$ 要 $d^2=9$ 次乘法；走 $B(Ax)$ 只要 $dr + rd = 6$ 次。$r$ 越小，越省，且**从不显式构造 $d\times d$ 的 $\Delta W$**，省显存。

## 6. 为什么参数高效——逐数对账

### 6.1 LoRA 参数量

单个 $d\times d$ 线性层：

$$\text{全量}=d^2,\qquad \text{LoRA}=2dr,\qquad \text{压缩比}=\frac{2dr}{d^2}=\frac{2r}{d}$$

**手算**：$d=4096$（7B 模型的隐藏维量级），$r=8$：

$$\frac{2\times8}{4096}=\frac{16}{4096}\approx 0.39\%$$

即每个被 LoRA 包裹的线性层，只训练原参数的 **0.39%**。

### 6.2 Prompt-Tuning 参数量

$$N_{\text{train}} = k\times d$$

与模型层数 $L$、模型总参数 $N$ **完全无关**——这是最极致的参数高效，但表达力也最受限。

### 6.3 显存对账（接第 1 节）

只有新增参数需要梯度+Adam 动量（16 字节/参数），冻结主体只存 fp16 权重（2 字节/参数）。LoRA 微调 7B：

```
冻结主体权重：  2 字节 × 7e9          = 14   GB
LoRA 可训练参数(设共 ~4.2M)：16 × 4.2e6 ≈ 0.067 GB
                                       ─────────
合计(不含激活)  ≈ 14 GB  ← vs 全量 112 GB
```

> 112 GB → 14 GB，单卡可训。**省的就是那 16 字节/参数的优化器开销，只施加在极少数新参数上。**

## 7. 何时用哪个（决策树）

```
要微调一个大模型，怎么选？
│
├─ 推理延迟极其敏感、想合并回原权重做到零额外延迟？
│        └─ ✅ LoRA / QLoRA（推理可合并）
│
├─ 任务多、要给同一底模挂很多套"插件"热切换？
│        ├─ 想要最小存储、每任务只存几 KB 向量 → Prompt-Tuning
│        └─ 想要更强效果、每任务存几 MB → LoRA（每任务一组 BA）
│
├─ 底模非常大(10B+)、任务相对简单(分类/意图)？
│        └─ ✅ Prompt-Tuning 往往够用，参数最少
│
├─ 任务复杂(生成/推理/代码)、要逼近全量微调效果？
│        └─ ✅ LoRA（必要时 r 调大、覆盖更多层 q,k,v,o,FFN）
│
└─ 显存极度紧张(单张 24G 卡训 7B+)？
         └─ ✅ QLoRA：底模 4-bit 量化 + LoRA 旁路
```

经验法则：

- **效果优先 / 生成任务**：LoRA（业界默认首选）。
- **极致省参数 / 海量任务热插拔 / 超大底模 + 简单任务**：Prompt-Tuning。
- **介于两者、要每层注入更强引导**：Prefix-Tuning（见 [[llm-train/peft/Prefix-Tuning]]）。

## 8. 数值示例：复现仓库里的 8192 个可训练参数

仓库示例用 **BLOOMZ-560m**，配置 `num_virtual_tokens=8`。BLOOMZ-560m 的隐藏维 $d=1024$。

$$N_{\text{train}} = k \times d = 8 \times 1024 = 8192$$

对账 `print_trainable_parameters()` 的输出：

```
trainable params: 8,192 || all params: 559,222,784 || trainable%: 0.00146%
```

- $8192 = 8\times1024$ ✔ 正好是软提示矩阵 $P\in\mathbb{R}^{8\times1024}$ 的元素数。
- 可训练占比：$\dfrac{8192}{559{,}222{,}784}\times100\% \approx 0.00146\%$ ✔
- `all params` $\approx 5.59\times10^8$ 含了冻结的 BLOOMZ 本体 + 8192 个软提示。

> **手算复现**：$8192 / 559222784 = 1.4649\times10^{-5} = 0.0014649\%$，与日志完全吻合。换 `num_virtual_tokens=20` 则变 $20\times1024=20480$ 个可训练参数——线性可控。

引入库与创建 Prompt-Tuning 配置：

```python
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
from transformers import AutoModelForCausalLM

model_name_or_path = "/data/nfs/llm/model/bloomz-560m"

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,                       # k = 8 个软提示
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_name_or_path,
)
```

## 9. PEFT 代码骨架（Prompt-Tuning 与 LoRA 一行之差）

```python
# 1) 包装基础模型 —— Prompt-Tuning
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# trainable params: 8,192 || all params: 559,222,784 || trainable%: 0.0014648902430985358

# 2) 换成 LoRA —— 只改 config
from peft import LoraConfig
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.05,      # r=8, alpha/r=2 缩放
    target_modules=["query_key_value"],          # BLOOM 注意力线性层
)
model = get_peft_model(base_model, lora_config)  # 旁路 BA 自动挂上
```

> `get_peft_model` 会自动冻结主体、只给新增参数置 `requires_grad=True`。训练循环与普通 `Trainer` 完全一致。更多 API 见 [[llm-train/peft/PEFT-API]]。

## 对照表：四种方法复杂度/适用一图流

| 维度 | Full FT | Prompt-Tuning | Prefix-Tuning | LoRA |
|------|------|------|------|------|
| 训练参数量 | $N$（全部）| $k\,d$ | $\sim 2Lkd$ | $\sum 2d r$ |
| 与层数 $L$ 相关 | 是 | 否 | 是（线性）| 是（看插几层）|
| 改原权重 | 是 | 否 | 否 | 否（旁路）|
| 推理额外延迟 | 0 | 序列变长 $+k$ | 序列变长 $+k$ | **0（可合并）** |
| 显存(7B 量级) | ~112 GB | 最低 | 低 | ~14 GB |
| 表达力 | 最强 | 弱~中 | 中~强 | 强 |
| 多任务热插拔 | 难 | 极易(几KB) | 易 | 易(几MB) |
| 典型超参 | lr | $k$, init | $k$, MLP | $r,\alpha,$ target |
| 首选场景 | 资源充足求极致 | 超大底模+简单任务 | 每层强引导 | **生成/通用默认** |

## 常见问题 / 高频追问

| 问题 | 踩点答案 | 追问与陷阱 |
|------|---------|-----------|
| Prompt-Tuning 和 Prefix-Tuning 啥区别？ | Prompt 只在**输入 embedding 层**插软提示；Prefix 在**每一层** K/V 前都插，参数随层数 $L$ 增长，表达力更强。 | 追问"为什么 Prompt 在小模型上不如 Prefix？"→ 只动一层引导太弱；模型越大软提示越够用。 |
| LoRA 为什么不增加推理延迟？ | 推理前可把 $W\!\leftarrow\!W+\frac{\alpha}{r}BA$ 合并回原权重，前向与原模型一模一样。 | 陷阱：合并后就**不能再热切换**别的 LoRA 了；要切换得保持不合并。 |
| 为什么 $B$ 初始化为 0？ | 让训练首步 $\Delta W=BA=0$，模型从预训练点平滑出发，不破坏已有能力。 | 追问"$A$ 也置 0 行不行？"→ 不行，$A,B$ 全 0 则梯度恒 0，学不动（对称性破缺需要 $A$ 随机）。 |
| LoRA 的 $r$ 怎么选？ | 简单任务 $r=4\!\sim\!8$；复杂/生成任务 $r=16\!\sim\!64$。$r$ 越大越接近全量但越贵。 | 追问"$\alpha$ 作用？"→ 缩放 $\frac{\alpha}{r}$ 调旁路影响强度，常 $\alpha=2r$。 |
| 软提示的参数量怎么算？ | $k\times d$，与模型总参数、层数**无关**。BLOOMZ-560m($d=1024$)、$k=8$ → 8192。 | 陷阱：$k$ 太大会显著挤占上下文窗口长度。 |
| 为什么 $\Delta W$ 能假设低秩？ | 大模型预训练已学到通用表示，下游适配只需在少数方向上微调，改变量内在维度低。 | 追问"哪些层放 LoRA 收益最大？"→ 注意力的 $q,v$ 通常性价比最高。 |
| QLoRA 是什么？ | 底模 4-bit(NF4)量化冻结 + LoRA 旁路(常 bf16)训练，单卡 24G 可训 7B~13B。 | 陷阱：量化的是冻结底模，LoRA 旁路本身仍是高精度，不量化。 |
| 显存到底省在哪？ | 冻结参数无梯度/无 Adam 动量，省掉 16 字节/参数的优化器开销；只对极少新参数施加。 | 追问"激活值省了吗？"→ 省得有限，激活仍随 batch/序列走，必要时配梯度检查点。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全站导航总图
- [[llm-train/peft/Prefix-Tuning]] — 每层注入的"加深版"软提示
- [[llm-train/peft/PEFT-API]] — `get_peft_model` / 各 Config 的 API 细节与训练流程
