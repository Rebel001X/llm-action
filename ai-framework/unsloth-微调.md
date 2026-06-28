# Unsloth (高效微调)

> Unsloth 是一个把 LoRA/QLoRA 微调中**所有热点算子用手写 Triton kernel 重写**的库，在单卡上把训练速度提升约 2 倍、显存占用降低约 50~70%，且**不损失精度**（数学等价）。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/huggingface-peft/README]] [[ai-framework/openai-triton/README]] [[llm-train/peft/PEFT-API]]

- 官方仓库：https://github.com/unslothai/unsloth （版本/API/支持模型列表均**以官方文档为准**）

## 阅读地图

| 章节 | 你将搞懂 | 关键词 |
|------|---------|--------|
| 0 | 一句话锚点 | Triton 重写 / 等价加速 |
| 1 | 地基：LoRA / 显存账本 / Triton 是什么 | 前置 |
| 2 | Unsloth 解决什么问题 | "PyTorch 默认实现不够省" |
| 3 | 整体架构与数据流 | 打补丁 / monkey-patch |
| 4 | 核心机制①：融合 kernel（fused） | kernel launch / 中间张量 |
| 5 | 核心机制②：手算反向（manual backward） | autograd vs 手写 grad |
| 6 | 核心机制③：QLoRA + 显存优化 | 4bit / 梯度检查点 |
| 7 | 速度与显存优势（数值） | 2x / -70% |
| 8 | 支持的模型 | Llama/Mistral/Qwen/Gemma... |
| 9 | 与同类对比（PEFT/TRL/axolotl） | 何时用 |
| 数值例子 | 7B QLoRA 显存手算 | 24GB 卡能不能跑 |
| FAQ | 精度/多卡/坑 | 常见问题 |

---

## 0. 一句话锚点

普通的 LoRA 微调 = `HuggingFace PEFT` 帮你插 LoRA 层 + `PyTorch autograd` 自动算梯度。它**能跑**，但每一步都走 PyTorch 的"通用"路径：算子一个个分开调用、产生大量中间张量、反向全交给自动微分。

**Unsloth 的全部价值**就一句话：

> 把这条路径上**最耗时、最耗显存的几个算子**（RoPE、RMSNorm、Cross-Entropy、LoRA 的矩阵乘、SwiGLU）用 **Triton 手写成融合 kernel**，并**手工推导它们的反向梯度**，从而少启动 kernel、少存中间结果、少读写显存——**结果与原版逐位（近似）等价，只是更快更省**。

它不是新算法，是**同一份数学的极致工程实现**。

```
          原版路径(PyTorch)                Unsloth 路径(Triton)
   ┌────────────────────────┐        ┌────────────────────────┐
   │ op1 → op2 → op3 → op4   │        │   一个融合 kernel        │
   │  ↓     ↓     ↓     ↓    │        │   (op1~op4 合并)         │
   │ 存4个中间张量到显存       │   ⇒    │   中间值留在寄存器/SRAM   │
   │ autograd 反向再来一遍     │        │   手写反向, 不存激活      │
   └────────────────────────┘        └────────────────────────┘
        慢、显存大                          快约2x、省约70%
```

---

## 1. 地基 / 前置（不假设你记得）

### 1.1 LoRA 是什么（一分钟原子化）

微调一个大模型，最朴素是更新它**所有**权重 $W$。但 $W$ 动辄几十亿个数，存梯度+优化器状态很贵。

LoRA 的想法：**冻结原权重 $W$，只在旁边加一个低秩"补丁"**。对某个线性层，前向变成：

$$ y = W x + \frac{\alpha}{r}\, (B A)\, x $$

- $W \in \mathbb{R}^{d\times d}$ 冻结、不算梯度。
- $A \in \mathbb{R}^{r\times d}$、$B \in \mathbb{R}^{d\times r}$ 是**新增的小矩阵**，秩 $r$ 很小（如 8/16/32）。
- $\alpha$ 是缩放系数。只训练 $A,B$，参数量从 $d^2$ 降到 $2dr$。

> 直觉：你不重画整张画，只在透明胶片上加几笔修改，原画不动。

详细见 [[ai-framework/huggingface-peft/README]] 与 [[llm-train/peft/PEFT-API]]。

### 1.2 微调的"显存账本"

训练一步要在显存里同时放下这些东西（这是理解 Unsloth 省显存的关键）：

```
显存 = 模型权重 + 梯度 + 优化器状态 + 激活值(activations) + 临时缓冲
        ↑冻结/4bit   ↑只LoRA小   ↑只LoRA小    ↑前向每层都要存,反向要用   ↑kernel中间张量
```

- **权重**：QLoRA 把它压到 4bit，省一大块。
- **梯度+优化器状态**：LoRA 只对 $A,B$ 有，很小。
- **激活值**：前向时每一层的输出都要**缓存**，因为反向求梯度要用。长序列 / 大 batch 时，**这才是显存第一大头**。
- **临时缓冲**：每个 PyTorch 算子产生的中间张量。

Unsloth 主攻的就是后两项：**用融合 kernel 减少中间缓冲，用梯度检查点 + 手写反向减少激活缓存**。

### 1.3 Triton 是什么

Triton 是 OpenAI 出的、用 **Python 语法写 GPU kernel** 的语言/编译器（见 [[ai-framework/openai-triton/README]]）。

- 普通 PyTorch：你调 `torch.matmul`、`F.softmax`，每个是一个**独立 CUDA kernel**，各自从显存读数据、算完写回显存。
- Triton：你能把好几步**合进一个 kernel**，数据读进 GPU 的高速片上内存（SRAM/寄存器）后**一口气算完**，中间不落显存。

```
PyTorch:  HBM ─读→ [kernel1] ─写→ HBM ─读→ [kernel2] ─写→ HBM ...
                每次都来回搬运(HBM 慢、带宽是瓶颈)

Triton :  HBM ─读→ [ 融合kernel: 算算算 ] ─写→ HBM
                只搬一次, 中间全在SRAM里跑
```

GPU 上**搬数据(显存带宽)往往比算数还慢**，所以"少搬运"= 快。这正是 Unsloth 的物理基础。

---

## 2. Unsloth 解决什么问题

PyTorch + HuggingFace 这套生态**通用、好用**，但为了通用，它的实现是"积木式"的——每个算子独立、保守地保存所有中间结果、反向完全交给 autograd。结果：

| 痛点 | 表现 |
|------|------|
| kernel 太多 | 一层 Transformer 几十个 kernel launch，启动开销+反复读写 HBM |
| 中间张量多 | RMSNorm、RoPE、attention scores 等都落显存 |
| 反向全自动 | autograd 为通用性保存大量中间量，并按通用规则反传，不一定最省 |
| 激活占满显存 | 长上下文微调时显存先于算力爆掉，被迫减小 batch/序列 |

**Unsloth 的答案**：针对 Transformer 微调这个**具体场景**，把热点路径手工优化到底。它**不改变你的训练数学**——loss、梯度、最终权重与原版一致（数值上几乎逐位等价），只是更快更省。

---

## 3. 整体架构与数据流

Unsloth 不是另起炉灶的训练框架，它**寄生/打补丁**在 HuggingFace `transformers` + `peft` + `trl` 之上：

```
┌──────────────────────────────────────────────┐
│  你的训练脚本 (SFTTrainer / 自定义 loop)        │
├──────────────────────────────────────────────┤
│  TRL (SFTTrainer)   PEFT (LoRA 插层)            │  ← 沿用生态
├──────────────────────────────────────────────┤
│            Unsloth 补丁层 (monkey-patch)        │
│  FastLanguageModel.from_pretrained(...)        │
│    把模型里的关键 forward 替换成 Triton 版       │
├──────────────┬──────────────┬─────────────────┤
│ Triton kernel│ Triton kernel│ Triton kernel   │
│   RMSNorm    │     RoPE     │  Cross-Entropy  │
│  (fused+bwd) │  (fused+bwd) │   (fused+bwd)   │
│   SwiGLU/MLP │  LoRA matmul │  (省激活检查点)   │
├──────────────┴──────────────┴─────────────────┤
│           PyTorch / CUDA / Triton runtime      │
└──────────────────────────────────────────────┘
```

**数据流（一次训练 step）**：

```
input_ids
   │  (FastLanguageModel 已把各层 forward 换成 Triton 版)
   ▼
[Embedding] → [N× Transformer Block] → [LM Head] → [Fused Cross-Entropy] → loss
                     │                                      │
        每个 Block 内: RMSNorm(融合) → Attention(RoPE融合)   │
                       → RMSNorm(融合) → SwiGLU MLP(融合)     │
                     LoRA 补丁挂在 q/k/v/o/gate/up/down 上    │
                                                             ▼
   loss.backward()  ←── 手写反向 kernel 直接给出梯度(不全靠 autograd)
   │  只更新 LoRA 的 A,B (其余冻结)
   ▼
 optimizer.step()
```

> 关键：你写的代码几乎和普通 HF/PEFT/TRL **一模一样**，只是入口换成 `unsloth.FastLanguageModel`。换言之，**学习成本极低，收益直接**。

---

## 4. 核心机制① —— 融合 kernel（fused kernels）

### 4.1 为什么"融合"会快

以 **RMSNorm** 为例。RMSNorm 把向量 $x$ 归一化：

$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_i x_i^2 + \epsilon}} \cdot g $$

朴素 PyTorch 写法会拆成多步：求平方 → 求均值 → 开方 → 除 → 乘 $g$。每一步是一个 kernel，每一步都要**把整个张量从 HBM 读出、算完写回**：

```
x ─读→[平方]─写→ HBM ─读→[均值]─写→ HBM ─读→[÷√]─写→ HBM ─读→[×g]─写→ HBM
     5 次往返显存, 4 个中间张量
```

Triton 融合版：把 $x$ 一行读进 SRAM，**平方、求和、开方、缩放一气呵成**，只写回最终结果：

```
x ─读→[ 平方+求和+开方+缩放 全在SRAM ]─写→ output
     1 次往返显存, 0 个落地中间张量
```

带宽往返从 5 次降到 1 次——这就是 2~3 倍加速的来源之一。

### 4.2 Unsloth 融合了哪些热点

| 算子 | 为什么是热点 | 融合后 |
|------|------------|--------|
| RMSNorm | 每层 2 次，全模型几十次 | 单 kernel 前向+反向 |
| RoPE 位置编码 | 每个 attention 都要旋转 Q/K | 融合进 attention 路径 |
| SwiGLU / MLP | `gate*silu(up)` 多步逐元素 | 融合逐元素 + matmul |
| Cross-Entropy | 词表大(10万+)，logits 巨大 | 融合 + 分块算，**不实例化整张 logits** |
| LoRA matmul | $W x + (BA)x$ | 融合缩放与加法 |

> Cross-Entropy 尤其关键：词表 V=128k、序列 4096 时，logits 张量是 `4096×128k` 个 float，约 **2GB**。融合 CE kernel **边算边规约**，不把整张 logits 落显存，省下巨量显存。

---

## 5. 核心机制② —— 手写反向（manual backward）

### 5.1 autograd vs 手写梯度

PyTorch 的 `autograd` 会**记录前向每一步**，反向时按链式法则自动回放。优点是通用，代价是：

1. 它要**保存很多中间张量**（为了反向能用）。
2. 它按"每个小算子"分别反传，又是一堆 kernel + HBM 往返。

Unsloth 对融合 kernel **亲手推导反向公式**，写成对应的 Triton 反向 kernel。这样：

- 反向也是**一个融合 kernel**，一次往返。
- 只保存**真正必要**的量（甚至能用前向输出反推，从而不存输入）。

### 5.2 手写反向一例（LoRA 的梯度）

LoRA 前向（忽略缩放）：$ y = Wx + B(Ax) $。$W$ 冻结，只对 $A,B$ 求梯度。设上游梯度为 $\delta = \dfrac{\partial \mathcal{L}}{\partial y}$，令 $h = Ax$：

$$ \frac{\partial \mathcal{L}}{\partial B} = \delta\, h^{\top}, \qquad \frac{\partial \mathcal{L}}{\partial A} = \big(B^{\top}\delta\big)\, x^{\top} $$

```
前向:  x ─[A]→ h ─[B]→ +Wx → y
反向:  δ ───────────────────┐
        ├─ dB = δ · hᵀ        │  (h 是前向算过的小矩阵, r 维, 很省)
        └─ B'δ ─[· xᵀ]→ dA    │  注意: W 冻结, 不需要 dW(省掉最大那块!)
```

> 重点：因为 $W$ 冻结，**完全不需要算也不需要存 $\partial\mathcal{L}/\partial W$**——而这本来是最大的一块。Unsloth 把这个事实"焊进"kernel，连带省掉相关中间量。这就是"为通用而保守"的 autograd 做不到的极致。

### 5.3 配合梯度检查点（gradient checkpointing）

激活值是显存大头。梯度检查点的思路：**前向时不存中间激活，反向时再重算一遍**——用算力换显存。Unsloth 提供优化版的检查点（把重算也走融合 kernel，并可把部分激活临时卸载到 CPU/内存），让"重算"的代价尽量小：

```
普通: 存下每层激活 → 显存大
检查点: 只存少数"检查点" → 反向时从检查点重算 → 显存小, 多花一点算力
Unsloth: 重算也用融合 kernel + 可选 offload → 省显存且重算不太慢
```

---

## 6. 核心机制③ —— QLoRA 与显存优化

QLoRA = **4bit 量化的基座权重 + LoRA 补丁**。基座 $W$ 用 4bit（NF4）存，前向时按需反量化成 bf16 参与计算，LoRA 的 $A,B$ 仍是高精度可训练。

```
权重存储:   [ 4bit NF4 量化 W ]   ← 体积 ≈ 全精度的 1/4
前向:       4bit ─反量化→ bf16 ─[matmul]→ 结果 + LoRA(BA)x
训练:       只更新 A,B (bf16), W 永不更新
```

Unsloth 在 QLoRA 上把**反量化也融合进 matmul kernel**，减少额外往返。综合"4bit 权重 + 小 LoRA 梯度/优化器态 + 融合 CE 不落 logits + 优化检查点"，把单卡可微调的模型规模显著拉大。

---

## 7. 速度与显存优势（数值，约 / 以官方为准）

> 以下为官方常引用的**量级**，具体随 GPU/序列长/batch/模型而变，**以官方 benchmark 为准**。

```
            训练速度            峰值显存(相对原版)
原版 HF+PEFT  1.0x  ████████      100%  ██████████
Unsloth(开源) ~2x   ████          ~40~50%  ████▌
长上下文场景  提升更明显(融合CE+检查点把显存压更狠 → 可上更长序列)
```

| 维度 | 量级（约） | 来源机制 |
|------|-----------|---------|
| 速度 | 单卡约 **2x**（开源版） | 融合 kernel + 手写反向，少 kernel/少 HBM 往返 |
| 显存 | 降低约 **50~70%** | 融合 CE 不落 logits + 优化检查点 + 4bit |
| 上下文 | 同卡可训**更长序列** | 激活/ logits 显存被压下来 |
| 精度 | **0 损失**（数学等价） | 手写 kernel 与原公式逐位/近似一致 |

> 注意：Unsloth 主打**单卡/消费级 GPU** 效率。多卡/分布式能力在**商业/Pro 版**或随版本演进，开源版以单卡为主——**以官方文档为准**。

---

## 8. 支持的模型（以官方列表为准）

Unsloth 为**主流开源 LLM/VLM**做了适配（因为要对每种结构手写 kernel/补丁）：

```
文本 LLM:  Llama 系列 / Mistral / Qwen 系列 / Gemma /
           Phi / DeepSeek / Yi / TinyLlama ...
多模态:    部分 VLM (如 Llava / Qwen-VL 类, 视版本)
量化:      4bit(QLoRA, NF4) / 16bit LoRA / 部分支持全量
任务:      SFT(指令微调) / 继续预训练 / DPO 等偏好对齐(配合 TRL)
```

> 模型支持是**逐个适配**的（不是任意模型都自动加速），新模型/新结构的支持随版本更新——**用前查官方支持列表**。

---

## 9. 与同类对比 & 何时用

```
                通用性   单卡速度  单卡省显存   多卡/规模   学习成本
HF PEFT+TRL      高★★★    ★         ★            中         低
Axolotl(配置式)   高★★★    ★         ★★           高★★★      中(YAML)
Unsloth          中(适配)  ★★★      ★★★          低(开源)    极低
```

| 选 Unsloth，当你… | 别用 / 谨慎，当你… |
|------------------|------------------|
| 单卡(消费级/单 A100)做 LoRA/QLoRA SFT | 需要大规模多卡分布式训练(开源版弱) |
| 想在 16/24GB 卡上微调 7B/13B | 你的模型结构 Unsloth 还没适配 |
| 追求最快迭代、最省显存、最低改动 | 需要训练框架级的复杂编排/特殊并行 |
| 跑长上下文 SFT | 需要改训练数学(它只做等价加速) |

> 一句话决策：**单卡 + LoRA/QLoRA + 主流模型 + 想又快又省又不掉点 → Unsloth 几乎是默认选择**；超出这个范围，回到 PEFT/TRL/Axolotl 或专业分布式框架。

---

## 数值例子 / 对照：24GB 卡能否 QLoRA 微调 7B？

**目标**：7B 模型（约 70 亿参数），QLoRA，序列 2048，看显存够不够。

**① 基座权重（4bit）**：
$$ 7\times10^9 \text{ 参数} \times 0.5\text{ Byte/参数(4bit)} \approx 3.5\ \text{GB} $$

**② LoRA 参数（r=16，作用于若干线性层）**：远小于基座，连同其梯度、优化器状态（AdamW 约 2 份状态）合计**约几百 MB**（量级 < 0.5GB）。

**③ 激活值**：这是变量。普通缓存全部激活会很大；开**梯度检查点**后大幅下降。设优化后约 **3~6GB**（随 batch/序列变）。

**④ 融合 Cross-Entropy 省下的**：若不融合，`2048×128k` logits ≈ `2048×128000×2B ≈ 0.5GB`（单步、单样本），batch 大时翻倍累加——融合 CE 把这块基本抹掉。

**合计（约）**：
```
权重 3.5 + (LoRA+优化器) 0.5 + 激活(检查点后) 3~6 + 缓冲 ≈ 8~12 GB
```

**结论**：**24GB 卡轻松容纳**，甚至能加大 batch 或拉长序列。若用原版 HF+PEFT（不融合 CE、激活更重），同配置可能逼近甚至超过 24GB——这正是 Unsloth 的实际意义：**让消费级显卡也能舒服地微调 7B/13B**。

> 数字为量级估算，便于建立直觉；精确值随实现/版本/配置变化，**以实测与官方为准**。

---

## 常见问题（FAQ）

| 问题 | 解答 |
|------|------|
| Unsloth 会降低模型精度吗？ | 不会。kernel 是原公式的**数学等价**实现，loss/梯度与原版一致（仅浮点级误差）。 |
| 它是新的微调算法吗？ | 不是。算法仍是 LoRA/QLoRA/DPO 等，Unsloth 只做**工程加速**。 |
| 为什么不直接用 PyTorch 编译(torch.compile)？ | compile 是通用图优化；Unsloth 针对 Transformer 微调**手写并手算反向**，省显存更狠、确定性更高。 |
| 多卡能用吗？ | 开源版以**单卡**为主；多卡/大规模能力看商业版或版本演进——**以官方为准**。 |
| 任意模型都能加速吗？ | 否。需对模型结构**逐个适配**，用前查官方支持列表。 |
| 代码改动大吗？ | 极小。入口换成 `FastLanguageModel.from_pretrained(...)`，其余沿用 TRL/PEFT。 |
| 和 PEFT 冲突吗？ | 不冲突，Unsloth 在 PEFT/TRL 之上打补丁，二者协同（见 [[ai-framework/huggingface-peft/README]]）。 |
| 省显存的最大功臣是谁？ | 融合 Cross-Entropy(不落整张 logits) + 优化的梯度检查点 + 4bit 权重。 |
| 速度提升的最大功臣是谁？ | 融合 kernel(少 HBM 往返) + 手写反向(少 kernel、不存多余激活)。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，返回总图
- [[ai-framework/huggingface-peft/README]] — PEFT/LoRA 的标准实现，Unsloth 在其上打补丁
- [[ai-framework/openai-triton/README]] — Triton 语言/编译器，Unsloth 手写 kernel 的工具
- [[llm-train/peft/PEFT-API]] — LoRA/QLoRA 的 API 用法与参数
