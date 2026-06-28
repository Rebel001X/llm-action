# PEFT 参数高效微调（Parameter-Efficient Fine-Tuning）总览

> 一句话定位：**冻结预训练大模型的绝大多数权重，只训练极少量（常 < 1%）新增/选中参数**，用 1 张消费级显卡也能把百亿模型微调到下游任务。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/peft/LoRA-QLoRA]]　[[llm-train/peft/Prompt-Tuning]]　[[llm-train/peft/Prefix-Tuning]]　[[llm-train/peft/PEFT-API]]　[[llm-train/README]]　[[llm-algo/mlp]]　[[llm-algo/transformer/模型架构]]

---

## 阅读地图

| 节 | 内容 | 你将学到 |
| --- | --- | --- |
| 0 | 一句话锚点 | PEFT 到底省了什么 |
| 1 | 地基：全量微调的代价 | 为什么 7B 全量微调要 ~112GB 显存 |
| 2 | PEFT 方法谱系（地图） | 加法 / 选择 / 重参数化 三大流派 |
| 3 | LoRA：低秩重参数化 | 旁路 $BA$、秩 $r$、$\alpha/r$ 缩放 |
| 4 | QLoRA：4-bit + LoRA | NF4 / 双重量化 / 分页优化器 |
| 5 | Adapter：插入小瓶颈层 | 串行/并行、瓶颈维 $m$ |
| 6 | Prefix / Prompt / P-Tuning | 软提示，连续向量当 token |
| 7 | IA³ / BitFit / $(IA)^3$ | 缩放与只调 bias 的极简法 |
| 8 | 显存账本 | 逐项手算谁占显存 |
| 9 | 数值手算合集 | LoRA 参数量、显存、压缩比 |
| 10 | 选型与坑 | 怎么选、合并、推理 |

---

## 0. 一句话锚点

把大模型权重 $W_0$ **锁住不动**（forward 仍用它），只学一个**很小的修正量** $\Delta W$ 或一小撮**插入参数**。
- 训练时：反向传播只更新这一小撮参数 → **优化器状态、梯度、激活全部缩小几个数量级**。
- 推理时：多数方法可把 $\Delta W$ **合并回** $W_0$，做到**零额外延迟**。

```
           全量微调                         PEFT（以 LoRA 为例）
   ┌──────────────────────┐        ┌──────────────────────┐
   │   W0  (全部可训练)    │        │   W0   ❄️ 冻结        │
   │   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  │        │   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  │
   │   ↑ 梯度+优化器 全量  │        │      + B·A  🔥 训练   │
   └──────────────────────┘        │      ░ (秩 r 很小)    │
     可训练参数 = 100%             └──────────────────────┘
                                     可训练参数 ≈ 0.1%~1%
```

---

## 1. 地基：全量微调到底贵在哪

要理解 PEFT 省了什么，先把**全量微调的显存账**摊开。设模型有 $\Phi$ 个参数，用 **Adam** 优化器、**混合精度（fp16/bf16）** 训练，显存四大块：

| 项 | 内容 | 每参数字节数 | 说明 |
| --- | --- | --- | --- |
| 模型权重 | fp16 一份 | 2 B | forward/backward 用 |
| 梯度 | fp16 一份 | 2 B | 每参数一个梯度 |
| 优化器状态 | Adam: fp32 权重副本 + 一阶矩 $m$ + 二阶矩 $v$ | 4+4+4 = 12 B | Adam 的"重灾区" |
| 激活值 | 与 batch、序列长、层数有关 | 单列 | 见第 8 节 |

> 经验公式（不含激活）：**全量 Adam 混合精度显存 ≈ $16\Phi$ 字节**（$2+2+12$）。

**手算（7B 模型）**：$\Phi = 7\times10^9$
$$ 16 \times 7\times10^9 = 1.12\times10^{11}\ \text{B} \approx 112\ \text{GB} $$
→ 单卡 A100-80G **装不下**，必须多卡 + ZeroOffload（见 [[ai-framework/deepspeed/README]]）。

**PEFT 的核心洞察**：上表中"梯度 + 优化器状态" = $14\Phi$ 字节是大头，而它们**只对可训练参数收取**。如果可训练参数从 $\Phi$ 降到 $0.005\Phi$，这 $14\Phi$ 直接缩 200 倍。

```
全量微调 7B 显存条形（约）        PEFT(LoRA) 7B 显存条形（约）
权重    ██ 14GB                   权重    ██ 14GB  (冻结，仍要存)
梯度    ██ 14GB                   梯度    · 0.07GB
优化器  ██████ 84GB               优化器  · 0.4GB
激活    ███ 可变                  激活    ███ 可变（基本不变）
总   ≈ 112GB+                    总   ≈ 14GB + 激活
```

> 注意：**权重那 14GB 省不掉**（forward 必须用）。所以 PEFT 砍的是"优化器+梯度"，不是权重。要再砍权重 → 量化（QLoRA，见第 4 节，见 [[llm-compression/quantization/量化基础]]）。

---

## 2. PEFT 方法谱系（一张地图）

按"参数从哪来"分三大流派：

```
                         PEFT
        ┌──────────────┬──────────────┬──────────────┐
        │  加法 Additive │ 选择 Selective │ 重参数 Reparam │
        │  (插新参数)    │  (只调原有子集) │ (低秩等价改造) │
   ┌────┴────┐     ┌────┴────┐      ┌────┴────┐
   │ Adapter │     │ BitFit  │      │  LoRA   │
   │ Prefix  │     │(只调bias)│      │  QLoRA  │
   │ Prompt  │     │ 部分层冻结│      │  DoRA   │
   │ P-Tuning│     │         │      │  AdaLoRA│
   │ IA³     │     └─────────┘      └─────────┘
   └─────────┘
```

| 流派 | 代表 | 参数加在哪 | 推理是否零延迟 |
| --- | --- | --- | --- |
| 加法-Adapter | Adapter、IA³ | Transformer 子层后插小模块 | 否（多算一层）/IA³近似零 |
| 加法-软提示 | Prefix/Prompt/P-Tuning | 在输入或每层 KV 前拼可学习向量 | 否（占序列/算力） |
| 选择 | BitFit | 原模型的 bias 等子集 | 是（无新结构） |
| 重参数化 | **LoRA / QLoRA** | 旁路低秩矩阵，可合并 | **是**（合并后） |

> LoRA 系是工业界默认首选，原因就是最后一列"可合并 → 推理零延迟"。

---

## 3. LoRA：低秩重参数化（重点）

**直觉**：微调对权重的改变量 $\Delta W$ 往往是**低秩**的（"内在维度低"）。既然低秩，就用两个瘦矩阵的乘积去近似它，不必学满秩。

对某个线性层 $W_0 \in \mathbb{R}^{d \times k}$，前向改成：
$$ h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\, B A\, x $$
其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d,k)$。
- **只训练 $A, B$**，$W_0$ 冻结。
- 初始化：$A \sim \mathcal{N}(0,\sigma^2)$，$B = 0$ → 训练开始时 $\Delta W = 0$，**不破坏预训练**。
- $\alpha/r$ 是缩放，把秩 $r$ 与学习率解耦（调 $r$ 时不必重调 lr）。

```
        输入 x (k 维)
          │
   ┌──────┴───────┐
   │              │
   ▼              ▼
┌──────┐     ┌─────┐  A: r×k  (降维到 r)
│  W0  │     │  A  │     r 很小, 比如 8
│ d×k  │     └──┬──┘
│ ❄冻结│        ▼
└──┬───┘     ┌─────┐  B: d×r  (升回 d), 初始=0
   │         │  B  │
   │         └──┬──┘
   │            │ ×(α/r)
   ▼            ▼
   └────►(＋)◄──┘
         │
         ▼  h = W0·x + (α/r)·B·A·x
```

**参数量**：原层 $d\times k$，LoRA 旁路 $= r(d+k)$。当 $r$ 小，$r(d+k) \ll dk$。

**加在哪些层？** 经典做法是注意力的 $W_q, W_v$（有时也 $W_k, W_o$）；现代实践常对所有线性层（含 MLP，见 [[llm-algo/mlp]]）都加，效果更稳。

---

## 4. QLoRA：4-bit 量化 + LoRA

LoRA 把"梯度+优化器"砍掉了，但**冻结权重那 14GB 还在**。QLoRA 把这份冻结权重**量化到 4-bit** 存储，让 65B 模型也能塞进单张 48GB 卡。三板斧：

1. **NF4（4-bit NormalFloat）**：针对正态分布权重设计的 4-bit 数据类型，分位点按正态分布密度划分，比普通 int4 更"贴合"权重分布，量化误差小。
2. **双重量化（Double Quant）**：连量化用的 scale 因子本身也再量化一次，平均每参数再省 ~0.37 bit。
3. **分页优化器（Paged Optimizer）**：优化器状态在显存不足时分页换出到内存（类似 OS 换页），抗 OOM 尖峰。

关键：**前向时把 4-bit 权重反量化回 bf16 参与计算，但梯度只流向 LoRA 的 $A,B$**（它们是 bf16/fp16）。所以"低精度只用于存储冻结权重，不用于训练参数"。

```
存储:  W0 ─量化→ NF4 (0.5 B/参数)     ← 省显存的关键
计算:  NF4 ─反量化→ bf16 → 参与 forward
训练:  梯度 ✗ 不进 W0
       梯度 ✓ 只进 LoRA 的 A,B (bf16)
```

> 详细原理见 [[llm-train/peft/LoRA-QLoRA]]；量化数据类型见 [[llm-compression/quantization/量化基础]]、[[llm-compression/quantization/fp8]]。

---

## 5. Adapter：插入小瓶颈层

在每个 Transformer 子层（注意力后、FFN 后）插入一个**瓶颈结构**：先降维到 $m$，非线性，再升回 $d$，外加残差。
$$ \text{Adapter}(x) = x + W_{\text{up}}\,\sigma(W_{\text{down}}\,x),\quad W_{\text{down}}\in\mathbb{R}^{m\times d},\ W_{\text{up}}\in\mathbb{R}^{d\times m} $$

```
   子层输出 x (d 维)
        │
        ├──────────────┐ (残差)
        ▼              │
   ┌─────────┐ down: d→m
   │  W_down │  (m 远小于 d)
   └────┬────┘
        ▼  σ (非线性)
   ┌─────────┐ up: m→d
   │  W_up   │
   └────┬────┘
        ▼
       (＋)◄──────────┘
        │
        ▼ 进入下一层
```

- 参数量：每个 Adapter $\approx 2md$（外加少量 bias/LayerNorm）。
- 缺点：**串行插入 → 推理多算一层，有延迟**（不像 LoRA 可合并）。这是它逐渐被 LoRA 取代的主因。

---

## 6. 软提示：Prefix-Tuning / Prompt-Tuning / P-Tuning

思路：不动模型权重，**学一段"连续向量"当虚拟 token** 拼在输入前，引导模型行为。区别在"拼在哪一层"。

| 方法 | 可学习参数拼在哪 | 直觉 |
| --- | --- | --- |
| **Prompt-Tuning** | 只在**输入嵌入层**前拼 $p$ 个软 token | 最轻量，模型越大越好用 |
| **P-Tuning (v1)** | 软 token + 一个小 LSTM/MLP 编码器生成它们 | 让软提示更"连贯" |
| **Prefix-Tuning / P-Tuning v2** | **每一层**的注意力 KV 前都拼前缀 | 更强，深层也受控 |

```
普通输入:        [tok1][tok2][tok3] ── 模型 ──► 输出
                                  ▲
软提示(Prompt):  [P1][P2]…[Pp][tok1][tok2][tok3]
                  └── 可学习, 其余冻结 ──┘

Prefix(每层):    Layer L 的注意力:
                 K = [ Pk ; K_real ]   ← 前缀 KV 可学习
                 V = [ Pv ; V_real ]
```

- 参数量极小：Prompt-Tuning 仅 $p \times d$（$p$=软 token 数，如 20，$d$=隐藏维）。
- 代价：软 token **占用序列长度/算力**，且效果对超参敏感，小模型上不如 LoRA 稳。
- 细节见 [[llm-train/peft/Prompt-Tuning]]、[[llm-train/peft/Prefix-Tuning]]。

---

## 7. 极简流派：BitFit / IA³

- **BitFit**：只训练模型里的 **bias 项**（约占总参数 0.1%），其余全冻。简单到极致，部分任务意外好用。
- **IA³**：为 key、value、FFN 中间激活各学一个**逐元素缩放向量** $l_k, l_v, l_{ff}$，即 $\text{attn}: K \leftarrow l_k \odot K$。参数量比 LoRA 还小，且缩放向量可乘进权重 → **近似零推理延迟**。

```
IA³:  K ─⊙ l_k→ K'      V ─⊙ l_v→ V'      FFN_mid ─⊙ l_ff→
      只学这几个向量(逐元素缩放), 维度 = d, 极小
```

---

## 8. 显存账本：逐项看谁占显存

PEFT 训练时显存分四块，对照第 1 节的全量版本看缩小幅度：

```
┌─ 冻结权重 W0 ────────────┐  LoRA: 2 B/参 (fp16)  ← 没省（QLoRA 才省到 0.5B）
├─ 可训练参数 (A,B) 副本 ──┤  2 B/参，但参数量极小
├─ 可训练参数 梯度 ────────┤  2 B/可训练参
├─ 优化器状态(Adam) ───────┤  12 B/可训练参  ← 全量时的重灾区，这里几乎归零
└─ 激活值 ────────────────┘  与全量基本相同（forward 路径没变）
```

**关键结论**：
- PEFT **省的是"梯度+优化器状态"**，因为它们按"可训练参数量"收费。
- PEFT **不省"激活值"**（forward 仍走全网络）。长序列/大 batch 时激活才是新瓶颈 → 配合梯度检查点 / FlashAttention（见 [[llm-optimizer/FlashAttention]]）。
- 想再省冻结权重那一份 → 上 QLoRA（4-bit）。

---

## 9. 数值手算合集

### 9.1 LoRA 可训练参数量

设隐藏维 $d=k=4096$，秩 $r=8$，对一层 $W_q$ 加 LoRA：
$$ \text{LoRA 参数} = r(d+k) = 8\times(4096+4096) = 65{,}536 $$
原层参数 $= d\times k = 4096^2 = 16{,}777{,}216$。
**压缩比** $= 16.7\text{M} / 65.5\text{K} \approx 256\times$。即这一层只训练原参数的 $0.39\%$。

### 9.2 全模型可训练比例（7B，只调 q,v）

LLaMA-7B：32 层，$d=4096$。每层给 $W_q,W_v$ 加 $r=8$ 的 LoRA：
$$ 2\ \text{矩阵} \times r(d+d) = 2 \times 8 \times 8192 = 131{,}072\ \text{参数/层} $$
$$ \times 32\ \text{层} = 4{,}194{,}304 \approx 4.2\text{M} $$
占比 $= 4.2\text{M} / 7\text{B} \approx 0.06\%$。**只训 0.06% 的参数。**

### 9.3 优化器显存对比

- **全量 Adam（fp32 状态）**：$7\text{B} \times 12\,\text{B} = 84\ \text{GB}$（仅优化器）。
- **LoRA Adam**：$4.2\text{M} \times 12\,\text{B} \approx 50\ \text{MB}$。
→ 优化器显存从 **84 GB → 0.05 GB**，缩 ~1680 倍。

### 9.4 QLoRA 冻结权重显存

7B 冻结权重：
- fp16 存：$7\text{B}\times 2 = 14\ \text{GB}$。
- NF4 存：$7\text{B}\times 0.5 = 3.5\ \text{GB}$（+双重量化再省一点）。
→ 配合 LoRA，**7B QLoRA 训练峰值常 < 8GB**，单张消费卡可跑。

### 9.5 $\alpha/r$ 缩放的意义

$r=8,\alpha=16$ → 缩放 $=16/8=2$。若把 $r$ 翻倍到 16 仍设 $\alpha=16$ → 缩放 $=1$。
作用：$\Delta W = \frac{\alpha}{r}BA$ 的"有效幅度"由 $\alpha$ 控制，**调 $r$（容量）时不必重调学习率**，工程上解耦了两个超参。

---

## 10. 选型与常见坑

| 场景 | 推荐 | 理由 |
| --- | --- | --- |
| 单卡微调大模型 | **QLoRA** | 4-bit 冻结权重，显存最省 |
| 多卡/显存够、要推理零延迟 | **LoRA**（合并） | 可 merge 回权重 |
| 多任务、热插拔 | LoRA 多适配器 | 同一底座挂多套 $A,B$，按需切 |
| 参数极致小 | IA³ / BitFit | 适配器体积 KB 级 |
| 仅需"提示风格"对齐 | Prompt/Prefix | 不改权重 |

---

## 常见问题

| 问题 | 解答 |
| --- | --- |
| LoRA 为什么不破坏预训练？ | $B$ 初始化为 0 → 起始 $\Delta W=0$，等价于原模型，再逐步学。 |
| 秩 $r$ 取多大？ | 常 4~64；任务难/数据多取大。先试 8/16。 |
| LoRA 推理慢吗？ | **合并后零延迟**（$W = W_0 + \frac{\alpha}{r}BA$ 算一次存好）；不合并则多两次小矩阵乘。 |
| QLoRA 推理也是 4-bit 吗？ | 训练 4-bit 存冻结权重；推理可合并 LoRA 后按需量化或还原 fp16，**以官方实现为准**。 |
| PEFT 省激活显存吗？ | 不省。激活由 forward 决定，PEFT 没改 forward 路径，需配梯度检查点/FlashAttention。 |
| Adapter 和 LoRA 区别？ | Adapter 串行插层、有推理延迟；LoRA 旁路低秩、可合并、零延迟 → LoRA 更主流。 |
| 能多个 LoRA 叠加吗？ | 能，按任务挂不同 $A,B$；也可加权融合做多任务（注意秩冲突）。 |
| 工具/库版本？ | HuggingFace `peft`、`bitsandbytes`、axololt 等，**具体 API 与默认值以官方文档为准**。 |

---

## 🔗 跳转链接

- 📍 总图：[[00-知识地图]]
- LoRA/QLoRA 深入：[[llm-train/peft/LoRA-QLoRA]]
- 软提示：[[llm-train/peft/Prompt-Tuning]]　[[llm-train/peft/Prefix-Tuning]]
- PEFT 接口实践：[[llm-train/peft/PEFT-API]]
- 训练总览：[[llm-train/README]]
- 模型结构（加在哪些层）：[[llm-algo/transformer/模型架构]]　[[llm-algo/mlp]]
- 量化（QLoRA 基础）：[[llm-compression/quantization/量化基础]]　[[llm-compression/quantization/fp8]]
- 显存/算力底座：[[llm-algo/FLOPs]]　[[ai-framework/deepspeed/README]]
- 激活优化配套：[[llm-optimizer/FlashAttention]]
- 工具：https://github.com/OpenAccess-AI-Collective/axolotl
