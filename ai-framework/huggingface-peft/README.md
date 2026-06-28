# HuggingFace PEFT

> 用「只训练一小撮新增/选中的参数」的方式微调大模型的统一库——把 LoRA / QLoRA / Prefix-Tuning / Prompt-Tuning / IA³ / (IA)³ 等一堆方法收进同一套 API，让你在一张消费级显卡上微调几十亿参数的模型。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-train/peft/Prompt-Tuning]] [[llm-train/peft/Prefix-Tuning]] [[ai-framework/huggingface-transformers/README]]

## 阅读地图

| 章节 | 你会得到什么 | 难度 |
| --- | --- | --- |
| 0. 一句话锚点 | PEFT 到底是什么，一行记住 | ★ |
| 1. 地基/前置 | 为什么「全量微调」会爆显存，PEFT 解决什么 | ★ |
| 2. 核心抽象 | `PeftModel` / `PeftConfig` / adapter 注入的统一框架 | ★★ |
| 3. LoRA | 低秩更新的数学与显存账（最重要的一节） | ★★★ |
| 4. QLoRA | 4-bit 量化基座 + LoRA，把显存再砍一截 | ★★★ |
| 5. Prefix / Prompt / P-Tuning | 软提示家族：往输入/KV 里塞可学习向量 | ★★ |
| 6. adapter 注入机制 | PEFT 是怎么「钻进」Transformers 模型的 | ★★★ |
| 7. 生态集成 | 与 Transformers / TRL / Accelerate / bitsandbytes 的关系 | ★★ |
| 8. 数值例子 | 7B 模型 LoRA/QLoRA 显存手算 | ★★★ |
| 对照表 / 常见问题 | 选型与踩坑 | ★★ |

---

## 0. 一句话锚点

> **PEFT = Parameter-Efficient Fine-Tuning（参数高效微调）。**
> 冻结预训练大模型的全部原始权重，只在旁边/前面挂一小撮「新参数」去训练。原始模型不动，新参数极小（常常是总量的 0.1%~2%），所以**显存、存储、训练成本全部暴跌**，而效果接近全量微调。

PEFT 库就是 HuggingFace 把这一类方法做成的**统一封装**：一行 `get_peft_model(model, config)` 就把任意 Transformers 模型改造成「只训小参数」的模式。

---

## 1. 地基/前置：它到底解决什么问题

### 1.1 全量微调为什么贵

先把「微调一次到底要存多少东西」拆到最原子。训练（用 Adam 优化器、混合精度）时，**每个可训练参数**要在显存里同时驻留：

```
每个参数的显存占用（Adam + fp16/bf16 混合精度，经典账）
┌─────────────────────────────────────────────┐
│  权重 weight (fp16)            → 2 bytes      │
│  梯度 grad   (fp16)            → 2 bytes      │
│  优化器: 权重 fp32 副本        → 4 bytes      │
│  优化器: Adam 一阶动量 m       → 4 bytes      │
│  优化器: Adam 二阶动量 v       → 4 bytes      │
├─────────────────────────────────────────────┤
│  合计 ≈ 16 bytes / 参数（不含激活值）         │
└─────────────────────────────────────────────┘
```

对一个 **7B（70 亿参数）** 模型做全量微调：

$$
7\times10^9 \times 16\ \text{bytes} = 112\times10^9\ \text{bytes} \approx 112\ \text{GB}
$$

——再加上前向激活值，单卡 80GB 都装不下。这还只是 7B；70B 直接没法玩。

### 1.2 PEFT 的核心洞察

研究（LoRA 论文等）发现：**大模型在「适应下游任务」时，权重的变化量 $\Delta W$ 是「低秩」的**——也就是说，让模型学会一个新任务并不需要改动那 112GB 里的每一个数，真正需要变化的「方向」很少。

于是思路变成：
- **冻结**原始权重 $W$（不算梯度、不进优化器，省掉上面 16 bytes 里的 14 bytes）。
- 只为那「一小撮真正要变的方向」引入新参数去训练。

可训练参数从 70 亿降到几百万，**优化器状态、梯度的显存几乎清零**，这就是 PEFT 省钱的根。

```
全量微调:  [████████████████████ 70 亿参数全部要算梯度+优化器状态]
PEFT:      [████████████████████ 70 亿冻结(只前向)] + [▏ 几百万可训练]
                    ↑ 不占梯度/优化器显存            ↑ 真正训练的部分
```

---

## 2. 核心抽象：PEFT 的统一框架

PEFT 把所有方法收敛到三个概念上，理解了它们就理解了整个库。

```
        ┌──────────────────────────────────────────────┐
        │              PeftConfig                        │
        │  「用哪种方法 + 超参」                          │
        │  例: LoraConfig(r=8, lora_alpha=16,            │
        │       target_modules=["q_proj","v_proj"])      │
        └───────────────────┬────────────────────────────┘
                            │  get_peft_model(base, config)
                            ▼
   base_model (Transformers) ───► ┌──────────────────────┐
   AutoModelForCausalLM 等         │     PeftModel         │
   (权重全部冻结 requires_grad=F)  │  = 基座 + 注入的adapter│
                                   │  只有 adapter 可训练   │
                                   └──────────┬────────────┘
                                              │ save_pretrained()
                                              ▼
                              只存 adapter 权重(几 MB ~ 几十 MB)
                              而不是整个几十 GB 的基座
```

- **`PeftConfig`**：声明式配置。每种方法一个子类（`LoraConfig` / `PrefixTuningConfig` / `PromptTuningConfig` / `IA3Config`…）。
- **`PeftModel`**：包住基座模型的壳。它负责把 adapter 注入、把基座参数冻结、并在 `forward` 时让 adapter 参与计算。
- **adapter 权重产物**：保存时**只保存新增的小参数**。一个 7B LoRA adapter 往往只有几十 MB，可以一个基座挂几十个 adapter，按需切换。

这就是「统一库」的价值：换方法 = 换 config 子类，训练循环、保存/加载、推理 API 全部不变。

---

## 3. LoRA：低秩更新（最核心的一节）

LoRA（Low-Rank Adaptation）是 PEFT 里最常用、最重要的方法。把它的数学讲到底。

### 3.1 它在改什么

Transformer 里大量是线性层 $h = Wx$，其中 $W$ 是一个 $d_{out}\times d_{in}$ 的大矩阵（例如注意力里的 $q\_proj, v\_proj$）。全量微调就是直接更新这个 $W$。

LoRA 不动 $W$，而是在旁边**并联**一条「低秩旁路」：

$$
h = Wx + \underbrace{\Delta W}_{\text{低秩}} x = Wx + B A x
$$

其中：
- $A$ 是 $r\times d_{in}$ 的矩阵，
- $B$ 是 $d_{out}\times r$ 的矩阵，
- $r$（秩，rank）是一个**很小**的数，典型取 $4, 8, 16, 32$。

「低秩」的意思就是：本来 $\Delta W$ 是 $d_{out}\times d_{in}$ 的大矩阵，现在用两个瘦长矩阵 $B$、$A$ 的乘积去逼近它。$BA$ 的秩最多是 $r$，所以叫「低秩更新」。

```
全量更新 ΔW (巨大):          LoRA 用 B·A 逼近 (两个小矩阵):

   d_in                          r          d_in
 ┌──────────┐               ┌──┐        ┌──────────┐
 │          │  d_out     d_ │  │   ×    │   A      │ r
 │  ΔW      │           out │ B│        └──────────┘
 │ (要学的) │               │  │
 └──────────┘               └──┘
 参数量 d_out×d_in          参数量 (d_out + d_in)×r  ← 小得多
```

### 3.2 参数量到底省多少（手算）

拿一个 $d_{in}=d_{out}=4096$ 的线性层（7B 模型注意力层量级）：

- 全量更新 $\Delta W$ 的参数量：$4096\times4096 = 16{,}777{,}216 \approx 1678\ \text{万}$。
- LoRA（$r=8$）的参数量：$(4096+4096)\times 8 = 65{,}536 \approx 6.5\ \text{万}$。

$$
\text{压缩比} = \frac{65536}{16777216} \approx 0.39\%
$$

**一层就省到 0.39%。** 全模型几百层叠加，可训练参数常常落在总参数的 0.1%~1%。

### 3.3 缩放因子 $\alpha$ 与初始化

LoRA 实际计算时还有一个缩放：

$$
h = Wx + \frac{\alpha}{r}\, B A x
$$

- $\alpha$（`lora_alpha`）是缩放超参，$\frac{\alpha}{r}$ 控制旁路的「话语权」。常见做法 $\alpha = 2r$ 或 $\alpha = r$。
- **初始化技巧**：$A$ 用随机高斯初始化，$B$ **初始化为 0**。于是训练刚开始 $BA=0$，旁路输出为 0，模型完全等价于原始预训练模型——这保证了「不破坏起点，从原模型平滑出发」。

### 3.4 训练完怎么用：可合并

推理时可以把旁路**合并**回主权重，得到一个普通模型，**零额外推理延迟**：

$$
W' = W + \frac{\alpha}{r} B A \quad\Rightarrow\quad h = W'x
$$

```
训练时(并联,有额外算路):        合并后(merge_and_unload):
       x                                x
       │                                │
   ┌───┴───┐                         ┌──┴──┐
   │ W(冻) │  ┌─B·A─┐                │  W' │  ← W' = W + (α/r)BA
   └───┬───┘  └──┬──┘                └──┬──┘
       └──(+)────┘                      │
          │                             ▼
          ▼                         一个普通模型,无旁路
```

这是 LoRA 相对「Adapter 串联层」的一大优势：**部署时可以零延迟**（不合并则保留切换灵活性，二者权衡）。

---

## 4. QLoRA：4-bit 量化基座 + LoRA

LoRA 把「可训练参数」砍没了，但**冻结的基座本身**仍要以 fp16 放在显存里——7B 基座光权重就 14GB。QLoRA 解决的就是这个。

### 4.1 三板斧

```
            QLoRA = 量化基座 + LoRA 旁路 + 显存优化
 ┌──────────────────────────────────────────────────────┐
 │ ① 4-bit NF4 量化:  基座权重从 fp16(2B) → 4-bit(0.5B)   │
 │    7B 基座: 14GB → ~3.5GB  (冻结,只读)                 │
 │ ② LoRA 旁路:       仍用 fp16/bf16 训练那一小撮新参数    │
 │ ③ 双重量化+分页优化器: 进一步抠显存,防 OOM 峰值         │
 └──────────────────────────────────────────────────────┘
```

- **NF4（4-bit NormalFloat）**：一种为「正态分布的权重」量身设计的 4-bit 数据类型，信息论上比普通 int4 更优。前向时反量化为 bf16 参与计算，存储时是 4-bit。
- **Double Quantization（双重量化）**：连量化用的「缩放常数」本身也再量化一次，进一步省一点。
- **Paged Optimizer（分页优化器）**：用类似操作系统虚拟内存的机制，把优化器状态在显存↔内存间分页，防止训练峰值瞬间 OOM。

### 4.2 关键直觉

- 基座只用来**前向**（提供特征），精度损失一点点没关系，量化到 4-bit 可接受。
- **真正学习的 LoRA 参数仍是高精度**，所以训练质量基本不掉。
- 这套组合让 **单张 24GB / 48GB 卡微调 33B、65B** 成为可能（数值见第 8 节）。

> QLoRA 在 PEFT 里就是「用 bitsandbytes 以 4-bit 加载基座 + 正常套 `LoraConfig`」，库层面几乎无缝，具体加载参数以官方文档为准。

---

## 5. 软提示家族：Prefix / Prompt / P-Tuning

LoRA 改的是「权重」；软提示家族改的是「输入/上下文」——往模型里塞**可学习的向量**，基座一个参数都不动。详见 [[llm-train/peft/Prompt-Tuning]] 与 [[llm-train/peft/Prefix-Tuning]]。

```
普通提示(离散token,人写):   "请翻译成法语:" → 查表得到固定 embedding
软提示(连续向量,机器学):    [v1][v2]...[vk] → 可训练向量,梯度直接优化它们
                            (没有对应的真实单词,纯粹是被学出来的向量)
```

| 方法 | 可学习向量塞在哪 | 直觉 |
| --- | --- | --- |
| **Prompt-Tuning** | 只在**输入 embedding 最前面**拼一段软 token | 最轻量；像学一个「最佳开场白」，但开场白是向量不是文字 |
| **Prefix-Tuning** | 在**每一层**的注意力 Key/Value 前面都拼一段可学前缀 | 比 Prompt-Tuning 表达力强，因为每层都能调控 |
| **P-Tuning / v2** | 用一个小网络生成软提示 / 多层插入 | 介于二者之间，提升小模型与难任务上的稳定性 |

```
Prefix-Tuning 示意（每层注意力的 KV 前面挂可学前缀）

   第 L 层注意力:
   K = [ P_k(可学前缀) | K_real(由输入算出) ]
   V = [ P_v(可学前缀) | V_real ]
              ↑
        只有 P_k, P_v 训练，基座全冻结
```

**特点**：软提示家族可训练参数往往比 LoRA 还少；但一般任务效果上 LoRA（尤其在生成/对齐任务）更稳更强，所以工业界主力是 LoRA/QLoRA，软提示更多用于轻量任务与研究对比。

---

## 6. adapter 注入机制：PEFT 是怎么「钻进」模型的

这是理解 PEFT 工程实现的关键。`get_peft_model` 做的核心动作是**模块替换 / 包裹**。

```
原始 Transformers 模型的某一层:
   self.q_proj = nn.Linear(4096, 4096)      ← 一个普通线性层

get_peft_model 之后(以 LoRA 为例):
   self.q_proj = lora.Linear(               ← 被替换成 LoRA 版线性层
        base_layer = 原 nn.Linear(冻结),
        lora_A     = nn.Linear(4096, r, bias=False),   ← 新,可训练
        lora_B     = nn.Linear(r, 4096, bias=False),   ← 新,可训练(初始 0)
   )
```

注入流程：

```
get_peft_model(model, LoraConfig(target_modules=["q_proj","v_proj"]))
        │
        ├─ 1. 遍历模型所有子模块,按名字匹配 target_modules
        │      (用正则/后缀匹配 "q_proj","v_proj" 等)
        │
        ├─ 2. 把命中的 nn.Linear 包成 LoRA 层(原层做 base_layer 冻结)
        │
        ├─ 3. 把基座所有参数 requires_grad = False
        │      只有 lora_A / lora_B 保持 requires_grad = True
        │
        └─ 4. 返回 PeftModel,其 forward 自动让旁路参与
```

`target_modules` 决定「给哪些层装旁路」——这是 LoRA 最重要的旋钮之一。常见选择：只装注意力的 `q_proj,v_proj`（最省），或扩展到 `k_proj,o_proj` 甚至 MLP 的 `gate/up/down_proj`（更强但参数更多）。

**为什么这种设计很优雅**：它不改 Transformers 源码，纯粹在运行时替换模块，所以**任何**符合命名规范的 Transformers 模型都能即插即用。

---

## 7. 生态集成：PEFT 在 HuggingFace 全家桶里的位置

PEFT 不是孤立的，它是黏在整个 HF 栈中间的一层。

```
┌──────────────────────────────────────────────────────────────┐
│  数据/任务层                                                    │
│   datasets ── 提供训练数据                                      │
├──────────────────────────────────────────────────────────────┤
│  训练编排层                                                     │
│   TRL (SFTTrainer/DPOTrainer/PPOTrainer/GRPO)                   │
│     │  直接吃 peft_config 参数,内部帮你 get_peft_model          │
│   Transformers Trainer ── 通用训练循环                          │
├──────────────────────────────────────────────────────────────┤
│  ★ PEFT 层 ★   get_peft_model / PeftModel                       │
│     把基座改造成「只训 adapter」                                 │
├──────────────────────────────────────────────────────────────┤
│  模型层                                                         │
│   Transformers (AutoModelForCausalLM 等) ← [[..huggingface-transformers/README]] │
├──────────────────────────────────────────────────────────────┤
│  底层加速                                                       │
│   bitsandbytes(4/8-bit 量化, 喂给 QLoRA) │ Accelerate(多卡/分布式)│
└──────────────────────────────────────────────────────────────┘
```

- **与 Transformers**：PEFT 直接包住 `AutoModelForCausalLM` 等基座；保存 adapter 后，推理时 `PeftModel.from_pretrained(base, adapter_path)` 即可挂回。详见 [[ai-framework/huggingface-transformers/README]]。
- **与 TRL**：TRL 的各种 Trainer（`SFTTrainer`、`DPOTrainer` 等）都接受一个 `peft_config` 参数，传进去它就自动用 LoRA/QLoRA 训练——**RLHF / SFT 里用 LoRA 几乎是默认操作**，是省显存训对齐模型的标配。
- **与 Accelerate**：PEFT 模型照样能被 Accelerate 包起来做多卡/FSDP/DeepSpeed 分布式（注意 QLoRA + FSDP 有一些已知配合细节，以官方文档为准）。
- **与 bitsandbytes**：QLoRA 的 4-bit 加载靠它。

一句话：**PEFT 负责「怎么少训参数」，Transformers 负责「模型本体」，TRL 负责「训练算法（SFT/DPO/PPO）」，bitsandbytes/Accelerate 负责「省显存/上多卡」。**

---

## 数值例子 / 典型场景

### 例 1：7B 模型，全量 vs LoRA vs QLoRA 显存对比（粗算，仅模型相关项）

设 7B = $7\times10^9$ 参数，注意力相关线性层约占可注入位置，LoRA 取 $r=8$ 装在 q/v 上，可训练参数约 **4M（400 万）**。

| 项目 | 全量微调(fp16+Adam) | LoRA(fp16 基座) | QLoRA(4-bit 基座) |
| --- | --- | --- | --- |
| 基座权重 | 7B×2B = **14 GB** | 7B×2B = **14 GB** | 7B×0.5B ≈ **3.5 GB** |
| 基座梯度 | 7B×2B = **14 GB** | 0（冻结） | 0（冻结） |
| 基座优化器状态(fp32×3) | 7B×12B = **84 GB** | 0 | 0 |
| adapter 权重+梯度+优化器 | — | 4M×16B ≈ **0.06 GB** | 4M×16B ≈ **0.06 GB** |
| **模型相关合计** | **≈ 112 GB** | **≈ 14 GB** | **≈ 3.6 GB** |

> 结论：全量微调要 A100-80G ×多卡；LoRA 一张 24GB 卡就够；QLoRA 甚至能在更小卡上跑更大的模型。（激活值另算，受 batch/序列长度影响。）

### 例 2：可训练参数占比

7B 模型用 LoRA(r=8, 只装 q/v)，可训练 ≈ 4M：

$$
\frac{4\times10^6}{7\times10^9} \approx 0.057\%
$$

**只训 0.057% 的参数**，就能把模型适配到一个新任务/新风格——这就是 PEFT 的威力。

### 例 3：adapter 体积与多任务部署

一个 7B LoRA adapter 落盘常常只有 **十几~几十 MB**。于是一种典型架构：

```
            一份 7B 基座(只加载一次, 14GB)
                       │
    ┌──────────┬───────┼───────┬──────────┐
 客服adapter  翻译adapter  代码adapter  法律adapter
  (30MB)      (30MB)       (30MB)      (30MB)
    └──── 推理时按请求动态切换/热插拔 adapter ────┘
```

省显存（基座只一份）、省存储（每任务才几十 MB）、上线快（换 adapter 不重载基座）。

---

## 对照表（与同类对比）

| 方法 | 改什么 | 可训练参数量级 | 推理额外延迟 | 典型效果 | 何时选 |
| --- | --- | --- | --- | --- | --- |
| **全量微调** | 全部权重 | 100% | 无 | 上限最高 | 数据多、显存充足、要极致效果 |
| **LoRA** | 选中线性层的低秩旁路 | 0.1%~1% | 可合并→0 | 接近全量 | **默认首选**，性价比之王 |
| **QLoRA** | LoRA + 4-bit 基座 | 0.1%~1% | 同 LoRA | 略低于 LoRA | 显存紧、模型大（33B/65B 单卡） |
| **Prefix-Tuning** | 每层 KV 前缀 | <0.1% | 有（前缀占上下文） | 中 | 不想动权重、轻量任务 |
| **Prompt-Tuning** | 输入软 token | 极小 | 有（占少量上下文） | 大模型上不错 | 最轻量、多任务共享基座 |
| **IA³** | 用学到的向量缩放激活 | 极小 | 可合并 | 中上 | 比 LoRA 还省、少样本 |
| **Adapter(串联)** | 层间插小 MLP | 1%~5% | 有（串联不可消） | 中上 | 经典方法，现多被 LoRA 取代 |

---

## 常见问题（表格）

| 问题 | 答案 |
| --- | --- |
| LoRA 的 `r` 怎么选？ | 从 8/16 起步；任务难、数据多就加大（32/64）。越大越接近全量但越占参数 |
| `lora_alpha` 怎么配？ | 常用 $\alpha=2r$ 或 $\alpha=r$；它通过 $\alpha/r$ 缩放旁路强度 |
| `target_modules` 装哪些层？ | 最省：只装 `q_proj,v_proj`；要更强：加 `k/o_proj` 和 MLP 的 `gate/up/down_proj` |
| LoRA 和全量微调差多少？ | 多数任务非常接近；数据极多/任务极难时全量仍有微弱上限优势 |
| QLoRA 会掉精度吗？ | 基座 4-bit 量化带来轻微损失，但 LoRA 参数仍高精度，总体质量基本持平 |
| 训练完一定要 merge 吗？ | 不必。merge 后零延迟但失去切换灵活性；不 merge 则可热插拔多 adapter |
| 一个基座能挂多个 adapter 吗？ | 能。PEFT 支持加载/切换多个 adapter，适合多任务部署 |
| 为什么 $B$ 初始化为 0？ | 让训练起点 $BA=0$，模型等价于原始模型，从原模型平滑出发不破坏起点 |
| PEFT 必须配 TRL 吗？ | 不必。可直接配 Transformers `Trainer`；但做 SFT/DPO/PPO 时配 TRL 最省事 |
| adapter 文件为什么这么小？ | 只存新增的低秩矩阵（几百万参数），不存几十 GB 的冻结基座 |

> 注：具体 API 签名、加载参数、版本特性请以 HuggingFace PEFT 官方文档为准；本文聚焦稳定的原理与机制。

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局知识地图，返回总览
- [[llm-train/peft/Prompt-Tuning]] — 软提示之 Prompt-Tuning 细节
- [[llm-train/peft/Prefix-Tuning]] — 软提示之 Prefix-Tuning 细节
- [[ai-framework/huggingface-transformers/README]] — PEFT 改造的基座来自这里
