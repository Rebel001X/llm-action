# LoRA 与 QLoRA：低秩适配与量化微调

> 一句话定位：**冻结大模型原权重，只训练一对小的低秩矩阵 $A,B$（LoRA）；再把冻结的底座压成 4bit 省显存（QLoRA）**——这是单卡/消费级显卡微调百亿模型的主力方案。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/huggingface-peft/README]] · [[llm-train/peft/PEFT-API]] · [[llm-compression/quantization/量化基础]] · [[llm-train/README]]

---

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|-------------|--------|
| 0 | 一句话锚点 | 冻结 + 低秩 + 量化 |
| 1 | 地基：全量微调贵在哪 | 优化器状态、显存账 |
| 2 | LoRA 数学原理 | $W_0 + \frac{\alpha}{r}BA$ |
| 3 | LoRA 三个超参怎么选 | r / alpha / target_modules |
| 4 | 代码：Baichuan2 LoRA 配置 | `LoraConfig` / `get_peft_model` |
| 5 | QLoRA：4bit 量化 + LoRA | NF4 / 双量化 / 分页优化器 |
| 6 | 4bit/8bit/16bit 线性层对照 | `Linear4bit` / `Linear8bitLt` |
| 7 | 实操：加载量化模型 | `BitsAndBytesConfig` |
| 8 | 调试：确认只有 LoRA 在算梯度 | `requires_grad` / `grad` |
| — | 常见坑 + 跳转 | — |

---

## 0. 一句话锚点

- **全量微调**：更新模型全部 $N$ 个参数，显存 ≈ 权重 + 梯度 + 优化器状态，约 **16×参数量字节**（fp16 + Adam）。70 亿参数要 >100GB 显存。
- **LoRA**：把每个要适配的权重矩阵 $W$ 的更新约束成低秩 $\Delta W = BA$，**只训练 $A,B$（占总参数 0.1%~1%）**，原权重 $W_0$ 全程冻结。
- **QLoRA**：在 LoRA 基础上，把冻结的底座 $W_0$ **量化成 4bit（NF4）** 存放，前向时反量化回 16bit 算，再叠 LoRA。显存再降一半多——**单张 24GB 卡可微调 33B，48GB 可微调 65B**。

一句话因果链：`贵 → 只训低秩(LoRA) → 底座还占显存 → 把底座压4bit(QLoRA)`。

---

## 1. 地基：全量微调的显存账（为什么需要 PEFT）

训练一个参数量为 $N$、用 fp16 + Adam 的模型，显存粗略分解：

```
显存 = 权重        : 2N 字节  (fp16)
     + 梯度        : 2N 字节  (fp16, 每个可训练参数一份)
     + Adam 一阶动量: 4N 字节  (fp32 m)
     + Adam 二阶动量: 4N 字节  (fp32 v)
     + fp32 权重副本: 4N 字节  (混合精度主权重)
     ─────────────────────────
     ≈ 16N 字节  + 激活值
```

**数值手算**：7B 模型 → $16 \times 7\times10^9 = 1.12\times10^{11}$ 字节 ≈ **112 GB**，单卡放不下。

PEFT（Parameter-Efficient Fine-Tuning）的核心洞察：**梯度/优化器状态只对“可训练参数”才有**。如果可训练参数从 7B 降到 7M（千分之一），那后面 4 项（梯度 + 两个动量 + fp32 副本，共 14N）几乎全部归零，只剩“权重 2N”这一项是大头。

```
全量:  权重2N + 梯度2N + Adam 8N + fp32副本4N        ← 14N 全压在 7B 上
LoRA:  权重2N(冻结) + (梯度+Adam+副本)只压在7M上 ≈ 0  ← 大头只剩权重
QLoRA: 权重压到 0.5N(4bit) + LoRA那一点点          ← 权重也被砍掉3/4
```

---

## 2. LoRA 数学原理：低秩更新

### 2.1 核心假设

论文（LoRA, Hu et al. 2021）的假设：**微调时权重的“变化量” $\Delta W$ 具有很低的内在秩（intrinsic rank）**。即虽然 $W \in \mathbb{R}^{d\times k}$ 很大，但适配某个下游任务真正需要的变化 $\Delta W$ 可以用一个秩为 $r \ll \min(d,k)$ 的矩阵很好地近似。

于是把 $\Delta W$ 分解为两个瘦长矩阵相乘：

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}\, B A
$$

其中：
- $W_0 \in \mathbb{R}^{d\times k}$：原始预训练权重，**冻结，不更新**。
- $B \in \mathbb{R}^{d\times r}$，$A \in \mathbb{R}^{r\times k}$：**只训练这两个**。
- $r$：秩（rank），通常 4/8/16/64。
- $\alpha$：缩放系数，实际缩放因子是 $\frac{\alpha}{r}$。

### 2.2 ASCII 图：前向通路

```
            输入 x  (维度 k)
              │
      ┌───────┴────────┐
      │                │
   ┌──▼───┐        ┌───▼────┐
   │  W0  │        │   A    │  r×k  (训练)   ← 先降维到 r
   │ d×k  │        └───┬────┘
   │冻结  │            │ (维度 r, 很小)
   └──┬───┘        ┌───▼────┐
      │            │   B    │  d×r  (训练)   ← 再升回 d
      │            └───┬────┘
      │                │ × (α/r)  缩放
      │                │
      └───────┬────────┘
           (相加)
              │
              ▼
        输出 h = W0·x + (α/r)·B·A·x
```

- 主干 $W_0 x$ 和旁路 $\frac{\alpha}{r}BAx$ **并行**计算后相加。
- 参数量对比：$W_0$ 有 $d\times k$ 个；LoRA 旁路只有 $r(d+k)$ 个。例如 $d=k=4096, r=8$：$W_0$=1677万，LoRA=6.6万，**仅 0.39%**。

### 2.3 初始化（为什么训练一开始不影响模型）

- $A$ 用高斯随机初始化，$B$ 用 **全零** 初始化。
- 于是初始 $\Delta W = B A = 0$，即训练第 0 步模型输出 **完全等于原模型**，从一个无损起点开始学习，训练更稳。

### 2.4 推理时零延迟（合并权重）

训练完可以把旁路合并回主干：

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r}BA
$$

合并后变回一个普通的 $W_{\text{merged}}$，推理时**没有任何额外算子、零延迟**。这是 LoRA 相比 Adapter（串行插层，推理变慢）的关键优势。也因此可以为不同任务存多套小 $A,B$，按需热插拔。

---

## 3. LoRA 三个核心超参怎么选

| 超参 | 含义 | 经验取值 | 调大的影响 |
|------|------|---------|-----------|
| `r` (rank) | 低秩维度，旁路容量 | 8 / 16 / 64 | 越大越能拟合复杂任务，参数与显存线性增加；过大易过拟合 |
| `lora_alpha` | 缩放，实际系数 $\alpha/r$ | 常设为 `2×r` 或 16/32 | 放大 LoRA 影响力；常与 `r` 联动，固定 $\alpha/r$ 比值 |
| `target_modules` | 给哪些线性层加 LoRA | 注意力的 q/k/v/o；MLP；或模型特定层 | 覆盖越多层效果越好但参数越多 |
| `lora_dropout` | 旁路 dropout，防过拟合 | 0.05 / 0.1 | 增强正则 |

> **`target_modules` 是模型相关的**：要填模型里真实存在的线性层名字。Baichuan 把 q/k/v 融合成了一个名为 `W_pack` 的层；LLaMA 则是 `q_proj/k_proj/v_proj/o_proj`；Bloom 是 `query_key_value/dense`（见第 6 节打印出的结构）。

---

## 4. 代码：Baichuan2 的 LoRA 配置（原文真料）

> 来源：`https://github.com/baichuan-inc/Baichuan2/blob/main/fine-tune/fine-tune.py`

```python
from peft import LoraConfig, TaskType, get_peft_model

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # 因果语言模型（GPT 类自回归）
    target_modules=["W_pack"],      # Baichuan 把 q/k/v 融合成 W_pack 一层
    inference_mode=False,           # 训练模式（True 则只推理、冻结 LoRA）
    r=1,                            # 秩，这里取了极小值 1
    lora_alpha=32,                  # 缩放，实际系数 α/r = 32/1 = 32
    lora_dropout=0.1,
)
model.enable_input_require_grads()  # 关键：让输入 require_grad，配合梯度检查点
model = get_peft_model(model, peft_config)   # 把原模型包成带 LoRA 旁路的模型
model.print_trainable_parameters()  # 打印可训练参数占比，验证 PEFT 生效
```

**逐行为什么**：
- `r=1, lora_alpha=32` → 缩放因子 $\alpha/r = 32$ 很大，用极小的秩配很大的缩放，是一种省到极致的配置。
- `enable_input_require_grads()`：当用 **梯度检查点（gradient checkpointing）** 时，冻结的 embedding 输出默认 `requires_grad=False`，会导致反向传播链断掉、LoRA 收不到梯度。这行手动给输入打开梯度，**是 PEFT + 梯度检查点的必备搭配**，漏了会报“没有需要梯度的张量”。
- `print_trainable_parameters()` 典型输出形如 `trainable params: X || all params: Y || trainable%: 0.0x%`，**务必看一眼**确认占比是千分级而非 100%（100% 说明 LoRA 没挂上）。

---

## 5. QLoRA：4bit 量化 + LoRA

LoRA 把“梯度/优化器”那部分显存砍没了，但**冻结的底座权重 $W_0$ 仍以 fp16 占着 2N 字节**。QLoRA（Dettmers et al. 2023）进一步把这块也压掉。

### 5.1 三大技术点

| 技术 | 作用 | 直觉 |
|------|------|------|
| **NF4（4-bit NormalFloat）** | 把冻结权重存成 4bit | 针对“权重近似正态分布”设计的 4bit 数据类型，信息论上对正态数据最优，比普通 int4/fp4 精度更高 |
| **双量化 Double Quantization** | 把“量化用的常数”也量化 | 每块有个 fp32 缩放常数，数量大；再用 8bit 量化这些常数，每参数再省约 0.37 bit |
| **分页优化器 Paged Optimizer** | 防显存尖峰 OOM | 长序列时优化器状态用 NVIDIA 统一内存，显存不够时自动换页到 CPU，避免崩溃 |

### 5.2 数据流 ASCII 图

```
存储:  W0 以 NF4(4bit) 躺在显存          ← 省 3/4
          │
   前向时 │ 反量化 (dequantize → fp16/bf16)
          ▼
       W0_fp16 ──┐
                 ├──► h = W0·x + (α/r)·B·A·x
   LoRA  A,B ────┘        (A,B 始终是 16bit, 可训练)
          ▲
   反向时 │ 梯度只流向 A,B；W0 不更新(它只是 4bit 只读底座)
```

关键点：**量化只作用在冻结的 $W_0$ 上**；LoRA 的 $A,B$ 永远是 16bit 全精度，所以训练精度损失很小，论文显示 QLoRA 能基本匹配 16bit 全量微调效果。

### 5.3 显存对比（数值直觉，7B 为例）

| 方案 | 底座权重 | 梯度+优化器 | 量级 |
|------|---------|------------|------|
| 全量 fp16 + Adam | 14 GB | ~84 GB | >100 GB |
| LoRA (fp16 底座) | 14 GB | ~0（只压在 LoRA 上） | ~15 GB |
| **QLoRA (4bit 底座)** | **~3.5 GB** | ~0 | **<10 GB，单卡可跑** |

---

## 6. 4bit / 8bit / 16bit 对应的线性层类（原文真料）

加载时通过参数 `load_in_4bit` / `load_in_8bit` / 默认，`transformers` + `bitsandbytes` 会把模型里的 `nn.Linear` 自动替换成不同的量化线性层类。下面用同一个 `bloom-2b6-zh` 打印 `print(model)` 对比三者，**注意看注意力里 `query_key_value`/`dense` 的类名变化**：

```
load_in_4bit=True  →  Linear4bit       (4bit 存储，QLoRA 用这个)
load_in_8bit=True  →  Linear8bitLt     (LLM.int8() 的 8bit)
默认 (fp16)        →  Linear           (普通全精度线性层)
```

### 6.1 4bit：`Linear4bit`（QLoRA 配置）

```python
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)

device_map = {'': 0}
model = AutoModelForCausalLM.from_pretrained(
    "/home/guodong.li/workspace/model/bloom-2b6-zh",
    device_map=device_map,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,   # 反量化后用 fp16 计算
        bnb_4bit_use_double_quant=True,         # 开启双量化，再省一点显存
        bnb_4bit_quant_type="nf4",              # 用 NF4 数据类型（QLoRA 推荐）
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    ),
)

print(model)
```

打印结果（注意 `Linear4bit`）：

```
BloomForCausalLM(
  (transformer): BloomModel(
    (word_embeddings): Embedding(46145, 2560)
    (word_embeddings_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
    (h): ModuleList(
      (0): BloomBlock(
        (input_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (self_attention): BloomAttention(
          (query_key_value): Linear4bit(in_features=2560, out_features=7680, bias=True)
          (dense): Linear4bit(in_features=2560, out_features=2560, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (mlp): BloomMLP(
          (dense_h_to_4h): Linear4bit(in_features=2560, out_features=10240, bias=True)
          (gelu_impl): BloomGelu()
          (dense_4h_to_h): Linear4bit(in_features=10240, out_features=2560, bias=True)
        )
      )
     (29): BloomBlock(
       ...
      )
    )
    (ln_f): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2560, out_features=46145, bias=False)
)
```

> 各参数含义：
> - `bnb_4bit_compute_dtype=fp16`：权重存 4bit，但实际矩阵乘是把它反量化成 fp16 再算，**存储精度 ≠ 计算精度**。
> - `bnb_4bit_use_double_quant=True`：双量化，对应 5.1 节，每参数再省 ~0.37 bit。
> - `bnb_4bit_quant_type="nf4"`：可选 `"nf4"`（NormalFloat，QLoRA 推荐）或 `"fp4"`。
> - `llm_int8_threshold` / `llm_int8_has_fp16_weight`：是 8bit（LLM.int8）路径的离群值阈值参数，4bit 路径里通常不起主导作用。
> - 注意 `lm_head` 仍是普通 `Linear`（输出头一般不量化，保精度）。

### 6.2 8bit：`Linear8bitLt`

```python
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)

device_map = {'': 0}
model = AutoModelForCausalLM.from_pretrained(
    "/home/guodong.li/workspace/model/bloom-2b6-zh",
    device_map=device_map,
    load_in_8bit=True,
    torch_dtype=torch.float16,
)

print(model)
```

打印结果（注意 `Linear8bitLt`）：

```
BloomForCausalLM(
  (transformer): BloomModel(
    (word_embeddings): Embedding(46145, 2560)
    (word_embeddings_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
    (h): ModuleList(
      (0): BloomBlock(
        (input_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (self_attention): BloomAttention(
          (query_key_value): Linear8bitLt(in_features=2560, out_features=7680, bias=True)
          (dense): Linear8bitLt(in_features=2560, out_features=2560, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (mlp): BloomMLP(
          (dense_h_to_4h): Linear8bitLt(in_features=2560, out_features=10240, bias=True)
          (gelu_impl): BloomGelu()
          (dense_4h_to_h): Linear8bitLt(in_features=10240, out_features=2560, bias=True)
        )
      )
      ...
      (29): BloomBlock(
        ...
      )
    )
    (ln_f): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2560, out_features=46145, bias=False)
)
```

### 6.3 16bit：普通 `Linear`（无量化基线）

```python
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)

device_map = {'': 0}
model = AutoModelForCausalLM.from_pretrained(
    "/home/guodong.li/workspace/model/bloom-2b6-zh",
    device_map=device_map,
    torch_dtype=torch.float16,
)

print(model)
```

打印结果（普通 `Linear`，没有量化后缀）：

```
BloomForCausalLM(
  (transformer): BloomModel(
    (word_embeddings): Embedding(46145, 2560)
    (word_embeddings_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
    (h): ModuleList(
      (0): BloomBlock(
        (input_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (self_attention): BloomAttention(
          (query_key_value): Linear(in_features=2560, out_features=7680, bias=True)
          (dense): Linear(in_features=2560, out_features=2560, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (mlp): BloomMLP(
          (dense_h_to_4h): Linear(in_features=2560, out_features=10240, bias=True)
          (gelu_impl): BloomGelu()
          (dense_4h_to_h): Linear(in_features=10240, out_features=2560, bias=True)
        )
      )
      (29): BloomBlock(
        ...
      )
    )
    (ln_f): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2560, out_features=46145, bias=False)
)
```

### 6.4 三者对照表

| 加载方式 | 线性层类名 | 每权重存储 | 用途 | 配套 |
|---------|-----------|-----------|------|------|
| `load_in_4bit=True` | `Linear4bit` | 4 bit (NF4) | **QLoRA 微调**、超省显存推理 | `BitsAndBytesConfig(nf4, double_quant)` |
| `load_in_8bit=True` | `Linear8bitLt` | 8 bit (LLM.int8) | 8bit 推理 / LoRA 微调 | 离群值用 fp16 旁路保精度 |
| 默认 fp16 | `Linear` | 16 bit | 全精度基线 | 显存翻几倍 |

> 从这三段打印能直观看到：**量化只是把 `Linear` 换成 `Linear4bit/Linear8bitLt`，模型结构（层数、维度 2560、29 个 BloomBlock）完全不变**，换的只是这些层内部权重的存储与计算方式。这也是为什么 QLoRA 能无缝套用到任意 HF 模型。

> 在量化模型上挂 LoRA 前，通常先调 `prepare_model_for_kbit_training(model)`（上面 import 已引入）：它会把 LayerNorm 转回 fp32、为输入开启梯度、打开梯度检查点兼容，**是 k-bit 训练的标准预处理步骤**。

---

## 7. 调试：确认“只有 LoRA 在算梯度”（原文真料）

PEFT 最容易翻车的点是：**以为冻结了底座，实际没冻结**，或反之 **LoRA 根本没收到梯度**。下面这段在 `backward()` 前后遍历所有参数，逐个打印 `requires_grad` 和 `grad`，用来肉眼核对：

```python
train_loss = lw[0] * loss0 + lw[1] * loss1 + lw[2] * loss2   # 多任务加权 loss

# loss backward 之前：grad 应全是 None
for name, parms in model.named_parameters():
    print('\nBefore backward\n')
    print('-->name:', name)
    print('-->para:', parms)
    print('-->grad_requirs:', parms.requires_grad)
    print('-->grad_value:', parms.grad)
    print("===========================")

train_loss.backward()

# loss backward 之后：只有 lora_A / lora_B 这些层 grad 非 None
for name, parms in model.named_parameters():
    print('\nAfter backward\n')
    print('-->name:', name)
    print('-->para:', parms)
    print('-->grad_requirs:', parms.requires_grad)
    print('-->grad_value:', parms.grad)
    print("===========================")
```

**怎么读这段输出（验收标准）**：

```
                  requires_grad   backward后 grad
底座 W0 (冻结)         False           None        ← 正确：底座没被训
LoRA  A / B           True          有数值张量      ← 正确：旁路在学习
─────────────────────────────────────────────
若底座 grad 非 None  → 底座没冻住，显存爆 + 不是 PEFT
若 LoRA  grad 全 None → LoRA 没接上 / 输入没 require_grad（回看第4节）
```

- `name` 里含 `lora_A` / `lora_B` 的就是可训练旁路；其余是冻结底座。
- backward **前**所有 `grad` 都应是 `None`（还没算）；backward **后** 只有 LoRA 那几个有值。

---

## 常见问题 / 坑

| 现象 | 原因 | 解法 |
|------|------|------|
| `print_trainable_parameters()` 显示 100% | LoRA 没真正挂上，或 `target_modules` 名字写错匹配不到层 | 用 `print(model)` 看真实层名（如 Bloom 是 `query_key_value`，Baichuan 是 `W_pack`）再填 |
| 报错“没有需要梯度的张量 / element 0 does not require grad” | 用了梯度检查点但没开输入梯度 | 加 `model.enable_input_require_grads()`（第4节） |
| 量化模型上 LoRA 不收敛 / loss 不降 | 漏了 k-bit 预处理 | 先 `prepare_model_for_kbit_training(model)` 再 `get_peft_model` |
| `target_modules=["W_pack"]` 换到 LLaMA 报错 | 层名是模型相关的 | LLaMA 用 `["q_proj","k_proj","v_proj","o_proj"]` |
| 显存仍很大（只用了 LoRA 没量化） | 底座 fp16 仍占 2N 字节 | 上 QLoRA：`load_in_4bit=True` + NF4（第6.1节） |
| 长序列训练偶发 OOM 崩溃 | 优化器状态显存尖峰 | QLoRA 用分页优化器（paged optimizer），换页到 CPU 防崩 |
| `r` 设很大但效果没变好 | 任务内在秩低，过大徒增参数还易过拟合 | 从 r=8/16 起步，必要时配 `lora_dropout`；保持 `α/r` 稳定 |
| 合并权重后想再换任务很麻烦 | merge 后旁路融进主干了 | 训练阶段不 merge，保留多套 `A,B` 按任务热插拔；只在最终部署时 merge |

---

## 🔗 跳转链接

**枢纽**
- [[00-知识地图]]
- [[llm-train/README]]
- [[llm-train/pytorch/distribution/README]]
- [[llm-train/megatron/README]]
- [[llm-train/megatron-deepspeed/README]]
- [[ai-framework/megatron-lm/README]]
- [[ai-framework/deepspeed/README]]
- [[ai-framework/pytorch/README]]
- [[ai-framework/huggingface-peft/README]]

**同族 PEFT 方法**
- [[llm-train/peft/PEFT-API]]
- [[llm-train/peft/Prompt-Tuning]]
- [[llm-train/peft/Prefix-Tuning]]

**底层依赖**
- [[llm-compression/quantization/量化基础]] —— NF4 / 双量化 的量化原理
- [[llm-algo/transformer/模型架构]] —— LoRA 加在哪些线性层（q/k/v/o、MLP）
- [[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/NCCL]] —— 多卡训练时的通信

**下游应用**
- [[llm-alignment/RLHF]] —— RLHF/DPO 常用 LoRA 训练奖励模型/策略
- [[B07:llm-inference/大模型推理张量并行]] —— 量化 + 并行推理
