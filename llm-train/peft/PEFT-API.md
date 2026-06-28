# PEFT 实战要点

> HuggingFace PEFT 库的工程落地手册：LoRA/QLoRA 怎么配、怎么合并、多 adapter 怎么切、显存怎么算、坑在哪。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-train/peft/Prompt-Tuning]] [[llm-train/peft/Prefix-Tuning]] [[ai-framework/huggingface-peft/README]]

## 阅读地图

| 节 | 主题 | 你会得到 |
|----|------|---------|
| 0 | 一句话锚点 | PEFT 的核心思想一句话 |
| 1 | 地基/前置 | 为什么不全参微调；冻结+插桩的数学 |
| 2 | LoRA 数学与 `LoraConfig` | `r`/`alpha`/`target_modules`/`dropout` 每个参数到底是什么 |
| 3 | `get_peft_model` 数据流 | 一个 Linear 是怎么被改写的 |
| 4 | QLoRA = 4bit 量化 + LoRA | NF4/双重量化/`prepare_model_for_kbit_training` |
| 5 | 合并权重 `merge_and_unload` | 为什么要合、合完为什么变快、QLoRA 不能直接合 |
| 6 | 多 adapter 管理 | `load_adapter`/`set_adapter`/`add_weighted_adapter` |
| 7 | 训练显存账 | 权重+梯度+优化器+激活逐项手算 |
| 数值例子 | 7B 全参 vs LoRA vs QLoRA | 一张对照表 |
| 常见问题 | 高频坑 | loss 不降/合并不生效/OOM/保存丢东西 |

---

## 0. 一句话锚点

> **PEFT（Parameter-Efficient Fine-Tuning）= 冻结预训练大模型的全部原始权重，只训练一小撮「外挂」参数（通常 < 总量的 1%），就拿到接近全参微调的效果。**

LoRA 是 PEFT 家族里最主流的方法；QLoRA 是「4bit 量化底座 + LoRA 外挂」的组合，让单张消费级显卡也能微调 7B~70B 模型。

```
全参微调:  [████████████ 7B 参数全部更新 ████████████]   显存爆炸
PEFT/LoRA: [████████████ 7B 冻结 ████████████] + [▏外挂 ~0.06B 训练▕]
                            ↑                          ↑
                       不算梯度/优化器              只有它吃梯度+优化器
```

---

## 1. 地基/前置

### 1.1 为什么不直接全参微调？

全参微调（Full Fine-Tuning）要为**每一个**参数维护：原值、梯度、以及优化器状态（Adam 是一阶动量 + 二阶动量两份）。以参数量 $N$、Adam + fp16 混合精度为例，仅「权重副本+梯度+动量」就要约 $16N$ 字节（见第 7 节）。7B 模型光这部分就 ~112 GB，远超单卡。

**核心矛盾**：预训练已经把通用知识压进了权重里，下游微调其实只需要一个「低秩的方向性修正」。全参微调是用大炮打蚊子。

### 1.2 PEFT 的统一思想：冻结主干 + 插入可训练模块

```
                  原始前向          PEFT 前向
  输入 x ──────►  W·x      vs    W·x  +  ΔW(x)
                   ↑                ↑        ↑
              冻结(requires_grad   冻结    只训这个 ΔW
                 = False)
```

不同 PEFT 方法的区别只在「ΔW(x) 长什么样」：

| 方法 | 插入位置 | 可训练形式 | 本笔记重点 |
|------|---------|-----------|-----------|
| **LoRA** | 线性层旁路 | 低秩矩阵 $BA$ | ✅ 主角 |
| Prompt-Tuning | 输入嵌入前 | 一串虚拟 token 向量 | [[llm-train/peft/Prompt-Tuning]] |
| Prefix-Tuning | 每层 K/V 前 | 每层前缀向量 | [[llm-train/peft/Prefix-Tuning]] |
| (IA)³ | 激活缩放 | 逐通道缩放向量 | — |

---

## 2. LoRA 数学与 `LoraConfig`

### 2.1 LoRA 的核心公式（先把数学讲透）

一个线性层本来是 $h = W_0 x$，其中 $W_0 \in \mathbb{R}^{d \times k}$ 是冻结的预训练权重。LoRA 假设「微调带来的更新 $\Delta W$ 是低秩的」，于是用两个小矩阵的乘积去近似它：

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\, B A\, x$$

- $A \in \mathbb{R}^{r \times k}$：降维矩阵，**用高斯随机初始化**。
- $B \in \mathbb{R}^{d \times r}$：升维矩阵，**初始化为全 0**（保证训练开始时 $\Delta W = 0$，不破坏预训练能力）。
- $r$：秩（rank），$r \ll \min(d,k)$，是 LoRA 的**信息瓶颈**。
- $\frac{\alpha}{r}$：缩放系数，把低秩增量放大/缩小到合适幅度。

```
        x (维度 k)
        │
   ┌────┴─────────────────────┐
   │                          │
 W0·x  (冻结, d×k)        A·x  (r×k, 降到 r 维) ── 随机初始化
   │                          │
   │                        B·(A·x) (d×r, 升回 d 维) ── 初始化为 0
   │                          │
   │                      ×(α/r) 缩放
   │                          │
   └──────────► (+) ◄─────────┘
                │
                h (维度 d)
```

**为什么 $B=0$ 初始化关键**：训练第 0 步 $\Delta W = B A = 0$，模型行为 = 原模型，不会一上来就把预训练知识搞乱，训练曲线平滑。

### 2.2 参数量怎么省

原层参数量 $d \times k$；LoRA 旁路参数量 $r(d+k)$。以 $d=k=4096$、$r=8$ 为例：

- 原层：$4096 \times 4096 \approx 16.7\text{M}$
- LoRA：$8 \times (4096+4096) = 65536 \approx 0.066\text{M}$
- 压缩比 ≈ **256×**

### 2.3 `LoraConfig` 逐参数拆解

```python
from peft import LoraConfig, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # 任务类型，决定包多大的外壳
    r=8,                            # 秩：表达能力 vs 参数量的旋钮
    lora_alpha=16,                  # 缩放分子，实际缩放 = alpha/r = 2.0
    target_modules=["q_proj", "v_proj"],  # 在哪些层注入 LoRA
    lora_dropout=0.05,              # 在 A·x 之后加 dropout，防过拟合
    bias="none",                    # 是否训练 bias: none/all/lora_only
    modules_to_save=None,           # 额外要"全量训练并保存"的模块(如分类头)
)
```

| 参数 | 直觉 | 调参经验 |
|------|------|---------|
| `r` | 低秩瓶颈宽度；越大越接近全参、越占显存 | 8/16 起步；任务难/数据多再加到 32/64 |
| `lora_alpha` | 增量幅度，配合 r 决定 $\alpha/r$ | 常设 `alpha = 2r`（即 scaling=2）；很多人固定 alpha 只调 r |
| `target_modules` | 注入哪些线性层 | 见 2.4，最关键 |
| `lora_dropout` | 对 LoRA 分支做 dropout | 0~0.1，小数据可调高 |
| `bias` | 是否一并训 bias | 一般 `none` |
| `modules_to_save` | 不走 LoRA、整层训练并随 adapter 保存 | 改了词表/加了分类头时必填，否则推理对不上 |

### 2.4 `target_modules` 怎么选（最容易踩坑）

`target_modules` 是模块**名字的子串匹配**列表。一个 Llama 解码层里的可注入点：

```
         ┌──────────── Attention ─────────────┐  ┌──── MLP ────┐
名字:    q_proj  k_proj  v_proj  o_proj          gate_proj up_proj down_proj
         └──── QKV 投影 ────┘  └输出投影┘          └──── 前馈网络 ────┘
最小配置:  ✅            ✅
推荐(更强): ✅    ✅     ✅     ✅              ✅       ✅      ✅
```

- **最小集** `["q_proj","v_proj"]`：原始 LoRA 论文配置，省、稳，效果常常已够。
- **全注意力+MLP**：QLoRA 论文建议把**所有线性层**都挂 LoRA，效果更接近全参（代价是参数翻几倍）。
- 偷懒写法：`target_modules="all-linear"`（新版 peft 支持，自动匹配所有 `nn.Linear`，但通常排除 `lm_head`）。
- **不同模型名字不同**：GPT-2 是 `c_attn`，ChatGLM 是 `query_key_value`，写错了**不报错但 LoRA 没注入**，表现为「可训练参数 = 0」或 loss 一直不动——务必用下一节的打印来核对。

---

## 3. `get_peft_model` 数据流：一个 Linear 是怎么被改写的

```python
from peft import get_peft_model
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

`get_peft_model` 做的事：遍历模型，把名字命中 `target_modules` 的 `nn.Linear` 包成 `lora.Linear`，原权重 `requires_grad=False`，新增 `lora_A`/`lora_B` `requires_grad=True`。

```
 原始模块树                         PEFT 改写后
 model                              model (PeftModel)
  └ ...layers.0                      └ base_model.model...layers.0
      └ self_attn                        └ self_attn
          └ q_proj: Linear  ──包装──►        └ q_proj: lora.Linear
                                                  ├ base_layer: Linear(冻结)
                                                  ├ lora_A.default: Linear(r×k, 训练)
                                                  └ lora_B.default: Linear(d×r, 训练)
```

> **自检铁律**：`print_trainable_parameters()` 的 `trainable%` 必须 > 0 且数量合理（百万级）。若是 0 或异常，几乎一定是 `target_modules` 名字写错。

---

## 4. QLoRA = 4bit 量化底座 + LoRA

### 4.1 动机

LoRA 把「梯度/优化器」省了，但**冻结的主干权重仍以 fp16 常驻显存**（7B ≈ 14 GB）。QLoRA 的杀招：**把冻结主干量化成 4bit 存**，显存再砍约 4×，腾出的空间训练时反量化成 bf16 做计算。

### 4.2 三板斧

```
┌─────────────────────────────────────────────────────────┐
│ ① NF4 (4-bit NormalFloat)                                 │
│   针对"权重近似正态分布"设计的 4bit 数据类型，             │
│   分位点按正态分布排布，比普通 int4 更省信息损失。         │
├─────────────────────────────────────────────────────────┤
│ ② 双重量化 (Double Quantization)                          │
│   连"量化用的缩放常数"也再量化一次，每参数再省 ~0.4 bit。 │
├─────────────────────────────────────────────────────────┤
│ ③ Paged Optimizer                                         │
│   优化器状态在显存不足时分页换到内存(CPU)，防 OOM 尖峰。  │
└─────────────────────────────────────────────────────────┘
        三者叠加: 65B 模型可在单张 48GB 卡上微调
```

### 4.3 代码骨架

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # ① 用 NF4
    bnb_4bit_use_double_quant=True,       # ② 双重量化
    bnb_4bit_compute_dtype=torch.bfloat16 # 计算时反量化成 bf16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)   # ★ 关键预处理
model = get_peft_model(model, lora_config)
```

### 4.4 `prepare_model_for_kbit_training` 到底做了什么

> 早期 `prepare_model_for_int8_training` 已 deprecated，统一用 `prepare_model_for_kbit_training`。它处理量化模型以适配训练：

```
1. 把 LayerNorm 等归一化层转回 fp32        → 量化下的归一化数值不稳，升精度保稳定
2. 让输出嵌入层 / lm_head 计算梯度并转 fp32 → 量化的输出层若不升精度会拖垮收敛
3. 在输入嵌入层挂 forward hook 计算输入隐状态的梯度
   → 让梯度能穿过被量化冻结的主干，流到 LoRA 分支
4. （可选）启用 gradient_checkpointing
```

参数 `use_gradient_checkpointing=True`：用梯度检查点节省激活显存，代价是反向传播变慢——**本质是用计算换显存**。

#### 补充：gradient_checkpointing（梯度检查点）原理

训练时反向传播需要前向阶段的中间激活值来算梯度，模型越深激活占的显存越多。梯度检查点的做法：**前向时只保存少数「检查点」处的激活，其余丢弃；反向时对每个局部段重新跑一遍前向把激活算出来**，从而用「多一次前向计算」换「少存大量激活」。

```
普通反向: 前向[存全部激活]──────────────► 反向[直接用激活算梯度]   省时间 费显存
检查点:   前向[只存检查点]──► 反向时[重算该段前向→拿激活→算梯度]   省显存 费时间
```

具体地：完整正向传播以 `torch.no_grad()` 运行（不存中间激活），每遇检查点保存「输入 tensor + 函数参数」；反向时把每个检查点当作一个局部阶段，取出保存的输入重算该段前向、跟踪激活、再算梯度。

> ⚠️ 配合坑：开 `gradient_checkpointing` 时务必 `model.config.use_cache=False`（否则警告/冲突），并对最新 transformers 用 `model.enable_input_require_grads()` 或 `prepare_model_for_kbit_training(..., gradient_checkpointing_kwargs={"use_reentrant": False})` 避免「输入不需要梯度」报错。

---

## 5. 合并权重 `merge_and_unload`

### 5.1 为什么要合并

训练时 LoRA 是「旁路」，推理每层要多算一次 $BAx$，有额外开销；而 LoRA 的妙处是**增量可以解析地折回主干**：

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$$

合并后得到一个和原模型结构完全一样的普通模型，**零额外推理延迟**，也方便用 vLLM/TensorRT 等部署。

```
合并前(训练态)            合并后(部署态)
   W0 ──┐                  W_merged = W0 + (α/r)·B·A
        ├─(+)─► h            └────────► h
 (α/r)BA┘                   单条路径, 无额外算子, 可被推理引擎吃掉
```

### 5.2 代码

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Llama-2-7b-hf", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, "my-lora-adapter")
merged = model.merge_and_unload()      # 折回主干并卸掉 LoRA 结构
merged.save_pretrained("llama2-merged")  # 存成普通模型
```

### 5.3 关键坑：QLoRA 不能直接合并到 4bit 底座

`merge_and_unload` 要做 $W_0 + \Delta W$ 的浮点加法，但 4bit 量化权重无法精确承接这个加法。正确姿势：**用 fp16 重新加载原始（非量化）基座模型，再挂 adapter 合并**。直接对 4bit 模型 merge 会报错或精度严重损失。

```
错误: 4bit 基座 + LoRA ──merge──► ✗ (精度崩/报错)
正确: fp16 基座(重载) + 同一份 LoRA ──merge──► fp16 完整模型 ✓
```

---

## 6. 多 adapter 管理

PEFT 支持一个基座挂多套 LoRA，按需切换/组合——典型用于多任务、多语言、多风格。

```
                 ┌── adapter: "math"   (lora_A/B)
  base model ────┼── adapter: "code"   (lora_A/B)
   (冻结一份)     └── adapter: "chat"   (lora_A/B)
                        ↑ 同一主干, 不同旁路, 内存只多几十 MB/个
```

```python
# 加载多个 adapter（adapter_name 是它们的标识）
model = PeftModel.from_pretrained(base, "math_adapter", adapter_name="math")
model.load_adapter("code_adapter", adapter_name="code")

model.set_adapter("math")   # 激活 math，推理时只走它
model.set_adapter("code")   # 切到 code

# 加权融合多个 adapter 成新 adapter
model.add_weighted_adapter(
    adapters=["math", "code"],
    weights=[0.7, 0.3],
    adapter_name="math_code_mix",
    combination_type="linear",
)
model.set_adapter("math_code_mix")

model.disable_adapter()     # 临时退回纯基座(上下文管理器/方法)
```

| API | 作用 |
|-----|------|
| `load_adapter(path, adapter_name=)` | 再加载一套 LoRA，共享同一基座 |
| `set_adapter(name)` | 切换当前激活的 adapter |
| `add_weighted_adapter(...)` | 线性/拼接等方式融合多套权重为新 adapter |
| `disable_adapter()` | 暂时禁用所有 adapter（拿基座行为做对照） |
| `delete_adapter(name)` | 删除某套 adapter 释放显存 |

> 优势：N 个任务只需 1 份主干 + N 份小 adapter（每份几十 MB），相比存 N 个全量模型省几个数量级的磁盘/显存。

---

## 7. 训练显存账（逐项手算）

训练时显存四大块，设可训练参数量 $P$、模型参数量 $N$：

```
┌───────────────┬──────────────────────────────────────────────┐
│ ① 模型权重     │ 全参: 2N (fp16)；QLoRA: ~0.5N (4bit)           │
│ ② 梯度         │ 只有可训练参数有: 2P (fp16)                    │
│ ③ 优化器状态   │ Adam 两份动量: 8P (fp32 m+v) [+ fp32 主权重4P] │
│ ④ 激活值       │ ∝ batch × seq_len × hidden × 层数(可被检查点压)│
└───────────────┴──────────────────────────────────────────────┘
```

**全参微调 7B（fp16 权重 + fp32 Adam，混合精度经验公式 ≈ 16N 字节）**：
$$16 \times 7\text{B} = 112\ \text{GB （还不含激活）} \Rightarrow \text{单卡 80G 都装不下}$$

**LoRA 7B**：主干仍 fp16 常驻（$2N=14$ GB），但 ②③ 只作用在 $P\approx0.06\text{B}$ 上：
$$\underbrace{14}_{权重} + \underbrace{2\times0.06}_{梯度} + \underbrace{12\times0.06}_{优化器} \approx 14 + 0.84 \approx 15\ \text{GB（+激活）}$$

**QLoRA 7B**：主干 4bit（约 3.5 GB）+ 同样的小 LoRA 开销：
$$\underbrace{3.5}_{4bit权重} + \underbrace{\sim0.84}_{LoRA梯度+优化器} \approx 4.3\ \text{GB（+激活）}$$
→ 这就是「**单张 24G/16G 消费卡能微调 7B**」的来源。

---

## 数值例子 / 对照表

设 Llama-2-7B，$d=k=4096$，注入 `q_proj,v_proj`，32 层，$r=8$。

| 项 | 全参微调 | LoRA | QLoRA |
|----|---------|------|-------|
| 主干权重精度 | fp16 (14 GB) | fp16 (14 GB) | NF4 4bit (~3.5 GB) |
| 可训练参数量 | 6.7 B | ~4.2 M (0.06%) | ~4.2 M |
| 梯度+优化器显存 | ~98 GB | ~0.8 GB | ~0.8 GB |
| 训练总显存(估,含激活) | >120 GB（多卡） | ~16–20 GB | ~6–10 GB |
| 单卡可行性 | ✗（需 A100×多） | 单 24G 卡可行 | 单 16G 卡可行 |
| 推理延迟 | 基准 | 旁路有微开销；合并后=基准 | 量化推理或合并到 fp16 |
| 产物大小 | ~14 GB | ~16 MB（adapter） | ~16 MB |

**单层 LoRA 参数手算**：$r(d+k)=8\times(4096+4096)=65{,}536$ 参数/层/投影。
注入 q、v 共 2 个投影 × 32 层：$65{,}536 \times 2 \times 32 = 4{,}194{,}304 \approx 4.2\text{M}$，与上表一致。

---

## 常见问题

| 现象 | 根因 | 解法 |
|------|------|------|
| `trainable% = 0` / loss 完全不动 | `target_modules` 名字写错，LoRA 没注入 | 打印模块名核对（GPT-2 用 `c_attn`，ChatGLM 用 `query_key_value`） |
| 训练正常但加载后推理像没微调 | 推理时未挂 adapter / 路径错 / 改了词表却没设 `modules_to_save` | 用 `PeftModel.from_pretrained` 挂 adapter；分类头/新 token 放进 `modules_to_save` |
| QLoRA `merge_and_unload` 报错或精度崩 | 直接对 4bit 基座做浮点合并 | 用 **fp16 重载基座** 再合并（见 5.3） |
| 开 gradient_checkpointing 报「输入不需要梯度」 | 量化冻结主干梯度断流 | `model.enable_input_require_grads()` 或经 `prepare_model_for_kbit_training` |
| gradient_checkpointing 警告 + 变慢且不省 | 没关 KV cache / reentrant 模式问题 | `config.use_cache=False`；`use_reentrant=False` |
| loss 抖动大/不收敛 | `alpha/r` 缩放过大、学习率偏高 | LoRA 学习率常比全参大（1e-4~3e-4）；调 `alpha=2r` |
| 显存仍 OOM | 激活太大 / batch 太大 | 开梯度检查点、减 `seq_len`/batch、用 paged optimizer |
| 多 adapter 推理结果串味 | 忘了 `set_adapter` 切换 | 推理前显式 `set_adapter(name)`；对照可 `disable_adapter()` |
| 保存 adapter 后体积巨大 | 误存了整个基座 | 只 `model.save_pretrained()` 存 adapter（PeftModel 自动只存 LoRA 权重） |

---

## 一图总结流程

```
                    ┌─────────── 训练阶段 ───────────┐
 加载基座(fp16/4bit) ─► [可选]prepare_for_kbit_training
       │                        │
       ▼                        ▼
 LoraConfig(r,alpha,target) ─► get_peft_model ─► print_trainable_parameters()
       │                                              │(核对 % > 0)
       ▼                                              ▼
   Trainer.train() ─────────────────────► model.save_pretrained("adapter") (仅 ~16MB)

                    ┌─────────── 部署阶段 ───────────┐
 fp16 重载基座 + PeftModel.from_pretrained(adapter)
       │
       ├─ 直接推理(可多 adapter: load/set/disable)
       └─ merge_and_unload() ─► save_pretrained() ─► 普通模型, 零额外延迟, 进 vLLM/TRT
```

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，回主图
- [[llm-train/peft/Prompt-Tuning]] — 软提示类 PEFT：在输入端学一串虚拟 token
- [[llm-train/peft/Prefix-Tuning]] — 在每层 K/V 前注入可训练前缀
- [[ai-framework/huggingface-peft/README]] — PEFT 库总览与更多方法（(IA)³、AdaLoRA、P-Tuning v2 等）
