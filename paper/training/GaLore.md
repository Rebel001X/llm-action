# GaLore：梯度低秩投影，让 24GB 单卡也能全参预训练 7B

> 一句话定位：GaLore 不压缩权重、不冻结权重，而是把**优化器状态（Adam 的一、二阶动量）压到低秩子空间**，从而在保持"全参数训练"语义的前提下，把优化器显存砍掉约 65%，让 7B 模型在单张 RTX 4090（24GB）上从零预训练成为可能。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] · [[llm-train/pytorch/distribution/README]] · [[llm-compression/quantization/量化基础]] · [[docs/transformer内存估算]]

- 代码：https://github.com/jiaweizzhao/GaLore
- 论文：https://arxiv.org/abs/2403.03507 （*GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection*，Zhao et al., ICML 2024 Oral）

---

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点：GaLore 到底压了谁 | 优化器状态 ≠ 权重 |
| 1 | 地基：训 LLM 的显存账，钱花在哪 | 权重/梯度/Adam 态/激活 |
| 2 | 前置对照：LoRA 为何"省但伤" | 低秩权重 vs 低秩梯度 |
| 3 | 核心洞察：梯度本身就低秩 | $G \approx U\Sigma V^\top$ |
| 4 | 方法主干：投影—优化—回投三步 | $P^\top G$ / $P R$ |
| 5 | 工程三件套：8bit + per-layer + 子空间切换 | `update_proj_gap` |
| 6 | 关键公式与算法伪码 | 完整逐行 |
| 7 | 显存数值手算 | 7B 省多少 GB |
| 8 | 实验结论（定性） | C4 / GLUE |
| 9 | 评价/局限/对照表 | 何时别用 |

---

## 0. 一句话锚点

训练一个 LLM，显存的"四座大山"是：**权重 W、梯度 G、优化器状态（Adam 的 $m,v$）、激活值**。
- **LoRA** 动的是第一座山：用 $W+BA$ 替代 $W$，让可训练参数变少 → 权重/梯度/优化器态都跟着小。代价：表达能力受限于低秩增量，预训练效果差。
- **GaLore** 动的是第三座山：**权重照样全量更新（full-parameter）**，只是把"维护 Adam 动量"这件事搬到一个低秩子空间里做。优化器状态从 $O(mn)$ 降到 $O(mr+nr)$（$r \ll \min(m,n)$）。

> 记忆钩子：LoRA 是"低秩的权重"，GaLore 是"低秩的梯度（统计量）"。前者改变了模型容量，后者不改变模型容量。

---

## 1. 地基：训 LLM 的显存账，钱到底花在哪

以一个含 $N$ 个参数、用 Adam + 混合精度训练的模型为例，逐项拆解（单位：字节/参数）：

```
                每参数显存占用（bf16 主跑 + fp32 优化器态，常见配置）
  ┌─────────────────────────────────────────────────────────┐
  │ 权重 W (bf16)          : 2 bytes/param                    │
  │ 梯度 G (bf16)          : 2 bytes/param                    │
  │ Adam 一阶动量 m (fp32) : 4 bytes/param  ← GaLore 攻击点    │
  │ Adam 二阶动量 v (fp32) : 4 bytes/param  ← GaLore 攻击点    │
  │ (可选)fp32 权重副本    : 4 bytes/param                    │
  └─────────────────────────────────────────────────────────┘
   优化器状态 = m+v = 8 bytes/param，是权重的 4 倍！
```

**关键结论**：在 Adam 训练里，**优化器状态（$m,v$）通常是最大的固定开销**，比权重本身还贵。对 7B 模型，光 $m+v$ 就是 $7\text{B} \times 8\text{B} = 56\,\text{GB}$ —— 单卡根本放不下。GaLore 的全部目标就是干掉这 56GB 的大部分。

（激活值是另一座山，但它随 batch/序列长度变化，且可用激活重计算/checkpointing 处理，与 GaLore 正交，可叠加。）

---

## 2. 前置对照：LoRA 为什么"省显存但伤效果"

LoRA 把一层权重的更新约束成低秩：$W = W_0 + BA$，其中 $B\in\mathbb{R}^{m\times r}, A\in\mathbb{R}^{r\times n}$，只训练 $A,B$。

```
   LoRA：冻结大矩阵，只训两条"细管子"
   ┌──────────────┐         ┌──┐
   │   W0 (冻结)   │   +     │B │ × ┌──A──┐   只有 B、A 进优化器
   │   m × n      │         │  │   └─────┘
   └──────────────┘         └──┘
   省显存：优化器只维护 r(m+n) 个参数
   伤效果：所有更新被锁死在 rank-r 子空间内 → 从头预训练学不动
```

LoRA 的两个硬伤（GaLore 要修的痛点）：
1. **表达受限**：整个训练轨迹的权重增量 $\Delta W$ 被强制限制为秩 $\le r$。但研究发现，从随机初始化做**预训练**时，所需的更新并不是天然低秩的，强行低秩会显著掉点。
2. **改变了优化对象**：你不再优化原始 $W$，而是优化 $A,B$ 的组合，损失曲面被改写。

> GaLore 的发问：能不能**保留对 $W$ 的全秩更新**，又只省优化器状态？答案藏在"梯度"里。

---

## 3. 核心洞察：梯度矩阵本身就是低秩的（随训练演化）

GaLore 的理论支点：在合理假设下（论文用"reversible network / 可逆网络"等条件），**权重梯度矩阵 $G_t = -\nabla_W \mathcal{L}$ 的秩会随训练逐渐降低，趋于低秩结构**。

直觉解释：训练初期梯度方向五花八门（高秩），随着模型收敛，有效的下降方向集中到少数几个主方向上 —— 梯度的能量被少数奇异值吃掉。既然 $G_t$ 近似低秩：

$$G_t \approx U_r \, \Sigma_r \, V_r^\top, \qquad U_r\in\mathbb{R}^{m\times r},\ V_r\in\mathbb{R}^{r\times n}$$

那就没必要在全空间 $\mathbb{R}^{m\times n}$ 里维护 Adam 的 $m,v$ —— 只需在梯度的**主子空间**里维护即可。

```
   梯度的奇异值谱（示意）：能量集中在前 r 个方向
   σ
   │█
   │██
   │███
   │████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  ← 尾部近乎为 0，可丢弃
   └──┬──────────────────► 奇异值序号
      r（保留前 r 个主方向）
```

这与 LoRA 的本质差别：
- LoRA 假设**权重增量 $\Delta W$ 低秩**（强约束，限制容量）；
- GaLore 假设**梯度 $G$ 低秩**（弱假设，且全秩权重照常更新，只是分步从不同子空间累积更新，最终 $\sum \Delta W$ 仍可达到高秩）。

---

## 4. 方法主干：投影 → 子空间内优化 → 回投，三步走

GaLore 对**每一个二维权重矩阵**（如 attention 的 $W_Q,W_K,W_V,W_O$、FFN 的 up/down/gate）独立做以下三步：

### 4.1 投影（Project down）
取一个投影矩阵 $P\in\mathbb{R}^{m\times r}$（来自 $G$ 的 SVD 左奇异向量），把全尺寸梯度压到低秩：
$$R_t = P^\top G_t \in \mathbb{R}^{r\times n}$$

### 4.2 在低秩空间里跑优化器（Adam on $R_t$）
Adam 的动量 $m,v$ 只在 $r\times n$ 的小矩阵上维护：
$$N_t = \text{Adam}(R_t) \quad (\text{即 } N_t = m_t / (\sqrt{v_t}+\epsilon))$$
**这里就是省显存的根本**：$m,v$ 的形状从 $m\times n$ 变成 $r\times n$。

### 4.3 回投（Project back）并更新权重
把低秩更新映射回原空间，再做全参更新：
$$W_t = W_{t-1} - \eta \cdot \alpha \cdot (P\, N_t)$$
其中 $\alpha$ 是缩放系数（代码里 `--galore_scale`，例 0.25）。注意 $W$ 仍是**全尺寸、全秩**地被更新。

```
  GaLore 单层一步的数据流（每个权重矩阵独立）

   G_t (m×n) ──P^T──►  R_t (r×n) ──Adam──►  N_t (r×n) ──P──►  ΔW (m×n)
   全尺寸梯度   投影下    低秩梯度   只在此处   低秩更新   投影回   全尺寸更新
                          ▲ m,v 在这里维护（小！）
   W_t = W_{t-1} - η·α·ΔW         （权重始终是全秩 full-param）
```

**对比一眼看懂省在哪：**

```
  普通 Adam:  G(m×n) → [m(m×n), v(m×n)] → ΔW(m×n)     优化器态 = 2mn
  GaLore   :  G(m×n) →P^T→ R(r×n) → [m(r×n), v(r×n)] →P→ ΔW(m×n)
                                       优化器态 = 2rn (+ P 占 mr)
```

---

## 5. 工程三件套：把"能跑"变成"单卡能跑 7B"

光有低秩投影还不够省到 24GB，GaLore 论文/代码叠了三个工程技巧：

### 5.1 周期性切换子空间（`update_proj_gap`）
梯度的主子空间会随训练**漂移**。若投影矩阵 $P$ 固定不变，就退化成"只在一个固定子空间里训"，等价于换了个 LoRA。所以 GaLore **每隔 $T$ 步（如 `--update_proj_gap 500`）重新对当前梯度做一次 SVD，刷新 $P$**。

```
  step:  0........500.......1000......1500
  子空间: [  P^(0)  ][  P^(1)  ][  P^(2)  ]  ← 每 500 步换一次主方向
  效果：不同阶段的更新落在不同子空间，累积起来 ΣΔW 可达高秩 → 逼近全参训练
```
- 切换太频繁（gap 小）：SVD 开销大、不稳；切换太稀（gap 大）：丢失梯度变化、退化为低秩。$T$ 是关键超参，需调。

### 5.2 8-bit 优化器（`galore_adamw8bit`）
低秩后的 $m,v$ 再叠加 bitsandbytes 的 8-bit Adam 量化，每个动量元素从 4 字节降到 1 字节，优化器显存再砍约 4×。这与低秩**正交可叠加**。
> 参见 [[llm-compression/quantization/量化基础]]：8-bit 优化器用分块量化保精度。

### 5.3 逐层权重更新（per-layer / `galore_adamw8bit_per_layer`）
常规流程是"反向传播算完所有层梯度 → 统一 step"，这要求所有层梯度同时在显存里。GaLore 借助 PyTorch 的 **per-parameter hook**：某一层梯度一算出来，**立刻做 GaLore 更新并释放该层梯度**，不必等整张图。这进一步压低梯度峰值显存。

```
  常规：  [算 L1 grad][算 L2 grad]...[算 Ln grad] → 一次性 step（峰值高）
  per-layer：[算 L1 grad → 立即更新 → 释放][算 L2 grad → 立即更新 → 释放]...
             ▲ 梯度在显存里只短暂存在，峰值显存大降
```
代价：**目前 per-layer 仅支持单卡（`--single_gpu`），不兼容 `nn.parallel.DistributedDataParallel`**（README 明确说明），因为 DDP 的梯度 all-reduce 与"算完即更新即释放"的时序冲突。

---

## 关键公式 / 算法伪码 / 数值示例

### 算法伪码（单个权重矩阵 $W\in\mathbb{R}^{m\times n}$，设 $m\le n$）

```
初始化：P = None
for t = 1, 2, ... :
    G_t = -∇_W L(W_{t-1})                      # 全尺寸梯度 (m×n)

    if t % update_proj_gap == 0 or P is None:   # 周期刷新子空间
        U, Σ, V^T = SVD(G_t)                    # 对当前梯度做 SVD
        P = U[:, :r]                            # 取前 r 个左奇异向量 (m×r)

    R_t = P^T @ G_t                             # 投影下：低秩梯度 (r×n)
    N_t = Adam_step(R_t; m_state, v_state)      # 仅在 (r×n) 上维护 m,v
    ΔW  = P @ N_t                               # 回投：全尺寸更新 (m×n)
    W_t = W_{t-1} - η · α · ΔW                  # 全参更新；α=galore_scale
```

> 实务细节：选 $m\le n$ 时投左奇异 $U$（投行空间），$m>n$ 时投右奇异 $V$（投列空间），始终把大维度压成 $r$。

### 数值手算：7B 模型优化器显存，GaLore 省多少？

设全部 7B 参数都走 Adam，先看**普通 8-bit Adam**与 **GaLore 8-bit** 的优化器态对比。取一个代表性方阵层 $W\in\mathbb{R}^{4096\times 4096}$，rank $r=1024$（约 1/4）：

- 普通 Adam 该层优化器态（$m+v$，fp32）：
  $$2 \times (4096\times4096) \times 4\,\text{B} = 2\times 16.7\text{M}\times 4 \approx 134\,\text{MB}$$
- GaLore（低秩 $r=1024$，fp32）：
  $$2 \times (1024\times4096) \times 4\,\text{B} = 2\times 4.19\text{M}\times 4 \approx 33.5\,\text{MB}$$
  外加投影矩阵 $P\in\mathbb{R}^{4096\times1024}$：$4.19\text{M}\times 4\text{B}\approx 16.8\,\text{MB}$ → 合计 $\approx 50\,\text{MB}$。
- **该层优化器显存：134MB → 50MB，省约 63%**（rank=1/4 时的典型量级）。

整模层面，论文给出：用 GaLore 训 7B（C4 预训练），**优化器状态显存约下降 65%**，配合 8-bit + per-layer + 激活重计算，**总显存可压到 24GB 以内（单张 RTX 4090）**。README 实测：LLaMA-7B、8-bit GaLore-Adam、单卡 + activation checkpointing、`bsz=16` 时约 **22.8GB**。

```
  优化器状态显存（7B，示意，数值见原文/约）
  普通 16-bit Adam : ████████████████████  ~56 GB  放不下
  8-bit Adam       : █████                 ~14 GB
  GaLore 8-bit     : ██                    ~ 省到能塞进 24GB 卡
```

> 数字以原论文 / 官方 README 为准；上面手算为单层量级演示，整模需逐层累加且含非二维参数（embedding/LayerNorm 不投影，照常 Adam）。

---

## 实验结论（定性，数字标"约/见原文"）

| 任务 | 设置 | 结论（定性） |
|---|---|---|
| C4 预训练 | LLaMA 60M~7B，从头训 | GaLore 的 perplexity **接近全秩 Adam**，显著优于同等显存预算的 LoRA / ReLoRA（见原文表） |
| 单卡可行性 | 7B + 4090 24GB | 全参预训练从"不可能"变"可行"，约 22.8GB（README） |
| GLUE 微调 | RoBERTa-base | 微调精度**与 LoRA 相当或更好**，且省优化器显存（见原文） |
| 内存占用 | 8-bit + per-layer | 优化器态显存约降 **65%**，端到端显存大幅下降（见原文） |

要点：GaLore 的卖点不是"比 Adam 更准"，而是**在逼近全参 Adam 质量的同时把显存压到消费级单卡**，这是 LoRA 系做不到的（LoRA 预训练会明显掉点）。

---

## 安装与最小复现（来自官方 README）

```bash
git clone git@github.com:jiaweizzhao/GaLore.git
cd GaLore
git checkout a6bc1650
# 依赖 torch 2.1.0+cu118（whl 见 download.pytorch.org/whl/cu118/...）
```

单卡 4090 预训练 LLaMA-7B 的最小命令（per-layer 8-bit，关键开关 `--optimizer galore_adamw8bit_per_layer`）：

```bash
# LLaMA-7B, 8-bit GaLore-Adam, single GPU, activation checkpointing
# bsz=16, ~22.8G
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_7b.json \
    --lr 0.005 \
    --galore_scale 0.25 \          # 回投缩放 α
    --rank 1024 \                  # 低秩 r（越小越省、越激进）
    --update_proj_gap 500 \        # 每 500 步刷新子空间 P
    --batch_size 16 \
    --total_batch_size 512 \
    --activation_checkpointing \   # 激活重计算，与 GaLore 正交叠加
    --num_training_steps 150000 \
    --warmup_steps 15000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --single_gpu \                 # per-layer 仅支持单卡
    --optimizer galore_adamw8bit_per_layer
```

**调参直觉**：`--rank` 越小越省显存但更激进（可能掉点）；`--galore_scale` 控制低秩更新幅度；`--update_proj_gap` 太小则 SVD 开销大、太大则退化为低秩。命令默认值以官方仓库为准。

> 注意：**per-layer 权重更新当前仅支持单 GPU（`--single_gpu`），不能配 `DistributedDataParallel`**。多卡场景请用非 per-layer 的 GaLore 变体（普通 `galore_adamw`），让 DDP 的 all-reduce 正常工作。

---

## 评价 / 对照 / 局限

| 维度 | LoRA | GaLore |
|---|---|---|
| 压什么 | 权重增量（$\Delta W$ 低秩） | 优化器状态（梯度统计量低秩） |
| 权重更新秩 | 受限 rank-r | **全秩**（子空间随时间切换累积） |
| 适合预训练 | 弱（明显掉点） | **强**（接近全参 Adam） |
| 适合微调 | 强 | 强（相当或更好） |
| 额外开销 | 几乎无 | **周期性 SVD**（每 `update_proj_gap` 步一次） |
| 可与量化叠加 | 是（QLoRA） | 是（8-bit Adam） |

**局限与后续：**
1. **SVD 开销**：每次刷新子空间要对每个权重梯度做 SVD，$O(mn^2)$，大模型上非零成本；后续工作用随机 SVD / 幂迭代近似来降本。
2. **超参敏感**：`rank`、`update_proj_gap`、`galore_scale` 都需调，调不好会掉点。
3. **per-layer 不兼容 DDP**：最省显存的模式锁死在单卡，限制了大规模分布式扩展（多卡需退回普通模式）。
4. **二维矩阵假设**：只对二维权重投影，embedding/bias/LayerNorm 等照常全态 Adam，省得相对少。
5. **后续演进**：Q-GaLore（投影矩阵+权重一起量化，进一步省）、GaLore 与 FSDP/ZeRO 的结合、自适应秩等都是接力方向。

> 一句话总结：GaLore = "全参训练的质量" × "接近 LoRA 的显存"，代价是周期性 SVD 与一些超参调优；它把"消费级单卡从零预训练 7B"从口号变成可复现的命令。

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 训练全景与显存账：[[llm-train/README]] · [[docs/transformer内存估算]]
- 分布式训练（多卡 / DDP / FSDP 背景）：[[llm-train/pytorch/distribution/README]]
- 量化与 8-bit 优化器：[[llm-compression/quantization/量化基础]]
- 对照阅读（参数高效微调思路）：[[llm-train/README]]
