# GLM-130B 训练经验：FP16 大模型如何不崩

> 用三招（嵌入梯度缩减 + FP32 Softmax + FP16 混合精度）让一个 130B 模型在不支持 BF16 的硬件上稳定训练完。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[llm-algo/transformer/模型架构]] · [[llm-optimizer/FlashAttention]]

原始资料：<https://github.com/THUDM/GLM-130B/blob/main/README_zh.md>

---

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|----|---------|--------|
| 0 | 一句话锚点 | 稳定性 = 防上溢/下溢 + 防梯度尖峰 |
| 1 | 地基：FP16/BF16/FP32 数值格式 | 指数位/尾数位/动态范围 |
| 2 | 为什么 GLM-130B 选 FP16 而非 BF16 | 硬件可移植性 vs 稳定性 |
| 3 | 招式一：嵌入层梯度缩减（gradient shrink）| `α=0.1`、5k step 崩溃 |
| 4 | 招式二：注意力 FP32 Softmax | CogView、PB-Relax、+1e4/-1e-3 |
| 5 | 三招的组合拳与训练崩溃排查 | loss spike、grad norm |
| 实操 | 可直接抄的代码片段 | embedding shrink / fp32 softmax |
| 坑 | 常见问题表 | |

---

## 0. 一句话锚点

> **大模型训练崩溃（loss 突然飞到 NaN/Inf）的根因只有两类：① 前向计算的数值上溢/下溢；② 反向传播里某一层的梯度范数异常尖峰。GLM-130B 用 FP32 Softmax 治第①类，用嵌入梯度缩减治第②类。**

记住这个因果链，后面所有技巧都是它的推论：

```
            训练崩溃 (loss → NaN)
                  ▲
        ┌─────────┴─────────┐
   异常梯度(尖峰)        前向数值溢出
        ▲                   ▲
   嵌入层梯度过大       Softmax 在 FP16 下上溢/下溢
        │                   │
   ❷ 嵌入梯度缩减      ❸ FP32 Softmax
   (α=0.1)             (注意力内部用 fp32)
```

---

## 1. 地基：FP16 / BF16 / FP32 三种格式

要理解 GLM-130B 的所有取舍，必须先把三种浮点格式的"位预算"刻进脑子。一个浮点数 = 符号位 + 指数位（决定**动态范围**，能表示多大/多小）+ 尾数位（决定**精度**，相邻两数之间多密）。

```
          符号  指数(range)        尾数(precision)
 FP32     [S]  [ E E E E E E E E ] [ M ×23 ]   range≈1e±38, 精度高
 FP16     [S]  [ E E E E E ]       [ M ×10 ]   range≈6e-8 ~ 6.5e4, 精度尚可
 BF16     [S]  [ E E E E E E E E ] [ M ×7  ]   range≈1e±38, 精度低
```

| 格式 | 指数位 | 尾数位 | 最大值 | 最小正规数 | 动态范围 | 谁在用 |
|------|-------|-------|--------|-----------|---------|--------|
| FP32 | 8 | 23 | ~3.4e38 | ~1.2e-38 | 大 | 主权重/优化器状态 |
| FP16 | 5 | 10 | ~65504 | ~6.1e-5 | **小** | GLM-130B、CogView |
| BF16 | 8 | 7 | ~3.4e38 | ~1.2e-38 | 大 | BLOOM、Megatron |

**核心矛盾**：FP16 的指数只有 5 位 → 一旦某个激活值超过 **65504** 就上溢成 `Inf`，小于 **~6e-5** 就下溢成 0。注意力分数动辄到 `1e4` 量级，离上溢只差一步。BF16 把指数位拉回 8 位、动态范围和 FP32 一样，所以"溢出"几乎不会发生——代价是尾数只剩 7 位，精度更差，但训练对精度不敏感、对范围敏感，所以 BF16 训练更稳。

> 一句话：**BF16 用精度换范围，天生抗溢出；FP16 范围窄，必须靠工程技巧补救。** 这就是 GLM-130B 全部经验的出发点。

---

## 2. 为什么 GLM-130B 偏要用 FP16（原文真料 1）

原文观点：

> FP16 混合精度已成为十亿~百亿规模模型训练框架的默认选项，但仍太容易遇到精度问题。NVIDIA Ampere GPU 提供 BF16（被 BLOOM 采用）来缓解；然而 **BF16 在其他平台上不被支持**，这大大缩小了它在更广泛应用中的潜力。为了让更多开发者使用，GLM-130B 仍选择 FP16 作为训练浮点格式。同时这意味着 GLM-130B 将面临更多稳定性挑战。

为什么"BF16 不被支持"是个大问题？BF16 的硬件原生支持始于 NVIDIA **Ampere（A100）**。在 GLM-130B 训练的年代，很多国产/旧卡（V100、以及各类非 Ampere 加速器）只有 FP16 单元。GLM-130B 的目标是"让更多开发者能复现/微调"，所以必须押注最通用的 FP16。

这是一个典型的 **可移植性 vs 易用性** 工程权衡：

```
   选 BF16  ──► 训练省心，但只能跑在 A100/Ampere+，门槛高
   选 FP16  ──► 到处能跑(V100/各类NPU)，但要自己解决稳定性
                          │
                          └─► 于是发明了下面两招补丁
```

**混合精度（mixed precision）回顾**：FP16 训练并不是所有东西都用 FP16。标准做法是——前向/反向用 FP16 算（省显存、快），但保留一份 FP32 主权重做更新，并用 **loss scaling**（把 loss 乘一个大系数 S，反传后梯度再除以 S）把过小的梯度"抬"进 FP16 可表示区间，防下溢。GLM-130B 的两招是在这套标准混合精度之上的**额外**补丁。

---

## 3. 招式一：嵌入层梯度缩减（原文真料 2）

### 3.1 现象与诊断

原文观察：

> 在训练早期，**嵌入层的梯度范数明显比其他层大**。根据经验，大多数训练崩溃都发生在其梯度范数激增之后。

为什么是嵌入层？词嵌入是一张 `[vocab, hidden]` 的查找表，每一步只有 batch 里出现过的 token 行被更新，但被更新的那些行收到的梯度很集中、很大。早期模型还没学好，这些梯度尤其凶猛，范数尖峰 → 优化器一步走太远 → loss 炸。

```
 grad_norm
   ▲
   │           ╱╲  ← 嵌入层梯度尖峰(spike)
   │          ╱  ╲
   │  ───────╯    ╲____  其他层平稳
   │
   └────────────────────► step
        尖峰之后 ~几步内 loss 崩溃
```

### 3.2 两条路线对比

| 方案 | 做法 | 效果 | 副作用 | 出处 |
|------|------|------|--------|------|
| Embedding Norm | 对嵌入输出做 LayerNorm | 能稳定训练 | **牺牲较大下游性能** | BLOOM |
| Embedding **梯度缩减** | 前向不变，只把回传梯度乘 α | 稳定且**不掉点** | 几乎无 | GLM-130B |

GLM-130B 没采用 BLOOM 的 Embedding Norm（因为它掉下游分），而是直接在梯度上动手。

### 3.3 一行代码的魔法（原文真料）

```python
word_embedding = word_embedding * α + word_embedding.detach() * (1 - α)
```

**为什么这一行就能把梯度缩到 α 倍？** 关键在 `detach()`——它切断梯度，被 detach 的那一支前向有值、反向梯度为 0。设 $w$ 为嵌入，输出 $y = \alpha w + (1-\alpha)\,\text{detach}(w)$：

- 前向：$y = \alpha w + (1-\alpha) w = w$ ✅ **数值完全不变**，模型看到的还是原嵌入。
- 反向：$\dfrac{\partial y}{\partial w} = \alpha$（detach 支贡献 0）→ 回传到嵌入的梯度被乘上 $\alpha$。

也就是说，**前向恒等、反向打折**。GLM-130B 取 $\alpha = 0.1$，即把嵌入层梯度缩到原来的 **1/10**。

数值手算：若原嵌入梯度范数为 $g=50$（远超其他层的 ~5），缩减后变成 $\alpha g = 0.1 \times 50 = 5$，立刻回到与其他层同量级，优化器步长正常。

```
   缩减前:  embed grad = 50  ←尖峰
   缩减后:  embed grad = 50 × 0.1 = 5  ←拉平
                          ▲
                   detach 让前向不变, 只缩反向
```

### 3.4 会不会拖慢收敛？

原文亲测：

> 缩小嵌入梯度并**没有减缓收敛速度**；相反，没有缩小梯度的模型会出现意外尖峰，并在 **5k 步左右**出现训练崩溃。

直觉解释：早期嵌入梯度本来就过大、是"过冲"，缩小它反而让训练更稳更平滑，不影响最终学到的表示。**收益（不崩）远大于成本（基本为零）。**

---

## 4. 招式二：注意力 FP32 Softmax（原文真料 3）

### 4.1 梯度缩减是"事后药"，溢出要"事前防"

原文一针见血：

> 梯度收缩是一种避免训练崩溃的**事后**技术。从本质上讲，崩溃是由异常的损失"梯度"形成的，要么是噪声数据，要么是**正向计算中的精度上溢或下溢**。我们观察到，在大型语言模型中，**注意力的计算操作是最容易上溢或下溢的**。

所以招式二直接堵住前向溢出的源头——注意力里的 Softmax。

### 4.2 为什么注意力分数会溢出

注意力分数 $s_{ij} = \dfrac{q_i \cdot k_j}{\sqrt{d}}$。原文引 **CogView** 的观察：

> 不同的注意力头对其注意力分数有非常不同的数值范围，有些头计算出的平均分数可达 **+1e4 或 -1e-3**。

`+1e4` 已经逼近 FP16 上限 65504；在 Softmax 里要先算 $e^{s}$，$e^{10000}$ 在 FP16 下直接 `Inf`（上溢）；而 `-1e-3` 这种极小值在做归一化时又可能下溢成 0。FP16 的窄动态范围（见 §1）在这里彻底暴露。

```
   注意力头分数分布(CogView 观察):
     head A:  均值 ≈ +1e4  ──► exp() 上溢 → Inf  (FP16 max 65504)
     head B:  均值 ≈ -1e-3 ──► 归一化下溢 → 0
                       ▲
          同一层 96 个头, 范围天差地别
```

### 4.3 CogView 的 PB-Relax 为什么被放弃

CogView 提出 **PB-Relax（精度瓶颈放松）**：做 Softmax 前，从每个头的注意力分数矩阵中扣掉该矩阵的**最大绝对值**，把数值压回安全区（这就是经典的"减最大值再 exp"稳定 Softmax 的思路，只是按整矩阵做）。

数学上 Softmax 减常数不变：$\text{softmax}(s_i) = \dfrac{e^{s_i - c}}{\sum_j e^{s_j - c}}$，取 $c = \max|s|$ 让指数项都 $\le e^0=1$，绝不上溢。

但原文实测它在 GLM-130B 上**很慢**：

> 可能是因为在 **96 个大小为 `2048×2048`** 的注意力分数矩阵中寻找最大值和操作标量，对 CUDA 内核不友好。

为什么慢？`argmax/max-reduce` 一整个 `2048×2048`（≈420 万元素）矩阵 × 96 个头，是大规模规约 + 标量分支操作，访存密集、并行度差，对 GPU kernel 极不友好。

### 4.4 最终方案：Softmax 内部用 FP32

> 经过几周艰苦探索，最快最简单的方法是在 **Softmax 计算中使用 FP32**。与完全 FP16 计算相比，几乎没有速度损失，但明显提高了训练稳定性。

机制：注意力分数算出来后，在进入 Softmax（`exp` + 求和 + 归一化）这一小段把张量 cast 到 FP32，算完再 cast 回 FP16。FP32 指数位 8 位、最大 ~3.4e38，`exp(1e4)` 这种在 FP32 里也只是 `Inf`？——不，关键是 FP32 范围足够大、且配合"减最大值"的标准实现，`exp(s - max)` 落在安全区，绝不溢出。

```
   QKᵀ/√d  ──(fp16)──► scores
                          │ cast → fp32
                          ▼
              Softmax 全程 fp32 (exp/sum/div)
                          │ cast → fp16
                          ▼
                   × V  ──(fp16)──► out
   只有窄窄一段用 fp32, 显存/速度几乎不变
```

为什么"几乎不损速度"？Softmax 在整个注意力里只占很小一段计算量（大头是两次 matmul：$QK^\top$ 和 $\cdot V$，仍跑 FP16/Tensor Core），只把这一小段升精度，开销可忽略。对比 PB-Relax 还要全矩阵 `max-reduce`，FP32 Softmax 既稳又快。

> 这也是后来主流框架（Megatron-LM 的 `--attention-softmax-in-fp32`、FlashAttention 内核）的默认做法 → 见 [[llm-optimizer/FlashAttention]]。

---

## 5. 三招组合拳与崩溃排查心法

```
┌───────────────────────────────────────────────┐
│  GLM-130B 稳定性三件套                          │
│                                                 │
│  ❶ FP16 混合精度 + loss scaling   (基础设施)    │
│       └ 省显存/快, 用 FP32 主权重兜底           │
│  ❷ 嵌入层梯度缩减 α=0.1           (治梯度尖峰)  │
│       └ word_emb*α + detach*(1-α), 前向不变     │
│  ❸ 注意力 FP32 Softmax            (治前向溢出)  │
│       └ 只升 softmax 这一段精度                 │
└───────────────────────────────────────────────┘
```

**排查崩溃的标准流程**（从 GLM-130B 经验提炼）：

1. 盯 `grad_norm` 曲线：哪一层先出尖峰？若是嵌入层 → 上梯度缩减。
2. 盯 loss：是否在某个固定步数（如 ~5k）规律性炸？规律性强多半是数值问题而非数据。
3. 定位前向溢出：打印各层激活的 max/min，注意力分数是不是逼近 65504？→ 上 FP32 Softmax。
4. 实在不行再考虑：调小 loss scale、跳过坏 batch、回滚 checkpoint 重启。

---

## 实操：可直接抄的代码

### 嵌入层梯度缩减（原文真料）

```python
# alpha=0.1 时把嵌入层回传梯度缩到 1/10, 前向数值不变
alpha = 0.1
word_embedding = word_embedding * alpha + word_embedding.detach() * (1 - alpha)
# 前向: = word_embedding  (恒等)
# 反向: d/dw = alpha       (梯度打折)
```

### 注意力 FP32 Softmax（按原文思路的等价实现）

```python
# scores: [batch, heads, seq, seq], 由 QK^T/sqrt(d) 得到, dtype=fp16
scores_fp32 = scores.float()                 # cast 到 fp32
attn = torch.softmax(scores_fp32, dim=-1)    # 全程 fp32, 自动减最大值, 不溢出
attn = attn.to(scores.dtype)                 # cast 回 fp16
out = attn @ value                           # 后续仍 fp16, 走 Tensor Core
```

> 在 Megatron-LM 中对应开关：`--attention-softmax-in-fp32`（或框架内 `attention_softmax_in_fp32=True`）。

### 关键数值速查

| 量 | 值 | 含义 |
|----|----|------|
| 嵌入梯度缩减系数 α | **0.1** | 梯度 ×0.1，前向不变 |
| 无缩减时崩溃步数 | ~**5k** step | 早期梯度尖峰导致 |
| 注意力头分数范围 | **+1e4 ~ -1e-3** | CogView 观察，FP16 易溢出 |
| FP16 最大可表示值 | **65504** | 超过即上溢 Inf |
| 注意力矩阵规模 | 96 × `2048×2048` | PB-Relax 在此规模上慢 |

---

## 常见问题 / 坑

| 现象 / 疑问 | 根因 | 对策 |
|------------|------|------|
| loss 在 ~5k 步规律性变 NaN | 嵌入层早期梯度尖峰 | 嵌入梯度缩减 α=0.1 |
| 注意力处出现 Inf/NaN | FP16 Softmax 对 `+1e4` 量级分数上溢 | Softmax 内部用 FP32 |
| 用了 PB-Relax 但训练很慢 | 对 96×`2048²` 矩阵做 max-reduce，CUDA 不友好 | 改用 FP32 Softmax，更快更稳 |
| 担心嵌入梯度缩减拖慢收敛 | 误以为缩梯度=学得慢 | 实测不掉收敛速度，反而更稳 |
| 为何不直接用 BF16 一劳永逸 | BF16 仅 Ampere+ 原生支持，旧卡/部分平台不支持 | 押 FP16 保可移植，靠工程补稳定性 |
| Embedding Norm 也能稳，为何不用 | BLOOM 的 Embedding Norm 牺牲较大下游性能 | 用梯度缩减，稳且不掉点 |
| `detach()` 那行看不懂 | 前向 α+( 1-α)=1 恒等；反向 detach 支梯度为 0 | 记住"前向不变、反向打折" |
| FP32 Softmax 会不会变慢/爆显存 | Softmax 计算量占比小，仅升这一段精度 | 速度/显存几乎无影响 |

---

## 🔗 跳转链接

枢纽与延伸阅读：

- [[00-知识地图]] — 全局导航
- 框架：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]]
- 架构原理：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 算子/优化：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 量化与精度延伸：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]] · [[llm-compression/quantization/GPTQ]]
- 训练/微调/对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 硬件与网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 推理：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 评测与估算：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
