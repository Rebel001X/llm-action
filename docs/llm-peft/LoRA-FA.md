# LoRA-FA：冻结 A 矩阵以省下激活显存的高效微调

> 一句话定位：LoRA-FA = LoRA 的"省激活显存"改良版，**冻结下投影 A、只更新上投影 B**，让权重变化锁死在 A 列空间张成的低秩子空间里，从而**无需存储全秩输入激活**。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/peft/PEFT-API]] · [[ai-framework/deepspeed/README]] · [[llm-compression/quantization/量化基础]] · [[docs/transformer内存估算]]

## 阅读地图

| 你想知道 | 看哪一节 |
| --- | --- |
| LoRA-FA 一句话是什么 | §0 一句话锚点 |
| 复习 LoRA 的 `W + αAB` | §1 地基：从 LoRA 说起 |
| 为什么 LoRA 还要存激活、激活显存从哪来 | §2 痛点：激活显存 |
| 冻结 A 为什么能省激活 | §3 核心机制：FA = Frozen-A |
| 权重变化为何被锁在低秩空间 | §4 低秩子空间的几何 |
| 可训练参数减半 / 显存公式 `2n+8n_r` | §5 显存复杂度手算 |
| 怎么和量化/ZeRO-3/重计算叠加 | §6 与其它显存优化组合 + 实操节 |
| 容易踩的坑 | 常见问题表 |

## 0. 一句话锚点

**LoRA** 把权重更新写成 $\Delta W = AB$，训练 A 和 B 两个低秩矩阵；
**LoRA-FA**（FA = **F**rozen-**A**ctivation / **F**rozen-**A**）只训练 B，**A 在随机初始化后就冻住不动**。

> 论文摘要原话：LoRA-FA "选择冻结 A 的投影下权重、更新 B 的投影上权重在每个 LoRA 层中"，"确保模型权重的变化存在于低秩空间中，同时消除存储全秩输入激活的要求"。
> 实验跨 **RoBERTa、T5、LLaMA** 多种模型与规模，结果：LoRA-FA 精度可优于全参数微调与 LoRA，且**相对 LoRA 把总体内存成本降低最多 1.4×**。

## 1. 地基：先把 LoRA 摆清楚

预训练权重 $W \in \mathbb{R}^{d_{out}\times d_{in}}$ 冻结，LoRA 引入旁路：

$$
h = Wx + \alpha\,\Delta W\,x,\qquad \Delta W = AB
$$

其中（按本文记号，A 是"下投影 / project-down"，B 是"上投影 / project-up"）：

- $A \in \mathbb{R}^{r \times d_{in}}$：把输入从 $d_{in}$ 维压到秩 $r$，故称 **down-projection**；
- $B \in \mathbb{R}^{d_{out}\times r}$：把 $r$ 维升回 $d_{out}$ 维，故称 **up-projection**；
- $r \ll d_{in}, d_{out}$（典型 $r=8/16/64$）。

> 原文：LoRA "通过更新 A 和 B 两个低秩矩阵，并使用 AB 作为预训练和冻结权重 W 的变化，即 $W + \alpha\Delta W = W + \alpha AB$"。

```
        x (d_in)
         │
    ┌────┴──────────────┐
    │                   │
 [ W 冻结 ]        ┌──[ A 训练 ]  下投影 d_in→r
 (d_out×d_in)      │      │  z (r)
    │              │   [ B 训练 ]  上投影 r→d_out
    │              └──────┤
    └────── + ────────────┘
              │
              h (d_out)
   LoRA：A、B 都训练
```

LoRA 已经把**可训练参数**从 $d_{out}d_{in}$ 砍到 $r(d_{in}+d_{out})$，优化器状态显存随之大降。**但它没有解决激活显存**——这正是 LoRA-FA 要补的洞。

## 2. 痛点：LoRA 仍需昂贵的"激活显存"

反向传播要算梯度，就得在前向时**缓存中间激活**。对一个线性层 $y=Mx$，求 $M$ 的梯度需要输入 $x$：$\partial L/\partial M = (\partial L/\partial y)\,x^\top$。

在 LoRA 里 **A 是可训练的**，A 的输入就是该层的**全秩输入 $x\in\mathbb{R}^{d_{in}}$**。于是每个 LoRA 层都要把这个 $d_{in}$ 维（× batch × seq_len）的大激活存下来等反向用。

> 原文：LoRA "仍需要昂贵的激活记忆更新低秩权重。减少 LoRA 层数或使用激活重计算可能会损害微调性能或增加计算开销。"

```
激活显存 ≈ batch × seq_len × d_in × 层数 × 字节数
                          ▲
                          └ d_in 是“全秩”，很大（如 4096）
```

两条老路都不理想：① 减 LoRA 层 → 掉精度；② 激活重计算（recompute）→ 多一次前向，增计算开销。LoRA-FA 给出第三条路。

## 3. 核心机制：冻结 A，激活只需存 r 维

LoRA-FA 的唯一改动：**A 随机初始化后冻结，只更新 B**。

> 原文：LoRA-FA "冻结了 W 和 A，并仅在细调过程中更新 B"；"在细调过程中，冻结初始化的 A 和预训练的 W，并更新投影上权重 B"。

为什么这能省激活？看 B 的梯度需要什么输入：B 的输入是 $z = Ax \in \mathbb{R}^{r}$（已经被 A 压成 $r$ 维了）。

$$
\frac{\partial L}{\partial B} = \frac{\partial L}{\partial h}\,z^\top,\qquad z = Ax\in\mathbb{R}^{r}
$$

- 训练 B → 只需缓存 **$z$（r 维）**，不需要 $x$（$d_{in}$ 维）；
- A 冻结 → 不需要 A 的梯度 → **不必缓存 A 的输入 $x$**。

```
        x (d_in)  ← 不再需要为反向缓存它！
         │
    ┌────┴──────────────┐
 [ W 冻结 ]        ┌──[ A 冻结 ]❄  下投影 d_in→r
    │              │      │  z (r)  ← 只缓存这 r 维
    │              │   [ B 训练 ]✎  上投影 r→d_out
    └────── + ─────┴──────┘
              │
              h (d_out)
   LoRA-FA：A 冻❄  B 训✎  → 激活由 d_in 降到 r
```

**直觉**：把"需要存全秩输入的可训练层"（A）变成"无需存输入的冻结层"，激活的瓶颈维度从 $d_{in}$ 降到 $r$。$d_{in}=4096, r=8$ 时，这一项激活缩小约 $512\times$。

## 4. 低秩子空间的几何：精度为何不掉

冻结 A 不是随便冻——它给权重更新加了一个**结构约束**：

$$
\Delta W = AB,\quad A\ \text{固定} \ \Rightarrow\ \Delta W \text{ 的每一列都是 } A \text{ 各行（即 } A^\top \text{ 各列）的线性组合}
$$

> 原文：在适应过程中"权重的变化将被限制在由 A 的列空间定义的低秩空间中"，且 LoRA-FA"确保模型权重的变化存在于低秩空间中"。

也就是说：

- LoRA 的 $\Delta W$ 在一个"可学习的低秩子空间"里漂移（A、B 都动）；
- LoRA-FA 的 $\Delta W$ 在一个"由随机 A 固定下来的低秩子空间"里漂移（只 B 动）。

只要 A 用合适的随机初始化，其 $r$ 行大概率近似正交、张成一个"信息充分"的随机子空间（类似 random projection / Johnson–Lindenstrauss 的思想）——B 在这个子空间内自由组合，表达力足够。这解释了为何**LoRA-FA 精度不降反而常优于 LoRA**：约束相当于一种正则，缓解过拟合。

```
全参数：ΔW 在整个 R^{d_out×d_in} 自由游走（参数最多，易过拟合/费显存）
LoRA  ：ΔW 锁在“可学习”的秩-r 子空间（A,B 都学）
LoRA-FA：ΔW 锁在“A 固定”的秩-r 子空间（只学 B）——子空间不动，落点 B 自由
```

## 5. 显存复杂度：可训练参数减半，公式手算

> 原文：LoRA-FA"仅计算 B 的梯度，其具有 $d_{out}\times r$ 个元素。在 GPT 类型的模型中，总的可训练参数是 $n_r/2$，即 LoRA 中可训练参数数量的一半。因此，在 16 位混合精度训练中，模型权重和适配器相关状态的内存成本为 $2n + 8n_r$ 字节。"

逐项拆解（设 $n$ = 模型总参数量，$n_r$ = LoRA 全部 A+B 的参数量）：

| 量 | LoRA | LoRA-FA | 说明 |
| --- | --- | --- | --- |
| 可训练参数 | $n_r$（A+B 都训） | $n_r/2$（只训 B） | B 占一半 → 减半 |
| 优化器状态(Adam) | 随 $n_r$ | 随 $n_r/2$ | 只给 B 存 m、v |
| 输入激活瓶颈维 | $d_{in}$（全秩） | $r$（低秩） | §3 的核心收益 |

**16-bit 混合精度下的权重+适配器状态显存 ≈ $2n + 8n_r$ 字节**，逐项对应：

```
2n      = 模型权重 W（fp16，2 字节/参数 × n 个参数，冻结，无梯度/优化器）
8 n_r   = 适配器相关状态（每个 LoRA 参数约 8 字节：
          fp16 权重 + fp16 梯度 + Adam 的一阶/二阶动量等汇总）
```

**数值示例**（粗算，便于建立量级感）：取 $n=7\text{B}$（LLaMA-7B），LoRA-FA 一般 $r=8$、作用于若干投影矩阵，设 $n_r\approx 4\text{M}$：

- $2n = 2\times 7\times10^9 = 14\ \text{GB}$（权重，量化后还能更低，见 §6）；
- $8n_r = 8\times 4\times10^6 = 32\ \text{MB}$（适配器状态，几乎可忽略）。

可见微调显存主体已不是优化器，而是**权重 + 激活**；LoRA-FA 同时压低了激活，这正是相对 LoRA 省到 **1.4×** 的来源。

## 6. 与其它显存优化方法的组合（正交叠加）

> 原文：LoRA-FA"可以与先进的内存优化方法相结合，如权重量化、权重分片和选择性激活重计算"，"可以与其他内存优化技术相结合，提高其利用率"。

三种可叠加手段，各打不同显存项：

| 技术 | 打哪块显存 | LoRA-FA 怎么配 | 对应枢纽 |
| --- | --- | --- | --- |
| 权重量化 | $2n$（权重） | 把冻结的 $W$ 量化到更低位宽（如 4/8-bit），**不影响微调性能**（原文语） | [[llm-compression/quantization/量化基础]] |
| 权重分片 / ZeRO-3 | $2n$（跨卡分摊） | 多 GPU 数据并行时把 $W$ 分片到各卡，每卡显存降 | [[ai-framework/deepspeed/README]] |
| 选择性激活重计算 | 激活 | 只重算部分组件输入，**无需存 LoRA 层输入**即可平衡激活/重算成本 | [[docs/transformer内存估算]] |

> 原文要点回顾：
> - **权重量化**："将模型权重量化为较低的位宽，以减少模型权重的内存开销，而不影响细调性能"；
> - **权重分片**："将权重分片或使用 ZeRO stage-3 技术与 LoRA-FA 相结合，将模型权重分片到不同的 GPU 上，从而降低每个 GPU 的内存开销"；
> - **选择性激活重计算**："重新计算部分模型组件的输入，以减少激活内存开销……在不需要存储 LoRA 层输入的情况下平衡激活成本和重计算成本"。

```
微调总显存 = 权重(2n)          ← 量化↓ / ZeRO-3 分片↓
           + 适配器状态(8 n_r) ← LoRA-FA 已减半（只训 B）
           + 激活               ← LoRA-FA 已从 d_in 降到 r；+选择性重计算再↓
```

## 实操：把 LoRA-FA 用起来（概念配置）

> 注：原文为方法论描述，未给出具体命令行；下面给出与论文方法一致的**配置心智**，便于在 PEFT/DeepSpeed 体系里落地。落地 API 细节见 [[llm-train/peft/PEFT-API]]。

LoRA-FA 与标准 LoRA 在工程上只差一处：**把 A 的 `requires_grad` 置为 False，优化器只收 B 的参数**。

```python
# 伪代码示意：与论文“冻结 A、只训 B”一一对应
for name, p in model.named_parameters():
    if "lora_A" in name:
        p.requires_grad = False     # ← FA 的关键：冻结下投影 A
    elif "lora_B" in name:
        p.requires_grad = True      # 只更新上投影 B
    else:
        p.requires_grad = False     # 预训练权重 W 冻结

# 优化器只拿到“还在训练”的参数（即各层的 B）
optim = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=...)
```

叠加显存优化（与 §6 对应）：

```
权重量化     ：加载 W 时用 4/8-bit（量化基础那篇的反量化前向）
ZeRO-3 分片  ：DeepSpeed zero_optimization.stage = 3，把 W 切到各卡
选择性重计算 ：对非 LoRA 的重组件开 activation/gradient checkpointing
```

## 常见问题 / 坑

| 现象 / 疑问 | 原因 | 对策 |
| --- | --- | --- |
| 把 B 冻结、只训 A，效果差很多 | A 的输入是全秩 $x$，冻 A 才省激活；冻 B 反而既不省激活又限制了输出维组合 | 严格"冻 A 训 B"，别冻反 |
| A 用全零初始化 | A=0 则 $z=Ax=0$ 恒为零，B 收不到任何信号，无法学习 | A 用随机初始化（保证列空间有效），B 可初始化为 0 |
| 以为可训练参数和 LoRA 一样 | LoRA-FA 只训 B → 参数是 LoRA 的一半（$n_r/2$） | 优化器状态、梯度按 $n_r/2$ 估，不要按 $n_r$ |
| 期望权重显存也大降 | $2n$（权重）由 LoRA-FA 本身不变，它省的是**激活 + 适配器状态** | 想降 $2n$ 要叠**量化 / ZeRO-3** |
| 激活仍然爆 | 只对 LoRA 层省了激活，模型主干前向激活仍在 | 叠**选择性激活重计算**处理主干 |
| 误以为精度必降 | LoRA-FA 的固定低秩子空间相当于正则 | 论文实测常优于 LoRA / 全参微调，放心用 |

## 🔗 跳转链接

- 知识地图枢纽：[[00-知识地图]]
- 同目录 PEFT：[[README]]（PEFT 总览）· [[ReLoRA]] · [[MAM_Adapter]]
- 训练与 PEFT API：[[llm-train/README]] · [[llm-train/peft/PEFT-API]]
- 显存 / 框架：[[docs/transformer内存估算]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 压缩 / 量化（配 LoRA-FA 降权重显存）：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 算子 / 注意力：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 架构基础：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 对齐（微调下游）：[[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 评测：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 硬件 / 网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]]
