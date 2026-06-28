# Mixtral (稀疏MoE)

> Mixtral 8x7B 是把 Llama 风格 Transformer 的每个 FFN 换成"8 个专家 + Top-2 路由"的稀疏混合专家(SMoE)模型：总参数 ~46.7B、每 token 只激活 ~12.9B，用稠密 13B 的算力跑出接近 70B 的质量。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-algo/moe/README]] [[llm-algo/llama/模型架构]] [[llm-compression/quantization/moe模型量化]]

## 阅读地图

| 节 | 内容 | 你将能回答 |
|----|------|-----------|
| 0 | 一句话锚点 | Mixtral 到底改了什么 |
| 1 | 地基：稠密 FFN 与"为什么要 MoE" | FFN 占多少参数、稀疏化省什么 |
| 2 | MoE 层结构 | 专家、门控、Top-k 怎么拼 |
| 3 | Top-2 路由逐步拆解 | softmax→argtop2→加权 的每一步 |
| 4 | 参数量手算 | 46.7B 与 12.9B 怎么来的 |
| 5 | 显存与吞吐手算 | 为什么"省算力不省显存" |
| 6 | 专家并行(EP) | 多卡怎么切、All-to-All 是什么 |
| 7 | 负载均衡与路由分析 | 专家塌缩、aux loss、专家专精 |
| 8 | 稠密 vs 稀疏对照 | 何时选 MoE |
| 9 | 规模配置与训练要点 | 8x7B/8x22B 超参 |
| 10 | 面试问答清单 | 高频考点+陷阱 |

## 0. 一句话锚点

把 Transformer 每一层里那个"又胖又贵"的前馈网络(FFN)复制成 **8 份(专家)**，再加一个小小的**路由器(门控)**，每个 token 进来时门控只挑 **2 个**专家干活、其余 6 个完全不算。于是：

- **参数很多**(8 份 FFN 都存着) → 模型容量大；
- **算力很少**(每次只跑 2 份) → 推理便宜。

这就是"**稀疏激活**"：参数稠密存储、计算稀疏触发。

## 1. 地基：稠密 FFN 与"为什么要 MoE"

### 1.1 先看一个普通 Transformer block

Llama 类的一个 decoder 层(参考 [[llm-algo/llama/模型架构]]):

```
        x (hidden, 维度 d)
         │
   ┌─────▼─────┐
   │ RMSNorm   │
   └─────┬─────┘
         ▼
   ┌───────────┐   Grouped-Query Attention
   │  Attn     │   (Mixtral 沿用 GQA + RoPE)
   └─────┬─────┘
         ▼   + 残差
   ┌─────▼─────┐
   │ RMSNorm   │
   └─────┬─────┘
         ▼
   ┌───────────┐   ←—— 这里！稠密里是一个 FFN
   │  FFN/MoE  │       Mixtral 里换成 8 专家 MoE
   └─────┬─────┘
         ▼   + 残差
```

### 1.2 稠密 FFN 长什么样

Llama/Mixtral 的 FFN 是 **SwiGLU**(3 个矩阵)：

$$\text{FFN}(x)=\big(\text{SiLU}(xW_{gate})\odot (xW_{up})\big)W_{down}$$

- $W_{gate}, W_{up}\in\mathbb{R}^{d\times d_{ff}}$，$W_{down}\in\mathbb{R}^{d_{ff}\times d}$。
- Mixtral 7B 量级：$d=4096$，$d_{ff}=14336$。
- 单个 FFN 参数 = $3\times d\times d_{ff}=3\times4096\times14336\approx 1.76\times10^8$ ≈ **176M**。

**为什么 FFN 是 MoE 的下手对象？** 因为在 Transformer 里 FFN 通常占了**约 2/3 的参数与算力**(注意力只占 1/3)。想用更多参数提升质量，最划算的就是把 FFN 做大；但把它简单加宽会让"每个 token 的算力"线性上涨。MoE 的诀窍是：**加宽容量(更多专家) 但不加算力(只激活少数)**。

> 核心矛盾：我们想要"更多参数(更聪明)"，又不想要"更多 FLOPs(更贵)"。稠密模型里这俩绑死，MoE 把它们**解耦**。

## 2. MoE 层结构

把上面那个 FFN 框替换成：

```
                 x  (一个 token, 维度 d)
                 │
        ┌────────▼─────────┐
        │  Router / Gate   │  线性层 W_g: d → 8
        │  logits = x·W_g  │  得到 8 个分数
        └────────┬─────────┘
                 │ 取 Top-2 (argmax 两次)
        ┌────────▼─────────────────────────────┐
        │ 选中 E_a, E_b ;  其余 6 个不计算       │
        └───┬───────────────────────┬──────────┘
            ▼                        ▼
      ┌──────────┐            ┌──────────┐         ┌────┐
      │ Expert a │            │ Expert b │  (E0..E7 各是一个 SwiGLU FFN)
      │ FFN_a(x) │            │ FFN_b(x) │   未选中的 E 全部跳过
      └────┬─────┘            └────┬─────┘
           │ × g_a                 │ × g_b   (g_a,g_b 是归一化后的门控权重)
           └──────────┬────────────┘
                      ▼
                  y = g_a·FFN_a(x) + g_b·FFN_b(x)
```

要点：
- **8 个专家**：每个专家就是一个独立参数的 SwiGLU FFN(结构相同、权重不同)。
- **路由器**只是一个 $d\times8$ 的小线性层(参数 $4096\times8=32768$，几乎可忽略)。
- 路由是 **per-token**(逐 token)、**per-layer**(每层独立路由)，不是整句话选一次。

## 3. Top-2 路由逐步拆解(每一步说"为什么")

设当前 token 的隐藏向量为 $x\in\mathbb{R}^d$，路由权重 $W_g\in\mathbb{R}^{d\times 8}$。

**第 1 步：算门控 logits。**
$$h = x W_g \in \mathbb{R}^{8}$$
*为什么：* 让每个专家对这个 token 打一个"我有多适合处理你"的分。

**第 2 步：选 Top-2。**
$$\mathcal{T}=\text{TopK}(h, k{=}2)=\{i,j\},\quad h_i,h_j \text{ 是最大的两个}$$
*为什么 k=2 不是 k=1？* k=1(Switch 风格)最省算力但路由"硬"、训练不稳；k=2 给了梯度两条路径、表达更平滑，是质量/成本的甜点。Mixtral 选 2。

**第 3 步：只对选中的 2 个做 softmax(关键细节)。**
$$g_i=\frac{e^{h_i}}{e^{h_i}+e^{h_j}},\quad g_j=\frac{e^{h_j}}{e^{h_i}+e^{h_j}},\quad g_i+g_j=1$$
*为什么只在 Top-2 上 softmax 而不是 8 个上？* 这样权重在被选中的 2 个专家间归一化，未选中的天然贡献 0，输出尺度稳定(权重和恒为 1)。这正是 Mixtral 官方实现的做法。

**第 4 步：稀疏加权求和。**
$$y=\sum_{e\in\mathcal{T}} g_e\cdot \text{FFN}_e(x)=g_i\,\text{FFN}_i(x)+g_j\,\text{FFN}_j(x)$$
*为什么是加权和：* 两个专家各出一份"意见"，门控权重决定听谁多一点。

### 逐数手算(单个 token)

假设 8 个专家的 logits：

```
专家:   E0    E1    E2    E3    E4    E5    E6    E7
logit:  0.5   2.1   0.3  -1.0   1.8   0.9   0.2   1.1
                ↑高               ↑次高
```

- Top-2 = {E1(2.1), E4(1.8)}。
- 在这两者上 softmax：
  $g_{E1}=\dfrac{e^{2.1}}{e^{2.1}+e^{1.8}}=\dfrac{8.166}{8.166+6.050}=\dfrac{8.166}{14.216}\approx 0.574$
  $g_{E4}=\dfrac{6.050}{14.216}\approx 0.426$ (验证：0.574+0.426=1.0 ✓)
- 输出 $y = 0.574\cdot\text{FFN}_{E1}(x) + 0.426\cdot\text{FFN}_{E4}(x)$。
- **E0,E2,E3,E5,E6,E7 这 6 个专家的 FFN 一次都没跑** → 这就是省下来的 6/8 = 75% 的 FFN 算力来源。

## 4. 参数量手算：46.7B 与 12.9B 从哪来

以 Mixtral 8x7B 公开配置为基(以官方为准)：

```
d_model (hidden)      = 4096
d_ff   (FFN 中间维)    = 14336
层数 L                = 32
专家数 E              = 8     每层
激活专家 k            = 2     Top-2
词表 V                = 32000
注意力: GQA, 32 头 / 8 KV 头
```

**(a) 每层 MoE 的专家参数**(SwiGLU 3 矩阵 × 8 专家)：
$$8\times 3\times d\times d_{ff}=8\times3\times4096\times14336\approx 1.41\times10^9 \approx 1.41\text{B/层}$$

**(b) 注意力 + norm + 路由**(每层，GQA 使 KV 投影变小，约)：
- Q 投影 $d\times d=4096^2\approx16.8\text{M}$；K,V 各 $4096\times1024\approx4.2\text{M}$；O 投影 $16.8\text{M}$ → 约 $42\text{M}$。
- 路由器 $4096\times8\approx0.03\text{M}$，norm 可忽略。
- 合计每层非专家 ≈ **42M**。

**(c) 每层总计** ≈ 1.41B + 0.042B ≈ **1.45B**。
**(d) 32 层** ≈ $1.45\times32\approx 46.4\text{B}$，加 embedding/输出 $2\times32000\times4096\approx0.26\text{B}$ → **总参数 ≈ 46.7B** ✓(与官方一致)。

**激活参数(每 token 实际参与计算)：** 每层只跑 2/8 专家：
$$\text{每层激活专家} = 2\times3\times4096\times14336\approx0.352\text{B}$$
$$\text{每层激活} \approx 0.352\text{B}+0.042\text{B(注意力)} \approx0.394\text{B}$$
$$\times 32 \text{层} + \text{embedding} \approx 12.9\text{B} \;\;(\text{即激活} \approx 12.9\text{B})\checkmark$$

> 关键对比：**存 46.7B 的参数，但每个 token 只算 12.9B 的量** → 这就是"稀疏激活"的全部魔力。算力 ≈ 稠密 13B，质量 ≈ 稠密 ~70B 级。

## 5. 显存与吞吐手算(为什么"省算力不省显存")

**显存看的是总参数(46.7B)，算力看的是激活参数(12.9B)。** 这是 MoE 最反直觉、也最常被面试追问的点。

权重显存(只算权重，不含 KV cache / 激活)：

| 精度 | 每参数字节 | 46.7B 权重显存 |
|------|-----------|----------------|
| FP16/BF16 | 2 | $46.7\times2\approx93.4$ GB |
| INT8 | 1 | $\approx46.7$ GB |
| INT4 | 0.5 | $\approx23.4$ GB |

- BF16 下 ~93GB → **单张 80GB A100/H100 放不下**，必须多卡或量化(见 [[llm-compression/quantization/moe模型量化]])。
- 但前向 FLOPs 只相当于 13B 稠密模型 → 在"放得下"的前提下，**吞吐/延迟接近 13B**。

**单 token 前向 FLOPs(粗算，FLOPs ≈ 2 × 激活参数)：**
$$2\times12.9\times10^9\approx 2.58\times10^{10}\ \text{FLOPs/token}$$
对比稠密 70B：$2\times70\times10^9=1.4\times10^{11}$ → Mixtral 约**省 5.4×** 算力。

> 一句话记忆：**MoE 用显存换算力。** 你为"更多参数"付的是显存代价，省下的是计算代价。

## 6. 专家并行(Expert Parallelism, EP)

8 个专家可以放在不同 GPU 上。这引出 MoE 特有的并行维度 **EP**，与 TP/PP/DP 正交。

```
            token 批 (路由后每个 token 知道去哪个专家)
                       │
        ┌──────────────┼───────────────┐
        │   All-to-All 通信(dispatch)   │  按目标专家把 token 发到对应卡
        └──────────────┼───────────────┘
            ▼           ▼            ▼           ▼
      ┌─────────┐ ┌─────────┐  ┌─────────┐ ┌─────────┐
      │ GPU0    │ │ GPU1    │  │ GPU2    │ │ GPU3    │
      │ E0,E1   │ │ E2,E3   │  │ E4,E5   │ │ E6,E7   │
      └────┬────┘ └────┬────┘  └────┬────┘ └────┬────┘
           │ 各自跑本地专家 FFN                  │
        ┌──┴───────────┼───────────────────────┴──┐
        │   All-to-All 通信(combine)              │  把结果发回原 token 所在卡
        └──────────────┼──────────────────────────┘
                       ▼
                  加权求和得 y
```

- **dispatch All-to-All**：每张卡按门控结果把自己的 token 发往持有目标专家的卡。
- **combine All-to-All**：算完再发回。每个 MoE 层 = **两次 All-to-All**(MoE 的主要通信开销)。
- **负载不均**会拖慢：若大量 token 涌向同一专家，那张卡成瓶颈 → 引出第 7 节的均衡。
- EP 常与 **capacity factor**(每专家容量上限)配合：超出容量的 token 被 drop 或溢出到次优专家，避免显存爆炸。
- EP vs TP：EP 切的是"哪个专家在哪张卡"，通信是 All-to-All；TP 切的是单个矩阵，通信是 All-Reduce。实际大规模训练常 **EP×TP×DP** 混用。

## 7. 负载均衡与路由分析

### 7.1 专家塌缩问题(为什么需要均衡)

训练初期若不加约束，门控容易陷入"富者越富"：少数专家总被选中、拿到更多梯度、变得更强、于是更容易被选中……最终 8 个专家退化成 1~2 个有用，其余成摆设。**这叫专家塌缩(routing collapse)。**

### 7.2 辅助负载均衡损失(aux loss)

经典做法(Switch/GShard 风格)：设一个 batch 内 token 数 $T$，专家 $i$ 的

- 实际被选中频率 $f_i = \dfrac{\text{选中专家 }i\text{ 的 token 数}}{T}$，
- 平均门控概率 $P_i = \dfrac{1}{T}\sum_t p_i(t)$。

$$\mathcal{L}_{aux}=\alpha\cdot E\cdot\sum_{i=1}^{E} f_i\,P_i$$

*为什么是 $f_i\cdot P_i$ 的和：* 当负载均匀($f_i\approx P_i\approx 1/E$)时该和最小；某专家被过度使用时 $f_iP_i$ 变大，损失惩罚它。$\alpha$ 是小系数(如 0.01)。

> 注意：Mixtral 官方推理代码里只做 Top-2 + 局部 softmax；aux loss 是**训练期**的均衡手段。Mixtral 技术报告也展示了路由**与领域/主题弱相关**——专家更多按"语法/token 级模式"专精，而非"数学专家 / 代码专家"这种语义分工。

### 7.3 路由分析手算：均衡 vs 塌缩

设一个 batch 有 $T=1000$ 个 token，8 专家。理想均衡时每专家命中 $1000\times2/8=250$ 次(因 Top-2，总分发 = $T\times k=2000$)。

```
均衡(健康):     塌缩(不健康):
E0 ███ 250      E0 ████████ 700
E1 ███ 250      E1 ██████ 520
E2 ███ 250      E2 ██ 180
E3 ███ 250      E3 █ 120
E4 ███ 250      E4 ▏ 90
E5 ███ 250      E5 ▏ 80
E6 ███ 250      E6 ▏ 110
E7 ███ 250      E7 ▏ 100
变异系数 CV≈0    CV 大,E0/E1 过载,EP 下成瓶颈
```

衡量指标：变异系数 $CV=\dfrac{\sigma}{\mu}$，越接近 0 越均衡。塌缩时少数专家过载，EP 的 All-to-All 被最慢的卡卡住。

## 8. 稠密 vs 稀疏对照

| 维度 | 稠密 13B (如 Llama-13B) | Mixtral 8x7B (稀疏) | 稠密 70B |
|------|------------------------|---------------------|----------|
| 总参数 | 13B | **46.7B** | 70B |
| 激活参数/token | 13B | **12.9B** | 70B |
| 权重显存(BF16) | ~26GB | **~93GB** | ~140GB |
| 推理 FLOPs/token | ~中 | **≈13B 量级(低)** | 高 |
| 质量 | 基线 | **≈接近/超 70B** | 高 |
| 通信 | 无特殊 | **每层 2× All-to-All** | All-Reduce |
| 适用 | 显存紧、要简单 | **显存够、要质量又省算力** | 不计成本要最强 |

**一句话决策：** 显存够(多卡/量化能放下 46.7B)、又想要高质量但不想付 70B 的算力账 → 选 MoE。显存极度受限的边缘部署 → 稠密更省心(MoE 把参数都压在显存里)。

## 9. 规模配置与训练要点

| 配置 | Mixtral 8x7B | Mixtral 8x22B(约,以官方为准) |
|------|--------------|------------------------------|
| hidden d | 4096 | 6144 |
| d_ff | 14336 | 16384 |
| 层数 | 32 | 56 |
| 专家数 / Top-k | 8 / 2 | 8 / 2 |
| 注意力 | GQA(32 头 / 8 KV) | GQA |
| 上下文 | 32K(滑窗+RoPE) | 64K |
| 总参 / 激活 | 46.7B / 12.9B | ~141B / ~39B |

训练要点：
- **基于 Llama 架构**：RMSNorm + RoPE + GQA + SwiGLU，只是 FFN→MoE。可复用 [[llm-algo/llama/模型架构]] 的大部分实现。
- **aux loss / z-loss** 维持负载均衡与 logit 稳定。
- **capacity factor + token drop** 控制专家溢出。
- **EP 通信优化**：All-to-All overlap、专家分组、避免热点专家。
- 详见通用 MoE 原理 [[llm-algo/moe/README]]，部署量化见 [[llm-compression/quantization/moe模型量化]]。

## 10. 面试问答清单

**Q1：Mixtral 总参 46.7B 但只激活 12.9B，显存按哪个算？**
踩点答：**显存按总参 46.7B**(8 个专家权重全要驻留)，算力/FLOPs 按激活 12.9B。所以 MoE "省算力不省显存"。
追问"那为什么还用它？"→ 因为在显存够的场景，能用 13B 的算力拿到接近 70B 的质量。

**Q2：为什么是 Top-2 而不是 Top-1 或 Top-8？**
踩点答：Top-1(Switch)最省但路由硬、训练不稳、梯度只有一条路；Top-8 等于稠密、不省算力。Top-2 在质量与成本间最优，给两条梯度路径且输出更平滑。

**Q3：Top-2 的 softmax 是在 8 个专家上做还是 2 个上做？**
踩点答：**只在被选中的 2 个上做** softmax，权重归一化到和为 1，未选中专家贡献天然为 0，输出尺度稳定。这是 Mixtral 官方实现细节,容易答错。

**Q4：什么是专家塌缩?怎么解决?**
踩点答：路由"富者越富"导致少数专家垄断、其余退化。解决：训练加**辅助负载均衡损失** $\alpha E\sum f_iP_i$、z-loss、capacity factor 限流。陷阱：aux loss 是训练期的,推理不需要。

**Q5：专家并行 EP 的通信瓶颈是什么?**
踩点答：每个 MoE 层有 **两次 All-to-All**(dispatch + combine)。负载不均时热点专家所在卡成瓶颈,被最慢的卡拖住整层。

**Q6：Mixtral 的专家是按"数学/代码/语言"分工的吗?**
踩点答：**不是**。技术报告显示路由与高层语义/领域**弱相关**,专家更多按 token 级/句法模式专精,且同一序列里相邻 token 常路由到不同专家。陷阱:别想当然说"有个数学专家"。

**Q7：MoE 相比稠密,哪个指标变好哪个变差?**
踩点答：参数量/容量↑、质量↑、相同算力下更强;但**显存↑↑、通信复杂度↑(All-to-All)、训练稳定性↓**(需均衡)。

**Q8：路由是逐 token 还是逐序列?每层共享路由吗?**
踩点答：**逐 token、逐层独立路由**。同一个 token 在第 3 层和第 10 层可能走完全不同的专家;每层有自己的门控 $W_g$。

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[llm-algo/moe/README]] — MoE 通用原理(门控、aux loss、Switch/GShard 谱系)
- [[llm-algo/llama/模型架构]] — Mixtral 复用的 Llama 骨架(RMSNorm/RoPE/GQA/SwiGLU)
- [[llm-compression/quantization/moe模型量化]] — MoE 部署量化(把 93GB 压到能放下)

参考：
- Mixtral 8x7B：https://www.promptingguide.ai/models/mixtral
- Mixtral 8x22B：https://www.promptingguide.ai/models/mixtral-8x22b
