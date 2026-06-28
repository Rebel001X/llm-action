# 混合专家模型 MoE(Mixture of Experts)深入浅出讲义

> 一句话定位:MoE 用"很多个小专家 + 一个门控调度员"替换 Transformer 里那块最重的 FFN,让模型**总参数量暴涨但每个 token 实际算的量几乎不变**——这是当今超大模型(DeepSeek-V3、Mixtral、GShard、Switch)能做到"万亿参数还跑得起"的核心魔法。📍 导航:[[00-知识地图]]
> 🔗 相关:[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-train/pytorch/distribution/README]] · [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点:稀疏激活到底省了什么 | 总参数 vs 激活参数 |
| 1 | 地基:稠密 FFN 为什么这么贵 | Transformer / FFN / 算力墙 |
| 2 | 核心思想:把一块大 FFN 拆成 N 个专家 | Expert / 条件计算 |
| 3 | 门控网络 Gating:谁来决定走哪个专家 | Top-K / Softmax / Router |
| 4 | 负载均衡:为什么会"专家偏科",怎么治 | Aux Loss / 容量因子 |
| 5 | 工程落地:专家并行 EP 与 All-to-All | EP / All-to-All / 通信账 |
| 6 | 名作巡礼:GShard→Switch→GLaM→Mixtral→DeepSeekMoE | 历史演进 |
| 7 | 关键公式 + 数值手算 | FLOPs / 显存 / 加速比 |
| 8 | 评价、对照、局限 | 优缺点表 |

## 0. 一句话锚点

**MoE = 条件计算(Conditional Computation)。**
普通(稠密 Dense)模型:每个 token 都要过**全部**参数。
MoE(稀疏 Sparse)模型:每个 token 只过**一小部分**参数(被路由到的那几个专家)。

于是你可以这样理解一个核心比值:

$$
\text{激活参数量} \approx \frac{K}{N} \times \text{专家总参数量} + \text{共享参数}
$$

其中 $N$ 是专家总数,$K$ 是每个 token 实际选中的专家数(通常 1~2,或 DeepSeek 的 8)。
**总参数量决定了"模型能记多少知识",激活参数量决定了"每次推理花多少算力"。** MoE 让这两者解耦——这就是它存在的全部意义。

```
       稠密 Dense                         稀疏 MoE
   ┌───────────────┐               ┌───────────────────────┐
   │   一块巨大FFN   │               │  Router 挑 Top-2       │
   │  100% 参数都算  │               │ ┌──┐┌──┐┌──┐ ... ┌──┐ │
   │               │               │ │E1││E2││E3│     │E64││
   └───────────────┘               │ └──┘└▲─┘└──┘     └▲─┘ │
   每 token: 算 1 份               │      └── 只算这2个 ─┘  │
                                   └───────────────────────┘
                                   每 token: 算 2/64 份的专家
```

## 1. 地基:稠密 FFN 为什么这么贵

回顾标准 Transformer 一层的结构(详见 [[llm-algo/transformer/模型架构]]):

```
x ──► [Multi-Head Attention] ──► +残差&LayerNorm ──► [FFN] ──► +残差&LayerNorm ──► out
```

其中 **FFN(前馈网络)** 是两个线性层夹一个激活:

$$
\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2,\quad W_1 \in \mathbb{R}^{d_{ff}\times d},\ W_2 \in \mathbb{R}^{d\times d_{ff}}
$$

通常 $d_{ff} = 4d$。**这块 FFN 占了 Transformer 单层参数的约 2/3**(注意力的 $Q,K,V,O$ 是 $4d^2$,FFN 是 $2 \cdot 4d^2 = 8d^2$)。

矛盾点:
- 我们想要**更多参数**(参数越多,能拟合/记忆的知识越多,Scaling Law 给的承诺)。
- 但稠密模型里"参数翻倍 ⇒ 每个 token 的计算量翻倍 ⇒ 训练和推理成本翻倍"。

MoE 的回答是:**只把 FFN 这块做大(变成很多个专家),但每个 token 只挑几个专家算。** 参数涨 64 倍,计算只涨约 2 倍。

> 直觉类比:一家医院如果让每个病人都看遍所有科室的所有医生(稠密),既慢又浪费;实际是分诊台(门控)根据症状把你导诊到 1~2 个对口科室(专家)。医院规模(总医生数)可以很大,但你的就诊耗时只取决于你看的那几个医生。

## 2. 核心思想:把一块 FFN 拆成 N 个专家

MoE 层的做法:用 $N$ 个**结构相同、参数独立**的小 FFN(称为"专家 Expert")替换原来那一块大 FFN。

```
                       ┌──────── MoE Layer ────────┐
   token x ──► Router(门控) ── 给出 Top-K 专家 & 权重
                       │
              ┌────────┼─────────┬──────── ... ────┐
              ▼        ▼         ▼                 ▼
           Expert_1 Expert_2  Expert_3    ...   Expert_N   (每个都是一个独立 FFN)
              │        │         │                 │
              └────────┴─── 加权求和(只对选中的)────┘
                       │
                       ▼
                    output y
```

设第 $i$ 个专家函数为 $E_i(x)$,门控给出的权重为 $g_i(x)$,则 MoE 层输出:

$$
y = \sum_{i \in \mathcal{T}(x)} g_i(x)\, E_i(x)
$$

关键在于 $\mathcal{T}(x)$ 是 Top-K 选出的专家集合(只有 $K$ 个非零),其余专家的 $g_i = 0$,**根本不参与计算**。这就是"稀疏激活"。

- **每个专家**:就是一个普通 FFN,$E_i(x) = W_2^{(i)}\sigma(W_1^{(i)}x)$。
- **专家之间不共享参数**(否则就退化成稠密了),所以总参数 $\approx N$ 倍单专家。
- **token 级路由**:每个 token 独立选专家,同一句话里相邻两个 token 可能去完全不同的专家。

## 3. 门控网络 Gating:谁来决定走哪个专家

门控(也叫 Router/路由器)是 MoE 的灵魂。它本身极小——通常就是一个线性层 $W_g \in \mathbb{R}^{N\times d}$,把每个 token 映射成 $N$ 个 logits,再决定去哪。

### 3.1 最经典的 Softmax Top-K 门控(GShard 式)

$$
h = W_g\, x \in \mathbb{R}^{N},\qquad
g = \mathrm{Softmax}(h),\qquad
\mathcal{T}(x) = \mathrm{TopK}(g,\,K)
$$

只保留 Top-K 个专家的权重,其余置零(并可重新归一化):

```
x ──► W_g ──► logits[64] ──► Softmax ──► 概率分布
                                            │
                                      取最大的 K 个
                                            │
                  ┌─────────────────────────┴──────────────┐
                  ▼                                         ▼
            选中 E_17 (权重0.62)                      选中 E_3 (权重0.38)
                  │                                         │
                  └────── y = 0.62·E_17(x) + 0.38·E_3(x) ───┘
```

### 3.2 为什么至少选两个专家(Top-2)?

这是本文件原始那句话的核心。GShard/早期工作发现:**如果只路由到 1 个专家(Top-1),门控就拿不到"对比信号"——它无法学会"这个专家比那个专家更合适"**,因为反向传播时梯度只流向被选中的那一个,门控不知道"别的选择会不会更好"。

把输入路由到**不止一个专家(至少两个)**,门控才能在两个专家的加权结果上获得有效的、可比较的梯度,从而**学会有效的路由选择**。Switch Transformers 后来就这一点做了更深入的研究:它证明在加入足够的负载均衡损失、合理初始化和容量设计后,**Top-1 也能稳定训练**,从而把通信和计算砍到最省。所以:

- **Top-2(GShard、Mixtral)**:路由更稳,门控学得好,代价是计算/通信约 2 倍。
- **Top-1(Switch)**:最省,但需要更精细的稳定性技巧(精度、容量因子、初始化缩放)。
- **Top-8 + 共享专家(DeepSeekMoE)**:用很多个"细粒度小专家"提升组合表达力。

> 一句话:Top-1 省钱但难训,Top-2 是稳健的默认值,"至少两个"的本质是**给门控提供可比较的学习信号**。

### 3.3 路由的不可微问题

TopK 选择是**离散、不可微**的(argmax 没梯度)。常见处理:把被选中专家的门控权重 $g_i$ **乘进输出里**(如上式),让梯度通过这个连续的权重系数回流到 $W_g$。也就是说,梯度不流经"选择"这个动作,而流经"权重大小"。这是一种实用的近似(类似 straight-through 思路)。

## 4. 负载均衡:专家会"偏科",怎么治

这是 MoE 工程上最大的坑,必须单列一节。

### 4.1 问题:路由坍缩(Routing Collapse)

门控会自我强化:某个专家一旦初期被选得多 → 它训练得更好 → 更容易被选 → 更好……最终**少数专家被挤爆,多数专家"饿死"没人用**。后果:

1. 模型有效容量退化(等于白养了一堆没用的专家)。
2. 在专家并行(EP)下,被挤爆的专家所在 GPU 成为瓶颈,其余 GPU 闲置 → 严重负载不均。

```
   理想(均衡)                       坍缩(偏科)
  E1 ████ 12.5%                   E1 ████████████████ 70%
  E2 ████ 12.5%                   E2 █ 4%
  E3 ████ 12.5%                   E3 █ 2%
  ...                             ...
  E8 ████ 12.5%                   E8 ░ 0.3%  ← 饿死
   ↑ 每个 GPU 活儿差不多            ↑ E1 这张卡累死,别的卡摸鱼
```

### 4.2 解法一:辅助负载均衡损失(Auxiliary Load-Balancing Loss)

加一个鼓励"流量平摊"的正则项。设一个 batch 有 $T$ 个 token,$N$ 个专家:

- $f_i$ = 被路由到专家 $i$ 的 token **比例**(实际分配,离散统计)。
- $P_i$ = 专家 $i$ 的门控**平均概率**(softmax 后,连续可微)。

$$
\mathcal{L}_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

直觉:当某专家既被频繁选中($f_i$ 大)又被高概率打分($P_i$ 大)时,这一项变大,被惩罚 → 逼门控把流量摊平。乘 $N$ 是为了让均衡时该损失约等于 1(与专家数无关)。系数 $\alpha$ 通常很小(如 0.01),太大会损伤主任务。

### 4.3 解法二:专家容量(Expert Capacity)与丢弃

给每个专家设一个**容量上限**:

$$
\text{capacity} = \left\lceil \frac{T}{N} \times \text{capacity\_factor} \right\rceil
$$

容量因子(capacity_factor)通常取 1.0~1.25。超过容量的 token 被**丢弃(dropped)**——它们跳过这个 MoE 层,只走残差连接。容量因子越大越不丢 token,但显存和通信开销越大(要为每个专家预留固定大小的缓冲区,做静态形状以便高效 All-to-All)。

```
专家 E5 容量 = 100
  到达 token: 130 个
  ├─ 前 100 个: 正常计算 ✅
  └─ 后 30 个:  溢出,被丢弃 ❌(只走残差,本层不更新)
```

### 4.4 解法三:DeepSeek 的"无辅助损失负载均衡"

DeepSeek-V3 提出给每个专家加一个**可学习的偏置 $b_i$**,只用于 Top-K 选择阶段($h_i + b_i$ 参与排序),但**不参与最终加权**。训练中根据各专家近期负载动态调整 $b_i$:过载就调低、欠载就调高。好处是**避免辅助损失对主任务造成的性能拉扯**。这是当前 SOTA 的均衡思路之一(具体调节规则以官方/原文为准)。

## 5. 工程落地:专家并行 EP 与 All-to-All

专家太多,单卡放不下,必须把不同专家**分散到不同 GPU**——这叫**专家并行(Expert Parallelism, EP)**。它带来一种独特的通信模式。详见 [[llm-train/pytorch/distribution/README]] 与 [[ai-infra/网络/集合通信原语]]。

### 5.1 为什么是 All-to-All

每张卡上的 token 路由结果是"分散"的:GPU0 上的 token 可能要去 GPU3 上的专家,GPU2 上的 token 要去 GPU0 的专家……需要把 token **按目标专家重新洗牌**到对应 GPU,算完再洗回来。这正是 **All-to-All** 集合通信原语干的事。

```
   ─────── Dispatch(分发, All-to-All #1)───────►
  GPU0: [t→E2, t→E5]              GPU0 收到所有要去 E0,E1 的 token
  GPU1: [t→E0, t→E3]   ───洗牌──►  GPU1 收到所有要去 E2,E3 的 token
  GPU2: [t→E1, t→E7]              GPU2 收到所有要去 E4,E5 的 token
  GPU3: [t→E4, t→E6]              GPU3 收到所有要去 E6,E7 的 token

        ┌──────────── 各 GPU 本地算自己的专家 FFN ────────────┐
        ▼                                                    ▼
   ◄────── Combine(回收, All-to-All #2)──── 把结果按原 token 洗回去
```

**一个 MoE 层 = 两次 All-to-All**(dispatch 分发 + combine 回收),夹着本地专家计算。

### 5.2 通信账(数值手算)

设隐藏维 $d=4096$,每卡 batch 内 $T=8192$ 个 token,Top-K=2,bf16(2 字节/元素)。

- 每个 token 要被发往 $K=2$ 个专家 → dispatch 发送量 $= T \times K \times d \times 2\text{B}$。
- $= 8192 \times 2 \times 4096 \times 2 = 134{,}217{,}728 \text{ B} \approx 128\ \text{MB}$(单次 All-to-All 的本卡发送量,量级估算)。
- combine 回收量同量级,再来 128 MB。
- **所以一个 MoE 层每卡每步约 256 MB 跨卡流量。** 在几十层 MoE 的大模型里,All-to-All 极易成为瓶颈——这也是为什么 MoE 训练高度依赖高带宽互联(NVLink/InfiniBand),以及为什么 DeepSeek 等会做 dispatch/combine 与计算的**重叠(overlap)** 来掩盖通信。

> 经验法则:MoE 把"算力受限"的问题部分转化成了"通信受限"。专家越多、EP 跨的节点越多,All-to-All 越贵。

### 5.3 与其它并行的叠加

实战中 EP 通常与张量并行(TP)、数据并行(DP)、流水并行(PP)叠加(3D/4D 并行)。常见布局:注意力部分用 TP+DP,FFN/MoE 部分用 EP。关键约束:**专家数最好能被 EP 度数整除**,否则负载天然不均。

## 6. 名作巡礼:MoE 的演进脉络(对照表)

| 模型 | 年代 | 路由 | 专家数(典型) | 一句话贡献 |
|---|---|---|---|---|
| **GShard** | 2020 | Top-2 | 数千 | 首次把 MoE 做到超大规模翻译,提出容量因子+aux loss+All-to-All 范式 |
| **Switch Transformer** | 2021 | **Top-1** | 数千~万级 | 证明 Top-1 也能稳,极简路由、省通信;研究稳定性(精度/初始化) |
| **GLaM** | 2021 | Top-2 | 64 | 大规模 decoder MoE,展示推理能耗远低于同质量稠密模型 |
| **Mixtral 8x7B** | 2023 | Top-2 | 8 | 开源旗舰,8 专家选 2,激活约 13B 而总参约 47B,效果对标更大稠密模型 |
| **DeepSeekMoE / V3** | 2024 | Top-K(细粒度) | 数百细粒度+共享专家 | 细粒度专家切分 + 共享专家 + 无辅助损失均衡,极致 token/参数效率 |

注:上述数字为典型/约值,具体配置以各自原文为准。两条主线值得记:
1. **路由从 Top-2 走向 Top-1(省)又走向更细粒度的多专家(强表达)**——没有银弹,看你卡通信预算。
2. **共享专家(Shared Expert)**:DeepSeekMoE 让一两个专家**对所有 token 都激活**,负责学"通用知识",其余路由专家学"专精知识"。这样减少了专家间的知识冗余。

```
   DeepSeekMoE 结构示意
   token ──┬──► [共享专家 E_s] ──(总是激活,学通用)──┐
           │                                        ├──► 加权和 ──► y
           └──► Router ─Top-K─► [路由专家×K] ──(学专精)─┘
```

## 7. 关键公式 / 算法 / 数值示例(汇总)

### 7.1 一个 MoE 前向的完整算法(伪代码思路)

```
输入: token 表示 X[T, d], 门控 W_g[N, d], 专家 {E_1..E_N}
1. logits = X @ W_g.T            # [T, N]
2. g = softmax(logits, axis=-1)  # 门控概率 [T, N]
3. topk_val, topk_idx = TopK(g, K)   # 每 token 选 K 个专家
4. (EP) All-to-All Dispatch: 按 topk_idx 把 token 发往目标专家所在 GPU
5. for 每个本地专家 E_i:  y_i = E_i(收到的 token)   # 本地 FFN
6. (EP) All-to-All Combine: 把 y_i 洗回原 token 位置
7. 输出 = Σ_k  topk_val[k] · y_{topk_idx[k]}     # 加权求和
8. 加上 L_aux 负载均衡损失到总 loss
```

### 7.2 参数量与激活量(数值手算)

以 Mixtral 式配置举例:$N=8$ 专家,$K=2$,$d=4096$,$d_{ff}=14336$,层数 $L=32$。

单个专家的 FFN 参数(SwiGLU 有 3 个矩阵 $W_1,W_3,W_2$):
$$
3 \times d \times d_{ff} = 3 \times 4096 \times 14336 \approx 1.76\times10^8 \approx 176\text{M}
$$

- **每层专家总参数** $\approx 8 \times 176\text{M} = 1.41\text{B}$
- **每个 token 实际激活的专家参数** $\approx 2 \times 176\text{M} = 352\text{M}$(只走 2 个)
- **激活比例** $= 2/8 = 25\%$ ——意味着 MoE 层算力只相当于"2 个专家"的稠密 FFN,却拥有"8 个专家"的知识容量。

把注意力(各层约 $4d^2 \approx 67\text{M}$)算进去后,总量级:总参约 47B、激活约 13B(与公开数字一致,精确值见原文)。

### 7.3 加速比直觉

相比"用同样总参数量的稠密模型",MoE 的每 token 计算量约为:

$$
\frac{\text{MoE FLOPs}}{\text{等参稠密 FLOPs}} \approx \frac{K + (\text{注意力等共享部分})}{N + (\text{共享部分})}
$$

专家越多($N$ 大)、选得越少($K$ 小),省得越狠——但通信、负载均衡难度、显存(总参数仍要全部存!)同步上升。**注意:MoE 省的是计算(FLOPs)和带宽内的激活,不省总显存——所有专家参数都得驻留显存。** 这也是 MoE 推理部署的核心约束(见 [[llm-inference/vllm/README]] 对 MoE 的支持)。

## 8. 评价 / 对照 / 局限

| 维度 | 稠密 Dense | 稀疏 MoE |
|---|---|---|
| 同等**计算预算**下的效果 | 基线 | 通常更好(等价更大模型) |
| 总参数 / 知识容量 | 受算力限制 | 可极大,解耦于算力 |
| 每 token 计算 FLOPs | 高 | 低(只激活 K/N) |
| 显存占用 | 中 | **高**(所有专家都要存) |
| 训练稳定性 | 稳 | 较难(路由坍缩、loss 抖动) |
| 通信开销 | 常规 AllReduce | 额外 **两次 All-to-All** |
| 推理部署复杂度 | 简单 | 复杂(EP、专家放置、batch 不均) |
| 微调/迁移 | 直接 | 路由可能漂移,需小心 |

**主要局限(护栏:以下为通识性结论,具体数值以原文为准):**
1. **显存墙**:总参数全驻留,小显存设备难部署(常配合量化,见 [[llm-compression/quantization/量化基础]])。
2. **通信墙**:All-to-All 对跨节点带宽敏感,扩展到多节点时易成瓶颈。
3. **训练不稳**:需要 aux loss / 容量因子 / z-loss / 精细初始化,调参成本高。
4. **批次不均**:推理时不同请求路由分布不同,导致专家负载抖动,batch 利用率下降。
5. **知识冗余**:专家间可能学到重复知识(共享专家、细粒度切分是缓解手段)。

**什么时候用 MoE?** 当你"有充足显存/带宽,想在固定算力预算下榨出最强效果",且能承担工程复杂度时。反之,边缘设备、小规模、追求部署简单的场景,稠密模型仍是更省心的选择。

---

## 🔗 跳转链接

- 总览枢纽:[[00-知识地图]]
- 架构前置:[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]]
- 并行与通信:[[llm-train/pytorch/distribution/README]] · [[ai-infra/网络/集合通信原语]] · [[B07:llm-inference/大模型推理张量并行]]
- 推理部署:[[llm-inference/vllm/README]] · [[llm-inference/PD分离]] · [[llm-inference/KV-Cache优化]]
- 压缩配套:[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]] · [[llm-compression/sparsity/README]]
- 性能口径:[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
