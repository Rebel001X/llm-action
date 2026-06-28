# MoE 并行（专家并行 Expert Parallelism）

> 把 FFN 换成"一群专家 + 一个分发器"，让总参数量爆炸式增长、而单 token 的计算量几乎不变——这就是稀疏激活；当专家放不进单卡时，就需要"专家并行"把不同专家切到不同设备上。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-algo/moe/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
| --- | --- | --- |
| 0. 一句话锚点 | MoE 与专家并行到底在干嘛 | 稀疏激活 / 专家切分 |
| 1. 地基 | 从 dense FFN 到 MoE 的演化 | FFN / Gating / 专家 |
| 2. 稀疏激活原理 | 为什么参数多了但算力没涨 | Top-k / 激活参数 |
| 3. 分发器（Gating） | token 怎么被路由到专家 | Top-2 / 负载均衡 loss |
| 4. 容量与丢弃 | 容量因子 C 与 token drop | capacity factor |
| 5. 专家并行的通信 | All-to-All 是 EP 的命门 | All-to-All / EP |
| 6. 经典模型谱系 | GShard→Switch→GLaM→Pathways | Top-1 vs Top-2 |
| 实操 | 框架与论文资料（原文真料） | FastMoE / SmartMoE |
| 常见问题 | 路由崩塌、显存、通信坑 | 负载均衡 / All-to-All |

## 0. 一句话锚点

- **MoE（Mixture-of-Experts）**：把 Transformer 里的 FFN 层，从"一个 FFN"换成"N 个 FFN（专家）+ 一个分发器（Gating）"。每个 token 只被送到 Top-k 个专家（通常 k=1 或 2），所以**总参数 ∝ N，但单 token 计算量 ∝ k**。
- **专家并行（Expert Parallelism, EP）**：当 N 个专家放不进一张卡时，把不同专家分配到不同 GPU 上。token 要"飞"到自己被路由到的专家所在的卡，再"飞"回来——靠 **All-to-All** 通信完成。
- **本质对照**：数据并行复制模型、张量并行切一层算子、流水并行切层；**专家并行切的是"专家维度"**，是一种和算力维度正交的稀疏切分。

## 1. 地基：从 dense FFN 到 MoE

> 引用自原文（GShard 论文笔记）：
> Mixture-of-Experts 结构的模型更像是一个**智囊团**，里面有多个专家，你的问题会分配给最相关的一个或多个专家，综合他们的意见得到最终结果。
> 为了实现这个结构，显而易见需要两部分：
> 1）**分发器**：根据你的问题决定应该问哪些专家
> 2）**一群各有所长的专家**：根据分发器分过来的问题做解答
> 3）（可选）**综合器**：很多专家如果同时给出了意见，决定如何整合这些意见……其实就是根据问题，给各个专家分配一个权重。

普通 Transformer 的 FFN（两层全连接）在 MoE 里被替换成红框里的 MoE 结构。**MoE 里面的"专家"依旧是 FFN**，只是从单个 FFN 换成了一群 FFN，又加了一个分发器（Gating）。分发器的任务是把不同的 token 分发给不同的专家。

```
        Dense Transformer FFN                MoE FFN（专家层）
   ┌─────────────────────────┐      ┌──────────────────────────────────┐
   │   x  ──► [ FFN ] ──► y   │      │   x ──► [Gating] ──► 选 Top-k     │
   │        (W1,W2 全激活)    │      │            │                     │
   └─────────────────────────┘      │     ┌──────┼──────┬──────┐        │
                                     │   [E1]  [E2]  [E3] ... [EN]      │
        所有参数都被算一遍           │     └──────┴──────┴──────┘        │
                                     │   只有被选中的 k 个专家被计算     │
                                     │   y = Σ g_i · E_i(x)             │
                                     └──────────────────────────────────┘
```

## 2. 稀疏激活原理：参数涨了，算力没涨

设隐藏维度 $d$，FFN 中间维度 $d_{ff}$，专家数 $N$，每 token 选 $k$ 个专家。

- **dense FFN 参数量**：约 $2 d \cdot d_{ff}$（W1、W2 两个矩阵）。
- **MoE 层参数量**：约 $N \cdot 2 d \cdot d_{ff}$（N 份专家）+ 一个很小的 Gating 矩阵 $d \cdot N$。
- **单 token 计算量（FLOPs）**：只算被选中的 $k$ 个专家，约 $k \cdot 2 \cdot 2 d \cdot d_{ff}$，与 $N$ **无关**。

**数值手算**（$d=4096,\ d_{ff}=4\times d=16384,\ N=64,\ k=2$）：

| 量 | dense | MoE |
| --- | --- | --- |
| FFN 参数（每层） | $2\cdot4096\cdot16384\approx1.34\times10^8$ | $\times64\approx8.6\times10^9$（涨 64 倍） |
| 单 token 计算 | $\propto 1$ | $\propto k=2$（只涨 2 倍） |

> 一句话：**用 64 倍的参数（更强的容量），换来只有 2 倍的单 token 算力**。这就是 MoE 能把模型做到万亿参数、却推理得起的根本原因。

## 3. 分发器（Gating）：token 怎么被路由

Gating 是一个小线性层：$h = \text{softmax}(x W_g)\in\mathbb{R}^{N}$，给每个专家打分，取分最高的 Top-k 个，门控权重做归一化后加权求和：$y=\sum_{i\in\text{TopK}} g_i\, E_i(x)$。

> 引用自原文（GShard 笔记）——为什么分发要"均匀"：
> 对于分发器来说，在训练过程中，**最好把 token 平均分配给各个专家**：不然有些专家闲着，有些专家一堆事，会影响训练速度，而且那些整天无所事事的专家肯定最后训练的效果不好。因此分发器有一个很重要的任务，就是尽可能把 token 均分给各个专家。
> 为了完成这个目标，有一些繁琐的设定：
> 1）**引入了一个 loss**，专门用来控制分发器分发得怎么样：如果把 token 都分给一个人，loss 就很高，分得越均匀（最好彻底均分），loss 越小。
> 2）**每个 token 最多分配给两个专家**。如果每个 token 哐叽一下发给了所有人，那多专家有什么意义？（专家之间的差别主要就是训练数据的不同引起的。）
> 3）**每个专家每次最多接手 C 个 token**。如果一个专家成天"教练我想打篮球/我想唱/我想 rapper"……那估计最后学出来也是四不像。

这三条对应三个核心机制：**负载均衡 loss**、**Top-k 路由**、**专家容量 C**。

```
  tokens: t1 t2 t3 t4 t5 t6 ...
           │  │  │  │  │  │
        ┌──┴──┴──┴──┴──┴──┴──┐
        │      Gating        │  softmax(x·Wg) 打分
        └──┬──────────┬──────┘
     Top-2 │          │ Top-2
       ┌───┴──┐   ┌───┴──┐
     [E1]   [E2] [E3]  [E4] ...   每个专家容量上限 = C
      ▲ 满了的专家会丢弃多余 token（token drop）
```

### 负载均衡 loss（直觉公式）

记 $f_i$ 为路由到专家 $i$ 的 token 占比，$P_i$ 为 Gating 给专家 $i$ 的平均概率，则辅助损失约为 $L_{aux}=N\cdot\sum_i f_i\,P_i$。当所有专家被均匀使用（$f_i=P_i=1/N$）时 $L_{aux}$ 最小，**逼迫路由别"扎堆"**。这个 loss 加到主损失上一起反传。

### 路由数值手算（N=4，Top-2）

某 token 的 Gating logits 经 softmax 得到概率 $[E_1,E_2,E_3,E_4]=[0.5,\,0.3,\,0.15,\,0.05]$：

1. 取 Top-2 → 选中 $E_1(0.5)$ 与 $E_2(0.3)$，丢弃 $E_3,E_4$。
2. 门控权重重归一化：$g_1=\frac{0.5}{0.5+0.3}=0.625$，$g_2=\frac{0.3}{0.8}=0.375$。
3. 输出 $y=0.625\cdot E_1(x)+0.375\cdot E_2(x)$。

若此刻 $E_1$ 已经收满 $C$ 个 token，则该 token 对 $E_1$ 的部分被 drop，只剩 $g_2\cdot E_2(x)$（甚至全 drop 只走残差）。这就是"容量"如何直接吃掉精度的微观过程。

## 4. 容量因子 C 与 token drop

每个专家每个 batch 只接收固定数量的 token，叫**专家容量**：

$$C=\text{capacity\_factor}\times \frac{\text{tokens}\times k}{N}$$

- `capacity_factor` 常取 **1.0 ~ 1.25**（训练）或更高（推理）。
- 超过容量的 token 被**丢弃（drop）**——它们的 FFN 输出为 0，只保留残差连接，相当于这一层对它"摆烂"。
- 容量太小 → 丢弃多、精度掉；容量太大 → 显存/通信浪费、padding 空算。这是 MoE 调参的核心权衡。

## 5. 专家并行的通信：All-to-All 是命门

专家放在不同 GPU 上时，token 必须先"飞"到目标专家的卡上算，再"飞"回来。两次跨卡搬运都是 **All-to-All** 集合通信（详见 [[ai-infra/网络/集合通信原语]]）。

```
  EP=4，每卡放 N/4 个专家
  GPU0[E0,E1]  GPU1[E2,E3]  GPU2[E4,E5]  GPU3[E6,E7]

  ① Dispatch（All-to-All）：本卡 token 按路由结果发往目标卡
       GPU0 的 t→E5 ──────────────►  GPU2
  ② Expert compute：各卡在本地专家上算 FFN
  ③ Combine（All-to-All）：把结果送回 token 原来的卡
       GPU2 算完 ──────────────────►  GPU0
```

- All-to-All 的通信量随 EP 规模、序列长度、隐藏维度线性增长，**通常是 MoE 训练/推理的主要瓶颈**，对网络带宽（NVLink / [[ai-infra/网络/InfiniBand]]）极敏感。
- 路由不均会让某些卡收到远超容量的 token → 通信与计算双双倾斜（落后者 straggler）。所以**负载均衡不只是精度问题，更是性能问题**。
- 工程上常与张量并行（[[ai-framework/megatron-lm/README]]）、数据并行组合成多维并行（[[multidimensional-hybrid-parallel/README]]）。

### All-to-All 通信量手算

每卡有 $T$ 个 token、隐藏维度 $d$、Top-k 路由、bf16（2 字节）。Dispatch 阶段每卡要发出 $T\cdot k$ 份 token 向量，Combine 对称返回同样大小：

$$\text{单卡单次 All-to-All 字节} \approx T\cdot k\cdot d\cdot 2$$

代入 $T=4096,\ k=2,\ d=4096$：约 $4096\times2\times4096\times2\ \text{B}\approx 134\ \text{MB}$；Dispatch + Combine 两趟即约 **268 MB/层/卡**。模型几十层叠加，**这就是为什么 MoE 训练对 NVLink / InfiniBand 带宽如此敏感**——通信量和 EP 规模、序列长度同步放大。

### EP 与其他并行的组合（正交切分）

| 并行维度 | 切什么 | 通信原语 |
| --- | --- | --- |
| 数据并行 DP | 复制整模型、切 batch | All-Reduce（梯度） |
| 张量并行 TP | 切一层算子的权重矩阵 | All-Reduce / All-Gather |
| 流水并行 PP | 切层（stage） | P2P send/recv |
| **专家并行 EP** | **切专家维度** | **All-to-All** |

EP 与 DP/TP/PP 维度正交，可叠加成 "DP×TP×PP×EP" 的多维网格。常见组合：MoE 层走 EP，非 MoE 部分（attention/共享 FFN）走 TP+PP，整体再套 DP。注意 **EP 组内的 token 必须能 All-to-All 互达**，所以 EP 一般优先放在带宽最高的同节点 NVLink 域内。

## 6. 经典模型谱系：GShard → Switch → GLaM → Pathways

代表作（原文）：**GShard，Switch-Transformer，GLaM**。

### GShard（Top-2）

> 引用自原文：
> GShard，按照文章的说法，是**第一个将 MoE 的思想拓展到 Transformer 上**的工作。具体的做法是，把 Transformer 的 encoder 和 decoder 中，**每隔一个（every other）的 FFN 层，替换成 position-wise 的 MoE 层**，使用的都是 **Top-2 gating network**。

### Switch Transformer（Top-1，最稀疏）

> 引用自原文：
> 跟其他 MoE 模型的一个显著不同就是，Switch Transformer 的 gating network 每次只 route 到 **1 个 expert**，而其他的模型都是至少 2 个。这样就是**最稀疏的 MoE** 了，因此单单从 MoE layer 的计算效率上讲是最高的了。

### Pathways（理想）与 PaLM（现实）

> 引用自原文：
> 当前模型的主要问题：基本都是一个模型做一个任务；在一个通用模型上继续 fine-tune 会遗忘很多其他知识；基本都是单模态；**基本都是 dense 模型，在完成一个任务时（不管难易）网络的所有参数都被激活和使用**。
> Pathways 的愿景——一个更接近人脑的框架：一个模型，可以做多任务、多模态；**sparse model，在做任务时只是 sparsely activated，只使用一部分参数**。

### 对照表

| 模型 | 路由 Top-k | 替换策略 | 一句话定位 |
| --- | --- | --- | --- |
| GShard | Top-2 | 每隔一层 FFN→MoE | 第一个把 MoE 搬上 Transformer |
| Switch Transformer | **Top-1** | FFN→MoE | 最稀疏，MoE 层算力最高效 |
| GLaM | Top-2 | 部分层 MoE | 大规模稀疏 LM，激活参数远小于总参数 |
| Pathways/PaLM | —— | 框架愿景 | sparse、多任务多模态的理想框架 |

## 实操：框架与论文资料（原文真料）

**MoE 训练框架 / 实现**

- FastMoE：https://github.com/laekov/fastmoe
- SmartMoE：https://github.com/zms1999/SmartMoE
- 飞桨-MOE：https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/moe_cn.html

**经典论文与笔记**

- GShard-MoE：https://arxiv.org/abs/2006.16668
- 代表模型：GShard、Switch-Transformer、GLaM
- Mixture-of-Experts (MoE) 经典论文一览：https://zhuanlan.zhihu.com/p/542465517
- Google 的 Pathways（理想）与 PaLM（现实）：https://zhuanlan.zhihu.com/p/541281939
- GShard 论文笔记（1）-MoE 结构：https://zhuanlan.zhihu.com/p/344344373
- 参考博客：https://blog.csdn.net/qq_41185868/article/details/103219988

> 选型直觉：研究/小规模快速验证可用 **FastMoE**；要自动化的专家放置与自适应并行策略，关注 **SmartMoE**；生产级大模型训练通常走 [[ai-framework/megatron-lm/README]]（Megatron-Core MoE）或 [[ai-framework/deepspeed/README]]（DeepSpeed-MoE）。

## 常见问题 / 坑

| 现象 | 根因 | 应对 |
| --- | --- | --- |
| 路由崩塌：所有 token 挤进少数专家 | 负载均衡 loss 权重太小 / 训练初期门控不稳 | 调大 aux loss 系数、加 router z-loss、用 noisy gating |
| 精度掉但 loss 看着正常 | 容量太小，大量 token 被 drop | 提高 capacity_factor（如 1.0→1.25），或换 dropless 实现 |
| 某些卡卡顿、整体变慢 | 专家负载倾斜 → All-to-All 与计算 straggler | 均衡路由 + 专家放置均衡 + 调度对齐 |
| 显存爆 | 总参数 ∝ N，专家全放一卡放不下 | 用专家并行 EP 把专家切到多卡 |
| All-to-All 成瓶颈 | EP 跨节点、带宽不足 | 优先同节点 NVLink、用 IB/RDMA、减小 EP 规模或重叠通信 |
| 推理 batch 小、专家利用率低 | token 太少摊不满 N 个专家 | 推理侧提高容量/合并请求、考虑专家合并或量化 |
| 增大 N 但效果不涨 | 专家间区分度不足（数据相似） | 专家差异来自训练数据分布；检查路由是否真在"分工" |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 上游算法：[[llm-algo/moe/README]] · [[llm-algo/transformer/模型架构]] · [[llm-algo/旋转编码RoPE]]
- 其他并行：[[data-parallelism/README]] · [[tensor-parallel/README]] · [[pipeline-parallelism/README]] · [[multidimensional-hybrid-parallel/README]] · [[auto-parallel/README]]
- 框架：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]]
- 硬件 / 网络：[[ai-infra/网络/集合通信原语]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 优化 / 推理：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]] · [[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 压缩 / 量化：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 训练 / 对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 评测 / 估算：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
