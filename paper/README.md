# paper · 大模型论文精读索引（Paper Reading MOC）

> 本目录是 llm-action 的「论文精读」分区：把一篇篇 LLM/AI-Infra 经典论文拆到「到底做了什么、为什么重要」的程度。本文件是这些精读笔记的**导航中枢 + 阅读地图**。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/PD分离]] · [[llm-optimizer/kv-cache]] · [[llm-alignment/RLHF]] · [[llm-compression/sparsity/README]]

---

## 阅读地图（先看这张表）

| 子目录/文件 | 论文 | 一句话定位 | 配套讲义（去看富笔记） |
|---|---|---|---|
| `PagedAttention.md` | PagedAttention / vLLM | 用 OS 分页思想管 KV Cache，几乎消灭显存碎片 | [[llm-optimizer/kv-cache]] · [[llm-inference/vllm/README]] |
| `inference/orca.md` | Orca | **迭代级（continuous batching）**调度，吞吐数量级提升 | [[llm-inference/连续批处理]] |
| `inference/llm-in-a-flash.md` | LLM in a Flash | 把权重放 Flash/SSD，靠稀疏激活做内存换 IO | [[llm-compression/sparsity/README]] |
| `inference/迈向高效的生成式大语言模型服务综述.md` | Efficient Serving Survey | 从「算法→系统」两层梳理推理优化全貌 | [[llm-inference/README]] |
| `training/A Survey on Efficient Training of Transformers.md` | 高效训练综述 | 计算/内存/通信三条线全景 | [[llm-train/README]] |
| `training/Reducing Activation Recomputation.md` | 选择性激活重计算 | 只重算「便宜」的层，省显存又少算 | [[docs/transformer内存估算]] |
| `training/GaLore.md` | GaLore | 梯度低秩投影，全参微调省优化器显存 | [[llm-train/pytorch/distribution/README]] |
| `parameter-pruning/SparseGPT.md` | SparseGPT | 一次性（one-shot）剪枝，无需重训 | [[llm-compression/sparsity/README]] |
| `parameter-pruning/Wanda.md` | Wanda | 权重×激活幅度做剪枝，极简无需求逆 | [[llm-compression/sparsity/README]] |
| `parameter-pruning/LLM-Pruner.md` | LLM-Pruner | 结构化剪枝 + LoRA 恢复 | [[llm-compression/sparsity/README]] |
| `moe/README.md` | MoE 论文集 | 稀疏专家、路由、负载均衡 | [[llm-algo/moe/README]] |
| `data/LESS.md` | LESS | 用梯度影响力挑「最有用」的指令数据 | [[llm-data-engineering/README]] |
| `llm对齐综述.md` | Alignment Survey | RLHF/RLAIF/PPO/DPO 全谱系 | [[llm-alignment/RLHF]] · [[llm-alignment/DPO]] |
| `LLM增强LLMS.md` | CALM | 用 cross-attention 把小模型「插」进大模型 | [[llm-algo/transformer/模型架构]] |

> 体例：每篇精读尽量包含——**问题背景 → 核心方法（拆到能懂）→ 关键创新 → 关键公式/算法 → 实验结论（数字标"约/见原文"）→ 工程启示 → 局限**。

---

## 0. 一句话锚点

> **这个目录回答一个问题：当我读到一篇 LLM 论文标题，它到底解决了什么瓶颈、用了什么招、对我的工程有什么用。** 论文很多，但底层就在抢三样东西——**算力（FLOPs）、显存（HBM 容量+带宽）、通信（卡间带宽）**。下面把论文按「抢哪样资源」归类，你就有了一张地图。

```
                    一篇 LLM 系统论文 = 在三角形里挪动瓶颈
                          算力 FLOPs
                            /\
                           /  \   GaLore(省优化器态)
        FlashAttn / 重计算/    \  激活重计算(算换存)
                       /        \
                      /__________\
              显存 HBM ---------- 通信 带宽
          PagedAttn/vLLM        TP/PP/EP(MoE)
          KV量化/剪枝            Orca(调度榨吞吐)
```

---

## 1. 地基：读这些论文前，先有这几个量纲感

读系统类论文，脑子里要先挂三本「账」。后面每篇论文都是在某本账上做文章。

### 1.1 显存账（一张卡到底装得下什么）

推理时一张卡的 HBM 主要被三块吃掉：

```
HBM 占用 ≈ 权重  +  KV Cache  +  激活/临时
          (静态)    (随并发×序列增长)   (前向用完即弃)
```

- **权重**：参数量 × 每参数字节。FP16 下 7B≈14GB，70B≈140GB（见原文/官方）。
- **KV Cache**：这是推理显存的「变量」，也是 PagedAttention 的战场。
  $$\text{KV bytes} = 2 \times L \times n_{kv} \times d_{head} \times \text{batch} \times \text{seqlen} \times \text{bytes}$$
  其中 $2$ 是 K 和 V，$L$ 层数，$n_{kv}$ 是 KV 头数（GQA 下远小于 Q 头数），$d_{head}$ 头维度。

**手算一笔**（LLaMA-7B：$L{=}32$，$n_{kv}{=}32$，$d_{head}{=}128$，FP16=2B，单条 2048 token）：

$$2 \times 32 \times 32 \times 128 \times 1 \times 2048 \times 2 \approx 1.07\times10^9 \text{ B} \approx 1.0\text{ GB / 条}$$

→ 16 并发就 ~16GB KV，比模型权重还大。**所以 KV Cache 怎么省、怎么不浪费，是推理论文的头号主题**（PagedAttention、KV 量化、剪枝、PD 分离都从这里出发）。

### 1.2 算力账（为什么 prefill 和 decode 不一样）

- **Prefill**（处理 prompt）：一次算 $S$ 个 token，是大矩阵乘 → **算力受限（compute-bound）**，GPU 利用率高。
- **Decode**（逐 token 生成）：一次算 1 个 token，但要读全部权重+KV → **访存受限（memory-bound）**，GPU 算力大量闲置。

这条「prefill 胖、decode 瘦」的差异，直接催生了 **continuous batching（Orca）** 和 **PD 分离** 两类论文。

### 1.3 通信账（多卡时谁在等谁）

- 张量并行 TP：每层 attention/FFN 后一次 `all-reduce`，通信频繁、要高带宽（NVLink）。
- 流水并行 PP：层间传激活，通信少但有「气泡」。
- 专家并行 EP（MoE）：`all-to-all` 把 token 路由到专家所在卡。

> 这三本账是后面所有论文的「计价单位」。看到一个方法，就问：它在省算力、省显存、还是省通信？代价是什么？

---

## 2. 推理派：榨显存与吞吐（inference/ + PagedAttention）

### 2.1 PagedAttention / vLLM —— 把 OS 分页搬进 KV Cache

**问题**：传统推理给每条请求**连续**预留「最大长度」的 KV 显存。结果是：① 序列没生成那么长 → **内部碎片**；② 不同请求长度不一、预留块大小不齐 → **外部碎片**。论文测得连续分配下显存有效利用率可低到「约 20–40%」（见原文）。

**核心方法（拆到能懂）**：照搬操作系统「虚拟内存分页」。

```
逻辑视角(每条请求看到的连续KV)        物理视角(显存里的非连续块)
请求A: [blk0][blk1][blk2]  --映射-->   物理块池: [P7][P3][P1][P9][P2]...
请求B: [blk0][blk1]        --映射-->   (块大小固定,如16 token/块)
       ↑ 块表(block table)把逻辑块号翻译成物理块号
```

- KV 切成固定大小的**块（block）**，逻辑连续、物理离散，用**块表**做翻译。
- 只在需要时分配新块 → 碎片几乎归零，显存利用率「接近满」（见原文，约 96%+）。
- **写时复制（Copy-on-Write）**：并行采样/beam search 共享同一段 prompt 的 KV 块，只在分叉处复制。

**关键创新**：① 非连续显存 + 块表；② 共享与 CoW 让「一个 prompt 多个输出」几乎零额外显存。
**实验结论**：相比 FasterTransformer/Orca，吞吐提升「约 2–4 倍」（见原文），在长序列/高并发下更明显。
**工程启示**：这就是 vLLM 成为事实标准推理引擎的根。**详细讲义 → [[llm-optimizer/kv-cache]] · [[llm-inference/vllm/README]]**。
**局限**：块表查找有少量 overhead；自定义 attention kernel 才能高效读非连续块。

### 2.2 Orca —— 迭代级调度（continuous batching 的起点）

**问题**：传统**请求级**批处理，整个 batch 必须等最慢的那条生成完才能返回、才能换下一批 → 短请求被长请求「绑架」，GPU 大量空转。

**核心方法**：把调度粒度从「一个请求」降到「一次迭代（一个 token step）」。

```
请求级批处理(旧):  [====A长====]
                   [==B==]......(B早完但被迫等A)  ← 空转
迭代级批处理(Orca): step1: A B C D
                   step2: A   C D E   (B完成立刻退出,E立刻补进来)
                   step3: A     D E F
```

- **iteration-level scheduling**：每个 step 重新组 batch，谁完成谁走、新请求随时插入。
- **selective batching**：attention 因各条 seqlen 不同不能简单堆叠，FFN/LN 等逐 token 算子则可拼成大 batch → 分开处理。

**实验结论**：相同延迟下吞吐较 FasterTransformer 提升「约一个数量级（见原文）」。
**工程启示**：今天 vLLM/TGI 的 **continuous batching** 就源于此。**讲义 → [[llm-inference/连续批处理]]**。
**局限**：实现复杂（要处理变长 attention）；KV 显存管理仍需配合 PagedAttention 才完整。

### 2.3 LLM in a Flash —— 让大模型在「装不下」的设备上跑

**问题**：模型权重 > DRAM 容量（手机/边缘）。能不能把权重放 Flash（SSD），按需取？瓶颈是 Flash 带宽远低于 DRAM。

**核心方法**：吃 FFN 的**激活稀疏性**（ReLU 系模型大量神经元输出 0，对应权重根本用不到）。
- **窗口化（windowing）**：复用最近若干 token 已加载的神经元，只增量加载新需要的。
- **行列捆绑（row-column bundling）**：按 Flash 读「大块连续」更快的特性，重排权重布局，减少随机小读。

**工程启示**：稀疏激活 = 可预测的「哪些权重不用」= 内存换 IO 的钥匙。**关联 → [[llm-compression/sparsity/README]]**。
**局限**：依赖高稀疏度（ReLU 类）；GeLU/SwiGLU 稠密模型收益小。

### 2.4 高效生成式服务综述 —— 一张「算法↔系统」全景图

把推理优化分两层：
- **算法层**：解码（投机解码/并行解码）、KV 压缩（量化/剪枝/共享）、注意力近似。
- **系统层**：批处理（continuous）、调度（PD 分离）、显存（PagedAttention）、并行（TP/PP）。

**用途**：当索引读。**主线讲义 → [[llm-inference/README]] · [[llm-inference/PD分离]] · [[llm-inference/KV-Cache优化]]**。

---

## 3. 训练派：省显存、省重算（training/）

### 3.1 选择性激活重计算（Reducing Activation Recomputation）

**问题**：训练显存被**激活（前向中间结果，反向要用）**吃掉一大块。经典「全量重计算（gradient checkpointing）」省显存，但反向时整层重算一遍前向 → **算力多花约 1/3**。

**核心思想**：不是全有或全无。激活按「存它省的显存 / 重算它花的算力」排序，**只丢弃那些重算便宜、占显存大的部分**（论文重点点名 attention 里的 softmax/dropout 等）。

```
全量重计算:  存极少 → 反向重算整层(算力↑↑)
选择性重计算: 存"贵算"的(matmul结果),丢"便算"的(softmax/dropout) → 显存省九成,算力只多几个%
```

配合**序列并行（sequence parallel）**把 LN/dropout 的激活沿序列维切到多卡，进一步降单卡显存。
**实验结论**：激活显存大幅下降，重计算开销从「约 30%+」降到「个位数 %」（见原文）。
**工程启示**：Megatron 默认开法之一。**讲义 → [[docs/transformer内存估算]]**。

### 3.2 GaLore —— 梯度低秩投影，全参微调也省显存

**问题**：Adam 优化器要为每个参数存一阶+二阶动量 → 优化器态约是参数量的 2 倍（FP32 下更多）。LoRA 省了这块但只训低秩旁路，**不是全参训练**。

**核心方法**：观察到**梯度矩阵本身近似低秩**。于是把梯度投影到低秩子空间 $G_{low}=P^\top G$，**优化器态只在低秩空间维护**，更新后再投影回去 $\tilde G = P G_{low}$。

$$G \in \mathbb{R}^{m\times n} \;\xrightarrow{\;P\in\mathbb{R}^{m\times r}\;}\; G_{low}\in\mathbb{R}^{r\times n}, \quad r \ll m$$

- 关键：**权重仍是全参更新**（区别于 LoRA），只有优化器**状态**被压到低秩 → 兼顾效果与省显存。
- 投影矩阵 $P$ 用梯度的 SVD 周期性刷新。

**实验结论**：可在「约 24GB 单卡（见原文）」预训练 7B；优化器显存大降。
**工程启示**：要全参效果又显存紧张时的备选。**关联 → [[llm-train/pytorch/distribution/README]]**。
**局限**：SVD 有周期性开销；超参（秩 $r$、刷新间隔）需调。

### 3.3 高效训练 Transformer 综述

按**计算 / 内存 / 通信**三轴归类所有招式：混合精度、重计算、ZeRO、并行策略、稀疏/量化训练等。**当训练优化的索引读 → [[llm-train/README]]**。

---

## 4. 剪枝派：让大模型「变稀疏」（parameter-pruning/）

三篇核心论文递进关系：

```
难度/效果轴
Magnitude(只看权重大小,最简单,效果差)
   │
   ▼  Wanda  = 权重幅度 × 输入激活幅度  (加一个激活项,几乎零成本,效果反超)
   │
   ▼  SparseGPT = 逐层最小化重建误差,用Hessian逆做最优补偿  (one-shot,效果最好,要算逆)
   │
   ▼  LLM-Pruner = 结构化剪整块(头/通道) + LoRA微调恢复  (能真加速,但要恢复训练)
```

### 4.1 SparseGPT —— 一次性剪枝，不重训

**问题**：传统剪枝要「剪→重训」循环，大模型重训不起。
**核心方法**：把剪枝看成**逐层的稀疏重建**：剪掉一部分权重后，**调整剩下的权重去补偿误差**，使该层输出尽量不变。用 OBS（Optimal Brain Surgeon）思想，借 Hessian 逆做闭式补偿，并用近似让它能跑在 175B 上。
$$\min_{\hat W}\; \| WX - \hat W X \|_2^2 \quad\text{s.t. }\hat W \text{ 满足稀疏掩码}$$
**结论**：50% 稀疏下精度损失「很小（见原文）」，单卡数小时完成 175B。

### 4.2 Wanda —— 极简到「不敢相信有效」

**核心创新**：剪枝重要性 = **权重绝对值 × 该列输入激活的 L2 范数**：
$$S_{ij} = |W_{ij}| \cdot \|X_j\|_2$$
不需要求 Hessian 逆、不需要更新权重，**一行公式**，效果却追平甚至超过 SparseGPT。**启示**：激活幅度承载了「这个权重实际有多重要」的信息。

### 4.3 LLM-Pruner —— 结构化 + 恢复

剪的是**整块结构**（注意力头、FFN 通道），所以能在硬件上**真实加速**（非结构化稀疏难加速）；剪完用少量数据 + LoRA 快速恢复精度。
**讲义汇总 → [[llm-compression/sparsity/README]]**，公式细节见同目录 `公式.md`。

---

## 5. MoE / 数据 / 对齐 / 组合（moe、data、对齐综述、CALM）

| 论文 | 它做了什么 | 关键创新 | 去哪看富笔记 |
|---|---|---|---|
| **MoE 系列**（`moe/`） | 用稀疏门控让「参数多但每 token 只激活少数专家」 | Top-k 路由 + 负载均衡损失，算力≈稠密小模型、容量≈稠密大模型 | [[llm-algo/moe/README]] |
| **LESS**（`data/`） | 从大数据池挑「对目标任务最有用」的子集 | 用**梯度影响力**（训练样本梯度与验证集梯度的相似度）打分，少量数据≈全量效果 | [[llm-data-engineering/README]] |
| **对齐综述** | RLHF/RLAIF/PPO/DPO 全谱系梳理 | 把「奖励建模→策略优化」与「免奖励模型（DPO）」两条路对照清楚 | [[llm-alignment/RLHF]] · [[llm-alignment/DPO]] |
| **CALM**（`LLM增强LLMS`） | 不改两个模型权重，用 **cross-attention** 把小模型能力「插进」大模型 | 冻结双模型，只训中间的组合层 → 低成本扩展能力 | [[llm-algo/transformer/模型架构]] |

**MoE 算力直觉（手算）**：稠密 $d{=}4096$ 的 FFN 每 token ≈ $2\times(d\times 4d \times 2)$ FLOPs；MoE 有 8 专家但只激活 Top-2 → **算力≈稠密的 2/8 那层规模**，而**总参数 ≈ 8 倍** → 「参数膨胀、算力不膨胀」正是 MoE 的卖点。代价是**显存（要存所有专家）+ all-to-all 通信**。详见 [[llm-algo/moe/README]]。

---

## 评价 / 对照 / 局限（一表看清这些论文在抢什么）

| 论文 | 主攻瓶颈 | 核心一招 | 代价 / 局限 | 是否已成工业标配 |
|---|---|---|---|---|
| PagedAttention | KV 显存碎片 | OS 分页 + 块表 | 需定制 kernel | ✅ vLLM 默认 |
| Orca | decode 吞吐 | 迭代级调度 | 变长 attention 难实现 | ✅ continuous batching |
| LLM in a Flash | DRAM 装不下 | 稀疏激活换 IO | 依赖高稀疏度 | 边缘端 |
| 激活重计算 | 训练显存 | 选择性重算 | 仍多算一点 | ✅ Megatron |
| GaLore | 优化器显存 | 梯度低秩投影 | SVD 开销/调参 | 备选 |
| SparseGPT | 模型大小 | one-shot 重建剪枝 | 非结构化难加速 | 研究/部分落地 |
| Wanda | 模型大小 | 权重×激活打分 | 同上 | 研究 |
| LLM-Pruner | 真实加速 | 结构化剪+恢复 | 需恢复训练 | 部分落地 |
| MoE | 算力↔容量 | 稀疏门控 | 显存+all-to-all | ✅ DeepSeek/Mixtral |
| LESS | 数据效率 | 梯度影响力选数据 | 计算影响力有成本 | 研究 |
| 对齐综述/CALM | 对齐/能力组合 | 谱系/cross-attn | — | 参考 |

> **护栏**：上表的吞吐/利用率等具体数字均以「约/见原文」为准，请回到各篇论文与官方实现核对；不同硬件/序列长度差异很大。

---

## 怎么用这个目录（建议路径）

1. **想搞推理**：先 PagedAttention → Orca → 服务综述，再跳 [[llm-inference/PD分离]] · [[llm-optimizer/kv-cache]]。
2. **想搞训练省显存**：激活重计算 → GaLore → [[docs/transformer内存估算]]。
3. **想搞压缩**：Wanda（最简）→ SparseGPT（最优）→ LLM-Pruner（能加速）→ [[llm-compression/sparsity/README]]。
4. **想搞 MoE/对齐**：MoE 论文集 → [[llm-algo/moe/README]]；对齐综述 → [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]。

---

## 🔗 跳转链接

- 导航中枢：[[00-知识地图]]
- 推理优化：[[llm-inference/PD分离]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/vllm/README]] · [[llm-inference/连续批处理]] · [[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 压缩剪枝：[[llm-compression/sparsity/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 训练并行：[[llm-train/README]] · [[llm-train/pytorch/distribution/README]] · [[B07:llm-inference/大模型推理张量并行]] · [[docs/transformer内存估算]]
- 算法/对齐：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- Infra：[[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/GPU工作原理]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[llm-data-engineering/README]]
