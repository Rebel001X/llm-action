# RS-4 投机解码的负收益、失效条件与真实基准数据

> 调研日期：2026-08-22
> 定位：本库最重要的一份调研。全库立场——**投机采样是一笔用算力换延迟的交易，这笔交易经常是亏的。** 本文只回答一件事：**什么时候亏，亏多少，怎么提前判定。**
> 本文是**调研底稿**，不是正文。所有数字都带口径；口径不全的一律标注，不参与横向比较。

---

## 0. 阅读须知：本文的取数规则

写正文的人必须先读这一节，否则会把本文的数字用错。

### 0.1 三条硬规则

1. **任何加速比 / 接受率数字都不许裸奔。** 每条记录尽量带齐七项口径：
   `batch size` · `γ 或树规模` · `接受率 α 或平均接受长度 τ` · `draft/target 组合与参数量` · `硬件与精度` · `测的是什么指标(TPOT / 端到端延迟 / 吞吐)` · `任务分布`。
   缺项一律在该条下写「原文未给出 X」，并整条标记 **【口径不全，不可横向比较】**。
2. **禁止跨来源合表比加速比。** 本文所有表格都**按来源分表**，每张表自带口径行。你在正文里想合表，必须先自己核对七项口径一致——通常核对不过。
3. **查不到就写「未查证」。** 第 11 节是诚实缺口清单，正文不许用推测填补。

### 0.2 一个必须先建立的区分：τ ≠ 加速比

这是本次调研中**最高频的错误来源**，先在这里钉死：

- **τ（平均接受长度，average acceptance length）**：目标模型每做一次前向，平均产出多少个 token。是**算法质量**指标。
- **加速比（speedup ratio）**：实测墙钟时间之比。是**系统收益**指标，已经扣掉草稿开销、验证开销、框架开销。

两者数值差距很大。例：EAGLE 在 Vicuna-7B / MT-bench / T=0 下 **τ = 3.94，但加速比只有约 2.90×**（见 §4.1）。
调研过程中我亲眼看到自动摘要工具把 τ=3.94 当成"3.94× 加速"输出——**这个错误在中文二手资料里极其普遍**。详见 §10.1。

同理还有第三个量：**接受率 α（每个草稿 token 被接受的概率）**。α、τ、加速比三者不能互相替代。

### 0.3 证据等级标注

| 标记 | 含义 |
|---|---|
| 【一手·论文自报】 | 方法作者自己测的，通常是最优配置下的上界 |
| 【一手·第三方】 | 独立团队/框架方/工程博客实测 |
| 【一手·issue】 | 用户在框架仓库报的真实现场，口径通常残缺但真实 |
| 【二手·无出处】 | 广泛流传但溯源失败，**不可引用**，仅作为"江湖传言"记录 |
| 【待核验】 | 我尝试核验但未能逐字确认 |

---

## 1. 第一节：什么时候**不要**开投机解码 —— 可执行判定清单

这一节是全文结论的压缩。每条都给出**可测量的触发条件**和**支撑证据的章节号**，不给模糊建议。

### 1.1 红灯：符合任意一条，默认关闭，除非你有本机实测反证

| # | 触发条件（可测量） | 为什么 | 证据 |
|---|---|---|---|
| R1 | **并发/批大小稳定 ≥ 32**，且上下文短（< 4K） | verify 前向从 memory-bound 掉进 compute-bound，草稿 token 不再"免费" | §2.1（EAGLE 在 bs32 = 0.93×，bs56 = 0.71×）、§2.2（Meta：bs2→48，1.3×→0.7×） |
| R2 | **草稿头训练窗口 < 你的实际输入长度**（典型：EAGLE3 公开 checkpoint 训练窗口 2K，你的请求 8K+） | 草稿头外推失败，τ 塌到 ~1.3，纯亏草稿开销 | §3.3（OWL：EAGLE3 在 4K–64K 输入上 **0.81×**，τ=1.28） |
| R3 | **目标模型是 4-bit 权重量化**（W4A16 / W4A8） | 量化已经把权重加载时间压掉，verify 的额外计算再无处可藏 | §7.4（W4A16 8B + EAGLE-2 在 RTX3090 上**零加速增益**；60 草稿 token 时 verify ratio 1.8 vs FP16 的 1.2） |
| R4 | **主要工作负载是低资源语言 / 多语言翻译** | 小草稿模型的多语言能力塌方，α ≈ 0.30–0.54 | §4.4（11 种低资源语言，ᾱ=0.40 → 加速 **1.02×**，等于白干） |
| R5 | **草稿是独立小模型且目标模型 ≤ 8B** | 草稿前向占比过高（bs1 时可达 47%），且草稿模型自己吃 KV cache 挤掉批容量 | §6.1（draft-model 使 per-token 显存 ×1.77）、§6.2（更大草稿模型接受率更高但吞吐更差） |
| R6 | **目标是超大 MoE 且已在 bs1 就只有 ~1.1× 收益** | verify 多个 token 会激活更多专家，验证不再近似免费 | §7.6（DeepSeek-V3.1 + MTP 仅 1.10×；Qwen3-235B + EAGLE-3 仅 1.22×，对比 dense Llama3.1-8B 的 4.40×） |
| R7 | **消费级 GPU + 已经是 memory-bound 到底** | 没有闲置算力可换 | §9.6（2×RTX 5060 Ti：88.3→88.2 tok/s，0%） |

### 1.2 黄灯：必须先做 A/B 实测，不许直接上线

| # | 条件 | 要测什么 |
|---|---|---|
| Y1 | 并发在 **8–32** 之间波动 | 测你自己的交叉点。不同来源交叉点从 bs8 到 bs64 都有（§2.6），**没有通用阈值** |
| Y2 | 温度 > 0 且业务用 T≈0.6–1.0 | T=0→1 加速掉约 30%（§5.1）；n-gram 类 T=0→0.6 掉 22%（§5.3） |
| Y3 | 任务分布里摘要 / 开放对话占比高 | 摘要是 EAGLE-3 全表最差的一档（CNN/DM 3.65× vs HumanEval 4.85×，§4.1） |
| Y4 | 同时开了 prefix caching 且是长上下文重复前缀场景 | MTP 可能让 prefix cache 命中减半，prefill 回退吃掉 decode 收益（§7.2） |
| Y5 | 用了 FlashMLA / 特定 attention backend | 多 token 投机可能被迫走 prefill path（§7.3） |
| Y6 | 服务要跑数小时不重启 | EAGLE 接受长度存在**随运行时间单调退化**的未修复 bug（§9.5） |

### 1.3 绿灯：这些场景收益最稳

| # | 条件 | 证据 |
|---|---|---|
| G1 | bs1–4 的低延迟单请求场景（本地助手、IDE 补全） | §2.1 全表最高档 |
| G2 | 输入输出高度重叠的任务（代码编辑、RAG 改写、agent 循环） | n-gram / suffix decoding 在 InstructCoder 上平均接受 7.27 token（§4.3）；SWE-Bench 上 suffix decoding 75.8→286 tok/s（§8.4） |
| G3 | **长上下文 + 大批**，且草稿侧用定长窗口（StreamingLLM）或自投机 | §3.1 MagicDec：bs32–128、32K prefill 下 1.91–2.0× |
| G4 | 推理/思维链长生成 | §4.3（SAM[EAGLE-3] 多轮思考 3.97×） |

### 1.4 一条元规则

> **投机解码的收益不是模型属性，是"模型 × 硬件 × 负载 × 任务"四元组的属性。**
> 本文收集到的同一个方法（EAGLE）在不同四元组下实测值域是 **0.71× ~ 4.40×**。
> 因此：**任何不带这四项的加速比宣称，都应当被当作营销文案处理。**

---

## 2. Batch 维度：什么时候投机解码开始降低吞吐

机制：投机解码的前提是 decode 阶段 **memory-bound**——GPU 在等权重/KV 从 HBM 搬进来，算力闲置，于是"顺手多验几个 token"几乎免费。批大小上升会把 batch GEMM 的算术强度推过 roofline 拐点，前向变成 **compute-bound**，此时多验的 token 是**真金白银的额外算力**，而被拒绝的部分就是纯浪费。

### 2.1 【一手·论文自报】EAGLE-3 论文 Table 5 —— 目前最干净的一张批大小扫描表

这是本次调研找到的**唯一一张明确穿过 1.0× 的公开批大小扫描表**，也是本文最重要的证据。

**口径**：GPU = RTX 3090；目标模型 = LLaMA-Instruct 3.1 8B；基线 = 未开投机的 vLLM，归一化为 1.00×；指标 = 吞吐比。
**口径缺项**：原文该表**未给出** γ / 树规模、逐 batch 的接受长度、温度、数据集、精度。 → **【口径不全，不可与其它来源横比】**

| batch size | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 56 |
|---|---|---|---|---|---|---|---|---|
| **EAGLE** | 1.30× | 1.25× | 1.21× | 1.10× | **1.03×** | **0.93×** | 0.82× | **0.71×** |
| **EAGLE-3** | 1.75× | 1.68× | 1.58× | 1.49× | 1.42× | 1.36× | 1.21× | 1.01× |

来源：https://arxiv.org/html/2503.01840 （EAGLE-3 论文 Table 5）

**读法（这是本文的核心结论之一）**：
- EAGLE 的**盈亏平衡点在 bs 24 与 32 之间**。bs32 起是净亏损，bs56 时**比不开投机慢 29%**。
- EAGLE-3 把平衡点推到 bs56 附近（1.01× ≈ 打平）。**改进草稿质量的作用，等价于把交叉点往右推**，而不是消除交叉点。
- 交叉点的存在是结构性的，不是实现 bug。任何投机方案都有自己的交叉点，只是位置不同。

⚠️ **该论文正文与表格自相矛盾**：正文写 "EAGLE shows the maximum throughput improvement at a batch size of 24"，但表里 EAGLE 的最大值明明在 bs2（1.30×），bs24 只有 1.03×。详见 §10.2。**引用时以表为准。**

### 2.2 【一手·第三方】Meta：EAGLE 在 vLLM 里从 1.3× 掉到 0.7×

> 原文逐字：*"when the batch size is increased from 2 to 48, the speed-up (measured using vLLM) of EAGLE-based speculative decoding compared to non-speculative decoding drops from 1.3× to 0.7×"*

来源：Meta GenAI & Infra，*Efficient Speculative Decoding for Llama at Scale: Challenges and Solutions*，https://arxiv.org/html/2508.08192

同文的机制陈述（逐字）：
> *"at large batch sizes, decoding becomes compute-bound, resulting in a decrease in speedup."*
> *"At large context lengths, attention dominates the computation even for large batch sizes, and therefore the workload stays memory-bound."*

**Meta 自己优化后的结果**（口径：8×H100；EAGLE-based；生产规模）：
- 大批下加速比 **1.4×–2.0×**（原文：*"our optimizations enable us to achieve a speed-up for large batch sizes between 1.4x and 2.0x at production scale"*）
- Llama4 Maverick：**约 4 ms/token @ batch size 1**，比此前最好方法快 10%

**Tokens-Per-Call (TPC) 表**（口径：MT-Bench；chain-like draft；**temperature=0, top-p=0.9**；**speculation length = 3**）：

| 模型 | TPC |
|---|---|
| Llama3.1 8B | 2.78 |
| Llama3.3 70B | 2.94 |
| Llama4 Scout | 2.87 |
| Llama4 Maverick | 2.75 |

> 注意 TPC ≈ τ，**不是加速比**。γ=3 时理论上限 TPC=4，实测 2.75–2.94，即约 70% 的理论上限。

⚠️ Meta 这句转述与 EAGLE-3 Table 5 **对不上**：Table 5 里 bs48 是 0.82×，0.71× 出现在 bs56。见 §10.3。

### 2.3 【一手·第三方】SqueezeBits：vLLM vs TensorRT-LLM 实测的并发阈值

**口径**：目标 = Llama-3.1-70B-Instruct，BF16，TP=4；硬件 = 4×A100-SXM-80G；框架 = vLLM v0.6.3 / TensorRT-LLM v0.14.0；数据集 = Dynamic-Sonnet，输入 1K/2K，输出 128 token。
来源：https://blog.squeezebits.com/vllm-vs-tensorrtllm-11-speculative-decoding-37301

| 草稿模型 | 输入长度 | 接受率 | 投机解码开始劣于标准解码的并发 |
|---|---|---|---|
| Qwama-0.5B-Instruct | 1K | 53.5%（另一处记为 54.0%，见 §10.5） | **max concurrency > 32** |
| Qwama-0.5B-Instruct | 2K | 50.9% | **max concurrency > 16** |
| Llama-3.1-8B-Instruct | 1K | 76.5% | 原文未给出对应并发阈值 |

**两个关键读数**：
1. **输入变长会把交叉点往左推**（1K→2K，阈值 32→16）。这与 §3 的"长上下文救投机解码"看似矛盾——不矛盾，见 §3.4 的调和。
2. 该文结论逐字：*"speculative decoding excels at reducing TPOT, but only under scenarios with very small batch sizes"*。
3. 当时的 **TensorRT-LLM v0.14.0 只支持 batch size 1 的投机解码**，原文称其对服务场景 *"impractical"*。（版本已老，仅作历史记录）

### 2.4 【一手·第三方】SpecDecode-Bench：批大小上收益单调收窄，但（在其测试范围内）未穿破 1.0×

**口径**：H100 80GB；vLLM v0.10.1.1（含 KV cache / continuous batching / chunked prefill / CUDA graph 等生产优化）；8B 单卡，70B/106B 用 TP=4；每步提议 3 个草稿 token（n-gram 敏感性分析用 5）。
模型：Llama-3.1-8B-Instruct、Llama-3-70B-Instruct、Qwen3-8B、GLM-4.5-Air-106B，草稿用 Qwen3-0.6B。
数据集：CNN/DailyMail、ShareGPT、InstructCoder、GSM8K、AIME22–24、GPQA-Main。
来源：https://arxiv.org/html/2601.11580v1 ；项目页 https://specdecode-bench.github.io/

| 配置 | bs=1 | bs=128 |
|---|---|---|
| EAGLE，Llama3.1-8B，GSM8K | 1.73× | 1.21× |
| EAGLE，Llama3-70B，ShareGPT | 1.96× | 1.72× |

论文明确陈述：*"Across all workloads, every SD variant outperforms the no-SD baseline."*

> ⚠️ **重要的口径冲突**：该项目**网站**的摘要里出现过"树验证 k=21（分支因子 4）在 bs64 跌破 1× 加速"的说法，但我对**论文正文**做定向抽取时，抽取结果称论文未包含该树验证敏感性分析、且明确说所有变体都优于基线。两次抽取互相矛盾。
> **处理**：以论文正文陈述为准；"k=21 跌破 1×"标记为 **【待核验】**，正文不得引用。

> ※ **2026-08-22 复核：此处记载有误 —— 根本不是矛盾，是我只读了 v1。【待核验】解除，"正文不得引用"的禁令撤销。**
> 上面这段对 **v1 是对的**：v1（2025-12-31 提交，PDF 148 684 字符）里 `Tree-Style` 0 次、`k=21` 0 次、`SGLang` 0 次，`tree` 仅 3 次且全在引言列举与参考文献里 —— **v1 确实不含树敏感性分析**。
> 但 **arXiv:2601.11580 有 v2（2026-03-18 修订）**，本库当时四处引用的都是写死的 `…/html/2601.11580v1`，**从未打开过 v2**。v2 新增了一整节 **"Tree-Style Verification" + Figure 2**，网站摘要说的就是它。**网站是对的；错的是我把 v1 当成了"论文正文"的全部。**
> **v2 逐字原句**：*"By batch size 64, the k=21 tree falls below 1× speedup on all workloads for both models, whereas the chain remains above 1× throughout."*
> **和 "every SD variant outperforms the no-SD baseline" 不矛盾**：那句在**引言**里（v1、v2 都有），说的是 **vLLM 上 chain（k=3）五种变体**的主实验；树实验是**另一套、在 SGLang 上跑的**，两者作用域不同。我当时把一句引言当成了全文范围的断言。
> **v2 树实验的完整口径（铁律二七项）**：引擎 **SGLang v0.5.9（不是 vLLM）**——原文自陈原因是 *"the draft-tree path in the vLLM version we evaluated is not yet sufficiently optimized for a fair comparison"*；模型 Qwen3-8B、Llama3-70B；负载 4 个非推理数据集（GSM8K / CNN-DailyMail / ShareGPT / InstructCoder）；**树深固定 3**；chain 基线 k=3，树 k=6（分支因子 2）与 k=21（分支因子 4，≈EAGLE 的树）；k = 目标模型一次并行验证的草稿 token 数；chain 用 **FlashAttention-3**、树用 **FlashInfer**；SGLang 默认走**动态草稿树**策略；图 2 横轴 batch = 1/16/64/128；测的是**吞吐比**（generated tokens/sec，开 SD vs 不开）；硬件按原文 *"Unless otherwise noted, all other settings follow those described earlier"* 承接 §3.1 的 **H100 80GB**（8B 单卡、70B TP=4）。
> **v2 给出的数字**：bs=1 时 Qwen3-8B/GSM8K **chain 1.65× → k=6 1.68× → k=21 1.85×**；Llama3-70B/ShareGPT **1.81× → 1.90× → 2.03×**。接受长度 Qwen3-8B/GSM8K **2.25 → 2.51 → 2.92**、接受率 **0.415 → 0.300 → 0.095**；Llama3-70B/ShareGPT 接受长度 **2.29 → 2.55 → 2.93**、接受率 **0.429 → 0.310 → 0.097**。
> ⚠️ **原文未指明这组 bs=1 数字属于 EAGLE 还是 EAGLE-3**（图 2 图例两族都有，正文只写 *"tree-based EAGLE/EAGLE-3"*）。引用时必须带这句"未指明"。
> **核验方式**：v1/v2 的 HTML 与 PDF 四份全部 `curl` 落盘，本地 `pdftotext -layout` + 正则比对；项目网站（末次更新 **2026-04-02**）单独落盘核对，其"Tree vs Chain Verification"一节与 v2 正文逐条对得上。

**为什么这篇没穿破 1.0× 而 EAGLE-3 论文穿破了**——这正是"不许跨来源合表"的活教材：
- 硬件不同（H100 vs RTX3090）：3090 的 算力/带宽 比更低，但 vLLM 生产优化程度、kernel 成熟度、批调度都不同；
- 框架版本不同（v0.10.1.1 vs EAGLE-3 论文当时的版本）；
- 草稿方案不同；批大小上限不同（128 vs 56）。
**结论：交叉点位置不可搬运，必须本机实测。**

### 2.5 【一手·论文自报】批量实现本身的开销：Batch Speculative Decoding Done Right

**口径**：A100 80GB；Vicuna-7B/68M、Qwen3-8B/0.6B、GLM-4-9B/0.6B；batch size 1–32。
来源：https://arxiv.org/html/2510.22876v3

核心问题（"ragged tensor problem"）：*"sequences in the same batch accept different numbers of draft tokens, desynchronizing position IDs, attention masks, and KV-cache state."*
论文主张**现有批量投机解码实现普遍破坏输出等价性**（即批量下结果和单条不一致）。

**修正它是要付钱的**：
- 对齐开销：bs=1 时约 13% → **bs=32 时接近 40%**；bs=16 时 46.7%（原文两处数值口径见原文）
- *"Beyond batch size 8, throughput degrades as sequence length diversity reduces grouping effectiveness, forcing more frequent fallbacks to realignment."*
- *"EqSpec exhibits negative scaling beyond BS=8"*

> 这条的意义：**即使算力账算得过来，"把批量投机做对"这件事本身就是一笔随 batch 增长的开销。** 这是很多人算 roofline 时漏掉的第三项。

### 2.6 汇总：交叉点在哪？——**没有统一答案**

| 来源 | 交叉点 / 收益消失点 | 硬件 | 目标模型 | 方法 | 口径完整度 |
|---|---|---|---|---|---|
| EAGLE-3 论文 Table 5 | EAGLE 在 bs 24→32 之间穿过 1.0× | RTX 3090 | Llama-3.1-8B | EAGLE | 缺 γ/τ/温度/数据集 |
| EAGLE-3 论文 Table 5 | EAGLE-3 在 bs 56 打平 | RTX 3090 | Llama-3.1-8B | EAGLE-3 | 同上 |
| Meta 2508.08192 | bs 48 时 0.7× | vLLM（原始 EAGLE 实现） | Llama 系 | EAGLE | 缺具体硬件说明 |
| SqueezeBits | 并发 32（1K 输入）/ 16（2K 输入） | 4×A100 | Llama-3.1-70B | 独立草稿 0.5B | 较完整 |
| Batch SD Done Right | bs 8 以后负向缩放（针对 EqSpec 对齐方案） | A100 | 多个 | 多个 | 较完整 |
| SpecDecode-Bench | 到 bs128 仍 > 1.0×（※ 见下方复核：这只对**链式 k=3** 成立） | H100 | 多个 | 4 种 | 完整 |
| Nightjar（§8.5） | 高 QPS 下最多比 vanilla 慢 30.25% | RTX 4090 | 7B | 独立草稿 0.5B | 较完整 |

**这张表的正确用法**：证明"交叉点存在且分布极广（bs8 ~ bs128+）"，**不是**用来取一个平均值当阈值。

---

## 3. 长上下文维度：MagicDec 的"仍有收益"成立在什么条件下

### 3.1 【一手·论文自报】MagicDec：长上下文 + 大批下投机解码**重新**有收益

核心论断（逐字）：
> *"For every model and hardware pair, there exists a **critical sequence length** beyond which LLM inference becomes memory bound even for large batch sizes."*

机制：KV cache 大小 ∝ batch × seqlen，而**注意力对 KV 的算术强度是常数**（读一遍 KV 只做一次很浅的计算）。当 KV cache 的搬运时间超过权重搬运 + GEMM 时间，前向重新被钉回 memory-bound，于是"闲置算力"又出现了。

**实测**（口径：8×A100 / 8×H100；目标 = LLaMA-2-7B-32K 与 LLaMA-3.1-8B；草稿 = TinyLlama-1.1B 或**自投机 + StreamingLLM（滑窗 + attention sink）**；batch 32–128）：

| 硬件 | 目标模型 | prefill 长度 | batch 32–128 加速比 |
|---|---|---|---|
| 8×A100 | Llama-2-7B | 8K | 1.18× – 1.63× |
| 8×A100 | Llama-2-7B | **32K** | **1.91× – 2.0×** |
| 8×A100 | Llama-3.1-8B | 32K | 1.22× – 1.47× |
| 8×H100 | Llama-2-7B | 8K | 1.19× – 1.65× |
| 8×H100 | Llama-2-7B | 32K | up to 1.64× |
| 8×H100 | Llama-3.1-8B | 32K | 1.23× – 1.43× |

来源：https://infini-ai-lab.github.io/MagicDec/ 、https://infini-ai-lab.github.io/MagicDec-part2/ 、Together AI 博客 https://www.together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference

**口径缺项**：上表未逐格给出 γ、接受率、温度。**【口径不全】**

### 3.2 MagicDec 自己承认的失效条件

论文原文逐字：
> *"For the GQA target model, the speed-up is expected to be lower because of better arithmetic intensity."*

翻译成可执行判据：
1. **GQA / MQA 模型的 critical sequence length 更高**。因为 GQA 把 KV head 数砍掉，KV cache 变小，算术强度变高，需要**更长的序列**才能重新掉回 memory-bound。表中 Llama-3.1-8B（GQA）确实全面低于 Llama-2-7B（MHA）：32K 下 1.22–1.47× vs 1.91–2.0×。
2. **今天的主流模型全是 GQA / MLA**，而 MagicDec 的最亮眼数字来自 MHA 的 Llama-2。**这是引用 MagicDec 时最容易踩的坑。**
3. 短上下文（低于 critical length）下大批仍然是 compute-bound，结论回到 §2。

**H100 上的数字整体低于 A100**，与"算力/带宽比更高的卡，memory-bound 窗口更窄"一致。推广到 B200/GB200 应当更不利——**但我未查到 Blackwell 上的实测，标记未查证（§11）。**

### 3.3 【一手·论文自报】反例：OWL —— EAGLE3 在长上下文上**变慢到 0.81×**

**口径**：目标 = Llama-3.1-8B-Instruct（1×H200）与 Llama-3.3-70B-Instruct（8×H200）；**batch size = 1**；benchmark = LongSpecBench，200 条样本，**输入长度 4K–64K**，取自 WildChat-4.8M。
来源：https://arxiv.org/abs/2510.07535 、https://arxiv.org/html/2510.07535v1
（作者含 Aurick Qiao、Samyam Rajbhandari，即 Snowflake Arctic Inference 团队）

**平均接受长度 τ**：

| 方法 | Llama-3.1-8B | Llama-3.3-70B |
|---|---|---|
| EAGLE3 | **1.28** | **1.35** |
| OWL | 4.00 | 4.27 |
| HOWL | 6.14 | 5.31 |

**加速比**：

| 方法 | 加速比 |
|---|---|
| EAGLE3 | **0.81×** |
| OWL | 2.35× |
| HOWL | 3.08× |

**归因**：EAGLE3 的公开草稿头**训练窗口是 2K**，而现有 benchmark（SpecBench 等）输入普遍在 2K 以内。一旦输入超出训练窗口，草稿头外推失败，τ 塌到 ~1.28（几乎等于不投机），此时**草稿开销 + 验证开销全是净亏**，于是 0.81×。论文归因逐字为 *"small acceptance length and drafting overhead"*。

**旁证**：Red Hat 的 EAGLE-3 实践文也提到，当时可得的投机模型有 *"a 2048 context length cap"*。
来源：https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding

### 3.4 调和：MagicDec 与 OWL 不矛盾——它们是两个独立的乘子

这是本节最重要的结论，正文必须讲清楚：

> **投机解码收益 ≈ (硬件侧：前向是否 memory-bound) × (算法侧：草稿在这段分布上准不准) − 开销**

- **MagicDec 谈的是第一个乘子**：长上下文 + 大批让 KV 搬运主导，硬件侧重新出现闲置算力。它用的草稿是 **StreamingLLM 定长窗口 / 自投机**——这类草稿**不存在训练窗口外推问题**，因为它本来就只看固定窗口。
- **OWL 谈的是第二个乘子**：EAGLE3 这类**学习型草稿头**在训练窗口外失效。硬件侧就算给了闲置算力，草稿侧交不出货。
- **SqueezeBits（§2.3）观察到 1K→2K 让交叉点从 32 降到 16**，看似又和 MagicDec 反向。原因：2K **远未到 critical sequence length**，这个区间里长输入只是增加了 KV 读取和调度压力、并压低了接受率（53.5%→50.9%），还没到"KV 主导"的拐点。

**可执行判据**：
1. 长上下文能救投机解码的前提是 **KV cache 搬运真正主导前向时间**——粗略地说要 `batch × seqlen` 大到 KV 字节数远超权重字节数。几 K 的输入不算长上下文。
2. 长上下文场景下，**必须换草稿方案**：用自投机 / 定长窗口 / 检索式草稿，或用在长序列上训过的草稿头（OWL 的 LSTM drafter）。**直接把 2K 训练的 EAGLE3 拿到 32K 上用，是本文找到的最确定的负收益配方。**
3. GQA/MLA 模型请把预期砍半（§3.2）。

### 3.5 【一手·论文自报】TriForce：另一类长上下文方案（分层投机）

**口径**：来源 https://arxiv.org/abs/2404.11912 、https://infini-ai-lab.github.io/TriForce/
- Llama2-7B-128K，**A100**：up to **2.31×**
- Llama2-13B-128K，**2×RTX 4090（offloading）**：TBT 低至 **0.22 s**，称比高度优化的 offloading 系统快 7.8×
- Llama2-7B-128K，2×RTX 4090：TBT **0.11 s**
- 单张 RTX 4090：比 DeepSpeed-Zero-Inference 快 4.86×
- 动机数字：Llama2-7B-128K 的 KV cache 达 **64 GB**，而权重仅 14 GB

**口径缺项**：上述数字未统一给出 batch size（多数为小批/单请求 offloading 场景）、接受率、温度。**【口径不全】** 这是"用投机解码换 offloading 带宽"的场景，与服务化大批场景不可比。

---

## 4. 接受率维度：不同任务类型差多少

### 4.1 【一手·论文自报】EAGLE-3 论文 Table 1：逐数据集的 τ 与加速比（严格分开）

**口径**：论文自报；温度分 T=0 / T=1 两组；**batch size 1**（该表遵循投机解码论文惯例）；原文未在该表给出硬件逐行标注（EAGLE 系列主实验通常为单卡 A100/3090，**以原文为准**）。**【口径不全：硬件与 γ/树规模未逐行给出】**
来源：https://arxiv.org/html/2503.01840

**T = 0（贪心）**

| 目标模型 | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM |
|---|---|---|---|---|---|
| **Vicuna 13B** 加速比 / τ | 5.58× / 6.65 | 6.47× / 7.54 | 5.32× / 6.29 | 5.16× / 6.17 | 5.01× / 6.47 |
| **Llama-3.1-Inst 8B** 加速比 / τ | 4.40× / 6.13 | **4.85× / 6.74** | 4.48× / 6.23 | 4.82× / 6.70 | **3.65× / 5.34** |
| **Llama-3.3-Inst 70B** 加速比 / τ | 4.11× / 5.63 | **4.79× / 6.52** | 4.34× / 6.15 | 4.30× / 6.09 | **3.27× / 5.02** |
| **DeepSeek-R1-Distill-Llama 8B** 加速比 / τ | 4.05× / 5.58 | 4.59× / 6.38 | **5.01× / 6.93** | 3.65× / 5.37 | **3.52× / 4.92** |

**T = 1**

| 目标模型 | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM |
|---|---|---|---|---|---|
| Vicuna 13B | 4.57× / 5.42 | 5.15× / 6.22 | 4.71× / 5.58 | 4.49× / 5.39 | 4.33× / 5.72 |
| Llama-3.1-Inst 8B | 3.07× / 4.24 | 4.13× / 5.82 | 3.32× / 4.59 | 3.90× / 5.56 | 2.99× / 4.39 |
| Llama-3.3-Inst 70B | 3.96× / 5.45 | 4.36× / 6.16 | 4.17× / 5.95 | 4.14× / 5.87 | 3.11× / 4.88 |
| DeepSeek-R1-Distill-Llama 8B | 3.20× / 4.49 | 3.77× / 5.28 | 4.38× / 6.10 | 3.16× / 4.30 | 3.08× / 4.27 |

**读数**：
- **代码（HumanEval）几乎总是最高，摘要（CNN/DM）几乎总是最低。** Llama-3.1-8B 上二者相差 **1.33×**（4.85 vs 3.65），τ 相差 1.40 个 token。
- 70B 上摘要与代码差距更大（3.27 vs 4.79，1.47×）。
- 这是**同一篇论文、同一套口径**内的比较，**可以**横比。跨论文不可。

### 4.2 【一手·论文自报】EAGLE-1 的同类数据（注意：与 EAGLE-3 不可混）

**口径**：Vicuna 7B 用 RTX 3090；LLaMA2-Chat 70B 用 4×A100 40G；MT-Bench；T=0。
来源：https://arxiv.org/html/2401.15077

| 指标 | Vicuna 7B | LLaMA2-Chat 70B |
|---|---|---|
| MT-bench τ（T=0） | **3.94** | 3.81 |
| MT-bench τ（T=1） | 3.17 | 3.46 |
| MT-bench 加速比（T=0） | 约 **2.90×** | 约 3.01× |

**批大小消融（Table 7，MT-Bench，T=0）**：

| batch size | Vicuna 7B | LLaMA2-Chat 70B |
|---|---|---|
| 1 | 2.90× | 3.01× |
| 2 | 2.87× | 2.81× |
| 3 | 2.65× | 2.50× |
| 4 | 2.76× | 2.40× |

原文：*"The speedup ratio diminishes with increasing batch size."*
> 注意这里只扫到 bs4，看不出交叉点；要看交叉点必须用 §2.1 的 EAGLE-3 Table 5。

**EAGLE 草稿头可训练参数量**（用于 §6 的开销讨论）：7B→0.24B，13B→0.37B，33B→0.56B，70B→**0.99B**，Mixtral 8x7B→0.28B。

### 4.3 【一手·第三方】SpecDecode-Bench：任务对方法选择的影响远大于方法本身

**口径**：H100；vLLM v0.10.1.1；每步 3 个草稿 token；单位 = 平均接受 token 数（含 std）。
来源：https://arxiv.org/html/2601.11580v1

| 数据集 | n-gram | EAGLE / EAGLE-3 |
|---|---|---|
| **InstructCoder**（代码编辑） | **7.27**（std 4.19） | EAGLE 4.24（std 1.07） |
| **CNN/DailyMail**（摘要） | 2.33（std 1.08） | EAGLE-3 3.08（std 0.61） |
| **GSM8K**（数学） | **1.41**（std 0.45） | EAGLE-3 3.02（std 0.47） |

**三个可执行结论**：
1. **n-gram 的方差极大**：代码编辑 7.27，数学 1.41，相差 **5.2 倍**。n-gram 是"看菜下饭"的方法，任务分布一变就废。
2. 论文给出切换判据：**当 prompt-output 重叠的 BLEU-4 超过约 0.6 时，n-gram 稳定优于 EAGLE / EAGLE-3。** 这是本文找到的**唯一一条可直接计算的方法选择判据**，正文务必收录。
3. EAGLE 系的 std 明显更小（0.47–1.07 vs 0.45–4.19），即**学习型草稿头的收益更可预测**——对有 SLO 的服务，可预测性本身值钱。

其它逐条口径数字（同源）：
- 执行时间分解：*"the verification stage generally takes the largest execution time, ranging from 42% to 95% in all execution methods."*
- 草稿开销：n-gram *"<2%"*；draft-model 方法在 Qwen3-8B 上 **bs=1 时 47%**，**bs=512 时降到 16%**。
- 显存：n-gram 0；EAGLE/EAGLE-3 *"adds <10%"*；draft-model（Qwen3-0.6B 配 8B）使 per-token 显存 **×1.77**。
- Oracle 上界：InstructCoder / bs1 / n-gram，oracle **2.75×** vs 最优定长 **约 2.1×**；自适应组合 oracle 达 **4.9×**，单方法 oracle 上界 2.2×。原文：*"the gap generally widens as batch size increases: the fixed-k curves drop much faster, while the oracle speedup degrades more gently."*
  → **定长 γ 在大批下损失的收益，比自适应 γ 大得多。这是 §8 自适应系统存在的理由。**

### 4.4 【一手·论文自报】多语言：最确定的负收益场景

**口径**：verifier = Qwen 3.5 9B；draft = Qwen 3.5 0.8B（另测 2B / 4B）；任务 = 英译 11 种语言；草稿前向 ~0.033 s，n-gram ~0.001 s。
语言：Amharic, Berber, Cherokee, Guarani, Hawaiian, Igbo, Nepali, Occitan, Quechua, Yoruba, Tamazight。
来源：https://arxiv.org/html/2605.30580v1 （*Speculative Decoding Across Languages*）

| 场景 | 平均接受率 ᾱ | 平均加速 f̄ |
|---|---|---|
| 翻译（域内），通用草稿 | **0.40**（值域 0.30 ber – 0.54 amh） | **1.02×**（≈ 完全白干） |
| 翻译（域内），任务特定蒸馏草稿 | 0.60 | 1.28× |
| 故事生成（域外），任务特定蒸馏草稿 | 0.43 | **1.03×**（域外泛化崩） |
| 故事生成（域外），n-gram | 0.30（更低！） | **1.39×**（却更快） |

**三个反直觉结论**：
1. **低资源语言上通用草稿模型的加速比是 1.02×——即完全不值得部署。**
2. **接受率高 ≠ 更快**：故事生成上蒸馏草稿 ᾱ=0.43 只有 1.03×，而 n-gram ᾱ=0.30 却有 1.39×。差别全在草稿前向成本（0.033 s vs 0.001 s，**33 倍**）。这是 §6 的核心论点的最干净证据。
3. 论文发现接受率与"模型该语言能力"相关性**很弱**（r = 0.170），即**不能用"模型这个语言好不好"来预测投机收益**，必须实测。

### 4.5 【一手·第三方】Red Hat：翻译任务上最优草稿长度是 1 甚至 0

**口径**：Llama 3.1 8B 单卡 A100；Llama 3.3 70B 四卡 A100；生成上限 1024 token；数据 = MT-Bench + SpecBench（数学、RAG、翻译、HumanEval）。
来源：https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding

- 8B：延迟最多降低 **1.8×**
- 70B：低请求率下最多降低 **1.6×**
- RAG / 数学：最多 **2.1×**
- **翻译（德译英）**：原文 *"Eagle performs so poorly that the optimal draft length is 1, or even 0"* —— **即最优策略是关闭投机**
- 高请求率下 70B 因 compute saturation 延迟上升

**口径缺项**：该文未给出逐数据集的接受率数值与 QPS 具体取值。**【口径不全】**

### 4.6 【一手·论文自报】推理 / 长思维链场景：高 τ 也能是负收益

**口径**：单张 RTX A6000 48GB；16-core Xeon Silver 4309Y；float16；**batch size 1**；目标 = DeepSeek-R1-Distill-Llama-8B（DSL-8B）与 Qwen3-8B（QW3-8B）；9 种方法。
来源：https://arxiv.org/html/2509.04474v1 （*Scaling Up, Speeding Up*）

| 方法 / 场景 | 平均接受 token（MAT） | 加速比 |
|---|---|---|
| **模型型 SpS** | **7.07（全表最高）** | **0.87×（比不开还慢）** |
| SAM[EAGLE-3]，多轮思考，T=0，DSL-8B | 4.72 | 3.97× |
| EAGLE-3，多轮思考，T=0，QW3-8B | 4.38 | 2.91× |
| SAM[EAGLE-3]，Best-of-N，T=0.6，QW3-8B | 3.91 | 2.77× |

**这一行是全文最有力的单条证据**：
> **SpS 的接受长度是全表最高的 7.07，加速比却是 0.87× —— 比不开投机还慢 13%。**
> 原文归因：*"draft generation … overhead … limits overall acceleration."*
> **结论：接受率/接受长度是过程指标，不是收益指标。只看 α 或 τ 做决策必然踩坑。**

同源还给出：n-gram 类方法在 T 从 0 升到 0.6 时加速比下降 **22%**（见 §5.3）；REST 表现差，作者归因于与推理数据集重叠度低。

---

## 5. 温度与采样参数

### 5.1 【一手·论文自报】温度 0 → 1 的代价：加速比掉约 30%

直接用 §4.1 的 EAGLE-3 Table 1 做差（**同论文同口径，可以横比**）：

| 目标模型 / 数据集 | T=0 加速比 | T=1 加速比 | 相对跌幅 | T=0 τ | T=1 τ | τ 跌幅 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B / MT-bench | 4.40× | 3.07× | **−30.2%** | 6.13 | 4.24 | −30.8% |
| Llama-3.1-8B / GSM8K | 4.48× | 3.32× | −25.9% | 6.23 | 4.59 | −26.3% |
| Llama-3.1-8B / CNN-DM | 3.65× | 2.99× | −18.1% | 5.34 | 4.39 | −17.8% |
| Llama-3.1-8B / HumanEval | 4.85× | 4.13× | −14.8% | 6.74 | 5.82 | −13.6% |
| Llama-3.3-70B / MT-bench | 4.11× | 3.96× | −3.6% | 5.63 | 5.45 | −3.2% |
| Vicuna 13B / MT-bench | 5.58× | 4.57× | −18.1% | 6.65 | 5.42 | −18.5% |

**读数**：
1. **温度惩罚不是常数**，从 −3.6%（70B / MT-bench）到 −30%（8B / MT-bench）都有。
2. **加速比跌幅与 τ 跌幅高度一致**（每行两者相差通常 <1 个百分点）→ 温度是通过压低接受率起作用的，而不是通过增加开销。
3. **代码任务对温度最不敏感**（−13.6% ~ −14.8%），**MT-bench 类开放对话最敏感**。直觉解释：代码的下一 token 分布本来就尖，升温也还是尖。

EAGLE-1 同向（§4.2）：Vicuna 7B MT-bench τ 3.94（T=0）→ 3.17（T=1），跌 19.5%。

### 5.2 机制：为什么升温一定伤接受率

标准投机采样的接受概率是 `min(1, p_target(x)/q_draft(x))`，整体接受率约等于 1 − TV(p, q)（总变差距离）。
- T→0 时两个分布都塌成 one-hot，只要 argmax 一致就接受 → α 高。
- T→1 时分布展开，草稿模型和目标模型在**尾部**的分歧被放大，TV 距离变大 → α 降。
> 注意：常见的直觉"高温让更多草稿都'说得通'所以更容易接受"是**错的**——投机采样验的不是"合理性"，是**分布一致性**。实测（§5.1）一致显示升温降低接受率。

**top-p / top-k 的正确处理**：截断必须**对 draft 和 target 施加同样的截断后**再算接受比，否则会破坏分布保真。这一点在实现层面是常见 bug 来源。**我未查到公开的 α-vs-top-p 实测曲线，标记未查证（§11）。**

### 5.3 【一手·论文自报】n-gram 类方法的温度敏感性更高

来源：https://arxiv.org/html/2509.04474v1
> n-gram 类方法在 T 从 0 升到 0.6 时，加速比下降 **22%**，原文描述为 *"sensitivity to sampling temperature"*。

**口径缺项**：原文未逐方法给出对应的 α 变化。**【口径不全】**

### 5.4 【一手·框架文档】vLLM 的温度相关限制

vLLM 文档逐字：
> *"use_heterogeneous_vocab currently supports greedy draft sampling only. Probabilistic acceptance (temperature > 0 draft sampling) is not yet supported"*

来源：https://docs.vllm.ai/en/latest/features/speculative_decoding/

即**异构词表（draft 与 target 词表不同）+ 温度采样，当前不支持**。要用跨词表草稿就得接受贪心草稿采样。

### 5.5 输出等价性：一个容易被忽略的风险

- **Batch Speculative Decoding Done Right**（§2.5）主张：现有批量投机解码实现因 ragged tensor 问题**普遍破坏输出等价性**，即批量下的输出与逐条推理不一致。修正方案带来 13%→40% 的对齐开销。
  来源：https://arxiv.org/html/2510.22876v3
- 这意味着"投机解码是无损的"这个常见说法，**在批量实现层面并非自动成立**——它是理论性质，不是实现保证。
- **我未能找到一个权威的、逐框架的"greedy 下投机与非投机输出逐 token 一致性"测试报告，标记未查证（§11）。**

---

## 6. 草稿开销：为什么业界从"独立小模型"转向"单层草稿头"

### 6.1 草稿开销到底占多少

**【一手·第三方】SpecDecode-Bench**（H100 / vLLM v0.10.1.1 / 每步 3 草稿 token）：

| 草稿方案 | 草稿阶段占执行时间 | GPU 显存开销 |
|---|---|---|
| n-gram | **< 2%** | **0** |
| EAGLE / EAGLE-3（单层草稿头） | 原文未单列百分比 | 权重 + KV cache **< 10%** |
| 独立 draft model（Qwen3-0.6B 配 Qwen3-8B） | **bs=1 时 47%**，bs=512 时 16% | per-token 显存 **×1.77** |

验证阶段占 **42%–95%**。
来源：https://arxiv.org/html/2601.11580v1

**这张表就是答案**：独立小模型在**低批**（也就是投机解码最该发光的区间）把将近**一半的时间**花在草稿上，还把每 token 显存开销抬到 1.77 倍——直接压缩可容纳的并发数，从另一个方向伤吞吐。

### 6.2 【一手·论文自报】更大的草稿模型：接受率更高，吞吐更低

*Decoding Speculative Decoding*（NAACL 2025）
**口径**：4×NVIDIA A100 80GB；**batch size 固定为 1**；贪心解码；目标 = OPT-66B / LLaMa-65B；草稿候选 = OPT-125M / 350M / 1.3B / 2.7B / 6.7B、LLaMa-7B / 13B。
来源：https://arxiv.org/html/2402.01528v3

核心结论（逐字）：
> *"The key bottleneck in speculative decoding is the draft model's latency."*

- 增大草稿模型**持续提升接受率，却持续降低吞吐**：OPT-350M / 1.3B / 6.7B 的端到端吞吐**都不如 OPT-125M**。
- 模型**深度**线性决定延迟 → **浅而宽优于深而窄**（同参数量下）。
- 另一条：*"a draft model with higher accuracy on language modeling task can have similar TAR to a model with lower accuracy"* —— 语言建模能力强不代表接受率高。

> ⚠️ **重要口径修正**：这篇论文**全程 batch size = 1**，**不能**用来论证批大小效应。我在调研早期一度以为它谈了批大小，核对后确认没有。引用时务必注意。

### 6.3 §4.4 的多语言实验给出了同一结论的独立复现

故事生成任务上：蒸馏草稿 ᾱ=0.43 → 1.03×；n-gram ᾱ=0.30 → **1.39×**。
草稿前向：0.033 s vs 0.001 s（**33×**）。
→ **接受率低 43% 的方法反而快 35%，唯一原因是草稿便宜。**
来源：https://arxiv.org/html/2605.30580v1

### 6.4 转向单层草稿头的四条理由（每条带证据）

| 理由 | 证据 |
|---|---|
| **1. 草稿延迟是主要瓶颈，草稿头比小模型快一个量级** | §6.2；§6.3（33× 前向成本差） |
| **2. 独立草稿模型要自己的 KV cache，吃掉批容量** | §6.1（per-token 显存 ×1.77 vs 草稿头 <10%） |
| **3. 草稿头复用目标模型的隐藏状态，接受率反而更高** | EAGLE 系在 §4.1 达到 τ=5.0–7.5，远高于典型独立草稿（§2.3 的 0.5B 草稿 α≈53%） |
| **4. 部署简化：单权重、单词表、无需两套调度** | vLLM 文档：*"Speculative decoding with draft models is not supported in `vllm<=0.10.0`"*（即 V1 引擎早期只支持草稿头/n-gram 类）<br>来源：https://docs.vllm.ai/en/latest/features/speculative_decoding/ |

**训练成本佐证**：EAGLE 草稿头可训练参数量 7B→0.24B ... 70B→0.99B；原文称 70B 的草稿头 *"trainable within 1-2 days on an A100 40G server"*。
来源：https://arxiv.org/html/2401.15077

**工程优化空间佐证**（Snowflake Arctic Inference）：
- Proposer 延迟从 **1.47 ms/token 降到 0.47 ms/token（约 3.1×）**
- Verifier 延迟从 **1.34 ms 降到 0.38 ms（约 3.5×）**
- 接受率：MLP-Speculator 原版 **13.7%** → Arctic 版 **42.7%**；LSTM-Speculator **44.5%**
- Suffix decoding：**CPU 上 20 微秒/草稿 token，零 GPU 开销**
来源：https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/

> 注意：Arctic 的这些吞吐数字是在 **max_concurrency = 1** 下测的（见 §8.4），且其插件自带 `disable_by_batch_size: 64`。**【口径不全，不可当作服务化收益】**

### 6.5 【待核验】DeepSeek-V3 MTP 的"85–90% 接受率 / 1.8× TPS"

这两个数字在中文/英文二手资料中被极其广泛引用，归因于 DeepSeek-V3 技术报告（通常指 §5.4.3 Multi-Token Prediction 相关讨论）：
- 第二个 token 预测的接受率约 **85%–90%**
- 带来约 **1.8 倍 TPS**

**我的核验结果**：对 arXiv HTML 版本（v1、v2、latest）以及 ar5iv 镜像做了 **4 次定向抽取**，均未能逐字定位到这两句；其中一次抽取明确报告文档在到达该章节前被**截断**（*"Content truncated due to length"*）。
来源尝试：https://arxiv.org/html/2412.19437v1 、https://arxiv.org/html/2412.19437v2 、https://arxiv.org/html/2412.19437 、https://ar5iv.labs.arxiv.org/html/2412.19437

**结论**：极可能确实存在于原文（截断导致抽取失败的可能性最大），但**本库标记为【待核验】**。写正文时若要引用，**必须人工打开 PDF 核对章节号与原句**，不得直接从本文搬运。
（该报告确定包含的一句是：*"we can also repurpose these MTP modules for speculative decoding to further improve the generation latency"*，§2.2。）

> ※ **2026-08-22 复核：此处的【待核验】已解除 —— 数字是真的，原判断"极可能确实存在于原文"正确。**
> **核验方式**（不走摘要模型，避免二次幻觉）：`curl` 把 arXiv HTML **v1**（549 443 B）、**v2**（549 352 B）与**官方 PDF**（1 887 366 B / 53 页）全部落盘，本地去标签 + `pdftotext -layout` 后正则定位。**三种渲染逐字一致。**
> **原文位置**：**§5.4.3 Multi-Token Prediction Evaluation**（PDF **第 35 页**，紧接 §5.4.2 Self-Rewarding 之后、§6 Conclusion 之前）。前四次抽取失败的原因确认是**抽取工具在到达 §5.4.3 之前截断**，不是原文没有。
> **逐字原句**：*"Based on our evaluation, the acceptance rate of the second token prediction ranges between 85% and 90% across various generation topics, demonstrating consistent reliability. This high acceptance rate enables DeepSeek-V3 to achieve a significantly improved decoding speed, delivering 1.8 times TPS (Tokens Per Second)."*
> **但按铁律二，口径仍然严重不全，不可横向比较**：batch size **原文未给出**；推理硬件**原文未在该节给出**（§3.4 说线上部署在 H800 集群、解码最小单元 40 节点 320 GPU、TP4+SP+DP80+EP320，**但论文没有说 §5.4.3 的 1.8× 是在该配置下测的 —— 不许替它连线**）；TPS **未区分 per-user 输出速度还是系统总吞吐**；数据集只写 *"across various generation topics"*；**接受率的定义口径未给**（是逐位接受率还是第二 token 的 argmax 一致率，无从判断）。γ 隐含 =1（*"predicts the next 2 tokens"*，$D=1$）。

---

## 7. 与其它优化的冲突

总原则：**投机解码的收益来自"闲置算力"。任何别的优化只要也在吃闲置算力，或者改变了 memory-bound / compute-bound 的平衡，就会和投机解码互相抵消。**

### 7.1 Chunked Prefill

- vLLM Issue #5016（2024-05-23，由 cadedaniel 提出，**closed / stale**，无实现）：请求将 chunked prefill 与投机解码结合。
  机制陈述：vLLM 默认吞吐优先调度 *"prioritizes prefills eagerly"*，而 chunked prefill 把 prefill 工作 *"spread out … over many different decode batches"*。
  来源：https://github.com/vllm-project/vllm/issues/5016
- vLLM Issue #10276：**投机解码 + spec worker 上的 TP + chunked prefill 三者同开会启动后立即失败**。
  来源：https://github.com/vllm-project/vllm/issues/10276
- 现状：不同方法支持度不同（n-gram 类 GPU 投机在较新版本已可与 chunked prefill 共存；draft-model 路径长期受限）。**具体到某个版本的支持矩阵，请查该版本 release notes——本文不给结论，标记未查证（§11）。**

**冲突的本质（这是正文要讲的）**：chunked prefill 的目的就是**把 prefill 的算力填进 decode batch 的闲置算力里**。而投机解码的收益也来自**同一份闲置算力**。二者在物理上抢同一个资源——**开了 chunked prefill，投机解码的收益会被系统性削薄**，即使它们在工程上兼容。

### 7.2 Prefix Caching / APC

- **可以同开**，但有观测坑：vLLM 论坛答复称，*"Prefix caching and speculative decoding can be enabled together in vLLM, but in some versions (including 0.9.1), prefix cache hit statistics may not be logged when speculative decoding is active."*
  来源：https://discuss.vllm.ai/t/can-speculative-decoding-and-prefix-caching-take-effect-simultaneously/1291/4
  → **即：你以为 prefix cache 没生效，其实只是没打日志。别据此做决策。**

- **真实的负收益案例（强证据）**：vllm-ascend Issue #9247，DeepSeek-V4-Flash + MTP：
  | 配置 | prefix cache 命中 | 46,240 token 的 prompt 需重算 |
  |---|---|---|
  | 不开 MTP | 32,768 token | 13,472 token |
  | 开 MTP | **16,384 token** | **29,856 token** |

  多付出约 **16,384 token 的重算**。成因：DeepSeek-V4 的 `HybridKVCacheCoordinator` 的 LCM 对齐块 + MTP/EAGLE 丢弃最后一个匹配块 + 后续 LCM 向下取整，三者叠加，报告者原话：*"a small MTP/EAGLE recomputation requirement is amplified into losing a full 16K prefix-cache segment."*
  报告结论：*"the extra prefill cost can outweigh MTP's decode speed benefit"*。状态：closed as not planned / stale。
  来源：https://github.com/vllm-project/vllm-ascend/issues/9247

  → **这是"开投机解码反而变慢"的一个非常干净的机制性案例：它不是在 decode 上亏的，是在 prefill 上亏的。** 长上下文 + 高前缀复用的场景（多轮对话、agent）必须专门验证这一项。

- 结构性难点：EAGLE 的第 i 个 KV cache 与第 i+1 个 token id 耦合，因此与前缀缓存的对齐需要特殊处理。

### 7.3 CUDA Graph

投机解码使**每步 token 数可变**（接受 0..γ 个），而 CUDA Graph 要求静态形状。

- vLLM PR #23679 逐字：*"The cudagraph ability of the EAGLE(3) model in V1 has been broken for a while since the launch of [PR #20059], because for now vllm torch.compile would never implicitly run a cudagraph until we explicitly dispatch it via the cudagraph dispatcher."*
  即 **EAGLE(3) 的 CUDA Graph 能力在 vLLM V1 中曾长期处于失效状态**。
  该 PR 的做法：主模型与 drafter **各自一个 dispatcher**（主模型 uniform decode query len = `1 + num_spec_tokens`，drafter 通常 = 1）；开投机时把 capture size 约束为能被 `1 + num_spec_tokens` **整除**。
  来源：https://github.com/vllm-project/vllm/pull/23679

- vLLM Issue #21984（Padded Speculative Decoding）逐字：*"FlashMLA currently assumes decode operations have `query_len=1`, forcing multi-token speculation requests to use the inefficient prefill path."*
  即 **多 token 投机在 FlashMLA 下被迫走 prefill kernel 路径**（该 issue 标注其为 *"Memory inefficient!; Compute Optimized"*）。
  padding 方案让所有 batch 保持 `len=4`，代价是为被拒 token 付 padding 计算。
  来源：https://github.com/vllm-project/vllm/issues/21984

**可执行判据**：开投机前确认 (a) 你的 attention backend 支持 `query_len > 1` 的 decode 路径；(b) CUDA Graph 在开投机后仍然生效（看是否回退 eager）。**这两条任一不满足，投机解码的账基本算不过来。**

### 7.4 量化（本节是 §1.1 R3 的依据）

*Speculative Decoding Meets Quantization*
**口径**：量化方案 = W8A8 (SmoothQuant) / W4A16 (GPTQ) / W4A8 (QoQ/QQQ) / FP16 基线；投机方法 = **EAGLE-2**（另用 EAGLE-3 验证）；模型 = Llama-3-8B-Instruct、Llama-3-70B-Instruct；硬件 = A100 80GB、RTX 3090；**单批**。
来源：https://arxiv.org/html/2505.22179v1

- 核心结论逐字：*"applying EAGLE-2 on 4-bit weight quantized models (W4A16 and W4A8) yields limited additional speedups"*
- **最强的负结果**：RTX 3090 上 **W4A16 的 8B 模型 + EAGLE-2 → 无加速增益**。
- 量化定量：60 个草稿 token 时，**W4A16 的 verification ratio 达 1.8，而 FP16 / W8A8 为 1.2**（理想值 1.0）。
  即 **4-bit 权重量化下，验证 60 个草稿 token 要花 1.8 倍于单 token 前向的时间**——"验证近似免费"这个前提直接破产。
- 归因逐字：*"the heavy computation time required during draft verification undermines the memory efficiency gained by 4-bit weight quantization."*
- 另有 profiling 观察：量化模型触发**更多 CUDA kernel launch**，并引入 blockwise dequantization 等额外步骤。

**机制**（写正文时讲这个）：
> 4-bit 权重量化把"权重加载时间"砍掉约 4 倍，等于**把 roofline 拐点往左搬**——原本 memory-bound 的 decode 被量化推向 compute-bound。**量化和投机解码抢的是同一块收益（decode 的空闲算力），二者高度重叠而非叠加。**

**未查证**：FP8（W8A8-FP8）下的同类实测数字，我未找到。§11。

### 7.5 Prefill / Decode 分离（PD 分离）

**定性**：投机解码只作用于 decode 侧。verify 步的 `query_len = 1 + γ`，算术强度高于普通 decode，理论上更适合 decode pool 的带宽受限特性。

**但**：我**未查到**任何 PD 分离系统（NVIDIA Dynamo / Mooncake / DistServe / vLLM disagg）给出的**定量**投机解码收益数据，也未查到"decode 节点开投机后如何影响 KV 传输和 decode 节点批容量"的实测。
**标记未查证（§11）。正文不许在这一节给数字。**

（可用的相关观察：§7.2 的 prefix cache 案例说明投机解码会**反向影响 prefill 侧成本**，这在 PD 分离下意味着两个 pool 的耦合比看起来更强。）

### 7.6 MoE：本节结论是**分裂的**，必须原样呈现分歧

这是本次调研中**分歧最大**的一个问题。四个来源给出四种口径下的不同结论：

**(A)【一手·论文自报】Meta：大 MoE 的加速比随 batch 下降，dense 8B 反而上升**
逐字：*"the Llama3.1 8B model exhibits greater speculative decoding speedup at large batch sizes compared to small batch sizes. In contrast, the speed-up for Llama4 Maverick, which has approximately 400 billion parameters, decreases with increasing batch size."*
来源：https://arxiv.org/html/2508.08192

**(B)【一手·论文自报】EcoSpec：超大 MoE 上即使 batch=1，收益也只有 1.10–1.22×**
口径：8×NVIDIA H200；**batch size 1**；T=0（附录另测 BS=2,4,8）。

| 目标模型 | 基线方法 | 加速比 | 平均激活专家数 |
|---|---|---|---|
| Qwen3-235B-A22B（Top-8） | EAGLE-3 | **1.22×** | 23.7 |
| GPT-OSS-120B（Top-4） | EAGLE-3 | **1.14×** | 11.6 |
| DeepSeek-V3.1（671B，Top-8） | MTP | **1.10×** | 31.4 |

对照：**dense 的 Llama-3.1-8B 上 EAGLE-3 是 4.40×（§4.1）。**
机制：验证成本取决于所有被验 token 激活专家的**并集** `⋃ₜ S_ℓ(xₜ)`；高置信草稿可能路由到**互不相交**的专家集（"expert scattering"），扩大每步专家足迹。
定量：DeepSeek-V3.1 FP8 下**单个专家 = 44.04 MB HBM 带宽**；每层期望专家足迹每减少 0.2 个，每投机步就省约 0.5 GB 专家权重流量。
逐字：*"Verification latency scales linearly with the number of active experts"*
来源：https://arxiv.org/html/2607.12696v1

**(C)【一手·论文自报】MoESD：中等 batch 下 MoE 反而比 dense 更受益**
机制：当 batch 大到**单个 decode step 已经激活了全部专家**时，再验证多个草稿 token **不会**带来额外的专家权重加载。此时验证是真免费。
但在**大 batch** 下退化为 compute-bound，收益消失或转负。
来源：https://arxiv.org/pdf/2505.19645
**【口径不全】** 我通过 PDF 抽取得到的具体数字（模型/加速比区间）可靠性不足，**本文不收录其数字**，只收录机制论断。

**(D)【一手·第三方】Cohere：MoE 因算术强度低而更受益，峰值在中等 batch**
口径：dense 基线 = Command A（111B）；MoE = Cohere MoE，K=3（128 选 8 的表述见原文）。
- MoE 的加速比-batch 曲线是**非单调**的，dense 是**单调下降**的
- BS=1：**1.95×**
- 验证成本比 `Tt(4)/Tt(1) ≈ 1.25×`，配合 AL=2.73 → **2.18×**
- 专家路由的时序相关性使唯一专家数减少 **20–31%**；*"verifying four tokens activates ~2.5× top-k"*
来源：https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding

**怎么调和这四条（正文要写的）**：

它们并不真矛盾，差异全在**三个未对齐的口径**：
1. **MoE 的规模与稀疏度**：GPT-OSS-120B（Top-4，5.1B 激活）和 DeepSeek-V3.1（671B，Top-8，37B 激活）完全不是一回事。EcoSpec 测的是超大稀疏模型，Cohere 测的是自家中等规模模型。
2. **所处的 batch 区间**：MoESD/Cohere 说的"MoE 更受益"发生在 **"全部专家已被激活但尚未 compute-bound"** 的中间窗口；EcoSpec 的 bs=1 在窗口左侧（专家足迹随草稿数线性涨）；Meta 的大 batch 在窗口右侧。
3. **对比基准**：与 dense 比，还是与自己不开投机比。

**可执行判据（这是本节唯一该进正文的结论）**：
> MoE 上投机解码的关键变量是 **"验证 γ+1 个 token 时激活的专家并集 / 验证 1 个 token 时激活的专家数"**。
> - 这个比值 ≈ 1（batch 已大到全专家激活）→ MoE 比 dense 更受益。
> - 这个比值接近 γ+1（bs 小、Top-k 小、路由分散）→ MoE 比 dense 差得多。
> **实测方法**：直接 profile 每步激活的唯一专家数，开投机前后各测一次。不要用别人的 MoE 结论。

**第三方旁证**（消费级硬件）：dev.to 的家用测试观察到 MoE 上 *"each speculative token may activate different experts"* 导致批处理效率下降（§9.6）。

### 7.7 张量并行

- vLLM 提供 `draft_tensor_parallel_size` 独立配置草稿模型的 TP 度。
  来源：https://docs.vllm.ai/en/latest/features/speculative_decoding/
  动机：小草稿模型在高 TP 下会变成通信受限（每层 all-reduce 的固定开销相对于极小的计算量占比过高）。
- vLLM 文档另有明确不兼容项：*"Pipeline parallelism is not composable with speculative decoding as of `vllm<=0.15.0`"*
- vLLM Issue #10276：投机解码 + spec worker TP + chunked prefill 三者同开会启动即失败。
  来源：https://github.com/vllm-project/vllm/issues/10276
- **我未查到 draft TP 度选择的定量实测（例如 draft TP=1 vs 8 的吞吐差），标记未查证（§11）。**

### 7.8 结构化输出 / 受限解码

- 冲突机制：grammar 的 FSM/PDA 状态机运行在 **scheduler 进程的 CPU 上**，而草稿阶段需要让 FSM **前向推进 γ 次**；把这个优化到 GPU 上需要每个草稿 token 都做 GPU↔CPU 往返序列化。
- vLLM PR #14702 已为 V1 启用「投机解码 + 结构化输出」，兼容 xGrammar 与 Guidance 后端。
  来源：https://github.com/vllm-project/vllm/pull/14702
- **性能上二者叠加的净收益，我未查到实测数字，标记未查证（§11）。**

---

## 8. 服务化视角：收益如何随负载变化，以及"按负载动态开关"的实践

### 8.1 官方承认的核心事实

**vLLM RFC #4565**（2024-05-02，**closed as not planned**）动机段逐字：
> *"under conditions of high request rates or low speculation accuracy, latency may actually increase."*

提出的三阶段路线：
- Milestone 1：运行队列超阈值时**手动禁用**
- Milestone 2：用 batch size + 离线 profile 的参数（接受率、模型开销）**动态决定提议长度**
- Milestone 3：改为运行时采集，去掉离线 profile 依赖
局限（原文）：当前只能做到 batch 级（全批共享同一提议长度），不能 per-request。
来源：https://github.com/vllm-project/vllm/issues/4565

**vLLM 官方文档**当前对方法选择的定性表述（逐字要点）：
- n-gram 和 suffix decoding 在**低 QPS** 下只有 *"Low to medium gain"*
- draft model 在**高 QPS**（吞吐导向）下 *"Medium gain"*
- *"real gains depend on your model family, traffic pattern, hardware, and sampling settings"*
来源：https://docs.vllm.ai/en/latest/features/speculative_decoding/

### 8.2 ⚠️ vLLM 的 `disable_by_batch_size` 在 V1 里**没有实现**（重要生产陷阱）

- V0 时代存在 `speculative_disable_by_batch_size`：入队请求数超过该值就对新请求关闭投机。SqueezeBits 的评测（§2.3）当年也是这么用的（阈值 64）。
- **但**：vLLM Issue #25112（vLLM **v0.10.1**）报告：配置 `"disable_by_batch_size": 12` **完全没有效果**，用户在代码库里搜索后发现 *"the parameter is nowhere used"*。
  用户请求确认是否还支持、是否会重新引入，称其为 *"a very useful feature to prevent throughput regressions at higher batch sizes with spec decoding"*。
  **状态：closed as not planned / stale，无维护者实质答复。**
  来源：https://github.com/vllm-project/vllm/issues/25112

> **可执行结论**：**不要假设你的 vLLM 会在高负载时自动关掉投机解码。** 请在你的版本上实测该参数是否生效（设一个极小值，压测看行为是否变化）。如果不生效，负载保护必须做在**网关/路由层**（例如按并发路由到开/不开投机的两组实例）。

### 8.3 SGLang：目前最完整的自适应实现

**`--speculative-adaptive`**：运行时调整 `speculative_num_steps` / `speculative_num_draft_tokens`，而不是整个服务生命周期用一个固定值。
文档动机逐字：*"one static step count is rarely optimal."*

配置与默认值：
| 参数 | 默认 |
|---|---|
| `ema_alpha` | 0.2 |
| `warmup_batches` | 10 |
| `update_interval` | 5（每 5 个 batch 重算一次） |
| `down_hysteresis` | −0.25 |
| `up_hysteresis` | 0.0 |
| `ceiling_coeff` | 0（默认关闭） |

决策逻辑：读每请求的实际接受草稿长度 → 求批平均 → EMA 平滑 → 在候选档位间切换。
`target_steps ≈ clamp(round(ema_accept_len) + 1, min(candidate_steps), max(candidate_steps))`
**关键设计**：**按 batch-size 区间维护独立的 tracker**，防止小批的观测污染大批的信号。
限制：仅支持 `--speculative-algorithm EAGLE` / `EAGLE3` 且 `--speculative-eagle-topk 1`，否则回退到静态配置。
**文档未给出任何 benchmark 数字。【口径不全】**
来源：https://docs.sglang.io/advanced_features/adaptive_speculative_decoding.html

另有 `--speculative-disable-by-batch-size`（SGLang 侧）。

SGLang 文档给出的唯一性能表（口径：**1×H100，LLaMA-Instruct 3.1 8B，MT-bench；原文未标注 batch size，推测为 bs1**）：

| 配置 | 吞吐 (tokens/s) |
|---|---|
| SGLang（无投机） | 158.34 |
| SGLang + EAGLE-2 | 244.10 |
| SGLang + EAGLE-3 | 373.25 |

**【口径不全：缺 batch size / 温度 / γ】**
来源：https://docs.sglang.io/advanced_features/speculative_decoding.html

### 8.4 Snowflake Arctic Inference

**口径**：8×H100；TP=2 / TP=8 测试；**主 benchmark 用 max_concurrency = 1**；插件含 `disable_by_batch_size: 64` 参数。

吞吐（tokens/s，Llama 3.1-70B）：

| 工作负载 | 无投机 | LSTM only | Suffix only | n-gram | LSTM + Suffix |
|---|---|---|---|---|---|
| ShareGPT | 76.0 | 172 | 113 | — | **179** |
| HumanEval | 77.2 | 204 | 148 | — | **217** |
| SWE-Bench | 75.8 | 123 | **286** | 175 | **302** |

**读数**：SWE-Bench（agent 重复模式强）上 suffix decoding 286 远超 LSTM 草稿头 123；ShareGPT（开放对话）上正好反过来（113 vs 172）。**同一套硬件同一个模型，仅换任务，最优方法就反转了。**

其它：*"91% of the theoretical maximum speedup"*；agentic 端到端完成时间降低 1.8×–4.5×。
来源：https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/

**gpt-oss 系列**（口径：TP=4；**max_concurrency 1**）：
- gpt-oss-120B：ShareGPT 377.3 tok/s（1.7×）、HumanEval 400.0（1.8×）
- gpt-oss-20B：ShareGPT 476.2（1.6×）、HumanEval 490.2（1.6×）
- *"The acceptance rate averages 44%-50% across our benchmark settings"*；*"With a speculation length of 3, this means that our model can, on average, accurately predict 2.3-2.5 tokens in advance"*
来源：https://www.snowflake.com/en/engineering-blog/faster-gpt-oss-reasoning-arctic-inference/

> ⚠️ **这些是并发=1 的数字。** Arctic 自己带 `disable_by_batch_size: 64` 说明他们清楚大批下要关。**引用 Arctic 的 "2.8× / 4×" 时必须带上"并发 1"这个口径**，否则就是误导。

**SuffixDecoding 的负载相关观察**（原始设计的局限，逐字）：*"At high concurrency levels (e.g., 64 requests), speculation overhead began to dominate runtime"*，此时 n-gram 反超。优化后的版本称在各并发档位下用单一配置都能取得一致收益。
来源：https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/ 、https://arxiv.org/html/2411.04975v2

SuffixDecoding 论文口径：8×NVIDIA H100 80G + 2TB 内存（AWS p5.48xlarge）；Llama-3.1-8B-Instruct 等。
- AgenticSQL，**batch size 1**：up to **5.3×** vs vanilla；比 EAGLE-2/3 快 2.8×；平均接受 **6.3** token/步 vs EAGLE-3 的 3.6
- SWE-Bench 端到端（**4×H100，8 并发任务**）：**1.8–4.5×**

### 8.5 【一手·论文自报】Nightjar：把"高负载下投机变慢"量化出来

**口径**：目标 = DeepSeek-R1-Distill-Qwen-7B、Vicuna-13B；草稿 = DeepSeek-R1-DRAFT-Qwen2.5-0.5B、Vicuna-68m；硬件 = RTX 4090 (24GB) / A100 (40GB) / 双 L20 (48GB)；数据 = ShareGPT、Alpaca、SpecBench、Azure LLM traces；**请求率 1–35 QPS，泊松到达**。
来源：https://arxiv.org/html/2512.22420v5

问题陈述逐字：
> *"speculative decoding improves throughput in low-load, memory-bound systems but degrades performance in high-load, compute-bound environments due to verification overhead."*

**关键数字**：
- γ=3 在 **15 QPS** 时带来 **15.5% 吞吐提升**
- **高负载下，vanilla 反超投机解码，投机解码相对 vanilla 的性能退化最多达 30.25%**

机制方案：contextual multi-armed bandit，按 batch size 选 γ，**必要时置 γ=0（完全关闭）**；高负载时把草稿模型 **offload 到 CPU 内存**，把腾出的显存还给 KV cache。
结果：比静态投机在 13B/Alpaca 上高 14.76% 吞吐、低 20.18% 延迟；比 vanilla 平均高 27.29%。

> **这是本文找到的、对"服务化收益随负载变化"最完整的一条定量证据。** 注意它的硬件是 RTX 4090，交叉点位置不可直接搬到 H100/H200。

### 8.6 【一手·论文自报】SmartSpec：goodput 框架

**定义**（逐字）：goodput = *"Number of Generated Tokens / Execution Time"*，其中 generated tokens **只算被接受的草稿 token + bonus token**——这正是与 throughput 的区别（throughput 会把被拒绝的草稿也算进"做过的工作"）。

高请求率下的退化证据：
- Vicuna-7B + Spider 数据集，**request rate 32** 时，提议 5 个 token 相对不投机**严重退化**
- Vicuna-33B + Spider，提议 ≥3 个 token 时，**request rate 超过 5 就开始增加延迟**
- 结论逐字：*"under higher request rates or low speculation accuracy, it paradoxically increases latency"*

**"up to 3.2×"**：原文为 *"SmartSpec consistently reduces average request latency by up to 3.2× compared to non-speculative decoding baselines across different sizes of target models, draft models, request rates, and datasets."*
**⚠️ 原文未指明这个 3.2× 出现在哪个具体配置**（哪个模型对、哪个 QPS、哪个数据集）。**【口径不全，不可引用为代表性数字】**
来源：https://arxiv.org/html/2406.14066v2

### 8.7 服务化小结：一张"该在哪一层做决策"的表

| 决策层 | 决策内容 | 现成实现 |
|---|---|---|
| 请求级 | 这条请求要不要投机、γ 取多少 | SmartSpec（研究）；vLLM RFC #4565 明确说当前**做不到** per-request |
| 批级 | 本批 γ 取多少 | **SGLang `--speculative-adaptive`**（EAGLE/EAGLE3 + topk=1） |
| 实例级 | 超过某并发就关投机 | SGLang `--speculative-disable-by-batch-size`；**vLLM V1 该参数疑似未实现（§8.2）** |
| 集群/网关级 | 把高并发流量路由到不开投机的实例池 | 无现成实现；**在 vLLM 上这是目前最可靠的兜底手段** |

---

## 9. "上线后发现变慢了"的具体案例

按证据强度排序。**注意：issue 类证据的口径普遍残缺，价值在于"这类事真的会发生"以及"根因分类"，不在于数字。**

### 9.1 vLLM #15025 —— 稳定慢 30%，最优解是关掉

**口径**：目标 1.7B；草稿 135M（SmolLMv2 家族，用 logits 蒸馏训练，作者称接受率高）；`num_speculative_tokens=5`；硬件 L4 / T4（Colab）；vLLM 0.6.2 与 0.7.3；`gpu_memory_utilization=0.9`；`max_model_len=2500`；采样 `temperature=0, top_k=1, max_tokens=256`。
现象逐字：*"consistent performance drop ~30% in terms of speed when using 5 speculative tokens"*。
**减少 spec tokens 会好转，但最优是完全禁用投机解码。**
状态：closed as not planned / stale，无维护者答复。
来源：https://github.com/vllm-project/vllm/issues/15025

> **根因分类：小目标模型 + 独立草稿模型。** 目标只有 1.7B，本身单次前向就极快，草稿 135M 的前向 + 调度开销占比过高（对照 §6.1：draft-model 在 bs1 时草稿占 47%）。**目标模型越小，投机解码越不划算**——这条判据在 §1 里对应 R5。

### 9.2 vLLM #8439 —— n-gram 比不开还慢

**口径**：7B 模型 + n-gram 投机。
数字：**投机 61.79 tokens/s vs 基线 69.79 tokens/s**（即 **0.885×**）。
状态：closed as not planned。
来源：https://github.com/vllm-project/vllm/issues/8439
**【口径不全：未给硬件、并发、数据集、prompt_lookup 参数】**

### 9.3 vLLM #5239 —— 低 QPS 下也没有收益

**口径**：目标 Llama-2-70B-chat；草稿 TinyLlama-1.1B-chat-**GPTQ**；300 条 prompt，平均输入 158 token，最大输出 100 token。
现象逐字：*"using Speculative Decoding way is almost same performance or lower than normal(Only using Target Model) even low query per second."*
状态：closed as not planned。原报告只附了图，**正文无可引用数字**。
来源：https://github.com/vllm-project/vllm/issues/5239

> **值得注意的细节：草稿模型是 GPTQ 量化的。** 结合 §7.4，量化草稿模型可能同时带来（a）草稿前向并不如预期快（低批下量化 kernel 开销占比高）、（b）与 FP16 目标模型的分布偏移压低接受率。**这是一个未被诊断但很可疑的根因。**

### 9.4 vLLM #19254 —— Qwen3-32B-FP8 + n-gram

**口径**：Qwen3-32B-FP8；**4×H20，TP=4**；vLLM **0.9.0.1**；n-gram 配置试过 `num_speculative_tokens/prompt_lookup_max` = 5/4、3/3、2/2；数据集 ShareGPT。
**可靠的一条数字**：**Draft acceptance rate 初始 38.8%，峰值负载时升到 61.3%。**
状态：closed as not planned / stale。
来源：https://github.com/vllm-project/vllm/issues/19254

> ⚠️ **口径警告**：该 issue 标题声称"比不开慢"，但我自动抽取到的吞吐数字方向与标题**相反**（抽到的"开投机"一侧吞吐更高）。我无法在不打开原页面逐字核对的情况下判定哪个是 before / 哪个是 after。
> **处理：本文只收录接受率数字，吞吐对比标记【存疑，需人工核对原 issue】，正文不得引用其吞吐数字。**

### 9.5 vLLM #41838 —— 接受长度**随运行时间单调退化**（最有生产价值的一条）

**口径**：
- Model A：约 250GB（BF16 权重 + KV cache），EAGLE2 与 EAGLE3
- Model B：FP8 权重/激活/KV cache + EAGLE3
- Model C：FP8 + EAGLE3
- vLLM **0.17.1**，在 **0.19.1** 上复现依旧存在

**现象**：接受长度 (AL) *"smoothly and monotonically regresses over time"*，不会自行恢复，**只有重启服务才能立刻回到初始值**。高流量的 prompt 类型退化更严重。

| 模型 | 起始 AL | 退化后 AL | 条件 |
|---|---|---|---|
| Model A | ~3.07 | ~2.8（数小时内） | — |
| Model B | ~4.5 | **~2.8** | batch = 4 |
| Model B | ~4.2 | ~3.8 | batch = 2 |
| Model C | ~3.9 | ~3.5 | vLLM |
| Model C | 稳定 | 稳定 | **TensorRT-LLM（不退化）** |

猜测根因：投机解码下 KV cache 状态处理问题（报告者关联到 #14649，涉及近似隐藏状态与 KV cache 覆写）。
**状态：Open（未解决）。**
来源：https://github.com/vllm-project/vllm/issues/41838

> **为什么这条最重要**：它意味着**你的上线基准测试是骗人的**。你压测 10 分钟看到 AL=4.5、加速 1.8×，跑几小时后 AL 掉到 2.8，实际收益腰斩，而监控上看不出任何报错。
> **可执行动作**：把 **AL 作为一等监控指标持续打点**，并设"AL 相对启动值下降超过 X%"的告警。**只测一次不算测过。** 对应 §1.2 的 Y6。

### 9.6 【一手·第三方】消费级双卡：零收益

**口径**：2×NVIDIA RTX 5060 Ti（16GB×2 = 32GB）；llama.cpp；n-gram（`--spec-type ngram-mod --draft-max 64 --draft-min 48 --spec-ngram-size-n 24 --spec-ngram-size-m 48`）；Q4_K_M 量化；flash attention；8K 上下文；双卡切分。

| 模型 | 基线 | 开 ngram-mod | 变化 |
|---|---|---|---|
| Gemma 4 26B-A4B（MoE） | 88.3 tok/s | 88.2 tok/s | **0%** |
| Qwen3-32B（Dense） | 20.4 tok/s | 20.6 tok/s | **+1%** |

跨 **8 个不同 prompt**（代码生成、API 设计、bash 脚本等）测试：*"Zero improvement"*。只有**重复同一 prompt** 时才出现 *"Almost 5x speedup by run 10"*（即纯 n-gram 缓存命中，不是泛化收益）。

作者诊断：瓶颈是 *"memory bandwidth, not compute"*，消费级 GPU 缺少 A100/H100 那样的 算力/带宽 比；MoE 上 *"each speculative token may activate different experts"* 进一步降低批处理效率。
来源：https://dev.to/defilan/i-tested-speculative-decoding-on-my-home-gpu-cluster-heres-why-it-didnt-help-3ej6

> **这条揭示了一个反直觉点**：很多人以为"memory-bound 越严重，投机解码越有用"。**错。** 投机解码需要的是**闲置的算力**。消费级卡是**算力和带宽同时都不够**，没有可供挪用的闲置算力，所以投机解码无从获利。对应 §1.1 的 R7。

### 9.7 vllm-ascend #9247 —— 开 MTP 反而让 TTFT 变差

见 §7.2 全文。**根因不在 decode 而在 prefill（prefix cache 命中减半）。**
这是"上线变慢"里**最难自查**的一类：你监控 TPOT 会看到改善，监控 TTFT 才发现总账是亏的。
来源：https://github.com/vllm-project/vllm-ascend/issues/9247

### 9.8 案例根因分类汇总（正文可直接用作排查树）

| 根因类别 | 案例 | 自查方法 |
|---|---|---|
| 目标模型太小 / 草稿太贵 | §9.1、§9.2 | 测草稿前向占单步时间的比例，>30% 就危险 |
| 草稿模型量化引入的分布偏移或 kernel 开销 | §9.3（疑似） | 换 FP16 草稿再测一遍 |
| 并发过高，掉入 compute-bound | §2.x、§8.5 | 扫并发画曲线，找自己的交叉点 |
| 草稿头训练窗口 < 实际输入 | §3.3 | 查 checkpoint 的训练 max_len |
| 接受长度随运行时间退化 | §9.5 | 持续监控 AL，设退化告警 |
| 与 prefix caching 冲突，亏在 prefill | §7.2、§9.7 | 同时监控 TTFT 与 prefix cache 命中率 |
| CUDA Graph 被禁用 / 走了 prefill kernel 路径 | §7.3 | 确认开投机后 CUDA Graph 仍生效 |
| 硬件本身没有闲置算力 | §9.6 | 看 GPU 的 算力/带宽 比与实测 MFU |

---

## 10. 口径混乱名人堂

本节记录调研中遇到的、同一件事在不同来源被报成差异很大数字的案例，并指出差异来自哪个口径。**这是本文对写正文的人最有价值的一节：它教你怎么不被骗。**

### 10.1 冠军：τ（接受长度）被当成加速比

- **事实**：EAGLE 在 Vicuna-7B / MT-bench / T=0 下，**τ = 3.94，加速比约 2.90×**（§4.2）。
- **混乱**：调研过程中，我使用的自动摘要工具在第一次抽取 EAGLE 论文时，直接输出了 "Vicuna 7B: 3.94x (T=0) → 3.17x (T=1)" —— 把 T=0 的 τ 和 T=1 的 τ 当成了两个温度下的**加速比**。我在第二次定向抽取（明确要求区分两个指标）后才拿到正确数据。
- **差异来源**：EAGLE 系论文的表格把 Speedup Ratio 和 τ 并排放，两列数值量级接近（都是 3–7），极易串行。
- **教训**：**看到任何 3–7 之间的"投机解码加速比"，先怀疑它是 τ。** 判断方法：如果同一篇里代码任务的数字比对话高、且各模型间数值高度接近，多半是 τ。

### 10.2 亚军：EAGLE-3 论文正文与自己的 Table 5 打架

- **正文说**：*"EAGLE shows the maximum throughput improvement at a batch size of 24, while EAGLE-3 shows this at 56."*
- **Table 5 说**：EAGLE 在 bs2 是 1.30×（全表最高），bs24 只有 1.03×（几乎打平）。
- **差异来源**：正文那句大概率想表达的是"EAGLE 保持正收益的最大 batch 是 24"（即 bs24 = 1.03× > 1.0，bs32 = 0.93× < 1.0），被写成了 "maximum throughput improvement"。
- **处理**：**以表为准。** 引用时说"EAGLE 的盈亏平衡点在 bs24 与 bs32 之间"，不要引用正文那句。
- 来源：https://arxiv.org/html/2503.01840

### 10.3 季军：Meta 转述 EAGLE 退化时的 batch size 对不上

| 来源 | 陈述 |
|---|---|
| Meta 2508.08192 | batch **2 → 48**，速度比 **1.3× → 0.7×** |
| EAGLE-3 Table 5 | bs2 = 1.30×，bs**48** = **0.82×**，bs**56** = **0.71×** |

- **差异来源**：1.3× 对得上（bs2）；0.7× 对应的是 **bs56 而不是 bs48**。Meta 要么是自己独立测的（口径与 EAGLE-3 论文不同），要么是转述时把 56 写成了 48。原文未标注该数据的来源。
- **处理**：两条**分别记录、各带各的口径**，不要合并成一句"EAGLE 在 bs48 会掉到 0.7×"。

### 10.4 DeepSeek-V3 MTP 的 "85–90% / 1.8× TPS"

见 §6.5 完整记录。**广泛引用，我 4 次定向抽取 arXiv HTML 未能逐字定位（其中一次明确报告文档被截断）。标记【待核验】。**
**教训**：一个数字被引用一万次，不等于有人核对过原文。写正文前必须自己打开 PDF。

> ※ **2026-08-22 复核：已核到，【待核验】解除。** 逐字原句在 **§5.4.3（PDF 第 35 页）**，arXiv HTML v1/v2 与官方 PDF 三种渲染一致，详见 §6.5 的复核记录。
> **教训要改写**：原来的教训"必须自己打开 PDF"是对的，但**还漏了半句 —— 抽取失败不等于原文没有**。四次失败里至少一次是明确的截断报告；把"我抽不到"直接读成"原文没有"，是本库这次差点犯的错。**正确的动作是换渲染（HTML→PDF）+ 落盘本地正则，而不是换一个摘要模型再问一遍。**

### 10.5 SqueezeBits 同一配置的接受率被记作两个值

- 同一篇文章、同一个草稿模型（Qwama-0.5B）、1K 输入下的接受率，我两次抽取分别得到 **53.5%** 和 **54.0%**。
- **差异来源**：极可能是文中不同图（Figure 7 vs Figure 8）测的是略有差异的配置（例如不同 draft token 数），或正文与图注取整方式不同。
- **处理**：本文记作 **"约 53.5%–54.0%"**，并注明来源单一。**这个 0.5 个百分点不影响结论，但它提醒你：连同一篇文章内部都未必自洽。**

### 10.6 SpecDecode-Bench 项目网站 vs 论文正文

- **网站摘要**（经自动抽取）声称：树验证 k=21（分支因子 4）在 **bs64 跌破 1× 加速**。
- **论文正文**（经自动抽取）声称：*"Across all workloads, every SD variant outperforms the no-SD baseline."*，且不含树验证敏感性分析。
- **差异来源**：无法确定——可能是网站含论文外的补充实验，也可能是某一次自动抽取产生了幻觉。
- **处理**：**以论文正文为准**；k=21 的说法标记【待核验】，**正文不得引用**。
- **教训**：项目主页的"亮点摘要"和论文正文经常不是同一套实验，引用请指向论文。

> ※ **2026-08-22 复核：此处记载有误，上面这条"冲突"根本不存在。**
> **真相是版本差**：我读的是 **v1**（无树实验），网站对齐的是 **v2**（2026-03-18 修订，新增 "Tree-Style Verification" 节 + Figure 2）。**网站摘要是对的，两个"差异来源"猜测都不对**（既不是网站的论文外实验，也不是抽取幻觉）。完整复核与全口径见 §2.4 的复核块。
> **教训要整个换掉**：原来那条"网站与论文常常不是同一套实验"在这里**不成立**，而且它把我导向了错误的处理动作（禁用一条真数据）。
> **正确的教训（本库新增踩坑）**：**网站摘要和论文正文"打架"时，第一件要查的不是谁可信，是它们是不是同一个版本。** arXiv 的 `abs/` 页会明写 *"last revised …(this version, v2)"*；而本库四处引用的全是写死的 `…/html/<编号>v1`，**把版本号焊死在 URL 里，等于让自己永远读不到修订**。
> **可操作的规矩两条**：① 引用 arXiv 正文时**先开 `abs/` 页看有没有 vN**，正文 URL 优先用**不带版本号**的形式（或显式写明"本库核的是 vX"）；② 记录"论文里没有 X"这类**否定断言**时，**必须同时记下核的是哪个版本、哪天核的** —— 否定断言的保质期比肯定断言短得多。

### 10.7 MoE 到底受益还是受损：四方混战

见 §7.6 全文。同一个问题四个来源四种结论：

| 来源 | 结论 | 真实口径 |
|---|---|---|
| Cohere | MoE **更**受益，BS=1 就有 1.95× | 自家中等规模 MoE，dense 对照是 111B |
| MoESD | 中等 batch 更受益，大 batch 转负 | 全专家是否已激活是分水岭 |
| Meta | Llama4 Maverick 加速比**随 batch 下降** | 400B MoE，8×H100，生产规模 |
| EcoSpec | 超大 MoE 上 EAGLE-3 只有 **1.10–1.22×** | bs=1，8×H200，671B/235B/120B |

- **差异来源**：(1) MoE 规模与稀疏度差异巨大；(2) 所处 batch 区间不同；(3) 对照基准不同（对比 dense vs 对比自己不开投机）。
- **处理**：**禁止跨来源合表。** 正文只讲 §7.6 末尾那条可执行判据（测激活专家并集的膨胀比）。

### 10.8 "并发 1" 的数字被当成服务化收益

- Snowflake Arctic 的 "2.8× faster / 4× faster"、SuffixDecoding 的 "5.3×"、几乎所有投机解码论文的主表——**绝大多数是 batch size 1 或 max_concurrency 1 下测的**。
- 佐证：Arctic 自己的插件里带 `disable_by_batch_size: 64`，说明他们清楚大批下要关。
- **处理**：任何投机解码加速比，**第一件事是找 batch size**。找不到就默认它是 1，并按 §2.1 的衰减曲线心里打个对折以上的折扣。

### 10.9 无出处的"经验数字"（不可引用，仅存档为反面教材）

某些流传较广的博客给出高度具体但**无任何引用**的数字，我逐条溯源均失败：

| 流传说法 | 出处状况 |
|---|---|
| "并发 4–8 以上投机解码就没用了" | 无引用 |
| "α=0.6 → 2.4× 加速；α=0.8 → 3.7×；α<0.5 就不划算" | 无引用，且与 §4.4（α=0.30 的 n-gram 跑出 1.39×）直接矛盾 |
| "H100 上草稿模型状态多占 10–20 GB 显存" | 无引用 |
| "EAGLE-3 在所有生成位置保持 70–80% 接受率" | 无引用 |
| 某流媒体平台"p50 延迟降到 190ms、成本降 40%" | 无引用，无法验证平台身份 |
来源（作为反面教材记录）：https://tianpan.co/blog/2026-04-17-speculative-decoding-production-hidden-traps

另一类：Modular/BentoML 推理手册明确自陈其性能数字 *"These results are from informal tests and for reference only."*，其中"TP=1 时吞吐在约 20–30 并发就提前见顶"可作为**方向性**参考，不作为数字引用。
来源：https://handbook.modular.com/inference-optimization/speculative-decoding

**教训**：**α 到加速比之间没有通用换算公式。** 那个换算必须知道草稿成本、验证成本、框架开销三项，而这三项是硬件和实现相关的。任何给出"α → 加速比"通用对照表的文章，都可以直接判定为不可信。

---

## 11. 诚实缺口：本次调研**未查证**的问题

正文写到这些地方时，必须写「未查证」，不许推测填补。

1. **Blackwell（B200 / GB200 / GB300）上的投机解码实测。** 按 §3.2 的逻辑，算力/带宽比继续升高应当让交叉点左移、收益变窄，但**我没有找到任何实测数据**。
2. **FP8（W8A8-FP8）下投机解码收益衰减的定量数字。** §7.4 只覆盖了 W8A8(SmoothQuant) / W4A16 / W4A8。
3. **PD 分离架构下投机解码的定量收益**，以及 decode 节点开投机后对 KV 传输、decode pool 批容量的影响。§7.5 只有定性推理。
4. **draft tensor parallel size 的定量选择依据**（draft TP=1 vs TP=8 的实测对比）。§7.7 只有参数存在性。
5. **结构化输出 + 投机解码同开时的净收益实测。** §7.8 只确认了功能兼容。
6. **α 随 top-p / top-k 变化的实测曲线。** §5.2 只有机制推理和温度维度的实测。
7. **一份权威的、逐框架的"greedy 下投机 vs 非投机输出逐 token 一致性"测试报告。** §5.5 只有单篇论文的主张。
8. **长数字串 / 随机 token 串等病理输入上的接受率实测。** 直觉上应当极低，但**未查到数据**（本轮 WebSearch 配额在此项前耗尽）。
9. **chunked prefill 与各类投机方法在具体 vLLM 版本上的完整支持矩阵**，以及二者同开时收益被削薄的**定量**幅度。§7.1 只有机制分析。
10. ~~**DeepSeek-V3 MTP 的 85–90% / 1.8× TPS 原句**（§6.5、§10.4）——需人工打开 PDF 核对。~~ ※ **2026-08-22 已核验关闭**：原句在 §5.4.3（PDF 第 35 页），HTML v1/v2 与 PDF 三渲染一致；口径仍不全，见 §6.5 复核块。
11. **vLLM V1 当前是否已重新实现按 batch size 禁用投机**（§8.2 基于 v0.10.1 的 issue，可能已在更新版本中改变）。上线前请在**你自己的版本**上实测。

---

## 12. 给正文作者的三条写作建议

1. **本库的正文应当以 §1 的判定清单开篇**，而不是以"投机解码原理"开篇。原理是手段，判定清单才是这份知识库的差异化价值。
2. **每引用一个加速比，强制带上 batch size。** 建议在正文里统一使用 `2.35×（bs=1，H200，Llama-3.1-8B，LongSpecBench 4K–64K，T=0）` 这种带括号的写法，让读者一眼看到口径。
3. **把 §10 单独做成一节放进正文**，标题可以叫「读投机解码论文时如何不被骗」。这一节的实用价值可能高于任何技术细节。

---

## 附录 A：本文引用的全部一手来源

**论文**
- EAGLE：https://arxiv.org/html/2401.15077
- EAGLE-3：https://arxiv.org/html/2503.01840
- OWL（长上下文，EAGLE3 = 0.81×）：https://arxiv.org/abs/2510.07535 ｜ https://arxiv.org/html/2510.07535v1
- MagicDec：https://infini-ai-lab.github.io/MagicDec/ ｜ https://infini-ai-lab.github.io/MagicDec-part2/
- TriForce：https://arxiv.org/abs/2404.11912 ｜ https://infini-ai-lab.github.io/TriForce/
- Meta *Efficient Speculative Decoding for Llama at Scale*：https://arxiv.org/html/2508.08192 ｜ https://ai.meta.com/research/publications/efficient-speculative-decoding-for-llama-at-scale-challenges-and-solutions/
- SpecDecode-Bench：https://arxiv.org/html/2601.11580v1 ｜ https://specdecode-bench.github.io/
- Batch Speculative Decoding Done Right：https://arxiv.org/html/2510.22876v3
- Decoding Speculative Decoding（NAACL 2025）：https://arxiv.org/html/2402.01528v3 ｜ https://aclanthology.org/2025.naacl-long.328.pdf
- Speculative Decoding Across Languages：https://arxiv.org/html/2605.30580v1
- Scaling Up, Speeding Up：https://arxiv.org/html/2509.04474v1
- Speculative Decoding Meets Quantization：https://arxiv.org/html/2505.22179v1
- EcoSpec（MoE 专家成本）：https://arxiv.org/html/2607.12696v1
- MoESD：https://arxiv.org/pdf/2505.19645
- SmartSpec（goodput）：https://arxiv.org/html/2406.14066v2
- Nightjar（负载自适应）：https://arxiv.org/html/2512.22420v5
- SuffixDecoding：https://arxiv.org/html/2411.04975v2
- Spec-Bench（综述与平台）：https://arxiv.org/html/2401.07851v2 ｜ https://sites.google.com/view/spec-bench ｜ https://github.com/hemingkx/Spec-Bench
- DeepSeek-V3 技术报告：https://arxiv.org/html/2412.19437

**框架文档**
- vLLM 投机解码：https://docs.vllm.ai/en/latest/features/speculative_decoding/
- SGLang 投机解码：https://docs.sglang.io/advanced_features/speculative_decoding.html
- SGLang 自适应投机解码：https://docs.sglang.io/advanced_features/adaptive_speculative_decoding.html

**GitHub Issue / PR**
- vLLM #4565（RFC 自动化投机解码）：https://github.com/vllm-project/vllm/issues/4565
- vLLM #5016（chunked prefill 结合）：https://github.com/vllm-project/vllm/issues/5016
- vLLM #5239（性能持平或更低）：https://github.com/vllm-project/vllm/issues/5239
- vLLM #8439（n-gram 更慢）：https://github.com/vllm-project/vllm/issues/8439
- vLLM #10276（spec + TP + chunked prefill 崩溃）：https://github.com/vllm-project/vllm/issues/10276
- vLLM #15025（稳定慢 30%）：https://github.com/vllm-project/vllm/issues/15025
- vLLM #19254（Qwen3-32B-FP8 n-gram）：https://github.com/vllm-project/vllm/issues/19254
- vLLM #21984（Padded Speculative Decoding / FlashMLA）：https://github.com/vllm-project/vllm/issues/21984
- vLLM #23679（spec-decode CUDA Graph 重构 PR）：https://github.com/vllm-project/vllm/pull/23679
- vLLM #25112（disable_by_batch_size 无实现）：https://github.com/vllm-project/vllm/issues/25112
- vLLM #14702（投机解码 + 结构化输出 PR）：https://github.com/vllm-project/vllm/pull/14702
- vLLM #41838（EAGLE 接受长度随时间退化，Open）：https://github.com/vllm-project/vllm/issues/41838
- vllm-ascend #9247（MTP 破坏 prefix cache）：https://github.com/vllm-project/vllm-ascend/issues/9247
- vLLM 论坛（prefix caching + spec decode）：https://discuss.vllm.ai/t/can-speculative-decoding-and-prefix-caching-take-effect-simultaneously/1291/4

**工程博客（第三方实测）**
- SqueezeBits vLLM vs TensorRT-LLM #11：https://blog.squeezebits.com/vllm-vs-tensorrtllm-11-speculative-decoding-37301
- Red Hat *Fly Eagle(3) fly*：https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding
- Snowflake Arctic 投机解码：https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/
- Snowflake SuffixDecoding 生产化：https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/
- Snowflake gpt-oss + Arctic：https://www.snowflake.com/en/engineering-blog/faster-gpt-oss-reasoning-arctic-inference/
- Cohere *Why MoE Models Get More From Speculative Decoding*：https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding
- dev.to 家用集群实测（零收益）：https://dev.to/defilan/i-tested-speculative-decoding-on-my-home-gpu-cluster-heres-why-it-didnt-help-3ej6

**反面教材（无出处，仅存档）**
- https://tianpan.co/blog/2026-04-17-speculative-decoding-production-hidden-traps
- https://handbook.modular.com/inference-optimization/speculative-decoding
