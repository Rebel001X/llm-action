# RS-2 社区讲解盘点与评点

> 调研笔记，**不参与双链**，不是正文。服务于本库第 **26-社区精彩解释精选-好在哪与错在哪**、**27-常见误解与判据**、**30-附录-优秀文章与资料清单**，以及散落在 04/05/06/07/18/19/20 各篇的引用与评点。
> 核验日期：**2026-08-22**。凡标「链接无法访问（2026-08 核验）」的，是本次实际抓取失败，**没有读过内容，不做内容评点**。

---

## 0. 调研方法与可信度声明

### 0.1 方法

- 一律用 WebFetch 实际打开页面读正文；论文用 `ar5iv.labs.arxiv.org` 取全文 HTML 后**本地正则抽取原句**，避免二手转述。
- 代码仓库不看 README 就下结论，**直接抓 `raw.githubusercontent.com` 的源码文件**，逐行核对接受判据与残差分布。
- 引擎实现同理：vLLM 的 `vllm/v1/sample/rejection_sampler.py` 与 `vllm/config/speculative.py` 是本次直接下载 main 分支源码核对的。

### 0.2 本次访问失败的来源（不评点，只记录）

| 来源 | 状态 | 备注 |
|---|---|---|
| 知乎专栏（`zhuanlan.zhihu.com/p/651359908`、`/p/15837326799`、`/p/671432448`、`/p/685282553`、`/p/7162909442`、`/p/15575453436`、`/p/27272034867`） | **HTTP 403**，WebFetch 与 curl 均被拒 | 链接无法访问（2026-08 核验）。搜索摘要可见标题，但**未读正文，不评点** |
| CSDN《论文导读 \| 投机解码加速模型推理》`blog.csdn.net/weixin_48167662/article/details/139006059` | **HTTP 521** | 链接无法访问（2026-08 核验） |
| WaytoAGI 飞书《（10）深入LLM投机采样(上)》 | **302 跳登录页** | 链接无法访问（2026-08 核验） |
| Andrej Karpathy 的投机执行推文 `x.com/karpathy/status/1697318534555336961` | **HTTP 402** | 链接无法访问（2026-08 核验）。**这条在社区被引用极广，本库若要引用必须转引二手，并显式标注"原推文未能直接核验"** |
| `openlm.ai/speculative-decoding-in-vllm/` | **HTTP 403** | 链接无法访问（2026-08 核验） |
| Aphrodite Engine 投机解码文档 `aphrodite.pygmalion.chat/spec-decoding/overview/` | **HTTP 404** | 链接已失效（2026-08 核验） |
| B站视频、微信公众号、YouTube 讲解 | 本次 **WebSearch 配额用尽（200/200）**，未能定位到可抓取的具体 URL | **本轮不覆盖视频类来源**，留给下一轮调研 |

> **给后续 agent 的提示**：知乎/CSDN/微信这条线在本环境里基本抓不动。若正文必须引用中文社区材料，优先用 **博客园（cnblogs.com，curl 可通，HTTP 200）**。B 站与公众号建议直接放弃"逐字引用"，改为只列条目 + 标注"未核验内容"。

### 0.3 两个必须先钉死的口径（贯穿全篇评点）

**(a) 记号冲突（本次逐字核实，两篇奠基论文的 p/q 恰好相反）**

| 论文 | 目标模型 | 草稿模型 | 原文出处 |
|---|---|---|---|
| Leviathan et al. 2211.17192 | **$p$**（$M_p$） | **$q$**（$M_q$） | Algorithm 1：`Sample γ guesses x_{1..γ} from M_q`；`Run M_p in parallel` |
| Chen et al. 2302.01318（DeepMind） | **$q$** | **$p$** | Algorithm 2 原文："Given auto-regressive **target model $q(.|.)$**, and auto-regressive **draft model $p(.|.)$**" |

**这不是小事**：社区材料在两套记号之间来回抄，导致大量"接受判据写反了"的观感错误。本库正文统一用 **Leviathan 记号（p=目标、q=草稿）**，并在第 04/05 篇显式给出这张对照表；引用任何材料的公式时，**先判定它用的是哪套**再翻译。

**(b) "无损"的三种口径（对应铁律一的 L1/L2/L3）**

本次核验发现的关键事实：**"输出一样"这个说法的源头，就在 Leviathan 论文的摘要本身。**

- 摘要（loose）："an algorithm to sample from autoregressive models faster **without any changes to the outputs**"、"show a 2X-3X acceleration ... **with identical outputs**"
- 正文（precise）："we are able to accelerate inference ... and **without changing the model output distribution**"

**同一篇论文，摘要说"输出不变"，正文说"输出分布不变"。** 社区绝大多数二手讲解只抄了摘要那句。这是本库第 07 篇最好的开场素材：**这个误解不是社区笨，是原文摘要给了口实。**

---

## 1. 一句话结论

> 社区在**"怎么做"**上讲得相当好（比喻、图、代码基本都对），在**"到底保证了什么"**和**"什么时候不划算"**上系统性偏弱：
> **接受判据与残差分布的错误率很低**（本次核到的所有源码实现全部正确，包括中文材料），
> **但"无损 = 输出一样"的错误率极高**（本次核到的 40 份材料里，明确踩坑的有 8 份，其中包括 NVIDIA 官方博客、PyTorch 官方博客、Wikipedia 词条），
> **而"什么时候变慢"几乎只有引擎方在说**（vLLM / TensorRT-LLM / IBM / HF-Whisper 给了可判定阈值，教学向材料几乎全部缺席）。

换句话说：**社区高频错误的重心，不在 §0.3(a) 的公式，而在 §0.3(b) 的口径和铁律二/三的账。**

---

## 2. 八条高频错误的命中矩阵

图例：**✗ = 踩了**；**◎ = 不但没踩，还主动纠正了（可直接采纳为正文论据）**；**○ = 未踩**；**— = 未涉及该话题**

| # | 材料 | E1 无损=结果同 | E2 判据丢随机性 | E3 拒绝后从 p 重采 | E4 草稿越小越好 | E5 提升吞吐 | E6 Medusa 说成无损 | E7 MTP 混淆 | E8 E[τ] 不提独立性 |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | Leviathan 2211.17192（论文） | ✗（摘要）/◎（正文） | ○ | ○ | ◎ | ○ | — | — | **◎ 原文明写 i.i.d. 假设** |
| 2 | Chen 2302.01318（论文） | ◎ | ○ | ○ | — | ○ | — | — | — |
| 3 | Google Research 回顾博客 | ◎（分布口径精确） | ○ | ○ | — | ○ | — | — | — |
| 4 | HF《Assisted Generation》 | ○ | — | — | — | ◎（明说仅 bs=1） | — | — | — |
| 5 | HF《Whisper 投机解码》 | **✗** | — | — | — | ◎（明写 bs>4 反而更慢） | — | — | — |
| 6 | HF《Universal Assisted Generation》 | ○ | — | — | — | ○ | — | — | — |
| 7 | HF《Dynamic Speculation Lookahead》 | ○ | — | — | — | ○ | — | — | — |
| 8 | HF《Optimal Lossy Variant》(Tran-Thien) | ◎ | ○ | ○ | — | ○ | ◎（专讲有损） | — | — |
| 9 | Jay Mody《Speculative Sampling》 | **◎ 全网最好的一条纠正** | ○ | ○ | — | ○ | — | — | — |
| 10 | PyTorch gpt-fast 博客（正文） | **✗** | **✗（"扔掉不匹配的"）** | ○ | ○ | ◎（全文注明 bs=1） | — | — | — |
| 11 | gpt-fast `generate.py`（代码） | — | ○ | ○ | — | — | — | — | — |
| 12 | PyTorch/IBM《Hitchhiker's Guide》 | **✗** | — | — | ○ | ◎（明写 bs>64 掉吞吐） | — | — | — |
| 13 | vLLM 博客 2.8x | **✗（"lossless"无口径）** | — | — | — | ◎（给出高 QPS 反向数字） | — | — | — |
| 14 | vLLM issue #10318 | — | — | — | — | ◎（实测证伪该博客） | — | — | — |
| 15 | vLLM 官方文档（2026） | **◎ 最严谨的一份** | — | — | — | ○ | — | — | — |
| 16 | vLLM `rejection_sampler.py` | — | ○ | ○ | — | — | — | — | — |
| 17 | vLLM `config/speculative.py` | — | ○ | ○ | — | — | — | **◎ 明写 MTP 复用告警** | **◎ 拒绝几何模型** |
| 18 | Aleksa Gordić《Inside vLLM》 | ○（措辞略松） | ○ | ○ | — | ○ | — | — | — |
| 19 | SGLang 官方文档 | — | — | — | — | ✗（把 tok/s 叫 throughput 且无 batch） | — | — | — |
| 20 | LMSYS SpecForge | — | — | — | — | ○ | — | — | — |
| 21 | NVIDIA《An Introduction to Speculative Decoding》 | **✗✗（最严重）** | **✗** | — | — | **✗** | — | — | — |
| 22 | NVIDIA《TensorRT-LLM 3.6x throughput》 | ○ | — | — | **◎ 数据直接证伪** | **✗（标题口径不全）** | — | — | — |
| 23 | TensorRT-LLM 官方文档 | ◎（明说只支持贪心） | — | — | — | ◎（明写只在小 batch 有效） | — | ✗（MTP relaxed 未标有损） | — |
| 24 | AMD ROCm Deep Dive | ○ | — | — | — | ○ | — | — | — |
| 25 | Snowflake Arctic Inference | **◎ 工程上把 L1/L2 掰开了** | — | — | — | ○ | — | — | — |
| 26 | Red Hat gpt-oss 投机解码（2026-04） | ✗（"no change to output quality"） | — | — | — | **◎ 高并发反例，重要** | — | — | — |
| 27 | Together.ai Medusa 博客 | — | — | — | — | ○ | **◎ 原文自己承认放宽了分布** | — | — |
| 28 | Medusa GitHub README | — | — | — | — | ○ | ✗（只写"typical acceptance"不提有损） | — | — |
| 29 | EAGLE GitHub README | ○（措辞精确） | — | — | — | ○ | — | — | — |
| 30 | Apple ReDrafter 页 | — | — | — | — | ○ | — | — | — |
| 31 | Spec-Bench | — | — | — | — | **◎ 口径最全的榜单** | — | — | — |
| 32 | Wikipedia《Speculative decoding》 | **✗** | — | ✗（完全不提残差分布） | — | ○ | — | — | — |
| 33 | LLM Inference Handbook | **✗** | — | — | ◎ | ◎（给并发拐点） | — | — | — |
| 34 | Charles Frye 系统综述 | ○ | — | — | — | ○ | — | — | — |
| 35 | feifeibear/LLMSpeculativeSampling | — | ○ | ○ | — | — | — | — | — |
| 36 | romsto/Speculative-Decoding | ◎（README 措辞精确） | ○ | **⚠ 提供了错误变体开关** | — | — | — | — | — |
| 37 | shreyansh26/Speculative-Sampling | — | ○ | ○ | ◎ | — | — | — | — |
| 38 | E2E Networks EAGLE-3 长文 | **✗** | ○ | — | — | ◎（batch 衰减表很硬） | — | — | — |
| 39 | 博客园·罗西的思考（中文） | ◎ | ○ | ○ | — | — | — | — | — |
| 40 | 博客园·SHICENT（中文） | ○ | ○ | ○ | — | ○ | ✗（漏讲） | — | **✗** |
| 41 | tbr8.org（中文） | ○（正文）/✗（标题） | ○ | ○ | — | **✗（标题）** | — | — | — |

**统计（只统计"明确涉及且判定得了"的格子）**：

- E1（无损=结果相同）：**踩坑 8 份，主动纠正 8 份**。踩坑方包含 NVIDIA 官方、PyTorch 官方、Wikipedia、Modular Handbook —— **这是权威来源污染率最高的一条**。
- E2（判据丢随机性）：**踩坑 2 份**（NVIDIA intro、gpt-fast 博客正文），其余全部正确。
- E3（拒绝后从 p 重采）：**没有一份材料写错**；只有 Wikipedia 完全不提残差分布（属于漏讲不是写错），romsto 仓库提供了一个可开关的错误变体（见 §3.F）。
- E4（草稿越小越好）：**没人明说这句话，但也几乎没人正面反驳**。反驳它的最硬证据在 NVIDIA 3.6x 博客的数据表和 Leviathan 论文表 3 里（见 §4.4）。
- E5（提升吞吐）：**教学向材料集体缺席，引擎向材料集体在场**。
- E6（Medusa 有损）：**Together.ai 原文自己说清楚了，下游转述几乎全丢**。
- E7（MTP 训练目标 vs 推理草稿）：本轮材料里几乎无人正面讨论；唯一硬证据在 vLLM 源码告警里。
- E8（E[τ] 独立性前提）：**Leviathan 原文写得清清楚楚"If we make the simplifying assumption that the βs are i.i.d."，中文长文照抄公式时把这句丢了。**

---

## 3. 分组盘点

### A. 奠基文本与官方口径

---

#### Fast Inference from Transformers via Speculative Decoding

- URL：https://arxiv.org/abs/2211.17192 ；全文 HTML https://ar5iv.labs.arxiv.org/html/2211.17192 （**均访问成功**）
- 作者/机构：Yaniv Leviathan、Matan Kalman、Yossi Matias（标注 equal contribution）　　- 年月：v1 **2022-11-30**，末版 **2023-05-18**（ICML 2023 oral）
- 类型：论文　　- 语言：英文　　- 层次：源码级（数学）
- **好在哪**：
  1. **它把"接受判据 + 残差分布"写成了一句可直接抄进代码的话**：原文 §2.3 "if $q(x)>p(x)$ we reject the sample with probability $1-\frac{p(x)}{q(x)}$ and sample $x$ again from an adjusted distribution $p'(x)=norm(max(0,p(x)-q(x)))$ instead." —— 注意它是**从"什么时候拒绝"讲起**而不是"什么时候接受"，这个叙述顺序比社区常见的 `min(1,p/q)` 更能让人看懂"为什么只在 q 高估时才需要拒"。
  2. **α 的定义是"两问式"的，而且给了闭式**：Definition 3.1 定义 $\beta_{x_{<t}}$ 为"给定前缀时接受一个 $x_t\sim q$ 的概率"，$\alpha = E(\beta)$；Corollary 3.6 给出 $\alpha = 1-E(D_{LK}(p,q)) = E(\min(p,q))$。**这一步是把"两个模型有多像"从玄学变成可计算量的关键**，本库第 05 篇应当直接采纳这条链路（β→α→TV 距离）。
  3. **它把"能不能赚"写成了一个可判定的不等式**：Corollary 3.9 —— "If $\alpha>c$, there exists $\gamma$ for which we'll get an improvement, and the improvement factor will be at least $\frac{1+\alpha}{1+c}$"，其中 $c$ 是草稿/目标的单步耗时比。**$\alpha > c$ 是全库第 19 篇（负收益全解）应该反复引用的那条线**。
  4. **它的动机段就是带宽墙**："inference from large models is often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication, so additional computation resources might be available"。第 01/03 篇可直接引。
  5. **它对速度提升的表述在正文里是精确的**："without changing the model output distribution"。
- **哪里不严谨或讲错了**：
  - **摘要与正文口径不一致，且摘要那句是全社区误解的源头**。摘要：`"an algorithm to sample from autoregressive models faster without any changes to the outputs"`、`"show a 2X-3X acceleration compared to the standard T5X implementation, with identical outputs"`。正文：`"without changing the model output distribution"`。**"changes to the outputs" 与 "changes to the output distribution" 差一个词，差的是 L2 与 L1。** 本库第 07 篇应把这两句并排贴出来。
  - $E[\tau]$ 公式的独立性假设**在原文是写明的**（"If we make the simplifying assumption that the $\beta$s are i.i.d., ... the number of tokens produced by a single run of Algorithm 1 is a **capped geometric variable**, with success probability $1-\alpha$ and cap $\gamma+1$"），但**论文没有做任何实证检验这个 i.i.d. 假设**。本库第 06 篇要接着往下做（相关性会怎样系统性地偏移预测）。
  - Theorem 3.8 的 walltime 模型 $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$ **只算了两个模型的串行耗时，没有 batch 维度**。原文自己也承认 "$c$ was always less than 0.05 and often negligibly close to 0" 是在他们那种"小两个数量级"的设置下测的。**把这个公式搬去解释 bs=64 的生产系统是越界使用**。
- 是否踩了 8 条：摘要踩 **E1**；正文与分析部分对 **E1/E8** 都是模范（◎）。
- 值得直接引用的一句话 / 一张表：
  > "if $q(x)>p(x)$ we reject the sample with probability $1-\frac{p(x)}{q(x)}$ and sample $x$ again from an adjusted distribution $p'(x)=norm(max(0,p(x)-q(x)))$ instead."
  以及 **§4.1 的实验表**（本次抓取到的行）：`CNNDM T5-small γ=5 α=0.53 → 2.3X`、`CNNDM T5-base γ=3 α=0.55 → 2.2X`、`CNNDM T5-large γ=3 α=0.56 → 1.7X`。
  **这三行是本库最有价值的一张表**：草稿从 small→base→large，$\alpha$ 单调上升（0.53→0.55→0.56），**加速比却单调下降（2.3X→2.2X→1.7X）**。它同时干掉两个直觉——"草稿越小越好"和"接受率越高越快"——真正的判据是 $\alpha$ 与 $c$ 的**联合**权衡（Corollary 3.9）。

---

#### Accelerating Large Language Model Decoding with Speculative Sampling

- URL：https://arxiv.org/abs/2302.01318 ；全文 https://ar5iv.labs.arxiv.org/html/2302.01318 （**均访问成功**）
- 作者/机构：Charlie Chen、Sebastian Borgeaud、Geoffrey Irving、Jean-Baptiste Lespiau、Laurent Sifre、John Jumper（DeepMind，正文首页署名 "from DeepMind"）　　- 年月：v1 **2023-02-02**
- 类型：论文　　- 语言：英文　　- 层次：源码级
- **好在哪**：
  1. **"within hardware numerics" 这个限定词是它发明的，而且是对的**。摘要："a novel modified rejection sampling scheme which **preserves the distribution of the target model within hardware numerics**"。**这是全社区唯一一个把浮点误差写进无损声明的原始出处**，vLLM 官方文档 2026 年那句 "theoretically lossless up to the precision limits of hardware numerics" 就是它的后代。本库铁律一的 L1 定义应当继承这个限定。
  2. **它明确指出了"至少产出 1 个 token""最多产出 K+1 个 token"这两条边界**：
     > "At least one token will always be generated from a draft-accept loop – if the first token is rejected, a valid token is resampled."
     > "Since the final token of the draft gives us the logits for the next token, if every drafted token is accepted, we can sample from it normally. This gives us a maximum of $K+1$ tokens per loop, over the naive implementation which would only return $K$ tokens."
     **第二条是很多人手写实现时漏掉的"bonus token"**（vLLM 源码里就叫 `bonus tokens`）。第 02 篇手算算例必须把这一条走一遍。
  3. **它把独立发现讲得很干净**："The work in this manuscript was undertaken concurrently and independently of the work on speculative decoding from Leviathan et al. 2022. We focus more heavily the distributed serving setting for large models and offer some incremental optimisations, but otherwise the core underlying idea is the same." —— 第 09 篇（两篇同期论文的异同）可以直接引这句作为定调。
  4. **一个很好的史料细节**：Supplementary 的 Author Contributions 明写 "**Modified Rejection Sampling Scheme: John Jumper**"。也就是说，投机采样那套修正拒绝采样是 AlphaFold 的第一作者写的。这条适合放在第 09 篇当花絮，但**必须注明出处是论文附录的贡献声明**。
- **哪里不严谨或讲错了**：
  - **记号与 Leviathan 相反**（$p$=草稿、$q$=目标），且论文里**没有任何一句提醒读者这个差异**，尽管它在正文里明确引用了 Leviathan。这是社区记号混乱的直接源头之一。
  - 加速比口径不全：摘要给 "2-2.5x decoding speedup in a distributed setup" —— **"distributed setup" 具体是多少芯片、什么 batch、$K$ 取几，摘要没写**。引用时必须标"口径不全"。
- 是否踩了 8 条：**一条都没踩**，且对 E1 是模范。
- 值得直接引用的一句话：
  > "a novel modified rejection sampling scheme which preserves the distribution of the target model **within hardware numerics**"

---

#### Looking back at speculative decoding（Google Research 官方回顾）

- URL：https://research.google/blog/looking-back-at-speculative-decoding/ （**访问成功**）
- 作者/机构：Yaniv Leviathan（Distinguished Engineer）、Matan Kalman（Software Engineer）、Yossi Matias（VP & Head, Google Research）　　- 年月：**2024-12-06**
- 类型：官方博客（作者本人写的回顾）　　- 语言：英文　　- 层次：入门
- **好在哪**：
  1. **它是"为什么是投机执行"这个类比的权威出处**："a well-known example of speculative execution is branch prediction in modern pipelined CPUs"。第 01 篇引类比时应该引这里，而不是引二手博客。
  2. **√7 的例子极其好用**：展示同一段生成里，"抄一个 7" 和 "算出 2.646" 的难度天差地别 —— 这就是"有些 token 是白送的"这个直觉的最短路径。**本库第 01 篇建议直接采纳这个例子并注明出处。**
  3. **对硬件动机的表述干净**："ample spare computational resources available when generating outputs from LLMs on modern hardware"。
  4. **它把确定性版本与随机版本拆成了两段来讲**，先讲经典投机执行（$f$、$f^*$、$g$），再讲"LLM 不是产出一个 token 而是一个分布"，然后引出 speculative sampling。**这个两段式结构值得本库第 04 篇借鉴。**
- **哪里不严谨或讲错了**：
  - **有一句话极容易被断章取义**："We are guaranteed identical outputs either way." 本次特意抓了它的上下文来判定：**它出现在"确定性投机执行"那一段**（前一句是 "If $f^*(X)$ output a different value, we can simply discard the computation of $g(f^*(X))$ and revert to calculating $g(Y)$ as in the serial case."），**不是在讲 LLM 采样**。到了 LLM 采样那一段，用词就变成了精确的："we are guaranteed that in spite of the lower cost, the generated samples come from **exactly the same probability distribution** as those produced by naïve decoding."
    **→ 结论：原文是对的，但这句话被大量二次引用时脱离了上下文。本库第 07 篇应当把"这句话原本在讲什么"讲清楚，作为"引用要看上下文"的范例。**
  - 加速比 "~2x–3x" + "11B T5-XXL 用 60M T5-small 做草稿，~3x" —— **没给 batch、没给 $\gamma$、没给硬件**。口径不全。
  - 完全没提 Medusa / EAGLE 的名字，只笼统说"后来有人用了多个草稿猜测、蒸馏、用目标模型的一部分做草稿"等。**做谱系时不能只靠这篇。**
- 是否踩了 8 条：**E1 上下文内正确（◎）**，但措辞给了误读空间。
- 值得直接引用的一句话：
  > "we are guaranteed that in spite of the lower cost, the generated samples come from exactly the same probability distribution as those produced by naïve decoding."

---

#### Wikipedia《Speculative decoding》

- URL：https://en.wikipedia.org/wiki/Speculative_decoding （**访问成功**）
- 作者/机构：Wikipedia 社区（无署名）　　- 年月：本次未取到具体修订日期，标「未查证」
- 类型：百科词条　　- 语言：英文　　- 层次：入门
- **好在哪**：
  - **史线给得很干净且日期对**：2018 Stern/Shazeer/Uszkoreit blockwise parallel decoding（并明确注明"worked only with greedy decoding, didn't preserve full sampling"）→ 2022-11 Leviathan/Kalman/Matias → 2023-02 Chen et al. → 2023-07 ICML oral。**第 08/09 篇的时间线可以拿它做交叉校验。**
  - 对验证机制的一句话概括抓住了要害："the first token that fails is resampled from a **corrected distribution**"（用了"corrected"这个词，方向是对的）。
- **哪里不严谨或讲错了**：
  - **明确踩 E1**："produces **the same results** as standard decoding while cutting latency by roughly two to three times"。同一篇里另一处又写 "The verification preserves the target model's original output distribution" 和 "the output distribution is the same as if each token had been generated one at a time"。**同一个词条里，"same results" 与 "same distribution" 混用**，这正是本库要点名的那个病。
  - **只字未提残差分布是什么**（"corrected distribution" 具体是 $\mathrm{norm}(\max(0,p-q))$ 这件事没写）。对一个技术词条来说，这是把整个算法的灵魂省掉了。
  - 加速比 "roughly two to three times" 完全裸奔。
- 是否踩了 8 条：**E1**；E3 属漏讲。
- 值得引用：**作为"高频错误的典型样本"引用**，不作为知识来源。第 27 篇可以直接拿这句 "produces the same results as standard decoding" 当反面靶子。

---

### B. 图解与直觉派

---

#### Accelerating Generative AI with PyTorch II: GPT, Fast（gpt-fast）

- URL：https://pytorch.org/blog/accelerating-generative-ai-2/ （**访问成功**）
- 作者/机构：Team PyTorch　　- 年月：页面显示 last updated **2024-11-14**（原发于 2023 年 11-12 月）
- 类型：官方工程博客　　- 语言：英文　　- 层次：进阶
- **好在哪**：
  1. **Verity / Drake 这个比喻是全网投机解码最好的比喻，没有之一。** 原文：资深工程师 Verity 决策对但写码慢，初级工程师 Drake 写得快但决策不一定对；Drake 先写，Verity 一次 review，"Verity might decide that the first 3 technical decisions Drake made are correct, but the last 2 need to be redone"。
     **它好在哪（具体）**：它一次性解决了三个理解障碍 ——
     (i) 为什么"验证比生成便宜"（review 比重写快）；
     (ii) 为什么被拒绝之后**后面的全部作废**而不是只废那一个（Drake 从第 4 个决策起全部重来，因为决策是链式依赖的）；
     (iii) 为什么最终质量不降（Verity 亲自把关）。
     普通的"小模型猜、大模型验"讲法只能覆盖 (i)。**本库第 02 篇建议直接采纳这个比喻并注明出处。**
  2. **它把动机放在 MBU（Model Bandwidth Utilization）的框架里**，前文已经算出 "72% MBU"，然后说 "In order to generate 100 tokens, we must load our weights 100 times" —— **投机解码是被逼出来的，因为量化已经把带宽账压到极限了**。这个叙事顺序（compile → int8 → 投机 → int4 → TP）是第 01/03/23 篇的好模板。
  3. **全文诚实标注了口径**："We will be focusing on latency (i.e. batch size=1) for all of these benchmarks... all benchmarks are run on an A100-80GB, power limited to 330W."
  4. **它给了两个对比鲜明的数字**：CodeLlama-34B + CodeLlama-7B → **2x**；Llama-7B + TinyLlama-1B → **约 1.3x**。并解释了原因："the runtime performance varies depending on the generated text, as well as how aligned the draft and verifier model are"。
- **哪里不严谨或讲错了**（这份材料是"文与码打架"的经典案例，非常适合当教学素材）：
  - **正文踩 E1**："one crucial property of speculative decoding is that **it does not change the quality of the output**"，以及 "**speculative decoding guarantees that we have mathematically identical results compared to regular generation**"。
  - **正文踩 E2**："we would generate 8 tokens using the draft model, and then process all eight tokens in parallel using the verifier model, **throwing out the ones that don't match**" —— "don't match" 是**贪心逐 token 比对**的语言，不是 $\min(1,p/q)$ 的语言。
  - **但它自己的代码是完全正确的**。本次直接抓 https://raw.githubusercontent.com/pytorch-labs/gpt-fast/main/generate.py 核对（记号为 DeepMind 系：`p`=draft、`q`=target）：
    ```python
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()
    ...
    new = q - p
    new = torch.where(new > 0, new, 0.0)
    new = new / new.sum()
    next_token = multinomial_sample_one_no_sync(new)
    ```
    **随机数有、min(1,·) 有、残差分布有。全对。**
  - **→ 这就是本库第 26 篇最值得写的一节：一份被引用了几万次的官方博客，比喻是满分，散文是错的，代码是对的。读者如果只读散文，会同时学会 E1 和 E2 两个错误。**
- 是否踩了 8 条：正文 **E1、E2**；代码 ○；对 E5 是模范（◎，全程标注 bs=1）。
- 值得直接引用：**Verity/Drake 那三段配图**（`image21.png` / `image6.png` / `image15.png`），以及那句诚实的口径声明 "We will be focusing on latency (i.e. batch size=1) for all of these benchmarks"。

---

#### Andrej Karpathy 关于 speculative execution 的推文

- URL：https://x.com/karpathy/status/1697318534555336961 —— **链接无法访问（2026-08 核验，HTTP 402）**
- 作者：Andrej Karpathy　　- 年月：2023-09（据引用它的第三方页面，未逐字核验）
- **状态说明**：这条推文在社区被反复称为"最清楚的投机解码解释"（Jim Fan 转发语："This is the clearest explanation I've ever seen on speculative decoding."，同样只见于搜索索引，**推文原文本次未能打开**）。
- **本库处理建议**：
  - **不要逐字引用**。若必须提，写成"据社区广泛转述（原推文 2026-08 核验时无法访问）"。
  - 从各处转述可见其核心是"forwarding an LLM on a single input token takes about as much time as forwarding an LLM on K input tokens in a batch"，**这是纯粹的贪心/匹配框架（"any tokens that agree with draft predictions"）**，即 **L2 口径**。如果本库要点评它，必须先拿到原文；否则只在第 30 篇附录里列一条"社区影响力最大但本次无法核验"的条目。

---

#### At the Intersection of LLMs and Kernels — Research Roundup（Charles Frye）

- URL：https://charlesfrye.github.io/programming/2023/11/10/llms-systems.html （**访问成功**）
- 作者/机构：Charles Frye　　- 年月：**2023-11-10**
- 类型：个人技术综述博客　　- 语言：英文　　- 层次：进阶
- **好在哪**：
  - **把"为什么并行验证几乎免费"讲成了一句可以背下来的话**："computing the logprobs for a prompt + K tokens can be done in parallel. That makes it much cheaper than sampling K tokens to follow a prompt, which must be done serially." —— **它强调的是"评分（scoring）"与"采样（sampling）"的代价差**，而不是笼统的"带宽 bound"，这个切法对第 03 篇（算术强度/roofline）很有用。
  - 分支预测类比给得克制且准确。
- **哪里不严谨或讲错了**：
  - **它没有把"并行评分几乎免费"量化**：没有算术强度、没有 roofline、没有说这个"几乎免费"在 K 多大时开始失效。本库第 03 篇必须补上这一层——这正是本库相对社区材料的增量。
  - 是综述性质，投机解码只是其中一小段，深度有限。
- 是否踩了 8 条：**未踩**（但对 E5 也未涉及）。
- 值得直接引用：上面那句 scoring vs sampling 的对比。

---

#### An Introduction to Speculative Decoding for Reducing Latency in AI Inference（NVIDIA）

- URL：https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/ （**访问成功**）
- 作者/机构：Jamie Li、Chenhan Yu、Hao Guo（NVIDIA）　　- 年月：**2025-09-17**
- 类型：厂商技术博客　　- 语言：英文　　- 层次：入门
- **好在哪**：
  - 比喻干净好记："Think of the target as the **meticulous scientist** ensuring correctness, while the draft is the **quick assistant** proposing possibilities that the scientist then verifies."
  - Figure 1 的走查很具体（草稿给出 "Brown / Fox / Hopped / Over"，目标接受前两个、拒绝其余、然后自己再产一个）——**它把"拒绝之后目标模型还会白送一个 token"画出来了**，这一点很多图解会漏。
  - 覆盖了 EAGLE-3 与 MTP 的对比，时效性好（2025-09）。
- **哪里不严谨或讲错了**（本篇是本次核到**错误最集中**的权威来源）：
  - **踩 E1 + E2，而且是同一句里踩两条**：
    > "**Only when a draft token matches what the target model would have generated, is it accepted** ... ensuring that the final output is **identical to what the target model would have produced**"
    这句话把 **L2 当成了全部**：它既取消了 $\min(1,p/q)$ 的随机性（E2），又把"分布相同"讲成了"输出相同"（E1）。
  - **踩 E5**："cutting latency and boosting throughput **without any impact on accuracy**"。全文**没有任何一句**讨论 batch size / 并发度对收益的影响 —— 而这恰恰是同一家公司自己的 TensorRT-LLM 文档里明写着 "speed ups are only observable at low batch sizes" 的事（见 §3.D）。
  - 另一处对判据的描述停在 "compares the proposed probability of the draft model, P(Draft), against the actual probability of the target model, P(Target)" —— **"compares" 是个含糊词，读者会自然理解成 `if p > q`**。
  - 完全没给 $\min(1,p/q)$，也完全没给残差分布。
  - 对草稿模型大小与接受率的权衡只字未提，只把 acceptance rate 当名词解释了一句。
- 是否踩了 8 条：**E1、E2、E5**。
- 值得直接引用：**"meticulous scientist / quick assistant" 这个比喻可以采纳**（比 Verity/Drake 更短，适合做一句话导入），但**必须紧跟一句纠正**："注意 NVIDIA 这篇同时把接受判据写成了确定性匹配，且宣称吞吐无损提升——两处都需要按本库铁律一、铁律二修正。"

---

### C. 数学派（本库第 04/05/06 篇的主要采纳对象）

---

#### Speculative Sampling（Jay Mody）

- URL：https://jaykmody.com/blog/speculative-sampling/ （**访问成功**）
- 作者/机构：Jay Mody（个人博客）　　- 年月：**2023-02-08**（代码于 2023-04-13 有一次修订，去掉了一次多余的草稿模型前向）
- 类型：个人博客 + 可跑 numpy/torch 实现　　- 语言：英文　　- 层次：进阶
- **好在哪**：
  1. **这是本次核到的、把 E1 讲得最透的一份材料，一句话解决问题**：
     > "Speculative sampling giving different result than autoregressive sampling is akin to running autoregressive sampling but **with a different seed**."
     **它好在哪（具体）**：把"分布相同但结果不同"翻译成了程序员每天都懂的东西——**换个随机种子**。读者不需要懂拒绝采样，就立刻明白"输出不一样"根本不是 bug。**本库第 07 篇建议把这句作为该篇的题眼直接采纳（注明出处）。** 它还补了例外情形：`temperature=0` 时才是逐 token 相同。
  2. **接受判据与残差分布都是可执行的代码，而且写法极简**：
     ```python
     if np.random.random() < min(1, q[i][j] / p[i][j]):   # accepted
     ...
     sample(max_fn(q[i] - p[i]))
     # max_fn: x_max = np.where(x > 0, x, 0); return x_max / np.sum(x_max)
     ```
     两行就把整个算法的灵魂交代了。**第 02 篇的手算算例可以照着这个骨架来。**
  3. **它给出了 `autoregressive_sampling()` 与 `speculative_sampling()` 的并排实现**，读者能直接跑对拍。
- **哪里不严谨或讲错了**：
  - **无损性它没有自己证，是甩链接给论文的**："The proof for this is shown in the paper (Theorem 1)"（指向 DeepMind 论文第 10 页）。**本库第 04 篇的完整证明正是要补这一块。**
  - **记号是 DeepMind 系（`p`=草稿、`q`=目标）**，但博客正文里并没有醒目地提醒读者这一点。读者若刚看完 Leviathan 论文再来看这段代码，会误以为分子分母写反了。**引用时必须补一句记号说明。**
  - 没有讨论 batch、没有讨论 $\gamma$ 怎么选、没有 $E[\tau]$。它是"讲清一个算法"，不是"讲清一笔账"。
- 是否踩了 8 条：**一条都没踩**，且对 **E1 是全网最好的纠正**。
- 值得直接引用的一句话：见上，"akin to running autoregressive sampling but with a different seed"。

---

#### An Optimal Lossy Variant of Speculative Decoding（Vivien Tran-Thien）

- URL：https://huggingface.co/blog/vivien/optimal-lossy-variant-of-speculative-decoding （**访问成功**）
- 作者/机构：Vivien Tran-Thien（HF community blog）　　- 年月：**2024-06-12**
- 类型：个人研究博客（带证明与 GitHub 代码）　　- 语言：英文　　- 层次：源码级（数学）
- **好在哪**：
  1. **它是本次核到唯一一份把"有损投机采样"当成一个最优化问题正面求解的材料**，而不是随手放宽阈值。目标：在 KL 散度预算 $D(q\|\pi)\le D$ 的约束下，最大化接受概率 $\sum_i p_i r_i$。
  2. **解的形式与标准投机采样长得几乎一样，但多了两个乘子**，非常有教学价值（记号为 DeepMind 系，$p$=草稿、$q$=目标、$\pi$=实际输出分布）：
     - 输出分布：$\pi_i = p_i r_i + s_i(1-\sum_j p_j r_j)$
     - 接受概率：$r_i = \min\left(1, \dfrac{q_i}{\alpha p_i}\right)$　　—— **令 $\alpha=1$ 就退回标准投机采样**
     - 残差：$s_i = \dfrac{1}{1-\sum_j p_j r_j}\max\left(0, \dfrac{q_i}{\beta}-p_i\right)$　　—— **令 $\beta=1$ 也退回标准**
     **它好在哪（具体）**：这套式子把"无损"从一个二元属性变成了**一条连续的曲线上的端点**。读者第一次能看清：Medusa 的 typical acceptance、各种阈值放宽，本质都是在这条曲线上往右挪，只是它们没算过自己挪了多远。**本库第 12 篇（Medusa）与第 07 篇（三种口径）应当把这套参数化直接采纳，作为 L1→L3 的连续谱。**
  3. **数学是真做完了的**：用了 KKT 条件、给了唯一解存在性、给了二分搜索算法及其单调性保证、链接了 GitHub 上的形式化证明，并在 WMT15 翻译上做了实验。
- **哪里不严谨或讲错了**：
  - **约束是 KL 而不是 TV**，而投机采样自然的度量是总变差（$\alpha=1-\mathrm{TV}$）。用 KL 做预算在优化上更好处理，但**和 $\alpha$ 的口径不在同一个度量下**，两者不能直接换算。本库引用时要标注这一点。
  - **它没有回答"KL 预算 $D$ 该取多少"这个工程问题**。给了旋钮，没给刻度。
  - 影响力有限（HF community blog，非官方博客位），社区几乎没有引用。**这恰恰是本库该采纳它的理由——它是被埋没的好材料。**
- 是否踩了 8 条：**一条都没踩**；对 **E6 是最好的解药**（把"有损"正面量化）。
- 值得直接引用：$r_i=\min(1, q_i/(\alpha p_i))$ 这一族公式，及"标准投机采样是 $\alpha=\beta=1$ 的特例"这个观察。

---

### D. 引擎与生产派（本库第 18/19/20 篇的主要采纳对象）

---

#### How Speculative Decoding Boosts vLLM Performance by up to 2.8x

- URL：https://vllm.ai/blog/2024-10-17-spec-decode （**访问成功**；旧地址 `blog.vllm.ai/2024/10/17/spec-decode.html` 会 301 到此）
- 作者/机构：vLLM Team　　- 年月：**2024-10-17**
- 类型：官方工程博客　　- 语言：英文　　- 层次：进阶
- **好在哪**：
  1. **它是少数把"反向数字"和"正向数字"并排放出来的官方材料**：
     - 正向：Llama3-70B + Qwama-0.5B 草稿，ShareGPT，4×H100，**QPS=1 → 1.5x**；n-gram，CNN/DailyMail，4×H100，**QPS=1 → 2.8x**
     - 反向：同样配置在**高 QPS 下 ShareGPT 慢 1.4x、CNN/DailyMail 慢 1.8x**
     **这组"同一套配置，低 QPS 赚、高 QPS 亏"的对照，是本库第 18/19 篇最直接的引用素材。**
  2. **它解释了为什么会亏，而且解释是对的**：
     > "However, in **high-QPS environments**, speculative decoding may introduce performance trade-offs. The extra compute required to propose and verify tokens can sometimes slow down the system **when it is already compute-bound**."
     —— 抓住了"系统已经算力 bound 时，投机拿不到免费算力"这个要害。
  3. Figure 8 的走查图很好用（草稿给 `["I","like","cooking","and","traveling"]`，第三个被拒并改成 "playing"）。
  4. 把 n-gram / prompt lookup 与 Medusa/EAGLE/MLPSpeculator 的定位分清楚了。
- **哪里不严谨或讲错了**：
  - **踩 E1（无口径的 lossless）**："speculative decoding accelerates generation without sacrificing accuracy, making it a **lossless** yet highly efficient method"。**没标是 L1 还是 L2**。
  - **完全没有接受判据的数学**（既无 $\min(1,p/q)$，也无残差分布），只有一句 "the reduction is less pronounced when the average token acceptance rate is high"。
  - **最重要的一条：这篇博客的数字被第三方公开质疑且未被回应。** 见下一条。
- 是否踩了 8 条：**E1**；对 **E5 是模范**（◎）。
- 值得直接引用：那段 high-QPS 的原话（见上）。

---

#### vLLM Issue #10318：博客数字不可复现

- URL：https://github.com/vllm-project/vllm/issues/10318 （**访问成功**）
- 作者：GitHub 用户 `yeonjoon-jung01`　　- 年月：2024-11 提出
- 类型：Issue 讨论　　- 语言：英文　　- 层次：进阶
- **好在哪**：
  - **这是本次调研里最有价值的"元材料"**：有人拿着官方博客的配置去复现，复现不出来，并把完整口径贴了出来。
  - 复现配置（口径齐全，可直接引用）：vLLM 0.6.3、**4×H100 PCIe**、Meta-Llama-3-70B-Instruct 目标（TP=4）、turboderp/Qwama-0.5B-Instruct 草稿（TP=1）、ShareGPT、**1 QPS**、**4 个投机 token**。
  - 结果：**最高只到 1.4x（且是在 batch size 1 时）**；batch size 256 时端到端延迟"exceeded 20 seconds"。
  - **结局**：issue 被打上 stale 标签，90 天无活动后以 "not planned" 关闭，**vLLM 团队未补充实验细节，也未回应复现差距**。
- **哪里不严谨或讲错了**：
  - 报告者与博客的口径**未必完全一致**（博客的 4xH100 未注明是 SXM 还是 PCIe，报告者用的是 PCIe；PCIe 版带宽显著更低）。所以严格说这是"口径不全导致无法复现"，不能直接判定博客造假。**本库引用时必须这样表述。**
- 是否踩了 8 条：不适用；它本身就是 **E5/铁律二的活教材**。
- 值得直接引用：**把"官方博客给 2.8x、独立复现只到 1.4x、口径差异未被澄清、issue 以 stale 关闭"这条完整链路写进第 19 篇**。这比任何"数字要标口径"的说教都有说服力。

---

#### vLLM 官方文档：Speculative Decoding（2026 版）

- URL：https://docs.vllm.ai/en/latest/features/speculative_decoding/ （**访问成功**；旧路径 `.../features/spec_decode.html` 会跳转到此）
- 作者/机构：vLLM 项目　　- 年月：持续更新，本次核验 **2026-08-22**
- 类型：官方文档　　- 语言：英文　　- 层次：源码级
- **好在哪**（**这是本次核到对"无损"表述最严谨的一份材料，没有之一**）：
  1. **它把无损声明拆成了三层，每层都有限定**：
     > "Speculative decoding sampling is **theoretically lossless up to the precision limits of hardware numerics**."
     > "vLLM's implementation of speculative decoding is **algorithmically validated** to be lossless."
     并紧跟浮点限定："**Floating-Point Precision**: Differences in hardware numerical precision may lead to slight discrepancies."
     **→ 这三句话正好对应本库铁律一想要的东西：理论口径（L1）、实现口径（有测试背书）、数值口径（浮点）。第 07 篇建议整段采纳。**
  2. **它写出了一条几乎没人提的坑**：
     > "**Changes in batch size may cause variations in logprobs and output probabilities.**"
     > "vLLM does not currently guarantee stable token log probabilities."
     **这条对"无损"的实践含义极大**：即使算法层面 L1 成立，**batch 组成变化会通过浮点归约顺序改变 logprobs**，从而改变实际采样结果。本库第 07 篇必须讲这一层——**"分布无损"在真实系统里还要再打一次折**。
  3. 方法清单（2026-08 快照）：EAGLE、MTP、Draft Model、**PARD（Parallel Draft Model）**、MLP、N-Gram、**Suffix Decoding**、Hidden State Extraction、**Dynamic Speculative Decoding**、**Adaptive Verification**。**注意：Medusa 已不在文档的当前方法清单中**（虽然代码里仍有 `medusa` 分支，见下一条）。
  4. 明确的版本边界："Speculative decoding with draft models is not supported in `vllm<=0.10.0`" —— **这一条直接说明 Aleksa Gordić 2025-08 那篇"vLLM V1 不支持独立草稿模型"的结论已经过期**。
- **哪里不严谨或讲错了**：
  - 没有给出接受判据的数学（这是文档，合理），需要配源码看。
  - "algorithmically validated to be lossless" 没有指向具体的测试文件，读者无法自查。
- 是否踩了 8 条：**一条都没踩**；对 **E1 是模范（◎）**。
- 值得直接引用：那三句无损声明 + "Changes in batch size may cause variations in logprobs"。

---

#### vLLM 源码：`vllm/v1/sample/rejection_sampler.py`

- URL：https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/sample/rejection_sampler.py （**访问成功**）
- 作者/机构：vLLM 项目　　- 年月：main 分支，核验于 2026-08-22
- 类型：源码　　- 层次：源码级
- **好在哪**：
  1. **docstring 直接把论文钉死**："The implementation **strictly follows** the algorithm described in https://arxiv.org/abs/2211.17192."
  2. **接受判据（Triton kernel `rejection_random_sample_kernel`）**：
     ```python
     accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
     ```
     —— 注意它**没有显式写 `min(1, ·)`**，因为当比值 ≥1 时该不等式自动成立，`min` 是多余的。**这是一个很好的"数学写法 vs 工程写法"对照点，第 04 篇可以用来讲清 `min` 的作用。**
  3. **残差分布（`sample_recovered_tokens_kernel`）用的是 Gumbel-max，而不是显式归一化**：
     ```python
     prob = tl.maximum(target_prob - draft_prob, 0.0)
     score = prob * inv_q
     ```
     然后取 `score` 在词表上的 argmax。
     **它好在哪（具体）**：这一步同时解决了两个工程问题——**不需要显式做 $\sum$ 归一化**（Gumbel-max 对未归一化权重成立），**也不需要为每个被拒位置单独跑一次多项式采样**。这正是本库"教学实现 vs 生产实现"那一节最好的对照素材：数学上的 $\mathrm{norm}(\max(0,p-q))$ 在生产里根本不会真的去 norm。
  4. **术语值得直接采纳**：源码把输出分成三类 —— **accepted tokens**（通过比值检验的草稿 token）、**recovered tokens**（拒绝后从残差分布补采的）、**bonus tokens**（全部接受时目标模型白送的那个）。**"bonus token" 这个词比中文社区常用的"额外 token"更精确，本库统一采用。**
- **哪里不严谨或讲错了**：
  - Gumbel-max 与显式归一化在**浮点上并不完全等价**（尤其是 `prob` 极小时），这与文档里那句 "up to the precision limits of hardware numerics" 是同一件事的两面。源码里未见对此的注释。
- 是否踩了 8 条：**一条都没踩**。
- 值得直接引用：docstring 那句 + accepted/recovered/bonus 三分法。

---

#### vLLM 源码：`vllm/config/speculative.py`（2026-08 快照，本次调研的最大收获之一）

- URL：https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/speculative.py （**访问成功**）
- 类型：源码/配置定义　　- 层次：源码级
- **好在哪**（这份文件里藏着四条社区材料里几乎找不到的硬事实）：
  1. **草稿采样默认是贪心，并且草稿概率被当成 one-hot**：
     ```python
     draft_sample_method: DraftSampleMethod = "greedy"
     """How the draft model samples tokens. 'greedy' always picks the argmax
     token, and the draft probabilities are treated as one-hot during rejection
     sampling. 'probabilistic' samples stochastically from the draft
     distribution and uses the full draft logits for the probability ratio test
     during rejection sampling. This comes at the cost of additional GPU memory
     usage."""
     ```
     **意义**：$q$ 取 one-hot 时，接受判据退化为 $u < \min(1, p(x)/1) = p(x)$，残差退化为 $\mathrm{norm}(\max(0, p - \mathbf{1}_x))$。**这仍然是 L1 无损的**（是标准算法的一个合法特例），但**接受率的数学完全不同**，且 `probabilistic` 模式要多花显存。社区讲解从来不提"生产系统默认根本不用草稿的完整分布"。**第 05 篇（α 的口径）和第 20 篇（引擎实现）必须写这一条。**
  2. **α 的口径在源码里被显式定义为"无条件/边际"概率，且必须单调不增**：
     ```python
     synthetic_acceptance_rates: list[float] | None = None
     """Per-position *unconditional* acceptance rates for synthetic rejection
     sampling. Position i's entry is the marginal probability that the first
     i+1 draft tokens are all accepted; the list must have length
     num_speculative_tokens, each entry in [0, 1], and be monotonically
     non-increasing."""
     ```
     **这是本次找到的、对"α 到底是什么"最硬的一条工程口径**：位置 $i$ 的接受率 = **前 $i+1$ 个草稿 token 全部被接受的边际概率**，而不是"给定前面都接受了、第 $i$ 个被接受的条件概率"。**这两者只有在 i.i.d. 假设下才互相换算。第 05 篇的"口径分歧"一节就写这个。**
  3. **vLLM 把"平均接受长度 → 每位置接受率"的换算，实现成了"最小方差调度"，而不是几何分布**：
     ```python
     @staticmethod
     def _acceptance_length_to_rates(length: float, n: int) -> list[float]:
         """Mean acceptance length to unconditional per-position rates, using
         the minimum-variance schedule."""
         num_drafts = length - 1  # expected number of accepted draft tokens
         num_full = int(num_drafts)
         return ([1.0] * num_full + [num_drafts - num_full] + [0.0] * (n - num_full - 1))[:n]
     ```
     **意义**：给定同一个平均接受长度，vLLM 选的是 `[1,1,...,frac,0,0,...]` 这种确定性调度，**而不是 $\alpha^i$ 的几何衰减**。这等于在工程上明确表态：**$E[\tau]=(1-\alpha^{\gamma+1})/(1-\alpha)$ 背后的 i.i.d. 几何模型不是唯一合理的模型**。**这是 E8 最有力的实证支撑，第 06 篇必须引。**
  4. **MTP 的推理期复用有明确告警**：
     ```python
     logger.warning(
         "Enabling num_speculative_tokens > 1 will run "
         "multiple times of forward on same MTP layer"
         ",which may result in lower acceptance rate")
     ```
     **意义**：MTP 头是按训练目标一次预测下一个 token 的；推理时想让它出 $\gamma>1$ 个草稿，只能把同一层反复跑，**接受率会掉**。**这正是 E7（训练目标 ≠ 推理草稿用法）在生产系统里的具体表现，第 14 篇必须引这条源码告警。**
  5. **拒绝采样方法已经有三种**：`RejectionSampleMethod = Literal["standard", "synthetic", "block"]`，其中 `block` 的注释写着 "**block verification (Sun et al.)**, which jointly verifies the draft tokens as a block instead of one at a time"。**"逐 token 验证"已经不是唯一方案了**，第 24/25 篇（前沿/未来）要跟这条线。
  6. **MTP 已经是新模型的标配**：`MTPModelTypes` 列了 **27 种**模型特化的 MTP 类型（deepseek_mtp、glm4_moe_mtp、qwen3_next_mtp、kimi_k3_mtp、minimax_m3_mtp、gemma4_mtp、ernie_mtp、longcat_flash_mtp……）。**这个数字本身就是第 14/25 篇的论据：2026 年，"自带草稿头"已经从论文技巧变成模型发布的默认配件。**
  7. 方法全集（2026-08）：`ngram`、`medusa`、`mlp_speculator`、`draft_model`、`suffix`、`custom_class`、`eagle`/`eagle3`/`extract_hidden_states`、`dflash`、MTP 全家、`ngram_gpu`、`dspark`。另有 `enable_adaptive_verification`（目前仅 `dspark` 支持）。
- **哪里不严谨或讲错了**：
  - 源码不是教材，`_acceptance_length_to_rates` 那句 "minimum-variance schedule" 没有给出推导或引用；本库要用它就得自己在 `_lab/` 里把"为什么这个调度方差最小"验证一遍。
  - **注意：本次在 main 分支中未找到 `disable_by_batch_size` 相关字段**（grep `disable_by_batch_size` 无命中）。这说明 V0 时代那个 `--speculative-disable-by-batch-size` 开关**在当前版本已不存在**，改由 "Dynamic Speculative Decoding" 承担。**中文材料里仍在教这个参数的（见 §3.G tbr8.org），已经过期。**
- 是否踩了 8 条：不适用（源码）；但它对 **E7、E8 提供了最硬的反证**。

---

#### Inside vLLM: Anatomy of a High-Throughput LLM Inference System（Aleksa Gordić）

- URL：https://www.aleksagordic.com/blog/vllm （**访问成功**；vLLM 官方博客转载版 https://vllm.ai/blog/2025-09-05-anatomy-of-vllm ）
- 作者/机构：Aleksa Gordić（个人）　　- 年月：**2025-08-29**（vLLM 博客转载 2025-09-05）
- 类型：长篇源码级博客　　- 语言：英文　　- 层次：源码级
- **好在哪**：
  1. **接受判据与残差分布，散文写得比大多数材料的公式还准**：
     > "If the large model's probability for the draft token ≥ the draft's probability, accept it. Otherwise, accept it with probability `p_large(token)/p_draft(token)`"
     > "If there was a rejection create a new rebalanced distribution at that position (`p_large - p_draft`, clamp min at 0, normalize to sum to 1)"
     **它好在哪（具体）**：它把判据拆成了"两种情形"来讲（$p\ge q$ 无条件接受 / 否则按比值概率接受），**而不是直接甩一个 $\min$**。对第一次学的人，两情形写法比 $\min$ 写法更容易看出"$\min$ 在防止什么"。第 04 篇可以先用两情形式再收成 $\min$。
  2. **把投机解码放在整个引擎的语境里讲**（scheduler、KV cache、chunked prefill 之后才讲它），读者能看清它在系统里的位置，而不是当成孤立技巧。
- **哪里不严谨或讲错了**：
  - **"in expectation" 这个词用错了层次**：
     > "the accept/reject rule guarantees that **in expectation** the sequence is distributed exactly as if we had sampled token by token from the large model."
     无损性是**逐点的分布相等**（对每个 $x$，$P(\text{输出}=x)=p(x)$），**不是"在期望意义下成立"**。"in expectation ... distributed exactly" 这个搭配自相矛盾。**本库第 07 篇可以拿它当一个"措辞精度"的小案例：意思对，词用错了，而这个词恰好会诱导读者以为无损是统计平均意义上的近似。**
  - **有一条结论已经过期**："vLLM V1 does **not** support the LLM draft model method, instead it implements faster—but less accurate—proposal schemes: n-gram, EAGLE, and Medusa."
    **对照 2026-08 的 vLLM 官方文档与源码：`draft_model` 已是一等方法**（文档明写 "not supported in `vllm<=0.10.0`"，即 0.10.0 之后已支持），且 `medusa` 反而从文档的方法清单里退了下去。**引用这篇时必须标注"结论截至 2025-08，2026 年已变"。**
  - 不讨论大 batch 下的负收益。
- 是否踩了 8 条：**未明确踩坑**，但 E1 的措辞不精确。
- 值得直接引用：那段"两情形式"的接受判据描述。

---

#### A Hitchhiker's Guide to Speculative Decoding（PyTorch / IBM）

- URL：https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/ （**访问成功**）
- 作者/机构：IBM Research + Team PyTorch（致谢段点名：speculator 架构与训练 Davis Wertheimer、Pavithra Ranganathan、Sahil Suneja；paged attention 集成 Josh Rosenkranz、Antoni Viros i Martin；推理服务集成 Thomas Parnell、Nick Hill、Prashant Gupta）　　- 年月：页面 last updated **2024-11-13**
- 类型：官方工程博客（生产部署报告）　　- 语言：英文　　- 层次：进阶/生产
- **好在哪**（**这是本次核到"上生产"这一档最好的一份**）：
  1. **它给了一条本库最想要的、可判定的失效条件**：
     > "We begin to observe **throughput reduction beyond a batch size of 64**, which happens rarely in practice."
     **可判定、有数、有前提。铁律三要的就是这种句子。**
  2. **它把"多几个头"的代价讲成了一笔账**：
     > "If the speculator is not accurate with more heads, it will result in **wasted compute increasing the latency and reducing the throughput**."
     并给了经验值："we find **3-4 heads works well** in practice, whereas we found that **code models can reap benefits from 6-8 heads**"。
     **它好在哪（具体）**：它把 $\gamma$ 的选取和**任务类型**绑定（代码任务可预测性高 → 能吃更长的草稿），而不是给一个万能数字。第 17 篇（动态草稿长度）应该从这条经验出发。
  3. **它是真的生产数据**："We have deployed these speculators in an internal production-grade environment with **thousands of daily users**"，2x（Llama3 8B / Llama2 13B / Granite 7B）、3x（Granite 20B code）。
  4. **指标选得对**：报 **TTFT 与 ITL 随并发用户数变化的两张图**，而不是笼统的"tokens/s"。**这是本库铁律二第 6 项（"测的是什么"）的正面范例。**
  5. **它讲了草稿头怎么训**，而且给了配比："两阶段训练——阶段一小 batch 长序列（4k）走标准 causal LM；阶段二大 batch 短序列（256）用基座模型生成的数据，把头对齐到基座输出；**5:2 的步数配比**"。**第 21 篇（草稿模型怎么训）可以直接采纳这套配方。**
  6. 架构上把 Medusa 改成了**层级式多阶段头**（每个头预测一个 token 再喂给下一个头），并说明理由。
- **哪里不严谨或讲错了**：
  - **踩 E1**，而且是开篇第一段就踩：
     > "It incorporates a verification mechanism to ensure the correctness of these speculated tokens, thereby **guaranteeing that the overall output of speculative decoding is identical to that of vanilla decoding**."
     以及后文 "enable speculative decoding **without deviating from the original model's output**"。
     **需要说明的是：他们的实现本身很可能确实是 L2（贪心逐 token 匹配验证），所以这句话在他们自己的系统里未必错——但它是以"投机解码的普遍性质"的口吻写出来的**，读者会把它当成通用结论。
  - "which happens rarely in practice"（bs>64 很少见）这句在 2024 年的 IBM 内部负载下可能成立，**在 2026 年的高并发推理服务里已经不成立**。引用时要加时效标注。
  - 没有任何接受判据的数学。
- 是否踩了 8 条：**E1**；对 **E5 是模范（◎）**，对 E4 也有正面处理。
- 值得直接引用：**"We begin to observe throughput reduction beyond a batch size of 64"** 与 **3-4 heads / 6-8 heads（代码任务）** 这两条。

---

#### TensorRT-LLM 官方文档：Speculative Decoding

- URL：https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/speculative-decoding.html （**访问成功**）
- 作者/机构：NVIDIA　　- 年月：文档版本 1.2.0rc6，核验于 2026-08
- 类型：官方文档　　- 语言：英文　　- 层次：源码级
- **好在哪**（**它是本次核到"最诚实地承认自己是 L2"的一份材料**）：
  1. **它直接说了自己只做贪心**：
     > "A draft token is accepted **if matches the previously decoded token exactly**."
     > "**only greedy sampling is supported for speculative decoding**"
     **意义极大**：NVIDIA 自家的 intro 博客（§3.B）说"输出与目标模型一致"，读者以为是通用结论；而 NVIDIA 自家的文档说"因为我们只支持贪心"。**把这两份并排贴出来，就是本库第 07 篇最有力的一张对照。**
  2. **它给了一条可判定的工程失效条件**：
     > "There is currently **no way to dynamically disable speculation**, thus **speed ups are only observable at low batch sizes**."
     **这句话应当被本库第 19 篇原样引用。** 它同时说明了"为什么会亏"（不能动态关）和"什么时候亏"（大 batch）。
  3. **one-model vs two-model 的取舍讲得清楚**：one-model 更快（"launches the entire drafting loop as a single CUDA graph"）但不支持动态草稿长度；two-model 更灵活但**"do not support overlap scheduler. It will be disabled automatically."** —— **"开了投机就关掉 overlap scheduler"这类隐性代价，是引擎文档才会告诉你的东西**，第 20 篇必须收录。
  4. 配置示例直白可抄：`DraftTargetDecodingConfig(max_draft_len=3, speculative_model=...)`、`EagleDecodingConfig(max_draft_len=3, ..., eagle3_one_model=False)`。
  5. **MTP 的参数暴露了一个有损开关**：`use_relaxed_acceptance_for_thinking`、`relaxed_topk`、`relaxed_delta`。
- **哪里不严谨或讲错了**：
  - **MTP 的 "relaxed acceptance" 参数没有任何一句提示它会改变输出分布**。这是 **E6 的同型错误**（放宽接受判据 = L3 有损），只不过主角从 Medusa 换成了 MTP。**本库第 14 篇要点名这一条：`use_relaxed_acceptance_for_thinking` 是有损开关，文档未标注。**
  - 方法清单里没有 Medusa/ReDrafter/Lookahead（而 NVIDIA 自己发过 ReDrafter 的支持），文档与生态的覆盖不一致。
- 是否踩了 8 条：**E7 相关的有损开关未标注（近似 E6）**；对 **E1、E5 都是模范（◎）**。
- 值得直接引用：那两句（"only greedy sampling is supported" 与 "speed ups are only observable at low batch sizes"）。

---

#### SGLang 官方文档：Speculative Decoding

- URL：https://docs.sglang.io/advanced_features/speculative_decoding.html （**访问成功**；`docs.sglang.ai` 会 301 到此）
- 作者/机构：SGLang 项目　　- 年月：持续更新，核验于 2026-08
- 类型：官方文档　　- 层次：进阶
- **好在哪**：
  1. **三个参数的语义被一句话各自钉死，这是本库第 16 篇（树形草稿）最需要的**：
     - `--speculative-num-steps`：**"Depth of autoregressive drafting"**（树的深度）
     - `--speculative-eagle-topk`：**"Branching factor per step"**（每步的分支数）
     - `--speculative-num-draft-tokens`：**"Maximum parallel verification capacity"**（一次并行验证的 token 预算）
     **它好在哪（具体）**：社区讲树形草稿时常常只说"构造一棵树"，这三个参数把树的**深度、宽度、总预算**分成三个独立旋钮，读者立刻明白树不是一个东西而是一个三维配置空间。
  2. **给了随模型族变化的默认值**：Llama/Grok 默认 `steps=5, topk=4, draft-tokens=8`，其他模型 `steps=3, topk=1, draft-tokens=4`。**"默认值按模型族分叉"这件事本身说明接受率高度依赖模型族**，可作为 E4 的侧证。
  3. **写出了隐性代价**："overlap scheduler（V2 默认）要求显式 `--speculative-eagle-topk 1`"，`topk>1` 会**关掉 overlap scheduler**；ngram 同样会"disables the overlap scheduler & mixed chunked prefill"。**和 TensorRT-LLM 那条是同一类坑，交叉印证。**
  4. 方法覆盖广：EAGLE-2、EAGLE-3、MTP、**DFLASH**（linear block verification）、STANDALONE、NGRAM（CUDA-only）。
- **哪里不严谨或讲错了**：
  - **数字口径不全，且用词误导（近似 E5）**：文档给 LLaMA-Instruct 3.1 8B / MT-bench / 1×H100：baseline **158.34 tokens/s**、EAGLE-2 **244.10**、EAGLE-3 **373.25**。**没有标 batch size / 并发数**，而 MT-bench 的标准跑法是 batch=1。把 tok/s 的提升口头描述成 "throughput gains" 会让读者以为是服务吞吐。**本库引用这组数字时必须补标"（未标注 batch，按 MT-bench 惯例应为 bs=1；口径不全）"。**
  - **完全没有关于 temperature / 采样是否支持、是否无损的任何说明**。示例全用 `temperature=0`。读者无从判断 SGLang 的 EAGLE 在 $T>0$ 下是 L1 还是 L2。**这是本库第 20 篇需要自己去读源码补的坑。**
- 是否踩了 8 条：**E5（口径不全 + 用词）**；E1 属于未表态。
- 值得直接引用：三个参数的那三句定义。

---

#### SpecForge: Accelerating Speculative Decoding Training for SGLang（LMSYS）

- URL：https://www.lmsys.org/blog/2025-07-25-spec-forge/ （**访问成功**）
- 作者/机构：The SGLang Team（LMSYS）　　- 年月：**2025-07-25**
- 类型：官方工程博客　　- 层次：进阶
- **好在哪**：
  1. **它把"投机解码的真正瓶颈"从推理挪到了训练**：
     > "the lack of robust open-source tools for **training draft models**—a key component of this process—has significantly hindered its adoption."
     **这是第 21 篇的立论句。** 投机解码的算法早就公开了，真正卡住落地的是"没人给你训好的草稿头"。
  2. **在线/离线两种训练模式的取舍给了硬数字**：online 边训边生成 hidden states，省磁盘但要更多 GPU；offline 预计算 hidden states，**"~12TB for UltraChat + ShareGPT"**，但可以少到 1 张 GPU。**"12TB"这个数字很有教学价值——它让人具体感受到"草稿模型训练的成本主要是存 hidden states"。**
  3. 把 EAGLE-3 的 TTT（Training-Time Test）讲清了目的："makes the draft model robust by **simulating multi-step generation**"，并诚实说明实现难点："specialized attention masks and recursive data loops"。**第 13 篇讲 EAGLE-3 时可以引这句作为"为什么 TTT 是工程上的硬骨头"。**
- **哪里不严谨或讲错了**：
  - **口径严重不全**：Llama 4 Scout 2.0×、Maverick 2.18×（MT-Bench），**未给硬件、未给 batch、未给接受长度**，只给了 `speculative-eagle-topk=8`、`speculative-num-draft-tokens=10`。按铁律二，这组数字**不可与任何其他来源横向比较**。
  - 没有任何关于无损性的表述。
- 是否踩了 8 条：**未踩**（但口径不全）。
- 值得直接引用："the lack of robust open-source tools for training draft models ... has significantly hindered its adoption" + 12TB 那个数字。

---

#### TensorRT-LLM Speculative Decoding Boosts Inference Throughput by up to 3.6x（NVIDIA）

- URL：https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/ （**访问成功**）
- 作者/机构：Carl (Izzy) Putterman、Lalit Vaidya、Anjali Shah、Sharan Chetlur、Laikh Tewari（NVIDIA）　　- 年月：**2024-12-02**
- 类型：厂商工程博客　　- 层次：进阶
- **好在哪**：
  1. **它无意中给出了"草稿越小越好"最直接的反证数据。** Llama 3.1 405B 目标、4×H200、FP8、`max_batch_size=32`：
     | 草稿模型 | 输出 tokens/s | 加速比 |
     |---|---|---|
     | 无草稿（baseline） | 33.46 | 1.00× |
     | Llama 3.2 **1B** | 111.34 | 3.33× |
     | Llama 3.2 **3B** | **120.75** | **3.61×** |
     | Llama 3.1 **8B** | 101.86 | 3.04× |
     **1B 不是最优，3B 才是；再大到 8B 就掉下去了。** 这是一条清清楚楚的**倒 U 形曲线**，本库第 05/21 篇应当直接引用。
     70B 目标（1×H200）上则是另一个形状：1B 2.86× > 3B 2.75× > 8B 2.23× —— **最优草稿尺寸随目标模型尺寸变化**，这比"倒 U 形"本身更有信息量。
  2. 口径给了大半：4×H200 / 1×H200、FP8、`max_batch_size=32`、`max_num_tokens=8192`、`max_seq_len=131072`，指标定义也写了（"Output tokens/second" 且含 TTFT）。
- **哪里不严谨或讲错了**：
  - **标题踩 E5，正文没有救回来。** 标题说 "Boosts Inference **Throughput** by up to 3.6x"，但报的指标是**单请求的 output tokens/second**（含首 token 时间），`max_batch_size=32` 只是上限，**全文没有任何并发度扫描**。**按本库铁律二第 7 项，这是"把延迟指标叫成吞吐"的典型。**
  - **完全没有失效条件**：没有一句话说明在什么并发下这 3.6× 会消失——尽管同公司的 TensorRT-LLM 文档明写 "speed ups are only observable at low batch sizes"。
  - 没有 $\gamma$（`max_draft_len`）的取值。
  - 对无损性未表态。
- 是否踩了 8 条：**E5**；对 **E4 提供了最好的反证数据（◎）**。
- 值得直接引用：**那张 405B 的四行草稿模型对比表**（并注明"NVIDIA 称其为 throughput，实为单请求 output tok/s，口径不全"）。

---

#### Speculative Decoding - Deep Dive（AMD ROCm Blogs）

- URL：https://rocm.blogs.amd.com/software-tools-optimization/speculative-decoding---deep-dive/README.html （**访问成功**）
- 作者/机构：Chang Liu（AMD）　　- 年月：**2025-03-24**
- 类型：厂商工程博客　　- 层次：进阶
- **好在哪**：
  - **口径在本次核到的厂商博客里算齐的**：MI300X、ROCm 6.3.1、vLLM v0.6.7、`--num_speculative_tokens 5`、`max-num-seqs 300`。
  - **它做了 request rate 扫描**，给出了衰减曲线：**2.21× @ 0.2 req/s → 2.06× @ 1.4 req/s**，并明说 "the speedup ratio declines slightly along with increasing request rates"。**这是"收益随负载衰减"的一条实测曲线，第 18 篇可用。**
  - 长上下文的数字很有价值：Llama 3.1-70B，**输入 32768 / 输出 128，单 MI300X，2.98×** —— **长 prompt + 短输出的场景收益最高**，这与"prefill 占比大时投机的相对收益"有关，第 22 篇（长上下文）可作为切入点。
  - 其他：70B + 3.2-1B 草稿，单 MI300X → **2.31×**；405B + 3.2-1B，4×MI300X → **2.13×**。
- **哪里不严谨或讲错了**：
  - **标题叫 "Deep Dive"，但完全没有算法深度**：没有 $\min(1,p/q)$、没有残差分布、没有 $E[\tau]$、没有独立性讨论。它是一篇 benchmark 报告，不是深潜。**本库引用它只作为数字来源，不作为原理来源。**
  - **batch size 大部分场景未标**（只有一条命令里能看到 `max-num-seqs 300`，那是上限不是实际并发）。
  - 只说"收益随请求率下降"，**没解释机制**，也没给拐点。
- 是否踩了 8 条：**未踩**（因为几乎不涉及算法表述）。
- 值得直接引用：request rate 衰减那两个点，以及 32768/128 长上下文的 2.98×。

---

#### Fastest Speculative Decoding in vLLM with Arctic Inference and Arctic Training（Snowflake）

- URL：https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/ （**访问成功**）
- 作者/机构：Ye Wang、Gabriele Oliaro、Jaeseong Lee、Yuxiong He、Aurick Qiao、Samyam Rajbhandari（Snowflake）　　- 年月：**2025-05-01**
- 类型：厂商工程博客　　- 层次：进阶/生产
- **好在哪**（**这份材料贡献了本次调研最锋利的一句引文**）：
  1. **它把 L1 与 L2 的取舍变成了一次真实的工程决策，并写在了博客里**：
     > "**Switched from rejection sampling to greedy verification** (accept tokens only if they match greedy decoding), ensuring **outputs are identical to the base model without lowering acceptance rate**."
     **它好在哪（具体）**：社区讲 L1 与 L2 时永远停在"理论上不一样"。这句话展示了**一个生产团队为什么会主动放弃 L1**：他们要的是"和不开投机时逐 token 一样"的可复现性（对 agent / 代码场景尤其重要），而 L1 给不了这个。**本库第 07 篇建议把这句作为"L2 不是退化，是另一种需求"的证据。**
  2. **suffix decoding 的成本被量化了**："**20 microseconds per token**" 的 CPU 端投机 —— **草稿可以便宜到不用 GPU**。第 11 篇（无模型草稿）必须引这个数字。
  3. **它把"草稿质量"量化成了接受率的倍数**："**3.1x higher acceptance rate**" vs 开源基线（LSTM/MLP speculator）。
  4. **它给了一个"agent 场景收益最高"的论断并配了数字**：SWE-Bench 上跑 CodeAct agent（OpenHands LM 32B），端到端 **1.8x–4.5x**。**理由是 agent 轨迹里有大量重复文本**，正好是 suffix decoding 的主场。第 24/25 篇可以顺着这条讲"投机解码在 agent 时代反而更重要"。
  5. 主要数字的口径较齐：Llama-3.1-70B-Instruct、**8×H100、TP=2、FP8**、0.5 req/s；ShareGPT 179 tok/s、HumanEval 217 tok/s、混合 209 tok/s。
- **哪里不严谨或讲错了**：
  - **SWE-Bench 那条 1.8x–4.5x 没标硬件**，口径不全。
  - **没有并发扫描**，也没说明在高并发下 suffix decoding 的 CPU 开销会不会成为瓶颈。
  - "without lowering acceptance rate" 这句需要谨慎读：**贪心验证在数学上不可能比拒绝采样有更高的接受率**（拒绝采样的接受概率 $\ge$ 贪心匹配概率）。他们大概是指在他们的草稿模型上实测差异可忽略，**但博客把它写成了一般性断言**。本库引用时要标注这一点。
- 是否踩了 8 条：**未踩，且对 E1 是最好的工程注脚（◎）**。
- 值得直接引用：那句 "Switched from rejection sampling to greedy verification..."（全文最值得引的一句）。

---

#### Performance improvements with speculative decoding in vLLM for gpt-oss（Red Hat）

- URL：https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss （**访问成功**）
- 作者/机构：Harshith Umesh（Red Hat）　　- 年月：**2026-04-16**
- 类型：厂商工程博客　　- 层次：进阶/生产
- **好在哪**（**这是本次调研里对本库立场冲击最大的一份材料，必须认真对待**）：
  1. **它给出了"高并发下投机解码依然赚"的反例，而且口径极全**：
     - 目标：`openai/gpt-oss-120b`（**MoE，MXFP4 量化**）；草稿：`nvidia/gpt-oss-120b-Eagle3-v2`（EAGLE3）
     - 硬件：**H200-PCIe-141GB**，TP=1 与 TP=2
     - 并发：**1 / 5 / 25 / 50 / 100 / 200**
     - 数据集：ShareGPT（平均输入 122 token）、MLPerf（5011）、SWE-bench（556）
     - 工具：GuideLLM v0.5.3、vLLM v0.13.0
     - 结果：ShareGPT 输出吞吐峰值 **2574 tok/s vs 基线 2024 tok/s（+27.2%）**，几何平均 **+20.7%**，且 **"consistent throughput and latency improvements that persist with up to 200 concurrent requests"**
     **→ 本库铁律三"投机是笔经常亏的交易"必须加一条限定：当目标模型是 MoE（激活参数远小于总参数、算力空转更严重）且草稿是高接受率的 EAGLE3 时，2026 年的证据表明高并发下依然赚。**"大 batch 一定亏"这个说法本身要被修正成一条**依赖 (激活/总参数比、量化精度、草稿接受率) 的判据**。第 18/19 篇必须把这份材料写进去，否则本库的结论会落后于 2026 年的事实。
  2. **给了 $\gamma$ 的可判定甜点区与代价**：
     > "For this model, **2 or 3 draft tokens is the sweet spot**"；"going from 3 to 4 draft tokens causes a modest **8% output throughput drop**"
     **"多一个草稿 token 掉 8% 吞吐"是本库能找到的最具体的一条 $\gamma$ 边际成本数据。**
  3. 三个数据集分开报（ShareGPT decode-heavy、MLPerf 长 prompt、SWE-bench 代码），并且不同数据集收益差异明显（+20.7% / +9.5% / +20.5%）——**任务分布对收益的影响被量化了**，正合铁律二第 7 项。
- **哪里不严谨或讲错了**：
  - **踩 E1**："no change to the model's output quality"、"no change to model weights or output quality"，**没有标口径**（EAGLE3 在 $T>0$ 下的分布保证与 $T=0$ 的逐 token 一致是两回事）。
  - **它是厂商博客，结论对自家栈有利**；且 MoE + MXFP4 是一个**对投机极其有利**的组合（权重加载代价大、激活算力少），**不能推广到稠密 fp16 模型**。本库引用时必须把"为什么这个组合特别适合投机"讲清楚，否则读者会得出"高并发也总是赚"的相反错误。
- 是否踩了 8 条：**E1**；对 **E5 提供了关键的反向证据（◎）**。
- 值得直接引用：并发 200 仍有提升那句 + "3 到 4 个草稿 token 掉 8% 吞吐"。

---

#### LLM Inference Handbook — Speculative decoding

- URL：https://handbook.modular.com/inference-optimization/speculative-decoding （**访问成功**；原 `bentoml.com/llm/inference-optimization/speculative-decoding` 301 至此）
- 作者/机构：原 BentoML，现挂在 Modular　　- 年月：页面未显示日期，标「未查证」
- 类型：手册式教程　　- 层次：入门/进阶
- **好在哪**：
  - **它给了一条并发拐点的实测描述**：
    > "With TP = 1, the total throughput **plateaued earlier (around 20–30 concurrent requests)** compared to the baseline...indicating that the coordination between the draft and target models might bring overhead at higher loads."
    > "Adding parallelism (TP = 2) improves throughput, but you need to **tune γ to avoid latency spikes at high load**."
    **"20–30 并发处提前饱和"是一个可判定的数字，第 18 篇可用。**
  - **对草稿模型的建议是对的方向**："How closely your draft model's distribution matches with the target model determines the acceptance rate...you'll likely get better results by **fine-tuning a draft model on your data**"，并配一句 "**Don't ignore wasted compute**"。
  - 方法覆盖齐（Medusa、MTP+DeepSeek-V3、n-gram、EAGLE 三代），适合当索引。
- **哪里不严谨或讲错了**：
  - **踩 E1，而且措辞是最强的那种**：
    > "This draft-then-verify pattern **guarantees the final output matches exactly what the original target model would have produced on its own**. Therefore, it does not sacrifice output quality."
  - **完全没有 $\min(1,p/q)$ 与残差分布**，整篇没有一个公式。这对一本自称 handbook 的材料是硬伤。
  - 引用的 Medusa 2.2–3.6× 与 EAGLE 3.0–6.5× **都是论文里的最大值，口径不全**，且并排放在一起做了隐性比较（违反铁律二"不许把不同来源的加速比放进同一张表"）。
- 是否踩了 8 条：**E1**；对 **E4/E5 有正面处理（◎）**。
- 值得直接引用：20–30 并发处提前饱和那句。

---

### E. 方法专篇

---

#### Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads（Together.ai 博客）

- URL：https://www.together.ai/blog/medusa （**访问成功**）
- 作者/机构：Tianle Cai、Yuhong Li、Zhengyang Geng、Hongwu Peng、Tri Dao（Cai 与 Li 共同一作）　　- 年月：**2023-09-11**
- 类型：项目博客　　- 层次：进阶
- **好在哪**：
  1. **它自己就把"Medusa 是有损的"说清楚了**，这一点必须替它正名：
     > "**Relaxing the requirement of matching the distribution of the original model** makes the non-greedy generation even faster than greedy decoding."
     **原文明确承认放宽了分布匹配要求。E6 这个锅不该扣在 Medusa 作者头上，该扣在下游转述者头上。** 本库第 12 篇要写清这个传播链。
  2. **它给出了"为什么多头能行"的关键数字**：第二个头预测 next-next token 的 **top-1 准确率约 60%，但 top-5 超过 80%**。
     **它好在哪（具体）**：这一对数字直接解释了**为什么必须有树**——单条链只能用 top-1（60% 太低），而树能同时验证 top-5（80% 就够用了）。**第 12 篇讲"树注意力为什么诞生"，这组数字是最好的动机。**
  3. **树的构造讲得直观**：各头 top-k 预测的**笛卡尔积**构成候选，注意力 mask 限制每个 token 只能看到自己的祖先，"enabling parallel processing of multiple candidates **without increasing batch size**"。**最后半句是要害：树是用 mask 换 batch，第 16 篇要抓住这一点。**
  4. 口径基本齐：**单张 A100-80G，batch size 1**，ShareGPT 训练一个 epoch，33B 用 8-bit 量化；Vicuna 7B/13B/33B 上约 2× 墙钟加速；"a 33B model with Medusa operates as fast as an unaccelerated 13B model"（这个类比很好懂）。
- **哪里不严谨或讲错了**：
  - **typical acceptance 的机制描述过于简略**："We set a threshold based on the original model's prediction probabilities, and if a candidate exceeds this, it's accepted." —— **没给阈值的具体形式**（论文里是 $\max(\epsilon, \delta\exp(-H(p)))$ 这类形式，与熵挂钩），读者无法判断偏离有多大。
  - **"更快"的代价没有量化**：说了放宽分布会更快，**但没有任何一个指标衡量放宽了多少**（既没有 TV 距离，也没有下游任务分数对照）。**这正是 Tran-Thien 那篇（§3.C）该被拿来配对阅读的原因。**
  - 2× 的加速没标 $\gamma$/树的规模。
- 是否踩了 8 条：**E6 上是模范（◎）**。
- 值得直接引用："Relaxing the requirement of matching the distribution of the original model..." 这句，以及 60%/80% 那对数字。

---

#### Medusa GitHub（FasterDecoding/Medusa）

- URL：https://github.com/FasterDecoding/Medusa （**访问成功**）
- 作者/机构：FasterDecoding；作者列表 Tianle Cai、Yuhong Li、Zhengyang Geng、Hongwu Peng、Jason D. Lee、Deming Chen、Tri Dao　　- 年月：README 本次未取到明确 last update，标「未查证」；**star 数 2.8k**
- 类型：代码仓库　　- 层次：源码级
- **好在哪**：
  - Medusa-1 / Medusa-2 的区分写得清楚：Medusa-1 冻结基座只训头；Medusa-2 支持全模型训练，用特殊配方"adds the speculative prediction ability while keeping the original model's performance"，并支持自蒸馏（不需要原始训练数据就能给已微调模型加头）。**第 12/21 篇需要这个区分。**
  - 报了 2.2–3.6× 的更新数字，并明确 "single-GPU inference, **batch size of 1**"。
- **哪里不严谨或讲错了**：
  - **踩 E6（漏讲型）**：README 只写 "a **typical acceptance scheme** is employed to pick the longest plausible prefix from the candidates"，**没有一个字提示这会改变输出分布**。相比之下 Together.ai 博客反而说清楚了。
    **→ 这正是 E6 传播链的断点：论文/博客说清楚了，仓库 README 没说，绝大多数人只读 README。第 12 篇要把这条链画出来。**
  - 本次抓取未见 `temperature` / `posterior_threshold` / `posterior_alpha` 这几个关键超参的文档说明（它们在代码里存在），**即"控制有损程度的旋钮没有被文档化"**。
- 是否踩了 8 条：**E6**。

---

#### EAGLE 官方仓库（SafeAILab/EAGLE）

- URL：https://github.com/SafeAILab/EAGLE （**访问成功**）
- 作者/机构：SafeAILab　　- 年月：README 显示 last update **2025-09-18**（EAGLE-3 被 NeurIPS'25 接收）；**star 数 2.5k**
- 类型：代码仓库　　- 层次：源码级
- **好在哪**：
  1. **三代的差异被压缩成了三句话，非常适合做谱系表**：
     - EAGLE-1："Extrapolating the **second-top-layer contextual feature vectors** of LLMs"
     - EAGLE-2：用草稿模型的 **confidence scores 近似接受率**，据此**动态调整草稿树的结构**
     - EAGLE-3："**Removes the feature prediction constraint** in EAGLE and simulates this process during training using **training-time testing**. Replaces top-layer features with **fusion of low-, mid-, and high-level semantic features**."
     **"EAGLE-3 把 EAGLE-1 的核心约束（预测特征）删掉了"——这是第 13 篇"什么被后来推翻"最干净的一条。**
  2. **无损表述用词是精确的**："provably maintaining the **consistency with vanilla decoding in the distribution of generated texts**" —— 说的是分布，不是结果。
- **哪里不严谨或讲错了**：
  - **README 的 Todo 列表里仍写着 "Support non-greedy inference (provably maintaining text distribution)"（2026-08 核验）**。也就是说，**仓库自己把"带证明的非贪心推理支持"列为待办**。这与正文那句 "provably maintaining ... distribution" 并列出现，**读者无法判断当前实现到底在 $T>0$ 下是不是 L1**。
    **→ 本库第 13 篇必须点名这条矛盾，并且不能替它下结论（要么去读源码验证，要么写"README 自相矛盾，本次未验证实现"）。**
  - **加速比口径不全**："3x faster than vanilla decoding (13B)"，给了 Vicuna 13B / **2×RTX 3090** / fp16，**但没给 batch size、没给 temperature、没给树的规模**。
  - 无任何 batch size 上限的说明。
- 是否踩了 8 条：**未明确踩坑**，但 Todo 与正文的矛盾会误导 E1 的判断。
- 值得直接引用：三代差异那三句 + Todo 那条（作为"官方自己都还没做完非贪心保证"的证据）。

---

#### Recurrent Drafter for Fast Speculative Decoding（Apple）

- URL：https://machinelearning.apple.com/research/recurrent-drafter （**访问成功**）
- 作者/机构：Aonan Zhang、Ray Zhang、Yunfei Cheng、Chong Wang、Yi Wang（Apple）　　- 年月：**2024-11**
- 类型：研究页　　- 层次：进阶
- **好在哪**：
  - **一句话点出了与 Medusa 的结构差异**：在 beam search 结果上做 **dynamic tree attention**，"to eliminate duplicated prefixes in candidate sequences"。**"树是从 beam 结果里去重出来的，而不是笛卡尔积拍出来的"——这是第 16 篇讲树构造演化的关键一步。**
  - 端侧数据有价值：Apple Silicon + MLX + Metal GPU 上 **最高 2.3×**（H100 上 Vicuna MT-Bench 最高 2.8×）。**"端侧也能跑投机"这条线在社区材料里很少见。**
- **哪里不严谨或讲错了**：
  - **页面信息极简**：没有接受判据、没有无损性表述、没有 batch/γ、没有与 Medusa 的定量对比（页面上那句"prior methods including Medusa either degrade acceptance or introduce overheads that limit scaling"来自关联文章而非本页正文，**引用时需注意出处**）。
  - 2.8× / 2.3× 均无 batch 标注。
- 是否踩了 8 条：**未涉及**。

---

### F. 代码仓库与可跑实现（本次逐行核对了接受判据与残差分布）

**总体结论：本次核到的所有实现，接受判据与残差分布全部正确。E2/E3 在代码层面几乎不存在。** 唯一需要警惕的是 romsto 的可选错误变体。

---

#### feifeibear/LLMSpeculativeSampling

- URL：https://github.com/feifeibear/LLMSpeculativeSampling （README 访问成功）；核心实现 https://raw.githubusercontent.com/feifeibear/LLMSpeculativeSampling/main/sampling/speculative_sampling.py （**访问成功**）
- 作者：feifeibear　　- 年月：README 显示 last update **2023-09-21**；**star 数 923**
- 类型：代码仓库　　- 层次：源码级
- **好在哪**：
  - **它是唯一一个把两篇论文的算法差异做成两份代码并排放的仓库**：README 明写 "The speculative sampling is proposed by Google and Deepmind independently. So I implement **two slightly different versions** of speculative sampling: Google's and Deepmind's."
    **它好在哪（具体）**：读者可以直接 diff 两个版本，亲眼看到"两篇论文的差异到底在哪"（主要在 KV cache 的回滚方式与最后一个 token 的处理），而不是靠文字描述。**第 09 篇（两篇同期论文的异同）应该直接推荐读者去 diff 这两个文件。**
  - 两版实现都正确：
    ```python
    # v1（用"拒绝"的写法）
    if r > (target_prob[..., j]) / (approx_prob[..., j]):   # 拒绝
    t = sample(max_fn(target_prob[:, n, :] - approx_prob[:, n, :]))
    # v2（用"接受"的写法，显式 min）
    if r < torch.min(torch.tensor([1], device=q.device), p[:, prefix_len+i-1, j] / q[:, prefix_len+i-1, j]):
    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
    ```
    **v1 与 v2 恰好演示了"写成拒绝"与"写成接受"两种等价写法**，教学价值高。
- **哪里不严谨或讲错了**：
  - **README 的 benchmark 口径严重不全**：llama2-7b 1084.86、llama2-70b 329.83、投机采样 427.02。**没有硬件、没有 $\gamma$、没有 batch，甚至单位（tokens/sec）也存疑**。按面值算只有 **1.29×**，远低于同期其它报告。**本库引用此仓库只用于"读代码"，绝不引用其数字。**
  - 2023-09 之后基本停更，不反映 2026 年的实现（无树、无 EAGLE）。
- 是否踩了 8 条：**代码未踩**；README 数字属铁律二问题。

---

#### romsto/Speculative-Decoding

- URL：https://github.com/romsto/Speculative-Decoding （**访问成功**）；核心实现 https://raw.githubusercontent.com/romsto/Speculative-Decoding/main/sampling/speculative_decoding.py （**访问成功**）
- 作者：romsto　　- **star 数 115**；last update 未取到，标「未查证」
- 类型：代码仓库　　- 层次：源码级
- **好在哪**：
  1. **README 的措辞是精确的（罕见）**："allows to generate sequences faster than the classic auto-regressive decoding **without changing the output distribution** or requiring further fine-tuning."
  2. **三种解码放在一个仓库里对拍**：经典自回归、beam search（带长度惩罚）、投机解码，外加 **NASD（Ngram Assisted Speculative Decoding）**。**"同一个仓库里能直接对拍"对做无损性实验极其友好，第 04/07 篇的实验建议参考它的组织方式。**
  3. 实现正确：
     ```python
     r = torch.rand(corrected_gamma, device=target.device)
     fractions = p / q
     for i in range(corrected_gamma):
         if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
             ...
     p_p = max_fn(p[..., n, :] - q[0, n, :])
     ```
- **哪里不严谨或讲错了 / 需要警惕的地方**：
  - **⚠ 它提供了一个可以关掉残差修正的开关**：
    ```python
    if not skip_sample_adjustment:
        p_p = max_fn(p[..., n, :] - q[0, n, :])
    else:
        p_p = p[..., n, :]          # ← 直接从目标分布 p 重采，这正是 E3
    ```
    **`skip_sample_adjustment=True` 就是本库列的第 3 条高频错误的可执行版本。**
    **两面看**：作为**消融实验开关**，它极其有价值——本库 `_lab/test_lossless.py` 完全可以做同样的事，用它来实证"跳过残差修正会让分布偏成什么样"。但作为**默认可用的 API**，它是个陷阱：任何抄这段代码的人一旦打开这个开关，就会在不知情的情况下破坏无损性，而 README 对此**没有任何警告**。
    **→ 本库第 26 篇应把这条写成"社区代码里的一个真实陷阱"，并在 `_lab/` 里做对应的证伪实验。**
  - star 数不高，社区验证不充分。
- 是否踩了 8 条：**默认路径未踩；提供了 E3 的开关且无警告**。

---

#### shreyansh26/Speculative-Sampling

- URL：https://github.com/shreyansh26/Speculative-Sampling （**访问成功**）
- 作者：shreyansh26　　- **star 数 112**；last update 未取到，标「未查证」
- 类型：代码仓库（DeepMind 版实现）　　- 层次：源码级
- **好在哪**：
  - **它是本次核到唯一一个把 temperature=0 与 temperature=0.5 分开报的仓库**：
    | 目标 / 草稿 | T=0 | T=0.5 |
    |---|---|---|
    | OPT-13b / OPT-1.3b | 1.78× | 1.75× |
    | OPT-6.7b / OPT-1.3b | 1.46× | 1.51× |
    **它好在哪（具体）**：社区反复说"采样会降低接受率"（HF 的 assisted generation 博客也这么说），**这组数字表明在 OPT 上 T=0.5 与 T=0 的差异极小，甚至在 6.7b 上 T=0.5 反而更快**。这是一条**值得本库在 `_lab/` 里复核的、与常识相左的观察**，第 05 篇可以拿它当悬念。
  - 另一条观察也值得引："The speedup ratio seems to **increase as the target model size increases**"（13b 比 6.7b 快），与 Leviathan 的 $\alpha>c$ 判据一致（目标越大，$c$ 越小）。
- **哪里不严谨或讲错了**：
  - **口径不全**：没有硬件、没有 $K$、没有 batch。上面那两条观察都只有 2 个数据点，**统计上不足以下结论**（本库踩过"单样本判定"的坑，引用时必须标注"仅 2 组配置，需复核"）。
- 是否踩了 8 条：**未踩**。

---

#### gpt-fast 的 `generate.py`

见 §3.B 的 gpt-fast 条目。**代码完全正确，且只有约 50 行，是"最短的正确实现"这个位置的最佳候选。**
URL：https://raw.githubusercontent.com/pytorch-labs/gpt-fast/main/generate.py （**访问成功**）

---

### G. 中文材料（C 级来源，但同样逐条评点）

> **前置说明**：本轮中文来源受抓取限制严重（知乎 403、CSDN 521、公众号/飞书需登录、B 站未覆盖）。以下三份是**实际读到正文**的。
> **一个正面的整体观察**：**这三份中文材料在接受判据和残差分布上全部写对了**，比英文侧的 NVIDIA intro 博客、Wikipedia、Modular Handbook 还准。中文社区在"数学怎么写"上并不弱。**弱的是口径（无损口径、加速比口径、batch 口径）和时效（参数过期）。**

---

#### 探秘Transformer系列之（30）--- 投机解码

- URL：https://www.cnblogs.com/rossiXYZ/p/18837229 （**访问成功**，curl 亦 200）
- 作者/平台：罗西的思考 / 博客园　　- 年月：**2025-04-23**
- 类型：博客长文（约 12000 字）　　- 语言：中文　　- 层次：进阶
- **好在哪**：
  1. **接受判据写成了可执行代码，并且带随机数**：
     > `r_i = torch.rand_like(probability_ratio); is_accepted = r_i <= probability_ratio`
     并配文字解释 "当 probability_ratio > 1 ... keep the token. Otherwise reject with p = 1 - probability_ratio"。**"大于 1 就留，否则按 1-ratio 拒"这个讲法与 Leviathan 原文的叙述顺序一致，比直接甩 min 更好懂。**
  2. **残差分布给了完整式子**：`p'(x) = norm(max(0,p(x) − q(x)))`，并解释其目的是"弥补从近似模型 Mq 中得到的猜测与目标模型 Mp 分布之间的差异"。
  3. **无损的表述是分布口径，没有踩 E1**：
     > "投机解码无需对输出进行任何更改，就可以保证和使用原始模型的**采样分布完全相同**"
     > "这种验证和重采样过程在**理论等价**于直接从目标 LLM 采样"
     **中文社区里能把"采样分布"三个字写对的不多，这份写对了。**
  4. 覆盖面广：Blockwise Parallel Decoding、两篇奠基论文、Token Tree Verification、SpecInfer、Medusa、EAGLE，且用了 CPU 分支预测类比。图多（流程对比图、并行验证示意、树注意力 mask）。
- **哪里不严谨或讲错了**：
  - **完全没有 $E[\tau]=(1-\alpha^{\gamma+1})/(1-\alpha)$，也没有任何加速比模型**。读者读完知道"怎么做"，不知道"能赚多少、什么时候赚"。
  - **完全没有 batch / 大并发下负收益的讨论**。这是中文长文的通病。
  - MTP 只在参考文献里出现，未展开（对 2025-04 的时间点尚可理解）。
- 踩了 8 条中的：**E1/E2/E3 均未踩（◎）**；**E5、E8 属漏讲**。
- 值得直接引用：那句"保证和使用原始模型的采样分布完全相同"——**可以作为"中文社区里表述正确的样本"，与 Wikipedia 的 "same results" 并排对照**。

---

#### Speculative Decoding（推测解码/投机推断）深度全景解析

- URL：https://www.cnblogs.com/SCCQ/p/19837997 （**访问成功**）
- 作者/平台：SHICENT / 博客园　　- 年月：**2026-04-09**（约 15000 字）
- 类型：博客长文　　- 语言：中文　　- 层次：进阶
- **好在哪**：
  1. **判据与残差全对，且写成了伪代码**：
     > `αᵢ = min(1.0, p_xi / q_xi)  # 接受概率`
     > `if random.uniform(0, 1) < α_i: ✅ 接受草稿 token`
     > `p'(x) = normalize(relu(target_probs[i] - draft_probs[i]))`
     **用 `relu` 表达 `max(0,·)` 是个好写法**，对写过深度学习代码的读者一秒就懂。
  2. **无损表述是分布口径**："在**数学上严格保证输出分布无损**"，并给了 `P(output=t) = p(t)` 的形式。
  3. **覆盖面是本次中文材料里最广的**：经典 Spec、SpecInfer、SpecTr、REST、PLD、Medusa、EAGLE-1/2/3、Lookahead、Hydra、Draft&Verify、Ouroboros，16+ 种方法。**作为中文侧的"方法索引"很好用。**
- **哪里不严谨或讲错了**：
  - **踩 E8，而且是最典型的那种**：给出了 `E[tokens per round] = (1-α^(γ+1))/(1-α)`，**但没有提这个式子成立需要"每步接受事件 i.i.d."这个假设**。Leviathan 原文明写 "If we make the simplifying assumption that the βs are i.i.d."，这份长文照抄了结论、丢掉了前提。**本库第 06 篇要专门讲这一步丢了什么。**
  - **踩 E6（漏讲型）**：讲了 Medusa，**但完全没有讨论 typical acceptance 是有损的**。读者会顺理成章地以为它和标准投机采样一样无损。
  - **对大 batch 的表述过软**："高 QPS 场景下，额外草稿开销**可能**降低收益。vLLM 正在开发动态推测解码" —— **"可能"是废话式表述**，没有给拐点、没有给数字，而 vLLM 官方博客给了"高 QPS 下慢 1.4×–1.8×"的硬数字。
  - **数据引用跨度 2023–2025 且未统一基准**，把不同来源的加速比并列（违反铁律二）。
  - "vLLM 正在开发动态推测解码"这条在 2026-04 已经过期（vLLM 文档已把 Dynamic Speculative Decoding 列为既有方法）。
- 踩了 8 条中的：**E6（漏讲）、E8**；E5 表述过软。
- 值得直接引用：`p'(x) = normalize(relu(...))` 这个写法可以采纳。

---

#### 投机解码深度解析：如何在不牺牲精度的前提下将大模型推理吞吐量提升2-3倍

- URL：https://tbr8.org/投机解码深度解析：如何在不牺牲精度的前提下将/ （**访问成功**）
- 作者/平台：tbr8.org（**作者未注明**；页脚称"本站文章皆为原创"）　　- 年月：**2026-08-01**
- 类型：博客长文　　- 语言：中文　　- 层次：进阶
- **好在哪**：
  1. **判据与残差全对，且给了伪代码行号级的细节**：
     > "以概率 min(1, q(x_t) / p(x_t)) 决定是否接受"（该文记号为 q=目标、p=草稿，DeepMind 系）
     > 伪代码第 24 行 `if random() < accept_prob:`
     > "从调整后的分布 (q(x) - p(x))_+ / Z 中重新采样一个替代 token"
     > 伪代码第 31 行 `corrected = (q[t] - draft_probs[t]).clamp(min=0)`
  2. **无损的正文表述是分布口径**："投机解码的输出分布与原始自回归解码完全一致，即零精度损失"、"这种修正采样确保了最终输出的**统计特性**与大模型逐 token 生成完全一致"。
  3. **正文里明确提了高并发收益递减**："高并发时投机收益递减"、"避免投机反噬吞吐"。**"投机反噬吞吐"这个中文表述很有画面感，可以采纳。**
- **哪里不严谨或讲错了**：
  - **标题直接踩 E5，且是最刺眼的位置**：《……将大模型推理**吞吐量**提升 2-3 倍》。正文里给的是"获得 2-3 倍甚至更高的**加速比**"，**加速比 ≠ 吞吐量**，而且正文自己承认"高并发时投机收益递减"。**标题与正文自相矛盾。**
    **→ 本库第 26 篇可以把这条写成"标题党如何制造社区共识"的样本：正文是对的，但绝大多数人只看到标题。**
  - **配置参数已过期**：文中给出 `--speculative-disable-by-batch-size 32` 作为"batch 超过阈值自动禁用"的做法。**本次核验 vLLM main 分支的 `vllm/config/speculative.py`（2026-08-22），grep `disable_by_batch_size` 无任何命中**，该参数在当前版本已不存在，能力由 "Dynamic Speculative Decoding" 承担。**引用中文材料的配置参数必须核到源码，这是一条通用教训。**
  - **作者未署名**，无法追溯；且"本站文章皆为原创"的声明与文中大量与英文材料高度同构的表述并存，**是否为编译/整合无法判断**，标「未查证」。
- 踩了 8 条中的：**E5（标题）**；正文 E1/E2/E3 均未踩。
- 值得直接引用："投机反噬吞吐"这个词；以及把标题拿来当反面案例。

---

### H. 目录、榜单与教程（第 30 篇附录的骨架）

---

#### Spec-Bench（hemingkx）

- URL：https://github.com/hemingkx/Spec-Bench （**访问成功**）；榜单 https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/Leaderboard.md （**访问成功**）
- 作者：hemingkx　　- **star 数 405**，榜单最后更新 **2025-04-22**
- 类型：基准 + 榜单　　- 层次：进阶
- **好在哪**（**这是本次核到"口径最全"的一份材料，堪称铁律二的模范**）：
  1. **它把口径全部写在榜单头上**：
     - RTX 3090 组："a single NVIDIA GeForce RTX 3090 GPU (24GB) with 12 CPU cores"、"Pytorch 2.5.1, under CUDA 12.1"
     - A100 组："a single NVIDIA A100 GPU (80GB) with 96 CPU cores"、"Pytorch 2.5.1, under CUDA 11.5"
     - 统一模型与设置：**Vicuna-7B/13B/33B-v1.3、greedy decoding、FP16、batch size = 1**
     **"batch size = 1" 明写在榜单上——这一条就把所有引用它的人从铁律二的坑里救出来了。**
  2. **它引入了 #MAT（Mean Accepted Tokens）作为与加速比并列的指标**。**这至关重要**：加速比混合了硬件与实现效率，#MAT 只反映草稿质量。**本库第 05 篇应当采纳"必须同时报 #MAT 与加速比"这个规范。**
  3. 榜单数据（可直接引用，口径齐）：
     - RTX 3090 / Vicuna-7B：SAMD[EAGLE2] **2.38× / 4.61 MAT**；EAGLE2 **2.19× / 4.35**；EAGLE **2.03× / 3.57**
     - A100 / Vicuna-7B：SAMD[EAGLE2] **2.73× / 4.58**；EAGLE2 **2.36× / 4.34**；Recycling **2.22× / 2.73**
     - A100 / Vicuna-13B：**EAGLE3 3.02× / 5.71 MAT**；SAMD[EAGLE2] 2.77× / 4.52；EAGLE2 2.46× / 4.43
     - A100 / Vicuna-33B：EAGLE2 2.59× / 4.05；SAMD[EAGLE2] 2.59× / 4.07；EAGLE 2.43× / 3.39
     **注意 Recycling 那一行：#MAT 只有 2.73，却拿到 2.22× —— 说明"接受长度短但草稿极便宜"也能赢。这一条正好呼应 Leviathan 的 $\alpha>c$ 判据。第 05/06 篇可用。**
  4. 关联论文明确：Xia et al., "Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding", Findings of ACL 2024, arXiv:2401.07851（2024-01）。
- **哪里不严谨或讲错了**：
  - **榜单本身没有一句关于"跨论文加速比不可比"的警示**。它靠"自己统一跑"来解决可比性，但没有把这个道理写出来。**本库引用时要替它把这层意思说明白：Spec-Bench 存在的理由，就是论文自报的加速比不可比。**
  - **全部是 batch size = 1 + greedy**。也就是说，**它测不出本库最关心的两件事：大 batch 下的负收益、$T>0$ 下的接受率变化**。这是它的边界，必须标注。
  - 榜单最后更新 2025-04，**未覆盖 2025 下半年到 2026 的方法**。
- 是否踩了 8 条：**未踩，且对 E5 是最好的解药（◎）**。
- 值得直接引用：整张榜单 + "batch size = 1" 这条口径声明。

---

#### Awesome list：hemingkx/SpeculativeDecodingPapers

- URL：https://github.com/hemingkx/SpeculativeDecodingPapers （**访问成功**）；README 原文 https://raw.githubusercontent.com/hemingkx/SpeculativeDecodingPapers/main/README.md （**访问成功**）
- 作者：hemingkx　　- **star 数 1.3k**　　- 描述："📰 Must-read papers and blogs on Speculative Decoding ⚡️"
- 类型：论文索引　　- 层次：全档
- **好在哪**：
  - **它的分类法本身就是一张谱系图**，本库第 15 篇（谱系图与被淘汰的分支）可以直接对照：
    `Survey` / `Speculative Decoding for Seq2Seq` / `Speculative Decoding for LLMs` / **`Multi-Token Prediction`** / **`Speculative Decoding for Diffusion LMs`** / `Multimodal Speculative Decoding` / **`Long-Context Speculative Decoding`** / **`Speculative Decoding for Mixture-of-Experts`** / `Alignment` / `Benchmarks` / `Applications` / **`Analysis`** / `Other Techniques`
    **值得注意的三个分类**：`Speculative Decoding for MoE`（呼应 Red Hat 那份 gpt-oss 材料）、`Long-Context`（第 22 篇）、`Analysis`（说明"分析类"已经自成一个门类——本库的定位正在这里）。
  - **覆盖 500+ 篇，时间跨度 2018 → 2026-06**，是本次找到的最全索引。
- **哪里不严谨或讲错了**：
  - 只列不评。**没有"哪些分支已经死了"的标记**，这正是本库第 15 篇要补的增量。
  - README 的 last update 本次未取到，标「未查证」。
- 是否踩了 8 条：不适用。

---

#### COLING 2025 Tutorial：Speculative Decoding for Efficient LLM Inference

- URL：https://speculative-decoding.github.io/ （**访问成功**）
- 主讲：Heming Xia、Yongqi Li、Wenjie Li（香港理工大学），Cunxiao Du（SEA AI Lab），Qian Liu（TikTok）　　- 年月：**2025-01-19**，Abu Dhabi National Exhibition Centre, Capital Suite 7, 09:00–12:30 GST
- 类型：会议 tutorial（含 slides 与录像）　　- 层次：入门→进阶
- **好在哪**：
  - **它是本次找到的唯一一份"体系化教学"材料**（不是博客、不是文档、不是论文），且是同一批人写的那篇 ACL Findings 综述的作者团队。
  - 议程结构可以直接借鉴：`Introduction & Definition (40min)` → `History and Taxonomy of Methods (45min)` → `Cutting-edge Algorithms (40min)` → `Downstream Adaptations (30min)` → `Final Remarks & Q&A (20min)`。**"先定义、再史与分类、再前沿、再下游适配"这个顺序与本库 A→B→C→D→E 的编排高度一致，可作为编排合理性的旁证。**
  - 资源：slides `https://tinyurl.com/speculative-decoding-tutorial`、录像 `https://tinyurl.com/spec-tutorial-recording`（**本次未跟进 tinyurl 跳转目标，标「未核验」**）。
- **哪里不严谨或讲错了**：
  - 主页对无损的表述是概括性的（"maintaining original distributions"），**未细分 L1/L2/L3**。
  - **本次只读到主页，slides 与录像未打开**，因此**不能对其内容质量下结论**。
- 是否踩了 8 条：**信息不足，不判定**。

---

### I. 负面发现（同样重要）

| 常被引用的来源 | 本次核验结论 |
|---|---|
| **Lilian Weng《Large Transformer Model Inference Optimization》** | **该文没有投机解码/投机采样章节。** 本次逐条抓取了全文目录：Methods Overview / Distillation / Quantization / Pruning / Sparsity（含 MoE）/ Architectural Optimization / Citation / References。**没有任何一节涉及 speculative decoding、speculative sampling 或 parallel decoding。** 该文发表 2023-01-10、更新 2023-01-24，早于两篇奠基论文进入主流视野。**中文材料里把它列为"投机解码必读"的，是错误转引。本库第 30 篇不收录它，并在第 26 篇点名这条误传。** |
| **Sebastian Raschka（Ahead of AI）** | 本次未找到其关于投机解码的专文。**因 WebSearch 配额用尽，检索不充分，结论标「未查证」，不写入附录。** |
| **HF `transformers` 文档的 Generation Strategies 页** | 本次抓取 https://huggingface.co/docs/transformers/en/generation_strategies （版本标识 v5.15.1，核验 2026-08）**全文只有 Greedy search / Sampling / Beam search + Custom generation methods（`custom_generate`）+ Resources**，**未见 "Assisted decoding / Speculative decoding" 小节**。页面末尾把 `custom_generate` 分成 Community 与 Tutorials 两个 collection，并说明后者是"reference implementations for methods that **previously were part of `transformers`**"。**这暗示 assisted generation 可能已从核心 API 迁往 `custom_generate` 社区方法**，但**本次未直接核到迁移公告，此推断标「待确认」，正文不得写成事实。** 无论如何：**指向 generation_strategies 页面讲 assisted decoding 的链接，在 2026-08 已经指不到内容了**，第 30 篇附录必须改指 HF 博客原文。 |
| **`--speculative-disable-by-batch-size`** | **在 vLLM main（2026-08-22 核验）的 `vllm/config/speculative.py` 中不存在。** 仍在教这个参数的中文材料已过期。 |
| **Aphrodite Engine 投机解码文档** | HTTP 404，**链接已失效（2026-08 核验）**。 |

---

## 4. 可直接采纳的「精彩解释」清单

> 每条给出：**出处 → 它解决了哪个具体理解障碍 → 本库在哪一篇用 → 用的时候必须补什么**

| # | 精彩解释 | 出处 | 它解决的理解障碍 | 用在 | 采纳时必须补 |
|---|---|---|---|---|---|
| 1 | **Verity（资深）/ Drake（初级）工程师 code review** | PyTorch gpt-fast 博客 | 一次性解决三件事：验证为什么比生成便宜、为什么拒绝后**后面全废**、为什么质量不降 | **02**、19 | 必须紧跟一句："原文说'扔掉不匹配的'是贪心口径，随机采样下用的是 $\min(1,p/q)$" |
| 2 | **"换了个随机种子"** | Jay Mody | **把"分布无损 ≠ 结果相同"讲成程序员的日常经验** | **07**（题眼）、27 | 补上例外："$T=0$ 时才逐 token 相同" |
| 3 | **√7 的例子：抄一个 7 很容易，算 2.646 很难** | Google Research 回顾 | 让"有些 token 是白送的"从抽象变具体 | **01**、05 | 无需补，但要注明是 Leviathan 本人写的 |
| 4 | **CPU 分支预测类比** | Leviathan 论文正文 + Google 博客 + Charles Frye | 为"猜错了就丢弃、不影响正确性"提供已有心智模型 | **01**、04 | 注明论文引的是 Burton 1985 / Hennessy & Patterson 2012 |
| 5 | **"meticulous scientist / quick assistant"** | NVIDIA intro 博客 | 比 Verity/Drake 更短，适合一句话导入 | 00、01 | **必须紧跟纠正**：同一篇里把判据写成了确定性匹配、把吞吐说成无条件提升 |
| 6 | **scoring 与 sampling 的代价差** | Charles Frye | 把"并行验证几乎免费"落到"评分可并行、采样必须串行"这个具体机制上 | **03** | 本库要补它没做的量化（算术强度/roofline/K 多大失效） |
| 7 | **"从拒绝讲起"的判据叙述** | Leviathan 论文 §2.3 + 罗西的思考 | 比直接甩 $\min$ 更容易看出"$\min$ 在防什么" | **04** | 之后再收成 $\min(1,p/q)$ 的紧凑形式 |
| 8 | **两情形式的判据描述** | Aleksa Gordić | 同上，且是散文版 | 04 | 纠正其 "in expectation" 的措辞 |
| 9 | **`relu` 表达 `max(0,·)`** | 博客园 SHICENT | 深度学习读者一秒理解残差分布 | 04、02 | — |
| 10 | **accepted / recovered / bonus 三分法** | vLLM `rejection_sampler.py` | 给"一步到底产出了哪几种 token"一套精确词汇 | **02**、04、16 | 全库统一采用这套术语 |
| 11 | **Gumbel-max 代替显式归一化** | vLLM `rejection_sampler.py` | 展示"教学实现 vs 生产实现"的真实差距 | **20** | 说明两者在浮点上并不严格等价 |
| 12 | **"Switched from rejection sampling to greedy verification"** | Snowflake Arctic | 证明 L2 不是退化，而是**另一种被主动选择的需求**（可复现性） | **07**、19、25 | 指出"接受率不降"是他们的实测而非定理 |
| 13 | **三层无损声明（理论 / 实现验证 / 浮点）** | vLLM 官方文档 | 把"无损"从一句口号拆成三层可核查的断言 | **07**（整段采纳） | — |
| 14 | **"batch size 变化会改变 logprobs"** | vLLM 官方文档 | 揭示"算法无损"之上还有一层系统层的不确定性 | **07**、19 | 本库应在 `_lab/` 之外指出这是硬件归约顺序问题 |
| 15 | **"speed ups are only observable at low batch sizes"** | TensorRT-LLM 文档 | 把失效条件写成一句可判定的话 | **19**（原样引用） | 与 Red Hat 2026 的反例并列，说明这条依赖模型结构 |
| 16 | **"throughput reduction beyond a batch size of 64"** | PyTorch/IBM | 给出了一个**具体数字**的拐点 | **18**、19 | 标注时效（2024 年 IBM 内部负载） |
| 17 | **深度 / 分支因子 / 验证预算 三旋钮** | SGLang 文档 | 把"树"从一个名词拆成三维配置空间 | **16**、17 | — |
| 18 | **"树 = 用 mask 换 batch"** | Together.ai Medusa 博客 | 说清树注意力省的到底是什么 | **16** | — |
| 19 | **next-next token top-1 60% / top-5 80%** | Together.ai Medusa 博客 | 直接解释了**为什么必须有树**（单链只能用 top-1） | **12**、16 | — |
| 20 | **"Relaxing the requirement of matching the distribution"** | Together.ai Medusa 博客 | Medusa 作者自认有损，堵死 E6 | **12**、07 | 指出下游 README 与转述丢了这句 |
| 21 | **$r_i=\min(1,q_i/(\alpha p_i))$ 的有损参数化** | Vivien Tran-Thien | 把 L1→L3 变成一条连续曲线上的移动 | **07**、12 | 说明其约束是 KL 而非 TV |
| 22 | **Leviathan §4.1 的 small/base/large 三行表** | Leviathan 论文 | **同时干掉"草稿越小越好"与"α 越高越快"** | **05**、21 | — |
| 23 | **NVIDIA 405B 的 1B/3B/8B 草稿倒 U 形** | NVIDIA 3.6x 博客 | 用生产级数字证明最优草稿尺寸存在且随目标尺寸变化 | **05**、21 | 标注"其 throughput 实为单请求 tok/s" |
| 24 | **#MAT 与加速比必须并报** | Spec-Bench | 把"草稿质量"与"实现效率"分离 | **05**、30 | — |
| 25 | **Recycling：#MAT 仅 2.73 却拿到 2.22×** | Spec-Bench 榜单 | 证明"接受长度短但草稿便宜"也能赢，呼应 $\alpha>c$ | **05**、06 | — |
| 26 | **`draft_sample_method="greedy"` 默认 + one-hot 处理** | vLLM 源码 | 揭示生产系统默认不用草稿的完整分布 | **05**、20 | 说明这仍是 L1，但接受率数学不同 |
| 27 | **α 定义为"前 i+1 个全接受的边际概率"** | vLLM 源码 | 给 α 的口径分歧一个工程侧的权威定义 | **05**（核心） | 与 Leviathan 的条件概率定义对照 |
| 28 | **"minimum-variance schedule" 而非几何分布** | vLLM 源码 | **工程上明确不采用 i.i.d. 几何模型** | **06**（E8 的核心论据） | 本库要在 `_lab/` 验证"为什么这个调度方差最小" |
| 29 | **MTP 复用同一层会掉接受率的告警** | vLLM 源码 | E7 在生产系统里的具体表现 | **14** | — |
| 30 | **suffix decoding 每 token 20 微秒（CPU）** | Snowflake Arctic | 草稿可以便宜到不占 GPU | **11** | — |
| 31 | **12TB：离线训 EAGLE3 要存的 hidden states** | LMSYS SpecForge | 让"训草稿的成本"变具体 | **21** | — |
| 32 | **3-4 头（语言）/ 6-8 头（代码）** | PyTorch/IBM | γ 的选择依赖任务可预测性 | **17**、21 | — |
| 33 | **"3 → 4 个草稿 token 掉 8% 吞吐"** | Red Hat 2026 | γ 的边际成本的具体数字 | **17**、18 | — |
| 34 | **"投机反噬吞吐"** | tbr8.org（中文） | 一个好用的中文表述 | 18、19 | 该文标题本身踩 E5，引用时只取词不取题 |
| 35 | **官方 2.8× vs 独立复现 1.4×，issue 以 stale 关闭** | vLLM 博客 + issue #10318 | 铁律二最有说服力的活教材 | **19**、26 | 说明口径差异（H100 PCIe vs 未注明）未被澄清 |

---

## 5. 错误传播链分析（第 26/27 篇的骨架）

### 5.1 E1（"无损 = 输出一样"）的传播链

```
Leviathan 摘要："without any changes to the outputs" / "with identical outputs"
        │  （正文其实写的是 "without changing the model output distribution"）
        ├──→ Karpathy 推文（贪心框架，"tokens that agree"）  ── 影响力最大，本次无法核验
        ├──→ HF Whisper 博客："mathematically ensuring exactly the same outputs"
        ├──→ PyTorch gpt-fast 博客："mathematically identical results"（但代码是对的）
        ├──→ PyTorch/IBM："guaranteeing that the overall output ... is identical to that of vanilla decoding"
        ├──→ NVIDIA intro 博客："the final output is identical to what the target model would have produced"
        ├──→ Modular Handbook："guarantees the final output matches exactly what the original target model would have produced"
        ├──→ Wikipedia："produces the same results as standard decoding"
        └──→ 中文材料（本次核到的三份反而没踩，踩的是英文权威源）
```

**关键观察（这是本库能给出的、社区没有的判断）：**

1. **E1 的重灾区是英文权威源，不是中文社区。** 本次核到的三份中文长文（罗西、SHICENT、tbr8 正文）**全部用了"分布"口径**；踩坑的是 NVIDIA、PyTorch、IBM、Wikipedia、Modular。
2. **踩坑的材料有一个共同特征：它们的实现本来就是贪心验证。** IBM 的 speculator、TensorRT-LLM、Arctic 都是 L2 实现。**对它们自己而言那句话不错，错在用普遍口吻说出来。**
3. **最干净的解药有两个**：Jay Mody 的"换了个种子"（直觉层）、vLLM 文档的三层声明（严谨层）。**第 07 篇建议先给前者再给后者。**

### 5.2 E6（Medusa typical acceptance 说成无损）的断点

```
Medusa 论文 / Together.ai 博客：明确写了 "Relaxing the requirement of matching the distribution"
        │
        ✂ 断在这里 ✂
        │
Medusa GitHub README：只写 "a typical acceptance scheme is employed"，一个字都没提有损
        │
        └──→ 下游所有二手讲解（包括本次核到的中文 SHICENT 长文）：讲了 Medusa，没讲它有损
```

**同型错误在 2026 年换了主角**：TensorRT-LLM 文档的 MTP 参数 `use_relaxed_acceptance_for_thinking` / `relaxed_topk` / `relaxed_delta`，**同样是放宽接受判据、同样没有任何有损标注**。**第 14 篇要点名这一条。**

### 5.3 E8（E[τ] 丢独立性前提）的断点

```
Leviathan 论文：明写 "If we make the simplifying assumption that the βs are i.i.d.,
                ... a capped geometric variable, with success probability 1−α and cap γ+1"
        │
        ✂ 断在"照抄公式不照抄前提" ✂
        │
社区长文（如 SHICENT）：直接给 E[tokens per round] = (1−α^(γ+1))/(1−α)，前提没了
        │
        └──→ 读者以为这是恒等式，拿去套自己的系统，对不上就以为是实现有 bug
```

**最硬的反证在工程侧**：vLLM 的 `_acceptance_length_to_rates` 用的是"最小方差调度"`[1,1,...,frac,0,...]`，**根本不是几何衰减**。**一个生产系统在同一个"平均接受长度"下选了完全不同的分布形状，就说明几何模型只是众多模型之一。第 06 篇要把这两个模型并排画出来，并在 `_lab/` 里跑数字对比。**

---

## 6. 给第 26 / 27 / 30 篇的写作建议

### 第 26 篇（社区精彩解释精选）建议结构

1. **开场用 gpt-fast 那个"散文错、代码对"的案例**，立刻建立本篇的价值：读社区材料要分层读。
2. **"该采纳的"**：按 §4 的 35 条，挑 12–15 条写成"每条一段"，每段必须写清"它解决了哪个具体障碍"。
3. **"该纠正的"**：按 §5 的三条传播链，每条画一张 mermaid 传播图。
4. **"最反直觉的三份材料"**：Leviathan §4.1 的 small/base/large 表、NVIDIA 405B 的倒 U 形、Spec-Bench 里 Recycling 的低 #MAT 高加速。
5. **"中文社区的真实水平"**：点明本次的反常发现——中文材料在公式上比英文权威源更准，弱在口径和时效。

### 第 27 篇（常见误解与判据）建议

把 §2 的命中矩阵直接做成本库的**自查表**：读到任何一份新材料，按这 8 列打勾。并给每条错误配一个**一句话判据**：

| 错误 | 一句话判据（读材料时问自己） |
|---|---|
| E1 | 它说的是 "same output" 还是 "same output **distribution**"？有没有说 $T=0$ 这个前提？ |
| E2 | 判据里有没有出现**随机数**？没有随机数的判据一定是贪心口径 |
| E3 | 拒绝后重采的分布，有没有减去 $q$？没减就是错的 |
| E4 | 它有没有同时给 $\alpha$ **和** $c$（草稿耗时占比）？只谈 $\alpha$ 的建议都不完整 |
| E5 | 它报的是**单请求 tok/s** 还是**系统吞吐**？有没有并发扫描？ |
| E6 | 讲 Medusa/relaxed acceptance 时，有没有一句"这会改变分布"？ |
| E7 | 讲 MTP 时，有没有区分"训练时预测 n 个 token"与"推理时要出 γ 个草稿"？ |
| E8 | 用 $(1-\alpha^{\gamma+1})/(1-\alpha)$ 时，有没有提 i.i.d.？ |

### 第 30 篇（附录）

见配套文件 **RS-2b-优秀文章附录原料.md**。

---

## 7. 本轮调研的缺口（留给下一轮）

1. **视频类来源完全未覆盖**（WebSearch 配额 200/200 用尽）：Trelis Research、Efficient NLP、Umar Jamil、Yannic Kilcher、Stanford CS336、MIT 6.5940、B 站 UP 主。**需要单独一轮。**
2. **知乎 / CSDN / 微信公众号在本环境不可抓**。若必须覆盖，建议改用浏览器工具（claude-in-chrome）走真实会话。
3. **COLING 2025 tutorial 的 slides 与录像未打开**（tinyurl 未跟进）。这份可能是最好的体系化教学材料，**下一轮优先。**
4. **交互式 / 可视化讲解一份都没找到**。投机解码的树注意力 mask 极适合做可视化，社区若真的没有，**这本身就是本库可以自己补的空白（`_lab/tree.py` 输出 mask 图）**。
5. **HF `transformers` 里 assisted generation 是否已迁出核心 API**，需要核到迁移公告或 changelog 才能写进正文。
6. **EAGLE 在 $T>0$ 下到底是 L1 还是 L2**，README 自相矛盾，**必须读源码才能下结论**。
7. **Karpathy 那条推文**若要引用，需要通过可访问的镜像或截图核验原文。
