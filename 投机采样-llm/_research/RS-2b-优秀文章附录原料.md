# RS-2b 优秀文章附录原料

> 调研笔记，**不参与双链**。这是第 **30-附录-优秀文章与资料清单** 的原料库。
> 与 `RS-2-社区讲解盘点与评点.md` 是姊妹文件：**RS-2 讲"每份材料好在哪、错在哪"，本文件讲"按什么顺序读、为什么排在这个位置"。**
> 所有 URL 均于 **2026-08-22** 实际访问；访问失败的集中列在 §7，**不进入任何推荐序列**。

---

## 0. 使用说明与三条编排原则

### 0.1 本附录的编排原则

1. **按用途分组，不按类型分组。** 读者来附录不是想知道"有哪些博客"，而是想知道"我现在这个问题该读哪一篇"。所以分组是 ①入门 ②数学 ③动手 ④生产 ⑤前沿，不是"论文/博客/代码"。
2. **每条必须回答"为什么排在这个位置"**，而不只是"它讲了什么"。排序理由要具体到"它接住了上一条留下的哪个问题"。
3. **每条必须带一条"读的时候要小心什么"。** 本库的立场是社区材料要分层读（散文层 / 公式层 / 代码层经常打架），附录不能只推荐不设防。

### 0.2 全库通用的口径标签

在每条推荐里用这三个标签快速标注质量：

- **【口径全】**：加速比同时给了模型组合 / batch 或并发 / γ / 硬件 / 精度 / 指标定义 / 任务，或者根本不报数字。
- **【口径缺】**：报了数字但缺项，**引用时必须加"口径不全，不可横向比较"**。
- **【口径错】**：把延迟类指标叫成吞吐，或把不同来源的加速比并排比较。

### 0.3 「无损口径」标签（对应本库铁律一）

- **L1**：分布无损（输出是目标分布 $p$ 的精确样本；同 prompt 两次结果可以不同）
- **L2**：贪心等价（$T=0$ 时逐 token 相同）
- **L3**：近似（改了分布）
- **L?**：材料自己没说清

---

## 1. 推荐阅读顺序

### ① 第一次入门：4 条，约 2 小时

> **目标**：看完能回答三个问题 —— 为什么 decode 慢、为什么并行验证几乎免费、"无损"到底保证了什么。

| 序 | 材料 | URL | 为什么排在这个位置 |
|---|---|---|---|
| 1 | **Looking back at speculative decoding**（Google Research，Leviathan / Kalman / Matias，2024-12-06） | https://research.google/blog/looking-back-at-speculative-decoding/ | **排第一是因为它是发明人自己写的科普，而且叙述顺序天然正确**：先讲经典投机执行（$f$、$f^*$、$g$）→ 再讲"LLM 输出的是分布不是值" → 才引出 speculative sampling。√7 的例子（抄一个 7 很容易、算 2.646 很难）三十秒就让人明白"有些 token 是白送的"。**它同时给了 CPU 分支预测这个心智模型，后面所有材料都会默认你已经有了它。**<br>【口径缺】"~2x–3x"、"11B T5-XXL + 60M T5-small → ~3x" 均无 batch / γ / 硬件。<br>⚠ **小心**："We are guaranteed identical outputs either way." 这句出现在**确定性投机执行**那一段，**不是在讲 LLM 采样**；讲 LLM 采样时它用的是精确措辞 "exactly the same probability distribution"。这句被大量断章取义引用。 |
| 2 | **Accelerating Generative AI with PyTorch II: GPT, Fast** 的 Step 3 一节（Team PyTorch，页面 last updated 2024-11-14） | https://pytorch.org/blog/accelerating-generative-ai-2/ | **排第二是因为它提供了全网最好的比喻（Verity/Drake），而且这个比喻恰好补上第 1 条没讲透的那一点**：为什么被拒绝之后**后面的全部作废**（Drake 的技术决策是链式依赖的）。此外它前半篇已经把 MBU（Model Bandwidth Utilization）算到 72%，**读者能看清投机解码是被带宽墙逼出来的最后一招**，而不是一个孤立技巧。<br>【口径全】全文明写 "batch size=1"、"A100-80GB, power limited to 330W"；给了两个对比数字：CodeLlama-34B+7B → **2×**，Llama-7B+TinyLlama-1B → **约 1.3×**。<br>⚠ **小心**：**这一篇的散文是错的，代码是对的。** 散文说 "mathematically identical results"（L1/L2 混淆）和 "throwing out the ones that don't match"（丢掉了 $\min(1,p/q)$ 的随机性）；但 `generate.py` 里 `torch.minimum(torch.ones(()), q/p)` 和 `max(0, q-p)` 归一化全都正确。**入门阶段只看比喻，别记那两句话。** |
| 3 | **Speculative Sampling**（Jay Mody，2023-02-08） | https://jaykmody.com/blog/speculative-sampling/ | **排第三是因为它专治第 2 条留下的那个错误认知。** 它用一句话解决了整个社区最高频的误解：<br>> "Speculative sampling giving different result than autoregressive sampling is akin to running autoregressive sampling but **with a different seed**."<br>并补了例外（`temperature=0` 时才逐 token 相同）。**"换了个随机种子"这个说法，是把 L1 讲给程序员听的最短路径。** 同时它的 numpy 实现只有几十行，入门阶段第一次看到正确的接受判据与残差分布。<br>【口径全】不报加速比数字。**无损口径：L1，且主动区分了 L1/L2。**<br>⚠ **小心**：记号是 DeepMind 系（`p`=草稿、`q`=目标），**与 Leviathan 论文相反**，看代码时别以为分子分母写反了。无损性它没自证，甩链接给了论文 Theorem 1。 |
| 4 | **vLLM 官方文档 Speculative Decoding 的"Lossless guarantees"部分**（核验 2026-08） | https://docs.vllm.ai/en/latest/features/speculative_decoding/ | **排第四是收口。** 前三条给了直觉，这一条给正式口径，而且是本次核到最严谨的一份：<br>> "Speculative decoding sampling is **theoretically lossless up to the precision limits of hardware numerics**."<br>> "vLLM's implementation ... is **algorithmically validated** to be lossless."<br>> "**Changes in batch size may cause variations in logprobs and output probabilities.**"<br>**三句话正好三层：理论层 / 实现层 / 数值层。** 尤其最后一句，是入门阶段就该知道的一记警钟——**即使算法无损，真实系统里 batch 组成变化也会通过浮点归约顺序改变结果。**<br>【口径全】不报加速比。**无损口径：L1，且把限定条件全写了。** |

**入门后应当能判定的三件事**：(a) "输出不一样"不是 bug；(b) 只有 $T=0$ 才谈得上逐 token 相同；(c) 任何不给 batch 的加速比数字都不可信。

**入门阶段明确不推荐**：Wikipedia 的 Speculative decoding 词条（说 "produces the same results as standard decoding"，直接教错）、NVIDIA 的 An Introduction to Speculative Decoding（同一句里既取消了随机性又把分布相同说成结果相同）。理由见 §6。

---

### ② 想搞懂数学：5 条

> **目标**：能自己推出无损性、能算 $\alpha$、能判断"这笔买卖赚不赚"。

| 序 | 材料 | URL | 为什么排在这个位置 |
|---|---|---|---|
| 1 | **Fast Inference from Transformers via Speculative Decoding**（Leviathan / Kalman / Matias；v1 2022-11-30，末版 2023-05-18） | https://arxiv.org/abs/2211.17192<br>全文 HTML：https://ar5iv.labs.arxiv.org/html/2211.17192 | **数学线必须从这里开始，因为整条链上的四个量都是它定义的**：<br>- §2.3 接受与修正："if $q(x)>p(x)$ we reject the sample with probability $1-\frac{p(x)}{q(x)}$ and sample $x$ again from an adjusted distribution $p'(x)=norm(max(0,p(x)-q(x)))$"<br>- Definition 3.1：$\beta_{x_{<t}}$ = 给定前缀时接受 $x_t\sim q$ 的概率；$\alpha=E(\beta)$<br>- Corollary 3.6：$\alpha=1-E(D_{LK}(p,q))=E(\min(p,q))$　**←把"两个模型有多像"变成可计算量的那一步**<br>- Theorem 3.8：墙钟提升 $=\dfrac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$；**Corollary 3.9：若 $\alpha>c$ 则存在 $\gamma$ 使其获益，且提升至少 $\frac{1+\alpha}{1+c}$**<br>**$\alpha>c$ 是全库最重要的一条判据，必须从原文读到。**<br>⚠ **小心两点**：(a) **摘要与正文口径不一致**——摘要写 "without any changes to the outputs" / "with identical outputs"，正文写 "without changing the model output **distribution**"。**社区那个最大误解的源头就在这里。**(b) $E[\tau]$ 的推导**明写了 i.i.d. 前提**："If we make the simplifying assumption that the $\beta$s are i.i.d., ... a **capped geometric variable**"。**看到这句就明白：丢前提的是社区，不是论文。** |
| 2 | **Accelerating Large Language Model Decoding with Speculative Sampling**（Chen / Borgeaud / Irving / Lespiau / Sifre / Jumper，DeepMind，v1 2023-02-02） | https://arxiv.org/abs/2302.01318<br>全文 HTML：https://ar5iv.labs.arxiv.org/html/2302.01318 | **排第二是因为它是同一件事的第二个视角，而且补上了第 1 条没强调的两条边界**：<br>> "**At least one token will always be generated** from a draft-accept loop – if the first token is rejected, a valid token is resampled."<br>> "if every drafted token is accepted, we can sample from it normally. This gives us a **maximum of $K+1$ tokens per loop**, over the naive implementation which would only return $K$ tokens."<br>**第二条就是 vLLM 源码里的 "bonus token"，手写实现最容易漏。**<br>它还发明了那个后来被所有人继承的限定词："preserves the distribution of the target model **within hardware numerics**"。<br>⚠ **小心**：**记号与第 1 条完全相反**——Algorithm 2 原文 "Given auto-regressive **target model $q(.|.)$**, and auto-regressive **draft model $p(.|.)$**"。**读这两篇的顺序里必须夹一张记号对照表，否则公式全看反。**<br>【口径缺】"2-2.5x in a distributed setup" 未给芯片数 / batch / K。<br>🎁 花絮：附录 Author Contributions 明写 "**Modified Rejection Sampling Scheme: John Jumper**"。 |
| 3 | **An Optimal Lossy Variant of Speculative Decoding**（Vivien Tran-Thien，2024-06-12） | https://huggingface.co/blog/vivien/optimal-lossy-variant-of-speculative-decoding | **排第三是因为它回答了前两篇留下的那个问题："既然无损是靠 $\min(1,p/q)$ 卡住的，放宽它会怎样？"** 它把这件事做成了一个带约束的最优化问题（在 $D(q\|\pi)\le D$ 的 KL 预算下最大化接受概率），并给出闭式解（记号为 DeepMind 系）：<br>$r_i=\min\!\left(1,\frac{q_i}{\alpha p_i}\right)$，$s_i=\frac{1}{1-\sum_j p_j r_j}\max\!\left(0,\frac{q_i}{\beta}-p_i\right)$，$\pi_i=p_i r_i+s_i(1-\sum_j p_j r_j)$<br>**令 $\alpha=\beta=1$ 就退回标准投机采样。** 这一步把"无损/有损"从二元属性变成了**一条连续曲线上的位置**——Medusa 的 typical acceptance、各种阈值放宽，本质都是在这条曲线上往右挪，只是它们没算过挪了多远。<br>用了 KKT 条件、给了唯一解存在性与二分搜索算法、链了形式化证明、在 WMT15 上做了实验。**是本次核到最严谨的一份社区数学材料，也是最被埋没的一份。**<br>⚠ **小心**：约束用的是 **KL** 而不是 TV，而 $\alpha=1-\mathrm{TV}$ 用的是 TV，**两者不能直接换算**。另外它给了旋钮没给刻度（KL 预算取多少没答）。 |
| 4 | **Spec-Bench 榜单**（hemingkx，榜单最后更新 2025-04-22） | https://github.com/hemingkx/Spec-Bench<br>榜单原文：https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/Leaderboard.md | **排第四是因为数学学到这里必须落回数字，而这是唯一一份口径齐全到可以直接用来验证公式的榜单。** 口径写在榜单头上：**"a single NVIDIA A100 GPU (80GB) with 96 CPU cores"、Pytorch 2.5.1、Vicuna-7B/13B/33B-v1.3、greedy decoding、FP16、batch size = 1**。<br>**它引入 #MAT（Mean Accepted Tokens）与加速比并列** —— 这正好把 $E[\tau]$（草稿质量）与墙钟提升（含实现效率）分开，是验证 Theorem 3.8 的现成数据。<br>**最该盯的一行**：A100 / Vicuna-7B 上 **Recycling 的 #MAT 只有 2.73，却拿到 2.22×**；而 EAGLE2 的 #MAT 是 4.34、只拿到 2.36×。**接受长度差 1.6 倍，加速比只差 6%** —— 这就是 $c$（草稿成本）在起作用，是 Corollary 3.9 最好的实证注脚。<br>【口径全】<br>⚠ **小心**：全部是 **batch=1 + greedy**，因此它**测不出**大 batch 负收益与 $T>0$ 的接受率变化；且榜单未覆盖 2025 下半年以后的方法。 |
| 5 | **vLLM 源码 `vllm/config/speculative.py`**（main 分支，核验 2026-08-22） | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/speculative.py | **排最后，因为它是"数学在工程里被怎么改写"的答案，读完前四条才看得懂它在拒绝什么。** 三处关键：<br>**(a) α 的工程定义是"边际"而非"条件"**：<br>> "Per-position ***unconditional*** acceptance rates ... Position i's entry is the **marginal probability that the first i+1 draft tokens are all accepted**; ... must be **monotonically non-increasing**."<br>这与 Leviathan 的条件概率定义**只有在 i.i.d. 下才互换**。<br>**(b) vLLM 明确不用几何模型**：`_acceptance_length_to_rates` 的 docstring 写 "using the **minimum-variance schedule**"，实现是 `[1.0]*num_full + [frac] + [0.0]*...`，**根本不是 $\alpha^i$ 衰减**。<br>**→ 这是"$E[\tau]=(1-\alpha^{\gamma+1})/(1-\alpha)$ 不是恒等式"最硬的证据。**<br>**(c) 草稿默认贪心、概率按 one-hot 处理**：<br>> `draft_sample_method: DraftSampleMethod = "greedy"` ... "the draft probabilities are treated as **one-hot** during rejection sampling"；`'probabilistic'` 模式 "comes at the cost of **additional GPU memory usage**"。<br>**即：生产系统默认根本不用草稿的完整分布。** 这仍是 L1，但接受率的数学完全不同。<br>⚠ **小心**：源码不解释"为什么这个调度方差最小"，要用就得自己验证。 |

---

### ③ 想动手实现：5 条

> **目标**：从 50 行的最短正确实现，到能读懂生产 kernel。
> **重要提示**：本次逐行核对了这一组的接受判据与残差分布，**全部正确**。唯一的陷阱是第 4 条里的一个开关。

| 序 | 材料 | URL | 为什么排在这个位置 |
|---|---|---|---|
| 1 | **gpt-fast 的 `speculative_decode`**（约 50 行原生 PyTorch） | https://raw.githubusercontent.com/pytorch-labs/gpt-fast/main/generate.py | **排第一因为它是"最短的正确实现"**，且被官方博客用一句话背书："Here's the entirety of the implementation, in about **50 lines** of native PyTorch."<br>关键三行（记号 `p`=draft、`q`=target）：<br>```python<br>accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)<br>rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()<br>new = q - p; new = torch.where(new > 0, new, 0.0); new = new / new.sum()<br>```<br>**三要素齐全：`min(1,·)`、`torch.rand_like`（随机性）、归一化残差。**<br>⚠ **小心**：**别读它的博客散文**（见 ①-2 的警告）。另外它带 KV cache 回滚与 `torch.compile`，第一次读要先跳过这两块。 |
| 2 | **Jay Mody 的 numpy 版**（2023-02-08） | https://jaykmody.com/blog/speculative-sampling/ | **排第二是因为它把第 1 条的 PyTorch 骨架翻译成了"没有框架也能懂"的版本**，且博客里 `autoregressive_sampling()` 与 `speculative_sampling()` 并排放着，**可以直接对拍**。<br>```python<br>if np.random.random() < min(1, q[i][j] / p[i][j]):   # accepted<br>...<br>sample(max_fn(q[i] - p[i]))     # max_fn: x_max = np.where(x>0,x,0); x_max/np.sum(x_max)<br>```<br>2023-04-13 的修订说明也值得看（去掉了一次多余的草稿模型前向），**是"实现里最容易多跑一次前向"这个坑的现成记录**。<br>⚠ **小心**：`p`=草稿、`q`=目标（与第 1 条同系，但与 Leviathan 论文相反）。 |
| 3 | **feifeibear/LLMSpeculativeSampling**（923 stars，last update 2023-09-21） | https://github.com/feifeibear/LLMSpeculativeSampling<br>核心：https://raw.githubusercontent.com/feifeibear/LLMSpeculativeSampling/main/sampling/speculative_sampling.py | **排第三的唯一理由：它是唯一一个把两篇论文的算法做成两份代码并排放的仓库。** README："The speculative sampling is proposed by Google and Deepmind independently. So I implement **two slightly different versions** of speculative sampling: Google's and Deepmind's."<br>**直接 diff 这两个函数，就能亲眼看到两篇论文的差异在哪（KV cache 回滚方式、最后一个 token 的处理），比读一万字对比文章都快。**<br>另外两版分别演示了"写成拒绝"和"写成接受"两种等价写法：<br>```python<br># v1: if r > p_target/p_draft:  → 拒绝<br># v2: if r < torch.min(torch.tensor([1]), p/q):  → 接受<br>```<br>【口径缺 / 近乎口径错】README 的 benchmark（7b 1084.86 / 70b 329.83 / spec 427.02）**没有硬件、没有 γ、没有 batch，单位也存疑**，按面值只有 1.29×。**只读代码，不要引用它的数字。**<br>⚠ 2023-09 后基本停更，无树、无 EAGLE。 |
| 4 | **romsto/Speculative-Decoding**（115 stars） | https://github.com/romsto/Speculative-Decoding<br>核心：https://raw.githubusercontent.com/romsto/Speculative-Decoding/main/sampling/speculative_decoding.py | **排第四是因为它做了前三条都没做的事：把自回归、beam search、投机解码、NASD（N-gram 辅助）放在一个仓库里，可以直接对拍。** README 的措辞也精确："without **changing the output distribution**"。<br>**⚠⚠ 但它带一个必须警惕的开关**：<br>```python<br>if not skip_sample_adjustment:<br>    p_p = max_fn(p[..., n, :] - q[0, n, :])<br>else:<br>    p_p = p[..., n, :]          # ← 直接从目标分布重采：这正是社区第 3 号高频错误<br>```<br>**`skip_sample_adjustment=True` 就是"拒绝后直接从 $p$ 重采"这个经典错误的可执行版本，而 README 没有任何警告。**<br>**两面看**：作为**消融开关**它非常有价值（可以实测跳过残差修正会把分布偏成什么样）；作为默认可用 API 它是陷阱。**本库 `_lab/test_lossless.py` 应该做同样的对照实验。** |
| 5 | **vLLM 的 `rejection_sampler.py`**（main 分支，核验 2026-08-22） | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/sample/rejection_sampler.py | **排最后，因为它展示了"数学写法"与"生产写法"的三处真实差距**，读完前四条才有对照。<br>docstring 先钉死出处："The implementation **strictly follows** the algorithm described in https://arxiv.org/abs/2211.17192."<br>**差距一：没有显式 `min`**——`accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob`（比值 ≥1 时不等式自动成立，`min` 是多余的）。<br>**差距二：残差不做归一化，改用 Gumbel-max**——`prob = tl.maximum(target_prob - draft_prob, 0.0); score = prob * inv_q`，然后取 argmax。**省掉了 $\sum$ 归一化，也省掉了逐位置的多项式采样。**<br>**差距三：术语更精确**——accepted / **recovered**（拒绝后补采的）/ **bonus**（全接受时目标模型白送的那个）。**本库统一采用这套术语。**<br>⚠ **小心**：Gumbel-max 与显式归一化在浮点上不严格等价，这与官方文档那句 "up to the precision limits of hardware numerics" 是同一件事的两面。 |

**动手阶段的一个建议练习（本库 `_lab/` 应当覆盖）**：拿第 1 或第 2 条的实现，做三个消融——(a) 把 `min(1,p/q)` 换成 `p > q`；(b) 把残差换成直接采 $p$（即第 4 条的开关）；(c) 把 $\gamma$ 从 1 扫到 16。**跑经验分布对拍，看前两个消融把分布偏成什么样、第三个的收益曲线在哪里拐头。**

---

### ④ 想上生产：6 条

> **目标**：能回答"我这个部署到底该不该开投机、开了会不会亏、参数怎么调"。
> **这一组的共同价值：它们全都给了失效条件。教学向材料几乎没有一份给。**

| 序 | 材料 | URL | 为什么排在这个位置 |
|---|---|---|---|
| 1 | **How Speculative Decoding Boosts vLLM Performance by up to 2.8x**（vLLM Team，2024-10-17） | https://vllm.ai/blog/2024-10-17-spec-decode | **排第一是因为它是唯一一份把"赚的数字"和"亏的数字"并排放出来的官方博客**，一进门就把期望值校准好：<br>- 赚：Llama3-70B + Qwama-0.5B 草稿，ShareGPT，**4×H100，QPS=1 → 1.5×**；n-gram，CNN/DailyMail，**QPS=1 → 2.8×**<br>- 亏：**同样配置在高 QPS 下 ShareGPT 慢 1.4×、CNN/DailyMail 慢 1.8×**<br>机制解释也对："The extra compute required to propose and verify tokens can sometimes slow down the system **when it is already compute-bound**."<br>【口径全（相对而言）】给了模型对、数据集、GPU 数、QPS。缺 γ。<br>⚠ **小心**：它把 lossless 写成无口径的一句话（**L?**）。 |
| 2 | **vLLM Issue #10318：博客数字不可复现** | https://github.com/vllm-project/vllm/issues/10318 | **必须紧跟第 1 条读，这是本附录里最重要的一次"打预防针"。** 有人拿着官方配置去复现：vLLM 0.6.3、**4×H100 PCIe**、Llama-3-70B（TP=4）+ Qwama-0.5B（TP=1）、ShareGPT、**1 QPS、4 个投机 token** ——<br>**实测最高只到 1.4×（bs=1 时）**；bs=256 时端到端延迟 "exceeded 20 seconds"。**issue 被打 stale 标签，90 天后以 "not planned" 关闭，官方未补实验细节、未回应差距。**<br>**读完这一条，你会永久性地不再相信任何不带完整口径的加速比。**<br>⚠ 严格说这是"口径不全导致无法复现"（博客未注明 H100 是 SXM 还是 PCIe，PCIe 带宽显著更低），**不能据此判定博客造假**。引用时必须这样表述。 |
| 3 | **A Hitchhiker's Guide to Speculative Decoding**（PyTorch / IBM，页面 last updated 2024-11-13） | https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/ | **排第三是因为它是本次核到唯一一份"真实生产部署报告"**："deployed these speculators in an internal production-grade environment with **thousands of daily users**"。三条可直接抄进运维手册的结论：<br>**(a) 一个具体的失效拐点**：<br>> "We begin to observe **throughput reduction beyond a batch size of 64**, which happens rarely in practice."<br>**(b) γ 与任务类型绑定**："we find **3-4 heads works well** in practice, whereas we found that **code models can reap benefits from 6-8 heads**"。<br>**(c) 指标选得对**：报的是 **TTFT 与 ITL 随并发用户数变化的两张曲线**，不是笼统的 tok/s。<br>另外给了草稿头训练配方（两阶段：4k 长序列小 batch 走 causal LM → 256 短序列大 batch 对齐基座输出，**5:2 步数配比**），是第 21 篇的现成材料。<br>⚠ **小心**：开篇踩了"输出完全一致"（**L2 当 L1 说**）；"bs>64 很少见"是 2024 年 IBM 内部负载的判断，**2026 年已不成立**。 |
| 4 | **TensorRT-LLM 官方文档 Speculative Decoding** | https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/speculative-decoding.html | **排第四是因为它最诚实地暴露了引擎的真实约束，而这些约束在厂商博客里一个字都不会写。** 三句话值回票价：<br>> "A draft token is accepted **if matches the previously decoded token exactly**."<br>> "**only greedy sampling is supported** for speculative decoding"<br>> "There is currently **no way to dynamically disable speculation**, thus **speed ups are only observable at low batch sizes**."<br>**它明确承认自己是 L2**，同时给了"为什么会亏（不能动态关）+ 什么时候亏（大 batch）"。<br>还有隐性代价：two-model 模式 "do not support **overlap scheduler**. It will be disabled automatically."<br>**无损口径：L2（文档自己说清楚了）。**<br>⚠ **小心**：MTP 的 `use_relaxed_acceptance_for_thinking` / `relaxed_topk` / `relaxed_delta` **是有损开关，文档没有任何标注**。 |
| 5 | **SGLang 官方文档 Speculative Decoding** | https://docs.sglang.io/advanced_features/speculative_decoding.html | **排第五是因为要调树形草稿，只有这份文档把三个旋钮的语义分开写清楚了**：<br>- `--speculative-num-steps`："**Depth** of autoregressive drafting"<br>- `--speculative-eagle-topk`："**Branching factor** per step"<br>- `--speculative-num-draft-tokens`："Maximum **parallel verification capacity**"<br>**深度 / 宽度 / 总预算 三维分离**，读完才知道"调树"到底在调什么。默认值还按模型族分叉（Llama/Grok：steps=5, topk=4, draft-tokens=8；其他：3/1/4），**这本身说明接受率高度依赖模型族**。<br>隐性代价也写了：`topk>1` 会**关掉 overlap scheduler**；ngram 会 "disables the overlap scheduler & mixed chunked prefill"。<br>【口径缺 / 近口径错】1×H100、MT-bench、LLaMA-3.1-8B：baseline **158.34 tok/s** → EAGLE-2 **244.10** → EAGLE-3 **373.25**，**未标 batch**（按 MT-bench 惯例应为 bs=1），却用 "throughput gains" 描述。<br>⚠ **小心**：**完全没有关于 temperature / 是否无损的表述**（**L?**），示例全用 `temperature=0`。 |
| 6 | **Performance improvements with speculative decoding in vLLM for gpt-oss**（Red Hat，Harshith Umesh，2026-04-16） | https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss | **排最后是因为它推翻了前五条读者刚建立起来的直觉，必须最后读、带着前面的判据来读。** 它给出了**高并发下投机依然赚**的反例，而且口径是本组最全的：<br>- 目标 `openai/gpt-oss-120b`（**MoE + MXFP4**）+ 草稿 `nvidia/gpt-oss-120b-Eagle3-v2`（EAGLE3）<br>- **H200-PCIe-141GB**，TP=1/2，**并发 1/5/25/50/100/200**，GuideLLM v0.5.3 + vLLM v0.13.0<br>- ShareGPT：峰值 **2574 vs 2024 tok/s（+27.2%）**，几何平均 **+20.7%**，且 "improvements that **persist with up to 200 concurrent requests**"<br>- MLPerf（长 prompt）TP=1 **+9.5%** / TP=2 **+16%**；SWE-bench **+20.5%**（并发 100 时 +24%）<br>- γ 的边际成本："**2 or 3 draft tokens is the sweet spot**"；"going from 3 to 4 draft tokens causes a modest **8% output throughput drop**"<br>**→ 结论要改写成条件式**："大 batch 一定亏"不成立；正确的表述是**收益取决于 (激活参数/总参数比、权重精度、草稿接受率) 三者**。MoE + 低比特量化 + 高接受率 EAGLE3 这个组合，把 compute-bound 的拐点推得很远。<br>⚠ **小心**：厂商博客，结论对自家栈有利；且 MoE+MXFP4 **不能推广到稠密 fp16 模型**。它也踩了 "no change to output quality"（**L?**）。 |

**生产阶段的决策清单（可直接做成第 19 篇的表）**：

| 问题 | 去哪条找答案 |
|---|---|
| 我的并发下还赚吗 | ④-1（vLLM 高 QPS 反向数字）、④-3（bs>64 拐点）、④-6（MoE 例外） |
| 引擎支不支持采样 | ④-4（TensorRT-LLM 只支持贪心）、④-5（SGLang 未表态） |
| 开了投机会关掉什么 | ④-4、④-5（overlap scheduler） |
| γ 取几 | ④-3（3-4 / 6-8 按任务）、④-6（2-3，第 4 个掉 8%） |
| 别人的加速比能不能信 | ④-2（不可复现案例）、②-4（Spec-Bench 才是可比的） |

---

### ⑤ 想跟前沿：6 条

> **目标**：知道 2024→2026 这条线上什么变了、什么死了、下一步在哪。

| 序 | 材料 | URL | 为什么排在这个位置 |
|---|---|---|---|
| 1 | **hemingkx/SpeculativeDecodingPapers**（1.3k stars，500+ 篇，2018 → 2026-06） | https://github.com/hemingkx/SpeculativeDecodingPapers<br>README 原文：https://raw.githubusercontent.com/hemingkx/SpeculativeDecodingPapers/main/README.md | **排第一是因为跟前沿的第一步是拿到地图。** 它的分类法本身就是一张谱系：<br>`Survey` / `Seq2Seq` / `LLMs` / **`Multi-Token Prediction`** / **`Diffusion LMs`** / `Multimodal` / **`Long-Context`** / **`Mixture-of-Experts`** / `Alignment` / `Benchmarks` / `Applications` / **`Analysis`** / `Other Techniques`<br>**三个分类值得单独注意**：`for MoE`（呼应 ④-6）、`Long-Context`、以及 **`Analysis` 已经自成门类**——说明"分析投机解码到底赚不赚"已经是一条独立研究线，**这正是本库的定位**。<br>⚠ **小心**：只列不评，**没有"哪些分支已经死了"的标记**——这恰恰是本库第 15 篇要补的增量。README last update 未取到，标「未查证」。 |
| 2 | **EAGLE 官方仓库**（SafeAILab，2.5k stars，README last update 2025-09-18） | https://github.com/SafeAILab/EAGLE | **排第二是因为 EAGLE 三代是这条线上唯一一个"每一代都推翻上一代核心假设"的谱系**，README 把差异压成了三句：<br>- EAGLE-1：外推**倒数第二层的上下文特征向量**<br>- EAGLE-2：用草稿的 **confidence 近似接受率**，据此**动态调整草稿树结构**<br>- EAGLE-3：**"Removes the feature prediction constraint in EAGLE"**，改用 **training-time test**，并把顶层特征换成**低/中/高层语义的融合**<br>**"EAGLE-3 把 EAGLE-1 的核心机制删掉了"——这是全库最干净的一条"被推翻"记录。**<br>⚠ **小心**：**README 的 Todo 里仍写着 "Support non-greedy inference (provably maintaining text distribution)"（2026-08 核验）**，与正文那句 "provably maintaining the consistency ... in the **distribution** of generated texts" 并列出现，**自相矛盾，无法判定 $T>0$ 下是 L1 还是 L2**（**L?**）。<br>【口径缺】"3x faster (13B)"、Vicuna 13B、**2×RTX 3090**、fp16，**无 batch、无 temperature、无树规模**。 |
| 3 | **SpecForge: Accelerating Speculative Decoding Training for SGLang**（SGLang Team / LMSYS，2025-07-25） | https://www.lmsys.org/blog/2025-07-25-spec-forge/ | **排第三是因为它标志了这条线的重心转移：从"算法"转到"怎么把草稿模型训出来"。** 立论句：<br>> "the lack of robust open-source tools for **training draft models**—a key component of this process—has significantly hindered its adoption."<br>两条硬信息：<br>- **离线模式要 "~12TB for UltraChat + ShareGPT"** 的磁盘来存 hidden states（在线模式省磁盘但要更多 GPU）。**这个数字让"训草稿的成本"变得具体。**<br>- EAGLE-3 的 TTT "makes the draft model robust by **simulating multi-step generation**"，实现难在 "specialized attention masks and recursive data loops"。<br>【口径缺】Llama 4 Scout **2.0×** / Maverick **2.18×**（MT-Bench），**无硬件、无 batch、无接受长度**；只给了 `speculative-eagle-topk=8`、`speculative-num-draft-tokens=10`。 |
| 4 | **Fastest Speculative Decoding in vLLM with Arctic Inference**（Snowflake，2025-05-01） | https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/ | **排第四是因为它同时代表了两个 2025-26 的新方向，而且贡献了本库最锋利的一句引文。**<br>**方向一：草稿可以不要 GPU** —— suffix decoding 在 CPU 上 "**20 microseconds per token**"，靠历史输出与当前输入里的重复结构建投机序列。<br>**方向二：agent 场景是投机解码的新主场** —— SWE-Bench 上跑 CodeAct agent（OpenHands LM 32B）端到端 **1.8×–4.5×**，因为 agent 轨迹里重复文本极多。<br>**那句引文**（本库第 07 篇必用）：<br>> "**Switched from rejection sampling to greedy verification** (accept tokens only if they match greedy decoding), ensuring **outputs are identical to the base model** without lowering acceptance rate."<br>**它证明 L2 不是"退化的 L1"，而是一个被生产团队主动选择的需求（可复现性）。**<br>【口径较全】8×H100、TP=2、FP8、0.5 req/s；ShareGPT 179 / HumanEval 217 / 混合 209 tok/s。SWE-Bench 那条无硬件。<br>⚠ **小心**："without lowering acceptance rate" 是他们的实测，**不是定理**（贪心验证的接受概率数学上不可能高于拒绝采样）。 |
| 5 | **vLLM `config/speculative.py` 的方法清单与新参数**（main，核验 2026-08-22） | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/speculative.py | **排第五是因为"哪些方法活下来了"这个问题，最诚实的答案在生产系统的枚举类型里，不在论文里。** 2026-08 快照：<br>`SpeculativeMethod = ngram / medusa / mlp_speculator / draft_model / suffix / custom_class / eagle / eagle3 / extract_hidden_states / dflash / <MTP 全家> / ngram_gpu / dspark`<br>**三条趋势读得出来**：<br>**(a) MTP 已成新模型标配** —— `MTPModelTypes` 列了 **27 种**模型特化 MTP（deepseek / glm4_moe / qwen3_next / kimi_k3 / minimax_m3 / gemma4 / ernie / longcat_flash / pangu_ultra_moe / step3p5 …）。**"自带草稿头"从论文技巧变成了模型发布的默认配件。**<br>**(b) 逐 token 验证不再是唯一方案** —— `RejectionSampleMethod = "standard" | "synthetic" | "block"`，其中 block 的注释是 "**block verification (Sun et al.)**, which jointly verifies the draft tokens **as a block instead of one at a time**"。<br>**(c) 自适应化** —— `enable_adaptive_verification`（"adaptively size the draft-verification budget from per-request confidence"，目前仅 `dspark`）、文档侧的 Dynamic Speculative Decoding、PARD。<br>**另外一条对 MTP 极重要的告警**：<br>> "Enabling `num_speculative_tokens > 1` will run multiple times of forward on same MTP layer, **which may result in lower acceptance rate**"<br>**这正是"MTP 的训练目标 ≠ 推理期草稿用法"在生产里的具体表现。** |
| 6 | **COLING 2025 Tutorial：Speculative Decoding for Efficient LLM Inference** | https://speculative-decoding.github.io/<br>slides：https://tinyurl.com/speculative-decoding-tutorial （**本次未跟进跳转，未核验**）<br>录像：https://tinyurl.com/spec-tutorial-recording （**未核验**） | **排最后是因为它是唯一一份体系化教学材料，适合当"跟完前沿之后回头做整理"的收口。** 主讲：Heming Xia、Yongqi Li、Wenjie Li（香港理工大学）、Cunxiao Du（SEA AI Lab）、Qian Liu（TikTok）；2025-01-19 于 Abu Dhabi National Exhibition Centre, Capital Suite 7, 09:00–12:30 GST。<br>议程：`Introduction & Definition (40min)` → `History and Taxonomy of Methods (45min)` → `Cutting-edge Algorithms (40min)` → `Downstream Adaptations (30min)` → `Final Remarks & Q&A (20min)`。**同一批人也是 ACL Findings 2024 那篇综述（arXiv:2401.07851）的作者。**<br>⚠ **小心**：**本次只读到主页，slides 与录像未打开，因此不能对内容质量下结论**。主页对无损的表述是概括性的 "maintaining original distributions"，**未细分 L1/L2/L3**。 |

---

## 2. 完整清单（按用途分组，含未进推荐序列的条目）

### 2.1 奠基论文与官方口径

| 材料 | URL | 年月 | 一句话定位 | 口径 |
|---|---|---|---|---|
| Leviathan / Kalman / Matias《Fast Inference from Transformers via Speculative Decoding》 | https://arxiv.org/abs/2211.17192 | v1 2022-11-30，末版 2023-05-18 | $\alpha$、$E[\tau]$、$\alpha>c$ 判据的定义者 | 摘要 **L2 措辞**，正文 **L1** |
| Chen 等（DeepMind）《Accelerating LLM Decoding with Speculative Sampling》 | https://arxiv.org/abs/2302.01318 | v1 2023-02-02 | "within hardware numerics" 的发明者；bonus token 的出处 | **L1** |
| Google Research《Looking back at speculative decoding》 | https://research.google/blog/looking-back-at-speculative-decoding/ | 2024-12-06 | 发明人写的科普；√7 例子、分支预测类比 | **L1**（但有易被断章取义的句子） |
| Xia 等《Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding》 | https://arxiv.org/abs/2401.07851 | Findings of ACL 2024 | Spec-Bench 的配套综述（**本次未逐页核验正文，仅从 Spec-Bench README 核到引用信息**） | 未查证 |
| Wikipedia《Speculative decoding》 | https://en.wikipedia.org/wiki/Speculative_decoding | 修订日期未查证 | 史线日期可交叉校验；**内容口径有错，不作知识来源** | **踩 L1/L2 混淆** |

### 2.2 教学与图解

| 材料 | URL | 年月 | 一句话定位 | 口径 |
|---|---|---|---|---|
| Jay Mody《Speculative Sampling》 | https://jaykmody.com/blog/speculative-sampling/ | 2023-02-08 | "换了个随机种子"——L1 的最佳直觉解释 | **L1，主动区分 L1/L2** |
| PyTorch《Accelerating Generative AI II: GPT, Fast》 | https://pytorch.org/blog/accelerating-generative-ai-2/ | last updated 2024-11-14 | Verity/Drake 比喻；**散文错、代码对** | 散文踩 L1/L2 混淆 |
| Charles Frye《At the Intersection of LLMs and Kernels》 | https://charlesfrye.github.io/programming/2023/11/10/llms-systems.html | 2023-11-10 | scoring vs sampling 的代价差 | 未表态 |
| HF《Assisted Generation》（Joao Gante） | https://huggingface.co/blog/assisted-generation | 2023-05-11 | Inception 类比、"latency-free oracle model"；**明写 "limited to a batch size of 1"** | 未宣称无损（**诚实**） |
| HF《Speculative Decoding for 2x Faster Whisper Inference》（Sanchit Gandhi） | https://huggingface.co/blog/whisper-speculative-decoding | 2023-12-20 | **给了 batch=4 的明确拐点**（"Above batch size 4, speculative decoding returns slower inference than the main model alone"）；T4 16GB、bs=1、73→33s（2.2×）/117→62s（1.9×） | **踩 L1/L2 混淆**（"exactly the same outputs"） |
| NVIDIA《An Introduction to Speculative Decoding》 | https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/ | 2025-09-17 | "meticulous scientist / quick assistant" 比喻可取；**其余错误密集** | **同一句里既取消随机性又混淆 L1/L2** |
| LLM Inference Handbook（原 BentoML，现 Modular） | https://handbook.modular.com/inference-optimization/speculative-decoding | 日期未查证 | 方法索引 + 并发拐点（TP=1 时 **20–30 并发**提前饱和） | **踩 L1/L2 混淆**；无任何公式 |

### 2.3 数学与理论

| 材料 | URL | 年月 | 一句话定位 |
|---|---|---|---|
| Vivien Tran-Thien《An Optimal Lossy Variant of Speculative Decoding》 | https://huggingface.co/blog/vivien/optimal-lossy-variant-of-speculative-decoding | 2024-06-12 | 把 L1→L3 参数化成一条连续曲线；KKT + 二分搜索 + 形式化证明 |
| Leviathan 论文 §3（Analysis） | 见 2.1 | — | $\beta$/$\alpha$ 定义、Corollary 3.6、Theorem 3.8、Corollary 3.9 |
| Spec-Bench 榜单 | https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/Leaderboard.md | 榜单 2025-04-22 | #MAT 与加速比并报；**batch=1、greedy、FP16 全标注** |

### 2.4 引擎、源码与生产

| 材料 | URL | 年月 | 一句话定位 | 无损口径 |
|---|---|---|---|---|
| vLLM 官方文档 Speculative Decoding | https://docs.vllm.ai/en/latest/features/speculative_decoding/ | 核验 2026-08 | **三层无损声明 + batch 影响 logprobs 的警告**；方法全集 | **L1（限定最完整）** |
| vLLM `v1/sample/rejection_sampler.py` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/sample/rejection_sampler.py | main，核验 2026-08 | Gumbel-max 残差；accepted/recovered/bonus 术语 | — |
| vLLM `config/speculative.py` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/speculative.py | main，核验 2026-08 | α 的边际口径、最小方差调度、草稿默认贪心、MTP 告警 | — |
| vLLM 博客 2.8× | https://vllm.ai/blog/2024-10-17-spec-decode | 2024-10-17 | 赚与亏的数字并排 | **L?**（无口径 lossless） |
| vLLM Issue #10318 | https://github.com/vllm-project/vllm/issues/10318 | 2024-11 提出 | 官方 2.8× vs 独立复现 1.4×，以 stale 关闭 | — |
| Aleksa Gordić《Inside vLLM》 | https://www.aleksagordic.com/blog/vllm | 2025-08-29（vLLM 转载 2025-09-05） | 两情形式的判据描述；引擎全景 | 措辞 "in expectation" 不精确 |
| TensorRT-LLM 官方文档 | https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/speculative-decoding.html | 核验 2026-08 | **明说只支持贪心 + 只在小 batch 有效**；one-model vs two-model | **L2（自己说清楚）** |
| SGLang 官方文档 | https://docs.sglang.io/advanced_features/speculative_decoding.html | 核验 2026-08 | 树的三旋钮语义；overlap scheduler 冲突 | **L?** |
| PyTorch / IBM《Hitchhiker's Guide》 | https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/ | last updated 2024-11-13 | 真实生产部署；**bs>64 掉吞吐**；草稿头训练配方 | **踩 L1/L2 混淆** |
| NVIDIA《TensorRT-LLM Speculative Decoding Boosts Throughput by up to 3.6x》 | https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/ | 2024-12-02 | **405B 上 1B/3B/8B 草稿的倒 U 形数据**（证伪"越小越好"） | 未表态；**标题口径错** |
| AMD ROCm《Speculative Decoding - Deep Dive》（Chang Liu） | https://rocm.blogs.amd.com/software-tools-optimization/speculative-decoding---deep-dive/README.html | 2025-03-24 | MI300X 基准；request rate 衰减曲线（2.21×@0.2 → 2.06×@1.4 req/s）；长上下文 32768/128 → 2.98× | 未涉及 |
| Snowflake Arctic Inference | https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/ | 2025-05-01 | suffix decoding 20µs/token；主动从 L1 切到 L2 | **L2（主动选择）** |
| Red Hat gpt-oss 投机解码 | https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss | 2026-04-16 | **高并发（200）仍 +20% 的反例**；γ 边际成本 8% | **L?** |

### 2.5 方法专篇

| 材料 | URL | 年月 | 一句话定位 | 无损口径 |
|---|---|---|---|---|
| Together.ai《Medusa》 | https://www.together.ai/blog/medusa | 2023-09-11 | **作者自认放宽了分布匹配**；next-next top-1 60% / top-5 80%（树的动机） | **L3（自己说清楚）** |
| Medusa GitHub | https://github.com/FasterDecoding/Medusa | star 2.8k | Medusa-1 / Medusa-2 区分；自蒸馏 | **L3 但 README 未标注** |
| EAGLE GitHub | https://github.com/SafeAILab/EAGLE | README 更新 2025-09-18，star 2.5k | 三代差异三句话；EAGLE-3 删掉了 EAGLE-1 的核心约束 | **L?（README 自相矛盾）** |
| Apple ReDrafter | https://machinelearning.apple.com/research/recurrent-drafter | 2024-11 | beam search 结果上做 dynamic tree attention 去重前缀；端侧 MLX 2.3× | 未表态 |
| LMSYS SpecForge | https://www.lmsys.org/blog/2025-07-25-spec-forge/ | 2025-07-25 | 训草稿的工具链；离线要 12TB | 未表态 |
| HF《Universal Assisted Generation》 | https://huggingface.co/blog/universal_assisted_generation | 2024-10-29 | 跨 tokenizer 的 2-way 翻译；CodeLlama-13b + tiny_starcoder_py **1.90×**（HumanEval，单 A6000）等 5 组 | 未表态 |
| HF《Faster Assisted Generation with Dynamic Speculation》 | https://huggingface.co/blog/dynamic_speculation_lookahead | 2024-10-08 | 动态 γ；`assistant_confidence_threshold` 早停；transformers 4.45.0 起为默认 | 未表态 |
| HF《Speeding Up LLM Decoding with Advanced UAG Techniques》 | https://huggingface.co/blog/jmamou/uag-tli | 未核验 | UAG + TLI；**本次未打开，仅从搜索索引见到标题，标「未核验」** | — |

### 2.6 代码仓库

| 仓库 | URL | stars | 判据正确？ | 残差正确？ | 备注 |
|---|---|---|---|---|---|
| pytorch-labs/gpt-fast | https://raw.githubusercontent.com/pytorch-labs/gpt-fast/main/generate.py | — | ✅ | ✅ | 最短的正确实现（~50 行） |
| Jay Mody 博客内实现 | https://jaykmody.com/blog/speculative-sampling/ | — | ✅ | ✅ | numpy，带对拍 |
| feifeibear/LLMSpeculativeSampling | https://github.com/feifeibear/LLMSpeculativeSampling | 923 | ✅（两版） | ✅ | **唯一并排放了 Google 版与 DeepMind 版**；README 数字口径不全 |
| romsto/Speculative-Decoding | https://github.com/romsto/Speculative-Decoding | 115 | ✅ | ✅ 默认路径 | **⚠ 带 `skip_sample_adjustment` 开关，打开即变成错误实现，README 无警告** |
| shreyansh26/Speculative-Sampling | https://github.com/shreyansh26/Speculative-Sampling | 112 | ✅ | ✅ | **唯一分别报了 T=0 与 T=0.5**（OPT-13b/1.3b：1.78× / 1.75×；OPT-6.7b/1.3b：1.46× / 1.51×）；仅 2 组配置，需复核 |
| vLLM `rejection_sampler.py` | https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/sample/rejection_sampler.py | — | ✅ | ✅（Gumbel-max） | 生产实现参考 |

### 2.7 索引、榜单与教程

| 材料 | URL | 规模 | 备注 |
|---|---|---|---|
| hemingkx/SpeculativeDecodingPapers | https://github.com/hemingkx/SpeculativeDecodingPapers | 1.3k stars，500+ 篇，2018→2026-06 | 分类法本身即谱系；**只列不评** |
| hemingkx/Spec-Bench | https://github.com/hemingkx/Spec-Bench | 405 stars，榜单 2025-04-22 | **口径最全的横向榜单**；边界是 batch=1 + greedy |
| COLING 2025 Tutorial | https://speculative-decoding.github.io/ | 3.5 小时 | 唯一体系化教学材料；**slides/录像未核验** |

### 2.8 中文材料（C 级来源）

> **一个反常但重要的观察**：本次实际读到的三份中文材料，**在接受判据和残差分布上全部写对了**，比 NVIDIA intro 博客、Wikipedia、Modular Handbook 还准。**中文社区弱的不是公式，是口径（无损口径、加速比口径）和时效（配置参数过期）。**

| 材料 | URL | 年月 | 好在哪 | 要小心 |
|---|---|---|---|---|
| 罗西的思考《探秘Transformer系列之（30）--- 投机解码》 | https://www.cnblogs.com/rossiXYZ/p/18837229 | 2025-04-23，约 12000 字 | 判据带随机数（`r_i = torch.rand_like(...); is_accepted = r_i <= probability_ratio`）；残差给了 `p'(x) = norm(max(0,p(x) − q(x)))`；**无损用的是分布口径**（"保证和使用原始模型的**采样分布**完全相同"） | **无 $E[\tau]$、无加速比模型、无 batch 讨论**；MTP 只在参考文献 |
| SHICENT《Speculative Decoding（推测解码/投机推断）深度全景解析》 | https://www.cnblogs.com/SCCQ/p/19837997 | 2026-04-09，约 15000 字 | 判据与残差全对（`p'(x)=normalize(relu(target−draft))` 写法好）；**方法覆盖最广（16+ 种）**，适合当中文索引 | **给了 $E[\tau]=(1-\alpha^{\gamma+1})/(1-\alpha)$ 但丢了 i.i.d. 前提**；讲了 Medusa 但**没说 typical acceptance 有损**；高并发只说"可能降低收益"无数字；"vLLM 正在开发动态推测解码"已过期 |
| tbr8.org《投机解码深度解析：如何在不牺牲精度的前提下将大模型推理吞吐量提升2-3倍》 | https://tbr8.org/投机解码深度解析：如何在不牺牲精度的前提下将/ | 2026-08-01，作者未注明 | **正文**判据与残差全对；无损用分布口径；明确提"高并发时投机收益递减"、"投机反噬吞吐"（好词） | **标题把加速比说成吞吐量（正文自己都不同意）**；**`--speculative-disable-by-batch-size 32` 已过期**（2026-08-22 核验 vLLM main 的 `config/speculative.py`，grep 无命中）；作者未署名 |

---

## 3. 一张速查表：我要解决 X 问题，读哪一条

| 我想知道 | 读 | 备注 |
|---|---|---|
| 为什么 decode 慢 / 为什么并行验证几乎免费 | ①-1、①-2 前半、②-1 的 Introduction | Leviathan 原文："inference from large models is often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication" |
| "无损"到底保证了什么 | ①-3（直觉）→ ①-4（正式）→ ②-1 摘要 vs 正文对照 | 本库铁律一的全部素材 |
| 接受判据为什么要带随机数 | ②-1 §2.3 → ③-1 或 ③-2 的代码 | 看不到 `rand` 的讲解一律不信 |
| 残差分布为什么是 $\mathrm{norm}(\max(0,p-q))$ | ②-1 §2.3 + ②-2 的 $(\cdot)_+$ 定义；然后跑 ③-4 的消融开关 | 亲手把它关掉看分布怎么偏 |
| $\alpha$ 到底怎么定义、怎么测 | ②-1 Definition 3.1 + Corollary 3.6 → ②-5 的边际口径 → ②-4 的 #MAT | **三处定义不完全一致，这正是第 05 篇的主题** |
| $E[\tau]$ 公式能不能直接用 | ②-1（看到 i.i.d. 前提）→ ②-5（看到 vLLM 不用几何模型） | 第 06 篇的核心冲突 |
| γ 取几 | ④-3（3-4 / 6-8 按任务）、④-6（2-3，第 4 个掉 8%）、⑤-3 的 topk=8/draft-tokens=10 | 没有万能值 |
| 草稿模型选多大 | ②-1 §4.1 的 small/base/large 三行 + ④-... 的 NVIDIA 405B 倒 U 形表 | **两处数据都证伪"越小越好"** |
| 我的并发下还赚吗 | ④-1 → ④-3 → ④-4 → ④-6 | 最后一条会推翻前面三条的绝对化结论 |
| 树形草稿怎么调 | ④-5（三旋钮）→ 2.5 的 Medusa 博客（树的动机）→ Apple ReDrafter（beam 去重） | |
| 别人的加速比能信吗 | ④-2 → ②-4 | 一个反例 + 一个正例 |
| Medusa / EAGLE / MTP 谁有损 | Medusa：2.5 的 Together 博客（自认 L3）；EAGLE：⑤-2 的 Todo 矛盾（L?）；MTP：④-4 的 relaxed 参数（未标注的 L3） | |
| 前沿在往哪走 | ⑤-1（地图）→ ⑤-5（生产里活下来的方法枚举）→ ⑤-4（agent + CPU 草稿） | |

---

## 4. 值得直接嵌进正文的引文库（已逐字核验）

> 每条标明来源与用在哪一篇。引用时必须带 URL 与年月。

**关于无损口径**

1. > "if $q(x)>p(x)$ we reject the sample with probability $1-\frac{p(x)}{q(x)}$ and sample $x$ again from an adjusted distribution $p'(x)=norm(max(0,p(x)-q(x)))$ instead."　—— Leviathan 论文 §2.3。**用在 04。**
2. > "a novel modified rejection sampling scheme which preserves the distribution of the target model **within hardware numerics**."　—— Chen 等摘要。**用在 04、07。**
3. > "Speculative decoding sampling is **theoretically lossless up to the precision limits of hardware numerics**." / "**Changes in batch size may cause variations in logprobs and output probabilities.**"　—— vLLM 官方文档。**用在 07。**
4. > "Speculative sampling giving different result than autoregressive sampling is akin to running autoregressive sampling but **with a different seed**."　—— Jay Mody。**用在 07（题眼）、27。**
5. > "**Switched from rejection sampling to greedy verification** (accept tokens only if they match greedy decoding), ensuring outputs are identical to the base model without lowering acceptance rate."　—— Snowflake Arctic。**用在 07、25。**
6. > "A draft token is accepted **if matches the previously decoded token exactly**." / "**only greedy sampling is supported** for speculative decoding"　—— TensorRT-LLM 文档。**用在 07、20。**
7. > "**Relaxing the requirement of matching the distribution of the original model** makes the non-greedy generation even faster than greedy decoding."　—— Together.ai Medusa 博客。**用在 12、07。**

**关于负收益与口径**

8. > "There is currently **no way to dynamically disable speculation**, thus **speed ups are only observable at low batch sizes**."　—— TensorRT-LLM 文档。**用在 19。**
9. > "We begin to observe **throughput reduction beyond a batch size of 64**, which happens rarely in practice."　—— PyTorch/IBM。**用在 18、19。**
10. > "The extra compute required to propose and verify tokens can sometimes slow down the system **when it is already compute-bound**."　—— vLLM 博客。**用在 19。**
11. > "**Above batch size 4, speculative decoding returns slower inference than the main model alone.**"　—— HF Whisper 博客。**用在 18、19**（注意这是 Whisper + T4 的场景，不可外推）。
12. > "With TP = 1, the total throughput **plateaued earlier (around 20–30 concurrent requests)** compared to the baseline"　—— LLM Inference Handbook。**用在 18。**
13. > "consistent throughput and latency improvements that **persist with up to 200 concurrent requests**"　—— Red Hat（MoE + MXFP4 + EAGLE3）。**用在 18、19，作为前面几条的反例。**
14. > "For this model, **2 or 3 draft tokens is the sweet spot**" / "going from 3 to 4 draft tokens causes a modest **8% output throughput drop**"　—— Red Hat。**用在 17。**

**关于 α 与 γ**

15. > "If we make the simplifying assumption that the $\beta$s are **i.i.d.**, ... the number of tokens produced by a single run of Algorithm 1 is a **capped geometric variable**, with success probability $1-\alpha$ and cap $\gamma+1$"　—— Leviathan 论文 §3.1。**用在 06（E8 的正本清源）。**
16. > "If $\alpha>c$, there exists $\gamma$ for which we'll get an improvement, and the improvement factor will be at least $\frac{1+\alpha}{1+c}$."　—— Leviathan Corollary 3.9。**用在 06、19。**
17. > "Per-position ***unconditional*** acceptance rates ... Position i's entry is the **marginal probability that the first i+1 draft tokens are all accepted**"　—— vLLM `config/speculative.py`。**用在 05。**
18. > "Mean acceptance length to unconditional per-position rates, using the **minimum-variance schedule**."　—— vLLM `config/speculative.py`。**用在 06。**
19. > "the draft probabilities are treated as **one-hot** during rejection sampling"（`draft_sample_method` 默认 `"greedy"`）　—— vLLM `config/speculative.py`。**用在 05、20。**
20. > "we find **3-4 heads works well** in practice, whereas we found that **code models can reap benefits from 6-8 heads**"　—— PyTorch/IBM。**用在 17、21。**

**关于机制与结构**

21. > "**At least one token will always be generated** from a draft-accept loop" / "This gives us a **maximum of $K+1$ tokens per loop**"　—— Chen 等 §4.2。**用在 02。**
22. > 约 60% top-1 / 超过 80% top-5（next-next token）　—— Together.ai Medusa 博客。**用在 12、16（树的动机）。**
23. > `--speculative-num-steps`="Depth of autoregressive drafting"；`--speculative-eagle-topk`="Branching factor per step"；`--speculative-num-draft-tokens`="Maximum parallel verification capacity"　—— SGLang 文档。**用在 16、17。**
24. > "Enabling `num_speculative_tokens > 1` will run multiple times of forward on same MTP layer, **which may result in lower acceptance rate**"　—— vLLM `config/speculative.py`。**用在 14。**
25. > "**Removes the feature prediction constraint** in EAGLE and simulates this process during training using **training-time testing**."　—— EAGLE README（EAGLE-3）。**用在 13。**
26. > suffix decoding "**20 microseconds per token**"（CPU 上）　—— Snowflake Arctic。**用在 11。**
27. > "the lack of robust open-source tools for **training draft models** ... has significantly hindered its adoption."　—— LMSYS SpecForge。**用在 21。**

---

## 5. 数字总表（引用前必看口径栏）

> **严禁把不同行放进同一张对比表**——除非你已核对模型对、batch、γ、硬件、精度、指标定义、任务七项一致（本库铁律二）。**下表的存在目的正是让人看清"几乎没有两行是可比的"。**

| 数字 | 来源 | 模型对 | batch/并发 | γ | 硬件/精度 | 指标 | 任务 | 口径评级 |
|---|---|---|---|---|---|---|---|---|
| 2×–3× | Leviathan 论文摘要 | T5-XXL 11B + T5-small 60M | 未标 | 表中 3–7 | TPU（T5X） | 墙钟 | 翻译/摘要 | 缺 batch |
| 2.3× / 2.2× / 1.7× | Leviathan §4.1 | T5-XXL + T5-small / base / large | 未标 | 5 / 3 / 3 | 同上 | 墙钟 | CNNDM | 缺 batch，**但三行内部可比，是最有价值的对照** |
| 2–2.5× | Chen 等摘要 | Chinchilla 70B + 草稿 | 未标 | 未标 | "distributed setup" | 解码加速 | — | **缺项多** |
| 2.2× / 1.9× | HF Whisper | Whisper large-v2 + distil | **bs=1** | 未标 | **T4 16GB** | 端到端秒数（73→33 / 117→62） | 英语 / 多语 ASR | **较全** |
| 2× / 1.3× | PyTorch gpt-fast | CodeLlama-34B+7B / Llama-7B+TinyLlama-1B | **bs=1** | 未标 | **A100-80GB @330W** | tok/s | 代码 / 通用 | **较全，缺 γ** |
| 2×（语言）/ 3×（代码） | PyTorch/IBM | Llama3-8B / Llama2-13B / Granite-7B / Granite-20B-code + 自训 speculator | 生产并发（图中标注） | 3-4 / 6-8 头 | 未标 | **TTFT + ITL** | 生产流量 | **指标定义最好，缺硬件** |
| 1.5× / 2.8× | vLLM 博客 | Llama3-70B + Qwama-0.5B / n-gram | **QPS=1** | 未标 | **4×H100**（未标 SXM/PCIe） | 端到端 | ShareGPT / CNNDM | 缺 γ、缺 H100 型号 |
| 慢 1.4× / 慢 1.8× | vLLM 博客 | 同上 | **高 QPS**（具体值未标） | 未标 | 同上 | 端到端 | 同上 | 缺具体 QPS |
| 最高 1.4× | vLLM issue #10318（独立复现） | Llama-3-70B TP=4 + Qwama-0.5B TP=1 | **1 QPS，bs=1** | **4** | **4×H100 PCIe**，vLLM 0.6.3 | 端到端 | ShareGPT | **口径最全的一条** |
| 3.33× / 3.61× / 3.04× | NVIDIA 3.6x | Llama 3.1 405B + 3.2-1B / 3.2-3B / 3.1-8B | `max_batch_size=32`（**非实际并发**） | 未标 | **4×H200，FP8** | "output tokens/s"（含 TTFT） | 未标 | **标题称 throughput，实为单请求速率——口径错** |
| 2.86× / 2.75× / 2.23× | NVIDIA 3.6x | Llama 3.1 70B + 同上三个草稿 | 同上 | 未标 | **1×H200，FP8** | 同上 | 未标 | 同上 |
| 2.31× / 2.13× / 2.98× | AMD ROCm | Llama3.1-70B+3.2-1B / 405B+3.2-1B / 70B 长上下文 | `max-num-seqs 300`（上限） | **5** | **MI300X**，ROCm 6.3.1，vLLM 0.6.7 | 加速比 | — / — / 32768-in,128-out | 缺实际并发 |
| 2.21× → 2.06× | AMD ROCm | 同上 | **0.2 → 1.4 req/s** | 5 | 同上 | 加速比 | — | **衰减曲线，较全** |
| 244.10 / 373.25 vs 158.34 tok/s | SGLang 文档 | LLaMA-3.1-8B-Instruct + EAGLE-2 / EAGLE-3 | **未标**（MT-bench 惯例 bs=1） | steps/topk/draft-tokens 未在该表标 | **1×H100** | tok/s（称 throughput） | MT-bench | **缺 batch，用词误导** |
| 2.0× / 2.18× | LMSYS SpecForge | Llama 4 Scout / Maverick + EAGLE3 | 未标 | topk=8, draft-tokens=10 | **未标** | 加速比 | MT-Bench | **缺项多** |
| +27.2% 峰值 / +20.7% 几何均值 | Red Hat | gpt-oss-120b(MXFP4) + Eagle3-v2 | **并发 1→200 全扫** | **2–3（甜点）** | **H200-PCIe-141GB**，TP=1/2 | 输出吞吐 tok/s | ShareGPT | **最全的一条** |
| 179 / 217 / 209 tok/s | Snowflake Arctic | Llama-3.1-70B + LSTM+Suffix | **0.5 req/s** | 未标 | **8×H100，TP=2，FP8** | tok/s | ShareGPT / HumanEval / 混合 | 较全，缺 γ |
| 1.8×–4.5× | Snowflake Arctic | OpenHands LM 32B（CodeAct agent） | 未标 | 未标 | **未标** | 端到端 | SWE-Bench | **缺项多** |
| ~2×（7B/13B/33B） | Together.ai Medusa | Vicuna + Medusa 头 | **bs=1** | 树规模未标 | **单张 A100-80G**（33B 用 8-bit） | 墙钟 | ShareGPT | 缺 γ |
| 2.2×–3.6× | Medusa README | 多个 LLM | **bs=1** | 未标 | 单 GPU | — | — | 缺项多 |
| 3×（13B） | EAGLE README | Vicuna 13B | **未标** | 未标 | **2×RTX 3090，fp16** | 加速比 | — | **缺 batch、缺 temperature** |
| 2.8×（H100）/ 2.3×（Apple Silicon） | Apple ReDrafter | Vicuna + ReDrafter | 未标 | 未标 | H100 / Apple Silicon MLX | 加速比 | MT-Bench | 缺 batch |
| 2.03–2.38×（3090）/ 2.22–3.02×（A100） | Spec-Bench 榜单 | Vicuna-7B/13B/33B-v1.3 + 各方法 | **bs=1** | 各方法自带 | **RTX 3090 24GB / A100 80GB，FP16，greedy** | 加速比 **+ #MAT** | Spec-Bench 多子任务 | **最全，唯一可横向比较的一组** |
| 1.90× / 1.52× / 1.76× / 1.78× / 1.91× | HF UAG | 5 组跨 tokenizer 配对 | 未标 | 未标 | A6000 / A100（分组标注） | 加速比 | HumanEval / CNNDM / scrolls | 缺 batch |
| 2.71× / 1.59× / 1.09× / 1.52× | HF 动态 SL | opt-6.7b+opt-125m / codegen / Llama-3.1-8B+3.2-1B | 未标 | **动态** | **RTX 4090** | 加速比 | 摘要 / 开放生成 / 代码 | 缺 batch；**注意 codegen 那组启发式基线只有 0.89×，即基线本身是负收益** |
| 1.78× / 1.75× / 1.46× / 1.51× | shreyansh26 仓库 | OPT-13b / OPT-6.7b + OPT-1.3b | 未标 | 未标 | **未标** | 加速比 | — | **缺项多，但唯一分列 T=0 与 T=0.5** |
| 1.29×（按面值） | feifeibear README | llama2-70b + llama2-7b | 未标 | 未标 | **未标** | 单位存疑 | — | **不可用** |

**从这张表能直接得出的三条结论（本库第 18/19 篇可用）**：

1. **口径最全的两条（vLLM issue #10318 的 1.4× 与 Spec-Bench 的 2.0–3.0×）都是 batch=1**；**所有报出 3× 以上的条目，没有一条给了真实并发扫描**（Red Hat 2026 是唯一例外，它给了并发扫描，报的提升是 **+20%~27%** 而不是数倍）。
2. **同一份材料内部可比的对照（Leviathan §4.1 的三行、NVIDIA 405B 的三行、AMD 的 req/s 扫描、Spec-Bench 的同硬件横比）信息量远大于跨材料的数字堆砌。** 附录应当引导读者只看这类内部对照。
3. **"数倍加速"与"生产吞吐提升"是两个不同量级的事**：bs=1 的墙钟能到 2–3×，真实并发下的吞吐提升在 +10%~+27% 这个量级。**把前者说成后者，就是社区高频错误第 5 条。**

---

## 6. 明确不推荐 / 慎读清单

| 材料 | 问题 | 处理方式 |
|---|---|---|
| **Wikipedia《Speculative decoding》** | "produces **the same results** as standard decoding"；且完全不提残差分布 | **不作为知识来源**；可作为第 27 篇的反面靶子引用 |
| **NVIDIA《An Introduction to Speculative Decoding》** | 一句话里同时取消随机性并混淆 L1/L2："Only when a draft token **matches** what the target model would have generated, is it accepted ... ensuring that the final output is **identical** to what the target model would have produced"；另有 "boosting throughput without any impact on accuracy" | **只采纳它的 "meticulous scientist / quick assistant" 比喻，且必须紧跟纠正** |
| **LLM Inference Handbook（Modular）** | "guarantees the final output **matches exactly** what the original target model would have produced on its own"；全篇零公式；把 Medusa 2.2–3.6× 与 EAGLE 3.0–6.5× 并排（违反铁律二） | **只引它的 20–30 并发拐点** |
| **Lilian Weng《Large Transformer Model Inference Optimization》** | **该文根本没有投机解码章节**（本次逐条核了全文目录：Methods Overview / Distillation / Quantization / Pruning / Sparsity / Architectural Optimization / Citation / References；发表 2023-01-10，更新 2023-01-24，早于两篇奠基论文进入主流） | **不收录**；并在第 26 篇点名"把它列为投机解码必读"是错误转引 |
| **tbr8.org 中文长文的标题与配置参数** | 标题"吞吐量提升 2-3 倍"与正文自相矛盾；`--speculative-disable-by-batch-size 32` 在 vLLM main（2026-08-22 核验）已不存在 | **正文可读，标题与参数不可引** |
| **feifeibear README 的 benchmark 数字** | 无硬件、无 γ、无 batch，单位存疑 | **只读代码，不引数字** |
| **romsto 的 `skip_sample_adjustment=True`** | 打开即变成"拒绝后直接从 $p$ 重采"的错误实现，README 无警告 | **作为消融开关可用，作为默认路径必须警告** |
| **Aleksa Gordić 关于 "vLLM V1 不支持独立草稿模型" 的结论** | 截至 2025-08 成立；vLLM 官方文档（2026-08）明写 draft model 在 `>0.10.0` 已支持 | **引用时必须标注时效** |
| **EAGLE README 的无损表述** | 正文说 "provably maintaining ... distribution"，Todo 里却写着 "Support non-greedy inference (provably maintaining text distribution)" | **自相矛盾，不得据此判定 L1；要么读源码，要么写"未验证"** |
| **TensorRT-LLM 的 MTP relaxed 参数** | `use_relaxed_acceptance_for_thinking` / `relaxed_topk` / `relaxed_delta` 是有损开关，文档零标注 | **第 14 篇必须点名** |

---

## 7. 访问失败记录（不进推荐序列）

| 材料 | URL | 状态（2026-08-22 核验） |
|---|---|---|
| Andrej Karpathy 投机执行推文 | https://x.com/karpathy/status/1697318534555336961 | **HTTP 402，链接无法访问**。社区引用极广，**若要引用必须转引二手并显式标注"原推文未核验"** |
| 知乎专栏（共 7 篇，含《大模型推理妙招—投机采样》p/651359908、《投机采样原理剖析》p/15837326799、《手撕LLM-Speculative Decoding》p/671432448、《投机解码算法快速理解与代码实现》p/685282553、《推测解码的拒绝采样》p/7162909442、《投机解码详解》p/15575453436、《What makes for efficient speculative decoding?》p/27272034867） | zhuanlan.zhihu.com | **HTTP 403（WebFetch 与 curl 均被拒）**，链接无法访问。**只见标题，未读正文，不评点** |
| CSDN《论文导读 \| 投机解码加速模型推理》 | https://blog.csdn.net/weixin_48167662/article/details/139006059 | **HTTP 521**，链接无法访问 |
| WaytoAGI 飞书《（10）深入LLM投机采样(上)》 | https://waytoagi.feishu.cn/wiki/U1BywrrxTibXOAkqF5Yc2uWynRh | **302 跳登录页**，需登录 |
| openlm.ai《Speculative Decoding in vLLM》 | https://openlm.ai/speculative-decoding-in-vllm/ | **HTTP 403** |
| Aphrodite Engine 投机解码文档 | https://aphrodite.pygmalion.chat/spec-decoding/overview/ | **HTTP 404，链接已失效** |
| COLING 2025 tutorial 的 slides / 录像 | https://tinyurl.com/speculative-decoding-tutorial / https://tinyurl.com/spec-tutorial-recording | **未跟进短链跳转，未核验内容** |
| HF《Speeding Up LLM Decoding with Advanced UAG Techniques》 | https://huggingface.co/blog/jmamou/uag-tli | **本次未打开，仅从搜索索引见到标题，标「未核验」** |
| HF 文档 Generation Strategies 的 assisted decoding 小节 | https://huggingface.co/docs/transformers/en/generation_strategies | **页面可访问（v5.15.1），但全文只有 Greedy / Sampling / Beam search + Custom generation methods + Resources，未见 assisted decoding 小节**。页面把 `custom_generate` Tutorials collection 描述为 "reference implementations for methods that **previously were part of `transformers`**"。**是否已迁出核心 API 本次未核到公告，标「待确认」。无论如何，附录不得再指向此页讲 assisted decoding，改指 HF 博客原文。** |
| 视频类来源（Trelis Research、Efficient NLP、Umar Jamil、Yannic Kilcher、Stanford CS336、MIT 6.5940、B 站 UP 主等） | — | **本次 WebSearch 配额用尽（200/200），未能定位到可抓取 URL。整类未覆盖，留给下一轮** |
| Sebastian Raschka 的投机解码专文 | — | **检索不充分，标「未查证」，不写入附录** |
| 交互式 / 可视化讲解 | — | **本次一份都没找到**。树注意力 mask 极适合可视化，**若社区确实空白，这是本库可以自建的空白（`_lab/tree.py` 输出 mask 图）** |

---

## 8. 给第 30 篇的落地建议

1. **附录的主体就是 §1 的五组推荐顺序**，每条保留"为什么排在这个位置"和"要小心什么"两栏。**只列 URL 不给排序理由的附录没有价值。**
2. **§5 的数字总表原样搬进去**，并保留最后那三条结论。它是铁律二最好的落地形式：**读者一眼看到"几乎没有两行可比"。**
3. **§6 的"不推荐清单"必须保留**，尤其 Lilian Weng 那条（澄清一个流传很广的错误转引）和 Wikipedia 那条。
4. **§7 的访问失败记录也要保留一部分**（至少 Karpathy 推文、知乎全线、Aphrodite 404 这三条），理由是：**本库承诺 URL 都实际访问过，那么"哪些没访问到"同样是要交代的信息。**
5. **每条 URL 后建议统一附 `（2026-08 核验）`**，并在第 30 篇开头写明核验日期与失效处理策略（失效的就地标注 404，不冒充读过）。
