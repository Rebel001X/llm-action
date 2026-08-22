# 18 batch 与吞吐 —— 收益衰减曲线

## 1. 一句话

投机采样的加速比不是一个常数，它是 batch 的**单调递减函数**，而且这条曲线有一个可以算出来的**渐近线**：一旦验证前向进入 compute-bound 区，加速比收敛到

$$
\frac{E[\tau]}{\gamma+1}\times\big(1-\text{草稿开销占比}\big)\ <\ 1
$$

这个极限**恒小于 1**，而且**调参救不了** —— 即使把接受长度顶到理论上限 $\gamma+1$，也只能勉强打平。所以"投机解码在大 batch 下会亏"不是工程没做好，是算术。

---

## 2. 从哪来

[[06-期望接受长度与加速比模型-完整推导]] 给出的 $\text{speedup}=E[\tau]/(\gamma c+1)$ 里，分母那个 $1$ 代表"一次目标模型前向"，并**默认它与生成 1 个 token 同价**。[[03-并行验证为什么几乎免费-算术强度与roofline]] 说明了这条只在 memory-bound 区成立。

本篇要回答的就是：**离开那个区之后会怎样，以及边界在哪。**

这不是一个学术问题。线上服务的 batch 由 QPS 决定，而 QPS 是波动的：凌晨 batch=2，高峰 batch=200。**同一套配置在一天之内会从"加速 2.8 倍"走到"降速 40%"**，而多数团队只在低负载下测过。

---

## 3. 机制拆解

### 3.1 三条时间线

一次投机迭代包含两段（模型见 `_lab/speedup.py`）：

$$
T_{\text{iter}}=\underbrace{\sum_{i=0}^{\gamma-1}T_{\text{draft}}(\text{batch},\ \text{seqlen}+i,\ 1)}_{\gamma\ \text{次串行草稿前向}}
+\underbrace{T_{\text{target}}(\text{batch},\ \text{seqlen},\ \gamma+1)}_{1\ \text{次验证前向}}
$$

其中

$$
T(\cdot)=\max\!\left(\frac{P b+\text{batch}\cdot\text{seqlen}\cdot\text{kv}}{\text{BW}},\ \frac{\text{batch}\cdot n_q\cdot(2P+4L\,\text{seqlen}\,d_{\text{model}})}{\text{peak}}\right)
$$

基线是 $T_{\text{target}}(\text{batch},\text{seqlen},1)$ 产出 batch 个 token。（$L$ 为层数；attention 项必须乘 $L$，见文末勘误记录。）

### 3.2 关键的标度差异

```mermaid
graph TD
    B["batch 增大"] --> M["访存项：权重 Pb 不变<br/>KV 项 ∝ batch"]
    B --> C["算力项：∝ batch × n_query"]
    M --> R{"谁更大？"}
    C --> R
    R -->|"memory 更大<br/>（小 batch / 长上下文）"| F["验证近乎免费<br/>加速比 ≈ E[τ]/(γc+1)"]
    R -->|"compute 更大<br/>（大 batch + 短上下文）"| G["验证成本 ∝ γ+1<br/>加速比 → E[τ]/(γ+1) < 1"]
```

**注意 $n_q$ 只出现在算力项里。** 这一条不对称同时解释了投机采样为什么能赢、以及为什么最终必输：

- memory 区：$n_q$ 不影响时间 → 验证 $\gamma+1$ 个 token 白送 → 赢；
- compute 区：时间 $\propto n_q=\gamma+1$ → 花了 $\gamma+1$ 份算力只换来 $E[\tau]$ 个 token → **算力利用率降为 $E[\tau]/(\gamma+1)$** → 输。

---

## 4. 逐步推导：渐近线

设已深入 compute 区（算力项主导）。此时

$$
T_{\text{target}}(\text{batch},\text{seqlen},n_q)=\frac{\text{batch}\cdot n_q\cdot F}{\text{peak}},\qquad F:=2P+4L\,\text{seqlen}\,d_{\text{model}}
$$

**基线吞吐**（tokens/s，全 batch 合计）：

$$
\text{TPS}_{\text{base}}=\frac{\text{batch}}{T_{\text{target}}(\cdot,1)}=\frac{\text{peak}}{F}
$$

**投机吞吐**：

$$
\text{TPS}_{\text{spec}}=\frac{\text{batch}\cdot E[\tau]}{T_{\text{iter}}}
=\frac{\text{batch}\cdot E[\tau]}{T_{\text{draft,total}}+\dfrac{\text{batch}(\gamma+1)F}{\text{peak}}}
$$

把分子分母同除 batch，令 $s:=T_{\text{draft,total}}/T_{\text{iter}}$ 为草稿占迭代的比例：

$$
\text{TPS}_{\text{spec}}=\frac{E[\tau]}{(1-s)^{-1}\cdot\dfrac{(\gamma+1)F}{\text{peak}}}
=\frac{\text{peak}}{F}\cdot\frac{E[\tau]}{\gamma+1}\cdot(1-s)
$$

两式相除：

$$
\boxed{\ \lim_{\text{batch}\to\infty}\text{speedup}=\frac{E[\tau]}{\gamma+1}\,(1-s)\ }
\tag{4.1}
$$

**三条推论，逐条都很硬**：

1. **$\text{TPS}_{\text{spec}}$ 与 batch 无关**（batch 被约掉了）—— 投机吞吐在 compute 区**封顶**，而基线吞吐也封顶但在更高的位置。所以交叉点必然出现，只是早晚。
2. **$E[\tau]/(\gamma+1)\le 1$ 恒成立**（$\tau$ 的上界就是 $\gamma+1$），再乘上 $(1-s)<1$，**极限严格小于 1**。
3. **$E[\tau]/(\gamma+1)$ 就是"算力有效利用率"**：你让目标模型算了 $\gamma+1$ 个位置，只有 $E[\tau]$ 个变成了产出，其余是**为了买信息而故意浪费的算力**。memory 区里这份浪费不要钱，compute 区里它按全价收费。

> **一句话总结机制**：投机采样是拿"算力"去买"延迟"。算力免费时（memory-bound）这笔交易稳赚；算力开始收费时（compute-bound），你买的东西一分没变，价格却涨到了全价。

---

## 5. 实测表

口径（**全表统一，不可与其它来源的数字横比**）：target = llama3-70b，draft = llama3.2-1b，$\gamma=4$，$E[\tau]=3.0$ 固定，fp16 权重与 KV，硬件 **8×H100 SXM5（TP=8，带宽/算力/容量按 8 倍线性放大，忽略通信）**，报的是**吞吐**（tokens/s，全 batch 合计）。70B fp16 权重 141 GB，单卡 80 GB 装不下，故必须多卡。**本模型忽略 TP 通信、norm/softmax、kernel launch 与调度，是乐观上界，真实翻转点只会更早。**

复跑：`python _lab/speedup.py --batch`

### seqlen = 1024（短上下文）

| batch | 基线 tok/s | 投机 tok/s | 加速比 | 验证阶段 | 草稿占迭代 | 显存 |
|---|---|---|---|---|---|---|
| 1 | 189.4 | 530.4 | **2.801** | memory | 6.6% | 144 GB |
| 16 | 2 925.6 | 8 109.0 | **2.772** | memory | 7.6% | 150 GB |
| 64 | 10 543.7 | 28 397.8 | **2.693** | memory | 10.2% | 167 GB |
| 128 | 18 628.3 | 30 367.7 | 1.630 | **compute** | 8.0% | 191 GB |
| 256 | 30 210.6 | 30 818.7 | 1.020 | compute | 6.6% | 238 GB |
| 384 | 38 108.6 | 30 972.1 | **0.813** ⚠ | compute | 6.2% | 285 GB |
| 512 | 43 839.2 | 31 049.3 | **0.708** ⚠ | compute | 5.9% | 333 GB |
| 768 | 51 598.1 | 31 127.0 | **0.603** ⚠ | compute | 5.7% | 427 GB |
| 1024 | 55 016.4 | 31 165.9 | **0.566** ⚠ | compute | 5.6% | 522 GB |

**请盯住"投机 tok/s"那一列**：从 batch=128 到 batch=1024，batch 涨了 8 倍，投机吞吐只从 30 368 涨到 31 166（**+2.6%**）—— 它封顶了，正如 (4.1) 推论 1 所言。而基线从 18 628 涨到 55 016（**+195%**），一路把投机甩在身后。

再验一次 (4.1)：$E[\tau]/(\gamma+1)=3.0/5=0.600$，草稿占比 $s=5.6\%$，预测极限 $0.600\times0.944=\mathbf{0.566}$ —— 与表中 batch=1024 的实测值**完全一致**（`_lab/test_speedup.py::test_compute_bound_asymptote_equals_wasted_compute_ratio`）。

### seqlen = 16384（长上下文）

| batch | 基线 tok/s | 投机 tok/s | 加速比 | 验证阶段 | 草稿占迭代 | 显存 |
|---|---|---|---|---|---|---|
| 1 | 182.8 | 506.8 | **2.772** | memory | 7.6% | 150 GB |
| 16 | 1 888.2 | 4 740.2 | **2.510** | memory | 16.3% | 238 GB |
| 64 | 3 538.0 | 8 139.6 | **2.301** | memory | 23.3% | 522 GB |
| 128 | — | — | — | — | — | 900 GB **装不下** |
| ≥256 | — | — | — | — | — | 1 656 GB+ **装不下** |

**这一段不翻转。** 长上下文的 KV 读把前向死死钉在 memory 区，验证依旧近乎白送。但它换来另一个约束：**显存装不下大 batch**。

于是流行说法"投机解码在大 batch 下没用"必须补上前提：**条件是短上下文**。长上下文场景下投机采样在能装下的全部 batch 范围内都是赚的（`_lab/test_speedup.py::test_long_context_keeps_speedup_at_large_batch`），只是那个范围本身被显存卡死（`::test_long_context_large_batch_may_not_fit`）。这条与 MagicDec 等工作的结论方向一致，详见 [[22-长上下文下的投机采样]]。

---

## 6. 代码验证

```python
# _lab/speedup.py
def spec_throughput(target, draft, hw, batch, seqlen, gamma, accept_len):
    base = fwd_time(tm, hw, batch, seqlen, 1)
    t_draft = sum(fwd_time(dm, hw, batch, seqlen + i, 1)["t"] for i in range(gamma))
    t_verify = fwd_time(tm, hw, batch, seqlen, gamma + 1)["t"]
    t_iter = t_draft + t_verify
    return dict(base_tps=batch / base["t"],
                spec_tps=batch * accept_len / t_iter,
                speedup=(accept_len / t_iter) * base["t"], ...)
```

三条关键断言：

- 渐近线等于"算力浪费比"（§4 的 (4.1)）：`::test_compute_bound_asymptote_equals_wasted_compute_ratio`
- 投机吞吐封顶而基线继续涨（推论 1）：`::test_spec_throughput_saturates_while_baseline_keeps_climbing` —— batch 涨 8 倍投机吞吐涨不到 10%，基线涨超过 150%
- 加速比对 batch 单调递减：`::test_speedup_is_monotone_decreasing_in_batch`

---

## 7. 口径与坑

1. **本表报的是吞吐，不是延迟。** 这两者在大 batch 下会给出**相反**的结论：batch=384 时投机的吞吐是 0.813 倍（亏），但**单请求的 TPOT 仍然可能更好**（每轮产出 3 个 token）。所以"投机解码在大 batch 下有没有用"这个问题**问得不完整**，必须先说清优化目标是延迟还是吞吐。SLO 里若写的是 P99 TPOT，结论可能与本表相反。
2. **$E[\tau]=3.0$ 是假设值不是实测值。** 真实的 $E[\tau]$ 还会随 batch 变化（不同请求处在不同难度的位置，一批里的接受长度由最短的那条拖累 —— 见 §8 第 3 条），所以真实曲线比本表更陡。
3. **本模型忽略 TP 通信。** all-reduce 在 decode 阶段占比可观，而投机验证的 activation 是 $\gamma+1$ 倍大，通信量也随之上升。计入之后翻转点更早。
4. **不要把本表的数字与论文/blog 的加速比横比。** 本表是解析模型的乐观上界，口径也不同（吞吐 vs 延迟、模型组合、硬件）。铁律二在这里是硬约束。
5. **"草稿占迭代"随 seqlen 显著变化**（1024 下约 6%，16384 下升到 23%）。原因是草稿模型也要读自己的 KV cache，长上下文下这笔开销按比例放大。**草稿不是永远便宜的**。

---

## 8. 失效条件（本篇结论本身什么时候不适用）

1. **优化目标是延迟而非吞吐时**：见 §7 第 1 条，结论可能相反。
2. **长上下文时**：§5 第二张表，不翻转（但受显存约束）。
3. **一批请求难度不齐时**：本模型假设整批共享同一个 $E[\tau]$。真实系统里一批里既有简单续写也有复杂推理，**同步验证意味着整批要等最慢的那条**，实际接受长度比单条平均值更低。这条会让真实曲线比本表更差，属于本模型**没有覆盖**的恶化因素。
4. **量化改变账本时**：权重量化到 int4 会把访存项砍到 1/4，屋脊点右移，**翻转点提前**（更容易 compute-bound）；KV cache 量化则相反地缓解长上下文的访存压力。见 [[23-与其它优化的相互作用-量化与KVcache与PD分离]]。
5. **MoE 模型 —— 有一个真实反例，必须写明。** 有效激活参数远小于总参数，访存与算力的比值整个变了：路由使得"每 token 的算力"按激活参数算，而"权重读取"在大 batch 下趋近全量，两者的比值远比稠密模型友好。本库调研（`_research/RS-2-社区讲解盘点与评点.md`）记录到 Red Hat 于 2026-04 报告，在 **gpt-oss-120b（MoE + MXFP4 量化）+ EAGLE3** 上并发到 **200 仍有约 +20% 吞吐** —— 这与本篇稠密模型的曲线方向相反。

   所以本篇 §4 的渐近线结论**有明确的适用范围**：它推的是"消耗 $\gamma+1$ 份算力换 $E[\tau]$ 个 token"这笔账，在稠密模型上成立；MoE 把"一份算力"的定义改了，账要重列。**本模型未覆盖 MoE，本篇不对 MoE 下的交叉点给出任何数值结论。** 该 Red Hat 数据的完整口径见 [[19-负收益全解-什么时候投机反而更慢]] 与 [[26-社区精彩解释精选-好在哪与错在哪]]。
6. **PD 分离架构下**：decode 实例的 batch 组织方式与本模型不同，且 prefill 与 decode 分开调度会改变 decode 侧的实际 batch 分布。

---

## 9. 自测题

1. 为什么 compute 区里"投机吞吐与 batch 无关"？用 §4 的推导说明。
   <details><summary>答案要点</summary>compute 区里 $T_{\text{verify}}\propto\text{batch}\cdot(\gamma+1)$，而吞吐 $=\text{batch}\cdot E[\tau]/T_{\text{iter}}$，batch 在分子分母同时出现被约掉。物理含义：算力已经打满，产出速率由"每份算力能换几个 token"决定，与并发数无关。</details>

2. 某团队把 $\gamma$ 从 4 降到 2 来"减少大 batch 下的浪费"。假设 $E[\tau]$ 相应从 3.0 降到 2.0，compute 极限下的加速比会变好吗？
   <details><summary>答案要点</summary>$E[\tau]/(\gamma+1)$ 从 $3/5=0.60$ 变成 $2/3=0.667$，**确实变好了**，但仍然 <1。降 $\gamma$ 能减轻亏损但不能扭亏 —— 因为只要 $E[\tau]<\gamma+1$ 就有浪费。极限情况 $\gamma\to0$ 时退化成不开投机（比值→1）。**这说明大 batch 下正确的动作是关掉投机，而不是调小 $\gamma$。**</details>

3. 如果把接受长度顶到理论上限 $E[\tau]=\gamma+1$（全部接受），compute 极限下加速比是多少？
   <details><summary>答案要点</summary>$E[\tau]/(\gamma+1)=1$，再乘 $(1-s)$，得到略小于 1 —— 只能**打平偏亏**，亏掉的正是草稿的开销。本库实测该情形加速比落在 0.9–1.0 之间（`_lab/test_speedup.py::test_higher_accept_length_raises_but_cannot_save_asymptote`）。含义：compute 区里投机采样**在吞吐意义上最好也就是不亏**。</details>

4. 表里 batch=256 时加速比 1.038，几乎打平。如果你是这套服务的负责人，会怎么设置策略？
   <details><summary>答案要点</summary>关键在于此处曲线很陡（256→384 就掉到 0.827），且真实系统比模型更差（§8 第 3、5 条）。合理做法是**按当前 batch 动态开关投机**，阈值设在明显低于模型给出的翻转点处（因为模型是乐观上界），并且监控的应当是实测吞吐与 P99 TPOT 两条线，而不是理论加速比。相关实践见 [[19-负收益全解-什么时候投机反而更慢]] 与 [[20-主流引擎实现-vLLM与SGLang与TensorRTLLM]]。</details>

5. 为什么长上下文那张表里"草稿占迭代"从 6% 升到 23%？这说明了什么？
   <details><summary>答案要点</summary>草稿模型也要读自己的 KV cache，KV 流量 $\propto$ batch·seqlen，长上下文下草稿的访存开销按比例放大，而它的权重很小、原本主要成本就是访存。含义：**"草稿很便宜"这个前提在长上下文下会松动**，选草稿时要看的不只是参数量，还有它的 KV 结构（层数 × KV 头数）。</details>

---

## 10. 延伸与双链

- 物理地基与"免费"的边界：[[03-并行验证为什么几乎免费-算术强度与roofline]]
- 分母那个 $1$ 的来历：[[06-期望接受长度与加速比模型-完整推导]]
- 完整的负收益判定清单：[[19-负收益全解-什么时候投机反而更慢]]
- 长上下文为什么不翻转：[[22-长上下文下的投机采样]]
- 量化/MoE/PD 分离如何改写这本账：[[23-与其它优化的相互作用-量化与KVcache与PD分离]]
- 树规模也算进 query token 数：[[16-树形草稿与树注意力-mask构造与验证]]
- 引擎里的自适应开关：[[20-主流引擎实现-vLLM与SGLang与TensorRTLLM]]
- 误解清单：[[27-常见误解与判据]]

---

### 本篇验证

- `_lab/test_speedup.py::test_compute_bound_asymptote_equals_wasted_compute_ratio` —— **验证 §4 的 (4.1)**：compute 极限下加速比 = $E[\tau]/(\gamma+1)\times(1-s)$，实测 0.566 与预测吻合到 0.01 以内。
- `_lab/test_speedup.py::test_spec_throughput_saturates_while_baseline_keeps_climbing` —— 验证推论 1：batch 涨 8 倍，投机吞吐涨不到 10%，基线涨超过 150%。
- `_lab/test_speedup.py::test_higher_accept_length_raises_but_cannot_save_asymptote` —— 验证推论 2：$E[\tau]$ 顶到 $\gamma+1$ 也只能打平偏亏。
- `_lab/test_speedup.py::test_speedup_collapses_at_large_batch_short_context` —— 短上下文大 batch 下加速比 <1 且验证已 compute-bound。
- `_lab/test_speedup.py::test_speedup_is_monotone_decreasing_in_batch` —— 加速比对 batch 单调递减。
- `_lab/test_speedup.py::test_long_context_keeps_speedup_at_large_batch` —— 长上下文下不翻转。
- `_lab/test_speedup.py::test_long_context_large_batch_may_not_fit` —— 但受显存约束。
- `_lab/test_speedup.py::test_verify_stops_being_free_in_compute_bound_region` —— compute 区里验证时间随 query token 数线性涨（机制）。
- 可复跑：`python _lab/speedup.py --batch` —— §5 的两张实测表

### 本篇勘误记录

本篇初稿的数字由 `_lab/speedup.py` 的早期版本算出，那个版本的 attention 浮点数写成
$4\,s\,d_{\text{model}}$，**漏乘了层数 $L$**（正确为 $4L\,s\,d_{\text{model}}$）。
该 bug 在写第 03 篇的过程中被查出并已修复，本篇表格已用修正后的模型重算。

影响：seqlen=1024 下 attention 项只占 $2P$ 的约 0.02%，故加速比只有第三位小数级变化
（如 batch=256 从 1.038 变为 1.020），**交叉点位置与 §4 的渐近线结论均未改变**；
seqlen=16384 的可行行仍全部落在 memory 区，结论不变。
回归测试：`_lab/test_speedup.py::test_attention_flops_include_layer_count`。

### 本篇来源

- 本篇的模型与数字全部来自本库自建的解析模型 `_lab/speedup.py`，**不是硬件实测**。它的价值在于把"为什么必然翻转"讲成可推导、可复算的形式；它的局限在 §7、§8 已逐条声明。真实硬件上的实测数据见 [[19-负收益全解-什么时候投机反而更慢]] 收集的第三方基准。
- 硬件规格：NVIDIA H100 SXM5，HBM3 带宽 3.35 TB/s，BF16 **稠密**峰值 989.5 TFLOPS（含稀疏的 1979 TFLOPS 不适用于本场景）。屋脊点 $989.5/3.35\approx295$ FLOP/Byte。
- 本仓库既有材料：`MLOPS/02-推理服务案例/推理08-投机解码的负收益.md` 记录了单个案例；本篇补的是**为什么必然如此**的解析推导与渐近线闭式。
