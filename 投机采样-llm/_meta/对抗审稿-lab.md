# 对抗审稿：`_lab/`（2026-08-22）

审稿口径：**假设 `_lab/` 是错的，去证明它错。** 目标是攻破"本库的数学断言都由可执行代码验证过"这条核心主张。

结论：**5 个真问题、3 个疑似/口径问题、11 项攻击过但没攻破。**
另有 5 项我独立找到、但在本次审稿进行中被并发编辑抢先修掉的问题（第 4 节，作为交叉验证留档）。

- 审稿开始时 `cd _lab && python -m pytest -q` = **200 passed**；审稿结束时 = **233 passed**（增量全部来自并发编辑，不是我）。
- **本次审稿未改动任何代码**。原因见 §5。所有结论都附可复跑脚本。

---

## 0. 一个必须先说明的事实：审稿期间有并发编辑

审稿从 21:30 开始。期间 `_lab/` 下有文件被**另一个进程**改写：

| 文件 | 我读到的时刻 | 被改写的时刻 |
|---|---|---|
| `moe.py` / `test_moe.py` | 21:31 | **21:34** |
| `test_speedup.py` | 21:31 | **21:47** |
| `test_verify_rules.py`（新增） | — | **21:48** |
| `verify.py` | 21:32 | **22:00** |

未被触碰（因此本报告的主要发现都落在这里）：`spec.py`、`tree.py`、`test_tree.py`、`accept.py`、`regime.py`、`caliber.py`、`prehistory.py`、`speedup.py`、`notation.py` 及其测试。

**这本身就是一条发现**：在没有版本控制（该目录不是 git 仓库）的情况下并发改 `_lab/`，两边都会以为自己看到的是最新状态。建议给该库加 git，否则"谁在什么时候改了哪条结论"无法追溯。

---

## 1. 真问题（5 个）

### 真-1 ⭐ `16 §5.3` 的「无论预算多大，链都赢」是**错的**，而且支撑它的测试恰好只扫到成立边界

**① 问题描述**

第 16 篇 §5.3 写：

> **流行说法二："草稿不准就该把树加宽。"** 也错，而且错得更彻底。看 agree=0.80 那张表：$c_2 = c_1 = 0.608$，**第二候选的边际贡献恰好是 0**……于是**无论预算多大，链都赢**（`_lab/test_tree.py::test_width_is_worthless_without_marginal_coverage`）。

§5.4 第 2 条重复了这个断言："草稿整体跑偏（边际覆盖率增益 ≈ 0）时……**无论预算多大树都不赢**"。
`tree.py::_shape()` 第 229–230 行也印了同一句话（对 agree=0.50）。

**用本库自己的模型、自己的覆盖率数字，把预算从 64 往上再走一格，这句话就被推翻。**

正文 §5.2 那张 agree=0.80 的表自己就写了 $c_4=0.828 > c_1=0.608$。既然 $c_4>c_1$，4 叉树的渐近上限是 $c_4/(1-c_4)=4.81$，而链的渐近上限是 $c_1/(1-c_1)=1.55$ —— **只要预算够大，4 叉树必然反超**，这是算术必然，不是数值巧合。表只扫到 $n=64$，恰好停在反超之前一格。

实算（与 `tree.py --shape` 同一 rng 序列，agree=0.80）：

| 预算 $n$ | 链 | 2 叉 | 4 叉 | 8 叉 | 最优 |
|---|---|---|---|---|---|
| 32 | **1.5517** | 1.3395 | 1.5128 | 0.8900 | 链 |
| 64 | **1.5517** | 1.4227 | 1.5128 | 0.8900 | 链 |
| **128** | 1.5517 | 1.4733 | **2.0799** | 1.6822 | **4 叉树（+34%）** |
| 256 | 1.5517 | 1.5040 | **2.0799** | 1.6822 | 4 叉树 |
| 512 | 1.5517 | 1.5227 | **2.5492** | 1.6822 | 4 叉树 |

agree=0.50 同理，也在 $n=128$ 翻转。

**正确的命题只对 $k=2$ 成立**：当 $c_2=c_1$ 时，2 叉树的渐近上限与链相同（同一个 $c/(1-c)$），而深度只有 $\log_2 n$ 对 $n$，所以 2 叉树**在任何预算下都不赢**。这条是对的。错的是把它推广到"任何 $k$"。

**② 复现**

```python
import sys; sys.path.insert(0, "_lab")
import numpy as np
from tree import zipf_dist, make_draft, coverage_topk, kary_depth, expected_accept
rng = np.random.default_rng(0)
p = zipf_dist(2000, 2.0, rng)
for ag in (0.95, 0.80, 0.50):          # 与 _shape() 同一 rng 消耗顺序
    q = make_draft(p, ag, rng)
cs = {k: coverage_topk(p, q, k) for k in (1, 2, 4, 8)}   # 这里 q 是 agree=0.50 那支
for budget in (64, 128, 256):
    print(budget, {k: round(expected_accept(cs[k], kary_depth(budget, k)), 4) for k in (1,2,4,8)})
```

**③ 影响范围**

- `16-树形草稿与树注意力-mask构造与验证.md` §5.3「流行说法二」的收尾句、§5.4 工程结论第 2 条
- `_lab/tree.py::_shape()` 第 229–230 行的读法 2
- `_lab/test_tree.py::test_width_is_worthless_without_marginal_coverage` 的 docstring 第一句
- 连带：§5.3 那句"两个条件合起来：树赢需要 (i) 边际覆盖率增益显著 **且** (ii) 预算足够大"——这句其实是**对的**，而"无论预算多大链都赢"与它自相矛盾（后者等于说 (ii) 不管用）。两句话在同一节里打架。

**④ 建议怎么修**

1. 把断言收窄成："当 $c_2=c_1$ 时，**加宽到 2 叉**在任何预算下都不赢；但只要存在某个 $k$ 使 $c_k>c_1$，预算够大时 $k$ 叉树仍会反超（本例 $c_4=0.828$，$n\ge128$ 时 4 叉树领先 34%）。"
2. §5.2 的表加一行 $n=128$，把反超点显式画出来 —— 这比删掉它更有教学价值：它正好演示"两个条件缺一不可"里的第 (ii) 条。
3. 测试改成断言**真正成立的那条**，并补一条**反向回归测试**钉住反超：

```python
def test_width_k2_is_worthless_without_marginal_coverage():
    """cov[2]==cov[1] 时，2 叉树在任何预算下都不赢（渐近上限相同、深度更浅）。"""
    ...
    for budget in (8, 16, 32, 64, 128, 256, 1024, 4096):
        assert expected_accept(c1, kary_depth(budget,1)) >= expected_accept(c2, kary_depth(budget,2))

def test_wider_tree_still_wins_at_large_budget_when_c4_exceeds_c1():
    """回归：c4>c1 时，预算够大 4 叉树必反超 —— '无论预算多大链都赢'是错的。"""
    ...
    assert expected_accept(c4, kary_depth(128,4)) > expected_accept(c1, kary_depth(128,1))
```

---

### 真-2 ⭐ `test_width_is_worthless_without_marginal_coverage` 是**单种子统计命题**：200 个种子里只过 87 个（43.5%）

**① 问题描述**

这条测试固定 `default_rng(0)`，`make_draft(p, agree=0.50, rng)`。换种子重跑，**200 个种子里只有 87 个能过（43.5%），seed=0 属于少数派**。

根因是 `make_draft` 的构造在 agree=0.50 与 0.80 上**恰好落在并列点**：

$q = \text{agree}\cdot p + (1-\text{agree})\cdot\text{other}$，两支都是同一组 Zipf(2.0) 值（$p_1=0.6081,\ p_2=p_1/4=0.1520$），只是排列不同。于是

| agree | top-1 之争 | top-2 之争 |
|---|---|---|
| 0.95 | $0.578$ vs $0.030$ —— 差 5.5e-1，稳 | $0.144$ vs $0.030$ —— 稳 |
| **0.80** | $0.486$ vs $0.122$ —— 稳 | $\mathbf{0.12160}$ vs $\mathbf{0.12162}$ —— **差 2e-5，抛硬币** |
| **0.50** | $\mathbf{0.30405}$ vs $\mathbf{0.30405}$ —— **完全相等，抛硬币** | — |

（agree=0.80 的并列是因为 Zipf(2.0) 恰好有 $p_2/p_1=1/4$，而 $0.2/0.8$ 也是 $1/4$。）

实测 400 个种子（agree=0.50）：

- `argmax(q) == argmax(p)` 的比例 = **0.472**（应该是 0 或 1，实际是抛硬币）
- `coverage_topk(p,q,1)` 的**中位数 = 0.0001**，有 **52.5%** 的种子里它 < 0.01
- 预算 64 下谁赢：链 189 / 树 211 —— **基本五五开**

也就是说，正文说"agree=0.50 时第二候选是垃圾"，而在**一半以上的种子里恰恰相反：第一候选才是垃圾，全部概率质量都在第二候选上**，树以压倒性优势赢。seed=0 抛到了正面。

这正是本库 `verify.py` 文件头自己列出的坑第 3 条（"只在一个样本上判统计命题 -> 结论随机"），只不过它落在 `_lab` 的实现里而不是自检器里。

**② 复现**

```python
import sys; sys.path.insert(0,"_lab")
import numpy as np
from tree import zipf_dist, make_draft, coverage_topk, kary_depth, expected_accept
ok = 0
for seed in range(200):
    rng = np.random.default_rng(seed)
    p = zipf_dist(2000, 2.0, rng); q = make_draft(p, 0.50, rng)
    c1 = coverage_topk(p, q, 1); gain = coverage_topk(p, q, 2) - c1
    good = gain / c1 < 1e-5
    if good:
        for b in (8, 16, 32, 64):
            if expected_accept(c1, kary_depth(b,1)) < max(
                   expected_accept(coverage_topk(p,q,k), kary_depth(b,k)) for k in (2,4,8)):
                good = False
    ok += good
print(ok, "/200")      # -> 87 /200
```

**③ 影响范围**

- `_lab/test_tree.py::test_width_is_worthless_without_marginal_coverage`
- `_lab/tree.py::_shape()` 打印的 agree=0.50 那一段读法
- 第 16 篇 §5.2 的 agree=0.50 行、§5.3「流行说法二」
- **注意**：`test_chain_beats_tree_at_small_budget` 与 `test_tree_beats_chain_at_large_budget`（都用 agree=0.95）我扫了 40 个种子，**0 个失败** —— agree=0.95 远离并列点，那两条是稳的。问题只在 0.50/0.80。

**④ 建议怎么修**

1. **换掉退化的 agree 取值**：0.50 与 0.80 都正好在并列点上。用 0.90 / 0.70 / 0.35 之类避开 $\{(1-a)/a\}$ 与 Zipf 相邻比 $1/4,\ 4/9$ 重合的点。
2. **或者**（更好）：不要靠 `agree` 间接控制覆盖率，直接**构造**一个 $c_2=c_1$ 的 $q$（把 $q$ 的第 2 名硬指到 $p$ 的一个尾部 token），这样命题的前提是被**构造**出来的而不是抽样抽出来的，测试就与种子无关。
3. 测试里加多种子断言（本库既有做法）：`for seed in range(20): ...`，并把分母写进 docstring。
4. 若保留 agree=0.50，正文必须改成"**在本例的这一次抽样里**"，并给出重复抽样的比例（189/400）。

---

### 真-3 `spec.py::residual_dist` 的 docstring 里两句断言**都不成立**

**① 问题描述**

`_lab/spec.py:36-41`：

```python
必须从残差分布采，否则输出分布有偏（见 test_spec.py::test_naive_resample_is_biased）。

退化情形：p == q 时 max(0,p-q) 全 0，此时接受概率恒为 1，残差分布永远用不到；
为了让函数总是返回合法分布，这里退回 p（并在测试里断言这条路不会被走到）。
```

三处不成立：

1. **`test_spec.py` 不存在。** `_lab/` 下没有这个文件；那条测试实际在 `test_lossless.py`。
   `verify.py --tests` 抓不到它 —— 因为 `check_tests()` 只扫 `ROOT.glob("*.md")`，**不扫 `_lab/*.py` 的 docstring，也不扫 `_meta/`、`_research/` 下的 md**。这是自检器的一个漏检面。
2. **"残差分布永远用不到"是假的。** 当 `draft is target`（同一个对象，例如自投机、或把 target 当自己的 draft 做单测）时，`iteration_dist(t, t, 0, gamma=2)` 就会走进兜底分支：`p_rej = 1.0 - np.minimum(p,p).sum()` 在浮点下等于 `1.11e-16 > 0`，于是进入拒绝分支并调用 `residual_dist(p, p)`，此时 `r.sum() == 0`，兜底触发。
   gamma=1 时 0 次，**gamma=2 时 1 次，gamma=3 时 7 次**（|V|=5, seed=0）。
   兜底返回 `p` 在数值上是**正确**的（那 1e-16 的质量本来就该按 p 分配），所以**没有产生错误结果** —— 但这段兜底是**承重的活代码，不是死代码**，文档把它写成了后者。
3. **"在测试里断言这条路不会被走到"—— 没有这样的测试。** 全库 grep 不到任何断言兜底不可达的测试。

**② 复现**

```python
import sys; sys.path.insert(0,"_lab")
import numpy as np, spec as S
hits = [0]; _o = S.residual_dist
def spy(p, q):
    if np.maximum(0.0, p-q).sum() <= 0.0: hits[0] += 1
    return _o(p, q)
S.residual_dist = spy
t = S.ToyMarkov(5, seed=0)
for g in (1, 2, 3):
    hits[0] = 0
    d = S.iteration_dist(t, t, 0, g, "correct")
    print("gamma=%d sum=%.15f fallback_hits=%d" % (g, sum(d.values()), hits[0]))
# gamma=1 ... 0 / gamma=2 ... 1 / gamma=3 ... 7
```

（顺带：我用 300 组**含硬零概率**的随机马尔可夫模型 —— 即"某 token 上 q=0 而 p>0"的情形 —— 逐个验证了 `iteration_dist` 概率和为 1、首 token 逐点无损，兜底 **0 次命中**。所以在 $p\ne q$ 的一切情形下兜底确实走不到；能走到的只有 $p\equiv q$ 的浮点残差。）

**③ 影响范围**

不影响任何数值结论（兜底行为恰好正确）。影响的是**文档诚实度**——本库的卖点就是"断言都可执行验证"，而这里有一句"在测试里断言了"是空头支票，还引了一个不存在的文件。第 04 篇与第 27 篇都引用 `test_naive_resample_is_biased`，引的是对的路径（`test_lossless.py`），只有 `spec.py` 的 docstring 引错。

**④ 建议怎么修**

1. `test_spec.py::` → `test_lossless.py::`。
2. 把"永远用不到 / 测试里断言不会被走到"改成事实描述：
   > 退化情形：`p == q` 时 `max(0,p-q)` 全 0。数学上此时接受概率恒为 1、残差用不到；
   > 但浮点下 `1 - sum(min(p,q))` 可能是 `+1e-16`，于是**当 draft 与 target 是同一模型时这条路会被走到**（|V|=5,γ=3 时 7 次）。
   > 兜底返回 `p` 在数值上是正确的分配。回归测试见 `test_lossless.py::test_residual_fallback_only_fires_when_p_equals_q`。
3. 补一条真的回归测试（钉住"只有 $p\equiv q$ 才会触发，且触发时结果仍无损"）：

```python
def test_residual_fallback_only_fires_when_p_equals_q():
    import spec as S
    hits = [0]; orig = S.residual_dist
    def spy(p, q):
        if np.maximum(0.0, p - q).sum() <= 0.0: hits[0] += 1
        return orig(p, q)
    S.residual_dist = spy
    try:
        tgt = S.ToyMarkov(5, seed=0)
        drf = S.perturb(tgt, eps=1.2, seed=1)          # p != q
        S.iteration_dist(tgt, drf, 0, 3); assert hits[0] == 0
        hits[0] = 0
        d = S.iteration_dist(tgt, tgt, 0, 3)           # p == q
        assert hits[0] > 0                              # 兜底确实是活代码
        assert abs(sum(d.values()) - 1.0) < 1e-12       # 且结果仍然合法
    finally:
        S.residual_dist = orig
```

4. 顺手把 `verify.py --tests` 的扫描面扩到 `_lab/*.py` 的 docstring 与 `_meta/`、`_research/*.md`（本条就是被这个漏检面漏掉的）。

---

### 真-4 `crossover_batch` 不校验显存，`--quant` 与 `test_long_context_still_no_crossover_even_quantized` 的"扫到 8192"落在**物理不可能区**

**① 问题描述**

`speedup.py::crossover_batch(..., bmax=8192)` 在 `[1, 8192]` 上二分找交叉点，**完全不调用 `feasible()`**。同一文件里 `_batch()` 是**校验**显存的（装不下就打 `"%.0fGB 装不下"`），`_quant()` 与 `_chunked()` 不校验。于是：

`_quant()` 的 seqlen=16384 那六行全部打印 `>8192`，而在同一口径（8×H100 = 640 GB）下，那些配置的**最大可行 batch 是 84–204**：

| 权重 | KV | seqlen | 打印的交叉点 | 实际最大可行 batch |
|---|---|---|---|---|
| fp16 | fp16 | 16384 | `>8192` | **84** |
| fp16 | fp8 | 16384 | `>8192` | 168 |
| fp8 | fp16 | 16384 | `>8192` | 96 |
| fp8 | fp8 | 16384 | `>8192` | 192 |
| int4 | fp16 | 16384 | `>8192` | 102 |
| int4 | fp8 | 16384 | `>8192` | **204** |

`test_long_context_still_no_crossover_even_quantized` 的 docstring 写"扫到 8192 仍未见交叉点"——**扫描区里 97.5% 的格子在物理上不存在**。

**结论本身仍然成立**（在最大可行 batch=204 处实测 speedup=2.039，仍然赚），但**证据的口径是裸奔的** —— 这正是本库铁律二要禁止的那类表述，而且同一个文件里的另一个函数（`_batch`）已经做对了。内部不一致。

**② 复现**

```python
import sys; sys.path.insert(0,"_lab")
from speedup import *
hw = scale(H100, 8)
W = (MODELS["llama3-70b"]["P"] + MODELS["llama3.2-1b"]["P"]) * 0.5
per = 16384 * (MODELS["llama3-70b"]["kv_per_tok"] + MODELS["llama3.2-1b"]["kv_per_tok"]) * 0.5
print("最大可行 batch =", int((hw["hbm"] - W) / per))                      # 204
print("crossover     =", crossover_batch("llama3-70b","llama3.2-1b",hw,16384,4,3.0,0.5,0.5))  # None
print("在 204 处：", spec_throughput("llama3-70b","llama3.2-1b",hw,204,16384,4,3.0,0.5,0.5)["speedup"])  # 2.039
```

**③ 影响范围**

- `_lab/speedup.py::crossover_batch`、`_quant()`、`_chunked()`
- `_lab/test_speedup.py::test_long_context_still_no_crossover_even_quantized`
- 第 23 篇 §4.1 的表（seqlen=1024 那几行没问题，265/133/67 都远在显存墙之内）与「本篇验证」第 339 行的那句"扫到 batch 8192 仍未见交叉点"

**④ 建议怎么修**

1. 给 `crossover_batch` 加一个 `bmax` 的**物理上界**：默认 `bmax = min(8192, max_feasible_batch(...))`，或者返回 `(crossover, bmax_used, feasible_bmax)` 三元组，让调用方能打印"扫描区上界是显存墙 204，非 8192"。
2. `_quant()` 的最后一列改成 `">204（显存墙）"` 而不是 `">8192"`。
3. 测试改成断言**可行区内**没有交叉点，并把可行上界写进断言：

```python
def test_long_context_still_no_crossover_within_the_memory_wall():
    hw = scale(H100, 8)
    bmax = max_feasible_batch("llama3-70b", "llama3.2-1b", hw, 16384, 0.5, 0.5)
    assert 150 < bmax < 260, bmax                       # 显存墙在两百上下
    assert crossover_batch(..., bmax=bmax) is None
    assert spec_throughput(..., bmax, ...)["speedup"] > 1.5
```

---

### 真-5 `moe.py` 的 attention FLOPs 用 `d_model` 当 $n_{\text{heads}}\times d_{\text{head}}$ 的代理 —— 对 gpt-oss 低估 1.42×、对 DeepSeek-V3 低估 2.86×

**① 问题描述**

`moe.py::fwd_time_moe` 与 `tokens_to_saturate_moe` 都用

```python
flops = n_q * (2 * active_params(m) + 4 * m["layers"] * seqlen * m["d_model"])
```

`4·L·s·d` 这个式子里的 `d` 正确含义是 $n_{\text{heads}}\times d_{\text{head}}$（QK^T 与 PV 各 $2\,s\,n_h d_h$），**不是 `hidden_size`**。两者对 Llama 系恰好相等，所以 `speedup.py` 里是对的；但对本文件的两个 MoE 模型都**不相等**：

| 模型 | $n_h\times d_h$ | `d_model` | 低估倍数 |
|---|---|---|---|
| llama3-8b / 70b / 3.2-1b | 4096 / 8192 / 2048 | 4096 / 8192 / 2048 | 1.00 ✅ |
| **gpt-oss-120b** | $64\times64=4096$ | 2880 | **1.42×** |
| **deepseek-v3**（MLA） | QK 用 $128\times192$、PV 用 $128\times128$，等效 40960 | $4\times7168=$ 等效 14336 | **2.86×** |

方向是**低估算力**→ 高估 memory-bound 区 → **让投机采样看起来更好**。这与本文件其余部分刻意保持的"乐观上界"方向一致，但这一条是**算错了**而不是"简化"，且文件里没有任何注释声明。

**② 复现 / 影响量化**

```python
# 免费额度：现行(用 d_model) vs 修正(用 n_h*d_h / MLA 真实维)，8xH100 TP=8, batch=64
gpt-oss-120b  s=1024    1858  vs 1828   高估   1.7%
gpt-oss-120b  s=8192    2187  vs 1978   高估  10.5%
gpt-oss-120b  s=32768   2690  vs 2167   高估  24.1%
deepseek-v3   s=1024    2603  vs 2345   高估  11.0%
deepseek-v3   s=8192    2344  vs 1334   高估  75.7%
deepseek-v3   s=32768   1829  vs  601   高估 204.4%
```

**③ 影响范围**

- **当前正文结论不受影响**：`moe.py` 的全部打印函数与测试都只用 seqlen=1024，那里误差 1.7%–11%，`--redhat` 表逐格重算后**没有一个格子翻转**（batch=200 仍是 memory-bound、speedup 仍是 2.846）。所以第 18 篇、第 23 篇 §4.7 的结论是安全的。
- **但这是个定时炸弹**：本库最看重的正是"长上下文对投机友好"这条，而 `moe.py` 一旦被拿去算 s≥8192，DeepSeek-V3 的免费额度会被高估 76%–204%。而且这跟本库已经踩过一次的坑（attention FLOPs 漏乘层数 L）是**同一类错误的另一个维度**：那次漏的是 $L$，这次错的是 $d$。`test_speedup.py::test_attention_flops_include_layer_count` 钉住了 $L$，**没有任何测试钉住 $d$**。

**④ 建议怎么修**

1. `MOE_MODELS`（以及 `MODELS`）加一个显式字段 `attn_dim`（= $n_h\times d_h$；MLA 按 $n_h(d_{qk}+d_v)/2$ 折算），`flops` 改用它；Llama 系填成与 `d_model` 相等，行为不变。
2. 补一条与 `test_attention_flops_include_layer_count` 对称的回归测试：

```python
@pytest.mark.parametrize("name,n_heads,head_dim", [
    ("llama3-8b", 32, 128), ("llama3-70b", 64, 128), ("llama3.2-1b", 32, 64)])
def test_attention_flops_use_head_dim_product_not_hidden_size(name, n_heads, head_dim):
    """回归：attention 项的宽度必须是 n_heads*head_dim。
    Llama 系恰好 == d_model，但 gpt-oss(4096 vs 2880) 与 MLA 不是 —— 那次是靠这条抓出来的。"""
    assert MODELS[name]["attn_dim"] == n_heads * head_dim
```
3. 在 `moe.py` 文件头的口径声明里补一句"本文件的数字只在 seqlen≈1k 下核对过；长上下文请先修 attn_dim"。

---

## 2. 疑似 / 口径问题（3 个，报 WARN 不报 ERROR）

### 疑-1 `2*P` 里含 embedding 表，但 embedding 是 gather 不是 matmul

`fwd_time` 的 `flops = n_q*(2*P + ...)` 与 `mem_bytes = P*bytes_per_param` 都把 `embed_tokens` 算了进去。embedding lookup 只读被用到的那几行，也不做矩阵乘。对 llama3-70b 这是 $2\times1.05\text{B}/70.6\text{B}\approx3\%$ 的高估，对 llama3.2-1b（tie_word_embeddings=true）反而是对的（那张表被 lm_head 真读一次）。量级在噪声内，但文件头的"忽略 norm/softmax/kernel launch"清单里没提这一条，建议补一句。

### 疑-2 `prehistory.py` 的"最优独立近似"随机试探是**弱检验**

`test_best_independent_approximation_is_product_of_marginals` 用**均匀随机** Dirichlet 当备选去比，4000 次无一更优 —— 但随机备选离最优极远，这个"无一更优"几乎是必然的，检验强度很低。我改用**近最优扰动**（在最优边缘上加 σ=0.01 高斯噪声再归一化）打了 9000 次，同样 0 次击败、最大改进 0.000e+00 —— **命题本身稳固**，只是测试写松了。建议把随机备选换成近最优扰动（同样便宜，检验强度高一个量级）。真正的证明其实是同文件那条恒等式测试 `test_independent_factorization_cost_equals_total_correlation`，那条是硬的。

### 疑-3 H100 的 989.5 TFLOPS 是 1979/2 的算术值

联网核验（见 §3）确认了**载重的那一半**：NVIDIA H100 产品页 BFLOAT16 行的 1,979 带星号，脚注是 "With sparsity" —— 所以本库"1979 含稀疏、不许裸奔"这条**站得住**。但 989.5 本身没能从 NVIDIA 官方文档里直接取到（whitepaper PDF 链接全部 404/302），而 Hopper whitepaper 被广泛引用的数字是 **989.4**。建议写成"989.5（≈ NVIDIA 白皮书表里的 989.4）"，或把白皮书那一行标"未查证"。差 0.01% 不影响任何结论。

---

## 3. 攻击过但**没攻破**的（11 项 —— 这些地方经得起打）

| # | 攻击点 | 攻法 | 结果 |
|---|---|---|---|
| 1 | `iteration_dist` 四个 variant 的概率是否和为 1 | $V\in\{2..5\}\times\gamma\in\{1,2,3\}\times\epsilon\in\{0,.5,2,6\}\times$ 全部起点 × 4 variant 穷举 | **全部 = 1（tol 1e-12）**。`threshold` 分支的拒绝概率 $\sum_{p<q}q(x)$ 与接受概率 $\sum_{p\ge q}q(x)$ 互补，算对了 |
| 2 | `residual_dist` 兜底在"$q=0$ 而 $p>0$"时是否可达 | 300 组随机打洞（35%/50% 格子硬置 0）的马尔可夫对，逐起点逐 γ 枚举 | **0 次命中**；且首 token 逐点无损（<1e-12）。只有 $p\equiv q$ 的浮点残差能触发（见真-3） |
| 3 | `no_bonus` 变体真的无偏吗（不只是首 token） | 对**整条流**（前 3 token 联合分布）算最大逐点误差 | **≤2.2e-16**，全 18 组配置。"不发赠品只慢不错"成立 |
| 4 | `marginal_gamma` 的判据 $\alpha^{g+1}>c\cdot\text{speedup}(g)$ | ① 手推：$\frac{N+\alpha^{g+1}}{D+c}>\frac ND \iff \alpha^{g+1}>c\frac ND$，精确等价，不是近似；② 20 万次随机 $(\alpha,c,g)$ 比对判据与真实增量方向；③ 199,800 格 $(\alpha\in[.001,.999],c\in[.001,.2])$ 对暴力搜索 | **不一致 0 次 / 0 格**。另扫了 $\alpha\to1$、$\alpha\to0$、$c\to0$、$c=10^6$ 四个边界，行为都正确（$\alpha=1,c<1$ 时无上界 → 返回 gmax，与暴力一致） |
| 5 | `speedup_ideal(g)` 是否真的单峰（判据靠这个） | $\alpha\in[.30,.99]\times c\in[.005,.2]$ 共 2760 格，每格扫 $g=1..300$ 找局部极大 | **无一多峰** |
| 6 | 更新过程闭式的分母该不该是 $E[\tau]$ | 重推：token 流按区制分解 $\pi_r = (\text{起点在 }r\text{ 的迭代数})\times E[\tau|r]$，反解得 $P(\text{起点}\in r)\propto \pi_r/E[\tau|r]$ | **分母就该是 $E[\tau]$**，这是长度偏置采样的标准形式，代码写对了。而且本库自己已经把"该式只在高粘性下准"做成了断言（`test_renewal_formula_accurate_only_when_sticky`），态度是对的 |
| 7 | `verify_tree_equals_chains` 是不是只比叶子 | 读代码：`for t, node in enumerate(path)` 遍历路径**全部**节点；7 节点样例树的 3 条路径覆盖全部 7 个节点。再给**内部**节点 1 注入 1e-3 误差 | **抓得到（返回 1.0e-3）**。docstring 说"对比叶子节点"是**说少了**，实际检查比宣称的强。建议只改 docstring |
| 8 | 券收集公式 $E(1-(1-k/E)^N)$ 是否因 top-k **不放回**而失效 | ① 理论：均匀随机 $k$-子集下 $P(\text{某专家未被选})=1-k/E$ **精确**成立，期望的线性性吃掉专家之间的相关性；② 向量化蒙特卡洛 $E\in\{4,8,128,256\}\times N\in\{1,2,5,20,100\}$，各 20000 次 | **20 组全部落在 4 SE 内**。公式是**精确**的，不是近似。（另注：逐层独立路由不改变期望值，因为期望对层数线性可加） |
| 9 | `tokens_to_saturate_moe` 的不动点会不会振荡 / 收敛到错解 | ① 320 组参数（2 模型 × 2 硬件 × 5 batch × 4 seqlen × 4 精度）检查返回值是否满足 $f(n)=n$；② 16 个代表格做 $[10^{-3},10^{9}]$ 全域根扫描找多解；③ 人为造极端格（E=4096、单专家 200 GB、算力项极小） | **320/320 都是真不动点；全域扫描每格恰好 1 个根；极端格也只有 1 个根且迭代收到它**。阻尼迭代是稳的 |
| 10 | 五个模型规格（P、层数、d_model、KV/token、激活参数） | 派子代理联网核 HuggingFace `config.json` / 官方 model card / 论文 | **无一为错**。llama3-8b 8,030,261,248；llama3-70b 70,553,706,496；llama3.2-1b 1,235,814,400（tie=true）；gpt-oss-120b 116,829,156,672 / 5.1B active / 128 专家 top-4 / 36 层 / SWA 128 交替；DeepSeek-V3 671B/37B/256+1 专家/61 层/MLA 576 元素。全部对上。H100 的 1979 含稀疏一条也核实了 |
| 11 | `caliber.py` 的 L2 贪心等价与 L3 阈值判据 | 逐行验算 `greedy_speculative` 的接受/拒绝两条路是否都吐 `argmax(p)`；核对 `_typical` 的三条读法（p 最小分量 0.1045 → 阈值 ≤0.10 时判据不生效 → 误差恒 0.1719） | **推导与打印数字全部对得上**。首 token 的边缘分布只由第 0 层决定，所以"看 p(.\|s=0) 的最小分量"这个论证是严谨的 |

另外两条我特意去打但没打破的"反直觉结论"：

- **第 06/19 篇的"公式高估而非低估"**：`test_formula_overestimates_when_sticky` 的两档 (0.50, [0.97,1.06]) 与 (0.99, [0.85,0.97])，n=40000 时 SE 约 0.3%，容差是 10 SE 量级，方向判定稳固。
- **第 18 篇的 compute-bound 渐近线 $E[\tau]/(\gamma+1)$**：`test_compute_bound_asymptote_equals_wasted_compute_ratio` 的恒等式我重推了一遍——在 baseline 也进入 compute 区时该式是**精确**的（$t_{\text{verify}}=(\gamma+1)t_{\text{base}}$），容差 0.01 绰绰有余。

---

## 4. 我独立找到、但审稿期间被并发编辑抢先修掉的（交叉验证留档）

这 5 条我是在 21:31 读到的旧版本上独立发现的，写报告时发现已被改。列出来是因为它们**互相印证**（两个独立审稿路径撞上同一批问题，说明这批问题是真的）：

| # | 问题 | 我的独立测算 | 现状 |
|---|---|---|---|
| 4-1 | `deepseek-v3` 的 `P_dense=13.0e9` 反推出激活 **33.56B**，而官方是 37B（低 9.3%）。由 $P_d+8x=37,\ P_d+256x=671$ 解得 $P_d\approx16.55$ | 我算出应为 16.5–17.0e9，并按 MLA/attention 逐项重建了 671B 的分解验证 | **21:34 已改为 17.0e9** |
| 4-2 | `test_expert_decomposition_is_self_consistent` 断言 `P_dense + n_experts*expert_size == P_total` 是**恒等式**（`expert_size` 就是这么定义的），**怎么填 P_dense 都过** —— 典型的空检查，正是它放跑了 4-1 | 我判定为"太松的测试" | **21:34 已补 `test_expert_decomposition_matches_published_active`**，正是我建议的那条 |
| 4-3 | `deepseek-v3` 的 `kv_per_tok = 2*61*512*2`。MLA 缓存的是**单个联合压缩潜向量** $512+64=576$，不是分开的 K 和 V，正确值 `61*576*2 = 70272`（旧值高估 1.78×） | 联网核验确认 576 与"非分离 K/V" | **21:34 已改**。子代理还额外提示：该 config 里 `num_key_value_heads: 128` 是 MLA 下的**遗留字段**，用它算 KV 会高估 ~28×，值得在正文加个警告 |
| 4-4 | `test_weight_quantization_shrinks_the_usable_batch_range` 的 docstring 写"交叉点 fp16 273 / fp8 137 / int4 69"，实际是 **265 / 133 / 67**（旧值是 attention FLOPs 漏乘 L 修复**之前**的） | 我逐格重算确认 265/133/67；正文第 23 篇 §7 用的是 265/67，是对的，只有测试 docstring 是旧的 | **21:47 已改** |
| 4-5 ⭐ | `verify.py` 的 `SPEEDUP_PAT = ...(?:[×xX]\b\|倍...)`：`×` 是非词字符，`×\b` 要求右邻必须是词字符，于是 `2.8×，` `2.8× ` `2.8×）` `**2.8×**` 全部漏检 | 我逐篇统计：全库形如 `N.N×` 的写法共 **894 处，铁律二只命中 1 处（漏检 99.9%）**；把 `\b` 去掉后铁律二 WARN 从 15 条涨到 **83 条（新增 68 条）** | **22:00 已写进 verify.py 文件头的"已知假阴性"并给出 63 处的实测，但明确标注"尚未修"**。我的独立测算（68）与他们的（63）在同一量级，互证 |

对 4-5 的补充意见（这条仍未修，是**当前最大的自检漏洞**）：文件头写的理由是"多数是表格行，口径写在几行之外的表注里，属于窗口太窄而非真裸奔"。我抽看了新增的 68 条，同意其中相当一部分是窗口问题（例如第 09 篇 Table 2 的六行，batch 口径写在表格上方的段落里）。但也有真裸奔的，例如：

- `08 §:268` "GPU 上最好只有 1.06–1.08×"
- `10 §:164` "**理想加速比 1.376×，而这是上界**"、`10 §:333` 同一数字
- `12 §:97/:105` Vicuna-7B 的 2.18× / 2.83×

建议按文件头自己写的方案办：**把上下文窗口从 ±6 行改成"到最近的 `##`/`###` 小节边界"**，再逐条复核。在那之前，`verify.py --rules` 报的"铁律二 0 处"这个数字**不应被当作铁律二已达标的证据**。

---

## 5. 为什么本次审稿没有改代码

任务允许"100% 确定且已用测试证明的错误可以直接修 + 补回归测试"。我没有改，原因有三：

1. **并发编辑正在进行**（§0）。目录不在 git 下，我改任何一个正在被对方改的文件都可能被静默覆盖，或反过来覆盖对方。
2. **真-1/真-2 的修法不是改代码，是改结论**（第 16 篇 §5.3/§5.4 的两句断言 + 表格加一行 + 测试重写）。这属于编辑决策，应由作者拍板取哪一版措辞，而不是审稿人替他改。
3. 真-3/真-4/真-5 都附了**可直接粘贴的补丁与回归测试**，改动面小、语义明确，作者按 §1 的"④ 建议怎么修"落地即可。

审稿全程**只读**，未产生任何代码改动；注入式检验（§ verify.py 那一节）用的 7 个临时 md 文件已在同一脚本的 `finally` 里全部删除，已核对目录洁净。

---

## 6. 复跑

本报告所有数字都可复跑。全套探针脚本在
`C:\Users\jianm\AppData\Local\Temp\claude\C--Users-jianm\3ff8b823-9e11-474b-96cd-c6e11996dd05\scratchpad\`
（`a1.py` 枚举与兜底、`a2.py` MoE attention 维、`a3.py` 显存墙、`a4.py` 边际判据、`a5.py` 不动点、`a6.py` 树等价+券收集、`a7–a10.py` 树形状的种子稳健性），临时目录会被回收，各条的核心片段已内联在上文的"② 复现"里。

基线：`cd _lab && python -m pytest -q` —— 审稿开始 200 passed，结束 233 passed（增量非本次审稿产生），无 failed、无 skipped。
库根目录已核对洁净：32 个 md，无注入残留文件。
