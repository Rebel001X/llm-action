# 并行策略：TP / PP / EP

> **一句话**：模型装不下一张卡，只有三个方向能切。
> **前置**：[[02-自回归解码为什么是访存瓶颈]]（本篇的通信量账、气泡账，全部建立在那篇的算术强度框架之上）

## 0. 这一篇要解决什么问题

从 `01-地基/` 到 `02-最小可跑引擎/` 为止，本库所有代码都悄悄假设了一件事：**整个模型的权重、激活、KV 缓存，都能放进一张卡的显存里**。`_lab/minigpt.py` 的 `init_weights()` 一次性把所有层的权重摆在一个进程的内存里，`_lab/paged.py` 的 `BlockAllocator` 管理的也是单一显存池的块。这个假设在教学玩具上成立，在真实场景里很快就会碎——一个几百亿参数的模型，权重本身就可能超过单卡显存；即便权重装得下，`04-连续批处理.md` 讲过的"用 batch 换算术强度"这条杠杆一旦拉起来，KV 缓存也会迅速把剩余显存吃光。

装不下的时候，出路只有一个：**把模型切成几份，分给几张卡**。但"切"不是一个笼统的动作，具体到一个 Transformer 的计算图上，只有三个方向可以切：

1. **切"同一层内部"**——一次矩阵乘、一次注意力，拆给几张卡各算一部分，再合并。这是**张量并行（Tensor Parallelism，TP）**。
2. **切"不同层之间"**——第 1~8 层给卡 A，第 9~16 层给卡 B，激活像流水线一样从 A 传到 B。这是**流水并行（Pipeline Parallelism，PP）**。
3. **切"不同专家之间"**——如果模型是 MoE，不同的专家网络放在不同卡上，token 按路由结果被送到对应的卡。这是**专家并行（Expert Parallelism，EP）**，只对 MoE 模型有意义。

本篇要回答的不是"三种切法分别怎么写代码"（那是工程实现，`## 4` 会看真实引擎怎么做），而是三个更根本的问题，每个都能在单机 numpy 上给出**可以手算复核**的答案：

- **切完还等不等价？** TP 和 EP 本篇都给了逐位对比（bit-for-bit 相同，不是"数值上差不多"）；PP 天然等价，因为它只是把同一串计算拆开执行，不改变计算本身。
- **通信量是多少？** 每种切法都要在卡之间传数据，传多少、什么时候传，直接决定了这种切法能不能在给定的网络条件下工作。
- **代价是什么，且代价怎么随"切得更细"变化？** TP 切得越细通信占比越高，PP 切得越多气泡越大，EP 切得越不均衡浪费越多——三条代价曲线形状不同，这是选型时真正要看的东西。

**红线声明**：本机是单机、无 GPU、没有任何真实互联。本篇计算的**元素数、字节数、气泡比例**，全部是可以拿计算器复核的纯计数，**不包含任何真实带宽、延迟、加速比**——那些数字只有在真卡上实测才有意义，教学库编不出来，也不该编。`_lab/parallel.py` 顶部的模块文档字符串（`_lab/parallel.py:20-21`）把这条红线写在了代码里，不是本文单方面的免责声明。

顺序：先看三张能跑出来的表（`## 1`），再逐一讲清楚每种切法"为什么必须这样切"（`## 2`），然后走读 `_lab/parallel.py` 的实现（`## 3`），最后对照真实引擎、讲清楚设计取舍与踩坑（`## 4`~`## 6`）。

## 1. 先看现象

进入 `_lab/` 跑一遍自检：

```bash
cd _lab
python parallel.py --selftest
```

`selftest()`（`_lab/parallel.py:173-259`）里压着 11 条断言，全部通过时打印"全部通过"。这 11 条覆盖了本篇要讲的每一个硬结论：TP 在 tp=1/2/4/8 下逐位等价、GELU 非线性证明切法不能反过来、通信量系数从 1.0 涨到 1.75、注意力按头切的等价性、头数除不尽时的报错、流水气泡公式与模拟逐一吻合、气泡随 micro-batch 数暴涨、MoE 路由不丢 token、负载不均衡度的对比、top-2 路由总负载翻倍。这些数字下面会逐条摊开讲。

再跑一遍不带 `--selftest` 的版本，看三张可手算复核的表：

```bash
cd _lab
python parallel.py
```

**第一张表：TP 的通信量**（`tokens=2048, d_model=4096` 时，每次 all-reduce 每卡收发的元素数）：

```
  tp    每次 all-reduce 元素数    相对 tp=2
   1                       0        0.00
   2               8,388,608        1.00
   4              12,582,912        1.50
   8              14,680,064        1.75
  16              15,728,640        1.88
```

**第二张表：PP 的气泡比例**（推理期只有前向，各级等长）：

```
  级数 P   micro M   总步数     气泡
     2         2        3     33.3%
     4         4        7     42.9%
     4         8       11     27.3%
     4        32       35      8.6%
     8         8       15     46.7%
     8        64       71      9.9%
    16        16       31     48.4%
```

**第三张表：EP 的负载不均**（8 个专家，4096 个 token，top-1，`skew` 越大路由越偏向前几个专家）：

```
  偏斜    最轻    最重   max/mean   等效浪费
   0.0     483     524      1.02        2%
   0.2     238     945      1.85       85%
   0.4      85    1422      2.78      178%
   0.8      10    2272      4.44      344%
   1.5       0    3184      6.22      522%
```

三张表分别对应三个"代价曲线形状不同"的例子：TP 的通信系数从 1.00 缓慢爬到 1.88（上界是 2.00，永远到不了）；PP 的气泡随 micro-batch 数增大而**剧烈**下降（46.7% → 9.9%，只差在 M 从 8 变到 64）；EP 的浪费倍数随路由偏斜**近乎线性**上升，偏斜到 1.5 时浪费已经超过 5 倍。`## 2` 把这三条曲线各自的成因用能手算的方式推一遍。

## 2. 原理

### 2.1 TP 的核心洞察：切法不是随便挑的，是被 GELU 逼出来的

一个 Transformer 的 MLP 块是 `y = gelu(x @ w1) @ w2`，`x` 形状 `(T, d)`，`w1` 形状 `(d, h)`，`w2` 形状 `(h, d)`。要把这计算摊到 `tp` 张卡上，`w1`/`w2` 该怎么切？

**关键观察一：矩阵乘的"列切"不产生部分和。** `x @ w1` 的第 `j` 列只依赖 `w1` 的第 `j` 列：`(x @ w1)[:, j] = sum_k x[:, k] * w1[k, j]`，跟 `w1` 的其它列毫无关系。所以把 `w1` 按列切成 `tp` 份 `w1_1, ..., w1_tp`（每份形状 `(d, h/tp)`），每张卡算出的 `x @ w1_r` 是**完整的、独立的**一部分隐藏单元——不是"需要跟别的卡加起来才对"的半成品。既然是完整的值，**GELU 可以就地逐元素做，不需要等任何别的卡**：`gelu(x @ w1_r)` 逐元素成立，因为 GELU 本来就是逐元素函数。

**关键观察二：矩阵乘的"行切"必须产生部分和。** 把上一步得到的 `h_r = gelu(x @ w1_r)`（形状 `(T, h/tp)`）过 `w2` 时，如果把 `w2` 按行切成对应的 `tp` 份 `w2_1, ..., w2_tp`（每份形状 `(h/tp, d)`），那么每张卡只能算出 `h_r @ w2_r`——这只是完整输出 `h @ w2` 里"贡献了 `h` 的一部分维度"的**部分和**，必须把 `tp` 张卡的结果按元素加起来才是真正的输出：`y = sum_r (h_r @ w2_r)`。这一步的求和就是**一次 all-reduce**。

**这两步合起来，整个 MLP 块只需要一次通信**——`w1` 列切、`w2` 行切，中间的 GELU 不用通信，只有最后要把 `tp` 份部分和加起来。`_lab/parallel.py:44-69` 的 `mlp_tensor_parallel()` 就是这套逻辑的精确实现：

```python
# _lab/parallel.py:61-69
d, h = w1.shape
assert h % tp == 0, "隐藏维必须能被 tp 整除"
parts = []
for r in range(tp):
    w1_shard = w1[:, r * h // tp:(r + 1) * h // tp]     # 按列切
    w2_shard = w2[r * h // tp:(r + 1) * h // tp, :]     # 按行切
    parts.append(gelu(x @ w1_shard) @ w2_shard)          # 本地算，无通信
out = sum(parts)                          # <- 唯一的一次 all-reduce
```

**反过来切会怎样？** 如果贪图"对称"，把 `w1` 也按行切（即按 `d` 维切，配合把 `x` 也切成 `tp` 份），每张卡算出的 `x_r @ w1_r` 就是`x @ w1` 的部分和——**在做 GELU 之前，必须先把这个部分和 all-reduce 出来**，因为 GELU 是非线性的：`gelu(a + b) != gelu(a) + gelu(b)`。这不是"效率差一点"，是**数学上不允许**：如果不在 GELU 前通信，每张卡算出的 `gelu(x_r @ w1_r)` 加起来之后，跟真正的 `gelu(x @ w1)` 完全是两个不同的函数值。`selftest()` 里专门有一条断言把这个不等式钉死：

```python
# _lab/parallel.py:198-201
a = rng.normal(0, 1, (4, 4)).astype(np.float32)
b = rng.normal(0, 1, (4, 4)).astype(np.float32)
chk(not np.allclose(gelu(a + b), gelu(a) + gelu(b), atol=1e-3), ...)
```

所以"错误"的切法（`w1` 按行切）不是慢一点，是要**多付一次通信**：GELU 前一次 all-reduce（把部分和加成真值）、`w2` 之后再一次（合并最终输出）——两次通信换到跟"正确"切法完全相同的结果。**Megatron 那套"列切+行切"的组合不是任选的约定，是"只用一次通信就能拿到正确结果"的唯一解**——这是本篇最容易被说反的一条：直觉上"对称地切"听起来更公平，但对称在这里恰恰是错的。

### 2.2 注意力按头切：天然对齐，但 GQA 之后约束更紧

多头注意力天然适合这套逻辑：`n_head` 个头彼此在 `softmax(QK^T)V` 这一步完全互不干扰（`_lab/parallel.py:85-92` 的 `attn_single()` 里每个头独立算完再拼接），所以按头切跟 MLP 的"列切"是同一个道理——每张卡拿 `n_head/tp` 个头的完整 Q/K/V 权重，算出的是这几个头**完整**的注意力输出，不是部分和；只有最后的输出投影 `wo` 需要按行切、all-reduce：

```python
# _lab/parallel.py:106-118（节选）
hpr = n_head // tp
for r in range(tp):
    sl = slice(r * hpr * dh, (r + 1) * hpr * dh)
    q = (x @ wq[:, sl])...
    ...
    parts.append(o.transpose(1, 0, 2).reshape(t, hpr * dh) @ wo[sl, :])
return sum(parts)
```

**硬约束**：`n_head % tp == 0`（`_lab/parallel.py:108`）。这不是实现偷懒——头是不可再分的计算单元（`## 2.1` 的"列切"只在"这一列独立、不依赖别的列"时才不用通信，头之间也是这个性质，但**头内部**再往下切没有意义），所以 TP 度只能取能整除头数的值：`n_head=8` 时能取 1/2/4/8，取不了 3、5、6、7。`selftest()` 用 `tp=3` 真实触发了这条断言（`_lab/parallel.py:223-227`）。

**GQA（分组查询注意力）之后，约束更紧一层。** GQA 下 KV 头数远少于 Q 头数（比如 32 个 Q 头共享 8 个 KV 头），这时候还要求 `n_kv_head % tp == 0` 才能把 KV 头也切开；如果 `tp` 比 KV 头数还多，就只能把每个 KV 头复制给多张卡（`tp % n_kv_head == 0`）而不是切分它。本库的 `attn_tensor_parallel()` 没有实现 GQA 分支（`_lab/parallel.py` 只处理 Q/K/V 头数相等的朴素多头），`## 4` 会看真实引擎怎么处理这层更紧的约束。

### 2.3 通信量：为什么切得越细，每卡通信量不降反升

一次 ring all-reduce 把 `N` 个元素在 `tp` 张卡间同步，标准结果是**每张卡收发 `2 * (tp-1)/tp * N` 个元素**——这个系数由两阶段组成：先做一轮 reduce-scatter（每卡收发约 `(tp-1)/tp * N`），再做一轮 all-gather（同样约 `(tp-1)/tp * N`），加起来正是 `2(tp-1)/tp`。`_lab/parallel.py:72-80` 的 `allreduce_elements()` 直接实现这个公式：

```python
# _lab/parallel.py:79-80
n = tokens * d_model
return 0 if tp <= 1 else int(2 * (tp - 1) / tp * n)
```

这个系数**随 `tp` 增大单调上升，且上界是 2.00**：`tp=2` 时系数恰好 1.00，`tp=8` 时 1.75，`tp=16` 时 1.875——`## 1` 那张表就是这个函数的取值。**关键不在"系数会不会涨"，而在于它涨的同时，每张卡分到的计算量却在缩小（`tp` 张卡平分同一份算力）**：`tp` 每翻倍，单卡算力需求减半，但单卡通信量的系数只从"逼近 2.00 的路上"往前挪了一点点、从没有下降过。**通信/计算比因此随 `tp` 增大而一路恶化**——这正是"TP 基本只在单机内（高速互联）做"这条结论的数字来源，`## 5` 会把它正式写成一条决策。

还有一个容易漏掉的乘数：**一个完整的 Transformer 层要做两次 all-reduce**，注意力块一次（`## 2.2` 的 `wo` 行切之后）、MLP 块一次（`## 2.1` 的 `w2` 行切之后）——`_lab/parallel.py:265` 的 `_demo()` 开头那行注释写得很直白："每层两次 all-reduce：注意力一次 + MLP 一次"。`## 1` 表里的数字是**每次**的通信量，一层实际要付两倍。

### 2.4 PP 的气泡：一个能徒手数格子证明的公式

流水并行把 `n_stage` 张卡各分到模型的一段连续层，请求被切成 `n_micro` 个 micro-batch 陆续送进流水线。第 `s` 级处理第 `m` 个 micro-batch 的绝对时刻是 `t = s + m`——`_lab/parallel.py:136-137` 就是这两行：

```python
# _lab/parallel.py:135-137
for m in range(n_micro):
    for s in range(n_stage):
        tl[s][s + m] = m
```

跑一个具体例子看时间线长什么样——4 级流水、4 个 micro-batch（`P=4, M=4`）：

```
stage0: [ 0  1  2  3  .  .  . ]
stage1: [ .  0  1  2  3  .  . ]
stage2: [ .  .  0  1  2  3  . ]
stage3: [ .  .  .  0  1  2  3 ]
```

（`.` 表示这一步这一级空闲，数字是正在处理第几个 micro-batch）每行长度是 `total = n_micro + n_stage - 1 = 7`（`_lab/parallel.py:133`）——这个数字本身也能手推：`stage0` 最早在 `t=0` 开工，`stage3`（最后一级）处理完最后一个 micro-batch（`m=3`）的时刻是 `t = 3 + 3 = 6`，算上 `t=0` 这一步，一共 `0..6` 共 7 步。

**关键的手算事实**：每一行恰好有 `n_stage - 1 = 3` 个空格——`stage0` 这一行填了 `m=0..3` 四个数字之后，`t=4,5,6` 三步必须空闲（因为它没有更多 micro-batch 要处理，得等整条流水排空）；`stage3` 这一行前 `t=0,1,2` 三步空闲（因为它要等前面三级把第一个 micro-batch 的激活传过来）。**这个"每行空 3 格"的数字，与 `s` 是第几级完全无关**——`_lab/parallel.py:138-139` 的实现直接验证了这一点：

```python
# _lab/parallel.py:138-139
idle = sum(1 for row in tl for c in row if c is None)
return tl, idle / (n_stage * total)
```

代入：总空格数 `idle = n_stage * (n_stage - 1)`（每行 `n_stage-1` 个，共 `n_stage` 行），总格子数 `n_stage * total = n_stage * (n_micro + n_stage - 1)`。两者相除，`n_stage` 直接约掉：

```
气泡 = idle / 总格子数
     = n_stage * (n_stage - 1) / [n_stage * (n_micro + n_stage - 1)]
     = (n_stage - 1) / (n_micro + n_stage - 1)
```

这正是 `_lab/parallel.py:129` 文档字符串写的公式，`selftest()` 对 5 组 `(p, m)` 逐一核对模拟结果与公式吻合（`_lab/parallel.py:230-237`）。**这条推导最值得记住的地方**：分子 `n_stage*(n_stage-1)` 是一个**跟 `n_micro` 完全无关的常数**——不管流水线上跑多少个 micro-batch，"灌满流水线"和"排空流水线"这两头总共要浪费掉这么多格子；而分母 `n_stage*(n_micro+n_stage-1)` 随 `n_micro` 线性增长。**固定的浪费，除以增长的总量，气泡比例自然被摊薄**——这就是"`M` 必须远大于 `P`"这条结论在数字上的来源，不是经验法则，是这个公式的直接推论。

### 2.5 EP：唯一重要的数是 max/mean

MoE 把 token 路由给 `n_expert` 个专家中的 `top_k` 个，`_lab/parallel.py:144-157` 的 `moe_route()` 模拟这个过程，`skew` 参数控制路由有多不均匀（`skew=0` 均匀，越大越偏向编号靠前的专家）。

**为什么 `max/mean` 是唯一重要的数**：一步计算里，`n_expert` 张卡各自处理分到自己的那批 token，这一步必须等**所有**专家都算完才能继续（下一层要用到这一层的完整输出）。如果负载均匀，每张卡耗时都约等于"平均负载对应的耗时"；如果负载不均，这一步的墙钟耗时由**负载最重**的那张卡决定，其它卡提前算完只能空等。所以这一步实际耗时相对于"理想均匀"情况的倍数，正好就是 `max_load / mean_load`——`_lab/parallel.py:160-168` 的 `imbalance()` 只有一行：

```python
# _lab/parallel.py:168
return float(load.max() / load.mean())
```

`## 1` 表里的数字是这套逻辑的直接体现：`skew=0.8` 时最重的专家拿到 2272 个 token、最轻的只有 10 个，`max/mean=4.44`，意味着**这一步四分之三以上的算力在空转等待**。`selftest()` 用两组种子相同、偏斜不同的路由直接对比（`_lab/parallel.py:246-252`）：均匀路由 `imbalance=1.02`，偏斜路由（`skew=0.6`）`imbalance=3.64`——相同的 4096 个 token、相同的 8 个专家，只是路由概率不均匀，一步的有效利用率就从接近满载掉到不足三成。

**`top_k` 增大不省通信量，只省单 token 算力**。`top_k=2` 时每个 token 要被送到 2 个专家，总的 `(token, expert)` 分发对数直接翻倍：`selftest()` 验证了这一点——`top_k=1` 时 4096 个 token 对应总负载 4096，`top_k=2` 时总负载变成 8192（`_lab/parallel.py:254-256`）。MoE 相对稠密模型的省算力体现在"每个 token 只激活一小部分专家参数"，但**路由到专家这件事本身要传数据**（EP 场景下是跨卡的 all-to-all），`top_k` 越大，要传的 `(token, expert)` 对越多，通信量只会更多不会更少——这条容易被直觉误解成"MoE 省，所以处处省"，`## 6` 会把这个错误单独列一条。

## 3. 自己动手：`_lab/` 里对应的实现

`_lab/parallel.py` 一共五节（模块文档字符串 `_lab/parallel.py:1-22` 已经把整体思路写明白），本节按 TP → PP → EP 的顺序逐段核对代码，配合能直接复制运行的脚本。

### 3.1 TP 参照实现与主角：`mlp_single` / `mlp_tensor_parallel`

`mlp_single()`（`_lab/parallel.py:39-41`）是不切的参照实现，两行：`gelu(x @ w1) @ w2`。`## 2.1` 已经逐行讲过 `mlp_tensor_parallel()`（`_lab/parallel.py:44-69`）的切法，这里补一个能亲手验证等价性的脚本：

```bash
cd _lab
python -c "
import numpy as np
import parallel as P

rng = np.random.default_rng(7)
x = rng.normal(0, 1, (16, 64)).astype(np.float32)
w1 = rng.normal(0, 0.05, (64, 256)).astype(np.float32)
w2 = rng.normal(0, 0.05, (256, 64)).astype(np.float32)

ref = P.mlp_single(x, w1, w2)
for tp in (1, 2, 4, 8, 16):
    got, comm = P.mlp_tensor_parallel(x, w1, w2, tp)
    diff = np.abs(ref - got).max()
    print(f'tp={tp:3d}  max_diff={diff:.3e}  all_reduce_elems={comm}')
"
```

实测输出（本机 CPU，只说明这套切法数学上的等价性，不代表任何真实吞吐）：

```
tp=  1  max_diff=0.000e+00  all_reduce_elems=0
tp=  2  max_diff=3.576e-07  all_reduce_elems=1024
tp=  4  max_diff=2.980e-07  all_reduce_elems=3072
tp=  8  max_diff=2.682e-07  all_reduce_elems=7168
tp= 16  max_diff=2.384e-07  all_reduce_elems=15360
```

`max_diff` 始终停在 `1e-7` 量级——这是 fp32 累加顺序不同带来的浮点舍入误差（`sum(parts)` 按 `tp` 份分别求和再合并，跟一次性 `gelu(x@w1)@w2` 的浮点运算顺序不同），不是切法本身引入的误差，`selftest()` 用 `atol=1e-4` 判定"一致"（`_lab/parallel.py:191`）也是同样的道理。**`all_reduce_elems` 是 `out.size * (tp - 1)`**（`_lab/parallel.py:69`，`out.size = 16*64 = 1024`）——这个数字**随 `tp` 线性增长**（`tp=16` 时是 `tp=2` 的 15 倍），因为它衡量的是"这次 all-reduce 一共要合并多少个待求和的部分和元素"（`tp` 份部分和两两相加，等价于要传输 `(tp-1)` 份完整输出）。这跟 `## 2.3`/`## 3.2` 讲的"每卡通信量"是**不同的量**：`allreduce_elements()` 算的是"用 ring 算法实现这次 all-reduce，每张卡自己收发多少元素"，那个数字的系数 `2(tp-1)/tp` 增长很慢（逼近 2.00 就不再涨）；这里的 `out.size*(tp-1)` 是"待归约的部分和总量"，会随 `tp` 线性涨上去。教学上刻意分成两个函数，避免把"要同步的数据总量"和"用具体算法实现同步、每卡分摊到的通信开销"混在一起。

### 3.2 通信量系数：`allreduce_elements`

`## 2.3` 已经推过公式，这里用脚本复现 `## 1` 第一张表，顺带验证系数上界确实是 2.00 而非其它数：

```bash
cd _lab
python -c "
import parallel as P
tokens, d = 2048, 4096
base = P.allreduce_elements(tokens, d, 2)
for tp in (1, 2, 4, 8, 16, 32, 64):
    e = P.allreduce_elements(tokens, d, tp)
    print(f'tp={tp:3d}  elems={e:>12,}  ratio_to_tp2={e/base if base else 0:.4f}')
"
```

```
tp=  1  elems=           0  ratio_to_tp2=0.0000
tp=  2  elems=   8,388,608  ratio_to_tp2=1.0000
tp=  4  elems=  12,582,912  ratio_to_tp2=1.5000
tp=  8  elems=  14,680,064  ratio_to_tp2=1.7500
tp= 16  elems=  15,728,640  ratio_to_tp2=1.8750
tp= 32  elems=  16,252,928  ratio_to_tp2=1.9375
tp= 64  elems=  16,515,072  ratio_to_tp2=1.9688
```

`tp=64` 时系数已经逼近 1.97，越往后涨得越慢——`2(tp-1)/tp = 2 - 2/tp`，`tp` 每翻倍，跟上界 `2.00` 的差距就减半，这是个收敛但永远到不了 2.00 的序列。`selftest()` 只断言了"单调上升"这个定性结论（`_lab/parallel.py:206-209`），没有断言具体收敛到哪——**验证代码只该断言"哪些结论是稳定成立的"，不该把一个连续函数的每个取值都写死成断言**，这也是本库测试设计上的一个惯例。

### 3.3 注意力 TP：头数除不尽时的报错现场

`## 2.2` 讲过 `n_head % tp == 0` 这条硬约束，这里亲手触发一次看报错长什么样：

```bash
cd _lab
python -c "
import numpy as np
import parallel as P

rng = np.random.default_rng(1)
x = rng.normal(0, 1, (4, 32)).astype(np.float32)
ws = [rng.normal(0, 0.05, (32, 32)).astype(np.float32) for _ in range(4)]
P.attn_tensor_parallel(x, *ws, n_head=8, tp=3)
"
```

```
AssertionError: 头数必须能被 tp 整除 —— 这是硬约束，不是实现限制
```

这条断言字符串本身（`_lab/parallel.py:108`）就是在提醒读者：**这不是"当前实现懒得处理不整除的情况"，而是"头是不可再分的计算单元，切不整除在数学上就没有意义"**。`selftest()` 用 `try/except AssertionError` 的方式把这个报错纳入自检（`_lab/parallel.py:223-227`），确认它**必须**报错，而不是默默算出一个错误答案。

### 3.4 PP：亲手把气泡公式核对一遍

`## 2.4` 推过 `(P-1)/(M+P-1)` 的公式，这里用脚本核对模拟结果与公式在更大范围内逐一吻合，同时观察"气泡的绝对格子数是常数"这条结论：

```bash
cd _lab
python -c "
import parallel as P
for p in (2, 4, 8):
    for m in (p, p * 2, p * 8, p * 32):
        tl, bub = P.pipeline_schedule(p, m)
        idle = sum(1 for row in tl for c in row if c is None)
        want = (p - 1) / (m + p - 1)
        print(f'P={p:2d} M={m:4d}  idle_cells={idle:4d}  bubble={bub:.4f}  formula={want:.4f}  match={abs(bub-want)<1e-9}')
"
```

```
P= 2 M=   2  idle_cells=   2  bubble=0.3333  formula=0.3333  match=True
P= 2 M=   4  idle_cells=   2  bubble=0.2000  formula=0.2000  match=True
P= 2 M=  16  idle_cells=   2  bubble=0.0588  formula=0.0588  match=True
P= 2 M=  64  idle_cells=   2  bubble=0.0154  formula=0.0154  match=True
P= 4 M=   4  idle_cells=  12  bubble=0.4286  formula=0.4286  match=True
P= 4 M=   8  idle_cells=  12  bubble=0.2727  formula=0.2727  match=True
P= 4 M=  32  idle_cells=  12  bubble=0.0857  formula=0.0857  match=True
P= 4 M= 128  idle_cells=  12  bubble=0.0229  formula=0.0229  match=True
P= 8 M=   8  idle_cells=  56  bubble=0.4667  formula=0.4667  match=True
P= 8 M=  16  idle_cells=  56  bubble=0.3043  formula=0.3043  match=True
P= 8 M=  64  idle_cells=  56  bubble=0.0986  formula=0.0986  match=True
P= 8 M= 256  idle_cells=  56  bubble=0.0266  formula=0.0266  match=True
```

**`idle_cells` 那一列是这段脚本最值得盯着看的地方**：固定 `P`、改变 `M`，`idle_cells` 纹丝不动（`P=8` 恒为 56 `= 8×7`），只有 `bubble`（`idle_cells` 除以总格子数）在往下掉。`## 2.4` 推过的道理在这里被数字直接坐实：**浪费的绝对格子数只取决于级数 `P`，跟 micro-batch 数 `M` 一点关系都没有**——`M` 大只是把这个固定的浪费摊到了更多的有效工作上。`selftest()` 里对应的断言是 `_lab/parallel.py:230-237`（5 组 `(p,m)` 与公式核对）和 `_lab/parallel.py:239-243`（`P=8` 时 `M=8` 与 `M=64` 的气泡比对，`b_few > b_many*3`）。

### 3.5 EP：路由不丢 token、`max/mean` 是唯一要看的数

亲手跑一遍 `## 2.5` 提到的具体数字：

```bash
cd _lab
python -c "
import parallel as P
even = P.moe_route(4096, 8, top_k=1, seed=1, skew=0.0)
skewed = P.moe_route(4096, 8, top_k=1, seed=1, skew=0.6)
print('even   ', even.tolist(), 'sum=', int(even.sum()), 'imbalance=', round(P.imbalance(even), 4))
print('skewed ', skewed.tolist(), 'sum=', int(skewed.sum()), 'imbalance=', round(P.imbalance(skewed), 4))
"
```

```
even    [517, 512, 521, 511, 524, 483, 507, 521] sum= 4096 imbalance= 1.0234
skewed  [1866, 1029, 550, 296, 190, 94, 47, 24] sum= 4096 imbalance= 3.6445
```

两行 `sum` 都精确等于 4096——**路由只决定 token 去哪个专家，不决定要不要处理它**，这是 `moe_route()`（`_lab/parallel.py:154-156`）内层循环"每个 token 恰好选 `top_k` 个专家"的直接结果，也是 `selftest()`（`_lab/parallel.py:248-249`）显式检查的东西：**如果这条不成立，意味着有 token 在路由过程中被静默丢弃了，这是比"负载不均"严重得多的 bug**——`## 6` 会把这条单独列出来。`imbalance` 从 1.02 涨到 3.64，同一批 token、同一批专家，只是概率分布从均匀变成 `skew=0.6` 的偏斜，浪费倍数就涨了 3 倍多。

最后验证 `top_k` 对总负载的影响：

```bash
cd _lab
python -c "
import parallel as P
for k in (1, 2, 4):
    load = P.moe_route(2000, 8, top_k=k, seed=0)
    print(f'top_k={k}  total={int(load.sum())}  expect={2000*k}')
"
```

```
top_k=1  total=2000  expect=2000
top_k=2  total=4000  expect=4000
top_k=4  total=8000  expect=8000
```

`total` 精确等于 `tokens * top_k`——`## 2.5` 讲过的"总负载线性正比于 `top_k`"在这里是一条能直接读出来的等式，`_lab/tests/test_advanced.py:129-131` 的 `test_moe_topk_does_not_lose_tokens` 把这条参数化到 `k in (1,2,4)` 三个值上测。

### 3.6 三个函数各自的测试落点

`_lab/tests/test_advanced.py` 的"二、并行：输出恒等"一节（`_lab/tests/test_advanced.py:91-131`）把上面六段手动脚本全部收进了 pytest：`test_tensor_parallel_mlp_is_exact`（`:91-99`，对 `tp in (1,2,4,8)` 参数化）对应 `## 3.1`；`test_gelu_is_not_additive`（`:101-105`）对应 `## 2.1` 的非线性证明；`test_head_count_must_divide_tp`（`:108-114`）对应 `## 3.3`；`test_allreduce_volume_grows_with_tp`（`:117-120`）对应 `## 3.2`；`test_pipeline_bubble_matches_formula`（`:123-126`，参数化 5 组 `(p,m)`）对应 `## 3.4`；`test_moe_topk_does_not_lose_tokens`（`:129-131`）对应 `## 3.5` 最后一段。**这一节测试与 `## 1` 的 `selftest()` 断言几乎一一对应**——教学库的惯例是同一条数学事实在 `--selftest` 和 `pytest` 里各留一份独立证据，`_PLAN.md` §2 把这条写成了"regul"：`test_lab.py`/`test_advanced.py` 这类恒等测试的判据永远是"算两遍，逐位比"。
