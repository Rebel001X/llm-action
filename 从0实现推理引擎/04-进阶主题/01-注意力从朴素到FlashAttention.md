# 注意力从朴素到FlashAttention

> **一句话**：FlashAttention 一个 FLOP 都没省，它省的是访存。
> **前置**：[[02-自回归解码为什么是访存瓶颈]]（decode 是访存受限，这条结论是本篇存在的全部理由）；
> [[04-数值精度-为什么softmax要减最大值]]（在线 softmax 就是那篇"减最大值"技巧的增量版）

## 0. 这一篇要解决什么问题

FlashAttention 这个名字自带一种误导性——"Flash"（闪电）容易让人以为它是一种更快的**算法**，
或者是一种为了速度牺牲了一点精度的**近似**。这两个印象都不对，本篇要做的第一件事就是纠正它们：

> **FlashAttention 一个 FLOP 都没省。它重排了同样的乘加运算，额外还多付了一点点缩放的开销，
> FLOPs 不降反有一点点升。它省的只有一件事：访存——那张 T×T 的注意力分数矩阵不再需要
> 整个存在显存里。**

`[[02-自回归解码为什么是访存瓶颈]]` 已经证明：decode 阶段的算术强度恒卡在一个很低的常数上，
无论模型多大、上下文多长，都无法通过"算得更快"去解决，因为硬件根本不缺算力，缺的是数据搬运的带宽。
在这个前提下，"省 FLOPs" 本来就不是一个值得追求的目标——**省访存才是**。这就是为什么这篇文章要先把
"FlashAttention 是不是更快的算法"这个问题，替换成一个更准确的问题："它省的到底是什么，省了多少，
拿什么换的"。

回答这个问题要讲清楚五件事，也是本篇 `## 2` 的顺序：

1. 朴素的注意力计算里，**唯一**逼着你把整张 T×T 矩阵存下来的硬依赖是什么？
2. "在线 softmax"具体怎么拆掉这条依赖——它维护了哪些状态，那条更新公式为什么恒不会溢出？
3. 把这套技巧从"求最大值和求和"推广到"求注意力输出"，需要多加哪一件事（提示：不止 `l`，还有 `acc`）？
4. 分块之后，输出还是不是原来那个输出？是**精确**相等，还是**近似**相等？
5. decode 阶段（Tq=1）为什么不能直接套用"按 query 分块"这套思路，需要另开一条路径？

⚠️ **红线先说清楚**：本机没有 GPU。本篇能证明的只有两件事——**输出等价**（分块前后逐行 `np.allclose` 一致）
和**峰值中间量**（那张分数矩阵在内存里最多同时存多少个数）。**"更快"这件事本身，发生在 GPU 的
SRAM/HBM 两级存储层级上，本机的 numpy 实现证明不了，也不打算证明**——`_lab/flashattn.py` 里用
numpy 做分块只会比不分块更慢（多了 Python 层的循环开销），这是本库诚实标准明确要求的边界
（见 `_lab/flashattn.py:26-28`）。哪块 GPU 具体快多少倍，请去读 FlashAttention 原论文的实测数字，
本篇不代替它给出任何硬件性能数字。

## 1. 先看现象（可跑的代码/可算的数）

进入 `_lab/` 目录，先跑 selftest 确认代码本身是对的，再跑不带参数的版本看数字：

```bash
cd _lab
python flashattn.py --selftest    # 10 项断言
python flashattn.py               # 峰值中间量表 + 不同块大小的输出差异表
```

`--selftest` 的输出（本篇写作时实跑一遍得到，10 项全部 `[ok]`）：

```
  [ok]   在线 softmax 的 (最大值, 和) 等于一次性算的
  [ok]   输入含 900 量级仍不溢出（exp(m_old-m_new) 恒 <= 1）
  [ok]   6 种块大小下输出全部与朴素一致 —— 是重写不是近似
  [ok]   带因果 mask 时也一致
  [ok]   被完全 mask 的块不产生 nan（整块跳过要显式处理）
  [ok]   峰值中间元素：朴素 4096（= T^2），分块 256（= 块面积，与 T 无关）
  [ok]   T 翻倍：朴素峰值 x4，分块峰值不变
  [ok]   FLOPs 完全没省：同样的 QK^T 与 PV，只是分块做 + O(T) 次缩放
  [ok]   decode 沿 KV 切成 1/2/3/8/64 段，合并结果全部一致（合并可结合）
  [ok]   不同块大小结果在 1e-5 内一致；但逐位相同？False —— 求和顺序变了，
         最后几位就可能不同，这是「确定性」开关的由来
  全部通过
```

不带参数运行打印的第一张表——**峰值中间元素数**（T 是序列长度；"朴素"是那张 T×T 分数矩阵需要同时
存在的元素个数；"分块"固定用 32×32 的块）：

```
       T        朴素 (T^2)      分块 32x32        倍数
     128          16,384         1,024        16
     512         262,144         1,024       256
    2048       4,194,304         1,024     4,096
    8192      67,108,864         1,024    65,536
   32768   1,073,741,824         1,024 1,048,576
```

第二张表——同一组输入，不同块大小 `(bq, bk)` 算出的输出，相对朴素结果的最大绝对差：

```
     块 (bq,bk)           最大绝对差
     (256,256)       5.924e-08
       (64,64)       4.611e-07
       (16,16)       9.387e-07
         (4,4)       3.069e-07
         (1,1)       7.647e-07
```

**读这两张表，三条线索缺一不可：**

1. **朴素峰值随 T 平方增长，分块峰值恒定。** T 从 128 翻到 32768（256 倍），朴素峰值从 16,384 涨到
   10.7 亿（涨了 65,536 倍 = 256²），分块峰值始终是 1,024——**一个数字都没变**。T=32768 这一行，
   两者相差 1,048,576 倍，约 105 万倍。**"上下文开不长"，字面意义上就卡在这一格**：朴素做法要求
   在显存里同时摆下 10.7 亿个 float 才能算出一步注意力，硬件显存根本装不下这么大的中间结果，
   分块做法则从头到尾只需要摆下一块（这里选的 32×32=1,024 个）。
2. **块大小完全不影响峰值的量级判断，但会影响误差的具体数值。** 第二张表里五种块大小的误差都在
   1e-7~1e-8 这个量级，比 `np.allclose` 默认判定为一致的 1e-5 容差小两到三个数量级——**这是
   浮点求和顺序不同带来的舍入误差，不是算法本身的误差**，`## 2.5` 会讲这条区分为什么重要。
3. **selftest 第 10 条最值得多看一眼**：它断言 `np.allclose(a, b, atol=1e-5)` 为真（在容差内一致），
   同时打印 `np.array_equal(a, b)` 的结果是 `False`（逐位不完全相同），**这条断言本身就是在教一件事**：
   **"分块前后输出一致"这句话的正确验证方式是"在容差内相等"，不是"逐位相等"**——`## 6` 会把这个坑
   单独讲透，因为它是这个领域最常见、最容易被写错的一条测试。

## 2. 原理

### 2.1 唯一的硬依赖：softmax 的分母要等所有分数都算完才知道

一步注意力，对某一个 query 位置 `i`，输出是：

```
out_i = (1 / l_i) · Σ_j  exp(s_ij - m_i) · v_j
其中   m_i = max_j  s_ij        （这一行分数里的最大值，减掉它是为了数值稳定，见下面 04 篇）
       l_i = Σ_j  exp(s_ij - m_i)  （减去最大值之后的指数和，即 softmax 的分母）
```

`m_i` 和 `l_i` 都是**对 j 的全局归约**——要遍历这一行全部 `Tk` 个 key 位置才能确定下来。朴素实现
（`_lab/flashattn.py:37-56` 的 `attention_naive()`）索性一口气把它们全部算完：

```python
scores = q @ k.T / np.sqrt(d)          # _lab/flashattn.py:47，一次性算出整张 (Tq, Tk) 矩阵
...
m = scores.max(axis=-1, keepdims=True) # _lab/flashattn.py:53，此时才能拿到每行的最大值
p = np.exp(scores - m)
out = (p / p.sum(axis=-1, keepdims=True)) @ v   # _lab/flashattn.py:54-55
```

代价写在函数自己的 docstring 里（`_lab/flashattn.py:42-43`）：**"峰值中间量是 Tq×Tk —— 那张分数矩阵
必须整个存在，因为 softmax 的分母要等所有分数都算完才知道。"** 这是唯一的硬依赖，也是分块技术唯一
需要拆掉的东西。

这里有一个容易被忽略的备选方案，值得先排除掉：**两遍扫描**——第一遍只扫过所有 key 块，只求
`(m_i, l_i)` 这两个标量（不需要存整张矩阵，只需要 O(Tq) 的额外空间）；第二遍再扫一遍 key/value，
用第一遍算好的 `(m_i, l_i)` 直接算出 `out_i`。这个方案在**内存**上确实也能把峰值降到 O(块大小)，
但它要求把同一批 K、V **从头到尾读两遍**——在 decode 这种访存受限的场景里，"读两遍"就是把本该省下的
访存搬运成本又还回去了一半。**在线 softmax 真正的贡献不只是省内存，是把两遍扫描压缩成一遍**，
让每一块 K、V 只被读一次。`## 2.2` 就是这条压缩是怎么做到的。

### 2.2 在线 softmax：用一点点缩放换掉第二遍扫描

`_lab/flashattn.py:61-81` 的 `online_softmax_sum()` 是整个技巧的最小单元——先不管注意力，
只看"怎么用一遍扫描算出一串数字的 (最大值, 以该最大值为基准的指数和)"：

```python
m, l = -np.inf, 0.0
for s in range(0, len(x), block):
    blk = x[s:s + block]
    m_blk = float(blk.max())
    m_new = max(m, m_blk)
    if m_new == -np.inf:
        continue
    l = l * np.exp(m - m_new) + float(np.exp(blk - m_new).sum())   # _lab/flashattn.py:79
    m = m_new
return m, l
```

关键就是 `_lab/flashattn.py:79` 这一行。每见到一个新块：

1. 新的运行最大值 `m_new = max(m_old, m_blk)`——**只可能变大，不会变小**；
2. 但是 `l`（旧的指数和）是按**旧**的 `m_old` 算出来的，直接加新块的贡献会对不上基准，所以要先把它
   按 `exp(m_old - m_new)` 缩一下，再加上新块（以 `m_new` 为基准）的贡献。

**这一步为什么恒不会溢出**：因为 `m_new >= m_old`（`max` 的定义），所以 `m_old - m_new <= 0`，
`exp(非正数)` 恒 `<= 1`。缩放因子永远在把旧值往小里缩，不可能把一个有限数缩成 `inf`。
这条性质和 `[[04-数值精度-为什么softmax要减最大值]]` 里"减最大值防止 `exp` 溢出"是同一个原理的
增量版——那一篇讲的是"减一次、用一整批数据的最大值"，这里讲的是"每见到新数据就重新减一次，
且缩放系数恒 `<=1`，所以中途换基准点也不会引入新的溢出风险"。`_lab/flashattn.py:69` 的 docstring
把这句话原样写了下来："这就是整个 FlashAttention 的数学内核，剩下的都是工程。"

selftest 用两条断言验证了这个最小单元本身：第一条（`_lab/flashattn.py:180-186`）验证分块算出的
`(m, l)` 和一次性算出来的完全一致；第二条（`_lab/flashattn.py:188-192`）故意塞进 700/800/900 这种
在 fp32 下单独做 `exp` 就会溢出的数字，验证在线版本依然不出问题——因为它永远在减去**当前已知的
运行最大值**，永远不会真的对着一个大到溢出的数字直接做 `exp`。

### 2.3 推广到注意力：分子（acc）和分母（l）必须用同一个缩放因子一起缩

只求 `(m, l)` 还不够——注意力真正要的是加权和 `out = (1/l) · Σ exp(s-m) · v`，多了一个"分子"
`acc = Σ exp(s-m) · v`。`acc` 和 `l` 的更新逻辑必须完全对称：**`l` 换基准要缩放，`acc` 换基准
也要用同一个缩放因子缩放，缩早了或者缩晚了、两者缩的系数不一致，输出就是错的。**
这正是 `_lab/flashattn.py:84-128` 的 `attention_online()` 在做的事，逐行走读：

外层循环按 query 分块（`_lab/flashattn.py:100-101`），内层循环按 key/value 分块
（`_lab/flashattn.py:107-108`）。每个 query 块进入内层循环前，先初始化三个 running 状态
（`_lab/flashattn.py:103-105`）：

```python
m = np.full((qe - qs, 1), -np.inf, np.float32)      # running 最大值
l = np.zeros((qe - qs, 1), np.float32)              # running 分母
acc = np.zeros((qe - qs, v.shape[1]), np.float32)   # running 分子（这是比 2.2 多出来的一个状态）
```

内层循环每见到一个新的 KV 块，先算这一块的分数、（可选）套因果 mask、算这一块的局部最大值
`m_blk`、合并出新的运行最大值 `m_new`（`_lab/flashattn.py:109-115`）。接下来是防 nan 的一步——
**如果某一整块因为因果 mask 全被设成 `-inf`**（比如 query 位置早于这一整块 key 的位置），
这一块的 `m_blk` 也是 `-inf`，`m_new` 可能仍然是 `-inf`（如果之前所有块也都被 mask 掉了）：

```python
alive = np.isfinite(m_new)                          # _lab/flashattn.py:116
if not alive.any():
    continue
m_safe = np.where(alive, m_new, 0.0)                 # _lab/flashattn.py:119，避免 -inf 参与减法
```

然后是全篇真正的核心，**这一行是全部**（原文件注释原话，`_lab/flashattn.py:120`）：

```python
rescale = np.where(np.isfinite(m), np.exp(m - m_safe), 0.0)   # _lab/flashattn.py:121
p = np.where(np.isfinite(s), np.exp(s - m_safe), 0.0)          # _lab/flashattn.py:122
l = l * rescale + p.sum(axis=-1, keepdims=True)                 # _lab/flashattn.py:123
acc = acc * rescale + p @ v[ks:ke]                               # _lab/flashattn.py:124
```

`l` 和 `acc` **用的是同一个 `rescale`**——这就是 `## 2.3` 标题那句话的代码实体。内层循环结束后，
`out[qs:qe] = acc / np.where(l == 0, 1.0, l)`（`_lab/flashattn.py:127`）才做最后一次除法；
分母保护是防守"这个 query 块所有 key 都被 mask 掉"这种边界情况（正常因果 mask 下不会发生，
因为对角线永远可见，但作为防御性代码写在这里）。

### 2.4 峰值中间量：O(T²) 到 O(块面积)，且与 T 无关

`attention_naive()` 返回的峰值中间量是 `tq * tk`（`_lab/flashattn.py:56`）；`attention_online()`
返回的峰值中间量是 `bq * bk`（`_lab/flashattn.py:128`）——**与 `tq`、`tk` 完全无关**，因为内层循环
里的 `s`、`p` 这些中间张量的形状永远是 `(块大小, 块大小)`，序列多长都不会让这个形状变大，变大的
只是循环要跑多少轮。这正是 `## 1` 那张表格的代码来源，也是 selftest 第 6、7 条断言
（`_lab/flashattn.py:213-216`、`218-225`）直接编码的结论："T 翻倍，朴素峰值 ×4，分块峰值不变"。

### 2.5 是重写，不是近似——但"逐位相等"是错误的验证方式

`attention_online()` 的 docstring 说得很直接（`_lab/flashattn.py:89-91`）：**"输出必须与
`attention_naive` 一致（浮点误差内）——这是本文件的核心断言，也是 FlashAttention 能被安全采用的
唯一理由：它是重写，不是近似。"** selftest 第 3 条（`_lab/flashattn.py:194-203`）拿 6 种块大小
`(1,1) (1,16) (8,8) (16,7) (64,64) (64,128)` 挨个和朴素结果比对，全部在 `atol=1e-5` 内一致。

但第 10 条（`_lab/flashattn.py:242-247`）紧接着演示了一件事：**"在容差内一致"不等于"逐位相同"**。
`bq=bk=64` 和 `bq=bk=8` 两种块大小算出的结果，`np.allclose` 判定一致，但 `np.array_equal` 返回
`False`。原因是 IEEE 754 浮点加法不满足结合律——`(a+b)+c` 和 `a+(b+c)` 在实数域里相等，
在浮点域里可能差最后一两位有效数字。**块大小不同，本质上就是把同一组数按不同的顺序分组求和**，
最后几位不同是正常的舍入现象，不是 bug。这条区分对应着真实系统里"deterministic / 确定性模式"这个
开关的存在理由——`## 4`、`## 6` 会继续展开。

### 2.6 decode 为什么需要另一条路径：沿 KV 切，而不是沿 query 切

`attention_online()` 的外层循环是按 query 分块。**decode 阶段 `Tq` 恒为 1**——`_lab/flashattn.py:49`
的注释写着"decode 时 tq=1 而 tk 很大"。外层循环只有一次迭代可跑，等于没有并行度：不管你把
`bq` 设成多大，反正只有 1 行 query，切不出第二块。真正又长又值得切的维度是 KV——一次 decode 要看
的历史 token 数（`ctx_len`）可以是几千甚至几万。

`_lab/flashattn.py:133-161` 的 `attention_split_kv()` 就是给这种场景准备的：断言 `Tq` 必须是 1
（`_lab/flashattn.py:146`），把 KV 切成 `n_split` 段（`_lab/flashattn.py:149`），每段独立算出自己的
`(局部最大值, 局部指数和, 局部加权和)`（`_lab/flashattn.py:150-156`），最后合并：

```python
m_all = max(p[0] for p in partial)
l_all = sum(l * np.exp(m - m_all) for m, l, _ in partial)     # _lab/flashattn.py:159
o_all = sum(o * np.exp(m - m_all) for m, _, o in partial)     # _lab/flashattn.py:160
```

合并用的还是 `## 2.2`、`## 2.3` 那同一套"缩放到统一基准再相加"的数学——只不过这次是把已经算好的
几个**段级**结果合并，而不是流式地处理一个个块。因为这条合并运算本身满足结合律（`## 2.2` 已经证明
缩放因子恒 `<=1` 且更新公式是纯粹的线性组合），**段怎么切、切几段都不影响最终结果**——
selftest 第 9 条（`_lab/flashattn.py:231-240`）用 `n_split ∈ {1,2,3,8,64}` 五种切法逐一验证，
全部与朴素结果一致。真实系统里这种"沿 KV 切、多段独立算再合并"的技术叫 FlashDecoding 或
split-K（`_lab/flashattn.py:140`）。

## 3. 自己动手：`_lab/` 里对应的实现

把 `## 2` 走过的每一段推导，收拢成一份可以按图索骥的清单（本库自引用，不加引擎前缀）：

**参照实现（问题的起点）**
- `_lab/flashattn.py:37-56` `attention_naive()`：朴素注意力的完整实现，返回 `(输出, 峰值中间元素数)`。
- `_lab/flashattn.py:42-43` docstring 原话："峰值中间量是 Tq×Tk —— 因为 softmax 的分母要等
  所有分数都算完才知道"，这是 `## 2.1` 的结论出处。
- `_lab/flashattn.py:47` `scores = q @ k.T / np.sqrt(d)`：一次性算出整张分数矩阵的那一行代码。
- `_lab/flashattn.py:53-55`：全局取最大值、算 `exp`、归一化、乘 `V` 四步，对应 `## 2.1` 公式里的
  `m_i`、`l_i`、`out_i`。

**在线 softmax 的数学内核**
- `_lab/flashattn.py:61-81` `online_softmax_sum()`：最小单元，只维护 `(m, l)` 两个标量。
- `_lab/flashattn.py:67` docstring 里写死的更新公式 `l_new = l_old*exp(m_old-m_new) + sum(...)`。
- `_lab/flashattn.py:69`："`exp(m_old-m_new)` 恒 `<=1`，所以这一步永远在缩小，不会溢出"——
  `## 2.2` 不溢出证明的原文出处。
- `_lab/flashattn.py:79`：唯一真正实现这条更新公式的一行代码。

**推广到注意力：分块 + 在线 softmax**
- `_lab/flashattn.py:84-128` `attention_online()`：`## 2.3` 逐行走读的对象。
- `_lab/flashattn.py:89-91` docstring："它是重写，不是近似"——`## 2.5` 论点的原文出处。
- `_lab/flashattn.py:103-105`：`m`/`l`/`acc` 三个 running 状态的初始化，`acc` 是比
  `online_softmax_sum` 多出来的那个状态。
- `_lab/flashattn.py:116-119`：`alive` 掩码，专门防止整块被 mask 掉时出现 `-inf` 参与减法产生 nan。
- `_lab/flashattn.py:120-124`：**"这一行是全部"**——`rescale` 因子的计算，以及 `l`、`acc` 用
  同一个 `rescale` 同步缩放再累加，`## 2.3` 标题那句话的代码实体。
- `_lab/flashattn.py:127`：最终除法，对 `l == 0`（整行被 mask，边界情况）做了防御性保护。

**decode 专用：沿 KV 切**
- `_lab/flashattn.py:133-161` `attention_split_kv()`：`## 2.6` 逐行走读的对象。
- `_lab/flashattn.py:146`：`assert q.shape[0] == 1`，显式限定这条路径只服务 `Tq=1` 的 decode 场景。
- `_lab/flashattn.py:149`：`step = max(1, (tk + n_split - 1) // n_split)`，沿 KV 切分的步长计算。
- `_lab/flashattn.py:158-160`：段级结果的合并公式，与 `## 2.2`/`## 2.3` 同一套缩放数学。

**正确性护栏（`## 1` 说"先信 selftest 再信表"，护栏就在这）**
- `_lab/flashattn.py:194-203`：6 种块大小逐一核对与朴素结果一致（`atol=1e-5`），`## 2.5` 论点一。
- `_lab/flashattn.py:210-211`：因果 mask 下整块被 mask 掉的场景不产生 nan，`## 6` 会展开这个坑。
- `_lab/flashattn.py:213-216`：峰值中间量与 T 无关的断言，`## 2.4` 结论的可执行版本。
- `_lab/flashattn.py:218-225`：T 翻倍时朴素峰值 ×4、分块峰值不变的断言。
- `_lab/flashattn.py:227-229`：**"FLOPs 完全没省"**——这句话本身被写成了一条（恒真的）断言，
  提醒读者这不是一个需要用数字验证的命题，而是需要被记住的事实。
- `_lab/flashattn.py:234-240`：decode 的 split-KV 在 `n_split ∈ {1,2,3,8,64}` 下结果全部一致，
  `## 2.6` "合并可结合"的可执行版本。
- `_lab/flashattn.py:242-247`：`np.allclose` 为真但 `np.array_equal` 为 `False` 的那条断言，
  `## 2.5` 第二个论点、也是 `## 6` 要讲的踩坑的正面示范。

**pytest 回归（独立于 selftest 的第二道护栏）**
- `_lab/tests/test_advanced.py:55` `test_tiling_does_not_change_output`：参数化 5 种块大小
  `(1,1) (1,8) (7,5) (16,16) (64,64)`，逐一断言与朴素一致——和 selftest 第 3 条互相独立地验证同一件事。
- `_lab/tests/test_advanced.py:63` `test_peak_intermediate_is_independent_of_sequence_length`：
  显式断言朴素峰值 `== Tq*Tk`、分块峰值 `== 64`（`bq=bk=8`）。
- `_lab/tests/test_advanced.py:72` `test_online_softmax_survives_huge_inputs`：700/800/900 量级
  输入下 `(m, l)` 仍然有限。
- `_lab/tests/test_advanced.py:79` `test_split_kv_merge_is_associative`：`n_split ∈ {1,2,5,16,48}`
  下与朴素结果全部一致，参数集合和 selftest 里的不完全相同（selftest 用 1/2/3/8/64），
  两边独立选取参数、独立通过，是更强的证据。

## 4. 真实引擎是怎么做的（对照 vLLM/SGLang 等，带 `引擎:文件:行`）

`## 2` 推出的第一个结论就足够颠覆直觉：**真实推理引擎几乎都不自己"写"一份从零实现的注意力
kernel。** 它们要么把这件事外包给专门的 kernel 库（FlashAttention 官方库、FlashInfer），
要么另起一份用 Triton 写的版本——但即便是"自己写"的那份，数学内核也和 `## 2.2`/`## 2.3`
讲的在线 softmax 完全同构，只是从 numpy 换成了 Triton、从 Python for 循环换成了 GPU 上的
tile 级并行。**"不自己重新发明这套数学"这件事本身，就是本篇一个重要的结论。**

**证据一：注意力在 vLLM 里不是"一个" kernel，是一整套可插拔的后端。**
`vllm:vllm/v1/attention/backends/registry.py:34-131` 的 `AttentionBackendEnum` 用 37 个具名
成员列出了所有支持的后端，每个成员是一个可以被覆盖的类路径，包括（举例）：

```python
FLASH_ATTN   = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"     # registry.py:44
TRITON_ATTN  = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"   # registry.py:48
FLASHINFER   = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"         # registry.py:65
```

（`vllm:vllm/v1/attention/backends/registry.py:44`、`vllm:vllm/v1/attention/backends/registry.py:48`、
`vllm:vllm/v1/attention/backends/registry.py:65`）——同一份代码库里同时存在给 CUDA 用的、
给 ROCm 用的、给 MLA（多头潜在注意力）架构用的、给稀疏注意力用的十几种实现，运行时按硬件和
模型结构选一种。这本身就说明"attention kernel"从来不是一个单点，而是一层需要按场景切换的抽象。

**证据二：`FLASH_ATTN` 这个默认后端，压根没有自己实现在线 softmax。**
`vllm:vllm/v1/attention/backends/fa_utils.py:18-24`：

```python
if current_platform.is_cuda():
    from vllm._custom_ops import reshape_and_cache_flash
    from vllm.vllm_flash_attn import (
        compile_flash_attn_varlen_func_from_specs,
        flash_attn_varlen_func,
        get_scheduler_metadata,
    )
```

`flash_attn_varlen_func` 是从 `vllm.vllm_flash_attn` 这个独立打包的 kernel 库里**导入**的，
不是在这个文件、也不是在 vLLM 仓库的 Python 层里重新实现的。vLLM 这一层做的事是调度、拼 batch、
管理 KV 缓存的排布——**把"怎么在 SRAM 里分块、怎么排 tile"这件事完全交给了下游那个专门的库**。

**证据三：`TRITON_ATTN` 是货真价实自己写的，但写出来的数学和 `_lab/flashattn.py` 逐字对得上。**
`vllm:vllm/v1/attention/ops/triton_unified_attention.py:561-562`：

```python
M, L, P, alpha = softmax_step(S, M, L)
acc = acc * alpha[:, None]
```

`softmax_step()` 本体在 `vllm:vllm/v1/attention/ops/triton_attention_helpers.py:417-440`：

```python
m_j = tl.maximum(M, tl.max(S, axis=1))          # 运行最大值，等价于 _lab/flashattn.py:115 的 m_new
alpha = tl.exp(M - m_j)                          # 缩放因子，等价于 _lab/flashattn.py:121 的 rescale
L_new = L * alpha + l_j                          # 等价于 _lab/flashattn.py:79/123 的更新公式
return m_j, L_new, P, alpha
```

对照看：`_lab/flashattn.py:121` 的 `rescale = np.exp(m - m_safe)` 对应这里的 `alpha = tl.exp(M - m_j)`；
`_lab/flashattn.py:123` 的 `l = l * rescale + p.sum(...)` 对应这里的 `L_new = L * alpha + l_j`；
`_lab/flashattn.py:124` 的 `acc = acc * rescale + p @ v[...]` 对应调用处的 `acc = acc * alpha[:, None]`
（乘法之后紧接着的累加在调用处的下一行，此处未摘录）。**变量名从 `m/l/rescale` 换成了 `M/L/alpha`，
公式的形状一模一样**——只不过一个跑在 numpy 的 Python 解释器里，一个作为一个 `@triton.jit` 编译单元
跑在 GPU 上，`S`（这一块的分数）是一整个 tile 一次性被硬件并行算出来的，不是 Python 里的
标量循环。这正是"更快"真正发生的地方：**同一套 `## 2.2`/`## 2.3` 的数学，换到能做大规模并行的
硬件执行模型上**——具体怎么发生，`[[05-GPU执行模型速成]]` 那一篇的主题，本篇证明不了。

**证据四：SGLang 独立地收敛到了同一种架构。**
`sglang:python/sglang/srt/layers/attention/attention_registry.py:31` 定义了自己的
`ATTENTION_BACKENDS = {}` 注册表，用 `@register_attention_backend(...)` 装饰器登记了 22 个后端
（`sglang:python/sglang/srt/layers/attention/attention_registry.py:42` 登记 `"flashinfer"`，
`:177` 登记 `"triton"`，`:209` 登记 `"fa3"`，`:235` 登记 `"fa4"`，`:202` 登记 `"flashmla"`）。
它的 FlashAttention 系后端同样是"外包"模式——
`sglang:python/sglang/srt/layers/attention/flashattention_backend.py:53-56`：

```python
from sglang.kernels.ops.attention.flash_attention import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
```

外加 `merge_state_v2` 从 `sgl_kernel`（一个独立编译的 kernel 包）导入。**两个互相独立开发的引擎，
用了同一种"注册表 + 可插拔后端 + 至少一个后端外包给编译好的 kernel 库"的结构**——这不是巧合，
是因为"给一块具体的 GPU 架构写出能榨干 SRAM 带宽的 tile 调度"是一门和"写调度器、管 KV 缓存"
完全不同的专精手艺，值得由专门的团队/专门的库来做，引擎本身只负责在合适的时候调用它、
在不同硬件/不同注意力变体之间做正确的选择。

## 5. 设计决策与代价（为什么这样 / 不这样会怎样 / 什么时候可以不这样）

**决策一：用"单遍扫描 + 在线缩放"，而不是"两遍扫描"去求 softmax 的归一化项。**
- 为什么这样：单遍扫描时每一块 K、V 只需要从存储里读一次；两遍扫描（先扫一遍拿到全局
  `(m,l)`，再扫一遍算加权和）需要把同一批 K、V 完整读两次。`## 2.1` 已经说明这是唯一被
  在线 softmax 真正省下来的东西——`_lab/flashattn.py:79` 那一行"多付"的只是一次标量乘法
  （`exp(m_old-m_new)`），换掉的是整整一趟数据搬运。
- 不这样会怎样：如果老实做两遍扫描，内存峰值确实也能降到 O(块大小)（两遍扫描本身不需要存整张
  矩阵），但每一块 KV 要被读两次——在 `[[02-自回归解码为什么是访存瓶颈]]` 证明的"decode 天生
  访存受限"这个前提下，多读一遍数据就是多付一遍本该省下的成本，等于只拿到了"省内存"这一半收益，
  却没拿到"省访存"这另一半。
- 什么时候可以不这样：当 K、V 本来就小到能整批常驻在离计算单元最近的存储里（比如序列很短，
  `## 6` 会展开这一条），扫一遍还是两遍在访存量上几乎没有差别，这时候选哪种实现方式更多是工程
  简洁性的考量，不是性能考量。

**决策二：用"分块 + 精确重写"，而不是用近似（截断/稀疏/低秩）去省内存。**
- 为什么这样：`## 2.5` 证明了 6 种块大小的输出与朴素结果在 `atol=1e-5` 内完全一致——**这是一条
  可以验证的等价性，不是"差不多"**。一项技术要在生产系统里被无条件默认启用，前提是它不会悄悄
  改变模型该有的输出分布；如果 FlashAttention 是近似算法，每换一个新模型、新任务都要重新验证
  "这点近似误差会不会影响这次的下游质量"，这件事没有尽头。精确重写不需要这层担保。
- 不这样会怎样：滑动窗口截断、稀疏注意力、低秩近似这些技术确实能省更多内存/算力，但每一种都
  需要单独做"近似质量够不够"的实验，而且往往**不能反悔**——如果训练阶段就用了某种近似结构，
  推理时想换回精确注意力，模型的权重已经是在那种近似假设下学出来的，换不回去，会掉点。
- 什么时候可以不这样：当你确实需要更激进的内存/算力节省，并且愿意为此重新训练或重新验证模型时
  （典型场景是真正需要支持远超正常上下文长度的超长序列），近似路线是合理选择——但那已经是一个
  比"要不要用 FlashAttention"更大的决策，`_PLAN.md` §4 里 `04-进阶主题` 目录下其他还没写的篇目
  （比如量化、结构化输出）在讨论"有损优化"时会展开这类权衡，本篇讨论的重写是无损的。

**决策三：给 decode 单独设计"沿 KV 切"的路径，而不是复用"沿 query 切"的双层循环。**
- 为什么这样：`_lab/flashattn.py:137-139` 的 docstring 把原因写得很直接——decode 时 `Tq` 恒为 1，
  按 query 分块的外层循环连一次真正意义上的"分块"都做不到（只有一整块，且这一块只有一行），
  等于外层循环形同虚设，没有并行度可言；而一次 decode 要看的历史 KV 长度动辄几千上万，
  是唯一真正"够长、值得切"的维度。
- 不这样会怎样：如果硬套 `attention_online()` 那套"先切 query 再切 KV"的双层循环去处理 decode，
  数学上不会错（依然是同一套在线 softmax），但外层循环永远只有一个块——整个计算退化成对一个
  长度为 1 的 query 块，在内层循环里对 KV 做顺序分块扫描，**没有把"KV 很长"这个唯一可用的
  并行维度显式暴露出来给下面的执行单元利用**，等于放弃了 decode 阶段本来能拿到的并行度。
- 什么时候可以不这样：当 KV 本身也很短（比如一次对话刚开始，上下文只有几十个 token）时，
  切不切、沿哪个维度切都无所谓，`attention_online()` 退化成一次几乎不用分块的朴素计算就够用——
  split-KV 是专门为"上下文很长"这个场景准备的手段，`_lab/flashattn.py:234-240` 那条断言测的
  `n_split ∈ {1,2,3,8,64}` 全部一致，说明"切不切、切几段"只是一个性能/并行度的选择，
  完全不影响正确性，什么时候用它是可以按场景自由决定的。

## 6. 常见错误与踩坑

**错误一：以为 FlashAttention 是一种近似算法。**
"更快"这个名字加上"省内存"这个效果，很容易让人联想到"牺牲一点精度换速度"。但 `## 2.5` 已经
用可执行的断言证明了：6 种块大小的输出与朴素结果在 `atol=1e-5` 内一致（`_lab/flashattn.py:194-203`），
`attention_online()` 的 docstring 直接写着"它是重写，不是近似"（`_lab/flashattn.py:89-91`）。
它和朴素实现算的是**同一个数学对象**，只是换了一种计算它的顺序。

**错误二：以为它省的是 FLOPs。**
selftest 里专门有一条断言把这句话焊死成了字面意思（`_lab/flashattn.py:227-229`）：
**"FLOPs 完全没省：同样的 QK^T 与 PV，只是分块做 + O(T) 次缩放。"** 分块只是把同一批乘加运算
换了个顺序、换了个粒度去做，额外还要为每个新的 KV 块多算一次 `rescale`（`_lab/flashattn.py:121`）
和一次 `exp`——FLOPs 不但没降，理论上还有一点点上升。真正被省下来的只有访存，`## 0` 那句总纲
"FlashAttention 一个 FLOP 都没省"不是修辞，是字面事实。

**错误三：整块被 mask 掉时不处理，得到 nan。**
causal mask 下，右上角那些"query 位置早于这一整块 key 位置"的块，分数会整块变成 `-inf`。
如果直接对这样一块取 `max`，`m_blk` 也是 `-inf`；如果不做防护，紧接着的 `exp(m_old - m_new)`
在 `m_old` 和 `m_new` 都是 `-inf` 时会算出 `exp(-inf - (-inf)) = exp(nan) = nan`——而 nan
具有传染性，一旦 `l` 或 `acc` 被污染，后面所有依赖它们的计算全部变成 nan，且**不会报错**，
只会在某个位置突然输出乱码。`_lab/flashattn.py:116-119` 的 `alive` 掩码就是专门防这个的，
selftest 第 5 条（`_lab/flashattn.py:210-211`）显式断言 causal 场景下的输出全部有限
（`np.isfinite(got_c).all()`）。**这是本文档开头 `_PLAN.md` 反复强调的"推理引擎的 bug 绝大多数是
静默的"这条铁律在这里的具体体现**——不处理这个边界情况，代码照样跑，只是结果悄悄错了。

**错误四：拿分块结果和朴素结果做逐位（bitwise）相等断言。**
`## 2.5` 已经详细讲过：selftest 第 10 条（`_lab/flashattn.py:242-247`）明确演示
`np.array_equal(a, b)` 返回 `False`——不同块大小意味着浮点加法按不同顺序分组求和，
IEEE 754 浮点加法不满足结合律，最后几位有效数字（ULP）可能不同。**正确的验证方式永远是
`np.allclose`（带一个合理的 atol/rtol），不是逐位相等。** 如果在 CI 里写了逐位相等的断言，
它会因为块大小、并行调度顺序、甚至硬件不同而随机地、间歇性地失败——制造出一个看起来像 bug、
实际上完全正常的假警报。这也是 `## 4` 提到的真实引擎里"deterministic / 确定性模式"这个开关
存在的原因：它通常意味着"强制用固定的块划分和固定的归约顺序"，用一部分性能换取"同一个输入
每次都拿到逐位相同的输出"这个特殊保证——大多数场景不需要它，默认关闭以换取更好的性能。

**错误五：以为 `n_split` 切几段会影响 decode 的计算精度。**
`## 2.6` 已经说明合并公式满足结合律，`_lab/flashattn.py:234-240` 用 `n_split ∈ {1,2,3,8,64}`
五种切法逐一验证与朴素结果一致。段数只影响并行度和调度方式，不影响数值上"应该等于什么"——
唯一会出现的差异和 `## 2.5` 说的一样，是不同求和顺序带来的最后几位舍入误差，落在
`atol=1e-5` 的容差之内，与"切几段"这件事本身无关。

## 7. 自测题与延伸阅读

**自测题（闭卷，做完再对照 `## 2` 或重新跑 `_lab/flashattn.py --selftest` 核对）：**

1. 用 `## 2.1` 给出的 `out_i`、`m_i`、`l_i` 三条公式说明：为什么朴素实现必须先把这一行全部
   `Tk` 个分数都算完，才能得到 `out_i` 里的任何一个数？"两遍扫描"能不能绕开这条依赖？
   代价是什么？
2. `online_softmax_sum()` 里的缩放因子写成 `np.exp(m_old - m_new)`。如果有人手误写反，
   写成 `np.exp(m_new - m_old)`，会在什么情况下让程序出问题？（提示：`m_new` 和 `m_old`
   谁大谁小是确定的，反过来意味着指数恒 `>=0`。）
3. `attention_online()` 的内层循环每处理一个新的 KV 块，要更新哪三个 running 状态？
   为什么 `l` 和 `acc` 必须乘同一个 `rescale` 因子，不能各自单独处理、用不同的缩放系数？
4. T 从 512 变到 2048（4 倍），`_lab/flashattn.py` 里朴素实现的峰值中间量变成几倍？
   固定块大小的分块实现峰值中间量变成几倍？用 `## 1` 那张实测表核对你的答案。
5. decode 阶段为什么不能直接复用 `attention_online()` 里"先切 query 再切 KV"的双层循环去
   获得并行度？换成 `attention_split_kv()` 之后，"切几段"这件事会不会改变输出的数值？
6.（附加）selftest 第 10 条断言 `np.allclose(a, b, atol=1e-5)` 为真，同时打印
   `np.array_equal(a, b)` 为 `False`，测试仍然判定为通过。如果哪天有人把这条断言改成要求
   `np.array_equal` 必须为 `True`，会发生什么？这样改是不是让测试"更严格"了？

**延伸阅读**：FlashAttention 这个技术的原始出处是 Dao 等人的论文（标题里包含 "IO-Awareness"
这个关键词），具体版本、发表年份、期刊/会议本篇未查证，不在此编造引用；感兴趣可以自己搜索核对。
但请记住 `## 0` 就说清楚的红线：**本篇能证明的只是等价性和峰值中间量，具体"快多少倍"是原论文在
特定 GPU、特定序列长度、特定精度下的实测数字，不是可以脱离硬件条件直接复述的通用真理**——
`[[05-GPU执行模型速成]]` 讲的 SRAM/HBM 两级存储模型，是理解"这些数字为什么会是那个量级"所需要的
背景知识，本篇没有覆盖，也不打算靠编数字去覆盖。

**双链**：
- `[[02-自回归解码为什么是访存瓶颈]]`——本篇整个立论"省访存而非省算力才有意义"的前提，
  来自那一篇证明的"decode 恒处于访存瓶颈"。
- `[[04-数值精度-为什么softmax要减最大值]]`——在线 softmax 的缩放公式是那一篇"减最大值防溢出"
  这条数值稳定性技巧的增量版：每见到新数据就重新减一次当前已知的最大值。
- `[[05-GPU执行模型速成]]`——`## 4` 反复强调"更快"发生在 GPU 的 SRAM/HBM 层级，
  具体是怎么发生的、为什么把数据留在片上就能更快，属于那一篇的主题，本篇证明不了。
- `[[03-投机解码]]`——`## 2.6`/`## 5` 决策三的前提是"decode 恒有 `Tq=1`"；投机解码验证阶段
  一次要验证 `γ+1` 个候选 token，`Tq` 不再是 1，split-KV"专为 `Tq=1` 设计"这条边界条件在那一篇
  会被重新审视。
