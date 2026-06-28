# 05 · 从零实现 FlashAttention（分块 tiling + 在线 softmax）

> 用 PyTorch 从零写一个 **分块 + online softmax** 的注意力，证明 FlashAttention
> **不是近似、而是精确的内存重排**：输出与朴素注意力逐位相等，额外显存却从
> `O(n²)` 降到 `O(n)`。CPU 几秒跑通，自包含、零外部依赖（不需要数据集/网络/GPU）。

---

## 1. 演示什么原理（简明）

标准（朴素）注意力：

```
O = softmax(Q Kᵀ / √d) · V
```

它会先把分数矩阵 `S = Q Kᵀ`（**n×n**）整张物化到显存，再 softmax 成 `P`（又一张
**n×n**），再乘 `V`。瓶颈是这两张 `n×n` 矩阵反复读写 HBM —— 显存 `O(n²)`，且注意力
在 GPU 上是**访存受限（memory-bound）**的。

**FlashAttention** 把 `Q/K/V` 沿序列维切成小块（tile），逐块在「片上」计算，
用 **online softmax** 为每个 query 行维护三个小状态，边算边累加，**永不物化完整的
`n×n` 矩阵**：

| 状态 | 含义 | 大小 |
|---|---|---|
| `m` | running max（当前见过的最大分数） | 标量 / 行 |
| `l` | running 分母（exp 之和） | 标量 / 行 |
| `acc` | running 未归一化输出 | `d` 维 / 行 |

online softmax 的核心递推（来一个新 K/V 块就更新一次）：

```
m_new   = max(m_old, rowmax(S_blk))           # running max 只增不减
rescale = exp(m_old − m_new)   (≤ 1)          # 把旧累加器翻译到新 max 基准
l   = rescale·l   + rowsum(exp(S_blk − m_new))
acc = rescale·acc + exp(S_blk − m_new) · V_blk
... 所有块处理完 ...
O = acc / l                                    # 收尾只除一次
```

`rescale` 因子是灵魂：max 一旦变大，就把「按旧 max 算出的旧累加值」乘上一个 `≤1`
的因子缩到新基准，保证最终结果与一次性看全行完全一致，且全程数值稳定。

> 关键认知：FlashAttention **省的是显存占用 + HBM 访存次数，不省 FLOPs**。本 demo
> 在 CPU 上跑（CPU 无 SRAM/HBM 层级），所以**不复现 GPU 的加速比**，但能精确复现
> 「分块 + online softmax = 标准注意力」的**数值等价性**与 `O(n²)→O(n)` 的**显存账**。

---

## 2. 怎么跑

环境：Python 3.13 / PyTorch（CPU 即可）。无需 GPU、无需联网。

```bash
cd practical-projects/05-flash-attention-tiled
python flash_attention.py
```

代码里特意把块大小设成**除不尽序列长**（`n=128, block_k=24`），用来测试尾块/边界的
正确性；并跑了「单块退化为标准 softmax」的边角案例和 20 组随机输入的压力测试。

---

## 3. 预期输出（跑通后摘的真实输出）

```
[correctness] flash vs naive attention output
  max  abs error = 4.441e-16
  mean abs error = 4.594e-17
  torch.allclose(atol=1e-10) = True
  -> RESULT: PASS (numerically identical)
----------------------------------------------------------------
[edge case] single block (Br=Bc=n) degenerates to plain softmax
  max abs error = 6.661e-16  (PASS)
----------------------------------------------------------------
[memory] extra intermediate-state elements (NOT counting Q/K/V/O)
       n |   naive O(n^2) | flash (per Qblk) |      ratio
     128 |         32,768 |              928 |      35.3x
     512 |        524,288 |              928 |     565.0x
    2048 |      8,388,608 |              928 |    9039.4x
    8192 |    134,217,728 |              928 |  144631.2x
----------------------------------------------------------------
[stress] 20 random trials, worst-case max abs error = 1.554e-15
  -> ALL PASS
DONE. FlashAttention reproduces naive attention exactly; extra memory is O(n), not O(n^2).
```

要点：
- **逐位等价** —— 最大误差 `~4e-16`，正好是 float64 的机器精度量级（即「精确」，非近似）。
- **边角正确** —— 单块退化、尾块除不尽都 PASS。
- **显存账** —— 朴素中间矩阵随 `n²` 爆炸（`n=8192` 时 1.34 亿个元素），Flash 的每块状态
  恒为 928 个元素、**不随 `n` 增长**；比值随 `n` 线性放大（35× → 144631×）。

---

## 4. 对应 llm-action 文档

- **原理详解**：[`../../llm-optimizer/FlashAttention.md`](../../llm-optimizer/FlashAttention.md)
  —— 含 GPU 内存层级、online softmax 完整推导、v1/v2/v3 演进、反向传播、手算示例。
  本代码的两层循环（外 Q 块 / 内 K-V 块）与该文档第 5 节的伪代码 **1:1 对应**。
- 相关：[`../../llm-optimizer/kv-cache.md`](../../llm-optimizer/kv-cache.md)、
  [`../../llm-optimizer/`](../../llm-optimizer/)（注意力/推理优化合集）。

---

## 5. 社区参考

- [FlashAttention 论文 arXiv:2205.14135](https://arxiv.org/abs/2205.14135) ——
  *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*。
- [xlite-dev/Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference)
  —— LLM 推理优化论文/代码合集。

---

## 6. 局限 / 与真实工程的差异

- **不复现加速比**：真实 FlashAttention 的提速来自把分块计算 **fuse 进一个 CUDA/Triton
  kernel**，让小块停在片上 SRAM、减少 HBM 往返。CPU 上没有这套内存层级，本 demo 只验证
  **数值等价 + 显存复杂度**，不验证 wall-clock 加速。
- **用 float64 对拍**：为了把误差从「算法」里彻底剥离、清楚展示「精确」，对比用了
  float64。真实实现跑 fp16/bf16，会有正常的低精度舍入误差（仍与同精度朴素实现等价）。
- **是教学循环，不是高性能 kernel**：用 Python `for` 循环逐块遍历，纯演示控制流；
  生产实现是 Triton/CUDA 手写 kernel，并融合 mask/dropout/causal、做双缓冲与 warp 调度。
- **只做前向、无 causal mask、单头**：未实现反向传播（论文用「重算代替存储」省显存）、
  因果掩码、多头/批维。这些都是在本框架上的直接扩展。
- **块大小是 toy 值**：`block_q/block_k` 取小值便于看清分块逻辑；真实 kernel 里它们由
  SRAM 容量决定（典型 64/128）。
```

---

> 这是 `llm-action` 的「动手可跑」实战系列之一：每个项目都是 CPU 可跑通、自包含的最小实现，
> 用来把对应文档里的原理「跑出来」。
