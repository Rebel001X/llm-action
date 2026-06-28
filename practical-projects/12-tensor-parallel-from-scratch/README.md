# 从零理解张量并行(Megatron 列/行切分)

在**单 CPU、单进程**里"模拟"出 Megatron-LM 风格的**张量并行(Tensor Parallelism, TP)**:
把一个 Transformer 的 MLP 块切到 N 个"虚拟设备"上各自独立计算,再用(模拟的)
**all-reduce** 合并,并用 `torch.allclose` 证明结果与"不切分的单卡前向"**逐元素等价**。

无需多卡 / NCCL / 外部数据集 / 联网,几十秒内跑完,适合用来建立张量并行的核心直觉。

---

## 1. 演示什么原理

一个 Transformer 的 MLP 块是 `Y = GeLU(X @ A) @ B`。Megatron 的做法是把这两个线性层
用**互补的切法**组合起来,使整个块前向**只需 1 次通信**:

| 线性层 | 切法 | 切的维度 | 是否需要通信 | 原理 |
|--------|------|----------|--------------|------|
| 第一层 `A` | **列并行 (column parallel)** | 输出维 `d_ff` 切成 N 列块 | 否 | GeLU 是**逐元素**的,`GeLU(X@A)` 的第 i 段列 == `GeLU(X@A_i)`,每个 rank 能独立算出自己那段隐藏激活 `H_i`,不需要通信 |
| 第二层 `B` | **行并行 (row parallel)** | 输入维 `d_ff` 切成 N 行块(切点与 A 对齐) | 是(1 次) | 每个 rank 算部分输出 `Y_i = H_i @ B_i`,它们是同形状的**部分和**;`all-reduce(求和)` 得到完整 `Y = Σ_i H_i @ B_i` |

**关键结论:列并行接行并行 => 一个 MLP 块前向只触发 1 次 all-reduce。**

为什么常说"Megatron 每个 Transformer 层前向有 2 次 all-reduce"?因为一层里有两个块:
- **Attention 块**:列并行 QKV/输出投影 + 行并行,最后 1 次 all-reduce;
- **MLP 块**:即本项目演示的,1 次 all-reduce。

合计前向 2 次;反向传播对称地各再来一次,所以**一个标准 Transformer 层一个 step 共 4 次 all-reduce**。

代码里用一个 `CommCounter` 把"模拟 all-reduce"的调用计数,从而把"通信次数"这件事**量化打印**出来。

---

## 2. 怎么跑

```bash
cd practical-projects/12-tensor-parallel-from-scratch
python tensor_parallel.py
```

依赖:Python 3.x + PyTorch(CPU 即可)。`matplotlib` 可选(用于画"误差 vs 分片数"图,
缺失也不会报错,会退化为文本打印)。

---

## 3. 预期输出(真实运行摘录)

```
[1] Single MLP block: column-parallel(A) + row-parallel(B)
    shards= 1 | all_reduce_calls=1 | max_err=0.000e+00 | vs single-device: MATCH
    shards= 2 | all_reduce_calls=1 | max_err=7.153e-07 | vs single-device: MATCH
    shards= 4 | all_reduce_calls=1 | max_err=9.537e-07 | vs single-device: MATCH
    shards= 8 | all_reduce_calls=1 | max_err=9.537e-07 | vs single-device: MATCH
    -> one column-parallel + row-parallel MLP needs exactly 1 all-reduce

[2] Stacked MLP blocks: #all_reduce should equal #layers (MLP view)
    layers=1 | total_all_reduce=1 | max_err=1.192e-06 | end-to-end: EQUIVALENT
    layers=2 | total_all_reduce=2 | max_err=9.537e-07 | end-to-end: EQUIVALENT
    layers=6 | total_all_reduce=6 | max_err=1.788e-07 | end-to-end: EQUIVALENT

RESULT
  shards tested        : [1, 2, 4, 8]
  worst max_abs_error  : 9.537e-07  (target < 1e-5)
  tensor-parallel == single-device : True
```

可量化的"成功"信号:
- 任意分片数(1/2/4/8)下,张量并行输出与单卡基线的最大绝对误差 **~1e-6**(纯浮点累加顺序差异,远小于 `1e-5` 阈值),`torch.allclose` 全部 `MATCH`;
- 单个 MLP 块**恒为 1 次** all-reduce;堆叠 N 层时总 all-reduce 次数**恰好等于 N**,印证"每块 1 次通信";
- 同时生成 `tp_error_vs_shards.png`,直观展示误差始终停在浮点精度量级。

---

## 4. 对应 llm-action 文档

- 张量并行所属的训练框架:[`../../ai-framework/megatron-lm`](../../ai-framework/megatron-lm)
- 训练框架总览:[`../../ai-framework`](../../ai-framework)
- 大模型分布式训练相关:[`../../llm-train`](../../llm-train)

---

## 5. 社区参考

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (arXiv:1909.08053)](https://arxiv.org/abs/1909.08053) —— 张量并行(列/行切分 + all-reduce)的原始论文。

---

## 6. 局限 / 与真实工程的差异

本项目是**教学模拟**,刻意省略了真实分布式系统的大量工程细节:

1. **单进程模拟,不是真并行**:用一个 `for` 循环依次算各分片、用一次张量加法"扮演"
   all-reduce。真实 Megatron 是多进程/多 GPU,每个 rank 独占一块卡,all-reduce 走 **NCCL** 集合通信,延迟/带宽是真实瓶颈。本项目**不会**带来任何加速,只验证**数值等价性**与**通信次数**。
2. **只演示了 MLP 块**:没有实现 Attention 块的列/行并行(QKV 投影、多头切分、输出投影)。
   真实一层是 MLP + Attention,前向各 1 次共 2 次 all-reduce。
3. **只有前向**:没有实现反向传播里对偶的通信(前向 all-reduce 的算子在反向里对应另一次通信),
   因此没有体现"每层每 step 4 次 all-reduce"中反向的那 2 次。
4. **没有 embedding / 词表并行、没有序列并行(sequence parallelism)、没有与数据并行/流水线并行的 3D 组合**。
5. **toy 规模 + 无混合精度/无 fused kernel**:真实训练里张量并行还要叠加 fp16/bf16、激活重计算、
   通信-计算重叠等优化,数值表现与本项目的浮点误差量级不可直接类比。

一句话:本项目帮你**看懂"为什么这样切、为什么是这个通信次数、为什么数学上等价"**,
真正的性能与扩展性请以 Megatron-LM / DeepSpeed 等框架的多卡实现为准。
