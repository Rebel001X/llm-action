# 13 · 从零实现 Ring All-Reduce（数据并行通信原语）

> 用纯 numpy 在单进程里**模拟 N 个 worker**，从零实现 NVIDIA NCCL / 百度最早推广的
> **Ring All-Reduce** 算法（Reduce-Scatter + All-Gather 两阶段），并验证它与直接求和/平均
> 完全一致、量化它的**带宽最优性**、再用它驱动一个 toy 数据并行训练把 loss 真正降下来。
>
> CPU 数秒跑完，只依赖 numpy（matplotlib 可选）。

---

## ① 演示什么原理

数据并行训练中，N 张 GPU 各自算出一份梯度，必须「求和再平均」后让**每张卡都拿到同一份
全局梯度**才能同步更新——这个「全员求和、人人受益」的操作叫 **All-Reduce**。

**Ring All-Reduce** 把它拆成两阶段，每阶段各 `N-1` 步，每步每个 worker 只与右邻通信一个分片：

1. **Reduce-Scatter**（求和阶段）：把每份张量切成 N 个 chunk，沿环传递并累加。
   N-1 步后，每个 worker 各自持有「最终结果的某一个分片（已是全局和）」。
2. **All-Gather**（收集阶段）：把这些完整分片再沿环传一圈（只搬不加），
   人人集齐全部分片，拼出完整结果。

**核心卖点——带宽最优**：每个 worker 收发的数据量为

```
2(N-1)/N · 数据量
```

当 N 增大时系数趋于常数 **2**，**与 worker 数 N 无关**；而 naive（中心节点收发的星型）
All-Reduce 的单点通信量随 N **线性增长**，会成为带宽瓶颈。这就是 Ring 能扩展到成千上万
张卡的原因。通信量 `2(N-1)/N` 的推导见对应文档 §8.3。

---

## ② 怎么跑

```bash
cd practical-projects/13-ring-allreduce
python ring_allreduce.py
```

环境：Python 3.13 / numpy 2.3（torch 2.x CPU 可选，本脚本未强制依赖）。
画图为可选项：装了 matplotlib 会把 loss 曲线存成 `training_loss.png`，没装则退化为
ASCII 火花线，**不会崩**。

---

## ③ 预期输出（真实跑通摘录）

```
[1] Correctness check (Ring vs naive sum/mean)
    all workers identical : True
    max abs err  (sum)    : 4.441e-16
    max abs err  (mean)   : 1.110e-16
    result == reference   : True

[2] Communication volume per worker (lower is better)
       N |  ring_coeff 2(N-1)/N |   ring_MB |  naive_MB | ring<naive
       4 |               1.5000 |     6.000 |    12.000 |       2.0x
       8 |               1.7500 |     7.000 |    28.000 |       4.0x
      64 |               1.9688 |     7.875 |   252.000 |      32.0x
     256 |               1.9922 |     7.969 |  1020.000 |     128.0x
    Note: ring coefficient -> 2.0 as N grows (independent of N) ... bandwidth optimal

[3] Toy data-parallel SGD with ring all-reduce gradient sync
    loss[  0] = 5.070283
    loss[ -1] = 0.000099
    loss reduction factor : 51012.1x  (loss went down)
    max|w_learned - w_true|: 0.0007  (recovered true weights)
    training converged    : True

 OVERALL: SUCCESS (correctness passed AND training converged)
```

可量化的「成功信号」：
- **正确性**：ring 结果与直接求和/平均逐元素误差 `~1e-16`（浮点极限），且所有 worker 拿到一致结果。
- **带宽最优**：ring 系数随 N 收敛到 2；N=256 时 ring 单点通信量仅 ~8 MB，naive 高达 ~1020 MB（**128x** 差距）。
- **能驱动训练**：toy 数据并行 SGD 用 ring all-reduce 同步梯度，loss 下降 **5 万倍**，权重误差 `7e-4`。

---

## ④ 对应 llm-action 文档

- [集合通信原语](../../ai-infra/网络/集合通信原语.md) —— §8 Ring-AllReduce 流程图解、§8.3 通信量 `2(N-1)/N` 推导、§10 小数字手算数据流
- [NCCL](../../ai-infra/网络/NCCL.md) —— 工业级实现 NCCL 的环/树拓扑与性能
- [集合通讯性能测试](../../ai-infra/网络/nccl-test-集合通讯的性能测试.md) —— 真实 all-reduce 带宽实测
- 网络目录总览：[../../ai-infra/网络](../../ai-infra/网络)

---

## ⑤ 社区参考

- [NCCL / Ring-AllReduce 原理（NVIDIA Developer Blog: Massively Scale Deep Learning with NCCL 2.4）](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)
- [Baidu Ring-AllReduce（Bringing HPC Techniques to Deep Learning，原始推广文）](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)
- [NCCL 官方文档 — Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

---

## ⑥ 局限 / 与真实工程的差异

- **单进程模拟**：本代码在一个进程里用 numpy 数组模拟 N 个 worker，没有真实的进程/网络
  通信。真实场景由 NCCL/Gloo/MPI 通过 NVLink、InfiniBand/RoCE 等互联实现，瓶颈是真实链路带宽与延迟。
- **无重叠（overlap）**：真实 NCCL 把 chunk 收发**流水线化**并与反向计算**重叠**以隐藏延迟；
  本实现是同步串行的「快照→累加」，只为讲清数据流，不追求性能。
- **拓扑简化**：只实现了单环（single ring）。生产 NCCL 会用双环、Tree、以及 NVLS（NVLink SHARP）
  在不同规模/拓扑下自动选择最优算法；大消息用 ring，小消息常用 tree。
- **数值/类型**：用 float64，未涉及 fp16/bf16 混合精度下的累加误差、梯度压缩、量化通信等工程细节。
- **约束**：toy 实现要求张量长度能被 N 整除；真实实现会对不整除做 padding/分块处理。
- **梯度平均位置**：这里在 all-gather 后统一除以 N；NCCL 的 `ReduceOp.AVG` 或 DDP 也可在
  reduce-scatter 阶段就缩放，二者数学等价。

> 一句话：本项目用最少代码把 Ring All-Reduce 的**算法骨架与带宽最优性**讲透并跑通，
> 是理解 DDP/NCCL「梯度同步为什么不随卡数变慢」的最小可运行模型。
