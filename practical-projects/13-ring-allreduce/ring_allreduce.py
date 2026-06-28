# -*- coding: utf-8 -*-
"""
从零实现 Ring All-Reduce（数据并行的核心通信原语）
=================================================

这是什么
--------
数据并行训练里，N 张 GPU 各自算出一份梯度，必须把它们「求和（再平均）」后
让每张卡都拿到同一份全局梯度，才能同步更新参数。这个「全员求和、人人受益」
的操作就是 **All-Reduce**。

本文件用纯 numpy 在单进程里**模拟** N 个 worker，从零实现 NVIDIA NCCL / 百度
最早推广的 **Ring All-Reduce** 算法，它把 All-Reduce 拆成两个阶段：

  1. Reduce-Scatter：N-1 步后，每个 worker 各自持有「最终结果的 1/N 分片」。
  2. All-Gather    ：再 N-1 步把这些分片在环上传一圈，人人集齐完整结果。

核心卖点是**带宽最优**：每个 worker 收发的数据量为 `2(N-1)/N · 数据量`，
当 N 很大时趋于常数 2 倍数据量，**与 worker 数 N 无关**——这正是它能扩展到
成千上万张卡的原因。作为对比，naive（朴素）All-Reduce 的单点通信量会随 N
线性增长，成为带宽瓶颈。

本脚本做三件事：
  * 实现 ring_all_reduce 并验证其结果 == 直接求和/平均（数值误差 ~1e-15）。
  * 打印通信量分析，量化 Ring vs Naive 的差距（per-worker 字节数）。
  * 跑一个 toy 的数据并行 SGD：N 个 worker 各看一部分数据，每步用
    ring all-reduce 同步梯度，观察 loss 下降——证明这套原语真能驱动训练。

对应文档：../../ai-infra/网络/集合通信原语.md  （§8 Ring-AllReduce，§8.3 通信量推导）
运行：python ring_allreduce.py   （CPU 数秒跑完，仅依赖 numpy）

注意：print 全部用 ASCII（Windows 控制台 GBK 下打印中文会 UnicodeEncodeError），
中文只出现在注释与 docstring 里。
"""

import numpy as np

# 固定随机种子，保证结果可复现
SEED = 42


# ---------------------------------------------------------------------------
# 1. 朴素 / 参考实现：直接把所有 worker 的张量加起来
#    这是「正确答案」，用来校验 ring 版本是否算对。
# ---------------------------------------------------------------------------
def naive_all_reduce_sum(tensors):
    """把 N 个 worker 的张量逐元素求和，返回 N 份相同的结果。

    语义上的 All-Reduce(SUM)：每个 worker 最终都拿到「全员之和」。
    这里直接 np.sum 一把梭，作为正确性基准（reference）。
    """
    total = np.sum(np.stack(tensors, axis=0), axis=0)  # 全局求和
    return [total.copy() for _ in range(len(tensors))]  # 人人一份


# ---------------------------------------------------------------------------
# 2. Ring All-Reduce 主体实现
#    我们把每个 worker 的一维张量切成 N 个 chunk（分片）。
#    环形拓扑：worker r 的右邻是 (r+1)%N，左邻（数据来源）是 (r-1)%N。
# ---------------------------------------------------------------------------
def ring_all_reduce(tensors, op="sum"):
    """对 N 个等长一维张量做 Ring All-Reduce。

    参数
    ----
    tensors : list[np.ndarray]，长度为 N，每个形状相同的一维数组。
              tensors[r] 代表 worker r 本地持有的数据（如本地梯度）。
    op      : "sum"（求和）或 "mean"（求平均，DDP 默认）。

    返回
    ----
    list[np.ndarray]：长度 N，每个都是相同的全局规约结果。

    算法分两阶段，各 N-1 步，每步每个 worker 只和右邻通信一个 chunk。
    """
    N = len(tensors)
    assert N >= 1, "need at least one worker"
    if N == 1:
        out = tensors[0].copy()
        return [out / 1 if op == "sum" else out]

    length = tensors[0].size
    assert length % N == 0, (
        "for this toy demo the tensor length must be divisible by N; "
        f"got length={length}, N={N}"
    )

    # 关键设计：把每个 worker 的向量切成 N 个 chunk。
    # buffers[r] 是一个长度为 N 的列表，buffers[r][c] = worker r 的第 c 个分片。
    # 整个算法只在这些分片上做「加法 + 沿环传递」。
    chunk = length // N
    buffers = [
        [t[c * chunk:(c + 1) * chunk].copy() for c in range(N)]
        for t in tensors
    ]

    # ---- 阶段一：Reduce-Scatter（N-1 步）----------------------------------
    # 直觉：让第 c 个分片的「全局和」最终汇聚到某一个 worker 手里。
    # 第 step 步：worker r 把自己「当前负责累加」的那个 chunk 发给右邻，
    #            右邻收到后累加到它对应位置上。经过 N-1 步，
    #            worker r 手里的 chunk[(r+1)%N] 恰好集齐了所有 N 份之和。
    # 这一步对应文档 §8：通信被均摊到每条环上的边，无单点热点。
    for step in range(N - 1):
        # 必须先把本轮所有「要发送的数据」快照下来，再统一加，
        # 否则同一步内边发边改会污染数据（模拟真实的并发收发）。
        send_snapshot = []
        for r in range(N):
            # worker r 本轮发送的 chunk 下标（精心设计的调度，使数据沿环累加）
            send_idx = (r - step) % N
            send_snapshot.append(buffers[r][send_idx].copy())

        for r in range(N):
            right = (r + 1) % N           # 右邻：接收者
            recv_idx = (r - step) % N     # 右邻要累加到的 chunk 下标
            # 右邻把「从 r 收到的分片」累加进自己对应的 chunk —— 这就是 reduce
            buffers[right][recv_idx] += send_snapshot[r]

    # 此刻：对每个 worker r，buffers[r][(r + 1) % N] 已是该分片的全局和。

    # ---- 阶段二：All-Gather（N-1 步）-------------------------------------
    # 直觉：把上面散落在各 worker 手里的「完整分片」沿环再传一圈，
    #      让每个 worker 集齐全部 N 个完整分片，拼出完整结果。
    # 这一步只搬运数据、不再做加法。对应文档 §8 的第二阶段。
    for step in range(N - 1):
        send_snapshot = []
        for r in range(N):
            # 本轮 worker r 把「已经完整」的那个 chunk 转发出去
            send_idx = (r + 1 - step) % N
            send_snapshot.append(buffers[r][send_idx].copy())

        for r in range(N):
            right = (r + 1) % N
            recv_idx = (r + 1 - step) % N
            # 直接覆盖（拷贝），因为这是已经规约完成的最终分片
            buffers[right][recv_idx] = send_snapshot[r]

    # ---- 拼回完整向量，并按需求平均 -------------------------------------
    results = []
    for r in range(N):
        full = np.concatenate(buffers[r])          # N 个分片拼成完整结果
        if op == "mean":
            full = full / N                        # DDP 默认对梯度取平均
        results.append(full)
    return results


# ---------------------------------------------------------------------------
# 3. 通信量分析：量化 Ring 的带宽最优性 vs Naive
# ---------------------------------------------------------------------------
def communication_cost_report(N, num_elements, bytes_per_elem=4):
    """返回 per-worker 收发字节数：ring 与 naive 两种方案。

    数学（文档 §8.3）：
      * Ring：两阶段各 N-1 步，每步收发 1 个 chunk = data/N。
              单向发送量 = 2*(N-1)*(data/N) = 2(N-1)/N * data。
              当 N -> 无穷，系数 -> 2，与 N 无关（带宽最优）。
      * Naive（reduce-to-0 然后 broadcast，简单星型）：
              发送侧热点 worker 要收 N-1 份、再发 N-1 份，随 N 线性增长。
    """
    data_bytes = num_elements * bytes_per_elem            # 一整份张量的字节数
    ring_per_worker = 2.0 * (N - 1) / N * data_bytes      # Ring 单 worker 收/发量
    naive_per_worker = (N - 1) * data_bytes               # Naive 中心节点的负担
    return {
        "data_bytes": data_bytes,
        "ring_send_bytes": ring_per_worker,
        "ring_coefficient": 2.0 * (N - 1) / N,            # 关键系数 2(N-1)/N
        "naive_hotspot_bytes": naive_per_worker,
        "speedup_vs_naive": naive_per_worker / ring_per_worker,
    }


# ---------------------------------------------------------------------------
# 4. Toy 数据并行训练：用 ring all-reduce 同步梯度，看 loss 是否下降
#    任务：N 个 worker 各持一部分样本，协同做线性回归 (y = X w + b)。
# ---------------------------------------------------------------------------
def toy_data_parallel_training(N=4, dim=8, total_samples=512, steps=60, lr=0.1):
    """模拟 N 个 worker 的数据并行 SGD，梯度用 ring_all_reduce 同步。

    返回训练过程的全局 loss 列表，以及 (学到的 w, 真值 w)。
    若每步都正确同步，效果应等价于「单机用全部数据」训练。
    """
    rng = np.random.default_rng(SEED)

    # 构造一个有真值的线性回归数据集（含噪声）
    true_w = rng.standard_normal(dim)
    X = rng.standard_normal((total_samples, dim))
    y = X @ true_w + 0.01 * rng.standard_normal(total_samples)

    # 把数据**切分**给 N 个 worker（数据并行的本质：各看一份不重叠的数据）
    shards_X = np.array_split(X, N)
    shards_y = np.array_split(y, N)

    # 所有 worker 从同一初始参数出发（这点很重要，否则模型会发散）
    w = np.zeros(dim)

    losses = []
    for step in range(steps):
        # 每个 worker 在自己的数据分片上算本地梯度
        local_grads = []
        for r in range(N):
            Xr, yr = shards_X[r], shards_y[r]
            pred = Xr @ w
            err = pred - yr
            # MSE 对 w 的梯度：2/m * X^T (Xw - y)
            grad = (2.0 / Xr.shape[0]) * (Xr.T @ err)
            local_grads.append(grad)

        # >>> 核心：用 ring all-reduce 把 N 份本地梯度同步成「全局平均梯度」<<<
        # 这一步就是真实 DDP 里每个 backward 之后发生的事。
        synced_grads = ring_all_reduce(local_grads, op="mean")
        global_grad = synced_grads[0]   # 同步后人人相同，取任意一个即可

        # 每个 worker 用同一份全局梯度更新 —— 参数自然保持一致
        w = w - lr * global_grad

        # 记录全局 loss（用全量数据评估，便于观察收敛）
        full_pred = X @ w
        loss = float(np.mean((full_pred - y) ** 2))
        losses.append(loss)

    return losses, w, true_w


# ---------------------------------------------------------------------------
# 5. Demo / 自检
# ---------------------------------------------------------------------------
def _demo_correctness():
    """验证 ring all-reduce 的结果与朴素求和/平均逐元素一致。"""
    rng = np.random.default_rng(SEED)
    N = 5
    length = 20                       # 必须能被 N 整除
    tensors = [rng.standard_normal(length) for _ in range(N)]

    ref_sum = naive_all_reduce_sum(tensors)[0]
    ring_sum = ring_all_reduce([t.copy() for t in tensors], op="sum")
    ring_mean = ring_all_reduce([t.copy() for t in tensors], op="mean")

    # 所有 worker 是否拿到一致结果？
    all_equal = all(np.allclose(ring_sum[0], r) for r in ring_sum)
    max_err_sum = max(np.max(np.abs(r - ref_sum)) for r in ring_sum)
    max_err_mean = max(np.max(np.abs(r - ref_sum / N)) for r in ring_mean)

    print("[1] Correctness check (Ring vs naive sum/mean)")
    print(f"    N(workers)={N}, tensor_length={length}")
    print(f"    all workers identical : {all_equal}")
    print(f"    max abs err  (sum)    : {max_err_sum:.3e}")
    print(f"    max abs err  (mean)   : {max_err_mean:.3e}")
    ok = all_equal and max_err_sum < 1e-9 and max_err_mean < 1e-9
    print(f"    result == reference   : {ok}")
    print()
    return ok


def _demo_communication():
    """打印 Ring vs Naive 的 per-worker 通信量，体现带宽最优性。"""
    num_elements = 1_000_000          # 假设 1M 参数的梯度
    print("[2] Communication volume per worker (lower is better)")
    print("    data = 1,000,000 floats (4 bytes each) = ~3.81 MB")
    print(f"    {'N':>4} | {'ring_coeff 2(N-1)/N':>20} | "
          f"{'ring_MB':>9} | {'naive_MB':>9} | {'ring<naive':>10}")
    print("    " + "-" * 66)
    for N in (2, 4, 8, 16, 64, 256):
        rep = communication_cost_report(N, num_elements)
        print(f"    {N:>4} | {rep['ring_coefficient']:>20.4f} | "
              f"{rep['ring_send_bytes'] / 1e6:>9.3f} | "
              f"{rep['naive_hotspot_bytes'] / 1e6:>9.3f} | "
              f"{rep['speedup_vs_naive']:>9.1f}x")
    print("    Note: ring coefficient -> 2.0 as N grows (independent of N),")
    print("    while naive hotspot grows linearly with N. <- bandwidth optimal")
    print()


def _demo_training():
    """跑 toy 数据并行训练，证明 ring all-reduce 能驱动 loss 下降。"""
    print("[3] Toy data-parallel SGD with ring all-reduce gradient sync")
    losses, w, true_w = toy_data_parallel_training()
    print(f"    loss[  0] = {losses[0]:.6f}")
    print(f"    loss[ 10] = {losses[10]:.6f}")
    print(f"    loss[ 30] = {losses[30]:.6f}")
    print(f"    loss[ -1] = {losses[-1]:.6f}")
    drop = losses[0] / losses[-1]
    w_err = float(np.max(np.abs(w - true_w)))
    print(f"    loss reduction factor : {drop:.1f}x  (loss went down)")
    print(f"    max|w_learned - w_true|: {w_err:.4f}  (recovered true weights)")
    converged = losses[-1] < losses[0] and w_err < 0.05
    print(f"    training converged    : {converged}")
    print()
    return losses, converged


def _maybe_plot(losses):
    """画 loss 曲线存 png；matplotlib 不可用则纯文本打印，绝不崩。"""
    try:
        import matplotlib
        matplotlib.use("Agg")          # 无显示环境，用文件后端
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(losses, marker=".")
        plt.yscale("log")
        plt.xlabel("step")
        plt.ylabel("global MSE loss (log scale)")
        plt.title("Data-Parallel Training via Ring All-Reduce")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = "training_loss.png"
        plt.savefig(out, dpi=110)
        plt.close()
        print(f"[4] Loss curve saved to {out}")
    except Exception as exc:           # matplotlib 缺失或后端问题都走这里
        print(f"[4] matplotlib unavailable ({type(exc).__name__}); "
              "ASCII loss sparkline instead:")
        lo, hi = min(losses), max(losses)
        ticks = " .:-=+*#%@"
        line = "".join(
            ticks[min(len(ticks) - 1,
                      int((np.log(l + 1e-12) - np.log(lo + 1e-12)) /
                          (np.log(hi + 1e-12) - np.log(lo + 1e-12) + 1e-12)
                          * (len(ticks) - 1)))]
            for l in losses
        )
        print("    high->low: " + line)
    print()


def main():
    print("=" * 70)
    print(" Ring All-Reduce from scratch (numpy)  |  data-parallel primitive")
    print("=" * 70)
    print()
    ok_correct = _demo_correctness()
    _demo_communication()
    losses, ok_train = _demo_training()
    _maybe_plot(losses)

    print("=" * 70)
    success = ok_correct and ok_train
    print(f" OVERALL: {'SUCCESS' if success else 'FAILURE'} "
          "(correctness passed AND training converged)" if success
          else " OVERALL: FAILURE")
    print("=" * 70)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
