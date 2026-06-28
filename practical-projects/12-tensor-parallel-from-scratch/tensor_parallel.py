"""
从零理解张量并行(Tensor Parallelism, Megatron 列/行切分)
====================================================================

这是什么
--------
本文件用纯 PyTorch(CPU)在**单进程**里"模拟"Megatron-LM 风格的张量并行(TP),
不需要多卡、不需要 NCCL、不需要外部数据集或网络。目的是把张量并行最核心的两个
工程直觉用可运行、可验证的代码讲清楚:

  1) 一个 Transformer MLP 块 `Y = GeLU(X @ A) @ B` 如何被切到 N 个"虚拟设备"上:
       - 第一个线性层 A 做 **列并行 (column parallel)**:按输出维切成 N 份,
         每个分片独立算出 `GeLU(X @ A_i)`,**无需通信**(因为 GeLU 是逐元素的,
         可以在切分后的列上独立施加)。
       - 第二个线性层 B 做 **行并行 (row parallel)**:按输入维切成 N 份,
         每个分片算出 `Y_i = GeLU(X @ A_i) @ B_i`,最后把所有分片的部分和
         **all-reduce(求和)** 合并成完整输出 Y。

  2) 为什么 Megatron 每个 Transformer 层在前向里有 **2 次 all-reduce**:
       - 一次来自 Attention 块(同样是 列并行 QKV/输出投影 + 行并行,最后 all-reduce);
       - 一次来自 MLP 块(本文件演示的就是这一个)。
     反向传播里对称地各再来一次,所以一个标准 Transformer 层前向+反向共 4 次。

我们做什么验证
--------------
把"切分并行 + 模拟 all-reduce"的前向结果,与"不切分的单卡前向"用
`torch.allclose` 逐元素对比,证明数学上**完全等价**(误差 ~ 浮点级别)。
同时打印:分片数、前向触发的通信(all-reduce)次数、与单卡基线的最大误差,
以及一个把多块 MLP 串起来的小"网络"的端到端等价性检查。

怎么跑
------
    python tensor_parallel.py

依赖:Python 3.x + PyTorch(CPU 即可)。可选 matplotlib(用于画误差随分片数变化图,
没有也不会崩,会退化为文本打印)。

与真实工程的差异见 README.md。
"""

import math

import torch
import torch.nn.functional as F


# 设随机种子,保证每次运行结果可复现(教学/可比对很重要)
SEED = 1234
torch.manual_seed(SEED)


# --------------------------------------------------------------------------- #
# 0. 一个"通信计数器" —— 模拟分布式里的集合通信原语 all-reduce                 #
# --------------------------------------------------------------------------- #
class CommCounter:
    """统计模拟的集合通信次数。

    在真实 Megatron 里,all-reduce 是跨 GPU 的 NCCL 集合通信(把各 rank 的张量
    求和后广播回每个 rank)。这里我们在单进程内用一次张量相加来"模拟"它,
    并对调用计数,以便量化"每个块发生了几次通信"。
    """

    def __init__(self):
        self.all_reduce_calls = 0

    def all_reduce_sum(self, shards):
        """模拟 all-reduce(sum):输入是各"虚拟 rank"的张量列表,输出其逐元素之和。

        语义对应真实分布式中的 dist.all_reduce(op=SUM):
        通信后每个 rank 都持有相同的"全局求和"结果。这里直接返回那个求和结果。
        """
        self.all_reduce_calls += 1
        out = shards[0].clone()
        for t in shards[1:]:
            out = out + t
        return out


# --------------------------------------------------------------------------- #
# 1. 基线:一个不切分的标准 MLP 块(单卡前向),作为"正确答案"                  #
# --------------------------------------------------------------------------- #
class BaselineMLP:
    """标准 Transformer MLP:Y = GeLU(X @ A) @ B。

    形状约定:
      X: [batch, d_model]
      A: [d_model, d_ff]      (升维,通常 d_ff = 4 * d_model)
      B: [d_ff,   d_model]    (降维回 d_model)
      Y: [batch, d_model]
    """

    def __init__(self, d_model, d_ff):
        # 用固定种子生成权重,后面张量并行版本会"切"这同一份权重,从而可比对
        self.A = torch.randn(d_model, d_ff) * (1.0 / math.sqrt(d_model))
        self.B = torch.randn(d_ff, d_model) * (1.0 / math.sqrt(d_ff))

    def forward(self, X):
        H = F.gelu(X @ self.A)   # 第一线性层 + 逐元素非线性
        Y = H @ self.B           # 第二线性层
        return Y


# --------------------------------------------------------------------------- #
# 2. 张量并行版 MLP:列并行(A) + 行并行(B) + all-reduce                       #
# --------------------------------------------------------------------------- #
class TensorParallelMLP:
    """把 BaselineMLP 的权重切到 N 个"虚拟设备"上并行计算。

    关键原理:
      * 列并行 A:A 按"输出维 d_ff"切成 N 列块 A_1..A_N。
        因为 GeLU 是逐元素的,GeLU(X @ A) 的第 i 段列 == GeLU(X @ A_i),
        所以每个 rank 能**独立**算出自己那段隐藏激活 H_i = GeLU(X @ A_i),
        前向到此**不需要通信**(这是列并行被选作第一层的根本原因)。
      * 行并行 B:B 按"输入维 d_ff"切成 N 行块 B_1..B_N,正好对齐 H_i 的列。
        每个 rank 算部分输出 Y_i = H_i @ B_i,它们是同形状的**部分和**;
        把所有 Y_i 求和(all-reduce)即得完整 Y = sum_i (H_i @ B_i)。
        => 列并行接行并行的组合,整个 MLP 前向**只需 1 次 all-reduce**。
    """

    def __init__(self, baseline_mlp, num_shards, comm):
        self.num_shards = num_shards
        self.comm = comm
        d_ff = baseline_mlp.A.shape[1]
        assert d_ff % num_shards == 0, "d_ff 必须能被分片数整除(教学简化)"
        chunk = d_ff // num_shards

        # 列并行:沿 dim=1(输出维 d_ff)切 A
        self.A_shards = [
            baseline_mlp.A[:, i * chunk:(i + 1) * chunk] for i in range(num_shards)
        ]
        # 行并行:沿 dim=0(输入维 d_ff)切 B —— 切点与 A 完全对齐
        self.B_shards = [
            baseline_mlp.B[i * chunk:(i + 1) * chunk, :] for i in range(num_shards)
        ]

    def forward(self, X):
        partial_outputs = []
        for A_i, B_i in zip(self.A_shards, self.B_shards):
            # --- 列并行段:本 rank 独立算自己的隐藏激活,无通信 ---
            H_i = F.gelu(X @ A_i)          # [batch, d_ff/N]
            # --- 行并行段:本 rank 算部分输出(部分和的一项) ---
            Y_i = H_i @ B_i               # [batch, d_model]
            partial_outputs.append(Y_i)

        # --- 唯一一次通信:把各 rank 的部分和 all-reduce(求和)成完整输出 ---
        # 这就是 Megatron 中 MLP 块前向的那"1 次 all-reduce"。
        Y = self.comm.all_reduce_sum(partial_outputs)
        return Y


# --------------------------------------------------------------------------- #
# 3. 一个把多块 MLP 串起来的小"网络",演示 N 层 => N 次 all-reduce            #
# --------------------------------------------------------------------------- #
def run_stacked_demo(num_layers, d_model, d_ff, num_shards, batch, comm):
    """堆叠 num_layers 个 MLP 块,验证端到端等价 + 统计总通信次数。"""
    torch.manual_seed(SEED + 7)
    X = torch.randn(batch, d_model)

    # 基线:逐层单卡前向
    baselines = [BaselineMLP(d_model, d_ff) for _ in range(num_layers)]
    Y_base = X
    for mlp in baselines:
        Y_base = mlp.forward(Y_base)

    # 张量并行:复用同一份权重切片,逐层前向
    comm.all_reduce_calls = 0
    tp_layers = [TensorParallelMLP(b, num_shards, comm) for b in baselines]
    Y_tp = X
    for tp in tp_layers:
        Y_tp = tp.forward(Y_tp)

    max_err = (Y_base - Y_tp).abs().max().item()
    ok = torch.allclose(Y_base, Y_tp, atol=1e-5, rtol=1e-4)
    return ok, max_err, comm.all_reduce_calls


# --------------------------------------------------------------------------- #
# 4. 误差随分片数变化的小图(可选,matplotlib 缺失则文本打印)                  #
# --------------------------------------------------------------------------- #
def plot_error_vs_shards(shard_list, err_list, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无界面后端,直接存文件
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(shard_list, err_list, marker="o")
        plt.yscale("log")
        plt.xlabel("number of shards (virtual devices)")
        plt.ylabel("max abs error vs single-device (log)")
        plt.title("Tensor Parallel MLP: TP output matches single-device")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(out_path, dpi=110)
        plt.close()
        return out_path
    except Exception as e:  # 缺 matplotlib 或其它问题:不崩,文本兜底
        print(f"[plot] skipped (matplotlib unavailable: {e}); text summary:")
        for s, err in zip(shard_list, err_list):
            print(f"[plot]   shards={s:2d}  max_err={err:.3e}")
        return None


# --------------------------------------------------------------------------- #
# 5. Demo 主程序                                                               #
# --------------------------------------------------------------------------- #
def main():
    # toy 规模:CPU 秒级跑完
    d_model = 64
    d_ff = 256          # = 4 * d_model,Megatron 典型比例
    batch = 32

    print("=" * 70)
    print("Tensor Parallelism from scratch (Megatron column/row split), CPU sim")
    print("=" * 70)
    print(f"config: d_model={d_model}, d_ff={d_ff}, batch={batch}, seed={SEED}")
    print()

    comm = CommCounter()

    # ---- (A) 单块 MLP:不同分片数都应与单卡基线一致,且各只触发 1 次 all-reduce ----
    print("[1] Single MLP block: column-parallel(A) + row-parallel(B)")
    baseline = BaselineMLP(d_model, d_ff)
    torch.manual_seed(SEED + 1)
    X = torch.randn(batch, d_model)
    Y_ref = baseline.forward(X)  # 正确答案

    shard_list, err_list = [], []
    for num_shards in [1, 2, 4, 8]:
        comm.all_reduce_calls = 0
        tp = TensorParallelMLP(baseline, num_shards, comm)
        Y_tp = tp.forward(X)
        max_err = (Y_ref - Y_tp).abs().max().item()
        match = torch.allclose(Y_ref, Y_tp, atol=1e-5, rtol=1e-4)
        shard_list.append(num_shards)
        err_list.append(max_err)
        status = "MATCH" if match else "MISMATCH"
        print(
            f"    shards={num_shards:2d} | all_reduce_calls={comm.all_reduce_calls} "
            f"| max_err={max_err:.3e} | vs single-device: {status}"
        )
    print("    -> one column-parallel + row-parallel MLP needs exactly 1 all-reduce")
    print()

    # ---- (B) 堆叠 N 层:总 all-reduce 次数应等于层数(MLP 块视角) ----
    print("[2] Stacked MLP blocks: #all_reduce should equal #layers (MLP view)")
    for num_layers in [1, 2, 6]:
        ok, max_err, calls = run_stacked_demo(
            num_layers, d_model, d_ff, num_shards=4, batch=batch, comm=comm
        )
        status = "EQUIVALENT" if ok else "DIFFERENT"
        print(
            f"    layers={num_layers} | total_all_reduce={calls} "
            f"| max_err={max_err:.3e} | end-to-end: {status}"
        )
    print("    -> a real Transformer layer = MLP(1) + Attention(1) = 2 all-reduce/fwd")
    print("       (backward mirrors it, so 4 all-reduce per layer per step)")
    print()

    # ---- (C) 误差图(可选) ----
    out_png = "tp_error_vs_shards.png"
    saved = plot_error_vs_shards(shard_list, err_list, out_png)
    if saved:
        print(f"[3] Saved error-vs-shards figure to: {saved}")
    print()

    # ---- 最终量化结论(可量化的"成功"信号) ----
    overall_max_err = max(err_list)
    all_match = overall_max_err < 1e-5
    print("=" * 70)
    print("RESULT")
    print(f"  shards tested        : {shard_list}")
    print(f"  worst max_abs_error  : {overall_max_err:.3e}  (target < 1e-5)")
    print(f"  tensor-parallel == single-device : {all_match}")
    print(f"  takeaway: column-then-row split => 1 all-reduce per MLP block;")
    print(f"            results are numerically identical to the unsharded forward.")
    print("=" * 70)


if __name__ == "__main__":
    main()
