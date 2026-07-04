# -*- coding: utf-8 -*-
"""
tiled_matmul.py — 分块矩阵乘(Tiled MatMul)与访存模型(Memory Model)
==================================================================

本模块是《Practical GPU Programming》第 5 章「共享内存分块」的配套实现。
它在纯 CPU / numpy 上,用两条腿把「分块为什么能减少访存」讲透:

  1) 计算腿:一个逐块(tile-by-tile)累加的 matmul,行为与 GPU 上
     "把 A、B 的小块搬进 shared memory 反复复用" 的 kernel 完全对应,
     并用 numpy 的 A @ B 校验数值正确性。

  2) 建模腿:一个解析(analytical)访存模型,不真正跑 kernel,而是
     用公式算出——在给定 tile 大小 T 下,朴素 kernel 与分块 kernel 分别
     需要从 HBM(高带宽显存)读多少字节、算术强度(arithmetic intensity)
     是多少、每个元素被复用了多少次。

核心结论(全部由本模块的函数定量给出):
  - 朴素 matmul:每个输出元素独立地把 A 的一行、B 的一列从 HBM 读进来,
    C 的每个元素触发 2N 次全局内存读 → 总读 ≈ 2*N^3 个元素。
  - 分块 matmul:一个 T×T 的输出块共享同一批 A、B 子块,子块进 shared
    memory 后被块内 T*T 个线程复用 → 总读 ≈ 2*N^3 / T 个元素。
  - 于是 tile 越大,HBM 访问越少(线性下降 1/T),算术强度越高
    (正比于 T),直到 tile 装不下 shared memory 为止 —— 这就是
    "屋顶线(roofline)" 里从「访存受限」走向「计算受限」的物理原因。

所有函数不依赖 GPU、不联网,CPU 上秒级可跑。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =====================================================================
# 第一部分:分块矩阵乘的「计算」实现(数值正确性由 numpy 校验)
# =====================================================================


def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """朴素三重循环矩阵乘,作为「行为可读」的基线参考实现。

    C[i, j] = sum_k A[i, k] * B[k, j]

    这段代码刻意写成三重 for 循环,而不是 A @ B —— 因为它一比一地
    对应「朴素 GPU kernel」里每个线程干的活:一个线程负责一个 C[i, j],
    自己把 A 的第 i 行和 B 的第 j 列从全局内存里读进来做点积。
    读者看这段循环,就能直观数出:内层 k 循环走了 N 步,每步读 2 个
    全局内存元素(A[i,k] 和 B[k,j])。这正是后面访存模型的依据。

    参数:
        A: 形状 (M, K) 的二维数组
        B: 形状 (K, N) 的二维数组
    返回:
        C: 形状 (M, N) 的二维数组
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("matmul_naive 只接受二维矩阵")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"内维不匹配: A 是 {A.shape}, B 是 {B.shape}")

    C = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += A[i, k] * B[k, j]  # 每步 2 次「全局内存读」
            C[i, j] = acc
    return C


def matmul_tiled(A: np.ndarray, B: np.ndarray, tile: int) -> np.ndarray:
    """分块(tiled)矩阵乘 —— 模拟 GPU shared-memory 分块 kernel 的访存顺序。

    关键思想(与 CUDA kernel 一一对应):
      - 把输出 C 切成 tile×tile 的小块;一个「线程块」负责算一个 C 子块。
      - 沿着内维 K 一段一段(每段 tile 长)推进:
          * 先把 A 的一个 (tile × tile) 子块、B 的一个 (tile × tile) 子块
            「搬进 shared memory」(这里用 numpy 切片代表这一次搬运);
          * 然后子块内做一次小矩阵乘,累加到 C 子块上。
      - 一个 A 子块被 C 子块里 tile 列复用,一个 B 子块被 tile 行复用 ——
        这就是分块省访存的来源:搬一次,用 tile 次。

    本函数用 numpy 的块乘 (Asub @ Bsub) 来代表「子块进片上后的高速计算」,
    因此数值结果与 A @ B 完全一致(见 tests)。它的价值不在于快,而在于
    它的**循环结构**精确复刻了 GPU 分块 kernel,让访存建模有据可依。

    参数:
        A:    (M, K)
        B:    (K, N)
        tile: 分块边长 T(>=1)。不要求整除;边界块会自动截断。
    返回:
        C:    (M, N),数值 == A @ B
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("matmul_tiled 只接受二维矩阵")
    if not isinstance(tile, (int, np.integer)) or tile < 1:
        raise ValueError("tile 必须是 >=1 的整数")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"内维不匹配: A 是 {A.shape}, B 是 {B.shape}")

    C = np.zeros((M, N), dtype=np.float64)
    # i0/j0 遍历输出块的左上角;k0 沿内维推进(对应 kernel 的 k-phase 循环)
    for i0 in range(0, M, tile):
        i1 = min(i0 + tile, M)
        for j0 in range(0, N, tile):
            j1 = min(j0 + tile, N)
            acc = np.zeros((i1 - i0, j1 - j0), dtype=np.float64)
            for k0 in range(0, K, tile):
                k1 = min(k0 + tile, K)
                Asub = A[i0:i1, k0:k1]   # 「搬进 shared memory」的 A 子块
                Bsub = B[k0:k1, j0:j1]   # 「搬进 shared memory」的 B 子块
                acc += Asub @ Bsub        # 子块在「片上」高速累乘
            C[i0:i1, j0:j1] = acc
    return C


# =====================================================================
# 第二部分:解析访存模型(不跑 kernel,用公式定量算 HBM / 强度 / 复用)
# =====================================================================


@dataclass(frozen=True)
class MemoryModel:
    """一次 matmul 的解析访存分析结果(单位见各字段注释)。

    约定:M=N=K=n 的方阵情形是最常用的对照(可用 from_square 构造);
    通用矩形情形用 analyze() 直接传 M/N/K。
    """

    scheme: str          # "naive" 或 "tiled"
    M: int               # C 的行数
    N: int               # C 的列数
    K: int               # 内维
    tile: int            # 分块边长 T(naive 记为 1)
    dtype_bytes: int     # 每个元素字节数(fp32=4, fp64=8, fp16=2)

    flops: float         # 浮点运算次数(乘加各算 1,即 2*M*N*K)
    hbm_read_elems: float    # 从 HBM 读入的「元素」个数
    hbm_read_bytes: float    # 从 HBM 读入的字节数
    hbm_write_elems: float   # 写回 HBM 的元素个数(= M*N,C 各写一次)
    hbm_bytes_total: float   # 读+写总字节数
    arithmetic_intensity: float  # 算术强度 = FLOPs / HBM总字节(FLOP/Byte)
    reuse_factor: float      # 复用率 = 若无分块的读量 / 实际读量


def _matmul_flops(M: int, N: int, K: int) -> float:
    """matmul 的 FLOPs:每个 C 元素做 K 次乘 + K 次加 = 2K,共 M*N 个元素。"""
    return 2.0 * M * N * K


def analyze(
    M: int,
    N: int,
    K: int,
    tile: int,
    scheme: str = "tiled",
    dtype_bytes: int = 4,
) -> MemoryModel:
    """对一次 M×N×K 的 matmul 做解析访存分析。

    ------------------------------------------------------------------
    模型推导(第一性原理,方阵 n 的直觉先记住,通用式在下面)
    ------------------------------------------------------------------
    朴素 kernel(scheme="naive",等价 tile=1):
      每个输出 C[i,j] 独立地把 A 的第 i 行(K 个元素)、B 的第 j 列
      (K 个元素)从 HBM 读进来 → 每个元素读 2K 个,共 M*N 个元素:
          read_elems_naive = 2 * M * N * K
      (对方阵就是 2 n^3;这就是「访存爆炸」的根源。)

    分块 kernel(scheme="tiled",tile=T):
      把 C 切成 T×T 块;算一个输出块时,沿 K 分 ceil(K/T) 个阶段,
      每阶段把一个 A 子块(T×T)和一个 B 子块(T×T)搬进 shared memory
      各一次,块内 T*T 个线程共享它们。
      - A 侧读入总量:C 有 ceil(M/T)*ceil(N/T) 个输出块,每个块沿 K
        读 ceil(K/T) 个 A 子块,每子块最多 T*T 元素。但同一「块行」
        上 A 子块与 j 无关,可在数学上化简 —— 更干净的记法是:
        A 的每个元素被读入的次数 = 「它所在块行对应的输出块列数」
        = ceil(N/T)。于是
            read_A ≈ (M*K) * ceil(N/T)
            read_B ≈ (K*N) * ceil(M/T)
        对方阵 M=N=K=n、且 T 整除 n:
            read_A = n^2 * (n/T) = n^3 / T,  read_B 同理 = n^3 / T
            read_elems_tiled = 2 n^3 / T
      → 相比朴素的 2 n^3,分块把 HBM 读**除以了 T**。这就是「tile 越大
        越省访存」的定量证据。

    复用率(reuse factor):
        reuse = read_elems_naive / read_elems_tiled
      方阵整除时正好 = T。它衡量「一个从 HBM 搬来的元素平均被算了几次」。

    算术强度(arithmetic intensity, AI):
        AI = FLOPs / HBM总字节
      FLOPs 与方案无关(都是 2*M*N*K);HBM 字节数分块后大幅下降,
      所以分块把 AI 抬高约 T 倍,把 kernel 从「访存受限」推向「计算受限」。

    ------------------------------------------------------------------
    参数:
        M, N, K:      矩阵维度(C 是 M×N,内维 K)
        tile:         分块边长 T;scheme="naive" 时此值被忽略(记为 1)
        scheme:       "naive" 或 "tiled"
        dtype_bytes:  元素字节(fp32=4 默认, fp64=8, fp16=2)
    返回:
        MemoryModel 数据类
    """
    if scheme not in ("naive", "tiled"):
        raise ValueError('scheme 必须是 "naive" 或 "tiled"')
    for name, v in (("M", M), ("N", N), ("K", K)):
        if not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError(f"{name} 必须是 >=1 的整数, 收到 {v!r}")
    if dtype_bytes < 1:
        raise ValueError("dtype_bytes 必须 >=1")

    flops = _matmul_flops(M, N, K)
    write_elems = float(M * N)  # C 每个元素写回一次

    if scheme == "naive":
        eff_tile = 1
        # 朴素:每个 C 元素读 A 一行 + B 一列 = 2K;共 M*N 个
        read_elems = 2.0 * M * N * K
    else:
        if not isinstance(tile, (int, np.integer)) or tile < 1:
            raise ValueError("tiled 方案要求 tile 是 >=1 的整数")
        eff_tile = int(tile)
        # 分块:A 每元素被读 ceil(N/T) 次;B 每元素被读 ceil(M/T) 次。
        # ceil 用整数算术:(x + T - 1) // T
        n_col_blocks = (N + eff_tile - 1) // eff_tile  # ceil(N/T)
        n_row_blocks = (M + eff_tile - 1) // eff_tile  # ceil(M/T)
        read_A = float(M * K) * n_col_blocks
        read_B = float(K * N) * n_row_blocks
        read_elems = read_A + read_B

    read_bytes = read_elems * dtype_bytes
    write_bytes = write_elems * dtype_bytes
    hbm_bytes_total = read_bytes + write_bytes

    ai = flops / hbm_bytes_total if hbm_bytes_total > 0 else float("inf")

    # 复用率:以「朴素读量」为分母基线 —— 分块相对朴素省了多少倍
    naive_read = 2.0 * M * N * K
    reuse = naive_read / read_elems if read_elems > 0 else float("inf")

    return MemoryModel(
        scheme=scheme,
        M=int(M),
        N=int(N),
        K=int(K),
        tile=int(eff_tile),
        dtype_bytes=int(dtype_bytes),
        flops=flops,
        hbm_read_elems=read_elems,
        hbm_read_bytes=read_bytes,
        hbm_write_elems=write_elems,
        hbm_bytes_total=hbm_bytes_total,
        arithmetic_intensity=ai,
        reuse_factor=reuse,
    )


def analyze_square(n: int, tile: int, scheme: str = "tiled",
                   dtype_bytes: int = 4) -> MemoryModel:
    """方阵便捷入口:M=N=K=n。最常用于教学对照与画曲线。"""
    return analyze(n, n, n, tile, scheme=scheme, dtype_bytes=dtype_bytes)


def max_tile_for_smem(smem_bytes: int, dtype_bytes: int = 4,
                      num_tiles: int = 2) -> int:
    """给定 shared memory 容量,反推能放下的最大方形 tile 边长 T。

    分块 kernel 通常同时把 A 子块和 B 子块放进 shared memory,共 num_tiles
    个 T×T 数组,占用 num_tiles * T^2 * dtype_bytes 字节。令其 <= smem_bytes:
        T <= sqrt( smem_bytes / (num_tiles * dtype_bytes) )

    这解释了「为什么 tile 不能无限大」:一旦 T 超过这个上限,子块塞不进
    片上 shared memory,分块就退化甚至启动失败。典型 NVIDIA SM 的 shared
    memory 约 48KB~ 228KB。

    参数:
        smem_bytes:  可用 shared memory 字节数
        dtype_bytes: 元素字节
        num_tiles:   同时驻留片上的 tile 个数(A、B 各一 → 2)
    返回:
        最大可行 T(向下取整,至少 1)
    """
    if smem_bytes < 1 or dtype_bytes < 1 or num_tiles < 1:
        raise ValueError("参数必须为正")
    t = int(np.floor(np.sqrt(smem_bytes / (num_tiles * dtype_bytes))))
    return max(1, t)


def sweep_tiles(n: int, tiles, dtype_bytes: int = 4):
    """对一组 tile 大小做扫描,返回并列的数组,方便画曲线/写表。

    返回 dict,键:
        tiles, hbm_read_bytes, hbm_bytes_total,
        arithmetic_intensity, reuse_factor
    另附 naive_* 标量作为基线对照。
    """
    tiles = list(tiles)
    read_bytes, total_bytes, ai, reuse = [], [], [], []
    for t in tiles:
        m = analyze_square(n, t, scheme="tiled", dtype_bytes=dtype_bytes)
        read_bytes.append(m.hbm_read_bytes)
        total_bytes.append(m.hbm_bytes_total)
        ai.append(m.arithmetic_intensity)
        reuse.append(m.reuse_factor)

    base = analyze_square(n, 1, scheme="naive", dtype_bytes=dtype_bytes)
    return {
        "n": n,
        "tiles": np.asarray(tiles, dtype=float),
        "hbm_read_bytes": np.asarray(read_bytes, dtype=float),
        "hbm_bytes_total": np.asarray(total_bytes, dtype=float),
        "arithmetic_intensity": np.asarray(ai, dtype=float),
        "reuse_factor": np.asarray(reuse, dtype=float),
        "naive_read_bytes": base.hbm_read_bytes,
        "naive_bytes_total": base.hbm_bytes_total,
        "naive_ai": base.arithmetic_intensity,
    }


if __name__ == "__main__":
    # 手动 smoke:方阵 n=512,看看不同 tile 的对照
    n = 512
    print(f"方阵 n={n}, fp32(4 字节)")
    base = analyze_square(n, 1, scheme="naive")
    print(f"  [naive]  HBM读={base.hbm_read_bytes/1e6:8.2f} MB  "
          f"AI={base.arithmetic_intensity:6.2f} FLOP/B  复用={base.reuse_factor:.1f}x")
    for t in (16, 32, 64, 128):
        m = analyze_square(n, t, scheme="tiled")
        print(f"  [tile={t:3d}] HBM读={m.hbm_read_bytes/1e6:8.2f} MB  "
              f"AI={m.arithmetic_intensity:6.2f} FLOP/B  复用={m.reuse_factor:.1f}x")
    print(f"  48KB shared memory 下最大方形 tile = "
          f"{max_tile_for_smem(48*1024)}")
