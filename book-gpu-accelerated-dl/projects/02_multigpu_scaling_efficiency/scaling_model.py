# -*- coding: utf-8 -*-
"""
scaling_model.py — 多 GPU 扩展效率模型(Multi-GPU Scaling Efficiency Model)

本质:把"用 N 张卡训练同一个模型能快多少"这件事,用两条第一性原理拆开:
  1) Amdahl 定律:一个程序里有一部分是"串行的"(serial),它拿再多卡也快不了;
     只有"可并行"(parallel)的那部分才能被 N 张卡分摊。
  2) 通信开销(communication overhead):多卡之间要同步梯度(all-reduce),
     这份开销 ≈ 数据量 / 带宽,而且它会随着卡数 N 增长(卡越多,越难同步)。

把这两件事写成公式,就能定量回答:
  - 加速比 Speedup(N):用 N 卡比单卡快几倍?
  - 扩展效率 Efficiency(N) = Speedup(N) / N ∈ (0, 1]:这 N 张卡"利用率"多高?

这就是工业界做"强扩展"(strong scaling)容量规划时最核心的手算模型。

术语中英对照:
  - Speedup            加速比         S(N) = T(1) / T(N)
  - Efficiency         扩展效率       E(N) = S(N) / N
  - Serial fraction    串行占比       s ∈ [0, 1]
  - Parallel fraction  并行占比       p = 1 - s
  - Strong scaling     强扩展         固定总问题规模,加卡求更快
  - Weak scaling       弱扩展         每卡问题规模固定,加卡求更大
  - All-reduce         全规约         多卡同步梯度的集合通信原语
  - Bandwidth          带宽           单位时间能传多少字节

所有函数都是纯 numpy / 纯 Python,CPU 即可运行,不需要 GPU、不联网、不下模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "amdahl_speedup",
    "amdahl_efficiency",
    "communication_time",
    "CommModel",
    "speedup_with_comm",
    "efficiency_with_comm",
    "scaling_curve",
    "max_speedup_amdahl",
    "half_efficiency_n",
]


# =============================================================================
# 第 1 部分:纯 Amdahl(无通信开销的理想上界)
# =============================================================================

def amdahl_speedup(n: int, serial_fraction: float) -> float:
    r"""
    Amdahl 定律加速比(不含通信开销)。

    公式(书中经典形式):
        S(N) = 1 / ( s + (1 - s) / N )

    其中:
      - N = 卡数(processors)
      - s = serial_fraction 串行占比 ∈ [0, 1]
      - (1 - s) = 并行占比 p,这部分被 N 张卡平摊为 (1-s)/N

    直觉:总时间被归一化成 1。串行的 s 无论多少卡都还是 s;
          并行的 (1-s) 被 N 卡分摊成 (1-s)/N。二者相加就是新的总时间,
          再取倒数就是"快了几倍"。

    参数
    ----
    n : int
        卡数,必须 >= 1。
    serial_fraction : float
        串行占比 s,必须在 [0, 1]。s=0 表示完全可并行(理想),
        s=1 表示完全串行(加卡毫无用处)。

    返回
    ----
    float : 加速比 S(N) >= 1。

    边界性质:
      - S(1) == 1(单卡,定义如此)
      - s == 0 时 S(N) == N(完美线性加速)
      - N -> ∞ 时 S -> 1/s(Amdahl 天花板,见 max_speedup_amdahl)
    """
    _check_n(n)
    _check_fraction(serial_fraction, "serial_fraction")
    p = 1.0 - serial_fraction
    return 1.0 / (serial_fraction + p / n)


def amdahl_efficiency(n: int, serial_fraction: float) -> float:
    r"""
    纯 Amdahl 扩展效率 E(N) = S(N) / N。

    效率衡量"每张卡的平均贡献",取值落在 (0, 1]:
      - E == 1 表示这 N 张卡每张都满负荷(线性加速,只有 s=0 时成立)
      - E 越小表示越多算力被浪费在"等串行部分 / 通信"上
    """
    return amdahl_speedup(n, serial_fraction) / n


def max_speedup_amdahl(serial_fraction: float) -> float:
    r"""
    Amdahl 天花板:N -> ∞ 时的极限加速比 = 1 / s。

    这是"再加卡也超不过"的硬上界。例如 s=0.05(5% 串行),
    则无论堆多少卡,最多也只能快 20 倍。这就是为什么"降低串行占比"
    往往比"买更多卡"更值钱。

    s == 0 时返回 +inf(理论上可无限加速)。
    """
    _check_fraction(serial_fraction, "serial_fraction")
    if serial_fraction == 0.0:
        return float("inf")
    return 1.0 / serial_fraction


# =============================================================================
# 第 2 部分:通信开销模型(comm ≈ 数据量 / 带宽,且随 N 增长)
# =============================================================================

@dataclass
class CommModel:
    r"""
    通信开销模型(communication overhead model)。

    核心思想:多卡训练每一步都要做一次梯度同步(all-reduce)。
    这份通信时间 ≈ 需要传输的数据量 / 带宽,并且会随卡数 N 增长。

    我们把单步计算时间归一化为 1(即单卡跑完并行部分记为 1 个"计算单位"),
    通信时间用一个与之可比的无量纲量表示:

        T_comm(N) = comm_base * f(N)

    其中 comm_base = message_bytes / bandwidth 是"一次通信相对一次计算"的基准占比,
    f(N) 描述通信如何随卡数增长(不同 all-reduce 拓扑增长曲线不同):

      - scaling="linear"     : f(N) = N            (最朴素的参数服务器,最差)
      - scaling="log"        : f(N) = log2(N)      (树形 / 递归折半)
      - scaling="ring"       : f(N) = (N - 1) / N  (Ring-AllReduce,近似常数,最好)

    字段
    ----
    message_bytes : float
        每步需要同步的数据量(字节),≈ 模型参数量 * 每参数字节数。
    bandwidth : float
        互联带宽(字节/单位时间),越大通信越快。必须 > 0。
    scaling : str
        通信随卡数的增长模式,取值 {"linear", "log", "ring"}。

    comm_base(基准通信占比)= message_bytes / bandwidth。
    这是"一次通信 vs 一次计算"的相对成本,是决定强扩展好坏的关键旋钮。
    """

    message_bytes: float = 1.0e8      # 100 MB 梯度(示意)
    bandwidth: float = 1.0e10         # 10 GB/s 互联(示意)
    scaling: str = "ring"

    _valid_scaling: tuple = field(default=("linear", "log", "ring"), repr=False)

    def __post_init__(self) -> None:
        if self.bandwidth <= 0:
            raise ValueError(f"bandwidth 必须 > 0,收到 {self.bandwidth}")
        if self.message_bytes < 0:
            raise ValueError(f"message_bytes 必须 >= 0,收到 {self.message_bytes}")
        if self.scaling not in self._valid_scaling:
            raise ValueError(
                f"scaling 必须是 {self._valid_scaling} 之一,收到 {self.scaling!r}"
            )

    @property
    def comm_base(self) -> float:
        """基准通信占比 = 数据量 / 带宽(一次通信相对一次计算的成本)。"""
        return self.message_bytes / self.bandwidth

    def growth(self, n: int) -> float:
        """通信随卡数增长的因子 f(N)。单卡 N=1 时通信为 0(无需同步)。"""
        _check_n(n)
        if n == 1:
            return 0.0  # 单卡不需要跨卡通信
        if self.scaling == "linear":
            return float(n)
        if self.scaling == "log":
            return float(np.log2(n))
        # ring:(N-1)/N,随 N 增大趋近于 1(近似常数)
        return (n - 1.0) / n

    def comm_time(self, n: int) -> float:
        """N 卡时的通信时间(与归一化计算时间同量纲)。"""
        return self.comm_base * self.growth(n)


def communication_time(
    n: int,
    message_bytes: float,
    bandwidth: float,
    scaling: str = "ring",
) -> float:
    """
    函数式封装:直接由数据量/带宽/拓扑算 N 卡通信时间。
    等价于 CommModel(message_bytes, bandwidth, scaling).comm_time(n)。
    """
    return CommModel(message_bytes, bandwidth, scaling).comm_time(n)


# =============================================================================
# 第 3 部分:Amdahl + 通信 —— 真实强扩展模型
# =============================================================================

def speedup_with_comm(
    n: int,
    serial_fraction: float,
    comm: CommModel | None = None,
) -> float:
    r"""
    含通信开销的加速比(真实强扩展)。

    把单卡总时间归一化为 1,拆成串行 s 与并行 (1-s)。N 卡时:
        T(N) = s + (1 - s)/N + T_comm(N)
             = 串行(不缩) + 并行(被 N 平摊) + 通信(随 N 增长)

    加速比:
        S(N) = T(1) / T(N) = 1 / ( s + (1-s)/N + comm_base * f(N) )

    注意 T(1) = s + (1-s) + f(1)*comm_base = 1 + 0 = 1(单卡通信为 0)。

    通信项 comm_base*f(N) 是"反派":它随 N 增长,
    最终会让 T(N) 掉头上升 —— 也就是"加卡反而更慢",强扩展的经典陷阱。

    参数
    ----
    n : int
        卡数 >= 1。
    serial_fraction : float
        串行占比 s ∈ [0, 1]。
    comm : CommModel | None
        通信模型;None 表示无通信开销(退化成纯 Amdahl)。

    返回
    ----
    float : 加速比 S(N) > 0。
    """
    _check_n(n)
    _check_fraction(serial_fraction, "serial_fraction")
    p = 1.0 - serial_fraction
    comp_time = serial_fraction + p / n          # 计算部分(串行 + 并行)
    comm_t = 0.0 if comm is None else comm.comm_time(n)
    total = comp_time + comm_t
    # total 恒 > 0:n>=1 时 p/n>=0,serial>=0,comm>=0,且 n=1 时 total=1
    return 1.0 / total


def efficiency_with_comm(
    n: int,
    serial_fraction: float,
    comm: CommModel | None = None,
) -> float:
    r"""
    含通信开销的扩展效率 E(N) = S(N) / N ∈ (0, 1]。

    性质保证(见测试):
      - E(1) == 1(单卡效率恒为 1)
      - 0 < E(N) <= 1 对所有合法输入成立(证明见 README 🔬)
      - 通信越大 / 串行越大,E(N) 越低
    """
    return speedup_with_comm(n, serial_fraction, comm) / n


def scaling_curve(
    n_list: Sequence[int] | Iterable[int],
    serial_fraction: float,
    comm: CommModel | None = None,
) -> dict:
    r"""
    批量计算一条扩展曲线,返回可直接绘图的字典。

    返回
    ----
    dict,含四个等长数组:
      - "n"          : 卡数数组
      - "speedup"    : 各卡数下的加速比 S(N)
      - "efficiency" : 各卡数下的效率 E(N)
      - "ideal"      : 理想线性加速(= n 本身),画对角参考线用

    这是 run_demo.py 画图的主入口。
    """
    ns = np.asarray(list(n_list), dtype=int)
    if ns.size == 0:
        raise ValueError("n_list 不能为空")
    sp = np.array([speedup_with_comm(int(n), serial_fraction, comm) for n in ns])
    ef = np.array([efficiency_with_comm(int(n), serial_fraction, comm) for n in ns])
    return {
        "n": ns,
        "speedup": sp,
        "efficiency": ef,
        "ideal": ns.astype(float),
    }


def half_efficiency_n(
    serial_fraction: float,
    comm: CommModel | None = None,
    n_max: int = 100_000,
) -> int | None:
    r"""
    半效率卡数:找到第一个使 E(N) <= 0.5 的 N(工程上常用的"扩展拐点")。

    含义:超过这个卡数,你花的钱有一半以上被浪费在串行 + 通信上了。
    这是容量规划里"该不该再加卡"的一个直观阈值。

    在 [2, n_max] 内线性搜索;若始终 > 0.5(扩展性极好)则返回 None。
    """
    for n in range(2, n_max + 1):
        if efficiency_with_comm(n, serial_fraction, comm) <= 0.5:
            return n
    return None


# =============================================================================
# 内部校验工具
# =============================================================================

def _check_n(n: int) -> None:
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"卡数 n 必须是整数,收到 {type(n).__name__}")
    if n < 1:
        raise ValueError(f"卡数 n 必须 >= 1,收到 {n}")


def _check_fraction(x: float, name: str) -> None:
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"{name} 必须在 [0, 1],收到 {x}")


# =============================================================================
# 自测:直接 `python scaling_model.py` 会打印一张小表
# =============================================================================

if __name__ == "__main__":
    print("=" * 66)
    print("多 GPU 扩展效率模型 · 快速自测")
    print("=" * 66)
    comm = CommModel(message_bytes=2.0e8, bandwidth=1.0e10, scaling="ring")
    s = 0.05
    print(f"串行占比 s = {s}, 通信基准 comm_base = {comm.comm_base:.4f}, "
          f"拓扑 = {comm.scaling}")
    print(f"Amdahl 天花板(1/s) = {max_speedup_amdahl(s):.2f}")
    print(f"{'N':>6} | {'纯Amdahl加速':>12} | {'含通信加速':>10} | {'效率E(N)':>9}")
    print("-" * 66)
    for n in [1, 2, 4, 8, 16, 32, 64, 128]:
        s_pure = amdahl_speedup(n, s)
        s_comm = speedup_with_comm(n, s, comm)
        e_comm = efficiency_with_comm(n, s, comm)
        print(f"{n:>6} | {s_pure:>12.3f} | {s_comm:>10.3f} | {e_comm:>9.3f}")
    hn = half_efficiency_n(s, comm)
    print("-" * 66)
    print(f"半效率卡数(E<=0.5 的首个 N)= {hn}")
