"""
collective_cost.py —— 集合通信(collective communication)AllReduce 的 α-β 代价模型

对应 ../../05_网络与通信_RDMA_NCCL_集合通信算法_拓扑.md。

用一个简单但抓住本质的 **Hockney α-β 模型** 量化并对比三种 AllReduce 算法在
不同卡数 P、不同消息大小 N(字节)下的耗时:

    T = α · steps(步数/消息数) + β · bytes(临界路径上传输的字节数)

其中:
  · α = 每条消息的固定延迟(startup/link latency),单位秒——决定"小消息"性能
  · β = 每字节传输时间 = 1 / 链路带宽,单位秒/字节——决定"大消息"性能
  · steps = 通信轮数(每轮至少一次 α 开销)
  · bytes = 临界路径上一个节点要搬运的字节数(乘 β 得带宽项耗时)

三种算法(全都实现 AllReduce = "每个节点最终拿到所有节点数据之和"):

  1. ring(环状,Baidu/NCCL 的 ring-allreduce = reduce-scatter + all-gather)
       steps = 2(P-1)                     ← 延迟随 P 线性增长(小消息吃亏)
       bytes = 2(P-1)/P · N               ← 带宽最优,P→∞ 时 →2N(大消息赢)

  2. tree(朴素二叉树 = reduce 上行 + broadcast 下行)
       steps = 2·log2(P)                  ← 延迟随 P 对数增长(小消息赢)
       bytes = 2·log2(P) · N              ← 临界路径带宽差,随 log P 增长(大消息吃亏)

  3. double_binary_tree(NCCL 的双二叉树,大规模默认算法)
       steps = 2·log2(P)                  ← 保留树的对数延迟
       bytes = 2(P-1)/P · N               ← 又拿到环的最优带宽 → 两全其美(Pareto 最优)

第一性原理:AllReduce 的两个物理下界是
  · 延迟下界 ~ log(P) 跳(信息至少要传播 log P 层)
  · 带宽下界 ~ 2N(每个节点至少要发出/收进 ~2N 数据:reduce-scatter + all-gather)
ring 打满带宽下界但延迟 O(P);tree 打满延迟下界但带宽 O(log P·N);
double-binary-tree 同时逼近两个下界——这正是 NCCL 在大集群默认用它的原因。
"""
from __future__ import annotations
from dataclasses import dataclass
import math

# ---------------------------------------------------------------------------
# 0) α-β 基本公式
# ---------------------------------------------------------------------------
def alpha_beta_time(steps: float, nbytes: float, alpha: float, beta: float) -> float:
    """Hockney α-β 模型:T = α·steps + β·bytes(秒)。"""
    return alpha * steps + beta * nbytes


def bandwidth_to_beta(gbps: float) -> float:
    """把链路带宽(GB/s,这里 1 GB = 1e9 字节)换算成 β(秒/字节)。"""
    if gbps <= 0:
        raise ValueError("带宽必须为正")
    return 1.0 / (gbps * 1e9)


def _log2_ceil(P: int) -> int:
    """⌈log2(P)⌉;P=1 时为 0(单卡无需通信)。对非 2 的幂也给出树的真实深度。"""
    if P < 1:
        raise ValueError("P 必须 >= 1")
    if P == 1:
        return 0
    return int(math.ceil(math.log2(P)))


# ---------------------------------------------------------------------------
# 1) 步数公式(α 项的系数)
# ---------------------------------------------------------------------------
def ring_steps(P: int) -> int:
    """ring:reduce-scatter(P-1 步) + all-gather(P-1 步) = 2(P-1)。"""
    return 2 * (P - 1)


def tree_steps(P: int) -> int:
    """tree:reduce 上行 log2(P) 步 + broadcast 下行 log2(P) 步 = 2·log2(P)。"""
    return 2 * _log2_ceil(P)


def dbt_steps(P: int) -> int:
    """double-binary-tree:两棵树深度仍是 log2(P),reduce+broadcast = 2·log2(P)。"""
    return 2 * _log2_ceil(P)


# ---------------------------------------------------------------------------
# 2) 每节点通信量(β 项的系数;临界路径上搬运的字节数)
# ---------------------------------------------------------------------------
def ring_coeff(P: int) -> float:
    """ring 的带宽系数:2(P-1)/P(乘 N 得字节数)。P→∞ 时 →2。"""
    return 2.0 * (P - 1) / P


def tree_coeff(P: int) -> float:
    """tree 的带宽系数:2·log2(P)(临界路径每一跳都搬满 N)。"""
    return 2.0 * _log2_ceil(P)


def dbt_coeff(P: int) -> float:
    """double-binary-tree 的带宽系数:与 ring 相同的最优 2(P-1)/P。"""
    return 2.0 * (P - 1) / P


def ring_bytes(N: float, P: int) -> float:
    """ring AllReduce 每节点通信量 = 2(P-1)/P · N。"""
    return ring_coeff(P) * N


def tree_bytes(N: float, P: int) -> float:
    """tree AllReduce 临界路径通信量 = 2·log2(P) · N。"""
    return tree_coeff(P) * N


def dbt_bytes(N: float, P: int) -> float:
    """double-binary-tree 每节点通信量 = 2(P-1)/P · N(带宽最优)。"""
    return dbt_coeff(P) * N


# ---------------------------------------------------------------------------
# 3) 算法注册表 + 统一入口
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Algo:
    """一个 AllReduce 算法:名字、步数函数、带宽系数函数、给人看的公式串。"""
    name: str
    steps_fn: "callable"
    coeff_fn: "callable"
    steps_formula: str
    bytes_formula: str


ALGOS: dict[str, Algo] = {
    "ring": Algo(
        "ring", ring_steps, ring_coeff,
        "2(P-1)", "2(P-1)/P · N",
    ),
    "tree": Algo(
        "tree", tree_steps, tree_coeff,
        "2·log2(P)", "2·log2(P) · N",
    ),
    "double_binary_tree": Algo(
        "double_binary_tree", dbt_steps, dbt_coeff,
        "2·log2(P)", "2(P-1)/P · N",
    ),
}


def allreduce_steps(algo: str, P: int) -> int:
    return ALGOS[algo].steps_fn(P)


def allreduce_bytes(algo: str, N: float, P: int) -> float:
    return ALGOS[algo].coeff_fn(P) * N


def allreduce_time(algo: str, N: float, P: int, alpha: float, beta: float) -> float:
    """某算法在 (N 字节, P 卡, α, β) 下的 AllReduce 耗时(秒)。"""
    steps = ALGOS[algo].steps_fn(P)
    nbytes = ALGOS[algo].coeff_fn(P) * N
    return alpha_beta_time(steps, nbytes, alpha, beta)


# ---------------------------------------------------------------------------
# 4) 有用的派生量:总线带宽 busbw、交叉点 crossover
# ---------------------------------------------------------------------------
def bus_bandwidth(algo: str, N: float, P: int, alpha: float, beta: float) -> float:
    """
    NCCL 口径的"总线带宽"busbw(GB/s):
        algbw = N / T                     ← 算法带宽(把 N 字节 AllReduce 完的等效速率)
        busbw = algbw · 2(P-1)/P          ← 归一化到硬件链路利用率,便于横比不同 P
    busbw 越接近链路峰值,说明算法越吃满硬件。ring/DBT 大消息时逼近峰值,tree 落后。
    """
    T = allreduce_time(algo, N, P, alpha, beta)
    if T <= 0:
        return 0.0
    algbw = N / T                       # 字节/秒
    busbw = algbw * (2.0 * (P - 1) / P)
    return busbw / 1e9                  # 转成 GB/s


def crossover_size(P: int, alpha: float, beta: float, a: str, b: str) -> float:
    """
    解算法 a、b 耗时相等的消息大小 N*(字节)。
        α·steps_a + β·coeff_a·N = α·steps_b + β·coeff_b·N
      ⇒ N* = α·(steps_a - steps_b) / (β·(coeff_b - coeff_a))
    典型用法:crossover_size(P, α, β, "ring", "tree") → 小于它 tree 赢、大于它 ring 赢。
    返回 <=0 表示不存在正交叉点(某算法在全区间恒优)。
    """
    sa, sb = ALGOS[a].steps_fn(P), ALGOS[b].steps_fn(P)
    ca, cb = ALGOS[a].coeff_fn(P), ALGOS[b].coeff_fn(P)
    denom = beta * (cb - ca)
    if denom == 0:
        return -1.0
    N = alpha * (sa - sb) / denom
    return N if N > 0 else -1.0


def best_algo(N: float, P: int, alpha: float, beta: float) -> str:
    """给定 (N, P, α, β),返回耗时最小的算法名。"""
    return min(ALGOS, key=lambda k: allreduce_time(k, N, P, alpha, beta))


# ---------------------------------------------------------------------------
# 5) 一个默认的网络画像(可按需替换),便于 demo/测试直接用
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Network:
    """一个链路画像:延迟 α(秒)与带宽(GB/s)。"""
    alpha: float = 5e-6              # 5 微秒:典型 NVLink/IB 单跳延迟量级
    bandwidth_gbps: float = 100.0   # 100 GB/s:一条高速链路的量级

    @property
    def beta(self) -> float:
        return bandwidth_to_beta(self.bandwidth_gbps)


DEFAULT_NET = Network()
