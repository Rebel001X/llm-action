"""两篇奠基论文的记号对照与「记号写反」的量化后果。

服务于正文 09-2023奠基-两篇同期论文的异同。

两篇论文的 p / q 含义**完全相反**：

    Leviathan, Kalman, Matias (arXiv 2211.17192)
        M_p / p = target（目标模型，要加速的那个）
        M_q / q = draft （草稿模型，近似模型）
        草稿长度记为 gamma
        接受判据（Algorithm 1）: n <- min({i-1 | r_i > p_i(x)/q_i(x)} u {gamma})
        即：接受当且仅当 r_i <= p_i(x)/q_i(x)

    Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper (arXiv 2302.01318)
        p = draft （"K draft tokens ... generated from p(.|.)"）
        q = target
        草稿长度记为 K（lookahead）
        接受判据（Algorithm 2）: if r < min(1, q(x~)/p(x~)) then accept

本模块验证三件事：

1. 把记号映射对之后，两条判据**逐点给出同一个接受概率**（两篇是同一个算法）。
2. 若照抄 Chen 的 `min(1, q/p)` 却沿用 Leviathan 的变量命名（q=draft, p=target），
   得到的是「反向判据」min(1, q_draft/p_target)。它**不会报错**，反而让
   **实测接受率上升**——这正是它危险的地方。
3. 反向判据把输出分布拽向草稿模型。在本模块的算例里，它**恰好精确复现草稿分布**。

全部为精确计算（不掷骰子），CPU 毫秒级。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 算例：与 04 篇 / spec.py --demo 同一组数字（|V|=5，起始 token s0=0 的那一步）
# ---------------------------------------------------------------------------

_TARGET_RAW = np.array([0.2024, 0.1564, 0.3386, 0.1982, 0.1045])
_DRAFT_RAW = np.array([0.2060, 0.2330, 0.3406, 0.0540, 0.1664])

# 上面两行是 04 篇 / `spec.py --demo` 打印出来的**四位小数**，本身带舍入残差
# （TARGET 和为 1.0001）。这里重新归一化，好让「无损」验出 1e-16 而不是 1e-4。
TARGET = _TARGET_RAW / _TARGET_RAW.sum()
DRAFT = _DRAFT_RAW / _DRAFT_RAW.sum()


def accept_prob_leviathan(p_target: np.ndarray, q_draft: np.ndarray) -> np.ndarray:
    """Leviathan Algorithm 1 的接受概率，逐 token。

    原文判据 `r_i > p_i(x)/q_i(x)` 触发拒绝，故接受概率 = min(1, p/q)，
    其中 **p 是 target、q 是 draft**。
    """
    return np.minimum(1.0, p_target / q_draft)


def accept_prob_chen(p_draft: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """Chen Algorithm 2 的接受概率，逐 token。

    原文判据 `r < min(1, q(x~)/p(x~))`，其中 **p 是 draft、q 是 target**。
    形参名刻意沿用 Chen 的含义，以便与上面的函数对照。
    """
    return np.minimum(1.0, q_target / p_draft)


def accept_prob_swapped(p_target: np.ndarray, q_draft: np.ndarray) -> np.ndarray:
    """**错误**判据：照抄 Chen 的 min(1, q/p) 但沿用 Leviathan 的变量命名。

    于是算成了 min(1, q_draft / p_target)——分子分母整个反过来。
    """
    return np.minimum(1.0, q_draft / p_target)


def residual(p_target: np.ndarray, q_draft: np.ndarray) -> np.ndarray:
    """残差分布 norm(max(0, p_target - q_draft))（两篇写法一致，只是记号相反）。"""
    r = np.maximum(0.0, p_target - q_draft)
    s = r.sum()
    if s <= 0.0:
        return p_target.copy()
    return r / s


def measured_acceptance_rate(accept: np.ndarray, q_draft: np.ndarray) -> float:
    """一次起草被接受的总概率 = sum_x q_draft(x) * a(x)。

    这就是引擎打点打出来的「接受率」。
    """
    return float((q_draft * accept).sum())


def first_token_dist(accept: np.ndarray, p_target: np.ndarray,
                     q_draft: np.ndarray) -> np.ndarray:
    """给定接受概率向量，算出「接受 + 拒绝后用残差重采」后首 token 的精确分布。

    残差始终按正确公式 norm(max(0, p_target - q_draft)) 计算——
    也就是说，本函数把「只写反了接受判据」这一种 bug 单独隔离出来。
    """
    accepted = q_draft * accept                 # Pr[采到 x 且接受]
    reject_mass = 1.0 - float(accepted.sum())
    return accepted + reject_mass * residual(p_target, q_draft)


def _demo() -> None:
    lev = accept_prob_leviathan(TARGET, DRAFT)
    chen = accept_prob_chen(p_draft=DRAFT, q_target=TARGET)
    bad = accept_prob_swapped(TARGET, DRAFT)

    np.set_printoptions(precision=4, suppress=True)
    print("target p (Leviathan 记号 p / Chen 记号 q):", TARGET)
    print("draft  q (Leviathan 记号 q / Chen 记号 p):", DRAFT)
    print()
    print("Leviathan min(1, p_target/q_draft) :", lev)
    print("Chen      min(1, q_target/p_draft) :", chen)
    print("两者最大逐点差                     :", float(np.max(np.abs(lev - chen))))
    print()
    print("写反的判据 min(1, q_draft/p_target):", bad)
    print()

    a_ok = measured_acceptance_rate(lev, DRAFT)
    a_bad = measured_acceptance_rate(bad, DRAFT)
    print(f"正确判据下的接受率 beta = {a_ok:.4f}")
    print(f"写反判据下的接受率      = {a_bad:.4f}   <- 看板上会「变好」")
    print()

    d_ok = first_token_dist(lev, TARGET, DRAFT)
    d_bad = first_token_dist(bad, TARGET, DRAFT)
    print("正确判据的首 token 分布:", d_ok)
    print("目标模型分布           :", TARGET)
    print("  最大逐点误差:", float(np.max(np.abs(d_ok - TARGET))))
    print()
    print("写反判据的首 token 分布:", d_bad)
    print("  与 target 的最大逐点误差:", float(np.max(np.abs(d_bad - TARGET))))
    print("  与 draft  的最大逐点误差:", float(np.max(np.abs(d_bad - DRAFT))))


if __name__ == "__main__":
    _demo()
