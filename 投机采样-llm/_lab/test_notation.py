"""两篇奠基论文记号对照的可执行验证（服务于正文 09）。"""

from __future__ import annotations

import numpy as np
import pytest

from notation import (
    DRAFT,
    TARGET,
    accept_prob_chen,
    accept_prob_leviathan,
    accept_prob_swapped,
    first_token_dist,
    measured_acceptance_rate,
    residual,
)
from spec import ToyMarkov, beta_overlap, perturb


def _random_pair(seed: int, v: int = 6):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(v))
    q = rng.dirichlet(np.ones(v))
    return p, q


# ---------------------------------------------------------------------------
# 1. 两篇论文的接受判据是同一个算法
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(12))
def test_two_papers_give_identical_acceptance_probability(seed):
    """把记号映射对之后，Leviathan 与 Chen 的接受概率逐点相等。

    Leviathan: min(1, p/q)，p=target, q=draft
    Chen:      min(1, q/p)，p=draft,  q=target
    """
    p_target, q_draft = _random_pair(seed)
    lev = accept_prob_leviathan(p_target, q_draft)
    chen = accept_prob_chen(p_draft=q_draft, q_target=p_target)
    assert np.max(np.abs(lev - chen)) < 1e-15


@pytest.mark.parametrize("seed", range(12))
def test_both_papers_are_l1_lossless(seed):
    """两篇的判据 + 残差重采都精确还原 target（L1 分布无损）。"""
    p_target, q_draft = _random_pair(seed)
    for accept in (accept_prob_leviathan(p_target, q_draft),
                   accept_prob_chen(p_draft=q_draft, q_target=p_target)):
        out = first_token_dist(accept, p_target, q_draft)
        assert np.max(np.abs(out - p_target)) < 1e-14


def test_acceptance_rate_equals_overlap_in_both_notations():
    """两篇口径下算出的接受率都等于 sum min(p,q) = 1 - TV(p,q)。"""
    beta = measured_acceptance_rate(accept_prob_leviathan(TARGET, DRAFT), DRAFT)
    assert beta == pytest.approx(float(np.minimum(TARGET, DRAFT).sum()), abs=1e-12)
    assert beta == pytest.approx(beta_overlap(TARGET, DRAFT), abs=1e-12)


# ---------------------------------------------------------------------------
# 2. 记号写反：接受率反而上升（所以看板发现不了）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(12))
def test_swapped_notation_inflates_measured_acceptance_rate(seed):
    """写反判据后，实测接受率**不降反升**——这正是它难以被发现的原因。"""
    p_target, q_draft = _random_pair(seed)
    good = measured_acceptance_rate(accept_prob_leviathan(p_target, q_draft), q_draft)
    bad = measured_acceptance_rate(accept_prob_swapped(p_target, q_draft), q_draft)
    assert bad >= good - 1e-15


def test_swapped_notation_inflates_on_the_article_example():
    """正文 §5 的具体数字：0.8558 -> 0.9607。"""
    good = measured_acceptance_rate(accept_prob_leviathan(TARGET, DRAFT), DRAFT)
    bad = measured_acceptance_rate(accept_prob_swapped(TARGET, DRAFT), DRAFT)
    assert good == pytest.approx(0.8558, abs=5e-4)
    assert bad == pytest.approx(0.9607, abs=5e-4)


# ---------------------------------------------------------------------------
# 3. 记号写反：分布被拽向草稿模型（正确性问题）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(12))
def test_swapped_notation_is_biased(seed):
    """写反判据破坏 L1 无损性：输出分布不再等于 target。"""
    p_target, q_draft = _random_pair(seed)
    out = first_token_dist(accept_prob_swapped(p_target, q_draft), p_target, q_draft)
    err = float(np.max(np.abs(out - p_target)))
    assert err > 1e-3, "构造上 draft != target 时必然有可观偏差"


@pytest.mark.parametrize("seed", range(12))
def test_swapped_notation_pulls_output_toward_the_draft(seed):
    """偏差方向是**朝向草稿模型**，不是随机乱偏。"""
    p_target, q_draft = _random_pair(seed)
    out = first_token_dist(accept_prob_swapped(p_target, q_draft), p_target, q_draft)
    to_target = float(np.abs(out - p_target).sum())
    to_draft = float(np.abs(out - q_draft).sum())
    assert to_draft < to_target


def test_swapped_notation_reproduces_the_draft_on_the_article_example():
    """本算例里只有一个 token 满足 p_target > q_draft（残差是 one-hot），
    此时写反判据**恰好精确复现草稿分布**。正文 §5 引用的就是这个数字。"""
    out = first_token_dist(accept_prob_swapped(TARGET, DRAFT), TARGET, DRAFT)
    assert np.max(np.abs(out - DRAFT)) < 1e-12
    assert np.max(np.abs(out - TARGET)) == pytest.approx(0.1442, abs=1e-4)


def test_residual_is_one_hot_on_the_article_example():
    """前一条的前提检查：残差确实集中在唯一被草稿低估的 token 上。"""
    r = residual(TARGET, DRAFT)
    assert int(np.argmax(r)) == 3
    assert r[3] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 4. 两篇 alpha 口径不同：解析估计量 vs 实测接受计数
# ---------------------------------------------------------------------------

def _chen_caliber(alpha: float, gamma: int) -> float:
    """Chen Figure 1 中图的纵轴：平均接受 token 数 / (K+1)。"""
    return ((1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)) / (gamma + 1)


def test_leviathan_and_chen_alpha_calibers_are_different_quantities():
    """Leviathan 报的 alpha 是 E[min(p,q)]（Corollary 3.6，逐位解析量）；
    Chen 报的是「平均接受 token 数 /(K+1)」（Figure 1 中图，整轮计数量）。

    在一对真实的玩具模型上，两个数明显不等。
    """
    target = ToyMarkov(V=6, seed=3)
    draft = perturb(target, eps=1.5, seed=4)

    rng = np.random.default_rng(0)
    last, betas = 0, []
    for _ in range(4000):
        p = target.dist(last)
        betas.append(beta_overlap(p, draft.dist(last)))
        last = int(rng.choice(len(p), p=p))
    alpha_lev = float(np.mean(betas))

    alpha_chen = _chen_caliber(alpha_lev, gamma=4)

    assert 0.0 < alpha_lev < 1.0
    assert 0.0 < alpha_chen < 1.0
    assert abs(alpha_chen - alpha_lev) > 0.05


def test_alpha_caliber_gap_changes_sign():
    """**两个口径的差值连符号都不固定** —— 所以不能"换算"，只能各报各的。

    小 alpha 时 Chen 口径偏高（赠品 token 把下界托到 1/(K+1)）；
    大 alpha 时 Chen 口径偏低（gamma 截断把上界压住）。
    """
    assert _chen_caliber(0.20, gamma=3) > 0.20      # 正号
    assert _chen_caliber(0.70, gamma=3) < 0.70      # 负号
    assert _chen_caliber(0.10, gamma=7) > 0.10
    assert _chen_caliber(0.70, gamma=7) < 0.70


def test_alpha_caliber_gap_is_large_at_leviathan_headline_setting():
    """Leviathan Table 2 的 EnDe / T5-small / temp=1 那一行：alpha=0.62, gamma=7。
    同一套系统若按 Chen 的口径画图，纵轴只有 0.32 —— 差 0.30。"""
    gap = 0.62 - _chen_caliber(0.62, gamma=7)
    assert gap == pytest.approx(0.298, abs=0.005)
