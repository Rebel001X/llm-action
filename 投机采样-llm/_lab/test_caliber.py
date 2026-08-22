"""三种无损口径（L1/L2/L3）的判定性测试。第 07 篇引用本文件。"""
import numpy as np
import pytest

from caliber import (greedy_generate, greedy_speculative, onehot_model,
                     typical_accept_first_token)
from spec import ToyMarkov, perturb


# ---------------------------------------------------------------- L2 贪心等价

@pytest.mark.parametrize("V,eps,gamma", [
    (5, 0.0, 2), (5, 3.0, 4), (5, 8.0, 3),
    (8, 1.0, 2), (8, 3.0, 4), (8, 8.0, 5),
])
def test_greedy_speculative_equals_greedy(V, eps, gamma):
    """**L2**：T=0 时投机采样的输出与贪心解码**逐 token 相同**，与草稿好坏无关。"""
    tgt = ToyMarkov(V, seed=V, temp=1.0)
    drf = perturb(tgt, eps=eps, seed=V + 7)
    assert greedy_generate(tgt, 0, 60) == greedy_speculative(tgt, drf, 0, 60, gamma)


def test_greedy_equivalence_holds_even_when_argmax_disagrees():
    """草稿的 argmax 与目标大量不一致时，L2 依然成立（只是变慢）。"""
    tgt = ToyMarkov(8, seed=8, temp=1.0)
    drf = perturb(tgt, eps=8.0, seed=15)
    agree = float(np.mean(tgt.T.argmax(1) == drf.T.argmax(1)))
    assert agree < 0.8, "这个用例要求 argmax 确实有不少分歧"
    assert greedy_generate(tgt, 0, 80) == greedy_speculative(tgt, drf, 0, 80, 4)


def test_onehot_model_is_deterministic():
    m = onehot_model(ToyMarkov(6, seed=3, temp=1.0))
    assert np.allclose(m.T.sum(axis=1), 1.0)
    assert set(np.unique(m.T)) <= {0.0, 1.0}


# ---------------------------------------------------------------- L3 近似判据

def test_typical_acceptance_is_biased():
    """**L3**：绝对阈值判据在任何阈值下都有偏（分布不等于 p）。"""
    tgt = ToyMarkov(5, seed=0, temp=1.0)
    drf = perturb(tgt, eps=1.2, seed=1)
    p = tgt.dist(0)
    for eps in (0.0, 0.05, 0.2, 0.5):
        d = typical_accept_first_token(tgt, drf, 0, 3, eps)
        assert float(np.abs(d - p).max()) > 1e-3, eps


def test_zero_threshold_reproduces_draft_distribution():
    """eps=0（无条件接受）时输出分布**就是草稿分布 q** —— 偏差的上界很直观。"""
    tgt = ToyMarkov(5, seed=0, temp=1.0)
    drf = perturb(tgt, eps=1.2, seed=1)
    d = typical_accept_first_token(tgt, drf, 0, 3, 0.0)
    assert np.abs(d - drf.dist(0)).max() < 1e-12


def test_raising_threshold_makes_bias_worse_not_better():
    """**反直觉但可复现**：把阈值调保守，偏差反而变大。

    机制：拒绝掉的质量走 norm(max(0,p-q)) 这条为 min(1,p/q) 配套设计的补偿路径，
    换了判据补偿就不匹配；拒绝越多，走错路的质量越多。
    => 判据与补偿必须成对更换，只改判据是修不好的。
    """
    tgt = ToyMarkov(5, seed=0, temp=1.0)
    drf = perturb(tgt, eps=1.2, seed=1)
    p = tgt.dist(0)
    errs = [float(np.abs(typical_accept_first_token(tgt, drf, 0, 3, e) - p).max())
            for e in (0.05, 0.20, 0.50)]
    assert errs[0] < errs[1] < errs[2], errs
