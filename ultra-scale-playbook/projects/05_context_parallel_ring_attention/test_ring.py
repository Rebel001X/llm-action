"""
test_ring.py —— 单进程验证 Ring Attention:在线 softmax 累加 == 标准全注意力(含因果)。
运行:python -m pytest -q
"""
import math
import torch
import pytest
from ring_attention import (
    full_attention, online_update, ring_attention, zigzag_indices,
)

torch.set_default_dtype(torch.float64)


def _qkv(s=12, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(s, d, generator=g),
            torch.randn(s, d, generator=g),
            torch.randn(s, d, generator=g))


@pytest.mark.parametrize("num_ranks", [1, 2, 3, 4, 6])
def test_ring_noncausal_equals_full(num_ranks):
    Q, K, V = _qkv(12, 8)
    ref = full_attention(Q, K, V, causal=False)
    out = ring_attention(Q, K, V, num_ranks, causal=False)
    assert torch.allclose(ref, out, atol=1e-10)


@pytest.mark.parametrize("num_ranks", [1, 2, 3, 4, 6])
def test_ring_causal_equals_full(num_ranks):
    Q, K, V = _qkv(12, 8)
    ref = full_attention(Q, K, V, causal=True)
    out = ring_attention(Q, K, V, num_ranks, causal=True)
    assert torch.allclose(ref, out, atol=1e-10)


def test_online_softmax_two_blocks_matches_full():
    """把 K/V 切成两块,在线累加 == 一次性 softmax。"""
    Q, K, V = _qkv(6, 8)
    d = Q.shape[-1]
    K1, K2 = K.split(3, dim=0)
    V1, V2 = V.split(3, dim=0)
    m = torch.full((6,), float("-inf")); l = torch.zeros(6); O = torch.zeros(6, 8)
    for Kb, Vb in [(K1, V1), (K2, V2)]:
        S = Q @ Kb.transpose(-1, -2) / math.sqrt(d)
        m, l, O = online_update(m, l, O, S, Vb)
    out = O / l.unsqueeze(-1)
    ref = full_attention(Q, K, V, causal=False)
    assert torch.allclose(out, ref, atol=1e-12)


def test_ring_independent_of_num_ranks():
    Q, K, V = _qkv(12, 8)
    a = ring_attention(Q, K, V, 2, causal=True)
    b = ring_attention(Q, K, V, 4, causal=True)
    assert torch.allclose(a, b, atol=1e-10)


def test_causal_first_token_only_attends_self():
    """因果下第 0 个 token 只能看到自己 → 输出应等于 V[0]。"""
    Q, K, V = _qkv(8, 8)
    out = ring_attention(Q, K, V, 4, causal=True)
    assert torch.allclose(out[0], V[0], atol=1e-10)


def test_zigzag_is_valid_permutation():
    s, P = 24, 3
    idx = zigzag_indices(s, P)
    assert idx.numel() == s
    assert set(idx.tolist()) == set(range(s))     # 是 0..s-1 的一个双射(合法重排)


def test_output_shape():
    Q, K, V = _qkv(12, 8)
    assert ring_attention(Q, K, V, 4).shape == (12, 8)
