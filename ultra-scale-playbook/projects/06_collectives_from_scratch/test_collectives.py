"""
test_collectives.py —— 单进程验证从零实现的集合通信:结果 == 手工求和/拼接。
运行:python -m pytest -q
"""
import torch
import pytest
from collectives import (
    reference_sum, ring_allreduce_sim, ring_broadcast_sim,
    all_gather_sim, reduce_scatter_sim, allreduce_bytes_per_rank,
)

torch.set_default_dtype(torch.float64)


def _tensors(P, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N, generator=g) for _ in range(P)]


@pytest.mark.parametrize("P", [1, 2, 3, 4, 6, 8])
def test_ring_allreduce_equals_sum(P):
    N = 24                                          # 可被各 P 整除
    ts = _tensors(P, N)
    ref = reference_sum(ts)
    outs = ring_allreduce_sim(ts)
    assert len(outs) == P
    for o in outs:                                  # 每个 rank 都拿到完整求和
        assert torch.allclose(o, ref, atol=1e-10)


def test_ring_allreduce_all_ranks_identical():
    ts = _tensors(4, 12)
    outs = ring_allreduce_sim(ts)
    for o in outs[1:]:
        assert torch.allclose(o, outs[0], atol=1e-12)


def test_reduce_scatter_then_all_gather_equals_allreduce():
    """ReduceScatter + AllGather 应等价于 AllReduce。"""
    ts = _tensors(4, 12)
    rs = reduce_scatter_sim(ts)                      # 每 rank 拿到求和的一段
    gathered = torch.cat(rs)                         # 拼起来 = 完整求和
    assert torch.allclose(gathered, reference_sum(ts), atol=1e-12)


def test_all_gather_concatenates():
    ts = _tensors(3, 8)
    outs = all_gather_sim(ts)
    expected = torch.cat(ts)
    for o in outs:
        assert torch.allclose(o, expected, atol=1e-12)


def test_broadcast_copies_source():
    ts = _tensors(4, 8)
    outs = ring_broadcast_sim(ts, src=2)
    for o in outs:
        assert torch.allclose(o, ts[2], atol=1e-12)


def test_allreduce_comm_volume_formula():
    # 每 rank 收发 2(P-1)/P·N,与 P 几乎无关(P 大时趋于 2N)
    assert allreduce_bytes_per_rank(1000, 2, 4) == pytest.approx(2 * 0.5 * 1000 * 4)
    big = allreduce_bytes_per_rank(1000, 128, 4)
    assert big < 2 * 1000 * 4                        # 恒 < 2N·bytes
    assert big > 1.9 * 1000 * 4                      # P 大时趋近 2N·bytes


def test_ring_allreduce_matches_naive_for_ints():
    # 换一组确定输入再验证一次(整数,结果精确)
    ts = [torch.arange(6.0) + i for i in range(3)]   # 长度 6,P=3
    outs = ring_allreduce_sim(ts)
    ref = reference_sum(ts)
    for o in outs:
        assert torch.allclose(o, ref, atol=1e-12)
