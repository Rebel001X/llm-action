"""
test_pipeline.py —— 单进程验证流水线并行:累加梯度 == 单进程整 batch 梯度;气泡公式正确。
运行:python -m pytest -q
"""
import torch
import pytest
from pipeline import (
    build_stages, reference_grads, pipeline_grads,
    schedule_afab, schedule_1f1b, bubble_ratio,
)

torch.set_default_dtype(torch.float64)


def _data(n=12, dim=8, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g), torch.randn(n, dim, generator=g)


@pytest.mark.parametrize("num_micro", [1, 2, 3, 4, 6])
def test_pipeline_grads_equal_full_batch(num_micro):
    stages = build_stages(num_stages=4, dim=8)
    x, y = _data(12)
    ref = reference_grads(stages, x, y)
    pipe = pipeline_grads(stages, x, y, num_micro)
    for a, b in zip(ref, pipe):
        assert torch.allclose(a, b, atol=1e-10)


def test_pipeline_result_independent_of_micro_count():
    stages = build_stages(num_stages=3, dim=8)
    x, y = _data(12)
    g2 = pipeline_grads(stages, x, y, 2)
    g4 = pipeline_grads(stages, x, y, 4)
    for a, b in zip(g2, g4):
        assert torch.allclose(a, b, atol=1e-10)


def test_afab_event_counts():
    p, m = 4, 5
    ev = schedule_afab(p, m)
    fwd = [e for e in ev if e[1] == "F"]
    bwd = [e for e in ev if e[1] == "B"]
    assert len(fwd) == p * m and len(bwd) == p * m       # 每 stage 每 micro 一次前向一次反向


def test_1f1b_has_all_forwards_and_backwards():
    p, m = 4, 6
    ev = schedule_1f1b(p, m)
    fwd = [e for e in ev if e[0].endswith("F")]
    bwd = [e for e in ev if e[0].endswith("B")]
    assert len(fwd) == m and len(bwd) == m


def test_bubble_formula_and_monotonic():
    assert bubble_ratio(4, 1) == pytest.approx(3 / 4)        # (p-1)/(m+p-1)=3/4
    assert bubble_ratio(4, 8) == pytest.approx(3 / 11)
    # micro-batch 越多气泡越小
    ratios = [bubble_ratio(4, m) for m in [1, 2, 4, 8, 16]]
    assert all(ratios[i] > ratios[i + 1] for i in range(len(ratios) - 1))


def test_bubble_goes_to_zero_with_many_micro():
    assert bubble_ratio(8, 1000) < 0.01
