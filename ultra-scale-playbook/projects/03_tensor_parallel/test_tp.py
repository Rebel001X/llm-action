"""
test_tp.py —— 单进程验证张量并行:并行前向/反向 == 单卡整体计算(逐元素相等)。
运行:python -m pytest -q
"""
import torch
import torch.nn.functional as F
import pytest
from tensor_parallel import (
    linear, split_cols_of_output, split_rows_of_input,
    column_parallel_forward, row_parallel_forward,
    reference_mlp, parallel_mlp, reference_attention, parallel_attention,
)

torch.set_default_dtype(torch.float64)


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


@pytest.mark.parametrize("p", [1, 2, 4])
def test_column_parallel_equals_full_linear(p):
    x = _rand(6, 16, seed=1)
    W = _rand(32, 16, seed=2)          # (out=32, in=16),32 可被 p 整除
    ref = linear(x, W)
    par = column_parallel_forward(x, split_cols_of_output(W, p))
    assert torch.allclose(ref, par, atol=1e-12)


@pytest.mark.parametrize("p", [1, 2, 4])
def test_row_parallel_equals_full_linear(p):
    x = _rand(6, 16, seed=1)
    W = _rand(8, 16, seed=2)           # in=16 可被 p 整除
    ref = linear(x, W)
    par = row_parallel_forward(x, split_rows_of_input(W, p))
    assert torch.allclose(ref, par, atol=1e-12)


@pytest.mark.parametrize("p", [1, 2, 4])
def test_parallel_mlp_forward_matches(p):
    h = 16
    x = _rand(5, h, seed=1)
    W1 = _rand(4 * h, h, seed=2); b1 = _rand(4 * h, seed=3)
    W2 = _rand(h, 4 * h, seed=4); b2 = _rand(h, seed=5)
    ref = reference_mlp(x, W1, b1, W2, b2)
    par = parallel_mlp(x, W1, b1, W2, b2, p)
    assert torch.allclose(ref, par, atol=1e-11)


@pytest.mark.parametrize("p", [2, 4])
def test_parallel_mlp_backward_matches(p):
    h = 16
    W1 = _rand(4 * h, h, seed=2); b1 = _rand(4 * h, seed=3)
    W2 = _rand(h, 4 * h, seed=4); b2 = _rand(h, seed=5)

    x1 = _rand(5, h, seed=1).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    reference_mlp(x1, W1, b1, W2, b2).sum().backward()
    parallel_mlp(x2, W1, b1, W2, b2, p).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-10)


@pytest.mark.parametrize("p", [1, 2, 4])
def test_parallel_attention_matches(p):
    B, S, H, nh = 2, 7, 16, 4
    x = _rand(B, S, H, seed=1)
    Wq = _rand(H, H, seed=2); Wk = _rand(H, H, seed=3)
    Wv = _rand(H, H, seed=4); Wo = _rand(H, H, seed=5)
    ref = reference_attention(x, Wq, Wk, Wv, Wo, nh)
    par = parallel_attention(x, Wq, Wk, Wv, Wo, nh, p)
    assert torch.allclose(ref, par, atol=1e-11)


def test_attention_backward_matches():
    B, S, H, nh, p = 2, 6, 16, 4, 2
    Wq = _rand(H, H, seed=2); Wk = _rand(H, H, seed=3)
    Wv = _rand(H, H, seed=4); Wo = _rand(H, H, seed=5)
    x1 = _rand(B, S, H, seed=1).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    reference_attention(x1, Wq, Wk, Wv, Wo, nh).sum().backward()
    parallel_attention(x2, Wq, Wk, Wv, Wo, nh, p).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-10)
