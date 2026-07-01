"""
test_zero_dp.py —— 单进程验证 DDP / ZeRO 的正确性(不依赖多进程,稳过)。
金标准:分布式逻辑的结果 == 单卡在全 batch 上做一步 Adam(逐元素相等)。
运行:python -m pytest -q
"""
import torch
from zero_dp import (
    make_model, make_batch, compute_grads, flatten, unflatten,
    adam_step, shard_ranges, ddp_average_grads, zero3_step, init_zero_states,
)

torch.set_default_dtype(torch.float64)   # 双精度让"逐元素相等"断言更干净


def test_shard_ranges_cover_and_disjoint():
    for n, w in [(10, 3), (8, 4), (7, 4), (5, 8)]:
        rs = shard_ranges(n, w)
        assert rs[0][0] == 0 and rs[-1][1] == n
        for (a1, b1), (a2, b2) in zip(rs, rs[1:]):
            assert b1 == a2              # 连续无缝
        assert sum(b - a for a, b in rs) == n   # 覆盖全部,无重叠


def test_ddp_avg_grads_equals_full_batch_grad():
    """把全 batch 切成 world 份分别算梯度再平均 == 全 batch 一次算的梯度。"""
    world = 4
    model = make_model()
    x, y = make_batch(world * 3)              # 12 条,可被 4 整除
    full_g, _ = compute_grads(model, x, y)
    shards = x.chunk(world), y.chunk(world)
    per_rank = [compute_grads(model, xs, ys)[0] for xs, ys in zip(*shards)]
    avg = ddp_average_grads(per_rank)
    for a, f in zip(avg, full_g):
        assert torch.allclose(a, f, atol=1e-10)


def test_zero3_single_step_equals_reference_adam():
    """ZeRO-3 分片一步 == 单卡整体一步 Adam(逐元素相等)。"""
    model = make_model()
    x, y = make_batch(8)
    grads, _ = compute_grads(model, x, y)
    flat_p = flatten([p.detach() for p in model.parameters()])
    flat_g = flatten(grads)
    n = flat_p.numel()

    # 参考:单卡整体 Adam
    m_ref, v_ref = torch.zeros(n), torch.zeros(n)
    p_ref, m_ref, v_ref = adam_step(flat_p.clone(), flat_g, m_ref, v_ref, t=1)

    # ZeRO-3:分 4 片,各自更新再拼回
    world = 4
    states = init_zero_states(n, world)
    p_zero = zero3_step(flat_p.clone(), flat_g, states, t=1, world_size=world)

    assert torch.allclose(p_ref, p_zero, atol=1e-12)


def test_zero3_multi_step_matches_reference():
    """连续多步:ZeRO-3(dp=4)训练轨迹 == 单卡 Adam 轨迹。"""
    world, steps = 4, 15
    model = make_model()
    x, y = make_batch(world * 2)

    like = [p for p in model.parameters()]
    flat_ref = flatten([p.detach().clone() for p in like])
    flat_zero = flat_ref.clone()
    n = flat_ref.numel()
    m_ref, v_ref = torch.zeros(n), torch.zeros(n)
    states = init_zero_states(n, world)

    for t in range(1, steps + 1):
        # 用当前(参考)参数灌回模型算 DDP 平均梯度
        for p, chunk in zip(model.parameters(), unflatten(flat_ref, like)):
            p.data.copy_(chunk)
        shards = x.chunk(world), y.chunk(world)
        per_rank = [compute_grads(model, xs, ys)[0] for xs, ys in zip(*shards)]
        flat_g = flatten(ddp_average_grads(per_rank))

        flat_ref, m_ref, v_ref = adam_step(flat_ref, flat_g, m_ref, v_ref, t)
        flat_zero = zero3_step(flat_zero, flat_g, states, t, world)
        assert torch.allclose(flat_ref, flat_zero, atol=1e-10), f"step {t} 偏离"


def test_zero_stages_store_less_but_same_result():
    """ZeRO-1/2/3 结果相同(数学不变),差别只在"存了多少"(此处验证结果一致)。"""
    model = make_model()
    x, y = make_batch(8)
    grads, _ = compute_grads(model, x, y)
    flat_p = flatten([p.detach() for p in model.parameters()])
    flat_g = flatten(grads)
    n = flat_p.numel()
    # 无论分几片,逐元素 Adam 结果一致
    for world in [1, 2, 3, 8]:
        states = init_zero_states(n, world)
        p_new = zero3_step(flat_p.clone(), flat_g, states, t=1, world_size=world)
        if world == 1:
            ref = p_new
        else:
            assert torch.allclose(ref, p_new, atol=1e-12)
