# -*- coding: utf-8 -*-
"""测试批量仿真与指标：完成率、平均步数、轨迹准确率、反思对比。"""

from simulation import (
    run_simulation,
    compare_with_without_reflection,
    SimResult,
)
from tasks import build_task_suite


def test_simulation_runs_all_tasks():
    sim = run_simulation()
    assert sim.n == len(build_task_suite())


def test_completion_rate_in_range():
    sim = run_simulation()
    assert 0.0 <= sim.completion_rate <= 1.0
    # 7 个任务里只有 1 个（unknown）注定失败，完成率应 >= 6/7
    assert sim.completion_rate >= 6 / 7 - 1e-9


def test_avg_steps_positive_and_bounded():
    sim = run_simulation(max_steps=6)
    assert sim.avg_steps > 0
    # 每个任务步数 <= max_steps，平均自然也 <=
    assert sim.avg_steps <= 6


def test_tool_trajectory_accuracy_high():
    """整套任务的工具轨迹准确率应很高（golden 与实际一致）。"""
    sim = run_simulation()
    assert sim.tool_trajectory_acc >= 6 / 7 - 1e-9


def test_reflection_improves_completion_rate():
    """核心结论：开启反思后完成率严格高于关闭反思。"""
    cmp = compare_with_without_reflection()
    without = cmp["without_reflection"].completion_rate
    with_ref = cmp["with_reflection"].completion_rate
    assert with_ref > without, f"反思未提升完成率: {with_ref} !> {without}"


def test_empty_simresult_defaults():
    """空结果的指标不应除零崩溃。"""
    s = SimResult(episodes=[])
    assert s.completion_rate == 0.0
    assert s.avg_steps == 0.0
    assert s.tool_trajectory_acc == 0.0


def test_by_kind_breakdown():
    sim = run_simulation()
    bk = sim.by_kind()
    assert set(bk.keys()) == {"calc", "lookup", "flaky", "unknown"}
    assert bk["calc"] == 1.0        # 计算题应全对
    assert bk["unknown"] == 0.0     # unknown 应全错
