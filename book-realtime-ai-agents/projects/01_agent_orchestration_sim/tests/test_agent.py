# -*- coding: utf-8 -*-
"""测试 Agent 编排循环：路由正确、终止性（不死循环）、反思提升成功率、步数上限。"""

import pytest

from agent import OrchestrationAgent, Planner, Reflector
from tools import build_default_registry, ToolRegistry, CalculatorTool, ToolResult
from tasks import Task, build_task_suite


def _agent(max_steps=6, max_reflections=2):
    return OrchestrationAgent(build_default_registry(),
                              max_steps=max_steps, max_reflections=max_reflections)


# ---------------- 工具路由正确性 ---------------- #
def test_calc_task_routes_to_calculator():
    ep = _agent().run(Task("c", "算 2+2", "calc",
                           payload={"expression": "2 + 2"}, expected=4.0,
                           golden_tools=["calculator"]))
    assert ep.success
    assert ep.answer == 4.0
    assert ep.used_tools == ["calculator"]      # 路由到了正确的工具
    assert ep.terminated_reason == "solved"


def test_lookup_task_routes_to_retriever():
    ep = _agent().run(Task("l", "光速?", "lookup",
                           payload={"query": "光速"},
                           expected="真空中光速约为 299792458 米/秒。",
                           golden_tools=["retriever"]))
    assert ep.success
    assert ep.used_tools == ["retriever"]


def test_tool_trajectory_matches_golden_for_suite():
    """整套任务里，成功任务的工具轨迹应与 golden 一致。"""
    agent = _agent()
    for task in build_task_suite():
        ep = agent.run(task)
        if ep.success:
            assert ep.used_tools == task.golden_tools, \
                f"任务 {task.tid} 轨迹不符：{ep.used_tools} != {task.golden_tools}"


# ---------------- 终止性：绝不死循环 ---------------- #
def test_unknown_task_terminates_without_infinite_loop():
    """无解任务必须安全终止，不能无限转。"""
    ep = _agent(max_steps=6, max_reflections=2).run(
        Task("u", "查不到的词", "unknown",
             payload={"query": "反重力引擎"}, expected=None,
             golden_tools=["retriever"]))
    assert not ep.success
    assert ep.terminated_reason in ("dead_end", "max_reflections", "max_steps")
    assert ep.steps <= 6


def test_every_task_respects_max_steps():
    """任何任务的执行步数都不超过 max_steps（硬护栏）。"""
    cap = 4
    agent = _agent(max_steps=cap, max_reflections=10)
    for task in build_task_suite():
        ep = agent.run(task)
        assert ep.steps <= cap, f"{task.tid} 步数 {ep.steps} 超过上限 {cap}"


def test_flaky_step_bounded_even_with_many_reflections():
    """flaky 任务即便给很多反思次数，也受 max_steps 约束，不会爆炸。"""
    agent = _agent(max_steps=3, max_reflections=99)
    ep = agent.run(Task("f", "flaky", "flaky",
                        payload={"payload": "x"}, expected="mock-成功结果",
                        golden_tools=["flaky", "flaky"]))
    assert ep.steps <= 3


# ---------------- 反思提升成功率 ---------------- #
def test_reflection_makes_flaky_succeed():
    """有反思时，先失败一次的 flaky 任务最终成功。"""
    ep = _agent(max_reflections=2).run(
        Task("f", "flaky", "flaky",
             payload={"payload": "x"}, expected="mock-成功结果",
             golden_tools=["flaky", "flaky"]))
    assert ep.success
    assert ep.reflections >= 1                 # 确实反思了
    assert ep.used_tools == ["flaky", "flaky"] # 调了两次 flaky


def test_no_reflection_makes_flaky_fail():
    """关掉反思（max_reflections=0），flaky 任务第一次失败后无法翻盘。"""
    ep = _agent(max_reflections=0).run(
        Task("f", "flaky", "flaky",
             payload={"payload": "x"}, expected="mock-成功结果",
             golden_tools=["flaky", "flaky"]))
    assert not ep.success
    assert ep.reflections == 0


# ---------------- 反思器 / 规划器单元 ---------------- #
def test_reflector_response_match_float():
    r = Reflector()
    task = Task("x", "", "calc", expected=84.0)
    solved, retry = r.reflect(task, ToolResult(ok=True, output=84.0))
    assert solved and not retry
    # 数值不匹配 -> 未解决
    solved2, _ = r.reflect(task, ToolResult(ok=True, output=83.0))
    assert not solved2


def test_reflector_unknown_does_not_retry():
    """unknown 任务失败后不应重试（否则会推向死循环）。"""
    r = Reflector()
    task = Task("x", "", "unknown", expected=None)
    solved, retry = r.reflect(task, ToolResult(ok=False, error="miss"))
    assert not solved and not retry


def test_planner_gives_up_on_unknown_when_reflecting():
    """规划器在反思态面对 unknown 会主动放弃（tool_name=None）。"""
    a = Planner().plan(Task("x", "", "unknown", payload={"query": "q"}),
                       last_result=ToolResult(ok=False), reflecting=True)
    assert a.tool_name is None


def test_route_miss_is_handled_as_failed_result():
    """若规划器给出不存在的工具名，Agent 应把它当失败结果处理而非崩溃。"""
    reg = build_default_registry()
    agent = OrchestrationAgent(reg, max_steps=3, max_reflections=1)

    # 用一个会返回幻觉工具名的规划器替身
    class HallucinatingPlanner(Planner):
        def plan(self, task, last_result, reflecting):
            from agent import Action
            return Action("不存在的工具", {}, thought="幻觉")
    agent.planner = HallucinatingPlanner()

    ep = agent.run(Task("h", "", "calc", expected=1.0, golden_tools=[]))
    assert not ep.success            # 没崩，只是失败
    assert ep.used_tools == []       # 幻觉工具不计入实际调用
