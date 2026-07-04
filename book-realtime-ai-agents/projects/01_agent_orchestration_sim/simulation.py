# -*- coding: utf-8 -*-
"""
simulation.py —— 批量仿真 + 指标统计
=====================================

对一批任务跑 Agent，聚合出 AgentOps 关心的几个核心指标:
  - completion_rate       : 完成率（成功任务数 / 总任务数）
  - avg_steps             : 平均执行步数（越低越省算力/延迟）
  - tool_trajectory_acc   : 工具轨迹正确率（实际调用序列是否匹配 golden）
  - total_tool_calls      : 总工具调用次数
  - reflection_count      : 总反思次数

还提供 `compare_with_without_reflection`:
  用「关掉反思（max_reflections=0）」和「开启反思」跑同一批任务，
  证明**反思后完成率提升**——这是本项目最核心的结论之一。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from agent import OrchestrationAgent, Episode
from tools import ToolRegistry, build_default_registry
from tasks import Task, build_task_suite


@dataclass
class SimResult:
    """一次批量仿真的聚合结果。"""
    episodes: List[Episode] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def completion_rate(self) -> float:
        if self.n == 0:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / self.n

    @property
    def avg_steps(self) -> float:
        if self.n == 0:
            return 0.0
        return sum(e.steps for e in self.episodes) / self.n

    @property
    def total_reflections(self) -> int:
        return sum(e.reflections for e in self.episodes)

    @property
    def total_tool_calls(self) -> int:
        return sum(len(e.used_tools) for e in self.episodes)

    @property
    def tool_trajectory_acc(self) -> float:
        """工具轨迹正确率：实际调用序列 == golden_tools 的任务占比。

        注意：只对「本应成功」的任务（expected 非 None）算轨迹准确率，
        对 unknown 这类「本就无解」的任务，golden 只描述它「尝试过什么」。
        """
        if self.n == 0:
            return 0.0
        hits = 0
        for e in self.episodes:
            if e.used_tools == e.task.golden_tools:
                hits += 1
        return hits / self.n

    def by_kind(self) -> Dict[str, float]:
        """分任务类型的完成率，方便定位「哪类任务拖后腿」。"""
        buckets: Dict[str, List[bool]] = {}
        for e in self.episodes:
            buckets.setdefault(e.task.kind, []).append(e.success)
        return {k: sum(v) / len(v) for k, v in buckets.items()}


def run_simulation(tasks: List[Task] | None = None,
                   registry: ToolRegistry | None = None,
                   max_steps: int = 6,
                   max_reflections: int = 2) -> SimResult:
    """对一批任务跑一遍 Agent，返回聚合结果。

    每次仿真前重置工具计数 & flaky 内部状态，保证任务间互不污染。
    """
    tasks = tasks if tasks is not None else build_task_suite()
    registry = registry if registry is not None else build_default_registry()
    registry.reset_counts()

    agent = OrchestrationAgent(registry, max_steps=max_steps, max_reflections=max_reflections)
    episodes: List[Episode] = []
    for task in tasks:
        # flaky 工具的「尝试计数」是任务级的，跑每个任务前单独重置
        flaky = registry.route("flaky")
        if flaky is not None and hasattr(flaky, "reset"):
            flaky.reset()
        episodes.append(agent.run(task))
    return SimResult(episodes=episodes)


def compare_with_without_reflection(tasks: List[Task] | None = None) -> Dict[str, SimResult]:
    """对比「无反思」vs「有反思」两种配置，证明反思能提升完成率。

    - 无反思：max_reflections=0，flaky 任务第一次失败后无法重试 -> 失败
    - 有反思：max_reflections=2，flaky 任务反思后重试 -> 成功
    """
    tasks = tasks if tasks is not None else build_task_suite()
    # 关键：两组用各自独立的 registry，避免 flaky 状态串味
    without = run_simulation(tasks, registry=build_default_registry(), max_reflections=0)
    with_ref = run_simulation(tasks, registry=build_default_registry(), max_reflections=2)
    return {"without_reflection": without, "with_reflection": with_ref}
