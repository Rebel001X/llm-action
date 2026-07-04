# -*- coding: utf-8 -*-
"""
agent.py —— Agent 编排核心：规划 → 执行（调工具） → 反思 循环
================================================================

这是整个项目的「大脑」。它把第 4 章反复强调的 Agent 骨架落成可跑代码:

    ┌─────────┐   ┌──────────────┐   ┌──────────┐
    │  规划   │──▶│  执行(调工具) │──▶│   反思   │
    │  Plan   │   │   Execute    │   │ Reflect  │
    └─────────┘   └──────────────┘   └────┬─────┘
         ▲                                 │
         └────────────  重试/换招  ────────┘

我们**刻意不接真 LLM**——用一个「确定性规划器」（rule-based planner）替代大模型。
原因（第一性原理）:
  1. 本机离线、无网络、无 key，仿真必须自给自足。
  2. 教学目标是「编排循环本身」，不是「模型有多聪明」。把 LLM 换成规则，
     循环的控制流（何时停、何时重试、步数上限）反而看得一清二楚。
  3. 可复现：同样输入永远同样输出，pytest 才能断言。
真实工程里，把 `Planner.plan()` 换成一次 LLM 调用即可，循环骨架原封不动。

关键安全阀（真实 Agent 的必备护栏 / guardrail）:
  - max_steps      : 步数上限，防「死循环烧钱」
  - max_reflections: 反思次数上限，防「无限反思」
  - 每步都记录一条 trace，最后能完整回放（这就是 AgentOps 里的 tracing）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from tools import ToolRegistry, ToolResult
from tasks import Task


# ------------------------------------------------------------------ #
# 1. 单步动作（Action）与轨迹记录（TraceStep）
# ------------------------------------------------------------------ #
@dataclass
class Action:
    """规划器产出的一步动作：调哪个工具、传什么参数。

    tool_name=None 表示「规划器认为该收尾了 / 无路可走」，用来触发终止。
    """
    tool_name: Optional[str]
    args: dict = field(default_factory=dict)
    thought: str = ""  # 规划时的「想法」，纯粹为了可解释性


@dataclass
class TraceStep:
    """一步执行的完整快照，用于事后回放（AgentOps 的 trace）。"""
    step: int
    phase: str            # "plan" / "execute" / "reflect"
    tool_name: Optional[str]
    args: dict
    thought: str
    result: Optional[ToolResult] = None
    note: str = ""


@dataclass
class Episode:
    """一个任务跑完后的完整结果（一集 episode）。"""
    task: Task
    success: bool
    answer: Any
    steps: int                       # 实际消耗的执行步数
    reflections: int                 # 反思了几次
    used_tools: List[str]            # 实际调用的工具轨迹
    trace: List[TraceStep] = field(default_factory=list)
    terminated_reason: str = ""      # 为什么停：solved / max_steps / dead_end


# ------------------------------------------------------------------ #
# 2. 规划器（Planner）—— LLM 的确定性替身
# ------------------------------------------------------------------ #
class Planner:
    """根据任务类型 + 上一步结果，产出下一步动作。

    这是「规划」阶段。真实系统里这里是一次 LLM 调用（带工具菜单的 prompt）。
    我们用 if/else 规则模拟，但**接口和真实规划器完全一致**:
        plan(task, last_result, reflecting) -> Action
    """

    def plan(self, task: Task, last_result: Optional[ToolResult], reflecting: bool) -> Action:
        kind = task.kind

        # 计算题：路由到 calculator
        if kind == "calc":
            return Action("calculator",
                          {"expression": task.payload.get("expression", "")},
                          thought="这是一道算术题，选用 calculator 工具")

        # 检索题：路由到 retriever
        if kind == "lookup":
            return Action("retriever",
                          {"query": task.payload.get("query", "")},
                          thought="这是一道事实查询题，选用 retriever 工具")

        # flaky 题：第一次正常调 flaky；如果在反思态（上一步失败）就「带重试意图」再调
        if kind == "flaky":
            thought = ("首次调用不稳定服务 flaky" if not reflecting
                       else "上一步失败，反思后决定重试 flaky（真实系统可加退避/换参）")
            return Action("flaky", {"payload": task.payload.get("payload")}, thought=thought)

        # unknown 题：只会去查 retriever；查不到就无路可走（返回 tool_name=None 触发终止）
        if kind == "unknown":
            if reflecting:
                # 反思后仍无解——规划器诚实地「认输」，避免死循环
                return Action(None, {}, thought="反思后仍找不到可用工具/答案，主动放弃以避免死循环")
            return Action("retriever",
                          {"query": task.payload.get("query", "")},
                          thought="尝试用 retriever 查询，但预期知识库中可能没有")

        # 兜底：未知类型直接放弃
        return Action(None, {}, thought=f"未知任务类型 {kind!r}，无法规划")


# ------------------------------------------------------------------ #
# 3. 反思器（Reflector）—— 判断「成功了没 / 要不要再来一次」
# ------------------------------------------------------------------ #
class Reflector:
    """反思阶段：看执行结果，决定 (是否已解决, 是否值得重试)。"""

    def reflect(self, task: Task, result: ToolResult) -> tuple[bool, bool]:
        """返回 (solved, should_retry)。

        - solved=True      : 结果满足任务，收工
        - should_retry=True: 没成，但值得再试一次（比如 flaky 的临时故障）
        两者互斥优先：solved 优先。
        """
        if result.ok and self._is_answer_valid(task, result.output):
            return True, False  # 成功且答案有效 -> 解决

        # 失败分两类：
        # 1) 临时故障（工具返回 ok=False 但类型题本身可解）-> 值得重试
        # 2) 本质无解（比如知识库确实没有）-> 不值得重试
        if task.kind == "flaky":
            return False, True   # flaky 的失败是「抖动」，反思后重试
        if task.kind == "unknown":
            return False, False  # unknown 的失败是「无解」，不重试，避免死循环
        # calc/lookup 若失败（比如表达式非法/查不到），给一次重试机会
        return False, True

    @staticmethod
    def _is_answer_valid(task: Task, output: Any) -> bool:
        """response match：把工具输出和 golden 标准答案比对。"""
        if task.expected is None:
            return False
        if isinstance(task.expected, float):
            try:
                return abs(float(output) - task.expected) < 1e-9
            except (TypeError, ValueError):
                return False
        return output == task.expected


# ------------------------------------------------------------------ #
# 4. Agent —— 把规划/执行/反思编排成一个带护栏的循环
# ------------------------------------------------------------------ #
class OrchestrationAgent:
    """规划→执行→反思 的编排器。

    参数:
      registry        : 工具注册表（负责工具路由）
      max_steps       : 单任务最大执行步数（硬护栏，防死循环）
      max_reflections : 单任务最大反思次数（软护栏，控制重试预算）
    """

    def __init__(self, registry: ToolRegistry, max_steps: int = 6, max_reflections: int = 2) -> None:
        self.registry = registry
        self.max_steps = max_steps
        self.max_reflections = max_reflections
        self.planner = Planner()
        self.reflector = Reflector()

    def run(self, task: Task) -> Episode:
        trace: List[TraceStep] = []
        used_tools: List[str] = []
        reflections = 0
        reflecting = False
        last_result: Optional[ToolResult] = None
        success = False
        answer: Any = None
        reason = "max_steps"  # 默认终止原因；被真正解决/放弃时会覆盖

        # 主循环：每一轮 = 规划 → 执行 → 反思。step 是硬上限，绝不会无限转。
        for step in range(1, self.max_steps + 1):
            # ---------- 规划 ----------
            action = self.planner.plan(task, last_result, reflecting)
            trace.append(TraceStep(step, "plan", action.tool_name, action.args,
                                   action.thought))

            # 规划器主动放弃（tool_name=None）-> 死路，安全终止
            if action.tool_name is None:
                reason = "dead_end"
                break

            # ---------- 执行（工具路由 + 调用）----------
            tool = self.registry.route(action.tool_name)
            if tool is None:
                # 路由失败：Agent「幻觉」了一个不存在的工具。记为失败结果，走反思。
                result = ToolResult(ok=False,
                                    error=f"工具路由失败：不存在的工具 {action.tool_name!r}",
                                    tool_name=action.tool_name)
            else:
                result = tool(**action.args)
                used_tools.append(action.tool_name)
            last_result = result
            trace.append(TraceStep(step, "execute", action.tool_name, action.args,
                                   action.thought, result=result))

            # ---------- 反思 ----------
            solved, should_retry = self.reflector.reflect(task, result)
            note = f"solved={solved}, should_retry={should_retry}"
            trace.append(TraceStep(step, "reflect", action.tool_name, action.args,
                                   "评估执行结果", result=result, note=note))

            if solved:
                success = True
                answer = result.output
                reason = "solved"
                break

            # 没解决：决定是否进入下一轮反思重试
            if should_retry and reflections < self.max_reflections:
                reflections += 1
                reflecting = True
                continue  # 回到循环顶，规划器会带着「重试意图」再规划
            else:
                # 不值得重试，或反思预算耗尽 -> 终止
                reason = "dead_end" if not should_retry else "max_reflections"
                break

        return Episode(
            task=task,
            success=success,
            answer=answer,
            steps=len([s for s in trace if s.phase == "execute"]),
            reflections=reflections,
            used_tools=used_tools,
            trace=trace,
            terminated_reason=reason,
        )
