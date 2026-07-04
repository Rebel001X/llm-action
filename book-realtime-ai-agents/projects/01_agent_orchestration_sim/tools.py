# -*- coding: utf-8 -*-
"""
tools.py —— 可插拔的「离线工具」集合（offline pluggable tools）
=================================================================

对应《Multimodal Real-Time AI Agent Systems》第 4 章「工具即 Python 函数」的思想：
Agent 的「技能」就是一组普通的 Python 函数。本文件把这个思想再抽象一层，
定义一个统一的工具协议（Tool Protocol），让工具**可插拔**（plug-and-play）：
新增一个工具，只要实现 `Tool` 接口并注册进 `ToolRegistry`，Agent 立刻就能用。

设计要点（第一性原理）:
  1. 每个工具都是一个「纯离线」函数——不联网、不调模型、无副作用（除计数）。
     这样仿真才可复现（reproducible），pytest 才能断言确定的结果。
  2. 工具的返回统一为 `ToolResult`，把「成功/失败」显式建模。
     真实 Agent 里工具会超时、会报错，反思（reflection）循环正是靠这个信号触发的。
  3. `ToolRegistry` 就是「工具路由（tool routing）」的物理载体：
     给一个工具名 -> 找到对应的可调用对象。路由错了，整个 Agent 就废了。

本项目内置 3 个工具:
  - calculator : 安全地算一个算术表达式（AST 白名单，绝不 eval 任意代码）
  - retriever  : 在一个本地小知识库里做关键词检索（mock RAG，不联网）
  - mock       : 一个「会按脚本失败再成功」的工具，用来演示「失败→反思→重试」
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ------------------------------------------------------------------ #
# 1. 工具的统一返回类型
# ------------------------------------------------------------------ #
@dataclass
class ToolResult:
    """工具调用的统一返回。

    把「成功与否」显式建模，是让 Agent 能「反思」的前提：
    Agent 看到 ok=False，才知道该重试 / 换工具 / 换参数。
    """
    ok: bool                      # 调用是否成功
    output: Any = None            # 成功时的输出（数字、文本、列表……）
    error: str = ""               # 失败时的错误信息（人类可读）
    tool_name: str = ""           # 是哪个工具产生的（便于追踪）

    def __repr__(self) -> str:  # 让轨迹打印更好看
        if self.ok:
            return f"ToolResult(ok=True, tool={self.tool_name!r}, output={self.output!r})"
        return f"ToolResult(ok=False, tool={self.tool_name!r}, error={self.error!r})"


# ------------------------------------------------------------------ #
# 2. 工具基类（Tool Protocol）
# ------------------------------------------------------------------ #
class Tool:
    """所有工具的基类。子类只需实现 `run(**kwargs) -> ToolResult`。

    - name        : 工具唯一名字（路由的 key）
    - description : 给「规划器」看的说明，规划时靠它来选工具
    - call_count  : 累计被调用次数（做「工具调用统计」用）
    """

    name: str = "base"
    description: str = "抽象工具基类，不可直接使用"

    def __init__(self) -> None:
        self.call_count: int = 0

    def __call__(self, **kwargs: Any) -> ToolResult:
        # 统一入口：先计数，再执行，异常统一兜底成 ToolResult(ok=False)
        self.call_count += 1
        try:
            result = self.run(**kwargs)
            # 保证子类忘了填 tool_name 时也能追踪
            if not result.tool_name:
                result.tool_name = self.name
            return result
        except Exception as exc:  # noqa: BLE001  故意宽泛：工具绝不能把整个 Agent 拖崩
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}", tool_name=self.name)

    def run(self, **kwargs: Any) -> ToolResult:  # pragma: no cover - 抽象方法
        raise NotImplementedError


# ------------------------------------------------------------------ #
# 3. 计算器工具 —— 安全 AST 求值（绝不 eval 任意代码）
# ------------------------------------------------------------------ #
# 白名单：只允许这些二元 / 一元运算，杜绝 __import__ / 属性访问等注入
_BIN_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """递归地、只按白名单求值一棵 AST。遇到不认识的节点就抛错。

    这是「计算器为什么不能用 eval」的标准答案：
    eval("__import__('os').system('rm -rf /')") 会真的执行；
    而我们只放行数字和四则运算节点，注入无从下手。
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    # Python 3.8+ 用 ast.Constant 表示数字字面量
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ValueError(f"不支持的常量: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"不支持的二元运算符: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
        return _UNARY_OPS[op_type](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


class CalculatorTool(Tool):
    """计算器：把一个算术表达式字符串安全求值成数字。"""

    name = "calculator"
    description = "计算一个算术表达式（支持 + - * / // % ** 与括号），入参 expression=字符串"

    def run(self, expression: str = "", **_: Any) -> ToolResult:
        if not isinstance(expression, str) or not expression.strip():
            return ToolResult(ok=False, error="expression 必须是非空字符串")
        try:
            tree = ast.parse(expression, mode="eval")
            value = _safe_eval(tree)
        except ZeroDivisionError:
            return ToolResult(ok=False, error="除零错误（division by zero）")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, error=f"表达式非法: {exc}")
        return ToolResult(ok=True, output=value)


# ------------------------------------------------------------------ #
# 4. 检索工具 —— 本地 mock RAG（关键词命中，不联网）
# ------------------------------------------------------------------ #
# 一个「玩具知识库」：真实系统里这里是向量库 / ES；仿真里用内存字典即可
_KNOWLEDGE_BASE: Dict[str, str] = {
    "光速": "真空中光速约为 299792458 米/秒。",
    "地球半径": "地球平均半径约为 6371 千米。",
    "重力加速度": "地表重力加速度约为 9.8 米/秒²。",
    "水的沸点": "标准大气压下水的沸点为 100 摄氏度。",
    "圆周率": "圆周率 π 约等于 3.14159。",
    "普朗克常数": "普朗克常数约为 6.626e-34 焦耳·秒。",
}


class RetrieverTool(Tool):
    """检索器：在本地知识库里按关键词找答案（mock RAG）。"""

    name = "retriever"
    description = "在本地知识库检索一个关键词，入参 query=关键词字符串"

    def __init__(self, knowledge_base: Optional[Dict[str, str]] = None) -> None:
        super().__init__()
        # 允许注入自定义知识库，方便测试
        self.kb: Dict[str, str] = dict(knowledge_base or _KNOWLEDGE_BASE)

    def run(self, query: str = "", **_: Any) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            return ToolResult(ok=False, error="query 必须是非空字符串")
        q = query.strip()
        # 先精确命中，再做「子串包含」的模糊命中
        if q in self.kb:
            return ToolResult(ok=True, output=self.kb[q])
        for key, val in self.kb.items():
            if key in q or q in key:
                return ToolResult(ok=True, output=val)
        return ToolResult(ok=False, error=f"知识库中未找到与 {q!r} 相关的内容")


# ------------------------------------------------------------------ #
# 5. Mock 工具 —— 「按脚本先失败后成功」，用于演示反思重试
# ------------------------------------------------------------------ #
class FlakyMockTool(Tool):
    """会「先失败 N 次、之后才成功」的 mock 工具。

    为什么要它？——为了在**确定性**仿真里复现「工具会抖动」这一现实：
    第 1 次调用失败（比如超时、限流），Agent 反思后带着「重试」意图再调，
    到第 fail_times+1 次就成功。这样我们能精确测「反思后成功率提升」。
    """

    name = "flaky"
    description = "一个会先失败若干次再成功的 mock 工具，入参 payload=任意"

    def __init__(self, fail_times: int = 1, success_output: str = "mock-成功结果") -> None:
        super().__init__()
        self.fail_times = fail_times            # 前几次注定失败
        self.success_output = success_output    # 成功时返回什么
        self._attempts = 0                      # 内部尝试计数（独立于 call_count）

    def reset(self) -> None:
        """重置尝试计数（新任务开始时调用，保证任务间独立）。"""
        self._attempts = 0

    def run(self, payload: Any = None, **_: Any) -> ToolResult:
        self._attempts += 1
        if self._attempts <= self.fail_times:
            return ToolResult(
                ok=False,
                error=f"临时故障（第 {self._attempts} 次尝试，注定失败 {self.fail_times} 次）",
            )
        return ToolResult(ok=True, output=self.success_output)


# ------------------------------------------------------------------ #
# 6. 工具注册表 —— 「工具路由」的物理载体
# ------------------------------------------------------------------ #
@dataclass
class ToolRegistry:
    """工具注册表：名字 -> 工具实例。这就是「工具路由」的核心数据结构。

    route(name) 做的事，等价于真实 Agent 里 LLM 输出一个 tool_name 后，
    运行时（runtime）去查「这个名字到底对应哪个可执行函数」。
    路由不到，就是一次「工具幻觉（tool hallucination）」——Agent 编了个不存在的工具。
    """

    _tools: Dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> "ToolRegistry":
        if tool.name in self._tools:
            raise ValueError(f"工具名冲突: {tool.name!r} 已注册")
        self._tools[tool.name] = tool
        return self  # 支持链式注册

    def route(self, name: str) -> Optional[Tool]:
        """按名字路由到工具；找不到返回 None（而不是抛错，交给 Agent 反思）。"""
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def describe(self) -> str:
        """给规划器看的「工具菜单」文本。"""
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def total_calls(self) -> int:
        return sum(t.call_count for t in self._tools.values())

    def reset_counts(self) -> None:
        for t in self._tools.values():
            t.call_count = 0
            if isinstance(t, FlakyMockTool):
                t.reset()


def build_default_registry() -> ToolRegistry:
    """构造本项目默认的工具注册表（计算器 + 检索 + flaky mock）。"""
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(RetrieverTool())
    reg.register(FlakyMockTool(fail_times=1))
    return reg
