# -*- coding: utf-8 -*-
"""测试工具层：计算器安全性、检索命中、flaky 脚本、工具路由正确性。"""

import pytest

from tools import (
    CalculatorTool,
    RetrieverTool,
    FlakyMockTool,
    ToolRegistry,
    build_default_registry,
)


# ----------------- 计算器 ----------------- #
def test_calculator_basic():
    calc = CalculatorTool()
    r = calc(expression="12 * (3 + 4)")
    assert r.ok
    assert r.output == 84.0
    assert r.tool_name == "calculator"


def test_calculator_division_and_power():
    calc = CalculatorTool()
    assert calc(expression="(100 - 58) / 6").output == 7.0
    assert calc(expression="2 ** 10").output == 1024.0


def test_calculator_rejects_injection():
    """安全护栏：绝不能执行任意代码，只放行算术。"""
    calc = CalculatorTool()
    # 属性访问 / 函数调用 / import 都必须被拒
    for evil in ["__import__('os')", "().__class__", "print(1)", "a + 1"]:
        r = calc(expression=evil)
        assert not r.ok, f"应当拒绝: {evil}"


def test_calculator_zero_division():
    calc = CalculatorTool()
    r = calc(expression="1 / 0")
    assert not r.ok
    assert "除零" in r.error


def test_calculator_empty_input():
    assert not CalculatorTool()(expression="").ok


# ----------------- 检索器 ----------------- #
def test_retriever_exact_hit():
    r = RetrieverTool()(query="光速")
    assert r.ok
    assert "299792458" in r.output


def test_retriever_fuzzy_hit():
    # 子串模糊命中
    r = RetrieverTool()(query="圆周率")
    assert r.ok and "3.14159" in r.output


def test_retriever_miss():
    r = RetrieverTool()(query="反重力引擎")
    assert not r.ok


def test_retriever_custom_kb():
    r = RetrieverTool(knowledge_base={"甲": "乙"})(query="甲")
    assert r.ok and r.output == "乙"


# ----------------- flaky mock ----------------- #
def test_flaky_fails_then_succeeds():
    f = FlakyMockTool(fail_times=1)
    r1 = f(payload="x")
    assert not r1.ok          # 第 1 次注定失败
    r2 = f(payload="x")
    assert r2.ok              # 第 2 次成功
    assert r2.output == "mock-成功结果"


def test_flaky_reset():
    f = FlakyMockTool(fail_times=1)
    f(payload="x")  # 失败
    f(payload="x")  # 成功
    f.reset()
    assert not f(payload="x").ok  # 重置后又从失败开始


# ----------------- 工具路由（核心）----------------- #
def test_registry_route_hit():
    reg = build_default_registry()
    assert reg.route("calculator") is not None
    assert reg.route("retriever") is not None
    assert reg.route("flaky") is not None


def test_registry_route_miss_returns_none():
    """路由不到的工具返回 None（而非抛异常），交给 Agent 去反思。"""
    reg = build_default_registry()
    assert reg.route("不存在的工具") is None


def test_registry_duplicate_register_raises():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    with pytest.raises(ValueError):
        reg.register(CalculatorTool())


def test_registry_names_and_describe():
    reg = build_default_registry()
    assert reg.names() == ["calculator", "flaky", "retriever"]
    text = reg.describe()
    assert "calculator" in text and "retriever" in text


def test_registry_call_counting():
    reg = build_default_registry()
    reg.route("calculator")(expression="1+1")
    reg.route("calculator")(expression="2+2")
    assert reg.total_calls() == 2
    reg.reset_counts()
    assert reg.total_calls() == 0


def test_tool_never_crashes_agent():
    """工具内部异常必须被兜底成 ToolResult(ok=False)，绝不向上抛。"""
    class Boom(CalculatorTool):
        name = "boom"
        def run(self, **kwargs):
            raise RuntimeError("炸了")
    r = Boom()(expression="anything")
    assert not r.ok
    assert "RuntimeError" in r.error
