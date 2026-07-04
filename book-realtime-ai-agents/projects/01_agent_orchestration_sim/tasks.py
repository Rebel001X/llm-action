# -*- coding: utf-8 -*-
"""
tasks.py —— 任务定义与「黄金轨迹」（golden trajectory）
========================================================

对应第 4 章「评估」一节的核心概念:
  - golden dataset  : 每个任务都带一个「标准答案」和「应当走的工具路径」
  - tool trajectory : Agent 实际调用了哪些工具、顺序对不对
  - response match  : 最终答案与标准答案是否一致

一个 `Task` 就是一道题:
  - tid          : 任务 id
  - prompt       : 人类可读的任务描述（模拟用户请求）
  - kind         : 任务类型（决定规划器怎么规划）
  - payload      : 结构化参数（比如要算的表达式、要查的关键词）
  - expected     : 标准答案（response match 用）
  - golden_tools : 期望调用的工具序列（tool trajectory 用）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Task:
    tid: str
    prompt: str
    kind: str                       # "calc" | "lookup" | "flaky" | "unknown"
    payload: dict = field(default_factory=dict)
    expected: Any = None            # 标准答案
    golden_tools: List[str] = field(default_factory=list)  # 期望的工具轨迹


def build_task_suite() -> List[Task]:
    """构造一批仿真任务（golden dataset）。

    覆盖 4 类:
      1) calc   —— 计算器题（response = 数字）
      2) lookup —— 检索题（response = 知识库文本）
      3) flaky  —— 会失败一次、需反思重试才成功的题（考验反思循环）
      4) unknown—— 故意给一个「无法完成」的题（考验循环能否安全终止，不死循环）
    """
    return [
        # --- 计算器题 ---
        Task("t1", "请计算 12 * (3 + 4)", "calc",
             payload={"expression": "12 * (3 + 4)"}, expected=84.0,
             golden_tools=["calculator"]),
        Task("t2", "请计算 (100 - 58) / 6", "calc",
             payload={"expression": "(100 - 58) / 6"}, expected=7.0,
             golden_tools=["calculator"]),
        Task("t3", "请计算 2 ** 10", "calc",
             payload={"expression": "2 ** 10"}, expected=1024.0,
             golden_tools=["calculator"]),
        # --- 检索题 ---
        Task("t4", "光速是多少？", "lookup",
             payload={"query": "光速"},
             expected="真空中光速约为 299792458 米/秒。",
             golden_tools=["retriever"]),
        Task("t5", "地球半径是多少？", "lookup",
             payload={"query": "地球半径"},
             expected="地球平均半径约为 6371 千米。",
             golden_tools=["retriever"]),
        # --- 反思重试题（flaky 工具先失败一次）---
        Task("t6", "调用不稳定服务拿一个结果（会先失败再成功）", "flaky",
             payload={"payload": "ping"}, expected="mock-成功结果",
             golden_tools=["flaky", "flaky"]),  # 期望：第一次失败、反思后再来一次
        # --- 无法完成题（考验终止 & 不死循环）---
        Task("t7", "查询一个知识库里没有的冷门词条", "unknown",
             payload={"query": "反重力引擎"}, expected=None,
             golden_tools=["retriever"]),
    ]
