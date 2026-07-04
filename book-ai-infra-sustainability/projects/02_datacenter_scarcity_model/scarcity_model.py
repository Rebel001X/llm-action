# -*- coding: utf-8 -*-
"""
数据中心算力增长 vs 碳预算 —— 稀缺性模型 (Scarcity Model)
================================================================

配套书籍:《AI Infrastructures and Sustainability》
主题:从"稀缺性 (scarcity)"视角建模 —— 碳预算 (carbon budget) 是有限的、
不可再生的"公共资源",而 AI 算力需求却在指数级增长。本模块回答一个问题:

    "按当前趋势,AI 基础设施的累计碳排 (cumulative CO2) 会在第几年
     耗尽给定的剩余碳预算?能效改进和可再生能源能不能救我们?"

第一性原理 (first principles)
-----------------------------
某一年的 AI 碳排放,可以拆成三个可乘因子 (multiplicative factors):

    年碳排 = 算力需求 (compute)          # 越大越多
           × 能耗强度 (energy per compute) # 越低越好 —— 靠能效改进
           × 碳强度 (carbon per energy)    # 越低越好 —— 靠可再生渗透

这就是经典的 **Kaya 恒等式 (Kaya identity)** 在 AI 场景的变体:把一个
总量指标分解成若干"驱动因子"的乘积,每个因子各自随时间演化。

- 算力 compute:以年增长率 g_compute 复利增长(需求侧,指数上涨)。
- 能耗强度 intensity:每年改进 r_eff(每 FLOP/每次推理耗能下降),
  下降是"抵消 (offset)"增长的关键杠杆。
- 碳强度 carbon:由电网可再生渗透率决定;可再生占比越高,每度电碳越低。

把这三条曲线逐年相乘,得到"年碳排",再逐年累加得到"累计碳排",
和"剩余碳预算"比较 —— 这就是稀缺性视角:预算是墙,曲线是逼近墙的过程。

设计原则
--------
- 纯 numpy,CPU,离线,无网络、无外部数据、无随机性(结果可复现)。
- 所有函数有清晰的量纲说明;单位统一在函数 docstring 标注。
- 归一化:第 0 年算力 = 1.0(相对单位),碳排单位为 GtCO2(十亿吨),
  由第 0 年的绝对碳排 base_emission_gt 校准。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. 情景参数 (Scenario) —— 一个不可变的数据类,承载所有输入假设
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    """一个稀缺性情景 (scarcity scenario) 的全部输入假设。

    字段单位说明:
      name                场景名(中文/英文皆可)
      years               模拟年数(含第 0 年),int，>=1
      compute_growth      算力年增长率,如 0.30 表示每年 +30%(复利)
      efficiency_gain     能效年改进率,如 0.15 表示每年能耗强度 ×(1-0.15)
      renewable_start     第 0 年可再生渗透率 (0~1)
      renewable_end       最后一年可再生渗透率 (0~1),线性插值
      grid_carbon_fossil  化石电力碳强度 (tCO2 / MWh 的相对单位),默认 1.0
      base_emission_gt    第 0 年 AI 相关碳排 (GtCO2)，用于把相对量校准成绝对量
      carbon_budget_gt    剩余碳预算 (GtCO2)，这是"稀缺资源"的总量,即那面墙
    """

    name: str
    years: int = 26
    compute_growth: float = 0.30
    efficiency_gain: float = 0.15
    renewable_start: float = 0.30
    renewable_end: float = 0.60
    grid_carbon_fossil: float = 1.0
    base_emission_gt: float = 0.10
    carbon_budget_gt: float = 5.0

    def __post_init__(self) -> None:
        # ⚠️ 坑:参数校验一定要有,否则负增长率/超范围渗透率会悄悄产出错误曲线
        if self.years < 1:
            raise ValueError("years 必须 >= 1")
        if self.compute_growth <= -1.0:
            raise ValueError("compute_growth 必须 > -1(否则算力变负)")
        if not (0.0 <= self.efficiency_gain < 1.0):
            raise ValueError("efficiency_gain 必须在 [0, 1) 内")
        for v, nm in ((self.renewable_start, "renewable_start"),
                      (self.renewable_end, "renewable_end")):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{nm} 必须在 [0, 1] 内")
        if self.base_emission_gt < 0:
            raise ValueError("base_emission_gt 必须 >= 0")
        if self.carbon_budget_gt <= 0:
            raise ValueError("carbon_budget_gt 必须 > 0")


# ---------------------------------------------------------------------------
# 2. 三条驱动因子曲线 (Kaya factors)
# ---------------------------------------------------------------------------
def compute_curve(years: int, growth: float) -> np.ndarray:
    """算力需求曲线(相对单位,第 0 年 = 1.0),按复利指数增长。

    数学:compute[t] = (1 + growth) ** t
    这是需求侧的"稀缺压力源":哪怕单位算力越来越省,总需求也可能压倒改进。
    """
    t = np.arange(years, dtype=float)
    return np.power(1.0 + growth, t)


def efficiency_curve(years: int, gain: float) -> np.ndarray:
    """能耗强度曲线(相对单位,第 0 年 = 1.0),每年按 (1-gain) 复利下降。

    数学:intensity[t] = (1 - gain) ** t
    gain=0 表示毫无改进(强度恒为 1);gain 越大,曲线下降越快 —— 这是"抵消"杠杆。
    """
    t = np.arange(years, dtype=float)
    return np.power(1.0 - gain, t)


def renewable_curve(years: int, start: float, end: float) -> np.ndarray:
    """可再生渗透率曲线 (0~1),从 start 线性过渡到 end。

    只用了最朴素的线性插值 np.linspace;真实电网是 S 曲线,但线性足以传达机制。
    years==1 时只有第 0 年,直接返回 [start]。
    """
    if years == 1:
        return np.array([start], dtype=float)
    return np.linspace(start, end, years, dtype=float)


def carbon_intensity_curve(years: int, start: float, end: float,
                           fossil: float = 1.0) -> np.ndarray:
    """电力碳强度曲线(相对单位)。可再生占比部分视为零碳。

    数学:carbon[t] = fossil × (1 - renewable[t])
    可再生渗透 100% ⇒ 碳强度 0;渗透 0% ⇒ 碳强度 = fossil。
    """
    ren = renewable_curve(years, start, end)
    return fossil * (1.0 - ren)


# ---------------------------------------------------------------------------
# 3. 年碳排 & 累计碳排 (Kaya 乘积 + 累加)
# ---------------------------------------------------------------------------
def annual_emissions(scn: Scenario) -> np.ndarray:
    """逐年 AI 碳排 (GtCO2)。核心:三因子逐元素相乘,再用第 0 年绝对量校准。

    第 0 年:compute=1, intensity=1, carbon=fossil*(1-renewable_start)。
    我们希望"第 0 年年碳排 == base_emission_gt",所以要除以第 0 年的原始乘积
    做归一化,再乘 base_emission_gt。这样单位就从"相对"变成"GtCO2"。
    """
    comp = compute_curve(scn.years, scn.compute_growth)
    eff = efficiency_curve(scn.years, scn.efficiency_gain)
    carb = carbon_intensity_curve(scn.years, scn.renewable_start,
                                  scn.renewable_end, scn.grid_carbon_fossil)

    raw = comp * eff * carb            # 相对单位的年碳排(逐元素相乘)
    raw0 = raw[0]                      # 第 0 年的相对量,用作归一化基准

    # ⚠️ 坑:若第 0 年可再生渗透 == 100%,raw0 会是 0,归一化会除零。
    # 物理含义是"起点就零碳",这时整条曲线都应为 0(碳强度恒 0)。
    if raw0 == 0.0:
        return np.zeros(scn.years, dtype=float)

    return scn.base_emission_gt * (raw / raw0)


def cumulative_emissions(scn: Scenario) -> np.ndarray:
    """累计碳排 (GtCO2),对年碳排做前缀和 (prefix sum / np.cumsum)。

    稀缺性的本质就在这条曲线:它单调不减,像水位不断上涨,
    碳预算是水坝高度,曲线穿过水坝的那一年就是"超预算年"。
    """
    return np.cumsum(annual_emissions(scn))


# ---------------------------------------------------------------------------
# 4. 稀缺性判定 (Scarcity verdict) —— 哪一年超预算?
# ---------------------------------------------------------------------------
def budget_exceeded_year(scn: Scenario) -> Optional[int]:
    """返回累计碳排首次 > carbon_budget_gt 的年份索引;整段都不超返回 None。

    实现:用 np.searchsorted 在单调递增的累计曲线上二分查找预算所在位置。
    - side='right' 表示找"第一个 > budget"的位置(严格超出才算)。
    - 若返回值 == years,说明整段都没超,返回 None。
    """
    cum = cumulative_emissions(scn)
    # np.searchsorted:在有序数组 cum 中,找到把 budget 插入后仍保持有序的位置。
    # side='right':相等也排在右边,于是拿到的是第一个"严格大于 budget"的下标。
    idx = int(np.searchsorted(cum, scn.carbon_budget_gt, side="right"))
    if idx >= scn.years:
        return None                    # 预算够用,整段不超
    return idx


def remaining_budget_curve(scn: Scenario) -> np.ndarray:
    """剩余碳预算随年份的曲线 (GtCO2) = budget - 累计碳排,可为负(赤字)。"""
    return scn.carbon_budget_gt - cumulative_emissions(scn)


# ---------------------------------------------------------------------------
# 5. 情景汇总 (Result) —— 供 CLI / demo / 报表使用
# ---------------------------------------------------------------------------
@dataclass
class ScenarioResult:
    """一个情景跑完后的完整结果快照。"""

    scenario: Scenario
    annual: np.ndarray = field(repr=False)
    cumulative: np.ndarray = field(repr=False)
    exceeded_year: Optional[int]
    total_emission_gt: float
    budget_gt: float
    within_budget: bool


def run_scenario(scn: Scenario) -> ScenarioResult:
    """把一个 Scenario 从头跑到尾,打包成 ScenarioResult。"""
    ann = annual_emissions(scn)
    cum = cumulative_emissions(scn)
    ex = budget_exceeded_year(scn)
    total = float(cum[-1])
    return ScenarioResult(
        scenario=scn,
        annual=ann,
        cumulative=cum,
        exceeded_year=ex,
        total_emission_gt=total,
        budget_gt=scn.carbon_budget_gt,
        within_budget=(ex is None),
    )


def scarcity_pressure(scn: Scenario) -> float:
    """稀缺性压力指数 = 累计总碳排 / 碳预算。

    >1 表示超预算(赤字),<1 表示预算内。这个无量纲比值方便跨情景横向比较。
    """
    total = float(cumulative_emissions(scn)[-1])
    return total / scn.carbon_budget_gt


# ---------------------------------------------------------------------------
# 6. 预置情景库 (便于 demo 与教学对比)
# ---------------------------------------------------------------------------
def default_scenarios() -> List[Scenario]:
    """返回一组对照情景:一览无余地看清每个杠杆的作用。"""
    return [
        Scenario(
            name="基准·放任 (BAU 高增长)",
            compute_growth=0.30, efficiency_gain=0.10,
            renewable_start=0.30, renewable_end=0.40,
            carbon_budget_gt=5.0,
        ),
        Scenario(
            name="强能效改进 (每年-25%强度)",
            compute_growth=0.30, efficiency_gain=0.25,
            renewable_start=0.30, renewable_end=0.40,
            carbon_budget_gt=5.0,
        ),
        Scenario(
            name="高可再生渗透 (30%→90%)",
            compute_growth=0.30, efficiency_gain=0.10,
            renewable_start=0.30, renewable_end=0.90,
            carbon_budget_gt=5.0,
        ),
        Scenario(
            name="双管齐下 (能效+可再生)",
            compute_growth=0.30, efficiency_gain=0.25,
            renewable_start=0.30, renewable_end=0.90,
            carbon_budget_gt=5.0,
        ),
    ]


if __name__ == "__main__":
    # ⚠️ Windows 坑:控制台默认 GBK 编码,print emoji(✅❌)会 UnicodeEncodeError。
    # 解决:把标准输出重配成 UTF-8。这是 llm-action 系列在 Windows 上的通用踩坑。
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Python 3.7+ 支持
    except Exception:
        pass

    # 简易自检:直接 python scarcity_model.py 就能看到每个情景的结论
    print("=" * 68)
    print("数据中心算力增长 vs 碳预算 —— 稀缺性模型自检")
    print("=" * 68)
    for s in default_scenarios():
        r = run_scenario(s)
        verdict = ("✅ 预算内" if r.within_budget
                   else f"❌ 第 {r.exceeded_year} 年超预算")
        print(f"[{s.name:<26}] 累计={r.total_emission_gt:6.2f} Gt / "
              f"预算={r.budget_gt:.1f} Gt | 压力={scarcity_pressure(s):.2f} | {verdict}")
