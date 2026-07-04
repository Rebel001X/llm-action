# -*- coding: utf-8 -*-
"""
AI 能耗-碳排-水耗计算器 (AI Carbon Footprint Calculator)
========================================================

配套书籍:《AI Infrastructures and Sustainability》

本模块是一个**纯 Python(仅依赖标准库 + dataclasses)** 的第一性原理计算器,
把「一次模型训练 / 一次推理服务」翻译成三笔环境账单:

    1. 能耗   Energy       —— 千瓦时 kWh
    2. 碳排   Carbon       —— 吨二氧化碳当量 tCO2e
    3. 水耗   Water        —— 升 L

核心物理链条(全书的骨架):

    GPU 电功率(W)
        ×  GPU 数量  ×  运行时长(h)          →  IT 设备电能(kWh)
        ×  PUE (数据中心整体用电效率)          →  设施总电能(kWh)
        ×  电网碳强度 gCO2/kWh                 →  碳排放(gCO2 → tCO2)
        ×  水耗系数 L/kWh                       →  水足迹(L)

推理侧再把「设施总电能」按 token 摊薄,得到「每千 token 的 gCO2」。

设计原则:
    - 不联网、不下模型、不需 key,纯算术,任何机器都能跑。
    - 每个函数只做一件事,单元可测(见 tests/)。
    - 单位在函数名 / docstring / 变量名里显式标注,杜绝「单位地狱」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# =============================================================================
# 常量:一些行业公认的默认值(可被调用方覆盖)
# =============================================================================

# 每千瓦时耗电对应的数据中心「现场 + 电厂」总水耗系数,单位 升/千瓦时 (L/kWh)。
# 综合现场冷却水(WUE)与发电取水,典型区间约 1.0 ~ 3.0 L/kWh。
DEFAULT_WATER_L_PER_KWH: float = 1.8

# 常见电网碳强度(gCO2e/kWh)。数值为教学近似,用于对比不同电网的清洁程度。
# 可再生占比越高 → 碳强度越低。
GRID_CARBON_INTENSITY: Dict[str, float] = {
    "全球平均 (Global avg)": 475.0,   # IEA 量级
    "中国 (China)": 580.0,            # 火电占比高
    "美国 (US avg)": 380.0,
    "欧盟 (EU avg)": 250.0,
    "法国 (France, 核电为主)": 55.0,   # 核电+水电,极低碳
    "北欧 (Nordic, 水电为主)": 30.0,   # 水电/风电,近乎零碳
    "煤电为主 (Coal-heavy)": 900.0,    # 近乎纯煤
}

# 一吨 CO2 = 1e6 克
GRAMS_PER_TONNE: float = 1_000_000.0


# =============================================================================
# 数据类:把「一次训练任务」的输入参数打包
# =============================================================================

@dataclass
class TrainingConfig:
    """一次模型训练任务的物理输入。

    只描述「硬件 + 运行」的物理量,不掺杂业务假设。
    """
    name: str = "unnamed-model"        # 模型名(仅用于展示)
    num_params_billion: float = 7.0    # 参数量(十亿),仅用于展示/推理摊薄
    num_gpus: int = 1024               # 参与训练的 GPU 数量
    train_hours: float = 720.0         # 训练墙钟时长(小时)
    gpu_power_watts: float = 400.0     # 单卡平均电功率(瓦),如 A100≈400W, H100≈700W
    pue: float = 1.4                   # 数据中心 PUE(设施总电 / IT 设备电),≥1.0
    grid_gco2_per_kwh: float = GRID_CARBON_INTENSITY["全球平均 (Global avg)"]
    water_l_per_kwh: float = DEFAULT_WATER_L_PER_KWH
    utilization: float = 1.0           # 卡的平均利用率 0~1(未跑满时功率折算)

    def __post_init__(self) -> None:
        # ⚠️ 坑:PUE 物理上不可能 < 1.0(设施总电必然 ≥ IT 电)。
        if self.pue < 1.0:
            raise ValueError(f"PUE 必须 ≥ 1.0,收到 {self.pue}")
        if not (0.0 < self.utilization <= 1.0):
            raise ValueError(f"utilization 必须在 (0,1],收到 {self.utilization}")
        if self.num_gpus <= 0 or self.train_hours < 0 or self.gpu_power_watts < 0:
            raise ValueError("GPU 数量/时长/功率必须为非负,且 GPU 数量 > 0")
        if self.grid_gco2_per_kwh < 0 or self.water_l_per_kwh < 0:
            raise ValueError("碳强度 / 水耗系数必须 ≥ 0")


@dataclass
class InferenceConfig:
    """一次推理服务(在线 serving)的物理输入。"""
    name: str = "unnamed-serving"
    num_gpus: int = 8                  # serving 集群 GPU 数
    gpu_power_watts: float = 400.0
    pue: float = 1.4
    grid_gco2_per_kwh: float = GRID_CARBON_INTENSITY["全球平均 (Global avg)"]
    water_l_per_kwh: float = DEFAULT_WATER_L_PER_KWH
    throughput_tokens_per_sec: float = 5000.0  # 整个集群的总吞吐(tokens/s)
    utilization: float = 1.0

    def __post_init__(self) -> None:
        if self.pue < 1.0:
            raise ValueError(f"PUE 必须 ≥ 1.0,收到 {self.pue}")
        if self.throughput_tokens_per_sec <= 0:
            raise ValueError("吞吐必须 > 0,否则每 token 能耗为无穷")
        if not (0.0 < self.utilization <= 1.0):
            raise ValueError(f"utilization 必须在 (0,1],收到 {self.utilization}")


# =============================================================================
# 结果容器
# =============================================================================

@dataclass
class FootprintResult:
    """一次核算的三笔账单 + 中间量,便于测试与展示。"""
    label: str
    it_energy_kwh: float        # IT 设备净电能(不含 PUE 开销)
    facility_energy_kwh: float  # 设施总电能(含 PUE)
    carbon_tco2: float          # 碳排放(吨)
    water_liters: float         # 水耗(升)
    extra: Dict[str, float] = field(default_factory=dict)  # 摊薄类附加指标


# =============================================================================
# 核心计算函数:每个都是一行物理公式,单独可测
# =============================================================================

def it_energy_kwh(num_gpus: int, gpu_power_watts: float,
                  hours: float, utilization: float = 1.0) -> float:
    """IT 设备净电能(kWh)= 功率(kW) × 卡数 × 时长(h) × 利用率。

    为什么:能量 = 功率 × 时间。这是整条链的地基。
    单位换算:瓦(W) / 1000 = 千瓦(kW),千瓦 × 小时 = 千瓦时(kWh)。
    """
    power_kw = (gpu_power_watts / 1000.0) * num_gpus * utilization
    return power_kw * hours


def apply_pue(it_kwh: float, pue: float) -> float:
    """把 IT 电能放大为设施总电能:facility = IT × PUE。

    为什么:数据中心除了服务器,还要给冷却/配电/照明供电。
    PUE = 设施总用电 / IT 用电,是「整栋楼的用电效率」指标,越接近 1 越好。
    """
    return it_kwh * pue


def energy_to_carbon_tco2(facility_kwh: float, grid_gco2_per_kwh: float) -> float:
    """设施电能 → 碳排(吨 CO2e)。

    碳(g)= 电能(kWh) × 电网碳强度(gCO2/kWh);再 /1e6 转吨。
    这一步是「同样的电,谁家的更脏」的核心:清洁电网碳强度低。
    """
    grams = facility_kwh * grid_gco2_per_kwh
    return grams / GRAMS_PER_TONNE


def energy_to_water_liters(facility_kwh: float, water_l_per_kwh: float) -> float:
    """设施电能 → 水耗(升)= 电能(kWh) × 水耗系数(L/kWh)。

    为什么算水:数据中心冷却(蒸发冷却塔)与上游发电(火电/核电冷却)都要取水,
    是《AI Infrastructures and Sustainability》强调的「被忽视的第二资源」。
    """
    return facility_kwh * water_l_per_kwh


# =============================================================================
# 组合函数:一步算出训练 / 推理的完整账单
# =============================================================================

def compute_training_footprint(cfg: TrainingConfig) -> FootprintResult:
    """核算一次训练任务的能耗 / 碳 / 水。"""
    it_kwh = it_energy_kwh(cfg.num_gpus, cfg.gpu_power_watts,
                           cfg.train_hours, cfg.utilization)
    fac_kwh = apply_pue(it_kwh, cfg.pue)
    carbon = energy_to_carbon_tco2(fac_kwh, cfg.grid_gco2_per_kwh)
    water = energy_to_water_liters(fac_kwh, cfg.water_l_per_kwh)

    # 附加:GPU·小时(算力规模的直观指标),以及每十亿参数的碳成本
    gpu_hours = cfg.num_gpus * cfg.train_hours
    carbon_per_bparam = carbon / cfg.num_params_billion if cfg.num_params_billion > 0 else 0.0

    return FootprintResult(
        label=cfg.name,
        it_energy_kwh=it_kwh,
        facility_energy_kwh=fac_kwh,
        carbon_tco2=carbon,
        water_liters=water,
        extra={
            "gpu_hours": gpu_hours,
            "carbon_tco2_per_billion_params": carbon_per_bparam,
        },
    )


def compute_inference_footprint(cfg: InferenceConfig,
                                total_tokens: float) -> FootprintResult:
    """核算一次推理服务处理 `total_tokens` 个 token 的账单,并摊薄到每千 token。

    推理与训练的差别:训练是「一锤子买卖」(固定时长),推理是「按流量计费」。
    我们先由吞吐反推需要的运行秒数,再走同一条物理链,最后按 token 摊。
    """
    if total_tokens < 0:
        raise ValueError("total_tokens 必须 ≥ 0")

    # 处理这些 token 需要的墙钟时间(秒 → 小时)
    seconds = total_tokens / cfg.throughput_tokens_per_sec
    hours = seconds / 3600.0

    it_kwh = it_energy_kwh(cfg.num_gpus, cfg.gpu_power_watts, hours, cfg.utilization)
    fac_kwh = apply_pue(it_kwh, cfg.pue)
    carbon = energy_to_carbon_tco2(fac_kwh, cfg.grid_gco2_per_kwh)
    water = energy_to_water_liters(fac_kwh, cfg.water_l_per_kwh)

    # 摊薄:每 1000 token 的 gCO2 / Wh / mL —— serving 团队最关心的单位指标
    tokens_k = total_tokens / 1000.0 if total_tokens > 0 else 0.0
    if tokens_k > 0:
        gco2_per_ktok = (carbon * GRAMS_PER_TONNE) / tokens_k
        wh_per_ktok = (fac_kwh * 1000.0) / tokens_k
        ml_water_per_ktok = (water * 1000.0) / tokens_k
    else:
        gco2_per_ktok = wh_per_ktok = ml_water_per_ktok = 0.0

    return FootprintResult(
        label=cfg.name,
        it_energy_kwh=it_kwh,
        facility_energy_kwh=fac_kwh,
        carbon_tco2=carbon,
        water_liters=water,
        extra={
            "total_tokens": float(total_tokens),
            "run_hours": hours,
            "gco2_per_1k_tokens": gco2_per_ktok,
            "wh_per_1k_tokens": wh_per_ktok,
            "ml_water_per_1k_tokens": ml_water_per_ktok,
        },
    )


# =============================================================================
# 对比工具:同一训练任务在不同电网下的碳排
# =============================================================================

def compare_grids(cfg: TrainingConfig,
                  grids: Dict[str, float] | None = None) -> List[FootprintResult]:
    """固定训练任务,只换电网碳强度,返回按碳排升序排序的结果列表。

    用途:回答「把训练搬到更清洁的电网能省多少碳?」这一决策问题。
    """
    grids = grids or GRID_CARBON_INTENSITY
    results: List[FootprintResult] = []
    for grid_name, gco2 in grids.items():
        # 复制配置只改电网(dataclass 不可变字段少,直接构造更清晰)
        variant = TrainingConfig(
            name=f"{cfg.name} @ {grid_name}",
            num_params_billion=cfg.num_params_billion,
            num_gpus=cfg.num_gpus,
            train_hours=cfg.train_hours,
            gpu_power_watts=cfg.gpu_power_watts,
            pue=cfg.pue,
            grid_gco2_per_kwh=gco2,
            water_l_per_kwh=cfg.water_l_per_kwh,
            utilization=cfg.utilization,
        )
        results.append(compute_training_footprint(variant))
    # 按碳排从低到高排序,最清洁的电网排最前
    results.sort(key=lambda r: r.carbon_tco2)
    return results


def renewable_share_to_intensity(renewable_share: float,
                                 fossil_gco2_per_kwh: float = 820.0,
                                 renewable_gco2_per_kwh: float = 20.0) -> float:
    """由「可再生占比」线性插值出电网碳强度(gCO2/kWh)。

    模型:电网 = 可再生部分 + 化石部分的加权平均。
        强度 = share × 清洁强度 + (1-share) × 化石强度
    可再生占比越高 → 碳强度单调下降。用于「可再生越高碳越低」的直觉验证。
    """
    if not (0.0 <= renewable_share <= 1.0):
        raise ValueError(f"renewable_share 必须在 [0,1],收到 {renewable_share}")
    return (renewable_share * renewable_gco2_per_kwh
            + (1.0 - renewable_share) * fossil_gco2_per_kwh)


def human_readable(result: FootprintResult) -> str:
    """把一次核算结果格式化成一行人类可读文本(用于 CLI / demo 打印)。"""
    return (
        f"[{result.label}] "
        f"设施能耗={result.facility_energy_kwh:,.1f} kWh | "
        f"碳排={result.carbon_tco2:,.3f} tCO2e | "
        f"水耗={result.water_liters:,.1f} L"
    )


# =============================================================================
# 直接运行时的自检(python carbon_calculator.py)
# =============================================================================

if __name__ == "__main__":
    demo = TrainingConfig(name="GPT-类 7B 演示", num_params_billion=7,
                          num_gpus=1024, train_hours=500, gpu_power_watts=400,
                          pue=1.4, grid_gco2_per_kwh=475.0)
    res = compute_training_footprint(demo)
    print(human_readable(res))
    print("每十亿参数碳成本:",
          f"{res.extra['carbon_tco2_per_billion_params']:.3f} tCO2e/B")
