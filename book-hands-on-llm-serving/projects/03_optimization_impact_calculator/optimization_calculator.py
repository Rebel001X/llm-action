# -*- coding: utf-8 -*-
"""
optimization_calculator.py
==========================
优化技术收益计算器（Optimization Impact Calculator）—— 核心解析模型。

对应《Hands-On LLM Serving and Optimization》第 5 / 6 / 7 章：
    - 第 5 章：GPU 物理底座（显存容量 / 带宽 / 算力）、KV 缓存公式、算术强度、屋顶线；
    - 第 6 章：量化（位宽）、连续批处理（continuous batching）、KV 量化；
    - 第 7 章：高级优化的叠加与权衡。

本模块**不加载真实模型、不联网、不需要 GPU**，全部用「第一性原理解析公式」估算：
给定「模型配置 + 硬件配置 + 一组优化开关」，输出三大指标的**叠加收益**：
    ① 显存占用（memory，GB）
    ② 吞吐（throughput，token/s）—— 受显存与带宽双重上限约束
    ③ 单请求延迟（latency，每 token 的 ms/token 与 TTFT）

设计目标：
    - 每个公式都能追溯到书里的物理原理（见注释里的「书 pXXX」标注）；
    - 各优化的收益可以**独立计算**，也可以**顺序叠加**（waterfall，瀑布），
      从而回答面试高频问题：「量化 + 连续批 + KV 量化叠起来到底省多少 / 快多少？」

术语中英并列，注释中文。纯 Python + dataclass，无第三方依赖（除了可选的 numpy，本文件不用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple


# =============================================================================
# 一、基础常量与精度表
# =============================================================================
# 每种数值精度（dtype）占多少「字节 / 参数」。书里反复用到（p.502）：
#   7B 参数 × 2 字节/参数（FP16）= 14 GB。
# 位宽（bit width）÷ 8 = 字节数。这是「量化省显存」的第一性原理：
#   显存 ∝ 位宽。FP16(16bit)→INT8(8bit) 直接砍半；→INT4(4bit) 砍到 1/4。
BYTES_PER_PARAM: Dict[str, float] = {
    "fp32": 4.0,   # 32 bit
    "fp16": 2.0,   # 16 bit（HBM 上最常见的服务精度，等价 bf16）
    "bf16": 2.0,   # 16 bit
    "fp8": 1.0,    # 8 bit（W8A8 / FP8，Hopper 起硬件支持）
    "int8": 1.0,   # 8 bit
    "int4": 0.5,   # 4 bit（W4A16，GPTQ/AWQ 最常用）
    "fp4": 0.5,    # 4 bit
}

# 各精度的位宽（bit），用于「量化降显存按位宽比例」这一核心断言的自洽校验。
BITS: Dict[str, int] = {
    "fp32": 32, "fp16": 16, "bf16": 16,
    "fp8": 8, "int8": 8, "int4": 4, "fp4": 4,
}

# 低精度是否能「加速计算（提 FLOPS）」。书 p.510 Table 6-2：
#   比特减半，Tensor Core 算力通常翻倍（FP8/INT8 相对 FP16 ×2）。
# 但注意：weight-only 量化（W4A16）**不加速计算**（要反量化回高比特），
#   只有 weight+activation（W8A8/FP8）才提 FLOPS。见 quantize() 里的处理。
COMPUTE_SPEEDUP_VS_FP16: Dict[str, float] = {
    "fp16": 1.0, "bf16": 1.0,
    "fp8": 2.0, "int8": 2.0,   # 8bit Tensor Core ≈ 2× FP16
    "int4": 1.0, "fp4": 1.0,   # weight-only：算力不变（默认按 W4A16 处理）
    "fp32": 0.5,
}


# =============================================================================
# 二、配置数据类（Config）
# =============================================================================
@dataclass(frozen=True)
class ModelConfig:
    """模型配置。字段名对应 HuggingFace config.json（书 p.351）。"""
    name: str = "Llama-2-7B"
    num_params_b: float = 7.0          # 参数量（单位：十亿 B）
    num_layers: int = 32               # num_hidden_layers，算 KV 缓存要用
    num_heads: int = 32                # 注意力头数（KV 头数，MHA 下等于 query 头数）
    head_dim: int = 128                # 每个头的维度（hidden_size / num_heads）
    hidden_size: int = 4096            # 隐藏维度 = num_heads × head_dim
    max_seq_len: int = 4096            # 单请求最大 token 数（prompt + 生成）

    def kv_bytes_per_token(self, kv_dtype: str = "fp16") -> float:
        """
        每 token 的 KV 缓存大小（字节）—— 书 p.399 的核心公式：

            每 token KV = 2 × 层数 × 注意力头数 × 头维度 × 精度字节数
                          └ K 和 V 两份 ┘

        例（Llama-7B，FP16）：2 × 32 × 32 × 128 × 2 = 524,288 字节 = 0.5 MB/token。
        这条公式是「长上下文 / 大 batch 显存爆炸」的根源（KV 随 batch×seq_len 线性增长）。
        """
        return 2 * self.num_layers * self.num_heads * self.head_dim * BYTES_PER_PARAM[kv_dtype]


@dataclass(frozen=True)
class HardwareConfig:
    """硬件配置。对应书第 5 章 GPU 规格表（p.163 等）。默认给一张 A100-80G。"""
    name: str = "A100-80GB"
    vram_gb: float = 80.0              # 显存容量（GB）—— 决定能装多大 batch
    mem_bandwidth_tb_s: float = 2.0    # 显存带宽（TB/s）—— decode 的命门
    fp16_tflops: float = 312.0         # FP16 Tensor Core 算力（TFLOPS）—— prefill 的命门


@dataclass(frozen=True)
class WorkloadConfig:
    """工作负载：一次请求的输入 / 输出长度，用于 prefill vs decode 的划分。"""
    prompt_len: int = 1024             # 输入 prompt token 数（决定 prefill 计算量）
    gen_len: int = 512                 # 生成 token 数（决定 decode 步数）


# =============================================================================
# 三、优化配置（一组开关）
# =============================================================================
@dataclass(frozen=True)
class OptimizationConfig:
    """
    一组优化开关。三大优化各自可开关：
      - weight_dtype：权重量化位宽（"fp16" 表示不量化，"int8"/"int4" 表示量化）；
      - activation_dtype：激活精度（区分 weight-only 与 weight+activation）；
      - batch_size：连续批处理的目标 batch（1 表示不批处理）；
      - kv_dtype：KV 缓存量化位宽（"fp16" 不量化，"int8"/"fp8" 量化）。
    """
    weight_dtype: str = "fp16"
    activation_dtype: str = "fp16"
    batch_size: int = 1
    kv_dtype: str = "fp16"

    def is_weight_and_activation(self) -> bool:
        """W8A8/FP8 这类「权重+激活都量化」才提 FLOPS；W4A16 只是 weight-only。"""
        return BITS[self.activation_dtype] < 16


# =============================================================================
# 四、结果数据类
# =============================================================================
@dataclass
class Metrics:
    """一次配置下算出的三大指标。"""
    weight_mem_gb: float          # 模型权重显存（GB）
    kv_mem_gb: float              # KV 缓存显存（GB，随 batch×seq 增长）
    total_mem_gb: float           # 总显存 = 权重 + KV + 激活余量
    throughput_tok_s: float       # 吞吐（token/s）
    latency_ms_per_token: float   # decode 每 token 延迟（ms）
    ttft_ms: float                # 首 token 延迟（prefill 时间，ms）
    effective_batch: int          # 实际可用 batch（可能被显存/带宽压低）
    oom: bool = False             # 是否显存溢出（Out Of Memory）

    def as_row(self) -> Dict[str, float]:
        return {
            "weight_mem_gb": round(self.weight_mem_gb, 3),
            "kv_mem_gb": round(self.kv_mem_gb, 3),
            "total_mem_gb": round(self.total_mem_gb, 3),
            "throughput_tok_s": round(self.throughput_tok_s, 2),
            "latency_ms_per_token": round(self.latency_ms_per_token, 4),
            "ttft_ms": round(self.ttft_ms, 2),
            "effective_batch": self.effective_batch,
            "oom": self.oom,
        }


# 激活/中间计算预留：书 p.394 说显存里除权重和 KV 还要给「一小块中间计算空间」。
# 用一个固定比例（相对权重）近似，避免把显存算得过满而不真实。
ACTIVATION_OVERHEAD_RATIO = 0.10


# =============================================================================
# 五、核心计算器
# =============================================================================
class OptimizationCalculator:
    """
    优化技术收益计算器。

    用法：
        calc = OptimizationCalculator(model, hardware, workload)
        m = calc.compute(OptimizationConfig(weight_dtype="int4", batch_size=16, kv_dtype="int8"))
        print(m.as_row())

    也可用 calc.waterfall([...]) 顺序叠加多个优化，得到瀑布图数据。
    """

    def __init__(self, model: ModelConfig, hardware: HardwareConfig, workload: WorkloadConfig):
        self.model = model
        self.hw = hardware
        self.wl = workload

    # ---------------------------------------------------------------------
    # 5.1 显存：权重 + KV + 激活
    # ---------------------------------------------------------------------
    def weight_memory_gb(self, weight_dtype: str) -> float:
        """
        权重显存（GB）= 参数量 × 每参数字节数。
        书 p.502/503：7B × 2B(FP16) = 14GB；量化到 INT8(1B) → 7GB；INT4(0.5B) → 3.5GB。
        这就是「量化降显存严格按位宽比例」的来源：mem ∝ bytes_per_param ∝ bit_width。
        """
        bytes_total = self.model.num_params_b * 1e9 * BYTES_PER_PARAM[weight_dtype]
        return bytes_total / (1024 ** 3)

    def kv_memory_gb(self, batch: int, kv_dtype: str, seq_len: Optional[int] = None) -> float:
        """
        KV 缓存显存（GB）= 每 token KV × 总 token 数（书 p.419）：
            总 KV = 每 token KV × (batch × seq_len)
        KV 量化（kv_dtype fp16→int8）同样严格按位宽砍：int8 是 fp16 的一半。
        """
        seq_len = seq_len if seq_len is not None else self.model.max_seq_len
        per_tok = self.model.kv_bytes_per_token(kv_dtype)
        bytes_total = per_tok * batch * seq_len
        return bytes_total / (1024 ** 3)

    # ---------------------------------------------------------------------
    # 5.2 「一张卡能装多大 batch」—— 用剩余显存反推（书 p.435 表 5-7）
    # ---------------------------------------------------------------------
    def max_batch_by_memory(self, weight_dtype: str, kv_dtype: str,
                            seq_len: Optional[int] = None) -> int:
        """
        显存允许的最大 batch：
            可用于 KV 的显存 = 总显存 − 权重 − 激活余量
            max_batch = 该显存 ÷ (每 token KV × seq_len)
        这是吞吐「有上限」的第一道天花板：显存装不下更多并发请求。
        """
        seq_len = seq_len if seq_len is not None else self.model.max_seq_len
        w = self.weight_memory_gb(weight_dtype)
        act = w * ACTIVATION_OVERHEAD_RATIO
        kv_budget_gb = self.hw.vram_gb - w - act
        if kv_budget_gb <= 0:
            return 0
        per_tok_gb = self.model.kv_bytes_per_token(kv_dtype) / (1024 ** 3)
        max_tokens = kv_budget_gb / per_tok_gb
        return max(0, int(max_tokens // seq_len))

    # ---------------------------------------------------------------------
    # 5.3 吞吐（throughput）—— 受「带宽 roofline」与「显存 batch 上限」双重约束
    # ---------------------------------------------------------------------
    def decode_throughput_tok_s(self, opt: OptimizationConfig) -> Tuple[float, int]:
        """
        decode 阶段吞吐（token/s），返回 (吞吐, 实际生效 batch)。

        🔬 第一性原理（书 p.61）：decode 是**带宽受限**的——每生成 1 个 token，
        都要把「全部权重 + 当前所有请求的 KV」从 HBM 读一遍。所以：

            单步 decode 读入字节 = 权重字节 + batch × seq × 每token KV字节
            单步耗时 ≈ 读入字节 / 显存带宽
            吞吐(token/s) = batch / 单步耗时      （一步给 batch 个请求各出 1 token）

        batching 提吞吐的本质：**权重只读一次，却服务 batch 个请求** → 权重的搬运成本被摊薄。
        但这个收益**有上限**：
          ① 显存上限：batch 不能超过 max_batch_by_memory（装不下）；
          ② 带宽上限：batch 很大时，KV 搬运项 (batch×seq×kv) 会超过权重项而主导，
             单步耗时随 batch 线性上升，吞吐趋于饱和（roofline 屋顶）。
        """
        # ① 先按显存夹紧目标 batch
        mem_cap = self.max_batch_by_memory(opt.weight_dtype, opt.kv_dtype)
        eff_batch = min(opt.batch_size, mem_cap) if mem_cap > 0 else 0
        if eff_batch <= 0:
            return 0.0, 0

        # ② 单步 decode 需要从 HBM 搬运的字节
        weight_bytes = self.model.num_params_b * 1e9 * BYTES_PER_PARAM[opt.weight_dtype]
        # decode 时每个请求平均缓存约 seq_len 个 token 的 KV（用负载均值近似）
        avg_ctx = self.wl.prompt_len + self.wl.gen_len // 2
        kv_bytes_per_req = self.model.kv_bytes_per_token(opt.kv_dtype) * avg_ctx
        step_bytes = weight_bytes + eff_batch * kv_bytes_per_req

        bandwidth_bytes_s = self.hw.mem_bandwidth_tb_s * 1e12  # TB/s → B/s
        step_time_s = step_bytes / bandwidth_bytes_s

        # 一步给 eff_batch 个请求各出 1 token
        throughput = eff_batch / step_time_s
        return throughput, eff_batch

    def decode_latency_ms_per_token(self, opt: OptimizationConfig, eff_batch: int) -> float:
        """
        单请求 decode 延迟（ms/token）= 单步耗时（因为一步内 batch 个请求并行出 token）。
        注意：batch 越大，单步搬的 KV 越多，**单请求延迟会略升**——这就是
        「吞吐 vs 延迟」的经典权衡（书 p.44）：批处理提吞吐，但拉高单请求延迟。
        """
        if eff_batch <= 0:
            return float("inf")
        weight_bytes = self.model.num_params_b * 1e9 * BYTES_PER_PARAM[opt.weight_dtype]
        avg_ctx = self.wl.prompt_len + self.wl.gen_len // 2
        kv_bytes_per_req = self.model.kv_bytes_per_token(opt.kv_dtype) * avg_ctx
        step_bytes = weight_bytes + eff_batch * kv_bytes_per_req
        bandwidth_bytes_s = self.hw.mem_bandwidth_tb_s * 1e12
        return step_bytes / bandwidth_bytes_s * 1000.0

    def ttft_ms(self, opt: OptimizationConfig) -> float:
        """
        首 token 延迟 TTFT ≈ prefill 时间。prefill 是**算力受限**（书 p.58）：
            prefill FLOPs ≈ 2 × 参数量 × prompt_len   （每参数一次乘加，×2）
            TTFT ≈ FLOPs / 有效算力
        weight+activation 量化（W8A8/FP8）会把算力翻倍 → TTFT 减半（书 p.510）。
        weight-only（W4A16）不提算力 → TTFT 基本不变。
        """
        flops = 2 * self.model.num_params_b * 1e9 * self.wl.prompt_len
        speedup = COMPUTE_SPEEDUP_VS_FP16.get(opt.weight_dtype, 1.0)
        if opt.is_weight_and_activation():
            speedup = max(speedup, COMPUTE_SPEEDUP_VS_FP16.get(opt.activation_dtype, 1.0))
        else:
            # weight-only：算力不变（书 p.511），无论权重是不是 4bit
            speedup = 1.0
        eff_tflops = self.hw.fp16_tflops * speedup
        seconds = flops / (eff_tflops * 1e12)
        return seconds * 1000.0

    # ---------------------------------------------------------------------
    # 5.4 一次性把三大指标算出来
    # ---------------------------------------------------------------------
    def compute(self, opt: OptimizationConfig) -> Metrics:
        w_mem = self.weight_memory_gb(opt.weight_dtype)
        throughput, eff_batch = self.decode_throughput_tok_s(opt)
        kv_mem = self.kv_memory_gb(max(eff_batch, 1), opt.kv_dtype)
        act_mem = w_mem * ACTIVATION_OVERHEAD_RATIO
        total_mem = w_mem + kv_mem + act_mem
        oom = total_mem > self.hw.vram_gb or eff_batch <= 0
        latency = self.decode_latency_ms_per_token(opt, eff_batch)
        ttft = self.ttft_ms(opt)
        return Metrics(
            weight_mem_gb=w_mem,
            kv_mem_gb=kv_mem,
            total_mem_gb=total_mem,
            throughput_tok_s=throughput,
            latency_ms_per_token=latency,
            ttft_ms=ttft,
            effective_batch=eff_batch,
            oom=oom,
        )

    # ---------------------------------------------------------------------
    # 5.5 瀑布叠加（waterfall）：顺序打开优化，逐级看收益
    # ---------------------------------------------------------------------
    def waterfall(self, stages: List[Tuple[str, OptimizationConfig]]) -> List[Dict]:
        """
        输入一串「(阶段名, 该阶段的完整 OptimizationConfig)」，
        输出每个阶段的指标 + 相对**上一阶段**的增量（delta），供瀑布图使用。

        典型顺序（回答「优化叠加到底赚多少」）：
            baseline(FP16, batch=1)
          → +量化 W4A16
          → +连续批处理 batch=32
          → +KV 量化 int8
        """
        rows: List[Dict] = []
        prev: Optional[Metrics] = None
        for stage_name, opt in stages:
            m = self.compute(opt)
            row = {"stage": stage_name, "config": opt, "metrics": m}
            if prev is not None:
                row["d_mem_gb"] = m.total_mem_gb - prev.total_mem_gb
                row["d_throughput"] = m.throughput_tok_s - prev.throughput_tok_s
                row["speedup_vs_prev"] = (
                    m.throughput_tok_s / prev.throughput_tok_s if prev.throughput_tok_s > 0 else float("inf")
                )
            else:
                row["d_mem_gb"] = 0.0
                row["d_throughput"] = 0.0
                row["speedup_vs_prev"] = 1.0
            rows.append(row)
            prev = m
        return rows


# =============================================================================
# 六、三大优化的「便捷变换」—— 把一个 OptimizationConfig 变成叠加了某优化后的新配置
# =============================================================================
def quantize(opt: OptimizationConfig, weight_dtype: str,
             activation_dtype: Optional[str] = None) -> OptimizationConfig:
    """叠加「权重量化」。W4A16 传 weight_dtype='int4'；W8A8 传两者都为 'int8'/'fp8'。"""
    return replace(
        opt,
        weight_dtype=weight_dtype,
        activation_dtype=activation_dtype if activation_dtype is not None else opt.activation_dtype,
    )


def continuous_batching(opt: OptimizationConfig, batch_size: int) -> OptimizationConfig:
    """叠加「连续批处理」，把 batch 调到目标值。"""
    return replace(opt, batch_size=batch_size)


def kv_quantize(opt: OptimizationConfig, kv_dtype: str) -> OptimizationConfig:
    """叠加「KV 缓存量化」。"""
    return replace(opt, kv_dtype=kv_dtype)


# =============================================================================
# 七、内置示例配置（供 demo / 测试复用）
# =============================================================================
def default_setup() -> Tuple[OptimizationCalculator, ModelConfig, HardwareConfig, WorkloadConfig]:
    """一套书里用过的默认组合：Llama-7B + A100-80G + (1k prompt, 512 gen)。"""
    model = ModelConfig()
    hw = HardwareConfig()
    wl = WorkloadConfig()
    return OptimizationCalculator(model, hw, wl), model, hw, wl


if __name__ == "__main__":
    # 快速自检：打印 baseline 与全优化的对比
    calc, model, hw, wl = default_setup()
    base = calc.compute(OptimizationConfig())
    full = calc.compute(OptimizationConfig(weight_dtype="int4", batch_size=32, kv_dtype="int8"))
    print("baseline:", base.as_row())
    print("full-opt:", full.as_row())
    print("吞吐提升 ×%.1f" % (full.throughput_tok_s / base.throughput_tok_s))
