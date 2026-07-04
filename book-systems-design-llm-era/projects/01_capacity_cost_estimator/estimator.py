# -*- coding: utf-8 -*-
"""
estimator.py —— LLM 系统「容量 & 成本」解析估算器（analytical capacity & cost model）

配套书：《Systems Design in the LLM Era》(Sampriti Mitra)
主题：给定 QPS、平均输入/输出 token、模型规模、目标 p99 延迟，
     解析地估算：所需 GPU 数、KV Cache/显存占用、带宽、$/月、$/百万 token。

设计哲学（第一性原理）：
    LLM 推理不是黑盒。它的成本 = 算力（FLOPs）+ 显存（weights + KV）+ 带宽（HBM 读）。
    这些量都能用「模型规模 × token 数」的简单公式**先验估出**，误差在 2× 以内。
    与其上线后被账单吓一跳，不如先用一张 Excel（或本模块）算清楚。

本模块**纯 Python + 标准库**（dataclasses / math / enum），不依赖 numpy/torch，
    保证离线、可移植、可被 pytest 精确验证每一条公式。

术语对照：
    QPS      = Queries Per Second，每秒请求数
    TTFT     = Time To First Token，首 token 时间（prefill 决定）
    TPOT     = Time Per Output Token，每输出 token 时间（decode 决定）
    KV Cache = Key/Value Cache，注意力缓存，显存第二大头
    HBM      = High Bandwidth Memory，GPU 显存
    MFU      = Model FLOPs Utilization，模型算力利用率
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ============================================================================
# 一、基础常量：单位换算与「第一性原理」系数
# ============================================================================

# 每秒 → 每月（按 30 天算）。成本估算按月出账，业界惯例。
SECONDS_PER_MONTH = 30 * 24 * 3600  # 2_592_000

# 每个参数每次前向传播的 FLOPs 系数。
# 第一性原理：一次矩阵乘 y = W·x，W 有 P 个元素，每个元素做「1 乘 + 1 加」= 2 FLOPs。
# 所以「前向 1 token」≈ 2·P FLOPs（P = 参数量）。这是 Kaplan/Chinchilla 都在用的经典近似。
FLOPS_PER_PARAM_PER_TOKEN_FWD = 2.0

# 字节数常量
BYTES_PER_GB = 1024 ** 3          # 1 GiB
BYTES_PER_TB = 1024 ** 4          # 1 TiB


class DType(Enum):
    """权重/KV 的数值精度。值 = 每个元素占的字节数（bytes/element）。

    这是显存估算的**核心旋钮**：FP16→INT8 显存直接砍半。
    """
    FP32 = 4   # 训练常用，推理很少用（太占显存）
    FP16 = 2   # 推理默认（bf16/fp16 都是 2 字节）
    BF16 = 2   # 与 FP16 同字节，数值范围更大，训练更稳
    FP8 = 1    # H100 起支持，推理量化前沿
    INT8 = 1   # 权重量化常用（GPTQ/AWQ 落地精度）
    INT4 = 0.5  # 激进量化，0.5 字节/元素（4 bit）


# ============================================================================
# 二、GPU 硬件规格表（GPU spec sheet）
# ============================================================================
# 只列**推理最相关**的三个量：FP16 算力（TFLOPS）、显存（GB）、显存带宽（TB/s）、单价（$/hr）。
# 单价用「按需云价」的粗略公开量级（不同云/合约差异大，改这里即可）。
# ⚠️ 坑：厂商标称 TFLOPS 常是「带稀疏（sparse）」的翻倍值，这里取稠密（dense）实际值。

@dataclass(frozen=True)
class GPUSpec:
    name: str
    fp16_tflops: float       # 稠密 FP16 算力，单位 TFLOPS（10^12 FLOPs/s）
    hbm_gb: float            # 显存容量，GB
    hbm_bandwidth_tbs: float  # 显存带宽，TB/s（decode 的真正瓶颈）
    hourly_cost_usd: float   # 按需单价，$/小时（含整机摊薄的粗略量级）


# 常见推理卡。数字为公开量级的近似，用于教学估算，非精确报价。
GPU_CATALOG: Dict[str, GPUSpec] = {
    "A100-40G": GPUSpec("A100-40G", fp16_tflops=312.0, hbm_gb=40.0,
                        hbm_bandwidth_tbs=1.555, hourly_cost_usd=1.8),
    "A100-80G": GPUSpec("A100-80G", fp16_tflops=312.0, hbm_gb=80.0,
                        hbm_bandwidth_tbs=2.039, hourly_cost_usd=2.5),
    "H100-80G": GPUSpec("H100-80G", fp16_tflops=989.0, hbm_gb=80.0,
                        hbm_bandwidth_tbs=3.35, hourly_cost_usd=4.0),
    "L40S-48G": GPUSpec("L40S-48G", fp16_tflops=362.0, hbm_gb=48.0,
                        hbm_bandwidth_tbs=0.864, hourly_cost_usd=1.2),
}


# ============================================================================
# 三、模型规格（model spec）
# ============================================================================
# KV Cache 大小取决于「层数 × KV 头数 × head_dim」，不能只看总参数量。
# 这里把 Transformer 的关键结构参数显式建模，才能算准 KV。

@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_params_b: float      # 总参数量，单位「十亿」（Billion）
    num_layers: int          # Transformer 层数 L
    hidden_dim: int          # 隐藏维度 d_model
    num_q_heads: int         # Query 头数
    num_kv_heads: int        # KV 头数（GQA/MQA 时 < num_q_heads，直接砍 KV 显存！）
    head_dim: Optional[int] = None  # 每头维度，默认 hidden_dim / num_q_heads

    def resolved_head_dim(self) -> int:
        """若未显式给出 head_dim，则用 hidden_dim / num_q_heads 推出。"""
        if self.head_dim is not None:
            return self.head_dim
        # ⚠️ 坑：hidden_dim 未必能被 num_q_heads 整除（少见），这里向下取整并断言常见情况
        return self.hidden_dim // self.num_q_heads


# 几个代表性模型的近似结构参数（用于教学估算）。
MODEL_CATALOG: Dict[str, ModelSpec] = {
    # Llama-3-8B：GQA，8 个 KV 头（Q 头 32），KV 显存只有 MHA 的 1/4
    "Llama3-8B": ModelSpec("Llama3-8B", num_params_b=8.0, num_layers=32,
                           hidden_dim=4096, num_q_heads=32, num_kv_heads=8),
    # Llama-3-70B：80 层，GQA 8 KV 头
    "Llama3-70B": ModelSpec("Llama3-70B", num_params_b=70.0, num_layers=80,
                            hidden_dim=8192, num_q_heads=64, num_kv_heads=8),
    # Qwen2-7B：类似 Llama-8B 结构
    "Qwen2-7B": ModelSpec("Qwen2-7B", num_params_b=7.6, num_layers=28,
                          hidden_dim=3584, num_q_heads=28, num_kv_heads=4),
    # 一个「假想 MHA 13B」用于对比 GQA 省显存效果（KV 头 = Q 头）
    "MHA-13B": ModelSpec("MHA-13B", num_params_b=13.0, num_layers=40,
                         hidden_dim=5120, num_q_heads=40, num_kv_heads=40),
}


# ============================================================================
# 四、工作负载（workload）与 SLA
# ============================================================================

@dataclass
class Workload:
    """描述线上流量特征与服务目标。"""
    qps: float                    # 每秒请求数
    avg_input_tokens: int         # 平均输入（prompt）token 数 —— 决定 prefill 成本
    avg_output_tokens: int        # 平均输出（generation）token 数 —— 决定 decode 成本
    target_p99_ms: float = 2000.0  # 目标 p99 端到端延迟，毫秒。默认 2s。

    def total_tokens_per_req(self) -> int:
        """单请求处理的总 token（输入 + 输出）。"""
        return self.avg_input_tokens + self.avg_output_tokens


# ============================================================================
# 五、估算结果容器
# ============================================================================

@dataclass
class Estimate:
    """一次估算的完整结果。字段名即物理量，便于 pytest 断言与报表打印。"""
    # —— 显存（单卡视角）——
    weight_gb: float                 # 权重显存
    kv_per_token_kb: float           # 每 token 的 KV 大小（KB）
    kv_per_request_gb: float         # 单请求满上下文时的 KV 显存（GB）
    kv_concurrent_gb: float          # 所有并发请求的 KV 总显存（GB）
    total_gpu_mem_needed_gb: float   # 权重 + 并发 KV + 激活裕量，总显存需求

    # —— 并发与吞吐 ——
    concurrency: float               # 系统内并发请求数（Little 定律）
    decode_tokens_per_s_per_gpu: float  # 单卡 decode 吞吐（tokens/s）
    prefill_tokens_per_s_per_gpu: float  # 单卡 prefill 吞吐（tokens/s）

    # —— GPU 数量（两条独立约束取 max）——
    gpus_for_compute: float          # 算力约束下所需卡数
    gpus_for_memory: float           # 显存约束下所需卡数
    num_gpus: int                    # 实际需要的卡数 = ceil(max(两者))

    # —— 带宽 ——
    hbm_read_gbps_per_gpu: float     # 单卡 decode 时的 HBM 读带宽需求（GB/s）
    bottleneck: str                  # "compute" / "memory" / "bandwidth"

    # —— 成本 ——
    cost_per_hour_usd: float         # 集群每小时成本
    cost_per_month_usd: float        # 集群每月成本
    cost_per_million_tokens_usd: float  # 每百万 token 成本（$/1M tok）—— 最常用的对外报价单位

    # —— 元信息 ——
    gpu: str = ""
    model: str = ""
    notes: List[str] = field(default_factory=list)


# ============================================================================
# 六、核心公式函数（每个都可被 pytest 单独验证）
# ============================================================================

def weight_memory_gb(model: ModelSpec, dtype: DType) -> float:
    """权重显存（GB）= 参数量 × 每参数字节数。

    第一性原理：P 个参数，每个占 `dtype.value` 字节 → 总字节数 / 2^30。
    例：8B 参数 × FP16(2B) = 16 GB。这就是「7B 模型至少要 ~14-16GB 显存」的由来。
    """
    total_bytes = model.num_params_b * 1e9 * dtype.value
    return total_bytes / BYTES_PER_GB


def kv_cache_bytes_per_token(model: ModelSpec, kv_dtype: DType) -> float:
    """每个 token 的 KV Cache 大小（字节）。

    公式（本模块最重要的公式之一）：
        kv_per_token = 2 × L × num_kv_heads × head_dim × bytes_per_elem
    逐项解释：
        2          —— K 和 V 两份缓存
        L          —— 每一层都要缓存
        num_kv_heads × head_dim —— 一层里 KV 的宽度（GQA/MQA 时用 kv_heads，不是 q_heads！）
        bytes_per_elem —— 精度字节数

    ⚠️ 最常见的错：用 num_q_heads 而非 num_kv_heads。GQA 下二者差 4-8 倍，
       算错直接高估显存好几倍。这也是面试高频陷阱。
    """
    head_dim = model.resolved_head_dim()
    return 2 * model.num_layers * model.num_kv_heads * head_dim * kv_dtype.value


def little_law_concurrency(qps: float, latency_s: float) -> float:
    """Little 定律：系统内平均并发数 L = λ × W。

    λ = 到达率（QPS），W = 每个请求在系统里停留的时间（延迟，秒）。
    直觉：每秒来 100 个请求，每个待 2 秒 → 任一时刻系统里平均有 200 个在处理。
    这是**容量规划的第一公式**：并发数决定了你要为多少「同时在飞」的请求预留 KV 显存。
    """
    return qps * latency_s


def decode_throughput_per_gpu(gpu: GPUSpec, model: ModelSpec, dtype: DType,
                              mfu: float = 0.4, batch_size: int = 32) -> float:
    """单卡 decode 阶段**聚合**吞吐（tokens/s），已计入连续批处理（continuous batching）。

    第一性原理：decode 是**显存带宽瓶颈（memory-bound）**，不是算力瓶颈！
    因为 decode 每步每个序列只生成 1 个 token，算术强度（arithmetic intensity）低，
    真正的限制是「把整个模型权重从 HBM 读一遍」的时间。

        batch=1 时：每步读一遍权重、产 1 token → 上限 = bandwidth / weight_bytes tok/s
        batch=B 时：一次权重读被 B 个序列**共享**（这正是 vLLM 连续批处理的精髓），
                    每步读一遍权重、产 B 个 token → 上限 ≈ B × bandwidth / weight_bytes

    所以 decode 吞吐 ≈ batch_size × (带宽 / 权重字节) × mfu。
    ⚠️ 坑：不能无限加 batch —— batch 越大 KV 显存越吃紧（见显存约束），
       且总有一刻会转成算力瓶颈。这里用固定 batch_size 给量级估计，真实系统动态调度。

    第一性原理小结：**batching 把「带宽墙」摊薄，是让 GPU 便宜的头号手段。**
    """
    weight_bytes = model.num_params_b * 1e9 * dtype.value
    bandwidth_bytes_per_s = gpu.hbm_bandwidth_tbs * BYTES_PER_TB
    # 单序列上限 × batch × 利用率
    per_seq = bandwidth_bytes_per_s / weight_bytes
    return per_seq * batch_size * mfu


def prefill_throughput_per_gpu(gpu: GPUSpec, model: ModelSpec,
                               mfu: float = 0.4) -> float:
    """单卡 prefill 阶段吞吐（tokens/s）。

    第一性原理：prefill 是**算力瓶颈（compute-bound）**。
    一次处理整个 prompt（几百上千 token），矩阵很「胖」，能吃满 GPU 算力。
        处理 1 token 需 2·P FLOPs → 吞吐 = 算力 / (2·P)
    乘 mfu（典型 0.3~0.5）修正实际利用率。
    """
    flops_per_token = FLOPS_PER_PARAM_PER_TOKEN_FWD * model.num_params_b * 1e9
    peak_flops = gpu.fp16_tflops * 1e12
    return (peak_flops / flops_per_token) * mfu


def estimate(
    workload: Workload,
    model: ModelSpec,
    gpu: GPUSpec,
    weight_dtype: DType = DType.FP16,
    kv_dtype: DType = DType.FP16,
    mfu: float = 0.4,
    mem_overhead_frac: float = 0.1,
    max_kv_util: float = 0.9,
    batch_size: int = 32,
) -> Estimate:
    """把上面所有公式串起来，输出一次完整的容量+成本估算。

    参数：
        weight_dtype     权重精度（默认 FP16）
        kv_dtype         KV 精度（默认 FP16，量化到 INT8 可省一半 KV 显存）
        mfu              算力/带宽利用率修正（0~1，典型 0.3~0.5）
        mem_overhead_frac 激活/碎片等额外显存裕量占比（默认 10%）
        max_kv_util      单卡显存里最多留给 KV 的比例上限（默认 90%，其余给权重+激活）
        batch_size       decode 连续批处理的批大小（摊薄带宽墙，默认 32）

    返回 Estimate。整个函数**无副作用、纯计算**，方便测试。
    """
    notes: List[str] = []

    # ---- 1) 显存：权重 ----
    w_gb = weight_memory_gb(model, weight_dtype)

    # ---- 2) 显存：KV ----
    kv_per_tok_bytes = kv_cache_bytes_per_token(model, kv_dtype)
    kv_per_tok_kb = kv_per_tok_bytes / 1024.0
    # 单请求满上下文（输入+输出）的 KV 显存
    kv_per_req_gb = kv_per_tok_bytes * workload.total_tokens_per_req() / BYTES_PER_GB

    # ---- 3) 并发数（Little 定律）----
    latency_s = workload.target_p99_ms / 1000.0
    concurrency = little_law_concurrency(workload.qps, latency_s)

    # 所有并发请求的 KV 总显存
    kv_concurrent_gb = kv_per_req_gb * concurrency

    # ---- 4) 吞吐（单卡）----
    dec_tps = decode_throughput_per_gpu(gpu, model, weight_dtype, mfu, batch_size)
    pre_tps = prefill_throughput_per_gpu(gpu, model, mfu)

    # ---- 5) GPU 数量：两条独立约束 ----
    # (a) 算力/吞吐约束：全系统每秒要产出的 token / 单卡吞吐
    # 输出 token 走 decode，输入 token 走 prefill，各自算需求再取「等效卡数」
    output_tps_needed = workload.qps * workload.avg_output_tokens
    input_tps_needed = workload.qps * workload.avg_input_tokens
    # 单卡在 decode 上的时间占比与 prefill 时间占比之和 ≤ 1，用「工作量/单卡产能」相加近似
    gpus_for_decode = output_tps_needed / dec_tps if dec_tps > 0 else float("inf")
    gpus_for_prefill = input_tps_needed / pre_tps if pre_tps > 0 else float("inf")
    gpus_for_compute = gpus_for_decode + gpus_for_prefill

    # (b) 显存约束：权重每卡都要放一份；KV 总量除以「每卡可用于 KV 的显存」
    kv_budget_per_gpu = gpu.hbm_gb * max_kv_util - w_gb
    if kv_budget_per_gpu <= 0:
        # 权重都放不下 → 需要模型并行（张量并行/流水并行），这里给出提示
        notes.append(
            f"⚠️ 权重 {w_gb:.1f}GB 超过单卡可用显存，需张量并行（TP）拆分，"
            f"本估算按至少能放下处理，请增大显卡或量化。"
        )
        # 退化处理：至少要能放权重，令每卡 KV 预算为一个很小的正数避免除零
        kv_budget_per_gpu = max(gpu.hbm_gb * 0.05, 0.1)
        gpus_for_weight = math.ceil(w_gb / (gpu.hbm_gb * max_kv_util))
    else:
        gpus_for_weight = 1
    gpus_for_kv = kv_concurrent_gb / kv_budget_per_gpu if kv_budget_per_gpu > 0 else float("inf")
    gpus_for_memory = max(float(gpus_for_weight), gpus_for_kv)

    # (c) 实际卡数 = ceil(max(算力约束, 显存约束))，至少 1 张
    raw = max(gpus_for_compute, gpus_for_memory)
    num_gpus = max(1, math.ceil(raw))

    # ---- 6) 瓶颈判定 ----
    if gpus_for_memory > gpus_for_compute:
        bottleneck = "memory"
    else:
        bottleneck = "compute"

    # ---- 7) 显存总需求（单卡视角，用于报表）----
    # 把并发 KV 摊到 num_gpus 张卡上，加权重和激活裕量
    kv_per_gpu = kv_concurrent_gb / num_gpus
    total_mem = w_gb + kv_per_gpu
    total_mem *= (1 + mem_overhead_frac)

    # ---- 8) 带宽：decode 每卡每秒读多少 HBM ----
    # 关键：连续批处理下，一次 decode step 读一遍权重、产 batch_size 个 token。
    # 所以每秒的 step 数 = dec_tps / batch_size，每 step 读整份权重。
    # 这样 hbm_read 会逼近 GPU 标称带宽 × mfu —— 印证「decode 是带宽瓶颈」。
    weight_bytes = model.num_params_b * 1e9 * weight_dtype.value
    steps_per_s = dec_tps / batch_size if batch_size > 0 else dec_tps
    hbm_read_bytes_per_s = steps_per_s * weight_bytes
    hbm_read_gbps = hbm_read_bytes_per_s / BYTES_PER_GB

    # ---- 9) 成本 ----
    cost_per_hour = num_gpus * gpu.hourly_cost_usd
    cost_per_month = cost_per_hour * (SECONDS_PER_MONTH / 3600.0)
    # 每百万 token 成本：月总成本 / 月总 token × 1e6
    total_tokens_per_s = workload.qps * workload.total_tokens_per_req()
    total_tokens_per_month = total_tokens_per_s * SECONDS_PER_MONTH
    if total_tokens_per_month > 0:
        cost_per_million = cost_per_month / total_tokens_per_month * 1e6
    else:
        cost_per_million = float("inf")

    return Estimate(
        weight_gb=w_gb,
        kv_per_token_kb=kv_per_tok_kb,
        kv_per_request_gb=kv_per_req_gb,
        kv_concurrent_gb=kv_concurrent_gb,
        total_gpu_mem_needed_gb=total_mem,
        concurrency=concurrency,
        decode_tokens_per_s_per_gpu=dec_tps,
        prefill_tokens_per_s_per_gpu=pre_tps,
        gpus_for_compute=gpus_for_compute,
        gpus_for_memory=gpus_for_memory,
        num_gpus=num_gpus,
        hbm_read_gbps_per_gpu=hbm_read_gbps,
        bottleneck=bottleneck,
        cost_per_hour_usd=cost_per_hour,
        cost_per_month_usd=cost_per_month,
        cost_per_million_tokens_usd=cost_per_million,
        gpu=gpu.name,
        model=model.name,
        notes=notes,
    )


# ============================================================================
# 七、便捷函数：用名字直接估算 + 打印报表
# ============================================================================

def estimate_by_name(
    qps: float,
    avg_input_tokens: int,
    avg_output_tokens: int,
    model_name: str = "Llama3-8B",
    gpu_name: str = "A100-80G",
    target_p99_ms: float = 2000.0,
    weight_dtype: DType = DType.FP16,
    kv_dtype: DType = DType.FP16,
    mfu: float = 0.4,
    batch_size: int = 32,
) -> Estimate:
    """按目录名快速估算。给不想构造 dataclass 的调用方用。"""
    if model_name not in MODEL_CATALOG:
        raise KeyError(f"未知模型 {model_name}，可选：{list(MODEL_CATALOG)}")
    if gpu_name not in GPU_CATALOG:
        raise KeyError(f"未知 GPU {gpu_name}，可选：{list(GPU_CATALOG)}")
    workload = Workload(qps=qps, avg_input_tokens=avg_input_tokens,
                        avg_output_tokens=avg_output_tokens,
                        target_p99_ms=target_p99_ms)
    return estimate(workload, MODEL_CATALOG[model_name], GPU_CATALOG[gpu_name],
                    weight_dtype=weight_dtype, kv_dtype=kv_dtype, mfu=mfu,
                    batch_size=batch_size)


def format_report(est: Estimate) -> str:
    """把 Estimate 渲染成一份人类可读的中文报表字符串。"""
    lines = [
        "=" * 60,
        f"  LLM 容量 & 成本估算报表",
        f"  模型: {est.model}   GPU: {est.gpu}",
        "=" * 60,
        f"[显存]",
        f"  权重显存           : {est.weight_gb:8.2f} GB",
        f"  KV / token         : {est.kv_per_token_kb:8.2f} KB",
        f"  KV / 请求(满上下文) : {est.kv_per_request_gb:8.3f} GB",
        f"  并发 KV 总量       : {est.kv_concurrent_gb:8.2f} GB",
        f"  单卡总显存需求     : {est.total_gpu_mem_needed_gb:8.2f} GB",
        f"[并发 & 吞吐]",
        f"  并发数(Little 定律): {est.concurrency:8.1f} 个在飞请求",
        f"  单卡 decode 吞吐   : {est.decode_tokens_per_s_per_gpu:8.1f} tok/s",
        f"  单卡 prefill 吞吐  : {est.prefill_tokens_per_s_per_gpu:8.1f} tok/s",
        f"[GPU 数量]",
        f"  算力约束需卡数     : {est.gpus_for_compute:8.2f}",
        f"  显存约束需卡数     : {est.gpus_for_memory:8.2f}",
        f"  >>> 实际需 GPU 数  : {est.num_gpus:8d}  (瓶颈: {est.bottleneck})",
        f"[带宽]",
        f"  单卡 HBM 读带宽    : {est.hbm_read_gbps_per_gpu:8.1f} GB/s",
        f"[成本]",
        f"  每小时             : ${est.cost_per_hour_usd:10.2f}",
        f"  每月               : ${est.cost_per_month_usd:12.2f}",
        f"  每百万 token       : ${est.cost_per_million_tokens_usd:10.4f} / 1M tok",
        "=" * 60,
    ]
    for n in est.notes:
        lines.append(n)
    return "\n".join(lines)


if __name__ == "__main__":
    # 直接跑本文件 → 打印一个示例报表
    e = estimate_by_name(qps=50, avg_input_tokens=1024, avg_output_tokens=256,
                         model_name="Llama3-8B", gpu_name="A100-80G")
    print(format_report(e))
