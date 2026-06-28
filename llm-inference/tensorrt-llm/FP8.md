# TensorRT-LLM FP8 量化推理

> 用 8 位浮点（FP8）做大模型推理加速：在 Hopper/Ada/Blackwell GPU 上把权重与激活压到 1 字节，借硬件 FP8 Tensor Core 把吞吐和显存一起打下来。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/tensorrt-llm/README]] [[llm-compression/quantization/量化基础]] [[llm-inference/tensorrt-llm/TRT-LLM引擎构建参数]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

## 阅读地图

| 章节 | 你会得到什么 |
| --- | --- |
| 0. 一句话锚点 | FP8 在 TRT-LLM 里到底是什么 |
| 1. 地基 | 推理为什么缺显存/缺算力，低精度解决什么 |
| 2. FP8 数制 | E4M3 / E5M2 两种格式从比特讲起 |
| 3. 缩放 | scale / amax 为什么是 FP8 的命门 |
| 4. 硬件 | Hopper FP8 Tensor Core 为什么快一倍 |
| 5. TRT-LLM 流程 | 校准 → 量化 → 构建引擎 → 部署的调用链 |
| 6. FP8 vs INT8/INT4 | 怎么选精度 |
| 配置示例 | quant 配置项各做什么（讲含义，不背默认值） |
| 常见坑 | 表格速查 |

## 0. 一句话锚点

**FP8 推理 = 把模型里"做矩阵乘法"的张量从 16 位浮点（FP16/BF16）换成 8 位浮点（FP8），让数据少占一半显存、让 GPU 的 FP8 Tensor Core 以接近两倍的速度算 GEMM，同时用"缩放因子"把精度损失压到几乎无感。**

在 TensorRT-LLM 里，FP8 是一种 **量化（quantization）模式**：构建引擎时指定 `fp8`，引擎内部的线性层就会用 FP8 计算。它属于 NVIDIA 主推的"Hopper 之后才好用"的精度，需要硬件支持 FP8 指令。

> 注意：本文讲机制与权衡。具体 CLI 参数名、默认值、支持矩阵以官方文档（`nvidia.github.io/TensorRT-LLM`、TensorRT-Model-Optimizer）和当前版本源码为准——这些细节随版本变化很快，不要背。

## 1. 地基：推理为什么要低精度

一个推理请求的瓶颈通常落在两件事上：

```
            ┌────────────── 推理瓶颈 ──────────────┐
            │                                      │
   Prefill（处理 prompt）              Decode（逐 token 生成）
   计算密集：大 GEMM 吃满算力        访存密集：每生成 1 token
   → 受 算力(FLOPS) 限              都要把全部权重从显存搬到 SM
                                    → 受 显存带宽(GB/s) 限
```

- **显存容量**：70B 模型用 FP16 存权重要 ~140 GB，单卡放不下。每个参数省 1 字节就是省一半显存。
- **显存带宽**：Decode 阶段每生成一个 token 都要把权重读一遍，权重越小，搬得越快，token/s 越高。
- **算力**：Prefill / 大 batch 时 GEMM 吃满 Tensor Core，低精度的 Tensor Core 吞吐更高。

FP8 **同时**缓解这三点：权重 1 字节（容量↓、带宽↓），FP8 Tensor Core 吞吐翻倍（算力↑）。这是它比 FP16 香、又比 INT4 通用的原因。

## 2. FP8 的数制：E4M3 与 E5M2 从比特讲起

浮点数 = 符号位 S + 指数位 E + 尾数位 M。8 位放不下太多，于是有两种切法（OCP / NVIDIA 标准）：

```
 E4M3  (动态范围小、精度高)        E5M2  (动态范围大、精度低)
 ┌─┬───────┬─────────┐            ┌─┬─────────┬───────┐
 │S│ E(4)  │  M(3)   │            │S│  E(5)   │ M(2)  │
 └─┴───────┴─────────┘            └─┴─────────┴───────┘
  1   4        3                    1     5       2
 ≈ ±448 量级，分辨更细            ≈ ±57344 量级，更抗溢出
```

直觉：
- **E4M3**：指数少、尾数多 → 表示的最大值小（约 ±448），但同一区间内"刻度"更密，**数值更准**。适合**前向激活与权重**（值域不大但要精度）。
- **E5M2**：指数多、尾数少 → 能表示很大/很小的数，但刻度粗，**更抗溢出**。常用于**梯度**（训练场景动态范围大），推理前向用得少。

推理（TRT-LLM 主战场）几乎都用 **E4M3** 做权重和激活。一个对比：

| 格式 | 位宽 | 近似最大正规值 | 相对精度 | 典型用途 |
| --- | --- | --- | --- | --- |
| FP16 | 16 | ~65504 | 高 | 通用 baseline |
| BF16 | 16 | ~3.4e38 | 中（尾数7位） | 训练/通用 |
| FP8 E4M3 | 8 | ~448 | 低 | 推理权重+激活 |
| FP8 E5M2 | 8 | ~57344 | 更低 | 梯度/特殊层 |

**核心矛盾**：E4M3 最大才 ~448，可真实激活动辄上千。直接塞进去就溢出/截断 → 这就引出第 3 节的缩放。

## 3. 缩放（Scaling）：FP8 能不能用的命门

因为 FP8 表示范围窄，必须先把张量"缩放"到 FP8 能容纳的区间，算完再缩放回去。这叫 **per-tensor scaling**（每个张量一个缩放因子）。

设原张量 $X$，缩放因子 $s$，量化与反量化：

$$X_{fp8} = \text{round\_to\_fp8}(X \cdot s), \qquad \hat{X} = X_{fp8} / s$$

$s$ 怎么定？让张量的最大绝对值 $\text{amax}$ 刚好落到 FP8 最大值附近：

$$s = \frac{\text{FP8\_MAX}}{\text{amax}(X)}, \quad \text{FP8\_MAX(E4M3)} \approx 448$$

- $s$ 太小 → 大量小值被压成 0（**下溢**，损失信息）。
- $s$ 太大 → 大值超过 448 被截断（**上溢**，更致命）。

**amax 从哪来？** 这是"静态量化 vs 动态量化"的分水岭：

```
权重（离线已知）            激活（运行时才知道）
   │                          │
   │ 量化时直接算 amax        │ 方案A 校准(calibration)：
   ▼                          │   推理前喂少量样本，统计每层激活 amax
 静态 scale 写进引擎          │   → 静态 scale，零运行时开销  ← TRT-LLM 常用
                              │ 方案B 动态：每次 forward 现算 amax
                              │   → 更准但有额外开销
```

一次 FP8 GEMM 的实际数据流：

```
  A(fp8) ──┐
           ├─► FP8 Tensor Core 累加(fp32) ──► 乘 (1/sA·1/sB) ──► 输出(fp16/bf16)
  B(fp8) ──┘        ↑                              ↑
              累加器用高精度避免误差         反缩放回真实尺度
```

关键点：**乘法在 FP8、累加在 FP32**。这样既享受 FP8 的速度，又不让长求和把误差滚雪球。

## 4. 硬件地基：FP8 Tensor Core 为什么快

FP8 不是软件模拟，而是 GPU 里 **Tensor Core 的原生指令**。能用 FP8 计算的卡：

| 架构 | 代表卡 | FP8 支持 |
| --- | --- | --- |
| Hopper | H100 / H200 | ✅ 原生（含 E4M3/E5M2） |
| Ada Lovelace | L40S / L4 / RTX 4090 | ✅ 原生 |
| Blackwell | B200 / GB200 | ✅ 原生（并新增更低精度） |
| Ampere | A100 / A30 | ❌ 无 FP8（只能 INT8/FP16） |

> 这是 NVIDIA 文档把 FP8 称作 "FP8 (Hopper)" 的原因——A100 用不了 FP8，强行指定会回退或报错。

为什么快？Tensor Core 一条指令算一小块矩阵乘加。位宽减半，单位时间能塞进去的元素翻倍，**FP8 的峰值算力大致是 FP16 的两倍**（具体倍数与累加精度、卡型有关，以官方白皮书为准）。再叠加显存占用减半带来的带宽收益，Decode 端往往也明显加速。

```
   同一块 Tensor Core，同一个时钟周期：
   FP16:  [■■■■]            处理 N 个 MAC
   FP8 :  [■■■■■■■■]        处理 ~2N 个 MAC   ← 吞吐≈2×
```

## 5. TRT-LLM 的 FP8 端到端流程

TensorRT-LLM 把 FP8 的量化职责交给 **TensorRT-Model-Optimizer（ModelOpt）** 工具，再由 TRT-LLM 构建引擎。整体调用链（具体命令名以版本文档为准）：

```
 HF 权重(fp16/bf16)
        │
        ▼
 ① 量化/校准 (ModelOpt)
    - 加载模型 + 少量校准数据集（如几十~几百条样本）
    - 跑前向，逐层统计激活 amax
    - 算出权重/激活的 per-tensor scale
    - 导出"带 FP8 量化信息"的 checkpoint
        │
        ▼
 ② 构建引擎 (trtllm-build / build API)
    - 指定 dtype=fp8（量化模式）
    - 线性层算子选 FP8 kernel，融合 scale/反scale
    - 产出 .engine（已含 scale，运行时零校准）
        │
        ▼
 ③ 运行时 (TRT-LLM Runtime / Triton)
    - KV Cache 可独立选 FP8（见下）
    - 加载引擎，正常推理
```

**三个可以独立选 FP8 的地方**（理解它们是分开的，很重要）：

```
 ┌──────────────────────────────────────────────┐
 │ 1) GEMM 权重+激活 FP8  → 省显存、加速计算       │
 │ 2) KV Cache FP8        → 省长上下文显存、加带宽 │
 │ 3) （部分层保留 FP16） → 敏感层不量化保精度     │
 └──────────────────────────────────────────────┘
```

- **GEMM FP8**：主收益来源，线性层用 FP8。
- **FP8 KV Cache**：长序列时 KV Cache 可能比权重还占显存，单独把它压成 FP8 收益巨大，且与 GEMM FP8 正交，可单开。
- **混合精度**：有些层（如 LM head、LayerNorm、个别对量化敏感的层）保留高精度，是精度与速度的折中。

## 配置示例与参数说明（讲含义，不背默认值）

下面是"概念层面"的量化配置（字段名以 ModelOpt / trtllm-build 当前文档为准，这里讲每项**干什么、怎么权衡**）：

| 配置项（语义） | 作用 | 怎么权衡 |
| --- | --- | --- |
| 量化算法 = FP8 | 选 FP8（E4M3）做权重+激活 | 对 Hopper+；精度/速度均衡的首选 |
| KV Cache dtype = FP8 | KV Cache 单独用 FP8 存 | 长上下文/大并发时显存收益最大 |
| 校准数据集 | 提供统计激活 amax 的样本 | 选**贴近实际业务分布**的样本，几十~几百条通常够；样本偏了 scale 就偏 |
| 校准样本数 | 跑多少条算 amax | 太少 amax 估不准，太多浪费时间，边际收益递减 |
| 排除层 / 跳过量化的层 | 指定哪些层保持高精度 | LM head、敏感层保 FP16 可救精度，代价是这些层不提速 |
| per-tensor / per-channel | scale 的粒度 | 更细粒度更准但更复杂；FP8 常用 per-tensor |

**实践节奏**：先全 FP8 跑通 → 用评测集（如 perplexity、下游任务准确率）对比 FP16 baseline → 若掉点明显，逐步把敏感层排除出量化集 / 改进校准数据，直到精度可接受。先 measure，再 tune。

一个**显存数值例子**（直观感受收益，非精确）：13B 模型权重，FP16 约 26 GB；FP8 约 13 GB。再把 KV Cache 也转 FP8，长上下文场景下 KV 显存约减半 → 同样的卡能塞更长上下文 / 更大 batch。

## 常见问题 / 坑

| 现象 / 问题 | 原因 | 应对 |
| --- | --- | --- |
| 指定 fp8 报错/回退 | 卡是 A100 等无 FP8 硬件 | 换 Hopper/Ada/Blackwell，或改用 INT8 |
| 量化后精度明显掉点 | 校准数据不代表真实分布；敏感层被量化 | 换更贴业务的校准集；把敏感层排除出量化 |
| 输出出现 NaN/异常大值 | 激活上溢（超过 ~448）/ scale 估偏 | 检查校准；个别层保 FP16；核对 amax 统计 |
| 加速不明显 | batch/序列太小，落在访存或非 GEMM 部分 | 看是否 Decode 受限；考虑 FP8 KV Cache；增大 batch |
| 显存没省多少 | 只量化了 GEMM，KV Cache 仍是 FP16 | 单独开启 FP8 KV Cache |
| 把校准当成训练 | 误以为 FP8 量化需要重训 | FP8 是**训练后量化(PTQ)**，只需少量样本做校准，不更新权重 |
| 期望 FP8 = INT8 行为 | FP8 是浮点、有指数位，缩放/范围逻辑不同 | 按浮点缩放理解（amax→scale），别套 INT8 的 zero-point 思路 |
| 跨版本配置名对不上 | TRT-LLM/ModelOpt 接口迭代快 | 永远以**当前版本官方文档/源码**为准 |

## 🔗 跳转链接

- 返回知识地图：[[00-知识地图]]
- TRT-LLM 总览：[[llm-inference/tensorrt-llm/README]]
- 引擎构建参数：[[llm-inference/tensorrt-llm/TRT-LLM引擎构建参数]]
- 量化原理打底：[[llm-compression/quantization/量化基础]]
- 量化专题总览：[[llm-compression/README]]
- KV Cache 优化：[[llm-optimizer/kv-cache]]
- 推理性能指标：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

> 官方权威参考：
> - TensorRT-LLM Precision（FP8/Hopper）：https://nvidia.github.io/TensorRT-LLM/reference/precision.html#fp8-hopper
> - TensorRT-Model-Optimizer：https://nvidia.github.io/TensorRT-Model-Optimizer/
