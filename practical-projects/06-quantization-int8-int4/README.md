# 从零实现权重量化(INT8 / INT4)

一个**端到端、CPU 几十秒可跑通、不依赖外部数据集/网络**的动手项目:
用 `numpy` + `torch`(CPU)在一个 toy 小 MLP 上,把"权重量化"的核心原理从最底层走一遍。
配合 llm-action 的[量化文档](../../llm-compression/quantization)一起看,可以把"读懂概念"和"亲手跑通"打通。

---

## 一、演示什么原理(简明)

量化 = 用**更少的比特**表示原本 FP32/FP16 的权重,换取**更小的存储 + 更高的带宽利用率**。
本项目聚焦最常用的**对称量化(symmetric quantization)**,核心只有三个公式:

```
量化:   q     = round(clip(w / scale, -Qmax, +Qmax))     # 浮点 -> 整数
反量化: w_hat = q * scale                                 # 整数 -> 近似浮点
scale  = absmax / Qmax,  Qmax = 2^(bits-1) - 1           # INT8=127, INT4=7
```

- **scale** 是桥梁:它决定浮点范围如何映射到有限的整数格点。`absmax` 取得越"局部",误差越小。
- **量化误差** = `w` 与反量化后 `w_hat` 的差。位宽越低、分组越粗,误差越大,但压缩比越高 —— 这就是**精度 vs 压缩**的权衡。

代码实现并对比了三种 scale 粒度:

| 方案 | scale 粒度 | 典型用途 |
|------|-----------|---------|
| **per-tensor absmax** | 整张权重 1 个 scale | 最简单,误差最大 |
| **per-channel(逐行)** | 每个输出通道 1 个 scale | LLM INT8 权重量化的标配,误差大幅下降 |
| **group-wise 分组** | 每行切成小组,每组 1 个 scale | INT4(GPTQ/AWQ 等)的常用布局,4bit 下压住误差 |

度量指标:**MSE**(越小越好)、**余弦相似度**(越接近 1 越好)、**压缩比**(越大越省),
以及一次 toy 前向中"量化前 vs 量化后"的输出差异 —— 把权重误差传导到真正关心的模型输出。

---

## 二、怎么跑

```bash
cd practical-projects/06-quantization-int8-int4
python quantization.py
```

环境:Python 3.13 / numpy 2.3 / torch 2.12(CPU)。无需 GPU、数据集或联网。
若装了 `matplotlib`,会额外存一张误差柱状图 `quant_error.png`;没装也不影响主流程(自动降级为纯文本)。

---

## 三、预期输出(真实跑通摘录)

```
[A] quantization error & compression on fc1.weight (FP32 baseline)
method         bits           MSE      cosine   compress
--------------------------------------------------------
per_tensor        8     8.115e-08    0.999992      4.00x
per_channel       8     7.751e-08    0.999992      3.88x
per_tensor        4     2.643e-05    0.997466      8.00x
per_channel       4     2.528e-05    0.997579      7.53x
group/g32         4     2.403e-05    0.997696      7.11x
group/g16         4     2.207e-05    0.997888      6.40x

[B] end-to-end forward: FP32 output vs quantized-weight output
scheme                       out-MSE    out-cosine     max|dy|
--------------------------------------------------------------
INT8 per_tensor            1.824e-06      0.999986      0.0043
INT8 per_channel           1.773e-06      0.999987      0.0056
INT4 per_channel           6.381e-04      0.995316      0.0948
INT4 group=32              5.970e-04      0.995377      0.0727

[summary]
  - INT8 per_channel: near-lossless (cosine ~1.0, tiny MSE).
  - INT4 error is larger but group-wise quant brings it back down.
  - finer groups -> lower error, slightly more scale storage.
  - compression ratio grows as bits drop (FP32 -> INT8 ~4x, INT4 ~8x).

DONE. Quantization demo finished successfully.
```

**怎么看"成功":**
- **INT8 近乎无损**:余弦 ~0.99999、MSE ~1e-8 级,前向输出几乎不变(`max|dy|` 仅 ~0.004)。
- **INT4 误差明显放大**(MSE 从 1e-8 跳到 1e-5,前向 out-MSE 从 1e-6 跳到 ~6e-4),但仍能用;
- **更细的粒度更准**:`group=16` 比 `group=32` 比 per-channel 比 per-tensor,误差逐级下降;
- **压缩比随位宽下降而上升**:FP32 → INT8 约 **4x**,INT4 约 **8x**(per-channel/分组因要存 scale 略低于理论值)。

脚本第 `[C]` 段还会逐个打印 `fp32 -> int8_q -> int8_hat / int4_q -> int4_hat`,直观看到一次"量化-反量化"的整数轨迹与 scale 的作用。

---

## 四、对应 llm-action 文档

- 量化文档总目录:[`../../llm-compression/quantization`](../../llm-compression/quantization)
- 量化基础:[`../../llm-compression/quantization/量化基础.md`](../../llm-compression/quantization/量化基础.md)
- 大模型量化概述:[`../../llm-compression/quantization/大模型量化概述.md`](../../llm-compression/quantization/大模型量化概述.md)
- LLM.int8():[`../../llm-compression/quantization/LLM-int8.md`](../../llm-compression/quantization/LLM-int8.md)
- GPTQ:[`../../llm-compression/quantization/GPTQ.md`](../../llm-compression/quantization/GPTQ.md)
- SmoothQuant:[`../../llm-compression/quantization/SmoothQuant.md`](../../llm-compression/quantization/SmoothQuant.md)

---

## 五、社区参考

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale — arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers — arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs — arXiv:2211.10438](https://arxiv.org/abs/2211.10438)

---

## 六、局限 / 与真实工程的差异

这是**教学最小实现**,刻意省去了工程细节,和生产级量化框架(bitsandbytes / GPTQ / AWQ / TensorRT-LLM 等)有明显差距:

1. **只做对称量化**:没有 zero-point(非对称/仿射量化)。激活值常偏态,真实方案多用非对称量化。
2. **只量化权重(W),不碰激活(A)**:真正的加速来自 W8A8 / W4A8 整数矩阵乘。LLM.int8()、SmoothQuant 的难点恰恰在**激活的离群值(outlier)**,本项目未涉及。
3. **没有真正的位打包(bit-packing)与整数算子**:这里 INT4 仍以浮点张量"模拟"整数,反量化后用 FP 做矩阵乘,因此**没有真实的内存节省和速度提升**,压缩比是"估算值"。真实框架会把 4bit 打包进 int32 并用定制 CUDA kernel。
4. **PTQ 且无误差补偿**:只是朴素的 round-to-nearest 训练后量化(PTQ),没有 GPTQ 的 Hessian/逐列误差补偿,也没有 QAT(量化感知训练)。
5. **toy 规模**:128×64 的小权重 + 随机初始化,数值分布比真实 LLM 权重温和得多;真实权重的离群值会让 per-tensor 量化误差远大于此处。

把这些差异作为延伸阅读的钩子:理解了本项目的 scale / per-channel / group 三件套后,再去读 GPTQ 的误差补偿、SmoothQuant 的激活平滑、LLM.int8() 的离群值分解,会顺畅很多。
