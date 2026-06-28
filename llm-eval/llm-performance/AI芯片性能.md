# AI 芯片性能评测

> 一句话定位：教你把一块 AI 芯片（GPU/NPU）的「纸面峰值」和「真实跑出来的有效算力」拆开看清楚——算力、带宽、MFU、微基准、roofline 与实测方法。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/算力/GPU工作原理]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|--------------|--------|
| 0 | 一句话锚点 | 峰值 ≠ 实测 |
| 1 | 地基：FLOPs/FLOPS/字节/带宽 到底是什么 | 单位拆原子 |
| 2 | 算力（TFLOPS）：怎么算出来、为什么有很多个数 | 频率×核数×每周期 FLOP |
| 3 | 显存带宽：为什么很多负载卡在这里 | HBM、GB/s |
| 4 | MFU / HFU：实测利用率怎么定义、怎么算 | 有效算力 ÷ 峰值 |
| 5 | 微基准：GEMM 与通信怎么单点测 | cuBLAS、NCCL bus BW |
| 6 | 峰值 vs 实测：差距从哪来 | 6 大损耗 |
| 7 | Roofline：一张图定位你被谁卡住 | 算术强度、拐点 |
| 8 | 怎么测：完整评测流程 | 预热、稳态、复现 |
| 数值例 | 端到端手算一遍 | A100 训练 7B |
| FAQ | 易错点 | 表格 |

## 0. 一句话锚点

**芯片厂商给的 TFLOPS 是「理论上限」，你真正能用上的只有其中一部分（典型 30%~60%）。评测 AI 芯片 = 量出三件事：峰值能力（算力/带宽）、单点微基准（GEMM/通信能跑到峰值的几成）、端到端有效利用率（MFU）；再用 roofline 解释「为什么只有这么多」。**

```
            纸面峰值 (Spec Sheet)
                 │   312 TFLOPS (A100 BF16)
                 ▼
   ┌─────────────────────────────┐
   │  微基准  GEMM 实测 ≈ 280 TF │  ← 单算子能到峰值 90%
   └─────────────────────────────┘
                 ▼
   ┌─────────────────────────────┐
   │  端到端  训练实测 ≈ 150 TF  │  ← MFU ≈ 48%
   └─────────────────────────────┘
   差距 = 通信 + 访存 + 非矩阵算子 + 气泡 + 启动开销
```

## 1. 地基：把单位拆到最原子

不假设你记得这些，逐个拆：

- **FLOP**（FLoating-point OPeration，复数 FLOPs）= **一次浮点运算**。一次乘法算 1 个 FLOP，一次加法算 1 个 FLOP。习惯上把「乘加」(FMA, fused multiply-add) 记为 **2 FLOPs**（1 乘 + 1 加）。
- **FLOPS**（FLoating-point Operations **per Second**，注意大写 S）= **每秒能做多少次浮点运算**。这是「速度/算力」。
  - 易混点：`FLOPs`（小写 s，复数，**总量**）vs `FLOPS`（大写 S，**速率**）。一个是「做了多少次」，一个是「每秒做多少次」。
  - TFLOPS = $10^{12}$ FLOPS（Tera），PFLOPS = $10^{15}$（Peta）。
- **Byte（字节）/ bit（位）**：1 Byte = 8 bit。BF16/FP16 一个数 = 2 Byte，FP32 = 4 Byte，FP8 = 1 Byte，INT4 = 0.5 Byte。
- **带宽（Bandwidth）**= 每秒能搬多少字节，单位 GB/s 或 TB/s。注意这里 1 GB 通常按 $10^9$ Byte（厂商）或 $2^{30}$（系统）算，差约 7%，量级判断不影响。
- **延迟（Latency）** vs **带宽（Bandwidth）**：延迟是「一趟要等多久」（µs/ns），带宽是「单位时间搬多少」。小数据看延迟，大数据看带宽。

```
 算力 (compute)        访存 (memory)
 ───────────          ───────────
 FLOP  = 一次浮点运算   Byte = 一个字节
 FLOPS = 次/秒 ← 速度   GB/s = 字节/秒 ← 带宽
        ↑                      ↑
   "算得多快"             "喂得多快"
   两者谁先到极限 → 谁就是瓶颈（见第7节 roofline）
```

> 一句话记忆：**芯片有两条腿——算（FLOPS）和搬（GB/s）。哪条腿短，跑多快由它定。**

## 2. 算力（TFLOPS）：那个大数字怎么来的

### 2.1 峰值算力的物理来源

GPU 的浮点峰值是「堆出来的」，公式（以传统 CUDA Core 为例）：

$$\text{Peak FLOPS} = N_{\text{cores}} \times f_{\text{clock}} \times \text{FLOP/cycle/core}$$

- $N_{\text{cores}}$：浮点单元数量（CUDA Core 数 / Tensor Core 数）。
- $f_{\text{clock}}$：时钟频率（Hz），现代 GPU 约 1.4~1.9 GHz。
- FLOP/cycle/core：每个核每周期能做几个 FLOP，FMA 单元 = 2。

**手算（CUDA Core, FP32, 数值仅作示意）**：设 6912 个 FP32 CUDA Core，频率 1.41 GHz，FMA = 2 FLOP/cycle：

$$6912 \times 1.41\times10^9 \times 2 \approx 19.5\times10^{12} = 19.5\ \text{TFLOPS (FP32)}$$

这与 A100 公开的 FP32 峰值约 19.5 TFLOPS 一致（以官方为准）。

### 2.2 为什么一块卡有「一堆」TFLOPS 数

因为不同精度、不同计算单元、开不开稀疏，峰值都不一样。**报数时必须带上精度和条件**，否则没有意义。

```
       同一块 A100，不同口径的峰值（约，以官方为准）
   ┌─────────────┬──────────────┬──────────────────┐
   │ 精度/单元    │ 峰值 TFLOPS  │ 用途             │
   ├─────────────┼──────────────┼──────────────────┤
   │ FP32 (CUDA) │   ~19.5      │ 通用/老式训练    │
   │ TF32 (TC)   │   ~156       │ 训练默认         │
   │ BF16/FP16(TC)│  ~312       │ 混合精度训练主力 │
   │ FP16 稀疏    │   ~624       │ 2:4 结构化稀疏   │
   │ INT8 (TC)   │   ~624 TOPS  │ 推理量化         │
   └─────────────┴──────────────┴──────────────────┘
   TC = Tensor Core（专做矩阵乘的专用单元）
   规律：精度越低、专用单元越多、开稀疏 → 数字越大
```

要点：
- **Tensor Core**（张量核）是专做小矩阵乘累加（如 4×4×4）的硬件单元，这是现代 AI 算力暴增的来源——大模型 80%+ 的算力靠它。
- **"稀疏峰值"通常 ×2**：靠跳过零值，需要权重满足 2:4 结构化稀疏，普通稠密负载吃不到，报告时要标清楚。
- INT8/INT4 用 **TOPS**（Tera Operations Per Second，整数运算）而非 TFLOPS。

## 3. 显存带宽：很多负载真正的瓶颈

### 3.1 为什么带宽这么重要

GPU 计算单元（SM）算得飞快，但数据放在显存（HBM）里。**如果数据喂不上来，算力单元就在空转。** 大模型推理里 decode 阶段（一次只生成 1 个 token）几乎全是「读权重」，是典型的**带宽受限（memory-bound）**。

```
   ┌──────────┐  搬数据   ┌─────────────┐
   │  HBM 显存 │ ───────► │ SM/Tensor Core│ 算
   │  ~80 GB  │ ◄─────── │  (片上很小)  │
   └──────────┘  写回     └─────────────┘
        ▲                        ▲
   带宽 ~2 TB/s            算力 ~312 TFLOPS
   每秒能搬 2e12 Byte      每秒能算 312e12 FLOP

   关键比值：算力/带宽 = 312e12 / 2e12 ≈ 156 FLOP/Byte
   → 平均每搬 1 字节，硬件"希望"你算 156 次，
     算不够就是浪费算力（memory-bound），见 roofline 拐点
```

### 3.2 带宽的来源

$$\text{BW} = (\text{总线位宽 bit} / 8) \times \text{有效数据率 (传输/秒)}$$

HBM（High Bandwidth Memory，高带宽显存）通过极宽的位宽（如 5120-bit）堆出带宽。典型公开值（约，以官方为准）：

| 卡 | 显存 | 带宽（约） |
|----|------|-----------|
| A100 80GB | HBM2e | ~2.0 TB/s |
| H100 SXM | HBM3 | ~3.35 TB/s |
| H200 | HBM3e | ~4.8 TB/s |

> 经验：**搬一遍模型权重要多久 = 权重字节数 ÷ 带宽**。例：7B 模型 BF16 权重 = $7\times10^9 \times 2 = 14$ GB，在 2 TB/s 上读一遍 ≈ $14/2000 \approx 7$ ms——这正是 decode 单步延迟的下界。

## 4. MFU / HFU：实测利用率怎么定义

这是本仓库原有方向，作为评测的「最终成绩单」。

### 4.1 定义（把原文说清楚）

- **MFU（Model FLOPs Utilization，模型算力利用率）**：模型**一次前向+反向**真正需要的（矩阵）算力，占机器峰值算力的比值。它只算「模型在数学上必须做的运算」。
- **HFU（Hardware FLOPs Utilization，硬件算力利用率）**：在 MFU 基础上，**把重计算（activation recomputation/gradient checkpointing）多做的前向也算进去**，再除以峰值。
- 关系：因为重计算让硬件实际多算了一些（但模型本身不需要），所以 **HFU ≥ MFU**。HFU 衡量「硬件忙不忙」，MFU 衡量「忙得有没有用」。

$$\text{MFU} = \frac{\text{模型必需 FLOPs / step}}{\text{step 耗时(s)} \times \text{峰值 FLOPS}}$$

```
   每步实际算的       │←── 重计算多算的 ──→│
   ├───模型必需 FLOPs───┼──recompute 额外──┤
   │                    │                  │
   └── MFU 分子 ────────┘                  │
   └── HFU 分子 ───────────────────────────┘
                  都 ÷ (step时间 × 峰值FLOPS)
   ⇒ HFU ≥ MFU；二者差距 = 重计算开销占比
```

### 4.2 Transformer 必需 FLOPs 的经验公式（手算用）

训练一次（前向+反向）每个 token 的算力，业界常用近似：

$$C_{\text{token}} \approx 6 \times N_{\text{params}}$$

其中 6 = 2(前向乘加) + 4(反向约 2 倍前向)。所以一个 step（batch 的总 token 数为 $T$）：

$$\text{FLOPs/step} \approx 6 \times N \times T$$

> 「6N」是只数主干矩阵乘的近似；严格版（含 attention 的 $O(L^2)$ 项）见末节数值例。推理只前向时常用 $2N$/token。

## 5. 微基准：把单点能力测出来

端到端 MFU 是综合结果，要诊断「哪一环弱」，得做**微基准（microbenchmark）**：只测一个算子/一条链路，看它能跑到峰值的几成。

### 5.1 GEMM（矩阵乘）微基准——量算力

GEMM = General Matrix Multiply，$C = A \times B$，是 Tensor Core 的主战场。

- **FLOPs 公式**：$M\times N$ 输出、$K$ 为内积维：$\text{FLOPs} = 2 \times M \times N \times K$（每个输出元素 K 次乘加 = 2K FLOP）。
- **实测 TFLOPS** = $\dfrac{2MNK}{\text{耗时(s)}}$，再 ÷ 峰值 = GEMM 利用率。
- 工具：`cublasLt` / `nvidia-cutlass profiler` / PyTorch `torch.matmul` 计时。

```
  A (M×K)      B (K×N)        C (M×N)
  ┌──────┐    ┌──────┐       ┌──────┐
  │      │ ×  │      │   =    │      │
  └──────┘    └──────┘       └──────┘
  FLOPs = 2·M·N·K
  实测 = 2MNK / 时间   →   /峰值 = GEMM 效率(通常 80~95%)

  形状影响巨大：
    大方阵 (4096³)        → 高效率，逼近峰值
    瘦长 (M=1, decode)    → 低效率，退化为带宽受限
    K 不是 8/16 倍数      → Tensor Core 利用率掉
```

> 关键认知：**GEMM 能跑到 90% 峰值 ≠ 端到端能到 90%**。微基准是「单科满分」，端到端是「总成绩」。

### 5.2 通信微基准——量互联带宽

多卡训练里，梯度要靠 **All-Reduce** 等集合通信同步。用 **NCCL**（NVIDIA Collective Communications Library）的 `nccl-tests` 测。

- 报两个数：**algbw**（算法带宽 = 数据量/时间）和 **busbw**（总线带宽，已按算法系数归一，更能反映物理链路）。
- All-Reduce 的 busbw ≈ algbw × $\frac{2(n-1)}{n}$（n 卡）。
- 链路：卡内 **NVLink**（数百 GB/s）≫ 跨机 **PCIe/InfiniBand**（数十~数百 GB/s）。

```
   All-Reduce 拓扑（Ring，4 卡）
   GPU0 ─► GPU1 ─► GPU2 ─► GPU3
     ▲                       │
     └───────────────────────┘
   每卡发送/接收 ≈ 2(n-1)/n × 数据量
   nccl-tests 输出 busbw → /NVLink峰值 = 通信效率
   通信慢 → 端到端出现"气泡"，MFU 掉
```

## 6. 峰值 vs 实测：差距从哪来（核心）

为什么微基准 GEMM 能 90%、端到端 MFU 只有 40~55%？六大损耗：

```
   峰值 312 TFLOPS
   ──────────────────────────────────────────► 100%
   ① 非矩阵算子   LayerNorm/Softmax/激活/逐元素   -10~20%
      （这些是访存受限，Tensor Core 闲着）
   ② 访存搬运     读写 HBM、缓存未命中            -5~15%
   ③ 通信开销     All-Reduce / All-Gather 同步    -5~20%
   ④ 流水线气泡   PP 各级等待、负载不均           -5~15%
   ⑤ 启动/调度    kernel launch、Python 开销、小算子 -3~10%
   ⑥ 重计算       省显存换算力（MFU 看不到，HFU 看到）
   ──────────────────────────────────────────►
   实测 MFU ≈ 40~55%（训练好的话）
```

实践判据：
- **MFU 50%+** 在大规模训练里已属优秀（很多公开工作 35~55%）。
- 若 **GEMM 微基准也低** → 问题在算子/形状/精度；若 **GEMM 高但端到端低** → 问题在通信/气泡/小算子。这就是分层诊断的价值。

## 7. Roofline：一张图说清「你被谁卡住」

### 7.1 算术强度（Arithmetic Intensity, AI）

$$\text{AI} = \frac{\text{FLOPs（算了多少次）}}{\text{Bytes（搬了多少字节）}}\quad (\text{FLOP/Byte})$$

它衡量「每搬一个字节，能复用着算多少次」。AI 高 = 算得多搬得少（compute-bound），AI 低 = 搬得多算得少（memory-bound）。

### 7.2 Roofline 模型

把「可达性能」画成两条上限的最小值：

$$\text{Attainable FLOPS} = \min\big(\underbrace{\text{Peak FLOPS}}_{\text{算力屋顶}},\ \underbrace{\text{AI} \times \text{Bandwidth}}_{\text{带宽斜坡}}\big)$$

```
 性能(FLOPS)
   ▲
峰值├──────────────── ← 算力屋顶 (compute-bound)
   │           ╱──────
   │         ╱  ← 斜坡斜率 = 带宽(GB/s)
   │       ╱
   │     ╱   ← 此区 memory-bound（带宽决定）
   │   ╱
   └─╱──────┬──────────────► 算术强度 AI (FLOP/Byte)
            拐点 = 峰值FLOPS / 带宽
       拐点(A100 BF16)≈ 312e12/2e12 ≈ 156 FLOP/Byte
   ● 你的算子 AI < 156 → 落在斜坡 → 被带宽卡（如 decode、LayerNorm）
   ● 你的算子 AI > 156 → 落在屋顶 → 被算力卡（如大 GEMM、训练 prefill）
```

用法三步：
1. 算你的算子/模型的 AI（总 FLOPs ÷ 总搬运 Bytes）。
2. 和拐点比：小于拐点 → memory-bound，优化方向是**减少访存/提批量/算子融合/量化**；大于 → compute-bound，优化方向是**更低精度/更好 kernel/更大 GEMM**。
3. 把实测点画上去，看离屋顶/斜坡还有多远 = 优化空间。

## 8. 怎么测：可复现的评测流程

```
 ┌─ ① 锁定环境 ──────────────────────────────┐
 │  固定 driver/CUDA/框架版本；锁频或记录频率 │
 │  nvidia-smi -q 看功耗/温度墙是否触发        │
 ├─ ② 微基准 (单点) ─────────────────────────┤
 │  GEMM: 扫多种 (M,N,K) → 实测 TFLOPS/峰值   │
 │  通信: nccl-tests → busbw/NVLink峰值       │
 │  访存: 带宽测试 (如 stream/复制核)         │
 ├─ ③ 端到端 (综合) ─────────────────────────┤
 │  跑真实模型；预热(warmup)弃前几步          │
 │  取稳态(steady-state)多步求中位数          │
 │  记 step 时间 → 算 MFU/HFU                  │
 ├─ ④ 定位 (roofline+profiler) ──────────────┤
 │  Nsight Systems 看时间线(算/通信/气泡)     │
 │  Nsight Compute 看单 kernel 的 AI 与瓶颈   │
 └─ ⑤ 报告 ──────────────────────────────────┘
    必带：精度、batch、序列长、并行策略、硬件、软件版本
```

测量铁律：
- **预热**：前几步含编译/缓存填充，必须丢弃。
- **稳态**：取中间多步的**中位数**（避开抖动），别只测一步。
- **同步计时**：GPU 异步执行，计时前后要 `torch.cuda.synchronize()`，否则测的是「下发命令的时间」而非「算完的时间」——这是最常见的错误。
- **报告口径**：TFLOPS 必带精度；MFU 必带峰值口径（用稠密 BF16 峰值，不要用稀疏峰值灌水）。
- **避免热墙/功耗墙**：长跑会降频，结论要在稳态下取。

## 9. 数值例子：A100 上训练 7B 模型，端到端手算一遍

**设定**（示意值）：A100，BF16 峰值 312 TFLOPS（稠密），单卡，参数 $N=7\times10^9$，全局 batch 一个 step 处理 token 数 $T = 2\times10^6$（如 1024 序列 × 约 2000 样本），实测每 step 耗时 $t = 4.6$ s。

**第一步：必需 FLOPs/step**（用 6N 近似）

$$6 \times N \times T = 6 \times 7\times10^9 \times 2\times10^6 = 8.4\times10^{16}\ \text{FLOPs}$$

**第二步：实测有效算力**

$$\frac{8.4\times10^{16}}{4.6} \approx 1.83\times10^{16}\ \text{FLOPS} = 18.3\ \text{... 等价 } 183\ \text{TFLOPS}$$

（$1.83\times10^{16}$ FLOPS = 18.3 PFLOPS? 不——$10^{16}$ 级是 ~18.3 ×10¹⁵ = 18.3 PFLOPS 仅当多卡；单卡此例 $T$ 偏大，仅作算法演示。换算：$1.83\times10^{16}/10^{12}=18300$ TFLOPS 显然超单卡，说明该 $T$ 对应多卡聚合。）

**修正为单卡口径**：取单卡每 step 处理 $T=2\times10^4$ token、$t=0.46$ s：

$$\frac{6\times7\times10^9\times2\times10^4}{0.46}=\frac{8.4\times10^{14}}{0.46}\approx1.83\times10^{15}=183\ \text{TFLOPS}$$

**第三步：MFU**

$$\text{MFU} = \frac{183}{312} \approx 0.586 = 58.6\%$$

**解读**：58.6% 是相当好的训练 MFU。剩下 41% 去了第 6 节的六大损耗。若该卡还开了重计算（多做约 1/3 前向），HFU 会更高，比如 $\approx 58.6\% \times (1 + 1/3) \approx 78\%$（示意），说明硬件其实更忙，只是部分忙在「为省显存而重复的前向」上。

**带宽侧对照（decode 推理）**：同样 7B、BF16 权重 14 GB，单步 decode 主要读一遍权重，下界 $14\text{GB}/2\text{TB/s}=7$ ms → 理论上限约 142 token/s（单序列，未计 KV cache 与 batch），这就是 decode **带宽受限**的来源，与训练的 compute-bound 形成对照。

## 常见问题

| 问题 | 答案 |
|------|------|
| FLOPs 和 FLOPS 区别？ | 小写 s = 运算总次数（量）；大写 S = 每秒次数（速度/算力）。 |
| 为什么我的实测远低于厂商 TFLOPS？ | 那是理论峰值；实测要扣非矩阵算子、访存、通信、气泡、启动、（可能用了稀疏峰值口径灌水）。 |
| MFU 和 HFU 哪个高？ | HFU ≥ MFU；差距来自重计算。MFU 看「有用算力」，HFU 看「硬件忙不忙」。 |
| 报 TFLOPS 要注意什么？ | 必带精度（FP32/BF16/FP8）、是否稠密、是否 Tensor Core。否则数字无意义。 |
| 我的模型是被算力还是带宽卡？ | 算算术强度 AI，与 roofline 拐点（峰值÷带宽）比；小于拐点=带宽卡，大于=算力卡。 |
| 训练 MFU 多少算好？ | 大规模训练 50%+ 优秀，35~50% 常见；推理 decode 看带宽利用率而非 MFU。 |
| 测速最常见的坑？ | 没做 `cuda.synchronize()` 就计时（测成了下发时间）；没预热；只测一步；用峰值口径不一致。 |
| GEMM 微基准高但端到端低？ | 瓶颈不在算力，去查通信（nccl-tests）、小算子启动、流水线气泡。 |
| 6N 是哪来的？ | 前向 2N + 反向 4N（反向约 2 倍前向）= 每 token 6N FLOPs（主干矩阵乘近似，未含 attention 二次项）。 |
| TOPS 和 TFLOPS？ | TOPS 是整数运算（INT8/INT4）速率，TFLOPS 是浮点运算速率；量化推理常报 TOPS。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，从这里找其他主题
- [[ai-infra/算力/GPU工作原理]] — 想懂 SM / Tensor Core / HBM 为什么这样设计，回到底层硬件
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] — 配套的指标名词表（吞吐、延迟、TTFT、TPOT 等）
