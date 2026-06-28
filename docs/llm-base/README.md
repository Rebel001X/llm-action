# LLM 基础与 AI-Infra 地基总览（llm-base 导览）

> 一句话定位：这是 `llm-base/` 目录的总入口——把训练/推理一台大模型所需的**最底层地基**（数值精度、分布式并行、GPU 集群与网络、显存与算力估算、监控工具链）拼成一张可手算、可导航的知识地图。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] · [[llm-inference/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[ai-infra/算力/GPU工作原理]] · [[ai-infra/网络/集合通信原语]]

## 阅读地图

| 你想知道的问题 | 跳到本文 | 富笔记深读 |
| --- | --- | --- |
| 模型用什么数据类型存？FP16/BF16/FP8 区别？ | §2 数值精度 | [[机器学习中常用的数据类型]] · [[llm-compression/quantization/量化基础]] |
| 一个 7B/70B 模型训练/推理要多少显存？ | §3 显存估算（手算） | [[llm-compression/quantization/fp8]] |
| 训练一步要多少 FLOPs？要训多久？ | §4 算力估算（手算） | [[llm-algo/FLOPs]] · [[FLOPS]] |
| 模型放不下一张卡怎么切？DP/TP/PP/MoE/ZeRO | §5 分布式并行总图 | [[distribution-parallelism/README]] · [[llm-inference/大模型推理张量并行]] |
| 多卡多机怎么连？NVLink/PCIe/IB？通信多大？ | §6 集群与网络 | [[ai-infra/网络/集合通信原语]] |
| 怎么把显存/算力换着花？重计算/卸载/累积 | §7 加速与省显存技术 | [[分布式训练加速技术]] · [[llm-optimizer/计算通信重叠]] |
| 怎么看 GPU 跑得健不健康？ | §8 监控工具链 | [[nvidia-smi]] · [[dcgmi]] |

## 0. 一句话锚点

> 训练/推理一台大模型 = **「把一堆浮点数（精度）搬进显存（容量）、用张量核做矩阵乘（算力）、再用网络在多卡间同步（通信）」**。本目录的每一节，都是这句话里某个名词的展开。四个地基缺一不可：

```
        ┌──────────── 一台大模型怎么跑起来 ────────────┐
        │                                             │
   [精度/数据类型]   [显存容量]    [算力FLOPs]   [网络通信]
     FP32/BF16/FP8   80GB HBM     TensorCore     NVLink/IB
        §2            §3            §4             §6
        │             │             │              │
        └────── 不够? → §5 并行切分 + §7 省显存换算力换通信 ──┘
```

## 1. 地基/前置：为什么"大"模型需要一整套基础设施

一个稠密 Transformer 的一切都是矩阵乘法（GEMM）。规模一旦上去，三件事同时爆炸：

- **参数量 $P$ 爆炸**：7B→70B→千亿，权重本身就装不下一张卡。
- **激活值爆炸**：训练要存中间激活做反向传播，激活显存随 `batch × seq × layers` 线性涨。
- **通信爆炸**：切到多卡后，每一步都要 All-Reduce 梯度 / All-Gather 权重。

所以 `llm-base` 不是讲"模型结构"（那在 [[llm-algo/transformer/模型架构]]），而是讲**承载模型的物理与系统地基**。先建立单位直觉：

```
1 个 FP32 数 = 4 字节   1 个 BF16/FP16 数 = 2 字节   1 个 FP8/INT8 数 = 1 字节
1 GiB = 2^30 B ≈ 1.07e9 B        1 GB(十进制) = 1e9 B
1 TFLOP/s = 1e12 次浮点运算/秒
A100(BF16) ≈ 312 TFLOP/s   H100(BF16) ≈ ~990 TFLOP/s(稀疏更高，以官方为准)
```

## 2. 数值精度：所有显存与速度的"计价单位"

> 深读 → [[机器学习中常用的数据类型]]。这里给"骨架 + 怎么选"。

浮点数 = `符号位 | 指数位(动态范围) | 尾数位(精度)`。**指数位决定能表示多大/多小，尾数位决定有多准。**

```
          符号  指数(range)   尾数(precision)   字节
 FP32   :  1  |  8 bits   |   23 bits      |  4   全精度，主权重
 TF32   :  1  |  8 bits   |   10 bits      | (内部计算型, Ampere+)
 FP16   :  1  |  5 bits   |   10 bits      |  2   范围小→易溢出，需loss scaling
 BF16   :  1  |  8 bits   |    7 bits      |  2   范围=FP32, 精度差, 训练首选
 FP8(E4M3): 1 |  4 bits   |    3 bits      |  1   H系列+, GEMM加速 → [[fp8]]
 INT8   :  整数 [-128,127]                 |  1   量化推理 → [[量化基础]]
```

记忆法：**FP16 范围小、BF16 精度差**。同样 2 字节，BF16 因为指数位和 FP32 一样宽，几乎不溢出，是大模型训练的事实标准；FP16 必须配 loss scaling。FP8 兼有"FP16 的稳定 + INT8 的速度"，靠 Transformer Engine 做 GEMM、用 BF16/FP32 保主权重。

**混合精度训练**的本质（一定要记牢，这是显存估算的前提）：

```
  master weight (FP32, 精确) ──复制──► weight (BF16) ──前向/反向──► grad (BF16)
        ▲                                                              │
        └──────────── 用 FP32 优化器状态更新主权重 ◄────────────────────┘
  推理时只需半精度权重 → 同样结果只用一半显存
```

## 3. 显存估算（必含手算）：一台模型到底吃多少 GB

显存 = **权重 + 梯度 + 优化器状态 + 激活**。设参数量 $P$。

**训练显存（Adam + 混合精度，每参数字节数）**：

| 部分 | 精度 | 字节/参数 |
| --- | --- | --- |
| 模型权重 | BF16 | 2 |
| 梯度 | BF16 | 2 |
| 优化器：FP32 主权重 | FP32 | 4 |
| 优化器：Adam 一阶动量 $m$ | FP32 | 4 |
| 优化器：Adam 二阶动量 $v$ | FP32 | 4 |
| **合计（不含激活）** |  | **16 字节/参数** |

这就是著名的 **"16×P 规则"**：$\text{Mem}_{\text{static}} \approx 16P$ 字节。

**手算 1 —— 7B 模型训练静态显存**：

```
P = 7e9
静态显存 = 16 B × 7e9 = 1.12e11 B = 112 GB / 1.024^3 ≈ 104 GiB
→ 一张 80GB 的 A100/H100 装不下 ! → 必须并行(§5) 或 ZeRO 切分优化器状态
```

**手算 2 —— 70B 模型推理（BF16）显存**：

```
权重 = 2 B × 70e9 = 1.4e11 B = 140 GB ≈ 130 GiB
→ 至少需要 2× 80GB 卡做张量并行 → [[llm-inference/大模型推理张量并行]]
  再叠加 KV Cache → [[llm-inference/KV-Cache优化]]
```

**手算 3 —— KV Cache 显存**（推理时除权重外的大头）：每 token 每层缓存 K、V 两份。

$$\text{KV} = 2 \times L \times n_{kv} \times d_{head} \times \text{batch} \times \text{seq} \times \text{bytes}$$

```
例: L=80 层, 隐藏维=8192(头数64×128), batch=1, seq=4096, BF16=2B, MHA(n_kv=头数)
KV = 2 × 80 × 8192 × 1 × 4096 × 2 B
   = 2 × 80 × 8192 × 4096 × 2 = 1.07e10 B ≈ 10 GiB  (单条序列!)
→ 这就是为什么要 GQA/MQA 减少 n_kv、要 PagedAttention 省碎片 → [[llm-optimizer/kv-cache]]
```

**激活显存（训练）** 直觉：$\propto \text{batch}\times\text{seq}\times\text{hidden}\times\text{layers}$，是开启**重计算/梯度检查点**（§7）的主要动机——用算力换掉它。

## 4. 算力估算（必含手算）：要多少 FLOPs、训多久

> 深读 → [[llm-algo/FLOPs]] · [[FLOPS]]。核心经验公式（稠密 Transformer）：

$$\text{训练总 FLOPs} \approx 6 \times P \times D$$

其中 $P$=参数量，$D$=训练 token 数。"6" = 前向 2 + 反向 4（每个 MAC 算 2 次浮点：1 乘 1 加；反向是前向的两倍）。推理（只前向）则是 $\approx 2 P$ 每 token。

**手算 4 —— 训练 7B 模型、1T tokens 要多久（1024×A100）**：

```
总FLOPs = 6 × 7e9 × 1e12 = 4.2e22 FLOPs
单卡 A100 BF16 峰值 = 312e12 FLOP/s
实际利用率(MFU) ≈ 40% → 有效 = 312e12 × 0.40 = 1.25e14 FLOP/s
集群有效算力 = 1.25e14 × 1024 ≈ 1.28e17 FLOP/s
时间 = 4.2e22 / 1.28e17 ≈ 3.28e5 s ≈ 91 小时 ≈ 3.8 天
```

```
  FLOPs ─┐
         ├─► 时间 = 6PD / (单卡峰值 × MFU × 卡数)
  卡数 ──┘
  关键变量是 MFU(模型算力利用率): 并行切得越碎、通信占比越高 → MFU 越低
```

**手算 5 —— 单 token 推理 FLOPs**：`2 × 7e9 = 1.4e10 FLOPs/token`。配合 GPU 算力和显存带宽（推理常是**带宽瓶颈**而非算力瓶颈，因为每 token 要把整套权重从 HBM 读一遍）即可估吞吐。

## 5. 分布式并行总图：模型放不下一张卡时怎么切

> 深读总入口 → [[distribution-parallelism/README]]。一张图记住五种切法的"切什么维度"：

```
                     ┌─────────── 并行家族 ───────────┐
   数据并行 DP        张量并行 TP        流水线并行 PP       MoE 并行        ZeRO/FSDP
  (切 batch)        (切单层权重)       (切层/stage)     (切专家)       (切优化器状态)
       │                 │                 │               │                │
 GPU0 GPU1         GPU0 ▏GPU1        GPU0:L0-3        GPU0:E0,E1     权重/梯度/状态
 全套权重 各算       一层横劈两半       GPU1:L4-7       GPU1:E2,E3     分片到各卡再聚合
 半个batch          算完 All-Reduce    像流水线接力     按router分发    → [[deepspeed/README]]
 → All-Reduce梯度   通信量最大         有气泡(bubble)   → [[moe-parallel]]
 → [[data-parallelism/README]] [[tensor-parallel/README]] [[pipeline-parallelism/README]]
```

| 并行方式 | 切的维度 | 通信原语 | 通信发生时机 | 适用 |
| --- | --- | --- | --- | --- |
| 数据并行 DP | batch | All-Reduce(梯度) | 每步反向后 | 模型能放下单卡 |
| 张量并行 TP | 单层权重(行/列) | All-Reduce / All-Gather | 每层前向+反向 | 单层都放不下，机内 NVLink |
| 流水线并行 PP | 层(stage) | P2P(send/recv) | stage 边界 | 层数多、跨机 |
| MoE 并行 | 专家 | All-to-All | router 分发/聚合 | 稀疏大模型 → [[llm-algo/moe/README]] |
| ZeRO/FSDP | 优化器状态/梯度/权重 | Reduce-Scatter+All-Gather | 反向/前向 | 省静态显存，等价 DP |

**通信量手算（TP All-Reduce）**：Megatron 风格 TP，每个 Transformer 层前向 2 次、反向 2 次 All-Reduce。Ring All-Reduce 单次通信量 $\approx 2\times \frac{N-1}{N}\times S$（$S$=张量字节，$N$=卡数）。

```
例: TP=8, 一层激活张量 S = batch×seq×hidden×2B = 1×4096×8192×2 = 6.4e7 B ≈ 64 MB
单次 All-Reduce 流量 ≈ 2 × (7/8) × 64MB ≈ 112 MB
一层 4 次 → 448 MB/层; 80 层 → ~35 GB/步 在 NVLink(~600GB/s)上 ≈ 60 ms
→ 这就是为什么 TP 必须放在机内 NVLink 域, 跨机做 TP 会被 IB 带宽拖死
```

**实践组合（3D/4D 并行）**：`TP(机内) × PP(跨机) × DP(跨副本) [× MoE]`，详见 [[multidimensional-hybrid-parallel/README]] 与 [[ai-framework/megatron-lm/README]]。自动并行（让框架搜最优切法）见 [[distribution-parallelism/auto-parallel/README]]。

## 6. 集群与网络：多卡多机怎么连、通信多快

> 深读 → [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/GPU工作原理]]。

**带宽层级（差一个数量级就决定并行怎么摆）**：

```
   ┌─ 机内 GPU↔GPU ─────────────────────────────┐
   │  NVLink/NVSwitch  ~600-900 GB/s   ← 放 TP   │
   │  PCIe Gen4/5       ~32-64 GB/s    ← 退而求其次│
   └────────────────────────────────────────────┘
   ┌─ 机间 节点↔节点 ───────────────────────────┐
   │  InfiniBand NDR  ~400 Gb/s(=50GB/s)/卡 ← 放 PP/DP │
   │  RoCE/TCP-IP      更慢            ← 兜底     │
   └────────────────────────────────────────────┘
   口诀: 机内 NVLink > PCIe ; 机间 IB > RoCE > TCP
        把通信最密的(TP)塞进最快的域(NVLink)
```

**集合通信原语速记**（深读 [[ai-infra/网络/集合通信原语]]）：

```
 All-Reduce  : 各卡求和后人人拿到结果   (DP 梯度同步、TP)
 All-Gather  : 各卡碎片拼成完整         (ZeRO 前向取权重)
 Reduce-Scatter: 求和后各拿一片         (ZeRO 反向)
 All-to-All  : 人人给人人发不同片        (MoE 专家分发)
 Broadcast/P2P: 一对多 / 点对点          (初始化 / PP 接力)
```

集群上层调度（`slurm` / `singularity`）、RDMA 网络压测（`多机RDMA性能测试`）、环境安装（`a800-env-install` / `h800-env-install`）都在本目录，是把上面理论落到真机的"运维层"。

## 7. 加速与省显存技术：三种"换"的艺术

> 深读 → [[分布式训练加速技术]] · [[llm-optimizer/计算通信重叠]]。

```
        想省什么            手段              代价(换什么)
  ┌──────────────────────────────────────────────────────┐
  │ 省显存(激活)   重计算/梯度检查点      用算力换 (反向重算前向)│
  │ 省显存(权重)   ZeRO / Offload         用通信换 (CPU↔GPU横跳)│
  │ 省显存(精度)   FP16/BF16/FP8          用精度换 (省一半~3/4)  │
  │ 省通信         梯度累积               用更新频率换 (攒大batch)│
  │ 省显存+提速    FlashAttention         用kernel融合(减HBM访问)│
  │ 省KV+提速      MQA/GQA                用表达力换 (共享KV)     │
  │ 藏通信         计算通信重叠           用调度换 (comm/comp叠)  │
  └──────────────────────────────────────────────────────┘
```

- **重计算**：前向只存少量激活，反向需要时重算——用约 +33% 算力换大幅激活显存下降。
- **梯度累积**：攒 $k$ 个 micro-batch 的梯度再更新一次，等效大 batch 而不增显存；梯度取 $k$ 个 batch 平均。
- **FlashAttention**：分块（tiling）+ kernel 融合，把 Attention 的 $O(seq^2)$ 中间矩阵不落 HBM → 训练推理都快、都省显存。深读 [[llm-optimizer/FlashAttention]]。
- **计算通信重叠**：反向算梯度的同时就开始 All-Reduce 已算好的层，把通信"藏"进计算窗口 → [[llm-optimizer/计算通信重叠]]。

## 8. 监控与工具链：怎么知道集群跑得健不健康

> 这些是本目录的"体检工具"，原理是读 GPU 的硬件计数器（利用率、显存、温度、功耗、ECC、NVLink 流量）。

```
 nvidia-smi      : 一眼看显存/利用率/进程    → [[nvidia-smi]]
 nvidia-smi dmon : 滚动刷新逐卡指标          → [[nvidia-smi-dmon]]
 dcgmi           : 数据中心级 GPU 健康/诊断    → [[dcgmi]]
 Nsight Systems  : 时间线级 profile(找气泡/通信瓶颈) → [[NVIDIA-Nsight-Systems性能分析]]
 nvprof/nvvp     : 老一代 kernel 级 profiler
```

看监控的逻辑闭环：**利用率低 → 看是算力瓶颈还是带宽/通信瓶颈 → 回到 §4 的 MFU / §6 的通信 → 调并行策略或开 §7 的重叠**。

## 数值手算速查（汇总）

| 量 | 公式 | 例子 |
| --- | --- | --- |
| 推理权重显存 | $2P$ 字节(BF16) | 70B → 140 GB |
| 训练静态显存 | $16P$ 字节(Adam混精) | 7B → 112 GB |
| KV Cache | $2 L\,n_{kv} d_{head}\,b\,s\cdot\text{bytes}$ | 见 §3 手算3 → 10 GiB |
| 训练总算力 | $6PD$ FLOPs | 7B×1T → 4.2e22 |
| 推理单 token 算力 | $2P$ FLOPs | 7B → 1.4e10 |
| 训练耗时 | $\dfrac{6PD}{\text{卡数}\times\text{峰值}\times\text{MFU}}$ | 7B×1T,1024卡 → ~3.8 天 |
| Ring All-Reduce 单次流量 | $2\frac{N-1}{N}S$ | 见 §5 → 112 MB |

## 常见问题

| 问题 | 答案 |
| --- | --- |
| 为什么大模型训练首选 BF16 而非 FP16？ | BF16 指数位与 FP32 同宽，动态范围一致几乎不溢出；FP16 范围小必须配 loss scaling。代价是 BF16 尾数少 3 位、精度略差。 |
| "16×P"里那 16 字节怎么来的？ | 权重2+梯度2+FP32主权重4+Adam一阶4+二阶4=16（混合精度+Adam）。SGD 或不存主权重会更小。 |
| TP 和 DP 什么时候用谁？ | 模型放得下单卡 → DP（只在反向 All-Reduce）；单层都放不下 → TP（每层都通信，必须机内 NVLink）。通常 TP×PP×DP 组合。 |
| ZeRO 和 DP 什么关系？ | ZeRO 是"省显存版 DP"：把优化器状态/梯度/权重分片到各 DP 卡上，需要时再 All-Gather，等价语义但显存大降。 |
| 推理瓶颈是算力还是带宽？ | 多为**带宽**：每生成一个 token 要把整套权重从 HBM 读一遍，算力（$2P$）反而吃不满 → 故有 KV Cache、量化、投机解码等优化。 |
| 重计算划算吗？ | 用约 +1/3 算力换大幅激活显存下降，在显存吃紧时几乎总是划算，是长序列/大模型标配。 |
| MFU 一般多少？ | 经验 30%~50%（以实际为准）。并行切得越碎、通信占比越高、bubble 越大，MFU 越低。 |
| FLOPs 公式里为什么是 6？ | 前向每参数 2 FLOPs（1乘1加），反向约 2×前向 = 4，合计 6；推理只前向故 2。 |

## 🔗 跳转链接

- 知识地图总入口：[[00-知识地图]]
- 数据类型与量化：[[机器学习中常用的数据类型]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/fp8]]
- 算力与结构：[[llm-algo/FLOPs]] · [[FLOPS]] · [[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[llm-algo/旋转编码RoPE]]
- 分布式并行：[[distribution-parallelism/README]] · [[tensor-parallel/README]] · [[data-parallelism/README]] · [[pipeline-parallelism/README]] · [[moe-parallel/README]] · [[multidimensional-hybrid-parallel/README]] · [[distribution-parallelism/auto-parallel/README]] · [[llm-algo/moe/README]]
- 推理与 KV：[[llm-inference/README]] · [[llm-inference/大模型推理张量并行]] · [[llm-inference/KV-Cache优化]] · [[llm-inference/解码策略]] · [[llm-optimizer/kv-cache]]
- 优化算子：[[llm-optimizer/FlashAttention]] · [[llm-optimizer/计算通信重叠]] · [[分布式训练加速技术]]
- 框架：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]]
- 集群/硬件/网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/ai-hardware/CUDA]]
- 训练与对齐：[[llm-train/README]] · [[llm-train/peft/Prompt-Tuning]] · [[llm-train/peft/Prefix-Tuning]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 本目录监控工具：[[nvidia-smi]] · [[nvidia-smi-dmon]] · [[dcgmi]] · [[NVIDIA-Nsight-Systems性能分析]]
