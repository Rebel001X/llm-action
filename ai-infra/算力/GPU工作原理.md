# GPU 工作原理

> 一句话定位：GPU 是一台"用上千个简单算术单元并行碾压数据、靠分层显存喂数据"的吞吐机器；看懂它，就是看懂"算力"和"带宽"哪个先饿死。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/ai-hardware/CUDA]] [[llm-algo/FLOPs]] [[llm-optimizer/FlashAttention]]

## 阅读地图

| 节 | 你将搞懂 | 一句话钥匙 |
|----|----------|-----------|
| 0 | 一句话锚点 | GPU = 算力 × 带宽，谁小谁封顶 |
| 1 | CPU vs GPU 设计哲学 | CPU 优化延迟，GPU 优化吞吐 |
| 2 | 硬件层级 GPC→SM→CUDA核 | SM 是真正的"核心" |
| 3 | Tensor Core | 一拍算一个 4×4 矩阵乘加 |
| 4 | SIMT 与 warp | 32 线程锁步走一条指令 |
| 5 | 显存层级 | 寄存器→共享→L2→HBM，越快越小 |
| 6 | 算力 vs 带宽 | TFLOPS 和 TB/s 是两个数 |
| 7 | Roofline 与算术强度 | AI = FLOPs / Bytes 决定命运 |
| 8 | 为什么 LLM decode 受带宽限 | 每个 token 要把权重整搬一遍 |
| 9 | 手算：判断 compute / memory bound | 拿 AI 和拐点比大小 |
| 10 | 对照表 / FAQ | 速查 |

## 0. 一句话锚点

一块 GPU 干活的速度被两个上限夹住：

$$\text{实际吞吐} = \min\big(\underbrace{\text{峰值算力}}_{\text{FLOPS}},\ \underbrace{\text{峰值带宽} \times \text{算术强度}}_{\text{Bytes/s} \times \text{FLOPs/Byte}}\big)$$

- **算力上限**：芯片每秒最多能做多少次乘加（FLOPS）。
- **带宽上限**：每秒最多能从显存搬多少字节（Bytes/s）。

如果你的算子"每搬 1 字节只算几次"，那带宽先饿死它（**memory bound**）；如果"每搬 1 字节算几百次"，那算力先饿死它（**compute bound**）。**LLM 的训练大矩阵乘是 compute bound，单 batch 的 decode 是 memory bound**——这是本文的灵魂结论。

## 1. 地基：CPU 和 GPU 是两种世界观

先把"核"这个词的歧义破掉。CPU 的核和 GPU 的核完全不是一个量级的东西。

- **CPU**：少量（4~64）非常复杂的核。每个核有大缓存、分支预测、乱序执行——目标是**让一条指令链尽快跑完**（低延迟）。
- **GPU**：上千个非常简单的算术单元（CUDA 核），靠"线程多到隐藏延迟"取胜——目标是**单位时间处理尽量多数据**（高吞吐）。

```
        CPU（延迟优化）                      GPU（吞吐优化）
   ┌──────────────────────┐         ┌────────────────────────────┐
   │ ┌────┐ 巨大缓存       │         │ 小缓存                      │
   │ │Core│ 复杂控制       │         │ ┌─┬─┬─┬─┬─┬─┬─┬─┐ ……数千个 │
   │ └────┘ 分支预测       │         │ │█│█│█│█│█│█│█│█│  简单ALU  │
   │ ┌────┐ 乱序执行       │         │ ├─┼─┼─┼─┼─┼─┼─┼─┤          │
   │ │Core│                │         │ │█│█│█│█│█│█│█│█│          │
   │ └────┘ 4~64 核        │         │ └─┴─┴─┴─┴─┴─┴─┴─┘          │
   └──────────────────────┘         └────────────────────────────┘
   "1 个聪明人快速做完一题"          "1 万个学生同时各做一题"
```

**为什么 GPU 这样设计能行？** 因为深度学习的核心运算——矩阵乘——是"一万个互不依赖的乘加"，天然适合"撒出去同时算"。GPU 不在乎单个乘加要等几百个时钟周期访存，因为它手上还有上万个别的乘加可以立刻顶上，访存延迟被"线程海"盖住了。这叫**延迟隐藏（latency hiding）**。

## 2. 硬件层级：从芯片到 CUDA 核

一块现代 NVIDIA GPU 自顶向下是这样套娃的：

```
GPU 芯片
 └── GPC（图形处理簇，若干个）
      └── TPC
           └── SM（流式多处理器，几十~上百个）★真正的"核心单元"
                ├── CUDA Core（FP32/INT 算术单元，每 SM 64~128 个）
                ├── Tensor Core（矩阵乘加单元，每 SM 4 个）
                ├── Warp Scheduler（决定下一拍哪组线程发指令）
                ├── Register File（寄存器堆，最快的存储）
                └── Shared Memory / L1（SM 内共享的便签内存）
```

**SM（Streaming Multiprocessor）是理解 GPU 的核心。** 你写的一个 CUDA block 会被整体分配到某个 SM 上执行，block 里的线程共享这个 SM 的寄存器和共享内存。一块 GPU 有 N 个 SM，就能同时跑 N 个（及以上）block。

- **CUDA Core**：标量算术单元，做单个 FP32/FP16/INT 乘加。"几千个 CUDA 核"= 所有 SM 的 CUDA 核加总。
- **Warp Scheduler**：每个 SM 有 2~4 个，每拍挑一个就绪的 warp 发射指令。它是延迟隐藏的指挥官——某个 warp 卡在访存上，调度器立刻切到另一个 warp。

> 类比：SM 是一个车间，CUDA 核是车间里的工人，Warp Scheduler 是工头，寄存器/共享内存是工人手边的工作台。

## 3. Tensor Core：为矩阵乘而生

CUDA 核一拍只能做一个标量乘加。**Tensor Core 一拍能做一整块小矩阵的乘加**（典型如 $4\times4$ 矩阵 $D = A\times B + C$），把矩阵乘的吞吐拉高一个数量级。

```
普通 CUDA 核：一拍 1 次乘加
   a × b + c  ─►  1 FLOP-pair

Tensor Core：一拍一个 4×4×4 矩阵乘加（D = A·B + C）
   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
   │ A   │ × │ B   │ + │ C   │ = │ D   │
   │ 4×4 │   │ 4×4 │   │ 4×4 │   │ 4×4 │
   └─────┘   └─────┘   └─────┘   └─────┘
   = 4×4×4 = 64 次乘加 / 拍（远超 1 次）
```

**为什么这么快？** $4\times4$ 矩阵乘需要 $4\times4\times4 = 64$ 次乘加。Tensor Core 把这 64 个乘加做成专用硬件阵列，一拍全出。这就是为什么用 FP16/BF16/INT8 喂 Tensor Core 时，标称算力（TFLOPS）能比纯 FP32 CUDA 核高 8~16 倍。

**代价/权衡**：Tensor Core 偏爱低精度（FP16/BF16/FP8/INT8）和规整的矩阵形状。所以现代训练用**混合精度**：矩阵乘走 Tensor Core 的低精度通道，累加和敏感运算保留 FP32。精度换吞吐，这是 AI-Infra 最重要的工程取舍之一。FLOPs 怎么数见 [[llm-algo/FLOPs]]。

## 4. SIMT 与 warp：32 个线程锁步前进

GPU 不是每个线程各管各的。硬件把线程 **32 个打成一捆**，叫一个 **warp（线程束）**。一个 warp 里的 32 个线程在同一拍**执行同一条指令**，只是各自处理不同的数据。这个模型叫 **SIMT（Single Instruction, Multiple Threads）**。

```
一个 warp = 32 个线程，锁步执行同一条指令
   指令: c[i] = a[i] * b[i]
   线程: T0  T1  T2  ... T31     ← 同一拍全部执行这条乘法
   数据: i=0 i=1 i=2 ... i=31    ← 各自一个不同下标
```

**为什么用 warp？** 32 个线程共享一套取指/译码电路，省下大量控制逻辑的面积和功耗——这正是 GPU 能堆这么多算术单元的原因。

### 分支发散（warp divergence）——必须懂的坑

如果 warp 里的线程走了不同的 if 分支，硬件只能**串行执行每个分支**，没走该分支的线程被屏蔽（masked）空等。

```
   if (x > 0)  A();   else  B();

   走 A 的线程: ██████░░░░░░   （执行 A 时，走 B 的线程空等）
   走 B 的线程: ░░░░░░██████   （执行 B 时，走 A 的线程空等）
                ↑ 总耗时 = A + B，而非 max(A,B)
```

**结论**：写 GPU kernel 要尽量让同一 warp 的线程走相同路径。这条原理深入见 [[ai-infra/ai-hardware/CUDA]]。

## 5. 显存层级：越快越小，越大越慢

这是理解"带宽限"的物理基础。GPU 的存储是一座金字塔，从上到下**速度递减、容量递增、离算术单元越来越远**：

```
   离 ALU 最近 ↑ 最快最小                       延迟  带宽(量级)  容量
 ┌──────────────────────────────────┐
 │ 寄存器 Registers (每线程私有)     │   ~1拍   ~10+ TB/s   每SM 256KB级
 ├──────────────────────────────────┤
 │ 共享内存 Shared / L1 (block 共享) │  ~20拍   ~数 TB/s    每SM ~100~200KB
 ├──────────────────────────────────┤
 │ L2 Cache (全片共享)               │ ~200拍   ~数 TB/s    几十 MB
 ├──────────────────────────────────┤
 │ HBM 显存 (全局 Global Memory)     │ ~400+拍  ~1~3+ TB/s  几十~上百 GB
 └──────────────────────────────────┘
   离 ALU 最远 ↓ 最慢最大
   （注：具体数值随架构不同，量级以官方白皮书为准）
```

关键认知：
- **寄存器**：每个线程私有，访问几乎零延迟。kernel 用寄存器太多会限制同时驻留的线程数（occupancy 下降）。
- **共享内存（Shared Memory）**：同一个 block 的线程共享，是程序员可手动管理的"便签纸"，用来缓存反复用到的数据，避免一次次回 HBM 取。和 L1 共用同一块物理 SRAM。
- **L2**：整片 GPU 共享的硬件缓存，自动管理。
- **HBM（High Bandwidth Memory）**：就是你说的"24GB / 80GB 显存"。容量大但**离算术单元最远、最慢**。所有权重、激活、KV cache 默认住在这里。

**核心矛盾**：算术单元一秒钟能吃下的数据量（带宽）远小于它一秒钟能算的运算量。所以**把数据尽量留在高层级（寄存器/共享内存）反复用**，是 GPU 优化的第一原则。[[llm-optimizer/FlashAttention]] 的全部精髓就是"把注意力计算切块塞进共享内存，避免把巨大的注意力矩阵写回 HBM 再读回来"。

## 6. 算力 vs 带宽：两个独立的数

一块 GPU 的规格表里有两个最关键的数，它们是**独立的**：

| 量 | 含义 | 单位 | 决定什么 |
|----|------|------|----------|
| 峰值算力 $\pi$ | 每秒最多多少次浮点运算 | FLOP/s（TFLOPS） | compute bound 时的天花板 |
| 峰值带宽 $\beta$ | 每秒最多从 HBM 搬多少字节 | Byte/s（TB/s） | memory bound 时的天花板 |

一个常被忽视的事实：**算力的增长远快于带宽的增长**。十年来 GPU 的 FLOPS 涨了几十倍，而 HBM 带宽只涨了几倍。结果就是**越来越多的算子从 compute bound 滑向 memory bound**——芯片算得飞快，却在等数据。这正是 LLM 推理时代"带宽为王"的根因。

## 7. Roofline 模型与算术强度

把上面两个上限画在一张图里，就是著名的 **Roofline 模型**。横轴是**算术强度（Arithmetic Intensity, AI）**，纵轴是可达到的实际吞吐。

**算术强度的定义**——一个算子每从 HBM 搬运 1 字节数据，能做多少次浮点运算：

$$\text{AI} = \frac{\text{总浮点运算次数 (FLOPs)}}{\text{总访存字节数 (Bytes)}} \quad [\text{FLOP/Byte}]$$

可达吞吐被两条线夹住：

$$\text{可达 FLOP/s} = \min\big(\underbrace{\pi}_{\text{算力屋顶}},\ \underbrace{\beta \times \text{AI}}_{\text{带宽斜坡}}\big)$$

```
 FLOP/s
   ↑
 π ┤            ┌──────────────────  ← 算力屋顶(水平)：compute bound 区
   │           /
   │          /  斜率 = 带宽 β
   │         /   (带宽斜坡)：memory bound 区
   │        /
   │       /
   └──────┴──────────────────────────→ 算术强度 AI (FLOP/Byte)
         I*=π/β
       (拐点/脊点 ridge point)
```

**怎么用这张图判断命运**——只看一个比较：把你算子的 AI 和拐点 $I^* = \pi/\beta$ 比大小：

- **AI < $I^*$**：落在左边斜坡 → **memory bound**（带宽限），加再多算力也没用，要省带宽。
- **AI > $I^*$**：落在右边平顶 → **compute bound**（算力限），要用 Tensor Core / 低精度提算力。
- **AI = $I^*$**：恰好在拐点，算力和带宽同时打满，最理想。

**拐点的物理意义**：$I^* = \pi/\beta$ 是"这块硬件要让算力刚好喂饱所需的最低算术强度"。比如 $\pi=300$ TFLOPS、$\beta=2$ TB/s，则 $I^* = 300{,}000 / 2{,}000 = 150$ FLOP/Byte——你的算子算术强度要 ≥150，才可能打满算力。

## 8. 为什么 LLM 的 decode 阶段受带宽限

这是把所有概念串起来的高潮。Transformer 推理分两个阶段：

- **Prefill（处理 prompt）**：一次性处理上千个 token，是大矩阵 × 大矩阵 → **算术强度高 → compute bound**。
- **Decode（逐 token 生成）**：每步只处理 **1 个**新 token，是大矩阵 × 一个向量（GEMV）→ **算术强度极低 → memory bound**。

**直觉解释**：decode 每生成 1 个 token，都要把模型**全部权重**从 HBM 读一遍（每个权重只参与极少量乘加就被丢掉），还要读 KV cache。算的次数少，搬的字节多，AI 自然趴在地上。

```
 Prefill: [大矩阵 W] × [N个token的大矩阵 X]   每个权重被 N 个 token 复用
          → 权重读一次，算 N 次 → AI 高 → compute bound

 Decode:  [大矩阵 W] × [1个token的向量 x]      每个权重只被 1 个 token 用
          → 权重读一次，算 1 次 → AI ≈ 2 → memory bound
                                          ↑搬权重的时间 >> 算的时间
```

**这解释了一堆现象**：
- decode 速度（token/s）几乎正比于**显存带宽**，而不是 TFLOPS——换块带宽更高的卡，decode 立刻变快；换块算力翻倍但带宽不变的卡，decode 几乎没变化。
- **批处理（batching）有效**：把多个请求的 token 拼在一起，让同一份权重被多个 token 复用，AI 上升 → 把 memory bound 往 compute bound 推。这就是为什么推理服务拼命攒 batch（continuous batching）。
- **量化（INT8/FP8/INT4）对 decode 极有效**：权重字节数减半/减四，搬运量直接腰斩，decode 直接变快——因为瓶颈本就是搬权重的字节数。
- **FlashAttention** 之所以快，也是同一逻辑：把注意力中间矩阵留在共享内存里算完，避免把 $N\times N$ 的大矩阵反复写回/读出 HBM，砍掉了访存字节，提高了有效 AI。详见 [[llm-optimizer/FlashAttention]]。

## 9. 手算：判断一个算子是 compute 还是 memory bound

> 目标：拿真实数字走一遍"数 FLOPs → 数 Bytes → 算 AI → 和拐点比"。
> 假设硬件：峰值算力 $\pi = 300$ TFLOPS（FP16），峰值带宽 $\beta = 2$ TB/s。
> 先算拐点：$I^* = \pi/\beta = \dfrac{300\times10^{12}}{2\times10^{12}} = 150$ FLOP/Byte。

### 例 A：矩阵乘（GEMM），训练里的主力

设 $C = A \times B$，其中 $A$ 是 $M\times K$、$B$ 是 $K\times N$，取 $M=N=K=4096$，数据为 FP16（2 字节/元素）。

**第 1 步：数 FLOPs。** 输出 $C$ 有 $M\times N$ 个元素，每个元素是一行点一列 = $K$ 次乘 + $K$ 次加 = $2K$ 次浮点运算。

$$\text{FLOPs} = 2 \times M \times N \times K = 2 \times 4096^3 \approx 1.37\times10^{11}$$

**第 2 步：数 Bytes（理想：每个矩阵只读/写一次）。**

$$\text{Bytes} = (\underbrace{M\!\cdot\!K}_{A} + \underbrace{K\!\cdot\!N}_{B} + \underbrace{M\!\cdot\!N}_{C})\times 2 = 3\times4096^2\times2 \approx 1.01\times10^{8}$$

**第 3 步：算 AI。**

$$\text{AI} = \frac{1.37\times10^{11}}{1.01\times10^{8}} \approx 1365 \ \text{FLOP/Byte}$$

**第 4 步：和拐点比。** $1365 \gg 150 = I^*$ → **compute bound**。✅
大矩阵乘是算力受限——所以训练才疯狂依赖 Tensor Core 提算力。

### 例 B：逐元素加法（Add），激活函数那类

设 $C = A + B$，三个张量都是 $N=10^8$ 个 FP16 元素。

**FLOPs**：每个元素 1 次加法 → $\text{FLOPs} = N = 10^8$。
**Bytes**：读 $A$、读 $B$、写 $C$，共 3 个张量 → $\text{Bytes} = 3\times N\times 2 = 6\times10^8$。
**AI**：

$$\text{AI} = \frac{10^8}{6\times10^8} \approx 0.17 \ \text{FLOP/Byte}$$

**比较**：$0.17 \lll 150$ → **memory bound**。✅
逐元素算子永远是带宽受限——所以框架要做**算子融合（fusion）**，把多个逐元素操作串起来，让数据在寄存器里算完再写回，只读写一次 HBM。

### 例 C：LLM decode 的一层 GEMV

decode 时一个权重矩阵 $W$（$d\times d$，取 $d=4096$，FP16）乘以单个 token 向量 $x$（$d\times1$）。

**FLOPs**：$2\times d\times d = 2\times4096^2 \approx 3.36\times10^7$。
**Bytes**：主要是读权重 $W$ → $d\times d\times2 = 4096^2\times2 \approx 3.36\times10^7$（$x$ 和输出向量字节可忽略）。
**AI**：

$$\text{AI} = \frac{2\,d^2}{2\,d^2} = 2 \ \text{FLOP/Byte}\ (\text{近似})$$

**比较**：$2 \lll 150$ → **memory bound**。✅✅
这就是第 8 节的数值证据：decode 的算术强度约等于 2，趴在 roofline 的最左侧斜坡上。**搬权重的字节数才是瓶颈**，所以量化（把 2 字节砍成 1 字节甚至 0.5 字节）和 batching（让分母不变但分子翻倍）是 decode 提速的两大法宝。

> 一张表总结三个例子：

| 算子 | FLOPs | Bytes | AI | vs $I^*{=}150$ | 结论 |
|------|-------|-------|----|----|------|
| GEMM 4096³ | 1.37e11 | 1.01e8 | ~1365 | ≫ | compute bound |
| 逐元素 Add | 1e8 | 6e8 | ~0.17 | ≪ | memory bound |
| decode GEMV | 3.36e7 | 3.36e7 | ~2 | ≪ | memory bound |

## 对照表：概念速查

| 概念 | 一句话 | 类比 |
|------|--------|------|
| SM | GPU 真正的"核心"，独立调度单元 | 车间 |
| CUDA Core | 标量乘加单元，一拍 1 次 | 工人 |
| Tensor Core | 矩阵乘加单元，一拍 64 次 | 流水线机床 |
| warp | 锁步执行的 32 线程 | 32 人方阵走同一步 |
| SIMT | 单指令多线程 | 一个口令全班执行 |
| 寄存器 | 最快最小，线程私有 | 手边便利贴 |
| 共享内存 | block 内共享的可控缓存 | 小组白板 |
| HBM | 大而慢的全局显存 | 仓库 |
| 算力 π | 每秒最多算多少 FLOP | 工人手速 |
| 带宽 β | 每秒最多搬多少字节 | 送料速度 |
| 算术强度 AI | 每搬 1 字节算几次 | 一车料能加工几次 |
| 拐点 I*=π/β | 算力刚好被喂饱的最低 AI | 料够不够喂满工人 |

## 复杂度/带宽直觉表

| 阶段/算子 | 典型 AI | 瓶颈 | 提速手段 |
|-----------|---------|------|----------|
| 训练 GEMM / prefill | 高（百~千） | compute | Tensor Core、低精度 |
| decode GEMV | 极低（~2） | memory | 量化、batching |
| 逐元素 / norm / 激活 | 极低（<1） | memory | 算子融合 |
| 注意力（朴素） | 中但访存重 | memory | FlashAttention（留共享内存） |

## 常见问题

| 疑问 | 真相 |
|------|------|
| "买算力更高的卡，LLM 推理就更快？" | 单请求 decode 是 **memory bound**，看的是**带宽**不是 TFLOPS。带宽不变则 decode 几乎不变。 |
| "GPU 的几千个核和 CPU 的核一样吗？" | 不一样。CUDA 核是简单 ALU，GPU 真正的调度单元是 **SM**（几十~上百个）。 |
| "为什么要用 FP16/INT8 这种低精度？" | 算得快（Tensor Core）+ 搬得少（字节减半）。对 memory bound 的 decode，省字节直接等于提速。 |
| "warp 里线程写 if-else 有什么问题？" | 分支发散，硬件串行跑两个分支、互相空等，吞吐打折。 |
| "共享内存和 L1、和显存是一回事吗？" | 共享内存与 L1 共用 SM 内 SRAM（快、小、可控）；显存(HBM)是片外、大、慢。 |
| "FlashAttention 为什么快？" | 把注意力中间结果留在共享内存里算完，砍掉对 HBM 的反复读写，提高有效算术强度。 |
| "batching 为什么能提推理吞吐？" | 让同一份权重被多个 token 复用，抬高算术强度，把 memory bound 往 compute bound 推。 |
| "怎么一眼判断算子被谁限？" | 算 AI = FLOPs/Bytes，和拐点 $I^*=\pi/\beta$ 比大小：小于→带宽限，大于→算力限。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全站导航总图
- [[ai-infra/ai-hardware/CUDA]] — CUDA 编程模型：grid/block/thread、warp、共享内存怎么写
- [[llm-algo/FLOPs]] — 怎么数模型的 FLOPs（本文 AI 分子的来源）
- [[llm-optimizer/FlashAttention]] — 用共享内存重写注意力，砍 HBM 访存的经典案例

---

- [GPU 工作原理解析](https://zhuanlan.zhihu.com/p/697694330)
- [GPU 架构与 CUDA 关系](https://zhuanlan.zhihu.com/p/697746975)
