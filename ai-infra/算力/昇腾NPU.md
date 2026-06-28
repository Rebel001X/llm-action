# 昇腾 NPU (Ascend)

> 华为面向 AI 训练/推理的专用处理器(NPU),以"达芬奇(Da Vinci)"架构 + CANN 软件栈对标 NVIDIA GPU+CUDA,是国产 AI 算力替代的核心载体。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/ai-hardware/README]] [[ai-infra/算力/推理芯片]]

## 阅读地图

| 节 | 主题 | 一句话 | 难度 |
|----|------|--------|------|
| 0 | 锚点 | NPU 是什么、为什么不是 GPU | ★ |
| 1 | 地基 | 张量计算 / 脉动阵列 / Cube 单元 | ★★ |
| 2 | 达芬奇架构 | Cube+Vector+Scalar 三引擎 | ★★★ |
| 3 | 昇腾 910/310 | 芯片规格与代际 | ★★ |
| 4 | 服务器/集群 | Atlas 系列 + HCCS/HCCL 互联 | ★★★ |
| 5 | CANN 软件栈 | AscendCL/ACLNN/算子/图编译 | ★★★ |
| 6 | vs CUDA | 生态逐层对照 | ★★★ |
| 7 | MindSpore/MindFormers | 框架与大模型套件 | ★★ |
| 8 | 迁移要点 | PyTorch→NPU 实操 | ★★★ |
| 9 | 数值例子 | 算力/显存/带宽手算 | ★★ |
| 10 | 国产替代 | 生态地位与瓶颈 | ★★ |

## 0. 一句话锚点

- **NPU = Neural Processing Unit**:为神经网络的核心运算——**矩阵乘加(MAC)**——量身定制的 ASIC。
- 与 GPU 的本质区别:GPU 是"通用并行处理器"(成千上万个 SIMT 标量核),NPU 把"乘加"硬连成一个**矩阵阵列单元(Cube)**,用更少的控制逻辑换取**单位面积/功耗下更高的矩阵吞吐**。
- 昇腾(Ascend)是华为海思的 NPU 产品线;**昇腾 910** 主打训练,**昇腾 310** 主打边缘/推理。软件栈叫 **CANN**,框架叫 **MindSpore**。
- 记忆锚:**昇腾之于华为 ≈ GPU 之于 NVIDIA**;**CANN 之于昇腾 ≈ CUDA 之于 NVIDIA**;**MindSpore ≈ PyTorch/TensorFlow**。

## 1. 地基:为什么需要专用矩阵单元

### 1.1 深度学习的计算本质是矩阵乘

一个全连接层 / 一个 Transformer 的 Q·Kᵀ,本质都是矩阵乘法 $C = A \times B$。对 $M\times K$ 乘 $K\times N$:

$$C_{ij} = \sum_{k=1}^{K} A_{ik} \cdot B_{kj}$$

总乘加次数 $= M \times N \times K$。一次"乘加"(MAC)= 1 个乘法 + 1 个加法。**整个大模型训练 99% 的浮点运算都是 MAC**。

### 1.2 GPU 怎么做 vs NPU 怎么做

```
GPU(SIMT 标量核):              NPU Cube(脉动阵列 systolic array):
每个线程算一个 C_ij,             数据像"水波"在 16x16 的 PE 阵列里流动
靠成千上万线程并行                每个时钟,每个 PE 做一次 MAC 并把结果传给邻居
┌──────────────────┐            A 行→ ┌──┬──┬──┬──┐
│ thread thread ...│                  │PE│PE│PE│PE│  B 列
│ thread thread ...│                  ├──┼──┼──┼──┤   ↓
│  (大量寄存器/调度) │                  │PE│PE│PE│PE│
└──────────────────┘                  └──┴──┴──┴──┘
控制逻辑占比高,灵活             控制逻辑极少,数据复用率极高、能效高
```

**脉动阵列(systolic array)**核心思想:让一个数据(如 A 的一行)在阵列里被多个 PE 复用,而不是反复从存储器搬运。**搬数据(访存)比算数据(计算)更耗能**,这是 NPU 节能的根本原因。

### 1.3 关键术语

- **PE(Processing Element)**:一个最小乘加单元。
- **Cube**:达芬奇里的矩阵运算单元,一拍可完成 $16\times16\times16$ 的矩阵乘累加。
- **FLOPS**:每秒浮点运算次数;一次 MAC 算 2 FLOPs。
- **FP16/BF16/INT8**:低精度,AI 训练/推理主力数据类型。

## 2. 达芬奇(Da Vinci)架构:三引擎异构

达芬奇是昇腾的 AI Core 微架构。一个 **AI Core** 内有三类计算单元 + 多级缓冲:

```
            ┌─────────────────── AI Core(达芬奇)───────────────────┐
   HBM/L2 →│  ┌────────┐  ┌──────────┐  ┌──────────┐               │
            │  │ Scalar │  │  Vector  │  │   Cube   │ ← 核心        │
            │  │  标量   │  │  向量化   │  │ 16x16x16 │ 矩阵乘加      │
            │  │ 控制/标量│  │ 激活/归一 │  │  MAC阵列  │              │
            │  └────────┘  └──────────┘  └──────────┘               │
            │   ↑ Buffer:  L0A / L0B(给Cube) L0C(累加) UB(给Vector)│
            │   ┌──────────────── L1 Buffer(片上)──────────────┐   │
            └───┴───────────────────────────────────────────────┴───┘
```

| 引擎 | 作用 | 类比 |
|------|------|------|
| **Cube** | 矩阵乘累加(MatMul/Conv) | GPU Tensor Core |
| **Vector** | 逐元素:激活、LayerNorm、Softmax、加偏置 | GPU CUDA Core(向量化) |
| **Scalar** | 标量运算、循环/分支控制、地址计算 | GPU 标量单元/调度 |

**为什么三引擎?** Transformer 一层里:`MatMul(Cube) → 加偏置/LayerNorm/激活(Vector) → 控制流(Scalar)` 是流水的。三引擎可**并行+流水**,Cube 算下一块时 Vector 处理上一块结果,隐藏延迟。

**多级 Buffer(显式 scratchpad)**:与 GPU 自动缓存不同,达芬奇的 L0/L1/UB 由**编译器/算子开发者显式管理**数据搬运(DMA),这既是高能效的来源,也是**手写算子门槛高**的原因。

## 3. 昇腾芯片代际与规格(以官方为准)

```
昇腾家族(典型公开规格,约):
  昇腾 310 (Ascend 310)  推理/边缘  ~8 TFLOPS FP16 / 16 TOPS INT8,低功耗(~8W级)
  昇腾 310P             推理增强
  昇腾 910 (Ascend 910)  训练旗舰   ~320 TFLOPS FP16(约),7nm,HBM
  昇腾 910B             训练主力(当前国产训练卡主力,对标 A100 量级)
  昇腾 910C             更新一代(对标更高,以官方为准)
```

要点:
- **910 系列 = 训练**,集成 **HBM** 高带宽显存,多卡互联(HCCS),用于 Atlas 训练服务器/集群。
- **310 系列 = 推理/边缘**,低功耗,INT8 主力,用于 Atlas 推理卡、Atlas 200/500 边缘盒、智能摄像头。
- 具体 TFLOPS/显存容量/制程随版本变化大,**精确数字以华为官方 datasheet 为准**,不可硬背。
- 顶部出现的 **Atlas 800-9000** 是训练服务器整机型号(见下节),不是芯片名。

## 4. 服务器与集群:Atlas + HCCL

单芯片不够,大模型要成百上千卡。昇腾的整机/集群体系:

```
芯片(910B) → 模组/卡 → Atlas 服务器 → 集群(Atlas 900 SuperPoD)
   │            │            │                  │
 AI Core×N   OAM/PCIe卡   8卡整机           成千上万卡
                          (Atlas 800)        + 光互联

单机 8 卡互联拓扑(HCCS,类比 NVLink):
   ┌────┐  HCCS   ┌────┐
   │ 910├────────┤ 910│      全互联/Mesh,卡间高带宽
   └─┬──┘         └──┬─┘     避免走 PCIe 瓶颈
   ┌─┴──┐         ┌──┴─┐
   │ 910├────────┤ 910│
   └────┘         └────┘
```

- **HCCS**:卡间高速互联(对标 NVIDIA **NVLink**),让 8 卡像一个大显存池,支撑张量并行。
- **HCCL(Huawei Collective Communication Library)**:集合通信库,提供 AllReduce/AllGather/Broadcast 等(对标 NVIDIA **NCCL**),是数据/张量/流水并行的通信底座。
- **Atlas 800 / 900**:训练服务器与超节点(SuperPoD),用 RoCE/光互联组大规模集群。

## 5. CANN 软件栈:昇腾的"CUDA"

**CANN(Compute Architecture for Neural Networks)**是昇腾的异构计算软件栈,从框架到硬件逐层:

```
┌───────────────────────────────────────────────┐
│  框架层  MindSpore / PyTorch(torch_npu)/TensorFlow │
├───────────────────────────────────────────────┤
│  CANN                                          │
│   ├ GE(Graph Engine):图编译/融合/调度          │
│   ├ AscendCL(ACL):运行时 C/C++/Python API      │
│   ├ ACLNN / aclnn 算子库(类 cuDNN/cuBLAS)      │
│   ├ AOL 算子库 + 自定义算子(Ascend C 语言写)    │
│   ├ TBE/AscendC:算子开发(类 CUDA kernel)       │
│   └ HCCL:集合通信(类 NCCL)                     │
├───────────────────────────────────────────────┤
│  Runtime / Driver(类 CUDA Runtime+Driver)      │
├───────────────────────────────────────────────┤
│  昇腾硬件(达芬奇 AI Core)                        │
└───────────────────────────────────────────────┘
```

逐层解释:
- **AscendCL(ACL)**:应用直接调用的运行时 API——分配 device 内存、拷贝 H2D/D2H、加载模型、下发执行(对应 CUDA Runtime API)。
- **算子库(aclnn/AOL)**:预置高性能算子(MatMul、Conv、Attention 等),对应 cuBLAS/cuDNN。
- **Ascend C**:用类 C++ 写自定义算子的编程语言/编译器,显式管理 Cube/Vector/Buffer(对应写 CUDA kernel)。**门槛是显式 Buffer 搬运**。
- **GE 图引擎**:把整网编译成图,做**算子融合**(如 Conv+BN+ReLU 融合成一个核)、内存复用、调度,是性能关键。
- **ATC(Ascend Tensor Compiler)**:离线把模型转成 `.om` 格式,推理时加载,类似 TensorRT 的 engine。

## 6. 与 CUDA 生态逐层对照(核心记忆表)

| 能力层 | NVIDIA(CUDA 生态) | 华为昇腾(CANN 生态) |
|--------|---------------------|----------------------|
| 芯片架构 | GPU(SIMT)+Tensor Core | NPU 达芬奇(Cube/Vector/Scalar) |
| 卡间互联 | NVLink / NVSwitch | HCCS |
| 集合通信 | NCCL | HCCL |
| 运行时 API | CUDA Runtime/Driver | AscendCL(ACL) |
| 数学库 | cuBLAS / cuDNN | aclnn / AOL 算子库 |
| 算子编程 | CUDA C++ kernel | Ascend C |
| 图/推理优化 | TensorRT | ATC + GE(.om) |
| 框架后端 | PyTorch CUDA | torch_npu / MindSpore |
| 训练大模型套件 | Megatron-LM | MindFormers / MindSpeed |
| 编程模型差异 | 隐式 L1/L2 缓存 | **显式 Buffer 管理** |

```
迁移心智图:把"cuda语义"映射到"ascend语义"
  tensor.cuda()      ──►  tensor.npu()
  torch.cuda.*       ──►  torch.npu.*  (torch_npu)
  nccl              ──►  hccl
  *.engine(TRT)     ──►  *.om(ATC)
  nvidia-smi        ──►  npu-smi info
```

## 7. MindSpore 与 MindFormers

- **MindSpore**:华为自研深度学习框架(对标 PyTorch/TensorFlow)。特点:
  - **动静统一**:动态图(调试友好)/静态图(`@ms.jit` 编译加速)可切换。
  - **自动并行**:声明式地自动切分数据/模型/流水并行,降低分布式编码量。
  - 与昇腾深度协同(也支持 GPU/CPU),在 NPU 上能拿到最佳图编译收益。
- **MindFormers**:基于 MindSpore 的大模型套件(对标 Megatron-LM + HuggingFace Transformers),内置 LLaMA/GLM/Qwen 等结构、并行配置、预训练/微调/推理流程。
- **MindSpeed**:面向 PyTorch on Ascend 的大模型分布式加速库(融合算子、并行策略),让 PyTorch 用户也能在 910B 上高效训大模型。

```
两条上手路线:
  路线A(原生最优): PyTorch/HF 代码 → 不变,装 torch_npu + MindSpeed → 跑在 NPU
  路线B(华为栈):   MindSpore + MindFormers → 图编译/自动并行收益最大,但需学新框架
实践:多数团队走路线A(改动小),性能调优时再深入 CANN/Ascend C。
```

## 8. PyTorch → 昇腾迁移要点(实操)

### 8.1 三步上手

```python
# 1) 安装并导入 torch_npu(关键:import 后 'npu' 设备才注册)
import torch
import torch_npu                      # 必须 import,才能用 .npu()

# 2) 把 cuda 全部替换为 npu
device = "npu:0"                       # 原 "cuda:0"
model = model.to(device)
x = x.to(device)

# 3) 分布式:后端用 hccl 替代 nccl
import torch.distributed as dist
dist.init_process_group(backend="hccl")   # 原 backend="nccl"
```

### 8.2 迁移检查清单

| 项 | CUDA | 昇腾 | 注意 |
|----|------|------|------|
| 设备 | `cuda` | `npu` | 需 `import torch_npu` |
| 通信后端 | `nccl` | `hccl` | 分布式必改 |
| 混合精度 | AMP fp16/bf16 | 支持,优先 **bf16** | 数值更稳 |
| 算子覆盖 | 全 | **大部分** | 个别算子未支持需替换/自定义 |
| 监控 | `nvidia-smi` | `npu-smi info` | 看利用率/显存/温度 |
| 性能分析 | Nsight | **msprof / MindStudio** | 抓 Cube 利用率 |

### 8.3 常见坑(为什么)

- **算子缺失/回退到 CPU**:某算子 NPU 未实现 → 隐式 D2H 到 CPU 算再传回,严重掉速。**用 profiler 抓"host bound/算子回退"**。
- **未触发图编译/融合**:动态图逐算子下发,Host 下发开销大 → 用图模式或绑定融合算子,提升 Cube 利用率。
- **AllReduce 慢**:确认走 HCCS 而非 PCIe;`HCCL_*` 环境变量配拓扑。
- **精度对齐**:优先 **bf16**(指数位与 fp32 同,溢出风险小),先对齐 loss 曲线再追性能。

## 9. 数值例子(手算,数字为示意/以官方为准)

### 例1:Cube 单元的算力来源

设一个 AI Core 的 Cube 每拍做 $16\times16\times16$ 矩阵乘累加 = $16^3 = 4096$ 次 MAC = $4096\times2 = 8192$ FLOPs/拍。

若主频 $f=1.0\text{ GHz}=10^9$ 拍/秒,单 Core:
$$8192 \times 10^9 \approx 8.2\ \text{TFLOPS (FP16)}$$
若一颗 910 有约 32 个 AI Core:$8.2 \times 32 \approx 262\ \text{TFLOPS}$,量级与公开宣称的"约 320 TFLOPS FP16"吻合(数字示意)。

> 直觉:**算力 ≈ 阵列规模³ × 频率 × Core 数 × 2**。把矩阵硬连成阵列,是 NPU 算力高的根源。

### 例2:训练一步要多少算力?

经验公式:训练 1 个 token 的浮点开销 $\approx 6N$(N=参数量,正向2N+反向4N):
$$\text{FLOPs} = 6 \times N \times D \quad (D=\text{训练 token 数})$$

训练 7B 模型、1T tokens:
$$6 \times 7\times10^9 \times 1\times10^{12} = 4.2\times10^{22}\ \text{FLOPs}$$

设单卡有效算力 200 TFLOPS、MFU(算力利用率)40%:
- 有效算力 $=200\times10^{12}\times0.4=8\times10^{13}$ FLOPS
- 单卡需 $4.2\times10^{22} / 8\times10^{13} \approx 5.25\times10^8$ 秒 ≈ **16.6 卡·年**
- 用 1024 卡:约 $16.6\times365/1024 \approx 5.9$ 天(理想线性扩展)

> **MFU 是 NPU 落地的胜负手**:峰值算力相近时,谁的编译/融合/通信好,谁的有效吞吐高。

### 例3:显存够不够装 7B?

参数+优化器(Adam,fp16 训练,混合精度,经验 ~16 字节/参数:参数2+梯度2+Adam态m/v各4+fp32主参4):
$$7\times10^9 \times 16\ \text{B} = 112\ \text{GB}$$

单卡(设 HBM 64GB,以官方为准)装不下 → **必须多卡张量/流水/ZeRO 并行**,这就是为什么要 HCCS+HCCL。推理时仅需参数(7B×2B≈14GB)+KV Cache,单卡可行。

## 10. 国产替代:地位、价值与瓶颈

```
国产 AI 算力替代图谱(NPU 阵营,昇腾为代表):
  硬件   昇腾 910B/C  ── 当前国产训练主力,已支撑多个国产大模型预训练
  互联   HCCS+HCCL   ── 自有方案,绕开 NVLink/NCCL
  软件   CANN        ── 在补齐 CUDA 多年生态(算子覆盖/调试/三方库)
  框架   MindSpore   ── 自研框架 + torch_npu 兼容 PyTorch 降低迁移门槛
  动因   出口管制     ── A100/H100 受限,推动昇腾成为"可获得的高端算力"
```

- **战略意义**:在高端 GPU 受出口管制的背景下,昇腾是国内能稳定获得、可规模化的训练算力,已实际承载多个国产大模型的训练/推理。
- **核心壁垒在软件生态**:硬件峰值算力可以追,但 CUDA 二十年积累的算子覆盖、三方库、调试工具、社区代码,是昇腾最需补齐的。**生态成熟度 > 单卡峰值**。
- **现实策略**:`torch_npu` 兼容 PyTorch 降低迁移成本;华为持续扩 CANN 算子覆盖、开源 MindFormers/MindSpeed;以"软硬件全栈一体优化"换 MFU。
- **选型建议**:推理/国产化合规场景昇腾性价比高;前沿训练若依赖大量自定义 CUDA 算子,迁移成本需评估算子覆盖率与 profiler 实测吞吐,不能只看峰值 TFLOPS。

## 常见问题

| 问题 | 答 |
|------|----|
| NPU 和 GPU 谁强? | 不同定位。NPU 在矩阵乘能效上有优势,GPU 在通用性/生态上领先,要看 MFU 实测而非峰值 |
| CANN 等于 CUDA 吗? | 定位等价(都是异构计算栈),但生态成熟度差距仍在补 |
| PyTorch 能用昇腾吗? | 能。`import torch_npu` + `.npu()` + `hccl`,多数代码小改即可 |
| 必须学 MindSpore 吗? | 不必须。走 torch_npu 路线即可;追极致性能再用 MindSpore/Ascend C |
| 达芬奇三引擎是什么? | Cube(矩阵)+Vector(逐元素)+Scalar(控制),并行流水 |
| 910 和 310 区别? | 910=训练(HBM、高算力),310=推理/边缘(低功耗、INT8) |
| HCCL/HCCS 对标谁? | HCCL≈NCCL(通信库),HCCS≈NVLink(卡间互联) |
| 迁移最大坑? | 算子缺失回退 CPU + 未触发图融合,务必用 profiler 抓 |
| .om 是什么? | ATC 离线编译产物,类似 TensorRT engine,推理加载 |

## 🔗 跳转链接

- [[00-知识地图]] — 全站导航总图
- [[ai-infra/ai-hardware/README]] — AI 硬件总览(GPU/NPU/TPU 对比)
- [[ai-infra/算力/推理芯片]] — 推理芯片专题(310/TensorRT/量化)
