# TensorRT

> NVIDIA 官方的**深度学习推理优化引擎(SDK)**：把训练好的模型「编译」成一份针对特定 GPU 高度优化的可执行体(engine)，靠图优化 + 层融合 + 低精度 + kernel 自动调优把推理延迟压到极限。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/TensorRT-Model-Optimizer]] [[llm-inference/README]]

---

## 阅读地图

| 节 | 你会得到什么 | 难度 |
|---|---|---|
| 0. 一句话锚点 | TensorRT 到底是「编译器」还是「框架」 | ★ |
| 1. 地基 | 训练框架直接推理为什么慢，TensorRT 解决什么 | ★ |
| 2. 整体架构 | Builder / Engine / Runtime 三件套 + 数据流图 | ★★ |
| 3. 图优化 & 层融合 | 算子怎么被「合并」，为什么能减访存 | ★★★ |
| 4. 精度校准 | FP16/INT8/FP8，INT8 校准的数学原理 | ★★★ |
| 5. Kernel 自动调优 | tactic 选择 / autotuning 是怎么回事 | ★★ |
| 6. Build Engine 全流程 | 从 ONNX 到 .engine 的完整调用链 | ★★ |
| 7. 与 TensorRT-LLM 关系 | 为什么 LLM 要单独一套 | ★★ |
| 8. 为什么快(总账) | 五个加速来源的归因 | ★★ |
| 9. 配置/流程示例 | trtexec、Python API 关键参数含义 | ★★ |
| 常见问题 | 踩坑速查 | ★ |

---

## 0. 一句话锚点

**TensorRT 不是一个像 PyTorch 那样的「框架」，而是一个「推理编译器 + 运行时」。**

- 输入：一个已训练好的网络结构 + 权重（通常以 ONNX 表示）。
- 处理：对计算图做一系列**离线优化**（融合、选精度、挑最快 kernel），并把结果固化。
- 输出：一个 **engine（`.engine`/`.plan` 二进制）**，它**绑定到一张特定 GPU 型号 + 一套配置**，换硬件/换 TensorRT 版本通常要**重新编译**。

类比：PyTorch 是「解释执行」（每次 forward 都临时决定怎么算），TensorRT 是「提前把整张图编译成最优机器码」。这就是它快的根因。

---

## 1. 地基：训练框架直接推理为什么慢

拿 PyTorch eager 模式跑推理，慢在哪？逐个原子拆开：

1. **逐算子分发开销**：每个 `conv`、`add`、`relu` 都是一次独立的 CUDA kernel 启动（launch），CPU 要不断给 GPU 下命令。小算子多时，GPU 大量时间在「等下一条指令」而不是算数。
2. **访存(memory-bound)瓶颈**：`conv → bias → relu` 三步，每步都把中间结果写回显存(DRAM)再读出来。GPU 的算力(TFLOPS)远超显存带宽(GB/s)，所以很多算子卡在「搬数据」而不是「算」。
3. **精度浪费**：训练用 FP32 是为了梯度稳定；推理时 FP16/INT8 往往够用，但框架默认不替你降精度。
4. **kernel 不是为你这块卡 + 这个 shape 选的**：通用 kernel ≠ 当前 GPU 架构(SM 数、共享内存、Tensor Core)下的最优实现。

TensorRT 就是逐条消灭上面 4 个浪费。下面一节一节看它怎么做。

```
        训练框架 eager 推理              TensorRT engine 推理
   ┌──────────────────────────┐   ┌──────────────────────────┐
   │ conv  → DRAM 写           │   │  ┌────────────────────┐  │
   │ bias  → DRAM 读/写        │   │  │  融合 kernel        │  │
   │ relu  → DRAM 读/写        │   │  │  conv+bias+relu    │  │
   │ (3 次 launch, 多次访存)    │   │  │  一次 launch        │  │
   └──────────────────────────┘   │  │  中间值留寄存器/SMEM │  │
                                   │  └────────────────────┘  │
                                   └──────────────────────────┘
```

---

## 2. 整体架构：Builder / Engine / Runtime 三件套

TensorRT 的世界可以归到三个核心对象。理解了它们的边界，整个 API 就通了。

```
┌─────────────────────────────────────────────────────────────────┐
│                       ① 构建期 (Build-time, 离线/一次性)            │
│                                                                   │
│   ONNX / API 搭网络                                                │
│        │                                                          │
│        ▼                                                          │
│   ┌──────────────┐   解析   ┌──────────────┐                      │
│   │ Parser       │ ───────► │ INetwork     │  (图的逻辑表示)        │
│   │ (ONNX/Caffe) │          │ Definition   │                      │
│   └──────────────┘          └──────┬───────┘                      │
│                                    │                              │
│                          ┌─────────▼──────────┐                   │
│                          │  Builder + Config   │ ◄── 精度/显存/校准 │
│                          │  ┌───────────────┐  │                   │
│                          │  │ 图优化/层融合   │  │                   │
│                          │  │ 精度选择       │  │                   │
│                          │  │ kernel autotune│  │                   │
│                          │  └───────────────┘  │                   │
│                          └─────────┬──────────┘                   │
│                                    │ 序列化                        │
│                                    ▼                              │
│                          ┌──────────────────┐                     │
│                          │ ② Engine (.plan)  │  ← 绑定 GPU 型号     │
│                          └─────────┬─────────┘                     │
└────────────────────────────────────┼──────────────────────────────┘
                                     │ 反序列化 (上线时加载)
┌────────────────────────────────────┼──────────────────────────────┐
│                       ③ 运行期 (Runtime, 每次请求)                  │
│                                     ▼                              │
│                    ┌────────────────────────────┐                 │
│                    │ ExecutionContext            │                 │
│                    │  - 绑定输入/输出显存 buffer    │                 │
│                    │  - enqueue() 异步执行 (stream)│                 │
│                    └────────────────────────────┘                 │
└───────────────────────────────────────────────────────────────────┘
```

核心对象速记表：

| 对象 | 角色 | 类比 |
|---|---|---|
| `Builder` | 编译器主体，跑所有优化 | gcc 这个程序 |
| `BuilderConfig` | 编译开关（精度、显存上限、校准器） | 编译 flag `-O3 -march=...` |
| `INetworkDefinition` | 网络的逻辑图 | 源代码 AST |
| `ICudaEngine` | 编译产物（已优化、已序列化） | 编译出的可执行文件 |
| `IExecutionContext` | 一次推理的运行态（含中间显存） | 进程运行实例 |

**关键认知**：构建（Build）很贵（几秒到几十分钟，要做 autotuning），但**只做一次**；运行（Runtime）很便宜，每个请求复用同一个 engine。所以工程上一定是「离线 build → 序列化存盘 → 线上反序列化加载」。

---

## 3. 图优化 & 层融合（最大加速来源）

这是 TensorRT 的「灵魂」。分两类：**纵向融合**和**横向融合**。

### 3.1 纵向融合（垂直，把串联的算子合并）

把数据流上前后相连的算子合成一个 kernel。最经典的是 **Conv + Bias + ReLU → CBR**：

```
融合前 (3 kernel, 中间结果两次往返 DRAM):
   x ──►[Conv]──► t1(写DRAM) ──►[+Bias]──► t2(写DRAM) ──►[ReLU]──► y

融合后 (1 kernel, 中间结果只在寄存器/共享内存里流转):
   x ──►[  Conv+Bias+ReLU 融合算子  ]──► y
```

**为什么省**：原来 `t1`、`t2` 都要写回显存再读出（每个元素一来一回）。融合后中间值不落地，直接在 on-chip 寄存器里传给下一步。对**访存受限**的算子，省掉的 DRAM 流量直接等于省掉的时间。

数值直觉：设激活张量 $N$ 个元素、每个 4 字节，融合前两次中间落地约 $2\times2\times4N = 16N$ 字节的 DRAM 读写；融合后接近 0。省下的字节数除以显存带宽就是省下的时间。

### 3.2 横向融合（水平，把并行的同构算子合并）

若多个算子**消费同一个输入、做同类运算**（例如 Inception 里多个并行的 1×1 Conv），TensorRT 把它们合成一个更宽的 kernel，一次启动算完，减少 launch 次数并提高 GPU 占用率。

```
融合前:           融合后:
        ┌─[1x1 Conv a]            ┌──────────────────┐
  x ────┼─[1x1 Conv b]    x ────► │ 合并的宽 Conv     │ ──► 拆回 a/b/c
        └─[1x1 Conv c]            └──────────────────┘
  (3 次 launch)                    (1 次 launch)
```

### 3.3 其它图级优化

- **常量折叠 (constant folding)**：编译期就能算出的子图（如对常量做的运算）直接预计算成常量。
- **消除无用层**：恒等(identity)、no-op 的 concat/reshape 在内存布局允许时被抹掉。
- **维度/布局优化**：选择对 Tensor Core 友好的内存排布（如 NHWC、`NC/xHWx` 这类 vectorized 格式），减少额外的 transpose。

> 注意：具体哪些算子能融合、融合成什么名字，**随 TensorRT 版本演进**，以官方 release note / `--verbose` 构建日志为准；这里讲的是**机制不变量**。

---

## 4. 精度校准（FP16 / INT8 / FP8）

降精度 = 用更少 bit 表示数 → 更小的访存 + 更快的 Tensor Core 吞吐。但要在「快」和「准」之间权衡。

| 精度 | 位宽 | 相对 FP32 吞吐(直觉) | 是否需校准 | 典型用途 |
|---|---|---|---|---|
| FP32 | 32 | 1× | 否 | 精度基线 |
| TF32 | 19(有效) | ~快 | 否 | Ampere+ 默认折中 |
| FP16 | 16 | ~2×+ | 否 | 最常用、几乎无损 |
| BF16 | 16 | ~2×+ | 否 | 动态范围大、训练同款 |
| INT8 | 8 | ~4× | **是**(PTQ) | 极致延迟、对精度敏感 |
| FP8 | 8 | ~4× | 视模型 | 新架构(Hopper+) LLM |

### 4.1 INT8 为什么需要「校准」

FP 是浮点（动态范围大）；INT8 只有 $[-128,127]$ 共 256 个台阶。要把一段连续的浮点激活值塞进 256 个整数台阶，必须定一个**缩放因子(scale)**：

$$ q = \mathrm{round}\!\left(\frac{x}{s}\right),\qquad s = \frac{|x|_{\max}}{127} $$

难点在于 **$|x|_{\max}$ 怎么定**。如果直接取真实最大值，少数离群点(outlier)会把 scale 撑得很大，导致大多数正常值都被压到 0 附近、量化噪声爆炸。

**校准(calibration)** 就是：拿一小批有代表性的真实数据（calibration dataset，几百~几千张即可）跑前向，**统计每层激活的分布直方图**，再用某种准则挑一个最优的截断阈值 $T$（不一定是真实 max）。

经典准则是 **KL 散度（熵校准, Entropy Calibration）**：在不同候选阈值 $T$ 下，把 FP32 分布 $P$ 截断+量化得到分布 $Q$，选使量化前后分布信息损失最小的 $T$：

$$ T^\* = \arg\min_{T}\; D_{\mathrm{KL}}\big(P \,\|\, Q_T\big) $$

```
       激活值直方图 (FP32)
  频次
   │      ███
   │    ███████
   │  ████████████        ← 选 T 截断这里，离群尾巴丢掉
   │ ██████████████░░░░  ← 尾部少量大值
   └────────────┬────────────► 幅值
                T (校准选出的阈值)
   [-T, T] 内均匀映射到 [-127,127]，|x|>T 截到 ±127
```

> 这叫 **PTQ（Post-Training Quantization，训练后量化）**：不改权重，只配 scale。若 PTQ 掉点太多，要用 **QAT（量化感知训练）** 或更细粒度的量化方案——这部分由 [[ai-framework/TensorRT-Model-Optimizer]] 负责（它是 TensorRT 生态里专门做量化/稀疏/蒸馏，再把优化结果喂给 TensorRT 编译的工具）。

### 4.2 权重 vs 激活、per-tensor vs per-channel

- 权重通常 **per-channel**（每个输出通道一个 scale），分布更紧、掉点小。
- 激活通常 **per-tensor**（整张一个 scale），便宜但更易受 outlier 影响。
- 这也是 LLM 量化难的根源：激活里有强 outlier，需要 SmoothQuant / AWQ / per-token 之类的招，见 [[ai-framework/TensorRT-Model-Optimizer]]。
## 5. Kernel 自动调优（Autotuning / Tactic 选择）

同一个算子（如某 shape 的矩阵乘），在一块 GPU 上可能有**几十种实现**（不同 tile 大小、是否用 Tensor Core、不同的内存布局、来自 cuBLAS/cuDNN/TensorRT 内置库等）。哪个最快**取决于具体 GPU + 具体输入 shape**，无法纯靠公式算。

TensorRT 的做法是**实测打擂台**：构建期对每个层枚举候选实现（称为 **tactic**），**实际在目标 GPU 上各跑几次计时**，选最快的那个固化进 engine。

```
   对某一层:
   候选 tactic 池
   ┌──────────────────────────┐
   │ tactic#1 cuDNN  →  2.1 ms │
   │ tactic#2 cuBLAS →  1.8 ms │
   │ tactic#3 TRT内置 → 1.3 ms │ ◄── 实测最快，选它
   │ tactic#4 ...    →  2.0 ms │
   └──────────────────────────┘
            │ 各自真机计时
            ▼
       写入 engine 的「执行计划」
```

推论：**build 慢**正是因为把这些 tactic 逐个跑一遍计时——拿构建时间换运行时间；engine 因此**绑定 GPU 型号**（换卡后最优 tactic 变了，旧 engine 即使能加载性能也不再最优→须重 build）；可用 **timing cache** 缓存计时结果加速后续构建。

---

## 6. Build Engine 全流程（调用链）

从一个 ONNX 模型到一个能上线的 `.engine`，标准链路：

```
 PyTorch/TF 模型
      │ torch.onnx.export / tf2onnx
      ▼
   model.onnx ────────────────────────────────┐
      │                                         │ (INT8 时)
      │                                  校准数据集 calib/*
      │                                         │
      ▼                                         ▼
 ┌───────────────────────────────────────────────────────┐
 │ TensorRT Builder                                       │
 │  1. Parser 解析 ONNX → INetworkDefinition              │
 │  2. 读 BuilderConfig: 精度(FP16/INT8)、workspace 上限   │
 │  3. 图优化 + 层融合 (第3节)                              │
 │  4. 精度/校准 (第4节)                                   │
 │  5. kernel autotuning，实测选 tactic (第5节)            │
 │  6. 序列化                                              │
 └───────────────────────────────────────────────────────┘
      │
      ▼
  model.engine (.plan)  ← 存盘，上线复用
      │ 反序列化
      ▼
 Runtime: deserialize → createExecutionContext
      │  bind 输入/输出显存
      ▼
  context.enqueueV3(stream)  → 异步推理 → 取输出
```

两条等价路径：
- **命令行**：`trtexec`（官方自带工具，最快验证）。
- **Python/C++ API**：`tensorrt` 包，灵活、可嵌入服务。

PyTorch 用户还有 **Torch-TensorRT**（`pytorch/TensorRT`）：在 PyTorch 图里把**能优化的子图**交给 TensorRT，**不能的回退**给 PyTorch 执行，省去手动导 ONNX。

---

## 7. 与 TensorRT-LLM 的关系

很关键，别混了：

| 维度 | TensorRT（本篇） | TensorRT-LLM |
|---|---|---|
| 定位 | 通用推理编译器(CV/NLP/任意 ONNX) | **专为大语言模型**的库，**构建在 TensorRT 之上** |
| 处理对象 | 静态图、固定/可变 shape | 自回归生成、KV Cache、变长序列 |
| 额外能力 | — | in-flight/continuous batching、paged KV cache、张量/流水并行、专用 attention kernel(如 paged attn)、量化(INT8/FP8/AWQ) |
| 产物 | engine | 同样是 TensorRT engine，但内含 LLM 专用插件/plugin |

```
        ┌──────────────────────────────┐
        │        TensorRT-LLM           │  ← LLM 专属:KV cache/批处理/并行
        │  (Python 组网 + 专用 plugin)   │
        └───────────────┬───────────────┘
                        │ 复用底层
        ┌───────────────▼───────────────┐
        │          TensorRT              │  ← 图优化/融合/精度/autotune
        └───────────────┬───────────────┘
                        ▼
                      CUDA / Tensor Core
```

一句话：**LLM 不是「一张静态图」**——它要循环解码 N 步、每步读写 KV Cache、不同请求长度不同。这些是通用 TensorRT 图模型表达不了的，所以 NVIDIA 单独做了 TensorRT-LLM，但**底层的融合/精度/调优仍然靠 TensorRT**。详见 [[llm-inference/README]] 同目录的 `tensorrt-llm/`。

---

## 8. 为什么快（一张总账）

把加速来源归因，便于回答面试题「TensorRT 凭什么快」：

```
┌───────────────────────┬─────────────────────────────────────┐
│ 加速来源               │ 省掉了什么                            │
├───────────────────────┼─────────────────────────────────────┤
│ ① 层融合 (纵/横)       │ 中间结果的 DRAM 往返 + kernel launch  │
│ ② 低精度 FP16/INT8/FP8 │ 一半~1/4 的访存 + Tensor Core 高吞吐  │
│ ③ kernel autotuning   │ 通用 kernel 的低效，挑到当前卡最优实现 │
│ ④ 常量折叠/去无用层    │ 编译期能算的就不放到运行期             │
│ ⑤ 显存复用/规划       │ 中间张量共用 buffer，少分配少碎片      │
└───────────────────────┴─────────────────────────────────────┘
        核心思想:把「每次推理都要决策的事」前置到「编译一次」。
```

一句话本质：**用一次昂贵的离线编译，换无数次廉价的在线推理。**

---

## 9. 配置 / 流程示例（讲含义）

> 下面是说明性示例，**精确的 CLI 默认值/参数名以官方文档与 `--help` 为准**，这里重在讲清每个旋钮在权衡什么。

### 9.1 trtexec（最快上手）

```bash
trtexec \
  --onnx=model.onnx \        # 输入网络
  --saveEngine=model.engine \# 序列化产物存盘(关键:只 build 一次)
  --fp16 \                   # 开 FP16 (几乎无损,首选)
  --int8 \                   # 开 INT8 (需配校准,极致延迟)
  --memPoolSize=workspace:4096 \ # 给 autotuning 的临时显存上限
  --minShapes=... --optShapes=... --maxShapes=... # 动态 shape 三档
```

关键参数的权衡：

| 参数 | 含义 | 调大/开启的代价与收益 |
|---|---|---|
| `--fp16` | 启用半精度 | 收益:~2× 吞吐；代价:极少数模型精度抖动 |
| `--int8` | 启用 INT8 | 收益:~4×；代价:需校准数据，可能掉点 |
| workspace/memPool | autotuning 可用临时显存 | 大→可选更快但更费显存的 tactic；小→可能选不到最优 |
| `opt/min/maxShapes` | 动态 shape 的优化点 | optShapes 设成线上最常见的 batch/seq，命中最优 |
| timing cache | 复用 tactic 计时 | 加快二次构建，跨相同环境复用 |

### 9.2 Python API 骨架（机制示意）

```python
import tensorrt as trt
logger  = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(...)        # 逻辑图
parser  = trt.OnnxParser(network, logger)
parser.parse(open("model.onnx","rb").read()) # ONNX → network

config  = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)        # 精度开关
# config.int8_calibrator = MyCalibrator(...)  # INT8 时挂校准器

engine_bytes = builder.build_serialized_network(network, config)  # ←最贵的一步
open("model.engine","wb").write(engine_bytes) # 存盘

# ---- 上线侧 ----
runtime = trt.Runtime(logger)
engine  = runtime.deserialize_cuda_engine(open("model.engine","rb").read())
context = engine.create_execution_context()
# 绑定输入/输出显存地址，context.execute_async_v3(stream) 触发推理
```

> 上面的方法名/枚举在不同大版本间有改动（如 `enqueueV2/V3`、`build_engine` vs `build_serialized_network`），**以你安装版本的官方 API 为准**，思路（parse→config→build→serialize→deserialize→execute）是稳定的。

---

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 换了张 GPU 后 engine 加载失败/变慢 | engine 绑定 GPU 架构 + TRT 版本 | 在目标卡上**重新 build** |
| INT8 精度掉很多 | 校准集不代表真实分布 / 激活有 outlier | 换更有代表性的校准集；上 QAT 或 SmoothQuant/AWQ（→ Model-Optimizer） |
| 构建特别慢 | autotuning 实测大量 tactic | 用 timing cache；减小 workspace 搜索面；CI 里缓存 engine |
| 动态 shape 性能不稳 | optShapes 没设在常用点 | 把 optShapes 设成线上高频 batch/seq |
| 想跑 LLM 却很别扭 | 通用 TensorRT 不含 KV cache/批处理 | 改用 **TensorRT-LLM**（见第 7 节） |
| 显存 OOM | workspace + 中间张量过大 | 调小 memPool；查看是否有未复用的大中间张量 |
| PyTorch 模型导不出 ONNX | 含不支持算子/控制流 | 改写算子；或用 Torch-TensorRT 让不支持子图回退 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，回到顶层
- [[ai-framework/TensorRT-Model-Optimizer]] — 量化/稀疏/蒸馏工具，产出喂给 TensorRT 编译
- [[llm-inference/README]] — 推理优化总览（与同目录 `tensorrt-llm/`、`vllm`、`triton` 对照）

### 官方资料

- TensorRT: https://github.com/NVIDIA/TensorRT
- Torch-TensorRT: https://github.com/pytorch/TensorRT
- 快速开始（pip 安装）: https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#installing-pip
- tar 包安装: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar
- NGC PyTorch 镜像: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
