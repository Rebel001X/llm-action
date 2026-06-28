# 大模型性能分析工具(Profiling Tools)

> 一句话定位:把"模型为什么慢/为什么爆显存"从玄学变成可测量、可定位、可优化的工程问题——这就是 Profiling 工具链要解决的事。📍 导航:[[00-知识地图]]
> 🔗 相关:[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]] · [[llm-optimizer/FlashAttention]] · [[llm-inference/vllm/README]]

## 阅读地图

| 你想解决的问题 | 看哪一节 | 用什么工具 |
| --- | --- | --- |
| 不跑代码,先估算训练/推理的显存与时延 | §1 §3 | LLM-Viewer / llm-analysis / llm_profiler(静态分析) |
| 想知道时间花在哪个算子(Python 视角) | §4 | PyTorch Profiler |
| 想看整机 CPU+GPU+IO+通信的时间线(系统视角) | §5 | Nsight Systems(nsys) |
| 想深挖某个 CUDA kernel 为什么慢(微观视角) | §6 | Nsight Compute(ncu) |
| 想给自己的代码段打标签便于在时间线里认出来 | §7 | NVTX |
| 想压测推理服务的吞吐/时延并顺带 profile | §8 | vLLM / SGLang benchmark |

## 0. 一句话锚点

> **静态分析告诉你"理论上应该多少",动态 Profiling 告诉你"实际上是多少",二者的差就是优化空间。**

性能分析永远围绕一个根本矛盾:GPU 算力(FLOPS)和显存带宽(GB/s)是两种独立资源,一个算子要么卡在算力(compute-bound),要么卡在带宽(memory-bound)。所有 profiling 工具,本质都是在帮你回答**"现在卡在哪一边"**这一个问题。

## 1. 地基:为什么需要 Profiling

### 1.1 大模型推理/训练的三种瓶颈

```
            ┌─────────────────────────────────────────────┐
            │              一次前向/反向的耗时              │
            └─────────────────────────────────────────────┘
                 │              │                │
        ┌────────┘       ┌──────┘         ┌──────┘
        ▼                ▼                ▼
  ┌───────────┐   ┌────────────┐   ┌──────────────┐
  │ 算力瓶颈   │   │ 带宽瓶颈    │   │ 通信/同步瓶颈 │
  │ compute   │   │ memory     │   │ comm/launch  │
  │ -bound    │   │ -bound     │   │ -bound       │
  ├───────────┤   ├────────────┤   ├──────────────┤
  │ 大矩阵乘   │   │ KV-Cache 读 │   │ AllReduce    │
  │ Prefill   │   │ Decode 逐 token│ │ kernel launch│
  │ FFN GEMM  │   │ LayerNorm  │   │ Host↔Device  │
  └───────────┘   └────────────┘   └──────────────┘
```

判定一个算子属于哪种瓶颈,看 **算术强度(Arithmetic Intensity)**:

$$I = \frac{\text{FLOPs}}{\text{Bytes accessed}} \quad(\text{单位: FLOP/Byte})$$

把它和硬件的"屋脊点"比较——这就是 **Roofline 模型**:

$$I^* = \frac{\text{峰值算力 (FLOP/s)}}{\text{峰值带宽 (Byte/s)}}$$

- $I < I^*$ → 带宽受限(增加算力没用,要减少访存)
- $I > I^*$ → 算力受限(要么提算力,要么降精度做更多有效计算)

**数值示例(A100,以 FP16 估算)**:峰值算力约 $312\ \text{TFLOP/s}$,显存带宽约 $2.0\ \text{TB/s}$,则
$$I^* = \frac{312\times10^{12}}{2.0\times10^{12}} \approx 156\ \text{FLOP/Byte}$$
意味着任何算术强度低于 ~156 的算子(典型如 Decode 阶段逐 token 的注意力、LayerNorm、逐元素激活)在 A100 上都是**带宽受限**的——这就是为什么 KV-Cache 优化、FlashAttention(减少 HBM 读写)如此关键。

### 1.2 静态分析 vs 动态 Profiling

| 维度 | 静态分析(LLM-Viewer/llm-analysis/llm_profiler) | 动态 Profiling(PyTorch/Nsight) |
| --- | --- | --- |
| 是否跑模型 | 否,纯公式推算 | 是,真实执行采样 |
| 输出 | 理论 FLOPs/显存/时延上界 | 真实耗时/占用/调用栈 |
| 优点 | 秒级、改超参即可对比、不占卡 | 反映真实 overhead(launch、碎片、同步) |
| 缺点 | 忽略实现/调度/碎片开销 | 需要真机、采样有扰动 |
| 适用阶段 | 选型、容量规划、可行性评估 | 落地后的瓶颈定位与调优 |

## 2. 静态分析三件套(原文工具,逐个解释)

原文给出三个静态分析项目,它们都不真正跑大模型,而是按 Transformer 结构用解析公式算出资源需求:

- **[LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer.git)**:可视化大语言模型(LLMs)并分析不同硬件平台上的性能。它把每一层映射到 Roofline 上,直观看出哪些层是 compute-bound、哪些是 memory-bound。
- **[llm-analysis](https://github.com/cli99/llm-analysis)**:对 Transformer 模型的训练和推理进行延迟和内存分析。给定模型尺寸、并行策略、序列长度,输出各部分显存(参数/梯度/优化器状态/激活)与时延拆分。
- **[llm_profiler](https://github.com/harleyszhang/llm_counts)**:大模型理论性能分析工具(仓库名 `llm_counts`),按公式统计 FLOPs、参数量、各阶段时延。

### 2.1 它们到底在算什么(原理)

以推理显存为例,核心四块:

```
  显存总占用 ≈ 模型权重 + KV-Cache + 激活 + 框架/碎片开销
              │           │          │
              │           │          └─ 与 batch、隐藏维成正比,前向后释放
              │           └─ 与 batch × seqlen × layers × 2(K,V) 成正比 ← 推理增长主因
              └─ 参数量 × 每参数字节数(FP16=2, INT8=1, INT4=0.5)
```

**手算示例(KV-Cache,与 [[docs/transformer内存估算]] 同源)**:一个 32 层、隐藏维 $d=4096$、FP16 的模型,单条序列长度 $L=2048$:

$$\text{KV} = 2 \times L \times \text{layers} \times d \times \text{bytes} = 2\times2048\times32\times4096\times2 \approx 1.07\ \text{GB}$$

batch=32 时 KV-Cache 就要 ~34 GB——这正是静态工具一眼能算出、而靠"感觉"很难估准的量,也是为什么要看 [[llm-optimizer/kv-cache]] 和 [[llm-inference/PD分离]]。

## 3. 推理框架自带的 Profiling 入口

原文引用了两个框架的官方 profiling 文档:

- **[vLLM 性能分析](https://vllm.hyper.ai/docs/contributing/profiling_index/)**(英文原文 https://docs.vllm.ai/en/stable/contributing/profiling/profiling_index.html)。vLLM 内部封装了两条路:**PyTorch Profiler**(算子级 trace)与 **NVIDIA Nsight Systems**(系统级 trace)。
- **[SGLang Benchmark and Profiling](https://docs.sglang.ai/references/benchmark_and_profiling.html)**。

### 3.1 两个关键启动参数(原文真料,解释为什么)

原文单独列出了两个 flag,它们是做 profiling/压测时极其常用的"省事开关":

| 参数 | 作用 | 为什么 profiling 时用它 |
| --- | --- | --- |
| `--json-model-override-args` | 用 JSON 覆盖模型配置参数(如层数、隐藏维) | 想构造一个"缩小版"或"自定义尺寸"模型来快速复现/隔离瓶颈,不必改原始 config |
| `--load-format dummy` | 用随机/假权重加载,**不读真实 checkpoint** | profiling 只关心计算图和算子耗时,与权重数值无关。跳过磁盘 I/O 和大文件下载,几秒就能起服务做压测 |

> 坑点:`--load-format dummy` 跑出来的模型**输出是乱的**,只能用于性能测量,不能用于正确性验证。

## 4. PyTorch Profiler(算子级,Python 视角)

> 适合回答:"我的 `model(x)` 里,哪个 op(matmul / softmax / layernorm)吃了最多时间和显存?"

官方教程(原文链接):
- 入门 https://pytorch.org/tutorials/beginner/profiler.html
- Recipe https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- TensorBoard 可视化 https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

进阶教程(原文链接):
1. Numeric Suite https://pytorch.org/tutorials/prototype/numeric_suite_tutorial.html
2. FX Profiling https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html

### 4.1 工作原理

```
  Python 代码  ── torch.profiler 钩子 ──► 记录每个 op 的
   model(x)         (CPU 端 + CUDA 端)        ├─ CPU 自身耗时
                                              ├─ CUDA kernel 耗时
                                              └─ 显存分配/释放
                          │
                          ▼
                  导出 trace.json ──► chrome://tracing 或 TensorBoard 看时间线
```

PyTorch Profiler 的视角是"以 PyTorch op 为单位"聚合,它能把 GPU kernel **归因回**你写的 Python 行,这是 nsys 不直接具备的优势(nsys 看到的是裸 kernel 名)。

## 5. NVIDIA Nsight Systems(nsys,系统级)

> 属于**系统级**性能分析工具,提供从全局视角对整个系统的性能进行监控和分析,包括 CPU、GPU、内存、IO 等多种硬件资源的使用情况,以及它们之间的交互信息。

参考资料(原文):
- 阿里云 ACK 用 Nsight System 做性能分析:https://help.aliyun.com/zh/ack/cloud-native-ai-suite/use-cases/using-nsight-system-to-realize-performance-analysis
- 知乎讲解:https://zhuanlan.zhihu.com/p/718956195

### 5.1 它擅长看什么

```
  时间轴 ──────────────────────────────────────────────►
  CPU:   [tokenize][   launch kernels  ][   wait sync   ]
  GPU:        ░░░░░[█GEMM█][█attn█]░░gap░░[█GEMM█]
  NCCL:                    [===AllReduce===]
                ▲          ▲                ▲
                │          │                └ 通信与计算是否重叠?
                │          └ kernel 之间有没有空泡(gap)?
                └ CPU 是否成了 GPU 的"喂数瓶颈"(launch-bound)?
```

系统级 profiler 的核心价值是发现**算子之外**的问题:kernel launch 开销、CPU 喂不上数、通信没和计算重叠、Host↔Device 拷贝阻塞。这些靠 PyTorch Profiler 不容易看清。

### 5.2 关键技巧与命令(原文真料)

- **看 CUDA kernel 的 Python 调用栈**:使用 `--python-backtrace=cuda` 查看所有 CUDA 内核的 Python 调用堆栈,就像在 PyTorch Profiler 中一样。
  > 注意(原文坑):这可能会导致基于 CUDA 事件计时的内核运行时间不准确——因为采样调用栈本身有开销。
- **快速看 trace**:生成的时间线可在浏览器 `chrome://tracing` 中打开查看。
- **结合 SGLang 压测做 profiling**(原文命令):

```bash
python -m sglang.bench_serving \
    --backend sglang \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-prompts 2 \
    --sharegpt-output-len 100 \
    --profile
```

该命令含义逐项:`--backend sglang` 指定后端;`--model` 指定模型;`--num-prompts 2` 只发 2 条请求(profiling 不需要大流量,够采到样即可);`--sharegpt-output-len 100` 控制每条输出 100 token;`--profile` 打开 profiling 采集。

## 6. NVIDIA Nsight Compute(ncu,内核级)

> 专注于 **GPU 内核级**的性能分析,主要针对 CUDA 应用程序,深入到 GPU 内部,分析 CUDA 内核的执行情况。

ncu 是放大镜中的放大镜:nsys 告诉你"哪个 kernel 慢",ncu 告诉你"这个 kernel 为什么慢"——它会给出该 kernel 的 SM 占用率(occupancy)、寄存器/共享内存用量、内存吞吐、是否 bank conflict、Roofline 上的具体位置。

```
  nsys(找慢的 kernel)  ──►  ncu(剖析这个 kernel 内部)
   "整桌菜哪盘凉了"            "这盘菜为啥凉:火候?锅?摆盘?"
```

> 实践节奏:**先 nsys 定位,再 ncu 深挖**。ncu 采集开销极大(常需重放 kernel 数十次),不要一上来就对整个程序跑 ncu。

## 7. NVTX(NVIDIA Tools Extension Library)

通过使用 NVTX,开发者可以在代码中添加注释,这些注释可以被 NVIDIA 的开发工具识别,从而在性能分析和调试过程中提供帮助。

**为什么需要它**:nsys/ncu 的时间线里全是底层 kernel 名(`ampere_sgemm_...`),你根本认不出哪段对应你的"注意力层"。NVTX 让你给代码段贴**人类可读的标签**,这些标签会作为彩色区间出现在 nsys 时间线上。

```
  没有 NVTX:  [sgemm][elementwise][sgemm][reduce]...   ← 看不懂
  加了 NVTX:  [══ Attention ══][══ FFN ══][══ Norm ══]  ← 一眼对应代码
```

安装(原文):

```bash
pip install nvtx
```

代码示例(原文,`@nvtx.annotate()` 装饰函数、`with nvtx.annotate(...)` 标注代码块):

```python
# example_lib.py

import time
import nvtx


def sleep_for(i):
    time.sleep(i)

@nvtx.annotate()          # 自动用函数名作为区间标签
def my_func():
    time.sleep(1)

with nvtx.annotate("for_loop", color="green"):   # 给代码块手动命名+上色
    for i in range(5):
        sleep_for(i)
        my_func()
```

配合 nsys 采集(原文命令):

```bash
nsys profile python demo.py
```

运行后,nsys 时间线里就会出现绿色的 `for_loop` 区间和 `my_func` 区间,把底层 kernel 归到你的逻辑段上。

## 8. 实操汇总:三个工具的协作流程

```
┌────────────────────────────────────────────────────────────┐
│  Step 1  静态估算(选型/容量)                                │
│    LLM-Viewer / llm-analysis / llm_profiler                 │
│      ↓ 知道理论显存、理论时延、瓶颈侧(算力还是带宽)         │
│  Step 2  系统级定位(找空泡/通信/launch)                     │
│    nsys profile + NVTX 标注 + --python-backtrace=cuda       │
│      ↓ 在 chrome://tracing 看时间线,锁定最慢的 kernel/阶段  │
│  Step 3  算子级 / 内核级深挖                                 │
│    PyTorch Profiler(归因到 Python 行) / ncu(剖析 kernel)  │
│      ↓ 改 kernel、改并行、改精度,回到 Step 1 重新估算闭环   │
└────────────────────────────────────────────────────────────┘
  压测入口: vLLM/SGLang benchmark + --load-format dummy + --profile
```

## 常见问题/坑

| 现象 / 误区 | 原因 | 正确做法 |
| --- | --- | --- |
| 静态工具算出的时延和实测差很多 | 静态忽略 kernel launch、显存碎片、同步、调度开销 | 静态只用于估上界与选型,落地用动态 profiling 校正 |
| `--load-format dummy` 跑出来结果是错的 | 用的是随机权重,只为测性能 | 仅用于性能/压测,正确性验证必须加载真实 checkpoint |
| 加了 `--python-backtrace=cuda` 后 kernel 计时变长 | 采集 Python 调用栈本身有开销,扰动 CUDA event 计时 | 定位调用来源时开,纯计时时关掉 |
| nsys 时间线全是 kernel 名,认不出自己的代码 | 缺少逻辑标签 | 用 NVTX `annotate` 给关键代码段打标签再 profile |
| 一上来就对整个程序跑 ncu,几小时跑不完 | ncu 需重放 kernel、采集开销巨大 | 先 nsys 缩小到少数 kernel,再用 ncu 精剖 |
| Decode 阶段提算力没用、加 batch 才有用 | Decode 是带宽受限(算术强度低于屋脊点) | 优化访存:KV-Cache 量化/PagedAttention,而非堆 FLOPS |
| profiling 时发的请求很多导致 trace 巨大难看 | 采样不需要大流量 | 像原文 `--num-prompts 2` 那样只发少量请求即可采到样 |

## 🔗 跳转链接

- 枢纽:[[00-知识地图]]
- 性能指标定义:[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
- 被 profiling 的对象(模型结构):[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 优化点(带宽受限的根因):[[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 推理框架(profiling 入口):[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 压缩降访存(优化带宽瓶颈):[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 训练/微调侧 profiling:[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 对齐训练:[[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 硬件与通信(系统级瓶颈根因):[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
