# 编译器路线：TVM 与 MLIR

> **一句话**：代码是人写的还是编译出来的，是和"用什么语言写"完全正交的第二条轴。
> **前置**：先读 `01-用什么语言写推理引擎-12个真实引擎的证据.md` 建立语言构成的事实基础；`02-自回归解码为什么是访存瓶颈.md`（`01-地基/`）证明的"decode 访存受限"，本篇第 2、5 节会直接借用它的结论——算子融合省的是谁，答案就在那一篇里。

## 0. 这一篇要解决什么问题

前三篇（`02-Python为什么还没被淘汰.md`、`03-C++路线-llama.cpp的取舍.md`、`04-Rust在推理栈里到底占了什么位置.md`）问的都是同一类问题：**这段代码用什么语言写**。本篇问的是一个不同维度的问题：**这段代码到底是不是"写"出来的**——还是人只写了一份更抽象的描述，剩下的活交给编译器生成。

这条轴之所以要单独拎出来，是因为它带着一个真实存在、而且相当有诱惑力的立论：**一个推理引擎要覆盖 N 个模型 × M 种硬件 × K 种精度，如果每种组合都手写一遍算子，工作量是一个乘法。**

具体到数字上：这份研究库 clone 里，vLLM 一家仓库里 `vllm/model_executor/models/` 下就有 292 个模型定义文件（自数，见 §3），MLC-LLM 的 `model/` 目录下有 44 个模型架构子目录（自数），每个模型至少要过一遍 attention、FFN、norm 几类算子；硬件侧 MLC-LLM 自己的后端矩阵列出 CUDA、ROCm、Vulkan、Metal、WebGPU、OpenCL 六条目标（`mlc-llm:README.md:22-58`）；精度侧还有 fp16/bf16/int8/int4/fp8 等好几档。手写全部组合，理论上限就是 N×M×K 条算子实现——这是真实存在的组合爆炸,不是编出来吓唬人的。

**编译器路线的赌注**：把"手写 N×M×K 个算子"换成"写 N 份模型描述 + 实现 M 个后端代码生成器"，让 K 种精度和大部分算子融合由编译器自动搜索/生成——**把乘法换成加法**。这个赌注在多大程度上兑现，是本篇要回答的核心问题。

但"编译器路线"这个说法本身是个陷阱——它把三种完全不同的技术方案压扁成了一个词。本篇要先把它们拆开，再逐个看它们各自兑现了赌注的哪一部分、又在哪里留下了新的账单：

1. **算子级 DSL**（以 Triton 为代表）：本质上还是人写 kernel，只是写在一个更高的抽象层（tile 而不是线程），寄存器分配、访存合并这类脏活交给编译器。
2. **图级编译**（以 TVM/Relax、torch.compile 为代表）：把一整张计算图接管过去，做融合、排布优化、代码生成。
3. **中间表示基础设施**（以 MLIR 为代表）：不是一个能直接拿来编译模型的编译器，是一套**用来造编译器的工具箱**——各家硬件/框架团队在它上面搭自己的方言（dialect）。

这三层解决的是完全不同的问题，各自的收益和代价互不相通。混着谈的后果在 2.3 节会具体展开，但先说结论：**"要不要走编译器路线"这个问题本身问错了层——正确的问题是"这三层里,你的引擎现在需要哪一层,需要到什么程度"。**

## 1. 先看现象（可跑的代码/可算的数）

在讲原理之前，先把已经能拿到的真实数字摆出来——这些数字会在第 2 节直接被用来支撑"三层"的划分。

| 项 | 数字 | 出处 |
|---|---:|---|
| LightLLM 总行数 | 148,574 | `_PLAN.md` §3.2 |
| LightLLM `.cu`/`.cuh` 文件数 | **0** | 自数，见 §3 |
| LightLLM 含 `@triton.jit` 的文件数 | 129 | 自数，见 §3 |
| MLC-LLM 总行数 | 100,240 | `_PLAN.md` §3.2 |
| MLC-LLM C++ 占比（TVM 运行时） | 16% | `_PLAN.md` §3.2 |
| MLC-LLM `model/` 下模型架构子目录数 | 44 | 自数，见 §3 |
| vLLM `model_executor/models/` 下模型文件数 | 292 | 自数，见 §3 |
| vLLM 含 `@triton.jit` 的 `.py` 文件数 | 202 | 自数，见 §3 |
| vLLM `@triton.jit` 出现总次数 | 499 | 自数，见 §3 |
| vLLM 含 `torch.compile(` 的 `.py` 文件数 | 61 | 自数，见 §3 |
| vLLM `torch.compile(` 出现总次数 | 104 | 自数，见 §3 |
| vLLM `.cu`/`.cuh` 文件数（对照） | 169 | 自数，见 §3 |
| SGLang 含 `@triton.jit` 的 `.py` 文件数 | 249 | 自数，见 §3 |
| SGLang `@triton.jit` 出现总次数 | 631 | 自数，见 §3 |
| SGLang 含 `torch.compile(` 的 `.py` 文件数 | 64 | 自数，见 §3 |
| SGLang `torch.compile(` 出现总次数 | 86 | 自数，见 §3 |
| SGLang `.cu`/`.cuh` 文件数（对照） | 246 | 自数，见 §3 |

这张表已经把本篇的立论提前摆出来了。三条读法：

**第一，LightLLM 是"层 1 走到底"的真实案例。** 148,574 行代码，0 个 CUDA 文件，129 个文件写着 Triton kernel——一个能跑、能服务真实请求的推理引擎，热点算子全部用算子级 DSL 写，一行 CUDA C++ 都没有。这不是理论上"应该可行"，是已经在生产级开源项目里跑通的事实。

**第二，MLC-LLM 是"层 2 放在架构中心"的真实案例。** 100,240 行里 16% 是 C++——不是引擎作者手写的业务逻辑，是 TVM 编译产物依赖的运行时（第 4 节会具体看这 16% 装的是什么）。

**第三，也是最容易被忽略的一条：vLLM 和 SGLang——这两个通常被归类为"Python 手写调 PyTorch"的引擎——本身就大量使用层 1 和层 2。** vLLM 有 202 个文件、499 处 `@triton.jit`；SGLang 有 249 个文件、631 处。vLLM 有 61 个文件、104 处调用 `torch.compile(`；SGLang 有 64 个文件、86 处。换句话说：**"手写派"和"编译派"从来不是两个互斥的阵营，vLLM/SGLang 内部本来就并存着大量层 1 和层 2 的代码**，只是没有像 MLC-LLM 那样把编译放在架构的中心位置。这条现象会在第 4、6 节反复回来。

还有一组数字值得在这里先埋个伏笔：vLLM 一家就支持 292 个模型文件，MLC-LLM 只有 44 个模型架构子目录——量级差了 6 倍多。这不是说 MLC-LLM 团队投入的精力更少，§4.2 会给出机制上的原因：vLLM 的模型定义是普通的 PyTorch eager `nn.Module`，跟上游 HuggingFace 生态几乎是同一套写法，接入成本低；MLC-LLM 的模型定义要用它自己的图追踪前端重新写一遍，还要经过编译流水线才能验证对不对。**"多支持一种硬件"和"多支持一个模型"，在编译器路线里要付的是两种不同的代价，第一种代价被架构选择一次性买断了，第二种代价每次都要重新付**——这条区分在 §4.2 会摊开算清楚。

## 2. 原理

### 2.1 出发点：把乘法变成加法，但这是一个有条件的承诺

重新把 §0 的账算清楚。假设手写路线下，每个模型 × 每种硬件 × 每种精度都要一份独立调优过的 kernel 实现，总工作量正比于 N×M×K。编译器路线想做的事是：

- 每个模型只写**一份**图/算子描述（不区分硬件、不区分精度）——工作量变成 N；
- 每种硬件只写**一份**后端代码生成器（接受任意合规的图描述，吐出这种硬件上能跑的代码）——工作量变成 M；
- 精度差异（K 那一维）尽量交给自动量化/自动调优处理，不需要为每个精度单独写代码。

如果这套分解真的成立，总工作量从 N×M×K 降到 N+M，量级上的差别是巨大的。**这就是编译器路线的全部吸引力所在，也是它唯一值得认真对待的理由**——不是"编译器听起来更先进"，而是这个乘法确实存在,确实值得想办法拆开。

但这个分解能不能兑现，取决于一个前提：**后端代码生成器必须真的对"任意"模型描述都好用**——新模型引入的新算子、新的张量形状模式、新的控制流，编译器的通用代码生成路径都要能兜住,不然还是要为这个模型单独写一份"后端专属优化"，N+M 就退化回了介于 N+M 和 N×M×K 之间的某个中间态。第 4、5、6 节会具体看这个前提在多大程度上成立。

这里先如实纠正一个容易被 §2.1 开头那句"K 种精度交给自动量化/自动调优处理"带偏的预期：**精度这一维并不是编译器凭空生成的，而是从一份预先手写好的量化策略库里选一个。**

MLC-LLM 自己的量化注册表就是证据——`q0f16`、`q4f16_0`、`q4f16_1`、`q4f16_autoawq`、`q4f16_ft` 这些精度变体,分别对应 `NoQuantize`、`GroupQuantize`、`AWQQuantize`、`FTQuantize` 几种不同的量化算法实现,在一个字典里逐条注册（`mlc-llm:python/mlc_llm/quantization/quantization.py:31-33`，`q4f16_0` 那一条在 `mlc-llm:python/mlc_llm/quantization/quantization.py:69-70`）。也就是说 K 这一维同样是"人先把每种量化算法写好,编译器负责的是把已经选定的那一种应用到具体的模型图上"——跟层 1、层 2 是同一种性质的分工：**编译器省的是"把已经想清楚的方案套到具体对象上"这一步机械劳动，不是"想清楚方案本身"这一步。**

N×M×K 里唯一被真正压缩掉的是"每个模型 × 每种硬件 × 每种精度都要重新实现一遍算法"这个组合爆炸,算法本身（无论是某个融合规则、某种量化策略,还是某个 Triton kernel 的分块方式）依然要由人想清楚、写一次,然后被编译器复用到更多组合上——这是本篇要传达的最核心的一层澄清,也是理解后面所有"代价"小节的前提。

### 2.2 三层，必须拆开

#### 层 1：算子级 DSL（Triton）——人还在写 kernel，只是换了一层抽象

用 Triton 写一个 kernel，你仍然要决定算法：分几块、每块多大、循环怎么走、要不要在线 softmax。Triton 编译器接管的是**从"tile 级描述"到"GPU 线程/寄存器/内存事务"这一层翻译**——寄存器分配、访存合并、部分调度决策，这些是 Triton 编译器做的，不是你写的。

LightLLM 的证据最直接：全仓 0 个 `.cu`/`.cuh` 文件，129 个文件里写着真实的 Triton kernel。举一个具体例子，`context_flashattention_nopad.py` 里的 prefill attention kernel：

```python
@triton.jit
def _fwd_kernel(
```
（`lightllm:lightllm/common/basemodel/triton_kernel/att/prefill_att/context_flashattention_nopad.py:13`）

这是一个完整的、手写的 FlashAttention 风格算法——分块、在线 softmax、causal mask，算法设计的活一点没少做，Triton 只是省掉了用 CUDA C++ 手写这一层线程索引/共享内存搬运的繁琐程度。**Triton 买到的是"表达算法更省力、后端可移植性更好（同一份 Triton 代码理论上能同时面向 NVIDIA 和 AMD 的后端生成不同的机器码）"，买不到"帮你想出算法"。** `_lab/flashattn.py:3` 那句话是同一个道理的另一面：`"FlashAttention 一个 FLOP 都没省"`——在线 softmax 这个算法洞察是人想出来的（`_lab/flashattn.py:84` 的 `attention_online()` 就是这个洞察的最小实现），Triton/CUDA 只是负责把这个已经想清楚的算法映到硬件上,不负责替你发现它。

vLLM 和 SGLang 内部同样大量使用这一层。vLLM 里一个融合 QK RMSNorm 的 kernel：

```python
@triton.jit
def _fused_q_kv_rmsnorm_kernel(
```
（`vllm:vllm/models/common/ops/fused_qk_rmsnorm.py:9`）

以及 LoRA 场景下的一组 Triton kernel（`vllm:vllm/lora/ops/triton_ops/lora_expand_op.py:23`）。SGLang 的 decode attention 也是同一条路子（`sglang:python/sglang/kernels/ops/attention/decode_attention.py:246`）。这些都是**人手写的算法，用 Triton 这个 DSL 表达，编译器负责把 tile 级描述降到硬件指令**——跟 LightLLM 的 129 个文件是同一类东西，只是 vLLM/SGLang 没有把这条路线走到"全仓 0 CUDA"的地步（§4.1 会给出确切占比）。

#### 层 2：图级编译（TVM/Relax、torch.compile）——接管的是一整张图

这一层的输入不是一个 kernel 的 tile 描述，是**一整张计算图**——模型的 forward 被表达成一系列算子节点，编译器可以看到全局，做三类事：**把相邻的小算子融合成一个大 kernel、决定张量的内存/数据排布、把子图分发给具体的代码生成后端**。

MLC-LLM 的编译入口把这条流程走了个完整闭环：

```python
mod, named_params, ext_mods = model.export_tvm(
    spec=model.get_default_spec(),
    allow_extern=True,
)
```
（`mlc-llm:python/mlc_llm/interface/compile.py:163`）——模型对象先被"导出"成一张 TVM 的 Relax 图,再交给编译流水线：

```python
pipeline=relax.get_pipeline(
    "mlc_llm",
    target=args.target,
    ...
```
（`mlc-llm:python/mlc_llm/interface/compile.py:209`）——`target=args.target` 就是"同一张图,换一个硬件目标,重新走一遍这条流水线"的入口,这正是层 2 相对层 1 多出来的东西:层 1 的一份 Triton kernel 只服务一个算子;层 2 的一份图描述服务的是整个模型,可以对着不同硬件重新走生成流程。

图级编译真正的收益点是**融合**——具体是一个例子，MLC-LLM 有一个专门的 pass：

```python
"""A compiler pass that fuses add + rms_norm."""
```
（`mlc-llm:python/mlc_llm/compiler_pass/fuse_add_norm.py:1`）——把残差相加（add）和 RMSNorm 融合成一个 kernel。这两个算子单独执行时，第一个算子的输出要写回显存，第二个算子再把它读回来；融合之后，这次写回和读取都省掉了。**这不是省 FLOPs——两次算子加起来做的浮点运算次数完全一样，融合前后一模一样——省的是访存**，而 `02-自回归解码为什么是访存瓶颈.md` 已经证明 decode 天生卡在访存的地板上（`_lab/minigpt.py:203` 的 `batch=1` 恒为 0.5 那条结论）。`_lab/minigpt.py:235` 那句注释说得直白：`"算术强度：每读 1 字节能做多少次浮点运算。越低越是访存瓶颈。"`——融合类的图级优化,打的就是这个访存分母,不是分子。

值得先在这里点破一件事，第 4.3 节会展开：**图级编译并不会把层 1 替换掉——它经常还是要落到层 1 头上。** MLC-LLM 自己就有一个 pass 专门"把 Triton kernel 派发进图里"（`mlc-llm:python/mlc_llm/compiler_pass/dispatch_triton_kernel.py:1`），说明即便是这 12 家里编译走得最深的一家，遇到需要极致性能的算子（比如 w8a8 量化矩阵乘），最终答案依然是"调用一份手写的 Triton kernel"，不是"让图级编译器自己生成"。

vLLM 和 SGLang 的图级编译走的是 `torch.compile` 这条路，接管的是 PyTorch eager 模式下捕获出来的 FX 图。vLLM 的编译封装里：

```python
self._compiled_callable = torch.compile(
    compiled_ptr,
    fullgraph=True,
    dynamic=False,
    backend=backend,
    options=options,
)
```
（`vllm:vllm/compilation/wrapper.py:148`）；SGLang 同样的调用点：

```python
compiled_callable = torch.compile(
    bound,
    fullgraph=fullgraph,
    backend=backend_factory,
)
```
（`sglang:python/sglang/srt/compilation/compile.py:217`）。两处都传了自定义 `backend`——这意味着它们没有直接用 PyTorch 默认的 Inductor 后端把 Python 全部吃掉，而是接了一层自己的 pass（做算子融合、CUDA Graph 整合等），这正是图级编译"接管一整张图,重写它"的标准动作，只是接入方式跟 MLC-LLM 的 Relax 流水线是两套完全不同的具体实现。

#### 层 3：中间表示基础设施（MLIR）——不是编译器，是造编译器的工具箱

MLIR 本身不提供"把我的模型编译到某个硬件"这条现成流水线。它提供的是：一套通用的 IR 数据结构、一套定义"方言"（dialect，也就是你自己的一组 IR 算子和类型）的机制、一套在方言之间做转换（lowering）的 pass 框架。**谁想要一个新的编译器，不用从零写解析器/IR 数据结构/pass 调度器，在 MLIR 上定义一个方言就能省下这一大块地基工程**——但方言本身，以及方言到目标硬件的转换 pass，还是要自己写。

这 12 个引擎里刚好有一个真实案例：TensorRT-LLM 的 `auto_deploy` 路径。它没有直接用 TVM 那种"整套编译器都替你搭好"的方案，而是自己在（MLIR 概念的 Python 原生实现）`xDSL` 上定义了一套专属方言：

```python
"""FX graph → MLIR (xDSL) converter.

Walks an FX ``GraphModule`` topologically and emits ``ad`` dialect ops.
```
（`tensorrt-llm:tensorrt_llm/_torch/auto_deploy/mlir/fx_to_mlir.py:16`），方言定义在：

```python
"""AutoDeploy MLIR dialect definition using xDSL.

Defines the ``ad`` dialect with ops that mirror the FX graph operations
```
（`tensorrt-llm:tensorrt_llm/_torch/auto_deploy/mlir/dialect.py:16`），具体的算子类型比如：

```python
class AdAdd(IRDLOperation):
```
（`tensorrt-llm:tensorrt_llm/_torch/auto_deploy/mlir/dialect.py:163`，同一文件还有 `AdRMSNorm`、`AdMul` 等一整套自定义算子类型）。

**这段代码是"层 3"最好的教材**：TensorRT-LLM 团队要做的事是把一张 PyTorch FX 图转成一种自己的中间表示再做优化，他们没有从零发明一套 IR 数据结构和 pass 框架，而是在 MLIR 的思路（这里具体是它的 Python 实现 xDSL）上定义了自己的 `ad` 方言。这正是"MLIR 是造编译器的框架"这句话的字面意思——**它没有替 TensorRT-LLM 决定"图怎么融合、怎么生成代码"，只是替他们省掉了"怎么设计一套 IR 系统"这一层地基**。

这里必须澄清一个极容易搞混、但对读者理解"MLIR 是什么"至关重要的事实：**TVM 不是建立在 MLIR 之上的。** TVM 有自己独立的 IR 栈（历史上是 Relay，现在的主线是 Relax，往下还有 TIR），跟 MLIR 是两套各自独立发展、解决相邻问题的编译器基础设施，没有依赖关系。也就是说，本篇标题里的"TVM"和"MLIR"根本不是同一种东西：**TVM 是一个（相对）开箱即用的机器学习编译器，MLIR 是一套要你自己动手搭方言的编译器地基。** 前者的用户是"想编译一个模型的人"，后者的用户是"想造一个新编译器的人"——这条区分在 §2.3 和 §6 的误区四会具体展开。

### 2.3 为什么混谈会得出错误结论

把三层压扁成一个"编译器路线"概念，会导致三种具体的错误推理：

**把层 1 和层 2 混为一谈**：会得出"我们已经用了 Triton 写 kernel，说明团队已经认同编译器路线，不如再把 TVM 也接上"这种结论。但层 1（写一个 kernel，编译器负责底层映射）和层 2（把整张图交出去，编译器决定怎么融合/生成）解决的是两件不同规模、不同性质的问题，各自的收益和代价要分开算，不能因为已经接受了其中一层就默认另一层的账也划算——vLLM/SGLang 的真实选择（本篇第 1 节的数字）恰好证明了这一点：两家都大量用了层 1，但都没有把层 2 放到 MLC-LLM 那种架构中心的位置。

**把层 2 和层 3 混为一谈**：会得出"TVM 是建立在 MLIR 上的，所以要学会用 TVM 得先学 MLIR"这种错误认知。实际上二者是平行的两套基础设施——用 TVM 编译一个已有架构支持的模型，完全不需要接触 MLIR 那一层；反过来，MLIR 本身也编不出一个能跑的推理引擎，它只提供地基，真正"能编译模型"的流水线（融合 pass、算子生成规则、量化策略）还是要在这套地基之上自己搭，TensorRT-LLM 的 `auto_deploy/mlir/` 那几百行 `dialect.py`/`fx_to_mlir.py` 就是这份"自己搭"的工作量的真实体量。

**把"用了某种编译技术"等同于"编译器优先的引擎"**：vLLM 有 499 处 `@triton.jit`、104 处 `torch.compile(`，SGLang 的数字更高——但没有人会把 vLLM/SGLang 归类为"编译器优先"的引擎，因为它们的架构核心仍然是 Python 手写的调度器 + eager 模式的 PyTorch 前向，编译技术是**加在局部热点上的加速手段**，不是"整个引擎先过一遍编译流水线才能跑"的架构基石。MLC-LLM 恰恰相反——不导出到 TVM、不跑完整个 Relax 流水线，模型根本不能执行（§4.4 具体展开这一点的代价）。**"用没用编译技术"和"编译在架构里占据什么位置"是两个独立的问题，前者 12 家里几乎全部命中，后者只有 MLC-LLM 一家。**

### 2.4 一个容易被忽略的边界：融合省的是访存，访存不总是瓶颈

2.2 节层 2 已经点破"算子融合省的是访存不是 FLOPs"这条结论,但这条结论本身还有一层边界经常被漏掉：**如果你的场景本来就不是访存受限,融合能带来的收益会跟着塌下去,因为它优化的那个分母压根不是瓶颈。**`02-自回归解码为什么是访存瓶颈.md` 用 `_lab/minigpt.py` 的解析模型把这条边界量化到了具体数字上——同样是 fp32,`ctx=16` 时 `bs=256` 能把算术强度抬到 21.74,KV 占访存的比例只有 54.7%；而 `ctx=16384` 时 `bs=256` 只能把算术强度抬到 0.53,KV 占访存的比例涨到 99.9%(`_PLAN.md` §3.1 的完整表格,`_lab/minigpt.py:203` 是这条结论的解析证明)。

这条曲线对编译器路线的含义是:**同一个"融合 add+RMSNorm"这样的 pass,在长上下文、小 batch 的 decode 场景里能带来实打实的收益(访存本来就是瓶颈,少一次读写就是少一段真实等待时间);换到大 batch 的 prefill 场景(一次性处理大量新 token,算术强度天然就高,矩阵乘本身已经把算力喂得比较饱和),同一个融合 pass 省下来的那点访存开销,占总耗时的比例会小得多——融合本身没有变差,是它能撬动的那部分开销在这类场景里本来就占比不高。**

这解释了一个容易让人困惑的现象:为什么图级编译器在某些场景宣传的收益很显著,换一个场景（比如离线批量跑大 batch prefill）却感觉不出明显差别——**不是编译器"失效"了,是它优化的那个变量在新场景里权重变小了。** 评估"值不值得为某个模型/某个场景引入图级编译"时,先用 `_lab/minigpt.py` 这类解析模型（或者对着自己真实的 batch/序列长度分布）估一下这个场景的算术强度落在哪个区间,比直接假设"编译器总是有用"更可靠。

## 3. 自己动手：去 `_src/` 里自己数这些数

上面所有数字，都能在这份研究库的 clone（`../VLLM-SGlang-研究-推进/_src/`）上自己跑出来。给出命令之前先提一个隔壁库真踩过的坑（见用户记忆 `inference-engines-kb` 的教训）：**短缩写统计必须用词边界，否则大量误报。** 这里最典型的陷阱是"triton"这个词本身有歧义——它既指 OpenAI 的 Triton 算子 DSL（本篇讨论的对象），也指 NVIDIA Triton Inference Server（一个完全不相关的服务框架，`tensorrt-llm/triton_backend`、`dynamo/examples/backends/tritonserver` 这些目录名都在用这同一个词的另一层含义）。所以下面的命令**一律锚定在 `@triton.jit` 这个具体的装饰器写法上**，不是裸的 `triton` 关键词——只有真正定义 kernel 的地方才会出现这个写法，服务框架的引用不会。

```bash
cd ../VLLM-SGlang-研究-推进/_src

# ── 层 1：算子级 DSL（Triton），按 @triton.jit 精确锚定 ──
# -l 数"含有它的文件数"，-o 数"出现的总次数"（一个文件可能有多个 kernel）
grep -rl "@triton\.jit" vllm    --include="*.py" | wc -l   # → 202
grep -ro "@triton\.jit" vllm    --include="*.py" | wc -l   # → 499
grep -rl "@triton\.jit" sglang  --include="*.py" | wc -l   # → 249
grep -ro "@triton\.jit" sglang  --include="*.py" | wc -l   # → 631
grep -rl "@triton\.jit" lightllm --include="*.py" | wc -l  # → 129
find lightllm -name '*.cu' -o -name '*.cuh' | wc -l        # → 0（全仓真的没有）

# ── 层 2：图级编译（torch.compile），锚定实际的函数调用写法 ──
grep -rl "torch\.compile(" vllm   --include="*.py" | wc -l  # → 61
grep -ro "torch\.compile(" vllm   --include="*.py" | wc -l  # → 104
grep -rl "torch\.compile(" sglang --include="*.py" | wc -l  # → 64
grep -ro "torch\.compile(" sglang --include="*.py" | wc -l  # → 86

# ── 对照组：手写 CUDA 文件数，看编译路线到底替代了多少 ──
find vllm   -name '*.cu' -o -name '*.cuh' | wc -l           # → 169
find sglang -name '*.cu' -o -name '*.cuh' | wc -l           # → 246

# ── N 的量级：模型文件/目录数 ──
find vllm/vllm/model_executor/models -maxdepth 1 -name '*.py' | wc -l   # → 292
find mlc-llm/python/mlc_llm/model -maxdepth 1 -type d | wc -l           # → 44（含父目录本身，模型子目录 43）
```

三点如实说明数法，避免读者拿这些数字去做超出口径的外推：

1. **`-l`（文件数）和 `-o`（出现次数）是两个不同的口径**，一个文件里可能有多个 `@triton.jit` kernel。上表两个数字都给了，别把它们混用。
2. **没有排除测试/benchmark/vendor 代码**，是仓库整体的口径。拆开看更精确：vLLM 202 个命中文件里，196 个在 `vllm/vllm/`（主源码），22 个在 `vllm/vllm/third_party/`（vendor 进来的第三方实现，跟 22 个有 5 个重叠计入了测试目录，数字不严格可加但量级可信），5 个在 `tests/`，1 个在 `benchmarks/`；`torch.compile(` 61 个命中文件里 23 个在主源码、32 个在 `tests/`——**测试代码里出现 `torch.compile(` 往往是在测编译路径本身是否正确，不代表主链路的使用密度**，想知道"生产路径实际用了多少"要看主源码那个子集的数字。SGLang 侧类似：`@triton.jit` 241/249 在 `python/sglang/` 主源码；`torch.compile(` 48/64 在主源码，14 个在 `test/`。
3. **`.cu`/`.cuh` 文件数只统计文件后缀**，不代表这些文件里的每一行都在被使用（可能有过时/实验性代码），也不代表 Triton kernel 数量就等于"被替代掉的 CUDA kernel 数量"——两者不是一一对应关系，这里只是给一个量级对照，不是精确的"替代率"。

## 4. 真实引擎是怎么做的

第 2 节已经用大量带行号的证据把三层原理立住了，这一节把镜头拉远，回答三个更贴近"选型"的问题：vLLM/SGLang 这种量级的引擎里，编译技术具体分布在哪些子系统？MLC-LLM 把编译放在架构中心，具体换来了什么、又具体付出了什么？"动态形状是编译路线的死穴"这句话，在真实源码里长什么样？

### 4.1 vLLM/SGLang：编译技术分布在哪些子系统，不是"整个引擎过一遍编译"

§1 的数字已经说明 vLLM/SGLang 内部大量使用层 1 和层 2，这里具体看它们分布在哪几类子系统，而不是笼统地说"用了 Triton"。

**层 1（Triton）出现的地方**：从 §2.2 举过的例子能看出规律——`vllm:vllm/lora/ops/triton_ops/lora_expand_op.py:23` 属于 LoRA（多套 adapter 权重的批量矩阵乘，形状组合太多，手写 CUDA 要为每种 rank 单独写一份，用 Triton 参数化省掉这个组合）；`vllm:vllm/models/common/ops/fused_qk_rmsnorm.py:9` 属于模型专属的小算子融合；`sglang:python/sglang/kernels/ops/attention/decode_attention.py:246` 属于 decode 阶段的 attention kernel（sglang 的这一份是官方维护的主力实现之一，不是外包给第三方库）。这三类共同点是：**要么形状/配置组合太多不值得为每种手写一份 CUDA，要么是团队自己维护的算法，本来就打算用比 CUDA C++ 更省力的方式写。**

**层 2（`torch.compile`）出现的地方**：vLLM 和 SGLang 都把它接在模型 forward 的主链路上，不是零星调用。vLLM 的 `support_torch_compile` 装饰器（`vllm:vllm/compilation/decorators.py:118` 起）挂在模型的顶层 `nn.Module` 上，`vllm:vllm/compilation/wrapper.py:148` 是实际发起编译调用的地方；这一层做的事情是把 PyTorch eager 模式下逐算子调用的开销，通过 `fullgraph=True` 尽量整图捕获再一次性优化/生成——这跟 `02-Python为什么还没被淘汰.md` 里 `T_py` 这条线是相关的：`torch.compile` 减少的正是"每一步 decode 要发起多少次 Python/CUDA API 调用"这部分开销,不是矩阵乘本身的计算量。

**还有一条容易被"编译器路线"这个词忽略的做法：外包给专家手写库，既不是自己写 Triton，也不是让编译器生成。** vLLM 和 SGLang 的核心 attention 后端都直接 `import` 第三方专家库——vLLM 的 FlashInfer 后端（`vllm:vllm/v1/attention/backends/flashinfer.py:12` 起 `from flashinfer import (...)`）、SGLang 的同名后端（`sglang:python/sglang/srt/layers/attention/flashinfer_backend.py:72` 起同样的导入）。

这条路径既不属于层 1（不是自己写 Triton kernel），也不属于层 2（不是把图交给编译器），是三层框架之外真实存在的第四个选项：**这个算子已经有别的团队用手写 CUDA/Triton 打磨到接近硬件极限了，直接调用，不重新发明。** §1 数字表里 vLLM 169 个、SGLang 246 个手写 `.cu`/`.cuh` 文件，有相当一部分正是这类外部依赖库编译产物或者围绕它们做的适配代码,不是引擎团队从零手写的算子实现。

**两层同时出现，但都不是架构中心**：vLLM/SGLang 的架构骨架依然是"Python 调度器 + PyTorch eager 前向"，Triton kernel、`torch.compile`、外部专家库，三者都是挂在这个骨架上的加速手段，用不用、用在哪，是可以逐个子系统单独决定的局部优化——去掉 `torch.compile`，vLLM 依然能跑（用 eager 模式），只是慢一些；这跟下面 MLC-LLM 的情况正好相反。

### 4.2 MLC-LLM：编译放在架构中心，换来了什么、付出了什么

这 12 家里只有 MLC-LLM 把"编译"放在了架构核心——**不导出到 TVM、不跑完整 Relax 流水线，模型根本没法执行**（`mlc-llm:python/mlc_llm/interface/compile.py:163` 的 `export_tvm` 和 `:209` 的 `relax.get_pipeline` 是必经步骤，不是可选加速）。这个架构选择具体换来了什么、要付出什么代价，值得摊开算。

**换来的东西是真实的、可验证的**：MLC-LLM 自己的 README 列出的后端矩阵——

```
Linux / Win：  AMD ✅ Vulkan, ROCm    NVIDIA ✅ Vulkan, CUDA    Intel ✅ Vulkan
macOS：        AMD ✅ Metal (dGPU)    Apple ✅ Metal            Intel ✅ Metal (iGPU)
Web Browser：  ✅ WebGPU and WASM
iOS/iPadOS：   ✅ Metal on Apple A-series GPU
Android：      ✅ OpenCL (Adreno / Mali)
```
（`mlc-llm:README.md:22-58`）——**这条矩阵是编译器路线在这 12 家里唯一真正兑现"一份描述、多种后端"承诺的地方**。vLLM/SGLang 没有一家能在浏览器里跑（它们的架构假设就是"有一块 GPU 装了 CUDA/ROCm 驱动的服务器"），MLC-LLM 靠着把模型表达成 TVM 的图、让 TVM 的后端代码生成器分别吐出 WebGPU/Metal/Vulkan/OpenCL 各自的机器码，做到了同一份 `LlamaForCausalLM` 定义能跑在从数据中心 GPU 到手机的整条谱系上。这不是一句营销话术,是可以在 `README.md` 的这张表里逐格核对的能力。

**付出的代价同样具体，而不是抽象的"多一层复杂度"**：新模型接入 MLC-LLM，要用它自己的图追踪前端重写一遍模型定义。以 Llama 为例，MLC-LLM 里的实现：

```python
from tvm.relax.frontend import nn
...
class LlamaAttention(nn.Module):
    ...
    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
```
（类定义在 `mlc-llm:python/mlc_llm/model/llama/llama_model.py:139`，`forward` 在 `:157`）——这里的 `nn.Module` 来自 `tvm.relax.frontend.nn`，**不是 PyTorch 的 `torch.nn.Module`**，整个文件没有 `import torch`（可以自己在 `mlc-llm/python/mlc_llm/model/llama/llama_model.py` 的 import 区核对）。对照 vLLM 同一个模型的实现：

```python
from torch import nn
...
class LlamaAttention(nn.Module):
```
（`vllm:vllm/model_executor/models/llama.py:31,122`）——**这是普通的 PyTorch eager `nn.Module`**，可以直接单步调试、直接跑通任意 Python 控制流,不需要经过任何编译步骤就能执行。

两份代码的类名、大体结构几乎一样，但背后的开发体验完全不同：vLLM 的版本写完就能在解释器里跑，出错了直接看 Python 栈；MLC-LLM 的版本写完还要经过 `export_tvm` 把它转成 Relax 图、再跑一遍 `relax.get_pipeline` 编译流水线才能拿到一个能执行的产物，图追踪过程中不允许任意的 Python 控制流（数据依赖的分支、非固定形状的循环这类东西在图追踪阶段是不允许的或者需要特殊处理）——**这是"一份描述、多种后端"这个承诺背后真实的迭代成本：写模型这一步的开发循环，从"改代码→直接跑"变成了"改代码→导出图→跑编译流水线→跑得动才能测"。**

本库不产未经查证的"谁比谁支持新模型更快"这类时间线数字（`03-语言选型/` 系列的诚实标准不允许），这里只如实描述机制上的因果关系：**多出来的"重写成图追踪前端 + 走一遍编译流水线"这一步，结构上必然比"改一份 PyTorch eager 代码就能跑"多出一段不产生业务价值、纯粹为了满足编译器要求而存在的工作量**——这段工作量具体会转化成多长的接入延迟，取决于团队规模和这个模型架构离已有模板有多远，本篇不替这件事编一个数字。

**这条代价—收益的取舍不是 MLC-LLM 团队的失误，是他们主动选的定位**：如果你的目标就是"同一个模型要能跑在浏览器、手机、桌面独显这些完全不同的硬件/运行时上"，vLLM/SGLang 的架构根本做不到这件事（它们的算子实现绑死在 CUDA/ROCm），这时候 MLC-LLM 付出的"每个新模型要重写一遍 + 走一遍编译流水线"的代价，换来的是一个其他 11 家都给不了的能力。反过来，如果你的目标只是"在一台装了 A100 的服务器上尽快跑通并优化吞吐"，这条代价换不来任何用得上的东西——这正是 §5 决策三要具体展开的判据。

### 4.3 "动态形状是死穴"——证据在 vLLM 自己的配置注释里

编译路线（尤其是层 2 图级编译）最常被低估的代价是：**编译产物往往只对编译时见过的形状有效，一个新形状出现，要么触发重新编译，要么直接不能用。** 推理引擎里 batch size 和序列长度天天在变，这条代价不是边缘情况，是主路径天天要面对的事。

vLLM 自己的配置代码把这件事写得非常直白。CUDA Graph 相关的配置项：

```python
cudagraph_capture_sizes: list[int] = None  # type: ignore[assignment]
"""Sizes to capture cudagraph.
- None (default): capture sizes are inferred from vllm config.
- list[int]: capture sizes are specified as given."""
```
（`vllm:vllm/config/compilation.py:648`）——这是一个**离散的、有限的整数列表**，不是"支持任意 batch size"。为什么必须是离散列表，代码注释里直接给了原因：

```
Why we have different sizes for cudagraph and inductor:
- cudagraph: a cudagraph captured for a specific size can only be used
    for the same size. We need to capture all the sizes we want to use.
```
（`vllm:vllm/config/compilation.py:434-436`）——**"为某个具体尺寸捕获的 CUDA Graph 只能用于同一个尺寸"，想用哪些尺寸就得把每一个都单独捕获一遍。** 这不是 vLLM 工程上的疏漏，是 CUDA Graph 这项技术本身的性质——它记录的是一串具体的、固定地址/固定形状的 GPU 操作序列，换一个 batch size 意味着张量形状变了，之前录好的那份操作序列直接不能复用。`torch.compile` 侧的处理稍微宽松一些（同一份 Inductor 编译产物在一定条件下能覆盖多个形状），但源码注释里紧接着承认这条路径也有自己的成本——为具体尺寸编译能拿到更多优化空间，泛化的动态形状路径优化空间更窄。

**结果是：想要编译路线的完整性能收益，实践中往往要为"这个引擎实际会遇到的形状集合"预先枚举、预先捕获/编译——这个集合越大，前期编译时间成本越高；集合覆盖不到的形状，要么退化到未优化路径，要么触发一次新的编译（新形状第一次出现时的那次请求会明显变慢）。**

这条证据是 §6 误区一的直接支撑——"编译器能自动解决动态形状问题"这句话，被 vLLM 自己的配置代码反证了：如果编译器真的能通用地处理任意动态形状且不损失性能，`cudagraph_capture_sizes` 这个需要手工枚举尺寸的配置项就不需要存在。

### 4.4 一条如实的补充：TensorRT-LLM 的 MLIR 用法是"beta 分支"，不是主路径

§2.2 层 3 举的 `tensorrt-llm:tensorrt_llm/_torch/auto_deploy/mlir/` 例子必须补一句准确的定位说明，否则容易被误读成"TensorRT-LLM 的主编译流程是基于 MLIR 的"——事实不是这样。TensorRT-LLM 自己的 README 把这条路径描述为：

> "AutoDeploy: A beta backend for TensorRT LLM to simplify and accelerate the deployment of PyTorch models."（`tensorrt-llm:README.md:343`）

**"beta backend"**——这是一条明确标注为测试阶段的备选后端，服务的场景是"直接部署 PyTorch 模型、少走 TensorRT-LLM 传统的引擎构建流程"，跟 TensorRT-LLM 的旗舰路径（基于 NVIDIA 自家 TensorRT 的图编译/引擎构建体系，不经过 MLIR 也不经过 TVM）是两条并行存在、服务不同场景的技术栈。这条补充跟 4.1 节 vLLM/SGLang 的情况是同一个教训的另一个例子：**一个引擎内部出现某种编译技术，不代表这种技术就是这个引擎的架构主干**——判断"主干还是旁支"要看这条路径是不是新用户默认会走的路径、是不是文档标注为主推方案，不能只看"仓库里存在这样的代码"。

### 4.5 编译时间本身是一项要被测量的成本，不是可以忽略的一次性开销

前面几节讲的都是编译产物的收益/代价，还有一项代价容易被低估：**编译/自动调优这个过程本身要花时间，而且往往花在引擎启动的关键路径上**——本库不产任何真实硬件上的秒数（红线：本机无 GPU），但"这段时间是否重要到值得专门测量并打日志"这件事，vLLM 自己的源码给出了明确的态度。

`torch.compile` 封装的调用点在完成编译后，会走进一段专门为这个阶段计时的代码：

```python
with monitor_torch_compile(
    self.vllm_config,
    "torch.compile and initial profiling/warmup "
    "run together took %.2f s in total",
    is_encoder=self._is_encoder,
):
```
（`vllm:vllm/compilation/decorators.py:674-675`）——注意这条日志模板把"`torch.compile` 编译"和"初始 profiling/warmup 运行"算作**同一段要被测量的总时间**。配套的计时器实现：

```python
start = time.perf_counter()
yield
elapsed = time.perf_counter() - start
...
logger.info_once(
    "Initial profiling/warmup run took %.2f s",
    elapsed,
)
```
（`vllm:vllm/compilation/monitor.py:82`）——`time.perf_counter()` 掐表、`logger.info_once` 在引擎启动时把这个数字打印出来。**一个成熟到已经被大量生产环境使用的引擎，专门为"编译+首次运行"这一段写了计时和日志基础设施，这件事本身就是最有力的证据：这段时间在真实使用中足够长、足够重要，值得被单独测量、单独展示给运维人员看。** 如果编译时间可以忽略不计，不会有人专门为它写一条日志模板。

这条代价跟 §4.3 的动态形状代价是同一枚硬币的两面：`cudagraph_capture_sizes` 决定了"要为多少种形状分别捕获/编译"，形状集合越大，上面那条日志打印出来的总时间就会越长——**覆盖面（多少种形状能吃到编译优化）和启动成本（这些形状分别要编译多久）是同一个旋钮的两端，图级编译没有办法同时把两端都做到最优，只能根据自己的真实工作负载去选一个折中点。** 这也是决策一"先手写一版能跑的"这条建议的另一层理由：手写路线没有这项启动期编译成本，想快速验证一个想法、跑一次实验，手写路径的"从改代码到看到结果"这个循环里不会插进来一段不确定时长的编译等待。

## 5. 设计决策与代价（为什么这样 / 不这样会怎样 / 什么时候可以不这样）

### 决策一：别从编译器路线起步，先手写一版能跑的

- **为什么这样**：编译器路线的全部收益都建立在"有一个正确的、可信的基准"之上——无论是层 1 的 Triton kernel 还是层 2 的图级编译，判断"融合/生成之后结果对不对"都需要一份手写的、已经验证过正确性的参照实现去比对。`_lab/flashattn.py:3` 那句"FlashAttention 一个 FLOP 都没省"背后是先有 `attention_naive()`（`_lab/flashattn.py:37`）这份朴素实现立住正确性基准，才有资格去写 `attention_online()`（`_lab/flashattn.py:84`）这份优化版本并且证明它输出一致。没有基准，编译器生成的代码错了你根本发现不了——它不会崩，只会安静地给出错误答案，这跟 `00-总览与学习路径.md` §3 强调的"KV 缓存/分页/批处理出错都是静默的"是同一类风险,编译器生成的代码只会让这个风险更隐蔽,因为你连"生成的代码长什么样"都不一定看得懂。
- **不这样会怎样**：直接从图级编译或者 DSL 起步，等于把"这段逻辑对不对"和"这段代码怎么被编译器改写"两个问题叠在一起调试——出错时你面对的是编译器吐出的中间产物,而不是自己写的、能一步步跟踪的代码,排查成本会远高于"先有基准再谈优化"这条顺序。
- **什么时候可以不这样**：如果目标本身就是学习/魔改某个成熟编译器（比如给 TVM 提交一个新的融合 pass、给 Triton 调试后端代码生成），那手写基准这一步的角色会反过来——这时候正确性基准由这个编译器项目自己现成的测试集提供,你不需要重新造一个,直接站在已有基础设施上工作是合理的起点。

### 决策二：第一层要引入的是 Triton，不是图级编译器

- **为什么这样**：三层里 Triton 的心智负担最小、回退成本最低——写一个 Triton kernel 本质上还是在写一个具体的算法（跟写 CUDA C++ 的心智模型是同一类），只是省掉了手动管理寄存器/共享内存这层麻烦；出问题时可以直接对着这一个 kernel 调试，不需要理解一整套图重写/融合规则。回退也简单：这个 kernel 效果不好，换回纯 Python/numpy 实现（哪怕慢），不影响引擎其他部分——LightLLM 的 129 个文件、vLLM 的 202 个文件，都是"一个算子一个文件、互相独立"的组织方式，这正是 Triton 这一层收益/风险比最好的地方：局部失败不牵连全局。
- **不这样会怎样**：如果第一层就上图级编译器（TVM/torch.compile），失败模式会牵连整个模型——一次编译要么整体成功要么整体失败，中间调试过程要面对一整张被重写过的图，出错定位的难度比调一个 kernel 高一个数量级；而且图级编译器的收益（融合、跨硬件代码生成）在引擎还没有稳定跑起来之前根本无从谈起，投入产出比很差。
- **什么时候可以不这样**：如果你的引擎从第一天起就有非常明确、已知的多硬件目标（不是"以后可能要支持"，是现在就要交付到两种以上完全不同的硬件/运行时上），并且团队已经有成熟的图编译经验，直接从层 2 起步是合理的——但这个前提在"从 0 开始学写引擎"这个场景里几乎不成立，见决策三。

### 决策三：不要为不存在的多后端需求提前上图级编译器

- **为什么这样**：图级编译器（尤其是 MLC-LLM 这种把编译放在架构中心的方案）最大的真实收益是"一份模型描述、多种硬件后端"（§4.2 的后端矩阵是证据），这个收益只有在你**真的需要覆盖多种硬件**时才能兑现；代价（每个新模型要用编译器自己的前端重写一遍、开发迭代循环变慢、编译时间成本、动态形状要预先枚举）却是从第一天就要背上的固定成本，不管你最终用不用得上多后端这个收益。绝大多数"从 0 写一个推理引擎"的场景，目标硬件从一开始就是确定的一种（你手头那块 GPU），这时候多后端能力是一个用不上的期权，代价却是实打实要付的。
- **不这样会怎样**：如果提前上了图级编译栈却始终只服务一种硬件，相当于长期背负决策换来的代价（模型接入变慢、调试难度上升、编译时间），却从来没有兑现那份"多后端"的收益——这是纯亏损的选型，而且这类选型的成本经常是隐性的、后知后觉才会发现（新模型接入总是比同类项目慢,一开始不容易意识到这是架构选型的代价,而不是团队能力问题）。
- **什么时候可以不这样**：目标硬件从一开始就明确不止一种，且几乎可以确定不会收窄到一种（比如产品定位就是"同一个模型要能在服务器和用户手机上都跑起来"）——这时候 MLC-LLM 式的架构才是在为一个真实存在、而不是假设存在的需求付费。判断标准很具体：**能不能现在就写出两个以上、具体到型号的目标硬件清单？写不出来，就不是"现在"该上图级编译器的时机。**

### 决策四：`torch.compile` 适合接在一个已经能跑的 PyTorch 引擎之后，不是从零开始的起点

- **为什么这样**：§4.1 已经看到 vLLM/SGLang 的真实用法——`torch.compile` 是挂在一个已经用 PyTorch eager 模式跑通的模型骨架上的加速层,不是骨架本身。这样做的好处是**编译失败可以直接回退到 eager 模式继续跑**（vLLM/SGLang 的架构都保留了这条退路），编译带来的收益（减少 kernel 提交次数、部分融合）是纯增量的，不影响"引擎能不能跑"这条底线。
- **不这样会怎样**：如果从项目一开始就假定"最终一定要靠 `torch.compile` 才能跑起来"，会把"模型逻辑对不对"和"编译器能不能正确处理这份逻辑"两个问题的调试耦合在一起——`torch.compile` 对动态控制流、某些 Python 内建操作有覆盖限制，这些限制在 eager 模式下不存在，先在 eager 模式下把模型写对，再逐步套上 `torch.compile`，能把"哪里踩了编译器的坑"这个问题单独隔离出来调试。
- **什么时候可以不这样**：如果引擎的目标场景从一开始就要求"启动即编译"（比如线上服务不能接受运行时性能抖动，必须提前把所有会用到的形状编译好、CUDA Graph 全部捕获完才对外提供服务），那么"编译是启动流程的必经步骤"本身就是设计要求，不是可选加速——但即便如此，开发阶段依然应该先用 eager 模式验证模型逻辑，"编译是必经步骤"和"先用 eager 调试模型"这两件事并不冲突。

### 决策五：只有真的要造硬件后端时才碰 MLIR 这一层，应用层引擎作者不需要

- **为什么这样**：MLIR/xDSL 这一层解决的是"我要造一个新编译器,不想从零写 IR/pass 基础设施"这个问题——TensorRT-LLM 的 `auto_deploy/mlir/dialect.py` 花了近 500 行去定义一套 `ad` 方言（`tensorrt-llm:tensorrt_llm/_torch/auto_deploy/mlir/dialect.py:16` 起），这是**造编译器的工程量**,不是"用编译器加速一个模型"的工程量。写一个推理引擎的目标是"让模型跑起来、跑得快",不是"发明一套新的中间表示"——除非你面对的是一个连 TVM/Triton/torch.compile 现成后端都覆盖不到的全新硬件目标（比如一颗全新的、还没有任何主流编译器支持的自研 ASIC）,否则你要解决的问题,层 1、层 2 已有的现成工具足够覆盖,不需要再往下一层挖到"造编译器地基"这个深度。
- **不这样会怎样**：把 MLIR 当成"更底层所以更值得学"而提前投入，会把大量时间花在跟"让我的模型跑起来"没有直接关系的基础设施工程上——定义方言、写 lowering pass 本身就是一个不小的项目,对于一个想学会"怎么造一个推理引擎"的人来说，这是在错误的抽象层上花时间。
- **什么时候可以不这样**：你的团队本身就是硬件厂商或者编译器基础设施团队，需要给一款还没有任何现成编译器支持的硬件从头搭一条编译路径——这种情况下 MLIR 式的方言机制确实能省下"从零写 IR 系统"这一块地基工程，是合理的起点，但这已经不是"写一个推理引擎"这个任务本身了，是一个规模大得多、性质也不同的项目（造编译器,而不是造引擎）。

### 一句收束：三层决策的判据可以叠成一条流程

把上面五条决策串起来，能得到一条对"从 0 开始的人"最实用的顺序：**先手写基准（决策一）→ 遇到具体的、局部的性能痛点时，第一个尝试的优化是把这个算子换成 Triton（决策二）→ 只有确认目标硬件真的不止一种时，才考虑把某个模型的编译流程接入图级编译器，而且优先考虑"给已经能跑的 PyTorch 骨架接上 `torch.compile`"这种侵入性最小的方式（决策三、四）→ MLIR 这一层几乎不会出现在这条路径上，除非你的项目性质已经变成"造编译器"而不是"造引擎"（决策五）。** 这条顺序的核心逻辑是：**编译器路线的每一层都是在拿"现在就要付的确定成本"换"未来可能兑现的收益"，层数越深，成本越前置、越确定，收益越依赖一个具体的、必须现在就能写清楚的前提条件是否成立。**

## 6. 常见错误与踩坑

**误区一：以为编译器能自动解决动态形状问题**

§4.3 已经给了反证：vLLM 自己的 `cudagraph_capture_sizes: list[int]`（`vllm:vllm/config/compilation.py:648`）是一份需要手工枚举的离散尺寸列表，源码注释直接写明原因——"a cudagraph captured for a specific size can only be used for the same size"（`vllm:vllm/config/compilation.py:434-436`）。**编译器不是把"动态形状"这个问题变没了，是把它变成了"要不要、值不值得为每个可能出现的形状单独付一次编译/捕获的成本"这个新问题。** 推理引擎里 batch size 和序列长度是运行时才知道的、持续变化的量，这条代价躲不掉，只能在"覆盖更多形状（编译时间/显存成本上升）"和"覆盖更少形状（未命中形状退化到慢路径或触发运行时重编译）"之间选一个点。以为上了编译器就不用再操心这件事，是本篇见过的最常见的误判。

**误区二：以为算子融合能省 FLOPs，其实主要省的是访存**

`mlc-llm:python/mlc_llm/compiler_pass/fuse_add_norm.py:1` 那条 pass 把 add 和 rms_norm 融合成一个 kernel，融合前后两个算子做的浮点运算次数完全相同——**融合省下来的是"中间结果写回显存、再读回来"这一次访存往返**，不是减少了任何一次乘法或加法。`02-自回归解码为什么是访存瓶颈.md` 已经证明 decode 天生卡在访存的地板上（`_lab/minigpt.py:203`），`_lab/flashattn.py:3` 那句"FlashAttention 一个 FLOP 都没省"是同一个道理在另一个算子上的体现。把"融合"简单理解成"算得更快了"，会让人对编译器路线的收益产生错误预期——**如果你的场景恰好不是访存受限（比如超大 batch 的 prefill，算术强度已经很高），算子融合能带来的收益会显著变小，因为它优化的那个分母（访存字节数）在这类场景里本来就不是瓶颈。**

**误区三：为不存在的多后端需求提前上图级编译栈**

MLC-LLM 的后端矩阵（`mlc-llm:README.md:22-58`）是真实收益，但它是用"新模型要用 `tvm.relax.frontend.nn` 重写一遍、经过 `export_tvm`（`mlc-llm:python/mlc_llm/interface/compile.py:163`）和整条 Relax 流水线（`:209`）才能跑"这份代价换来的（§4.2 已经拿 `LlamaAttention` 在 MLC-LLM 和 vLLM 里的两份不同实现对比过）。如果你的引擎实际只服务一种硬件——这是"从 0 写一个推理引擎"场景里的大多数情况——这份代价从第一天就要背上，收益却永远兑现不了。**"以后可能要支持更多硬件"不是一个够格的理由**：决策三的判据很具体——写不出两个以上、具体到型号的目标硬件清单,就不该现在为这件事预付图级编译栈的成本。

**误区四：把 Triton 和 TVM 当成同一层东西**

这是本篇要拆解的核心混淆。Triton 是层 1——你仍然要写一个具体算法的 kernel，编译器负责把 tile 级描述映射到硬件指令；TVM/Relax 是层 2——接管一整张计算图，做融合/排布/跨后端代码生成，你写的是模型的图描述,不是某一个算子的实现。两者的相似之处只在于都用了"编译"这个词、都能生成可执行代码；差异之处才是真正决定选型的地方：**Triton 的失败模式是局部的（一个 kernel 写错/编译失败，不影响其他算子）,图级编译的失败模式是全局的（一次编译失败,整个模型编不出来）；Triton 几乎不需要为它单独设计一套模型描述语言,图级编译需要（MLC-LLM 的 `tvm.relax.frontend.nn` 就是这样一套语言）。** 把两者混为一谈，容易产生"已经用了 Triton 说明团队认同编译器路线，该顺势上 TVM 了"这种不成立的推论（§2.3 已经具体拆过）。

MLIR 更是第三层完全不同的东西——它不生成任何可执行代码，只是给"造一个新编译器"这件事提供地基,`tensorrt-llm:tensorrt_llm/_torch/auto_deploy/mlir/dialect.py:16` 那套 `ad` 方言证明了这一点：TensorRT-LLM 团队要做的不是"调用 MLIR 编译模型",是"用 MLIR 的机制自己搭一套编译器",这跟"调用 TVM 编译模型"是完全不同量级的工程。

**误区五：以为"要么全手写要么全编译"是唯二选择**

§1 的数字已经反驳了这个假二分：vLLM 有 499 处 `@triton.jit`、104 处 `torch.compile(`，同时还有 169 个手写 CUDA 文件；SGLang 的三个数字分别是 631、86、246。**主流引擎的真实架构是三种东西并存**——某些算子外包给 FlashAttention/FlashInfer 这类专家手写库、某些算子自己用 Triton 写、模型主链路挂一层 `torch.compile` 做增量优化、极少数极致场景保留手写 CUDA。选型不是"选一个阵营"，是**逐个子系统单独决定**：这个算子形状组合太多、值得用 Triton 参数化吗？这段图结构值得接一层 `torch.compile` 吗？这个热点值不值得单独手写 CUDA 抠到极致？把选型问题错误地简化成"我们是手写派还是编译派"，会让人跳过这些本该逐个做的判断，用一个不存在的全局立场替代了一堆本该独立回答的具体问题。

**误区六：以为编译器生成的代码可以不用像手写代码一样做"优化前后输出一致"的测试**

`00-总览与学习路径.md` §3 立的规矩是"KV 缓存/分页/批处理这几类优化出错都是静默的，唯一护栏是算两遍、逐位比"——这条规矩对编译器生成的代码**只会更重要，不会更不重要**。决策一已经点破了原因：出错时你面对的是编译器吐出的中间产物，排查成本天然更高；但更容易被忽略的是,"编译器很成熟、经过大量测试"这个印象,会让人误以为编译产物天然正确、不需要再验一遍。

这个印象是错的——`torch.compile` 对动态控制流、部分 Python 内建操作的处理有已知的覆盖限制（决策四已经提到）,MLC-LLM 的 Relax 流水线要经过导出图、跑量化、跑融合 pass 好几道工序（§4.2）,任何一道工序都可能因为某个新模型的某种张量形状/算子组合而生成出跟 eager 模式不一致的结果——**这不是"编译器有 bug"这种小概率事件,是"图追踪/自动融合本身要处理的组合数太大,没有覆盖到所有情况"这种结构性风险。**

真正安全的做法是把这条测试哲学原样搬过来：**同一个模型,编译前跑一遍、编译后跑一遍,拿同样的输入比对 logits（或者至少比对生成的 token 序列）,是不是完全一致（或者在数值误差可接受的范围内一致）。** 没有这一步，"编译器帮我优化了"和"编译器悄悄改变了我的模型行为"这两件事,你是分不清的——而后者恰恰是这个领域里最容易被放过的一类 bug。

## 7. 自测题与延伸阅读

**自测题（闭卷回答，答案都在本篇里）**：

1. 本篇说"编译器路线"其实是三层不同的东西。分别说出层 1、层 2、层 3 各自的输入是什么（一个 kernel 的描述？一整张图？还是别的？），以及编译器在这三层里各自接管的是流程的哪一部分。
2. LightLLM 全仓 0 个 CUDA 文件、129 个 Triton 文件，这证明了什么、没有证明什么？如果有人说"这说明 Triton 已经能完全取代 CUDA"，这句话哪里说过了头？
3. `mlc-llm:python/mlc_llm/compiler_pass/fuse_add_norm.py:1` 那条融合 pass，省下来的具体是什么？为什么说它"没有减少任何一次乘法或加法"？这条结论跟 `_lab/minigpt.py` 里算术强度那张表是什么关系？
4. vLLM 的 `cudagraph_capture_sizes` 为什么必须是一个需要手工枚举的整数列表，而不能是"支持任意 batch size"？这跟 CUDA Graph 这项技术本身的什么性质有关？
5. MLC-LLM 的 `LlamaAttention` 和 vLLM 的 `LlamaAttention`，类名一样、结构相似，但导入的 `nn.Module` 来自完全不同的两个地方。分别说出这两处 `import` 分别来自哪里，以及这个差异背后对应的开发迭代循环有什么不同。
6. TensorRT-LLM 的 `auto_deploy/mlir/` 是不是这个引擎的主编译路径？证据是什么？如果只看到"TensorRT-LLM 的代码里有 MLIR 相关目录"就下结论"TensorRT-LLM 是基于 MLIR 的编译器"，这个推理错在哪一步？
7. 如果你现在要给 `02-最小可跑引擎/` 里搭的玩具引擎引入第一层编译技术，会选层 1、层 2 还是层 3？决策二给出的判据，代入"这是一个只在你自己电脑上跑的学习项目"这个场景，落在哪一边？
8. "TVM 建立在 MLIR 之上"这句话对不对？如果不对，正确的关系应该怎么描述？这条误解如果被当真，会导致读者在学习路径上做出什么错误的先后顺序安排？
9. §2.4 说融合的收益在大 batch prefill 场景会"跟着塌下去"。用 `_PLAN.md` §3.1 那张算术强度表里的哪两组数字能支撑这句话？如果一个引擎的实际工作负载几乎全是长上下文、小 batch 的 decode，这条边界对它成不成立？
10. vLLM 的源码里专门写了计时代码去测量"`torch.compile` 编译 + 首次 warmup"这段耗时并打印日志（`vllm:vllm/compilation/monitor.py:82`）。这件事本身能推出什么结论？如果编译时间可以忽略不计，这段计时/日志代码大概率会是什么样子？

**延伸阅读（本库内）**：

- `01-用什么语言写推理引擎-12个真实引擎的证据.md` —— 本篇引用的 12 家语言构成表的完整出处和方法论，本篇讨论的编译技术分布，是这张表之外的另一条正交轴。
- `02-自回归解码为什么是访存瓶颈.md`（`01-地基/`）—— 本篇反复借用的"算子融合省的是访存不是 FLOPs"这条结论的完整推导来源，`_lab/minigpt.py` 那张算术强度表就在这一篇里。
- `06-选型决策表.md` —— 把本篇和前四篇的判据收拢到一张总表里，回答"我该怎么选"这个最终问题。

双链：[[01-用什么语言写推理引擎-12个真实引擎的证据]] [[02-自回归解码为什么是访存瓶颈]] [[06-选型决策表]]
