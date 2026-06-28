# PyTorch 源码解读

> 从「一行 Python 调用」一路追到「C++/CUDA 内核」，看清 PyTorch 这台机器内部是怎么转的：分发(dispatch)、自动微分(autograd)、算子注册、内存与设备管理。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/pytorch/README]] [[llm-train/README]] [[llm-train/pytorch/distribution/README]]

## 阅读地图

| 节 | 主题 | 你将得到什么 | 何时回看 |
|----|------|-------------|---------|
| 0 | 一句话锚点 | 一行 `a+b` 背后发生了什么 | 迷路时 |
| 1 | 地基：为什么要读源码 | Python 壳 + C++ 核的分工 | 入门 |
| 2 | 仓库目录地图 | 各 `torch/`、`aten/`、`c10/` 是干嘛的 | 找代码时 |
| 3 | 一次调用的完整旅程 | Python→C++→kernel 调用链 | 想知道"怎么走到内核" |
| 4 | Tensor 与 Storage | data_ptr / stride / view 的底层 | 看懂 view/contiguous |
| 5 | Dispatcher 分发机制 | 同一算子如何选 CPU/CUDA/autograd | 理解多设备/多后端 |
| 6 | autograd 引擎 | 计算图、grad_fn、反向调度 | 调试梯度 |
| 7 | 算子注册与 ATen | native_functions.yaml 怎么生成代码 | 想加/改算子 |
| 8 | 编译与调试入口 | 怎么 build、怎么打断点跟进去 | 真要动手读源码 |
| 9 | 常见问题 | 踩坑与术语速查 | Debug 时 |

---

## 0. 一句话锚点

你写下 `c = a + b`，PyTorch 内部其实走了一条**五层管道**：

```
Python:  a + b
   │  (1) Python 调 torch._C 绑定的 C++ 函数
   ▼
C++ API: at::add(a, b)
   │  (2) Dispatcher 按「设备 + 是否需梯度」选实现
   ▼
Autograd 层: 若 requires_grad → 记一个 AddBackward 节点进图
   │  (3) 再分发到真正算数据的内核
   ▼
ATen kernel: add_kernel (CPU / CUDA)
   │  (4) 在 Storage 的连续内存上逐元素相加
   ▼
返回新 Tensor c（带 grad_fn=AddBackward）
```

**记住这张图**：本文其余部分就是把这五层逐层拆开。读 PyTorch 源码迷路时，先问自己「我现在在哪一层」。

---

## 1. 地基：为什么读源码、Python 与 C++ 怎么分工

### 1.1 它解决什么问题

只会用 `loss.backward()` 时，很多事是「黑盒」：

- 为什么 `x.view()` 不复制数据、`x.reshape()` 有时复制？
- 自定义算子、自定义 autograd Function 时该往哪里挂钩子？
- 性能分析看到一堆 `aten::add`、`AddBackward0`，它们是谁？
- 报错栈里出现 `c10::Error`、`Dispatcher`、`TensorIterator`，这些是什么层？

读源码就是把这些黑盒打开，知道**改哪里、为什么慢、为什么报这个错**。

### 1.2 Python 是「壳」，C++ 是「核」

PyTorch 是一个**多语言分层**项目：

```
┌──────────────────────────── Python 层 (torch/) ────────────────────────────┐
│  torch.nn / torch.optim / torch.Tensor 的 Python 方法 / autograd.Function   │
│  —— 用户写的、调试方便、灵活，但不直接算数                                    │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                 │  pybind11 / torch._C 绑定
┌────────────────────────────────▼─────────────────────────── C++ 层 ──────────┐
│  torch/csrc/   Python↔C++ 胶水、autograd 引擎、JIT、分布式                    │
│  aten/         ATen：张量库 + 所有算子实现 (CPU/CUDA kernel)                  │
│  c10/          Core：最底层抽象（Tensor/Storage/Device/Dispatcher/类型）     │
└──────────────────────────────────────────────────────────────────────────────┘
```

一句话分工：**Python 决定「做什么」，C++/CUDA 决定「怎么快速地做」**。绝大多数算子的真正计算都在 C++/CUDA，Python 只是一层方便人用的接口。

> 注意：下文出现的目录名、文件名是 PyTorch 长期稳定的组织方式；**具体文件路径、函数签名会随版本变动，以你本地 clone 的源码与官方文档为准**。

---

## 2. 仓库目录地图：各文件夹是干嘛的

第一次打开 PyTorch 仓库会被几百个目录吓到。其实记住**四个核心目录**就够入门：

```
pytorch/
├── torch/                 ← Python 包（你 import 的就是它）
│   ├── nn/                  神经网络层、损失函数
│   ├── optim/               优化器 (SGD/Adam...)
│   ├── autograd/            自动微分的 Python 侧接口
│   ├── distributed/         分布式 (DDP/FSDP/通信)
│   └── csrc/              ← C++ 源码！(C source) Python↔C++ 的桥
│       ├── autograd/        autograd 引擎 (反向调度) 的 C++ 实现
│       ├── jit/             TorchScript / 图编译
│       └── api/             C++ 前端
│
├── aten/                  ← "A Tensor library"，张量与算子的家
│   └── src/ATen/
│       ├── native/          ← 算子的真实现 (add/matmul/conv...) ★最常看
│       │   ├── cpu/          CPU kernel
│       │   └── cuda/         CUDA kernel
│       └── native_functions.yaml  ← 算子"声明清单"，代码生成的源头 ★
│
├── c10/                   ← "Core / Caffe2 tensor lib"，最底层抽象
│   ├── core/                TensorImpl / StorageImpl / Device / Dispatcher
│   └── cuda/                CUDA 设备/流/缓存分配器抽象
│
└── tools/                 代码生成脚本（从 yaml 生成大量 C++ 绑定）
```

记忆口诀（自底向上）：

```
c10   = 地基（最核心的类型与抽象，谁都依赖它）
ATen  = 算子库（站在 c10 上，实现所有数学运算）
csrc  = 桥（把 ATen 接到 Python，并实现 autograd 引擎/JIT/分布式）
torch = 门面（你天天 import 的 Python API）
```

**找代码的经验法则**：

- 想看「某个算子怎么算的」→ `aten/src/ATen/native/`（CPU 看 `cpu/`，GPU 看 `cuda/`）。
- 想看「这个算子叫什么、有几个重载、什么 dispatch key」→ `native_functions.yaml`。
- 想看「Tensor 内部长什么样」→ `c10/core/TensorImpl.h`。
- 想看「反向是怎么调度的」→ `torch/csrc/autograd/`。

---

## 3. 一次调用的完整旅程（调用链）

以 `c = a + b`（两个 `requires_grad=True` 的张量）为例，逐站点追：

```
① Python:  c = a + b
            └─ Tensor.__add__ → torch.add → torch._C._TensorBase.add
                                              （pybind 绑定，进入 C++）
                                   │
② 进入 C++:  at::add(a, b)
                                   │   交给 Dispatcher 决定调谁
③ Dispatcher 查表（见 §5）:
     当前 Tensor 的 dispatch key 集合 = {Autograd, CUDA}
     先命中优先级最高的 Autograd key
                                   │
④ Autograd kernel (VariableType):
     - 调用底层真正算数的 add（见 ⑤）得到结果
     - 因为 requires_grad，新建一个 AddBackward 节点，
       把 a、b 的"反向边"连进计算图，挂到 c.grad_fn
                                   │  重新分发，这次跳过 Autograd
⑤ 真正的 add kernel:  CUDA 后端的 add_kernel
     - 通过 TensorIterator 处理广播/类型提升/步长
     - 启动 CUDA kernel，在 Storage 连续内存上逐元素相加
                                   │
⑥ 返回 Tensor c：data 是相加结果，grad_fn=AddBackward0
```

**关键洞察**：Dispatcher 会**分发两次**。第一次命中 Autograd 层（负责建图），Autograd 层处理完「记录反向」后，**再分发一次**到真正算数据的 CPU/CUDA 内核。这就是「前向时顺手把反向图建好」的机制所在。

> 想亲眼看这条链：用 PyTorch Profiler 或 `torch.autograd.profiler`，你会看到 `aten::add`（前向算子）；反向时会看到 `AddBackward0`（反向节点）。

---

## 4. Tensor 与 Storage：view 为什么不复制

### 4.1 「视图」与「数据」是分离的

一个 `Tensor` 不直接持有数据，而是持有一个**视图描述**，指向底层的 `Storage`（一维连续内存）：

```
        Tensor (TensorImpl)                Storage (StorageImpl)
        ┌───────────────────┐             ┌──────────────────────────────┐
        │ sizes   [2,3]      │   data_ptr  │  一维连续内存                 │
        │ strides [3,1]      │ ──────────► │ [a00 a01 a02 a10 a11 a12 ...]│
        │ storage_offset 0   │             └──────────────────────────────┘
        │ dtype  float32     │
        │ device cuda:0      │
        │ → storage ─────────┼─────────────┘
        └───────────────────┘
```

- **sizes**：每个维度的长度（逻辑形状）。
- **strides**：在每个维度上「走一步」要跳过多少个元素。
- **storage_offset**：从 Storage 哪个位置开始。

逻辑下标 $(i,j)$ 映射到内存线性地址：

$$\text{addr} = \text{offset} + i\cdot\text{stride}_0 + j\cdot\text{stride}_1$$

### 4.2 为什么 view/transpose 不复制数据

`view`、`transpose`、`[:, ::2]` 这些操作**只改 sizes/strides/offset，不动 Storage**——同一块内存，换个「读法」：

```
原始 x  shape (2,3) strides (3,1)        内存: [0 1 2 3 4 5]

x.transpose(0,1)  shape (3,2) strides (1,3)   ← 内存没变！只是 strides 交换
   读 (0,0)=offset0 → 0
   读 (0,1)=0+1*3   → 3   （跳 3 个）
```

代价：转置后**内存不再连续**（按行读会跳着读）。某些算子要求连续内存，就需要 `x.contiguous()`——**这一步才真复制**，把数据按新顺序搬成连续块。

```
x.transpose(0,1)          → 非连续视图（共享内存，零拷贝）
x.transpose(0,1).contiguous() → 触发一次真实拷贝，得到新 Storage
reshape = 能 view 就 view，不能就 contiguous 再 view（可能复制）
```

> 数值直觉：`view` 几乎零成本（只改元数据）；`contiguous`/`reshape` 在不连续时是 $O(n)$ 拷贝。看源码时遇到 `is_contiguous()`、`as_strided`、`TensorIterator`，都是围绕这套 stride 机制转的。

---

## 5. Dispatcher：同一个 add，怎么选到对的实现

### 5.1 它解决什么问题

`add` 这一个名字，背后要支持：CPU、CUDA、是否 autograd、是否 autocast(混合精度)、是否量化、是否稀疏…… 如果用 `if/else` 判设备，代码会爆炸。PyTorch 用 **Dispatcher（分发器）** 解耦：算子是一张「**操作名 × DispatchKey → 函数指针**」的表。

### 5.2 DispatchKey 与查表

每个 Tensor 带一组 **DispatchKey**（如 `CPU`/`CUDA`/`Autograd`/`Autocast`），调用算子时，Dispatcher 取出参数张量的 key 集合，按**固定优先级**选最高的那个 key 对应的 kernel：

```
            算子注册表 (operator × key → kernel)

 op="aten::add"
   ┌─────────────┬──────────────────────────────┐
   │ Autograd    │ VariableType::add  (建反向图) │ ← 优先级高
   │ Autocast    │ 自动半精度包装               │
   │ CUDA        │ add_cuda_kernel              │
   │ CPU         │ add_cpu_kernel               │ ← 优先级低
   └─────────────┴──────────────────────────────┘

 调用 add(a,b)，a 的 keys={Autograd, CUDA}
   → 先选 Autograd（建图）→ 该 kernel 内"屏蔽 Autograd key 后再分发"
   → 这次选 CUDA → add_cuda_kernel 真正算
```

**「重新分发(redispatch)」** 就是上面的箭头：每层 kernel 干完自己的活（建图/转精度），把 Autograd/Autocast 这类 key 临时屏蔽掉，再丢回 Dispatcher，直到落到真正算数据的后端 kernel。这是一条**洋葱式的分层管道**，每层只负责一件事。

```
请求 ─► [Autocast] ─► [Autograd] ─► [后端 CPU/CUDA] ─► 算完
         转精度        建反向图        真正计算
       (每层做完自己的事就 redispatch 给下一层)
```

### 5.3 为什么这套设计好

- **可扩展**：加一个新后端（如某 NPU、新量化方案），只需注册自己 key 的 kernel，不动核心代码。
- **正交**：autograd、autocast、量化各是一层，互相独立组合。
- 理解了它，就理解了 PyTorch 「能在 CPU/GPU/各种加速器/各种精度上跑同一份 Python 代码」的根。

> 国产/异构后端（如昇腾、各类加速卡）接入 PyTorch，本质就是「注册自己 DispatchKey 的一套 kernel」。参见 [[ai-infra/算力/昇腾NPU]]。

---

## 6. autograd 引擎：计算图怎么建、反向怎么调度

### 6.1 前向时悄悄建图

第 3、5 节已经说到：当张量 `requires_grad=True`，前向算子在 Autograd 层会**新建一个反向节点**（如 `AddBackward0`、`MulBackward0`），并记录输入张量的「反向边」。这些节点连成一张**有向无环图(DAG)**：

```
前向:  d = (a * b) + c          反向图（自动建好）:
                                   d.grad_fn = AddBackward
   a ─┐                                  │
       ├─[MulBackward]──┐        AddBackward ─┬─► (传给 c)
   b ─┘                 │                     └─► MulBackward ─┬─► (传给 a)
   c ───────────────────┴──► d                                └─► (传给 b)
                                  每个节点知道：怎么把"上游梯度"变成"对各输入的梯度"
```

- 叶子张量（参数、输入）：`grad_fn` 为 `None`，反向结果累加到它的 `.grad`。
- 中间张量：`grad_fn` 指向产生它的反向节点。

### 6.2 backward() 时反向调度

`loss.backward()` 触发 C++ 里的 **autograd 引擎**：

```
loss.backward()
   │
   ▼  从 loss 节点出发，按计算图的"反拓扑序"调度
┌──────────────── Engine ────────────────┐
│ 1. 把起点梯度设为 1（dL/dL=1）           │
│ 2. 维护一个就绪队列，谁的上游梯度齐了就跑 │
│ 3. 每个节点：用保存的前向中间量          │
│    计算"对各输入的梯度"（链式法则一格）   │
│ 4. 把梯度沿反向边传给上游节点            │
│ 5. 到达叶子 → 累加进 .grad（注意是累加！）│
└─────────────────────────────────────────┘
   多个反向节点之间可并行（多线程 worker）
```

几个**关键源码概念**（看 `torch/csrc/autograd/`）：

| 概念 | 是什么 | 对应你写的代码 |
|------|--------|---------------|
| `Node` / `Function` | 反向图的一个节点（一个反向算子） | 每个前向算子对应一个 `XxxBackward` |
| `Edge` | 反向边（指向上游 Node 的第几个输出） | 连接梯度流向 |
| `Engine` | 反向调度引擎（拓扑序+多线程） | `backward()` 背后那只手 |
| `AccumulateGrad` | 叶子节点专用，把梯度累加进 `.grad` | 为啥要 `zero_grad()` 的根 |
| `saved_tensors` | 前向保存、反向要用的中间量 | 占显存的主因之一 |

### 6.3 自定义 autograd.Function 时你在做什么

当你写 `class MyFn(torch.autograd.Function)` 并实现 `forward`/`backward`，本质就是**手动往这张图里塞一个自定义反向节点**：`forward` 用 `ctx.save_for_backward` 存中间量，`backward` 收到上游梯度、返回对各输入的梯度。理解了 §6.1–6.2，就知道这两个函数分别对应「建节点」和「节点的反向计算」。

> 显存视角：`saved_tensors` 是激活显存的主要来源；梯度检查点(checkpointing)就是「反向时重算前向、少存 saved_tensors」来省显存——与 [[llm-train/pytorch/distribution/README]] 里的大模型训练优化直接相关。

---

## 7. 算子注册与 ATen：代码是「生成」出来的

### 7.1 native_functions.yaml 是清单

PyTorch 有上千个算子，不可能手写每一处 Python 绑定、autograd 包装、dispatch 注册。它用**代码生成**：核心是一份声明清单 `aten/src/ATen/native/native_functions.yaml`，里面每个算子声明大致长这样（示意，**确切字段以源码为准**）：

```
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  dispatch:
    CPU:  add_cpu          # CPU 用哪个 C++ 函数实现
    CUDA: add_cuda         # CUDA 用哪个
  # （还可声明是否支持 autograd、别名信息等）
```

构建时，`tools/` 下的生成脚本读这份 yaml，**自动产出**：

```
native_functions.yaml ──► 代码生成器 (tools/) ──►
   ├─ Python 绑定代码 (torch._C.add 怎么调到 at::add)
   ├─ Dispatcher 注册代码 (把 add_cpu/add_cuda 挂到对应 key)
   ├─ autograd 包装 (VariableType::add，自动建反向节点)
   └─ 形状/类型推断等样板
```

所以你在仓库里**搜不到**很多「绑定函数」的手写源码——它们是 build 时生成的，藏在 `build/` 目录。

### 7.2 加/改一个算子的心智模型

```
想加一个新算子？典型流程（机制层面，细节以官方贡献指南为准）：
  1. 在 native_functions.yaml 声明 func 签名 + dispatch
  2. 在 aten/src/ATen/native/ 写 CPU/CUDA 实现
  3. 若要可微 → 在 derivatives.yaml 声明它的反向公式
  4. 重新 build → 生成器自动补齐绑定/注册/autograd 包装
```

这就解释了一个常见困惑：「我改了 C++ 实现，为什么 Python 调用没变化」——多半是**没重新 build / 没触发代码生成**。

---

## 8. 编译与调试：怎么真正读进去

### 8.1 读源码的两种姿势

```
姿势 A：只读不跑（轻量）
  - clone 仓库，靠 IDE 跳转：从 torch/ 的 Python 往下，
    遇到 C++ 绑定 → 跳到 aten/native 看 kernel
  - 配合 native_functions.yaml 反查"这个算子注册在哪"

姿势 B：从源码编译 + 断点调试（重量，能看到运行时真相）
  - 从源码 build（debug 模式带符号），用 gdb/lldb 跟 C++ 栈
  - Python 侧断点 → 进入 C++ 后用 gdb 接管
```

### 8.2 不用编译也能"看见"内部的工具

很多时候不必啃 C++ 就能验证你对机制的理解：

| 工具 | 看到什么 | 对应本文哪节 |
|------|---------|-------------|
| `torch.autograd.profiler` / Profiler | 实际执行了哪些 `aten::*` 算子、反向节点 | §3 §6 |
| `Tensor.stride()` / `.storage()` / `.data_ptr()` | view 是否共享内存 | §4 |
| `Tensor.grad_fn` | 这个张量挂在哪个反向节点 | §6 |
| `torch.fx` / `make_fx` | 把前向 trace 成一张可读的图 | §3 §6 |
| `TORCH_SHOW_DISPATCH_TRACE`(调试用环境变量) | 一次调用经过了哪些 dispatch key | §5 |

> 环境变量、构建脚本名称随版本变化较大，**具体编译命令、调试开关请以官方 `CONTRIBUTING.md`、`setup.py`/`CMake` 文档为准**，不要照抄记忆里的旧参数。

### 8.3 一张「我在哪一层」的速查图

```
报错/疑问出现这些字样 → 你在这一层 → 去这里看
 ─────────────────────────────────────────────────────
 KeyError/Python 栈        Python 层      torch/*.py
 pybind / torch._C         绑定层         (生成代码，看 yaml)
 Dispatcher / DispatchKey  分发层         c10/core/Dispatcher
 AddBackward / Engine      autograd 层    torch/csrc/autograd/
 TensorIterator / kernel   ATen 算子层    aten/src/ATen/native/
 StorageImpl / data_ptr    c10 核心层     c10/core/
 c10::Error / CUDA error   底层/驱动      看设备、驱动、内存
```

---

## 9. 常见问题

| 现象 / 疑问 | 原因（机制层面） | 处理思路 |
|------------|-----------------|---------|
| `x.view()` 报错说不连续 | transpose/切片后内存非连续，view 要求连续 | 先 `.contiguous()` 或改用 `.reshape()` |
| 改了 C++ 实现 Python 没变 | 没重新 build、代码生成没跑 | 重新编译；确认改的是被注册的那份 |
| `grad_fn` 是 None | 张量 `requires_grad=False` 或在 `no_grad()` 里建图 | 检查是否需梯度、是否在推理上下文 |
| 反向特别耗显存 | `saved_tensors`（激活）存太多 | 用梯度检查点、减小 batch、看 §6.3 |
| Profiler 里满屏 `aten::*` | 那是 ATen 算子的真名（§7） | 对照 native_functions.yaml 反查 |
| 同名算子在不同设备结果路径不同 | Dispatcher 按 DispatchKey 选了不同 kernel | 理解 §5 的分发优先级 |
| 找不到某绑定函数源码 | 它是 build 时生成的，不在源码树里 | 去 `build/` 找生成产物，或读 yaml |
| 新后端(NPU/加速卡)接入 | 缺对应 DispatchKey 的 kernel 注册 | 注册自定义 key 的算子集（§5.3） |
| C++ 栈里 `TensorIterator` | 处理广播/类型提升/步长的统一框架 | 它是逐元素算子的"调度器" |

**术语速查**：`c10`=最底层核心库；`ATen`=张量与算子库；`native`=算子真实现目录；`Dispatcher`=按 key 选 kernel 的分发器；`DispatchKey`=CPU/CUDA/Autograd 等标签；`grad_fn`=反向节点；`Storage`=一维连续内存；`stride`=维度步长；`TensorIterator`=逐元素算子的广播/步长处理框架。

---

## 🔗 跳转链接

- 知识地图总览：[[00-知识地图]]
- 框架基础（张量/autograd/nn 的"怎么用"）：[[ai-framework/pytorch/README]]
- 训练流程（前向→反向→更新五步）：[[llm-train/README]]
- 分布式训练（DDP 的 all-reduce、FSDP 分片，建立在本文的 autograd/dispatch 之上）：[[llm-train/pytorch/distribution/README]]
- 异构后端接入（注册自己 DispatchKey 的一例）：[[ai-infra/算力/昇腾NPU]]

> 外部参考（以官方为准，版本会变动）：PyTorch 官方源码仓库 github.com/pytorch/pytorch；贡献/构建指南见仓库内 `CONTRIBUTING.md`；ezyang 的「PyTorch internals」博文与系列讲座是经典的源码导读。
> 中文导读：PyTorch 源码解读系列 https://zhuanlan.zhihu.com/p/328674159 ；PyTorch 训练推理（gemfield 专栏）https://www.zhihu.com/column/gemfield ；PyTorch 分布式 https://juejin.cn/post/7026144707591815175
