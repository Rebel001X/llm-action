# OpenAI Triton (kernel编程)

> 用 Python 写 GPU kernel：你只管"按 block(数据块)分活、算逻辑"，编译器替你把 thread 排布、共享内存、向量化、流水线这些苦活自动搞定。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/ai-hardware/CUDA]] [[llm-optimizer/FlashAttention]] [[ai-infra/算力/GPU工作原理]]

---

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | Python-DSL、block 级、自动优化 |
| 1 | 它解决什么问题 | CUDA 太难、kernel 启动开销、算子融合 |
| 2 | GPU 执行模型回顾(地基) | SM、warp、thread、SRAM、HBM |
| 3 | block 级 vs CUDA thread 级 | 编程心智模型对比 |
| 4 | Triton 程序长什么样 | `@triton.jit`、`program_id`、`tl.load/store` |
| 5 | 编译栈与自动优化 | Triton-IR → LLVM → PTX、autotune |
| 6 | 内存层级与 mask | HBM↔SRAM、越界掩码 |
| 7 | 经典案例:向量加 / 矩阵乘 | 思路拆解 |
| 8 | FlashAttention 为什么用它写 | 在线 softmax、tiling、融合 |
| 9 | 数值例子 / 何时用 | 显存、带宽、加速比手算 |
| — | 对照表、常见问题、跳转 | CUDA / CUTLASS / torch.compile |

---

## 0. 一句话锚点

**Triton = 一个内嵌在 Python 里的领域专用语言(DSL)+ 编译器**。你写一个看起来像"对一整块张量做 numpy 运算"的函数，标上 `@triton.jit`，Triton 就把它编译成在 GPU 上跑的真正机器码(PTX/SASS)。

核心卖点一句话：**"CUDA 的性能，numpy 的开发体验"** ——你以 **block(数据块)** 为单位思考，而不是以单个 **thread(线程)** 为单位思考。

```
              CUDA 世界                      Triton 世界
          ┌──────────────┐              ┌──────────────┐
你写的是  │ 每个线程做什么 │              │ 每个数据块做什么│
          │ (thread 级)   │              │ (block 级)    │
          └──────┬───────┘              └──────┬───────┘
                 │ 你手动管:                    │ 编译器自动管:
                 │  线程号↔数据下标             │  线程排布
                 │  共享内存搬运                │  SRAM 分配
                 │  向量化/bank冲突             │  向量化/流水线
                 ▼                            ▼
            难、易错、慢                   快、好写、接近手写 CUDA 性能
```

---

## 1. 地基：它到底解决什么问题

### 1.1 痛点一：手写 CUDA 太难
写高性能 CUDA kernel 要同时操心：线程块尺寸、共享内存(SRAM)手动搬运、bank conflict、寄存器压力、内存合并访问(coalescing)、warp 调度、double buffering 流水线……这些细节决定你能不能吃满硬件，但写对它们需要硬件专家级经验。Triton 把这些**绝大部分自动化**了。

### 1.2 痛点二：framework 的"算子粒度"不够灵活
PyTorch 的每个算子(op)都是一次独立的 kernel 启动。一个 `softmax` 可能拆成 `max → sub → exp → sum → div` 好几个 kernel，每个 kernel 都要：

```
读 HBM(慢) → 算 → 写回 HBM(慢) → 下一个 kernel 再读 HBM(慢) ...
```

中间结果在 **HBM(显存)** 和计算单元之间反复往返，**带宽**被浪费。把这几步**融合(fusion)** 成一个 kernel，中间值只待在片上 **SRAM** 里，就能大幅省带宽。手写融合 kernel 用 CUDA 很痛苦，用 Triton 很自然——这正是 FlashAttention 的核心套路(见第 8 节)。

### 1.3 痛点三：可移植 + 可自动调优
同一份 Triton 代码，换 GPU(不同 SM 数量、SRAM 大小)时，靠 `autotune` 自动搜最优 block 尺寸，而不用你重写。

> 一句话：**Triton 站在 "CUDA(太底层)" 和 "PyTorch 算子(太粗、不可控)" 之间的甜点区。**

---

## 2. 地基回顾：GPU 是怎么执行的(不假设你记得)

要懂 Triton 的"block 级"，先要有 GPU 执行模型的最小图景。详见 [[ai-infra/算力/GPU工作原理]]，这里给最小必要版：

```
 GPU
 ┌─────────────────────────────────────────────┐
 │  SM 0      SM 1      SM 2   ...   SM N         │  ← SM=流多处理器,真正干活的核
 │ ┌──────┐ ┌──────┐ ┌──────┐                    │
 │ │warp  │ │warp  │ │warp  │  warp=32个线程齐步走 │
 │ │warp  │ │warp  │ │...   │  (锁步 SIMT)         │
 │ │SRAM  │ │SRAM  │ │SRAM  │  ← 片上, 极快极小    │
 │ └──────┘ └──────┘ └──────┘   (KB~百KB级)       │
 └───────────────────┬─────────────────────────┘
                     │ 带宽大但延迟高
              ┌──────▼──────┐
              │ HBM / 显存  │  ← 几十 GB, 慢, 一切瓶颈常在这
              └─────────────┘
```

关键名词(原子级)：
- **thread(线程)**：最小执行单元，处理一个或几个数据元素。
- **warp(线程束)**：32 个 thread 锁步执行同一条指令(SIMT)。这是硬件真正的调度粒度。
- **block / CTA(线程块)**：一组线程，共享一块 SRAM，能互相同步。
- **SM(流多处理器)**：物理核，一次跑很多 warp，靠超线程隐藏内存延迟。
- **HBM**：显存，容量大(几十 GB)但慢；**SRAM**：片上，飞快但只有 KB~百 KB。

**所有 GPU 优化的母题**：少碰 HBM，多在 SRAM/寄存器里算。Triton 让你不用手写 SRAM 搬运就能贴近这个目标。

---

## 3. 核心：block 级 vs CUDA thread 级(本文心脏)

### 3.1 CUDA 的心智模型：thread 级(SIMT)
在 CUDA 里你写的是**单个线程**的视角。你必须自己算"我是第几个线程，我负责哪个数据下标"：

```
// CUDA 伪代码: 向量加 c = a + b
int i = blockIdx.x * blockDim.x + threadIdx.x;  // 我是谁? 手算下标
if (i < N)                                       // 边界自己判
    c[i] = a[i] + b[i];                          // 我只管 1 个元素
```

一个数据块由很多线程协作完成，**线程之间怎么分工、怎么搬 SRAM、怎么向量化，全是你的事**。

### 3.2 Triton 的心智模型：block 级(SPMD on blocks)
在 Triton 里你写的是**一整个数据块(BLOCK)** 的视角。一个 Triton "program(程序实例)" 负责一整块数据，块内的多个线程对你**透明**——编译器替你生成：

```python
# Triton 伪代码: 向量加, 一个 program 处理一整块 BLOCK_SIZE 个元素
pid    = tl.program_id(0)                       # 我是第几块?
offs   = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 这一块的下标向量
mask   = offs < N                               # 整块的越界掩码(向量)
a      = tl.load(a_ptr + offs, mask=mask)       # 一次搬一整块进来
b      = tl.load(b_ptr + offs, mask=mask)
tl.store(c_ptr + offs, a + b, mask=mask)        # 一次写回一整块
```

注意区别：`tl.arange(0, BLOCK_SIZE)` 直接生成**一个向量**的下标，`a + b` 是**整块向量**相加。你完全没有写"第 threadIdx 个线程做什么"。

```
   thread 级 (CUDA)            block 级 (Triton)
  ─────────────────          ──────────────────
  for each thread:            for each program(block):
    算我的下标 i                算这一块的下标向量 offs[]
    搬我的 1 个元素              tl.load 一整块 → SRAM(编译器排)
    算我的 1 个结果             整块向量运算
    写我的 1 个元素             tl.store 一整块
    ⟂ SRAM/向量化你来管         ⟂ SRAM/向量化编译器管
```

### 3.3 一句话总结这个范式转换

| 维度 | CUDA(thread 级) | Triton(block 级) |
|---|---|---|
| 你思考的单位 | 单个线程 | 一块数据(tile) |
| 下标计算 | 手算 `blockIdx*dim+threadIdx` | `program_id` + `arange` 向量 |
| 块内线程分工 | **你管** | **编译器管** |
| SRAM 搬运 | **你写** | **编译器生成** |
| 向量化/合并访存 | **你调** | **编译器做** |
| 边界处理 | `if (i<N)` 标量 | `mask` 向量 |
| 心智负担 | 高 | 中(像写 numpy) |

> 关键直觉：**Triton 把"线程"这一层抽象掉了，让你在"数据块"这一层编程**。这就是它既好写、又能快的根本原因——粒度恰好够编译器做强优化，又不至于像 PyTorch 算子那样把融合机会全埋掉。

---

## 4. 一个 Triton 程序由哪些部件构成

```
┌──────────────────────────────────────────────────────────┐
│  @triton.jit                       ← 装饰器:这是一个 kernel │
│  def kernel(x_ptr, y_ptr, ...,     ← 指针(张量首地址)       │
│             N, BLOCK_SIZE: tl.constexpr):  ← constexpr=编译期常量│
│      pid  = tl.program_id(axis=0)  ← 我是第几个 block       │
│      offs = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)      │
│      mask = offs < N               ← 越界掩码               │
│      x = tl.load(x_ptr+offs, mask=mask)   ← HBM→SRAM       │
│      ...   tl 提供 exp/sum/max/dot 等向量与矩阵原语          │
│      tl.store(y_ptr+offs, out, mask=mask) ← SRAM→HBM       │
│                                                            │
│  # 主机侧启动:                                              │
│  grid = (triton.cdiv(N, BLOCK_SIZE),)  ← 要多少个 block     │
│  kernel[grid](x, y, ..., N, BLOCK_SIZE=1024)              │
└──────────────────────────────────────────────────────────┘
```

部件含义(原子级)：
- **`@triton.jit`**：标记函数为 kernel，首次调用时即时(JIT)编译。
- **指针参数 `x_ptr`**：传进来的是张量在显存里的首地址，你用 `ptr + offs` 做地址算术。
- **`tl.constexpr`**：编译期常量(如 `BLOCK_SIZE`)，编译器据此展开循环、定尺寸、做特化。
- **`tl.program_id(axis)`**：当前是第几个 program 实例(对应 CUDA 的 blockIdx)。
- **`tl.arange / tl.load / tl.store`**：生成下标向量、整块读、整块写。
- **`grid`**：启动多少个 program。`triton.cdiv(N, B)` = ⌈N/B⌉ 向上取整。

> API 名称与签名以 [官方文档](https://github.com/openai/triton) 为准，此处讲稳定的机制与心智模型。

---

## 5. 编译栈：Python 怎么变成 GPU 机器码 + 自动优化

```
你的 @triton.jit Python 函数
        │  (1) 追踪/解析 AST
        ▼
   Triton-IR  ← 块级中间表示(还看得见 tile 语义)
        │  (2) 优化 pass:
        │      自动选 SRAM 分块、向量化、
        │      软件流水线(double buffer)、合并访存
        ▼
   Triton-GPU-IR → LLVM-IR
        │  (3) 后端
        ▼
   PTX(NVIDIA) / 类似中间码(AMD)
        │  (4) ptxas / ROCm
        ▼
   SASS / 机器码  ── 在 SM 上执行
```

**自动优化都做了啥(你免费拿到的)**：
1. **线程排布**：把你的 block 级运算映射到 warp/thread，自动满足合并访存(coalescing)。
2. **SRAM 分配与搬运**：`tl.load` 的整块数据自动放进共享内存/寄存器。
3. **向量化**：把多个元素打包成一条向量访存/算术指令。
4. **软件流水线**：在算当前块时预取下一块(double buffering)，隐藏 HBM 延迟。
5. **`@triton.autotune`**：你给一组候选配置(不同 `BLOCK_SIZE`、`num_warps`、`num_stages`)，运行时实测选最快的那组并缓存。

```
@triton.autotune(configs=[
    Config(BLOCK_M=64,  BLOCK_N=64,  num_warps=4),
    Config(BLOCK_M=128, BLOCK_N=64,  num_warps=8),
    Config(BLOCK_M=128, BLOCK_N=128, num_warps=8),
], key=['M','N','K'])         # 输入尺寸变了才重新搜
@triton.jit
def matmul_kernel(...): ...
```

> 直觉：你写"算什么"，编译器决定"怎么在硬件上排"。这把 CUDA 里最难、最吃经验的部分自动化了——但也意味着遇到极端场景，可调的旋钮比裸 CUDA 少。

---

## 6. 内存层级与 mask：为什么处处是 `tl.load/store` 和掩码

GPU 性能 = 尽量把数据从 HBM 搬进 SRAM 后，**多算几次再写回**。Triton 里这体现为显式的 `tl.load`(HBM→片上) 和 `tl.store`(片上→HBM)，中间的运算都发生在片上。

```
 HBM ──tl.load(mask)──▶ SRAM/寄存器 ──向量运算──▶ SRAM ──tl.store(mask)──▶ HBM
  慢                       快(算一切)                          慢
  └────────── 目标: load/store 次数最少, 中间多算 ──────────┘
```

**为什么需要 `mask`**：数据长度 `N` 通常不能被 `BLOCK_SIZE` 整除，最后一块会"越界"。`mask = offs < N` 是一个**布尔向量**，告诉 `load/store`：越界的元素别真去读/写显存(读时给个 `other=0` 占位)。这把 CUDA 里的标量 `if (i<N)` 升级成了**整块的向量化条件**。

```
 N=1000, BLOCK_SIZE=256 → 4 块 (0..255,256..511,512..767,768..1023)
 第4块 offs = 768..1023, 但 N=1000
        offs:   768 ... 999 1000 ... 1023
        mask:    1   ...  1    0   ...   0   ← 后24个不读写
```

---

## 7. 经典案例的"思路"(讲套路，不背 API)

### 7.1 向量加 `c = a + b`(入门 Hello World)
思路：N 个元素切成 ⌈N/B⌉ 块；每个 program 取 `pid`，算 `offs` 向量，`load` 两块、相加、`store` 回去，末块用 `mask`。这是**带宽受限(memory-bound)** 任务的模板——没有复用，瓶颈纯在 HBM 带宽。

### 7.2 融合 softmax(展示"融合"价值)
朴素 PyTorch：`max, sub, exp, sum, div` 是多次 kernel、多次 HBM 往返。Triton 思路：一个 program 负责一行，把**整行**一次性 `load` 进 SRAM，在片上完成 `max→减→exp→sum→除`，**只读一次写一次** HBM。带宽省了好几倍。

### 7.3 矩阵乘 `C = A @ B`(展示"复用 + tiling")
思路：把输出 `C` 切成 `BLOCK_M × BLOCK_N` 的瓦片(tile)，每个 program 算一个瓦片；沿 K 维做循环，每次 `load` 一小条 `A`、一小条 `B` 进 SRAM，用 `tl.dot` 累加到寄存器里的累加器 `acc`，循环结束再写回。这是**计算受限(compute-bound)** 任务，靠 SRAM 复用把 HBM 访问摊薄。`tl.dot` 在新硬件上会映射到 **Tensor Core**。

```
        K
   ┌────────┐         每个 program 负责 C 的一个 tile:
 M │   A    │  沿 K 分块循环:
   └────────┘    load A_tile, B_tile → SRAM
        K            acc += dot(A_tile, B_tile)   ← 留在寄存器
   ┌────────┐    循环完 → store acc → C
 K │   B    │
   └────────┘   关键: A/B 的每块进一次 SRAM, 被复用多次
```

---

## 8. FlashAttention：为什么"用 Triton 写"是天作之合

[[llm-optimizer/FlashAttention]] 的核心痛点：标准注意力要显式造出 $S = QK^\top$ 这个 $N\times N$ 的大矩阵(N=序列长度)，再 softmax 再乘 V。这个 $N\times N$ 矩阵：
- **显存爆炸**：$O(N^2)$ 存中间结果，长序列直接 OOM。
- **带宽爆炸**：$S$ 反复写读 HBM。

FlashAttention 的解法 = **tiling + 在线 softmax + 永不落地 $S$**：把 Q、K、V 切块，逐块累积结果，**整个 $N\times N$ 的 $S$ 从不写进 HBM**，只在 SRAM 里以小块形式存在。

```
 标准注意力                       FlashAttention
 ┌─────────┐                     ┌──────────────────────────┐
 │ S=QKᵀ   │  ← O(N²) 落 HBM     │ for 每块 K,V:              │
 │ P=softmax(S)                  │   在 SRAM 算 Sᵢⱼ=QᵢKⱼᵀ     │
 │ O=P·V   │                     │   在线更新 running max/sum │
 └─────────┘                     │   累加到 Oᵢ (永不落 S)     │
 显存 O(N²), 带宽炸              └──────────────────────────┘
                                 显存 O(N), 带宽省, 更快
```

**为什么必须是 Triton(或手写 CUDA)**：
1. 这是一个**深度融合**的 kernel——matmul、逐块 softmax、再 matmul 全揉进一个 kernel，中间值绝不落 HBM。PyTorch 算子拼不出来(每个 op 都会落地)。
2. 需要**精细的 tiling 与 SRAM 编排**——而 Triton 的 block 级模型恰好让你直接以"块"为单位写 tiling，自动管 SRAM，比手写 CUDA 省一个数量级的工程量。
3. **在线 softmax(online softmax)**：用一个 running 的最大值 $m$ 和归一化和 $\ell$，边读块边修正，避免存全行：对每个新块更新
$$m_{new}=\max(m,\ \tilde m),\quad \ell_{new}=e^{m-m_{new}}\ell+e^{\tilde m-m_{new}}\tilde\ell$$
这套"边走边修正"的逻辑，在 Triton 里就是块循环里的几行向量运算。

> 结论：FlashAttention 是"融合 + tiling + 不落地中间矩阵"的典范，而 Triton 的 block 级编程模型是写这类 kernel 的最佳工具之一——官方 Triton 仓库就自带 FlashAttention 教程实现。

---

## 9. 数值例子 / 手算 / 何时用

### 9.1 该用多少个 block？(grid 手算)
向量加，`N = 1,000,000`，`BLOCK_SIZE = 1024`：
$$\text{num\_blocks}=\left\lceil \frac{N}{\text{BLOCK\_SIZE}}\right\rceil=\left\lceil\frac{1000000}{1024}\right\rceil=977$$
即启动 977 个 program 实例，最后一个用 mask 处理 `1000000 - 976×1024 = 576` 个有效元素，其余 448 个被掩掉。

### 9.2 融合省了多少带宽？(softmax 手算)
设一行 $D=4096$ 个 float32(4 字节)。**朴素多 kernel** 版本(`max,sub,exp,sum,div` 约 5 趟，每趟读+写)粗估每元素往返 HBM 约 8~10 次；**Triton 融合**只需读一次、写一次 ≈ 2 次。
$$\text{带宽节省}\approx \frac{10}{2}=5\times$$
对带宽受限的 softmax，这几乎线性变成 ~5× 加速。每行数据量 $4096\times4=16\text{KB}$，恰能放进 SM 的 SRAM 在片上算完。

### 9.3 FlashAttention 省了多少显存？
$N=8192$ 序列，标准注意力的 $S=QK^\top$ 是 $N\times N$ float16(2 字节)：
$$8192\times8192\times2\ \text{B}=128\ \text{MB / 头 / 样本}$$
多头(如 32 头)× batch(如 8)= $128\text{MB}\times32\times8\approx 32\ \text{GB}$ —— 单这一个中间矩阵就能撑爆一张卡。FlashAttention 让它降到 $O(N)$，**这块显存直接归零**(只留 SRAM 里的小块)。

### 9.4 何时用 Triton(决策清单)

| 场景 | 选 Triton？ | 原因 |
|---|---|---|
| PyTorch 已有高效算子(如普通 matmul) | 否 | 直接用 cuBLAS/cuDNN，别重复造轮子 |
| 多个 element-wise/归约想**融合**省带宽 | ✅ 是 | Triton 融合一把梭，省 HBM 往返 |
| 新型注意力/自定义 attention 变体 | ✅ 是 | FlashAttention 类,框架没有现成 op |
| 量化/稀疏/奇怪 dtype 的自定义算子 | ✅ 是 | 灵活，比手写 CUDA 快得多 |
| 要榨干硬件最后 5%、用满 Tensor Core 特殊指令 | 看情况 | 极限场景手写 CUDA/CUTLASS 仍有上限优势 |
| 只是想加速整段模型、不想写 kernel | 否 | 先试 `torch.compile`(它后端就用 Triton) |

---

## 对照表：Triton 与同类技术

| 维度 | **Triton** | **CUDA C++** | **CUTLASS** | **torch.compile (Inductor)** |
|---|---|---|---|---|
| 抽象层级 | block 级(tile) | thread 级 | 模板化 tile 原语 | 图级,自动生成 |
| 写什么 | Python DSL | C++ | C++ 模板 | 不用写 kernel |
| 谁管 SRAM/向量化 | 编译器 | 你 | 半自动(模板) | 编译器(底层用 Triton) |
| 学习曲线 | 中 | 陡 | 陡 | 几乎零 |
| 灵活度上限 | 高 | 最高 | 高(主攻 GEMM/卷积) | 受图捕获限制 |
| 典型用途 | 自定义/融合 kernel | 极限手调、特殊指令 | 高性能 GEMM 库 | 一键加速整模型 |
| 关系 | — | Triton 编译到它 | 同为高性能路线 | **后端正是 Triton** |

> 记忆锚点：**torch.compile 自动帮你生成 Triton kernel**；你手写 Triton 是为了拿下编译器自动生成不出来的那些**深度融合/新算子**。

---

## 常见问题

| 问题 | 解答 |
|---|---|
| Triton 会取代 CUDA 吗？ | 不会。它降低了写高性能 kernel 的门槛，但极限场景(特殊指令、最后 5% 性能)仍需 CUDA。两者是分工。 |
| block 级真比 thread 级慢吗？ | 通常不慢。编译器生成的线程排布往往接近甚至追平手写 CUDA，开发量却少一个数量级。 |
| 为什么处处要 `mask`？ | 因为 `N` 很少能被 `BLOCK_SIZE` 整除,末块会越界,mask 把越界元素从读写中屏蔽。 |
| `BLOCK_SIZE` 怎么选？ | 通常取 2 的幂(128/256/512/1024);别死调,交给 `@triton.autotune` 自动搜。 |
| Triton 和 FlashAttention 是一回事吗？ | 不是。FlashAttention 是**算法**;Triton 是**写它的工具**之一(也可用 CUDA 写)。 |
| 只支持 NVIDIA 吗？ | 早期主攻 NVIDIA,后续向 AMD 等后端扩展;具体支持以官方文档为准。 |
| 我该先学 Triton 还是 CUDA？ | 先有 GPU 执行模型直觉([[ai-infra/算力/GPU工作原理]]),再用 Triton 上手写 kernel 性价比最高;深究底层再补 CUDA。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航总入口
- [[ai-infra/ai-hardware/CUDA]] — thread 级编程模型、SM/warp/SRAM 的底层细节
- [[llm-optimizer/FlashAttention]] — Triton 最经典的应用:tiling + 在线 softmax + 不落地中间矩阵
- [[ai-infra/算力/GPU工作原理]] — SM、warp、HBM/SRAM 内存层级、带宽 vs 计算瓶颈

### 参考资料
- OpenAI Triton 官方仓库与教程(含 FlashAttention 实现)：https://github.com/openai/triton
- OpenAI Triton 入门教程：https://zhuanlan.zhihu.com/p/684473453

> 一句话收尾：**Triton 让你"以数据块为单位用 Python 写 GPU kernel"，把线程排布/SRAM/向量化/流水线交给编译器——这恰好是写 FlashAttention 这类深度融合算子的甜点工具。** 具体 API/版本以官方文档为准。
