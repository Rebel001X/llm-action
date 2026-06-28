# Triton 语言与编译器

> 用 Python 写 GPU kernel：你只管"按 block 切数据"，编译器自动处理共享内存、合并访存、寄存器分配，最后编译到 PTX。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/openai-triton/README]] [[ai-infra/ai-hardware/CUDA]] [[llm-optimizer/FlashAttention]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | block 编程 / Python DSL |
| 1 | GPU 地基：SM/线程/显存层级 | warp / SMEM / coalescing |
| 2 | Triton 的核心抽象：program 与 block | `program_id` / `tl.arange` |
| 3 | 它替你自动做的三件事 | 调度 / 共享内存 / 合并访存 |
| 4 | 编译流水线：Python → PTX | AST → Triton IR → LLVM IR → PTX |
| 5 | 第一个 kernel：向量加法 | mask / `tl.load` / `tl.store` |
| 6 | 进阶：矩阵乘法 tiling | 分块 / 累加器 |
| 7 | 写 FlashAttention 类 kernel | online softmax / 不落盘 |
| 8 | Triton vs CUDA 全面对比 | 抽象层 / 控制力 |
| 9 | 生态与工具链 | autotune / 集成 |
| 数值例子 | 带宽与手算 | GB/s / FLOPs |
| FAQ | 高频疑问 | — |

## 0. 一句话锚点

**Triton = 一个嵌入 Python 的领域专用语言（DSL）+ 编译器**，让你以"**一个 program 处理一个数据 block**"的视角写 GPU kernel。你写的是接近 NumPy 的张量级代码，编译器负责把它降低（lower）成高效的 GPU 机器码（NVIDIA 上是 PTX）。它由 OpenAI 开源，现已是 PyTorch 2.x `torch.compile` 的默认 GPU 代码生成后端之一。

```
   你写的（Python，张量级）            编译器替你做的（底层）
 ┌───────────────────────────┐     ┌──────────────────────────────┐
 │ pid = tl.program_id(0)    │     │ 线程↔数据映射、warp 调度       │
 │ x = tl.load(ptr+offs,mask)│ ==> │ 合并访存、共享内存分配          │
 │ y = x * 2                 │     │ 寄存器分配、指令选择            │
 │ tl.store(out+offs,y,mask) │     │ 生成 PTX → SASS                │
 └───────────────────────────┘     └──────────────────────────────┘
```

> 直觉：CUDA 让你管"每个线程做什么"；Triton 让你管"每个 block 做什么"，线程内部的事交给编译器。

## 1. 地基/前置：GPU 是怎么算的

要理解 Triton "替你做了什么"，得先有 GPU 硬件的原子图像。以 NVIDIA 为例（规格"约/以官方为准"）：

```
GPU
 ├─ SM (Streaming Multiprocessor) ×几十~上百个
 │   ├─ CUDA Core / Tensor Core   ← 真正做乘加
 │   ├─ 寄存器堆 (Registers)       ← 最快，每线程私有
 │   ├─ 共享内存 SMEM (~几十~228KB/SM，可配置) ← block 内共享，片上，快
 │   └─ Warp 调度器                ← 一次发射 32 线程(=1 warp)
 └─ 全局显存 Global Memory (HBM, 几十 GB) ← 最大但最慢
```

**两个必须刻进脑子的层级关系：**

1. **执行层级**：`thread`（最小执行单元）→ `warp`（32 个 thread 锁步执行，SIMT）→ `block / CTA`（一组 thread，共享 SMEM）→ `grid`（所有 block）。
2. **存储层级（越往下越慢越大）**：寄存器 → 共享内存(SMEM) → L2 缓存 → 全局显存(HBM)。

```
速度  容量
快 小  寄存器     ~几十 KB/SM     ← ns 级
 │     共享内存    ~百 KB/SM       ← 片上，~几十 cycle
 │     L2 Cache   ~几十 MB/GPU
慢 大  HBM 显存    ~几十 GB        ← 几百 cycle，瓶颈常在这
```

**为什么这决定性能？** GPU 算力（FLOPs）远超访存带宽（GB/s）。绝大多数 LLM kernel 是**访存受限（memory-bound）**的——慢在搬数据，不慢在算。所以优化的核心就两条：

- **合并访存（coalescing）**：让一个 warp 的 32 个线程访问**连续**地址，硬件合并成一次大事务，而不是 32 次小事务。
- **复用数据**：把从 HBM 读来的数据放进 SMEM/寄存器，反复用，少回 HBM。

Triton 的全部价值，就是**让你不用手写这两条，编译器自动做**。

## 2. 核心抽象：program 与 block

CUDA 的世界观是"我是第几个线程"。Triton 的世界观是"**我是第几个 program，负责哪一块数据**"。

- 一个 **program**（也叫 kernel instance）≈ CUDA 里的一个 block，但你**看不到也不写线程**。
- `tl.program_id(axis)`：拿到当前 program 在 grid 中的编号。
- `tl.arange(0, BLOCK)`：生成 `[0,1,...,BLOCK-1]` 这个向量，用来算这一块的偏移。
- 你操作的是**张量（向量/矩阵）**，不是标量。`x = tl.load(...)` 一次加载一整块。

```
数据(长度 N=10):  [ a b c d e f g h i j ]
BLOCK=4, grid = ceil(10/4)=3 个 program
                 ┌─────┬─────┬─────┐
   program_id =  │  0  │  1  │  2  │
                 │ abcd│ efgh│ ij  │  ← program 2 越界，靠 mask 屏蔽
                 └─────┴─────┴─────┘
   每个 program 内部的 4 个元素 → 编译器分给若干线程，你不管
```

关键心智转变：**SPMD（单程序多数据）的粒度从"线程"上升到了"block"**。你写一段代码，它被复制到每个 program 上跑，靠 `program_id` 区分各自负责的数据段。

## 3. 编译器替你自动做的三件事

这是 Triton 相比 CUDA 最大的卖点。你写张量级代码，下面这些**全部自动完成**：

```
你写： x = tl.load(ptr + offs, mask=...)   ← 一句话
       │
       ▼ 编译器分析
 ┌─────────────────────────────────────────────┐
 │ ① 线程↔数据映射：把 BLOCK 个元素分给 warp     │
 │    并保证相邻线程访相邻地址 → 合并访存          │
 │ ② 共享内存：发现 tiling 复用 → 自动分配 SMEM、  │
 │    自动插入同步、做 double-buffer / 软流水      │
 │ ③ 寄存器 & 指令：寄存器分配、向量化、用 Tensor  │
 │    Core（mma 指令）、调度隐藏访存延迟           │
 └─────────────────────────────────────────────┘
```

| 手写 CUDA 你必须操心 | Triton 里谁来做 |
|---|---|
| 线程索引算偏移 `blockIdx*blockDim+threadIdx` | 你写 block 级，编译器映射 |
| `__shared__` 声明、大小、bank conflict | 编译器自动分配/优化 |
| `__syncthreads()` 放哪 | 编译器自动插入 |
| 访存是否合并 | 编译器布局保证 |
| 寄存器用量 / occupancy 调优 | 编译器 + autotune |
| 用 Tensor Core 写 wmma/mma | `tl.dot` 自动降低到 mma |

> 边界：Triton 自动做"会做的部分"。极致的、奇形怪状的手工优化（特殊 swizzle、warp 专用化），CUDA 仍有更细的控制力——见第 8 节权衡。

## 4. 编译流水线：Python 怎么变成 PTX

Triton 不是解释执行，它是**真编译**。一个 `@triton.jit` 函数第一次被调用时触发即时编译（JIT），按 `(参数类型, BLOCK 等常量)` 缓存结果，后续直接复用。

```
@triton.jit 的 Python 函数
        │  (1) 抽取 Python AST
        ▼
   Triton-IR        ← 块级、张量级的中间表示（基于 MLIR）
        │  (2) 优化 pass：layout 推导、合并访存分析、SMEM 流水
        ▼
   Triton-GPU-IR    ← 绑定到具体 GPU 的线程布局
        │  (3) 下降 lowering
        ▼
   LLVM-IR
        │  (4) LLVM 后端 + NVPTX
        ▼
   PTX              ← NVIDIA 的"虚拟汇编"
        │  (5) ptxas 汇编（运行时/驱动）
        ▼
   SASS             ← 真正在 SM 上跑的机器码
```

- **PTX**（Parallel Thread Execution）：NVIDIA 的中间汇编，跨 GPU 架构稳定，由 `ptxas` 再编成具体架构的 SASS。
- **MLIR**：现代编译器基础设施，Triton 用它管理多层 IR 和优化 pass（具体 pass 列表"以官方源码为准"）。
- AMD GPU 后端走的是 LLVM AMDGPU → GCN/RDNA ISA（不经 PTX）。

> 一句话：**Triton 把"语言"和"为某代 GPU 优化"解耦**——你写一份 Python，编译器按目标硬件生成最优代码。

## 5. 第一个 kernel：向量加法（带逐行解释）

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid   = tl.program_id(axis=0)          # 我是第几个 block
    start = pid * BLOCK                     # 我负责的数据起点
    offs  = start + tl.arange(0, BLOCK)     # 这一块的所有下标（向量）
    mask  = offs < N                        # 越界保护：尾块不满 BLOCK
    x = tl.load(x_ptr + offs, mask=mask)    # 一次读一整块；越界处不读
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)  # 一次写一整块

def add(x, y):
    out  = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK']),)  # 几个 block
    add_kernel[grid](x, y, out, x.numel(), BLOCK=1024)
    return out
```

**为什么 `mask` 是关键？** 数据长度往往不是 `BLOCK` 的整数倍，最后一个 program 会"越界"。`mask=offs<N` 让 `tl.load`/`tl.store` 对越界元素**跳过**（读到默认值、不写回），避免非法访存。这就是 Triton 的"边界处理"哲学——不用写 `if`，用布尔向量遮罩。

```
N=10, BLOCK=4, pid=2:
  offs = 8 + [0,1,2,3] = [8, 9, 10, 11]
  mask =                 [T, T,  F,  F]   ← 10,11 越界被屏蔽
  load → [x8, x9, ?, ?]，store 只写回 8,9
```

## 6. 进阶：矩阵乘法的 tiling（数据复用思想）

$C = A \times B$，$A$ 是 $M\times K$，$B$ 是 $K\times N$。朴素做法每算一个 $C$ 元素都要从 HBM 读一行一列，复用率极低。**tiling（分块）**把输出切成 `BLOCK_M × BLOCK_N` 的小块，沿 $K$ 维循环累加：

```
        K
   ┌──────────┐         ┌────┐
 M │   A 行块  │   ×   K │ B  │ 列块
   └──────────┘         │块  │
                        └────┘
   每步取 A 的一个 (BM×BK) 子块、B 的一个 (BK×BN) 子块
   → 放进 SMEM/寄存器 → tl.dot 累加到 acc → 沿 K 滑动
```

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    acc = tl.zeros((BM, BN), dtype=tl.float32)   # 累加器在寄存器
    for k in range(0, K, BK):
        a = tl.load(...)   # (BM, BK) 子块 —— 编译器自动放 SMEM/做流水
        b = tl.load(...)   # (BK, BN) 子块
        acc += tl.dot(a, b)  # tl.dot 自动降低到 Tensor Core 的 mma 指令
    tl.store(c_ptr + ..., acc)
```

注意：你**只写了 `tl.load`/`tl.dot`/累加循环**。"把子块放进共享内存、做双缓冲（一边算一边预取下一块）、插同步、用 Tensor Core"——全是编译器自动做的。CUDA 里这是上百行 + 手调 `__shared__` + `__syncthreads` 的活。

## 7. 写 FlashAttention 类 kernel

注意力的瓶颈是中间矩阵 $S=QK^\top$（$N\times N$，$N$ 是序列长度）巨大，写回 HBM 再读回来做 softmax，访存爆炸。**FlashAttention 的核心：把 $S$ 分块，在片上算完 softmax 直接用掉，永不把 $N\times N$ 落盘 HBM**——这正是 Triton 擅长表达的"分块 + 片上复用"。详见 [[llm-optimizer/FlashAttention]]。

```
朴素注意力：           FlashAttention（Triton）：
 Q,K,V                  按 K/V 的块循环：
   │                     ┌── 取 K_j, V_j 块进 SMEM
 S=QKᵀ  (N×N) → HBM      │   s = Q·K_jᵀ            （片上）
   │                     │   在线更新 max / 分母    （online softmax）
softmax(S)→HBM           │   acc += p · V_j        （片上累加）
   │                     └── 循环结束，一次写回 O
 O=PV                    ❌ N×N 从不进 HBM
```

**online softmax（数值稳定的增量归一化）** 是关键。普通 softmax 要先看到整行才能算分母 $\sum e^{s_i}$；分块时一次只看到一块。维护**运行最大值 $m$** 和**运行分母 $\ell$**，每来一块就"修正"已累加的结果：

$$m_{\text{new}}=\max(m,\, \tilde m),\quad \ell_{\text{new}}=e^{m-m_{\text{new}}}\ell + e^{\tilde m - m_{\text{new}}}\tilde\ell$$

$$\text{acc}_{\text{new}}=e^{m-m_{\text{new}}}\,\text{acc} + e^{\tilde m - m_{\text{new}}}\,(P_j V_j)$$

减去 $m_{\text{new}}$ 防止 $e^{\text{大数}}$ 溢出（数值稳定的标准技巧）。

```python
# FlashAttention 内层循环骨架（Triton 伪代码）
m_i = -inf; l_i = 0; acc = zeros(BM, d)
for j in range(0, N, BN):                  # 遍历 K/V 的块
    k = tl.load(K_block_j); v = tl.load(V_block_j)
    s = tl.dot(q, k.T) * scale             # (BM, BN) 片上
    m_new = max(m_i, max(s, axis=1))       # 在线最大值
    p = tl.exp(s - m_new[:, None])         # 稳定指数
    alpha = tl.exp(m_i - m_new)            # 旧结果的衰减因子
    l_i = l_i * alpha + sum(p, axis=1)     # 更新分母
    acc = acc * alpha[:, None] + tl.dot(p, v)  # 更新累加器
    m_i = m_new
o = acc / l_i[:, None]                      # 最后归一化，一次写回 HBM
```

**为什么 Triton 写 FlashAttention 比 CUDA 香？** 在线 softmax 的逻辑本质是"块上的张量运算 + 标量状态更新"，Triton 的张量级抽象正好贴合；而合并访存、SMEM、流水线这些底层苦活又被编译器接管。社区里 FlashAttention 的 Triton 版常只有数百行，可读、可改、可 autotune。

## 8. Triton vs CUDA 全面对比

```
抽象层级阶梯（越上越省心，越下越能压榨）：
  PyTorch 算子  ──→  Triton  ──→  CUDA C++  ──→  PTX/SASS 内联汇编
   最高层               中层            低层            最底层
   写得快               写得快+够快      控制全           极致但难
```

| 维度 | Triton | CUDA |
|---|---|---|
| 编程视角 | **block/program 级**，张量运算 | **thread 级**，标量运算 |
| 语言 | 嵌入 Python 的 DSL | C/C++ 扩展 |
| 共享内存 | **编译器自动** | 手写 `__shared__` |
| 同步 `__syncthreads` | **自动插入** | 手写 |
| 合并访存 | **编译器保证** | 手工保证 |
| Tensor Core | `tl.dot` 自动 | 手写 wmma/mma |
| 调参 | `@autotune` 自动搜 | 手工/脚本扫 |
| 控制粒度 | 中（够大多数场景） | 极细（warp/寄存器/swizzle） |
| 上手成本 | 低（会 PyTorch 即可起步） | 高 |
| 代码量（一个 FA） | 数百行 | 上千行 |
| 可移植性 | 后端可换（NV/AMD…） | 主要绑 NVIDIA |
| 极限性能 | 接近手写、多数追平 | 理论上限最高 |

**何时用 Triton？** ① 需要写自定义/融合算子但不想陷进 CUDA；② 研究迭代快、要频繁改 kernel；③ 让 `torch.compile` 自动生成。**何时还得 CUDA？** ① 要榨干最后 5–10% 性能；② 需要 Triton 暂不支持的硬件特性/指令；③ 已有成熟 CUDA 库（cuBLAS/cuDNN/CUTLASS）直接用更划算。

**权衡一句话**：Triton 用"放弃一部分极限控制力"换"巨大的开发效率和可读性"，对绝大多数 LLM 融合算子，这笔交易非常值。

## 9. 生态与工具链

```
        ┌──────────── PyTorch 生态 ────────────┐
        │  torch.compile  ──(Inductor 后端)──►  生成 Triton kernel
        └───────────────────────────────────────┘
                          │
        ┌─────────────── Triton ────────────────┐
        │  triton.language(tl.*)  @triton.jit    │
        │  @triton.autotune  @triton.heuristics  │  ← 自动调 BLOCK/num_warps/num_stages
        │  TRITON_INTERPRET=1 (CPU 解释器调试)    │
        └────────────────────────────────────────┘
                          │
        ┌──── 后端 ────┐   缓存：~/.triton（编译产物缓存）
        │ NVIDIA→PTX   │   调试：TRITON_PRINT_AUTOTUNING、看 IR/PTX dump
        │ AMD→AMDGPU   │   性能：do_bench、Nsight Compute 看 SASS
        └──────────────┘
```

- **`@triton.autotune`**：给一组候选配置（`BLOCK`、`num_warps`、`num_stages`），运行时实测选最快的并缓存。`num_stages` 控制软件流水的级数（预取深度）。
- **关键集成**：是 PyTorch 2.x `torch.compile`（Inductor）在 GPU 上的主力代码生成器；vLLM、SGLang、Unsloth、xFormers 等推理/训练框架大量用 Triton 写融合算子。
- **多后端**：NVIDIA 成熟；AMD（ROCm）官方支持；Intel/其他硬件在推进（"以官方文档为准"）。
- **调试**：`TRITON_INTERPRET=1` 让 kernel 在 CPU 上以 Python 解释执行，可以 `print`/打断点（具体环境变量"以官方文档为准"）。

> 版本/具体 API/环境变量请始终**以官方文档与所装版本为准**，Triton 迭代很快。

## 数值例子：带宽与"够不够快"的手算

判断一个 kernel 写得好不好，最实用的标尺是**有效带宽**，对比 GPU 的峰值 HBM 带宽。

**例：向量加法 `out = x + y`，长度 $N=2^{24}\approx 1.6\times10^7$，FP32（4 字节）。**

- 访存量：读 $x$、读 $y$、写 $out$，共 3 个数组：
  $$\text{bytes}=3 \times N \times 4 = 3\times16{,}777{,}216\times4 \approx 2.01\times10^8 \text{ B} \approx 192\,\text{MiB}$$
- 假设实测耗时 $t=0.4\,\text{ms}=4\times10^{-4}\,\text{s}$：
  $$\text{带宽}=\frac{2.01\times10^8}{4\times10^{-4}} \approx 5.0\times10^{11}\,\text{B/s} = 500\,\text{GB/s}$$
- 若该卡峰值 HBM 带宽约 2000 GB/s（如某高端卡，"以官方为准"），则达到峰值的 $500/2000=25\%$ → **还有优化空间**（可能 BLOCK 太小、occupancy 不足）。理想的 memory-bound kernel 应逼近 80–90% 峰值。

**计算量对照（matmul，看是否算力受限）**：$M=N=K=4096$ 的矩阵乘，
$$\text{FLOPs}=2MNK=2\times4096^3\approx 1.37\times10^{11}\,\text{FLOPs}=137\,\text{GFLOP}$$
若耗时 1 ms，则 $137\,\text{GFLOP}/10^{-3}\text{s}=137\,\text{TFLOPS}$。对照该卡 Tensor Core 峰值（几百 TFLOPS 量级，"以官方为准"）判断是否充分利用 Tensor Core。

> 经验法则（roofline 直觉）：先算**算术强度** $I=\frac{\text{FLOPs}}{\text{bytes}}$；$I$ 小 → memory-bound（优化访存/融合）；$I$ 大 → compute-bound（优化用满 Tensor Core）。

## 常见问题

| 问题 | 解答 |
|---|---|
| Triton 是要取代 CUDA 吗？ | 不是。它在 CUDA 之上提供更高抽象，多数 kernel 够用且更快开发；极限优化仍可能回到 CUDA。 |
| 它只能跑 NVIDIA 吗？ | 不。NVIDIA 走 PTX，AMD 走 AMDGPU，后端可扩展；同一份 Python 换后端编译。 |
| 我要手写共享内存吗？ | 不用。编译器分析数据复用后自动分配 SMEM 并插同步——这是它最大卖点之一。 |
| `BLOCK` 怎么选？ | 用 `@triton.autotune` 自动搜；手工时一般取 2 的幂、配合 `num_warps`/`num_stages` 一起调。 |
| 为什么我的 kernel 慢？ | 多半访存没合并 / BLOCK 不当 / occupancy 低。先算有效带宽对照峰值，再看 IR/PTX 与 Nsight。 |
| `tl.dot` 会自动用 Tensor Core 吗？ | 在支持的 dtype/形状下会自动降低到 mma 指令；细节随版本，"以官方为准"。 |
| 和 `torch.compile` 什么关系？ | Inductor 后端会**自动生成 Triton kernel**，你常常"已经在用 Triton 而不自知"。 |
| online softmax 为什么要减最大值？ | 防止 $e^{\text{大数}}$ 溢出，是数值稳定 softmax 的标准技巧；分块时维护运行 max/分母增量更新。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航总图
- [[ai-framework/openai-triton/README]] — OpenAI Triton 框架与 API 细节
- [[ai-infra/ai-hardware/CUDA]] — CUDA 编程模型与 GPU 硬件基础（Triton 的地基）
- [[llm-optimizer/FlashAttention]] — FlashAttention 算法原理（第 7 节 kernel 的理论来源）
