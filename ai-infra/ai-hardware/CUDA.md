# CUDA 编程模型

> CUDA 是 NVIDIA 把 GPU 上「成千上万个线程」组织成 grid→block→thread 三级层级、用同一段 kernel 代码并行处理海量数据的编程模型。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-infra/算力/GPU工作原理]]

## 阅读地图

| 你想搞清楚的问题 | 去哪一节 |
| --- | --- |
| GPU 为什么要分 grid/block/thread 三级？ | 1、2 |
| 一个线程怎么知道自己该算哪个数据？ | 3 |
| warp 是什么，为什么 32 这个数字到处出现？ | 4 |
| block 内线程怎么共享数据、怎么同步？ | 5 |
| 一个最小可运行的 kernel 长什么样？ | 6 |
| 怎么知道我的 kernel 把 GPU 用满了没有？ | 7（occupancy） |
| CUDA 和 cuDNN/cuBLAS 是什么关系？ | 8 |
| 来点具体数字，把 grid 配置算出来 | 9（手算） |

## 0. 一句话锚点

**写一份代码（kernel），描述「一个线程做什么」；启动时告诉 GPU「开多少个线程」；GPU 用硬件把这些线程铺到上千个计算核心上同时跑。** 你不写循环遍历数据，而是「每个数据点配一个线程」。

## 1. 地基：CPU 思维 vs GPU 思维

先从最底层的差别讲起，否则后面所有层级都会显得莫名其妙。

**CPU 做向量加法**：一个（或几个）强力核心，串行地一个一个加。

```
for i in 0..N:        # 一个核心跑 N 次循环
    C[i] = A[i] + B[i]
```

**GPU 做向量加法**：开 N 个弱小线程，每个线程只负责一个 `i`，全部「同时」跑。

```
thread 0 → C[0] = A[0] + B[0]
thread 1 → C[1] = A[1] + B[1]
...
thread N-1 → C[N-1] = A[N-1] + B[N-1]
```

**为什么这样划算？** 向量加法里，`C[i]` 之间互不依赖（叫**数据并行 / embarrassingly parallel**）。CPU 有 8~64 个强核，GPU 有上万个弱核。当任务是「同样的操作 × 海量数据」时，「核多」碾压「核强」。这正是矩阵乘、卷积、注意力的形状——所以深度学习长在 GPU 上。

**关键约束（埋个伏笔，第 4 节展开）**：GPU 的核不是各自独立的，而是 32 个绑在一起、被迫执行同一条指令。这决定了后面所有性能问题。

## 2. 三级层级：grid / block / thread

CUDA 把线程组织成**严格的三层**，这不是为了好看，而是直接对应硬件结构。

```
                         Grid（一次 kernel 启动 = 一个 grid）
   ┌───────────────────────────────────────────────────────┐
   │  Block(0,0)      Block(1,0)      Block(2,0)   ...       │
   │  Block(0,1)      Block(1,1)      Block(2,1)   ...       │
   └───────────────────────────────────────────────────────┘
                          │ 放大一个 Block
                          ▼
        Block = 一组 thread（最多 1024 个）
   ┌───────────────────────────────────────┐
   │ T0  T1  T2  T3  T4  T5  T6  T7 ...      │
   │ ... 这些线程能共享内存、能互相同步      │
   └───────────────────────────────────────┘
```

| 层级 | 是什么 | 关键能力 / 限制 |
| --- | --- | --- |
| **thread** | 最小执行单元，跑一份 kernel | 有自己的寄存器、私有局部内存 |
| **block** | 一组 thread（≤1024） | **块内**可共享内存、可 `__syncthreads()` 同步 |
| **grid** | 一次启动的所有 block | block 之间**不能**直接同步/通信（默认） |

**为什么 block 之间不能同步？** 因为 GPU 一次装不下所有 block。block 是**调度单位**：GPU 上有若干个流多处理器（SM），block 被分发到 SM 上，跑完一批再上下一批，顺序不保证。如果允许 block A 等 block B，而 B 还没被调度，就死锁了。**「block 独立、可任意顺序执行」是 CUDA 可扩展性的根基**——同一段代码，小 GPU 慢慢轮，大 GPU 一次铺开，无需改代码。

grid 和 block 都可以是 1D/2D/3D（用 `dim3` 描述），这只是为了让「线程编号」更自然地映射到向量(1D)、图像(2D)、体数据(3D)，本质都是一维线性展开。

## 3. 核心技巧：线程 → 数据的映射

这是 CUDA 编程**最重要**的一行公式。每个线程靠内置变量算出「我是谁」：

| 内置变量 | 含义 |
| --- | --- |
| `threadIdx.x` | 我在 block 内的局部编号（0 .. blockDim.x-1） |
| `blockIdx.x` | 我所在 block 在 grid 内的编号 |
| `blockDim.x` | 每个 block 有多少线程 |
| `gridDim.x` | grid 里有多少 block |

**全局唯一编号**（1D 情形）：

$$\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

为什么是这个式子？因为线程是「分段编号」的：第 0 个 block 占 `0..blockDim-1`，第 1 个 block 占 `blockDim..2*blockDim-1`……所以「前面有 blockIdx 个满块」+「块内偏移 threadIdx」。

```
blockDim.x = 4

block 0:        block 1:        block 2:
[t0 t1 t2 t3]   [t0 t1 t2 t3]   [t0 t1 t2 t3]
 0  1  2  3      4  5  6  7      8  9  10 11      ← 全局 idx
 │                                               
 idx = 1*4 + 2 = 6  ← block1 的 t2
```

**为什么需要 `if (idx < N)` 守卫？** 线程总数 = `gridDim × blockDim` 几乎总是 ≥ N（向上取整出来的），多出来的线程必须什么都不做，否则会越界读写。这是新手最常踩的坑。

## 4. warp：被隐藏却决定性能的一层

block 看起来是一堆独立线程，但**硬件不是一个一个执行的**。SM 把 block 里的线程**每 32 个打包成一个 warp**，一个 warp 的 32 个线程**锁步执行同一条指令**（SIMT：Single Instruction, Multiple Threads）。

```
一个 block（128 线程）被切成 4 个 warp：
 warp0: t0..t31   ┐
 warp1: t32..t63  │  每个 warp 内 32 线程同一时刻执行同一条指令，
 warp2: t64..t95  │  但各自带不同数据
 warp3: t96..t127 ┘
```

这一层带来两个**必须知道**的性能法则：

**① warp divergence（分支发散）**。同一 warp 内若走不同分支：

```
if (threadIdx.x % 2 == 0)   // 偶数线程
    do_A();
else                        // 奇数线程
    do_B();
```
硬件没法让 32 个线程同时跑两条不同指令，只能**先让偶数线程跑 A（奇数线程闲置）、再让奇数线程跑 B（偶数闲置）**，两段串行执行 → 有效算力直接腰斩。**结论：尽量让同一 warp 内的线程走相同路径。**

**② 内存合并（coalescing）**。warp 一次访存，若 32 个线程访问的是**连续地址**，硬件合并成 1 笔大事务；若访问分散，则拆成多笔，带宽利用率暴跌。**结论：让相邻线程访问相邻内存**（这也是为什么 `idx` 映射要让 `threadIdx.x` 对应最内层连续维度）。

「block 大小最好取 32 的倍数」就是因为它：取 100 个线程 = 4 个满 warp + 1 个只用了 4 线程的 warp，那个 warp 浪费了 28/32 的算力。

## 5. 共享内存与块内同步

线程默认各算各的，但很多算法需要「block 内线程协作」。CUDA 给 block 配了一块**片上共享内存（shared memory）**：

```
内存层级（越往上越快越小）：
  寄存器 register   ── 每线程私有，最快，~几十 KB/SM
  共享内存 shared   ── 每 block 共享，片上，~快 (≈寄存器量级延迟)，几十 KB
  全局内存 global   ── 整卡共享，显存(DRAM)，大(GB级)但慢(几百周期延迟)
```

**为什么需要它？** 全局内存（显存）访问要几百个时钟周期。如果一份数据要被 block 内很多线程反复用，从全局内存重复读非常亏。把它**一次性搬到共享内存**，之后大家从共享内存（快几十倍）反复读。这是矩阵乘 tiling、卷积、reduce 的核心套路。

**`__syncthreads()`：块内屏障**。共享内存协作必然引出一个问题：线程 A 往共享内存写、线程 B 要读，必须保证「A 写完了 B 才读」。`__syncthreads()` 让 block 内**所有线程都到达这一行后**才一起继续。

```
共享内存版「block 内求和」示意：
  ① 每个线程把自己那份从 global 搬进 shared[threadIdx.x]
  ② __syncthreads();   ← 等所有人搬完，否则有人还没写就被读
  ③ 折半相加 reduce：
       step1:  s[0]+=s[4]  s[1]+=s[5] ...   __syncthreads();
       step2:  s[0]+=s[2]  s[1]+=s[3]        __syncthreads();
       step3:  s[0]+=s[1]                     ← 结果在 s[0]
```

**注意**：`__syncthreads()` 只能同步**同一个 block**，不能跨 block（回到第 2 节：block 间默认不能同步）。跨 block 协作要么拆成多次 kernel 启动，要么用更高级的协作组/原子操作。

## 6. 完整例子：向量加法 kernel

把前面所有概念串起来。这是 CUDA 的「Hello World」。

```cuda
// __global__ 表示这是个 kernel：CPU 调用、GPU 执行
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 第3节的映射公式
    if (idx < N) {            // 第3节的越界守卫
        C[idx] = A[idx] + B[idx];   // 每个线程只做 1 次加法
    }
}

// ---- 主机端（CPU）调用 ----
int N = 1 << 20;                       // 1,048,576 个元素
int threadsPerBlock = 256;             // block 大小（32 的倍数，见第4节）
int blocksPerGrid =
    (N + threadsPerBlock - 1) / threadsPerBlock;   // 向上取整，见第9节

// <<<grid, block>>> 就是「启动配置」：开 blocksPerGrid 个 block，每块 256 线程
vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
cudaDeviceSynchronize();   // kernel 是异步的，CPU 在此等 GPU 算完
```

> 注：`d_A/d_B/d_C` 是**显存**上的指针，需先 `cudaMalloc` 分配、`cudaMemcpy` 把数据从内存拷到显存；算完再拷回。具体 API 名以官方 CUDA Runtime 文档为准。这里重点是**编程模型**：`<<<...>>>` 描述并行结构，kernel 描述「一个线程做什么」。

**数据流全貌**：

```
 CPU内存(host)  ──cudaMemcpy(H2D)──▶  显存(device global)
                                          │
                        kernel<<<grid,block>>> 启动
                                          │
              上万线程并行  C[idx]=A[idx]+B[idx]
                                          │
 CPU内存(host)  ◀──cudaMemcpy(D2H)──   显存(device global)
```

## 7. occupancy（占用率）：GPU 用满了吗

光「开很多线程」不等于「跑得快」。每个 SM 是一座有固定资源的工厂：固定数量的寄存器、固定大小的共享内存、固定的「同时常驻 warp 数」上限。

**occupancy（占用率）** = SM 上**实际常驻的 warp 数** ÷ **硬件支持的最大常驻 warp 数**。

$$\text{occupancy} = \frac{\text{活跃 warp 数}}{\text{SM 最大 warp 数}}$$

**为什么 occupancy 重要？** GPU 隐藏延迟靠的不是缓存，而是「**切换 warp**」：当 warp0 在等显存（几百周期），SM 立刻切到 warp1、warp2 接着算，让计算单元一刻不停。常驻 warp 越多，越有「备胎」可切，越能把访存延迟藏住。occupancy 太低 → 没足够 warp 掩盖延迟 → 计算单元干等。

**什么会压低 occupancy？** 每个 SM 资源是**被瓜分**的：

- **寄存器**：每线程用太多寄存器 → 同时能容纳的线程变少。
- **共享内存**：每 block 用太多共享内存 → 同时能容纳的 block 变少。
- **block 大小**：太小（如 32）可能撞上「每 SM 最多 N 个 block」的上限而塞不满。

```
某 SM 上限：最多 2048 线程常驻（= 64 warp）
 case A: block=256, 用很少寄存器 → 能塞 8 个block = 2048线程 → occupancy=100%
 case B: block=256, 每线程寄存器翻倍 → 只塞 4 个block = 1024线程 → occupancy=50%
```

**重要权衡**：occupancy 不是越高越好。有时**故意降低 occupancy**、给每个线程更多寄存器/共享内存，反而更快（指令级并行、减少访存）。occupancy 是**诊断指标**，不是优化目标本身。

## 8. CUDA 与 cuDNN / cuBLAS 生态

CUDA 本身只是「用 GPU 跑并行代码」的底座。要高效做深度学习，没人手写矩阵乘 kernel——而是站在 NVIDIA 调好的**库**之上。

```
┌──────────────────────────────────────────────┐
│  框架层  PyTorch / TensorFlow / JAX            │  ← 你写 model(x)
├──────────────────────────────────────────────┤
│  库层   cuBLAS(矩阵/向量)  cuDNN(卷积/RNN/注意力)│  ← 高度优化的算子
│         cuFFT(FFT) cuRAND(随机数) ...           │
├──────────────────────────────────────────────┤
│  CUDA Runtime / Driver API（本文的编程模型）    │  ← kernel、内存、流
├──────────────────────────────────────────────┤
│  硬件   GPU（SM × N，显存，Tensor Core）        │
└──────────────────────────────────────────────┘
```

| 库 | 干什么 | 在深度学习里对应 |
| --- | --- | --- |
| **cuBLAS** | GPU 上的 BLAS：矩阵乘 (GEMM)、向量运算 | 全连接层、attention 的 QKᵀ、注意力加权 |
| **cuDNN** | 深度学习专用原语：卷积、池化、归一化、激活、RNN | 卷积层、BatchNorm、LSTM |
| **cuFFT / cuRAND** | 快速傅里叶变换 / 随机数生成 | 信号处理、dropout/初始化的随机源 |

**为什么用库而不自己写？** 一个高性能 GEMM 要处理 tiling、共享内存、寄存器分块、Tensor Core 指令、不同形状的特判——NVIDIA 用上千工程师月调到极致，手写很难追平。**何时仍要自己写 kernel？** 当你的算子是「访存密集 + 多个小算子能融合」时（如 LayerNorm+激活+残差融成一个 kernel），自定义/融合 kernel 能省去中间结果反复进出显存的开销——这正是 FlashAttention、各类 fused kernel 的价值所在。

> 注：上述为库的稳定职责划分；各库的具体 API 签名、版本与对应 driver 版本以 NVIDIA 官方文档为准（见底部链接）。

## 9. 数值示例 / 逐数手算

**任务**：把 `N = 1,000,000` 个 float 相加，`threadsPerBlock = 256`。

**① 算需要多少 block（向上取整）**

$$\text{blocks} = \left\lceil \frac{N}{\text{tpb}} \right\rceil = \left\lceil \frac{1{,}000{,}000}{256} \right\rceil$$

$1{,}000{,}000 \div 256 = 3906.25$，向上取整 = **3907 个 block**。
CUDA 没有 `ceil`，用整数技巧：$(N + \text{tpb} - 1) / \text{tpb} = (1{,}000{,}000 + 255)/256 = 1{,}000{,}255 / 256 = 3907$（整数除）。✓

**② 一共开了多少线程？多出来几个？**

$$\text{线程总数} = 3907 \times 256 = 1{,}000{,}192$$

多出 $1{,}000{,}192 - 1{,}000{,}000 = 192$ 个线程 → 它们的 `idx ≥ N`，被 `if(idx<N)` 挡掉，什么都不做。这 192 个「空转」线程正是越界守卫存在的理由。

**③ 一共多少 warp？最后一个 warp 满不满？**

每 block 256 线程 = $256 / 32 = 8$ 个满 warp（256 是 32 的倍数，零浪费，印证第 4 节）。
全网格 warp 数 = $3907 \times 8 = 31{,}256$ 个 warp。

**④ occupancy 粗估**（假设某 SM 上限 2048 线程 / 64 warp，且寄存器/共享内存不成瓶颈）

每 block 256 线程，$2048 / 256 = 8$ 个 block 可同时常驻 → $8 \times 8 = 64$ warp = 满载。

$$\text{occupancy} = \frac{64}{64} = 100\%$$

**⑤ 显存够吗？** 三个数组 A、B、C，各 $10^6$ 个 float（4 字节）：

$$3 \times 10^6 \times 4\,\text{B} = 12 \times 10^6\,\text{B} \approx 11.4\,\text{MiB}$$

对动辄几十 GB 显存的 GPU 来说微不足道——所以这个例子瓶颈不在显存容量，而在**访存带宽**（向量加是访存密集型：每个元素 3 次访存只换 1 次加法）。

## 对照 / 复杂度表

| 维度 | thread | warp | block | grid |
| --- | --- | --- | --- | --- |
| 典型规模 | 1 | 32（固定） | ≤1024 线程 | 任意多 block |
| 同步手段 | —（自己） | 锁步(隐式) | `__syncthreads()` | 默认无（需多次启动/原子） |
| 共享资源 | 寄存器(私有) | — | 共享内存 | 全局内存(显存) |
| 调度单位 | — | 执行单位 | 调度单位(分发到 SM) | 一次 kernel 启动 |
| 编程时关注 | idx 映射 | 分支发散/合并访存 | 大小取 32 倍数、shared 用量 | block 独立性 |

| 内存 | 作用域 | 相对速度 | 容量 |
| --- | --- | --- | --- |
| 寄存器 | 线程私有 | 最快 | 最小（每 SM 几十 KB 分摊） |
| 共享内存 | block 共享 | 很快（片上） | 小（每 block 几十 KB） |
| 全局内存(显存) | 全卡 | 慢（几百周期） | 大（GB 级） |

## 常见问题

| 疑问 | 真相 |
| --- | --- |
| block 大小是不是越大越好？ | 否。受 1024 上限 + 寄存器/共享内存瓜分约束；常用 128/256，取 32 倍数即可，需实测调。 |
| occupancy 100% 就是最优？ | 否。它只是「延迟掩盖能力」的指标；低 occupancy 配高寄存器/ILP 有时更快。 |
| 不同 block 能互相通信吗？ | 默认不能、不保证执行顺序。要跨 block 同步通常拆成多次 kernel 启动。 |
| warp 是 32，未来会变吗？ | 至今 NVIDIA GPU 都是 32；写代码别硬编码，用 `warpSize` 更稳。 |
| 我必须手写 kernel 吗？ | 通常不必。矩阵乘/卷积用 cuBLAS/cuDNN；只有需要算子融合或特殊算子才自己写。 |
| 线程多就一定快吗？ | 否。向量加这类是**访存带宽受限**，加再多线程也被 DRAM 带宽卡住，不是算力问题。 |
| kernel 调用是同步的吗？ | 否，是异步的。CPU 发起后立即返回，需 `cudaDeviceSynchronize` 等结果或测时间。 |

## 🔗 跳转链接

- [[00-知识地图]] —— 全局索引，从这里回到 AI-Infra 知识地图
- [[ai-infra/算力/GPU工作原理]] —— SM / Tensor Core / 显存带宽等硬件细节，是本文「为什么这样编程」的物理根源

> 参考（以官方文档为准）：
> - CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
> - CUDA 编程手册中文版: https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese
> - CUDA Toolkit Release Notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
