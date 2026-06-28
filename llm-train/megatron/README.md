# Megatron 训练实战

> Megatron-LM 是 NVIDIA 出品的大模型预训练引擎，把"张量并行 + 流水线并行 + 数据并行"三种切分方式工程化，让单卡放不下的模型能跨成百上千张 GPU 高效训练。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/megatron-lm/README]] [[llm-train/megatron-deepspeed/README]] [[llm-train/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|--------|
| 0 | 一句话锚点：Megatron 到底解决什么 | "切大模型" |
| 1 | 地基：为什么单卡装不下、三种并行的本质 | 显存账、通信原语 |
| 2 | 端到端训练流程总览 | 预处理→训练→转换→推理 |
| 3 | 数据预处理：indexed dataset（.bin/.idx） | mmap、零拷贝 |
| 4 | 数据加载与 epoch 索引、融合 CUDA 内核 | DataLoader、kernel fusion |
| 5 | 张量并行 TP | 行/列切分、all-reduce |
| 6 | 流水线并行 PP | micro-batch、气泡、1F1B |
| 7 | 数据并行 DP 与三者组合 | world_size 公式 |
| 8 | Checkpoint：保存/恢复/重切分 | 分片、merge/split |
| 9 | Megatron-LM vs Megatron-DeepSpeed | 谁管什么 |
| 数值例子 | 显存/通信/world_size 手算 | 拍脑袋前先算 |
| 常见问题 | 典型坑速查表 | 救火手册 |

---

## 0. 一句话锚点

**Megatron-LM = 把一个 Transformer 模型沿三个维度切开，分摊到多张 GPU 上，并保证数学结果和单卡训练完全等价。**

- 沿"模型内部矩阵"切 → **张量并行 TP**（Tensor Parallel）
- 沿"模型的层"切 → **流水线并行 PP**（Pipeline Parallel）
- 沿"数据批次"复制 → **数据并行 DP**（Data Parallel）

三者正交，可叠加。一个 175B 的模型，单卡（80GB）连参数都放不下，但 TP=8 × PP=8 × DP=N 就能在集群上跑起来。

---

## 1. 地基：为什么需要它

### 1.1 先算一笔显存账（这是一切的出发点）

训练时一个参数要占多少字节？以混合精度 + Adam 优化器为例，**每个参数**大约需要：

- fp16 权重：2 字节
- fp16 梯度：2 字节
- fp32 权重副本（master weight）：4 字节
- Adam 一阶动量 m：4 字节
- Adam 二阶动量 v：4 字节

合计约 **16 字节/参数**（这就是著名的"模型状态 ≈ 16×参数量"经验值，以官方/DeepSpeed 文档为准）。

```
参数量 P=7B 的模型，仅"模型状态"显存：
  7e9 × 16 字节 ≈ 112 GB
单张 A100-80G 装不下 → 必须切分
```

再加上**激活值**（activation，前向中间结果，反向要用），它随 batch、序列长度、层数线性增长，往往和模型状态同量级甚至更大。所以"单卡放不下"是常态。

### 1.2 三种并行的本质（一张图看懂）

```
                 一个 Transformer 模型
        ┌─────────────────────────────────────┐
        │  Layer0  Layer1  ... LayerL          │
        │   每层内含: QKV / Attn / MLP 大矩阵    │
        └─────────────────────────────────────┘

  TP(张量并行): 把"每个大矩阵"横/竖切，多卡共算一层
       ┌────┬────┐
       │ W列0│ W列1│   ← 同一层的权重被切成两半，各占一卡
       └────┴────┘      前向/反向需要 all-reduce 把结果拼回

  PP(流水线并行): 把"层"分段，不同卡负责不同层段
       卡A: Layer0~3  →  卡B: Layer4~7  →  卡C: Layer8~11
       数据像流水线一样在卡间逐段流过(p2p 通信)

  DP(数据并行): 整个模型复制多份，各吃不同数据
       副本1(吃batch切片1) | 副本2(吃batch切片2) | ...
       反向后 all-reduce 同步梯度
```

**通信代价的直觉排序**（决定怎么排布局）：
- TP 通信最频繁、最重（每层前向反向都要 all-reduce），所以 **TP 一般只放在单机内**（走 NVLink，约数百 GB/s），跨机会被网络拖死。
- PP 通信是点对点（p2p），只在层段边界传一次激活，**通信量小**，适合跨机。
- DP 通信是每个 step 末尾一次梯度 all-reduce，可被反向计算重叠隐藏。

> 经验布局口诀：**TP 锁机内、PP 跨机器、DP 填满剩余卡**。

---

## 2. 端到端训练流程总览

```
 原始语料(jsonl)
      │  ① 预处理: tokenize + 建索引
      ▼
 indexed dataset (xxx.bin + xxx.idx)   ← 只算一次, 训练时 mmap 直接读
      │  ② pretrain_gpt.py 启动
      ▼
 ┌──────────────────────────────────────────────┐
 │ 构建并行进程组 (TP/PP/DP) → 切分并加载模型     │
 │ DataLoader 取 batch → 前向 → 反向 → 优化器更新 │
 │ 周期性保存 checkpoint (分片)                   │
 └──────────────────────────────────────────────┘
      │  ③ checkpoint 转换 (改变 TP/PP 切分, 或转 HF 格式)
      ▼
 ④ 推理 / 评测 / 部署
```

四个阶段，每个都有独立脚本，下面逐个拆。

---

## 3. 数据预处理：indexed dataset

### 3.1 它解决什么问题

如果训练时才做 tokenize（文本→token id），每个 epoch 都要重复一遍，CPU 成为瓶颈，且字符串读取慢。Megatron 的做法：**预处理阶段一次性把所有文本 tokenize 成整数序列，存成二进制 + 索引**，训练时通过内存映射（mmap）零拷贝读取，几乎不占额外内存、随机访问 O(1)。

### 3.2 产物结构（.bin / .idx）

```
  data_text_document.bin   ← 所有 token id 顺序拼接的"大数组"(uint16/uint32)
  data_text_document.idx   ← 索引: 第i条样本在.bin中的[起始偏移, 长度]

  .idx 内容示意:
    doc0 → offset=0       len=512
    doc1 → offset=512     len=480
    doc2 → offset=992     len=600
       ...
  训练取第k条 → 查idx拿到(offset,len) → mmap切片 → 即得token序列
```

为什么用 mmap：操作系统把文件映射进虚拟地址空间，**按页惰性加载**，多个 DataLoader worker 进程共享同一份页缓存，不会把整个数据集读进内存。TB 级语料也能轻松处理。

### 3.3 预处理命令的关键参数（讲含义，不背默认值）

调用 `tools/preprocess_data.py`，核心参数语义：

| 参数 | 含义 | 权衡 |
|------|------|------|
| `--input` | 输入 jsonl，每行一条，含文本字段 | 字段名由 `--json-keys` 指定 |
| `--tokenizer-type` | 分词器类型（如 GPT2BPE、SentencePiece、HF） | 必须与训练时一致 |
| `--vocab-file` / `--merge-file` | 词表与 BPE 合并表 | 决定 token id 空间 |
| `--append-eod` | 每条文档末尾追加 EOD（end-of-document）token | 让模型学会"文档边界"，必开 |
| `--workers` | 并行进程数 | 越大越快，受 CPU 核数限制 |
| `--output-prefix` | 输出前缀，生成 `<prefix>.bin/.idx` | — |

> 具体参数名/默认值以你所用 Megatron 版本的 `preprocess_data.py` 为准。

---

## 4. 数据加载与融合 CUDA 内核

### 4.1 高效 DataLoader 与 epoch 索引

Megatron-LM 带有一个高效的 DataLoader：数据在训练前已被 tokenize 和 shuffle，并拆分为带索引的编号序列，索引被持久化存储，因此 tokenize 只需计算一次。

构建索引的逻辑：先根据训练参数（要训多少 token / 多少 step）计算需要多少个 epoch，预先生成一个跨 epoch 的样本排列（ordering），再整体 shuffle。这与"迭代整个数据集直到用尽，再重复第二个 epoch"的常规写法不同——Megatron 把多个 epoch 的样本顺序**一次性规划好并打乱**，这平滑了学习曲线（避免 epoch 边界的分布突变）并节省训练时间。

```
 常规做法:   [epoch0 顺序读完][epoch1 顺序读完]...  ← 边界处分布突变
 Megatron:   预先规划 N 个 epoch 的全局样本序 → 整体 shuffle
             → 一条平滑、无明显 epoch 接缝的样本流
```

### 4.2 融合 CUDA 内核（fused kernel）

当一个计算在 GPU 上运行时，必要的数据从内存取出加载到 GPU 计算，结果再保存回内存。**融合内核的思想：把通常由 PyTorch 分别执行的相邻操作合并成一个硬件操作**，从而减少中间结果在显存（HBM）与计算单元之间的来回搬运。

```
 未融合:  f → x'(写回HBM) → 读x' → g → y'(写回HBM) → 读y' → h
          ↑ 多次 HBM 读写, 访存即瓶颈(memory-bound)

 融合后:  [f g h 合成一个kernel]  中间结果 x',y' 只留在GPU寄存器
          → 立刻被下一步使用, 不落 HBM → 显著加速
```

典型被融合的算子：LayerNorm、Softmax（含 mask/scale）、bias+GeLU、bias+dropout 等。此外 Megatron-LM 还用 Apex 的 **fused AdamW**，比原生 PyTorch 实现更快。

> 实践提示：DataLoader 与 fused 优化器可以在 transformers 生态里复用，但**自定义融合 CUDA 内核对新手非常不友好**（要写 CUDA、对齐数值），通常直接用现成的即可。

---

## 5. 张量并行 TP（Tensor Parallel）

### 5.1 核心机制：按列切 + 按行切

以 MLP 的两层矩阵 $Y = \text{GeLU}(XA)$，$Z = YB$ 为例（Megatron 的经典切法）：

- 第一层 $A$ **按列切**：$A=[A_1, A_2]$，则 $XA=[XA_1, XA_2]$，GeLU 逐元素，可各卡独立算 → **无需通信**。
- 第二层 $B$ **按行切**：$B=[B_1; B_2]$，则 $Z = Y_1B_1 + Y_2B_2$，各卡算一半再 **all-reduce 求和**。

```
 输入 X (各卡相同)
   ├── 卡0: X·A1 → GeLU → Y1 → Y1·B1 ┐
   └── 卡1: X·A2 → GeLU → Y2 → Y2·B2 ┘ all-reduce相加 → Z
   每层前向1次all-reduce, 反向再1次 → 通信非常频繁
```

Attention 同理：多头注意力天然按 head 维度切分，QKV/输出投影分摊到各卡。

### 5.2 关键参数与权衡

- `--tensor-model-parallel-size`（TP size）：一层被切成几份。
- **权衡**：TP 越大，单卡显存压力越小，但 all-reduce 通信越频繁。因为通信极重，**TP size 通常 ≤ 单机 GPU 数**（如 8），跨机做 TP 会被网络带宽（数十 GB/s）拖垮。

---

## 6. 流水线并行 PP（Pipeline Parallel）

### 6.1 核心机制与"气泡"问题

把 L 层切成若干 stage，分给不同卡，激活在 stage 间用 p2p 传递。但朴素流水线有**气泡（bubble）**——前面 stage 算完在等后面，GPU 空转。

```
 朴素(每stage算完整batch才传): 大量空泡(idle)
   卡A: ███········
   卡B: ···███·····
   卡C: ······███··   ← 利用率低

 解法: 把 batch 切成多个 micro-batch, 像流水线一样填满
   卡A: █1█2█3█4·····
   卡B: ··█1█2█3█4···
   卡C: ····█1█2█3█4·  ← 稳态时各卡都在忙
```

micro-batch 越多，气泡占比越小。气泡时间占比近似 $\frac{p-1}{m}$，其中 $p$ 是 stage 数（PP size），$m$ 是 micro-batch 数。所以 **m 要远大于 p**。

### 6.2 1F1B 调度

Megatron 用 **1F1B（one-forward-one-backward）** 调度：稳态时每个 stage 交替做一次前向、一次反向，相比"全部前向再全部反向"能**显著降低激活显存峰值**（不必同时缓存所有 micro-batch 的激活）。还有 interleaved（虚拟流水线）变体，进一步压缩气泡。

### 6.3 关键参数

- `--pipeline-model-parallel-size`（PP size）：切成几个 stage。
- `--micro-batch-size` 与 `--global-batch-size`：micro-batch 是单次前向喂入的小批；global = micro × DP数 × 梯度累积数。**调大 global/micro 的比值 = 增加 micro-batch 数 = 减小气泡**。
- **权衡**：PP 通信小、适合跨机，但 stage 越多气泡越大、对负载均衡（每段计算量均匀）越敏感。

---

## 7. 数据并行 DP 与三者组合

### 7.1 DP 机制

模型（在 TP×PP 切分后的"一份完整副本"）被复制 DP 份，各吃不同数据切片，反向后对梯度做 all-reduce 同步。Megatron 支持配合 **分布式优化器**（distributed optimizer，类似 ZeRO-1：把优化器状态按 DP 维分片）进一步省显存。

### 7.2 三维如何拼成集群（核心公式）

```
  world_size = TP_size × PP_size × DP_size
  （= 总 GPU 数）

  布局直觉(以 64 卡为例, TP=8 PP=2 DP=4):
  ┌─────────────── PP stage 0 ───────────────┬──── PP stage 1 ────┐
  │ DP副本0: [TP0..TP7] (机器0, NVLink内通信) │  DP副本0: [TP0..7]  │
  │ DP副本1: [TP0..TP7] (机器1)               │  DP副本1: [TP0..7]  │
  │ DP副本2 ...  DP副本3 ...                   │   ...               │
  └───────────────────────────────────────────┴─────────────────────┘
   TP 锁在机内走NVLink ; PP/DP 跨机走IB网络
```

启动时通常用 `torchrun`/`torch.distributed` 拉起 `world_size` 个进程，Megatron 内部根据三个 size 自动建立对应的通信进程组（process group）。

---

## 8. Checkpoint：保存、恢复与重切分

### 8.1 为什么 checkpoint 是"分片"的

模型被 TP×PP 切开，每个进程只持有自己那一片参数。所以保存时**每个 rank 存自己的分片**，目录里是一堆按 TP/PP rank 命名的子目录：

```
 checkpoint/iter_0001000/
   ├── mp_rank_00_000/   ← TP rank0, PP rank0 的参数分片
   ├── mp_rank_01_000/   ← TP rank1, PP rank0
   ├── mp_rank_00_001/   ← TP rank0, PP rank1
   └── ...
 latest_checkpointed_iteration.txt  ← 记录最新迭代号
```

恢复时各 rank 读回自己的分片即可，**前提是 TP/PP 切分必须和保存时一致**。

### 8.2 关键参数

- `--save` / `--load`：保存与加载目录。
- `--save-interval`：每多少 step 存一次（权衡：太密占磁盘+拖慢，太疏断点损失大）。
- `--no-load-optim` / `--no-load-rng`：只加载权重、不加载优化器状态/随机数（常用于"接着别的 run 微调"或换并行度）。

### 8.3 重切分（resharding）——一个高频实战需求

训练用 TP=8/PP=8，但推理只想用 TP=1，或要转成 HuggingFace 格式部署，就需要**改变切分**。Megatron 提供转换脚本（如 `tools/checkpoint/` 下的 util / convert 脚本）来 **merge（合并分片）/ split（重新切分）/ 转格式**。

```
 训练态: TP8×PP8 (64分片)  ──merge──▶  单卡完整权重  ──convert──▶  HF格式
                            ◀──split──  改成 TP2×PP4 等其它布局
```

> 转换脚本的精确名称/参数随版本变化，以官方 `tools/checkpoint` 文档为准。

---

## 9. Megatron-LM vs Megatron-DeepSpeed

二者常被混淆。简记：**Megatron-LM 强在"模型并行（TP/PP）"，DeepSpeed 强在"显存优化（ZeRO）"，Megatron-DeepSpeed 把两者缝合。**

| 维度 | Megatron-LM（NVIDIA） | Megatron-DeepSpeed（缝合版） |
|------|----------------------|------------------------------|
| 出品 | NVIDIA | 微软在 Megatron-LM 上集成 DeepSpeed（也有 BigScience 分支） |
| 并行强项 | TP + PP（3D 并行的"模型切分"两维由它实现） | 继承 Megatron 的 TP/PP，**额外接入 DeepSpeed 的 ZeRO/DP** |
| 显存优化 | 分布式优化器（类 ZeRO-1） | **ZeRO-1/2/3、ZeRO-Offload、激活检查点**更全 |
| 配置方式 | 命令行参数为主 | 额外有 DeepSpeed 的 `ds_config.json` |
| 适用 | 纯 NVIDIA 栈、追求极致 TP/PP 性能 | 想叠加 ZeRO 省显存、用 offload 跑超大模型 |
| checkpoint | Megatron 分片格式 | 多了 DeepSpeed 的 zero 分片，转换更绕（典型坑） |

> 详见 [[llm-train/megatron-deepspeed/README]]。选型直觉：模型并行够用且追求性能 → Megatron-LM；需要 ZeRO-3/offload 把更大模型塞进有限卡 → Megatron-DeepSpeed。

---

## 数值例子 / 对照 / 实践

### 例 1：world_size 与显存分摊（手算）

> 目标：用 A100-80G 训练 7B 模型，估算最小并行布局。

模型状态 ≈ $7\text{e}9 \times 16 = 112$ GB > 80GB，单卡放不下。若只开 TP=2：每卡模型状态约 $112/2 = 56$ GB，再留出激活与碎片，**TP=2 起步可行**；若 TP=4 则每卡约 28GB，更宽裕、可加大 batch。

```
  TP=2, PP=1, DP=4  → world_size = 2×1×4 = 8 张卡
  每卡模型状态 ≈ 112/2 = 56 GB  (TP切了模型状态)
  DP=4 → 吞吐近似 ×4 (各副本并行吃数据)
```

### 例 2：TP all-reduce 通信量（直觉）

一次 all-reduce 传输的数据量约为 $2(N-1)/N \times S$（ring all-reduce，$N$=参与卡数，$S$=张量字节数）。Transformer 每层有多次 all-reduce，层数 ×2（前向+反向），所以 **TP 通信总量 ∝ 层数 × hidden 维度 × batch**。这就是为什么 TP 必须走 NVLink（约数百 GB/s）而非 IB 网络（约数十 GB/s）——否则通信时间会吃掉所有收益。

### 例 3：流水线气泡占比（手算）

PP=4（p=4），若 micro-batch 数 m=4：气泡占比 $\approx (p-1)/m = 3/4 = 75\%$，几乎全在等！把 m 提到 32：$3/32 \approx 9.4\%$，可接受。**结论：global_batch 要足够大以喂出足够多 micro-batch。**

### 实践要点速记

- **先跑通小配置再放大**：TP=2/PP=1/DP=1 单机能跑通，再逐步加维度。
- **TP 锁机内、PP 跨机、DP 填满**——按通信代价排布局。
- **micro-batch 数尽量大**以压气泡；激活检查点（recompute）换显存。
- 预处理的 **tokenizer/vocab 必须与训练严格一致**，否则 token id 错位、loss 不收敛。
- 改并行度后**必须重切分 checkpoint**，不能直接 `--load` 旧分片。

---

## 常见问题

| 现象 / 问题 | 根因 | 处理 |
|------------|------|------|
| 加载 checkpoint 报 shape/分片不匹配 | 改了 TP/PP size 却直接 load 旧分片 | 用转换脚本 reshard，或恢复原并行度 |
| loss 一开始就发散/乱码 | 预处理与训练的 tokenizer/vocab 不一致 | 统一 tokenizer-type 与 vocab/merge 文件 |
| GPU 利用率低、卡间等待 | PP 气泡大（micro-batch 太少） | 增大 global batch / micro-batch 数；用 interleaved |
| 跨机 TP 后吞吐暴跌 | TP 走了 IB 网络而非 NVLink | 把 TP size 限制在单机卡数内 |
| OOM（显存爆） | 激活/优化器状态过大 | 开激活检查点、调大 TP、用分布式优化器/ZeRO |
| `--append-eod` 没加 | 文档边界缺失 | 预处理必加 EOD token |
| Megatron-DeepSpeed 的 ckpt 转不出 HF | zero 分片 + megatron 分片双层结构 | 先 merge zero，再用 megatron→HF 脚本，分两步 |
| 数据预处理巨慢 | `--workers` 太小、CPU 瓶颈 | 调大 workers，分片并行预处理后再合并 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，先看这张图定位
- [[ai-framework/megatron-lm/README]] — Megatron-LM 框架原理（TP/PP/通信组实现细节）
- [[llm-train/megatron-deepspeed/README]] — 缝合 DeepSpeed 的版本与 ZeRO 配置
- [[llm-train/README]] — 训练专题总目录（各框架对照）

### 参考项目 / 资料
- [CodeGeeX](https://github.com/THUDM/CodeGeeX) — 基于 Megatron-LM 实现的代码大模型
- [如何使用 Megatron-LM 训练语言模型](https://huggingface.co/blog/zh/megatron-training) — 数据预处理、训练、模型转换、推理全流程
