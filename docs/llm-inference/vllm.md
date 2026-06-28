# vLLM —— 大模型高吞吐推理服务框架

> 一句话定位：vLLM 用 **PagedAttention** 把 KV Cache 像操作系统管理虚拟内存一样分页管理，再叠加 **Continuous Batching**，把 GPU 利用率和吞吐量推到极限。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/KV-Cache优化]] · [[llm-optimizer/kv-cache]] · [[llm-inference/解码策略]] · [[llm-inference/大模型推理张量并行]] · [[llm-optimizer/FlashAttention]] · [[llm-inference/README]]

参考资料（理解原理用，命令/版本以官方文档为准）：
- vLLM 论文《Efficient Memory Management for Large Language Model Serving with PagedAttention》(SOSP'23)
- VLLM 推理流程梳理（一）: https://zhuanlan.zhihu.com/p/649974825
- VLLM 推理流程梳理（二）: https://zhuanlan.zhihu.com/p/649977422
- 大模型推理服务框架 vLLM 要点简析 (上): https://zhuanlan.zhihu.com/p/654259045
- PagedAttention—大模型推理服务框架 vLLM 要点简析 (中): https://zhuanlan.zhihu.com/p/655561941

---

## 阅读地图

| 节 | 主题 | 你会得到 |
|----|------|----------|
| 0 | 一句话锚点 | vLLM 解决的核心痛点 |
| 1 | 地基 | 自回归推理 / Prefill vs Decode / KV Cache |
| 2 | 显存浪费的真相 | 内/外碎片、过度预留——手算浪费率 |
| 3 | PagedAttention | 分页 KV、块表、逻辑/物理块映射 |
| 4 | Continuous Batching | 迭代级调度 vs 静态批 |
| 5 | 内存共享与 Copy-on-Write | Beam Search / Parallel Sampling 省显存 |
| 6 | 调度器与抢占 | 抢占、Swap、Recompute |
| 7 | 张量并行 | 多卡切分与通信 |
| 8 | 整体架构 | Engine / Scheduler / Worker / KV Manager |
| 9 | 数值手算 | KV 块数、碎片、吞吐对比 |
| - | 常见问题 + 跳转链接 | |

---

## 0. 一句话锚点

LLM 推理慢、贵，**不是因为算力不够，而是因为显存（KV Cache）被浪费了**。传统框架给每个请求**连续、按最大长度**预留 KV Cache 空间，导致 60%~80% 显存闲置；批处理还用「等最长的人」的静态批。

vLLM 两板斧：

```
痛点                       vLLM 解法
─────────────────────────  ──────────────────────────────
KV Cache 连续分配 → 碎片    PagedAttention：分页 + 块表(非连续)
静态批 → GPU 空转           Continuous Batching：迭代级动态调度
─────────────────────────  ──────────────────────────────
结果：吞吐 ↑ 2~24×（相对 HF Transformers / 旧版 FasterTransformer）
```

---

## 1. 地基 / 前置

### 1.1 自回归解码：一次一个 token

LLM 推理是「逐 token」的：输出第 $t$ 个 token 要依赖前面所有 token。

```
输入: "中国的首都是"
step1: 中国的首都是 → 北     (forward 一次)
step2: 中国的首都是北 → 京   (forward 一次)
step3: ...北京 → <eos>       (停止)
```

每个 step 都要把前面所有 token 重新经过 Attention。若每步都重算 Key/Value，复杂度爆炸 → 引出 **KV Cache**。

### 1.2 KV Cache：用显存换算力

Attention 里每个 token 算出的 Key、Value 向量，后续 step 都要复用。把它们缓存下来，新 token 只算自己的 Q/K/V，再和缓存拼接：

```
            缓存的 K,V (历史)        新 token 的 q
Attention(  K[0..t-1],V[0..t-1],     q_t        )
```

单 token、单层的 KV Cache 大小（字节）：

$$\text{bytes} = 2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times \text{dtype}$$

> 详细推导见 [[llm-inference/KV-Cache优化]] / [[llm-optimizer/kv-cache]]。

### 1.3 两个阶段：Prefill 与 Decode

```
┌──────── Prefill (一次性) ────────┐ ┌──── Decode (逐token循环) ────┐
│ 把整段 prompt 并行喂入           │ │ 每次只喂 1 个 token           │
│ 计算密集 (compute-bound)        │ │ 访存密集 (memory-bound)       │
│ 一次填满 prompt 的 KV Cache     │ │ 每步往 KV Cache 追加 1 格     │
└─────────────────────────────────┘ └──────────────────────────────┘
```

- Prefill：GPU 算得满，瓶颈在算力。
- Decode：每步只算一个 token，瓶颈在**搬 KV Cache 的带宽**和**显存容量**。
- vLLM 的优化主战场是 **Decode 阶段的显存管理**。

---

## 2. 显存浪费的真相（vLLM 要解决什么）

传统框架（如早期 HF / FasterTransformer）给每个请求**预留一块连续显存**，长度按 `max_seq_len`。问题来了：

```
请求实际只输出 100 token，但预留了 2048：
┌──────────────────────────────────────────────┐
│ 已用 100 │■■■■■■■■ 浪费 1948 (预留未用) ■■■■■■■│  ← 内部碎片(internal)
└──────────────────────────────────────────────┘
请求间残块凑不出连续大块：
[req A 用][空 50][req B 用][空 30] ← 外部碎片(external)，新请求放不进
```

三类浪费：

| 浪费类型 | 成因 | 占比（论文实测） |
|----------|------|------------------|
| 预留浪费（reserved） | 按 max_len 预占，但还没生成到 | 大 |
| 内部碎片（internal） | 生成结束后，预留块剩余部分 | 大 |
| 外部碎片（external） | 请求间隙凑不出连续块 | 中 |

论文实测：传统方案**真正存有效 KV 的显存只有约 20%~40%**，其余被上面三者吃掉。

> 核心洞察：操作系统几十年前就用「**虚拟内存 + 分页**」解决了一模一样的碎片问题。把这套思想搬到 KV Cache，就是 PagedAttention。

---

## 3. PagedAttention：KV Cache 的「分页」

### 3.1 类比操作系统虚拟内存

```
操作系统                       vLLM PagedAttention
──────────────────────────     ─────────────────────────────
进程的虚拟地址空间          →   一条 sequence 的逻辑 KV 块序列
物理内存页 (page)           →   GPU 显存里的物理 KV 块 (block)
页表 (page table)           →   块表 (block table)
按页分配，非连续            →   按块分配，物理上可不连续
```

### 3.2 把 KV Cache 切成固定大小的「块」

每个 **KV 块（block）** 存固定 `block_size` 个 token 的 K 和 V（典型 `block_size=16`）。一条序列的 KV 是若干**逻辑块**，通过**块表**映射到**物理块**——物理上可以散落各处：

```
逻辑视图 (sequence "中国的首都是北京")  block_size=4
  逻辑块0: [中 国 的 首]   逻辑块1: [都 是 北 京]
                │                       │
          ┌─────┴── 块表 (Block Table) ─┴─────┐
          │ logical 0 → physical 7            │
          │ logical 1 → physical 2            │
          └───────────────┬──────────────────┘
                          ▼  物理显存（非连续）
  物理块: [0][1][2:都是北京][3][4][5][6][7:中国的首]...
                    └ 逻辑块1            └ 逻辑块0
```

### 3.3 PagedAttention 的算子改造

普通 Attention 假定 K、V 在显存里**连续**，可直接切片。分页后 KV 散落，需要改 kernel：**按块取 K/V，块内连续、块间跳转（靠块表）**。

```
for 每个 KV 块 b in 块表[seq]:
    K_b, V_b = 取物理块(b)          # 块内连续可向量化
    部分 attention 分数 = q · K_b^T  # 在线 softmax 累加
合并各块结果 → 输出
```

这和 [[llm-optimizer/FlashAttention]] 的「分块 + 在线 softmax」精神一致：都是把大矩阵切块、流式累加，避免一次性物化全部注意力矩阵。vLLM 后续版本直接把 FlashAttention 作为底层 attention backend。

### 3.4 收益：碎片几乎归零

- 内部碎片：只剩**最后一个块**没填满，最多浪费 `block_size-1` 个 token（如 15 个），与序列长度无关。
- 外部碎片：**消失**——块大小统一，任何空闲块都能复用。
- 按需增长：序列变长才追加新块，不再按 max_len 预占。

---

## 4. Continuous Batching：迭代级调度

### 4.1 静态批 vs 连续批

静态批（static / synchronous batching）：一批请求**同进同出**，必须等批内最慢（最长）的那个生成完，整批才释放。短请求被长请求拖死，GPU 大量空转。

```
静态批（行=请求，█=在算，·=已结束仍占坑空转）
req1 ███·····················   (5 token 就结束，但要陪跑)
req2 ███████████████████████   (最长，决定整批时间)
req3 ████████····             ·(8 token 结束，陪跑到底)
     └─ GPU 在 · 处空转，吞吐被最长请求锁死 ─┘
```

连续批（continuous / iteration-level batching）：调度粒度是**每一步（iteration）**，不是整批。哪个请求生成完 `<eos>` 就立刻退出、**释放它的 KV 块**，调度器**马上把等待队列里的新请求塞进来**填空位：

```
连续批（每步重新组批，结束即退出，新请求即时补位）
step:  1   2   3   4   5   6   7 ...
req1  ███ ███ ███ EOS                 ← 第4步结束，立刻退出
req5            ▲ 新请求第4步加入 ███ ███ ...
req2  ███ ███ ███ ███ ███ ███ ...
槽位始终被填满 → GPU 利用率接近 100%
```

### 4.2 为什么连续批离不开 PagedAttention

连续批要频繁地**为新请求分配、为完成请求回收** KV 空间。如果 KV 必须连续分配，回收/再分配会立刻产生外部碎片，连续批根本玩不转。**分页让「即来即分、即走即收」变得廉价**——这就是两板斧必须配套的原因。

---

## 5. 内存共享与 Copy-on-Write

分页还自然支持**多序列共享同一物理块**，这对以下场景是显存暴击优化：

- **Parallel Sampling**：同一 prompt 采样 $n$ 个不同续写。
- **Beam Search**：多条候选束共享公共前缀。

```
Prompt 相同 → 共享前缀块（引用计数 ref=3）
seq A ─┐
seq B ─┼─→ [共享 prompt 块: ref=3] → 各自分叉块
seq C ─┘
当某序列要写共享块 → 触发 Copy-on-Write：
  复制该块 → ref-1，新块 ref=1，只复制被修改的那一块
```

效果：prompt 只在显存里存**一份**。论文里 Parallel Sampling / Beam Search 的显存节省可达 55%+。机制与 OS 的 `fork()` COW 完全同构。

---

## 6. 调度器与抢占（显存不够时怎么办）

显存有限，等待队列可能塞不下所有请求。调度器按策略（默认 FCFS 类）准入；当**正在跑的请求要新块但没块了**，触发**抢占（preemption）**，把某些序列暂时踢出：

```
两种抢占恢复策略
┌── Swapping（换出到 CPU 内存）──────────────┐
│ 把被抢占序列的 KV 块拷到 CPU RAM，腾出 GPU │
│ 恢复时再拷回 GPU。省算力，花 PCIe 带宽      │
└────────────────────────────────────────────┘
┌── Recomputation（重算）────────────────────┐
│ 直接丢弃 KV，恢复时把该序列 prompt 重新     │
│ prefill 一遍。省显存搬运，花算力            │
└────────────────────────────────────────────┘
```

- 序列短 → 重算便宜，倾向 Recompute。
- 序列长 → 重算贵，倾向 Swap。
（具体默认行为以官方实现为准。）

---

## 7. 张量并行（多卡部署）

单卡放不下大模型时，vLLM 用**张量并行（TP）**把每层权重按列/行切到多张卡。每张卡各自维护**自己那部分 head 的 KV Cache**，PagedAttention 在每卡内独立分页。

```
TP=2，注意力头 32 → 每卡 16 头
GPU0: head 0..15  权重分片 + 自己的 KV 块池
GPU1: head 16..31 权重分片 + 自己的 KV 块池
每层结束 → All-Reduce 合并部分结果
```

每个 Transformer 层有 2 次 All-Reduce（Attention 后、MLP 后）。通信量推导与拓扑见 [[llm-inference/大模型推理张量并行]] 与 [[ai-infra/网络/集合通信原语]]。一个 vLLM 进程通过多 Worker 进程（每卡一个）协同，中央调度器统一发指令。

---

## 8. 整体架构（一张图串起来）

```
                ┌──────────── LLMEngine ────────────┐
   请求 ──────► │  Scheduler（连续批 + 抢占决策）     │
   (prompt)     │      │                              │
                │      ▼                              │
                │  Block Manager（块表/分配/回收/COW）│
                │      │  逻辑块→物理块映射            │
                │      ▼                              │
                │  Worker(s)（每张GPU一个）           │
                │   ├ Model Runner: 前向计算          │
                │   └ Cache Engine: PagedAttention    │
                │        kernel + KV 物理块池          │
                └────────────────────────────────────┘
   兼容 OpenAI API 的 Server 包在最外层（/v1/chat/completions 等）
```

执行一拍（iteration）：
1. Scheduler 选出本步要跑的序列集合（连续批）。
2. Block Manager 给需要新块的序列分配物理块、更新块表；不够则抢占。
3. Worker 用 PagedAttention 做一次前向，每个序列产出 1 个新 token。
4. 完成（`<eos>`/到长度）的序列退出并释放块，循环回到 1。

---

## 9. 数值手算

> 设定：模型 13B 级别，`n_layers=40`，`n_kv_heads=40`，`d_head=128`，dtype=FP16（2 字节），`block_size=16`。单卡 A100-40GB。

### 9.1 单 token 的 KV Cache 大小

$$\text{bytes/token} = 2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times 2$$
$$= 2 \times 40 \times 40 \times 128 \times 2 = 1{,}638{,}400 \text{ B} \approx 1.56\ \text{MB}$$

（factor 2 = K 和 V 各一份。）

### 9.2 一个 KV 块多大

$$\text{bytes/block} = \text{bytes/token} \times block\_size = 1.56\ \text{MB} \times 16 \approx 25\ \text{MB}$$

### 9.3 留给 KV 的显存能放多少块、多少 token

设权重等占 30GB，留 ~10GB 给 KV：
$$\text{块数} = \frac{10 \times 1024\ \text{MB}}{25\ \text{MB}} \approx 409\ \text{块}$$
$$\text{总 token} = 409 \times 16 \approx 6{,}544\ \text{token}$$

即可同时容纳约 6.5k token 的活跃 KV（如 12 个并发、每个 ~500 token）。

### 9.4 碎片浪费：分页前 vs 分页后

某请求实际只生成 **100 token**，而 `max_len=2048`：

```
传统连续预留：
  预留 = 2048 token，实用 = 100 → 浪费 1948 token
  浪费率 = 1948/2048 ≈ 95.1%   （单请求极端情形）

PagedAttention（block_size=16）：
  需要块数 = ceil(100/16) = 7 块，覆盖 112 token
  浪费 = 112-100 = 12 token（只在最后一块）
  浪费率 = 12/112 ≈ 10.7%，且与 max_len 无关
```

把单请求浪费从 95% 压到 ~11%。批量场景下，传统方案有效显存利用常 < 40%，vLLM 可逼近 96%——**等效地把 batch / 吞吐放大数倍**。

### 9.5 共享省显存（Parallel Sampling）

prompt 长 512 token，采样 $n=4$ 个续写：

```
不共享：4 × 512 = 2048 token 的 prompt KV
共享后：1 × 512 = 512 token（prompt 一份）+ 各自分叉部分
仅 prompt 部分就省 (2048-512)/2048 = 75%
```

---

## 常见问题

| 问题 | 答 |
|------|-----|
| PagedAttention 会损失精度吗？ | 不会。只改 KV 的**存储/取数布局**，数学等价于普通 attention。 |
| block_size 怎么选？ | 太小→块表开销/kernel 跳转多；太大→内部碎片回升。常用 16。以官方默认为准。 |
| 连续批和 PagedAttention 谁更重要？ | 互为前提：连续批要频繁分配/回收 KV，分页让其无碎片地廉价完成。 |
| Prefill 和 Decode 能混在一批吗？ | 可以（chunked prefill / 混合调度），把 prefill 切块与 decode 共批，平滑算力与延迟。具体策略以版本为准。 |
| 和 FlashAttention 冲突吗？ | 不冲突，互补。FlashAttention 优化**单次** attention 的算/访存；PagedAttention 优化**KV 的显存管理**。vLLM 用 FA 作 backend。 |
| 抢占用 Swap 还是 Recompute？ | 取决于序列长度与带宽/算力权衡；短序列偏重算，长序列偏换出。 |
| GPU 利用率为何能接近满？ | 连续批让结束的槽位即时被新请求填补，消除「陪跑」空转。 |
| 多卡时 KV 怎么分？ | 按张量并行的 head 切分，每卡存自己 head 的 KV，各自独立分页。 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引
- [[llm-inference/README]] — 推理章总览
- [[llm-inference/KV-Cache优化]] / [[llm-optimizer/kv-cache]] — KV Cache 显存估算与优化
- [[llm-inference/解码策略]] — Greedy / Beam / Parallel Sampling，与 5 节内存共享呼应
- [[llm-inference/大模型推理张量并行]] — 第 7 节多卡 TP 的通信量推导
- [[llm-optimizer/FlashAttention]] — 第 3.3 节分块 attention 的底层 kernel
- [[llm-optimizer/计算通信重叠]] — TP 通信与计算 overlap
- [[ai-infra/网络/集合通信原语]] — All-Reduce 等原语
- [[ai-infra/算力/GPU工作原理]] — 显存带宽 / compute-bound vs memory-bound
- [[llm-compression/quantization/fp8]] — KV Cache / 权重量化进一步压显存
- [[llm-algo/transformer/模型架构]] — Attention / MLP 结构地基
