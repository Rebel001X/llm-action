# 推理优化总览

> LLM 推理优化 = 用「算子 / 内存 / 批处理 / 并行 / 解码 / 重叠」六类手段，把同一个模型在同一块卡上跑得更快、更省、更便宜。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-optimizer/FlashAttention]] [[llm-optimizer/kv-cache]] [[llm-optimizer/计算通信重叠]]

---

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|--------|
| 0 | 一句话锚点：推理优化到底在优化什么 | 延迟/吞吐/成本 |
| 1 | 地基：自回归推理两阶段、瓶颈的本质（算力 vs 带宽） | Prefill/Decode、Roofline |
| 2 | 算子优化：FlashAttention 为代表的 kernel 重写 | 融合、IO-aware |
| 3 | 内存优化：KV Cache 与 PagedAttention | 显存、碎片 |
| 4 | 批处理：连续批 / 分块预填充 | continuous batching、chunked prefill |
| 5 | 并行：TP / PP / EP / SP，把模型摊到多卡 | 张量/流水/专家并行 |
| 6 | 解码优化：投机解码、并行解码 | speculative decoding |
| 7 | 重叠：计算与通信、计算与计算的 overlap | overlap、kernel fusion |
| 8 | 按瓶颈选 + 组合拳：决策树与实战配方 | 选型、叠加 |
| 9 | 数值例子：显存/带宽/吞吐手算 | 一张卡能跑多大 |
| — | 常见问题 + 跳转链接 | FAQ |

---

## 0. 一句话锚点

推理优化的目标只有三个，且常相互冲突：

```
                ┌──────────────┐
                │   三个目标    │
                └──────────────┘
   Latency 延迟      Throughput 吞吐     Cost 成本
 (单请求多快)        (每秒多少 token)    (每百万 token 多少钱)
        │                  │                   │
        └─── TTFT/TPOT ────┴── tokens/s/GPU ───┘
```

- **TTFT**（Time To First Token，首 token 延迟）：用户发请求到看到第一个字的时间，由 **Prefill** 决定。
- **TPOT**（Time Per Output Token，每输出 token 时间）= ITL（inter-token latency），由 **Decode** 决定。
- **Throughput**：单位时间所有请求的总 token 数，决定每张卡的服务能力，直接换算成钱。

> 核心矛盾：把 batch 调大 → 吞吐↑成本↓，但单请求延迟↑。所有优化都在这条权衡曲线上挪动。本总览讲的「六类手段」，本质是把这条权衡曲线整体往右下方推（同样延迟下吞吐更高）。

---

## 1. 地基 / 前置

### 1.1 自回归推理：两个阶段

LLM 一次生成是「读完输入 → 一个一个吐字」。这天然分两段：

```
输入: "中国的首都是"
                        ┌─────────────────────────────────────────┐
  Prefill（预填充）     │ 一次性处理全部 6 个输入 token            │
  ─ 计算密集 ─          │ 6 个 token 同时过 Transformer，并行度高  │
                        │ 产出：每层每个 token 的 K、V 存进 Cache  │
                        └─────────────────────────────────────────┘
                                        │ 生成第 1 个 token "北"
                        ┌─────────────────────────────────────────┐
  Decode（解码）        │ 每步只处理 1 个新 token                  │
  ─ 访存密集 ─          │ 新 token 的 Q 去和 Cache 里所有 K、V 算  │
                        │ 自注意力，产出 1 个新 token，循环往复    │
                        └─────────────────────────────────────────┘
                          "北"→"京"→"。"→<eos>  逐字蹦
```

| 阶段 | 一次处理 token 数 | 瓶颈 | 优化重点 | 影响指标 |
|------|-----------------|------|---------|---------|
| Prefill | 全部输入（几十~几千） | **算力**（compute-bound） | 算子融合、并行 | TTFT |
| Decode | 1（每步） | **访存带宽**（memory-bound） | KV Cache、批处理、投机 | TPOT |

### 1.2 为什么 Decode 是「访存密集」？—— Roofline 直觉

一个算子快不快，看它的 **算术强度**（Arithmetic Intensity）：

$$\text{算术强度} = \frac{\text{浮点运算次数 FLOPs}}{\text{读写字节数 Bytes}}$$

- 强度高 → 卡在算力（compute-bound）→ 该上更快的算子/更高精度的 Tensor Core。
- 强度低 → 卡在带宽（memory-bound）→ 该减少数据搬运（融合、量化、批处理）。

Decode 阶段 batch=1 时，每生成 1 个 token 要把**整个模型权重**从显存读一遍，却只做了「1 个 token」那点计算 → 算术强度极低 → 严重 memory-bound。

```
 Roofline 图（log-log）
 性能(FLOP/s)
   │            ┌──────────── 峰值算力（屋顶）
   │           /│
   │          / │
   │  带宽墙 /  │ compute-bound 区
   │ (斜线) /   │
   │      /     │
   │     /  ●Decode(强度低,贴在斜线上,被带宽卡死)
   │    /       │      ●Prefill(强度高,贴近屋顶)
   └───┴────────┴──────────────► 算术强度(FLOP/Byte)
```

> **一句话**：Prefill 缺算力，Decode 缺带宽。优化手段要对症——给 Decode 上「降带宽」的招（KV 复用、量化、批处理），给 Prefill 上「提算力利用」的招（融合、并行）。

---

## 2. 算子优化：FlashAttention 为代表

### 2.1 朴素注意力的痛点

标准注意力 $\text{softmax}(QK^\top/\sqrt{d})V$，序列长 $N$ 时要显式生成 $N\times N$ 的注意力矩阵 $S$ 写回显存，再读回来做 softmax，再读回来乘 $V$。

```
朴素: Q,K → [写] S(N×N) → [读] softmax → [写] P → [读] P@V
                 ▲ 这块 N×N 在 HBM 上反复读写，O(N²) 访存，N=4096 时巨贵
```

显存占用与访存量都是 $O(N^2)$，长序列时既爆显存又被带宽卡死。

### 2.2 FlashAttention 的核心思想：分块 + 在线 softmax + 不落盘

```
FlashAttn: 把 Q,K,V 切成小块，逐块搬进 SRAM（片上高速缓存）
           在 SRAM 里算局部注意力，用"在线 softmax"增量合并
           N×N 的中间矩阵从不写回 HBM → 访存降到 O(N)

   HBM(慢,大) ──load块──► SRAM(快,小) ──算──► 累加器 ──► 只写最终 O
                                  ▲ 全程不物化 N×N
```

- **IO-aware**：优化目标不是 FLOPs，而是 HBM 读写字节数。
- **kernel 融合**：matmul + mask + softmax + matmul 融成一个 GPU kernel，省掉中间张量的来回搬运。
- 效果：长序列下显存 $O(N^2)\to O(N)$，速度数倍提升，且**数学上等价**（不是近似）。

> 详见 → [[llm-optimizer/FlashAttention]]。同类算子优化还有：FlashDecoding（Decode 阶段沿 KV 长度切分提并行）、融合的 RMSNorm/RoPE/SwiGLU、量化 GEMM kernel（W8A8/W4A16）。

---

## 3. 内存优化：KV Cache 与 PagedAttention

### 3.1 KV Cache：用空间换时间

Decode 每步只新增 1 个 token，但它要和「前面所有 token 的 K、V」算注意力。若每步重算所有历史 K、V，则第 $t$ 步要重算 $t$ 个 token，总复杂度 $O(N^2)$。

**KV Cache**：把每个 token 在每层算出的 K、V 缓存下来，下一步直接复用，Decode 每步只算 1 个新 token 的 K、V。复杂度 $O(N^2)\to O(N)$。

KV Cache 显存（单序列、FP16）：

$$\text{KV} = 2 \times L \times n_{kv} \times d_{head} \times \text{seq} \times 2\text{B}$$

其中 $2$=K和V，$L$=层数，$n_{kv}$=KV 头数（GQA 下远小于 Q 头），$d_{head}$=每头维度，末尾 $2$B=FP16 字节。

### 3.2 痛点：KV Cache 太大 + 碎片化

KV Cache 随并发数 × 序列长线性膨胀，常常比模型权重还大，是限制并发的头号瓶颈。早期实现给每个序列预留「最大长度」的连续显存 → 内部碎片严重，利用率常不到一半。

### 3.3 PagedAttention：像操作系统分页一样管 KV

```
传统(连续预留):  序列A [████░░░░░░░░░░]  ← 预留 max_len，大量░浪费
                序列B [██░░░░░░░░░░░░]

Paged(分块按需): 物理块池  [B1][B2][B3][B4][B5][B6]...
                序列A 逻辑→物理:  B1→B4→B2   (按需分配，不连续也行)
                序列B 逻辑→物理:  B3→B6
                ▲ 块表(block table)做逻辑到物理映射，碎片≈0
```

- 把 KV Cache 切成固定大小的 **block**（如 16 token/块），用 **block table** 映射逻辑位置到物理块，按需分配。
- 显存利用率从 ~50% 拉到 ~96%，相同显存可服务的并发数翻倍。
- 天然支持 **前缀共享**（多个请求共享相同 system prompt 的 KV 块，copy-on-write）。

> 详见 → [[llm-optimizer/kv-cache]]。配套招数：量化 KV（KV 存 INT8/FP8 砍半显存）、KV 卸载到 CPU/NVMe、稀疏 KV（只留重要 token）。

---

## 4. 批处理：把吞吐榨干

### 4.1 静态批 vs 连续批（Continuous Batching）

GPU 喜欢「一次算一大批」。但请求长度不一、到达时间不一，静态批要等最慢的请求做完才能放新请求 → GPU 大量空转。

```
静态批(static):  R1 ████████████ done
                R2 ████ done......(空等)......│ 整批一起退，短请求干等长请求
                R3 ██████ done....(空等)......│
                                              └► 才能进下一批

连续批(continuous): 槽位级调度，谁完成谁立刻被新请求顶上
                R1 ████████████
                R2 ████|R4 ███████      ← R2 完成立刻插入 R4
                R3 ██████|R5 █████|R6 ██ ← 槽位不空转
                ▲ 每个 decode step 重新组批，GPU 利用率拉满
```

连续批（也叫 in-flight batching）按 **iteration 级**调度：每生成一步就检查谁结束了、谁能进来，是现代推理引擎吞吐翻几倍的关键。

### 4.2 分块预填充（Chunked Prefill）

长 Prefill 会「霸占」GPU，让正在 Decode 的请求卡顿（TPOT 抖动）。把长 Prefill 切成小块，和 Decode 步交错调度，平滑延迟、提高整体利用率。

```
不分块:  [─────── 长Prefill 独占 ───────][decode][decode]...
        ▲ 这期间所有 decode 请求被饿死，TPOT 飙升

分块:    [pf块][dec][pf块][dec][pf块][dec]...  ← prefill 与 decode 交织
        ▲ 单步算力预算固定，延迟平滑
```

> 进阶：**PD 分离**（Prefill/Decode Disaggregation）——把 Prefill（算力密集）和 Decode（带宽密集）放到不同的 GPU 池，各自用最优配置，通过 KV 传输衔接，避免互相干扰。

---

## 5. 并行：把模型摊到多卡

单卡放不下、或要更高吞吐时，沿不同维度切分。

```
张量并行 TP（层内切）        流水并行 PP（层间切）
  GPU0  GPU1                  GPU0: 层 1-8
  ┌──┬──┐  每个矩阵按列/行切   GPU1: 层 9-16
  │½ │½ │  算完 all-reduce 合  GPU2: 层 17-24
  └──┴──┘  通信频繁,需高带宽    像流水线,需 micro-batch 填气泡
  (适合卡间 NVLink)            (适合跨节点,通信少但有气泡)

专家并行 EP（MoE 专属）       序列并行 SP（长序列切）
  把不同 expert 放不同卡       沿序列维切分,配 TP 省激活显存
  按 router 结果 all-to-all   长上下文必备
```

| 维度 | 切什么 | 通信原语 | 通信量 | 适用 |
|------|--------|---------|--------|------|
| TP（张量） | 单层内矩阵 | all-reduce | 大、频繁 | 卡内 NVLink，单机 8 卡 |
| PP（流水） | 不同层组 | P2P send/recv | 小 | 跨节点，需调度填「气泡」|
| EP（专家） | MoE 的 expert | all-to-all | 中 | MoE 模型 |
| SP（序列） | 序列维 | all-gather/RS | 中 | 超长上下文 |
| DP（数据） | 不同请求批 | 无（独立副本） | 无 | 横向扩并发 |

> 实战常**组合**：单机内 TP=8，跨机 PP；MoE 再叠 EP。并行带来通信，通信要靠第 7 节的「重叠」藏起来。

---

## 6. 解码优化：投机解码

### 6.1 痛点：Decode 一次只蹦一个字，且 memory-bound

Decode 每步读一遍全部权重只产出 1 token，算力大量闲置。能不能「一步验证多个 token」？

### 6.2 投机解码（Speculative Decoding）

```
        ┌─ 小模型(draft) 便宜地连猜 K 个 token ─┐
草稿:   "北" "京" "是" "中" "国"   (猜 5 个)
        └────────────────┬───────────────────┘
                         ▼ 一次性喂给大模型
验证:   大模型并行验证这 5 个(一次前向,而非 5 次)
        ✓北 ✓京 ✓是 ✗中→改为"的"  接受前3+纠正1
        ▲ 接受 3 个 → 一步顶 4 步,大模型前向次数↓
```

- **draft-then-verify**：小模型（或同模型的浅层、或 Medusa 多头、或 EAGLE 特征外推）快速产出 K 个候选，大模型**一次前向并行验证**。
- 利用 Decode 本就 memory-bound 的特性——并行验证 K 个 token 几乎和验证 1 个一样快（带宽瓶颈不变，算力本来就闲）。
- **保证输出分布与原模型一致**（接受/拒绝采样），是无损加速。加速比取决于 draft 命中率，常见 1.5~3×。

> 同类：并行解码（Lookahead、Jacobi）、Medusa（加几个预测头）、EAGLE（在特征空间外推，命中率更高）。

---

## 7. 重叠：把等待时间藏起来

GPU 既要**算**（compute），又要**搬数据**（communication / memory）。串行做就是等；让它们**同时发生**就是赚。

### 7.1 计算与通信重叠

```
不重叠:  [── 计算 ──][── all-reduce 通信 ──][── 计算 ──]   总时长=算+通+算
重叠:    [── 计算块1 ──][── 计算块2 ──]
                       [── 通信块1 ──][── 通信块2 ──]      通信藏在计算下
         ▲ 把张量切块,算完一块就开始传它,同时算下一块
```

并行（第 5 节）必然引入 all-reduce / all-to-all 通信。把计算和通信**切块流水**，让通信「躲」在计算背后，几乎免费。kernel 融合（如把 GEMM 与 all-reduce 融成一个 kernel）能进一步消除启动开销。

### 7.2 计算与计算 / 计算与访存重叠

- **多流（multi-stream）**：不同 CUDA stream 上的独立算子并发执行。
- **双缓冲（double buffering）**：搬下一块数据的同时算当前块（FlashAttention 内部就在用）。
- **MoE 的 dispatch/combine 与专家计算重叠**。

> 详见 → [[llm-optimizer/计算通信重叠]]。判断重叠有没有用：看 GPU timeline 上有没有「气泡」（空白）——有气泡就有重叠空间。

---

## 8. 按瓶颈选 + 组合拳

### 8.1 先定位瓶颈，再下药（决策树）

```
                ┌── 你的痛点是什么？ ──┐
                ▼                      ▼
        TTFT 太高(首字慢)        TPOT 太高(蹦字慢)
                │                      │
        Prefill 是 compute-bound  Decode 是 memory-bound
                │                      │
   ┌────────────┼──────────┐   ┌───────┼─────────────┐
 FlashAttn   TP并行    chunked  KV优化  连续批     投机解码
 算子融合    提算力   prefill   /Paged  增 batch   /量化降带宽
                                量化KV
                ▼                      ▼
        ┌──────────────────────────────────┐
        │ 显存不够装下模型+KV？             │
        │  → TP/PP 并行 + PagedAttn + 量化  │
        │ 吞吐不够(成本高)？               │
        │  → 连续批 + 大 batch + PD 分离    │
        └──────────────────────────────────┘
```

### 8.2 优化手段全景对照

| 类别 | 代表技术 | 主攻瓶颈 | 主要收益 | 代价/风险 |
|------|---------|---------|---------|----------|
| 算子 | FlashAttention | Prefill 访存 | 显存 O(N²)→O(N)、提速 | 实现复杂、需对应 kernel |
| 内存 | KV Cache | Decode 重算 | O(N²)→O(N) 计算 | 吃显存 |
| 内存 | PagedAttention | 显存碎片 | 利用率↑、并发翻倍 | 需块表管理 |
| 内存 | KV/权重量化 | 带宽+显存 | 砍半显存、提速 | 精度损失（需评测）|
| 批处理 | 连续批 | GPU 空转 | 吞吐数倍 | 调度复杂 |
| 批处理 | chunked prefill / PD 分离 | 延迟抖动 | TPOT 平滑 | 工程复杂 |
| 并行 | TP/PP/EP/SP | 单卡装不下 | 可扩展 | 通信开销 |
| 解码 | 投机解码 | Decode 算力闲 | 1.5~3× 无损 | 需 draft、命中率 |
| 重叠 | 计算/通信 overlap | 通信等待 | 近免费提速 | 需切块/融合 |

### 8.3 组合拳示例（一个现代推理引擎的标配）

```
PagedAttention(管KV) + FlashAttention(算注意力)
        + 连续批(榨吞吐) + chunked prefill(平滑延迟)
        + TP=8(单机摊模型) + 计算通信重叠(藏通信)
        + 投机解码(可选,降TPOT) + FP8/INT8量化(降带宽显存)
        ────────────────────────────────────────────
        这些是"乘法关系"叠加: 各自 1.5~3×，组合后可达 10×+
```

> 关键认知：这六类**不是互斥的选项，而是要叠在一起的层**。算子优化打底，内存优化解显存，批处理拿吞吐，并行解规模，解码与重叠抠最后的延迟。

---

## 9. 数值例子 / 对照（手算）

> 以下用 7B 类模型的典型公开规格做量级估算，**具体以官方为准**。设：层数 $L=32$，隐藏维 $d=4096$，注意力头 32，FP16。

### 例 1：模型权重显存

$$7\times10^9 \text{ 参数} \times 2\text{B(FP16)} \approx 14\text{ GB}$$

一张 24GB 卡（如约 A10/4090 级别，以官方为准）装权重后剩 ~10GB 给 KV 和激活。

### 例 2：KV Cache 一个 token 占多少？（MHA，无 GQA）

$$\text{KV/token} = 2(\text{K,V}) \times L \times d \times 2\text{B} = 2\times32\times4096\times2 \approx 524\,\text{KB/token}$$

- 一条 2048 token 的序列：$524\text{KB}\times2048 \approx 1.0\text{ GB}$。
- 剩 10GB 显存 → 最多约 **10 条**这样的并发序列（这就是为什么 PagedAttention 省显存如此关键）。
- 若用 **GQA** 把 KV 头从 32 降到 8：KV/token 砍到约 131KB，同显存并发 **×4**。
- 若再把 **KV 量化到 INT8**：再砍半，并发再 ×2。

### 例 3：Decode 是不是 memory-bound？（带宽手算）

设卡显存带宽约 1 TB/s（以官方为准）。batch=1 时每生成 1 token 要读一遍权重 14GB：

$$t_{\text{token}} \ge \frac{14\text{ GB}}{1000\text{ GB/s}} = 14\text{ ms} \Rightarrow \le 71 \text{ token/s}$$

而这 14ms 里实际计算量只有约 $2\times7\text{B}=14\text{ GFLOP}$，对几十 TFLOP/s 的卡是「一瞬间」→ **确认 memory-bound**。
结论：单请求加 batch 几乎不增加耗时（权重只读一遍被多请求共享），所以**加大 batch 是 Decode 提吞吐的第一性手段**——这正是连续批的价值。

### 例 4：投机解码加速比

draft 一次猜 4 个、平均接受 2.5 个：理论加速 ≈ 接受数 / 大模型前向次数 ≈ $2.5\times$（扣除 draft 与验证开销后实测常 1.5~2×）。

---

## 常见问题

| 问题 | 答 |
|------|----|
| Prefill 和 Decode 哪个该量化？ | 两个都受益。Decode 更吃带宽，量化（降读取字节）收益更直接。 |
| KV Cache 和 PagedAttention 什么关系？ | KV Cache 是「存什么」，PagedAttention 是「怎么管这块显存」（分页防碎片）。 |
| 连续批会增加单请求延迟吗？ | 几乎不会增加 TPOT（Decode 共享权重读取），还能让请求更快被服务。 |
| 投机解码会改变输出吗？ | 标准实现**无损**（接受/拒绝采样保证分布一致），只快不变质。 |
| TP 和 PP 怎么选？ | 卡间带宽高（NVLink）优先 TP；跨节点带宽低用 PP（通信少但有流水气泡）。 |
| FlashAttention 是近似吗？ | 不是，数学上**精确等价**，只是改了计算顺序与访存方式。 |
| 长上下文最该上哪些？ | FlashAttention（控 N²）+ PagedAttention（控 KV 显存）+ KV 量化 + 序列并行。 |
| 这些优化能同时用吗？ | 能且应该。它们作用在不同层，收益近似相乘。 |

---

## 🔗 跳转链接

- 知识地图总览 → [[00-知识地图]]
- 算子优化（IO-aware 注意力）→ [[llm-optimizer/FlashAttention]]
- 内存优化（KV 缓存与分页）→ [[llm-optimizer/kv-cache]]
- 通信优化（计算通信重叠）→ [[llm-optimizer/计算通信重叠]]

---

### 综述与延伸论文

- LLM Inference Serving: Survey of Recent Advances and Opportunities
- Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems
- [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/pdf/2404.14294)
- [A Survey on Model Compression for Large Language Models](https://arxiv.org/pdf/2308.07633.pdf)
- BlendServe: Optimizing Offline Inference for Auto-regressive Large Models with Resource-aware Batching — https://arxiv.org/pdf/2411.16102
- FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion — https://arxiv.org/abs/2406.06858
- BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching — https://arxiv.org/abs/2412.03594
