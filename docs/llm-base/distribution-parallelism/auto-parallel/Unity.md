# Unity：代数变换与并行化的联合自动优化

> 一句话定位：Unity 把"改图等价变换（代数变换）"和"切图分布式并行"统一成同一张 **PCG（并行计算图）**，在一个搜索空间里一起优化，找到单看任一方面都看不到的全局加速点。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-base/distribution-parallelism/auto-parallel/Alpa]] · [[llm-base/distribution-parallelism/auto-parallel/Flexflow]] · [[llm-base/distribution-parallelism/auto-parallel/README]] · [[ai-infra/网络/集合通信原语]] · [[llm-algo/FLOPs]]

> 论文：Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization（Zhihao Jia 等，OSDI 2022）。前作血脉：TASO / PET（代数变换）+ FlexFlow（SOAP 并行搜索 + 模拟器）。

---

## 阅读地图

| 节 | 你将搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | 联合优化 / PCG |
| 1 | 两条以前各走各的优化路线 | 代数变换 vs 并行化 |
| 2 | 为什么"分开优化"会丢解 | 局部最优陷阱 |
| 3 | PCG：把两者塞进同一张图 | 6 类并行算子 |
| 4 | 等价性如何保证 | substitution / verifier |
| 5 | 三层搜索算法 | 子图替换 + 图切分 + 配置搜索 |
| 6 | 代价模型与模拟器 | delta simulation |
| 7 | 数值手算：通信量/收益估算 | AllReduce / 字节数 |
| 8 | 与 Alpa / FlexFlow 对比 | 谁管什么 |
| FAQ | 易错点表格 | — |

---

## 0. 一句话锚点

传统流程是**两段式**：先用编译器把计算图做**代数等价变换**（如把两个小 matmul 融合成一个大 matmul、把 conv+conv 合并），再把**变换后的固定图**交给并行系统去切分到多卡。

Unity 的洞见：**这两步互相影响**。某个代数变换单独看让单卡变慢，但它改变了张量形状，从而让并行切分的通信骤减——只有把两步放进**同一个搜索空间联合优化**，才能发现这种"先退一步、再进两步"的全局最优。Unity 用一张 **PCG（Parallel Computation Graph，并行计算图）** 统一表达"算什么"和"怎么并行"，让同一套**子图替换（graph substitution）**机制既能做代数变换、也能做并行变换。

```
传统(分离):   代数变换 ──► 固定图 ──► 并行切分      (两次局部最优, 中间墙)
Unity(联合):  ┌───────────── 同一张 PCG ─────────────┐
              │ 代数substitution + 并行substitution  │  (一个搜索空间)
              └──────────────────────────────────────┘
```

---

## 1. 地基：两条本来各走各的优化路线

要懂 Unity，先把它要"缝合"的两件事拆成原子。

### 1.1 代数变换（Algebraic Transformation）

在**单设备**视角下，对计算图做**数学上等价**的改写，让它算得更快。例子：

- 两个连续的逐元素加法 $((x+a)+b) \equiv (x+(a+b))$，常量可预计算。
- 把 `relu(conv(x))` 中的 conv 与相邻 conv 融合（kernel fusion）。
- 矩阵结合律：$(AB)C \equiv A(BC)$，根据形状选乘法顺序，FLOPs 可差几倍。

代表系统：**TASO**（自动用图替换规则搜索等价图）、**PET**（允许"部分等价 + 修正"的更激进变换）。

### 1.2 并行化（Parallelization）

把一个算子/张量沿某个维度切开，分到多卡，引入集合通信把结果拼回来。这正是 [[llm-base/distribution-parallelism/auto-parallel/Flexflow]] 的 **SOAP** 四维度（Sample / Operator / Attribute / Parameter），也是 [[llm-base/distribution-parallelism/auto-parallel/Alpa]] 的 intra-op / inter-op 两级。

```
代数变换空间                 并行化空间
  图长什么样               图怎么切到多卡
(TASO/PET管这个)         (FlexFlow/Alpa管这个)
        \                     /
         \                   /
          ▼                 ▼
        Unity: 合二为一的搜索
```

---

## 2. 为什么"分开优化"会丢解（核心动机）

关键：**代数变换会改变张量的形状和算子结构，而并行的通信代价高度依赖形状**。两步分离时，第一步只用"单卡更快"当目标，看不到第二步的通信账。

一个被反复引用的直觉例子（矩阵乘 + 转置场景）：

```
原图:                          代数等价变换后:
  A·B  然后  转置(A·B)            先 转置(B)·转置(A)  (= 转置(A·B))
  └ 切 B 列做并行              └ 切法变了, 通信张量更小/更对齐
  通信: 中等                     通信: 更小

单卡角度: 变换后可能 FLOPs 略增  ← 两段式编译器会"拒绝"这个变换
全局角度: 通信骤降, 端到端更快   ← 只有联合优化才会"接受"它
```

> 一句话：**最优代数图依赖于最优并行方案，反之亦然；二者是耦合的，不能贪心地先定一个再定另一个**。这与 [[llm-base/distribution-parallelism/auto-parallel/README]] 里反复强调的"组合空间爆炸 + 耦合"问题一脉相承。

---

## 3. PCG：把代数与并行塞进同一张图

Unity 的统一抽象是 **Parallel Computation Graph（PCG）**。普通计算图里：节点 = 数学算子（matmul/conv/...），边 = 张量。PCG **额外引入一类"并行算子"作为图中的真实节点**，于是"怎么并行"也变成了图结构的一部分，能被同一套图替换机制操作。

PCG 中的并行算子（直观分 6 类，描述其语义，具体命名以论文为准）：

| 并行算子 | 作用 | 对应通信原语 |
|----------|------|--------------|
| **Partition（切分）** | 沿某维把张量切成 N 份分发 | scatter / 本地切片 |
| **Combine（合并）** | 把 N 份切片在某维拼回完整张量 | all-gather |
| **Replicate（复制）** | 把张量复制到多设备 | broadcast |
| **Reduce（规约）** | 把多设备的部分和相加 | all-reduce / reduce-scatter |
| **Pipeline（流水）** | 把子图划成 stage 跨设备流水 | P2P send/recv |
| **MapReduce/重排** | 跨 mesh 的 resharding | all-to-all 等 |

```
普通图节点:   [matmul] ── tensor ── [relu]

PCG (把并行也画进去):
   [Partition d=cols]
        │  (切成2片)
   ┌────┴────┐
[matmul]  [matmul]      ← 两卡各算一半
   └────┬────┘
   [Combine d=cols]     ← all-gather 拼回
        │
     [relu]
```

**为什么这是关键设计**：一旦"并行"是图里的节点，那么"代数变换"和"并行变换"就都退化为**同一种操作——子图替换（graph substitution）**。Unity 不再需要两套优化器，一套替换引擎通吃。

---

## 4. 等价性如何保证：substitution + verifier

Unity 的所有优化都表达为 **graph substitution（子图替换）规则**：`匹配某个子图模式 → 替换成另一个等价子图`。

- **代数替换**：如 `matmul(matmul(A,B),C) → matmul(A,matmul(B,C))`。
- **并行替换**：如 `matmul → Partition; matmul×N; Combine`（把单算子换成并行版）。
- **混合替换**：并行算子之间也能等价改写，例如 `Combine→Partition`（先合再切）在某些维度上可对消、或换成更便宜的 `Reduce`。

**正确性**：替换规则不是人工凭直觉写死的，而是可由系统**自动生成 + 自动验证**（沿用 TASO 的思路）：用随机数值在小张量上**实测两边输出是否一致**，再辅以符号化检查，确保替换在数学上等价。这样既能扩出海量规则，又不会引入错误梯度。

```
substitution 引擎(只此一套):
   ┌─ 代数规则:  结合律/分配律/融合 ……
   ├─ 并行规则:  Partition/Combine/Reduce 插入与消解
   └─ 混合规则:  并行算子互相对消/下沉
        ↓ 每条规则都过 verifier(随机数值+符号)
     保证: 替换后图 ≡ 原图 (数学等价)
```

---

## 5. 三层搜索算法（怎么在巨大空间里找解）

PCG 把空间统一了，但空间也因此爆炸。Unity 用**分层搜索**控制规模：

```
┌──────────────────────────────────────────────┐
│ ① 图替换搜索 (graph substitution)             │
│    在 PCG 上反复套用等价规则, 生成候选图       │
│    (代数+并行混在一起搜, 这是 Unity 的灵魂)    │
├──────────────────────────────────────────────┤
│ ② 图切分 (graph split)                        │
│    大 PCG 太大搜不动 → 切成子图分而治之        │
│    (类似 Alpa 把全图切 stage 的思路)           │
├──────────────────────────────────────────────┤
│ ③ 设备/配置搜索                                │
│    每个子图选并行度、维度、设备映射             │
│    用代价模型/模拟器评分, 选端到端最优          │
└──────────────────────────────────────────────┘
        ▲ 三层交替迭代, 上层用下层的评分反馈
```

- **①** 是 Unity 相对前作的核心增量：代数与并行规则**在同一轮替换里交错应用**，于是能走出第 2 节那种"单卡变慢、全局变快"的路径。
- **②③** 借鉴 FlexFlow/Alpa：用**图切分 + 配置枚举**把指数空间压到可解。

对比记忆：[[llm-base/distribution-parallelism/auto-parallel/Alpa]] 是 **DP(切 stage) + ILP(切 op)**；Unity 是 **substitution(改图+并行) + split + 配置搜索**，多了"改图"这一维。

---

## 6. 代价模型与模拟器（如何给候选打分）

搜索每一步都要"这个图在这套硬件上要跑多久"。Unity 沿用并增强 **FlexFlow 式执行模拟器**：

- **构建 timeline**：按 PCG 拓扑模拟每个算子的开始/结束时间，得到端到端 iteration 时间。
- **算子时间从 profile 表查**：同 input-size 的同类算子执行时间近似不变 → 一次 profile，多次复用（见 [[llm-base/distribution-parallelism/auto-parallel/Flexflow]]）。
- **通信时间**：由"通信张量字节数 / 链路带宽 + 固定延迟"估计，区分卡内 NVLink 与卡间网络（见 [[ai-infra/网络/集合通信原语]]）。
- **delta simulation（增量模拟）**：一次替换只改了图的一小块，**只重算受影响的子时间线**，不从头模拟整张图 → 把单次评分从"秒级"降到"毫秒级"，这是搜索能跑起来的关键工程点。

```
候选图 ──► [模拟器] ──► 预测 iter 时间
               ▲                │
               └── delta: 只重算改动子图(快3个数量级)
```

---

## 7. 数值手算：通信量与收益怎么估

并行的收益最终落在 **FLOPs 摊薄** 与 **通信开销** 的拉锯上。下面用一个张量并行片段把账算清楚（思路同 [[llm-algo/FLOPs]] 与 [[llm-base/distribution-parallelism/tensor-parallel/tensor-parallel]]）。

### 7.1 一个 matmul 切列并行的通信量

设 $Y = XW$，$X\in\mathbb{R}^{B\times K}$，$W\in\mathbb{R}^{K\times N}$，输出 $Y\in\mathbb{R}^{B\times N}$。用 fp16（2 字节）。沿 $N$（列）切到 $P$ 卡：每卡算 $Y_{:, n/P}$，最后 **all-gather** 拼成完整 $Y$。

- 完整 $Y$ 的字节数：$B\cdot N\cdot 2$。
- Ring all-gather 单卡进出流量 $\approx \dfrac{P-1}{P}\cdot(B\cdot N\cdot 2)$。

代入 $B=4096,\ N=4096,\ P=8$：

$$
Y_{\text{bytes}} = 4096\times4096\times2 = 33{,}554{,}432\ \text{B} \approx 32\ \text{MB}
$$
$$
\text{all-gather}\approx \frac{7}{8}\times32\ \text{MB} = 28\ \text{MB}\ (\text{每卡})
$$

若卡间带宽 $\approx 200\ \text{GB/s}$，则通信时间 $\approx \dfrac{28\times10^6}{200\times10^9}\approx 0.14\ \text{ms}$。

### 7.2 联合优化如何省钱（手算对比）

假设有个代数变换：把上面"切列 + all-gather"改写成"**切行 + all-reduce 部分和**"的等价路径，且变换顺手把一个相邻转置消掉，使需要通信的张量从 $Y(B\times N)$ 变成更小的 $Z(B\times N/2)$（形状对齐后维度减半）。

- 变换前通信：$\approx 28\ \text{MB/卡}$，$0.14\ \text{ms}$。
- 变换后通信张量减半：$\approx 14\ \text{MB/卡}$，$\approx 0.07\ \text{ms}$。
- 代价：该代数变换让**单卡 FLOPs 多了约 10%**。设单卡该段计算 $\approx 0.5\ \text{ms}$，多算 $0.05\ \text{ms}$。

净账：$\underbrace{-0.07\,\text{ms}}_{\text{省通信}} + \underbrace{+0.05\,\text{ms}}_{\text{多算}} = -0.02\ \text{ms}$，**端到端更快**。

```
两段式编译器视角:  "FLOPs +10% ⇒ 拒绝变换"   → 停在 0.14ms 通信
Unity 联合视角:    "+0.05算 −0.07通信 = 净赚" → 接受变换, 落到更优解
```

> 这正是 Unity 的存在理由：**通信账只有把并行也建模进来才看得见**，分离优化天然丢掉这类解。注意：以上数字为**说明性手算**（带宽/比例为示意），真实收益以具体模型+集群 profile 为准。

### 7.3 FLOPs 摊薄侧的直觉

一个 $B\times K \times N$ 的 matmul，FLOPs $\approx 2BKN$。切到 $P$ 卡后单卡 $\approx \dfrac{2BKN}{P}$。只要 **通信时间 < 计算节省的时间**，并行就划算；Unity 的搜索本质就是在 PCG 上为每个算子找这个不等式成立、且**叠加代数变换后总账最优**的配置。

---

## 8. 与 FlexFlow / Alpa 的分工对比

| 维度 | FlexFlow | Alpa | **Unity** |
|------|----------|------|-----------|
| 优化对象 | 仅并行（SOAP） | 仅并行（intra/inter-op） | **代数变换 + 并行（联合）** |
| 统一抽象 | 算子图 + 设备拓扑 | HLO + device mesh | **PCG（并行算子入图）** |
| 核心算法 | MCMC 随机搜索 | DP + ILP 两级 | substitution + split + 配置搜索 |
| 等价性来源 | — | — | **自动生成+验证的替换规则** |
| 评分 | 执行模拟器 | cost model（通信量/带宽） | 模拟器 + delta simulation |
| 改图能力 | 否 | 否 | **是（灵魂特性）** |

记忆钩子：**FlexFlow 给了模拟器，Alpa 给了两级搜索，Unity 在它们之上多加了"改图"这一维，并用 PCG 把"改图"和"切图"统一成同一动作。**

```
能力叠层:
  FlexFlow ── 模拟器/SOAP
      └─ Alpa ── 两级搜索(DP+ILP)/cross-mesh
            └─ Unity ── 代数变换 ⊕ 并行  (PCG 统一 + 联合搜索)
```

---

## 常见问题

| 问题 | 答 |
|------|----|
| Unity 一句话贡献是什么？ | 用 PCG 把"代数等价变换"和"并行化"统一进同一搜索空间联合优化，发现分离优化看不到的全局最优。 |
| 为什么非得"联合"？分开做不行？ | 最优代数图与最优并行方案**相互依赖**；分离优化是贪心，会丢掉"单卡变慢但通信骤降"的解（见 §2、§7.2）。 |
| PCG 和普通计算图差在哪？ | PCG 把 Partition/Combine/Reduce 等**并行算子也当作真实图节点**，于是并行也能被图替换操作。 |
| 怎么保证改完图结果还对？ | 每条 substitution 规则经**随机数值实测 + 符号验证**确认数学等价（承袭 TASO）。 |
| 搜索空间爆炸怎么办？ | 分层：图替换搜索 → 图切分 → 配置搜索；评分用**delta simulation** 只重算改动部分。 |
| 和 Megatron 手工并行的关系？ | Megatron 是人工指定固定切法；Unity 是自动搜索，且能叠加代数变换，理论上能覆盖并超过手工方案。 |
| 适合谁用？ | 模型结构多变、形状不规则、想榨干异构集群带宽的训练场景；规整的标准 Transformer 上手工方案也已很强。 |

> 工具/接口细节（具体 API、规则集规模、支持的算子覆盖度）随版本演进，**以官方论文与代码仓库为准**。

---

## 🔗 跳转链接

- 同目录自动并行家族：[[llm-base/distribution-parallelism/auto-parallel/Alpa]] · [[llm-base/distribution-parallelism/auto-parallel/Flexflow]] · [[llm-base/distribution-parallelism/auto-parallel/Galvatron]] · [[llm-base/distribution-parallelism/auto-parallel/gspmd]] · [[llm-base/distribution-parallelism/auto-parallel/Mesh-Tensorflow]] · [[llm-base/distribution-parallelism/auto-parallel/README]]
- 并行基础：[[llm-base/distribution-parallelism/tensor-parallel/tensor-parallel]] · [[llm-base/distribution-parallelism/pipeline-parallelism/README]] · [[llm-base/distribution-parallelism/multidimensional-hybrid-parallel/README]] · [[llm-inference/大模型推理张量并行]]
- 代价模型相关原子：[[llm-algo/FLOPs]] · [[ai-infra/网络/集合通信原语]] · [[llm-optimizer/计算通信重叠]] · [[ai-infra/算力/GPU工作原理]]
- 框架落地：[[ai-framework/megatron-lm/README]] · [[ai-framework/deepspeed/README]] · [[llm-train/README]]
- 全局导航：[[00-知识地图]]

---

### 参考资料

- Unity (OSDI 2022)：https://www.usenix.org/conference/osdi22/presentation/unger
- Unity：通过代数变换和并行化的联合优化加速 DNN 训练：https://www.victorlamp.com/article/7387511088
- 【论文赏读】Unity: Accelerating DNN Training Through Joint Opt of Algebraic Transform and Parallelization：https://zhuanlan.zhihu.com/p/560247608
- 前作 TASO / PET（代数变换基础）、FlexFlow（并行搜索 + 模拟器基础）
