# DeepSpeed

> 微软开源的大模型训练/推理"省显存 + 提速"引擎；核心是 ZeRO（把模型状态切片分散到多卡，让单卡也能放下巨型模型），外加 offload、3D 并行、对齐训练（DeepSpeed-Chat）一整套工具箱。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/megatron-lm/README]] [[llm-train/README]] [[ai-framework/pytorch/README]]

## 阅读地图

| 你想搞清楚的问题 | 跳到哪节 |
| --- | --- |
| 训练一个模型到底占多少显存？为什么 OOM？ | §1 显存账的"原子拆解" |
| ZeRO-1/2/3 各切什么？省多少？ | §2 ZeRO 三段切分 |
| 显存还不够怎么办（搬到 CPU/NVMe）？ | §3 Offload |
| 单纯切分不够，怎么和张量/流水并行组合？ | §4 3D 并行 |
| 怎么用它做 RLHF/对齐？ | §5 DeepSpeed-Chat |
| 我该用 ZeRO 几？什么时候别用？ | §6 选型决策 |
| 一个 13B 模型 8 卡的显存到底够不够？ | §7 数值手算 |
| ZeRO-3 和 PyTorch FSDP 到底啥关系？ | §8 对照表 |
| 踩坑/高频疑问 | §9 常见问题 |

## 0. 一句话锚点

数据并行（DDP）每张卡都存一份**完整**的模型参数、梯度、优化器状态——这部分叫**模型状态（model states）**，是大模型显存的大头。**ZeRO（Zero Redundancy Optimizer，零冗余优化器）的核心思想只有一句：这三样东西每张卡只存 $1/N$ 片，需要时再临时凑齐。** 别的全是工程细节。

## 1. 地基：训练显存到底花在哪（原子拆解）

不先把"显存账"算清楚，后面所有"省显存"都是空中楼阁。训练时显存分四块：

```
┌──────────────────────────────────────────────────────────┐
│  GPU 显存占用 = 模型状态 + 激活值 + 临时缓冲 + 碎片         │
│                                                            │
│  模型状态 (model states)  ← ZeRO 专门优化这块               │
│   ├─ 参数 Parameters   (P)                                 │
│   ├─ 梯度 Gradients    (G)                                 │
│   └─ 优化器状态 Optimizer states (O)                       │
│                                                            │
│  激活值 (activations)     ← 重计算/checkpoint 优化这块      │
│  临时缓冲 (buffers) + 显存碎片 (fragmentation)              │
└──────────────────────────────────────────────────────────┘
```

**为什么模型状态这么大？** 关键在混合精度训练（fp16/bf16 + fp32）。设参数量为 $\Psi$（个数），用 Adam 优化器、fp16 训练：

- **参数（fp16）**：$2\Psi$ 字节（每个数 2 字节）
- **梯度（fp16）**：$2\Psi$ 字节
- **优化器状态（fp32）**：Adam 要存 3 份 fp32——主参数副本、动量 momentum、方差 variance，即 $4\Psi + 4\Psi + 4\Psi = 12\Psi$ 字节

合计每个参数要 $2+2+12 = 16$ 字节。所以经典公式：

$$\text{模型状态显存} = 16\Psi \text{ 字节} = \frac{16\Psi}{10^9}\text{ GB（按 } \Psi \text{ 为参数个数）}$$

**手算锚点**：一个 7.5B（$\Psi = 7.5\times10^9$）的模型，仅模型状态就要 $16 \times 7.5 = 120$ GB。一张 A100 只有 80GB——**一张卡根本放不下**。这就是 ZeRO 要解决的问题。

而 DDP（PyTorch 默认数据并行）的做法是：**每张卡都存这 120GB 的完整副本**。8 张卡 = 8 份重复的 120GB。这就是"冗余（redundancy）"，ZeRO 要"归零（zero）"的就是它。

## 2. ZeRO 三段切分（本引擎的灵魂）

ZeRO 把模型状态 $P/G/O$ 逐级切片到 $N$ 张卡。设并行度（卡数）为 $N$：

```
DDP（基线，每卡全量）        每卡显存 = 16Ψ
┌────┐┌────┐┌────┐┌────┐
│PGOO││PGOO││PGOO││PGOO│   ← 4 张卡，每张都全量复制
└────┘└────┘└────┘└────┘

ZeRO-1：只切优化器状态 O      每卡 = 4Ψ + 12Ψ/N
┌────┐┌────┐┌────┐┌────┐
│PG o││PG o││PG o││PG o│   ← O 被切成 4 片，P/G 仍全量
└────┘└────┘└────┘└────┘

ZeRO-2：切 O + 梯度 G         每卡 = 2Ψ + 14Ψ/N
┌────┐┌────┐┌────┐┌────┐
│P g o││P g o││...│ │...│   ← G、O 都切片，P 仍全量
└────┘└────┘└────┘└────┘

ZeRO-3：切 O + G + 参数 P     每卡 = 16Ψ/N
┌────┐┌────┐┌────┐┌────┐
│p g o││p g o││...│ │...│   ← 三样全切，每卡只剩 1/N
└────┘└────┘└────┘└────┘
（小写 = 切片，大写 = 全量）
```

逐档显存公式（$N$ = 数据并行度）：

| 阶段 | 切什么 | 每卡模型状态显存 | 通信量（相对 DDP） |
| --- | --- | --- | --- |
| DDP | 不切 | $16\Psi$ | $2\Psi$（一次 all-reduce） |
| **ZeRO-1** | O | $4\Psi + \dfrac{12\Psi}{N}$ | $2\Psi$（基本不变） |
| **ZeRO-2** | O+G | $2\Psi + \dfrac{14\Psi}{N}$ | $2\Psi$（基本不变） |
| **ZeRO-3** | O+G+P | $\dfrac{16\Psi}{N}$ | $3\Psi$（约 1.5×） |

### 2.1 为什么 ZeRO-1/2 通信几乎不增加？

关键洞察：DDP 本来就要做梯度 **all-reduce**。ZeRO 把 all-reduce 拆成 **reduce-scatter（每卡只收自己负责那片的梯度和）+ all-gather（更新完参数再凑齐）** 两步，总通信量和 all-reduce 相同（$2\Psi$）。所以 ZeRO-1/2 是"**几乎免费的午餐**"——显存大降，速度几乎不掉。

```
DDP 梯度同步:   all-reduce(G)                      = 2Ψ 通信
ZeRO-2 梯度同步: reduce-scatter(g) + all-gather(p)  = Ψ + Ψ = 2Ψ 通信
（数学上等价，只是把一次 all-reduce 拆成两半）
```

### 2.2 为什么 ZeRO-3 要多一份通信？

ZeRO-3 连**参数**都切了，每卡平时只有 $1/N$ 的参数。但前向/反向要算某层时，**必须先把这层的完整参数临时 all-gather 凑齐**，算完立刻丢弃（释放显存）。所以多了一次参数 all-gather（$\Psi$），总通信约 $3\Psi$。

```
ZeRO-3 前向计算第 k 层的时序：
  ① all-gather 第 k 层参数（从各卡收齐）→ 临时占满
  ② 计算这一层 forward
  ③ 立即释放第 k 层参数（只留自己那 1/N 片）
  ④ 进入第 k+1 层，重复 ①②③
                  ↑ 用"时间换空间"：参数随用随取、用完即弃
```

这就是 ZeRO-3 的灵魂：**参数不常驻，按层流式 gather**。代价是通信变多、对带宽敏感（NVLink/InfiniBand 越快越好）。

## 3. Offload：显存还不够就往 CPU/NVMe 搬

即便 ZeRO-3 把每卡降到 $16\Psi/N$，超大模型仍可能放不下。**ZeRO-Offload / ZeRO-Infinity** 把"暂时不用"的数据搬到更便宜、更大但更慢的存储层级：

```
存储金字塔（容量↑ 速度↓ 价格↓）
        ┌─────────────┐
        │ GPU HBM 80G │  ←最快(TB/s)，最贵，最小
        ├─────────────┤
        │ CPU DRAM 1T │  ←ZeRO-Offload：优化器状态/梯度搬这里
        ├─────────────┤
        │ NVMe SSD 8T │  ←ZeRO-Infinity：连参数也能搬到 SSD
        └─────────────┘
```

- **ZeRO-Offload**：把**优化器状态 + 优化器计算（Adam 的 step）** 放到 CPU。GPU 算前向/反向，CPU 算参数更新，两者流水重叠。能在单卡上训练 10B+ 级别模型。
- **ZeRO-Infinity**：进一步把参数也卸到 NVMe，理论上单机能"训练"万亿参数（速度很慢，但能跑通）。

**权衡**：搬运走 PCIe（约几十 GB/s）比 HBM（约 2 TB/s）慢一两个数量级。**Offload 是"以速度换显存的最后手段"**——能在 HBM 里放下就别 offload。

## 4. 3D 并行：ZeRO 不是万能的

ZeRO 本质是**数据并行（DP）的省显存改良版**。但超大模型（百亿~万亿）光靠 DP 不够，要叠三种并行（DeepSpeed 与 Megatron-LM 联合实现，见 [[ai-framework/megatron-lm/README]]）：

```
3D 并行 = 数据并行(DP) × 张量并行(TP) × 流水线并行(PP)

         ┌── DP 维度（ZeRO 切模型状态）──┐
         │                              │
   PP ───┼── GPU0  GPU1  GPU2  GPU3 ────┤  每个 DP 组内
   流水  │   GPU4  GPU5  GPU6  GPU7      │  再叠 TP×PP
   层切  └──────────────────────────────┘
              └── TP 维度（切单层矩阵）──┘
```

| 并行方式 | 切什么 | 解决的瓶颈 | 通信特征 |
| --- | --- | --- | --- |
| 数据并行 DP（含 ZeRO） | 切 batch / 模型状态 | 显存 + 吞吐 | all-reduce/gather，可跨机 |
| 张量并行 TP | 切单层的权重矩阵 | 单层太大放不下 | 每层都通信，须高带宽（机内 NVLink） |
| 流水线并行 PP | 按层切成多段（stage） | 层数太多、跨机扩展 | 段间传激活，通信少，可跨机 |

**经验摆放**：TP 放机内（吃 NVLink 带宽）、PP 跨机（通信少）、DP/ZeRO 最外层。这正是训练 GPT-3/百亿大模型的标准配方。

## 5. DeepSpeed-Chat：把 ZeRO 用到 RLHF 对齐

RLHF（基于人类反馈的强化学习，见 [[llm-train/README]]）是大模型对齐的主流路线，但工程上很重，因为同时要管多个模型。DeepSpeed-Chat 把三阶段流水线打包：

```
DeepSpeed-Chat 三阶段（对齐流水线）
①SFT 监督微调 → ②RM 奖励模型 → ③RLHF(PPO 强化学习)
                                    │
        PPO 阶段同时活着 4 个模型：  │
        ┌──────────────────────────▼─────────────────┐
        │ Actor(被训练) │ Critic(被训练)              │
        │ Reference(冻结，算KL) │ Reward(冻结，给分)   │
        └────────────────────────────────────────────┘
          ↑ 4 个模型同驻显存 → 显存爆炸 → 正是 ZeRO 的舞台
```

亮点：

- **Hybrid Engine（混合引擎）**：PPO 一轮里既要"生成"（推理，要快）又要"训练"（反向，要省显存）。混合引擎在**生成阶段切到推理优化的并行布局**、**训练阶段切回 ZeRO 布局**，避免两者互相拖累。
- 冻结模型（Reference/Reward）用 ZeRO-3 切片或 offload，进一步压显存。

## 6. 选型决策：我该用 ZeRO 几？

```
            模型能放进单卡 HBM 吗？
                    │
        ┌───────是──┴──否────────┐
        ▼                        ▼
   普通 DDP / ZeRO-1     单卡放得下"一层"吗？
   （最快，省心）          ┌────是────┴──否────┐
                          ▼                   ▼
              ZeRO-2 → 还 OOM？           必须切层 → TP/PP
                 │                        （3D 并行 + ZeRO-3）
            ┌─否─┴─是─┐
            ▼         ▼
         ZeRO-2    ZeRO-3 → 还不够？→ +Offload(CPU)
        （稳）              → 还不够？→ +Infinity(NVMe)
```

口诀：**能 ZeRO-2 就别 ZeRO-3（通信少），能不 offload 就别 offload（速度快），HBM 是越用满越好的资源**。

## 7. 数值手算：13B 模型，8×A100(80G) 够吗？

设 $\Psi = 13\times10^9$，Adam + fp16，$N=8$ 卡。只看模型状态（$16\Psi = 208$ GB 总量）：

| 方案 | 每卡模型状态 | 计算 | 单卡 80G 够吗 |
| --- | --- | --- | --- |
| DDP | $16\Psi = 208$ GB | 每卡全量 | ❌ 远超 |
| ZeRO-1 | $4\Psi + 12\Psi/8$ | $52 + 19.5 = 71.5$ GB | ⚠️ 勉强，激活一加就爆 |
| ZeRO-2 | $2\Psi + 14\Psi/8$ | $26 + 22.75 = 48.75$ GB | ✅ 留 31G 给激活 |
| ZeRO-3 | $16\Psi/8$ | $208/8 = 26$ GB | ✅ 留 54G 给激活/batch |
| ZeRO-3 + CPU offload | ≪ 26 GB | O/G 搬 CPU | ✅ 还能开更大 batch |

**结论**：13B 用 ZeRO-2 即可舒服训练，要开大 batch / 长序列就上 ZeRO-3。再上 65B 这种，单卡 ZeRO-3 也是 $16\times65/8=130$ GB > 80G，**必须叠 TP/PP（3D 并行）或加 offload**。

**通信量手算**：ZeRO-3 跑一步，参数 all-gather + 梯度 reduce-scatter + 参数 all-gather ≈ $3\Psi = 3\times13\times2\text{B} = 78$ GB 数据要在卡间流动。这就是为什么 ZeRO-3 在没有 NVLink/IB 的廉价机器上会"显存够了但慢得离谱"。

## 8. 对照表：ZeRO-3 vs PyTorch FSDP

二者**思想几乎一样**（参数/梯度/优化器全切片，按需 gather），FSDP 实际上是 PyTorch 把 ZeRO-3 的理念吸收进官方（见 [[ai-framework/pytorch/README]]）。

| 维度 | DeepSpeed ZeRO-3 | PyTorch FSDP |
| --- | --- | --- |
| 切分思想 | 参数/梯度/优化器全切片 | 同（FullyShard） |
| 出身 | 微软外部库，先行者 | PyTorch 官方原生 |
| 配置方式 | JSON 配置 `ds_config` | Python API / `accelerate` |
| Offload | CPU + NVMe（Infinity 最强） | 支持 CPU offload，NVMe 较弱 |
| 3D 并行 | 与 Megatron 深度集成，成熟 | 靠 `device_mesh`/组合，演进中 |
| 对齐工具 | DeepSpeed-Chat 现成 | 需自己拼 |
| 生态/上手 | 配置项多、文档以官方为准 | 与原生训练循环更贴合 |

一句话：**要极致省显存 + Infinity offload + 现成 RLHF，选 DeepSpeed；要和 PyTorch 原生栈无缝、少装外部依赖，选 FSDP。** 二者性能在多数场景接近，按团队栈和习惯选。

> 注：以上为稳定机制对比，具体配置项/CLI 参数/API 签名以各自官方文档为准，不同版本可能不同。

## 9. 常见问题

| 问题 | 一句话答案 |
| --- | --- |
| ZeRO 和数据并行什么关系？ | ZeRO 就是"零冗余版的数据并行"，不改变 DP 的语义，只是不再每卡存全量。 |
| ZeRO-1/2/3 选哪个最划算？ | ZeRO-2 通常性价比最高（省显存多、通信几乎不增）；放不下才上 ZeRO-3。 |
| 为什么开了 ZeRO-3 反而变慢？ | 多了参数 all-gather 通信，带宽不足（无 NVLink/IB）时被通信卡住。 |
| Offload 一定能省到？ | 能，但 PCIe 慢一两个数量级，是"以速度换显存"的兜底手段。 |
| 模型状态省了，激活值怎么办？ | 那是另一条线：用激活重计算（gradient checkpointing）+ 序列并行，与 ZeRO 正交叠加。 |
| ZeRO-3 等于把模型切开了吗？ | 不是张量并行式的"切层内矩阵"，而是把"整层参数"散在各卡、用时再聚齐，DP 语义不变。 |
| 单卡能用 ZeRO 吗？ | 单卡 ZeRO 无切分意义，但 ZeRO-Offload 单卡仍能把优化器搬 CPU，省下大量 HBM。 |
| ZeRO 和 Megatron 冲突吗？ | 不冲突，互补：Megatron 管 TP/PP，ZeRO 管最外层 DP，组成 3D 并行。 |

## 🔗 跳转链接

- [[00-知识地图]] —— 全局总览，先看这里定位 DeepSpeed 在训练栈中的位置
- [[ai-framework/megatron-lm/README]] —— 张量并行/流水线并行，与 ZeRO 组成 3D 并行
- [[llm-train/README]] —— 训练全流程（含 SFT/RLHF），DeepSpeed-Chat 的上层场景
- [[ai-framework/pytorch/README]] —— FSDP 原生分片，与 ZeRO-3 思想对照
