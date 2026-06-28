# 分布式深度学习路线图（Distributed DL Roadmap）

> 一句话定位：把"单卡装不下、单卡跑太慢"的大模型训练，系统拆成**数据/流水线/张量/序列/MoE/多维混合/自动**七类并行，并落到 DeepSpeed / Megatron-LM / Alpa 三大框架的学习地图。
> 📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[llm-algo/transformer/模型架构]] · [[llm-algo/FLOPs]] · [[llm-optimizer/计算通信重叠]] · [[llm-algo/moe/README]]

## 阅读地图

| 节 | 内容 | 核心问题 | 关键数字/公式 |
|---|---|---|---|
| 0 | 一句话锚点 | 为什么需要分布式 | 模型显存 ≫ 单卡显存 |
| 1 | 地基：单卡显存账本 | 一张卡到底装了什么 | 参数+梯度+优化器≈ $16\Psi$ 字节 |
| 2 | 集合通信原语 | 卡与卡怎么对话 | AllReduce $=2(N-1)/N$ |
| 3 | 数据并行 DP/DDP | 复制模型、切分数据 | 通信量 $\approx 2\Psi$ |
| 4 | ZeRO（DeepSpeed） | 把冗余的状态切开 | 显存 $16\Psi \to 16\Psi/N$ |
| 5 | 流水线并行 PP | 按层切，造流水线 | 气泡率 $(p-1)/(m+p-1)$ |
| 6 | 张量并行 TP | 把一个矩阵切开 | 每层 2 次 AllReduce |
| 7 | 序列并行 SP | 切 LayerNorm/Dropout 的序列维 | 省激活显存 |
| 8 | MoE 并行 | 专家分到不同卡 | All-to-All 路由 |
| 9 | 3D/多维混合并行 | DP×PP×TP 组合 | $N=d\times p\times t$ |
| 10 | 自动并行（Alpa） | 让编译器找切法 | inter/intra-op |
| 11 | 框架与模型对照 | 学什么、怎么练 | DeepSpeed/Megatron |
| — | 数值手算 | 175B 怎么放进集群 | 端到端估算 |

## 0. 一句话锚点

大模型训练的全部分布式技术，都在回答同一个矛盾：

> **一个模型（及其训练状态）太大、计算太多，一张 GPU 既装不下也算不快。**

解决思路只有两个原子动作：

- **切（partition）**：把"放不下的东西"分到多张卡 —— 切数据、切层、切矩阵、切序列、切专家。
- **传（communicate）**：切开后各卡只有局部，必须用**集合通信**把结果对齐回来。

**所有并行 = 一种切法 + 一种通信原语。** 记住这句，下面每一节都只是它的实例。

```
            放不下 / 算太慢
                  │
        ┌─────────┴─────────┐
       切                   传
   (partition)         (communicate)
   ┌──┬──┬──┬──┬──┐    ┌──────┬──────┐
   数 层 矩 序 专      AllReduce  All-to-All
   据    阵 列 家      Broadcast  AllGather
   DP PP TP SP MoE    ReduceScatter
```

## 1. 地基：单卡显存账本（一切的起点）

不理解显存构成，就理解不了为什么要切。设模型参数量为 $\Psi$（个数），混合精度（fp16 计算 + fp32 主权重）训练时，单卡常驻显存：

| 项 | 精度 | 每参数字节 | 说明 |
|---|---|---|---|
| fp16 参数 | 2 字节 | $2\Psi$ | 前向/反向用 |
| fp16 梯度 | 2 字节 | $2\Psi$ | 反向产出 |
| fp32 主参数 | 4 字节 | $4\Psi$ | 优化器更新用 |
| fp32 momentum | 4 字节 | $4\Psi$ | Adam 一阶矩 |
| fp32 variance | 4 字节 | $4\Psi$ | Adam 二阶矩 |

合计 **模型状态 $\approx (2+2+4+4+4)\Psi = 16\Psi$ 字节**（Adam 混合精度的经典账）。

```
 单参数的"随身行李"(Adam, 混合精度):
 ┌────┬────┬────────┬────────┬────────┐
 │fp16│fp16│ fp32   │ fp32 m │ fp32 v │
 │参数│梯度│主参数  │一阶矩  │二阶矩  │
 │ 2B │ 2B │  4B    │  4B    │  4B    │  = 16 B / 参数
 └────┴────┴────────┴────────┴────────┘
```

**手算**：GPT-3 175B → $16 \times 175\times10^9 = 2.8\times10^{12}$ B $= 2800$ GB。一张 A100（80GB）只能装 $80/2.8 \approx 1/35$。**结论：不切根本不可能训。**

> 还有第二大头 —— **激活值（activation）**，与 batch、序列长度成正比，是序列并行/重计算要对付的对象（见第 7 节）。详见 [[llm-algo/transformer/模型架构]] 与 [[transformer内存估算]]。

## 2. 地基：集合通信原语（卡与卡的对话方式）

切开后必须传。一切分布式训练只用到几种**集合通信（collective）**原语，详见 [[ai-infra/网络/集合通信原语]]：

```
 4 张卡, 每卡一块数据 a/b/c/d:

 AllReduce      : 每卡都拿到 a+b+c+d        (DP 同步梯度)
 ┌─a─┐┌─b─┐┌─c─┐┌─d─┐ →  每卡: a+b+c+d

 ReduceScatter  : 求和后"各分一段"          (ZeRO/环算法的前半)
 AllGather      : 各持一段 → 每卡拿到全部    (ZeRO 取参数/环算法后半)
 Broadcast      : 一卡 → 所有卡             (发初始权重)
 All-to-All     : 卡 i 的第 j 段 → 卡 j     (MoE 路由 token)
```

**关键成本公式（Ring-AllReduce）**：$N$ 卡、数据量 $V$，总传输 $= 2\frac{N-1}{N}V$，**与卡数几乎无关**（这是它能扩展的原因）。其中 ReduceScatter 传 $\frac{N-1}{N}V$，AllGather 再传 $\frac{N-1}{N}V$。

```
 Ring-AllReduce (N=4, 环形):
 GPU0 → GPU1 → GPU2 → GPU3
   ↑___________________↓
 阶段1 ReduceScatter: 转 N-1 步, 每步传 V/N
 阶段2 AllGather    : 转 N-1 步, 每步传 V/N
 总量 = 2(N-1)/N · V  ←  N 大时趋近 2V
```

## 3. 数据并行 DP / DDP（最常用的入门并行）

**切法**：切数据，不切模型。每张卡**复制一份完整模型**，各吃不同的 mini-batch，反向后用 **AllReduce 同步梯度**。

```
 全局 batch = 4×B, 4 卡各拿 B:

 GPU0  GPU1  GPU2  GPU3
 [模型] [模型] [模型] [模型]   ← 完全相同的副本
   │     │     │     │
  本地梯度 g0  g1   g2   g3
   └─────┴──AllReduce──┴────┘
        g = (g0+g1+g2+g3)/4   ← 每卡同步成相同梯度
   各自用 g 更新 → 权重始终一致
```

- **PyTorch DDP** 是工程标准：用 bucket 把梯度分桶，反向算一桶就异步 AllReduce 一桶，与计算**重叠**（见 [[llm-optimizer/计算通信重叠]]）。
- **通信量**：每步 AllReduce 梯度 $\approx 2\Psi$ 字节（fp16），与 batch 无关。
- **致命缺点**：每卡都存完整 $16\Psi$ 模型状态 —— **大模型根本放不下**。这就引出 ZeRO。

**手算**：7B 模型，fp16 梯度 $2\times7\times10^9=14$ GB，Ring-AllReduce 实际传 $2\times\frac{N-1}{N}\times14 \approx 28$ GB/步。

## 4. ZeRO：把数据并行里的"冗余"切掉（DeepSpeed 核心）

观察：DP 下 $N$ 张卡存了 $N$ 份**完全相同**的优化器状态/梯度/参数，纯冗余。**ZeRO（Zero Redundancy Optimizer）** 把这 $16\Psi$ 沿数据并行维切成 $N$ 份。详见 [[ai-framework/deepspeed/README]]。

```
 朴素 DP: 每卡 16Ψ        ZeRO-3: 每卡 16Ψ/N
 ┌────────────┐          ┌──┐┌──┐┌──┐┌──┐
 │ P G O ...  │×N 份      │P0││P1││P2││P3│  参数也切
 └────────────┘          └──┘└──┘└──┘└──┘
                         用到某层时 AllGather 临时拼回
```

| 阶段 | 切什么 | 每卡显存 | 额外通信 |
|---|---|---|---|
| ZeRO-1 | 优化器状态 | $4\Psi + \frac{12\Psi}{N}$ | 同 DP |
| ZeRO-2 | + 梯度 | $2\Psi + \frac{14\Psi}{N}$ | 同 DP（ReduceScatter 代 AllReduce） |
| ZeRO-3 | + 参数 | $\frac{16\Psi}{N}$ | 约 1.5× DP（多一次 AllGather 参数） |

**ZeRO-Offload / Infinity**：把状态再卸载到 CPU 内存甚至 NVMe，用带宽换显存，让单机训更大模型。

**手算**：175B、$N=64$，ZeRO-3 模型状态/卡 $= 2800/64 \approx 43.75$ GB，加上激活后可放进 80GB A100 —— 这正是 ZeRO 的威力。

## 5. 流水线并行 PP（按层切）

**切法**：把模型**按层（深度）切段**，每段放一张卡（一个 stage）。前向像流水线一样逐段传激活，反向逐段传梯度。

```
 4 层模型切到 4 卡:
 GPU0: L0 → GPU1: L1 → GPU2: L2 → GPU3: L3
       激活→      激活→      激活→
   反向:  ←梯度      ←梯度     ←梯度

 朴素 PP 的"气泡"(只 1 个 microbatch 时):
 t→  GPU0 ███░░░░░░░░░███
     GPU1 ░░░███░░░░░███░
     GPU2 ░░░░░░███░███░░
     GPU3 ░░░░░░░░░███░░░   ░=空闲(气泡)
```

朴素 PP 大量空闲。**GPipe/1F1B** 把一个 batch 拆成 $m$ 个 **microbatch** 喂入，填满流水线：

$$\text{气泡率} = \frac{p-1}{m+p-1}$$

```
 微批化 (m=4, p=4) 后流水更满:
 GPU0 1 2 3 4 ········
 GPU1 · 1 2 3 4 ······
 GPU2 ·· 1 2 3 4 ·····
 GPU3 ··· 1 2 3 4 ····   头尾仍有小气泡
```

**手算**：$p=4,m=1$ 气泡率 $=3/4=75\%$（极差）；$m=16$ 时 $=3/19\approx16\%$。**结论：microbatch 数 $m$ 要远大于 stage 数 $p$。** PP 通信量小（只传段间激活），但有气泡，适合跨节点（带宽较低）。详见 [[ai-framework/megatron-lm/README]]。

## 6. 张量并行 TP（把单个矩阵切开）

**切法**：在**单层内部**把权重矩阵切到多卡，每卡算一部分，再用通信拼回。这是 Megatron-LM 的招牌。详见 [[llm-inference/大模型推理张量并行]]。

以 MLP（$Y=\text{GeLU}(XA)B$）为例：

```
 列切 A=[A1,A2], 行切 B=[B1;B2]:

   X ──┬──→ XA1 → GeLU → ·B1 ─┐
       │                       ├─ AllReduce → Y
       └──→ XA2 → GeLU → ·B2 ─┘
   A 按列切: GeLU 可在本地独立做(逐元素)
   B 按行切: 两半相加 = 完整结果 → 一次 AllReduce
```

注意力同理：按**注意力头**切，每卡算若干个 head，输出投影行切，再 AllReduce。

**通信代价**：每个 Transformer 层前向 **2 次 AllReduce**（MLP 一次、Attention 一次），反向再 2 次，共 4 次。通信极频繁且在关键路径上 → **TP 只适合机内高带宽（NVLink）**，跨机会被慢网络拖死。

**手算**：隐藏维 $h=12288$（175B），单层激活 AllReduce 量约 $b\cdot s\cdot h\cdot 2$ 字节；$b{\cdot}s=2048$ 时 $\approx 2048\times12288\times2 \approx 50$ MB/次，每层 4 次 → 必须 NVLink（~600 GB/s）才能压住延迟。

## 7. 序列并行 SP（切掉 TP 没切的激活）

TP 切了矩阵乘法，但 **LayerNorm 和 Dropout** 是沿隐藏维做的、TP 没法切，这些算子的激活仍在每卡全量保存。**序列并行**沿**序列维度 $s$** 把这些算子切开，与 TP 配合消除冗余激活。

```
 Transformer 层内 (TP+SP 配合):
 [LayerNorm] ← SP 切序列维 s  (省激活)
      │ AllGather(g) 还原序列
 [Attention/MLP] ← TP 切隐藏维 h
      │ ReduceScatter(g̅) 再切回序列
 [Dropout]   ← SP 切序列维 s
 通信总量与纯 TP 持平(AllReduce 拆成 RS+AG), 但激活显存大降
```

**收益**：在不增加通信总量的前提下，把 LayerNorm/Dropout/残差的激活显存按 TP 度数 $t$ 削减，常配合**激活重计算（recomputation）** 进一步省显存。这是把超长序列、超大模型放进集群的关键拼图。

## 8. MoE 并行（专家分到不同卡）

**混合专家（MoE）**：每个 token 只激活少数专家（如 top-2/8），用稀疏换"大参数量、小计算量"。详见 [[llm-algo/moe/README]]。

**专家并行（EP）**：把 $E$ 个专家分到不同卡，token 经 router 选中专家后，用 **All-to-All** 把 token 送到对应卡，算完再 All-to-All 送回。

```
 Router 给每个 token 选专家:
 GPU0[E0,E1] GPU1[E2,E3]
   tokens ──router──┐
   ┌── All-to-All ──┴── 按目标专家重排到对应卡
   各卡跑本地专家 FFN
   └── All-to-All ── 把结果送回原位置
```

- **关键挑战**：负载不均（热门专家挤爆某卡）→ 需 **辅助负载均衡损失** + **容量因子（capacity factor）** 丢弃溢出 token。
- **All-to-All** 通信量随 token 数线性增长，是 MoE 训练的主瓶颈。

**手算**：8 专家 top-2，理论计算量 $\approx$ 稠密的 $2/8=1/4$，但参数量是稠密 FFN 的 8 倍 —— 这就是"大模型小算力"的来源。

## 9. 3D/多维混合并行（组合拳）

实际训 100B+ 模型，**单一并行都不够**，要把它们正交组合：

$$N_{\text{total}} = d_{\text{DP}} \times p_{\text{PP}} \times t_{\text{TP}} \ (\times e_{\text{EP}})$$

**经验摆放（按通信频率从高到低配带宽）**：

```
 一个 8 卡节点内 (NVLink 高带宽):
 ┌───────── Node 0 ─────────┐ ┌──── Node 1 ────┐
 │ TP=2  TP=2  TP=2  TP=2   │ │  ... PP stage1 │
 │ └PP stage0, DP group───┘ │ │                │
 └──────────────────────────┘ └────────────────┘
  TP → 机内(NVLink)   PP/DP → 跨机(IB/以太网)
  频繁通信放快网, 稀疏通信放慢网
```

**摆放原则**：
- **TP 放机内**（每层 4 次 AllReduce，最吃带宽，必须 NVLink）。
- **PP/DP 可跨机**（通信少、可与计算重叠）。
- 维度乘积要整除集群规模，且让每个并行组落在合适的网络层级。

**手算**：1024 卡训 175B，可取 $t=8$（机内）、$p=8$、$d=16$，则 $8\times8\times16=1024$，每卡模型状态 $\approx 2800/(8\times8)\approx 44$ GB（TP+PP 共切 64 份）。

## 10. 自动并行（Alpa：让编译器替你切）

手工调 $d,p,t,e$ 和摆放极费力。**Alpa** 把并行分两层自动搜索：

```
 Alpa 两层并行:
 ┌─ inter-op (算子间)  → 对应 流水线并行(切计算图)
 │     用动态规划切 stage
 └─ intra-op (算子内)  → 对应 数据/张量并行(切单算子)
       用整数规划选每个算子的切分策略
 输入: 计算图 + 集群拓扑 → 输出: 自动并行方案
```

- **intra-op**：单个算子怎么切（≈ TP/DP），用代价模型 + ILP 求最省通信的 sharding。
- **inter-op**：计算图怎么分段流水（≈ PP），用 DP 切 stage。
- **意义**：把"人肉调并行"变成"编译器搜并行"，是自动并行的代表作。生产中 Megatron/DeepSpeed 仍以手工+模板为主，自动并行是趋势。

## 11. 框架与模型对照（学什么、怎么练）

| 框架 | 强项 | 主打技术 | 何时选 |
|---|---|---|---|
| **DeepSpeed** | 省显存、易用 | ZeRO-1/2/3、Offload、Infinity | DP 为主、想用 ZeRO 放大模型 → [[ai-framework/deepspeed/README]] |
| **Megatron-LM** | 极致 TP/PP | 张量并行、序列并行、1F1B | 大规模 3D 并行、自研大模型 → [[ai-framework/megatron-lm/README]] |
| **Alpa** | 自动并行 | inter/intra-op 自动搜索 | 想免手调、研究自动并行 |
| **Megatron-DeepSpeed** | 二者融合 | TP/PP(Megatron)+ZeRO(DS) | 训 100B+（BLOOM 即用此） |

**练手模型阶梯**（结构见 [[llm-algo/transformer/模型架构]]）：

```
 GPT-2 (345M) ── 单卡/DDP 跑通训练循环
      ↓
 LLaMA / LLaMA2 ── RMSNorm + RoPE([[llm-algo/旋转编码RoPE]]) + SwiGLU
      ↓
 ChatGLM / ChatGLM2 ── Prefix-LM / 2D 位置
      ↓
 Bloom (176B) ── 真·3D 并行(Megatron-DeepSpeed)
```

**学习顺序建议**：① 吃透第 1-2 节显存账与通信原语 → ② DDP 跑通 GPT-2 → ③ DeepSpeed ZeRO 放大到 7B → ④ Megatron TP+PP 理解 3D 并行 → ⑤ 读 Alpa 理解自动并行。

## 数值手算：把 175B 放进 64 卡 A100 集群

目标：GPT-3 175B（$\Psi=175\times10^9$），混合精度 Adam，A100-80GB。

**1) 不切（朴素 DP）**：模型状态 $16\Psi = 2800$ GB。单卡 80GB → 装不下（差 35 倍）。**否决。**

**2) 仅 ZeRO-3，$N=64$**：模型状态/卡 $= 2800/64 = 43.75$ GB。剩 $80-43.75\approx 36$ GB 给激活 —— 勉强可行但激活紧张。

**3) 3D 并行 TP=8（机内）+ PP=8 + DP=1，共 64 卡**：
- 模型状态被 TP×PP $=64$ 份切：$2800/64 = 43.75$ GB/卡。
- TP 通信走 NVLink（机内 8 卡），PP 走 IB 跨机，DP=1 无梯度同步。
- 配激活重计算 + 序列并行后，激活降到约 10-20 GB，**总占用约 55-65 GB，安全放进 80GB。**

**4) 通信量校验（DP 梯度同步，若 DP>1）**：单次 fp16 梯度 AllReduce $\approx 2\times\frac{N-1}{N}\times 2\Psi$。$\Psi=175$B → $2\Psi=350$ GB，实际传 $\approx 690$ GB/步 —— 必须靠**计算通信重叠**（[[llm-optimizer/计算通信重叠]]）摊掉。

**结论**：单一技术解决不了 175B，**ZeRO + TP + PP + SP + 重计算 协同**才放得下、训得动。

## 常见问题

| 问题 | 解答 |
|---|---|
| DP 和 ZeRO 什么关系？ | ZeRO 是"无冗余的 DP"，沿 DP 维把 $16\Psi$ 切成 $N$ 份，逻辑上仍是数据并行。 |
| TP 为什么不能跨机？ | 每层 4 次 AllReduce 在关键路径上，跨机带宽（IB ~几十 GB/s）远低于 NVLink（~600 GB/s），会成瓶颈。 |
| PP 气泡怎么消？ | 增大 microbatch 数 $m$，气泡率 $\frac{p-1}{m+p-1}$，$m\gg p$ 时趋近 0；配 1F1B 调度。 |
| 序列并行省的是什么？ | 省 LayerNorm/Dropout/残差的**激活显存**，且不增加 TP 的通信总量。 |
| MoE 的瓶颈在哪？ | All-to-All 通信 + 专家负载不均；靠负载均衡损失 + 容量因子缓解。 |
| 三者怎么排？ | 通信频率：TP>SP>PP>DP；带宽配比：TP 放机内，PP/DP 可跨机。 |
| Megatron 还是 DeepSpeed？ | 要极致 TP/PP 选 Megatron；要省显存易上手选 DeepSpeed ZeRO；超大模型用 Megatron-DeepSpeed 融合。 |
| 自动并行成熟了吗？ | Alpa 等是趋势但生产仍以手工+模板为主，工具/版本以官方为准。 |

## 🔗 跳转链接

- 知识地图：[[00-知识地图]]
- 集合通信与硬件：[[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/CUDA]]
- 张量并行（推理视角）：[[llm-inference/大模型推理张量并行]] · [[llm-inference/README]]
- 框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]
- 重叠/优化：[[llm-optimizer/计算通信重叠]] · [[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 模型结构：[[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 算力账：[[llm-algo/FLOPs]] · [[transformer内存估算]]
- 训练全景：[[llm-train/README]]
