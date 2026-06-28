# BLOOM-176B 训练经验

> BigScience 团队用 384 张 A100-80GB、3D 并行 + BF16 在 Megatron-DeepSpeed 上训练 176B 多语言大模型的"实战工程笔记"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-infra/网络/集合通信原语]] · [[llm-base/distribution-training/FP16-BF16]] · [[llm-algo/FLOPs]]

## 阅读地图

| 节 | 你会得到什么 | 关键数字 |
|----|-------------|---------|
| 0 锚点 | 一句话记住 BLOOM 训练范式 | 176B / 384 GPU / BF16 |
| 1 地基 | 模型与集群规格、为什么用 3D 并行 | A100-80GB×384 |
| 2 BF16 为什么 | FP16 溢出 → BF16 + FP32 master | 8 位指数 |
| 3 3D 并行 | TP=4 / PP=12 / DP=8 怎么排布 | 4×12×8=384 |
| 4 张量并行 TP | 单层切分、通信量手算 | 2 次 all-reduce/层 |
| 5 流水线并行 PP | micro-batch 气泡、1F1B | bubble=(p-1)/m |
| 6 数据并行 DP | ZeRO 切分优化器状态 | ZeRO-1 |
| 7 显存账本 | 逐字节算 176B 放不放得下 | ~2.8TB 状态 |
| 8 吞吐与算力 | FLOPs / TFLOPS 利用率 | ~150 TFLOPS/GPU |
| 9 稳定性 | loss spike、硬件故障、断点续训 | embedding norm |
| 数值手算 | 通信量/显存/FLOPs 全流程 | — |
| 常见问题 | 速查表 | — |

---

## 0. 一句话锚点

> **BLOOM-176B = "把一个 3.5TB 的训练状态，切成 384 份塞进 80GB 显存里，再用 BF16 让它不溢出，跑 3.5 个月不崩。"**

三件事缺一不可：
1. **3D 并行**（TP×PP×DP）把模型和数据都切开，单卡装得下；
2. **BF16 混合精度**让数值不溢出、优化器在 FP32 里精确累加；
3. **工程鲁棒性**（断点续训、故障自愈、loss spike 监控）让超长训练不前功尽弃。

---

## 1. 地基：模型规格与集群

### 1.1 模型超参（决定一切显存/算力账）

```
BLOOM-176B
├── 层数         L = 70
├── 隐藏维度     h = 14336
├── 注意力头数   a = 112        (每头 128 维)
├── 词表         V = 250880     (多语言, 巨大!)
├── 序列长度     s = 2048
├── 参数量       N ≈ 176 B
└── 训练 token   ≈ 366 B token  (ROOTS 语料, 46 语言 + 13 编程语言)
```

参数量粗算（忽略 bias/LN）：每层 = 注意力 $4h^2$ + MLP $8h^2$ = $12h^2$。
$$N \approx L\cdot 12h^2 + V\cdot h = 70\cdot 12\cdot14336^2 + 250880\cdot14336$$
$$\approx 70\cdot 2.466\text{e}9 + 3.6\text{e}9 \approx 172.6\text{e}9 + 3.6\text{e}9 \approx 176\text{B} ✓$$
注意 **词表 embedding 单独贡献 3.6B**，多语言模型这块不可忽略。

### 1.2 集群规格

```
集群 (Jean Zay, IDRIS, 法国)
┌──────────────────────────────────────────────┐
│  48 节点 × 8 GPU = 384 张 A100-80GB           │
│  节点内: NVLink (8 卡全互联, ~600 GB/s)        │
│  节点间: Omni-Path/InfiniBand                  │
│  另有 64 卡冗余用于硬件故障替换                  │
└──────────────────────────────────────────────┘
```

### 1.3 为什么必须 3D 并行？

176B 参数光是 **FP32 权重就 704GB**，加优化器状态接近 **2.8TB**，单张 80GB 卡装不下零头。
- 只切数据（DP）→ 每卡仍要存整模型，装不下 ❌
- 只切层（PP）→ 70 层切 12 段，每段 ~15B，仍偏大且通信难平衡 ❌
- 只切张量（TP）→ TP 跨机通信巨大，带宽撑不住 ❌

→ **必须三者组合**：TP 在机内吃 NVLink 带宽，PP 跨机但只传激活，DP 用 ZeRO 省优化器内存。

---

## 2. 为什么是 BF16，不是 FP16

### 2.1 FP16 会溢出（核心痛点）

FP16 最大可表示 ~65504。模型越大，激活/梯度里偶尔出现的大数越容易 **overflow → NaN → 训练崩**。
```
FP16 乘法溢出演示
  250 × 250 = 62500   ✓ (< 65504)
  255 × 255 = 65025   ✓ 勉强
  256 × 256 = 65536   ✗ Inf!  ← 巨型模型常踩中
```
loss scaling（把 loss 放大再缩回）能缓解，但 176B 规模下仍频繁触顶。

### 2.2 BF16：用精度换动态范围

```
位布局对比 (都是 16 bit, 2 字节)
            符号  指数      尾数
FP16   :     1     5         10      范围窄~6.5e4, 精度高
BF16   :     1     8          7      范围 ~3.4e38, 精度差
FP32   :     1     8         23      基准
            └── BF16 指数位与 FP32 相同 → 不溢出
```
BF16 指数 8 位，动态范围跟 FP32 一样大 → **几乎不溢出**，代价是尾数只有 7 位（精度差）。
为什么精度差能接受？SGD/Adam 是"蹒跚前行"：单步方向不完美，后续步骤会纠正。

### 2.3 FP32 master weight + FP32 梯度累积

```
混合精度数据流 (每个 micro-batch)
  FP32 master weight ──cast──▶ BF16 weight
                                   │ forward (BF16 计算, 省显存/快)
                                   ▼
                              BF16 activations
                                   │ backward
                                   ▼
                              BF16 grads ──accumulate──▶ FP32 grad buffer
                                                              │
  FP32 master weight ◀──Adam update (FP32)───────────────────┘
```
**关键：梯度累积必须在 FP32 中做**。PP 的特征就是每个 micro-batch 累加梯度，若在 BF16 里累加，7 位尾数会丢掉小梯度（吃掉舍入）。`BF16Optimizer` 保证累加用 FP32 → 把"潜在噩梦变成平稳过程"。

> 实测经验：BF16 下**不需要 loss scaling**（指数范围够），少一个调参旋钮，训练更稳。

---

## 3. 3D 并行总布局：TP=4 / PP=12 / DP=8

$$\text{TP}\times\text{PP}\times\text{DP} = 4\times12\times8 = 384 \text{ GPU} ✓$$

```
384 GPU = 8 个 DP 副本, 每副本 = TP4 × PP12 = 48 GPU
                                                                 
  DP副本0   DP副本1   ...   DP副本7                              
  ┌─────┐  ┌─────┐         ┌─────┐                              
  │48GPU│  │48GPU│  ...    │48GPU│   ← 各持完整模型分片         
  └─────┘  └─────┘         └─────┘                              
     │ 每个 48GPU 内部:                                          
     ▼                                                          
   PP 阶段 (纵向 12 段, 每段 ~6 层)                              
   stage0  stage1  ...  stage11                                 
   ┌──┐    ┌──┐         ┌──┐                                    
   │TP│    │TP│         │TP│   ← 每段内 TP=4 横向切张量          
   │×4│ ──▶│×4│ ──▶ ... │×4│                                    
   └──┘    └──┘         └──┘                                    
```

放置原则（吃带宽特性）：
- **TP=4 放机内**：TP 每层 2 次 all-reduce，通信最密集 → 必须走 NVLink（600GB/s）。
- **PP 跨机**：PP 只在阶段边界传激活，通信稀疏 → 可走较慢的机间网络。
- **DP=8 + ZeRO**：跨副本只在 step 末做一次梯度 all-reduce。

---

## 4. 张量并行 TP（Megatron 式，机内 NVLink）

### 4.1 切法：MLP 列切 + 行切

```
MLP: Y = GeLU(X·A)·B,  把 A 按列切, B 按行切
                                                       
   X ──┬──▶ A1 ─▶ GeLU ─▶ B1 ─┐                        
       │                       ├─(+)─ all-reduce ─▶ Y  
       └──▶ A2 ─▶ GeLU ─▶ B2 ─┘                        
       (TP=4 则切 4 份, 这里画 2 份示意)                 
```
列切后 GeLU 可独立算（非线性不跨设备），行切后求和需 **1 次 all-reduce**。
注意力同理：QKV 按头切（112 头 / 4 = 28 头/卡），输出投影行切再 all-reduce。

### 4.2 通信量手算（每层 forward）

每层 forward 有 2 次 all-reduce（注意力 1 次 + MLP 1 次），backward 再 2 次。
单次 all-reduce 传输量（Ring，每卡收发）≈ $2\cdot\frac{N_{data}(p-1)}{p}$。
激活张量大小（一个 micro-batch）= $b\cdot s\cdot h$ 元素 × 2 字节(BF16)。

设 micro-batch $b=1$，$s=2048$，$h=14336$，TP $p=4$：
$$\text{激活字节} = 1\cdot2048\cdot14336\cdot2 = 58.7\text{ MB}$$
单次 all-reduce 每卡通信 $\approx 2\cdot58.7\cdot\frac{3}{4} \approx 88\text{ MB}$。
全模型 forward = 70 层 × 2 次 = 140 次 → $\approx 12.3\text{ GB}$ 机内通信。
在 600GB/s NVLink 下 $\approx 20\text{ ms}$ → 机内放 TP 才扛得住，跨机（~25GB/s）会慢 24 倍。

---

## 5. 流水线并行 PP（跨机，1F1B）

### 5.1 气泡问题

70 层切 12 段，朴素 PP 像接力，前后段必须等：

```
朴素 PP (4 stage 示意), F=forward B=backward, 空格=气泡
stage0: F1 F2 F3 F4 ......... B4 B3 B2 B1
stage1:    F1 F2 F3 F4 ..... B4 B3 B2 B1
stage2:       F1 F2 F3 F4 . B4 B3 B2 B1
stage3:          F1 F2 F3 F4 B4 B3 B2 B1
                 └气泡(空转)┘
```

气泡占比：
$$\text{bubble fraction} = \frac{p-1}{m}$$
其中 $p$=PP 段数=12，$m$=micro-batch 数。要把气泡压到 < 10%：
$$\frac{12-1}{m} < 0.1 \Rightarrow m > 110$$
所以 BLOOM 用了**大量 micro-batch**（global batch 很大，约 2048 序列，拆成上百个 micro-batch）来摊薄气泡。

### 5.2 1F1B 调度省激活显存

```
1F1B: 稳态下每卡交替 1 次 forward / 1 次 backward
  ...F F F B F B F B F B B B...
        └ 及时 backward 释放激活 → 峰值激活只需存 ~p 份, 而非 m 份
```
朴素全 forward 再全 backward 要存 $m$ 份激活；1F1B 只存约 $p$ 份 → 显存从 O(m) 降到 O(p)。

### 5.3 PP 通信量

PP 只在段边界传一个激活张量（forward）+ 一个梯度张量（backward）。
单次 = $b\cdot s\cdot h\cdot 2$ 字节 = 58.7MB（同上）。每段 2 次 × 11 个边界，远小于 TP 的 140 次 → 故 PP 可跨机。

---

## 6. 数据并行 DP + ZeRO

8 个 DP 副本，每副本持完整模型分片。BLOOM 用 **ZeRO Stage-1**：把**优化器状态**切到 8 个副本上，每副本只存 1/8。

```
ZeRO-1: 优化器状态 (FP32 momentum+variance+master) 切 8 份
  DP0  DP1  ...  DP7
  [os] [os]      [os]   ← 各存 1/8 优化器状态
   ▲ step 时 all-gather 回完整, 更新后再切回
  梯度仍 all-reduce, 参数仍各存完整副本(因 PP/TP 已切过了)
```
为什么只到 Stage-1，不上 Stage-2/3？因为 TP+PP 已经把参数和激活切得很碎，再叠 ZeRO-3 切参数会与 TP 通信冲突、收益低且复杂。**优化器状态是 ZeRO-1 的大头**（占训练状态 75%，见第 7 节），切它最划算。

---

## 7. 显存账本：176B 到底占多少（逐字节）

### 7.1 训练状态（与 batch 无关，纯模型）

BF16 混合精度 + Adam，每个参数需要：
```
  BF16 weight        : 2 字节
  BF16 grad          : 2 字节
  FP32 master weight : 4 字节
  FP32 Adam momentum : 4 字节
  FP32 Adam variance : 4 字节
  ──────────────────────────────
  合计               : 16 字节/参数
```
$$\text{训练状态} = 176\text{e}9 \times 16 = 2.816\text{ TB}$$
其中优化器相关（master+m+v=12 字节）= $176e9\times12 = 2.11$TB ≈ **75%** → 印证 ZeRO-1 切优化器最值。

### 7.2 切分后每卡占多少

```
2816 GB 训练状态  ÷  TP4 × PP12  (参数被切到 48 卡)
                 ÷  DP8 (优化器再被 ZeRO 切 8 份)
```
参数+梯度（4 字节/参 × 176B = 704GB）÷48 ≈ **14.7 GB/卡**。
优化器状态（12 字节 × 176B = 2112GB）÷48÷8 ≈ **5.5 GB/卡**。
小计 ≈ **20 GB/卡**，剩 60GB 留给激活和碎片。

### 7.3 激活显存（与 batch、序列相关）

单层单 micro-batch 激活 ≈ $s\cdot b\cdot h\cdot(\text{常数}\sim34)$ 字节（含 attention 中间量）。
1F1B 下每卡约存 $p_{local}$ 段 × 每段 6 层。粗估每卡激活峰值数 GB ~十几 GB，配合 **激活重计算（gradient checkpointing）** 可再砍：只存每段输入，backward 时重算 → 激活内存 $O(\sqrt{L})$ 量级，换约 33% 额外算力。

```
激活重计算权衡
  不重算: 存所有中间激活, 显存大, 算力 1×
  重算  : 只存段边界, 显存小, 算力 ~1.33× (多一次 forward)
  176B 这种规模 → 选重算, 显存是硬约束
```

---

## 8. 吞吐、FLOPs 与利用率

### 8.1 一次前向+反向的 FLOPs

经验公式：训练每 token 计算量 $\approx 6N$（forward 2N + backward 4N），重计算再 +2N → $\approx 8N$。
$$C_{token} \approx 8\times176\text{e}9 = 1.41\text{e}12 \text{ FLOPs/token}$$
全程 366B token：
$$C_{total} \approx 1.41\text{e}12 \times 366\text{e}9 = 5.16\text{e}23 \text{ FLOPs}$$

### 8.2 时间手算（验证 ~3.5 个月）

A100 BF16 峰值 ~312 TFLOPS，实测 MFU（利用率）~30-40%，取有效 ~100-150 TFLOPS/卡。
384 卡有效算力 $\approx 384\times120\text{e}12 = 4.6\text{e}16$ FLOP/s。
$$T = \frac{5.16\text{e}23}{4.6\text{e}16} \approx 1.12\text{e}7 \text{ s} \approx 130 \text{ 天} \approx 3.5 \text{ 月} ✓$$
与官方"约 3.5 个月"吻合，说明 MFU ~30-40% 的估计合理。

### 8.3 为什么 MFU 上不去？

```
理想 312 TF ──气泡(PP)──▶ ──通信(TP all-reduce)──▶ ──重计算(33%)──▶ 实测 ~120 TF
   100%        -~8%            -~15%                  -~25%          ~38%
```
每个并行维度都"吃"一口利用率，巨型模型 MFU 能到 ~40% 已是优秀工程。

---

## 9. 训练稳定性：真正的难点

### 9.1 Loss spike（损失尖峰）

超长训练偶发 loss 突然飙升。BLOOM 经验：
- **embedding 层加 LayerNorm**（"embedding norm"）显著稳住早期训练；
- 监控 **梯度范数（grad norm）**，异常飙升时该 step 容易炸；
- 必要时回滚到最近 checkpoint、跳过坏数据 batch 再续。

```
loss 曲线
 loss
  │＼
  │ ＼___              ← 正常下降
  │     ＼  ╱＼ spike! ← 检测到 → 回滚 checkpoint, 跳 batch
  │      ＼╱   ＼___
  └────────────────▶ steps
```

### 9.2 硬件故障（384 卡跑 3.5 月必然遇到）

GPU/网络/节点几乎一定会坏。应对：
- **频繁 checkpoint**（如每 ~3 小时），故障后从最近点续；
- 预留 **64 卡冗余**，坏节点立刻替换不停训；
- checkpoint 要存 **所有并行维度的分片状态 + 优化器 + RNG 种子 + 数据加载器位置**，否则续训不等价。

### 9.3 断点续训的"等价性"

```
完整 checkpoint 必须包含:
  ├── 模型分片 (TP×PP 切片)
  ├── 优化器状态 (ZeRO 切片)
  ├── lr scheduler 步数
  ├── 数据加载器 consumed_samples (从哪条数据接着读)
  └── 各 rank 的随机数种子 (保证 dropout/重计算可复现)
缺一 → 续训后曲线偏离, 等于"换了次实验"
```

### 9.4 Slurm 编排

集群用 Slurm 调度，作业脚本声明节点数/GPU/时长，配合"超时前自动 checkpoint + 重新排队"实现长时间无人值守训练（参考 BigScience 的 `tr11-176B-ml.slurm`）。具体 Slurm 参数/命令默认值以官方为准。

---

## 数值手算汇总（一页速查）

| 量 | 公式 | 代入 | 结果 |
|----|------|------|------|
| 参数量 | $L\cdot12h^2 + Vh$ | $70,14336,250880$ | 176B |
| 训练状态总字节 | $16N$ | $16\times176e9$ | 2.82 TB |
| 优化器占比 | $12/16$ | — | 75% |
| 参数+梯度/卡 | $4N/(\text{TP·PP})$ | $704/48$ | 14.7 GB |
| 优化器/卡(ZeRO1) | $12N/(\text{TP·PP·DP})$ | $2112/384$ | 5.5 GB |
| 单激活张量(b=1) | $bsh\times2$ | $2048\times14336\times2$ | 58.7 MB |
| TP 单 all-reduce/卡 | $2\cdot\text{act}\cdot\frac{p-1}{p}$ | $2\times58.7\times0.75$ | 88 MB |
| PP 气泡占比 | $(p-1)/m$ | $11/m$ | <10% 需 m>110 |
| 训练总 FLOPs | $8N\cdot T_{tok}$ | $8\times176e9\times366e9$ | 5.16e23 |
| 训练时长 | $C/(\text{卡数·有效TF})$ | $5.16e23/4.6e16$ | ~130 天 |

---

## 常见问题

| 问题 | 答案 |
|------|------|
| 为什么不用 FP16？ | 巨型模型激活/梯度易超 6.5e4 → 溢出 NaN；BF16 指数 8 位范围同 FP32，几乎不溢出。 |
| BF16 精度差怎么办？ | 计算用 BF16，**master weight 和梯度累积用 FP32**，优化器全精度更新，精度不丢。 |
| 为什么梯度累积一定要 FP32？ | PP 每个 micro-batch 累加梯度，BF16 的 7 位尾数会把小梯度舍掉，FP32 累加才精确。 |
| TP 为什么放机内？ | TP 每层 2 次 all-reduce，通信最密，必须 NVLink 600GB/s；跨机带宽差 24 倍会卡死。 |
| PP 为什么能跨机？ | PP 只在段边界传激活，通信稀疏，机间网络扛得住。 |
| 为什么只用 ZeRO-1？ | 优化器状态占 75% 是大头，切它收益最高；TP/PP 已切参数和激活，再上 ZeRO-3 冲突且复杂。 |
| 气泡怎么压？ | 增大 micro-batch 数 $m$，$(p-1)/m$ 随 $m$ 变小；BLOOM 用上百 micro-batch。 |
| 激活放不下？ | 用激活重计算（gradient checkpointing），显存换 ~33% 算力，巨型模型必选。 |
| MFU 为什么只有 ~38%？ | 气泡 + TP 通信 + 重计算各吃一口，巨型模型能到 40% 已优秀。 |
| loss spike 怎么处理？ | embedding 加 LayerNorm 稳定训练；监控 grad norm；炸了就回滚 checkpoint 跳坏 batch。 |
| 3.5 个月硬件坏了？ | 频繁 checkpoint + 冗余节点热替换 + 完整状态（含数据位置/RNG 种子）保证等价续训。 |

---

## 🔗 跳转链接

- 框架与并行实现：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]
- 混合精度细节：[[llm-base/distribution-training/FP16-BF16]] · [[llm-base/distribution-training/自动混合精度]]
- 同类巨模经验：[[llm-base/distribution-training/GLM-130B训练经验]] · [[llm-base/distribution-training/OPT-175B训练经验]]
- 并行原语与网络：[[ai-infra/网络/集合通信原语]] · [[llm-optimizer/计算通信重叠]]
- 算力与显存估算：[[llm-algo/FLOPs]] · [[ai-infra/算力/GPU工作原理]]
- 张量并行（推理侧对照）：[[llm-inference/大模型推理张量并行]]
- 模型结构基础：[[llm-algo/transformer/模型架构]] · [[llm-algo/mlp]]
- 知识地图：[[00-知识地图]]

---
参考：BigScience / HuggingFace 博客《如何丝滑训练 BLOOM》<https://huggingface.co/blog/zh/bloom-megatron-deepspeed>，Slurm 脚本 <https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/tr11-176B-ml.slurm>。具体框架版本/命令默认值以官方仓库为准。
