# 大模型训练面试题：性能指标 / 混合精度 / DeepSpeed 并行

> 训练面试三板斧：怎么**衡量**训练快不快（指标）、怎么**省显存又不掉精度**（混合精度）、怎么**多卡切分**（ZeRO/PP/TP）。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]

## 阅读地图

| 节 | 主题 | 一句话 | 面试高频度 |
| :-- | :-- | :-- | :-- |
| §0 | 锚点 | 三条主线：指标 / 精度 / 并行 | - |
| §1 | 地基 | FLOPs、显存四大块、通信原语 | ⭐⭐ |
| §2 | 训练性能指标 | 吞吐率→MFU 的优先级链 | ⭐⭐⭐ |
| §3 | MFU vs HFU | 重计算为什么让 HFU>MFU | ⭐⭐⭐ |
| §4 | 混合精度 | FP32 备份 + loss scale + LN 兜底 | ⭐⭐⭐ |
| §5 | bf16 vs fp16 | 动态范围 vs 精度的取舍 | ⭐⭐⭐ |
| §6 | DeepSpeed ZeRO | Stage 1/2/3 切什么 | ⭐⭐⭐ |
| §7 | PP + ZeRO | 为什么 PP+ZeRO2/3 不兼容 | ⭐⭐ |
| 实操 | 命令/配置 | ds_config、loss scale 配置 | ⭐⭐ |

## 0. 一句话锚点

- **指标**：优先级 = 吞吐率 > 单步时间 > 线性度 > 内存占用 > 带宽占用 > 训练效率 > FLOPS > 算力利用率。
- **混合精度**：半精度跑得快+省显存，但有下溢和舍入误差；用 **FP32 权重备份 + loss scale + LayerNorm 用 FP32** 三件套兜底。
- **并行**：ZeRO 切的是「优化器状态/梯度/参数」三类显存；**PP + ZeRO 2/3 不兼容**，要组合就用 **PP + ZeRO 1**。

## 1. 地基：训练显存与算力都花在哪

一次训练 step 做三件事：**前向**算 loss、**反向**算梯度、**优化器**更新权重。显存被四类东西吃掉：

```
            ┌──────── 训练显存四大块（以 Adam + 混合精度为例）────────┐
   参数 P   │ fp16 权重 2P  +  fp32 权重备份 4P                       │  ← §4 备份在这
   梯度 G   │ fp16 梯度 2P                                            │
   优化器   │ Adam 一阶动量 m 4P  +  二阶动量 v 4P                    │  ← ZeRO 主要切这块
   激活值 A │ 与 batch、seq_len、层数成正比，可被重计算换显存         │  ← §3 重计算在这
            └────────────────────────────────────────────────────────┘
   合计 ≈ 16P（参数相关） + 激活值A
```

数值手算：7B 模型（$P=7\times10^9$），仅「参数相关」就要 $16P = 16\times7\text{B} = 112\text{GB}$，已超单张 A100-80G——这正是为什么需要 ZeRO 把这 16P 拆到多卡（§6）。

算力侧：一次前反向的理论计算量约 $C \approx 6\,P\,D$（D 为训练 token 数，前向 2、反向 4 倍 FLOPs/参数/token），「算力利用率」（§3）就是衡量这 $6PD$ 里真正被矩阵乘吃掉的比例。

## 2. 训练通常关注哪些性能指标

> **面试题：模型训练通常关注的性能指标有哪些？**

| 指标名称 | 单位 | 指标含义 |
| :---- | :----------------- | ----- |
| 吞吐率 | samples/s、tokens/s | 单位时间（例如 1s）内处理的 Token 数 / 训练样本数 |
| 单步时间 | s | 执行一个 step 所花费的时间 |
| 线性度、加速比 | values | 单卡训练扩展到多卡、单机扩展到集群的效率度量指标 |
| 内存占用 | 百分比 | - |
| 带宽占比 | 百分比 | - |
| 训练效率 | tokens/day | - |
| 浮点运算 | TFLOPS | 每秒浮点运算次数，是计算设备的计算性能指标 |
| 模型算力利用率（MFU） | 百分比 | 模型一次前反向计算消耗的矩阵算力与机器算力的比值 |
| 硬件算力利用率（HFU） | 百分比 | 考虑重计算后，模型一次前反向计算消耗的矩阵算力与机器算力的比值 |

**优先级排序（原文要点，务必记）**：

$$\text{吞吐率} > \text{单步迭代时间} > \text{线性度} > \text{内存占用} > \text{带宽占用} > \text{训练效率} > \text{FLOPS} > \text{算力利用率}$$

**为什么是这个顺序？** —— 越靠前越「贴近用户实际收益、越端到端、越不受规模影响」：

```
   吞吐率 ─── 端到端总产出，一个数概括快慢，最直观
     │
   单步时间 ─ 吞吐率 = batch_size / 单步时间，是其分母，可定位瓶颈
     │
   线性度 ─── 扩到多卡后效率掉多少（理想 N 卡=N 倍），决定能否堆规模
     │
   内存/带宽 ─ 子资源利用，解释「为什么慢」
     │
   训练效率(tokens/day) ─ 工程排期视角，但受卡数堆叠影响，不纯
     │
   FLOPS / 算力利用率 ─ 最底层硬件视角，离用户最远，故排最后
```

记忆口诀：**「先看总产出（吞吐），再看每步（单步），再看扩展性（线性度），最后才追硬件细节」**。

## 3. MFU vs HFU：重计算（激活重计算）是分水岭

两者都是「模型前反向真正用的矩阵算力 ÷ 机器峰值算力」，差别只在**分子算不算重计算的那部分白干的 FLOPs**：

```
   理论必需 FLOPs ─────────────────► 分子1 (MFU 用)
        +
   激活重计算多算的前向 FLOPs ─────► 分子2 (HFU 额外加上)
        ───────────────────────────
   都 ÷ 机器峰值算力
```

- **MFU（Model FLOPs Utilization）**：分子 = 模型「应该」算的 FLOPs，不含重计算的浪费。它衡量「这次训练对硬件的本质利用」。
- **HFU（Hardware FLOPs Utilization）**：分子 = 硬件「实际」算的 FLOPs，**含重计算重复跑的前向**。所以 **HFU ≥ MFU**。

为什么开了重计算 HFU 升高但 MFU 不变？因为重计算（gradient checkpointing）为省激活显存，在反向时**重跑一遍前向**——硬件确实更忙了（HFU↑），但这些是「为省显存而重复的白工」，不增加模型的本质进展（MFU 不变）。**面试一句话**：HFU 看硬件忙不忙，MFU 看忙得有没有用；重计算用「多算 ~33% 前向」换「激活显存大幅下降」。

## 4. 混合精度训练：半精度的优缺点与三件套兜底

> **面试题：混合精度训练使用半精度训练的优缺点？**

- **优点**：跑得快（半精度矩阵乘走 Tensor Core，吞吐翻倍）+ 省显存（激活/梯度 2 字节而非 4 字节）。
- **缺点**：精度问题 = **下溢（underflow）** + **舍入误差（rounding error）**。

用了半精度，一般配一套「捆绑技术」弥补缺点：

```
   ┌─ FP32 权重备份 ──┐   ┌─ loss scale ───┐   ┌─ LayerNorm 用 FP32 ─┐
   │ 解决「舍入误差」 │   │ 解决「下溢」    │   │ 解决「均值方差累加」│
   │ 主权重存 fp32，  │   │ loss×S 放大，   │   │ 加法/除法多，fp16  │
   │ 更新时 fp16 小   │   │ 链式法则梯度也  │   │ 累加易丢精度，整层 │
   │ 梯度不被吃掉     │   │ ×S，回写前 ÷S   │   │ 走 fp32 更稳       │
   └──────────────────┘   └─────────────────┘   └─────────────────────┘
```

**① FP32 权重备份**：对权重额外备份一份 float32 版本。梯度更新时若用 fp16 权重，会因 fp16 精度不够发生**舍入误差**——大权重 + 小梯度相加，小梯度被「舍掉」变成无效更新。fp32 主权重避免这个问题。代价是多占一份权重显存（即 §1 里的 4P），但相对总显存通常不致命。

> 数值直觉：fp16 在 1.0 附近的最小可分辨间隔约 $2^{-10}\approx 0.001$。若权重 $w=1.0$、梯度更新 $-0.0003$，fp16 下 $1.0-0.0003$ 仍 round 回 $1.0$——更新被「吃掉」。fp32 间隔约 $2^{-23}$，能正常累加。

**② loss scale**：训练后期梯度很小，fp16 容易 **underflow**（小于 fp16 最小正规数 $\approx 6\times10^{-5}$ 直接变 0）。对 loss 乘一个大的 scale $S$，由链式法则该 $S$ 会作用到**每个梯度**上，把它们抬出下溢区；优化器更新前再把梯度 $\div S$ 还原。比「逐个梯度去 scale」划算得多（一次 loss×S 顶全部梯度×S）。

**③ LayerNorm 用 FP32**：LN 要算一组值的均值和方差，涉及大量加法和除法，fp16 累加容易「出岔子」（累加丢精度、除法放大误差），所以这层可完全用 float32。

## 5. bf16 vs fp16：用 bf16 / fp16 半精度训练的优缺点

> **面试题：使用 bf16 和 fp16 进行半精度训练的优缺点？**

两者都是 16 位，区别在「指数位 vs 尾数位」如何分配——决定了**动态范围 vs 精度**的取舍：

```
   fp16 (IEEE half):  1 符号 | 5 指数 | 10 尾数   → 范围窄、精度高
   bf16 (bfloat16) :  1 符号 | 8 指数 |  7 尾数   → 范围宽(同 fp32)、精度低
   fp32            :  1 符号 | 8 指数 | 23 尾数
```

| 维度 | fp16 | bf16 |
| :-- | :-- | :-- |
| 指数位 / 尾数位 | 5 / 10 | 8 / 7 |
| 动态范围 | 窄，$\sim 6\times10^{-5}\sim 6.5\times10^4$，**易上溢/下溢** | 与 fp32 同，$\sim 10^{-38}\sim 3\times10^{38}$，几乎不溢出 |
| 相对精度 | 高（10 尾数位） | 低（7 尾数位） |
| 是否需要 loss scale | **需要**（范围窄，靠 §4②兜底） | 基本**不需要**（范围足够） |
| 硬件要求 | 较老 GPU 也支持 | Ampere（A100）及以后 |
| 适合场景 | 老硬件、对精度敏感的小模型 | 大模型预训练主流选择，更稳不易 NaN |

**一句话**：bf16 用「牺牲尾数精度」换「fp32 级动态范围」，所以大模型训练几乎不发散、可省掉 loss scale 调参；fp16 精度更高但范围窄，必须搭配 loss scale，否则后期梯度下溢、或激活上溢成 NaN。

## 6. DeepSpeed ZeRO：各 Stage 切什么

> **面试题：DeepSpeed 的特点是什么？各个 ZeRO Stage 都有什么用？**

ZeRO（Zero Redundancy Optimizer）的核心思想：数据并行下，每张卡本来都存一份**完整**的「优化器状态 + 梯度 + 参数」（§1 那 16P），其实是冗余的——把它们**按卡切片**，用到时再 all-gather 临时拼回。

```
   ┌──────────────── ZeRO 三级切分（每卡显存逐级下降）────────────────┐
   Stage 1 │ 切【优化器状态】(Adam m,v ≈ 8P)  → 每卡 8P/N            │
   Stage 2 │ 切【优化器状态 + 梯度】          → 再省 2P/N            │
   Stage 3 │ 切【优化器状态 + 梯度 + 参数】    → 参数也 2P/N，最省    │
   └──────────────────────────────────────────────────────────────────┘
     省显存：Stage3 > Stage2 > Stage1
     通信量：Stage3 > Stage2 > Stage1   ← 切得越细，跑时拼回的通信越多
```

- **Stage 1**：只切优化器状态（Adam 的一阶/二阶动量，约 8P）。省显存最少，通信最少。
- **Stage 2**：再切梯度。常用甜点档——显存省得多、通信增加可接受。
- **Stage 3**：连参数也切（前向/反向用到某层才 all-gather 该层参数）。最省显存，可训超大模型，但通信开销最大。
- **ZeRO-Offload / Infinity**：把切片进一步搬到 CPU/NVMe，单卡也能跑大模型，代价是 PCIe/磁盘带宽成瓶颈。

## 7. 流水线并行能与 ZeRO 2/3 一起训练吗

> **面试题：流水线并行（PP）能与 DeepSpeed ZeRO 2/3 一起训练吗？**

**结论：PP + ZeRO 2/3 不推荐一起训练。**

根本冲突在「梯度怎么处理」上：

```
   PP    需要：梯度累积 (accumulate gradients)  ← 多个 micro-batch 累加完整梯度
   ZeRO2 需要：梯度分块  (chunk gradients)       ← 把梯度切片分散到各卡
            ▲                       ▲
            └──────── 互相打架 ──────┘
   一个要「完整累加」，一个要「切碎分散」，语义冲突
```

要点（来自原文）：

- PP 需要**累积梯度**，但 ZeRO2 需要对梯度进行**分块（chunk）**，两者语义冲突。
- 即便强行实现，也**没有真正的性能提升**：PP + ZeRO2 实际上比单纯 ZeRO2（无 PP）**更慢且内存效率更低**。
- 如果显存不足，**用 ZeRO3 代替 ZeRO2 + PP** 更划算。
- 正因如此，在 DeepSpeed 中 **PP 与 ZeRO 2/3 不兼容**；要组合请用 **PP + ZeRO 1**。
- ColossalAI 为支持更多并行方式，仍提供了 **ZeRO 3 + PP + TP** 的组合方案（即便效率不高）。

**记忆**：DeepSpeed 里 **「PP 只配 ZeRO 1」**；想更省显存就别叠 PP，直接上 ZeRO 3。

参考：
- https://www.zhihu.com/question/652836990/answer/3468210626
- https://github.com/microsoft/DeepSpeed/issues/1110
- https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/engine.py#L71
- https://github.com/hpcaitech/ColossalAI/issues/682
- https://github.com/hpcaitech/ColossalAI/pull/477

## 实操：DeepSpeed 混合精度 + ZeRO 配置

把上面的原理落到一份典型 `ds_config.json`（对应 §4 loss scale、§5 bf16/fp16、§6 ZeRO Stage）：

```json
{
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,

  "fp16": {                         // §5 选 fp16 时需 loss scale（§4②）
    "enabled": true,
    "loss_scale": 0,                // 0 = 动态 loss scale，自动找最大不溢出的 S
    "initial_scale_power": 16,      // 初始 S = 2^16
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": { "enabled": false },     // §5 选 bf16 则置 true、fp16 置 false，无需 loss scale

  "zero_optimization": {
    "stage": 2,                     // §6 切「优化器状态+梯度」；想最省显存改 3
    "offload_optimizer": {          // §6 ZeRO-Offload：把优化器状态搬 CPU
      "device": "cpu"
    },
    "contiguous_gradients": true,
    "overlap_comm": true            // 通信与计算重叠，缓解切片带来的通信开销
  }
}
```

启动（对应 PP+ZeRO 选择，注意 §7：用 PP 时 stage 设为 1）：

```bash
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json
```

要点对照：
- `"loss_scale": 0` = **动态** loss scale，DeepSpeed 自动找「不溢出的最大 S」，溢出就回退缩小——这是 §4② 的工程实现。
- 选 **bf16** 时把 `bf16.enabled=true`、`fp16.enabled=false`，且**通常不配 loss scale**（§5：范围够大）。
- ZeRO `stage` 与 §6 一一对应；与 PP 同用时务必把 `stage` 降到 **1**（§7）。

## 常见问题 / 坑

| 现象 / 问题 | 原因 | 处理 |
| :-- | :-- | :-- |
| fp16 训练后期 loss 变 NaN / 不降 | 梯度下溢被截成 0（underflow） | 开 loss scale（动态），或直接换 **bf16** |
| 混合精度后权重「更新无效」 | fp16 大权重 + 小梯度相加，小梯度被舍掉（舍入误差） | **FP32 权重备份**（§4①） |
| LayerNorm 数值不稳 / 输出异常 | fp16 累加均值方差丢精度 | 该层用 **FP32**（§4③） |
| HFU 高但训练并没更快 | 开了重计算，硬件多跑了重复前向 | 看 **MFU** 而非 HFU 判断本质效率（§3） |
| PP + ZeRO2 比纯 ZeRO2 还慢 | 梯度累积与梯度分块语义冲突 | **PP 只配 ZeRO1**；要省显存用 **ZeRO3**（§7） |
| 多卡扩展后吞吐没翻倍 | 线性度差，通信成瓶颈 | 查带宽占比、开 `overlap_comm`、减小切分粒度 |
| bf16 老 GPU 跑不了 | bf16 需 Ampere（A100）及以后 | 老卡退回 fp16 + loss scale |
| 单卡放不下 7B | 仅参数相关就 16P≈112GB（§1） | 上 ZeRO2/3 切分，必要时 Offload 到 CPU |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 训练：[[llm-train/README]] · [[llm-train/peft/PEFT-API]]
- 对齐：[[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 硬件/网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]]
- 评测/估算：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · [[docs/transformer内存估算]]
