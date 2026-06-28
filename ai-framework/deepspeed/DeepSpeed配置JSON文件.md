# DeepSpeed 配置详解

> `ds_config.json` 是 DeepSpeed 的"总开关面板"——一份 JSON 决定 ZeRO 切几片、精度用 fp16 还是 bf16、优化器/调度器怎么配、显存往哪 offload。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/deepspeed/README]] [[llm-train/megatron-deepspeed/README]]

> 官方字段全集：https://www.deepspeed.ai/docs/config-json/（精确字段名/默认值/新增项以官方文档与源码为准）

## 阅读地图

| 你想搞清楚的问题 | 跳到哪节 |
| --- | --- |
| 这份 JSON 到底在控制什么、谁来读它？ | §0 锚点 §1 地基 |
| batch 三个字段为什么互相约束？ | §2 batch 三元组 |
| ZeRO stage 0/1/2/3 在 JSON 里怎么写？ | §3 zero_optimization |
| offload 把什么搬到哪、代价多大？ | §4 offload |
| fp16 / bf16 / 混合精度怎么选？ | §5 精度 |
| 优化器、学习率调度怎么在 JSON 里配？ | §6 optimizer §7 scheduler |
| 梯度裁剪/累积/激活检查点放哪？ | §8 其它常用块 |
| 一份完整 ds_config 长啥样、每行啥意思？ | §9 完整示例逐行讲 |
| 字段写错/不生效/和框架打架 | §10 常见问题 |

## 0. 一句话锚点

**DeepSpeed 把所有"训练加速 + 省显存"的策略，统一抽象成一份 JSON 配置文件 `ds_config.json`。** 你的训练代码几乎不动，只在 `deepspeed.initialize(...)` 时把这份 JSON 喂进去，引擎就按里面的指令决定：模型状态切几片、用什么精度、优化器搬不搬 CPU、batch 怎么拆。**改策略 = 改 JSON，不是改代码**，这是 DeepSpeed 易用性的核心设计。

## 1. 地基：这份 JSON 谁读、什么时候读

```
你的训练脚本                          DeepSpeed 引擎
─────────────                        ──────────────
model = MyModel()                          │
optim  = ...(可选, 也可写进JSON)            │
                                           ▼
engine, optim, _, sched =          ┌──────────────────────┐
  deepspeed.initialize(            │ 1. 解析 ds_config.json │
     model=model,                  │ 2. 包裹 model→engine   │
     config="ds_config.json",  ──▶ │ 3. 按 zero stage 切片  │
     model_parameters=...)         │ 4. 建 optimizer/sched  │
                                   │ 5. 设精度/通信/offload │
for batch in data:                 └──────────┬───────────┘
   loss = engine(batch)                       │ 引擎接管:
   engine.backward(loss)  ───────────────────▶│ 自动 all-gather/
   engine.step()          ───────────────────▶│ reduce-scatter/
                                               │ offload/精度转换
```

**关键认知**：JSON 里的字段不是"建议"，而是**引擎运行时真正执行的指令**。同一个字段在不同 stage 下含义可能变化（如 `overlap_comm` 只在 stage≥1 有意义）。命令行 `deepspeed --num_gpus ... train.py --deepspeed ds_config.json` 把它传进来；HuggingFace Trainer 则用 `--deepspeed ds_config.json`，并允许用 `"auto"` 占位让 Trainer 回填字段。

JSON 顶层是一个大对象，**每个顶层 key 是一个"功能模块"**，本文逐块拆解：

```
ds_config.json
├── train_batch_size / *_micro_* / gradient_accumulation_steps  ← §2 batch 三元组
├── zero_optimization { stage, offload_*, ... }                 ← §3 §4 核心
├── fp16 { } / bf16 { }                                         ← §5 精度
├── optimizer { type, params }                                  ← §6
├── scheduler { type, params }                                  ← §7
├── gradient_clipping / gradient_accumulation_steps             ← §8
├── activation_checkpointing { }                                ← §8
├── steps_per_print / wall_clock_breakdown / comms_logger       ← 监控
└── ...
```

## 2. batch 三元组：三个字段的隐藏等式

最容易写错的就是 batch 相关三字段，因为它们**不是独立的**，受一条等式约束：

$$\text{train\_batch\_size} = \text{train\_micro\_batch\_size\_per\_gpu} \times \text{gradient\_accumulation\_steps} \times N_{gpu}$$

```
全局有效 batch (train_batch_size = 64)
        │  拆给 8 张卡 (N_gpu=8)
        ▼
每卡要处理 8 个样本 / step
        │  显存放不下 8 个 → 拆成 2 次微批，每次 4
        ▼
micro_batch_per_gpu=4 , gradient_accumulation_steps=2
   ↓ 微批1 fwd+bwd(存梯度,不更新)
   ↓ 微批2 fwd+bwd(梯度累加)
   ↓ 累计够2步 → 一次 optimizer.step() 更新
4 × 2 × 8 = 64 ✓ 等式成立
```

| 字段 | 含义 | 权衡 |
| --- | --- | --- |
| `train_batch_size` | 全局有效批量（语义上的 batch） | 决定收敛行为/学习率，是"科学超参" |
| `train_micro_batch_size_per_gpu` | 单卡单次前向的样本数 | 受显存约束的"工程参数"，越大越省通信但越吃显存 |
| `gradient_accumulation_steps` | 累积几次微批再更新 | 用时间换显存：放不下大 batch 就多累积几步，不增显存 |

**实践**：三者只需指定任意两个，第三个由等式推出；HF Trainer 里常写 `"auto"` 让它自动算。**梯度累积是"穷人版大 batch"**——不增显存却能模拟大 batch 的收敛特性，但每个 micro step 的归一化要正确（DeepSpeed 内部已处理梯度平均）。

## 3. zero_optimization：JSON 的心脏

ZeRO 的"为什么省显存"在 [[ai-framework/deepspeed/README]] §2 已讲透（参数 P/梯度 G/优化器状态 O 每卡只存 $1/N$）。这里只讲**怎么在 JSON 里表达**：

```json
"zero_optimization": {
  "stage": 2,
  "allgather_partitions": true,
  "reduce_scatter": true,
  "overlap_comm": true,
  "contiguous_gradients": true,
  "reduce_bucket_size": 5e8,
  "allgather_bucket_size": 5e8
}
```

```
stage=0  关闭 ZeRO，退化为普通 DDP（每卡全量 P+G+O）
stage=1  切 优化器状态 O          省最多、通信几乎不增  ← 几乎白嫖
stage=2  切 O + 梯度 G            省更多、通信仍很少    ← 性价比之王
stage=3  切 O + G + 参数 P        省到底、多一次参数all-gather通信
         ┌─────────────────────────────────────────┐
         │ O ───切─┐                                │
         │ G ──切──┤ stage 越高，切的越多，省得越多   │
         │ P ─切───┘ 但 stage3 跑前向要临时凑齐参数   │
         └─────────────────────────────────────────┘
```

**核心字段语义与权衡**：

| 字段 | 含义 | 权衡/建议 |
| --- | --- | --- |
| `stage` | 0/1/2/3，切分等级 | 能放下就用低 stage（通信少）；OOM 才往上加 |
| `overlap_comm` | 通信与计算重叠 | 开启提速，但多占一点临时显存做缓冲 |
| `contiguous_gradients` | 梯度连续内存存放 | 减少显存碎片，stage≥2 推荐开 |
| `reduce_bucket_size` | 梯度规约的分桶字节数 | 桶大→通信效率高但峰值显存高；桶小→反之 |
| `allgather_bucket_size` | 参数聚合分桶字节数 | 同上权衡，OOM 时调小 |

**stage 3 专属字段**（参数也被切，前向时要"凑参数"）：

| 字段 | 含义 | 权衡 |
| --- | --- | --- |
| `stage3_prefetch_bucket_size` | 提前预取下一层参数的桶大小 | 大→隐藏通信延迟好但占显存 |
| `stage3_param_persistence_threshold` | 小于此阈值的参数不切片（常驻） | 小参数切片不划算，留着省通信 |
| `stage3_max_live_parameters` | 同时"活着"的参数上限 | 上限越小越省显存但通信越频繁 |
| `stage3_gather_16bit_weights_on_model_save` | 存模型前把切片权重合并 | **必开**，否则存出来的 `state_dict` 是残缺的分片 |

> 上一版本本文件仅有 `stage3_gather_16bit_weights_on_model_save` 一项——它的作用正是：参数被切在各卡上不在 `state_dict` 里，调用 `save_16bit_model()` 时引擎自动 all-gather 凑齐再存 fp16 权重。

## 4. offload：显存不够，搬到 CPU/NVMe

当 ZeRO-3 切片后单卡仍 OOM，最后的兜底是 **offload：把优化器状态/参数搬出 HBM**。它是 `zero_optimization` 的子块：

```json
"zero_optimization": {
  "stage": 3,
  "offload_optimizer": { "device": "cpu", "pin_memory": true },
  "offload_param":     { "device": "cpu", "pin_memory": true }
}
```

```
        HBM (快, 贵, 小)        PCIe       CPU 内存 (慢, 便宜, 大)
   ┌────────────────────┐   ~16-32GB/s  ┌────────────────────┐
   │ 参数P / 梯度G       │◀────────────▶│ offload_optimizer  │
   │ 前向反向计算在此     │              │ offload_param      │
   └────────────────────┘              └─────────┬──────────┘
                                                 │ 还放不下
                                        ┌────────▼──────────┐
                                        │ NVMe SSD (更慢更大) │ ← ZeRO-Infinity
                                        └───────────────────┘
```

| 字段 | 取值 | 含义与代价 |
| --- | --- | --- |
| `offload_optimizer.device` | `"cpu"` / `"nvme"` | 把 Adam 的 fp32 状态(12Ψ)搬出，省 HBM 最多；优化器 step 在 CPU 算，会慢 |
| `offload_param.device` | `"cpu"` / `"nvme"`（仅 stage3） | 把参数也搬出，省到极致；前向要从 CPU 拉参数，PCIe 成瓶颈 |
| `pin_memory` | true/false | 锁页内存加速 CPU↔GPU 拷贝，但占用不可换出的物理内存 |
| `nvme_path` | 路径 | NVMe offload 的落盘目录，需高速 SSD |
| `buffer_count`/`buffer_size` | 数值 | NVMe 读写缓冲，调优 I/O 吞吐 |

**铁律**：offload 是**以速度换显存**的最后手段。PCIe 比 HBM 慢一两个数量级，"显存够了但慢得离谱"通常就是 offload + 弱互联导致。能不 offload 就别 offload。

## 5. 精度：fp16 vs bf16

混合精度让参数/梯度用 16 位省一半显存，但 fp16 动态范围窄（易溢出），需要"损失缩放"兜底；bf16 范围宽（同 fp32）但精度低，无需缩放。**二选一，不要同时开。**

```
fp16:  1符号 + 5指数 + 10尾数   范围窄→梯度易下溢→需 loss scaling
bf16:  1符号 + 8指数 +  7尾数   范围同fp32→稳→不需缩放→新硬件首选
fp32:  1符号 + 8指数 + 23尾数   精度高但占2倍显存(只在优化器主副本用)
```

```json
"fp16": {
  "enabled": true,
  "loss_scale": 0,            // 0 = 动态损失缩放(推荐)
  "loss_scale_window": 1000,  // 多少步无溢出就上调 scale
  "initial_scale_power": 16,  // 初始 scale = 2^16
  "hysteresis": 2,            // 连续溢出几次才下调
  "min_loss_scale": 1
}
```
或者（Ampere/Hopper 及以上推荐）：
```json
"bf16": { "enabled": true }
```

| 字段 | 含义 | 权衡 |
| --- | --- | --- |
| `fp16.enabled` | 开启 fp16 混合精度 | 老卡(V100)/无 bf16 支持时用 |
| `loss_scale` | 固定缩放值，`0`=动态 | 动态最省心，自动找最大不溢出的 scale |
| `loss_scale_window` | 上调 scale 的观察窗口 | 太小抖动，太大反应慢 |
| `bf16.enabled` | 开启 bf16 | A100/H100 首选，训练更稳，不用调 scale |

**为什么不能两个都开**：精度模式互斥，引擎只认一个。**经验**：硬件支持 bf16（Ampere 以上）就无脑 bf16，省去 fp16 调 loss scale 的烦恼；只有老卡或追求极致显存/吞吐才考虑 fp16。

## 6. optimizer：在 JSON 里建优化器

DeepSpeed 可以替你创建优化器（也可由你在代码里建好传入）。写进 JSON 的好处是与 offload/ZeRO 自动对齐（如自动启用 fused/CPU Adam）：

```json
"optimizer": {
  "type": "AdamW",
  "params": {
    "lr": 1e-4,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01
  }
}
```

| 字段 | 含义 | 要点 |
| --- | --- | --- |
| `type` | `Adam`/`AdamW`/`Lamb`/`OneBitAdam`... | offload 时 DeepSpeed 会用 `DeepSpeedCPUAdam`(CPU 高效实现) |
| `params.lr` | 学习率 | 随 `train_batch_size` 缩放，是收敛关键 |
| `params.betas`/`eps` | Adam 一/二阶动量系数、数值稳定项 | 一般用默认 |
| `params.weight_decay` | 权重衰减 | AdamW 解耦衰减，常用 0.01~0.1 |

**`OneBitAdam`/`ZeroOneAdam`** 这类是 DeepSpeed 的通信压缩优化器，把动量通信量压到 1-bit，弱互联集群提速明显——这是"JSON 配优化器"才能白嫖的能力。

## 7. scheduler：学习率调度

```json
"scheduler": {
  "type": "WarmupDecayLR",
  "params": {
    "warmup_min_lr": 0,
    "warmup_max_lr": 1e-4,
    "warmup_num_steps": 1000,
    "total_num_steps": 100000
  }
}
```

```
lr
1e-4┤        ___________
    │       /           \____
    │      / warmup       decay\____
    │     /(线性升)        (线性/余弦降)\___
   0└────┴──────────────────────────────────▶ step
        0   1000(warmup_num_steps)      total_num_steps
```

| 字段 | 含义 |
| --- | --- |
| `type` | `WarmupLR`(只热身) / `WarmupDecayLR`(热身+衰减) / `WarmupCosineLR` |
| `warmup_num_steps` | 学习率从 min 线性升到 max 的步数 | 防训练初期梯度爆 |
| `total_num_steps` | 总步数，用于衰减进度 |

**为什么要 warmup**：训练初期参数随机、梯度方向噪声大，直接用大 lr 容易发散；先用小 lr"试探"再升上去更稳。

## 8. 其它常用块

```json
"gradient_clipping": 1.0,
"gradient_accumulation_steps": 4,
"steps_per_print": 100,
"wall_clock_breakdown": false,
"activation_checkpointing": {
  "partition_activations": false,
  "cpu_checkpointing": false,
  "contiguous_memory_optimization": false,
  "number_checkpoints": null
}
```

| 字段 | 含义 | 权衡 |
| --- | --- | --- |
| `gradient_clipping` | 梯度范数裁剪阈值 | 防梯度爆炸，大模型常设 1.0 |
| `activation_checkpointing` | 激活重计算 | 反向时重算激活而非存它，**省激活显存**，换约 30% 算力。与 ZeRO 正交叠加 |
| `partition_activations` | 激活也切片到各卡 | 进一步省显存，配合 TP |
| `steps_per_print` | 多少步打印一次 loss/吞吐 | 仅监控 |
| `wall_clock_breakdown` | 打印各阶段耗时 | 调优时开，平时关（有开销） |

**关键认知**：ZeRO 省的是**模型状态**（P/G/O），`activation_checkpointing` 省的是**激活值**——两条独立的省显存路线，可同时开，叠加效果最猛。

## 9. 完整示例逐行讲（ZeRO-2 + bf16 标准训练配置）

```json
{
  "train_batch_size": 64,                          // 全局有效 batch
  "train_micro_batch_size_per_gpu": 4,             // 单卡单次前向样本数
  "gradient_accumulation_steps": 2,                // 累积2次再更新 (4×2×8卡=64)
  "gradient_clipping": 1.0,                        // 梯度裁剪防爆

  "bf16": { "enabled": true },                     // A100/H100 用 bf16，省一半显存且稳

  "zero_optimization": {
    "stage": 2,                                    // 切 优化器状态+梯度，性价比之王
    "overlap_comm": true,                          // 通信计算重叠提速
    "contiguous_gradients": true,                  // 连续梯度内存，减碎片
    "reduce_bucket_size": 5e8,                      // 梯度规约分桶(OOM时调小)
    "allgather_bucket_size": 5e8,
    "offload_optimizer": { "device": "none" }      // 显存够，不 offload(速度优先)
  },

  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01 }
  },

  "scheduler": {
    "type": "WarmupDecayLR",
    "params": { "warmup_min_lr": 0, "warmup_max_lr": 1e-4,
                "warmup_num_steps": 1000, "total_num_steps": 100000 }
  },

  "activation_checkpointing": {                    // 激活重计算，省激活显存
    "partition_activations": false,
    "cpu_checkpointing": false
  },

  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

**这份配置的决策链**：8 卡 A100 → 支持 bf16 故用 bf16（省 fp16 调缩放）→ 模型 ZeRO-2 能放下故不上 stage3（少一次参数 all-gather 通信）→ 显存够故不 offload（速度优先）→ batch 拆成 micro=4 累积=2 满足等式且不 OOM。**升级路线**：若 OOM，依次：调小 micro batch → 开 activation_checkpointing → stage 升 3 → 加 offload_optimizer cpu → 加 offload_param/NVMe。

```
显存不够的"求救阶梯"（从便宜到昂贵）：
micro↓ → 激活检查点 → stage2→3 → offload优化器 → offload参数 → NVMe
 省一点    省激活        省模型状态    省O(慢)       省P(更慢)    兜底(最慢)
```

## 10. 常见问题

| 问题 | 一句话答案 |
| --- | --- |
| batch 三字段填了报错？ | 它们受等式约束，只填两个让第三个自动算，或都设 `"auto"`（HF Trainer） |
| fp16 和 bf16 都写 enabled:true？ | 互斥，只能开一个；新卡选 bf16，老卡选 fp16 |
| stage3 存模型后权重残缺？ | 必须开 `stage3_gather_16bit_weights_on_model_save`，否则存的是分片 |
| 开了 offload 反而很慢？ | 正常，offload 是以 PCIe 速度换显存的兜底，弱互联尤其慢 |
| ZeRO stage 怎么选？ | 能放下用低 stage（通信少），OOM 才往上加；ZeRO-2 通常最划算 |
| `reduce_bucket_size` 调它干嘛？ | OOM 时调小降峰值显存，带宽足时调大提通信效率 |
| optimizer 写代码里还是 JSON 里？ | 用 offload/1-bit 优化器时写 JSON，引擎才能自动对齐；否则两者皆可 |
| `"auto"` 是什么？ | HF Trainer 占位符，让 Trainer 用命令行参数回填，避免两处配置打架 |
| 改了 JSON 不生效？ | 确认传入路径正确、被 `deepspeed.initialize` 真正读取；HF 下用 `--deepspeed` 指定 |
| 激活检查点和 ZeRO 冲突吗？ | 不冲突，正交：ZeRO 省模型状态，检查点省激活，叠加用 |

## 🔗 跳转链接

- [[00-知识地图]] —— 全局总览，先定位 DeepSpeed 配置在训练栈中的位置
- [[ai-framework/deepspeed/README]] —— ZeRO 原理/显存账/三段切分/选型决策（本文是它的"配置落地篇"）
- [[llm-train/megatron-deepspeed/README]] —— Megatron-DeepSpeed 把 TP/PP 与 ZeRO 组成 3D 并行，配置在此基础上扩展
