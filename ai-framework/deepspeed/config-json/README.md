# DeepSpeed Config JSON 全参数详解

> 一句话定位：`ds_config.json` 是 DeepSpeed 的"中央控制台"——一张 JSON 表格里同时声明 batch 编排、优化器、混合精度、ZeRO 显存切分、卸载、激活重计算、监控与压缩，DeepSpeed 引擎读它来决定训练怎么跑。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[llm-train/README]] · [[docs/transformer内存估算]]

官方参考：https://www.deepspeed.ai/docs/config-json/

## 阅读地图

| 节 | 你将得到 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | config 即声明式控制台 |
| 1 | 地基：DeepSpeed 引擎如何吃 config | `deepspeed.initialize` |
| 2 | Batch Size 三件套与约束公式 | global / micro / accum |
| 3 | Optimizer 优化器 | Adam / fused / torch_adam |
| 4 | Scheduler 学习率调度 | WarmupLR / step() |
| 5 | 通讯选项 | dtype / prescale / predivide |
| 6 | 混合精度：FP16 / BF16 / AMP | loss_scale / 三选一 |
| 7 | 梯度裁剪 | gradient_clipping |
| 8 | ZeRO 与卸载 | offload_param / offload_optimizer |
| 9 | 激活重计算 | activation_checkpointing |
| 10 | 稀疏注意力 / 监控 / 压缩 / Checkpoint / 数据类型 | sparse_attention 等 |
| 实操 | 一份可拼装的完整 config | 真料汇总 |
| 坑 | 高频报错与排查 | 互斥 / 公式不符 |

## 0. 一句话锚点

DeepSpeed 把"训练怎么跑"从代码里抽出来，变成**一个 JSON 文件**。你不改训练循环，只改这张表，就能在数据并行、ZeRO-1/2/3、CPU/NVMe 卸载、FP16/BF16、激活重计算之间自由切换。理解 config-json = 理解 DeepSpeed 的全部能力面。

## 1. 地基：引擎如何读 config

训练脚本里只有一行把 config 喂进引擎：
```python
model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=params,
    config="ds_config.json",   # 或 config=dict(...)
)
# 之后训练循环里：
loss = model_engine(batch)
model_engine.backward(loss)
model_engine.step()            # ← scheduler.step() 也在这里被触发
```

```
 ds_config.json ──► deepspeed.initialize(解析JSON→装配各模块)
                       ├─ batch 编排(累积/通信)   ├─ 混合精度(fp16/bf16)  ├─ 监控/压缩
                       ├─ 优化器(Adam…)           └─ ZeRO/卸载(切分显存)
```

为什么用声明式？**训练逻辑与系统配置解耦**：同一份训练代码，跑单卡调试、跑 8 卡 ZeRO-2、跑 64 卡 ZeRO-3+NVMe 卸载，只换 config，不动 Python。

## 2. Batch Size 相关参数

DeepSpeed 用三个量描述"一步全局更新到底吃了多少样本"，并强制一条**铁律约束**：

```
train_batch_size
   = train_micro_batch_size_per_gpu × gradient_accumulation_steps × GPU数量
```

- **train_batch_size**：一次梯度更新（一次 `step()`）涉及的**全局有效批量**。
- **train_micro_batch_size_per_gpu**：单张 GPU 单次前向/反向真正塞进显存的样本数——**显存压力由它决定**。
- **gradient_accumulation_steps**：在平均并应用梯度之前累积梯度的训练 step 数。

> 原文要点：累积梯度**有时对提高可扩展性很有用，因为它降低了 step 之间梯度通信的频率**；另一个影响是**能在每个 GPU 上用更大的"等效"批量进行训练**（用时间换显存）。累积如何用"小 micro-batch"凑出"大全局 batch"：

```
 每卡:  micro micro micro micro    ← accum=4，每个 micro 单独 fwd/bwd，显存只需 1 个 micro
         └──累加梯度──┘ ▼ 4 步后才 all-reduce + step()  ← 通信频率降为 1/4
 global = micro(8) × accum(4) × gpu(4) = 128
```

数值手算：micro=8、accum=4、8 卡 ⇒ `8×4×8 = 256` = train_batch_size。若你只填 train_batch_size=256 与 micro=8、8 卡，DeepSpeed 会**自动反推** accum=4；三者中给两个，第三个可被推断，但若三者都给且不满足等式则直接报错。

| 想达成的目标 | 调哪个 | 代价 |
|------------|--------|------|
| 显存不够，OOM | 调小 micro，调大 accum | 单 step 变慢（多次 fwd/bwd）|
| 减少跨卡通信开销 | 调大 accum | 梯度更新变稀疏 |
| 增大全局批量稳定训练 | 调大 micro 或 accum 或加卡 | 显存 / 通信 |

## 3. Optimizer 优化器参数

- **type**：优化器名。DeepSpeed 原生支持 **Adam、AdamW、OneBitAdam、Lamb、OneBitLamb**，同时也可从 torch 导入其他优化器。
- **params**：实例化优化器的参数字典，**参数名必须与构造函数签名匹配**（如 `torch.optim.Adam` 的 `lr/betas/eps/weight_decay`）。

Adam 优化器示例（原文真料）：
```json
"optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [0.8, 0.999],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
}
```

特别参数 **torch_adam**：使用 torch 自带 adam 实现而非 DeepSpeed 的 fused adam，**默认为 false**。为什么默认用 fused adam？Adam 更新对每个参数做一串逐元素运算（$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$，$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$，$\theta_t = \theta_{t-1} - \eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$）；fused 版把这些 kernel 融合，减少 GPU kernel 启动与显存读写，**更快**。`torch_adam=true` 只在需对齐 PyTorch 数值或排查差异时用。

参考：optimizers https://deepspeed.readthedocs.io/en/latest/optimizers.html ；torch https://pytorch.org/docs/stable/optim.html

## 4. Scheduler 学习率调度参数

当执行 `model_engine.step()` 时，DeepSpeed 在**每个训练步骤**调用 scheduler 的 `step()`。

- **type**：调度器名。DeepSpeed 提供 **LRRangeTest、OneCycle、WarmupLR、WarmupDecayLR** 的实现。
- **params**：参数字典，名称应与调度器构造函数签名匹配。

scheduler 示例（原文真料）：
```json
"scheduler": {
    "type": "WarmupLR",
    "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 0.001,
        "warmup_num_steps": 1000
    }
}
```

WarmupLR 直觉：前 `warmup_num_steps` 步把学习率从 `warmup_min_lr` 线性升到 `warmup_max_lr`，之后保持。

```
lr 0.001 ┤        ┌────────  ← warmup_max_lr 之后恒定
         │      ╱  ← 线性 warmup
   0     ┼────╱──────────► step
         0  1000
```

为什么要 warmup？训练初期参数随机、梯度噪声大，直接用大 lr 易发散；先小步"热身"让二阶动量 $v_t$ 统计稳定，再放大 lr。`WarmupDecayLR` 则在 warmup 后再线性衰减到 0，适合有总步数的预训练。

## 5. 通讯选项

| 字段 | 作用 | 直觉 |
|------|------|------|
| **communication_data_type** | 梯度归约（all-reduce）时用的数据类型 | 用低精度（如 fp16）通信省带宽，但可能损精度 |
| **prescale_gradients** | 在归约**前**缩放梯度 | 防止 fp16 下梯度求和溢出 |
| **gradient_predivide_factor** | 归约前先除以的因子 | 大规模数据并行时把"先求和再除 N"拆开，避免中间和溢出 |
| **sparse_gradients** | 启用稀疏梯度通信 | 仅对 Embedding 等稀疏更新有效 |

为什么有 `gradient_predivide_factor`？数据并行 all-reduce 默认"先把 N 张卡梯度求和、再除 N 取平均"；当 N 大且用 fp16 时中间和可能溢出（fp16 最大约 65504），先按因子预除把数值压回安全区再求和，等效但更稳。

## 6. 混合精度训练选项

**三选一互斥**：FP16 / BF16 / AMP 三种模式**不能同时启用**；且 AMP 目前**与 ZeRO 不兼容**。

```
精度模式(任意时刻只能开一个): fp16(需动态 loss scaling) | bf16(范围同fp32,免缩放,需A100) | amp(Apex O1/O2,与ZeRO冲突)
```

### 6.1 FP16（原文真料）— 注意：此模式不能与下述 amp 模式结合使用

```json
"fp16": {
    "enabled": true,
    "auto_cast": false,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "consecutive_hysteresis": false,
    "min_loss_scale": 1
}
```

字段直觉：`loss_scale:0` 表示**动态损失缩放**（非 0 则固定）；`initial_scale_power:16` 初始缩放因子 $2^{16}=65536$；`loss_scale_window:1000` 连续 1000 步无溢出就尝试**放大**缩放；`hysteresis:2` 连续 2 次溢出才**减小**缩放（迟滞防抖）；`min_loss_scale:1` 缩放下限。

为什么 fp16 要 loss scaling？fp16 最小正规数约 $6\times10^{-5}$，小梯度会"下溢"成 0。把 loss 乘以大因子 $S$，梯度按链式法则同样放大 $S$ 倍，从下溢区拉回可表示区；优化器更新前再除回 $S$。若某步产生 Inf/NaN（上溢），就跳过该步并把 $S$ 减半。

```
真实梯度 1e-7 ──×S(65536)──► 6.5e-3 ✓ fp16 可表示 ──优化前÷S──► 还原 1e-7 用于更新
```

### 6.2 BFLOAT16（原文真料）— 注意：不能与 amp 结合，也不能与上述 fp16 结合

使用 bfloat16 作为 FP16 替代方案。**BFLOAT16 需要硬件支持（例如 NVIDIA A100）**。**使用 bfloat16 训练不需要损失缩放**。

```json
"bf16": {
   "enabled": true
}
```

为什么 bf16 不需要 loss scaling？bf16 与 fp32 **指数位都是 8 位**，动态范围一样大（约 $10^{\pm38}$），只是尾数少（7 位，精度低）。梯度不会下溢，所以省掉了 fp16 那套缩放机制——配置因此极简，只需 `enabled: true`。

| | fp16 | bf16 |
|--|------|------|
| 指数位 | 5 | 8（同 fp32）|
| 尾数位 | 10 | 7 |
| 动态范围 | 窄，需 loss scaling | 宽，**免** loss scaling |
| 硬件 | 较普遍 | 需 A100 / 较新架构 |

### 6.3 自动混合精度 AMP（原文真料）— 注意：不能与 fp16 结合，且**目前与 ZeRO 不兼容**

```json
"amp": {
    "enabled": true,
    "opt_level": "O1"
}
```

AMP 是 NVIDIA Apex 风格的自动混合精度（`O1` 局部 cast、`O2` 更激进）。因与 ZeRO 不兼容，DeepSpeed 训练**实践中几乎都用 fp16 或 bf16**，AMP 仅作兼容选项。

## 7. 梯度裁剪 Gradient Clipping

**gradient_clipping**：梯度全局范数的裁剪阈值。训练中偶发"梯度爆炸"会让一步更新冲飞。裁剪在 `step()` 前把所有梯度拼成大向量算 $L_2$ 范数 $g_{norm}$，若 $g_{norm} > c$ 则整体乘 $c/g_{norm}$ 缩回，方向不变、长度封顶：

$$ g \leftarrow g \cdot \min\Big(1, \frac{c}{\lVert g \rVert_2}\Big) $$

数值：阈值 $c=1.0$，某步 $g_{norm}=5$ ⇒ 缩放因子 $1/5=0.2$，所有梯度按 0.2 缩放，等效"踩刹车"。

## 8. ZeRO 优化与卸载

DeepSpeed 的看家本领：**ZeRO（Zero Redundancy Optimizer）** 把数据并行里每张卡重复存的"优化器状态 / 梯度 / 参数"切分到各卡，不再人人存全量。Stage 越高切得越多、省得越狠、通信越多。

```
        参数P 梯度G 优化器O      卸载: 把切到本卡的那份再丢去 CPU/NVMe
ZeRO-1   全量  全量  切分  ← 省一点
ZeRO-2   全量  切分  切分  ← 省更多
ZeRO-3   切分  切分  切分  ← 最省，三者全切
```

### 8.1 参数卸载 offload_param（原文真料）

启用并配置 ZeRO 优化，将参数卸载到 CPU/NVMe。**仅适用于 ZeRO 阶段 3**；若 `device` 未指定或不支持则**触发断言**。

```json
"offload_param": {
    "device": "[cpu|nvme]",
    "nvme_path": "/local_nvme",
    "pin_memory": [true|false],
    "buffer_count": 5,
    "buffer_size": 1e8,
    "max_in_cpu": 1e9
}
```

### 8.2 优化器卸载 offload_optimizer（原文真料）

启用并配置 ZeRO 优化，将优化器**计算卸载到 CPU**、优化器**状态卸载到 CPU/NVMe**。**CPU 卸载适用于 ZeRO 阶段 1、2、3；NVMe 卸载仅适用于阶段 3**。注意：`device` 未指定或不支持则触发断言。

```json
"offload_optimizer": {
    "device": "[cpu|nvme]",
    "nvme_path": "/local_nvme",
    "pin_memory": [true|false],
    "buffer_count": 4,
    "fast_init": false
}
```

字段直觉：`device` 卸载目的地（NVMe 容量大但慢）；`pin_memory` 页锁定内存让 CPU↔GPU 传输更快（多占物理内存）；`buffer_count/buffer_size` 多缓冲流水线深度与单缓冲大小（越大重叠越好越占内存）；`max_in_cpu` 常驻 CPU 的参数元素上限。

为什么卸载有用？以 7B 模型为例，Adam 在 fp32 下优化器状态约 $7\text{B}\times(4_{m}+4_{v}+4_{p_{fp32}})\approx 84$GB，单卡放不下。把这部分丢到 CPU 内存或 NVMe 盘，GPU 只在 `step()` 时按需取回，**显存换内存/磁盘**，让小卡也能训大模型——代价是搬运变慢。详见 [[docs/transformer内存估算]]。

```
 GPU 显存(贵,小) ──offload──► CPU 内存(中) ──offload──► NVMe 盘(便宜,大)
   活跃参数      ◄──用时取回── 优化器状态  ◄────────── 优化器状态(溢出)
```

## 9. 激活重计算 Activation Checkpointing（原文真料）

```json
"activation_checkpointing": {
    "partition_activations": false,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
}
```

原理：前向时**不保存**所有中间激活，只在分界点存少量"检查点"；反向需要某段激活时**重新前向算一遍**。用算力换显存。字段：`partition_activations` 配合模型并行把激活切分到各卡；`cpu_checkpointing` 把检查点激活进一步卸载到 CPU；`contiguous_memory_optimization` 让检查点激活连续存放减少碎片。

数值直觉：$L$ 层网络，普通方式存 $O(L)$ 份激活；分段重计算可降到约 $O(\sqrt L)$ 份，额外多算约一次前向。

## 10. 其余模块

### 10.1 稀疏注意力 Sparse Attention（原文真料）

```json
"sparse_attention": {
 "mode": "fixed",
 "block": 16,
 "different_layout_per_head": true,
 "num_local_blocks": 4,
 "num_global_blocks": 1,
 "attention": "bidirectional",
 "horizontal_global_attention": false,
 "num_different_global_patterns": 4,
 "num_random_blocks": 0,
 "local_window_blocks": [4],
 "global_block_indices": [0],
 "global_block_end_indices": None,
 "num_sliding_window_blocks": 3
}
```

思路：标准注意力是 $O(n^2)$ 全连接；把序列切成 `block`（如 16），只让每块关注"局部窗口 + 少量全局/随机块"，把稠密注意力变稀疏降复杂度。延伸 [[llm-optimizer/FlashAttention]]。

### 10.2 Logging

字段：`steps_per_print` 每多少步打印进度/loss；`wall_clock_breakdown` 打印各阶段（fwd/bwd/step）墙钟耗时（**性能调优必开**）；`dump_state` 初始化后转储引擎状态，用于排查 config 是否被正确解析。

### 10.3 Flops 分析器 Flops Profiler（原文真料）

字段：`detailed` 是否打印详细模型配置；`output_file` 输出文件路径（None 则打印到标准输出）。

```json
"flops_profiler": {
    "enabled": false,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
}
```

用途：自动统计模型 FLOPs、参数量、各算子耗时，定位计算瓶颈，配合 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] 一起读。

### 10.4 监控模块 TensorBoard / WandB / CSV（原文真料）
```json
"tensorboard": {
    "enabled": true,
    "output_path": "output/ds_logs/",
    "job_name": "train_bert"
}
```

DeepSpeed 内置三种监控后端（TensorBoard / WandB / CSV），开启后自动把 loss、lr、吞吐等指标写出，无需在训练代码里手动埋点。

### 10.5 压缩 Compression

DeepSpeed Compression 提供一组训练期压缩手段，均在 config 里声明：

| 子模块 | 做什么 | 相关笔记 |
|--------|--------|----------|
| Layer Reduction | 知识蒸馏式减层 | [[llm-compression/README]] |
| 权重量化 Weight Quantization | 权重降位宽 | [[llm-compression/quantization/量化基础]] |
| 激活量化 Activation Quantization | 激活降位宽 | [[llm-compression/quantization/fp8]] |
| 稀疏剪枝 Sparse Pruning | 按比例置零权重 | [[llm-compression/README]] |
| 头剪枝 Head Pruning | 裁掉注意力头 | [[llm-algo/transformer/模型架构]] |
| 通道剪枝 Channel Pruning | 裁掉卷积/线性通道 | [[llm-compression/README]] |

### 10.6 Checkpoint 选项（原文真料）

```json
"checkpoint": {
    "tag_validation": "Warn",
    "load_universal": false,
    "use_node_local_storage": false,
    "parallel_write": {
        "pipeline_stage": false
    }
}
```

字段：`tag_validation` 标签校验级别（Warn/Ignore/Fail）；`load_universal` 加载"通用 checkpoint"可跨并行度迁移；`use_node_local_storage` 用节点本地盘存 checkpoint 减共享存储压力；`parallel_write.pipeline_stage` 流水线并行时并行写盘加速。

### 10.7 数据类型选项（原文真料）

```json
"data_types": {
    "grad_accum_dtype": ["fp32"|"fp16"|"bf16"]
}
```

`grad_accum_dtype` 控制**梯度累积**用的精度。bf16 训练时常把累积放在 fp32 以保数值稳定——计算用 bf16 省显存，累加用 fp32 防误差堆积。

### 10.8 Data Efficiency

提供课程学习（curriculum learning）与数据采样优化，声明后可用更少数据/步数达到同等效果（详见官方文档）。

## 实操：拼一份最小可用 config

把上面真料按需拼装，下面是"ZeRO-2 + bf16 + WarmupLR"常见组合骨架（字段均来自上文真料）：
```json
{
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "bf16": { "enabled": true },
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 3e-7 }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": { "warmup_min_lr": 0, "warmup_max_lr": 0.001, "warmup_num_steps": 1000 }
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu", "pin_memory": true }
  },
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

校验铁律：若 8 卡运行，则 `train_batch_size = 8 × 4 × 8 = 256`，可不写让 DeepSpeed 反推，但一旦你显式写错就会报错。

## 常见问题 / 坑

| 现象 / 报错 | 根因 | 解法 |
|------------|------|------|
| `train_batch_size` 断言失败 | 三件套不满足 `global = micro × accum × gpu` | 改其中之一，或只填两个让引擎推断 |
| 开了 fp16 又开 amp/bf16 报错 | 三种精度**互斥** | 只留一个 |
| bf16 训练 NaN | 硬件不支持 bf16，或累积精度不足 | 确认 A100 级硬件；`grad_accum_dtype` 设 fp32 |
| `offload_param` 不生效 | 它**仅 ZeRO-3** 可用 | 升到 stage 3，或改用 `offload_optimizer` |
| device 未指定触发断言 | `offload_*` 的 `device` 必填且须支持 | 显式写 `"cpu"` 或 `"nvme"`（nvme 还要 `nvme_path`）|
| AMP + ZeRO 报不兼容 | AMP **目前与 ZeRO 不兼容** | 换 fp16/bf16 |
| fp16 loss 一直缩放到 min 仍 NaN | 梯度真的爆炸 | 加 `gradient_clipping`、调小 lr |
| 卸载后训练巨慢 | NVMe/CPU↔GPU 搬运成瓶颈 | 优先 CPU 卸载、开 `pin_memory`、加大 `buffer_count` |
| 参数名报 `unexpected keyword` | optimizer `params` 名与构造函数签名不符 | 对照 torch.optim 文档改名 |

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 同框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]] · [[ai-framework/pytorch/README]]
- 训练/对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 显存与算子：[[docs/transformer内存估算]] · [[llm-optimizer/FlashAttention]] · [[llm-optimizer/kv-cache]]
- 模型结构：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 压缩量化：[[llm-compression/README]] · [[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 推理侧：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 硬件网络：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 评测：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
