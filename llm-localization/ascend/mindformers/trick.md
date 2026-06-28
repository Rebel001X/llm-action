# MindFormers 训练调优 Trick:批大小、并行配比与吞吐

> 在昇腾 MindFormers 大模型训练套件里,把 `global_batch_size` 拆成 DP/PP/micro-batch/多副本几个因子,并由此推出每卡每秒吞吐(throughput),是所有性能调优的算账起点。📍 导航:[[00-知识地图]]
> 🔗 相关:[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] [[ai-framework/megatron-lm/README]] [[llm-train/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|--------------|--------|
| 0 | 一句话锚点 | global_batch_size 公式 |
| 1 | MindFormers 在昇腾栈的定位 + 昇腾↔英伟达对照 | 套件层 / Megatron 对标 |
| 2 | global_batch_size 五因子拆解 | DP / PP / micro / 多副本 |
| 3 | 吞吐 throughput 怎么算 | samples/s/p |
| 4 | 五个并行维度与显存的关系(ASCII 图) | DP/TP/PP/SP/OP |
| 5 | micro_batch_num 与流水线气泡 | bubble / 1F1B |
| 6 | micro_batch_interleave_num 多副本 | 通信隐藏 |
| 7 | 与 DeepSpeed 记账方式对照 | gradient_accumulation |
| — | 迁移要点 / 注意事项与坑 | 从 Megatron 迁来 |
| — | 常见问题 / 跳转链接 | FAQ |

## 0. 一句话锚点

```
global_batch_size = batch_size × data_parallel × micro_batch_num × micro_batch_interleave_num
                  = 16        = 2          × 1             × 8               × 1

throughput (samples/s/p) = global_batch_size / device_num / (per_step_seconds / 1000)
```

- `global_batch_size`:一个优化器 step(一次参数更新)实际"吃进"的样本总数,决定梯度的统计有效性。
- `throughput`:每一步、每一卡、每一秒能处理的样本数(samples per second per processor/卡),是横向比较硬件/配置效率的硬指标。
- **调优的本质**:在显存装得下的前提下,把 `global_batch_size` 顶上去、把 `per_step_seconds` 压下来,从而抬高 throughput。下面逐个因子拆开讲它们各自影响什么。

## 1. 地基:MindFormers 在昇腾栈的定位 + 对标英伟达生态

**MindFormers 是什么、在哪一层**:它是华为昇腾上的大模型训练/微调/推理一体化套件,定位等价于英伟达世界的 **Megatron-LM + 部分 HuggingFace Transformers**。它向上提供"配置即模型"(YAML 描述结构+并行+优化器),向下调用 MindSpore 框架,再由 MindSpore 把图编译成 CANN 算子,最终落到昇腾 NPU 的达芬奇 Cube/Vector 单元执行。

昇腾软件栈自下而上的四层:

```
┌───────────────────────────────────────────────┐
│ 套件层   MindFormers / ModelLink   ← 本文在这里 │  对标 Megatron-LM
├───────────────────────────────────────────────┤
│ 框架层   MindSpore (动态图/静态图/PyNative)     │  对标 PyTorch
├───────────────────────────────────────────────┤
│ 异构计算 CANN (算子库 + 图引擎 GE + HCCL通信)   │  对标 CUDA + cuDNN + NCCL
├───────────────────────────────────────────────┤
│ 硬件层   昇腾 NPU(达芬奇架构 Cube/Vector)      │  对标 GPU(Tensor Core)
└───────────────────────────────────────────────┘
```

**昇腾 ↔ 英伟达生态对照表(迁移心智图)**:

| 维度 | 昇腾(华为) | 英伟达 | 说明 |
|------|------------|--------|------|
| 加速硬件 | NPU(达芬奇架构) | GPU | Cube 单元 ≈ Tensor Core |
| 异构计算平台 | CANN | CUDA | 编译/运行时/驱动总成 |
| 算子/数学库 | CANN 算子库(AOL/aclnn) | cuDNN / cuBLAS | 卷积、GEMM、Attention 等 |
| 集合通信 | HCCL | NCCL | AllReduce / AllGather 等原语 |
| 深度学习框架 | MindSpore | PyTorch | 也支持 PyTorch+torch_npu |
| 训练大模型套件 | **MindFormers** / ModelLink | **Megatron-LM** | 并行训练、配置化建模 |
| 推理引擎 | MindIE | TensorRT-LLM / vLLM | 推理服务化 |
| 量化压缩工具 | msModelSlim | GPTQ / AWQ 工具 | 权重量化 |
| 性能采集 | msprof / MindStudio Insight | Nsight / nvprof | profiling |

> 一句话:把"Megatron 上跑大模型"的经验整体平移过来,MindFormers 的并行抽象(DP/TP/PP)几乎一一对应,主要差异在框架(MindSpore vs PyTorch)和底层通信/算子(HCCL/CANN vs NCCL/CUDA)。

## 2. global_batch_size 五因子拆解

```
global_batch_size = batch_size × data_parallel × micro_batch_num × micro_batch_interleave_num
```

| 因子 | 含义 | 调它影响什么 | 何时用 |
|------|------|--------------|--------|
| `batch_size` | 单个数据通路一次前向的样本数 | 直接放大显存占用与计算量 | 总在用 |
| `data_parallel` | 数据并行路数(DP) | 线性扩样本数,几乎不增单卡显存(权重各卡都有一份) | 卡多时首选 |
| `micro_batch_num` | 流水线并行(PP)的微批次个数 | 把一个 batch 切成多个 micro-batch 喂入流水线,摊薄气泡 | `pipeline_stage > 1` 时 |
| `micro_batch_interleave_num` | batch_size 的再拆份数(多副本并行) | 用计算与通信重叠来隐藏模型并行(TP)的通信开销 | 开 model_parallel 时 |

**约束关系(易踩坑)**:
- 开流水并行时需满足 `micro_batch_num >= pipeline_stage`,否则流水线填不满,气泡占比极高,等于白白浪费算力。
- `micro_batch_interleave_num` 在**纯流水并行**时不建议开:它的收益来自隐藏 TP 的通信,纯 PP 场景没有这层通信可隐藏,反而徒增调度复杂度。
- 五个因子的乘积必须能被数据集/采样器整除,否则最后一个 step 数据不足会触发补齐或丢弃,影响 loss 统计。

## 3. 吞吐 throughput 怎么算

```
# compute throughput (samples/s/p) 每一步、每一卡、每一秒能处理的样本数
throughput = global_batch_size / device_num / (per_step_seconds / 1000)
```

- 分子 `global_batch_size`:一步处理的总样本。
- `device_num`:参与训练的卡数,除以它得到"每卡"。
- `per_step_seconds / 1000`:单步耗时(毫秒转秒),除以它得到"每秒"。

直觉:**throughput 越高,等价于训练越快、同样预算下能跑更多 token**。优化方向有二——
1. 抬分子:在显存允许下增大 micro_batch / 序列长度利用率;
2. 压分母:减小 `per_step_seconds`,手段包括减小通信(合理切 TP/PP)、用多副本隐藏通信、开融合算子、用 BF16/FP16 混合精度让 Cube 单元打满。

注意 throughput 是 **samples/s/p**;若想换算成业界常说的 **tokens/s**,再乘以序列长度 `seq_length` 即可。

## 4. 五个并行维度与显存的关系

大模型训练把"模型 + 数据"沿不同轴切开。MindFormers 与 Megatron 的并行语义一致:

```
                     一个大模型 + 一批数据
                              │
        ┌──────────────┬──────┴───────┬───────────────┐
       DP             TP             PP              SP/OP
   数据并行        张量并行        流水线并行     序列/优化器并行
        │              │              │               │
  复制整模型     切单层权重      按层切成 stage   切序列维 / 切优化器状态
  样本分到各卡   层内通信(TP)   stage 间传激活   降激活/状态显存
        │              │              │               │
  ─────────────────────────────────────────────────────────
   省不了显存     省权重显存      省权重显存      省激活/状态显存
   只扩吞吐       但通信最重      引入流水气泡    通信换显存
```

| 并行 | 缩写 | 切什么 | 主要省什么 | 主要代价 | 对标 Megatron |
|------|------|--------|-----------|----------|---------------|
| 数据并行 | DP / `data_parallel` | 数据 | — | 梯度 AllReduce | DP |
| 张量并行 | TP / `model_parallel` | 层内权重 | 权重显存 | 层内高频通信 | TP |
| 流水线并行 | PP / `pipeline_stage` | 按层分段 | 权重显存 | 流水气泡 | PP |
| 序列并行 | SP | 序列维 | 激活显存 | AllGather/ReduceScatter | SP |
| 优化器并行 | OP(类 ZeRO) | 优化器状态 | 状态显存 | 状态聚合通信 | ZeRO/分布式优化器 |

调优口诀:**先用 DP 扩吞吐;单卡装不下了再加 TP(优先放在单机 8 卡内,因为 TP 通信最频繁,要走机内高速互联);层数太多权重还装不下,再上 PP;激活/优化器状态吃紧,再叠 SP/OP。**

## 5. micro_batch_num 与流水线气泡

流水线并行把模型按层切成若干 stage,放在不同卡上。若一次只喂一个大 batch,前面 stage 算完在等后面 stage,形成"气泡"(bubble),利用率低。把 batch 拆成 `micro_batch_num` 个微批,让它们像流水线一样首尾相接,就能摊薄气泡:

```
stage0: [m1][m2][m3][m4]........[反向]
stage1: ...[m1][m2][m3][m4]....[反向]
stage2: ......[m1][m2][m3][m4][反向]
            ↑填充期        ↑稳态(高利用)↑排空期
   micro 越多,稳态越长,首尾气泡占比越小
```

- 气泡占比 ≈ `(pipeline_stage - 1) / (micro_batch_num + pipeline_stage - 1)`。
- 所以 `micro_batch_num` 越大,气泡越小,但每个 micro-batch 太小又会让单算子打不满 Cube 单元——需要折中。
- 这也是 `micro_batch_num >= pipeline_stage` 这条硬约束的由来:至少要填满一遍流水线。

## 6. micro_batch_interleave_num:多副本并行隐藏 TP 通信

多副本并行(multi-copy / interleaved)把一个 `batch_size` 再切成 `micro_batch_interleave_num` 份,在张量并行(TP)产生通信时,用另一份的计算去**掩盖通信延迟**,提升 NPU 利用率:

```
单副本:  [计算A]→[通信A 等待]→[计算B]→[通信B 等待]   ← 通信时算力空转
多副本:  [计算A][计算B]
                └[通信A]┘ 与计算B重叠,通信被隐藏    ← 算力不空转
```

- **收益来源**:TP 的层内 AllReduce/AllGather 在 model_parallel 时频繁发生,多副本让"算"和"传"并行,等价于把 HCCL 通信藏到计算背后。
- **何时开**:开了 `model_parallel`(TP)时考虑;纯流水并行场景没有这层通信,不建议开。
- **代价**:每份变小,可能让单个 GEMM 打不满 Cube;切份数需实测找甜点。

## 7. 与 DeepSpeed 记账方式对照

很多人从 GPU + DeepSpeed 迁来,记账名词不同但本质一致:

```
DeepSpeed:
global_train_batch_size = train_micro_batch_size_per_gpu
                        × gradient_accumulation_steps
                        × number_of_GPUs
```

| 概念 | MindFormers | DeepSpeed | 备注 |
|------|-------------|-----------|------|
| 单通路微批 | `batch_size` | `train_micro_batch_size_per_gpu` | 一次前向样本数 |
| 梯度累积 | 由 `micro_batch_num`(PP)体现 | `gradient_accumulation_steps` | 攒多步再更新 |
| 数据并行路数 | `data_parallel` | `number_of_GPUs`(纯 DP 时) | 扩样本数 |
| 全局批 | `global_batch_size` | `global_train_batch_size` | 一次参数更新的总样本 |

差异点:MindFormers 把"梯度累积"与"流水线微批"统一到 `micro_batch_num` 这一概念里(PP 的微批天然就是累积),而 DeepSpeed 把 `gradient_accumulation_steps` 单列。理解了这层映射,Megatron/DeepSpeed 的配置经验就能平移。

## 迁移要点 / 注意事项与坑

1. **并行语义可平移,但底层通信换了引擎**:DP 的 AllReduce、TP 的 AllGather 在昇腾上由 **HCCL** 承担(对标 NCCL)。HCCL 的环/树算法、机内 vs 跨机带宽差异会显著影响 `per_step_seconds`——TP 尽量压在单机内,跨机优先走 DP/PP。
2. **混合精度让 Cube 单元打满**:达芬奇 Cube 对 FP16/BF16 矩阵乘有专门通路。务必开混合精度,否则 throughput 上不去(对标 GPU 的 Tensor Core 必须喂半精度)。
3. **`micro_batch_num >= pipeline_stage` 是硬约束**,违反会让流水线填不满,气泡吞掉收益。
4. **多副本只在有 TP 通信可隐藏时才开**,纯 PP 别开。
5. **五因子乘积要能整除数据量与设备数**,否则末步补齐影响 loss 统计与吞吐计算。
6. **算 throughput 时分母用稳态 per_step_seconds**,跳过前几个 warmup step(图编译、缓存预热会让前几步偏慢),否则低估吞吐。
7. **环境/镜像/版本类一律以官方为准**:MindFormers、MindSpore、CANN 三者有严格版本配套关系,装错会出现算子不支持或精度异常。**具体安装命令、镜像名、配套版本号以华为昇腾官方文档(Ascend 社区)为准**——本文只讲"为什么要对齐版本"(套件依赖框架、框架依赖 CANN、CANN 依赖驱动,任一层错位都会编译/运行失败),不给可能过期的具体命令。

## 常见问题

| 问题 | 解答 |
|------|------|
| global_batch_size 越大越好吗? | 不是。受显存上限约束,且过大会改变优化器的统计特性(等效学习率/收敛),需配合 lr 调整。 |
| throughput 单位 samples/s/p 怎么换 tokens/s? | 乘以 `seq_length`,再乘 `device_num` 得全局 tokens/s。 |
| 为什么我的吞吐比理论低很多? | 多半是通信瓶颈(TP 跨机)、气泡(micro 太少)、或没开混合精度/融合算子;先用 msprof profiling 定位。 |
| micro_batch_num 和 micro_batch_interleave_num 区别? | 前者服务流水线并行(填满流水、摊薄气泡);后者服务张量并行(多副本隐藏通信)。两者解决不同瓶颈。 |
| 从 Megatron 迁来配置改动大吗? | 并行维度概念一一对应,主要改框架写法(MindSpore/YAML)与底层依赖;并行配比经验可直接复用。 |
| 纯流水并行要不要开多副本? | 不要,没有 TP 通信可隐藏,反增复杂度。 |

## 🔗 跳转链接

- 枢纽:[[00-知识地图]]
- 硬件与生态:[[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]] · [[ai-infra/ai-hardware/CUDA]]
- 通信:[[ai-infra/网络/NCCL]] · [[ai-infra/网络/集合通信原语]]
- 训练框架/套件:[[ai-framework/megatron-lm/README]] · [[ai-framework/huggingface-transformers/README]]
- 训练与算法:[[llm-train/README]] · [[llm-algo/transformer/模型架构]]
- 推理与压缩:[[llm-inference/README]] · [[llm-compression/quantization/量化基础]]
