# tq-llm 训练子系统：大模型微调任务的启动与参数

> 一句话定位：tq-llm 的 `train` 模块是一个「大模型微调任务启动器」——把一条 `bootstrap-llm.sh` 命令行翻译成一次可在集群上运行的 SFT / LoRA 训练作业，统一管理数据、模型、输出、指标与显存。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] [[ai-framework/deepspeed/README]] [[llmops/README]] [[llmops/kubernetes]]

## 阅读地图

| 小节 | 你将搞清楚 |
| --- | --- |
| 0. 一句话锚点 | 这个 `train` 模块到底干啥 |
| 1. 地基 | 为什么需要一个「启动脚本」而不是手敲 torchrun |
| 2. 整体架构 | 从命令行到 GPU 进程的调用链 |
| 3. bootstrap 参数全景 | 每个 `--xxx` 是干什么用的、怎么权衡 |
| 4. 全量 SFT vs LoRA | 两种微调路线的取舍 |
| 5. 降显存四件套 | gradient_checkpointing / 累积 / batch / seq_length |
| 6. S3 数据与模型流转 | 训练前拉取、训练后回传 |
| 典型流程 | 一次完整作业的生命周期 |
| 常见坑 | 易踩的雷与排查 |

## 0. 一句话锚点

`train` 模块 = **一层「人话 → 训练进程」的转换器**。
你写：`sh bootstrap-llm.sh --sft_type=full --gpu_num=2 ...`；
它负责：装环境 → 从对象存储拉数据/基座模型 → 拼出底层分布式训练命令（`torchrun` / `deepspeed`）→ 跑训练 → 把权重和指标回传到对象存储。

> 注意：本文讲的是该启动器的**通用机制与参数含义**。具体参数的精确默认值、脚本内部实现，请以 `tq-llm/train/` 下的实际脚本与官方文档为准，下文不臆造默认值。

## 1. 地基：为什么要一个启动脚本

裸手跑一次大模型微调，你至少要操心这些事：

```
环境       Python/CUDA/PyTorch 版本对齐、装 transformers/peft/deepspeed
数据       从哪拉、什么格式、要不要切分
基座模型    几十 GB 权重从对象存储下载到本地盘
分布式      torchrun 的 --nproc_per_node / --nnodes / rdzv 配置
显存        OOM 了改哪几个旋钮
产物        权重存哪、指标(loss/lr)写哪、谁来收
```

如果每个算法同学都自己写一遍，会出现：参数命名五花八门、忘了回传指标、环境对不齐、复现困难。
**启动脚本的价值 = 把上面这堆「样板工程」收敛成一组稳定的命名参数**，让使用者只关心「数据 + 模型 + 超参」三件事，其余交给脚本。这是 LLMOps 里最朴素也最高频的一层抽象。

$$\text{使用成本} \;=\; \underbrace{f(\text{业务超参})}_{\text{你该关心的}} \;+\; \underbrace{C_{\text{样板}}}_{\text{脚本帮你吃掉}}$$

## 2. 整体架构：从命令行到 GPU 进程

```
            用户 / 调度平台
                 │  sh bootstrap-llm.sh --sft_type=full --gpu_num=2 ...
                 ▼
        ┌───────────────────────┐
        │   bootstrap-llm.sh    │  ① 解析参数 / 激活 conda 环境
        │   (启动器/胶水层)      │  ② 校验路径、补默认值
        └───────────┬───────────┘
                    │
        ┌───────────┼────────────────────────────┐
        │           │                            │
        ▼           ▼                            ▼
   拉数据/模型   拼分布式命令                  回传产物
  (S3 → 本地)  torchrun/deepspeed             (本地 → S3)
        │           │                            ▲
        │           ▼                            │
        │   ┌────────────────────┐               │
        │   │  train.py 训练入口  │──── 写 ───────┤
        │   │  HF Trainer / DS    │   权重 + 指标 │
        │   └─────────┬──────────┘               │
        │             │                          │
        ▼             ▼                          │
   /local/data   GPU0  GPU1 ... GPUn ────────────┘
                 (DDP / ZeRO 数据并行)
```

关键点：启动器本身**不做反向传播**，它只是「准备 → 拼命令 → 收尾」。真正的训练在 `train.py`（HuggingFace `Trainer` 或 DeepSpeed 引擎）里跑，由 `torchrun` 在 `gpu_num` 个进程上并行。

> 对应原始命令行（来自本目录草稿）：
> ```sh
> sh /task/script/bootstrap-llm.sh --sft_type=full --conda_env=torch1131-venv \
>   --train_dataset_path=s3://.../train_data_1k.json,s3://.../train_data_100.json \
>   --pre_model_path=s3://.../model-bloom-2b6 \
>   --checkpoint_path=s3://.../model-bloom-2b6 \
>   --model_output_path=s3://.../model-bloom-2b6 \
>   --model_metrics_path=s3://.../processx.json \
>   --gpu_num=2 \
>   --epoch=1 --batch_size=8 --learning_rate=1e-5 \
>   --max_seq_length=512 --logging_steps=1 --warmup_ratio=0.1 --weight_decay=0
> ```

## 3. bootstrap 参数全景（讲含义，不背默认值）

把参数按「职责」分四类来记，比死记顺序好用得多。

### 3.1 环境 & 模式类

| 参数 | 作用 | 权衡 / 说明 |
| --- | --- | --- |
| `--sft_type` | 微调类型，常见取值 `full` / `lora` | `full` 改全部权重，效果上限高但贵；`lora` 只训练低秩旁路，省显存省盘 |
| `--conda_env` | 指定要激活的 conda/venv 环境名 | 保证 PyTorch/CUDA 版本与镜像一致，避免 ABI 不匹配 |
| `--gpu_num` | 单机使用的 GPU 数（数据并行进程数） | 等价于 `torchrun --nproc_per_node`；越多吞吐越高，但要够显存放下模型副本 |

### 3.2 路径类（数据进、模型进出）

| 参数 | 作用 | 说明 |
| --- | --- | --- |
| `--train_dataset_path` | 训练集路径，**支持逗号分隔多个** | 多文件会被拼接/采样成一个训练集；通常是对象存储 `s3://` 地址 |
| `--pre_model_path` | 基座（预训练）模型路径 | 微调的起点权重，如 `model-bloom-2b6` |
| `--checkpoint_path` | 断点续训的检查点目录 | 任务中断后从这里恢复优化器/调度器状态；首训可与基座相同或留空 |
| `--model_output_path` | 训练完成后**权重回传**的目标路径 | 收尾阶段把本地产物 push 回对象存储 |
| `--model_metrics_path` | 指标文件（如 `processx.json`）写入路径 | 存 loss/lr/进度，供平台做曲线和进度展示 |

> 经验提醒：示例里 `pre_model_path / checkpoint_path / model_output_path` 指向**同一个** S3 前缀。生产中建议把「输出」与「输入」分目录，避免覆盖基座权重。

### 3.3 训练超参类

| 参数 | 作用 | 直觉 |
| --- | --- | --- |
| `--epoch` | 训练轮数 | 小数据多看几遍；大数据 1~3 轮常够，过多易过拟合 |
| `--batch_size` | 每个 GPU 的微批大小 | 越大越稳越快，但吃显存（详见第 5 节） |
| `--learning_rate` | 学习率，如 `1e-5` | 全量 SFT 一般取小（$10^{-5}$ 量级）；LoRA 可大一两个数量级 |
| `--max_seq_length` | 截断/填充的最大序列长度，如 `512` | 显存随它**近似平方**增长（注意力 $O(L^2)$） |
| `--warmup_ratio` | 预热步数占比，如 `0.1` | 前 10% 步把 lr 从 0 线性升到峰值，稳住早期训练 |
| `--weight_decay` | 权重衰减（L2 正则） | `0` 表示不加；适度正则可抗过拟合 |
| `--logging_steps` | 每多少步打一次日志/指标 | `1` 表示每步都记，调试方便但 I/O 多 |

学习率调度直觉（warmup + 衰减）：

```
 lr
  │        ___________
  │      /            \___
  │    /                  \____
  │  /  ← warmup_ratio        \___  ← 后期衰减
  │/   (前10%线性升)              \____
  └──────────────────────────────────────► step
```

### 3.4 全局有效 batch 的换算

很多人会困惑「我设了 batch_size=8，到底一步喂了多少样本？」记住这个公式：

$$\text{global\_batch} = \text{batch\_size} \times \text{gpu\_num} \times \text{grad\_accum\_steps}$$

数值例子：`batch_size=8`、`gpu_num=2`、若梯度累积 `=4`，则一次参数更新看到 $8\times2\times4 = 64$ 条样本。**调显存时改前三个旋钮，但要意识到它们一起决定了「等效 batch」**，进而影响学习率该取多大。

## 4. 全量 SFT vs LoRA：两条微调路线

```
            全量 SFT (full)                     LoRA
   ┌─────────────────────────┐      ┌─────────────────────────────┐
   │  W  (d×d, 全部可训练)    │      │  W (冻结)  +  B·A (低秩可训) │
   │  ▲ 更新整张权重           │      │            ▲ 只训练 r 维旁路 │
   └─────────────────────────┘      └─────────────────────────────┘
   显存: 权重+梯度+优化器态×全参    显存: 只为 A,B (≪全参) 存梯度/态
   产物: 一份完整大模型            产物: 一个几 MB~几十 MB 的适配器
```

LoRA 的核心思想：冻结原权重 $W$，只学一个低秩增量 $\Delta W = BA$，其中 $A\in\mathbb{R}^{r\times d}$、$B\in\mathbb{R}^{d\times r}$，秩 $r$ 通常取 8/16/64：

$$W' = W + \frac{\alpha}{r}\,BA$$

| 维度 | 全量 SFT | LoRA |
| --- | --- | --- |
| 可训练参数 | 100% | 通常 < 1% |
| 显存占用 | 高（优化器态 ×全参） | 低 |
| 产物大小 | 整个模型（GB 级） | 适配器（MB 级） |
| 效果上限 | 更高 | 略低，但多数任务够用 |
| 多任务切换 | 各存一份大模型 | 共享基座 + 换适配器 |

选择直觉：**资源紧 / 任务多 / 要快速试错 → LoRA；追求极致效果且资源充足 → full。**

## 5. 降显存四件套（草稿里点名的旋钮）

当遇到 CUDA OOM，按「副作用从小到大」依次调这四个：

```
显存压力大？按顺序拧旋钮：

① gradient_checkpointing  ─ 用算力换显存：前向不存全部激活，
   (重计算激活)              反向时重算 → 省激活显存，慢 ~20-30%

② gradient_accumulation    ─ 拆 batch：累积 N 个微批的梯度再更新，
   _steps  (梯度累积)         等效大 batch 但峰值显存只占 1 个微批

③ batch_size ↓             ─ 直接减每步样本数：最直接，但吞吐降

④ max_seq_length ↓         ─ 砍序列长度：注意力显存 ~O(L²)，
   (seq_length)              砍一半能省一大块，但截断长样本
```

权衡口诀：
- `gradient_checkpointing` —— **省显存最划算**，代价只是慢一点，优先开。
- `gradient_accumulation_steps` —— 想要大 batch 的训练稳定性又放不下时用它，**不降等效 batch**。
- `batch_size` / `seq_length` —— 最后才动，因为它们直接影响吞吐或样本完整性。

数值直觉：序列从 512 砍到 256，注意力相关显存约降到 $\left(\frac{256}{512}\right)^2 = \frac14$；而 `batch_size` 减半，激活显存大致线性减半。

## 6. S3 数据与模型流转

启动器在「训练前后」各做一次对象存储 I/O，这是云上训练区别于本地的关键环节：

```
 训练前(拉取)                 训练中                 训练后(回传)
 s3://.../train.json ─┐                          ┌─► s3://.../output (权重)
 s3://.../bloom-2b6  ─┼─► /local/scratch ──训练──┤
                      │   (本地高速盘)            └─► s3://.../metrics.json
 (多个数据用逗号分隔)─┘                              (loss/lr 曲线给平台用)
```

为什么要先落本地盘？因为训练对随机读极敏感，**直接读对象存储会被网络延迟拖垮**；先一次性下载到本地 NVMe，训练全程读本地，结束再批量回传。`model_metrics_path` 指向的 JSON 会被平台轮询，用于画进度条和 loss 曲线。

## 典型流程：一次作业的生命周期

```
1. 平台/用户提交  sh bootstrap-llm.sh --sft_type=full ...
2. 启动器激活 conda_env，校验各路径可达
3. 从 S3 拉 train_dataset_path（逗号分隔→合并）+ pre_model_path 到本地
4. 若 checkpoint_path 有断点 → 恢复优化器/lr 调度器状态
5. 拼底层命令：torchrun --nproc_per_node=$gpu_num train.py \
        --epoch --batch_size --learning_rate --max_seq_length ...
6. 训练循环：每 logging_steps 步把 loss/lr 写进 model_metrics_path
7. 训练结束：本地权重 push 到 model_output_path
8. 平台读 metrics 画曲线、标记任务完成
```

## 常见问题 / 坑

| 现象 | 可能原因 | 处理方向 |
| --- | --- | --- |
| 一启动就 CUDA OOM | seq_length/batch 太大或没开 checkpointing | 按第 5 节四件套依次拧 |
| loss 一直 NaN | 学习率过大 / 数据有脏 token | 降 lr、检查数据、确认 warmup 生效 |
| 多卡只用到 1 张 | `gpu_num` 没传对，或 torchrun 进程数与可见卡不符 | 核对 `gpu_num` 与 `CUDA_VISIBLE_DEVICES` |
| 输出覆盖了基座模型 | output 与 pre_model 同一 S3 前缀 | 输入输出分目录 |
| 断点续训没生效 | `checkpoint_path` 为空或目录里没有 optimizer 状态 | 确认上次保存了完整检查点 |
| 指标曲线不更新 | `model_metrics_path` 写入失败/路径错 | 检查 S3 权限与 `logging_steps` |
| LoRA 效果不如预期 | 秩 r 太小 / lr 没调大 | 提高 r、LoRA 学习率比全量大 1~2 个量级 |
| 环境 ABI 报错 | `conda_env` 与镜像 CUDA/torch 不匹配 | 锁定环境名，统一基础镜像 |

> 反复强调：以上参数名（`--sft_type` 等）来自本目录草稿示例，**精确默认值与脚本内部行为请以 `tq-llm/train/` 实际源码及官方文档为准**，切勿照搬本文当作准确默认配置。

## 🔗 跳转链接

- 返回总图：[[00-知识地图]]
- 训练总览：[[llm-train/README]]
- 分布式训练框架：[[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]
- PyTorch 多机/分布式：[[llm-train/pytorch/distribution/README]]
- 对齐与 SFT 上游：[[llm-alignment/RLHF]]
- LLMOps 平台：[[llmops/README]] · [[llmops/kubernetes]]
