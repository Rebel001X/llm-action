# HelloDeepSpeed：从原生 PyTorch 到 DeepSpeed 的 BERT 预训练第一课

> 用一个最小可跑的 BERT 预训练例子，对照看清「原生 HF 训练循环」与「接入 DeepSpeed 后」的差异，理解 ZeRO 切分后权重/优化器状态如何落盘。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/deepspeed/README]] · [[ai-framework/pytorch/README]] · [[ai-framework/megatron-lm/README]] · [[llm-train/README]]

---

## 阅读地图

| 小节 | 你将搞懂 | 关键词 |
| --- | --- | --- |
| 0. 一句话锚点 | 这个例子到底在演示什么 | HelloDeepSpeed / 对照 |
| 1. 地基 | BERT 预训练循环的四步 + 谁吃显存 | MLM / 优化器状态 |
| 2. 原生训练循环 | `zero_grad → forward → backward → step` 为什么是这四步 | 自动微分 / 梯度累积 |
| 3. 接入 DeepSpeed | 三处替换：`initialize` / `model.backward` / `model.step` | 引擎接管 / API 对齐 |
| 4. 落盘文件解剖 | 为什么单卡是 1 个 `.pt`，DeepSpeed 是一堆 `zero_pp_rank_*` | ZeRO 切分 / checkpoint |
| 5. 多卡子集与转换 | `--include localhost:2,3` 与 `zero_to_fp32.py` | 卡选择 / 状态合并 |
| 实操 | 原文真实命令与目录树 | 命令速查 |
| 常见问题/坑 | local_rank、卡数、断点续训 | 排错 |

---

## 0. 一句话锚点

**HelloDeepSpeed 是 DeepSpeedExamples 里的「Hello World」**：同一个 BERT 预训练任务写两份代码——`train_bert.py`（纯 HuggingFace + 原生 PyTorch）和 `train_bert_ds.py`（同样的模型，但训练循环交给 DeepSpeed 引擎）。两份对照，你能用最少的代码差异看清 DeepSpeed 改变了什么。

- 源码：https://github.com/microsoft/DeepSpeedExamples/tree/master/training/HelloDeepSpeed

```
原生 PyTorch                          DeepSpeed
┌──────────────────────┐             ┌──────────────────────────────────┐
│ model = create_model │             │ model = create_model             │
│ optimizer = Adam(...) │   ───►      │ model,_,_,_ = deepspeed.initialize│
│ loss.backward()      │             │ model.backward(loss)             │
│ optimizer.step()     │             │ model.step()                     │
└──────────────────────┘             └──────────────────────────────────┘
  你手动管梯度/优化器                    引擎接管：ZeRO 切分、混合精度、通信
```

---

## 1. 地基：BERT 预训练循环里，谁在吃显存

BERT 用 **MLM（Masked Language Model，掩码语言模型）** 自监督预训练：把输入句子里约 15% 的 token 替换为 `[MASK]`，让模型预测被遮住的原词。损失就是这些位置上的交叉熵。本例的模型由几个超参完全决定：

| 超参 | 含义 | 原文出现处 |
| --- | --- | --- |
| `num_layers` | Transformer 层数 | `create_model(num_layers=...)` |
| `num_heads` | 多头注意力的头数 | `num_heads=...` |
| `h_dim` | 隐藏维度（hidden size） | `h_dim=...` |
| `ff_dim` | FFN 中间层维度 | `ff_dim=...` |
| `dropout` | 丢弃率 | `dropout=...` |

**为什么要先看「谁吃显存」？** 因为 DeepSpeed 的核心价值就是省显存。训练一个参数量为 $\Psi$ 的模型，用 Adam + fp16 混合精度，显存大头是：

$$
M \approx \underbrace{2\Psi}_{\text{fp16 权重}} + \underbrace{2\Psi}_{\text{fp16 梯度}} + \underbrace{12\Psi}_{\text{Adam 状态(fp32 权重+momentum+variance)}} = 16\Psi \text{ bytes}
$$

**数值手算**：本例若是个 ~0.2B 参数的小 BERT，$16\Psi = 16 \times 2\times10^8 = 3.2\,\text{GB}$ 仅状态；真正训练时再叠加激活值。单卡跑得动，但一旦放大就爆显存——这正是后面 DeepSpeed ZeRO 把这 $16\Psi$ 沿 GPU 数 $N$ 切分的动机（详见 [[ai-framework/deepspeed/README]]）。

> 内存估算的完整推导见 [[docs/transformer内存估算]]。

---

## 2. 原生训练循环：四步为什么是这四步

原文 `train_bert.py` 的核心（HF + 原生 PyTorch）：

```python
model = create_model(
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        h_dim=h_dim,
        dropout=dropout,
    )
model.train()

for step, batch in enumerate(data_iterator, start=start_step):
    optimizer.zero_grad()
    # Forward pass
    loss = model(**batch)
    # Backward pass
    loss.backward()
    # Optimizer Step
    optimizer.step()
```

**逐步说为什么：**

```
 ┌─ optimizer.zero_grad() ── 清空上一步残留的 .grad（PyTorch 默认梯度累加，不清会叠加）
 │
 ├─ loss = model(**batch) ── 前向：建立计算图，记录每个张量的 grad_fn
 │
 ├─ loss.backward() ──────── 反向：autograd 沿计算图回传，把 ∂loss/∂w 填进每个参数的 .grad
 │
 └─ optimizer.step() ─────── 用 .grad 更新权重，w ← w - lr·m̂/(√v̂+ε)（Adam）
```

- `model.train()`：切到训练模式，开启 dropout、让 BatchNorm 用 batch 统计量。
- `start=start_step`：从断点续训时不从 0 开始计步。
- **关键认知**：这里梯度、优化器状态、权重全在「这一张卡」上，没有任何切分或跨卡通信。简单，但天花板低。

**运行命令（原文真料，原样保留）：**

```
python train_bert.py --checkpoint_dir ./experiments --local_rank 0
```

`--local_rank 0`：标识本进程在本机的 GPU 编号；单卡训练填 0。它是分布式启动器（如 `torch.distributed`）约定的参数。

**模型输出权重文件（原文真料）：**

```
tree experiments/
experiments/
└── bert_pretrain.2023.6.13.5.34.39.addjtvxg
    ├── checkpoint.iter_1000.pt
    ├── checkpoint.iter_2000.pt
    ├── checkpoint.iter_3000.pt
    ├── checkpoint.iter_4000.pt
    ├── checkpoint.iter_5000.pt
    ├── checkpoint.iter_6000.pt
    ├── checkpoint.iter_7000.pt
    ├── checkpoint.iter_8000.pt
    ├── checkpoint.iter_9000.pt
    ├── gitdiff.log
    ├── githash.log
    ├── hparams.json
    └── tb_dir
        └── events.out.tfevents.1686659679.ai-app-2-46.54673.0
```

**怎么读这棵树：**

- 目录名 `bert_pretrain.<时间戳>.<随机串>`：每次实验自动建独立目录，避免覆盖。
- `checkpoint.iter_N.pt`：每 1000 步存一次，**整模型一个文件**——因为没切分，单卡全量保存。
- `gitdiff.log` / `githash.log`：记录代码版本与未提交改动，保证实验**可复现**（AI-Infra 工程化好习惯）。
- `hparams.json`：本次实验超参快照。
- `tb_dir/events.out.tfevents.*`：TensorBoard 日志，看 loss 曲线。

---

## 3. 接入 DeepSpeed：只改三处

原文 `train_bert_ds.py` 的核心。注意它和第 2 节**几乎一样**，只有三处替换：

```python
model = create_model(
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        h_dim=h_dim,
        dropout=dropout,
    )
model, _, _, _ = deepspeed.initialize(model=model,
                                          model_parameters=model.parameters(),
                                          config=ds_config)
model.train()
for step, batch in enumerate(data_iterator, start=start_step):
    # Forward pass
    loss = model(**batch)
    # Backward pass
    model.backward(loss)
    # Optimizer Step
    model.step()
```

**三处替换对照表（这是本课最值钱的一张表）：**

| 原生 PyTorch | DeepSpeed | 引擎在背后多做了什么 |
| --- | --- | --- |
| `optimizer = Adam(...)` | `model,_,_,_ = deepspeed.initialize(...)` | 根据 `ds_config` 创建 ZeRO 优化器、配置混合精度、建分布式通信组 |
| `optimizer.zero_grad()` | （省略，引擎内部处理） | 引擎在 `step` 后自动清梯度 |
| `loss.backward()` | `model.backward(loss)` | 反向 + 梯度切分/all-reduce + loss scaling（fp16 防下溢） |
| `optimizer.step()` | `model.step()` | 用切分后的优化器状态更新对应分片 + 学习率调度 |

```
deepspeed.initialize() 返回 4 元组：
  ┌─────────────┬──────────────┬──────────────┬───────────────┐
  │ engine(model)│ optimizer    │ dataloader   │ lr_scheduler  │
  └─────────────┴──────────────┴──────────────┴───────────────┘
        ▲ 本例只接 model，其余用 _ 忽略
        └─ 这个 model 已是 DeepSpeedEngine，包住原模型 + ZeRO 逻辑
```

**为什么 backward/step 要换成 `model.*`？** 因为 ZeRO 把梯度和优化器状态切到不同卡上，`loss.backward()` 只会在本地算梯度、不会触发跨卡通信与切分；必须走引擎的 `model.backward()` / `model.step()`，才能在合适时机插入 all-reduce / reduce-scatter 等[[ai-infra/网络/集合通信原语]]。

`ds_config`（DeepSpeed 配置字典/JSON）决定一切：ZeRO stage、fp16/bf16、优化器、梯度累积步数等——它是 DeepSpeed 的「方向盘」，详见 [[ai-framework/deepspeed/README]]。

---

## 4. 落盘文件解剖：为什么变成一堆 `zero_pp_rank_*`

**运行命令（原文真料）：**

```
# 默认使用当前服务器所有GPU卡
deepspeed train_bert_ds.py --checkpoint_dir ./experiments_ds
```

`deepspeed` 启动器**默认拉起本机所有 GPU**，每张卡一个进程。

**输出目录树（原文真料）：**

```
tree experiments_ds/
experiments_ds/
└── bert_pretrain.2023.6.13.18.58.44.addjtvxg
    ├── gitdiff.log
    ├── githash.log
    ├── global_step1000
    │   ├── mp_rank_00_model_states.pt
    │   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
    ...
    │   └── zero_pp_rank_7_mp_rank_00_optim_states.pt
    ├── global_step9000
    │   ├── mp_rank_00_model_states.pt
    │   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
    ...
    │   └── zero_pp_rank_7_mp_rank_00_optim_states.pt
    ├── hparams.json
    ├── latest
    ├── tb_dir
    │   └── events.out.tfevents.1686707924.ai-app-2-46.599.0
    └── zero_to_fp32.py
```

**和单卡的关键差异——文件名解码：**

| 文件 | 含义 | 为什么这样 |
| --- | --- | --- |
| `global_stepN/` | 第 N 步的 checkpoint **目录**（不再是单文件） | 分布式 checkpoint 由多分片组成，用目录归拢 |
| `mp_rank_00_model_states.pt` | 模型权重，`mp_rank` = 模型并行 rank | 本例无张量并行，只有 `mp_rank_00` 一份权重 |
| `zero_pp_rank_0..7_mp_rank_00_optim_states.pt` | **8 份优化器状态**，对应 8 张卡 | ZeRO 把 Adam 状态（$12\Psi$ 那块）沿 8 卡切分，每卡只存 $1/8$ |
| `latest` | 文本文件，记录最新 `global_stepN` | 续训时引擎读它定位最新断点 |
| `zero_to_fp32.py` | **状态合并脚本** | 把切散的分片重组为单个 fp32 全量权重，供推理/导出 |

```
ZeRO 切分直观图（8 卡为例，Adam 状态占大头）：
                  整体优化器状态 12Ψ
   ┌────┬────┬────┬────┬────┬────┬────┬────┐
   │GPU0│GPU1│GPU2│GPU3│GPU4│GPU5│GPU6│GPU7│   每卡 = 12Ψ/8
   └────┴────┴────┴────┴────┴────┴────┴────┘
   zero_pp_rank_0  ...                rank_7
   ⇒ 单卡显存压力降到约 1/N，这就是 ZeRO 的核心收益
```

**为什么单卡是 1 个 `.pt`、DeepSpeed 是 1+N 个分片？** 单卡全量状态在一张卡上，自然一个文件；DeepSpeed 用 ZeRO 把状态切到 N 卡，**每卡只持有自己那片**，落盘时各存各的，于是出现 `zero_pp_rank_0 ... rank_7` 共 8 份。要拿回完整模型，就用 `zero_to_fp32.py` 合并。

---

## 5. 多卡子集 + 大小对照 + 状态转换

**只用部分卡 + 控制步数（原文真料）：**

```
deepspeed --include localhost:2,3,4,5 train_bert_ds.py --checkpoint_dir ./experiments_multigpu --num_iterations=500 --checkpoint_every=250
```

| 参数 | 作用 |
| --- | --- |
| `--include localhost:2,3,4,5` | 只用本机 2/3/4/5 号 GPU（共 4 卡），不占满全机 |
| `--num_iterations=500` | 总训练步数 |
| `--checkpoint_every=250` | 每 250 步存一次（所以会有 `global_step250`） |

**带大小的目录树（原文真料，`tree -h`）：**

```
tree -h ./experiments_multigpu
./experiments_multigpu
├── [  36]  bert_pretrain.2023.6.13.19.37.59.addjtvxg
│   └── [  63]  global_step250
│       └── [ 47M]  zero_pp_rank_3_mp_rank_00_optim_states.pt
└── [ 169]  bert_pretrain.2023.6.13.19.38.0.addjtvxg
    ├── [ 45K]  gitdiff.log
    ├── [  41]  githash.log
    ├── [ 207]  global_step250
    │   ├── [ 31M]  mp_rank_00_model_states.pt
    │   ├── [ 47M]  zero_pp_rank_0_mp_rank_00_optim_states.pt
    │   ├── [ 47M]  zero_pp_rank_1_mp_rank_00_optim_states.pt
    │   └── [ 47M]  zero_pp_rank_2_mp_rank_00_optim_states.pt
    ├── [ 298]  hparams.json
    ├── [  14]  latest
    ├── [  77]  tb_dir
    │   └── [2.4K]  events.out.tfevents.1686710280.ai-app-2-46.14672.0
    └── [ 18K]  zero_to_fp32.py
```

**从大小读出门道：**

- `mp_rank_00_model_states.pt` = **31M**：模型权重（全量，一份）。
- 每个 `zero_pp_rank_*_optim_states.pt` = **47M**：单卡优化器分片。**为什么优化器分片比权重还大？** 因为 Adam 每个参数要存 momentum + variance 两份 fp32 状态（约 $8\Psi$），再加 fp32 主权重副本，远大于 fp16 模型权重（$2\Psi$）。这正印证了第 1 节「$12\Psi$ 才是显存大头」。
- 出现**两个**实验目录（19.37.59 与 19.38.0）：DeepSpeed 多进程各自带时间戳建目录，第一个只落了一个 rank 的分片是写入时序/进程差异所致——实操中**以信息完整、含 `latest` 和 `zero_to_fp32.py` 的那个目录为准**。

**转换：分片 → 单个 fp32 全量权重**

`zero_to_fp32.py` 随 checkpoint 自动生成，用于把所有 `zero_pp_rank_*` 分片重新拼回一个完整的 fp32 `state_dict`，方便做推理部署、上传 HF Hub 或脱离 DeepSpeed 环境加载：

```
python zero_to_fp32.py  <checkpoint目录>  <输出.pt>
# 它读 global_stepN/ 下所有分片，gather 后写出单一全量权重
```

---

## 实操 · 命令速查（原文真料汇总）

```bash
# 1) 原生 HF + PyTorch，单卡
python train_bert.py --checkpoint_dir ./experiments --local_rank 0

# 2) DeepSpeed，默认用本机所有 GPU
deepspeed train_bert_ds.py --checkpoint_dir ./experiments_ds

# 3) DeepSpeed，仅用 2/3/4/5 号卡，跑 500 步、每 250 步存档
deepspeed --include localhost:2,3,4,5 train_bert_ds.py \
    --checkpoint_dir ./experiments_multigpu \
    --num_iterations=500 --checkpoint_every=250

# 4) 把 ZeRO 分片合并为单个 fp32 全量权重
python <ckpt>/zero_to_fp32.py <ckpt>/global_step250 ./pytorch_model.bin
```

---

## 常见问题 / 坑

| 现象 / 问题 | 原因 | 处理 |
| --- | --- | --- |
| 单卡跑 `train_bert.py` 报 `local_rank` 相关错 | 没传 `--local_rank 0` | 单卡显式补 `--local_rank 0` |
| `deepspeed` 把所有卡都占了 | 默认拉起本机全部 GPU | 用 `--include localhost:2,3,4,5` 指定卡子集 |
| checkpoint 是一堆 `zero_pp_rank_*`，下游加载不了 | ZeRO 切分后状态是分片 | 用 `zero_to_fp32.py` 合并成单一全量权重再加载 |
| 优化器分片(47M)比模型(31M)还大，怀疑出错 | Adam 状态 $\approx 12\Psi$ 本就 > fp16 权重 $2\Psi$ | 正常现象，非 bug |
| 出现多个时间戳实验目录，分片不全 | 多进程并发建目录 + 写入时序 | 认含 `latest`/`zero_to_fp32.py` 的完整目录 |
| 续训不知从哪步开始 | 需读 `latest` 文件 | 引擎据 `latest` 定位最新 `global_stepN`，配合 `start_step` |
| 改了模型却不确定跑的哪版代码 | 忘了看版本记录 | 看 `githash.log` / `gitdiff.log` 复现实验 |
| 加了 DeepSpeed 但显存没省 | `ds_config` 没开 ZeRO 或 stage 太低 | 在 `ds_config` 里设 `zero_optimization.stage`，详见 [[ai-framework/deepspeed/README]] |

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- DeepSpeed 全景与 ZeRO 三阶段：[[ai-framework/deepspeed/README]]
- 张量/流水并行对照：[[ai-framework/megatron-lm/README]]
- PyTorch 分布式基础：[[ai-framework/pytorch/README]]
- 训练总览：[[llm-train/README]] · 高效微调：[[llm-train/peft/PEFT-API]]
- 模型架构与 Transformer：[[llm-algo/transformer/模型架构]]
- 通信原语（all-reduce / reduce-scatter）：[[ai-infra/网络/集合通信原语]] · 互联：[[ai-infra/网络/InfiniBand]]
- 显存与硬件：[[docs/transformer内存估算]] · [[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]]
- 性能指标名词：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] · 评测：[[llm-eval/README]]
