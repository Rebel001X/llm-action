# PyTorch 训练：学习资源与工程实践

> 一份"从入门到工程化"的 PyTorch 训练学习地图：把零散资源沉淀成一条可复用的流水线，让训练代码可复审、可复用、可交接、可部署。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] [[llm-train/pytorch/distribution/README]] [[ai-framework/pytorch/README]] [[ai-framework/deepspeed/README]]

## 阅读地图

| 节 | 你会得到什么 | 适合谁 |
|----|--------------|--------|
| 0. 一句话锚点 | 这篇到底讲什么 | 所有人 |
| 1. 地基：为什么需要"工程化" | 框架不统一带来的真实代价 | 团队负责人/新人 |
| 2. PyTorch 学习的四层心智模型 | Tensor→Autograd→Module→分布式 | 入门者 |
| 3. 训练流水线 6 大环节 | 数据→增强→建模→损失→训练→部署 | 落地工程师 |
| 4. 工程化要素拆解 | 配置/日志/checkpoint/可复现 | 进阶 |
| 5. 从单卡到分布式的演进路径 | DP→DDP→FSDP→框架 | Infra 方向 |
| 6. 学习资源分级清单 | 官方→源码→实战 | 自学者 |
| 配置/目录示例 | 一个可复用的工程骨架 | 落地工程师 |
| 常见问题/坑 | 踩过的坑提前避开 | 所有人 |

## 0. 一句话锚点

**PyTorch 训练 = `Tensor`（数据载体）+ `Autograd`（自动求导）+ `Module`（模型容器）+ `Optimizer`（参数更新）+ `分布式后端`（多卡扩展）**，五者用一条标准化流水线串起来。学习的目标不是"会写一个 demo"，而是写出**能被别人复审、复用、接手、部署**的训练代码。

> 本文是"资源 + 实践索引"型文档，讲**稳定的思想与方法论**；具体 API 名称、默认值、版本特性请以 [官方文档 pytorch.org/docs](https://pytorch.org/docs/) 与源码为准。

## 1. 地基：为什么"工程化"比"会调 API"更重要

很多人学 PyTorch 停在"能跑通一个脚本"，但团队协作时痛点全部暴露——**框架/写法不统一**会带来连锁代价：

| 维度 | 不统一时的真实代价 |
|------|--------------------|
| 代码复审 | 每个人写法不同，reviewer 无从下手，评审效率趋近于零 |
| 代码复用 | 数据/模型/训练逻辑耦合，根本无法复用 |
| 项目交接 | 没有约定的目录与配置，新人接手如考古 |
| 模型部署 | 训练态与推理态脱节，导出（`torch.export`/ONNX/TorchScript）一言难尽 |
| 经验复用 | 调参经验散落在个人脑子里，"想都别想"沉淀下来 |
| 性能调试 | 没有统一 profiler/日志，瓶颈定位全靠猜 |

**结论**：学习 PyTorch 训练，要同时学两件事——**(a) 框架机制**（懂底层才能调对）和 **(b) 工程约定**（让训练可工业化）。本文把这两条线都铺开。

## 2. PyTorch 学习的四层心智模型

理解 PyTorch，最好按"由底向上"的四层来建立心智模型，每一层都解决上一层留下的问题：

```
            ┌───────────────────────────────────────────┐
   第4层    │  分布式 / 大规模  DDP · FSDP · 框架(DeepSpeed/Megatron) │
   (扩展)   │  解决: 单卡放不下 / 训得太慢                │
            └───────────────────────────────────────────┘
                              ▲ 把单卡训练横向放大
            ┌───────────────────────────────────────────┐
   第3层    │  Module + Optimizer + DataLoader  (训练循环) │
   (组装)   │  解决: 把"求导"组织成可训练的模型与流程     │
            └───────────────────────────────────────────┘
                              ▲ 用计算图驱动参数更新
            ┌───────────────────────────────────────────┐
   第2层    │  Autograd  自动微分 / 动态计算图            │
   (求导)   │  解决: 不用手写反向传播,自动算梯度          │
            └───────────────────────────────────────────┘
                              ▲ 记录每一步算子,反向回放
            ┌───────────────────────────────────────────┐
   第1层    │  Tensor  多维数组 + device(cpu/cuda) + dtype │
   (载体)   │  解决: 数据在 CPU/GPU 上高效存取与运算      │
            └───────────────────────────────────────────┘
```

- **第 1 层 Tensor**：一切的载体。要点是 `device`（数据在 CPU 还是 GPU）、`dtype`（fp32/fp16/bf16，影响显存与精度）、`shape/stride`（内存布局，决定 `view`/`reshape` 是否拷贝）。
- **第 2 层 Autograd**：PyTorch 的灵魂。前向时把算子记录进**动态计算图**，`loss.backward()` 时按链式法则反向回放求梯度。理解 `requires_grad`、`grad`、`no_grad()`、计算图何时被释放，是 debug 显存与梯度问题的根基。
- **第 3 层 Module + 训练循环**：`nn.Module` 是参数与子模块的容器；`Optimizer` 持有参数引用并按梯度更新；`DataLoader` 负责批量喂数据。三者组成标准训练循环（见第 3 节）。
- **第 4 层 分布式**：当单卡放不下或太慢，把第 3 层的训练横向放大——数据并行（DDP）、参数分片（FSDP）、再到 [[ai-framework/deepspeed/README]] / [[ai-framework/megatron-lm/README]] 这类训练框架。

> 学习顺序建议严格按 1→2→3→4：跳过 Autograd 直接学分布式，遇到 OOM / 梯度异常会完全失去判断力。

## 3. 训练流水线的 6 大环节

任何一个训练项目，都可以拆成下面这条**标准流水线**。把它当作"工程化模板"，每个新项目都按同一套骨架填空，复审与复用问题迎刃而解。

```
 ┌────────┐   ┌────────┐   ┌──────────┐   ┌────────┐   ┌─────────┐   ┌──────────┐
 │①数据集 │ → │②数据增强│ → │③模型搭建 │ → │④损失   │ → │⑤超参&训练│ → │⑥移植&部署│
 │Dataset │   │Augment │   │ 与裁剪   │   │函数    │   │ 训练循环 │   │ 导出/上线 │
 └────────┘   └────────┘   └──────────┘   └────────┘   └─────────┘   └──────────┘
      │            │             │              │            │              │
   读取/清洗    随机变换       nn.Module     设计目标     forward/      TorchScript
   划分train    标准化         结构+剪枝     度量差距     backward/     ONNX/export
   /val/test    /tokenize      /冻结层       /正则       step/eval     量化/服务化
```

**① 数据集（Dataset）**
- 职责：把原始数据 → 可索引的样本。自定义类实现"取第 i 条"与"总长度"两个语义。
- 要点：训练/验证/测试划分要固定（设随机种子）；大数据用流式/分片避免一次性读入。

**② 数据增强（Augmentation）**
- 职责：在喂入模型前做随机变换，提升泛化、缓解过拟合。
- CV：翻转/裁剪/色彩抖动/归一化；NLP：分词（tokenize）、截断/填充、随机 mask。
- 坑：增强只应作用于**训练集**，验证/测试集只做确定性的标准化。

**③ 模型搭建与裁剪（Model）**
- 职责：用 `nn.Module` 组织网络；必要时裁剪（剪枝、冻结部分层、只训 LoRA 适配器，参见 [[llm-train/README]] 下的 peft/lora）。
- 要点：把"结构超参"（层数/隐藏维度）抽到配置里，不要写死。

**④ 损失函数（Loss）**
- 职责：度量预测与目标的差距，给出可反传的标量。
- 要点：分类用交叉熵、回归用 MSE、对齐用专门目标（见 [[llm-alignment/RLHF]]）；注意数值稳定（如 logits + `log_softmax` 合一，避免溢出）。

**⑤ 超参与训练（Train Loop）**
- 标准五步循环（伪代码思想）：
  ```
  for epoch:
    for batch in dataloader:
        optimizer.zero_grad()      # 1. 清空上一步梯度
        out  = model(batch.x)      # 2. 前向,构建计算图
        loss = loss_fn(out, batch.y)  # 3. 算损失
        loss.backward()            # 4. 反向,自动求梯度
        optimizer.step()           # 5. 按梯度更新参数
    evaluate(model, val_loader)    # 周期性验证 + 存 checkpoint
  ```
- 关键超参的"作用与权衡"：

| 超参 | 作用 | 调大 / 调小的权衡 |
|------|------|-------------------|
| learning rate | 每步更新幅度 | 太大发散、太小收敛慢；常配 warmup + 衰减 |
| batch size | 一次喂多少样本 | 大→稳定/吃显存；小→噪声大/可能更泛化 |
| epoch / steps | 训练总量 | 太多过拟合，太少欠拟合，看验证曲线早停 |
| weight decay | L2 正则强度 | 抑制过拟合，过大欠拟合 |
| 梯度裁剪 | 限制梯度范数 | 防梯度爆炸，尤其 RNN/大模型 |
| 混合精度 (amp) | fp16/bf16 计算 | 省显存提速；需 loss scaling 防下溢 |
| 梯度累积 | 多个小 batch 累加再更新 | 用时间换显存，等效放大 batch |

**⑥ 移植与部署（Deploy）**
- 职责：把训练得到的权重导出为推理可用的格式并上线。
- 路径：`state_dict` → TorchScript / `torch.export` / ONNX → 量化（[[llm-compression/quantization/量化基础]]）→ 推理引擎（[[llm-inference/README]]：vLLM / TensorRT-LLM 等）。
- 坑：训练态算子（dropout、BN 的训练统计）要切到 `eval()`；自定义算子可能不被导出格式支持。

## 4. 工程化要素拆解（让训练"可工业化"）

光有流水线还不够，要让它**可复审、可复现、可交接**，需要四个横切要素：

```
   训练脚本 (train.py)
        │
        ├── 配置(Config) ──── 所有超参/路径集中,命令行可覆盖(argparse/hydra/yaml)
        ├── 日志(Logging) ─── 结构化日志 + 指标可视化(TensorBoard/W&B)
        ├── Checkpoint ────── 定期存权重+optimizer+epoch,支持断点续训
        └── 可复现(Repro) ─── 固定随机种子 + 记录环境/git commit/依赖版本
```

- **配置外置**：把超参、数据路径、模型结构参数全部抽到 YAML/命令行，禁止硬编码。这是"代码复用"和"实验可追溯"的前提。
- **结构化日志 + 指标可视化**：训练 loss、验证指标、学习率曲线必须可视化，否则无法判断收敛与过拟合。
- **Checkpoint 与断点续训**：长训练必崩，要能从最近 checkpoint 恢复 `model + optimizer + scheduler + step` 完整状态。
- **可复现**：固定 `random/numpy/torch/cuda` 种子，并记录 git commit、依赖版本、硬件，让结果可被他人复现（注意：完全确定性可能牺牲性能）。

> 这四点正是第 1 节"框架不统一"痛点的解药——团队约定一套骨架，所有人填同样的格子。

## 5. 从单卡到分布式的演进路径

当模型/数据规模上来，按下面阶梯逐级演进（细节见 [[llm-train/pytorch/distribution/README]]）：

```
单卡          →   单机多卡          →   单机多卡(更省显存)   →   多机多卡 / 大模型框架
torch(cuda)      DataParallel(DP)     DistributedDataParallel   FSDP / DeepSpeed / Megatron
                 [已不推荐]            (DDP, 每卡一进程)         (参数/梯度/优化器分片)
                 单进程多线程,         多进程,NCCL 通信,         解决"单卡放不下整模型"
                 主卡瓶颈/不均衡       近线性扩展               + 张量/流水线并行
```

- **DP（DataParallel）**：单进程多线程，主卡聚合，存在负载不均与 GIL 瓶颈，现已不推荐。
- **DDP（DistributedDataParallel）**：每卡一个进程，反向时用 [[ai-infra/网络/NCCL]] 做 all-reduce 同步梯度，扩展性接近线性，是数据并行的事实标准；启动用 `torchrun`（见 [[llm-train/pytorch/distribution/README]] 下的 torchrun 文档）。
- **FSDP（Fully Sharded Data Parallel）**：把参数/梯度/优化器状态**分片**到各卡，按需 all-gather，显著降低单卡显存，适合大模型。
- **训练框架**：再往上是 [[ai-framework/deepspeed/README]]（ZeRO 分片）与 [[ai-framework/megatron-lm/README]]（张量并行 + 流水线并行），用于千亿级模型。

> 通信原语（all-reduce / all-gather / broadcast）是分布式训练的底层语言，理解它们才能读懂上面每一层，参见 [[ai-infra/网络/集合通信原语]]。

## 学习资源分级清单（怎么由浅入深地学）

| 层级 | 资源类别 | 学什么 / 注意 |
|------|----------|---------------|
| L0 入门 | 官方 Tutorials + 60 分钟闪电战 | Tensor / Autograd / 第一个训练循环 |
| L1 API 手册 | 官方 docs（pytorch.org/docs） | 查准确签名/默认值——**以此为准，勿凭记忆** |
| L2 机制 | Autograd 机制、`nn.Module` 内部、DataLoader 原理 | 懂底层才能 debug 显存/梯度/性能 |
| L3 工程 | 本仓库 [[llm-train/pytorch/distribution/README]]、torchrun 文档 | 配置/日志/checkpoint/分布式启动 |
| L4 源码 | 同目录 `Pytorch源码解读.md` | C++/ATen/Autograd 引擎实现细节 |
| L5 实战 | 本仓库 peft/lora/qlora、megatron 等目录 | 真实大模型训练落地 |

> 同目录可继续看：`README.md`（PyTorch 总览）、`api.md`（常用 API 速查）、`torchrun.md`（分布式启动器）、`Pytorch源码解读.md`（源码导读）。
> 工程最佳实践参考：PyTorch 工程实践讨论（如知乎 zhuanlan.zhihu.com/p/371978706 等社区文章），注意甄别版本与时效。

## 一个可复用的工程目录骨架（示例，讲结构不背命名）

```
project/
├── configs/            # YAML 配置:数据/模型/训练超参全部外置
│   └── base.yaml
├── data/
│   ├── dataset.py      # ① Dataset:取样本+长度
│   └── transforms.py   # ② 数据增强(仅训练集随机)
├── models/
│   └── model.py        # ③ nn.Module 结构定义
├── losses/
│   └── loss.py         # ④ 损失函数
├── engine/
│   ├── trainer.py      # ⑤ 训练/验证循环 + AMP + 梯度累积
│   └── checkpoint.py   # Checkpoint 存取 / 断点续训
├── utils/
│   ├── logging.py      # 结构化日志 + TensorBoard
│   └── seed.py         # 固定随机种子,可复现
├── export/
│   └── export.py       # ⑥ 导出 TorchScript/ONNX,切 eval()
├── train.py            # 入口:解析 config → 组装 → 训练
└── requirements.txt    # 锁版本,保证可复现
```

这套骨架的价值：**新项目复制即用，每个人都在同样的格子里填代码**——这正是把第 1 节六大痛点（复审/复用/交接/部署/经验/调试）一次性解决的关键。

## 常见问题 / 坑

| 现象 / 坑 | 根因 | 应对 |
|-----------|------|------|
| CUDA out of memory | batch 太大 / 计算图未释放 / 缓存碎片 | 减 batch、用梯度累积、`no_grad` 做评估、清理引用 |
| 验证集忘了 `model.eval()` | dropout/BN 仍在训练态 | 评估前 `eval()`，评估后切回 `train()` |
| loss 不下降 / NaN | 学习率过大、未做梯度裁剪、数值溢出 | 调小 lr + warmup、梯度裁剪、fp16 配 loss scaling |
| 多卡指标比单卡差 | DDP 下未正确同步/未用分布式采样器 | 用分布式采样器，确保各卡数据不重叠 |
| 结果不可复现 | 未固定种子、cuDNN 非确定算子 | 固定全部种子，必要时开确定性（牺牲速度） |
| 增强用错数据集 | 把随机增强用到 val/test | 增强只作用训练集，评估只做确定性标准化 |
| checkpoint 续训对不上 | 只存了权重没存 optimizer/step | 完整保存 model+optimizer+scheduler+step |
| 导出推理失败 | 自定义/动态算子不被 ONNX/Script 支持 | 改用支持的算子或 `torch.export`，先 `eval()` |
| 显存随训练增长 | 把含计算图的张量累加进 list 做日志 | 记录标量时用 `.item()` / `.detach()` |
| 版本/默认值记错 | 凭记忆写 API 默认值 | **一律查官方 docs/源码，不靠记忆** |

## 🔗 跳转链接

- 知识总图：[[00-知识地图]]
- 训练总览：[[llm-train/README]]
- 分布式训练：[[llm-train/pytorch/distribution/README]]
- PyTorch 框架：[[ai-framework/pytorch/README]]
- DeepSpeed 框架：[[ai-framework/deepspeed/README]]
- Megatron-LM：[[ai-framework/megatron-lm/README]]
- 集合通信原语：[[ai-infra/网络/集合通信原语]]
- NCCL：[[ai-infra/网络/NCCL]]
- 量化基础：[[llm-compression/quantization/量化基础]]
- RLHF 对齐：[[llm-alignment/RLHF]]
- 推理引擎总览：[[llm-inference/README]]
