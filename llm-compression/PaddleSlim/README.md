# PaddleSlim

> 百度飞桨(PaddlePaddle)生态的官方模型压缩工具库：把量化、剪枝、蒸馏、NAS 四件套打包成开箱即用的 API，让 Paddle 训练出来的模型"又小又快"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/paddlepaddle/README]] [[llm-compression/README]]

## 阅读地图

| 你想搞清楚的问题 | 跳到 | 一句话结论 |
| --- | --- | --- |
| PaddleSlim 到底是个啥？ | [§0 锚点](#0-一句话锚点) | Paddle 生态的"压缩四件套"工具库 |
| 它解决了什么麻烦？ | [§1 地基](#1-地基前置它解决什么问题) | 把零散的压缩算法变成一行 API |
| 整体长什么样？ | [§2 架构](#2-整体架构四大能力一条主线) | 四大能力共享一套"图改写"底座 |
| 量化怎么用？ | [§3 量化](#3-量化quantization把-fp32-变-int8int4) | PTQ/QAT 两条路，LLM 走 advanced 子模块 |
| 剪枝/稀疏怎么玩？ | [§4 剪枝](#4-剪枝pruning把不重要的通道删掉) | 敏感度分析→结构化剪通道→蒸馏恢复 |
| 蒸馏在 Paddle 里怎么搭？ | [§5 蒸馏](#5-蒸馏distillation大模型教小模型) | 中间层对齐 + 软标签，零侵入挂蒸馏 |
| NAS 是搜什么的？ | [§6 NAS](#6-nas神经网络结构搜索让机器选结构) | 在搜索空间里自动搜出又快又准的结构 |
| 跟 PyTorch 系工具比呢？ | [§7 对比](#7-与-pytorch-系工具对比) | 框架绑定 vs 通用，端侧 vs 大模型 |
| 我该不该用它？ | [§8 选型](#8-选型什么时候该用-paddleslim) | 你在 Paddle 生态里就用，否则慎入 |
| 来段真实代码 | [配置示例](#典型流程与配置示例讲含义) | LLM 量化的 Shift/Smooth/GPTQ 流程 |

## 0. 一句话锚点

**PaddleSlim = 飞桨(PaddlePaddle)官方的模型压缩库，把「量化 + 剪枝 + 蒸馏 + NAS」四类压缩技术封装成统一 API。**

它不是一个新算法，而是一个**算法集合 + 工程脚手架**：你已经有一个 Paddle 模型，想让它更小更快上线（手机、边缘盒子、服务器低延迟），就用它。记住四个轴：

- **量化(Quant)**：每个数用更少比特（FP32→INT8/INT4）。
- **剪枝(Prune)**：删掉不重要的参数/通道。
- **蒸馏(Distill)**：让小模型学大模型的"暗知识"。
- **NAS**：让机器自动搜出最优网络结构。

> ⚠️ 一句话铁律：**PaddleSlim 强绑定 Paddle**。它吃的是 Paddle 的 `Program`/动态图，吐的也是 Paddle 模型。不是 PyTorch 工具，别拿来压 `.pt`。

## 1. 地基/前置（它解决什么问题）

### 1.1 没有 PaddleSlim 时的痛

假设你训了一个 Paddle 图像分类模型，要塞进手机。你得自己：

1. 找一篇量化论文，手动在每个 conv 后插入"伪量化"节点；
2. 自己写校准(calibration)逻辑，统计每层激活的 min/max；
3. 自己改计算图，把 FP32 算子换成 INT8 算子；
4. 剪枝时还要自己做敏感度分析、自己重训恢复精度。

这些都是**和模型业务无关的重复体力活**，而且极易踩坑（量化误差、图改写出 bug）。

### 1.2 PaddleSlim 把这些"标准化"了

```
        没有 PaddleSlim                  有了 PaddleSlim
   ┌────────────────────────┐      ┌────────────────────────┐
   │ 读论文 → 手插伪量化节点 │      │  quant = QAT(config)    │
   │ 手写校准统计           │ ───► │  q_model = quant.quantize(model) │
   │ 手改计算图换算子        │      │  一行 API,底层全自动     │
   │ 手做敏感度+重训         │      │                          │
   └────────────────────────┘      └────────────────────────┘
        几百行 + 易错                    几行 + 久经验证
```

它解决的核心问题：**把"压缩"从研究代码变成生产 API**，并且和 Paddle 的训练/导出/推理(Paddle Inference、Paddle Lite)无缝衔接——压完直接能部署。

### 1.3 三个前置概念（看不懂后面会糊）

- **计算图改写(Graph Transform)**：压缩的本质是改模型的计算图（插节点/删通道/换算子）。Paddle 用 `IrGraph`/`Program` 表示图，PaddleSlim 就在这层上动刀（文件顶部那两行 `from paddle.base.framework import IrGraph` 就是干这个的）。
- **静态图 vs 动态图**：早期 Paddle 是静态图(`fluid`，现已并入 `paddle.base`)，新版主推动态图。PaddleSlim 同时支持两套，量化既有静态图 PTQ 也有动态图 QAT。
- **伪量化(Fake Quant)**：训练时不真的用 INT8 算，而是"模拟"量化误差（先量化再反量化），让模型在训练中适应量化噪声。这是 QAT 的核心。

## 2. 整体架构：四大能力，一条主线

```
                         PaddleSlim
   ┌──────────────────────────────────────────────────────────┐
   │  ① 量化 Quant      ② 剪枝 Prune    ③ 蒸馏 Distill   ④ NAS │
   │  - PTQ 离线量化    - 通道剪枝       - 软标签         - 结构 │
   │  - QAT 量训        - 敏感度分析     - 中间层对齐      搜索  │
   │  - LLM advanced    - L1/FPGM/ASP   - 自蒸馏                │
   │  (Shift/Smooth/GPTQ)                                       │
   └───────────────────────────┬──────────────────────────────┘
                               │ 共享底座
                ┌──────────────▼──────────────┐
                │   计算图改写层 (Graph IR)     │
                │   IrGraph / Program / 动态图   │
                └──────────────┬──────────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │  Paddle 框架 (训练) + 部署后端        │
            │  Paddle Inference(服务器) / Lite(端) │
            └──────────────────────────────────────┘
```

**主线**：所有压缩能力最终都落到"改计算图"这一层，再交给 Paddle 的部署后端（Paddle Inference 跑服务器、Paddle Lite 跑手机/嵌入式、Paddle.js 跑浏览器）。这是它相比"压完还要自己想办法部署"的工具的最大优势：**端到端闭环**。

### 核心模块速查

| 模块 | 命名空间(以官方为准) | 干什么 |
| --- | --- | --- |
| 量化 | `paddleslim.quant` | PTQ/QAT 入口 |
| LLM 量化 | `paddleslim.quant.advanced` | Shift/Smooth/GPTQ/PieceWiseSearch 等大模型专用 |
| 剪枝 | `paddleslim.prune` | 通道剪枝、敏感度分析 |
| 蒸馏 | `paddleslim.dist` | 蒸馏 loss 挂载 |
| NAS | `paddleslim.nas` | 搜索空间 + 搜索策略 |
| 自动压缩 | ACT(Auto Compression Toolkit) | 无需训练代码，喂模型自动压 |

## 3. 量化(Quantization)：把 FP32 变 INT8/INT4

量化是 PaddleSlim 用得最多的能力。原理一句话：用缩放因子 $s$ 把浮点映射到低比特整数。

$$
q = \text{round}\!\left(\frac{x}{s}\right),\qquad s = \frac{\max(|x|)}{2^{b-1}-1}
$$

其中 $b$ 是比特数。INT8 时 $2^{b-1}-1 = 127$。

### 3.1 两条路线：PTQ vs QAT

```
   PTQ (训练后量化)              QAT (量化感知训练)
   ┌─────────────┐              ┌─────────────────┐
   │ 训好的模型   │              │ 训好的模型       │
   │     │        │              │     │ 插伪量化节点 │
   │  喂少量校准数据│             │     ▼            │
   │  统计min/max  │             │  继续训练几个epoch│
   │     │        │              │  让权重适应量化   │
   │     ▼        │              │     │            │
   │ INT8 模型    │              │     ▼            │
   │ (快,可能掉点) │             │ INT8 模型(慢但准) │
   └─────────────┘              └─────────────────┘
```

| 维度 | PTQ(离线量化) | QAT(量化感知训练) |
| --- | --- | --- |
| 是否要训练 | 否，只需几百条校准数据 | 是，要回炉重训 |
| 速度 | 分钟级 | 小时级 |
| 精度 | 一般，敏感模型会掉点 | 高，逼近 FP32 |
| 适用 | 快速上线、对精度不极致 | 精度敏感、可承受重训 |

**经验**：先无脑 PTQ，掉点能接受就完事；掉点太多再上 QAT。

### 3.2 LLM 专用：`paddleslim.quant.advanced`

大模型量化有个老大难：**激活里有"离群值(outlier)"**，几个维度数值特别大，按 max 量化会把其余维度压成 0。PaddleSlim 在 `advanced` 子模块里给了一整套对策：

```
  激活异常值问题            advanced 子模块的解法
  ┌──────────────┐
  │ 某些通道值爆炸 │ ──► Shift  : 把激活整体平移,去掉偏置
  │ 量化精度崩塌   │ ──► Smooth : 把激活的难度"搬"到权重上
  └──────────────┘ ──► PieceWiseSearch: 分段搜最优缩放
                   ──► GPTQ   : 逐层最小化量化误差(二阶信息)
                   ──► LayerWiseQuantError: 量化误差诊断/排查
```

- **Shift / Smooth**：SmoothQuant 思路在 Paddle 的实现。Smooth 通过一个逐通道缩放 $s$，把"难量化的激活"用 $X/s$ 变好量化，代价转嫁给权重 $W\cdot s$（权重好量化）。等价变换、不改数学结果。
- **PieceWiseSearch**：把数值范围**分段**，每段搜一个最优 alpha/scale，比全局一个缩放更精细。
- **GPTQ**：经典的逐层量化，用 Hessian(二阶信息)逐列量化并补偿误差，`act_order=True` 按激活重要性排序量化，精度更好。
- **LayerWiseQuantError**：不真量化，只**测量**每层若被量化会引入多大误差，用来定位"哪些层不能量化"。

## 4. 剪枝(Pruning)：把不重要的通道删掉

剪枝是删参数。但**非结构化剪枝**（随机置零）在通用硬件上不加速（还是稠密计算），所以 PaddleSlim 主打**结构化剪枝**——直接砍掉整个卷积通道，模型真的变小变快。

```
   原始 conv: 64 输出通道
   ┌─┬─┬─┬─┬─┬─┬─┬─┐  敏感度分析: 哪些通道砍了精度掉得少?
   │■│■│░│■│░│■│░│■│  ░ = L1范数小/FPGM判定不重要
   └─┴─┴─┴─┴─┴─┴─┴─┘
            │ 剪掉 ░
            ▼
   剪后 conv: 40 输出通道  → 计算量↓、参数↓、真加速
            │
            ▼ 重训/蒸馏恢复精度
   精度回到接近原模型
```

三步法：

1. **敏感度分析(Sensitivity)**：逐层试剪不同比例，看精度掉多少，画出"敏感度曲线"。敏感的层少剪，鲁棒的层多剪。
2. **剪枝准则**：常见 `L1Norm`（按权重绝对值和）、`FPGM`（按几何中位数，剪冗余而非剪小值）、`ASP`（2:4 结构化稀疏，配合支持的硬件加速）。
3. **恢复**：剪完精度掉，重训(fine-tune)或挂蒸馏把精度找回来。

数值感受：一个 ResNet 剪掉 30% 通道，FLOPs 大约也降 30%（卷积量级近似 $\propto C_{in}\times C_{out}$），精度掉 1% 以内是常见水平。

## 5. 蒸馏(Distillation)：大模型教小模型

蒸馏让小模型(student)模仿大模型(teacher)的输出分布(软标签)和中间特征。PaddleSlim 的 `dist` 把这做成"零侵入挂载"——不用改 student 网络定义，指定要对齐的层名即可。

```
   Teacher(大,固定)         Student(小,在训)
   ┌───────────┐           ┌───────────┐
   │  conv_5    │ ──对齐──► │  conv_5'   │  中间层 loss (FSP/L2)
   │  conv_8    │ ──对齐──► │  conv_8'   │
   │  logits    │ ──对齐──► │  logits'   │  软标签 KL loss
   └───────────┘           └───────────┘
        |                        |
        └──── 总loss = 任务loss + λ·蒸馏loss ────┘
```

支持的对齐方式：软标签蒸馏(KL 散度)、中间特征对齐(L2 / FSP 矩阵)、DML(互学习)等。**蒸馏常和剪枝/量化组合**：剪完掉的点，用 teacher 蒸回来。

## 6. NAS（神经网络结构搜索）：让机器选结构

前三招是"压缩一个给定结构"，NAS 是**直接搜一个更优的结构**。

```
   定义搜索空间(可选的层数/通道数/算子)
            │
            ▼
   搜索策略采样一个子结构 ──► 训练/评估 ──► 反馈奖励
            ▲                                    │
            └──────────── 迭代优化 ◄─────────────┘
            │
            ▼
   输出帕累托最优结构(精度↑ 同时 延迟↓)
```

PaddleSlim 提供 SANAS(模拟退火)、OFA(Once-For-All，一次训练得到一族子网)等。**OFA 思路**：训一个超网，里面包含无数子网，部署时按硬件约束直接抽一个最优子网，不用重训。代价是 NAS 搜索本身算力开销大，工业界用得比量化/剪枝少。

## 7. 与 PyTorch 系工具对比

```
              框架绑定                    适用规模
   PaddleSlim ──── Paddle 专用 ───── CV/小模型强, LLM 量化在补
   PyTorch量化 ─── torch 专用 ────── 通用,生态最大
   GPTQ/AWQ ────── 框架无关(HF) ──── 专攻 LLM 权重量化
   llm-compressor─ HF/vLLM 生态 ──── 大模型量化+稀疏,产线化
```

| 工具 | 框架 | 覆盖能力 | 强项 | 短板 |
| --- | --- | --- | --- | --- |
| **PaddleSlim** | PaddlePaddle | 量化+剪枝+蒸馏+NAS(全) | 端到端闭环、端侧部署(Lite)、CV 成熟 | 强绑 Paddle，PyTorch 用户用不了 |
| torch 原生量化(`torch.ao`) | PyTorch | 量化为主 | 生态大、文档全 | 剪枝/NAS 需另找库 |
| GPTQModel / AWQ | 框架无关(偏 HF) | LLM 权重量化 | LLM INT4 效果好、社区热 | 只管量化，不管剪枝/蒸馏 |
| llm-compressor | HF + vLLM | 量化+稀疏 | 大模型产线、和 vLLM 直连 | 偏 LLM，不做 CV/NAS |

**一句话区分**：PaddleSlim 是"**一个框架内的全能压缩套件**"，PyTorch 系是"**一堆专精单点的库拼起来**"。你在哪个生态，就用哪边——跨框架几乎不可能直接复用。

## 8. 选型：什么时候该用 PaddleSlim

```
   你的模型是 Paddle 训的吗?
        │是                          │否
        ▼                            ▼
   要不要端侧/边缘部署?          用 PyTorch 系
        │要                  (torch量化 / GPTQ / llm-compressor)
        ▼                    别硬转 Paddle,转换成本高于收益
   PaddleSlim + Paddle Lite ✅
        │否(服务器)
        ▼
   PaddleSlim + Paddle Inference ✅
   LLM? → paddleslim.quant.advanced(Smooth/GPTQ)
```

- **用它**：你已经在 Paddle 生态（PaddleOCR、PaddleDetection、PaddleNLP 这些都基于 Paddle），要压缩部署，尤其是端侧/嵌入式。
- **慎用**：你的模型是 PyTorch/HF 的。把 PyTorch 模型转 Paddle 再用 PaddleSlim，转换损耗 + 调试成本通常不划算，不如直接用 PyTorch 系工具。

## 典型流程与配置示例（讲含义）

### 例 1：LLM 量化前的 Smooth（去激活离群值）

```python
from paddleslim.quant.advanced import Smooth, MultiStepSampler

model = LLM()
model_config = {}
smooth = Smooth(model, model_config, sample_function=MultiStepSampler())
for data in dataloader():       # 喂校准数据,采样激活分布
    model(data)
    smooth.step += 1
smooth.update_weight()          # 用采样到的统计量,把缩放搬到权重
```

**含义**：`MultiStepSampler` 多步采样激活的统计量；`update_weight()` 真正执行"激活难度→权重"的等价迁移。做完这步，权重和激活都变得"好量化"，后面再走 INT8 量化就不容易掉点。

### 例 2：分段搜索更优缩放（PieceWiseSearch）

```python
from paddleslim.quant.advanced import Smooth, MultiStepSampler, PieceWiseSearch, mse_loss

search_func = PieceWiseSearch(
    k_piece=3,                 # 把数值范围分 3 段
    bits_length=8,             # 目标 INT8
    search_alpha_min=0.2, search_alpha_max=0.8,  # smooth 强度搜索区间
    search_scale_min=1., search_scale_max=5.,    # 缩放搜索区间
    weight_quant_method='abs_max_channel_wise',  # 权重逐通道量化
    act_quant_method='abs_max',                  # 激活按绝对最大值量化
    loss_function=mse_loss,    # 用 MSE 衡量量化误差,选误差最小的参数
)
smooth = Smooth(model, model_config, sample_function=MultiStepSampler(), search_function=search_func)
```

**关键参数权衡**：`k_piece` 越大越精细但越慢；`weight_quant_method` 选 `channel_wise`(逐通道)比 `abs_max`(逐张量)精度高但 scale 更多；搜索区间越宽越可能找到更优解，代价是搜索时间线性增长。

### 例 3：逐层 GPTQ + 量化误差诊断

```python
from paddleslim.quant.advanced import GPTQ, LayerWiseQuantError
import paddle

model = LLM()
for cur_name, cur_layer in model.named_sublayers():
    if type(cur_layer) == paddle.nn.Linear:
        gptq_layer = GPTQ(cur_layer)
        for data in dataloader():        # 累积该层的 Hessian 统计
            model(data)
        gptq_layer.fasterquant(act_order=True)  # 按激活重要性排序逐列量化
```

**含义**：GPTQ 对每个 `Linear` 层独立做量化，`act_order=True` 让"重要的列先量化、误差先补偿"，精度更稳。`LayerWiseQuantError` 则是配套的"体检"工具——不真量化，只打印每层的量化误差 `cur_layer.losses.mean()`，帮你定位哪些层是精度杀手、该不该跳过量化。

> 以上 API 以 PaddleSlim 官方文档/源码为准；`paddle.fluid` 已并入 `paddle.base`（见 [Paddle issue #55108](https://github.com/PaddlePaddle/Paddle/issues/55108)），老代码里的 `fluid` 命名空间在新版可能需替换为 `paddle.base`。

## 常见问题

| 问题 | 答案 |
| --- | --- |
| PaddleSlim 能压 PyTorch 模型吗？ | 不能，强绑 Paddle。要么转 Paddle，要么用 PyTorch 系工具 |
| PTQ 和 QAT 先用哪个？ | 先 PTQ；掉点不可接受再上 QAT |
| LLM 量化掉点严重怎么办？ | 用 `advanced` 里的 Smooth/Shift 先压离群值，再 GPTQ |
| 剪枝后为什么没变快？ | 八成是非结构化剪枝；要用结构化(剪通道)才在通用硬件加速 |
| `act_order=True` 有什么用？ | GPTQ 里按激活重要性排序量化，精度更好，几乎无副作用 |
| 压完怎么部署？ | 服务器 Paddle Inference，端侧/嵌入式 Paddle Lite，无缝衔接 |
| 不想改训练代码也能压吗？ | 用 ACT(自动压缩)，喂模型 + 少量数据自动搜压缩策略 |
| `fluid` 报错找不到？ | 已并入 `paddle.base`，把 `paddle.fluid.*` 换成 `paddle.base.*` |
| NAS 值得用吗？ | 算力够、追极致才用；多数场景量化+剪枝性价比更高 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局知识地图，回总览
- [[ai-framework/paddlepaddle/README]] — Paddle 框架本体，PaddleSlim 的地基
- [[llm-compression/README]] — 模型压缩总览，量化/剪枝/蒸馏/低秩的通用原理
