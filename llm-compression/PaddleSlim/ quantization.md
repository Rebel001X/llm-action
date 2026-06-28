# PaddleSlim 量化（quantization）

> PaddleSlim 在飞桨(PaddlePaddle)生态里把大模型量化拆成一组可组合的「等价变换 + 误差最小化」算子：Shift / Smooth / PieceWiseSearch / GPTQ / LayerWiseQuantError，先把激活"捋顺"再压低比特。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-compression/PaddleSlim/README]] [[llm-compression/quantization/量化基础]] [[llm-compression/quantization/SmoothQuant]] [[llm-compression/quantization/GPTQ]]

## 阅读地图

| 你想搞清楚的问题 | 跳到 | 一句话结论 |
| --- | --- | --- |
| 量化到底在算什么？ | [§0 锚点](#0-一句话锚点) | 用缩放因子把浮点映射到低比特整数 |
| LLM 量化为什么特别难？ | [§1 地基](#1-地基前置llm-量化难在哪) | 激活有离群值(outlier)，一刀切量化会崩 |
| 这套 API 怎么组织的？ | [§2 架构](#2-整体架构采样器--变换器--搜索器--量化器) | 采样器+变换器+搜索器+量化器四件套 |
| Shift 干什么？ | [§3 Shift](#3-shift把激活平移到零附近) | 平移消偏置，把分布拉回零附近 |
| Smooth 干什么？ | [§4 Smooth](#4-smooth把激活的难度搬到权重上) | 逐通道缩放，激活的难度转嫁给权重 |
| 缩放系数怎么定最优？ | [§5 PieceWiseSearch](#5-piecewisesearch分段搜索最优缩放) | 分段+网格搜，按误差挑最优 alpha/scale |
| GPTQ 的精度从哪来？ | [§6 GPTQ](#6-gptq用二阶信息逐列量化并补偿) | 用 Hessian 逐列量化并补偿误差 |
| 哪些层不能量化？ | [§7 诊断](#7-layerwisequanterror量化误差体检) | LayerWiseQuantError 只测不量、定位精度杀手 |
| 完整流程怎么串？ | [配置示例](#典型流程与配置示例讲含义) | Shift→Smooth/Search→GPTQ→导出 |
| 容易踩什么坑？ | [常见问题](#常见问题坑) | 采样不足、step 没自增、误用逐张量量化 |

## 0. 一句话锚点

**PaddleSlim 量化 = 在飞桨里把"高精度浮点权重/激活"压成 INT8/INT4，核心是 `paddleslim.quant.advanced` 这组面向大模型的算子。**

量化的数学本质只有一行：用一个缩放因子 $s$（scale）把浮点 $x$ 映射到定点整数 $q$，推理时再用 $s$ 还原：

$$
q = \text{round}\!\left(\frac{x}{s}\right),\qquad \hat{x} = q\cdot s,\qquad s = \frac{\max(|x|)}{2^{b-1}-1}
$$

其中 $b$ 是比特数，INT8 时 $2^{b-1}-1 = 127$，INT4 时是 7。量化误差就是 $|x-\hat{x}|$。**整个 advanced 子模块做的全部事情，就是想方设法让这个误差更小**——要么先把分布变好量化（Shift/Smooth），要么更聪明地选 $s$（PieceWiseSearch），要么逐列补偿误差（GPTQ）。

> ⚠️ 铁律：PaddleSlim 强绑 Paddle，吃 `paddle.nn` 模型、吐 Paddle 模型。本文 API 名称/默认值以官方文档与源码为准，下文重在讲清"每个组件解决什么、参数怎么权衡"。

## 1. 地基/前置：LLM 量化难在哪

### 1.1 小模型量化 vs 大模型量化

CV 小模型直接 PTQ（训练后量化）统计一下 min/max 就能 INT8，掉点很小。但 LLM 不行，根因是**激活里的离群值(outlier)**：Transformer 的某些隐藏维度（通常是固定的几个 channel）数值特别大，比其余维度大 10~100 倍。

```
   正常激活通道                离群通道(outlier)
   ┌──┬──┬──┬──┬──┐          ┌──┬──┬──┬████┬──┐
   │▁▁│▁▁│▁▁│▁▁│▁▁│          │▁▁│▁▁│▁▁│████│▁▁│  ← 这一列数值爆炸
   └──┴──┴──┴──┴──┘          └──┴──┴──┴████┴──┘
   按 max 量化 → 步长合适      按 max 量化 → 步长被撑大
                              其余通道全被压成 0~1 个量化级
```

按全张量的 $\max(|x|)$ 定 $s$，离群通道把 $s$ 撑得很大，于是正常通道的值除以 $s$ 后全挤在 0 附近，量化后几乎都变成同一个数 —— **信息丢光，精度崩塌**。

### 1.2 三类对策（advanced 子模块的思路）

| 对策 | 代表算子 | 一句话 |
| --- | --- | --- |
| 改分布：平移 | **Shift** | 把激活整体减去一个偏置，拉回零附近 |
| 改分布：缩放迁移 | **Smooth** | 逐通道缩放，把激活的"难"转给权重（权重好量化）|
| 选更优 scale | **PieceWiseSearch** | 分段 + 网格搜，按误差挑最优缩放 |
| 逐列补偿误差 | **GPTQ** | 量化一列、把误差补给后面没量化的列 |
| 只诊断不量化 | **LayerWiseQuantError** | 测量每层量化误差，定位"不能量化"的层 |

### 1.3 两个必须先懂的概念

- **逐张量(per-tensor) vs 逐通道(per-channel)量化**：整张权重共用一个 $s$ 叫 per-tensor，简单但粗；每个输出通道一个 $s$ 叫 per-channel(channel-wise)，scale 更多、精度更高。代码里 `abs_max` 多指逐张量，`abs_max_channel_wise` 指逐通道。
- **权重量化 vs 激活量化**：权重是静态的（训练完就固定），好统计、好量化；激活是动态的（随输入变），还带离群值，难量化。所以业界常见 **W8A8**（权重激活都 INT8）、**W4A16**（只压权重到 INT4，激活留 FP16）。难点几乎都在激活侧，这也是 Shift/Smooth 存在的理由。

## 2. 整体架构：采样器 + 变换器 + 搜索器 + 量化器

advanced 子模块不是一个大函数，而是一组可拼装的角色，靠"喂校准数据 → 累积统计 → 更新权重"的循环驱动：

```
                  校准数据 dataloader
                         │  每个 batch 前向一次
                         ▼
   ┌─────────────────────────────────────────────────────────┐
   │  ① 采样器 Sampler                                        │
   │     EMASampler / MultiStepSampler                        │
   │     职责: 在前向时 hook 住激活, 累积统计量(min/max/均值)  │
   └───────────────┬─────────────────────────────────────────┘
                   │ 把统计量喂给
   ┌───────────────▼─────────────────────────────────────────┐
   │  ② 变换器 Transformer (等价变换,不改模型数学结果)         │
   │     Shift  : x → x - shift_bias                          │
   │     Smooth : x → x / s , 配套 W → W * s                   │
   └───────────────┬─────────────────────────────────────────┘
                   │ (Smooth 可选挂)
   ┌───────────────▼─────────────────────────────────────────┐
   │  ③ 搜索器 Searcher (可选)                                 │
   │     PieceWiseSearch: 分段 + 网格搜 alpha/scale, 选误差最小 │
   └───────────────┬─────────────────────────────────────────┘
                   │ 分布"捋顺"后
   ┌───────────────▼─────────────────────────────────────────┐
   │  ④ 量化器 / 诊断                                          │
   │     GPTQ              : 逐 Linear 层, Hessian 逐列量化补偿 │
   │     LayerWiseQuantError: 不量化, 只测每层量化误差          │
   └─────────────────────────────────────────────────────────┘
```

**关键心智模型**：①②③ 都不真的把数变成整数，它们只是把权重/激活的**数值分布预处理成"好量化"的样子**（等价变换，输出不变）；真正落 INT 的是 ④（GPTQ 或后续的 PTQ/QAT 流程）。所以典型管线是 **先 Shift/Smooth 捋分布，再 GPTQ 落比特**。

### 为什么要"喂数据循环 + step 自增"

Shift/Smooth/Search 都需要知道激活的真实分布，而激活只有把数据喂进去前向才会产生。所以代码模式固定为：

```
for data in dataloader():
    model(data)          # 前向触发 hook,采样器累积本 step 统计
    smooth.step += 1     # 告诉组件"又过了一个采样步"
smooth.update_weight()   # 采样够了,一次性更新权重/缩放
```

> 坑预警：`xxx.step += 1` 必须手动自增（采样器靠 step 判断采到第几步、要不要更新滑动平均）；漏掉它统计会不对。`update_weight()` 必须在循环**之后**调用一次。

## 3. Shift：把激活平移到零附近

**解决的问题**：有些激活分布不是关于 0 对称的，而是整体偏向一侧（有个直流偏置）。对称量化假设以 0 为中心，偏置会浪费掉一半量化级。Shift 先估计这个偏置 $b$，把激活平移成 $x' = x - b$，再把 $b$ 等价地吸收进下一层（如 LayerNorm 的 bias 或 Linear 的 bias），保证数学结果不变。

```
   平移前: 分布偏右,0 不在中心        平移后: 拉回以 0 为中心
   0        ┌──┐                      ┌──┐
   │     ┌──┤  ├──┐                ┌──┤  ├──┐
   │  ┌──┤  │  │  ├──┐          ┌──┤  │  │  ├──┐
   ───┴──┴──┴──┴──┴──┴──►   ──┴──┴──┴──┴──┴──┴──►
        对称量化浪费左半区间        对称量化区间用满
```

- **采样器**：示例用 `EMASampler`（指数滑动平均），逐 step 平滑地估计偏置，避免被单个 batch 的抖动带偏。
- **作用对象**：通常是激活的逐通道偏置（per-channel shift）。
- **何时用**：分布明显非零中心、或与 Smooth 配合时打前站。它本身不改变"离群值大"这个问题（那是 Smooth 的活），主要解决"不对称"。

```python
from paddleslim.quant.advanced import Shift, EMASampler

model = LLM()
model_config = {}
shift = Shift(model, model_config, sample_function=EMASampler())
for data in dataloader():
    model(data)
    shift.step += 1
shift.update_weight()   # 把估计到的偏置吸收进相邻层,完成等价平移
```

## 4. Smooth：把激活的难度搬到权重上

这是 LLM 量化最核心的一招（PaddleSlim 里 SmoothQuant 思想的实现）。

**核心洞察**：一个 Linear 层算的是 $Y = X W$。引入一个逐通道（per-channel，沿输入维度）的正缩放向量 $s$，做等价改写：

$$
Y = X W = \underbrace{(X / s)}_{\tilde{X}}\,\underbrace{(s\, W)}_{\tilde{W}}
$$

数学上完全相等（$s$ 一除一乘抵消）。但**量化难度被重新分配了**：

- 激活里离群通道很大 → 让对应的 $s$ 大一点，$\tilde{X}=X/s$ 就被压小，离群没了，好量化；
- 代价转给权重 $\tilde{W}=sW$，但权重本来分布规整、且可以逐通道量化，吸收得了这点放大。

```
   原始:  X(有离群,难量化)  ×  W(规整,易量化)
                │  按通道引入缩放 s
                ▼
   等价:  X/s (离群被抹平,易量化) × s·W (略放大,仍易量化)
          └──────── 难度从激活"搬"到了权重 ────────┘
   结果: 激活和权重都变得好量化, 整体量化误差↓
```

- **采样器**：示例用 `MultiStepSampler`，多步采样累积激活的统计（如逐通道最大值），用来计算每个通道该用多大的 $s$。
- **平衡强度 alpha**：$s$ 通常按 $s_j = \dfrac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}}$ 这类形式算，$\alpha\in[0,1]$ 调"搬多少难度给权重"。$\alpha$ 偏大 → 更照顾激活；偏小 → 更照顾权重。最优 $\alpha$ 因层而异，这正是 PieceWiseSearch 要搜的东西。

```python
from paddleslim.quant.advanced import Smooth, MultiStepSampler

smooth = Smooth(model, model_config, sample_function=MultiStepSampler())
for data in dataloader():
    model(data)
    smooth.step += 1
smooth.update_weight()   # 把逐通道缩放真正写进激活前的缩放和权重
```

## 5. PieceWiseSearch：分段搜索最优缩放

固定一个全局 $\alpha$ 往往不是最优——同一层里不同数值段，最优的平滑强度可能不同。PieceWiseSearch 干两件事：

1. **分段(piece-wise)**：把通道按数值大小分成 `k_piece` 段，每段单独搜一组缩放/平滑参数，比"全层一个值"更精细。
2. **网格搜索(search)**：在给定区间里枚举候选 $\alpha$ 与 scale，对每个候选**真的算一遍量化误差**（用 `loss_function`，如 `mse_loss`），选误差最小的那组。

```
   数值范围按大小切 k_piece=3 段
   ┌───────────┬───────────┬───────────┐
   │  小值段    │  中值段    │  大值段(含离群)│
   └─────┬─────┴─────┬─────┴─────┬─────┘
         │           │           │  每段独立搜:
         ▼           ▼           ▼
   alpha∈[0.2,0.8] × scale∈[1,5] 网格
         │           │           │
         ▼           ▼           ▼
   算 mse_loss, 取最小 → 该段最优 (alpha*, scale*)
```

它作为 `search_function` 挂到 `Smooth` 上协同工作：

```python
from paddleslim.quant.advanced import Smooth, MultiStepSampler, PieceWiseSearch, mse_loss

search_func = PieceWiseSearch(
    k_piece=3,                 # 分 3 段;越大越精细也越慢
    bits_length=8,             # 目标比特数(INT8)
    search_piece=False,        # 是否连"分几段"也一起搜
    search_alpha_min=0.2,      # smooth 强度 alpha 的搜索下界
    search_alpha_max=0.8,      # 上界
    search_scale_min=1.,       # 额外缩放的搜索下界
    search_scale_max=5.,       # 上界
    weight_quant_method='abs_max_channel_wise',  # 权重逐通道量化(精度高)
    act_quant_method='abs_max',                  # 激活按绝对最大值量化
    loss_function=mse_loss,    # 用 MSE 当评判标准,误差最小者胜出
)
smooth = Smooth(model, model_config,
                sample_function=MultiStepSampler(),
                search_function=search_func)
for data in dataloader():
    model(data)
    smooth.step += 1
smooth.update_weight()
```

**参数权衡速记**：

| 参数 | 调大的效果 | 调小的效果 |
| --- | --- | --- |
| `k_piece` | 更精细、误差更小，但搜索更慢 | 快但粗 |
| `[alpha_min, alpha_max]` | 搜索空间更广、更可能找到优解 | 范围窄、快但可能错过最优 |
| `[scale_min, scale_max]` | 同上，区间越宽搜得越久 | 收敛快 |
| `weight_quant_method=channel_wise` | 精度高（每通道一个 scale） | 用 `abs_max`(逐张量)更省但更粗 |

> 注意：搜索是"对每个候选都跑一遍量化+算 loss"，开销随 `k_piece × alpha 网格 × scale 网格` 近似线性增长，区间别盲目开很大。

## 6. GPTQ：用二阶信息逐列量化并补偿

前面的 Shift/Smooth/Search 都在"预处理分布"，GPTQ 则是真正把权重压成低比特的**量化器**，而且是目前精度最好的 PTQ 路线之一。

**核心思想**：量化是有损的，关键在"损得聪明"。GPTQ 把一个 Linear 的权重**逐列**量化；每量化完一列、产生一点误差，就用该层的二阶信息（Hessian $H = X^\top X$，来自校准激活）把这点误差**补偿到还没量化的列**上，让整体输出 $XW$ 的偏差最小。

```
   权重矩阵 W (按列处理)
   col0  col1  col2  col3 ...
   ┌──┬──┬──┬──┐
   │██│  │  │  │  ① 量化 col0 → 产生误差 e0
   └──┴──┴──┴──┘
        │ 用 Hessian 把 e0 分摊补偿给 col1..coln
        ▼
   ┌──┬──┬──┬──┐
   │██│██│  │  │  ② 量化 col1(已带补偿) → 误差 e1 再往后补
   └──┴──┴──┴──┘
        ... 逐列推进, 误差不断被后续列吸收 ...
   ┌──┬──┬──┬──┐
   │██│██│██│██│  完成: 整层输出偏差被最小化
   └──┴──┴──┴──┘
```

- **`act_order=True`**：按激活重要性（Hessian 对角元，约等于该列对输出影响大小）**从重要到次要排序**再逐列量化。重要的列先量化、误差还有大量后续列可补偿，精度更稳。几乎是"开了就有、副作用极小"的选项。
- **逐层独立**：对每个 `paddle.nn.Linear` 单独建一个 `GPTQ` 包装，喂数据累积该层 Hessian，再 `fasterquant()` 落地。

```python
from paddleslim.quant.advanced import GPTQ
import paddle

model = LLM()
for cur_name, cur_layer in model.named_sublayers():
    if type(cur_layer) == paddle.nn.Linear:
        gptq_layer = GPTQ(cur_layer)
        for data in dataloader():       # 前向累积该层的 Hessian(X^T X)
            model(data)
        gptq_layer.fasterquant(act_order=True)  # 按重要性排序,逐列量化+补偿
```

> GPTQ 与 Smooth 不冲突，反而互补：**先 Smooth 把激活离群压掉、把难度匀给权重，再 GPTQ 精细量化权重**，是常见组合拳。原理可对照 [[llm-compression/quantization/GPTQ]] 与 [[llm-compression/quantization/SmoothQuant]]。

## 7. LayerWiseQuantError：量化误差体检

不是所有层都该量化——有些层一量化就掉点（比如最后输出层、某些对误差极敏感的层）。`LayerWiseQuantError` 是个**只测不量**的诊断工具：它在每个 Linear 上模拟"如果把你量化，输出会偏多少"，把误差记下来，但**不真的改权重**。

```python
from paddleslim.quant.advanced import LayerWiseQuantError
import paddle

model = LLM()
for cur_name, cur_layer in model.named_sublayers():
    if type(cur_layer) == paddle.nn.Linear:
        cur_layer = LayerWiseQuantError(cur_layer)   # 包一层"误差探针"

for data in dataloader():
    model(data)                                       # 前向触发,累积误差

for cur_name, cur_layer in model.named_sublayers():
    if type(cur_layer) == LayerWiseQuantError:
        print(cur_name, cur_layer.losses.mean())      # 打印每层平均量化误差
```

**用法**：跑一遍打印各层 `losses.mean()`，误差排前几名的层就是"精度杀手"。决策：

```
   各层量化误差排序
   layer.31.mlp  ████████████  ← 误差最大: 考虑跳过/混精度(留 FP16)
   layer.0.qkv   ███████
   layer.15.out  ████
   ...           ▁▁           ← 误差小: 放心量化
```

据此做**混合精度**：误差大的层留高精度（FP16/INT8），其余压到更低比特（INT4），在"压缩率"和"精度"之间找平衡点。

## 典型流程与配置示例（讲含义）

把上面五个组件串成一条实战管线，顺序是有讲究的：

```
   ① 诊断(可选)   LayerWiseQuantError → 看哪些层敏感,决定混精度方案
        │
        ▼
   ② 平移         Shift  + EMASampler        → 分布拉回零中心
        │
        ▼
   ③ 平滑         Smooth + MultiStepSampler  → 抹平激活离群,难度搬给权重
        │         (可选挂 PieceWiseSearch 搜最优 alpha/scale)
        ▼
   ④ 落比特       GPTQ(act_order=True)        → 逐层逐列量化权重 + 误差补偿
        │
        ▼
   ⑤ 导出/部署    保存量化模型 → Paddle Inference(服务器)/Paddle Lite(端侧)
```

**每一步在做什么、为什么按这个顺序**：

1. **先诊断**：知道敌情（哪些层不能动），后面才好定混精度策略，避免在敏感层上白费力气。
2. **Shift 在 Smooth 前**：先把分布平移对称，Smooth 的逐通道缩放统计才准。
3. **Smooth 在 GPTQ 前**：Smooth 把激活离群压掉、权重难度可控后，GPTQ 再量化权重才不会被离群拖累。
4. **Search 挂在 Smooth 上**：搜的就是 Smooth 的 alpha/scale，二者是一体的。
5. **GPTQ 最后落地**：前面都是等价变换（输出不变），到这里才真正把权重变成 INT4/INT8。

> 上述 API 的精确签名、默认值、可选项以 PaddleSlim 官方文档与源码为准。本文重在讲清每个组件的职责与参数权衡，不背具体默认值。

## 常见问题/坑

| 问题 | 答案 / 排查 |
| --- | --- |
| `xxx.step` 不自增会怎样？ | 采样器靠 step 判断采到第几步、更新滑动平均；漏掉统计就不对，缩放算偏 |
| `update_weight()` 忘了调用？ | 那只是采了样、没真正更新权重/缩放，等于白跑，必须在循环后调一次 |
| 校准数据要多少？ | 几百条有代表性的样本通常够；太少统计不稳，离群估计不准 |
| 激活量化掉点严重？ | 八成是离群值；先上 Shift+Smooth 捋分布，再考虑 PieceWiseSearch 搜更优缩放 |
| `abs_max` 还是 `channel_wise`？ | `channel_wise`(逐通道)精度高、scale 多；`abs_max`(逐张量)省但粗。权重优先逐通道 |
| `act_order=True` 该开吗？ | GPTQ 里几乎总该开：按重要性排序量化，精度更好、副作用极小 |
| PieceWiseSearch 太慢？ | 开销 ≈ `k_piece × alpha网格 × scale网格`；缩小搜索区间或减 `k_piece` |
| 哪些层该跳过量化？ | 用 LayerWiseQuantError 测误差，排前几名的层留高精度（混合精度） |
| Shift 和 Smooth 谁先？ | Shift 先（平移对称）→ Smooth 后（缩放迁移），统计才准 |
| 能压 PyTorch 模型吗？ | 不能，PaddleSlim 强绑 Paddle，吃 `paddle.nn` 模型 |
| `paddle.fluid` 找不到？ | 已并入 `paddle.base`，老代码把 `paddle.fluid.*` 换 `paddle.base.*` |
| 量化完更慢/没变快？ | 检查部署后端是否真用了 INT 算子内核；伪量化只是模拟、不加速 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局知识地图，回总览
- [[llm-compression/PaddleSlim/README]] — PaddleSlim 总览（量化/剪枝/蒸馏/NAS 四件套）
- [[llm-compression/quantization/量化基础]] — 量化的通用原理（scale/零点/对称非对称/per-tensor vs per-channel）
- [[llm-compression/quantization/SmoothQuant]] — Smooth 背后的 SmoothQuant 算法详解
- [[llm-compression/quantization/GPTQ]] — GPTQ 二阶补偿量化的原理推导
- [[llm-compression/README]] — 模型压缩总览
