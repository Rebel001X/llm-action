# 量化可视化：把权重 / 激活的"分布与离群点"画出来

> 量化难不难，肉眼能看见——用 3D 柱状图把 LLM 的权重张量和激活张量画出来，离群点（outlier）一眼就现形，这正是 SmoothQuant 等算法的出发点。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-compression/quantization/量化基础]] · [[llm-compression/quantization/SmoothQuant]] · [[llm-compression/quantization/大模型量化概述]] · [[llm-algo/transformer/模型架构]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|-------|
| §0 锚点 | 一句话讲清"为什么要画" | 离群点可视化 |
| §1 地基 | 权重 / 激活张量长什么样、为何要可视化 | tensor / outlier |
| §2 取张量 | 从 HF 模型里取出 weight 和 activation | `q_proj.weight` / forward hook |
| §3 降采样 | 张量太大，怎么 strip / 分块 max 池化 | reshape / max pool |
| §4 画 3D 柱图 | `bar3d` 的每个参数都干什么 | meshgrid / bar3d |
| §5 配色 | 原文的 matplotlib 颜色表怎么用 | named_colors |
| §6 读图 | 从图里看出"权重好量化、激活难量化" | per-channel outlier |
| 实操 | 两个 notebook 的完整真实代码 | qwen_visual / qwen_activate_visual |
| 坑 | 中文乱码 / 渲染慢 / OOM | Agg / simplify |

---

## 0. 一句话锚点

**量化的本质是"把浮点数塞进更少的 bit"，而塞得进塞不进，取决于数值的分布有多"平"。** 分布平整、没有极端值 → 量化误差小；分布里有几个特别大的"离群点"→ 它们会撑大量化范围，把绝大多数正常值挤进很窄的格子里，精度崩塌。

可视化就是把这件抽象的事**画成肉眼可见的 3D 柱状图**：

```
   平整的权重分布(好量化)         有离群点的激活分布(难量化)
   Value                          Value
    │ ▁▂▂▁▂▁▂▂▁▂▁▂                 │              █  ← outlier 一柱擎天
    │ ▂▁▂▂▁▂▁▂▂▁▂▁                 │              █
    │ ▁▂▁▂▂▁▂▂▁▂▁▂                 │ ▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁  ← 其余被压扁
    └──────────────► feature       └──────────────────► channel
```

这张对照图，就是 **SmoothQuant 论文那张经典图的复刻**——本目录的两个 notebook 干的就是这件事。

---

## 1. 地基：权重张量 vs 激活张量

LLM 的每个 Linear 层都做 $Y = XW^\top$。量化要同时面对两类张量：

| | 权重 W (weight) | 激活 X (activation) |
|---|----------------|---------------------|
| 来源 | 训练好后**固定**存在模型里 | 推理时**随输入动态**算出来 |
| 形状(以 q_proj 为例) | `[out_feature, in_feature]` | `[token, channel/hidden]` |
| 分布特点 | 相对平整、近高斯 | **per-channel 有离群点**，某些通道值特别大 |
| 量化难度 | ✅ 容易（INT8 几乎无损） | ❌ 难（直接 INT8 掉点严重） |

> SmoothQuant 的核心洞察（notebook 里直接贴了它的摘要）：**"weights are easy to quantize while activations are not"**——权重好量化、激活难量化。所以它把激活的量化难度，用一个数学等价变换"搬"一部分到权重上去。要理解这句话，最好的方式就是**亲眼看一眼两者的分布差异**，这就是可视化的价值。

**为什么激活会有离群点？** Transformer 在某些隐藏维度（channel）上会系统性地产生大幅值（与 LayerNorm、残差累积有关），这些"系统性大通道"在不同 token 上一致地大，形成图里那一排"高墙"。

---

## 2. 第一步：从 HF 模型里取出张量

### 2.1 取权重（qwen_visual.ipynb）

权重是静态的，直接从 `state_dict` 里取，无需前向：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_id = '/model/ModelScope/Qwen/Qwen3-0.6B'
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True,
)

# 取第 0 层 注意力的 q_proj 权重 -> [out_feature, in_feature]
weights_origin = model.model.layers[0].self_attn.q_proj.weight.cpu().detach().float().numpy()
rows_origin, cols_origin = weights_origin.shape   # 例如 [2048, 1024]
```

逐项拆解，每个动词都不能少：

```
.weight              拿到 nn.Parameter（带梯度、在 GPU 上）
   │
.cpu()               搬回 CPU —— numpy 不认 GPU 张量
   │
.detach()            切断计算图 —— 否则 .numpy() 报 "requires grad"
   │
.float()             bf16/fp16 → fp32 —— matplotlib 只吃 float32/64
   │
.numpy()             变成 ndarray，才能 reshape / 画图
```

> 顺序记忆：**先 cpu → 再 detach → 再 float → 再 numpy**。`.detach()` 必须在 `.numpy()` 前，否则 PyTorch 抛 `RuntimeError: Can't call numpy() on Tensor that requires grad`。

### 2.2 取激活（qwen_activate_visual.ipynb）

激活是动态的，必须**注册 forward hook**，跑一次前向才能抓到：

```python
import torch
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

activations = {}
def get_activation(name):
    def hook(module, input, output):
        activations[name] = {
            'input':  input[0].detach().cpu().numpy(),
            'output': output[0].detach().cpu().numpy(),
        }
    return hook

# 给每一层 decoder 都挂上钩子
for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(get_activation(f'layer_{i}'))

# 喂一段文本（notebook 里用的正是 SmoothQuant 论文摘要），触发前向
# inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
# model(**inputs)

# 取第 26 层输出，摊平成 [token, hidden=1024]
activate_origin = activations['layer_26']['output'].reshape(-1, 1024)
```

**forward hook 机制图解：**

```
input ──► [layer_26 forward 计算] ──► output
                    │
                    └─►  hook(module, input, output) 被自动调用
                              │
                              └─► activations['layer_26'] = {input, output}
                                  （旁路抓取，不改变前向结果）
```

> hook 是"旁路监听"：它在该层算完后被框架自动回调，把中间结果存进字典，不影响主计算流。这是抓任意中间层激活最干净的方式，不用改模型源码。

---

## 3. 第二步：降采样——张量太大画不动

`q_proj.weight` 动辄 `[2048, 1024] ≈ 200 万` 个元素，3D 柱图一柱一格会卡死。两个 notebook 给了两种降采样：

### 3.1 简单跳点（strip / 切片）

```python
activate_strip = 2
token_strip = 1
weights = activate_origin[::token_strip, ::activate_strip]   # 每隔 N 个取 1 个
```

`a[::strip]` 即每 `strip` 个采样一个，最省事但会**漏掉离群点**（采样点可能正好跳过那根高墙）。

### 3.2 分块 max 池化（更能保留离群点，权重 notebook 用的）

```python
sub_mat_size = 8
weights = (weights_origin
    .reshape(rows_origin//sub_mat_size, sub_mat_size,
             cols_origin//sub_mat_size, sub_mat_size)
    .swapaxes(1, 2)                       # 把两个 block 维度凑到一起
    .reshape(-1, sub_mat_size, sub_mat_size)
    .max(axis=(1, 2))                     # 每个 8×8 小块取最大值
    .reshape(rows_origin//sub_mat_size, cols_origin//sub_mat_size))
```

把矩阵切成 `8×8` 的小块，每块用 **max** 代表。为什么用 max 不用 mean？

> **量化关心的是极值（决定量化范围 = max-min），不是均值。** 用 max 池化降采样，能保证离群点不被平均掉、仍然"冒头"。这是这段代码最值得学的一点。

```
原始 16×16              max 池化后 2×2
┌──┬──┬──┬──┐           每个 8×8 块 → 1 个 max
│  ███     │  ← 离群    ┌────┬────┐
│          │           │ 9.2│ 0.3│  离群点 9.2 被保留
│          │     ═══►   ├────┼────┤
│          │           │ 0.2│ 0.4│
└──┴──┴──┴──┘           └────┴────┘
```

---

## 4. 第三步：画 3D 柱状图 `bar3d`

两个 notebook 用同一套 `ax.bar3d(...)` 绘图骨架。先建网格坐标，再把每个矩阵元素画成一根柱子：

```python
import numpy as np, matplotlib, random
matplotlib.use('Agg')          # 无界面后端，见 §坑
import matplotlib.pyplot as plt
plt.ioff()

# 渲染加速：路径简化
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 0.1

rows, cols = weights.shape
x = np.arange(0, cols)         # 列 → x 轴
y = np.arange(0, rows)         # 行 → y 轴
X, Y = np.meshgrid(x, y)       # 生成每根柱子的 (x,y) 坐标

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
bottom = np.zeros_like(weights)  # 所有柱子从 z=0 起

ax.bar3d(
    X.ravel(), Y.ravel(), bottom.ravel(),   # 每根柱子的 (x, y, z) 起点
    0.01, 0.01,                             # 柱子的 dx, dy（底面宽度）
    np.abs(weights).ravel(),                # dz = 柱高 = |权重值|
    color=colors_np, shade=True,
)
```

**`bar3d` 的 6 个位置参数，就是"画很多个长方体"：**

```
bar3d(x, y, z,  dx, dy, dz)
       └─起点坐标─┘  └─长宽高─┘
   一根柱子 = 从 (x,y,z) 出发，底面 dx×dy，高 dz

  z=|value| ┌─┐         dz 越高 = 该权重绝对值越大
            │ │         => 离群点 = 鹤立鸡群的一根高柱
        ────┴─┴──── (x,y) 平面：feature × feature 网格
```

> 取 `np.abs(weights)` 是因为只关心**幅值大小**（量化范围由 |max| 决定），正负无关紧要。`shade=True` 加光照让 3D 立体感更强。

轴标签也对应了两类张量的物理含义：

| notebook | x 轴 | y 轴 | z 轴 |
|---|---|---|---|
| 权重 qwen_visual | `in feature` | `out feature` | `Value` |
| 激活 qwen_activate | `channel` | `token` | `Value` |

---

## 5. 配色：原文的 matplotlib 颜色表

notebook 给柱子上色用的是 matplotlib 的**具名颜色（named colors）**。原 README 收录了完整的颜色名→十六进制对照表，正是这里要查的参考表。

上色逻辑（两个 notebook 一致）：**同一行的柱子用同一组随机色**，让 3D 图按行有色带、更易分辨：

```python
colors_element = ["red", "yellow", "blue", "green", "orange"]
colors = []
temp = random.choices(colors_element, k=cols)   # 先为一行随机选 cols 个颜色
for i in range(0, rows):
    colors.extend(temp)                          # 每一行复用同一组
colors_np = np.array(colors)
```

> 注释里还给了另一种调色板：`["blue", "cornflowerblue", "mediumturquoise", "goldenrod"]`——这些名字都能在下面的对照表里查到十六进制值。

**matplotlib 具名颜色对照表**（官方：https://matplotlib.org/stable/gallery/color/named_colors.html ）：

```python
{
	'aliceblue': '#F0F8FF',          'antiquewhite': '#FAEBD7',
	'aqua': '#00FFFF',               'aquamarine': '#7FFFD4',
	'azure': '#F0FFFF',              'beige': '#F5F5DC',
	'bisque': '#FFE4C4',             'black': '#000000',
	'blanchedalmond': '#FFEBCD',     'blue': '#0000FF',
	'blueviolet': '#8A2BE2',         'brown': '#A52A2A',
	'burlywood': '#DEB887',          'cadetblue': '#5F9EA0',
	'chartreuse': '#7FFF00',         'chocolate': '#D2691E',
	'coral': '#FF7F50',              'cornflowerblue': '#6495ED',
	'cornsilk': '#FFF8DC',           'crimson': '#DC143C',
	'cyan': '#00FFFF',               'darkblue': '#00008B',
	'darkcyan': '#008B8B',           'darkgoldenrod': '#B8860B',
	'darkgray': '#A9A9A9',           'darkgreen': '#006400',
	'darkkhaki': '#BDB76B',          'darkmagenta': '#8B008B',
	'darkolivegreen': '#556B2F',     'darkorange': '#FF8C00',
	'darkorchid': '#9932CC',         'darkred': '#8B0000',
	'darksalmon': '#E9967A',         'darkseagreen': '#8FBC8F',
	'darkslateblue': '#483D8B',      'darkslategray': '#2F4F4F',
	'darkturquoise': '#00CED1',      'darkviolet': '#9400D3',
	'deeppink': '#FF1493',           'deepskyblue': '#00BFFF',
	'dimgray': '#696969',            'dodgerblue': '#1E90FF',
	'firebrick': '#B22222',          'floralwhite': '#FFFAF0',
	'forestgreen': '#228B22',        'fuchsia': '#FF00FF',
	'gainsboro': '#DCDCDC',          'ghostwhite': '#F8F8FF',
	'gold': '#FFD700',               'goldenrod': '#DAA520',
	'gray': '#808080',               'green': '#008000',
	'greenyellow': '#ADFF2F',        'honeydew': '#F0FFF0',
	'hotpink': '#FF69B4',            'indianred': '#CD5C5C',
	'indigo': '#4B0082',             'ivory': '#FFFFF0',
	'khaki': '#F0E68C',              'lavender': '#E6E6FA',
	'lavenderblush': '#FFF0F5',      'lawngreen': '#7CFC00',
	'lemonchiffon': '#FFFACD',       'lightblue': '#ADD8E6',
	'lightcoral': '#F08080',         'lightcyan': '#E0FFFF',
	'lightgoldenrodyellow': '#FAFAD2','lightgray': '#D3D3D3',
	'lightgreen': '#90EE90',         'lightpink': '#FFB6C1',
	'lightsalmon': '#FFA07A',        'lightseagreen': '#20B2AA',
	'lightskyblue': '#87CEFA',       'lightslategray': '#778899',
	'lightsteelblue': '#B0C4DE',     'lightyellow': '#FFFFE0',
	'lime': '#00FF00',               'limegreen': '#32CD32',
	'linen': '#FAF0E6',              'magenta': '#FF00FF',
	'maroon': '#800000',             'mediumaquamarine': '#66CDAA',
	'mediumblue': '#0000CD',         'mediumorchid': '#BA55D3',
	'mediumpurple': '#9370DB',       'mediumseagreen': '#3CB371',
	'mediumslateblue': '#7B68EE',    'mediumspringgreen': '#00FA9A',
	'mediumturquoise': '#48D1CC',    'mediumvioletred': '#C71585',
	'midnightblue': '#191970',       'mintcream': '#F5FFFA',
	'mistyrose': '#FFE4E1',          'moccasin': '#FFE4B5',
	'navajowhite': '#FFDEAD',        'navy': '#000080',
	'oldlace': '#FDF5E6',            'olive': '#808000',
	'olivedrab': '#6B8E23',          'orange': '#FFA500',
	'orangered': '#FF4500',          'orchid': '#DA70D6',
	'palegoldenrod': '#EEE8AA',      'palegreen': '#98FB98',
	'paleturquoise': '#AFEEEE',      'palevioletred': '#DB7093',
	'papayawhip': '#FFEFD5',         'peachpuff': '#FFDAB9',
	'peru': '#CD853F',               'pink': '#FFC0CB',
	'plum': '#DDA0DD',               'powderblue': '#B0E0E6',
	'purple': '#800080',             'rebeccapurple': '#663399',
	'red': '#FF0000',                'rosybrown': '#BC8F8F',
	'royalblue': '#4169E1',          'saddlebrown': '#8B4513',
	'salmon': '#FA8072',             'sandybrown': '#F4A460',
	'seagreen': '#2E8B57',           'seashell': '#FFF5EE',
	'sienna': '#A0522D',             'silver': '#C0C0C0',
	'skyblue': '#87CEEB',            'slateblue': '#6A5ACD',
	'slategray': '#708090',          'snow': '#FFFAFA',
	'springgreen': '#00FF7F',        'steelblue': '#4682B4',
	'tan': '#D2B48C',                'teal': '#008080',
	'thistle': '#D8BFD8',            'tomato': '#FF6347',
	'turquoise': '#40E0D0',          'violet': '#EE82EE',
	'wheat': '#F5DEB3',              'white': '#FFFFFF',
	'whitesmoke': '#F5F5F5',         'yellow': '#FFFF00',
	'yellowgreen': '#9ACD32',
}
```
（上表为节选展示，完整 148 个 CSS4 颜色名以官方链接为准。）

---

## 6. 怎么读图：看出"权重好量化、激活难量化"

把两张图并排看，量化算法的动机就清楚了：

```
   q_proj 权重图(layer 0)        layer_26 激活图
   柱高均匀、参差不大             少数 channel 高耸如墙
   ┌─────────────────┐          ┌─────────────────┐
   │ ▃▄▃▄▃▄▃▄▃▄▃▄▃▄▃ │          │       █         │ ← 这一列 channel
   │ ▄▃▄▃▄▃▄▃▄▃▄▃▄▃▄ │          │       █         │   在所有 token 上都大
   │ ▃▄▃▄▃▄▃▄▃▄▃▄▃▄▃ │          │ ▂▂▂▂▂▂█▂▂▂▂▂▂▂▂ │   = per-channel outlier
   └─────────────────┘          └─────────────────┘
   → 整体动态范围小              → 动态范围被离群柱撑大
   → INT8 几乎无损              → 直接 INT8 把正常值压扁 → 掉点
```

**量化误差与动态范围的关系**（per-tensor 对称量化）：

$$\Delta = \frac{\max(|X|)}{2^{b-1}-1}, \qquad x_q = \text{round}\!\left(\frac{x}{\Delta}\right)$$

离群点把 $\max(|X|)$ 抬高 → 步长 $\Delta$ 变大 → 正常值都落进同一个量化格子 → 信息丢失。

**数值手算**：设某激活通道正常值在 $[-1, 1]$，但有一个离群点 $= 64$。INT8（$b=8$，$2^7-1=127$）：

$$\Delta = \frac{64}{127} \approx 0.504$$

那么正常值 $0.4$ 量化为 $\text{round}(0.4/0.504)=\text{round}(0.79)=1$，反量化回 $1\times0.504=0.504$——误差超过 25%！而若没有离群点（$\max=1$），$\Delta=1/127\approx0.0079$，$0.4$ 量化为 $51$，反量化 $0.402$，几乎无损。**一个离群点，让步长放大了 64 倍。**

> 这正是 SmoothQuant 要解决的：通过 $\hat X = X/s,\ \hat W = W\cdot s$ 的等价变换，把激活的离群"幅度"按通道搬一部分到权重上（权重本来很好量化、有余量），让两者都好量化。可视化就是验证这个变换前后分布变化的最直观工具。详见 [[llm-compression/quantization/SmoothQuant]]。

---

## 实操：两个 notebook 的真实代码与产物

| notebook | 画什么 | 输入 | 关键参数 |
|----------|--------|------|---------|
| `qwen_visual.ipynb` | 第 0 层 `q_proj` **权重** | 无需文本 | `sub_mat_size=8` 分块 max 池化 |
| `qwen_activate_visual.ipynb` | 第 26 层 **激活输出** | SmoothQuant 摘要文本 | `activate_strip=2, token_strip=1` |

**保存图片并计时（权重 notebook 末尾真实代码）：**

```python
ax.set_xlabel('in feature', labelpad=15)
ax.set_ylabel('out feature', labelpad=15)
ax.set_zlabel('Value', labelpad=10)
ax.set_title('Linear Layer Tensor', pad=20)
ax.set_zlim(0, weights.max().item())

start = time.perf_counter()
plt.savefig("demo_strip8.pdf", dpi=100)   # 保存为 PDF 矢量图
end = time.perf_counter()
print(f"执行时间：{end - start:.6f} 秒")
```

> 选 **PDF 矢量格式**而非 PNG：3D 柱图的边线在 PDF 里无限放大不糊，适合放论文/汇报；代价是文件大、渲染慢——所以前面才要 `path.simplify` 加速。

**复现步骤：**
1. 准备 Qwen3-0.6B 权重（notebook 里路径为 `/model/ModelScope/Qwen/Qwen3-0.6B`，按本地实际改）。
2. `pip install transformers matplotlib numpy torch`。
3. 跑 `qwen_visual.ipynb` 看权重分布；跑 `qwen_activate_visual.ipynb`（先取消注释 tokenize + 前向那两行）看激活分布。
4. 对比两张 `*.pdf`，亲眼确认"权重平、激活有离群"。

---

## 常见问题 / 坑

| 现象 | 原因 | 解决 |
|------|------|------|
| 标签 / 标题中文变方框乱码 | 源码注释本就是 GBK 乱码；matplotlib 默认字体无中文 | 轴标签用英文（notebook 即如此）；要中文设 `plt.rcParams['font.sans-serif']=['SimHei']` |
| `Can't call numpy() on Tensor that requires grad` | 漏了 `.detach()` | 严格按 `.cpu().detach().float().numpy()` 顺序 |
| 显存/内存 OOM 或画图卡死 | 张量太大（百万元素逐柱画） | 先降采样：strip 切片 或 §3.2 的 8×8 max 池化 |
| 渲染奇慢 | 3D 柱图柱子数巨大 | `path.simplify=True` + `simplify_threshold=0.1`；用 `matplotlib.use('Agg')` 无界面后端 |
| 离群点没画出来 | 用简单 strip 切片正好跳过了高柱 | 改用 **max 池化**降采样，保证极值冒头 |
| hook 抓不到激活 | 没跑前向 / 取错层名 | 注册 hook 后必须 `model(**inputs)`；`activations` 字典在前向后才有值 |
| `bar3d` 柱子全黑一团 | 颜色数组长度 ≠ 柱子数 | `colors` 长度须等于 `rows*cols`（见 §5 的拼接逻辑） |
| 取了负权重柱子朝下 | 没取绝对值 | 画幅值用 `np.abs(weights)` |

---

## 🔗 跳转链接

- 枢纽：[[00-知识地图]]
- 量化原理：[[llm-compression/quantization/量化基础]] · [[llm-compression/README]] · [[llm-compression/quantization/GPTQ]] · [[llm-compression/quantization/fp8]]
- 本图直接服务的算法：[[llm-compression/quantization/SmoothQuant]]（激活离群点搬移） · [[llm-compression/quantization/大模型量化概述]]
- 模型结构（理解 q_proj / hidden / token 维度）：[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]] · [[llm-algo/旋转编码RoPE]]
- 推理侧消费量化模型：[[llm-inference/README]] · [[llm-inference/vllm/README]] · [[llm-inference/解码策略]] · [[llm-inference/PD分离]]
- 显存 / KV：[[llm-optimizer/kv-cache]] · [[llm-optimizer/FlashAttention]] · [[docs/transformer内存估算]]
- 训练 / 对齐：[[llm-train/README]] · [[llm-train/peft/PEFT-API]] · [[llm-alignment/RLHF]] · [[llm-alignment/DPO]]
- 框架：[[ai-framework/pytorch/README]] · [[ai-framework/deepspeed/README]] · [[ai-framework/megatron-lm/README]]
- 硬件 / 算力：[[ai-infra/算力/GPU工作原理]] · [[ai-infra/ai-hardware/硬件对比]] · [[ai-infra/网络/InfiniBand]] · [[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/昇腾NPU]] · [[ai-infra/ai-hardware/AI芯片软件生态]]
- 评测：[[llm-eval/README]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
