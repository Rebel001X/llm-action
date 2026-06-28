# PaddlePaddle

> 百度自研、国产化最成熟的产业级深度学习框架：用「动静统一」一套代码兼顾调试灵活与部署高效，靠 Fleet 做大规模分布式训练，再以 PaddleNLP 提供大模型端到端套件。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/pytorch/README]] [[ai-framework/README]]

## 阅读地图

| 节 | 看什么 | 一句话收获 |
| --- | --- | --- |
| 0 | 一句话锚点 | PaddlePaddle 到底是啥、解决什么 |
| 1 | 地基/前置 | 计算图、动态图 vs 静态图的最底层概念 |
| 2 | 整体架构 | 从 Python API 到底层 kernel 的分层 |
| 3 | 动静统一 | `@to_static` 怎么把动态图变静态图 |
| 4 | 张量与自动微分 | Tensor、反向传播在 Paddle 里怎么跑 |
| 5 | 分布式训练 Fleet | 数据/模型/流水线/混合并行怎么做 |
| 6 | PaddleNLP 大模型套件 | 训练→微调→压缩→推理一条龙 |
| 7 | 国产生态 | 昆仑芯/昇腾/飞桨硬件适配与社区 |
| 8 | 与 PyTorch 对比 | API、生态、部署、何时选谁 |
| 9 | 数值例子 | 手算一次前向+反向，看清梯度 |
| 10 | 常见问题 | 踩坑速查 |

---

## 0. 一句话锚点

**PaddlePaddle（飞桨）= 百度开源的深度学习框架**，定位对标 PyTorch / TensorFlow，但主打三件事：

1. **动静统一**：写代码用「动态图」（像 PyTorch 一样逐行执行、好调试），训练/部署时一键转「静态图」（先建图再跑、可优化可导出），同一份代码两种模式。
2. **产业级分布式**：内置 `Fleet` 高层 API，把数据并行、模型并行、流水线并行、参数服务器封装成几行配置。
3. **国产软硬件生态**：是国内 star 最多、产业落地最广的自研框架，深度适配昆仑芯、昇腾、海光等国产芯片，符合「自主可控」诉求。

> 一句话：**PyTorch 的易用 + TensorFlow 的可部署 + 国产化生态**，三者的并集。

---

## 1. 地基/前置：计算图、动态图与静态图

要理解 Paddle 的「动静统一」卖点，必须先把最底层的「计算图」讲清。

### 1.1 什么是计算图（Computation Graph）

任何神经网络的运算本质都是一串数学操作。把每个操作画成节点、数据流画成边，就得到一张「计算图」。例如 $y = relu(w \cdot x + b)$：

```
   x ──┐
        ▼
 w ──► [MatMul] ──► [Add] ──► [ReLU] ──► y
                      ▲
                 b ───┘
```

有了这张图，框架就能：① 正向求值 ② 反向自动求导（沿边反向传播梯度）③ 做图优化（算子融合、常量折叠）。

### 1.2 动态图 vs 静态图——核心矛盾

| 维度 | 动态图(Dynamic / Imperative) | 静态图(Static / Declarative) |
| --- | --- | --- |
| 何时建图 | **边执行边建**，写一行算一行 | **先把整张图建好再一次性跑** |
| 调试 | 易，可随时 print、打断点 | 难，图建好后看不到中间值 |
| 灵活性 | 高，可写 if/for 控制流 | 低，控制流要用特殊算子 |
| 性能/部署 | 慢，无法全局优化 | 快，可融合算子、导出独立模型 |
| 类比 | Python 解释执行 | C 编译后执行 |

PyTorch 早期纯动态图（易用但难部署），TensorFlow 1.x 纯静态图（难用但好部署）。**Paddle 的答案是：两者都要——这就是「动静统一」**（第 3 节展开）。

### 1.3 为什么部署要静态图

线上服务不能依赖 Python 解释器（慢、依赖重）。静态图能导出成一个「自包含的图文件 + 权重」，交给 C++ 推理引擎（Paddle Inference / Paddle Lite）执行，速度快、无 Python 依赖。

---

## 2. 整体架构：从 Python 到 Kernel 的分层

```
┌──────────────────────────────────────────────────────────┐
│  应用层套件：PaddleNLP / PaddleOCR / PaddleDetection ...    │  ← 开箱即用模型库
├──────────────────────────────────────────────────────────┤
│  高层 API：paddle.Model / Fleet(分布式) / 高阶组网          │  ← 少写代码
├──────────────────────────────────────────────────────────┤
│  基础 API：paddle.nn / paddle.optimizer / paddle.Tensor    │  ← 搭网络、定义层
├──────────────────────────────────────────────────────────┤
│  动静统一引擎：动态图执行器  ◄──@to_static──►  静态图(Program)│  ← 本框架核心
├──────────────────────────────────────────────────────────┤
│  中间表示 & 编译：IR / Pass 图优化 / CINN 编译器            │  ← 算子融合、加速
├──────────────────────────────────────────────────────────┤
│  算子(OP) & Kernel：CPU / CUDA / 国产芯片(昆仑/昇腾)        │  ← 真正干活的计算
└──────────────────────────────────────────────────────────┘
```

**自上而下读**：用户在套件或高层 API 写代码 → 引擎决定用动态还是静态执行 → 图经过 Pass 优化 → 最终落到具体硬件的 Kernel 上计算。**国产化适配主要发生在最底两层**（换 Kernel 后端即可支持新芯片）。

---

## 3. 动静统一（核心机制，重点）

这是 Paddle 区别于早期 PyTorch / TF 的招牌设计。

### 3.1 默认动态图，调试无障碍

Paddle 2.0 起**默认动态图**，写法和 PyTorch 几乎一样：

```python
import paddle
import paddle.nn as nn

class Net(nn.Layer):              # 对应 PyTorch 的 nn.Module
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)         # 逐行执行，可 print(x) 调试
```

### 3.2 一行装饰器转静态图：`@paddle.jit.to_static`

加上装饰器，Paddle 会用「**动转静**」技术，把你的 Python 函数（含 if/for 等控制流）翻译成等价的静态图 `Program`：

```python
@paddle.jit.to_static
def forward(self, x):
    return self.fc(x)
```

原理示意：

```
你写的动态图代码
      │
      ▼  ① AST 解析 + 控制流转写(把 Python if/for → 图算子 cond/while_loop)
   静态 Program(完整计算图)
      │
      ▼  ② 图优化 Pass(算子融合 / 显存复用 / 常量折叠)
   优化后的图
      │
      ├──► 训练：执行器高效跑(比动态图快)
      └──► 部署：paddle.jit.save 导出 → Paddle Inference 加载
```

### 3.3 为什么这很关键

- **开发期**：用动态图，断点、print 随便上，bug 好定位。
- **上线期**：`to_static` + `jit.save` 导出静态模型，C++ 引擎吃进去，**零 Python 依赖、可量化、可融合**。
- **同一份代码**：不用为部署重写一遍模型——这正是 PyTorch 早年（要靠 TorchScript/ONNX 折腾）和 TF 1.x（开发就痛苦）各自的短板。

```
          ┌──────── 同一份组网代码 ────────┐
          ▼                                ▼
     [动态图模式]                      [静态图模式 @to_static]
     调试 / 快速实验                    训练加速 / 导出部署
```

---

## 4. 张量与自动微分

### 4.1 Tensor——一切的基本数据

`paddle.Tensor` 是多维数组（标量0维、向量1维、矩阵2维、更高维张量），带 `dtype`（float32 等）和 `place`（CPU / GPU / 国产卡）。和 NumPy `ndarray`、PyTorch `Tensor` 同构，可零拷贝互转。

```python
import paddle
x = paddle.to_tensor([[1., 2.], [3., 4.]])   # 2x2 矩阵, float32
print(x.shape)     # [2, 2]
print(paddle.matmul(x, x))                    # 矩阵乘
```

### 4.2 自动微分（autograd）——反向传播怎么来的

训练 = 让损失 $L$ 对参数 $w$ 的梯度 $\frac{\partial L}{\partial w}$ 指导更新。Paddle 在前向时记录每个算子，反向时按**链式法则**自动求梯度：

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial w}$$

```python
w = paddle.to_tensor([2.0], stop_gradient=False)  # 需要梯度(对应 requires_grad=True)
x = paddle.to_tensor([3.0])
y = w * x          # 前向
loss = y * y       # L = (w·x)^2
loss.backward()    # 反向：自动算 dL/dw
print(w.grad)      # 见第 9 节手算验证
```

> 注意术语：PyTorch 用 `requires_grad=True`，Paddle 用 `stop_gradient=False`（默认参数不停梯度、叶子张量默认停梯度）。

---

## 5. 分布式训练：Fleet（重点）

模型一张卡放不下、或想加速，就要多卡多机。Paddle 把这套能力封装进 **`paddle.distributed.fleet`**，几行配置切换并行策略。

### 5.1 四种并行——先把概念讲透

```
数据并行(Data Parallel)                模型并行(Tensor Parallel)
每卡完整模型, 切数据                     一层权重切到多卡, 各算一片
┌GPU0┐ ┌GPU1┐ ┌GPU2┐                   ┌GPU0──┐┌GPU1──┐
│模型││模型││模型│ ← 各算各的            │权重左││权重右│ ← 拼一层
│data0││data1││data2│                  └──┬──┘└──┬──┘
└─┬──┘└─┬──┘└─┬──┘                       └──通信拼接──┘
  └──AllReduce 同步梯度──┘

流水线并行(Pipeline Parallel)          混合并行 = 上面几种叠加
按"层"切分到不同卡, 像流水线            DP × TP × PP 三维组合
GPU0:[L1-L4] → GPU1:[L5-L8] → ...      训练千亿参数大模型的标配
```

| 策略 | 切什么 | 解决 | 主要开销 |
| --- | --- | --- | --- |
| 数据并行 DP | 切数据 | 加速、扩 batch | 梯度 AllReduce 通信 |
| 张量并行 TP | 切单层权重 | 单层太大放不下 | 层内频繁通信 |
| 流水线并行 PP | 切层（阶段） | 模型整体太深太大 | 流水线气泡（bubble）|
| 分组切片 Sharding | 切优化器状态/梯度/参数 | 省显存（类 ZeRO）| 通信换显存 |

### 5.2 Fleet 用法骨架

```python
import paddle
from paddle.distributed import fleet

fleet.init(is_collective=True)            # 初始化集合通信
strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {               # 三维混合并行
    "dp_degree": 2,   # 数据并行 2
    "mp_degree": 2,   # 张量(模型)并行 2
    "pp_degree": 2,   # 流水线并行 2
}                                         # 2×2×2 = 至少 8 卡
model = fleet.distributed_model(model)
optimizer = fleet.distributed_optimizer(optimizer)
# 之后训练循环和单卡几乎一样
```

启动（多卡）：`python -m paddle.distributed.launch train.py`（具体参数以官方文档为准）。

> Paddle 还保留**参数服务器（PS）模式**，在搜广推（CTR、Embedding 巨大）场景里是强项——这块生态比 PyTorch 原生更成熟。

---

## 6. PaddleNLP：大模型端到端套件（重点）

PaddleNLP 是建在 Paddle 之上的 NLP/大模型库，相当于 Paddle 生态里的「HuggingFace Transformers + TRL + 推理 + 压缩」合集。

### 6.1 它覆盖的全链路

```
数据 → 预训练 → 微调(SFT) → 对齐(RLHF/DPO) → 压缩(量化/裁剪) → 推理部署
 │       │         │            │              │              │
 └───────┴─────────┴── PaddleNLP 一套 API 全包 ──┴──────────────┘
```

| 环节 | PaddleNLP 提供 |
| --- | --- |
| 模型库 | ERNIE（百度文心）、Llama、Qwen、GPT 等主流结构（以版本为准）|
| 高效微调 | LoRA、Prefix-Tuning 等 PEFT |
| 训练加速 | 融合算子、混合精度、与 Fleet 集成做大模型分布式 |
| 压缩 | 量化（INT8/INT4）、裁剪、蒸馏 |
| 推理 | 高性能推理（动态插入、量化推理），对接 Paddle Inference |

### 6.2 最小上手（Auto 系列接口，和 HF 风格一致）

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("模型名")         # 具体名以官方为准
model = AutoModelForCausalLM.from_pretrained("模型名")
inputs = tok("你好，介绍一下飞桨", return_tensors="pd")  # pd = paddle 张量
out = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out[0][0]))
```

> 接口风格刻意贴近 HuggingFace（`from_pretrained` / `AutoModel`），降低迁移成本；张量格式标记是 `"pd"` 而非 `"pt"`。**具体模型名、API 以官方仓库为准**：https://github.com/PaddlePaddle/PaddleNLP

---

## 7. 国产生态：自主可控的护城河

```
            ┌─────────────── 飞桨 PaddlePaddle ───────────────┐
   软件套件 │ NLP / OCR / 检测 / 语音 / 科学计算 / 强化学习      │
            ├──────────────────────────────────────────────────┤
   硬件适配 │ NVIDIA GPU │ 昆仑芯 XPU │ 昇腾 NPU │ 海光 │ 寒武纪 │
            ├──────────────────────────────────────────────────┤
   生态     │ AI Studio 在线平台 │ 模型库 │ 中文社区/文档/课程    │
            └──────────────────────────────────────────────────┘
```

- **硬件「全场景」适配**：通过统一算子层，Paddle 对接多家国产 AI 芯片，是「信创/自主可控」语境下的首选框架（**具体支持型号与成熟度以官方硬件支持列表为准**）。
- **中文友好**：官方中文文档、AI Studio 免费算力、丰富中文教程，国内学习门槛低。
- **产业落地**：工业质检、OCR、搜广推等场景大量真实部署，社区活跃度国内第一梯队。

---

## 8. 与 PyTorch 对比（重点）

| 维度 | PaddlePaddle | PyTorch |
| --- | --- | --- |
| 主导方 | 百度（国产）| Meta（海外，社区主导）|
| 编程范式 | **动静统一**（动态开发+静态部署一份代码）| 动态图为主，部署靠 TorchScript/`torch.compile`/ONNX |
| 模块基类 | `nn.Layer` | `nn.Module` |
| 梯度开关 | `stop_gradient` | `requires_grad` |
| 张量标记 | `return_tensors="pd"` | `"pt"` |
| 分布式 | Fleet（集合通信 + **参数服务器**双形态）| DDP / FSDP（集合通信为主，PS 弱）|
| 部署 | Paddle Inference / Lite，端到端成熟 | 需 ONNX/TensorRT 等外部链路 |
| 大模型套件 | PaddleNLP（自带，中文模型强如 ERNIE）| HuggingFace 生态（第三方但极繁荣）|
| 国产芯片 | 适配深、官方支持 | 主要 NVIDIA，国产靠厂商插件 |
| 社区规模 | 国内强、国际相对小 | **全球第一**，论文/开源默认 |
| 何时选 | 国产化合规、搜广推 PS、想训部署一体、中文场景 | 做前沿研究、要海量社区资源/复现论文、国际协作 |

**一句话权衡**：

```
研究/追前沿/对齐国际社区 ───► PyTorch
国产化合规 / 端到端部署 / 搜广推 PS / 中文大模型 ───► PaddlePaddle
```

二者底层概念（Tensor、autograd、计算图）完全相通，**迁移成本主要在 API 命名**，理解一个能很快上手另一个。可对照阅读 [[ai-framework/pytorch/README]]。

---

## 9. 数值例子：手算一次前向 + 反向

验证第 4.2 节那段代码。设 $w=2,\ x=3$，损失 $L=(w\cdot x)^2$。

**前向**：

$$y = w\cdot x = 2\times 3 = 6,\qquad L = y^2 = 36$$

**反向**（链式法则）：

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial w} = (2y)\cdot(x) = (2\times 6)\times 3 = 36$$

所以 `w.grad` 应得 **36**。再做一步梯度下降，学习率 $\eta=0.01$：

$$w \leftarrow w - \eta\frac{\partial L}{\partial w} = 2 - 0.01\times 36 = 1.64$$

更新后 $w=1.64$，新损失 $L=(1.64\times3)^2 = (4.92)^2 \approx 24.2 < 36$，损失下降，方向正确。这就是 Paddle `loss.backward()` + `optimizer.step()` 内部干的事。

数据流图：

```
w=2 ─┐
      ▼
     [×] ──► y=6 ──► [平方] ──► L=36
x=3 ─┘
  反向: dL/dy=2y=12 ──► dL/dw = dL/dy · x = 12·3 = 36
```

---

## 安装 / 镜像 / 可视化（速查）

> 版本号以官方文档为准：https://www.paddlepaddle.org.cn/install/quick

```bash
# 安装（清华源加速）
python -m pip install paddlepaddle==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade paddlenlp
```

```bash
# Docker 镜像
docker pull paddlepaddle/paddle:2.5.2-gpu-cuda11.7-cudnn8.4-trt8.4   # GPU
docker pull paddlepaddle/paddle:2.5.2                                 # CPU
```

```bash
# 训练可视化（类似 TensorBoard）
visualdl --logdir ./log
```

```python
# 验证安装
import paddle
paddle.utils.run_check()   # 打印是否成功、是否检测到 GPU
```

---

## 常见问题

| 问题 | 答案 |
| --- | --- |
| Paddle 和 PyTorch 像吗？ | API 高度相似，`nn.Layer`↔`nn.Module`、`stop_gradient`↔`requires_grad`，底层概念一致，迁移主要是改命名 |
| 「动静统一」到底解决啥？ | 开发用动态图好调试，部署用静态图好优化/导出，**同一份代码**两种模式，免去重写 |
| 动转静怎么做？ | 给函数加 `@paddle.jit.to_static`，框架解析 AST 把控制流转成图算子 |
| 训大模型用什么？ | `fleet` 配 `hybrid_configs` 做 DP×MP×PP 混合并行 + Sharding 省显存 |
| 大模型套件叫什么？ | **PaddleNLP**，覆盖预训练→微调→对齐→压缩→推理全链路，接口仿 HuggingFace |
| 张量标记为什么是 "pd"？ | Paddle 的简写（PyTorch 是 "pt"、TF 是 "tf"）|
| 支持国产芯片吗？ | 适配昆仑芯/昇腾等，信创首选；**具体型号与成熟度以官方支持列表为准** |
| 部署比 PyTorch 强在哪？ | 自带 Paddle Inference / Lite，静态图直接导出，无需绕 ONNX/TensorRT |
| 什么时候别用 Paddle？ | 追前沿研究、要复现海外论文、依赖庞大国际社区时，PyTorch 资源更多 |
| 版本/CLI 不确定怎么办？ | 一律以官方文档为准，本文版本号仅为示例 |

---

## 🔗 跳转链接

- 知识地图总览：[[00-知识地图]]
- 同类框架对照：[[ai-framework/pytorch/README]]
- 框架总览：[[ai-framework/README]]
- 官方安装：https://www.paddlepaddle.org.cn/install/quick
- PaddleNLP 仓库：https://github.com/PaddlePaddle/PaddleNLP
