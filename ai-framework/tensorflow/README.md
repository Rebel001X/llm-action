# TensorFlow

> TensorFlow 是 Google 开源的深度学习框架，它的灵魂是把神经网络的计算抽象成一张「数据流图（dataflow graph）」，先把图编译/优化好再喂数据高速执行——靠这套「静态图 + XLA 编译 + 原生 TPU 支持」在大规模工业部署里站稳脚跟，又用 Keras + eager 模式补上易用性。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/pytorch/README]] [[ai-framework/README]]

## 阅读地图

| 节 | 主题 | 你会得到什么 | 难度 |
|----|------|--------------|------|
| 0 | 一句话锚点 | 框架的本质定位 | ★ |
| 1 | 地基/前置 | 它解决什么问题、「数据流图」是什么 | ★★ |
| 2 | 静态图 vs eager | 两种执行模式的根本区别 | ★★★ |
| 3 | tf.function / AutoGraph | eager 写、静态图跑：怎么做到的 | ★★★★ |
| 4 | Keras | 高层 API：怎么三行搭网络 | ★★ |
| 5 | 自动微分 GradientTape | 反向传播在 TF 里怎么算 | ★★★ |
| 6 | 分布式策略 Strategy | 多卡/多机训练的统一接口 | ★★★★ |
| 7 | TPU + XLA | 专用芯片为什么快、怎么用 | ★★★★ |
| 8 | 部署生态 | Serving / Lite / JS：训练之外的半壁江山 | ★★ |
| 9 | 历史地位与现状 | 从王者到守成，发生了什么 | ★★ |
| - | 数值例子 / 对照 / FAQ / 跳转 | 横向对比与查漏 | ★ |

## 0. 一句话锚点

**TensorFlow = 把神经网络的计算画成一张「图」，再让引擎把整张图编译、优化、调度到 CPU/GPU/TPU 上高速执行的框架。**

名字本身就是定义：**Tensor（张量，多维数组）+ Flow（在图里流动）= 张量在数据流图中流动**。

```
   "TensorFlow" 这个名字的字面意思
   ┌──────────────────────────────────────────────┐
   │   Tensor (张量)  ──flow──▶  在一张计算图里流动 │
   │                                                │
   │   节点 = 运算 (matmul, add, relu...)           │
   │   边   = 张量 (数据沿边从一个运算流到下一个)   │
   └──────────────────────────────────────────────┘
```

它和 PyTorch 解决的是**同一个问题**（自动微分 + GPU 张量计算 + 网络/分布式工具箱），但**哲学相反**：TensorFlow 起家于「**先把整张图定义清楚、再整体优化执行**」（静态图），PyTorch 起家于「**边写边跑**」（动态图）。理解这条主线，整个框架就通了。

## 1. 地基/前置：什么是「数据流图」，为什么要它

回顾训练一个网络的本质循环（和 PyTorch 篇一致）：

```
   ┌────────────────────────────────────────────────────┐
   │  1. 前向：输入 x → 模型 → 预测 ŷ                    │
   │  2. 损失：loss = L(ŷ, y)                             │
   │  3. 反向：∂loss/∂每个参数 (梯度)                    │
   │  4. 更新：参数 ← 参数 - 学习率 × 梯度              │
   │  5. 回到 1，直到收敛                                 │
   └────────────────────────────────────────────────────┘
```

任何框架都要把第 3 步（手算梯度）和「搬上 GPU、管显存、多机通信」自动化。**TensorFlow 的独特选择**是：不直接执行你的代码，而是先把你的计算**翻译成一张图**，再把图交给引擎。

### 为什么「先建图」是个好主意？

把计算变成一张**与具体编程语言解耦的图**后，引擎能做很多人手做不到的全局优化：

```
   你写的代码 (Python)
        │  翻译
        ▼
   ┌─────────────────────────┐
   │   数据流图 (Graph)        │   ← 一种"中间表示"(IR)，
   │   y = relu(W·x + b)      │     纯粹描述"算什么"、不含 Python
   │                          │
   │   x ─┐                   │
   │      ├─[MatMul]─[Add]─[Relu]─ y
   │   W ─┘         │         │
   │   b ───────────┘         │
   └─────────────────────────┘
        │  引擎拿到整张图后能做：
        ▼
   ① 算子融合：MatMul+Add+Relu 合成一个 kernel，少读写显存
   ② 跨设备调度：自动把某些节点放 GPU、某些放 CPU/TPU
   ③ 死代码消除、常量折叠、内存复用规划
   ④ 序列化：图能存成文件，脱离 Python 在 C++/手机/浏览器里跑
```

第 ④ 点尤其关键：**图是「可序列化的程序」**。训练用 Python，部署可以完全不要 Python——直接把图文件（`SavedModel`）丢给 C++ 服务、手机、浏览器执行。这就是 TensorFlow 在**工业部署**上的传统强项，也是它和「为科研易用而生」的 PyTorch 的分水岭。

> 一句话记忆：**PyTorch 把「框架」当成「会自动求导的 NumPy」；TensorFlow 把「框架」当成「一门会编译优化的张量计算语言」。**

## 2. 静态图 vs eager：两种执行模式的根本区别

这是理解 TensorFlow 历史和现状的核心，必须讲透。

### 2.1 静态图（TF 1.x，"define-and-run"）

老 TensorFlow 1.x 是**纯静态图**：你写的 Python 代码**不计算任何数字**，只是在「搭建图」；搭完后要开一个 `Session`，把数据「喂（feed）」进去，引擎才真正执行。

```
   TF 1.x 的两段式 (先定义，后执行)

   阶段一：建图 (此时不出任何数字!)
   ┌──────────────────────────────────────┐
   │ x = placeholder()   # 占位符，先没数据│
   │ W = Variable(...)                      │
   │ y = matmul(x, W)    # 只是往图里加节点│
   └──────────────────────────────────────┘
                  │
   阶段二：开 Session 喂数据，才真正算
   ┌──────────────────────────────────────┐
   │ sess.run(y, feed_dict={x: 真实数据})  │ ──▶ 现在才出数字
   └──────────────────────────────────────┘
```

**好处**：图建好后可整体优化、可序列化、可跨设备——性能与部署友好。
**致命缺点**：**写起来反直觉、极难调试**。你不能直接 `print(y)` 看中间结果（`y` 只是个图节点不是数字）；想要 `if`/`for` 控制流得用 `tf.cond`/`tf.while_loop` 这种「图内控制流」算子，而不是普通 Python；报错信息指向图的内部，离你的代码很远。这正是当年大量研究者「逃向 PyTorch」的直接原因。

### 2.2 eager 模式（TF 2.x 默认，"define-by-run"）

TensorFlow 2.0（2019）把**默认行为彻底改成 eager（即时执行）**，向 PyTorch 的动态图看齐：写一行、算一行、立刻出数字，可以 `print`、可以用原生 Python 的 `if/for`、可以单步调试。

```
   TF 2.x eager (边写边算，和 PyTorch 一样直观)
   ┌──────────────────────────────────────┐
   │ x = tf.constant([[1.,2.]])            │
   │ W = tf.Variable([[3.],[4.]])          │
   │ y = tf.matmul(x, W)   # 立刻就算!     │
   │ print(y)  # → [[11.]]  真的能打印!    │
   └──────────────────────────────────────┘
```

**但纯 eager 丢掉了静态图的性能/部署优势。** 怎么办？TF 2 的答案是第 3 节的 `tf.function`——**让你用 eager 写代码，需要时再一键转成静态图**，鱼和熊掌兼得。这是 TF 2 设计的精髓。

```
   两种模式的取舍一图流
                     易调试/灵活      性能/可部署
   TF 1.x 静态图        ✗ 差            ✓ 强
   TF 2 纯 eager        ✓ 强            ✗ 弱
   TF 2 + tf.function   ✓ 写时强        ✓ 跑时强   ← 兼得
```

## 3. tf.function / AutoGraph：eager 写、静态图跑

这是 TF 2 最聪明、也最该理解的机制。

### 3.1 它做了什么

给一个普通的 Python 函数加上 `@tf.function` 装饰器，TensorFlow 会在**第一次调用时**把这个函数「追踪（trace）」一遍：跑一次记录下所有 TF 运算，**编译成一张静态图**；之后每次调用直接执行这张图，不再走 Python 解释器。

```
   @tf.function
   def train_step(x, y):
       ...               # 你照常用 eager 风格写

   第一次调用 train_step(x, y)：
   ┌─────────────────────────────────────────────┐
   │ ① Trace：执行一遍，记录所有 tf 运算         │
   │ ② AutoGraph：把 Python 的 if/for/while       │
   │    自动改写成图内控制流 (tf.cond/while_loop) │
   │ ③ 生成并优化静态图 (可走 XLA 编译)           │
   └─────────────────────────────────────────────┘
            │  缓存这张图 (按输入签名)
            ▼
   第二次起：直接执行已编译的图 ── 快，无 Python 开销
```

**AutoGraph** 是其中的魔法：它把你写的普通 `for`/`if`（依赖张量值的那种）**源码级改写**成 `tf.while_loop`/`tf.cond`，于是图里也能表达控制流，而你不用手写那些反人类的图算子。

### 3.2 关键收益与陷阱

- **收益**：消除 Python 逐行解释开销、允许算子融合、可被 XLA 进一步编译、可序列化部署。和 PyTorch 的 `torch.compile` 是**同一类思想**（捕获图→编译加速），只是 TF 因为图根基深，这条路更「原生」。
- **陷阱——retrace（重复追踪）**：图是**按输入签名缓存**的。如果每次传入不同 shape/dtype、或传 Python 原生数（而非 tf.Tensor），会触发**重新追踪**，反而变慢。所以要尽量传张量、固定 shape。
- **陷阱——副作用只在 trace 时执行一次**：函数里的 `print()`(Python 的) 只在追踪时打印一次；想在每次执行都打印要用 `tf.print`。这是 trace 语义的直接后果。

```
   retrace 直觉：
   train_step(shape=[32,10]) → 编译图A，缓存
   train_step(shape=[32,10]) → 命中图A，快
   train_step(shape=[16,10]) → shape 变了 → 重新编译图B (慢一次)
```

## 4. Keras：高层 API，三行搭网络

`tf.keras` 是 TensorFlow 的**官方高层 API**，把「定义层、组装模型、训练循环」封装到极简。它的地位类似 PyTorch 里 `nn.Module` + 训练循环 + 一堆 Lightning 式糖，但更「开箱即用」。

```
   Keras 三种建模风格 (按灵活度递增)

   ① Sequential —— 层像叠积木，一条直线
      model = Sequential([Dense(128, 'relu'), Dense(10)])
      适合：简单前馈、新手最快上手

   ② Functional API —— 把层当函数调用，可分叉/合并
      inp = Input(784); h = Dense(128,'relu')(inp); out = Dense(10)(h)
      model = Model(inp, out)
      适合：多输入多输出、残差/分支结构

   ③ Subclassing —— 继承 Model，自己写 call()，像 PyTorch
      class Net(Model):
          def call(self, x): ...
      适合：最大灵活度、动态结构、研究
```

训练闭环被压缩成 `compile` + `fit`：

```
   model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
   model.fit(x_train, y_train, epochs=5, batch_size=32)
   # 内部自动完成：前向→算loss→GradientTape反向→optimizer更新→指标统计
   #              还自动处理 batch 切分、进度条、验证集评估
```

```
   compile/fit 在背后替你做的事 (对应第1节的循环)
   ┌─────────────────────────────────────────────┐
   │ for epoch:                                    │
   │   for batch in 自动切分的数据:                │
   │     with GradientTape: y_hat=model(x); loss   │  ② 前向+建带
   │     grads = tape.gradient(loss, 参数)         │  ③ 反向
   │     optimizer.apply_gradients(...)            │  ④ 更新
   │   打印进度/指标，跑验证集                     │
   └─────────────────────────────────────────────┘
```

> 历史补充：Keras 原是一个**独立的高层库**（François Chollet 作），能接 TensorFlow/Theano/CNTK 等多个后端。后来被 TensorFlow 深度整合为 `tf.keras` 成为「官方门面」。Keras 3（2023+）又回归**多后端**（可跑在 TF / JAX / PyTorch 之上），定位为跨框架的高层 API。具体版本/能力**以官方文档为准**。

## 5. 自动微分 GradientTape：反向传播在 TF 里怎么算

eager 模式下没有「先建好的图」可逆向遍历，TF 用 **`tf.GradientTape`（梯度带）** 来记录前向、再回放算梯度。「带」这个比喻很贴切：它像一盘磁带，**录下你在它作用域里做的运算**，之后能「倒带」算导数。

```
   x = tf.Variable(3.0)
   with tf.GradientTape() as tape:   # 开始"录音"
       y = x * x                     # 记录: y = x²
   dy_dx = tape.gradient(y, x)       # 倒带求导: dy/dx = 2x = 6.0
```

```
   GradientTape 的工作流
   ┌──────────────────────────────────────────────┐
   │  进入 with：tape 开始监视                      │
   │     y = x*x   → 磁带记下这步运算 + 局部导数    │
   │  退出 with：录制结束                           │
   │  tape.gradient(y, x)：                          │
   │     沿记录逆向套链式法则 → ∂y/∂x               │
   └──────────────────────────────────────────────┘
```

几个必懂细节（和 PyTorch autograd 一一对应）：

| 概念 | TensorFlow | PyTorch 对应 | 含义 |
|------|-----------|--------------|------|
| 哪些张量自动被追踪 | `tf.Variable` 自动追，常量需 `tape.watch(t)` | `requires_grad=True` | 只追要算梯度的 |
| 默认用一次就丢 | 默认 `gradient()` 调一次磁带就释放 | backward 后图释放 | 省显存 |
| 要多次求导 | `GradientTape(persistent=True)` | `retain_graph=True` | 保留记录 |
| 关闭追踪 | 在 tape 外 / `stop_gradient` | `torch.no_grad()` | 推理时省显存 |

> 注意：在 `@tf.function` 里用 GradientTape，整个前向+反向会被一起追踪成静态图——这就是 Keras `fit` 高效的底层原因（eager 写法 + 静态图执行）。

## 6. 分布式策略 Strategy：多卡/多机的统一接口

TensorFlow 把「单机怎么写的代码，几乎不改就能扩到多卡/多机」抽象成 **`tf.distribute.Strategy`**。核心思路：**你照常写模型和训练，只把它们放进一个 strategy 的作用域里，由 strategy 负责变量放哪、梯度怎么同步。**

```
   strategy = tf.distribute.MirroredStrategy()   # 选一种策略
   with strategy.scope():
       model = build_model()      # 变量自动在各卡建副本
       model.compile(...)
   model.fit(...)                 # 训练自动分布式
```

主要策略（按场景）：

| 策略 | 场景与机制 | PyTorch 类比 |
|------|-----------|--------------|
| `MirroredStrategy` | 单机多卡。每卡完整镜像，梯度 All-Reduce 同步 | DDP（单机） |
| `MultiWorkerMirroredStrategy` | 多机多卡。镜像 + 跨机 All-Reduce | DDP（多机） |
| `TPUStrategy` | 在 TPU Pod 上分布式（见第 7 节） | — |
| `ParameterServerStrategy` | 异步参数服务器：worker 推梯度给 ps 汇总更新，适合超大稀疏/推荐 | — |

MirroredStrategy 的数据并行图（和 PyTorch DDP 同构）：

```
   GPU0      GPU1      GPU2      GPU3
   模型镜像  模型镜像  模型镜像  模型镜像
   数据片A   数据片B   数据片C   数据片D
     │         │         │         │
   各自前向+GradientTape反向 → 梯度 g0..g3
     └──── All-Reduce（求和/平均后广播回每卡）────┘
   各自 apply_gradients —— 镜像变量保持一致
```

**和 PyTorch 的差异**：PyTorch 用 DDP/FSDP 两套类，FSDP/ZeRO 式参数分片是大模型训练主流；TF 的 Strategy 接口更「声明式统一」，**参数服务器**这条线在 Google 内部推荐系统等超大稀疏场景历史更深。极大模型的纯参数分片，PyTorch 生态（DeepSpeed/FSDP，见 [[ai-framework/pytorch/README]]）当前更主流、工具更全。

## 7. TPU + XLA：专用芯片为什么快、怎么用

### 7.1 XLA：图编译器

**XLA（Accelerated Linear Algebra）** 是 TensorFlow（也被 JAX 用）的**线性代数专用编译器**。它接过计算图，做**算子融合**和**为目标硬件生成专用机器码**，是「图编译加速」的引擎。

```
   计算图 ──▶ ┌──────────┐ ──▶ 针对 CPU/GPU/TPU 的高效机器码
              │   XLA     │
              │ 融合算子   │   例：a*b 再 +c 再 relu，
              │ 内存规划   │   本来3个kernel/3次显存往返，
              │ 生成代码   │   融合成1个kernel/1次往返
              └──────────┘
```

`tf.function(jit_compile=True)` 即开启 XLA。它和 PyTorch 的 Inductor、JAX 的默认 jit 是同一生态位的东西。

### 7.2 TPU：为矩阵乘而生的芯片

**TPU（Tensor Processing Unit）** 是 Google 自研的深度学习专用芯片（ASIC）。GPU 本是图形通用并行处理器，TPU 则**专门为神经网络的核心运算——矩阵乘/卷积——设计**，核心是一个巨大的 **脉动阵列（systolic array）**。

```
   脉动阵列 (Systolic Array) 直觉：
   把矩阵乘 C = A×B 摊到一个 N×N 的乘加单元(MAC)网格上，
   数据像"心脏泵血"一样在网格里流动、边流边乘加，
   一次加载、多步复用，省去反复读写寄存器/显存的开销。

        b b b b          ← 权重/B 列从上方注入
        │ │ │ │
   a ─▶ ⊞─⊞─⊞─⊞ ─▶      每个 ⊞ = 一次乘加(MAC)
   a ─▶ ⊞─⊞─⊞─⊞ ─▶      数据横向流入、纵向累加
   a ─▶ ⊞─⊞─⊞─⊞ ─▶      → 极高的算力/功耗比
   a ─▶ ⊞─⊞─⊞─⊞ ─▶
```

**为什么 TPU 快又省电？** 普通处理器算矩阵乘要反复「取数→乘→写回」，瓶颈在访存。脉动阵列让数据在阵列里**流动复用**，一个数读进来后被很多个乘加单元用到，**大幅减少访存次数**，于是同样功耗下算力远高。

**TPU Pod**：把成千上万块 TPU 用高速专用互联连成一个巨型集群（`单 TPU 芯片 ──► Board ──► Pod`），配合 `TPUStrategy` 把训练摊到整个 Pod，是 Google 训练超大模型的基础设施。

> 数字提示：各代 TPU 的算力/显存（HBM）规格请**以 Google Cloud 官方文档为准**，此处不写具体精确值以免过时。直觉上「TPU 在大 batch、规整的矩阵密集型负载上性价比突出，对动态/不规则计算不如 GPU 灵活」。

## 8. 部署生态：训练之外的半壁江山

TensorFlow 的真正护城河之一是**端到端部署生态**——图可序列化带来的红利。

```
   训练好的模型 (SavedModel / 一张可序列化的图)
        ├─► TF Serving : 高性能 C++ 服务，gRPC/REST 上线，
        │               热更新/多版本/批处理 (服务器端推理)
        ├─► TF Lite    : 量化成手机/嵌入式/IoT 小模型 (端侧推理)
        ├─► TF.js      : 在浏览器/Node.js 里直接跑 (前端推理)
        └─► TFX        : 生产级 ML 流水线 (校验→训练→评估→上线)
```

这套「从训练到 serving 到端侧到流水线」的全家桶，是 TF 在**工业生产环境**长期被采用的根本原因：模型脱离 Python、跨语言跨设备运行的能力，是它从 1.x 静态图时代就刻进基因的优势。

## 9. 历史地位与现状

```
   时间线（约，以官方资料为准）
   2015 ── TF 1.0 开源。纯静态图，Google 背书，迅速成为
           工业界与学界事实标准（取代 Theano/Caffe）。
   2016-18 ─ 王座期。但静态图难用，PyTorch(2016)以动态图
            在科研圈快速蚕食份额。
   2019 ── TF 2.0：默认 eager + tf.function + 主推 tf.keras，
           一次大转向，试图补回易用性。
   2020+ ─ 科研主流明显倒向 PyTorch（论文实现/HuggingFace
           生态几乎以 PyTorch 为底座）；Google 内部新研究
           大量转向 JAX。
   现状 ── TF 在"已有工业部署 + 端侧 + TPU 生态"里仍稳固，
          但大模型时代的科研与开源前沿，PyTorch 是主场。
```

**怎么客观看待 TF 的「衰落」？** 它不是技术差，而是**易用性起步太晚**叠加**生态正反馈**：研究者图易用先选 PyTorch → 论文代码是 PyTorch → 复现/二次开发都用 PyTorch → HuggingFace/DeepSpeed/vLLM 等围绕 PyTorch 建 → 工业界为对接前沿也转向 PyTorch。这是典型的**网络效应**，与单点技术优劣关系不大。

**但 TF 远未出局**：海量存量工业系统、移动端（TF Lite）、浏览器（TF.js）、Google 云上的 TPU 训练、推荐系统（参数服务器）等场景，它依然是稳健选择。同时 Google 把前沿研究的赌注分到了 **JAX**（函数式 + XLA，见 [[ai-framework/README]] 下 jax）。

## 数值例子 / 对照

### 例 1：tf.function 为什么能提速（手算直觉）

设一个 step 里有 100 个小算子，eager 模式下每个算子要过一次 Python 解释器 + 启动一次 GPU kernel。假设单个算子 Python 调度约 $5\,\mu s$、kernel 启动约 $5\,\mu s$，纯开销 $\approx 100 \times 10\,\mu s = 1\,\text{ms}$。若真实计算本身也只要 $1\,\text{ms}$，那**一半时间浪费在调度上**。

`tf.function` 把 100 个算子编成一张图、经 XLA 融合成（比如）20 个 kernel：调度/启动开销 $\approx 20 \times 5\,\mu s = 0.1\,\text{ms}$，且融合减少显存往返让计算也更快。**小算子越多、batch 越小，编译收益越大**——这与 PyTorch `torch.compile` 的收益来源完全一致。

### 例 2：MirroredStrategy 的梯度通信（手算）

一个 $P = 1$ 亿参数模型（100M），用 4 卡 MirroredStrategy，梯度 FP32 共 $4P = 0.4\,\text{GB}$。每步做一次 Ring All-Reduce，每卡收发约 $2 \times \frac{N-1}{N} \times 数据量 = 2 \times \frac{3}{4} \times 0.4 \approx 0.6\,\text{GB}$/卡/步。

直觉同 DDP：**通信量正比于模型大小、与卡数关系不大**；模型越大越吃机间带宽，所以 TPU Pod / NVLink / InfiniBand 这类高速互联才关键。

### 例 3：TPU 脉动阵列省了多少访存（直觉）

朴素算 $N\times N$ 矩阵乘 $C=A\times B$ 共 $N^3$ 次乘加，若每次乘加都从内存取两个操作数，访存约 $2N^3$。脉动阵列里一个元素**加载一次后在阵列中被复用 $\approx N$ 次**，访存量降到约 $O(N^2)$ 量级。$N$ 越大，访存节省比例越高——这就是 TPU 在大矩阵密集负载下「算力/功耗比」碾压的来源。具体倍数随实现/数据流而变，**以官方白皮书为准**。

### 对照表（TensorFlow vs PyTorch vs JAX）

| 维度 | TensorFlow 1.x | TensorFlow 2 / Keras | PyTorch | JAX |
|------|----------------|----------------------|---------|-----|
| 默认计算图 | 静态(define-and-run) | eager + `@tf.function` 转静态 | 动态(define-by-run) | 函数式 + `jit` 编译 |
| 调试体验 | 差（图与执行分离） | 较好 | 最好（纯 Python） | 中（函数纯度约束） |
| 上手难度 | 高 | 中 | 低 | 中高（函数式思维） |
| 自动微分 | 图反向 | `GradientTape` | `autograd` | `grad`（函数变换） |
| 图编译/加速 | 天生静态图 | `tf.function`/XLA | `torch.compile`(Inductor) | XLA（默认强） |
| 专用硬件 | TPU 原生 + GPU | TPU 原生 + GPU | GPU 为主，TPU 较弱 | TPU/GPU 皆强 |
| 分布式 | 较繁琐 | `tf.distribute.Strategy` | DDP/FSDP 生态成熟 | `pmap`/`shard_map` |
| 部署生态 | ★ Serving/Lite/JS/TFX | ★ 同左，最完整 | 改善中(TorchServe/ExecuTorch) | 较弱 |
| 科研主流 | 已淘汰 | 工业仍有 | ★ 绝对主流 | 增长中(数值/RL/Google) |
| 大模型开源生态 | 弱 | 弱 | ★ HF/DeepSpeed 全围绕它 | 中（部分实验室） |

**何时选 TensorFlow？**
1. **已有 TF 存量系统**、团队熟悉、要继续维护演进。
2. **重端侧/浏览器部署**：TF Lite / TF.js 成熟度高。
3. **要用 TPU**：TF 是 TPU 的一等公民（JAX 同样强）。
4. **生产级 ML 流水线**：TFX 体系完整。
若是**从零做大模型科研/训练/对齐/推理**，当前默认选 PyTorch（生态与社区压倒性）。

## 常见问题

| 问题 | 答案 |
|------|------|
| 静态图和动态图到底差在哪？ | 静态图先把整张计算图定义完再喂数据执行（利于全局优化/序列化部署，但难调试）；动态图边运行边建（直观可调试，控制流随意）。TF 1.x 纯静态、PyTorch 纯动态、TF 2 用 `tf.function` 在二者间切换。 |
| TF 2 还需要 Session 吗？ | 不需要。`Session`/`placeholder`/`feed_dict` 是 TF 1.x 静态图的产物，TF 2 默认 eager 已弃用这套（`tf.compat.v1` 里仍可访问）。 |
| `tf.function` 一定更快吗？ | 不一定。有首次编译开销和 retrace 风险（shape/dtype 频繁变化会反复重编译）。适合稳定形状的训练/服务，一次性小脚本未必划算——与 `torch.compile` 同理。 |
| Keras 和 TensorFlow 是一回事吗？ | 不是。Keras 是高层 API，`tf.keras` 是它整合进 TF 的版本；Keras 3 又支持多后端（TF/JAX/PyTorch）。TF 是底层引擎，Keras 是门面。 |
| GradientTape 用一次就没了？ | 默认是，调一次 `gradient()` 后磁带释放（省显存）。要多次求导用 `persistent=True`，记得手动 `del tape`。 |
| 为什么 PyTorch 反超了 TF？ | 主要是易用性起步晚 + 生态网络效应：科研先选 PyTorch → 论文/开源/工具链都围绕它 → 工业为对接前沿也转向它。非单点技术优劣。 |
| TPU 一定比 GPU 快吗？ | 在大 batch、规整矩阵密集负载上性价比突出；对小 batch、动态/不规则计算不如 GPU 灵活。要看负载形状，具体规格以官方为准。 |
| 现在还该学 TF 吗？ | 看场景：做端侧/浏览器部署、用 TPU、维护存量工业系统值得；纯大模型科研优先 PyTorch。两者核心概念（张量/自动微分/数据并行/图编译）相通，学透一个迁移很快。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局知识地图，从这里找其他主题
- [[ai-framework/pytorch/README]] — 动态图阵营的对照参考：autograd / DDP / FSDP / torch.compile
- [[ai-framework/README]] — AI 框架总览：TensorFlow / PyTorch / JAX / 各类训练推理框架的全景定位
