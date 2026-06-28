# JAX

> JAX 是用「函数式编程 + 可组合的程序变换（grad/jit/vmap/pmap）+ XLA 编译」重新设计的数值计算框架：你写一个普通的纯函数，它能自动帮你求导、编译加速、批量化、跨多卡并行——而且这些能力可以像乐高一样任意叠加。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/pytorch/README]] [[ai-framework/README]]

> Jax 是我看过那么多项目中，唯一一个让我看了之后觉得「哇，软件还可以这么写，一切都很有道理」的项目。Google 把很多 TensorFlow 的经验都吸取进了 JAX 里。这份笔记从最底层把这种「有道理」讲清楚。

## 阅读地图

| 节 | 主题 | 你会得到什么 | 难度 |
|----|------|--------------|------|
| 0 | 一句话锚点 | JAX 的本质定位 | ★ |
| 1 | 地基/前置 | 它解决什么问题、为什么要函数式 | ★★ |
| 2 | NumPy 同款 API | `jax.numpy` 与不可变数组 | ★★ |
| 3 | `grad`：自动微分 | 一个纯函数怎么变出梯度函数 | ★★★ |
| 4 | `jit` 与 XLA | 追踪 → 中间表示 → 编译融合 | ★★★★ |
| 5 | `vmap`：自动向量化 | 不写循环就能批量化 | ★★★ |
| 6 | `pmap`/`shard_map`：多设备并行 | 一行把计算铺到多卡/TPU | ★★★★ |
| 7 | 可组合：变换叠变换 | 为什么科研友好 | ★★★ |
| 8 | PyTree 与函数式状态 | 参数/优化器状态怎么传 | ★★★ |
| 9 | Flax / Haiku | 在函数式之上怎么写网络 | ★★ |
| 10 | TPU 亲和 | 为什么 JAX 在 TPU 上如鱼得水 | ★★★ |
| 数值例子 | 手算 `grad`/`vmap` | 验证变换确实对 | ★★★ |
| 对照表 | JAX vs PyTorch | 怎么选 | ★ |
| - | FAQ / 跳转 | 查漏与导航 | ★ |

---

## 0. 一句话锚点

**JAX = 一个会自动求导、能被 XLA 编译、能自动批量化和并行化的「NumPy」。**

它和 PyTorch 最大的哲学差异，一句话：

- **PyTorch**：面向对象 + 有状态。模型是对象（`nn.Module`），参数藏在对象里，`loss.backward()` 把梯度「填」回参数的 `.grad` 上——副作用驱动。
- **JAX**：函数式 + 无状态。一切都是纯函数 `输出 = f(参数, 输入)`，没有隐藏状态。求导、编译、并行都是「把一个函数变成另一个函数」的**程序变换（transformation）**。

```
   PyTorch 思维：  对象.方法()  改变对象内部状态（副作用）
   JAX 思维：     新函数 = 变换(旧函数)   函数进，函数出，无副作用
```

JAX 的灵魂是 4 个**可组合的变换**：

| 变换 | 作用 | 类比 |
|------|------|------|
| `grad` | 求导：`f` → `∂f/∂x` | 自动微分 |
| `jit`  | 编译：`f` → 编译加速版 `f` | XLA 编译 |
| `vmap` | 向量化：`f(单样本)` → `f(一批样本)` | 自动 batch |
| `pmap` | 并行：`f` → 跨多设备并行的 `f` | 多卡/TPU |

> 注：题面写的 `pmax` 应为 **`pmap`**（parallel map）。`pmax` 不是 JAX 变换；`jax.lax.pmax` 只是 `pmap` 内部做跨设备求最大值的归约算子。下文统一用 `pmap`。

## 1. 地基/前置：为什么要"函数式"

回忆训练一个网络的核心循环（和 PyTorch 一节里画的是同一个）：

```
   ┌──────────────────────────────────────────────────────┐
   │  1. 前向：  ŷ = f(params, x)                          │
   │  2. 损失：  loss = L(ŷ, y)                            │
   │  3. 反向：  grads = ∂loss/∂params                     │
   │  4. 更新：  params ← params - lr × grads              │
   │  5. 回到 1                                            │
   └──────────────────────────────────────────────────────┘
```

PyTorch 把第 1~4 步用「对象 + 副作用」串起来：`params` 藏在 `model` 里，`backward()` 偷偷把 `grads` 写到 `.grad`，`optimizer.step()` 偷偷改 `params`。方便，但「谁改了什么」不透明。

**JAX 的赌注：如果把所有计算都写成纯函数（同样的输入永远给同样的输出、不改外部状态），那么"求导/编译/并行"就有了干净的数学定义，可以被自动实现、还能自由组合。**

什么叫纯函数（pure function）：

```
   纯：  def f(x): return x*2          # 只依赖入参，无副作用 ✅
   不纯：count = 0
        def f(x):
            global count; count += 1   # 改了外部状态 ✗
            return x*2
```

为什么纯函数对编译/求导是天大的好处：

| 痛点 | 函数式的回答 |
|------|--------------|
| 编译器不敢优化（怕副作用） | 纯函数无副作用 → 可大胆重排/融合/缓存 |
| 求导要追踪隐藏状态 | 状态都在入参里 → 求导只是数学变换 |
| 并行有数据竞争 | 无共享可变状态 → 天然可并行 |
| 随机数难复现 | 把随机性变成"显式 key 入参"（见 §8） |

代价：**你必须把状态显式地传进传出**（参数、优化器动量、随机 key 都得手动当参数传）。这就是 JAX「啰嗦但透明」的来源。

## 2. `jax.numpy`：同款 API，但数组不可变

JAX 的张量库几乎就是 NumPy 的克隆：

```python
import jax.numpy as jnp
x = jnp.array([1., 2., 3.])
y = jnp.sin(x) @ jnp.ones((3,))      # 和 numpy 写法一模一样
```

关键差异：**JAX 数组不可变（immutable）**。`x[0] = 5` 会报错。要"改"得用函数式写法返回新数组：

```python
x = x.at[0].set(5.0)     # 返回一个新数组，原 x 不变
```

```
   NumPy:  x[0] = 5        就地改（mutate）
   JAX:    x = x.at[0].set(5)   返回新数组（functional update）
                │
                └─ 编译器看到"无就地改" → 能放心地把整段融合成一个 kernel
```

另外 JAX 默认在加速器（GPU/TPU）上、默认 FP32（`x32`）；想要 FP64 需手动开 `jax.config.update("jax_enable_x64", True)`。

## 3. `grad`：把一个函数变成它的梯度函数

这是 JAX 最优雅的地方。求导不是「方法」，而是「函数 → 函数」的变换：

```python
import jax
def f(x):           # f(x) = x^2 + 3x
    return x**2 + 3*x

df = jax.grad(f)    # df 是一个新函数，等于 f'(x) = 2x + 3
df(2.0)             # = 7.0
```

```
        ┌─────────┐    jax.grad     ┌──────────────┐
   f(x) │  x²+3x  │  ───────────▶   │ f'(x)= 2x+3  │  df(x)
        └─────────┘   （变换）       └──────────────┘
```

它怎么做到的？**追踪（tracing）+ 反向模式自动微分**：JAX 用一个抽象的「追踪器」对象当 `x` 喂进 `f`，记录下所有原语运算（加、乘、sin…）形成一张计算图（jaxpr），再对这张图套链式法则反向传播。

训练里最常用的组合是 **同时拿到值和梯度**：

```python
loss_value, grads = jax.value_and_grad(loss_fn)(params, batch)
```

进阶变换（都可组合）：

| 变换 | 含义 |
|------|------|
| `jax.grad(f)` | 反向模式一阶梯度（标量输出） |
| `jax.jacfwd` / `jax.jacrev` | 前向/反向模式雅可比矩阵 |
| `jax.hessian` | 二阶（= `jacfwd(jacrev(f))`） |
| `jax.jvp` / `jax.vjp` | 雅可比-向量积 / 向量-雅可比积（自微分的两块积木） |

## 4. `jit` 与 XLA：从 Python 到融合 kernel

`jax.jit` 把一个 Python 函数**编译**成一段高度优化的机器码。底层引擎是 **XLA（Accelerated Linear Algebra）**——Google 给 TensorFlow 写的线性代数编译器，JAX 直接复用。

它的两段式工作流：**追踪（trace）→ 编译（compile）**。

```
  Python 函数 f
       │  ① 用抽象形状追踪（不看具体数值，只看 shape/dtype）
       ▼
   jaxpr（JAX 中间表示：一串原语运算的 DAG）
       │  ② 交给 XLA
       ▼
   XLA HLO（高层算子图）
       │  ③ 优化：算子融合 / 内存布局 / 常量折叠 / 死代码消除
       ▼
   针对 CPU/GPU/TPU 的机器码（编译产物会被缓存）
       │  ④ 之后每次调用直接跑机器码，不再走 Python
       ▼
     结果
```

**最大收益是「算子融合（fusion）」**：

```
  没编译： sin → tmp → mul → tmp → add        三个 kernel，三趟读写显存
  XLA 融合：  ┌──────────────────────────┐
             │ 一个 kernel 里算完 sin·mul·add │   一趟读写，省带宽
             └──────────────────────────┘
```

用法就是套一层：

```python
@jax.jit
def step(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    params = jax.tree.map(lambda p, g: p - 0.1*g, params, grads)
    return params, loss
```

**必须知道的两个坑**：

1. **追踪只发生在"抽象形状"上**：`jit` 看不到具体数值，所以 `if x > 0` 这种**依赖数值**的 Python 控制流会报错——要用 `jax.lax.cond` / `jax.lax.scan`。形状（shape）必须在编译期固定。
2. **重新编译**：每遇到一个**新的输入形状/dtype**，会重新追踪+编译一次（首次调用慢，叫 warm-up）。形状反复变会反复编译，很慢。需要变化的标量用 `static_argnums` 标成「静态」。

> 心智模型：`jit` = 「**第一次调用时把 Python 翻译成一张静态图并编译，之后都跑编译产物**」。和 PyTorch 的 `torch.compile` 目标一致，但 JAX 是「编译优先」原生设计，PyTorch 是「动态优先、后补编译」。

## 5. `vmap`：不写循环的自动批量化

你写好「处理一个样本」的函数，`vmap` 自动把它变成「处理一批样本」的函数——而且是向量化、并行的，不是 Python for 循环。

```python
def predict(params, x):      # x 是单个样本（无 batch 维）
    return params @ x

batched = jax.vmap(predict, in_axes=(None, 0))   # 对 x 的第 0 维做批量
batched(params, X)            # X 形状 (batch, dim)，一次算完整批
```

```
   你只写：  f(单样本)
                │  jax.vmap
                ▼
   自动得到： f(一批样本)   ── 在加速器上并行，无 Python 循环
   in_axes 告诉它哪个参数有 batch 维（None=不批量，0=第0维是batch）
```

为什么这是杀手锏：在 PyTorch 里你得**手动**在每个算子里塞 batch 维、维护广播规则；JAX 让你写「数学上最干净的单样本逻辑」，批量化交给变换。计算 per-sample 梯度（隐私/影响函数里常用）一行搞定：`vmap(grad(loss))`。

## 6. `pmap` / `shard_map`：一行铺到多设备

`pmap`（parallel map）把函数复制到**多个设备**（多 GPU / 多 TPU 核）上，每个设备跑数据的一个分片，需要时用 `jax.lax.psum`/`pmax` 等做**跨设备归约**（这就是 `pmax` 真正出现的地方）。

```
        设备0    设备1    设备2    设备3      ← 8卡/TPU pod 的一个切片
       ┌────┐  ┌────┐  ┌────┐  ┌────┐
  数据 │分片0│  │分片1│  │分片2│  │分片3│    每卡算自己那份梯度
       └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘
         └───────┴── psum / pmax ──┴───────┘   跨卡归约（all-reduce）
                     ▼
              全卡同步的平均梯度（数据并行）
```

```python
@jax.pmap
def train_step(params, batch):
    grads = jax.grad(loss_fn)(params, batch)
    grads = jax.lax.pmean(grads, axis_name='batch')   # 跨设备求平均
    return params - 0.1*grads
```

现代 JAX 更推荐用 **`jax.sharding` + `jit`（GSPMD）** 或 **`shard_map`** 来表达更复杂的并行（张量并行、流水并行、FSDP 式分片）：你只描述「数组怎么切到设备网格（mesh）上」，XLA 的 **GSPMD** 自动插入通信。这套是 JAX 在 TPU pod 上训超大模型（如 Gemini/PaLM）的基石。

## 7. 可组合：变换叠变换 —— 为什么科研友好

JAX 真正的魔法不是 4 个变换本身，而是**它们能任意嵌套组合**，每个组合都恰好对应一个数学概念：

```
   jit(grad(f))             编译后的梯度          ← 训练 step
   vmap(grad(f))            每样本梯度（per-sample）
   grad(grad(f))            二阶导 / Hessian
   pmap(jit(grad(f)))       多卡上的编译梯度
   vmap(vmap(f))            处理"批的批"
   jit(vmap(grad(loss)))    编译 + 批量 + 求导，全叠
```

为什么科研友好（这点是 JAX 被研究者偏爱的核心原因）：

| 科研需求 | JAX 怎么满足 |
|----------|--------------|
| 想要高阶导（元学习/物理仿真） | `grad(grad(...))` 直接叠，无需特殊支持 |
| 想要 per-sample 梯度（隐私/影响函数） | `vmap(grad(...))` 一行 |
| 想试新优化器/新自微分玩法 | 函数式 → 数学怎么写代码就怎么写，无隐藏状态干扰 |
| 想要可复现 | 纯函数 + 显式随机 key → 结果逐位可复现 |
| 想从单卡无痛扩到 TPU pod | 同一份函数，外面套 `pmap`/sharding |

> 一句话：**PyTorch 让"写网络"很顺手，JAX 让"做数学实验"很顺手。** 谁的研究里充满了高阶导、自定义梯度、奇怪的并行模式，谁就会爱上 JAX。

## 8. PyTree 与函数式状态：参数/随机数怎么传

既然无隐藏状态，**所有状态都得当参数传**。JAX 用 **PyTree** 统一处理「嵌套的容器」（dict/list/自定义类里装着数组）：

```python
params = {'w': jnp.zeros((3,3)), 'b': jnp.zeros(3)}   # 这就是一个 PyTree
# tree.map 对树里每个叶子（数组）同时操作
new_params = jax.tree.map(lambda p, g: p - 0.1*g, params, grads)
```

```
   params (PyTree)
   ├─ 'w' → ndarray   ┐
   └─ 'b' → ndarray   ┘ 叶子（leaves）；grad/vmap/jit 都按相同结构作用到每个叶子
```

**随机数也是显式状态**。JAX 不用全局随机种子（那是副作用），而是传一个显式的 `key`，并要求你手动 `split` 出新 key——这样并行/重跑都完全可复现：

```python
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)      # 用一次 split 一次，绝不复用同一个 key
noise = jax.random.normal(subkey, (3,))
```

## 9. Flax / Haiku：在函数式之上写网络

纯函数式手写大网络很啰嗦，所以社区在 JAX 之上加了神经网络库，帮你管理参数这棵 PyTree：

| 库 | 出品方 | 风格 | 现状 |
|----|--------|------|------|
| **Flax**（`flax.linen` / 新版 `nnx`） | Google | 模块化、显式 `apply(params, x)` | JAX 生态**事实主流**，Gemma 等开源模型用它 |
| **Haiku** | DeepMind | 类 Sonnet，`transform` 把有状态写法转成纯函数 | 仍在用，但 DeepMind 也在向 Flax 靠拢 |
| **Equinox** | 社区 | 把模块直接当 PyTree，最"纯函数" | 轻量、研究者喜欢 |
| **Optax** | DeepMind | 优化器库（梯度变换链） | 几乎人人都用，搭配上面任意一个 |

Flax 的关键认知：模块只负责**定义结构**，参数是**外部的 PyTree**，前向是 `model.apply(params, x)`——参数永远在你手上、显式传递，这才符合 JAX 哲学：

```
   model = MLP(...)                      # 只是结构定义，不含参数
   params = model.init(key, x)           # 参数是独立的 PyTree
   y = model.apply(params, x)            # 前向：参数显式传入（纯函数）
                  ▲
                  └─ 于是 grad/jit/vmap 能直接作用在 params 上
```

## 10. TPU 亲和：为什么 JAX 在 TPU 上如鱼得水

JAX 和 XLA、TPU 是同一个 Google 生态长出来的，三者天然咬合：

```
   JAX 函数 ──trace──▶ XLA ──编译──▶ TPU 机器码
                        │
            同一个 XLA 后端也能编 CPU/GPU
```

- **XLA 是 TPU 唯一的编译路径**：TPU 不像 GPU 有海量手写 CUDA kernel，它**只吃 XLA 编译出来的代码**。JAX 原生产出 XLA，所以在 TPU 上是"一等公民"，几乎零摩擦。
- **GSPMD 自动并行**：在 TPU pod（成百上千个芯片，靠高速 ICI 互联成 2D/3D 环面网络）上，你只需用 `jax.sharding` 描述数组怎么切到设备 mesh，XLA 自动插入跨芯片通信。把模型从 8 卡扩到 4096 芯片，代码改动极小。
- **典型公开规格（约/以官方为准）**：TPU v4 单芯片约 275 TFLOPS（bf16），一个 v4 pod 约 4096 芯片、整 pod 约 1.1 EFLOPS 级；芯片间 ICI 带宽远高于普通以太网。Google 的 Gemini、PaLM、T5 等大模型主要在 JAX + TPU 上训练。
- **GPU 也支持**：JAX 在 NVIDIA GPU 上同样跑（XLA-GPU 后端），但生态/kernel 丰富度上 PyTorch 更成熟，调试期工具也更多。

> 选择信号：**有 TPU（尤其大规模 pod）→ JAX 几乎是默认选择；只有 GPU 且要用海量现成库 → PyTorch 通常更省心。**

## 数值例子：手算验证变换

**例 1：`grad` 对不对。** 取 $f(x)=x^2+3x$，理论导数 $f'(x)=2x+3$。在 $x=2$：

$$f'(2) = 2\times 2 + 3 = 7$$

`jax.grad(f)(2.0)` 应返回 `7.0`，与手算一致。再叠一层：`grad(grad(f))(2.0)` 求二阶导，$f''(x)=2$，返回 `2.0`。

**例 2：`vmap` 对不对。** 设单样本函数 `dot(w, x) = w·x`，$w=[1,2,3]$。批量输入

$$X=\begin{bmatrix}1&0&0\\0&1&0\\1&1&1\end{bmatrix}$$

`vmap(dot, in_axes=(None,0))(w, X)` 对每一行算点积：

- 行0 $[1,0,0]\cdot w = 1$
- 行1 $[0,1,0]\cdot w = 2$
- 行2 $[1,1,1]\cdot w = 6$

得 `[1, 2, 6]`——和「写个 for 循环逐行点积」结果完全一致，但 JAX 把它编译成一次矩阵-向量乘，无 Python 循环。

**例 3：`pmap` 数据并行平均梯度。** 4 卡各算出本地梯度 $g_0,g_1,g_2,g_3$，`jax.lax.pmean` 做 all-reduce：

$$\bar g = \frac{g_0+g_1+g_2+g_3}{4}$$

每卡都拿到同一个 $\bar g$ 再各自更新——这就是数据并行的数学本质（和 PyTorch DDP 的 all-reduce 一模一样，只是 JAX 用一行变换表达）。

## 对照表：JAX vs PyTorch

| 维度 | JAX | PyTorch |
|------|-----|---------|
| 编程范式 | 函数式、无状态、纯函数 | 面向对象、有状态、副作用 |
| 计算图 | 追踪后编译（trace-then-compile） | 动态图（define-by-run），`torch.compile` 后补编译 |
| 求导 | `grad` 变换（函数→函数） | `loss.backward()`（写 `.grad`，有副作用） |
| 批量化 | `vmap` 自动向量化 | 手动加 batch 维 |
| 并行 | `pmap` / `sharding`+GSPMD | DDP / FSDP |
| 高阶导/per-sample 梯度 | 天生强（变换任意叠） | 能做但更费劲 |
| 状态（参数/优化器/随机） | 显式传参（啰嗦但透明） | 藏在对象里（方便但隐式） |
| 调试 | 追踪期报错较抽象；`jax.debug.print` | eager 模式，`print`/断点直观 |
| 硬件 | TPU 一等公民，GPU 也行 | GPU 生态最成熟，TPU 较弱 |
| 生态 | 研究/Google 系（Flax/Optax/Gemma） | 工业/社区最大，模型/库最多 |
| 何时选 | 重数学实验、高阶导、TPU 大规模 | 重工程落地、用现成模型/库、GPU |

## 常见问题（FAQ）

| 问题 | 答案 |
|------|------|
| `pmax` 是变换吗？ | 不是。4 个变换是 `grad/jit/vmap/pmap`；`pmax` 是 `pmap`/`lax` 里的跨设备求最大归约算子 |
| 为什么 `if x>0` 在 `jit` 里报错？ | 追踪只看抽象形状不看数值；用 `jax.lax.cond` 表达数值控制流 |
| 为什么数组不能就地改？ | 不可变是函数式前提，换来编译器可大胆融合；用 `x.at[i].set(v)` |
| 首次调用为什么特别慢？ | 那是追踪+XLA 编译（warm-up），之后跑缓存的编译产物 |
| 形状一变就卡？ | 新形状触发重编译；固定形状/padding，或用 `static_argnums` |
| 随机数为什么要传 key？ | 全局种子是副作用、破坏可复现；显式 key + `split` 保证逐位可复现 |
| Flax 和 Haiku 选哪个？ | 新项目优先 Flax（生态主流）；老 DeepMind 代码常见 Haiku |
| JAX 能只用 GPU 吗？ | 能（XLA-GPU 后端），但库/kernel 丰富度不如 PyTorch |
| 等价于 `torch.compile` 吗？ | 思路相近（都编译融合），但 JAX 是编译优先的原生设计 |
| 大模型真有人用 JAX 训吗？ | 有：Gemini、PaLM、T5、Gemma 等主要在 JAX+TPU 上训练 |

## 🔗 跳转链接

- 知识地图总览：[[00-知识地图]]
- 对照：动态图 + autograd 的有状态范式 → [[ai-framework/pytorch/README]]
- 框架全景（底座/并行/微调/推理分工）→ [[ai-framework/README]]
