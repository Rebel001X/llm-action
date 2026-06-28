# PyTorch 训练流程

> 从最底层把一次「前向 → 反向 → 更新」讲透：训练循环就是「让参数沿着梯度反方向走一步」的不断重复。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[ai-framework/pytorch/README]] [[llm-train/README]] [[llm-train/pytorch/distribution/README]]

## 阅读地图

| 节 | 主题 | 你将得到什么 | 何时回看 |
|----|------|-------------|---------|
| 0 | 一句话锚点 | 训练循环的最小骨架 | 忘记顺序时 |
| 1 | 地基：它解决什么问题 | 为什么需要 autograd / optimizer | 入门 |
| 2 | 张量与计算图 | `requires_grad` / 动态图机制 | 报错 "grad" 时 |
| 3 | 标准训练循环五步 | forward/loss/backward/step/zero_grad | 写训练脚本时 |
| 4 | DataLoader 数据流 | Dataset→Sampler→collate→batch | 数据卡顿/OOM 时 |
| 5 | AMP 混合精度 | autocast + GradScaler | 想提速省显存时 |
| 6 | 梯度累积 | 用小卡模拟大 batch | 显存不够时 |
| 7 | Checkpoint | 存/取 state_dict | 断点续训时 |
| 8 | 最小可跑示例思路 | 串起来的完整骨架 | 从零搭建时 |
| 9 | 常见问题 | 踩坑速查 | Debug 时 |

---

## 0. 一句话锚点

PyTorch 训练 = **重复执行五步**：

```
for batch in dataloader:        # 取数据
    optimizer.zero_grad()       # ① 清空上一轮梯度
    out  = model(x)             # ② 前向：算预测
    loss = criterion(out, y)    # ③ 算损失
    loss.backward()             # ④ 反向：算梯度
    optimizer.step()            # ⑤ 更新：参数 -= lr * grad
```

记住这张图，后面所有内容（AMP、梯度累积、分布式）都是**在这五步上打补丁**。

---

## 1. 地基：它解决什么问题

训练一个神经网络，本质是解一个优化问题：找一组参数 $\theta$，让损失 $L(\theta)$ 最小。

梯度下降的更新规则是：

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t)$$

其中 $\eta$ 是学习率，$\nabla_\theta L$ 是损失对每个参数的偏导（梯度）。

**手算这个梯度是不可能的**——一个模型有上亿参数，损失是几十层复合函数。PyTorch 解决两件苦差事：

- **autograd（自动微分）**：你只写前向，框架自动帮你求 $\nabla_\theta L$。
- **optimizer（优化器）**：把「$\theta -= \eta\nabla$」这步以及各种变体（动量、自适应学习率）封装好。

```
你的职责：定义 model / loss / optimizer ── 前向 ──>
PyTorch：记录计算图(前向时) → backward()自动反传梯度 → step()套用更新公式
```

---

## 2. 张量、requires_grad 与动态计算图

### 2.1 张量是一切的载体

`torch.Tensor` 既存数据，也存「是否需要梯度」的标记。模型参数（`nn.Parameter`）默认 `requires_grad=True`，输入数据默认 `False`。

```
Tensor 内部关键字段
  data          实际数值 (如 [0.3, -1.2])
  requires_grad 是否追踪梯度 (True/False)
  grad          反传后累积的梯度
  grad_fn       由哪个运算产生（连成计算图）
```

### 2.2 动态图：边算边建

PyTorch 是**动态图（define-by-run）**：每次前向都**重新构建**一张计算图，前向走到哪图就建到哪。这就是为什么 PyTorch 里能直接写 `if`/`for`、能 `print` 中间张量——图是 Python 代码运行出来的。

```
前向 y = (w*x + b).relu()  时，自底向上记录：

   x ──┐
        ├─[mul]──┐
   w ──┘          ├─[add]──[relu]── loss
   b ─────────────┘
                     每个节点记住 grad_fn，
                     backward() 时沿箭头反向走（链式法则）
```

### 2.3 链式法则就是反向传播

反向传播 = 对计算图从输出到输入应用链式法则。例如 $L = f(g(\theta))$：

$$\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial g}\cdot\frac{\partial g}{\partial \theta}$$

`loss.backward()` 帮你把整张图的链式乘法做完，结果累加进每个叶子张量的 `.grad`。**注意是「累加」**，这正是后面梯度累积能成立、也是必须 `zero_grad()` 的根源。

---

## 3. 标准训练循环：五步逐字拆解

```
┌──────────────────────────── 一个 step ────────────────────────────┐
│                                                                    │
│  ① optimizer.zero_grad()   把 param.grad 清零（否则会累加上一轮）   │
│           │                                                        │
│  ② output = model(x)       前向：构建计算图，得到预测              │
│           │                                                        │
│  ③ loss = criterion(out,y) 把预测和标签压成一个标量损失           │
│           │                                                        │
│  ④ loss.backward()         反向：沿图算梯度，写入各 param.grad     │
│           │                                                        │
│  ⑤ optimizer.step()        用 grad 更新参数（θ -= lr*grad …）      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 为什么是这个顺序？每一步「为什么」

| 步 | 调用 | 为什么必须 | 漏了会怎样 |
|----|------|-----------|-----------|
| ① | `zero_grad()` | `.grad` 是累加的，要先清零 | 梯度叠加多轮，更新爆炸/乱掉 |
| ② | `model(x)` | 建图 + 得预测 | 无图可反传 |
| ③ | `criterion` | 反传必须从**标量**出发 | `backward()` 对非标量报错 |
| ④ | `backward()` | 真正算梯度 | `.grad` 全是 None |
| ⑤ | `step()` | 套用更新公式 | 参数永远不变，loss 不降 |

### 数值例子（单参数线性回归）

设 $y=wx$，真值 $w^*=2$；当前 $w=1$，样本 $x=3,\ y=6$，用 MSE：

- 前向：$\hat y = w x = 1\times3 = 3$
- 损失：$L=(\hat y - y)^2=(3-6)^2=9$
- 反向：$\dfrac{\partial L}{\partial w}=2(\hat y-y)\cdot x = 2(3-6)\cdot3 = -18$
- 更新（$\eta=0.01$）：$w \leftarrow 1 - 0.01\times(-18) = 1.18$

$w$ 从 1 朝 2 移动了一点——这就是「学习」。重复几百步即可逼近 2。

### 训练 / 评估两种模式

- `model.train()`：开启 Dropout、让 BatchNorm 用 batch 统计（训练时建图，要反传）。
- `model.eval()`：关闭 Dropout、BatchNorm 用滑动均值/方差。
- 评估时还要套 `with torch.no_grad():` —— **不建图、不存中间激活**（不反传，只看结果），显存和速度都更省。

---

## 4. DataLoader：数据怎么流进训练循环

模型要吃 **batch**，而原始数据往往散落在磁盘。`DataLoader` 负责把 `Dataset` 组织成一个个 batch 并（可选）多进程预取。

```
磁盘/内存里的样本
      │
 ┌────▼─────┐   __getitem__(i) 取第 i 条
 │ Dataset  │
 └────┬─────┘
      │   Sampler 决定「取哪些索引、什么顺序」(shuffle 在这)
 ┌────▼─────┐
 │ Sampler  │──> [3, 17, 5, ...]  一个 batch 的索引
 └────┬─────┘
      │   num_workers 个子进程并行调用 __getitem__
 ┌────▼──────────┐
 │ collate_fn    │  把 N 条样本拼成一个 batch 张量 (含 padding)
 └────┬──────────┘
      │   pin_memory 把 batch 钉在锁页内存，加速 CPU→GPU 拷贝
      ▼
   一个 batch  ──> 进入训练循环的 for
```

### Dataset 的两类

- **Map-style**：实现 `__getitem__` + `__len__`，支持随机索引（最常用）。
- **Iterable-style**：实现 `__iter__`，适合流式/超大数据集（无法随机访问）。

### 关键参数与权衡

| 参数 | 含义 | 调大的好处 | 调大的代价 |
|------|------|-----------|-----------|
| `batch_size` | 每批样本数 | 吞吐高、梯度稳 | 显存涨、泛化可能略降 |
| `shuffle` | 每轮打乱顺序 | 防止学到顺序偏置 | 训练集才开，验证集别开 |
| `num_workers` | 加载子进程数 | 喂数据更快，GPU 不挨饿 | 内存/CPU 占用上升 |
| `pin_memory` | 锁页内存 | `to(cuda)` 更快（可异步） | 多占一点内存 |
| `drop_last` | 丢掉凑不满的尾批 | BN/分布式更稳 | 浪费少量样本 |
| `collate_fn` | 自定义拼批 | 处理变长序列/padding | 需自己写 |

> 经验：若 GPU 利用率忽高忽低，多半是数据没喂上——优先调大 `num_workers`、开 `pin_memory`、把预处理搬离主进程。具体默认值与平台差异以官方文档为准。

---

## 5. AMP 混合精度：又快又省

### 5.1 它解决什么

默认参数和计算都是 FP32（32 位）。很多算子用 **FP16/BF16（16 位）** 算就够了：**显存减半、矩阵乘在 Tensor Core 上更快**。混合精度（Automatic Mixed Precision）= 大部分算子用 16 位，少数对精度敏感的（如求和、softmax、loss）保留 FP32。

### 5.2 两个核心工具

- `torch.autocast`：上下文管理器，自动把区域内算子选择性地降到半精度。
- `GradScaler`：**梯度缩放器**，专为 FP16 设计，解决「小梯度下溢成 0」的问题。

为什么要缩放？FP16 能表示的最小正数远大于 FP32，很多 $10^{-7}$ 量级的梯度直接变 0。做法是：**先把 loss 乘一个大数 $S$ 再反传**（梯度同比放大，躲开下溢区），更新前再 **除回 $S$**。

```
普通：          loss ──backward──> grad（可能下溢成0）

AMP+Scaler：
  loss × S ──backward──> grad×S（落在可表示区间）
                          │ unscale (÷S)
                          ▼
                       真实 grad ──> step
   若发现 inf/nan：跳过本次 step，自动调小 S（动态缩放）
```

### 5.3 接进训练循环（思路）

```
scaler = GradScaler()
for x, y in loader:
    optimizer.zero_grad()
    with autocast(device_type='cuda', dtype=float16):  # 自动半精度
        out  = model(x)
        loss = criterion(out, y)
    scaler.scale(loss).backward()   # 放大后反传
    scaler.step(optimizer)          # 内部先 unscale 再 step（遇 inf 则跳过）
    scaler.update()                 # 动态调整缩放系数 S
```

### 5.4 FP16 vs BF16

| | FP16 | BF16 |
|---|------|------|
| 指数位 | 少（动态范围小） | 多（范围≈FP32） |
| 是否需 GradScaler | **需要**（易下溢） | 通常**不需要** |
| 硬件要求 | 较老卡也支持 | 需较新 GPU（如 Ampere+） |
| 大模型训练常用 | 较少 | **更常用**（更稳） |

> 选择以你的 GPU 与框架文档为准：能用 BF16 时通常优先 BF16，省去 GradScaler 的麻烦。

---

## 6. 梯度累积：用小卡跑大 batch

### 6.1 痛点

大 batch 通常更稳、对某些任务更好，但显存放不下。梯度累积（gradient accumulation）让你**不增加显存**也能等效大 batch。

### 6.2 原理：利用「.grad 累加」

回忆第 2 节——`backward()` 是把梯度**累加**进 `.grad`。于是：连续做 $K$ 个小 batch 的 `backward()`、**先不 step**，`.grad` 会自动叠加；攒够 $K$ 次再 `step()` 一次。等效 batch 大小 = `micro_batch × K`。

为保证和真正大 batch 数学等价，要把每个 micro-batch 的 loss **除以 $K$**（因为大 batch 的 loss 是平均，不是求和）。

```
micro1 ─backward─┐
micro2 ─backward─┤ grad 不断累加（不 step / 不 zero）
 ...             │
microK ─backward─┘
                 ▼
            step()  +  zero_grad()   ←── 每 K 步才更新一次
```

### 6.3 写法（思路）

```
K = 4
optimizer.zero_grad()
for i, (x, y) in enumerate(loader):
    loss = criterion(model(x), y) / K     # 关键：除以 K
    loss.backward()                       # 累加梯度
    if (i + 1) % K == 0:
        optimizer.step()
        optimizer.zero_grad()
```

| 维度 | 真·大 batch | 梯度累积 |
|------|------------|---------|
| 显存 | 高 | **低**（只放 micro batch） |
| 速度 | 快（一次算完） | 略慢（多次前/反向） |
| 数学等价 | — | 基本等价（BN 统计除外） |

> 注意：**BatchNorm** 是在单个 micro-batch 上统计的，梯度累积**不能**让 BN 看到大 batch 的统计量；分布式下与 `no_sync()` 配合可省去中间几次梯度同步。

---

## 7. Checkpoint：存档与续训

### 7.1 存什么

训练随时可能中断（机器挂、抢占、调参重启）。要能**从断点恢复**，需要保存的不只是模型权重：

```
一个完整 checkpoint 应包含
┌────────────────────────────────────────────┐
│ model.state_dict()       参数（必存）        │
│ optimizer.state_dict()   动量/二阶矩等状态   │  ← 漏了它，Adam 等会"失忆"
│ lr_scheduler.state_dict()学习率调度进度      │
│ epoch / global_step      训练到哪了          │
│ scaler.state_dict()      AMP 缩放系数（用AMP时）│
│ 随机数种子 / rng_state   保证可复现           │
└────────────────────────────────────────────┘
```

### 7.2 为什么存 `state_dict` 而不是整个模型对象

`state_dict` 是一个「参数名 → 张量」的字典，**与代码结构解耦**，跨版本、跨重构更稳。直接 `torch.save(model)` 会序列化类定义，换了代码就可能加载失败。**推荐只存 `state_dict`**。

### 7.3 存 / 取（思路）

```
# 保存
torch.save({
    'step': step,
    'model':     model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler':    scaler.state_dict(),
}, 'ckpt.pt')

# 恢复
ckpt = torch.load('ckpt.pt', map_location='cpu')   # 先到 CPU 再分发，避免显存冲突
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
start_step = ckpt['step'] + 1                       # 从下一步继续
```

> 实践：定期存（每 N 步）+ 只保留最近几个；分布式下通常**只让 rank 0 写盘**，避免多进程抢同一文件。大模型分片（FSDP/ZeRO）的 checkpoint 形态不同，详见 [[llm-train/pytorch/distribution/README]]。

---

## 8. 最小可跑示例：把所有补丁串起来

下面是一段「带 AMP + 梯度累积 + checkpoint」的训练骨架思路，注释标出每一步对应前文哪一节：

```
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler    = torch.cuda.amp.GradScaler()           # §5
loader    = DataLoader(ds, batch_size=32,         # §4
                       shuffle=True, num_workers=4, pin_memory=True)
K = 4                                              # §6 累积步数

model.train()                                      # §3 训练模式
optimizer.zero_grad()
for step, (x, y) in enumerate(loader):
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

    with torch.autocast('cuda', dtype=torch.bfloat16):   # §5 混合精度
        out  = model(x)                            # ② forward
        loss = criterion(out, y) / K               # ③ loss，除以 K（§6）

    scaler.scale(loss).backward()                  # ④ backward（放大反传）

    if (step + 1) % K == 0:                         # §6 攒够 K 步才更新
        scaler.step(optimizer)                      # ⑤ step（内部 unscale）
        scaler.update()
        optimizer.zero_grad()                       # ① 清梯度

    if (step + 1) % 1000 == 0:                       # §7 定期存档
        torch.save({'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict()}, 'ckpt.pt')
```

整体调用链一图收尾：

```
DataLoader ─batch─> [autocast: forward → loss/K] ─> scaler.scale(loss).backward()
   每 K 步 ─> scaler.step → scaler.update → zero_grad
   每 N 步 ─> save checkpoint     （多卡时）backward 后梯度 all-reduce 同步 → 见 distribution
```

### 从单卡到多卡

上面是单卡骨架。要扩到多卡数据并行（DDP），核心改动是：每个进程一份模型副本，`backward()` 时框架自动把各卡梯度 **all-reduce** 求平均，使各副本参数保持一致。分布式相关的进程组、`get_rank()` / `get_world_size()` / `set_device()`、FSDP 分片等，见 [[llm-train/pytorch/distribution/README]]。

---

## 9. 常见问题

| 现象 | 可能原因 | 处理思路 |
|------|---------|---------|
| loss 不下降、参数不动 | 漏了 `optimizer.step()` 或忘了把 grad 接上 | 检查五步是否齐全 |
| loss 越训越爆 / NaN | 漏 `zero_grad()`、学习率过大、FP16 下溢 | 加 clip、降 lr、用 BF16 或 GradScaler |
| `.grad` 全是 None | 张量 `requires_grad=False`、在 `no_grad()` 里建图、loss 非标量 | 检查模式与 loss 维度 |
| `backward()` 报「retain_graph」 | 同一张图反传两次 | 一般是结构写错；确需复用才加 `retain_graph=True` |
| 显存 OOM | batch 太大、没释放中间量 | 减 batch、用梯度累积、AMP、`no_grad()` 评估 |
| GPU 利用率忽高忽低 | 数据加载是瓶颈 | 调大 `num_workers`、`pin_memory`、预取 |
| 续训后 loss 突然变高 | 没存/没载 optimizer 状态 | checkpoint 必须含 optimizer state |
| 评估结果不稳/偏高 | 忘 `model.eval()` 或没 `no_grad()` | 评估前切 eval + no_grad |
| 多卡结果对不上 | 各进程随机种子/数据划分不一致 | 用 `DistributedSampler`、固定种子 |

分布式排查：`export NCCL_DEBUG=INFO` 打印 NCCL 通信详细日志定位卡死/超时；`export NCCL_SOCKET_IFNAME=eth0` 显式指定网卡避免选错接口。

---

## 🔗 跳转链接

- 知识地图总览：[[00-知识地图]]；框架基础（张量/autograd/nn）：[[ai-framework/pytorch/README]]
- 训练总览（数据/优化/调参全景）：[[llm-train/README]]；分布式训练（DDP/FSDP/通信）：[[llm-train/pytorch/distribution/README]]

> 外部参考（以官方文档为准）：PyTorch 官方教程与示例 pytorch/examples、pytorch/tutorials；FSDP 教程 pytorch.org/tutorials/intermediate/FSDP_tutorial.html；官方镜像 hub.docker.com/r/pytorch/pytorch。
