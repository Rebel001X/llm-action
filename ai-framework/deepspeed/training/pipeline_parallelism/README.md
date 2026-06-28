# DeepSpeed 流水线并行(Pipeline Parallelism)

> 把一个深层网络"按层切成几段"放到不同 GPU 上,用"流水线"让多段 GPU 同时干活,从而训练单卡放不下的大模型。📍 导航:[[00-知识地图]]
> 🔗 相关:[[llm-train/README]] · [[llm-train/pytorch/distribution/README]] · [[ai-infra/网络/集合通信原语]] · [[docs/transformer内存估算]]

## 阅读地图

| 章节 | 你会学到 | 关键词 |
| --- | --- | --- |
| 0. 一句话锚点 | PP 到底是什么 | 层切分 / 微批 / 气泡 |
| 1. 地基 | 为什么单卡放不下、三种并行怎么分工 | DP / TP / PP |
| 2. 朴素层切分的问题 | 为什么"切了层"还不够 | 串行等待 / 利用率 |
| 3. 微批与流水线调度 | GPipe 与 1F1B 怎么填满流水线 | micro-batch / bubble |
| 4. DeepSpeed 的实现 | PipelineModule / 张量约定 / 接口 | nn.Sequential / LayerSpec |
| 5. 显存与通信账 | 切了 PP 到底省多少显存 | 激活 / P2P 通信 |
| 6. 启动命令拆解 | 文件顶部那条命令每个参数啥意思 | `-p` / `--include` |
| 关键公式/数值 | 气泡占比、加速比、显存手算 | $1-\frac{1}{1+(m-1)/(p-1)}$ |
| 评价/对照/局限 | PP vs TP vs DP,什么时候用 PP | 通信量 / 负载均衡 |

---

## 0. 一句话锚点

**流水线并行 = 把模型"纵向"按层切成 $p$ 段(stage),每段放一张 GPU;再把一个大 batch 切成 $m$ 个微批(micro-batch),像工厂流水线一样让 GPU 们错峰并行,从而既能放下超大模型、又尽量不让 GPU 闲着。**

- "纵向切层" → 解决**显存放不下**(每张卡只存模型的一段)。
- "微批 + 流水线调度" → 解决**朴素切层时 GPU 互相干等**的利用率问题。
- 代价:存在**流水线气泡(bubble)**——开头填、结尾排空的那段时间里部分 GPU 在空转。

---

## 1. 地基:为什么需要 PP,它在三种并行里是什么位置

### 1.1 单卡放不下的来源

训练显存大致由四块构成(以混合精度 + Adam 为例,$\Psi$ 为参数量):

| 项 | 量级(每参数字节) | 说明 |
| --- | --- | --- |
| 参数(fp16) | 2 | 前向/反向用 |
| 梯度(fp16) | 2 | 反向产出 |
| Adam 优化器状态 | 12 | fp32 参数副本 4 + 一阶动量 4 + 二阶动量 4 |
| **小计(状态)** | **16** | 即经典的 $16\Psi$ 字节 |
| 激活值 | 随 batch、序列长度变化 | 前向缓存,反向要用 |

一个 100 亿(10B)参数的模型,光"状态"就需 $16 \times 10^{10} = 160\,\text{GB}$,远超单张 80GB 卡。于是必须把模型**拆到多卡**。详见 [[docs/transformer内存估算]]。

### 1.2 三种并行的分工(一张图看懂)

```
            一个大模型 + 一个大 batch
                      │
   ┌──────────────────┼──────────────────────┐
   │                  │                       │
 数据并行 DP        张量并行 TP            流水线并行 PP
 (复制整模型,      (把单层矩阵            (把不同层
  切 batch)         横/竖切到多卡)         分到不同卡)
   │                  │                       │
 每卡同一套层      每卡同一层的一部分      每卡不同的层
 通信:梯度 AllReduce  通信:每层多次AllReduce  通信:段间一次P2P
 适合:模型放得下    适合:单层太大/卡间带宽高  适合:模型太深/跨节点
```

- **DP(数据并行)**:每张卡放**完整模型**,各喂不同数据,反向后对梯度做 AllReduce。模型放不下时无能为力。参考 [[llm-train/pytorch/distribution/README]]。
- **TP(张量并行)**:把**同一层**的大矩阵切成多块分到多卡,每个矩阵乘都要卡间通信。通信极频繁,通常只在**单节点内 NVLink** 用。参考 [[B07:llm-inference/大模型推理张量并行]]。
- **PP(流水线并行,本文主角)**:把**不同的层**分到不同卡,只在**相邻段的边界**传一次激活/梯度。通信量小、对带宽要求低,**适合跨节点**扩展,是 3D 并行(DP×TP×PP)里"纵向"的那一维。

> 记忆口诀:**DP 切数据,TP 切矩阵,PP 切层。** 大模型训练常三者叠加成 3D 并行。

---

## 2. 朴素层切分的问题:为什么"切了层"还不够

最自然的想法:把 8 层网络切成 2 段,第 0–3 层放 GPU0,第 4–7 层放 GPU1。前向时 GPU0 算完把激活传给 GPU1;反向时 GPU1 算完把梯度传回 GPU0。

```
朴素串行(一个 batch,p=2):

时间 →
GPU0:  [F0─────]                       [B0─────]
GPU1:           [F1─────] [B1─────]
       ↑前向算  ↑传激活           ↑传梯度  ↑反向算

任意时刻只有 1 张卡在干活,另一张干等 → 利用率 ≈ 1/p
```

**问题本质**:GPU 之间有**数据依赖**(GPU1 必须等 GPU0 的输出),朴素做法让它们**串行**执行。切了 $p$ 段,利用率反而只有 $\approx 1/p$,GPU 越多越浪费。这就是流水线并行真正要解决的核心矛盾。

---

## 3. 微批与流水线调度:把流水线填满

### 3.1 关键招数:把 batch 切成微批(micro-batch)

把一个 batch 切成 $m$ 个小微批,依次喂入流水线。当 GPU0 算完微批 0 的前向、把它交给 GPU1 时,GPU0 **不闲着**,马上开始算微批 1 的前向。这样多张卡就能在**不同微批**上同时工作。

```
GPipe 调度(p=2, m=4):F=前向, B=反向, 数字=微批号

时间 →   1    2    3    4    5    6    7    8
GPU0:  [F0][F1][F2][F3]              [B3][B2][B1][B0]
GPU1:       [F0][F1][F2][F3][B3][B2][B1][B0]
            └填充┘  └──稳态(两卡都忙)──┘ └排空┘
              ↑bubble                        ↑bubble
```

- **填充(warm-up)**:流水线刚开始,后面的段还没拿到数据 → 气泡。
- **稳态(steady)**:所有段都在不同微批上忙 → 满载。
- **排空(cool-down)**:最后几个微批走完,前面的段空出来 → 气泡。

微批越多($m$ 越大),稳态越长,气泡占比越小;但 $m$ 太大会增加缓存的激活、且每个微批太小会降低算子效率,需折中。

### 3.2 GPipe vs 1F1B(DeepSpeed 默认)

| 调度 | 顺序 | 激活显存峰值 | 说明 |
| --- | --- | --- | --- |
| **GPipe** | 先把所有微批的 F 全做完,再统一做 B | 高:要同时缓存 $m$ 份激活 | 直观,但显存压力大 |
| **1F1B**(One-Forward-One-Backward) | 进入稳态后,每做一个 F 紧跟一个 B | 低:每段只需缓存约 $p$ 份激活 | DeepSpeed `PipelineEngine` 默认调度 |

```
1F1B 稳态(关键差别:F 和 B 交错,B 尽早执行以释放激活):

GPU0: F0 F1 F2 F3 B0 F4 B1 F5 B2 ...   ← 做完 B0 立刻释放微批0的激活
                   ↑一前一后交替
```

**为什么 1F1B 省显存**:GPipe 要等所有 $m$ 个前向都做完才开始反向,故必须同时缓存 $m$ 份激活;1F1B 让每个微批的反向**尽早**执行,激活用完即释放,峰值缓存数从 $O(m)$ 降到 $O(p)$。这对长序列大模型至关重要。

---

## 4. DeepSpeed 的实现:PipelineModule 与张量约定

### 4.1 模型必须表达成"层的序列"

DeepSpeed 用 `PipelineModule` 把模型描述成一串**有序的层**,然后自动按 stage 切分、分配到各 rank。两种写法:

```python
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

# 写法 A:直接传 nn.Sequential / 层列表
net = nn.Sequential(layer0, layer1, ..., layerN)
model = PipelineModule(layers=net, num_stages=2)

# 写法 B:用 LayerSpec 延迟构造(省显存:切分后才在目标卡上真正建层)
specs = [LayerSpec(TransformerLayer, hidden, heads) for _ in range(N)]
model = PipelineModule(layers=specs, num_stages=2,
                       loss_fn=nn.CrossEntropyLoss())
```

- **`num_stages`** = 流水线段数 $p$(= 命令里的 `-p`)。
- **`LayerSpec`** 只记录"怎么建这一层",**真正实例化推迟到切分之后**,避免每张卡都先把整模型建一遍再丢掉——这对大模型是必须的。
- **`partition_method`**:层到 stage 的切分策略(如按层数均分 `uniform`、按参数量均分 `parameters`),目标是让各 stage **负载均衡**,否则最慢的 stage 拖累整条流水线。

### 4.2 张量约定:层间只能传"张量"

流水线把激活在 stage 间用 **P2P send/recv** 传递,因此每一层的 `forward` 必须满足:

- 输入/输出要么是**单个张量**,要么是**张量元组**(不能是 dict / 任意 python 对象,否则没法跨进程传)。
- 形状在各微批间要一致(便于预分配通信缓冲)。

### 4.3 训练循环:engine.train_batch()

```python
engine, _, _, _ = deepspeed.initialize(
    args=args, model=model,
    model_parameters=[p for p in model.parameters() if p.requires_grad])

for step in range(steps):
    loss = engine.train_batch()   # 内部完成:取数据→微批切分→1F1B 调度
                                   #          →F/B→段间通信→优化器更新
```

`train_batch()` 一行就跑完"一个全局 batch"的完整流水线:**它内部自动做微批切分、F/B 调度、段间 P2P 通信、梯度累积与优化器 step**,使用者无需手写 forward/backward。

### 4.4 数据如何进入流水线

只有**第一个 stage** 真正读取输入数据、**最后一个 stage** 才有标签并计算 loss。DeepSpeed 用一个可重复(可被多次取 next)的数据迭代器,在每个微批向第一段喂数据;loss 在最后一段算出后,沿流水线反向回传梯度。

---

## 5. 显存与通信账:PP 到底省多少

### 5.1 显存:模型状态按 stage 均分

设总参数 $\Psi$、流水线段数 $p$。理想均分下,**每张卡只存 $1/p$ 的模型状态**:

$$
M_{\text{状态/卡}} \approx \frac{16\Psi}{p} \quad(\text{字节, 混合精度 + Adam})
$$

加上每段缓存的激活(1F1B 下约 $p$ 份微批激活)。所以 PP 直接把"模型放不下"问题缩小 $p$ 倍。

### 5.2 通信:只在段边界传一次

PP 的通信发生在**相邻 stage 之间**,每个微批:

- 前向:把该段输出激活 **send** 给下一段(1 次 P2P)。
- 反向:把对输入的梯度 **send** 回上一段(1 次 P2P)。

单次传输量 = 激活张量大小 $\approx b_\mu \cdot s \cdot h \cdot 2\,\text{字节}$($b_\mu$ 微批大小,$s$ 序列长,$h$ 隐藏维,fp16)。**注意:这与层数无关、与 TP 的"每层多次 AllReduce"相比小得多**,所以 PP 适合**跨节点/低带宽**场景。集合/点对点通信原语见 [[ai-infra/网络/集合通信原语]]。

```
通信量对比(直觉):
 TP:  ──每层都要 AllReduce──  通信∝层数, 必须高带宽(NVLink)
 PP:  ─段边界传一次激活─        通信∝段数, 容忍跨节点(InfiniBand/以太网)
```

---

## 6. 启动命令拆解(文件顶部那条)

```bash
deepspeed --include localhost:3,4,5,6 train.py \
          --deepspeed_config=ds_config.json -p 2 --steps=200
```

| 片段 | 含义 |
| --- | --- |
| `deepspeed` | DeepSpeed 的分布式启动器(类似 `torchrun`),负责拉起多进程、设环境变量 |
| `--include localhost:3,4,5,6` | 在本机**只用 3、4、5、6 号 GPU**(共 4 张),即 world_size=4 |
| `train.py` | 训练脚本 |
| `--deepspeed_config=ds_config.json` | DeepSpeed 配置文件(batch 大小、优化器、ZeRO、fp16 等) |
| `-p 2` | **流水线并行度 = 2**(脚本里据此设 `num_stages=2`);此处脚本参数,以脚本定义为准 |
| `--steps=200` | 训练 200 个 step |

**这台机器上的并行布局推演**:4 张 GPU + 流水线度 $p=2$ ⟹ 每条流水线占 2 张卡,4/2 = **2 条流水线并行做数据并行(DP=2)**。即 $\text{DP} \times \text{PP} = 2 \times 2 = 4$。

```
            GPU3        GPU4        GPU5        GPU6
流水线A:   stage0  ───►  stage1
流水线B:                            stage0  ───►  stage1
            └──── DP 副本0 ────┘    └──── DP 副本1 ────┘
            (两条流水线之间对梯度做 AllReduce)
```

> 具体 `-p` 是否对应 `num_stages`、是否还叠加 TP,取决于 `train.py` 怎么解析参数,**以该脚本与 DeepSpeed 官方文档为准**。

---

## 关键公式 / 算法 / 数值示例

### 公式 1:流水线气泡占比

总时间 = 填充 + 稳态 + 排空。气泡(空转)占比近似为:

$$
\text{Bubble Fraction} = \frac{p-1}{m + p - 1}
\quad\Longleftrightarrow\quad
\text{利用率} = \frac{m}{m + p - 1}
$$

其中 $p$ = 段数,$m$ = 微批数。**结论:$m \gg p$ 时气泡可忽略。**

### 公式 2:理想加速比

相对单段串行,流水线的理想加速比:

$$
\text{Speedup} = \frac{m}{m + p - 1}\cdot p
$$

### 数值手算示例

设 $p=4$,分别取 $m=4$ 和 $m=16$:

| $m$ | 气泡占比 $\frac{p-1}{m+p-1}$ | 利用率 | 相对 4 卡的有效加速 |
| --- | --- | --- | --- |
| 4 | $3/7 \approx 43\%$ | $4/7 \approx 57\%$ | $0.57 \times 4 \approx 2.3\times$ |
| 16 | $3/19 \approx 16\%$ | $16/19 \approx 84\%$ | $0.84 \times 4 \approx 3.4\times$ |

**解读**:同样 4 段流水线,微批从 4 提到 16,利用率从 57% 升到 84%——这正是"为什么 DeepSpeed 鼓励用较大的梯度累积步数(更多微批)"。

### 数值手算:显存对比

10B 参数模型,混合精度 + Adam,状态 $16\Psi = 160\,\text{GB}$:

- 单卡:需 160GB(+激活)→ 80GB 卡放不下。
- $p=4$ 流水线:每卡 $160/4 = 40\,\text{GB}$ 状态 + 激活 → 单张 80GB 卡可容纳。
- 若再叠加 ZeRO-1 把优化器状态在 DP 维再切分,可进一步降低。

### 数值手算:单次 P2P 通信量

$b_\mu=2$(微批大小),$s=2048$,$h=4096$,fp16:

$$
2 \times 2048 \times 4096 \times 2\,\text{字节} = 67\,\text{MB / 次}
$$

每段边界每个微批前向/反向各传一次,对 InfiniBand(几十~上百 GB/s)是轻量负载——再次印证 PP 对带宽宽容。

---

## 评价 / 对照 / 局限

| 维度 | 流水线并行 PP | 张量并行 TP | 数据并行 DP |
| --- | --- | --- | --- |
| 切什么 | 不同**层**到不同卡 | 同层**矩阵**切块 | 复制整模型,切 **batch** |
| 通信频率 | 低(段边界 1 次 P2P) | 高(每层多次 AllReduce) | 中(每 step 一次梯度 AllReduce) |
| 带宽要求 | 低,**可跨节点** | 高,**宜单节点 NVLink** | 中 |
| 显存收益 | 模型状态 ÷ $p$ | 模型状态 ÷ TP度 | 不省(ZeRO 才省) |
| 主要代价 | **流水线气泡** + 负载均衡难 | 频繁通信 | 不解决"放不下" |
| 何时首选 | 模型很深、跨节点扩展 | 单层巨大、卡间高带宽 | 模型放得下、要吞吐 |

**PP 的主要局限:**

1. **气泡浪费**:小 $m$ 时利用率低;需足够多微批,但微批太小又损算子效率。
2. **负载均衡难**:各 stage 计算量须接近,否则最慢 stage 卡脖子;不规则模型切分困难。
3. **接口约束**:模型要能表达成层序列、层间只传张量,改造非 `nn.Sequential` 结构有成本。
4. **首尾 stage 特殊**:只有首段读数据、末段算 loss,工程上要小心数据/标签的供给。

**实战建议**:大模型训练通常 **TP(节点内)× PP(跨节点)× DP/ZeRO(再扩展)** 组合成 3D 并行——TP 解决单层太大、PP 解决模型太深且省跨节点带宽、DP/ZeRO 提升吞吐并进一步分摊优化器状态。

---

## 🔗 跳转链接

- 知识地图枢纽:[[00-知识地图]]
- 训练总览:[[llm-train/README]] · [[llm-train/pytorch/distribution/README]]
- 张量并行对照:[[B07:llm-inference/大模型推理张量并行]]
- 通信原语:[[ai-infra/网络/集合通信原语]] · [[ai-infra/算力/GPU工作原理]]
- 显存估算:[[docs/transformer内存估算]]
- 性能指标:[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 模型结构:[[llm-algo/transformer/模型架构]] · [[llm-algo/moe/README]]
