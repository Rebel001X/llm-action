# 使用 DDP + 流水线并行训练 Transformer 模型（2D 并行入门）

> 一句话定位：在 4 张 GPU 上把一个约 10 亿参数的 Transformer **沿层切成 2 段流水线**（PP），再用 **DDP 复制 2 份流水线** 并行喂数据 —— 这是「数据并行 × 流水线并行」最小可跑的 2D 并行样板。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/pytorch/distribution/README]] · [[B07:llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]] · [[docs/transformer内存估算]]

本文是对官方教程 *TRAINING TRANSFORMER MODELS USING DISTRIBUTED DATA PARALLEL AND PIPELINE PARALLELISM* 的逐行精读，配套脚本为同目录 `ddp_pipeline.py`。所有数字均来自该脚本与本目录 `README.md` 的真实运行日志，未核实处标注「约/见原文」。

---

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 锚点 | 一句话讲清 2D 并行拓扑 | PP×DP、4 卡 |
| 1 地基 | 为什么单纯 DP 或单纯 PP 不够 | 显存墙、气泡 |
| 2 模型定义 | Encoder/Decoder 如何拆成 `nn.Sequential` | 序列化、(S,N) |
| 3 切分 PP | 8 层如何切成 2 段、`partition_len` 怎么算 | `Pipe`、micro-batch |
| 4 DDP 包裹 | 为什么 `DistributedDataParallel(model)` 不传 device_ids | 多设备模块 |
| 5 数据切分 | `batchify` 里 rank 维度切分的玄机 | DP 不重叠数据 |
| 6 训练循环 | `model(data).local_value()` 与 target 搬设备 | RRef、跨设备 loss |
| 7 公式/数值 | 显存账、参数量手算、气泡率 | FLOPs、bubble |
| 8 评价/局限 | `checkpoint="never"`、不支持 Windows | 兼容性坑 |

---

## 0. 一句话锚点

> **进程 P0 用 GPU{0,1} 跑一条流水线，进程 P1 用 GPU{2,3} 跑同一条流水线；两条流水线是同一模型的两个副本，它们之间用 DDP（NCCL all-reduce）同步梯度。**

这就是 2D 并行：
- **流水线并行（PP）**：把模型「按层」切到 2 张卡上，单卡放不下的模型现在放得下了 → 解决 **显存墙**。
- **数据并行（DP / DDP）**：把整条流水线**复制**成 2 份，各吃半个 batch → 解决 **吞吐**。

```
                     全局 batch (DDP 切成 2 半)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  进程 rank0                  进程 rank1
  ┌─────────────┐            ┌─────────────┐
  │   Pipe A    │            │   Pipe B    │   ← 同一模型的两个副本
  │ GPU0 → GPU1 │            │ GPU2 → GPU3 │   ← 每条 = 流水线并行(2 段)
  └─────────────┘            └─────────────┘
        │  反向后梯度              │
        └───────── DDP all-reduce ┘   ← NCCL 跨进程同步梯度
```

记住坐标：**DP 维度 = 2（rank0/rank1），PP 维度 = 2（每条流水线 2 段），合计 2×2 = 4 张 GPU。**

---

## 1. 地基：为什么要 DDP × PP 一起上？

### 1.1 两堵墙

| 矛盾 | 现象 | 单一手段为何不够 |
|---|---|---|
| 显存墙 | 10 亿参 + 激活值 + 优化器状态，单卡 OOM | **纯 DDP** 要求每张卡放下**整个模型副本**，放不下就跪 |
| 吞吐墙 | 模型切到多卡后单条流水线利用率低（有气泡） | **纯 PP** 只切一份，batch 大不起来，卡间通信串行，扩展性差 |

**结论**：PP 先把「装不下」变成「装得下」，DP 再把「装得下的那份」复制多份来提吞吐。这是从「模型并行」走向「大规模训练」的第一块拼图。

### 1.2 前置依赖（教程明确列出）

- 流水线并行 `torch.distributed.pipeline.sync.Pipe`（PyTorch 1.8+ 的实验 API）
- `nn.Transformer` + TorchText 的序列到序列建模（本文模型与 [[2-使用torchtext训练transformer模型]] 相同，只是放大并切段）
- DDP 入门 `DistributedDataParallel`

> ⚠️ 平台护栏：脚本开头 `if sys.platform == 'win32': sys.exit(0)` —— **Windows 不支持流水线并行**；同时 `torch.cuda.device_count() < 4` 直接退出，**本例硬性需要 4 张 GPU**。

---

## 2. 模型定义：把 Transformer 拆成「可切的乐高」

`Pipe` 只能切 `nn.Sequential`。所以第一步是把原来一体的 Transformer 拆成「编码端 / N 个 Encoder 层 / 解码端」三类积木，再串成一条 `Sequential`。

### 2.1 三块积木

```
Encoder(词嵌入+位置编码)  →  TransformerEncoderLayer × 8  →  Decoder(线性投影)
   nn.Embedding                  自注意力 + FFN                nn.Linear
   PositionalEncoding            （参数量大头在这）            映射回词表 ntokens
```

- **PositionalEncoding**：用不同频率的 $\sin/\cos$ 注入位置信息，与词嵌入同维度可直接相加。脚本里 `pe` 被包成 `nn.Parameter(..., requires_grad=False)`（不参与梯度，但随模块迁移设备）。
  $$PE_{(pos,2i)}=\sin\!\Big(\frac{pos}{10000^{2i/d}}\Big),\quad PE_{(pos,2i+1)}=\cos\!\Big(\frac{pos}{10000^{2i/d}}\Big)$$
- **Encoder.forward**：先 `src = src.t()` 转成 `(S, N)`（序列在前），再 `embedding * sqrt(ninp)` 缩放，最后加位置编码。注意这里的 `Encoder` 是「输入嵌入端」，不是 Transformer 的编码器堆栈。
- **Decoder.forward**：`nn.Linear(ninp, ntoken)` 把隐状态投回词表，再 `.permute(1,0,2)` 把 batch 维放回最前，方便流水线输出对接 loss。

### 2.2 为什么必须拆？

`Pipe` 的输入是 `nn.Sequential`，它会**按子模块在序列中的位置**决定哪段放哪张卡。一体的 `nn.TransformerEncoder` 是个黑盒，无法在中间切开；只有拆成「每层一个子模块」的扁平序列，才能在第 4 层和第 5 层之间画一刀。

---

## 3. 流水线切分：8 层 → 2 段，`Pipe` 怎么放

### 3.1 模型规模（脚本实参）

| 超参 | 值 | 含义 |
|---|---|---|
| `emsize` | 4096 | 嵌入维度 $d_{model}$ |
| `nhid` | 4096 | FFN 隐藏维度 |
| `nhead` | 16 | 注意力头数 |
| `nlayers` | 8 | `TransformerEncoderLayer` 层数 |
| `num_gpus` | 2 | **每条流水线**的段数 = GPU 数 |

> 注意：注释里写「~1 billion」，但真实日志打印 **`Total parameters: 1,061,924,974`（约 10.6 亿）**，以日志为准。

### 3.2 切分算法（逐行拆）

```python
partition_len = ((nlayers - 1) // num_gpus) + 1   # = ((8-1)//2)+1 = 4
```
即每段放 `partition_len = 4` 层。日志印证：`partition_len: 4 rank: 0/1`。

放置逻辑（核心循环）：

```python
tmp_list = [Encoder(...).cuda(2 * rank)]          # 嵌入端放到本进程第 0 张卡
module_list = []
for i in range(nlayers):                          # i = 0..7
    block = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
    if i != 0 and i % partition_len == 0:         # i==4 时触发切段
        module_list.append(nn.Sequential(*tmp_list))
        tmp_list = []
    device = i // partition_len                   # 0..3 → 0；4..7 → 1
    tmp_list.append(block.to(2 * rank + device))  # 关键：物理设备号 = 2*rank + 段号
tmp_list.append(Decoder(...).cuda(2 * rank + num_gpus - 1))  # 解码端放最后一张卡
module_list.append(nn.Sequential(*tmp_list))
```

**`2 * rank + device` 是整篇的灵魂**：

| rank（进程） | 段 device=0 | 段 device=1 |
|---|---|---|
| 0 | GPU **0** | GPU **1** |
| 1 | GPU **2** | GPU **3** |

这样两个进程自动占用互不重叠的 4 张卡。设备拓扑：

```
进程 rank0:  [Encoder + Layer0..3]──激活前传──▶[Layer4..7 + Decoder]
              └── GPU 0 ───────┘   (P2P/NVLink)  └── GPU 1 ──────┘

进程 rank1:  [Encoder + Layer0..3]──激活前传──▶[Layer4..7 + Decoder]
              └── GPU 2 ───────┘                └── GPU 3 ──────┘
```

> 教程强调：**传给 `Pipe` 的 `nn.Sequential` 只放 2 个元素**（对应 2 张卡），让 `Pipe` 只处理两个分区、避免跨分区开销。

### 3.3 micro-batch：流水线的发动机

```python
chunks = 8
model = Pipe(nn.Sequential(*module_list), chunks=chunks, checkpoint="never")
```

`Pipe` 把每个 mini-batch 再切成 `chunks=8` 个 **micro-batch**，让它们像工厂流水线一样在两段之间错峰流动 —— 段 1 处理第 2 个 micro-batch 时，段 2 正在处理第 1 个，从而**减少气泡（bubble）**。

```
时间 →
GPU0(段1):  m1  m2  m3  m4  ......
GPU1(段2):      m1  m2  m3  m4 ...
            ↑bubble        填满后两卡同时忙
```

> `checkpoint="never"` 是硬约束：截至 PyTorch 1.8，**`Pipe` 的激活重计算（checkpoint）与 DDP 不兼容**，所以这里必须关掉重计算，代价是显存占用更高。

### 3.4 RPC 框架：`Pipe` 的隐藏依赖

```python
rpc.init_rpc(name="worker", rank=0, world_size=1,
    rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
        init_method="file://{}".format(tmpfile.name),
        _transports=["ibv", "uv"], _channels=["cuda_ipc", "cuda_basic"]))
```

`Pipe` 通过 **RRef** 依赖 RPC（为将来「跨主机流水线」留口子）。本例是「单进程驱动多 GPU」，所以 RPC 的 `world_size=1`（只有一个 worker）。`_transports/_channels` 在 PyTorch ≥1.8.1 后可省略，是当时的 workaround。

> 易混点：这里有**两套 world_size**。RPC 的 `world_size=1`（管流水线内部），DDP 的 `world_size=2`（管两条流水线之间，见下文 `mp.spawn`）。别搞混。

---

## 4. DDP 包裹：把整条流水线当成一个「模块」复制

```python
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)  # world_size=2
model = DistributedDataParallel(model)     # 注意：没有 device_ids！
```

**关键反直觉点：`DistributedDataParallel(model)` 不传 `device_ids`。**

平时单卡 DDP 写 `DDP(model, device_ids=[local_rank])`，因为模型在**一张**卡上。但这里 `model`（`Pipe`）是**横跨两张卡的多设备模块**，没有单一 device_id 可填，所以留空 —— DDP 会把它当作「多设备模块」处理，只负责**对所有参数做 all-reduce 梯度同步**，不负责搬运。

日志里那条警告正是这个原因：
```
[W logger.cpp:317] Warning: Cuda time stats are not collected for multi-device modules.
```

```
反向传播完成后：
  Pipe A 的梯度(分布在 GPU0,GPU1)  ┐
                                   ├─ DDP all-reduce (NCCL) ─▶ 两副本梯度求平均
  Pipe B 的梯度(分布在 GPU2,GPU3)  ┘
  → 两条流水线参数始终保持一致
```

---

## 5. 数据切分：DDP 副本之间数据不能重叠

`batchify` 里有段容易被忽略的 rank 切分逻辑：

```python
def batchify(data, bsz, rank, world_size, is_train=False):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)          # 裁掉除不尽的尾巴
    data = data.view(bsz, -1).t().contiguous()      # 排成 (序列, bsz) 列
    if is_train:                                    # 只对训练集切 rank
        data_per_rank = data.size(0) // world_size
        data = data[rank * data_per_rank : (rank + 1) * data_per_rank]
    return data.to(device)
```

- `device = torch.device(2 * rank)` —— 数据落到**本进程第 0 张卡**（GPU0 或 GPU2），即流水线**入口**那张卡。
- `is_train=True` 时按 `rank` 切分：rank0 拿前半段、rank1 拿后半段。**这是 DDP 的精髓**：两个副本看不同数据，梯度 all-reduce 后等价于在大 batch 上训练。验证/测试集不切（每个 rank 各自完整评估）。

数据形状流转（`bptt=35`，batch_size=20）：

```
原始 token 流 ──batchify──▶ (序列长, 20)  ──按rank切──▶ (序列长/2, 20)
   ──get_batch(i)──▶ data:(20, 35) target:(700,)   ← 注意 data 已 .t() 成 batch-first
```

`get_batch` 里 `return data.t(), target` 把 data 转成 batch-first，因为流水线对外接口要求 batch 维在前。

---

## 6. 训练循环：跨设备的 loss 怎么算

```python
output = model(data).local_value()                 # ① Pipe 返回 RRef，取本地值
loss = criterion(output.view(-1, ntokens),
                 targets.cuda(2 * rank + 1))        # ② target 搬到流水线"出口"卡
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)   # ③ 防梯度爆炸
optimizer.step()
```

三个工程细节：

1. **`.local_value()`**：`Pipe.forward` 返回的是 `RRef`（远程引用，为跨主机预留）。因为本例流水线就在本进程内，输出是本地的，直接 `.local_value()` 取出真实张量。
2. **`targets.cuda(2 * rank + 1)`**：流水线的输出停在**最后一段那张卡**上（rank0 → GPU1，rank1 → GPU3，即 `2*rank+1`）。loss 必须在同一张卡上算，所以 target 要手动搬过去。**这是模型并行最常见的「设备不匹配」陷阱。**
3. 超参：`SGD(lr=5.0)`、`StepLR(gamma=0.95)`、`clip_grad_norm_(0.5)`；为缩短跑时只训 50 个 batch（`nbatches = min(50*bptt, ...)`），跑 3 个 epoch。

### 真实运行结果（来自本目录 README 日志）

```
partition_len: 4 rank: 0 / 1
[RANK 0/1]: Total parameters in model: 1,061,924,974
epoch1  ms/batch ~386~518   valid ppl 2.48
epoch2  ms/batch ~387       valid ppl 1.26
epoch3  ms/batch ~387       valid ppl 1.37
End of training | test loss 0.27 | test ppl 1.31
```

> 现象解读：epoch1 的 loss/ppl 是天文数字（lr=5.0 过大、未热身），属正常的剧烈下降过程；几个 epoch 后 ppl 收敛到 ~1.3。**这只是流程演示，不追求 SOTA 质量**。

### 启动方式

```python
world_size = 2     # DDP 副本数 = 流水线条数
mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
```
命令行就一句 `python ddp_pipeline.py`，`mp.spawn` 拉起 2 个进程，每进程一条流水线。

---

## 关键公式 / 数值示例

### 7.1 参数量手算（验证那 10.6 亿）

单个 `TransformerEncoderLayer`（$d=4096$，FFN 隐藏 $h=4096$）：
- 自注意力 QKVO：$4 \times d^2 = 4 \times 4096^2 \approx 6.71\times10^7$
- FFN 两层：$2 \times d \times h = 2 \times 4096^2 \approx 3.36\times10^7$（注意本例 $h=d$，比常见 $4d$ 小）
- 合计每层 ≈ $1.0\times10^8$（1 亿），8 层 ≈ **8 亿**

加上嵌入与输出投影：$ntokens \times d$。WikiText-2 词表约 2.8 万，$2.8\times10^4 \times 4096 \approx 1.15\times10^8$，嵌入+解码两处共 ≈ 2.3 亿。

合计 ≈ $8\text{亿} + 2.3\text{亿} \approx 10.3$ 亿，与日志 **1,061,924,974 ≈ 10.6 亿** 同量级（手算未含 LayerNorm/bias，量级吻合即可）。

### 7.2 显存账（对照 README 的 nvidia-smi）

真实占用（README 日志）：

| GPU | 进程 | 占用 | 角色 |
|---|---|---|---|
| 0 | 29461(rank0) | 11612 MiB | rank0 段1 |
| 0 | 29462(rank1) | 556 MiB | rank1 借用(NCCL/上下文) |
| 1 | 29461(rank0) | 11748 MiB | rank0 段2 |
| 2 | 29462(rank1) | 11354 MiB | rank1 段1 |
| 3 | 29462(rank1) | 11748 MiB | rank1 段2 |

每张主力卡约 **11.5 GB**。粗略验证（FP32）：
- 每段 ≈ 5.3 亿参 → 参数 $5.3\times10^8 \times 4\text{B} \approx 2.1\,\text{GB}$
- 梯度同量 ≈ 2.1 GB；SGD 无动量则优化器状态 ≈ 0
- 剩余 ~7 GB 来自**激活值**（因 `checkpoint="never"` 不重计算，激活全留显存）+ CUDA 上下文 + micro-batch 缓冲

→ 与实测 11.5 GB 同量级。结论：**关掉激活重计算是这里的显存大户**，但它是 DDP 兼容性的必要代价。

### 7.3 流水线气泡率

PP 段数 $p=2$，micro-batch 数 $m=\text{chunks}=8$，理想气泡率：
$$\text{bubble ratio} = \frac{p-1}{m+p-1} = \frac{1}{8+1} \approx 11.1\%$$
即约 11% 的时间一张卡空转。**增大 `chunks`（micro-batch 数）可降低气泡率**，但 micro-batch 太小又会让单卡 kernel 利用率下降，需权衡。

### 7.4 加速比直觉

- DP 维度 2：理想吞吐 ×2（梯度 all-reduce 通信换来的扩展性）。
- PP 维度 2：**不是为了加速，而是为了「装得下」**；扣掉 ~11% 气泡，单条流水线吞吐 < 单卡理想值，但换来了 2× 模型容量。

---

## 评价 / 对照 / 局限

| 维度 | 本方案（PyTorch `Pipe` + DDP） | 说明 |
|---|---|---|
| 解决的问题 | 显存墙(PP) + 吞吐(DP) 的 2D 组合 | 入门级 3D 并行的前两维 |
| 切分粒度 | 按层切（inter-layer），同步流水线（GPipe 式） | 非 1F1B、非张量切分 |
| 激活重计算 | **必须 `checkpoint="never"`** | 1.8 时与 DDP 不兼容，显存代价大 |
| 平台 | 仅 Linux，需 ≥4 GPU | Windows 直接退出 |
| API 状态 | `torch.distributed.pipeline.sync.Pipe` 实验性 | 新代码建议看 PiPPy / Megatron / DeepSpeed，**以官方最新为准** |
| 缺失维度 | 无张量并行(TP)、无 ZeRO 切优化器 | 见 [[B07:llm-inference/大模型推理张量并行]] |
| 调试坑 | RPC/DDP 两套 world_size；target 要 `.cuda(2*rank+1)`；DDP 不传 device_ids | 多设备模块的常见雷 |

**与真正的大模型训练栈对照**：现代框架（Megatron-LM / DeepSpeed）用 **TP×PP×DP 三维并行 + ZeRO** 切分优化器状态，并用 1F1B 调度进一步压气泡。本教程是理解这套机制的「最小可跑骨架」——**先把 PP×DP 的设备映射、梯度同步、跨设备 loss 这三件事吃透，再上生产框架。**

---

## 🔗 跳转链接

- 上游基础：[[3-使用流水线并行训练Transformer模型]]（纯 PP，无 DDP）· [[2-使用torchtext训练transformer模型]]（原始单卡模型）· [[1-流水线]]（流水线并行原理）
- 并行全景：[[llm-train/pytorch/distribution/README]] · [[llm-train/README]] · [[B07:llm-inference/大模型推理张量并行]]
- 通信底座：[[ai-infra/网络/集合通信原语]]（all-reduce/NCCL）· [[ai-infra/算力/GPU工作原理]]
- 模型本体：[[llm-algo/transformer/模型架构]]
- 算账工具：[[docs/transformer内存估算]] · [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- 知识枢纽：[[00-知识地图]]
