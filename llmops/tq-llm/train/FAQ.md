# 大模型训练常见问题 FAQ（排错速查）

> 大模型训练任务（SFT / LoRA / 全参微调）跑挂、报错、OOM、loss 异常时的"症状 → 根因 → 处方"速查手册。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-framework/deepspeed/README]] [[llm-train/README]] [[llm-train/pytorch/distribution/README]] [[llmops/README]]

## 阅读地图

| 你想解决的问题 | 跳到 |
| --- | --- |
| 这份 FAQ 怎么用、排错总思路 | §0、§1 |
| 显存炸了（OOM / CUDA out of memory） | §2 |
| 多卡 / 分布式起不来、卡死、超时 | §3 |
| loss 变 NaN / 不下降 / 爆炸 | §4 |
| 模型/Tokenizer 加载报错（版本、依赖） | §5 |
| 数据加载慢、worker 被 kill、共享内存 | §6 |
| DeepSpeed ZeRO / Offload 相关坑 | §7 |
| 保存/恢复 checkpoint 出问题 | §8 |
| 原始报错条目（Baichuan2 / PyTorch）速查 | §9 |
| 常见坑汇总表 | 文末 |

---

## 0. 一句话锚点

**大模型训练报错的 90% 落在四类：显存不够、分布式没起来、数值不稳定（NaN）、环境/版本不匹配。** 先用一句话给症状归类，再按本表对症下药，比逐个 google 报错栈快得多。

---

## 1. 地基：排错总思路（先归类，再处方）

大模型训练是一条长链路，任何一环断了都会报错。先建立"链路心智图"，看报错发生在哪一段，能极大缩小排查范围。

```
                  一次训练任务的链路（哪一段挂了？）
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ 环境/依赖 │ → │ 数据加载 │ → │ 模型加载 │ → │ 分布式   │ → │ 前向/反向│
  │ pip/cuda │   │ Dataset  │   │ from_    │   │ 通信初始 │   │ + 优化器 │
  │ /transf. │   │ DataLoader│  │ pretrained│  │ 化(NCCL) │   │ step     │
  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
       │              │              │              │              │
   §5 版本错      §6 worker      §5 tokenizer    §3 卡死/超时   §2 OOM
   ImportError    被 kill        /BnB 报错      address in use  §4 NaN/loss
```

排错四步法：

1. **先看是哪一段**：报错栈最底层那一行（`File ... line ...`）落在数据、模型还是优化器？
2. **看是否所有 rank 都报**：只有 rank 0 报 → 多半是数据/保存；所有 rank 都卡 → 多半是通信/同步。
3. **二分定位**：把 batch_size 调到 1、把数据集截到 100 条、把多卡降成单卡——能跑通就说明问题在被裁掉的那一维。
4. **最小复现**：单卡 + 小模型 + 几十条数据，能稳定复现再去改配置，不要在 8 卡大任务上反复试错（贵且慢）。

---

## 2. 显存炸了：CUDA out of memory（OOM）

### 2.1 显存都被谁吃了？

训练时显存 = **模型权重 + 梯度 + 优化器状态 + 激活值（activations）+ 框架碎片**。理解这五块，才知道该砍哪一块。

```
   单卡显存占用（混合精度 + Adam，按"占模型参数量的倍数"粗估）
   ┌───────────────────────────────────────────────┐
   │ 模型权重(fp16/bf16)        ~ 2×P 字节          │  P=参数量
   │ 梯度(fp16/bf16)            ~ 2×P 字节          │
   │ 优化器状态(Adam: m,v + fp32主权重) ~ 12×P 字节 │  ← 最大头！
   │ 激活值(activations)       随 batch×seq 增长    │  ← 第二大头，可变
   │ 框架/碎片/通信缓冲          数 GB              │
   └───────────────────────────────────────────────┘
   经验：全参 Adam 训练 ≈ 16×P 字节起步（7B → ~112GB，必须切分/offload）
```

> 直觉：Adam 给每个参数额外存一阶动量 $m$、二阶动量 $v$，混合精度下还要存一份 fp32 主权重，所以优化器状态往往比模型本身还大。这就是为什么"模型只有 7B、单卡 80G 还是装不下"。

### 2.2 降显存处方（按"性价比/代价"从低到高）

| 手段 | 砍的是哪块 | 代价 | 备注 |
| --- | --- | --- | --- |
| 调小 `batch_size` / `max_seq_length` | 激活值 | 吞吐下降 | 最快验证 OOM 的开关 |
| 梯度累积 `gradient_accumulation_steps` | 激活值（等效大 batch） | 步数变多、略慢 | 小 micro-batch 凑出大 global-batch |
| 梯度检查点 `gradient_checkpointing` | 激活值（用算力换显存） | 反向多算一遍，慢 ~20-30% | 长序列必备 |
| 用 **LoRA/QLoRA** 替代全参 | 梯度+优化器状态 | 表达能力略降 | 只训低秩适配器，显存断崖式下降 |
| **DeepSpeed ZeRO-2/3** | 梯度/优化器/权重切分到多卡 | 通信变多 | 见 §7，多卡时首选 |
| ZeRO **Offload** 到 CPU/NVMe | 把状态甩到内存/磁盘 | 极慢（受 PCIe 带宽限制） | 单卡装不下时的兜底 |

> README 里"lora 降显存"列的正是前几招：`gradient_checkpointing`、`gradient_accumulation_steps`、`batch_size`、`seq_length`。本节把它们的"作用对象"讲清，方便按需取舍。

### 2.3 排查要点

- **OOM 不一定在第一步**：训练几步后才 OOM，常是某条样本特别长（激活值随 seq 暴涨）→ 做长度排序/截断，或开 `gradient_checkpointing`。
- **碎片化（fragmentation）**：报错显示"已 reserved 很多但 allocated 不够"→ 设置环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（具体键名以官方文档为准）可缓解。
- **eval/save 阶段 OOM**：评估时 batch 可能更大、生成时 KV-Cache 占显存；保存 ZeRO-3 时要 gather 全量权重。单独调小 eval batch。

---

## 3. 多卡/分布式起不来、卡死、超时

分布式训练靠 NCCL 做集合通信，"卡死"几乎都出在初始化或某个 rank 掉队。

```
        torchrun / deepspeed 拉起 N 个进程（每卡一个 rank）
   rank0 ──┐
   rank1 ──┤   都要在 init_process_group 处"集合"  ←── 任何一个没到 → 全体 hang
   rank2 ──┤   靠 MASTER_ADDR:MASTER_PORT 互相发现
   rank3 ──┘   靠 NCCL 走 NVLink/PCIe/IB 通信
```

| 症状 | 根因 | 处方 |
| --- | --- | --- |
| `Address already in use` | `MASTER_PORT` 被上次残留进程占用 | 换端口；`nvidia-smi` 杀掉僵尸进程 |
| 启动即 hang，无报错 | 某 rank 没起来 / 卡数与 `gpu_num` 不符 | 核对进程数 = GPU 数；查防火墙/端口连通 |
| `NCCL timeout` / `Watchdog` | 某 rank 在做耗时操作（数据预处理/保存）超过同步窗口 | 加大超时时间；把重操作放到所有 rank 一起做 |
| `NCCL ... unhandled system error` | NCCL 走错网卡/IB 不可用 | 显式指定网卡（`NCCL_SOCKET_IFNAME`）、检查 IB；调试可设 `NCCL_DEBUG=INFO` 看日志 |
| 多机连不上 | `MASTER_ADDR` 不可达、NCCL 没走对网络 | ping 通主节点；多机 IB 见 [[ai-infra/网络/NCCL]] |

> 黄金法则：**所有 rank 必须执行相同次数的集合通信操作**。常见死锁是"只有 rank0 做保存/打印"里偷偷调用了带通信的算子，其它 rank 没调 → 全体 hang。

---

## 4. loss 异常：NaN / 不下降 / 爆炸

```
   正常       loss ↘↘↘ 平滑下降
   NaN/inf    loss = nan 突然出现 ──→ 梯度/数值溢出
   不降        loss 横线 ──────────→ lr 太小 / 数据或 mask 错 / 没真正在学
   爆炸        loss ↗↗↗ 飞了 ──────→ lr 太大 / 没做梯度裁剪
```

| 症状 | 常见根因 | 处方 |
| --- | --- | --- |
| loss 变 `nan/inf` | fp16 下数值溢出；学习率过大；脏数据 | 改用 **bf16**（动态范围大、更稳）；开梯度裁剪 `gradient_clipping`；查 label 是否含非法值 |
| loss 一开始就很大且不动 | label/mask 错位（如把 prompt 也算进 loss） | 检查 SFT 的 loss mask：只对 response 段算 loss |
| loss 几乎不降 | lr 太小 / warmup 没结束 / 数据没洗 | 调 `learning_rate`、`warmup_ratio`；确认 dataset 真有梯度信号 |
| loss 周期性尖刺 | 某些 batch 含超长/异常样本 | 长度分桶、过滤异常样本、加大梯度裁剪 |

> fp16 vs bf16：两者都用 2 字节，但 bf16 指数位多、表示范围大（约 $10^{38}$），代价是尾数精度低。大模型训练**优先 bf16**（A100/H100/910 等支持），能避开绝大多数 NaN。配置里 `"bf16": {"enabled": "auto"}` 即让框架自动开启。

---

## 5. 模型/Tokenizer 加载报错（环境与版本）

这一类是 `transformers`/依赖版本与模型权重不匹配导致，**最易踩、也最易修**——锁版本即可。

| 报错 | 根因 | 处方 |
| --- | --- | --- |
| `'BitsAndBytesConfig' object is not subscriptable` | 量化配置 API 与 `transformers`/`bitsandbytes` 版本不匹配（Baichuan2 早期建模代码） | 见官方 discussion；对齐 transformers 与 bitsandbytes 版本 |
| `'BaichuanTokenizer' object has no attribute 'sp_model'` | 新版 transformers 改了 tokenizer 初始化顺序，老建模代码不兼容 | 降到兼容版本 `pip install transformers==4.34.0`（以模型卡说明为准） |
| `trust_remote_code` 相关报错 | 自定义建模代码需显式信任 | `from_pretrained(..., trust_remote_code=True)` |
| `size mismatch` / 权重 shape 对不上 | 模型结构与 config 不一致、断点与代码版本错配 | 用配套的 config 与建模代码；别混用不同来源权重 |
| `ImportError: cannot import name ...` | transformers/peft/accelerate 互相版本不兼容 | 用一套经过验证的版本组合；固定到 `requirements` |

> 处方总纲：**模型卡（model card）/官方 discussion 给的版本组合是唯一权威**。本仓库不写死具体版本号，遇到就去模型 HF 页面对照。锁版本（pin version）后写进环境，避免下次重装又踩。

---

## 6. 数据加载：慢、worker 被 kill、共享内存

```
   主进程 ──fork──> DataLoader worker0 ─┐
                    DataLoader worker1 ─┼─> 各自读数据/做预处理 ──> 通过共享内存(/dev/shm) 把 batch 传回主进程
                    workerN ───────────┘
                    ↑ 共享内存太小 → worker 被 OS 以 SIGKILL 杀掉
```

| 报错 | 根因 | 处方 |
| --- | --- | --- |
| `DataLoader worker (pid xxx) is killed by signal: Killed` | 容器 **共享内存 /dev/shm 太小**，worker 传 batch 时 OOM 被内核杀 | 启动容器加 `--shm-size 4G`（或更大） |
| 数据加载很慢，GPU 利用率低 | `num_workers` 太小 / 预处理太重 / IO 慢（如 S3 直读） | 加大 `num_workers`、`prefetch`、`pin_memory`；预处理离线化、数据落本地盘 |
| `RuntimeError: unable to mmap` / `Bus error` | 同样是 /dev/shm 不足 | 同上，扩 shm |
| 卡在 epoch 之间 | worker 反复重建 | 设 `persistent_workers=True` |

> 为什么是"共享内存"？多进程 DataLoader 用 `/dev/shm` 这块共享内存把整理好的 batch 张量从 worker 传回主进程。Docker 默认 `/dev/shm` 只有 64MB，大 batch 一传就爆——所以容器跑训练几乎必加 `--shm-size`。

---

## 7. DeepSpeed ZeRO / Offload 相关坑

ZeRO（Zero Redundancy Optimizer）把"梯度/优化器状态/参数"按 stage 逐级切分到各卡，消除冗余；Offload 进一步把它们甩到 CPU 内存甚至 NVMe。本目录的 `zero3-offload.json` 就是一个 ZeRO-3 + CPU offload 配置。

```
   ZeRO 三级切分（消除多卡间的重复存储）
   ─────────────────────────────────────────
   ZeRO-1 : 切【优化器状态】           省最多、通信最少
   ZeRO-2 : 再切【梯度】               省更多
   ZeRO-3 : 再切【模型参数本身】       省最多、通信最重(参数用时现 all-gather)
   Offload: 把上面切出来的块放到 CPU/NVMe  单卡兜底，但极慢(受 PCIe 带宽限制)
```

`zero3-offload.json` 关键项的含义（**讲作用，不背默认值**）：

| 配置项 | 作用 | 权衡 |
| --- | --- | --- |
| `zero_optimization.stage: 3` | 选 ZeRO-3，连参数也切分 | 省显存最多，但通信开销最大 |
| `offload_optimizer.device: cpu` | 优化器状态放 CPU 内存 | 省显存，但更新慢，吃 PCIe 带宽 |
| `offload_param.device: cpu` | 参数也放 CPU，用时再拉回 | 进一步省显存，更慢 |
| `overlap_comm: true` | 通信与计算重叠 | 提速，但占额外缓冲显存 |
| `contiguous_gradients: true` | 梯度连续存放，减碎片 | 利于带宽 |
| `stage3_gather_16bit_weights_on_model_save: true` | 保存时把切分的权重 gather 成完整模型 | 保存时显存/内存峰值升高 |
| 各种 `"auto"`（如 `train_batch_size`） | 让框架从训练脚本/HF Trainer 参数自动推导 | 省去手算，但要保证脚本里有对应参数 |

常见坑：

- **`"auto"` 推不出来**：DeepSpeed 配合 HF Trainer 时，`train_batch_size = micro_batch × grad_accum × world_size` 必须自洽，三者只能手填两个、另一个交给 auto，全填或冲突会报错。
- **Offload 后巨慢**：这是机制本身——参数/优化器在 CPU，每步要在 CPU↔GPU 间搬运。只有"单卡真的装不下"才用，能加卡/能 ZeRO-3 不 offload 就别 offload。
- **ZeRO-3 下取模型参数**：直接 `model.xxx.weight` 拿到的可能是空壳（参数被切分了），需用 DeepSpeed 提供的 gather 接口。
- **保存的是分片**：ZeRO 保存的是各 rank 分片，导出可用整模型时要做 consolidate（或开 `stage3_gather_16bit_weights_on_model_save`）。

> 更系统的 ZeRO 原理见 [[ai-framework/deepspeed/README]]。

---

## 8. 保存 / 恢复 Checkpoint

| 症状 | 根因 | 处方 |
| --- | --- | --- |
| 保存时 OOM / 内存暴涨 | ZeRO-3 保存要 gather 全量权重到单卡/CPU | 用 sharded 保存；或确保 CPU 内存足够 |
| 恢复后 loss 跳变 | 没恢复优化器状态/学习率调度器/随机种子 | 恢复完整 state（optimizer + scheduler + rng），别只 load 权重 |
| 只想要可部署的 HF 权重 | 训练 checkpoint 含 DeepSpeed 分片，不能直接推理 | 用脚本把 ZeRO 分片转成 `pytorch_model.bin`/safetensors |
| 多机保存只有部分文件 | 各 rank 写到了本地盘而非共享存储 | checkpoint 路径放共享存储（NFS/对象存储，如本目录用的 S3） |

> 实践口诀：**断点续训要"三件套全恢复"**——权重、优化器状态、调度器/步数。只恢复权重等于丢掉了动量和 lr 进度，loss 会抖。

---

## 9. 原始报错条目速查（保留 + 展开）

### 9.1 Baichuan2

- **`'BitsAndBytesConfig' object is not subscriptable`**
  量化配置 API 与依赖版本不匹配。参考官方 discussion：
  `https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/discussions/2`

- **`AttributeError: 'BaichuanTokenizer' object has no attribute 'sp_model'`**
  新版 transformers 与 Baichuan2 建模代码不兼容。降低版本到 4.34.0 及以下：
  `pip install transformers==4.34.0`
  （根因：新版改了 tokenizer 初始化流程，老的自定义 `BaichuanTokenizer` 在调用 `super().__init__` 前就访问了 `sp_model`，于是属性还没建好就被引用。锁版本是最稳的修法，具体兼容版本以模型卡为准。）

### 9.2 PyTorch

- **`RuntimeError: DataLoader worker (pid xxxxx) is killed by signal: Killed.`**
  大概率是**共享内存太小**。容器启动加：`--shm-size 4G`（详见 §6）。
  （根因：多进程 DataLoader 经 `/dev/shm` 回传 batch，Docker 默认 64MB 不够用，worker 被内核 OOM-Killer 杀掉。）

---

## 常见问题/坑（汇总表）

| 现象 | 第一反应 | 真正根因方向 |
| --- | --- | --- |
| CUDA out of memory | 调小 batch / seq，开 grad checkpointing | 优化器状态/激活值占大头，必要时上 LoRA/ZeRO，见 §2 §7 |
| 启动即 hang 不报错 | 核对进程数=GPU 数 | 某 rank 没起来或集合通信不对称，见 §3 |
| NCCL timeout | 加大超时 | 有 rank 在做耗时操作掉队，见 §3 |
| loss = nan | 换 bf16、开梯度裁剪 | fp16 溢出 / lr 过大 / 脏数据，见 §4 |
| loss 不降 | 查 lr 和 loss mask | SFT 把 prompt 也算进了 loss，见 §4 |
| tokenizer/BnB 报错 | 锁 transformers 版本 | 建模代码与依赖版本错配，见 §5 §9 |
| worker killed by signal | `--shm-size` 调大 | /dev/shm 不足，见 §6 §9 |
| Offload 后极慢 | 别 offload，能加卡就加卡 | CPU↔GPU 搬运受 PCIe 带宽限制，见 §7 |
| 保存的 checkpoint 不能直接推理 | 用转换脚本导 HF 权重 | ZeRO 存的是分片，见 §7 §8 |
| 断点续训 loss 抖 | 恢复"权重+优化器+调度器"三件套 | 只 load 权重丢了动量/lr 进度，见 §8 |

> 通用免责声明：本 FAQ 给的是**机制与排查方向**；具体 CLI 参数名、默认值、版本号、API 签名请以对应框架（PyTorch / transformers / DeepSpeed）的官方文档与本仓库脚本实际实现为准，不要凭记忆写死。

---

## 🔗 跳转链接

- 知识地图总览：[[00-知识地图]]
- DeepSpeed / ZeRO 原理：[[ai-framework/deepspeed/README]]
- 大模型训练总览：[[llm-train/README]]
- PyTorch 分布式训练：[[llm-train/pytorch/distribution/README]]
- Megatron-LM（更大规模并行）：[[ai-framework/megatron-lm/README]]
- NCCL 集合通信与网络：[[ai-infra/网络/NCCL]]
- 训练/推理性能指标名词：[[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- LLMOps 总览：[[llmops/README]]
