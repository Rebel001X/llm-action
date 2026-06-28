# Megatron-LM GPT-2 检查点合并/重切分（TP/PP）与推理服务

> 一句话定位：用 `checkpoint_util.py` 把一份并行度为 (TP, PP) 的 Megatron 权重「先合成完整权重、再按目标 (TP', PP') 重新切分落盘」，然后用 `run_text_generation_server.py` 起一个 Flask 推理服务对外提供文本生成。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/README]] · [[B07:llm-inference/大模型推理张量并行]] · [[ai-infra/网络/集合通信原语]] · [[llm-inference/连续批处理]]

## 阅读地图

| 小节 | 你会得到什么 | 关键词 |
|------|--------------|--------|
| 0 | 一句话锚点：这套脚本到底干了啥 | merge / repartition / serve |
| 1 | 前置：为什么 TP/PP 切分的权重不能直接换并行度加载 | 分片布局 / mp_rank 目录 |
| 2 | 总架构：loader→queue→saver 的「生产者-消费者」管道 | 多进程 / 全量权重中转 |
| 3 | TP 切分的数学：哪些权重按行切、哪些按列切、为什么 | Column/Row Parallel |
| 4 | PP 切分的数学：层怎么分到不同 stage | 流水线 / 层映射 |
| 5 | 合并命令逐参数解读 + 落盘目录结构 | checkpoint_util.py |
| 6 | 推理服务：4 个启动脚本 + REST/CLI 调用 | flask / torchrun |
| 关键公式 | 分片维度、显存账、数值手算 | $d_{shard}$ |
| 评价 | 何时合、何时拆、坑位对照 | 局限 |

## 0. 一句话锚点

训练时为了塞进显存，GPT-2 被切成了 **张量并行 TP × 流水并行 PP** 的若干分片，落盘成 `mp_rank_xx[_yyy]/model_optim_rng.pt`。当你想：

- 换一套硬件（比如训练用 4 卡、推理只想用 1 卡）；
- 或把模型导出给别的框架；

就必须**先把分片拼回完整权重，再按新并行度重新切**。`checkpoint_util.py` 就是这个「重切分工具」（repartition / resharding）。完成后用 `run_text_generation_server.py` 起 HTTP 服务做推理。

本目录文件清单：

```
merge_ck_and_inference/
├── checkpoint_util.py                          # 入口：拉起 loader + saver 两个进程
├── checkpoint_loader_megatron.py               # loader 插件：读分片→拼全量→丢进队列
├── checkpoint_saver_megatron.py                # saver 插件：从队列取全量→按新 TP/PP 切→落盘
├── run_text_generation_server.py               # Flask 文本生成服务入口
├── run_text_generation_server_345M.sh          # 起服务：TP=1 PP=1（合并后单卡）
├── run_text_generation_server_345M_2tp_2dp.sh  # 起服务：TP=2 PP=2
├── run_text_generation_server_345M_4_tensor_parallel.sh  # 起服务：TP=4 PP=1
├── text_generation_cli.py                      # 命令行客户端（交互式输入 prompt）
└── eval_gpt2_lambada.sh                         # 在 LAMBADA 上评测合并后的模型
```

## 1. 地基：为什么不能「直接换并行度加载」

Megatron 把每个 GPU 上**那一份分片**单独存盘。以 345M（24 层、hidden=1024、heads=16）为例，落盘目录因并行度而异：

```
TP=2,PP=2  ->  mp_rank_00 mp_rank_01 mp_rank_02 mp_rank_03   (4 份, 每份只有 1/4 权重)
TP=1,PP=1  ->  mp_rank_00                                    (1 份, 完整 1.3G)
TP=2,PP=1  ->  mp_rank_00 mp_rank_01                         (2 份, 各 ~680M)
TP=1,PP=4  ->  mp_rank_00_000 .. mp_rank_00_003              (4 份, 按层切, 大小不均)
```

> 命名规律（以官方实现为准）：`mp_rank_{tp:02d}` 是纯 TP；`mp_rank_{tp:02d}_{pp:03d}` 带 PP 时多一段 stage 编号。

**核心矛盾**：分片里的权重张量是「被切过一刀的半成品」。比如 attention 的 QKV 投影矩阵，TP=2 时每张卡只持有「一半的 head」对应的列。直接用 TP=1 去 load，形状对不上、语义也错。所以必须经过「**先 all-gather 成完整张量，再 chunk 成新份数**」。这正是本工具做的事。

## 2. 总架构：loader → queue → saver 的多进程管道

`checkpoint_util.py` 本质是一个**两进程生产者-消费者**：

```
   ┌──────────────────────┐        mp.Queue (maxsize=50)        ┌──────────────────────┐
   │   loader 进程         │   put(完整张量, 按固定协议顺序)      │   saver 进程          │
   │  (主进程里直接跑)     │ ─────────────────────────────────▶ │ (mp.Process 子进程)   │
   │                       │                                     │                       │
   │ 1. 读旧分片           │   {"name":"embeddings", ...}        │ 1. 收 metadata        │
   │ 2. all-gather 拼全量  │   {"name":"transformer layer 0"}    │ 2. 按新 TP chunk 张量 │
   │ 3. 一条条 put 到队列  │   ...                               │ 3. 按新 PP 分到 stage │
   │ 4. 最后 put "done"    │   {"name":"final layer norm"}       │ 4. 写 model_optim_rng │
   └──────────────────────┘   "done"                            └──────────────────────┘
```

为什么用队列中转、而不是直接函数调用？

- **解耦**：loader 只懂「怎么读旧格式」，saver 只懂「怎么写新格式」。两者通过一个固定的**消息协议**通信（见 `checkpoint_util.py` 顶部注释：embeddings → 每层 transformer → final layernorm → lm head → done）。这样支持 GPT/BERT/T5 用不同 loader/saver 插件组合。
- **限流**：`mp.Queue(maxsize=50)` 限制在途张量数，避免 loader 把整个模型一次性铺进内存。
- **流水重叠**：loader 在拼下一层时，saver 可以同时在切/写上一层（输出日志里 `sending layer 12` 和 `received layer 10` 交错出现就是证据）。

关键代码（`checkpoint_util.py`）：

```python
queue = mp.Queue(maxsize=args.max_queue_size)        # 默认 50
saver_proc = mp.Process(target=saver.save_checkpoint, args=(queue, args))
saver_proc.start()                                    # 先起消费者
loader.load_checkpoint(queue, args)                   # 主进程当生产者
saver_proc.join()                                     # 等 saver 落盘完成
```

> 协议里有一句话很重要：「the weight sent over the queue are the **full model weights, nothing split**」。即队列里跑的永远是**未切分的完整张量**，TP 的切分完全发生在 saver 一侧。这把「重切分」简化成了「拼全 → 切新」两个独立步骤。

## 3. TP 切分的数学：谁按行切、谁按列切

这是整个工具最容易踩坑的地方。Transformer 里两类线性层的切法**正好相反**（参见 Megatron 原论文）：

```
                  Column Parallel (按输出维/列切)        Row Parallel (按输入维/行切)
                  ┌─────────────────────┐                ┌─────────────────────┐
   用于           │ QKV 投影、MLP 第1层  │                │ Attn dense、MLP 第2层│
                  │ (fc1, 把 h 升到 4h)  │                │ (fc2, 把 4h 降回 h)  │
   切分维度       │ dim=0  (输出通道)    │                │ dim=1  (输入通道)    │
   前向通信       │ 无 (各算各的)        │                │ all-reduce 求和      │
                  └─────────────────────┘                └─────────────────────┘
```

在 `checkpoint_saver_megatron.py` 里这一切非常直白：

```python
qkv_weight   = torch.chunk(msg.pop("qkv weight"),   target_tp, dim=0)   # 列切：每卡一组 head
dense_weight = torch.chunk(msg.pop("dense weight"), target_tp, dim=1)   # 行切
mlp_l0_weight= torch.chunk(msg.pop("mlp l0 weight"),target_tp, dim=0)   # fc1 列切
mlp_l1_weight= torch.chunk(msg.pop("mlp l1 weight"),target_tp, dim=1)   # fc2 行切
out_word_embed = torch.chunk(full_word_embed,       target_tp, dim=0)   # 词表沿 vocab 维切
```

直觉记忆法：

```
   x ──[fc1 列切]──▶ [各卡持有部分 4h 通道，互不依赖]──[fc2 行切]──▶ ⊕ all-reduce ──▶ y
        ColumnParallel                                    RowParallel
   先把维度「拆宽」让各卡并行算，再「拍扁求和」拼回来——一进一出刚好一次 all-reduce。
```

- **QKV 列切**：注意力按 head 天然可并行，TP=k 就是把 16 个 head 分给 k 张卡。所以 `dim=0`（输出维=heads×head_dim）。
- **dense（attn 输出投影）行切**：它的输入正是上面各卡的部分 head 输出，所以按输入维 `dim=1` 切，前向需 all-reduce 把各卡贡献加起来。
- **word_embedding 沿 vocab 维（dim=0）切**：词表大、且和 LM head 共享权重，需要 padding 到能被 TP 整除（日志里 `padded vocab 50257 -> 50304/50432` 就是 `make_vocab_size_divisible_by` 在起作用）。
- **LayerNorm / bias（部分）是「复制」而非切分**：每张卡各存一份完整副本。

> 反过来「合并」(merge, TP→1) 就是把这些 `chunk` 的逆操作：loader 端把同名分片 `torch.cat` 回去（列切的 cat dim=0，行切的 cat dim=1），得到完整张量再丢进队列。

## 4. PP 切分的数学：层如何映射到 stage

PP（pipeline parallel）不切单个张量，而是**把 24 层整层整层地分给不同 stage**。345M 重切到 PP=4 的落盘大小印证了这点：

```
mp_rank_00_000 -> 489M   (stage0: 含 word/pos embedding + 前 6 层)   ← 多了 embedding 所以大
mp_rank_00_001 -> 288M   (stage1: 中间 6 层)
mp_rank_00_002 -> 288M   (stage2: 中间 6 层)
mp_rank_00_003 -> 485M   (stage3: 后 6 层 + final LN + LM head)      ← 多了输出头所以大
```

层映射公式：第 $i$ 层（$0 \le i < L$，$L$=24）落到 stage

$$\text{stage}(i) = \left\lfloor \frac{i}{L / P} \right\rfloor,\quad P = \text{pipeline\_parallel\_size}$$

PP=4 时每段 $L/P = 6$ 层。首段额外背 embedding、末段额外背 LM head，所以两头的 `.pt` 文件更大、中间均匀。这也是为什么 PP 的负载均衡常需要手动微调每段层数（Megatron 提供 `--num-layers-per-virtual-pipeline-stage` 等参数，本工具默认均分）。

## 5. 合并/重切分命令逐参数解读

最常见用法：把 TP=2,PP=2 的训练产物**合并成 TP=1,PP=1 的单卡可加载权重**。

```bash
python tools/checkpoint_util.py \
        --model-type GPT \
        --load-dir /workspace/model/megatron-models/345m-init-mp \
        --save-dir /workspace/model/megatron-models/345m-init-mp-out \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1
```

| 参数 | 含义 | 备注 |
|------|------|------|
| `--model-type` | `GPT` / `BERT`，决定加载哪套 loader/saver 协议 | T5 也支持，以官方实现为准 |
| `--loader` / `--saver` | 插件模块名，默认 `megatron` → `checkpoint_{loader,saver}_megatron.py` | 想导出别的格式就换 saver |
| `--load-dir` | 旧分片所在目录（含 `latest_checkpointed_iteration.txt`） | 工具自动从 ckpt 读 num_layers/hidden 等结构，无需手填 |
| `--save-dir` | 新分片输出目录 | 会写出 `iter_xxxx/mp_rank_*/model_optim_rng.pt` |
| `--target-tensor-parallel-size` | **目标** TP；不填则沿用旧 TP | 合并→1，拆分→更大值 |
| `--target-pipeline-parallel-size` | **目标** PP；不填则沿用旧 PP | |
| `--max-queue-size` | 在途张量上限，默认 50 | 内存吃紧可调小 |
| `--no-checking` | 关闭消息名/顺序校验 | 默认开校验，能抓出协议错位 |

落盘结果（合并到 TP=1,PP=1）：

```
345m-init-mp-out/
├── iter_0005000/
│   └── mp_rank_00/
│       └── model_optim_rng.pt   (1.3G, 完整权重)
└── latest_checkpointed_iteration.txt   (内容: "5000")
```

> saver 内部会**伪造 WORLD_SIZE = target_tp × target_pp** 来骗过 Megatron 的 world-size 合法性检查，并强制 `--use-cpu-initialization --no-initialization`（不真初始化、不占显存），这样在**纯 CPU、无 GPU** 的机器上也能跑重切分。这是该工具能在 CI / 小机器上运行的关键。

其它常见目标（命令同形，改两个 `--target-*` 即可）：

```bash
# 拆成 TP=2,PP=1  ->  mp_rank_00 / mp_rank_01 各 ~680M
--target-tensor-parallel-size 2 --target-pipeline-parallel-size 1
# 拆成 TP=1,PP=4  ->  mp_rank_00_000..003，大小不均（见第4节）
--target-tensor-parallel-size 1 --target-pipeline-parallel-size 4
```

## 6. 推理服务：起 server + 调用

合并/重切分完成后，用对应并行度的脚本起服务。**启动脚本里的 `--tensor-model-parallel-size / --pipeline-model-parallel-size 必须与权重目录的并行度一致**，否则形状对不上。

```
┌─────────────────────────────────────────────────────────────────────┐
│  权重并行度        →   对应启动脚本                  →   nproc_per_node │
├─────────────────────────────────────────────────────────────────────┤
│  TP=1 PP=1 (合并后)    run_text_generation_server_345M.sh                1   │
│  TP=4 PP=1            ..._345M_4_tensor_parallel.sh                      4   │
│  TP=2 PP=2            ..._345M_2tp_2dp.sh (注:实为2tp2pp)                4   │
└─────────────────────────────────────────────────────────────────────┘
```

单卡（TP=1）启动核心命令：

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1          # Megatron 推荐，保证 kernel 提交顺序
pip install flask-restful
torchrun --nproc_per_node 1 --master_port 6000 tools/run_text_generation_server.py \
       --tensor-model-parallel-size 1  --pipeline-model-parallel-size 1 \
       --num-layers 24 --hidden-size 1024 --num-attention-heads 16 \
       --max-position-embeddings 1024 --seq-length 1024 --out-seq-length 1024 \
       --load ${CHECKPOINT} --tokenizer-type GPT2BPETokenizer --fp16 \
       --micro-batch-size 1 --temperature 1.0 --top_p 0.9 --seed 42 \
       --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE
```

> 注意 `..._2tp_2dp.sh` 文件名写的是「2dp」，但脚本里实际是 `--tensor-model-parallel-size 2 --pipeline-model-parallel-size 2`（即 2TP×2PP，world_size=4，data-parallel=1）。命名有误导，**以脚本内参数为准**。

服务起来后默认监听 `0.0.0.0:5000`，提供 `PUT /api`。三种调用方式：

```bash
# 1) curl 直接打
curl 'http://localhost:5000/api' -X PUT -H 'Content-Type: application/json' \
     -d '{"prompts":["Hello world"], "tokens_to_generate":1}'
# -> {"logprobs":null,"segments":[["Hello"," world",","]],"text":["Hello world,"]}

# 2) 自带交互式 CLI（text_generation_cli.py）
python tools/text_generation_cli.py localhost:5000
#   Enter prompt: hello
#   Enter number of tokens to generate: 5
#   Megatron Response: hello! Until that protagonist receive
```

`text_generation_cli.py` 逻辑极简：把 `{"prompts":[句子], "tokens_to_generate":N}` POST(PUT) 到 `/api`，取回 `response.json()['text'][0]`。

评测：`eval_gpt2_lambada.sh` 用合并后的权重在 LAMBADA 上跑 `tasks/main.py --task LAMBADA --strict-lambada`，注意它带 `--no-load-optim --no-load-rng`（只评测、不需要优化器状态）。

## 关键公式 / 数值手算

**(1) 单张 TP 卡分到的 QKV 权重形状**
QKV 投影完整权重形状 $[3 H, H]$（$H$=hidden=1024）。TP=$k$ 沿 `dim=0` 切：

$$\text{每卡 QKV} = \left[\frac{3H}{k},\, H\right]$$

TP=4：每卡 $[3072/4, 1024] = [768, 1024]$，即每卡负责 $16/4 = 4$ 个 head。

**(2) 重切分对单分片体积的影响（345M 实测）**

$$\text{单分片} \approx \frac{\text{总参数} \times \text{bytes}}{\text{TP} \times \text{PP}} + \text{每卡复制的部分}$$

| 目标并行度 | 分片数 | 单片大小（约） | 解释 |
|-----------|--------|----------------|------|
| TP1×PP1 | 1 | 1.3 G | 全量 fp16+部分fp32 |
| TP2×PP1 | 2 | 680 M | ≈1.3G/2 |
| TP1×PP4 | 4 | 489/288/288/485 M | 两头带 embed/head 偏大 |

注意 TP2 时 2×680M=1.36G > 1.3G：因为 LayerNorm/bias 等是**复制**的，切得越多复制冗余越多，**总落盘略增**。

**(3) 推理显存粗估（参数部分）**
fp16 下参数显存 $\approx 2 \times P$ bytes。345M：$2 \times 3.55\times10^8 \approx 0.71$ GB（仅权重，不含 KV cache / 激活）。TP=4 时每卡只需 $\approx 0.18$ GB 权重——这就是合并/拆分换并行度的现实意义：用更多小卡分摊，或合并到单张大卡省通信。

## 评价 / 对照 / 局限

| 维度 | 说明 |
|------|------|
| 何时**合并**(→1) | 导出给 HF/vLLM、单卡推理、做离线评测；消除分片管理负担 |
| 何时**拆分**(→大) | 单卡放不下、想用 TP 降低单卡显存或用 PP 跨机 |
| 优点 | 纯 CPU 可跑、不依赖 GPU；插件式支持多模型；流式低内存 |
| 局限1 | 只重切 **TP/PP**，不改模型结构/不做量化；DP 不影响落盘（DP 只是数据副本） |
| 局限2 | `2tp_2dp.sh` 命名与实际参数不符，易误导，**以脚本内 `--*-parallel-size` 为准** |
| 局限3 | 协议顺序严格，loader/saver 版本要匹配；错位会在 `check_message` 处直接 exit |
| 局限4 | 词表会按 `make_vocab_size_divisible_by` 被 padding，导出到其它框架时需注意还原 50257 |
| 数值护栏 | 文中 1.3G/680M/489M 等为该仓库示例输出的实测值；显存/FLOPs 为按公式估算，精确值以实际硬件与 Megatron 版本为准 |

## 🔗 跳转链接

- [[00-知识地图]]
- [[llm-train/README]] · [[llm-train/pytorch/distribution/README]]
- [[B07:llm-inference/大模型推理张量并行]]（TP 切分细节）
- [[ai-infra/网络/集合通信原语]]（all-reduce / all-gather 是 TP/PP 的通信底座）
- [[llm-inference/vllm/README]]（合并后的权重常导给 vLLM 做高吞吐推理）
- [[llm-inference/连续批处理]] · [[llm-inference/KV-Cache优化]]
- [[llm-algo/transformer/模型架构]]（QKV / MLP 两类线性层是切分对象）
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]
- [[docs/transformer内存估算]]（参数/激活/KV cache 显存账）
