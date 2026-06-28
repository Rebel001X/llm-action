# Megatron-DeepSpeed 源码精读：数据集构建调用链

> 顺着一行 `build_train_valid_test_datasets(...)` 往下钻，看清 Megatron-DeepSpeed 是怎么把磁盘上的二进制 token 流，组装成训练循环每个 step 直接能吃的 `(tokens, labels, loss_mask, attention_mask, position_ids)` 张量的。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-train/megatron-deepspeed/README]] [[llm-train/megatron-deepspeed/microsoft/代码]] [[llm-train/pytorch/distribution/README]] [[llm-train/README]]

---

## 阅读地图

| 你想搞清楚的问题 | 跳到哪一节 |
| --- | --- |
| 训练入口是怎么"要"到数据的? | [§1 入口与三件套](#1-入口与三件套dataset-provider) |
| `build_*` 和带下划线的 `_build_*` 啥区别? | [§2 两层函数为什么分两层](#2-两层为什么分两层public-vs-private) |
| 单数据集和多数据集走的代码分叉在哪? | [§3 分叉点单数据集-vs-多数据集](#3-分叉点单数据集-vs-多数据集) |
| 905,5,90 这种 split 怎么切的? | [§4 split三段如何切片](#4-split三段如何被切成索引区间) |
| .bin / .idx 二进制到底长啥样? | [§5 底层indexed-dataset与-mmap](#5-底层indexedmmap-dataset二进制布局) |
| 一条样本怎么从 token 流里抠出来? | [§6 GPTDataset一条样本的诞生](#6-gptdataset一条样本是怎么诞生的) |
| 多数据集按比例混合是怎么实现的? | [§7 BlendableDataset按比例混采](#7-blendabledataset按比例混采) |
| 通信/显存/采样数怎么算? | [关键数值示例](#关键算法与数值示例) |
| 和纯 Megatron / HF datasets 怎么比? | [对照与局限](#对照对比与局限) |

---

## 0. 一句话锚点

**一句话：** `build_train_valid_test_datasets` 是数据侧的总入口,它根据你传的 `--data-path` 是「一个路径前缀」还是「权重+路径交替的列表」,决定走**单数据集**(直接造一个 `GPTDataset`)还是**多数据集**(给每个子数据集各造一个,再用 `BlendableDataset` 按权重混采),返回 `train/valid/test` 三个 `Dataset` 对象;真正"从二进制里取 token"这件脏活,落在最底层的 **mmap indexed dataset** 上。

> 类名、函数签名、参数默认值以你 clone 的那个 commit 为准。本文讲**调用链与机制**,不背死细节;涉及具体行为时标注"以源码为准"。

---

## 1. 入口与三件套：dataset provider

训练主循环(`megatron/training.py` 的 `pretrain()`)在 setup 阶段会调用一个**你传进去的回调** `train_valid_test_dataset_provider`(在 `pretrain_gpt.py` 里定义)。这个回调内部就调 `build_train_valid_test_datasets`。

```
pretrain_gpt.py: train_valid_test_datasets_provider(train_val_test_num_samples)
        │  把 args 里的 data-path / seq-length / split 等揉成参数
        ▼
megatron/data/gpt_dataset.py: build_train_valid_test_datasets(
        data_prefix,            ← --data-path,可能是单个 or 列表
        splits_string,          ← --split,如 "949,50,1"
        train_valid_test_num_samples,  ← 三段各要采多少条样本
        seq_length, seed, ...)
        ▼
    返回 (train_ds, valid_ds, test_ds)  ← 三个 torch Dataset
```

**为什么要"要多少条样本(num_samples)"而不是"要多少 epoch"?**
大模型预训练通常**不按 epoch 走**,而是按 `train_iters × global_batch_size` 反推出总共需要多少条样本。数据可能被**重复采样**(epoch > 1)也可能**用不到一遍**(epoch < 1)。所以入口必须显式告诉 dataset"我要 N 条",而不是"我要几轮"——这是和 HuggingFace `datasets` 思路最大的不同。

```
train_val_test_num_samples = [
    train_iters        * global_batch_size,   # 训练要采的总条数
    eval_iters * (train_iters//eval_interval) * gbs,  # 验证(近似,见源码)
    eval_iters         * gbs ]                 # 测试
```

---

## 2. 两层：为什么分两层(public vs private)

源码里你会看到**成对**出现的两个函数(这正是空文件里那行提示的来源):

```
build_train_valid_test_datasets          ← 公开入口(对外)
   └─ _build_train_valid_test_datasets   ← 私有实现(干活)
```

分两层的**唯一目的是处理"单 vs 多"这层分叉**:

```
build_train_valid_test_datasets(data_prefix, ...):
    if data_prefix 是单个路径(len==1):
        return _build_train_valid_test_datasets(单路径, ...)   # 直接干活
    else:
        # 多数据集:解析出 [权重, 路径, 权重, 路径, ...]
        prefixes, weights = parse_and_normalize_blend(data_prefix)
        # 对每个子数据集分别调私有函数
        for prefix in prefixes:
            t, v, te = _build_train_valid_test_datasets(prefix, 按权重缩放后的num_samples, ...)
            堆进 train_datasets/valid_datasets/test_datasets
        # 再用 BlendableDataset 把它们按 weights 缝起来
        return (BlendableDataset(train_datasets, weights, ...),
                BlendableDataset(valid_datasets, weights, ...),
                BlendableDataset(test_datasets,  weights, ...))
```

> 记忆:**带下划线的 `_build_*` 只认识"一个数据集"**;不带下划线的 `build_*` 负责"要不要把多个 `_build_*` 的结果混起来"。

```
            data_prefix
                │
     ┌──────────┴───────────┐
   单路径                  多路径(带权重)
     │                       │
 _build_*(一次)        循环 _build_*(N 次) ──► BlendableDataset 混采
     │                       │
   GPTDataset×3          (GPTDataset×3)×N  ─缝─► Blendable×3
```

---

## 3. 分叉点：单数据集 vs 多数据集

`--data-path` 的两种写法,决定了走哪条分支:

```
# 单数据集:一个前缀(对应 my-corpus_text_document.bin / .idx)
--data-path /data/my-corpus_text_document

# 多数据集:权重 路径 权重 路径 ... 交替
--data-path 0.3 /data/wiki_text_document  0.7 /data/books_text_document
```

`_build_train_valid_test_datasets`(私有,处理**单个**)内部做三件事:

1. **建底层 indexed dataset**:`get_indexed_dataset_(prefix, ...)`,把 `.bin/.idx` mmap 进来,得到一个能"按 doc 索引取 token 序列"的对象,并拿到**文档总数 / token 总数**。
2. **算 split 区间**:`get_train_valid_test_split_(splits_string, total_num_docs)`,把 `[0, total_docs)` 按比例切成三段(见 §4)。
3. **为每段造一个 `GPTDataset`**:用一个内部 `build_dataset(index, name)` 闭包,把"这段文档区间 + 要采多少条样本"打包成一个真正的 `GPTDataset`(见 §6),分别给 train/valid/test。

```
_build_train_valid_test_datasets(单 prefix):
   indexed_ds = get_indexed_dataset_(prefix)        # ① mmap .bin/.idx
   splits     = get_train_valid_test_split_(...)    # ② 切文档区间
   def build_dataset(idx, name):                    # ③ 闭包造 GPTDataset
        documents = arange(splits[idx], splits[idx+1])
        return GPTDataset(name, prefix, documents, indexed_ds,
                          num_samples[idx], seq_length, seed)
   return build_dataset(0,'train'), build_dataset(1,'valid'), build_dataset(2,'test')
```

---

## 4. split：三段如何被切成索引区间

`--split "949,50,1"` 不是"949 条 / 50 条 / 1 条",而是**比例**(会被归一化成 0.949 / 0.050 / 0.001)。切的是**文档(document)下标区间**,不是 token,也不是样本。

`get_train_valid_test_split_` 大致逻辑:

```python
splits = [int(s) for s in splits_string.split(',')]   # [949,50,1]
splits_sum = sum(splits)                              # 1000
# 归一化后乘以文档总数,得到每段的"文档数",再前缀和成切点
index = [0]
for s in splits:
    index.append(index[-1] + round(s / splits_sum * total_num_docs))
# index = [0, train_end, valid_end, test_end(=total_docs)]
```

```
 文档 0 ───────────────────────────────── total_docs
 │██████████ train(94.9%) ██████████│val│t│
 0                          train_end   ↑   ↑
                                   valid_end test_end
```

**关键点 / 易踩坑:**
- 切的是**文档边界**,所以 valid/test 看到的是**完全不同的文档**,不会和 train 有 token 级泄漏(前提是文档本身不重复)。
- 比例 → 文档数用 `round`,极端比例(如 "1")在小语料上可能**取整成 0 个文档**,导致 valid/test 为空 → 报错。多数据集时这点更隐蔽。
- 这是**确定性**切分:同样的 `splits_string` + 同样的文档总数,切点完全可复现,无需固定随机种子。

---

## 5. 底层：indexed/mmap dataset 二进制布局

预处理脚本 `tools/preprocess_data.py` 把 `corpus.jsonl` 变成**一对文件**:

```
my-corpus_text_document.bin   ← 所有文档 token id 首尾相接的紧凑流(无分隔)
my-corpus_text_document.idx   ← 元数据:每篇文档的 (起始指针, 长度)、dtype 等
```

`MMapIndexedDataset` 用 `numpy.memmap` 把 `.bin` 映射进**虚拟地址空间**,不真正读进内存——只有访问到某段时,OS 缺页才把对应 4KB 页从磁盘调进来。

```
.bin (磁盘):  [doc0 tok...][doc1 tok...][doc2 tok...] ...   一条连续 uint16/uint32 流
                ▲              ▲
.idx 记录:  doc0: ptr=0,len=L0   doc1: ptr=L0,len=L1  ...
                │
   ds[i]  ──►  从 ptr[i] 起读 len[i] 个 token  ──► np.array
```

**为什么用 mmap 而不是一次读进内存?**
- 语料 `.bin` 动辄几百 GB ~ TB,**装不进内存**;mmap 让"按需分页"自然解决。
- **多进程 DataLoader worker 共享同一份页缓存**(page cache),N 个 worker 不会把同一文件读 N 遍 → 省内存、省 IO。
- 随机访问 `ds[i]` 是 O(1):先查 `.idx` 拿指针,再切 `.bin` 一段,无需扫描。

> 代价:首个 epoch 因为缺页会有冷启动 IO 抖动;之后页缓存热了就快。这也是为什么 `--data-path` 最好放在**本地高速盘**而非网络盘。

---

## 6. GPTDataset：一条样本是怎么诞生的

`GPTDataset.__init__` 里会预先算好**三张索引表**(并缓存到磁盘 `.npy`,第二次启动直接 load,省掉重算):

| 索引表 | 作用 | 形状/含义 |
| --- | --- | --- |
| `doc_idx` | 把"本段文档"按需重复/打乱,排成一条逻辑文档流 | 长度 ≈ epoch 数 × 文档数 |
| `sample_idx` | 标记**第 i 条样本**在那条 token 长流里的 (起点 doc, 起点 offset) | (num_samples+1, 2) |
| `shuffle_idx` | 把样本顺序整体打乱 | 长度 = num_samples |

`__getitem__(idx)` 的流程:

```
idx ──shuffle_idx──► 打乱后的样本号 s
   │
   ├─ sample_idx[s]   = (doc_start, offset_start)
   ├─ sample_idx[s+1] = (doc_end,   offset_end)
   │
   ▼  从 doc_start 跨到 doc_end,拼出 (seq_length+1) 个 token
 sample = [t0, t1, ..., t_{S}]          ← 长度 S+1 = seq_length+1
   │
   ├─ tokens = sample[:-1]   ← 输入(0..S-1)
   ├─ labels = sample[1:]    ← 目标(右移一位,next-token)
   └─ loss_mask / position_ids / attention_mask 由 collate 阶段补
```

**关键设计:为什么要 `seq_length + 1`?**
因果语言模型用"预测下一个 token"训练。一条长 $S+1$ 的片段,前 $S$ 个当输入 `tokens`,后 $S$ 个(右移一位)当 `labels`,正好对齐 next-token 任务,不浪费任何 token。

**为什么样本可以跨文档?**
GPT 预训练把整段语料当作一条"无穷长 token 流",样本是在这条流上**滑窗截取**,允许一条样本横跨两篇文档(中间靠 EOD/eos token 隔开,模型自己学边界)。这和 BERT 那种"一条样本=一句/一对句"完全不同——所以叫 `GPTDataset` 而非通用 dataset。

```
逻辑 token 流(由 doc_idx 拼成):
  ...[docA eos][docB 很长很长....][docC]...
        │← 一条样本(seq_len+1)→│
        起点可落在 docB 中间,终点可跨到下一篇
```

---

## 7. BlendableDataset：按比例混采

当 `--data-path` 给了多个(权重, 路径)对,§2 里 `build_*` 会把 N 个子数据集塞进 `BlendableDataset`。它的核心也是**预先算一张索引表**:

```
weights = [0.3, 0.7]              # 归一化后
size    = sum(各子数据集要采的样本数)
# 用一个 C/cython 加速的 build_blending_indices,按权重把
# 全局样本号 i ──► (属于哪个子数据集 dataset_idx[i], 子数据集内下标 dataset_sample_idx[i])
```

`__getitem__(i)`:

```
i ──► dataset_idx[i]        = 选中第几个子数据集(按 weights 概率)
   ──► dataset_sample_idx[i]= 在那个子数据集里的样本号
   ──► 转交 self.datasets[dataset_idx[i]][dataset_sample_idx[i]]
```

```
 全局样本流(被预排好):
 [w][b][w][w][b][w][b][b]...   w=wiki(0.3)  b=books(0.7)
   │                     按长程比例 ≈ 30% / 70%
   ▼
 BlendableDataset[i] 直接路由到对应子 GPTDataset
```

**为什么不在 `__getitem__` 里现场随机抽?**
现场随机会让**不同进程/不同 epoch 看到不同序列**,破坏可复现性,也无法保证全局精确比例。**预先生成确定性索引表**(给定 seed)→ 所有 DP rank 看到一致的混采序列,比例也精确收敛到权重。

---

## 关键算法与数值示例

### 例1:训练总样本数怎么来的

假设 `train_iters=300000`,`global_batch_size=2048`:

$$\text{train\_num\_samples} = 300000 \times 2048 = 6.14\times10^8 \approx 6.14\text{ 亿条样本}$$

若语料共 $2\times10^9$ 条样本(文档切窗后),则训练**只走约 0.3 个 epoch**——大模型预训练数据"用不完"很常见,`doc_idx` 这时 epoch 数 = 1 即可。

### 例2:一条样本占多少 token / 显存

设 `seq_length=4096`,token 用 `uint16`(词表 < 65536)或 `uint32`:

| 项 | 计算 | 结果 |
| --- | --- | --- |
| 单样本 token 数(含+1) | 4097 | 4097 |
| 单样本字节(uint16) | $4097\times2$ | ≈ 8 KB |
| 一个 micro-batch=4 | $8\text{KB}\times4$ | ≈ 32 KB(仅 token,激活另算) |

DataLoader 侧只搬"几十 KB/样本"的 token,**数据本身从不是显存瓶颈**;瓶颈在前向激活(见 [[docs/transformer内存估算]])。

### 例3:多数据集权重 → 各采多少条

`--data-path 0.3 wiki 0.7 books`,总样本数 $N=6.14\times10^8$:

$$N_{wiki}=0.3N\approx1.84\times10^8,\quad N_{books}=0.7N\approx4.30\times10^8$$

`_build_*` 给 wiki / books 各传缩放后的 `num_samples`,所以**小权重数据集会被少采(可能 epoch<1),大权重会被多采(可能 epoch>1)**——这正是"按比例混合"的工程实现:不是真的复制数据,而是**调采样次数**。

---

## 对照、对比与局限

| 维度 | Megatron-DeepSpeed 数据栈 | HuggingFace `datasets` | 朴素 `torch Dataset` |
| --- | --- | --- | --- |
| 存储 | 自定义 `.bin/.idx` + mmap | arrow/parquet,memory-map | 任意(常 JSON/CSV) |
| 取数粒度 | "无穷 token 流"滑窗,跨文档 | 一条 record | 一条 record |
| 规模 | TB 级,O(1) 随机访问 | GB~TB | 受内存限 |
| 混采 | `BlendableDataset` 确定性按权重 | `interleave_datasets` | 自己写 |
| 可复现 | 索引表 + seed,全 rank 一致 | 取决于实现 | 取决于实现 |
| 多卡感知 | 与 DP/TP/PP group 解耦,各 rank 同序后再分片 | 需自己接 sampler | 需自己接 sampler |

**局限 / 注意:**
- 必须**先离线 `preprocess_data.py`** 生成 `.bin/.idx`,改一次数据要重跑,不如 HF 即开即用灵活。
- `doc_idx/sample_idx/shuffle_idx` 的 `.npy` 缓存**和 `num_samples`、`seq_length`、`seed` 绑定**:改任一参数会触发重算(冷启动慢),且旧缓存不删可能踩坑。
- 跨文档拼接对**长文档语料**友好,对"短句独立样本"任务(如分类)反而要专门 dataset,别硬套 GPTDataset。
- 极端 split 比例 + 小语料 → valid/test 文档数取整为 0,启动即报错(见 §4)。
- 不同 fork(微软 vs bigscience vs NVIDIA 原版)函数名/路径有差异,**以你 clone 的 commit 为准**;本文讲的是各家共有的机制骨架。

---

## 🔗 跳转链接

- 上层框架与 3D 并行总览:[[llm-train/megatron-deepspeed/README]]
- 入口脚本→训练循环全调用链:[[llm-train/megatron-deepspeed/microsoft/代码]]
- 分布式基础(DP/TP/PP、进程组):[[llm-train/pytorch/distribution/README]]
- 集合通信原语(all-reduce/all-gather):[[ai-infra/网络/集合通信原语]]
- 训练框架总览:[[llm-train/README]]
- 显存估算(激活才是大头):[[docs/transformer内存估算]]
- 知识地图总枢纽:[[00-知识地图]]
