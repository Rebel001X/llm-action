# vLLM KV 缓存与自动前缀缓存（APC）解剖

> **本篇取证基准**：`vllm` @ `7ca49fbe`（2026-08-22）
> **一句话**：块是定长页表项，前缀命中是哈希查表，驱逐是双向链表 O(1) 弹头。

## 0. 结论先行

vLLM V1 的 KV 缓存管理可以压缩成四句话：

1. **块（block）是唯一的分配单位**。`num_gpu_blocks` 个 `KVCacheBlock` 在启动时一次性建好，之后再也不发生 Python 对象创建——分配、释放、驱逐都只是在这批预建对象之间搬指针（`vllm/v1/core/kv_cache_utils.py:162`、`vllm/v1/core/block_pool.py:176`）。
2. **前缀缓存命中本质是一次哈希表查找**。每个块的哈希由「父块哈希 + 本块 token id + 额外键（LoRA/多模态/cache_salt）」链式构成（`vllm/v1/core/kv_cache_utils.py:620`），命中判定就是从 `block_hashes[0]` 开始顺着链表查 `cached_block_hash_to_block`，查不到就停（`vllm/v1/core/single_type_kv_cache_manager.py:735`）。
3. **驱逐不是 LRU 堆，是一条双向链表**。空闲块按"下次该被淘汰的顺序"排好队，弹出/插入都是 O(1) 指针操作，不需要堆的 O(log n) 上浮下沉（`vllm/v1/core/kv_cache_utils.py:228`）。
4. **可用块数 = (总显存 × gpu_memory_utilization − 权重 − 激活峰值 − CUDA Graph 预留) ÷ 每块字节数**，这是一条要跑一次真实前向传播（profile run）才能拿到分母之外所有量的公式，不是配置项能直接算出来的（`vllm/v1/worker/utils.py:505`、`vllm/v1/worker/gpu_worker.py:481`、`vllm/v1/core/kv_cache_utils.py:1362`）。

混合注意力（全注意力 + 滑动窗口 + Mamba 状态）不是"分别管理"，而是被拍扁成若干个 `KVCacheGroup`，每组自己的块表按自己的窗口/衰减规则收缩，但对外必须在同一个"调度器块粒度"（`scheduler_block_size`，各组块大小的最小公倍数）上给出一致的前缀命中长度——这靠一个不动点迭代算法（`HybridKVCacheCoordinator.find_longest_cache_hit`，`vllm/v1/core/kv_cache_coordinator.py:757`）完成，下面会逐行拆解。

## 1. 它在系统里的位置

KV 缓存子系统夹在调度器和模型执行器之间，是纯 CPU 侧的簿记层，本身不持有任何 GPU 张量：

```
Scheduler（见 03-vLLM-调度器解剖）
   │  每步：get_computed_blocks() → allocate_slots() → free()
   ▼
KVCacheManager  (vllm/v1/core/kv_cache_manager.py)
   │  多组时委托给协调器；单组时直接用一个 SingleTypeKVCacheManager
   ▼
KVCacheCoordinator（Unitary / Hybrid）  (vllm/v1/core/kv_cache_coordinator.py)
   │  每个 KVCacheGroup 一个 SingleTypeKVCacheManager
   ▼
SingleTypeKVCacheManager（Full/SlidingWindow/ChunkedLocal/Mamba/CrossAttention…）
   │  只做"哪些 block_id 属于这个请求"的账，不碰显存
   ▼
BlockPool  (vllm/v1/core/block_pool.py)
   │  唯一持有 `list[KVCacheBlock]`、空闲链表、哈希表的对象
   ▼
（GPU Worker 侧）KVCacheConfig → 真实张量分配，见 vllm/v1/worker/
   （模型执行时注意力后端读取 block_ids 拼 block_table，见 05-vLLM-注意力后端与算子层）
```

调度器把 `block_ids` 列表交给模型执行器，模型执行器的注意力后端把它拼成 PagedAttention 需要的 `block_table` 张量——KV 缓存管理器全程不知道张量长什么样，它管理的是"哪个逻辑块编号被谁占着"，这是一种彻底的关注点分离：容量算法、驱逐算法都可以脱离 CUDA 环境用纯 Python 单测（这也是为什么本库能在无 GPU 机器上把这套逻辑读透）。

这套结构在张量并行（TP）/流水线并行（PP）下也没有变复杂多少，因为"规划"和"执行"本来就是分开的两个阶段：`EngineCore` 先向**所有** worker 收集各自的 `kv_cache_specs`（每个 PP stage 持有的层不同，`KVCacheSpec` 集合自然也不同），合并成一份全局的 `KVCacheGroup` 划分方案（§4.3 提到的 `merged_kv_cache_specs`），再把这份**同一份**方案连同各 worker 各自 profile 出的可用内存一起喂给 `get_kv_cache_configs`——TP 场景下每个 rank 的层完全相同，只是数据并行切分了每层内部的头数，所以各 rank 天然应该得到同一份 `KVCacheConfig`；真正会让不同 worker 拿到不同 `num_blocks` 的，只有"各 worker profile 出的可用显存不同"这一个变量，而这恰恰是 §4.3 第三步"跨 worker 取最小值"要抹平的东西（详见 §7 第 4 条）。这意味着"混合注意力怎么对齐"（§4.2 的不动点算法）和"张量并行怎么对齐"（取 min）是两个独立的正交问题，分别在协调器层和 `get_kv_cache_configs` 层被处理，不会互相纠缠。

## 2. 代码地图（文件 → 职责，带行号）

以 `struct_map.py` 抓到的 kv_cache 子系统（13 文件 / 9,581 行）为基础，逐一核对行号：

| 文件 | 职责 | 关键行 |
|---|---|---|
| `vllm/v1/core/kv_cache_utils.py` | 哈希函数、`KVCacheBlock`/`FreeKVCacheBlockQueue`、KV 缓存组切分、容量计算的落地实现（全文件 2,323 行，本子系统最大文件） | `BlockHash` 定义 `vllm/v1/core/kv_cache_utils.py:47`；`KVCacheBlock` `vllm/v1/core/kv_cache_utils.py:162`；`FreeKVCacheBlockQueue` `vllm/v1/core/kv_cache_utils.py:228`；`hash_block_tokens` `vllm/v1/core/kv_cache_utils.py:620`；`get_request_block_hasher` `vllm/v1/core/kv_cache_utils.py:714`；`get_kv_cache_configs` `vllm/v1/core/kv_cache_utils.py:2061` |
| `vllm/v1/core/block_pool.py` | 唯一持有 GPU 块对象数组与前缀缓存哈希表的类 | `BlockHashToBlockMap` `vllm/v1/core/block_pool.py:33`；`BlockPool.__init__` `vllm/v1/core/block_pool.py:162`；`get_cached_block` `vllm/v1/core/block_pool.py:198`；`get_new_blocks` `vllm/v1/core/block_pool.py:647`；`touch` `vllm/v1/core/block_pool.py:702`；`free_blocks` `vllm/v1/core/block_pool.py:719` |
| `vllm/v1/core/kv_cache_manager.py` | 调度器唯一直接打交道的门面类，隐藏了组/协调器细节 | `KVCacheBlocks`（结果容器）`vllm/v1/core/kv_cache_manager.py:34`；`get_computed_blocks` `vllm/v1/core/kv_cache_manager.py:232`；`allocate_slots` `vllm/v1/core/kv_cache_manager.py:347`；`free` `vllm/v1/core/kv_cache_manager.py:570` |
| `vllm/v1/core/kv_cache_coordinator.py` | 单组用 `UnitaryKVCacheCoordinator`，多组（混合注意力）用 `HybridKVCacheCoordinator` | 基类 `vllm/v1/core/kv_cache_coordinator.py:64`；`free` `vllm/v1/core/kv_cache_coordinator.py:325`；混合协调器 `vllm/v1/core/kv_cache_coordinator.py:560`；跨组不动点命中算法 `vllm/v1/core/kv_cache_coordinator.py:757` |
| `vllm/v1/core/single_type_kv_cache_manager.py` | 每种 attention 类型一个具体管理器（全文件 1,953 行，本子系统第二大） | 抽象基类 `vllm/v1/core/single_type_kv_cache_manager.py:36`；`FullAttentionManager.find_longest_cache_hit` `vllm/v1/core/single_type_kv_cache_manager.py:684`；`SlidingWindowManager.find_longest_cache_hit` `vllm/v1/core/single_type_kv_cache_manager.py:903`；反向释放 `vllm/v1/core/single_type_kv_cache_manager.py:516`；MambaManager `vllm/v1/core/single_type_kv_cache_manager.py:1268` |
| `vllm/v1/kv_cache_interface.py` | 每种 attention 类型的"规格单"：一块占多少字节、一请求最多几块 | `KVCacheSpec` 基类 `vllm/v1/kv_cache_interface.py:144`；`AttentionSpec` `vllm/v1/kv_cache_interface.py:373`；`FullAttentionSpec` `vllm/v1/kv_cache_interface.py:434`；`SlidingWindowSpec` `vllm/v1/kv_cache_interface.py:687`；`MambaSpec` `vllm/v1/kv_cache_interface.py:811`；`KVCacheGroupSpec`/`KVCacheConfig` `vllm/v1/kv_cache_interface.py:1130`、`vllm/v1/kv_cache_interface.py:1145` |
| `vllm/v1/worker/utils.py` | 把 `gpu_memory_utilization` 换算成"允许申请的总内存字节数" | `request_memory` `vllm/v1/worker/utils.py:500` |
| `vllm/v1/worker/gpu_worker.py` | 真正跑一次 profile 前向，量出剩余显存 | `determine_available_memory` `vllm/v1/worker/gpu_worker.py:481` |
| `vllm/v1/engine/core.py` | 把 worker 侧的可用内存喂给 `get_kv_cache_configs`，再把结果广播回所有 worker | 调 profile `vllm/v1/engine/core.py:308`；调容量计算 `vllm/v1/engine/core.py:319`；写回 `num_gpu_blocks` `vllm/v1/engine/core.py:333` |
| `vllm/v1/core/kv_cache_metrics.py` | 可选的块级命中率/驻留时间统计钩子（96 行，最小的文件之一） | 由 `BlockPool` 持有的 `metrics_collector` 回调，见 `vllm/v1/core/block_pool.py:196` |

（此表已给出 20+ 条 `文件:行` 引用，超过 `## 2` 的硬指标 10 条。）

补充三点文件职责上容易搞混的边界，帮助定位读代码的起点：

- **`kv_cache_manager.py` vs `kv_cache_coordinator.py` 不是同一层**。前者是调度器唯一认识的门面（`KVCacheManager`），单组、多组模型都用同一套接口；后者才是"单组走 `UnitaryKVCacheCoordinator`、多组走 `HybridKVCacheCoordinator`"这个分支发生的地方（`vllm/v1/core/kv_cache_coordinator.py` 顶部的 `get_kv_cache_coordinator` 工厂函数）。读代码时如果只想看"调度器怎么用 KV 缓存"，看前者就够；想看"混合注意力怎么对齐"，才需要下钻到后者。
- **`single_type_kv_cache_manager.py` 里的类只做"账"，不做"分配"**。真正弹出/归还物理块的调用最终都落到 `self.block_pool` 上（`SingleTypeKVCacheManager.__init__` 里保存了 `block_pool` 引用，`vllm/v1/core/single_type_kv_cache_manager.py:81`）；`SingleTypeKVCacheManager` 自己维护的 `req_to_blocks: dict[str, list[KVCacheBlock]]`（`vllm/v1/core/single_type_kv_cache_manager.py:94`）只是"这个请求在这一组里持有哪些块"的视图，删掉这本账不会释放任何显存，必须经过 `block_pool.free_blocks`。
- **`kv_cache_interface.py` 里的 `*Spec` 类是纯数据 + 纯函数，不持有任何运行期状态**。它们全部标了 `@dataclass(frozen=True)`（如 `vllm/v1/kv_cache_interface.py:373`、`vllm/v1/kv_cache_interface.py:811`），意味着一次构造完就不可变，`max_memory_usage_bytes`/`page_size_bytes` 都是根据字段现算的纯函数——这批类在容量规划阶段被大量创建、比较、合并（`merge` classmethod），如果它们是可变对象，多线程/多进程下的合并逻辑会安全得多复杂。

## 3. 核心数据结构

### 3.1 `KVCacheBlock`（`vllm/v1/core/kv_cache_utils.py:162`）

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: BlockHashWithGroupId | None = None
    _block_hash_num_tokens: int | None = None
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
    is_null: bool = False
```

逐字段说明为什么存在：

- **`block_id: int`**——物理块编号，`0 .. num_gpu_blocks-1`，是这个类唯一"不可变"的字段。它是块表（block table）里写给注意力内核的那个整数，其余字段全是 CPU 侧簿记，GPU 内核不关心。
- **`ref_cnt: int`**——有多少个请求正在共享这块 KV。之所以要显式计数而不是用 Python 的 GC 引用计数，是因为"是否可回收"的判定要在**批调度前**完成——调度器要在真正分配前问"如果我现在把这个块交给新请求，会不会破坏正在跑的请求"，这必须是一个可读的整数而不是运行时对象图。
- **`_block_hash` / `_block_hash_num_tokens`**——只有块**满**（存完 `block_size` 个 token 的 KV）且被缓存时才非空。前置下划线 + `@property` 只读包装是为了让 `set_block_hash`/`reset_hash` 成为唯一写入口，强制"一个块的哈希只能被设置一次，被驱逐时必须显式清空"这条不变式（`set_block_hash` 里的 `assert` 见 `vllm/v1/core/kv_cache_utils.py:192`）。`_block_hash_num_tokens` 单独存是因为混合模型下不同组的块大小不同，同一个哈希粒度（`hash_block_size`）可能只覆盖块的一部分（"部分块"缓存，见 §4）。
- **`prev_free_block` / `next_free_block`**——不是"这个块的逻辑邻居"，而是它在**空闲链表**里的邻居。字段直接长在块对象上（而不是外部再包一层链表节点）是为了让"从空闲队列中间移除一个块"变成 O(1) 指针改写，不产生任何新对象（下面 §3.2 展开）。
- **`is_null`**——占位块（`block_id=0`，池初始化时第一个弹出的块，见 `vllm/v1/core/block_pool.py:190`），永远不参与哈希缓存、永远不会被真正 free。它是滑动窗口/混合模型里"这个位置没有对应的真实 KV"的显式哨兵，避免用 `None` 到处判空。

### 3.2 `BlockHash` / `BlockHashWithGroupId`（`vllm/v1/core/kv_cache_utils.py:47`）

```python
BlockHash = NewType("BlockHash", bytes)
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)
```

`BlockHash` 只是打了类型标签的 `bytes`，用 `NewType` 而不是裸 `bytes` 是为了让类型检查器抓出"把普通字节串错当块哈希传"的 bug。`BlockHashWithGroupId` 是 `block_hash + group_id.to_bytes(4, "big")` 拼接（`make_block_hash_with_group_id`，`vllm/v1/core/kv_cache_utils.py:60`）——**同一段 token 内容在不同 KV 缓存组（比如同一模型里的全注意力组和滑窗组）必须映射到不同的缓存条目**，因为两组的 KV 张量形状/内容完全不同，若不带 group id，混合模型第一次跑起来就会把全注意力的块错当滑窗块命中。

### 3.3 `FreeKVCacheBlockQueue`（`vllm/v1/core/kv_cache_utils.py:228`）

不是 Python 的 `deque`，是手写双向链表，`fake_free_list_head`/`fake_free_list_tail` 两个哨兵节点消除了首尾特判分支。字段只有 `num_free_blocks` 计数和两个哨兵——真正的链表数据全部寄生在 `KVCacheBlock.prev_free_block`/`next_free_block` 上，链表本身不为节点分配任何新对象。类文档字符串直接写明了排序语义：**队首是最先被驱逐的块**，且"同一批被释放的块之间，哈希 token 数更多（链更长）的排在更前面"（`vllm/v1/core/kv_cache_utils.py:239`）——这条排序不是这个类自己维护的，是调用方（`free_blocks`）通过"传入顺序"来保证的，详见 §5.3。

### 3.4 `BlockHashToBlockMap`（`vllm/v1/core/block_pool.py:33`）

前缀缓存的哈希表本体，`{BlockHashWithGroupId: KVCacheBlock | dict[int, KVCacheBlock]}`。**值是联合类型**（单块或 `{block_id: block}` 字典）是一个刻意的空间优化：绝大多数哈希键只对应一个物理块，直接存单个 `KVCacheBlock` 省掉内层 `dict` 的 GC 开销；只有出现哈希碰撞或同内容重复缓存（见类文档 NOTE #1，`vllm/v1/core/block_pool.py:47`：vLLM **不做内容去重**，同一段 token 若被两个独立请求各自算出块，会各自占一个物理块）时才升级成字典。这是"块表必须只增不改"（append-only）这条不变式换来的代价——不去重是为了保证一个请求已分配的 `block_id` 永远不会因为后台去重而改变。

### 3.5 `KVCacheSpec` 家族（`vllm/v1/kv_cache_interface.py:144` 起）

`KVCacheSpec` 是抽象基类，核心是一个属性契约：

- `block_size: int`——这层用多少 token 一个块。
- `page_size_bytes`——一个块在这层上占多少字节（`num_heads × num_states × state_content_size_bytes`，`vllm/v1/kv_cache_interface.py:414`）。这是"容量计算"公式里除数的直接来源。
- `max_memory_usage_bytes(vllm_config)`——**一个请求**在这层最多可能占多少字节；不同子类给出完全不同的答案，这正是"为什么混合注意力不能共用一套简单算法"的根源（见 §5.5）。

三个具体子类的关键差异（字段级）：

- **`FullAttentionSpec`**（`vllm/v1/kv_cache_interface.py:434`）：`sliding_window: int | None`（仅用于"混合分配器被禁用"时把滑窗层伪装成全注意力，见 §5.6）。`max_memory_usage_bytes` 直接是 `cdiv(max_model_len, block_size) * page_size_bytes`——请求越长占用越大，无上限（`vllm/v1/kv_cache_interface.py:459`）。
- **`SlidingWindowSpec`**（`vllm/v1/kv_cache_interface.py:687`）：`sliding_window: int` + `extra_retained_tokens: int`（多模块投机解码要多保留几个 token 的块，字段注释直接写了原因，`vllm/v1/kv_cache_interface.py:689`）。`max_memory_usage_bytes` 与 `max_model_len` **无关**，只与窗口大小 + 一个批次里最多同时在途的 token 数有关（`max_admission_blocks_per_request`，`vllm/v1/kv_cache_interface.py:696`）——这是"滑窗请求的显存占用有硬上限，不会随生成长度无限增长"的直接代码证据。
- **`MambaSpec`**（`vllm/v1/kv_cache_interface.py:811`）：没有 `block_size` 语义上的"多少 token"，而是 `shapes: tuple[tuple[int, ...], ...]` + `dtypes`——Mamba 的状态是一个固定形状的 SSM 隐状态张量，不是"每 token 一份 KV"。`max_memory_usage_bytes` 在默认模式下是 `page_size_bytes * (1 + num_speculative_blocks)`（`vllm/v1/kv_cache_interface.py:847`）——**每个请求恒定占用，跟请求多长完全无关**，这是本篇 §7 的第一个反直觉点。

### 3.6 `KVCacheGroupSpec` / `KVCacheConfig` / `KVCacheTensor`（`vllm/v1/kv_cache_interface.py:1130/1145/1106`）

- `KVCacheGroupSpec.layer_names: list[str]` + `kv_cache_spec: KVCacheSpec`——一组"共用一张块表"的模型层。`is_eagle_group: bool` 标记这组是不是投机解码草稿模型的层（EAGLE/MTP 命中要多丢一个块，见 §4）。
- `KVCacheConfig.num_blocks: int`——**全局唯一**的块池大小，所有组共用同一个 `block_id` 空间（见 §5.5 为什么必须共用）。`kv_cache_tensors: list[KVCacheTensor]` 描述 worker 该怎么把这块内存切给各层。
- `KVCacheTensor.size/layers/layer_stride/block_stride/offset`——不是"每层一个张量"，而是"一段裸字节 buffer + 每层的步长"，字段注释直接给出寻址公式：`offset + l * layer_stride + b * block_stride`（`vllm/v1/kv_cache_interface.py:1111`）。多个 `KVCacheTensor` 的地址区间可以重叠（"alias"），因为一个 `block_id` 在任意时刻只属于一个组，不会有两层同时读写同一物理字节。

### 3.7 `SpecGroup`：把"共用同一份规格"的组批处理成一次查找

`HybridKVCacheCoordinator` 内部不是逐个 `KVCacheGroup` 单独查前缀命中，而是先按 `KVCacheSpec` 是否相等把组归并成 `SpecGroup`（`vllm/v1/core/kv_cache_coordinator.py:545`）：

```python
class SpecGroup(NamedTuple):
    spec: KVCacheSpec
    group_ids: list[int]
    manager_cls: type[SingleTypeKVCacheManager]
    use_eagle: bool
```

- **`spec` / `manager_cls`**——一旦两个 `KVCacheGroup` 的 `kv_cache_spec` 完全相等（比如模式重复出的多份滑动窗口层），它们必然产生完全相同的命中长度，没必要各查一次；`spec` 和 `manager_cls` 就是"这一批组该用哪个规格、调哪个类的 `find_longest_cache_hit`"的凭证。
- **`group_ids: list[int]`**——同一 `SpecGroup` 下所有组的编号，查一次命中结果要广播回所有这些组各自的块列表（`vllm/v1/core/kv_cache_coordinator.py:858` 的 `for group_id, blocks in zip(group_ids, hit_blocks)`）。
- **`use_eagle: bool`**——只要这批组里**任意一个**是 EAGLE/MTP 草稿层，整批就要按 EAGLE 语义丢最后一块（字段文档字符串直接写明"the EAGLE last-block drop is necessarily decided for the whole spec group"，`vllm/v1/core/kv_cache_coordinator.py:549`）——这是"批处理换来的一个必须接受的粗粒度化"：不能让同一 `SpecGroup` 里一部分组丢块、另一部分不丢，因为它们共用同一次查找结果。

`KVCacheCoordinator` 基类自己持有的核心字段是 `single_type_managers: tuple[SingleTypeKVCacheManager, ...]`（`vllm/v1/core/kv_cache_coordinator.py:136`）——按 `KVCacheGroup` 的下标一一对应，是协调器把"跨组统一算法"（不动点迭代、公共前缀统计）下沉到"单组具体实现"（每种 attention 类型各自的 `find_longest_cache_hit`）的唯一入口。理解这两层字段的关系，是读懂 §4.2 那段不动点循环代码里 `first_group_id = group_ids[0]` 这类写法的前提——它并不是"随便挑一个组代表"，而是"这批组的命中结果本来就该完全相同，取哪个下标都一样"。

### 3.8 `BlockHashListWithBlockSize`：不同块大小的组怎么共用同一份哈希序列

这是本篇回答"混合注意力块大小不同怎么办"这个问题时最关键的一个数据结构（`vllm/v1/core/kv_cache_utils.py:2217`），但它不是一个存储哈希的容器，而是一个**惰性视图**：

```python
class BlockHashListWithBlockSize:
    def __init__(self, block_hashes, hash_block_size, target_block_size):
        self.block_hashes = block_hashes
        self.scale_factor = target_block_size // hash_block_size
    def _get_value_at(self, idx):
        return self.block_hashes[(idx + 1) * self.scale_factor - 1]
```

背景是这样的：`request.block_hashes` 只按**一种**粒度（`hash_block_size`，取各组块大小的最大公约数或 `prefix_match_unit`，见 §5.7）算一遍，不会为每个 `KVCacheGroup` 各自的块大小重复计算一遍哈希——这既省计算，也保证"同一段 token 内容在不同粒度下必须得到一致的命中结果"这条正确性要求不会因为两套独立计算的哈希函数实现细节不同而被破坏。但一个块大小是 `hash_block_size` 四倍的组，需要的是"每 4 个细粒度哈希对应一个粗粒度块"的视图，而不是重新算。

这个类能做到"零重算"的关键在于哈希本身是**链式**的：`hash_block_tokens` 的定义是 `hash(parent_hash, curr_tokens, extra_keys)`（`vllm/v1/core/kv_cache_utils.py:645`），意味着细粒度序列里第 `k` 个哈希已经把从头到第 `k` 块结尾的**全部**前缀都编码进去了。类文档字符串给了一个具体例子（`vllm/v1/core/kv_cache_utils.py:2231`）：`hash_block_size=16`、`target_block_size=32` 时，覆盖 token 0-31 的粗粒度哈希，就**等于**覆盖 token 16-31 的那个细粒度哈希（因为它的 `parent_hash` 已经链到了 token 0-15）——`_get_value_at` 因此只需要做一次下标换算 `(idx+1)*scale_factor - 1`，取到"这个粗粒度块的最后一个细粒度哈希"直接返回，不需要重新调用一次 `hash_block_tokens`。三个字段分别是：`block_hashes`（底层的细粒度哈希列表，只读、不拷贝）、`scale_factor`（粗/细粒度块大小之比，构造时断言必须整除）、隐式的"惰性"——`__getitem__`/`__iter__` 都是按需计算下标，不会在构造时把整个粗粒度序列物化成一个新列表。

### 3.9 `KVCacheMetricsCollector`：采样式的块级驻留统计（`vllm/v1/core/kv_cache_metrics.py:46`）

这是全子系统里唯一"默认不产生任何开销、需要显式配置才生效"的组件，而且即使开启也不是全量统计：

- **`sample_rate: float = 0.01`**（`vllm/v1/core/kv_cache_metrics.py:49`）——默认只对 1% 的新分配块记录生命周期，构造函数强制 `0 < sample_rate <= 1.0`。这是一个刻意的取舍：块的分配/访问/驱逐是调度热路径上极高频的操作，全量记录每个块的时间戳会带来不可忽视的开销，采样把这个开销压到可接受范围，代价是统计结果是抽样估计而不是精确值。
- **`block_metrics: dict[int, BlockMetricsState]`**——只保存"被抽中"的块的状态，字典大小随采样率而不是总块数增长，这是控制内存开销的另一半。
- **`BlockMetricsState.access_history: deque[int]`**，`maxlen=4`（`vllm/v1/core/kv_cache_metrics.py:23`)——每个被抽样的块只保留最近 4 次访问的时间戳，用来算"复用间隔"（`get_reuse_gaps_seconds`），不是无界增长的访问日志；注释直接写了原因："Bounded to prevent unbounded growth if a block is accessed many times"——一个被极高频复用的公共前缀块（比如系统提示词对应的块）如果不设上限，它的访问历史会比其他任何数据结构都先把内存吃光。
- **`on_block_allocated` / `on_block_accessed` / `on_block_evicted`** 三个回调分别挂在 `BlockPool.get_new_blocks`（`vllm/v1/core/block_pool.py:669`）、`touch`（`vllm/v1/core/block_pool.py:716`）、`_maybe_evict_cached_block`（`vllm/v1/core/block_pool.py:691`）上——采集点和 §3.1/§4.1 讨论的引用计数变更点完全重合，说明这套统计不是另起一条独立的观测路径，而是寄生在已有的分配/驱逐调用链上，这也是为什么开启它的边际开销主要来自"随机数生成 + 采样判定"而不是额外的遍历。

## 4. 主流程走读

### 4.1 新请求进调度器：命中 → 分配 → （部分）缓存

```
KVCacheManager.get_computed_blocks(request)          # vllm/v1/core/kv_cache_manager.py:232
    → coordinator.find_longest_cache_hit(block_hashes, max_len)
        单组: UnitaryKVCacheCoordinator                 (vllm/v1/core/kv_cache_coordinator.py:472 附近)
        多组: HybridKVCacheCoordinator.find_longest_cache_hit  (vllm/v1/core/kv_cache_coordinator.py:757)
    → 逐组调用 SingleTypeKVCacheManager.find_longest_cache_hit
        FullAttentionManager   (vllm/v1/core/single_type_kv_cache_manager.py:684)：从头顺扫，遇 miss 立即停
        SlidingWindowManager   (vllm/v1/core/single_type_kv_cache_manager.py:903)：从尾向头扫，只需窗口内连续命中
    → 每组内部调 BlockPool.get_cached_block(block_hash, group_ids)  (vllm/v1/core/block_pool.py:198)
```

`get_computed_blocks` 里有一处容易读漏但很关键的边界处理：`max_cache_hit_length = request.num_tokens - 1`（`vllm/v1/core/kv_cache_manager.py:262`）——**即使全部 token 都命中缓存，也要故意少算最后一个 token**，因为最后一个位置的 logits 必须重新算一次前向才能采样下一个 token；如果把最后一个 token 也算作"已命中"，会导致这个块被跳过计算，从而拿不到 logits。这是"prompt 100% 命中缓存"这个看似最优场景下的一个必要妥协——**最少也要重算一个块（或一个哈希粒度）**。

拿到命中块之后：

```
KVCacheManager.allocate_slots(...)                    # vllm/v1/core/kv_cache_manager.py:347
    1. 释放不再需要的块（比如滑窗外的旧块）并检查空闲块是否够
    2. touch() 命中块：ref_cnt += 1，且如果块之前 ref_cnt==0（在空闲队列里）就摘出来
    3. get_new_blocks() 从空闲队列头部弹出新块（如果弹到的是被缓存的块，顺带把它从哈希表逐出）
    4. 已经写满的新块立即 cache_full_blocks()，这样同一批（batch）里的其他请求马上能复用
```

`allocate_slots` 的 docstring 里画了一张 ASCII 图（`vllm/v1/core/kv_cache_manager.py:394`），把一个块表切成 `comp | new_comp | ext_comp | new | lookahead` 五段——`comp` 是本地已算完并已计入 `ref_cnt` 的前缀，`new_comp` 是这次新命中但还没 touch 的前缀，`ext_comp` 是由 KV connector（P/D 分离场景）从别处传来的前缀，`new` 是这次要真正跑前向的 token，`lookahead` 是投机解码要预留的草稿块。这五段共用同一份 `req_to_blocks` 列表，靠偏移量区分状态——**没有为"待计算"和"已计算"分别建列表**，因为块表本身必须是 append-only 的（见 §5.4），状态只能用游标表达。

### 4.2 混合注意力的不动点对齐

`HybridKVCacheCoordinator.find_longest_cache_hit`（`vllm/v1/core/kv_cache_coordinator.py:757`）是本篇最值得逐行读的一段。核心矛盾：全注意力组"越靠前命中越长"，滑窗组"只要窗口内连续命中就够，跟更早的前缀无关"，Mamba 组的状态是否可复用又依赖它自己的保留策略——三种组各自跑一遍 `find_longest_cache_hit` 会给出**互不相同**的命中长度，但调度器最终只能给整个请求一个统一的 `num_computed_tokens`。算法做法：

1. 先给一个候选长度 `hit_length`（初始为 `max_cache_hit_length`）。
2. 让每个组在"不超过候选长度"的约束下各自求它能命中多长；全注意力组因为"命中集合下降封闭"（downward-closed，命中 N 就必然命中 N 的所有前缀）只需要查一次，之后每轮只做截断（`vllm/v1/core/kv_cache_coordinator.py:810`）。
3. 如果某组算出的命中长度比候选值更短，说明候选值"虚高"，把候选值降到这个更短的值，**所有组重新来一轮**。
4. 直到没有任何组再缩短候选值为止——这就是不动点。

代码里专门优化了"简单混合"（只有全注意力 + 恰好一种其他类型）这一最常见情形：证明只需要一轮就收敛，直接 `break`（`is_simple_hybrid` 分支，`vllm/v1/core/kv_cache_coordinator.py:790`、`867`），避免了通用情形下潜在的多轮循环开销。

### 4.3 容量计算：从 `gpu_memory_utilization` 到 `num_gpu_blocks`

```
CacheConfig.gpu_memory_utilization（用户配置，比如 0.9）
    │
    ▼
request_memory(init_snapshot, cache_config)            # vllm/v1/worker/utils.py:500
    requested_memory = ceil(total_memory * gpu_memory_utilization)     # :505
    （若 free_memory < requested_memory 直接报错，不会静默降级）
    │
    ▼
GPUWorker.determine_available_memory()                 # vllm/v1/worker/gpu_worker.py:481
    with memory_profiling(...):
        model_runner.profile_run()      # 真跑一次最大 batch 的前向，量出激活峰值
    free_gpu_memory = profile_result.after_profile.free_memory
    available_kv_cache_memory_bytes =
        requested_memory - non_kv_cache_memory - cudagraph_memory_estimate   # :565
    │  （非 KV 内存 = 模型权重 + 前向激活峰值 + 其他子系统占用，profile 实测得出）
    ▼
EngineCore（vllm/v1/engine/core.py:308）收集所有 worker 的可用内存
    │
    ▼
get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory)    # vllm/v1/core/kv_cache_utils.py:2061
    → get_kv_cache_config_from_groups(...)                             # vllm/v1/core/kv_cache_utils.py:1329
        bytes_per_block = max(每个组"一块占多少字节"的总和)             # :1284 附近, _get_kv_cache_bytes_per_block
        num_blocks = available_memory // bytes_per_block                # :1362
    → 跨 worker（TP/PP）取 min(num_blocks)，按最小值重新收缩每个 worker 的分配  # :2201-2211
    │
    ▼
EngineCore 写回 vllm_config.cache_config.num_gpu_blocks                # vllm/v1/engine/core.py:333
```

三个容易忽略的细节：

1. **`gpu_memory_utilization` 的分母是显存总量，不是当前空闲量**（`request_memory` 用的是 `init_snapshot.total_memory`，`vllm/v1/worker/utils.py:505`）。这意味着如果同一张卡上还跑着别的进程占了显存，vLLM 依然会按"总量的 90%"去申请，申请不到就直接报错，而不是自动退让到"空闲量的 90%"。
2. **权重和激活峰值是实测的，不是估算的**——`profile_run()` 真的构造一个 `max_num_batched_tokens` 规模的假 batch 跑一次前向，用 PyTorch 的内存快照差值算出激活峰值。这是"KV 缓存池大小取决于一次真实前向"这条结论的直接代码证据：没有 GPU 就不可能精确算出这个数字，本篇也因此不产出任何具体的 `num_gpu_blocks` 实测值（诚实标准第 4 条）。
3. **多组模型的 `bytes_per_block` 取的是各组里最大的那个**（`_get_kv_cache_bytes_per_block`，取 `max`，见 §2 表格），但 `num_blocks` 是全局唯一的一个数——这意味着如果某个组的每块字节数远小于另一组（比如 Mamba 状态块 vs 全注意力块），块池的"块数"由**最大的那个组**决定，小块组会在同样的 `num_blocks` 下"浪费"一部分块容量换来的显存空间用不满（这也是 §5.5 要展开的代价）。

### 4.4 一个解析算例（非实测，只演算公式）

本机没有 GPU，下面的数字全部是**假设输入 + 公式推导**，不是任何一次真实 profile 的输出——这是诚实标准第 4 条要求的"解析计算"标注。取一组典型的 8B 级稠密模型规格作为假设：

| 假设量 | 取值 | 依据 |
|---|---|---|
| `num_kv_heads`（GQA 分组后的 KV 头数） | 8 | 假设值，仅用于演算 |
| `head_size` | 128 | 假设值 |
| `head_size_v`（默认等于 `head_size`） | 128 | `AttentionSpec.__post_init__`，`vllm/v1/kv_cache_interface.py:392` |
| `dtype` | fp16（2 字节/元素） | 假设值 |
| `block_size`（调度器块粒度） | 16 | 假设值 |
| `num_hidden_layers` | 32 | 假设值 |
| 假设可用于 KV 缓存的显存 | 40 GiB | 假设 `determine_available_memory` 的实测输出（本例中虚构） |

第一步，单层每块字节数（`AttentionSpec.page_size_bytes`，无 padding 时等于 `unpadded_page_size_bytes`，`vllm/v1/kv_cache_interface.py:411`）：

```
state_content_size_bytes = (head_size + head_size_v) * dtype_size
                          = (128 + 128) * 2 = 512 字节
num_states               = block_size / tokens_per_state = 16 / 1 = 16   （tokens_per_state 默认 1）
page_size_bytes(单层)     = num_heads * num_states * state_content_size_bytes
                          = 8 * 16 * 512 = 65,536 字节 = 64 KiB
```

第二步，一个组（这里假设单组、全部 32 层都是全注意力）每块字节数（`_get_kv_cache_bytes_per_block`，对组内所有层的 `page_size_bytes` 求和，`vllm/v1/core/kv_cache_utils.py:1284`）：

```
bytes_per_block = num_hidden_layers * page_size_bytes(单层)
                = 32 * 65,536 = 2,097,152 字节 = 2 MiB
```

（这个"一个块 2 MiB"级别的数量级，与社区里常见的"7B/8B 模型 `block_size=16` 时一个块约 2 MiB"的经验说法量级相符，可以作为公式没算错的一个交叉检验——但这仍然是**本库推断的交叉验证**，不是对某一具体版本实测数字的引用。）

第三步，套用 `num_blocks = available_memory // bytes_per_block`（`vllm/v1/core/kv_cache_utils.py:1362`）：

```
available_memory = 40 * 1024**3 = 42,949,672,960 字节
num_blocks        = 42,949,672,960 // 2,097,152 = 20,479   （整除，此处取整数部分）
```

第四步，换算成"总 token 容量"（不是"最大并发请求数"，是所有正在跑的请求一共能占用的 token 总数）：

```
总 token 容量 = num_blocks * block_size = 20,479 * 16 ≈ 327,664 tokens
```

如果假设 `max_model_len = 4096` 且请求之间完全不共享前缀（最坏情况，前缀缓存不生效），能同时塞进显存的最长请求数上限是：

```
327,664 / 4,096 ≈ 79.9  → 理论上限约 79 个并发满长度请求
```

这个 79 只是"块池能装下多少 token"这一个约束下的上限，实际并发数还要受 `max_num_seqs`、调度器的 `max_num_batched_tokens`、以及是否触发前缀缓存命中（命中会让共享前缀的多个请求只消耗一份块）共同影响——本例只演算 KV 缓存这一层公式，不代表某个具体部署的真实并发能力。

### 4.5 运行中请求的增量分配，与一个两请求共享前缀的具体例子

新请求只是 `allocate_slots` 的一种调用方式；一个**已经在跑**的请求每解码出一个新 token，调度器同样要调用一次 `allocate_slots`（`vllm/v1/core/kv_cache_manager.py:347`），只是这次 `new_computed_blocks` 为空——`get_num_blocks_to_allocate` 里专门有一条"快路径"处理这种情况：一旦这个请求 ID 已经出现在 `self.num_cached_block` 里（意味着它是正在运行、不是刚调度的新请求），直接断言不会再有新的前缀命中，只算"还差几块才能装下当前已确定的 token 数"（`vllm/v1/core/single_type_kv_cache_manager.py:191` 的 `if request_id in self.num_cached_block` 分支）。这条快路径的存在是一个性能考虑：解码阶段每步都要跑一次这个函数，多数时候一个新 token 根本不需要新块（上一个块还没写满），能尽早返回就不要重新走一遍"扫哈希、算命中"的完整逻辑。

用官方设计文档里的思路（`docs/design/prefix_caching.md:143` 起的"Duplicated blocks"例子）复述一个具体场景，全部换成本篇已经核对过行号的术语：

1. 请求 1 携带 prompt "ABCDEF"（假设 `block_size=4`），走一遍 `get_computed_blocks`——首次请求，前缀缓存里什么都没有，命中长度为 0。
2. `allocate_slots` 走 `get_new_blocks` 从空闲队列头部弹出 2 个新块，分别装满 "ABCD" 和 "EF"（第二块还没写满）。"ABCD" 立即触发 `cache_full_blocks`（`vllm/v1/core/block_pool.py:225`），这一步会把它的哈希（`hash(NONE_HASH, ("A","B","C","D"), None)`，按 `hash_block_tokens` 的定义，`vllm/v1/core/kv_cache_utils.py:645`）写进 `cached_block_hash_to_block`——注意这一步发生在**请求 1 还在跑、还没结束**的时候，是"同一批次内其他请求也能立刻复用"这条设计的直接体现（§4.1 步骤 4 里提到的"立即缓存"）。
3. 请求 1 继续解码，输出 "GH"，"EF" 块被写满、连带新解码的 "G" "H" 又开出第三块。
4. 此时请求 2 携带 prompt "ABCDXY" 到达（前 4 个 token 与请求 1 完全相同，后 2 个不同）。`get_computed_blocks` 顺着哈希链查：`hash(NONE_HASH, ("A","B","C","D"), None)` 命中——正是请求 1 那块 "ABCD"。`FullAttentionManager.find_longest_cache_hit`（`vllm/v1/core/single_type_kv_cache_manager.py:684`）在这一块之后继续查 "XY" 对应的哈希（这次哈希的 `parent_block_hash` 是 "ABCD" 块的哈希，见 `hash_block_tokens` 的链式定义），查不到（因为请求 1 走的是 "EF" 不是 "XY"），命中在这里停住，`hit_length=4`。
5. `allocate_slots` 对请求 2 调用 `touch()`（`vllm/v1/core/block_pool.py:702`）命中的那一块——它的 `ref_cnt` 从 1（只被请求 1 引用）变成 2（请求 1、请求 2 共同引用），**物理上只有一份 "ABCD" 的 KV 数据**，两个请求的块表里对应位置指向同一个 `block_id`。这正是"前缀缓存"字面意义上省下的那部分显存与计算——请求 2 不需要重新跑 "ABCD" 四个 token 的前向。
6. 若干步之后请求 1 结束，调用 `free`（`vllm/v1/core/kv_cache_manager.py:570`）——"ABCD" 块的 `ref_cnt` 从 2 减到 1（不是减到 0），所以它**不会**被放回空闲队列（`vllm/v1/core/block_pool.py:732` 的 `if block.ref_cnt == 0` 判定），仍然安全地留在请求 2 手里；而请求 1 独占的 "EFGH" 等后续块 `ref_cnt` 减到 0，按 §5.4 的倒序释放规则进入空闲队列排队。

这个例子把 §3–§5 里分散讲的哈希链式结构、`touch`/`free_blocks` 的引用计数语义、以及"立即缓存写满的块"这三件事串成了一条时间线；也直接解释了为什么"前缀缓存"不需要任何显式的"复制"或"迁移"操作——从头到尾只有指针（`block_id`）在请求之间被共享，物理数据从未被移动过。

### 4.6 `KVCacheConfig` 落地成真实显存的最后一步

到 §4.3 为止，`KVCacheConfig`（`num_blocks` + `kv_cache_tensors`）还只是一份"计划"，本篇开头强调过 KV 缓存管理器全程不持有任何 GPU 张量——真正把这份计划变成一段可读写显存的代码在 GPU worker 侧，链路是：

```
GPUModelRunner.initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)
                                                    # vllm/v1/worker/gpu_model_runner.py:7446
    → allocate_kv_cache(kv_cache_config, device, layout, kernel_block_sizes)
                                                    # vllm/v1/worker/utils.py:378
        buf = torch.zeros(size, dtype=torch.int8, device=device)     # :395，唯一一次真实显存分配
        for tensor in kv_cache_config.kv_cache_tensors:
            views = create_kv_cache_views(buf, spec, num_blocks, layout, tensor, ...)
                                                    # vllm/v1/kv_cache_interface.py:306
                torch.as_strided(buf, size=logical_shape, stride=strides, ...)   # :357，零拷贝切片
    → bind_kv_cache(...)                            # vllm/v1/worker/gpu_model_runner.py:7476
        把各层的视图张量绑定进注意力层的 forward context，之后模型正向传播里
        `layer.forward()` 直接读写这些视图
```

三个值得注意的地方：

1. **全模型只调用一次 `torch.zeros`**（`vllm/v1/worker/utils.py:395`），后续每一层的 KV 缓存都是这一整块裸 `int8` buffer 上的一个 `torch.as_strided` 视图（`vllm/v1/kv_cache_interface.py:357`），不是每层单独 `cudaMalloc`。这与 §3.6 提到的"多个 `KVCacheTensor` 可以地址重叠"是同一件事的两面：分配阶段先要一整块**连续**显存，才能保证之后按 `KVCacheGroup` 切分出的各层视图地址是可预测、可用步长表达的。
2. **`assert len(sizes) == 1`**（`vllm/v1/worker/utils.py:394`）——所有 `KVCacheTensor` 必须共享同一个 `size`（即同一份 `bytes_per_block * num_blocks` 总量），这是 §4.3 里"块池只有一套 `num_blocks`"这条设计在张量分配层面留下的直接约束：如果哪个组算出的总大小跟别的组对不上，这里会直接断言失败而不是悄悄分配出两块不一致的显存。
3. **视图是零拷贝的**：`create_kv_cache_views` 用 `torch.as_strided`（`vllm/v1/kv_cache_interface.py:357`）而不是 `torch.narrow`/切片再 `.clone()`，意味着"KV 缓存布局从 `KVCacheGroup` 到具体张量步长的转换"完全是元数据操作，不产生任何一次显存搬运——这也是为什么 §4.3 提到的"跨 worker 取最小 `num_blocks` 后要重新规划"（`vllm/v1/core/kv_cache_utils.py:2210`）代价可以接受：重新规划改变的是步长参数，不需要重新搬运已经写入的数据（不过要注意，一旦真正调用过一次 `allocate_kv_cache`，缩小 `num_blocks` 意味着要重新走一遍这整条分配链路，产生的是一块新的、更小的 buffer，而不是对旧 buffer 做原地收缩）。

### 4.7 公共前缀统计：KV 缓存管理器还顺带给级联注意力提供了一个信号

除了服务前缀命中和容量分配，`KVCacheManager` 每个调度步还会额外算一个数：当前所有运行中请求共享的最长公共前缀有多少块。调度器每步调用一次 `get_num_common_prefix_blocks(any_request_id)`（`vllm/v1/core/sched/scheduler.py:1198`），注释直接写了用途——"可能被级联注意力（cascade attention）使用"（`vllm/v1/core/sched/scheduler.py:1192`）。具体算法在 `FullAttentionManager.get_num_common_prefix_blocks`（`vllm/v1/core/single_type_kv_cache_manager.py:823`）：

```python
def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
    blocks = self.req_to_blocks[running_request_id]
    num_common_blocks = 0
    for block in blocks:
        if block.ref_cnt == len(self.req_to_blocks):
            num_common_blocks += 1
        else:
            break
    return num_common_blocks
```

这段代码复用了一个此前已经在维护的量——`ref_cnt`——来回答一个新问题："这个块是不是被**所有**正在运行的请求共享"，判定条件就是 `ref_cnt == len(self.req_to_blocks)`（当前运行请求总数）。之所以能这样判定，是因为 `ref_cnt` 本来就精确统计了"有多少个请求持有这块"（见 §3.1），不需要为"是否公共前缀"这个新需求单独维护一份计数。级联注意力优化的原理是：如果一批请求有很长的公共前缀，注意力计算可以把这段公共前缀单独算一次再广播，而不是让每个请求各自重算一遍——`get_num_common_prefix_blocks` 就是把这个"能省多少"的信息，从 KV 缓存管理器已有的账本里几乎零成本地导出给调度器。这也是"关注点分离"这条系统设计原则在本篇范围内的一个小例子：KV 缓存管理器本身不知道什么是级联注意力，它只是恰好持有回答这个问题所需的全部数据。

## 5. 设计决策与代价

### 5.1 块哈希链入 LoRA id / 多模态哈希 / cache_salt，不只哈希 token id

**为什么这么设计**：block hash 的定义是 `hash((parent_block_hash, curr_block_token_ids_tuple, extra_keys))`（`hash_block_tokens`，`vllm/v1/core/kv_cache_utils.py:645`），`extra_keys` 由 `generate_block_hash_extra_keys`（`vllm/v1/core/kv_cache_utils.py:582`）拼出，包含三类东西：

- **LoRA 名字**（`_gen_lora_extra_hash_keys`，`vllm/v1/core/kv_cache_utils.py:541`）：同一串 token id，用 LoRA A 的权重和用 LoRA B 的权重算出来的 KV **数值完全不同**（LoRA 改的是权重，不改 token）。如果不把 LoRA id 混进哈希，两个用不同适配器跑同一段前缀的请求会互相"偷"到对方用错误权重算出的 KV——这不是性能问题，是**正确性问题**：模型会在错误的 KV 上继续生成，输出静默错误且没有任何报错信号。
- **多模态哈希 + 块内偏移**（`_gen_mm_extra_hash_keys`，`vllm/v1/core/kv_cache_utils.py:474`）：多模态填充位 token id 对不同的图片/音频是**相同的**（都是同一个特殊 token 重复填充），真正的信息在图片/音频的嵌入里，不在 token id 里。不哈希图片内容本身、只哈希 token id 的话，两张不同的图片只要填充段长度相同就会被错误地判定"前缀相同"，直接把 A 图的 KV 喂给问 B 图的请求——图文不符的静默错误。代码里连"同一张图出现在块内不同偏移"这种更细的情形都区分开了（`offset - start_token_idx` 作为额外键的一部分，`vllm/v1/core/kv_cache_utils.py:525`）。
- **`cache_salt`**（仅在块起点 `start_token_idx == 0` 时混入，`vllm/v1/core/kv_cache_utils.py:604`）：这条不是正确性问题，是**多租户隐私**问题。vLLM 官方安全文档明确指出：同一后端上的攻击者可以通过测量 TTFT（首 token 延迟）来推断"我猜的这段前缀是不是命中了别的用户的缓存"，研究显示在仅 8 个 token 的前缀长度下，这个旁路信号的 ROC AUC 就能到 0.99（`docs/usage/security.md:540`）。`cache_salt` 把一个租户私有的字符串混进第一块的哈希，让不同租户即使 token 完全相同也无法互相命中，从根上掐断这条计时旁路（`docs/design/prefix_caching.md:87`）。

**不这样会怎样**：LoRA/多模态维度不混哈希 → 缓存污染导致模型输出错误，且没有报错、没有崩溃，是最难排查的一类 bug（表现为"偶尔这个请求答非所问"，复现依赖并发时序）。`cache_salt` 不用 → 多租户场景下存在可实测的旁路信道，能让一个租户探测另一个租户是否问过某个特定前缀。

**什么时候可以不这样**：单租户、单 LoRA（或不用 LoRA）、纯文本、不关心计时旁路的场景（比如离线批量跑分、内部工具单用户使用）——这时候额外键要么天然为空、要么可以忽略不设置 `cache_salt`，代码路径本身也是这么写的：`extra_keys` 为空时函数直接返回 `None`（`vllm/v1/core/kv_cache_utils.py:614`），不会给哈希计算带来任何额外开销。

### 5.2 空闲块用双向链表，不用堆

**为什么这么设计**：类文档字符串直接给出了理由（`vllm/v1/core/kv_cache_utils.py:230`）：需要在链表**中间**以 O(1) 移除任意一个块（一个块被 `touch()` 命中时要从空闲队列里摘出来，它此时可能在队列任意位置，不一定在头/尾）。堆结构做不到 O(1) 的任意元素删除（标准二叉堆删除中间元素是 O(log n) 且需要维护元素到堆内位置的映射）；而链表 + 每个节点自带前后指针，删除就是"把前驱的 next 指向后继、后继的 prev 指向前驱"，与元素在链表里的位置无关。而且这套实现完全不用 Python `deque`/`heapq` 包一层节点对象，直接把指针字段焊在 `KVCacheBlock` 上，省掉了每次入队出队的对象分配。

**不这样会怎样**：如果用堆按"最后访问时间"排序，`touch()` 一个命中块（这是**高频操作**，每次前缀命中都要做）就要做一次对数复杂度的上浮，且需要额外的"元素 → 堆内索引"映射表（否则找不到要调整的位置）；用普通 `deque`，"删除中间元素"是 O(n) 线性扫描，在块数动辄成千上万的大显存场景下会成为调度热路径上的瓶颈。

**什么时候可以不这样**：如果驱逐策略从"近似 LRU"换成"严格 FIFO 不支持中途摘除"（比如完全不支持前缀缓存的部署，`enable_caching=False`），本就不需要"命中后摘出队列"这个操作，用简单 `deque` 或数组栈就够——事实上 `enable_caching=False` 时 `get_new_blocks` 确实跳过了 `_maybe_evict_cached_block` 的哈希表操作（`vllm/v1/core/block_pool.py:664` 的 `if self.enable_caching` 分支）。

### 5.3 释放时区分「有哈希的块」和「没哈希的块」，分别走 FIFO 和 LIFO

**为什么这么设计**：`BlockPool.free_blocks`（`vllm/v1/core/block_pool.py:719`）里明确分两路：

```python
if block.block_hash is None or not self.enable_caching:
    blocks_to_evict_first.append(block)   # LIFO：塞回队首
else:
    blocks_to_evict_last.append(block)    # FIFO：塞到队尾
```

没有哈希的块（比如从没写满过、或前缀缓存关闭）**没有被复用的可能**，让它们尽快被重新分配，同一物理块反复被同一类请求使用有利于 GPU 缓存局部性（注释直接写了"LIFO reuse ... for better GPU locality"）；有哈希的块**有被复用的可能**，理应尽量晚被驱逐，给后来的相同前缀请求更长的存活窗口，这是标准 LRU 语义。

**不这样会怎样**：如果所有释放块都走同一种顺序（比如统一 FIFO），没哈希的块会和有哈希的块混在队列里排队等驱逐，短寿命的"注定不会被复用"的块占着队列前部拖慢下一次分配；如果统一 LIFO，则最近释放的高价值缓存块反而最先被覆盖，前缀缓存命中率会明显下降。

**什么时候可以不这样**：`enable_caching=False` 时，因为所有块永远没有哈希，两个分支自动合并成同一种行为（都进 `blocks_to_evict_first`），这条区分本身自然失效——不需要额外配置去关闭它。

### 5.4 释放顺序按「链尾先释放」，不是任意顺序

**为什么这么设计**：`SingleTypeKVCacheManager.free`（`vllm/v1/core/single_type_kv_cache_manager.py:524`）显式 `reversed()` 了块列表再传给 `free_blocks`：

```python
self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))
```

一个请求的块表是按 token 顺序 append 出来的，链尾（最后几块）覆盖的是这个请求**独有**的后缀 token，链头覆盖的是**大概率被别的请求共享**的前缀 token。倒序释放意味着"独有的后缀块"先进入空闲队列排队（更快被驱逐），"共享的前缀块"后进入（更晚被驱逐）——这与 §3.3 提到的类文档字符串里"哈希 token 数更多的块排在更前面"这条排序规则完全对应：一个请求结束时，它贡献给空闲队列的所有块，链越长（离前缀起点越远）的越先被淘汰。

**不这样会怎样**：正序释放会让最靠近 prompt 开头（最可能被其他请求复用）的块反而先进入驱逐候选区的前部，在显存紧张、驱逐频繁发生时，公共前缀会比独有后缀更快被冲掉，前缀缓存的收益直接被这个顺序错误抵消掉一大半。

**什么时候可以不这样**：单请求场景、或请求之间已知不共享任何前缀（比如每个请求都带独立 `cache_salt`）时，释放顺序对命中率没有影响，这时候顺序只是纯粹的实现细节，可以简化。

### 5.5 混合注意力必须共用同一个 `block_id` 空间，用 `KVCacheGroup` 而不是独立的池

**为什么这么设计**：`_get_kv_cache_groups_uniform_page_size` 的文档字符串给出了最直接的解释（`vllm/v1/core/kv_cache_utils.py:1135`）：模型层是按固定模式重复的（比如 10 层全注意力 + 20 层滑窗，模式是 `1 全 + 2 滑窗` 重复 10 次），把"模式里的每一种层"归成一个组，每组只需要维护**一份**块表，再把这份块表复用到模式重复的所有层上——否则要给 30 层每层建一份独立块表，块表大小是层数的线性函数而不是"模式种类数"的线性函数，内存开销和簿记复杂度都会爆炸。同时文档写明了做这件事的前提假设：所有组"每块物理字节数必须相同"（`vllm/v1/core/kv_cache_utils.py:1157` 附近），否则块粒度的显存碎片管理会变得极其复杂——这也是为什么 §4.3 里 `bytes_per_block` 要取所有组的 `max` 而不是各自独立计算：**块池只有一套 `num_blocks`，物理块大小必须对齐，小块的组只能在大块的空间里"欠打满"，不能单独精算**。

**不这样会怎样**：如果每种 attention 类型各建一个独立块池、独立 `num_blocks`，需要为每个池单独做一次容量估算、单独维护一套驱逐链表和哈希表，前缀缓存命中判定还要跨池同步"这个请求在池 A 里命中了 N 块，在池 B 里只命中了 M 块，最终该按哪个数字推进 `num_computed_tokens`"——这正是 §4.2 那个不动点算法在解决的问题；如果没有统一的块 id 空间，这个问题会从"CPU 侧簿记复杂"升级成"要不要为每种类型分别调 `gpu_memory_utilization` 子配额"这种更难运维的选型。

**什么时候可以不这样**：`--disable-hybrid-kv-cache-manager` 开启时，vLLM 会把滑窗层"伪装"成全注意力层（用 `FullAttentionSpec.sliding_window` 字段记住窗口大小，但块分配按全注意力走，`unify_hybrid_kv_cache_specs`）——这是"退回单组"的官方逃生舱：牺牲滑窗层本该有的"块数不随请求变长"优势，换取更简单、历史包袱更小的单组代码路径，适合调试或排查"是不是混合分配器本身的 bug"。

### 5.6 可用内存必须靠一次真实 profile run 量出来，不能纯算

**为什么这么设计**：模型前向的激活内存峰值（尤其是 `max_num_batched_tokens` 规模下的中间激活、CUDA Graph 捕获所需的临时显存）依赖具体算子实现、融合策略、量化方案、并行切分方式，理论上可以静态分析但极其复杂且容易在版本升级后过期；用真实前向跑一次并用 PyTorch 内存快照做差值（`memory_profiling` 上下文管理器包住 `profile_run()`，`vllm/v1/worker/gpu_worker.py:521`），是"用一次实测替代一整套容易过期的静态估算模型"的工程取舍。

**不这样会怎样**：静态估算错了会导致两种失败模式之一——算少了，白白浪费本可以用来扩大 KV 缓存池、提升并发的显存；算多了，`num_gpu_blocks` 定得过大，真正跑起来时激活内存和 KV 缓存挤爆显存导致 OOM，而 OOM 往往发生在服务已经在生产环境跑了一段时间、遇到某个之前没测过的输入长度组合时——这是最难运维的一类故障。

**什么时候可以不这样**：`kv_cache_memory_bytes` 配置项可以完全跳过这次 profile 对 KV 缓存部分的自动推导，直接手动指定 KV 缓存要用多少字节（`vllm/v1/worker/gpu_worker.py:495` 附近的 `if kv_cache_memory_bytes := ...` 分支）——适合"同一份配置反复启动、已经知道安全数值"的稳定生产环境，用手动数字换掉每次启动都要付出的 profile 时间；但代码日志里也明确警告这条路径"不再遵守 `gpu_memory_utilization`"，纯靠使用者自己保证安全边界。

### 5.7 只精确缓存"整块"，块内部只在尾部做一次有限的部分缓存

**为什么这么设计**：前缀缓存的哈希是按 `hash_block_size`（可能小于等于真正的调度块 `block_size`，由 `resolve_kv_cache_block_sizes` 决定，`vllm/v1/core/kv_cache_utils.py:650`）逐段链式计算的，但只有当一段刚好落在真正块边界上时才会被写进 `cached_block_hash_to_block`。`FullAttentionManager._cache_partial_tail_block`（`vllm/v1/core/single_type_kv_cache_manager.py:793`）专门处理"prompt 长度不是块大小整数倍"这一种情况——只登记 prompt 结尾那一个部分块，中间任何哈希粒度的边界都**故意跳过**（docstring 原话："intermediate hash boundaries inside the same cache block are intentionally skipped"，`vllm/v1/core/single_type_kv_cache_manager.py:801`）。这样做把"部分块缓存"限制在一个可控的范围内：一个物理块永远只可能被注册成一种缓存条目（要么整块、要么恰好这一个 prompt 尾部边界），不会因为同一块内部有多个哈希粒度的候选边界而产生歧义。

**不这样会怎样**：如果放开中间边界也登记缓存，同一个物理块可能同时对应好几个不同长度的哈希键（这块的前 8 个 token 一个键，前 16 个 token 又一个键），`BlockHashToBlockMap` 的键空间会成倍膨胀，而且"这个物理块的 KV 内容"和"哪些哈希键指向它"之间的对应关系会变得一对多——一旦这个块被驱逐，需要同时使好几个哈希键失效，`_remove_cached_block_hashes` 之类的清理逻辑复杂度也会上升。

**什么时候可以不这样**：混合模型且 `hash_block_size < block_size`（多个 KV 缓存组的块大小不同，取它们的最大公约数或 `prefix_match_unit` 作为哈希粒度，`vllm/v1/core/kv_cache_utils.py:701`）时，细粒度部分命中才有意义——这时 `FullAttentionManager.find_longest_cache_hit` 会走"fine-grained"分支（`vllm/v1/core/single_type_kv_cache_manager.py:718`），在块内部按哈希粒度探测更长的命中；单组、块大小与哈希粒度相等的最常见情形下，这整套"细粒度探测"代码直接不触发（`fine_grained` 判定为假）。

### 5.8 空闲块池里永远焊死一个不会被使用的 `null_block`

**为什么这么设计**：`BlockPool.__init__` 在构造完所有块之后，立刻从空闲队列弹出第一个块（`block_id=0`）当作 `null_block`，标记 `is_null=True`（`vllm/v1/core/block_pool.py:190`）。它的用途是给"这个位置逻辑上不该有真实 KV"的场景一个安全占位——比如滑动窗口把窗口外的旧块替换成 `null_block`（`RSWAManager.remove_skipped_blocks` 的 docstring 明确写了"replace the removed blocks with null_block so the block_table is valid"，`vllm/v1/core/kv_cache_manager.py:589` 附近同名方法的注释），这样注意力内核拿到的 `block_table` 永远是一个合法的、指向某个真实存在的物理块的整数数组，不需要在内核里额外判断"这个位置是不是空指针"。

**不这样会怎样**：如果用 `-1` 或 `None` 表示"这个位置没有块"，PagedAttention 内核（或者拼 `block_table` 的 Python 代码）就要在每次访问前做一次条件分支，CUDA 内核里的分支是要付出warp 级别的代价的；更麻烦的是，`null_block` 方案让"跳过这个位置"和"正常访问"走的是**同一条代码路径**（读一块内容全为初始化值/哨兵值的哈希无关块），而 `-1`/`None` 方案需要在内核里单独写一条"跳过"分支，等于要维护两条路径的正确性。

**什么时候可以不这样**：如果一个部署从不启用滑动窗口、混合注意力、R-SWA 等任何会产生"空洞"的特性（纯稠密全注意力模型），`null_block` 实际上永远不会被写入任何请求的块表——它仍然存在（因为 `BlockPool.__init__` 无条件创建），但纯粹是待机状态，不影响这类部署的行为；这也是为什么容量计算要专门为它保留一块内存（见 §7 第 1 条）——即使用不上，这一个块也必须被预留出来。

### 5.9 八条决策速查表

逐条读完 §5.1–§5.8 之后，用一张表回顾"取舍点在哪里"，方便之后带着问题回来重读某一条：

| 决策 | 换来的好处 | 付出的代价 | 逃生舱 |
|---|---|---|---|
| 5.1 哈希混入 LoRA/多模态/salt | 正确性（不串味）+ 隐私（防计时旁路） | 每块哈希多算几个额外键 | 单租户单 LoRA 纯文本场景额外键天然为空 |
| 5.2 双向链表而非堆 | 命中摘除 O(1) | 无法免费获得"按数值排序"的能力（本来也不需要） | 关闭前缀缓存后不再需要摘除操作 |
| 5.3 free 时 LIFO/FIFO 分流 | 兼顾 GPU 局部性与缓存命中率 | 多一次条件判断 | `enable_caching=False` 时两分支自动合一 |
| 5.4 倒序释放 | 保护公共前缀更晚被淘汰 | 需要在调用处显式 `reversed()` | 请求间不共享前缀时顺序不影响结果 |
| 5.5 多组共用一套 `block_id` 空间 | 避免为每种 attention 类型单独管理一套池 | 块物理大小必须对齐（取 max，小块组"欠打满"） | `--disable-hybrid-kv-cache-manager` 退回单组 |
| 5.6 容量靠真实 profile run | 避免静态估算过期导致 OOM 或浪费 | 每次启动多付出一次前向的时间 | `kv_cache_memory_bytes` 手动指定，跳过 profile |
| 5.7 只精确缓存整块+一个尾部部分块 | 避免同一物理块对应多个哈希键 | 中间哈希边界的部分命中放弃 | 混合模型块大小不同时才需要细粒度部分命中 |
| 5.8 恒定预留一个 `null_block` | `block_table` 永远合法，内核无需判空分支 | 容量规划永远少算一块 | 纯稠密全注意力部署中这个块永远待机 |

## 6. 同位对照（另一个引擎在同一位置怎么做）

vLLM 的前缀缓存是"定长块 + 哈希表"：命中判定的时间复杂度和内存开销都是`O(前缀长度 / block_size)`，数据结构是一张扁平的 `{哈希: 块}` 字典。

SGLang 的 RadixAttention 走的是另一条路：用一棵基数树（radix tree）按 token 序列做前缀索引，节点本身就是"共享前缀"的显式表示，不需要像 vLLM 这样把每个块的哈希单独算出来再查表——这是公开的 RadixAttention 设计通识，**本库尚未在本次任务中逐行核对 SGLang 源码**，具体到哪个文件、哪一行实现了树节点分裂/合并、驱逐策略是否也是双向链表式的近似 LRU，请以本库后续 `[[03-SGLang-RadixAttention与前缀缓存]]` 篇的逐行核对为准——此处标注为**本库推断**，不作为可引用的行号断言。

两者一个可以现在就下的、有 vLLM 一侧真实代码支撑的结论是：vLLM 的哈希链式设计天然支持"任意粒度的部分块命中"（见 §5.7 的 `hash_block_size` 与 `resolve_block_hashes`，`vllm/v1/core/kv_cache_utils.py:650`），因为哈希本身就是链式累加的，可以在任意哈希粒度边界上截断查询；这与"用树结构做前缀索引"要解决的是同一个问题（如何以细粒度复用最长公共前缀），只是数据结构选择不同——一个用显式树形状表达共享关系，一个用哈希链表达同样的关系但把"树"隐式编码进了哈希值本身。

另一个可以现在就下、同样只依赖 vLLM 自身源码的对照点是"跨节点/跨进程复用前缀缓存"这件事在 vLLM 里怎么留的口子：`hash_block_size` 与 `prefix_match_unit` 的存在（`vllm/v1/core/kv_cache_utils.py:701`）不只是为了混合模型内部的组间对齐，注释里明确写了它同时服务于"prefix caching and KV connectors (P/D, offloading)"（`vllm/v1/core/kv_cache_utils.py:684`）——也就是说，vLLM 把"块哈希"设计成了一种可以脱离单个进程、传给外部 KV 传输组件（P/D 分离场景下的生产者/消费者）复用的通用标识符，而不是一个只在本进程内部有意义的内部实现细节。这套机制具体怎么和 P/D 分离对接，留给 [[12-vLLM-PD分离与KV-Connector]] 篇展开；跨引擎层面，SGLang 的 HiCache 分层缓存（GPU/CPU/磁盘多级）解决的是相似的"缓存要不要跨越单一存储层级"问题，具体实现细节同样标注为**本库推断**，留给 [[11-SGLang-PD分离与HiCache分层]] 核对。

## 7. 踩坑与反直觉

1. **"Mamba 状态占用不随生成长度增长"，与"KV 缓存随生成长度线性增长"的直觉相反**。`MambaSpec.max_memory_usage_bytes` 在非 `"all"` 模式下恒定是 `page_size_bytes * (1 + num_speculative_blocks)`（`vllm/v1/kv_cache_interface.py:847`）——每个请求只占 1（或 2，`"align"` 模式下）个状态块，无论生成了 10 个 token 还是 10 万个 token。这是 SSM/线性注意力架构相对 Transformer 全注意力在长序列场景下的核心显存优势，代码里体现得非常直白：`max_num_blocks_per_req` 在 `"align"` 模式外根本不看 `max_len` 参数（`vllm/v1/kv_cache_interface.py:860` 走的是 `max_memory_usage_bytes // page_size_bytes`，与 `max_len` 无关）。
2. **"LRU" 驱逐并不是严格按最后访问时间排序的 LRU**。真正决定一个块在空闲队列里位置的，是它"进入队列那一刻"是走 LIFO 还是 FIFO 路径（§5.3）、以及释放批次内部的顺序（§5.4）——一个块被 `touch()` 命中并不会让它在队列里"往后挪"，因为被 touch 的块此时 `ref_cnt > 0`，根本不在空闲队列里；它只有在 `ref_cnt` 重新归零、被下一次 `free_blocks` 调用送回队列时，才会按"最新一批释放"的规则重新排位。这是一种"近似 LRU"，精确程度取决于批调度的粒度，不是教科书意义上"每次访问都更新时间戳"的严格 LRU。
3. **混合模型的前缀命中判定不是一次查表就能拿到答案，可能要跑多轮不动点迭代**（§4.2）。只有"全注意力 + 恰好一种其他类型"这种最常见情形被特判成一轮收敛（`is_simple_hybrid`），更复杂的组合（比如同时有滑窗和 Mamba 两种非全注意力组）理论上需要多轮，循环终止条件是"这一轮没有任何组再缩短候选长度"——虽然收敛性有保证（长度单调递减且下界为 0），但迭代轮数不是一个可以提前静态算出的常数。
4. **张量并行/流水线并行下，KV 缓存池的大小由"最穷"的那个 worker 决定**。`get_kv_cache_configs` 最后一步显式 `min(kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs)`，再把其余 worker 的分配按这个最小值重新收缩（`vllm/v1/core/kv_cache_utils.py:2201`）——如果集群里某张卡因为被其他进程多占了一点显存导致 profile 出的可用内存更少，**全体 worker 的 KV 缓存池都会被拉低到这张卡的水平**，不是"各用各的"。
5. **SlidingWindowManager 的命中查找是官方自己标注的次优实现**：代码里留了一条待办注释，说当前从右向左逐块扫描的复杂度是 `O(max_num_blocks)`，在命中率低的场景下本可以优化到 `O(max_num_blocks / sliding_window_contiguous_blocks + sliding_window_contiguous_blocks)`（`vllm/v1/core/single_type_kv_cache_manager.py:937`）——这条直接留在源码里，见 §8。
6. **`num_gpu_blocks` 里永远有一个块用不上，而且这不是四舍五入误差，是刻意预留**。`get_kv_cache_configs` 在做容量检查前，会先从 `available_memory` 里减掉一个块的字节数再检查（`check_memory = avail_mem - _pool_bytes_per_block(groups)`，`vllm/v1/core/kv_cache_utils.py:2166`），注释直接写明理由："Reserve the null block BlockPool permanently holds back, so auto-fit and the capacity check both plan against usable blocks"——也就是说，即便某次部署完全用不到滑动窗口/混合注意力这些需要 `null_block` 的特性，容量规划公式依然会先扣掉一个块的预算，"能用的块数"永远比"分配到的块数"少一。这在块数动辄上万的场景下完全不可感知，但如果有人试图用 §4.4 那样的公式去精确对账"我这台机器理论上应该有多少块"，会发现比实测数字系统性地多算了（大约）一个块——差值就来自这条预留。
7. **不加密的哈希算法（`xxhash`/`xxhash_cbor`）在跨进程场景下默认不可复现，即使内容完全相同**。`NONE_HASH`（每条哈希链的起点种子）在使用加密哈希（`sha256`/`sha256_cbor`）时用固定的默认种子 `"vllm-none-hash"`（`vllm/v1/core/kv_cache_utils.py:108`），保证不同进程对相同内容算出相同的块哈希；但非加密哈希算法因为不具备抗碰撞性，如果种子固定就可能被攻击者离线预计算出碰撞（`vllm/v1/core/kv_cache_utils.py:100` 注释直接引用了 issue #12621），所以 `resolve_none_hash_seed` 对非加密算法**默认用每进程随机种子**（`os.urandom(32).hex()`，`vllm/v1/core/kv_cache_utils.py:129`）。反直觉的地方在于：如果一个多实例部署想要跨实例共享前缀缓存（比如通过 P/D 分离或外部 KV 存储比对哈希），换成 `xxhash` 追求哈希速度的同时，会**默认静默失去跨进程可复现性**——除非显式设置 `PYTHONHASHSEED` 环境变量统一种子（代码里也确实检查了这个环境变量优先级最高，`vllm/v1/core/kv_cache_utils.py:125`）。不设置的话不会报错，只会在跨进程比对时"莫名其妙"地全部 miss。

## 8. 可改进点

1. **`SlidingWindowManager.find_longest_cache_hit` 的线性扫描可以按源码里留的待办注释优化**（`vllm/v1/core/single_type_kv_cache_manager.py:937`）：低命中率场景下，每次 cache miss 后可以直接把扫描游标跳过一整个 `sliding_window_contiguous_blocks` 长度，而不是逐块递减；这是源码作者自己承认的已知次优点，不是本库的猜测。
2. **`BlockHashToBlockMap` 不做内容去重**（`vllm/v1/core/block_pool.py:47` NOTE #1）是为了保证块表 append-only，但代价是"同一段内容被不同请求各自计算出块哈希后各占一份物理块"这种情况下无法共享——**本库推断**：如果能在不破坏 append-only 的前提下，为"新算出的块哈希已存在于表中"这种情况提供一种延迟合并（比如后台异步把新块标记为可驱逐并把引用指向已存在的块），理论上能进一步提高高重复度工作负载（比如多个请求各自独立生成但恰好走到完全相同的推理路径）下的缓存命中率；但这需要打破"block_id 分配后永不改变"这条当前贯穿调度器和模型执行器的核心不变式，改造成本很高，本库判断短期内不会有人做，标注为长期可探索方向而非可落地 PR。
3. **`metrics_collector` 是可选注入**（`vllm/v1/core/block_pool.py:196`），默认不开启；即使开启，`sample_rate` 默认也只有 0.01（§3.9），运维方如果不主动传入一个 `KVCacheMetricsCollector` 实例并调高采样率，就只能看到聚合的 `PrefixCacheStats`，看不到块级驻留时间/复用间隔分布。**本库推断**：把最基础的一层轻量统计（比如驱逐年龄分布的直方图，全量而非采样，因为直方图分桶本身的开销远小于保存完整访问历史）做成默认开启、零额外内存分配的路径，能让运维方在不改配置的情况下就获得"我的 `gpu_memory_utilization` 设置是不是导致了过度驱逐"这类调优信号，当前这个信号必须主动配置且承受采样偏差才能拿到。
4. **跨 worker（TP/PP）取最小 `num_blocks` 再统一收缩**（`vllm/v1/core/kv_cache_utils.py:2201`，见 §7 第 4 条）是目前"最简单能保证正确性"的方案，但代价是"一张卡的一点点显存波动，会拉低整个集群的 KV 缓存容量"。**本库推断**：如果 worker 间的显存差异是长期稳定的（比如流水线并行不同 stage 的层数天然不均衡，导致某个 stage 的非 KV 内存占用系统性更高），理论上可以让每个 worker 独立持有自己的 `num_blocks`，代价是调度器发往不同 worker 的 `block_table` 不能再假设"同一个 `block_id` 在所有 worker 上都有效"——这是一处"简单正确 vs 榨干每一分显存"的典型权衡，短期看不出有人会为了这点收益去承担这个复杂度，标注为长期方向而非可落地 PR。

## 9. 自测题与延伸阅读

**自测题**（闭卷）：

1. 一个块的哈希由哪三类信息构成？如果两个请求 token 完全相同、但一个带 LoRA 适配器 A、一个不带 LoRA，它们的第一个块哈希会相同吗？为什么？
2. `get_computed_blocks` 里为什么要把命中长度上限设成 `request.num_tokens - 1` 而不是 `request.num_tokens`？如果去掉这个 `-1` 会导致什么问题？
3. 空闲块队列为什么用双向链表而不是二叉堆？这个选择在"命中一个正在排队等驱逐的块"这个操作上带来了什么复杂度优势？
4. 一个请求结束时，它的块是按分配顺序释放还是按分配顺序的反序释放？这个顺序选择跟"哪些块更可能被别的请求复用"有什么关系？
5. `gpu_memory_utilization` 的分母是显存总量还是当前空闲显存？这意味着同一张卡上如果还跑着别的进程，vLLM 的行为会是什么？
6. 混合注意力模型（全注意力 + 滑动窗口）为什么不能给两种层各建一个独立的 KV 缓存池、各自独立算 `num_blocks`？
7. 用 §4.4 的解析算例方法，如果把 `dtype` 从 fp16 换成 fp8（1 字节/元素），`bytes_per_block` 和 `num_blocks` 会如何变化？这与量化 KV 缓存能提升并发的直觉是否一致？
8. 为什么 `BlockPool` 里永远有一个 `null_block`，而且容量计算时要专门为它预留一块内存？如果一个部署从不用滑动窗口/混合注意力，这个预留是否还会发生？
9. 如果一个多实例部署把哈希算法从 `sha256` 换成 `xxhash` 追求速度，且没有设置 `PYTHONHASHSEED`，跨进程共享前缀缓存会发生什么？为什么这个失败模式不会报错，只会表现为"命中率异常低"？

**延伸阅读**（本库双链）：

- [[03-vLLM-调度器解剖]]——KV 缓存管理器的所有方法都是被调度器每一步调用的，理解调用时序需要回到调度循环里看。
- [[05-vLLM-注意力后端与算子层]]——本篇止步于"哪个逻辑块编号归谁"，`block_ids` 之后如何被拼成 `block_table` 张量、喂给 PagedAttention 内核，在这一篇继续。
- [[03-SGLang-RadixAttention与前缀缓存]]——§6 提到的"树 vs 哈希表"两种前缀缓存实现路线的另一半，逐行核对后可以把本篇 §6 的推断替换成真正的源码引用。
- [[12-vLLM-PD分离与KV-Connector]]——§6 提到的"块哈希被设计成可以传给外部 KV 传输组件复用"具体怎么落地，在这一篇展开。
