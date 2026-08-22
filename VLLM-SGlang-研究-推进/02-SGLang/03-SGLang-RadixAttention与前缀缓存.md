# SGLang RadixAttention 与前缀缓存解剖

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：树替代块表，命中任意长度前缀，代价是深度遍历与分裂开销。

## 0. 结论先行

1. **RadixAttention 的本质不是"用了一棵树"，是把 KV 复用的最小单元从"定长块"换成"变长树节点"。** 一个节点的 `key` 可以是 1 个 token，也可以是几千个 token——只要没有别的请求在中间分叉过（`python/sglang/srt/mem_cache/radix_cache.py:238`）。
2. **分裂（split）是这套设计的心脏。** 当新请求在某个节点内部分叉时，`_split_node` 用一次指针重接把该节点切成"共享前缀父节点 + 两个各自的子节点"，只拷贝分裂点之前那一段 KV 索引，不触碰分裂点之后任何数据（`python/sglang/srt/mem_cache/radix_cache.py:705`）。这是 radix tree 相对"定长块 + 哈希表"方案的核心差异，第 3、6 节会逐行拆开讲。
3. **引用计数 `lock_ref` 沿"叶子→根"整条路径传播**，把"这个前缀正被谁占用"编码进树本身，而不是额外维护一张"正在使用的块"集合（`python/sglang/srt/mem_cache/radix_cache.py:623`）。
4. **驱逐只从叶子开始，用一个按 `last_access_time` 排序的堆**：弹出一个叶子后，检查它的父节点是否因此变成了新叶子，是就推进堆继续弹——LRU 在这里是"沿树坍缩"而不是"全局排序"（`python/sglang/srt/mem_cache/radix_cache.py:593`）。
5. **一个必须先说清楚的反直觉事实**：本篇精读的 863 行 `RadixCache`（`radix_cache.py`）是当前代码库里仍然存在、可独立读懂的经典实现，但 SGLang 默认的生产路径走的其实是另一个文件里 2,894 行的 `UnifiedRadixCache`（`python/sglang/srt/mem_cache/unified_radix_cache.py:148`）。两者共享同一套"匹配→分裂→驱逐"算法骨架，`UnifiedRadixCache` 把它重新组织成组件化架构以支持混合注意力（SWA/Mamba）与分层缓存。第 7 节会给出证据链，说明为什么本篇仍然选 `RadixCache` 作为精读对象。

## 1. 它在系统里的位置

前缀缓存夹在调度器和 KV 物理存储之间，只做"哪段 token 序列已经算过、算过的 KV 存在哪些槽位"的簿记，本身不持有 GPU 张量：

```
Scheduler（见 [[02-SGLang-Scheduler事件循环]]）
   │ 请求入队/续跑前: match_prefix() 找可复用前缀
   │ prefill/decode 写完新 token 后: cache_unfinished_req() / cache_finished_req()
   ▼
BasePrefixCache 抽象接口 (python/sglang/srt/mem_cache/base_prefix_cache.py:235)
   │ 由 registry.py::create_tree_cache 按 ServerArgs 选一个具体实现
   ├── ChunkCache        (python/sglang/srt/mem_cache/chunk_cache.py:35)      —— --disable-radix-cache 时用它，不做前缀复用
   ├── RadixCache        (python/sglang/srt/mem_cache/radix_cache.py:303)    —— 本篇精读对象；单树，HiRadixCache/PureSWARadixCache 的父类
   ├── UnifiedRadixCache (python/sglang/srt/mem_cache/unified_radix_cache.py:148) —— 默认工厂链兜底选中的实现，组件化
   └── RadixCacheCpp     (python/sglang/srt/mem_cache/radix_cache_cpp.py)    —— 需要 SGLANG_EXPERIMENTAL_CPP_RADIX_TREE，默认关闭
   ▼
TreeNode.value: torch.Tensor[int64]  —— 存的是 KV cache 里的"槽位下标"，不是 KV 本身
   ▼
ReqToTokenPool (python/sglang/srt/mem_cache/memory_pool.py:256) + TokenToKVPoolAllocator —— 真正持有 GPU KV 张量的地方
```

关键的关注点分离：`RadixCache` 全程不知道 KV 张量长什么样，它管理的是"哪个逻辑 token 位置对应哪个槽位编号"（`TreeNode.value`），真实的 K/V 数据在 `MHATokenToKVPool`/`MLATokenToKVPool` 这类 `KVCache` 子类里（`python/sglang/srt/mem_cache/memory_pool.py:1628`、`python/sglang/srt/mem_cache/memory_pool.py:1759`）。这个分离和 vLLM 的 `KVCacheManager` vs `BlockPool` 是同一个思路（见 [[04-vLLM-KV缓存与前缀缓存]] §1），差别只在"逻辑索引"的组织方式——一个用树，一个用哈希表，这是第 6 节的主题。

调度器如何、在哪一步调用 `match_prefix`/`cache_unfinished_req`，属于调度循环的时序问题，留给 [[02-SGLang-Scheduler事件循环]] 展开；本篇只关心这棵树内部怎么运作。

## 2. 代码地图（文件 → 职责，带行号）

`kv_cache` 子系统按 `struct_map.py` 统计有 47 文件 / 32,786 行，与前缀缓存直接相关的核心文件如下：

| 文件 | 职责 | 关键行 |
|---|---|---|
| `radix_cache.py` | `RadixKey`/`TreeNode`/`RadixCache` 的完整教科书级实现，863 行，本篇主体 | `RadixKey` 类 `python/sglang/srt/mem_cache/radix_cache.py:59`；`RadixKey.match`（最长公共前缀）`python/sglang/srt/mem_cache/radix_cache.py:181`；`RadixKey.child_key` `python/sglang/srt/mem_cache/radix_cache.py:217`；`RadixKey.page_aligned` `python/sglang/srt/mem_cache/radix_cache.py:150`；`TreeNode` 类 `python/sglang/srt/mem_cache/radix_cache.py:238`；`RadixCache` 类 `python/sglang/srt/mem_cache/radix_cache.py:303`；`match_prefix` `python/sglang/srt/mem_cache/radix_cache.py:377`；`insert` `python/sglang/srt/mem_cache/radix_cache.py:437`；`evict` `python/sglang/srt/mem_cache/radix_cache.py:593`；`inc_lock_ref` `python/sglang/srt/mem_cache/radix_cache.py:623`；`dec_lock_ref` `python/sglang/srt/mem_cache/radix_cache.py:638`；`_match_prefix_helper` `python/sglang/srt/mem_cache/radix_cache.py:679`；`_split_node` `python/sglang/srt/mem_cache/radix_cache.py:705`；`_insert_helper` `python/sglang/srt/mem_cache/radix_cache.py:738`；`_delete_leaf` `python/sglang/srt/mem_cache/radix_cache.py:811`；`_update_leaf_status` `python/sglang/srt/mem_cache/radix_cache.py:821`；文件自带可运行 demo `python/sglang/srt/mem_cache/radix_cache.py:849` |
| `base_prefix_cache.py` | 所有前缀缓存实现共享的抽象接口与参数/结果数据契约 | `BasePrefixCache` `python/sglang/srt/mem_cache/base_prefix_cache.py:235`；`MatchPrefixParams` `python/sglang/srt/mem_cache/base_prefix_cache.py:50`；`InsertParams` `python/sglang/srt/mem_cache/base_prefix_cache.py:61`；`MatchResult` `python/sglang/srt/mem_cache/base_prefix_cache.py:171` |
| `chunk_cache.py` | `--disable-radix-cache` 时的替代实现，语义上"完全不做前缀复用" | `ChunkCache` 类 `python/sglang/srt/mem_cache/chunk_cache.py:35`；`disable` 恒为 `True` `python/sglang/srt/mem_cache/chunk_cache.py:60`；`match_prefix` 直接返回空命中 `python/sglang/srt/mem_cache/chunk_cache.py:67` |
| `unified_radix_cache.py` | 生产默认路径实际选中的实现，2,894 行，组件化架构 | `UnifiedRadixCache` 类 `python/sglang/srt/mem_cache/unified_radix_cache.py:148`；`match_prefix` `python/sglang/srt/mem_cache/unified_radix_cache.py:499`；`insert` `python/sglang/srt/mem_cache/unified_radix_cache.py:518`；`evict` `python/sglang/srt/mem_cache/unified_radix_cache.py:537`；`inc_lock_ref` `python/sglang/srt/mem_cache/unified_radix_cache.py:683` |
| `registry.py` | 按 `ServerArgs` 选具体前缀缓存实现的工厂函数 | `default_radix_cache_factory` `python/sglang/srt/mem_cache/registry.py:80`；`ChunkCache` 分支 `python/sglang/srt/mem_cache/registry.py:93`；兜底选中 `UnifiedRadixCache` `python/sglang/srt/mem_cache/registry.py:143`；`create_tree_cache` 入口 `python/sglang/srt/mem_cache/registry.py:199` |
| `evict_policy.py` | 可插拔的驱逐优先级策略（LRU/LFU/FIFO/SLRU/Priority…） | `LRUStrategy` `python/sglang/srt/mem_cache/evict_policy.py:16`；`LFUStrategy` `python/sglang/srt/mem_cache/evict_policy.py:21`；`PriorityStrategy` `python/sglang/srt/mem_cache/evict_policy.py:41`；`SLRUStrategy` `python/sglang/srt/mem_cache/evict_policy.py:49` |
| `utils.py` | 策略工厂 + 哈希工具 | `get_eviction_strategy` `python/sglang/srt/mem_cache/utils.py:67`；`split_node_hash_value` `python/sglang/srt/mem_cache/utils.py:187` |
| `server_args.py` | 相关 CLI 旋钮（全文件 10,142 行，本子系统之外的最大文件） | `disable_radix_cache` `python/sglang/srt/server_args.py:966`；`radix_eviction_policy` `python/sglang/srt/server_args.py:949`；`page_size` `python/sglang/srt/server_args.py:921` |
| `memory_pool.py` | `TreeNode.value` 实际指向的物理存储（本子系统最大文件之一） | `ReqToTokenPool` `python/sglang/srt/mem_cache/memory_pool.py:256`；`KVCache` 抽象基类 `python/sglang/srt/mem_cache/memory_pool.py:1628`；`MHATokenToKVPool` `python/sglang/srt/mem_cache/memory_pool.py:1759` |
| `pure_swa_radix_cache.py` / `hiradix_cache.py` | 复用 `RadixCache` 的两个子类，证明它不是废弃代码 | `PureSWARadixCache(RadixCache)` `python/sglang/srt/mem_cache/pure_swa_radix_cache.py:24`；`HiRadixCache(RadixCache)` `python/sglang/srt/mem_cache/hiradix_cache.py:77` |

（以上共 30+ 条 `文件:行` 引用，超过 `## 2` 硬指标的 10 条。）

补充两点容易搞混的文件边界：

- **`radix_cache.py` vs `unified_radix_cache.py` 不是"旧版 vs 新版"的替换关系，而是"仍在并存的两套实现"。** 前者被 `HiRadixCache`/`PureSWARadixCache` 继承、被 `create_simulated` 用于纯 CPU 测试；后者是 `registry.py` 默认工厂链兜底选中的实现。读代码时如果想看"radix tree 算法本身"，看前者就够；想看"生产环境实际跑的是什么"，需要下钻到后者——第 7 节会给出这个判断的完整证据链。
- **`evict_policy.py` 和 `utils.py` 里的 `get_eviction_strategy` 不是 `RadixCache` 独有的。** `EvictionStrategy` 抽象类和它的几个具体策略（`LRUStrategy`/`LFUStrategy`/…）只依赖 `TreeNode` 的几个字段（`last_access_time`/`hit_count`/`creation_time`/`priority`），不依赖 `RadixCache` 类本身的任何方法，这是一处干净的策略模式（Strategy Pattern）——理论上任何持有 `TreeNode` 风格对象的缓存实现都可以复用这套策略而不需要继承 `RadixCache`。

## 3. 核心数据结构

### 3.1 `RadixKey`——匹配用的键，而不是原始 token 列表

`RadixKey`（`python/sglang/srt/mem_cache/radix_cache.py:59`）包一层 token 数组，带上 `extra_key`（LoRA/命名空间隔离）和 `cache_salt`（采样盐），保证"token 完全相同但语义不同"的两个请求不会共享树节点（`_check_compatible` 强制校验，`python/sglang/srt/mem_cache/radix_cache.py:169`）。它最重要的方法是 `match`：

```python
# python/sglang/srt/mem_cache/radix_cache.py:181
def match(self, other: RadixKey, page_size: int = 1) -> int:
    """Logical-unit prefix length shared with ``other``. Result is rounded down to ``page_size``."""
```

`match` 不是逐 token 循环比较，而是**倍增窗口的指数搜索**（`python/sglang/srt/mem_cache/radix_cache.py:191`-`206`）：

```python
# python/sglang/srt/mem_cache/radix_cache.py:191-206（节选，变量名与原文一致）
matched_tokens = n
lo = 0
step = 1
while lo < n:
    hi = lo + step if lo + step < n else n
    if t0[lo:hi] != t1[lo:hi]:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if t0[lo:mid] == t1[lo:mid]:
                lo = mid
            else:
                hi = mid
        matched_tokens = lo
        break
    lo = hi
    step *= 2
```

先按 1、2、4、8… 个 token 一批做整段切片比较（Python 层是 C 级别的数组切片相等判断，`array` 类型的 `!=`/`==` 不逐元素走 Python 字节码），一旦某个窗口 `t0[lo:hi] != t1[lo:hi]` 出现不相等，再在这个窗口内二分定位第一个分叉点。这比"一个 token 一个 token 比"快（长匹配区间只需要 `O(log n)` 次切片而不是 `O(n)` 次单 token 比较），但仍然是 `O(log(匹配长度))` 次操作而不是 vLLM 那种"一次哈希查表"的 `O(1)`——这是"树"这条路线从根子上要付出的代价，第 6 节会展开对比。

### 3.2 `TreeNode` 字段逐个讲

（`python/sglang/srt/mem_cache/radix_cache.py:238`）

```python
# python/sglang/srt/mem_cache/radix_cache.py:238-266
class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        self.write_through_pending_id: Optional[int] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # Namespace-aware hashes used only for external KV events.
        self.event_hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
```

| 字段 | 类型 | 为什么存在 |
|---|---|---|
| `key: RadixKey` | 变长 token 序列 | 节点的"身份"——这段 token 序列是从父节点到这个节点唯一的路径标签。和 vLLM 的块不同，长度不固定，只在两个方向上有边界：page_size 对齐（见 3.3）和分裂点。 |
| `value: torch.Tensor` | KV 槽位下标（int64） | 和 `key` 等长、逐位置对应的物理存储地址。`match_prefix` 命中之后，调度器直接把这段 `value` 拼进请求的 `req_to_token` 行，不需要重新计算 KV。 |
| `children: dict` | `{child_key: TreeNode}` | 用首 token（或 `page_size` 个 token 组成的元组，见 3.3）做 key 的哈希表，让"往下走一步"是 `O(1)` 期望而不是遍历所有兄弟节点。定义成 `defaultdict(TreeNode)`（`python/sglang/srt/mem_cache/radix_cache.py:243`），但实际所有读取点都先做了 `in node.children.keys()` 判断（`python/sglang/srt/mem_cache/radix_cache.py:686`、`759`），`defaultdict` 的自动创建行为从未被真正触发——第 7 节会说明这是一个潜在的地雷而非有意设计。 |
| `lock_ref: int` | 引用计数 | 有多少个"活跃请求"正把这个节点当作自己前缀链条的一部分。`>0` 时这个节点及其所有祖先都不能被驱逐（详见 4.3）。 |
| `last_access_time: float`（`time.monotonic()`） | 单调时钟时间戳 | LRU 驱逐用的优先级键。用 `monotonic()` 而不是 `time.time()`是为了不受系统时钟被 NTP 校准/手动调整影响（本库推断，是标准 Python 计时惯用法，未在注释里显式说明理由）。 |
| `creation_time` | 单调时钟时间戳 | 供 `FIFOStrategy`/`FILOStrategy` 使用（`python/sglang/srt/mem_cache/evict_policy.py:26`、`36`），和 `last_access_time` 分开存是因为"多久没被访问"和"多久前被创建"是两种不同的驱逐哲学，二者不能用同一个时间戳互相推导（一个节点可能创建后从未被再次命中，此时两者理应相等；但一旦被命中一次，二者就分叉了）。 |
| `hit_count: int` | 命中次数 | 供 `LFUStrategy`/`SLRUStrategy` 使用（`python/sglang/srt/mem_cache/evict_policy.py:21`、`49`）；`_inc_hit_count` 在分块请求（`chunked=True`）时故意跳过计数（`python/sglang/srt/mem_cache/radix_cache.py:730`），避免同一个请求的多个 chunk 把自己制造出来的节点反复给自己加命中数，制造虚假的"热点"。 |
| `priority: int` | 请求携带的优先级 | 供 `PriorityStrategy` 使用；插入时沿路径取 `max`（`python/sglang/srt/mem_cache/radix_cache.py:752`），根节点初始化成 `-sys.maxsize`（`python/sglang/srt/mem_cache/radix_cache.py:356`）保证任何真实请求的优先级都压不过根——虽然根本身因为 `lock_ref=1` 永远不会被驱逐，这层保险更多是防御性的。 |
| `host_value`/`host_ref_counter`/`hash_value`/`write_through_pending_id` | HiCache 相关字段 | 服务分层缓存（GPU/CPU/磁盘）时才用得上，本篇不展开，留给 [[11-SGLang-PD分离与HiCache分层]]。 |

`TreeNode.__lt__`（`python/sglang/srt/mem_cache/radix_cache.py:299`）只比较 `last_access_time`——这是因为驱逐用的堆存的是 `(priority_tuple, node)` 二元组（见 4.3），当两个节点的 `priority_tuple` 相同（比如 LRU 策略下两个节点时间戳恰好相等，或 SLRU 策略下 segment 与 hit_count 都一样）时，`heapq` 会退化去比较元组的第二个元素——也就是 `TreeNode` 对象本身，如果没有 `__lt__`，这里会直接抛 `TypeError`。

`TreeNode` 还有两个属性和两个方法值得单独一提，因为它们揭示了"驱逐"在这份代码里实际上分两个独立的轴：

```python
# python/sglang/srt/mem_cache/radix_cache.py:268-285
@property
def evicted(self):
    return self.value is None

@property
def backuped(self):
    return self.host_value is not None

def protect_host(self):
    """Protect the host value from eviction."""
    self.host_ref_counter += 1

def release_host(self):
    """Release the host value, allowing it to be evicted."""
    if self.host_ref_counter > 0:
        self.host_ref_counter -= 1
    else:
        raise RuntimeError("Host reference counter is already zero.")
```

`evicted` 判定的是"GPU 上的 `value` 还在不在"，`backuped` 判定的是"CPU/磁盘上的 `host_value` 备份还在不在"——本篇精读的纯 `RadixCache` 路径下 `host_value` 恒为 `None`（`backuped` 恒为 `False`），这一对属性真正发挥作用是在 HiCache 分层场景：一个节点可以"GPU 上已驱逐但 CPU 上还有备份"（`evicted=True, backuped=True`），这时候重新命中它不需要重算，只需要把 `host_value` 从 CPU 搬回 GPU。`protect_host`/`release_host` 是给这层 CPU 备份单独准备的引用计数（`host_ref_counter`），和保护 GPU `value` 的 `lock_ref` 是两套独立账本——一个节点可以 GPU 侧没有任何请求在用（`lock_ref=0`，允许被 GPU 侧驱逐），但 CPU 侧的备份因为正在被写回流程占用而不能删（`host_ref_counter>0`）。这套双层保护机制本篇不再展开，留给 [[11-SGLang-PD分离与HiCache分层]]。

### 3.3 `page_size` 怎么进入匹配

`RadixKey.page_aligned`（`python/sglang/srt/mem_cache/radix_cache.py:150`）在 `match_prefix`/`insert` 入口处把请求的 key 长度向下取整到 `page_size` 的整数倍；`RadixKey.match` 的返回值同样按 `page_size` 取整（`python/sglang/srt/mem_cache/radix_cache.py:213`-`215`）。`page_size == 1` 时这两步都是恒等操作。这意味着树节点的分裂边界**永远落在 page 边界上**，不会出现"半个 page 归 A 半个 page 归 B"的情况——这条约束和 vLLM 的"块是原子单位"看似矛盾，其实是同一个物理约束（底层 KV 存储的分配粒度）在两种数据结构里各自的投影，第 6 节会把这一点讲透。

### 3.4 `is_bigram`：EAGLE 投机解码借用同一棵树的方式

`RadixKey` 的 `__slots__`（`python/sglang/srt/mem_cache/radix_cache.py:62`）里有一个不太直观的字段 `is_bigram`，配合 `maybe_to_bigram_view`（`python/sglang/srt/mem_cache/radix_cache.py:156`-`167`）使用：

```python
# python/sglang/srt/mem_cache/radix_cache.py:156-167
def maybe_to_bigram_view(
    self,
    is_eagle: bool,
    value: Optional[torch.Tensor] = None,
) -> Tuple[RadixKey, Optional[torch.Tensor]]:
    # O(1): flip the bigram flag instead of materializing a tuple list.
    # value is paired with raw tokens and gets truncated to the bigram count.
    if is_eagle and not self.is_bigram:
        self.is_bigram = True
        if value is not None:
            value = value[: len(self)]
    return self, value
```

`is_bigram=True` 时，`RadixKey` 不再把单个 token 当作匹配单位，而是把**相邻两个 token 组成的 pair**（bigram）当作匹配单位——`__len__`（`python/sglang/srt/mem_cache/radix_cache.py:99`-`103`）和 `__iter__`（`python/sglang/srt/mem_cache/radix_cache.py:106`-`116`）在这个模式下都会返回"少一个"的逻辑长度（`N` 个 token 只有 `N-1` 个 bigram）。`match_prefix`/`insert` 的公共入口（`python/sglang/srt/mem_cache/radix_cache.py:415`、`446`）在 `self.is_eagle` 为真时都会先调用 `maybe_to_bigram_view` 切换视图，**这样同一棵 `RadixCache` 既能服务普通请求，也能服务 EAGLE 投机解码请求，不需要为后者单独建一棵树**——这是"用同一套 `key.child_key()`/`key.match()` 接口，通过一个布尔标志切换匹配粒度"的复用设计，本篇不展开 EAGLE 本身的算法，留给 [[10-SGLang-投机解码EAGLE]]。

### 3.5 调用契约：`MatchPrefixParams`/`MatchResult`/`InsertParams`

`RadixCache` 不是直接接收 token 列表的裸函数接口，而是走一套所有前缀缓存实现（`ChunkCache`/`RadixCache`/`UnifiedRadixCache`……）共享的参数/结果数据类（`python/sglang/srt/mem_cache/base_prefix_cache.py`）。`InsertParams`（`python/sglang/srt/mem_cache/base_prefix_cache.py:61`）除了 `key`/`value` 之外还挂了 `mamba_value`、`c128_value`、`prev_prefix_len`、`swa_evicted_seqlen` 等字段——这些是 `UnifiedRadixCache` 为了支持混合模型（Mamba/DSA/SWA）加上去的，`RadixCache` 只读 `key`/`value`/`chunked`/`priority` 四个字段，其余字段对它是死代码，从一个侧面印证了"这是一套为组件化设计预留、但 `RadixCache` 本身只用了最小子集"的接口。`MatchResult`（`python/sglang/srt/mem_cache/base_prefix_cache.py:171`）同理：`swa_host_hit_length`、`mamba_host_hit_length`、`mamba_branching_seqlen` 这些字段在 `RadixCache.match_prefix`（`python/sglang/srt/mem_cache/radix_cache.py:430`-`435`）里全部保持默认值 `0`/`None`，只有 `device_indices`/`last_device_node`/`last_host_node`/`best_match_node` 四个字段被真正赋值——`last_host_node` 在没有 HiCache 时**必须**和 `last_device_node` 相等，这一点在字段注释里明确写了（`python/sglang/srt/mem_cache/base_prefix_cache.py:177`-`179`）。

## 4. 主流程走读——一次分裂的完整推演

### 4.1 从空树开始

文件末尾自带一段可运行 demo（`python/sglang/srt/mem_cache/radix_cache.py:849`-`863`），依次插入 `[1,2,3]`、`[1,2,3]`（重复）、`[1,2,4,5]`、`[1,2,4,5,6,7]`、`[8,9,10,11,12]`，最后查询 `[1,2,3,13,14]`。下面按 `_insert_helper`（`python/sglang/srt/mem_cache/radix_cache.py:738`）与 `_split_node`（`python/sglang/srt/mem_cache/radix_cache.py:705`）的实际逻辑手工推演，`page_size=1`。

**插入请求 A = `[1,2,3]`（空树）**

`root.children` 里没有以 `1` 开头的条目，直接在根下挂一个新叶子：

```
root
└── [1,2,3]  (A, lock_ref=0)
```

**插入请求 B = `[1,2,4,5]`（关键：与 A 共享前 2 个 token，第 3 个分叉）**

沿树下行：`root.children[1]` 命中 A。`A.key.match([1,2,4,5])` 返回 `2`（token `1,2` 相同，`3` 与 `4` 在 index 2 分叉）。因为 `prefix_len(2) < len(A.key)(3)`，触发 `_split_node(key=A.key, child=A, split_len=2)`（`python/sglang/srt/mem_cache/radix_cache.py:705`-`728`）：

1. 新建节点 `P`，`P.key = A.key[:2] = [1,2]`，`P.value = A.value[:2].clone()`——**只拷贝分裂点之前那一段**，不碰 A 原来 `[3]` 对应的那部分 KV 索引。
2. 原地改写 `A`：`A.key = A.key[2:] = [3]`，`A.value = A.value[2:].clone()`，`A.parent = P`。
3. `P.parent = A` 原来的父节点（这里是 `root`），`root.children[1]` 从指向 `A` 改成指向 `P`；`P.children[3] = A`。

对照 `_split_node` 的完整实现（`python/sglang/srt/mem_cache/radix_cache.py:705`-`728`），上面三步分别对应下面这段代码：

```python
# python/sglang/srt/mem_cache/radix_cache.py:705-728
def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
    # new_node -> child
    # New node inherits child's priority (represents shared prefix)
    new_node = TreeNode(priority=child.priority)
    new_node.hit_count = child.hit_count
    new_node.children = {key[split_len:].child_key(self.page_size): child}
    new_node.parent = child.parent
    new_node.lock_ref = child.lock_ref
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len].clone()
    child.parent = new_node
    child.key = child.key[split_len:]
    child.value = child.value[split_len:].clone()
    new_node.parent.children[key.child_key(self.page_size)] = new_node

    # Split hash_value if it was already computed, otherwise leave as None
    new_node.hash_value, child.hash_value = split_node_hash_value(
        child.hash_value, split_len, self.page_size
    )
    new_node.event_hash_value, child.event_hash_value = split_node_hash_value(
        child.event_hash_value, split_len, self.page_size
    )

    return new_node
```

注意 `new_node.lock_ref = child.lock_ref` 这一行（第 712 行）：分裂时新父节点**继承**原节点当时的锁引用计数，而不是从 0 开始——这是必要的，因为如果 `A` 在分裂前正被某个运行中请求锁住（`lock_ref=1`），分裂之后这把锁在逻辑上属于"从根到 A 这条路径"，`P` 作为这条路径新插入的一环，必须同样被认为是"锁住的"，否则驱逐算法会把 `P` 误判成可驱逐节点，进而破坏 `A` 依赖的前缀完整性。`hit_count` 同理被完整复制给新节点（第 709 行），保留原节点的"热度"历史，而不是让分裂后的父节点从零热度重新累积。

```
root
└── [1,2]  (P) ── A、B 的共享前缀
    └── [3] (A) ── A 的私有尾部
```

回到 `_insert_helper`，剩余 key `[4,5]` 在 `P.children` 里找不到以 `4` 开头的条目，于是新建叶子 `C`：

```
root
└── [1,2]  (P)
    ├── [3]   (A)
    └── [4,5] (C) ── B 的私有尾部
```

**这就是两个请求"共享前 N 个 token、第 N+1 个分叉"时树的完整改法**：分裂发生在共享前缀的终点，产生一个新的共享父节点 `P`，原节点 `A` 收缩成只保留自己的私有尾部，新请求的私有尾部作为 `P` 的另一个子节点插入。全程只有 `P` 和 `A` 两次 tensor `.clone()`（各 `O(split_len)`），没有任何数据被复制两次或丢弃重建。

**继续插入 `[1,2,4,5,6,7]`**：命中 `P`（完全匹配 `[1,2]`）→ 命中 `C`（完全匹配 `[4,5]`，因为 `[4,5]` 恰好是 `C.key` 的全部，不需要分裂）→ `C.children` 里没有以 `6` 开头的条目，新建叶子 `D=[6,7]`。

**继续插入 `[8,9,10,11,12]`**：`root.children` 里没有以 `8` 开头的条目，直接挂新叶子 `E`。

最终树形：

```
root
└── [1,2] (P)
    ├── [3]   (A)
    └── [4,5] (C)
        └── [6,7] (D)
    [8,9,10,11,12] (E, 挂在 root 下)
```

### 4.2 `match_prefix` 沿树下行

`_match_prefix_helper` 的完整实现（`python/sglang/srt/mem_cache/radix_cache.py:679`-`703`）：

```python
# python/sglang/srt/mem_cache/radix_cache.py:679-703
def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
    access_time = time.monotonic()
    node.last_access_time = access_time

    child_key = key.child_key(self.page_size)

    value = []
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        child.last_access_time = access_time
        prefix_len = child.key.match(key, page_size=self.page_size)
        if prefix_len < len(child.key):
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            value.append(child.value)
            node = child
            key = key[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

    return value, node
```

对照查询 `[1,2,3,13,14]`：`root.children[1]` → `P`，`P.key.match([1,2,3,13,14])=2`，等于 `len(P.key)`（完全匹配，走 `else` 分支不分裂），前进；剩余 key `[3,13,14]`，`P.children[3]` → `A`，`A.key.match([3,13,14])=1`，等于 `len(A.key)`（完全匹配），前进；剩余 key `[13,14]`，`A.children` 里没有以 `13` 开头的条目，`while` 循环条件 `child_key in node.children.keys()` 为假，循环终止。返回 `value = concat(P.value, A.value)`（对应原始 token `[1,2,3]`）、`last_node = A`。

这段代码里 `if prefix_len < len(child.key)` 分支（第 690-693 行）就是"匹配在节点内部终止"的判定：一旦触发就立刻 `break`，说明**一次 `match_prefix` 调用最多触发一次分裂**——分裂只可能发生在遍历路径的最后一步，不会在下行过程中反复分裂已经走过的节点。

**如果查询在节点内部终止会怎样**：`match_prefix` 和 `insert` 走的是同一套"匹配到中途 `prefix_len < len(child.key)`就分裂"的逻辑（`python/sglang/srt/mem_cache/radix_cache.py:690`、`767`）——也就是说，**单纯的读操作（match_prefix）也可能触发写副作用（分裂树结构）**。这是一个容易在并发场景下被忽略的细节：函数注释里也明确写了"This method may mutate internal structure by splitting an existing node"（`python/sglang/srt/mem_cache/radix_cache.py:404`）。分裂本身不改变任何已缓存数据的语义（只是把一个节点在逻辑上切两段），但如果调用方以为 `match_prefix` 是纯读操作、在没持锁的情况下并发调用，可能观察到树结构在两次调用之间发生变化。

### 4.3 引用计数怎么保护正在跑的请求

`inc_lock_ref`（`python/sglang/srt/mem_cache/radix_cache.py:623`）从命中节点开始，沿 `parent` 指针一路走到根，每个节点 `lock_ref += 1`；第一次从 `0` 变成 `1` 的节点，把自己的 `len(key)` 从 `evictable_size_` 转移到 `protected_size_`。`dec_lock_ref`（`python/sglang/srt/mem_cache/radix_cache.py:638`）做镜像操作。调度器在请求开始跑之前对 `match_prefix` 返回的 `last_node` 调 `inc_lock_ref`，请求结束或让出时调 `dec_lock_ref`（例子见 `cache_unfinished_req`，`python/sglang/srt/mem_cache/radix_cache.py:571`-`572`：先 `dec_lock_ref(req.last_node)` 释放旧节点，再 `inc_lock_ref(new_last_node)` 锁住新节点）。

关键点：**锁的是从叶子到根的整条路径，不是单个节点**。因为子节点被锁住时，它的存在依赖父节点的 `key`/`value` 完整（父节点是它的前缀），如果只锁子节点不锁父节点，驱逐算法可能会把父节点删掉，导致子节点的语义悬空。

### 4.4 驱逐：从叶子开始的堆

`evict`（`python/sglang/srt/mem_cache/radix_cache.py:593`）：

```python
leaves = list(self.evictable_leaves)
eviction_heap = [(self.eviction_strategy.get_priority(node), node) for node in leaves]
heapq.heapify(eviction_heap)
while num_evicted < num_tokens and len(eviction_heap):
    _priority, x = heapq.heappop(eviction_heap)
    self.token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)
    num_evicted += len(x.value)
    self._delete_leaf(x)
    if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
        new_priority = self.eviction_strategy.get_priority(x.parent)
        heapq.heappush(eviction_heap, (new_priority, x.parent))
```

`self.evictable_leaves` 是一个**增量维护**的集合，不是每次驱逐时现算的——`_update_leaf_status`（`python/sglang/srt/mem_cache/radix_cache.py:821`）在每次插入、加锁、解锁、删除后被调用，判断一个节点是否同时满足"未被驱逐（有 value）"、"`lock_ref==0`"、"没有未驱逐的子节点"三个条件，是则加入集合，否则移出。这让 `evict` 入口不需要一次 DFS 扫全树找叶子，直接 `list(self.evictable_leaves)` 拿到候选集。

驱逐一个叶子后，检查它的父节点是否**因此**变成了新叶子（子节点都被删空，且没被锁住），是的话把父节点也推进堆——这就是"沿树坍缩"：一条从未被复用过的长前缀链，会像剥洋葱一样从最深的叶子往根方向逐层被驱逐，而不是被当成一个整体一次性回收。

## 5. 设计决策与代价

**决策 1：变长树节点而非定长块**

- *为什么这么设计*：请求之间共享前缀的实际长度是任意的（系统提示词、多轮对话历史、few-shot 示例都不会天然对齐到某个固定块大小），用变长节点可以让分裂点精确落在"两个请求真正开始不同"的那个 token 上，不多复用、也不少复用。
- *不这样会怎样*：如果用定长块，两个请求即使前 4097 个 token 完全相同、只是块大小设成 4096，也会在第二块整体判定为不命中（因为第二块里前 1 个 token 相同、后面全不同，哈希不等），白白浪费 4095 个 token 的可复用前缀（除非引入 vLLM 那样的细粒度块内探测，见第 6 节）。
- *什么时候可以不这样*：如果上游流量本身就是"要么完全不共享、要么整段整段共享"（比如离线批处理跑同一个长模板的 N 份变体，变体只在结尾几个 token 不同），定长块的浪费趋近于零，此时块方案的实现简单性和内存局部性反而更有利，见 6.3。

**决策 2：`_split_node` 只拷贝分裂点之前的那一段**

- *为什么这么设计*：分裂点之后的数据本来就属于原节点收缩后的私有部分，没有理由动它；只对分裂点之前的那段 `value` 做 `.clone()` 给新的父节点，是把"必须复制的量"压到最小（`O(split_len)` 而不是 `O(len(child.key))`）。
- *不这样会怎样*：如果分裂时把整个节点的 `value` 都复制一遍（包括尾部），内存和时间开销会随节点长度线性增长而不是随分裂点位置增长，在长上下文、深树场景下会显著变慢。
- *什么时候可以不这样*：如果 KV 存储本身允许"视图切片而不实际拷贝底层内存"（比如某种引用计数的 tensor view），这两次 `.clone()` 可以进一步省掉——这正是第 8 节的一个可改进方向，目前代码库里没有查到这样的实现（未查证）。

**决策 3：`lock_ref` 沿整条路径传播而不是只记叶子**

- *为什么这么设计*：驱逐算法只看叶子，如果祖先节点没有被标记为"被占用"，一个正在被某个运行中请求依赖的中间节点可能因为它自己"看起来"不是叶子而被忽略保护检查，进而在它的某个未被使用的兄弟分支被清空后意外暴露成叶子并被驱逐。整条路径加锁把这个风险从"需要额外分析"变成"结构上不可能发生"。
- *不这样会怎样*：只锁叶子的话，`_update_leaf_status` 判断"是否所有子节点都已驱逐"这一步会失真——父节点可能因为某个未加锁的兄弟分支被删空而被误判为"可驱逐叶子"，即使它自己的另一个子节点正被使用。
- *什么时候可以不这样*：如果树的分支因子恒为 1（退化成链表，不会有"兄弟节点"这个概念），只锁叶子和锁整条路径是等价的——但这种情况下用树本身就没有意义了。

**决策 4：leaf 优先的堆式 LRU，而不是全局排序表**

- *为什么这么设计*：可驱逐的候选集合（`evictable_leaves`）本身就随插入/命中/解锁增量变化，用堆维护"当前候选里最该被淘汰的是谁"，避免每次驱逐都重新排序整个树；leaf 优先保证了"父节点在其所有子节点都被清空前不会被淘汰"，这是维持树结构完整性的硬约束，不是优化选项。
- *不这样会怎样*：如果允许直接淘汰非叶子节点（比如单纯按 `last_access_time` 全局排序，不管是不是叶子），会出现"父节点被删了，但它的子节点还指向一个已经不存在的父"这种悬空引用，除非同时递归删除整棵子树——但那样会连带清掉可能还有效的、只是暂时没被访问的子孙节点，这违背 LRU"只淘汰最不该留的那一个单位"的初衷。
- *什么时候可以不这样*：`--radix-eviction-policy` 支持切到 `lfu`/`slru`/`priority`（`python/sglang/srt/server_args.py:949`-`960`），换的只是堆里 `get_priority` 返回的排序键（`python/sglang/srt/mem_cache/evict_policy.py:16`-`63`），"leaf 优先 + 堆" 这个骨架本身在所有策略下都不变——这说明"堆" 和 "LRU" 是两个正交的选择，堆负责满足树结构约束，`LRUStrategy` 只是堆里默认的排序函数。

**决策 5：`match` 用倍增窗口指数搜索而不是逐 token 循环或 Python 内建的 `zip` 比较**

- *为什么这么设计*：token 序列在共享前缀场景下经常是"要么很长一段完全相同，要么很快分叉"，指数搜索对"长匹配"场景是 `O(log n)` 次 C 级别切片比较，比 `for i in range(n): if t0[i]!=t1[i]` 这种 Python 逐元素循环快得多——后者哪怕匹配几千个 token，也要在 Python 字节码层面跑几千次循环迭代。
- *不这样会怎样*：如果改成朴素逐 token 比较，长系统提示词（几千 token）场景下每次 `match_prefix`/`insert` 调用都要多付出与匹配长度成正比的纯 Python 解释开销，这在高 QPS 场景下会成为 CPU 侧调度瓶颈的一部分（调度器每步都要为每个新请求跑一次前缀匹配）。
- *什么时候可以不这样*：如果 `page_size` 很大（比如以后出现 `page_size=64` 这样的配置）且树本身很浅（分叉很少），单次 `match` 调用处理的 token 数量有限，指数搜索的优势会被它自身的分支判断开销抵消——但这只是理论上的边界情况，本篇没有找到 SGLang 对小规模输入做特殊分支优化的证据（未查证）。

**决策 6：一条边可以压缩多个 token，而不是经典 trie"一层一个字符"**

- *为什么这么设计*：如果严格按经典字典树（trie）的写法，每个 `TreeNode` 只对应 1 个 token，那么插入请求 A=`[1,2,3]` 会创建 3 层节点而不是 1 层；`RadixCache` 的 `TreeNode.key` 允许是任意长度的 token 序列，只有出现分叉时才会分裂出新边界（第 4.1 节的推演里，`[1,2,3]` 在没有分叉之前始终是**一个**节点，直到 `[1,2,4,5]` 插入才裂开成 `[1,2]+[3]`）。这正是"radix tree"（基数树/PATRICIA trie）相对朴素 trie 的定义性优化：压缩掉所有只有单一子节点的中间层。
- *不这样会怎样*：朴素 trie 在共享前缀很长、分叉很少的场景下（典型如长系统提示词）会创建大量只有一个子节点的"直链"节点，每个节点单独占一份 Python 对象开销（`TreeNode.__init__` 里 `last_access_time`/`creation_time`/`children` 等字段都要初始化一次），相当于用节点数量把本该是 `O(1)` 个对象的信息拆成了 `O(前缀长度)` 个对象。
- *什么时候可以不这样*：如果几乎每个 token 位置都会被后续某个请求在该位置分叉（也就是树的分支因子处处很高），压缩边和不压缩边的节点数量趋于相同，此时"边压缩"这个优化几乎不产生收益——但代价也没有变坏，所以这更像是一个"总是划算或持平，几乎不可能变差"的设计，不像前面几条决策那样存在真正的取舍。

**决策 7：提供 `--disable-radix-cache` 整体关掉树，退化成 `ChunkCache`**

- *为什么这么设计*：维护一棵树本身有成本——每次请求结束/续跑都要走一遍 `match_prefix`/`insert`（`O(树深)` 查找 + 可能的分裂），如果上游流量本身几乎不存在共享前缀（比如每个请求都是完全独立的一次性 embedding 计算，或者压测场景故意构造互不重叠的输入），这些开销纯粹是浪费，不产生任何命中收益。`ChunkCache`（`python/sglang/srt/mem_cache/chunk_cache.py:35`）就是为这种场景准备的极简替代：`match_prefix` 恒返回空命中，`insert` 是纯 no-op：

  ```python
  # python/sglang/srt/mem_cache/chunk_cache.py:67-77（节选）
  def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
      return MatchResult(
          device_indices=torch.empty((0,), dtype=torch.int64),
          last_device_node=None,
          last_host_node=None,
          best_match_node=None,
      )

  def insert(self, params: InsertParams) -> InsertResult:
      # ChunkCache does not support prefix caching, so insert is a no-op
      return InsertResult(prefix_len=0)
  ```

  `disable` 属性硬编码为 `True`（`python/sglang/srt/mem_cache/chunk_cache.py:60`-`62`），`evict`/`inc_lock_ref` 也都是直接返回零值的空实现（`python/sglang/srt/mem_cache/chunk_cache.py:96`-`100`）——没有树、没有堆、没有引用计数账本，KV 槽位的生命周期完全跟着请求的生命周期走，请求结束立刻整段释放（`cache_finished_req`，`python/sglang/srt/mem_cache/chunk_cache.py:79`-`87`）。
- *不这样会怎样*：如果没有这条开关，即使流量里完全没有可复用前缀，调度器每一步仍然要为每个请求走一次 `_match_prefix_helper` 的树遍历（哪怕大概率一步就在根节点 miss）、`_insert_helper` 的节点创建与 `_update_leaf_status` 维护、以及 `evictable_leaves` 集合的增删——这些操作单次都不贵，但在高 QPS、短请求场景下会累积成不可忽视的 CPU 侧调度开销，而且这部分开销完全没有对应的命中收益。
- *什么时候可以不这样（即什么时候该真的关掉它）*：CLI 参数是 `--disable-radix-cache`（`python/sglang/srt/server_args.py:966`-`968`，帮助文本原文 "Disable RadixAttention for prefix caching."），触发条件由 `registry.py::default_radix_cache_factory` 判定——当 `ctx.disable_radix_cache` 为真且请求了分块预填充（`ctx.effective_chunked_prefill_size is not None`）时才会真的选中 `ChunkCache`（`python/sglang/srt/mem_cache/registry.py:91`-`102`），否则即使传了这个开关，某些路径仍然会退回 `UnifiedRadixCache`（`python/sglang/srt/mem_cache/registry.py:87`-`89`，服务于 host-pool 回退场景）——**开关的实际生效条件比"传了参数就一定关闭"更复杂**，这是本篇核对源码后才发现的细节，只读 `--help` 文本容易漏掉。典型该关的场景：纯粹的一次性 embedding/scoring 服务、每个请求输入都不重叠的压测基线测量（想单独测"不算前缀缓存"时的吞吐上限）、或者显存本身就紧张到连一个最小可用的树都嫌贵的极端部署。

## 6. 同位对照：vLLM 的块哈希表 vs SGLang 的基数树

### 6.1 vLLM 一侧的数据结构证据

vLLM V1 用一张扁平哈希表做前缀缓存：`BlockHashToBlockMap`（`` `vllm:vllm/v1/core/block_pool.py:33` ``）把 `{block_hash: KVCacheBlock}` 存成普通 dict；块哈希由"父块哈希 + 本块全部 token id + 额外键"链式构成（`hash_block_tokens`，`` `vllm:vllm/v1/core/kv_cache_utils.py:620` ``）；命中判定是从第一个块开始顺着哈希链查表，**查不到就立即停**（`FullAttentionManager.find_longest_cache_hit`，`` `vllm:vllm/v1/core/single_type_kv_cache_manager.py:684` ``，第 733 行开始的循环：`for block_hash in itertools.islice(...)`, `cached_block = block_pool.get_cached_block(block_hash, ...)`, `if not cached_block: break`）。

这个"链式哈希、遇miss即停"的设计，正确性建立在一个前提上：**块必须先满（`curr_block_token_ids` 是完整的一块），哈希才有意义**——一个只填了一半的块永远不会进哈希表（`cache_full_blocks`，`` `vllm:vllm/v1/core/block_pool.py:225`ff ``）。

### 6.2 SGLang 一侧的数据结构证据

`TreeNode.children` 是 `{首 token(或 page_size 个 token 的元组): TreeNode}`（`python/sglang/srt/mem_cache/radix_cache.py:243`），树深由实际共享前缀的层级决定，不由固定块大小决定；命中判定是从根开始逐层查 `children` 字典（每层 `O(1)` 期望），层数等于"匹配到的公共前缀里出现过多少次分叉"，不等于"匹配长度 / page_size"（`_match_prefix_helper`，`python/sglang/srt/mem_cache/radix_cache.py:679`）。当匹配在某个节点内部终止时，用 `_split_node` 在线性时间内把命中边界精确暴露出来（第 4 节已推演）。

### 6.3 复杂度与内存的正面对照

| 维度 | vLLM（块哈希表） | SGLang（基数树） |
|---|---|---|
| 命中判定复杂度 | `O(前缀长度 / block_size)` 次哈希表查找，每次 `O(1)` 期望 | `O(树深)` 次字典查找，每次 `O(1)` 期望；树深上界是 token 数，下界取决于实际分叉次数 |
| 命中粒度 | 恒为 `block_size` 的整数倍（新版本加了 `alignment_tokens` 细粒度探测，见 `` `vllm:vllm/v1/core/single_type_kv_cache_manager.py:713`ff ``，本篇不展开） | 恒为 `page_size` 的整数倍，但节点长度本身随实际共享前缀自适应，不受固定块大小限制 |
| 结构维护开销 | 插入即写一个哈希表条目，`O(1)`；从不需要"重组已有条目" | 分裂需要新建节点、`.clone()` 两段 tensor、重接父子指针，`O(split_len)` |
| 内存对象开销 | `KVCacheBlock` 数组在启动时一次性预建（`` `vllm:vllm/v1/core/kv_cache_utils.py:162` ``），之后只搬指针，不再创建 Python 对象 | 每次分叉都新建一个 `TreeNode` Python 对象（`TreeNode.counter` 自增，`python/sglang/srt/mem_cache/radix_cache.py:265`），对象数量随分叉次数增长，GC 压力更大 |
| 跨进程/跨层传输的天然接口 | 块哈希本身就是可以脱离单进程传给 KV Connector 复用的标识符（见 [[04-vLLM-KV缓存与前缀缓存]] §6） | 需要额外的 hash_value 字段（`TreeNode.hash_value`，`python/sglang/srt/mem_cache/radix_cache.py:259`）才能获得类似语义，服务于 HiCache/KV 事件，不是树结构本身自带的 |

### 6.4 什么工作负载下哪一方更划算

**块哈希表更划算的场景**：请求间共享前缀天然对齐到较大粒度（长系统提示词 + 结尾少量差异化 token，比如同一个 agent 模板反复调用），此时块级命中已经能吃掉几乎全部可复用长度，分裂带来的"精确到 token 级"的额外收益趋近于零，而块哈希表"预建数组只搬指针、从不建新对象"的低开销优势是纯赚。这也是为什么它更容易和跨进程 KV 传输（P/D 分离）对接——哈希本身就是可以直接发出去的扁平标识符。

**基数树更划算的场景**：请求间的分叉点不对齐、且分叉频繁——比如多轮对话在任意历史位置被编辑重发、大量 few-shot / in-context 样例以不同顺序拼接、投机解码/树形搜索场景下同一个前缀有大量变长分支。此时块哈希表要么在分叉点前一个块整体判失败（浪费该块内已经相同的部分），要么需要额外的细粒度探测逻辑来弥补；基数树天然把"最长公共前缀"作为一次树遍历的直接产物，不需要额外机制。

代价是反过来的：分叉越频繁，`TreeNode` 对象创建、`.clone()` 拷贝、树深增长带来的查找开销也越高——**基数树把"块哈希表要花在细粒度探测上的复杂度"转移成了"分裂与遍历的复杂度"，不是免费获得任意粒度命中的**。哪一方净胜，取决于具体流量的分叉密度，本库没有可执行的基准测试来给出量化阈值（本机无 GPU，且需要真实流量分布），这里只给出结构性证据，不产出"谁比谁快 X 倍"这类没有口径的数字。

### 6.5 同一批请求，两种数据结构长什么样

以第 4 节推演的 A=`[1,2,3]`、B=`[1,2,4,5]` 为例（`block_size = page_size = 2` 便于对齐对比），两条路线对同一份数据的组织方式完全不同：

```
vLLM：一张扁平哈希表（block_size=2）
  A 切成块: [1,2] [3]         —— 第二块不满 block_size，不会进哈希表参与前缀缓存
  B 切成块: [1,2] [4,5]
  cached_block_hash_to_block = {
      hash([1,2])            : Block#7,   # A、B 共用同一个块——因为块内容完全相同
      hash([1,2]) + [4,5]    : Block#9,   # B 的第二块，父哈希链到 Block#7
  }
  查 B 的命中: 查 hash([1,2]) 命中 Block#7 → 查 hash([1,2])+[4,5] 命中 Block#9 → 命中 4 个 token
  查 A 的命中: 查 hash([1,2]) 命中 Block#7 → A 只有 3 个 token，第 3 个 token [3] 不满一块，不参与哈希查找，
              只能命中 2 个 token（第 3 个 token 要重新计算，即使这是 A 自己已经算过的）

SGLang：一棵基数树（page_size=2 只影响分裂点对齐，不影响节点长度本身）
  root
  └── [1,2] (P，A、B 共享)
      ├── [3]   (A 的私有尾部，长度 1，不要求对齐到 page_size)
      └── [4,5] (B 的私有尾部)
  查 B 的命中: root→P→C，命中 4 个 token
  查 A 的命中: root→P→A，命中 3 个 token —— A 自己的第 3 个 token 不需要凑够一整个 page 就能命中
```

这个对比暴露了 6.3 表格里"命中粒度"那一行的实际含义：vLLM 的块必须整块填满才能进入哈希表参与后续复用，A 请求自己重新访问自己缓存过的第 3 个 token 时，因为这个 token 落在一个不满的块里，**不会被当作前缀命中**，要重新计算（除非用第 6.3 表格提到的 `alignment_tokens` 细粒度探测机制去弥补，这属于 vLLM 一侧的后续优化，本篇不展开）；而 SGLang 的树节点长度不受 page_size 限制，`A` 节点的私有尾部 `[3]` 长度是 1，照样能被完整命中。反过来，SGLang 为此付出的代价是维护了两个额外的 `TreeNode` Python 对象（`P` 和 `A` 分裂后各一个），vLLM 只多写了一条哈希表目录项。

## 7. 踩坑与反直觉

1. **默认生产路径不是本篇精读的这份代码。** `registry.py::default_radix_cache_factory` 的兜底分支直接构造 `UnifiedRadixCache`（`python/sglang/srt/mem_cache/registry.py:143`），全文件搜索找不到任何地方直接把 `RadixCache(` 构造出来用于常规服务路径——`RadixCache` 目前的角色是 `HiRadixCache`（`python/sglang/srt/mem_cache/hiradix_cache.py:77`）和 `PureSWARadixCache`（`python/sglang/srt/mem_cache/pure_swa_radix_cache.py:24`）的父类，以及 `create_simulated`（`python/sglang/srt/mem_cache/radix_cache.py:334`）提供的纯 CPU 模拟/测试入口。本篇选它精读，是因为它是全库里唯一一份"匹配-分裂-驱逐"算法自包含在 800 多行、没有组件化间接层的实现——读懂它等于读懂了 `UnifiedRadixCache` 的算法内核，只是后者把同样的骨架摊开到 2,894 行去支持 SWA/Mamba/C128 多分量协同和 HiCache 分层。
2. **`HiRadixCache` 本身没有在默认工厂链里被直接实例化，但也不是死代码。** 全仓库搜索 `HiRadixCache(` 只在它自己的类定义处出现一次；但它被 `hybrid_pool_assembler.py` 当作类型标注使用、被 `unified_radix_cache.py` 的注释引用为"行为对照对象"（第 2650 行附近："Mirrors `HiRadixCache` so the disagg decode restore state machine…"）、并有专门的单元测试 `test/registered/unit/mem_cache/test_hiradix_cache_unit.py` 覆盖。合理推断（本库推断，未继续深挖它真实的实例化入口）：分层缓存的实现正在从独立的 `HiRadixCache` 向 `UnifiedRadixCache.init_hicache`（`python/sglang/srt/mem_cache/registry.py:187`-`193`）收敛，`HiRadixCache` 处在"仍被测试、仍被引用，但不再是默认新增功能落脚点"的过渡状态。
3. **`TreeNode.children` 声明成 `defaultdict(TreeNode)`，但这个自动创建行为从未被真正用到。** 全文件搜索 `.children[` 只出现 4 处（`python/sglang/srt/mem_cache/radix_cache.py:687`、`718`、`760`、`784`），其中两处读取（687、760）都在 `in node.children.keys()` 判断为真之后才发生，两处写入（718、784）是显式赋值。也就是说，如果未来有人在别处写一行 `node.children[some_key]` 且忘了先判断是否存在，会静默创建一个 `key=None`、`value=None` 的"幽灵节点"挂在树上——`defaultdict` 在这里更像一个未被清理的历史遗留，而非有意为之的设计。
4. **`match_prefix` 不是纯读操作。** 前面 4.2 节已经指出：命中边界落在某个节点内部时，`match_prefix` 会调用 `_split_node` 改写树结构（`python/sglang/srt/mem_cache/radix_cache.py:690`-`693`），函数自己的 docstring 也承认了这一点（`python/sglang/srt/mem_cache/radix_cache.py:404`-`412`）。对只读过"radix tree 就是查前缀"这类概念性介绍的人来说，这是一个容易漏掉的实现细节。
5. **`page_size > 1` 时，"任意长度前缀命中"这个卖点会打折扣。** `RadixKey.match` 把命中长度向下取整到 `page_size` 的整数倍（`python/sglang/srt/mem_cache/radix_cache.py:213`-`215`），意味着即使两个请求实际在第 `N+1` 个 token 才分叉，只要 `N` 不是 `page_size` 的整数倍，记录下来的分裂点也会退到 `N` 之前最近的 page 边界——最多损失 `page_size - 1` 个 token 的复用长度。第 6 节强调的"SGLang 能命中任意长度前缀"这个优势，**只在 `page_size == 1` 时完全成立**；`page_size > 1`（多数生产部署的常见配置，见 `python/sglang/srt/server_args.py:921`-`924`）下，SGLang 和 vLLM 的命中粒度差异，本质上收窄成"page 内如何组织"的差异，而不是"有没有粒度限制"的差异。
6. **`extra_key`/`cache_salt` 隔离会主动放弃看起来可以复用的公共前缀，这是有意为之的正确性权衡，不是漏洞。** `RadixKey._check_compatible`（`python/sglang/srt/mem_cache/radix_cache.py:169`-`179`）在 `extra_key` 或 `cache_salt` 不一致时直接抛异常，`child_key`（`python/sglang/srt/mem_cache/radix_cache.py:217`-`229`）把 `extra_key`/`cache_salt` 编进哈希 key 本身（第 227-229 行），保证两个 token 完全相同、但一个带 LoRA 适配器 A、一个不带的请求，从树的第一层开始就落在不同的 `children` 条目里，永远不会共享任何节点——哪怕它们此后的 token 序列一模一样。这是用"多花内存/少复用"换"绝不会把两个语义不同的请求的 KV 混在一起"的正确性保证，第 9 节自测题第 1 条延续了这个点。
7. **KV 事件不是调试日志，是给外部观测者用的信号通道。** `insert`（`python/sglang/srt/mem_cache/radix_cache.py:789`）在新建叶子节点时调用 `self.kv_events.record_store(new_node)`，`evict`（`python/sglang/srt/mem_cache/radix_cache.py:618`）在驱逐节点时调用 `self.kv_events.record_remove(x)`，`reset`（`python/sglang/srt/mem_cache/radix_cache.py:375`）在清空整棵树时调用 `self.kv_events.record_all_cleared()`——这套事件由 `KVCacheEventRecorder`（`python/sglang/srt/mem_cache/events.py`，通过 `python/sglang/srt/mem_cache/radix_cache.py:48` 导入）统一收集，默认关闭（`enable_kv_cache_events` 参数，构造 `RadixCache` 时传入，`python/sglang/srt/mem_cache/radix_cache.py:313`-`315`），开启后可以被外部消费者订阅——这是为 P/D 分离场景下 Prefill 和 Decode 两个进程各自维护一棵树、需要互相同步彼此树结构变化留的口子，具体怎么消费这些事件留给 [[11-SGLang-PD分离与HiCache分层]] 核对，本篇只确认了事件在源码里确实被触发。

## 8. 可改进点

1. **把 `TreeNode.children` 从 `defaultdict(TreeNode)` 改成普通 `dict`，配合 `.get()`。** 现有代码所有访问路径都已经手动做了存在性判断（第 7 节第 3 条），`defaultdict` 除了掩盖潜在 bug 外没有提供任何被使用到的行为，去掉它不改变任何现有语义，纯粹是移除一个风险点。
2. **`_split_node` 里两次 `.clone()` 是否可以在高频分叉场景下摊薄。** 目前每次分裂都对 `value` 做两次张量拷贝（父节点一次、收缩后的子节点一次），在长上下文 + 高并发分叉（比如大量 agentic 树搜索请求同时命中同一批共享前缀）场景下，这可能成为 CPU 侧热点；能否用写时复制的 tensor view 替代实际拷贝，需要先有真实 profiling 数据支撑，本库没有 GPU 环境无法验证，只能提出方向。
3. **`RadixCacheCpp`（C++ 版本）与 Python 版 `RadixCache` 的功能对等性缺一份文档化的对照表。** `SGLANG_EXPERIMENTAL_CPP_RADIX_TREE` 默认关闭（`python/sglang/srt/mem_cache/registry.py:104`-`109`），这条 opt-in 路径存在但收益/维护成本、与 `page_size`/`radix_eviction_policy` 等旋钮的兼容范围没有在本篇查证范围内确认，值得单独一篇做逐行核对。
4. **`RadixCache` vs `UnifiedRadixCache` 的算法一致性目前只能靠人工读代码确认。** 二者独立维护同一套"匹配-分裂-驱逐"逻辑（`UnifiedRadixCache` 没有直接继承或复用 `radix_cache.py` 的 `TreeNode`/`_split_node`，只共享了 `RadixKey`），如果两边逻辑出现分歧（比如 `page_size` 对齐规则、`lock_ref` 传播规则），不会有任何编译期或类型系统信号提示——一份跨实现的等价性单元测试（相同插入序列在两个实现上产生相同的命中/驱逐结果）会是比人工读代码更可靠的保障，本库没有在 `_lab/` 里发现这样的测试脚手架。
5. **`_lab/struct_map.py` 目前只统计类和方法名，不统计"哪个类实际在默认路径上被实例化"。** 本篇第 7 节第 1、2 条的发现（`RadixCache` 不是默认路径、`HiRadixCache` 没有直接构造点）完全靠人工用 `grep` 在 `registry.py` 里追工厂函数才확认出来；如果 `_lab/struct_map.py` 能追加一遍"从 `create_tree_cache` 开始的静态可达性分析"（哪些类可以通过默认参数组合被构造出来），后续每一篇涉及"哪个实现是默认值"的文章都能省掉这一步人工核对，也能在源码演进后自动发现"某个类不再可达"这类漂移。

## 9. 自测题与延伸阅读

**自测题**（闭卷）：

1. 两个请求 token 序列前 5 个完全相同、第 6 个开始分叉，`_split_node` 被调用时 `split_len` 参数会是多少？分裂后原节点和新建的父节点各自的 `key`/`value` 长度分别是多少？
2. `evictable_leaves` 为什么要增量维护而不是每次 `evict` 调用时现算？如果改成现算，复杂度会从什么变成什么？
3. 为什么 `inc_lock_ref`/`dec_lock_ref` 要沿整条路径操作而不是只操作命中的最后一个节点？举一个"只锁叶子"会导致父节点被误驱逐的具体场景。
4. `page_size > 1` 时，如果两个请求在 token 位置 5 分叉、`page_size = 4`，实际记录的分裂点会落在 token 位置几？为什么？
5. vLLM 的块哈希表在"遇 miss 即停"这一步依赖什么前提？如果允许哈希表存储"未满的块"，这个"遇 miss 即停"的正确性还成立吗？
6. `match_prefix` 会不会在没有任何新 token 被缓存的情况下改变树的节点数量？如果会，是通过哪个函数、在什么条件下发生的？
7. 为什么本篇选择精读 863 行的 `RadixCache` 而不是 2,894 行的默认生产实现 `UnifiedRadixCache`？这个选择本身依赖的证据链是什么（提示：见第 7 节第 1 条）？
8. 两个请求 token 完全相同，但一个带 LoRA 适配器、一个不带，它们会共享任何 `TreeNode` 吗？决定这件事的是哪个函数、哪个字段？
9. `_split_node` 里 `new_node.lock_ref = child.lock_ref`（而不是让新节点从 0 开始）这一行为什么是必要的？如果去掉它，一个正在被运行中请求占用的前缀在分裂后会面临什么风险？

**延伸阅读**（本库双链）：

- [[02-SGLang-Scheduler事件循环]]——本篇止步于"树内部怎么匹配/分裂/驱逐"，`match_prefix`/`cache_unfinished_req` 在调度循环的哪一步被调用、和批处理决策如何交织，在这一篇展开。
- [[04-vLLM-KV缓存与前缀缓存]]——第 6 节"块哈希表 vs 基数树"对照的另一半，该篇 §6 早先把 SGLang 一侧标注为"本库推断"占位，本篇提供了可核对的源码引用，可以回去把那个占位替换掉。
- [[04-SGLang-内存池与KV布局]]——本篇把 `TreeNode.value` 当作"KV 槽位下标"一笔带过，这些下标具体如何映射到物理显存布局（MHA/MLA、page-major 布局等），在这一篇继续。
