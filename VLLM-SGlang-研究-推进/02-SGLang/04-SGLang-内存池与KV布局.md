# SGLang 内存池分层与 KV 张量布局解剖

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：请求管 token 位置的账，token 位置管 KV 数据的账，两本账分开算才不用给每个请求预留满血显存。

## 0. 结论先行

SGLang 的显存管理压缩成一句话：**两级映射表 + 一族按注意力类型分化的物理 KV 池**。

1. **`ReqToTokenPool` 和 `TokenToKVPool`（Allocator + KVCache）是两本完全独立计价的账**。前者把"请求"映射到"token 逻辑位置"，大小只跟并发请求数与 `max_context_len` 有关；后者把"token 位置"映射到"KV 显存里的物理槽位"，大小由可用显存除以每 token 字节数决定——两者的容量公式互不相关，这正是源码模块 docstring 里"two levels of memory pool"这句话的字面含义（`python/sglang/srt/mem_cache/memory_pool.py:17`）。
2. **KV 张量默认是"按层拆开的一组张量"，不是一整块 5D 大张量**。`MHATokenToKVPool` 的 `k_buffer`/`v_buffer` 各自是长度为 `layer_num` 的 Python list，每个元素形状 `(size+page_size, head_num, head_dim)`（NHD 布局，`python/sglang/srt/mem_cache/memory_pool.py:2056`-`2060`）——这是默认摆法，但同一份代码里还有把所有层揉进一整块连续 `uint8` buffer 的 `PageMajorMHATokenToKVPool`（`python/sglang/srt/mem_cache/memory_pool.py:3139`），两种摆法共存，摆法本身是可切换的实现细节而非唯一真理。
3. **MLA 池只有一份 `kv_buffer`，K 和 V 是同一块内存的两种切片视图**，不是两块独立显存。`MLATokenToKVPool` 存的是压缩后的 `kv_lora_rank + qk_rope_head_dim` 维向量（`python/sglang/srt/mem_cache/memory_pool.py:4046`-`4050`），`get_value_buffer` 直接对同一个张量做 `[..., :kv_lora_rank]` 切片（`python/sglang/srt/mem_cache/memory_pool.py:4109`-`4117`）——这不是"节省一半"，是"K 和 V 根本没有分开存过"。
4. **`page_size` 默认是 1（一页一个 token），不是 vLLM 式的"一页一批 token"**。`_page_size_default` 在绝大多数平台上直接返回 `{"page_size": 1}`（`python/sglang/srt/arg_groups/overrides.py:2396`），而 vLLM 的 `block_size` 默认是 16（`vllm:vllm/config/cache.py:79`）。更关键的是：page_size 在很多场景下**不是用户能随便调的旋钮**，而是被具体 attention 后端的内核约束反过来钉死的——FlashMLA 后端会把它强制改写成 64（`python/sglang/srt/arg_groups/overrides.py:2136`-`2142`），DSA 池在非 HIP 平台上直接 `assert self.page_size == 64`（`python/sglang/srt/mem_cache/memory_pool.py:4497`）。
5. **显存预算链路不做一次真实前向 profile，靠一条静态启发式公式估出"非 KV 内存"**，这与 vLLM 用真实 `profile_run()` 量出激活峰值的做法（`vllm:vllm/v1/worker/gpu_worker.py:481`）是两条完全不同的工程路线（详见 §6）。`mem_fraction_static` 默认值本身就是从 `reserved_mem = 512 + activation_tokens*1.5 + ...` 这类启发式公式反推出来的（`python/sglang/srt/server_args.py:5027`-`5038`），源码注释直接承认"未来可以做更好的估计，甚至跑一次 dummy run"（`python/sglang/srt/server_args.py:4880`）。

一条本库需要先澄清的取证结果：任务提纲里提到的"双稀疏（DoubleSparse）池"在当前 sha 下**查无此类**——全仓 `grep -r "DoubleSparse\|double_sparse"` 零命中，推测是曾经存在的实验特性已被移除或被更通用的稀疏池取代。当前代码库里承担"稀疏注意力 KV 池"这个角色的是 `DSATokenToKVPool`（DeepSeek Sparse Attention，`python/sglang/srt/mem_cache/memory_pool.py:4431`）和 `MiniMaxSparseKVPool`（`python/sglang/srt/mem_cache/memory_pool.py:4762`），本篇以两者为准，对不存在的类不编造内容。

## 1. 它在系统里的位置

内存池子系统夹在调度器和注意力后端之间，是"逻辑 token 位置"和"物理显存槽位"之间的两级间接层：

```
Scheduler / ScheduleBatch
   │  每步：req_to_token_pool.alloc(reqs) → token_to_kv_pool_allocator.alloc(need_size) → req_to_token_pool.write(...)
   ▼
ReqToTokenPool                                    (python/sglang/srt/mem_cache/memory_pool.py:256)
   │  req_pool_idx → req_to_token[req_pool_idx, :seq_len] = 一串 KV 物理槽位下标（稠密，逐 token）
   ▼
BaseTokenToKVPoolAllocator（Token* / Paged* / SWA* / Mamba*）  (python/sglang/srt/mem_cache/allocator/base.py:27)
   │  只做"哪些整数下标是空闲的"这本账，不碰显存；page_size=1 时下标就是 token 下标本身，
   │  page_size>1 时下标是"页号 × page_size + 页内偏移"（见 §3.2）
   ▼
KVCache（MHATokenToKVPool / MLATokenToKVPool / DSATokenToKVPool / ...）  (python/sglang/srt/mem_cache/memory_pool.py:1628)
   │  唯一真正持有 GPU 张量的对象：按下标 gather/scatter 出 K/V（或 MLA 的单一 latent 向量）
   ▼
（Attention 后端）RadixAttention.forward 读 get_kv_buffer(layer_id) 拼进 kernel 调用
```

模块自己的 docstring 把这条链路总结成三句话（`python/sglang/srt/mem_cache/memory_pool.py:15`-`21`）：

> SGLang has two levels of memory pool.
> ReqToTokenPool maps a request to its token locations.
> TokenToKVPoolAllocator manages the indices to kv cache data.
> KVCache actually holds the physical kv cache.

注意这里的"两级"指的是 **req→token位置** 和 **token位置→KV数据** 两次映射，而"token位置→KV数据"这一级在实现上又拆成了两个协作对象：`*TokenToKVPoolAllocator`（只管哪些下标空闲/占用，不碰显存）和 `KVCache`（只管下标对应的显存里到底存了什么，不管下标怎么分配）。RadixAttention 前缀缓存树（见 [[03-SGLang-RadixAttention与前缀缓存]]）复用的正是 `ReqToTokenPool` 与 allocator 之间的这套下标账本；本篇只讲账本本身和它背后的物理显存长什么样，树结构如何驱动分配不在本篇范围。

## 2. 代码地图（文件 → 职责，带行号）

| 文件 | 职责 | 关键行 |
|---|---|---|
| `python/sglang/srt/mem_cache/memory_pool.py` | 头号文件，5125 行：`ReqToTokenPool`、抽象基类 `KVCache`、`MHATokenToKVPool`/`MLATokenToKVPool`/`DSATokenToKVPool` 等全部物理池实现 | `ReqToTokenPool` `python/sglang/srt/mem_cache/memory_pool.py:256`；`KVCache` `python/sglang/srt/mem_cache/memory_pool.py:1628`；`MHATokenToKVPool` `python/sglang/srt/mem_cache/memory_pool.py:1759`；`PageMajorMHATokenToKVPool` `python/sglang/srt/mem_cache/memory_pool.py:3139`；`MLATokenToKVPool` `python/sglang/srt/mem_cache/memory_pool.py:4009`；`DSATokenToKVPool` `python/sglang/srt/mem_cache/memory_pool.py:4431`；`MiniMaxSparseKVPool` `python/sglang/srt/mem_cache/memory_pool.py:4762` |
| `python/sglang/srt/mem_cache/allocator/base.py` | 分配器抽象基类：只管整数下标的空闲/占用状态 | `BaseTokenToKVPoolAllocator` `python/sglang/srt/mem_cache/allocator/base.py:27`；`resize`（随 `max_total_num_tokens` 重建）`python/sglang/srt/mem_cache/allocator/base.py:109` |
| `python/sglang/srt/mem_cache/allocator/token.py` | `page_size=1` 时的朴素分配器 | `TokenToKVPoolAllocator` `python/sglang/srt/mem_cache/allocator/token.py:28`；把 `page_size` 硬编码成 1 传给基类 `python/sglang/srt/mem_cache/allocator/token.py:36` |
| `python/sglang/srt/mem_cache/allocator/paged.py` | `page_size>1` 时的页对齐分配器 | `PagedTokenToKVPoolAllocator` `python/sglang/srt/mem_cache/allocator/paged.py:105`；`alloc`（页号→token 下标展开）`python/sglang/srt/mem_cache/allocator/paged.py:149`-`170` |
| `python/sglang/srt/mem_cache/index_key_cache.py` | DSA 索引器的独立 K 缓存（稀疏 top-k 选择用，不是主 KV） | `IndexKeyCache` `python/sglang/srt/mem_cache/index_key_cache.py:14`；缓冲区形状 `python/sglang/srt/mem_cache/index_key_cache.py:32`-`38` |
| `python/sglang/srt/mem_cache/allocation.py` | 把分配到的物理下标写回 `req_to_token` 表的粘合代码 | `write_req_to_token_pool`（triton/CPU 双路径）`python/sglang/srt/mem_cache/allocation.py:95`-`99` |
| `python/sglang/srt/mem_cache/memory_pool_host.py` | HiCache 分层缓存的宿主（CPU 侧）内存池，本篇只标接口，细节见 [[11-SGLang-PD分离与HiCache分层]] | `LogicalHostPool` `python/sglang/srt/mem_cache/memory_pool_host.py:56` |
| `python/sglang/srt/mem_cache/pool_host/base.py` | 宿主池抽象基类，显式持有 `device_pool: KVCache` 引用 | `HostKVCache.__init__` `python/sglang/srt/mem_cache/pool_host/base.py:113`-`150` |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py` | 2252 行：显存预算 profiling → 池子构造的编排类 | `KVCacheConfigurator.configure` `python/sglang/srt/mem_cache/kv_cache_configurator.py:276`；`_profile_available_bytes` `python/sglang/srt/mem_cache/kv_cache_configurator.py:1826`；`_resolve_memory_pool_config` `python/sglang/srt/mem_cache/kv_cache_configurator.py:1990`；`resolve_max_num_reqs` `python/sglang/srt/mem_cache/kv_cache_configurator.py:1935`；`calculate_mla_kv_cache_dim` `python/sglang/srt/mem_cache/kv_cache_configurator.py:2201` |
| `python/sglang/srt/model_executor/pool_configurator.py` | "每 token 多少字节"（cell_size）的具体算法，按 MHA/MLA/MiniMax 稀疏分支 | `DefaultPoolConfigurator._compute_cell_size` `python/sglang/srt/model_executor/pool_configurator.py:236`；`calculate_pool_sizes`（budget÷cell_size）`python/sglang/srt/model_executor/pool_configurator.py:416`-`425` |
| `python/sglang/srt/model_executor/model_runner.py` | 调用编排类、把结果挂到 `self.req_to_token_pool` 等属性上 | `alloc_memory_pool` `python/sglang/srt/model_executor/model_runner.py:822` |
| `python/sglang/srt/distributed/bootstrap.py` | 在加载权重**之前**量一次"pre_model_load_memory"，作为预算公式的锚点 | `python/sglang/srt/distributed/bootstrap.py:132`-`137` |
| `python/sglang/srt/utils/common.py` | 跨平台"当前空闲显存"查询，返回 GiB | `get_available_gpu_memory` `python/sglang/srt/utils/common.py:414`；单位换算 `python/sglang/srt/utils/common.py:545` |
| `python/sglang/srt/server_args.py` | `page_size`/`mem_fraction_static` 等旋钮定义与默认值兜底公式 | `page_size` 定义 `python/sglang/srt/server_args.py:921`-`925`；`mem_fraction_static` 定义 `python/sglang/srt/server_args.py:799`-`802`；默认值启发式公式 `python/sglang/srt/server_args.py:5027`-`5038` |
| `python/sglang/srt/arg_groups/overrides.py` | `page_size` 的两处后处理：平台默认值、attention 后端约束 | `_page_size_default` `python/sglang/srt/arg_groups/overrides.py:2378`-`2397`；`_mla_backend_page_constraints` `python/sglang/srt/arg_groups/overrides.py:2129`-`2142` |

（此表 15 条 `文件:行` 引用，已超过 `## 2` 硬指标 10 条；正文其余小节还会补充更细的行号。）

## 3. 核心数据结构

### 3.1 `ReqToTokenPool`（`python/sglang/srt/mem_cache/memory_pool.py:256`）

```python
class ReqToTokenPool:
    def __init__(self, size, max_context_len, device, enable_memory_saver):
        self.size = size
        self._alloc_size = size + 1
        self.max_context_len = max_context_len
        self.req_to_token = torch.zeros(
            (self._alloc_size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation = torch.zeros(self._alloc_size, dtype=torch.int64)
```

逐字段说明（行号见 `python/sglang/srt/mem_cache/memory_pool.py:272`-`283`）：

- **`req_to_token: torch.Tensor`**，形状 `(size+1, max_context_len)`，`int32`。这是**稠密的、逐 token 的**下标表：`req_to_token[req_idx, pos]` 直接存着第 `pos` 个 token 在 KV 物理池里的槽位号，不是"块号"，attention 后端读它时不需要再乘任何 `page_size`（对照 §6 vLLM 的 block_table）。
- **`_alloc_size = size + 1`**：多出来的第 0 行是给 CUDA Graph 补齐 batch 时 `req_pool_idx` 默认填 0 用的占位行——填 0 的哑请求读写这一行不会踩到任何真实请求的数据（`python/sglang/srt/mem_cache/memory_pool.py:273`-`274` 注释原话）。
- **`free_slots: list[int]`**：空闲的 `req_pool_idx` 列表，从队尾弹出分配（`self.free_slots[-need_size:]`，`python/sglang/srt/mem_cache/memory_pool.py:312`），注释直接写明这是为了 O(need_size) 而不是从队首弹出的 O(len(free_slots))（`python/sglang/srt/mem_cache/memory_pool.py:310`-`311`）。
- **`req_generation: torch.Tensor`**，`int64`，每次一个 `req_pool_idx` 被重新分配就 `+= 1`（`python/sglang/srt/mem_cache/memory_pool.py:321`）。这是一个 ABA 问题的显式解法：**本库推断**（依据 `python/sglang/srt/managers/overlap_utils.py:210`-`230` 的用法）——在 overlap 调度下，某个异步产出的结果（比如投机解码的置信度）可能在写回时，它原本对应的 `req_pool_idx` 已经被释放并分配给了另一个新请求；`req_generation` 让消费者能判断"这份异步结果还是不是当初那个请求产生的"，不必靠时序假设。
- **`available_size()` / `alloc(reqs)` / `free(req)` / `clear()`**：`alloc` 支持"部分请求复用已有 `req_pool_idx`"（chunked prefill 跨 chunk 续用同一个槽位，`python/sglang/srt/mem_cache/memory_pool.py:291`-`323`），`free` 只是把下标塞回 `free_slots`（`python/sglang/srt/mem_cache/memory_pool.py:325`-`328`），两者都不碰任何 KV 显存——这本账里根本没有 KV 数据的影子。

### 3.2 `BaseTokenToKVPoolAllocator` 家族（`python/sglang/srt/mem_cache/allocator/base.py:27`）

抽象基类字段（`python/sglang/srt/mem_cache/allocator/base.py:38`-`48`）：

- **`size: int`**：这个分配器能分配的 token 槽位总数上限（等于 `max_total_num_tokens`）。
- **`page_size: int`**：一次分配的最小粒度，见 §3.3。
- **`_kvcache: KVCache`**：持有对物理池对象的引用，但只用来转发 `get_cpu_copy`/`load_cpu_copy` 这类需要知道物理布局的操作，分配逻辑本身完全不读它。
- **`free_pages` / `release_pages`**：两个整数张量。`free_pages` 是"随时可分配"的下标（或页号），`release_pages` 是"本轮刚被 free 但还没排序合并"的下标——`merge_and_sort_free()`（`python/sglang/srt/mem_cache/allocator/base.py:77`-`83`）才把两者合并排序，这是一个"批量释放先攒着、需要时再排序"的惰性优化，避免每次 `free()` 都触发一次 `torch.sort`。
- **`free_group` / `is_not_in_free_group`**：`free_group_begin()`/`free_group_end()`（`python/sglang/srt/mem_cache/allocator/base.py:63`-`70`）让调用方把一批 `free()` 调用先收集起来最后合并成一次 `torch.cat`，用于批量释放场景（比如一次调度步里同时结束多个请求）。

两个具体子类：

- **`TokenToKVPoolAllocator`**（`python/sglang/srt/mem_cache/allocator/token.py:28`）：把 `page_size` 硬编码为 1 传给基类（`python/sglang/srt/mem_cache/allocator/token.py:36`）。`free_pages` 就是"空闲 token 槽位号"本身，`alloc(need_size)` 直接从 `free_pages` 头部切走 `need_size` 个（`python/sglang/srt/mem_cache/allocator/token.py:53`-`61`）——**分配粒度=物理槽位粒度=一个 token**，没有"页"这个中间层。
- **`PagedTokenToKVPoolAllocator`**（`python/sglang/srt/mem_cache/allocator/paged.py:105`）：`free_pages` 里存的是**页号**，`alloc(need_size)` 先换算成 `num_pages = need_size // page_size`（`python/sglang/srt/mem_cache/allocator/paged.py:156`），再把页号展开成 token 下标：`out_indices = out_pages[:, None] * page_size + arange(page_size)`（`python/sglang/srt/mem_cache/allocator/paged.py:165`-`168`）——**返回给上层（最终写进 `req_to_token`）的仍然是逐 token 的物理槽位号**，"页"只是分配器内部记账的粒度，不会泄漏到 `ReqToTokenPool` 那一层。

### 3.3 `KVCache` 抽象基类（`python/sglang/srt/mem_cache/memory_pool.py:1628`）

公共字段（`__init__`，`python/sglang/srt/mem_cache/memory_pool.py:1633`-`1667`）：

- **`size: int`**：本池能容纳的 token 槽位数（`max_total_num_tokens`，不含 padding 行）。
- **`page_size: int`**：物理布局的页粒度——**不是**分配粒度的另一份拷贝，而是真正决定张量形状的那个数（见 §3.4）。等于 1 时代表"一页一个 token 槽位"。
- **`dtype` / `store_dtype`**：`store_dtype` 在 KV dtype 是 fp8 系列时强制退化成 `torch.uint8`（`python/sglang/srt/mem_cache/memory_pool.py:1649`-`1653`），注释写明原因：`Tensor.index_put` 当时不支持 `float8_e5m2` 直接索引赋值，只能先当 `uint8` 存、读的时候再 `.view(dtype)` 转回去。
- **`layer_num` / `start_layer` / `end_layer`**：支持流水线并行下"这个 rank 只持有 `[start_layer, end_layer)` 区间的层"，池子对象本身不知道全局层数。
- **`mem_usage`**：仅用于日志，由 `_finalize_allocation_log`（`python/sglang/srt/mem_cache/memory_pool.py:1674`-`1700`）在分配完成后填入实际字节数。

抽象方法契约：`get_key_buffer(layer_id)` / `get_value_buffer(layer_id)` / `get_kv_buffer(layer_id)` / `set_kv_buffer(layer, loc, k, v)`（`python/sglang/srt/mem_cache/memory_pool.py:1706`-`1726`）——**所有子类必须能按 `layer_id` 单独取出这一层的 K/V**，这是 attention 后端逐层调用的接口，也是"为什么物理布局要按层拆开"的直接原因之一（见 §5.2）。

### 3.4 `MHATokenToKVPool`（`python/sglang/srt/mem_cache/memory_pool.py:1759`）

默认（NHD）布局下的物理张量形状（`_kv_buffer_shapes`，`python/sglang/srt/mem_cache/memory_pool.py:2049`-`2060`）：

```python
def _kv_buffer_shapes(self):
    if self.use_hnd:
        return (
            (self.num_pages, self.head_num, self.page_size, self.head_dim),
            (self.num_pages, self.head_num, self.page_size, self.v_head_dim),
        )
    rows = self.size + self.page_size
    return (
        (rows, self.head_num, self.head_dim),
        (rows, self.head_num, self.v_head_dim),
    )
```

- **`k_buffer` / `v_buffer`：`list[torch.Tensor]`，长度 `layer_num`**（构造见 `_create_buffers_normal`，`python/sglang/srt/mem_cache/memory_pool.py:2062`-`2113`）。默认 NHD 布局每层一个形状 `(size+page_size, head_num, head_dim)` 的张量——**行下标就是 token 物理槽位号**，`get_key_buffer(layer_id)` 直接 `self.k_buffer[layer_id - self.start_layer]`（`python/sglang/srt/mem_cache/memory_pool.py:2276`-`2298`），O(1) 定位到这一层，不用在一整块大张量里切片。
- **`+page_size` 而不是 `+1`**：第一行到第 `page_size` 行是"padding 页"，专门吸收 CUDA Graph 补齐 batch 时的哑写入（`python/sglang/srt/mem_cache/memory_pool.py:2069` 注释），page_size 越大这块 padding 也越大。
- **`use_hnd`（HND 布局，`SGLANG_USE_HND_KVCACHE` 开）**：形状变成 `(num_pages, head_num, page_size, head_dim)`——**页维度被拆到最外层，token 在页内的位置退到第三维**，专门服务"按 KV 头做稀疏页表"的分页后端（`python/sglang/srt/mem_cache/memory_pool.py:1810`-`1815` 注释）。
- **`get_v_head_dim()` / `v_head_dim`**：MHA 的 V 头维可以和 K 头维不同（非对称头维模型），池子显式区分 `head_dim`（K）和 `v_head_dim`（V），不假设两者相等（`python/sglang/srt/mem_cache/memory_pool.py:1804`-`1808`）。
- **`PageMajorMHATokenToKVPool`**（`python/sglang/srt/mem_cache/memory_pool.py:3139`）是同一接口下的另一种物理摆法：所有层共享一个连续 `uint8` `_raw` buffer，每层的 K/V 是这块大 buffer 上的 4D strided view（"page-major, layer-major within a page"），docstring 明确写了代价——`tiled KV copy` 内核、CPU offload、投机解码前缀提交内核都假设"每层是独立连续 3D 张量"，在这个 strided-view 布局下会**直接报错而不是静默错误索引**（`python/sglang/srt/mem_cache/memory_pool.py:3148`-`3151`）。这说明"按层拆张量列表"不是唯一实现，只是默认实现和目前生态覆盖最全的实现。

### 3.5 `MLATokenToKVPool`（`python/sglang/srt/mem_cache/memory_pool.py:4009`）

```python
self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim   # memory_pool.py:4046-4050
self.kv_buffer = [
    torch.zeros((self.size + self.page_size, 1, self.kv_cache_dim),
                dtype=self.store_dtype, device=self.device)
    for _ in range(self.layer_num)
]                                                       # memory_pool.py:4071-4078
```

- **只有一个 `kv_buffer`，没有 `k_buffer`/`v_buffer` 之分**。`get_key_buffer` 直接返回整个 `kv_buffer[layer]`（`python/sglang/srt/mem_cache/memory_pool.py:4100`-`4107`），`get_value_buffer` 对同一个张量切片 `[..., :kv_lora_rank]`（`python/sglang/srt/mem_cache/memory_pool.py:4109`-`4117`）——**"K"和"V"是同一段内存的两种读法**：MLA 数学上的 KV cache 本来就是一个压缩后的单一 latent 向量（`kv_lora_rank` 维）拼上 RoPE 位置编码（`qk_rope_head_dim` 维），不存在独立的 per-head K 向量和 V 向量,自然也没必要分成两块显存。
- **中间维恒为 1（`(rows, 1, kv_cache_dim)`）**：MLA 不像 MHA 那样每个注意力头存一份 KV，这一维是"每 token 只有一份 latent"的直接体现——对照 MHA 的 `(rows, head_num, head_dim)`，MLA 少掉了 `head_num` 这个乘数，这是 MLA 显存优势的物理来源，不是量化或压缩算法带来的，是数据结构本身就没有那个维度。
- **`get_kv_size_bytes`**（`python/sglang/srt/mem_cache/memory_pool.py:4083`-`4088`）只累加一份 `kv_buffer`，不像 MHA 池要分别累加 K 和 V。

### 3.6 `DSATokenToKVPool`（`python/sglang/srt/mem_cache/memory_pool.py:4431`，继承 `MLATokenToKVPool`）

DeepSeek Sparse Attention（DSA，DeepSeek-V3.2 系）在 MLA 的单一 `kv_buffer` 之外，**额外挂了一个独立的索引器 K 缓存**：

```python
self.index_key_cache = self._create_index_key_cache()   # memory_pool.py:4498
```

- **`IndexKeyCache`**（`python/sglang/srt/mem_cache/index_key_cache.py:14`）自己按层建一组独立 buffer，形状 `(num_pages, page_size * (index_head_dim + 量化 scale 字节))`（`python/sglang/srt/mem_cache/index_key_cache.py:32`-`38`），`dtype=torch.uint8`——这份缓存服务的是 DSA 的稀疏 top-k 选择机制（先用低维索引向量算相关性再选真正参与 attention 的 KV 子集），和主 `kv_buffer` 是两块完全独立的物理内存，靠 `move_kv_cache` 里显式的"lockstep 搬运"保持下标对齐（`python/sglang/srt/mem_cache/memory_pool.py:4514`-`4517` 附近的 `move_kv_cache` override）。
- **`page_size` 被写死**：非 HIP 平台 `assert self.page_size == 64`（`python/sglang/srt/mem_cache/memory_pool.py:4497`），HIP 平台要么等于 1、要么是 16 的倍数（`python/sglang/srt/mem_cache/memory_pool.py:4487`-`4495`）——DSA 池不接受默认的 `page_size=1`，这是内核实现（分块 top-k 索引计算）对物理布局的硬约束，见 §5.4。
- 另一个"组合式稀疏池"的例子是 `MiniMaxSparseKVPool`（`python/sglang/srt/mem_cache/memory_pool.py:4762`），它不继承 MLA/DSA，而是**组合**了一个主 MHA 池（稠密层）+ 一个索引 KV 池（稀疏层，K+V 都要读）+ 一个只存 K 的池（稀疏层里 V 从不被读的那部分，`python/sglang/srt/mem_cache/memory_pool.py:4779`-`4809`）——这印证了一个通用模式：**SGLang 的"稀疏注意力 KV 池"普遍不是发明新的张量形状，而是把已有的 MHA/MLA/K-only 池按层的角色重新组合**。

### 3.7 其余变体一览（不逐一展开，只标坐标）

`KVCache` 这一层抽象下还有一批更窄场景的子类，本篇不逐行拆，但值得知道它们存在、解决什么问题，方便后续按需查表：

- **`NoOpMHATokenToKVPool`**（`python/sglang/srt/mem_cache/memory_pool.py:2871`）：embedding-only、prefill-only 且用 `fa_skip_kv_cache` 路径时，**根本不分配真实 K/V 显存**，只留 `(page_size, head_num, head_dim)` 的占位张量满足接口——docstring 明确写了"调用方必须保证不会对这个池真正读写，否则显式报错"（`python/sglang/srt/mem_cache/memory_pool.py:2884`-`2886`），这是"调度器眼里池子容量正常、物理上根本没分配"的一种合法状态。
- **`MHATokenToKVPoolFP4`**（`python/sglang/srt/mem_cache/memory_pool.py:2985`）/ **`MHATokenToKVPoolMXFP8`**（`python/sglang/srt/mem_cache/memory_pool.py:3293`）：同一套 NHD 形状之上叠加 FP4/MXFP8 量化，需要额外的 per-block scale 缓冲区——这也是 §4.3 里 `cell_size` 公式要为 `is_float4_e2m1fn_x2`/`mxfp8` 单独加一段 scale 开销的原因（`python/sglang/srt/model_executor/pool_configurator.py:336`-`349`）。
- **`HybridLinearKVPool`**（`python/sglang/srt/mem_cache/memory_pool.py:3654`）：docstring 一句话"KV cache with separate pools for full and linear attention layers"——给 Mamba/线性注意力混合模型用，全注意力层走标准 MHA/MLA 子池，线性注意力层的状态走独立的 `mamba_pool`，两者物理上互不相关，因为线性注意力状态根本不是"per-token 的 KV"，形状语义完全不同。
- **`MHATokenToKOnlyPool`**（`python/sglang/srt/mem_cache/memory_pool.py:4663`）：docstring 直接说明用途——MiniMax 稀疏层里"索引分支从不读 V"的那部分层，分配 V 缓冲区纯粹是浪费，所以干脆只建 K 池（`python/sglang/srt/mem_cache/memory_pool.py:4663`-`4665`）。
- **`SWATokenToKVPoolAllocator`**（`python/sglang/srt/mem_cache/allocator/swa.py:20`）：滑动窗口混合模型的分配器，内部按 `page_size==1` 与否分别持有一个 `full_attn_allocator`（`python/sglang/srt/mem_cache/allocator/swa.py:42`-`44`）——这与 [[04-vLLM-KV缓存与前缀缓存]] §5.5 讲的"全注意力组与滑窗组共用同一个 `block_id` 空间"是两条不同的路线：vLLM 强制全局唯一下标空间，SGLang 这里是"两个子分配器分别管自己的下标，靠 `full_kv_pool`/`swa_kv_pool` 两个物理池对应"，代价与收益的对照留给横向对比篇。

这些变体共享同一套 `KVCache` 抽象接口（§3.3），意味着 attention 后端和调度器的调用代码不需要为每种变体写分支——**"一族形状不同的池子实现同一份契约"是这套抽象存在的全部意义**。

## 4. 主流程走读

### 4.1 启动期：从 `mem_fraction_static` 到两个池子的实例

```
distributed/bootstrap.py:132  pre_model_load_memory = get_available_gpu_memory(...)   # 加载权重之前的空闲显存快照
    │
    ▼  （权重被加载，占用一部分显存）
model_runner.py:822  ModelRunner.alloc_memory_pool()
    │
    ▼
kv_cache_configurator.py:276  KVCacheConfigurator.configure(pre_model_load_memory=...)
    │
    ▼
kv_cache_configurator.py:1990  _resolve_memory_pool_config(pre_model_load_memory)
    │
    ├─ kv_cache_configurator.py:1826  _profile_available_bytes(pre_model_load_memory)
    │      available_gpu_memory = get_available_gpu_memory(...)          # 此刻的空闲显存（GiB）
    │      slack_gb = pre_model_load_memory * (1 - mem_fraction_static)  # :1837，非静态开销的预留
    │      rest_memory = available_gpu_memory - slack_gb - mm_reservation_gb   # :1851
    │      return int(rest_memory * (1 << 30))                          # :1873，换算成字节
    │
    ├─ pool_configurator.py:236  DefaultPoolConfigurator._compute_cell_size(...)
    │      MHA:  cell_size = n_kv_heads * (head_dim + v_head_dim) * num_layers * dtype_size   # :328-334
    │      MLA:  cell_size = (kv_lora_rank + qk_rope_head_dim) * num_layers * dtype_size       # :258-266
    │            （DSA 再加一段索引器开销，:280-284）
    │
    ├─ pool_configurator.py:419  max_total_num_tokens = available_bytes // cell_size，再向下取整到 page_size 的倍数（:424）
    │
    └─ kv_cache_configurator.py:1935  resolve_max_num_reqs(max_total_num_tokens)
           estimated = max_total_num_tokens / context_len * 512，夹在 [2048, 4096] 之间   # :1939-1940
           max_num_reqs = min(estimated 或用户指定值, max_total_num_tokens // 2)          # :1945/1948
    │
    ▼
kv_cache_configurator.py:652  req_to_token_pool = ReqToTokenPool(size=max_num_reqs, max_context_len=context_len+extra, ...)
kv_cache_configurator.py:_init_pools  token_to_kv_pool_allocator + token_to_kv_pool（MHA/MLA/DSA 二选一）
```

三处容易读漏的关键点：

1. **`pre_model_load_memory` 是"加载权重前"的快照，`available_gpu_memory` 是"配置这一刻"的实时读数**——两者不是同一个量。`slack_gb` 用的是前者乘以 `(1 - mem_fraction_static)`，也就是说**无论权重实际占了多少显存，非静态开销的预留量只跟"加载前的显存总量"和 `mem_fraction_static` 有关，跟权重实际吃掉多少无关**（`python/sglang/srt/mem_cache/kv_cache_configurator.py:1827`-`1829` 注释原话："Whatever is already resident (model weights, etc.) is thus charged against it"）。
2. **`cell_size`（每 token 字节数）是纯静态公式算出来的，不依赖任何一次真实前向**——`n_kv_heads * (head_dim+v_head_dim) * num_layers * dtype_size` 全部是模型配置里能直接读到的整数，MLA 分支同理（`kv_lora_rank + qk_rope_head_dim`，见 `python/sglang/srt/mem_cache/kv_cache_configurator.py:2201`-`2215` 的 `calculate_mla_kv_cache_dim`）。这部分公式是精确的（KV cache 大小本来就能解析算出来），**静态估算不精确的地方在"非 KV 内存"那一段**（见 §6）。
3. **`ReqToTokenPool` 的大小是从已经算好的 `max_total_num_tokens` 反推出来的，不是独立预算的**——`resolve_max_num_reqs` 的入参就是 `token_capacity`（`python/sglang/srt/mem_cache/kv_cache_configurator.py:1935`），这意味着 `ReqToTokenPool` 永远是"后决定"的那一个，它的容量被设计成"不会成为瓶颈"（`min(estimated, token_capacity // 2)`），真正的显存瓶颈始终在 KV 池这一侧。

### 4.2 运行期：一次 extend/decode 的分配 → 写回 → 读取

```
调度器决定这一步要 extend 的 token 数
    │
    ▼
token_to_kv_pool_allocator.alloc(need_size)          # allocator/token.py 或 allocator/paged.py
    → 返回一串物理 KV 槽位下标 out_cache_loc
    │
    ▼
mem_cache/allocation.py:95  req_to_token_pool.write((req_idx, slice(prefix_len, seq_len)), out_cache_loc)
    → req_to_token[req_idx, prefix_len:seq_len] = out_cache_loc   （稠密逐 token 写入）
    │
    ▼
模型前向：RadixAttention 层调 token_to_kv_pool.set_kv_buffer(layer, loc, cache_k, cache_v)
    → k_buffer[layer_id][loc] = cache_k；v_buffer[layer_id][loc] = cache_v（MHA）
    → kv_buffer[layer_id][loc] = cache_k（MLA，V 是同一份数据的切片，不需要单独写）
    │
    ▼
下一步：attention 后端读 req_to_token[req_idx, :seq_len] 拿到这个请求全部 token 的物理槽位号，
    据此 gather 出 K/V 传给 kernel（page_size>1 时 kernel 自己按页组织访存，见 §5.4）
```

## 5. 设计决策与代价

### 5.1 两级间接（req → token 位置 → KV 物理槽位），不直接 req → KV

**为什么这么设计**：`ReqToTokenPool` 按 `max_num_reqs`（并发请求数上限）计价，`TokenToKVPool` 按 `max_total_num_tokens`（GPU 显存能装下的 token 总数）计价，这是两个数量级完全不同、也没有代数关系的量——`max_num_reqs` 一般在几千以内（`resolve_max_num_reqs` 夹在 `[2048, 4096]`，`python/sglang/srt/mem_cache/kv_cache_configurator.py:1940`），`max_total_num_tokens` 在几十万到百万级（取决于显存和模型）。如果没有这层间接，要么按"每个请求预留 `max_context_len` 份 KV"分配（`max_num_reqs × max_context_len` 份物理槽位，对绝大多数没跑满上下文的请求是巨大浪费），要么退化成"KV 槽位数=并发请求数"（完全没法支持长上下文）。两级间接让 `ReqToTokenPool` 只管"哪个请求占用了哪些逻辑位置"（一张小表，按并发数计价），`TokenToKVPool` 只管"这些位置对应的显存实际有没有被占用"（一份大池子，按显存计价、按需分配），二者独立伸缩。

**不这样会怎样**：合并成一级会强制"请求数上限"和"KV 总容量"用同一个分母做除法——要么请求数上限被显存打得很低（每个请求都按最大上下文预留），要么 KV 容量被"允许的并发数"打得很低。前缀缓存（RadixAttention）依赖的"同一段 token 被多个请求共享同一批物理槽位"这件事，在没有间接层的情况下也无法表达——共享意味着两个不同的 `req_pool_idx` 在各自的逻辑位置上指向同一批物理槽位号，这正是两级映射天然支持、单级映射做不到的。

**什么时候可以不这样**：单请求、离线批处理、上下文长度固定且总是跑满（比如某些评测脚本一次只推一条定长序列）——这时候 `max_num_reqs=1` 且请求上下文固定，两级间接退化成事实上的一一对应，多出来的一层查表只是常数开销，可以忽略但没必要专门绕过。

### 5.2 KV 张量默认按层拆成 Python list，而不是一整块 5D 张量

**为什么这么设计**：`KVCache` 抽象基类要求所有子类支持"给一个 `layer_id`，O(1) 取出这一层的 K/V"（`get_key_buffer`/`get_value_buffer`/`get_kv_buffer`，`python/sglang/srt/mem_cache/memory_pool.py:1706`-`1726`），这直接对应模型前向"逐层调用 attention"的执行顺序——每层的 `RadixAttention.forward` 只需要、也只能拿到自己这一层的 KV。把 KV 存成 `list[layer_num]` 张量，`k_buffer[layer_id]` 就是一次 Python list 索引，没有额外的切片计算；如果存成一整块 `(layer_num, size, head_num, head_dim)` 的 5D 张量，取一层要做 `kv_buffer[layer_id]`（等价的一次索引，PyTorch 视图切片），理论上开销相近，**真正的差异在流水线并行**：分层存储天然支持"这个 rank 只创建 `[start_layer, end_layer)` 区间对应的那几个张量"（`layer_num` 直接等于本地层数，`python/sglang/srt/mem_cache/memory_pool.py:1655`-`1656`），而一整块大张量的第一维如果要按 PP 切片，需要额外的偏移量簿记。

**不这样会怎样**：如果坚持用一整块 5D 大张量，PP 场景下要么每个 rank 都分配完整 `layer_num` 那一维（浪费显存），要么在张量视图层面做偏移切片（多一层索引间接，且这份视图在跨进程/跨设备转移 KV，比如 PD 分离场景，`get_contiguous_buf_infos`，`python/sglang/srt/mem_cache/memory_pool.py:2217`-`2231`）时更难描述成一段连续裸内存供 RDMA 直接搬运。

**什么时候可以不这样**：单卡、不需要 PD 分离/跨进程零拷贝传输、且刻意想省一点 Python list 遍历开销的场景——`PageMajorMHATokenToKVPool`（`python/sglang/srt/mem_cache/memory_pool.py:3139`）就是这个方向的实现：所有层揉进一块连续 `uint8` buffer，用 strided view 取代 list 索引，代价是放弃了对 tiled KV copy / CPU offload / 投机解码前缀提交这几个假设"每层独立连续张量"的内核路径的支持（这些路径在该类下会显式报错，见 §3.4）。

### 5.3 MLA 池只存压缩后的单一 latent 向量，K/V 共享同一份物理内存

**为什么这么设计**：这不是内存池层面的工程选择，是 MLA（Multi-head Latent Attention）这个注意力变体本身的数学结构决定的——原始 MHA 每个头有独立的 K/V 向量（`head_num × head_dim` 维），MLA 把所有头的信息压缩进一个共享的低秩 latent（`kv_lora_rank` 维）加一份 RoPE 位置分量（`qk_rope_head_dim` 维），解码时用投影矩阵把 latent 展开回各个头的 K/V。既然缓存的东西本来就是"一份共享 latent"，`kv_buffer` 就只需要 `(rows, 1, kv_lora_rank+qk_rope_head_dim)` 这一份，`get_value_buffer` 对它切片取前 `kv_lora_rank` 维（`python/sglang/srt/mem_cache/memory_pool.py:4113`-`4117`）而不是另开一块内存——这是"缓存该缓存的东西"，不是刻意省内存的技巧。

**不这样会怎样**：如果在池子层面强行把 latent 拆成"假装独立的 K 和 V 两块内存"（比如都存一份 `kv_lora_rank` 维的拷贝当 K、当 V），会凭空多出一倍的显存占用和一次冗余拷贝，且这份"K/V"本身在数学上根本不是独立信息——多存的那一份纯粹是浪费。

**什么时候可以不这样**：非 MLA 架构（标准 MHA/GQA）没有这个选择余地，K 和 V 本来就是两份独立算出来的张量，必须分开存；DSA 在 MLA 之上又加了一份独立的索引器 K 缓存（§3.6），说明"要不要额外开一块内存"取决于这份数据在数学上是不是独立信息，不是一个可以随意选的开关。

### 5.4 `page_size` 由 attention 后端的内核约束反过来钉死，不是纯粹的用户配置项

**为什么这么设计**：`page_size` 在 CLI 层是 `Optional[int] = None`（`python/sglang/srt/server_args.py:921`-`925`），默认值由 `_page_size_default` 后处理填充为 1（`python/sglang/srt/arg_groups/overrides.py:2396`），但紧接着 `_mla_backend_page_constraints`（`python/sglang/srt/arg_groups/overrides.py:2129`）会按选定的 attention 后端把它改写：FlashMLA 强制 64、Cutlass MLA 强制 128、TensorRT-LLM MLA 只接受 32 或 64（`python/sglang/srt/arg_groups/overrides.py:2136`-`2159`）。这么设计的原因是 page_size 本质上描述的是"attention kernel 一次访存处理多少个 token 的 KV"，这是 kernel 实现（尤其是用了 TMA/分块 gather 的现代 kernel）的物理约束，用户设一个 kernel 不支持的值，唯一后果是运行时报错或性能塌陷——与其让用户自己踩坑，不如让后端选择在参数解析阶段就把不兼容的值改掉并打警告。

**不这样会怎样**：如果 `page_size` 纯粹按用户输入生效、不做后端一致性校验，用户很容易配出"FlashMLA + page_size=16"这种组合，要么在 kernel 内部触发难以理解的越界/精度错误，要么根本跑不起来——这类错误通常在模型跑到一半才暴露，比参数解析阶段直接报错的排查成本高得多。

**什么时候可以不这样**：用朴素后端（比如 `triton`/`torch_native`）且不追求 kernel 极致性能时，`page_size=1` 的默认值本身就没有额外约束——这也是为什么"一页一个 token"能成为跨平台默认值：它对 kernel 实现的假设最少，兼容面最广，代价是放弃了大 page 才能拿到的访存合并收益。

### 5.5 显存预算用启发式公式估"非 KV 内存"，不做真实前向 profile

**为什么这么设计**：`_profile_available_bytes` 把"当前空闲显存"减去一段静态算出的 `slack_gb`（`python/sglang/srt/mem_cache/kv_cache_configurator.py:1837`），`slack_gb` 的默认来源是 `mem_fraction_static` 的默认值本身就由 `reserved_mem = 512 + activation_tokens*1.5 + tp*pp/8*1024 + reserve_for_graph_mb()`（`python/sglang/srt/server_args.py:5019`-`5038`）这类线性启发式公式反推出来的——`activation_tokens` 取 `chunked_prefill_size`/`max_prefill_tokens`/`decode 并发×draft token 数` 中与当前 serving 模式匹配的那个（`python/sglang/srt/server_args.py:5019`-`5026`）。这是"用一个足够快、足够简单的公式换取启动速度"的工程取舍：不需要真的跑一次前向就能给出一个可用的 `mem_fraction_static`，配合"欠估计就报错提示调大 `mem_fraction_static`、过估计就在实际跑批时 OOM"的双向纠错（`python/sglang/srt/mem_cache/kv_cache_configurator.py:1856`-`1871` 的报错分支），把风险交给运行时而不是启动时。

**不这样会怎样**：静态系数（`1.5`、`512`）在某些模型架构或量化方案下会算多或算少——系数低了会导致 `mem_fraction_static` 定得过高，KV 池分配后真正跑大 batch 前向时激活显存不够，触发运行期 OOM（这类故障通常在服务已经上线、遇到某个没测过的输入组合时才暴露）；系数高了则是白白浪费本可以扩大 KV 池的显存。源码作者自己承认了这一点："The coefficient 1.5 is a heuristic value, in the future, we can do better estimation by looking at the model types, hidden sizes or even do a dummy run"（`python/sglang/srt/server_args.py:4880`）。

**什么时候可以不这样**：`mem_fraction_static` 被用户显式指定时，这整套启发式公式直接被跳过（`if self.mem_fraction_static is None:` 分支，`python/sglang/srt/server_args.py:5006`）——这是"同一份配置反复启动、已经用实测经验校准过安全值"的生产环境该走的路径,用一次性的人工校准换掉每次启动都要承受的估算误差。

## 6. 同位对照（vLLM 在同一位置怎么做）

**逻辑表的粒度：token 级稠密表 vs 块级稀疏表。** SGLang 的 `req_to_token`（`python/sglang/srt/mem_cache/memory_pool.py:279`）形状是 `(size+1, max_context_len)`，**每个 token 一个 int32 物理槽位号**，即使开了分页（`page_size>1`），写进这张表的仍然是逐 token 展开后的物理下标（`python/sglang/srt/mem_cache/allocator/paged.py:165`-`168`）。vLLM 的 `BlockTable` 反过来：`self.block_table` 形状 `(max_num_reqs, max_num_blocks_per_req)`（`vllm:vllm/v1/worker/block_table.py:114`-`116`），**存的是块 ID**，真正的逐 token 物理槽位（`slot_mapping`）是一个只覆盖"当前批次" `max_num_batched_tokens` 大小的瞬时缓冲区（`vllm:vllm/v1/worker/block_table.py:119`-`120`），每一步调度都要重新算。两种设计对应两种取舍：SGLang 的表更大（`max_num_reqs × max_context_len × 4` 字节，一次分配长期持有）但attention 内核不需要做"块号×块大小+偏移"的换算；vLLM 的表更小（块粒度，`max_context_len / block_size` 分之一）但每步都要重新展开 slot_mapping。

**分页粒度的默认值：1 vs 16。** SGLang `page_size` 默认 1（`python/sglang/srt/arg_groups/overrides.py:2396`），vLLM `block_size` 默认 16（`vllm:vllm/config/cache.py:79`，应用于 `vllm:vllm/config/cache.py:317`）。这不是两边team品味不同，而是和上一条互为因果：SGLang 因为逻辑表本身就是稠密逐 token 的，"页"只在分配器内部起作用（§3.2），page_size=1 时几乎不产生额外抽象；vLLM 的块概念直接嵌进了逻辑表的形状定义里，16 是"分配粒度"和"内核访存粒度"共用的一个数字，选得太小（比如 1）会让块表变得和逐 token 表一样大，失去块级索引的优势。

**显存预算：真实 profile run vs 静态公式。** vLLM 的 `GPUWorker.determine_available_memory` 真的构造一个 `max_num_batched_tokens` 规模的假 batch 跑一次前向，用 PyTorch 内存快照差值量出激活峰值（`vllm:vllm/v1/worker/gpu_worker.py:481`）；SGLang 用一条不依赖任何前向执行的线性公式估算同一个量（`python/sglang/srt/server_args.py:5019`-`5038`，见 §5.5）。**本库推断**：这个差异可能与两边的批调度策略有关——vLLM V1 的 `max_num_batched_tokens` 是一个相对固定的调度上限，跑一次这个规模的 profile 前向成本可控；SGLang 支持更灵活的 `chunked_prefill_size`/动态 batch 组合方式，如果要对每种组合都 profile 一次，启动开销会显著增加，用公式换启动速度可能是更划算的取舍——但这只是本库基于两边代码结构的推测，没有找到 SGLang 侧明确解释"为什么不做 profile run"的 commit message 或设计文档，标注为推断而非源码断言。

MLA 池"K/V 共享同一份内存"这条（§3.5、§5.3）在 vLLM 侧目前没有可比对的**已核实**结论——vLLM 的 MLA 支持路径本库尚未在本篇任务范围内逐行核对，此处不编造对照，留给后续横向对比篇（`04-横向对比/03-KV缓存与前缀复用横向对比.md`，不在双链名册内暂不引用）处理。

## 7. 踩坑与反直觉

1. **"page"这个词在 SGLang 里几乎只活在分配器内部，逻辑表从不知道页的存在**。`ReqToTokenPool.req_to_token` 永远是逐 token 稠密表，`PagedTokenToKVPoolAllocator.alloc` 返回的也是展开后的逐 token 下标（`python/sglang/srt/mem_cache/allocator/paged.py:165`-`168`）——如果照搬 vLLM 的直觉去读 SGLang 代码，会本能地去找"块表在哪"，但块/页在这里只是分配器内部一本记账粒度更粗的账，不是贯穿整个系统的核心抽象。
2. **`page_size` 经常不是你在命令行里写的那个数**。`_mla_backend_page_constraints`（`python/sglang/srt/arg_groups/overrides.py:2129`）会按 attention 后端静默改写它并打一条 warning（`python/sglang/srt/arg_groups/overrides.py:2139`-`2142`）；DSA 池甚至在构造时 `assert` 死（`python/sglang/srt/mem_cache/memory_pool.py:4497`）。读日志时如果只看命令行参数不看这条 warning，会以为自己配的 page_size 生效了。
3. **MLA 的 `get_value_buffer` 不是"读取"操作那么简单，它是一次 tensor 切片，返回的是原 `kv_buffer` 的视图**（`python/sglang/srt/mem_cache/memory_pool.py:4109`-`4117`）——对返回值做 in-place 写会直接改到 latent 缓存本身。这与 MHA 池里 `k_buffer`/`v_buffer` 是两个完全独立张量的直觉不同,混着读两种池的代码容易在"能不能对返回值做原地操作"这件事上出错。
4. **SGLang 没有类似 vLLM `profile_run()` 的机制**——全仓 `grep` 找不到 `def profile_run` 这样的函数（本库核实结论，见 §6）。这意味着 `mem_fraction_static` 的默认值本质上是"经验系数",不是"实测值",在没见过的模型架构 / 极端长上下文配置下,默认值给出的安全边界可能不准,这也是为什么源码里专门保留了"显式设置 `mem_fraction_static` 会跳过启发式公式"这条路径（§5.5）。
5. **`ReqToTokenPool` 的容量从来不是独立预算出来的,它是 `max_total_num_tokens` 算完之后的派生量**（`resolve_max_num_reqs`,`python/sglang/srt/mem_cache/kv_cache_configurator.py:1935`-`1948`）。这意味着"调大 `--max-running-requests`"这个直觉上应该只影响并发数的旋钮,实际上会受到 KV 池容量的硬约束——`min(requested_per_worker, token_capacity // 2)`（`python/sglang/srt/mem_cache/kv_cache_configurator.py:1945`),用户要的值拿不到时只是打一条日志警告降级,不是报错拒绝启动。

## 8. 可改进点

1. **`mem_fraction_static` 的默认值公式可以按源码自己留下的方向演进**——`python/sglang/srt/server_args.py:4880` 的注释明确写了"未来可以按模型类型/隐藏层大小更精细地估算,甚至跑一次 dummy run"。**本库推断**：一个成本较低的中间方案是"仅在首次为某个 `(模型架构, chunked_prefill_size, max_bs)` 组合启动时跑一次轻量 profile 并缓存结果",而不必对每次启动都付出真实前向的开销——这不是本库凭空设想,是直接沿着源码作者留下的方向做的最小化落地,具体实现成本本库未评估,标注为长期方向而非可以立刻提 PR 的改动。
2. **`DSATokenToKVPool` 的 `page_size` 断言（`assert self.page_size == 64`,`python/sglang/srt/mem_cache/memory_pool.py:4497`）目前发生在池对象构造期,而不是参数解析期**——用户如果手动指定了一个和 DSA 不兼容的 `page_size` 又没有触发 `_mla_backend_page_constraints` 里已覆盖的那几个后端分支(比如未来新增了一种 DSA 专用后端但忘记同步更新 `_mla_backend_page_constraints`),会在模型加载走到很后面才因为这个 `assert` 崩溃,而不是在 CLI 参数校验阶段就报错。**本库推断**：把这条约束也搬进 `overrides.py` 的后处理链路(类似其它几个 MLA 后端的做法),能让这类配置错误更早暴露、报错信息也能给出更明确的"该用哪个 page_size"提示,而不是一条裸的 `AssertionError`。
3. **`_profile_available_bytes` 的报错分支（`python/sglang/srt/mem_cache/kv_cache_configurator.py:1856`-`1871`）给出的 `suggested_mem_fraction_static` 是"恰好不报错"的临界值**,没有再往上留任何缓冲——**本库推断**：按这条建议值直接设置,大概率会在下一步的真实前向里因为激活显存不够而 OOM(因为这个建议值本身就是从"静态估算,不是实测"这条限制里算出来的,见 §5.5 与 §7 第 4 条),建议信息如果能同时提示"这是理论下限,实际请再留 5%-10% 余量",能减少用户按提示改完参数后又立刻撞见下一个 OOM 的排查成本。

## 9. 自测题与延伸阅读

**自测题（闭卷）**：

1. `ReqToTokenPool` 和 `TokenToKVPool`（Allocator + KVCache）分别按什么单位计价？如果把两者合并成一张表会带来什么问题？
2. `MHATokenToKVPool` 默认布局下,`k_buffer`/`v_buffer` 是什么 Python 类型、每个元素的形状是什么？行下标代表什么？
3. `MLATokenToKVPool` 的 `get_key_buffer` 和 `get_value_buffer` 返回的是不是两块独立显存？为什么 MLA 可以这样设计而 MHA 不行？
4. SGLang 的 `page_size` 默认值是多少？为什么在使用 FlashMLA 之类的后端时它会被自动改写？
5. `PagedTokenToKVPoolAllocator.alloc()` 返回的下标是"页号"还是"逐 token 展开后的物理槽位号"？这个设计对 `req_to_token` 表的形状有什么影响？
6. SGLang 计算 `max_total_num_tokens` 的公式里,有没有一步依赖真实跑一次模型前向？如果没有,它是怎么估算"非 KV 内存"占用的？这和 vLLM 的做法有什么本质区别？
7. `resolve_max_num_reqs` 的输入是什么？这说明 `ReqToTokenPool` 的容量和 `TokenToKVPool` 的容量谁先谁后？

**延伸阅读（本库双链）**：

- [[03-SGLang-RadixAttention与前缀缓存]]——本篇讲的两级映射表是前缀缓存树复用 KV 的物理基础,树节点如何驱动 `alloc`/`free` 在那一篇展开。
- [[05-SGLang-注意力后端矩阵]]——本篇止步于 `get_kv_buffer(layer_id)` 返回什么形状的张量,这些张量之后如何被具体的 attention kernel（FlashInfer/FlashMLA/Triton…）消费,在那一篇继续。
- [[11-SGLang-PD分离与HiCache分层]]——本篇 §2 提到的 `memory_pool_host.py`/`HostKVCache` 只给了接口入口,KV 如何在 GPU/CPU/SSD 之间分层迁移的完整机制在那一篇细讲。
- [[04-vLLM-KV缓存与前缀缓存]]——§6 同位对照的另一半,vLLM 侧块表与容量计算的完整逐行核对在那一篇。
