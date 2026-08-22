# SGLang 投机解码解剖：EAGLE / EAGLE3 与多套 Worker 实现

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：树是每轮动态剪枝的候选池，KV 靠双缓冲窗口原地覆盖，不是逐轮 free

## 0. 结论先行

1. **SGLang 的"投机解码"不是一个 worker，是一张 8 算法 × 多 worker 的注册表。** `SpeculativeAlgorithm` 枚举有 `DFLASH`/`DSPARK`/`EAGLE`/`EAGLE3`/`FROZEN_KV_MTP`/`STANDALONE`/`NGRAM`/`NONE` 八个成员（`python/sglang/srt/speculative/spec_info.py:38`-`45`），`create_worker` 按算法名把服务器实际启动的 worker 类分派到 7 个不同文件（`python/sglang/srt/speculative/spec_info.py:254`-`306`），此外还留了一张给插件算法用的独立注册表 `_REGISTRY`（`python/sglang/srt/speculative/spec_registry.py:166`）。EAGLE/EAGLE3 只是其中最常用的一支，本篇以它为主线，第 2、6、7 节交代其余几支各是什么、什么时候会被选中。
2. **文件名里的 `_v2` 是历史包袱，不是"v1/v2 并存"。** 整个 `speculative/` 目录搜不到任何非 `_v2` 版本的 `eagle_worker.py`；`spec_registry.py` 的类文档直接写明"the spec V1 worker path has been removed"（`python/sglang/srt/speculative/spec_registry.py:37`-`40`）——`_v2` 后缀留在文件名里纯粹是因为没人做过一次批量改名，读代码时不要被这个后缀误导成"还有个 v1 分支要兼容"。
3. **草稿"树"不是 `topk^num_steps` 的固定形状，是一个每轮动态剪枝的候选池。** 每步只保留 `topk` 个累计分数最高的候选继续展开（beam 宽度恒为 `topk`），但每步产生的全部 `topk*topk` 个候选都记进候选池；`num_steps` 步之后候选池大小固定为 `topk*(1+(num_steps-1)*topk)`（公式来自 `python/sglang/srt/speculative/eagle_utils.py:113` 的注释），而真正发给目标模型验证的树只有 `num_draft_tokens-1` 个节点——从候选池里按累计分数**全局** `torch.topk` 选出（`python/sglang/srt/speculative/eagle_utils.py:117`）。同一组 `(topk, num_steps)` 配置下，每一轮实际验证的树形状（哪层留下几个节点）会随 softmax 分数逐轮变化，第 3、4 节手工推演会具体演示这一点。
4. **验证同时支持贪心比对和 rejection sampling 两条路径，由采样温度门控。** `sampling_info.is_all_greedy` 为真时走 `verify_tree_greedy_func`（对目标模型 argmax 逐层匹配）（`python/sglang/srt/speculative/eagle_utils.py:730`-`743`），否则走 `chain_speculative_sampling_triton` 或 `tree_speculative_sampling_target_only`（`python/sglang/srt/speculative/eagle_utils.py:770`-`850`）——是否用严格意义的 rejection sampling（而不是近似的 tree-only 采样）还要看 `speculative_use_rejection_sampling` 这个独立旋钮。第 4 节手工走一遍贪心路径的树遍历。
5. **接受后的 KV 裁剪不是"逐轮 free"，是一个滚动双缓冲窗口原地覆盖。** 每轮 decode 预留的 KV 长度是 `kv_committed_len + 2 * get_alloc_len_per_decode()`（`python/sglang/srt/mem_cache/allocation_sizing.py:48`-`54`），被拒绝分支占用的物理槽位不会立刻还给分配器，而是留在这个"预留窗口"里等下一轮草稿前向直接覆盖写入；真正调用分配器归还物理槽位，只发生在请求彻底结束时的 `release_kv_cache`（`python/sglang/srt/mem_cache/common.py:198`-`241`）。第 4、5 节详细讲这条设计和它的代价。
6. **反直觉：SGLang 确实内置了"高并发自动关草稿"的机制，但默认关闭。** `--speculative-adaptive` 默认 `False`（`python/sglang/srt/server_args.py:2293`-`2297`）；打开后按批大小分档调整 `num_steps`，默认配置里 batch_size≥64 那一档的候选步数只有 `[0]`（`python/sglang/srt/speculative/adaptive_spec_params.py:41`-`46`）——`num_steps=0` 时源码注释直接写着"Drafting disabled (high batch size)"，验证核会"永远接受 root、从目标 logits 采一个新 bonus token——功能上等价于普通逐 token 解码"（`python/sglang/srt/speculative/eagle_worker_v2.py:1206`-`1250`）。第 7 节给出完整证据链和这个机制自身的适用范围限制。
7. **不产出任何实测加速比。** 本机无 GPU，凡涉及"树比链快多少""topk=4 比 topk=1 好多少"，本篇只给源码里能核实的机制性因果链，不编数字。

## 1. 它在系统里的位置

投机解码 worker 夹在调度器的 decode 循环和目标模型之间，同时管着**两个**模型的前向：一个小草稿模型（多步自回归展开候选树）和目标模型本身（一次前向验证整棵树）。调度器不区分算法细节，统一通过 `ServerArgs.speculative_algorithm` 解析出的 `SpeculativeAlgorithm` 枚举拿到一个 worker 类（`python/sglang/srt/speculative/spec_info.py:254`），这个类必须实现 `draft()`/`draft_extend()`（`EagleDraftWorkerBase`，`python/sglang/srt/speculative/base_spec_worker.py:57`-`72`）和 `verify()`（`BaseSpecWorker`，`python/sglang/srt/speculative/base_spec_worker.py:147`）。

以 EAGLE/EAGLE3 为例，一次 decode 轮次的完整时序在 `EAGLEWorkerV2.forward_batch_generation`（`python/sglang/srt/speculative/eagle_worker_v2.py:1147`）里可以逐行读出：

```
Scheduler（见 [[02-SGLang-Scheduler事件循环]]）
   │ 每个 decode step 调一次 forward_batch_generation()
   ▼
EAGLEWorkerV2.forward_batch_generation (eagle_worker_v2.py:1147)
   │ 1. activate_step_by_batch(bs)          —— 按批大小切换自适应 num_steps（eagle_worker_v2.py:1187）
   │ 2. if num_steps == 0:
   │        _build_trivial_verify_input()   —— 1 节点树，functionally 等价普通解码（eagle_worker_v2.py:1206-1250）
   │    else:
   │        draft_worker.draft(batch)       —— EagleDraftWorker，草稿模型多步展开+建树（eagle_worker_v2.py:1219）
   ▼
self.verify(batch)  →  run_eagle_verify (eagle_worker_common.py:461)
   │ 目标模型一次前向验证整棵树 + 贪心/rejection 采样 + KV 裁剪
   ▼
draft_worker._draft_extend_for_decode(batch, batch_output)  (eagle_worker_v2.py:1240)
   │ 用本轮接受的 token 补齐草稿模型自己的 KV/隐状态，为下一轮 draft() 做种子
   ▼
下一轮 decode
```

### 1.1 七种算法 × Worker 实现清单

`SpeculativeAlgorithm.create_worker`（`python/sglang/srt/speculative/spec_info.py:254`-`306`）是唯一的分派入口，按检查顺序（`DFLASH` → `DSPARK` → `FROZEN_KV_MTP` → `EAGLE`+multi-layer → `EAGLE` → `STANDALONE` → `NGRAM`）把算法名映射到 7 个 worker 类中的一个。逐个核对每个类的继承关系（`grep "^class.*Worker"`）后，实现清单是：

| 算法 | Worker 类:定义行 | 继承自 | 是什么 | 何时选中 |
|---|---|---|---|---|
| `EAGLE` / `EAGLE3`（未开 multi-layer） | `EAGLEWorkerV2`，`python/sglang/srt/speculative/eagle_worker_v2.py:1050` | `BaseSpecWorker` | 标准 EAGLE 动态剪枝树草稿；**`EAGLE3` 与 `EAGLE` 复用同一个 worker 文件**，二者区别在草稿模型架构本身要不要拼接目标模型多层 aux hidden states（`get_draft_input_from_target_hidden_dim`，`python/sglang/srt/speculative/eagle_utils.py:446`-`485`），不在 worker 分发这一层 | `--speculative-algorithm EAGLE` 或 `EAGLE3`，且不开 `--enable-multi-layer-eagle` |
| `EAGLE` + `--enable-multi-layer-eagle` | `MultiLayerEagleWorkerV2`，`python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py:918` | `BaseSpecWorker`（**独立实现，不继承 `EAGLEWorkerV2`**，只是复用 `eagle_worker_common.py` 里的共享函数） | 多层草稿头变体；`run_eagle_verify` 用 `finalize_tree_path=False` 调用它（对照 `EAGLEWorkerV2.verify` 传 `True`，见 `python/sglang/srt/speculative/eagle_worker_common.py:484`-`486` 的文档说明），因为它"从未跑过"接受路径压缩 | `--speculative-algorithm EAGLE --enable-multi-layer-eagle` |
| `STANDALONE` | `StandaloneWorkerV2`，`python/sglang/srt/speculative/standalone_worker_v2.py:147` | **`EAGLEWorkerV2`（直接子类，复用绝大部分 verify/KV 编排逻辑）** | 独立小模型草稿，不与目标模型共享 embedding/lm_head（类文档，`python/sglang/srt/speculative/standalone_worker_v2.py:34`-`35`） | `--speculative-algorithm STANDALONE`，默认 `(num_steps, topk, num_draft_tokens)=(3,1,4)`（`_auto_choose_speculative_params`，`python/sglang/srt/arg_groups/speculative_hook.py:819`-`820`） |
| `FROZEN_KV_MTP` | `FrozenKVMTPWorkerV2`，`python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py:676` | **`EAGLEWorkerV2`（直接子类）** | 只读目标模型 KV、自己不持有 KV 池的 MTP 草稿（模块文档，`python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py:14`-`19`），对应 DeepSeek/Gemma4 assistant 一类模型 | 显式指定，或 Gemma4 assistant 草稿模型被自动提升（`_resolve_speculative_algorithm_alias`，`python/sglang/srt/arg_groups/speculative_hook.py:24`-`60`） |
| `DFLASH` | `DFlashWorkerV2`，`python/sglang/srt/speculative/dflash_worker_v2.py:256` | `BaseSpecWorker`（独立实现） | 线性（非树）block 草稿，`DFlashVerifyInput.topk` 恒为 1（`python/sglang/srt/speculative/dflash_info.py:24`-`33` 的注释直接写"DFLASH verify is linear (non-tree)"） | `--speculative-algorithm DFLASH` |
| `DSPARK` | `DSparkWorkerV2`，`python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py:83` | `BaseSpecWorker`（独立实现） | ragged verify + 全批次逐 token 预算分配的变体，与 `DFLASH` 同属"KDA fold-every-commit family"（`python/sglang/srt/server_args.py:6489`-`6498`） | `--speculative-algorithm DSPARK` |
| `NGRAM` | `NGRAMWorker`，`python/sglang/srt/speculative/ngram_worker.py:71` | `BaseSpecWorker`（独立实现） | 没有神经网络草稿模型，从历史生成语料的 n-gram 表查续写候选（`python/sglang/srt/speculative/cpp_ngram/ngram_corpus.py`），复用 `eagle_sample` 做验证（`python/sglang/srt/speculative/ngram_worker.py:22` import） | `--speculative-algorithm NGRAM` |
| `NONE` | 不创建 worker | — | 投机解码关闭，`create_worker` 对 `NONE` 直接断言拒绝调用（`python/sglang/srt/speculative/spec_info.py:257`-`259`） | 默认值（`speculative_algorithm=None`，`python/sglang/srt/server_args.py:2116`-`2120`） |

三个要点：**（a）没有 v1/v2 并存**——第 0、7 节已给出证据链，`_v2` 只是命名遗留；**（b）"独立文件"不等于"独立实现"**——`StandaloneWorkerV2`/`FrozenKVMTPWorkerV2` 直接继承 `EAGLEWorkerV2`，只重写草稿模型的具体前向方式，`verify()`/KV 裁剪等编排逻辑是继承来的同一份代码；**（c）真正另起炉灶的是 `MultiLayerEagleWorkerV2`/`DFlashWorkerV2`/`DSparkWorkerV2`/`NGRAMWorker` 四个**——它们的验证/KV 语义和 `EAGLEWorkerV2` 有实质差异（非树验证、ragged 布局、无神经网络草稿），继承共享基类换不来代码复用，只能各自实现。

草稿模型和目标模型各自持有**独立的** KV 池：草稿模型的 KV 由 `EagleDraftWorker` 管，目标模型的 KV 走正常的 `req_to_token_pool`/`token_to_kv_pool_allocator`（见 [[04-SGLang-内存池与KV布局]]）。两者的接口靠 `EagleDraftInput`/`EagleVerifyInput`/`EagleDraftExtendInput` 三个数据结构（`python/sglang/srt/speculative/eagle_info.py`）串联，第 3 节展开。

## 2. 代码地图（文件 → 职责，带行号）

`speculative/` 目录共 99 文件（含 `cpp_ngram/`、`dspark_components/` 两个子目录），本篇只精读 EAGLE 家族相关的部分：

| 文件:行 | 职责 |
|---|---|
| `python/sglang/srt/speculative/spec_info.py:31`-`126` | `SpeculativeAlgorithm` 枚举：8 个算法成员（`:38`-`45`）+ `is_eagle()`/`need_topk()` 等判定方法 |
| `python/sglang/srt/speculative/spec_info.py:254`-`306` | `create_worker`：算法名 → worker 类的完整分派表 |
| `python/sglang/srt/speculative/spec_registry.py:25`-`260` | `CustomSpecAlgo` 插件基类 + `_REGISTRY`（`:166`）+ "spec V1 已删除"的类文档（`:37`-`40`） |
| `python/sglang/srt/speculative/base_spec_worker.py:57`-`145` | `EagleDraftWorkerBase`：`draft()`/`draft_extend()` 抽象接口 |
| `python/sglang/srt/speculative/eagle_worker_v2.py:129`-`765` | `EagleDraftWorker`：草稿侧主体，`draft()`（`:501`）、`draft_forward()`（`:564`，per-step 展开循环 `:624`-`708`） |
| `python/sglang/srt/speculative/eagle_worker_v2.py:1050`-`1575` | `EAGLEWorkerV2`：调度器实际持有的顶层 worker，`forward_batch_generation`（`:1147`）、`verify`（`:1539`）、自适应挂钩（`:1329`-`1421`） |
| `python/sglang/srt/speculative/eagle_worker_common.py:316`-`403` | `build_eagle_verify_input`：draft() 的收尾，调用建树 kernel 组装 `EagleVerifyInput` |
| `python/sglang/srt/speculative/eagle_worker_common.py:461`-`660` | `run_eagle_verify`：verify 的完整流水线（target 前向 → 采样 → KV 裁剪） |
| `python/sglang/srt/speculative/eagle_utils.py:107`-`136` | `organize_draft_results`：候选池全局 `torch.topk` 剪枝（公式注释在 `:113`） |
| `python/sglang/srt/speculative/eagle_utils.py:151`-`293` | `build_tree_kernel_efficient`：Python 侧建树入口，按硬件分派到 CUDA/Triton/CPU/NPU 核 |
| `python/sglang/srt/speculative/eagle_utils.py:653`-`911` | `eagle_sample`：贪心 vs rejection 双路径验证与采样 |
| `python/sglang/srt/speculative/eagle_info.py:16`-`138` | `EagleVerifyInput` 数据结构，含 `max_tree_depth`（`:44`）/`tree_topk`（`:50`）两个属性 |
| `python/sglang/srt/speculative/spec_utils.py:286`-`341` | `select_top_k_tokens`：单步 topk×topk 展开 + beam 剪枝的核心数学 |
| `python/sglang/srt/speculative/spec_utils.py:694`-`756` | `move_accept_tokens_to_target_kvcache`：接受路径 KV 从散列位置搬到连续区 |
| `python/sglang/kernels/ops/speculative/spec_tree.py:18`-`173` | Triton 版建树 kernel（mask + retrieve_index/next_token/next_sibling），逐行可读 |
| `python/sglang/kernels/ops/speculative/spec_tree.py:176`-`281` | Triton 版贪心验证 kernel：树遍历 + 逐层匹配 |
| `python/sglang/srt/speculative/adaptive_spec_params.py:22`-`259` | `DEFAULT_ADAPTIVE_CONFIG`（`:22`-`47`）+ `AdaptiveStepSlot` EMA 步数调节器 |
| `python/sglang/srt/speculative/adaptive_runtime_state.py:61`-`134` | `AdaptiveController`：预建每档 `SpecRuntimeState`，切换时整体替换而非现建 |
| `python/sglang/srt/arg_groups/speculative_hook.py:543`-`844` | `_handle_eagle_family` + `_auto_choose_speculative_params`（`:813`）：三个核心旋钮的默认值来源 |
| `python/sglang/srt/mem_cache/allocation_sizing.py:17`-`77` | `get_alloc_len_per_decode`/`get_alloc_reserve_per_decode`/`page_aligned_decode_alloc_lens`：双缓冲窗口的算术 |
| `python/sglang/srt/mem_cache/common.py:198`-`268` | `release_kv_cache`/`_release_overallocated_kv_indices`：请求结束时才真正归还超额 KV |
| `python/sglang/srt/mem_cache/memory_pool.py:2801`-`2814` | `move_kv_cache`：接受路径 KV 的物理拷贝实现 |

（以上 20 条，超过 `## 2` 硬指标的 10 条；第 3-7 节还会补充更多。）

## 3. 核心数据结构

三个 `SpecInput` 子类串起草稿-验证-补种三个阶段（都在 `python/sglang/srt/speculative/eagle_info.py`），三者都继承自同一个抽象基类 `SpecInput`（`python/sglang/srt/speculative/spec_info.py:322`-`365`），后者定义了跨算法通用的阶段判断方法 `is_draft_input()`/`is_verify_input()`（`spec_info.py:351`-`365`），供调度器和 attention 后端在不知道具体是哪个算法的情况下也能判断"现在处于哪个阶段"。

### 3.1 `EagleDraftInput`——草稿阶段的输入/输出

定义在 `python/sglang/srt/speculative/eagle_info.py:142`-`268`。关键字段：

- `topk_p`/`topk_index`（`:148`-`149`）——当前展开层的 `topk` 个候选及其条件概率，形状 `(b, topk)`，是 4.1 节 `select_top_k_tokens` 每一步读写的核心状态。
- `hidden_states`（`:153`-`156`）——喂给草稿模型下一步前向的隐状态；注释明确写了 `None` 是合法值，`STANDALONE` 这类不读隐状态的算法（独立小模型，不接目标模型的 hidden states）会让它保持 `None`。
- `bonus_tokens`（`:163`-`165`）——上一轮验证结束后确定要继续展开的"种子" token，即 4.5 节 draft_extend 追平之后交给下一轮 `draft()` 的起点。
- `draft_probs`（`:150`-`152`）——只在 `speculative_use_rejection_sampling=True` 时才非空,这是贪心路径和 rejection 路径共享同一数据结构、但字段含义分叉的地方：贪心路径完全不填这个字段。

### 3.2 `EagleVerifyInput`——建树后交给目标模型验证的完整树描述

定义在 `python/sglang/srt/speculative/eagle_info.py:16`-`138`。核心不是 token 本身，而是三个索引张量：

| 字段 | 行号 | 含义 |
|---|---|---|
| `draft_token` | `:17` | 树里每个节点代表的 token id，摊平成一维 |
| `custom_mask` | `:18` | 4.2 节推演出的树注意力可见性矩阵 |
| `positions` | `:19` | 每个节点的位置编码（`seq_len` + 树深度） |
| `retrieve_index` | `:20` | 每个树节点在摊平数组里的全局下标 |
| `retrieve_next_token` | `:21` | 每个节点的第一个孩子（邻接表头指针） |
| `retrieve_next_sibling` | `:22` | 同层兄弟链表指针 |
| `spec_steps`/`topk`/`draft_token_num` | `:24`-`26` | 建树时的三个配置参数本身，随 `EagleVerifyInput` 一起传给验证 kernel |
| `draft_probs` | `:32` | 只在 rejection sampling 下非空，语义同 `EagleDraftInput.draft_probs` |

`retrieve_index`/`retrieve_next_token`/`retrieve_next_sibling` 这三者合起来就是一棵用邻接表表示的树；`max_tree_depth` 属性恒等于 `spec_steps + 1`（`eagle_info.py:44`-`48`，含 root），`tree_topk` 属性直接返回 `topk`（`eagle_info.py:50`-`54`），作为"这是不是规则分支"的标记，供验证 kernel 判断能否走定长快速路径——`DFLASH` 这种非树算法会把 `topk` 恒设为 1（3.3 节之外的 `DFlashVerifyInput` 有自己的等价字段，本篇不展开）。

### 3.3 `EagleDraftExtendInput`——验证之后的草稿 KV 补种输入

定义在 `python/sglang/srt/speculative/eagle_info.py:271`-`389`。`num_correct_drafts`/`num_accept_tokens`（`num_accept_tokens = num_correct_drafts + 1`，`:286`-`289` 注释）告诉草稿模型这一轮目标模型到底接受了几个 token，草稿模型要用这些真实 token（而不是自己猜的）重新跑一遍 extend，把自己的 KV 追上目标模型的进度——这是 EAGLE 系列"草稿模型的输入必须是已验证 token"这一约束的直接体现，4.5 节已经手工走过这一步。

### 3.4 `SpecRuntimeState`/`AdaptiveController`——自适应切换的状态载体

自适应侧还有一个跨轮次持久的状态对象：`SpecRuntimeState`（`python/sglang/srt/speculative/adaptive_runtime_state.py:18`-`43`）打包了某个 `(num_steps, num_draft_tokens)` 配置下全部形状相关资源：

| 字段 | 阶段 | 内容 |
|---|---|---|
| `draft_attn_backend`/`cuda_graph_runner` | 草稿 | 草稿模型多步展开用的 attention 后端与 CUDA Graph |
| `target_attn_backend`/`target_graph_runner` | 验证 | 目标模型一次前向验证树用的 attention 后端与 CUDA Graph |
| `draft_extend_attn_backend`/`cuda_graph_runner_for_draft_extend` | 补种 | 4.5 节 draft_extend 阶段用的 attention 后端与 CUDA Graph |

`AdaptiveController`（`adaptive_runtime_state.py:61`-`134`）为 `candidate_steps` 里的每个候选步数**预先构建**一份 `SpecRuntimeState` 并缓存在 `self._states` 字典里，切换步数时 `_activate`（`:128`-`134`）只是换一次字典查找、整体替换引用，不涉及任何 GPU 端重建——第 5 节第 5 条详细讲这个设计的代价。

## 4. 主流程走读

### 4.1 建树：从"分数最高的候选"到"发给目标模型的固定形状"

以 `topk=2, num_steps=3` 为例手工推演（真实默认值是 Llama 架构的 `(num_steps, topk, num_draft_tokens) = (5, 4, 8)`，来自 `_auto_choose_speculative_params`，`python/sglang/srt/arg_groups/speculative_hook.py:821`-`822`；这里用更小的数字方便画图）。

`draft_forward` 的循环体（`python/sglang/srt/speculative/eagle_worker_v2.py:624`-`708`）每步调 `select_top_k_tokens`（`python/sglang/srt/speculative/spec_utils.py:343`-`355`），第 0 步走 `_select_top_k_tokens_first`（`:286`-`303`，直接取 root 的 topk 个孩子，不需要剪枝），第 1 步起走 `_select_top_k_tokens_later`（`:306`-`340`，每个在展开的节点各生成 `topk` 个孩子，`topk*topk` 个候选全部计入分数，再用 `fast_topk` 在这 `topk*topk` 个候选里**全局**选 `topk` 个继续展开）。

假设 root 的两个孩子概率是 `a1=0.60`、`a2=0.40`：

```
第 0 步（root 的 topk=2 个孩子，全部继续展开）
  root(1.0)
   ├─ a1  score=0.60
   └─ a2  score=0.40

第 1 步（a1、a2 各展开 topk=2 个孩子，共 2*2=4 候选，全部记入候选池；
        但只有全局 top-2 继续展开 —— 结果两个都来自 a1，a2 整条分支被剪掉）
  a1 ─┬─ a1x1  score=0.60*0.5=0.30   ← 继续展开
      └─ a1x2  score=0.60*0.5=0.30   ← 继续展开
  a2 ─┬─ a2x1  score=0.40*0.7=0.28   （记入候选池，但不再展开）
      └─ a2x2  score=0.40*0.3=0.12   （记入候选池，但不再展开）

第 2 步（最后一步，只记分不再展开：a1x1、a1x2 各生 2 个孩子，共 4 候选）
  a1x1 ─┬─ a1x1y1  score=0.30*0.9=0.270
        └─ a1x1y2  score=0.30*0.1=0.030
  a1x2 ─┬─ a1x2y1  score=0.30*0.4=0.120
        └─ a1x2y2  score=0.30*0.6=0.180
```

候选池共 `2 + 4 + 4 = 10` 个（吻合 `topk*(1+(num_steps-1)*topk) = 2*(1+2*2) = 10`），按分数排序：

```
0.60 a1   0.40 a2   0.30 a1x1   0.30 a1x2   0.28 a2x1
0.270 a1x1y1   0.180 a1x2y2   0.120 a1x2y1   0.12 a2x2   0.030 a1x1y2
```

设 `num_draft_tokens = 6`，`organize_draft_results`（`python/sglang/srt/speculative/eagle_utils.py:117`）取全局 top-5（`num_draft_tokens-1`）：**a1、a2、a1x1、a1x2、a2x1**——注意 `a1x1y1`（0.270）虽然分数不低，还是以 0.01 之差输给了 `a2x1`（0.28），第 2 层的两个候选整个没能进入这一轮的验证树。这就是第 0 节第 3 条说的"实际验证的树形状每轮都不同"：同样 `(topk=2, num_steps=3)` 配置，若这一轮 `a1` 的两个孩子分数没那么接近（比如 `a1x1=0.55`），第 2 层的候选就可能挤掉 `a2` 分支进入最终树。

最终发给目标模型验证的树是 `bonus(root) + {a1, a2, a1x1, a1x2, a2x1}` 共 6 个节点，编号 `0..5`。

### 4.2 树注意力 mask：谁能看见谁

`build_tree_kernel_efficient`（`python/sglang/srt/speculative/eagle_utils.py:151`）按硬件分派到 CUDA/Triton/CPU/NPU 的同名 kernel；仓库里能读到完整 Python 源码的是 Triton 版（`python/sglang/kernels/ops/speculative/spec_tree.py:18`-`173`，XPU 后端用它，语义与 CUDA 版一致；CUDA/C++ 版本在独立的 `sgl-kernel` 包里，不在本仓库 clone 范围内，标注为未查证）。

核心循环（`python/sglang/kernels/ops/speculative/spec_tree.py:59`-`104`）从最后一个节点往前找父节点：`selected_index[i-1] // topk` 得到"这是第几个继续展开的父节点分支"，再在 `parent_list` 里查出父节点在候选池里的原始 token 位置，找到匹配后把当前节点接到父节点的孩子链表（`retrieve_next_token`）或兄弟链表（`retrieve_next_sibling`）上。对我们的 6 节点树，链表结构是：

```
retrieve_next_token[root]=a1     retrieve_next_sibling[a1]=a2      retrieve_next_sibling[a2]=-1
retrieve_next_token[a1]=a1x1     retrieve_next_sibling[a1x1]=a1x2  retrieve_next_sibling[a1x2]=-1
retrieve_next_token[a2]=a2x1     retrieve_next_sibling[a2x1]=-1
retrieve_next_token[a1x1]=-1（叶子）  retrieve_next_token[a1x2]=-1（叶子）  retrieve_next_token[a2x1]=-1（叶子）
```

mask 构造（`python/sglang/kernels/ops/speculative/spec_tree.py:109`-`173`）逐节点填：先把节点自己标为可见（`python/sglang/kernels/ops/speculative/spec_tree.py:121`），再沿着上面这条父指针链一路往上标记祖先（`python/sglang/kernels/ops/speculative/spec_tree.py:125`-`169`，循环上限是 `depth` 即 `spec_steps`，遇到父节点是 root 就停），从不触碰非祖先节点——每个节点最终能看到的只有"目标模型已生成的前缀 + 自己在树里的祖先链"：

```
root  : [root]
a1    : [root, a1]
a2    : [root, a2]
a1x1  : [root, a1, a1x1]
a1x2  : [root, a1, a1x2]
a2x1  : [root, a2, a2x1]
```

同一次前向里，`a1x1` 看不见 `a1x2`（同层兄弟，路径不同），`a2x1` 也看不见 `a1` 这一支的任何节点——这正是"一次前向验证一整棵树"能够成立的前提：树里每条根到叶的路径，在注意力意义上都是一条独立的因果链,只是共享了同一次前向、同一批 KV 写入。`positions` 的计算（`python/sglang/kernels/ops/speculative/spec_tree.py:170`-`173`）也呼应这一点：每个节点的位置编码是 `seq_len + 它在树里的深度`，所以 `a1x1` 和 `a2x1` 虽然是不同 token、不同父节点，但深度相同（2），会拿到相同的位置 id——这对 RoPE 是正确的，因为"深度"就是这条假设路径上的逻辑序列位置。

### 4.3 验证与接受：手工遍历一次贪心比对

`verify_tree_greedy_kernel_triton`（`python/sglang/kernels/ops/speculative/spec_tree.py:176`-`281`）从 `accept_index[:,0] = root` 出发，每一层用目标模型在"当前已接受节点"这一位置的 argmax 预测（`target_predict`），去匹配该节点**所有孩子**（通过 `retrieve_next_token` 起步、`retrieve_next_sibling` 遍历同层兄弟）里是否有 token 与之相同；命中就把接受指针移到那个孩子、进入下一层，全部孩子都不匹配就停止。

接上面的例子，假设目标模型在验证这棵树时给出的贪心预测是：`predict(root 之后) = a1 的 token`，`predict(a1 之后) = a1x2 的 token`（不是 `a1x1`），`predict(a1x2 之后) = 某个新 token Z`：

```
第 1 层：候选 = {a1, a2}（root 的孩子）。predict(root 之后) == a1 的 token → 命中，接受 a1。
第 2 层：候选 = {a1x1, a1x2}（a1 的孩子）。先试 a1x1：不匹配；再试兄弟 a1x2：predict(a1 之后) == a1x2 的 token → 命中，接受 a1x2。
第 3 层：a1x2 是叶子（没有孩子进入本轮验证树，见 4.1 节末尾的取舍）→ 无候选可比，链条在此终止。
```

最终 `num_correct_drafts = 2`（a1、a1x2 两个真正被验证通过的草稿 token），`accept_index = [root, a1, a1x2, -1]`，加上循环结束后的收尾（`python/sglang/kernels/ops/speculative/spec_tree.py:276`-`281`：把 `predict(a1x2 之后)` 的 Z 也写回 `predicts[a1x2]` 作为"bonus token"）——这一步不需要再验证，因为它就是目标模型自己算出来的下一个 token，天然正确。所以这一轮实际提交给用户的是 3 个 token（`num_correct_drafts + 1`，对应 `python/sglang/srt/speculative/eagle_utils.py:911` 的返回值），`a2`、`a2x1`、`a1x1` 三个未被走到的节点连同它们的 KV 一起被丢弃——不是"错误"，只是这条假设路径没有被目标模型证实。

### 4.4 接受后的 KV 裁剪

`accept_index` 指向的 3 个节点（`root, a1, a1x2`）在树里的物理槽位是散的（它们在 `out_cache_loc` 里的位置取决于建树时的展开顺序，不连续），但下一轮 decode 需要这个请求的 KV 是连续的一段。`move_accept_tokens_to_target_kvcache`（`python/sglang/srt/speculative/spec_utils.py:694`-`756`）计算一段目标连续区间 `[kv_committed_len, kv_committed_len + num_correct_drafts + 1)`（`:723`-`747`），再调用 `TokenToKVPool.move_kv_cache`（`python/sglang/srt/mem_cache/memory_pool.py:2801`）把 K/V 张量从散列的 `accept_out_cache_loc` **物理拷贝**到这段连续区间——这是真实的显存搬运，不是指针重排。`a2`、`a2x1`、`a1x1` 三个节点占用的槽位不参与这次拷贝，直接被撇下；它们的物理槽位去了哪里、什么时候被回收，是第 5 节的主题。

### 4.5 draft_extend：草稿模型自己的 KV 追平

`verify()` 结束后，目标模型的 KV 已经通过 4.4 节的搬运追平到本轮实际接受的 token；但草稿模型自己那份独立 KV 池还停在上一轮的进度——它在 `draft()` 阶段写入的是"猜测"出来的候选 token 的 KV（含被拒绝的分支），不是这一轮真正被接受的序列。`EagleDraftWorker._draft_extend_for_decode`（`python/sglang/srt/speculative/eagle_worker_v2.py:898`-`933`）用真实接受的 token 重新跑一遍草稿模型的 extend 前向，把草稿模型的 KV 也追到同一个进度：

- `EagleDraftExtendInput.num_correct_drafts = accept_lens - 1`（`eagle_worker_v2.py:905`，注释直接写"accept_lens includes the bonus token; correct drafts exclude it"）——和 4.3 节手工推演里 `num_correct_drafts=2` 是同一个量。
- `num_tokens_per_req` 被设成 `speculative_num_draft_tokens`（整棵树的宽度）而不是 `num_steps+1`（`eagle_worker_v2.py:908`-`909` 的注释解释：这是为了让 DP 场景下的 MLP-sync 填充在 `topk>1` 时保持形状一致，不是说每次都真的写这么多 token）。
- `select_index`（`eagle_worker_v2.py:912`-`920`）算出每个请求在"整棵树摊平之后的缓冲区"里、它真正被接受的最后一个节点落在哪个下标——这一步是给下游"只对被接受的那些位置跑 lm_head/写 KV"用的定位索引，避免对整棵树的每个节点（包括被拒绝的）都重复计算。

这一步跑完，草稿模型的隐状态和 KV 都追平到"目标模型刚刚认证过的最后一个 token"，作为下一轮 `draft()` 的种子（`EagleDraftInput.hidden_states`/`bonus_tokens`）——整条 decode 循环因此形成一个稳定的三段闭环：**draft（猜）→ verify（目标模型认证）→ draft_extend（草稿模型追平真相）**，任何一段的状态都不会比另外两段"更超前"或"更落后"太久，这是能支持"下一轮接着从上一轮的猜测继续猜"这件事成立的前提。

## 5. 设计决策与代价

1. **候选池"每步 topk×topk 展开，全局剪到 topk 继续"，而不是维护 topk 条独立链或展开满树。**
   - **为什么这么设计**：满树（`topk^num_steps`）在 `topk=4, num_steps=5` 下是 1024 个节点，验证一次前向的注意力和采样开销随之爆炸；`topk` 条独立链又会把每一步的展开预算平均分给可能已经不靠谱的分支。`_select_top_k_tokens_later`（`python/sglang/srt/speculative/spec_utils.py:306`-`340`）的做法是标准 beam search：把预算集中到累计分数最高的路径上，4.1 节的例子里 `a2` 分支第 1 层分数还有 0.40，但因为 `a1` 的两个孩子都恰好比它高，`a2` 分支被整个剪掉——这是"树"相对"topk 条独立链"的核心收益：分支预算是动态再分配的，不是静态平分的。
   - **不这样会怎样**：如果改成 `topk` 条独立线性链（等价于 `topk` 次独立的 chain 投机），每条链固定占 `1/topk` 的展开预算，即使某条链前几步命中率很高、后面很可能继续对，也不能从别的链"借"预算；总候选数固定为 `topk*num_steps` 而不是 `topk*(1+(num_steps-1)*topk)`，覆盖到的假设路径少得多。
   - **什么时候可以不这样**：`topk=1` 时这套机制自动退化成纯链（`_select_top_k_tokens_later` 的 `topk*topk` 项退化为 `1*1`），`python/sglang/srt/arg_groups/speculative_hook.py:675`-`683` 也在这种情况下把 `speculative_num_draft_tokens` 强制改写为 `speculative_num_steps + 1`——即"topk=1 时干脆别装作是棵树，按链处理"。`STANDALONE`（独立小模型草稿）和 `FROZEN_KV_MTP`（DeepSeek/Gemma 风格 MTP）的默认参数就是 `topk=1`（`_auto_choose_speculative_params`，`python/sglang/srt/arg_groups/speculative_hook.py:819`-`820`、`:823`-`839`），说明"要不要用树"本身是一个按算法特性做的选择，不是 EAGLE 家族的强制约定。

2. **KV 分配用"滚动双缓冲预留窗口，原地覆盖"，而不是"逐轮为草稿分配、拒绝分支立刻 free"。**
   - **为什么这么设计**：`get_alloc_reserve_per_decode`（`python/sglang/srt/mem_cache/allocation_sizing.py:48`-`54`）把每轮预留长度设成 `2 * get_alloc_len_per_decode()`，`page_aligned_decode_alloc_lens`（`:57`-`77`）里 `nxt = max(cur, kv_committed_len + reserve 向上取整)`——只要现有预留窗口还没被已提交长度追上，`nxt - cur == 0`，`alloc_for_spec_decode`（`python/sglang/srt/mem_cache/allocation.py:646`-`689`）就完全不调用分配器，直接复用上一轮已经分配好的物理页，让这一轮草稿前向把新数据写进去覆盖旧数据。逐轮 free+realloc 需要走一次分配器的空闲链表操作，在高吞吐 decode 循环里这是要反复付的税；原地覆盖把这笔税降到"预留窗口需要扩容时才付一次"。
   - **不这样会怎样**：如果每轮都严格按"这一轮实际写了多少就分配多少、多余的立刻 free"来做，等价于把 `_release_overallocated_kv_indices`（`python/sglang/srt/mem_cache/common.py:244`-`268`）从"请求结束时调一次"变成"每个 decode step 调一次"——每步都要做一次 `allocator.free_segment` + 下一步的重新 `alloc_token_slots`，对分配器空闲表的读写频率乘以 decode 步数,且如果分配器在两次操作之间把这批页分给了别的请求，还得处理"这批页面刚被别人认领"的竞争。
   - **什么时候可以不这样**：非投机解码路径（`speculative_algorithm is None`）就是"这样"——`_release_overallocated_kv_indices` 里那条断言 `spec_algo is None and not strip_thinking_cache` 时要求 `start_p == end_p`（`python/sglang/srt/mem_cache/common.py:253`-`256`），即普通解码根本不允许有"已分配未提交"的悬空区间，每步分配即等于每步提交。换句话说，双缓冲窗口是投机解码专属的让步，代价是**这条路径上,一个请求在存活期间可能一直占着比它实际序列长度更多的物理 KV 页**——这正是第 8 节可改进点第 1 条要处理的问题。

3. **验证按采样温度门控到贪心比对或 rejection sampling 两条独立代码路径，而不是统一走 rejection sampling。**
   - **为什么这么设计**：贪心解码下，"接受"的定义是"草稿 token 是否等于目标模型 argmax"，这是一个确定性谓词，`verify_tree_greedy_func`（`python/sglang/srt/speculative/eagle_utils.py:378`-`443`）不需要任何随机数、不需要草稿模型给出概率分布 `draft_probs`，比通用 rejection sampling 更省（不用算 softmax、不用生成 uniform 随机数、不用比较接受率阈值）。真正需要以正确的边际分布采样时（非贪心），才用 `chain_speculative_sampling_triton`/`tree_speculative_sampling_target_only`（`:770`-`850`），这条路径确实需要 `draft_probs` 和逐 token 的 rejection 系数。
   - **不这样会怎样**：如果所有请求（包括全贪心的）都统一走 rejection sampling 代码路径，贪心请求要多算一整套没必要的 softmax/top-k/top-p 归一化和随机采样，且贪心场景下 rejection sampling 在数学上等价于比较 argmax 是否相等，多出来的计算是纯浪费；反过来如果只保留贪心比对、砍掉 rejection sampling，非贪心（temperature 采样、top_p 等）请求就没有理论上无偏的验证算法可用。
   - **什么时候可以不这样**：混合批次（同一批里有的请求贪心、有的不贪心）会让"哪条路径"这件事变成逐请求判断而不是逐批次判断——`eagle_sample`（`python/sglang/srt/speculative/eagle_utils.py:730`）用 `sampling_info.is_all_greedy` 判断整批，但真正是否走贪心核还看 `sampling_info.is_all_greedy or _is_cpu or _is_npu or _is_hip or _is_xpu`（`:730`）——CPU/NPU/HIP/XPU 后端目前**不管温度设置，一律走贪心比对核**，说明这几个后端还没有把 rejection sampling 的采样核移植过去,这是一条值得在第 8 节标注的功能缺口而非有意的设计取舍。

4. **7 种算法各有独立 worker 文件，而不是一个参数化的统一 worker。**
   - **为什么这么设计**：`FROZEN_KV_MTP` 的草稿"模型"只读目标模型的 KV、自己不持有 KV 池（`python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py:14`-`19` 模块文档），`DFLASH` 的验证是线性窗口而非树（`DFlashVerifyInput.topk` 恒为 1，`python/sglang/srt/speculative/dflash_info.py:24`-`33` 的注释直接写"DFLASH verify is linear (non-tree)"），`STANDALONE` 用一个完全独立、不共享 embedding/lm_head 的小模型（`StandaloneDraftWorker` 类文档，`python/sglang/srt/speculative/standalone_worker_v2.py:34`-`35`），`NGRAM` 干脆没有神经网络草稿模型、从历史文本语料表查 n-gram 续写（`python/sglang/srt/speculative/ngram_worker.py`）。这几种"草稿从哪来""KV 怎么管""验证是链还是树"的组合差异大到用 if/else 参数化会让单个文件承担太多分支；独立文件让每种算法的实现读起来是线性流程,不用在心里叠加"如果是这个算法就跳过这一段"。
   - **不这样会怎样**：一个统一 worker 要么把所有分支塞进同一批方法（`draft`/`verify`/`draft_extend`），可读性和可测试性都会下降；要么退化成"共享一个基类,每个算法重写大半个类",这其实就是现在的样子（`EagleDraftWorker`/`StandaloneDraftWorker`/`MultiLayerEagleWorkerV2` 都直接或间接继承共享基类,见 `python/sglang/srt/speculative/base_spec_worker.py:57`、`147`）——独立文件只是把"重写大半个类"这件事在物理文件层面显式化。
   - **什么时候可以不这样**：`StandaloneDraftWorker(EagleDraftWorker)`（`python/sglang/srt/speculative/standalone_worker_v2.py:34`）和 `MultiLayerEagleWorkerV2` 复用 `eagle_worker_common.py` 里的 `run_eagle_verify`/`build_eagle_verify_input` 就是"能共享就共享"的证据——真正独立成新文件的,是验证算法或 KV 管理方式发生根本变化的算法（DFLASH 的线性验证、FROZEN_KV_MTP 的只读 KV），不是随便一个变体都单开文件。

5. **自适应步数切换是"预建每档完整 `SpecRuntimeState`、整体替换引用"，而不是运行时连续调参。**
   - **为什么这么设计**：`num_steps` 一变,草稿/验证/补种三个阶段各自的 attention backend 元数据形状和 CUDA Graph 捕获的 batch 形状全部要跟着变——`AdaptiveController.init_states`（`python/sglang/srt/speculative/adaptive_runtime_state.py:94`-`111`）在启动时就为 `candidate_steps` 里的每个候选值都跑一遍 `build_adaptive_runtime_state`（`python/sglang/srt/speculative/eagle_worker_v2.py:1343`-`1419`,里面会重新构建 attention backend、重新捕获 CUDA Graph）,把结果全部缓存住;运行时切换只是 `_activate`（`python/sglang/srt/speculative/adaptive_runtime_state.py:128`-`134`）换一个字典查找,不涉及任何 GPU 端重建,代价是纳秒级的。
   - **不这样会怎样**：如果不预建、每次触发切换都现场重建 attention backend + 重新捕获 CUDA Graph,这个操作本身（`build_adaptive_runtime_state` 里同时有 `init_attention_backend`、`_capture_cuda_graphs`、`DecodeCudaGraphRunner` 构造）在真实 GPU 上通常是秒级到十秒级的开销——一次决策失误（比如噪声导致 EMA 短暂越过阈值又弹回来）就要为一次不必要的重建买单,这在需要低延迟响应的 decode 循环里是不可接受的。
   - **什么时候可以不这样**：预建的代价是显存——每个候选步数都要留一份自己的 CUDA Graph 显存池,`candidate_steps` 档位越多、`cuda_graph_bs` 覆盖的 batch size 越多,启动时预建的总显存开销越大;`adaptive_unsupported_reason`（`python/sglang/srt/speculative/adaptive_spec_params.py:50`-`87`）里列的一串不支持场景（DP attention、multi-layer eagle、TBO、PD-mux）某种程度上也是"这套预建-切换机制目前只覆盖得起最简单的部署形态"的证据——显存紧张或部署形态复杂时,直接不开 `--speculative-adaptive`（默认就是关的）就是"不这样"的选项。

## 6. 同位对照：vLLM 的投机解码

对照仓库是 `_src/vllm`（clone sha `7ca49fbe`；本篇取证基准仍是 `sglang @ 15a43983`，以下引用按要求带 `vllm:` 前缀，本篇不为 vLLM 声明独立取证基准头）。

- **草稿结构：SGLang 是动态剪枝的树，vLLM 主线代理目前是链。** `EagleProposer` 只是 `SpecDecodeBaseProposer` 的一个空子类（`vllm:vllm/v1/spec_decode/eagle.py:10`-`22`），真正的 `propose` 逻辑在 `SpecDecodeBaseProposer`（`vllm:vllm/v1/spec_decode/llm_base_proposer.py:510`）；这个通用 proposer 里有一条明确的 `FIXME`："when using tree-based specdec, adjust number of forward-passes according to the depth of the tree"（`vllm:vllm/v1/spec_decode/llm_base_proposer.py:1645`-`1646`）——即树形投机在这条主线路径上截至本篇取证时点还是待办项,不是已落地能力。vLLM 的 `RejectionSampler.rejection_sample` 输入签名里 `draft_token_ids.ndim == 1`（断言在 `vllm:vllm/v1/sample/rejection_sampler.py:413`）也印证了这一点：草稿 token 是攤平的一维序列,没有 `retrieve_index`/`retrieve_next_token`/`retrieve_next_sibling` 这类树结构索引。这不代表 vLLM 完全没有树相关工作——`vllm/v1/worker/gpu/spec_decode/` 下另有 `eagle/`、`dflash/`、`dspark/`、`mtp/` 等子目录和一套更新的 GPU worker 实现,与 SGLang 的算法命名高度重合,但那套实现的树支持程度本篇未继续深挖,留给 [[10-vLLM-投机解码]] 专篇核实。
- **算法命名高度重合，`DFLASH`/`DSPARK`/`Gemma4→MTP` 三个例子都能在两边独立找到对应文件。** vLLM 有一个专门的 `DFlashProposer(SpecDecodeBaseProposer)`（`vllm:vllm/v1/spec_decode/dflash.py:23`），文件名和类名都与 SGLang 的 `DFLASH`/`DFlashWorkerV2` 直接对应；vLLM 还有一个专门处理 Gemma4 assistant 草稿模型的 `gemma4.py`，模块文档写"Gemma4 MTP (Multi-Token Prediction) proposer... shares KV cache with the target model via cross-model KV sharing"（`vllm:vllm/v1/spec_decode/gemma4.py:3`-`8`）——这和 SGLang 的 `_resolve_speculative_algorithm_alias` 把 Gemma4 assistant 草稿自动提升成 `FROZEN_KV_MTP`（只读目标 KV，1.1 节表格已给出，`python/sglang/srt/arg_groups/speculative_hook.py:24`-`60`）是同一个模型架构特性在两个引擎里各自被识别、各自命名的结果。这条不是"谁抄谁"的证据（本库没有查证两边代码的时间先后或依赖关系），只说明"哪些投机解码变体值得单独实现"这件事，在独立的工程判断下收敛到了相近的答案。
- **验证判据：两边都是"贪心比对 vs rejection sampling 双路径",門控方式一致。** vLLM 的 `rejection_sample`（`vllm:vllm/v1/sample/rejection_sampler.py:394`）同样按 `sampling_metadata.all_greedy`（`vllm:vllm/v1/sample/rejection_sampler.py:435`）分流,全贪心走 `rejection_greedy_sample_kernel`（`vllm:vllm/v1/sample/rejection_sampler.py:454`-`467`,只在目标 argmax 与草稿 token 之间做等值比较）,否则走随机采样核。这条设计和 SGLang `eagle_sample` 里 `sampling_info.is_all_greedy` 的判据（`python/sglang/srt/speculative/eagle_utils.py:730`）几乎是同一个思路的两次独立实现——说明"贪心路径可以走比对而非采样"是投机解码这个问题本身的性质,不是某一个引擎的独创设计。
- **自适应机制：两边都有,但覆盖面不同,不是同一层级的对照。** SGLang 的 `AdaptiveController`（第 0、5、7 节）是按**批大小**分档调整**草稿步数**,只覆盖 EAGLE/EAGLE3（`adaptive_unsupported_reason`,`python/sglang/srt/speculative/adaptive_spec_params.py:54`-`58`）。vLLM 侧能找到对照物的是 `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`,但它的文档字符串写的是"Adaptive verification for DSpark speculative decoding"（`vllm:vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:3`）,`_assign_draft_token_budget`（`vllm:vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:34`-`60`）做的是**按逐 token 生存概率**（`confidence_probs` 的累乘）在全批次范围内重新分配验证预算,是"预算怎么在 token 间分配"层面的自适应,而不是"要不要开草稿"层面的自适应,且这条机制目前也只服务 vLLM 的 DSpark 实现（与 SGLang 的 `DSPARK` 算法同名但归属不同引擎）,不是 vLLM EAGLE 主线路径的能力——两边"都有自适应"这句话本身是真的,但适用范围、调节粒度、覆盖算法完全不是同一件事,不能直接类比成"谁的自适应更好"。

## 7. 踩坑与反直觉

1. **反直觉：高并发下投机解码常是负收益,SGLang 确实内置了自动关闭机制,但默认关闭且要满足一串前提。** 完整证据链：`--speculative-adaptive` 默认 `False`（`python/sglang/srt/server_args.py:2293`-`2297`）→ 打开后 `DEFAULT_ADAPTIVE_CONFIG` 里 `"64"` 档的 `candidate_steps` 只有 `[0]`（`python/sglang/srt/speculative/adaptive_spec_params.py:41`-`46`,意味着批大小达到 64 这一档时,EMA 调节器能选的步数只有 0,没有任何机会回到正步数,除非批大小掉回更低的档位）→ `num_steps=0` 时 `forward_batch_generation` 走 `_build_trivial_verify_input`（`python/sglang/srt/speculative/eagle_worker_v2.py:1206`-`1209`）→ 这个函数的文档字符串直接写"Used when `speculative_num_steps == 0` to skip drafting... functionally a plain decode"（`:1244`-`1250`）。但 `adaptive_unsupported_reason`（`python/sglang/srt/speculative/adaptive_spec_params.py:50`-`87`）同时要求：只支持 EAGLE/EAGLE3、`speculative_eagle_topk` 必须是 1、不能开 DP attention、不能开 multi-layer eagle、不能开 two-batch-overlap、不能开 pdmux——也就是说这套"高并发自动降级"机制,覆盖的其实是相对简单的部署形态,复杂部署形态下即使显式打开 `--speculative-adaptive` 也会在启动时被拒绝。
2. **`_v2` 文件名后缀是历史包袱,别读成"v1 兼容层还在"。** `spec_registry.py` 的类文档明确写"the spec V1 worker path has been removed, so such algorithms run on the V2 scheduler schema"（`python/sglang/srt/speculative/spec_registry.py:37`-`40`）,`create_worker` 分派表（`python/sglang/srt/speculative/spec_info.py:254`-`306`）返回的类清一色带 `V2` 后缀（`EAGLEWorkerV2`、`MultiLayerEagleWorkerV2`、`DFlashWorkerV2`、`FrozenKVMTPWorkerV2`、`StandaloneWorkerV2`）,但目录里已经找不到任何非 `_v2` 版本可供对比——`_v2` 现在纯粹是命名遗留,不影响任何运行时行为判断。
3. **KV 回收不是"逐轮 free",这一点最容易在读代码时想当然。** 直觉上"树里被拒绝的分支应该立刻把 KV 还回内存池",但实测证据链是：`_compact_accept_to_front` 的文档字符串自己承认"trailing unaccepted slots stay and are freed as overshoot"（`python/sglang/srt/speculative/eagle_worker_common.py:449`-`450`）——这里的"freed"指的是逻辑上不再被引用,不是立刻调用分配器的 free；真正调用 `allocator.free_segment` 归还物理槽位的 `_release_overallocated_kv_indices`（`python/sglang/srt/mem_cache/common.py:244`-`268`）只在请求结束的 `release_kv_cache`（`:198`-`241`）里触发。一个请求如果长期存活、每轮都有部分草稿被拒绝,这些"未提交但已分配"的物理页会一直挂在它的 `req_to_token` 行里,直到请求彻底结束才归还——不是内存泄漏（下一轮会覆盖写,数据不会累积增长）,但确实是"账面上看起来分配了但没用满"的常驻占用,如果拿 `nvidia-smi` 或显存统计工具去对"这个请求应该占多少 KV"做算术,不知道这条机制会算错。
4. **`topk=1` 时 `speculative_num_draft_tokens` 会被强制覆盖,命令行传的值不一定生效。** `python/sglang/srt/arg_groups/speculative_hook.py:675`-`683` 在 `speculative_eagle_topk == 1` 且用户传的 `speculative_num_draft_tokens != speculative_num_steps + 1` 时,直接打日志警告并覆盖成 `speculative_num_steps + 1`——这条和 [[09-SGLang-ServerArgs旋钮全景]] 第 0 节第 4 条讲的"有一类旋钮设了也白设"是同一类现象在投机解码子系统里的具体案例。
5. **DFLASH 虽然和 EAGLE 共享同一套 `speculative/` 目录、同一套 server_args 命名空间,但它的"验证"根本不是树。** `DFlashVerifyInput.topk` 字段的注释直接写"DFLASH verify is linear (non-tree)... always 1"（`python/sglang/srt/speculative/dflash_info.py:24`-`33`）——如果只看 `SpeculativeAlgorithm.is_eagle()` 之类的判定方法就以为"能过 `create_worker` 分派表这些算法都在做同一件事,只是模型架构不同",会漏掉这个根本性的结构差异（第 1 节和第 5 节第 4 条已经从"为什么独立成文件"的角度讲过这一点）。

## 8. 可改进点

1. **自适应机制目前只调"要不要展开、展开几步",没有联动 `topk`。** `adaptive_unsupported_reason` 强制要求 `speculative_eagle_topk in (None, 1)`（`python/sglang/srt/speculative/adaptive_spec_params.py:59`-`66`）,即自适应功能开启后树宽被锁死为 1（纯链）——EMA 只能在"链多长"这一个维度上调节,不能在"批大小上升时先收窄 topk 再收窄步数"这样更细粒度的代价曲线上调节。要支持 `topk` 联动,需要 `SpecRuntimeState` 预建的候选集从"一维步数列表"扩展成"`(steps, topk)` 二维网格",预建开销会显著增加,这也是为什么现状选择了只调一个维度——但这确实是一个没有被源码验证覆盖的假设,值得先用真实 profiling 数据验证"降 topk 比降 steps 更划算"是否成立,再决定要不要做。
2. **`DEFAULT_ADAPTIVE_CONFIG` 的四档边界（1/8/32/64）是硬编码常量,不随目标模型或实测接受率自动插值。** `python/sglang/srt/speculative/adaptive_spec_params.py:22`-`47` 这张表对所有模型、所有硬件都是同一组默认阈值;不同模型的"接受长度随批大小衰减曲线"形状不一样（大模型通常比小模型在同样并发下衰减更慢,因为目标模型前向本身更贵、投机解码的相对收益窗口更宽）,用同一组硬编码阈值大概率不是所有场景下都恰好合适。`speculative_adaptive_config` 已经支持传自定义 JSON 覆盖（`python/sglang/srt/server_args.py:2298`-`2302`）,缺的是一条从"实测接受率曲线"到"生成这份 JSON"的工具链,目前只能手工调。
3. **"预留窗口"里长期挂着的未提交 KV 页没有主动收缩机制。** 第 7 节第 3 条指出这些页只在请求结束时才归还;对于生成长度很长、且投机解码接受率长期偏低（比如输出进入了模型不擅长的领域,持续小步长命中）的请求,这个"应分配未提交"的常驻窗口会一直占着显存,直到整条生成结束。可以考虑在请求生命周期内定期（比如每 N 轮）主动做一次收缩判断——如果连续多轮 `num_correct_drafts` 都远低于 `speculative_num_steps`,主动把预留窗口收窄而不是保持满額预留。
4. **CPU/NPU/HIP/XPU 后端目前一律走贪心验证核,非贪心采样在这些后端上退化成什么行为需要专门核实。** 第 5 节第 3 条已经指出 `eagle_sample` 的分支条件里这四种硬件后端被并进了 `is_all_greedy` 分支（`python/sglang/srt/speculative/eagle_utils.py:730`）——这是功能缺口还是有意为之（比如这些后端上非贪心投机解码本来就不被支持,上层已经拦掉）,本篇没有找到明确的上游文档说明,值得单独查证并把结论文档化,否则容易被误读成"这些后端上温度采样被静默改成了贪心"。

## 9. 自测题与延伸阅读

**自测题**（闭卷）：

1. `topk=2, num_steps=3` 时候选池总大小是多少？如果 `num_steps` 改成 4、`topk` 不变,候选池大小怎么变？公式在哪一行？
2. 4.1 节的例子里,`a2` 分支在第 1 层的分数是 0.40,不算低,为什么最终一个 `a2` 的孩子都没能继续展开？这说明"树"和"topk 条独立链"在预算分配上的本质区别是什么？
3. 一个草稿树节点的 `positions`（位置编码）是由什么决定的？为什么两个不同父节点、不同 token 的兄弟节点可能拿到相同的 position id？这对 RoPE 为什么是正确的？
4. `verify_tree_greedy_kernel_triton` 在某一层如果目标模型的 argmax 预测不匹配任何一个候选孩子,会发生什么？这一步之后紧跟着的"bonus token"是怎么来的,为什么它不需要再验证？
5. 为什么"接受路径的 KV 需要从散列位置搬到连续区间"？如果不搬,下一轮 decode 的 KV 读取逻辑会遇到什么问题？
6. `speculative_num_steps=0` 时,`_build_trivial_verify_input` 构造的是一棵几个节点的树？它和"完全不经过投机解码 worker、走普通解码路径"在功能上等价吗？在实现路径上等价吗？
7. `--speculative-adaptive` 默认是开还是关？打开后,批大小达到 64（用默认配置）时,`num_steps` 会被调到多少？这和"投机解码在高并发下常是负收益"这个说法是什么关系？
8. `topk=1` 时,`speculative_num_draft_tokens` 会被强制设成什么？如果命令行显式传了一个不同的值,会发生什么？
9. DFLASH 和 EAGLE 都在 `speculative/` 目录下、都能通过 `SpeculativeAlgorithm.is_eagle()` 判定方法族群相关判断,但它们在"验证是不是树结构"这一点上有什么根本区别？

**延伸阅读**（本库双链）：

- [[10-vLLM-投机解码]]——本篇 `## 6` 只核对了 vLLM 主线 `EagleProposer` 的链式结构和 `rejection_sampler.py` 的贪心/随机双路径,`vllm/v1/worker/gpu/spec_decode/` 下 `eagle/`、`dflash/`、`dspark/`、`mtp/` 那套更新的 GPU worker 实现（含树支持程度）留给这一篇专门核实。
- [[04-SGLang-内存池与KV布局]]——本篇 `## 4.4`、`## 5` 第 2 条讲的"预留窗口原地覆盖"建立在 `req_to_token_pool`/`token_to_kv_pool_allocator` 的物理布局之上,这些底层结构的完整解剖在这一篇。
- [[09-SGLang-ServerArgs旋钮全景]]——本篇涉及的 `speculative_num_steps`/`speculative_eagle_topk`/`speculative_num_draft_tokens`/`speculative_adaptive` 等旋钮只挑了投机解码相关的部分讲,该篇覆盖全部 476 个字段的通用解析机制和"设了也白设"现象的完整清单。

