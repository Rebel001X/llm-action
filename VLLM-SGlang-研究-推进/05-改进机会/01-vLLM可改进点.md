# vLLM 可改进点全景

> **本篇取证基准**：`vllm` @ `7ca49fbe`（2026-08-22）
> **一句话**：鉴权默认放行最致命,其余是可见性负债

## 0. 结论先行

- 本篇不是"vLLM 代码质量差"的报告——机器信号本身就说明相反的事：静默吞异常的热路径密度只有 1.9 / 万行热路径代码,是本库扫过的 12 个引擎里**第二低**（仅次于代码规模小两个数量级的 `tokasaurus`），`TODO` 认领率 34%（205/603）也是本库见过最高的水平。这是一份"在一个总体扎实的工程里,哪几个具体地方值得动刀"的清单,不是拆台报告。
- 全篇给出 **17 条改进点**（分布在鉴权 / 可观测性 / 配置一致性 / 死引用 / 文档同步 / 调度器 / 投机解码 / 注意力后端 / API 表面八个子系统）,以及 **6 条"读机器信号读到了但不建议改"** 的条目——这六条是本篇诚实度的自查项,来源分别是可选依赖兜底、探测型文件的静默降级、非阻塞轮询里的正常控制流、机器信号自身对一处间接日志的误标,以及一处"规模大不等于该拆分"的判断。
- 最重的一条是**鉴权模型**：`--api-key` 只保护 `GUARDED_PREFIX` 白名单前缀,不在白名单里的路径（`/invocations`、`/sleep`、`/collective_rpc` 等）配了 key 也完全不设防,其中 `/collective_rpc` 能在 worker 进程上以 `getattr` 方式派发任意方法名。**这是上游自己在 `docs/usage/security.md` 里写清楚的已知限制,不是本库的新发现**——本篇要做的是把"上游已经写在文档里"和"代码层面默认方向仍然是反的"这两件事分开说清楚。
- 823 处 `raise NotImplementedError` 是全场最多,但机械识别（形式化 `ABC`/`Protocol` 基类 + `@abstractmethod` 装饰器 + "必须被子类实现"类消息）能确认至少 40% 是接口占位；对余下部分做人工抽样复核后,真实占位比例估计落在 45%~50% 区间——**这 823 这个数字本身不能当"vLLM 里有 823 处没写完的代码"来读**,详见 `## 8` 的"不建议改"第一条。

## 1. 它在系统里的位置

本篇是 `05-改进机会/` 目录的第一篇,坐标上承接 `01-vLLM/` 全部十篇已完成正文,横向对齐 `_lab/out/improve.json` 的机器信号产物。三类原材料的分工：

- **机器信号**（`_lab/out/improve.json` 的 `engines.vllm`）负责"在哪扫、扫出多少个"——它给的是密度和候选位置,不是结论,必须逐条打开源码复核,`## 7` 会展开至少一处机器信号本身出错的例子。
- **兄弟篇的 `## 8` 段落**（`01-vLLM/01` 到 `01-vLLM/10`）负责"读代码时已经想清楚的具体缺口"——这十篇每篇末尾都有一段"可改进点",本篇的任务是把这十段**收集、复核、按主题重组**,而不是重新读一遍全部源码。
- **本篇自己的复核**负责"把前两类原材料的每一条断言都重新打开一次源码"——本篇引用的每个 `文件:行` 都经过本篇作者独立 `grep`/`Read` 核实,不是照抄兄弟篇的行号（个别行号因为兄弟篇写作时间早于本篇,存在 1~2 行的漂移,本篇一律以自己重新核实的行号为准）。

下游消费者是 `05-改进机会/03-跨引擎共性缺口.md`（⬜ 未写）——那篇要把本篇和未来的 `02-SGLang可改进点.md` 做交集,挑出"两个引擎都有的同类问题",本篇不做这一步。

## 2. 代码地图（文件 → 职责，带行号）

| 文件:行 | 职责 |
|---|---|
| `vllm/entrypoints/serve/middleware/authenticate.py:11` | `GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")`——鉴权白名单前缀元组 |
| `vllm/entrypoints/serve/middleware/authenticate.py:59` | `if url_path.startswith(GUARDED_PREFIX) and not self.verify_token(headers):`——只有命中白名单前缀才校验 token |
| `vllm/entrypoints/serve/dev/rpc/api_router.py:23` | `@router.post("/collective_rpc")`——不在白名单前缀下的高危路由 |
| `vllm/v1/executor/multiproc_executor.py:1038` | `func = getattr(self.worker, method)`——`method` 来自 HTTP 请求体,worker 进程内按字符串反射派发 |
| `docs/usage/security.md:231` | 上游文档原话："Execute arbitrary RPC methods on the engine (extremely dangerous)" |
| `vllm/compilation/cuda_graph.py:238` | `CUDAGraphWrapper.__call__` 第一处静默直转发：无 forward context 时直接跑 `self.runnable(*args, **kwargs)` |
| `vllm/compilation/cuda_graph.py:254` | 第二处静默直转发：`cudagraph_runtime_mode` 为 `NONE` 或与本 wrapper 模式不匹配时同样直接转发,不打日志 |
| `vllm/v1/worker/gpu_model_runner.py:4105` | `dispatch_cudagraph(num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output)`——cascade attention / encoder 输出强制关闭 FULL 图 |
| `vllm/config/observability.py:68` | `cudagraph_metrics: bool = False`——CUDA Graph 统计表默认关闭 |
| `vllm/config/vllm.py:1387` | `enforce_eager=True` 覆盖 `compilation_config.mode`/`cudagraph_mode`（这条**有** `warning_once`） |
| `vllm/sampling_params.py:529` | `temperature < _SAMPLING_EPS` 时静默重置 `top_p`/`top_k`/`min_p`（**无任何日志**） |
| `vllm/v1/attention/backends/registry.py:112` | `NO_ATTENTION = "vllm.v1.attention.backends.no_attention.NoAttentionBackend"`——指向一个不存在的模块 |
| `vllm/entrypoints/openai/api_server.py:24` | `warnings.warn("... is deprecated ...")`——59 行文件本身就是一个弃用 shim |
| `vllm/v1/core/sched/scheduler.py:652` | `preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))`——PRIORITY 抢占的 O(n) 扫描 |
| `vllm/v1/core/sched/scheduler.py:964`-`971` | `enable_chunked_prefill=False` 时,队首请求排不下直接 `break`,不尝试后面更短的请求 |
| `vllm/v1/spec_decode/metrics.py:41` | `def observe_draft(...)`——采集接受率但没有下游读取者 |
| `vllm/config/speculative.py:976`-`977` | `hf_config.model_type == "mlp_speculator"` 时自动把 `method` 设成这个值 |
| `vllm/v1/worker/gpu_model_runner.py:701` | `self.drafter` 分发链的兜底分支,`mlp_speculator` 会落到这里报 `Unknown speculative decoding method` |
| `vllm/v1/worker/gpu_input_batch.py:708` | `def condense(self) -> None:`——十几个并行数组必须手工同步搬移的压缩逻辑 |
| `rust/src/server/src/routes.rs:83`-`94` | Rust 前端路由表,与 Python 侧 `api_router.py` 各自独立维护 |
| `vllm/entrypoints/scale_out/token_in_token_out/api_router.py:79` | 第一处 `/abort_requests` 注册（`--tokens-only` 时挂载） |
| `vllm/entrypoints/serve/dev/rlhf/api_router.py:94` | 第二处 `/abort_requests` 注册（`VLLM_SERVER_DEV_MODE=1` 时挂载） |
| `vllm/v1/attention/backend.py:263` | `validate_configuration()` 定义处,只在 `Attention` 层真正构建时被调用 |
| `vllm/platforms/cuda.py:385` | `validate_configuration()` 的一处调用点,发生在模型构建阶段而非 CLI 解析阶段 |
| `vllm/platforms/cuda.py:505`-`519` | `get_supported_vit_attn_backends()`——手写的 ViT backend 候选列表,与 `AttentionBackendEnum` 注册各自独立 |
| `vllm/entrypoints/generate/api_router.py:75` | `if envs.VLLM_ENABLE_COHERE_API:`——`/cohere/v2/chat` 唯一的门控条件 |
| `vllm/entrypoints/pooling/embed/api_router.py:47` | `/v2/embed` 无条件注册,不受 `VLLM_ENABLE_COHERE_API` 影响 |
| `vllm/entrypoints/pooling/scoring/api_router.py:106` | `/v2/rerank` 同样无条件注册 |
| `vllm/v1/worker/gpu_model_runner.py:501` | `class GPUModelRunner(...)`——全场最大 god_file 的主体,一个类贯穿到文件末尾 |
| `vllm/entrypoints/openai/cli_args.py:140`-`141` | `--enable-tokenizer-info-endpoint`——`/tokenizer_info` 的独立门控旗标 |
| `vllm/entrypoints/serve/dev/server_info/api_router.py:43` | `/server_info` 无条件注册,风险等级由外层 `VLLM_SERVER_DEV_MODE` 判断兜底 |
| `vllm/entrypoints/launchers/api_server/routers.py:34`-`37` | dev 族路由（含 `/server_info`、`/collective_rpc`、`/sleep`）统一挂在 `VLLM_SERVER_DEV_MODE` 下,粒度过粗 |

## 3. 核心数据结构

本篇没有自己的运行时数据结构,但依赖两套"记录格式"，读懂它们是读懂全篇的前提。

**一、`improve.json` 的产物 schema**（`_lab/out/improve.json` → `engines.vllm`）：

```
{
  "ref": {sha, sha_short, commit_date, ...},   # 取证基准,写正文前必须核对
  "summary": {                                  # 全仓聚合数字,写数字前从这里抄
    "silent_except": 449, "silent_except_by_role": {"hot":88,"other":158,"probe":203},
    "silent_except_hot_per_10k_hot_lines": 1.9,  # 真正该看的口径,不是 silent_except_per_10k_lines
    "todo": 603, "todo_owned": 205, "not_impl": 823, "god_file": 7, ...
  },
  "god_files": [...],           # >=3000 行的文件,带 line:1 起点
  "hotspots_silent_except": [...],           # 全仓口径的密度榜(会被探测型文件带偏)
  "hotspots_silent_except_hot_path": [...],  # 热路径口径的密度榜(更可信)
  "samples": {bare_except, deprecated, god_file, not_impl, silent_except, todo},  # 每类的样例,带 file/line
  "samples_silent_except_hot_path": [...]    # 热路径样例,role 都标 "hot"
}
```

`role` 字段（`probe`/`hot`/`other`）是这份产物里最关键的一列——`caveat` 字段原话："热点前几名全是 `check_env`/`collect_env`/`platforms` 这类**探测型**文件,那里吞异常本来就正当。"`## 5`/`## 7` 会展开为什么这条 caveat 必须当真。

**二、本篇 `## 8` 每条改进点的记录格式**（任务规定的六字段模板）：`现状`（带 `文件:行` 的事实）→ `问题`（具体失败场景）→ `证据等级`（源码为证 / 本库推断）→ `改进方向`（具体到函数）→ `代价与反驳`（不许空）→ `难度`。这个格式本身是一种强制——`代价与反驳` 字段不许空,逼着每条建议回答"上游为什么可能故意不这么做",这是本篇和一般"发现问题就列出来"式清单的区别。

## 4. 主流程走读

本篇的写作过程是一条三级过滤流水线,不是"读一遍代码列问题"：

1. **收集候选**：读 `_lab/out/improve.json` 的 `summary`/`god_files`/`hotspots_*`/`samples*` 六个字段,记下每条带 `file:line` 的信号；再逐篇打开 `01-vLLM/01` 到 `01-vLLM/10` 的 `## 8` 段落,摘出全部具体断言（约 30 条原始候选）。
2. **复核**：对每条候选重新 `grep -n` 定位、`Read` 打开上下文——这一步淘汰了不少候选：比如兄弟篇 09 篇提到 `enforce_eager` 覆盖"无条件"发生,复核发现它其实**有** `logger.warning_once`（`vllm/config/vllm.py:1387`-`1393`），"无条件覆盖"和"无日志覆盖"是两件事,不能混为一谈,本篇按复核结果重新措辞（详见 `## 7` 第 3 条）。
3. **交叉验证机器信号与人工断言**：比如 `/collective_rpc` 的"能执行任意方法"这句话,不满足于兄弟篇的转述,而是亲自追到 `vllm/v1/executor/multiproc_executor.py:1038` 的 `func = getattr(self.worker, method)` 和 `vllm/v1/executor/multiproc_executor.py:415`-`420` 的 `send_method = method` 入队逻辑,确认这条链路的每一跳都成立。
4. **给"不建议改"留位置**：凡是机器信号里出现、但复核后发现是正当设计的候选（探测型文件的 `except ImportError: pass`、轮询循环里的 `queue.Empty`），不是简单丢弃,而是移进 `## 8` 的"不建议改"小节——这一步是为了让"信号被过滤掉了"这件事本身可见,而不是让读者以为本篇没扫到。
5. **归类与去重**：把幸存的候选按子系统（鉴权/可观测性/配置/调度器/投机解码/注意力后端/API 表面）分组,同一个根因（比如"覆盖类日志级别不一致"）出现在多个候选里的合并成一条,最终收敛到 17 条正式改进点。

### 4.1 一个完整案例：`/collective_rpc` 是怎么被复核出来的

`## 8.1` 是全篇最重的一条,值得把复核过程完整走一遍,作为"三级过滤流水线"具体长什么样的例子（其余 16 条走的是同一套流程,不再逐条重复展开）：

1. **起点**：`01-vLLM/08-vLLM-HTTP-API表面全解.md` 的 `## 8` 提到"`AuthenticationMiddleware` 的 `GUARDED_PREFIX` 白名单模型值得改成默认保护、显式豁免",并点名了 `/invocations`、`/generative_scoring`、`/pooling`、`/classify`、`/score`、`/rerank` 这几个具体端点。
2. **第一跳**：`grep -n "GUARDED_PREFIX" --include=*.py vllm/` 定位到唯一定义点 `vllm/entrypoints/serve/middleware/authenticate.py:11`,打开整个文件核实判断逻辑在 `:59`,确认"只有命中白名单前缀才校验 token"这句话成立。
3. **第二跳**：既然白名单外的端点不设防,哪个端点风险最高？没有直接沿用兄弟篇列出的六个,而是回到 `docs/usage/security.md` 通读一遍上游自己的分类——发现"Development endpoints"一节里 `/collective_rpc` 被上游自己标成"extremely dangerous"（`docs/usage/security.md:231`）,风险等级明显高于普通的推理端点,于是把这条作为本篇的重点案例,而不是简单照抄兄弟篇的六个端点列表。
4. **第三跳**：`/collective_rpc` 具体"危险"在哪？打开 `vllm/entrypoints/serve/dev/rpc/api_router.py`,确认处理函数把请求体的 `method`/`args`/`kwargs` 原样转发给 `engine_client(raw_request).collective_rpc(...)`——但这一步还只是"HTTP 层把参数转发出去了",不足以证明"能执行任意方法"这句话。
5. **第四跳**：顺着 `collective_rpc` 调用链往下追（`vllm/engine/protocol.py` → `vllm/v1/engine/async_llm.py` → `vllm/v1/engine/core.py` → `vllm/v1/executor/multiproc_executor.py`）,在 `vllm/v1/executor/multiproc_executor.py:1038` 找到 `worker_busy_loop` 里的 `func = getattr(self.worker, method)`——这一行才是"字符串 `method` 变成真正函数调用"的证据,链路走完,`## 8.1` 的"现状"字段才算真正立住。
6. **收尾**：核实"上游是否知道"——再读一遍 `docs/usage/security.md:147`-`180` 关于 `--api-key` 限制的完整说明,确认这不是本库的新发现,措辞上明确区分"上游已经写清楚"和"代码默认值方向仍然是反的"这两件事,避免把一个文档化的已知限制包装成"重大发现"。

这个过程说明本篇的"复核"不是简单地把兄弟篇的一句话配上一个 `grep` 结果,而是要**沿着调用链走到底**,直到找到能支撑"现状"字段里那句具体断言的最后一跳代码。

### 4.2 复现命令清单

本篇每条"现状"都可以用一条 `grep -n` 在 `_src/vllm`（取证基准 `7ca49fbe`）里独立复现,不需要相信本篇的转述。按 `## 8` 编号列出关键命令：

```
# 8.1 鉴权白名单与 collective_rpc 派发
grep -n "GUARDED_PREFIX" vllm/entrypoints/serve/middleware/authenticate.py
grep -n "getattr(self.worker, method)" vllm/v1/executor/multiproc_executor.py

# 8.2 CUDA Graph 静默回退
grep -n "return self.runnable(\*args, \*\*kwargs)" vllm/compilation/cuda_graph.py
grep -n "cudagraph_metrics: bool" vllm/config/observability.py

# 8.3 配置静默覆盖清单
grep -n "compilation_config.mode = CompilationMode.NONE" vllm/config/vllm.py
grep -n "self.top_p = 1.0" vllm/sampling_params.py

# 8.4 NO_ATTENTION 死引用
grep -n "NO_ATTENTION" vllm/v1/attention/backends/registry.py
find vllm/v1/attention/backends -iname "no_attention*"   # 预期：空结果

# 8.9 mlp_speculator 死配置
grep -n "mlp_speculator" vllm/config/speculative.py
grep -n "Unknown speculative decoding method" vllm/v1/worker/gpu_model_runner.py

# 8.16 Cohere 端点门控不一致
grep -n "VLLM_ENABLE_COHERE_API" vllm/entrypoints/generate/api_router.py
grep -n "VLLM_ENABLE_COHERE_API" vllm/entrypoints/pooling/embed/api_router.py   # 预期：空结果
```

最后一组命令(`8.16`)的"预期：空结果"是本篇断言"`/v2/embed` 不受该环境变量影响"的直接证据——如果这条命令哪天不再返回空结果,说明上游已经把这两个端点接入了同一套门控,`## 8.16` 这条改进点就该标记为已解决。

## 5. 设计决策与代价

本篇本身的三条方法论决策,同样按"为什么这么设计 / 不这样会怎样 / 什么时候可以不这样"展开：

**决策 1：用"热路径口径"而不是"全仓密度"作为静默异常的主要判据**

- 为什么这么设计：`improve.json` 的 `caveat` 字段已经说明,全仓密度会被 `check_env`/`collect_env`/`platforms` 这类探测型文件带偏——按全仓密度排序,最靠前的会是这些"吞异常完全正当"的文件,读者会被误导去"修"根本不需要修的地方。热路径口径只统计被判定为热路径角色的文件,更接近"这个静默异常真的会在服务运行时悄悄发生"这件事。
- 不这样会怎样：直接按全仓密度写,`## 8` 的候选池会被 `vllm/platforms/cuda.py`（10 处）、`vllm/_aiter_ops.py`（8 处）这类文件塞满,而这些恰恰是 `## 8` "不建议改" C 条已经核实为正当设计的文件——等于把体检报告的篇幅浪费在健康指标上。
- 什么时候可以不这样：如果任务目标是审计所有异常处理的代码风格一致性,而不是找运行时会造成实际影响的隐患,全仓密度反而是更合适的口径——本篇的选择服务于"哪里改了有实际收益"这个目标,不是普适真理。

**决策 2："不建议改"是硬性要求,不是锦上添花**

- 为什么这么设计：一份只列"问题"、不列"扫到但判断不是问题"的清单,读者没有办法分辨作者是"认真核实过、判断这里没问题",还是"根本没扫到这里"。强制写"不建议改"且要求给出具体理由,是让"过滤"这个动作本身可审计。
- 不这样会怎样：读者面对 823 处 `NotImplementedError`、449 处静默异常这类大数字,容易产生"这个代码库到处是问题"的错误印象,尤其这些数字经常被不熟悉代码库的人直接拿来做引擎间的横向比较,而不去看数字背后的构成。
- 什么时候可以不这样：如果目标读者是 vLLM 核心维护者(对设计意图有先验知识,不需要被提醒某处 `except` 是正当的),这一节的边际价值会降低——但对本库的目标读者(通过读源码理解引擎设计的学习者)而言,这一节几乎是全篇最重要的部分。

**决策 3：改进点模板强制"代价与反驳"不许空**

- 为什么这么设计：单纯罗列"这里可以改进"是廉价的——几乎任何一行代码都能找到"理论上可以更好"的角度,真正有价值的是判断"为什么现在没人这么改",这决定了一条建议是"举手之劳"还是"需要先说服团队接受一个权衡"。
- 不这样会怎样：一份只谈"怎么改"不谈"改了牺牲什么"的清单,读起来像是在说"上游没想到这一点",但本篇核实的多条(尤其 `## 8.1` 的鉴权模型)恰恰是上游明确知道并写了文档的已知取舍,不写代价等于误导读者以为这是维护者的疏忽。
- 什么时候可以不这样：对于纯粹的死引用类问题(比如 `## 8.4` 的 `NO_ATTENTION`),代价确实很小,这一栏会显得单薄——但即使单薄也要写(比如指出加测试可能在特定硬件 CI 上引入假失败),完全没有代价的改动在本篇里反而值得怀疑是不是遗漏了什么。

## 6. 同位对照

不同引擎在同一套 `_lab/out/improve.json` 口径下的对比(数字直接抄自各引擎 `summary` 字段;取证基准以各引擎自己的 `ref.sha` 为准,本篇不重复声明其余引擎的 sha)：

| 引擎 | 热路径静默异常密度(/万行) | `TODO` 认领率 | `NotImplementedError` 计数 | god_file 数 | Python 行数 |
|---|---|---|---|---|---|
| `tokasaurus` | 0.0 | 0/14（0%） | 0 | 0 | 9,860 |
| `lmdeploy` | 1.2 | 4/48（8.3%） | 250 | 0 | 137,796 |
| **`vllm`** | **1.9** | **205/603（34.0%）** | **823** | **7** | **827,314** |
| `sglang` | 4.9 | 167/706（23.7%） | 775 | 23 | 1,143,185 |
| `tensorrt-llm` | 5.7 | 49/418（11.7%） | 443 | 25 | 685,891 |
| `lightllm` | 8.0 | 0/33（0%） | 69 | 0 | 102,485 |
| `mlc-llm` | 8.6 | 6/16（37.5%） | 19 | 0 | 58,643 |
| `ktransformers` | 9.5 | 0/166（0%） | 50 | 0 | 94,658 |
| `tgi` | 11.5 | 14/145（9.7%） | 144 | 0 | 94,284 |
| `mooncake` | 15.5 | 0/2（0%） | 0 | 0 | 1,291 |
| `dynamo` | 21.8 | 20/75（26.7%） | 46 | 2 | 141,124 |

三点读数字时要注意的地方：

- **vLLM 的热路径静默异常密度(1.9)是除 `tokasaurus`(代码规模只有 vLLM 的 1.2%,样本小到不适合比较)之外最低的**——`## 0` 已经强调过,这里再次确认不是本篇选择性展示的结果。
- **`todo_owned` 比例上 `mlc-llm`(37.5%)名义高于 vLLM(34.0%)**,但 `mlc-llm` 的 `TODO` 总数只有 16 条,分母太小,单条认领与否就能让比例大幅波动;按绝对认领数量看,vLLM 的 205 条比第二名 `sglang` 的 167 条多出近四成,且是在 `TODO` 总数(603)本身也不小的情况下做到的——这个对比支持"vLLM 的 `TODO` 治理在样本量有意义的引擎里领先"这个更谨慎的表述,而不是简单说"认领率全场最高"。
- **`god_file` 数量上 vLLM(7)远少于 SGLang(23)和 TensorRT-LLM(25)**,但不能反过来读成"vLLM 代码组织比这两者更健康"——`sglang`/`tensorrt-llm` 的 Python 总行数分别是 vLLM 的 1.38 倍和 0.83 倍,规模差异本身就会影响超大文件出现的概率,`## 7` 会再展开这一点。
- **`NotImplementedError` 计数上 vLLM(823)和 SGLang(775)量级接近,TensorRT-LLM(443)明显更少**——但这个数字本身混合了"抽象占位"和"具体不支持的组合报错"两类性质完全不同的东西(`## 8` "不建议改" A 条已经拆解 vLLM 自己的构成),三个引擎的 823/775/443 谁的"真正待办"更多,不做同样的人工抽样复核就没法比较,本篇没有对 SGLang/TensorRT-LLM 做这项工作,这里只呈现原始计数,不做跨引擎的"谁更完善"排名。

## 7. 踩坑与反直觉

1. **机器信号会把"间接记录日志"误标成"完全没记录"**——`## 8` "不建议改" E 条是最具体的例子：`vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:415` 的 `except Exception` 块实际通过调用 `_fail_task` 把错误信息传给了 `logger.error`,但静态扫描器只检查 `except` 代码块字面上是否直接出现日志调用,看不穿这层函数调用,于是打上了 `no-raise-no-log` 的标签。教训是：`improve.json` 里任何一条 `swallow` 标注,写进正文前都必须去看它调用的辅助函数的完整实现,不能只信字段本身。
2. **"静默覆盖"这个词本身容易被滥用,`enforce_eager` 是一个反例**——早期收集候选时曾把 `enforce_eager=True` 改写 `compilation_config` 的行为直接归入"无条件静默覆盖",本篇复核 `vllm/config/vllm.py:1387`-`1393` 后确认这里**有** `logger.warning_once`。"无条件"（不问用户是否显式设置过）和"完全静默"（没有任何提示）是两个独立的维度,前者成立不代表后者也成立——`## 8.3` 按这个区分重新措辞,避免读者以为这条和 `SamplingParams` 那条(真正的零日志)是同一严重程度。
3. **热路径静默异常的"热"不等于"高频"**——`role=hot` 的判定依据是文件所在子系统(比如整个 KV 连接器目录默认按热路径处理),不是逐个函数做调用频率分析。KV 连接器目录里既有真正每次前向都可能触碰的路径,也有像 `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:82` 这种只在进程启动时执行一次的可选依赖 import 守卫（`except Exception: HF3FS_AVAILABLE = False`）——按文件粒度打的 `role` 标签,不能替代"这一行具体多久执行一次"的判断。
4. **god_file 数量的引擎间对比容易掉进规模谬误**——`## 6` 已经展开,这里补一句更直接的：即使按"每万行 Python 代码有多少个 god_file"归一化重算（vLLM 约 0.085 个/万行,SGLang 约 0.20 个/万行,TensorRT-LLM 约 0.36 个/万行）,vLLM 依然显著更低,但本篇没有进一步核实这是"vLLM 团队更严格地拆分大文件"还是"vLLM 覆盖的算子/模型种类恰好没有强烈催生超大文件的场景"。**未查证**：两种解释哪个（或两者共同）成立,本库没有做进一步的因果分析,只呈现归一化后的数字本身依然支持"vLLM 相对更低"这个观察。
5. **兄弟篇的行号会随时间漂移,写作时必须重新核实,不能直接照抄**——`## 4` 第 2 步已经提到 `enforce_eager` 那处的措辞辨析,行号本身也有同类问题：本篇初稿收集候选阶段沿用了兄弟篇给出的 `vllm/v1/core/sched/scheduler.py:651`-`655`（PRIORITY 抢占逻辑）,本篇重新 `grep` 定位后确认实际行号是 `:652` 起（`## 8.6`）,一行之差；类似地 WAITING 循环的 `break` 逻辑从兄弟篇给出的 `:966`-`972` 核实为 `:964`-`971`（`## 8.7`）。差距都不大,但足以说明"引用兄弟篇的断言"和"引用兄弟篇给出的具体行号"必须分开处理——前者可以复用,后者必须重新核实,这也是本篇所有 `01-vLLM/*` 子系统相关的行号都经过本篇作者独立复核、而不是直接复制的原因。

## 8. 可改进点

以下 17 条按子系统分组排序,重要性大致递减；每条固定六字段。全部标注为**源码为证**的断言均已由本篇作者亲自 `grep`/`Read` 复核,行号以本篇复核结果为准。

### 8.1 `--api-key` 白名单是"默认放行",`/collective_rpc` 能在 worker 上派发任意方法

- **现状**：`vllm/entrypoints/serve/middleware/authenticate.py:11` 定义 `GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")`,`:59` 的 `if url_path.startswith(GUARDED_PREFIX) and not self.verify_token(headers):` 只在路径命中这四个前缀时才校验 `Authorization` 头。`vllm/entrypoints/serve/dev/rpc/api_router.py:23` 注册的 `/collective_rpc` 不在白名单里,处理函数把请求体里的 `method`（字符串）、`args`、`kwargs` 原样转发给 `engine_client(...).collective_rpc(...)`；链路最终落到 `vllm/v1/executor/multiproc_executor.py:1038` 的 `func = getattr(self.worker, method)`,worker 进程按字符串反射取出方法直接调用。
- **问题**：只要 `VLLM_SERVER_DEV_MODE=1`（该路由本身挂在这个开关下）且网络可达,攻击者不需要任何 `--api-key` 就能让 worker 进程调用 `self.worker` 上的任意可调用属性；同一个端口上还挂着一批完全不设防的操作端点（`/invocations`、`/sleep`、`/scale_elastic_ep` 等,`docs/usage/security.md:179`-`231` 完整列出）,一个以为设了 `--api-key` 就万事大吉的运维很容易漏防。
- **证据等级**：源码为证（鉴权白名单逻辑、`/collective_rpc` 的 `getattr` 派发链均已逐跳打开源码核实）；"上游已知并有详尽文档"这一判断依据 `docs/usage/security.md:147`-`231`,同样是文档为证,不是本库猜测。
- **改进方向**：把 `AuthenticationMiddleware.__call__`（`vllm/entrypoints/serve/middleware/authenticate.py:47`）的判断方向反过来——默认所有非显式豁免路径都需要 `--api-key`,只给 `/health`、`/ping` 这类真正无害的路径开白名单,`GUARDED_PREFIX` 改名成 `EXEMPT_PREFIX` 并反转判断逻辑,是这条改动的核心。
- **代价与反驳**：这是一个破坏性变更——任何依赖"非 `/v1` 路径默认不需要 key"这个当前行为的现有部署,升级后都会突然要求带 token,可能造成生产环境的静默中断。上游选择"文档警告 + 保持向后兼容"而不是"改默认值",大概率就是权衡了这一点；`docs/usage/security.md` 的推荐做法直接是"部署在反向代理后面,只放行你要暴露的端点",说明上游把解法定位在部署层而不是默认值层,是经过取舍的选择,不是疏忽。
- **难度**：中等（改动本身是把一个布尔判断取反,量级很小；成本在于评估现有部署的兼容性影响并给出迁移期）。

### 8.2 CUDA Graph 静默回退到 eager,观测默认关闭

- **现状**：`vllm/compilation/cuda_graph.py:238` 和 `:254` 两处 `CUDAGraphWrapper.__call__` 分支,分别在"无 forward context"和"运行时模式不匹配"时直接 `return self.runnable(*args, **kwargs)`,不打任何日志；`vllm/v1/worker/gpu_model_runner.py:4105` 的 `dispatch_cudagraph(num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output)` 会在 cascade attention 或有 encoder 输出时强制不让本步走 FULL 图；唯一能看见这些回退的入口 `cudagraph_metrics` 默认是 `False`（`vllm/config/observability.py:68`）。
- **问题**：一个显式配置了 `cudagraph_mode=FULL` 的用户,如果请求里混入触发 cascade attention 的长共享前缀 batch,这一步会静默退化成 eager,日志里没有任何提示,只能凭"感觉变慢了"去怀疑,再手动打开 `--cudagraph-metrics` 才能证实。
- **证据等级**：源码为证。
- **改进方向**：在 `CUDAGraphWrapper.__call__` 的两处静默转发分支里,补一条基于 `logger.warning_once`(以 `(reason, batch_descriptor)` 做 key,避免刷屏)的低频提示；或者把 `cudagraph_stats` 的采集从"仅在 `cudagraph_metrics=True` 时构造"改成默认按较低采样率构造并聚合,只把"打印这一步"留给开关控制。
- **代价与反驳**：`CUDAGraphWrapper.__call__` 是真正的热路径,即使一次 `warning_once` 判断也要考虑极高 QPS 下的开销；更现实的顾虑是误报疲劳——warm-up、profile run 阶段本来就会大量走这条分支,如果不精确排除这些阶段,新用户第一次启动就会看到一堆警告。上游默认关闭 `cudagraph_metrics` 未必是没想到,也可能是判断"多数用户不需要这层信息,默认噪音大于收益"。
- **难度**：小改（`warning_once` 版本）；采样统计版本需要先厘清 warm-up 阶段如何排除,升级为中等。

### 8.3 配置静默覆盖的日志级别不统一,一处完全无日志

- **现状**：至少 7 处"用户显式设置的字段被下游 `__post_init__` 无条件改写"的例子分布在 4 个文件,可见度并不统一——`enforce_eager` 联动关闭编译走 `logger.warning_once`（`vllm/config/vllm.py:1387`-`1393`）；`TORCH_COMPILE_DISABLE` 环境变量同样走 `warning_once`；RISC-V 平台强制关闭 `enable_chunked_prefill`/`enable_prefix_caching` 走 `logger.info`；`enable_sleep_mode=True` 时强制打开 `enable_cumem_allocator` 走 `logger.info_once`（`vllm/config/model.py:606`-`613`）；`SamplingParams` 贪心采样时重置 `top_p`/`top_k`/`min_p` 完全没有日志（`vllm/sampling_params.py:529`-`534`）。
- **问题**：同样是"你设的值被丢弃了",五种不同的可见度。最后一种影响面最大——这是**每请求级别**的路径,用户传 `temperature=0` 做贪心解码时如果同时传了 `top_p=0.9`,这个值被悄悄清空,且没有任何字段能告诉用户"这次请求实际生效的采样参数不是你传的那组"。
- **证据等级**：源码为证（五处均已逐一打开核实；`enforce_eager` 这条经复核确认**不是**完全静默,而是"有 warning 但级别和其余几处不统一",见 `## 7` 第 2 条的措辞辨析）。
- **改进方向**：不要求所有覆盖都报错（`## 8.1` 的教训是"统一严格化"往往有兼容性代价）,但可以要求所有覆盖都至少打一条统一格式的 `logger.warning`,格式固定为"字段名：原值 → 新值（触发原因）"。`SamplingParams.__post_init__` 补一行日志的成本尤其低,是这条里最值得优先做的子项。
- **代价与反驳**：`SamplingParams` 的校验发生在每个请求,贪心采样又是最常见的用法之一——如果日志级别设成 `warning` 且没有做去重/采样,高 QPS 场景下这条日志本身可能变成新的噪音源；用 `logger.warning_once` 又会带来"用户可能只在第一次请求时看到警告,后续请求的静默覆盖仍然不可见"的新问题,完全没有免费的解法。
- **难度**：小改（`SamplingParams` 补一行日志）；统一 4 个文件的日志规范需要跨模块协调命名和格式,升级为中等。

### 8.4 `NO_ATTENTION` 枚举值指向不存在的模块,没有 CI 兜底

- **现状**：`vllm/v1/attention/backends/registry.py:112` 定义 `NO_ATTENTION = "vllm.v1.attention.backends.no_attention.NoAttentionBackend"`,但仓库里不存在 `vllm/v1/attention/backends/no_attention.py` 这个文件（本篇用 `find`/`grep` 全仓核实,唯一命中就是这一处枚举定义）。
- **问题**：如果任何代码路径真的尝试 `AttentionBackendEnum.NO_ATTENTION.get_class()`,会在运行时抛 `ModuleNotFoundError`,而不是在开发/CI 阶段被发现；且没有测试遍历全部 `AttentionBackendEnum` 成员逐个尝试 `import`,这类死引用可以在代码库里存在任意长时间而不被察觉。
- **证据等级**：源码为证。
- **改进方向**：加一个参数化测试,遍历 `AttentionBackendEnum` 全部成员（`CUSTOM` 除外,它本来就要求先注册）调用 `get_class()`,断言不抛异常——一旦以后有人重命名/删除某个 backend 文件而忘记同步枚举,CI 会立刻报错。
- **代价与反驳**：这类测试的代价是"每个 backend 文件的 import 副作用都会在 CI 里跑一遍"——如果某个 backend 依赖特定硬件/驱动（比如 ROCm-only 的实现）,在没有对应硬件的 CI runner 上跑测试反而会引入新的假失败,需要 `pytest.mark.skipif` 之类的机制按平台跳过,不是零成本。
- **难度**：小改（测试本身很短；需要多花心思处理跨平台 skip 逻辑）。

### 8.5 `api_server.py` 是 59 行废弃 shim,文档与真实装配点脱节

- **现状**：`vllm/entrypoints/openai/api_server.py` 全文件只有 59 行,内容是从 `vllm/entrypoints/launchers/api_server/*` 重新导出一批符号并在模块级 `warnings.warn`（`:24`-`30`）提示已弃用,真正的路由装配、`init_app_state`、`build_and_serve` 等逻辑都在 `vllm/entrypoints/launchers/` 下。
- **问题**：这本身不是 bug,但历史文档、教程、issue 里大量引用 `vllm.entrypoints.openai.api_server` 这条老路径——读者如果直接去这个文件找路由注册逻辑,只能看到一层重导出,得再跳一层才能找到真正实现。
- **证据等级**：源码为证。
- **改进方向**：不建议删除这个 shim（会立刻破坏所有存量导入）,但可以在文件顶部补一行结构性注释,直接点名"真实实现在 `vllm/entrypoints/launchers/`",并考虑在文档站搜索索引里把旧路径标记为"仅重定向"。
- **代价与反驳**：这条建议价值有限——`warnings.warn` 已经在做同样的事情,补注释只是把同一条信息从"运行时警告"复制到"静态阅读时可见",边际收益不大；真正的解法（统一替换文档站里引用旧路径的示例代码）超出了这个文件本身能解决的范围。
- **难度**：小改（加注释）；文档站层面的清理属于内容运营工作,不计入代码改动难度。

### 8.6 PRIORITY 抢占的受害者选择是 O(n) 线性扫描

- **现状**：`vllm/v1/core/sched/scheduler.py:652` 的 `preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))` 每次抢占都要线性扫描一遍 `self.running`。
- **问题**：如果一步调度内连续触发多次抢占（比如新来一个超大 prompt 一次性挤出多个低优先级请求）,这段逻辑退化成 O(n²)。正常部署下 `self.running` 规模不大,影响有限,但在 `SchedulingPolicy.PRIORITY` + 高并发 + 频繁抢占的组合场景下值得关注。
- **证据等级**：源码为证。
- **改进方向**：把 `self.running` 换成按 `(priority, arrival_time)` 排序的堆结构,能把单次选择降到 O(log n)。
- **代价与反驳**：当前实现里 `self.running` 的下标顺序本身承载着语义（其余调度循环按下标遍历并假设顺序稳定）,换成堆之后"移除任意一个元素后剩余顺序还满足其余代码的隐含假设"这件事需要重新证明——牵涉到调度器里好几处依赖 list 下标语义的代码,不是孤立的小改动。
- **难度**：中等偏设计讨论（数据结构本身简单,但影响面需要先摸清）。

### 8.7 WAITING 准入循环"一次拒绝就 break",可能造成队头阻塞

- **现状**：`enable_chunked_prefill=False` 时,`vllm/v1/core/sched/scheduler.py:964`-`971` 显示队首一个长 prompt 排不下 token 预算会直接 `break` 终止本步准入,不像 RUNNING 循环那样 `continue` 尝试队列里的下一个请求。
- **问题**：队列里排在长 prompt 后面、原本能一次性塞进剩余预算的短 prompt,本步完全没有机会被调度——即便它本可以立刻跑完,这是经典的队头阻塞。只在 `enable_chunked_prefill=False` 这个**非默认配置**下出现。
- **证据等级**：源码为证。
- **改进方向**：可以考虑参照 RUNNING 循环放宽成 `continue`,让调度器跳过排不下的长 prompt 继续尝试后续更短的请求。
- **代价与反驳**：放宽成 `continue` 会破坏 FCFS 的可预测性——本来排在前面的长 prompt 可能被无限期饿死在队列里,只要后面持续有短请求插队。SGLang 式的做法是在队列层面重排（`LPM`/`LOF` 策略,替换排序本身而不是放宽单次准入判断）,两种思路对"公平性 vs 吞吐"的取舍方向不同,贸然移植其中一半反而两头不讨好。**未查证**：本库没有核实上游是否已有相关讨论评估这个权衡。
- **难度**：需要设计讨论（实现本身是几行代码,难点在于要先决定清楚公平性语义）。

### 8.8 `SpecDecodingStats` 采集了接受率,但没有任何反馈通道

- **现状**：`vllm/v1/spec_decode/metrics.py:41` 的 `observe_draft(...)` 逐位记录接受 token 数,最终只流向 `SpecDecodingLogging`（Prometheus/日志展示）；本篇未在仓库中找到任何代码路径把这份统计读回来动态调整 `num_speculative_tokens` 或动态 K 表。
- **问题**：一个配置了投机解码但草稿模型/数据分布不匹配、接受率长期偏低的部署,唯一的发现方式是运维主动去看 Prometheus 面板——系统本身不会主动提示"这套投机解码配置可能不划算"。
- **证据等级**：源码为证（`SpecDecodingStats` 的写入点和展示点均已确认；"没有任何读回调整的路径"是全仓搜索后的否定性结论,不能排除本库检索遗漏）。
- **改进方向**：不需要做成全自动 EMA 闭环,先把"当前接受率是否长期低于某阈值"做成一条 `logger.warning_once`——`SpecDecodingLogging` 已经在周期性聚合这份数据,加一次阈值判断的成本很低。
- **代价与反驳**：阈值本身没有普适的默认值——不同草稿模型架构、不同任务类型的"正常接受率"差异很大,写死的阈值要么太敏感（制造无意义警告）,要么太迟钝（该警告的没警告）；做成可配置阈值又把"要不要调这个数"的负担重新丢回给用户,和当前"用户自己盯 Prometheus"的成本本质上是同一类问题。
- **难度**：小改（把已有的聚合统计接一次阈值判断和一条日志,不改动执行路径）。

### 8.9 `mlp_speculator` 在类型系统里活着、执行路径里已经死掉

- **现状**：`SpeculativeConfig.__post_init__` 的自动识别逻辑（`vllm/config/speculative.py:976`-`977`,`hf_config.model_type == "mlp_speculator"` 时把 `method` 设成这个值）完整存在,但 `GpuModelRunner.__init__` 的 `self.drafter` 分发链（`vllm/v1/worker/gpu_model_runner.py:701` 是这条链兜底分支所在处）没有任何分支处理这个取值。
- **问题**：一个 `hf_config.model_type` 恰好等于 `"mlp_speculator"` 的草稿模型,大概率会在初始化阶段撞上 `Unknown speculative decoding method` 报错——配置层"认得"这个方法名、执行层"不认得",诊断成本被留给了第一个踩到这个组合的用户。
- **证据等级**：源码为证（自动识别逻辑与分发链兜底分支均已确认；"没有对应实现"是通过阅读 `self.drafter` 分发链全部分支得出的否定性结论）。
- **改进方向**：要么在 `SpeculativeMethod` 定义处给这个取值加一条弃用说明并在 `__post_init__` 解析到它时提前报出更明确的错误（"`mlp_speculator` 在当前版本没有对应的 proposer 实现"）,要么干脆移除这条自动识别分支,让这类模型走通用的"未识别 speculative method"报错路径。
- **代价与反驳**：补充报错信息不改变任何行为,是纯粹的诊断体验改进；唯一的顾虑是这个方法名可能在更早版本里确实有实现、后来被移除时漏删了自动识别分支——如果是这样,更彻底的解法（直接删除该分支）需要先核实这一点。**未查证**：本库没有核实这条自动识别逻辑的历史沿革。
- **难度**：小改（补报错信息）；彻底删除分支需要先核实历史,风险略高但仍属小改量级。

### 8.10 `condense()` 依赖十几处状态数组手工同步搬移,没有字段清单自检

- **现状**：`vllm/v1/worker/gpu_input_batch.py:708` 起的 `condense()` 需要在同一次"空位/末位请求互换"操作里,对 `token_ids_cpu`、`spec_token_ids`、`block_table`、`num_computed_tokens_cpu` 等十几个并行数组逐一搬移。
- **问题**：`InputBatch` 上任何一个新增的、形状为 `(max_num_reqs, ...)` 的字段,如果实现者忘了在 `condense()` 里补一行对应的搬移逻辑,就会造成该字段在"请求被移除导致压缩"这条路径上状态错位——且这条路径不是最常被触发的路径,日常小规模测试很难覆盖到,错位可能潜伏很久才被发现。
- **证据等级**：源码为证。
- **改进方向**：给 `InputBatch` 补一个自检——遍历 `__init__` 里所有 `(max_num_reqs,)` 形状的属性,断言 `condense()` 的实现确实处理过每一个字段名。
- **代价与反驳**：这类自检本身容易写成形式主义——如果只是简单地按字段名字符串匹配,遇到字段名重构会产生误报；要做得可靠,需要自检逻辑理解"这个字段是否真的需要跟随请求移动"这个语义（有些字段可能是全局统计量）,这部分判断目前没有显式元数据支撑,加自检本身可能需要先给字段打标签,不是纯粹的测试代码。
- **难度**：中等（自检思路简单,但要避免误报/漏报需要先给字段分类）。

### 8.11 Rust 与 Python 路由表没有单一真源

- **现状**：Python 侧路由分散声明在 `vllm/entrypoints/*/api_router.py` 系列文件里,Rust 前端路由集中声明在 `rust/src/server/src/routes.rs:83`-`94`（该文件共 162 行）——`01-vLLM/01-vLLM-全景与代码地图.md` 的 `## 5` 已核实两侧路径集合存在真实差异（Rust 侧缺少若干 Python 侧已有的路径）。
- **问题**：打开 `VLLM_USE_RUST_FRONTEND` 的用户,如果调用了一个只在 Python 路由表里注册、Rust 路由表里没有的端点,得到的是普通的 404,而不是"这个端点在 Rust 前端暂不支持"这类明确提示——两套路由表的漂移只有在用户踩到的那一刻才会被发现。
- **证据等级**：源码为证（两个文件的存在与内容已复核；具体差异集合的完整核实工作由 `01-vLLM/01` 完成,本篇复用其结论）。
- **改进方向**：写一个从两侧分别抽取路径集合、跑集合差的 CI 检查——Python 侧可复用 `_lab/api_surface.py` 同款的 AST 抽取思路,Rust 侧对 `.route(...)` 调用做一次简单的正则抽取,不需要完整解析 Rust AST。
- **代价与反驳**：两套前端本身就不是承诺完全对等的实现——Rust 前端大概率是为了性能只覆盖高频端点的子集,如果 CI 检查不能区分"故意不实现"和"忘了实现",差集报告会变成一份长期挂红、没人认领的噪音清单;要做好这条,需要先有一份"Rust 前端目标覆盖范围"的显式声明作为比对基准。
- **难度**：中等（抽取脚本本身不难,难点是先要确定"差集里哪些是预期内的、哪些不是"这份基准）。

### 8.12 `/abort_requests` 双重注册没有冲突检测

- **现状**：`register_api_routers`（`vllm/entrypoints/launchers/api_server/routers.py:12`）在两个独立条件下分别引入两处 `/abort_requests` 注册——`"generate" in supported_tasks or "render" in supported_tasks` 时经 `register_scale_out_api_routers` 引入 `vllm/entrypoints/scale_out/token_in_token_out/api_router.py:79` 的版本（还需要 `args.tokens_only` 为真）；`envs.VLLM_SERVER_DEV_MODE` 为真时经 `register_vllm_dev_api_routers` 引入 `vllm/entrypoints/serve/dev/rlhf/api_router.py:94` 的版本。这两个条件互相独立,同时为真是可能发生的组合。
- **问题**：当 `VLLM_SERVER_DEV_MODE=1` 且 `--tokens-only` 同时成立时,同一个路径 `/abort_requests` 会被注册两次；按 Starlette/FastAPI 的路由匹配语义,先注册的一份会一直生效,后注册的一份变成永远匹配不到的死代码——这个组合在启动阶段没有任何检测或提示。
- **证据等级**：源码为证（两处注册条件与触发点均已确认；"其中一份变成死代码"这一后果依据 Starlette 路由"先注册先匹配"的公开行为推出,本库未针对这个具体组合额外写测试复现,推理链最后一环标注**本库推断**）。
- **改进方向**：在 `register_api_routers` 末尾加一次遍历 `app.routes`,按 `(path, method)` 找重复项,发现重复时打印一条 `logger.warning`,列出重复的路径和涉及的模块——这是一个通用的启动期自检,未来任何新增路由引入同类冲突都能被同一段代码捕获。
- **代价与反驳**：这类检测在启动路径上运行一次,开销可以忽略；唯一的成本是维护这段检测逻辑需要理解 FastAPI 路由对象的内部结构,如果上游未来升级 FastAPI/Starlette 大版本改变了内部结构,这段自检代码需要跟着维护,是一份和框架版本耦合的新技术债。
- **难度**：小改。

### 8.13 `AsyncScheduler` 的乐观计数字段缺少运行时不变量校验

- **现状**：`num_output_placeholders`/`num_in_flight_tokens`/`num_stale_output_tokens` 等字段要靠 `_update_after_schedule` 和 `update_from_output` 在调度和输出两个不同时间点分别加、分别减才能保持"乐观值"与"实际值"最终一致（例如投机解码 token 被拒绝时需要回滚 `num_computed_tokens`/`num_output_placeholders`）；本篇在 `vllm/v1/core/sched/scheduler.py` 和 `vllm/v1/core/sched/async_scheduler.py` 通读未发现有运行时断言在每步调度结束后统一校验这些计数器的不变量。
- **问题**：这类"两处分别维护、必须相互对齐"的状态,是并发/异步系统里最容易滋生难复现 bug 的地方之一——如果未来新的投机解码路径、新的抢占分支忘记同步更新其中一个计数器,现有代码结构下不会有任何东西提前报警,只会在下游某处出现看起来无关的症状（比如 KV 缓存分配数量对不上）。
- **证据等级**：本库推断（通读代码后未发现统一校验,是一个否定性结论；不能完全排除校验存在于本篇未覆盖到的路径,比如仅在 debug 构建/特定测试夹具里启用）。
- **改进方向**：在调度器每步收尾处（比如 `schedule()` 方法返回前）加一个仅在 debug/测试模式下启用的不变量校验函数,遍历这组"两两对齐"字段做一次一致性检查,发现不一致立刻 `assert` 报错并打印涉及的 `request_id`。
- **代价与反驳**：如果这段校验逻辑放在生产路径上无条件启用,会给每一步调度增加额外开销；即使做成仅 debug 模式启用,校验逻辑本身也需要精确定义"什么时候这些字段应该满足什么关系",这份不变量清单可能比 `01-vLLM/03-vLLM-调度器解剖.md` `## 3.4` 列出的字段表更复杂,写错校验条件反而会在正常场景下触发假警报。
- **难度**：中等（校验逻辑本身不难写,难点是先要把这组字段的不变量关系梳理完整并验证清楚,避免假阳性）。

### 8.14 组合校验只在真正构建这一层时才跑,用户可能要等模型加载到一半才发现某个 backend 选不了

- **现状**：`validate_configuration()`（`vllm/v1/attention/backend.py:263`）是在 `Attention` 层真正构建、经由 `get_attn_backend` 选择具体 backend 类时才被调用（调用点例如 `vllm/platforms/cuda.py:385`、`:417`）,发生在模型权重开始加载之后的模型构建阶段。
- **问题**：如果一个大模型有几十层,且某个不兼容的 backend 选择要到某一层构建时才暴露,用户已经等过了 CLI 参数解析、模型配置读取、（可能的）权重下载这些更耗时的步骤,才在权重加载中途看到 `validate_configuration` 报错——排查成本和等待成本都比"提前几秒钟报错"高得多。
- **证据等级**：源码为证（调用点与定义均已核实；"用户体验上要多等"这一后果基于调用时序推出,未实测具体等待时长）。
- **改进方向**：在 CLI 参数解析完成、模型权重还没开始下载/加载之前,先用模型配置文件里已知的 `head_size`/`dtype`/结构信息跑一遍"影子校验"——`ModelConfig` 在权重加载前就已经能提供这些字段,理论上可以把不兼容的 backend 选择在几秒内报出来。
- **代价与反驳**：这需要把 `validate_configuration` 的调用点从"真正构建 `Attention` 层"这个天然拥有全部运行时信息的位置,提前挪到一个只有静态配置、还没有真正实例化模型结构的位置——如果某些校验逻辑依赖的字段(比如某些从权重文件元数据里才能读到的信息)在这个更早的阶段还不可得,"影子校验"可能得不出和真实构建时一致的结论,反而制造一个"提前报的错和实际情况不符"的新问题。
- **难度**：中等（需要梳理 `validate_configuration` 依赖的全部字段,确认哪些在模型加载前就已可得）。

### 8.15 ViT 侧注意力 backend 候选列表和解码器侧独立维护,容易在新增 backend 时漏掉一边

- **现状**：`get_supported_vit_attn_backends()`（`vllm/platforms/cuda.py:505`-`519`）里的候选列表是手写的固定 `AttentionBackendEnum` 成员清单,不是从 `AttentionBackendEnum` 全量派生再过滤出"支持 encoder-only 场景"的子集。
- **问题**：如果未来给某个新的 attention backend 加上了 ViT 场景支持,`AttentionBackendEnum` 本身的注册（`vllm/v1/attention/backends/registry.py`）不会自动让它出现在 ViT 候选集里,需要有人记得手动把它加进 `vllm/platforms/cuda.py:505` 这个列表——这是一处容易在代码评审时被忽略的"两处独立维护、需要人工同步"的耦合点,和 `## 8.11` 的 Rust/Python 路由表是同一类问题的另一个实例。
- **证据等级**：源码为证。
- **改进方向**：给 `AttentionBackendEnum` 补一个显式的能力标记（比如一个 `supports_vit` 类属性或注册时的元数据字段）,`get_supported_vit_attn_backends()` 改成遍历全部枚举成员按这个标记过滤,而不是手写列表——这样新增 backend 时"是否支持 ViT"这个决策留在 backend 自己的定义处做出,不需要额外记得去改一个不相关文件里的列表。
- **代价与反驳**：这个改动需要先确认目前手写列表里的顺序是否承载语义（比如列表顺序是否代表某种优先级,`has_device_capability(80)` 前后两个分支的顺序差异就暗示这一点：`FLASHINFER` 在两个分支里都排最后,`TRITON_ATTN`/`TORCH_SDPA` 的顺序则互换）——如果顺序确实是优先级信号,改成"遍历过滤"就需要额外设计一个显式的优先级字段来替代当前隐含在列表书写顺序里的信息,不是单纯的"列表转标记"就能等价替换。
- **难度**：小改（如果顺序不承载语义）；如果顺序承载语义则升级为中等（需要额外设计优先级字段）。

### 8.16 Cohere 兼容端点的可用性策略不统一,容易让用户以为一个开关能控制全部

- **现状**：`VLLM_ENABLE_COHERE_API` 只门控 `/cohere/v2/chat`——`vllm/entrypoints/generate/api_router.py:75` 的 `if envs.VLLM_ENABLE_COHERE_API:` 是这条路由唯一的注册条件；但 `/v2/embed`（`vllm/entrypoints/pooling/embed/api_router.py:47` 的 `@router.post("/v2/embed", ...)`）和 `/v2/rerank`（`vllm/entrypoints/pooling/scoring/api_router.py:106`）的注册代码里都没有出现这个环境变量,是无条件注册的。
- **问题**：一个只设了 `VLLM_ENABLE_COHERE_API=1` 就以为自己"开启了 Cohere 兼容"的用户,不会意识到 `/v2/embed`/`/v2/rerank` 从一开始就是可用的、和这个环境变量完全无关——反过来,一个**没有**设这个环境变量、以为自己没有暴露任何 Cohere 兼容面的用户,`/v2/embed`/`/v2/rerank` 其实一直在线。
- **证据等级**：源码为证（三个端点的注册条件均已逐一打开核实）。
- **改进方向**：要么让 `/v2/embed`/`/v2/rerank` 也接受同一个环境变量门控（统一到"一个开关控制全部 Cohere 兼容面"）,要么在文档和 `/v1/models`/`server_info` 之类的自省端点里显式列出"当前这次启动实际生效的 Cohere 兼容端点有哪些",让用户不需要翻源码就能确认。
- **代价与反驳**：把 `/v2/embed`/`/v2/rerank` 也接入门控是一个行为变更——如果已经有依赖这两个端点默认可用的现有集成,加上门控会让它们突然 404;这两个端点本质上是"通用的 embedding/rerank 能力套了一层 Cohere 兼容的请求/响应格式",和"是否启用 Cohere SDK 集成"（`/cohere/v2/chat` 依赖真正的 `cohere` SDK 做校验,见 `vllm/entrypoints/cohere/api_router.py:238`）不是同一件事,现状的"不统一"某种程度上反映的是这两类端点本来就不该被同一个开关笼统地管——真正该改的可能是文档层面把这个区分讲清楚,而不是代码层面强行统一。
- **难度**：小改（自省端点补充信息）；统一门控行为属于破坏性变更,升级为中等。

### 8.17 `/tokenizer_info` 和 dev 族 `/server_info` 的门控方式不统一,无法只开其中风险较低的一个

- **现状**：`/tokenizer_info` 由独立 CLI 旗标 `--enable-tokenizer-info-endpoint` 控制（`vllm/entrypoints/openai/cli_args.py:140`-`141`,注册代码 `vllm/entrypoints/serve/tokenize/api_router.py:87`-`90` 的 `if getattr(app.state.args, "enable_tokenizer_info_endpoint", False):`）；`/server_info` 则完全绑在 `VLLM_SERVER_DEV_MODE` 这一个大开关上（`vllm/entrypoints/serve/dev/server_info/api_router.py:43` 无条件注册,但整个 `dev` 族模块只在 `register_api_routers`（`vllm/entrypoints/launchers/api_server/routers.py:34`-`37`）判断 `envs.VLLM_SERVER_DEV_MODE` 为真时才被 import/attach）。
- **问题**：一个运维只想开 `/server_info` 用于内部监控,却不想连带打开 `/collective_rpc` 这种"extremely dangerous"级别的端点（`## 8.1`）,目前没有办法只开一部分 dev 族路由,只能全开或全不开——`VLLM_SERVER_DEV_MODE=1` 是一个粒度过粗的开关,把"暴露一些配置信息"和"暴露任意 RPC 执行能力"绑在同一个布尔值上。
- **证据等级**：源码为证。
- **改进方向**：可以考虑把 `/server_info`、`/reset_prefix_cache` 这类"只读或低风险"操作从 `VLLM_SERVER_DEV_MODE` 里拆出来,参照 `/tokenizer_info` 的模式单独给一个风险级别更低的开关（比如 `--enable-server-info-endpoint`）,`/collective_rpc`、`/sleep`、`/wake_up` 这类真正高危的端点继续留在 `VLLM_SERVER_DEV_MODE` 下。
- **代价与反驳**：拆分开关意味着要重新审视 dev 族里每一个端点的风险等级并逐一归类,这本身是一份需要谨慎评估的清单（错误分类比不分类更危险——如果一个本该算高危的端点被误归进"低风险"新开关,反而降低了它原来"至少需要设 dev mode"这道门槛）；`VLLM_SERVER_DEV_MODE` 这个名字本身也暗示了"这一整组端点都只该在开发环境使用"的设计意图,拆细粒度开关某种程度上是在鼓励生产环境局部启用本该整体视为"开发态"的功能,这和维护者最初的意图可能是相反方向。
- **难度**：中等（拆分本身是配置层改动,但需要先完成一份可靠的风险分级)。

---

### 不建议改的几条

机器信号里有几处"看起来像问题",打开源码复核后判断是合理设计,列在这里作为本篇的诚实度自查——不是凑数,是明确标注"扫到了、看过了、判断不需要改"。

**A. 823 处 `raise NotImplementedError` 里,多少是抽象基类的正常占位？**

机械识别标准（三选一命中即算"占位"：紧邻 `@abstractmethod`/`@abstractproperty` 装饰器；所在 class 显式继承 `ABC`/`Protocol`/`ABCMeta`；异常消息匹配"必须被子类实现/override/subclass must"一类措辞）在全仓 823 处里能确认 **338 处、约 40%**——这是一个下界,因为 vLLM 里有相当一部分接口是"非正式基类"：既不用 `abc` 模块也不用 `Protocol`,就是一个普通类里放几个 `raise NotImplementedError`（比如 `vllm/distributed/device_communicators/base_device_communicator.py:76` 的 `All2AllManagerBase.get_handle`,`vllm/compilation/base_static_graph.py:38`/`:57` 的 `AbstractStaticGraphWrapper` 虽然用了 `Protocol` 但两处占位本身没有 `@abstractmethod` 装饰,`vllm/v1/worker/worker_base.py:190` 附近的 `WorkerBase` 也是同一模式）。本库对机械识别未命中的 497 处做了系统抽样（每 17 条抽 1 条,共 30 条）人工复核,其中 6 条（20%）是这类"非正式基类占位",其余 24 条（80%）是真正的"某个具体软硬件/精度/并行度组合当前不支持"运行时报错（例如 `vllm/_custom_ops.py:2009` 的"XPU 上不支持非对称 int8 量化"、`vllm/v1/attention/backends/mla/flashattn_mla.py:298` 的"不支持 encoder attention type"、`vllm/vllm_flash_attn/flash_attn_interface.py:316` 的"FA2 不支持 `aux_tensors`"）。按这个抽样比例外推,823 处里**抽象占位的真实比例估计落在 45%~50% 区间**,另外约一半是具体的"这个组合暂不支持"报错——这两类的改进方向完全不同（前者是接口设计问题,后者多数是诚实地把能力边界写清楚,本身就是好实践）,不能把 823 当成一个待办事项数字来读。（本库用 `grep -rn "raise NotImplementedError"` 复核到全仓 835 处,比 `improve.json` 记录的 823 处多出约 12 处,大概率是两边扫描器对 `vllm/vllm_flash_attn/`、`vllm/benchmarks/` 这类子目录的收编口径不同,未逐处对账,不影响本条的比例结论。）

**B. `except ImportError` 处理可选依赖是正当设计,不是异常处理缺陷**

`vllm/_aiter_ops.py:25` 的 `except ImportError: pd = PlaceholderModule("pandas")` 和 `vllm/_custom_ops.py:29` 的 `except ImportError: from torch.library import impl_abstract as register_fake` 都是这个模式：前者是可选依赖延迟报错（真正用到 `pandas` 时才会因为 `PlaceholderModule` 报出清晰的安装提示,而不是在 import 阶段就让不需要这个功能的用户装不上 vLLM）；后者是新旧 PyTorch API 名称的兼容层（`register_fake` 是新名字,`impl_abstract` 是旧 PyTorch 版本的名字）。这两处都被 `improve.json` 的 `silent_except` 信号收录,但都是教科书式的正当用法,不需要改。

**C. 探测型文件的静默降级是设计意图,不是疏漏**

`vllm/platforms/cuda.py` 是 `silent_except` 密度全仓最高的文件（10 处）,但逐条打开后发现全是同一类模式：`:204` 的 `_win_kernel_ver` 在 WSL2 内核版本号解析失败时返回 `None`（仅用于一条非关键的 WSL2 专属警告判断）；`:555` 在探测 ViT 专用 attention backend 时,遇到 `ImportError` 就 `pass` 并回退到安全默认值 `TORCH_SDPA`；`:1021` 附近探测 NVML 是否可用时,注释直接写明"On Jetson, NVML is not supported"——这三处以及同文件其余若干处,共同点是**探测的结果只用来决定一个安全的默认值/告警,不影响正确性**,这正是 `improve.json` 的 `caveat` 字段提前说明的情形。

**D. 非阻塞轮询循环里的 `queue.Empty`/`zmq.Again` 是正常控制流,不是异常处理缺陷**

88 处"热路径静默异常"里,KV 连接器目录（`vllm/distributed/kv_transfer/kv_connector/v1/`）贡献了将近一半,但相当一部分是这个模式：`vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py:719` 的 `try: notif = self._pending_completion_notifs.get_nowait()` 配 `except queue.Empty: break`,这是"非阻塞取队列、取空了就退出本轮循环"的标准写法,`queue.Empty`/`zmq.Again` 在这里是预期内的控制流信号,不是需要处理的错误——把它算进"静默吞异常"的密度统计里会高估这类文件的风险,这也是 `improve.json` 要求按 `role` 拆分而不是直接看总数的原因。

**E. 一处机器信号自身的误标,提醒读数字时别停在第一层**

`vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:415` 被 `improve.json` 标注为 `swallow: "no-raise-no-log"`——字面意思是"异常被吞了,既不重新抛出也不记日志"。但打开上下文会发现,这个 `except Exception as e:` 块的返回值是 `self._fail_task("Loaded", f"Task execution error: {e}", request_id, future)`,而 `_fail_task`（定义于 `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py:422`）内部第一行就是 `logger.error(...)`,把这个错误信息真实地打了日志。静态扫描器只看 `except` 代码块字面上有没有直接出现 `raise`/`logger.*` 调用,看不到"日志调用被封装进了一个辅助函数"这种间接路径,于是给出了"未记录"的假阴性标注。这条本身不该算进 vLLM 的问题清单,而是给本库自己提了个醒——`improve.json` 的每一条 `swallow` 标注,写进正文前都必须去看那个函数体的完整实现,不能只信字段值。

**F. `gpu_model_runner.py` 是全场最大的 god_file(7753 行),但不建议按行数拆分**

`vllm/v1/worker/gpu_model_runner.py` 从 `improve.json` 的 `god_files` 榜单看是 vLLM 里最大的单文件,超第二名 `_custom_ops.py`(4343 行)近一倍。但打开文件核实其结构后发现,7752 行(本篇 `wc -l` 实测值,与 `improve.json` 记录的 7753 行相差 1 行,属正常的换行符计数口径差异)里绝大部分属于**同一个类**——`GPUModelRunner`(定义于 `vllm/v1/worker/gpu_model_runner.py:501`),一路延伸到文件末尾。这个类是每一步前向推理的核心状态机：输入批次准备、`condense()` 压缩(`## 8.10`)、CUDA Graph 分发(`## 8.2`)、attention 元数据构建、模型前向调用、采样与输出组装,全部是**同一次调度步骤内、必须共享大量中间状态的连续操作**(本库推断,依据是通读类内方法调用关系后的判断,未逐一核实每个方法之间的具体数据依赖)。把这类"一个高度内聚的热路径状态机"按文件大小机械拆成几个文件,大概率不会减少真实的复杂度,只会把原本能在一个类定义里看到的完整调用链拆散到多个文件的 import 关系里,让"这一步到底做了什么"变得更难跟踪——`god_file` 计数本身是一个中性的规模信号,不能直接当"应该拆分"的证据,这条和 `## 6`/`## 7` 里"god_file 数量不能跨引擎直接排名"是同一类提醒,只是这里落在同一个引擎内部的单个文件上。

## 9. 自测题与延伸阅读

### 自测题

1. `--api-key` 的 `GUARDED_PREFIX` 具体保护哪四个路径前缀？`/collective_rpc` 为什么不在其中却格外危险——它在 worker 进程里是怎么把一个字符串 `method` 变成真正的函数调用的？
2. CUDA Graph 静默回退到 eager 有哪两类触发条件？想在日志里看到这类回退,需要打开哪个默认关闭的配置项？
3. `enforce_eager=True` 覆盖 `compilation_config` 时到底有没有日志？这和 `SamplingParams` 贪心采样覆盖 `top_p`/`top_k`/`min_p` 相比,可见度差在哪里？为什么不能把前者也叫"完全静默覆盖"？
4. `NO_ATTENTION` 这个死引用具体错在哪一行？为什么现有代码库里没有任何测试能在 CI 阶段提前抓到它？
5. 823 处 `raise NotImplementedError` 里,机械识别能确认的抽象占位下界是多少？人工抽样复核后,本篇给出的真实比例区间是多少？这个数字为什么不能直接当"823 处待办"来读？
6. 举一个具体例子说明：为什么"88 处热路径静默异常"不能直接等同于"88 个潜在 bug"？
7. vLLM 只有 7 个 god_file（≥3000 行）,SGLang 有 23 个、TensorRT-LLM 有 25 个——这是否说明 vLLM 的代码组织比另外两个引擎更健康？这个推论哪里站不住脚？
8. `validate_configuration()` 现在在哪个阶段被调用？如果要把它挪到 CLI 解析阶段就跑,最大的技术障碍是什么？
9. `/cohere/v2/chat` 和 `/v2/embed`/`/v2/rerank` 的可用性分别由什么条件决定？一个只设了 `VLLM_ENABLE_COHERE_API=1` 的用户,容易对哪个端点的可用性产生误解？
10. `vllm/v1/worker/gpu_model_runner.py` 全场行数最多,为什么本篇不建议单纯按行数把它拆分成多个文件？拆分真正应该权衡的是什么？
11. `/tokenizer_info` 和 `/server_info` 的门控方式分别是什么？为什么现在没有办法只开 `/server_info` 而不连带打开 `/collective_rpc`？

### 延伸阅读

- [[00-总览与阅读地图]] —— 本库的整体地图与三类原材料的关系
- [[09-vLLM-Python-API与EngineArgs]] —— `## 8.3` 复用的"静默覆盖"完整清单来自这篇
- [[08-vLLM-HTTP-API表面全解]] —— `## 8.1`/`## 8.12` 的鉴权与路由细节来自这篇
- [[03-vLLM-调度器解剖]] —— `## 8.6`/`## 8.7`/`## 8.13` 的调度器背景
- [[06-vLLM-模型执行与CUDA-Graph]] —— `## 8.2` 的 CUDA Graph 背景
- [[10-vLLM-投机解码]] —— `## 8.8`/`## 8.9` 的投机解码背景
