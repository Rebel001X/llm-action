# SGLang 前端 DSL 与编程模型解剖

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：前端把提示排成共享前缀，后端的树才有命中可捡。

**范围声明**：本篇只讲 `python/sglang/lang/` 这一层——DSL 的对象模型、解释器、`fork`/`select`/`join` 的语义、多后端适配、以及它和前缀缓存之间的因果链。radix 树本身怎么匹配/分裂/驱逐前缀，见 [[03-SGLang-RadixAttention与前缀缓存]]；`regex`/`json_schema` 在 `srt` 侧具体怎么落成逐 token 约束解码，见 [[06-SGLang-约束解码与语法后端]]；两篇都不在本篇重复展开。

**证据分级**：正文里标了行号的断言都是"源码为证"；涉及提交历史、CI 归属、路由是否存在这几条（集中在 §0.5、§0.6、§7）额外标注了取证方法（GitHub API / 本地 `grep`），因为本地浅克隆验不了历史；明确写"本库推断"的几处都是没有源码或提交记录直接支撑、需要读者自己判断可信度的推测，集中在 §6 对 vLLM 设计动机的猜测。

## 0. 结论先行

1. **"SG" = Structured Generation，这套 DSL 的对象模型是一棵 `SglExpr` 子类构成的表达式节点集合**：`SglConstantText`/`SglGen`/`SglSelect`/`SglRoleBegin`/`SglFork`/`SglVariable`… 全部定义在 `python/sglang/lang/ir.py`（`SglExpr` 基类 `python/sglang/lang/ir.py:327`，`SglGen` `python/sglang/lang/ir.py:451`，`SglSelect` `python/sglang/lang/ir.py:533`）。但 `@sgl.function` 包出来的 Python 函数**不是"先建完整棵树再统一解释"**——它在调用时立刻用真实的 Python 控制流（`if`/`for`/递归）跑一遍函数体（`python/sglang/lang/interpreter.py:44`），每次 `s += expr` 只是把当前这一个 IR 节点丢进一个队列（`python/sglang/lang/interpreter.py:342`），由**另一条专门的后台线程**异步消费执行（`python/sglang/lang/interpreter.py:422`）。这是生产者-消费者模型，不是"编译再运行"。
2. **`fork()` 根本不走 IR 节点队列，是纯 Python 值拷贝**。`ir.py` 里定义了 `SglFork`/`SglGetForkItem` 两个 IR 节点类（`python/sglang/lang/ir.py:552`、`python/sglang/lang/ir.py:565`），但 `interpreter.py` 的执行分发表（`_execute` 方法，`python/sglang/lang/interpreter.py:461`）里没有任何一个分支处理它们——全仓 `grep` 证实这两个类只被 `python/sglang/lang/tracer.py:114`、`python/sglang/lang/tracer.py:123` 引用，即只服务于静态前缀追踪子系统。真正跑起来的 `StreamExecutor.fork()`（`python/sglang/lang/interpreter.py:370`）直接 `dict(self.variables)`/`str(self.text_)` 做值拷贝，`BaseBackend.fork_program()` 这个专门为"通知后端有分叉发生"留的钩子（`python/sglang/lang/backend/base_backend.py:38`）在全仓库里从未被任何代码调用过。
3. **`select()` 的默认打分口径是"长度归一化的平均 log-prob 取 argmax"，不是约束解码**：`TokenLengthNormalized.__call__`（`python/sglang/lang/choices.py:34`）直接 `np.argmax(normalized_prompt_logprobs)`，而 `normalized_prompt_logprobs` 由 `RuntimeEndpoint.select()` 通过 `compute_normalized_prompt_logprobs`（`python/sglang/lang/backend/runtime_endpoint.py:351`）算出——就是输入 token logprob 的算术平均。且这个语义只在 `RuntimeEndpoint`（打到 SGLang 自家 server）上完整成立；换到 `OpenAI` 后端，`select()` 整个换了一套算法：靠 `logit_bias` 强制逐 token 贪心比对、用命中 token 数当分数，且明确声明忽略 `choices_method`（`python/sglang/lang/backend/openai.py:319`）；`Anthropic`/`LiteLLM`/`VertexAI` 三个后端根本没有覆写 `select()`，调用即 `NotImplementedError`（继承自 `python/sglang/lang/backend/base_backend.py:70`）。
4. **DSL 与 RadixAttention 是一条完整因果链，不是两个独立特性凑在一起**：`fork()` 在真正分叉前会先把已经攒好的文本原样发一次 `max_new_tokens=0` 的 `/generate`（`StreamExecutor.fork` 里的 `SglCommitLazy` 提交，`python/sglang/lang/interpreter.py:376`，落到 `RuntimeEndpoint.commit_lazy_operations` `python/sglang/lang/backend/runtime_endpoint.py:105`），把共享前缀预先"写"进服务端的前缀缓存；随后每个分支各自独立发请求，因为它们的 `text_` 字节级相同，服务端的 radix 树天然把它们匹配到同一个前缀节点上。`select()` 更直接：一次 HTTP 请求里打包 `[s.text_ + c for c in choices]`（`python/sglang/lang/backend/runtime_endpoint.py:265`）——这一批 prompt 共享同一个前缀字符串，radix 树见到的是"一批带共同前缀的请求"，不是"几个互不相干的请求"。`run_batch()` 默认还会先做一次纯静态追踪把公共前缀抽出来预热缓存（`enable_precache_with_tracing` 默认 `True`，`python/sglang/lang/global_config.py:21`；调用点 `python/sglang/lang/interpreter.py:106`）。第 4、6 节会把这条链逐步拆开。
5. **现状：这套 DSL 在仓库里活着，但已经退到"维护模式"**——90 天窗口内（2026-05-24 至今）`python/sglang/lang/` 只有 6 次提交触达，`python/sglang/srt/` 同期有 2339 次（GitHub 提交历史 API 实测，本地 `_src` 是 `--depth 1` 浅克隆没有历史，方法见 §7）；2025-11-16 的提交 `e019f233` 整体删除了 `python/sglang/lang/compiler.py`（231 行，标题为 "Remove unused code / testcases in `lang`"），三天后 `196b940a` 把 `lang` 的测试从 CI 自动发现的 `test/lang/` 迁到"非 CI、仅手动触发"的 `test/manual/lang_frontend/`（`test/README.md:15`）。如今仓库自己的 CI（`test/registered/`）里唯一还在用这套 DSL 的地方，是 32 个精度评测文件里各自复制粘贴的同一段 `few_shot_gsm8k` 批量推理套路（`@sgl.function` + `.run_batch()`），`fork`/`select` 在 `test/registered/` 里出现次数为 0。
6. **比"没有 CI"更硬的一条证据：`join()` 的其中一条代码路径调用的服务端接口，在当前 `srt` 上已经不存在了**。`RuntimeEndpoint.concatenate_and_append()`（`python/sglang/lang/backend/runtime_endpoint.py:317`-`324`）请求的是 `/concate_and_append_request`；把 `python/sglang/srt/entrypoints/http_server.py`（2827 行）里全部 `@app.post`/`@app.api_route` 路由列一遍，没有任何一条匹配这个路径。也就是说 `forks.join(mode="concate_and_append")`（第 3、7 节详述）这条路径，在这份取证基准对应的服务端代码上调用即失败。

## 1. 它在系统里的位置

`python/sglang/lang/` 是一个**纯客户端库**：它不在 `srt`（SGLang Runtime，实际跑模型、管调度器、管 KV 缓存的那部分）进程里执行。依赖方向是**单向**的，而且方向容易被想反：全仓 `grep "from sglang.lang" python/sglang/srt/` 零命中——`srt` 从不引用 `lang`，`srt` 完全可以在没有 `lang/` 这个目录的情况下独立跑起来。反过来，`lang/` 会在几个具体的地方**懒加载** `srt`：`Runtime()`（`python/sglang/lang/api.py:35`-`39`）懒加载 `Runtime` 类（`python/sglang/lang/backend/runtime_endpoint.py:356`），后者的 `__init__` 直接 `from sglang.srt.entrypoints.http_server import launch_server`（`python/sglang/lang/backend/runtime_endpoint.py:383`）并用 `multiprocessing.get_context("spawn")` 把 `srt` 服务器整个拉起成一个子进程（`python/sglang/lang/backend/runtime_endpoint.py:403`-`408`）；`Engine()`（`python/sglang/lang/api.py:42`-`44`）懒加载 `srt` 的 `Engine`；`separate_reasoning()` 执行到 `_execute_separate_reasoning` 时（`python/sglang/lang/interpreter.py:754`），会当场 `from sglang.srt.parser.reasoning_parser import ReasoningParser`（`python/sglang/lang/interpreter.py:767`）借用 `srt` 侧的推理内容解析器。这几处 import 全部写成函数体内的延迟导入而不是文件顶部的模块级 import，理由在源码注释里写得很直白："Avoid importing unnecessary dependency"（`python/sglang/lang/api.py:36`、`python/sglang/lang/api.py:43`）——不走 `Runtime`/`Engine`/`separate_reasoning` 这几条路的用户，装 `sglang` 时可以完全不触达 `torch`/CUDA 这类 `srt` 的重依赖。它和 `srt` 之间真正跑起来时只有一条通路：HTTP。

```
用户 Python 脚本                          用户 Python 脚本                裸 HTTP 客户端
   │  @sgl.function 装饰                     │  sgl.Engine(...)                 │
   │  s += expr、s.fork()/select()…          │  .generate(prompt)               │
   ▼                                         ▼                                  ▼
python/sglang/lang/                     （不经过 lang/，                  python/sglang/srt/entrypoints/
   │  ir.py          —— IR 节点定义         直接用 srt 的 Engine）           http_server.py
   │  interpreter.py —— StreamExecutor                                     （/generate、
   │  tracer.py      —— 影子解释器                                          /v1/chat/completions…）
   │  backend/       —— 翻译成对应后端的请求                                       │
   ▼                                                                              │
BaseBackend 子类之一（python/sglang/lang/backend/base_backend.py:9）              │
   ├── RuntimeEndpoint —— HTTP 打到 SGLang 自家 srt server（本地或远程） ─────────┤
   └── OpenAI / Anthropic / LiteLLM / VertexAI / Crusoe —— HTTP 打到第三方 API    │
   ▼                                                                              ▼
（只有 RuntimeEndpoint 这条路径才会走到）
python/sglang/srt/ —— 调度器 + RadixAttention + KV 缓存 + 模型执行（三条入口最终都汇到这里，或汇到某个第三方 API 自己的服务端）
```

三条入口里，`lang/` 是唯一"程序运行到一半"有个 `ProgramState` 对象可以查询、可以继续追加内容的路径；`Engine`/裸 HTTP 都是一次调用给一个完整输入、拿一个完整输出，调用之间不维护任何 DSL 层面的状态。

这个位置关系在源码里有一句非常直白的自白：`Runtime` 类（`python/sglang/lang/backend/runtime_endpoint.py:356`）的文档字符串写着——

> "It is mainly used for the frontend language. You should use the Engine class if you want to do normal offline processing without the frontend language."（`python/sglang/lang/backend/runtime_endpoint.py:362`-`363`）

也就是说，SGLang 自己把"要不要用这套 DSL"当成了一个**入口选择**：走 `Runtime`/`RuntimeEndpoint` 就是在用 DSL；走 `Engine`（`python/sglang/srt/entrypoints/engine.py`，由 `python/sglang/lang/api.py:44` 懒加载）就是"正常的离线处理"，完全绕开 `lang/`。`python/sglang/__init__.py:26` 起的 import 块导出 `Runtime`，`python/sglang/__init__.py:68` 用 `LazyImport` 导出 `Engine`——两者是并列导出的两个入口，选哪个由用户决定，不是包含关系。还有第三个入口完全不经过 Python：直接起 `srt` 的 HTTP server（`python/sglang/srt/entrypoints/http_server.py`）配 OpenAI 兼容的 `/v1/chat/completions`（`python/sglang/srt/entrypoints/http_server.py:1722`），这条路径连 `Engine` 都不用，更谈不上 `lang/`——第 6 节要说的正是：**这第三条路径，在 vLLM 那边几乎是唯一的路径。**

`lang/` 内部再往细分两条线：
- **执行线**（`interpreter.py`）——`@sgl.function` 真正跑起来走的路径，第 3、4 节的主角；
- **追踪线**（`tracer.py`）——不真正生成文本，只用一套结构相似但语义是"记录节点"的影子解释器（`TracerProgramState._execute`，`python/sglang/lang/tracer.py:144`）跑一遍函数体，抽出**常量前缀**喂给 `cache_program`（`python/sglang/lang/interpreter.py:242`）。这条线只服务一件事：批量调用前，把公共前缀预热进后端缓存。

`lang/` 怎么把一次 `s += sgl.gen(...)` 变成一次 HTTP `/generate` 调用，属于第 4 节主流程的问题；`srt` 收到请求之后怎么用 radix 树匹配前缀，属于 [[03-SGLang-RadixAttention与前缀缓存]] 的范围，本篇只负责讲清楚"前端为什么会制造出可复用的前缀"这一半。

## 2. 代码地图（文件 → 职责，带行号）

`lang/` 目录一共 14 个 `.py` 文件、4,644 行（`find python/sglang/lang -name '*.py' | xargs wc -l` 实测，命令与结果见 §7），比 `srt/`（1,646 个 `.py` 文件、89,798 行）小两个数量级。核心文件：

| 文件 | 职责 | 关键行 |
|---|---|---|
| `api.py`（292 行） | 面向用户的顶层函数：`function`/`gen`/`select`/`system`/`user`/`assistant`/`Runtime`/`Engine`，把调用参数打包成 IR 节点 | `function()` `python/sglang/lang/api.py:23`；`gen()`（`choices` 走 `SglSelect` 分支）`python/sglang/lang/api.py:75`、`python/sglang/lang/api.py:102`-`108`；`select()` `python/sglang/lang/api.py:236`；`Runtime()` 懒加载 `python/sglang/lang/api.py:35`；`Engine()` 懒加载 `python/sglang/lang/api.py:42`-`44` |
| `ir.py`（643 行） | IR 节点类定义 + `SglFunction`（`@sgl.function` 的产物）+ `SglSamplingParams`（多后端参数适配） | `SglFunction` 类 `python/sglang/lang/ir.py:141`；`SglFunction.run` `python/sglang/lang/ir.py:160`；`SglFunction.__call__` `python/sglang/lang/ir.py:316`；`SglExpr` 基类（`+`/`+=` 运算符重载）`python/sglang/lang/ir.py:327`、`python/sglang/lang/ir.py:350`-`359`；`SglSamplingParams` `python/sglang/lang/ir.py:18`-`138`；`SglGen` `python/sglang/lang/ir.py:451`；`SglSelect` `python/sglang/lang/ir.py:533`；`SglFork`/`SglGetForkItem`（只服务 tracer，见 §0.2）`python/sglang/lang/ir.py:552`、`python/sglang/lang/ir.py:565` |
| `interpreter.py`（1098 行，全库最大） | 真正执行 DSL 的地方：`StreamExecutor`（生产者-消费者）+ `ProgramState`/`ProgramStateGroup`（fork/join） | `run_program` `python/sglang/lang/interpreter.py:57`；`StreamExecutor` 类 `python/sglang/lang/interpreter.py:274`；worker 线程启动 `python/sglang/lang/interpreter.py:329`-`332`；`submit()` `python/sglang/lang/interpreter.py:342`；`fork()` `python/sglang/lang/interpreter.py:370`；`_execute` 分发表 `python/sglang/lang/interpreter.py:461`；`_execute_gen` `python/sglang/lang/interpreter.py:593`；`_execute_select` `python/sglang/lang/interpreter.py:647`；`ProgramState` 类 `python/sglang/lang/interpreter.py:852`；`ProgramState.fork` `python/sglang/lang/interpreter.py:888`；`ProgramStateGroup` 类 `python/sglang/lang/interpreter.py:1045`；`ProgramStateGroup.join` `python/sglang/lang/interpreter.py:1052`-`1076` |
| `tracer.py`（279 行） | 影子解释器，只为批量调用抽公共前缀服务 | `extract_prefix_by_tracing` `python/sglang/lang/tracer.py:29`-`51`；`TracerProgramState.fork`（真正生成 `SglFork` 节点的唯一地方）`python/sglang/lang/tracer.py:108` |
| `choices.py`（164 行） | 三种 `select()` 打分策略 | `ChoicesSamplingMethod` 抽象类 `python/sglang/lang/choices.py:14`；`TokenLengthNormalized`（默认）`python/sglang/lang/choices.py:32`；`GreedyTokenSelection` `python/sglang/lang/choices.py:56`；`UnconditionalLikelihoodNormalized` `python/sglang/lang/choices.py:110` |
| `chat_template.py`（679 行） | 各家模型的 chat 角色前后缀模板 | `ChatTemplate` 类 `python/sglang/lang/chat_template.py:13`；`get_chat_template` `python/sglang/lang/chat_template.py:69` |
| `global_config.py`（27 行） | 进程级开关 | `enable_precache_with_tracing`（默认 `True`）`python/sglang/lang/global_config.py:21`；`enable_parallel_encoding` `python/sglang/lang/global_config.py:22` |
| `backend/base_backend.py`（82 行） | 后端抽象接口 | `BaseBackend` 类 `python/sglang/lang/backend/base_backend.py:9`；`select()` 默认 `NotImplementedError` `python/sglang/lang/backend/base_backend.py:63`-`70`；`fork_program()`（从未被调用的钩子）`python/sglang/lang/backend/base_backend.py:38`-`44` |
| `backend/runtime_endpoint.py`（551 行） | 打到 SGLang 自家 `srt` server 的后端；唯一"语义完整"的后端 | `RuntimeEndpoint` 类 `python/sglang/lang/backend/runtime_endpoint.py:26`；`cache_prefix` `python/sglang/lang/backend/runtime_endpoint.py:80`；`commit_lazy_operations` `python/sglang/lang/backend/runtime_endpoint.py:105`；`select()` `python/sglang/lang/backend/runtime_endpoint.py:248`；`compute_normalized_prompt_logprobs` `python/sglang/lang/backend/runtime_endpoint.py:351`-`353`；`Runtime` 类（内嵌启动 `srt` 子进程）`python/sglang/lang/backend/runtime_endpoint.py:356` |
| `backend/openai.py`（475 行） | 远程 OpenAI 兼容 API 后端 | `OpenAI` 类 `python/sglang/lang/backend/openai.py:56`；`select()`（logit_bias 贪心比对，忽略 `choices_method`）`python/sglang/lang/backend/openai.py:312`-`380` |
| `backend/anthropic.py`（73 行）/`litellm.py`（90 行）/`vertexai.py`（148 行） | 远程后端，均未覆写 `select()` | 只实现 `generate`/`generate_stream`（如 `Anthropic` 类 `python/sglang/lang/backend/anthropic.py:12`、`generate` `python/sglang/lang/backend/anthropic.py:26`），`select` 继承基类直接抛异常 |

（以上 20+ 条 `文件:行` 引用，超过 `## 2` 硬指标的 10 条。）

`lang/` 只通过 HTTP 触达 `srt`，具体打到哪几条路由，对照 `python/sglang/srt/entrypoints/http_server.py`（2827 行）核过一遍：

| `lang/` 里的调用点 | 打到的 `srt` 路由 | 路由是否存在 |
|---|---|---|
| `RuntimeEndpoint.generate()` `python/sglang/lang/backend/runtime_endpoint.py:159` | `/generate` | 存在，`python/sglang/srt/entrypoints/http_server.py:889` |
| `RuntimeEndpoint.__init__` `python/sglang/lang/backend/runtime_endpoint.py:41`-`46` | `/get_model_info` | 存在，`python/sglang/srt/entrypoints/http_server.py:728` |
| `RuntimeEndpoint.flush_cache()` `python/sglang/lang/backend/runtime_endpoint.py:59`-`66` | `/flush_cache` | 存在，`python/sglang/srt/entrypoints/http_server.py:966` |
| `RuntimeEndpoint.cache_prefix()`/`commit_lazy_operations()` `python/sglang/lang/backend/runtime_endpoint.py:80`、`:105` | `/generate`（`max_new_tokens=0`，复用同一条路由） | 存在，同上 |
| `RuntimeEndpoint.concatenate_and_append()` `python/sglang/lang/backend/runtime_endpoint.py:317`-`324` | `/concate_and_append_request` | **不存在**，见 §7.1 |

## 3. 核心数据结构

### 3.1 `SglExpr` 家族——IR 节点，但只是"待执行指令"不是"完整程序"

`SglExpr`（`python/sglang/lang/ir.py:327`）是所有 IR 节点的基类，自己只做两件事：分配一个自增 `node_id`（`python/sglang/lang/ir.py:328`-`334`），以及重载 `+`/`+=` 让字符串和表达式能拼接成 `SglExprList`（`concatenate_ir`，`python/sglang/lang/ir.py:350`-`359`）。它的子类分三类：

- **内容类**：`SglConstantText`（普通文本，`python/sglang/lang/ir.py:506`）、`SglImage`/`SglVideo`（多模态输入）；
- **求值类**：`SglGen`（调用模型生成，携带一份 `SglSamplingParams`，`python/sglang/lang/ir.py:451`）、`SglSelect`（从候选里选一个，`python/sglang/lang/ir.py:533`）；
- **控制/结构类**：`SglRoleBegin`/`SglRoleEnd`（chat 角色边界）、`SglVarScopeBegin`/`SglVarScopeEnd`（截取一段文本存变量）、`SglFork`/`SglGetForkItem`（只在 `tracer.py` 出现，见 §0.2）、`SglCommitLazy`（触发 `backend.commit_lazy_operations`）。

注意 `SglExpr` 本身不携带"下一步做什么"的信息——顺序完全由 `ProgramState.__iadd__` 调用 `stream_executor.submit()` 的**先后次序**决定（`python/sglang/lang/interpreter.py:1023`-`1027`），本质上是一条指令流，不是一棵会被整体遍历执行的语法树。`print_graph_dfs`（`python/sglang/lang/ir.py:361`-`394`）能把节点打印成一张依赖图，但那是调试可视化用的，不参与实际执行。

### 3.2 `SglSamplingParams`——一份要适配 5 种不同 API 的采样参数

`python/sglang/lang/ir.py:18`-`138` 定义的这个 dataclass 装了 `max_new_tokens`/`temperature`/`regex`/`json_schema` 等近 20 个字段，自带 5 个 `to_xxx_kwargs()` 方法（`to_openai_kwargs` `python/sglang/lang/ir.py:64`、`to_vertexai_kwargs` `python/sglang/lang/ir.py:79`、`to_anthropic_kwargs` `python/sglang/lang/ir.py:93`、`to_litellm_kwargs` `python/sglang/lang/ir.py:109`、`to_srt_kwargs` `python/sglang/lang/ir.py:121`），每个方法负责把统一参数降级成对应 API 能接受的子集——比如 `to_openai_kwargs` 直接丢弃 `top_k`（OpenAI 不支持），`regex` 字段在四个远程后端各自的 `to_xxx_kwargs` 里全部只是 `warnings.warn` 后丢弃（`python/sglang/lang/ir.py:66`-`67`、`python/sglang/lang/ir.py:80`-`83`、`python/sglang/lang/ir.py:95`-`98`、`python/sglang/lang/ir.py:110`-`111`），只有 `to_srt_kwargs` 把它原样传下去（`python/sglang/lang/ir.py:136`）。这份"最大公约数参数 + 每个后端各自阉割"的设计，是第 5 节要展开的一条决策。

### 3.3 `StreamExecutor` —— 一个程序实例的运行时状态机

`python/sglang/lang/interpreter.py:274` 起的 `StreamExecutor` 是真正"跑程序"的对象，关键字段（原文摘录）：

```python
# python/sglang/lang/interpreter.py:295-319（节选）
self.variables = {}  # Dict[name: str -> value: str]
self.variable_event = {}  # Dict[name: str -> event: threading.Event]
self.meta_info = {}  # Dict[name: str -> info: str]
...
self.text_ = ""  # The full text
...
self.messages_ = []  # The messages in the OpenAI API format
...
# For fork/join
self.fork_start_text_pos = None
```

它同时持有一个 `queue.Queue`（`python/sglang/lang/interpreter.py:324`）和一条独立的后台线程（`python/sglang/lang/interpreter.py:326`-`332`），`submit()` 把 IR 节点丢进队列，后台线程的 `_thread_worker_func`（`python/sglang/lang/interpreter.py:422`）循环 `queue.get()` 并调用 `_execute()`。这意味着**一次 `@sgl.function` 调用至少有两条线程**：调用者线程跑 Python 函数体本身（决定控制流），worker 线程跑实际的网络请求。

### 3.4 `ProgramState` / `ProgramStateGroup` —— fork/join 的容器

`ProgramState`（`python/sglang/lang/interpreter.py:852`）包一层 `StreamExecutor`，给用户暴露 `+=`、`s["var"]`、`s.fork()` 等语法糖；`fork()` 返回的 `ProgramStateGroup`（`python/sglang/lang/interpreter.py:1045`）持有一组 `ProgramState`，重载了 `+=` 让"给所有分支同时追加同一段表达式"变成一行代码（`python/sglang/lang/interpreter.py:1084`-`1098`），`join()`（`python/sglang/lang/interpreter.py:1052`-`1076`）负责收尾。

### 3.5 `ChoicesSamplingMethod` —— 打分策略的抽象类

`python/sglang/lang/choices.py:14` 定义的 ABC，签名要求实现者接收 `normalized_prompt_logprobs`/`input_token_logprobs`/`output_token_logprobs`（可选还要 `unconditional_token_logprobs`），返回一个 `ChoicesDecision(decision, meta_info)`。三个实现分别对应"长度归一化平均 logprob"（`python/sglang/lang/choices.py:32`）、"逐 token 贪心保留候选集合"（`python/sglang/lang/choices.py:56`）、"减去无条件 logprob 再归一化"（`python/sglang/lang/choices.py:110`）三种口径，差异在第 5 节展开。

### 3.6 流式输出与"API 投机执行"——两个只在特定条件下才触发的分支

`_execute_gen`（`python/sglang/lang/interpreter.py:593`）不是只有一条路径。除了 §4 走读的普通同步生成，还有两个专门的分支：

**流式输出**：`state.text_iter()`/`text_async_iter()`（`python/sglang/lang/interpreter.py:918`、`python/sglang/lang/interpreter.py:956`）要求逐 token 消费结果，`_execute_gen` 在 `self.stream` 为真时走另一半代码：

```python
# python/sglang/lang/interpreter.py:630-645（节选）
generator = self.backend.generate_stream(
    self, sampling_params=sampling_params
)
self.variables[name] = ""
self.stream_var_event[name].set()
for comp, meta_info in generator:
    self.text_ += comp
    self.variables[name] += comp
    self.meta_info[name] = meta_info
    self.stream_var_event[name].set()
    self.stream_text_event.set()
```

每收到一个增量片段就把对应的 `threading.Event` `set()` 一次，`text_iter()` 那边在另一条线程上 `event.wait()`/`event.clear()` 轮着消费——这是 §5 决策一"两条线程"设计在流式场景下的具体落地。

**"API 投机执行"**：`@function(num_api_spec_tokens=256)`（真实用法见 `examples/frontend_language/usage/openai_chat_speculative.py:23`-`24`）给某个函数开一个"每次少发几次 HTTP 请求"的模式，专门针对**只暴露 chat 接口、按轮次计费/计延迟的远程 API**（比如 OpenAI 的 chat 模型）。原理是：同一个 `assistant()` 角色块里如果连续写了好几个 `sgl.gen(...)`，正常情况下每个 `gen` 都要单独打一次 HTTP；开了这个模式后，第一次 `gen` 会请求生成 `num_api_spec_tokens` 个 token（比每次要的都多），存进 `self.speculated_text`（`python/sglang/lang/interpreter.py:319`），后续同一个 assistant 块内的 `gen`/`fill` 优先从这段"投机文本"里就地切（`_execute_fill` 在满足条件时直接调 `self.backend.spec_fill(value)` 而不产生新请求，`python/sglang/lang/interpreter.py:508`-`515`；`_spec_gen` 负责按 `stop` 条件从投机文本里切出这次要的片段，不够才真正 `regen()`，`python/sglang/lang/interpreter.py:543`-`557`），直到 `SglRoleEnd("assistant")` 触发 `role_end_generate` 把真正攒下来的调用一次性执行（`python/sglang/lang/interpreter.py:683`-`691`）。这条机制要求 `self.backend.is_chat_model` 为真（`python/sglang/lang/interpreter.py:604`、`:687`），且示例文件的用法说明写得很直接："for speculative execution to work, user must put all `gen` in `assistant`"（`examples/frontend_language/usage/openai_chat_speculative.py:3`）——一旦把几个 `gen` 拆到不同的 `s += sgl.assistant(...)` 语句里，这个优化就不生效。这条机制和 RadixAttention 无关，是**为远程闭源 API 减少往返次数**这条完全独立的动机，`test/registered/`、`test/manual/lang_frontend/` 里都没有覆盖它，仅存在于 `examples/` 两个文件（`openai_chat_speculative.py`、`openai_speculative.py`）里。

### 3.7 `SglConcateAndAppend`——`join()` 的第二条路径：显式拼接子分支的 KV cache

第 3.4 节的 `ProgramStateGroup.join()` 默认走 `mode="gather_variable"`，那是纯 Python 层面的字典合并，不触达后端。但 `join()` 还接受 `mode="concate_and_append"`（`python/sglang/lang/interpreter.py:1067`-`1071`），对应一个专门的 IR 节点 `SglConcateAndAppend`（`python/sglang/lang/ir.py:602`-`608`）。它想做的事是：把 `fork` 出来的几个分支各自新增的那段文本、以及它们各自服务端已经算出来的 KV，**在服务端拼接成一条序列**，而不是像默认模式那样只把 Python 变量收集回来：

```python
# python/sglang/lang/interpreter.py:738-752
def _execute_concatenate_and_append_kv_cache(self, expr: SglConcateAndAppend):
    self_len = len(self.text_)
    for i, s in enumerate(expr.states):
        exe = s.stream_executor
        exe.submit(SglCommitLazy())
    for i, s in enumerate(expr.states):
        exe = s.stream_executor
        exe.sync()
        assert exe.fork_start_text_pos == self_len
        self.text_ += exe.text_[exe.fork_start_text_pos :]
    src_rids = [state.stream_executor.sid for state in expr.states]
    self.backend.concatenate_and_append(src_rids, self.sid)
```

这条路径只在 `global_config.enable_parallel_encoding`（默认 `True`）**且** `self.backend.support_concate_and_append` 同时成立时才会被选中（`python/sglang/lang/interpreter.py:492`-`499`）；`support_concate_and_append` 只有 `RuntimeEndpoint` 置为 `True`（`python/sglang/lang/backend/runtime_endpoint.py:35`），`BaseBackend` 默认 `False`（`python/sglang/lang/backend/base_backend.py:11`），条件不满足时会退化成纯文本拼接（`_execute_concatenate_and_append_text`，`python/sglang/lang/interpreter.py:729`-`736`，只把子分支的文本片段接起来，不做任何 KV 层面的操作）。`RuntimeEndpoint.concatenate_and_append()`（`python/sglang/lang/backend/runtime_endpoint.py:317`-`324`）把这件事翻译成一次 HTTP 请求，`src_rids`/`dst_rid` 分别是子分支和父分支各自 `StreamExecutor.sid`（`python/sglang/lang/interpreter.py:289`）。这条路径当前是否还能真正跑通，第 7 节给出证据。

### 3.8 chat 角色如何被翻译——以及一次"隐式注入"

`sgl.system(...)`/`sgl.user(...)`/`sgl.assistant(...)`（`python/sglang/lang/api.py:246`-`262`）本质上都是 `_role_common` 包一层 `SglRoleBegin`/`SglRoleEnd`。真正的翻译工作在 `_execute_role_begin`（`python/sglang/lang/interpreter.py:665`-`681`）里：

```python
# python/sglang/lang/interpreter.py:665-681
def _execute_role_begin(self, expr: SglRoleBegin):
    assert self.cur_role is None, "Nested roles are not allowed."

    if len(self.messages_) == 0 and expr.role != "system":
        # Insert the default system message
        default_system = self.chat_template.default_system_prompt
        if default_system:
            self._execute_role_begin(SglRoleBegin("system"))
            self._execute_fill(default_system)
            self._execute_role_end(SglRoleEnd("system"))

    self.cur_role = expr.role
    prefix, _ = self.chat_template.get_prefix_and_suffix(expr.role, self.messages_)
    self._execute_fill(prefix, prefix=True)
    self.cur_role_begin_pos = len(self.text_)
```

值得注意的两点：第一，这是一处**递归自调用**——如果程序第一句是 `s += sgl.user(...)` 而没有先手写 `sgl.system(...)`，`_execute_role_begin` 会先拿 `ChatTemplate.default_system_prompt`（`python/sglang/lang/chat_template.py:15`，例如 `"chatml-llava"` 模板给的默认值就是 `"You are a helpful assistant."`，`python/sglang/lang/chat_template.py:122`）自己造一轮 `system` 角色塞进去，再继续处理原本这句 `user`——这意味着两段字面上一样的 DSL 程序，如果一个显式写了 `sgl.system(...)`、另一个没写，最终喂给模型的 `text_` 可能是等价的（因为默认值刚好补上了同一句话），也可能不等价（如果默认值和你本来想写的系统提示不一样），这是一处容易被忽略的隐式行为。第二，真正决定每个角色前后缀长什么样的是 `ChatTemplate.get_prefix_and_suffix`（`python/sglang/lang/chat_template.py:22`），这部分模型特定的模板差异被完全封装在 `chat_template.py` 里，`interpreter.py` 侧的角色处理逻辑对所有模型都是同一套代码。

### 3.9 `var_scope`/`copy()` —— 两个建在 `submit`/`fork` 之上的语法糖

`with s.var_scope("draft"):` 这种写法（`ProgramState.var_scope`，`python/sglang/lang/interpreter.py:882`-`886`）用来"框出一段文本，事后当一个变量取出来"：

```python
# python/sglang/lang/interpreter.py:882-886, 719-724
@contextmanager
def var_scope(self, name: str):
    self.stream_executor.submit(SglVarScopeBegin(name))
    yield
    self.stream_executor.submit(SglVarScopeEnd(name))
# 对应的两个 _execute 方法：
# _execute_var_scope_begin: self.variables[expr.name] = int(len(self.text_))   —— 先记一个文本长度当起点
# _execute_var_scope_end:   self.variables[expr.name] = self.text_[self.variables[expr.name]:]  —— 结束时切片取出这段新增文本
```

`with` 块内不管塞多少个 `sgl.gen()`/普通文本，`var_scope` 都只是在进入/离开时各提交一个标记节点，实际"取文本"发生在 `_execute_var_scope_end`（`python/sglang/lang/interpreter.py:722`-`724`）里对 `self.text_` 做一次切片——复用的还是 §3 讲过的"`text_` 是一条不断增长的字符串"这个基础事实，没有引入新的存储结构。

`ProgramState.copy()`（`python/sglang/lang/interpreter.py:898`-`904`）则直接暴露了"`fork` 才是真正原语"这件事：

```python
# python/sglang/lang/interpreter.py:898-904
@contextmanager
def copy(self, position_ids_offset: Optional[List[int]] = None):
    state_group = self.fork(1, position_ids_offset)
    try:
        yield state_group[0]
    finally:
        state_group.join()
```

`s.copy()` 就是 `s.fork(1)` 再在 `with` 块结束时自动 `join()`——"复制一份状态出去改、改完扔掉"这个常见需求，不是一套独立机制，只是 `fork`/`join` 套上 `size=1` 和上下文管理器语法糖后的特例。

### 3.10 `SglFunction.bind` —— 给 `@sgl.function` 做柯里化

`SglFunction.bind(**kwargs)`（`python/sglang/lang/ir.py:154`-`158`）不改原对象，而是拿 `self.bind_arguments` 和新传入的 `kwargs` 合并出一份新字典，`return SglFunction(self.func, bind_arguments=new_bind_dict)` 返回一个新的 `SglFunction`。效果类似 `functools.partial`：如果 `parallel_sample`（§4 例子）要固定 `n=5`，可以 `fixed = parallel_sample.bind(n=5)`，之后 `fixed.run(question=...)` 不用每次都传 `n`。`run_program` 里 `func_kwargs.update(program.bind_arguments)`（`python/sglang/lang/interpreter.py:70`）说明绑定的参数优先级高于 `.run()` 调用时同名的普通参数——这是文档里没有明说、需要读源码才知道的顺序细节。

### 3.11 多模态输入的一个隐藏限制：`RuntimeEndpoint` 一次只吃一张图

`sgl.image(path)`（`python/sglang/lang/api.py:228`-`229`）产出 `SglImage`（`python/sglang/lang/ir.py:434`-`439`），`_execute_image`（`python/sglang/lang/interpreter.py:524`-`531`）把图片 base64 编码后追加进 `self.images_` 列表，并在 `text_` 里插入模板对应的图片占位符（`self.chat_template.image_token`）。`s.images_` 本身是个列表，一个程序里理论上可以多次 `s += sgl.image(...)` 累积多张图。但 `RuntimeEndpoint._add_images`（`python/sglang/lang/backend/runtime_endpoint.py:337`-`340`）在真正发请求前有一句断言：

```python
# python/sglang/lang/backend/runtime_endpoint.py:337-340
def _add_images(self, s: StreamExecutor, data):
    if s.images_:
        assert len(s.images_) == 1, "Only support one image."
        data["image_data"] = s.images_[0][1]
```

也就是说走 `RuntimeEndpoint` 这条路径，一次 `/generate` 请求最多只能带一张图——`OpenAI` 后端走 `_execute_role_end` 里单独攒 `content` 列表的路径（`python/sglang/lang/interpreter.py:698`-`714`），对多图没有这条硬性限制。这是又一处"同一段 DSL 代码换后端行为不同"的例子，且这次是以断言报错的形式出现，而不是 §5 决策四那种"警告后静默丢弃"。

### 3.12 `gen(dtype=...)` 才是这套 DSL 里真正的约束解码；`select()` 不是

§0.3 说了 `select()` 不是约束解码，那这套 DSL 里有没有真正意义上的约束解码？有，是 `gen(dtype=int/float/str/bool)`（`gen_int`/`gen_string` 是它的两个具名快捷方式，`python/sglang/lang/api.py:142`、`python/sglang/lang/api.py:185`）。`ir.py` 头部定义了四个正则表达式常量（`python/sglang/lang/ir.py:11`-`14`）：

```python
# python/sglang/lang/ir.py:11-14
REGEX_INT = r"[-+]?[0-9]+[ \n]*"
REGEX_FLOAT = r"[-+]?[0-9]*\.?[0-9]+[ \n]*"
REGEX_BOOL = r"(True|False)"
REGEX_STR = r"\"[\w\d\s]*\""  # bugs with regex r"\".*\"" in interegular pkg
```

`RuntimeEndpoint._handle_dtype_to_regex`（`python/sglang/lang/backend/runtime_endpoint.py:127`-`157`）在真正发请求前把 `dtype` 翻译成对应的正则表达式塞进 `sampling_params.regex`，这个字段最终会被 `srt` 侧独立的语法后端用来做真正的逐 token 约束解码（那条路径的实现细节见 [[06-SGLang-约束解码与语法后端]]，本篇不重复）。`OpenAI` 后端完全走另一套：没有正则约束能力，`dtype=int` 靠 `logit_bias=self.logit_bias_int`（`python/sglang/lang/backend/openai.py:214`，`logit_bias_int` 在 `__init__` 里预先算好、只允许数字 token，`python/sglang/lang/backend/openai.py:81`）把输出限制在数字上，`dtype=str` 靠一个更粗糙的技巧：在 prompt 末尾手动拼一个 `"`、把 `stop` 设成 `"`（`python/sglang/lang/backend/openai.py:193`-`194`），逼模型在遇到下一个引号时停下——这不是约束解码，是"拿字符串边界当停止符"的土办法。两个 dtype 分支都在开头 `assert not self.is_chat_model`（`python/sglang/lang/backend/openai.py:183`-`185`、`python/sglang/lang/backend/openai.py:203`-`205`），即 chat 模型完全不支持 `gen(dtype=...)`。

这里还带出一个比 §3.6 讲得更精确的事实：`OpenAI.generate()` 对 chat 模型、且没开"API 投机执行"时，如果 `s.text_` 不是恰好停在 `self.chat_prefix`（assistant 角色的前缀）末尾，会直接 `raise RuntimeError`（`python/sglang/lang/backend/openai.py:149`-`154`），报错信息原文就是 "sgl.gen must be right after sgl.assistant... adding api speculative execution: `@function(num_api_spec_tokens=128)`"。也就是说，对 OpenAI 的 chat 模型，"一个 `assistant` 块里只能有一个紧跟在角色开头的 `gen()`，否则必须开投机执行"不是文档建议，是**不满足就直接抛异常**的硬约束——§3.6 例子里 `@function(num_api_spec_tokens=256)` 那一行装饰器，对着 OpenAI chat 模型时几乎是必需品，不是可选的性能优化。

## 4. 主流程走读

用一个真实存在的示例文件走一遍：`examples/frontend_language/usage/parallel_sample.py`（全文 41 行，仓库里原样存在，下面按该文件自身行号引用）：

```python
# examples/frontend_language/usage/parallel_sample.py:9-26
@sgl.function
def parallel_sample(s, question, n):
    s += (
        "Question: Compute 1 + 2 + 3\n"
        "Reasoning: I need to use a calculator.\n"
        "Tool: calculator\n"
        "Answer: 6\n"
        "Question: Compute 3 + 2 + 2\n"
        "Reasoning: I will try a calculator.\n"
        "Tool: calculator\n"
        "Answer: 7\n"
    )
    s += "Question: " + question + "\n"
    forks = s.fork(n)
    forks += "Reasoning:" + sgl.gen("reasoning", stop="\n") + "\n"
    forks += "Tool:" + sgl.gen("tool", choices=["calculator", "browser"]) + "\n"
    forks += "Answer:" + sgl.gen("answer", stop="\n") + "\n"
    forks.join()
```

调用方式（同文件 `examples/frontend_language/usage/parallel_sample.py:29`-`40`）：

```python
sgl.set_default_backend(sgl.OpenAI("gpt-3.5-turbo-instruct"))
# sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
state = parallel_sample.run(question="Compute 5 + 2 + 4.", n=5, temperature=1.0)
for i in range(5):
    obj = {"reasoning": state["reasoning"][i], "tool": state["tool"][i], "answer": state["answer"][i]}
```

一步步对到源码（假设把默认后端换成注释掉的 `RuntimeEndpoint`，这样才走得到 §0.4 的完整因果链）：

1. **`@sgl.function` 装饰**：`function()`（`python/sglang/lang/api.py:23`-`32`）把 `parallel_sample` 包成一个 `SglFunction` 实例（`python/sglang/lang/ir.py:141`），此时函数体**完全没有执行**——只是在 `SglFunction.__init__` 里用 `inspect.getfullargspec` 解析出参数名（`python/sglang/lang/ir.py:149`-`151`），断言第一个参数必须叫 `s`。
2. **`.run(question=..., n=5, temperature=1.0)`**：`SglFunction.run`（`python/sglang/lang/ir.py:160`）把关键字参数打包成 `SglSamplingParams`，调 `run_program`（`python/sglang/lang/interpreter.py:57`）。`run_program` 建一个 `StreamExecutor`（构造时立刻启动它自己的 worker 线程）、包成 `ProgramState`，再调 `run_internal`（`python/sglang/lang/interpreter.py:42`-`44`）——**这一步是同步的**（`stream=False` 是默认值，`python/sglang/lang/interpreter.py:88`-`90`），也就是说 `program.func(state, question, n)` 在调用者线程上直接、立刻执行，函数体里的 `for`/`if` 走的是真 Python 解释器，不是任何自建的图执行器。
3. **`s += "Question: ...\n"`**：`ProgramState.__iadd__`（`python/sglang/lang/interpreter.py:1023`-`1027`）把字符串包成 `SglConstantText`，调 `stream_executor.submit()`（`python/sglang/lang/interpreter.py:342`）——因为 `use_thread` 默认 `True`，这一步只是 `queue.put(expr)`（`python/sglang/lang/interpreter.py:346`），**立刻返回**，不等实际发生任何网络请求。调用者线程继续往下跑，worker 线程在后台把队列里的节点一个个 `_execute`。
4. **`forks = s.fork(n)`**：`ProgramState.fork`（`python/sglang/lang/interpreter.py:888`-`896`）调 `StreamExecutor.fork(5)`（`python/sglang/lang/interpreter.py:370`）。因为此时 `text_` 非空，先 `submit(SglCommitLazy())`（`python/sglang/lang/interpreter.py:375`-`376`）——worker 线程执行到这条时调用 `RuntimeEndpoint.commit_lazy_operations`（`python/sglang/lang/backend/runtime_endpoint.py:105`-`114`），拿当前 `text_`（两个 few-shot 例子 + 当前问题）发一次 `max_new_tokens=0` 的 `/generate`，**把这段前缀原样"跑"一遍但不生成任何 token**——这是主动把前缀预热进服务端 radix 树的动作。接着 `self.sync()`（`python/sglang/lang/interpreter.py:378`）阻塞等队列清空，确认预热请求真正发出去了，然后创建 5 个新的 `StreamExecutor`（`python/sglang/lang/interpreter.py:381`-`390`），每个都 `dict(self.variables)`/`str(self.text_)` 值拷贝父状态（`python/sglang/lang/interpreter.py:391`-`398`）——5 份**字节完全相同**的文本副本。
5. **`forks += "Reasoning:" + sgl.gen("reasoning", stop="\n") + "\n"`**：右边先算出一个 `SglExprList`（字符串 `+` `SglGen` 触发 `SglExpr.__radd__`/`__add__`，`python/sglang/lang/ir.py:336`-`359`），再由 `ProgramStateGroup.__iadd__`（`python/sglang/lang/interpreter.py:1084`-`1098`）广播给 5 个子 `ProgramState`——每个子状态各自 `submit` 一份**同一个** `SglExprList` 对象。5 个 worker 线程各自异步执行，各自往 `RuntimeEndpoint.generate()`（`python/sglang/lang/backend/runtime_endpoint.py:159`）发请求，`data["text"]` 就是各自的 `s.text_`——此刻五份 `text_` 仍然完全相同（都停在 "Reasoning:" 之前），所以五个并发请求打到服务端时前缀字节相同，radix 树按最长公共前缀匹配，只需要为超出已缓存部分的新增文本做 prefill。
6. **`sgl.gen("tool", choices=["calculator", "browser"])`**：因为传了 `choices`，`gen()`（`python/sglang/lang/api.py:75`）在第 102-108 行短路，直接返回 `SglSelect` 而不是 `SglGen`（`python/sglang/lang/api.py:102`-`108`）——**`choices=` 是 `select()` 的语法糖，不是约束解码**。执行到这个节点时走 `_execute_select`（`python/sglang/lang/interpreter.py:647`-`658`）→ `RuntimeEndpoint.select()`（`python/sglang/lang/backend/runtime_endpoint.py:248`）：先发一次零 token `/generate` 确定当前 prompt 的真实 token 数（用于 token-healing 修正，`python/sglang/lang/backend/runtime_endpoint.py:257`-`261`），再把 `[s.text_ + "calculator", s.text_ + "browser"]` 打包成**一次批量请求**（`python/sglang/lang/backend/runtime_endpoint.py:264`-`273`），拿回每个候选的 `input_token_logprobs`，用 `compute_normalized_prompt_logprobs`（`python/sglang/lang/backend/runtime_endpoint.py:351`-`353`，就是算术平均）算出长度归一化 logprob，交给 `token_length_normalized`（`python/sglang/lang/choices.py:53`）取 `argmax`。这一批请求内部也共享前缀——两条候选序列除最后几个 token 外完全相同。
7. **`forks.join()`**：`ProgramStateGroup.join()`（`python/sglang/lang/interpreter.py:1052`-`1076`），默认 `mode="gather_variable"`：对每个子状态先 `sync()` 确保队列跑完，再把子状态 `variables` 里"父状态原本没有"的键收集成列表塞回父状态（`python/sglang/lang/interpreter.py:1057`-`1066`）——`reasoning`/`tool`/`answer` 三个变量名在父状态里各自变成一个长度为 5 的列表。之后每个子 `StreamExecutor` 调 `end()`（`python/sglang/lang/interpreter.py:1076`）。
8. **`state["reasoning"][i]`**：`ProgramState.__getitem__` → `get_var`（`python/sglang/lang/interpreter.py:1029`-`1030`、`python/sglang/lang/interpreter.py:354`-`357`），此时 `join()` 已经把结果写回父状态，直接读字典即可。

这条走读证明了 §0.4 的因果链在这一个例子里全部成立：`fork` 前的显式预热 + fork 后天然相同的文本前缀 + `select` 内部的批量共前缀请求，三处都在制造"radix 树能命中"的条件，而这些都是**语言层面的编排结果**，不是运行时引擎自己猜出来的。

补一句：`parallel_sample.py` 里的 `forks.join()` 不带参数，走的是默认的 `mode="gather_variable"`——纯 Python 层面把 5 个分支的变量收回来，不触达后端。如果把最后一行换成 `forks.join(mode="concate_and_append")`，代码路径会完全不同（§3.7），这也是第 7 节要专门核实的地方。

## 5. 设计决策与代价

**决策一：生产者-消费者执行模型（调用者线程跑控制流，worker 线程跑 IO）。**
- 为什么这么设计：`text_iter()`（`python/sglang/lang/interpreter.py:918`）这类流式读取 API 要求"一边生成新内容一边把已经生成的部分吐给用户"，如果构造 prompt 的 Python 控制流和真正发请求、等回包的 IO 挤在同一条线程上，就没法在等待网络响应的同时继续往前推进后续节点的入队。把"决定接下来提交什么"和"真正把请求发出去、等回包"拆成两条线程，才能撑住这种边生成边消费的用法。
- 不这样会怎样：`submit()` 确实保留了同步分支（`use_thread=False` 时直接 `self._execute(expr)`，`python/sglang/lang/interpreter.py:345`-`348`），一旦启用，每次 `gen()` 调用都会阻塞到 HTTP 响应回来才能往下走。错误处理也因此变复杂：worker 线程里的异常要专门通过 `self.error_` 字段跨线程传回调用者（`python/sglang/lang/interpreter.py:454`），调用者必须显式调 `.error()` 才能看到——这是异步模型必须承担的调试成本。
- 什么时候可以不这样：一条直线跑到底、不需要流式输出、也不在意"提交完立刻返回"这层解耦的简单脚本，`use_thread=False` 更好调试（异常直接在调用栈里抛出，不用跨线程收集）；不过高层 `SglFunction.run()` 目前没有把这个开关暴露给最终用户，要用就得绕开 `api.py`/`ir.py` 直接调 `interpreter.run_program(..., use_thread=False)`。

**决策二：`fork()` 绕过 IR 队列，直接做同步的 Python 值拷贝。**
- 为什么这么设计：分叉出的每个分支都需要一条独立的执行线（各自的 `queue`/worker 线程），如果 fork 本身也只是"提交一个节点，由某条 worker 线程负责创建 N 条新 worker 线程"，就会把"决定要不要分叉"和"分叉后的并发执行"绑在同一条工作线程上，多一层没有必要的间接。直接在调用 `fork()` 的那条线程里同步完成分叉，`fork()` 一返回，N 个独立的 `StreamExecutor` 就已经就绪、各自的线程也已经启动——语义更直接。
- 不这样会怎样：如果 fork 也走队列异步执行，`forks = s.fork(n)` 这行代码返回时 `forks` 内部的子 `StreamExecutor` 列表可能还没被真正创建出来，要么强迫用户到处显式等待，要么得让 `ProgramStateGroup` 支持"延迟绑定"。`tracer.py` 那套确实做了类似的延迟处理（因为它的目标不是执行，是收集图结构），代价是同一套"fork 语义"要在两个文件里各写一遍（第 7 节详述）。
- 什么时候可以不这样：如果一开始就接受"这套 DSL 只服务批处理场景，不需要在单个程序内部做交互式分叉"，`fork` 完全可以不做成运行时特性，退化成 `run_batch`（`python/sglang/lang/ir.py:223`）——从 §0.5 的 CI 证据看，SGLang 自己的测试套件事实上正是这么用的：`test/registered/` 里从未出现过 `.fork(`。

**决策三：`select()` 默认用"长度归一化平均 logprob"，而不是直接比较总 logprob 或做约束解码。**
- 为什么这么设计：不同候选字符串的 token 数不同，直接比较总 log-likelihood 天然偏向短选项（负数越少越好），归一化到"平均每 token"能消掉长度这个混杂因子；用打分而不是约束解码，是因为候选可以是任意长度的字符串，约束解码要为 N 个候选临时拼一个只接受这 N 条路径的自动机，工程上比"一次批量 prefill 拿 logprob"（`python/sglang/lang/backend/runtime_endpoint.py:264`-`273`）更复杂——后者可以完全复用已有的 `/generate` + `return_logprob` 接口，不用给 `srt` 另开一条约束解码的路（SGLang 已经有独立的语法后端处理 `regex`/`json_schema`，见 [[06-SGLang-约束解码与语法后端]]；`select` 不复用那条路，说明这是刻意的产品切分：`select` 面向"从确定的少量候选里选一个"，语法后端面向"生成过程中动态收窄词表"）。
- 不这样会怎样：如果用总 logprob 不归一化，长候选会系统性地吃亏；`GreedyTokenSelection`（`python/sglang/lang/choices.py:56`-`104`）其实是在往"逐 token 比较"方向走了半步（重叠选项延伸到公共长度取平均值再比较），说明作者确实考虑过别的口径，但没有把它设成默认。
- 什么时候可以不这样：候选之间长度、置信度分布都接近时，三种 `ChoicesSamplingMethod` 往往选出同一个答案，这时用哪种口径几乎不影响结果；`UnconditionalLikelihoodNormalized`（`python/sglang/lang/choices.py:110`-`161`）专门处理"某个候选本身在语言模型里天然更常见"这类先验偏置问题，需要额外一次"不带 prompt"的 logprob 请求，只有候选之间先验概率差异明显大时才值得多付这次请求的代价。但"往往选出同一个答案"不是必然——仓库自己的（非 CI）单测 `test/manual/lang_frontend/test_choices.py:12`-`52` 用一组人为构造的 mock logprob 数据，让三种方法在同一组 3 个候选（`"organ"`/`"organism"`/`"antidisestablishmentarianism"`）上分别选出三个不同的答案（`test/manual/lang_frontend/test_choices.py:57`-`87`）——这组数据是专门设计出来演示分歧的，不代表真实模型输出的典型分布，但足以证明"换一种打分口径，结果真的可能变"不是纸上谈兵。

**决策四：参数不兼容时"警告后丢弃"而非"直接报错"——`regex`/`dtype` 在四个远程后端上被静默忽略。**
- 为什么这么设计：如果换后端就直接抛异常，会让同一段 DSL 代码只能锁死跑在一个后端上，违背"一套代码打多个后端"的初衷（这个初衷本身在第 6 节还要再检验）；`warnings.warn`（`python/sglang/lang/ir.py:67`、`python/sglang/lang/ir.py:81`、`python/sglang/lang/ir.py:96`、`python/sglang/lang/ir.py:111`）保留了"用户至少能跑通"这个优先级，把决定权交给外层日志配置。
- 不这样会怎样：静默降级的风险是 `regex` 这种直接影响输出正确性的约束被吞掉后，程序仍然"成功"跑完，但拿到的文本可能完全不满足格式要求——如果调用方没盯着 `warnings` 输出（生产环境常见配置：warnings 被重定向或过滤掉），这是一个真实的"看起来对但其实错"的坑。
- 什么时候可以不这样：只在单一固定后端（比如永远只用 `RuntimeEndpoint`）上跑的场景，这条代价根本不会触发，因为 `to_srt_kwargs()`（`python/sglang/lang/ir.py:121`）不丢任何字段；一旦决定要跨后端复用同一段 DSL 代码，就应该在 CI 里显式测每个用到的后端，而不是依赖警告——这正是当前仓库没做到的地方（§0.5、§7）。

**决策五：`tracer.py` 与 `interpreter.py` 是两套结构相似但语义不同的并行解释器。**
- 为什么这么设计：真正执行（`interpreter.py`）需要处理线程、事件同步、流式输出等一大堆运行时关注点；纯粹为了抽前缀（`tracer.py`）只需要"跑一遍控制流，把常量文本段落连起来，遇到第一个非常量节点就停"（`extract_prefix_by_tracing`，`python/sglang/lang/tracer.py:29`-`51`），两者的性能特征和正确性要求完全不同——硬塞进一套代码里会让"执行路径"里混入"仅供追踪用"的分支判断，增加核心执行路径的复杂度。
- 不这样会怎样：合并成一套意味着每次 `_execute` 都要判断"这是真执行还是假执行"，`StreamExecutor` 要多背一个 dry-run 标志位穿透到每个 `_execute_xxx` 方法里——`interpreter.py` 已经有十几个 `_execute_xxx` 方法，每个都要小心处理 dry-run 分支，比维护两个各自更短的类风险更高。
- 什么时候可以不这样：如果"追踪抽前缀"这个特性本身价值不大（`enable_precache_with_tracing` 完全可以设成 `False`，`python/sglang/lang/global_config.py:21`），维护两条并行逻辑的成本就没有对应收益——这正是第 7 节要给出的观察：`compiler.py`（属于这条追踪/编译线的一部分）已经在 2025-11 被整体删除，`tracer.py` 目前的调用者只剩 `cache_program`（`python/sglang/lang/interpreter.py:242`-`247`）和 `SglFunction.trace`（`python/sglang/lang/ir.py:304`），后者本身也没有在 `examples/`、`test/` 里搜到任何调用点。

**决策六：`join()` 默认只做 Python 变量收集，KV 级拼接要显式选 `mode="concate_and_append"`。**
- 为什么这么设计：绝大多数 fork 出去的场景（§4 的 `parallel_sample` 就是典型）分支之间是"平行采样、互不需要拼回一条序列"的关系——每个分支产出自己的答案，父状态只需要把这些答案收集成列表。默认给一个更"便宜"、不需要额外网络往返、任何后端都能满足的行为（`gather_variable` 纯操作 Python 字典，`python/sglang/lang/interpreter.py:1053`-`1066`），把"真的需要把几段并行生成的内容在服务端拼成一条连续序列"这种更少见、代价更高、且只有支持它的后端才能满足的需求，作为一个要显式声明的选项（`mode="concate_and_append"`）。
- 不这样会怎样：如果反过来把 KV 级拼接设成默认，`OpenAI`/`Anthropic` 等 `support_concate_and_append=False` 的后端（`python/sglang/lang/backend/base_backend.py:11`）就必须在每次 `join()` 时都退化成文本拼接（`python/sglang/lang/interpreter.py:729`-`736`），语义上等价但要多一层隐式判断，用户完全看不出"这次 join 到底有没有真的拼 KV"。
- 什么时候可以不这样：如果一个部署场景里**所有**分叉-汇合都是"多分支各自算一段、最后必须拼成一条连续上下文继续生成"（比如并行编码长文档的不同片段再拼起来续写），显式写 `mode="concate_and_append"` 反而比每次都要记得传参更省心；但目前看（第 7 节），这条路径连服务端接口都对不上，这一条"什么时候可以不这样"更多是设计意图上的假设，而非当前可用的现实选项。

**决策七："API 投机执行"是一个和 RadixAttention 完全独立的优化维度。**
- 为什么这么设计：RadixAttention 省的是"重复计算前缀的 GPU 时间"，对**本地** `RuntimeEndpoint` 后端才有意义；但打 OpenAI 这类远程 chat API 时，真正的瓶颈往往是**往返次数**本身（每次 HTTP 调用的排队、网络、计费都是独立成本），而这类 API 通常不暴露"续写同一段文本"的原始 completion 接口，只能整轮对话一次性提交。`num_api_spec_tokens`（`python/sglang/lang/ir.py:142`-`144`）这条机制用"一次多要一些 token，本地按 `stop` 条件切"的方式把同一个 `assistant` 块里的多个 `gen()` 合并成尽量少的请求，直接针对的是这个跟 KV 缓存无关的瓶颈。
- 不这样会怎样：不开这个模式，`gen_character_spec`（`examples/frontend_language/usage/openai_chat_speculative.py:23`-`24`）这类"一个 assistant 块里连续问名字/生日/工作"的函数，每个字段都要单独打一次 OpenAI API；调用次数直接和 `gen()` 调用次数成正比。
- 什么时候可以不这样：打本地 `RuntimeEndpoint`（`support_concate_and_append`/RadixAttention 生效的那条路径）时，HTTP 往返本身很便宜（同机或同机房），这条机制几乎没有收益，反而要承担"必须把所有 `gen` 塞进同一个 `assistant` 块"这条限制（`examples/frontend_language/usage/openai_chat_speculative.py:3`）；`num_api_spec_tokens` 参数留空（默认 `None`）就完全不触发这条代码路径（`python/sglang/lang/interpreter.py:597`）。

## 6. 同位对照

vLLM 没有对应的前端 DSL。`import vllm` 之后能拿到的顶层编程接口就是 `LLM` 类（`` `vllm:vllm/entrypoints/llm.py:67` ``）——`generate()`（`` `vllm:vllm/entrypoints/llm.py:418` ``）和 `chat()`（`` `vllm:vllm/entrypoints/llm.py:612` ``）的签名形状是：

```python
# vllm:vllm/entrypoints/llm.py:418-428（generate，节选签名）
def generate(
    self,
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    *,
    use_tqdm: bool | Callable[..., tqdm] = True,
    lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
    priority: list[int] | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> list[RequestOutput]:
```

`prompts` 直接标成 `PromptType | Sequence[PromptType]`——一个或一批，进去多少个 prompt 就出来多少个 `RequestOutput`，整个签名里没有"程序状态"“角色”“分支”这类概念的位置。`chat()`（`` `vllm:vllm/entrypoints/llm.py:612` ``）的 `messages` 参数同理是 `list[...] | Sequence[list[...]]`，一次调用打包好整段对话，不支持在多次调用之间维护一个可以被 DSL 语法追加的运行时对象。

都是**一次调用、一批 prompt 进、一批结果出**的批处理接口，没有 `fork`/`select`/`s +=` 这类可以在 Python 函数体内部动态编排出多分支、多轮"程序"的语法，也没有 `ProgramState` 这种承载"一次程序运行到一半的状态"的对象。要在 vLLM 上做"5 个分支各自续写再汇总"这种事，用户得自己写 Python 循环调 5 次 `generate()`，自己拼 prompt 字符串、自己保证这些字符串共享前缀。

（顺带排除一个容易搞混的名字：vLLM 仓库里确实有个 `vllm/ir/` 目录，但那是编译期算子注册/包装用的中间表示（`enable_torch_wrap`/`register_op` 等），和"提示语言"毫无关系——这是**本库推断**，依据是该目录只被 `vllm/compilation/` 一类模块引用，命名撞车但语义完全不同，提醒读者不要顺着字面意思联想。）

这个差异不代表 vLLM 完全没有"前缀复用"能力——vLLM 的自动前缀缓存（见 [[04-vLLM-KV缓存与前缀缓存]]）是**引擎侧**基于哈希表的能力，任意两次 `generate()` 调用只要 prompt 文本字节前缀相同就能命中，不需要任何"语言"编排。真正的差异在于：SGLang 的 DSL 是在**语言层面主动排布**出"哪些请求应该共享前缀"（few-shot 例子、fork 前的公共部分、`select` 候选的公共部分，全部被显式地放在同一次函数调用、同一个 `text_` 前缀之下），而 vLLM 把"是否共享前缀"完全交给调用方自己保证——两个不同脚本发出的两次 `generate()` 调用，只要 prompt 字符串凑巧前缀相同，一样能在 vLLM 的前缀缓存上命中，跟"有没有 DSL"无关。换句话说，DSL 解决的不是"能不能复用前缀"（两边引擎都能），而是"**让程序员几乎不用主动思考就自然写出可复用前缀的程序结构**"——`fork` 把"共享的部分只写一次"变成语法内建，`select` 把"多个候选打包成一批共前缀请求"变成一次函数调用。

这也部分解释了为什么"前端 DSL"这件事在两边工程里的重要性不对等：RadixAttention 这种以"任意长度前缀树匹配"为卖点的缓存机制，配一套天然制造树状共享结构的 DSL，收益是相乘的（见 [[03-SGLang-RadixAttention与前缀缓存]] §0 的树 vs 定长块对比）；vLLM 的前缀缓存对"请求是否来自同一次函数调用里天然的分支结构"没有额外敏感度，DSL 对它的增益边际更小。这一句"边际更小是 vLLM 没做同类 DSL 的原因之一"是**本库推断**，没有查到 vLLM 官方就此给出的正面表态，标注为未查证的动机推测，不作为定论。

再往回看 §1 提到的"三个入口"：`lang/`（DSL）、`Engine`（Python 批处理）、裸 HTTP server（OpenAI 兼容）在 SGLang 里是三条并列的路，而 vLLM 基本只有后两条的对应物——`LLM` 类（对应 `Engine`）和它自带的 OpenAI 兼容 server。§0.5、§7 已经用提交历史和 CI 归属证明了 SGLang 自己的 `lang/` 这条路正在往边缘退——如果把这条趋势线拉长看，"SGLang 有一条 vLLM 没有的路"和"这条路在 SGLang 内部的权重也在下降"这两件事同时成立，并不矛盾：DSL 曾经是 SGLang 论文里和 RadixAttention 并列的核心卖点，但仓库自己现在的资源分配（提交频率、CI 覆盖）显示，真正被持续投入的是 `srt`（尤其是它暴露的 OpenAI 兼容 HTTP 层）——这一条趋势判断本身也标注为**本库推断**，依据是 §0.5/§7 给出的确定性数据，但"资源分配"和"产品重要性"之间不能画等号，只是相关，不是全部证据。

## 7. 踩坑与反直觉

1. **`join(mode="concate_and_append")` 依赖的服务端接口，在当前 `srt` 上找不到**。`RuntimeEndpoint.concatenate_and_append()`（`python/sglang/lang/backend/runtime_endpoint.py:317`-`324`）请求路径是 `/concate_and_append_request`；把 `srt` 的 HTTP 入口 `python/sglang/srt/entrypoints/http_server.py`（2827 行）里所有 `@app.get`/`@app.post`/`@app.api_route` 装饰的路由整段列出来核对（从 `/health`（`python/sglang/srt/entrypoints/http_server.py:654`）到 `/v1/chat/completions`（`python/sglang/srt/entrypoints/http_server.py:1722`），一共几十条），没有任何一条路径叫这个名字或近似名字。换句话说，§3.7 讲的那条"显式拼接子分支 KV cache"的路径，在这份取证基准对应的服务端代码上调用即报错——这不是"功能小众没人测"，是**前后端接口已经对不上**。作为旁证：`srt` 现在有 `/open_session`（`python/sglang/srt/entrypoints/http_server.py:1573`）/`/close_session`（`python/sglang/srt/entrypoints/http_server.py:1587`）这一套显式 session 机制，`RuntimeEndpoint.generate()`/`async_generate()` 也确实带了 `session_id` 参数（`python/sglang/lang/backend/runtime_endpoint.py:464`-`483`、`python/sglang/lang/backend/runtime_endpoint.py:505`-`523`）——这套 session 机制是否是 `concate_and_append` 的替代品，**未查证**，没有找到直接说明两者关系的提交记录，只是两者同时存在于同一份快照里的事实，不作为因果结论。
2. **`compiler.py` 曾经存在，本篇取材的这一版仓库里已经没有了**。`_PLAN.md` 给出的线索清单里点名要看 `compiler.py`，但在 `15a43983` 这个快照里 `find python/sglang/lang -iname "*compil*"` 零结果——本地 `_src/sglang` 是 `--depth 1` 浅克隆没有历史，验不了"以前有没有"，所以改用 GitHub 提交历史 API 独立核实（`commits?path=python/sglang/lang/compiler.py`）：`compiler.py` 在提交 `e019f233`（2025-11-16，标题 "Remove unused code / testcases in `lang`"）里被整体删除（231 行，0 新增），同一提交还删掉了 `test/lang/test_anthropic_backend.py`（24 行）、`test/lang/test_tracing.py`（129 行）、`test/lang/test_vertexai_backend.py`（53 行），改了 `test/lang/run_suite.py`（把这些测试从发现列表里摘掉）。三天后的提交 `196b940a`（2025-11-19，"[3/N] CI refactor: move some manually triggered tests"）把剩下的 `test/lang/` 整体搬到 `test/manual/lang_frontend/`——这两步合起来是"DSL 的自测代码从 CI 自动发现区被移出"的完整过程，时间点在本篇取证基准之前约 9 个月。
3. **`SglFork`/`SglGetForkItem` 是"活在 IR 定义里、死在执行器外"的类**。它们在 `ir.py` 里定义齐全（`python/sglang/lang/ir.py:552`、`python/sglang/lang/ir.py:565`），`SglExpr.print_graph_dfs`（`python/sglang/lang/ir.py:383`）甚至专门为它们写了打印格式，但 `interpreter.py` 的 `_execute` 分发表（`python/sglang/lang/interpreter.py:461`）压根没有对应分支——真正跑 `@sgl.function` 程序时永远不会实例化它们。它们唯一的用武之地是 `python/sglang/lang/tracer.py:114`、`python/sglang/lang/tracer.py:123` 那套只服务于抽前缀的影子解释器。第一次读代码容易望文生义，以为 `fork()` 的运行时实现一定是走这两个 IR 节点。
4. **`BaseBackend.fork_program()` 是从未被调用过的钩子**。全仓 `grep "fork_program" python/sglang/` 只在 `python/sglang/lang/backend/base_backend.py:38`-`44` 的定义处命中一次，没有任何调用点，也没有任何后端子类覆写它。它看起来像是为"分叉这件事应该通知后端"预留的扩展点，但实际的 fork 完全在 `StreamExecutor` 内部用纯 Python 拷贝完成，从未触达这个钩子——真正"通知后端"的是 `SglCommitLazy`（经 `commit_lazy_operations`），但那是 `fork()` 内部主动多发的一次独立请求，不经过这个钩子。
5. **`select()` 里的 token-healing 修正比想象中隐蔽**。`RuntimeEndpoint.select()` 先用 `logprob_start_len = max(prompt_len - 2, 0)`（`python/sglang/lang/backend/runtime_endpoint.py:261`）多要两个 token 的 logprob，是为了让服务端的 token healing（把 prompt 末尾可能被过早截断的 token 重新展开）在打分时也能被看见；紧接着一段逻辑专门检测"如果 healing 实际没发生"就要把多算的那个 token 从归一化平均里减掉、并把它从 `input_token_logprobs` 里剔除（`python/sglang/lang/backend/runtime_endpoint.py:284`-`292`）——这段代码不读注释根本看不出它在干什么，是"两次不同来源的边界效应互相打架，用一段专门的算术把其中一次撤销"的典型例子。
6. **32 份几乎相同的 DSL 用户代码，全都不是在写"结构化生成程序"**。`test/registered/` 里能匹配到 `sgl.function` 的 32 个文件（`grep -rl "sgl\.function" test/registered/ | wc -l` 实测），无一例外是各硬件/各模型的精度评测脚本（`test_*_eval_*.py`），每个文件里都各自复制了一份几乎逐字相同的 `few_shot_gsm8k` 函数——一次 `sgl.gen`，没有分支、没有 `fork`、没有 `select`——用 `.run_batch(..., num_threads=parallel, progress_bar=True)` 纯粹当"自带线程池和进度条的并发 HTTP 客户端"用。这批用户从未触达 `fork`/`select`/chat 角色构造等本篇讲的大部分机制——`test/registered/` 里 `.fork(` 和 `sgl.select(` 的命中数都是 0。这提示一个容易被忽略的落差：仓库文档和 `examples/` 里演示的"结构化编程"能力，和这套 DSL 在仓库自己 CI 里实际被使用的方式，是两回事。
7. **"API 投机执行"这条机制，示例齐全但测试为零**。`examples/frontend_language/usage/openai_chat_speculative.py`、`examples/frontend_language/usage/openai_speculative.py` 把用法讲得很细（包括"必须把 `gen` 都塞进同一个 `assistant` 块"这条容易踩的坑），但 `grep -rl "num_api_spec_tokens" test/` 在整个 `test/` 目录（含 `registered/` 和 `manual/`）零命中——这条机制没有任何自动化测试兜底，连"非 CI 但至少手动能跑"的覆盖都没有，比 §7.6 提到的 `fork`/`select` 处境更边缘。
8. **纯"续写"风格的代码打到 `Anthropic` 后端会被悄悄改写成一轮 chat**。`Anthropic.generate()`（`python/sglang/lang/backend/anthropic.py:26`-`49`）不管 `s.messages_` 是不是空的都要凑出一个 `messages` 列表：如果程序里从来没写过 `sgl.user()`/`sgl.assistant()`（纯 `s += "..." + sgl.gen(...)` 的老式续写写法），会把整段 `s.text_` 包成 `[{"role": "user", "content": s.text_}]`（`python/sglang/lang/backend/anthropic.py:31`-`34`）当一轮用户消息发出去，而不是像 `RuntimeEndpoint`/`OpenAI`（非 chat 模式）那样把 `s.text_` 当成一段要续写的裸文本。同一段"续写风格"的 DSL 代码，换到 `Anthropic` 后端后，模型看到的输入结构其实变了，只是这层转换完全在 `lang/` 内部悄悄完成，用户很难从 DSL 代码本身看出来。

**附：本篇几处规模/活跃度数字的复现命令**（`_src/sglang` 是浅克隆，前两组本地就能跑；第三组要打 GitHub API，网络请求需脱离沙箱执行）：

```bash
# lang/ 与 srt/ 的文件数、行数（§2、§0.5）
find python/sglang/lang -name '*.py' | xargs wc -l | tail -1     # 4644 total
find python/sglang/srt  -name '*.py' | wc -l                     # 1646
find python/sglang/srt  -name '*.py' | xargs wc -l | tail -1     # 89798 total

# test/registered/ 里 DSL 用户代码的分布（§7.6）
grep -rl "sgl\.function" test/registered/ | wc -l                # 32
grep -rl "\.fork(\|sgl\.select(" test/registered/ | wc -l        # 0

# 90 天窗口内触达 lang/ 与 srt/ 的提交数（§0.5，today = 2026-08-22）
curl -sI "https://api.github.com/repos/sgl-project/sglang/commits?path=python/sglang/lang&since=2026-05-24T00:00:00Z&per_page=1" | grep -i '^link'
curl -sI "https://api.github.com/repos/sgl-project/sglang/commits?path=python/sglang/srt&since=2026-05-24T00:00:00Z&per_page=1"  | grep -i '^link'
# Link 响应头里 rel="last" 的 page 号就是命中的提交数：lang/ 是 6，srt/ 是 2339
```

## 8. 可改进点

1. **`concate_and_append` 这条路径要么修好要么删掉**。§7.1 已经证实 `RuntimeEndpoint.concatenate_and_append()` 请求的 `/concate_and_append_request` 在当前 `srt` 的 HTTP 路由表里不存在——现状是"用户完全没法预先知道这条路径已经打不通，只有实际调用 `join(mode="concate_and_append")` 才会报错"。要么在 `srt` 里补上对应路由（如果这个能力还有需求），要么把 `SglConcateAndAppend`/`_execute_concatenate_and_append_kv_cache`/`concatenate_and_append` 这条链路整体标记废弃或删除，避免看代码的人以为它是一条能用的路。
2. **给 `lang/` 补一条哪怕最小的 CI 覆盖**。当前 `test/manual/lang_frontend/` 下 6 个测试文件（`test_bind_cache.py`/`test_choices.py`/`test_jump_forward.py`/`test_openai_backend.py`/`test_separate_reasoning.py`/`test_separate_reasoning_execution.py`）都标注为"非 CI、仅手动触发"（`test/README.md:15`）——这意味着任何一次 `interpreter.py`/`ir.py`/后端文件的改动，理论上可以在不破坏 CI 的情况下悄悄改坏 `select()` 的打分逻辑或 `fork()` 的状态拷贝，而没有任何自动信号。哪怕只把 `test_choices.py` 这类不依赖起真实 server、能用 mock 跑的测试挂回 `test/registered/`，也能补上这个盲区。
3. **多后端 `select()` 语义差异需要一处集中的文档说明**。`RuntimeEndpoint` 用长度归一化 logprob，`OpenAI` 用 `logit_bias` 贪心逐 token 匹配且忽略 `choices_method`（这句话目前只写在 `python/sglang/lang/backend/openai.py:319` 一行 docstring 里），`Anthropic`/`LiteLLM`/`VertexAI` 直接不支持——用户从 `api.py`/`choices.py` 的公开接口完全看不出这个落差，只有翻到每个后端实现内部才知道。给 `choices_method` 参数的文档（或者 `gen(choices=...)` 本身的 docstring）加一句"不同后端行为不同"的提示，成本很低。
4. **`fork_program` 这个从未被调用的钩子该删或该补**。要么承认"通知后端有分叉发生"这件事目前没有任何后端需要（因为分叉纯靠文本前缀隐式复用，不需要显式 session），把这个方法从 `BaseBackend` 里删掉，减少读者对"这个钩子应该干什么"的困惑；要么如果未来想让某个后端（比如支持显式 session 的自定义部署）在 fork 时做点什么，把它真正接进 `StreamExecutor.fork()` 的调用路径里。当前"定义了但没人用也没人删"是最差选项。
5. **几处"换后端就不一样"的硬限制应该被集中列出来，而不是散落在各自实现里**。`RuntimeEndpoint` 一次请求最多带一张图（`python/sglang/lang/backend/runtime_endpoint.py:339`，§3.11）、`OpenAI` 后端的 `dtype` 约束和 `select()` 都不支持 chat 模型（`python/sglang/lang/backend/openai.py:183`-`185`、`python/sglang/lang/backend/openai.py:320`-`324`，§3.12/§0.3）、chat 模型不开投机执行时一个 `assistant` 块只能有一个 `gen()`（`python/sglang/lang/backend/openai.py:149`-`154`，§3.12）——这些限制目前只能通过运行时抛出的 `AssertionError`/`RuntimeError`/`NotImplementedError` 发现，没有一处文档把它们汇总成一张"后端能力矩阵"。哪怕只是在 `BaseBackend` 的类文档字符串里加一张这样的表，也能省下不少"跑到一半才发现这条路走不通"的调试时间。
6. **`tracer.py` 这条并行解释器线的价值应该被重新评估**。`compiler.py` 已经因为"unused"被删过一次，`tracer.py` 目前唯二的调用方（`cache_program`/`SglFunction.trace`）里，前者服务的 `enable_precache_with_tracing` 默认开着但触发条件很窄（只在 `run_batch` 且 `len(batch_arguments) > 1` 时触发，`python/sglang/lang/interpreter.py:106`-`107`），后者（`.trace()`）在 `examples/`、`test/` 里没有搜到任何调用。如果 `.trace()` 已经名存实亡，应该考虑把 `extract_prefix_by_tracing` 这一半单独抽出来（它是 `cache_program` 唯一真正依赖的部分），把 `TracerProgramState` 里为通用追踪保留的那些方法一起清掉，避免"两套解释器都要维护"的成本继续背下去。

## 9. 自测题与延伸阅读

**自测题**（闭卷回答）：

1. `@sgl.function` 装饰一个函数之后，函数体是在装饰的那一刻执行，还是在 `.run()` 调用时执行？如果是后者，是在调用者线程上同步执行，还是在某条后台线程上异步执行？
2. `s.fork(5)` 返回的 5 个分支，它们各自的 `text_` 字段是"5 个指向同一份字符串对象的引用"还是"5 份独立的字符串拷贝"？这个选择对后续每个分支各自 `+=` 不同内容有什么影响？
3. `SglFork`/`SglGetForkItem` 这两个 IR 节点类，在真正执行 `@sgl.function` 程序（不是追踪抽前缀）的路径上会不会被实例化？为什么？
4. `sgl.gen("tool", choices=["calculator", "browser"])` 返回的是 `SglGen` 还是 `SglSelect`？决定走哪个分支的判断条件写在哪个文件的哪一行？
5. `RuntimeEndpoint.select()` 打分时，是所有候选文本各自单独发一次请求，还是把所有候选打包成一次批量请求？这个选择跟 radix 树命中率有什么关系？
6. 如果把 `sgl.set_default_backend` 从 `RuntimeEndpoint` 换成 `Anthropic`，`parallel_sample` 这个例子里 `sgl.gen("tool", choices=[...])` 那一行会发生什么？
7. `test/registered/` 里 32 份精度评测脚本用这套 DSL 做什么？它们有没有用到 `fork` 或 `select`？这说明了什么？
8. `forks.join()` 不传参数时走哪种模式？如果换成 `join(mode="concate_and_append")`，在这份取证基准对应的 `srt` 上会发生什么？
9. "API 投机执行"（`num_api_spec_tokens`）省的是 GPU 计算时间还是 HTTP 往返次数？它和 RadixAttention 省的是不是同一种成本？
10. `gen(dtype=int)` 和 `gen(choices=[...])`（即 `select()`）都会限制模型的输出，但限制的机制是同一种吗？哪一个在 `srt` 侧走的是真正的逐 token 约束解码？
11. 同一段只用 `s += "..." + sgl.gen(...)`、从未调用过 `sgl.user()`/`sgl.assistant()` 的"续写风格"代码，分别打到 `RuntimeEndpoint` 和 `Anthropic` 后端，模型实际看到的输入结构一样吗？

**延伸阅读**：

- [[03-SGLang-RadixAttention与前缀缓存]] —— 本篇讲"前端怎么制造出共享前缀"，这一篇讲"服务端的树怎么匹配、分裂、驱逐这些前缀"，两篇合起来才是完整的因果链。
- [[01-SGLang-全景与代码地图]] —— `lang/` 在整个仓库里的位置坐标，以及它和 `srt/` 各子系统的边界总览。
- [[09-vLLM-Python-API与EngineArgs]] —— 对照 vLLM 那条"只有 `LLM` 类、没有 DSL"的路径，理解 §6 的差异到底差在哪。
- [[10-SGLang-投机解码EAGLE]] —— 名字都带"投机"，但机制完全不同：那一篇讲的是服务端草稿模型级别的投机解码（省 GPU 计算），本篇 §5 决策七讲的"API 投机执行"省的是客户端到远程 API 的 HTTP 往返次数，两者不要混为一谈。
