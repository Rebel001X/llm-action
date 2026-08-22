# SGLang 前端 DSL 与编程模型解剖

> **本篇取证基准**：`sglang` @ `15a43983`（2026-08-22）
> **一句话**：前端把提示排成共享前缀，后端的树才有命中可捡。

## 0. 结论先行

1. **"SG" = Structured Generation，这套 DSL 的对象模型是一棵 `SglExpr` 子类构成的表达式节点集合**：`SglConstantText`/`SglGen`/`SglSelect`/`SglRoleBegin`/`SglFork`/`SglVariable`… 全部定义在 `python/sglang/lang/ir.py`（`SglExpr` 基类 `python/sglang/lang/ir.py:327`，`SglGen` `python/sglang/lang/ir.py:451`，`SglSelect` `python/sglang/lang/ir.py:533`）。但 `@sgl.function` 包出来的 Python 函数**不是"先建完整棵树再统一解释"**——它在调用时立刻用真实的 Python 控制流（`if`/`for`/递归）跑一遍函数体（`python/sglang/lang/interpreter.py:44`），每次 `s += expr` 只是把当前这一个 IR 节点丢进一个队列（`python/sglang/lang/interpreter.py:342`），由**另一条专门的后台线程**异步消费执行（`python/sglang/lang/interpreter.py:422`）。这是生产者-消费者模型，不是"编译再运行"。
2. **`fork()` 根本不走 IR 节点队列，是纯 Python 值拷贝**。`ir.py` 里定义了 `SglFork`/`SglGetForkItem` 两个 IR 节点类（`python/sglang/lang/ir.py:552`、`python/sglang/lang/ir.py:565`），但 `interpreter.py` 的执行分发表（`_execute` 方法，`python/sglang/lang/interpreter.py:461`）里没有任何一个分支处理它们——全仓 `grep` 证实这两个类只被 `python/sglang/lang/tracer.py:114`、`python/sglang/lang/tracer.py:123` 引用，即只服务于静态前缀追踪子系统。真正跑起来的 `StreamExecutor.fork()`（`python/sglang/lang/interpreter.py:370`）直接 `dict(self.variables)`/`str(self.text_)` 做值拷贝，`BaseBackend.fork_program()` 这个专门为"通知后端有分叉发生"留的钩子（`python/sglang/lang/backend/base_backend.py:38`）在全仓库里从未被任何代码调用过。
3. **`select()` 的默认打分口径是"长度归一化的平均 log-prob 取 argmax"，不是约束解码**：`TokenLengthNormalized.__call__`（`python/sglang/lang/choices.py:34`）直接 `np.argmax(normalized_prompt_logprobs)`，而 `normalized_prompt_logprobs` 由 `RuntimeEndpoint.select()` 通过 `compute_normalized_prompt_logprobs`（`python/sglang/lang/backend/runtime_endpoint.py:351`）算出——就是输入 token logprob 的算术平均。且这个语义只在 `RuntimeEndpoint`（打到 SGLang 自家 server）上完整成立；换到 `OpenAI` 后端，`select()` 整个换了一套算法：靠 `logit_bias` 强制逐 token 贪心比对、用命中 token 数当分数，且明确声明忽略 `choices_method`（`python/sglang/lang/backend/openai.py:319`）；`Anthropic`/`LiteLLM`/`VertexAI` 三个后端根本没有覆写 `select()`，调用即 `NotImplementedError`（继承自 `python/sglang/lang/backend/base_backend.py:70`）。
4. **DSL 与 RadixAttention 是一条完整因果链，不是两个独立特性凑在一起**：`fork()` 在真正分叉前会先把已经攒好的文本原样发一次 `max_new_tokens=0` 的 `/generate`（`StreamExecutor.fork` 里的 `SglCommitLazy` 提交，`python/sglang/lang/interpreter.py:376`，落到 `RuntimeEndpoint.commit_lazy_operations` `python/sglang/lang/backend/runtime_endpoint.py:105`），把共享前缀预先"写"进服务端的前缀缓存；随后每个分支各自独立发请求，因为它们的 `text_` 字节级相同，服务端的 radix 树天然把它们匹配到同一个前缀节点上。`select()` 更直接：一次 HTTP 请求里打包 `[s.text_ + c for c in choices]`（`python/sglang/lang/backend/runtime_endpoint.py:265`）——这一批 prompt 共享同一个前缀字符串，radix 树见到的是"一批带共同前缀的请求"，不是"几个互不相干的请求"。`run_batch()` 默认还会先做一次纯静态追踪把公共前缀抽出来预热缓存（`enable_precache_with_tracing` 默认 `True`，`python/sglang/lang/global_config.py:21`；调用点 `python/sglang/lang/interpreter.py:106`）。第 4、6 节会把这条链逐步拆开。
5. **现状：这套 DSL 在仓库里活着，但已经退到"维护模式"**——90 天窗口内（2026-05-24 至今）`python/sglang/lang/` 只有 6 次提交触达，`python/sglang/srt/` 同期有 2339 次（GitHub 提交历史 API 实测，本地 `_src` 是 `--depth 1` 浅克隆没有历史，方法见 §7）；2025-11-16 的提交 `e019f233` 整体删除了 `python/sglang/lang/compiler.py`（231 行，标题为 "Remove unused code / testcases in `lang`"），三天后 `196b940a` 把 `lang` 的测试从 CI 自动发现的 `test/lang/` 迁到"非 CI、仅手动触发"的 `test/manual/lang_frontend/`（`test/README.md:15`）。如今仓库自己的 CI（`test/registered/`）里唯一还在用这套 DSL 的地方，是 32 个精度评测文件里各自复制粘贴的同一段 `few_shot_gsm8k` 批量推理套路（`@sgl.function` + `.run_batch()`），`fork`/`select` 在 `test/registered/` 里出现次数为 0。

## 1. 它在系统里的位置

`python/sglang/lang/` 是一个**纯客户端库**：它不在 `srt`（SGLang Runtime，实际跑模型、管调度器、管 KV 缓存的那部分）进程里执行，本身也不 import `srt` 的核心模块——全仓 `grep "from sglang.lang" python/sglang/srt/` 零命中。它和 `srt` 之间只有一条通路：HTTP。

```
用户 Python 脚本
   │  @sgl.function 装饰、s += expr、s.fork()/select()…
   ▼
python/sglang/lang/            ← 本篇范围：纯 Python，不依赖 torch/CUDA
   │  ir.py          —— IR 节点定义（SglExpr 及其子类）
   │  interpreter.py —— 真正跑程序的地方：StreamExecutor + ProgramState
   │  tracer.py      —— 另一套只做"静态追踪抽前缀"的影子解释器
   │  backend/       —— 把 IR 节点翻译成对应后端的请求
   ▼
BaseBackend 子类之一（python/sglang/lang/backend/base_backend.py:9）
   ├── RuntimeEndpoint —— HTTP 打到 SGLang 自家 srt server（本地或远程）
   └── OpenAI / Anthropic / LiteLLM / VertexAI / Crusoe —— HTTP 打到第三方 API
   ▼
（只有 RuntimeEndpoint 这条路径才会走到）
python/sglang/srt/ —— 调度器 + RadixAttention + KV 缓存 + 模型执行
```

这个位置关系在源码里有一句非常直白的自白：`Runtime` 类（`python/sglang/lang/backend/runtime_endpoint.py:356`）的文档字符串写着——

> "It is mainly used for the frontend language. You should use the Engine class if you want to do normal offline processing without the frontend language."（`python/sglang/lang/backend/runtime_endpoint.py:362`-`363`）

也就是说，SGLang 自己把"要不要用这套 DSL"当成了一个**入口选择**：走 `Runtime`/`RuntimeEndpoint` 就是在用 DSL；走 `Engine`（`python/sglang/srt/entrypoints/engine.py`，由 `python/sglang/lang/api.py:44` 懒加载）就是"正常的离线处理"，完全绕开 `lang/`。`python/sglang/__init__.py:26` 起的 import 块导出 `Runtime`，`python/sglang/__init__.py:68` 用 `LazyImport` 导出 `Engine`——两者是并列导出的两个入口，选哪个由用户决定，不是包含关系。

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
- 什么时候可以不这样：候选数量少、长度接近时，三种 `ChoicesSamplingMethod` 往往选出同一个答案，这时用哪种口径几乎不影响结果；`UnconditionalLikelihoodNormalized`（`python/sglang/lang/choices.py:110`-`161`）专门处理"某个候选本身在语言模型里天然更常见"这类先验偏置问题，需要额外一次"不带 prompt"的 logprob 请求，只有候选之间先验概率差异明显大时才值得多付这次请求的代价。

**决策四：参数不兼容时"警告后丢弃"而非"直接报错"——`regex`/`dtype` 在四个远程后端上被静默忽略。**
- 为什么这么设计：如果换后端就直接抛异常，会让同一段 DSL 代码只能锁死跑在一个后端上，违背"一套代码打多个后端"的初衷（这个初衷本身在第 6 节还要再检验）；`warnings.warn`（`python/sglang/lang/ir.py:67`、`python/sglang/lang/ir.py:81`、`python/sglang/lang/ir.py:96`、`python/sglang/lang/ir.py:111`）保留了"用户至少能跑通"这个优先级，把决定权交给外层日志配置。
- 不这样会怎样：静默降级的风险是 `regex` 这种直接影响输出正确性的约束被吞掉后，程序仍然"成功"跑完，但拿到的文本可能完全不满足格式要求——如果调用方没盯着 `warnings` 输出（生产环境常见配置：warnings 被重定向或过滤掉），这是一个真实的"看起来对但其实错"的坑。
- 什么时候可以不这样：只在单一固定后端（比如永远只用 `RuntimeEndpoint`）上跑的场景，这条代价根本不会触发，因为 `to_srt_kwargs()`（`python/sglang/lang/ir.py:121`）不丢任何字段；一旦决定要跨后端复用同一段 DSL 代码，就应该在 CI 里显式测每个用到的后端，而不是依赖警告——这正是当前仓库没做到的地方（§0.5、§7）。

**决策五：`tracer.py` 与 `interpreter.py` 是两套结构相似但语义不同的并行解释器。**
- 为什么这么设计：真正执行（`interpreter.py`）需要处理线程、事件同步、流式输出等一大堆运行时关注点；纯粹为了抽前缀（`tracer.py`）只需要"跑一遍控制流，把常量文本段落连起来，遇到第一个非常量节点就停"（`extract_prefix_by_tracing`，`python/sglang/lang/tracer.py:29`-`51`），两者的性能特征和正确性要求完全不同——硬塞进一套代码里会让"执行路径"里混入"仅供追踪用"的分支判断，增加核心执行路径的复杂度。
- 不这样会怎样：合并成一套意味着每次 `_execute` 都要判断"这是真执行还是假执行"，`StreamExecutor` 要多背一个 dry-run 标志位穿透到每个 `_execute_xxx` 方法里——`interpreter.py` 已经有十几个 `_execute_xxx` 方法，每个都要小心处理 dry-run 分支，比维护两个各自更短的类风险更高。
- 什么时候可以不这样：如果"追踪抽前缀"这个特性本身价值不大（`enable_precache_with_tracing` 完全可以设成 `False`，`python/sglang/lang/global_config.py:21`），维护两条并行逻辑的成本就没有对应收益——这正是第 7 节要给出的观察：`compiler.py`（属于这条追踪/编译线的一部分）已经在 2025-11 被整体删除，`tracer.py` 目前的调用者只剩 `cache_program`（`python/sglang/lang/interpreter.py:242`-`247`）和 `SglFunction.trace`（`python/sglang/lang/ir.py:304`），后者本身也没有在 `examples/`、`test/` 里搜到任何调用点。

## 6. 同位对照

vLLM 没有对应的前端 DSL。`import vllm` 之后能拿到的顶层编程接口就是 `LLM` 类（`` `vllm:vllm/entrypoints/llm.py:67` ``）——`generate()`（`` `vllm:vllm/entrypoints/llm.py:418` ``）和 `chat()`（`` `vllm:vllm/entrypoints/llm.py:612` ``）都是**一次调用、一批 prompt 进、一批结果出**的批处理接口，没有 `fork`/`select`/`s +=` 这类可以在 Python 函数体内部动态编排出多分支、多轮"程序"的语法。要在 vLLM 上做"5 个分支各自续写再汇总"这种事，用户得自己写 Python 循环调 5 次 `generate()`，自己拼 prompt 字符串、自己保证这些字符串共享前缀。

（顺带排除一个容易搞混的名字：vLLM 仓库里确实有个 `vllm/ir/` 目录，但那是编译期算子注册/包装用的中间表示（`enable_torch_wrap`/`register_op` 等），和"提示语言"毫无关系——这是**本库推断**，依据是该目录只被 `vllm/compilation/` 一类模块引用，命名撞车但语义完全不同，提醒读者不要顺着字面意思联想。）

这个差异不代表 vLLM 完全没有"前缀复用"能力——vLLM 的自动前缀缓存（见 [[04-vLLM-KV缓存与前缀缓存]]）是**引擎侧**基于哈希表的能力，任意两次 `generate()` 调用只要 prompt 文本字节前缀相同就能命中，不需要任何"语言"编排。真正的差异在于：SGLang 的 DSL 是在**语言层面主动排布**出"哪些请求应该共享前缀"（few-shot 例子、fork 前的公共部分、`select` 候选的公共部分，全部被显式地放在同一次函数调用、同一个 `text_` 前缀之下），而 vLLM 把"是否共享前缀"完全交给调用方自己保证——两个不同脚本发出的两次 `generate()` 调用，只要 prompt 字符串凑巧前缀相同，一样能在 vLLM 的前缀缓存上命中，跟"有没有 DSL"无关。换句话说，DSL 解决的不是"能不能复用前缀"（两边引擎都能），而是"**让程序员几乎不用主动思考就自然写出可复用前缀的程序结构**"——`fork` 把"共享的部分只写一次"变成语法内建，`select` 把"多个候选打包成一批共前缀请求"变成一次函数调用。

这也部分解释了为什么"前端 DSL"这件事在两边工程里的重要性不对等：RadixAttention 这种以"任意长度前缀树匹配"为卖点的缓存机制，配一套天然制造树状共享结构的 DSL，收益是相乘的（见 [[03-SGLang-RadixAttention与前缀缓存]] §0 的树 vs 定长块对比）；vLLM 的前缀缓存对"请求是否来自同一次函数调用里天然的分支结构"没有额外敏感度，DSL 对它的增益边际更小。这一句"边际更小是 vLLM 没做同类 DSL 的原因之一"是**本库推断**，没有查到 vLLM 官方就此给出的正面表态，标注为未查证的动机推测，不作为定论。

## 7. 踩坑与反直觉

1. **`compiler.py` 曾经存在，本篇取材的这一版仓库里已经没有了**。`_PLAN.md` 给出的线索清单里点名要看 `compiler.py`，但在 `15a43983` 这个快照里 `find python/sglang/lang -iname "*compil*"` 零结果——本地 `_src/sglang` 是 `--depth 1` 浅克隆没有历史，验不了"以前有没有"，所以改用 GitHub 提交历史 API 独立核实（`commits?path=python/sglang/lang/compiler.py`）：`compiler.py` 在提交 `e019f233`（2025-11-16，标题 "Remove unused code / testcases in `lang`"）里被整体删除（231 行，0 新增），同一提交还删掉了 `test/lang/test_anthropic_backend.py`（24 行）、`test/lang/test_tracing.py`（129 行）、`test/lang/test_vertexai_backend.py`（53 行），改了 `test/lang/run_suite.py`（把这些测试从发现列表里摘掉）。三天后的提交 `196b940a`（2025-11-19，"[3/N] CI refactor: move some manually triggered tests"）把剩下的 `test/lang/` 整体搬到 `test/manual/lang_frontend/`——这两步合起来是"DSL 的自测代码从 CI 自动发现区被移出"的完整过程，时间点在本篇取证基准之前约 9 个月。
2. **`SglFork`/`SglGetForkItem` 是"活在 IR 定义里、死在执行器外"的类**。它们在 `ir.py` 里定义齐全（`python/sglang/lang/ir.py:552`、`python/sglang/lang/ir.py:565`），`SglExpr.print_graph_dfs`（`python/sglang/lang/ir.py:383`）甚至专门为它们写了打印格式，但 `interpreter.py` 的 `_execute` 分发表（`python/sglang/lang/interpreter.py:461`）压根没有对应分支——真正跑 `@sgl.function` 程序时永远不会实例化它们。它们唯一的用武之地是 `python/sglang/lang/tracer.py:114`、`python/sglang/lang/tracer.py:123` 那套只服务于抽前缀的影子解释器。第一次读代码容易望文生义，以为 `fork()` 的运行时实现一定是走这两个 IR 节点。
3. **`BaseBackend.fork_program()` 是从未被调用过的钩子**。全仓 `grep "fork_program" python/sglang/` 只在 `python/sglang/lang/backend/base_backend.py:38`-`44` 的定义处命中一次，没有任何调用点，也没有任何后端子类覆写它。它看起来像是为"分叉这件事应该通知后端"预留的扩展点，但实际的 fork 完全在 `StreamExecutor` 内部用纯 Python 拷贝完成，从未触达这个钩子——真正"通知后端"的是 `SglCommitLazy`（经 `commit_lazy_operations`），但那是 `fork()` 内部主动多发的一次独立请求，不经过这个钩子。
4. **`select()` 里的 token-healing 修正比想象中隐蔽**。`RuntimeEndpoint.select()` 先用 `logprob_start_len = max(prompt_len - 2, 0)`（`python/sglang/lang/backend/runtime_endpoint.py:261`）多要两个 token 的 logprob，是为了让服务端的 token healing（把 prompt 末尾可能被过早截断的 token 重新展开）在打分时也能被看见；紧接着一段逻辑专门检测"如果 healing 实际没发生"就要把多算的那个 token 从归一化平均里减掉、并把它从 `input_token_logprobs` 里剔除（`python/sglang/lang/backend/runtime_endpoint.py:284`-`292`）——这段代码不读注释根本看不出它在干什么，是"两次不同来源的边界效应互相打架，用一段专门的算术把其中一次撤销"的典型例子。
5. **32 份几乎相同的 DSL 用户代码，全都不是在写"结构化生成程序"**。`test/registered/` 里能匹配到 `sgl.function` 的 32 个文件（`grep -rl "sgl\.function" test/registered/ | wc -l` 实测），无一例外是各硬件/各模型的精度评测脚本（`test_*_eval_*.py`），每个文件里都各自复制了一份几乎逐字相同的 `few_shot_gsm8k` 函数——一次 `sgl.gen`，没有分支、没有 `fork`、没有 `select`——用 `.run_batch(..., num_threads=parallel, progress_bar=True)` 纯粹当"自带线程池和进度条的并发 HTTP 客户端"用。这批用户从未触达 `fork`/`select`/chat 角色构造等本篇讲的大部分机制——`test/registered/` 里 `.fork(` 和 `sgl.select(` 的命中数都是 0。这提示一个容易被忽略的落差：仓库文档和 `examples/` 里演示的"结构化编程"能力，和这套 DSL 在仓库自己 CI 里实际被使用的方式，是两回事。

## 8. 可改进点

1. **给 `lang/` 补一条哪怕最小的 CI 覆盖**。当前 `test/manual/lang_frontend/` 下 6 个测试文件（`test_bind_cache.py`/`test_choices.py`/`test_jump_forward.py`/`test_openai_backend.py`/`test_separate_reasoning.py`/`test_separate_reasoning_execution.py`）都标注为"非 CI、仅手动触发"（`test/README.md:15`）——这意味着任何一次 `interpreter.py`/`ir.py`/后端文件的改动，理论上可以在不破坏 CI 的情况下悄悄改坏 `select()` 的打分逻辑或 `fork()` 的状态拷贝，而没有任何自动信号。哪怕只把 `test_choices.py` 这类不依赖起真实 server、能用 mock 跑的测试挂回 `test/registered/`，也能补上这个盲区。
2. **多后端 `select()` 语义差异需要一处集中的文档说明**。`RuntimeEndpoint` 用长度归一化 logprob，`OpenAI` 用 `logit_bias` 贪心逐 token 匹配且忽略 `choices_method`（这句话目前只写在 `python/sglang/lang/backend/openai.py:319` 一行 docstring 里），`Anthropic`/`LiteLLM`/`VertexAI` 直接不支持——用户从 `api.py`/`choices.py` 的公开接口完全看不出这个落差，只有翻到每个后端实现内部才知道。给 `choices_method` 参数的文档（或者 `gen(choices=...)` 本身的 docstring）加一句"不同后端行为不同"的提示，成本很低。
3. **`fork_program` 这个从未被调用的钩子该删或该补**。要么承认"通知后端有分叉发生"这件事目前没有任何后端需要（因为分叉纯靠文本前缀隐式复用，不需要显式 session），把这个方法从 `BaseBackend` 里删掉，减少读者对"这个钩子应该干什么"的困惑；要么如果未来想让某个后端（比如支持显式 session 的自定义部署）在 fork 时做点什么，把它真正接进 `StreamExecutor.fork()` 的调用路径里。当前"定义了但没人用也没人删"是最差选项。
4. **`tracer.py` 这条并行解释器线的价值应该被重新评估**。`compiler.py` 已经因为"unused"被删过一次，`tracer.py` 目前唯二的调用方（`cache_program`/`SglFunction.trace`）里，前者服务的 `enable_precache_with_tracing` 默认开着但触发条件很窄（只在 `run_batch` 且 `len(batch_arguments) > 1` 时触发，`python/sglang/lang/interpreter.py:106`-`107`），后者（`.trace()`）在 `examples/`、`test/` 里没有搜到任何调用。如果 `.trace()` 已经名存实亡，应该考虑把 `extract_prefix_by_tracing` 这一半单独抽出来（它是 `cache_program` 唯一真正依赖的部分），把 `TracerProgramState` 里为通用追踪保留的那些方法一起清掉，避免"两套解释器都要维护"的成本继续背下去。

## 9. 自测题与延伸阅读

**自测题**（闭卷回答）：

1. `@sgl.function` 装饰一个函数之后，函数体是在装饰的那一刻执行，还是在 `.run()` 调用时执行？如果是后者，是在调用者线程上同步执行，还是在某条后台线程上异步执行？
2. `s.fork(5)` 返回的 5 个分支，它们各自的 `text_` 字段是"5 个指向同一份字符串对象的引用"还是"5 份独立的字符串拷贝"？这个选择对后续每个分支各自 `+=` 不同内容有什么影响？
3. `SglFork`/`SglGetForkItem` 这两个 IR 节点类，在真正执行 `@sgl.function` 程序（不是追踪抽前缀）的路径上会不会被实例化？为什么？
4. `sgl.gen("tool", choices=["calculator", "browser"])` 返回的是 `SglGen` 还是 `SglSelect`？决定走哪个分支的判断条件写在哪个文件的哪一行？
5. `RuntimeEndpoint.select()` 打分时，是所有候选文本各自单独发一次请求，还是把所有候选打包成一次批量请求？这个选择跟 radix 树命中率有什么关系？
6. 如果把 `sgl.set_default_backend` 从 `RuntimeEndpoint` 换成 `Anthropic`，`parallel_sample` 这个例子里 `sgl.gen("tool", choices=[...])` 那一行会发生什么？
7. `test/registered/` 里 32 份精度评测脚本用这套 DSL 做什么？它们有没有用到 `fork` 或 `select`？这说明了什么？

**延伸阅读**：

- [[03-SGLang-RadixAttention与前缀缓存]] —— 本篇讲"前端怎么制造出共享前缀"，这一篇讲"服务端的树怎么匹配、分裂、驱逐这些前缀"，两篇合起来才是完整的因果链。
- [[01-SGLang-全景与代码地图]] —— `lang/` 在整个仓库里的位置坐标，以及它和 `srt/` 各子系统的边界总览。
- [[09-vLLM-Python-API与EngineArgs]] —— 对照 vLLM 那条"只有 `LLM` 类、没有 DSL"的路径，理解 §6 的差异到底差在哪。
