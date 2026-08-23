# 把它变成一个HTTP服务

> **一句话**：HTTP 是并发的，引擎是单线程的，中间要靠两道队列缝合。
> **前置**：`[[06-采样-停止条件-流式输出]]`（本篇 `Job` 携带的采样器、`StreamGuard`、停止原因都在那一篇定义）、`[[04-连续批处理]]`（本篇的引擎线程每一步调用的仍然是那个 `step()`）

## 0. 这一篇要解决什么问题

从 `[[01-第一版-朴素前向]]` 到 `[[06-采样-停止条件-流式输出]]`，六篇文章加起来做成的东西，本质上只是**一个函数**：喂一段 token 进去，吐一段 token 出来，中间不管套了多少层优化（KV 缓存、分页、连续批处理、前缀缓存、采样），对外的形状始终是"调用、等待、拿到结果"。这一篇要解决的，是横在"一个函数"和"一个能被外部程序连上、同时服务很多人的服务"之间的那件事——而且只有这一件事是真正难的：

> **HTTP 是并发的（N 个连接同时来），引擎是单线程的（一步一步走）。**

难在哪里，先把结论摆在最前面，本篇余下的篇幅都是在把这张图撑开揉碎讲清楚：

```
HTTP handler 线程（N 个）              引擎线程（恰好 1 个）
  收请求 --> in_q.put(job)   ----->    step() 循环：准入/推进/回收
  发响应 <-- job.out_q.get() <-----    每出一个 token 就 put 一次
```

这张图（`_lab/serve.py:14-16`）是本篇的骨架：**一进一出两道队列，全局只有一个线程碰引擎**。为什么必须长成这样、为什么"加个 Flask 就完事了"是错的想法、这个结构下还要额外补哪几件事才能算是一个能上线的服务——是 `## 2`、`## 5` 要展开的内容。围绕这张图，本篇要讲清楚六件事，缺一件都不能说这是一个"服务"，只能说是一个"挂了个 HTTP 外壳的函数"：

1. **为什么 handler 线程不能直接调 `engine.step()`**——不是风格问题，是正确性问题，而且错了不会报错。
2. **为什么本文件不用 asyncio**——这是一条本篇要单独摆出来的决策，不是随手的实现选择。
3. **背压**——队列满了要立刻说不，不能笑着全收下来再慢慢拖。
4. **客户端断连必须回收**——本篇最容易被漏掉、也最值钱的一条。
5. **`ServingEngine` 为什么要把父类整段抄一遍**——一条关于"从 0 写的话该在哪里留缝"的设计教训。
6. **指标必须分层**——TTFT 和 TPOT 混成一个数字，等于什么都没测。

## 1. 先看现象（可跑的代码/可算的数）

先把 `_lab/serve.py` 真的跑起来。第一步是自检，`--selftest` 会拉起一个真实的 `ThreadingHTTPServer`，用标准库的 `urllib` 当客户端打一遍全部接口，跑完关掉：

```bash
cd _lab
python serve.py --selftest
```

这台机器上跑出来的真实输出（Windows、CPU、numpy 玩具引擎）：

```
  [ok]   GET /health 可用
  [ok]   非流式：跑满 max_tokens 时 finish_reason == 'length'
  [ok]   usage 里的 token 数对得上
  [ok]   TTFT 被量到了（且与 TPOT 分开报）
  [ok]   **流式各片拼起来 == 非流式整段**（流式唯一的正确性标准）
  [ok]   SSE 以 [DONE] 收尾
  [ok]   max_tokens + prompt 超过 max_seq 时返回 400（准入前拒）
  [ok]   6 条并发全部返回，块全部归还（在用 0）
  全部通过
```

9 项断言全过（比表面看到的多一条——最后一条不是校验某个值对不对，是把一个数字**报出来**，`## 3.6`、`## 6.6` 会讲这条为什么长这样）。这份"全过"背后有一段真实的返工史：本篇最初写作时，这份自检脚本自己的并发客户端不处理 503，5 次里有 2 次在第 8 项上炸出未捕获的 `HTTPError`，被当场记录成了一次"flake"；后来确认这不是环境抖动，是客户端自己没接住背压——`_lab/serve.py` 已经把这个坑补上，`## 6.6` 会把现象、根因、修法完整地讲一遍，因为它本身就是"背压不是服务端一家的事"这条道理最朴素的一次示范。这里先按下不表，继续往下看服务本身怎么用。

前台起服务：

```bash
python serve.py
```

```
listening on http://127.0.0.1:8000
  POST /v1/completions   GET /health  /v1/models  /metrics
```

另开一个终端，先看非流式：

```bash
curl -s -X POST localhost:8000/v1/completions -H "content-type: application/json" \
     -d "{\"prompt\":\"Hello\",\"max_tokens\":8}"
```

真实返回（本地实测，一次性打印）：

```json
{"id": "cmpl-2", "object": "text_completion", "model": "minigpt-toy",
 "choices": [{"index": 0, "text": "OPOPHello\n\nHelloOPHello", "finish_reason": "length"}],
 "usage": {"prompt_tokens": 1, "completion_tokens": 8},
 "timings": {"ttft_ms": 2.73, "tpot_ms": 0.2}}
```

`text` 字段读起来毫无意义（`"OPOPHello\n\nHelloOPHello"`）——这不是 bug，是意料之中的事：`CFG`（`_lab/serve.py:59-60`）用的是**随机初始化的权重**（`M.init_weights(CFG, ...)`，种子固定但没有任何训练），模型本身不承担"说人话"这个任务，本篇要验的是**服务这一层的行为**（并发、背压、断连回收、指标），不是生成质量。`timings` 里 `ttft_ms=2.73`、`tpot_ms=0.2` 这两个数字**只说明这个玩具在这台机器这一次请求上的行为**，`## 5`、`## 6` 会反复提醒这一点，不可外推到任何真实引擎。

再看流式（`-N` 关掉 curl 的缓冲，逐行看 SSE）：

```bash
curl -s -N -X POST localhost:8000/v1/completions -H "content-type: application/json" \
     -d "{\"prompt\":\"Hello\",\"max_tokens\":8,\"stream\":true}"
```

真实返回（节选前几行 + 收尾）：

```
data: {"id": "cmpl-1", "object": "text_completion", "choices": [{"index": 0, "text": "OP", "finish_reason": null}]}

data: {"id": "cmpl-1", "object": "text_completion", "choices": [{"index": 0, "text": "OP", "finish_reason": null}]}

data: {"id": "cmpl-1", "object": "text_completion", "choices": [{"index": 0, "text": "Hello", "finish_reason": null}]}

...(中略五行)...

data: {"id": "cmpl-1", "object": "text_completion", "choices": [{"index": 0, "text": "", "finish_reason": "length"}]}

data: [DONE]
```

把这几个 `text` 片段按顺序拼起来，正好就是刚才非流式返回的那串 `"OPOPHello\n\nHelloOPHello"`——这不是巧合，是 `## 3.6` 要讲的 selftest 第 5 条断言在验的东西：**流式各片拼起来必须等于非流式整段**，这是流式输出唯一的正确性标准。

最后看 `/metrics`（跑完上面两条请求之后）：

```bash
curl -s localhost:8000/metrics
```

```json
{"completed": 2, "queue_depth": 0, "running": 0, "waiting": 0, "blocks_in_use": 0,
 "ttft_ms": {"p50": 2.73, "p90": 2.73}, "tpot_ms": {"p50": 0.25, "p90": 0.25},
 "engine": {"prefill_tokens": 2, "decode_steps": 14, "cached_tokens": 0},
 "note": "numpy toy on CPU; NOT comparable to any real engine"}
```

留意最后一个字段 `"note"`——它不是可有可无的装饰，`_lab/serve.py:457` 把这句免责声明**焊进了返回值本身**，而不只是写在文档里：任何拿这份 `/metrics` 数据做汇报、画图的人，哪怕根本没读过这篇文章，也会在原始 JSON 里撞见这句话。这跟 `## 5.6`（TTFT/TPOT 分层）是同一个态度的两种落地方式——一个是"怎么测"，一个是"测完了怎么防止被误用"。

再验一条准入前的拒绝：

```bash
curl -s -X POST localhost:8000/v1/completions -H "content-type: application/json" \
     -d "{\"prompt\":\"Hello\",\"max_tokens\":1000000}" -w "\nHTTP_CODE:%{http_code}\n"
```

```
{"error": "prompt + max_tokens 超过 max_seq"}
HTTP_CODE:400
```

这四组真实交互——非流式、流式、指标、越界拒绝——覆盖了本篇要讲的大半个表面。`## 2` 开始把"为什么必须长这样"从现象拆成原理。

## 2. 原理

### 2.1 为什么 handler 线程不能直接调 `engine.step()`

最直接的写法，也是几乎所有人第一次接触"给引擎包一层 HTTP"时会写出来的版本：`ThreadingHTTPServer` 每来一个连接就开一个线程处理，`do_POST` 里直接 `engine.add(req)`、`while not req.finished: engine.step()`，用完就返回。这个写法在**单个请求**下完全正确，能通过所有正确性测试——问题在于它对"并发"这个词的理解是错的：`ThreadingHTTPServer` 名字里的 Threading 恰恰意味着，一旦有第二个连接同时进来，第二个 handler 线程会在**同一时刻**执行同一份 `engine.step()`。

`engine.step()`（`Engine.step`，`_lab/engine.py:190-203`，`## 3.2` 会展开 `ServingEngine` 对它的改造）内部改写的，是几样**可变共享状态**：`self.pool`（`PagedKVCache`，`_lab/paged.py:61-77`）里的物理 K/V 数组、`self.pool.alloc`（`BlockAllocator`，`_lab/paged.py:22-58`）的空闲栈和引用计数、`self.prefix`（`PrefixCache`，`_lab/engine.py:36-81`，`[[05-前缀缓存]]` 整篇讲的就是它）的哈希表、以及 `self.waiting`/`self.running` 两张列表本身。两个线程同时跑 `step()`，等价于两段代码同时对同一个 Python list 做 `pop(0)`/`append()`、同时对同一个 dict 做写入、同时往同一块 numpy 数组里写不同的内容——CPython 的 GIL 只保证单条字节码指令是原子的，**不保证"读出旧值、算新值、写回"这一整套操作是原子的**，`BlockAllocator.alloc()`（`_lab/paged.py:35-40`）里 `b = self.free.pop()` 和 `self.ref[b] = 1` 之间，另一个线程完全可能插进来也 `pop()` 到同一个块号，或者把这个刚分配出去的块又判定成空闲塞回栈里。

而这条错误最要命的地方，`_lab/serve.py:8-10` 的模块 docstring 已经把结论写在最前面：**它坏得静悄悄**。`[[05-前缀缓存]]` `## 2.5`、`## 6.2` 已经完整推演过一次"引用计数被算错导致两条请求共享同一块 KV"的后果——块被过早释放、分给不相关的第三条请求、原持有者读到别人写的数据；本篇讲的是这条推演的**并发版本**：不需要任何逻辑错误，光是"两个线程同时改一份状态"这一件事本身，就足以制造出跟 `[[05-前缀缓存]]` 完全同一种症状——程序不崩溃、两条请求各自都拿到了一个语法通顺的输出，只是它们的 KV 在某个时刻被交叉写过，答案是错的。**没有任何异常、没有任何日志、没有任何提示**，唯一能看出问题的办法是像 `[[05-前缀缓存]]` `## 2.3` 那样，把两条请求的输出拿去跟"完全不并发、串行执行"的参照结果做逐位比较——而这恰恰是并发场景里最难做到的事，因为 bug 本身**依赖具体的调度时序**，同一份代码这次跑对了，不代表下次还对。

结论只有一条：**引擎内部状态不能被两个线程同时触碰，一步都不行。** 剩下的问题是，多线程的 HTTP 服务器要怎么在这条约束下工作。

### 2.2 一进一出两道队列：请求在系统里的完整旅程

答案不是"给 `step()` 加锁"。加锁能防止两个线程同时改状态，但换不来别的东西——一条请求在拿到锁、跑完它的那一步之前，会让所有其它线程在锁外面排队等着，等于把"并发服务器"退化成了"每次只服务一个人、其它人卡在锁上"，而且拿锁的顺序完全没有调度可言，先抢到锁的先跑，`[[04-连续批处理]]` 辛辛苦苦搭起来的连续批处理（`_admit`/推进/回收按步交替，让请求互相不拖后腿）在这种锁的调度下毫无意义——因为压根没有一个统一的地方来决定"这一步该跑哪些请求"。

`_lab/serve.py:12-16` 给出的答案是**职责分离**：HTTP 线程只负责"接活、等结果、把结果发出去"，一步都不碰引擎内部状态；引擎内部状态只由**恰好一个**线程碰，这个线程只负责"按顺序执行 `step()`"。两边中间横着两道 `queue.Queue`——`queue.Queue` 本身是线程安全的，它的 `put`/`get` 内部自己拿锁，调用方不需要操心——**入队列**（`EngineLoop.in_q`）负责把新请求从 HTTP 线程搬到引擎线程；**出队列**（每条请求自己的 `job.out_q`）负责把新生成的 token 从引擎线程搬回对应的 HTTP 线程。一条请求从进来到出去，完整旅程是这样的：

1. HTTP 线程收到 POST，解析出 `prompt`/`SamplingParams`，构造一个 `Job`（`_lab/serve.py:83-97`，`## 3.1` 展开），塞进 `EngineLoop.submit()`（`_lab/serve.py:222-227`）——这一步只是把 `Job` 对象 `put` 进 `in_q`，不碰引擎的任何东西；
2. 引擎线程的 `run()` 循环（`_lab/serve.py:281-289`）先调 `_intake()`（`_lab/serve.py:236-248`）把 `in_q` 里排队的 `Job` 逐个取出来，各自包成一个 `E.Request`，`self.engine.add(req)`（走 `[[04-连续批处理]]` 的 `Engine.add`，`_lab/engine.py:124-125`）扔进等待队列；
3. 引擎线程调 `self.engine.step()`（`ServingEngine.step`，`_lab/serve.py:175-191`）推进一步——**这一步、也只有这一步，真正碰了共享状态**，而它只在这一个线程里发生，不存在竞争；
4. 引擎线程调 `_drain()`（`_lab/serve.py:250-279`），把这一步新出现在 `req.out` 里的 token 逐个翻译成文本片段，塞进对应 `Job.out_q`；
5. HTTP 线程一直在各自的 `job.out_q.get()` 上阻塞等着（`_blocking`，`_lab/serve.py:382-399`；或 `_stream`，`_lab/serve.py:402-438`），拿到就发给客户端，拿到 `"done"` 就收尾。

这条旅程回答了 `## 2.1` 留下的问题：**共享状态只在第 3 步被触碰，而第 3 步全程只有一个线程在跑**。第 1、5 步（HTTP 线程干的事）和第 2、4 步（引擎线程干的事）都不碰 `pool`/`alloc`/`prefix` 这些东西，`queue.Queue` 本身的线程安全保证了两边交接数据这件事是安全的，不需要额外加锁。这就是"一进一出两道队列"这句话具体指的是什么。

### 2.3 这依然是"同一个引擎"，只是外面裹了两层队列

有一个容易被忽略、但值得单独点出来的事实：`ServingEngine.step()`（`## 3.2` 逐行拆开看）在做的事情，跟 `[[04-连续批处理]]`、`[[05-前缀缓存]]` 里的 `Engine.step()` 相比，**调度逻辑一个字都没有变**——还是先 `_admit()` 再推进再回收，前缀缓存的匹配/存储逻辑原样保留。改的只有两处 `argmax` 变成 `self._pick()`（采样，见 `## 3.2`），以及新增了几处 `abort` 判断（取消，见 `## 3.2`、`## 2.4`）。这意味着：**前面几篇「优化不许改变输出」的正确性契约，在服务这一层依然有效**——只要采样退化成贪心（`temperature=0`，`sampling.py` 里 `SamplingParams` 的默认值就是贪心），`ServingEngine` 在给定同一批请求、同一个到达顺序时，算出来的 token 必须和 `[[04-连续批处理]]`、`[[05-前缀缓存]]` 里的 `Engine` 完全一致——这条契约不需要专门证明，因为代码本身没有引入新的计算路径，只是给"谁能碰这份计算"这件事上了一把结构性的锁（两道队列）。`## 3.6` 会看到 `selftest` 和 pytest 里确实没有为这条"服务层不改变输出"单独写断言——因为它是 `[[04-连续批处理]]`、`[[05-前缀缓存]]` 已经验过的东西的直接推论，不是本篇新引入的性质。本篇真正新增、需要专门验证的正确性命题是另一件事：**并发下资源必须归零**（`selftest` 第 8 条、pytest 的 `test_concurrent_requests_leak_no_blocks`，`## 3.6` 展开）。

### 2.4 取消是这条旅程里唯一的"逆向"分支

`## 2.2` 描述的旅程有一个隐含假设：请求乖乖地跑完、乖乖地把结果取走。真实世界里，HTTP 连接会断——用户关掉了浏览器标签页，反向代理超时了，客户端的 `curl -N` 被 Ctrl-C 了——这时候第 5 步（HTTP 线程在 `out_q.get()` 上等）不再会发生，可第 2、3、4 步（引擎线程还在推进这条请求、还在往它的 `out_q` 里塞东西）完全不知道对面已经没人在收了。`_lab/serve.py:26-27` 的模块 docstring 把这件事的后果直接摆出来：**它占的 KV 块不会自己还回来**，因为 `## 2.1` 已经讲过，判断"这条请求该不该继续跑"这件事，只有引擎线程自己知道，而引擎线程从来没有被告知"对面已经断了"。

这条链路需要三处配合，缺一不可，`_lab/serve.py:434-437` 的 `_stream` 是链路的起点：SSE 是一个长连接，`self.wfile.write()`（`chunk()` 内部，`_lab/serve.py:410-414`）在客户端已经断开的情况下会抛 `BrokenPipeError`（或 `ConnectionResetError`）——这是**HTTP 线程唯一能感知到"对面没人了"的时刻**，因为 HTTP 协议本身没有"我不要了"这种主动信号，只能靠"写不进去了"这个副作用反推。捕获到之后，`job.cancelled = True`（`_lab/serve.py:437`）——注意这里只是**打了个标记**，HTTP 线程自己不做任何清理，因为它压根碰不到引擎状态（`## 2.1` 的约束在这里同样成立：取消也不能由 HTTP 线程直接动手）。真正的清理留给引擎线程在下一次 `_drain()`（`_lab/serve.py:253-254`）里做：`if job.cancelled and not getattr(req, "abort", False): req.abort = True`——把 HTTP 线程打的标记翻译成引擎线程认识的信号。最后一棒在 `ServingEngine.step()`（`_lab/serve.py:186-190`）：**"取消也是一种结束"**，`getattr(req, "abort", False)` 一旦为真，这条请求会跟正常结束的请求走同一段回收代码——`req.tab.free()` 真正把块还给空闲池。三处任何一处缺了，链路都断在半路：`_stream` 不捕获异常，HTTP 线程会直接崩掉这个 handler 线程（`ThreadingHTTPServer` 会吞掉这个异常，但标记永远打不上）；`_drain` 不翻译，标记永远留在 `Job` 对象上，引擎线程从来看不到；`step()` 不检查 `abort`，请求会硬跑到 `max_new`，`## 6.2` 会把这条错误单独展开成一条"常见错误"，因为它是本篇最容易被漏掉、后果也最隐蔽的一条——不报错、不崩溃，只是显存（这里是块池）曲线只涨不跌。

## 3. 自己动手：`_lab/` 里对应的实现

### 3.1 `Job`：HTTP 线程与引擎线程之间唯一的通信对象

`_lab/serve.py:83-97` 定义了 `Job`：

```python
@dataclass
class Job:
    """一条在飞的请求。**HTTP 线程与引擎线程之间只通过它通信。**"""
    rid: int
    prompt: list[int]
    params: SamplingParams
    stream: bool
    out_q: "queue.Queue[tuple[str, object]]" = field(
        default_factory=queue.Queue)          # ("piece"|"done"|"error", payload)
    cancelled: bool = False                   # 客户端断连时由 HTTP 线程置位
    t_arrive: float = field(default_factory=time.perf_counter)
    t_first: float = -1.0                     # 第一个 token 出来的时刻
    t_end: float = -1.0
    n_out: int = 0
    finish_reason: str = "length"
```

留意它的字段构成：`rid`/`prompt`/`params`/`stream` 是 HTTP 线程创建时就定死的输入；`out_q`/`cancelled`/`t_first`/`t_end`/`n_out`/`finish_reason` 全部由**引擎线程写、HTTP 线程读**（`cancelled` 是唯一反过来的例外——HTTP 线程写、引擎线程读，`## 2.4` 已经讲过它的作用）。`Job` 不持有任何 `numpy` 数组、不引用 `pool`、`prefix` 这些共享状态的任何一部分——它是一个纯粹的"信封"，`## 2.1` 强调的边界（HTTP 线程不碰引擎内部状态）在类型层面就被这个设计保证了：`Job` 上没有任何字段能让 HTTP 线程摸到 `PagedKVCache` 或 `PrefixCache`。

`ttft`（`_lab/serve.py:99-102`）和 `tpot`（`_lab/serve.py:104-114`）是 `Job` 上两个只读属性：

```python
@property
def ttft(self) -> float:
    """Time To First Token：**排队 + prefill**。用户感知的"卡不卡"就是它。"""
    return self.t_first - self.t_arrive if self.t_first > 0 else -1.0

@property
def tpot(self) -> float:
    """Time Per Output Token：出了第一个之后，平均每个 token 多久。"""
    if self.n_out <= 1 or self.t_first <= 0:
        return -1.0
    return (self.t_end - self.t_first) / (self.n_out - 1)
```

`ttft` 算的是 `t_first - t_arrive`——从这条请求被创建（`t_arrive` 是 `Job` 的默认工厂函数在构造时打的戳，`_lab/serve.py:93`）到它第一个 token 真正出现在 `out_q` 里，这段时间既包含它在 `in_q`/等待队列里排队的时间，也包含它自己 prefill 的时间，两者在这一个数字里天然叠在一起，无法分离——这不是缺陷，是 TTFT 这个指标定义本身决定的：用户在意的是"发出请求到看到第一个字"这段体感等待，不关心这段等待里有多少是排队多少是真算。`tpot` 算的是从第一个 token 到最后一个 token 之间，平均每个 token 花多久——`n_out <= 1` 时返回 `-1.0`（哨兵值），因为只有一个输出 token 时根本没有"两个 token 之间的间隔"可言，`## 5.6` 会把"为什么这两个数必须分开报"当一条独立的设计决策讲。

`CFG`（`_lab/serve.py:59-60`）的定义值得停一下：

```python
CFG = M.Config(n_layer=2, n_head=2, d_model=32, vocab=len(TOY_VOCAB),
               max_seq=256, seed=1)
```

`vocab=len(TOY_VOCAB)`——词表大小被**刻意设成**跟 `sampling.py` 里那个 12 个 token 的玩具词表（`TOY_VOCAB`）长度相等。这不是巧合也不是省事：一旦两者相等，模型吐出来的 id 就可以**直接**拿去查 `TOY_VOCAB` 解码，不需要再写一层"模型词表 id → 展示用 token id"的映射表。真实模型的词表是几万到几十万，`## 4` 会看到真实引擎里这一层映射（tokenizer 的 vocab 和模型输出的 logits 维度对齐）是一个专门的、不轻的子系统；本库跳过它，是为了不让这层无关的复杂度掩盖本篇真正要讲的东西（并发结构），`toy_encode`（`_lab/serve.py:63-78`）用最长匹配分词把字符串切成 id，注释里也提醒了"真实 BPE 不是这么做的"。

### 3.2 `ServingEngine`：为什么要把父类整段抄一遍

`ServingEngine`（`_lab/serve.py:119-191`）继承自 `[[04-连续批处理]]`、`[[05-前缀缓存]]` 里那个 `Engine`（`_lab/engine.py:101-212`）。类文档字符串（`_lab/serve.py:120-132`）已经把结论写在最前面：

```python
class ServingEngine(E.Engine):
    """在 `Engine` 上补两件服务必需的能力：**每请求采样** 与 **中途取消**。

    ⚠ 下面把父类的 `_admit()` 和 `step()` 整个抄了一遍，只为了把两处 `argmax`
    换成 `self._pick()`。**这个重复本身是一个设计教训，值得单独说一句**：

        `engine.py` 把"选下一个 token"写死成 `argmax`，是为了让前面几章
        「优化不许改变输出」的逐位对比能成立 —— 采样一旦引入随机性，那套测试就没法写。
        代价就是这里：**采样点没有被设计成一个可替换的钩子，于是只能整段覆盖。**
    """
```

把 `_admit()`（`_lab/serve.py:142-173`）和 `Engine._admit()`（`_lab/engine.py:152-188`）逐行摆在一起对比，能确认这句话字面上是准的：调度逻辑（前缀缓存匹配、`todo` 计算、全部命中时回退一个块、`prefill` 统计）**一行都没变**，唯一的差异卡在这一行：

```python
req.out.append(self._pick(req, logits))     # <- 唯一的改动    # _lab/serve.py:167
```

对应父类的：

```python
req.out.append(int(logits.argmax()))                          # engine.py:182
```

`_pick`（`_lab/serve.py:138-140`）本身极短：

```python
def _pick(self, req: E.Request, logits: np.ndarray) -> int:
    s = self.samplers.get(req.rid)
    return int(logits.argmax()) if s is None else s(logits, req.out)
```

如果这条请求没有注册采样器（`self.samplers` 是 `ServingEngine.__init__` 新增的字典，`_lab/serve.py:134-136`），退化回 `argmax`；有的话，就把 `logits` 和这条请求已经生成的 `req.out`（重复惩罚要用到历史）交给 `[[06-采样-停止条件-流式输出]]` 里的 `Sampler.__call__`。

但对着 `step()`（`_lab/serve.py:175-191` vs `_lab/engine.py:190-203`）再逐行比一遍，会发现类文档字符串这句"只为了把两处 `argmax` 换成 `self._pick()`"其实**没有说全**——`step()` 里除了那处替换，还多了两处判断和一行清理，都不在文档字符串提到的范围内：

```python
def step(self) -> None:
    self._admit()
    for req in list(self.running):
        if req.finished or getattr(req, "abort", False):      # <- 多了 abort 判断
            continue
        logits = self._forward(req, [req.out[-1]])
        req.out.append(self._pick(req, logits))                # <- 唯一提到的改动
        self.stats["decode_steps"] += 1
    for req in list(self.running):
        if req.finished or getattr(req, "abort", False):       # <- 多了 abort 判断
            req.done = self.step_id
            req.tab.free()
            self.running.remove(req)
            self.samplers.pop(req.rid, None)                   # <- 多了这一行清理
    self.step_id += 1
```

这两处 `getattr(req, "abort", False)` 和最后那行 `samplers.pop`，正是 `## 2.4` 讲的取消链路真正落地的地方——**"取消也是一种结束"**这句注释（`_lab/serve.py:184-185`）就写在这里。这不是在挑文档字符串的错，而是想借这个真实存在的落差提醒一件事：**读源码的docstring 只能当地图，不能替代亲手把两份代码摆在一起逐行对比**——本库反复强调的"静默 bug"心态，同样适用于读文档本身：文档字符串没有撒谎（两处 `argmax` 确实是"为了采样"这一个目的做的改动），只是没把"顺手在同一份重写里也接上了取消"这件事说完整。

`_admit()`（`_lab/serve.py:142-173`）除了这行采样替换，值得单独指出它跟 `[[05-前缀缓存]]` `## 3.2` 讲过的父类版本**完全共享**同一套前缀缓存逻辑——命中匹配、全部命中时回退一个块（`_lab/serve.py:154-160`）、`store` 登记，一步都没有为了适配"服务化"而改动。这再次印证了 `## 2.3` 的判断：服务这一层是**纯粹外挂**的，调度和缓存这些"引擎内部"的事情不知道自己被 HTTP 包住了，也不需要知道。

### 3.3 `EngineLoop`：全进程唯一碰引擎的线程

`EngineLoop`（`_lab/serve.py:196-289`）是 `## 2.2` 那条旅程里第 2、3、4 步的载体，类文档字符串（`_lab/serve.py:197-206`）：

```python
class EngineLoop(threading.Thread):
    """**全进程唯一碰引擎的线程。** 它做四件事，循环做：
      1. 从 `in_q` 收新请求（非阻塞，收空为止）；
      2. 调 `engine.step()` 推进一步；
      3. 把这一步新产生的 token 增量解码后塞进各自的 `out_q`；
      4. 结束/取消的请求收尾。

    **没有请求时要让出 CPU**，否则这个 while 会把一个核吃满 ——
    很多人第一版会忘，表现是"服务一起来风扇就转"。
    """
```

`__init__`（`_lab/serve.py:210-219`）里唯一需要留意的是 `self.engine = ServingEngine(CFG, M.init_weights(CFG), **engine_kw)`（`_lab/serve.py:212`）——**引擎实例本身是 `EngineLoop` 的私有成员**，没有暴露给外面任何一个 HTTP handler 直接持有，handler 手里能拿到的只有 `loop.submit()`、`loop.depth()` 这两个方法（`_lab/serve.py:222-227`、`229-230`）：

```python
def submit(self, job: Job) -> bool:
    """收下返回 True；**队列满则返回 False，调用方必须回 503**。"""
    if self.in_q.qsize() + len(self.engine.waiting) >= self.max_queue:
        return False
    self.in_q.put(job)
    return True

def depth(self) -> int:
    return self.in_q.qsize() + len(self.engine.waiting) + len(self.engine.running)
```

`submit()` 的背压检查（`## 5.4` 展开）比较的是 `in_q.qsize() + len(self.engine.waiting)`，不是只看 `in_q` 一边——原因很直接：一条请求被 `_intake()`（`_lab/serve.py:236-248`）从 `in_q` 搬进 `self.engine.waiting` 之后，`in_q` 那一侧的计数会归零，但这条请求显然还没有被服务，仍然占着系统的容量。只看 `in_q.qsize()` 会在"引擎线程刚搬完一批"这个时间点上产生一个**错误的低估**，让本该被拒绝的新请求被误判为"还有空位"而放进来——`depth()`（供 `/health`、`/metrics` 用）同理，还要再加上 `running` 那一段，才是"系统里现在实打实压着多少条请求"的真实数字。

`_intake()`（`_lab/serve.py:236-248`）和 `_drain()`（`_lab/serve.py:250-279`）是 `## 2.2` 旅程第 2、4 步的具体实现，`_drain()` 里有一处容易被跳过但很关键的注释（`_lab/serve.py:268-270`）：

```python
# **只认引擎的回收信号**（`req.done >= 0` 由 `Engine.step()` 置位）。
# 不能自己判断"输出够了就算完"—— 那样会在块还没归还时就宣告结束，
# 于是 `/metrics` 的 blocks_in_use 永远对不上账。
if req.done >= 0:
```

`_drain()` 本可以用一个更"直观"的判据来决定要不要给 `Job` 发送 `"done"`——比如检查 `len(req.out) >= job.params.max_new`。这个判据在数值上大概率是对的，但它跟"引擎真正回收了这条请求的 KV 块"这件事**不是同一个时刻**：`req.done` 是 `ServingEngine.step()` 在真正调用 `req.tab.free()` 之后才置位的（`_lab/serve.py:187-188`），如果 `_drain()` 抢在这之前就宣告"完了"，`Job.out_q` 会先收到 `"done"`，HTTP 线程会先把响应发出去、`selftest`/客户端也会认为这条请求已经结束——但此刻块可能还没真正还给空闲池，`/metrics` 里 `blocks_in_use`（`_lab/serve.py:453`）这一瞬间读出来的数字就会比实际应有的偏高，账对不上。**只认引擎的回收信号**，保证了"HTTP 层认为请求结束"和"引擎层认为资源已回收"这两件事被绑定成同一个时刻，不会有任何窗口期。

`run()`（`_lab/serve.py:281-289`）是这整个类的主循环：

```python
def run(self) -> None:
    while not self._stop.is_set():
        self._intake()
        if self.engine.waiting or self.engine.running:
            self.engine.step()
            self._drain()
        else:
            time.sleep(0.002)      # 空转时让出 CPU，别把核吃满
```

`else` 分支的 `time.sleep(0.002)` 是 `## 6.4` 要讲的一条常见错误的正面对照——没有请求时，`self.engine.waiting`、`self.engine.running` 都是空的，如果这里没有让出 CPU，`while` 会在没有任何工作可做的情况下疯狂空转，把一个 CPU 核心吃满，`_lab/serve.py:204-206` 的类文档字符串直接点出了这条错误的现象："服务一起来风扇就转"。

### 3.4 HTTP 层：解析、准入、背压、两种响应模式

`_parse()`（`_lab/serve.py:302-317`）把请求体的 JSON 翻译成 `(ids, SamplingParams)`：

```python
def _parse(body: dict) -> tuple[list[int], SamplingParams]:
    prompt = body.get("prompt", "")
    ids = list(prompt) if isinstance(prompt, list) else toy_encode(str(prompt))
    ...
    return ids, SamplingParams(
        temperature=float(body.get("temperature", 0.0)),
        ...
        max_new=int(body.get("max_tokens", 16)),
        eos=body.get("eos", 0),
        stop=tuple(stop),
    )
```

`prompt` 既可以是一段文本（走 `toy_encode`）也可以直接是一串 id 列表——这个小分支是为了方便 `selftest`/调试时绕过分词直接指定 token，跟真实引擎的 `prompt` 字段能力对应但简化了很多，`## 4` 会看到真实引擎在这里要处理的是完整的 tokenizer/chat template 流水线。

`Handler.do_POST`（`_lab/serve.py:347-379`）是接进整条旅程的入口，按顺序做了四件必须做在**准入之前**的检查：

```python
n = int(self.headers.get("content-length", 0))
body = json.loads(self.rfile.read(n) or b"{}")          # 1) JSON 解析失败 -> 400，不是 500
...
ids, params = _parse(body)
if not ids:
    self._json(400, {"error": "empty prompt"})           # 2) 空 prompt -> 400
    return
if len(ids) + params.max_new > CFG.max_seq:
    self._json(400, {"error": "prompt + max_tokens 超过 max_seq"})   # 3) 越界 -> 400
    return

job = Job(_new_rid(), ids, params, bool(body.get("stream")))
if not self.loop.submit(job):
    self.send_response(503)                               # 4) 队列满 -> 503
    self.send_header("retry-after", "1")
    ...
```

第 3 条 `len(ids) + params.max_new > CFG.max_seq` 的检查位置很关键——它发生在 `Job` 被 `submit()` 之前，也就是这条请求**还没有被引擎线程看到**、更没有被分配任何物理块。`_lab/serve.py:363` 的注释把理由写得很直接："放进去再撑爆，代价是已经占掉的块和已经算掉的 prefill"——如果这条检查挪到 `_admit()` 内部（引擎线程已经真正给它分配了块表、甚至已经跑了一部分 prefill 之后）才做，等于是先花掉真实的计算和内存，再决定要不要拒绝，这笔已经花出去的成本全部打了水漂。**越晚发现一个必然要拒绝的请求，浪费的资源越多**——这是本篇除了并发结构之外，另一条贯穿始终的原则。

第 4 条背压检查（`## 5.4` 展开），`submit()` 返回 `False` 时，`do_POST` 直接 `send_response(503)` 并带上 `retry-after: 1` 头（`_lab/serve.py:370-373`）——不排队等、不阻塞这个 HTTP 线程，立刻把"我现在满了"这件事告诉客户端。

非流式响应 `_blocking()`（`_lab/serve.py:382-399`）是最简单的消费者：在 `job.out_q.get()` 上死循环，收集所有 `"piece"`，碰到非 `"piece"`（也就是 `"done"`）就跳出，拼起来一次性用 `_json` 发回去，`timings` 字段里带上 `job.ttft`、`job.tpot`（`## 5.6` 展开为什么这里是两个数不是一个）。

流式响应 `_stream()`（`_lab/serve.py:402-438`）稍微复杂，先看响应头：

```python
self.send_response(200)
self.send_header("content-type", "text/event-stream; charset=utf-8")
self.send_header("cache-control", "no-cache")
self.send_header("transfer-encoding", "chunked")   # _lab/serve.py:407
self.end_headers()
```

`transfer-encoding: chunked` 这一行的注释（`_lab/serve.py:406`）点出了理由：**SSE 长度未知，只能分块传**——一次性发送需要提前知道 `content-length`，可流式响应在开始发送的那一刻，谁都不知道最终会生成多少个 token、总共多少字节，`## 6.3` 会把"漏掉这个头"单独列成一条常见错误。`chunk()` 这个内嵌函数（`_lab/serve.py:410-414`）负责按 HTTP/1.1 chunked 编码的格式手动拼包（十六进制长度 + `\r\n` + 内容 + `\r\n`），每次都 `self.wfile.flush()`——不 flush 的话数据可能被操作系统的发送缓冲区攒住不发，客户端会看着连接"卡住"，跟真正意义上的流式背道而驰。

主循环里每收到一个 `"piece"` 就包一层 `data: {...}\n\n` 发出去；收到 `"done"` 就发最后一条带 `finish_reason` 的消息，再发 `data: [DONE]\n\n`（`## 1` 已经用真实输出验过这个收尾格式），最后 `chunk("")` 发一个长度为零的块作为 chunked 编码本身的结束标记。`## 2.4` 已经完整讲过的 `except (BrokenPipeError, ConnectionResetError, OSError): job.cancelled = True`（`_lab/serve.py:434-437`）就挂在这个循环外面。

### 3.5 `_metrics`：把指标分层落到接口上

`_metrics()`（`_lab/serve.py:440-458`）是 `## 5.6`（TTFT/TPOT 分层）在 `/metrics` 接口上的具体体现：

```python
def _metrics(loop: EngineLoop) -> dict:
    done = list(loop.done)
    ttfts = sorted(j.ttft for j in done if j.ttft > 0)
    tpots = sorted(j.tpot for j in done if j.tpot > 0)

    def p(xs, q):
        return round(xs[min(len(xs) - 1, int(len(xs) * q))] * 1000, 2) if xs else -1

    return {
        ...
        "ttft_ms": {"p50": p(ttfts, 0.5), "p90": p(ttfts, 0.9)},
        "tpot_ms": {"p50": p(tpots, 0.5), "p90": p(tpots, 0.9)},
        "engine": dict(loop.engine.stats),
        "note": "numpy toy on CPU; NOT comparable to any real engine",
    }
```

`ttft_ms` 和 `tpot_ms` 是**两个独立的字典**，各自有自己的 p50/p90，而不是合并成一个 `latency_ms`——这不是随手的接口设计，是 `## 5.6` 那条决策在数据结构层面的落地：一旦两者被合并，`p90` 这种分位数统计会立刻失真，`## 5.6` 会具体展开为什么。`"engine": dict(loop.engine.stats)` 直接把 `[[04-连续批处理]]`、`[[05-前缀缓存]]` 里 `Engine.stats`（`prefill_tokens`/`decode_steps`/`cached_tokens`）原样透出——服务层没有另起一套统计口径，用的是引擎自己已经在维护的账本，这也是 `## 2.3`"服务层是纯外挂"这条判断的又一处印证。

### 3.6 `selftest` 与 pytest：每条断言在验什么

`selftest()`（`_lab/serve.py:472-562`）的 9 条断言，跟 `## 1` 已经跑出来的输出逐条对上：

1. `GET /health` 可用（`_lab/serve.py:517`）——最基本的存活检查；
2. 非流式跑满 `max_tokens` 时 `finish_reason == "length"`（`_lab/serve.py:520-521`）——`[[06-采样-停止条件-流式输出]]` 定义的 `should_stop` 在服务层被正确接了进来；
3. `usage.completion_tokens` 与真实生成的 token 数一致（`_lab/serve.py:522`）；
4. `timings.ttft_ms > 0` 且与 `tpot_ms` 分开报（`_lab/serve.py:523`）——`## 5.6` 的最小可执行验证；
5. 流式各片拼起来等于非流式整段（`_lab/serve.py:530-531`）——`## 1` 已经手动核对过一次，这里是自动化版本；
6. SSE 以 `data: [DONE]` 收尾（`_lab/serve.py:532`）；
7. `max_tokens` 过大时准入前返回 400（`_lab/serve.py:538-539`，`## 3.4` 第 3 条检查的验证）；
8. 6 条并发请求全部返回、块全部归零（`_lab/serve.py:553-554`）——这是本篇**唯一**专门验证"并发正确性"的断言；
9. 过程中被背压拒绝并重试了几次（`_lab/serve.py:555-557`）——这条断言的判据是 `n_503[0] >= 0`，恒真，它不是在验证某个值对不对，是在借 `chk()` 这个已有的打印通道，把一个原本无人看见的计数**报出来**。`## 6.6` 要讲的整段返工史，就是从第 8 条断言在没有第 9 条、也没有 `## 3.4`/`post()` 里这段重试逻辑的旧版本上偶发失败开始的。

`_lab/tests/test_serving.py` 把这些断言原样搬进了 pytest，并且补充了 `selftest` 没覆盖的几条：`test_health_and_models`（`_lab/tests/test_serving.py:202`）、`test_completion_reports_length_finish_reason`（`:210`）、`test_stream_and_nonstream_agree`（`:217`）、`test_ttft_and_tpot_are_reported_separately`（`:234`）、`test_oversized_request_rejected_before_admission`（`:241`）、`test_bad_json_is_400_not_500`（`:249`，验的是 `## 3.4` 第 1 条检查——JSON 解析失败要拿到 400 而不是让异常一路捅穿到 500）、`test_concurrent_requests_leak_no_blocks`（`:258`，把并发请求数从 6 提到 8）、`test_metrics_endpoint_accounts_for_blocks`（`:276`，专门断言 `"NOT comparable"` 这句免责声明确实出现在返回值里）。

跑一遍全部相关测试：

```bash
cd _lab
python -m pytest tests -q
```

```
50 passed in 0.82s
```

## 4. 真实引擎是怎么做的（对照 vLLM / SGLang / TGI）

`## 2.2` 那张"一进一出两道队列"的图，在三个真实引擎里都能找到，但**解耦的手段**各不相同：vLLM 用 asyncio 的协程调度 + 独立进程，SGLang 直接拆成三个操作系统进程，TGI 用 Rust 的 `tokio::mpsc` channel。三者的共同点，恰好印证了 `## 2` 讲的那条原则不是本库自己发明的教条。

### 4.1 vLLM：HTTP → 队列 → 引擎核心，但队列两端都升了级

vLLM 的 `AsyncLLM.generate()`（`vllm:vllm/v1/engine/async_llm.py:550-624`）文档字符串（`vllm:vllm/v1/engine/async_llm.py:576-578`）几乎是本篇 `## 2.2` 那句话的原话重复：「A separate output_handler loop runs in a background AsyncIO task, pulling outputs from EngineCore and putting them into the per-request AsyncStream」——一个独立的后台任务从 EngineCore 拉输出，塞进每条请求自己的队列。这个后台任务是 `_run_output_handler` 里定义的 `output_handler()`（`vllm:vllm/v1/engine/async_llm.py:665-748`），主循环形状和 `_lab/serve.py:281-289` 的 `EngineLoop.run()` 完全对应：

```python
# vllm:vllm/v1/engine/async_llm.py:686-694（节选）
async def output_handler():
    try:
        while True:
            # 1) Pull EngineCoreOutputs from the EngineCore.
            outputs = await engine_core.get_output_async()
            ...
            # 2) Process EngineCoreOutputs.
            processed_outputs = output_processor.process_outputs(...)
            # NOTE: RequestOutputs are pushed to their queues.
```

对比本库的 `_drain()`（`_lab/serve.py:250-279`），结构是同一件事的两种写法：拉取这一步新产生的输出、逐条处理、推进各自的队列。区别在于 vLLM 的"引擎"（`EngineCore`）跑在**另一个操作系统进程**里，`get_output_async()`（`vllm:vllm/v1/engine/async_llm.py:688`）背后是跨进程 IPC，不是像本库这样同一进程内的 `queue.Queue`——`## 2.1` 讲的"共享可变状态不能被多线程同时碰"这条约束，vLLM 用了一种更彻底的解法：**压根不共享地址空间**，HTTP 服务和真正跑 `step()` 的调度器/模型是两个进程，连"同一份内存"这个前提都不存在了。

请求进入的一侧，`_add_request`（`vllm:vllm/v1/engine/async_llm.py:429-436`）对应本库的 `EngineLoop.submit()` + `_intake()`：

```python
# vllm:vllm/v1/engine/async_llm.py:433-436
# Add the request to OutputProcessor (this process).
self.output_processor.add_request(request, prompt, parent_req, index, queue)
# Add the EngineCoreRequest to EngineCore (separate process).
await self.engine_core.add_request_async(request)
```

两步分别对应本库 `_intake()`（`_lab/serve.py:236-248`）里"注册 `Job`/`Sampler`/`StreamGuard`"和"`self.engine.add(req)`"——先在本地登记好这条请求要用的辅助状态，再把请求本体真正递给跑推理的那一侧。每条请求自己的出队列是 `RequestOutputCollector`（`vllm:vllm/v1/engine/output_processor.py:48-87`），本库的 `Job.out_q` 是 `queue.Queue`（阻塞式），它是 `asyncio.Event` 驱动的：

```python
# vllm:vllm/v1/engine/output_processor.py:65-86（节选）
def put(self, output) -> None:
    """Non-blocking put operation."""
    if self.output is None or isinstance(output, Exception):
        self.output = output
        self.ready.set()
    ...
async def get(self):
    """Get operation blocks on put event."""
    while (output := self.output) is None:
        await self.ready.wait()
```

`put`/`get` 的语义跟 `queue.Queue` 的 `put`/`get` 一致（塞一个、等一个），只是本库用线程 + 阻塞队列，vLLM 用协程 + `asyncio.Event`——`## 5.2`（"为什么本文件不用 asyncio"）要讲的正是这一层：结构相同，事件循环把实现细节包了一层糖。

客户端断连这条链路（`## 2.4` 的核心），vLLM 走的是 Python 的异常传播机制，`generate()` 的 `except` 分支（`vllm:vllm/v1/engine/async_llm.py:616-624`）：

```python
# 如果客户端断连，generate() 会被 cancel，或者这个 generator 被垃圾回收。
# 所以我们在这里 abort 这条请求。
except (asyncio.CancelledError, GeneratorExit):
    if q is not None:
        await self.abort(q.request_id, internal=True)
    ...
    raise
```

跟本库 `## 2.4` 描述的三段链路对比：本库靠 `wfile.write()` 抛出的 `BrokenPipeError` 主动感知断连；vLLM 靠 asyncio 在检测到底层连接关闭时，把封装这条请求的协程任务标记为 `CancelledError`（或者这个异步生成器被垃圾回收触发 `GeneratorExit`），两种机制的共同点是**都不是主动轮询"客户端还在不在"，而是被动地在某个必然会发生的操作上捕获失败信号**——本库是写失败，vLLM 是协程被取消——这印证了 `## 2.4` 强调的那句话不是本库的巧合设计，是这一类系统必须面对的同一个约束下收敛出的同一类解法。

### 4.2 SGLang：三进程 + ZMQ，取消靠一个 2 秒的兜底任务

SGLang 的架构比 vLLM 走得更远，直接拆成三个独立的操作系统进程，`launch_server` 的文档字符串（`sglang:python/sglang/srt/entrypoints/http_server.py:2779-2783`）写得很明确：

```
- The engine consists of three components:
    1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
    2. Scheduler (subprocess): Receives requests from the Tokenizer Manager,
       schedules batches, forwards them, and sends the output tokens to the
       Detokenizer Manager.
    3. DetokenizerManager (subprocess): Detokenizes the output tokens and
       sends the result back to the Tokenizer Manager.
Note:
1. The HTTP server, Engine, and TokenizerManager all run in the main process.
2. Inter-process communication is done through IPC (each process uses a
   different port) via the ZMQ library.
```

HTTP 层（FastAPI）和 `TokenizerManager` 在主进程，真正跑连续批处理调度的 `Scheduler` 在一个独立子进程，`DetokenizerManager` 又是另一个独立子进程，三者之间用 ZMQ 做进程间通信——`## 2.1` 讲的"引擎状态不能被多个执行体同时碰"这条约束，SGLang 用了比 vLLM 更彻底的隔离：**调度器所在的进程连 HTTP 请求长什么样都不知道**，它只通过 ZMQ 收发经过序列化的消息。

`/generate` 的 HTTP handler（`sglang:python/sglang/srt/entrypoints/http_server.py:894-928`）流式分支里，客户端断连的处理方式跟 vLLM 不同：

```python
# sglang:python/sglang/srt/entrypoints/http_server.py:906-913（节选）
except ValueError as e:
    # A client disconnect also surfaces here. It's a client-side
    # cancellation, not a server error or bad input -- log it and
    # stop (the request was already aborted upstream) instead of
    # emitting a 400.
    if request is not None and await request.is_disconnected():
        logger.info(f"[http_server] Client disconnected: {e}")
        return
```

这一段处理的是"流式生成过程中途抛出异常，且这个异常恰好是客户端已经断开导致的"这种情况，但 SGLang 真正负责"通知调度器把这条请求收尾掉"的，是挂在 `StreamingResponse` 上的一个后台任务：

```python
return StreamingResponse(
    stream_results(),
    media_type="text/event-stream",
    background=_global_state.tokenizer_manager.create_abort_task(obj),   # sglang:.../http_server.py:928
)
```

`create_abort_task`（`sglang:python/sglang/srt/managers/tokenizer_manager.py:2112-2124`）：

```python
def create_abort_task(self, obj: GenerateReqInput):
    # Abort the request if the client is disconnected.
    async def abort_request():
        await asyncio.sleep(2)
        if obj.is_single:
            self.abort_request(obj.rid)
        else:
            for rid in obj.rid:
                self.abort_request(rid)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(abort_request)
    return background_tasks
```

这段代码里没有任何一处判断"这条请求是不是已经正常结束了"——它无条件地在响应发送完之后等 2 秒，再补一刀 `abort_request`。这跟本库、跟 vLLM 的做法都不一样：本库和 vLLM 都是**在检测到断连的那一刻立刻回收**；SGLang 这里是**不管三七二十一，固定延迟 2 秒后兜底回收一次**——正常完成的请求被再 `abort` 一次大概率是幂等的空操作（`abort_request` 内部会检查这条请求是否还存在），代价可以忽略；但对于真正中途断连的请求，这意味着**它占用的资源最多要多等 2 秒才会被真正释放**，比本库"下一次 `_drain()` 就能翻译成 `abort` 标记"（毫秒级）要慢得多。这是三个引擎里唯一一处，本篇能明确指出"取消回收的及时性"上存在真实设计差异的地方——`## 5` 会把这条也当一条决策来看：**用一个固定延迟的兜底任务换取实现简单**（不需要在流式生成器内部精确传递取消信号），代价是资源回收变慢。

### 4.3 TGI：Rust 的 mpsc channel，同一套结构换了一门语言写

TGI 的 v3 后端（`tgi:backends/v3/src/queue.rs`）里，`Entry` 结构体（`tgi:backends/v3/src/queue.rs:20-35`）自带一个 `response_tx`：

```rust
// tgi:backends/v3/src/queue.rs:20-27（节选）
pub(crate) struct Entry {
    pub request: ValidGenerateRequest,
    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: mpsc::UnboundedSender<Result<InferStreamResponse, InferError>>,
    ...
}
```

`mpsc::UnboundedSender` 就是本库 `Job.out_q` 的 Rust 版本——一条请求自带一根用来往外发结果的管道。`Queue` 结构体（`tgi:backends/v3/src/queue.rs:41-58`）自己也是靠 channel 实现的：

```rust
// tgi:backends/v3/src/queue.rs:41-58（节选）
pub(crate) struct Queue {
    queue_sender: mpsc::UnboundedSender<QueueCommand>,
}
impl Queue {
    pub(crate) fn new(...) -> Self {
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();
        tokio::spawn(queue_task(..., queue_receiver));
        Self { queue_sender }
    }
```

`Queue::new` 里 `tokio::spawn` 启动的后台任务持有等待队列的真实状态（`VecDeque<Entry>`），外部只能通过 `queue_sender` 发命令——跟本库 `EngineLoop.in_q`（外部只能 `put`，真正的 `self.engine.waiting` 只有引擎线程自己碰）是同一个模式。真正跑连续批处理调度、调用模型推理的 `batching_task`（`tgi:backends/v3/src/backend.rs:121-138`）：

```rust
// tgi:backends/v3/src/backend.rs:121-138（节选）
/// Batching logic
/// Will be launched in a background Tokio task
pub(crate) async fn batching_task(...) {
    loop {
        notifier.notified().await;
        while let Some((mut entries, batch, span)) = queue
            .next_batch(...)
            .await
        {
            let mut cached_batch = prefill(&mut client, batch, None, &mut entries)...
```

跟本库 `EngineLoop.run()`（`_lab/serve.py:281-289`）一样是一个无限循环，区别在于没有请求时它不是靠 `sleep` 空转，而是靠 `notifier.notified().await` 真正把这个 Tokio 任务挂起，等有新请求时被唤醒——这是 `## 6.4`（空转吃满一个核）在异步运行时下的正确解法，`asyncio`/`tokio` 这类事件循环天然提供"挂起等通知"的原语，不需要 `## 3.3` 里那种手写的 `time.sleep(0.002)` 折中方案，`## 5.2` 会具体展开这层差异。

客户端断连的检测，TGI 走的是第三种机制——检查 channel 是否已经被丢弃：

```rust
// tgi:backends/v3/src/queue.rs:276-284（节选）
'entry_loop: while let Some((id, entry)) = self.entries.pop_front() {
    // Filter entries where the response receiver was dropped (== entries where
    // the request was dropped by the client)
    if entry.response_tx.is_closed() {
        metrics::counter!("tgi_request_failure", "err" => "dropped").increment(1);
        continue;
    }
```

`response_tx.is_closed()`——如果对应的接收端（连着 HTTP 响应流）已经被丢弃，说明客户端已经断了，这条检查发生在**从等待队列里往外弹、准备组批**的那一刻，也就是说 TGI 对"断连的请求"最迟能在它被排进下一批之前拦下来，比本库"必须先写一次才能发现断连"更早一步；但如果一条请求已经在批里跑到一半（正在 decode）才断连，这里的检查就够不着了——`is_closed()` 只在 `queue.rs` 这个"弹出待跑批次"的路径上生效，跟本库、vLLM 在**生成过程中**捕获写失败/协程取消是不同的检测时机，各自覆盖了断连发生的不同阶段。TGI 用 Rust 写整个 router 层这件事本身，`[[04-Rust在推理栈里到底占了什么位置]]` 会专门展开，本篇只取跟"两道队列"这个结构直接相关的部分。

## 5. 设计决策与代价

### 5.1 全局只用一个引擎线程，不开多个引擎副本抢锁或分片

- **为什么这样**：`## 2.1` 已经证明过，多个执行体同时碰同一份引擎状态会产生静默错误。只用一个线程碰引擎，从根上让"竞争"这件事不可能发生——不需要锁，因为没有第二个人跟它抢。
- **不这样会怎样**：要么加锁（代价是所有请求排成一条队在锁外面等，`[[04-连续批处理]]` 的调度优势没了），要么彻底分片成多个独立的引擎实例（每个有自己的一份 `pool`/`prefix`），后者能提升吞吐，但前缀缓存的命中率会被分片打散——两条本该命中同一段缓存的请求如果被分到了不同分片，谁也看不到谁，`[[05-前缀缓存]]` 的收益直接打折。
- **什么时候可以不这样**：单机多 GPU、需要数据并行时，真实引擎（`## 4.1` 的 vLLM `EngineCore` 支持多副本）确实会跑多个引擎实例，但**每个实例内部**依然只有一个线程/一套调度逻辑碰自己的状态——"单引擎线程"这条约束是在"一个引擎实例的内部"成立的，不是在说"整个系统只能有一个引擎实例"。

### 5.2 不用 asyncio，用线程 + `queue.Queue`

- **为什么这样**：`_lab/serve.py:18-20` 的模块 docstring 已经把理由摆明——真实引擎多数用 asyncio，结构和本库完全一样（`## 4.1`、`## 4.3` 已经逐行验证过），但 `async`/`await` 关键字、事件循环调度这些语法会把"两道队列"这个主干藏进语言机制里，第一次读代码的人容易先被 `async def` 绕晕，看不清"这本质上就是一个入队列一个出队列"这件事。本库刻意用最朴素的 `threading.Thread` + `queue.Queue`，让队列本身可见。
- **不这样会怎样**：教学之外，纯线程模型在真实生产场景下有实打实的代价——`## 4.3` 里 TGI 用 `notifier.notified().await` 在没有请求时真正挂起、零 CPU 开销；本库只能用 `time.sleep(0.002)` 定期轮询（`## 3.3`），存在最多 2ms 的响应延迟，也存在"轮询本身"这一点点持续开销。异步运行时能更精细地管理大量并发连接（几千个 HTTP 长连接用几千个线程去扛，线程本身的内存和调度开销会成为瓶颈；同样数量的协程要轻得多）。
- **什么时候可以不这样**（即什么时候线程模型本身就够用，不必上 asyncio）：并发连接数不大、单条请求生命周期内不需要频繁在多个"等待中"的操作之间切换的场景——本库的教学定位就是如此。一旦要同时管理成千上万条长连接（真实的生产级推理服务），asyncio 或者干脆像 TGI 那样上 Rust 的 `tokio`，会是更合适的选择。

### 5.3 用 SSE，不用 WebSocket

- **为什么这样**：`_lab/serve.py` 的流式协议是单向的——服务端往外推 token，客户端不需要往回发任何东西（除了 HTTP 请求本身）。SSE（Server-Sent Events）是为"单向、服务端推、纯文本"这个场景量身设计的：建立在普通 HTTP 之上，`text/event-stream` + `chunked` 编码就能实现，`## 3.4` 已经完整展示过实现只需要几行手写的分块协议，任何 HTTP/1.1 客户端（`curl -N`）不需要额外协议升级就能读。`## 4.1`、`## 4.2` 的 vLLM、SGLang 走的都是同一条路。
- **不这样会怎样**（换成 WebSocket）：WebSocket 是全双工的，需要一次协议升级握手（`Upgrade: websocket`），实现和调试的复杂度比 SSE 高一截，换来的是"客户端也能随时往回发东西"这个能力——但一次文本补全请求根本用不上这个能力：客户端唯一可能想做的"往回发"的事是"我不要了"，这件事 `## 2.4` 已经证明了不需要一个双向协议，靠 HTTP 连接本身的关闭（进而触发 `BrokenPipeError`）就能间接传达。
- **什么时候可以不这样**：如果协议本身需要双向交互——比如客户端流式发送音频、服务端流式返回转写结果并且客户端需要根据中间结果实时调整发送内容（真正的双工对话），或者需要在一条连接上并发多路请求/取消而不想为每条开一个新的 HTTP 连接，这些场景 WebSocket（或者更贴近推理场景的 gRPC 双向流）会比 SSE 更合适。

### 5.4 背压用 503 + Retry-After，不用无限排队

- **为什么这样**：`_lab/serve.py:24-25` 的模块 docstring——"收下来只会让所有人的延迟一起涨，而且没人知道发生了什么"。`## 3.3` 的 `submit()` 在队列（`in_q` + `waiting`）达到 `max_queue` 时立刻返回 `False`，`do_POST`（`## 3.4`）立刻回 503 并带上 `Retry-After: 1`——这是把"我现在满了"这件事**显式、立刻**地告诉调用方，调用方可以据此决定是重试、降级还是报错给它自己的上游，而不是被晾在一个不知道要等多久的队列里。
- **不这样会怎样**（无限收，排队等）：队列会无限增长，每条新请求的排队时间（TTFT 的排队部分，`## 3.1` 已经讲过 TTFT 把排队和 prefill 混在一起算）会越来越长，而且这个变慢是**渐进且沉默**的——没有任何一个明确的时间点告诉调用方"现在系统已经不行了"，只有所有人的延迟曲线一起缓慢抬升，出问题时很难第一时间定位到"是队列堆太深了"还是"是某个环节真的变慢了"。
- **什么时候可以不这样**：如果调用方本身有能力、也愿意承受长时间的排队（比如一个离线批处理任务，只要求最终跑完、不在乎单条请求等多久），或者上游已经有一层限流/排队机制（比如一个专门的调度队列服务），服务这一层直接拒绝反而是重复劳动——但即便如此，服务这一层依然应该有一个**显式的上限**，只是这个上限可以设得很大，而不是彻底取消这层检查。

### 5.5 采样点整段复制父类，而不是从一开始留一个可替换的钩子

- **为什么这样**（这是既定现状，不是"应该"）：`## 3.2` 已经完整讲过——`[[04-连续批处理]]`、`[[05-前缀缓存]]` 里的 `Engine` 把"选下一个 token"写死成 `argmax`，是为了让"优化不许改变输出"这条贯穿全库的正确性契约能用最简单的逐位对比去验证。这个选择在写那几篇的当下是对的：**确定性是那几篇的教学目标，可替换的采样钩子不是**。
- **不这样会怎样**（选了 argmax 写死之后，服务化这一步的代价）：`## 3.2` 已经具体展示了代价——`ServingEngine` 必须把 `_admit()`、`step()` 两个方法**整段复制**，只为了改一行。这不是"重复代码"这种可以靠重构轻松解决的表面问题：两份 `_admit()` 未来如果 `[[05-前缀缓存]]` 那边的调度逻辑发生任何改动（哪怕只是修一个边界条件的 bug），这里的复制品**不会自动跟着更新**，需要人工同步，而人工同步这件事在真实项目里几乎必然会有遗漏的一天。
- **什么时候可以不这样**：**从 0 写的话，这个位置一开始就要留缝**——`_lab/serve.py:129-131` 的类文档字符串把这条结论直接摆明了：真实引擎从第一天就把"选下一个 token"做成一个独立对象（sampler / logits processor 管线），不是因为预见到了"以后要加采样"，而是因为这个位置天然是一个会持续长出新需求的地方——logit bias、结构化输出的 token 掩码（`[[04-结构化输出]]`）、投机解码的校验（`[[03-投机解码]]`）——全都要在"选下一个 token"这一步插一手。如果一开始就把这里做成 `self._select_next(logits, req) -> int` 这样一个可注入的钩子（哪怕第一版实现就是 `return int(logits.argmax())`），后续加采样、加任何一种 logits 处理器，都只需要替换这一个钩子的实现，不需要复制整个调度方法。这是本篇留给"如果你要自己从 0 写一个引擎"的读者的一条具体建议，不是事后诸葛亮式的批评——本库自己在 `[[01-第一版-朴素前向]]` 到 `[[05-前缀缓存]]` 这几篇写的时候，同样没有一开始就留这个缝，本篇就是留下的这笔债务真实到期的地方。

### 5.6 TTFT 与 TPOT 必须分开报，不能合成一个"延迟"

- **为什么这样**：`_lab/serve.py:108-110` 的属性文档字符串——TTFT 归排队和 prefill 管，TPOT 归 decode 的访存带宽管，`[[02-加上KV缓存]]`、`[[03-分页KV与块管理]]` 已经从算术强度的角度讲过这两个阶段受完全不同的因素支配（prefill 是计算密集、decode 是访存密集，`_PLAN.md` §3.1 那张算术强度表就是这条判断的数值依据）。把两个受不同瓶颈支配的量平均成一个数字，等于是"把两台机器的体温加起来除以二"——这个平均数不对应任何真实存在的物理量，看着有一个数字，实际什么信息都没传达。
- **不这样会怎样**：假设一个系统 TTFT 涨到了 2 秒（排队严重）但 TPOT 依然是 20ms/token（decode 完全正常），另一个系统 TTFT 只有 50ms 但 TPOT 涨到了 200ms/token（decode 本身变慢了，可能是批太大或者显存带宽打满）——两个系统合并出来的"平均延迟"完全可能长得差不多，但故障原因南辕北辙：前者要去查调度器和准入策略，后者要去查 decode 这一步的批大小和内存访问模式。合并成一个数字之后，这两种截然不同的故障会看起来一模一样，排查时会先南辕北辙地查错方向。
- **什么时候可以不这样**：如果场景里从一开始就只有单条请求、不存在排队（比如本地单用户离线跑一个脚本），TTFT 里"排队"这部分天然是零，TTFT 退化成纯粹的 prefill 时间，这时候一个综合的"总耗时"确实也能大致反映情况——但即便如此，把 prefill 和 decode 分开报依然是更细的粒度，"可以不分开"只是"不分开的坏处变小了"，不代表分开没有意义。

## 6. 常见错误与踩坑

### 6.1 多线程直接调 `engine.step()`——静默串 KV

`## 2.1` 已经完整推演过这条错误。**现象**：单条请求测试全部通过，正确性看起来毫无问题；一旦有真实并发流量，会开始出现**极少数**、**无法稳定复现**的输出异常——某条请求的输出读起来通顺但答非所问，或者两条本该完全独立的请求偶尔生成出诡异地相似/纠缠的内容。因为 bug 依赖具体的线程调度时序，同一份代码、同一批输入，这次跑对了不代表下次还对，几乎不可能靠"重跑一遍复现"来定位，是本库到目前为止**最难排查**的一类错误，因为它连"稳定复现"这个调试的第一步都做不到。

### 6.2 客户端断连不回收——显存（这里是块池）只涨不跌

`## 2.4` 完整讲过这条链路。**现象**：单独压测（不主动断连）完全正常，块池用量随负载起伏、空闲时能回落到 0（`## 1` 的 `/metrics` 里 `blocks_in_use` 可以验证）；一旦流量里存在真实的客户端断连（超时重试、用户中途关闭页面、反向代理主动切断慢连接），块池用量会呈现一条只涨不跌的曲线——每一条被断连的请求都会按 `max_new` 老老实实跑到底才释放资源，在此之前它占的块对系统来说是完全"沉默"的占用，不会出现在任何一条错误日志里，`/metrics` 的 `blocks_in_use` 是唯一能看出异常的地方，但前提是有人在盯着它看。

### 6.3 SSE 响应漏了 `transfer-encoding: chunked`

**现象**：服务端代码逻辑完全正确，token 也确实按顺序 `write` 出去了，但客户端表现为**一直卡着不返回任何东西，直到整个响应发送完毕才一次性吐出全部内容**，或者更糟——直接报错说响应格式不对。原因是 HTTP/1.1 在没有声明 `content-length` 也没有声明 `transfer-encoding: chunked` 时，客户端不知道这个响应体的边界在哪里，很多 HTTP 客户端库会等到连接关闭才认定"响应结束"，于是所谓的"流式"变成了"服务端悄悄攒了半天，最后一口气吐出来"，跟同步阻塞没有任何区别，只是延迟更差（多等了一整个生成过程）。

### 6.4 空转 `while` 吃满一个 CPU 核

`## 3.3` 的 `EngineLoop.run()` 里 `time.sleep(0.002)` 那一行如果被漏掉（比如把 `else` 分支整个删掉，让 `while` 循环在没有任何请求时也一刻不停地转），**现象**：服务进程启动之后，哪怕完全没有流量、一个请求都没打进来，CPU 占用率也会常驻接近 100%（占满一个核），`_lab/serve.py:204-206` 的类文档字符串把这条现象总结成一句话——"服务一起来风扇就转"。这条错误在本地开发时很容易被忽略（反正就自己一个人用，卡不卡感觉不出来），部署到共享的机器/容器上会变成一个持续消耗计费资源、却什么活都没干的空转进程。

### 6.5 把 TTFT 和 TPOT 混成一个"延迟"数字

`## 5.6` 已经讲过后果。**现象**：监控面板上只有一条"平均延迟"曲线，曲线抖动或者抬升时，无法判断问题出在"排队/准入变慢了"还是"每步 decode 本身变慢了"——这两类问题的排查方向、修复手段完全不同（前者调 `max_running`/`max_queue`/准入策略，后者查 batch 大小、显存带宽、算子本身），混在一起报的这一个数字，制造了一种"看着有监控，实际定位不了故障"的假象，比完全没有监控更容易让人产生虚假的安全感。

### 6.6 真实插曲：`selftest` 自己的并发客户端一度不接背压，被当成崩溃

`## 1` 留了个尾巴，这里把完整的返工史讲一遍——因为它是一个真事，从发现到定位到修复都留下了可以核对的痕迹，比任何虚构的反面教材都更有说服力。

**现象。** 本篇最初写作时，连续跑了 5 次 `python serve.py --selftest`，其中 2 次第 8 条断言（`## 3.6` 的编号，6 条并发请求全部返回、块全部归零）直接在客户端线程里炸出一个未捕获的 `HTTPError: 503`，而不是干净地打印 `[FAIL]`。`selftest()` 起服务时把 `max_queue` 设成了 4（`_lab/serve.py:483-484`），第 8 条断言起了 **6** 个线程同时打请求——当这 6 个线程的调度恰好挤在同一小段时间窗口内同时到达 `submit()`，`## 5.4` 讲的背压机制会**如实**拒绝掉超出 `max_queue` 的那几条，返回 503。乍一看，这跟"服务在并发下崩了"长得一模一样：一个未捕获的异常、一次不完整的返回、一次失败的断言。

**根因。** 但拒绝这几条请求，恰恰是 `## 5.4` 讲的背压机制**该做的事**——满了就说不，不能悄悄全收下来再拖慢所有人。真正没接住这件事的，是当时那个简化过的测试客户端：`post()` 辅助函数只用 `urllib.request.urlopen()` 直接拿返回值，`urlopen` 在遇到 4xx/5xx 状态码时会抛 `HTTPError`，这个异常从并发线程的 `lambda` 里一路往外冒，`outs` 列表永远等不到这一条的结果，断言自然失败。这条链路指向一个容易被忽略的事实：**背压不是服务端一家的事**。服务端负责在满了的时候及时说不，客户端负责听得懂"说不"是什么意思——如果客户端把 503 当成一种未预料的崩溃，那么一次完全符合设计的正常拒绝，在客户端这一侧就会表现成一次随机的、看起来毫无规律的失败。

**修法。** `_lab/serve.py` 现在的 `post()`（`_lab/serve.py:491-513`）把这件事补上了，文档字符串（`_lab/serve.py:492-499`）直接把这条教训写进了代码本身：

```python
# _lab/serve.py:504-513（节选）
for attempt in range(60):
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.read().decode() if stream else json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code != 503:
            raise
        n_503[0] += 1
        time.sleep(float(e.headers.get("retry-after", 0) or 0) or 0.05)
raise RuntimeError("重试 60 次仍被拒 —— 这才是真的有问题")
```

三个细节都不是随手写的：**只对 `503` 重试**（`_lab/serve.py:508-510`）——`if e.code != 503: raise` 把其它任何状态码原样重新抛出，因为 `400`（`## 3.4` 第 3 条检查）代表的是请求本身有问题，重试不会让一个越界的 `max_tokens` 变得合法，对 400 重试是把"客户端的错"错当成"服务端暂时的忙"，会掩盖真正的 bug；**退避而不是立刻重打**（`_lab/serve.py:512`）——优先读服务端返回的 `retry-after` 头（`## 3.4` 已经讲过 `do_POST` 会带上这个头），没有的话退化成 0.05 秒，不是"服务说了满了，客户端立刻又打一发"这种会让背压形同虚设的重试；**有上限**（`_lab/serve.py:504,513`）——最多重试 60 次，仍然被拒就 `raise RuntimeError`，因为如果 60 次退避之后系统还是满的，那就不再是"背压偶发命中"，而是真的有问题（比如引擎线程卡死、`max_queue` 设得离谱地小），这时候应该让测试如实失败，而不是无限重试掩盖一个真实故障。

**推广。** 这条教训不是 `selftest` 自己的特例。**任何带背压的服务，只要压测脚本、SDK、网关不处理它的限流状态码（503、429……），测出来的错误率就是假的**——它量的不是"服务扛不住了"，量的是"客户端没听懂服务在说什么"。真实世界里这个坑比想象的更容易踩：很多压测工具、很多手写的调用脚本默认只关心 2xx，一旦目标服务真的启用了背压，这些工具会把"系统在正确地自我保护"报告成"系统在随机地失败"，如果照着这份误报去扩容或者回滚，解决的是一个不存在的问题。

**这本身也是一课。** 加上重试之后，本地连续跑了 5 次 `--selftest`，被背压拒绝并重试的次数分别是 `0、0、2、1、0`——全部 9 条断言都通过，但**背压确实在大约一半的运行里真的被触发了**，只是现在有了第 9 条断言（`## 3.6`）把这个数字如实报出来，而不是让它在触发时表现成一次崩溃、不触发时又完全隐身。这条数据序列本身也值得多看一眼：`0、0、2、1、0`——多数时候一次都不会触发，少数时候触发一到两次，从来没有触发过更多次。**这正是最难查的一类问题**：它不是"从不发生"（那样可以放心忽略），也不是"总是发生"（那样第一次测试就会暴露），而是"大部分时候看起来是对的"——这种概率分布下的 bug，靠"跑一次看看"几乎不可能建立起对它存在与否的信心，需要像这里一样连续跑上好几次、并且让偶发事件本身变得可观测（第 9 条断言），才有机会把它从"运气不好"甄别成"设计缺口"。

## 7. 自测题与延伸阅读

### 自测题（闭卷回答，答不上去就回对应小节重新推一遍）

1. `## 2.1` 说多线程直接调 `engine.step()` 是"静默"的错误。具体解释：为什么这类错误几乎不可能靠"重跑一遍看看还错不错"来定位？跟 `[[05-前缀缓存]]` `## 6.2` 讲的"忘了 incref"错误相比，两者在"依赖具体时序才会触发"这一点上是不是同一类问题的两种表现？
2. `## 2.2` 描述的旅程里，`Job` 对象上哪些字段是"引擎线程写、HTTP 线程读"，哪些是反过来？`cancelled` 字段为什么是唯一的例外？如果颠倒了某个字段的读写方向（比如让 HTTP 线程去写 `t_first`），会破坏什么约束？
3. `## 2.4` 讲的客户端断连回收链路有三处配合缺一不可。具体写出这三处分别是哪个函数、哪一行，以及如果只缺中间那一处（`_drain` 不做标记翻译），系统会表现出什么现象——`job.cancelled` 会不会被置位？`req.abort` 会不会被置位？块最终会不会被回收？
4. `## 3.2` 指出 `ServingEngine.step()` 比文档字符串描述的"只改了两处 argmax"多做了两件事。这两件事分别是什么？如果把它们删掉（只留 argmax 换成 `_pick` 这一处改动），`## 2.4` 的取消链路会在哪一步彻底失效？
5. `_lab/serve.py:224` 的背压检查用的是 `self.in_q.qsize() + len(self.engine.waiting)`，不是只看 `in_q.qsize()`。构造一个具体场景，说明只看 `in_q.qsize()` 会导致背压检查在什么时间点上产生错误的低估。
6. `## 4.1`、`## 4.2`、`## 4.3` 里，vLLM、SGLang、TGI 三者检测"客户端断连"分别用的是什么机制？三者在"检测到断连后多快能真正回收资源"这件事上，哪一个明确慢于其它两个，为什么？
7. `## 5.5` 说"从 0 写的话这个位置一开始就要留缝"。具体设想一下：如果 `[[04-连续批处理]]` 的 `Engine._admit()`/`step()` 一开始就调用一个 `self._select_next(logits, req) -> int` 钩子（默认实现是 `argmax`），本篇的 `ServingEngine` 还需不需要整段复制这两个方法？需要改的会剩下什么？
8. `## 6.6` 讲的那次真实插曲里，旧版 `selftest` 的根因是代码本身哪一处假设不成立（不是"哪一次跑挂了"）？现在的 `post()`（`_lab/serve.py:504-513`）为什么只对 `503` 重试、遇到其它状态码要 `raise`——如果改成"任何 `HTTPError` 都重试"，`## 3.4` 第 3 条准入前拒绝（400）的那条断言会变成什么样？如果把 `raise RuntimeError` 那一行也删掉、改成重试次数用完就默默放弃，`## 3.6` 第 8 条"6 条并发全部返回"这条断言还能不能测出真正的资源泄漏？

### 延伸阅读

- `[[06-采样-停止条件-流式输出]]`——本篇 `Job` 携带的 `SamplingParams`/`Sampler`/`StreamGuard`/`should_stop` 全部来自这一篇，`## 3.1`、`## 3.2`、`## 3.4` 反复用到它定义的类型和函数。
- `[[04-连续批处理]]`——`ServingEngine` 继承的 `Engine`、`EngineLoop` 每一步调用的 `step()`，调度逻辑的主干都在这一篇，本篇没有改动它，只是在外面包了两道队列。
- `[[05-前缀缓存]]`——`## 2.1` 讲的"静默串 KV"跟这一篇 `## 2.5`、`## 6.2` 讲的引用计数错误是同一个症状家族的两种成因（一个是并发竞争，一个是逻辑遗漏），`ServingEngine._admit()` 原样复用了这一篇的前缀缓存匹配逻辑。
