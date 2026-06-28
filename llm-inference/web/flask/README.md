# Flask 部署 LLM 推理服务

> 用最轻量的 WSGI 框架，把"推理引擎 + tokenizer"包成一个 HTTP 接口（如 `/predict`），让别的服务用一条 `curl`/HTTP 请求就能调起大模型生成。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/web/fastapi/README]] [[llm-inference/README]] [[ai-infra/算力/昇腾NPU]] [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | Flask = WSGI + 同步 + 最小内核 |
| 1 | 它解决什么问题 | 命令行脚本 → 网络服务；为什么要"包一层 HTTP" |
| 2 | 整体架构与请求生命周期 | WSGI / route / 全局引擎单例 |
| 3 | 同步模型的代价 | 一线程一请求；GIL；GPU 串行 |
| 4 | 本目录的真实例子拆解 | Qwen + MindSpore-Lite + Ascend，`/predict` |
| 5 | 模型只加载一次（最关键的坑） | 启动期 warm-up vs 每请求重载 |
| 6 | 流式输出在 Flask 里怎么做 | 生成器 + `stream_with_context` |
| 7 | 并发与背压 | 串行队列、信号量、为什么要排队 |
| 8 | 生产部署 | 开发服务器 ≠ 生产；gunicorn / 绑卡 |
| — | 配置/参数说明（讲含义）| 采样参数、worker 数、超时 |
| — | 常见问题 | 表格速查 |

> 说明：本文讲**机制与思路**。涉及精确版本号、CLI 默认值、具体 API 签名处，一律以**官方文档/源码为准**，本文不编造。本目录的代码示例来自同级 `llm-qwen-mindspore-lite.py`（Qwen-7B + MindSpore-Lite，跑在昇腾 Ascend 上）。

---

## 0. 一句话锚点

**Flask 是一个基于 WSGI 标准、默认同步、内核极小的 Python Web 微框架。** 在 LLM 部署里，它扮演最简单的"前台"：接住 HTTP 请求 → 取出 `input_text` 与采样参数 → 调一次推理引擎 → 把生成结果打成 JSON 返回。它本身**不做 GPU/NPU 推理**，真正算 logits 的是 MindSpore-Lite / transformers / vLLM 这些引擎。

记住一条主线：

```
客户端 ──HTTP/JSON──► Flask（WSGI 前台，CPU 轻活）──►推理引擎（GPU/NPU 重活）
                       同步、一次一请求              tokenizer + 自回归生成
```

Flask 的定位词是"**简单、够用、上手快**"。当你只是想"把一个已经能跑的推理脚本，变成别人能 `curl` 的接口"，且**并发要求不高**时，Flask 是最低心智负担的选择。

---

## 1. 地基：它解决什么问题

### 1.1 从"命令行脚本"到"网络服务"

一个能跑的推理脚本长这样：加载模型 → 输入一句话 → 打印生成结果。问题是：

- 它只能在**本机命令行**跑，别的程序/同事/前端用不上；
- 每次跑都要**重新加载几十 GB 权重**（几十秒到几分钟），无法复用；
- 没有标准协议，无法被监控系统、网关、其他微服务集成。

"包一层 HTTP"就是把这个脚本变成**常驻进程**：模型**启动时加载一次**常驻显存/内存，之后通过 HTTP 接口反复调用。

```
之前（脚本）：             之后（Flask 服务）：
$ python infer.py "问题"   进程常驻，模型只加载一次
  └ 每次重载权重(慢)        客户端A ─┐
  └ 只能本机用             客户端B ─┼─HTTP─► Flask ─► 已加载好的引擎
                          客户端C ─┘         （权重一直在显存里）
```

### 1.2 为什么先选 Flask

| 维度 | Flask 的特点 |
|---|---|
| 内核大小 | 极小，只做"路由 + 请求/响应"，其余靠扩展 |
| 上手成本 | 低，几行就起一个接口，适合 demo / 内部工具 |
| 同步心智 | 直白：一个请求从头到尾在一个线程里跑完，好调试 |
| 生态 | 成熟，文档多，踩坑帖海量 |

### 1.3 Flask vs FastAPI：一句话说清取舍

| 维度 | Flask（WSGI，**同步**） | FastAPI（ASGI，**异步**） |
|---|---|---|
| 并发模型 | 线程/进程，**等待即占坑** | event loop，等待即让出 |
| 高并发 I/O-bound | 弱，靠堆线程/进程 | 强，单线程扛海量等待 |
| 流式 SSE | 能做但别扭（生成器手撸）| 原生 `StreamingResponse` |
| 入参校验 | 手写 if 判断 | Pydantic 自动校验 + 文档 |
| 自动 API 文档 | 需插件 | 内置 OpenAPI/Swagger |
| 适用场景 | demo / 内部工具 / 低并发 | 生产高并发 / 流式网关 |

> 结论：**对外扛高并发、要流式、要 OpenAI 兼容网关 → 上 [[llm-inference/web/fastapi/README]]**；**只想快速把脚本变接口、并发不高 → Flask 够用。** 关键是：无论哪个框架，真正的吞吐都来自**引擎的批处理**，而不是 Web 框架本身。

---

## 2. 整体架构与请求生命周期

### 2.1 WSGI 三层栈

```
┌────────────────────────────────────────────────┐
│ gunicorn / uWSGI（进程管理器，生产用）            │  管多个 worker、重启、信号
│  ├── worker-0 ┐                                  │
│  ├── worker-1 ┤ 每个 worker = 一个独立进程        │  各自一份 Flask app + 引擎
│  └── worker-2 ┘   ├── Flask app（WSGI 应用）      │  你的应用：路由、业务
│                   │    ├── 路由表 /predict ...     │
│                   │    └── 全局引擎句柄 pipeline    │  ← 模型常驻在这里
└────────────────────────────────────────────────┘
                    │
                    ▼
            GPU / NPU 推理引擎（MindSpore-Lite / vLLM ...）
```

- **WSGI**：Web Server Gateway Interface，Python 同步 Web 的标准接口（一个 `app(environ, start_response)` 可调用对象）。Flask app 就是一个 WSGI 应用。
- **开发服务器**（`app.run()` 起的那个）：**只供开发调试**，单线程/简单多线程，**绝不能上生产**（见 §8）。
- **生产服务器**：gunicorn / uWSGI 拉起多个 worker 进程，做存活监控、平滑重启。

### 2.2 一条请求的生命周期（同步）

```
① 客户端 POST /predict {"input_text": "保持健康的秘诀"}
   ─► ② WSGI 服务器收下，分给一个空闲 worker/线程
   ─► ③ Flask 路由匹配到 @app.route('/predict')
   ─► ④ 取出 request.json，手动取字段（无自动校验）
   ─► ⑤ 调推理引擎：tokenizer 编码 → 自回归生成 → 解码
            └─ 这一步占住整个线程，期间该线程不能服务别的请求 ★
   ─► ⑥ jsonify({'result': outputs}) 打成 JSON
   ─► ⑦ WSGI 写回 socket ─► 客户端收到结果
```

★ 是 Flask（同步）的核心特征：**第 ⑤ 步等 GPU 算完的整段时间，这个线程被完全占住**。这与 FastAPI 的"`await` 时让出控制权"形成鲜明对比，是后面所有并发讨论的根。

---

## 3. 同步模型的代价（必须先理解清楚）

LLM 生成一个回答动辄几百毫秒到几十秒，这段时间服务端几乎不耗 CPU——它在**等 GPU/NPU 算完**（典型 I/O 等待）。Flask 默认同步：

```
Flask 同步（开发服务器，假设单线程）：

请求A: [====生成中 占住唯一线程 5s====]
请求B:                                [等A完才轮到B] [====5s====]
请求C:                                                          [等...]
      └─ 后来的请求被前面的"卡"住，延迟线性叠加
```

即使开多线程（`threaded=True`）或多进程 worker：

- **每个在途请求都实打实占住一个线程/进程**。要扛 100 并发就要 ~100 个执行单元，线程切换 + 内存开销大；
- Python 有 **GIL**，纯 Python 计算无法真正并行；好在 GPU/NPU 推理时会**释放 GIL**（C 扩展里），所以多线程对"等 GPU"这类 I/O 等待**有一定帮助**，但远不如异步优雅；
- **更现实的瓶颈是 GPU 本身**：一张卡通常只放一份权重、一次只能高效跑有限的 batch。多个请求最终还是要在 GPU 上**排队**。

> 一句话：Flask 的同步不是"错"，而是"心智简单但并发天花板低"。低并发内部工具完全 OK；要扛公网高并发，要么上异步框架，要么把吞吐压力交给**会做连续批处理的引擎**（vLLM/TGI），Flask 只当薄网关。

---

## 4. 拆解本目录的真实例子（Qwen + MindSpore-Lite + Ascend）

同级的 `llm-qwen-mindspore-lite.py` 是一个**完整可用**的 Flask 推理服务骨架。它把 Qwen-7B-Chat 导出的 MINDIR 模型，用 MindSpore-Lite 在昇腾 Ascend NPU 上拉起，对外暴露 `/predict` 与一个仿 Triton 的 `/v2/.../generate_stream`。整体结构：

```
启动期（进程拉起时，只做一次）
  ├─ ms.set_context(device_target='Ascend')   选 NPU 后端
  ├─ QwenTokenizer(vocab_file=...)             加载 tiktoken 词表
  ├─ get_mindir_path(...)  取 prefill + increment 两个 MINDIR
  │      ├─ mindir_full_checkpoint  → 全量图（首次 prefill）
  │      └─ mindir_inc_checkpoint   → 增量图（逐 token decode）
  ├─ InferConfig(...) + InferTask.get_infer_task("text_generation")
  │      └─ 这里就是"推理引擎句柄" pipeline_task（常驻）
  └─ run_mslite_infer(pipeline_task, "hello")  ★ warm-up 预热一次

运行期（每来一个 HTTP 请求）
  /predict ──► data=request.json ──► input_text
            ──► run_mslite_infer(pipeline_task, input_text, args)
            ──► jsonify({'result': outputs})
```

### 4.1 为什么有"两个图"：prefill 与 increment

这是自回归推理的工程化体现，和 KV-cache 思想一脉相承（详见 [[llm-optimizer/kv-cache]]）：

- **prefill（全量图）**：把整段 prompt 一次性喂进去，并行算出所有位置、建立 KV-cache。这一步是"理解输入"。
- **increment（增量图）**：之后每生成一个 token，只算**新位置**，复用前面缓存的 K/V，避免重复计算整段历史。这一步是"逐字生成"。

```
prompt: "保持健康的秘诀"
  ┌─ prefill 图 ─► 并行处理全部 prompt token，建 KV-cache ─► 出第1个新 token
  └─ increment 图 ─► 用新 token + KV-cache ─► 第2个 ─► 第3个 ─► ... ─► EOS
       （每步只算 1 个位置，靠缓存省掉重复算 history）
```

### 4.2 采样参数：`/predict` 与 `/generate_stream` 的差别

例子里把采样参数封装在 `InferParam` 这个 dataclass 上。`/generate_stream` 允许请求体覆盖这些参数，而 `/predict` 用的是启动时固定的那套。各参数含义（**讲作用，不背默认值**）：

| 参数 | 作用 | 怎么权衡 |
|---|---|---|
| `temperature` | 缩放 logits，控制随机性 | 大→更发散有创意；小→更确定保守；趋 0 近似贪心 |
| `top_k` | 只在概率最高的 k 个候选里采样 | 小→更聚焦；大→更多样 |
| `top_p`（nucleus）| 取累积概率达到 p 的最小候选集 | 与 top_k 类似目的，按"概率质量"动态截断 |
| `repetition_penalty` | 惩罚已出现 token，抑制复读 | >1 减少重复；过大会伤通顺度 |
| `seq_length` | 模型支持的最大序列长度 | 受导出图与显存约束，超了要截断 |
| `predict_length`/`max_length` | 本次最多生成多少 token | 越大越慢越占显存；代码里会 `min(它, seq_length)` |
| `eos_token_id` | 命中即停止生成 | Qwen 这里 EOS 与 PAD 同为 `151643` |

> 这些采样参数是**所有**生成式推理通用的（vLLM/TGI/transformers 同样有），不是 Flask 的东西。Flask 只负责把它们从 JSON 里取出来、转交给引擎。

### 4.3 接口对比

| 路由 | 入参字段 | 行为 | 备注 |
|---|---|---|---|
| `/predict` | `input_text` | 用固定 `args` 跑一次，返回 `{'result': [...]}` | 最简单，参数不可调 |
| `/v2/models/ensemble/generate_stream` | `text_input` + 采样参数 | 允许覆盖采样参数，返回 `{'text_output': ...}` | 字段名仿 Triton/Ensemble 风格 |

注意：名字里虽带 `generate_stream`，但例子的实现是**先把结果全算完再一次性返回**（先 `for output in outputs: print` 再 `jsonify`），并不是真正逐 token 的流式。真要做流式见 §6。

---

## 5. 模型只加载一次：Flask 部署最关键的坑

### 5.1 致命反例 vs 正确做法

```
✗ 反例：把加载写进路由函数
   @app.route('/predict')
   def predict():
       pipeline = create_mslite_pipeline(args)   # 每请求重载几十 GB 权重！
       ...                                        # 几十秒延迟，显存反复抖动

✓ 正例（本目录例子的做法）：模块顶层加载一次，路由里复用
   pipeline_task = create_mslite_pipeline(args)   # ← 进程启动时执行一次
   run_mslite_infer(pipeline_task, "hello", args) # ← warm-up 预热
   @app.route('/predict')
   def predict():
       outputs = run_mslite_infer(pipeline_task, input_text, args)  # 复用
```

### 5.2 为什么要 warm-up（预热）

例子里加载完模型后特意跑了一次 `run_mslite_infer(..., "hello", ...)`。原因：**首次推理往往触发一堆一次性开销**——图编译/算子编译、显存分配、缓存建立、JIT。如果不预热，**第一个真实用户**会吃掉这段几秒甚至更久的冷启动延迟。预热把这笔账在"对外服务前"提前结清。

```
不预热：      [服务就绪] ─► 第1个真请求 [冷启动编译 + 推理 慢得离谱]
预热（推荐）：[服务就绪 → 自己跑一次 hello 把冷启动吃掉] ─► 第1个真请求 [正常速度 ✓]
```

### 5.3 多进程 worker 下的隐含代价

注意：**每个 gunicorn worker 进程都会各自执行一遍模块顶层代码**，也就是**各自加载一份模型**。所以"worker 数"直接等于"加载几份权重"。LLM 权重几十 GB，盲目加 worker → 多份权重抢同一张卡 → 显存 OOM。这条约束在 §8 详述。

---

## 6. 流式输出在 Flask 里怎么做

### 6.1 为什么要流式

非流式：用户盯着空白等十几秒，一次性收到全文，**首字延迟（TTFT）= 总生成时间**，体验差。流式：每吐一个 token 就推给前端，几百毫秒就看到"打字"，TTFT 大幅下降。指标定义见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]。

```
非流式：[........生成中(黑屏10s)........] ─► 全文
流式：  第50ms 首token ► token ► token ► ... ► [DONE]   用户立刻有反馈
```

### 6.2 Flask 的流式机制：生成器 + stream_with_context

Flask 同步框架做流式的办法是：让视图函数**返回一个生成器**而不是字符串。Flask 会边迭代边把 `yield` 出来的块写回 socket。配合 `stream_with_context` 可以在生成期间保持请求上下文可用。

```python
from flask import Response, stream_with_context

@app.route('/stream', methods=['POST'])
def stream():
    def gen():
        for tok in engine_token_iter(input_text):   # 引擎需能"逐 token 产出"
            yield f"data: {tok}\n\n"                  # SSE 文本块
        yield "data: [DONE]\n\n"
    return Response(stream_with_context(gen()),
                    mimetype='text/event-stream')
```

> 前提：**底层引擎必须支持逐 token 产出**。本目录例子的 `pipeline_task.infer(...)` 是"算完整段再返回"的，要做真流式得换成支持迭代式产出的引擎接口，或交给 vLLM/TGI 这类原生流式引擎。

### 6.3 Flask 流式的两个易错点

1. **反代缓冲**：Nginx 等默认会缓冲响应，把"流"攒成"块"，打字效果消失。需关闭缓冲（如 `proxy_buffering off;`，具体以 Nginx 文档为准）。
2. **WSGI 同步天花板**：每个流式连接会**长时间占住一个 worker/线程**（整段生成期间都占着）。这正是 Flask 做大规模流式不如 FastAPI 的根本原因——后者一个 event loop 就能扛海量并发流式连接。

---

## 7. 并发与背压

### 7.1 没有背压会发生什么

GPU/NPU 显存有限，能同时跑的请求数（KV-cache 容量）也有限。来多少收多少、无脑塞给设备，就会：`请求洪峰 ──► 全部接收 ──► 显存 OOM / 延迟雪崩 ──► 全员超时`。

**背压（backpressure）= 当下游处理不过来时，主动拒绝或排队上游请求**，保护系统不崩。

### 7.2 Flask 场景下的三道闸门

```
① worker/线程数本身就是天然限流：
   gunicorn -w 4  → 最多 4 个请求"在跑"，第 5 个在 WSGI 层排队

② 应用层信号量（更精细地限制"同时在推理"的数量）：
   sem = threading.Semaphore(MAX_CONCURRENCY)
   with sem:               # 超过 N 的请求在此等待，不冲进设备
       outputs = run_infer(...)

③ 引擎层批调度（最强一道，但要换引擎）：
   vLLM/TGI 内部按显存动态决定 batch 大小（连续批处理）
```

```
进来 100 并发，能力上限 16：

[16 个在设备上跑] ◄── 闸门(worker数/信号量) ──► [84 个排队/被拒]
        │                                          │
   正常出结果                            排队等名额 or 直接 429（背压）
```

### 7.3 数值直觉

设单请求平均占用设备 $\bar{t}=0.5\text{s}$、并发上限 $N=16$，理论吞吐上限约

$$\text{QPS} \approx \frac{N}{\bar{t}} = \frac{16}{0.5} = 32 \text{ req/s}$$

超过 32 req/s 持续涌入，队列会无限增长 → 必须靠限流把超额请求**快速拒绝**（fail fast，返回 429/503），而不是排队到超时。**宁可明确拒绝，也不要全员雪崩。**

---

## 8. 生产部署：开发服务器 ≠ 生产服务器

### 8.1 `app.run()` 为什么不能上生产

例子结尾的 `app.run(host='0.0.0.0', port=5000)` 启动的是 **Flask 内置开发服务器**：它面向调试（单/少线程、无优雅重启、无存活监控、性能与安全都不为生产设计）。**生产必须换成 WSGI 服务器**（gunicorn / uWSGI）来托管。

### 8.2 worker 数怎么定（和普通 Web 服务完全不同）

普通 CPU-bound Web：`workers ≈ CPU 核数`。**LLM 服务被 GPU/NPU 数量卡死**：一张卡通常只放一份权重，而**每个 worker 进程各加载一份模型**（见 §5.3）。所以经验起点是**每张卡一个 worker，并绑定独立设备**：

```
4 卡机器：
  worker-0 → 引擎实例 → 卡0   （各 worker 用环境变量绑定不同设备，
  worker-1 → 引擎实例 → 卡1     如 CUDA_VISIBLE_DEVICES / 昇腾的 DEVICE_ID/RANK_ID）
  worker-2 → 引擎实例 → 卡2
  worker-3 → 引擎实例 → 卡3
  └ 盲目加 worker → 多份权重抢同一张卡 → 显存 OOM
```

> 注意本目录例子里就读了 `DEVICE_ID` / `RANK_ID` 环境变量来选昇腾卡——这正是"每进程绑一张卡"的落地方式。具体环境变量名以昇腾/MindSpore 官方文档为准，见 [[ai-infra/算力/昇腾NPU]]。

### 8.3 命令示例（含义为主，参数以官方文档为准）

```bash
# 开发：内置服务器，仅调试用
python llm-qwen-mindspore-lite.py     # 即 app.run(...)，不要上生产

# 生产：gunicorn 托管（worker 数 ≈ 卡数，按显存定）
gunicorn 'llm-qwen-mindspore-lite:app' \
  -w 1 \                  # ★ LLM 通常每卡 1 worker；多了会抢卡 OOM
  -b 0.0.0.0:5000 \       # 绑定地址端口
  --timeout 300 \         # 长生成需调大，避免被误判卡死而杀掉
  --graceful-timeout 30   # 优雅退出窗口，等在途请求收尾
```

要点：长生成时 `--timeout` 太小会被当"卡死"杀掉，需放宽；每 worker 用环境变量绑定不同设备避免抢卡；生产前置一层 Nginx 做 TLS / 负载均衡 / 关闭 `proxy_buffering` 保流式。

### 8.4 全景部署图

```
客户端 ─► Nginx（TLS / 负载均衡 / 关闭 proxy_buffering 保流式）
            └─► gunicorn（进程主管：存活监控 + 平滑重启）
                  ├─ worker → Flask app + 引擎实例 │卡0
                  ├─ worker → Flask app + 引擎实例 │卡1   各 worker 绑卡，
                  └─ worker → Flask app + 引擎实例 │卡2   互不抢显存
                       （真正吞吐由引擎/批处理决定，Flask 只是薄网关）
```

---

## 配置/参数说明（讲含义，非照抄默认值）

把前面拼成一条最小可用骨架的"心智模型"：

```
1. 模块顶层：create_pipeline() 一次 + warm-up 预热（吃掉冷启动）
2. 路由 /predict：取 request.json → input_text + 采样参数
3. 调引擎：run_infer(pipeline, input_text, args)  ← 占住整个线程
4. jsonify 返回结果
5. （可选）流式：返回 Response(stream_with_context(gen()), 'text/event-stream')
6. （可选）信号量限流做背压，超额 429 fail fast
7. 部署：gunicorn -w {卡数}，每 worker 绑一张卡；前置 Nginx
```

各配置项的取舍：

| 配置 | 调大 | 调小 | 取舍 |
|---|---|---|---|
| `-w`（worker 数）| 更多实例，但每个各加载一份模型抢显存 | 省显存，吞吐低 | 受**卡数 / 单实例显存**硬约束，LLM 常 = 卡数 |
| 信号量 `N` | 更高并发，逼近 OOM | 更稳，吞吐降 | 按显存 / KV-cache 容量定，配合背压 |
| `--timeout` | 长生成不被误杀 | 早释放卡死请求 | 流式 / 长输出要放宽 |
| `max_length` 上限 | 允许长回答 | 省显存省时延 | 代码里 `min(max_length, seq_length)` 兜底 |
| `temperature/top_k/top_p` | 更发散多样 | 更确定保守 | 按任务调（创意写作 vs 事实问答）|

---

## 常见问题

| 问题 | 原因 | 处置 |
|---|---|---|
| 每次请求都重载模型、奇慢 | 把加载写进路由函数 | 模块顶层加载一次 + 全局复用 |
| 第一个用户特别慢 | 没预热，冷启动编译/分配落在首请求 | 启动后 warm-up 跑一次（如 `infer("hello")`）|
| 高并发就 OOM/雪崩 | 没背压；或 worker 数 > 卡承载，多份权重抢卡 | worker≈卡数、信号量限流、429 fail fast |
| 多 worker 显存爆 | 每个 worker 各加载一份模型 | worker 数按显存定，环境变量绑卡 |
| 名字叫 stream 却不流式 | `infer()` 算完整段才返回 | 用支持逐 token 产出的引擎 + 生成器 + `stream_with_context` |
| 流式没有"打字"效果 | 反代缓冲攒块 | 关闭 `proxy_buffering`，确认 `text/event-stream` |
| 中文输出成 `\uXXXX` | `json.dumps` 默认转义 | 加 `ensure_ascii=False` |
| 上线后又慢又不稳 | 用了内置开发服务器 `app.run()` | 换 gunicorn/uWSGI 托管 |
| 长生成被进程管理器杀 | `--timeout` 太小被当卡死 | 调大 timeout，对长输出/流式放宽 |
| 选错框架 | 要高并发/流式却用了同步 Flask | 高并发上 [[llm-inference/web/fastapi/README]]，或让引擎做连续批处理 |
| 以为换框架能提吞吐 | Flask/FastAPI 都不做批处理 | 吞吐靠**引擎**（vLLM/TGI）的连续批处理，Web 层只是薄网关 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，定位本文在"推理服务化"中的位置
- [[llm-inference/web/fastapi/README]] — 异步姊妹篇：要高并发/流式/OpenAI 兼容网关，看这篇
- [[llm-inference/README]] — 推理总览：引擎选型（vLLM/TGI/MindSpore-Lite）、连续批处理、KV-cache，与本文"前台 + 引擎"分工互补
- [[ai-infra/算力/昇腾NPU]] — 本例跑在昇腾 Ascend 上，设备选型 / DEVICE_ID-RANK_ID 绑卡背景
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] — TTFT / tokens-per-second 等流式体验指标定义
