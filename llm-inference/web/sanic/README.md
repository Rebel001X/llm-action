# Sanic 部署 LLM 推理服务

> 用"原生异步 + 多进程 worker"的高性能 Python Web 框架，把推理引擎包成高并发、可流式、可水平扩展的 HTTP 服务。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/README]] [[llm-inference/vllm/README]] [[llmops/kubernetes]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | Sanic = 异步框架 + 自带服务器 + 多进程 worker |
| 1 | 它解决什么问题 | 同步阻塞 vs 异步；为什么 LLM 服务必须异步 |
| 2 | 整体架构 | 主进程 / Worker Manager / worker 进程 / event loop |
| 3 | 路由、listener、ctx 单例 | 引擎只加载一次、按进程隔离 |
| 4 | 流式响应 | `response.stream` / SSE、边生成边吐 token |
| 5 | 与推理引擎对接 | 进程内 vs 进程外、同步引擎丢线程池 |
| 6 | 多进程 worker 与并发 | `--workers`、信号量背压、绑卡 |
| 7 | **fork vs spawn 启动方式**（本仓库踩坑点）| 多进程启动方法、CUDA 与 fork 冲突 |
| 8 | 部署与可观测性 | 命令示例、健康检查、优雅退出 |
| — | 配置示例（讲含义）| `sanic server.app --workers=N` |
| — | 常见问题 | 表格速查 |

> 说明：本文讲**机制与思路**。涉及精确版本号、CLI 参数默认值、具体 API 签名处，一律以**官方文档/源码为准**（https://sanic.dev），本文不编造。

---

## 0. 一句话锚点

**Sanic 是一个原生 `async/await`、且"自带高性能 Web 服务器与多进程进程管理器"的 Python 框架。** 在 LLM 部署里，它和 FastAPI 扮演同一角色——"前台/网关"：接 HTTP 请求 → 校验参数 → 把生成任务交给后端**推理引擎** → 把 token 流式吐回。它本身**不做 GPU 推理**，真正算 logits 的是 vLLM / TGI / Triton / transformers。

和 FastAPI 最大的区别是定位：

```
FastAPI：只是"应用框架"，要配 uvicorn(ASGI 服务器) + gunicorn(进程管理器) 才能跑生产
Sanic： "框架 + 服务器 + 进程管理器" 三合一，自带 server，一条命令拉起多 worker
```

记住主线：

```
客户端 ──HTTP/SSE──► Sanic（异步前台，自带 server，多 worker）──►推理引擎（GPU 重活）
                         等待即让出 loop                          连续批处理
```

---

## 1. 地基：它解决什么问题

### 1.1 同步阻塞为什么害死 LLM 服务

LLM 生成一个回答动辄要**几百毫秒到几十秒**。这段时间服务端几乎不耗 CPU——它在**等 GPU 算完**。这是典型的 **I/O 等待（I/O-bound）**，不是 CPU 算不过来。

传统**同步**模型里，一个线程处理一个请求，等待时线程被占死：

```
同步模型（4 个工作线程，每个生成耗时 5s）：

线程1: [req A 等GPU 5s............] [req E ...]
线程2: [req B 等GPU 5s............] [req F ...]
线程3: [req C 等GPU 5s............]
线程4: [req D 等GPU 5s............]
      └─ 第 5 个请求必须排队，因为 4 个线程全被"等待"占满
```

线程在等 GPU 期间**什么都没干，却占着坑**。扛 100 并发就要 100 线程，切换 + 内存开销爆炸。

### 1.2 异步并发：等待时把控制权让出去

`async def` + `await` 的本质是**协作式调度**：一个请求 `await` 在等 GPU/等网络写回时，event loop 把控制权交给**别的请求**，单线程就能"同时"推进成千上万个等待中的请求。

```
异步模型（单 worker 的 event loop）：

t0: req A 提交任务 → await（让出）
t0: req B 提交任务 → await（让出）
t0: req C 提交任务 → await（让出）
    ... loop 在多个 await 点之间穿梭，谁好了先处理谁
t5: req A 的 token 来了 → 唤醒 A 的协程 → 写回客户端
```

> 铁律：**异步函数里绝不能放同步阻塞调用**（同步 `requests`、`time.sleep`、CPU 密集纯 Python 循环、阻塞式 `.generate()`）。一旦阻塞，整个 worker 的 event loop 卡死，这个 worker 上所有并发请求一起冻结。阻塞活儿要丢**线程池**或**独立进程**。

### 1.3 Sanic 的特别之处：异步 + 多进程一起上

FastAPI 单进程靠一个 event loop 扛 I/O 并发；Sanic 同样是异步，但它内置一个**多进程 worker 模型**——可以一条命令起 N 个 worker 进程，每个进程各跑一个 event loop。

为什么对 LLM 服务有用？因为 LLM 服务的瓶颈往往不是单 loop 的 I/O，而是 **GPU**：一张卡装一个引擎实例。Sanic 的"多 worker"恰好可以**一个 worker 绑一张 GPU**，天然贴合多卡部署。

| 维度 | Flask（WSGI 同步） | FastAPI（ASGI 异步） | Sanic（异步 + 自带 server/manager） |
|---|---|---|---|
| 并发模型 | 线程/进程，等待即占坑 | event loop | event loop（每 worker 一个）|
| 服务器 | 需 gunicorn/uWSGI | 需 uvicorn | **自带**，无需外挂 ASGI server |
| 进程管理 | 需 gunicorn | 需 gunicorn | **自带 Worker Manager** |
| 多 worker 拉起 | 外部工具 | 外部工具 | `--workers=N` 一条命令 |
| 流式 | 别扭 | `StreamingResponse` | `response.stream` / SSE |

---

## 2. 整体架构：主进程 + Worker Manager + worker

Sanic 在生产模式下不是"一个进程"，而是一棵**进程树**：

```
┌─────────────────────────────────────────────────────────────┐
│ 主进程（serve_start）                                         │
│   └── Worker Manager（进程主管：拉起/监控/重启 worker）         │
│         ├── Worker 进程 0 ── event loop ── Sanic app ──► GPU0  │
│         ├── Worker 进程 1 ── event loop ── Sanic app ──► GPU1  │
│         └── Worker 进程 2 ── event loop ── Sanic app ──► GPU2  │
│              每个 worker：                                     │
│                ├── 路由表 /v1/...                              │
│                ├── 中间件（日志/限流/CORS）                     │
│                └── app.ctx.engine（推理引擎句柄，本进程单例）   │
└─────────────────────────────────────────────────────────────┘
```

- **主进程**：解析配置、绑定 socket（或交给 worker 复用端口）、启动 Worker Manager。
- **Worker Manager**：Sanic 自带的进程主管，相当于 gunicorn 之于 uvicorn——负责把 worker 拉起 N 份、做存活监控与重启。这是 Sanic"不需要外挂 gunicorn"的关键。
- **Worker 进程**：每个跑一个独立 event loop 和一份 Sanic app 实例。**注意：每个 worker 是独立进程，内存不共享**——引擎、模型、全局变量在每个 worker 里都是各自一份。

### 2.1 一条请求的生命周期（单个 worker 内）

```
① TCP/HTTP 入 ─► ② Sanic server 解析为 Request 对象 ─► ③ request 中间件链（日志/限流/鉴权）
   ─► ④ 路由匹配（按 method + path）
   ─► ⑤ 手动/Pydantic 校验请求体（错就 400/422，不进业务）
   ─► ⑥ 取 app.ctx.engine（本 worker 的引擎单例）
   ─► ⑦ 业务 handler：把 prompt + 采样参数 提交给引擎
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
        非流式：await 完整结果            流式：response.stream + async for token
                    ▼                             ▼
        ⑧ json(...)                  ⑧ 逐块写回（SSE：text/event-stream）
                    └──────────────┬──────────────┘
                                   ▼
            ⑨ response 中间件回程 ─► server 写回 socket ─► 客户端
```

任何一步抛异常 → 异常处理器兜底 → 返回结构化错误（4xx/5xx），不让连接裸崩。

---

## 3. 路由、listener、ctx 单例

### 3.1 用 listener 在每个 worker 启动时加载一次引擎

**致命错误是把模型加载写进路由函数**——那样每请求重载几十 GB 权重。正确做法：worker 进程启动时加载一次，挂到 `app.ctx`，全局复用。

Sanic 用 **listener**（生命周期钩子，如 `before_server_start` / `after_server_start` / `before_server_stop`）做这件事，相当于 FastAPI 的 `lifespan`。

```python
from sanic import Sanic, json

app = Sanic("llm-gateway")

@app.before_server_start
async def load(app, loop):
    app.ctx.engine = load_engine()   # 每个 worker 启动时各加载一次（重）

@app.before_server_stop
async def cleanup(app, loop):
    app.ctx.engine.shutdown()        # 退出时清理本 worker 的引擎

@app.post("/v1/chat/completions")
async def chat(request):
    data = request.json              # 取请求体（也可接 Pydantic 校验）
    engine = app.ctx.engine          # 复用本 worker 的引擎单例
    text = await engine.generate(data["messages"], data.get("max_tokens", 256))
    return json({"choices": [{"message": {"role": "assistant", "content": text}}]})
```

> 关键认知：`before_server_start` 在**每个 worker 进程**里都会各跑一遍。所以"加载一次"是指"每 worker 一次"，N 个 worker 就有 N 份权重——这正是为什么 worker 数受**显存**硬约束（见 §6、§7）。

### 3.2 入参校验

Sanic 本身不像 FastAPI 那样内建 Pydantic 自动校验，但可以：手动校验 `request.json`、用 `sanic-ext`（官方扩展，提供基于类型注解的校验/OpenAPI）、或自己接 Pydantic。**先校验再进业务**，把越界 `max_tokens`、缺字段挡在 GPU 之外，省算力也省 bug。具体 API 以官方文档为准。

---

## 4. 流式响应（让 token 边生成边吐）

### 4.1 为什么要流式

非流式：用户盯空白等 10 秒一次性收全文 → 体验差，**首字延迟 TTFT = 总生成时间**。
流式：模型每吐一个 token 就推给前端 → 几百毫秒就"打字"，TTFT 大幅下降。

```
非流式：  [........生成中(黑屏10s)........] ─► 全文
流式：    第50ms 首token ► token ► token ► ... ► [DONE]
          └ 用户立刻看到反馈，逐字出现
```

### 4.2 SSE 是什么

**Server-Sent Events**：基于普通 HTTP 的**单向、长连接**流。响应头 `Content-Type: text/event-stream`，body 不断追加文本块，每块形如：

```
data: {"delta":"你"}\n\n
data: {"delta":"好"}\n\n
data: [DONE]\n\n
```

每条以 `data: ` 开头、`\n\n` 结尾。比 WebSocket 轻（无需双向握手），天然适配"服务器单向推 token"。

### 4.3 Sanic 怎么实现：流式响应 + 异步生成

Sanic 提供流式响应能力（如 `sanic.response.ResponseStream` / 在 handler 里拿到可写的 response 对象后逐块 `await response.write(...)`，具体 API 名以官方文档为准）。核心思路：**从引擎 `async for` 拿 token，逐块写出 SSE 文本**，而不是攒满再返回。

```python
import json as _json
from sanic.response import ResponseStream

@app.post("/v1/chat/completions")
async def chat(request):
    data = request.json
    engine = app.ctx.engine
    if not data.get("stream"):
        text = await engine.generate(data["messages"], data.get("max_tokens", 256))
        return json({"choices": [{"message": {"role": "assistant", "content": text}}]})

    async def streaming(resp):
        async for tok in engine.stream(data["messages"], data.get("max_tokens", 256)):
            await resp.write(f"data: {_json.dumps({'delta': tok}, ensure_ascii=False)}\n\n")
        await resp.write("data: [DONE]\n\n")

    return ResponseStream(streaming, content_type="text/event-stream")
```

数据流：`引擎产 token(GPU) ─► async for(协程) ─► resp.write("data:...")(SSE) ─► server flush(写 socket) ─► 客户端逐块收(打字效果)`。

### 4.4 流式三个易错点

1. **客户端断线**：用户关页面后，要能停止生成、释放 GPU 名额，否则白烧算力。可在循环里探测连接是否还在，或捕获写入异常后停止。
2. **代理缓冲**：Nginx 等反代默认会缓冲响应，把"流"攒成"块"，打字效果消失。需关闭缓冲（如 `proxy_buffering off;`，以 Nginx 文档为准）。
3. **中文编码**：`json.dumps` 记得 `ensure_ascii=False`，避免一堆 `\uXXXX`。

---

## 5. 与推理引擎对接：进程内 vs 进程外

### 5.1 两种拓扑

```
A. 进程内（引擎和 Sanic worker 同进程）
   Sanic worker ──直接调用──► transformers / vLLM 引擎对象（同进程，共享 GPU）
   优点：简单、零网络跳；缺点：扩缩容与框架耦合，阻塞风险需小心

B. 进程外（Sanic 只做网关，引擎独立服务）
   Sanic ──HTTP/gRPC──► vLLM Server / TGI / Triton（独立进程或独立机器）
   优点：解耦、各自扩缩、引擎可换；缺点：多一跳网络
```

### 5.2 同步引擎怎么塞进异步世界

很多引擎的 `generate()` 是**同步阻塞**的。直接在 `async def` 里调它 = 卡死这个 worker 的 event loop（§1.2）。两条出路：

- **丢线程池**：`await loop.run_in_executor(None, engine.generate, ...)` / `await asyncio.to_thread(...)`，让阻塞调用在别的线程跑，event loop 不被卡。
- **用原生异步引擎**：vLLM 这类提供异步引擎 API（如 `AsyncLLMEngine`，名以官方为准），可直接 `async for` token，并在内部做**连续批处理（continuous batching）**，吞吐远高于"一个请求一个线程"。

```
同步 .generate() → run_in_executor（不卡 loop，但批处理弱）
异步 .stream()   → 直接 async for（连续批处理，吞吐高 ✓）
```

### 5.3 为什么生产更爱"进程外 + 异步引擎"

LLM 推理吞吐核心是**批处理**：把多个请求的 token 拼成一个 batch 喂 GPU。Sanic 自己不会批处理，但 vLLM/TGI 引擎内部有连续批处理调度器。让 Sanic 当**薄网关**、引擎专心批处理，是高吞吐部署的主流形态。详见 [[llm-inference/vllm/README]]。

---

## 6. 多进程 worker 与并发背压

### 6.1 worker 数怎么定（被 GPU 卡死）

普通 CPU-bound Web：`workers ≈ CPU 核数`。
**LLM 服务被 GPU 数量卡死**：一张卡通常只装一个引擎实例（显存装不下多份权重）。所以经验起点是**每张 GPU 一个 worker**：

```
4 卡机器 → --workers=4，每个 worker 用 CUDA_VISIBLE_DEVICES 绑一张卡
盲目加 worker → 多份权重抢同一张卡 → 显存 OOM
```

> 注意：每个 worker 是独立进程，`before_server_start` 各跑一遍 → 各自加载一份权重。`--workers=8` 在 4 卡机上若不绑卡，就是 8 份权重抢 4 张卡，必爆显存。

### 6.2 进程间不共享内存这件事

```
worker-0 内存            worker-1 内存            worker-2 内存
├ engine（权重副本0）    ├ engine（权重副本1）    ├ engine（权重副本2）
├ 全局计数器=10          ├ 全局计数器=3           ├ 全局计数器=7
└ 各自的 event loop      └ 各自的 event loop      └ 各自的 event loop
```

含义：进程内全局变量（缓存、计数器、限流状态）**不跨 worker 共享**。要全局一致（如全局限流、共享缓存），得靠外部存储（Redis 等）或 Sanic 的跨 worker 通信机制（如 worker 间共享上下文/消息，名以官方为准）。

### 6.3 背压：下游忙不过来就拒/排队

GPU 显存有限，同时在批里跑的请求数（KV-cache 容量）也有限。无脑收满 → `请求洪峰 → 全收 → OOM/延迟雪崩 → 全员超时`。

**背压 = 下游处理不过来时主动拒绝或排队上游请求。** 单 worker 内可用信号量做闸门：

```
进来 100 并发，每 worker MAX_CONCURRENCY=16：

[16 个在 GPU 上跑] ◄── asyncio.Semaphore 闸门 ──► [84 个排队 / 直接 429]
        │                                            │
   正常出结果                                 排队等名额 or fail fast（背压）
```

### 6.4 数值直觉

设单请求平均占 GPU `0.5s`、单 worker 并发上限 `N=16`，则该 worker 理论吞吐上限约

$$\text{QPS} \approx \frac{N}{\bar{t}} = \frac{16}{0.5} = 32 \text{ req/s}$$

4 个 worker（4 卡）合计 ~128 req/s。超过就要靠限流**快速拒绝**（fail fast），而不是排队到超时——**宁可明确拒绝，也不要全员雪崩**。

---

## 7. fork vs spawn 启动方式（本仓库实测踩坑点）

这是 Sanic 部署 LLM 时**最容易撞的坑**，也是本目录原始笔记记录的报错：

```
RuntimeError: Start method 'spawn' was requested, but 'fork' was already set.
```

### 7.1 多进程的两种"出生方式"

Python 多进程创建子进程有两种方法，Sanic 的 Worker Manager 拉起 worker 时也涉及它：

```
fork（Linux 默认，老方式）：
  父进程 ──克隆整块内存──► 子进程（继承一切：已 import 的库、已建的 CUDA 上下文、锁）
  优点：起得快、共享已加载内容
  致命缺点：CUDA / 某些 C 库在 fork 后状态损坏 → 子进程里 GPU 不可用 / 死锁

spawn（更干净的方式，部分平台默认）：
  父进程 ──只传必要参数──► 子进程从零 import、重新初始化
  优点：干净、CUDA 友好（每个 worker 自己初始化 GPU 上下文）
  缺点：起得慢、要求被传的对象可序列化（picklable）
```

### 7.2 为什么 LLM 场景常需要 spawn

如果在**主进程里就 import 了 CUDA / 初始化了 GPU**（比如导入 torch 并 `.cuda()`），再用 **fork** 拉 worker，子进程继承了一个"半死"的 CUDA 上下文 → GPU 报错或卡死。所以 LLM/torch 多进程**通常推荐 spawn**：让每个 worker 自己干净地初始化 GPU。

```
错误姿势：主进程已 import torch + 占了 CUDA ─fork─► worker 继承坏掉的 CUDA 上下文 ✗
正确姿势：主进程不碰 CUDA，worker 用 spawn 从零初始化 ─► 每 worker 干净拿到自己的 GPU ✓
```

### 7.3 那条报错到底在说什么

```
RuntimeError: Start method 'spawn' was requested, but 'fork' was already set.
```

含义：**多进程启动方法在进程里只能"定型"一次**。如果某处（某个库、某段代码）**已经把启动方式设成 fork** 了，你后面再要求 spawn 就冲突报错。

```
程序启动
   │
   ├─ 某库/某行代码偷偷 set_start_method('fork')  ← 提前定型为 fork
   │
   └─ Sanic / 你的代码再要求 spawn  ──► 冲突！RuntimeError
```

修复方向（具体 API 以官方文档为准，本文只给思路）：

1. **尽早、且只设置一次启动方式**：在程序最开头（任何重库 import / 任何进程创建之前）显式定为 `spawn`，避免被别处抢先设成 fork。
2. **用 Sanic 提供的入口约定**：把启动方式设置和 `app.run(...)` / `Sanic.serve(...)` 放进 `if __name__ == "__main__":` 保护块里——spawn 模式下子进程会重新 import 主模块，没有这层保护会无限递归创建进程。
3. **主进程不要提前初始化 CUDA**：把 torch/引擎的加载推迟到 worker 的 listener（`before_server_start`）里做，让每个 worker 自己初始化 GPU。
4. 参考官方说明：Sanic and start methods（https://sanic.dev/en/guide/running/manager.html#sanic-and-start-methods）。

### 7.4 一句话记牢

> **"CUDA + 多进程 = 用 spawn，且启动方式要在最早处只设一次，主进程别碰 GPU。"** fork 快但和 CUDA 不合；spawn 干净但要求 `__main__` 保护和可序列化。

---

## 8. 部署与可观测性

### 8.1 命令示例（含义为主，参数以官方文档为准）

```bash
# 开发：单进程，便于打断点/看日志（可加自动重载，生产关掉）
sanic server.app --host=0.0.0.0 --port=1337 --dev

# 生产：自带 server + Worker Manager，一条命令起多 worker
sanic server.app --host=0.0.0.0 --port=1337 --workers=4
#      └─ server.py 里的 app 对象     └─ worker 数 ≈ GPU 数（受显存硬约束）
```

要点：
- `--workers=N`：**N 受 GPU 数 / 单实例显存约束**，不是越大越好（§6.1）。
- 配合 `CUDA_VISIBLE_DEVICES` 给每个 worker 绑不同卡，避免抢显存。
- 开发用自动重载（监视文件 + 反复重载模型），**生产必须关**，否则反复重载几十 GB 权重。
- 自带 server 性能够用；若要 TLS / 复杂负载均衡，前面再挂 Nginx（记得关 `proxy_buffering` 保流式）。

### 8.2 全景部署图

```
客户端 ─► Nginx（TLS / 负载均衡 / 关闭 proxy_buffering 保流式）
            └─► Sanic 主进程 + Worker Manager（存活监控 + 重启，自带，无需 gunicorn）
                  ├─ worker-0 → Sanic app + 引擎实例 │GPU0
                  ├─ worker-1 → Sanic app + 引擎实例 │GPU1   worker 绑卡，
                  └─ worker-2 → Sanic app + 引擎实例 │GPU2   互不抢显存
                       （连续批处理在每个引擎实例内部完成）
```

### 8.3 生产稳定性清单

| 能力 | 怎么做 | 为什么 |
|---|---|---|
| 健康检查 | `/health`、`/ready` 路由 | 让 K8s/LB 知道该不该转流量；引擎没加载好别接客 |
| 请求取消 | 探测连接断开后停止生成 | 客户端跑了就别白烧 GPU |
| 超时控制 | 业务层 `asyncio.wait_for` / 框架超时配置 | 防单请求拖死名额 |
| 指标 | 暴露 TTFT、tokens/s、队列长度、并发数 | 流式体验命门 + 扩容/告警依据 |
| 优雅退出 | `before_server_stop` 清理 + Worker Manager 平滑停 | 重启时让在途请求收尾，不掉连接 |
| 结构化错误 | 统一异常处理器返回 4xx/5xx + JSON | 客户端可机读，不裸崩 |

---

## 典型流程/配置示例（讲含义，非照抄）

把前面拼成最小可用骨架的"心智模型"：

```
1. 程序最早处：设多进程启动方式为 spawn（只设一次），别在主进程碰 CUDA
2. before_server_start：load_engine() → app.ctx.engine（每 worker 各一份）
3. 校验请求体（手动 / sanic-ext / Pydantic）：卡死 max_tokens 边界
4. 路由 /v1/chat/completions：
     - 取 app.ctx.engine（本 worker 单例）
     - async with Semaphore(N) 做并发闸门（背压）
     - stream=True  → ResponseStream（SSE 逐 token resp.write）
     - stream=False → await engine.generate(...)（一次性 JSON）
5. 同步引擎调用 → run_in_executor 包一层，别卡 loop
6. 异常 → 统一处理器 → 结构化错误
7. /health 探针 + 指标暴露
8. 部署：sanic server.app --workers={GPU数}，每 worker 绑一张卡
```

各配置项的取舍：

| 配置 | 调大 | 调小 | 取舍 |
|---|---|---|---|
| `--workers` | 更多实例，但抢显存 | 省显存，吞吐低 | 受 **GPU 数 / 单实例显存** 硬约束 |
| `Semaphore(N)` | 更高并发，逼近 OOM | 更稳，吞吐降 | 按显存/KV-cache 容量定，配背压 |
| 请求超时 | 长生成不被误杀 | 早释放卡死请求 | 流式要放宽 |
| `max_tokens` 上限 | 允许长回答 | 省显存省时延 | 校验层就卡死，保护 GPU |
| 启动方式 | spawn（CUDA 友好） | fork（起得快但和 CUDA 冲突） | LLM 场景选 spawn |

---

## 常见问题

| 问题 | 原因 | 处置 |
|---|---|---|
| `Start method 'spawn' requested, but 'fork' was already set` | 启动方式被别处提前设成 fork，后面又要 spawn | 在程序最早处只设一次 spawn，放进 `__main__` 保护块（§7） |
| 多 worker 下 GPU 报错/卡死 | 主进程提前初始化了 CUDA，再 fork 出 worker | 用 spawn；CUDA/引擎加载推迟到 `before_server_start` |
| 加了 `async` 还是卡 | 协程里调了同步阻塞（同步 requests / `.generate()` / `time.sleep`） | 丢 `run_in_executor` 或用异步引擎 |
| 每次请求都重载模型 | 把加载写进路由函数 | listener `before_server_start` 加载一次 + `app.ctx` 复用 |
| `--workers` 调大就 OOM | 每 worker 各一份权重，超过 GPU 承载 | worker 数 ≈ GPU 数，`CUDA_VISIBLE_DEVICES` 绑卡 |
| 全局计数器/缓存各 worker 不一致 | 进程间内存不共享 | 用 Redis 等外部存储，或跨 worker 通信机制 |
| 流式没有"打字"效果 | 反代缓冲（Nginx `proxy_buffering on`）攒块 | 关闭缓冲；确认 `text/event-stream` |
| 中文输出成 `\uXXXX` | `json.dumps` 默认转义 | 加 `ensure_ascii=False` |
| spawn 下进程无限递归创建 | 启动代码没放 `if __name__ == "__main__":` | spawn 会重 import 主模块，必须加保护块 |
| 客户端关页面 GPU 还在烧 | 没做断线检测/取消 | 探测连接断开，及时停生成 |
| 这框架能直接提吞吐吗 | Sanic 不做批处理 | 吞吐靠**引擎**的连续批处理，Sanic 只做薄网关 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，定位本文在推理服务化中的位置
- [[llm-inference/README]] — 推理总览：引擎选型（vLLM/TGI/Triton）、连续批处理、KV-cache，与本文"前台 + 引擎"分工互补
- [[llm-inference/vllm/README]] — vLLM 异步引擎 + 连续批处理，Sanic 做薄网关时的首选后端
- [[llmops/kubernetes]] — 多 worker/多副本上 K8s：健康探针、滚动更新、GPU 调度与本文部署衔接

> 参考：Sanic 官方文档 https://sanic.dev ；多进程启动方式说明 https://sanic.dev/en/guide/running/manager.html#sanic-and-start-methods （精确 API/参数以官方为准）。
