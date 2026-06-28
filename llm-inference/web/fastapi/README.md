# FastAPI 部署 LLM
> 用异步 Web 框架把"推理引擎"包成高并发、可流式、可观测的 HTTP/SSE 服务。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/openai]] [[llm-inference/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|---|---|---|
| 0 | 一句话锚点 | FastAPI = ASGI + 类型校验 + 异步 I/O |
| 1 | 它解决什么问题 | 同步阻塞 vs 异步并发；为什么 LLM 服务必须异步 |
| 2 | 整体架构与请求生命周期 | ASGI / event loop / uvicorn worker |
| 3 | 路由、Pydantic、依赖注入 | 入参校验、模型加载单例 |
| 4 | 流式响应 SSE | `StreamingResponse` + 异步生成器 |
| 5 | 与推理引擎对接 | 进程内 vs 进程外（vLLM/TGI/Triton）|
| 6 | 并发与背压 | 信号量、队列、连续批处理、限流 |
| 7 | 部署：uvicorn / gunicorn | worker 数、GPU 亲和、健康检查 |
| 8 | 可观测性与稳定性 | 超时、取消、指标、优雅退出 |
| — | 配置示例（讲含义）| uvicorn / gunicorn 命令 |
| — | 常见问题 | 表格速查 |

> 说明：本文讲**机制与思路**。涉及精确版本号、CLI 默认值、具体 API 签名处，一律以**官方文档/源码为准**，本文不编造。

---

## 0. 一句话锚点

**FastAPI 是一个基于 ASGI 标准、用 Python 类型注解做请求校验、用 `async/await` 做高并发 I/O 的 Web 框架。** 在 LLM 部署里，它扮演"前台"角色：接住 HTTP 请求 → 校验参数 → 把生成任务交给后端**推理引擎** → 把 token 流式吐回客户端。它本身**不做 GPU 推理**，真正算 logits 的是 vLLM / TGI / Triton / transformers 这些引擎。

记住一条主线：

```
客户端 ──HTTP/SSE──► FastAPI（ASGI 前台，CPU 轻活）──►推理引擎（GPU 重活）
                         异步、不阻塞                     连续批处理
```

---

## 1. 地基：它解决什么问题

### 1.1 同步阻塞为什么害死 LLM 服务

LLM 生成一个回答动辄要 **几百毫秒到几十秒**。这段时间里，服务端几乎不消耗 CPU——它在**等 GPU 算完**。这是典型的 **I/O 等待（I/O-bound）**，不是 CPU 算不过来。

如果用传统**同步**模型（一个线程处理一个请求，等待时线程被占死）：

```
同步模型（4 个工作线程，每个生成耗时 5s）：

线程1: [req A 等GPU 5s............] [req E ...]
线程2: [req B 等GPU 5s............] [req F ...]
线程3: [req C 等GPU 5s............]
线程4: [req D 等GPU 5s............]
      └─ 第 5 个请求必须排队，因为 4 个线程全被"等待"占满
```

线程在 `await` GPU 结果期间**什么也没干，却占着坑**。要扛 100 并发就要 100 线程，线程切换 + 内存开销爆炸。

### 1.2 异步并发：等待时把控制权让出去

`async def` + `await` 的本质是 **协作式调度**：当一个请求 `await` 在等 GPU / 等网络写回时，event loop 把控制权交给**别的请求**，单个线程就能"同时"推进成千上万个等待中的请求。

```
异步模型（单线程 event loop）：

时刻 t0: req A 提交任务 → await（让出）
时刻 t0: req B 提交任务 → await（让出）
时刻 t0: req C 提交任务 → await（让出）
       ... event loop 在多个 await 点之间穿梭，谁好了先处理谁
时刻 t5: req A 的 token 来了 → 唤醒 A 的协程 → 写回客户端
```

> 关键纪律：**异步函数里绝不能放同步阻塞调用**（如同步 `requests.post`、`time.sleep`、CPU 密集的纯 Python 循环、阻塞式模型 `.generate()`）。一旦阻塞，整个 event loop 卡死，所有并发请求一起冻结。阻塞活儿要丢进**线程池**（`run_in_threadpool` / `asyncio.to_thread`）或**独立进程**。

### 1.3 为什么是 FastAPI 而不是 Flask

| 维度 | Flask（WSGI，同步） | FastAPI（ASGI，异步） |
|---|---|---|
| 并发模型 | 线程/进程，等待即占坑 | event loop，等待即让出 |
| I/O-bound 高并发 | 差，靠堆线程 | 强，单线程扛海量等待 |
| 流式 SSE | 可做但别扭 | 原生 `StreamingResponse` |
| 入参校验 | 手写 | Pydantic 自动校验 + 文档 |
| 自动 API 文档 | 需插件 | 内置 OpenAPI / Swagger |
| 类型提示 | 弱 | 强（IDE 友好）|

LLM 服务 = 高并发 + 长连接 + 流式输出，**正好踩中 FastAPI 的强项**。

---

## 2. 整体架构与请求生命周期

### 2.1 三层栈：进程管理器 / ASGI 服务器 / 应用

```
┌──────────────────────────────────────────────┐
│ gunicorn（进程管理器，可选）                    │  管多个 worker、重启、信号
│  ├── worker-0 ┐                                │
│  ├── worker-1 ┤ 每个 worker = 一个 uvicorn      │  ASGI 服务器：协议解析、event loop
│  └── worker-2 ┘   ├── FastAPI app              │  你的应用：路由、校验、业务
│                   │    ├── 路由表 /v1/...        │
│                   │    ├── 中间件（日志/限流）    │
│                   │    └── 推理引擎句柄（单例）   │
└──────────────────────────────────────────────┘
                    │
                    ▼
            GPU / 推理引擎
```

- **uvicorn**：ASGI 服务器，负责把 HTTP/字节流解析成 ASGI 事件，跑 event loop，调用你的 app。
- **gunicorn**：进程主管，把 uvicorn 当 worker 拉起多份，做存活监控与平滑重启（生产常用 `gunicorn -k uvicorn.workers.UvicornWorker`，具体类名以官方文档为准）。
- **FastAPI app**：你的代码——路由、Pydantic 模型、依赖注入、与引擎对接。

### 2.2 一条请求的生命周期

```
① TCP/HTTP 入 ─► ② uvicorn 解析为 ASGI 事件 ─► ③ 中间件链（CORS/日志/限流/鉴权）
   ─► ④ 路由匹配 ─► ⑤ Pydantic 校验请求体（错就 422，根本不进业务）
   ─► ⑥ 依赖注入（引擎单例 / 限流信号量 / user 上下文）
   ─► ⑦ 业务函数：把 prompt + 采样参数 提交给推理引擎
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
        非流式：await 完整结果            流式：async for token: yield
                    ▼                             ▼
        ⑧ 序列化 JSONResponse         ⑧ StreamingResponse（SSE 逐块写）
                    └──────────────┬──────────────┘
                                   ▼
            ⑨ 经中间件回程 ─► uvicorn 写回 socket ─► 客户端
```

任何一步抛异常 → 异常处理器兜底 → 返回结构化错误（4xx/5xx），不让连接裸崩。

---

## 3. 路由、Pydantic 校验、依赖注入

### 3.1 用 Pydantic 把"协议契约"写成类型

请求体先过 Pydantic：字段缺失、类型不对、越界，**在进入业务前就被拦下返回 422**。这把"垃圾输入"挡在 GPU 之外，省算力也省 bug。

```python
from typing import Optional, List
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    max_tokens: int = Field(256, ge=1, le=4096)      # 越界即 422
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = False                              # 是否走 SSE
```

> 想做成 OpenAI 兼容接口，就把字段对齐 `/v1/chat/completions` 的 schema，详见 [[llm-inference/openai]]。

### 3.2 模型/引擎只加载一次：lifespan + 依赖注入

**致命错误是把模型加载写进路由函数**——那样每请求都重载几十 GB 权重。正确做法：进程启动时加载一次，全局复用。

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = load_engine()   # 启动时加载一次（重）
    yield
    app.state.engine.shutdown()        # 退出时清理

app = FastAPI(lifespan=lifespan)

def get_engine():                      # 依赖：每请求注入同一句柄
    return app.state.engine
```

流向：`进程启动 ─► 加载权重到 GPU（一次，慢）─► app.state.engine`；之后 `请求1/2/N ─► Depends(get_engine) ─► 复用同一引擎（快）`。依赖注入的另一高价值用途：把**限流信号量**、**鉴权**、**请求级超时**做成可复用依赖，路由函数保持干净。

---

## 4. 流式响应 SSE（让 token 边生成边吐）

### 4.1 为什么要流式

非流式：用户盯着空白等 10 秒，一次性收到全文 → 体验差，且**首字延迟（TTFT）= 总生成时间**。
流式：模型每吐一个 token 就推给前端 → 用户几百毫秒就看到字在"打字"，TTFT 大幅下降，感知速度天差地别。

```
非流式：  [........生成中(黑屏10s)........] ─► 全文
流式：    第50ms 首token ► token ► token ► ... ► [DONE]
          └ 用户立刻看到反馈，逐字出现
```

### 4.2 SSE 是什么

**Server-Sent Events**：基于普通 HTTP 的**单向、长连接**流。响应头 `Content-Type: text/event-stream`，body 是不断追加的文本块，每块形如：

```
data: {"delta":"你"}\n\n
data: {"delta":"好"}\n\n
data: [DONE]\n\n
```

每条以 `data: ` 开头、`\n\n` 结尾。比 WebSocket 轻（无需双向握手），天然适配"服务器单向推 token"。

### 4.3 FastAPI 怎么实现：异步生成器 + StreamingResponse

核心：写一个 `async def` 生成器，从引擎 `async for` 拿 token，`yield` SSE 文本块；交给 `StreamingResponse`。

```python
import json
from fastapi.responses import StreamingResponse

async def sse_gen(req: ChatRequest, engine):
    async for tok in engine.stream(req.messages, req.max_tokens, req.temperature):
        yield f"data: {json.dumps({'delta': tok}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, engine = Depends(get_engine)):
    if req.stream:
        return StreamingResponse(sse_gen(req, engine),
                                 media_type="text/event-stream")
    text = await engine.generate(req.messages, req.max_tokens, req.temperature)
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}
```

数据流：`引擎产 token(GPU) ─► async for(协程) ─► yield "data:..."(SSE) ─► uvicorn flush(写 socket) ─► 客户端逐块收(打字效果)`。

### 4.4 流式三个易错点

1. **客户端断线检测**：用户关页面后，要能停止生成、释放 GPU 名额，否则白烧算力。可用 `request.is_disconnected()` 在循环里探测，或捕获取消异常。
2. **代理缓冲**：Nginx 等反代默认会缓冲响应，把"流"攒成"块"，打字效果消失。需关闭缓冲（如 `proxy_buffering off;`，具体以 Nginx 文档为准）。
3. **编码**：中文 `json.dumps` 记得 `ensure_ascii=False`，避免一堆 `\uXXXX`。

---

## 5. 与推理引擎对接：进程内 vs 进程外

### 5.1 两种拓扑

```
A. 进程内（引擎和 FastAPI 同进程）
   FastAPI app ──直接调用──► transformers / vLLM 引擎对象（同一进程，共享 GPU）
   优点：简单、零网络跳；缺点：扩缩容耦合，GIL/阻塞风险需小心

B. 进程外（FastAPI 只做网关，引擎独立服务）
   FastAPI ──HTTP/gRPC──► vLLM Server / TGI / Triton（独立进程或独立机器）
   优点：解耦、各自扩缩、引擎可换；缺点：多一跳网络
```

### 5.2 同步引擎怎么塞进异步世界

很多引擎的 `generate()` 是**同步阻塞**的。直接在 `async def` 里调它 = 卡死 event loop（见 §1.2）。两条出路：

- **丢线程池**：`await run_in_threadpool(engine.generate, ...)` / `await asyncio.to_thread(...)`，让阻塞调用在别的线程跑，event loop 不被卡。
- **用原生异步引擎**：vLLM 这类提供 `AsyncLLMEngine`（异步 API，具体名以官方为准），可直接 `async for` token，并在内部做**连续批处理（continuous batching）**，吞吐远高于"一个请求一个线程"。对比：`同步.generate() → run_in_threadpool（不卡 loop 但批处理弱）` vs `异步.stream() → 直接 async for（连续批处理，吞吐高 ✓）`。

### 5.3 为什么生产更爱"进程外 + 异步引擎"

LLM 推理的吞吐核心是 **批处理**：把多个请求的 token 拼成一个 batch 喂 GPU。FastAPI 自己不会批处理，但 vLLM/TGI 这类引擎内部有连续批处理调度器。让 FastAPI 当**薄网关**、引擎专心批处理，是高吞吐部署的主流形态。要做 OpenAI 兼容网关时，FastAPI 转发到引擎即可，协议细节见 [[llm-inference/openai]]。

---

## 6. 并发与背压

### 6.1 没有背压会发生什么

GPU 显存有限，能同时在批里跑的请求数（KV-cache 容量）也有限。如果来多少收多少、无脑塞给 GPU，就会：`请求洪峰 ──► 全部接收 ──► 显存 OOM / 延迟雪崩 ──► 全员超时`。

**背压（backpressure）= 当下游处理不过来时，主动拒绝或排队上游请求**，保护系统不崩。

### 6.2 三道闸门

```
① 限流/信号量：限制"同时在推理"的请求数
   sem = asyncio.Semaphore(MAX_CONCURRENCY)
   async with sem:        # 超过 N 的请求在此等待，不冲进 GPU
       await engine.generate(...)

② 队列 + 超时：等不到名额就快速失败（返回 429/503），别让客户端干等

③ 引擎层批调度：vLLM 内部按显存动态决定 batch 大小（最强一道）
```

```
进来 100 并发，MAX_CONCURRENCY=16：

[16 个在 GPU 上跑] ◄── Semaphore 闸门 ──► [84 个排队/被拒]
        │                                      │
   正常出结果                          排队等名额 or 直接 429（背压）
```

### 6.3 数值直觉

设单请求平均占用 GPU `0.5s`、并发上限 `N=16`，则理论吞吐上限约

$$\text{QPS} \approx \frac{N}{\bar{t}} = \frac{16}{0.5} = 32 \text{ req/s}$$

超过 32 req/s 持续涌入，队列就会无限增长 → 必须靠限流把超额请求**快速拒绝**（fail fast），而不是让它们排队到超时。这就是背压存在的意义：**宁可明确拒绝，也不要全员雪崩**。

---

## 7. 部署：uvicorn / gunicorn 思路

### 7.1 worker 数怎么定（和普通 Web 服务不一样）

普通 CPU-bound Web：`workers ≈ CPU 核数`。
**LLM 服务被 GPU 数量卡死**：一张卡通常只服务一个引擎实例（显存装不下多份权重）。所以经验起点是**每张 GPU 一个 worker**：`4 卡机器 → 4 个 worker，各绑一张卡（CUDA_VISIBLE_DEVICES）`；盲目加 worker → 多份权重抢同一张卡 → 显存 OOM。

### 7.2 单进程 vs 多进程

- **开发/调试**：`uvicorn` 单进程 + `--reload` 热重载（别上生产）。
- **生产**：`gunicorn` 管多个 uvicorn worker——worker 崩了自动拉起（存活监控）、平滑重启（先起新 worker 再停旧的，不断流）、每 worker 绑定独立 GPU。

### 7.3 命令示例（含义为主，参数以官方文档为准）

```bash
# 开发：单进程 + 热重载（生产禁用 --reload）
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 生产：gunicorn 拉多个 uvicorn worker
gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \   # 用 uvicorn 作为 worker 类型
  -w 4 \                               # worker 数 ≈ GPU 数（按显存定）
  -b 0.0.0.0:8000 \                    # 绑定地址
  --timeout 120 \                      # 长生成需调大，避免被误杀
  --graceful-timeout 30                # 优雅退出窗口，等在途请求收尾
```

要点：`--reload` 只在开发用（监视文件 + 反复重载模型）；长流式生成时 `--timeout` 太小会被当"卡死"杀掉，需放宽；每 worker 用 `CUDA_VISIBLE_DEVICES` 绑定不同 GPU，避免抢卡。

### 7.4 全景部署图

```
客户端 ─► Nginx（TLS / 负载均衡 / 关闭 proxy_buffering 保流式）
            └─► gunicorn（进程主管：存活监控 + 平滑重启）
                  ├─ uvicorn worker → FastAPI app + 引擎实例 │GPU0
                  ├─ uvicorn worker → FastAPI app + 引擎实例 │GPU1   worker 绑卡，
                  └─ uvicorn worker → FastAPI app + 引擎实例 │GPU2   互不抢显存
                       （连续批处理在每个引擎实例内部完成）
```

---

## 8. 可观测性与稳定性（生产分水岭）

| 能力 | 怎么做 | 为什么 |
|---|---|---|
| 健康检查 | `/health`、`/ready` 探针 | 让 K8s/负载均衡知道该不该转流量；引擎没加载好别接客 |
| 请求取消 | 探测 `is_disconnected()`，停止生成 | 客户端跑了就别白烧 GPU |
| 超时控制 | 业务层 `asyncio.wait_for` | 防单请求拖死名额 |
| 指标 | 暴露 TTFT、tokens/s、队列长度、并发数 | 既是流式体验命门，也是扩容与告警依据 |
| 优雅退出 | lifespan 清理 + graceful-timeout | 重启时让在途请求收尾，不掉连接 |
| 结构化错误 | 统一异常处理器返回 4xx/5xx + JSON | 客户端可机读，不裸崩 |

---

## 典型流程/配置示例（讲含义，非照抄）

把前面拼成一条最小可用骨架的"心智模型"：

```
1. lifespan 启动：load_engine() 一次 → app.state.engine
2. 定义 ChatRequest（Pydantic 校验 max_tokens/temperature 边界）
3. 路由 /v1/chat/completions：
     - Depends(get_engine) 拿引擎单例
     - async with Semaphore(N) 做并发闸门（背压）
     - stream=True  → StreamingResponse(sse_gen)（SSE 逐 token）
     - stream=False → await engine.generate(...)（一次性 JSON）
4. 同步引擎调用 → run_in_threadpool 包一层，别卡 loop
5. 异常 → 统一处理器 → 结构化错误
6. /health 探针 + 指标暴露
7. 部署：gunicorn -k UvicornWorker -w {GPU数}，每 worker 绑一张卡
```

各配置项的取舍：

| 配置 | 调大 | 调小 | 取舍 |
|---|---|---|---|
| `-w`（worker 数） | 更多实例，但抢显存 | 省显存，吞吐低 | 受 **GPU 数 / 单实例显存** 硬约束 |
| `Semaphore(N)` | 更高并发，逼近 OOM | 更稳，吞吐降 | 按显存/KV-cache 容量定，配合背压 |
| `--timeout` | 长生成不被误杀 | 早释放卡死请求 | 流式要放宽 |
| `max_tokens` 上限 | 允许长回答 | 省显存省时延 | 校验层就卡死，保护 GPU |

---

## 常见问题

| 问题 | 原因 | 处置 |
|---|---|---|
| 加了 `async` 却还是慢/卡 | 在协程里调了**同步阻塞**调用（同步 requests / `.generate()` / `time.sleep`） | 丢 `run_in_threadpool` 或用异步引擎 |
| 每次请求都重载模型 | 把加载写进路由函数 | lifespan 启动加载一次 + 依赖注入复用 |
| 流式没有"打字"效果 | 反代缓冲（Nginx `proxy_buffering on`）攒块 | 关闭缓冲；确认 `text/event-stream` |
| 中文输出成 `\uXXXX` | `json.dumps` 默认转义 | 加 `ensure_ascii=False` |
| 高并发直接 OOM/雪崩 | 没有背压，来多少塞多少 | Semaphore/队列限流 + 引擎批调度 + 429 fail fast |
| 多 worker 显存爆 | worker 数超过 GPU 承载，多份权重抢卡 | worker 数 ≈ GPU 数，`CUDA_VISIBLE_DEVICES` 绑卡 |
| 客户端关页面 GPU 还在烧 | 没做断线检测/取消 | `is_disconnected()` 探测，及时停生成 |
| 长生成被进程管理器杀 | `--timeout` 太小被当卡死 | 调大 timeout，对流式放宽 |
| 重启掉连接 | 没有优雅退出 | lifespan 清理 + `--graceful-timeout` |
| 这框架能直接提吞吐吗 | FastAPI 不做批处理 | 吞吐靠**引擎**的连续批处理，FastAPI 只做薄网关 |

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，定位本文在推理服务化中的位置
- [[llm-inference/openai]] — OpenAI 兼容接口规范，FastAPI 做兼容网关时对齐其 schema
- [[llm-inference/README]] — 推理总览：引擎选型（vLLM/TGI/Triton）、连续批处理、KV-cache，与本文"前台 + 引擎"分工互补
