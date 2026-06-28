# OpenAI 兼容接口

> 一句话定位：OpenAI 的 HTTP API（尤其 `/v1/chat/completions`）已成为大模型推理服务的"事实标准插座"，几乎所有开源推理引擎（vLLM、SGLang、TGI、llama.cpp、Ollama…）都实现了与它"形状一致"的接口，从而让上层应用一次接入、随处可换。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/chatgpt]] [[llm-maas/OpenAI-ChatGPT]] [[llm-inference/vllm/README]]

## 阅读地图

| 节 | 你会得到什么 | 关键词 |
|----|------------|--------|
| 0 | 一句话锚点：什么叫"OpenAI 兼容" | 协议 ≠ 模型 |
| 1 | 地基：为什么需要一个"统一接口"，它解决了谁的痛 | 标准化、解耦 |
| 2 | API 形状：endpoint、请求体、响应体逐字段拆解 | `/v1/chat/completions` |
| 3 | 三大核心端点：chat / completions / embeddings 的差异 | messages vs prompt |
| 4 | SSE 流式：为什么用 SSE、`data:` 帧逐字节长啥样 | `stream=true`、chunk |
| 5 | 为什么各推理引擎都来兼容它（生态飞轮） | 网络效应、迁移成本 |
| 6 | 客户端生态：官方 SDK、LangChain、各类 UI | `base_url` 一改即换后端 |
| 7 | 调用全生命周期：从 SDK 到 token 流的完整链路 | 请求生命周期 |
| 8 | 函数调用 / tools / 结构化输出 | tool_calls、JSON mode |
| 9 | 鉴权、错误、重试、超时、ConnectionError | 401/429/5xx |
| 流程 | 一个可运行的最小示例（讲含义，非照抄） | Python / curl |
| FAQ | 高频坑与对照表 | 兼容性差异 |

## 0. 一句话锚点

**"OpenAI 兼容"= 你的推理服务说的是 OpenAI 定义的那套 HTTP 方言。**

它和"用 OpenAI 的模型"是两件事，务必分清：

```
        协议(API 形状)              ≠              模型(权重)
   ┌──────────────────────┐              ┌──────────────────────┐
   │ POST /v1/chat/...    │              │ GPT-4 / Qwen / Llama │
   │ {"model","messages"} │   解耦       │ DeepSeek / GLM ...   │
   │ 返回 {choices,usage} │ ───────────▶ │ 谁在后面跑无所谓      │
   └──────────────────────┘              └──────────────────────┘
   只要"插座"一样，灯泡(模型/引擎)可以随便换
```

只要后端实现了这套"插座"，前端代码（哪怕是 OpenAI 官方 SDK）只改一个 `base_url` 就能从"调 OpenAI 云"切到"调本地 vLLM"，模型、引擎、机房全换掉而业务代码一行不动。这就是它的全部价值所在。

## 1. 地基：它解决什么问题

### 1.1 没有标准之前的世界

想象 2022 年之前：每家推理框架自定义 HTTP 接口。A 框架要你 POST `{"text": ...}`，B 框架要 `{"inputs": ...}`，C 框架返回 `{"generated": ...}`，D 框架返回 `{"output": [...]}`。结果：

```
应用层 ──┬──▶ 框架A 适配器A
         ├──▶ 框架B 适配器B      每换一个后端，
         ├──▶ 框架C 适配器C      上层就要重写一次胶水代码
         └──▶ 框架D 适配器D      → N×M 适配地狱
```

每个"应用 × 后端"组合都要写一份适配代码，复杂度是 $O(N\times M)$（N 个应用、M 个后端）。

### 1.2 标准化把 N×M 降成 N+M

OpenAI 因为 ChatGPT（见 [[llm-inference/chatgpt]]）的爆发占据了开发者心智，它的 API 文档成了大家"第一份学会的 LLM API"。于是社区做了一个朴素而强力的决定：**与其各自发明，不如都长成 OpenAI 那个样子**。

```
应用层(都用 OpenAI SDK) ──▶ ┌─────────────────┐
                            │  统一接口契约     │ ──▶ vLLM
                            │ /v1/chat/comp.   │ ──▶ SGLang
                            │ {model,messages} │ ──▶ TGI
                            └─────────────────┘ ──▶ Ollama / llama.cpp
   复杂度从 O(N×M) 降到 O(N+M)：应用学一次，后端各实现一次
```

这就是"事实标准（de facto standard）"：没有任何标准委员会投票，纯粹靠生态自发收敛。**它解决的核心问题是：把"模型供给"和"应用需求"解耦，让两边可以独立演化、自由组合。**

### 1.3 三个受益方

| 受益方 | 得到了什么 |
|--------|-----------|
| 应用开发者 | 一套代码，随处换后端；本地调试用 Ollama，上线换 vLLM 集群 |
| 推理引擎作者 | 不必教育用户学新 API，"兼容 OpenAI"四个字即可获客 |
| 平台/网关 | 可以做统一路由、计费、限流，对上游屏蔽后端差异（见 [[llm-maas/OpenAI-ChatGPT]]） |

## 2. API 形状：逐字段拆解

最核心、被兼容得最彻底的端点是 `POST /v1/chat/completions`。先看它的"骨架"：

```
POST https://<host>/v1/chat/completions
Authorization: Bearer <API_KEY>      ← 鉴权，放在 HTTP 头
Content-Type: application/json
                                       请求体(JSON)
{
  "model":    "qwen2.5-7b-instruct",  ← 选哪个模型(后端自己解析)
  "messages": [                        ← 对话历史，按角色分条
    {"role": "system",    "content": "你是助手"},
    {"role": "user",      "content": "你好"},
    {"role": "assistant", "content": "你好，请问..."},
    {"role": "user",      "content": "讲个笑话"}
  ],
  "temperature": 0.7,    ← 采样温度，越高越随机
  "top_p":       0.9,    ← 核采样，与 temperature 二选一为主
  "max_tokens":  512,    ← 生成上限(部分实现叫 max_completion_tokens)
  "stream":      false,  ← 是否流式返回
  "stop":        ["\n\n"]← 命中即停的字符串
}
```

**为什么是 `messages` 而不是一个大字符串？** 因为对话有"角色"结构：`system` 设定人设、`user` 是用户、`assistant` 是模型既往回答。引擎拿到结构化 `messages` 后，会用该模型的 **chat template**（对话模板）把它们拼成模型真正吃的那一长串 token：

```
messages(结构化)            chat template(每个模型不同)         真正喂给模型的 token 串
[system, user, ...]  ──────────────────────────────▶  <|im_start|>system\n你是助手<|im_end|>
                          引擎内部按模型规则拼接         <|im_start|>user\n你好<|im_end|>
                                                       <|im_start|>assistant\n
```

这一步"模板拼接"被藏在服务端，正是兼容接口的精妙处：**同样的 `messages`，换不同模型，引擎用各自模板渲染——上层完全无感**。

### 2.1 响应体（非流式）

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1718900000,
  "model": "qwen2.5-7b-instruct",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "为什么..."},
    "finish_reason": "stop"      ← 停止原因: stop/length/tool_calls
  }],
  "usage": {
    "prompt_tokens": 42,          ← 输入消耗
    "completion_tokens": 88,      ← 输出消耗(计费/限流核心)
    "total_tokens": 130
  }
}
```

逐字段要点：

| 字段 | 含义 | 为什么重要 |
|------|------|-----------|
| `choices[].message` | 模型这一轮的回答 | 取 `choices[0].message.content` 即答案 |
| `finish_reason` | 为何停止 | `length`=被 `max_tokens` 截断；`stop`=正常；`tool_calls`=要调工具 |
| `usage` | token 计数 | 计费、限流、成本核算全靠它 |
| `id` | 请求唯一标识 | 排障、日志关联 |

## 3. 三大核心端点

| 端点 | 输入形态 | 适用 | 备注 |
|------|---------|------|------|
| `/v1/chat/completions` | `messages`（多轮，带角色） | 对话、Agent、绝大多数场景 | **当今主力**，兼容度最高 |
| `/v1/completions` | `prompt`（裸文本续写） | 续写、补全、base 模型 | 早期接口，仍被保留 |
| `/v1/embeddings` | `input`（文本/数组） | RAG、检索、聚类 | 返回向量数组，配合向量库 |

```
chat/completions:   [{role,content},...] ─▶ 引擎套模板 ─▶ 续写 ─▶ {message}
completions:        "一段裸 prompt 文本"   ─▶ 直接续写       ─▶ {text}
embeddings:         "要编码的文本"         ─▶ 取隐层向量     ─▶ {embedding:[...]}
```

**为什么 chat 端点赢了？** 因为 instruct/chat 模型已成主流，且"角色化多轮"天然贴合 Agent、function calling、system prompt 等需求；`/v1/completions` 更适合 base 模型续写，逐渐退居二线。其它端点如 `/v1/models`（列出可用模型）、`/v1/audio`、`/v1/images` 等，开源引擎按需实现，**chat/completions + embeddings 是兼容性的"地基三件套"**。

## 4. SSE 流式：逐帧拆解

### 4.1 为什么要流式

非流式：用户等模型生成完 500 个 token 才看到第一个字，首字延迟（TTFT 之外的"整段等待"）可能好几秒，体验差。流式：模型每吐一个 token 就推一次，像打字机一样逐字出现。

```
非流式:  请求 ──────────[等全部生成]──────────▶ 一次性返回整段   (体验:卡)
流式:    请求 ─▶ 字 ─▶ 字 ─▶ 字 ─▶ ... ─▶ [DONE]              (体验:顺滑打字机)
```

### 4.2 为什么用 SSE 而不是 WebSocket

服务端 → 客户端的**单向**token 推送，用 **SSE（Server-Sent Events）** 足矣：它就是一个 `Content-Type: text/event-stream` 的长连 HTTP 响应，按 `data:` 前缀分帧。相比 WebSocket，SSE 更轻、走标准 HTTP、天然适配现有网关/代理，且无需双向通道。

请求只需加一个字段：`"stream": true`。响应变成一串"事件帧"：

```
HTTP/1.1 200 OK
Content-Type: text/event-stream      ← 关键头，告诉客户端这是流

data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"为"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"什"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"么"},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]                          ← 哨兵：流结束，客户端据此收尾
```

### 4.3 delta 拼接机制

流式里每帧不是完整 `message` 而是增量 `delta`。客户端把所有 `delta.content` **按序拼接**就得到完整答案：

```
delta:"为" + delta:"什" + delta:"么" + ... ─▶ 客户端累加 ─▶ "为什么..."
帧之间用空行分隔；遇到 data:[DONE] 即停止读取并关闭连接
```

要点：
- 每个 SSE 帧以 `data: ` 开头，以**空行**结尾；
- 第一帧通常带 `delta.role`，后续帧只带 `delta.content`；
- 最后一帧 `finish_reason` 非空，紧接着一行 `data: [DONE]` 作为哨兵；
- 流式下默认**不返回 usage**，部分实现支持 `stream_options:{"include_usage":true}` 在末帧附带 token 统计。

## 5. 为什么各推理引擎都来兼容（生态飞轮）

这是一个自我强化的**网络效应**循环：

```
        更多应用按 OpenAI API 写
                 │
                 ▼
   引擎"兼容 OpenAI"=零迁移成本接入
                 │
        ┌────────┴────────┐
        ▼                 ▼
  引擎为获客              应用因后端多
  竞相兼容                更愿意按此 API 写
        │                 │
        └────────┬────────┘
                 ▼
        标准越强 → 越多人用 → 标准更强(飞轮)
```

| 引擎 | OpenAI 兼容入口 | 一句话 |
|------|----------------|--------|
| vLLM | `vllm serve` 起 OpenAI-compatible server | 高吞吐，PagedAttention，详见 [[llm-inference/vllm/README]] |
| SGLang | OpenAI 兼容 server | RadixAttention，复杂控制流强 |
| TGI | Messages API 提供兼容层 | HuggingFace 出品 |
| Ollama | `/v1/chat/completions` 兼容端点 | 本地一键跑，开发者最爱 |
| llama.cpp | `llama-server` 暴露 OpenAI 接口 | 纯 C/C++，CPU/边缘友好 |

对引擎作者而言，**"兼容 OpenAI"几乎是免费的获客承诺**：用户已有的 OpenAI SDK 代码、LangChain 应用、各种 UI，改个 `base_url` 就能直接用你的引擎。不兼容＝主动把生态拒之门外。于是兼容成了"入场券"而非"加分项"。

## 6. 客户端生态

兼容接口最大的红利在客户端侧——**一个 `base_url` 切换整个后端**：

```python
from openai import OpenAI, AsyncOpenAI

# 指向本地 vLLM / Ollama，而非 OpenAI 云
client = OpenAI(
    base_url="http://localhost:8000/v1",   # ← 改这一行即切后端
    api_key="EMPTY",                        # 本地常用占位符
)
```

围绕这套 API 形成的生态层层叠叠：

```
┌──────────────────────────────────────────────┐
│ 上层应用 / UI: Open WebUI, LobeChat, 各类 Bot │
├──────────────────────────────────────────────┤
│ 框架: LangChain(ChatOpenAI), LlamaIndex,      │
│       AutoGen, dify ... 都默认走 OpenAI 形状   │
├──────────────────────────────────────────────┤
│ 官方/多语言 SDK: openai-python, openai-node,  │
│       go/java/rust 社区 SDK                    │
├──────────────────────────────────────────────┤
│ 协议: POST /v1/chat/completions (本文主角)     │
├──────────────────────────────────────────────┤
│ 后端引擎: vLLM / SGLang / TGI / Ollama ...     │
└──────────────────────────────────────────────┘
```

- **官方 SDK** `openai-python` 提供同步 `OpenAI` 与异步 `AsyncOpenAI`；异步版用于高并发场景，避免阻塞事件循环。
- **LangChain / LlamaIndex** 的 `ChatOpenAI` 类同样支持传 `base_url`，于是整条 RAG/Agent 链路无缝切到本地引擎。
- **UI 类**（Open WebUI、LobeChat 等）配置页基本都有"OpenAI API Base URL + Key"两栏，填上你的引擎地址即可。

## 7. 调用全生命周期

一个 `client.chat.completions.create(...)` 背后发生了什么：

```
应用代码 client.chat.completions.create(model,messages,stream)
   │  ① SDK 把参数序列化成 JSON 请求体
   ▼
HTTP 层 POST {base_url}/v1/chat/completions  (带 Bearer Key)
   │  ② 经网关/反代/负载均衡 → 路由到某个引擎实例
   ▼
推理引擎  ③ 解析 model → 找到权重; 用 chat template 渲染 messages → token
   │       ④ tokenizer 编码 → 进入调度队列(批处理/连续批 continuous batching)
   ▼
GPU 前向  ⑤ prefill(处理 prompt) → decode(逐 token 自回归生成)
   │       ⑥ 每生成一个 token，若 stream=true 立即封成 SSE 帧推回
   ▼
返回路径  ⑦ 非流式: 攒齐后组装完整 JSON(含 usage) 一次返回
   │       ⑦' 流式: data: {delta} ... data: [DONE]
   ▼
SDK 解析  ⑧ 反序列化为对象 / 迭代器, 交回应用
```

关键观察：**②③④⑤完全由后端实现，对应用透明**。应用只关心 ①⑦⑧ 这三层"形状"。这正是兼容接口"解耦"的体现——后端如何批处理、用什么注意力优化、跑在哪块卡上，上层一概不知也不必知。

## 8. 函数调用 / tools / 结构化输出

现代 Agent 依赖"让模型调工具"，这一能力也被纳入了兼容接口：

```json
// 请求里声明可用工具
"tools": [{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "查询某地天气",
    "parameters": {"type":"object","properties":{"city":{"type":"string"}}}
  }
}]
```

```json
// 模型决定调用工具时，响应的 finish_reason 为 tool_calls
"message": {
  "role": "assistant",
  "tool_calls": [{
    "id": "call_1",
    "type": "function",
    "function": {"name":"get_weather","arguments":"{\"city\":\"北京\"}"}
  }]
}
```

调用环（应用层负责的循环）：

```
模型 ──tool_calls──▶ 应用执行 get_weather("北京") ──结果──▶
   把结果作为 role:"tool" 消息塞回 messages ──▶ 再次请求模型 ──▶ 模型给出最终自然语言答复
```

此外：
- **JSON mode / 结构化输出**：`response_format: {"type":"json_object"}` 约束模型只输出合法 JSON，便于程序解析；
- **兼容性提示**：function calling 的解析质量依赖后端引擎的模板与解析器实现，**不同引擎对 `tool_calls` 的支持成熟度不一**，以官方文档/源码为准。

## 9. 鉴权、错误与 ConnectionError

鉴权：HTTP 头 `Authorization: Bearer <key>`。本地自托管常用 `EMPTY` 或自定义 key（引擎用 `--api-key` 设置）。

常见状态码与含义：

| 状态码 | 含义 | 应对 |
|--------|------|------|
| 401 | 鉴权失败 | 检查 key / `base_url` 是否多/少 `/v1` |
| 404 | 端点或模型不存在 | 核对 `model` 名、URL 路径 |
| 422/400 | 请求体非法 | 字段名/类型错，看错误 message |
| 429 | 限流 / 排队满 | 退避重试（exponential backoff） |
| 5xx | 后端异常 / OOM | 重试 + 排查显存、batch 配置 |

**ConnectionError（`openai-python` 高频坑，对应注释里 issue 链接）** 多数不是 OpenAI 服务的问题，而是本地/网络层：

```
ConnectionError 排查决策树
        │
   base_url 对吗? ──否──▶ 修正(注意要带 /v1，host/port 正确)
        │是
   服务起来了吗? ──否──▶ 先 curl http://host:port/v1/models 自测
        │是
   代理/防火墙? ──是──▶ 检查 HTTP(S)_PROXY 是否把本地请求也劫持了
        │否            (本地调试常需把 127.0.0.1/localhost 加入 NO_PROXY)
   超时/重试配置 ──▶ 调大 timeout，配置 max_retries
```

- 参考 issue：`https://github.com/openai/openai-python/issues/`（具体编号以仓库 issue 列表为准）。
- 实践要点：本地连本地时，务必把 `localhost,127.0.0.1` 加入 `NO_PROXY`，否则系统代理会把本地请求转出去导致连接失败。

## 典型流程：最小可运行示例（讲含义）

下面示例的"形状"是稳定知识，精确字段以官方文档为准。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # 指向本地 vLLM
    api_key="EMPTY",
    timeout=60, max_retries=2,            # 抗抖动: 超时与重试
)

# 1) 非流式: 拿完整答案
resp = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "用一句话解释 SSE"}],
    temperature=0.7, max_tokens=256,
)
print(resp.choices[0].message.content)
print(resp.usage)                         # token 计数

# 2) 流式: 逐 token 打印(打字机效果)
stream = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "讲个笑话"}],
    stream=True,
)
for chunk in stream:                      # 迭代 SSE 帧
    delta = chunk.choices[0].delta
    if delta.content:                     # 末帧 delta 可能为空
        print(delta.content, end="", flush=True)
```

对应的纯 `curl` 形态（看清"协议"本质）：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b-instruct","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

**异步版**（高并发服务里用）：

```python
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

async def ask(q):
    r = await client.chat.completions.create(
        model="qwen2.5-7b-instruct",
        messages=[{"role": "user", "content": q}],
    )
    return r.choices[0].message.content
# 多个 ask() 用 asyncio.gather 并发，单进程内打满后端吞吐
```

## 常见问题

| 问题 | 答案 |
|------|------|
| "OpenAI 兼容"是不是就要联网调 OpenAI？ | 否。只是 API 形状相同，后端可以是完全本地、离线的引擎 |
| `base_url` 要不要带 `/v1`？ | 通常要。SDK 会在其后拼 `/chat/completions`；多写少写 `/v1` 是最常见 404/401 来源 |
| 流式为什么拿不到 `usage`？ | 默认不返回；试 `stream_options={"include_usage":true}`，由后端是否支持决定 |
| 各引擎是否 100% 兼容？ | 否。chat/embeddings 高度一致；但 tools、`logprobs`、`response_format`、罕用参数差异较大，以各引擎文档为准 |
| chat vs completions 用哪个？ | instruct/chat 模型一律用 `/v1/chat/completions`；base 模型纯续写才用 `/v1/completions` |
| 本地连接报 ConnectionError？ | 多为代理劫持本地请求；把 `localhost,127.0.0.1` 加进 `NO_PROXY`，并先 `curl /v1/models` 自测 |
| 为什么 `messages` 而非一段字符串？ | 角色结构让引擎能按各模型 chat template 正确渲染，且支持 system/tool 等角色 |
| 同步还是异步 SDK？ | 脚本/低并发用 `OpenAI`；服务端高并发用 `AsyncOpenAI` 配 `asyncio` |
| `finish_reason=length` 意味着什么？ | 输出被 `max_tokens` 截断了，需调大上限或精简 prompt |
| temperature 和 top_p 同时调？ | 一般二选一为主。`temperature=0` 近似贪心、可复现；二者都调易难以预测 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引与学习路径
- [[llm-inference/chatgpt]] — ChatGPT：让这套 API 火遍全网的起点
- [[llm-maas/OpenAI-ChatGPT]] — MaaS 视角下的 OpenAI 平台与计费/网关
- [[llm-inference/vllm/README]] — vLLM：最主流的 OpenAI 兼容推理引擎
