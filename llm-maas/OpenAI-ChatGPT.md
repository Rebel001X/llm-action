# MaaS / OpenAI API

> 把"训练好的大模型"包装成一个 HTTP 接口卖出去——这就是 MaaS（Model-as-a-Service）；OpenAI 的 `/v1/chat/completions` 是这套接口事实上的"行业标准插座"。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-maas/README]] [[llm-inference/vllm/README]]

## 阅读地图

| 节 | 你会搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | MaaS、API 即产品 |
| 1 | 地基：HTTP / REST / JSON / Token | 前置概念 |
| 2 | MaaS 是什么、解决什么 | 商业模式、分工 |
| 3 | 请求-响应全链路（架构图） | 网关→鉴权→推理 |
| 4 | Completions vs Chat Completions | 两代接口 |
| 5 | messages 协议与角色 | system/user/assistant |
| 6 | 采样参数：temperature/top_p/... | 含义与权衡 |
| 7 | 流式 SSE | 逐 token 吐字 |
| 8 | Function Calling / tools | 让模型调外部能力 |
| 9 | Embeddings 接口 | 向量化、检索 |
| 10 | Token 与计费 | 怎么算钱 |
| 11 | 兼容生态：vLLM 也实现 | 一套代码切后端 |
| 数值例 | 计费/带宽/吞吐手算 | 落地直觉 |
| FAQ | 高频坑 | 排错 |

## 0. 一句话锚点

**MaaS = 你不买卡、不部署、不运维，只发一个 HTTP 请求，按 token 付钱，拿到模型的输出。** OpenAI 把这套接口的"形状"（URL 路径、请求字段、返回结构）定了下来，后来者（vLLM、DeepSeek、通义、Together、Groq……）大多**照抄这个形状**，于是有了"OpenAI 兼容 API"这个生态。学会这一套，等于学会了 90% 的 LLM 调用方式。

## 1. 地基 / 前置（不假设你记得）

把最底层的几个词拆到原子：

- **HTTP 请求**：客户端给服务器发一段文本，含**方法**（GET/POST）、**URL**、**头部 Header**（元信息，如鉴权）、**Body**（正文，这里是 JSON）。LLM API 几乎全是 `POST`，因为要把一大段 prompt 放进 Body。
- **REST**：一种"用 URL 表示资源、用方法表示操作"的约定。`/v1/chat/completions` 里 `v1` 是版本、`chat/completions` 是资源。
- **JSON**：用 `{}`（对象/字典）和 `[]`（数组/列表）表示数据的纯文本格式。请求和返回都是 JSON。
- **API Key（密钥）**：一串 `sk-...` 的字符串，放在头部 `Authorization: Bearer sk-...`，服务器据此知道"是谁在调、扣谁的钱"。**为什么放 Header 不放 Body**？因为它是身份元信息，且网关层在解析 Body 前就要先鉴权。
- **Token（词元）**：模型不认识"字"，只认识被切分后的整数 ID。一段文本先经**分词器（tokenizer）** 切成 token 序列再喂给模型。**计费、限流、上下文长度都按 token 算，不是按字符**。英文约 1 token ≈ 4 字符 ≈ 0.75 个单词；中文一个汉字常占 1~2 个 token（**以官方分词器实测为准**）。

```
 你发的文本                 分词器                    模型看到的
"你好, world"  ──切分──▶  ["你","好",",", " world"]  ──映射──▶  [56821, 6313, 11, 1917]
                          （token 字符串）              （token id 整数）
```

## 2. MaaS 是什么、解决什么

**问题**：自己跑一个大模型，要买 GPU（一张 H100 公开价约几万美元）、装 CUDA/驱动、部署推理框架、做扩缩容和高可用、盯显存 OOM……门槛极高。
**MaaS 的解法**：供应商把这些全包了，只暴露一个 API。你方只关心"发 prompt、收 completion"。

分工对照：

```
┌─────────────── 自建（IaaS/自部署）───────────────┐   ┌──────────── MaaS ────────────┐
│  买卡 → 装驱动 → 部署 vLLM → 扩缩容 → 监控 → 调模型 │   │            调模型             │
└──────────────────────────────────────────────────┘   └──────────────────────────────┘
        你全包                                                   你只管最后一步
```

权衡：MaaS **省心、起步快、按量付费**，但**数据出境/合规、单价更贵、受限于供应商模型与限流**；自部署反之。很多团队走**混合**：原型用 MaaS，规模化后把高频流量切到自部署的 vLLM（接口同形，几乎零改造，见第 11 节）。

## 3. 请求-响应全链路（架构图）

一个 `chat/completions` 请求在服务端经历的层：

```
客户端
  │  POST /v1/chat/completions   { model, messages, ... }
  ▼
┌──────────────┐  鉴权失败→401
│  API 网关    │  ① 校验 API Key ② 限流(RPM/TPM) ③ 路由到对应模型集群
└──────┬───────┘
       ▼
┌──────────────┐
│  调度/排队   │  ④ 进入该模型的请求队列，做 continuous batching
└──────┬───────┘
       ▼
┌──────────────┐
│ 推理引擎     │  ⑤ tokenizer 编码 → ⑥ prefill(读入prompt) → ⑦ decode(逐token生成)
│ (如 vLLM)    │     ⑧ 命中 KV-Cache 则跳过重复计算
└──────┬───────┘
       ▼
┌──────────────┐
│ 计费/日志    │  ⑨ 统计 prompt_tokens + completion_tokens → 扣费、记账
└──────┬───────┘
       ▼
客户端  ◀── JSON 响应（或 SSE 流式分片）
```

记住三件事：**鉴权与限流在最前**、**生成是 prefill+decode 两阶段**、**返回里一定带 usage 用于计费**。

## 4. Completions vs Chat Completions（两代接口）

| 维度 | `/v1/completions`（旧/补全） | `/v1/chat/completions`（主流/对话） |
|------|------------------------------|--------------------------------------|
| 输入 | 单个字符串 `prompt` | 结构化 `messages` 数组 |
| 心智 | "续写这段文字" | "多轮对话，带角色" |
| 适配模型 | 基座/instruct 类 | 对话微调模型（绝大多数新模型） |
| 现状 | 多被视为 legacy | **事实标准**，新项目首选 |

旧接口（续写）：

```
curl https://api.openai.com/v1/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "model": "gpt-3.5-turbo-instruct",
        "prompt": "Say this is a test", "max_tokens": 7, "temperature": 0 }'
```

新接口（对话，**你以后基本只用这个**）：

```
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "model": "gpt-3.5-turbo",
        "messages": [
          {"role": "system",    "content": "You are a helpful assistant."},
          {"role": "user",      "content": "Hello!"}
        ] }'
```

返回（节选，注意 `choices` 和 `usage`）：

```
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [
    { "index": 0,
      "message": {"role": "assistant", "content": "Hi! How can I help?"},
      "finish_reason": "stop" }      ← 为什么停：stop=自然结束 / length=到max_tokens / tool_calls=要调工具
  ],
  "usage": { "prompt_tokens": 19, "completion_tokens": 8, "total_tokens": 27 }
}
```

## 5. messages 协议与角色

`messages` 是一个**按时间顺序排列的对话历史数组**，每条 `{role, content}`。三个核心角色：

```
┌─────────┐  全局指令/人设/约束。优先级最高，决定模型"是谁、守什么规矩"
│ system  │  例：你是严谨的法律助手，只用中文，不确定就说不知道
├─────────┤
│ user    │  用户说的话
├─────────┤
│assistant│  模型之前的回复（多轮时把上一轮答案放回来）
├─────────┤
│ tool    │  工具执行结果回填（配合第 8 节 function calling）
└─────────┘
```

**关键认知：API 是无状态的**。服务器**不记得**你上一次说了什么——每一轮你都要把**完整历史**重新发过去。所谓"多轮对话"，本质是客户端不断把 `messages` 数组追加变长再整体重发。

```
第1轮发: [system, user1]                       → assistant1
第2轮发: [system, user1, assistant1, user2]    → assistant2   ← 历史全带上！
```

代价：历史越长，`prompt_tokens` 越多、越贵、越慢，且受**上下文窗口**（context window，模型一次能看的最大 token 数）上限约束。超了就要做截断或摘要。

## 6. 采样参数：含义与权衡（讲含义，不背默认值）

模型每步输出的是**下一个 token 的概率分布**；采样参数控制"怎么从分布里挑"：

- `temperature`（温度，$\ge 0$）：把概率分布"拉平或拉尖"。原理是对 logits 除以 $T$ 再 softmax：
  $$p_i=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$
  $T\to 0$ 趋近贪心（最确定、可复现，适合代码/抽取）；$T$ 大则更随机有创意（适合头脑风暴），过大会胡言乱语。
- `top_p`（核采样）：只在"累计概率达到 $p$ 的最小 token 集合"里采。`top_p=0.1` 即只考虑最可能的那一小撮。**一般 `temperature` 和 `top_p` 调一个就好，别同时大改**。
- `max_tokens` / `max_completion_tokens`：**只限输出长度**，不限输入。设小可省钱但可能被截断（`finish_reason=length`）。
- `n`：一次生成几个候选。`n=3` 大致按 3 倍输出 token 计费。
- `stop`：遇到指定字符串就停。
- `frequency_penalty` / `presence_penalty`：降低重复。前者按出现**次数**惩罚，后者只看**是否出现过**。
- `seed`：配合低温**尽量**复现（"尽量"，不保证 bit 级一致，以官方为准）。

```
logits ──/T──▶ softmax ──top_p 截断──▶ 在保留集合里按概率抽一个 token ──▶ 拼回输出
        温度越低分布越尖           p 越小候选越少
```

## 7. 流式输出 SSE（逐 token 吐字）

不加 `"stream": true` 时，服务器**算完整段才一次性返回**——长回答要等很久，体验差。流式则**生成一个 token 就推一片**，前端打字机效果，**首字时延（TTFT）** 大幅降低。

底层用 **SSE（Server-Sent Events）**：一条长连接的 HTTP 响应，`Content-Type: text/event-stream`，服务器不断写出 `data: {...}\n\n` 这样的小块，最后以 `data: [DONE]` 收尾。

```
请求: {... "stream": true}
响应流（一行一行陆续到达）:
  data: {"choices":[{"delta":{"role":"assistant"}}]}
  data: {"choices":[{"delta":{"content":"Hi"}}]}        ← delta 是增量，不是全文
  data: {"choices":[{"delta":{"content":"!"}}]}
  data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
  data: [DONE]                                           ← 流结束标记
```

要点：①每片是 **delta（增量）**，客户端自己拼接；②默认流式响应里**没有 usage**，要算 token 得自己数或显式请求（以官方为准）；③网络中断要能续/重试。SSE vs WebSocket：SSE 是**单向、基于 HTTP、实现简单**，正好契合"服务器单向吐 token"，所以成了主流选择。

## 8. Function Calling / tools（让模型调外部能力）

模型本身**不会查天气、不会算账、不会读你的数据库**。Function Calling 让你**声明一组工具（带 JSON Schema 参数）**，模型在需要时**不直接作答，而是输出"我要调用 `get_weather(city='北京')`"这个结构化意图**，由**你的代码**真正执行，再把结果回填给模型续答。

完整闭环（关键：模型只"提议"，执行权永远在你手里）：

```
① 你: messages + tools(声明 get_weather: {city:string})
        │
        ▼
② 模型: finish_reason="tool_calls"
        message.tool_calls=[{id:"call_1", function:{name:"get_weather",
                             arguments:"{\"city\":\"北京\"}"}}]   ← 只是文本意图
        │
        ▼
③ 你的代码: 真正去调天气服务 → "晴, 26℃"
        │
        ▼
④ 你: 把结果作为 {role:"tool", tool_call_id:"call_1", content:"晴,26℃"}
       连同历史再发一次
        │
        ▼
⑤ 模型: "北京今天晴，26℃。"   ← 自然语言最终答复
```

权衡与坑：① `arguments` 是**模型生成的字符串**，可能不合 schema，要校验、要容错；②可设 `tool_choice` 强制/禁止调用；③这是 **Agent、MCP、RAG 编排**的基础原语——"让 LLM 操作世界"几乎都靠它；④多工具时模型可能一次返回多个 tool_calls。

## 9. Embeddings 接口（向量化）

`/v1/embeddings` 与生成无关：它把一段文本**映射成一个定长浮点向量**（如 1536 维），语义相近的文本向量也相近。用途：**语义检索、RAG、聚类、去重、推荐**。

```
"猫"  ─embeddings→ [0.01, -0.22, ..., 0.07]  ┐
"猫咪" ─embeddings→ [0.02, -0.20, ..., 0.06]  ├ 余弦相似度高 → 判为语义相近
"汽车" ─embeddings→ [0.41,  0.10, ..., -0.3]  ┘ 与"猫"相似度低
```

相似度常用**余弦相似度** $\cos\theta=\dfrac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}\in[-1,1]$，越接近 1 越像。RAG 流程：把知识库文档逐段 embed 存进向量库 → 用户问题也 embed → 取最相近的若干段 → 拼进 prompt 交给 chat 接口作答。Embeddings 也**按输入 token 计费**，但通常单价远低于生成。

```
curl https://api.openai.com/v1/embeddings \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d '{ "model": "text-embedding-3-small", "input": "你好，世界" }'
```

## 10. Token 与计费（怎么算钱）

**计费公式**（最核心）：

$$\text{费用}=\frac{\text{prompt\_tokens}}{10^6}\times P_{in}+\frac{\text{completion\_tokens}}{10^6}\times P_{out}$$

其中 $P_{in}$、$P_{out}$ 是每百万 token 的输入/输出单价（**输出通常比输入贵数倍**，因为 decode 阶段一个一个算、最耗算力）。具体单价**以官方价目表为准**，且随模型不同差异巨大。

省钱/省时的几个杠杆：

- **输入侧**：精简 system、多轮对话做历史截断/摘要，少发冗余上下文。
- **Prompt Caching（提示缓存）**：很多供应商对**重复出现的前缀**（如固定的长 system）给缓存折扣，命中后这部分输入更便宜（**以官方为准**）。
- **输出侧**：用 `max_tokens` 封顶、让模型"简短作答"。
- **限流（rate limit）**：两个维度——**RPM**（每分钟请求数）和 **TPM**（每分钟 token 数）。打满会收到 `429 Too Many Requests`，要做**指数退避重试**。

## 11. 兼容生态：vLLM 也实现（一套代码切后端）

这是本文最有价值的工程认知：**OpenAI 的接口形状已成行业标准**。[[llm-inference/vllm/README]] 启动后会暴露一个 **OpenAI 兼容 server**，路径同样是 `/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`，请求/返回字段也对齐。于是：

```
                       只改两行：base_url + api_key
                                  │
┌──────────────┐                  ▼
│ 你的应用代码 │ ──OpenAI SDK──▶  ┌─ api.openai.com/v1   （MaaS，按量付费）
│ (一份不变)   │                  ├─ 你的 vLLM:8000/v1   （自部署，省钱可控）
└──────────────┘                  └─ DeepSeek/通义/.../v1（其它供应商）
```

启动一个本地兼容服务（命令与参数**以 vLLM 官方文档为准**，此处示意）：

```
python -m vllm.entrypoints.openai.api_server \
  --model <你的模型> --port 8000
# 然后客户端把 base_url 指到 http://localhost:8000/v1，api_key 随便填一个占位
```

价值：**原型期用 MaaS 快速验证，规模化后把高频流量切到自建 vLLM**，应用层几乎零改动。兼容也意味着**LangChain / LlamaIndex / 各类 Agent 框架开箱即用**。注意：兼容**不等于 100% 对齐**——某些高级字段（特定的 function calling 细节、logprobs、特殊采样参数）各家支持程度不一，**以各自文档为准**。

## 数值例子 / 对照 / 实践

**例 1：单次对话费用手算**。设某模型 $P_{in}=1$、$P_{out}=4$（美元/百万 token，**示例值非真实价**）。一次客服问答 prompt 用了 800 token、回答 200 token：
- 输入：$800/10^6\times 1=0.0008$ 美元
- 输出：$200/10^6\times 4=0.0008$ 美元
- 合计约 **\$0.0016/次**；日活 10 万次/天 → 约 **\$160/天 ≈ \$4800/月**。直觉：**输出虽少但单价高，常和输入费用相当甚至更高**，所以"让模型简短回答"很省钱。

**例 2：多轮对话的隐藏成本**。每轮历史全带，第 $k$ 轮的输入 token 约 $\sum_{i=1}^{k}(\text{第}i\text{轮新增})$，**随轮数近似线性增长**。10 轮后单次请求可能比首轮贵 5~10 倍——这就是为什么要做历史摘要。

**例 3：流式 vs 非流式体感**。设 decode 速度 40 token/s、回答 200 token。非流式：用户等约 $200/40=5$ 秒才见到第一个字；流式：**TTFT 仅约首 token 的几十毫秒~数百毫秒**，后续逐字出，总时长不变但**体感快得多**。

**实践清单**：① Key 放环境变量/密钥管理，**绝不进代码仓库与前端**（前端直连会泄露 Key，应走自己的后端转发）；②对 `429/5xx` 做指数退避重试；③记录每次 `usage` 做成本归因；④对 function calling 的 `arguments` 严格校验；⑤长上下文做截断或摘要防超窗与高费；⑥用 OpenAI SDK + 改 `base_url` 保持后端可切换。

## 常见问题

| 问题 | 答 |
|------|-----|
| 为什么我每轮都要重发历史？ | API **无状态**，服务器不记历史，多轮靠客户端拼 `messages` 重发。 |
| `finish_reason=length` 是出错吗？ | 不是，是**到了 `max_tokens` 被截断**。想要完整答案就调大上限。 |
| temperature 和 top_p 一起调？ | 一般**只动一个**。要确定性/可复现就 `temperature=0`。 |
| 流式里怎么算 token？ | 默认流式分片**不含 usage**，需自己数或显式请求统计（以官方为准）。 |
| function calling 模型会自己执行函数吗？ | **不会**。它只输出调用意图，真正执行永远是你的代码。 |
| 429 怎么办？ | 触发了 RPM/TPM 限流，做**指数退避重试**或申请提额。 |
| 中文 1 字几个 token？ | 常 1~2 个，**以官方分词器实测为准**，别按字符估。 |
| 能无缝从 OpenAI 换到 vLLM 吗？ | 基础能力可以，改 `base_url` 即可；高级字段不一定全兼容，看各自文档。 |
| 数据会被用于训练吗？ | 各供应商策略不同，**合规/隐私须看官方条款**，敏感数据慎用公有 MaaS。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全局索引，先回这里
- [[llm-maas/README]] — MaaS 专题总览（本文上层）
- [[llm-inference/vllm/README]] — 自部署 OpenAI 兼容服务（第 11 节落地）
- 官方参考：
  - https://platform.openai.com/docs/api-reference/chat/create
  - https://platform.openai.com/docs/guides/chat-completions
  - https://platform.openai.com/docs/guides/text-generation/managing-tokens
  - https://platform.openai.com/docs/api-reference/completions
  - https://platform.openai.com/docs/api-reference/chat
