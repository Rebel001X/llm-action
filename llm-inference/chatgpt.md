# ChatGPT API 调用

> 以 HTTP 请求把"对话历史"喂给托管在云端的大模型，拿回一段（可流式的）补全文本——这就是 Chat Completions API。📍 导航：[[00-知识地图]]
> 🔗 相关：[[00-知识地图]] [[llm-inference/openai]] [[llm-inference/解码策略]] [[llm-maas/OpenAI-ChatGPT]]

## 阅读地图

| 节 | 你会搞懂 | 关键词 |
|----|----------|--------|
| 0 | 一句话锚点 | 请求/响应/无状态 |
| 1 | 它解决什么问题 | 托管推理、为什么是"chat"接口 |
| 2 | 请求生命周期（调用链 ASCII） | TLS→鉴权→tokenize→解码→返回 |
| 3 | messages 与三/四种角色 | system/user/assistant/tool |
| 4 | 采样参数语义 | temperature/top_p/max_tokens/penalty |
| 5 | 流式 SSE | stream=true、data: chunk、[DONE] |
| 6 | function / tool calling | tools、tool_calls、回填 |
| 7 | 成本与 token 计费 | prompt/completion token、缓存 |
| 8 | 错误、重试、限流 | 429/5xx、指数退避 |
| 示例 | 端到端调用范例 | curl / Python |
| FAQ | 高频坑 | 表格 |

## 0. 一句话锚点

ChatGPT API = 一个**无状态的 HTTP POST 接口** `POST /v1/chat/completions`。你每次把**完整对话历史**（`messages` 数组）连同**采样参数**发过去，服务端在 GPU 上做一次自回归解码，把生成的 token 解码成文本回给你。**服务端不记得上一轮**——上下文记忆完全由你在客户端拼接历史来实现。

```
你的代码  ──(整段历史 + 参数)──▶  OpenAI 服务端 ──(一段补全)──▶  你的代码
   ▲                                                                 │
   └───────────  把回复 append 进 messages，下一轮再发  ◀────────────┘
```

## 1. 地基：它解决什么问题

**问题**：GPT 这类大模型有几百亿到上万亿参数，单次推理要几十 GB 显存 + 专用 GPU。个人/中小团队既没硬件也没工程能力自己部署。

**方案**：OpenAI 把模型托管在自己的集群里，对外只暴露一个 REST 接口。你按"用了多少 token"付费，省去采购 GPU、加载权重、做 KV-Cache/批处理/调度的全部工程。

**为什么是"chat"而不是"补全"**：早期的 `/v1/completions`（文本补全）只接收一个纯字符串 prompt。但 GPT-3.5/4 是用**对话格式 + 角色**做指令微调（RLHF）的，模型内部期望看到结构化的"谁说了什么"。于是新接口把输入设计成 `messages` 数组，每条带 `role`。这让模型能区分"系统设定 / 用户提问 / 助手历史回复"，对齐效果显著更好。**新项目一律用 Chat Completions**；老的 `/v1/completions` 已是 legacy。

```
旧: /v1/completions          新: /v1/chat/completions
   prompt = "纯字符串"           messages = [
                                   {role:system,  ...},
                                   {role:user,    ...},
                                   {role:assistant,...},
                                 ]
   ──► 模型只能"续写"          ──► 模型理解"对话结构 + 角色指令"
```

> 模型名（如 `gpt-3.5-turbo`、`gpt-4o` 等）和具体可用列表请以官方文档为准；本文用占位名讲机制，不锁定版本。

## 2. 请求生命周期（一次调用从发出到返回）

```
┌─────────────┐   HTTPS POST /v1/chat/completions
│  Client     │   Header: Authorization: Bearer sk-...
│ (你的程序)  │   Body:   {model, messages, temperature, ...} (JSON)
└──────┬──────┘
       │ 1. TLS 握手 + 鉴权（校验 API Key、组织、配额）
       ▼
┌─────────────┐   2. 解析 JSON、校验参数合法性（model 是否存在等）
│  API 网关   │   3. 限流：检查 RPM / TPM 是否超额 → 超则 429
└──────┬──────┘
       │ 4. 把 messages 按模板拼成单一 token 序列
       ▼
┌─────────────┐   5. tokenizer：文本 → token id（BPE 分词）
│  Tokenizer  │      "Hello!" → [9906, 0]  (示意)
└──────┬──────┘
       │  prompt_tokens 数量在此确定（影响计费）
       ▼
┌─────────────┐   6. 自回归解码：逐 token 生成
│  GPU 推理   │      每步算 logits → 按 temperature/top_p 采样下一个 token
│ (forward)   │      用 KV-Cache 复用历史，直到 EOS / 撞 max_tokens / 命中 stop
└──────┬──────┘
       │  completion_tokens 在此累加
       ▼
┌─────────────┐   7. detokenize：token id → 文本
│  组装响应   │   8. 算用量、finish_reason，封装成 JSON（或逐块 SSE）
└──────┬──────┘
       ▼
   返回给 Client：{choices:[{message:{...}}], usage:{...}}
```

**两条关键认知**：
1. **无状态**：第 4 步只看你这次发来的 `messages`。服务端不存历史。要多轮，就把上一轮的 assistant 回复 append 回数组再发——所以多轮对话每次都重传全部历史，token 越发越多。
2. **计费点**：`prompt_tokens`（第 5 步算出）+ `completion_tokens`（第 6 步累加）= `total_tokens`，这是账单的唯一依据。

## 3. messages 与角色

`messages` 是一个有序数组，每个元素是 `{"role": ..., "content": ...}`。角色决定模型如何理解这段文本。

```
messages = [
  ┌───────────────────────────────────────────────┐
  │ role: "system"     ← 全局设定/人设/规则        │  通常放第 1 条
  │   "你是一位严谨的中文 AI-Infra 讲师..."        │  权重高、约束行为
  ├───────────────────────────────────────────────┤
  │ role: "user"       ← 真人这一轮说的话          │
  │   "解释一下 top_p"                             │
  ├───────────────────────────────────────────────┤
  │ role: "assistant"  ← 模型上一轮的回复（历史）  │  你回填进来给模型"记忆"
  │   "top_p 是核采样..."                          │
  ├───────────────────────────────────────────────┤
  │ role: "user"       ← 真人本轮新问题            │
  │   "那它和 temperature 冲突吗？"               │
  ├───────────────────────────────────────────────┤
  │ role: "tool"       ← 工具执行结果（见第 6 节） │  function calling 专用
  └───────────────────────────────────────────────┘
]
```

| 角色 | 谁产生 | 作用 | 要点 |
|------|--------|------|------|
| `system` | 你 | 设定身份、风格、硬规则 | 放最前；不是越长越好；模型不保证 100% 服从 |
| `user` | 你（代表终端用户） | 真实输入/指令 | 多轮里有多条 |
| `assistant` | 模型（或你回填历史） | 模型的回复 | 多轮时把上轮回复原样放回，模型才"记得" |
| `tool` | 你（执行函数后） | 把函数返回值喂回模型 | 须带对应 `tool_call_id` |

> 提示词注入风险：`system` 的优先级高于 `user`，但**不是绝对隔离**。不要把未经清洗的外部内容直接拼进 `system` 当作可信指令。

## 4. 采样参数语义（决定"怎么挑下一个 token"）

模型每一步输出的是词表上的一个概率分布（logits 经 softmax）。这些参数控制**如何从分布里采样**。详细原理见 [[llm-inference/解码策略]]。

```
logits ──/temperature──▶ softmax ──top_p/top_k 截断──▶ 重新归一化 ──采样──▶ 下一个 token
```

| 参数 | 取值 | 含义 | 调高/调低的效果 | 权衡 |
|------|------|------|----------------|------|
| `temperature` | 0~2 | 把 logits 除以 $T$ 再 softmax，缩放分布"尖锐度" | $T\to0$ 趋近贪心、确定、保守；$T$ 大→更随机有创意 | 高 $T$ 易胡说；代码/抽取类任务建议 $0\sim0.3$ |
| `top_p`（核采样） | 0~1 | 只在累计概率达 $p$ 的最小 token 集合里采样 | $p$ 小→候选集小、更稳；$p=1$ 不截断 | 与 temperature 二选一调，别同时大改 |
| `max_tokens` / `max_completion_tokens` | 整数 | **生成部分**的 token 上限 | 太小→回答被截断（`finish_reason:length`） | 是硬上限，不含 prompt；直接影响成本 |
| `stop` | 字符串/数组 | 命中即停止生成 | 用于定界（如 `"\n\n"`） | 停止符本身不返回 |
| `n` | 整数 | 一次返回几条候选 | `n=3` 返回 3 个 choices | 计费按 $n$ 倍 completion_tokens |
| `presence_penalty` | -2~2 | 对**已出现过**的 token 施加惩罚 | 调高→更愿意引入新话题/新词 | 过高→偏题 |
| `frequency_penalty` | -2~2 | 按 token **出现频次**惩罚 | 调高→抑制重复啰嗦 | 过高→用词生硬 |
| `seed` | 整数 | 尽量可复现采样（best-effort） | 同 seed+同输入→尽量同输出 | 不保证 100% 确定，配合 `system_fingerprint` 看 |
| `response_format` | 对象 | 约束输出格式（如 JSON） | 强制返回合法 JSON | 需配合 prompt 说明 schema |
| `logprobs` / `top_logprobs` | 布尔/整数 | 返回每个 token 的对数概率 | 用于置信度分析 | 增大响应体 |

**temperature 数值例子**：设某步两个候选 logits 为 $z_A=2.0, z_B=1.0$。
- $T=1$：$p_A=\dfrac{e^{2}}{e^{2}+e^{1}}=\dfrac{7.389}{7.389+2.718}\approx0.731$，$p_B\approx0.269$。
- $T=0.5$（除以 0.5 即放大）：用 $z/T=(4,2)$，$p_A=\dfrac{e^{4}}{e^{4}+e^{2}}=\dfrac{54.6}{54.6+7.39}\approx0.881$，更确定。
- $T=2$：用 $z/T=(1,0.5)$，$p_A=\dfrac{e^{1}}{e^{1}+e^{0.5}}\approx0.622$，更平、更随机。

**经验法则**：`temperature` 和 `top_p` **不要同时大幅调**，选一个为主旋钮即可，否则相互干扰难以预测。

## 5. 流式输出（streaming, SSE）

默认 `stream=false`：服务端把整段回答生成完，一次性返回完整 JSON——首字延迟（TTFT）= 整段生成时间，长回答体验差。

设 `stream=true`：改用 **Server-Sent Events**，每生成几个 token 就推一个 `data:` 块过来，前端可像打字机一样边收边显。

```
client ──POST {... "stream": true}──▶ server
                                       │
   ◀── data: {"choices":[{"delta":{"role":"assistant"}}]}
   ◀── data: {"choices":[{"delta":{"content":"top"}}]}
   ◀── data: {"choices":[{"delta":{"content":"_p"}}]}
   ◀── data: {"choices":[{"delta":{"content":" 是"}}]}
            ... 逐块 ...
   ◀── data: {"choices":[{"finish_reason":"stop","delta":{}}]}
   ◀── data: [DONE]          ← 哨兵，收到即结束
```

**与非流式的差异**：
- 字段从 `message` 变成 `delta`（增量），你需要把每块的 `delta.content` 拼接起来。
- 默认流式响应**不带 `usage`**（需要时通过 `stream_options:{include_usage:true}` 等方式获取，以官方文档为准），所以流式下要自己估算 token。
- 错误可能发生在流中途，要处理"已收到一半又断开"的情况。

**何时用流式**：聊天 UI、长文生成、需要低 TTFT 的交互式场景。**何时不用**：后端批处理、需要拿到完整结构化 JSON 再解析、要精确 usage 的计费场景。

## 6. function / tool calling（让模型调用你的函数）

**它解决什么**：模型只会生成文本，不能自己查数据库、调天气 API、做精确计算。Tool calling 让模型在需要时**输出一个"调用意图"**（函数名 + JSON 参数），由**你的代码**实际执行，再把结果喂回去，模型据此给最终答复。模型本身从不执行任何代码——它只负责"决定调谁、传什么参数"。

```
① 你声明工具          tools=[{type:"function",
                              function:{name:"get_weather",
                                        parameters:{city:string}}}]
        │
        ▼
② 用户提问  "北京今天天气？" ──POST(messages + tools)──▶ 模型
        │
        ▼
③ 模型返回意图（注意：不是答案）
   finish_reason: "tool_calls"
   message.tool_calls = [{ id:"call_1",
                           function:{name:"get_weather",
                                     arguments:'{"city":"北京"}'}}]
        │
        ▼
④ 你的代码真正执行 get_weather("北京") → "晴, 28℃"
        │
        ▼
⑤ 把结果作为 role:"tool" 回填（带 tool_call_id），连同原历史再发一次
   messages += {role:"assistant", tool_calls:[...]}   ← 模型那条意图
   messages += {role:"tool", tool_call_id:"call_1", content:"晴, 28℃"}
        │
        ▼
⑥ 模型这次生成自然语言答复  "北京今天晴，28℃。"
```

**关键字段**：
- `tools`：请求里声明可用函数，每个含 `name`、`description`、JSON-Schema 形式的 `parameters`。
- `tool_choice`：`"auto"`（模型自己决定）/ `"none"`（禁用）/ 指定某函数（强制调用）。
- 响应里 `message.tool_calls`：数组，支持**并行**多调用；`arguments` 是**字符串化的 JSON**，用前要 `json.loads` 且**做校验**（模型可能给出不合 schema 的参数）。
- 回填那条必须是 `role:"tool"` 且 `tool_call_id` 对上号，否则模型对不上是哪个调用的结果。

> 早期的 `functions`/`function_call` 字段是 tool calling 的前身，现已被 `tools`/`tool_calls` 取代；语义类似，新代码用 `tools`。

## 7. 成本与 token 计费

**计费单位是 token，不是字**。一个英文词≈1.3 token，一个中文字大致 1~2+ token（取决于 tokenizer），标点/空格也算。

$$\text{费用} = \frac{\text{prompt\_tokens}}{1000}\times P_{in} + \frac{\text{completion\_tokens}}{1000}\times P_{out}$$

其中 $P_{out}$（输出价）通常**显著高于** $P_{in}$（输入价）。**具体单价随模型与时间变化，务必以官方 pricing 页为准**，下面只示意算法。

```
一次请求的账单构成
┌────────────────────────────────────────────┐
│ prompt_tokens     = system + 全部历史 + 本轮 │  ← 多轮里会越滚越大
│ completion_tokens = 这次生成的内容           │  ← 受 max_tokens 限制
│ total_tokens      = 上两者之和               │
└────────────────────────────────────────────┘
响应 usage 字段会原样给出这三个数。
```

**数值例子**（单价为假设，仅演示）：设 $P_{in}=0.5\$/1\text{M}$ tok，$P_{out}=1.5\$/1\text{M}$ tok。一次请求 `prompt=1000`、`completion=500`：
$$\frac{1000}{10^6}\times0.5 + \frac{500}{10^6}\times1.5 = 0.0005 + 0.00075 = 0.00125\ \$$$

**省钱要点**：
1. **控制历史长度**：多轮无状态意味着历史每轮重发。超长对话要做**摘要/截断/滑窗**，否则 prompt_tokens 线性爆炸。
2. **设 `max_tokens`**：给输出封顶，防止跑飞。
3. **利用 prompt 缓存**：相同前缀（如固定的长 system）在支持缓存的模型上可享折扣输入价（机制以官方为准）。
4. **选对模型**：能用小模型/便宜模型完成的任务（分类、抽取）别上最贵的。
5. **批量/异步**：若有 Batch 接口，离线任务可换更低价。

## 8. 错误、重试与限流

```
状态码        含义              客户端应对
─────────────────────────────────────────────
200          成功              正常解析
400          请求体非法        修参数/消息，不要重试
401          鉴权失败          查 API Key / 组织
403          无权限/区域受限   查账号
404          model 不存在      改 model 名
429          限流(RPM/TPM)或欠费  指数退避重试 / 降速 / 充值
500/503      服务端错误        指数退避重试
```

**指数退避**（对 429/5xx）：第 $k$ 次重试等待约 $\text{base}\times2^{k}$ 秒并加随机抖动，最多重试若干次。

```
失败 → wait 1s±jitter → 失败 → wait 2s±jitter → 失败 → wait 4s±jitter → ... → 放弃
```

**限流两个维度**：RPM（每分钟请求数）和 TPM（每分钟 token 数），与账户等级挂钩。批量任务要主动节流，别一股脑并发打满。

## 端到端示例（讲含义，非锁定语法）

**A. curl 非流式**：

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

返回（精简）：

```json
{
  "id": "chatcmpl-...",
  "choices": [
    { "index": 0,
      "message": { "role": "assistant", "content": "Hi! How can I help?" },
      "finish_reason": "stop" }
  ],
  "usage": { "prompt_tokens": 19, "completion_tokens": 8, "total_tokens": 27 }
}
```

`finish_reason` 取值：`stop`（自然结束/命中 stop）、`length`（撞 max_tokens 被截断）、`tool_calls`（要调函数）、`content_filter`（被过滤）。

**B. curl 流式**（沿用原文件方向，加 `"stream": true`）：

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role":"user","content":"写一句鼓励的话"}],
    "stream": true
  }'
# 输出是一串 data: {...} 行，最后以 data: [DONE] 结束
```

**C. Python（多轮 + 流式骨架，伪代码）**：

```python
from openai import OpenAI
client = OpenAI()                       # 读环境变量 OPENAI_API_KEY

messages = [{"role": "system", "content": "你是中文 AI-Infra 讲师"}]
messages.append({"role": "user", "content": "什么是 KV-Cache?"})

stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.3,
    stream=True,
)
reply = ""
for chunk in stream:                    # 逐块拼接 delta
    delta = chunk.choices[0].delta.content or ""
    reply += delta
    print(delta, end="", flush=True)

messages.append({"role": "assistant", "content": reply})  # 回填，供下一轮记忆
```

> 真实 SDK 的类名/方法名以官方文档为准；上面演示的是"拼历史 → 发请求 → 流式收 delta → 回填"这条不变的主线。

## 常见问题

| 问题 | 答案 |
|------|------|
| 为什么模型"忘了"上一轮？ | API 无状态，你没把上轮回复回填进 `messages`。记忆 = 客户端拼历史。 |
| `temperature=0` 就完全确定吗？ | 趋近确定但不保证 100% 可复现；要复现还需 `seed` 且仍是 best-effort。 |
| `temperature` 和 `top_p` 一起调？ | 不建议同时大改，选一个主旋钮，否则行为难预测。 |
| 回答被截断了？ | `finish_reason:length`，调大 `max_tokens`，或缩短输入。 |
| 多轮越用越贵正常吗？ | 正常。历史每轮重发，prompt_tokens 累积。需摘要/截断/滑窗。 |
| tool_calls 的 arguments 能直接信吗？ | 不能。是模型生成的 JSON 字符串，须 `json.loads` + schema 校验后再用。 |
| 流式为什么拿不到 usage？ | 默认流式不返回 usage，需开 `include_usage` 或自行估算 token。 |
| 遇到 429 怎么办？ | 区分是限流还是欠费；限流就指数退避+降并发，必要时升级配额。 |
| system 能 100% 约束模型吗？ | 不能，只是强先验；别把它当安全边界，注意提示注入。 |
| 中文一个字几个 token？ | 取决于 tokenizer，常见 1~2+ token/字，请以实际 tokenizer 统计为准。 |

## 🔗 跳转链接

- [[00-知识地图]] — 全站导航总入口
- [[llm-inference/openai]] — OpenAI 接口/SDK 体系全貌
- [[llm-inference/解码策略]] — temperature / top_p / top_k / 贪心 / beam 的底层原理
- [[llm-maas/OpenAI-ChatGPT]] — 以 MaaS（模型即服务）视角看 ChatGPT 的产品与计费形态
