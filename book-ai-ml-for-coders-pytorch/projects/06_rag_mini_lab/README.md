# 06 · 迷你 RAG 实战：检索增强生成 + 可插真实 LLM

> 配套章节：**第 18 章（RAG 检索增强生成）**，并串起 **第 14 章（Hugging Face Hub）**、**第 15 章（Transformer / transformers）**、**第 16 章（微调 LLM）**、**第 17 章（Ollama 本地部署）**。

一个**完全离线可跑**的迷你 RAG 系统：不联网、不下载任何数据/模型、纯 CPU、秒级完成，`pytest` 全绿。同时留好了**一键切到真实 LLM**（Ollama 或微调后的 HF 模型）的口子。

---

## 1. 这个项目在讲什么

第 18 章的一句话本质：**RAG = 用「检索」补上 LLM 不知道的私有事实 + 用「生成」把这些事实组织成答案。它不改一个模型参数，却能让没读过你数据的模型答得像读过一样。**

完整链路（本项目一比一复刻，只是把书里的重依赖换成零依赖的小实现）：

```
问题 ──embed──► 向量 ──检索(余弦 top-k)──► 命中片段 ──拼 prompt──► 生成答案
```

| 书里用的 | 本项目替代 | 为什么 |
|---|---|---|
| OpenAI Embeddings（需付费 key、联网） | `DeterministicEmbedder`（hashing 词袋，纯 numpy） | 离线、确定性、可单元测试 |
| ChromaDB（SQLite 向量库） | `VectorStore`（numpy 余弦 top-k） | 几十行看清检索内核 |
| PDF 小说 → 切块 | `knowledge.py` 里 16 条明确事实 | 免下载、结果可验证 |
| Ollama / GPT 生成 | 默认离线模板生成 + 可选 Ollama/HF 分支 | 默认绿；真实 LLM 一键接上 |

---

## 2. 三分钟跑通

```bash
# 1) 进目录
cd projects/06_rag_mini_lab

# 2) 依赖（本仓库环境已装好 numpy / pytest，通常无需再装）
pip install -r requirements.txt

# 3) 跑测试（应全绿，秒级）
python -m pytest -q

# 4) 看演示：问几个问题，打印检索片段 + 生成答案
python run_demo.py
```

预期：`pytest` 输出 `11 passed`；`run_demo.py` 对每个问题打印 top-3 检索片段（带相似度）和一段离线生成的答案。

---

## 3. 每个文件干嘛

| 文件 | 作用 | 对应章节要点 |
|---|---|---|
| `embed.py` | `DeterministicEmbedder`：离线确定性句向量（hashing 词袋 → L2 归一化）；另有可选 `SentenceTransformerEmbedder`（try import，缺失自动放弃） | 第 5/6 章：文本编码 + embedding 让语义可计算 |
| `store.py` | `VectorStore.add(id, text, vec)` / `.search(query_vec, k)`，用**余弦相似度**返回 top-k | 第 18 章：向量库 + 相似度检索 |
| `knowledge.py` | 16 条关于 PyTorch / 本书章节的明确事实（RAG 里的"私有数据"） | 第 18 章：待检索的源材料 |
| `rag.py` | `build_store()` 建库、`retrieve()` 检索、`answer()` 生成；含 `generate_offline / generate_with_ollama / generate_with_hf` 三条生成分支 | 第 18 章主流程 + 14/16/17 章的 LLM 接入 |
| `run_demo.py` | 命令行演示：问题 → 检索片段 → 答案 | — |
| `tests/test_rag.py` | 覆盖确定性/检索/答案/离线四类断言 | — |
| `conftest.py` | 让 `tests/` 能直接 import 项目模块 | — |

**为什么嵌入要用 `hashlib` 而不是内置 `hash()`？** Python 内置 `hash()` 对字符串默认带随机盐（`PYTHONHASHSEED`），跨进程结果会变；`hashlib.md5` 保证**逐字节可复现**——这是"同句同向量"的地基，也是单元测试能断言的前提。

---

## 4. 测试都验了什么（对应题面 a/b/c/d）

- **(a) 嵌入确定性**：同一句话 `encode` 两次、甚至换新实例，向量**逐位相同**；不同句向量不同、余弦 < 1。
- **(b) 检索正确**：对"怎么用 Ollama 部署大模型"这类明显相关的 query，正确文档稳居 **top-1**。
- **(c) 答案含事实**：`answer()` 返回的字符串**逐字包含**被检索命中的关键事实（RAG"用证据说话"）。
- **(d) 全程离线**：用 `monkeypatch` 把 `socket.socket` / `socket.create_connection` 全部替换成"一连接就报错"的守卫，证明离线 RAG 流程**完全不碰网络**依然跑通。

---

## 5. 如何换成"真实数据" / "真实 LLM"（落地到 LLM）

这一节是本项目的重点——离线版是脚手架，下面把它逐步升级成生产形态。

### 5.1 换成真实数据（你的 PDF / 文档）

`knowledge.py` 里的 `FACTS` 就是"私有数据"。真实场景按第 18 章的四步做：

1. **加载**：用 `PyPDFLoader`（LangChain）或 `pypdf` 读 PDF。
2. **切块**：`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`，优先在句子/换行等自然边界切，重叠 200 字符避免切断语义。
3. **嵌入**：对每个 chunk 调 `embedder.encode(chunk)`。
4. **入库**：`store.add(chunk_id, chunk_text, vec)`。

代码几乎不用改——把 `build_store()` 里遍历 `get_facts()` 换成"遍历你的 chunk 列表"即可。**铁律：建库和查询必须用同一个 `embedder`**，否则两套向量空间不可比，检索全是噪声却不报错。

### 5.2 换成真实句向量（更好的检索质量）

离线词袋只看"词面重叠"，换成语义嵌入能理解同义改写：

```python
from embed import get_embedder
embedder = get_embedder(prefer_real=True)   # 装了 sentence-transformers 就用它，否则自动回退
store, embedder = build_store(embedder=embedder)
```

`prefer_real=True` 会尝试 `sentence-transformers`（如 `all-MiniLM-L6-v2`）；缺库或离线则**静默回退**到离线确定性嵌入，测试照样绿。

### 5.3 换成 Ollama 本地生成（第 17 章）

第 17 章讲的就是用 Ollama 在本地跑开源大模型（默认端口 `11434`）。步骤：

```bash
# 装并拉一个小模型（一次性）
ollama pull llama3.1        # 或 gemma2:2b 等
ollama serve                # 起服务（通常安装后自动常驻）
pip install requests
```

然后把生成后端切成 `ollama`：

```python
from rag import answer
print(answer("什么是 RAG？", k=3, backend="ollama"))
# 或命令行： python run_demo.py ollama
```

`rag.generate_with_ollama()` 会把命中片段拼成 `context`，按第 18 章的对话结构组成 `messages`（一条 system 强约束"只依据 context 回答"、一条 user 带 `Context + Question`），`stream=False` POST 给 `http://localhost:11434/api/chat`。**连不上/未装 requests 会自动回退离线**，所以就算没装 Ollama，`backend="ollama"` 也不会崩。

> ⚠️ 上下文窗口陷阱（第 18 章原书强调）：`k × chunk_size` 必须留在模型上下文窗口内，还要给 system prompt 和答案留空间。小模型（如 Gemma 2B 仅 2k token）尤其容易爆窗。

### 5.4 换成微调后的 HF 模型（第 15 / 16 章）

第 16 章讲用自定义数据微调 / 提示微调 LLM。微调完你会得到一个本地模型目录，直接喂给 `generate_with_hf()`：

```python
from rag import build_store, retrieve, _build_context, generate_with_hf
store, emb = build_store()
q = "什么是 RAG？"
hits = retrieve(q, k=3, store=store, embedder=emb)
ctx  = _build_context(hits)
print(generate_with_hf(q, ctx, model_name="./my-finetuned-model"))  # 换成你微调后的目录
```

`generate_with_hf()` 用第 15 章的 `transformers.pipeline("text-generation", model=...)` 一行加载模型；缺库/缺权重/离线时返回 `None` 自动回退离线。把 `model_name` 从 `distilgpt2` 换成你微调产物的路径即可"落地到自己的 LLM"。

> 默认路径**不会**触发下载：只有你显式 `backend="hf"` 或直接调 `generate_with_hf` 才会尝试加载模型，因此 `pytest` 始终离线、秒级。

---

## 6. 生产化再进一步（延伸）

- **检索优化**：top-k 相似度容易召回重复片段，换 **MMR（最大边际相关）** 让 k 个片段各带新信息（第 18 章）。
- **真向量库**：数据量大了换 **Chroma / FAISS**；嵌入 L2 归一化后，余弦≡内积，FAISS 可用内积索引加速。
- **引用溯源**：给每个 chunk 存来源元数据（页码/起始位置），答案里带出处，就是 citation。
- **切块按 token 计**：`length_function` 用真正的 tokenizer（如 `tiktoken`）而非字符数，避免悄悄超上下文窗口。

---

## 7. 环境

Windows + Python 3.13 + CPU torch 2.12 + numpy + scikit-learn + transformers（已装）。`datasets` / `diffusers` **未安装**，本项目不依赖它们。默认离线路径只用 `numpy`。
