# 02 · NLP 分词 + 嵌入情感分类（对应第 5–6 章）

用 PyTorch 复刻《AI and Machine Learning for Coders》第 5–6 章的完整链路：

```
原始文本  →  分词/建词表  →  转 id 序列 + padding  →  nn.Embedding  →  掩码均值池化  →  Linear → 情感概率
   ↑第5章「Introduction to NLP / Tokenization」        ↑第6章「Making Sentiment Programmable with Embeddings」
```

全程 **离线、CPU、秒级**：数据是确定性合成的正/负情感小语料，不下载任何真实数据集或模型，`pytest` 全绿。

---

## 3 分钟跑通

```bash
cd 02_nlp_embeddings_lab

# 1) 跑测试（应全绿）
python -m pytest -q

# 2) 看端到端演示：训练 + 对新句子打印情感概率
python run_demo.py

# 3) 额外导出词向量到 Embedding Projector 格式（vectors.tsv / metadata.tsv）
python run_demo.py --export
```

`run_demo.py` 会打印类似：

```
最终验证集准确率：1.000
=== 新句子情感预测（正面概率）===
  [正面 👍  p=0.97]  i love this amazing wonderful movie
  [负面 👎  p=0.03]  this film was terrible and boring
```

导出的 `vectors.tsv` / `metadata.tsv` 可直接上传到
<https://projector.tensorflow.org/> 可视化词向量（书里第 6 章的经典操作）。

---

## 每个文件干嘛

| 文件 | 作用 | 对应书里 |
|------|------|---------|
| `tokenizer.py` | `SimpleTokenizer`：小写化、去标点、按空格切词、建 vocab、`texts_to_sequences`、`pad_sequences`；含 `<pad>`(id 0) / `<oov>`(id 1) | 第 5 章 Tokenizer |
| `data.py` | 确定性合成的正/负情感语料（正例含 great/love/amazing…，负例含 terrible/hate/awful…），返回训练/验证 split | 第 5 章 语料准备 |
| `model.py` | `SentimentNet` = `nn.Embedding` + **掩码均值池化**（mask 掉 `<pad>`）+ `Linear` → 1 个 logit（配 `BCEWithLogitsLoss`） | 第 6 章 Embedding 分类 |
| `engine.py` | `SentimentClassifier`：`fit` / `evaluate` / `classify(text)` 返回正类概率 | 第 6 章 训练与推理 |
| `run_demo.py` | 训练后对新句子打印情感概率；`--export` 导出嵌入 tsv | 第 6 章 可视化嵌入 |
| `tests/test_nlp.py` | 4 组断言：分词往返&padding、模型前向形状、验证集准确率 >0.8、classify 极性正确 | — |

---

## 核心手法（书里的要点）

1. **特殊 token**：`<pad>` 补齐、`<oov>` 兜底未登录词——真实语料里必有没见过的词。
2. **padding**：把不等长句子补/截到同一长度，才能塞进一个张量做 batch 前向。
3. **掩码均值池化**：均值池化时**必须**用 `(id != PAD)` 掩码只对真实词求平均，否则一堆补零会把句向量拉向 0。`nn.Embedding(..., padding_idx=0)` 进一步保证 `<pad>` 向量恒为 0 且不吃梯度。
4. **BCEWithLogitsLoss**：模型输出原始 logit（不加 sigmoid），损失函数内部做数值稳定的 sigmoid+BCE；推理时再手动 `sigmoid` 得概率。

---

## 如何换成真实数据 / 真实 LLM

- **换真实数据集**：把 `data.load_sentiment_data()` 换成读你自己的 CSV / IMDB / Sarcasm 数据即可，只要返回 `(texts, labels)`。其余代码（分词、模型、训练）完全不用改。
  ```python
  import pandas as pd
  df = pd.read_csv("reviews.csv")            # 含 text / label 两列
  texts, labels = df["text"].tolist(), df["label"].tolist()
  ```
  若装了 `datasets`（本环境未装），也可 `load_dataset("imdb")` 后喂进来。

- **换更强的分词器**：把 `SimpleTokenizer` 换成 `transformers` 的子词分词器（本环境已装 `transformers`）：
  ```python
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("bert-base-uncased")  # 首次需联网下载
  enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
  ```
  然后把 `enc["input_ids"]` 喂给一个以 BERT 词表大小新建的 `nn.Embedding`，或直接接预训练模型。

- **换真实 LLM 做情感分类**：把 `SentimentNet` 换成预训练模型的分类头，例如
  `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")`
  （首次联网下载权重）。本项目默认走**离线合成分支**保证 `pytest` 绿；真实 LLM 作为可选升级路径，不影响测试。

- **加载预训练词向量**（GloVe/word2vec）：用外部向量初始化 `model.embedding.weight`，冻结或微调即可，对应书里"用预训练嵌入"的进阶。

---

## 环境

见 `requirements.txt`。核心只需 `torch` + `numpy`；`pytest` 用于测试。
本项目在 Windows + Python 3.13 + CPU torch 2.12 下验证通过。
