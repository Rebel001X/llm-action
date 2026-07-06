# 📦 交付分支 · 《AI and ML for Coders in PyTorch》中文逐章精讲 + 实战合集

本分支是一个**自包含交付包**，内容位于 [`book-ai-ml-for-coders-pytorch/`](book-ai-ml-for-coders-pytorch/)。

- 原书：**AI and ML for Coders in PyTorch — A Coder's Guide to Generative AI and Machine Learning**（Laurence Moroney 著，O'Reilly）
- **21 篇逐章精讲**（`book-guide/`，约 1.1 万行）：20 章原书精讲 + 自撰第 21 章「从本书基础到 LLM 落地实战（合流篇）」
- **6 个 CPU 可跑实战项目**（`projects/`，共 56 个 `pytest` 全绿）：视觉 DNN+CNN / NLP 嵌入 / LSTM 文本生成 / 时间序列预测 / Flask 推理服务 / 迷你 RAG（可插真实 LLM）

👉 从 [`book-ai-ml-for-coders-pytorch/README.md`](book-ai-ml-for-coders-pytorch/README.md) 开始阅读。

```bash
# 跑任意一个项目
cd book-ai-ml-for-coders-pytorch/projects/01_vision_pytorch_lab
pip install -r requirements.txt
python -m pytest -q
python run_demo.py
```

> 说明：本分支为独立交付分支（无上游历史），便于单独 review / 合并进主库对应目录。
