# ✍️ 项目 03 · LSTM 文本生成（对应原书第 7–8 章）

用一个**词级 LSTM** 学一小批结构化短句，然后**自回归地生成文本**——这正是 LLM 解码的玩具版：
预测下一个词 → 把它接回输入 → 再预测，滚雪球。

## 🎯 它做了什么
1. `corpus.py`：确定性地铺出 64 句结构一致的良性短句（颜色+动物+动作+地点），建词级词表（0=`<pad>`，1=`<unk>`）。
2. `dataset.py`：定长滑动窗口切成 `(输入 window 个词, 下一个词)` 监督样本。
3. `model.py`：`nn.Embedding → nn.LSTM → nn.Linear`，输出每个词的 logits。
4. `engine.py`：标准训练循环（Adam + CrossEntropyLoss）。
5. `generate.py`：自回归生成。`temperature=0` 走贪心（**确定性、可复现**）；`temperature>0` 走温度采样。
6. `run_demo.py`：训练后从几个种子词各续写一句。

## 🚀 怎么跑
```bash
pip install -r requirements.txt
python -m pytest -q      # 4 项测试全绿
python run_demo.py       # 看训练损失下降 + 生成结果
```

## 🧠 对应书里的概念
- **windowing / n-gram 输入序列**（第 8 章）→ `dataset.py` 的滑动窗口。
- **Embedding + LSTM 建模序列**（第 7 章）→ `model.py`。
- **复合预测滚雪球生成**（第 8 章）→ `generate.py` 的自回归循环。
- **temperature 采样** → 控制生成的确定性 vs 多样性，和 LLM 解码里的 `temperature` 完全同源。

## 🔗 如何升级成"真实 LLM"
- 把**词级**换成 **BPE 子词**分词（如 `tokenizers` 库），词表更小、能处理未登录词。
- 把 **LSTM 换成 Transformer Decoder**（`nn.TransformerDecoderLayer` 或直接看 [nanoGPT](https://github.com/karpathy/nanoGPT)），就是现代 LLM 的结构。
- 把生成循环加上 **KV-cache** 就是推理加速的起点。
- 想直接用现成 LLM 生成，见第 16 章（微调）与第 17 章（Ollama 本地部署）。

> ⚠️ 说明：这里的语料是**刻意规则化**的，目的是让小模型几秒收敛、贪心生成可复现、`pytest` 稳定。真实文本生成请换成更大、更自然的语料与更强的模型。
