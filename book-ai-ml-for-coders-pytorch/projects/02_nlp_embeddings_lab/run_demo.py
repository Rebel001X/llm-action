# -*- coding: utf-8 -*-
"""
run_demo.py —— 端到端演示

1) 生成合成语料 -> 分词 -> 训练 Embedding 情感分类器；
2) 打印验证集准确率；
3) 对几句新句子打印"正面概率"；
4) 可选：把学到的词向量导出成 Embedding Projector 需要的
   vectors.tsv / metadata.tsv（对应书里第 6 章可视化嵌入的部分）。

用法：
    python run_demo.py            # 训练 + 打印情感概率
    python run_demo.py --export   # 额外导出嵌入 tsv 到当前目录
"""

import argparse

from data import load_sentiment_data
from engine import train_classifier


# 一些"明显"的测试句子，展示模型的情感倾向
DEMO_SENTENCES = [
    "i love this amazing wonderful movie",
    "what a great and fantastic book",
    "this film was terrible and boring",
    "the worst awful horrible product ever",
    "it was so brilliant and enjoyable",
    "a dreadful disappointing pathetic story",
]


def export_embeddings(clf, vectors_path="vectors.tsv", metadata_path="metadata.tsv"):
    """
    把 nn.Embedding 权重导出为 TensorFlow Embedding Projector 格式：
        vectors.tsv  : 每行一个词向量，制表符分隔
        metadata.tsv : 每行一个对应的词（第一行是表头）
    上传到 https://projector.tensorflow.org/ 即可可视化。
    """
    weights = clf.model.embedding.weight.detach().cpu().numpy()
    index_word = clf.tokenizer.index_word

    with open(vectors_path, "w", encoding="utf-8") as fv, \
         open(metadata_path, "w", encoding="utf-8") as fm:
        fm.write("word\n")
        # 按 id 顺序导出，与 metadata 一一对应
        for idx in range(weights.shape[0]):
            word = index_word.get(idx, "<unk>")
            fm.write(word + "\n")
            fv.write("\t".join(f"{v:.6f}" for v in weights[idx]) + "\n")
    print(f"已导出词向量到 {vectors_path} / {metadata_path}（共 {weights.shape[0]} 个词）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="导出嵌入到 tsv")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    # 1) 数据
    train, val = load_sentiment_data(n_per_class=120, val_ratio=0.2, seed=0)
    print(f"训练样本 {len(train[0])} 条，验证样本 {len(val[0])} 条")

    # 2) 训练
    clf, history = train_classifier(train, val, epochs=args.epochs, verbose=True)
    print(f"词表大小：{clf.tokenizer.vocab_size}")
    print(f"最终验证集准确率：{history[-1][1]:.3f}")

    # 3) 对新句子做情感预测
    print("\n=== 新句子情感预测（正面概率）===")
    for sent in DEMO_SENTENCES:
        prob = clf.classify(sent)
        tag = "正面 POS" if prob >= 0.5 else "负面 NEG"
        print(f"  [{tag}  p={prob:.3f}]  {sent}")

    # 4) 可选导出嵌入
    if args.export:
        print()
        export_embeddings(clf)


if __name__ == "__main__":
    main()
