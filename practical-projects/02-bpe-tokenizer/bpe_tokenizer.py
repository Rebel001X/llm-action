"""
从零实现 Byte-level BPE 分词器 (Byte-Pair Encoding Tokenizer from scratch)
====================================================================

这是什么
--------
本文件用纯 Python(仅依赖 numpy 做统计可视化,核心算法零第三方库)实现了一个
**字节级 BPE(Byte-level Byte-Pair Encoding)分词器**,也就是 GPT-2 / GPT-3 /
LLaMA 等大模型所用分词方案的"教学最小版"。它完整覆盖了一个分词器的三件事:

  1. train  —— 在语料上"学词表":统计相邻 token 对(pair)的频率,反复把
              出现次数最高的那一对合并成一个新 token,直到达到目标词表大小。
              这就是 BPE 的核心:用数据驱动地把高频字节组合"压"成一个符号。
  2. encode —— 把任意文本(先转成 UTF-8 字节序列)按照学到的 merges 规则
              逐步合并,得到一串整数 token id。
  3. decode —— 把 token id 还原回字节,再解码回字符串。

为什么是"字节级(byte-level)"
-----------------------------
我们不在"字符"上做 BPE,而是先把字符串编码成 UTF-8 字节(0..255)。这样:
  - 初始词表天然只有 256 个,且**永远不会有未登录字符(OOV)**:任何 Unicode
    文本(中文、emoji、生僻符号)都能被表示为字节,因此 round-trip 永远无损。
  - 这正是 GPT-2 论文采用 byte-level BPE 的关键原因。

可量化的"成功信号"(运行后会打印)
--------------------------------
  - 学到的前若干个 merge 规则(高频字节对被合并成新符号)。
  - 压缩率 = token 数 / 原始字节数:训练后这个比值应明显 < 1,且词表越大压得越狠。
  - round-trip 校验:decode(encode(text)) == text 必须为 True(无损)。

这是教学 toy 实现,CPU 几十秒内跑完。与生产实现(tiktoken / HuggingFace
tokenizers)的差异见文末 README 的"局限"小节。

对应 llm-action 文档:../../llm-data-engineering
社区参考:karpathy/minbpe、GPT-2 (Radford et al., 2019)
"""

import os
import random
from collections import Counter

import numpy as np

# 设随机种子,保证(本 demo 中任何随机环节)可复现
random.seed(42)
np.random.seed(42)


# ======================================================================
# 1) 统计相邻 pair 频率
#    原理:BPE 每一步都要找"出现最多的相邻符号对"。这里统计的是在所有词里
#    相邻两个 token 一起出现的总次数(带词频加权)。
# ======================================================================
def get_pair_counts(word_freqs):
    """统计语料中所有相邻 token 对的加权频率。

    word_freqs: dict[tuple(int,...), int]
        key 是一个"词"被切成的 token 序列(初始为字节序列),value 是该词的出现次数。
    返回: Counter[(int,int) -> int],每个相邻 pair 的总频率。
    """
    pair_counts = Counter()
    for symbols, freq in word_freqs.items():
        # zip(symbols, symbols[1:]) 取出所有相邻对 (a,b)
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += freq  # 用词频加权:这个词出现 freq 次,pair 也算 freq 次
    return pair_counts


# ======================================================================
# 2) 在一个词(token 序列)中,把指定 pair 合并成新 token id
#    原理:这是"应用一条 merge 规则"的最小操作。
# ======================================================================
def merge_pair_in_word(symbols, pair, new_id):
    """把序列 symbols 中所有相邻出现的 pair=(a,b) 替换成单个 new_id。"""
    a, b = pair
    merged = []
    i = 0
    while i < len(symbols):
        # 命中相邻 (a,b) 就合并成 new_id,跳过两个位置
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            merged.append(new_id)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


# ======================================================================
# BPE 分词器主体
# ======================================================================
class BPETokenizer:
    """最小可用的 byte-level BPE 分词器。"""

    def __init__(self):
        # merges: dict[(int,int) -> int]
        #   学到的合并规则,key=被合并的 pair,value=合并后的新 token id。
        #   dict 的"插入顺序"= 学习顺序 = encode 时应用规则的优先级顺序(很关键)。
        self.merges = {}
        # vocab: dict[int -> bytes]
        #   每个 token id 对应的字节串。0..255 是原始单字节;>=256 是合并产物。
        self.vocab = {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------
    # 训练:学 merges 词表
    # ------------------------------------------------------------------
    def train(self, corpus, vocab_size, verbose=False):
        """在 corpus(字符串列表)上训练 BPE。

        vocab_size: 目标词表大小,必须 > 256(因为前 256 个是字节)。
        训练过程 = 反复执行 (vocab_size - 256) 次"合并最高频 pair"。
        """
        assert vocab_size > 256, "vocab_size must be > 256 (256 byte base tokens)"
        num_merges = vocab_size - 256

        # 把语料按空白切成"词",每个词转成 UTF-8 字节元组,并统计词频。
        # 注:真实 GPT-2 用正则做更细的预切分(pre-tokenization),这里用空白做 toy 版。
        word_freqs = Counter()
        for line in corpus:
            for word in line.split():
                # 保留前导空格信息:用 "Ġ" 风格的话会更像 GPT-2,这里 toy 版直接用字节。
                word_bytes = tuple(word.encode("utf-8"))
                word_freqs[word_bytes] += 1

        # 反复合并:每轮找当前最高频 pair,记为一条 merge 规则,并应用到所有词上。
        for step in range(num_merges):
            pair_counts = get_pair_counts(word_freqs)
            if not pair_counts:
                break  # 没有可合并的 pair 了(语料太小)
            # argmax:取频率最高的 pair。这一步对应 BPE"贪心选最高频对"的核心思想。
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break  # 再合并也只是合并只出现 1 次的对,意义不大,提前停

            new_id = 256 + step  # 新 token id 紧接在字节之后递增分配
            self.merges[best_pair] = new_id
            # 新 token 的字节 = 两个旧 token 字节的拼接(这样 decode 才能无损还原)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # 把这条规则应用到所有词,更新 word_freqs(下一轮在新序列上继续统计)
            word_freqs = {
                merge_pair_in_word(symbols, best_pair, new_id): freq
                for symbols, freq in word_freqs.items()
            }

            if verbose:
                a = self.vocab[best_pair[0]].decode("utf-8", errors="replace")
                b = self.vocab[best_pair[1]].decode("utf-8", errors="replace")
                merged = self.vocab[new_id].decode("utf-8", errors="replace")
                print(
                    "  merge {:>3}: ({!r:>6}, {!r:>6}) -> id {:>3}  {!r:>8}  (count={})".format(
                        step, a, b, new_id, merged, best_count
                    )
                )

    # ------------------------------------------------------------------
    # 编码:文本 -> token id 列表
    # ------------------------------------------------------------------
    def encode(self, text):
        """把字符串编码为 token id 列表。

        原理:先转成字节序列,然后按"学习顺序"反复应用 merges。每一步都找当前序列里
        优先级最高(= 最早学到 = id 最小)的可合并 pair 并合并,直到无规则可用。
        """
        ids = list(text.encode("utf-8"))  # 起点:原始字节
        while len(ids) >= 2:
            # 找出当前序列中所有相邻 pair
            pairs = set(zip(ids, ids[1:]))
            # 在可合并的 pair 里,选 merges 中"最先学到"(new_id 最小)的那个。
            # 用 self.merges.get(p, inf) 做 key:越早学的 new_id 越小,优先级越高。
            candidate = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if candidate not in self.merges:
                break  # 没有任何可应用的规则了,结束
            ids = list(merge_pair_in_word(ids, candidate, self.merges[candidate]))
        return ids

    # ------------------------------------------------------------------
    # 解码:token id 列表 -> 文本
    # ------------------------------------------------------------------
    def decode(self, ids):
        """把 token id 列表还原为字符串(byte-level,因此无损)。"""
        # 每个 id 查 vocab 得到字节串,拼接后用 UTF-8 解码。
        token_bytes = b"".join(self.vocab[i] for i in ids)
        return token_bytes.decode("utf-8", errors="replace")


# ======================================================================
# 内置小语料(toy):重复出现的高频字节组合便于 BPE 学到有意义的 merge。
# 用英文以保证 print 安全(中文只放注释/README)。
# ======================================================================
TINY_CORPUS = [
    "low low low low low",
    "lower lower lowest",
    "newest newest newest widest widest",
    "the lowest of the newest is the widest",
    "low newer lower newest low newer",
    "tokenization learns to merge frequent byte pairs",
    "tokenization tokenization tokenization is fun",
    "byte pair encoding encoding encoding",
    "the the the the of of of and and and",
    "a quick brown fox the lazy dog the quick fox",
] * 5  # 重复几遍放大频率信号,让 toy 训练能学出清晰的 merge


def compression_ratio(tokenizer, texts):
    """压缩率 = token 总数 / 原始字节总数。越小说明压得越好。"""
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    return total_tokens / total_bytes, total_tokens, total_bytes


def try_plot(vocab_sizes, ratios, out_path):
    """画"词表大小 vs 压缩率"曲线;matplotlib 不可用就退化为文本打印,不崩。"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无界面后端,直接存 png
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(vocab_sizes, ratios, marker="o")
        plt.xlabel("vocab size")
        plt.ylabel("compression ratio (tokens / bytes)")
        plt.title("BPE: larger vocab -> better compression")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        return True
    except Exception as e:  # noqa: BLE001 (教学代码:任何画图问题都不该让 demo 崩)
        print("[plot skipped] matplotlib unavailable: {}".format(e))
        return False


def main():
    print("=" * 64)
    print("Byte-level BPE Tokenizer - from scratch (toy demo)")
    print("=" * 64)

    n_words = sum(len(line.split()) for line in TINY_CORPUS)
    n_bytes = sum(len(line.encode("utf-8")) for line in TINY_CORPUS)
    print("corpus: {} lines, {} words, {} bytes".format(len(TINY_CORPUS), n_words, n_bytes))
    print()

    # ---- 训练一个词表,并打印学到的前若干 merge(可量化信号 1)----
    target_vocab = 300  # 256 字节 + 44 个学到的合并 token
    print("[train] target vocab_size = {}  (= 256 bytes + {} merges)".format(
        target_vocab, target_vocab - 256))
    print("learned merges (highest-frequency byte pairs first):")
    tok = BPETokenizer()
    tok.train(TINY_CORPUS, vocab_size=target_vocab, verbose=True)
    print("[train] done. learned {} merges, final vocab = {}".format(
        len(tok.merges), len(tok.vocab)))
    print()

    # ---- round-trip 无损校验(可量化信号 2:必须全 True)----
    print("[round-trip] decode(encode(x)) == x  ?")
    samples = [
        "the lowest newest widest",
        "tokenization is fun",
        "a quick brown fox",
        "unseen words like hippopotamus!!!",  # 训练没见过 -> 字节级仍无损
        "中文也能无损 round-trip 🚀",           # 非 ASCII -> byte-level 的关键优势
    ]
    all_ok = True
    for s in samples:
        ids = tok.encode(s)
        back = tok.decode(ids)
        ok = (back == s)
        all_ok = all_ok and ok
        # 只 print ASCII 安全字段:样本里的非 ASCII 不直接打印,改打印长度/是否相等
        ascii_preview = s.encode("ascii", "replace").decode("ascii")
        print("  ok={!s:>5}  bytes={:>3} -> tokens={:>3}   {!r}".format(
            ok, len(s.encode("utf-8")), len(ids), ascii_preview))
    print("[round-trip] ALL LOSSLESS = {}".format(all_ok))
    print()

    # ---- 压缩率随词表增大而下降(可量化信号 3)----
    print("[compression] ratio = tokens / bytes  (lower is better)")
    vocab_sizes = [256, 280, 300, 350, 400]
    ratios = []
    # vocab_size=256 表示不学任何 merge(纯字节),压缩率应 = 1.0 作为 baseline
    for vs in vocab_sizes:
        t = BPETokenizer()
        if vs > 256:
            t.train(TINY_CORPUS, vocab_size=vs)
        ratio, n_tok, n_byt = compression_ratio(t, TINY_CORPUS)
        ratios.append(ratio)
        print("  vocab={:>4}  tokens={:>5}  bytes={:>5}  ratio={:.4f}".format(
            vs, n_tok, n_byt, ratio))

    base, best = ratios[0], ratios[-1]
    speedup = base / best  # 序列变短的倍数 = 等价的"上下文/算力"节省
    print("[compression] baseline(no-merge) ratio = {:.4f}".format(base))
    print("[compression] best ratio = {:.4f}  -> sequence shrunk {:.2f}x".format(best, speedup))
    print()

    # ---- 画图(可选,失败不崩)----
    here = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(here, "compression_curve.png")
    if try_plot(vocab_sizes, ratios, png):
        print("[plot] saved compression curve -> {}".format(os.path.basename(png)))
    print()

    # ---- 总结性"成功"判定 ----
    success = all_ok and (best < 1.0) and (speedup > 1.0)
    print("=" * 64)
    print("RESULT: round_trip_lossless={}  compression_ratio={:.4f}  shrink={:.2f}x".format(
        all_ok, best, speedup))
    print("SUCCESS" if success else "FAILED")
    print("=" * 64)


if __name__ == "__main__":
    main()
