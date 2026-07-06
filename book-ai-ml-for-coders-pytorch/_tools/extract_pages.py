# -*- coding: utf-8 -*-
"""Extract cleaned text from a physical page range of the book PDF.
Usage: python _tools/extract_pages.py START END
Page numbers are the physical PDF pages shown in the TOC (1-indexed).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import fitz

PDF = r"C:\Users\jianm\Desktop\llm-action\AI and ML for Coders in PyTorch A Coders Guide to Generative AI and Machine Learning (Laurence Moroney) (z-library.sk, 1lib.sk, z-lib.sk).pdf"

def clean(t: str) -> str:
    # normalize the mangled smart-quotes / hyphenation the PDF encodes badly
    reps = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "ﬁ": "fi", "ﬂ": "fl",
        "‐": "-", "­": "",
    }
    for a, b in reps.items():
        t = t.replace(a, b)
    # the specific double-question-mark artifact for smart apostrophes
    t = t.replace("’", "'")
    t = t.replace("��", "'")
    t = t.replace("‑", "-")
    return t

def main():
    start, end = int(sys.argv[1]), int(sys.argv[2])
    doc = fitz.open(PDF)
    out = []
    for p in range(start - 1, min(end, doc.page_count)):
        out.append(doc[p].get_text())
    print(clean("".join(out)))

if __name__ == "__main__":
    main()
