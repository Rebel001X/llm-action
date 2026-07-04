"""
run_demo.py —— 模拟一批 LLM 推理请求,对比三种 KV Cache 显存管理方案:
    (1) 朴素连续分配  naive contiguous  : 每序列按 max_seq_len 预留一整段连续显存
    (2) 分页(无共享) paged, no sharing : 固定 block + 页表,消除外部碎片
    (3) 分页 + 前缀共享 paged + prefix  : 相同系统提示的请求共享物理块 + COW

输出:
    · 终端打印三方案的「显存占用 / 利用率 / 碎片率 / 共享省块」对比表
    · 生成 kv_fragmentation.png(碎片与利用率对比图,2 子图)

运行:python run_demo.py
"""
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")                       # 无界面后端,纯出图
import matplotlib.pyplot as plt             # noqa: E402

from paged_kv_cache import PagedKVCacheManager  # noqa: E402

# 中文字体(按项目约定)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BLOCK_SIZE = 16                             # 每块 16 个 token 的 KV(vLLM 默认量级)
NUM_BLOCKS = 4096                           # 池子够大,专注比碎片而非 OOM


# ── 构造一批「聊天式」请求:共享一段系统提示 + 各自的用户输入/输出 ────────────
def make_workload(n=64, seed=7):
    rnd = random.Random(seed)
    # 一段较长的公共系统提示(所有请求前缀相同 → 前缀共享的用武之地)
    system_prompt = list(range(1, 49))     # 48 个 token,正好 3 个满块
    reqs = []
    for i in range(n):
        user_len = rnd.randint(4, 40)      # 用户输入长度不一
        out_len = rnd.randint(8, 120)      # 生成长度差异很大(碎片的根源)
        # token id 用负数区分「私有内容」,保证不同请求的非共享部分互不相同
        user = [-(i * 1000 + k) for k in range(user_len)]
        total_len = len(system_prompt) + user_len + out_len
        reqs.append({"prompt": system_prompt + user, "out_len": out_len,
                     "total_len": total_len})
    return system_prompt, reqs


# ── 用管理器真实地把整批请求「跑」一遍(prefill + decode) ──────────────────
def run_paged(reqs, sharing):
    mgr = PagedKVCacheManager(NUM_BLOCKS, BLOCK_SIZE, enable_prefix_sharing=sharing)
    for i, r in enumerate(reqs):
        mgr.add_sequence(i, r["prompt"])         # prefill
        for t in range(r["out_len"]):            # decode:逐 token 追加
            mgr.append_token(i, 900000 + t)
    return mgr


def main():
    system_prompt, reqs = make_workload()
    n = len(reqs)
    total_tokens = sum(r["total_len"] for r in reqs)
    max_len = max(r["total_len"] for r in reqs)

    # (1) 朴素连续分配:每序列预留 max_len
    naive_slots = PagedKVCacheManager.naive_reserved_slots(n, max_len)
    naive_util = total_tokens / naive_slots

    # (2) 分页,无共享
    paged = run_paged(reqs, sharing=False)
    paged_slots = paged.num_used_blocks() * BLOCK_SIZE
    paged_util = paged.utilization()

    # (3) 分页 + 前缀共享
    shared = run_paged(reqs, sharing=True)
    shared_slots = shared.num_used_blocks() * BLOCK_SIZE

    print("=" * 74)
    print(f"分页 KV Cache 显存管理:{n} 条聊天请求(共享 {len(system_prompt)}-token 系统提示)")
    print(f"block_size={BLOCK_SIZE}, 池={NUM_BLOCKS} 块, 逻辑 token 总量={total_tokens}, "
          f"最长序列={max_len}")
    print("=" * 74)
    hdr = f"{'方案':<20}{'预留KV槽':>12}{'相对朴素':>10}{'利用率':>10}{'内部碎片率':>12}"
    print(hdr)
    print("-" * 74)
    print(f"{'朴素连续(预留max)':<20}{naive_slots:>12}{'1.00x':>10}"
          f"{naive_util*100:>9.1f}%{(1-naive_util)*100:>11.1f}%")
    print(f"{'分页(无共享)':<20}{paged_slots:>12}{naive_slots/paged_slots:>9.2f}x"
          f"{paged_util*100:>9.1f}%{paged.internal_fragmentation()*100:>11.1f}%")
    print(f"{'分页+前缀共享':<20}{shared_slots:>12}{naive_slots/shared_slots:>9.2f}x"
          f"{'—':>10}{'—':>12}")
    print("-" * 74)
    print(f"前缀共享额外省下物理块:{shared.blocks_saved_by_prefix} 块 "
          f"(= {shared.blocks_saved_by_prefix*BLOCK_SIZE} 槽);"
          f"COW 触发 {shared.cow_count} 次")
    print(f"结论:分页把利用率从 {naive_util*100:.0f}% 拉到 {paged_util*100:.0f}%;"
          f"前缀共享再把显存压到朴素的 1/{naive_slots/shared_slots:.1f}。")

    # ── 出图:左=预留槽对比(越低越好),右=利用率对比 ─────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    names = ["朴素连续\n(预留max)", "分页\n(无共享)", "分页+前缀共享"]
    slots = [naive_slots, paged_slots, shared_slots]
    colors = ["#C44E52", "#DD8452", "#55A868"]
    bars = ax1.bar(names, slots, color=colors)
    ax1.axhline(total_tokens, color="black", ls="--", lw=1)
    ax1.text(2.02, total_tokens, "  实际需要的\n  最少槽数", va="center",
             fontsize=9, color="black")
    ax1.set_ylabel("预留 KV 槽数(显存 ∝ 此值,越低越好)")
    ax1.set_title("显存占用:分页 + 前缀共享大幅降低", fontsize=12, weight="bold")
    for b, s in zip(bars, slots):
        ax1.text(b.get_x() + b.get_width() / 2, s, f"{s:,}",
                 ha="center", va="bottom", fontsize=9)

    utils = [naive_util * 100, paged_util * 100]
    ubars = ax2.bar(["朴素连续", "分页"], utils, color=["#C44E52", "#55A868"])
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("有效利用率 (%)  = 1 − 内部碎片率")
    ax2.set_title("利用率:朴素被最长序列拖成大碎片", fontsize=12, weight="bold")
    for b, u in zip(ubars, utils):
        ax2.text(b.get_x() + b.get_width() / 2, u, f"{u:.1f}%",
                 ha="center", va="bottom", fontsize=10, weight="bold")

    fig.tight_layout()
    fig.savefig("kv_fragmentation.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\n已生成 kv_fragmentation.png")
    print("[OK] demo 结束。")


if __name__ == "__main__":
    main()
