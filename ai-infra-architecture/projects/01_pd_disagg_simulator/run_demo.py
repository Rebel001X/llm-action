"""
run_demo.py —— 跑一个带 prefill 突发的工作负载,对比"合置 vs 分离"的 decode 体验。
运行:python run_demo.py
"""
import random
from pd_sim import Request, simulate_colocated, simulate_disaggregated


def make_workload(n=40, seed=0):
    rnd = random.Random(seed)
    reqs = []
    for i in range(n):
        # 混合负载:大多数是短 prompt 长输出(聊天),少数是长 prompt(文档/RAG)会突发占算力
        if rnd.random() < 0.3:
            prompt, out = rnd.randint(400, 900), rnd.randint(4, 12)     # 预填充型
        else:
            prompt, out = rnd.randint(8, 40), rnd.randint(30, 80)       # 解码型
        reqs.append(Request(i, arrival=rnd.uniform(0, 60), prompt_len=prompt, output_len=out))
    return reqs


def row(name, res):
    print(f"{name:<14}{res['mean_ttft']:>10.1f}{res['p99_tpot']:>12.2f}"
          f"{res['max_decode_gap']:>14.1f}{res['throughput_tok_per_s']:>16.1f}")


def main():
    wl = make_workload()
    print("=" * 70)
    print("PD 分离 vs 合置(混合负载:70% 解码型 + 30% 预填充突发)")
    print("=" * 70)
    print(f"{'方案':<14}{'平均TTFT(ms)':>10}{'p99 TPOT(ms)':>12}{'最大decode停顿':>14}{'吞吐(tok/s)':>16}")
    # 合置:4 个 slot 共用
    row("合置 4-slot", simulate_colocated([Request(r.rid, r.arrival, r.prompt_len, r.output_len) for r in wl], n_slots=4))
    # 分离:2 prefill + 2 decode(总卡数相同)
    row("分离 2P+2D", simulate_disaggregated([Request(r.rid, r.arrival, r.prompt_len, r.output_len) for r in wl], n_prefill=2, n_decode=2))
    print("\n解读:同样 4 张卡,分离把 prefill 突发与 decode 隔离 → decode 停顿更小、p99 TPOT 更稳。")
    print("[OK] demo 结束。")


if __name__ == "__main__":
    main()
