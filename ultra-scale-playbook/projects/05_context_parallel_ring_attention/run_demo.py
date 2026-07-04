"""
run_demo.py —— 真·多进程 Ring Attention(gloo/CPU)。
运行:python run_demo.py

序列沿长度维切到 num_ranks 个 rank,每个 rank 持有自己那段 Q。
K/V 块在**环 ring** 上逐跳传递(isend/irecv):走完一圈,每个 rank 就"见过"全部 K/V,
用在线 softmax 累加出自己那段的输出——全程不物化完整 S×S 注意力。
每个 rank 用完整 K/V(同种子可本地复算)校验自己的输出 == 标准全注意力对应块。
真实场景:序列极长时,K/V 永远只在环上一块一块过,显存 O(S/P)。
"""
import os
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
S = 24          # 序列长度(可被 WORLD 整除)
D = 8


def full_attention(Q, K, V):
    S_ = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
    return torch.softmax(S_, dim=-1) @ V


def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29551")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_default_dtype(torch.float64)

    g = torch.Generator().manual_seed(0)
    Q = torch.randn(S, D, generator=g)
    K = torch.randn(S, D, generator=g)
    V = torch.randn(S, D, generator=g)
    b = S // world
    Qr = Q[rank * b:(rank + 1) * b]                 # 本 rank 的 query 段

    kv = torch.stack([K[rank * b:(rank + 1) * b], V[rank * b:(rank + 1) * b]])  # (2,b,D) 自己的 K/V

    m = torch.full((b,), float("-inf")); l = torch.zeros(b); O = torch.zeros(b, D)
    for step in range(world):
        Kb, Vb = kv[0], kv[1]
        Sblk = Qr @ Kb.transpose(-1, -2) / math.sqrt(D)
        block_max = Sblk.max(dim=-1).values
        m_new = torch.maximum(m, block_max)
        alpha = torch.exp(m - m_new)
        p = torch.exp(Sblk - m_new.unsqueeze(-1))
        l = alpha * l + p.sum(-1)
        O = alpha.unsqueeze(-1) * O + p @ Vb
        m = m_new
        if step < world - 1:                        # 环上把 K/V 传给下一个 rank
            recv = torch.empty_like(kv)
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, kv.contiguous(), (rank + 1) % world),
                dist.P2POp(dist.irecv, recv, (rank - 1) % world),
            ])
            for r in reqs:
                r.wait()
            kv = recv
    out = O / l.unsqueeze(-1)

    ref = full_attention(Q, K, V)[rank * b:(rank + 1) * b]
    dev = (out - ref).abs().max()
    dist.all_reduce(dev, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"[Ring-Attn] world={world} 序列={S}  ring 输出 vs 全注意力 最大偏差 = {dev.item():.2e} (应≈0)")
        print("[OK] gloo 多进程 Ring Attention demo 结束(K/V 环上逐跳,显存 O(S/P))。")
    dist.destroy_process_group()


def main():
    print(f"启动 {WORLD} 进程(gloo/CPU)做 Ring Attention,序列 {S} 切成 {WORLD} 段 ...")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
