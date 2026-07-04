"""
run_demo.py —— 真·多进程:只用 P2P(isend/irecv)从零搭 ring AllReduce,并和官方
torch.distributed.all_reduce 对拍。运行:python run_demo.py

ring AllReduce 两阶段(每阶段 P-1 步,数据沿环流动):
  reduce-scatter:每步把收到的一段"加"到本地对应段
  all-gather:每步把已求和的一段"覆盖"传给下一环
每个 rank 收发数据量 2(P-1)/P·N,与卡数几乎无关 → 带宽最优。
真实多卡:backend 换 nccl(NCCL 内部正是类似 ring/tree 算法)。
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
N = 24            # 向量长度(可被 WORLD 整除)


def ring_allreduce(vec, rank, world):
    chunks = list(torch.chunk(vec.clone(), world))     # 本 rank 的 P 段
    nxt, prv = (rank + 1) % world, (rank - 1) % world

    def exchange(send_c, recv_c):
        recv_buf = torch.empty_like(chunks[recv_c])
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, chunks[send_c].contiguous(), nxt),
            dist.P2POp(dist.irecv, recv_buf, prv),
        ])
        for r in reqs:
            r.wait()
        return recv_buf

    # reduce-scatter
    for i in range(world - 1):
        send_c = (rank - i) % world
        recv_c = (rank - 1 - i) % world
        chunks[recv_c] = chunks[recv_c] + exchange(send_c, recv_c)
    # all-gather
    for i in range(world - 1):
        send_c = (rank + 1 - i) % world
        recv_c = (rank - i) % world
        chunks[recv_c] = exchange(send_c, recv_c)
    return torch.cat(chunks)


def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29561")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_default_dtype(torch.float64)

    g = torch.Generator().manual_seed(rank)            # 每个 rank 不同数据
    vec = torch.randn(N, generator=g)

    mine = ring_allreduce(vec, rank, world)            # 自己搭的 ring all-reduce
    ref = vec.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)         # 官方 all-reduce 作参考

    dev = (mine - ref).abs().max()
    dist.all_reduce(dev, op=dist.ReduceOp.MAX)
    if rank == 0:
        from collectives import allreduce_bytes_per_rank
        vol = allreduce_bytes_per_rank(N, world, 8)
        print(f"[Ring-AllReduce] world={world}  自实现 vs 官方 all_reduce 最大偏差 = {dev.item():.2e} (应≈0)")
        print(f"    每 rank 收发 ≈ {vol:.0f} 字节 = 2(P-1)/P·N·bytes(与卡数几乎无关)")
        print("[OK] gloo 从零 ring AllReduce demo 结束。")
    dist.destroy_process_group()


def main():
    print(f"启动 {WORLD} 进程(gloo/CPU),用 P2P 从零搭 ring AllReduce ...")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
