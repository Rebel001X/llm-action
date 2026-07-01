"""
run_demo.py —— 真·多进程流水线并行(gloo/CPU):每个 rank 持有一个 stage,
用 dist.send/recv 在 stage 间逐段传递激活(这就是 PP 的"点对点通信")。
运行:python run_demo.py

演示前向流水:rank0→rank1→...→rank(p-1)。最后一个 rank 用完整模型(各 rank 权重
用同种子生成,故它能本地复算全模型)校验流水线输出 == 单进程整模型输出。
反向的数值正确性已在 test_pipeline.py 用单进程累加梯度证明。
真实多卡:backend 换 nccl、device 换 cuda。
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

WORLD = 4          # 4 个 stage
DIM = 8
BATCH = 6
LAYERS_PER_STAGE = 2


def build_all_stages(num_stages, dim, seed=0):
    torch.manual_seed(seed)
    stages = []
    for _ in range(num_stages):
        blocks = []
        for _ in range(LAYERS_PER_STAGE):
            blocks += [nn.Linear(dim, dim), nn.Tanh()]
        stages.append(nn.Sequential(*blocks))
    return stages


def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29541")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_default_dtype(torch.float64)

    stages = build_all_stages(world, DIM, seed=0)   # 每个 rank 都有全部权重(同种子)
    my_stage = stages[rank]

    g = torch.Generator().manual_seed(1)
    x = torch.randn(BATCH, DIM, generator=g)         # 所有 rank 用同一份输入(仅 rank0 真正用)

    # ---- 前向流水:激活逐段 send/recv ----
    if rank == 0:
        a = my_stage(x)
        dist.send(a.contiguous(), dst=1)
    else:
        buf = torch.empty(BATCH, DIM)
        dist.recv(buf, src=rank - 1)
        a = my_stage(buf)
        if rank < world - 1:
            dist.send(a.contiguous(), dst=rank + 1)

    if rank == world - 1:
        # 最后一个 stage:a 就是流水线输出;本地用完整模型复算做参考
        ref = x
        for s in stages:
            ref = s(ref)
        max_dev = (a - ref).abs().max().item()
        print(f"[PP 前向] stage 数={world}  流水线输出 vs 单进程整模型 最大偏差 = {max_dev:.2e} (应≈0)")
        print("[OK] gloo 多进程流水线前向 demo 结束(激活经 send/recv 逐段传递)。")
        print("     反向梯度正确性见 test_pipeline.py(累加梯度==单进程整 batch)。")
    dist.destroy_process_group()


def main():
    print(f"启动 {WORLD} 个进程(gloo/CPU),每个持有一个 pipeline stage ...")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
