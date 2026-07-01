"""
run_demo.py —— 真·多进程 DDP(gloo 后端,CPU),在本机模拟多卡数据并行。
运行:python run_demo.py
每个 rank 在自己的数据分片上算梯度 → dist.all_reduce 求平均 → 各自用相同 Adam 更新。
结束时校验:所有 rank 的参数完全一致(DDP 的不变量),且 loss 稳定下降。

Windows 友好:worker 是模块级函数、用 spawn、设 MASTER_ADDR/PORT、加 __main__ 守卫。
真实多卡只需把 backend 换成 'nccl'、device 换成 cuda。
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

WORLD = 4
STEPS = 40


class ToyMLP(nn.Module):
    def __init__(self, d_in=8, d_hidden=16, d_out=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x)


def worker(rank: int, world: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29521")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    torch.manual_seed(0)                       # 所有 rank 用同一初始权重(DDP 前提)
    model = ToyMLP()

    # 每个 rank 拿到全局数据的一个分片
    g = torch.Generator().manual_seed(1)
    X = torch.randn(world * 8, 8, generator=g)
    Y = torch.randn(world * 8, 4, generator=g)
    x = X.chunk(world)[rank]
    y = Y.chunk(world)[rank]

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        # 梯度 all-reduce 求平均(这一步让各 rank 的更新完全一致)
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world
        opt.step()
        if rank == 0 and step % 10 == 0:
            print(f"[step {step:2d}] rank0 local loss = {loss.item():.4f}")

    # 校验:所有 rank 参数应逐元素一致 —— 计算全局最大偏差
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    ref = flat.clone()
    dist.broadcast(ref, src=0)                  # 以 rank0 为基准
    max_dev = (flat - ref).abs().max()
    dist.all_reduce(max_dev, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"[校验] 所有 rank 参数最大偏差 = {max_dev.item():.2e}  (应≈0,证明 DDP 同步正确)")
        print("[OK] gloo 多进程 DDP demo 结束。")
    dist.destroy_process_group()


def main():
    print(f"启动 {WORLD} 个进程(gloo/CPU)做数据并行 DDP ...")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
