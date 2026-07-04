"""
run_demo.py —— 真·多进程张量并行 MLP(gloo/CPU),复现 Megatron 的"列并行→GELU→行并行"。
运行:python run_demo.py

每个 rank 只持有 W1、W2 的一个分片:
  · 列并行:h_r = GELU(x @ W1_r^T)        —— 各 rank 得到 h 的一段(无需通信)
  · 行并行:y_partial_r = h_r @ W2_r^T     —— 各 rank 得部分和
  · all_reduce(SUM) 得完整 y             —— 每个 MLP 前向只 1 次 all-reduce
最后 rank0 用完整权重算参考值,校验并行结果逐元素一致。
真实多卡:backend 换 nccl、device 换 cuda,并把 W1/W2 换成真正分片存储的参数。
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

WORLD = 4
H = 16


def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_default_dtype(torch.float64)

    # 所有 rank 用同一份"全局权重"种子生成,再各取自己的分片(演示用;真实场景是分片存储)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(5, H, generator=g)
    W1 = torch.randn(4 * H, H, generator=g)      # (4h, h)
    W2 = torch.randn(H, 4 * H, generator=g)      # (h, 4h)

    W1_r = torch.chunk(W1, world, dim=0)[rank]   # 列并行:按 out 切
    W2_r = torch.chunk(W2, world, dim=1)[rank]   # 行并行:按 in 切

    h_r = F.gelu(F.linear(x, W1_r))              # (5, 4h/p)
    y_partial = F.linear(h_r, W2_r)              # (5, h) 部分和
    y = y_partial.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)     # → 完整 y

    if rank == 0:
        ref = F.linear(F.gelu(F.linear(x, W1)), W2)   # 单卡参考
        max_dev = (y - ref).abs().max().item()
        print(f"[TP-MLP] world={world}  并行结果 vs 单卡参考 最大偏差 = {max_dev:.2e}  (应≈0)")
        print("[OK] gloo 多进程张量并行 MLP demo 结束(每层前向仅 1 次 all-reduce)。")
    dist.destroy_process_group()


def main():
    print(f"启动 {WORLD} 个进程(gloo/CPU)做张量并行 MLP ...")
    mp.spawn(worker, args=(WORLD,), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
