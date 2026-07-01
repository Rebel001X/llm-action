"""
run_demo.py —— 打印几个真实规模模型在不同并行配置下的显存账单。
运行:python run_demo.py
"""
from calc import PRESETS, param_count, bill, training_flops, GB


def fmt(g):
    return f"{g:8.2f} GB"


def show(name, cfg, batch, **cfgs):
    pc = param_count(cfg)
    print(f"\n=== {name}  (参数 {pc['total']/1e9:.2f} B, seq={cfg.seq_len}, batch={batch}) ===")
    print(f"{'配置':<46}{'模型状态':>12}{'激活':>12}{'单卡合计':>12}")
    for label, kw in cfgs.items():
        b = bill(cfg, batch=batch, **kw)
        print(f"{label:<46}{fmt(b['model_state_GB']):>12}{fmt(b['activation_GB']):>12}{fmt(b['total_GB']):>12}")


def main():
    print("=" * 82)
    print("Transformer 训练显存 & 算力计算器 —— 复刻《Ultra-Scale Playbook》第 2、9 章")
    print("=" * 82)

    show("Llama-7B", PRESETS["llama-7b"], batch=1,
         **{
             "单卡 fp16+Adam,无优化(放不下 80GB!)": dict(zero_stage=0, dp=1, recompute="none"),
             "+ 激活全重算":                          dict(zero_stage=0, dp=1, recompute="full"),
             "+ ZeRO-3 (dp=8) + 全重算":              dict(zero_stage=3, dp=8, recompute="full"),
             "+ ZeRO-3 (dp=8) + TP=2 + 选择性重算":   dict(zero_stage=3, dp=8, tp=2, recompute="selective"),
         })

    show("Llama-70B", PRESETS["llama-70b"], batch=1,
         **{
             "单卡(天文数字)":                        dict(zero_stage=0, dp=1, recompute="none"),
             "ZeRO-3(dp=64)+TP=8+PP=4+全重算":        dict(zero_stage=3, dp=64, tp=8, pp=4, recompute="full"),
         })

    # 算力:70B 训 1.4T token 需要多少 FLOPs、H100(~1e15 bf16 FLOP/s,MFU 0.4)要多久
    N = param_count(PRESETS["llama-70b"])["total"]
    D = 1.4e12
    flops = training_flops(N, D)
    gpu_hours = flops / (1e15 * 0.4) / 3600      # 单卡有效算力 0.4e15 FLOP/s
    print(f"\n=== 算力估算:Llama-70B 训 {D/1e12:.1f}T token ===")
    print(f"总 FLOPs ≈ {flops:.2e}")
    print(f"若单卡有效 4e14 FLOP/s(H100 @ MFU=0.4):≈ {gpu_hours:,.0f} GPU·小时"
          f" ≈ {gpu_hours/1024:,.0f} 张 H100 跑 {gpu_hours/1024/24:.1f} 天")
    print("\n[OK] demo 结束。")


if __name__ == "__main__":
    main()
