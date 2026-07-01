# 零基础(三)· Transformer 的显存与计算:参数 / 激活 / 优化器 / FLOPs 估算

> 会算账,才谈得上选并行。读完你能徒手估出"7B 模型训练要多少显存、多少算力"。
>
> 配套:`../projects/01_memory_flops_calculator`(把本文公式做成可跑的计算器 + 13 个测试)。

---

## 🧱 训练显存的四块

| 块 | 是什么 | 混合精度 + Adam 的字节数 |
|---|---|---|
| 参数 parameters | 权重(低精度副本) | 2 B/param |
| 梯度 gradients | 反向算出的梯度 | 2 B/param |
| 优化器状态 optimizer states | fp32 主参数 + Adam 的 m、v | 4 + 4 + 4 = 12 B/param |
| 激活 activations | 前向存给反向用的中间量 | 随 batch×seq 变化 |

**模型状态合计 = 16 B/param**。这就是那句"训练一个参数要 16 字节"的由来。

> 💡 所以 7B 模型光模型状态就 `7e9 × 16 ≈ 112 GB` → 单张 80GB 卡放不下 → 需要 ZeRO/TP/PP。

---

## 1) 参数量:`N ≈ 12 L h²`

一个标准 decoder 层(FFN 倍数 4):
- 注意力 Q/K/V/O 投影:`4h²`
- MLP 两层 `h→4h→h`:`8h²`
- 合计 `12h²`,乘层数 `L`,加嵌入 `V·h`。

$$N \approx 12\,L\,h^2 + V h$$

**纯 Python 算一下:**
```python
def param_count(L, h, V, ffn_mult=4):
    per_layer = 4*h*h + 2*ffn_mult*h*h     # 注意力 + MLP
    return L * per_layer + V * h           # + 嵌入

print(param_count(12, 768, 50257) / 1e6)   # GPT-2 small ≈ 124 (M) ✅
print(param_count(32, 4096, 32000) / 1e9)  # Llama-7B ≈ 6.6 (B) ✅
```

## 2) 激活显存:随 `s²` 爆炸

Megatron 的经典估计(单层,约,元素数):
$$A_{\text{layer}} \approx s\,b\,h\left(34 + \frac{5\,a\,s}{h}\right)$$
- `34sbh`:线性层/LayerNorm/dropout 的中间激活
- `5as²b`:**注意力分数矩阵**,随序列 `s²` 增长 → **长序列杀手**(要靠上下文并行 + 激活重算)

**激活重算 activation recomputation**(用算力换显存):
- `selective`:只重算注意力那部分(去掉 `s²` 项)
- `full`:每层只存输入 `2sbh`,反向时整层重算 → 激活显存暴降(demo 里 7B 从 194GB 降到 2GB)

## 3) FLOPs:`6ND`

$$\text{一次训练 FLOPs} \approx 6 N D$$
- `N` = 参数量,`D` = 训练 token 数
- 前向 `2ND`(每参数每 token 约 2 次乘加),反向约 2 倍 → 共 `6ND`

**MFU(Model FLOPs Utilization)** = 实测算力 / 峰值算力,是衡量训练效率的核心指标(H100 上跑到 0.4~0.5 就不错了)。

```python
def train_flops(N, D):          # N 参数量, D token 数
    return 6 * N * D

flops = train_flops(70e9, 1.4e12)      # 70B 训 1.4T token
print(f"{flops:.2e} FLOPs")            # ≈ 5.9e23
# H100 有效算力 ~4e14 FLOP/s(MFU 0.4):
print(flops / 4e14 / 3600 / 1024, "张 H100·天")   # ≈ 数百卡 × 十几天
```

---

## 🧮 把 7B 的账算全(混合精度 + Adam)

| 项 | 公式 | 数值(N=6.6e9) |
|---|---|---|
| 模型状态 | `16·N` | ≈ 106 GB |
| 激活(seq=4096,bs=1,无重算) | Megatron 公式 | ≈ 190+ GB(!) |
| 激活(全重算) | `2sbh·L` | ≈ 2 GB |

**结论**:不做任何优化,7B 训练要 ~300GB;全激活重算 + ZeRO-3(dp=8)后单卡能压到 ~15GB。这正是 `../projects/01_memory_flops_calculator` 的 demo 打印的。

## ⚠️ 常见坑
- 只算参数,忘了优化器状态(是参数的 6 倍字节)和激活。
- 以为 fp16 就省一半——梯度、优化器状态仍在,Adam 还多两份 fp32 动量。
- 长序列下激活常常比模型状态还大 → 必须重算 / 上下文并行。

## 📌 速查
- 参数 `≈12Lh²+Vh`;模型状态 `16 B/param`;训练算力 `6ND`;激活 `~sbh(34+5as/h)`。

## 🔗 延伸
- 计算器实战:`../projects/01_memory_flops_calculator`(13 个测试对拍这些公式)
- 理论:`../book-guide/01_单卡训练...md`、`../book-guide/08_寻找最优训练配置...md`
- 数量级直觉:`../appendix/D_LLM训练的典型规模与数字...md`
