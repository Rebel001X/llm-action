# 项目 01 · PD 分离调度模拟器

> 对应 [`../../01_PD分离架构_Prefill_Decode_Disaggregation.md`](../../01_PD分离架构_Prefill_Decode_Disaggregation.md)。
>
> 用一个**离散时间 tick 仿真**,把"合置 vs 分离"的本质差异量化出来:**prefill 突发时,合置会阻塞 decode(TPOT 尖峰),分离不会**。纯 Python、零依赖、秒级可跑。

---

## 🎯 模型(简化但抓住本质)

- **一个 tick = 一个 decode 步**(`DECODE_STEP` ms)。
- **prefill 算力受限**:耗时 ∝ `prompt_len`,占用一个 slot 若干 tick。
- **decode 访存受限**:每 tick 让所有活跃 decode 请求各 +1 token(连续批处理),**但只有本 tick 有空闲 slot 时才能推进**。
- **合置 colocated**:prefill 和 decode 抢同一批 slot;slot 被 prefill 占满时 decode **停一拍**。
- **分离 disaggregated**:prefill 池与 decode 池独立;decode 永不被 prefill 阻塞;代价是 prefill→decode 的 **KV 传输开销** ∝ `prompt_len`。

## 📁 文件
| 文件 | 作用 |
|---|---|
| `pd_sim.py` | 模拟器核心:`simulate_colocated` / `simulate_disaggregated`,输出 TTFT/TPOT/停顿/吞吐 |
| `test_pd_sim.py` | 6 个 pytest:请求都完成、分离 decode 不阻塞、合置会停顿、分离尾延迟更优 |
| `run_demo.py` | 40 个混合请求(70% 解码型 + 30% 预填充突发),对比两种方案 |

## ▶️ 如何运行
```bash
python -m pytest -q      # 6 passed
python run_demo.py       # 打印合置 vs 分离的 TTFT/TPOT/停顿/吞吐对比
```

## 📊 典型输出解读
| 方案 | 平均TTFT | p99 TPOT | 最大decode停顿 | 吞吐 |
|---|---|---|---|---|
| 合置 4-slot | 低 | **高(~71ms)** | **大(~260ms)** | 高 |
| 分离 2P+2D | higher | **低(~1ms)** | **~1ms** | 取决于 P:D 配比 |

> 🔬 **本质**:分离把 decode 的**尾延迟 p99 TPOT 打平**(用户体验的"流畅度"),代价是 KV 传输 + 需要合理的 prefill:decode 卡数配比。这正是 DistServe/Mooncake 要按 SLO 分别配比两阶段资源的原因。**没有免费午餐**——demo 诚实地展示了这个权衡,而不是把分离吹成全面碾压。

## 💡 面试高频
- "PD 分离到底改善了什么?" → **decode 的尾延迟稳定性(p99 TPOT)**,消除 prefill 对 decode 的阻塞。
- "分离的代价?" → KV 传输开销 + 需要调 prefill:decode 配比(配错反而更差)。
- "怎么决定 P:D 配比?" → 按负载画像与 SLO:prefill 重(长 prompt/RAG)就多给 prefill 卡,反之多给 decode。

## ⚠️ 模型的简化(诚实声明)
- 未建模显存容量上限、PagedAttention 碎片、投机解码;decode 批大小对每步时间的影响做了"恒定"近似(实际 memory-bound 下近似成立,但有上限)。
- 目的是**教学与直觉**,不是精确性能预测。真实系统请看 vLLM/SGLang/DistServe 的 benchmark。

## 🔗 延伸
- 架构:[`../../01_PD分离架构...md`](../../01_PD分离架构_Prefill_Decode_Disaggregation.md)
- 瓶颈本质:[`../../02_GPU结构...md`](../../02_GPU结构_从SM到集群_全面本质.md) 的 roofline
- 仓库既有:`../../../llm-inference/PD分离.md`、`Mooncake.md`
