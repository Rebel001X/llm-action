# 11 · 连续批处理 + 分页 KV 缓存 调度模拟

> 用一个 **纯 Python 离散事件模拟器**(CPU 几秒跑通,无需 GPU / 数据集 / 联网),
> 把 vLLM 的两大核心思想 —— **连续批处理 (Continuous / iteration-level Batching)** 与
> **分页 KV 缓存 (PagedAttention)** —— 拆开,**定量** 对比它相对传统静态批处理的收益。

---

## ① 演示什么原理

LLM 推理服务的两大瓶颈:**怎么把多条请求批在一起算(批处理)** 和 **怎么管 KV 缓存显存**。
本项目对照实现两种策略,跑同一份负载,看指标差距:

| | (a) 静态批处理 Static Batching | (b) 连续批处理 + 分页 KV (vLLM 思想) |
|---|---|---|
| 调度粒度 | **请求级**:凑齐一批一起发车 | **迭代级**:每个 forward step 重新调度 |
| 短请求能否早退 | 不能,**右 padding 等批内最慢的那条**(队头阻塞) | 能,**生成完 EOS 当步立即退出**并释放显存 |
| 新请求何时进来 | 等整批结束 / 等下一批凑齐 | **任意 step 即时补位** |
| KV 显存管理 | 按 `max_seq_len` **静态预留连续显存** → 碎片大 | 按 `block_size=16` 的块 **按需分配/释放** → 碎片极小 |

核心结论(论文 PagedAttention / arXiv:2309.06180 想说的):
**把"显存分页化 + 调度迭代化",能在不改模型的前提下大幅提升吞吐、降低延迟、几乎消除显存碎片。**

模拟器里关键的"原理点"都在代码注释里标了:
- `PagedKVCache`:空闲块列表 = vLLM block allocator 极简版,`allocate()` 演示"序列变长就追加一块"的**按需分配**;
- `run_static_batching`:step 成本按 **整批宽度** 计 + 步数 = 批内 `max(output_len)`,复现 **padding 浪费 + 队头阻塞**;
- `run_continuous_batching`:step 成本按 **当前活跃请求数** 计 + 完成即 `kv.free()` + 每步 `admit_new()` 补位,复现 **iteration-level 调度**;
- 成本模型 `step_time_ms = 固定开销 + 每请求边际开销`,近似 roofline:批越大、单 token 越省。

---

## ② 怎么跑

```bash
cd practical-projects/11-paged-attention-batching-sim
python paged_batching_sim.py
```

环境:Python 3.13 / numpy 2.3 / torch 2.12 (CPU)。约 **3 秒** 跑完。
有 matplotlib 会额外存一张 `comparison.png`(吞吐 / 延迟 / 碎片 三联柱状图);没有也不影响,文本报告已足够。

---

## ③ 预期输出(真实跑通后摘录)

```
 RESULTS
  [static_batching]
    forward steps      : 327
    makespan (ms)      :     9127.6
    throughput (tok/s) :      241.6
    avg latency (ms)   :     3971.0
    p99 latency (ms)   :     8508.6
    peak KV slots      : 2528
    avg KV frag (%)    :       64.5
  [continuous_paged]
    forward steps      : 211
    makespan (ms)      :     4620.8
    throughput (tok/s) :      477.2
    avg latency (ms)   :     1772.2
    p99 latency (ms)   :     3754.3
    peak KV slots      : 1024
    avg KV frag (%)    :       10.8
--------------------------------------------------------------------
 SUMMARY (continuous+paged vs static)
   makespan speedup        :   1.98x  faster
   throughput gain         :   1.98x  more tok/s
   avg-latency improvement :   2.24x  lower
   KV fragmentation        :  64.5%  ->  10.8%
   peak KV slots           : 2528  ->  1024
 CHECK PASSED: continuous+paged wins on throughput AND fragmentation.
```

**怎么读这几个数字:**
- **吞吐 ↑ 1.98x**:同样 2205 个输出 token,连续批处理用 211 个 step(静态批要 327 个)就跑完 —— 省下来的全是 padding 步和队头阻塞。
- **平均延迟 ↓ 2.24x**:短请求不再被长请求拖着陪跑,完成即返回。
- **碎片 64.5% → 10.8%**:静态预留按最坏序列长度,2/3 显存是空着的;分页只有"每条最后一块半空",浪费极小。
- **peak KV slots 2528 → 1024**:静态预留量甚至**超过了 1024 的物理显存池**(说明真实显卡上这份负载静态方案会 OOM),分页则把峰值压在池子容量内 —— 这正是 PagedAttention 让显存"装得下更多并发"的直观体现。

---

## ④ 对应 llm-action 文档

- KV 缓存优化原理:[`../../llm-inference/KV-Cache优化.md`](../../llm-inference/KV-Cache优化.md)
- vLLM 与 PagedAttention 专题:[`../../llm-inference/vllm/`](../../llm-inference/vllm/)
  (含 `vllm.md`、`请求处理流程.md`、`源码.md` 等)
- 推理优化总览:[`../../llm-inference/README.md`](../../llm-inference/README.md)
- 相关动手项目:[`../04-kv-cache`](../04-kv-cache)(KV 缓存本身)、[`../07-speculative-decoding`](../07-speculative-decoding)(投机解码)

---

## ⑤ 社区参考

- [vLLM: Efficient Memory Management for LLM Serving with PagedAttention — arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System (blog.vllm.ai)](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)

---

## ⑥ 局限 / 与真实工程的差异

本项目是 **教学用离散事件模拟**,刻意做小、做透原理,不等于工程实现:

1. **不跑真实算子**:没有真实的矩阵乘 / attention,"一个 step 的耗时"用线性成本模型近似。
   真实延迟受 GPU 算力、内存带宽、kernel 调度、量化等影响,绝对数字不可外推,**只反映趋势**。
2. **简化的抢占**:显存不够时这里只是"该步不推进";真实 vLLM 有 **抢占 (preemption)** 与
   **换出/重算 (swap / recompute)** 策略来腾显存。
3. **未实现的高级特性**:prefix caching(共享前缀复用 KV)、chunked prefill(长 prompt 切块)、
   copy-on-write 块共享(beam / parallel sampling)、speculative decoding、多卡 TP/PP 等,均未建模。
4. **成本模型是单机近似**:`STEP_FIXED_MS / STEP_PER_REQ_MS` 是手调常数,用于演示"批越大越省";
   想体验不同硬件/负载,可调这两个常数以及 `BLOCK_SIZE / TOTAL_KV_BLOCKS / N` 重跑。
5. **负载是合成的**:用泊松到达 + 长尾输出长度合成,刻意放大静态批的劣势;
   真实流量分布不同,收益幅度会变,但"连续批 + 分页全面占优"的方向稳定成立。

> 一句话:这份代码帮你**建立直觉、看懂 vLLM 在解决什么问题**;真正上生产请直接用 vLLM / SGLang。
