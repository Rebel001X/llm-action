# 从零实现 KV Cache 推理加速

用一个 toy 规模的 GPT(decoder-only Transformer),在 CPU 上从零实现并对比
自回归生成的两条路径,直观理解 KV Cache 为什么是大模型推理的基石。

## 1. 演示什么原理

自回归生成是一个 token 一个 token 往外蹦的过程。朴素做法每生成一个新 token,
都把"到目前为止的整段序列"重新喂进模型,从头算一遍所有 token 的 Q/K/V 和注意力。

但注意一个关键事实:**模型是因果(causal)的——已经算过的 token,它们的
Key/Value 不会因为后面追加新 token 而改变。** 既然不变,就没必要重算。

- **(a) 无 cache(naive):** 第 t 步要处理 t 个 token,总计算量
  `1 + 2 + ... + n ≈ O(n²)`。
- **(b) KV cache:** 把每层每个历史 token 的 K、V 缓存下来;每步只对"新来的
  1 个 token"算 Q/K/V,把它的 K/V 追加进 cache,再与全部历史 K/V 做注意力。
  每步只算 1 个 token,总计算量 `≈ O(n)`。

两条路径**在数学上完全等价**,因此生成的 token 序列必须逐一相同。脚本用
`assert` 强制校验这一点,并打印随序列变长而增大的加速比。

真实的 LLM 服务(vLLM / TGI / TensorRT-LLM)都依赖 KV Cache;后续的
PagedAttention、KV Cache 量化 / offload、prefix caching 等,本质都是"如何更省
内存、更高吞吐地管理这块 cache"。理解了这个 toy 版,再看系统级优化就有地基了。

## 2. 怎么跑

```bash
cd practical-projects/04-kv-cache
python kv_cache.py
```

依赖:`numpy`、`torch`(CPU 即可)。无需数据集 / 网络,CPU 几十秒内跑完。

## 3. 预期输出(真实跑通,摘录)

```
 gen_len |  no_cache(s) |   cache(s) |  speedup |  match
----------------------------------------------------------------
      16 |       0.0420 |     0.0268 |    1.57x |   True
      32 |       0.0663 |     0.0378 |    1.75x |   True
      64 |       0.1339 |     0.0661 |    2.03x |   True
     128 |       0.1868 |     0.0801 |    2.33x |   True
----------------------------------------------------------------
All sequences MATCH -> KV cache is mathematically equivalent. PASS

Complexity intuition (work = #tokens whose Q/K/V are computed):
  generating 8 tokens, no_cache attends to: [1, 2, 3, 4, 5, 6, 7, 8]  -> total 36 (O(n^2))
  generating 8 tokens, with_cache computes:  [1, 1, 1, 1, 1, 1, 1, 1]  -> total 8 (O(n))

speedup grows with length: ['1.57x', '1.75x', '2.03x', '2.33x']
max speedup observed: 2.33x at gen_len=128
```

两个可量化的"成功信号":

1. **正确性:** 所有生成长度上 `match=True`,KV cache 与朴素重算逐 token 一致。
2. **加速比随长度增大:** `1.57x → 1.75x → 2.03x → 2.33x`,正是 O(n²) 退化为
   O(n) 的体现(序列越长、收益越大;真实模型规模下差距远比 toy 显著)。

脚本还会用 `matplotlib`(Agg 后端)输出 `kv_cache_speedup.png`;若环境无
matplotlib,则自动降级为文本提示,不会崩。

## 4. 对应 llm-action 文档

- [`../../llm-inference/KV-Cache优化.md`](../../llm-inference/KV-Cache优化.md) —— KV Cache 优化专题
- [`../../llm-inference`](../../llm-inference) —— 大模型推理目录(含 vLLM、PD 分离、Flash-Decoding 等)
- [`../../llm-inference/解码策略.md`](../../llm-inference/解码策略.md) —— 解码策略(本例用贪心解码保证输出确定可比较)

## 5. 社区参考

- [vLLM PagedAttention 论文 arXiv:2309.06180](https://arxiv.org/abs/2309.06180) —— 把 KV Cache 当作"分页内存"管理,解决碎片与浪费
- [Inside vLLM anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) —— vLLM 内部结构剖析,包含 KV Cache / 调度的工程视角

## 6. 局限 / 与真实工程的差异

- **toy 规模:** 仅 4 层、64 维、词表 64,纯为教学;真实模型加速比可达数十倍,
  且无 cache 在长序列上几乎不可用。
- **未训练:** 权重是随机初始化的,生成内容无语义;本例只验证"两条路径等价 +
  加速",不关心生成质量。
- **cache 无上限增长:** 这里 K/V 沿时间维不断 `cat` 拼接,内存随长度线性增长。
  真实系统用 **PagedAttention**(分页、按需分配、可共享 prefix)来避免碎片与
  浪费——这正是 vLLM 的核心贡献。
- **绝对位置编码:** 用了简单的 learned 绝对位置嵌入;真实大模型多用 RoPE 等
  相对位置编码,KV cache 的拼接逻辑需相应适配(对已缓存 K 应用对应位置旋转)。
- **未做的工程优化:** 量化(KV int8/fp8)、offload(显存 ↔ 内存 ↔ 磁盘)、
  连续批处理(continuous batching)、MQA/GQA(多 query 共享 K/V 头以省 cache)
  等均未涉及,留作进阶阅读。
- **CPU 单线程计时:** 为让复杂度差异更稳定可见而设 `set_num_threads(1)`;GPU 上
  因并行度与访存模式不同,绝对数字会变,但 O(n²)→O(n) 的趋势不变。
```
