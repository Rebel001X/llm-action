# 🛠️ practical-projects · 大模型全栈实战项目（可跑代码）

> 本目录是对 llm-action（24.3k★ 中文 AI-Infra 教程）的**动手实战补充**：仓库以原理文档为主，这里提供 **14 个端到端、CPU 几十秒即可跑通**的最小实战项目，把"推理优化 / 训练 / 压缩 / 对齐 / 分布式 / 应用"的核心思想用**可运行、可验证**的代码讲清楚。
>
> 全部纯 `numpy` / `torch`（CPU），自包含（不联网、不依赖外部数据集/大模型），设了随机种子可复现，每个都经过独立 `python` smoke test 验证。**适合面试准备、教学、和"读完文档想动手"时的第一站。**

> 📚 **每个项目都是一个完整学习单元（五件套）**：
> - `<脚本>.py` —— 可跑代码（CPU 几十秒，smoke test 验证过）
> - `README.md` —— 速览（原理/怎么跑/预期输出/参考）
> - `手把手教学.md` —— 350~750 行手把手详解：背景 → 原理数学推导(mermaid) → **代码逐段精讲** → 逐行读输出 → 动手练习 → 坑 → 速查
> - `面试题.md` —— 分级面试题库（基础/进阶/手撕 + 追问，带参考答案），对接 `llm-interview/`
> - `进阶与生产实践.md` —— 从 toy 到生产：vLLM/Megatron/TRT-LLM/PEFT 等业界系统怎么做、关键变体与优化、真实权衡与坑
>
> 建议顺序：手把手教学 → 跑代码改参数 → 进阶与生产实践 → 面试题自测。

## 怎么跑
```bash
cd practical-projects/<项目目录>
python <脚本>.py     # 跑完会打印可量化的"成功信号"，多数还会存一张学习/对比曲线 png
cat 手把手教学.md     # ← 配套的手把手详细教学（强烈建议先读）
```
需要 `pip install torch numpy matplotlib`（matplotlib 仅用于存曲线，缺了也不影响主逻辑）。
每个项目目录下是**五件套**：`<脚本>.py` + `README.md` + `手把手教学.md` + `面试题.md` + `进阶与生产实践.md`（详见上方 📚）。

## 14 个项目一览（均实跑验证 ✅）

| # | 项目 | 对应仓库分区 | 演示要点 | 实跑结果 |
|---|---|---|---|---|
| 01 | [tiny-gpt-from-scratch](01-tiny-gpt-from-scratch/) | `llm-algo` | 从零搭字符级 GPT 并预训练+采样 | eval loss 3.99→0.078，采样出风格化文本 |
| 02 | [bpe-tokenizer](02-bpe-tokenizer/) | `llm-data-engineering` | BPE 分词器训练 + 无损 round-trip | 含中文/emoji 无损，压缩 2.97× |
| 03 | [lora-from-scratch](03-lora-from-scratch/) | `llm-train/peft` | LoRA 低秩适配器微调 | 仅训 2336 参数，准确率 0.16→0.99 |
| 04 | [kv-cache](04-kv-cache/) | `llm-inference` | KV Cache 自回归加速 | 输出逐 token 一致，加速 1.57→2.33× |
| 05 | [flash-attention-tiled](05-flash-attention-tiled/) | `llm-optimizer` | 分块 + 在线 softmax 注意力 | 与朴素 allclose，误差 ~1e-15，显存 O(n) |
| 06 | [quantization-int8-int4](06-quantization-int8-int4/) | `llm-compression` | INT8/INT4 权重量化 | INT8 cosine 0.99999 / INT4 0.9976，压缩 4×/8× |
| 07 | [speculative-decoding](07-speculative-decoding/) | `llm-inference` | 投机采样（draft+target 验证） | 2.38× 加速且分布无损 |
| 08 | [moe-from-scratch](08-moe-from-scratch/) | `llm-algo/moe` | top-k 路由 MoE + 负载均衡损失 | 不均衡 inf→1.24（逼近均匀） |
| 09 | [dpo-from-scratch](09-dpo-from-scratch/) | `llm-alignment` | DPO 偏好对齐（无需奖励模型/PPO） | 偏好准确率 0.58→1.00 |
| 10 | [rag-minimal](10-rag-minimal/) | `llm-application` | 最小 RAG（检索增强生成） | 命中率 0%→100%（缓解凭空编造） |
| 11 | [paged-attention-batching-sim](11-paged-attention-batching-sim/) | `llm-inference` / `llmops` | 连续批处理 + 分页 KV 调度模拟 | 吞吐 1.98×，KV 碎片 64.5%→10.8% |
| 12 | [tensor-parallel-from-scratch](12-tensor-parallel-from-scratch/) | `ai-framework` | Megatron 列/行切分张量并行 | 与单卡 allclose，误差 ~1e-7 |
| 13 | [ring-allreduce](13-ring-allreduce/) | `ai-infra/网络` | Ring All-Reduce 通信原语 | 与朴素求和等价，通信省 128× |
| 14 | [llm-eval-mini](14-llm-eval-mini/) | `llm-eval` | 困惑度 + 解码策略对比评测 | PPL 24.1→1.07，下游 acc 0.03→1.0 |

## 学习路线建议
- **想搞懂推理优化（面试高频）**：04 KV Cache → 05 FlashAttention → 07 投机采样 → 11 连续批/分页KV。
- **想搞懂训练/微调**：01 tiny-GPT → 03 LoRA → 09 DPO。
- **想搞懂分布式**：12 张量并行 → 13 Ring All-Reduce。
- **想搞懂压缩/部署**：06 量化 → 04 KV Cache。
- **想搞懂应用落地**：10 RAG → 02 BPE → 14 评测。

## 社区优秀参考（动手向）
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) · [karpathy/minbpe](https://github.com/karpathy/minbpe) — 从零搭 GPT / BPE 的经典极简实现
- [walkinglabs/modern-llm-notebook](https://github.com/walkinglabs/modern-llm-notebook) — 26 个 notebook 覆盖 tokenizer→attention→MoE→RLHF→inference
- [xlite-dev/Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference) — 推理优化论文+代码合集（FlashAttention/PagedAttention/量化/并行）
- [vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) — PagedAttention/连续批处理拆解
- [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm) · [mlabonne LLM course](https://huggingface.co/blog/mlabonne/llm-course) — 系统化从零课程
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 业界标准评测框架

## 与文档的关系
每个项目 README 顶部都链回 llm-action 对应的原理文档（相对路径）。**先读原理，再来这里跑代码、改参数、看现象**，理解会牢得多。

---
*本目录为 llm-action 的实战补充，由多 agent 自动化产出并逐个 smoke test 跑通。欢迎提 issue / PR 增补更多项目。*
