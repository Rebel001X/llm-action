export const meta = {
  name: 'llmaction-补齐-run9-docs',
  description: 'llm-action补齐docs/: 45篇核心知识(并行/FlashAttn/KV/PEFT/RLHF等)+交叉链接富笔记',
  phases: [
    { title: 'D1-docs批1', detail: '15篇' },
    { title: 'D2-docs批2', detail: '15篇' },
    { title: 'D3-docs批3', detail: '15篇' },
  ],
}

const DIR = 'C:/Users/jianm/Desktop/llm-action/'
const SCHEMA = { type: 'object', properties: { file: { type: 'string' }, lines: { type: 'number' }, hasAscii: { type: 'boolean' }, links: { type: 'number' } }, required: ['file', 'lines', 'hasAscii', 'links'] }

// 已补齐的核心富笔记, 供 docs 交叉链接(路径式 Obsidian 双链)
const HUBS = '00-知识地图 / llm-algo/transformer/模型架构 / llm-algo/FLOPs / llm-algo/mlp / llm-algo/moe/README / llm-algo/旋转编码RoPE / llm-optimizer/FlashAttention / llm-optimizer/kv-cache / llm-optimizer/计算通信重叠 / llm-inference/KV-Cache优化 / llm-inference/解码策略 / llm-inference/大模型推理张量并行 / llm-inference/README / llm-compression/quantization/量化基础 / llm-compression/quantization/fp8 / llm-train/README / llm-train/peft/Prompt-Tuning / llm-train/peft/Prefix-Tuning / llm-alignment/RLHF / llm-alignment/DPO / ai-framework/deepspeed/README / ai-framework/megatron-lm/README / ai-infra/网络/集合通信原语 / ai-infra/算力/GPU工作原理 / ai-infra/ai-hardware/CUDA / B07相关:llm-inference/大模型推理张量并行'

function buildPrompt(path) {
  return `你是顶尖 AI-Infra 讲师。把 llm-action 仓库 docs/ 下一个【薄文件】补齐到最大细节，用【中文】，并用 Write 工具覆盖写回原路径。

【文件】${path}
【主题来自路径】目录/文件名即主题线索。例:"docs/llm-base/distribution-parallelism/tensor-parallel/tensor-parallel.md"=张量并行;"docs/transformer内存估算.md"=Transformer训练/推理内存估算;"docs/flash-attention/FlashAttention.md"=FlashAttention;"docs/llm-base/FLOPS.md"=FLOPs与计算量估算。
【第一步】Read ${DIR}${path} 看已有方向并尊重;近空按主题从零写。
【第二步】用 Write 写到绝对路径：${DIR}${path}

【风格】"从最底层讲清+逐数手算+ASCII可视化":概念拆原子、每步说为什么、【必含ASCII图】、【适用必含数值手算】(并行通信量/内存估算/FLOPs/显存等)、公式用$...$。

【交叉链接(重要)】本仓库已有大量同主题的"原子级富笔记"。请在顶部"相关"和底部"跳转链接"用路径式 Obsidian 双链指向最相关的几个(从下方枢纽清单选)，让 docs 与正文互通。例如讲张量并行就链 [[llm-inference/大模型推理张量并行]]、[[ai-infra/网络/集合通信原语]];讲FlashAttention就链 [[llm-optimizer/FlashAttention]]。
枢纽清单：${HUBS}

【护栏】讲稳定原理与机制;工具/环境类不编造精确版本/命令默认值, 不确定标"以官方为准"。

【固定结构】
# <主题>
> 一句话定位。📍 导航：[[00-知识地图]]
> 🔗 相关：[[最相关富笔记]] ...
## 阅读地图(表格)
## 0. 一句话锚点
## 1. 地基/前置
## 2.~N. 逐步拆解(每节配ASCII图; 适用处给公式/手算)
## (单独一节)数值手算/示例
## 常见问题(表格)
## 🔗 跳转链接(列相关富笔记双链)

【硬要求】200~400 行, 高密度; 顶部导航[[00-知识地图]]+底部双链必须有。返回 JSON {file,lines,hasAscii,links}。直接开始。`
}

const FILES = [
  'docs/README.md','docs/conda.md','docs/flash-attention/FlashAttention.md','docs/llm-base/FLOPS.md','docs/llm-base/NVIDIA-Nsight-Systems性能分析.md','docs/llm-base/README.md','docs/llm-base/ai-algo.md','docs/llm-base/distribution-parallelism/auto-parallel/Mesh-Tensorflow.md','docs/llm-base/distribution-parallelism/auto-parallel/Unity.md','docs/llm-base/distribution-parallelism/auto-parallel/auto-parallel.md','docs/llm-base/distribution-parallelism/auto-parallel/gspmd.md','docs/llm-base/distribution-parallelism/auto-parallel/分布式训练自动并行概述.md','docs/llm-base/distribution-parallelism/data-parallelism/README.md','docs/llm-base/distribution-parallelism/moe-parallel/moe-framework.md','docs/llm-base/distribution-parallelism/moe-parallel/moe-parallel.md',
  'docs/llm-base/distribution-parallelism/pipeline-parallelism/README.md','docs/llm-base/distribution-parallelism/tensor-parallel/README.md','docs/llm-base/distribution-parallelism/tensor-parallel/tensor-parallel.md','docs/llm-base/distribution-training/Bloom-176B训练经验.md','docs/llm-base/distribution-training/FP16-BF16.md','docs/llm-base/distribution-training/README.md','docs/llm-base/distribution-training/自动混合精度.md','docs/llm-base/gpu-env-var.md','docs/llm-base/multimodal/sora.md','docs/llm-base/rlhf/README.md','docs/llm-base/scenes/cv/README.md','docs/llm-base/scenes/cv/paddle/README.md','docs/llm-base/scenes/cv/pytorch/README.md','docs/llm-base/singularity命令.md','docs/llm-base/分布式训练加速技术.md',
  'docs/llm-experience.md','docs/llm-inference/DeepSpeed-Inference.md','docs/llm-inference/KV-Cache.md','docs/llm-inference/LLM服务框架对比.md','docs/llm-inference/README.md','docs/llm-inference/llm推理框架.md','docs/llm-inference/vllm.md','docs/llm-peft/MAM_Adapter.md','docs/llm-peft/README.md','docs/llm-peft/ReLoRA.md','docs/llm-summarize/README.md','docs/llm-summarize/distribution_dl_roadmap.md','docs/llm-summarize/文档大模型.md','docs/llm-summarize/金融大模型.md','docs/transformer内存估算.md',
]

const runWave = (paths, phaseTitle) =>
  parallel(paths.map(p => () => agent(buildPrompt(p), { label: `docs:${p}`, phase: phaseTitle, schema: SCHEMA })))

const b1 = FILES.slice(0, 15), b2 = FILES.slice(15, 30), b3 = FILES.slice(30)
phase('D1-docs批1'); const r1 = await runWave(b1, 'D1-docs批1'); log(`D1: ${r1.filter(Boolean).length}/${b1.length}`)
phase('D2-docs批2'); const r2 = await runWave(b2, 'D2-docs批2'); log(`D2: ${r2.filter(Boolean).length}/${b2.length}`)
phase('D3-docs批3'); const r3 = await runWave(b3, 'D3-docs批3'); log(`D3: ${r3.filter(Boolean).length}/${b3.length}`)

const all = [...r1, ...r2, ...r3].filter(Boolean)
return { total: FILES.length, written: all.length, files: all.map(x => x.file) }
