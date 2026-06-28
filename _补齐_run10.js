export const meta = {
  name: 'llmaction-补齐-run10-paper工具',
  description: 'llm-action补齐: paper论文精读17+blog/工具/流水序列并行等40篇+双链',
  phases: [
    { title: 'P1-论文与推理', detail: 'paper精读+推理相关 14篇' },
    { title: 'P2-训练与压缩', detail: 'paper训练/剪枝+流水序列并行+压缩 14篇' },
    { title: 'P3-工具与杂项', detail: 'nsight/nvtx/blog/paddle/template 12篇' },
  ],
}

const DIR = 'C:/Users/jianm/Desktop/llm-action/'
const SCHEMA = { type: 'object', properties: { file: { type: 'string' }, lines: { type: 'number' }, hasAscii: { type: 'boolean' }, links: { type: 'number' } }, required: ['file', 'lines', 'hasAscii', 'links'] }

const HUBS = '00-知识地图 / llm-algo/transformer/模型架构 / llm-algo/moe/README / llm-optimizer/FlashAttention / llm-optimizer/kv-cache / llm-inference/PD分离 / llm-inference/KV-Cache优化 / llm-inference/vllm/README / llm-inference/连续批处理? / llm-compression/quantization/量化基础 / llm-compression/sparsity/README / llm-compression/quantization/fp8 / llm-train/README / B07:llm-inference/大模型推理张量并行 / llm-train/pytorch/distribution/README / llm-alignment/RLHF / llm-alignment/DPO / ai-infra/网络/集合通信原语 / ai-infra/算力/GPU工作原理 / llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释 / docs/transformer内存估算'

function buildPrompt(path) {
  return `你是顶尖 AI-Infra 讲师/研究者。把 llm-action 仓库一个【薄文件】补齐到最大细节，用【中文】，并用 Write 工具覆盖写回原路径。

【文件】${path}
【第一步】Read ${DIR}${path} 看已有方向并尊重;近空按路径主题从零写。
【第二步】用 Write 写到绝对路径：${DIR}${path}

【按路径选体裁】
- 若路径以 "paper/" 开头 → 写成【论文精读笔记】:论文要解决的问题/核心方法(拆到能懂)/关键创新点/关键公式或算法(给出并解释)/实验结论(定性,数字标"约/见原文")/对工程的启示/局限与后续。务必讲清"这篇到底做了什么、为什么重要"。
- 若是 "blog/" 或科普向 → 写成【深入浅出讲义】:从直觉到原理,大白话+严谨并存。
- 若是工具(nsight/nvtx/可视化) → 讲【它测什么、原理、怎么用来定位LLM性能瓶颈、看什么指标】。
- 若是流水/序列并行/RPC教程 → 讲【机制+为什么+ASCII图+通信/显存账+最小示例思路】。
- 其它(源码/paddle/模板) → 讲架构/机制/流程/用途。

【通用风格】概念拆原子、每步说为什么、【必含ASCII图】、适用就给【数值手算】(通信量/显存/FLOPs/加速比等)、公式$...$。
【护栏】不编造精确版本/命令默认值/未核实数字;论文数字标"约/见原文";不确定标"以官方/原文为准"。

【固定结构】
# <主题>
> 一句话定位。📍 导航：[[00-知识地图]]
> 🔗 相关：[[相关富笔记]] ...(从枢纽清单选最相关的2-4个)
## 阅读地图(表格)
## 0. 一句话锚点
## 1. 地基/前置(问题背景)
## 2.~N. 逐步拆解(方法/机制, 每节配图)
## (单独一节)关键公式/算法/数值示例
## 评价/对照/局限(表格)
## 🔗 跳转链接
枢纽清单(可链): ${HUBS}

【硬要求】200~400 行, 高密度; 顶部导航[[00-知识地图]]+底部双链必须有。返回 JSON {file,lines,hasAscii,links}。直接开始。`
}

const FILES = [
  // P1 论文与推理
  'paper/PagedAttention.md','paper/inference/orca.md','paper/inference/llm-in-a-flash.md','paper/inference/迈向高效的生成式大语言模型服务综述-从算法到系统.md','paper/README.md','paper/moe/README.md','paper/LLM增强LLMS.md','llm-inference/ascend/mindformers/baichuan2/README.md','llm-inference/ascend/mindformers/chatglm3/README.md','llm-inference/sglang/source-code.md','blog/reference/高性能 LLM 推理框架的设计与实现.md','blog/llm-algo/moe.md','blog/llm-algo/大白话Transformer架构.md','ai-framework/deepspeed/training/pipeline_parallelism/README.md',
  // P2 训练与压缩
  'paper/training/GaLore.md','paper/training/Reducing Activation Recomputation in Large Transformer Models.md','paper/training/A Survey on Efficient Training of Transformers.md','paper/parameter-pruning/SparseGPT.md','paper/parameter-pruning/Wanda.md','paper/parameter-pruning/LLM-Pruner.md','paper/parameter-pruning/公式.md','paper/llm对齐综述.md','paper/LESS-选择有影响力的数据进行目标指令精调.md','paper/data/LESS-选择有影响力的数据进行目标指令精调.md','llm-train/pytorch/distribution/pipeline-parallel/3-使用流水线并行训练Transformer模型.md','llm-train/pytorch/distribution/pipeline-parallel/4-使用DDP与流水线并行训练Transformer模型.md','llm-train/pytorch/distribution/sequence-parallelism/README.md','llm-train/pytorch/distribution/rpc/README.md',
  // P3 工具与杂项
  'llm-tools/nsight.md','llm-tools/nvtx.md','llm-tools/可视化.md','blog/llm-compression/大模型量化技术原理-ZeroQuant系列.md','llm-compression/llm-compressor/source-code.md','llm-compression/quantization/llm-qat/cfd70ff/README.md','llm-train/megatron-deepspeed/source-code.md','llm-train/megatron-deepspeed/microsoft/slurm/README.md','llm-train/megatron/gpt2/merge_ck_and_inference/README.md','llm-train/paddle/paddlenlp/baichuan2/README.md','llm-train/paddle/paddlenlp/bloom/README.md','template/server.md',
]

const runWave = (paths, phaseTitle) =>
  parallel(paths.map(p => () => agent(buildPrompt(p), { label: `fill:${p}`, phase: phaseTitle, schema: SCHEMA })))

const b1 = FILES.slice(0, 14), b2 = FILES.slice(14, 28), b3 = FILES.slice(28)
phase('P1-论文与推理'); const r1 = await runWave(b1, 'P1-论文与推理'); log(`P1: ${r1.filter(Boolean).length}/${b1.length}`)
phase('P2-训练与压缩'); const r2 = await runWave(b2, 'P2-训练与压缩'); log(`P2: ${r2.filter(Boolean).length}/${b2.length}`)
phase('P3-工具与杂项'); const r3 = await runWave(b3, 'P3-工具与杂项'); log(`P3: ${r3.filter(Boolean).length}/${b3.length}`)

const all = [...r1, ...r2, ...r3].filter(Boolean)
return { total: FILES.length, written: all.length, files: all.map(x => x.file) }
