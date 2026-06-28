export const meta = {
  name: 'llmaction-补齐-run8-收尾',
  description: 'llm-action收尾: 补齐最后66篇工具配置/源码/部署类(通用路径派生)+双链',
  phases: [
    { title: 'T1-收尾批1', detail: '22篇' },
    { title: 'T2-收尾批2', detail: '22篇' },
    { title: 'T3-收尾批3', detail: '22篇' },
  ],
}

const DIR = 'C:/Users/jianm/Desktop/llm-action/'
const SCHEMA = { type: 'object', properties: { file: { type: 'string' }, lines: { type: 'number' }, hasAscii: { type: 'boolean' }, links: { type: 'number' } }, required: ['file', 'lines', 'hasAscii', 'links'] }

// 可链接的主要枢纽文件(已补齐)，供 agent 选相关链接
const HUBS = '00-知识地图 / llm-inference/README / llm-inference/vllm/README / llm-inference/sglang/README / llm-inference/tensorrt/README / llm-optimizer/kv-cache / llm-optimizer/FlashAttention / llm-compression/quantization/量化基础 / llm-compression/README / llm-train/README / llm-train/pytorch/distribution/README / ai-framework/README / ai-framework/deepspeed/README / ai-framework/megatron-lm/README / ai-framework/pytorch/README / ai-infra/网络/集合通信原语 / ai-infra/网络/InfiniBand / ai-infra/网络/NCCL / ai-infra/ai-cluster/README / ai-infra/ai-hardware/README / ai-infra/算力/昇腾NPU / llm-algo/transformer/模型架构 / llm-alignment/RLHF / llm-eval/README / llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释 / llm-application/README / llm-application/rag/README / llm-maas/README / llmops/README / llmops/kubernetes'

function buildPrompt(path) {
  return `你是顶尖 AI-Infra 讲师。把 llm-action 仓库里一个【薄文件】补齐到尽量详细，用【中文】，并用 Write 工具覆盖写回原路径。

【文件(相对仓库根)】${path}
【如何确定主题】路径本身就是主题线索:目录名是大类、文件名(或README的父目录名)是具体主题。例如 "llm-inference/tensorrt-llm/TRT-LLM引擎构建参数.md" 主题=TensorRT-LLM 引擎构建参数; "llm-train/pytorch/distribution/多机训练.md" 主题=PyTorch 多机训练; "ai-infra/网络/IB流量监控.md" 主题=InfiniBand 流量监控。
【第一步】先用 Read 读 ${DIR}${path}，看是否已有内容/方向并尊重它;近空则按路径主题从零写。
【第二步】用 Write 写到绝对路径：${DIR}${path}

【风格】"从最底层讲清 + 可视化":概念拆原子、每步说为什么、【必含至少1个 ASCII 图】(架构/流程/调用链/拓扑);适用就给数值例子;公式用 $...$。

【极重要的护栏——这是工具/配置/源码/部署类主题】
- 讲【稳定知识】:是什么、解决什么问题、整体架构、核心机制/流程、各类参数/配置项是【做什么用的、怎么权衡】、与同类对比、实践注意点、常见坑。
- 【绝对不要编造】精确版本号、CLI 参数的确切名字/默认值、API 函数签名、具体源码行号、未经证实的性能数字。不确定就讲【类别与机制】并明确写"具体以官方文档/源码为准"。宁可讲通用原理，不要造假细节。

【固定结构】
# <主题>
> 一句话定位。📍 导航：[[00-知识地图]]
> 🔗 相关：[[xxx]] [[yyy]]  (从下方枢纽清单选2-4个最相关的，或仓库内其他你确定存在的文件)
## 阅读地图(表格)
## 0. 一句话锚点
## 1. 地基/前置(它解决什么问题)
## 2.~N. 逐步拆解(配ASCII图)
## (单独一节)典型流程/配置示例/参数说明(讲含义,不背默认值)
## 常见问题/坑(表格)
## 🔗 跳转链接

可选枢纽双链清单(只链相关的):
${HUBS}

【硬要求】180~360 行(工具类不强求很长但要扎实); 顶部导航[[00-知识地图]]+底部双链必须有。返回 JSON {file,lines,hasAscii,links}。直接开始,不要解释。`
}

const FILES = [
  'ai-compiler/Treebeard/README.md','ai-compiler/treelit/README.md','ai-compiler/treelit/xgb.md',
  'ai-framework/deepspeed/config-json/deepspeed-nvme.md','ai-framework/deepspeed/deepspeed-slurm.md','ai-framework/mxnet/README.md','ai-framework/pai-torchacc.md','ai-framework/pytorch/install.md',
  'ai-infra/ai-hardware/OEM-DGX.md','ai-infra/ai-hardware/TSMC-台积电.md','ai-infra/ai-hardware/cuda镜像.md','ai-infra/网络/IB-docker.md','ai-infra/网络/IB流量监控.md','ai-infra/网络/README.md',
  'llm-algo/bloom/README.md','llm-alignment/README.md',
  'llm-application/Higress.md','llm-application/agent/OpenClaw.md','llm-application/gradio/README.md','llm-application/one-api.md','llm-application/rag/embedding.md',
  'llm-compression/PaddleSlim/ quantization.md','llm-compression/quantization/llm-qat/log.md','llm-compression/tools.md',
  'llm-data-engineering/dataset/baichuan2.md','llm-data-engineering/dataset/english-corpus-all.md','llm-data-engineering/sft-dataset/jinja.md',
  'llm-eval/llm-performance/README.md','llm-eval/llm-performance/tgi-benchmark.md','llm-eval/llm-performance/vllm/README.md','llm-eval/llm-performance/wrk-性能测试工具.md',
  'llm-inference/faster-transformer/gpt/README.md','llm-inference/faster-transformer/llama/README.md','llm-inference/sglang/服务器启动参数.md','llm-inference/tensorrt-llm/FP8.md','llm-inference/tensorrt-llm/Memory Usage of TensorRT-LLM.md','llm-inference/tensorrt-llm/TRT-LLM引擎构建参数.md','llm-inference/tensorrt-llm/Triton服务启动参数.md','llm-inference/tensorrt/install.md','llm-inference/triton/REAEME.md','llm-inference/triton/onnx/README.md','llm-inference/vllm/FAQ.md','llm-inference/vllm/FP8.md','llm-inference/vllm/cmd.md','llm-inference/vllm/vllm.md','llm-inference/web/flask/README.md','llm-inference/web/sanic/README.md',
  'llm-interview/README.md',
  'llm-train/chatglm-lora/README.md','llm-train/megatron-deepspeed/bigscience/bloom-note.md','llm-train/megatron-deepspeed/microsoft/H800多机多卡训练坑点.md','llm-train/megatron-deepspeed/microsoft/代码.md','llm-train/megatron-deepspeed/microsoft/环境准备.md','llm-train/megatron-deepspeed/microsoft/训练日志分析.md','llm-train/megatron/codegeex/README.md','llm-train/megatron/gpt2/README.md','llm-train/paddle/README.md','llm-train/peft/conditional_generation/README.md','llm-train/pytorch/Pytorch源码解读.md','llm-train/pytorch/distribution/torchrun.md','llm-train/pytorch/distribution/多机训练.md','llm-train/pytorch/resource.md',
  'llmops/FAQ.md','llmops/tq-llm/train/FAQ.md','llmops/tq-llm/train/README.md','llmops/使用docker进行多机多卡训练.md',
]

const runWave = (paths, phaseTitle) =>
  parallel(paths.map(p => () => agent(buildPrompt(p), { label: `fill:${p}`, phase: phaseTitle, schema: SCHEMA })))

const b1 = FILES.slice(0, 22), b2 = FILES.slice(22, 44), b3 = FILES.slice(44)

phase('T1-收尾批1')
const r1 = await runWave(b1, 'T1-收尾批1')
log(`T1: ${r1.filter(Boolean).length}/${b1.length}`)
phase('T2-收尾批2')
const r2 = await runWave(b2, 'T2-收尾批2')
log(`T2: ${r2.filter(Boolean).length}/${b2.length}`)
phase('T3-收尾批3')
const r3 = await runWave(b3, 'T3-收尾批3')
log(`T3: ${r3.filter(Boolean).length}/${b3.length}`)

const all = [...r1, ...r2, ...r3].filter(Boolean)
return { total: FILES.length, written: all.length, files: all.map(x => x.file) }
