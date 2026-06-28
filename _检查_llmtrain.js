export const meta = {
  name: 'llmaction-检查-llmtrain',
  description: 'llm-train 27个未改造文件: 薄的重构补齐(保留真实命令)+厚的仅加导航(保原文)',
  phases: [
    { title: 'E-薄中文件重构补齐', detail: '16篇<8KB重构为深度笔记(保留原文真料)' },
    { title: 'N-厚巨文件加导航', detail: '11篇≥8KB仅插入导航+双链, 原文不动' },
  ],
}

const DIR = 'C:/Users/jianm/Desktop/llm-action/'
const HUBS = '00-知识地图 / llm-train/README / llm-train/pytorch/distribution/README / llm-train/megatron/README / llm-train/megatron-deepspeed/README / ai-framework/megatron-lm/README / ai-framework/deepspeed/README / ai-framework/pytorch/README / ai-framework/huggingface-peft/README / llm-train/peft/PEFT-API / llm-train/peft/Prompt-Tuning / llm-train/peft/Prefix-Tuning / ai-infra/网络/集合通信原语 / ai-infra/网络/NCCL / llm-alignment/RLHF / llm-algo/transformer/模型架构 / llm-compression/quantization/量化基础 / B07:llm-inference/大模型推理张量并行'

const ENRICH_SCHEMA = { type: 'object', properties: { file: { type: 'string' }, lines: { type: 'number' }, keptOriginal: { type: 'boolean' }, hasAscii: { type: 'boolean' }, links: { type: 'number' } }, required: ['file', 'lines', 'keptOriginal', 'hasAscii', 'links'] }
const NAV_SCHEMA = { type: 'object', properties: { file: { type: 'string' }, action: { type: 'string' }, links: { type: 'number' }, bodyPreserved: { type: 'boolean' } }, required: ['file', 'action', 'links', 'bodyPreserved'] }

function enrichPrompt(path) {
  return `你是顶尖 AI-Infra 讲师。把 llm-action 仓库一个【已有部分内容但未达深度笔记标准】的文件提升为完整深度笔记，用【中文】，Write 覆盖写回。

【文件】${path}
【第一步】先 Read ${DIR}${path}。判断:它已有哪些【真实有价值信息】(真实命令/配置/代码片段/参数/踩坑)。
【第二步】用 Write 写回 ${DIR}${path}，要求:
- 【保留原文所有真实可用信息】(命令/配置/代码原样保留或更清晰呈现), 在其基础上【补全】:原理、为什么、ASCII图、对照表、数值示例、常见坑。
- 【不要删减真实信息, 不要编造新命令/版本/参数】。把零散内容重组为下面结构。

【风格】概念拆原子、每步说为什么、必含ASCII图、适用就给数值手算、公式$...$。
【结构】
# <主题>
> 一句话定位。📍 导航：[[00-知识地图]]
> 🔗 相关：[[相关]] ...(从枢纽选2-4)
## 阅读地图(表格)
## 0. 一句话锚点
## 1. 地基/前置
## 2.~N. 原理+机制(配图; 把原文真实命令/配置编排进来并解释)
## (单独一节)实操/命令/配置(保留原文真料)
## 常见问题/坑(表格)
## 🔗 跳转链接
枢纽: ${HUBS}
【硬要求】不破坏原文真料; 顶部导航+底部双链必须有。返回 JSON {file,lines,keptOriginal,hasAscii,links}。直接开始。`
}

function navPrompt(path) {
  return `这是一个【内容已很丰富的大文件】(真实训练日志/教程/代码), 【绝对不要重写或删减正文】。你只需给它【加上导航与双链】, 用【中文】。

【文件】${path}
【步骤】
1. 用 Read 只读该文件【前 40 行】(offset=0, limit=40), 找到第一行 H1 标题(以 "# " 开头那行)。
2. 用 Edit 工具, 把那一行 H1 标题, 替换为:
   "<原H1标题行>
   > 📍 导航：[[00-知识地图]]
   > 🔗 相关：[[xx]] [[yy]] [[zz]]   ← 根据本文件主题(看路径与标题)从枢纽清单选3-4个最相关的
   > 📖 本文导读：<一句话说明这篇大文档讲了什么、适合什么场景看>"
   即:只在 H1 标题【下方插入】导航/相关/导读三行, 原标题保留、正文其余【一字不动】。
3. 不要做任何其它修改, 不要 Write 整个文件。

【枢纽清单(选相关的)】${HUBS}
【路径主题线索】如 megatron/gpt2/model_train=Megatron GPT2 训练实操; microsoft/llama-note=昇腾?不,是LLaMA训练笔记; peft/README=PEFT微调总览; alpaca/README=Alpaca指令微调。
返回 JSON {file, action:"加导航", links:数量, bodyPreserved:true}。直接开始, 不解释。`
}

const ENRICH = [
  'llm-train/pytorch/distribution/tensor-parallel/README.md','llm-train/qlora/README.md','llm-train/slurm/README.md','llm-train/pytorch/distribution/分布式通信包.md','llm-train/deepspeedchat/README.md','llm-train/pytorch/distribution/data-parallel/使用DDP训练真实世界的模型.md','llm-train/pytorch/distribution/data-parallel/README.md','llm-train/alpaca-lora/README.md','llm-train/megatron/source-code.md','llm-train/pytorch/distribution/pipeline-parallel/1-流水线.md','llm-train/firefly/dockerfile.md','llm-train/paddle/paddlenlp/README.md','llm-train/pytorch/distribution/pipeline-parallel/2-使用torchtext训练transformer模型.md','llm-train/pytorch/api.md','llm-train/peft/LoRA-QLoRA.md','llm-train/pytorch/distribution/pipeline-parallel/README.md',
]
const NAV = [
  'llm-train/megatron-deepspeed/microsoft/README.md','llm-train/pytorch/distribution/data-parallel/minGPT-ddp/README.md','llm-train/megatron-deepspeed/bloom-megatron-deepspeed.md','llm-train/megatron/project.md','llm-train/megatron/gpt2/gpt-data-preprocess.md','llm-train/megatron-deepspeed/microsoft/项目结构-202312228.md','llm-train/alpaca/README.md','llm-train/peft/README.md','llm-train/megatron-deepspeed/microsoft/llama-note.md','llm-train/megatron/gpt2/model_merge_eval_inference.md','llm-train/megatron/gpt2/model_train.md',
]

phase('E-薄中文件重构补齐')
const re = await parallel(ENRICH.map(p => () => agent(enrichPrompt(p), { label: `enrich:${p}`, phase: 'E-薄中文件重构补齐', schema: ENRICH_SCHEMA })))
log(`E: ${re.filter(Boolean).length}/${ENRICH.length}`)

phase('N-厚巨文件加导航')
const rn = await parallel(NAV.map(p => () => agent(navPrompt(p), { label: `nav:${p}`, phase: 'N-厚巨文件加导航', schema: NAV_SCHEMA })))
log(`N: ${rn.filter(Boolean).length}/${NAV.length}`)

return { enriched: re.filter(Boolean).length, navAdded: rn.filter(Boolean).length, total: ENRICH.length + NAV.length }
