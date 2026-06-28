export const meta = {
  name: 'llmaction-检查-全库其余122',
  description: '全库其余122未改造: 63薄重构补齐(保真料)+59厚仅加导航(保原文)',
  phases: [
    { title: 'E1-重构', detail: '' }, { title: 'E2-重构', detail: '' }, { title: 'E3-重构', detail: '' },
    { title: 'N1-加导航', detail: '' }, { title: 'N2-加导航', detail: '' }, { title: 'N3-加导航', detail: '' },
  ],
}

const DIR = 'C:/Users/jianm/Desktop/llm-action/'
const HUBS = '00-知识地图 / llm-algo/transformer/模型架构 / llm-algo/moe/README / llm-algo/旋转编码RoPE / llm-optimizer/FlashAttention / llm-optimizer/kv-cache / llm-inference/README / llm-inference/vllm/README / llm-inference/解码策略 / llm-inference/PD分离 / llm-compression/README / llm-compression/quantization/量化基础 / llm-compression/quantization/GPTQ / llm-compression/quantization/fp8 / llm-train/README / llm-train/peft/PEFT-API / llm-alignment/RLHF / llm-alignment/DPO / ai-framework/deepspeed/README / ai-framework/megatron-lm/README / ai-framework/pytorch/README / ai-infra/算力/GPU工作原理 / ai-infra/ai-hardware/硬件对比 / ai-infra/网络/InfiniBand / ai-infra/网络/集合通信原语 / ai-infra/算力/昇腾NPU / ai-infra/ai-hardware/AI芯片软件生态 / llm-eval/README / llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释 / docs/transformer内存估算'

const E_SCHEMA = { type: 'object', properties: { file: { type: 'string' }, lines: { type: 'number' }, keptOriginal: { type: 'boolean' }, hasAscii: { type: 'boolean' }, links: { type: 'number' } }, required: ['file', 'lines', 'keptOriginal', 'hasAscii', 'links'] }
const N_SCHEMA = { type: 'object', properties: { file: { type: 'string' }, action: { type: 'string' }, links: { type: 'number' }, bodyPreserved: { type: 'boolean' } }, required: ['file', 'action', 'links', 'bodyPreserved'] }

function enrichPrompt(path) {
  return `你是顶尖 AI-Infra 讲师。把 llm-action 仓库一个【已有部分内容但未达深度笔记标准】的文件提升为完整深度笔记，用【中文】，Write 覆盖写回。
【文件】${path}
【第一步】先 Read ${DIR}${path}, 找出其中【真实有价值信息】(真实命令/配置/代码/参数/数据/踩坑)。
【第二步】用 Write 写回 ${DIR}${path}:
- 【保留原文所有真实可用信息】(命令/配置/代码原样或更清晰), 在其上【补全】原理/为什么/ASCII图/对照表/数值示例/常见坑, 重组为下面结构。
- 【不删真实信息, 不编造新命令/版本/参数】。主题以路径与原文为准。
【风格】概念拆原子、每步说为什么、必含ASCII图、适用就给数值手算、公式$...$。
【结构】
# <主题>
> 一句话定位。📍 导航：[[00-知识地图]]
> 🔗 相关：[[相关]] ...(从枢纽选2-4)
## 阅读地图(表格)
## 0. 一句话锚点
## 1. 地基/前置
## 2.~N. 原理+机制(配图; 原文真实命令/配置编排进来并解释)
## (单独一节)实操/命令/配置(保留原文真料)
## 常见问题/坑(表格)
## 🔗 跳转链接
枢纽: ${HUBS}
【硬要求】不破坏原文真料; 顶部导航+底部双链必须有; 220~430行。返回 JSON {file,lines,keptOriginal,hasAscii,links}。直接开始。`
}

function navPrompt(path) {
  return `这是一个【内容已很丰富的大文件】, 【绝对不要重写或删减正文】, 只给它【加导航与双链】, 用【中文】。
【文件】${path}
【步骤】
1. 用 Read 只读【前40行】(offset=0,limit=40)。
2. 确定插入点:若前40行有第一个 "# " H1 标题行 → 在其【下方】插入;若无H1 → 在【文件第一行之前】插入。
3. 用 Edit:把锚点行(H1行 或 文件第一行)替换为 "<锚点行>\n> 📍 导航：[[00-知识地图]]\n> 🔗 相关：[[xx]] [[yy]] [[zz]]\n> 📖 导读：<一句话:这篇大文档讲什么、适合什么场景看>"(H1情形把导航放锚点行下方; 无H1情形把导航块放在原第一行之前)。
4. 相关双链根据本文主题(看路径与标题)从枢纽清单选3-4个最相关的。其余正文【一字不动】, 不要 Write 整个文件。
【枢纽清单】${HUBS}
返回 JSON {file, action:"加导航", links:数量, bodyPreserved:true}。直接开始, 不解释。`
}

const ENRICH = [
  'llm-tools/nsight/README.md','llm-algo/transformer.md','llm-eval/EvalScope.md','llm-localization/ascend/ascend-infra/docker环境升级cann.md','llm-tools/README.md','llm-compression/quantization/fp6.md','docs/llm-base/monitor.md','docs/llm-base/distribution-training/GLM-130B训练经验.md','llm-localization/ascend/mindie/README.md','docs/llm-base/distribution-parallelism/README.md','docs/llm-base/scenes/multi-modal/README.md','ai-infra/网络/Spine-Leaf和InfiniBand网络架构区别简述.md','docs/llm-summarize/领域大模型.md','docs/llm-peft/LoRA-FA.md','docs/llm-base/机器学习中常用的数据类型.md','llm-compression/quantization/README.md','docs/llm-base/slurm.md','ai-infra/算力/NVIDIA-GPU型号.md','llm-eval/opencompass.md','docs/llm-inference/llm推理优化技术.md','llm-compression/quantization/kv-cache-quant.md',
  'llm-interview/llm-train.md','docs/llm-inference/flexflow/投机采样.md','llm-alignment/基本概念.md','llm-eval/llm-performance/llmperf.md','llm-interview/llm-inference.md','llm-localization/ascend/mindie/model-test.md','ai-infra/存储/固态硬盘.md','ai-infra/ai-hardware/gpudirect.md','ai-framework/deepspeed/hello_bert/README.md','docs/llm-base/distribution-parallelism/moe-parallel/README.md','llm-localization/ascend/mindie/mindie-1.0.RC2.md','llm-compression/quantization/可视化/README.md','docs/llm-base/h800-env-install.md','ai-infra/算力/AI芯片.md','llm-inference/tensorrt-llm/README.md','llm-localization/ascend/mindie/mindie-1.0.md','llm-eval/llm-precision/README.md','llm-inference/tensorrt-llm/安装.md','llm-compression/distillation/README.md','ai-framework/llama-cpp/README.md','ai-framework/deepspeed/1.DeepSpeed入门.md',
  'llm-compression/大模型压缩综述.md','llm-inference/vllm/服务启动参数.md','llm-pipeline/REAEMD.md','docs/llm-base/nvidia-smi-dmon.md','llm-localization/ascend/ascend-infra/ascend-llm下载.md','docs/llm-base/distribution-parallelism/auto-parallel/Galvatron.md','llm-algo/README.md','llm-inference/FlexFlow-Serve.md','docs/llm-base/distribution-parallelism/auto-parallel/README.md','llm-compression/quantization/大模型量化概述.md','llm-localization/ascend/mindspore/镜像.md','llm-tools/Pytorch-Profiler.md','llm-compression/distillation/大模型蒸馏概述.md','llm-inference/faster-transformer/bloom/README.md','llm-localization/ascend/ascend-infra/ascend-npu-smi.md','ai-framework/deepspeed/config-json/README.md','faq/FAQ.md','llm-localization/ascend/modellink/环境-20240521.md','llm-eval/llm-precision/模型质量评估.md','llmops/千帆大模型平台.md','docs/llm-base/distribution-parallelism/auto-parallel/Flexflow.md',
]

const NAV = [
  'llm-compression/quantization/llm-qat/README.md','llm-localization/ascend/mindie/mindid-1.0-offical.md','llm-eval/llm-performance/训练性能测试.md','llm-compression/quantization/llm-qat/LLM-QAT.md','ai-compiler/README.md','llm-eval/llm-performance/mindie/locust-lantency-throughput/README.md','paper/data/LESS 实践：仅用少量的数据完成目标指令微调.md','ai-framework/deepspeed/2.安装DeepSpeed.md','docs/llm-base/distribution-parallelism/multidimensional-hybrid-parallel/README.md','docs/llm-base/distribution-training/OPT-175B训练经验.md','blog/llm-peft/大模型参数高效微调技术原理综述（一）-背景、参数高效微调简介.md','llm-localization/ascend/mindformers/权重格式转换.md','llm-localization/ascend/standford-alpaca/README.md','blog/distribution-parallelism/大模型分布式训练并行技术（一）-概述.md','docs/llm-base/a800-env-install.md','docs/llm-inference/blog.md','llm-localization/ascend/mindie/mindie-api.md','llm-localization/ascend/昇腾LLM支持概览.md','llm-inference/lmdeploy/服务启动参数.md','llm-inference/faster-transformer/README.md',
  'llm-localization/ascend/mindformers/README.md','ai-framework/deepspeed/3.基于CIFAR-10使用DeepSpeed进行分布式训练 .md','blog/distribution-parallelism/大模型分布式训练并行技术（九）-总结.md','ai-framework/pai-megatron-patch/README.md','llm-compression/quantization/ZeroQuant(4+2).md','blog/distribution-parallelism/大模型分布式训练并行技术（六）-多维混合并行.md','llm-algo/bert/模型架构.md','llm-localization/ascend/mindie/docker/README.md','blog/llm-peft/大模型参数高效微调技术原理综述（五）-LoRA、AdaLoRA、QLoRA.md','docs/llm-base/distribution-parallelism/auto-parallel/Alpa.md','docs/llm-base/scenes/README.md','llm-localization/ascend/mindie/mindie-20240411.md','llm-localization/ascend/ascend910-env-install.md','docs/llm-base/autoregressive-lm-decoding-methods.md','docs/llm-base/distribution-parallelism/auto-parallel/飞桨面向异构场景下的自动并行设计与实践.md','ai-framework/transformer-engine/mnist/README.md','blog/llm-localization/大模型国产化适配1-华为昇腾AI全栈软硬件平台总结.md','paper/A Survey on Efficient Training of Transformers.md','llm-eval/llm-performance/mindie/lantency/README.md','blog/llm-inference/大模型推理框架概述.md',
  'blog/llm-localization/大模型国产化适配4-基于昇腾910使用LLaMA-13B进行多机多卡训练.md','llm-compression/quantization/ZeroQuant.md','docs/llm-base/nvidia-smi.md','llm-data-engineering/sft-dataset/数据集格式.md','ai-infra/网络/nvbandwidth.md','paper/data/LESS.md','llm-algo/chatglm/模型架构.md','llm-localization/ascend/mindie/2.0.RC2/qwen.md','docs/llm-summarize/大模型实践总结-20230930.md','llm-compression/quantization/FP6-LLM.md','blog/llm-compression/大模型量化技术原理：QoQ量化及QServe推理服务系统.md','llm-localization/ascend/mindformers/chatglm/README.md','docs/llm-summarize/大模型实践总结.md','README.md','llm-algo/chatglm2/模型架构.md','blog/ai-infra/AI 集群基础设施 InfiniBand 详解.md','blog/ai-infra/AI 集群基础设施 NVMe SSD 详解.md','llm-inference/faster-transformer/megatron-gpt2/megatron-gpt2-fp8.md','docs/llm-base/dcgmi.md',
]

const runE = (arr, ph) => parallel(arr.map(p => () => agent(enrichPrompt(p), { label: `enrich:${p}`, phase: ph, schema: E_SCHEMA })))
const runN = (arr, ph) => parallel(arr.map(p => () => agent(navPrompt(p), { label: `nav:${p}`, phase: ph, schema: N_SCHEMA })))

phase('E1-重构'); const e1 = await runE(ENRICH.slice(0,21), 'E1-重构'); log(`E1:${e1.filter(Boolean).length}/21`)
phase('E2-重构'); const e2 = await runE(ENRICH.slice(21,42), 'E2-重构'); log(`E2:${e2.filter(Boolean).length}/21`)
phase('E3-重构'); const e3 = await runE(ENRICH.slice(42), 'E3-重构'); log(`E3:${e3.filter(Boolean).length}/${ENRICH.length-42}`)
phase('N1-加导航'); const n1 = await runN(NAV.slice(0,20), 'N1-加导航'); log(`N1:${n1.filter(Boolean).length}/20`)
phase('N2-加导航'); const n2 = await runN(NAV.slice(20,40), 'N2-加导航'); log(`N2:${n2.filter(Boolean).length}/20`)
phase('N3-加导航'); const n3 = await runN(NAV.slice(40), 'N3-加导航'); log(`N3:${n3.filter(Boolean).length}/${NAV.length-40}`)

const enr = [...e1,...e2,...e3].filter(Boolean), nav = [...n1,...n2,...n3].filter(Boolean)
return { enriched: enr.length, navAdded: nav.length, total: ENRICH.length + NAV.length }
