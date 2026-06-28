export const meta = {
  name: 'llmaction-补齐-run11-国产化',
  description: 'llm-action补齐llm-localization/: 昇腾国产化50篇(机制/对标CUDA/迁移要点,不编命令)+双链',
  phases: [
    { title: 'L1-昇腾架构与基础设施', detail: '达芬奇/CANN/HCCL/镜像/监控等 17篇' },
    { title: 'L2-框架与训练栈', detail: 'MindSpore/MindFormers/ModelLink/训练 17篇' },
    { title: 'L3-推理压缩与其它国产', detail: 'MindIE/msmodelslim/vllm-ascend/天数/modelscope 16篇' },
  ],
}

const DIR = 'C:/Users/jianm/Desktop/llm-action/'
const SCHEMA = { type: 'object', properties: { file: { type: 'string' }, lines: { type: 'number' }, hasAscii: { type: 'boolean' }, links: { type: 'number' } }, required: ['file', 'lines', 'hasAscii', 'links'] }

const HUBS = '00-知识地图 / ai-infra/算力/昇腾NPU / ai-infra/ai-hardware/AI芯片软件生态 / ai-infra/ai-hardware/CUDA / ai-infra/网络/NCCL / ai-infra/网络/集合通信原语 / ai-framework/megatron-lm/README / ai-framework/huggingface-transformers/README / llm-compression/quantization/量化基础 / llm-inference/README / llm-train/README / llm-algo/transformer/模型架构'

function buildPrompt(path) {
  return `你是顶尖 AI-Infra 讲师, 精通昇腾(Ascend)国产化生态。把 llm-action 仓库 llm-localization/ 下一个【薄文件】补齐, 用【中文】, 并用 Write 工具覆盖写回原路径。

【文件】${path}
【主题来自路径】例:"ascend/ascend-infra/达芬奇架构.md"=昇腾达芬奇架构;"ascend/ascend-infra/HCCL.md"=HCCL集合通信;"ascend/mindformers/..."=MindFormers训练套件;"ascend/mindie/..."=MindIE推理引擎;"ascend/msmodelslim/..."=昇腾量化工具;"ascend/modellink/..."=ModelLink训练;"tianshuzhixin/..."=天数智芯;"modelscope/..."=魔搭。
【第一步】Read ${DIR}${path} 看已有方向并尊重;近空按主题从零写。
【第二步】用 Write 写到绝对路径：${DIR}${path}

【这是国产化/部署主题——核心讲法】
1. 【是什么+在昇腾栈里的定位】这个组件/概念解决什么、处于昇腾软件栈(硬件→CANN→框架→套件)的哪一层。
2. 【对标 CUDA 世界谁】给一张"昇腾 ↔ 英伟达生态"对照(如 NPU↔GPU、CANN↔CUDA、HCCL↔NCCL、CANN算子↔cuDNN/cuBLAS、MindSpore↔PyTorch、MindFormers↔Megatron/HF、MindIE↔TensorRT-LLM/vLLM、msmodelslim↔GPTQ/AWQ工具、ModelLink↔Megatron)。这是读者最需要的迁移心智图。
3. 【机制/原理】达芬奇Cube/Vector单元、图模式、HCCL环算法等讲原理。
4. 【迁移要点与常见坑】从CUDA迁到昇腾要改什么、易踩的坑、性能调优思路(机制层面)。

【极重要护栏】
- 【绝不编造】精确命令行/包名/版本号/路径/确切性能数字。环境安装/docker/镜像/配置类:只讲【整体流程步骤的含义、依赖关系、注意点、常见坑】, 凡涉及具体命令/版本一律写"具体命令与版本以华为昇腾官方文档(Ascend社区)为准"。宁可讲清"为什么要这步"也不要造假命令。

【固定结构】
# <主题>
> 一句话定位。📍 导航：[[00-知识地图]]
> 🔗 相关：[[ai-infra/算力/昇腾NPU]] [[ai-infra/ai-hardware/AI芯片软件生态]] ...(选相关)
## 阅读地图(表格)
## 0. 一句话锚点
## 1. 地基:在昇腾栈的定位 + 对标英伟达生态(对照表)
## 2.~N. 机制/原理/流程(配ASCII图)
## (单独一节)迁移要点 / 注意事项与坑
## 常见问题(表格)
## 🔗 跳转链接
枢纽清单: ${HUBS}

【硬要求】180~360 行; 顶部导航[[00-知识地图]]+底部双链必须有; 含至少1个ASCII图和1个"昇腾↔英伟达"对照表。返回 JSON {file,lines,hasAscii,links}。直接开始。`
}

const FILES = [
  // L1 架构与基础设施
  'llm-localization/README.md','llm-localization/ascend/README.md','llm-localization/ascend/FAQ.md','llm-localization/ascend/ascend-c/README.md','llm-localization/ascend/ascend-infra/达芬奇架构.md','llm-localization/ascend/ascend-infra/HCCL.md','llm-localization/ascend/ascend-infra/network.md','llm-localization/ascend/ascend-infra/npu监控.md','llm-localization/ascend/ascend-infra/ascend-dmi.md','llm-localization/ascend/ascend-infra/昇腾卡-soc版本.md','llm-localization/ascend/ascend-infra/昇腾卡注意事项.md','llm-localization/ascend/ascend-infra/昇腾镜像.md','llm-localization/ascend/ascend-infra/ascend-docker.md','llm-localization/ascend/ascend-infra/ascend-docker-runtime.md','llm-localization/ascend/ascend-infra/操作系统.md','llm-localization/ascend/ascend-infra/服务器配置.md','llm-localization/ascend/ascend-infra/环境安装.md',
  // L2 框架与训练栈
  'llm-localization/ascend/ascend-infra/MacOS环境.md','llm-localization/ascend/mindspore/README.md','llm-localization/ascend/mindspore/MindSpore-note.md','llm-localization/ascend/mindspore/bert.md','llm-localization/ascend/mindformers/env.md','llm-localization/ascend/mindformers/trick.md','llm-localization/ascend/mindformers/llama/README.md','llm-localization/ascend/mindformers/baichuan2/baichuan2训练.md','llm-localization/ascend/mindformers/qwen/qwen1训练.md','llm-localization/ascend/mindformers/qwen1.5/qwen1.5训练.md','llm-localization/ascend/modellink/README.md','llm-localization/ascend/modellink/llm.md','llm-localization/ascend/modellink/qwen.md','llm-localization/ascend/modellink/dataset.md','llm-localization/ascend/modellink/环境安装.md','llm-localization/ascend/pytorch/README.md','llm-localization/ascend/transformers/README.md',
  // L3 推理压缩与其它国产
  'llm-localization/ascend/mindie/mindie-2.0.rc2.md','llm-localization/ascend/mindie/性能调优.md','llm-localization/ascend/mindie/mindid-performance.md','llm-localization/ascend/mindie/docker/TEST.md','llm-localization/ascend/msmodelslim/README.md','llm-localization/ascend/vllm-ascend/README.md','llm-localization/ascend/peft/README.md','llm-localization/ascend/openmind/README.md','llm-localization/ascend/firefly-ascend.md','llm-localization/ascend/fabric-insight/README.md','llm-localization/ascend/优质学习资料.md','llm-localization/ascend/昇腾卡注意事项.md','llm-localization/modelscope/README.md','llm-localization/paddle/PaddleNLP.md','llm-localization/tianshuzhixin/README.md','llm-localization/tianshuzhixin/ixsmi.md',
]

const runWave = (paths, phaseTitle) =>
  parallel(paths.map(p => () => agent(buildPrompt(p), { label: `loc:${p}`, phase: phaseTitle, schema: SCHEMA })))

const b1 = FILES.slice(0, 17), b2 = FILES.slice(17, 34), b3 = FILES.slice(34)
phase('L1-昇腾架构与基础设施'); const r1 = await runWave(b1, 'L1-昇腾架构与基础设施'); log(`L1: ${r1.filter(Boolean).length}/${b1.length}`)
phase('L2-框架与训练栈'); const r2 = await runWave(b2, 'L2-框架与训练栈'); log(`L2: ${r2.filter(Boolean).length}/${b2.length}`)
phase('L3-推理压缩与其它国产'); const r3 = await runWave(b3, 'L3-推理压缩与其它国产'); log(`L3: ${r3.filter(Boolean).length}/${b3.length}`)

const all = [...r1, ...r2, ...r3].filter(Boolean)
return { total: FILES.length, written: all.length, files: all.map(x => x.file) }
