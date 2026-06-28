export const meta = {
  name: 'llmaction-补齐-run5',
  description: 'llm-action补齐run5: 28篇(框架/AI集群网络/模型变体/蒸馏量化补全)填到最大细节+双链',
  phases: [
    { title: 'W9-框架与AI集群网络', detail: 'JAX/Paddle/OneFlow/TF/TRL/集群/GPU网络/存储 14篇' },
    { title: 'W10-模型变体与压缩补全', detail: 'GPT/ChatGPT/ChatGLM/GLM-130B/InternLM/蒸馏/量化 14篇' },
  ],
}

const DIR = 'C:/Users/jianm/Desktop/llm-action/'
const SCHEMA = { type: 'object', properties: { file: { type: 'string' }, lines: { type: 'number' }, hasHandCalc: { type: 'boolean' }, hasAscii: { type: 'boolean' }, links: { type: 'number' } }, required: ['file', 'lines', 'hasHandCalc', 'hasAscii', 'links'] }

function buildPrompt(n) {
  const rel = n.related.map(r => `[[${r}]]`).join(' ')
  return `你是顶尖 AI-Infra 讲师。把 llm-action 仓库里一个【薄文件】补齐到【最大细节】，用【中文】，并用 Write 工具覆盖写回原路径。

【文件(相对仓库根)】${n.file}
【主题】${n.title}
【必须覆盖】${n.cover}
【相关链接】${rel}

【第一步】先 Read ${DIR}${n.file}，尊重已有方向;近空则从零写。
【第二步】用 Write 写到绝对路径：${DIR}${n.file}

【风格——"从最底层讲清 + 可视化"】
- 不假设记得公式;概念拆到最原子;每步说"为什么"。
- 【必含 ASCII 可视化图】(架构/数据流/结构/对比)。
- 适用就给【数值例子/手算】;框架/硬件类给典型公开规格并标"约/以官方为准", 不编造未知精确数字。
- 公式行内 $...$、独立 $$...$$。
- 框架类: 讲清【解决什么问题、整体架构、核心机制、与同类对比、何时用、权衡】。
- 模型变体类: 讲清【相对前代/同类的关键创新、架构、规模、训练、贡献】, 数字标"约/以官方为准"。
- 不确定的版本/CLI/API 标"以官方文档为准", 不编造。

【固定结构】
# ${n.title}
> 一句话定位。📍 导航：[[00-知识地图]]
> 🔗 相关：${rel}
## 阅读地图(表格)
## 0. 一句话锚点
## 1. 地基/前置
## 2.~N. 逐步拆解(每节配ASCII图)
## (单独一节)数值例子/对照
## 常见问题(表格)
## 🔗 跳转链接

【硬要求】250~430 行, 高密度; 顶部导航+底部双链必须有。返回 JSON {file,lines,hasHandCalc,hasAscii,links}。直接开始。`
}

const W9 = [
  { file: 'ai-framework/jax/README.md', title: 'JAX', cover: '函数式+可组合变换(grad/jit/vmap/pmax);XLA编译;为什么科研友好;与PyTorch对比;Flax/Haiku;TPU亲和', related: ['00-知识地图','ai-framework/pytorch/README','ai-framework/README'] },
  { file: 'ai-framework/paddlepaddle/README.md', title: 'PaddlePaddle', cover: '百度深度学习框架;动静统一;分布式训练(Fleet);PaddleNLP大模型套件;国产生态;与PyTorch对比', related: ['00-知识地图','ai-framework/pytorch/README','ai-framework/README'] },
  { file: 'ai-framework/oneflow/README.md', title: 'OneFlow', cover: 'SBP抽象统一数据/模型并行;Global Tensor;一致性视角;与Megatron/DeepSpeed对比;分布式易用性', related: ['00-知识地图','ai-framework/megatron-lm/README','ai-framework/README'] },
  { file: 'ai-framework/tensorflow/README.md', title: 'TensorFlow', cover: '静态图与eager;Keras;TF分布式策略;TPU;历史地位与现状;与PyTorch对比', related: ['00-知识地图','ai-framework/pytorch/README','ai-framework/README'] },
  { file: 'ai-framework/megatron-deepspeed/README.md', title: 'Megatron-DeepSpeed (框架)', cover: '把Megatron的TP/PP与DeepSpeed的ZeRO融合;3D并行;BigScience用它训BLOOM;配置;定位', related: ['00-知识地图','ai-framework/megatron-lm/README','ai-framework/deepspeed/README','llm-train/megatron-deepspeed/README'] },
  { file: 'ai-framework/huggingface-trl/README.md', title: 'HuggingFace TRL (对齐训练库)', cover: 'SFTTrainer/RewardTrainer/PPOTrainer/DPOTrainer/GRPOTrainer;与PEFT/Accelerate集成;对齐流程落地', related: ['00-知识地图','llm-alignment/RLHF','llm-alignment/DPO','ai-framework/huggingface-peft/README'] },
  { file: 'ai-framework/unsloth-微调.md', title: 'Unsloth (高效微调)', cover: '手写Triton kernel加速LoRA微调;显存与速度优势;支持的模型;原理(融合kernel/优化反向);何时用', related: ['00-知识地图','ai-framework/huggingface-peft/README','ai-framework/openai-triton/README','llm-train/peft/PEFT-API'] },
  { file: 'ai-framework/TensorRT-Model-Optimizer.md', title: 'TensorRT Model Optimizer', cover: 'NVIDIA量化/稀疏/蒸馏工具库;PTQ/QAT;导出到TRT-LLM;支持FP8/INT4;与TRT-LLM协作;流程', related: ['00-知识地图','llm-compression/quantization/量化基础','llm-inference/DeepSpeed-Inference'] },
  { file: 'ai-compiler/triton-lang/README.md', title: 'Triton 语言与编译器', cover: 'block级GPU编程模型;自动处理共享内存/合并访存;编译到PTX;写FlashAttention类kernel;与CUDA对比;生态', related: ['00-知识地图','ai-framework/openai-triton/README','ai-infra/ai-hardware/CUDA','llm-optimizer/FlashAttention'] },
  { file: 'ai-infra/ai-cluster/README.md', title: 'AI 训练集群', cover: '集群组成(GPU节点/网络/存储/调度);Slurm/K8s调度;网络拓扑(胖树/rail);故障容错;利用率;万卡集群挑战', related: ['00-知识地图','ai-infra/网络/InfiniBand','ai-infra/网络/网络硬件','llmops/kubernetes'] },
  { file: 'ai-infra/ai-hardware/GPU-network.md', title: 'GPU 互联 (NVLink/NVSwitch)', cover: 'NVLink点对点带宽;NVSwitch全互联;单机8卡拓扑;NVLink vs PCIe;为什么TP需要高带宽互联;跨机IB', related: ['00-知识地图','ai-infra/ai-hardware/硬件对比','ai-infra/网络/InfiniBand','ai-infra/网络/集合通信原语'] },
  { file: 'ai-infra/ai-hardware/AI芯片软件生态.md', title: 'AI 芯片软件生态', cover: 'CUDA护城河;cuDNN/cuBLAS/NCCL;为什么生态比硬件更难追;国产(CANN/ROCm)迁移挑战;算子覆盖;编译器路线', related: ['00-知识地图','ai-infra/ai-hardware/CUDA','ai-infra/算力/昇腾NPU','ai-compiler/triton-lang/README'] },
  { file: 'ai-infra/网络/网络硬件.md', title: '数据中心网络硬件', cover: '网卡NIC/交换机/光模块/线缆;IB vs 以太网交换机;400G/800G;胖树多轨;为什么网络是训练瓶颈;成本', related: ['00-知识地图','ai-infra/网络/InfiniBand','ai-infra/网络/roce','ai-infra/ai-cluster/README'] },
  { file: 'ai-infra/存储/nvme-ssd.md', title: 'NVMe SSD', cover: 'NVMe vs SATA;PCIe通道;IOPS/带宽/延迟;在AI里的角色(数据加载/checkpoint/KV offload);为什么快;选型', related: ['00-知识地图','ai-infra/存储/存储','llm-inference/offload'] },
]

const W10 = [
  { file: 'llm-algo/gpt/README.md', title: 'GPT (初代)', cover: 'GPT-1生成式预训练+判别式微调范式;Decoder-only起点;与ELMo/BERT路线对比;贡献', related: ['00-知识地图','llm-algo/gpt2/模型架构','llm-algo/bert'] },
  { file: 'llm-algo/gpt2/README.md', title: 'GPT-2 总览', cover: 'GPT-2规模与zero-shot;WebText;为什么"语言模型是无监督多任务学习者";与GPT-1/3关系', related: ['00-知识地图','llm-algo/gpt2/模型架构','llm-algo/gpt3/README'] },
  { file: 'llm-algo/chatgpt/README.md', title: 'ChatGPT / InstructGPT', cover: 'InstructGPT三阶段RLHF;为什么对齐让模型可用;与GPT-3差异;PPO;贡献与影响', related: ['00-知识地图','llm-alignment/RLHF','llm-algo/gpt3/README'] },
  { file: 'llm-algo/chatglm2/README.md', title: 'ChatGLM2', cover: 'ChatGLM2架构改进(GQA/FlashAttn/更长上下文);相对ChatGLM1;中文优化;训练', related: ['00-知识地图','llm-algo/chatglm/README','llm-algo/chatglm3/README'] },
  { file: 'llm-algo/chatglm3/README.md', title: 'ChatGLM3', cover: 'ChatGLM3能力(工具调用/代码/Agent);对话格式;相对ChatGLM2改进;部署', related: ['00-知识地图','llm-algo/chatglm2/README','llm-algo/glm4'] },
  { file: 'llm-algo/glm-130b/README.md', title: 'GLM-130B', cover: '130B双语开源;GLM训练目标;后归一化/缩放;INT4量化部署;训练稳定性经验;意义', related: ['00-知识地图','llm-algo/chatglm/README','llm-algo/bloom'] },
  { file: 'llm-algo/InternLM-20B.md', title: 'InternLM', cover: 'InternLM架构与规模;长上下文;训练数据;工具链(InternEvo);中文;特点', related: ['00-知识地图','llm-algo/llama/模型架构','llm-algo/qwen/README'] },
  { file: 'llm-algo/deepseek/README.md', title: 'DeepSeek 系列总览', cover: 'DeepSeek LLM→V2(MLA/MoE)→V3→R1演进主线;核心创新串讲;开源与影响;为什么经济高效', related: ['00-知识地图','llm-algo/deepseek/DeepSeek-V2','llm-algo/deepseek/DeepSeek-V3','llm-algo/deepseek/DeepSeek-R1'] },
  { file: 'llm-compression/distillation/GKD.md', title: 'GKD (广义知识蒸馏)', cover: 'on-policy蒸馏解决曝光偏差;学生自生成序列上蒸馏;广义JS散度;与序列级KD对比;为什么更好', related: ['00-知识地图','llm-compression/distillation/MINILLM','llm-compression/README'] },
  { file: 'llm-compression/distillation/SCOTT.md', title: 'SCOTT (思维链蒸馏)', cover: '把大模型推理链CoT蒸馏到小模型;一致性约束;反事实推理;为什么蒸推理能力;与R1蒸馏对比', related: ['00-知识地图','llm-compression/distillation/MINILLM','llm-algo/deepseek/DeepSeek-R1'] },
  { file: 'llm-compression/quantization/QQQ-W4A8.md', title: 'QQQ (W4A8 量化)', cover: 'W4A8权重4bit激活8bit;为什么W4A8平衡精度/速度;离群处理;GEMM kernel;与W4A16/W8A8对比', related: ['00-知识地图','llm-compression/quantization/量化基础','llm-compression/quantization/SmoothQuant','llm-compression/quantization/GPTQ'] },
  { file: 'llm-compression/quantization/PEQA.md', title: 'PEQA (量化感知微调)', cover: '参数高效+量化结合;只微调量化scale;冻结量化权重;省显存又适配;与QLoRA对比', related: ['00-知识地图','llm-compression/quantization/量化基础','llm-train/peft/PEFT-API'] },
  { file: 'llm-compression/llm-compressor/量化方案.md', title: 'llm-compressor 量化方案', cover: 'vLLM生态量化工具;支持GPTQ/AWQ/SmoothQuant/FP8;recipe配置思路;导出兼容vLLM;流程(讲机制不编CLI)', related: ['00-知识地图','llm-compression/quantization/GPTQ','llm-compression/quantization/量化基础','llm-inference/vllm/README'] },
  { file: 'llm-compression/gptqmodel/README.md', title: 'GPTQModel', cover: 'GPTQ量化工具(AutoGPTQ后继);支持多模型/多bit;打包与推理;与llm-compressor/AWQ对比;流程', related: ['00-知识地图','llm-compression/quantization/GPTQ','llm-compression/llm-compressor/量化方案'] },
]

const runWave = (notes, phaseTitle) =>
  parallel(notes.map(n => () => agent(buildPrompt(n), { label: `fill:${n.file}`, phase: phaseTitle, schema: SCHEMA })))

phase('W9-框架与AI集群网络')
const r9 = await runWave(W9, 'W9-框架与AI集群网络')
log(`W9 完成: ${r9.filter(Boolean).length}/${W9.length}`)
phase('W10-模型变体与压缩补全')
const r10 = await runWave(W10, 'W10-模型变体与压缩补全')
log(`W10 完成: ${r10.filter(Boolean).length}/${W10.length}`)

const all = [...r9, ...r10].filter(Boolean)
return { total: W9.length + W10.length, written: all.length, files: all.map(x => x.file) }
