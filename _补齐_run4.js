export const meta = {
  name: 'llmaction-补齐-run4',
  description: 'llm-action补齐run4: 28篇(模型变体/硬件网络存储/评测数据/应用优化)填到最大细节+双链',
  phases: [
    { title: 'W7-模型变体与Infra硬件', detail: 'Transformer/GPT3/GLM4/BLOOM/硬件对比/NPU/RoCE/存储 14篇' },
    { title: 'W8-评测数据应用优化', detail: '评测/数据工程/LangChain/向量库/优化补全 14篇' },
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
- 【必含 ASCII 可视化图】(结构/数据流/对比/拓扑)。
- 适用就给【数值例子/手算】(参数量/显存/带宽/吞吐/数据规模等);硬件类给典型公开规格但标"约/以官方为准", 不编造精确未知数字。
- 公式行内 $...$、独立 $$...$$。
- 模型变体类: 讲清【该模型相对前代/同类的关键创新、架构选择、规模、训练数据、贡献与影响】。
- 硬件/评测/数据/应用类: 讲清【是什么、为什么、核心机制/指标/流程、对比、实践要点】这类稳定知识;不确定的版本/参数标注"以官方为准"。

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

const W7 = [
  { file: 'llm-algo/transformer/README.md', title: 'Transformer 总览', cover: '2017原始Transformer;Encoder-Decoder;自注意力为什么取代RNN;并行性;后续Decoder-only演化;影响', related: ['00-知识地图','llm-algo/transformer/模型架构','llm-algo/bert','llm-algo/gpt2/模型架构'] },
  { file: 'llm-algo/transformer/Transformer中FFN的记忆功能.md', title: 'FFN 的记忆功能', cover: 'FFN作为key-value记忆(Geva论文);第一层=pattern检测,第二层=词表分布;为什么FFN存知识;与MoE/知识编辑联系', related: ['00-知识地图','llm-algo/mlp','llm-algo/moe/README'] },
  { file: 'llm-algo/llama.md', title: 'LLaMA 系列总览', cover: 'LLaMA1/2/3/3.1演进;开源影响;架构要点(RoPE/RMSNorm/SwiGLU/GQA);数据与规模;生态', related: ['00-知识地图','llm-algo/llama/模型架构','llm-algo/qwen/README'] },
  { file: 'llm-algo/gpt3/README.md', title: 'GPT-3', cover: '175B规模;in-context learning/few-shot涌现;与GPT-2差异(主要是规模);Scaling效应;影响', related: ['00-知识地图','llm-algo/gpt2/模型架构','llm-algo/基本概念'] },
  { file: 'llm-algo/glm4.md', title: 'GLM-4', cover: 'GLM-4架构与能力;长上下文;与ChatGLM演进;多模态/Agent;中文优化;对比', related: ['00-知识地图','llm-algo/chatglm/README','llm-algo/qwen/README'] },
  { file: 'llm-algo/qwen2.md', title: 'Qwen2 / Qwen2.5', cover: 'Qwen2架构升级;GQA;长上下文;MoE版;多语言;数学/代码;与Qwen差异', related: ['00-知识地图','llm-algo/qwen/README','llm-algo/llama/模型架构'] },
  { file: 'llm-algo/bloom.md', title: 'BLOOM', cover: 'BigScience开源多语言176B;ALiBi位置编码;训练公开;多语言语料;意义', related: ['00-知识地图','llm-algo/gpt3/README','llm-algo/旋转编码RoPE'] },
  { file: 'llm-algo/baichuan2/baichuan.md', title: 'Baichuan2', cover: 'Baichuan2架构;中文优化;7B/13B;RoPE/ALiBi;训练数据;NormHead/max-z loss', related: ['00-知识地图','llm-algo/llama/模型架构','llm-algo/qwen/README'] },
  { file: 'ai-infra/ai-hardware/README.md', title: 'AI 硬件总览', cover: 'GPU/NPU/TPU;训练卡vs推理卡;算力(FP16/BF16/FP8 TFLOPS)/显存/带宽/互联;选型;国产芯片', related: ['00-知识地图','ai-infra/算力/GPU工作原理','ai-infra/ai-hardware/硬件对比','ai-infra/算力/昇腾NPU'] },
  { file: 'ai-infra/ai-hardware/硬件对比.md', title: 'GPU 硬件对比', cover: 'A100/H100/H800/H20/B200等;算力/显存/HBM带宽/NVLink/功耗;为什么H800阉割互联;选型;以官方规格为准', related: ['00-知识地图','ai-infra/ai-hardware/README','ai-infra/算力/GPU工作原理'] },
  { file: 'ai-infra/算力/推理芯片.md', title: '推理芯片', cover: '推理专用芯片vs训练卡;低延迟/能效;Groq/TPU/国产推理卡;为什么推理需求不同(带宽/批);趋势', related: ['00-知识地图','ai-infra/ai-hardware/README','ai-infra/算力/昇腾NPU'] },
  { file: 'ai-infra/算力/昇腾NPU.md', title: '昇腾 NPU (Ascend)', cover: '华为昇腾910/达芬奇架构;CANN软件栈;与CUDA生态对比;MindSpore/MindFormers;迁移要点;国产替代', related: ['00-知识地图','ai-infra/ai-hardware/README','ai-infra/算力/推理芯片'] },
  { file: 'ai-infra/网络/roce.md', title: 'RoCE (RDMA over Ethernet)', cover: 'RoCEv2原理;在以太网上跑RDMA;与IB对比(成本/性能/运维);PFC/ECN无损网络;大规模训练用途', related: ['00-知识地图','ai-infra/网络/InfiniBand','ai-infra/网络/集合通信原语'] },
  { file: 'ai-infra/存储/存储.md', title: 'AI 训练存储', cover: '训练数据/checkpoint/KV的存储需求;并行文件系统(Lustre/GPFS);对象存储;NVMe;checkpoint写放大;带宽瓶颈', related: ['00-知识地图','ai-infra/存储/nvme-ssd','llm-inference/Mooncake'] },
]

const W8 = [
  { file: 'llm-eval/README.md', title: 'LLM 评测总览', cover: '评测维度(知识/推理/代码/对齐/安全);自动指标vs人评vs裁判模型;benchmark地图;污染;性能vs精度两类', related: ['00-知识地图','llm-eval/大模型测评集','llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释'] },
  { file: 'llm-eval/llm-performance/推理性能测试.md', title: '推理性能测试方法', cover: '压测目标(吞吐/延迟分布);并发/请求率;TTFT/TPOT/P99;warmup;工具(vllm bench/genai-perf);如何画吞吐-延迟曲线', related: ['00-知识地图','llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释','llm-eval/llm-performance/vllm-benchmark'] },
  { file: 'llm-eval/llm-precision/C-Eval.md', title: 'C-Eval 中文评测', cover: 'C-Eval中文知识/推理;学科覆盖;few-shot;与MMLU/CMMLU关系;榜单解读;局限', related: ['00-知识地图','llm-eval/大模型测评集','llm-eval/README'] },
  { file: 'llm-eval/llm-performance/AI芯片性能.md', title: 'AI 芯片性能评测', cover: '算力(TFLOPS)/显存带宽/实测MFU;微基准(GEMM/通信);峰值vs实测差距;roofline定位;怎么测', related: ['00-知识地图','ai-infra/算力/GPU工作原理','llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释'] },
  { file: 'llm-data-engineering/README.md', title: 'LLM 数据工程总览', cover: '预训练数据(来源/清洗/去重/配比/质量);SFT数据(指令多样性);为什么数据决定上限;数据闭环;污染', related: ['00-知识地图','llm-data-engineering/dataset/README','llm-data-engineering/sft-dataset/数据格式设计'] },
  { file: 'llm-data-engineering/dataset/README.md', title: '预训练数据集', cover: '常用语料(Common Crawl/C4/RedPajama/中文);清洗流程(去重/质量过滤/去毒);token配比;规模与scaling', related: ['00-知识地图','llm-data-engineering/README','llm-algo/FLOPs'] },
  { file: 'llm-data-engineering/sft-dataset/数据格式设计.md', title: 'SFT 数据格式设计', cover: 'chat模板(system/user/assistant);多轮;loss mask只算回答;特殊token;packing;工具调用格式;质量要点', related: ['00-知识地图','llm-data-engineering/README','llm-train/peft/Prompt-Tuning'] },
  { file: 'llm-application/langchain/README.md', title: 'LangChain', cover: '为什么编排框架;Chain/Agent/Tool/Memory;RAG集成;Prompt模板;LCEL;何时用/不用;批评', related: ['00-知识地图','llm-application/rag/README','llm-application/应用场景'] },
  { file: 'llm-application/vector-db/README.md', title: '向量数据库', cover: 'embedding向量检索;ANN算法(HNSW/IVF/PQ)原理;Faiss/Milvus/Qdrant;召回vs延迟;混合检索;选型', related: ['00-知识地图','llm-application/rag/README','llm-application/embbedding-model'] },
  { file: 'llm-application/embbedding-model.md', title: 'Embedding 模型', cover: '文本embedding原理;对比学习训练;BGE/M3E/GTE;池化;相似度;为什么RAG核心;评测MTEB;选型', related: ['00-知识地图','llm-application/vector-db/README','llm-application/rag/embedding'] },
  { file: 'llm-application/rag/方案.md', title: 'RAG 工程方案', cover: '切块策略(固定/语义/重叠);多路召回;重排rerank;query改写;父子块;混合检索;评估;常见坑', related: ['00-知识地图','llm-application/rag/README','llm-application/rag/存在的一些问题','llm-application/vector-db/README'] },
  { file: 'llm-optimizer/README.md', title: '推理优化总览', cover: '优化全景:算子(FlashAttn)/内存(KV/Paged)/批处理/并行/解码(投机)/重叠;按瓶颈选;组合拳', related: ['00-知识地图','llm-optimizer/FlashAttention','llm-optimizer/kv-cache','llm-optimizer/计算通信重叠'] },
  { file: 'llm-optimizer/xformers.md', title: 'xFormers', cover: 'Meta注意力优化库;memory-efficient attention;与FlashAttention关系;可组合块;何时用', related: ['00-知识地图','llm-optimizer/FlashAttention','llm-inference/FlashInfer'] },
  { file: 'llm-train/peft/PEFT-API.md', title: 'PEFT 实战要点', cover: 'LoRA配置(r/alpha/target_modules);QLoRA 4bit+LoRA;合并权重;多adapter;训练显存账;常见坑', related: ['00-知识地图','llm-train/peft/Prompt-Tuning','llm-train/peft/Prefix-Tuning','ai-framework/huggingface-peft/README'] },
]

const runWave = (notes, phaseTitle) =>
  parallel(notes.map(n => () => agent(buildPrompt(n), { label: `fill:${n.file}`, phase: phaseTitle, schema: SCHEMA })))

phase('W7-模型变体与Infra硬件')
const r7 = await runWave(W7, 'W7-模型变体与Infra硬件')
log(`W7 完成: ${r7.filter(Boolean).length}/${W7.length}`)
phase('W8-评测数据应用优化')
const r8 = await runWave(W8, 'W8-评测数据应用优化')
log(`W8 完成: ${r8.filter(Boolean).length}/${W8.length}`)

const all = [...r7, ...r8].filter(Boolean)
return { total: W7.length + W8.length, written: all.length, files: all.map(x => x.file) }
