export const meta = {
  name: 'llmaction-补齐-run2',
  description: 'llm-action补齐run2: 28篇(模型架构家族/对齐/面试八股/评测/压缩)填到最大细节+双链',
  phases: [
    { title: 'W3-模型架构与对齐', detail: 'DeepSeek/LLaMA/GPT/Qwen/BERT/RLHF/DPO/PEFT 14篇' },
    { title: 'W4-面试八股与评测', detail: '面试base/算法/RLHF/微调/压缩+评测+量化 14篇' },
  ],
}

const DIR = 'C:/Users/jianm/Desktop/llm-action/'

const SCHEMA = {
  type: 'object',
  properties: {
    file: { type: 'string' }, lines: { type: 'number' },
    hasHandCalc: { type: 'boolean' }, hasAscii: { type: 'boolean' }, links: { type: 'number' },
  },
  required: ['file', 'lines', 'hasHandCalc', 'hasAscii', 'links'],
}

function buildPrompt(n) {
  const rel = n.related.map(r => `[[${r}]]`).join(' ')
  return `你是顶尖 AI-Infra 讲师。把 llm-action 仓库里一个【薄文件】补齐到【最大细节】，用【中文】，并用 Write 工具覆盖写回原路径。

【要补齐的文件(相对仓库根)】${n.file}
【主题】${n.title}
【必须覆盖】${n.cover}
【相关链接(底部"相关"用这些 Obsidian 双链)】${rel}

【第一步】先用 Read 读现有文件 ${DIR}${n.file}，尊重已有方向;近空则从零写。
【第二步】用 Write 把补齐后的完整内容写到绝对路径：${DIR}${n.file}

【风格——"从最底层讲清 + 逐数手算 + ASCII可视化"】
- 不假设读者记得公式;每个概念拆到最原子;每步说"为什么"。
- 【必含 ASCII 可视化图】(结构/数据流/流程/对比)。
- 【适用就必含逐数手算/数值例子】(参数量/显存/FLOPs/量化scale/损失/注意力等)。
- 数学公式行内 $...$、独立 $$...$$;该推导就推导。
- 模型架构类: 讲清【设计动机、关键创新、与前代/同类差异、规模配置、训练要点】;数字以公开技术报告为准, 不确定的具体超参标注"约/以官方为准", 不编造。
- 面试类: 组织成【高频问题→踩点答案→追问】, 覆盖该主题核心考点, 给出可背诵的要点与陷阱。

【固定结构】
# ${n.title}
> 一句话定位。📍 导航：[[00-知识地图]]
> 🔗 相关：${rel}

## 阅读地图(表格)
## 0. 一句话锚点
## 1. 地基/前置
## 2.~N. 逐步拆解(原理+为什么+ASCII图; 适用处给公式/手算)
## (单独一节)数值示例/手算 或 面试问答清单
## 对照/复杂度表
## 常见问题/高频追问(表格)
## 🔗 跳转链接

【硬要求】长度 250~450 行, 高密度不灌水; 顶部导航+底部双链必须有。写完返回 JSON {file,lines,hasHandCalc,hasAscii,links}。直接开始。`
}

const W3 = [
  { file: 'llm-algo/deepseek/DeepSeek-V2.md', title: 'DeepSeek-V2 架构', cover: 'MLA低秩KV压缩+解耦RoPE(核心);DeepSeekMoE(细粒度+共享专家);辅助损失负载均衡;规模配置;经济高效原因', related: ['00-知识地图','llm-algo/moe/README','llm-algo/旋转编码RoPE','llm-algo/deepseek/DeepSeek-V3','llm-optimizer/kv-cache'] },
  { file: 'llm-algo/deepseek/DeepSeek-V3.md', title: 'DeepSeek-V3 架构', cover: 'MLA+DeepSeekMoE延续;无辅助损失负载均衡(bias);MTP多token预测;FP8训练;671B/37B激活;工程亮点', related: ['00-知识地图','llm-algo/deepseek/DeepSeek-V2','llm-algo/deepseek/DeepSeek-R1','llm-compression/quantization/fp8','llm-train/fp8'] },
  { file: 'llm-algo/deepseek/DeepSeek-R1.md', title: 'DeepSeek-R1 推理模型', cover: 'R1-Zero纯RL(GRPO)涌现推理;R1冷启动+多阶段;推理链CoT;蒸馏到小模型;与o1对标;RL奖励设计', related: ['00-知识地图','llm-algo/deepseek/DeepSeek-V3','llm-alignment/RLHF','llm-interview/llm-rlhf'] },
  { file: 'llm-algo/llama/模型架构.md', title: 'LLaMA 模型架构', cover: 'RMSNorm+RoPE+SwiGLU+GQA;Pre-norm;与原始Transformer差异;LLaMA1/2/3演进;参数配置;为什么这些选择', related: ['00-知识地图','llm-algo/transformer/模型架构','llm-algo/旋转编码RoPE','llm-algo/mlp'] },
  { file: 'llm-algo/gpt2/模型架构.md', title: 'GPT-2 模型架构', cover: 'Decoder-only;因果注意力;可学习位置编码;Pre-LN;权重共享;与GPT/GPT-3规模;自回归语言建模', related: ['00-知识地图','llm-algo/transformer/模型架构','llm-algo/gpt3/README'] },
  { file: 'llm-algo/qwen/README.md', title: 'Qwen 系列架构', cover: 'Qwen架构要点(RoPE/RMSNorm/SwiGLU/GQA);Qwen2/2.5演进;长上下文;MoE版本;多语言;与LLaMA差异', related: ['00-知识地图','llm-algo/llama/模型架构','llm-algo/qwen2'] },
  { file: 'llm-algo/mixtral/README.md', title: 'Mixtral (稀疏MoE)', cover: 'Mixtral 8x7B;每层8专家Top-2路由;稀疏激活;专家并行;与稠密对比;路由分析', related: ['00-知识地图','llm-algo/moe/README','llm-algo/llama/模型架构','llm-compression/quantization/moe模型量化'] },
  { file: 'llm-algo/bert.md', title: 'BERT (双向编码器)', cover: 'Encoder-only;MLM+NSP预训练;双向注意力;与GPT(单向)对比;[CLS]/[SEP];微调范式;为什么不能生成', related: ['00-知识地图','llm-algo/transformer/模型架构','llm-algo/gpt2/模型架构'] },
  { file: 'llm-algo/t5/README.md', title: 'T5 (Encoder-Decoder)', cover: 'text-to-text统一范式;Encoder-Decoder;相对位置偏置;span corruption预训练;与Decoder-only对比', related: ['00-知识地图','llm-algo/transformer/模型架构','llm-algo/bert'] },
  { file: 'llm-algo/chatglm/README.md', title: 'ChatGLM / GLM 架构', cover: 'GLM自回归填空预训练;2D位置编码;ChatGLM演进;Prefix-LM;与纯Decoder差异;中文优化', related: ['00-知识地图','llm-algo/transformer/模型架构','llm-algo/旋转编码RoPE'] },
  { file: 'llm-alignment/RLHF.md', title: 'RLHF 人类反馈强化学习', cover: '三阶段SFT→RM→RL;奖励模型pairwise损失;PPO目标(裁剪+KL惩罚+优势)推导;actor-critic;为什么KL锚;手算PPO裁剪;不稳定性', related: ['00-知识地图','llm-algo/训练范式','llm-alignment/DPO','llm-interview/llm-rlhf'] },
  { file: 'llm-alignment/DPO.md', title: 'DPO 与 GRPO', cover: 'DPO把RLHF变直接偏好分类损失(推导为什么等价);免奖励模型;β温度;GRPO组内采样归一优势免critic;与PPO对比;手算损失', related: ['00-知识地图','llm-alignment/RLHF','llm-algo/deepseek/DeepSeek-R1','llm-interview/llm-rlhf'] },
  { file: 'llm-train/peft/Prefix-Tuning.md', title: 'Prefix-Tuning / P-Tuning', cover: '在每层加可训练前缀向量(冻结主干);与全量微调对比;参数量;reparameterization;P-Tuning v1/v2;手算前缀如何进注意力', related: ['00-知识地图','llm-train/peft/Prompt-Tuning','llm-train/peft/PEFT-API'] },
  { file: 'llm-train/peft/Prompt-Tuning.md', title: 'Prompt-Tuning / LoRA 对比', cover: 'soft prompt软提示;与Prefix/LoRA对比;LoRA低秩更新原理+手算ΔW=BA;为什么参数高效;何时用哪个', related: ['00-知识地图','llm-train/peft/Prefix-Tuning','llm-train/peft/PEFT-API'] },
]

const W4 = [
  { file: 'llm-interview/base.md', title: '面试·基础八股', cover: 'Transformer/注意力/位置编码/归一化/激活/为什么这样设计 等高频基础题;每题踩点答案+追问', related: ['00-知识地图','llm-algo/transformer/模型架构','llm-interview/comprehensive','llm-interview/llm-algo'] },
  { file: 'llm-interview/comprehensive.md', title: '面试·综合大题', cover: '系统设计/训练推理全链路/scaling/长上下文/显存优化 等综合题;按情境组织答题框架', related: ['00-知识地图','llm-interview/base','llm-interview/llm-algo','llm-interview/llm-eval'] },
  { file: 'llm-interview/llm-algo.md', title: '面试·算法与模型', cover: 'MHA/MQA/GQA/MLA/MoE/RoPE/各家模型架构 高频题;数学推导点;陷阱', related: ['00-知识地图','llm-algo/moe/README','llm-algo/旋转编码RoPE','llm-algo/deepseek/DeepSeek-V2'] },
  { file: 'llm-interview/llm-rlhf.md', title: '面试·对齐与RLHF', cover: 'RLHF三阶段/PPO/DPO/GRPO/奖励模型/KL/R1 高频题;推导与取舍;陷阱', related: ['00-知识地图','llm-alignment/RLHF','llm-alignment/DPO','llm-algo/deepseek/DeepSeek-R1'] },
  { file: 'llm-interview/llm-ft.md', title: '面试·微调与训练', cover: 'SFT/LoRA/QLoRA/PEFT/分布式并行/混合精度 高频题;参数高效原理;显存账', related: ['00-知识地图','llm-train/peft/Prompt-Tuning','llm-train/peft/Prefix-Tuning'] },
  { file: 'llm-interview/llm-compress.md', title: '面试·压缩与量化', cover: '量化(GPTQ/AWQ/SmoothQuant/INT8/FP8)/蒸馏/剪枝 高频题;量化数学;掉点原因', related: ['00-知识地图','llm-compression/quantization/量化基础','llm-compression/quantization/GPTQ','llm-compression/quantization/SmoothQuant'] },
  { file: 'llm-interview/llm-eval.md', title: '面试·评测与性能', cover: '评测集/指标(困惑度/准确率)/推理性能(TTFT/TPOT/吞吐/MFU) 高频题;压测方法', related: ['00-知识地图','llm-eval/大模型测评集','llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释'] },
  { file: 'llm-interview/llm-app.md', title: '面试·应用与RAG/Agent', cover: 'RAG/向量库/Agent/Function calling/Prompt工程 高频题;RAG召回-重排-生成链路;陷阱', related: ['00-知识地图','llm-application/rag/README','llm-application/应用场景'] },
  { file: 'llm-eval/大模型测评集.md', title: '大模型测评集', cover: 'MMLU/C-Eval/GSM8K/HumanEval/MT-Bench等;考什么;few-shot;污染;裁判模型;局限', related: ['00-知识地图','llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释','llm-eval/README'] },
  { file: 'llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释.md', title: '训练/推理性能指标名词', cover: 'TTFT/TPOT/吞吐tokens/s/延迟/QPS/MFU/HFU/利用率/带宽;每个定义+公式+怎么测+优化方向', related: ['00-知识地图','ai-infra/算力/GPU工作原理','llm-algo/FLOPs','llm-eval/大模型测评集'] },
  { file: 'llm-compression/quantization/fp4.md', title: 'FP4 / NVFP4 / MXFP4', cover: 'FP4格式(E2M1)位分配;block scaling(MXFP4/NVFP4微缩放);为什么需要;Blackwell支持;与INT4/FP8对比;手算一个FP4表示', related: ['00-知识地图','llm-compression/quantization/fp8','llm-compression/quantization/量化基础'] },
  { file: 'llm-compression/quantization/moe模型量化.md', title: 'MoE 模型量化', cover: 'MoE专家多导致权重大;量化专家权重的挑战(专家激活稀疏/校准数据);per-expert量化;路由保精度;实践要点', related: ['00-知识地图','llm-algo/moe/README','llm-compression/quantization/量化基础','llm-algo/mixtral/README'] },
  { file: 'llm-compression/经验.md', title: '模型压缩实战经验', cover: '量化/蒸馏/剪枝选型;校准数据;精度回退排查;KV量化注意;不同硬件支持;组合策略', related: ['00-知识地图','llm-compression/quantization/量化基础','llm-compression/README'] },
  { file: 'llm-inference/offload.md', title: '推理 Offload(显存卸载)', cover: '显存放不下时把权重/KV卸载到CPU/NVMe;ZeRO-Inference/FlexGen思路;带宽瓶颈与调度;何时用;吞吐取舍', related: ['00-知识地图','llm-optimizer/kv-cache','ai-framework/deepspeed/README'] },
]

const runWave = (notes, phaseTitle) =>
  parallel(notes.map(n => () => agent(buildPrompt(n), { label: `fill:${n.file}`, phase: phaseTitle, schema: SCHEMA })))

phase('W3-模型架构与对齐')
const r3 = await runWave(W3, 'W3-模型架构与对齐')
log(`W3 完成: ${r3.filter(Boolean).length}/${W3.length}`)

phase('W4-面试八股与评测')
const r4 = await runWave(W4, 'W4-面试八股与评测')
log(`W4 完成: ${r4.filter(Boolean).length}/${W4.length}`)

const all = [...r3, ...r4].filter(Boolean)
return { total: W3.length + W4.length, written: all.length, files: all.map(x => x.file) }
