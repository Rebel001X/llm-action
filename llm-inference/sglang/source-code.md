# SGLang 源码阅读工程指南（Fork / 切 tag / 断点 / 读码路线）

> 一句话定位：这篇不讲「SGLang 的架构长什么样」（那是 [[llm-inference/sglang/项目代码结构]] 的活），而讲「**你怎么把这套迭代极快的源码搭起来、钉死在一个版本上、打上断点、一步步跟着一条请求走进 GPU**」——即**读源码的工程方法学**。📍 导航：[[00-知识地图]]
> 🔗 相关：[[llm-inference/sglang/项目代码结构]] [[llm-inference/sglang/README]] [[llm-inference/vllm/源码]] [[llm-inference/sglang/服务器启动参数]]

---

## 阅读地图

| 你想知道 | 看哪一节 |
| --- | --- |
| 为什么读 SGLang 源码必须先「钉版本」 | §0 锚点、§1 地基 |
| fork 怎么连上 upstream、怎么同步官方更新（保留原始 git 流程） | §2 三条远程关系 + 命令逐行解释 |
| 怎么 checkout 一个 tag 到可读可改的开发分支 | §3 切 tag 工作流 |
| 装成「可编辑安装」让改源码立刻生效 | §4 editable install |
| 一条请求怎么跨 3 个进程、怎么打断点跟进去 | §5 断点策略 + 跨进程调试 |
| 应该按什么顺序读、每个文件先看哪个函数 | §6 读码路线图 |
| 改了源码怎么用最小压测脚本验证 | §7 验证回路 |
| 迭代太快、类名/签名对不上怎么办 | §8 抗漂移策略 |
| 常见踩坑（编译、ZMQ、CUDA Graph 挡断点） | §9 常见问题 |

> 说明：SGLang 迭代极快。本文的**目录骨架、读码顺序、调试手法是稳定的方法**；凡涉及精确函数名、行号、命令默认值，一律以你 checkout 的那个 tag 源码与官方 `--help` 为准，本文不编造。

---

## 0. 一句话锚点

读 SGLang 源码的第一性原理只有一句：

> **先把版本钉死，再读。**

SGLang 主分支几乎每天都在动，类名、文件位置、调度细节随版本漂移。你今天对着博客学的 `get_next_batch_to_run`，下个月可能改名、拆函数、挪文件。所以「读源码」这件事的工程起点不是 `git clone` 完就读 `main`，而是：

```
fork 一份(你能改、能 push)
   → 接上 upstream(官方,只读同步源)
   → checkout 一个 tag(如 0.4.x) 到开发分支(钉死,世界不再变)
   → editable 安装(改 .py 立刻生效,不用重装)
   → 打断点,跟一条请求走完三进程
```

本文就是把这条链每一步讲透。原始文件里那几条 `git` 命令（fork 同步 + 切 tag）正是这条链的第 2、3 步，下面逐行解释「**它在干嘛、为什么这么干**」。

---

## 1. 地基：为什么读 SGLang 必须先解决「版本」与「多进程」两件事

读一个普通 Python 库，`pip install` + 跳转定义就够了。读 SGLang 不行，有两个特殊性：

**特殊性一：迭代速度。** SGLang 是研究驱动项目，主分支变化剧烈。如果你对着 `main` 读，今天的笔记明天就对不上。**解法 = 切 tag 钉死**（§3）。

**特殊性二：多进程 + ZMQ。** SGLang 不是单进程。一条 `launch_server` 起的是 **HTTP / Scheduler(GPU) / Detokenizer** 多个进程，进程间用 **ZeroMQ** 传消息（架构细节见 [[llm-inference/sglang/项目代码结构]] §2）。这意味着：

```
普通库：一个 pdb 断点能跟到底
SGLang：一条请求会"消失"在进程边界——
        你在 TokenizerManager 打的断点,跟到 ZMQ.send() 就断了,
        因为后面的活在另一个进程(Scheduler)里,要在那边单独打断点。
```

所以读 SGLang 源码 = **版本管理学 + 跨进程调试学**。下面分别给工具链。

---

## 2. 三条远程关系：fork、upstream 与同步官方更新

### 2.1 为什么要 fork 而不是直接 clone 官方

你要**改源码做实验**（加日志、打点、改调度），就得有个能 `push` 的仓库——官方仓库你没写权限。所以标准姿势是：在 GitHub 上 **Fork** 官方 `sgl-project/sglang` 到你自己名下，clone 你的 fork，再把官方仓库挂成一个**只读的上游远程（upstream）**用于同步。

```
        GitHub: sgl-project/sglang  ← 官方(你只读)
                      ▲  fork
                      │
        GitHub: <你>/sglang         ← 你的 fork(origin, 可push)
                      │ clone
                      ▼
        本地工作副本  ── remote: origin   → 你的 fork
                     └─ remote: upstream → 官方(同步用)
```

### 2.2 原始命令逐行解释（fork 同步流程）

原始文件给的这段，就是把本地副本「追上官方最新」再推回你 fork 的标准动作：

```bash
git clone git@github.com:liguodongiot/sglang.git        # ① clone 你自己的 fork(origin)
git remote add upstream https://github.com/sgl-project/sglang.git   # ② 把官方挂成 upstream(只读源)

git fetch upstream --tags          # ③ 从官方拉取所有提交"和所有 tag"(--tags 关键!)
git rebase upstream/main           # ④ 把你的分支"垫"到官方 main 之上(线性历史)
git push                           # ⑤ 推回你的 fork
git push --tags                    # ⑥ 把刚同步下来的 tag 也推到你的 fork
```

逐行的「**为什么**」：

| 行 | 在干嘛 | 为什么这么干 |
| --- | --- | --- |
| ① | clone 自己的 fork | 默认远程名 `origin` 指向你 fork，你能 push |
| ② | `remote add upstream 官方` | 给官方一个名字，之后才能 `fetch upstream` |
| ③ | `fetch upstream --tags` | **拉提交也拉 tag**——读源码要切 tag（§3），没 `--tags` 就没 tag 可切 |
| ④ | `rebase upstream/main` | 把你 fork 的改动「重放」到官方最新之上，得到干净线性历史；比 `merge` 少一堆合并提交 |
| ⑤⑥ | `push` / `push --tags` | 把同步结果（含 tag）落到你 fork，别的机器也能拉到 |

> 注释里那句「如果你的主分支不是叫 master 就换成你的名字（main 之类）」很重要：SGLang 主分支叫 `main`，所以是 `git rebase upstream/main`。`--tags` 这步**绝不能省**，因为读源码的下一步全靠 tag。
>
> ⚠️ `rebase` 会改写历史。只在**你私有的、没人协作的 fork** 上这么干；若你 fork 已被别人引用，改用 `git merge upstream/main` 更安全。

```
fetch upstream --tags 之后，本地多了什么:
   原来:  o─o─o (你的 main, origin)
   现在:  o─o─o─o─o─o (upstream/main, 更新)
          并且本地多了一堆 tag: 0.4.0  0.4.1  0.4.2 ...  ← 关键资产!
```

---

## 3. 切 tag 工作流：把世界钉死在一个版本

### 3.1 为什么是 tag 而不是分支或某个 commit

- **tag = 官方盖章的发布点**：`0.4.1` 这种 tag 对应一次正式 release，代码自洽、能跑通、有对应文档/博客。对着它读，笔记不会随主分支漂移。
- 直接 `git checkout 0.4.1` 会进入「**detached HEAD（游离头指针）**」——你能看不能稳妥地改和 push。所以要**从 tag 拉一个分支**出来再工作。

### 3.2 原始命令逐行解释（切 tag 到开发分支）

```bash
git tag                                   # ① 列出所有 tag(确认 0.4.1 在不在)
git checkout -b dev-code-0.4.1 0.4.1      # ② 从 tag 0.4.1 新建分支 dev-code-0.4.1 并切过去
git push -u origin dev-code-0.4.1         # ③ 把这个开发分支推到你 fork,并建立追踪
```

| 行 | 在干嘛 | 为什么 |
| --- | --- | --- |
| ① | `git tag` | 看有哪些版本可钉；没看到想要的 tag → 回 §2 补 `fetch upstream --tags` |
| ② | `checkout -b 新分支 <tag>` | **从 tag 长出一个可写分支**，避开 detached HEAD；以后你的所有打点/笔记改动都提交在这个分支上，与官方版本一一对应 |
| ③ | `push -u origin <分支>` | `-u` 建立 upstream 追踪，以后 `git push` 直接生效；换机器能拉到你这份「带注解的 0.4.1」 |

```
切 tag 之后的心智模型:
   官方 0.4.1 (不可变 tag)
        │ checkout -b
        ▼
   dev-code-0.4.1  ← 你的"实验室"分支
        ├─ 加 print/logger
        ├─ 写代码注释
        └─ 小改调度逻辑做对照实验
   想换版本读? 再 checkout -b dev-code-0.4.5 0.4.5,互不干扰。
```

> 命名约定 `dev-code-<版本>` 是个好习惯：一眼知道这是「为读 `<版本>` 源码而开的实验分支」。

---

## 4. Editable 安装：让「改源码」立刻生效

钉好版本只是能**读**。要**改了立刻跑**，必须用「可编辑安装（editable / develop install）」，否则你改的是源码树、跑的却是 `site-packages` 里那份旧拷贝。

```
普通安装(pip install .):
   源码树 ──复制──> site-packages/sglang/   ← 真正被 import 的是这份
   你改源码树 → 不生效(还得重装)

可编辑安装(pip install -e .):
   site-packages/sglang  ──软链/路径指针──> 你的源码树
   你改源码树 → import 立刻生效   ← 读源码做实验必须这样
```

要点（命令以仓库 `README/docs` 为准，本文讲原理）：

- SGLang 有 C++/CUDA 扩展（如 `sgl-kernel`），**纯 `pip install -e` 通常只让 Python 层可编辑**；改到 CUDA kernel 那层往往要重新编译。所以：**先改 Python 层（managers/mem_cache/model_executor）做实验最省事**，那是读码主战场，纯 Python，editable 即时生效。
- 建议在**独立虚拟环境/conda 环境**里装，避免污染系统；CUDA / torch 版本要和该 tag 的 `requirements` 对齐（版本不匹配是头号踩坑，见 §9）。

> 一句话：**editable + 只改 Python 层 = 读码实验的甜区**。要碰 kernel 再谈编译。

---

## 5. 跨进程断点：一条请求怎么跟进去

这是读 SGLang **最难也最关键**的一关——因为请求会跨进程「消失」。

### 5.1 进程边界在哪断你的断点

```
HTTP 进程                      Scheduler 进程(GPU)            Detokenizer 进程
─────────────                  ───────────────────            ─────────────
TokenizerManager
  tokenize
  ZMQ.send() ──────断点跟到这就"断了"────▶ recv_requests()
  await future                            event_loop
                                          get_next_batch_to_run()  ← 调度核心,要在这"另起炉灶"打断点
                                          run_batch()/forward
                                          process_batch_result()
                                          ZMQ.send() ───────────────────────▶ detokenize
                          回填 future ◀──────────────────────────────── ZMQ.send()
```

**核心认知**：一个 `pdb`/IDE 断点只活在一个进程里。你不能从 `TokenizerManager` 一路 step 进 `Scheduler`——中间是 ZMQ + 进程边界。要**在每个进程各打各的断点**，靠「同一个 `rid`（请求 id）」把三段拼起来。

### 5.2 三种实战调试手法（从轻到重）

**手法 A：日志打点（最稳，跨进程首选）。**
在三个关键函数各加一行带 `rid` 的日志，靠日志时间线把跨进程的「一条请求」串起来：

```python
# TokenizerManager.generate_request 里
logger.info(f"[TRACE] rid={rid} tokenized len={len(input_ids)}")
# Scheduler.get_next_batch_to_run 里
logger.info(f"[TRACE] rid in batch={[r.rid for r in batch.reqs]} mode={forward_mode}")
# process_batch_result 里
logger.info(f"[TRACE] rid={req.rid} new_token={tok} finished={req.finished()}")
```

为什么首选日志：**断点会卡住事件循环**，而 Scheduler 是个高频 `while True` 循环 + 还要喂 GPU，断在那儿容易把整套机器卡死或触发超时/心跳问题。日志无侵入、可在生产级负载下跑。

**手法 B：单进程化 + pdb（适合精读调度逻辑）。**
想真正 step 进 `get_next_batch_to_run`，把干扰降到最低：
- 用**单请求、单并发**复现（一条 prompt），让批里只有它，逻辑最干净。
- 尽量**关掉会绕过 Python 的加速路**：CUDA Graph、`torch.compile`、overlap 调度——它们会让 decode 路径「跳过」你想看的 Python 代码或打乱时序（见 §9）。
- 在 `scheduler.py` 的目标函数下 `breakpoint()`，单步看 `RadixCache.match_prefix` 返回了什么、批怎么组的。

**手法 C：attach 到已起进程（Scheduler 是子进程时）。**
Scheduler 常被 `launch_server` 以子进程方式拉起，直接在启动命令里下断点未必断在子进程。可用「按进程 attach」的远程调试（如 `debugpy` 监听端口、IDE attach 到对应 PID）。原理就一句：**找到持有 GPU 的那个 PID，把调试器接上去**。

### 5.3 用 `rid` 把三段缝起来（端到端跟踪心法）

```
你在日志里 grep 同一个 rid=abc123:
  [t0] HTTP        rid=abc123 tokenized len=812
  [t1] Scheduler   rid in batch=[abc123] mode=PREFILL   ← 命中前缀后只 prefill 失配段
  [t2] Scheduler   rid=abc123 new_token=15043 finished=False  (decode 第1步)
  [t3] Scheduler   rid=abc123 new_token=29871 finished=False
  ...
  [tk] Scheduler   rid=abc123 finished=True
  [tk] Detok       rid=abc123 text="..."
一条时间线 = 一条请求的完整一生。这是跨进程"看清调用链"的唯一可靠办法。
```

---

## 6. 读码路线图：先读哪个文件的哪个函数

别从头到尾平铺读。按「**一条请求的生命周期**」逆向钻，命中率最高。下面给「文件 → 先看的函数 → 看完该回答的问题」三元组（精确名以你 tag 源码为准）：

```
读码 8 步(对照 [[llm-inference/sglang/项目代码结构]] 的模块图):

① launch_server.py            → 入口:起了哪几个进程? 怎么 spawn?
② srt/server.py / entrypoints → HTTP 路由:/generate /v1/chat 落到哪个 handler?
③ managers/tokenizer_manager  → generate_request:text→Req,怎么 ZMQ.send,怎么 await rid?
④ managers/scheduler.py       → event_loop + get_next_batch_to_run  ★最该精读★
⑤ mem_cache/radix_cache.py    → match_prefix / insert / evict:前缀树四方法怎么动?
⑥ managers/schedule_batch.py  → Req / ScheduleBatch:一个"批"的状态字段有哪些?
⑦ model_executor/model_runner → forward:ScheduleBatch→ForwardBatch→logits→采样?
⑧ managers/detokenizer_manager→ token_id→文本增量,怎么回流到 HTTP 流?
```

**重点提示**：整台机器 80% 的逻辑在 **④ `scheduler.py` 的 `get_next_batch_to_run` 和 `process_batch_result`** 两个函数里。**先把这两个读穿**，再往两边（前面 ③ tokenizer、后面 ⑦ model_runner）扩，是收敛最快的路径。RadixCache（⑤）是 ④ 的「记忆」，读完 ④ 自然要钻 ⑤。

> 读码搭档：把这份路线图和 [[llm-inference/sglang/项目代码结构]] 的「§8 端到端时序图」并排看——那张图是**地图**，本节是**进山的脚印顺序**。

---

## 7. 验证回路：改了源码怎么确认有效

读源码常伴随「我猜这里改了会怎样」的对照实验。要有**最小、可复现、能量化**的验证回路，否则改了等于白改。

```
改源码(dev-code-0.4.1 分支)
   │
   ▼
editable 已生效 → 直接重启 launch_server
   │
   ▼
跑一个"最小压测"基线:
   · 单请求正确性: curl 一条 /generate 看输出对不对
   · 吞吐/延迟: 用仓库自带 bench_* 脚本(bench_offline_throughput 等)
   │
   ▼
对照: 改之前 vs 改之后的 吞吐(tok/s) / 命中率 / 显存
   │
   ▼
git commit 到 dev 分支(把"这个改动→这个效果"记下来)
```

要点：
- **先验「正确」再验「快」**：改调度/缓存最容易悄悄改坏正确性（比如错误复用前缀导致串扰），先 curl 一条确认输出没乱。
- **控制变量**：每次只改一处，跑同一个 prompt 集、同一并发，吞吐数才有可比性。
- **基线先存**：动手前先把「原版」的 bench 数跑一遍存下来，否则没有对照系。

> 性能指标怎么读（TTFT / TPOT / 吞吐 / goodput）见 [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]]。

---

## 8. 抗漂移：迭代太快、对不上号怎么办

SGLang 三个月就能让一篇博客「过时」。建立几条抗漂移习惯：

| 症状 | 抗漂移做法 |
| --- | --- |
| 博客提到的类/函数找不到 | 用 `git log -p -- <file>` 看该文件历史，或在仓库**全局搜函数名**（被改名/挪文件了） |
| 不确定某行为属于哪个版本 | 永远以「**我 checkout 的那个 tag**」为准，笔记里写清版本号 |
| 想知道两个版本差异 | `git diff 0.4.1 0.4.5 -- srt/managers/scheduler.py` 直接看调度器演进 |
| 官方又发新版想跟 | 回 §2 `fetch upstream --tags` → §3 `checkout -b dev-code-<新版> <新 tag>`，老分支留着对照 |
| 默认值/参数变了 | 以 `python -m sglang.launch_server --help` 实测为准，别信旧文档（含本仓库笔记） |

```
版本对照阅读法(看懂"为什么这么演进"):
   git diff 0.4.0 0.4.4 -- python/sglang/srt/managers/scheduler.py
        └─ 读 diff 比读最终态更长见识:
           能看到"零开销重叠调度"这类优化是怎么一步步加进调度循环的。
```

> 核心心法：**版本号是源码笔记的一等公民**。任何「SGLang 怎么做 X」的结论，都默认隐含「在版本 V 下」。脱离版本谈源码细节，就是给自己挖坑。

---

## 9. 常见问题（读码踩坑）

| 问题 | 答 |
| --- | --- |
| `pip install -e .` 报编译错误 | 多半是 **CUDA / torch / 编译器版本和该 tag 不匹配**。先对齐该 tag `requirements`，纯读 Python 层可考虑跳过 kernel 编译路径。 |
| 断点打了但 decode 阶段不进 | **CUDA Graph / torch.compile** 把 decode 路径捕获/编译走了，绕过 Python。读码实验时**关掉**这些加速开关（相关参数见 [[llm-inference/sglang/服务器启动参数]]），让代码走纯 Python 路径。 |
| 在 Scheduler 打断点整个服务卡死/超时 | Scheduler 是高频循环且喂着 GPU，断在那容易触发心跳/超时。优先用**日志打点**（§5.2 手法 A），或单请求 + 关 overlap 再 pdb。 |
| 跟到 `ZMQ.send` 就跟不下去了 | 正常——后面在**另一个进程**。去那个进程的入口函数（`recv_requests`/`detokenize`）另打断点，用 `rid` 缝起来（§5.3）。 |
| 改了 `.py` 没生效 | 没装成 editable（`-e`），import 的还是 `site-packages` 旧拷贝。重新 `pip install -e .`。 |
| `git checkout 0.4.1` 后提示 detached HEAD | 正常但别在上面长期改。用 `git checkout -b dev-code-0.4.1 0.4.1` 拉分支再工作（§3）。 |
| `git tag` 看不到想要的版本 | 没拉 tag。回 §2 跑 `git fetch upstream --tags`。 |
| 类名和这篇/那篇博客对不上 | 版本漂移。以你 tag 源码为准，必要时全局搜函数名或看 `git log`（§8）。 |
| 想读源码从哪进 | `launch_server.py` → `server.py` → `tokenizer_manager.py` → **`scheduler.py` 的 `get_next_batch_to_run`** → `radix_cache.py` → `model_runner.py`（§6）。 |

---

## 10. 一页速查（把这篇压成一张卡）

```
读 SGLang 源码 = 5 步:
  1. fork 官方 → clone → git remote add upstream 官方
  2. git fetch upstream --tags   (拉提交+tag,缺一不可)
  3. git checkout -b dev-code-0.4.1 0.4.1   (从 tag 拉可写分支,钉死版本)
  4. pip install -e .  (editable,改 .py 立刻生效;先攻 Python 层)
  5. 打断点/日志,用 rid 缝起跨进程调用链,精读 scheduler.py 两函数

抗漂移: 版本号是一等公民; 对不上就 git diff <tag1> <tag2> / 全局搜函数名 / --help 实测
跨进程: 一个断点只活一个进程; 日志打点 > pdb; 关 CUDA Graph/compile/overlap 才能走纯 Python
读码序: ④ scheduler 先读穿 → 往两边(③tokenizer / ⑦model_runner)扩,⑤radix_cache 是④的记忆
```

---

## 🔗 跳转链接

- [[00-知识地图]] — 全局导航
- [[llm-inference/sglang/项目代码结构]] — **本文搭档**：架构/模块/端到端时序图（地图）；本文是「怎么把这张地图走一遍」（脚印）
- [[llm-inference/sglang/README]] — RadixAttention 的「为什么」（概念层）
- [[llm-inference/sglang/服务器启动参数]] — 读码时要关的加速开关、要看的调度参数都在这
- [[llm-inference/vllm/源码]] — 对照阅读：vLLM 源码用同样的「钉版本 + 跟一条请求」方法读
- [[llm-eval/llm-performance/大模型场景下训练和推理性能指标名词解释]] — §7 验证回路里 TTFT/TPOT/吞吐怎么算
- 参考：https://blog.csdn.net/sdujava2011/article/details/138312278
