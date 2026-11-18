# CyberBrain 架构与实现

[English](ARCHITECTURE.en.md) | **中文**

本文描述 CyberBrain 的核心流程和实现细节，内容以当前代码为准。设计思路见 [README.md](../README.md)，依赖的数据库见 [QUICKSTART.md](../QUICKSTART.md)。

## 目录

1. [总体结构](#总体结构)
2. [存储布局](#存储布局)
3. [对话主流程 react](#对话主流程-react)
4. [System1：本能反应与认知门控](#system1本能反应与认知门控)
5. [认知路径选择](#认知路径选择)
6. [System2：社交决策](#system2社交决策)
7. [System3：语言表达](#system3语言表达)
8. [说话人识别](#说话人识别)
9. [谎言与事实来源](#谎言与事实来源)
10. [记忆系统](#记忆系统)
11. [Memory Gate：记忆门控](#memory-gate记忆门控)
12. [主体性：3F 模型](#主体性3f-模型)
13. [待办与自主行为](#待办与自主行为)
14. [主动触达](#主动触达)
15. [后台 Worker](#后台-worker)
16. [生命事件](#生命事件)
17. [管理接口与运行时配置](#管理接口与运行时配置)
18. [模型调用一览](#模型调用一览)
19. [已知问题与限制](#已知问题与限制)
20. [理论参考](#理论参考)

---

## 总体结构

```
                         ┌────────────────────────────┐
  POST /brain/react ───▶ │   cyber_brain.react()      │
                         │  System1 → 路由 → System2   │
                         │          → System3          │
                         └──────┬───────────┬─────────┘
                                │           │
          ┌─────────────────────┼───────────┼──────────────────────┐
          ▼                     ▼           ▼                      ▼
  记忆检索（Qdrant）      工作记忆 / 人格 / 主体性 / 关系      生命事件（MongoDB）
  Brief / Episodic /           （Redis）
  Knowledge
          ▲                     ▲
          │                     │
  ┌───────┴─────────────────────┴───────────────────────────────────┐
  │ 后台 Worker：记忆衰减 · 工作记忆每日整理 · 念头生成 · 待办提醒     │
  └─────────────────────────────────────────────────────────────────┘
```

代码分层：

| 层 | 目录 | 说明 |
|---|---|---|
| 接口 | `app/controller/` | `cyberbrain_controller`（`/brain`）、`admin_controller`（`/admin`）、`health_controller`（`/health`） |
| 业务 | `app/service/` | 主流程、System1、Memory Gate、主体性、工作记忆、知识、人格、任务执行、生命事件 |
| 数据 | `app/repository/` | Brief/Episodic 记忆（封装 mem0 与 Qdrant 原生操作）、人际关系、生命事件 |
| 客户端 | `app/client/` | 豆包（Ark）、Embedding、Rerank、Qdrant、Redis、MongoDB |
| 后台 | `app/worker/` | 四个随服务启动的后台任务 |
| 配置 | `app/config/` | 配置项、运行时可调参数、提示词 |

启动顺序（`app/main.py` 的 lifespan）：初始化 mem0 双客户端（Brief / Episodic）→ 初始化 Knowledge 集合 → 连接 MongoDB（失败只告警）→ 启动四个后台 Worker。

网络方面，`app/common/httpx_patch.py` 在导入时把 httpx 客户端的 `trust_env` 默认改为 `False`，避免 macOS 系统代理拦截到 localhost 的连接。

---

## 存储布局

### Redis

| Key | 类型 | 内容 | 写入方 |
|---|---|---|---|
| `persona:{agent_id}:system1` | String | System1 人格片段：情绪基线、触发点、价值观 | 初始化 / 更新接口 |
| `persona:{agent_id}:system2` | String | System2 人格片段：身份、能力边界、元认知、诚信倾向 | 同上 |
| `persona:{agent_id}:speaking_manner` | String | System3 的说话习惯 | 同上 |
| `subject_context:{agent_id}` | String（JSON） | 主体性上下文：`situation`、`overall_state`、`thoughts`、`pending_tasks`（只含待执行的任务）、`recent_done`（近期完成/取消任务，用于去重）、`last_update`。**每次写入都会把过期时间重置为 30 天** | 主体性服务 |
| `agent:{agent_id}:relationships` | Hash | field 为人名，value 为 JSON `{kind, agent_id, relationship, description}`；旧格式 `"关系类型，描述"` 按 kind=human 读取 | 初始化 / 更新接口、每日整理 |
| `agent:{agent_id}:capabilities` | Set | agent 的能力，目前只有 `web_search`；不存在即没有任何能力 | 初始化 / 更新接口 |
| `outbox:{agent_id}:{人名}` | List | agent 主动想对这个人（kind=human）说的话，最多保留 50 条，`GET /brain/fetch_talk` 取走即删除 | 主动触达 |
| `wm:{agent_id}:stm` | List | 工作记忆原文，每条一个 JSON：`{type: "raw", role, speaker, content, lies?}` | react |
| `wm:{agent_id}:stm:compress` | List | 压缩块或当日摘要：`{type: "compress" \| "daily", summary, speakers, count?}` | 滚动压缩、每日整理 |
| `subject_context:{agent_id}:tasks_lock` | String | 待办读-改-写的锁（10 秒过期） | 主体性服务 |
| `wm:{agent_id}:lock` | String | 压缩与每日整理共用的锁 | 同上 |
| `worker:decay:last_run` | String | Decay Worker 上次执行的 unix 时间戳 | Decay Worker |

工作记忆 key 中 `{agent_id}` 两侧的花括号是字面量，作为 Redis Cluster 的 hash tag，保证同一 agent 的 key 落在同一 slot、可以用事务。

persona 未配置时，`persona_manager.get()` 返回内置的缺省描述，保证流程不报错。

### Qdrant

| 集合 | 写入方式 | 向量 | 主要 payload |
|---|---|---|---|
| `brief_memories` | mem0 `add(infer=False)`，再补写 BM25 | 稠密 + `bm25` 稀疏（IDF） | `data`（原文）、`user_id`（即 agent_id）、`hash`、`created_at`、`updated_at`，以及写入时传入的 metadata（平铺） |
| `episodic_memories` | 同上 | 同上 | 同上，另有 `decay_weight`、`last_decay_update` |
| `knowledge_memories` | 直接写 Qdrant | 稠密 | `agent_id`、`title`、`content`、`knowledge_type`、`domain`、`sub_domain`、`proficiency`、`confidence`、`source`、`tags`、`related_knowledge`、`prerequisite_knowledge`、`used_count` 等 |

- 写 mem0 时固定 `infer=False`：**原文直接入库，不经过 LLM 抽取，也不做去重合并**。因此同样的内容写两次会得到两条记录。
- mem0 不会写 BM25 稀疏向量，`memory_repository.add()` 在 mem0 写入后，用 `qdrant/bm25` 模型（`multilingual` 分词）在服务端补写。启动时会检查 `bm25` 槽位，缺少 IDF 修饰时原地开启。
- Knowledge 写入时的 embedding 文本是 `"{title}\n{content}"`，每条生成新的 uuid。

### MongoDB

集合 `agent_life_event`（可配置），索引：`(agent_id, timestamp desc)`、`event_type`、`timestamp desc`。详见[生命事件](#生命事件)。

### SQLite

mem0 的记忆变更历史库：`./data/brief_history.db`、`./data/episodic_history.db`。

---

## 对话主流程 react

入口：`POST /brain/react`，参数 `agent_id`、`signal`（对方说的话）、`speaker`（说话人姓名，可选）。实现在 `cyber_brain.react()`。

```
记录 react_input 事件
  │
  ├─ 串行前置：
  │    1. 说话人识别（关系表 → 工作记忆 → 陌生人）
  │    2. 读 subject_context，格式化成两份文本：
  │         state_and_situation（给 System1/2）、state_only（给 System3）
  │    3. 把 signal 追加进工作记忆，再读出完整工作记忆文本
  │
  ├─ 预计算一次查询向量（写入缓存，三路召回共用）
  │
  ├─ 并行：
  │    ├─ Episodic 召回（单独的 task，与 System1 并行，路由前等它完成）
  │    ├─ Brief 召回
  │    └─ Knowledge 预取（余弦粗筛 + rerank）
  ├─ System1（看得到 Brief 和"我对这个话题掌握什么"）
  │
  ├─ 判断是否使用知识、是否被勾起了往事（Episodic 最高分）→ 选择认知路径
  │    ├─ fast：不带 Episodic，直觉计划
  │    ├─ memory：直觉计划 + 想起的往事 + 知识
  │    └─ deliberate：System2 → 决策计划（失败时退回直觉计划）
  │
  ├─ System3 生成说出口的话（System2 决定不说话时跳过）
  ├─ 解析 [谎言] 标记；把"我"的回应（或"（没有接话）"）写入工作记忆
  │
  ├─ 同步执行（返回前完成）：记录 react_output 事件（含完整 trace）
  │
  ├─ 异步执行（不阻塞返回）：Memory Gate 评估并持久化对方这句话；判断这轮对话是否产生待办；
  │    根据情绪调整整体状态、可能生成念头
  └─ 返回
```

返回字段：

| 字段 | 说明 |
|---|---|
| `response` | 说出口的话，已去掉 `[谎言]` 标记；不说话时为空 |
| `lies` | 被标记为谎言的句子 |
| `silent` | 是否选择不说话 |
| `route` | `fast` / `memory` / `deliberate` |
| `route_reason` | 选择该路径的原因，见[认知路径选择](#认知路径选择) |
| `emotion` | System1 全部输出 |
| `decision` | System2 决策（仅 deliberate 路径） |
| `brief` / `episodic` / `knowledge` | 本轮召回结果 |
| `requires_knowledge` | 本轮是否使用了知识 |
| `subjectivity` | 处境进了哪些系统、是否有整体状态 |
| `timings_ms` | 各阶段耗时：`subjectivity`、`pfc`、`embedding`、`brief`、`system1`、`knowledge_prefetch`、`knowledge`、`episodic`、`episodic_wait`、`system2`、`system3`、`total`、`react_total` |

查询向量缓存：`embedding_client.embed_query()` 带一个 512 条的 LRU 缓存（线程安全），Brief、Episodic（mem0 的检索 embedding 被替换为走这个缓存）和 Knowledge 三路召回共用同一次 embedding。

---

## System1：本能反应与认知门控

实现：`app/service/system1_tev1.py`。调用火山引擎方舟的 **System One 决策接口**（`doubao_client.decide()`，请求头带 `ark-beta-systemone-api: true`，默认模型 `DOUBAO_MODEL_LITE`）。它不生成文本，而是对一组问题做选择或打分，并返回概率。

**输入上下文**（按顺序拼接）：对方是谁 → 主体性上下文（整体状态 + 处境）→ System1 人格片段 → Brief 记忆 → 我对这个话题掌握的资料（rerank 后命中知识的标题，或"没有直接讲这个话题的资料"）→ 刚才的对话（工作记忆末尾 `BRAIN_SYSTEM1_RECENT_TURNS` 条原文，默认 4 条，不含本句和压缩摘要）→ 我此刻的状态（`describe_state()` 把整体状态转成一句话，如"心情不太好，压力很大，有点累"）→ 外部信号（信号前再标一次说话人，小模型对紧邻信号的内容权重更高）。

**整体状态要用文字、紧挨信号：** 以前状态只以三行数字出现在上下文开头，实测去掉它 System1 的判断几乎不变（林诗雨在"很差 / 基线 / 很好"三种状态下负面情绪 7/6/6 次，共 12 次）。改为文字放在信号前，并在情绪判断的说明里写明一般规律（心情差、压力大、疲惫时同一句话更容易让人烦躁、敏感或提不起劲；状态好时更宽容），三种状态下负面情绪变为 12/3/1，投入度 0.47/0.74/0.83。不会形成恶性循环：状态回归基线靠时间（见[整体状态](#feeling整体状态overall_state)），对话中的增量又有边际递减。

**关系里的牵挂是背景：** 情绪判断的说明里写明，情绪是对这句话的反应；关系描述里对对方的长期牵挂、担心只有在这句话触及那件事时才会被激起。以前关系描述写着"她快撑不住了"，对方说"晚安"也会判成焦虑（0.99）；实测与这件事无关的话判为焦虑从 4/15 降到 0/15，触及的话仍是 7/9。关系描述只写稳定的关系本身："最近状态越来越差"这类会变的情况写进处境（会随对话更新），"上个月借了十万"这类过去的事写进经历（Episodic）。

给 System1 最近对话，是因为"需要""好的""不用了"这类省略回答的意思全在上一句；只看孤立的两个字，System1 只能猜。

**一次请求问 11 个问题：**

| 问题 | 类型 | 选项 / 含义 |
|---|---|---|
| `situation` | choice | 场景性质 7 类：正常交流、表达关心、提出请求、开玩笑调侃、质疑批评、冒犯挑衅、提及共同经历 |
| `emotion` | choice | 16 种情绪，按效价 × 唤醒度分四组。正高：兴奋、喜悦、惊喜、自豪；正低：平静、满足、信任、感激；负高：愤怒、焦虑、恐惧、沮丧；负低：悲伤、无聊、厌恶、困惑 |
| `impulse` | choice | 12 种第一冲动。积极：探索、创造、分享、帮助、庆祝；防御：回避、对抗、掩饰、僵住；社交调节：亲近、退让；中性：无 |
| `intensity` | score | 情绪强度 |
| `engagement` | score | 投入意愿（想不想多聊） |
| `conflict` | score | 话题与"我"的价值观、专业、在意的事的冲突程度 |
| `requires_knowledge` | noul | 是否需要专业知识 |
| `user_intent` | choice | 对方意图 14 类（寒暄问候、闲聊、开玩笑、分享近况、提问、求助、征求意见、倾诉、邀约、批评质疑、挑衅冒犯、提起往事、正常讨论观点、其他） |
| `quick_action` | choice | 直觉行为 13 种（随口接话、调侃回去、反问对方、敷衍应付、关心对方、爽快答应、直接拒绝、顶回去、简单回答、岔开话题、愣住或追问、凭印象说、得想一想）。问到自己不掌握的东西时，按人格自然反应 |
| `needs_memory` | noul | 是否必须想起具体往事才能接话 |
| `needs_reasoning` | noul | 是否需要停下来认真想 |

**结果映射：**

- score 类（0–4 连续分）线性映射到 0–1：`intensity`、`engagement`、`conflict`
- `requires_knowledge`：概率 > 0.5 即为真
- `emotion`：取概率最高的作为主情绪；概率最高的前 3 个另存为 `emotion_probs`，写入接口返回和 trace，只用于观测，不参与路由和状态计算。它表示"选哪一个"的确定程度，不是几种情绪同时存在的比例；实测多数时候第一名 ≥ 0.9（如"考研上岸了"：喜悦 0.81 / 惊喜 0.14）
- `quick_action`：从概率分布里**单独拿出"得想一想"的概率**作为 `think_prob`，其余选项取概率最大的作为直觉行为，其概率记为 `action_confidence`
- `needs_memory`、`needs_reasoning`：保留原始概率，交给路由按阈值判断

失败时返回中性结果：平静、强度 0，`think_prob = 1.0`、`needs_reasoning = 1.0`，保证兜底走 System2。

---

## 认知路径选择

`_choose_route()` 按顺序判断，第一条满足的生效，返回路径和原因（`route_reason`，写入返回值和 trace）。阈值均可在运行时调整：

| 顺序 | 条件 | 路径 | route_reason |
|---|---|---|---|
| 1 | `needs_reasoning` ≥ `BRAIN_ROUTE_REASONING_THRESHOLD`（0.6） | deliberate | `needs_reasoning` |
| 2 | `think_prob` ≥ `BRAIN_ROUTE_THINK_THRESHOLD`（0.5） | deliberate | `think_prob` |
| 3 | 说话人在关系表里，直觉行为是"直接拒绝 / 顶回去"，且 `conflict` < `BRAIN_ROUTE_HOSTILE_MAX_CONFLICT`（0.4） | deliberate | `hostile_to_known` |
| 4 | `action_confidence` < `BRAIN_ROUTE_MIN_ACTION_CONFIDENCE`（0.55） | deliberate | `low_action_confidence` |
| 4.5 | 对方在说过去的事（意图"提起往事"、场景"提及共同经历"或 `needs_memory` 达阈值，任一成立），且召回的 Episodic 最高分 < `BRAIN_EPISODIC_CUE_MIN_SCORE` | deliberate | `recall_uncertain` |
| 5 | 使用知识 | memory | `knowledge` |
| 6 | `needs_memory` ≥ `BRAIN_ROUTE_MEMORY_THRESHOLD`（0.6） | memory | `needs_memory` |
| 7 | 并行召回的 Episodic 中 rerank 最高分 ≥ `BRAIN_EPISODIC_CUE_MIN_SCORE`（0.75） | memory | `episodic_cue` |
| 8 | 以上都不满足 | fast | `intuitive` |

- 第 3 条：对认识的人选了拒绝、话题却几乎没有冲突，多半是 System1 理解错了，交给 System2 复核。对陌生人的拒绝、或冲突确实很高时的拒绝，仍按直觉执行
- 第 4 条：直觉行为的概率分散时（多个选项都差不多），这个选择接近随机，不应该直接变成回答
- 第 4.5 条（想不起来）：被问到想不起来的往事，人会停一下想想。实测把"想不起来"作为事实写进直觉计划不管用，System3（角色模型）照样补出"绿萝还活着""钢笔是凌美的"；System2 会把怎么回（直说、含糊、反问，按人格）明确写进决策，System3 只负责说出来。System1 的意图和场景定义里明确包含"问起"往事（原来只写"说起"，"那支笔什么牌子来着"会被归为普通提问）
- 命中知识不会强制进 System2，但最低走 memory 路径，保证把相关经历一起召回。
- 第 6、7 条满足一条就带回忆。`needs_memory` 是小模型的一次二选一判断，很不稳定：同一句"我之前跟你说要去哪儿开民宿来着？"、同样的输入，两次分别给出 0.47 和 0.05，改问题措辞也只是换一种不稳定。而往事往往是被对方的话勾起来的：Episodic 召回本来就和 System1 并行，召回到和这句话高度相关的记忆，就该想起来，与知识"rerank 后还剩没剩"同理。实测（rerank 分数）：需要回想往事的句子 0.81–0.98；不需要的句子 0–0.66，其中较高的是"你觉得《百年孤独》好看吗"勾起的读书经历（0.66），这类被话题勾起的自身经历本来就该想起
- Episodic 召回在路由前等待完成，因为第 7 条要看它。它与 System1 并行，实测（150–230ms，System1 约 370–720ms）总是先完成，通常不额外耗时；等待时间记在 `episodic_wait`。走 fast 时不把召回结果带进计划。trace 的 `recall.episodic_cue` 记录这一轮的最高分

**知识注入时带可信度：** 每条知识的标题后附可信度标签（高 ≥ 0.75 / 中等 ≥ 0.5 / 低，多为商家或营销说法），给文字不给数值。System2 的提示词和直觉计划里都写明：可信度不高的内容只能说成"有资料称"，不能当成结论。

**是否使用知识：** 只看 rerank 之后还剩没剩（见 [Knowledge](#knowledge专业知识)）。剩下的条目按 rerank 分数排序，取前 `BRAIN_KNOWLEDGE_INJECT_TOP_K`（2）条，每条正文截断到 `BRAIN_KNOWLEDGE_CONTENT_CHARS`（1200）字。

System1 认为需要知识、但 rerank 后一条都不剩，**并且对方是在提问或征求意见**时，记为"知识缺口"（对方在下指令时，如"去查一下 X"，要回应的是这个请求，不算缺口），直觉计划里只写一句事实"我手上的资料里没有直接讲这个的内容"，不规定该怎么应对。

**不知道时怎么表现由人格决定**：直说不知道、凭印象说、岔开话题、反问对方，都是元认知的体现，因人而异。System1 在知识预取完成后才运行，输入里有"我对这个话题掌握的资料"（命中知识的标题，或"没有直接讲这个话题的资料"），由它按人格选出直觉行为（如"简单回答""凭印象说""岔开话题""反问对方"），System3 照此执行。元认知要写在人格里（`persona_system1` 决定第一反应，`persona_system2` 决定深思时的判断），不要写进说话习惯。

**人格三段各写什么：** 每条路径都必须守住的底线（身份、关系边界、诚实底线）写在 `persona_system1`：System1 和 System3 每一轮都看得到它，fast / memory 路径也一样。`persona_system2` 只有走 deliberate 时才用，只写深思时的判断方式（怎么权衡、怎么给建议、何时提醒风险）。底线只写在 system2 时，fast 路径看不到它：实测隐私、授权、风险提示这类边界在 system1 里也写了，两条路径的表现一致。但要注意，把"不知道就说不知道"写进 system1 并不能阻止 fast 路径编造它不可能知道的信息：Jarvis 被问天气、比特币价格、邮件数量、停车位、体检血压，18 次里 17 次编出具体数字（只有在人设里点名"实时天气"时天气一项才会如实回答，属于举例的拟合）。这类"不可能知道的现状"需要结构上的处理，尚未解决。

**直觉计划（fast / memory 路径）：** 由 `_quick_plan()` 生成，内容包括对方意图、"我的第一反应"及其说明；memory 路径附上最多 3 条想起的往事和命中的知识；往事由 `_format_recall()` 渲染（直觉计划和 System2 共用），附上记忆的边界："这就是我记得的全部，没写到的部分我不知道"；对方在说过去的事时还写明召回只是候选、不一定有对方说的那件，一条都没想起时写"想不起和这件事有关的具体经历"。给 System2 的往事一律按候选呈现（System1 常识别不出对方在说往事，而这句说明对任何一句话都成立）。只陈述事实，想不起来时怎么回由人格决定；对方在提问时，要求先正面回答再反问；并且始终提醒：没有真正做过的事（联系了谁、发了什么、安排好了什么）不能说成已经做了，只能说打算做或问对方要不要做。

**本轮时序事实（`THIS_TURN_FACT`）：** 一轮 react 里，回复先生成、先返回；对方托付的事（建待办、提醒、联系、查资料）在回复之后才在后台判断和执行，而且可能不执行（漏斗判为不需要、锁忙、模型失败）。所以直觉计划、System2 的输入、System2 的决策计划末尾都写明一句事实："这一轮我除了说话，还没有做任何事；对方刚才让我做的事，现在都还没做，之后才会去做"。只陈述时序，不规定措辞，回复自然是"好的，我来办 / 到时提醒您"这类答应，而不是"已设置提醒"。以前只有"计划里没写明已完成的不能说已完成"这条规则，模型会把"设置提醒""记下来"这类答应了就做的事当成说完即完成（Jarvis 实测 10/16 说成完成态，改后 2/33）。System2 的事实边界也补了一条：处境里的待办、事件只说明有这件事，不说明已经做了。fast / memory 路径没有 System2 的事实检查，这条约束只能放在计划里。

---

## System2：社交决策

实现：`cyber_brain._system2()`，模型 `DOUBAO_MODEL_PRO`，温度 `BRAIN_SYSTEM2_TEMPERATURE`（0.3），最大 token `BRAIN_SYSTEM2_MAX_TOKENS`（400）。只在 deliberate 路径调用。

**输入：**

- system 提示（`SYSTEM2_PROMPT`）：对方是谁；System1 给出的场景性质、第一冲动、投入意愿和冲突程度（都转成文字，不给数值）；人格画像（system1 + system2 两段）；决策步骤（前提检查 → 判断个人重要性 → 整合本能与人格 → 决定行为 → 事实边界 → 区分事实/推测/计划 → 撒谎规则）
- user 内容（`SYSTEM2_USER_TEMPLATE`）：当前状态与处境、Brief、本能反应、Episodic、知识（标注"仅供决策参考，技术细节不会说出口"）、工作记忆、对方这句话

**输出：** 为节省 token，按编号逐行输出，不写字段名：

| 编号 | 字段 | 说明 |
|---|---|---|
| 1 | `sanity` | 这话合理吗：合理 / 前提矛盾 / 缺少依据 |
| 2 | `thought` | 心里想（只用于决策，不交给 System3） |
| 3 | `action` | 怎么做 |
| 4 | `core` | 核心意思；不说话写"不说话" |
| 5 | `extra` | 补充 |
| 6 | `withhold` | 不说出口的内容 |
| 7 | `manner` | 分寸：语气、深度、主动性 |
| 8 | `lies` | 假话（没有则整行省略） |
| 9 | `source` | 事实来源（没有则整行省略） |

**解析（`_parse_decision`）：** 兼容"4 核心意思：xxx""4. xxx""4、xxx"等写法；同一编号重复时取第一次；"怎么做"和"核心意思"都缺失时视为失败。`lies` 按分号拆成列表。**沉默判定**：核心意思为空或含沉默词，并且"怎么做 + 核心意思"中出现沉默词（不说话、沉默、不吭声、不搭理、不理会、不回应）。

**失败处理：** 调用异常或解析失败时返回 `None`，由主流程退回直觉计划，保证总能说出话。

解析成功后，决策被格式化成 System3 的行为计划（`_format_decision`）：我决定、核心意思、补充、心里有但不说出口、分寸、要说的假话、事实来源。"心里想"不进入计划，避免 System3 把理由说出来。

---

## System3：语言表达

实现：`cyber_brain._system3()`，模型 `DOUBAO_MODEL_CHARACTER`，温度 `BRAIN_SYSTEM3_TEMPERATURE`（0.5），最大 token `BRAIN_SYSTEM3_MAX_TOKENS`（1000）。

**刻意只给少量信息：** 本能层的自我认知（`persona_system1`，标注为"用来把握身份和分寸，不是要说出来的内容"）、说话习惯（`speaking_manner`）、对方是谁、整体状态（不含处境）、工作记忆、对方这句话、本能反应（文字版）、行为计划。**不给 Brief、Episodic、知识和处境**：要说的事实都已经在计划里，多给只会让它自己补内容。

给自我认知的原因：认识对方（说话人信息）和认识自己一样是本能，每一轮都应在场，不能依赖是否走了 System2。否则 fast 路径上被问"你是谁、你能做什么"时，System3 不知道自己是谁，只能套用说话习惯里的句式。

提示词（`SYSTEM3_PROMPT`）要求：忠实执行计划、只表达对外内容、自然口语、长度由计划决定、不增加任何具体人物/地点/数字/经历、不得声称已经完成了计划里没写明已完成的动作（如"已为您联系"）、不用括号旁白、使用对方的语言、只在计划指定的假话句末加 `[谎言]`。

**后处理：**

- 信号中汉字占比 ≤ 10% 时，在输入末尾追加"用对方的语言回复"的提示
- `_only_spoken()` 去掉 1–30 字的括号旁白（如"（叹气）"）和包裹整句的引号；整段都是旁白时返回空串，按没接话处理
- 调用失败返回空串

---

## 说话人识别

`_format_speaker_info()` 把说话人转成一段描述，交给 System1/2/3：

1. 人际关系表（`agent:{agent_id}:relationships`）中有记录 → 使用关系描述，如"直属上级，关系一般，比较严格"
2. 工作记忆中出现过（原文记录的 `speaker`，或压缩块 / 当日摘要的 `speakers`）→ "有过简单接触的人"
3. 都没有，或未传 speaker → "一个不认识的纯陌生人"

每日整理时，前一天对话中出现过、但关系表里没有的人，会被自动记为"萍水相逢的陌生人，{日期}见过一面，说过几句话"。

---

## 谎言与事实来源

**谎言的完整链路：**

1. System2 在第 8 项写出要说的假话
2. System3 在说出口的对应句子末尾加 `[谎言]`
3. `parse_lies()` 按句切分，取出带标记的句子作为 `lies`，并从对外回复中去掉标记。System3 漏标时，用 System2 决定的假话兜底
4. 写入工作记忆时，`lies` 随记录一起保存；渲染工作记忆文本时，`_mark_lies()` 会把这些句子重新标上 `[谎言]`。因此后续轮次的 System2，以及滚动压缩和每日整理的 LLM，都能知道哪些话是"我"编的

**事实来源校验（`_validate_fact_sources`）：** 解析第 9 项中形如 `[事件]=来源` 的标注。来源看起来像记忆 ID（以 `memory_` 或 `mem_` 开头，或纯数字），但不在本轮召回的 Brief / Episodic 中时，改写为 `[事件]=推测` 并记录告警。

---

## 记忆系统

### Brief（语义记忆）

`memory_repository.search_brief()`，绕过 mem0 直接查 Qdrant：

1. **混合召回**（并行）：稠密向量检索（相似度阈值 `BRIEF_DENSE_SCORE_THRESHOLD`，0.3）+ BM25 检索，各取 `BRIEF_RERANK_CANDIDATES`（10）条
2. **RRF 融合**：`score = Σ 1 / (rank + 60)`，去重后取前 `BRIEF_RERANK_CANDIDATES` 条
3. **Rerank 过滤**（`BRIEF_RERANK_ENABLED`）：`qwen3-rerank` 打分，低于 `BRIEF_RERANK_MIN_SCORE`（0.42）的丢弃
4. **新旧冲突排序**：rerank 分差 ≤ `BRIEF_RECENCY_TIEBREAK_MARGIN`（0.05）且两条记忆的向量余弦 ≥ `BRIEF_CONFLICT_SIMILARITY`（0.85）时，视为同一事物的新旧版本，`updated_at` 较新的排前面（例如"住在上海"与"搬到杭州"）
5. 返回前 `BRAIN_BRIEF_TOP_K`（10）条

Brief 只用于 System1 和 System2。

### Episodic（情景记忆）

`memory_repository.search_episodic()`：

1. **查询改写**：生成 `QUERY_REWRITE_NUM` 个查询（含原查询）。默认值为 1，即不改写、不调用 LLM
2. **召回**：每个查询通过 mem0 做稠密向量检索，各取 `RERANK_CANDIDATE_SIZE / 查询数 + 1` 条，按 id 去重。**Episodic 不使用 BM25**
3. **跳过 rerank**：候选中最高分 > `RERANK_SKIP_THRESHOLD`（0.8）时，直接按向量分数返回前 top_k 条
4. **Rerank**：`qwen3-rerank` 重排，低于 `EPISODIC_RERANK_MIN_SCORE`（0.35）的丢弃
5. **冲突排序**：规则同 Brief，按 `created_at` 新的排前面
6. 返回前 `BRAIN_EPISODIC_TOP_K`（5）条

**写入来源：** [经历整理](#经历整理工作记忆--episodic)（滚动压缩前、每日整理时从工作记忆原文提取，`source: consolidation`）、待办执行结果、主动对话摘要、管理接口。Memory Gate 不再逐句写入。

**遗忘与重激活：**

- 遗忘见 [Decay Worker](#decay-worker记忆衰减)
- 重激活（`memory_repository.reactivate_related()`）：Memory Gate 评估对方的每句话时，**不论这句话本身是否值得记**，都会用它的向量在该 agent 的 Episodic 中检索（只用稠密向量），余弦 ≥ `REACTIVATION_THRESHOLD`（0.55）的记忆（最多 5 条）权重重置为 1.0，并写入 `last_reactivated_at`。重激活在本轮写入新记忆之前完成，不会命中刚写入的那条；查询向量复用 react 时已缓存的 embedding
- 召回排序不参考 `decay_weight`，权重只决定何时被删除

### Knowledge（专业知识）

`knowledge_memory_service`，通过 `GenericQdrantClient` 直接读写 Qdrant：

- 类型：`concept`、`howto`、`best_practice`、`troubleshooting`、`example`；来源：`experience`、`learning`、`project`、`documentation`。非法值会被替换为默认值
- 检索分两步：
  1. 稠密向量召回，按 `agent_id` 过滤，取 5 条，余弦 ≥ `BRAIN_KNOWLEDGE_SCORE_THRESHOLD`（0.5）
  2. `qwen3-rerank` 判断每条是否回答了这句话，低于 `BRAIN_KNOWLEDGE_RERANK_MIN_SCORE`（0.5）的丢掉

  只靠余弦分不开"话题相关"和"回答了问题"：知识库主题单一时，同主题的任何问题都能召回 0.55 以上的条目。实测（17 条抗衰老知识）：库里没有答案的问题（NAD+、二甲双胍、断食等），召回条目的 rerank 分最高 0.484；库里有答案的问题，正确条目的 rerank 分最低 0.529。rerank 调用失败时，退回只保留余弦 ≥ 0.6 的条目
- 召回条数（5）仍写在代码里，不在配置中
- 每轮 react 都会预取，是否注入见[认知路径选择](#认知路径选择)
- 写入来源：初始化接口、追加记忆接口。Memory Gate 不会写 Knowledge

### Relationship（人际关系）

Redis Hash，只做精确的人名匹配，用于[说话人识别](#说话人识别)。写入来源：初始化 / 更新接口、每日整理时自动记录的新陌生人（kind=human）。

每条关系记录对方的**类型**：`kind = human`（真人）或 `agent`（另一个 agent，必须同时记录对方的 `agent_id`）。类型只决定"我"[主动联系对方的方式](#主动触达)，不进入提示词；提示词里的关系描述与以前一样是"关系类型，描述"。校验在接口模型和存储层各做一次：kind=agent 必须有 agent_id，kind=human 不能有 agent_id。

### 工作记忆（PFC）

`prefrontal_lobe_memory`，两个 Redis List，均按时间从旧到新追加。每条原文记录带 `id` 和 `ts`（unix 秒），但渲染上下文时不输出，不增加 token：`id` 供 Memory Gate 和 System1 在后台给这条记录打上重要度、情绪标签，`ts` 供经历整理按时间间隔切分事件、标注发生日期。

- **追加**（热路径，不调用 LLM）：每轮先写对方的话，生成回应后再写"我"的话（沉默时写"（没有接话）"），各一次 `RPUSH`
- **滚动压缩**：`stm` 长度 > `PFC_STM_MAX_LEN`（100）时，在后台任务中处理最早的 `PFC_STM_COMPRESS_COUNT`（50）条：先[整理出经历](#经历整理工作记忆--episodic)写入 Episodic（压成摘要后细节就没了），再用 LLM 压缩成一个摘要块。先只读、两步都成功后才在事务里 `LTRIM stm + RPUSH compress`，任一步失败时原文保留。压缩块超过 `PFC_COMPRESS_MAX_BLOCKS`（16）时，把最早两块合并为一块。全程加锁（`PFC_LOCK_TTL`，300 秒）
- **读取**：一次 pipeline 读两个 list，压缩块在前、原文在后，渲染成 `[说话人] 内容` 的行。压缩块显示为 `[摘要]`，当日摘要显示为 `[昨日日程]`
- **每日整理**：见 [PFC Worker](#pfc-worker工作记忆每日整理)

工作记忆进入 System2 和 System3，也用于说话人识别。

---

## Memory Gate：记忆门控

实现：`memory_gate_service.evaluate_and_persist()`。每轮 react 生成回复后在后台执行，不阻塞本轮返回，**只评估对方说的话**（signal）。它**不写 Episodic，也不写 Brief**：经历由[经历整理](#经历整理工作记忆--episodic)从工作记忆原文中提取（那时看得到上下文和"我"说的话），信念只由[反思](#反思reflection-worker)写入。这里只做必须实时的四件事：

1. **重激活**：与 Episodic 重激活并行；对方再次提起旧事时，相似经历权重恢复、`updated_at` 更新（衰减和反思依赖它）
2. **打标签**：把 `importance`、`memory_type` 写到工作记忆里这条记录上（按记录 id，Lua 原子更新），作为整理经历时的"第一印象"参考；System1 的情绪与强度也同样标上
3. **个人信息**：见下文 profile
4. **处境与状态**：`importance ≥ 3` 时更新处境和整体状态（"明天要交方案"要马上让压力上升）

**评估**：一次 System One 请求问三个问题：

- `importance`（0–4 分，四舍五入为整数）：
  - 0：纯社交 / 语气，与用户本身无关
  - 1：用户在问问题，或只是此刻的瞬时状态
  - 2：近期确定的事（明天做 X），或不确定的意向（打算 / 想）
  - 3：用户的稳定特征（我是 / 我喜欢）、已完成的事件、通用经验规律
  - 4：安全 / 健康约束（过敏、禁忌）
- `memory_type`：`EPISODIC`（对话、事件、经历、约定、承诺）或 `SEMANTIC`（相对稳定的身份、偏好、资料、约束）
- `person_fact`：说话人有没有主动说出关于他自己的稳定个人信息

评估失败时按 importance = 2、SEMANTIC 处理。

**对话不直接写 Brief。** Brief 是"我的信条"（对自我的判断），对话里是"对方说的话"。对方一句重要的话落地时是经历，要经过反思（见[反思](#反思reflection-worker)）才可能变成对自己的认识。直接铸成信条等于替 agent 决定了它该怎么被影响，而且 Brief 不衰减、不重激活、没有删除接口。所以 Brief 只由初始化和反思写入。

**个人信息（profile）：** `person_fact` 概率 ≥ 0.6 时（与重要度无关），用一次 LLM 抽取到受控槽位（姓名、住址、职业、家庭成员、过敏与禁忌……共 16 个，见 `PROFILE_SLOTS`），融合进关系表里这个人的 `profile`。规则：

- **陌生人忽略**：不在关系表里的人说了个人信息也不记（路人一上来就说自己的事，人也会听着、但不会认真记下）
- **去重**：同一槽位、同一值只累计 `mentions` 和 `last_confirmed_at`，不新增条目
- **覆盖**：单值槽位（住址、职业……）新值生效，旧值进 `history`（最多 2 条），答得出"他以前住北京"
- **列表槽位**（家庭成员、过敏与禁忌、长期偏好、忌讳、习惯、重要日期）：同一值不重复，最多 8 条
- **不覆盖关系**：`relationship` / `description` 不受影响；更新关系时不传 profile 会保留已有资料
- 与关系描述共用一个 Hash（`agent:{agent_id}:relationships`）的 value，不额外开 key

读取时随说话人描述一起给 System1/2/3：`我对{人名}的了解（对方自己说过的，不是我自己的事）：- 住址：上海……`，与"我的信条"明确分开。

**更新处境：** 再发一次 System One 请求，判断我的处境有没有变化（有了新的、有确定日期的事；已有的事改期、解决或取消；我或我在意的人的持续状况变了；心里挂着的事变了）。只是请我做事、我答应做事（待办），或只是讲了一件已经过去的事（经历），都判为不变。有变化时用 `SITUATION_UPDATE_PROMPT` 让 LLM 输出新的完整处境 JSON，通过 `replace_situation()` 整体替换保存。提示词里有：
- 今天的日期和星期：相对时间（明天、下周三、月底）才能换算成准确日期。以前没有日期，"下周三"会被写成明天
- 说话人和这句话之前的几句原文：省略回答（"批了，月底最后一天上班"）要靠上文
- 处境与待办、经历的界限：我要去做的动作是待办，已经发生完的事是经历，都不写进处境
- 同一件事改期或有进展时改写原条目，不另加；已解决、取消、过期的直接删掉，不留"已完成"标记（渲染时不看状态字段，留着会被当成还没发生）；每类最多 3 条

新处境中 3 天内到期的每个事件让压力 +0.1。

---

## 经历整理：工作记忆 → Episodic

实现：`app/service/episode_consolidation_service.py`。对应人在睡眠中的情景记忆巩固：白天的对话先留在工作记忆里，再按"一件事"整理成经历存进长期记忆。和逐句实时判断相比：

- **看得到上下文**：单独一句没价值（"跟我。"），和上一句连起来才是一件事（离婚后孩子跟谁）
- **"我"说过的话也进经历**：答应了什么、拒绝了什么、哪句是假话（带 `[谎言]` 标记的会写明是假话），以后被追问时才想得起来
- **与旧经历合并**：同一件事再次被提起时，更新旧经历而不是重复存；`updated_at` 随之更新，反思据此找到反复发生的事

**两个入口**，处理的都是"即将从工作记忆里消失的原文"：滚动压缩之前（最早的 50 条要被压成摘要）、每日整理时（当天剩下的原文要被压成当日摘要）。已被滚动压缩的原文在压缩前已整理过，每日整理只处理 `stm` 里剩下的部分。

**提取**：一次 LLM 调用（`EPISODE_PROMPT`），输入是人格片段和这批原文（带说话人、谎言标记、Memory Gate 的重要度标签；相邻两句间隔超过 30 分钟时插入分隔）。按事件切分，输出 `{content, speakers, importance, emotion, emotion_charge, has_commitment, has_lie}`；寒暄、无实质内容的问答不记。经历用第一人称写，写清楚对方的名字，只写原文里有的内容；地名、人名、书名、数字、时间必须照原文的字写，不省略、不换说法（如不能把"大理洱海边上"写成"洱海边"或"云南"），合并时同样保留。

**合并**：同一件事换个说法，余弦约 0.72–0.81；但"不同的人、类似的事"（张三离婚 / 李四离婚）也有 0.73，单靠阈值分不开。所以和待办去重一样，先用 embedding 粗筛：余弦 ≥ `EPISODE_MERGE_SIMILARITY`（0.65）的最多 3 条作为候选，再由 LLM（`MERGE_PROMPT`）判断是不是同一件事。是同一件事且有新进展时，把两条合写成一条，就地改写（mem0 update，`created_at` 不变、`updated_at` 刷新、权重恢复、补写 BM25）；没有新内容时只重激活。不是同一件事则新增，metadata 带 `source: consolidation`、`speakers`、`occurred_at`（这批对话的日期）等。

**失败处理**：LLM 调用或解析失败时返回失败，调用方不压缩、不删除原文，下次再整理。

---

## 主体性：3F 模型

实现：`subjectivity_service`，所有数据保存在一个 `subject_context:{agent_id}` JSON 中。

```json
{
  "agent_id": "...",
  "situation": {
    "current_activity": "准备游戏上线",
    "time_sensitive_events": [{"description": "Steam首周销量统计", "deadline": "2026-11-20", "status": "pending"}],
    "ongoing_states": ["最近睡眠不好"],
    "attention_focus": ["游戏稳定性"]
  },
  "overall_state": {"mood": 0.0, "stress": 0.3, "energy": 0.7},
  "thoughts": [{"time": "...", "thought": "...", "related_to": "...", "target_person": "...", "urgency": 0.5, "expressed": false}],
  "pending_tasks": [{"task_id": "...", "content": "...", "source": "dialogue", "priority": 0.8, "deadline": "...", "related_person": "...", "interaction_type": "dialogue", "created_at": "...", "status": "pending", "vector": [...]}],
  "recent_done": [{"task_id": "...", "content": "...", "vector": [...], "status": "completed", "done_at": "..."}],
  "baseline": {"mood": 0.35, "stress": 0.45, "energy": 0.55},
  "state_updated_at": "...",
  "last_update": "..."
}
```

### Fact：处境（situation）

- 写入：初始化 / 更新接口、Memory Gate。待办执行不写处境：普通待办的"执行结果"是 LLM 编的一句话，以前会把"已与某某完成深谈"这类编出来、已经结束的事追加进持续状态
- 进入 prompt 的格式：`render_situation()` 统一渲染，System1/2（`format_for_prompts`）和念头生成（`format_situation_for_prompt`）共用，两边看到的处境一致。内容为当前活动；时间敏感事件（过期、无日期或格式错误的不展示，按截止日期由近到远取前 3 个，每个标注"今天 / 明天 / 还有 N 天"，只给日期时模型分不清是过去还是将来）；最多 3 个持续状态；最多 3 个当前关注（`attention_focus`）
- 给 System1、System2，以及 fast / memory 路径的直觉计划（"我近期的处境（对方问到时照此回答，没问到的不必提）"）。处境本来就是近期一直挂在心上的几件事，每类最多 3 条，不需要按相关性挑选：fast / memory 路径没有 System2，说话的 System3 拿不到处境时，被问到"证书哪天到期""读书会读哪本书"只能编。实测（Jarvis、林诗雨，问处境 12 句、闲聊 20 句）：不给处境时问处境答对 2/12；整段给处境后 12/12，闲聊中没有硬塞处境（偶尔顺口关心一句，如听说要去打网球时提醒这周睡得少，是自然的）。曾试过按相似度只挑相关的几条（余弦粗筛 + rerank），效果不比整段给更好（9/12），已去掉

### Feeling：整体状态（overall_state）

- `mood`（-1 最差 ~ 1 最好，0 为平常）、`stress`（0 ~ 1）、`energy`（0 ~ 1）三个连续值，三者互相独立。进入提示词时带上刻度（`format_state_lines()`），否则模型分不清 0.2 是好是坏
- **基线（`baseline`）**：这个人平时的状态，存在主体性上下文里。由初始化、或 `PUT` 传入 overall_state 时设定（同时把当前状态设成这个值）。旧数据没有基线时用中性值（0 / 0.3 / 0.7）
- **按时间结算（`settle_state()`）**：每次读取主体性上下文时，按 `state_updated_at` 到现在流逝的时间补算，不需要定时任务（没人读的值不必实时更新，读时算出来就是对的）：
  - 心情、压力指数回归到基线：`x += (基线 − x)·(1 − e^(−小时/τ))`，τ = `STATE_REGRESSION_TAU_HOURS`（6 小时：6 小时回去约 63%，一天基本回到基线；几分钟内几乎不变，连续对话不会被抹平）
  - 精力向基线恢复，τ = `ENERGY_RECOVERY_TAU_HOURS`（4 小时）×（1 + 压力），压力越大恢复越慢
  - 结算后把 `state_updated_at` 设为当前时刻。它和 `last_update` 分开：`last_update` 任何保存都会刷新（如加一条待办），用它计时会把回归错误地清零。分段结算和一次结算结果相同，所以读而不写、或读后因别的原因保存，都不会重复回归
- **调整（`adjust_overall_state`）**：先结算时间，再加增量。增量做边际递减：乘以"离边界还剩多少"占取值范围的比例，越接近边界越难再推、往回拉几乎不受影响，连续同向刺激会饱和而不是几轮就贴边
- **对话后异步调整**（按 System1 的情绪与强度，`EMOTION_STATE_EFFECT`，覆盖 System1 的全部 16 种情绪）。增量 = 系数 × 强度：

  | 分组 | 情绪（心情系数 / 压力系数） |
  |---|---|
  | 正面 · 高唤醒 | 兴奋 0.30/−0.05、喜悦 0.30/−0.10、惊喜 0.25/−0.05、自豪 0.25/−0.10 |
  | 正面 · 低唤醒 | 平静 0.05/−0.15、满足 0.20/−0.15、信任 0.10/−0.10、感激 0.20/−0.10 |
  | 负面 · 高唤醒 | 愤怒 −0.30/0.20、焦虑 −0.20/0.20、恐惧 −0.30/0.25、沮丧 −0.30/0.15 |
  | 负面 · 低唤醒 | 悲伤 −0.30/0.05、无聊 −0.10/0、厌恶 −0.20/0.10、困惑 −0.05/0.05 |

  以前只处理了 9 种（还包括 System1 不会输出的"高兴""放松"），信任、悲伤、无聊等不改变任何状态：主要被判成"信任"的 Jarvis 状态从不变化，主要被判成"焦虑"的林诗雨没有回落，累积到心情 −1、压力 1 后一直贴在边界
- **精力消耗**：每轮对话消耗 `ENERGY_COST_PER_TURN`（0.02）×（1 + 唤醒度 × 强度），高唤醒情绪最多翻倍。精力不再由压力换算（以前是 `1 − stress × 0.7`，表达不了"累但没压力"）
- 心情或压力的变化超过 0.01 时记录 `emotion` 生命事件（通过 `trace_id` 关联本轮对话）；只扣精力时不记。处境更新时，3 天内到期的事件让压力 +0.1（同样受边际递减）
- 给 System1、System2 和 System3

### Fabrication：念头（thoughts）

- 生成（`generate_thought`）：给 LLM 当前时间、处境、整体状态，让它输出一个 20 字以内的内心独白，以及想找的人、紧迫度、关联事件。**处境为空时不生成**
- 来源一：对话后异步生成。System1 的情绪强度 > 0.1 或投入意愿 > 0.5 时触发，结果保存到 `thoughts`（按时间只保留最近 3 个），并记录 `thought` 事件
- 来源二：Thought Generator Worker 定时生成。这类念头**不保存到 `thoughts`**，只送进待办漏斗判断是否转成待办
- 念头不进入 react 的任何 prompt

---

## 待办与自主行为

### 能力（Capabilities）

联网搜索是一个很具体的机制：对话中答应"去查"的事几秒后就交来一份带出处的整理，结果写进知识库，再主动通知对方。这适合个人知识库助手，但不适合拟人角色：一个说"回头帮你查查"的朋友，半分钟后交来结构化清单，就不像人了；知识库也会被搜索摘要填满。

所以能不能联网是 **agent 级别的能力开关**（`agent:{agent_id}:capabilities`），不写进人格、也不交给提示词判断：这是 agent 客观上有没有这个工具，不是性格。由 `initialize_agent` / `PUT /admin/agent` 的 `capabilities` 设置，**默认没有任何能力**。

| | 有 `web_search` | 没有（默认） |
|---|---|---|
| 对话 / 念头漏斗 | 问 `needs_web_search`（对话漏斗还问 `execute_now`） | 不问，任务里两项都是 false，System One 请求少两个问题 |
| 立即执行 | 联网任务写入后立即执行 | 不立即执行 |
| 执行 | 联网搜索 → 写入 Knowledge → 通知委托人 | 按普通任务执行，不联网 |

没有联网能力时，"回头帮你查查"仍然是一个普通待办，到时候按普通任务执行。

### 待办的产生（`add_todo`）

**来自对话**（每轮 react 结束时在后台执行，不阻塞返回）。判断和提炼都会带上最近几条对话（与 System1 相同）："查一下""好，去吧"这类确认的内容全在上一句，只看当前一句会丢掉要做的事：

1. System One 一次请求判断以下几件事（后两项只在 agent 有 `web_search` 能力时问，见[能力](#能力capabilities)）：
   - `has_commitment`：对方明确请求了某个行动，且"我"明确答应了
   - `needs_interaction`：完成任务是否需要去找某人沟通
   - `needs_web_search`：完成任务是否需要上网查资料获取还不知道的信息
   - `execute_now`：是不是现在就该着手（对方要求现在做，或没提时间且立刻能动手）；指定了以后的时间、要等条件、到时候的提醒都算否
2. `has_commitment` 的 yes 概率 ≥ 0.85 时，交给 lite 模型（`DOUBAO_MODEL_LITE`，温度 0，最多 80 token）完成两件事（不再二次判断有没有任务：实测同一输入约 1/4 的概率误答"无"）：
   - 把对方的原话提炼成任务描述（30 字内，不照抄原话）
   - 按话里的时间推算截止日期（`YYYY-MM-DD`），提示词里带今天的日期和星期，"明早""下周三""月底"都能推算；没提时间则为空
   
   提示词约 150 token，输出是一两行纯文本（不用 JSON），实测单次约 1 秒
3. 通过后创建：优先级 0.8，相关人为说话人

需要沟通时，任务带 `interaction_type: "dialogue"`；需要联网时带 `needs_web_search: true`；现在就该做时带 `execute_now: true`。

**立即执行：** 待办照常写入（去重仍然生效），写入后如果 `execute_now = true`、是**联网任务**且 agent 有 `web_search` 能力，`cyber_brain` 立即调用 `task_executor.run_task()` 执行，不等 Task Reminder 排期，执行完从待办中移除；执行失败则留在待办里，由 worker 下次检查重试。只放开联网任务，原因是：
- 普通任务的"执行"是 LLM 编一句完成结果，立即执行会让下一轮对话就出现没做过的事
- 对话中生成的社交任务，联系对象固定是当前说话人（正在对话的人），立即"主动联系"没有意义

`run_task()` 用一个进程内集合记录正在执行的任务，对话中立即执行和 worker 定时检查碰到同一个任务时不会重复执行。

**来自念头**（Thought Generator Worker）：一次 System One 请求同时判断 `need_task`（有没有想去做的具体事情，包括想去了解、查明、补充某个知识）和 `needs_web_search`（只在有 `web_search` 能力时问）。`need_task` 的 yes 概率 ≥ 0.8 直接创建（优先级即该概率），0.6–0.8 再让 LLM 确认，更低不创建。

> 判断用的是 System One 返回的 `probabilities["yes"]`，不是 `confidence`：后者表示判断有多确定（约为 1 − 熵），yes 概率 0.755 时 `confidence` 只有 0.197，不能当作"yes 的概率"用。

**存储与去重：**

- `pending_tasks` 只存待执行的任务，按优先级排序，**不设数量上限**
- 创建前去重，比较范围是全部待执行任务，以及 `TODO_DEDUP_WINDOW_HOURS`（24 小时）内完成或取消的任务（`recent_done`）：
  1. 内容完全相同 → 重复
  2. 否则用 embedding 粗筛：余弦 ≥ `TODO_DEDUP_SIMILARITY`（0.6）的任务作为候选，最多 3 个
  3. 由 System One 判断新任务是否和某个候选是同一件事（做的事、针对的人、目的都相同），是则跳过。System One 不可用时，只把余弦 ≥ 0.9 的当作重复

  只靠相似度阈值分不开两类情况：同一件事换个说法，余弦常在 0.65 左右；"提醒明早吃药"和"提醒明早开会"却有 0.78。所以由 embedding 缩小范围，再让模型判断
- 任务的向量随任务一起存储（`vector` 字段，对外接口不返回）；早期没有向量的任务在第一次比较时补算
- `pending_tasks` 的读-改-写由 Redis 锁串行化，对话待办、念头待办、任务完成同时发生时不会互相覆盖
- task_id 使用 uuid
- 创建时记录 `todo` 事件。待办只能由 Task Reminder 执行完成，目前没有取消入口（`cancel_task()` 已实现）

### 待办的执行（Task Reminder Worker + `task_executor`）

**执行时机（`_should_execute`）：**

| 情况 | 规则 |
|---|---|
| 已过截止日期 / 截止日当天 | 执行 |
| 距截止 1–2 天 | 按任务 ID 哈希约 50% 提前执行（创建满 1 小时后） |
| 距截止 3–7 天且优先级 > 0.7 | 按任务 ID 哈希约 30% 提前执行（创建满 2 小时后） |
| 无截止日期 | 优先级 > 0.7：创建满 2 小时；> 0.5：满 12 小时；其他：满 48 小时 |

用任务 ID 的哈希代替随机数，同一任务每次检查的结论一致。

**执行方式：**（`execute_task` 按以下顺序判断，联网优先：决策模型常把"查一下补进知识库"同时判为需要联系人，但这类任务的实质是获取信息）

- **联网任务**（`needs_web_search = true`，且 agent 有 `web_search` 能力；执行时再查一次能力，没有能力的按普通任务执行。开关关闭前生成的、或通过接口直接写入的联网任务不经过漏斗，靠这一步拦住）：
  1. LLM 把待办改写成**两个**查询词：一个直接查这件事，一个查它的风险、局限、反面证据或争议。两次 Tavily 搜索并行（`app/client/web_search_client.py`，每次取 `WEB_SEARCH_MAX_RESULTS` 条），结果按链接去重合并。只搜一次时，结果偏向哪一边，知识就偏向哪一边
  2. 搜索优先返回 `WEB_SEARCH_PREFER_DOMAINS` 中的站点（学术期刊、PubMed、卫健委等，Tavily 的 `prefer` 模式：只是优先，不够时仍会补充其他站点）
  3. LLM 先判断每条结果的**来源类型**（学术研究 / 官方或专家共识 / 新闻媒体 / 商家或营销），再**只根据摘录**整理出 1–3 条知识。正文要写明说法出自哪类来源，正反证据都写，只有动物实验或小样本时要写明，不能把商家说法写成结论，并用 `[1][2]` 标注出处
  4. 每条知识写入 Knowledge：可信度按引用来源中**最可靠的类型**给（学术研究、官方或专家共识 0.75，新闻媒体 0.55，商家或营销 0.35），正文末尾附每条来源的标题、类型和链接，`source = learning`
  5. 写入一条 Episodic 经历（"为了 X 上网查了资料，整理进知识库：……"或"……没找到可靠内容"），metadata 带两个查询词和来源链接
  
  未配置 `TAVILY_API_KEY` 或搜索失败时，任务返回失败、**保持待执行**，下次检查再试，不编造结果
- **普通任务**：LLM 用第一人称模拟一句完成结果，以"[自主完成] …"写入 Episodic，再让 LLM 判断是否更新处境（当前活动 / 事件完成 / 新的持续状态）。模拟结果不再写入 Knowledge
- **社交任务**（`interaction_type = dialogue`）：先校验 `related_person` 是已存在的 agent（有 persona），不是则**取消任务**（不调用 react，避免以人名为 agent_id 创建出游离数据，也避免 worker 每次检查都重试）。校验通过后，LLM 生成一句 15 字以内的开场白，以发起方为说话人，调用**目标 Agent**（`related_person` 作为目标 agent_id）的 `react()`；把双方的话写成一段经历存入发起方的 Episodic，并在发起方处境中追加 `recent_contacts`（保留最近 10 条）

执行成功后任务**从 `pending_tasks` 移除**，内容和向量转入 `recent_done`（只用于去重，超出窗口自动清理），完整任务记录写入 `todo` 生命事件（action=completed），并记录 `task_execution` 事件。服务启动时会把旧版本遗留在 `pending_tasks` 中的已完成/已取消任务迁移出去（幂等）。

---

## 主动触达

实现：`app/service/outreach_service.py`。agent 主动开口说话的两种来源：

- **任务完成**：替某人做的联网任务查完后（`notify_task_done`），告诉委托人查到了什么；社交任务（`interaction_type = dialogue`）执行时主动联系任务里的那个人
- **念头**：Thought Generator 定时生成的念头带有 `target_person`（生成时只能从关系表里的人名中选）。`maybe_express_thought` 用一次 System One 判断此刻要不要说给对方听，输入是"我是谁"（`persona_system1`）、对方是谁、最近的对话和这个念头；yes 概率 ≥ 0.6 才说。要不要说由人格决定，刚说过类似的话也会被判为不说

**投递**（`deliver`）按关系表里对方的 kind 决定，不靠名字推测：

| 对方 | 方式 |
|---|---|
| kind = human | 写入发件箱 `outbox:{agent_id}:{人名}`，前端调用 `GET /brain/fetch_talk?agent_id=&human=` 取走（取走即删除） |
| kind = agent | 用关系里记录的 agent_id 调用对方的 react，说话人用对方关系表里对"我"的称呼（对方没有记录时用"我"的 agent_id）；对方的回复写入"我"的工作记忆 |
| 不在关系表里 | 不主动联系（陌生人），任务取消，也不会以人名为 agent_id 创建游离数据 |

说出口的话由 System3 生成（按说话习惯，说话人是对方，"此刻对方说"是"对方没有说话，这是我主动开口"），然后写入"我"的工作记忆，标注"（主动找某人说）"，之后的对话知道自己说过；同时记录 `proactive_talk` 生命事件。

### 反思（Reflection Worker）

实现：`app/worker/reflection_worker.py`。把经历沉淀成信念，模拟人在睡眠中的记忆重放与巩固。**Brief 只由初始化和这里写入。**

**不单独定时**：每日整理（PFC Worker）把所有 agent 的经历整理完后调用 `run_once()`。反思要看到今天刚整理出、刚被重激活的经历，必须排在整理之后。执行前先把时间记进 Redis，20 小时内不会重复执行。

**候选**：Episodic 中 `updated_at != created_at`（被重新提起过；重激活时会更新 `updated_at`）且 24 小时内更新过的记忆。事实型（整理出的经历 `source = consolidation`，以及旧版 Memory Gate 写入的 `gate_type = SEMANTIC`）必选；`importance ≥ 3` 的经历按 `REFLECTION_INCLUDE_HIGH_IMPORTANCE` 纳入；`type = task_completion` 的系统回执排除。按重要性排序取前 `REFLECTION_MAX_CANDIDATES`（30）条。

**判断**：一次 LLM 调用，输入是人格片段（`persona_system1`）、全部现有 Brief、候选经历（带编号）。提示词见 `REFLECTION_PROMPT`，按七条标准逐条判定：归属（别人的话不是信念）、稳定性（一次性事件不是信念）、独立性、可检验性、不重复、不越界（个人事实归 profile）、不反噬（不把别人的指责照搬成对自己的判断）。输出 JSON：`{memory_ids, kind, action, target_index, text, reason, confidence}`。

**写回（有界步长）**：每个 agent 每轮最多 `REFLECTION_MAX_ACTIONS`（2）条；`confidence < REFLECTION_MIN_CONFIDENCE`（0.75）不写；`add` 前先查重（与本轮已写的文本比，并用 Brief 检索现有信念，rerank ≥ 0.85 视为同一件事），避免反思模型写出"多一个限定词"的近似条目；`revise` / `merge` 不删除旧条目，只给它写 `superseded_by`，检索时忽略但保留历史（可回滚、可解释）。每条反射信念带 `source: reflection`、`evidence_ids`、`confidence`、`reason`，并记一条 `thought_type = reflection` 的生命事件。

## 后台 Worker

四个 Worker 在服务启动时通过 `asyncio.create_task()` 启动，随服务停止。

| Worker | 执行时机 | 对象 |
|---|---|---|
| Decay Worker | 距上次执行满 `DECAY_WORKER_INTERVAL`（86400 秒）时执行；上次执行时间记在 Redis，重启不会额外触发 | Episodic 集合中出现过的所有 agent |
| PFC Worker | 每天 `PFC_DAILY_HOUR`（4）点；整理完后触发反思 | 有工作记忆的所有 agent |
| Thought Generator | 启动后**先等待**一个 `THOUGHT_GENERATOR_INTERVAL`（3600 秒），之后每个间隔执行 | 有主体性上下文的所有 agent |
| Task Reminder | **启动时立即执行一次**，之后每 `TASK_REMINDER_INTERVAL`（3600 秒） | 有主体性上下文的所有 agent |

### Decay Worker：记忆衰减

对每条 Episodic 记忆：`decay_weight × DECAY_FACTOR`（0.96），结果低于 `DECAY_MIN_WEIGHT`（0.2）就删除，否则写回。按默认参数，一条一直没被再次提起的记忆大约在第 40 次衰减（约 40 天）时被删除；被重激活后权重回到 1.0，重新计算。只作用于 Episodic，Brief 和 Knowledge 不衰减。

调度：上次执行的 unix 时间戳存在 Redis 的 `worker:decay:last_run`。Worker 每次醒来都重新计算距下一次执行还有多久，到期才执行，执行后（无论成败）更新时间戳。因此服务重启不会多衰减一轮，频繁重启也不会让衰减一直轮不到；从未执行过时启动即执行。

### PFC Worker：工作记忆每日整理

对每个 agent（与滚动压缩共用锁，锁时长固定 3600 秒）：

1. 以当前 `stm` 长度为游标，读取全部压缩块和游标之前的原文
2. 从游标之前的原文中[整理出经历](#经历整理工作记忆--episodic)写入 Episodic
3. LLM 生成一段 150 字以内的第一人称当日摘要
4. 用这一块当日摘要（带当天所有说话人）**替换**整个压缩队列，删除游标之前的原文（整理期间新到的消息保留）
5. 把前一天出现过的新说话人记入人际关系表
6. **复核处境**（`_review_situation`，`SITUATION_REVIEW_PROMPT`）：对话中的处境更新每次只看一句话，久了会留下已经结束的状况、重复的条目、混进来的待办和已经过去的事。先用代码删掉过期事件，再把当前处境（逐条编号）、待办列表、当日摘要和当天的对话原文交给 LLM，**对每一条分两步判断**：
   - 第一步 `kind`：这一条本身是什么：处境（持续的状况、有日期的事、挂着的事）、待办（我要做的动作）、已经发生完的事。只看条目本身，和今天有没有提到无关；待办和已发生完的事由代码直接删掉
   - 第二步 `action`：只对处境判断 keep / delete（今天的对话表明已结束、解决、取消，或重复）/ rewrite（有新进展）。**没有证据表明已经结束的 keep**：很久没被提起不等于结束了。带相对时间的说法（"这周""最近两周"）只在今天的对话表明不成立时才改，不换算成具体日期（不知道原话是哪天写的）

   新增只收处境，与已有条目高度相似的不再加。改写和新增只写今天对话里出现过的信息。由代码按决定组装：keep 的原样保留，漏判的按 keep；状况、关注每类最多 3 条，超出时丢新增的。当天没有对话原文时不调用模型；输出不合法时重试一次，仍失败则处境保持原样。

   为什么拆两步：最初让模型直接输出整份新处境，删除是否生效每次不一样；改成逐条决定后，模型常以"今天没提到"为由把待办、已发生完的事留下，因为"没有证据就保留"和"不属于处境就删"放在同一层判断。拆开后实测 5 轮全部符合预期。

经历整理或摘要失败时不做任何清理，数据留到下一次。当日摘要只留在工作记忆中，不写入长期记忆。所有 agent 整理完后触发[反思](#反思reflection-worker)。

### Thought Generator 与 Task Reminder

见[主体性](#fabrication念头thoughts)和[待办与自主行为](#待办与自主行为)。

---

## 生命事件

`agent_life_event_service` 把 Agent 经历的事写入 MongoDB。MongoDB 未连接时只打一条告警并跳过，不影响主流程。

| event_type | 记录时机 |
|---|---|
| `react_input` | 收到一句话 |
| `react_output` | 说出一句话（或沉默）。顶层带耗时 `rt` 和完整 `trace`：说话人描述、谎言、路由、System1 全字段、System2 决策、三路召回的 id/分数/文本、对话前的整体状态、工作记忆长度、各阶段耗时、本轮每次 LLM 调用的 token 用量 |
| `thought` | 对话后生成念头 |
| `todo` | 待办创建、完成（取消方法已实现，但目前没有调用方） |
| `situation` | 处境被设置或更新 |
| `emotion` | 整体状态变化（含前后值和增量） |
| `task_execution` | 待办执行成功或失败 |
| `proactive_talk` | 主动对某人说的话（任务完成的通知、想说出口的念头），含投递方式 outbox / react |

LLM token 用量通过 `contextvars` 收集：`react()` 开始时调用 `start_usage_collection()`，同一上下文中的 `chat` 调用都会把用量追加进来。

查询：`GET /admin/timeline/{agent_id}?event_types=react_input,react_output&page=1&page_size=50`。

---

## 管理接口与运行时配置

| 接口 | 行为 |
|---|---|
| `POST /admin/initialize_agent` | 一次性写入人格、Brief/Episodic/Knowledge、人际关系、处境、整体状态、念头、待办。用 Redis 锁防止并发重复初始化；agent 在 Redis 或 Qdrant 中已有任何数据时返回 409。**只有同时提供 system1 和 system2 时才会写入人格**。`capabilities` 设置能力（目前只有 `web_search`），不传即没有任何能力 |
| `GET /admin/agents` | persona（`persona:*:system1`）与 subject_context 的 agent 取并集 |
| `GET /admin/agent/{agent_id}` | 按初始化接口的结构返回完整信息 |
| `PUT /admin/agent/{agent_id}` | 覆盖式更新，只更新传了的字段。situation / overall_state / thoughts / pending_tasks 整体替换（pending_tasks 中 status 不是 pending 的会被忽略）；relationships 先清空再写入（传 `[]` 即清空；仍在列表里的人保留从对话中积累的个人信息 profile）；capabilities 整体替换（传 `[]` 即关闭全部能力）。agent 不存在返回 404 |
| `POST /admin/agent/{agent_id}/memories` | 追加 Brief / Episodic / Knowledge，每条都作为新记录写入。agent 不存在返回 404 |
| `DELETE /admin/agent/{agent_id}/pfc` | 删除 `stm` 和 `stm:compress` |
| `GET /brain/fetch_talk` | 参数 `agent_id`、`human`：取走 agent 主动想对这个人说的话，取走即删除，见[主动触达](#主动触达) |
| `GET /admin/config` | 返回 36 个可调参数的名称、当前值、默认值、类型、描述、取值范围 |
| `PUT /admin/config` | 修改参数，见下文 |

### 运行时配置

- 白名单在 `app/config/runtime_config.py`，包括 Brief/Episodic 检索、记忆衰减、工作记忆、react 路由与知识召回和注入、System2/3 温度和最大 token、记忆重激活阈值、待办去重、路由兜底、System1 最近对话条数等 36 项。这些参数在代码中都是调用时读取 `settings.X`，修改后下一次使用即生效
- 修改流程：整体校验（白名单、严格类型、取值范围，以及 `PFC_STM_COMPRESS_COUNT ≤ PFC_STM_MAX_LEN`），任意一项不合法则全部不改；校验通过后**先写回 `.env`**（临时文件 + 原子替换，保留其他行和注释），**再修改内存**中的 `settings`
- 配置优先级：环境变量 > `.env` > `settings.py` 默认值。`settings` 设置了 `extra="forbid"`，`.env` 中出现未定义的配置项时服务无法启动

---

## 模型调用一览

| 用途 | 模型 / 接口 |
|---|---|
| System1、Memory Gate 评估、是否更新处境、待办漏斗 | System One 决策接口（`DOUBAO_MODEL_LITE`） |
| System2 | `DOUBAO_MODEL_PRO` |
| System3 | `DOUBAO_MODEL_CHARACTER` |
| 工作记忆压缩与每日摘要、处境更新、念头生成、待办 LLM 判断、任务模拟执行、开场白、查询改写 | `DOUBAO_MODEL_LITE`（`chat` / `generate` 的默认模型） |
| Embedding | `EMBEDDING_MODEL`（OpenAI 兼容接口），维度 `EMBEDDING_DIMENSION` |
| Rerank | `RERANK_MODEL`（`qwen3-rerank`，DashScope） |

mem0 内部的 LLM 被替换为关闭深度思考的 `DoubaoLLM`，但由于写入时使用 `infer=False`，当前流程不会调用它。

---

## 已知问题与限制

**设计层面的限制：**

- **Episodic 冲突排序实际不生效**：排序依赖记忆向量，但 mem0 的检索结果不带向量，相似度恒为 0，结果只按 rerank 分数排序
- **念头不参与对话**：念头不进入任何 prompt；Thought Generator 生成的念头只用于产生待办，不保存
- **Memory Gate 只看对方的话**："我"说过的话（包括承诺和谎言）只留在工作记忆里，不会进入长期记忆
- **整体状态取值范围不一致**：初始化 / 更新接口要求 `mood` 在 [0, 1]，对话中的调整允许 [-1, 1]
- Knowledge 的召回条数（5）写在代码里，不在配置中
- Knowledge 的 rerank 阈值是在一个 17 条、单一主题的知识库上标定的，两侧余量都不大（0.484 / 0.529）；知识库规模和主题变化后需要重新验证
- `/admin` 接口没有鉴权

---

## 理论参考

- Kahneman (2011). *Thinking, Fast and Slow*：双系统理论
- Ebbinghaus (1885). *Über das Gedächtnis*：遗忘曲线
- Baddeley & Hitch (1974). Working Memory Model：工作记忆
- Scoville & Milner (1957). H.M. 病例：海马体与情景记忆
- Plutchik (1980). Emotion Wheel：离散情绪
- Russell (1980). Circumplex Model of Affect：效价 × 唤醒度的情绪分组
- Cannon (1915). Fight-or-Flight Response：应激反应与防御类冲动
- Raichle et al. (2001). Default Mode Network：自发念头
