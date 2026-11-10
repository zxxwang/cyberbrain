# CyberBrain 架构与实现

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
14. [后台 Worker](#后台-worker)
15. [生命事件](#生命事件)
16. [管理接口与运行时配置](#管理接口与运行时配置)
17. [模型调用一览](#模型调用一览)
18. [已知问题与限制](#已知问题与限制)
19. [理论参考](#理论参考)

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
| `agent:{agent_id}:relationships` | Hash | field 为人名，value 为 `"关系类型，描述"` | 初始化 / 更新接口、每日整理 |
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
  │    ├─ Episodic 召回（单独的 task，fast 路径会被取消）
  │    ├─ Brief 召回 ──▶ System1
  │    └─ Knowledge 预取
  │
  ├─ 判断是否使用知识 → 选择认知路径
  │    ├─ fast：取消 Episodic，直觉计划
  │    ├─ memory：等 Episodic，直觉计划 + 想起的往事 + 知识
  │    └─ deliberate：等 Episodic → System2 → 决策计划（失败时退回直觉计划）
  │
  ├─ System3 生成说出口的话（System2 决定不说话时跳过）
  ├─ 解析 [谎言] 标记；把"我"的回应（或"（没有接话）"）写入工作记忆
  │
  ├─ 同步执行（返回前完成）：
  │    ├─ Memory Gate：评估并持久化对方这句话
  │    └─ 记录 react_output 事件（含完整 trace）
  │
  ├─ 异步执行（不阻塞返回）：判断这轮对话是否产生待办；根据情绪调整整体状态、可能生成念头
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

**输入上下文**（按顺序拼接）：对方是谁 → 主体性上下文（整体状态 + 处境）→ System1 人格片段 → Brief 记忆 → 刚才的对话（工作记忆末尾 `BRAIN_SYSTEM1_RECENT_TURNS` 条原文，默认 4 条，不含本句和压缩摘要）→ 外部信号（信号前再标一次说话人，小模型对紧邻信号的内容权重更高）。

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
| `quick_action` | choice | 直觉行为 12 种（随口接话、调侃回去、反问对方、敷衍应付、关心对方、爽快答应、直接拒绝、顶回去、简单回答、岔开话题、愣住或追问、得想一想） |
| `needs_memory` | noul | 是否必须想起具体往事才能接话 |
| `needs_reasoning` | noul | 是否需要停下来认真想 |

**结果映射：**

- score 类（0–4 连续分）线性映射到 0–1：`intensity`、`engagement`、`conflict`
- `requires_knowledge`：概率 > 0.5 即为真
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
| 5 | 使用知识 | memory | `knowledge` |
| 6 | `needs_memory` ≥ `BRAIN_ROUTE_MEMORY_THRESHOLD`（0.6） | memory | `needs_memory` |
| 7 | 以上都不满足 | fast | `intuitive` |

- 第 3 条：对认识的人选了拒绝、话题却几乎没有冲突，多半是 System1 理解错了，交给 System2 复核。对陌生人的拒绝、或冲突确实很高时的拒绝，仍按直觉执行
- 第 4 条：直觉行为的概率分散时（多个选项都差不多），这个选择接近随机，不应该直接变成回答
- 命中知识不会强制进 System2，但最低走 memory 路径，保证把相关经历一起召回。
- fast 路径不等 Episodic 召回，直接取消该任务。

**是否使用知识：** System1 判断需要知识，或预取结果的最高分 ≥ `BRAIN_KNOWLEDGE_SCORE_THRESHOLD`（0.5）。使用时取前 `BRAIN_KNOWLEDGE_INJECT_TOP_K`（2）条，每条正文截断到 `BRAIN_KNOWLEDGE_CONTENT_CHARS`（1200）字。由于预取本身的相似度阈值也是 0.5，**实际上只要预取到知识就会使用**。

**直觉计划（fast / memory 路径）：** 由 `_quick_plan()` 生成，内容包括对方意图、"我的第一反应"及其说明；memory 路径附上最多 3 条想起的往事（并说明"这里没有的细节就是记不清了"）和命中的知识；对方在提问时，要求先正面回答再反问；并且始终提醒：没有真正做过的事（联系了谁、发了什么、安排好了什么）不能说成已经做了，只能说打算做或问对方要不要做。fast / memory 路径没有 System2 的事实检查，这条约束只能放在计划里。

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

**刻意只给少量信息：** 说话习惯（`speaking_manner`）、对方是谁、整体状态（不含处境）、工作记忆、对方这句话、本能反应（文字版）、行为计划。**不给 Brief、Episodic、知识和处境**：要说的事实都已经在计划里，多给只会让它自己补内容。

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

**写入来源：** Memory Gate（重要经历，原文前加"{说话人}对我说："）、待办执行结果、主动对话摘要、管理接口。

**遗忘与重激活：**

- 遗忘见 [Decay Worker](#decay-worker记忆衰减)
- 重激活（`memory_repository.reactivate_related()`）：Memory Gate 评估对方的每句话时，**不论这句话本身是否值得记**，都会用它的向量在该 agent 的 Episodic 中检索（只用稠密向量），余弦 ≥ `REACTIVATION_THRESHOLD`（0.55）的记忆（最多 5 条）权重重置为 1.0，并写入 `last_reactivated_at`。重激活在本轮写入新记忆之前完成，不会命中刚写入的那条；查询向量复用 react 时已缓存的 embedding
- 召回排序不参考 `decay_weight`，权重只决定何时被删除

### Knowledge（专业知识）

`knowledge_memory_service`，通过 `GenericQdrantClient` 直接读写 Qdrant：

- 类型：`concept`、`howto`、`best_practice`、`troubleshooting`、`example`；来源：`experience`、`learning`、`project`、`documentation`。非法值会被替换为默认值
- 检索：稠密向量，按 `agent_id` 过滤，取 5 条，相似度阈值 0.5（这两个值目前写在代码里，不在配置中）
- 每轮 react 都会预取，是否注入见[认知路径选择](#认知路径选择)
- 写入来源：初始化接口、追加记忆接口。Memory Gate 不会写 Knowledge

### Relationship（人际关系）

Redis Hash，只做精确的人名匹配，用于[说话人识别](#说话人识别)。写入来源：初始化 / 更新接口、每日整理时自动记录的新陌生人。

### 工作记忆（PFC）

`prefrontal_lobe_memory`，两个 Redis List，均按时间从旧到新追加，不存时间戳（相对顺序即时间先后，节省 token）。

- **追加**（热路径，不调用 LLM）：每轮先写对方的话，生成回应后再写"我"的话（沉默时写"（没有接话）"），各一次 `RPUSH`
- **滚动压缩**：`stm` 长度 > `PFC_STM_MAX_LEN`（100）时，在后台任务中把最早的 `PFC_STM_COMPRESS_COUNT`（50）条用 LLM 压缩成一个摘要块。先只读、压缩成功后才在事务里 `LTRIM stm + RPUSH compress`，LLM 失败时原文保留。压缩块超过 `PFC_COMPRESS_MAX_BLOCKS`（16）时，把最早两块合并为一块。全程加锁（`PFC_LOCK_TTL`，300 秒）
- **读取**：一次 pipeline 读两个 list，压缩块在前、原文在后，渲染成 `[说话人] 内容` 的行。压缩块显示为 `[摘要]`，当日摘要显示为 `[昨日日程]`
- **每日整理**：见 [PFC Worker](#pfc-worker工作记忆每日整理)

工作记忆进入 System2 和 System3，也用于说话人识别。

---

## Memory Gate：记忆门控

实现：`memory_gate_service.evaluate_and_persist()`。在每轮 react 返回前执行，**只评估对方说的话**（signal），不评估"我"的回应。评估与 Episodic [重激活](#episodic情景记忆)并行进行。

**评估**：一次 System One 请求问两个问题：

- `importance`（0–4 分，四舍五入为整数）：
  - 0：纯社交 / 语气，与用户本身无关
  - 1：用户在问问题，或只是此刻的瞬时状态
  - 2：近期确定的事（明天做 X），或不确定的意向（打算 / 想）
  - 3：用户的稳定特征（我是 / 我喜欢）、已完成的事件、通用经验规律
  - 4：安全 / 健康约束（过敏、禁忌）
- `memory_type`：`EPISODIC`（对话、事件、经历、约定、承诺）或 `SEMANTIC`（相对稳定的身份、偏好、资料、约束）

评估失败时按 importance = 2、SEMANTIC 处理。

**分流：**

| 条件 | 处理 |
|---|---|
| importance = 0 | 丢弃 |
| EPISODIC 且 importance < 3 | 不写长期记忆（只留在工作记忆里） |
| EPISODIC 且 importance ≥ 3 | 写入 Episodic |
| SEMANTIC 且 importance ≥ 1 | 写入 Brief |

说话人不是 `user` / `self` 时，写入内容为"{说话人}对我说：{原话}"，避免丢失说话人。metadata 包含 `speaker`、`importance`、`source: memory_gate`、`timestamp`。

**写入后更新处境：** 再发一次 System One 请求，判断这条信息是否需要更新处境。需要时用 `SITUATION_UPDATE_PROMPT` 让 LLM 输出新的完整处境 JSON，通过 `replace_situation()` 整体替换保存；新处境中 3 天内到期的每个事件让压力 +0.1。不需要更新时也会调用一次整体状态调整（增量为 0，相当于按当前压力重算精力）。

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
  "last_update": "..."
}
```

### Fact：处境（situation）

- 写入：初始化 / 更新接口、Memory Gate、待办执行
- 进入 prompt 的格式（`format_for_prompts`）：当前活动、最多 3 个时间敏感事件、最多 3 个持续状态。`attention_focus` 不进入 react 的 prompt
- 给 System1 和 System2，不给 System3

### Feeling：整体状态（overall_state）

- `mood`、`stress`、`energy` 三个连续值
- 调整（`adjust_overall_state`）：`mood` 限制在 [-1, 1]，`stress` 限制在 [0, 1]，**`energy` 每次都按 `max(0.1, 1 - stress × 0.7)` 重新计算**
- 对话后异步调整（按 System1 的情绪与强度）：

  | 情绪 | mood 变化 | stress 变化 |
  |---|---|---|
  | 喜悦、兴奋、感激 | +强度 × 0.3 | −强度 × 0.1 |
  | 焦虑、愤怒、沮丧、恐惧 | −强度 × 0.3 | +强度 × 0.2 |
  | 平静 | 0 | −强度 × 0.15 |

  变化超过 0.01 时写入，并记录 `emotion` 生命事件（通过 `trace_id` 关联本轮对话）
- 给 System1、System2 和 System3

### Fabrication：念头（thoughts）

- 生成（`generate_thought`）：给 LLM 当前时间、处境、整体状态，让它输出一个 20 字以内的内心独白，以及想找的人、紧迫度、关联事件。**处境为空时不生成**
- 来源一：对话后异步生成。System1 的情绪强度 > 0.1 或投入意愿 > 0.5 时触发，结果保存到 `thoughts`（按时间只保留最近 3 个），并记录 `thought` 事件
- 来源二：Thought Generator Worker 定时生成。这类念头**不保存到 `thoughts`**，只送进待办漏斗判断是否转成待办
- 念头不进入 react 的任何 prompt

---

## 待办与自主行为

### 待办的产生（`add_todo`）

**来自对话**（每轮 react 结束时在后台执行，不阻塞返回）：

1. System One 粗判两件事：
   - `has_commitment`：对方明确请求了某个行动，且"我"明确答应了
   - `needs_interaction`：完成任务是否需要去找某人沟通
2. `has_commitment = yes` 且置信度 > 0.5 时，交给 lite 模型（`DOUBAO_MODEL_LITE`，温度 0，最多 80 token）一次完成三件事：
   - 再确认"我"是否真的答应了要做某件事，没有就只输出"无"
   - 把对方的原话提炼成任务描述（30 字内，不照抄原话）
   - 按话里的时间推算截止日期（`YYYY-MM-DD`），提示词里带今天的日期和星期，"明早""下周三""月底"都能推算；没提时间则为空
   
   提示词约 150 token，输出是一两行纯文本（不用 JSON），实测单次约 1 秒
3. 通过后创建：优先级 0.8，相关人为说话人

需要沟通时，任务带 `interaction_type: "dialogue"`。

**来自念头**（Thought Generator Worker）：

1. 念头含行动类关键词（得、要、应该、找、联系、问、准备、安排、约等）→ 直接创建，内容为完整念头，优先级 0.6
2. 否则问 System One 是否需要转为待办：置信度 > 0.7 直接创建；0.5–0.7 再让 LLM 判断

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

**执行方式：**

- **普通任务**：LLM 用第一人称模拟一句完成结果，以"[自主完成] …"写入 Episodic；学习类任务（含"学习、了解、研究"等）还会尝试写入 Knowledge；再让 LLM 判断是否更新处境（当前活动 / 事件完成 / 新的持续状态）
- **社交任务**（`interaction_type = dialogue`）：LLM 生成一句 15 字以内的开场白，以发起方为说话人，调用**目标 Agent**（`related_person` 作为目标 agent_id）的 `react()`；把双方的话写成一段经历存入发起方的 Episodic，并在发起方处境中追加 `recent_contacts`（保留最近 10 条）

执行成功后任务**从 `pending_tasks` 移除**，内容和向量转入 `recent_done`（只用于去重，超出窗口自动清理），完整任务记录写入 `todo` 生命事件（action=completed），并记录 `task_execution` 事件。服务启动时会把旧版本遗留在 `pending_tasks` 中的已完成/已取消任务迁移出去（幂等）。

---

## 后台 Worker

四个 Worker 在服务启动时通过 `asyncio.create_task()` 启动，随服务停止。

| Worker | 执行时机 | 对象 |
|---|---|---|
| Decay Worker | 距上次执行满 `DECAY_WORKER_INTERVAL`（86400 秒）时执行；上次执行时间记在 Redis，重启不会额外触发 | Episodic 集合中出现过的所有 agent |
| PFC Worker | 每天 `PFC_DAILY_HOUR`（4）点 | 有工作记忆的所有 agent |
| Thought Generator | 启动后**先等待**一个 `THOUGHT_GENERATOR_INTERVAL`（3600 秒），之后每个间隔执行 | 有主体性上下文的所有 agent |
| Task Reminder | **启动时立即执行一次**，之后每 `TASK_REMINDER_INTERVAL`（3600 秒） | 有主体性上下文的所有 agent |

### Decay Worker：记忆衰减

对每条 Episodic 记忆：`decay_weight × DECAY_FACTOR`（0.96），结果低于 `DECAY_MIN_WEIGHT`（0.2）就删除，否则写回。按默认参数，一条一直没被再次提起的记忆大约在第 40 次衰减（约 40 天）时被删除；被重激活后权重回到 1.0，重新计算。只作用于 Episodic，Brief 和 Knowledge 不衰减。

调度：上次执行的 unix 时间戳存在 Redis 的 `worker:decay:last_run`。Worker 每次醒来都重新计算距下一次执行还有多久，到期才执行，执行后（无论成败）更新时间戳。因此服务重启不会多衰减一轮，频繁重启也不会让衰减一直轮不到；从未执行过时启动即执行。

### PFC Worker：工作记忆每日整理

对每个 agent（与滚动压缩共用锁，锁时长固定 3600 秒）：

1. 以当前 `stm` 长度为游标，读取全部压缩块和游标之前的原文
2. LLM 生成一段 150 字以内的第一人称当日摘要
3. 用这一块当日摘要（带当天所有说话人）**替换**整个压缩队列，删除游标之前的原文（整理期间新到的消息保留）
4. 把前一天出现过的新说话人记入人际关系表

LLM 失败时不做任何清理，数据留到下一次。当日摘要只留在工作记忆中，不写入长期记忆。

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

LLM token 用量通过 `contextvars` 收集：`react()` 开始时调用 `start_usage_collection()`，同一上下文中的 `chat` 调用都会把用量追加进来。

查询：`GET /admin/timeline/{agent_id}?event_types=react_input,react_output&page=1&page_size=50`。

---

## 管理接口与运行时配置

| 接口 | 行为 |
|---|---|
| `POST /admin/initialize_agent` | 一次性写入人格、Brief/Episodic/Knowledge、人际关系、处境、整体状态、念头、待办。用 Redis 锁防止并发重复初始化；agent 在 Redis 或 Qdrant 中已有任何数据时返回 409。**只有同时提供 system1 和 system2 时才会写入人格** |
| `GET /admin/agents` | persona（`persona:*:system1`）与 subject_context 的 agent 取并集 |
| `GET /admin/agent/{agent_id}` | 按初始化接口的结构返回完整信息 |
| `PUT /admin/agent/{agent_id}` | 覆盖式更新，只更新传了的字段。situation / overall_state / thoughts / pending_tasks 整体替换（pending_tasks 中 status 不是 pending 的会被忽略）；relationships 先清空再写入（传 `[]` 即清空）。agent 不存在返回 404 |
| `POST /admin/agent/{agent_id}/memories` | 追加 Brief / Episodic / Knowledge，每条都作为新记录写入。agent 不存在返回 404 |
| `DELETE /admin/agent/{agent_id}/pfc` | 删除 `stm` 和 `stm:compress` |
| `GET /admin/config` | 返回 35 个可调参数的名称、当前值、默认值、类型、描述、取值范围 |
| `PUT /admin/config` | 修改参数，见下文 |

### 运行时配置

- 白名单在 `app/config/runtime_config.py`，包括 Brief/Episodic 检索、记忆衰减、工作记忆、react 路由与知识注入、System2/3 温度和最大 token、记忆重激活阈值、待办去重、路由兜底、System1 最近对话条数等 35 项。这些参数在代码中都是调用时读取 `settings.X`，修改后下一次使用即生效
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
- Knowledge 的召回条数（5）和相似度阈值（0.5）写在代码里，不在配置中
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
