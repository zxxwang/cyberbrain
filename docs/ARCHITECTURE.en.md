# CyberBrain Architecture and Implementation

**English** | [中文](ARCHITECTURE.md)

This document describes the core flow and implementation details of CyberBrain. It follows the current code. For the design ideas, see [README](../README.en.md). For the databases it depends on, see [QUICKSTART](../QUICKSTART.en.md).

## Table of Contents

1. [Overall Structure](#overall-structure)
2. [Storage Layout](#storage-layout)
3. [Main Conversation Flow: react](#main-conversation-flow-react)
4. [System1: Instinctive Reaction and Cognitive Gating](#system1-instinctive-reaction-and-cognitive-gating)
5. [Cognitive Route Selection](#cognitive-route-selection)
6. [System2: Social Decision](#system2-social-decision)
7. [System3: Verbal Expression](#system3-verbal-expression)
8. [Speaker Identification](#speaker-identification)
9. [Lies and Fact Sources](#lies-and-fact-sources)
10. [Memory System](#memory-system)
11. [Memory Gate](#memory-gate)
12. [Subjectivity: The 3F Model](#subjectivity-the-3f-model)
13. [To-dos and Autonomous Behavior](#to-dos-and-autonomous-behavior)
14. [Background Workers](#background-workers)
15. [Life Events](#life-events)
16. [Admin API and Runtime Configuration](#admin-api-and-runtime-configuration)
17. [Model Calls at a Glance](#model-calls-at-a-glance)
18. [Known Issues and Limitations](#known-issues-and-limitations)
19. [Theoretical References](#theoretical-references)

---

## Overall Structure

```
                         ┌────────────────────────────────┐
  POST /brain/react ───▶ │   cyber_brain.react()          │
                         │  System1 → routing → System2   │
                         │          → System3             │
                         └──────┬───────────┬─────────────┘
                                │           │
          ┌─────────────────────┼───────────┼────────────────────────────┐
          ▼                     ▼           ▼                            ▼
  Memory retrieval       Working memory / persona /              Life events
  (Qdrant)               subjectivity / relationships            (MongoDB)
  Brief / Episodic /     (Redis)
  Knowledge
          ▲                     ▲
          │                     │
  ┌───────┴─────────────────────┴──────────────────────────────────────────┐
  │ Background workers: memory decay · daily working memory consolidation  │
  │                     · thought generation · to-do reminders             │
  └────────────────────────────────────────────────────────────────────────┘
```

Code layers:

| Layer | Directory | Description |
|---|---|---|
| API | `app/controller/` | `cyberbrain_controller` (`/brain`), `admin_controller` (`/admin`), `health_controller` (`/health`) |
| Service | `app/service/` | Main flow, System1, Memory Gate, subjectivity, working memory, knowledge, persona, task execution, life events |
| Data | `app/repository/` | Brief/Episodic memory (wraps mem0 and native Qdrant operations), relationships, life events |
| Client | `app/client/` | Doubao (Ark), Embedding, Rerank, Qdrant, Redis, MongoDB |
| Background | `app/worker/` | Four background tasks that start with the service |
| Config | `app/config/` | Settings, runtime-tunable parameters, prompts |

Startup order (the lifespan in `app/main.py`): initialize the two mem0 clients (Brief / Episodic) → initialize the Knowledge collection → connect to MongoDB (a failure only logs a warning) → start the four background workers.

On the network side, `app/common/httpx_patch.py` changes the default `trust_env` of httpx clients to `False` at import time. This stops the macOS system proxy from intercepting connections to localhost.

---

## Storage Layout

### Redis

| Key | Type | Content | Written by |
|---|---|---|---|
| `persona:{agent_id}:system1` | String | System1 persona fragment: emotional baseline, triggers, values | Init / update API |
| `persona:{agent_id}:system2` | String | System2 persona fragment: identity, ability boundaries, metacognition, honesty tendency | Same as above |
| `persona:{agent_id}:speaking_manner` | String | Speaking manner for System3 | Same as above |
| `subject_context:{agent_id}` | String (JSON) | Subject context: `situation`, `overall_state`, `thoughts`, `pending_tasks` (only tasks still to be executed), `recent_done` (recently completed/cancelled tasks, used for dedup), `last_update`. **Every write resets the expiry to 30 days** | Subjectivity service |
| `agent:{agent_id}:relationships` | Hash | Field is a person's name; value is `"关系类型，描述"` ("relationship type, description") | Init / update API, daily consolidation |
| `wm:{agent_id}:stm` | List | Raw working memory entries, one JSON per entry: `{type: "raw", role, speaker, content, lies?}` | react |
| `wm:{agent_id}:stm:compress` | List | Compressed blocks or daily summaries: `{type: "compress" \| "daily", summary, speakers, count?}` | Rolling compression, daily consolidation |
| `subject_context:{agent_id}:tasks_lock` | String | Lock for read-modify-write of to-dos (expires in 10 seconds) | Subjectivity service |
| `wm:{agent_id}:lock` | String | Lock shared by compression and daily consolidation | Same as above |
| `worker:decay:last_run` | String | Unix timestamp of the last Decay Worker run | Decay Worker |

In the working memory keys, the braces around `{agent_id}` are literal. They act as a Redis Cluster hash tag, so all keys of one agent land in the same slot and can be used in a transaction.

If no persona is configured, `persona_manager.get()` returns a built-in default description, so the flow does not fail.

### Qdrant

| Collection | How it is written | Vectors | Main payload |
|---|---|---|---|
| `brief_memories` | mem0 `add(infer=False)`, then BM25 is added | Dense + `bm25` sparse (IDF) | `data` (raw text), `user_id` (the agent_id), `hash`, `created_at`, `updated_at`, plus the metadata passed at write time (flattened) |
| `episodic_memories` | Same as above | Same as above | Same as above, plus `decay_weight`, `last_decay_update` |
| `knowledge_memories` | Written directly to Qdrant | Dense | `agent_id`, `title`, `content`, `knowledge_type`, `domain`, `sub_domain`, `proficiency`, `confidence`, `source`, `tags`, `related_knowledge`, `prerequisite_knowledge`, `used_count`, etc. |

- Writes to mem0 always use `infer=False`: **the raw text is stored as is, with no LLM extraction and no dedup or merging**. So writing the same content twice produces two records.
- mem0 does not write BM25 sparse vectors. After the mem0 write, `memory_repository.add()` adds them on the server side using the `qdrant/bm25` model (`multilingual` tokenizer). At startup the `bm25` slot is checked; if the IDF modifier is missing, it is enabled in place.
- For Knowledge, the embedding text is `"{title}\n{content}"`, and each entry gets a new uuid.

### MongoDB

Collection `agent_life_event` (configurable). Indexes: `(agent_id, timestamp desc)`, `event_type`, `timestamp desc`. See [Life Events](#life-events).

### SQLite

mem0's memory change history databases: `./data/brief_history.db`, `./data/episodic_history.db`.

---

## Main Conversation Flow: react

Entry point: `POST /brain/react`, with parameters `agent_id`, `signal` (what the other person said) and `speaker` (the speaker's name, optional). Implemented in `cyber_brain.react()`.

```
Record react_input event
  │
  ├─ Serial preparation:
  │    1. Speaker identification (relationship table → working memory → stranger)
  │    2. Read subject_context and format it into two texts:
  │         state_and_situation (for System1/2), state_only (for System3)
  │    3. Append signal to working memory, then read back the full working memory text
  │
  ├─ Compute the query vector once (cached, shared by all three retrievals)
  │
  ├─ In parallel:
  │    ├─ Episodic retrieval (separate task; cancelled on the fast route)
  │    ├─ Brief retrieval ──▶ System1
  │    └─ Knowledge prefetch
  │
  ├─ Decide whether to use knowledge → choose the cognitive route
  │    ├─ fast: cancel Episodic, intuitive plan
  │    ├─ memory: wait for Episodic, intuitive plan + recalled past events + knowledge
  │    └─ deliberate: wait for Episodic → System2 → decision plan
  │                   (falls back to the intuitive plan on failure)
  │
  ├─ System3 produces the spoken words (skipped if System2 decides not to speak)
  ├─ Parse [谎言] markers; write "my" reply (or "（没有接话）") into working memory
  │
  ├─ Synchronous (done before returning):
  │    ├─ Memory Gate: evaluate and persist what the other person said
  │    └─ Record react_output event (with the full trace)
  │
  ├─ Asynchronous (does not block the return): check whether this turn produced a to-do;
  │    adjust the overall state based on emotion; possibly generate a thought
  └─ Return
```

In the diagram, `[谎言]` ("lie") is the lie marker, and `（没有接话）` ("did not respond") is what is written when "I" stay silent.

Response fields:

| Field | Description |
|---|---|
| `response` | The spoken words, with `[谎言]` markers removed; empty when not speaking |
| `lies` | Sentences marked as lies |
| `silent` | Whether "I" chose not to speak |
| `route` | `fast` / `memory` / `deliberate` |
| `route_reason` | Why this route was chosen; see [Cognitive Route Selection](#cognitive-route-selection) |
| `emotion` | Full System1 output |
| `decision` | System2 decision (deliberate route only) |
| `brief` / `episodic` / `knowledge` | Retrieval results for this turn |
| `requires_knowledge` | Whether knowledge was used this turn |
| `subjectivity` | Which systems the situation was given to, and whether there is an overall state |
| `timings_ms` | Time per stage: `subjectivity`, `pfc`, `embedding`, `brief`, `system1`, `knowledge_prefetch`, `knowledge`, `episodic`, `episodic_wait`, `system2`, `system3`, `total`, `react_total` |

Query vector cache: `embedding_client.embed_query()` has a 512-entry LRU cache (thread-safe). Brief, Episodic (mem0's retrieval embedding is replaced so it goes through this cache) and Knowledge all share the same embedding call.

---

## System1: Instinctive Reaction and Cognitive Gating

Implementation: `app/service/system1_tev1.py`. It calls the **System One decision API** of Volcano Engine Ark (`doubao_client.decide()`, with the request header `ark-beta-systemone-api: true`, default model `DOUBAO_MODEL_LITE`). It does not generate text. Instead it makes a choice or gives a score for a set of questions, and returns probabilities.

**Input context** (concatenated in this order): who the other person is → subject context (overall state + situation) → System1 persona fragment → Brief memories → the recent conversation (the last `BRAIN_SYSTEM1_RECENT_TURNS` raw entries of working memory, default 4, excluding the current sentence and compressed summaries) → the external signal (the speaker is labeled again right before the signal, because a small model gives more weight to content right next to the signal).

System1 gets the recent conversation because the meaning of elliptical replies like "需要" ("yes, I need it"), "好的" ("okay") or "不用了" ("no need") lies entirely in the previous sentence. Given only two isolated characters, System1 can only guess.

**One request asks 11 questions:**

| Question | Type | Options / meaning |
|---|---|---|
| `situation` | choice | 7 situation types: 正常交流 (normal exchange), 表达关心 (showing care), 提出请求 (making a request), 开玩笑调侃 (joking / teasing), 质疑批评 (questioning / criticizing), 冒犯挑衅 (offending / provoking), 提及共同经历 (mentioning a shared experience) |
| `emotion` | choice | 16 emotions in four groups by valence × arousal. Positive-high: 兴奋 (excited), 喜悦 (joyful), 惊喜 (pleasantly surprised), 自豪 (proud); positive-low: 平静 (calm), 满足 (content), 信任 (trusting), 感激 (grateful); negative-high: 愤怒 (angry), 焦虑 (anxious), 恐惧 (afraid), 沮丧 (frustrated); negative-low: 悲伤 (sad), 无聊 (bored), 厌恶 (disgusted), 困惑 (confused) |
| `impulse` | choice | 12 first impulses. Positive: 探索 (explore), 创造 (create), 分享 (share), 帮助 (help), 庆祝 (celebrate); defensive: 回避 (avoid), 对抗 (confront), 掩饰 (cover up), 僵住 (freeze); social regulation: 亲近 (get closer), 退让 (give way); neutral: 无 (none) |
| `intensity` | score | Emotion intensity |
| `engagement` | score | Willingness to engage (whether "I" want to keep talking) |
| `conflict` | score | How much the topic conflicts with "my" values, expertise and the things "I" care about |
| `requires_knowledge` | noul | Whether domain knowledge is needed |
| `user_intent` | choice | 14 intents of the other person: 寒暄问候 (greeting / small talk), 闲聊 (chatting), 开玩笑 (joking), 分享近况 (sharing recent news), 提问 (asking a question), 求助 (asking for help), 征求意见 (asking for advice), 倾诉 (venting / confiding), 邀约 (inviting), 批评质疑 (criticizing / questioning), 挑衅冒犯 (provoking / offending), 提起往事 (bringing up the past), 正常讨论观点 (discussing views normally), 其他 (other) |
| `quick_action` | choice | 12 intuitive actions: 随口接话 (reply casually), 调侃回去 (tease back), 反问对方 (ask back), 敷衍应付 (brush off), 关心对方 (show concern), 爽快答应 (agree readily), 直接拒绝 (refuse outright), 顶回去 (push back), 简单回答 (answer briefly), 岔开话题 (change the subject), 愣住或追问 (be taken aback or ask for more), 得想一想 (need to think it over) |
| `needs_memory` | noul | Whether a specific past event must be recalled to reply |
| `needs_reasoning` | noul | Whether "I" need to stop and think carefully |

**Result mapping:**

- Score questions (continuous 0–4) are mapped linearly to 0–1: `intensity`, `engagement`, `conflict`
- `requires_knowledge`: true if the probability > 0.5
- `quick_action`: **the probability of 得想一想 is taken out separately** as `think_prob`; among the remaining options, the one with the highest probability becomes the intuitive action, and its probability is recorded as `action_confidence`
- `needs_memory`, `needs_reasoning`: the raw probabilities are kept and passed to routing, which applies thresholds

On failure, a neutral result is returned: 平静, intensity 0, `think_prob = 1.0`, `needs_reasoning = 1.0`. This guarantees the fallback goes through System2.

---

## Cognitive Route Selection

`_choose_route()` checks the conditions in order, and the first one that matches wins. It returns the route and the reason (`route_reason`, written to the response and the trace). All thresholds can be changed at runtime:

| Order | Condition | Route | route_reason |
|---|---|---|---|
| 1 | `needs_reasoning` ≥ `BRAIN_ROUTE_REASONING_THRESHOLD` (0.6) | deliberate | `needs_reasoning` |
| 2 | `think_prob` ≥ `BRAIN_ROUTE_THINK_THRESHOLD` (0.5) | deliberate | `think_prob` |
| 3 | The speaker is in the relationship table, the intuitive action is 直接拒绝 / 顶回去, and `conflict` < `BRAIN_ROUTE_HOSTILE_MAX_CONFLICT` (0.4) | deliberate | `hostile_to_known` |
| 4 | `action_confidence` < `BRAIN_ROUTE_MIN_ACTION_CONFIDENCE` (0.55) | deliberate | `low_action_confidence` |
| 5 | Knowledge is used | memory | `knowledge` |
| 6 | `needs_memory` ≥ `BRAIN_ROUTE_MEMORY_THRESHOLD` (0.6) | memory | `needs_memory` |
| 7 | None of the above | fast | `intuitive` |

- Rule 3: if System1 picks a refusal toward someone "I" know, but the topic has almost no conflict, System1 has most likely misunderstood. System2 reviews it. A refusal toward a stranger, or a refusal when conflict really is high, is still carried out as intuited
- Rule 4: when the intuitive action's probability is spread out (several options are about equal), the choice is close to random and should not turn directly into a reply
- A knowledge hit does not force System2, but the route is at least memory, so related experiences are recalled too.
- The fast route does not wait for Episodic retrieval; it cancels that task.

**Whether to use knowledge:** System1 judges that knowledge is needed, or the top prefetch score ≥ `BRAIN_KNOWLEDGE_SCORE_THRESHOLD` (0.5). When used, the top `BRAIN_KNOWLEDGE_INJECT_TOP_K` (2) entries are taken, and each body is truncated to `BRAIN_KNOWLEDGE_CONTENT_CHARS` (1200) characters. Because the prefetch's own similarity threshold is also 0.5, **in practice any prefetched knowledge gets used**.

**Intuitive plan (fast / memory routes):** generated by `_quick_plan()`. It contains the other person's intent, "my first reaction" and its explanation. On the memory route it adds up to 3 recalled past events (with the note "details not here are things I don't remember clearly") and the matched knowledge. When the other person is asking a question, it requires answering directly first and then asking back. It always includes this reminder: things not actually done (who was contacted, what was sent, what was arranged) must not be described as done; "I" can only say "I" plan to do them, or ask whether the other person wants them done. The fast / memory routes have no System2 fact check, so this constraint can only live in the plan.

---

## System2: Social Decision

Implementation: `cyber_brain._system2()`, model `DOUBAO_MODEL_PRO`, temperature `BRAIN_SYSTEM2_TEMPERATURE` (0.3), max tokens `BRAIN_SYSTEM2_MAX_TOKENS` (400). Called only on the deliberate route.

**Input:**

- System prompt (`SYSTEM2_PROMPT`): who the other person is; System1's situation type, first impulse, engagement and conflict level (all converted to words, no numbers); the persona (the system1 and system2 fragments); the decision steps (premise check → judge personal importance → combine instinct and persona → decide the action → fact boundaries → separate facts / guesses / plans → rules for lying)
- User content (`SYSTEM2_USER_TEMPLATE`): current state and situation, Brief, instinctive reaction, Episodic, knowledge (labeled "for decision reference only; technical details will not be said out loud"), working memory, and what the other person just said

**Output:** to save tokens, the model outputs numbered lines without field names:

| No. | Field | Description |
|---|---|---|
| 1 | `sanity` | Does the statement make sense: 合理 (reasonable) / 前提矛盾 (contradictory premise) / 缺少依据 (lacks grounds) |
| 2 | `thought` | What "I" think inside (used only for the decision, not passed to System3) |
| 3 | `action` | What to do |
| 4 | `core` | Core meaning; write 不说话 ("don't speak") to stay silent |
| 5 | `extra` | Additions |
| 6 | `withhold` | What is not said out loud |
| 7 | `manner` | Tact: tone, depth, initiative |
| 8 | `lies` | Lies to tell (omit the whole line if none) |
| 9 | `source` | Fact sources (omit the whole line if none) |

**Parsing (`_parse_decision`):** accepts forms like "4 核心意思：xxx", "4. xxx" and "4、xxx". If a number repeats, the first occurrence wins. If both "what to do" and "core meaning" are missing, parsing fails. `lies` is split into a list on semicolons. **Silence check**: the core meaning is empty or contains a silence word, and a silence word appears in "what to do + core meaning". The silence words are 不说话 (not speak), 沉默 (silent), 不吭声 (say nothing), 不搭理 (ignore), 不理会 (pay no attention), 不回应 (not respond).

**Failure handling:** on a call exception or a parse failure, it returns `None`, and the main flow falls back to the intuitive plan. This guarantees there is always something to say.

After successful parsing, the decision is formatted into a behavior plan for System3 (`_format_decision`): what I decided, core meaning, additions, what is on my mind but not said, tact, lies to tell, fact sources. "What I think inside" is not included in the plan, so that System3 does not say the reasoning out loud.

---

## System3: Verbal Expression

Implementation: `cyber_brain._system3()`, model `DOUBAO_MODEL_CHARACTER`, temperature `BRAIN_SYSTEM3_TEMPERATURE` (0.5), max tokens `BRAIN_SYSTEM3_MAX_TOKENS` (1000).

**It is deliberately given little information:** speaking manner (`speaking_manner`), who the other person is, overall state (without the situation), working memory, what the other person just said, the instinctive reaction (in words), and the behavior plan. **It is not given Brief, Episodic, knowledge or the situation**: any facts to be said are already in the plan, and giving more only makes it fill in content by itself.

The prompt (`SYSTEM3_PROMPT`) requires it to: follow the plan faithfully; express only outward-facing content; use natural spoken language; let the plan decide the length; add no specific people, places, numbers or experiences; never claim to have completed an action that the plan does not state as completed (such as "I've already contacted them for you"); use no bracketed stage directions; use the other person's language; and append `[谎言]` only to the end of the lies the plan specifies.

**Post-processing:**

- If Chinese characters make up ≤ 10% of the signal, a hint to "reply in the other person's language" is appended to the end of the input
- `_only_spoken()` removes bracketed stage directions of 1–30 characters (such as "（叹气）" ("(sighs)")) and quotes wrapping the whole sentence. If the whole text is stage directions, it returns an empty string, which is treated as not replying
- On call failure, it returns an empty string

---

## Speaker Identification

`_format_speaker_info()` turns the speaker into a description that is passed to System1/2/3:

1. The relationship table (`agent:{agent_id}:relationships`) has an entry → use the relationship description, e.g. "直属上级，关系一般，比较严格" ("direct manager, so-so relationship, fairly strict")
2. The speaker appears in working memory (the `speaker` of a raw entry, or the `speakers` of a compressed block / daily summary) → "有过简单接触的人" ("someone I've had brief contact with")
3. Neither, or no speaker was passed → "一个不认识的纯陌生人" ("a complete stranger I don't know")

During daily consolidation, people who appeared in the previous day's conversations but are not in the relationship table are automatically recorded as "萍水相逢的陌生人，{日期}见过一面，说过几句话" ("a stranger met by chance; met once on {date} and exchanged a few words").

---

## Lies and Fact Sources

**The full chain for lies:**

1. System2 writes the lies to tell in item 8
2. System3 appends `[谎言]` to the end of the corresponding spoken sentences
3. `parse_lies()` splits the text into sentences, takes the marked ones as `lies`, and removes the markers from the outward reply. If System3 forgets to mark them, the lies decided by System2 are used as a fallback
4. When written to working memory, `lies` is saved with the entry. When rendering the working memory text, `_mark_lies()` re-marks those sentences with `[谎言]`. So System2 in later turns, as well as the LLMs used in rolling compression and daily consolidation, all know which statements "I" made up

**Fact source validation (`_validate_fact_sources`):** parses annotations in item 9 of the form `[事件]=来源` (`[event]=source`). If a source looks like a memory ID (starts with `memory_` or `mem_`, or is all digits) but is not among the Brief / Episodic memories retrieved this turn, it is rewritten as `[事件]=推测` (`[event]=guess`) and a warning is logged.

---

## Memory System

### Brief (semantic memory)

`memory_repository.search_brief()` bypasses mem0 and queries Qdrant directly:

1. **Hybrid retrieval** (in parallel): dense vector search (similarity threshold `BRIEF_DENSE_SCORE_THRESHOLD`, 0.3) + BM25 search, each taking `BRIEF_RERANK_CANDIDATES` (10) entries
2. **RRF fusion**: `score = Σ 1 / (rank + 60)`; after dedup, the top `BRIEF_RERANK_CANDIDATES` entries are kept
3. **Rerank filter** (`BRIEF_RERANK_ENABLED`): scored by `qwen3-rerank`; entries below `BRIEF_RERANK_MIN_SCORE` (0.42) are dropped
4. **Old/new conflict ordering**: when the rerank score difference is ≤ `BRIEF_RECENCY_TIEBREAK_MARGIN` (0.05) and the vector cosine between two memories is ≥ `BRIEF_CONFLICT_SIMILARITY` (0.85), they are treated as old and new versions of the same thing, and the one with the newer `updated_at` goes first (for example, "lives in Shanghai" vs. "moved to Hangzhou")
5. Return the top `BRAIN_BRIEF_TOP_K` (10) entries

Brief is used only by System1 and System2.

### Episodic (episodic memory)

`memory_repository.search_episodic()`:

1. **Query rewriting**: generates `QUERY_REWRITE_NUM` queries (including the original). The default is 1, meaning no rewriting and no LLM call
2. **Retrieval**: each query does a dense vector search through mem0, taking `RERANK_CANDIDATE_SIZE / number of queries + 1` entries each, deduplicated by id. **Episodic does not use BM25**
3. **Skip rerank**: if the top candidate score > `RERANK_SKIP_THRESHOLD` (0.8), return the top top_k entries directly by vector score
4. **Rerank**: reranked by `qwen3-rerank`; entries below `EPISODIC_RERANK_MIN_SCORE` (0.35) are dropped
5. **Conflict ordering**: same rule as Brief, with newer `created_at` first
6. Return the top `BRAIN_EPISODIC_TOP_K` (5) entries

**Write sources:** Memory Gate (important experiences, with "{说话人}对我说：" ("{speaker} said to me:") prepended to the raw text), to-do execution results, summaries of proactive conversations, and the admin API.

**Forgetting and reactivation:**

- For forgetting, see [Decay Worker](#decay-worker-memory-decay)
- Reactivation (`memory_repository.reactivate_related()`): when Memory Gate evaluates each sentence from the other person, **whether or not the sentence itself is worth remembering**, its vector is used to search that agent's Episodic memories (dense vectors only). Memories with cosine ≥ `REACTIVATION_THRESHOLD` (0.55) (at most 5) have their weight reset to 1.0, and `last_reactivated_at` is written. Reactivation finishes before this turn writes any new memory, so it never hits the entry just written. The query vector reuses the embedding cached during react
- Retrieval ranking does not consider `decay_weight`; the weight only decides when a memory is deleted

### Knowledge

`knowledge_memory_service` reads and writes Qdrant directly through `GenericQdrantClient`:

- Types: `concept`, `howto`, `best_practice`, `troubleshooting`, `example`; sources: `experience`, `learning`, `project`, `documentation`. Invalid values are replaced with defaults
- Retrieval: dense vectors, filtered by `agent_id`, 5 entries, similarity threshold 0.5 (these two values are currently hard-coded, not in the configuration)
- Prefetched on every react turn; for whether it is injected, see [Cognitive Route Selection](#cognitive-route-selection)
- Write sources: the init API and the add-memories API. Memory Gate does not write Knowledge

### Relationships

A Redis Hash with exact name matching only, used for [Speaker Identification](#speaker-identification). Write sources: the init / update API, and new strangers recorded automatically during daily consolidation.

### Working memory (PFC)

`prefrontal_lobe_memory` uses two Redis Lists. Both are appended in time order, oldest to newest, with no timestamps (the relative order is the time order, which saves tokens).

- **Append** (hot path, no LLM call): each turn first writes what the other person said, then, after the reply is generated, writes what "I" said (or "（没有接话）" when silent). Each is one `RPUSH`
- **Rolling compression**: when the `stm` length > `PFC_STM_MAX_LEN` (100), a background task uses an LLM to compress the oldest `PFC_STM_COMPRESS_COUNT` (50) entries into one summary block. It only reads first; only after compression succeeds does it run `LTRIM stm + RPUSH compress` in a transaction. If the LLM fails, the raw entries are kept. When there are more than `PFC_COMPRESS_MAX_BLOCKS` (16) compressed blocks, the two oldest are merged into one. The whole process holds a lock (`PFC_LOCK_TTL`, 300 seconds)
- **Read**: one pipeline reads both lists, compressed blocks first and raw entries after, rendered as `[speaker] content` lines. Compressed blocks are shown as `[摘要]` ("summary") and daily summaries as `[昨日日程]` ("yesterday's schedule")
- **Daily consolidation**: see [PFC Worker](#pfc-worker-daily-consolidation-of-working-memory)

Working memory goes into System2 and System3, and is also used for speaker identification.

---

## Memory Gate

Implementation: `memory_gate_service.evaluate_and_persist()`. It runs before each react turn returns and **evaluates only what the other person said** (the signal), not "my" reply. Evaluation runs in parallel with Episodic [reactivation](#episodic-episodic-memory).

**Evaluation:** one System One request asks two questions:

- `importance` (0–4, rounded to an integer):
  - 0: purely social / tone, nothing about the user
  - 1: the user is asking a question, or it is just a momentary state
  - 2: something settled for the near term (doing X tomorrow), or an uncertain intention (planning / wanting to)
  - 3: a stable trait of the user (I am / I like), a completed event, a general rule of experience
  - 4: a safety / health constraint (allergies, things to avoid)
- `memory_type`: `EPISODIC` (conversations, events, experiences, agreements, promises) or `SEMANTIC` (relatively stable identity, preferences, profile information, constraints)

If evaluation fails, it is treated as importance = 2, SEMANTIC.

**Routing:**

| Condition | Handling |
|---|---|
| importance = 0 | Drop |
| EPISODIC and importance < 3 | Not written to long-term memory (stays only in working memory) |
| EPISODIC and importance ≥ 3 | Write to Episodic |
| SEMANTIC and importance ≥ 1 | Write to Brief |

When the speaker is not `user` / `self`, the stored content is "{说话人}对我说：{原话}" ("{speaker} said to me: {original words}"), so the speaker is not lost. The metadata includes `speaker`, `importance`, `source: memory_gate` and `timestamp`.

**Updating the situation after writing:** another System One request decides whether this information requires updating the situation. If so, `SITUATION_UPDATE_PROMPT` has the LLM output a new, complete situation JSON, which replaces the old one as a whole via `replace_situation()`. Each event in the new situation that is due within 3 days adds +0.1 to stress. If no update is needed, an overall state adjustment is still called once (with zero delta, which amounts to recomputing energy from the current stress).

---

## Subjectivity: The 3F Model

Implementation: `subjectivity_service`. All data is stored in one `subject_context:{agent_id}` JSON.

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

The example values in this JSON mean: 准备游戏上线 ("preparing the game launch"), Steam首周销量统计 ("Steam first-week sales numbers"), 最近睡眠不好 ("sleeping badly lately"), 游戏稳定性 ("game stability").

### Fact: situation

- Written by: the init / update API, Memory Gate, to-do execution
- Format in the prompt (`format_for_prompts`): current activity, up to 3 time-sensitive events, up to 3 ongoing states. `attention_focus` does not go into the react prompts
- Given to System1 and System2, not System3

### Feeling: overall state

- Three continuous values: `mood`, `stress`, `energy`
- Adjustment (`adjust_overall_state`): `mood` is clamped to [-1, 1], `stress` to [0, 1], and **`energy` is recomputed every time as `max(0.1, 1 - stress × 0.7)`**
- Asynchronous adjustment after a conversation (based on System1's emotion and intensity):

  | Emotion | mood change | stress change |
  |---|---|---|
  | 喜悦, 兴奋, 感激 | +intensity × 0.3 | −intensity × 0.1 |
  | 焦虑, 愤怒, 沮丧, 恐惧 | −intensity × 0.3 | +intensity × 0.2 |
  | 平静 | 0 | −intensity × 0.15 |

  The change is written when it exceeds 0.01, and an `emotion` life event is recorded (linked to the conversation turn via `trace_id`)
- Given to System1, System2 and System3

### Fabrication: thoughts

- Generation (`generate_thought`): the LLM gets the current time, the situation and the overall state, and outputs an inner monologue of at most 20 characters, plus the person "I" want to reach, the urgency and the related event. **Nothing is generated when the situation is empty**
- Source 1: generated asynchronously after a conversation. Triggered when System1's emotion intensity > 0.1 or engagement > 0.5. The result is saved to `thoughts` (only the 3 most recent are kept, by time), and a `thought` event is recorded
- Source 2: generated periodically by the Thought Generator Worker. These thoughts are **not saved to `thoughts`**; they only go into the to-do funnel, which decides whether to turn them into to-dos
- Thoughts do not go into any react prompt

---

## To-dos and Autonomous Behavior

### Creating to-dos (`add_todo`)

**From conversations** (run in the background at the end of each react turn, without blocking the return):

1. System One makes a rough judgment on two things:
   - `has_commitment`: the other person explicitly asked for some action, and "I" explicitly agreed
   - `needs_interaction`: whether completing the task requires talking to someone
2. When `has_commitment = yes` with confidence > 0.5, the lite model (`DOUBAO_MODEL_LITE`, temperature 0, at most 80 tokens) does three things in one call:
   - Re-check whether "I" really agreed to do something; if not, output only 无 ("none")
   - Turn the other person's words into a task description (at most 30 characters, not copying the original words)
   - Work out the deadline (`YYYY-MM-DD`) from any time mentioned. The prompt includes today's date and weekday, so expressions like 明早 ("tomorrow morning"), 下周三 ("next Wednesday") and 月底 ("end of the month") can all be resolved. If no time is mentioned, it is left empty

   The prompt is about 150 tokens, and the output is one or two lines of plain text (not JSON). Measured at about 1 second per call
3. If it passes, the task is created with priority 0.8, and the related person is the speaker

When talking to someone is needed, the task carries `interaction_type: "dialogue"`.

**From thoughts** (Thought Generator Worker):

1. If the thought contains action keywords (得 (have to), 要 (need to), 应该 (should), 找 (look for), 联系 (contact), 问 (ask), 准备 (prepare), 安排 (arrange), 约 (make an appointment), etc.) → create directly, with the full thought as content and priority 0.6
2. Otherwise, ask System One whether it should become a to-do: confidence > 0.7 creates it directly; 0.5–0.7 lets the LLM decide

**Storage and dedup:**

- `pending_tasks` stores only tasks still to be executed, sorted by priority, **with no limit on count**
- Dedup runs before creation. It compares against all pending tasks, plus tasks completed or cancelled within `TODO_DEDUP_WINDOW_HOURS` (24 hours) (`recent_done`):
  1. Identical content → duplicate
  2. Otherwise, a rough embedding filter: tasks with cosine ≥ `TODO_DEDUP_SIMILARITY` (0.6) become candidates, at most 3
  3. System One decides whether the new task is the same thing as any candidate (same action, same person, same purpose); if so, it is skipped. When System One is unavailable, only cosine ≥ 0.9 counts as a duplicate

  A similarity threshold alone cannot separate two cases: the same task phrased differently often has a cosine around 0.65, while "remind me to take medicine tomorrow morning" and "remind me about the meeting tomorrow morning" reach 0.78. So the embedding narrows the range, and the model makes the call
- A task's vector is stored with the task (the `vector` field, not returned by the external API). Older tasks without a vector get one computed on their first comparison
- The read-modify-write of `pending_tasks` is serialized by a Redis lock, so to-dos from conversations, to-dos from thoughts and task completions happening at the same time do not overwrite each other
- task_id is a uuid
- A `todo` event is recorded on creation. To-dos can only be completed by the Task Reminder; there is currently no entry point for cancelling (`cancel_task()` is implemented)

### Executing to-dos (Task Reminder Worker + `task_executor`)

**When to execute (`_should_execute`):**

| Case | Rule |
|---|---|
| Past the deadline / on the deadline day | Execute |
| 1–2 days before the deadline | About 50% execute early, based on a hash of the task ID (at least 1 hour after creation) |
| 3–7 days before the deadline and priority > 0.7 | About 30% execute early, based on a hash of the task ID (at least 2 hours after creation) |
| No deadline | Priority > 0.7: 2 hours after creation; > 0.5: 12 hours; otherwise: 48 hours |

A hash of the task ID is used instead of a random number, so every check of the same task reaches the same result.

**How tasks are executed:**

- **Ordinary tasks**: the LLM simulates a one-sentence result in the first person, which is written to Episodic as "[自主完成] …" ("[done on my own] …"). Learning tasks (containing words like 学习 (study), 了解 (learn about), 研究 (research)) are also written to Knowledge if possible. Then the LLM decides whether to update the situation (current activity / event completed / new ongoing state)
- **Social tasks** (`interaction_type = dialogue`): the LLM generates an opening line of at most 15 characters and calls `react()` of the **target agent** (`related_person` is used as the target agent_id), with the initiator as the speaker. Both sides' words are written as one experience into the initiator's Episodic, and `recent_contacts` is appended to the initiator's situation (keeping the 10 most recent)

After successful execution, the task is **removed from `pending_tasks`**. Its content and vector move to `recent_done` (used only for dedup and cleaned up automatically once outside the window). The full task record is written to a `todo` life event (action=completed), and a `task_execution` event is recorded. At service startup, completed/cancelled tasks left in `pending_tasks` by older versions are migrated out (idempotent).

---

## Background Workers

The four workers are started with `asyncio.create_task()` when the service starts, and stop with the service.

| Worker | When it runs | Targets |
|---|---|---|
| Decay Worker | When `DECAY_WORKER_INTERVAL` (86400 seconds) has passed since the last run. The last run time is stored in Redis, so a restart does not trigger an extra run | All agents that appear in the Episodic collection |
| PFC Worker | Every day at `PFC_DAILY_HOUR` (4) o'clock | All agents with working memory |
| Thought Generator | **Waits** one `THOUGHT_GENERATOR_INTERVAL` (3600 seconds) after startup, then runs every interval | All agents with a subject context |
| Task Reminder | **Runs once immediately at startup**, then every `TASK_REMINDER_INTERVAL` (3600 seconds) | All agents with a subject context |

### Decay Worker: memory decay

For each Episodic memory: `decay_weight × DECAY_FACTOR` (0.96). If the result is below `DECAY_MIN_WEIGHT` (0.2), the memory is deleted; otherwise the weight is written back. With the default parameters, a memory that is never mentioned again is deleted at about the 40th decay (about 40 days). After reactivation the weight returns to 1.0 and the count starts over. Only Episodic decays; Brief and Knowledge do not.

Scheduling: the unix timestamp of the last run is stored in Redis under `worker:decay:last_run`. Each time the worker wakes up, it recomputes how long until the next run, and runs only when due. After a run (successful or not), it updates the timestamp. So a service restart does not cause an extra decay round, and frequent restarts do not keep pushing decay back. If it has never run, it runs at startup.

### PFC Worker: daily consolidation of working memory

For each agent (sharing the lock with rolling compression; the lock duration is fixed at 3600 seconds):

1. Using the current `stm` length as a cursor, read all compressed blocks and the raw entries before the cursor
2. The LLM generates a first-person daily summary of at most 150 characters
3. This one daily summary block (with all of the day's speakers) **replaces** the whole compressed queue, and the raw entries before the cursor are deleted (messages that arrive during consolidation are kept)
4. New speakers from the previous day are added to the relationship table

If the LLM fails, nothing is cleaned up, and the data waits for the next run. The daily summary stays only in working memory and is not written to long-term memory.

### Thought Generator and Task Reminder

See [Subjectivity](#fabrication-thoughts) and [To-dos and Autonomous Behavior](#to-dos-and-autonomous-behavior).

---

## Life Events

`agent_life_event_service` writes what the agent experiences to MongoDB. If MongoDB is not connected, it logs one warning and skips, without affecting the main flow.

| event_type | When it is recorded |
|---|---|
| `react_input` | A sentence is received |
| `react_output` | A sentence is said (or silence). The top level carries the latency `rt` and the full `trace`: speaker description, lies, routing, all System1 fields, the System2 decision, ids/scores/texts from the three retrievals, the overall state before the conversation, working memory length, per-stage timings, and token usage of every LLM call in this turn |
| `thought` | A thought is generated after a conversation |
| `todo` | A to-do is created or completed (the cancel method is implemented, but nothing calls it yet) |
| `situation` | The situation is set or updated |
| `emotion` | The overall state changes (with before/after values and the delta) |
| `task_execution` | A to-do execution succeeds or fails |

LLM token usage is collected with `contextvars`: `react()` calls `start_usage_collection()` at the start, and every `chat` call in the same context appends its usage.

Query: `GET /admin/timeline/{agent_id}?event_types=react_input,react_output&page=1&page_size=50`.

---

## Admin API and Runtime Configuration

| Endpoint | Behavior |
|---|---|
| `POST /admin/initialize_agent` | Writes persona, Brief/Episodic/Knowledge, relationships, situation, overall state, thoughts and to-dos in one go. A Redis lock prevents concurrent duplicate initialization. Returns 409 if the agent already has any data in Redis or Qdrant. **The persona is written only when both system1 and system2 are provided** |
| `GET /admin/agents` | The union of agents from persona (`persona:*:system1`) and subject_context |
| `GET /admin/agent/{agent_id}` | Returns full information in the same structure as the init API |
| `PUT /admin/agent/{agent_id}` | Overwrite update; only the fields passed are updated. situation / overall_state / thoughts / pending_tasks are replaced as a whole (entries in pending_tasks whose status is not pending are ignored); relationships are cleared and then written (passing `[]` clears them). Returns 404 if the agent does not exist |
| `POST /admin/agent/{agent_id}/memories` | Appends Brief / Episodic / Knowledge; each entry is written as a new record. Returns 404 if the agent does not exist |
| `DELETE /admin/agent/{agent_id}/pfc` | Deletes `stm` and `stm:compress` |
| `GET /admin/config` | Returns the name, current value, default value, type, description and value range of the 35 tunable parameters |
| `PUT /admin/config` | Changes parameters; see below |

### Runtime configuration

- The whitelist is in `app/config/runtime_config.py`. It has 35 items, covering Brief/Episodic retrieval, memory decay, working memory, react routing and knowledge injection, System2/3 temperature and max tokens, the memory reactivation threshold, to-do dedup, routing fallbacks, the number of recent turns for System1, and more. The code reads all of these as `settings.X` at call time, so a change takes effect the next time the value is used
- Update flow: everything is validated as a whole (whitelist, strict types, value ranges, plus `PFC_STM_COMPRESS_COUNT ≤ PFC_STM_MAX_LEN`). If any item is invalid, nothing is changed. After validation passes, the values are **first written back to `.env`** (temp file + atomic replace, keeping other lines and comments), **then changed in memory** in `settings`
- Configuration precedence: environment variables > `.env` > defaults in `settings.py`. `settings` uses `extra="forbid"`, so the service will not start if `.env` contains an undefined setting

---

## Model Calls at a Glance

| Purpose | Model / API |
|---|---|
| System1, Memory Gate evaluation, whether to update the situation, the to-do funnel | System One decision API (`DOUBAO_MODEL_LITE`) |
| System2 | `DOUBAO_MODEL_PRO` |
| System3 | `DOUBAO_MODEL_CHARACTER` |
| Working memory compression and daily summary, situation updates, thought generation, LLM to-do judgment, simulated task execution, opening lines, query rewriting | `DOUBAO_MODEL_LITE` (the default model of `chat` / `generate`) |
| Embedding | `EMBEDDING_MODEL` (OpenAI-compatible API), dimension `EMBEDDING_DIMENSION` |
| Rerank | `RERANK_MODEL` (`qwen3-rerank`, Alibaba Cloud Model Studio (DashScope)) |

The LLM inside mem0 is replaced with `DoubaoLLM` with deep thinking turned off. But since writes use `infer=False`, the current flow never calls it.

---

## Known Issues and Limitations

**Design limitations:**

- **Episodic conflict ordering does not actually take effect**: the ordering relies on memory vectors, but mem0's search results do not include vectors, so the similarity is always 0 and results are ordered only by rerank score
- **Thoughts do not take part in conversations**: thoughts do not go into any prompt; thoughts from the Thought Generator are used only to create to-dos and are not saved
- **Memory Gate only looks at the other person's words**: what "I" said (including promises and lies) stays only in working memory and never enters long-term memory
- **Inconsistent overall state ranges**: the init / update API requires `mood` in [0, 1], while adjustments during conversation allow [-1, 1]
- Knowledge's retrieval count (5) and similarity threshold (0.5) are hard-coded, not in the configuration
- The `/admin` API has no authentication

---

## Theoretical References

- Kahneman (2011). *Thinking, Fast and Slow*: dual-system theory
- Ebbinghaus (1885). *Über das Gedächtnis*: the forgetting curve
- Baddeley & Hitch (1974). Working Memory Model: working memory
- Scoville & Milner (1957). The H.M. case: the hippocampus and episodic memory
- Plutchik (1980). Emotion Wheel: discrete emotions
- Russell (1980). Circumplex Model of Affect: grouping emotions by valence × arousal
- Cannon (1915). Fight-or-Flight Response: stress response and defensive impulses
- Raichle et al. (2001). Default Mode Network: spontaneous thoughts
