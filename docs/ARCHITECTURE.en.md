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
14. [Proactive Outreach](#proactive-outreach)
15. [Background Workers](#background-workers)
16. [Life Events](#life-events)
17. [Admin API and Runtime Configuration](#admin-api-and-runtime-configuration)
18. [Model Calls at a Glance](#model-calls-at-a-glance)
19. [Known Issues and Limitations](#known-issues-and-limitations)
20. [Theoretical References](#theoretical-references)

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
| `agent:{agent_id}:relationships` | Hash | Field is a person's name; value is JSON `{kind, agent_id, relationship, description}`; the legacy format `"关系类型，描述"` ("relationship type, description") is read as kind=human | Init / update API, daily consolidation |
| `agent:{agent_id}:capabilities` | Set | The agent's capabilities, currently only `web_search`; missing means no capabilities | Init / update API |
| `outbox:{agent_id}:{name}` | List | What the agent proactively wants to say to this person (kind=human), at most 50 kept; `GET /brain/fetch_talk` takes them out and deletes them | Proactive outreach |
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
  │    ├─ Episodic retrieval (separate task, parallel with System1; awaited before routing)
  │    ├─ Brief retrieval
  │    └─ Knowledge prefetch (cosine filter + rerank)
  ├─ System1 (sees Brief and "the material I have on this topic")
  │
  ├─ Decide whether to use knowledge and whether a past event was evoked
  │    (highest Episodic score) → choose the cognitive route
  │    ├─ fast: no Episodic, intuitive plan
  │    ├─ memory: intuitive plan + recalled past events + knowledge
  │    └─ deliberate: System2 → decision plan
  │                   (falls back to the intuitive plan on failure)
  │
  ├─ System3 produces the spoken words (skipped if System2 decides not to speak)
  ├─ Parse [谎言] markers; write "my" reply (or "（没有接话）") into working memory
  │
  ├─ Synchronous (done before returning): record react_output event (with the full trace)
  │
  ├─ Asynchronous (does not block the return): Memory Gate evaluates and persists what the
  │    other person said; check whether this turn produced a to-do;
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

**Input context** (concatenated in this order): who the other person is → subject context (overall state + situation) → System1 persona fragment → Brief memories → the material I have on this topic (titles of knowledge that survived rerank, or "no material directly on this topic") → the recent conversation (the last `BRAIN_SYSTEM1_RECENT_TURNS` raw entries of working memory, default 4, excluding the current sentence and compressed summaries) → my state right now (`describe_state()` turns the overall state into one sentence, e.g. "not in a great mood, under a lot of stress, a bit tired") → the external signal (the speaker is labeled again right before the signal, because a small model gives more weight to content right next to the signal).

**The overall state is given in words, right before the signal:** it used to appear only as three numbers at the top of the context, and removing it barely changed System1's judgments (Lin Shiyu had 7/6/6 negative emotions out of 12 under "very bad / baseline / very good"). In words next to the signal, plus a general rule in the emotion instructions (when in a bad mood, stressed or tired, the same sentence is more likely to irritate, sting or drain; in a good state one is more tolerant), the counts became 12/3/1 and engagement 0.47/0.74/0.83. There is no runaway loop: the return to baseline is driven by time (see [overall state](#feeling-overall-state)), and in-conversation deltas have diminishing returns.

**Concern within a relationship is background:** the emotion instructions state that the emotion is a reaction to this sentence; long-standing worry about the other person from the relationship description is aroused only when this sentence touches on it. The relationship description used to say "she's about to break", so even "good night" from her was judged anxious (0.99); measured, sentences unrelated to the worry went from 4/15 anxious to 0/15, while related ones stayed at 7/9. The relationship description should hold only the stable relationship itself: changing circumstances such as "she has been getting worse lately" belong in the situation (which updates with conversations), and past events such as "borrowed 100k last month" belong in Episodic.

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
| `quick_action` | choice | 13 intuitive actions: 随口接话 (reply casually), 调侃回去 (tease back), 反问对方 (ask back), 敷衍应付 (brush off), 关心对方 (show concern), 爽快答应 (agree readily), 直接拒绝 (refuse outright), 顶回去 (push back), 简单回答 (answer briefly), 岔开话题 (change the subject), 愣住或追问 (be taken aback or ask for more), 凭印象说 (answer from vague impression), 得想一想 (need to think it over). When asked about something it doesn't know, it reacts according to its persona |
| `needs_memory` | noul | Whether a specific past event must be recalled to reply |
| `needs_reasoning` | noul | Whether "I" need to stop and think carefully |

**Result mapping:**

- Score questions (continuous 0–4) are mapped linearly to 0–1: `intensity`, `engagement`, `conflict`
- `requires_knowledge`: true if the probability > 0.5
- `emotion`: the most probable option becomes the main emotion; the top 3 probabilities are also kept as `emotion_probs` in the API response and the trace, for observation only, not used by routing or state updates. They express how certain the choice is, not a mix of co-existing emotions; measured, the top one is usually ≥ 0.9 (e.g. "I passed the grad-school exam!": 喜悦 0.81 / 惊喜 0.14)
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
| 4.5 | The other person is talking about the past (intent 提起往事, situation 提及共同经历, or `needs_memory` at threshold — any one), and the top Episodic score < `BRAIN_EPISODIC_CUE_MIN_SCORE` | deliberate | `recall_uncertain` |
| 5 | Knowledge is used | memory | `knowledge` |
| 6 | `needs_memory` ≥ `BRAIN_ROUTE_MEMORY_THRESHOLD` (0.6) | memory | `needs_memory` |
| 7 | The highest rerank score among the Episodic memories retrieved in parallel ≥ `BRAIN_EPISODIC_CUE_MIN_SCORE` (0.75) | memory | `episodic_cue` |
| 8 | None of the above | fast | `intuitive` |

- Rule 3: if System1 picks a refusal toward someone "I" know, but the topic has almost no conflict, System1 has most likely misunderstood. System2 reviews it. A refusal toward a stranger, or a refusal when conflict really is high, is still carried out as intuited
- Rule 4: when the intuitive action's probability is spread out (several options are about equal), the choice is close to random and should not turn directly into a reply
- Rule 4.5 (cannot recall): asked about a past event they can't recall, people pause and think. Writing "I can't recall" into the intuitive plan as a fact proved ineffective: System3 (the character model) still invented answers ("the pothos is doing fine", "the pen was a Lamy"). System2 writes how to respond (say so plainly, stay vague, ask back — per persona) into its decision, and System3 only voices it. System1's intent and situation definitions now explicitly include *asking about* the past (they used to say only "bringing up", so "what brand was that pen again?" was classed as an ordinary question)
- A knowledge hit does not force System2, but the route is at least memory, so related experiences are recalled too.
- Either rule 6 or rule 7 is enough to bring memories in. `needs_memory` is a single yes/no judgment by a small model and is quite unstable: for the same sentence 我之前跟你说要去哪儿开民宿来着？ ("where did I say I'd open a guesthouse?") with identical input, two runs gave 0.47 and 0.05, and rewording the question only changes the kind of instability. Past events are usually brought back by what the other person says: Episodic retrieval already runs in parallel with System1, and if it finds a memory highly relevant to the sentence, it should be recalled, just as knowledge is used whenever something survives rerank. Measured (rerank scores): sentences that need recalling the past scored 0.81–0.98; those that don't scored 0–0.66, the highest being the reading experience evoked by 你觉得《百年孤独》好看吗 ("do you like One Hundred Years of Solitude?", 0.66), which is the kind of personal experience a topic should bring back anyway
- Episodic retrieval is awaited before routing, since rule 7 depends on it. It runs in parallel with System1 and was measured to always finish first (150–230 ms vs about 370–720 ms for System1), so it usually costs nothing extra; the wait is recorded as `episodic_wait`. On the fast route the retrieved memories are not put into the plan. The trace's `recall.episodic_cue` records the highest score of the turn

**Knowledge is injected with its confidence:** each entry's title carries a confidence label (high ≥ 0.75 / medium ≥ 0.5 / low, mostly commercial or marketing claims), in words rather than numbers. Both the System2 prompt and the intuitive plan state that content without high confidence may only be presented as "some sources say", never as a conclusion.

**Whether to use knowledge:** this depends only on whether anything is left after rerank (see [Knowledge](#knowledge)). The remaining entries are sorted by rerank score, the top `BRAIN_KNOWLEDGE_INJECT_TOP_K` (2) are taken, and each body is truncated to `BRAIN_KNOWLEDGE_CONTENT_CHARS` (1200) characters.

When System1 thinks knowledge is needed but nothing survives rerank, **and the other person is asking a question or for advice**, it is marked as a "knowledge gap" (when they are giving an instruction such as "go look up X", the request is what needs a response, so it is not a gap). The intuitive plan only states the fact, "the material I have says nothing directly about this", and does not prescribe a response.

**How to behave when not knowing is decided by the persona**: saying "I don't know", answering from vague impressions, changing the subject, or turning the question back are all expressions of metacognition and differ from person to person. System1 runs after the knowledge prefetch, and its input includes "the material I have on this topic" (titles of the matched knowledge, or "no material directly on this topic"). Based on the persona, it picks an intuitive action (such as 简单回答 "simple answer", 凭印象说 "answer from impression", 岔开话题 "change the subject", 反问对方 "ask back"), and System3 carries it out. Metacognition belongs in the persona (`persona_system1` decides the first reaction, `persona_system2` decides judgment when deliberating), not in the speaking manner.

**What each of the three persona parts holds:** bottom lines that every route must keep (identity, relationship boundaries, honesty) go in `persona_system1`: System1 and System3 see it every turn, including on the fast / memory routes. `persona_system2` is used only on the deliberate route and should hold how to judge when thinking things through (how to weigh things, how to advise, when to warn about risk). A bottom line written only in system2 is invisible to the fast route: measured, boundaries such as privacy, authorization and risk warnings were also in system1, and both routes behaved consistently. Note, however, that putting "say you don't know when you don't" into system1 does not stop the fast route from inventing information it cannot have: asked about the weather, the bitcoin price, the number of emails, parking spaces or a past blood-pressure reading, Jarvis invented specific numbers in 17 of 18 replies (the weather was answered honestly only when the persona named "real-time weather" as an example, which is overfitting to the example). Such "current facts I cannot know" need a structural fix and are not yet solved.

**Intuitive plan (fast / memory routes):** generated by `_quick_plan()`. It contains the other person's intent, "my first reaction" and its explanation. On the memory route it adds up to 3 recalled past events and the matched knowledge; the past events are rendered by `_format_recall()` (shared with System2) together with the boundary of memory: "this is all I remember; anything not written here I don't know". When the other person is talking about the past, it also states that the retrieved memories are only candidates and may not include the event they mean, and if nothing was recalled it says "I can't recall anything specific about this". System2 always gets the memories framed as candidates (System1 often fails to recognize talk about the past, while this note holds for any sentence). It only states facts; how to respond when not remembering is left to the persona. When the other person is asking a question, it requires answering directly first and then asking back. It always includes this reminder: things not actually done (who was contacted, what was sent, what was arranged) must not be described as done; "I" can only say "I" plan to do them, or ask whether the other person wants them done.

**This-turn timing fact (`THIS_TURN_FACT`):** within a react turn the reply is generated and returned first; what the other person asked for (creating a to-do, a reminder, contacting someone, looking something up) is judged and executed in the background after the reply, and may not happen at all (the funnel decides it is not needed, the lock is busy, a model call fails). So the intuitive plan, the System2 input and the end of the System2 decision plan all state one fact: "this turn I have done nothing except speak; what they just asked me to do has not been done yet and will be done later". It states only the timing and prescribes no wording, so replies naturally become commitments ("Certainly, I'll take care of it / I'll remind you then") rather than "Reminder set". Previously there was only the rule "don't claim something done unless the plan says it is done", and the model treated "set a reminder" or "note it down" as done the moment it agreed (Jarvis measured 10/16 in the completed form, 2/33 after the change). System2's fact boundary also gained a line: to-dos and events in the situation only show that the thing exists, not that it has been done. The fast / memory routes have no System2 fact check, so this constraint can only live in the plan.

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

**It is deliberately given little information:** the instinctive self-knowledge (`persona_system1`, labeled "for holding identity and tone, not content to be said"), speaking manner (`speaking_manner`), who the other person is, overall state (without the situation), working memory, what the other person just said, the instinctive reaction (in words), and the behavior plan. **It is not given Brief, Episodic, knowledge or the situation**: any facts to be said are already in the plan, and giving more only makes it fill in content by itself.

Why self-knowledge is included: knowing who you are is as instinctive as knowing who you're talking to (speaker info), so it should be present every turn, not only when System2 runs. Otherwise, asked "who are you / what can you do" on the fast route, System3 doesn't know who it is and can only fall back on phrasings from the speaking manner.

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

**Write sources:** [episode consolidation](#episode-consolidation-working-memory--episodic) (extracted from the raw working-memory text before rolling compression and during daily consolidation, `source: consolidation`), to-do execution results, summaries of proactive conversations, and the admin API. The Memory Gate no longer writes sentence by sentence.

**Forgetting and reactivation:**

- For forgetting, see [Decay Worker](#decay-worker-memory-decay)
- Reactivation (`memory_repository.reactivate_related()`): when Memory Gate evaluates each sentence from the other person, **whether or not the sentence itself is worth remembering**, its vector is used to search that agent's Episodic memories (dense vectors only). Memories with cosine ≥ `REACTIVATION_THRESHOLD` (0.55) (at most 5) have their weight reset to 1.0, and `last_reactivated_at` is written. Reactivation finishes before this turn writes any new memory, so it never hits the entry just written. The query vector reuses the embedding cached during react
- Retrieval ranking does not consider `decay_weight`; the weight only decides when a memory is deleted

### Knowledge

`knowledge_memory_service` reads and writes Qdrant directly through `GenericQdrantClient`:

- Types: `concept`, `howto`, `best_practice`, `troubleshooting`, `example`; sources: `experience`, `learning`, `project`, `documentation`. Invalid values are replaced with defaults
- Retrieval has two steps:
  1. Dense vector recall, filtered by `agent_id`, 5 entries, cosine ≥ `BRAIN_KNOWLEDGE_SCORE_THRESHOLD` (0.5)
  2. `qwen3-rerank` judges whether each entry answers the sentence; entries below `BRAIN_KNOWLEDGE_RERANK_MIN_SCORE` (0.5) are dropped

  Cosine alone cannot tell "on the same topic" from "answers the question": with a single-topic knowledge base, any question on that topic recalls entries above 0.55. Measured on 17 anti-aging entries: for questions the base cannot answer (NAD+, metformin, fasting, etc.), the highest rerank score of recalled entries was 0.484; for questions it can answer, the lowest score of the correct entry was 0.529. If the rerank call fails, only entries with cosine ≥ 0.6 are kept
- The recall count (5) is still hard-coded, not in the configuration
- Prefetched on every react turn; for whether it is injected, see [Cognitive Route Selection](#cognitive-route-selection)
- Write sources: the init API and the add-memories API. Memory Gate does not write Knowledge

### Relationships

A Redis Hash with exact name matching only, used for [Speaker Identification](#speaker-identification). Write sources: the init / update API, and new strangers recorded automatically during daily consolidation (kind=human).

Each relationship records the other party's **type**: `kind = human` (a real person) or `agent` (another agent, in which case its `agent_id` is required). The type only decides [how "I" reach out to them](#proactive-outreach) and does not go into prompts; the relationship description in prompts is still "relationship type, description". Validation happens both in the API model and in the storage layer: kind=agent requires agent_id, kind=human must not have one.

### Working memory (PFC)

`prefrontal_lobe_memory` uses two Redis Lists. Both are appended in time order, oldest to newest. Each raw entry carries an `id` and `ts` (unix seconds), which are not rendered into the context, so they cost no tokens: `id` lets the Memory Gate and System1 tag the entry in the background with importance and emotion, and `ts` lets episode consolidation split events by time gaps and date them.

- **Append** (hot path, no LLM call): each turn first writes what the other person said, then, after the reply is generated, writes what "I" said (or "（没有接话）" when silent). Each is one `RPUSH`
- **Rolling compression**: when the `stm` length > `PFC_STM_MAX_LEN` (100), a background task handles the oldest `PFC_STM_COMPRESS_COUNT` (50) entries: it first [extracts episodes](#episode-consolidation-working-memory--episodic) into Episodic (the details are gone once summarized), then uses an LLM to compress them into one summary block. It only reads first; only after both steps succeed does it run `LTRIM stm + RPUSH compress` in a transaction. If either step fails, the raw entries are kept. When there are more than `PFC_COMPRESS_MAX_BLOCKS` (16) compressed blocks, the two oldest are merged into one. The whole process holds a lock (`PFC_LOCK_TTL`, 300 seconds)
- **Read**: one pipeline reads both lists, compressed blocks first and raw entries after, rendered as `[speaker] content` lines. Compressed blocks are shown as `[摘要]` ("summary") and daily summaries as `[昨日日程]` ("yesterday's schedule")
- **Daily consolidation**: see [PFC Worker](#pfc-worker-daily-consolidation-of-working-memory)

Working memory goes into System2 and System3, and is also used for speaker identification.

---

## Memory Gate

Implementation: `memory_gate_service.evaluate_and_persist()`. It runs in the background after each react turn has produced its reply, without blocking the return, and **evaluates only what the other person said** (the signal). It **writes neither Episodic nor Brief**: episodes are extracted from the raw working-memory text by [episode consolidation](#episode-consolidation-working-memory--episodic) (which sees the context and what "I" said), and beliefs are written only by [reflection](#reflection-reflection-worker). It does only the four things that must happen in real time:

1. **Reactivation**: runs in parallel with Episodic reactivation; when the other person brings up something again, similar episodes get their weight restored and `updated_at` refreshed (decay and reflection depend on it)
2. **Tagging**: writes `importance` and `memory_type` onto this entry in working memory (by entry id, an atomic Lua update) as the "first impression" for consolidation; System1's emotion and intensity are tagged the same way
3. **Personal information**: see profile below
4. **Situation and state**: when `importance ≥ 3`, update the situation and overall state ("the proposal is due tomorrow" must raise stress right away)

**Evaluation:** one System One request asks three questions:

- `importance` (0–4, rounded to an integer):
  - 0: purely social / tone, nothing about the user
  - 1: the user is asking a question, or it is just a momentary state
  - 2: something settled for the near term (doing X tomorrow), or an uncertain intention (planning / wanting to)
  - 3: a stable trait of the user (I am / I like), a completed event, a general rule of experience
  - 4: a safety / health constraint (allergies, things to avoid)
- `memory_type`: `EPISODIC` (conversations, events, experiences, agreements, promises) or `SEMANTIC` (relatively stable identity, preferences, profile information, constraints)
- `person_fact`: did the speaker volunteer stable information about themselves?

If evaluation fails, it is treated as importance = 2, SEMANTIC.

**A conversation never writes to Brief directly.** Brief is "my beliefs" (judgments about myself); what a conversation contains is "what the other person said". A weighty sentence lands as an experience, and only reflection (see [Reflection](#reflection-reflection-worker)) can turn it into a belief about myself. Casting it straight into a belief would decide for the agent how it should be affected, and Brief does not decay, is not reactivated, and has no delete API. Brief is therefore written only by initialization and by reflection.

**Personal information (profile):** when the `person_fact` probability is ≥ 0.6 (regardless of importance), one LLM call extracts it into controlled slots (name, address, occupation, household, allergies… 16 in all, see `PROFILE_SLOTS`), and it is merged into that person's `profile` in the relationship table. The rules:

- **Strangers are ignored**: personal information from someone not in the relationship table is not recorded (a passer-by who announces their life story gets heard, but not diligently memorized)
- **Dedup**: the same slot with the same value only bumps `mentions` and `last_confirmed_at`, adding no entry
- **Overwrite**: for single-value slots (address, occupation…) the new value takes effect and the old one moves into `history` (at most 2), so "he used to live in Beijing" is still answerable
- **List slots** (household, allergies, long-term preferences, taboos, habits, important dates): no duplicate values, at most 8
- **The relationship is never overwritten**: `relationship` / `description` are untouched, and updating a relationship without passing profile keeps what has been gathered
- It lives in the value of the same Hash as the relationship description (`agent:{agent_id}:relationships`), with no extra key

When read, it goes to System1/2/3 together with the speaker description: `我对{人名}的了解（对方自己说过的，不是我自己的事）：- 住址：上海……`, kept clearly apart from "my beliefs".

**Updating the situation:** another System One request decides whether my situation changed (a new event with a definite date; an existing one rescheduled, resolved or cancelled; a lasting circumstance of mine or of someone I care about changed; what is on my mind changed). Being asked to do something or agreeing to do it (a to-do), or just hearing about something that is already over (an experience), counts as no change. If it changed, `SITUATION_UPDATE_PROMPT` has the LLM output a new, complete situation JSON, which replaces the old one via `replace_situation()`. The prompt includes:
- today's date and weekday, so relative times (tomorrow, next Wednesday, end of month) convert to exact dates; without it, "next Wednesday" used to be written as tomorrow
- the speaker and the few raw lines before this sentence, since elliptical replies ("approved, my last day is the end of the month") need context
- the boundary with to-dos and experiences: actions I am going to take are to-dos, things already over are experiences, and neither goes into the situation
- the same matter rescheduled or progressing rewrites its entry instead of adding one; resolved, cancelled or past items are deleted rather than marked "done" (rendering ignores status, so a kept entry would read as still pending); at most 3 per category

Each event in the new situation that is due within 3 days adds +0.1 to stress.

---

## Episode Consolidation: Working Memory → Episodic

Implementation: `app/service/episode_consolidation_service.py`. It corresponds to episodic consolidation during sleep: the day's conversations first stay in working memory, then are organized into experiences, one per event, and stored in long-term memory. Compared with judging sentence by sentence in real time:

- **It sees the context**: a sentence alone can be worthless ("With me."), yet together with the previous one it is an event (who got the child after the divorce)
- **What "I" said becomes part of experience too**: what I promised, refused, or said falsely (anything marked `[谎言]` is written down as a lie), so it can be recalled when someone follows up
- **It merges with old experiences**: when the same event comes up again, the old episode is updated instead of stored twice; `updated_at` is refreshed, which is how reflection finds recurring events

**Two entry points**, both handling "raw text about to leave working memory": before rolling compression (the oldest 50 entries are about to become a summary) and during daily consolidation (the rest of the day's raw text is about to become the daily summary). Raw text already rolled up was consolidated before compression, so daily consolidation only handles what remains in `stm`.

**Extraction**: one LLM call (`EPISODE_PROMPT`), given the persona fragment and this batch of raw text (with speakers, lie markers and the Memory Gate's importance tags; a separator is inserted when two adjacent entries are more than 30 minutes apart). It splits by event and outputs `{content, speakers, importance, emotion, emotion_charge, has_commitment, has_lie}`; small talk and content-free Q&A are not kept. Episodes are written in the first person, name the other person, and include only what the raw text contains; place names, personal names, titles, numbers and times must be kept in the original words, neither dropped nor paraphrased (e.g. "大理洱海边上" must not become "by the lake" or just "Yunnan"), and the same applies when merging.

**Merging**: the same event reworded scores a cosine of about 0.72–0.81, but "different people, similar events" (Zhang San's divorce / Li Si's divorce) also scores 0.73, so a threshold alone cannot separate them. As with to-do dedup, embeddings first narrow the field: up to 3 entries with cosine ≥ `EPISODE_MERGE_SIMILARITY` (0.65) become candidates, and then the LLM (`MERGE_PROMPT`) decides whether it is the same event. If it is and there is new progress, the two are rewritten as one, updated in place (mem0 update: `created_at` unchanged, `updated_at` refreshed, weight restored, BM25 rewritten); if there is nothing new, the old one is only reactivated. Otherwise a new entry is added, with `source: consolidation`, `speakers`, `occurred_at` (the date of the batch) and so on in the metadata.

**Failure handling**: if the LLM call or parsing fails, it reports failure, and the caller neither compresses nor deletes the raw text, so it is retried next time.

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
  "baseline": {"mood": 0.35, "stress": 0.45, "energy": 0.55},
  "state_updated_at": "...",
  "last_update": "..."
}
```

The example values in this JSON mean: 准备游戏上线 ("preparing the game launch"), Steam首周销量统计 ("Steam first-week sales numbers"), 最近睡眠不好 ("sleeping badly lately"), 游戏稳定性 ("game stability").

### Fact: situation

- Written by: the init / update API and Memory Gate. To-do execution does not write it: an ordinary to-do's "result" is a sentence the LLM makes up, which used to append invented, already-finished things such as "had a deep talk with so-and-so" to the ongoing states
- Format in the prompt: rendered by a single `render_situation()`, shared by System1/2 (`format_for_prompts`) and thought generation (`format_situation_for_prompt`), so both see the same situation. It contains the current activity; time-sensitive events (expired, undated or malformed ones are skipped; the nearest 3 by deadline are shown, each marked "today / tomorrow / N days left", because with a bare date the model cannot tell past from future); up to 3 ongoing states; and up to 3 current focuses (`attention_focus`)
- Given to System1, System2, and the intuitive plan on the fast / memory routes ("my recent situation (answer from this when asked; no need to bring up what isn't asked)"). The situation is by design a few things constantly on one's mind, at most 3 per category, so no relevance selection is needed: the fast / memory routes have no System2, and without the situation the speaking System3 could only make up answers to "when does the certificate expire" or "which book is the reading club doing". Measured (Jarvis and Lin Shiyu, 12 situation questions, 20 small-talk lines): without the situation 2/12 were correct; with the whole situation 12/12, and small talk did not get the situation forced in (an occasional natural mention, such as reminding about short sleep when told about playing tennis, is expected). Selecting only relevant items by similarity (cosine filter + rerank) was tried and did no better (9/12), so it was removed

### Feeling: overall state

- Three continuous, mutually independent values: `mood` (−1 worst ~ 1 best, 0 normal), `stress` (0 ~ 1), `energy` (0 ~ 1). They enter prompts with their scales (`format_state_lines()`), otherwise the model cannot tell whether 0.2 is good or bad
- **Baseline (`baseline`)**: the person's usual state, stored in the subject context. Set by initialization or by `PUT` with overall_state (which also sets the current state to it). Old data without a baseline uses a neutral one (0 / 0.3 / 0.7)
- **Settling by time (`settle_state()`)**: whenever the subject context is read, the time elapsed since `state_updated_at` is applied; no scheduled task is needed (a value nobody reads need not be current, and when read it is computed correctly):
  - mood and stress regress exponentially to the baseline: `x += (baseline − x)·(1 − e^(−hours/τ))`, τ = `STATE_REGRESSION_TAU_HOURS` (6 hours: about 63% back after 6 hours, essentially back after a day; nearly unchanged within minutes, so a continuous conversation is not flattened)
  - energy recovers toward its baseline with τ = `ENERGY_RECOVERY_TAU_HOURS` (4 hours) × (1 + stress), slower under higher stress
  - after settling, `state_updated_at` is set to now. It is separate from `last_update`, which any save refreshes (e.g. adding a to-do) and would wrongly reset the clock. Settling in steps equals settling at once, so reading without writing, or reading and saving for another reason, never regresses twice
- **Adjustment (`adjust_overall_state`)**: settle time first, then add the deltas. Deltas have diminishing returns: they are multiplied by the remaining room to the bound as a fraction of the range, so the nearer the bound the harder to push further while pulling back is barely affected; repeated same-direction stimuli saturate instead of hitting the bound in a few turns
- **Asynchronous adjustment after a conversation** (by System1's emotion and intensity, `EMOTION_STATE_EFFECT`, covering all 16 System1 emotions). Delta = coefficient × intensity:

  | Group | Emotion (mood coef / stress coef) |
  |---|---|
  | Positive · high arousal | 兴奋 0.30/−0.05, 喜悦 0.30/−0.10, 惊喜 0.25/−0.05, 自豪 0.25/−0.10 |
  | Positive · low arousal | 平静 0.05/−0.15, 满足 0.20/−0.15, 信任 0.10/−0.10, 感激 0.20/−0.10 |
  | Negative · high arousal | 愤怒 −0.30/0.20, 焦虑 −0.20/0.20, 恐惧 −0.30/0.25, 沮丧 −0.30/0.15 |
  | Negative · low arousal | 悲伤 −0.30/0.05, 无聊 −0.10/0, 厌恶 −0.20/0.10, 困惑 −0.05/0.05 |

  Previously only 9 were handled (including 高兴 and 放松, which System1 never outputs); 信任, 悲伤, 无聊 and others changed nothing: Jarvis, mostly judged 信任, never changed, while Lin Shiyu, mostly judged 焦虑, never recovered and stuck at mood −1, stress 1
- **Energy cost**: each conversation turn costs `ENERGY_COST_PER_TURN` (0.02) × (1 + arousal × intensity), at most double for high-arousal emotions. Energy is no longer derived from stress (it used to be `1 − stress × 0.7`, which could not express "tired but not stressed")
- An `emotion` life event is recorded when mood or stress changes by more than 0.01 (linked to the turn via `trace_id`); energy-only changes are not recorded. On situation updates, each event due within 3 days adds +0.1 stress (also with diminishing returns)

- Given to System1, System2 and System3

### Fabrication: thoughts

- Generation (`generate_thought`): the LLM gets the current time, the situation and the overall state, and outputs an inner monologue of at most 20 characters, plus the person "I" want to reach, the urgency and the related event. **Nothing is generated when the situation is empty**
- Source 1: generated asynchronously after a conversation. Triggered when System1's emotion intensity > 0.1 or engagement > 0.5. The result is saved to `thoughts` (only the 3 most recent are kept, by time), and a `thought` event is recorded
- Source 2: generated periodically by the Thought Generator Worker. These thoughts are **not saved to `thoughts`**; they only go into the to-do funnel, which decides whether to turn them into to-dos
- Thoughts do not go into any react prompt

---

## To-dos and Autonomous Behavior

### Capabilities

Web search is a very concrete mechanism: something "I" agreed to look up comes back seconds later as a sourced summary, written into the knowledge base, followed by a proactive notice. That suits a personal knowledge base assistant but not a human-like character: a friend who says "I'll look it up later" and returns a structured list half a minute later no longer feels human, and the knowledge base fills up with search snippets.

So whether an agent can search the web is an **agent-level capability switch** (`agent:{agent_id}:capabilities`). It is not written into the persona or left to a prompt to decide: it is whether the agent objectively has this tool, not a personality trait. It is set through `capabilities` in `initialize_agent` / `PUT /admin/agent`, and **by default an agent has no capabilities**.

| | With `web_search` | Without (default) |
|---|---|---|
| Conversation / thought funnel | Asks `needs_web_search` (the conversation funnel also asks `execute_now`) | Not asked; both are false on the task, and the System One request has two fewer questions |
| Immediate execution | Web tasks run right after being written | Not run immediately |
| Execution | Web search → written to Knowledge → requester notified | Runs as a normal task, no web access |

Without the web capability, "I'll look it up for you later" is still an ordinary to-do, executed as a normal task when its time comes.

### Creating to-dos (`add_todo`)

**From conversations** (run in the background at the end of each react turn, without blocking the return). Both the judgment and the extraction see the recent turns (the same ones System1 sees): in confirmations like "look it up" or "OK, go ahead" the content is all in the previous turn, so looking at the current sentence alone loses what is to be done:

1. One System One request judges the following (the last two are asked only when the agent has the `web_search` capability, see [Capabilities](#capabilities)):
   - `has_commitment`: the other person explicitly asked for some action, and "I" explicitly agreed
   - `needs_interaction`: whether completing the task requires talking to someone
   - `needs_web_search`: whether completing the task requires searching the web for information "I" don't have yet
   - `execute_now`: whether to start right now (the other person asks for it now, or no time is given and it can be started immediately); a later time, a condition to wait for, or a reminder for later all count as no
2. When the yes probability of `has_commitment` is ≥ 0.85, the lite model (`DOUBAO_MODEL_LITE`, temperature 0, at most 80 tokens) does two things in one call (it no longer re-checks whether there is a task: on the same input it wrongly answered 无 "none" about 1 time in 4):
   - Turn the other person's words into a task description (at most 30 characters, not copying the original words)
   - Work out the deadline (`YYYY-MM-DD`) from any time mentioned. The prompt includes today's date and weekday, so expressions like 明早 ("tomorrow morning"), 下周三 ("next Wednesday") and 月底 ("end of the month") can all be resolved. If no time is mentioned, it is left empty

   The prompt is about 150 tokens, and the output is one or two lines of plain text (not JSON). Measured at about 1 second per call
3. If it passes, the task is created with priority 0.8, and the related person is the speaker

When talking to someone is needed, the task carries `interaction_type: "dialogue"`; when web search is needed, `needs_web_search: true`; when it should be done now, `execute_now: true`.

**Immediate execution:** the to-do is written as usual (dedup still applies). If `execute_now = true`, it is a **web search task** and the agent has the `web_search` capability, `cyber_brain` immediately calls `task_executor.run_task()` without waiting for the Task Reminder schedule; on success it is removed from the to-dos, on failure it stays and the worker retries at the next check. Only web search tasks are allowed because:
- "Executing" an ordinary task means the LLM invents a one-sentence result; doing it immediately would put something that never happened into the very next turn
- A social task created from conversation always targets the current speaker (the person already talking to "me"), so "reaching out" right away makes no sense

`run_task()` keeps an in-process set of running tasks, so immediate execution and the worker's check never run the same task twice.

**From thoughts** (Thought Generator Worker): one System One request judges both `need_task` (is there something concrete "I" want to do, including wanting to find out, verify or fill in some piece of knowledge) and `needs_web_search` (asked only with the `web_search` capability). If the yes probability of `need_task` is ≥ 0.8, the to-do is created directly (its priority is that probability); 0.6–0.8 lets the LLM confirm; anything lower is not created.

> The judgment uses `probabilities["yes"]` from System One, not `confidence`. The latter measures how certain the judgment is (about 1 − entropy): with a yes probability of 0.755, `confidence` is only 0.197, so it cannot be used as "the probability of yes".

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

`execute_task` checks in this order, web search first (the decision model often also marks "look it up and add it to the knowledge base" as needing contact, but the substance of such a task is getting information):

- **Web search tasks** (`needs_web_search = true` and the agent has the `web_search` capability; the capability is checked again at execution and tasks without it run as normal tasks. Web tasks created before the switch was turned off, or written directly through the API, skip the funnel and are caught here):
  1. The LLM rewrites the to-do into **two** queries: one about the thing itself, and one about its risks, limitations, counter-evidence or controversy. The two Tavily searches run in parallel (`app/client/web_search_client.py`, `WEB_SEARCH_MAX_RESULTS` results each) and are merged and deduplicated by URL. With a single search, the knowledge leans whichever way the results happen to lean
  2. Searches prefer the sites in `WEB_SEARCH_PREFER_DOMAINS` (journals, PubMed, health authorities, etc.; Tavily's `prefer` mode only prioritizes them and still fills in from other sites)
  3. The LLM first labels the **source type** of each result (学术研究 academic research / 官方或专家共识 official or expert consensus / 新闻媒体 news media / 商家或营销 commercial or marketing), then turns **only the excerpts** into 1–3 knowledge entries. The body must say which kind of source a claim comes from, include both sides when there is evidence on both, say when evidence comes only from animal or small-sample studies, never state commercial claims as conclusions, and mark citations with `[1][2]`
  4. Each entry is written to Knowledge with confidence set by the **most reliable type** among its cited sources (academic research and official/expert consensus 0.75, news media 0.55, commercial or marketing 0.35); the title, type and link of each source are appended to the body, `source = learning`
  5. One Episodic experience is written ("I searched the web for X and added to the knowledge base: …" or "… found nothing reliable"), with both queries and the source links in its metadata
  
  If `TAVILY_API_KEY` is not configured or the search fails, the task returns failure and **stays pending**, to be retried at the next check. No result is invented
- **Ordinary tasks**: the LLM simulates a one-sentence result in the first person, which is written to Episodic as "[自主完成] …" ("[done on my own] …"). Then the LLM decides whether to update the situation (current activity / event completed / new ongoing state). Simulated results are no longer written to Knowledge
- **Social tasks** (`interaction_type = dialogue`): first check that `related_person` is an existing agent (has a persona); if not, the task is **cancelled** (react is not called, so no stray data is created under a person's name, and the worker does not retry it forever). Once that passes, the LLM generates an opening line of at most 15 characters and calls `react()` of the **target agent** (`related_person` is used as the target agent_id), with the initiator as the speaker. Both sides' words are written as one experience into the initiator's Episodic, and `recent_contacts` is appended to the initiator's situation (keeping the 10 most recent)

After successful execution, the task is **removed from `pending_tasks`**. Its content and vector move to `recent_done` (used only for dedup and cleaned up automatically once outside the window). The full task record is written to a `todo` life event (action=completed), and a `task_execution` event is recorded. At service startup, completed/cancelled tasks left in `pending_tasks` by older versions are migrated out (idempotent).

---

## Proactive Outreach

Implementation: `app/service/outreach_service.py`. Two sources of the agent speaking first:

- **Task completed**: when a web task done on someone's behalf finishes (`notify_task_done`), tell the requester what was found; a social task (`interaction_type = dialogue`) reaches out to the person in the task
- **Thoughts**: thoughts from the Thought Generator carry a `target_person` (chosen only from the names in the relationship table). `maybe_express_thought` makes one System One call on whether to say it now, given "who I am" (`persona_system1`), who the other person is, the recent conversation and the thought; it is said only if the yes probability is ≥ 0.6. Whether to speak is up to the persona, and having just said something similar also leads to "no"

**Delivery** (`deliver`) is decided by the other party's kind in the relationship table, never guessed from the name:

| Recipient | How |
|---|---|
| kind = human | Pushed to the outbox `outbox:{agent_id}:{name}`; the frontend calls `GET /brain/fetch_talk?agent_id=&human=` to take the messages (taking deletes them) |
| kind = agent | Calls the other agent's react using the agent_id recorded in the relationship; the speaker is the name the other agent knows "me" by (or "my" agent_id if it has no record); the reply is written to "my" working memory |
| Not in the relationship table | No outreach (a stranger); the task is cancelled, and no stray data is created under a person's name |

The words are produced by System3 (with the speaking manner; the speaker is the recipient and "what they just said" is "they said nothing, I'm speaking first"), then written to "my" working memory marked "(reached out to someone)", so later conversations know it was said; a `proactive_talk` life event is also recorded.

### Reflection (Reflection Worker)

Implementation: `app/worker/reflection_worker.py`. It turns experiences into beliefs, modelling the replay and consolidation that happen during sleep. **Brief is written only by initialization and by this worker.**

**It is not scheduled on its own**: after daily consolidation (the PFC Worker) has processed every agent, it calls `run_once()`. Reflection must see the episodes consolidated and reactivated today, so it has to run after consolidation. The run time is recorded in Redis before running, so it does not run again within 20 hours.

**Candidates**: Episodic memories whose `updated_at != created_at` (they were mentioned again; reactivation updates `updated_at`) and were updated within the last 24 hours. Fact-type entries (consolidated episodes with `source = consolidation`, and `gate_type = SEMANTIC` entries written by the old Memory Gate) are always included; experiences with `importance ≥ 3` are included according to `REFLECTION_INCLUDE_HIGH_IMPORTANCE`; system receipts (`type = task_completion`) are excluded. Sorted by importance, the top `REFLECTION_MAX_CANDIDATES` (30) are taken.

**Judgment**: one LLM call, given the persona fragment (`persona_system1`), every current Brief entry, and the candidate experiences (numbered). The prompt is `REFLECTION_PROMPT` and applies seven tests to each candidate: ownership (someone else's words are not a belief), stability (a one-off event is not a belief), independence, testability, non-duplication, non-transgression (personal facts belong in the profile), and non-self-harm (not copying someone's accusation into a judgment about oneself). The output is JSON: `{memory_ids, kind, action, target_index, text, reason, confidence}`.

**Writing back (bounded step)**: at most `REFLECTION_MAX_ACTIONS` (2) entries per agent per round; anything with `confidence < REFLECTION_MIN_CONFIDENCE` (0.75) is dropped; before an `add` the text is checked for duplicates (against what was written this round, and against existing beliefs retrieved from Brief, with a rerank score ≥ 0.85 meaning the same thing), so the model cannot pile up near-identical variants that differ by one qualifier; `revise` / `merge` never delete the old entry, they only write `superseded_by` on it, which excludes it from retrieval while keeping the history (rollback and explanation stay possible). Every reflected belief carries `source: reflection`, `evidence_ids`, `confidence` and `reason`, and a `thought_type = reflection` life event is recorded.

## Background Workers

The four workers are started with `asyncio.create_task()` when the service starts, and stop with the service.

| Worker | When it runs | Targets |
|---|---|---|
| Decay Worker | When `DECAY_WORKER_INTERVAL` (86400 seconds) has passed since the last run. The last run time is stored in Redis, so a restart does not trigger an extra run | All agents that appear in the Episodic collection |
| PFC Worker | Every day at `PFC_DAILY_HOUR` (4) o'clock; triggers reflection afterwards | All agents with working memory |
| Thought Generator | **Waits** one `THOUGHT_GENERATOR_INTERVAL` (3600 seconds) after startup, then runs every interval | All agents with a subject context |
| Task Reminder | **Runs once immediately at startup**, then every `TASK_REMINDER_INTERVAL` (3600 seconds) | All agents with a subject context |

### Decay Worker: memory decay

For each Episodic memory: `decay_weight × DECAY_FACTOR` (0.96). If the result is below `DECAY_MIN_WEIGHT` (0.2), the memory is deleted; otherwise the weight is written back. With the default parameters, a memory that is never mentioned again is deleted at about the 40th decay (about 40 days). After reactivation the weight returns to 1.0 and the count starts over. Only Episodic decays; Brief and Knowledge do not.

Scheduling: the unix timestamp of the last run is stored in Redis under `worker:decay:last_run`. Each time the worker wakes up, it recomputes how long until the next run, and runs only when due. After a run (successful or not), it updates the timestamp. So a service restart does not cause an extra decay round, and frequent restarts do not keep pushing decay back. If it has never run, it runs at startup.

### PFC Worker: daily consolidation of working memory

For each agent (sharing the lock with rolling compression; the lock duration is fixed at 3600 seconds):

1. Using the current `stm` length as a cursor, read all compressed blocks and the raw entries before the cursor
2. [Extract episodes](#episode-consolidation-working-memory--episodic) from the raw entries before the cursor into Episodic
3. The LLM generates a first-person daily summary of at most 150 characters
4. This one daily summary block (with all of the day's speakers) **replaces** the whole compressed queue, and the raw entries before the cursor are deleted (messages that arrive during consolidation are kept)
5. New speakers from the previous day are added to the relationship table
6. **Review the situation** (`_review_situation`, `SITUATION_REVIEW_PROMPT`): in-conversation situation updates see one sentence at a time and over time leave behind finished circumstances, duplicates, to-dos that slipped in, and things already over. Expired events are first removed in code; then the current situation (numbered item by item), the to-do list, the daily summary and the day's raw conversation go to the LLM, which **judges each item in two steps**:
   - step 1 `kind`: what the item itself is: situation (a lasting circumstance, a dated event, something on my mind), to-do (an action I am to take), or past (something already over). This looks only at the item, regardless of whether it came up today; to-dos and past items are deleted by code
   - step 2 `action`, for situation items only: keep / delete (today's conversation shows it ended, was resolved or cancelled, or it duplicates another) / rewrite (new progress). **Items with no evidence of having ended are kept**: not being mentioned for a long time does not mean something is over. Relative-time phrases ("this week", "the last two weeks") are changed only when today's conversation shows they no longer hold, and are never converted to specific dates (the original writing date is unknown)

   Additions accept only situation items, and those highly similar to a kept item are dropped. Rewrites and additions contain only information that appeared in today's conversation. Code assembles the result from the decisions: kept items are copied verbatim, undecided items count as keep; states and focuses are capped at 3 each, dropping additions first. The model is not called if there was no raw conversation that day; invalid output is retried once, and if it still fails the situation stays unchanged.

   Why two steps: asking the model to output a whole new situation made deletions hit or miss from run to run; with per-item decisions in a single layer, the model often kept to-dos and past items "because they weren't mentioned today", since "keep without evidence of ending" and "delete what isn't a situation" competed at the same level. Separated, 5 out of 5 runs matched expectations.

If episode consolidation or the summary fails, nothing is cleaned up, and the data waits for the next run. The daily summary stays only in working memory and is not written to long-term memory. After every agent is processed, [reflection](#reflection-reflection-worker) is triggered.

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
| `proactive_talk` | Something said proactively to someone (a task-done notice or a thought worth saying), with the delivery method outbox / react |

LLM token usage is collected with `contextvars`: `react()` calls `start_usage_collection()` at the start, and every `chat` call in the same context appends its usage.

Query: `GET /admin/timeline/{agent_id}?event_types=react_input,react_output&page=1&page_size=50`.

---

## Admin API and Runtime Configuration

| Endpoint | Behavior |
|---|---|
| `POST /admin/initialize_agent` | Writes persona, Brief/Episodic/Knowledge, relationships, situation, overall state, thoughts and to-dos in one go. A Redis lock prevents concurrent duplicate initialization. Returns 409 if the agent already has any data in Redis or Qdrant. **The persona is written only when both system1 and system2 are provided**. `capabilities` sets the capabilities (currently only `web_search`); omitted means none |
| `GET /admin/agents` | The union of agents from persona (`persona:*:system1`) and subject_context |
| `GET /admin/agent/{agent_id}` | Returns full information in the same structure as the init API |
| `PUT /admin/agent/{agent_id}` | Overwrite update; only the fields passed are updated. situation / overall_state / thoughts / pending_tasks are replaced as a whole (entries in pending_tasks whose status is not pending are ignored); relationships are cleared and then written (passing `[]` clears them; people still in the list keep the profile gathered from conversations); capabilities are replaced as a whole (passing `[]` turns all off). Returns 404 if the agent does not exist |
| `POST /admin/agent/{agent_id}/memories` | Appends Brief / Episodic / Knowledge; each entry is written as a new record. Returns 404 if the agent does not exist |
| `DELETE /admin/agent/{agent_id}/pfc` | Deletes `stm` and `stm:compress` |
| `GET /brain/fetch_talk` | Params `agent_id`, `human`: takes what the agent proactively wants to say to this person; taking deletes them. See [Proactive Outreach](#proactive-outreach) |
| `GET /admin/config` | Returns the name, current value, default value, type, description and value range of the 36 tunable parameters |
| `PUT /admin/config` | Changes parameters; see below |

### Runtime configuration

- The whitelist is in `app/config/runtime_config.py`. It has 36 items, covering Brief/Episodic retrieval, memory decay, working memory, react routing and knowledge retrieval and injection, System2/3 temperature and max tokens, the memory reactivation threshold, to-do dedup, routing fallbacks, the number of recent turns for System1, and more. The code reads all of these as `settings.X` at call time, so a change takes effect the next time the value is used
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
- Knowledge's retrieval count (5) is hard-coded, not in the configuration
- The Knowledge rerank threshold was calibrated on a 17-entry single-topic knowledge base, with little margin on either side (0.484 / 0.529); re-check it when the size or topics of the knowledge base change
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
