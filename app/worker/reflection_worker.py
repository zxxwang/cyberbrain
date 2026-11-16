"""
Reflection Worker: 把经历沉淀成信念（模拟睡眠中的记忆重放与巩固）

白天发生的事先作为"经历"存在 Episodic。只有经过反复出现、情绪确认、或与已有认识反复印证的，
才沉淀成"信念"（Brief）。对话不直接写 Brief（见 memory_gate_service），Brief 只由这里写入。

每天每日整理之后跑一次（由 prefrontal_lobe_memory_worker 调用 run_once）：
1. 取每个 agent 的 Episodic 中"被碰过"（updated_at != created_at）且 24 小时内更新过的记忆
2. 连同该 agent 现有的人格片段和全部 Brief 一起交给 LLM，判断哪些够格沉淀成信念
3. 有界步长：每个 agent 每轮最多 2 条；confidence 低于阈值一律不写
4. revise / merge 不删除旧条目：给旧条目写 superseded_by，检索时忽略但保留历史（可回滚、可解释）
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.client.doubao_client import doubao_client
from app.client.redis_client import redis_client
from app.config.settings import settings
from app.repository.memory_repository import memory_repository
from app.service.agent_life_event_service import agent_life_event_service
from app.service.persona_manager import persona_manager

logger = logging.getLogger(__name__)

LAST_RUN_KEY = "worker:reflection:last_run"  # 上次执行的 unix 时间戳（秒）

REFLECTION_PROMPT = """你是这个 Agent 的【反思】模块，模拟人在睡眠中的记忆重放与巩固。

白天发生的事先作为"经历"存在情景记忆里。只有经过反复出现、情绪确认、
或与已有认识反复印证的，才会沉淀成"信念"。你现在只做这一件事：
判断下面这批被重新提起的经历里，有没有够格沉淀成信念的，以及沉淀成什么。

## 输入

【我是谁】
{persona}

【我现在的信念】（带编号的现有 Brief）
{beliefs}

【最近被重新提起的经历】（过去 24 小时内被对话重新激活的情景记忆）
{memories}

注意：经历里的"我"指的是说这句话的人，不是你自己。例如
"张三对我说：我住在北京"里的"我"是张三。

## 判定一个候选是否够格，逐条问，全部为"是"才可能够格

1. 归属：这是"我对自己的认识 / 我对某段关系的判断 / 我对世界的判断"，
   还是"别人对我说的一句话"？后者不是信念。
2. 稳定性：这是反复出现的模式，还是一次性的事件？一次性的事件哪怕再激烈，
   也只是经历，不是信念。
3. 独立性：这条信念能不能脱离那件事单独成立？必须依赖"那天他骂了我"才成立的，
   是记忆，不是信念。
4. 可检验性：将来的对话里，有没有可能发现它是错的？无法被检验的空话不算信念。
5. 不重复：现有信念里是否已经有同一条？有则用 revise 或 merge，不要 add。
6. 不越界：这是关于某一个具体的人的事实（住址、职业、家人、身体状况、联系方式）吗？
   那是"我对这个人的了解"，不是"我的信念"→ action 填 pass，kind 填 person_fact。
7. 不反噬：这条信念是否只是把别人一时的情绪、评价、指责照搬成了对自己的判断？
   "你什么都做不好" → "我什么都做不好" 是错的。除非有多条独立证据、且反复出现，
   否则 pass。

## 输出

只输出 JSON，不要任何解释文字、不要 Markdown 代码块：

{{
  "decisions": [
    {{
      "memory_ids": ["M1", "M3"],
      "kind": "self_belief | relationship_belief | world_fact | person_fact | noise",
      "action": "add | revise | merge | pass",
      "target_index": 3,
      "text": "……",
      "reason": "……",
      "confidence": 0.0
    }}
  ]
}}

字段说明：
- memory_ids：支撑这条判断的经历编号，至少一个
- target_index：action 为 revise / merge 时填被修改的现有信念编号，其他情况填 null
- text：action 为 add / revise 时给出新信念。第一人称、具体、朴素、能指导以后的行为，
  不超过 40 个字。不要用"我意识到""我明白了""我学会了"这类空话开头
- reason：不超过 60 字，说明它满足了上面哪几条，以及依据是什么
- confidence：0 到 1，你对"这条够格"的确信程度

硬性约束：
- 最多 2 条 action 属于 add / revise / merge，其余全部 pass
- confidence 低于 0.75 的一律 pass
- revise 只能修改一条现有信念，merge 只用于两条现有信念明显是同一件事
- 宁可少写。今天没有够格的，返回 {{"decisions": []}}

## 示例

例 1（add：反复出现的模式）
经历：
[M2] 主人熬夜的第二天，情绪明显烦躁，讲道理会顶回来
[M5] 主人没睡好的时候，只想听结论，不想听过程
[M8] 主人熬夜后，对细节的追问容忍度很低
输出：
{{"decisions":[{{"memory_ids":["M2","M5","M8"],
"kind":"relationship_belief","action":"add","target_index":null,
"text":"主人熬夜之后容易烦躁，那时先把事办完，少讲道理",
"reason":"三条独立经历指向同一规律，可检验、不依赖单次事件","confidence":0.85}}]}}

例 2（pass：一次性的激烈评价）
经历：
[M1] 张三对我说：你根本不懂怎么做生意，这店迟早毁在你手里
输出：
{{"decisions":[{{"memory_ids":["M1"],"kind":"noise","action":"pass","target_index":null,
"text":"","reason":"一次性激烈情绪，只有一条证据，不能变成对自己的判断","confidence":0.9}}]}}

例 3（pass：关于某个人的事实）
经历：
[M4] 李四对我说：我刚搬到上海，现在在浦东上班
输出：
{{"decisions":[{{"memory_ids":["M4"],"kind":"person_fact","action":"pass","target_index":null,
"text":"","reason":"这是关于李四本人的事实，应记入对他的了解，不是我的信念","confidence":0.95}}]}}

例 4（revise：新证据与旧信念冲突）
现有信念：
7. 我不适合做需要长期坚持的事
经历：
[M3] 我已经连续三个月每天写稿，没有断过
[M6] 上次想放弃的时候，第二天还是坐回了书桌前
输出：
{{"decisions":[{{"memory_ids":["M3","M6"],
"kind":"self_belief","action":"revise","target_index":7,
"text":"我能坚持的事，得是我自己真心想做的",
"reason":"多条证据与旧信念冲突，是修正不是推翻","confidence":0.8}}]}}

## 绝对不要

- 不要把"别人说了什么"写成"我是什么样的人"
- 不要把人名、住址、职业、家人、身体状况写进信念
- 不要因为一次冲突就生成"这段关系靠不住了"之类的信念
- 不要输出和现有信念意思相同、只换了说法的内容
- 不要在 JSON 之外输出任何文字
"""


def _parse_ts(value: Any) -> Optional[datetime]:
    """payload 里的时间是带时区或 naive 的 ISO 字符串，统一转成 naive 本地时间比较"""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone().replace(tzinfo=None) if dt.tzinfo else dt


class ReflectionWorkerService:
    def __init__(self):
        self.running = False

    async def run_once(self) -> None:
        """
        跑一轮反思。由每日整理（prefrontal_lobe_memory_worker）在整理完经历后调用：
        反思要看到今天刚整理出、刚被重激活的经历，必须排在整理之后，所以不再单独定时。
        LAST_RUN_KEY 先记再跑，防止同一天被重复触发
        """
        try:
            last_run = await redis_client.get(LAST_RUN_KEY)
            if last_run and time.time() - float(last_run) < 20 * 3600:
                logger.info("Reflection skipped: already ran within 20 hours")
                return
        except Exception as e:
            logger.warning(f"Failed to read {LAST_RUN_KEY}: {e}")
        await redis_client.set(LAST_RUN_KEY, str(time.time()))
        try:
            await self._reflect_all()
        except Exception as e:
            logger.error(f"Error in reflection run: {e}", exc_info=True)

    # ---------------- 主流程 ----------------

    async def _reflect_all(self):
        agent_ids = await self._discover_agents()
        logger.info(f"Reflection: processing {len(agent_ids)} agents")
        for agent_id in agent_ids:
            lock = f"reflection:{agent_id}"
            if not await redis_client.acquire_lock(lock, 1800):
                logger.info(f"Reflection skipped for {agent_id}: already running elsewhere")
                continue
            try:
                await self.reflect_agent(agent_id)
            except Exception as e:
                logger.error(f"Reflection failed for {agent_id}: {e}", exc_info=True)
            finally:
                await redis_client.release_lock(lock) if hasattr(redis_client, "release_lock") else None

    async def _discover_agents(self) -> List[str]:
        agents = set()
        try:
            for mt in ("episodic", "brief"):
                client = memory_repository.episodic_client if mt == "episodic" else memory_repository.brief_client
                if not client:
                    continue
                vs = client.vector_store
                offset = None
                while True:
                    points, offset = vs.client.scroll(
                        collection_name=vs.collection_name, limit=100, offset=offset, with_payload=["user_id"])
                    for pt in points:
                        if pt.payload.get("user_id"):
                            agents.add(pt.payload["user_id"])
                    if offset is None:
                        break
        except Exception as e:
            logger.error(f"Failed to discover agents for reflection: {e}", exc_info=True)
        return sorted(agents)

    async def _candidates(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        候选：Episodic 中被"碰过"（updated_at != created_at）且最近 24 小时更新的记忆。
        事实型（gate_type=SEMANTIC）必选；高重要度经历按开关纳入。
        """
        cutoff = datetime.now() - timedelta(hours=24)
        out = []
        for mem in await memory_repository.get_all(agent_id, memory_type="episodic"):
            created = _parse_ts(mem.get("created_at"))
            updated = _parse_ts(mem.get("updated_at"))
            if not created or not updated or updated == created or updated < cutoff:
                continue
            if mem.get("type") == "task_completion":
                continue  # 系统执行回执，不是经历
            # 事实型：旧版 Memory Gate 写入的 SEMANTIC，或整理时提取的经历（本身就是筛过的"值得记住的事"）
            is_fact = mem.get("gate_type") == "SEMANTIC" or mem.get("source") == "consolidation"
            is_important = int(mem.get("importance") or 0) >= 3
            if is_fact or (settings.REFLECTION_INCLUDE_HIGH_IMPORTANCE and is_important):
                out.append(mem)
        out.sort(key=lambda m: (int(m.get("importance") or 0), m.get("updated_at", "")), reverse=True)
        return out[:settings.REFLECTION_MAX_CANDIDATES]

    async def reflect_agent(self, agent_id: str) -> int:
        """对一个 agent 做一轮反思，返回实际写入的信念条数"""
        candidates = await self._candidates(agent_id)
        if not candidates:
            logger.info(f"Reflection {agent_id}: no candidates")
            return 0

        persona = (await persona_manager.get(agent_id))["system1"]
        beliefs = await memory_repository.get_all(agent_id, memory_type="brief")
        beliefs = [b for b in beliefs if not b.get("superseded_by")]  # 已被修正的旧信念不参与

        memory_block = "\n".join(
            f"[M{i + 1}] {m.get('updated_at', '')[:16]}（重要度 {m.get('importance', '?')}，"
            f"{'事实型' if m.get('gate_type') == 'SEMANTIC' else '经历'}）\n{m.get('memory', '')}"
            for i, m in enumerate(candidates))
        belief_block = "\n".join(f"{i + 1}. {b.get('memory', '')}" for i, b in enumerate(beliefs)) or "（还没有）"

        prompt = REFLECTION_PROMPT.format(persona=persona, beliefs=belief_block, memories=memory_block)
        try:
            raw = await doubao_client.generate(prompt, max_tokens=1200)
            decisions = self._parse_decisions(raw)
        except Exception as e:
            logger.error(f"Reflection LLM failed for {agent_id}: {e}", exc_info=True)
            return 0

        applied = 0
        written: List[str] = []  # 本轮已写入的文本，避免同一轮里写两条意思相同的
        for d in decisions:
            if applied >= settings.REFLECTION_MAX_ACTIONS:
                break
            if d["action"] not in ("add", "revise", "merge"):
                continue
            if d["confidence"] < settings.REFLECTION_MIN_CONFIDENCE:
                logger.info(f"Reflection {agent_id}: skip low confidence {d['confidence']:.2f} — {d['text'][:30]}")
                continue
            if d["action"] == "add" and await self._already_known(agent_id, d["text"], beliefs, written):
                logger.info(f"Reflection {agent_id}: skip duplicate belief 「{d['text'][:30]}」")
                continue
            ok = await self._apply(agent_id, d, candidates, beliefs)
            if ok:
                written.append(d["text"])
            applied += ok
        logger.info(f"Reflection {agent_id}: candidates={len(candidates)} applied={applied}")
        return applied

    async def _already_known(self, agent_id: str, text: str, beliefs: List[Dict], written: List[str]) -> bool:
        """
        这条信念是否已经有了：先和本轮已写入的比（去掉标点后包含即算重复），
        再用 Brief 检索现有信念，rerank 分数 ≥ 0.85 视为同一件事。
        反思模型常给出"多一个限定词"的变体，只靠它自己判断会写出一堆近似条目
        """
        norm = lambda s: "".join(ch for ch in s if ch not in "，。、；：,.;:!！?？ ")
        n = norm(text)
        if any(n in norm(w) or norm(w) in n for w in written):
            return True
        try:
            hits = await memory_repository.search_brief(agent_id, text, top_k=3)
        except Exception as e:
            logger.warning(f"Duplicate-belief check failed, treat as new: {e}")
            return False
        for h in hits:
            if (h.get("rerank_score") or 0) >= 0.85:
                logger.info(f"Reflection: 「{text[:26]}」与已有信念「{h.get('memory', '')[:26]}」重复")
                return True
        return False

    @staticmethod
    def _parse_decisions(raw: str) -> List[Dict[str, Any]]:
        text = (raw or "").strip()
        if "{" in text and "}" in text:
            text = text[text.index("{"): text.rindex("}") + 1]
        try:
            data = json.loads(text)
        except Exception:
            logger.warning(f"Reflection output is not JSON: {raw[:200]}")
            return []
        out = []
        for d in data.get("decisions") or []:
            action = (d.get("action") or "").lower()
            text_value = (d.get("text") or "").strip()
            if action in ("add", "revise", "merge") and not text_value:
                continue
            out.append({
                "action": action,
                "text": text_value,
                "target_index": d.get("target_index"),
                "memory_ids": d.get("memory_ids") or [],
                "confidence": float(d.get("confidence") or 0.0),
                "reason": (d.get("reason") or "")[:200],
                "kind": d.get("kind") or "",
            })
        return out

    async def _apply(self, agent_id: str, d: Dict[str, Any], candidates: List[Dict], beliefs: List[Dict]) -> int:
        evidence = [candidates[int(m[1:]) - 1]["id"] for m in d["memory_ids"]
                    if isinstance(m, str) and m.startswith("M") and m[1:].isdigit()
                    and 0 < int(m[1:]) <= len(candidates)]
        new_id = await memory_repository.add(
            agent_id=agent_id,
            messages=[{"role": "user", "content": d["text"]}],
            memory_type="brief",
            metadata={
                "source": "reflection",
                "kind": d["kind"],
                "evidence_ids": evidence,
                "confidence": d["confidence"],
                "reason": d["reason"],
                "created_at": datetime.now().isoformat(),
                "tag": "reflection",
            },
        )
        if not new_id:
            return 0
        # revise / merge：旧条目标记 superseded_by，不删除（可回滚、可解释）
        target = d.get("target_index")
        if d["action"] in ("revise", "merge") and isinstance(target, int) and 0 < target <= len(beliefs):
            await memory_repository.mark_superseded(beliefs[target - 1]["id"], new_id[0], memory_type="brief")
        logger.info(f"Reflection {agent_id}: {d['action']} 「{d['text']}」 ({d['reason']})")
        await agent_life_event_service.record_thought(
            agent_id=agent_id,
            thought=f"[反思] {d['action']}：{d['text']}（{d['reason']}）",
            thought_type="reflection",
            context={"kind": d["kind"], "evidence_ids": evidence, "confidence": d["confidence"]},
        )
        return 1


reflection_worker = ReflectionWorkerService()
