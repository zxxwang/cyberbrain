"""
Episode Consolidation Service: 从工作记忆的原文里整理出经历，写入 Episodic

对应人在睡眠中的情景记忆巩固：白天的对话先留在工作记忆（PFC）里，之后再按"一件事"
切分、整理成经历存进长期记忆。和实时逐句判断相比：
- 看得到上下文：单独一句没价值（"跟我。"），连起来才是一件事（离婚后孩子跟谁）
- "我"说过的话也进经历：答应过什么、拒绝了什么、哪句是假话，以后被追问时才想得起来
- 和旧经历合并：这件事以前就记过时，重激活旧的那条，不重复存；反思据此找到反复发生的事

两个入口，处理的都是"即将从工作记忆里消失的原文"：
- 滚动压缩之前（prefrontal_lobe_memory.compress）：stm 超长，最早一批要被压成摘要
- 每日整理（prefrontal_lobe_memory_worker）：当天剩下的原文要被压成当日摘要
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.client.doubao_client import doubao_client
from app.config.settings import settings
from app.repository.memory_repository import memory_repository
from app.service.agent_life_event_service import agent_life_event_service
from app.service.persona_manager import persona_manager

logger = logging.getLogger(__name__)

SELF_SPEAKERS = ("我",)

EPISODE_PROMPT = """你是这个 Agent 的【情景记忆整理】模块。下面是"我"一段时间内的对话原文，
这些原文马上要从工作记忆里清掉。你要把其中**值得长期记住的事**整理成经历，其余的放手忘掉。

## 我是谁
{persona}

## 对话原文（按时间先后；"[我]"是我说的，其他方括号里是对方的名字；[谎言] 标记的是我故意说的假话；
## 「重要度」是当时对这句话的第一印象（0–4），仅供参考，要结合上下文重新判断）
{records}

## 什么算一件"值得记住的事"

按事件切分，不按句子：一问一答、来回几句说的是同一件事，就合成一条。单独一句看似无关紧要（"跟我。"），
但和上下文连在一起是重要的事（孩子判给了谁），就要记；反过来，一句话本身很重，但上下文说明只是玩笑，就不记。

值得记的：
- 对方的重要经历、处境变化、做出的决定、遇到的困难（离职、生病、搬家、分手、考试结果……）
- 双方的约定、承诺、请求，以及我的回应（答应了 / 拒绝了 / 说要考虑）
- 我说过的、以后可能被追问的话：给过的建议、表过的态度、**说过的假话**（要写明是假话，真相是什么如果知道也写上）
- 让我或对方情绪明显波动的事（争执、感动、被冒犯、和好）
- 对方明确说过的偏好、忌讳、底线（以事件的形式记：谁、在什么情况下说的）

不记的：
- 寒暄、客套、没有实质内容的闲聊
- 只是问了个问题、我回答了常识（没有承诺、没有情绪、没有对方的个人信息）
- 已经在原文里反复出现的同一件事，只记一次
- 我的内心独白、对规则的复述

## 写法

- 第一人称（"我"就是我），写清楚是谁：用对方的名字，不要写"对方""用户"
- 写成一段完整的经历（40–120 字）：谁、什么事、双方怎么说的、结果怎样
- **关键细节必须照原文的字写，不能省略、换说法或概括**：
  - 地名：原文说"大理""洱海边上"，就写"大理洱海边上"，不能只写"洱海边"或改成"云南"
  - 人名、店名、书名、机构名：照原文写全（"实验二小"不能写成"一所小学"）
  - 数字和时间：原文说"十二本""下周六""三个月"，就写"十二本""下周六""三个月"，不能写成"很多本""周末""一段时间"
  - 这些细节是以后被问"在哪儿""什么时候""多少"时唯一的依据；字数不够时删修饰语和我的感受，不删细节
- 只写原文里有的，不要补充、推测、润色
- 原文里我说了假话的，写成"我跟某某说……（这是假话，实际上……）"

## 输出

只输出 JSON，不要任何其他文字：

{{"episodes": [
  {{"content": "……", "speakers": ["张三"], "importance": 3, "emotion": "担忧", "emotion_charge": 0.6,
    "has_commitment": true, "has_lie": false}}
]}}

- importance：1–4（1 一般；2 有点意义；3 重要；4 改变关系、涉及健康安全或重大承诺）
- emotion：这件事带给我的主要情绪（一个词）；emotion_charge：情绪强度 0–1
- 没有值得记的，输出 {{"episodes": []}}
"""


MERGE_PROMPT = """判断一条新整理出的经历，和下面几条已有经历里的某一条是不是**同一件事**。

同一件事：同样的人、同一个事件（或它的后续进展、再次提起）。
不是同一件事：不同的人遇到类似的事（张三离婚 / 李四离婚）；同一个人的不同事情（张三离婚 / 张三出差）。

新经历：
{new}

已有经历：
{candidates}

只输出 JSON：
{{"same": 编号或 0, "merged": "……"}}

- same：是同一件事时填那条已有经历的编号，都不是填 0
- merged：same 不为 0 时，把新旧两条合写成一条完整经历（120 字以内，有新进展就写上进展；只写两条里有的内容，不补充）。
  两条里的地名、人名、数字、时间都要照原字保留，不能在合写时丢掉或换说法。新经历没有带来任何新内容时，merged 留空
"""


class EpisodeConsolidationService:

    async def consolidate(self, agent_id: str, records: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
        """
        把一批工作记忆原文整理成经历写入 Episodic。

        Returns:
            {"episodes": 整理出的条数, "added": 新写入, "merged": 合并进旧经历（重激活）}；
            LLM 调用或解析失败返回 None（调用方据此保留原文，不压缩、不删除）
        """
        raw = [r for r in records if r.get("type") == "raw" and (r.get("content") or "").strip()]
        if not raw:
            return {"episodes": 0, "added": 0, "merged": 0}

        episodes = await self._extract(agent_id, raw)
        if episodes is None:
            return None

        day = self._day_of(raw)
        added = merged = 0
        for ep in episodes:
            try:
                if await self._merge_into_existing(agent_id, ep):
                    merged += 1
                    continue
                await memory_repository.add(
                    agent_id=agent_id,
                    messages=[{"role": "user", "content": ep["content"]}],
                    memory_type="episodic",
                    metadata={
                        "source": "consolidation",
                        "speakers": ep["speakers"],
                        "importance": ep["importance"],
                        "emotion": ep["emotion"],
                        "emotion_charge": ep["emotion_charge"],
                        "has_commitment": ep["has_commitment"],
                        "has_lie": ep["has_lie"],
                        "occurred_at": day,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                added += 1
                await agent_life_event_service.record_memory(
                    agent_id=agent_id, memory_type="episodic", memory_content=ep["content"],
                    importance=ep["importance"], metadata={"source": "consolidation", "speakers": ep["speakers"]},
                )
            except Exception as e:
                logger.error(f"Write episode failed for {agent_id}: {e}", exc_info=True)

        logger.info(f"Episode consolidation {agent_id}: {len(raw)} records -> {len(episodes)} episodes "
                    f"(added {added}, merged {merged})")
        return {"episodes": len(episodes), "added": added, "merged": merged}

    # ---------------- 内部 ----------------

    async def _extract(self, agent_id: str, raw: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        persona = (await persona_manager.get(agent_id))["system1"]
        prompt = EPISODE_PROMPT.format(persona=persona, records=self._render(raw))
        try:
            text = (await doubao_client.generate(prompt, max_tokens=3000) or "").strip()
            if "{" in text and "}" in text:
                text = text[text.index("{"): text.rindex("}") + 1]
            items = json.loads(text).get("episodes") or []
        except Exception as e:
            logger.error(f"Episode extraction failed for {agent_id}: {e}", exc_info=True)
            return None

        out = []
        for it in items:
            content = (it.get("content") or "").strip()
            if not content:
                continue
            out.append({
                "content": content,
                "speakers": [s for s in (it.get("speakers") or []) if isinstance(s, str) and s not in SELF_SPEAKERS],
                "importance": max(1, min(4, int(it.get("importance") or 2))),
                "emotion": (it.get("emotion") or "").strip(),
                "emotion_charge": max(0.0, min(1.0, float(it.get("emotion_charge") or 0.0))),
                "has_commitment": bool(it.get("has_commitment")),
                "has_lie": bool(it.get("has_lie")),
            })
        return out

    @staticmethod
    def _render(raw: List[Dict[str, Any]]) -> str:
        """渲染原文：带说话人、谎言标记和第一印象标签；相邻两句间隔超过 30 分钟时插一条分隔，帮助按事件切分"""
        lines, last_ts = [], None
        for r in raw:
            ts = r.get("ts")
            if last_ts and ts and ts - last_ts > 1800:
                lines.append(f"—— 过了 {int((ts - last_ts) / 60)} 分钟 ——")
            last_ts = ts or last_ts
            content = r.get("content", "")
            for lie in r.get("lies") or []:
                if lie in content:
                    content = content.replace(lie, f"{lie}[谎言]", 1)
            tag = f"（重要度 {r['importance']}）" if r.get("importance") is not None else ""
            lines.append(f"[{r.get('speaker') or r.get('role', 'user')}] {content}{tag}")
        return "\n".join(lines)

    @staticmethod
    def _day_of(raw: List[Dict[str, Any]]) -> str:
        """这批对话发生的日期（取第一条有时间戳的记录；旧记录没有 ts 时用今天）"""
        ts = next((r["ts"] for r in raw if r.get("ts")), None)
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else datetime.now().strftime("%Y-%m-%d")

    async def _merge_into_existing(self, agent_id: str, ep: Dict[str, Any]) -> bool:
        """
        这件事以前是否已经记过？是则合并进旧经历（不重复存），返回 True。

        只靠向量阈值分不开两类情况：同一件事换个说法余弦约 0.72–0.81，
        而"不同的人、类似的事"（张三离婚 vs 李四离婚）也有 0.73。
        所以和待办去重一样：embedding 粗筛候选（余弦 ≥ EPISODE_MERGE_SIMILARITY），再由 LLM 判断是否同一件事；
        是同一件事时，有新进展就把两条合写成一条（就地改写，created_at 不变），没有新内容就只重激活。
        两种情况都会更新 updated_at，反思据此找到"反复发生的事"
        """
        candidates = await memory_repository.find_similar(
            agent_id, ep["content"], min_score=settings.EPISODE_MERGE_SIMILARITY, limit=3)
        if not candidates:
            return False

        listing = "\n".join(f"[{i + 1}] {text}" for i, (_, _, text) in enumerate(candidates))
        prompt = MERGE_PROMPT.format(new=ep["content"], candidates=listing)
        try:
            text = (await doubao_client.generate(prompt, max_tokens=400) or "").strip()
            if "{" in text and "}" in text:
                text = text[text.index("{"): text.rindex("}") + 1]
            verdict = json.loads(text)
            idx = int(verdict.get("same") or 0)
        except Exception as e:
            logger.warning(f"Episode merge judgment failed for {agent_id}, store as new: {e}")
            return False
        if not 1 <= idx <= len(candidates):
            return False

        memory_id, score, old_text = candidates[idx - 1]
        merged_text = (verdict.get("merged") or "").strip()
        if merged_text and merged_text != old_text:
            ok = await memory_repository.update_text(memory_id, merged_text, metadata={
                "importance": ep["importance"], "emotion": ep["emotion"], "emotion_charge": ep["emotion_charge"],
                "has_commitment": ep["has_commitment"], "has_lie": ep["has_lie"],
            })
            action = "updated"
        else:
            ok = await memory_repository.reactivate_ids([memory_id], memory_type="episodic")
            action = "reactivated"
        if ok:
            logger.info(f"Episode merged ({action}, cos={score:.2f}) for {agent_id}: "
                        f"「{ep['content'][:30]}」→「{(merged_text or old_text)[:30]}」")
        return ok

episode_consolidation_service = EpisodeConsolidationService()
