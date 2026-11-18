"""
Prefrontal Lobe Memory Worker: 每日凌晨的工作记忆整理

职责（每天 PFC_DAILY_HOUR 点，对每个agent）：
1. 读取 wm:{agent_id}:stm 剩余记录 + wm:{agent_id}:stm:compress 全部块
2. 从 stm 原文里整理出经历写入 Episodic（episode_consolidation_service），失败则本次不整理、原文保留
3. 使用 LLM 压缩成简洁的当日摘要，放入 compress 队列
4. 清理：stm 只删除参与整理的那部分，整理期间新到的消息保留
5. 结合当天完整的对话复核处境（_review_situation）：删掉已结束的、重复的、混进来的待办和已经过去的事，
   补上新出现的持续状况；没有结束证据的保留
6. 全部 agent 整理完后触发反思（reflection_worker）：反思要看到今天刚整理、刚重激活的经历

注意：
- 经历在这里（以及滚动压缩前）提取，而不是 Memory Gate 逐句实时写：整理时看得到上下文和"我"说的话
- 已被滚动压缩掉的原文，在压缩前就已经整理过经历，这里只处理 stm 里剩下的原文
- compress 摘要用于工作记忆召回，让 AI 快速了解历史上下文
- Situation 由 Memory Gate 实时更新，不在此处更新
"""
import asyncio
import json
import logging
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from typing import Any, Dict, List

from app.client.doubao_client import doubao_client
from app.client.redis_client import redis_client
from app.config.settings import settings
from app.repository.relationship_repository import relationship_repository
from app.service.prefrontal_lobe_memory import (
    compress_key, format_records, lock_key, prefrontal_lobe_memory, stm_key,
)

logger = logging.getLogger(__name__)


# 当日摘要压缩 Prompt
DAILY_SUMMARY_PROMPT = """你是一个记忆整理助手。

任务：将过去24小时的对话记录压缩成一段简洁的摘要（150字以内）。

要求：
1. 用第一人称视角（"我"指 AI 助手）
2. 按时间顺序概括主要话题和事件
3. 保留关键信息（任务、决策、重要讨论）
4. 忽略寒暄和重复内容
5. 不要提取单独的记忆片段，只输出一段连贯的摘要

日期：{date}

对话记录：
{records}

输出格式：直接输出摘要文本，不要加任何标记或解释。
"""

class PrefrontalLobeMemoryWorker:
    def __init__(self):
        self.lock_ttl = 3600
        self.running = False

    async def consolidate_all_agents(self) -> Dict[str, Dict[str, int]]:
        """对所有agent执行每日整理"""
        agent_ids = await prefrontal_lobe_memory.list_agents()
        if not agent_ids:
            logger.info("No agents to consolidate")
            return {}

        logger.info(f"Daily memory consolidation for {len(agent_ids)} agents")

        results = {}
        for agent_id in agent_ids:
            try:
                results[agent_id] = await self.consolidate_agent(agent_id)
            except Exception as e:
                logger.error(f"Failed daily consolidation for {agent_id}: {e}", exc_info=True)
        return results

    async def consolidate_agent(self, agent_id: str) -> Dict[str, int]:
        """
        单个agent的每日整理
        
        逻辑：
        - 使用 LLM 压缩当日对话成简洁摘要
        - 不再提取 episodes（由 Memory Gate 实时处理）

        Returns:
            {"records": 参与整理条数}
        """
        lock = lock_key(agent_id)
        # 与滚动压缩共用一把锁，避免两者同时改写同一批数据
        if not await redis_client.acquire_lock(lock, self.lock_ttl):
            logger.warning(f"Skip daily consolidation for {agent_id}: lock held")
            return {"records": 0}

        try:
            # 记下stm当前长度作为游标：只整理并删除游标之前的记录
            stm_len = await redis_client.llen(stm_key(agent_id))
            compressed_raw, stm_raw = await redis_client.lrange_many(compress_key(agent_id), stm_key(agent_id))
            stm_raw = stm_raw[:stm_len]

            records = [json.loads(r) for r in compressed_raw] + [json.loads(r) for r in stm_raw]
            if not records:
                return {"records": 0}

            # 原文被压成当日摘要之前，先整理出经历（压缩块里的原文在滚动压缩前已经整理过）
            from app.service.episode_consolidation_service import episode_consolidation_service
            stm_records = [json.loads(r) for r in stm_raw]
            episode_result = await episode_consolidation_service.consolidate(agent_id, stm_records)
            if episode_result is None:
                logger.error(f"Episode consolidation failed for {agent_id}, keep all records")
                return {"records": len(records)}

            # 使用 LLM 生成当日摘要（不提取 episodes）
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            daily_summary = await self._compress_daily_summary(date, records)
            
            if not daily_summary:
                # LLM 失败时不清理，数据留到下一次
                logger.error(f"Daily summary compression failed for {agent_id}, keep all records")
                return {"records": len(records)}

            # 提取昨天所有对话的 speakers（用于统计 dialogue_state）
            yesterday_speakers_set = set()
            for r in records:
                if r.get("type") == "raw" and r.get("speaker"):
                    # raw 记录：直接提取 speaker
                    spk = r["speaker"]
                    if spk not in ("我", "user"):
                        yesterday_speakers_set.add(spk)
                elif r.get("type") == "compress" and "speakers" in r:
                    # compress 块：提取块内的 speakers 列表
                    yesterday_speakers_set.update(r["speakers"])

            # compress 替换为当日摘要一块；stm 删除参与整理的前 stm_len 条
            daily_block = {
                "type": "daily",
                "summary": daily_summary,
                "speakers": list(yesterday_speakers_set),  # 保留昨天对话的所有 speakers
            }
            await redis_client.replace_list(
                compress_key(agent_id), [json.dumps(daily_block, ensure_ascii=False)]
            )
            if stm_len:
                await redis_client.ltrim(stm_key(agent_id), stm_len)

            # 记录今天遇到的新陌生人到人际关系表
            await self._record_new_strangers(agent_id, records, date)

            # 结合今天完整的对话复核一遍处境（失败不影响整理结果，处境保持原样）
            situation_changes = await self._review_situation(agent_id, stm_records, daily_summary)

            logger.info(f"Daily consolidation for {agent_id}: {len(records)} records -> daily summary "
                        f"({len(daily_summary)} chars), episodes {episode_result}, situation changes {situation_changes}")
            return {"records": len(records), **episode_result, "situation_changes": situation_changes}

        finally:
            await redis_client.release_lock(lock)

    async def _review_situation(self, agent_id: str, stm_records: List[Dict[str, Any]], daily_summary: str) -> int:
        """
        每日复核处境（SITUATION_REVIEW_PROMPT）。对话中的处境更新只看一句话，久了会留下已经结束的状况、
        重复的条目、混进来的待办和已经过去的事；这里结合当天完整的对话逐条复核。

        - 过期事件先用代码删掉（日期是确定的，不需要模型判断）
        - 每一条都编号，要求模型逐条给出 keep / delete / rewrite，再由代码组装：
          keep 的原样保留（不会被顺手改写），没有被逐条判断过的东西也不会悄悄留下。
          以前让模型直接输出整份新处境，它常把条目原样抄过去而不真正判断，删除是否生效每次不一样
        - 当天没有对话原文时不调用模型：没有新证据，只靠"很久没提"不能判断一件事结束了
        - 输出不合法时重试一次，仍失败则处境保持原样

        Returns:
            改动条数（delete + rewrite + add + 正在做的事的变化；没有调用或失败时为 0）
        """
        from app.config.prompts import SITUATION_REVIEW_PROMPT
        from app.service.subjectivity_service import subjectivity_service

        try:
            await subjectivity_service.clean_expired_events(agent_id)
            raw = [r for r in stm_records if r.get("type") == "raw"]
            context = await subjectivity_service.get_context(agent_id)
            if not raw or not context or not context.situation:
                return 0
            sit = context.situation

            # 逐条编号：E 事件 / S 状况 / F 关注
            items: Dict[str, Dict[str, Any]] = {}
            for n, e in enumerate(sit.get("time_sensitive_events") or [], 1):
                items[f"E{n}"] = {"type": "E", "text": e.get("description") or e.get("event") or "",
                                  "deadline": e.get("deadline")}
            for n, t in enumerate(sit.get("ongoing_states") or [], 1):
                items[f"S{n}"] = {"type": "S", "text": t}
            for n, t in enumerate(sit.get("attention_focus") or [], 1):
                items[f"F{n}"] = {"type": "F", "text": t}
            if not items:
                return 0
            listing = "\n".join(f"[{k}] {v['text']}" + (f"（{v['deadline']}）" if v.get("deadline") else "")
                                 for k, v in items.items())

            tasks = [t.get("content", "") for t in context.pending_tasks][:20]
            now = datetime.now()
            prompt = SITUATION_REVIEW_PROMPT.format(
                today=now.strftime("%Y-%m-%d"),
                weekday="星期" + "一二三四五六日"[now.weekday()],
                current_activity=sit.get("current_activity") or "（无）",
                items=listing,
                pending_tasks="\n".join(f"- {t}" for t in tasks) or "（无）",
                daily_summary=daily_summary,
                records=format_records(raw)[-6000:],  # 只留最近的部分，控制长度
            )
            # 模型偶尔输出不合法的 JSON（多为文字里夹了英文双引号）：重试一次，仍失败则处境保持原样
            data = None
            for attempt in range(2):
                text = (await doubao_client.generate(prompt, max_tokens=1500) or "").strip()
                if "{" in text and "}" in text:
                    text = text[text.index("{"): text.rindex("}") + 1]
                try:
                    data = json.loads(text)
                    break
                except json.JSONDecodeError as e:
                    logger.warning(f"Situation review for {agent_id}: invalid JSON (attempt {attempt + 1}): {e}")
            if data is None:
                return 0

            decisions = {d.get("id"): d for d in data.get("decisions") or [] if isinstance(d, dict)}
            kept: Dict[str, List[Dict[str, Any]]] = {"E": [], "S": [], "F": []}
            changes: List[str] = []
            for key, item in items.items():
                d = decisions.get(key) or {}
                kind = (d.get("kind") or "situation").lower()
                action = (d.get("action") or "keep").lower()   # 漏判的条目按 keep 处理，不丢数据
                # 先看这一条本身是什么：待办、已经发生完的事不属于处境，直接删（和今天有没有提到无关）。
                # 以前和"没有结束证据就保留"放在同一层判断，模型会以"今天没提到"为由把它们留下
                if kind in ("todo", "past"):
                    changes.append(f"删除 [{key}] {item['text']}（不是处境：{kind}，{d.get('reason', '')}）")
                    continue
                if action == "delete":
                    changes.append(f"删除 [{key}] {item['text']}（{d.get('reason', '')}）")
                    continue
                if action == "rewrite" and (d.get("text") or "").strip():
                    item = {**item, "text": d["text"].strip(),
                            "deadline": (d.get("deadline") or item.get("deadline")) if item["type"] == "E" else None}
                    changes.append(f"改写 [{key}] → {item['text']}（{d.get('reason', '')}）")
                kept[item["type"]].append(item)
            for a in data.get("add") or []:
                typ, text = (a.get("type") or "").upper(), (a.get("text") or "").strip()
                if typ not in kept or not text or (typ == "E" and not a.get("deadline")):
                    continue
                if (a.get("kind") or "situation").lower() != "situation":
                    continue
                # 已有条目改写过一次、又在新增里写了一遍：和已保留的某条高度相似就不再加
                if any(SequenceMatcher(None, text, i["text"]).ratio() >= 0.6 for i in kept[typ]):
                    continue
                kept[typ].append({"type": typ, "text": text, "deadline": a.get("deadline") if typ == "E" else None})
                changes.append(f"新增 [{typ}] {text}（{a.get('reason', '')}）")
            new_activity = (data.get("current_activity") or "").strip()
            if new_activity and new_activity != (sit.get("current_activity") or ""):
                changes.append(f"正在做的事 → {new_activity}")
            if not changes:
                return 0

            await subjectivity_service.replace_situation(agent_id, {
                **sit,
                "current_activity": new_activity or sit.get("current_activity", ""),
                "time_sensitive_events": [{"description": i["text"], "deadline": i["deadline"]} for i in kept["E"]],
                # 状况、关注每类最多 3 条：保留和改写过的在前、新增的在后，超出时丢掉新增的
                "ongoing_states": [i["text"] for i in kept["S"]][:3],
                "attention_focus": [i["text"] for i in kept["F"]][:3],
            })
            logger.info(f"Situation review for {agent_id}: {changes}")
            return len(changes)
        except Exception as e:
            logger.error(f"Situation review failed for {agent_id}, keep as is: {e}", exc_info=True)
            return 0

    async def _compress_daily_summary(self, date: str, records: List[Dict[str, Any]]) -> str:
        """
        使用 LLM 压缩当日对话成简洁摘要
        
        Args:
            date: 日期字符串
            records: 对话记录列表
            
        Returns:
            压缩后的摘要文本，失败返回空字符串
        """
        try:
            prompt = DAILY_SUMMARY_PROMPT.format(
                date=date,
                records=format_records(records)
            )
            
            summary = (await doubao_client.generate(prompt, max_tokens=3000)).strip()
            
            if not summary or len(summary) < 10:
                logger.warning(f"Generated summary too short: {summary}")
                return ""
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to compress daily summary: {e}", exc_info=True)
            return ""

    async def _record_new_strangers(self, agent_id: str, records: List[Dict[str, Any]], date: str) -> None:
        """
        每日压缩时，自动将今天遇到的新陌生人记录到人际关系表

        逻辑：
        1. 从今天的对话记录中提取所有 speaker（排除 "我"/"user"）
        2. 检查每个 speaker 是否已在 relationship 表中
        3. 如果不在，自动添加为"萍水相逢的陌生人，{日期}见过一面"

        效果：
        - 第一次见面：完全陌生人（查不到关系，"一个不认识的人"）
        - 第二次见面：有"见过一面"的记录，略微放松警惕
        - 多次见面后：可手动升级为"经常遇到的熟人"
        """
        try:
            # 提取今天所有对话的 speaker（排除自己和系统）
            today_speakers = {
                r.get("speaker")
                for r in records
                if r.get("speaker")
                and r.get("speaker") not in ("我", "user")
                and r.get("type") == "raw"
            }

            if not today_speakers:
                return

            # 检查每个 speaker 是否已在 relationship 表中
            new_strangers = []
            for speaker in today_speakers:
                existing = await relationship_repository.get_relationship(agent_id, speaker)
                if not existing:
                    new_strangers.append(speaker)

            # 批量添加新陌生人
            for speaker in new_strangers:
                await relationship_repository.add_relationship(
                    agent_id,
                    speaker,
                    f"萍水相逢的陌生人，{date}见过一面，说过几句话"
                )

            if new_strangers:
                logger.info(f"Recorded {len(new_strangers)} new strangers for {agent_id}: {new_strangers}")

        except Exception as e:
            logger.error(f"Failed to record new strangers for {agent_id}: {e}", exc_info=True)



    # ---------------- 定时任务管理 ----------------

    async def start(self):
        """启动每日定时任务（每天凌晨执行压缩）"""
        self.running = True

        while self.running:
            try:
                # 计算距离下次执行的秒数
                now = datetime.now()
                target = now.replace(hour=settings.PFC_DAILY_HOUR, minute=0, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)

                wait_seconds = (target - now).total_seconds()
                logger.info(f"Next consolidation at {target}, waiting {wait_seconds:.0f}s")

                # 等待到目标时间
                await asyncio.sleep(wait_seconds)

                # 执行整理，随后反思：反思要基于今天刚整理、刚重激活的经历，必须排在整理之后
                if self.running:
                    await self.consolidate_all_agents()
                    from app.worker.reflection_worker import reflection_worker
                    await reflection_worker.run_once()

            except Exception as e:
                logger.error(f"Error in prefrontal worker loop: {e}", exc_info=True)
                # 出错后等待1小时再重试
                await asyncio.sleep(3600)

    def stop(self):
        """停止定时任务"""
        self.running = False
        logger.info("PrefrontalLobeMemoryWorker stopped")


prefrontal_lobe_memory_worker = PrefrontalLobeMemoryWorker()
