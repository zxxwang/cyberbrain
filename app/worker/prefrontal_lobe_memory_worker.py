"""
Prefrontal Lobe Memory Worker: 每日凌晨的工作记忆整理

职责（每天 PFC_DAILY_HOUR 点，对每个agent）：
1. 读取 wm:{agent_id}:stm 剩余记录 + wm:{agent_id}:stm:compress 全部块
2. 使用 LLM 压缩成简洁的当日摘要（不提取 episodes）
3. 将当日摘要放入 compress 队列
4. 清理：stm 只删除参与整理的那部分，整理期间新到的消息保留

注意：
- 使用 LLM 压缩当日对话，生成简洁摘要
- 不再提取和升级 episodes（由 Memory Gate 在 react 时实时处理）
- compress 摘要用于工作记忆召回，让 AI 快速了解历史上下文
- Situation 由 Memory Gate 实时更新，不再在此处更新
"""
import asyncio
import json
import logging
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

# Situation更新 Prompt
SITUATION_UPDATE_PROMPT = """你是一个状态追踪助手。

任务：根据过去24小时的对话，更新"我"当前的状态。

当前状态：
{current_situation}

昨日对话：
{records}

要求：
1. 判断"我"当前的主要活动（current_activity）是否有变化
2. 更新时间敏感事件（time_sensitive_events）：
   - 保留未完成且未过期的事件
   - 添加昨日提到的新事件（必须有明确日期）
   - 格式：{{"description": "事件描述", "deadline": "YYYY-MM-DD", "status": "pending"}}
3. 更新持续状态（ongoing_states）：
   - 例如"最近失恋了"、"正在学习新技术"
   - 没有日期的状态
4. 更新关注焦点（attention_focus）：当前主要关注什么

输出格式：只输出JSON，不要markdown代码块
{{
  "current_activity": "...",
  "time_sensitive_events": [...],
  "ongoing_states": [...],
  "attention_focus": [...]
}}
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

            logger.info(f"Daily consolidation for {agent_id}: {len(records)} records -> daily summary ({len(daily_summary)} chars)")
            return {"records": len(records)}

        finally:
            await redis_client.release_lock(lock)

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

                # 执行压缩
                if self.running:
                    await self.consolidate_all_agents()

            except Exception as e:
                logger.error(f"Error in prefrontal worker loop: {e}", exc_info=True)
                # 出错后等待1小时再重试
                await asyncio.sleep(3600)

    def stop(self):
        """停止定时任务"""
        self.running = False
        logger.info("PrefrontalLobeMemoryWorker stopped")


prefrontal_lobe_memory_worker = PrefrontalLobeMemoryWorker()
