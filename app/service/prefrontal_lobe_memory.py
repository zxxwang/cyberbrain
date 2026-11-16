"""
Prefrontal Lobe Memory: 前额叶工作记忆（短时记忆）

结构（每个agent两个Redis list，均为RPUSH追加、时间从旧到新）：
- wm:{agent_id}:stm           原始短时记忆，每条一个JSON
- wm:{agent_id}:stm:compress  压缩块，每块一个JSON（摘要）

每条原文记录带 id 和 ts（unix 秒），但渲染上下文（format_records）时不输出，不增加 token：
- id：Memory Gate 在后台评估完这句话后，按 id 给它打上重要度、情绪等标签，供整理经历时参考
- ts：整理经历时按时间间隔切分对话片段，并给经历标上发生日期

流程：
- append: RPUSH入stm，长度超过PFC_STM_MAX_LEN时异步压缩最早的PFC_STM_COMPRESS_COUNT条
- compress: 先LRANGE只读 -> 从这批原文里整理出经历写入 Episodic -> LLM压缩
  -> 事务内LTRIM stm + RPUSH compress（成功后才删原文；整理经历失败也不压缩，原文保留）
- get_context: 一次pipeline读两个list，compress在前、stm在后拼成时间线
热路径（append/get_context）不调用LLM。
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from app.client.doubao_client import doubao_client
from app.client.redis_client import redis_client
from app.config.prompts import PFC_COMPRESS_PROMPT
from app.config.settings import settings

logger = logging.getLogger(__name__)

# 花括号是Redis Cluster的hash tag，保证同一agent的key落在同一slot，事务可用
KEY_PREFIX = "wm:"


def stm_key(agent_id: str) -> str:
    return f"{KEY_PREFIX}{{{agent_id}}}:stm"


def compress_key(agent_id: str) -> str:
    return f"{KEY_PREFIX}{{{agent_id}}}:stm:compress"


def lock_key(agent_id: str) -> str:
    return f"{KEY_PREFIX}{{{agent_id}}}:lock"


# 按 id 找到 stm 里的那条记录，把标签合并进去（原子操作，不需要锁；找不到说明已被压缩，返回 0）
_TAG_SCRIPT = """
local items = redis.call('LRANGE', KEYS[1], 0, -1)
for i, raw in ipairs(items) do
  local ok, rec = pcall(cjson.decode, raw)
  if ok and type(rec) == 'table' and rec['id'] == ARGV[1] then
    local tags = cjson.decode(ARGV[2])
    for k, v in pairs(tags) do rec[k] = v end
    redis.call('LSET', KEYS[1], i - 1, cjson.encode(rec))
    return 1
  end
end
return 0
"""


def format_records(records: List[Dict[str, Any]]) -> str:
    """把记录渲染成 '[说话人] 内容' 的行（按顺序即时间先后），供System2上下文和LLM压缩/总结使用"""
    lines = []
    for r in records:
        if r.get("type") == "daily":
            lines.append(f"[昨日日程] {r.get('summary', '')}")
        elif r.get("type") == "compress":
            lines.append(f"[摘要] {r.get('summary', '')}")
        else:
            content = _mark_lies(r.get("content", ""), r.get("lies") or [])
            lines.append(f"[{r.get('speaker') or r.get('role', 'user')}] {content}")
    return "\n".join(lines)


def _mark_lies(content: str, lies: List[str]) -> str:
    """把记录里的谎言句子重新标上[谎言]，让System2和LLM压缩/总结知道哪些话是"我"编的"""
    for lie in lies:
        if lie in content:
            content = content.replace(lie, f"{lie}[谎言]", 1)
    return content


class PrefrontalLobeMemory:
    def __init__(self):
        # 持有后台压缩任务的引用，防止被GC回收
        self._tasks: set = set()

    # ---------------- 写入 ----------------

    async def append(
        self,
        agent_id: str,
        content: str,
        role: str = "user",
        speaker: Optional[str] = None,
        lies: Optional[List[str]] = None,
        record_id: Optional[str] = None,
    ) -> int:
        """
        追加一条短时记忆（热路径：一次RPUSH）

        Args:
            lies: content中"我"有意撒谎的句子，渲染上下文时会重新标出，供圆谎和每日巩固识别
            record_id: 记录 id，不传时自动生成；调用方需要稍后给这条记录打标签时传入

        Returns:
            追加后stm的长度
        """
        record = {
            "type": "raw",
            "id": record_id or uuid.uuid4().hex[:12],
            "ts": int(time.time()),
            "role": role,
            "speaker": speaker,
            "content": content,
        }
        if lies:
            record["lies"] = lies
        length = await redis_client.rpush(stm_key(agent_id), json.dumps(record, ensure_ascii=False))

        if length > settings.PFC_STM_MAX_LEN:
            self._spawn(self.compress(agent_id))
        return length

    async def tag(self, agent_id: str, record_id: str, tags: Dict[str, Any]) -> bool:
        """给 stm 里的一条记录合并标签（如 Memory Gate 的重要度、System1 的情绪）。记录已被压缩时返回 False"""
        if not record_id or not tags:
            return False
        try:
            found = await redis_client.client.eval(
                _TAG_SCRIPT, 1, stm_key(agent_id), record_id, json.dumps(tags, ensure_ascii=False))
            return bool(found)
        except Exception as e:
            logger.warning(f"Tag PFC record {record_id} failed for {agent_id}: {e}")
            return False

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ---------------- 压缩 ----------------

    async def compress(self, agent_id: str) -> int:
        """
        压缩最早的compress_count条，直到stm长度不再超过max_len

        加锁防止并发重复压缩同一批；先读后删，LLM失败时原文保留不丢。

        Returns:
            本次压缩产生的块数
        """
        lock = lock_key(agent_id)
        if not await redis_client.acquire_lock(lock, settings.PFC_LOCK_TTL):
            return 0

        blocks = 0
        try:
            while await redis_client.llen(stm_key(agent_id)) > settings.PFC_STM_MAX_LEN:
                raw = await redis_client.lrange(stm_key(agent_id), 0, settings.PFC_STM_COMPRESS_COUNT - 1)
                records = [json.loads(r) for r in raw]
                if not records:
                    break

                # 原文被压成摘要之前，先把其中值得长期记住的经历整理进 Episodic（压缩后细节就没了）
                from app.service.episode_consolidation_service import episode_consolidation_service
                if await episode_consolidation_service.consolidate(agent_id, records) is None:
                    logger.error(f"Episode consolidation failed for {agent_id}, keep raw records uncompressed")
                    break

                summary = await self._summarize(records)
                if not summary:
                    logger.error(f"Compress failed for {agent_id}, keep raw records")
                    break

                # 提取这批对话涉及的所有 speaker（用于统计对话状态）
                speakers_set = {r.get("speaker") for r in records if r.get("speaker")}
                speakers = list(speakers_set)

                block = {
                    "type": "compress",
                    "count": len(records),
                    "summary": summary,
                    "speakers": speakers,  # 记录这批对话涉及的人
                }
                await redis_client.trim_and_push(
                    stm_key(agent_id), len(records),
                    compress_key(agent_id), json.dumps(block, ensure_ascii=False),
                )
                blocks += 1

            await self._merge_blocks(agent_id)
            if blocks:
                logger.info(f"Compressed {blocks} block(s) for {agent_id}")
            return blocks

        except Exception as e:
            logger.error(f"Failed to compress stm for {agent_id}: {e}", exc_info=True)
            return blocks
        finally:
            await redis_client.release_lock(lock)

    async def _merge_blocks(self, agent_id: str) -> None:
        """压缩块超过上限时，把最早的两块合并成一块，保证上下文总长度有界"""
        key = compress_key(agent_id)
        while await redis_client.llen(key) > settings.PFC_COMPRESS_MAX_BLOCKS:
            raw = await redis_client.lrange(key, 0, 1)
            first, second = (json.loads(r) for r in raw)
            summary = await self._summarize([first, second])
            if not summary:
                logger.error(f"Merge blocks failed for {agent_id}, keep blocks")
                return

            # 合并两个块的 speakers 列表（去重）
            merged_speakers_set = set(first.get("speakers", [])) | set(second.get("speakers", []))
            merged_speakers = list(merged_speakers_set)

            merged = {
                "type": "compress",
                "count": first.get("count", 0) + second.get("count", 0),
                "summary": summary,
                "speakers": merged_speakers,  # 保留说话人列表
            }
            # 事务内删掉头部两块并把合并块放回头部，读取方不会看到中间状态
            await redis_client.replace_head(key, 2, json.dumps(merged, ensure_ascii=False))

    async def _summarize(self, records: List[Dict[str, Any]]) -> str:
        try:
            prompt = PFC_COMPRESS_PROMPT.format(records=format_records(records))
            return (await doubao_client.generate(prompt, max_tokens=2000)).strip()
        except Exception as e:
            logger.error(f"LLM summarize failed: {e}", exc_info=True)
            return ""

    # ---------------- 读取 ----------------

    async def get_records(self, agent_id: str) -> List[Dict[str, Any]]:
        """一次pipeline读取两个list，按时间线拼接（compress在前，stm在后）"""
        compressed, raw = await redis_client.lrange_many(compress_key(agent_id), stm_key(agent_id))
        return [json.loads(r) for r in compressed] + [json.loads(r) for r in raw]

    async def get_context(self, agent_id: str) -> str:
        """渲染成可直接放进System2上下文的短期记忆文本"""
        records = await self.get_records(agent_id)
        return format_records(records)

    async def clear(self, agent_id: str) -> None:
        await redis_client.delete(stm_key(agent_id), compress_key(agent_id))

    async def list_agents(self) -> List[str]:
        """从Redis中发现所有有短时记忆的agent"""
        agents = set()
        for key in await redis_client.scan_keys(f"{KEY_PREFIX}{{*}}:stm*"):
            start, end = key.find("{"), key.find("}")
            if start != -1 and end > start:
                agents.add(key[start + 1:end])
        return sorted(agents)


prefrontal_lobe_memory = PrefrontalLobeMemory()
