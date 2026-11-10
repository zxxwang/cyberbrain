"""
Decay Worker: 周期性衰减episodic记忆的权重，清理低权重记忆

职责：
- 定期扫描所有agent的episodic记忆
- 应用衰减公式：new_weight = old_weight * decay_factor
- 删除低于阈值的记忆

执行时机：上次执行时间记在 Redis（LAST_RUN_KEY），距上次满 DECAY_WORKER_INTERVAL 才执行。
服务重启不会额外触发一次衰减，频繁重启也不会让衰减永远轮不到。
"""
import asyncio
import logging
import time
from typing import List

from app.client.redis_client import redis_client
from app.config.settings import settings
from app.repository.memory_repository import memory_repository

logger = logging.getLogger(__name__)

LAST_RUN_KEY = "worker:decay:last_run"  # 上次衰减执行的 unix 时间戳（秒）


class DecayWorkerService:
    def __init__(self):
        self.running = False

    async def start(self):
        """启动worker"""
        self.running = True
        logger.info(f"DecayWorker started: interval={settings.DECAY_WORKER_INTERVAL}s, factor={settings.DECAY_FACTOR}, "
                    f"min_weight={settings.DECAY_MIN_WEIGHT}")

        while self.running:
            wait = await self._seconds_until_due()
            if wait > 0:
                logger.info(f"DecayWorker: next run in {wait:.0f}s")
                await asyncio.sleep(wait)
                continue  # 醒来后重新计算：期间间隔可能被修改，或其他进程已经执行过

            try:
                await self._decay_all_agents()
            except Exception as e:
                logger.error(f"Error in decay worker loop: {e}", exc_info=True)
            finally:
                # 失败也记录时间，避免出错时立即重试形成死循环
                await redis_client.set(LAST_RUN_KEY, str(time.time()))

    async def _seconds_until_due(self) -> float:
        """距下一次应执行还有多少秒；从未执行过或 Redis 读取失败时返回 0（立即执行）"""
        try:
            last_run = await redis_client.get(LAST_RUN_KEY)
        except Exception as e:
            logger.warning(f"Failed to read {LAST_RUN_KEY}: {e}")
            return 0.0
        if not last_run:
            return 0.0
        return max(0.0, float(last_run) + settings.DECAY_WORKER_INTERVAL - time.time())

    def stop(self):
        """停止worker"""
        self.running = False
        logger.info("DecayWorker stopped")

    async def _decay_all_agents(self):
        """衰减所有agent的记忆"""
        # 发现所有agent
        agent_ids = await self._discover_agents()
        logger.info(f"Decaying memories for {len(agent_ids)} agents")

        for agent_id in agent_ids:
            try:
                await self._decay_agent(agent_id)
            except Exception as e:
                logger.error(f"Failed to decay memories for {agent_id}: {e}", exc_info=True)

    async def _discover_agents(self) -> List[str]:
        """从episodic collection中发现所有agent"""
        try:
            from qdrant_client import models

            client = memory_repository.episodic_client
            if not client:
                return []

            vector_store = client.vector_store
            agent_ids = set()
            offset = None

            while True:
                result = vector_store.client.scroll(
                    collection_name=vector_store.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["user_id"],
                )
                points, offset = result
                for pt in points:
                    user_id = pt.payload.get("user_id")
                    if user_id:
                        agent_ids.add(user_id)
                if offset is None:
                    break

            return list(agent_ids)

        except Exception as e:
            logger.error(f"Failed to discover agents: {e}", exc_info=True)
            return []

    async def _decay_agent(self, agent_id: str):
        """衰减单个agent的所有记忆"""
        memories = await memory_repository.get_all(agent_id, memory_type="episodic")
        if not memories:
            return

        updated_count = 0
        deleted_count = 0

        for mem in memories:
            memory_id = mem.get("id")
            current_weight = mem.get("decay_weight", 1.0)

            # 应用衰减
            new_weight = current_weight * settings.DECAY_FACTOR

            if new_weight < settings.DECAY_MIN_WEIGHT:
                # 删除低权重记忆
                success = await memory_repository.delete(memory_id, memory_type="episodic")
                if success:
                    deleted_count += 1
            else:
                # 更新权重
                success = await memory_repository.update_decay_weight(memory_id, new_weight, memory_type="episodic")
                if success:
                    updated_count += 1

        logger.info(f"Decayed agent {agent_id}: {updated_count} updated, {deleted_count} deleted")


decay_worker = DecayWorkerService()
