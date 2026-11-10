"""
Thought Generator Worker: 念头生成器

职责：
- 定期生成Agent的内在念头（间隔见 THOUGHT_GENERATOR_INTERVAL，默认1小时）
- 检测situation变化，生成相应念头
- 永久保留，只保留最近3个
- 不推进世界状态，只生成内在独白

与World Clock的区别：
- World Clock：推进世界（事件发生、deadline到达）
- Thought Generator：生成内心活动（念头、感受）
"""
import asyncio
import logging
from datetime import datetime, timedelta

from app.service.subjectivity_service import subjectivity_service

from app.config.settings import settings

logger = logging.getLogger(__name__)


class ThoughtGeneratorWorker:
    """念头生成器"""

    def __init__(self):
        self.interval = settings.THOUGHT_GENERATOR_INTERVAL  # 秒，见 settings / .env
        self.running = False

    async def start(self):
        """启动念头生成器"""
        self.running = True
        logger.info("Thought Generator Worker started")

        while self.running:
            try:
                await asyncio.sleep(self.interval)

                if self.running:
                    await self.generate_all_agents()

            except Exception as e:
                logger.error(f"Error in thought generator loop: {e}", exc_info=True)
                await asyncio.sleep(3600)  # 出错后1小时再试

    def stop(self):
        """停止念头生成器"""
        self.running = False
        logger.info("Thought Generator Worker stopped")

    async def generate_all_agents(self):
        """为所有Agent生成念头"""
        agent_ids = await subjectivity_service.list_agents()
        logger.info(f"Thought Generator tick: {len(agent_ids)} agents")

        for agent_id in agent_ids:
            try:
                await self.generate_for_agent(agent_id)
            except Exception as e:
                logger.error(f"Failed to generate thought for {agent_id}: {e}", exc_info=True)

    async def generate_for_agent(self, agent_id: str):
        """为单个Agent生成念头"""
        context = await subjectivity_service.get_context(agent_id)
        if not context:
            return
        # todo 念头的生成是否需要结合人物性格？
        # 生成念头
        thought = await subjectivity_service.generate_thought(
            agent_id=agent_id,
            trigger="spontaneous"
        )

        if thought:
            logger.info(f"Generated thought for {agent_id}: {thought.get('thought', '')[:30]}...")

            # 🆕 判断是否需要转为待办
            await subjectivity_service.add_todo(
                agent_id=agent_id,
                content=thought.get("thought", ""),
                source="thought",
                context={
                    "intensity": thought.get("urgency", 0.5),
                    "trigger": "spontaneous"
                }
            )

    async def generate_on_trigger(
        self,
        agent_id: str,
        trigger: str
    ):
        """事件触发时生成念头

        Args:
            agent_id: Agent ID
            trigger: 触发原因 (situation_change/deadline_approaching)
        """
        try:
            thought = await subjectivity_service.generate_thought(
                agent_id=agent_id,
                trigger=trigger
            )

            if thought:
                logger.info(f"Generated thought on trigger '{trigger}' for {agent_id}: {thought.get('thought', '')[:30]}...")

                # 🆕 判断是否需要转为待办
                await subjectivity_service.add_todo(
                    agent_id=agent_id,
                    content=thought.get("thought", ""),
                    source="thought",
                    context={
                        "intensity": thought.get("urgency", 0.5),
                        "trigger": trigger
                    }
                )
        except Exception as e:
            logger.error(f"Failed to generate thought on trigger: {e}", exc_info=True)


# 单例
thought_generator_worker = ThoughtGeneratorWorker()
