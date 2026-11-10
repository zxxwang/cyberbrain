"""
Task Reminder Worker: 定时检查并执行待办任务（优化时间真实性）

执行策略：
1. 有 deadline：提前1-2天开始（概率性），deadline 当天必执行
2. 无 deadline：根据优先级延迟 2-72 小时
"""

import asyncio
import logging
from datetime import datetime
from typing import List

from app.service.task_executor import task_executor
from app.service.subjectivity_service import subjectivity_service

from app.config.settings import settings

logger = logging.getLogger(__name__)


class TaskReminderWorker:
    """任务提醒器"""

    def __init__(self):
        self.interval = settings.TASK_REMINDER_INTERVAL  # 秒，见 settings / .env
        self.running = False

    async def start(self):
        """启动任务提醒器"""
        self.running = True
        logger.info("Task Reminder Worker started")

        while self.running:
            try:
                await self.check_and_execute_tasks()
            except Exception as e:
                logger.error(f"Error in task reminder worker: {e}", exc_info=True)

            await asyncio.sleep(self.interval)

    def stop(self):
        """停止任务提醒器"""
        self.running = False
        logger.info("Task Reminder Worker stopped")

    async def check_and_execute_tasks(self):
        """检查并执行所有 Agent 的待办"""
        agent_ids = await subjectivity_service.list_agents()

        if not agent_ids:
            logger.debug("No active agents found")
            return

        logger.info(f"Checking tasks for {len(agent_ids)} agents")

        for agent_id in agent_ids:
            try:
                await self._process_agent_tasks(agent_id)
            except Exception as e:
                logger.error(f"Failed to process tasks for {agent_id}: {e}", exc_info=True)

    async def _process_agent_tasks(self, agent_id: str):
        """处理单个 Agent 的待办"""
        tasks = await subjectivity_service.get_pending_tasks(agent_id, status="pending")

        if not tasks:
            return

        logger.debug(f"Agent {agent_id} has {len(tasks)} pending tasks")

        for task in tasks:
            if self._should_execute(task):
                await self._execute_and_complete(agent_id, task)

    def _should_execute(self, task: dict) -> bool:
        """
        判断任务是否应该执行（考虑时间真实性）

        执行策略：
        1. 有 deadline：提前1-2天开始（随机），deadline当天必执行
        2. 无 deadline：高优先级2-4h，中12-24h，低48-72h
        """
        now = datetime.now()

        try:
            created = datetime.fromisoformat(task["created_at"])
        except (ValueError, TypeError, KeyError):
            logger.warning(f"Invalid created_at for task {task.get('task_id')}")
            return False

        task_age_hours = (now - created).total_seconds() / 3600

        # 策略1: 有 deadline
        deadline = task.get("deadline")
        if deadline:
            try:
                task_date = datetime.fromisoformat(deadline).date()
                days_until = (task_date - now.date()).days

                # 已过期 → 立即执行
                if days_until < 0:
                    logger.info(f"Task overdue: {task['content']}")
                    return True

                # deadline 当天 → 必执行
                if days_until == 0:
                    logger.info(f"Task deadline today: {task['content']}")
                    return True

                # 1-2天内 → 50%概率提前
                if days_until <= 2:
                    import hashlib
                    task_hash = int(hashlib.md5(task["task_id"].encode()).hexdigest()[:8], 16)
                    if (task_hash % 100) < 50 and task_age_hours >= 1:
                        logger.info(f"Task deadline in {days_until}d, early: {task['content']}")
                        return True

                # 3-7天内 → 高优先级30%提前
                if 3 <= days_until <= 7 and task.get("priority", 0.5) > 0.7:
                    import hashlib
                    task_hash = int(hashlib.md5(task["task_id"].encode()).hexdigest()[:8], 16)
                    if (task_hash % 100) < 30 and task_age_hours >= 2:
                        logger.info(f"High priority early: {task['content']}")
                        return True

                return False

            except (ValueError, TypeError):
                logger.warning(f"Invalid deadline: {deadline}")

        # 策略2: 无 deadline（根据优先级）
        priority = task.get("priority", 0.5)

        if priority > 0.7 and task_age_hours >= 2:
            logger.info(f"High priority ready ({task_age_hours:.1f}h): {task['content']}")
            return True
        elif priority > 0.5 and task_age_hours >= 12:
            logger.info(f"Medium priority ready ({task_age_hours:.1f}h): {task['content']}")
            return True
        elif task_age_hours >= 48:
            logger.info(f"Low priority ready ({task_age_hours:.1f}h): {task['content']}")
            return True

        return False

    async def _execute_and_complete(self, agent_id: str, task: dict):
        """执行任务并标记完成"""
        try:
            result = await task_executor.execute_task(agent_id, task)

            if result["success"]:
                await subjectivity_service.complete_task(agent_id, task["task_id"])
                logger.info(f"✅ Task completed: {task['content']} → {result['result']}")
            else:
                logger.warning(f"❌ Task failed: {task['content']}")

        except Exception as e:
            logger.error(f"Failed to execute task {task['task_id']}: {e}", exc_info=True)


# 单例
task_reminder_worker = TaskReminderWorker()
