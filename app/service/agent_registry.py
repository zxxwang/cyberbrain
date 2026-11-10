"""
Agent Registry: Agent 注册表

职责：
- 管理所有 Agent 的身份信息
- 通过名字/别名解析 Agent ID
- 支持多对一映射（多个名字指向同一个 Agent）
"""

import json
import logging
from typing import Dict, List, Optional

from app.client.redis_client import redis_client

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Agent 注册表"""

    def __init__(self):
        self.redis_key = "agent_registry"

    async def register(self, agent_id: str, profile: Dict) -> bool:
        """
        注册 Agent

        Args:
            agent_id: Agent ID (唯一标识)
            profile: {
                "name": "张三",
                "aliases": ["老张", "张总", "小张"],
                "department": "技术部",
                "role": "工程师",
                "description": "后端工程师"
            }

        Returns:
            是否注册成功
        """
        try:
            await redis_client.hset(
                self.redis_key,
                agent_id,
                json.dumps(profile, ensure_ascii=False)
            )

            logger.info(f"Registered agent: {agent_id} ({profile.get('name')})")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}", exc_info=True)
            return False

    async def get_profile(self, agent_id: str) -> Optional[Dict]:
        """
        获取 Agent 档案

        Args:
            agent_id: Agent ID

        Returns:
            Agent 档案，或 None
        """
        try:
            profile_str = await redis_client.hget(self.redis_key, agent_id)
            if not profile_str:
                return None

            return json.loads(profile_str)

        except Exception as e:
            logger.error(f"Failed to get profile for {agent_id}: {e}", exc_info=True)
            return None

    async def resolve_agent_id(self, name: str) -> Optional[str]:
        """
        通过名字解析 Agent ID

        支持：
        - 精确匹配名字
        - 匹配别名

        Args:
            name: "张三" | "老张" | "张总"

        Returns:
            "zhang_san_001" or None
        """
        try:
            all_agents = await redis_client.hgetall(self.redis_key)

            if not all_agents:
                logger.warning("No agents registered in registry")
                return None

            for agent_id, profile_str in all_agents.items():
                profile = json.loads(profile_str)

                # 精确匹配名字
                if profile.get("name") == name:
                    logger.debug(f"Resolved '{name}' to {agent_id} (exact match)")
                    return agent_id

                # 匹配别名
                aliases = profile.get("aliases", [])
                if name in aliases:
                    logger.debug(f"Resolved '{name}' to {agent_id} (alias match)")
                    return agent_id

            logger.warning(f"No agent found for name: {name}")
            return None

        except Exception as e:
            logger.error(f"Failed to resolve agent ID for {name}: {e}", exc_info=True)
            return None

    async def list_all(self) -> List[Dict]:
        """
        列出所有 Agent

        Returns:
            [
                {
                    "agent_id": "zhang_san_001",
                    "name": "张三",
                    "aliases": ["老张"],
                    "department": "技术部",
                    "role": "工程师"
                },
                ...
            ]
        """
        try:
            all_agents = await redis_client.hgetall(self.redis_key)

            result = []
            for agent_id, profile_str in all_agents.items():
                profile = json.loads(profile_str)
                result.append({
                    "agent_id": agent_id,
                    **profile
                })

            return result

        except Exception as e:
            logger.error(f"Failed to list agents: {e}", exc_info=True)
            return []

    async def unregister(self, agent_id: str) -> bool:
        """
        注销 Agent

        Args:
            agent_id: Agent ID

        Returns:
            是否注销成功
        """
        try:
            deleted = await redis_client.hdel(self.redis_key, agent_id)

            if deleted:
                logger.info(f"Unregistered agent: {agent_id}")
                return True
            else:
                logger.warning(f"Agent not found: {agent_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}", exc_info=True)
            return False


# 单例
agent_registry = AgentRegistry()
