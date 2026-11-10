"""
Persona Manager: 管理System1、System2和speaking_manner的人格画像

每个agent在Redis存三个字段：
- persona:{agent_id}:system1: System1提示词片段（情绪反应参数）
- persona:{agent_id}:system2: System2提示词片段（认知与表达参数）
- persona:{agent_id}:speaking_manner: System3的说话习惯（用词、句式、语气）

react时取这三个片段拼接到对应的系统提示词里，决定"我"的气质、品性、认知风格、能力边界和说话方式。
"""
import logging
from typing import Dict

from app.client.redis_client import redis_client

logger = logging.getLogger(__name__)

PERSONA_KEY_PREFIX = "persona:"

# 缺省persona：agent没有配置时使用，保证不报错
DEFAULT_SYSTEM1 = """情绪基线平稳，反应性中等，对威胁与奖赏的敏感度均衡。没有明显的核心恐惧或渴望。压力下倾向回避，\
尴尬时有轻微掩饰冲动。"""

DEFAULT_SYSTEM2 = """身份未明，没有特定专业技能。元认知校准中等，遇到不懂的会含糊带过。性格平和，话不多也不少，\
较为诚实但也会说善意的谎。"""

DEFAULT_SPEAKING_MANNER = """说话方式普通，语气平和，不带明显口音或方言。句式简单直接，用词中性，没有特殊的说话习惯。"""


def persona_key(agent_id: str) -> str:
    return f"{PERSONA_KEY_PREFIX}{agent_id}"


class PersonaManager:
    """管理每个agent的人格画像配置"""

    async def get(self, agent_id: str) -> Dict[str, str]:
        """
        获取System1、System2和speaking_manner的persona片段

        Returns:
            {"system1": str, "system2": str, "speaking_manner": str}，未配置时返回缺省描述
        """
        key = persona_key(agent_id)
        values = await redis_client.mget(f"{key}:system1", f"{key}:system2", f"{key}:speaking_manner")
        return {
            "system1": values[0] or DEFAULT_SYSTEM1,
            "system2": values[1] or DEFAULT_SYSTEM2,
            "speaking_manner": values[2] or DEFAULT_SPEAKING_MANNER,
        }

    async def set(self, agent_id: str, system1: str, system2: str, speaking_manner: str = None) -> None:
        """设置System1、System2和speaking_manner的persona片段"""
        key = persona_key(agent_id)
        data = {
            f"{key}:system1": system1.strip(),
            f"{key}:system2": system2.strip(),
        }
        if speaking_manner is not None:
            data[f"{key}:speaking_manner"] = speaking_manner.strip()
        await redis_client.set_many(data)
        logger.info(f"Persona set for {agent_id}: s1={len(system1)} chars, s2={len(system2)} chars, speaking_manner={len(speaking_manner or '')} chars")

    async def delete(self, agent_id: str) -> bool:
        """删除persona配置"""
        key = persona_key(agent_id)
        count = await redis_client.delete(f"{key}:system1", f"{key}:system2", f"{key}:speaking_manner")
        logger.info(f"Persona deleted for {agent_id}: {count} keys")
        return count > 0

    async def exists(self, agent_id: str) -> bool:
        """检查agent是否配置了persona"""
        key = persona_key(agent_id)
        values = await redis_client.mget(f"{key}:system1", f"{key}:system2", f"{key}:speaking_manner")
        return any(values)


persona_manager = PersonaManager()
