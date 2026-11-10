"""
人际关系存储库（Redis Hash）

Key:   agent:{agent_id}:relationships
Field: 人名
Value: "关系类型，描述"，如 "直属上级，关系一般，比较严格"
"""
from typing import Dict, Optional

from app.client.redis_client import redis_client


def _key(agent_id: str) -> str:
    return f"agent:{agent_id}:relationships"


class RelationshipRepository:

    async def get_relationship(self, agent_id: str, person_name: str) -> Optional[str]:
        """查询某人与agent的关系描述，不存在返回None（即陌生人）"""
        return await redis_client.client.hget(_key(agent_id), person_name)

    async def add_relationship(self, agent_id: str, person_name: str, relationship: str, description: str = "") -> None:
        """添加/覆盖一条关系"""
        value = f"{relationship}，{description}" if description else relationship
        await redis_client.client.hset(_key(agent_id), person_name, value)

    async def load_all_relationships(self, agent_id: str) -> Dict[str, str]:
        return await redis_client.client.hgetall(_key(agent_id))

    async def delete_relationship(self, agent_id: str, person_name: str) -> bool:
        return await redis_client.client.hdel(_key(agent_id), person_name) > 0

    async def clear_all_relationships(self, agent_id: str) -> None:
        await redis_client.delete(_key(agent_id))


relationship_repository = RelationshipRepository()
