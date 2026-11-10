"""
Agent Life Event Repository: Agent 生命事件数据访问层

职责：
- 封装 agent_life_event 集合的所有数据库操作
- 提供 CRUD 接口
- 管理索引创建
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.client.mongodb_client import mongodb_client
from app.config.settings import settings

logger = logging.getLogger(__name__)


class AgentLifeEventRepository:
    """Agent 生命事件仓储"""

    def __init__(self):
        self.collection_name = settings.MONGODB_COLLECTION_LIFE_EVENT
        self._indexes_created = False

    async def ensure_indexes(self):
        """确保索引已创建（幂等操作）"""
        if self._indexes_created:
            return

        try:
            collection = mongodb_client.get_collection(self.collection_name)

            # 复合索引：agent_id + timestamp（倒序）
            await collection.create_index([
                ("agent_id", 1),
                ("timestamp", -1)
            ], name="agent_time_idx")

            # 单字段索引：event_type
            await collection.create_index("event_type", name="type_idx")

            # 单字段索引：timestamp（倒序）
            await collection.create_index([("timestamp", -1)], name="time_idx")

            self._indexes_created = True
            logger.info(f"Indexes created for collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to create indexes: {e}", exc_info=True)

    async def insert_event(
        self,
        agent_id: str,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        rt: Optional[float] = None,
        trace: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        插入事件

        Args:
            agent_id: Agent ID
            event_type: 事件类型
            data: 事件数据
            metadata: 元数据
            rt: 产生这条事件的耗时（毫秒），react_output / thought / todo 写这个顶层字段
            trace: 中间过程的结构化记录，目前只有 react_output 写（System1/路由/System2/召回/耗时/token）

        Returns:
            插入的文档 ID，失败返回 None
        """
        if not mongodb_client.is_connected:
            logger.warning("MongoDB not connected, skip event recording")
            return None

        try:
            collection = mongodb_client.get_collection(self.collection_name)

            event = {
                "agent_id": agent_id,
                "event_type": event_type,
                "timestamp": datetime.utcnow(),
                "data": data,
                "metadata": metadata or {}
            }
            if rt is not None:
                event["rt"] = round(rt, 1)
            if trace is not None:
                event["trace"] = trace

            result = await collection.insert_one(event)
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Failed to insert event: {e}", exc_info=True)
            return None

    async def find_events(
        self,
        agent_id: str,
        event_types: Optional[List[str]] = None,
        limit: int = 100,
        skip: int = 0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        查询事件

        Args:
            agent_id: Agent ID
            event_types: 事件类型过滤
            limit: 返回数量限制
            skip: 跳过数量
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            事件列表（按时间倒序）
        """
        if not mongodb_client.is_connected:
            logger.warning("MongoDB not connected")
            return []

        try:
            collection = mongodb_client.get_collection(self.collection_name)

            # 构建查询条件
            query = {"agent_id": agent_id}

            if event_types:
                query["event_type"] = {"$in": event_types}

            if start_time or end_time:
                query["timestamp"] = {}
                if start_time:
                    query["timestamp"]["$gte"] = start_time
                if end_time:
                    query["timestamp"]["$lte"] = end_time

            # 查询
            cursor = collection.find(query).sort(
                "timestamp", -1
            ).skip(skip).limit(limit)

            events = await cursor.to_list(length=limit)

            # 转换 ObjectId 为字符串
            for event in events:
                event["_id"] = str(event["_id"])

            return events

        except Exception as e:
            logger.error(f"Failed to find events: {e}", exc_info=True)
            return []

    async def count_events(
        self,
        agent_id: str,
        event_types: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        """
        统计事件数量

        Args:
            agent_id: Agent ID
            event_types: 事件类型过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            事件数量
        """
        if not mongodb_client.is_connected:
            return 0

        try:
            collection = mongodb_client.get_collection(self.collection_name)

            query = {"agent_id": agent_id}

            if event_types:
                query["event_type"] = {"$in": event_types}

            if start_time or end_time:
                query["timestamp"] = {}
                if start_time:
                    query["timestamp"]["$gte"] = start_time
                if end_time:
                    query["timestamp"]["$lte"] = end_time

            count = await collection.count_documents(query)
            return count

        except Exception as e:
            logger.error(f"Failed to count events: {e}", exc_info=True)
            return 0

    async def get_all_agent_ids(self) -> List[str]:
        """获取所有有事件记录的 Agent ID 列表"""
        if not mongodb_client.is_connected:
            return []

        try:
            collection = mongodb_client.get_collection(self.collection_name)
            agent_ids = await collection.distinct("agent_id")
            return agent_ids
        except Exception as e:
            logger.error(f"Failed to get agent IDs: {e}", exc_info=True)
            return []


# 全局单例
agent_life_event_repository = AgentLifeEventRepository()
