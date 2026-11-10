"""
MongoDB Client: 通用的 MongoDB 连接客户端（低耦合）
"""
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure

from app.config.settings import settings

logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    通用 MongoDB 客户端

    职责：
    - 管理 MongoDB 连接
    - 提供数据库访问接口
    - 不关心具体的集合和业务逻辑
    """

    def __init__(self):
        self.uri = settings.MONGODB_URI
        self.db_name = settings.MONGODB_DB_NAME
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self._connected = False

    async def connect(self) -> bool:
        """连接 MongoDB"""
        try:
            self.client = AsyncIOMotorClient(
                self.uri,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50
            )

            # 测试连接
            await self.client.admin.command('ping')

            self.db = self.client[self.db_name]
            self._connected = True

            logger.info(f"MongoDB connected: {self.db_name}")
            return True

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"MongoDB connection error: {e}", exc_info=True)
            self._connected = False
            return False

    async def disconnect(self):
        """断开连接"""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("MongoDB disconnected")

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected

    def get_collection(self, collection_name: str):
        """
        获取集合对象

        Args:
            collection_name: 集合名称

        Returns:
            Collection 对象
        """
        if not self.is_connected:
            raise RuntimeError("MongoDB not connected")

        return self.db[collection_name]


# 全局单例
mongodb_client = MongoDBClient()
