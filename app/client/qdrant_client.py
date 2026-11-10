"""
Generic Qdrant Client: 通用的Qdrant客户端适配器

提供基础的向量数据库操作，不绑定具体业务场景。
任何脑区（海马体、颞叶、前额叶等）都可以复用这个客户端。
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition
)

from app.config.settings import settings

logger = logging.getLogger(__name__)


class GenericQdrantClient:
    """通用的Qdrant客户端，可用于任何collection"""

    def __init__(self, collection_name: str, dimension: int = None):
        self.client = None  # 延迟初始化
        self.collection_name = collection_name
        self.dimension = dimension or settings.QDRANT_DIMENSION
        self._initialized = False

    async def initialize(self) -> bool:
        """初始化collection（在async context中运行同步Qdrant操作）"""
        if self._initialized:
            return True

        try:
            # 在线程池中运行同步初始化
            await asyncio.to_thread(self._init_sync)
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize collection {self.collection_name}: {e}", exc_info=True)
            return False

    def _init_sync(self):
        """同步初始化逻辑"""
        import os
        # 创建client（增加超时到60秒）
        # macOS系统代理会劫持localhost连接，qdrant-client用requests不读httpx patch，强制豁免
        old_no_proxy = os.environ.get("NO_PROXY", "")
        os.environ["NO_PROXY"] = "localhost,127.0.0.1," + old_no_proxy
        try:
            self.client = QdrantClient(url=settings.QDRANT_URL, timeout=60)
        finally:
            os.environ["NO_PROXY"] = old_no_proxy

        # 检查collection是否存在
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            # 创建collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.info(f"Collection {self.collection_name} already exists")

        # 创建索引（非阻塞，失败不影响启动）
        try:
            self._create_indexes()
        except Exception as e:
            logger.warning(f"Failed to create indexes (non-blocking): {e}")

    def _create_indexes(self, fields: List[str] = None):
        """创建索引以加速过滤查询（幂等操作）"""
        if fields is None:
            fields = ["agent_id"]  # 默认只索引agent_id

        for field in fields:
            try:
                # 尝试创建索引，如果已存在会静默失败
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                    wait=False  # 不等待索引完成
                )
                logger.debug(f"Requested index creation for {field}")
            except Exception as e:
                # 索引可能已存在或Qdrant版本不支持，忽略
                logger.debug(f"Skipped index {field}: {e}")

    def add_point(
        self,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any]
    ) -> bool:
        """添加一个点到collection"""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )

            logger.info(f"Added point: {point_id} to {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add point {point_id}: {e}", exc_info=True)
            return False

    def search_points(
        self,
        query_vector: List[float],
        filter_conditions: Optional[List[FieldCondition]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """通用的向量搜索"""
        try:
            # 搜索
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=Filter(must=filter_conditions) if filter_conditions else None,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            results = search_result.points

            # 转换结果
            points = []
            for point in results:
                item = {
                    "id": str(point.id),
                    "score": point.score,
                    **point.payload
                }
                points.append(item)

            logger.debug(f"Vector search in {self.collection_name}: found {len(points)} points")
            return points

        except Exception as e:
            logger.error(f"Failed to search in {self.collection_name}: {e}", exc_info=True)
            return []

    def get_by_id(self, point_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取点"""
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )

            if not points:
                return None

            point = points[0]
            return {
                "id": str(point.id),
                **point.payload
            }

        except Exception as e:
            logger.error(f"Failed to get point {point_id}: {e}", exc_info=True)
            return None

    def get_all_by_filter(
        self,
        filter_conditions: List[FieldCondition],
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """根据过滤条件获取所有点"""
        try:
            # 使用scroll获取所有记录
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=filter_conditions),
                limit=limit
            )

            items = []
            for point in points:
                item = {
                    "id": str(point.id),
                    **point.payload
                }
                items.append(item)

            logger.info(f"Retrieved {len(items)} points from {self.collection_name}")
            return items

        except Exception as e:
            logger.error(f"Failed to get points from {self.collection_name}: {e}", exc_info=True)
            return []

    def update_payload(
        self,
        point_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """更新点的payload"""
        try:
            # 更新updated_at时间戳
            updates["updated_at"] = datetime.utcnow().isoformat()

            self.client.set_payload(
                collection_name=self.collection_name,
                payload=updates,
                points=[point_id]
            )

            logger.info(f"Updated point: {point_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update point {point_id}: {e}", exc_info=True)
            return False

    def update_vector(
        self,
        point_id: str,
        vector: List[float]
    ) -> bool:
        """更新点的向量"""
        try:
            self.client.update_vectors(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector
                    )
                ]
            )

            logger.info(f"Updated vector for point: {point_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update vector for {point_id}: {e}", exc_info=True)
            return False

    def delete_point(self, point_id: str) -> bool:
        """删除点"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id]
            )
            logger.info(f"Deleted point: {point_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete point {point_id}: {e}", exc_info=True)
            return False


# 不再提供全局实例，由各个service根据需要创建自己的实例
