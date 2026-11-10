"""
Knowledge Memory Service: 知识记忆业务逻辑（Semantic Memory - External Temporal Lobe Knowledge）

神经科学映射：外侧颞叶（Lateral Temporal Lobe）存储程序性记忆和专业知识
"""
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from qdrant_client.models import FieldCondition, MatchValue

from app.client.embedding_client import embedding_client
from app.client.qdrant_client import GenericQdrantClient
from app.config.settings import settings

logger = logging.getLogger(__name__)


class KnowledgeType(str, Enum):
    """技能类型枚举"""
    CONCEPT = "concept"                    # 概念："什么是React Hooks"
    HOWTO = "howto"                        # 方法："如何配置Tailwind"
    BEST_PRACTICE = "best_practice"        # 最佳实践："CSS命名规范"
    TROUBLESHOOTING = "troubleshooting"    # 排错："解决Tailwind purge问题"
    EXAMPLE = "example"                    # 示例："Tailwind响应式布局代码"


class KnowledgeSource(str, Enum):
    """技能来源枚举"""
    EXPERIENCE = "experience"      # 工作经验
    LEARNING = "learning"          # 学习获得
    PROJECT = "project"            # 项目实践
    DOCUMENTATION = "documentation"  # 文档记录


class KnowledgeMemoryService:
    """技能记忆服务（使用通用Qdrant客户端）"""

    def __init__(self):
        self.client = GenericQdrantClient(
            collection_name="knowledge_memories",
            dimension=settings.QDRANT_DIMENSION
        )

    async def initialize(self) -> bool:
        """初始化服务"""
        success = await self.client.initialize()
        if success:
            # 创建技能相关的索引
            self.client._create_indexes(["agent_id", "domain", "sub_domain", "knowledge_type"])
        return success

    def add_knowledge(
        self,
        agent_id: str,
        title: str,
        content: str,
        knowledge_type: str,
        domain: str,
        sub_domain: Optional[str] = None,
        proficiency: float = 0.5,
        confidence: float = 0.8,
        source: str = "experience",
        tags: Optional[List[str]] = None,
        related_knowledge: Optional[List[str]] = None,
        prerequisite_knowledge: Optional[List[str]] = None,
        knowledge_id: Optional[str] = None
    ) -> Optional[str]:
        """
        添加技能记忆

        Args:
            agent_id: Agent ID
            title: 技能标题
            content: 技能详细内容
            knowledge_type: 技能类型（concept/howto/best_practice/troubleshooting/example）
            domain: 大领域（如"前端开发"）
            sub_domain: 子领域（如"CSS框架"）
            proficiency: 熟练度 0-1
            confidence: 可信度 0-1
            source: 来源（experience/learning/project/documentation）
            tags: 标签列表
            related_knowledge: 关联技能ID列表
            prerequisite_knowledge: 前置技能ID列表
            knowledge_id: 指定ID（可选，不指定则自动生成）

        Returns:
            技能ID，失败返回None
        """
        try:
            # 验证枚举值
            if knowledge_type not in [t.value for t in KnowledgeType]:
                logger.warning(f"Invalid knowledge_type: {knowledge_type}, using 'concept'")
                knowledge_type = KnowledgeType.CONCEPT.value

            if source not in [s.value for s in KnowledgeSource]:
                logger.warning(f"Invalid source: {source}, using 'experience'")
                source = KnowledgeSource.EXPERIENCE.value

            # 验证数值范围
            proficiency = max(0.0, min(1.0, proficiency))
            confidence = max(0.0, min(1.0, confidence))

            # 生成或使用指定的ID
            if not knowledge_id:
                knowledge_id = str(uuid.uuid4())

            # 生成embedding
            text = f"{title}\n{content}"
            vector = embedding_client.embed(text)

            # 构建payload
            payload = {
                "agent_id": agent_id,
                "title": title,
                "content": content,
                "knowledge_type": knowledge_type,
                "domain": domain,
                "sub_domain": sub_domain or "",
                "proficiency": proficiency,
                "confidence": confidence,
                "source": source,
                "tags": tags or [],
                "related_knowledge": related_knowledge or [],
                "prerequisite_knowledge": prerequisite_knowledge or [],
                "used_count": 0,
                "last_used_at": None,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            # 调用通用客户端写入
            success = self.client.add_point(
                point_id=knowledge_id,
                vector=vector,
                payload=payload
            )

            if success:
                logger.info(f"Added knowledge '{title}' for agent {agent_id}")
                return knowledge_id
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}", exc_info=True)
            return None

    def search_knowledge(
        self,
        agent_id: str,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.5,
        domain: Optional[str] = None,
        knowledge_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索技能记忆

        Args:
            agent_id: Agent ID
            query: 查询文本
            limit: 返回结果数量
            score_threshold: 相似度阈值
            domain: 过滤领域
            knowledge_type: 过滤技能类型

        Returns:
            技能列表，按相似度排序
        """
        try:
            # 生成query向量
            query_vector = embedding_client.embed_query(query)

            # 构建过滤条件
            filter_conditions = [
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            ]

            if domain:
                filter_conditions.append(
                    FieldCondition(key="domain", match=MatchValue(value=domain))
                )

            if knowledge_type:
                filter_conditions.append(
                    FieldCondition(key="knowledge_type", match=MatchValue(value=knowledge_type))
                )

            # 调用通用搜索
            knowledge_list = self.client.search_points(
                query_vector=query_vector,
                filter_conditions=filter_conditions,
                limit=limit,
                score_threshold=score_threshold
            )

            # 按 相似度 * proficiency 重新排序
            knowledge_list.sort(
                key=lambda x: x["score"] * x.get("proficiency", 0.5),
                reverse=True
            )

            return knowledge_list

        except Exception as e:
            logger.error(f"Failed to search knowledge: {e}", exc_info=True)
            return []

    def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """获取单个技能"""
        return self.client.get_by_id(knowledge_id)

    def get_all_knowledge(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取agent的所有技能"""
        filter_conditions = [
            FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
        ]
        knowledge_list = self.client.get_all_by_filter(filter_conditions, limit)

        # 按更新时间倒序
        knowledge_list.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return knowledge_list

    def update_knowledge(
        self,
        knowledge_id: str,
        **kwargs
    ) -> bool:
        """
        更新技能记忆

        支持的字段：title, content, knowledge_type, domain, sub_domain,
        proficiency, confidence, source, tags, related_knowledge, prerequisite_knowledge
        """
        try:
            # 过滤None值
            updates = {k: v for k, v in kwargs.items() if v is not None}

            if not updates:
                logger.warning("No updates provided")
                return False

            # 验证proficiency和confidence范围
            if "proficiency" in updates:
                updates["proficiency"] = max(0.0, min(1.0, updates["proficiency"]))

            if "confidence" in updates:
                updates["confidence"] = max(0.0, min(1.0, updates["confidence"]))

            # 如果更新了title或content，需要重新生成embedding
            if "title" in updates or "content" in updates:
                # 先获取当前数据
                current = self.get_knowledge(knowledge_id)
                if not current:
                    logger.warning(f"Knowledge {knowledge_id} not found for update")
                    return False

                # 合并更新
                title = updates.get("title", current["title"])
                content = updates.get("content", current["content"])

                # 重新生成embedding
                text = f"{title}\n{content}"
                vector = embedding_client.embed(text)

                # 更新payload
                success = self.client.update_payload(knowledge_id, updates)
                if not success:
                    return False

                # 更新向量
                return self.client.update_vector(knowledge_id, vector)
            else:
                # 只更新metadata，不更新向量
                return self.client.update_payload(knowledge_id, updates)

        except Exception as e:
            logger.error(f"Failed to update knowledge {knowledge_id}: {e}", exc_info=True)
            return False

    def delete_knowledge(self, knowledge_id: str) -> bool:
        """删除技能记忆"""
        return self.client.delete_point(knowledge_id)

    def increment_knowledge_usage(self, knowledge_id: str) -> bool:
        """增加技能使用次数（召回时调用）"""
        try:
            current = self.get_knowledge(knowledge_id)
            if not current:
                return False

            updates = {
                "used_count": current.get("used_count", 0) + 1,
                "last_used_at": datetime.utcnow().isoformat()
            }

            return self.client.update_payload(knowledge_id, updates)

        except Exception as e:
            logger.error(f"Failed to increment usage for {knowledge_id}: {e}", exc_info=True)
            return False

    def get_knowledges_by_domain(
        self,
        agent_id: str,
        domain: str
    ) -> List[Dict[str, Any]]:
        """获取指定领域的所有技能"""
        try:
            # 使用get_all_by_filter直接过滤
            filter_conditions = [
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                FieldCondition(key="domain", match=MatchValue(value=domain))
            ]

            knowledge_list = self.client.get_all_by_filter(filter_conditions, limit=100)
            logger.info(f"get_knowledge_by_domain: agent={agent_id}, domain='{domain}', found={len(knowledge_list)} items")

            # 按proficiency排序
            knowledge_list.sort(key=lambda x: x.get("proficiency", 0.0), reverse=True)

            return knowledge_list

        except Exception as e:
            logger.error(f"Failed to get knowledge by domain: {e}", exc_info=True)
            return []

    def get_knowledge_domains(self, agent_id: str) -> List[str]:
        """获取agent的所有技能领域"""
        try:
            knowledge_list = self.get_all_knowledge(agent_id)

            # 提取唯一的domain
            domains = set()
            for knowledge in knowledge_list:
                domain = knowledge.get("domain")
                if domain:
                    domains.add(domain)

            return sorted(list(domains))

        except Exception as e:
            logger.error(f"Failed to get knowledge domains: {e}", exc_info=True)
            return []


# 全局实例
knowledge_memory_service = KnowledgeMemoryService()
