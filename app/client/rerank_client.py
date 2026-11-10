"""Qwen3-rerank 重排序客户端（阿里云DashScope）"""
import asyncio
import logging
from http import HTTPStatus
from typing import Any, Dict, List, Optional

import dashscope

from app.config.settings import settings

logger = logging.getLogger(__name__)


class RerankClient:
    def __init__(self):
        dashscope.api_key = settings.EMBEDDING_API_KEY

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        对文档进行重排序（返回 index 和 relevance_score）

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_n: 返回top N个结果（None时返回全部）

        Returns:
            [{"index": 原文档索引, "relevance_score": 相关度分数}, ...]
        """
        if not documents:
            return []

        try:
            # dashscope.TextReRank.call是同步HTTP调用，必须放到线程池
            response = await asyncio.to_thread(
                dashscope.TextReRank.call,
                model=settings.RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=top_n or len(documents),
                return_documents=False
            )

            if response.status_code != HTTPStatus.OK:
                logger.error(f"Rerank API error: {response.code} - {response.message}")
                raise Exception(f"Rerank API error: {response.message}")

            results = []
            if hasattr(response, 'output') and hasattr(response.output, 'results'):
                for result in response.output.results:
                    results.append({
                        "index": result.index,
                        "relevance_score": result.relevance_score,
                    })

            return results

        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
            # 降级：返回原始顺序，分数递减
            return [
                {"index": i, "relevance_score": 1.0 - i * 0.05}
                for i in range(min(top_n or len(documents), len(documents)))
            ]


rerank_client = RerankClient()
