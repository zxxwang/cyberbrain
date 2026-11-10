"""
查询改写服务：多角度改写查询以提高召回率
使用LLM生成同义、反义、上下位等视角的查询变体
"""
import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List

from app.client.doubao_client import doubao_client
from app.config.settings import settings

logger = logging.getLogger(__name__)


class QueryRewriteService:
    def __init__(self):
        self._cache = {}
        self._cache_ttl = timedelta(hours=1)

    async def rewrite(self, query: str, num_queries: int = None) -> List[str]:
        """
        多角度改写查询（含原查询）

        Returns:
            长度为num_queries的查询列表，第一个是原查询
        """
        num_queries = num_queries or settings.QUERY_REWRITE_NUM

        # 优化：num_queries=1时直接返回原query，不调用LLM
        if num_queries <= 1:
            return [query]

        cache_key = hashlib.md5(f"{query}:{num_queries}".encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached and (datetime.utcnow() - cached["time"]) < self._cache_ttl:
            return cached["result"]

        try:
            rewritten = await asyncio.wait_for(self._rewrite_llm(query, num_queries), timeout=10.0)
            self._cache[cache_key] = {"result": rewritten, "time": datetime.utcnow()}
            logger.info(f"Rewrote query: {query[:50]} -> {len(rewritten)} variants")
            return rewritten
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Query rewriting failed ({e}), using fallback")
            return self._fallback(query, num_queries)

    async def _rewrite_llm(self, query: str, num_queries: int) -> List[str]:
        prompt = f"""你是查询改写专家。给定一个查询，生成{num_queries - 1}个不同视角的改写版本，以便从记忆库中召回更全面的结果。

                    改写策略：
                    1. 同义改写：换成同义表达
                    2. 反面改写：从否定/对立面问
                    3. 上下位改写：泛化或细化概念
                    4. 隐含改写：挖掘查询背后的真实意图
                    
                    原查询：{query}
                    
                    请输出{num_queries - 1}个改写版本，每行一个，不要编号、不要解释。"""

        response = await doubao_client.generate(prompt, max_tokens=2000)
        variants = [line.strip() for line in response.strip().split("\n") if line.strip()][:num_queries - 1]
        return [query] + variants

    def _fallback(self, query: str, num_queries: int) -> List[str]:
        """降级策略：简单的规则变体"""
        variants = [query]
        if num_queries > 1:
            variants.append(f"关于{query}")
        if num_queries > 2:
            variants.append(f"{query}是什么")
        if num_queries > 3:
            variants.append(f"有关{query}的信息")
        return variants[:num_queries]


query_rewrite_service = QueryRewriteService()
