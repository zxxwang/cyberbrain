"""
Memory Repository: 记忆的持久化层（封装mem0双collection + Qdrant原生操作）

职责：
- 初始化mem0双客户端（brief/episodic）
- 记忆的CRUD（add/search/update/delete）
- BM25稀疏向量补写
- 查询改写 + 混合检索 + Rerank
- 重激活和衰减权重管理
"""
import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.client.embedding_client import embedding_client
from app.client.rerank_client import rerank_client
from app.config.settings import settings

logger = logging.getLogger(__name__)

# mem0抽取指令：强制输出语言与输入一致（否则中文输入会被抽成英文记忆，BM25失效）
EXTRACTION_INSTRUCTIONS = (
    "Language Requirement (CRITICAL): Write every extracted memory in the SAME LANGUAGE "
    "and SCRIPT as the user's input messages. If the input is Chinese, the memory MUST be "
    "written in Simplified Chinese. Do NOT translate into English unless the input is "
    "already in English. Preserve technical terms, proper nouns and brand names as they "
    "appear in the input.\n"
    "Source Restriction: Extract memories ONLY from the New Messages section. "
    "Last k Messages are context for resolving references, never a source of new memories."
)

EPISODIC_EXTRACTION_INSTRUCTIONS = EXTRACTION_INSTRUCTIONS + (
    "\nEpisode Integrity: Each distinct event or experience in New Messages MUST be stored as "
    "exactly ONE memory that keeps its time, place, people, what happened and how the user "
    "felt together. Do NOT split one episode into multiple smaller facts, and do NOT emit "
    "standalone background facts (e.g. 'User owns a dog') that are only mentioned as part "
    "of the episode."
)

# BM25稀疏向量配置（服务端推理，必须用multilingual分词器才能切分中文）
BM25_VECTOR_NAME = "bm25"
BM25_MODEL = "qdrant/bm25"
BM25_OPTIONS = {"tokenizer": "multilingual", "language": "none"}


def _bm25_document(text: str):
    """构造服务端BM25推理用的Document（写入/查询共用）"""
    from qdrant_client import models
    return models.Document(text=text, model=BM25_MODEL, options=BM25_OPTIONS)


def _mem0_search(client, query: str, filters: Dict[str, Any], top_k: int, **kwargs) -> Dict[str, Any]:
    """
    mem0 Memory.search 的唯一入口。

    mem0 2.2.1 的 search(**kwargs) 会静默吞掉未知参数：传 limit 不报错，
    但会按默认 top_k=20 召回。这里显式拒绝 limit，避免再次踩坑。
    """
    if "limit" in kwargs:
        raise TypeError("mem0 search 使用 top_k，不要传 limit")
    return client.search(query=query, filters=filters, top_k=top_k, **kwargs)


def _share_query_embedding(client) -> None:
    """
    让 mem0 检索（memory_action="search"）复用 embedding_client 的查询向量缓存。
    写入（add / update）仍走 mem0 自己的 embedder，不受影响。
    两边模型、维度一致（实测同一文本余弦=1.0）。
    """
    model = client.embedding_model
    original_embed = model.embed

    def embed(text, memory_action=None):
        if memory_action == "search":
            return embedding_client.embed_query(text)
        return original_embed(text, memory_action)

    model.embed = embed


def _ensure_parent_dir(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


class MemoryRepository:
    def __init__(self):
        self.brief_client = None
        self.episodic_client = None
        self.enabled = False

    async def initialize(self) -> bool:
        """初始化mem0双客户端"""
        # 强制豁免 localhost 代理（macOS 系统代理会劫持 localhost 连接，httpx/qdrant-client 不读 httpx patch）
        import os
        old_no_proxy = os.environ.get("NO_PROXY", "")
        os.environ["NO_PROXY"] = "localhost,127.0.0.1," + old_no_proxy

        try:
            from mem0 import Memory
            from mem0.configs.llms.openai import OpenAIConfig
            from app.config.doubao_llm import DoubaoLLM

            llm_config_dict = {
                "provider": "openai",
                "config": {
                    "model": settings.DOUBAO_MODEL_LITE,
                    "api_key": settings.ARK_API_KEY,
                    "openai_base_url": settings.ARK_BASE_URL,
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
            }

            embedder_config_dict = {
                "provider": "openai",
                "config": {
                    "model": settings.EMBEDDING_MODEL,
                    "api_key": settings.EMBEDDING_API_KEY,
                    "openai_base_url": settings.EMBEDDING_BASE_URL,
                },
            }

            vector_store_base = {
                "provider": "qdrant",
                "config": {
                    "url": settings.QDRANT_URL,
                    "embedding_model_dims": settings.QDRANT_DIMENSION,
                },
            }

            # Brief客户端
            self.brief_client = Memory.from_config({
                "vector_store": {**vector_store_base, "config": {
                    **vector_store_base["config"],
                    "collection_name": settings.QDRANT_BRIEF_COLLECTION
                }},
                "llm": llm_config_dict,
                "embedder": embedder_config_dict,
                "version": "v1.1",
                "history_db_path": _ensure_parent_dir(settings.MEM0_BRIEF_HISTORY_DB),
                "custom_instructions": EXTRACTION_INSTRUCTIONS,
            })

            # Episodic客户端
            self.episodic_client = Memory.from_config({
                "vector_store": {**vector_store_base, "config": {
                    **vector_store_base["config"],
                    "collection_name": settings.QDRANT_EPISODIC_COLLECTION
                }},
                "llm": llm_config_dict,
                "embedder": embedder_config_dict,
                "version": "v1.1",
                "history_db_path": _ensure_parent_dir(settings.MEM0_EPISODIC_HISTORY_DB),
                "custom_instructions": EPISODIC_EXTRACTION_INSTRUCTIONS,
            })

            # 替换mem0内部LLM为关闭thinking的DoubaoLLM
            llm_config = OpenAIConfig(
                model=settings.DOUBAO_MODEL_LITE,
                api_key=settings.ARK_API_KEY,
                openai_base_url=settings.ARK_BASE_URL,
                temperature=0.7,
                max_tokens=1000
            )
            llm_config.disable_thinking = True
            custom_llm = DoubaoLLM(config=llm_config)
            self.brief_client.llm = custom_llm
            self.episodic_client.llm = custom_llm

            # mem0 检索时的查询向量走共享缓存（与 brief / knowledge 召回共用同一次 embedding）
            for client in (self.brief_client, self.episodic_client):
                _share_query_embedding(client)

            # 确保BM25稀疏向量开启IDF
            for client in (self.brief_client, self.episodic_client):
                await asyncio.to_thread(self._ensure_bm25_idf, client)

            self.enabled = True
            logger.info(f"MemoryRepository initialized: brief={settings.QDRANT_BRIEF_COLLECTION}, "
                        f"episodic={settings.QDRANT_EPISODIC_COLLECTION}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MemoryRepository: {e}", exc_info=True)
            self.enabled = False
            return False
        finally:
            # 恢复代理设置
            os.environ["NO_PROXY"] = old_no_proxy

    @staticmethod
    def _ensure_bm25_idf(client) -> None:
        """检查collection的bm25稀疏向量配置，缺少IDF时原地开启"""
        from qdrant_client import models

        vector_store = client.vector_store
        name = vector_store.collection_name
        sparse = vector_store.client.get_collection(name).config.params.sparse_vectors or {}
        if BM25_VECTOR_NAME not in sparse:
            logger.error(f"Collection '{name}' has no '{BM25_VECTOR_NAME}' sparse vector, BM25 disabled")
            return
        if sparse[BM25_VECTOR_NAME].modifier != models.Modifier.IDF:
            vector_store.client.update_collection(
                collection_name=name,
                sparse_vectors_config={BM25_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            logger.info(f"Enabled IDF for '{BM25_VECTOR_NAME}' in '{name}'")

    @staticmethod
    def _write_bm25_vectors(client, memories: List[Dict[str, Any]]) -> None:
        """为记忆补写bm25稀疏向量（必须在mem0写入后调用，mem0不会写bm25）"""
        from qdrant_client import models

        points = [
            models.PointVectors(id=m["id"], vector={BM25_VECTOR_NAME: _bm25_document(m["memory"])})
            for m in memories if m.get("id") and m.get("memory")
        ]
        if points:
            vector_store = client.vector_store
            vector_store.client.update_vectors(collection_name=vector_store.collection_name, points=points)

    async def add(
        self,
        agent_id: str,
        messages: List[Dict[str, str]],
        memory_type: str = "episodic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        添加记忆（根据memory_type路由到对应client）

        Returns:
            本次新增的记忆id列表（未抽取出新记忆或失败时为空列表）
        """
        if not self.enabled:
            return []

        client = self.brief_client if memory_type == "brief" else self.episodic_client
        if not client:
            logger.error(f"Client for {memory_type} not initialized")
            return []

        try:
            result = await asyncio.to_thread(
                lambda: client.add(messages=messages, user_id=agent_id, metadata=metadata or {}, infer=False)
            )

            # 🔍 调试：打印 Mem0 返回结果
            logger.info(f"Mem0 add() returned: {result}")

            # mem0 2.2.1返回: {'results': [{'id': '...', 'memory': '...', 'event': 'ADD'}]}
            if isinstance(result, dict):
                added = result.get("results", [result] if result.get("id") else [])
            elif isinstance(result, list):
                added = result
            else:
                added = []
            added = [m for m in added if isinstance(m, dict) and m.get("id")]

            # 补写bm25向量
            try:
                await asyncio.to_thread(self._write_bm25_vectors, client, added)
            except Exception as e:
                logger.error(f"Failed to write bm25 vectors: {e}", exc_info=True)

            memory_ids = [m["id"] for m in added]
            if memory_ids:
                logger.info(f"Added {len(memory_ids)} {memory_type} memories for {agent_id}")
            return memory_ids

        except Exception as e:
            logger.error(f"Failed to add {memory_type} memory: {e}", exc_info=True)
            return []

    async def search_episodic(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        搜索episodic记忆（query rewriting + 混合检索 + 简单去重）

        流程：
        1. Query Rewriting: 改写为多个角度
        2. 并行召回（每个改写查询召回 candidate_size / num_queries 条）
        3. 合并去重（按id）
        4. Rerank: 用qwen3-rerank重排
        5. 返回top_k
        6. Episodic冲突排序：按创建时间倒排（新的排前面）
        """
        if not self.enabled or not self.episodic_client:
            return []

        try:
            # 1. Query rewriting
            from app.service.query_rewrite_service import query_rewrite_service
            rewritten_queries = await query_rewrite_service.rewrite(query, settings.QUERY_REWRITE_NUM)
            per_query_limit = settings.RERANK_CANDIDATE_SIZE // len(rewritten_queries) + 1

            # 2. 并行召回
            async def _search(sub_query: str):
                return await asyncio.to_thread(
                    _mem0_search,
                    self.episodic_client,
                    sub_query,
                    {"user_id": agent_id},
                    per_query_limit,
                )

            results_list = await asyncio.gather(*[_search(q) for q in rewritten_queries])

            # 3. 合并去重
            seen = set()
            candidates = []
            for res in results_list:
                items = res.get("results", []) if isinstance(res, dict) else []
                for item in items:
                    mid = item.get("id")
                    if mid and mid not in seen:
                        seen.add(mid)
                        candidates.append(item)

            if not candidates:
                return []

            # 优化：如果Top1分数很高（> RERANK_SKIP_THRESHOLD），跳过rerank直接返回
            # 按score排序找Top1
            candidates_sorted = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
            top1_score = candidates_sorted[0].get("score", 0.0) if candidates_sorted else 0.0

            if top1_score > settings.RERANK_SKIP_THRESHOLD:
                logger.info(f"Skip rerank: top1_score={top1_score:.4f} > {settings.RERANK_SKIP_THRESHOLD}")
                # 直接返回top_k个候选，添加score作为rerank_score
                result = []
                for c in candidates_sorted[:top_k]:
                    mem = c.copy()
                    mem["rerank_score"] = c.get("score", 0.0)
                    result.append(mem)
                return result

            # 4. Rerank
            documents = [c.get("memory", "") for c in candidates]
            rerank_results = await rerank_client.rerank(query, documents, top_n=top_k)

            # 5. 按rerank顺序返回，过滤低相关（实测 <0.35 基本是不相关的噪声，只会占 System2 上下文）
            reranked = []
            dropped = 0
            for rr in rerank_results:
                idx = rr["index"]
                if idx < len(candidates):
                    if rr["relevance_score"] < settings.EPISODIC_RERANK_MIN_SCORE:
                        dropped += 1
                        continue
                    mem = candidates[idx].copy()
                    mem["rerank_score"] = rr["relevance_score"]
                    reranked.append(mem)

            # 6. Episodic冲突排序：按创建时间倒排（新的排前面）
            # 对于情景记忆，如果相似度接近，新事件优先（"我搬到上海" 比 "我住在北京" 更重要）
            reranked_with_conflict_sort = await self._episodic_conflict_sort(reranked, agent_id)

            logger.info(f"Searched episodic: {len(candidates)} candidates -> {len(reranked_with_conflict_sort)} after rerank+sort "
                        f"(dropped {dropped} below {settings.EPISODIC_RERANK_MIN_SCORE})")
            return reranked_with_conflict_sort

        except Exception as e:
            logger.error(f"Failed to search episodic: {e}", exc_info=True)
            return []

    async def _episodic_conflict_sort(
        self,
        memories: List[Dict[str, Any]],
        agent_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Episodic记忆冲突排序：相似度接近的记忆按创建时间倒排（新的在前）

        场景：
        - "我住在北京" (created_at: 2024-01-01)
        - "我搬到了上海" (created_at: 2024-06-01)
        召回时，上海应该排在北京前面（更新的地址信息）

        逻辑：
        1. 如果两条记忆的 rerank_score 差距小于 margin（分数接近）
        2. 且向量相似度 >= threshold（内容冲突）
        3. 则按 created_at 降序排列（新事件优先）
        """
        if not memories:
            return []

        import numpy as np

        # 使用与brief相同的配置
        margin = settings.BRIEF_RECENCY_TIEBREAK_MARGIN  # 0.05
        threshold = settings.BRIEF_CONFLICT_SIMILARITY    # 0.85

        def _cosine(v1, v2):
            if v1 is None or v2 is None:
                return 0.0
            a = np.array(v1)
            b = np.array(v2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        # 按 rerank_score 降序排列
        adjusted = sorted(memories, key=lambda m: m.get("rerank_score", 0.0), reverse=True)

        # 重新排序：分数接近且冲突的，按 created_at 降序
        final = []
        i = 0
        while i < len(adjusted):
            mem = adjusted[i]
            score = mem.get("rerank_score", 0.0)

            # 找到所有与 mem 分数接近的记忆
            group = [mem]
            j = i + 1
            while j < len(adjusted):
                other = adjusted[j]
                other_score = other.get("rerank_score", 0.0)
                if abs(score - other_score) <= margin:
                    # 检查是否与 group 中任意一个冲突
                    for g in group:
                        sim = _cosine(g.get("_vector"), other.get("_vector"))
                        if sim >= threshold:
                            group.append(other)
                            break
                    j += 1
                else:
                    break

            # group 内按 created_at 降序排列（新事件优先）
            group.sort(key=lambda m: m.get("created_at", ""), reverse=True)
            final.extend(group)
            i += len(group)

        return final

    async def search_brief(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        搜索brief记忆（混合检索 + rerank过滤 + 冲突排序）

        流程：
        1. 混合检索：向量 + BM25，加权融合 （直接面向数据库，而非mem0）
        2. Rerank过滤：qwen3-rerank打分，过滤低相关 （保留相似度评分大于settings.BRIEF_RERANK_MIN_SCORE的）
        3. 冲突排序：分数接近且相关性分数差距很小（settings.BRIEF_RECENCY_TIEBREAK_MARGIN小于 0.05），updated_at新的排前面
        4. 返回top_k
        """
        if not self.enabled or not self.brief_client:
            return []

        try:
            import numpy as np
            from qdrant_client import models

            vector_store = self.brief_client.vector_store
            agent_filter = models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=agent_id))]
            )

            # 1. 混合检索（向量 + BM25）
            # 注意：mem0创建的collection使用匿名向量，不能用named vector的prefetch语法
            # 改为分别搜索然后手动融合
            # embedding 是远程同步调用、qdrant-client 也是同步的，全部放线程池，避免阻塞事件循环
            # （阻塞时与之并行的 episodic / knowledge 任务会被迫排队）

            def _dense_search() -> Any:
                query_vector = embedding_client.embed_query(query)
                return vector_store.client.query_points(
                    collection_name=vector_store.collection_name,
                    query=query_vector,
                    limit=settings.BRIEF_RERANK_CANDIDATES,
                    with_payload=True,
                    with_vectors=True,
                    query_filter=agent_filter,
                    score_threshold=settings.BRIEF_DENSE_SCORE_THRESHOLD,
                )

            def _bm25_search() -> Any:
                return vector_store.client.query_points(
                    collection_name=vector_store.collection_name,
                    query=_bm25_document(query),
                    using=BM25_VECTOR_NAME,
                    limit=settings.BRIEF_RERANK_CANDIDATES,
                    with_payload=True,
                    with_vectors=True,
                    query_filter=agent_filter,
                )

            # 1a/1b. 向量搜索（含 embedding）与 BM25 搜索互不依赖，并行
            vector_result, bm25_result = await asyncio.gather(
                asyncio.to_thread(_dense_search),
                asyncio.to_thread(_bm25_search),
            )

            # 1c. 手动融合（RRF: Reciprocal Rank Fusion）
            # RRF公式: score = sum(1 / (rank + k)), k通常取60
            from collections import defaultdict
            rrf_scores = defaultdict(float)
            k = 60

            for rank, pt in enumerate(vector_result.points):
                rrf_scores[str(pt.id)] += 1.0 / (rank + k)

            for rank, pt in enumerate(bm25_result.points):
                rrf_scores[str(pt.id)] += 1.0 / (rank + k)

            # 合并去重，保留完整的point信息
            seen_ids = set()
            merged_points = []
            for pt in list(vector_result.points) + list(bm25_result.points):
                pt_id = str(pt.id)
                if pt_id not in seen_ids:
                    seen_ids.add(pt_id)
                    merged_points.append((rrf_scores[pt_id], pt))

            # 按RRF分数排序，取top N
            merged_points.sort(key=lambda x: x[0], reverse=True)
            top_points = [pt for _, pt in merged_points[:settings.BRIEF_RERANK_CANDIDATES]]

            candidates = [
                {
                    "id": str(pt.id),
                    "memory": pt.payload.get("data", ""),
                    "score": pt.score,
                    "_vector": pt.vector if isinstance(pt.vector, list) else pt.vector.get(""),  # 匿名向量直接是list
                    **pt.payload,
                }
                for pt in top_points
            ]

            if not candidates:
                return []

            # 2. Rerank过滤
            if settings.BRIEF_RERANK_ENABLED:
                documents = [c["memory"] for c in candidates]
                rerank_results = await rerank_client.rerank(query, documents, top_n=None)

                filtered = []
                for rr in rerank_results:
                    idx = rr["index"]
                    if idx < len(candidates) and rr["relevance_score"] >= settings.BRIEF_RERANK_MIN_SCORE:
                        mem = candidates[idx].copy()
                        mem["rerank_score"] = rr["relevance_score"]
                        filtered.append(mem)
                candidates = filtered

            if not candidates:
                return []

            # 3. 冲突排序（分数接近 + 语义重合 -> 按updated_at新的排前面）
            if settings.BRIEF_RECENCY_TIEBREAK_MARGIN > 0:
                candidates = self._apply_recency_tiebreak(candidates)

            # 删除内部字段，返回top_k
            for c in candidates:
                c.pop("_vector", None)

            return candidates[:top_k]

        except Exception as e:
            logger.error(f"Failed to search brief: {e}", exc_info=True)
            return []

    def _apply_recency_tiebreak(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        冲突排序：分数差≤margin 且 记忆间余弦≥threshold 时，按updated_at把新的排前面
        （解决"住上海"和"搬到杭州"这类同义冲突）
        """
        import numpy as np

        margin = settings.BRIEF_RECENCY_TIEBREAK_MARGIN
        threshold = settings.BRIEF_CONFLICT_SIMILARITY

        def _cosine(v1, v2):
            if not v1 or not v2:
                return 0.0
            a = np.array(v1)
            b = np.array(v2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        # 按rerank_score降序排列
        sorted_memories = sorted(memories, key=lambda m: m.get("rerank_score", 0.0), reverse=True)

        adjusted = []
        for i, mem in enumerate(sorted_memories):
            score = mem.get("rerank_score", 0.0)
            updated = mem.get("updated_at", "")

            # 检查是否有分数接近且语义重合的记忆
            should_reorder = False
            for j in range(i + 1, len(sorted_memories)):
                other = sorted_memories[j]
                other_score = other.get("rerank_score", 0.0)
                if abs(score - other_score) > margin:
                    break  # 分数差太大，后面的更远

                # 计算语义相似度
                sim = _cosine(mem.get("_vector"), other.get("_vector"))
                if sim >= threshold:
                    # 是冲突，比较updated_at
                    other_updated = other.get("updated_at", "")
                    if updated > other_updated:
                        # mem更新，应该排在other前面（已经满足）
                        pass
                    else:
                        # other更新，mem应该排在后面
                        should_reorder = True
                    break

            if not should_reorder:
                adjusted.append(mem)
            else:
                # 暂时跳过，后面会被other推到前面
                adjusted.append(mem)

        # 重新排序：分数接近且冲突的，按updated_at降序
        final = []
        i = 0
        while i < len(adjusted):
            mem = adjusted[i]
            score = mem.get("rerank_score", 0.0)

            # 找到所有与mem分数接近的记忆
            group = [mem]
            j = i + 1
            while j < len(adjusted):
                other = adjusted[j]
                other_score = other.get("rerank_score", 0.0)
                if abs(score - other_score) <= margin:
                    # 检查是否与group中任意一个冲突
                    for g in group:
                        sim = _cosine(g.get("_vector"), other.get("_vector"))
                        if sim >= threshold:
                            group.append(other)
                            break
                    j += 1
                else:
                    break

            # 对group按updated_at降序排序
            group.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
            final.extend(group)
            i += len(group)

        return final

    async def get_all(self, agent_id: str, memory_type: str = "episodic") -> List[Dict[str, Any]]:
        """获取某agent的所有记忆（用于衰减worker）"""
        if not self.enabled:
            return []

        client = self.brief_client if memory_type == "brief" else self.episodic_client
        if not client:
            return []

        try:
            # 用Qdrant scroll获取全量，不受top_k限制
            from qdrant_client import models

            vector_store = client.vector_store
            all_memories = []
            offset = None

            while True:
                result = vector_store.client.scroll(
                    collection_name=vector_store.collection_name,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=agent_id))]
                    ),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                )
                points, offset = result
                for pt in points:
                    all_memories.append({
                        "id": str(pt.id),
                        "memory": pt.payload.get("data", ""),
                        **pt.payload,
                    })
                if offset is None:
                    break

            return all_memories

        except Exception as e:
            logger.error(f"Failed to get all {memory_type} memories: {e}", exc_info=True)
            return []

    async def update_decay_weight(self, memory_id: str, decay_weight: float, memory_type: str = "episodic") -> bool:
        """更新单条记忆的decay_weight（用于衰减worker）"""
        if not self.enabled:
            return False

        client = self.brief_client if memory_type == "brief" else self.episodic_client
        if not client:
            return False

        try:
            vector_store = client.vector_store
            vector_store.client.set_payload(
                collection_name=vector_store.collection_name,
                payload={
                    "decay_weight": decay_weight,
                    "last_decay_update": datetime.utcnow().isoformat(),
                },
                points=[memory_id],
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update decay_weight for {memory_id}: {e}")
            return False

    async def delete(self, memory_id: str, memory_type: str = "episodic") -> bool:
        """删除单条记忆（用于低权重清理）"""
        if not self.enabled:
            return False

        client = self.brief_client if memory_type == "brief" else self.episodic_client
        if not client:
            return False

        try:
            await asyncio.to_thread(client.delete, memory_id=memory_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    async def reactivate_related(
        self,
        agent_id: str,
        messages: List[Dict[str, str]],
        memory_type: str = "episodic",
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        重激活机制：mem0 2.2.1的add只做新增，重复事实会被静默丢弃且不返回事件，
        因此写入前自行检索，与新消息高度相似的已有记忆视为"再次提及"，重置decay_weight

        Args:
            agent_id: agent标识
            messages: 待写入的消息列表
            memory_type: 记忆类型
            threshold: 相似度阈值，默认取 settings.REACTIVATION_THRESHOLD

        Returns:
            被重激活的记忆id列表
        """
        if not self.enabled:
            return []

        client = self.episodic_client if memory_type == "episodic" else self.brief_client
        if not client:
            return []

        # 提取用户消息文本
        texts = [m["content"] for m in messages if m.get("role") == "user" and m.get("content")]
        if not texts:
            return []
        if threshold is None:
            threshold = settings.REACTIVATION_THRESHOLD

        try:
            def _search_and_reactivate() -> List[str]:
                """在线程池中执行搜索和重激活（embedding和qdrant操作都是同步的）"""
                from qdrant_client import models

                # 查询向量走共享缓存：react 时同一句 signal 已经算过，不再额外调用 embedding
                vectors = [embedding_client.embed_query(t) for t in texts]

                hit_ids = set()
                vector_store = client.vector_store

                # 对每条用户消息搜索相似记忆
                for text, vec in zip(texts, vectors):
                    # 只用向量检索（用户原话 vs 抽取事实，BM25关键词可能差异大）
                    search_result = vector_store.client.query_points(
                        collection_name=vector_store.collection_name,
                        query=vec,
                        query_filter=models.Filter(
                            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=agent_id))]
                        ),
                        limit=5,
                        score_threshold=threshold,
                    )

                    # 记录相似度分数用于调优阈值
                    if search_result.points:
                        logger.debug(f"Reactivation similarity scores for '{text[:50]}': "
                                   f"{[f'{pt.id[:8]}={pt.score:.3f}' for pt in search_result.points]}")

                    hit_ids.update(str(pt.id) for pt in search_result.points)

                # 批量重置decay_weight（只改payload，不触发重新embedding）
                if hit_ids:
                    vector_store.client.set_payload(
                        collection_name=vector_store.collection_name,
                        payload={
                            "decay_weight": 1.0,
                            "last_reactivated_at": datetime.utcnow().isoformat(),
                        },
                        points=list(hit_ids),
                    )

                return list(hit_ids)

            # 在executor中执行
            reactivated_ids = await asyncio.to_thread(_search_and_reactivate)

            if reactivated_ids:
                logger.info(f"Reactivated {len(reactivated_ids)} {memory_type} memories for {agent_id}: {reactivated_ids}")

            return reactivated_ids

        except Exception as e:
            logger.error(f"Failed to reactivate related memories: {e}", exc_info=True)
            return []


memory_repository = MemoryRepository()
