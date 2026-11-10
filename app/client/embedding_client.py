"""Qwen Embedding客户端（OpenAI-compatible API）"""
import logging
import threading
from collections import OrderedDict

from openai import OpenAI

from app.config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    # 查询向量缓存：一次 react 里 brief / episodic / knowledge 三路召回用的是同一句 signal，
    # 只需算一次。embedding 对同一文本是确定的，缓存不需要过期，只限制条数。
    _QUERY_CACHE_SIZE = 512

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
        )
        self._query_cache: "OrderedDict[str, list[float]]" = OrderedDict()
        self._query_cache_lock = threading.Lock()

    def embed_query(self, text: str) -> list[float]:
        """
        检索用的查询向量（带缓存）。

        换行替换为空格，与 mem0 写入记忆时的处理保持一致。
        线程安全：召回都在 asyncio.to_thread 里调用。
        """
        key = text.replace("\n", " ")
        with self._query_cache_lock:
            vec = self._query_cache.get(key)
            if vec is not None:
                self._query_cache.move_to_end(key)
                return vec

        vec = self.embed(key)

        with self._query_cache_lock:
            self._query_cache[key] = vec
            self._query_cache.move_to_end(key)
            while len(self._query_cache) > self._QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return vec

    def embed(self, text: str) -> list[float]:
        return self.client.embeddings.create(
            input=text,
            model=settings.EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_DIMENSION
        ).data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            input=texts,
            model=settings.EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_DIMENSION
        )
        return [item.embedding for item in response.data]


embedding_client = EmbeddingClient()
