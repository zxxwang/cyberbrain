"""Redis客户端（前额叶工作记忆的存储层）"""
import logging
from typing import Dict, List, Optional

import redis.asyncio as redis

from app.config.settings import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """
    使用redis.asyncio：同步客户端会阻塞事件循环。
    连接池懒加载，首次调用时建立连接。
    """

    def __init__(self):
        self.client = redis.from_url(settings.REDIS_URL, decode_responses=True)

    async def ping(self) -> bool:
        try:
            return await self.client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

    async def close(self) -> None:
        await self.client.aclose()

    # ---------- list ----------

    async def rpush(self, key: str, *values: str) -> int:
        """追加到尾部，返回追加后的长度"""
        return await self.client.rpush(key, *values)

    async def lpush(self, key: str, *values: str) -> int:
        """插入到头部，返回插入后的长度"""
        return await self.client.lpush(key, *values)

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        return await self.client.lrange(key, start, end)

    async def llen(self, key: str) -> int:
        return await self.client.llen(key)

    async def ltrim(self, key: str, count: int) -> None:
        """删除头部count条"""
        await self.client.ltrim(key, count, -1)

    async def lrange_many(self, *keys: str) -> List[List[str]]:
        """一次往返读取多个list的全部内容"""
        async with self.client.pipeline(transaction=False) as pipe:
            for key in keys:
                pipe.lrange(key, 0, -1)
            return await pipe.execute()

    async def trim_and_push(self, trim_key: str, trim_count: int, push_key: str, *values: str) -> None:
        """
        原子提交：删除trim_key头部trim_count条，同时把values追加到push_key尾部。
        新消息只会追加到尾部，头部下标不变，因此并发写入下LTRIM是安全的。
        """
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.ltrim(trim_key, trim_count, -1)
            if values:
                pipe.rpush(push_key, *values)
            await pipe.execute()

    async def replace_head(self, key: str, count: int, value: str) -> None:
        """原子操作：删除头部count条，再把value插回头部（用于合并最早的若干块）"""
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.ltrim(key, count, -1)
            pipe.lpush(key, value)
            await pipe.execute()

    async def replace_list(self, key: str, values: List[str]) -> None:
        """原子替换整个list"""
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            if values:
                pipe.rpush(key, *values)
            await pipe.execute()

    # ---------- key / lock ----------

    async def delete(self, *keys: str) -> int:
        return await self.client.delete(*keys)

    async def scan_keys(self, pattern: str) -> List[str]:
        return [k async for k in self.client.scan_iter(match=pattern, count=500)]

    async def acquire_lock(self, key: str, ttl: int) -> bool:
        """NX锁，ttl秒后自动过期防止死锁"""
        return bool(await self.client.set(key, "1", nx=True, ex=ttl))

    async def release_lock(self, key: str) -> None:
        await self.client.delete(key)

    async def get(self, key: str) -> Optional[str]:
        return await self.client.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        await self.client.set(key, value, ex=ex)

    async def mget(self, *keys: str) -> List[Optional[str]]:
        """一次往返读取多个string key，不存在的返回None"""
        return await self.client.mget(*keys)

    async def set_many(self, mapping: Dict[str, str]) -> None:
        """事务内写入多个string key，保证一组相关key同时生效"""
        async with self.client.pipeline(transaction=True) as pipe:
            for key, value in mapping.items():
                pipe.set(key, value)
            await pipe.execute()

    # ---------- hash ----------

    async def hset(self, name: str, key: str, value: str) -> int:
        """设置 hash 字段，返回是否新增（1=新增，0=更新）"""
        return await self.client.hset(name, key, value)

    async def hget(self, name: str, key: str) -> Optional[str]:
        """获取 hash 字段"""
        return await self.client.hget(name, key)

    async def hgetall(self, name: str) -> Dict[str, str]:
        """获取整个 hash"""
        return await self.client.hgetall(name)

    async def hdel(self, name: str, *keys: str) -> int:
        """删除 hash 字段，返回删除的数量"""
        return await self.client.hdel(name, *keys)


redis_client = RedisClient()
