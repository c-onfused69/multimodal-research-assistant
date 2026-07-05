"""Semantic cache using Redis. Hits bypass the LLM/Agent entirely."""
import hashlib
import json

import redis.asyncio as aioredis

from config.settings import settings

CACHE_TTL = 3600 * 24  # 24 hours


class SemanticCache:
    def __init__(self):
        self.redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def get(self, query: str) -> dict | None:
        """Exact match check first. (Semantic check could be added here via embeddings)"""
        key = f"cache:q:{self._hash(query)}"
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    async def set(self, query: str, response: dict) -> None:
        key = f"cache:q:{self._hash(query)}"
        try:
            await self.redis.set(key, json.dumps(response), ex=CACHE_TTL)
        except Exception:
            pass
