"""Manifest of indexed documents keyed by source URI → content hash.
Fixes the placeholder `_already_indexed` in pipeline.py."""
import redis.asyncio as aioredis

from config.settings import settings

MANIFEST_KEY = "mra:manifest"


class SyncManifest:
    def __init__(self):
        self.redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    async def needs_index(self, source_uri: str, content_hash: str) -> bool:
        existing = await self.redis.hget(MANIFEST_KEY, source_uri)
        return existing != content_hash

    async def mark_indexed(self, source_uri: str, content_hash: str) -> None:
        await self.redis.hset(MANIFEST_KEY, source_uri, content_hash)

    async def indexed_uris(self) -> set[str]:
        return set(await self.redis.hkeys(MANIFEST_KEY))

    async def remove(self, source_uri: str) -> None:
        await self.redis.hdel(MANIFEST_KEY, source_uri)

    async def clear(self) -> None:
        await self.redis.delete(MANIFEST_KEY)
