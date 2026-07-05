"""Qdrant integration for late-interaction multivector visual storage."""
import uuid

from qdrant_client import AsyncQdrantClient, models

from config.settings import settings


class VisualStore:
    def __init__(self):
        self.client = AsyncQdrantClient(url=settings.qdrant_url)

    async def setup(self):
        if not await self.client.collection_exists(settings.visual_collection):
            await self.client.create_collection(
                collection_name=settings.visual_collection,
                vectors_config={
                    # ColPali outputs 128-dim vectors per patch
                    "colpali": models.VectorParams(
                        size=128, distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    )
                }
            )

    async def upsert(self, pages: list[dict], multivectors: list[list[list[float]]]):
        if not pages:
            return
        points = []
        for page, vecs in zip(pages, multivectors):
            points.append(models.PointStruct(
                id=uuid.uuid4().hex,
                vector={"colpali": vecs},
                payload=page
            ))
        await self.client.upsert(
            collection_name=settings.visual_collection,
            points=points, wait=False
        )
