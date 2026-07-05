"""Qdrant integration for text chunks and tables (hybrid search)."""
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from config.settings import settings


class VectorStore:
    def __init__(self):
        self.client = AsyncQdrantClient(url=settings.qdrant_url)

    async def setup(self):
        for collection in [settings.text_collection, settings.table_collection]:
            if not await self.client.collection_exists(collection):
                await self.client.create_collection(
                    collection_name=collection,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=1024, distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            modifier=models.Modifier.IDF
                        )
                    }
                )

    async def upsert(self, chunks: list[dict[str, Any]], embeddings: list[dict], collection: str):
        if not chunks:
            return

        points = []
        for chunk, emb in zip(chunks, embeddings):
            sparse_indices = [int(k) for k in emb["sparse"].keys()]
            sparse_values = list(emb["sparse"].values())

            points.append(models.PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    "dense": emb["dense"],
                    "sparse": models.SparseVector(
                        indices=sparse_indices, values=sparse_values
                    )
                },
                payload=chunk,
            ))

        await self.client.upsert(
            collection_name=collection,
            points=points,
            wait=False
        )
