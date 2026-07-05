"""Reciprocal Rank Fusion (RRF) combining Dense and Sparse Qdrant results."""
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from config.settings import settings
from ingestion.indexing.embedder import Embedder


@dataclass
class RetrievedDoc:
    chunk_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


def _rrf(dense_hits: list, sparse_hits: list, k: int = 60) -> list[RetrievedDoc]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits):
        scores[str(hit.id)] = scores.get(str(hit.id), 0.0) + 1.0 / (k + rank + 1)
        payloads[str(hit.id)] = hit.payload

    for rank, hit in enumerate(sparse_hits):
        scores[str(hit.id)] = scores.get(str(hit.id), 0.0) + 1.0 / (k + rank + 1)
        payloads[str(hit.id)] = hit.payload

    fused = [
        RetrievedDoc(chunk_id=cid, score=score, payload=payloads[cid])
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


class HybridRetriever:
    def __init__(self):
        self.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        self.embedder = Embedder()

    async def retrieve(self, query: str, top_k: int = 50, user_groups: list[str] = None) -> list[RetrievedDoc]:
        user_groups = user_groups or ["public"]
        emb = await self.embedder.embed_query(query)

        # ACL filter: chunk must allow at least one of the user's groups
        acl_filter = {
            "must": [{"key": "acl", "match": {"any": user_groups}}]
        }

        # Multi-collection search (text + tables)
        collections = [settings.text_collection, settings.table_collection]
        all_dense = []
        all_sparse = []

        for col in collections:
            dense = await self.qdrant.query_points(
                collection_name=col,
                query=emb["dense"],
                using="dense",
                limit=settings.dense_top_k,
                query_filter=acl_filter,
                with_payload=True
            )
            all_dense.extend(dense.points)

            sparse_indices = [int(k) for k in emb["sparse"].keys()]
            sparse_values = list(emb["sparse"].values())
            sparse = await self.qdrant.query_points(
                collection_name=col,
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=settings.sparse_top_k,
                query_filter=acl_filter,
                with_payload=True
            )
            all_sparse.extend(sparse.points)

        return _rrf(all_dense, all_sparse, k=settings.rrf_k)[:top_k]
