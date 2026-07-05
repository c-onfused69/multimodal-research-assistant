"""Late-interaction retrieval using ColPali against page screenshots."""
from qdrant_client import AsyncQdrantClient

from config.settings import settings
from ingestion.indexing.visual_embedder import VisualEmbedder
from retrieval.retrievers.hybrid_retriever import RetrievedDoc


class VisualRetriever:
    def __init__(self):
        self.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        self.embedder = VisualEmbedder()

    async def retrieve(self, query: str, top_k: int = 5,
                       user_groups: list[str] = None) -> list[RetrievedDoc]:
        user_groups = user_groups or ["public"]
        query_vecs = await self.embedder.embed_query(query)

        acl_filter = {"must": [{"key": "acl", "match": {"any": user_groups}}]}

        res = await self.qdrant.query_points(
            collection_name=settings.visual_collection,
            query=query_vecs,
            using="colpali",
            limit=top_k,
            query_filter=acl_filter,
            with_payload=True
        )

        return [
            RetrievedDoc(chunk_id=str(hit.id), score=hit.score, payload=hit.payload)
            for hit in res.points
        ]
