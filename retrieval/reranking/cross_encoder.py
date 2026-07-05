"""Re-ranks retrieved candidates using a Cross-Encoder."""
from FlagEmbedding import FlagReranker

from config.settings import settings
from retrieval.retrievers.hybrid_retriever import RetrievedDoc


class CrossEncoderReranker:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = FlagReranker(settings.reranker_model, use_fp16=False)
        return cls._model

    async def rerank(self, query: str, docs: list[RetrievedDoc], top_k: int = 8) -> list[RetrievedDoc]:
        if not docs:
            return []

        model = self.get_model()
        pairs = [[query, d.payload.get("text", "")] for d in docs]
        scores = model.compute_score(pairs)

        # FlagReranker can return a single float if len == 1
        if isinstance(scores, float):
            scores = [scores]

        for doc, score in zip(docs, scores):
            doc.score = float(score)

        docs.sort(key=lambda x: x.score, reverse=True)
        return docs[:top_k]
