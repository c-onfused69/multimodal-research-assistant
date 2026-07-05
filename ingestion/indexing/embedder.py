"""Dense (bge-m3) and Sparse (BM25 / SPLADE) embedding generation."""
from FlagEmbedding import BGEM3FlagModel

from config.settings import settings


class Embedder:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            # bge-m3 outputs both dense and sparse (lexical) representations
            cls._model = BGEM3FlagModel(settings.text_embedding_model, use_fp16=True)
        return cls._model

    async def embed(self, texts: list[str]) -> list[dict]:
        """Returns list of {"dense": [float], "sparse": {str: float}}"""
        if not texts:
            return []
        model = self.get_model()
        # compute_margin=False for standard embedding
        out = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)

        results = []
        for d, s in zip(out["dense_vecs"], out["lexical_weights"]):
            # Convert token IDs in lexical weights to string keys for Qdrant sparse vectors
            sparse_dict = {str(k): float(v) for k, v in s.items()}
            results.append({"dense": d.tolist(), "sparse": sparse_dict})
        return results

    async def embed_query(self, text: str) -> dict:
        res = await self.embed([text])
        return res[0]
