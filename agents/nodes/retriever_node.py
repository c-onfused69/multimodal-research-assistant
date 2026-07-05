from retrieval.retrievers.hybrid_retriever import HybridRetriever
from retrieval.retrievers.visual_retriever import VisualRetriever
from retrieval.retrievers.graph_retriever import GraphRetriever
from retrieval.reranking.cross_encoder import CrossEncoderReranker

_hybrid = HybridRetriever()
_visual = VisualRetriever()
_graph = GraphRetriever()
_reranker = CrossEncoderReranker()


def _rank_fuse(lists_of_docs):
    # simple RRF across different retriever outputs
    scores = {}
    payloads = {}
    for docs in lists_of_docs:
        for rank, d in enumerate(docs):
            cid = d.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
            payloads[cid] = d.payload
    
    from retrieval.retrievers.hybrid_retriever import RetrievedDoc
    fused = [RetrievedDoc(chunk_id=cid, score=score, payload=payloads[cid])
             for cid, score in scores.items()]
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


async def run(state: dict) -> dict:
    query = state.get("rewritten_query") or state["query"]
    groups = state.get("user_groups", ["public"])

    text_candidates = await _hybrid.retrieve(query, user_groups=groups)
    visual_candidates = await _visual.retrieve(query, user_groups=groups)

    graph_candidates = []
    if state.get("route") == "graph" or \
       state.get("variant_config", {}).get("always_fuse_graph"):
        graph_candidates = await _graph.retrieve(query)

    fused = _rank_fuse([text_candidates, visual_candidates, graph_candidates])

    top_k = state.get("variant_config", {}).get("rerank_top_k", 8)
    top = await _reranker.rerank(query, fused, top_k=top_k)

    return {
        "documents": [d.payload for d in top],
        "trace_events": state.get("trace_events", []) +
                        [f"Retrieved {len(text_candidates)} text, {len(visual_candidates)} visual, {len(graph_candidates)} graph → reranked to {len(top)}"]
    }
