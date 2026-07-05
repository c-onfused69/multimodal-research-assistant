"""Query → linked entities → k-hop subgraph → relation paths as pseudo-chunks.
Answers multi-hop questions ("How does X relate to Z via Y?") that flat
chunk retrieval misses. Degrades gracefully when no graph exists."""
import json
from pathlib import Path

import networkx as nx
from qdrant_client import AsyncQdrantClient

from config.settings import settings
from ingestion.indexing.embedder import Embedder
from ingestion.indexing.graph_builder import ENTITY_COLLECTION, GRAPH_PATH
from retrieval.retrievers.hybrid_retriever import RetrievedDoc


class GraphRetriever:
    def __init__(self):
        self.embedder = Embedder()
        self.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        self._graph: nx.MultiDiGraph | None = None

    def _load_graph(self) -> nx.MultiDiGraph | None:
        if self._graph is None and Path(GRAPH_PATH).exists():
            self._graph = nx.node_link_graph(
                json.loads(Path(GRAPH_PATH).read_text()),
                directed=True, multigraph=True)
        return self._graph

    async def _link_entities(self, query: str, top_n: int = 3) -> list[str]:
        q = await self.embedder.embed_query(query)
        res = await self.qdrant.query_points(
            collection_name=ENTITY_COLLECTION, query=q["dense"], using="dense",
            limit=top_n, score_threshold=0.5, with_payload=True)
        return [p.payload["entity"] for p in res.points]

    async def retrieve(self, query: str, top_k: int = 8,
                       hops: int = 2) -> list[RetrievedDoc]:
        graph = self._load_graph()
        if graph is None:
            return []
        try:
            seeds = await self._link_entities(query)
        except Exception:
            return []
        seeds = [s for s in seeds if s in graph]
        if not seeds:
            return []

        # Collect edges within `hops` of any seed entity
        nodes: set[str] = set(seeds)
        frontier = set(seeds)
        for _ in range(hops):
            nxt: set[str] = set()
            for n in frontier:
                nxt.update(graph.successors(n))
                nxt.update(graph.predecessors(n))
            nodes |= nxt
            frontier = nxt

        docs: list[RetrievedDoc] = []
        seen: set[tuple] = set()
        for u, v, data in graph.edges(nodes, data=True):
            key = (u, data.get("relation"), v)
            if key in seen or (u not in nodes and v not in nodes):
                continue
            seen.add(key)
            triple = f"{u} —[{data.get('relation')}]→ {v}"
            # Seed-adjacent triples rank first
            score = 1.0 if (u in seeds or v in seeds) else 0.5
            docs.append(RetrievedDoc(
                chunk_id=f"graph::{data.get('chunk_id', u + v)}",
                score=score,
                payload={
                    "text": triple,
                    "display_text": f"[Knowledge graph] {triple}",
                    "chunk_type": "graph",
                    "doc_id": data.get("doc_id", "graph"),
                    "filename": "knowledge_graph",
                    "source_chunk_id": data.get("chunk_id"),
                    "acl": ["public"],
                }))
        docs.sort(key=lambda d: d.score, reverse=True)
        return docs[:top_k]
