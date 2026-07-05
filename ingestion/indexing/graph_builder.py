"""Builds a knowledge graph from indexed chunks:
LLM extracts (entity, relation, entity) triples → networkx graph on disk
+ entity-name embeddings in Qdrant for query→node linking."""
import asyncio
import json
from pathlib import Path
from uuid import uuid4

import networkx as nx
from qdrant_client import AsyncQdrantClient, models

from config.settings import settings
from generation.llm_client import LLMClient
from ingestion.indexing.embedder import Embedder

GRAPH_PATH = Path("data/graph/knowledge_graph.json")
ENTITY_COLLECTION = "graph_entities"

EXTRACT_PROMPT = """Extract entities and relations from this text for a knowledge graph.
Entities: concepts, methods, models, datasets, metrics, people, organizations.
Relations: short verb phrases (e.g., "uses", "outperforms", "trained_on").

Text:
{text}

Respond ONLY with JSON:
{{"entities": [{{"name": "...", "type": "..."}}],
 "relations": [{{"source": "...", "relation": "...", "target": "..."}}]}}"""


class GraphBuilder:
    def __init__(self):
        self.llm = LLMClient(model=settings.llm_small_model)
        self.embedder = Embedder()
        self.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        self.graph = self._load()
        self.sem = asyncio.Semaphore(8)

    @staticmethod
    def _load() -> nx.MultiDiGraph:
        if GRAPH_PATH.exists():
            return nx.node_link_graph(json.loads(GRAPH_PATH.read_text()),
                                      directed=True, multigraph=True)
        return nx.MultiDiGraph()

    def _save(self) -> None:
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        GRAPH_PATH.write_text(json.dumps(nx.node_link_data(self.graph)))

    async def _ensure_entity_collection(self) -> None:
        if not await self.qdrant.collection_exists(ENTITY_COLLECTION):
            await self.qdrant.create_collection(
                collection_name=ENTITY_COLLECTION,
                vectors_config={"dense": models.VectorParams(
                    size=1024, distance=models.Distance.COSINE)})

    async def _extract(self, text: str, doc_id: str) -> tuple[list, list]:
        async with self.sem:
            out = await self.llm.complete_json(
                EXTRACT_PROMPT.format(text=text[:3000]))
        return out.get("entities", []), out.get("relations", [])

    async def build_from_chunks(self, chunks: list[dict]) -> None:
        """chunks: [{'text': str, 'doc_id': str, 'chunk_id': str}, ...]"""
        await self._ensure_entity_collection()
        results = await asyncio.gather(
            *[self._extract(c["text"], c["doc_id"]) for c in chunks])

        new_entities: set[str] = set()
        for chunk, (entities, relations) in zip(chunks, results):
            for e in entities:
                name = e.get("name", "").strip().lower()
                if not name:
                    continue
                if name not in self.graph:
                    new_entities.add(name)
                self.graph.add_node(name, type=e.get("type", "concept"))
            for r in relations:
                s = r.get("source", "").strip().lower()
                t = r.get("target", "").strip().lower()
                if s and t and r.get("relation"):
                    self.graph.add_edge(
                        s, t, relation=r["relation"],
                        doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"])

        if new_entities:
            names = sorted(new_entities)
            embs = await self.embedder.embed(names)
            await self.qdrant.upsert(
                collection_name=ENTITY_COLLECTION,
                points=[models.PointStruct(
                    id=uuid4().hex, vector={"dense": e["dense"]},
                    payload={"entity": n})
                    for n, e in zip(names, embs)])
        self._save()


async def build_graph_from_index(limit: int = 2000) -> None:
    """CLI: extract graph from every indexed text chunk."""
    from ingestion.indexing.vector_store import VectorStore
    store = VectorStore()
    points, _ = await store.client.scroll(
        collection_name=settings.text_collection, limit=limit, with_payload=True)
    chunks = [{"text": p.payload.get("text", ""),
               "doc_id": p.payload.get("doc_id", "?"),
               "chunk_id": str(p.id)} for p in points]
    builder = GraphBuilder()
    bs = 32
    for i in range(0, len(chunks), bs):
        await builder.build_from_chunks(chunks[i:i + bs])
        print(f"Graph: {builder.graph.number_of_nodes()} nodes, "
              f"{builder.graph.number_of_edges()} edges "
              f"({i + bs}/{len(chunks)} chunks)")


if __name__ == "__main__":
    asyncio.run(build_graph_from_index())
