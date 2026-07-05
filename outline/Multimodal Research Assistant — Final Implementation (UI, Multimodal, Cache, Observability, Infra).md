# Multimodal Research Assistant — Final Implementation

> Continues the Core Implementation drop. Covers the remaining work in the gated order:
> **ColPali visual retrieval → audio transcriber → semantic cache → Langfuse tracing → `ui/` frontend → `infra/`**
> Run `make eval` after each numbered section before advancing.

---

## 0. `pyproject.toml` — additions

```toml
[project.optional-dependencies]
multimodal = [
    "colpali-engine>=0.3.4",
    "torch>=2.3",
    "pypdfium2>=4.30",
    "pillow>=10.3",
    "faster-whisper>=1.0",
]
```

Install: `uv sync --extra multimodal --extra eval --extra dev`

---

## 1. Visual Retrieval (ColPali) — Build Order Step 7a

### `ingestion/parsers/page_renderer.py`

```python
"""Render PDF pages to PIL images for ColPali embedding + UI thumbnails."""
import io

import pypdfium2 as pdfium
from PIL import Image


def render_pdf_pages(pdf_bytes: bytes, scale: float = 2.0,
                     max_pages: int = 200) -> list[tuple[int, Image.Image]]:
    """Returns [(page_number_1_based, PIL.Image), ...]."""
    pdf = pdfium.PdfDocument(pdf_bytes)
    pages: list[tuple[int, Image.Image]] = []
    try:
        for i in range(min(len(pdf), max_pages)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pages.append((i + 1, bitmap.to_pil()))
            page.close()
    finally:
        pdf.close()
    return pages


def thumbnail_bytes(img: Image.Image, width: int = 320) -> bytes:
    ratio = width / img.width
    thumb = img.resize((width, int(img.height * ratio)))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return buf.getvalue()
```

### `ingestion/indexing/visual_embedder.py`

```python
"""ColPali multi-vector page embeddings (late-interaction / MaxSim)."""
import asyncio

import torch
from PIL import Image

from config.settings import settings


class VisualEmbedder:
    _model = None
    _processor = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            from colpali_engine.models import ColPali, ColPaliProcessor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model = ColPali.from_pretrained(
                settings.visual_embedding_model,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map=device,
            ).eval()
            cls._processor = ColPaliProcessor.from_pretrained(
                settings.visual_embedding_model)
        return cls._model, cls._processor

    # ---- pages (documents) ----
    def _embed_images_sync(self, images: list[Image.Image]) -> list[list[list[float]]]:
        model, processor = self._load()
        out: list[list[list[float]]] = []
        bs = 4
        for i in range(0, len(images), bs):
            batch = processor.process_images(images[i:i + bs]).to(model.device)
            with torch.no_grad():
                embs = model(**batch)
            out.extend(e.cpu().float().tolist() for e in embs)
        return out

    async def embed_pages(self, images: list[Image.Image]) -> list[list[list[float]]]:
        return await asyncio.to_thread(self._embed_images_sync, images)

    # ---- queries ----
    def _embed_query_sync(self, query: str) -> list[list[float]]:
        model, processor = self._load()
        batch = processor.process_queries([query]).to(model.device)
        with torch.no_grad():
            embs = model(**batch)
        return embs[0].cpu().float().tolist()

    async def embed_query(self, query: str) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_query_sync, query)
```

### `ingestion/indexing/visual_store.py`

```python
"""Qdrant multivector collection for ColPali page embeddings (MaxSim)."""
import base64
from uuid import uuid4

from qdrant_client import AsyncQdrantClient, models

from config.settings import settings


class VisualStore:
    def __init__(self):
        self.client = AsyncQdrantClient(url=settings.qdrant_url)

    async def ensure_collection(self, name: str | None = None, dim: int = 128) -> None:
        name = name or settings.visual_collection
        if await self.client.collection_exists(name):
            return
        await self.client.create_collection(
            collection_name=name,
            vectors_config={
                "colpali": models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM),
                ),
            },
        )

    async def upsert_pages(self, doc_id: str, filename: str,
                           pages: list[dict], acl: list[str]) -> None:
        """pages: [{'page': int, 'embedding': multivec, 'page_text': str,
                    'thumbnail': bytes}, ...]"""
        points = []
        for p in pages:
            points.append(models.PointStruct(
                id=uuid4().hex,
                vector={"colpali": p["embedding"]},
                payload={
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": p["page"],
                    "chunk_type": "visual_page",
                    "text": p.get("page_text", ""),        # for cross-encoder rerank
                    "display_text": f"[Page {p['page']} of {filename}] "
                                    + p.get("page_text", "")[:800],
                    "thumbnail_b64": base64.b64encode(p["thumbnail"]).decode(),
                    "acl": acl,
                },
            ))
        await self.client.upsert(
            collection_name=settings.visual_collection, points=points)

    async def delete_doc(self, doc_id: str) -> None:
        await self.client.delete(
            collection_name=settings.visual_collection,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=doc_id))])))

    async def search(self, query_multivec: list[list[float]], top_k: int,
                     user_groups: list[str]) -> list:
        res = await self.client.query_points(
            collection_name=settings.visual_collection,
            query=query_multivec, using="colpali", limit=top_k,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="acl", match=models.MatchAny(any=user_groups))]),
            with_payload=True)
        return res.points
```

### `retrieval/retrievers/visual_retriever.py`

```python
"""ColPali visual page retrieval — answers figure/layout/scan questions
that text chunking misses."""
from config.settings import settings
from ingestion.indexing.visual_embedder import VisualEmbedder
from ingestion.indexing.visual_store import VisualStore
from retrieval.retrievers.hybrid_retriever import RetrievedDoc


class VisualRetriever:
    def __init__(self):
        self.embedder = VisualEmbedder()
        self.store = VisualStore()

    async def retrieve(self, query: str, user_groups: list[str] | None = None,
                       top_k: int = 5) -> list[RetrievedDoc]:
        user_groups = user_groups or ["public"]
        try:
            qvec = await self.embedder.embed_query(query)
            points = await self.store.search(qvec, top_k, user_groups)
        except Exception:
            return []   # visual index optional — degrade gracefully
        return [RetrievedDoc(str(p.id), float(p.score), p.payload) for p in points]
```

### `agents/nodes/retriever_node.py` — updated (text + visual fusion)

```python
"""Hybrid text retrieval + ColPali visual retrieval, fused at rank level
(RRF is rank-based, so heterogeneous scorers fuse cleanly)."""
from retrieval.reranking.cross_encoder import CrossEncoderReranker
from retrieval.retrievers.hybrid_retriever import HybridRetriever, RetrievedDoc
from retrieval.retrievers.visual_retriever import VisualRetriever

_retriever = HybridRetriever()
_visual = VisualRetriever()
_reranker = CrossEncoderReranker()


def _rank_fuse(lists: list[list[RetrievedDoc]], k: int = 60) -> list[RetrievedDoc]:
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievedDoc] = {}
    for docs in lists:
        for rank, doc in enumerate(docs):
            scores[doc.chunk_id] = scores.get(doc.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            by_id[doc.chunk_id] = doc
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [by_id[cid] for cid, _ in ranked]


async def run(state: dict) -> dict:
    query = state.get("rewritten_query") or state["query"]
    groups = state.get("user_groups", ["public"])

    text_candidates = await _retriever.retrieve(query, user_groups=groups)
    visual_candidates = await _visual.retrieve(query, user_groups=groups)

    fused = _rank_fuse([text_candidates, visual_candidates])
    top = await _reranker.rerank(query, fused)

    docs = []
    for d in top:
        payload = dict(d.payload)
        payload.pop("thumbnail_b64", None)          # keep LLM context lean
        docs.append({"chunk_id": d.chunk_id, "score": d.score,
                     "thumbnail_available": "thumbnail_b64" in d.payload,
                     **payload})
    return {
        "documents": docs,
        "trace_events": state.get("trace_events", []) +
                        [f"Retrieved {len(text_candidates)} text + "
                         f"{len(visual_candidates)} visual → reranked to {len(docs)}"],
    }
```

---

## 2. Audio Transcription — Build Order Step 7b

### `ingestion/parsers/audio_transcriber.py`

```python
"""Whisper (faster-whisper) transcription with timestamps."""
import asyncio
from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    start: float          # seconds
    end: float
    text: str


class AudioTranscriber:
    _model = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            from faster_whisper import WhisperModel
            cls._model = WhisperModel("large-v3", compute_type="auto")
        return cls._model

    def _transcribe_sync(self, path: str) -> list[TranscriptSegment]:
        model = self._load()
        segments, _info = model.transcribe(path, vad_filter=True)
        return [TranscriptSegment(s.start, s.end, s.text.strip())
                for s in segments if s.text.strip()]

    async def transcribe(self, path: str) -> list[TranscriptSegment]:
        return await asyncio.to_thread(self._transcribe_sync, path)
```

### `ingestion/chunking/transcript_chunker.py`

```python
"""Groups transcript segments into token-budgeted chunks; timestamps are
kept in metadata and rendered as [MM:SS] markers for citations."""
from config.settings import settings
from ingestion.chunking.base import Chunk, Chunker
from ingestion.parsers.audio_transcriber import TranscriptSegment


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class TranscriptChunker(Chunker):
    async def chunk(self, doc_id: str, elements: list[TranscriptSegment],
                    **kwargs) -> list[Chunk]:
        target = settings.chunk_size_tokens * 4          # chars-per-token heuristic
        acl = kwargs.get("acl", ["public"])
        chunks: list[Chunk] = []
        buf: list[TranscriptSegment] = []
        size = 0

        def flush():
            if not buf:
                return
            text = " ".join(s.text for s in buf)
            display = f"[{_ts(buf[0].start)}–{_ts(buf[-1].end)}] {text}"
            chunks.append(Chunk(
                doc_id=doc_id, chunk_type="transcript",
                text=text, display_text=display, acl=list(acl),
                metadata={"start_sec": buf[0].start, "end_sec": buf[-1].end}))

        for seg in elements:
            if size + len(seg.text) > target and buf:
                flush()
                buf, size = [], 0
            buf.append(seg)
            size += len(seg.text)
        flush()
        return chunks
```

### `ingestion/pipeline.py` — updated (full replacement)

```python
"""Orchestrates: connect → parse → chunk → embed → index.
Now handles: PDF/DOCX (text+tables+figures+visual pages) and audio."""
import argparse
import asyncio
import logging

from config.settings import settings
from ingestion.chunking.contextual_chunker import ContextualChunker
from ingestion.chunking.layout_chunker import LayoutChunker
from ingestion.chunking.transcript_chunker import TranscriptChunker
from ingestion.connectors.local_files import LocalFilesConnector
from ingestion.indexing.embedder import Embedder
from ingestion.indexing.vector_store import VectorStore
from ingestion.indexing.visual_embedder import VisualEmbedder
from ingestion.indexing.visual_store import VisualStore
from ingestion.parsers.audio_transcriber import AudioTranscriber
from ingestion.parsers.docling_parser import DoclingParser
from ingestion.parsers.page_renderer import render_pdf_pages, thumbnail_bytes

log = logging.getLogger("ingestion")


class IngestionPipeline:
    def __init__(self, source: str, index_visual: bool = True):
        self.connector = LocalFilesConnector(source)
        self.parser = DoclingParser()
        self.text_chunker = ContextualChunker()
        self.layout_chunker = LayoutChunker()
        self.transcript_chunker = TranscriptChunker()
        self.transcriber = AudioTranscriber()
        self.embedder = Embedder()
        self.store = VectorStore()
        self.index_visual = index_visual
        if index_visual:
            self.visual_embedder = VisualEmbedder()
            self.visual_store = VisualStore()

    async def run(self) -> None:
        await self.store.ensure_collection(settings.text_collection)
        await self.store.ensure_collection(settings.table_collection)
        if self.index_visual:
            await self.visual_store.ensure_collection()

        uris = await self.connector.list_documents()
        log.info("Found %d documents", len(uris))
        for uri in uris:
            try:
                await self._process(uri)
            except Exception:
                log.exception("Failed: %s", uri)

    async def _process(self, uri: str) -> None:
        doc = await self.connector.fetch(uri)

        if doc.content_type == "audio":
            segments = await self.transcriber.transcribe(doc.source_uri)
            chunks = await self.transcript_chunker.chunk(
                doc.doc_id, segments, acl=doc.acl)
            for c in chunks:
                c.metadata["filename"] = doc.metadata.get("filename")
            await self._index(settings.text_collection, doc.doc_id, chunks)
            log.info("Indexed audio %s: %d transcript chunks",
                     doc.metadata.get("filename"), len(chunks))
            return

        if doc.content_type not in ("pdf", "docx", "pptx"):
            log.warning("Unsupported: %s", doc.content_type)
            return

        elements = await asyncio.to_thread(self.parser.parse, doc)
        text_chunks = await self.text_chunker.chunk(doc.doc_id, elements, acl=doc.acl)
        layout_chunks = await self.layout_chunker.chunk(doc.doc_id, elements, acl=doc.acl)

        figures = [c for c in layout_chunks if c.chunk_type == "figure"]
        tables = [c for c in layout_chunks if c.chunk_type == "table"]
        for c in text_chunks + layout_chunks:
            c.metadata["filename"] = doc.metadata.get("filename")

        await self._index(settings.text_collection, doc.doc_id, text_chunks + figures)
        await self._index(settings.table_collection, doc.doc_id, tables)

        if self.index_visual and doc.content_type == "pdf":
            await self._index_visual_pages(doc, elements)

        log.info("Indexed %s: %d text, %d figures, %d tables",
                 doc.metadata.get("filename"),
                 len(text_chunks), len(figures), len(tables))

    async def _index(self, collection: str, doc_id: str, chunks: list) -> None:
        if not chunks:
            return
        embeddings = await self.embedder.embed([c.text for c in chunks])
        await self.store.delete_doc(collection, doc_id)      # idempotent re-index
        await self.store.upsert_chunks(collection, chunks, embeddings)

    async def _index_visual_pages(self, doc, elements) -> None:
        pages = await asyncio.to_thread(render_pdf_pages, doc.raw_bytes)
        if not pages:
            return
        # Per-page extracted text (payload for cross-encoder + display)
        page_text: dict[int, list[str]] = {}
        for el in elements:
            if el.element_type == "text" and el.page:
                page_text.setdefault(el.page, []).append(el.content)

        images = [img for _, img in pages]
        embeddings = await self.visual_embedder.embed_pages(images)
        payloads = [{
            "page": num,
            "embedding": emb,
            "page_text": " ".join(page_text.get(num, []))[:2000],
            "thumbnail": thumbnail_bytes(img),
        } for (num, img), emb in zip(pages, embeddings)]

        await self.visual_store.delete_doc(doc.doc_id)
        await self.visual_store.upsert_pages(
            doc.doc_id, doc.metadata.get("filename", "?"), payloads, doc.acl)
        log.info("Indexed %d visual pages for %s",
                 len(payloads), doc.metadata.get("filename"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--no-visual", action="store_true")
    args = ap.parse_args()
    asyncio.run(IngestionPipeline(args.source, index_visual=not args.no_visual).run())
```

**Gate:** figure/scan questions in `questions.jsonl` (type=`figure`) must now pass `make eval` before moving on.

---

## 3. Semantic Cache — `api/caching.py`

```python
"""Two-tier cache:
  1. Redis exact-match (hash of normalized query + ACL scope) — sub-ms hits
  2. Qdrant semantic match (bge-m3 dense, cosine ≥ threshold) — near-duplicate hits
ACL groups are part of the cache scope so users never see cached answers
they aren't permitted to see. Only high-confidence, grounded, non-escalated
answers are cached."""
import hashlib
import json
import time
from uuid import uuid4

import redis.asyncio as aioredis
from qdrant_client import AsyncQdrantClient, models

from config.settings import settings
from ingestion.indexing.embedder import Embedder

CACHE_COLLECTION = "semantic_cache"
SIM_THRESHOLD = 0.95
TTL_SECONDS = 60 * 60 * 24          # 24h — corpus freshness bound
MIN_CONFIDENCE_TO_CACHE = 0.75


def _scope(user_groups: list[str]) -> str:
    return ",".join(sorted(user_groups))


def _exact_key(query: str, scope: str) -> str:
    h = hashlib.sha256(f"{scope}::{query.strip().lower()}".encode()).hexdigest()
    return f"mra:cache:{h}"


class SemanticCache:
    def __init__(self):
        self.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        self.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        self.embedder = Embedder()
        self._ready = False

    async def _ensure(self) -> None:
        if self._ready:
            return
        if not await self.qdrant.collection_exists(CACHE_COLLECTION):
            await self.qdrant.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config={"dense": models.VectorParams(
                    size=1024, distance=models.Distance.COSINE)})
        self._ready = True

    async def get(self, query: str, user_groups: list[str]) -> dict | None:
        await self._ensure()
        scope = _scope(user_groups)

        # Tier 1: exact
        hit = await self.redis.get(_exact_key(query, scope))
        if hit:
            return json.loads(hit)

        # Tier 2: semantic
        q = await self.embedder.embed_query(query)
        res = await self.qdrant.query_points(
            collection_name=CACHE_COLLECTION, query=q["dense"], using="dense",
            limit=1, score_threshold=SIM_THRESHOLD,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="scope", match=models.MatchValue(value=scope))]),
            with_payload=True)
        if not res.points:
            return None
        p = res.points[0]
        if time.time() - p.payload.get("created_at", 0) > TTL_SECONDS:
            return None
        return json.loads(p.payload["answer_json"])

    async def set(self, query: str, user_groups: list[str], answer: dict) -> None:
        if answer.get("escalated") or \
           answer.get("confidence", 0.0) < MIN_CONFIDENCE_TO_CACHE:
            return
        await self._ensure()
        scope = _scope(user_groups)
        blob = json.dumps(answer)

        await self.redis.set(_exact_key(query, scope), blob, ex=TTL_SECONDS)
        q = await self.embedder.embed_query(query)
        await self.qdrant.upsert(
            collection_name=CACHE_COLLECTION,
            points=[models.PointStruct(
                id=uuid4().hex,
                vector={"dense": q["dense"]},
                payload={"scope": scope, "query": query,
                         "answer_json": blob, "created_at": time.time()})])

    async def invalidate_all(self) -> None:
        """Call after re-indexing the corpus."""
        await self._ensure()
        await self.qdrant.delete_collection(CACHE_COLLECTION)
        self._ready = False
        async for key in self.redis.scan_iter("mra:cache:*"):
            await self.redis.delete(key)


semantic_cache = SemanticCache()
```

---

## 4. Observability — Langfuse Tracing

### `observability/tracing.py`

```python
"""Langfuse tracing: one trace per /chat request, one span per graph node,
plus generation-level token/cost capture. Also exposes score() for feedback."""
import functools
import time
from typing import Any, Awaitable, Callable

from langfuse import Langfuse

from config.settings import settings

_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    global _langfuse
    if _langfuse is None and settings.langfuse_public_key:
        _langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host)
    return _langfuse


class RequestTrace:
    """Lightweight wrapper: no-ops cleanly when Langfuse isn't configured."""

    def __init__(self, name: str, user_id: str | None = None,
                 metadata: dict | None = None):
        lf = get_langfuse()
        self.trace = lf.trace(name=name, user_id=user_id,
                              metadata=metadata or {}) if lf else None

    @property
    def trace_id(self) -> str | None:
        return self.trace.id if self.trace else None

    def span(self, name: str, input: Any = None):
        return _Span(self.trace.span(name=name, input=input)) if self.trace else _Span(None)

    def update(self, **kwargs) -> None:
        if self.trace:
            self.trace.update(**kwargs)


class _Span:
    def __init__(self, span):
        self._span = span
        self._t0 = time.time()

    def end(self, output: Any = None) -> None:
        if self._span:
            self._span.end(output=output,
                           metadata={"duration_s": round(time.time() - self._t0, 3)})


def score_trace(trace_id: str, name: str, value: float, comment: str = "") -> None:
    lf = get_langfuse()
    if lf and trace_id:
        lf.score(trace_id=trace_id, name=name, value=value, comment=comment)


def traced_node(name: str):
    """Decorator for LangGraph node functions: emits a span per invocation
    when the state carries a `_trace` (RequestTrace) reference."""
    def deco(fn: Callable[[dict], Awaitable[dict]]):
        @functools.wraps(fn)
        async def wrapper(state: dict) -> dict:
            trace: RequestTrace | None = state.get("_trace")
            span = trace.span(name, input={"query": state.get("query")}) if trace else None
            result = await fn(state)
            if span:
                span.end(output={k: v for k, v in result.items()
                                 if k in ("route", "confidence", "is_grounded",
                                          "retry_count", "trace_events")})
            return result
        return wrapper
    return deco


def flush() -> None:
    lf = get_langfuse()
    if lf:
        lf.flush()
```

### `agents/graph.py` — wrap nodes (only the changed lines)

```python
# imports: add
from observability.tracing import traced_node

# in build_graph(), wrap each node:
    g.add_node("router", traced_node("router")(router.run))
    g.add_node("retrieve", traced_node("retrieve")(retriever_node.run))
    g.add_node("grade", traced_node("grade")(grader_node.run))
    g.add_node("rewrite", traced_node("rewrite")(rewrite_node.run))
    g.add_node("generate", traced_node("generate")(generator_node.run))
    g.add_node("reflect", traced_node("reflect")(reflection_node.run))
    g.add_node("escalate", traced_node("escalate")(escalation_node.run))
```

Also add to `agents/state.py`'s `AgentState`:

```python
    _trace: object          # RequestTrace (excluded from serialization concerns)
```

### `observability/online_eval.py`

```python
"""Daily LLM-as-judge faithfulness audit on sampled production traces."""
import asyncio
import random

from generation.llm_client import LLMClient
from observability.tracing import get_langfuse

JUDGE = """Sources:\n{context}\n\nAnswer:\n{answer}\n
Is every factual claim in the answer supported by the sources?
Respond ONLY with JSON: {{"faithful": bool}}"""


async def run(sample_size: int = 25) -> None:
    lf = get_langfuse()
    if not lf:
        print("Langfuse not configured.")
        return
    traces = lf.fetch_traces(limit=200).data
    sample = random.sample(traces, min(sample_size, len(traces)))
    judge = LLMClient()
    for t in sample:
        out = (t.output or {}) if isinstance(t.output, dict) else {}
        context, answer = out.get("context", ""), out.get("answer", "")
        if not context or not answer:
            continue
        verdict = await judge.complete_json(
            JUDGE.format(context=context[:12000], answer=answer))
        lf.score(trace_id=t.id, name="online_faithfulness",
                 value=1.0 if verdict.get("faithful") else 0.0)
    lf.flush()
    print(f"Scored {len(sample)} traces.")


if __name__ == "__main__":
    asyncio.run(run())
```

### `api/routers/feedback.py` — updated (push scores to Langfuse)

```python
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from observability.tracing import score_trace

router = APIRouter()
_feedback_store: list[dict] = []      # production: Postgres


class Feedback(BaseModel):
    trace_id: str | None = None
    query: str
    answer: str
    rating: int          # 1 = 👍, -1 = 👎
    comment: str = ""


@router.post("/feedback")
async def submit_feedback(fb: Feedback):
    _feedback_store.append({**fb.model_dump(), "ts": datetime.utcnow().isoformat()})
    if fb.trace_id:
        score_trace(fb.trace_id, "user_feedback",
                    1.0 if fb.rating > 0 else 0.0, comment=fb.comment)
    # 👎 → triage queue → golden dataset candidates
    return {"status": "recorded", "queued_for_triage": fb.rating < 0}
```

---

## 5. `api/routers/chat.py` — updated (cache + tracing + token streaming)

```python
"""POST /chat — SSE: trace events → streamed tokens → final answer w/ citations.
Adds: semantic cache lookup/write, Langfuse request trace, trace_id in response."""
import asyncio
import json
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agents.graph import agent_graph
from api.caching import semantic_cache
from guardrails.document_rails import sanitize_documents
from guardrails.input_rails import check_input
from guardrails.output_rails import check_output
from observability.tracing import RequestTrace, flush

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    user_groups: list[str] = ["public"]


def _token_chunks(text: str, size: int = 3):
    """Yield word groups so the UI streams even though the graph is atomic."""
    words = re.findall(r"\S+\s*", text)
    for i in range(0, len(words), size):
        yield "".join(words[i:i + size])


@router.post("/chat")
async def chat(req: ChatRequest):
    rail = check_input(req.message)
    if not rail.allowed:
        raise HTTPException(status_code=400, detail=f"Request blocked: {rail.reason}")

    async def event_stream():
        trace = RequestTrace("chat", metadata={"groups": req.user_groups})

        # ---- Cache fast path ----
        cached = await semantic_cache.get(req.message, req.user_groups)
        if cached:
            yield {"event": "trace", "data": json.dumps({"step": "Cache hit ⚡"})}
            for tok in _token_chunks(cached["text"]):
                yield {"event": "token", "data": json.dumps({"t": tok})}
                await asyncio.sleep(0)
            cached["trace_id"] = trace.trace_id
            cached["cached"] = True
            yield {"event": "answer", "data": json.dumps(cached)}
            yield {"event": "done", "data": "{}"}
            trace.update(output={"cached": True, "answer": cached["text"]})
            flush()
            return

        # ---- Full agentic run ----
        state = {"query": req.message, "history": req.history,
                 "user_groups": req.user_groups, "_trace": trace}
        final: dict = {}
        async for event in agent_graph.astream(state, stream_mode="updates"):
            for node_name, update in event.items():
                for step in update.get("trace_events", []):
                    yield {"event": "trace", "data": json.dumps({"step": step})}
                if node_name == "grade" and update.get("documents"):
                    clean, flagged = sanitize_documents(update["documents"])
                    if flagged:
                        update["documents"] = clean
                        yield {"event": "trace", "data": json.dumps(
                            {"step": f"⚠ Dropped {len(flagged)} suspicious chunks"})}
                final.update(update)

        answer_text, violations = check_output(final.get("answer", ""))

        for tok in _token_chunks(answer_text):
            yield {"event": "token", "data": json.dumps({"t": tok})}
            await asyncio.sleep(0)

        payload = {
            "text": answer_text,
            "citations": final.get("citations", []),
            "confidence": final.get("confidence", 0.0),
            "escalated": final.get("needs_escalation", False),
            "pii_masked": bool(violations),
            "trace_id": trace.trace_id,
            "cached": False,
        }
        yield {"event": "answer", "data": json.dumps(payload)}
        yield {"event": "done", "data": "{}"}

        await semantic_cache.set(req.message, req.user_groups, payload)
        trace.update(output={"answer": answer_text,
                             "context": final.get("context", "")[:20000],
                             "confidence": payload["confidence"]})
        flush()

    return EventSourceResponse(event_stream())
```

---

## 6. UI — Next.js Frontend (`ui/`)

### `ui/package.json`

```json
{
  "name": "mra-ui",
  "private": true,
  "scripts": {
    "dev": "next dev -p 3001",
    "build": "next build",
    "start": "next start -p 3001"
  },
  "dependencies": {
    "next": "^15.0.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "typescript": "^5.5"
  }
}
```

### `ui/lib/api.ts`

```typescript
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Citation {
  index: number; chunk_id: string; doc_id?: string;
  source: string; page?: number; chunk_type: string; score: number;
}
export interface AnswerPayload {
  text: string; citations: Citation[]; confidence: number;
  escalated: boolean; pii_masked: boolean; trace_id?: string; cached?: boolean;
}
export interface StreamHandlers {
  onTrace: (step: string) => void;
  onToken: (token: string) => void;
  onAnswer: (a: AnswerPayload) => void;
  onError: (msg: string) => void;
}

/** POST /chat and parse the SSE stream (fetch-based; EventSource can't POST). */
export async function streamChat(
  message: string, history: { role: string; content: string }[],
  h: StreamHandlers, signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
    signal,
  });
  if (!res.ok || !res.body) {
    h.onError(`Request failed (${res.status})`);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      try {
        const parsed = JSON.parse(data);
        if (event === "trace") h.onTrace(parsed.step);
        else if (event === "token") h.onToken(parsed.t);
        else if (event === "answer") h.onAnswer(parsed as AnswerPayload);
      } catch { /* ignore malformed frames */ }
    }
  }
}

export async function fetchSource(chunkId: string) {
  const res = await fetch(`${API_URL}/sources/${chunkId}`);
  if (!res.ok) throw new Error("Source not found");
  return res.json();
}

export async function sendFeedback(fb: {
  trace_id?: string; query: string; answer: string; rating: number; comment?: string;
}) {
  await fetch(`${API_URL}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(fb),
  });
}
```

### `ui/app/layout.tsx`

```tsx
import "./globals.css";
import type { ReactNode } from "react";

export const metadata = { title: "Research Assistant" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
```

### `ui/app/globals.css`

```css
:root {
  --bg: #0f1117; --panel: #171a23; --border: #2a2f3d;
  --text: #e6e8ee; --muted: #8b91a3; --accent: #6d8dff; --ok: #3ecf8e;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font-family: system-ui, -apple-system, sans-serif; }
.app { display: flex; height: 100vh; }
.chat-pane { flex: 1; display: flex; flex-direction: column; max-width: 860px; margin: 0 auto; }
.messages { flex: 1; overflow-y: auto; padding: 24px 16px; }
.msg { margin-bottom: 20px; line-height: 1.6; }
.msg.user .bubble { background: var(--accent); color: #fff; margin-left: auto; }
.bubble { background: var(--panel); border: 1px solid var(--border);
  border-radius: 12px; padding: 12px 16px; max-width: 85%; width: fit-content;
  white-space: pre-wrap; }
.input-row { display: flex; gap: 8px; padding: 16px; border-top: 1px solid var(--border); }
.input-row input { flex: 1; background: var(--panel); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 14px; color: var(--text); font-size: 15px; }
.input-row button { background: var(--accent); color: #fff; border: none;
  border-radius: 10px; padding: 0 20px; font-size: 15px; cursor: pointer; }
.input-row button:disabled { opacity: 0.5; cursor: not-allowed; }
.trace { font-size: 12.5px; color: var(--muted); margin: 6px 0 10px;
  border-left: 2px solid var(--border); padding-left: 10px; }
.trace-step { padding: 1px 0; }
.chip { display: inline-flex; align-items: center; justify-content: center;
  min-width: 20px; height: 20px; padding: 0 5px; margin: 0 2px;
  background: rgba(109,141,255,.15); color: var(--accent);
  border: 1px solid rgba(109,141,255,.4); border-radius: 6px;
  font-size: 11px; cursor: pointer; vertical-align: text-top; }
.chip:hover { background: rgba(109,141,255,.3); }
.source-pane { width: 380px; border-left: 1px solid var(--border);
  background: var(--panel); padding: 16px; overflow-y: auto; }
.source-pane h3 { margin-top: 0; }
.source-pane .close { float: right; cursor: pointer; color: var(--muted); }
.source-text { font-size: 13.5px; white-space: pre-wrap; color: #cfd3de; }
.meta { font-size: 12px; color: var(--muted); margin-bottom: 10px; }
.feedback { margin-top: 8px; }
.feedback button { background: none; border: 1px solid var(--border);
  border-radius: 8px; padding: 4px 10px; margin-right: 6px; cursor: pointer;
  color: var(--muted); font-size: 13px; }
.feedback button.active { border-color: var(--ok); color: var(--ok); }
.followups { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.followups button { background: var(--panel); border: 1px solid var(--border);
  color: var(--text); border-radius: 16px; padding: 6px 12px; font-size: 13px;
  cursor: pointer; }
.followups button:hover { border-color: var(--accent); }
.badge { font-size: 11px; color: var(--muted); margin-left: 8px; }
```

### `ui/app/page.tsx`

```tsx
import ChatStream from "../components/ChatStream";

export default function Page() {
  return <ChatStream />;
}
```

### `ui/components/ChatStream.tsx`

```tsx
"use client";
import { useRef, useState } from "react";
import { streamChat, AnswerPayload, Citation } from "../lib/api";
import AgentTrace from "./AgentTrace";
import CitationChip from "./CitationChip";
import SourceViewer from "./SourceViewer";
import FeedbackButtons from "./FeedbackButtons";
import FollowUpSuggestions from "./FollowUpSuggestions";

interface Message {
  role: "user" | "assistant";
  content: string;
  traces?: string[];
  answer?: AnswerPayload;
}

export default function ChatStream() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [openChunk, setOpenChunk] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const send = async (text: string) => {
    if (!text.trim() || busy) return;
    setBusy(true);
    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    setMessages((ms) => [...ms,
      { role: "user", content: text },
      { role: "assistant", content: "", traces: [] }]);
    setInput("");

    const patch = (fn: (m: Message) => Message) =>
      setMessages((ms) => {
        const copy = [...ms];
        copy[copy.length - 1] = fn(copy[copy.length - 1]);
        return copy;
      });

    try {
      await streamChat(text, history, {
        onTrace: (step) => patch((m) => ({ ...m, traces: [...(m.traces ?? []), step] })),
        onToken: (t) => {
          patch((m) => ({ ...m, content: m.content + t }));
          bottomRef.current?.scrollIntoView({ behavior: "smooth" });
        },
        onAnswer: (a) => patch((m) => ({ ...m, content: a.text, answer: a })),
        onError: (msg) => patch((m) => ({ ...m, content: `⚠ ${msg}` })),
      });
    } finally {
      setBusy(false);
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  };

  const renderWithCitations = (text: string, citations?: Citation[]) => {
    if (!citations?.length) return text;
    const parts = text.split(/(\[\d+\])/g);
    return parts.map((p, i) => {
      const m = p.match(/^\[(\d+)\]$/);
      if (!m) return <span key={i}>{p}</span>;
      const c = citations.find((c) => c.index === Number(m[1]));
      return c ? (
        <CitationChip key={i} citation={c} onOpen={() => setOpenChunk(c.chunk_id)} />
      ) : <span key={i}>{p}</span>;
    });
  };

  return (
    <div className="app">
      <div className="chat-pane">
        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              {m.role === "assistant" && m.traces && m.traces.length > 0 && (
                <AgentTrace steps={m.traces} done={!!m.answer || !busy} />
              )}
              <div className="bubble">
                {m.role === "assistant"
                  ? renderWithCitations(m.content, m.answer?.citations)
                  : m.content}
                {m.answer?.cached && <span className="badge">⚡ cached</span>}
                {m.answer?.escalated && <span className="badge">⚑ escalated</span>}
              </div>
              {m.role === "assistant" && m.answer && (
                <>
                  <FeedbackButtons
                    traceId={m.answer.trace_id}
                    query={messages[i - 1]?.content ?? ""}
                    answer={m.answer.text}
                  />
                  {i === messages.length - 1 && (
                    <FollowUpSuggestions
                      lastAnswer={m.answer.text}
                      onSelect={(q) => send(q)}
                    />
                  )}
                </>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
        <div className="input-row">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send(input)}
            placeholder="Ask about your documents…"
            disabled={busy}
          />
          <button onClick={() => send(input)} disabled={busy || !input.trim()}>
            {busy ? "…" : "Send"}
          </button>
        </div>
      </div>
      {openChunk && (
        <SourceViewer chunkId={openChunk} onClose={() => setOpenChunk(null)} />
      )}
    </div>
  );
}
```

### `ui/components/AgentTrace.tsx`

```tsx
"use client";
import { useState } from "react";

export default function AgentTrace({ steps, done }: { steps: string[]; done: boolean }) {
  const [open, setOpen] = useState(!done);
  const latest = steps[steps.length - 1];
  return (
    <div className="trace">
      <div style={{ cursor: "pointer" }} onClick={() => setOpen(!open)}>
        {done ? `✓ ${steps.length} steps` : `⟳ ${latest}`} {open ? "▾" : "▸"}
      </div>
      {open && steps.map((s, i) => <div key={i} className="trace-step">• {s}</div>)}
    </div>
  );
}
```

### `ui/components/CitationChip.tsx`

```tsx
"use client";
import { Citation } from "../lib/api";

export default function CitationChip({
  citation, onOpen,
}: { citation: Citation; onOpen: () => void }) {
  const label = citation.page
    ? `${citation.source}, p.${citation.page}`
    : citation.source;
  return (
    <span className="chip" title={label} onClick={onOpen}>
      {citation.index}
    </span>
  );
}
```

### `ui/components/SourceViewer.tsx`

```tsx
"use client";
import { useEffect, useState } from "react";
import { fetchSource } from "../lib/api";

interface Source {
  chunk_id: string; doc_id?: string; text: string;
  page?: number; section?: string; chunk_type: string;
  structured_json?: Record<string, unknown>[];
}

export default function SourceViewer({
  chunkId, onClose,
}: { chunkId: string; onClose: () => void }) {
  const [src, setSrc] = useState<Source | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    setSrc(null); setErr("");
    fetchSource(chunkId).then(setSrc).catch(() => setErr("Source not found"));
  }, [chunkId]);

  return (
    <div className="source-pane">
      <span className="close" onClick={onClose}>✕</span>
      <h3>Source</h3>
      {err && <p>{err}</p>}
      {src && (
        <>
          <div className="meta">
            {src.chunk_type}{src.page ? ` · page ${src.page}` : ""}
            {src.section ? ` · ${src.section}` : ""}
          </div>
          {src.chunk_type === "table" && src.structured_json ? (
            <table style={{ fontSize: 12, borderCollapse: "collapse" }}>
              <thead>
                <tr>{Object.keys(src.structured_json[0] ?? {}).map((k) => (
                  <th key={k} style={{ border: "1px solid #2a2f3d", padding: 4 }}>{k}</th>
                ))}</tr>
              </thead>
              <tbody>
                {src.structured_json.map((row, i) => (
                  <tr key={i}>{Object.values(row).map((v, j) => (
                    <td key={j} style={{ border: "1px solid #2a2f3d", padding: 4 }}>
                      {String(v)}
                    </td>
                  ))}</tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="source-text">{src.text}</div>
          )}
        </>
      )}
    </div>
  );
}
```

### `ui/components/FeedbackButtons.tsx`

```tsx
"use client";
import { useState } from "react";
import { sendFeedback } from "../lib/api";

export default function FeedbackButtons({
  traceId, query, answer,
}: { traceId?: string; query: string; answer: string }) {
  const [sent, setSent] = useState<1 | -1 | 0>(0);
  const submit = async (rating: 1 | -1) => {
    if (sent) return;
    setSent(rating);
    await sendFeedback({ trace_id: traceId, query, answer, rating });
  };
  return (
    <div className="feedback">
      <button className={sent === 1 ? "active" : ""} onClick={() => submit(1)}>👍</button>
      <button className={sent === -1 ? "active" : ""} onClick={() => submit(-1)}>👎</button>
      {sent !== 0 && <span className="badge">thanks — recorded</span>}
    </div>
  );
}
```

### `ui/components/FollowUpSuggestions.tsx`

```tsx
"use client";

const TEMPLATES = [
  "Can you cite the specific table for that?",
  "What figure supports this?",
  "Summarize the methodology behind this result.",
];

export default function FollowUpSuggestions({
  lastAnswer, onSelect,
}: { lastAnswer: string; onSelect: (q: string) => void }) {
  if (!lastAnswer || lastAnswer.startsWith("I could not find")) return null;
  return (
    <div className="followups">
      {TEMPLATES.map((t) => (
        <button key={t} onClick={() => onSelect(t)}>{t}</button>
      ))}
    </div>
  );
}
```

---

## 7. Infra — Docker & Kubernetes

### `infra/docker/Dockerfile.api`

```dockerfile
FROM python:3.12-slim AS base
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml --extra multimodal
COPY config/ config/
COPY ingestion/ ingestion/
COPY retrieval/ retrieval/
COPY generation/ generation/
COPY agents/ agents/
COPY guardrails/ guardrails/
COPY observability/ observability/
COPY api/ api/
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### `infra/docker/Dockerfile.ingestion`

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml --extra multimodal
COPY config/ config/
COPY ingestion/ ingestion/
COPY generation/ generation/
ENTRYPOINT ["python", "-m", "ingestion.pipeline"]
```

### `infra/docker/Dockerfile.ui`

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY ui/package.json ./
RUN npm install
COPY ui/ .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
EXPOSE 3001
CMD ["node", "server.js"]
```

### `infra/k8s/api-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mra-api
spec:
  replicas: 2
  selector: { matchLabels: { app: mra-api } }
  template:
    metadata: { labels: { app: mra-api } }
    spec:
      containers:
        - name: api
          image: ghcr.io/yourorg/mra-api:latest
          ports: [{ containerPort: 8000 }]
          envFrom:
            - secretRef: { name: mra-secrets }
          env:
            - { name: QDRANT_URL, value: "http://qdrant:6333" }
            - { name: REDIS_URL, value: "redis://redis:6379" }
          resources:
            requests: { cpu: "1", memory: 4Gi }
            limits: { cpu: "2", memory: 8Gi }
          readinessProbe:
            httpGet: { path: /readyz, port: 8000 }
            initialDelaySeconds: 30      # model warm-up
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /healthz, port: 8000 }
            periodSeconds: 15
---
apiVersion: v1
kind: Service
metadata: { name: mra-api }
spec:
  selector: { app: mra-api }
  ports: [{ port: 8000, targetPort: 8000 }]
```

### `infra/k8s/ingestion-cronjob.yaml`

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: mra-ingestion-sync
spec:
  schedule: "0 */6 * * *"          # incremental sync every 6h
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: ingestion
              image: ghcr.io/yourorg/mra-ingestion:latest
              args: ["--source", "/data"]
              envFrom:
                - secretRef: { name: mra-secrets }
              env:
                - { name: QDRANT_URL, value: "http://qdrant:6333" }
              volumeMounts:
                - { name: corpus, mountPath: /data, readOnly: true }
              resources:
                requests: { cpu: "2", memory: 8Gi }
          volumes:
            - name: corpus
              persistentVolumeClaim: { claimName: mra-corpus-pvc }
```

### `infra/k8s/hpa.yaml`

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: mra-api-hpa }
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mra-api
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: { type: Utilization, averageUtilization: 70 }
```

### `infra/k8s/ingress.yaml`

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mra-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-buffering: "off"   # SSE requires this
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  rules:
    - host: assistant.example.com
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend: { service: { name: mra-api, port: { number: 8000 } } }
          - path: /
            pathType: Prefix
            backend: { service: { name: mra-ui, port: { number: 3001 } } }
```

---

## 8. `Makefile` — final

```makefile
.PHONY: dev ingest eval eval-adv online-eval test reindex ui build

dev:
	docker-compose up -d && uvicorn api.main:app --reload

ui:
	cd ui && npm run dev

ingest:
	python -m ingestion.pipeline --source ./data

eval:
	python eval/run_eval.py --dataset eval/golden_dataset/questions.jsonl

eval-adv:
	python eval/run_eval.py --dataset eval/golden_dataset/adversarial.jsonl

online-eval:
	python observability/online_eval.py

test:
	pytest tests/ -x --cov

reindex:
	python scripts/rebuild_index.py --blue-green && \
	python -c "import asyncio; from api.caching import semantic_cache; \
	asyncio.run(semantic_cache.invalidate_all())"

build:
	docker build -f infra/docker/Dockerfile.api -t mra-api .
	docker build -f infra/docker/Dockerfile.ingestion -t mra-ingestion .
	docker build -f infra/docker/Dockerfile.ui -t mra-ui .
```

---

## 9. Verification Checklist (final steps of Build Order)

| Step | Command / action | Pass criteria |
|---|---|---|
| 7a | `make ingest` (with visual) → `make eval` | `type=figure` questions answered; `recall@10` did not regress |
| 7b | Add an `.mp3` to corpus → ask timestamped question | Answer cites `[MM:SS]` transcript chunk |
| 8 | `make eval-adv` | `refusal_accuracy = 1.0` still holds with visual context |
| 9a | Repeat identical query twice via `/chat` | 2nd response: `⚡ cached`, latency < 200ms |
| 9b | `cd ui && npm run dev` → chat at `:3001` | Trace steps stream, `[n]` chips open SourceViewer, 👍/👎 recorded |
| 9c | Langfuse UI at `:3000` | Trace per request with per-node spans; feedback scores attached |
| 10 | `make build` → `kubectl apply -f infra/k8s/` | `/readyz` green, SSE works through ingress, HPA scales under load |
| Ongoing | `make online-eval` (daily cron) | `online_faithfulness ≥ 0.9` on sampled traffic |

**Project complete.** Remaining optional enhancements (Phase 9 of the master plan): GraphRAG for multi-hop entity queries, LoRA fine-tuning of bge-m3 on domain pairs, multi-agent researcher/writer/reviewer workflows, and A/B pipeline testing on live traffic — each gated by the same eval discipline: `ci_gate.py` blocks any regression > 3%.