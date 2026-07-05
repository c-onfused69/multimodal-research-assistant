# Multimodal Research Assistant — Complete Project Structure & Build Guide

> **Use case:** Ingest PDFs, images, tables, and audio transcripts → hybrid + multimodal retrieval → agentic RAG generation with citations, guardrails, and full observability.

---

## 1. Complete Monorepo Structure

```
multimodal-research-assistant/
│
├── README.md
├── Makefile                          # make ingest / make eval / make dev / make deploy
├── pyproject.toml                    # Python 3.12+, uv/poetry managed
├── .env.example                      # Template for all secrets (never commit .env)
├── .pre-commit-config.yaml           # ruff, mypy, secret-scanning hooks
├── docker-compose.yaml               # Local dev: Qdrant, Postgres, Langfuse, Redis
│
├── config/
│   ├── settings.py                   # Pydantic Settings (env-driven, typed config)
│   ├── models.yaml                   # LLM/embedding model registry + routing rules
│   ├── chunking.yaml                 # Chunking strategy params per doc type
│   ├── retrieval.yaml                # top_k, fusion weights, reranker config
│   └── guardrails.yaml               # NeMo Guardrails rails definitions
│
├── ingestion/                        # ============ PHASE 2 ============
│   ├── __init__.py
│   ├── connectors/
│   │   ├── base.py                   # AbstractConnector (list, fetch, watch)
│   │   ├── local_files.py            # PDF/DOCX/PPTX/images from disk
│   │   ├── web.py                    # HTML scraping connector
│   │   ├── notion.py                 # Notion API connector
│   │   └── audio.py                  # Audio → Whisper transcription connector
│   ├── parsers/
│   │   ├── docling_parser.py         # Layout-aware PDF parsing (tables, figures)
│   │   ├── table_extractor.py        # Tables → structured JSON + markdown
│   │   ├── image_captioner.py        # VLM captions for figures/diagrams
│   │   └── audio_transcriber.py      # Whisper/Deepgram transcription + timestamps
│   ├── preprocessing/
│   │   ├── cleaner.py                # Dedup, boilerplate stripping, encoding fix
│   │   └── pii_redactor.py           # Presidio-based PII detection & masking
│   ├── chunking/
│   │   ├── base.py                   # Chunker interface + Chunk dataclass
│   │   ├── semantic_chunker.py       # Embedding-similarity boundary splitting
│   │   ├── hierarchical_chunker.py   # Parent-child chunks
│   │   ├── contextual_chunker.py     # ★ Anthropic-style: LLM-prepended context
│   │   └── layout_chunker.py         # Tables, slides, code blocks
│   ├── indexing/
│   │   ├── embedder.py               # Dense embeddings (batch, retry, rate-limit)
│   │   ├── visual_embedder.py        # ColPali-style image/page embeddings
│   │   ├── sparse_indexer.py         # BM25 index builder
│   │   ├── vector_store.py           # Qdrant client wrapper (upsert w/ metadata+ACL)
│   │   └── sync.py                   # Incremental sync / change-data-capture logic
│   └── pipeline.py                   # Orchestrates: connect→parse→clean→chunk→index
│
├── retrieval/                        # ============ PHASE 3 ============
│   ├── __init__.py
│   ├── query_processing/
│   │   ├── intent_classifier.py      # Route: retrieval / SQL / web / direct answer
│   │   ├── query_rewriter.py         # Expansion, multi-query, history condensing
│   │   └── decomposer.py             # Multi-hop question decomposition
│   ├── retrievers/
│   │   ├── dense_retriever.py        # Vector search (top 50)
│   │   ├── sparse_retriever.py       # BM25 (top 50)
│   │   ├── visual_retriever.py       # ColPali visual page retrieval
│   │   ├── hybrid_retriever.py       # ★ RRF fusion of dense + sparse + filters
│   │   └── acl_filter.py             # Document-level permission enforcement
│   ├── reranking/
│   │   └── cross_encoder.py          # bge-reranker-v2 / Cohere Rerank 3.5
│   └── context/
│       ├── assembler.py              # Dedup, sibling merge, parent expansion
│       ├── compressor.py             # LLMLingua-style compression (optional)
│       └── formatter.py              # Citation-tagged context [doc_id] formatting
│
├── generation/                       # ============ PHASE 4.1 ============
│   ├── __init__.py
│   ├── prompts/
│   │   ├── system_grounded.txt       # "Answer ONLY from context, cite, refuse..."
│   │   ├── grader.txt                # Chunk relevance grading prompt
│   │   ├── reflection.txt            # Hallucination self-check prompt
│   │   └── registry.py               # Versioned prompt loader (prompts as code)
│   ├── llm_client.py                 # Unified client (API + vLLM), model routing
│   ├── generator.py                  # Grounded generation w/ structured citations
│   └── schemas.py                    # Pydantic output schemas (Answer, Citation)
│
├── agents/                           # ============ PHASE 4.2–4.4 ============
│   ├── __init__.py
│   ├── graph.py                      # ★ LangGraph state machine definition
│   ├── state.py                      # AgentState TypedDict (query, docs, grades…)
│   ├── nodes/
│   │   ├── router.py                 # retrieve / tool / direct-answer decision
│   │   ├── retriever_node.py         # Wraps hybrid retriever
│   │   ├── grader_node.py            # Corrective RAG: grade chunk relevance
│   │   ├── rewrite_node.py           # Re-retrieval loop: rewrite & retry
│   │   ├── generator_node.py         # Grounded answer generation
│   │   ├── reflection_node.py        # Hallucination + relevance self-check
│   │   └── escalation_node.py        # Low-confidence → human handoff
│   ├── tools/
│   │   ├── web_search.py             # Fresh info tool (Tavily/Brave)
│   │   ├── sql_tool.py               # Structured data queries
│   │   ├── code_interpreter.py       # Calculator / analysis sandbox
│   │   └── mcp_client.py             # MCP server connections
│   └── memory/
│       ├── conversation.py           # Summarization + follow-up condensing
│       └── entity_memory.py          # Entity store in app state
│
├── eval/                             # ============ PHASE 5 ============
│   ├── __init__.py
│   ├── golden_dataset/
│   │   ├── questions.jsonl           # 100–300 Q/A/source triplets
│   │   ├── adversarial.jsonl         # Injection attempts, "not in corpus" cases
│   │   └── generate_synthetic.py     # LLM-assisted QA generation (human-reviewed)
│   ├── metrics/
│   │   ├── retrieval_metrics.py      # recall@k, MRR, context precision/recall
│   │   ├── generation_metrics.py     # Ragas: faithfulness, answer relevance
│   │   └── citation_accuracy.py      # Do citations support claims?
│   ├── run_eval.py                   # Full suite runner → Langfuse experiment
│   ├── error_analysis.py             # Failure categorization report
│   └── ci_gate.py                    # Fails CI if metrics regress vs. baseline
│
├── guardrails/                       # ============ PHASE 6 ============
│   ├── __init__.py
│   ├── input_rails.py                # Prompt-injection & jailbreak detection
│   ├── document_rails.py             # ★ Injection scan INSIDE retrieved docs
│   ├── output_rails.py               # PII leakage, toxicity, unsupported claims
│   ├── rate_limiter.py               # Per-user/tenant rate limiting
│   └── audit_logger.py               # Compliance logging (EU AI Act)
│
├── api/                              # ============ PHASE 7.1 ============
│   ├── __init__.py
│   ├── main.py                       # FastAPI app factory + middleware
│   ├── routers/
│   │   ├── chat.py                   # POST /chat — SSE streaming endpoint
│   │   ├── feedback.py               # POST /feedback — 👍/👎 capture
│   │   ├── sources.py                # GET /sources/{doc_id} — source viewer
│   │   └── health.py                 # /healthz, /readyz
│   ├── auth.py                       # JWT/OIDC auth → user ACL claims
│   ├── caching.py                    # Semantic cache (Redis + embedding sim)
│   └── deps.py                       # Dependency injection (DB pools, clients)
│
├── ui/                               # ============ PHASE 7.2 ============
│   ├── package.json                  # Next.js 15+ / React
│   ├── app/
│   │   ├── page.tsx                  # Chat page
│   │   └── layout.tsx
│   ├── components/
│   │   ├── ChatStream.tsx            # SSE token streaming
│   │   ├── CitationChip.tsx          # Inline [1] → click opens source
│   │   ├── SourceViewer.tsx          # PDF/image/table viewer panel
│   │   ├── AgentTrace.tsx            # "Searching… Grading… Reflecting…" steps
│   │   ├── FeedbackButtons.tsx
│   │   └── FollowUpSuggestions.tsx
│   └── lib/api.ts                    # Typed API client
│
├── observability/                    # ============ PHASE 8 ============
│   ├── tracing.py                    # OpenTelemetry + Langfuse setup
│   ├── dashboards/                   # Grafana dashboard JSON exports
│   ├── online_eval.py                # Daily LLM-as-judge on sampled traffic
│   └── drift_monitor.py              # Embedding drift, corpus staleness alerts
│
├── infra/                            # ============ PHASE 7.3 ============
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.ingestion
│   │   └── Dockerfile.ui
│   ├── k8s/
│   │   ├── api-deployment.yaml
│   │   ├── ingestion-cronjob.yaml    # Scheduled incremental sync
│   │   ├── hpa.yaml                  # Autoscaling
│   │   └── ingress.yaml
│   └── terraform/                    # Cloud resources (optional)
│
├── scripts/
│   ├── seed_corpus.py                # Load sample research papers for dev
│   ├── rebuild_index.py              # Blue-green index rebuild + alias swap
│   └── export_traces.py
│
├── tests/
│   ├── unit/                         # Per-module unit tests
│   ├── integration/                  # Pipeline integration tests
│   └── conftest.py                   # Fixtures: fake vector store, mock LLM
│
└── .github/
    └── workflows/
        ├── ci.yaml                   # lint → typecheck → unit → EVAL GATE
        └── deploy.yaml               # staging → canary → prod
```

---

## 2. Tech Stack (Pinned Choices)

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** | Best-in-class agentic state machines |
| LLM | Claude / GPT API (dev) → vLLM + Qwen 3 (scale) | Fast start, cost path later |
| VLM (captions) | GPT-4o-class or Qwen-VL | Figure/diagram captioning |
| Embeddings | **bge-m3** (text) + **ColPali** (visual pages) | Multilingual + multimodal |
| Vector DB | **Qdrant** | Native hybrid search, ACL payload filters |
| Reranker | **bge-reranker-v2** (local) or Cohere Rerank 3.5 | Biggest quality win |
| Parsing | **Docling** | Tables, layout, OCR — open source |
| Audio | Whisper large-v3 | Transcripts w/ timestamps |
| Eval | **Ragas + DeepEval** | Retrieval + generation metrics |
| Observability | **Langfuse** (self-hosted) | OTel-based, free tier |
| Guardrails | NeMo Guardrails + Presidio | Rails + PII |
| API / UI | FastAPI (SSE) + Next.js | Streaming-native |

---

## 3. Key File Blueprints

### 3.1 `config/settings.py`
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    llm_provider: str = "anthropic"          # anthropic | openai | vllm
    llm_model: str = "claude-sonnet-latest"
    vlm_model: str = "gpt-4o"                # image captioning
    # Embeddings
    text_embedding_model: str = "BAAI/bge-m3"
    visual_embedding_model: str = "vidore/colpali-v1.3"
    # Retrieval
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rerank_top_k: int = 8
    # Infra
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379"
    langfuse_host: str = "http://localhost:3000"

    class Config:
        env_file = ".env"

settings = Settings()
```

### 3.2 `agents/state.py` — the agent's shared state
```python
from typing import TypedDict, Literal

class AgentState(TypedDict):
    query: str
    rewritten_query: str
    route: Literal["retrieve", "web_search", "sql", "direct"]
    documents: list[dict]          # retrieved chunks w/ metadata
    grades: list[bool]             # per-chunk relevance
    retry_count: int               # max 2 re-retrieval loops
    answer: str
    citations: list[dict]
    is_grounded: bool              # reflection verdict
    confidence: float
    needs_escalation: bool
```

### 3.3 `agents/graph.py` — the agentic RAG loop
```python
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import (router, retriever_node, grader_node,
                          rewrite_node, generator_node,
                          reflection_node, escalation_node)

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", router.run)
    g.add_node("retrieve", retriever_node.run)
    g.add_node("grade", grader_node.run)
    g.add_node("rewrite", rewrite_node.run)
    g.add_node("generate", generator_node.run)
    g.add_node("reflect", reflection_node.run)
    g.add_node("escalate", escalation_node.run)

    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: s["route"], {
        "retrieve": "retrieve", "direct": "generate",
        "web_search": "retrieve", "sql": "retrieve",
    })
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade",
        lambda s: "generate" if any(s["grades"]) or s["retry_count"] >= 2
                  else "rewrite",
        {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")          # re-retrieval loop
    g.add_edge("generate", "reflect")
    g.add_conditional_edges("reflect",
        lambda s: END if s["is_grounded"] and s["confidence"] > 0.7
                  else ("escalate" if s["retry_count"] >= 2 else "rewrite"),
        {END: END, "escalate": "escalate", "rewrite": "rewrite"})
    g.add_edge("escalate", END)
    return g.compile()
```

### 3.4 `ingestion/chunking/contextual_chunker.py` (2026 best practice)
```python
CONTEXT_PROMPT = """<document>{doc_summary}</document>
Here is a chunk from the document:
<chunk>{chunk}</chunk>
Write 1-2 sentences situating this chunk within the document
for search retrieval. Respond only with the context."""

async def contextualize(chunk: str, doc_summary: str, llm) -> str:
    ctx = await llm.complete(CONTEXT_PROMPT.format(
        doc_summary=doc_summary, chunk=chunk))
    return f"{ctx}\n\n{chunk}"     # prepend before embedding
```

### 3.5 `retrieval/retrievers/hybrid_retriever.py` (RRF fusion)
```python
def rrf_fuse(dense: list, sparse: list, k: int = 60) -> list:
    scores = {}
    for results in (dense, sparse):
        for rank, doc in enumerate(results):
            scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

### 3.6 `docker-compose.yaml` (local dev)
```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: [qdrant_data:/qdrant/storage]
  postgres:
    image: pgvector/pgvector:pg16
    environment: {POSTGRES_PASSWORD: dev}
    ports: ["5432:5432"]
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
  langfuse:
    image: langfuse/langfuse:latest
    depends_on: [postgres]
    ports: ["3000:3000"]
volumes:
  qdrant_data:
```

### 3.7 `.env.example`
```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
COHERE_API_KEY=            # reranker (optional)
TAVILY_API_KEY=            # web search tool
QDRANT_URL=http://localhost:6333
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
JWT_SECRET=
```

### 3.8 `.github/workflows/ci.yaml` (eval gate)
```yaml
name: CI
on: [pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: uv sync
      - run: ruff check . && mypy .
      - run: pytest tests/unit -x
      - name: Eval Gate (blocks merge on regression)
        run: python eval/ci_gate.py --baseline main --threshold-drop 0.03
```

---

## 4. Multimodal Ingestion Flow (the differentiator)

```
                      ┌─ text blocks ──→ contextual_chunker ──→ bge-m3 ──┐
PDF ──→ Docling ──────┼─ tables ───────→ layout_chunker (md+json) ───────┤
                      └─ figures ──────→ VLM caption + ColPali embed ────┼──→ Qdrant
Images ──→ VLM caption + ColPali visual embedding ───────────────────────┤    (multi-vector
Audio ──→ Whisper ──→ timestamped transcript ──→ semantic_chunker ───────┘     collections
                                                                                + BM25 + ACL)
```

**Qdrant collections:**
| Collection | Vector | Payload |
|---|---|---|
| `text_chunks` | bge-m3 dense + sparse | source, page, section, acl, parent_id |
| `visual_pages` | ColPali multi-vector | doc_id, page_num, thumbnail_url, acl |
| `tables` | bge-m3 (of md rendering) | structured_json, source, acl |

At query time: text query embedded both ways → retrieve from all 3 collections → RRF fuse → rerank → assemble (images returned as captioned references + thumbnails).

---

## 5. Build Order (Follow Exactly)

| Step | What to build | Verify with |
|---|---|---|
| 1 | `docker-compose up` + `config/` + secrets | Qdrant & Langfuse UIs load |
| 2 | `ingestion/`: local_files connector → Docling → contextual chunking → index 10 papers | Inspect chunks in Qdrant UI |
| 3 | **`eval/golden_dataset/`: write 50 Q/A pairs NOW** | Human review |
| 4 | `retrieval/`: dense-only → add BM25 → RRF → reranker | recall@10 ≥ 0.9 on golden set |
| 5 | `generation/`: grounded prompt + citations (single-pass RAG) | Faithfulness ≥ 0.9 (Ragas) |
| 6 | `agents/`: LangGraph — router → grader → rewrite loop → reflection | Adversarial set: refusals correct |
| 7 | Multimodal: image captioning + ColPali + audio transcripts | Figure-based questions answered |
| 8 | `guardrails/` + ACL filters + audit logs | Injection test suite passes |
| 9 | `api/` SSE streaming + semantic cache → `ui/` chat | P95 latency ≤ 5s |
| 10 | `infra/` + CI eval gate + `observability/` | Canary deploy succeeds |

**Golden rule:** never advance a step if the eval metric for the previous step regressed.

---

## 6. Makefile Targets

```makefile
dev:        docker-compose up -d && uvicorn api.main:app --reload
ingest:     python -m ingestion.pipeline --source ./data --incremental
eval:       python eval/run_eval.py --dataset eval/golden_dataset/questions.jsonl
eval-adv:   python eval/run_eval.py --dataset eval/golden_dataset/adversarial.jsonl
reindex:    python scripts/rebuild_index.py --blue-green
test:       pytest tests/ -x --cov
```