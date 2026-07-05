# LLM-Powered Application with RAG Pipelines — Complete Project Plan (2026 Edition)

> A production-grade, evaluation-driven roadmap reflecting the 2026 landscape: agentic RAG, hybrid retrieval, multimodal ingestion, long-context-aware chunking, guardrails, and LLMOps.

---

## Phase 0 — Project Definition & Scoping (Week 1)

### 0.1 Define the Use Case
Pick one concrete problem. Popular 2026-relevant options:
- **Enterprise knowledge assistant** (internal docs, wikis, Slack, tickets)
- **Legal/compliance research copilot** (contracts, regulations)
- **Customer support agent** (product docs + past resolutions)
- **Multimodal research assistant** (PDFs, images, tables, audio transcripts)

### 0.2 Define Success Criteria (before writing any code)
- **Answer faithfulness** ≥ 95% (no hallucinated claims)
- **Retrieval recall@10** ≥ 90% on a golden test set
- **P95 latency** ≤ 3–5 seconds end-to-end
- **Cost per query** budget (e.g., ≤ $0.01)
- Target users, expected QPS, data freshness requirements (real-time vs. daily sync)

### 0.3 Decide Key Architecture Questions
| Decision | Options (2026) |
|---|---|
| LLM hosting | API (GPT-5.x, Claude 4.x, Gemini 2.x) vs. self-hosted (Llama 4, Qwen 3, Mistral) via vLLM/SGLang |
| RAG style | Naive RAG → Advanced RAG → **Agentic RAG** (recommended) |
| Deployment | Cloud (AWS/GCP/Azure), on-prem (data-sensitive), or hybrid |
| Multimodality | Text-only vs. text + tables + images (vision-language models) |

---

## Phase 1 — Environment & Tech Stack Setup (Week 1–2)

### 1.1 Recommended 2026 Stack
```
Language:        Python 3.12+
Orchestration:   LangGraph / LlamaIndex Workflows / Haystack 2.x (agentic pipelines)
LLM Serving:     OpenAI/Anthropic API  OR  vLLM / SGLang (self-hosted)
Embeddings:      text-embedding-3-large, voyage-3, bge-m3 (multilingual), ColBERT-style late interaction
Vector DB:       Qdrant / Weaviate / pgvector / Milvus (with hybrid search support)
Keyword search:  BM25 (built into vector DB or Elasticsearch/OpenSearch)
Reranker:        Cohere Rerank 3.5 / bge-reranker-v2 / cross-encoder
Doc parsing:     Docling / unstructured.io / LlamaParse (handles tables, layout, OCR)
Eval:            Ragas / DeepEval / promptfoo + custom golden datasets
Observability:   LangSmith / Langfuse / Arize Phoenix (OpenTelemetry-based)
Guardrails:      NeMo Guardrails / Guardrails AI / LLM-as-judge filters
API layer:       FastAPI + streaming (SSE/WebSockets)
Frontend:        Next.js / React with streaming UI
Infra:           Docker, Kubernetes, GitHub Actions CI/CD
```

### 1.2 Setup Tasks
1. Create mono-repo structure: `ingestion/`, `retrieval/`, `generation/`, `agents/`, `eval/`, `api/`, `ui/`
2. Set up secrets management (API keys via Vault / cloud secret manager)
3. Provision vector DB (start with Docker-hosted Qdrant/pgvector locally)
4. Set up experiment tracking (Langfuse/LangSmith project)

---

## Phase 2 — Data Ingestion Pipeline (Week 2–3)

### 2.1 Data Source Connectors
- Build/configure connectors: PDFs, DOCX, HTML, Confluence, Notion, Slack, databases, SharePoint
- Implement **incremental sync** (change data capture / webhooks) — 2026 standard is near-real-time freshness

### 2.2 Parsing & Preprocessing
1. Use layout-aware parsers (Docling/LlamaParse) — extract **tables as structured data**, images with VLM-generated captions
2. Clean: deduplicate, strip boilerplate, normalize encodings
3. **PII detection & redaction** (Presidio or LLM-based) before storage

### 2.3 Chunking Strategy (critical — test multiple)
| Strategy | When to use |
|---|---|
| Semantic chunking | Default for prose (split on embedding-similarity boundaries) |
| Hierarchical (parent-child) | Retrieve small chunks, feed parent context to LLM |
| **Late chunking / contextual retrieval** | 2026 best practice — embed with document context, or prepend LLM-generated chunk context (Anthropic-style contextual retrieval) |
| Layout-based | Tables, slides, code files |

- Typical size: 256–512 tokens per chunk with contextual headers
- Attach rich **metadata**: source, date, author, section, access permissions (ACLs!)

### 2.4 Indexing
1. Generate dense embeddings (batch, with retry/rate-limit handling)
2. Build sparse index (BM25) in parallel → enables **hybrid search**
3. Optional: build **knowledge graph** (GraphRAG) for multi-hop/entity-heavy domains
4. Store with metadata filters + tenant/ACL fields

---

## Phase 3 — Retrieval Pipeline (Week 3–4)

### 3.1 Query Processing
1. **Query understanding**: intent classification, language detection
2. **Query transformation**:
   - Query rewriting / expansion (HyDE optional, less used in 2026)
   - Multi-query generation for ambiguous questions
   - Query decomposition for complex/multi-hop questions
3. **Routing**: send queries to the right index/tool (vector search vs. SQL vs. web search vs. direct LLM answer)

### 3.2 Hybrid Retrieval (2026 default)
```
Query → [Dense vector search (top 50)] ─┐
      → [BM25 keyword search (top 50)] ─┼→ RRF fusion → Reranker (cross-encoder, top 5–10) → Context
      → [Metadata/ACL filters]        ─┘
```

### 3.3 Reranking & Context Assembly
1. Cross-encoder rerank fused results (biggest single quality win)
2. Deduplicate + merge sibling chunks; pull parent context if using hierarchical chunks
3. **Context compression** if needed (LLMLingua-style) — though 2026 long-context models (1M+ tokens) reduce the need, retrieval precision still matters for cost & faithfulness
4. Format context with source citations `[doc_id]` for grounded generation

---

## Phase 4 — Generation & Agentic Layer (Week 4–6)

### 4.1 Prompt Design
- System prompt: role, grounding rules ("answer ONLY from context, cite sources, say 'I don't know' when unsupported")
- Structured output (JSON schema / tool calling) for citations
- Version prompts in code; A/B test via eval harness

### 4.2 Agentic RAG (the 2026 differentiator)
Build with LangGraph (or similar) a graph with these nodes:
1. **Router** — decide: retrieve, use tool, or answer directly
2. **Retriever** — hybrid retrieval as above
3. **Grader** — LLM judges retrieved chunks relevant/irrelevant (Corrective RAG pattern)
4. **Re-retrieval loop** — if grading fails: rewrite query, try alternate index, or fall back to web search
5. **Generator** — grounded answer with citations
6. **Self-check (reflection)** — hallucination check + answer-relevance check before responding
7. **Human escalation** — for low-confidence answers

### 4.3 Tool Use Beyond Retrieval
- SQL/analytics tool for structured data questions
- Web search tool for fresh information
- Calculator/code interpreter
- **MCP (Model Context Protocol) servers** — 2026 standard for connecting tools/data sources

### 4.4 Conversation Features
- Multi-turn memory: conversation summarization + entity memory (store in app state, not just raw history)
- Follow-up question rewriting (condense question with chat history)
- Streaming responses with progressive citations

---

## Phase 5 — Evaluation (Continuous, starting Week 4)

### 5.1 Build a Golden Dataset (non-negotiable)
- 100–300 question/answer/source triplets, human-curated + LLM-assisted generation
- Include: easy lookups, multi-hop, tables, "not in corpus" (should refuse), adversarial queries

### 5.2 Metrics (Ragas / DeepEval)
| Component | Metrics |
|---|---|
| Retrieval | context precision, context recall, MRR, recall@k |
| Generation | faithfulness, answer relevance, citation accuracy |
| End-to-end | correctness vs. golden answers (LLM-as-judge + human spot checks) |
| System | latency (P50/P95), cost/query, token usage |

### 5.3 Eval-Driven Development Loop
1. Run eval suite on every pipeline change (CI/CD gate)
2. Error analysis → categorize failures (retrieval miss vs. hallucination vs. bad chunking)
3. Fix the biggest failure category → re-run → repeat
4. Track regression across versions in Langfuse/LangSmith

---

## Phase 6 — Safety, Security & Guardrails (Week 6–7)

1. **Input guardrails**: prompt-injection detection (including injected content inside retrieved documents!), jailbreak filters, topic restrictions
2. **Output guardrails**: PII leakage checks, toxicity, unsupported-claim detection
3. **Access control**: enforce document-level ACLs at retrieval time (users only retrieve what they can see)
4. **Rate limiting & abuse prevention**
5. **Audit logging** of all queries/answers/sources (compliance: EU AI Act obligations are enforced in 2026 — document your system, risk assessment, transparency notices)

---

## Phase 7 — API, Frontend & Deployment (Week 7–8)

### 7.1 Backend
- FastAPI service: `/chat` (streaming SSE), `/feedback`, `/sources`
- Async pipeline execution; connection pooling to vector DB
- Semantic caching (cache answers for similar queries) → big cost savings

### 7.2 Frontend
- Chat UI with: streaming tokens, inline citations (click → source viewer), feedback buttons (👍/👎), suggested follow-ups
- Show agent reasoning steps ("Searching docs… Grading results…") — improves trust

### 7.3 Deployment
1. Dockerize all services; deploy on Kubernetes or serverless containers
2. CI/CD: lint → unit tests → **eval suite gate** → staging → canary → prod
3. Autoscaling for embedding/LLM inference workloads
4. Blue-green deploys for index updates (rebuild index → swap alias)

---

## Phase 8 — Observability & LLMOps (Week 8+, ongoing)

1. **Tracing**: every request traced end-to-end (query → retrieval → rerank → generation) via Langfuse/Phoenix with OpenTelemetry
2. **Dashboards**: latency, cost, token usage, retrieval hit rates, feedback scores
3. **Online evaluation**: sample production traffic, run LLM-as-judge on faithfulness daily
4. **Drift monitoring**: embedding drift, corpus staleness, query-topic shifts
5. **Feedback loop**: user 👎 → triage queue → add to golden dataset → improve pipeline
6. **Cost optimization**: model routing (small model for easy queries, large for hard), caching, batch embedding

---

## Phase 9 — Iteration & Advanced Enhancements (Month 3+)

- **GraphRAG** for multi-hop reasoning over entity-rich corpora
- **Multimodal RAG**: image/diagram retrieval with ColPali-style visual embeddings
- **Fine-tuning**: LoRA fine-tune embedding model on domain data; SFT the generator for tone/format
- **Multi-agent workflows**: researcher + writer + reviewer agents for complex tasks
- **Personalization**: user-level memory and preferences
- A/B testing of pipeline variants on live traffic

---

## Timeline Summary

| Weeks | Milestone |
|---|---|
| 1 | Scoping, success metrics, stack setup |
| 2–3 | Ingestion pipeline + indexing MVP |
| 3–4 | Hybrid retrieval + reranking |
| 4–6 | Agentic generation layer + golden eval dataset |
| 6–7 | Guardrails, security, compliance |
| 7–8 | API, UI, production deployment |
| 8+ | Observability, LLMOps, continuous eval |
| Month 3+ | GraphRAG, multimodal, fine-tuning, multi-agent |

## Key 2026 Principles (Cheat Sheet)
1. **Evaluation-driven development** — golden dataset before features
2. **Hybrid retrieval + reranking** — never dense-only
3. **Agentic loops beat single-pass RAG** — grade, reflect, retry
4. **Contextual chunking** — chunks embedded with document context
5. **Guardrails & ACLs are table stakes** — especially prompt injection via documents
6. **Long context ≠ no RAG** — retrieval still wins on cost, latency, faithfulness
7. **Trace everything** — you can't improve what you can't observe