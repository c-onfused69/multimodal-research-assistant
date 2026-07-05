"""Planner → Researcher → Writer → Reviewer loop (max 2 revisions).
Used for complex synthesis requests via mode="deep". Each finding is
grounded through the standard retrieve→rerank→generate path, so all
guardrails and citations carry through."""
from typing import TypedDict

from langgraph.graph import END, StateGraph

from config.settings import settings
from generation.generator import GroundedGenerator
from generation.llm_client import LLMClient
from retrieval.context.formatter import format_context
from retrieval.query_processing.decomposer import QueryDecomposer
from retrieval.reranking.cross_encoder import CrossEncoderReranker
from retrieval.retrievers.hybrid_retriever import HybridRetriever

_decomposer = QueryDecomposer()
_retriever = HybridRetriever()
_reranker = CrossEncoderReranker()
_generator = GroundedGenerator()
_reviewer_llm = LLMClient(model=settings.llm_model)

WRITER_PROMPT = """You are a research writer. Synthesize the findings below into a
coherent, well-structured answer to the main question. Preserve every inline
citation marker exactly as written (e.g., [1], [2]) — do not renumber or drop them.
{issues_block}
Main question: {question}

Findings:
{findings}

Synthesized answer:"""

REVIEWER_PROMPT = """You are reviewing a research answer for quality.

Question: {question}

Answer:
{answer}

Check: (1) fully addresses the question, (2) logically coherent,
(3) every major claim carries a citation marker, (4) no contradictions.
Respond ONLY with JSON:
{{"approved": bool, "issues": ["specific fix instructions..."]}}"""


class ResearchState(TypedDict, total=False):
    query: str
    user_groups: list[str]
    sub_questions: list[str]
    findings: list[dict]           # {question, answer, citations}
    draft: str
    review_issues: list[str]
    revision_count: int
    answer: str
    citations: list[dict]
    confidence: float
    needs_escalation: bool
    trace_events: list[str]


async def plan(state: ResearchState) -> dict:
    subs = await _decomposer.decompose(state["query"])
    return {"sub_questions": subs, "findings": [], "revision_count": 0,
            "trace_events": state.get("trace_events", []) +
                            [f"Plan: {len(subs)} sub-questions"]}


async def research(state: ResearchState) -> dict:
    groups = state.get("user_groups", ["public"])
    findings, all_citations = [], []
    offset = 0
    for sq in state["sub_questions"]:
        candidates = await _retriever.retrieve(sq, user_groups=groups)
        top = await _reranker.rerank(sq, candidates, top_k=5)
        if not top:
            findings.append({"question": sq, "answer": "(no sources found)",
                             "citations": []})
            continue
        context, citations = format_context(top)
        answer = await _generator.generate(sq, context)
        # Re-offset citation indices so they stay unique across findings
        for c in citations:
            c["index"] += offset
        answer = _shift_markers(answer, offset)
        offset += len(citations)
        findings.append({"question": sq, "answer": answer,
                         "citations": citations})
        all_citations.extend(citations)
    return {"findings": findings, "citations": all_citations,
            "trace_events": state.get("trace_events", []) +
                            [f"Researched {len(findings)} sub-questions, "
                             f"{len(all_citations)} sources"]}


def _shift_markers(text: str, offset: int) -> str:
    import re
    return re.sub(r"\[(\d+)\]",
                  lambda m: f"[{int(m.group(1)) + offset}]", text)


async def write(state: ResearchState) -> dict:
    findings_text = "\n\n".join(
        f"### {f['question']}\n{f['answer']}" for f in state["findings"])
    issues = state.get("review_issues", [])
    issues_block = ("Reviewer feedback to address:\n- " + "\n- ".join(issues) + "\n") \
        if issues else ""
    draft = await _generator.llm.complete(WRITER_PROMPT.format(
        question=state["query"], findings=findings_text,
        issues_block=issues_block))
    return {"draft": draft,
            "trace_events": state.get("trace_events", []) +
                            [f"Draft written (revision {state.get('revision_count', 0)})"]}


async def review(state: ResearchState) -> dict:
    verdict = await _reviewer_llm.complete_json(REVIEWER_PROMPT.format(
        question=state["query"], answer=state["draft"]))
    approved = bool(verdict.get("approved", False))
    return {"review_issues": verdict.get("issues", []),
            "revision_count": state.get("revision_count", 0) + (0 if approved else 1),
            "answer": state["draft"] if approved else state.get("answer", ""),
            "confidence": 0.85 if approved else 0.5,
            "trace_events": state.get("trace_events", []) +
                            [f"Review: {'approved ✓' if approved else 'revision requested'}"]}


async def finalize(state: ResearchState) -> dict:
    # Reviewer never approved → ship best draft, flagged for human review
    if not state.get("answer"):
        return {"answer": state.get("draft", ""), "needs_escalation": True,
                "confidence": 0.5,
                "trace_events": state.get("trace_events", []) +
                                ["Max revisions reached — escalated"]}
    return {"needs_escalation": False}


def build_deep_research_graph():
    g = StateGraph(ResearchState)
    g.add_node("plan", plan)
    g.add_node("research", research)
    g.add_node("write", write)
    g.add_node("review", review)
    g.add_node("finalize", finalize)

    g.set_entry_point("plan")
    g.add_edge("plan", "research")
    g.add_edge("research", "write")
    g.add_edge("write", "review")
    g.add_conditional_edges(
        "review",
        lambda s: "finalize" if s.get("answer") or s.get("revision_count", 0) >= 2
                  else "write",
        {"finalize": "finalize", "write": "write"})
    g.add_edge("finalize", END)
    return g.compile()


deep_research_graph = build_deep_research_graph()
