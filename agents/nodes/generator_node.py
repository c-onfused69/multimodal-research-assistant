from generation.generator import GroundedGenerator
from retrieval.context.formatter import format_context
from retrieval.retrievers.hybrid_retriever import RetrievedDoc

_generator = GroundedGenerator()


async def run(state: dict) -> dict:
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("documents", [])

    if not docs:
        return {
            "answer": "I could not find the answer in the provided documents.",
            "citations": [],
            "trace_events": state.get("trace_events", []) + ["Generated fallback (no docs)"]
        }

    # Convert dict back to RetrievedDoc for formatter
    retrieved_docs = [
        RetrievedDoc(chunk_id=d.get("chunk_id", ""), score=0.0, payload=d)
        for d in docs
    ]

    context_xml, citations = format_context(retrieved_docs)
    answer = await _generator.generate(query, context_xml)

    return {
        "answer": answer,
        "citations": citations,
        "context": context_xml,      # saved for eval/tracing
        "trace_events": state.get("trace_events", []) + ["Generated answer"]
    }
