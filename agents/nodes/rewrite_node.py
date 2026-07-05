from retrieval.query_processing.query_rewriter import QueryRewriter

_rewriter = QueryRewriter()


async def run(state: dict) -> dict:
    q = state.get("rewritten_query") or state["query"]
    hist = state.get("history", [])

    # Simple heuristic: if we already rewrote, append "different keywords"
    # Otherwise, do a full history-aware rewrite.
    if state.get("retries", 0) > 0:
        new_q = f"{q} (alternative keywords)"
    else:
        new_q = await _rewriter.rewrite(q, hist)

    return {
        "rewritten_query": new_q,
        "retries": state.get("retries", 0) + 1,
        "trace_events": state.get("trace_events", []) + [f"Rewrote query: {new_q}"]
    }
