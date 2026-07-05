"""FastAPI routes for the agentic RAG."""
import traceback

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.cache import SemanticCache
from api.dependencies import UserSession, check_rate_limit

router = APIRouter()
_cache = SemanticCache()


class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []
    mode: str = "fast"  # 'fast' or 'deep'


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, user: UserSession = Depends(check_rate_limit)):
    # Lazy imports to avoid module-level crashes when LLM keys are missing
    from guardrails.input_scanner import InputScanner
    from guardrails.output_scanner import OutputScanner
    from agents.memory.conversation import ConversationMemory
    from agents.graph import agent_graph
    from agents.workflows.deep_research import deep_research_graph

    # 1. Input Guardrails
    try:
        scanner = InputScanner(use_llm=False)  # regex-only when LLM may be unavailable
        is_safe, reason = await scanner.scan(req.query)
        if not is_safe:
            raise HTTPException(status_code=400, detail=f"Input blocked: {reason}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Input scan error (skipped): {e}")

    # 2. Semantic Cache Check (only for empty history)
    if not req.history:
        cached = await _cache.get(req.query)
        if cached:
            cached["cached"] = True
            return cached

    # 3. Memory Condensation
    try:
        memory = ConversationMemory()
        condensed_history = await memory.condense(req.history)
    except Exception:
        condensed_history = req.history  # fallback: pass raw history

    # 4. Agent Graph Execution
    state = {
        "query": req.query,
        "history": condensed_history,
        "user_groups": user.groups,
    }

    try:
        if req.mode == "deep":
            final = await deep_research_graph.ainvoke(state)
        else:
            final = await agent_graph.ainvoke(state)
    except ConnectionError as e:
        # LLM not configured — surface a helpful message
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")

    # 5. Output Guardrails
    ans = final.get("answer", "")
    context = final.get("context", "")
    try:
        out_scanner = OutputScanner()
        out_safe, out_reason = await out_scanner.scan(ans, context)
        if not out_safe:
            ans = "The generated answer was blocked by safety filters."
    except Exception:
        out_safe = True  # fail-open if output scanning fails

    # 6. Build Response
    res = {
        "answer": ans,
        "citations": final.get("citations", []),
        "confidence": final.get("confidence", 0.0),
        "needs_escalation": final.get("needs_escalation", False),
        "cached": False
    }

    # Cache if confident and safe
    if out_safe and res["confidence"] > 0.8 and not req.history:
        await _cache.set(req.query, res)

    return res
