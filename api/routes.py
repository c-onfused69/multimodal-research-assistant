"""FastAPI routes for the agentic RAG."""
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.graph import agent_graph
from agents.memory.conversation import ConversationMemory
from agents.workflows.deep_research import deep_research_graph
from api.cache import SemanticCache
from api.dependencies import UserSession, check_rate_limit
from guardrails.input_scanner import InputScanner
from guardrails.output_scanner import OutputScanner

router = APIRouter()
_cache = SemanticCache()
_in_scan = InputScanner()
_out_scan = OutputScanner()
_memory = ConversationMemory()


class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []
    mode: str = "fast"  # 'fast' or 'deep'


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, user: UserSession = Depends(check_rate_limit)):
    # 1. Input Guardrails
    is_safe, reason = await _in_scan.scan(req.query)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Input blocked: {reason}")

    # 2. Semantic Cache Check (only for empty history)
    if not req.history:
        cached = await _cache.get(req.query)
        if cached:
            cached["cached"] = True
            return cached

    # 3. Memory Condensation
    condensed_history = await _memory.condense(req.history)

    # 4. Agent Graph Execution
    state = {
        "query": req.query,
        "history": condensed_history,
        "user_groups": user.groups,
    }

    try:
        if req.mode == "deep":
            # Pass to the research subgraph
            final = await deep_research_graph.ainvoke(state)
        else:
            final = await agent_graph.ainvoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 5. Output Guardrails
    ans = final.get("answer", "")
    context = final.get("context", "")
    out_safe, out_reason = await _out_scan.scan(ans, context)
    if not out_safe:
        ans = "The generated answer was blocked by safety filters."

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
