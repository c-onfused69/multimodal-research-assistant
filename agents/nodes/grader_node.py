import asyncio

from generation.llm_client import LLMClient
from generation.prompts.registry import registry
from generation.schemas import GradeResult

_llm = LLMClient()


async def _grade_doc(query: str, doc: dict) -> bool:
    prompt = f"Question: {query}\n\nDocument:\n{doc.get('display_text', '')}"
    res = await _llm.complete_structured(
        prompt, GradeResult, system=registry.get("grader")
    )
    return res.is_relevant


async def run(state: dict) -> dict:
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("documents", [])

    if not docs:
        return {"documents": []}

    tasks = [_grade_doc(query, d) for d in docs]
    results = await asyncio.gather(*tasks)

    filtered = [d for d, keep in zip(docs, results) if keep]
    dropped = len(docs) - len(filtered)

    return {
        "documents": filtered,
        "trace_events": state.get("trace_events", []) + [f"Graded: {len(filtered)} kept, {dropped} dropped"]
    }
