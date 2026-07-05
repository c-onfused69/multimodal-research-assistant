from agents.tools.web_search import WebSearchTool

_web = WebSearchTool()


async def run(state: dict) -> dict:
    query = state.get("rewritten_query") or state["query"]
    docs = await _web.search(query)
    return {
        "documents": docs,
        "trace_events": state.get("trace_events", []) +
                        [f"Web search: {len(docs)} results"],
    }
