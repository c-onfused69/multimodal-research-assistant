from generation.llm_client import LLMClient

ROUTER_PROMPT = """Classify this user query into exactly one route:
- "retrieve": needs information from the research document corpus
- "graph": multi-hop question connecting multiple entities/concepts
  ("how does X relate to Z", "trace the lineage of", "which methods that use X also...")
- "web_search": needs fresh/current information not in documents
- "sql": asks about structured/tabular data aggregation
- "direct": greeting, chit-chat, or general question needing no lookup

Query: {query}

Respond with ONLY the route word."""

VALID = {"retrieve", "graph", "web_search", "sql", "direct"}

_llm = LLMClient()

async def run(state: dict) -> dict:
    query = state["query"]
    route = await _llm.complete(ROUTER_PROMPT.format(query=query))
    route = route.strip().lower()
    
    # Strip any punctuation
    import re
    route = re.sub(r'[^a-z_]', '', route)

    if route not in VALID:
        route = "retrieve"

    return {
        "route": route,
        "trace_events": state.get("trace_events", []) + [f"Route: {route}"]
    }
