"""LangGraph definition for the Agentic RAG loop."""
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from agents.state import AgentState
from agents.nodes import (
    escalation_node, generator_node, grader_node, reflection_node,
    retriever_node, rewrite_node, router, web_search_node
)
from config.settings import settings


def traced_node(name: str):
    """Decorator to log node entry/exit into the trace."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(state: AgentState) -> dict[str, Any]:
            trace = state.get("_trace")
            if trace:
                span = trace.span(name=name, input=state)
            try:
                update = await func(state)
                if trace:
                    span.end(output=update)
                return update
            except Exception as e:
                if trace:
                    span.end(output={"error": str(e)}, level="ERROR")
                raise
        return wrapper
    return decorator


def build_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("router", traced_node("router")(router.run))
    g.add_node("retrieve", traced_node("retrieve")(retriever_node.run))
    g.add_node("grade", traced_node("grade")(grader_node.run))
    g.add_node("rewrite", traced_node("rewrite")(rewrite_node.run))
    g.add_node("generate", traced_node("generate")(generator_node.run))
    g.add_node("reflect", traced_node("reflect")(reflection_node.run))
    g.add_node("escalate", traced_node("escalate")(escalation_node.run))
    g.add_node("web_search", traced_node("web_search")(web_search_node.run))

    # Edges
    g.set_entry_point("router")

    g.add_conditional_edges("router", lambda s: s.get("route", "retrieve"), {
        "retrieve": "retrieve",
        "direct": "generate",
        "web_search": "web_search",
        "sql": "retrieve",
        "graph": "retrieve",
    })

    g.add_edge("retrieve", "grade")
    g.add_edge("web_search", "grade")

    def grade_router(state: AgentState) -> str:
        if not state.get("documents") and state.get("retries", 0) < settings.max_retries:
            return "rewrite"
        return "generate"

    g.add_conditional_edges("grade", grade_router, {
        "rewrite": "rewrite",
        "generate": "generate"
    })

    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "reflect")

    def reflect_router(state: AgentState) -> str:
        if state.get("needs_escalation"):
            return "escalate"
        if state.get("confidence", 1.0) < settings.confidence_threshold:
            if state.get("retries", 0) < settings.max_retries:
                return "rewrite"
            return "escalate"
        return END

    g.add_conditional_edges("reflect", reflect_router, {
        "rewrite": "rewrite",
        "escalate": "escalate",
        END: END
    })
    g.add_edge("escalate", END)

    return g.compile()


agent_graph = build_graph()
