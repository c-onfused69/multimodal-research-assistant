"""Agent state definition."""
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    history: list[dict]
    user_groups: list[str]

    route: str                           # direct | retrieve
    documents: list[dict[str, Any]]      # retrieved chunks
    rewritten_query: str                 # if rewritten

    answer: str
    citations: list[dict]
    confidence: float
    needs_escalation: bool

    retries: int

    _trace: Any                          # Langfuse trace object
    trace_events: list[str]              # internal audit log
    variant_config: dict                 # A/B testing overrides
