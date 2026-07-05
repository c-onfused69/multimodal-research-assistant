async def run(state: dict) -> dict:
    return {
        "needs_escalation": True,
        "trace_events": state.get("trace_events", []) + ["Escalated to human"]
    }
