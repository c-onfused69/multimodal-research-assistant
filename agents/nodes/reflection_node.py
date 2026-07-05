from generation.generator import GroundedGenerator

_generator = GroundedGenerator()


async def run(state: dict) -> dict:
    query = state.get("rewritten_query") or state["query"]
    answer = state.get("answer", "")

    if "could not find the answer" in answer:
        return {"confidence": 0.0}

    eval_res = await _generator.evaluate_answer(query, answer)

    return {
        "confidence": eval_res.confidence,
        "trace_events": state.get("trace_events", []) + [f"Reflection confidence: {eval_res.confidence:.2f}"]
    }
