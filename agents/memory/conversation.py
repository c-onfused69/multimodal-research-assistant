"""Rolling conversation memory: keep last N turns verbatim, summarize the rest.
Used by the chat endpoint to bound history token cost."""
from config.settings import settings
from generation.llm_client import LLMClient

SUMMARY_PROMPT = """Summarize this conversation in 3-5 sentences, preserving
entities, decisions, and open questions:\n\n{history}"""

KEEP_TURNS = 6


class ConversationMemory:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient(model=settings.llm_small_model)

    async def condense(self, history: list[dict]) -> list[dict]:
        if len(history) <= KEEP_TURNS:
            return history
        old, recent = history[:-KEEP_TURNS], history[-KEEP_TURNS:]
        text = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in old)
        summary = await self.llm.complete(SUMMARY_PROMPT.format(history=text))
        return [{"role": "system",
                 "content": f"(Earlier conversation summary) {summary.strip()}"}] + recent
