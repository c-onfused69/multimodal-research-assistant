from config.settings import settings
from generation.llm_client import LLMClient

REWRITE_PROMPT = """You are an AI assistant helping to formulate a search query.
User's original query: {query}
Conversation history: {history}

Rewrite the query so it is fully self-contained and optimized for keyword and
semantic search. Do not include introductory text, just the query."""


class QueryRewriter:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient(model=settings.llm_small_model)

    async def rewrite(self, query: str, history: list[dict]) -> str:
        if not history:
            return query
        hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-3:])
        return await self.llm.complete(
            REWRITE_PROMPT.format(query=query, history=hist_text)
        )
