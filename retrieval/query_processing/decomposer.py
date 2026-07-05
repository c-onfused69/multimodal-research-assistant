from config.settings import settings
from generation.llm_client import LLMClient

DECOMPOSE_PROMPT = """Break this complex research question into 2-5 self-contained
sub-questions that can each be answered independently from a document corpus.
Order them so later sub-questions can build on earlier ones.

Question: {question}

Respond ONLY with JSON: {{"sub_questions": ["...", "..."]}}"""


class QueryDecomposer:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient(model=settings.llm_small_model)

    async def decompose(self, question: str) -> list[str]:
        out = await self.llm.complete_json(
            DECOMPOSE_PROMPT.format(question=question))
        subs = [s for s in out.get("sub_questions", []) if isinstance(s, str)]
        return subs[:5] or [question]
