"""Main grounded generation component."""
from generation.llm_client import LLMClient
from generation.prompts.registry import registry
from generation.schemas import AnswerResult


class GroundedGenerator:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()
        self.system_prompt = registry.get("system_grounded")

    async def generate(self, query: str, context_xml: str) -> str:
        prompt = f"Context documents:\n{context_xml}\n\nUser Question:\n{query}"
        return await self.llm.complete(prompt, system=self.system_prompt)

    async def evaluate_answer(self, query: str, answer: str) -> AnswerResult:
        prompt = f"Question: {query}\nAnswer: {answer}\nEvaluate the answer."
        return await self.llm.complete_structured(prompt, AnswerResult, system=registry.get("reflection"))
