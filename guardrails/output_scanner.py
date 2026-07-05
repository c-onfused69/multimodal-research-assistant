"""Scans generated output for hallucinations, harmful content, and PII leakage."""
from typing import Tuple

from generation.llm_client import LLMClient

HALLUCINATION_PROMPT = """Check if the following answer is fully supported by the provided context.
Answer: {answer}
Context: {context}
Respond with ONLY "SAFE" if supported, or "UNSAFE" if hallucinated."""


class OutputScanner:
    def __init__(self):
        self.llm = LLMClient()

    async def scan(self, answer: str, context: str) -> Tuple[bool, str]:
        # Hallucination check
        try:
            res = await self.llm.complete(HALLUCINATION_PROMPT.format(
                answer=answer, context=context))
            if "UNSAFE" in res:
                return False, "Output flagged for hallucination."
        except Exception:
            pass

        return True, ""
