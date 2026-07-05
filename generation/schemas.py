from typing import Any

from pydantic import BaseModel, Field


class GradeResult(BaseModel):
    is_relevant: bool = Field(..., description="True if the document contains info relevant to the query.")
    reasoning: str = Field(..., description="Brief explanation.")


class AnswerResult(BaseModel):
    answer: str
    citations: list[int]
    confidence: float
