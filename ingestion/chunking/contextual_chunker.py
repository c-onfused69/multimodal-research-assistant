"""Semantic chunking with LLM-generated document context prepended."""
import tiktoken

from config.settings import settings
from generation.llm_client import LLMClient
from ingestion.chunking.base import AbstractChunker
from ingestion.parsers.docling_parser import ParsedElement

CONTEXT_PROMPT = """You are analyzing a document. Here is the full text:
<document>
{text}
</document>
Write a 2-3 sentence summary of the whole document. This will be prepended
to chunks to give them context."""


class ContextualChunker(AbstractChunker):
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient(model=settings.llm_small_model)
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.size = settings.chunk_size_tokens
        self.overlap = settings.chunk_overlap_tokens

    async def chunk(self, elements: list[ParsedElement]) -> list[dict]:
        text_elements = [e for e in elements if e.element_type == "text"]
        full_text = "\n\n".join(e.content for e in text_elements)
        if not full_text:
            return []

        doc_context = await self.llm.complete(
            CONTEXT_PROMPT.format(text=full_text[:50000])  # limit input
        )

        tokens = self.enc.encode(full_text)
        chunks = []
        for i in range(0, len(tokens), self.size - self.overlap):
            chunk_toks = tokens[i:i + self.size]
            chunk_text = self.enc.decode(chunk_toks)
            chunks.append({
                "text": f"{doc_context}\n\n{chunk_text}",
                "display_text": chunk_text,
                "chunk_type": "text",
            })
        return chunks
