"""Passes tables and figures through directly; text is handled by contextual."""
from typing import Any

from ingestion.chunking.base import AbstractChunker
from ingestion.parsers.docling_parser import ParsedElement


class LayoutChunker(AbstractChunker):
    def chunk(self, elements: list[ParsedElement]) -> list[dict[str, Any]]:
        chunks = []
        for e in elements:
            if e.element_type == "table":
                chunks.append({
                    "text": e.content,
                    "display_text": e.content,
                    "chunk_type": "table",
                    "structured_json": e.structured,
                    "page": e.page,
                    "section": e.section,
                })
            elif e.element_type == "figure" and e.image_bytes:
                chunks.append({
                    "text": e.metadata.get("caption", e.content),
                    "display_text": e.content,
                    "chunk_type": "figure",
                    "image_bytes": e.image_bytes,
                    "page": e.page,
                    "section": e.section,
                })
        return chunks
