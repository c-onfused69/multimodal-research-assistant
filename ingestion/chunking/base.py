from abc import ABC, abstractmethod
from typing import Any

from ingestion.parsers.docling_parser import ParsedElement


class AbstractChunker(ABC):
    @abstractmethod
    def chunk(self, elements: list[ParsedElement]) -> list[dict[str, Any]]:
        pass
