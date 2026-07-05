from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator


@dataclass
class RawDocument:
    doc_id: str
    source_uri: str
    content_type: str                    # pdf | docx | image | audio | html
    raw_bytes: bytes | None = None
    text: str | None = None
    metadata: dict = field(default_factory=dict)
    acl: list[str] = field(default_factory=lambda: ["public"])
    fetched_at: datetime = field(default_factory=datetime.utcnow)


class AbstractConnector(ABC):
    """All connectors implement: list → fetch → (optional) watch."""

    @abstractmethod
    async def list_documents(self) -> list[str]:
        """Return document URIs available at the source."""

    @abstractmethod
    async def fetch(self, uri: str) -> RawDocument:
        """Fetch a single document by URI."""

    async def watch(self) -> AsyncIterator[RawDocument]:  # optional CDC hook
        raise NotImplementedError
