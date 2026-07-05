import hashlib
import mimetypes
from pathlib import Path

from ingestion.connectors.base import AbstractConnector, RawDocument

SUPPORTED = {".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".mp3", ".wav", ".html", ".md", ".txt"}

CONTENT_TYPE_MAP = {
    ".pdf": "pdf", ".docx": "docx", ".pptx": "pptx",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".mp3": "audio", ".wav": "audio",
    ".html": "html", ".md": "text", ".txt": "text",
}


class LocalFilesConnector(AbstractConnector):
    def __init__(self, root: str | Path):
        self.root = Path(root)

    async def list_documents(self) -> list[str]:
        return [
            str(p) for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED
        ]

    async def fetch(self, uri: str) -> RawDocument:
        path = Path(uri)
        raw = path.read_bytes()
        doc_id = hashlib.sha256(raw).hexdigest()[:16]
        return RawDocument(
            doc_id=doc_id,
            source_uri=str(path),
            content_type=CONTENT_TYPE_MAP[path.suffix.lower()],
            raw_bytes=raw,
            metadata={
                "filename": path.name,
                "mime": mimetypes.guess_type(path.name)[0],
                "size_bytes": len(raw),
                "mtime": path.stat().st_mtime,
            },
        )
