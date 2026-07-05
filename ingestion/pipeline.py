"""Master ingestion pipeline."""
import argparse
import asyncio
import logging

from ingestion.chunking.contextual_chunker import ContextualChunker
from ingestion.chunking.layout_chunker import LayoutChunker
from ingestion.connectors.local_files import LocalFilesConnector
from ingestion.indexing.embedder import Embedder
from ingestion.indexing.vector_store import VectorStore
from ingestion.indexing.sync import SyncManifest
from ingestion.parsers.docling_parser import DoclingParser

log = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, source_path: str):
        self.connector = LocalFilesConnector(source_path)
        self.parser = DoclingParser()
        self.text_chunker = ContextualChunker()
        self.layout_chunker = LayoutChunker()
        self.embedder = Embedder()
        self.store = VectorStore()
        self.manifest = SyncManifest()          # from ingestion.indexing.sync

    async def run(self) -> None:
        await self.store.setup()
        uris = await self.connector.list_documents()
        log.info("Found %d documents", len(uris))

        for uri in uris:
            await self._process(uri)

    async def _process(self, uri: str) -> None:
        doc = await self.connector.fetch(uri)
        if not await self.manifest.needs_index(doc.source_uri, doc.doc_id):
            log.info("Skip (unchanged): %s", uri)
            return

        log.info("Parsing: %s", uri)
        elements = self.parser.parse(doc)

        log.info("Chunking...")
        text_chunks = await self.text_chunker.chunk(elements)
        layout_chunks = self.layout_chunker.chunk(elements)
        all_chunks = text_chunks + layout_chunks

        # Annotate chunks with document metadata and ACLs
        for c in all_chunks:
            c["doc_id"] = doc.doc_id
            c["filename"] = doc.metadata.get("filename", "")
            c["acl"] = doc.acl

        log.info("Embedding %d chunks...", len(all_chunks))
        texts_to_embed = [c["text"] for c in all_chunks]
        embeddings = await self.embedder.embed(texts_to_embed)

        log.info("Indexing...")
        await self.store.upsert(
            [c for c in all_chunks if c["chunk_type"] == "text"],
            [e for c, e in zip(all_chunks, embeddings) if c["chunk_type"] == "text"],
            collection="text_chunks"
        )
        await self.store.upsert(
            [c for c in all_chunks if c["chunk_type"] == "table"],
            [e for c, e in zip(all_chunks, embeddings) if c["chunk_type"] == "table"],
            collection="tables"
        )
        
        await self.manifest.mark_indexed(doc.source_uri, doc.doc_id)
        log.info("Done: %s", uri)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    asyncio.run(IngestionPipeline(args.source).run())
