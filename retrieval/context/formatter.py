"""Formats chunks into an XML block and generates inline citation mappings."""
from retrieval.retrievers.hybrid_retriever import RetrievedDoc


def format_context(docs: list[RetrievedDoc]) -> tuple[str, list[dict]]:
    """Returns the formatted XML string and a list of citation dicts."""
    blocks = []
    citations = []

    for i, doc in enumerate(docs, start=1):
        # Determine the source label (filename or doc_id)
        source = doc.payload.get("filename") or doc.payload.get("doc_id", f"doc_{i}")
        page = doc.payload.get("page")
        page_str = f", page {page}" if page else ""

        # Build XML block
        blocks.append(
            f'<source id="{i}">\n'
            f'  <metadata>Source: {source}{page_str}</metadata>\n'
            f'  <content>\n{doc.payload.get("display_text", doc.payload.get("text", ""))}\n  </content>\n'
            f'</source>'
        )

        # Record citation mapping for the UI
        citations.append({
            "index": i,
            "chunk_id": doc.chunk_id,
            "doc_id": doc.payload.get("doc_id"),
            "source": source,
            "page": page,
            "chunk_type": doc.payload.get("chunk_type", "text"),
            "score": doc.score,
        })

    return "\n\n".join(blocks), citations
