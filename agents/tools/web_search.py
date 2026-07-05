"""Tavily web search — the `web_search` route's real backend.
Results are formatted like retrieved docs so grading/generation reuse works."""
import asyncio
import os


class WebSearchTool:
    def __init__(self):
        from tavily import TavilyClient
        self.client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))

    def _search_sync(self, query: str, max_results: int) -> list[dict]:
        res = self.client.search(query=query, max_results=max_results,
                                 include_answer=False)
        return res.get("results", [])

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            results = await asyncio.to_thread(self._search_sync, query, max_results)
        except Exception:
            return []
        docs = []
        for i, r in enumerate(results):
            docs.append({
                "chunk_id": f"web_{i}",
                "score": float(r.get("score", 0.0)),
                "text": r.get("content", ""),
                "display_text": r.get("content", ""),
                "chunk_type": "web",
                "filename": r.get("url", "web"),
                "doc_id": r.get("url", "web"),
                "acl": ["public"],
            })
        return docs
