"""VLM captioning of figures/diagrams → searchable text."""
import base64

from config.settings import settings
from generation.llm_client import LLMClient

CAPTION_PROMPT = (
    "Describe this figure for search retrieval. Include: what it shows, "
    "axis labels / legend items, key trends or values, and any text visible "
    "in the image. Be factual and dense. 2-4 sentences."
)


class ImageCaptioner:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient(provider="openai", model=settings.vlm_model)

    async def caption(self, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        return await self.llm.complete_with_image(CAPTION_PROMPT, b64, mime="image/png")
