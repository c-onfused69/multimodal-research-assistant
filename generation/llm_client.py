"""Unified async wrapper for Anthropic/OpenAI."""
import json
from typing import Any, Type

from pydantic import BaseModel

from config.settings import settings


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self.client = None

        if self.provider == "anthropic":
            api_key = settings.anthropic_api_key
            if api_key:
                from anthropic import AsyncAnthropic
                self.client = AsyncAnthropic(api_key=api_key)
        elif self.provider == "openai":
            api_key = settings.openai_api_key
            if api_key:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(api_key=api_key)

    def _check_client(self):
        """Raise a clear error if no LLM client is configured."""
        if self.client is None:
            raise ConnectionError(
                f"LLM provider '{self.provider}' is not configured. "
                f"Set the appropriate API key in your .env file "
                f"(ANTHROPIC_API_KEY or OPENAI_API_KEY)."
            )

    async def complete(self, prompt: str, system: str = "") -> str:
        self._check_client()
        if self.provider == "anthropic":
            res = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            return res.content[0].text
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            res = await self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return res.choices[0].message.content

    async def complete_structured(self, prompt: str, schema: Type[BaseModel], system: str = "") -> BaseModel:
        # Pydantic structured output mapping
        if self.provider == "openai":
            self._check_client()
            messages = [{"role": "user", "content": prompt}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            res = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=schema
            )
            return res.choices[0].message.parsed
        else:
            # Fallback to JSON mode for Anthropic/others with prompt injection
            augmented_prompt = f"{prompt}\n\nRespond ONLY with valid JSON matching this schema:\n{schema.model_json_schema()}"
            raw = await self.complete(augmented_prompt, system=system)
            return schema.model_validate_json(raw)

    async def complete_json(self, prompt: str, system: str = "") -> dict[str, Any]:
        """Generic JSON parsing when strict Pydantic isn't needed."""
        raw = await self.complete(f"{prompt}\n\nRespond ONLY with JSON.", system)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return {}

    async def complete_with_image(self, prompt: str, image_b64: str, mime: str = "image/png") -> str:
        self._check_client()
        if self.provider == "openai":
            res = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
                ]}]
            )
            return res.choices[0].message.content
        raise NotImplementedError("Image input only implemented for OpenAI provider here.")
