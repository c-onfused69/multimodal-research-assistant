"""Langfuse tracing integration."""
from langfuse import Langfuse
from config.settings import settings


class Tracer:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            # Langfuse keys read from env automatically
            cls._instance = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host="https://cloud.langfuse.com"
            )
        return cls._instance
