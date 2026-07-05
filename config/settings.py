from functools import lru_cache
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM ---
    llm_provider: str = "anthropic"              # anthropic | openai | vllm
    llm_model: str = "claude-sonnet-4-20250514"
    llm_small_model: str = "claude-3-5-haiku-latest"   # grading / rewriting / routing
    vlm_model: str = "gpt-4o"                    # image captioning
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    vllm_base_url: str = "http://localhost:8000/v1"

    # --- Embeddings ---
    text_embedding_model: str = "BAAI/bge-m3"
    visual_embedding_model: str = "vidore/colpali-v1.3"
    embedding_batch_size: int = 32

    # --- Retrieval ---
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rerank_top_k: int = 8
    rrf_k: int = 60
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Agent ---
    max_retries: int = 2
    confidence_threshold: float = 0.7

    # --- Chunking ---
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50

    # --- Infra ---
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379"
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    jwt_secret: str = "change-me"
    text_collection: str = "text_chunks"
    visual_collection: str = "visual_pages"
    table_collection: str = "tables"

    @model_validator(mode="after")
    def set_default_models(self) -> 'Settings':
        if self.llm_provider == "openai":
            if self.llm_model == "claude-sonnet-4-20250514":
                self.llm_model = "gpt-4o"
            if self.llm_small_model == "claude-3-5-haiku-latest":
                self.llm_small_model = "gpt-4o-mini"
        elif self.llm_provider == "gemini":
            if self.llm_model == "claude-sonnet-4-20250514":
                self.llm_model = "gemini-2.5-flash"
            if self.llm_small_model == "claude-3-5-haiku-latest":
                self.llm_small_model = "gemini-2.5-flash"
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
