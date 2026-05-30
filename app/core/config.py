"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All settings are loaded from .env file or environment variables."""

    # App
    APP_NAME: str = "ai-agent-mvp"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Telegram — get token from @BotFather
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_ADMIN_IDS: str = ""

    # LLM — Groq (free tier, OpenAI-compatible API)
    # Get key: https://console.groq.com/keys
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_MODEL: str = "llama3-8b-8192"

    # Embeddings — local model, no API key needed
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # Vector DB — Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "business_docs"

    # Redis (optional, for caching)
    REDIS_URL: str = ""

    # Webhook base URL (set automatically on Render via RENDER_EXTERNAL_URL)
    WEBHOOK_BASE_URL: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
