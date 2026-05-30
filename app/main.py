"""FastAPI application entry point.

Production (webhook mode):
  uvicorn app.main:app --host 0.0.0.0 --port 8000

Development (polling mode):
  python -m app.main bot
"""

import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, webhook
from app.bot.handlers import setup_webhook, remove_webhook, get_bot, start_polling
from app.core.config import settings
from app.core.security import SECURITY_HEADERS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: auto-ingest docs + setup webhook. Shutdown: cleanup."""
    print(f"🚀 Starting {settings.APP_NAME}")
    print(f"   LLM: {settings.LLM_MODEL}")
    print(f"   Embedding: {settings.EMBEDDING_MODEL}")
    print(f"   Qdrant: {settings.QDRANT_URL}")

    # Auto-ingest documents if Qdrant is empty
    try:
        from app.services.rag import RAGPipeline
        rag = RAGPipeline()
        if rag.vector.count() == 0:
            print("📤 No documents found. Running initial ingestion...")
            from scripts.ingest_docs import ingest_all
            result = await ingest_all()
            print(f"   Indexed {result['chunks']} chunks from {result['files']} files")
    except Exception as e:
        print(f"   ⚠️ Auto-ingest skipped: {e}")

    # Setup webhook in production (RENDER_EXTERNAL_URL is set by Render)
    base_url = settings.WEBHOOK_BASE_URL if hasattr(settings, 'WEBHOOK_BASE_URL') else ""
    if base_url:
        bot = get_bot()
        await setup_webhook(bot, base_url)
        print(f"   Webhook set: {base_url}/bot/webhook")

    yield

    # Shutdown
    if base_url:
        bot = get_bot()
        await remove_webhook(bot)
    print("👋 Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    description="AI Agent MVP with RAG and Telegram",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
)

# Security headers
@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    return response

# Routers
app.include_router(health.router)
app.include_router(webhook.router)


@app.get("/")
async def root():
    return {"name": settings.APP_NAME, "version": "1.0.0", "status": "ok"}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bot":
        # Polling mode for local development
        asyncio.run(start_polling())
    else:
        # API server (with webhook support)
        import uvicorn
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
