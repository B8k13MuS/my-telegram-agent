"""Health check endpoints for monitoring."""

from fastapi import APIRouter

from app.core.config import settings
from app.services.rag import RAGPipeline

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "1.0.0",
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check — verifies RAG pipeline."""
    try:
        rag = RAGPipeline()
        stats = rag.stats()
        return {
            "status": "ready",
            "documents": stats["documents_indexed"],
            "embedding_model": stats["embedding_model"],
            "llm_model": stats["llm_model"],
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "error": str(e),
        }


@router.get("/stats")
async def system_stats():
    """System statistics."""
    rag = RAGPipeline()
    return rag.stats()
