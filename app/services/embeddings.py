"""Local embedding service using sentence-transformers.

No API key required. Runs entirely on CPU.
Model: all-MiniLM-L6-v2 (22MB, 384-dim vectors).

Why local embeddings?
- Zero API cost — critical for $0 budget
- Fast for small document sets (< 1000 docs)
- Privacy — documents never leave your server
- No rate limits or network latency
"""

import asyncio
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingService:
    """Generate text embeddings locally."""

    def __init__(self):
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
        )
        self._loop = asyncio.get_event_loop()

    def _encode_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous encoding (runs in executor)."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts asynchronously."""
        if not texts:
            return []
        embeddings = await self._loop.run_in_executor(
            None, self._encode_sync, texts
        )
        return embeddings.tolist()

    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        embeddings = await self.embed([text])
        return embeddings[0]

    @property
    def dimension(self) -> int:
        """Return vector dimension (384 for all-MiniLM-L6-v2)."""
        return self.model.get_sentence_embedding_dimension()


@lru_cache()
def get_embedder() -> EmbeddingService:
    return EmbeddingService()
