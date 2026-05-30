"""Qdrant vector store for document search.

Qdrant runs in Docker, stores vectors + metadata locally.
Free, open-source, no API limits.

Collection: business_docs
Vector size: 384 (from all-MiniLM-L6-v2)
Distance: Cosine similarity
"""

import uuid
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.core.config import settings


class VectorStore:
    """Qdrant vector store for business documentation."""

    def __init__(self, vector_size: int = 384):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            prefer_grpc=False,
        )
        self.collection = settings.QDRANT_COLLECTION
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it does not exist."""
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            # Index for filtering by category
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="category",
                field_schema="keyword",
            )
            print(f"Created Qdrant collection: {self.collection}")

    def upsert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Insert or update documents with embeddings."""
        ids = [str(uuid.uuid4()) for _ in texts]
        points = [
            PointStruct(
                id=doc_id,
                vector=embedding,
                payload={"text": text, **(meta or {})},
            )
            for doc_id, text, embedding, meta in zip(
                ids, texts, embeddings, metadata or [{}] * len(texts)
            )
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return ids

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search documents by vector similarity."""
        query_filter = None
        if category:
            query_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

        return [
            {
                "text": r.payload.get("text", ""),
                "title": r.payload.get("title", "Unknown"),
                "category": r.payload.get("category", ""),
                "score": r.score,
            }
            for r in results
        ]

    def delete_by_source(self, source_file: str) -> int:
        """Delete all chunks from a specific source file."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            ),
        )
        return 1

    def count(self) -> int:
        """Return total number of vectors in collection."""
        info = self.client.get_collection(self.collection)
        return info.points_count

    def clear_all(self) -> None:
        """Delete all documents from collection."""
        self.client.delete(collection_name=self.collection, points_selector={})
