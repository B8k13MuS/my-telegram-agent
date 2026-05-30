"""RAG (Retrieval-Augmented Generation) pipeline.

Orchestrates: embedding -> vector search -> LLM generation.
Keeps conversation history per user for context-aware responses.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.vector_store import VectorStore


@dataclass
class RAGResult:
    """RAG query result."""
    answer: str
    sources: List[Dict] = field(default_factory=list)
    has_context: bool = True


class RAGPipeline:
    """End-to-end RAG pipeline."""

    def __init__(self):
        self.embedder = EmbeddingService()
        self.vector = VectorStore(vector_size=self.embedder.dimension)
        self.llm = LLMService()
        # In-memory conversation history: {user_id: [messages]}
        self._history: Dict[str, List[Dict[str, str]]] = {}

    async def query(
        self,
        question: str,
        user_id: str = "default",
        category: Optional[str] = None,
        top_k: int = 4,
    ) -> RAGResult:
        """Execute full RAG pipeline."""
        # 1. Embed query
        query_vector = await self.embedder.embed_query(question)

        # 2. Retrieve relevant documents
        docs = self.vector.search(
            query_vector=query_vector,
            limit=top_k,
            category=category,
        )

        if not docs:
            # No relevant docs found
            return RAGResult(
                answer=(
                    "I couldn't find relevant information in the documentation. "
                    "Try rephrasing your question or check if the documents are indexed."
                ),
                sources=[],
                has_context=False,
            )

        # 3. Build context from retrieved docs
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}: {doc['title']}]\n{doc['text']}")
        context = "\n\n---\n\n".join(context_parts)

        # 4. Get conversation history
        history = self._history.get(user_id, [])

        # 5. Generate answer via LLM
        response = await self.llm.generate_with_context(
            question=question,
            context=context,
            history=history,
        )

        # 6. Store in history
        self._add_to_history(user_id, question, response.text)

        return RAGResult(
            answer=response.text,
            sources=[
                {"title": d["title"], "score": round(d["score"], 3)}
                for d in docs
            ],
        )

    def _add_to_history(self, user_id: str, question: str, answer: str) -> None:
        """Add Q&A pair to conversation history."""
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append({"role": "user", "content": question})
        self._history[user_id].append({"role": "assistant", "content": answer})
        # Keep last 10 messages (5 Q&A pairs)
        if len(self._history[user_id]) > 10:
            self._history[user_id] = self._history[user_id][-10:]

    def clear_history(self, user_id: str) -> None:
        """Clear conversation history for a user."""
        self._history.pop(user_id, None)

    def stats(self) -> Dict:
        """Return RAG system statistics."""
        return {
            "documents_indexed": self.vector.count(),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": self.embedder.dimension,
            "llm_model": self.llm.model,
            "active_conversations": len(self._history),
        }
