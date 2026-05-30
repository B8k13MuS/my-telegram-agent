"""LLM service using Groq API (free tier).

Groq provides fast inference for open-source models.
- Free tier: 20 requests/minute, 1,500,000 tokens/day
- Models: llama3-8b, llama3-70b, mixtral-8x7b
- API is OpenAI-compatible — easy to switch providers later

Get API key: https://console.groq.com/keys

Fallback chain:
1. Groq API (free, fast)
2. If key missing or quota exceeded — use echo mode for testing
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import httpx

from app.core.config import settings


@dataclass
class LLMResponse:
    """Structured LLM response."""
    text: str
    model: str = ""
    usage: Dict[str, int] = None


class LLMService:
    """Generate text responses via Groq API."""

    SYSTEM_PROMPT = (
        "You are a helpful assistant for business process automation. "
        "Answer questions based on the provided context. "
        "If the context does not contain enough information, say so honestly. "
        "Always cite sources by referring to document titles. "
        "Answer in the same language as the user's question."
    )

    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_BASE_URL.rstrip("/")
        self.model = settings.LLM_MODEL
        self.timeout = 30.0

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate response from message history."""
        if not self.api_key:
            # Echo mode for testing without API key
            last_msg = messages[-1]["content"] if messages else ""
            return LLMResponse(
                text=f"[ECHO MODE — no LLM key configured]\n\nReceived: {last_msg[:200]}",
                model="echo",
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
        )

    async def generate_with_context(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> LLMResponse:
        """RAG-style generation with retrieved context."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "system",
                "content": f"Context from business documentation:\n{context}",
            },
        ]
        if history:
            messages.extend(history[-4:])  # Last 4 messages for context
        messages.append({"role": "user", "content": question})

        return await self.generate(messages)

    async def generate_simple(self, prompt: str) -> str:
        """Simple single-prompt generation."""
        response = await self.generate([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        return response.text
