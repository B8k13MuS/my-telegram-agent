"""Security utilities."""

import hashlib
import hmac
import re
from typing import Optional


def sanitize_input(text: str, max_length: int = 4096) -> str:
    """Clean user input: remove control chars, limit length."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_length]


def hash_token(token: str) -> str:
    """Hash a token for safe comparison."""
    return hashlib.sha256(token.encode()).hexdigest()


def mask_secret(value: str) -> str:
    """Mask a secret for logging."""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


# Security headers for HTTP responses
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self';",
}
