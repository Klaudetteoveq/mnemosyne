"""
Embedding client for Mnemosyne RAG.

Calls Ollama's /api/embeddings endpoint to generate vector embeddings.
Zero additional Python dependencies — uses httpx (already in requirements).

Configuration:
    OLLAMA_URL          - Ollama server URL (default: http://localhost:11434)
    OLLAMA_EMBED_MODEL  - Embedding model (default: nomic-embed-text)
    MNEMOSYNE_MOCK_RAG  - when set (1/true), return deterministic mock embeddings if Ollama is unavailable
"""

import logging
import os
import hashlib

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_TIMEOUT = 30.0
USE_MOCK = os.environ.get("MNEMOSYNE_MOCK_RAG", "0").strip() in ("1", "true", "yes")


def check_ollama_available() -> bool:
    """Synchronous check whether the Ollama server is reachable.

    If MNEMOSYNE_MOCK_RAG is set, the system treats Ollama as 'available' in the sense
    that the code will return mocked embeddings / answers for testing purposes.
    """
    if USE_MOCK:
        return True
    try:
        r = httpx.get(OLLAMA_URL, timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


async def get_embedding(text: str) -> list[float] | None:
    """Get embedding vector for a single text string.

    Returns None if the embedding service is unavailable and mocks are disabled.
    """
    if not text or not text.strip():
        return None

    # If Ollama is unreachable but mocking is enabled, return deterministic mock vector
    if not check_ollama_available() and USE_MOCK:
        # Deterministic pseudo-embedding: hash the input and expand to fixed dimensions
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # produce 64-d float vector in range [-1,1]
        vec = [((b / 255.0) * 2.0 - 1.0) for b in h[:64]]
        return vec

    try:
        async with httpx.AsyncClient(timeout=EMBED_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text.strip()},
            )
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if embedding and isinstance(embedding, list):
                return embedding
            logger.warning("Unexpected embedding response: %s", data)
            return None
    except Exception as e:
        logger.warning("Embedding request failed: %s", e)
        # If mocking enabled, return deterministic vector
        if USE_MOCK:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [((b / 255.0) * 2.0 - 1.0) for b in h[:64]]
            return vec
        return None


async def get_embeddings_batch(texts: list[str]) -> list[list[float] | None]:
    """Get embeddings for multiple texts.

    Processes sequentially to avoid overwhelming the Ollama server.
    Returns a list of embeddings (or None for failures) in the same order.
    """
    results = []
    for text in texts:
        embedding = await get_embedding(text)
        results.append(embedding)
    return results


def embedding_text_for_memory(title: str, content_compact: str | None = None) -> str:
    """Build the text string to embed for a memory item.

    Combines title and compact content for a balanced representation.
    """
    parts = [title.strip()]
    if content_compact and content_compact.strip():
        parts.append(content_compact.strip())
    return " ".join(parts)
