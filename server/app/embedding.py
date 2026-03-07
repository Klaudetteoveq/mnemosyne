"""
Embedding client for Mnemosyne RAG.

Calls Ollama's /api/embeddings endpoint to generate vector embeddings.
Zero additional Python dependencies — uses httpx (already in requirements).

Configuration:
    OLLAMA_URL          - Ollama server URL (default: http://localhost:11434)
    OLLAMA_EMBED_MODEL  - Embedding model (default: nomic-embed-text)
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_TIMEOUT = 30.0


async def get_embedding(text: str) -> list[float] | None:
    """Get embedding vector for a single text string.

    Returns None if the embedding service is unavailable.
    """
    if not text or not text.strip():
        return None

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
