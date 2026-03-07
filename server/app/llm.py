"""
LLM client for Mnemosyne RAG generation.

Calls Ollama's /api/generate endpoint for answer synthesis.
Zero additional Python dependencies — uses httpx (already in requirements).

Configuration:
    OLLAMA_URL          - Ollama server URL (default: http://localhost:11434)
    OLLAMA_CHAT_MODEL   - Chat/generation model (default: qwen2.5-coder:32b)
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5-coder:32b")
GENERATE_TIMEOUT = 120.0

SYSTEM_PROMPT = """You are Mnemosyne, a knowledge assistant. Answer questions based ONLY on the provided context.

Rules:
- Only use information from the provided context to answer.
- Cite sources by their title when referencing specific memories.
- If the context doesn't contain enough information to answer, say "I don't have enough context to answer this question."
- Be concise and direct.
- Do not make up information not in the context."""


async def generate_answer(
    question: str,
    context_items: list[dict],
    model: str | None = None,
) -> str | None:
    """Generate an answer from retrieved context using Ollama.

    Args:
        question: The user's question.
        context_items: List of memory items with 'title' and 'content' keys.
        model: Override the default chat model.

    Returns:
        Generated answer string, or None on failure.
    """
    model = model or OLLAMA_CHAT_MODEL

    # Build context block
    context_parts = []
    for i, item in enumerate(context_items, 1):
        title = item.get("title", "Untitled")
        kind = item.get("kind", "note")
        content = item.get("content", "")
        context_parts.append(f"[{i}] ({kind}) {title}\n{content}")

    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""Context:
{context_block}

Question: {question}

Answer:"""

    try:
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response")
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return None


RERANK_PROMPT = """Rate the relevance of each document to the question on a scale of 0-10.
Return ONLY a JSON array of scores in the same order as the documents.

Question: {question}

Documents:
{documents}

Return a JSON array of integer scores, e.g. [8, 3, 9, 1]. Nothing else."""


async def rerank_items(
    question: str,
    items: list[dict],
    model: str | None = None,
) -> list[dict]:
    """Rerank items using LLM-as-judge scoring.

    Returns items sorted by relevance (highest first), with _rerank_score added.
    Falls back to original order on failure.
    """
    if not items:
        return items

    model = model or OLLAMA_CHAT_MODEL

    docs_text = "\n".join(
        f"[{i}] {item.get('title', '')} — {(item.get('content', '') or '')[:200]}"
        for i, item in enumerate(items)
    )

    prompt = RERANK_PROMPT.format(question=question, documents=docs_text)

    try:
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "").strip()
            # Extract JSON array from response
            import json
            # Find the first [...] in the response
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end > start:
                scores = json.loads(raw[start:end + 1])
                if isinstance(scores, list) and len(scores) == len(items):
                    for item, score in zip(items, scores):
                        item["_rerank_score"] = int(score) if isinstance(score, (int, float)) else 0
                    return sorted(items, key=lambda x: x.get("_rerank_score", 0), reverse=True)
            logger.warning("Rerank: could not parse scores from: %s", raw[:200])
            return items
    except Exception as e:
        logger.warning("Rerank failed, keeping original order: %s", e)
        return items
