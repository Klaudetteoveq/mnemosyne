"""
LLM client for Mnemosyne RAG generation.

Calls Ollama's /api/generate endpoint for answer synthesis.
Zero additional Python dependencies — uses httpx (already in requirements).

Configuration:
    OLLAMA_URL          - Ollama server URL (default: http://localhost:11434)
    OLLAMA_CHAT_MODEL   - Chat/generation model (default: qwen2.5-coder:32b)
    MNEMOSYNE_MOCK_RAG  - when set (1/true), return deterministic mock answers if Ollama is unavailable
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5-coder:32b")
GENERATE_TIMEOUT = 300.0
USE_MOCK = os.environ.get("MNEMOSYNE_MOCK_RAG", "0").strip() in ("1", "true", "yes")

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

    When Ollama is unavailable and MNEMOSYNE_MOCK_RAG is set, returns a deterministic
    synthesized answer built from context item titles / snippets for testing.
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

    prompt = f"""Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"""

    # If Ollama is unreachable but mocking enabled, synthesize a simple answer
    if USE_MOCK:
        # Compose a concise answer that quotes the top context items and lists sources.
        if not context_items:
            return "I don't have enough context to answer this question."
        top = context_items[:min(4, len(context_items))]
        lines = []
        # Simple heuristic: if any context item contains the exact question terms, prefer it
        qwords = set(question.lower().split())
        best = None
        for it in top:
            txt = (it.get('content') or '').lower()
            if any(w in txt for w in qwords):
                best = it; break
        if not best:
            best = top[0]
        answer_snippet = (best.get('content') or best.get('content_compact') or '')[:400]
        sources = [it.get('title', '') for it in top]
        return f"{answer_snippet.strip()}\n\nSources: {', '.join(sources)}"

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
        # fallback to a simple synthesized answer when mocking is enabled
        if USE_MOCK:
            if not context_items:
                return "I don't have enough context to answer this question."
            top = context_items[:min(4, len(context_items))]
            answer_snippet = (top[0].get('content') or top[0].get('content_compact') or '')[:400]
            sources = [it.get('title', '') for it in top]
            return f"{answer_snippet.strip()}\n\nSources: {', '.join(sources)}"
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

    When mocking is enabled this produces a naive score based on token overlap.
    """
    if not items:
        return items

    if USE_MOCK:
        qset = set(question.lower().split())
        for it in items:
            txt = (it.get('content') or '').lower()
            overlap = len(qset.intersection(set(txt.split())))
            it['_rerank_score'] = overlap
        return sorted(items, key=lambda x: x.get('_rerank_score', 0), reverse=True)

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
