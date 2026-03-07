"""
Text chunking for Mnemosyne RAG document ingestion.

Recursive character splitter — zero external dependencies.
Splits text into overlapping chunks suitable for embedding and retrieval.
"""

import math

# Approximate chars per token (conservative)
CHARS_PER_TOKEN = 4

# Default separators, tried in order
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Split text into overlapping chunks.

    Args:
        text: The text to chunk.
        chunk_size: Target chunk size in tokens (~4 chars/token).
        chunk_overlap: Overlap between chunks in tokens.

    Returns:
        List of dicts with keys: content, position, token_count.
    """
    if not text or not text.strip():
        return []

    max_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = chunk_overlap * CHARS_PER_TOKEN

    pieces = _recursive_split(text.strip(), max_chars, SEPARATORS)

    # Merge small pieces and apply overlap
    chunks = []
    position = 0

    i = 0
    while i < len(pieces):
        # Accumulate pieces up to max_chars
        current = pieces[i]
        i += 1

        while i < len(pieces) and len(current) + len(pieces[i]) + 1 <= max_chars:
            current = current + " " + pieces[i]
            i += 1

        token_count = math.ceil(len(current) / CHARS_PER_TOKEN)
        chunks.append({
            "content": current.strip(),
            "position": position,
            "token_count": token_count,
        })
        position += 1

        # Apply overlap: back up by overlap_chars worth of text
        if i < len(pieces) and overlap_chars > 0:
            overlap_text = current[-overlap_chars:] if len(current) > overlap_chars else current
            # Prepend overlap to the next iteration
            pieces[i] = overlap_text.strip() + " " + pieces[i]

    return chunks


def _recursive_split(text: str, max_chars: int, separators: list[str]) -> list[str]:
    """Recursively split text using a hierarchy of separators."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    if not separators:
        # Last resort: hard split at max_chars
        parts = []
        for start in range(0, len(text), max_chars):
            piece = text[start:start + max_chars].strip()
            if piece:
                parts.append(piece)
        return parts

    sep = separators[0]
    remaining_seps = separators[1:]

    splits = text.split(sep)
    results = []
    current = ""

    for split in splits:
        candidate = current + sep + split if current else split
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                results.append(current)
            # If this single split is too large, recurse with finer separators
            if len(split) > max_chars:
                results.extend(_recursive_split(split, max_chars, remaining_seps))
                current = ""
            else:
                current = split

    if current:
        results.append(current)

    return results
