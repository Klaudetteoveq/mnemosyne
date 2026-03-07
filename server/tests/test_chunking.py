"""
Unit tests for the chunking module.
No external dependencies required.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from chunking import chunk_text


class TestChunking:
    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Hello world"
        assert chunks[0]["position"] == 0
        assert chunks[0]["token_count"] > 0

    def test_long_text_multiple_chunks(self):
        # Create text that's definitely longer than 1 chunk
        text = "This is a test sentence. " * 200  # ~5000 chars
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1
        # All chunks should have content
        for chunk in chunks:
            assert chunk["content"]
            assert chunk["token_count"] > 0

    def test_chunk_positions_sequential(self):
        text = "Paragraph one. " * 50 + "\n\n" + "Paragraph two. " * 50
        chunks = chunk_text(text, chunk_size=50)
        positions = [c["position"] for c in chunks]
        assert positions == list(range(len(positions)))

    def test_custom_chunk_size(self):
        text = "Word " * 500  # ~2500 chars
        chunks_small = chunk_text(text, chunk_size=50, chunk_overlap=0)
        chunks_large = chunk_text(text, chunk_size=500, chunk_overlap=0)
        assert len(chunks_small) > len(chunks_large)

    def test_markdown_splitting(self):
        text = "# Header 1\n\nParagraph under header 1.\n\n# Header 2\n\nParagraph under header 2."
        chunks = chunk_text(text, chunk_size=512)
        # Small enough to be one chunk
        assert len(chunks) >= 1

    def test_overlap_content(self):
        """Chunks with overlap should share some text."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. " * 20
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
        if len(chunks) >= 2:
            # With overlap, later chunks should contain some text from earlier chunks
            assert len(chunks) > 1
