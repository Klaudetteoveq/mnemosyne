import os
import asyncio

import pytest

os.environ['MNEMOSYNE_MOCK_RAG'] = '1'

from server.app import embedding as embed_mod
from server.app import llm as llm_mod

@pytest.mark.asyncio
async def test_mock_embedding_returns_vector():
    vec = await embed_mod.get_embedding('test embedding')
    assert isinstance(vec, list)
    assert len(vec) >= 16  # deterministic mock produces 64 in our implementation

@pytest.mark.asyncio
async def test_mock_generate_answer():
    items = [
        {'title': 'Doc1', 'kind': 'note', 'content': 'This is a test memory about cats.'},
        {'title': 'Doc2', 'kind': 'decision', 'content': 'We decided to adopt a cat.'},
    ]
    ans = await llm_mod.generate_answer('Tell me about cats', items)
    assert isinstance(ans, str)
    assert len(ans) > 0
