# Mnemosyne RAG Upgrade Plan

## Overview

Upgrade Mnemosyne from a memory store with keyword retrieval into a full-fledged
Retrieval-Augmented Generation (RAG) system. All changes are additive — existing
clients and tools keep working with zero breaking changes.

## Infrastructure

- **Embeddings**: Ollama `nomic-embed-text` (local or remote)
- **Generation**: Ollama `qwen2.5-coder:32b` (local or remote)
- **Vector storage**: Neo4j 5 native vector indexes
- **Zero new Python dependencies** — all LLM/embedding calls via httpx

## Phase 1: Vector Embeddings + Semantic Search

The single most impactful change. Adds vector representations so semantic search
works even when exact keywords don't appear.

### Changes

1. **Embedding client** (`server/app/embedding.py`)
   - Async client that calls Ollama `/api/embeddings` endpoint
   - Model: `nomic-embed-text` (768 dimensions)
   - Configurable via `OLLAMA_URL` env var
   - Batch embedding support for backfill

2. **Neo4j vector index**
   ```cypher
   CREATE VECTOR INDEX memory_embedding IF NOT EXISTS
   FOR (m:MemoryItem)
   ON (m.embedding)
   OPTIONS {indexConfig: {
     `vector.dimensions`: 768,
     `vector.similarity_function`: 'cosine'
   }}
   ```

3. **Embed on write** — After MERGE in `neo4j_storage.py`, compute embedding
   of `title + " " + content_compact` and store as `m.embedding`.

4. **Vector search** — Add `vector_search()` method to storage layer.

5. **Backfill** — Admin tool to embed all existing memories that lack embeddings.

## Phase 2: Hybrid Search (Keyword + Vector + Graph)

Combine keyword fulltext, vector similarity, and graph traversal.

### Changes

1. **Reciprocal Rank Fusion (RRF)** — Merge keyword + vector results:
   ```
   rrf_score(d) = sum(1 / (k + rank_i(d))) for each retriever i
   ```

2. **Graph-augmented retrieval** — After initial fetch, expand 1-hop:
   - Related items via `RELATES_TO`
   - Items with shared tags via `TAGGED_WITH`
   - Items from same session via `DECIDED_IN`

3. **Upgrade `mnemosyne_search`** — Add `method` parameter:
   - `keyword` — existing fulltext search
   - `semantic` — vector-only search
   - `hybrid` — RRF fusion (new default)

## Phase 3: Document Ingestion Pipeline

Support ingesting larger documents, chunking them, and making them searchable.

### Schema

```
(:Document {title, source, mime_type, ingested_at, workspace_hint, space_id})
  -[:HAS_CHUNK]-> (:Chunk {content, embedding, position, token_count})
(:Document) -[:IN_WORKSPACE]-> (:Workspace)
```

### Changes

1. **Chunking** — Recursive character splitter:
   - Default chunk size: 512 tokens (~2048 chars)
   - Overlap: 50 tokens (~200 chars)
   - Preserve markdown headers as metadata

2. **New tool: `mnemosyne_ingest`**
   - Input: `{title, content, source?, mime_type?, workspace_hint?, chunk_size?, chunk_overlap?}`
   - Pipeline: parse → chunk → embed each chunk → store Document + Chunks
   - Returns: `{ok, document_id, chunk_count}`

3. **Search over chunks** — Vector search returns chunks, linked back to
   source documents. Include surrounding chunks for coherence.

## Phase 4: Generation Layer

The "G" in RAG — synthesize answers from retrieved context.

### Changes

1. **LLM client** (`server/app/llm.py`)
   - Async client for Ollama `/api/generate` endpoint
   - Model: configurable via `OLLAMA_CHAT_MODEL` (default `qwen2.5-coder:32b`)

2. **New tool: `mnemosyne_ask`**
   - Input: `{question, workspace_hint?, max_context_items?, method?}`
   - Pipeline: embed question → hybrid search → assemble context → generate → return
   - Output: `{answer, sources: [{id, title, score}]}`

3. **Prompt template** — System prompt instructs LLM to:
   - Answer based only on provided context
   - Cite source memories by title
   - Say "I don't have enough context" when appropriate

## Phase 5: Advanced Enhancements

1. **Reranking** — Score-based reranking of hybrid results using
   LLM-as-judge or lightweight heuristic.
2. **Query decomposition** — Break complex questions into sub-queries.
3. **Agentic RAG** — Let LLM request additional retrieval in a loop.
4. ~~**Evaluation** — Automated quality metrics for retrieval + generation.~~ ✅ Implemented in `server/eval/evaluate.py` — benchmarks keyword, semantic, hybrid, and auto-context methods with direct recall, cross-reference, and negative test categories.
5. **Caching** — Cache embeddings and frequent queries.
6. **Compressed Knowledge Index** ✅ — `mnemosyne_index` tool generates a structural map of the memory store for cross-reference improvement.
7. **Auto-Context Endpoint** ✅ — `POST /auto-context` for pre-message memory injection in agent frameworks.

## Configuration (Environment Variables)

```
# Existing
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
MNEMOSYNE_BIND, MNEMOSYNE_PORT, MNEMOSYNE_MULTI_TENANT

# New for RAG
OLLAMA_URL=http://localhost:11434    # Ollama server URL
OLLAMA_EMBED_MODEL=nomic-embed-text  # Embedding model
OLLAMA_CHAT_MODEL=qwen2.5-coder:32b # Generation model
MNEMOSYNE_EMBED_ON_WRITE=1          # Enable auto-embedding (default: 1)
MNEMOSYNE_DEFAULT_SEARCH=hybrid     # Default search method
```

## Migration Strategy

- All changes are additive — no breaking changes
- `mnemosyne_search` gains optional `method` param (default "hybrid", falls back to "keyword" if no embeddings)
- New tools are purely additive
- Existing memories get embeddings via backfill
- Old clients keep working with zero changes
