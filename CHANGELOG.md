# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-03-07

### Added

- **Vector Embeddings** — Automatic embedding on write via Ollama (`nomic-embed-text`), stored in Neo4j native vector indexes
- **Hybrid Search** — `mnemosyne_search` gains `method` parameter: `keyword` (fulltext), `semantic` (vector), `hybrid` (Reciprocal Rank Fusion of both)
- **Document Ingestion** — New `mnemosyne_ingest` tool: chunk, embed, and store documents for RAG retrieval
- **Question Answering** — New `mnemosyne_ask` tool: retrieve context → generate answer with citations via Ollama LLM
- **LLM Reranking** — Optional LLM-as-judge reranking for improved search precision
- **Backfill** — New `mnemosyne_backfill_embeddings` tool to vectorize existing memories
- **Graph-Augmented Retrieval** — `graph_expand` method traverses RELATES_TO, shared tags, and session links
- **Chunking module** — Recursive character splitter for document ingestion (`server/app/chunking.py`)
- **Embedding client** — Async Ollama embedding client (`server/app/embedding.py`)
- **LLM client** — Async Ollama generation client (`server/app/llm.py`)
- **Startup health check** — Server logs Ollama reachability status at startup
- **Upgrade guide** — `docs/UPGRADING.md` for v1 → v2 migration

### Changed

- `MNEMOSYNE_EMBED_ON_WRITE` defaults to `0` (off) — server works without Ollama out of the box
- RAG tools (`mnemosyne_ingest`, `mnemosyne_ask`, `mnemosyne_backfill_embeddings`) return clear error messages when Ollama is unavailable
- `.env.example` now includes all RAG configuration variables
- Server version reported as `2.0.0` in MCP `initialize` response

### Fixed

- `graph_expand` Cypher query — Neo4j does not allow parameters in variable-length path patterns; now uses literal hop range

## [1.0.1] - 2026-02-14

### Added

- **Context Pollution Prevention** — Three-lever system (write-time hygiene, store-time structure, read-time shaping)
- **`mnemosyne_read` tool** — Retrieve a single memory item by ID with full or compact content
- **Bootstrap modes** — `thin`, `hybrid`, `full` with token budgeting and `max_items` limit
- **Auto-compact** — Server auto-generates compact summaries for long content
- **Ranking** — Bootstrap ranks items by `kind_weight × recency_decay × importance × workspace_match`
- **Content fields** — `content_compact`, `importance`, `workspace_hint`, `source` on memory items

### Changed

- `mnemosyne_write` accepts optional `content_compact`, `importance`, `workspace_hint`, `source`
- `mnemosyne_search` accepts optional `prefer` (compact/full) and `snippet_chars`
- `mnemosyne_bootstrap` accepts optional `mode`, `max_tokens`, `max_items`, `include_sessions`

### Fixed

- Defensive arguments guard — prevent 500 on malformed `tools/call` input
- `Join-Path` calls for PowerShell 5.1 compatibility in deploy scripts
- Legacy backward compatibility — default parameters match v1.0.0 behavior

## [1.0.0] - 2026-01-24

### Added

- Initial public release
- **5 MCP tools**: `mnemosyne_bootstrap`, `mnemosyne_write`, `mnemosyne_search`, `mnemosyne_commit_session`, `mnemosyne_last_session`
- Neo4j knowledge graph backend with fulltext search indexes
- VS Code extension with auto-bootstrap and auto-commit
- Stdio proxy (`mnemosyne_proxy.py`) for bridging stdio MCP to HTTP
- Docker Compose deployment (Neo4j + MCP server)
- PowerShell deployment and backup scripts
- GitHub Actions CI workflow

[2.0.0]: https://github.com/oveku/mnemosyne/compare/v1.0.1...v2.0.0
[1.0.1]: https://github.com/oveku/mnemosyne/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/oveku/mnemosyne/releases/tag/v1.0.0
