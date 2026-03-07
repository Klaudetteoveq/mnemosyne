# Upgrading from Mnemosyne v1 to v2

## What Changed

v2 adds **RAG (Retrieval-Augmented Generation)** capabilities on top of v1. All v1 features continue to work **exactly as before** — RAG is purely additive and **optional**.

### Without Ollama (default)

All 6 original tools work identically to v1:

| Tool | Status |
|------|--------|
| `mnemosyne_bootstrap` | Works as before |
| `mnemosyne_write` | Works as before |
| `mnemosyne_read` | Works as before |
| `mnemosyne_search` | Works as before (keyword fulltext) |
| `mnemosyne_commit_session` | Works as before |
| `mnemosyne_last_session` | Works as before |

### With Ollama

Three new tools become available, and search gains semantic/hybrid modes:

| Tool | Requires Ollama | Description |
|------|----------------|-------------|
| `mnemosyne_search` (method=hybrid) | Yes | Keyword + vector similarity with Reciprocal Rank Fusion |
| `mnemosyne_search` (method=semantic) | Yes | Vector similarity only |
| `mnemosyne_ingest` | Yes | Chunk, embed, and store documents for retrieval |
| `mnemosyne_ask` | Yes | RAG pipeline: retrieve context → generate answer with citations |
| `mnemosyne_backfill_embeddings` | Yes | Vectorize existing v1 memories for semantic search |

---

## Upgrade Steps

### Step 1: Pull the latest code

```bash
cd mnemosyne
git pull
```

### Step 2: Rebuild and restart

```bash
cd server
docker compose down
docker compose up -d --build
```

That's it — you're running v2. Neo4j vector indexes are created automatically on startup. Your existing memories and sessions are untouched.

At startup you'll see one of:

```
Ollama NOT reachable — RAG features disabled. Core memory tools work normally.
```
```
Ollama reachable at http://localhost:11434 — RAG features enabled
```

---

## Enabling RAG (optional)

### Step 3: Install Ollama

Download from [ollama.com](https://ollama.com/) or:

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download installer from https://ollama.com/download/windows
# macOS — download from https://ollama.com/download/mac
```

### Step 4: Pull the models

```bash
# Embedding model (required for all RAG features, ~275 MB)
ollama pull nomic-embed-text

# Generation model (required for mnemosyne_ask, large — ~18 GB)
ollama pull qwen2.5-coder:32b
```

> **Smaller alternative:** If you don't need `mnemosyne_ask`, you only need the embedding model. If you want a smaller generation model, set `OLLAMA_CHAT_MODEL` to any model you have (e.g. `qwen2.5-coder:7b`).

### Step 5: Configure and restart

Create or edit `server/.env`:

```ini
# Enable auto-embedding when writing memories
MNEMOSYNE_EMBED_ON_WRITE=1

# Ollama URL (default: http://localhost:11434)
# If Ollama runs on a different machine:
# OLLAMA_URL=http://192.168.1.100:11434

# Models (defaults shown — change to match what you pulled)
# OLLAMA_EMBED_MODEL=nomic-embed-text
# OLLAMA_CHAT_MODEL=qwen2.5-coder:32b
```

Restart:

```bash
cd server
docker compose down
docker compose up -d --build
```

You should now see:

```
Ollama reachable at http://localhost:11434 — RAG features enabled
```

### Step 6: Backfill existing memories (optional)

Your v1 memories don't have embeddings yet. To vectorize them for semantic search, call the backfill tool from your AI agent:

```
Use mnemosyne_backfill_embeddings with limit 100
```

Or via curl:

```bash
curl -X POST http://localhost:8010/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"mnemosyne_backfill_embeddings","arguments":{"limit":100}}}'
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `OLLAMA_CHAT_MODEL` | `qwen2.5-coder:32b` | Generation model name |
| `MNEMOSYNE_EMBED_ON_WRITE` | `0` | Auto-embed memories on write (`1` to enable) |

---

## Rollback

If anything goes wrong, v2 is fully backward-compatible. To disable RAG:

```ini
# server/.env
MNEMOSYNE_EMBED_ON_WRITE=0
```

Restart, and Mnemosyne behaves identically to v1. Embeddings already stored are simply ignored.
