# Upgrading Mnemosyne

## Upgrading from v2.0 to v2.1

### TL;DR

v2.1 adds a **compressed knowledge index** (`mnemosyne_index`), an **auto-context endpoint** (`POST /auto-context`), a **health endpoint** (`GET /health`), and a **retrieval evaluation benchmark**. All v2.0 features keep working — these are additive.

### What's New

| Feature | Description |
|---------|-------------|
| `mnemosyne_index` tool | Generates a compressed ~800-token structural map of your memory store |
| `include_index` in bootstrap | Pass `include_index: true` to get the knowledge index inline with bootstrap |
| `POST /auto-context` | Pre-message memory injection endpoint for agent frameworks |
| `GET /health` | Liveness check endpoint |
| `server/eval/evaluate.py` | Benchmark suite comparing retrieval methods |

### Upgrade Steps

1. **Pull latest code** and rebuild:
   ```bash
   cd mnemosyne && git pull
   cd server && docker compose down && docker compose up -d --build
   ```

2. **Restart VS Code MCP connection** — the new `mnemosyne_index` tool won't appear until VS Code reconnects. Open Command Palette → "MCP: List Servers" → restart ↻. You should now see 10 tools (was 9 in v2.0).

3. **Try the knowledge index**:
   ```
   Use mnemosyne_index with workspace_hint "my-project"
   ```

4. **Try auto-context** (for agent framework integration):
   ```bash
   curl -X POST http://localhost:8010/auto-context \
     -H "Content-Type: application/json" \
     -d '{"text": "How do I deploy?", "limit": 5, "min_score": 0.3}'
   ```

5. **Run the eval benchmark** (optional, requires populated memory store):
   ```bash
   cd server && python eval/evaluate.py --quick
   ```

No configuration changes or data migration needed.

---

# Upgrading from Mnemosyne v1 to v2

## TL;DR

v2 adds **RAG** (vector search, document ingestion, LLM-powered Q&A).
All v1 features keep working — RAG is additive and optional.

**Upgrade in 6 steps:**
1. Pull latest code
2. Pull the Ollama embedding model
3. Edit `.env` on the server
4. Rebuild & restart the Docker containers
5. Restart the VS Code MCP connection
6. Backfill embeddings for existing memories

---

## What Changes

### Without Ollama (safe default)

All 6 original tools work identically to v1 — nothing breaks.

### With Ollama

Three new tools and two new search modes appear:

| Tool | Description |
|------|-------------|
| `mnemosyne_search` (method=`semantic`) | Vector similarity search |
| `mnemosyne_search` (method=`hybrid`) | Keyword + vector with Reciprocal Rank Fusion |
| `mnemosyne_ingest` | Chunk, embed, and store documents for retrieval |
| `mnemosyne_ask` | Retrieve context → generate answer with LLM |
| `mnemosyne_backfill_embeddings` | Vectorize existing v1 memories |

---

## Step-by-Step Upgrade

### Step 1 — Pull the latest code

```bash
cd mnemosyne
git pull
```

### Step 2 — Pull the Ollama embedding model

The embedding model **must** be pulled before the server can generate vectors.
Without it, every embedding call returns 404 and all RAG features silently fail.

```bash
# Required (~275 MB) — generates 768-dim vectors for search
ollama pull nomic-embed-text

# Optional — only needed for mnemosyne_ask (LLM answer generation)
# Pick ONE based on your hardware:
ollama pull qwen2.5-coder:7b    # ~4.7 GB, fast, good for most uses
ollama pull qwen2.5-coder:14b   # ~9 GB, better quality
ollama pull qwen2.5-coder:32b   # ~18 GB, best quality, slow on CPU
```

> **Tip:** `qwen2.5-coder:7b` is recommended unless you have a powerful GPU.
> The 32b model can time out on slower hardware.

Verify the model is available:

```bash
ollama list   # should show nomic-embed-text in the output
```

### Step 3 — Configure the server `.env`

Edit the `.env` file **on the deployment target** (not your local machine).
If you use `deploy.ps1`, edit `server/.env` locally — it gets copied during deploy.

```ini
# --- RAG Configuration ---

# Ollama URL — point to wherever Ollama is running
OLLAMA_URL=http://localhost:11434
# If Ollama runs on a different machine:
# OLLAMA_URL=http://192.168.1.91:11434

# Models
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=qwen2.5-coder:7b

# Auto-embed every memory on write (recommended)
MNEMOSYNE_EMBED_ON_WRITE=1
```

> **Important:** If a `.env` file already exists on the remote server, it overrides
> the defaults in `docker-compose.yml`. Make sure to update **that** file,
> not just your local copy.

### Step 4 — Rebuild and restart the containers

```bash
cd server
docker compose down
docker compose up -d --build
```

Or, if you use the deploy script:

```powershell
.\deploy\deploy.ps1 -SshHost your-host -RemoteDir /path/to/mnemosyne
```

Check the server logs to confirm RAG is enabled:

```bash
docker logs --tail 5 mnemosyne-mcp
```

You should see:

```
Ollama reachable at http://...:11434 — RAG features enabled
Mnemosyne MCP server starting on 0.0.0.0:8010 (neo4j)
```

If you see `Ollama NOT reachable`, check that `OLLAMA_URL` is correct
and the Ollama server is running.

### Step 5 — Restart the VS Code MCP connection

**This step is easy to miss.** VS Code caches the tool list from the MCP server.
After upgrading, the new tools (`mnemosyne_ingest`, `mnemosyne_ask`,
`mnemosyne_backfill_embeddings`) and the `method` parameter on `mnemosyne_search`
**will not appear until VS Code reconnects**.

How to restart:

1. Open the Command Palette (`Ctrl+Shift+P`)
2. Run **"MCP: List Servers"**
3. Find the `mnemosyne` server and click the **restart** icon ↻

Verify by checking that the agent now sees all 10 tools (was 6 in v1).

### Step 6 — Backfill embeddings for existing memories

Your v1 memories have no vector embeddings. Backfill them so they appear
in semantic and hybrid searches.

Ask your AI agent:

```
Use mnemosyne_backfill_embeddings with limit 50
```

Repeat until it returns `"total": 0` (no items left).

Or via curl:

```bash
curl -s -X POST http://localhost:8010/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"mnemosyne_backfill_embeddings","arguments":{"limit":50}}}'
```

Expected output: `{"ok": true, "embedded": 50, "failed": 0, "total": 50}`

> Run this in batches of 50. Each item makes one Ollama call,
> so large batches take proportionally longer.

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model (768-dim) |
| `OLLAMA_CHAT_MODEL` | `qwen2.5-coder:7b` | LLM for `mnemosyne_ask` |
| `MNEMOSYNE_EMBED_ON_WRITE` | `0` | Auto-embed on write (`1` = on) |

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Backfill says `"failed": 5, "embedded": 0` | Embedding model not pulled | Run `ollama pull nomic-embed-text` |
| `mnemosyne_ask` returns "Failed to generate" | LLM timeout (model too large) | Switch to `qwen2.5-coder:7b` in `.env` and restart |
| New tools not visible in VS Code | MCP client cached old tool list | Restart the MCP server in VS Code (see Step 5) |
| Server log: `404 Not Found` for `/api/embeddings` | Model not pulled on Ollama | Run `ollama pull nomic-embed-text` and verify with `ollama list` |
| Remote `.env` overrides your changes | Old `.env` on server has stale values | SSH to server, edit the `.env` there, then restart container |

---

## Rollback

v2 is fully backward-compatible. To disable RAG:

```ini
# server/.env
MNEMOSYNE_EMBED_ON_WRITE=0
```

Restart, and Mnemosyne behaves identically to v1. Embeddings already stored are simply ignored.
