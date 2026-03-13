# Mnemosyne Retrieval Evaluation

Benchmark suite for measuring retrieval quality across search methods.

## Methodology

Compares three retrieval approaches:
- **Mode A (Keyword):** Fulltext search only
- **Mode B (Semantic):** Vector similarity only
- **Mode C (Hybrid):** Keyword + vector with Reciprocal Rank Fusion (default)

### Test Categories

- **Direct recall:** Single-fact retrieval ("What is X?")
- **Cross-reference:** Multi-domain connections ("How does X relate to Y?")
- **Negative:** Questions about facts NOT in memory (tests false-positive rejection)

### Scoring

Each test case is scored on recall (0–100%):
- What fraction of expected key facts appear in the retrieved context?
- Cross-references: are both domains connected?
- Negatives: are irrelevant results correctly excluded?

## Running

```bash
# Full evaluation (requires running Mnemosyne + Neo4j + Ollama)
python eval/evaluate.py

# Quick smoke test (fewer cases, faster)
python eval/evaluate.py --quick

# Keyword-only (no Ollama required)
python eval/evaluate.py --method keyword
```

### Environment Variables

- `MNEMOSYNE_URL` — Server endpoint (default: `http://localhost:8010`)
- `MNEMOSYNE_METHOD` — Default search method: `keyword`, `semantic`, `hybrid`

## Limitations

- Test cases are generated from the live memory store (potential overfitting)
- Hardware-specific latency numbers
- Single-run evaluation (no confidence intervals)
- Requires populated memory store for meaningful results
