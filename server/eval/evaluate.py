#!/usr/bin/env python3
"""
Mnemosyne Retrieval Evaluation Suite

Measures retrieval quality across keyword, semantic, and hybrid search.
Generates test cases from the live memory store, then evaluates recall
for each search method across three categories:
  - Direct recall: single-fact retrieval
  - Cross-reference: multi-domain connections
  - Negative: false-positive rejection

Inspired by zer0dex evaluation methodology.

Usage:
    python evaluate.py                     # Full eval, hybrid method
    python evaluate.py --quick             # Quick eval (10 cases)
    python evaluate.py --method keyword    # Keyword-only (no Ollama needed)
    python evaluate.py --method semantic   # Vector-only
    python evaluate.py --method all        # Compare all three methods

Requires:
    - Running Mnemosyne server (default: http://localhost:8010)
    - For semantic/hybrid: Ollama with nomic-embed-text
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error

MNEMOSYNE_URL = os.environ.get("MNEMOSYNE_URL", "http://localhost:8010")
TIMEOUT = 15


def mcp_call(tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool on the Mnemosyne server."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }).encode()
    req = urllib.request.Request(
        f"{MNEMOSYNE_URL}/mcp",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=TIMEOUT)
    data = json.loads(resp.read())
    text = data.get("result", {}).get("content", [{}])[0].get("text", "{}")
    return json.loads(text)


def auto_context_call(text: str, limit: int = 5) -> list[dict]:
    """Call the /auto-context endpoint."""
    payload = json.dumps({"text": text, "limit": limit}).encode()
    req = urllib.request.Request(
        f"{MNEMOSYNE_URL}/auto-context",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=TIMEOUT)
    data = json.loads(resp.read())
    return data.get("memories", [])


def server_reachable() -> bool:
    """Check if the Mnemosyne server is reachable."""
    try:
        urllib.request.urlopen(f"{MNEMOSYNE_URL}/health", timeout=3)
        return True
    except Exception:
        return False


def fetch_all_memories(limit: int = 200) -> list[dict]:
    """Fetch recent memories via bootstrap to use as test data source."""
    result = mcp_call("mnemosyne_bootstrap", {
        "limit_pinned": 25,
        "limit_recent": limit,
        "mode": "full",
        "max_items": limit,
    })
    items = result.get("pinned", []) + result.get("recent", [])
    # Deduplicate by id
    seen = set()
    unique = []
    for item in items:
        item_id = item.get("id", "")
        if item_id not in seen:
            seen.add(item_id)
            unique.append(item)
    return unique


def generate_test_cases(memories: list[dict], max_cases: int = 100) -> list[dict]:
    """Generate test cases from stored memories."""
    tests = []

    for mem in memories:
        title = mem.get("title", "")
        content = mem.get("content", "")
        kind = mem.get("kind", "note")
        if not title or len(title) < 5:
            continue

        # Extract key terms for expected facts
        words = title.split()
        key_fragments = []
        for word in words:
            word_clean = word.strip("[]():,.")
            if len(word_clean) > 3 and (word_clean[0].isupper() or any(c.isdigit() for c in word_clean)):
                key_fragments.append(word_clean)

        if not key_fragments:
            key_fragments = [words[0]] if words else []

        if not key_fragments:
            continue

        # Create a question from the title
        templates = [
            f"What do you know about {title}?",
            f"Tell me about {' '.join(words[:4])}",
            f"What is {' '.join(words[:3])}?",
        ]
        question = templates[hash(title) % len(templates)]

        tests.append({
            "question": question,
            "expected_facts": key_fragments[:4],
            "source_title": title,
            "type": "direct_recall",
            "source_kind": kind,
        })

    # Cross-reference: find items with shared tags
    tag_items: dict[str, list[dict]] = {}
    for mem in memories:
        tags_raw = mem.get("tags", "[]")
        if isinstance(tags_raw, str):
            try:
                tags = json.loads(tags_raw)
            except (json.JSONDecodeError, TypeError):
                tags = []
        else:
            tags = tags_raw
        for tag in tags:
            tag_items.setdefault(tag, []).append(mem)

    for tag, items in tag_items.items():
        if len(items) >= 2:
            a, b = items[0], items[1]
            a_title = a.get("title", "")
            b_title = b.get("title", "")
            # Extract a key word from each title
            a_key = next((w for w in a_title.split() if len(w) > 3), a_title[:10])
            b_key = next((w for w in b_title.split() if len(w) > 3), b_title[:10])
            tests.append({
                "question": f"How does {a_key} relate to {b_key}?",
                "expected_facts": [a_key, b_key],
                "source_title": f"{a_title} + {b_title}",
                "type": "cross_reference",
                "source_kind": "cross",
            })
            if len(tests) > max_cases:
                break

    # Negative cases (should return nothing relevant)
    negatives = [
        {"question": "What is the weather in Tokyo right now?", "expected_facts": [], "type": "negative"},
        {"question": "How do I cook pasta?", "expected_facts": [], "type": "negative"},
        {"question": "What is Bitcoin's current price?", "expected_facts": [], "type": "negative"},
        {"question": "Tell me about quantum computing at MIT", "expected_facts": [], "type": "negative"},
        {"question": "What is the population of Brazil?", "expected_facts": [], "type": "negative"},
    ]
    tests.extend(negatives)

    random.seed(42)
    random.shuffle(tests)
    return tests[:max_cases]


def score_retrieval(retrieved_text: str, expected_facts: list[str]) -> dict:
    """Score recall: what fraction of expected facts appear in retrieved text."""
    if not expected_facts:
        # Negative case: score 1.0 if retrieval is empty or very short
        is_clean = len(retrieved_text.strip()) < 50
        return {"recall": 1.0 if is_clean else 0.5, "type": "negative", "found": [], "missed": []}

    text_lower = retrieved_text.lower()
    found = [f for f in expected_facts if f.lower() in text_lower]
    recall = len(found) / len(expected_facts) if expected_facts else 0
    return {
        "recall": round(recall, 3),
        "found": found,
        "missed": [f for f in expected_facts if f.lower() not in text_lower],
    }


def run_search(question: str, method: str) -> tuple[str, float]:
    """Run a search query and return (retrieved_text, latency_ms)."""
    t0 = time.time()
    results = mcp_call("mnemosyne_search", {
        "query": question,
        "limit": 10,
        "method": method,
        "prefer": "full",
    })
    latency_ms = (time.time() - t0) * 1000

    if isinstance(results, list):
        text = "\n".join(r.get("title", "") + " " + r.get("content", "") for r in results)
    else:
        text = str(results)
    return text, latency_ms


def run_auto_context(question: str) -> tuple[str, float]:
    """Run auto-context query and return (retrieved_text, latency_ms)."""
    t0 = time.time()
    memories = auto_context_call(question, limit=5)
    latency_ms = (time.time() - t0) * 1000
    text = " ".join(m.get("text", "") for m in memories)
    return text, latency_ms


def run_eval(methods: list[str], test_cases: list[dict], include_auto_context: bool = True) -> dict:
    """Run the evaluation for given methods and return results."""
    results = {m: [] for m in methods}
    latencies = {m: [] for m in methods}
    by_type = {}

    if include_auto_context:
        results["auto_context"] = []
        latencies["auto_context"] = []

    for qtype in ("direct_recall", "cross_reference", "negative"):
        by_type[qtype] = {m: [] for m in methods}
        if include_auto_context:
            by_type[qtype]["auto_context"] = []

    total = len(test_cases)
    for i, tc in enumerate(test_cases, 1):
        q = tc["question"]
        expected = tc["expected_facts"]
        qtype = tc["type"]

        if i % 10 == 0 or i == 1:
            print(f"  [{i}/{total}] {qtype}: {q[:60]}...")

        for method in methods:
            text, lat = run_search(q, method)
            score = score_retrieval(text, expected)
            results[method].append(score["recall"])
            latencies[method].append(lat)
            by_type[qtype][method].append(score["recall"])

        if include_auto_context:
            text, lat = run_auto_context(q)
            score = score_retrieval(text, expected)
            results["auto_context"].append(score["recall"])
            latencies["auto_context"].append(lat)
            by_type[qtype]["auto_context"].append(score["recall"])

    return {"results": results, "latencies": latencies, "by_type": by_type}


def print_results(eval_data: dict, test_cases: list[dict]):
    """Print formatted evaluation results."""
    results = eval_data["results"]
    latencies = eval_data["latencies"]
    by_type = eval_data["by_type"]
    n = len(test_cases)

    type_counts = {}
    for tc in test_cases:
        t = tc["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n{'=' * 70}")
    print(f"MNEMOSYNE RETRIEVAL EVALUATION — n={n}")
    print(f"{'=' * 70}")
    print(f"  Direct recall:  {type_counts.get('direct_recall', 0)}")
    print(f"  Cross-reference: {type_counts.get('cross_reference', 0)}")
    print(f"  Negative:        {type_counts.get('negative', 0)}")

    all_methods = list(results.keys())
    for method in all_methods:
        scores = results[method]
        avg = sum(scores) / len(scores) if scores else 0
        passing = sum(1 for s in scores if s >= 0.75)
        avg_lat = sum(latencies[method]) / len(latencies[method]) if latencies[method] else 0

        label = method.replace("_", " ").title()
        print(f"\n  {label}:")
        print(f"    Avg recall:  {avg:.1%}")
        print(f"    ≥75% pass:   {passing}/{len(scores)} ({passing / len(scores):.0%})" if scores else "")
        print(f"    Avg latency: {avg_lat:.1f}ms")

    # By question type
    print(f"\n{'—' * 70}")
    print("BY QUESTION TYPE:")
    for qtype in ("direct_recall", "cross_reference", "negative"):
        if qtype not in by_type:
            continue
        print(f"\n  {qtype}:")
        for method in all_methods:
            scores = by_type[qtype].get(method, [])
            if scores:
                avg = sum(scores) / len(scores)
                print(f"    {method}: {avg:.1%} (n={len(scores)})")

    # Head-to-head comparison (if multiple methods)
    if len(all_methods) >= 2:
        print(f"\n{'=' * 70}")
        print("HEAD-TO-HEAD:")
        for i, m1 in enumerate(all_methods):
            for m2 in all_methods[i + 1:]:
                wins = ties = losses = 0
                for s1, s2 in zip(results[m1], results[m2]):
                    if s1 > s2:
                        wins += 1
                    elif s1 == s2:
                        ties += 1
                    else:
                        losses += 1
                print(f"  {m1} vs {m2}: {wins} wins, {ties} ties, {losses} losses")

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Mnemosyne Retrieval Evaluation")
    parser.add_argument("--method", default="all",
                        choices=["keyword", "semantic", "hybrid", "all"],
                        help="Search method to evaluate (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick eval with fewer test cases")
    parser.add_argument("--max-cases", type=int, default=0,
                        help="Max test cases (0 = auto)")
    parser.add_argument("--no-auto-context", action="store_true",
                        help="Skip auto-context evaluation")
    parser.add_argument("--url", default=None,
                        help="Mnemosyne server URL")
    args = parser.parse_args()

    if args.url:
        global MNEMOSYNE_URL
        MNEMOSYNE_URL = args.url

    if not server_reachable():
        print(f"Error: Mnemosyne server not reachable at {MNEMOSYNE_URL}")
        print("Start the server first, or use --url to specify the endpoint.")
        sys.exit(1)

    print(f"Mnemosyne Retrieval Evaluation")
    print(f"Server: {MNEMOSYNE_URL}")

    # Determine methods to test
    if args.method == "all":
        methods = ["keyword", "semantic", "hybrid"]
    else:
        methods = [args.method]

    # Fetch memories for test generation
    print("Fetching memories...")
    memories = fetch_all_memories(limit=200)
    print(f"  Found {len(memories)} memories")

    if len(memories) < 3:
        print("Error: Not enough memories to generate test cases. Seed some data first.")
        sys.exit(1)

    # Generate test cases
    max_cases = args.max_cases or (15 if args.quick else 100)
    test_cases = generate_test_cases(memories, max_cases=max_cases)
    print(f"  Generated {len(test_cases)} test cases")

    # Run evaluation
    print(f"\nEvaluating methods: {', '.join(methods)}")
    if not args.no_auto_context:
        print("  + auto-context endpoint")

    eval_data = run_eval(
        methods=methods,
        test_cases=test_cases,
        include_auto_context=not args.no_auto_context,
    )

    print_results(eval_data, test_cases)


if __name__ == "__main__":
    main()
