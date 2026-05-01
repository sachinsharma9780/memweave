"""
Memweave × LongMemEval Benchmark

Evaluates memweave retrieval recall on the LongMemEval dataset using the same
conditions as mempalace's published baseline:
  - Embedding model : all-MiniLM-L6-v2 (via Ollama)
  - Primary metric  : Recall@5 (any correct session in top-5)
  - Split           : 50 dev / 450 held-out (seed=42, lme_split_50_450.json)

Usage:
    # quick sanity check (10 questions from dev split)
    .venv/bin/python benchmarks/longmemeval_bench.py \\
        --dataset benchmarks/data/longmemeval/longmemeval_s_cleaned.json \\
        --split   benchmarks/data/lme_split_50_450.json \\
        --subset  dev --limit 10

    # full dev run
    .venv/bin/python benchmarks/longmemeval_bench.py \\
        --dataset benchmarks/data/longmemeval/longmemeval_s_cleaned.json \\
        --split   benchmarks/data/lme_split_50_450.json \\
        --subset  dev

    # held-out (run once, after dev is validated)
    .venv/bin/python benchmarks/longmemeval_bench.py \\
        --dataset benchmarks/data/longmemeval/longmemeval_s_cleaned.json \\
        --split   benchmarks/data/lme_split_50_450.json \\
        --subset  held_out
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from math import log2
from pathlib import Path

import litellm

litellm.suppress_debug_info = True

from memweave import MemWeave, MemoryConfig
from memweave.config import (
    CacheConfig,
    EmbeddingConfig,
    HybridConfig,
    MMRConfig,
    QueryConfig,
    SyncConfig,
)

# ── Constants ─────────────────────────────────────────────────────────────────

OLLAMA_MODEL = "ollama/all-minilm:l6-v2"
OLLAMA_BASE = "http://localhost:11434"

# Fetch enough results to cover all chunks in the workspace.
# Avg ~52 chunks per question (1.1 chunks/session × 47 sessions), max ~86.
_MAX_RESULTS = 200

_KS = [1, 3, 5, 10]


# ── Preflight ─────────────────────────────────────────────────────────────────


def check_ollama() -> None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
        models = [m["name"] for m in tags.get("models", [])]
    except urllib.error.URLError:
        raise SystemExit(
            f"Ollama not running at {OLLAMA_BASE}. Start Ollama and re-run."
        )

    if not any("all-minilm" in m for m in models):
        raise SystemExit(
            "all-minilm:l6-v2 not found in Ollama.\n"
            "Run:  ollama pull all-minilm:l6-v2"
        )


def warmup_ollama() -> None:
    """Send one embedding request so the model is loaded before timing starts."""
    payload = json.dumps(
        {"model": "all-minilm:l6-v2", "prompt": "warmup"}
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except Exception:
        pass  # non-fatal — timing will just be slightly off for Q1


# ── Data loading ──────────────────────────────────────────────────────────────


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_split(path: Path) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


def filter_questions(
    data: list[dict], split: dict[str, list[str]], subset: str
) -> list[dict]:
    if subset == "all":
        return data
    ids = set(split[subset])
    return [e for e in data if e["question_id"] in ids]


# ── Session formatting ────────────────────────────────────────────────────────


def write_sessions(workspace: Path, entry: dict) -> int:
    """Write each haystack session as {session_id}.md under workspace/memory/.

    Content: user turns only, matching mempalace raw mode exactly.
    Returns number of files written (sessions with at least one user turn).
    """
    memory_dir = workspace / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    dates = entry["haystack_dates"]

    written = 0
    for session, sid, date in zip(sessions, session_ids, dates):
        user_turns = [t["content"] for t in session if t["role"] == "user"]
        if not user_turns:
            continue
        content = f"# Session: {sid}\nDate: {date}\n\n" + "\n\n".join(user_turns)
        (memory_dir / f"{sid}.md").write_text(content, encoding="utf-8")
        written += 1

    return written


# ── MemWeave config ───────────────────────────────────────────────────────────


def make_config(workspace: Path, mode: str) -> MemoryConfig:
    """Build a MemoryConfig for the given benchmark mode."""
    shared = dict(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(
            model=OLLAMA_MODEL,
            api_base=OLLAMA_BASE,
        ),
        sync=SyncConfig(on_search=False),
        cache=CacheConfig(enabled=False),
        progress=False,
    )

    if mode == "raw":
        query = QueryConfig(
            max_results=_MAX_RESULTS,
            min_score=0.0,
            strategy="vector",
        )

    elif mode == "hybrid":
        query = QueryConfig(
            max_results=_MAX_RESULTS,
            min_score=0.0,
            strategy="hybrid",
            hybrid=HybridConfig(vector_weight=0.7, text_weight=0.3),
        )

    elif mode == "hybrid_mmr":
        query = QueryConfig(
            max_results=_MAX_RESULTS,
            min_score=0.0,
            strategy="hybrid",
            hybrid=HybridConfig(vector_weight=0.7, text_weight=0.3),
            mmr=MMRConfig(enabled=True, lambda_param=0.7),
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return MemoryConfig(**shared, query=query)


def search_strategy_for_mode(mode: str) -> str:
    if mode == "raw":
        return "vector"
    return "hybrid"


# ── Chunk → session mapping ───────────────────────────────────────────────────


def results_to_session_ranking(results: list) -> list[str]:
    """Deduplicate SearchResults by session ID (filename stem), preserving rank.

    SearchResult.path is relative, e.g. 'memory/sharegpt_abc_0.md'.
    Path(...).stem gives 'sharegpt_abc_0' which is the session ID.
    """
    seen: set[str] = set()
    ranked: list[str] = []
    for r in results:
        sid = Path(r.path).stem
        if sid not in seen:
            seen.add(sid)
            ranked.append(sid)
    return ranked


# ── Metrics ───────────────────────────────────────────────────────────────────


def recall_any_at_k(ranked: list[str], correct: set[str], k: int) -> float:
    return float(any(sid in correct for sid in ranked[:k]))


def recall_all_at_k(ranked: list[str], correct: set[str], k: int) -> float:
    top_k = set(ranked[:k])
    return float(all(cid in top_k for cid in correct))


def _dcg(relevances: list[float], k: int) -> float:
    return sum(r / log2(i + 2) for i, r in enumerate(relevances[:k]))


def ndcg_at_k(ranked: list[str], correct: set[str], k: int) -> float:
    relevances = [1.0 if sid in correct else 0.0 for sid in ranked[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg(ideal, k)
    return _dcg(relevances, k) / idcg if idcg > 0 else 0.0


# ── Per-question runner ───────────────────────────────────────────────────────


async def run_question(entry: dict, mode: str) -> dict:
    """Run a single question: index sessions → search → score. Returns a result row."""
    tmpdir = Path(tempfile.mkdtemp(prefix="mw_bench_"))
    try:
        workspace = tmpdir / "ws"
        write_sessions(workspace, entry)

        config = make_config(workspace, mode)
        strategy = search_strategy_for_mode(mode)

        t0 = time.monotonic()
        async with MemWeave(config) as mem:
            await mem.index()
            results = await mem.search(
                entry["question"],
                strategy=strategy,
                max_results=_MAX_RESULTS,
                min_score=0.0,
            )
        duration_ms = (time.monotonic() - t0) * 1000

        ranked = results_to_session_ranking(results)
        correct = set(entry["answer_session_ids"])

        row: dict = {
            "question_id": entry["question_id"],
            "question_type": entry["question_type"],
            "question": entry["question"],
            "answer": entry["answer"],
            "answer_session_ids": entry["answer_session_ids"],
            "ranked_session_ids": ranked[:50],  # cap stored list to top-50
            "num_sessions": len(entry["haystack_sessions"]),
            "num_results": len(results),
            "duration_ms": round(duration_ms),
        }
        for k in _KS:
            row[f"recall_any@{k}"] = recall_any_at_k(ranked, correct, k)
            row[f"recall_all@{k}"] = recall_all_at_k(ranked, correct, k)
            row[f"ndcg@{k}"] = round(ndcg_at_k(ranked, correct, k), 4)

        return row

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Aggregation ───────────────────────────────────────────────────────────────


def build_summary(rows: list[dict], mode: str, subset: str) -> dict:
    n = len(rows)
    per_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_type[r["question_type"]].append(r)

    summary: dict = {
        "__summary__": True,
        "mode": mode,
        "subset": subset,
        "n": n,
        "embedding_model": OLLAMA_MODEL,
        "avg_duration_ms": round(sum(r["duration_ms"] for r in rows) / n),
    }
    for k in _KS:
        summary[f"recall_any@{k}"] = round(
            sum(r[f"recall_any@{k}"] for r in rows) / n, 4
        )
        summary[f"recall_all@{k}"] = round(
            sum(r[f"recall_all@{k}"] for r in rows) / n, 4
        )
        summary[f"ndcg@{k}"] = round(sum(r[f"ndcg@{k}"] for r in rows) / n, 4)

    summary["per_type"] = {
        qtype: {
            "n": len(type_rows),
            "recall_any@5": round(
                sum(r["recall_any@5"] for r in type_rows) / len(type_rows), 4
            ),
            "recall_all@5": round(
                sum(r["recall_all@5"] for r in type_rows) / len(type_rows), 4
            ),
            "ndcg@5": round(
                sum(r["ndcg@5"] for r in type_rows) / len(type_rows), 4
            ),
        }
        for qtype, type_rows in sorted(per_type.items())
    }
    return summary


def print_summary(summary: dict, out_path: Path) -> None:
    sep = "─" * 65
    print(f"\n{sep}")
    print(f"  Results  →  {out_path}")
    print(sep)
    print(
        f"  {'Overall':<32}  "
        f"R@5={summary['recall_any@5']:.4f}  "
        f"R@10={summary['recall_any@10']:.4f}  "
        f"NDCG@5={summary['ndcg@5']:.4f}"
    )
    print()
    for qtype, stats in summary["per_type"].items():
        print(
            f"  {qtype:<32}  "
            f"R@5={stats['recall_any@5']:.4f}  "
            f"n={stats['n']}"
        )
    print(sep)
    print(f"  Avg time per question: {summary['avg_duration_ms']} ms")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memweave × LongMemEval Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to longmemeval_s_cleaned.json",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Path to lme_split_50_450.json",
    )
    parser.add_argument(
        "--mode",
        default="raw",
        choices=["raw", "hybrid", "hybrid_mmr"],
        help="Retrieval mode (default: raw)",
    )
    parser.add_argument(
        "--subset",
        default="dev",
        choices=["dev", "held_out", "all"],
        help="Which question split to run (default: dev)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N questions (for quick iteration)",
    )
    parser.add_argument(
        "--out",
        default="benchmarks/results",
        help="Output directory for JSONL results (default: benchmarks/results)",
    )
    args = parser.parse_args()

    check_ollama()
    print("Warming up Ollama model...", end=" ", flush=True)
    warmup_ollama()
    print("ready.\n")

    data = load_dataset(Path(args.dataset))
    split = load_split(Path(args.split))
    questions = filter_questions(data, split, args.subset)
    if args.limit:
        questions = questions[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"results_mw_{args.mode}_{args.subset}_{ts}.jsonl"

    print(
        f"Memweave × LongMemEval  |  "
        f"mode={args.mode}  subset={args.subset}  "
        f"n={len(questions)}  model={OLLAMA_MODEL}"
    )
    print(f"Output → {out_path}\n")

    rows: list[dict] = []
    with open(out_path, "w") as f:
        for i, entry in enumerate(questions, 1):
            row = await run_question(entry, args.mode)
            rows.append(row)

            hit = "✓" if row["recall_any@5"] == 1.0 else "✗"
            print(
                f"  [{i:>3}/{len(questions)}] {hit}  "
                f"{row['question_type']:<32}  "
                f"R@5={row['recall_any@5']:.0f}  "
                f"NDCG@5={row['ndcg@5']:.3f}  "
                f"({row['duration_ms']} ms)"
            )
            f.write(json.dumps(row) + "\n")
            f.flush()

        summary = build_summary(rows, args.mode, args.subset)
        f.write(json.dumps(summary) + "\n")

    print_summary(summary, out_path)


if __name__ == "__main__":
    asyncio.run(main())
