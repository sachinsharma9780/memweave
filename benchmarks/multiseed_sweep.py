"""
Multi-seed cross-validated (CV) sweep — memweave × LongMemEval.

Runs 5 independent stratified train/eval splits to produce mean ± std
estimates of retrieval performance — more reliable than a single held-out
evaluation.

For each of 5 seeds:
  1. Cache raw vector-search results for all 50 dev questions (one embedding
     pass per question — no post-processors registered).
  2. Sweep 27 ECR × IDF × CAATB alpha combinations on the cached raw rows
     (offline, near-zero cost) and select the combination with the highest
     dev R@5.
  3. Evaluate the held-out 450 questions with the best alphas.
  4. Report per-seed metrics and mean ± std across all 5 seeds.

No data leakage:
  - Alpha selection uses only the dev set.  Held-out is never consulted.
  - Each seed tunes on its own dev set independently.
  - Raw rows are deterministic outputs of the embedding model; reusing them
    across alpha combos is equivalent to re-running 27 full pipelines.
  - Post-processors use dataclasses.replace() — they never mutate the cached
    raw rows, so each combo sees the same unmodified input.

Stratified splits: each split preserves question-type proportions using
largest-remainder rounding to hit exactly DEV_SIZE=50.

Seeds: 42, 0, 1, 2, 3.  Splits are saved to --splits-dir.

Usage:
    .venv/bin/python benchmarks/multiseed_sweep.py \\
        --dataset benchmarks/data/longmemeval/longmemeval_s_cleaned.json

Output:
    benchmarks/results/multiseed/
        results_mw_cv_seed{N}_held_out_{ts}.jsonl   (cv = cross-validated, one per seed)
        summary_mw_cv_{N}seeds_{ts}.json            (aggregate across all seeds)
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
from math import log2
import random
import shutil
import statistics
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import litellm
litellm.suppress_debug_info = True

from memweave import MemWeave, MemoryConfig
from memweave.config import CacheConfig, EmbeddingConfig, QueryConfig, SyncConfig
from strategies.caatb import ConfidenceAdaptiveTemporalBooster
from strategies.entity_confidence_reranker import EntityConfidenceReranker
from strategies.idf_keyword_boost import IDFKeywordBooster

# ── Sweep config ──────────────────────────────────────────────────────────────

SEEDS    = [42, 0, 1, 2, 3]
DEV_SIZE = 50
_KS      = [1, 3, 5, 10, 15, 25]

# Alpha search grids — 3 × 3 × 3 = 27 combinations
ECR_ALPHAS   = [0.2, 0.3, 0.4]
IDF_ALPHAS   = [0.4, 0.6, 0.8]
CAATB_ALPHAS = [0.1, 0.2, 0.3]
ALPHA_GRID   = list(itertools.product(ECR_ALPHAS, IDF_ALPHAS, CAATB_ALPHAS))

OLLAMA_MODEL = "ollama/all-minilm:l6-v2"
OLLAMA_BASE  = "http://localhost:11434"
_MAX_RESULTS = 200

# ── Ollama helpers ────────────────────────────────────────────────────────────


def check_ollama() -> None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
        models = [m["name"] for m in tags.get("models", [])]
    except urllib.error.URLError:
        raise SystemExit(f"Ollama not running at {OLLAMA_BASE}. Start Ollama and re-run.")
    if not any("all-minilm" in m for m in models):
        raise SystemExit("all-minilm:l6-v2 not found in Ollama.\nRun: ollama pull all-minilm:l6-v2")


def warmup_ollama() -> None:
    payload = json.dumps({"model": "all-minilm:l6-v2", "prompt": "warmup"}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except Exception:
        pass  # non-fatal


# ── Stratified split ──────────────────────────────────────────────────────────


def make_stratified_split(
    seed: int,
    data: list[dict],
    splits_dir: Path,
    dev_size: int = DEV_SIZE,
) -> dict:
    """Create or load a stratified DEV_SIZE/remainder split for the given seed.

    Stratification preserves question-type proportions using largest-remainder
    rounding so the dev set contains exactly dev_size questions.

    Splits are saved to splits_dir so re-runs produce identical results.
    """
    split_path = splits_dir / f"lme_stratified_split_seed{seed}.json"
    if split_path.exists():
        with open(split_path) as f:
            return json.load(f)

    by_type: dict[str, list[str]] = defaultdict(list)
    for entry in data:
        by_type[entry["question_type"]].append(entry["question_id"])

    total      = len(data)
    type_names = sorted(by_type.keys())  # sorted for reproducibility

    # Largest-remainder proportional allocation
    exact     = {t: len(by_type[t]) * dev_size / total for t in type_names}
    floors    = {t: int(v) for t, v in exact.items()}
    remainder = dev_size - sum(floors.values())
    extras    = set(
        sorted(type_names, key=lambda t: exact[t] - floors[t], reverse=True)[:remainder]
    )

    rng      = random.Random(seed)
    dev: list[str]      = []
    held_out: list[str] = []

    for t in type_names:
        ids = list(by_type[t])
        rng.shuffle(ids)
        n_dev = floors[t] + (1 if t in extras else 0)
        dev.extend(ids[:n_dev])
        held_out.extend(ids[n_dev:])

    split = {
        "seed":       seed,
        "dev":        dev,
        "held_out":   held_out,
        "dev_size":   len(dev),
        "total":      total,
        "stratified": True,
        "dev_per_type": {
            t: sum(1 for qid in dev if qid in set(by_type[t]))
            for t in type_names
        },
    }

    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"  Created stratified split: seed={seed} → {split_path}", flush=True)
    return split


# ── Session writing ───────────────────────────────────────────────────────────


def write_sessions(workspace: Path, entry: dict) -> None:
    memory_dir = workspace / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    for session, sid, date in zip(
        entry["haystack_sessions"],
        entry["haystack_session_ids"],
        entry["haystack_dates"],
    ):
        user_turns = [t["content"] for t in session if t["role"] == "user"]
        if not user_turns:
            continue
        content = f"# Session: {sid}\nDate: {date}\n\n" + "\n\n".join(user_turns)
        (memory_dir / f"{sid}.md").write_text(content, encoding="utf-8")


# ── MemWeave config ───────────────────────────────────────────────────────────


def make_config(workspace: Path) -> MemoryConfig:
    return MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=OLLAMA_MODEL, api_base=OLLAMA_BASE),
        query=QueryConfig(max_results=_MAX_RESULTS, min_score=0.0, strategy="vector"),
        sync=SyncConfig(on_search=False),
        cache=CacheConfig(enabled=False),
        progress=False,
    )


# ── Chunk → session mapping ───────────────────────────────────────────────────


def results_to_session_ranking(results: list) -> list[str]:
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


def ndcg_at_k(ranked: list[str], correct: set[str], k: int) -> float:
    relevances = [1.0 if sid in correct else 0.0 for sid in ranked[:k]]
    ideal = sorted(relevances, reverse=True)
    dcg  = sum(r / log2(i + 2) for i, r in enumerate(relevances))
    idcg = sum(r / log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ── Raw search ────────────────────────────────────────────────────────────────


class _RawCapture:
    """No-op post-processor that captures RawSearchRow objects.

    mem.search() converts RawSearchRow → SearchResult after post-processors run.
    Registering this as the only post-processor intercepts rows in their raw
    form (with .text and .vector_score intact) before that conversion.
    """

    def __init__(self) -> None:
        self.rows: list = []

    async def apply(self, rows: list, query: str, **kwargs: object) -> list:
        self.rows = list(rows)
        return rows


async def run_question_raw(entry: dict) -> tuple[list, str]:
    """Index and search, returning RawSearchRow objects (not SearchResult).

    Uses _RawCapture to intercept rows before mem.search() converts them to
    SearchResult.  Returns (raw_rows, question_date).  raw_rows are
    RawSearchRow dataclasses — safe after tmpdir cleanup.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="mw_raw_"))
    try:
        workspace = tmpdir / "ws"
        write_sessions(workspace, entry)
        config   = make_config(workspace)
        capture  = _RawCapture()
        async with MemWeave(config) as mem:
            mem.register_postprocessor(capture)
            await mem.index()
            await mem.search(
                entry["question"],
                strategy="vector",
                max_results=_MAX_RESULTS,
                min_score=0.0,
            )
        return capture.rows, entry.get("question_date", "")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Dev sweep (offline alpha search on cached raw rows) ───────────────────────


async def run_dev_sweep(
    dev_questions: list[dict],
    alpha_grid: list[tuple[float, float, float]],
) -> tuple[float, float, float, float]:
    """Cache raw rows for all dev questions, then sweep alpha combos offline.

    Returns (best_ecr_alpha, best_idf_alpha, best_caatb_alpha, best_dev_r5).

    Fairness guarantee: alpha selection is based solely on dev R@5.
    The held-out set is never accessed here.  Post-processors return new
    dataclass instances (dataclasses.replace), so the cached raw_rows list
    is never mutated between combos.
    """
    n = len(dev_questions)
    print(f"  [dev] Caching raw results for {n} questions ...", flush=True)

    # (raw_rows, question_date, query_text, correct_session_ids)
    dev_cache: list[tuple[list, str, str, set[str]]] = []
    for i, entry in enumerate(dev_questions, 1):
        raw_rows, qdate = await run_question_raw(entry)
        dev_cache.append((
            raw_rows,
            qdate,
            entry["question"],
            set(entry["answer_session_ids"]),
        ))
        print(f"  [dev] {i}/{n}", end="\r", flush=True)

    print(
        f"  [dev] {n}/{n} — cache ready.  "
        f"Sweeping {len(alpha_grid)} alpha combos ...\n",
        flush=True,
    )

    best_combo: tuple[float, float, float] = alpha_grid[0]
    best_r5 = -1.0

    for ecr_a, idf_a, caatb_a in alpha_grid:
        reranker    = EntityConfidenceReranker(alpha=ecr_a)
        idf_booster = IDFKeywordBooster(alpha=idf_a)
        caatb       = ConfidenceAdaptiveTemporalBooster(alpha=caatb_a)

        total_r5 = 0.0
        for raw_rows, qdate, query, correct in dev_cache:
            rows = await reranker.apply(raw_rows, query)
            rows = await idf_booster.apply(rows, query)
            rows = await caatb.apply(rows, query, question_date=qdate)
            ranked = results_to_session_ranking(rows)
            total_r5 += recall_any_at_k(ranked, correct, 5)

        r5     = total_r5 / n
        marker = "  ◀ best" if r5 > best_r5 else ""
        print(
            f"    ECR={ecr_a}  IDF={idf_a}  CAATB={caatb_a}"
            f"  dev R@5={r5*100:.2f}%{marker}",
            flush=True,
        )

        if r5 > best_r5:
            best_r5   = r5
            best_combo = (ecr_a, idf_a, caatb_a)

    best_ecr, best_idf, best_caatb = best_combo
    print(
        f"\n  [dev] Best: ECR={best_ecr}  IDF={best_idf}  CAATB={best_caatb}"
        f"  R@5={best_r5*100:.2f}%\n",
        flush=True,
    )
    return best_ecr, best_idf, best_caatb, best_r5


# ── Per-question held-out runner ──────────────────────────────────────────────


async def run_question(
    entry: dict,
    reranker: EntityConfidenceReranker,
    idf_booster: IDFKeywordBooster,
    caatb: ConfidenceAdaptiveTemporalBooster,
) -> dict:
    """Run a single held-out question: index → search → post-process → score."""
    tmpdir = Path(tempfile.mkdtemp(prefix="mw_held_"))
    try:
        workspace     = tmpdir / "ws"
        question_date = entry.get("question_date", "")
        write_sessions(workspace, entry)
        config = make_config(workspace)

        t0 = time.monotonic()
        async with MemWeave(config) as mem:
            mem.register_postprocessor(reranker)
            mem.register_postprocessor(idf_booster)
            mem.register_postprocessor(caatb)
            await mem.index()
            results = await mem.search(
                entry["question"],
                strategy="vector",
                max_results=_MAX_RESULTS,
                min_score=0.0,
                question_date=question_date,
            )
        duration_ms = (time.monotonic() - t0) * 1000

        ranked  = results_to_session_ranking(results)
        correct = set(entry["answer_session_ids"])

        row: dict = {
            "question_id":        entry["question_id"],
            "question_type":      entry["question_type"],
            "question":           entry["question"],
            "answer":             entry["answer"],
            "answer_session_ids": entry["answer_session_ids"],
            "ranked_session_ids": ranked[:50],
            "num_sessions":       len(entry["haystack_sessions"]),
            "num_results":        len(results),
            "_duration_ms":       round(duration_ms),
        }
        for k in _KS:
            row[f"recall_any@{k}"] = recall_any_at_k(ranked, correct, k)
            row[f"ndcg@{k}"]       = round(ndcg_at_k(ranked, correct, k), 4)

        return row
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Per-seed summary ──────────────────────────────────────────────────────────


def build_seed_summary(
    rows: list[dict],
    seed: int,
    best_ecr: float,
    best_idf: float,
    best_caatb: float,
    dev_r5: float,
) -> dict:
    n = len(rows)
    per_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_type[r["question_type"]].append(r)

    summary: dict = {
        "seed":            seed,
        "n":               n,
        "embedding_model": OLLAMA_MODEL,
        "best_alphas": {
            "ecr":   best_ecr,
            "idf":   best_idf,
            "caatb": best_caatb,
        },
        "dev_r5":          round(dev_r5, 4),
    }
    for k in _KS:
        summary[f"recall_any@{k}"] = round(
            sum(r[f"recall_any@{k}"] for r in rows) / n, 4
        )
        summary[f"ndcg@{k}"] = round(
            sum(r[f"ndcg@{k}"] for r in rows) / n, 4
        )

    summary["per_type"] = {}
    for qtype, type_rows in sorted(per_type.items()):
        tn = len(type_rows)
        type_stats: dict = {"n": tn}
        for k in _KS:
            type_stats[f"recall_any@{k}"] = round(
                sum(r[f"recall_any@{k}"] for r in type_rows) / tn, 4
            )
            type_stats[f"ndcg@{k}"] = round(
                sum(r[f"ndcg@{k}"] for r in type_rows) / tn, 4
            )
        summary["per_type"][qtype] = type_stats

    return summary


# ── Aggregate across seeds ────────────────────────────────────────────────────


def _stat(values: list[float]) -> dict:
    mean = statistics.mean(values)
    std  = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "std":  round(std,  4),
        "min":  round(min(values), 4),
        "max":  round(max(values), 4),
    }


def build_aggregate(per_seed_summaries: list[dict]) -> dict:
    metric_keys = (
        [f"recall_any@{k}" for k in _KS]
        + [f"ndcg@{k}" for k in _KS]
    )

    agg: dict = {
        "n_seeds":              len(per_seed_summaries),
        "seeds":                [s["seed"] for s in per_seed_summaries],
        "n_held_out_per_split": per_seed_summaries[0]["n"],
        "split_type":           "stratified",
        "alpha_grid": {
            "ecr":   ECR_ALPHAS,
            "idf":   IDF_ALPHAS,
            "caatb": CAATB_ALPHAS,
        },
        "best_alphas_per_seed": {
            str(s["seed"]): s["best_alphas"] for s in per_seed_summaries
        },
        "aggregate": {
            key: _stat([s[key] for s in per_seed_summaries])
            for key in metric_keys
        },
    }

    all_types = sorted(
        set().union(*[set(s["per_type"].keys()) for s in per_seed_summaries])
    )
    agg["per_type_aggregate"] = {}
    for qtype in all_types:
        n_per_split = [s["per_type"].get(qtype, {}).get("n", 0) for s in per_seed_summaries]
        type_agg: dict = {"n_per_split": n_per_split}
        for key in metric_keys:
            vals = [s["per_type"].get(qtype, {}).get(key, 0.0) for s in per_seed_summaries]
            type_agg[key] = _stat(vals)
        agg["per_type_aggregate"][qtype] = type_agg

    return agg


# ── Printing ──────────────────────────────────────────────────────────────────


def print_seed_summary(summary: dict) -> None:
    sep    = "─" * 72
    alphas = summary["best_alphas"]
    print(f"\n{sep}")
    print(
        f"  Seed {summary['seed']}  n={summary['n']}  "
        f"best α: ECR={alphas['ecr']} IDF={alphas['idf']} CAATB={alphas['caatb']}  "
        f"dev R@5={summary['dev_r5']*100:.2f}%"
    )
    print(sep)
    print(f"  {'Type':<35}  {'n':>4}  {'R@5':>8}  {'R@10':>8}  {'R@15':>8}  {'NDCG@5':>8}")
    print(f"  {'─'*35}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for qtype, stats in summary["per_type"].items():
        print(
            f"  {qtype:<35}  {stats['n']:>4}  "
            f"{stats['recall_any@5']*100:>7.2f}%  "
            f"{stats['recall_any@10']*100:>7.2f}%  "
            f"{stats['recall_any@15']*100:>7.2f}%  "
            f"{stats['ndcg@5']*100:>7.2f}%"
        )
    print(f"  {'─'*35}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    print(
        f"  {'Overall':<35}  {summary['n']:>4}  "
        f"{summary['recall_any@5']*100:>7.2f}%  "
        f"{summary['recall_any@10']*100:>7.2f}%  "
        f"{summary['recall_any@15']*100:>7.2f}%  "
        f"{summary['ndcg@5']*100:>7.2f}%"
    )
    print(sep)


def print_aggregate(agg: dict) -> None:
    sep = "═" * 72
    print(f"\n{sep}")
    print(
        f"  AGGREGATE  {agg['n_seeds']} seeds {agg['seeds']}  "
        f"n={agg['n_held_out_per_split']} per split  (stratified CV)"
    )
    print(sep)

    print(f"  {'Seed':<6}  {'ECR α':>6}  {'IDF α':>6}  {'CAATB α':>8}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}")
    for seed_str, alphas in agg["best_alphas_per_seed"].items():
        print(f"  {seed_str:<6}  {alphas['ecr']:>6}  {alphas['idf']:>6}  {alphas['caatb']:>8}")
    print()

    def fmt(key: str) -> str:
        s = agg["aggregate"][key]
        return (
            f"{s['mean']*100:>6.2f}%"
            f"  ±{s['std']*100:.2f}"
            f"  [{s['min']*100:.2f}–{s['max']*100:.2f}]"
        )

    print(f"  {'Metric':<12}  {'Mean':>7}  {'±Std':>6}  {'[Min–Max]'}")
    print(f"  {'─'*12}  {'─'*7}  {'─'*6}  {'─'*20}")
    for k in _KS:
        print(f"  R@{k:<10}  {fmt(f'recall_any@{k}')}")
    print()
    for k in [5, 10]:
        print(f"  NDCG@{k:<7}  {fmt(f'ndcg@{k}')}")
    print()
    print(f"  Per-type R@5 (mean ± std):")
    print(f"  {'─'*60}")
    for qtype, stats in agg["per_type_aggregate"].items():
        s  = stats["recall_any@5"]
        ns = stats["n_per_split"]
        print(
            f"  {qtype:<35}  "
            f"{s['mean']*100:>6.2f}% ±{s['std']*100:.2f}  "
            f"n≈{round(sum(ns)/len(ns))}"
        )
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed CV sweep — memweave × LongMemEval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to longmemeval_s_cleaned.json",
    )
    parser.add_argument(
        "--splits-dir",
        default="benchmarks/data/splits",
        dest="splits_dir",
        help="Directory for per-seed stratified split files (default: benchmarks/data/splits)",
    )
    parser.add_argument(
        "--out",
        default="benchmarks/results/multiseed",
        help="Output directory (default: benchmarks/results/multiseed)",
    )
    args = parser.parse_args()

    check_ollama()
    print("Warming up Ollama model ...", flush=True)
    warmup_ollama()
    print("Ready.\n", flush=True)

    print(f"Alpha grid : ECR {ECR_ALPHAS} × IDF {IDF_ALPHAS} × CAATB {CAATB_ALPHAS} = {len(ALPHA_GRID)} combos")
    print(f"Seeds      : {SEEDS}")
    print(f"K values   : {_KS}")
    print(f"Output     : {args.out}\n")

    data       = json.loads(Path(args.dataset).read_text())
    out_dir    = Path(args.out)
    splits_dir = Path(args.splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    per_seed_summaries: list[dict] = []

    for seed_idx, seed in enumerate(SEEDS, 1):
        print(f"\n{'─'*72}")
        print(f"  Run {seed_idx}/{len(SEEDS)}  seed={seed}")
        print(f"{'─'*72}")

        try:
            split              = make_stratified_split(seed, data, splits_dir)
            dev_ids            = set(split["dev"])
            held_out_ids       = set(split["held_out"])
            dev_questions      = [e for e in data if e["question_id"] in dev_ids]
            held_out_questions = [e for e in data if e["question_id"] in held_out_ids]

            dev_by_type: dict[str, int] = defaultdict(int)
            for e in dev_questions:
                dev_by_type[e["question_type"]] += 1
            print(
                f"  Split: {len(dev_questions)} dev / {len(held_out_questions)} held-out",
                flush=True,
            )
            print(
                "  Dev type counts: "
                + "  ".join(f"{t}={n}" for t, n in sorted(dev_by_type.items())),
                flush=True,
            )
            print()

            # ── Phase 1: tune alphas on dev ───────────────────────────────────
            best_ecr, best_idf, best_caatb, dev_r5 = await run_dev_sweep(
                dev_questions, ALPHA_GRID
            )

            # ── Phase 2: evaluate held-out with best alphas ───────────────────
            reranker    = EntityConfidenceReranker(alpha=best_ecr)
            idf_booster = IDFKeywordBooster(alpha=best_idf)
            caatb       = ConfidenceAdaptiveTemporalBooster(alpha=best_caatb)

            out_path = out_dir / f"results_mw_cv_seed{seed}_held_out_{ts}.jsonl"
            rows: list[dict] = []

            with open(out_path, "w") as f:
                for i, entry in enumerate(held_out_questions, 1):
                    row = await run_question(entry, reranker, idf_booster, caatb)
                    rows.append(row)
                    hit = "✓" if row["recall_any@5"] == 1.0 else "✗"
                    print(
                        f"  [{i:>3}/{len(held_out_questions)}] {hit}  "
                        f"{row['question_type']:<32}  "
                        f"R@5={row['recall_any@5']:.0f}  "
                        f"NDCG@5={row['ndcg@5']:.3f}  "
                        f"({row['_duration_ms']} ms)",
                        flush=True,
                    )
                    json_row = {k: v for k, v in row.items() if not k.startswith("_")}
                    f.write(json.dumps(json_row) + "\n")
                    f.flush()

                seed_summary = build_seed_summary(
                    rows, seed, best_ecr, best_idf, best_caatb, dev_r5
                )
                f.write(json.dumps({"__summary__": True, **seed_summary}) + "\n")

            per_seed_summaries.append(seed_summary)
            print_seed_summary(seed_summary)
            print(f"  → {out_path}", flush=True)

        except Exception as exc:
            print(f"\n  ERROR on seed={seed}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            print(f"  Skipping seed={seed} and continuing ...\n", flush=True)
            continue

    if not per_seed_summaries:
        print("\nNo seeds completed successfully — no aggregate to compute.")
        return

    print(f"\nCompleted {len(per_seed_summaries)}/{len(SEEDS)} seeds.")

    agg      = build_aggregate(per_seed_summaries)
    agg_path = out_dir / f"summary_mw_cv_{len(per_seed_summaries)}seeds_{ts}.json"
    agg_path.write_text(json.dumps(agg, indent=2))

    print_aggregate(agg)
    print(f"\nAggregate summary → {agg_path}")


if __name__ == "__main__":
    asyncio.run(main())
