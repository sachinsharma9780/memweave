"""
Memweave × LongMemEval Benchmark

Evaluates memweave on LongMemEval-S using the ECR + IDF + CAATB retrieval
pipeline. Primary metric: Recall@5 (any correct session in top-5) — directly
comparable to mempalace's published results.

Pipeline
--------
  vector search (200 candidates)
    → EntityConfidenceReranker  (--alpha,      default 0.3)
    → IDFKeywordBooster         (--idf-alpha,  default 0.6)
    → CAATB                     (--caatb-alpha, default 0.2)
    → top-K cutoff

Typical workflow
----------------
Step 1  Find optimal hyperparameters on dev (≈10 min, uses smart caching):

    .venv/bin/python -u benchmarks/longmemeval_bench.py \\
        --dataset benchmarks/data/longmemeval/longmemeval_s_cleaned.json \\
        --split   benchmarks/data/lme_split_50_450.json \\
        --subset  dev --sweep

Step 2  Reproduce held-out with the best alphas from Step 1 (≈60 min):

    .venv/bin/python -u benchmarks/longmemeval_bench.py \\
        --dataset     benchmarks/data/longmemeval/longmemeval_s_cleaned.json \\
        --split       benchmarks/data/lme_split_50_450.json \\
        --subset      held_out \\
        --alpha       0.3 --idf-alpha 0.6 --caatb-alpha 0.2
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
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
import sys

sys.path.insert(0, str(Path(__file__).parent))

import litellm

litellm.suppress_debug_info = True

from memweave import MemWeave, MemoryConfig
from memweave.config import CacheConfig, EmbeddingConfig, QueryConfig, SyncConfig
from strategies.caatb import ConfidenceAdaptiveTemporalBooster
from strategies.entity_confidence_reranker import EntityConfidenceReranker
from strategies.idf_keyword_boost import IDFKeywordBooster

# ── Constants ─────────────────────────────────────────────────────────────────

OLLAMA_MODEL = "ollama/all-minilm:l6-v2"
OLLAMA_BASE  = "http://localhost:11434"

_MAX_RESULTS = 200
_KS          = [1, 3, 5, 10, 15, 25]

# Alpha grid swept by --sweep  (ECR × IDF × CAATB = 27 combinations)
_ECR_ALPHAS   = [0.2, 0.3, 0.4]
_IDF_ALPHAS   = [0.4, 0.6, 0.8]
_CAATB_ALPHAS = [0.1, 0.2, 0.3]


# ── Ollama preflight ──────────────────────────────────────────────────────────


def check_ollama() -> None:
    """Verify Ollama is running and all-minilm:l6-v2 is available.

    Raises
    ------
    SystemExit
        If Ollama is not reachable at ``OLLAMA_BASE`` or ``all-minilm:l6-v2``
        is not installed — message includes the corrective command to run.
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
        models = [m["name"] for m in tags.get("models", [])]
    except urllib.error.URLError:
        raise SystemExit(
            f"Ollama not running at {OLLAMA_BASE}.\n"
            "Start Ollama and re-run."
        )
    if not any("all-minilm" in m for m in models):
        raise SystemExit(
            "all-minilm:l6-v2 not found in Ollama.\n"
            "Run:  ollama pull all-minilm:l6-v2"
        )


def warmup_ollama() -> None:
    """Send one embedding request so the model is hot before timing starts."""
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
        pass  # non-fatal — timing will be slightly off for Q1


# ── Data loading ──────────────────────────────────────────────────────────────


def load_dataset(path: Path) -> list[dict]:
    """Load the full LongMemEval-S JSON dataset from disk."""
    with open(path) as f:
        return json.load(f)


def load_split(path: Path) -> dict[str, list[str]]:
    """Load a split file mapping subset names to lists of question IDs.

    Expected keys: "dev" (50 IDs) and "held_out" (450 IDs).
    """
    with open(path) as f:
        return json.load(f)


def filter_questions(
    data: list[dict], split: dict[str, list[str]], subset: str
) -> list[dict]:
    """Return only the questions belonging to the requested subset.

    Parameters
    ----------
    data:
        Full dataset loaded by ``load_dataset``.
    split:
        Split mapping from ``load_split`` (keys: ``"dev"``, ``"held_out"``).
    subset:
        One of ``"dev"``, ``"held_out"``, or ``"all"`` (returns everything).
    """
    if subset == "all":
        return data
    ids = set(split[subset])
    return [e for e in data if e["question_id"] in ids]


# ── Session formatting ────────────────────────────────────────────────────────


def write_sessions(workspace: Path, entry: dict) -> None:
    """Write each haystack session as {session_id}.md — user turns only.

    Parameters
    ----------
    workspace:
        Root directory for the MemWeave workspace; files are written under
        ``workspace/memory/``.
    entry:
        Single LongMemEval dataset record containing ``haystack_sessions``,
        ``haystack_session_ids``, and ``haystack_dates``.
    """
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
    """Build a MemoryConfig for a single benchmark question.

    Caching and on-search sync are disabled so each question starts cold and
    results are purely deterministic. ``min_score=0.0`` and
    ``max_results=_MAX_RESULTS`` ensure the re-rankers always receive the full
    200-candidate pool.
    """
    return MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=OLLAMA_MODEL, api_base=OLLAMA_BASE),
        sync=SyncConfig(on_search=False),
        cache=CacheConfig(enabled=False),
        progress=False,
        query=QueryConfig(max_results=_MAX_RESULTS, min_score=0.0, strategy="vector"),
    )


# ── Chunk → session mapping ───────────────────────────────────────────────────


def results_to_session_ranking(results: list) -> list[str]:
    """Deduplicate chunk-level search results to a session-level ranking.

    MemWeave indexes each session file as multiple chunks, so the raw result
    list may contain several chunks from the same session. This function
    collapses them to one entry per session, keeping only the first (highest-
    ranked) chunk's position. Later chunks from the same session are discarded
    because the session is already represented — and once a session is
    retrieved, all its content is available to the reader.
    """
    seen: set[str] = set()
    ranked: list[str] = []
    for r in results:
        # turns the full path into just session_A, session_B, etc.
        sid = Path(r.path).stem
        # skips any session it has already seen
        if sid not in seen:
            seen.add(sid)
            ranked.append(sid)
    return ranked


# ── Metrics ───────────────────────────────────────────────────────────────────


def recall_any_at_k(ranked: list[str], correct: set[str], k: int) -> float:
    """Return 1.0 if at least one correct session appears in the top-k results, else 0.0.

    Parameters
    ----------
    ranked:
        Ordered list of session IDs, best match first.
    correct:
        Set of ground-truth session IDs for the question.
    k:
        Cutoff rank.
    """
    return float(any(sid in correct for sid in ranked[:k]))


def ndcg_at_k(ranked: list[str], correct: set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at cutoff k.

    Parameters
    ----------
    ranked:
        Ordered list of session IDs, best match first.
    correct:
        Set of ground-truth session IDs for the question.
    k:
        Cutoff rank.

    Steps
    -----
    1. Build a binary relevance vector of length k: 1.0 if ``ranked[i]`` is a
       correct session ID, else 0.0.
    2. Compute DCG — sum each relevance score divided by log2(position + 2),
       where position is 0-indexed, so position 0 has discount log2(2)=1.
    3. Compute IDCG — the *ideal* DCG achieved by a perfect ranking (all
       relevant items first). Obtained by sorting the relevance vector
       descending and applying the same formula.
    4. Normalise: NDCG = DCG / IDCG. Returns 0.0 when IDCG = 0 (no correct
       session exists in the top-k pool at all).
    """
    relevances = [1.0 if sid in correct else 0.0 for sid in ranked[:k]]
    ideal = sorted(relevances, reverse=True)
    dcg  = sum(r / log2(i + 2) for i, r in enumerate(relevances))
    idcg = sum(r / log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ── Raw-row capture (used by --sweep) ─────────────────────────────────────────


class _RawCapture:
    """No-op post-processor that intercepts RawSearchRow objects.

    mem.search() converts RawSearchRow → SearchResult after post-processors run.
    SearchResult lacks .text and .vector_score, which the re-rankers need.
    Registering this as the only post-processor captures rows in raw form before
    that conversion, so --sweep can apply alpha combos offline without re-embedding.
    """

    def __init__(self) -> None:
        self.rows: list = []

    async def apply(self, rows: list, query: str, **kwargs: object) -> list:
        """Store the incoming raw rows and return them unchanged.

        Parameters
        ----------
        rows:
            ``RawSearchRow`` objects passed in by the MemWeave post-processor chain.
        query:
            The search query string (unused; kept for interface compatibility).
        **kwargs:
            Additional context forwarded by MemWeave (e.g. ``question_date``).
        """
        self.rows = list(rows)
        return rows


# ── Question runners ──────────────────────────────────────────────────────────


async def run_question_raw(entry: dict) -> tuple[list, str, str]:
    """Vector-only search for one question; captures raw rows before re-ranking.

    Used by ``sweep_dev`` to cache vector results once and evaluate all alpha
    combos offline — no re-embedding per combo.

    Parameters
    ----------
    entry:
        Single LongMemEval dataset record.

    Returns
    -------
    tuple[list, str, str]
        ``(raw_rows, query, question_date)`` where ``raw_rows`` are
        ``RawSearchRow`` objects captured before any post-processing.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="mw_raw_"))
    try:
        workspace = tmpdir / "ws"
        write_sessions(workspace, entry)
        capture = _RawCapture()
        async with MemWeave(make_config(workspace)) as mem:
            mem.register_postprocessor(capture)
            await mem.index()
            await mem.search(
                entry["question"],
                strategy="vector",
                max_results=_MAX_RESULTS,
                min_score=0.0,
                question_date=entry.get("question_date", ""),
            )
        return capture.rows, entry["question"], entry.get("question_date", "")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def run_question(
    entry: dict,
    ecr: EntityConfidenceReranker,
    idf: IDFKeywordBooster,
    caatb: ConfidenceAdaptiveTemporalBooster,
) -> dict:
    """Run the full pipeline for one question: index → search → ECR → IDF → CAATB → score.

    Parameters
    ----------
    entry:
        Single LongMemEval dataset record.
    ecr:
        Configured ``EntityConfidenceReranker`` post-processor.
    idf:
        Configured ``IDFKeywordBooster`` post-processor.
    caatb:
        Configured ``ConfidenceAdaptiveTemporalBooster`` post-processor.

    Returns
    -------
    dict
        Per-question result row with metrics for all k values in ``_KS``,
        ranked session IDs, and ``_duration_ms`` (console-only, stripped
        before JSONL write).
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="mw_bench_"))
    try:
        workspace = tmpdir / "ws"
        write_sessions(workspace, entry)
        qdate = entry.get("question_date", "")

        t0 = time.monotonic()
        async with MemWeave(make_config(workspace)) as mem:
            mem.register_postprocessor(ecr)
            mem.register_postprocessor(idf)
            mem.register_postprocessor(caatb)
            await mem.index()
            results = await mem.search(
                entry["question"],
                strategy="vector",
                max_results=_MAX_RESULTS,
                min_score=0.0,
                question_date=qdate,
            )
        duration_ms = round((time.monotonic() - t0) * 1000)

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
            "_duration_ms":       duration_ms,   # console only, excluded from JSONL
        }
        for k in _KS:
            row[f"recall_any@{k}"] = recall_any_at_k(ranked, correct, k)
            row[f"ndcg@{k}"]       = round(ndcg_at_k(ranked, correct, k), 4)
        return row
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Dev sweep ─────────────────────────────────────────────────────────────────


async def sweep_dev(questions: list[dict]) -> tuple[float, float, float, float]:
    """Find best alpha combo on dev using smart caching.

    Phase 1 — run each question once with vector-only search, capture raw rows.
    Phase 2 — apply all 27 alpha combos to cached rows offline (no re-embedding).
    Ties in R@5 are broken by NDCG@5.

    Parameters
    ----------
    questions:
        Dev-split question records from ``filter_questions``.

    Returns
    -------
    tuple[float, float, float, float]
        ``(best_ecr_alpha, best_idf_alpha, best_caatb_alpha, best_r5)``.
    """
    n = len(questions)
    print(f"[sweep] Caching raw vector rows for {n} dev questions …", flush=True)

    cached: list[tuple[list, str, str, set[str]]] = []
    for i, entry in enumerate(questions, 1):
        raw_rows, query, qdate = await run_question_raw(entry)
        cached.append((raw_rows, query, qdate, set(entry["answer_session_ids"])))
        print(f"  [{i:>3}/{n}] cached  ({len(raw_rows)} rows)", flush=True)

    grid = list(itertools.product(_ECR_ALPHAS, _IDF_ALPHAS, _CAATB_ALPHAS))
    print(
        f"\n[sweep] Evaluating {len(grid)} alpha combos "
        f"(ECR×IDF×CAATB = {len(_ECR_ALPHAS)}×{len(_IDF_ALPHAS)}×{len(_CAATB_ALPHAS)}) …\n",
        flush=True,
    )

    print(f"  {'ECR α':>6}  {'IDF α':>6}  {'CAATB α':>8}    {'R@5':>7}  {'NDCG@5':>8}", flush=True)
    print("  " + "─" * 52, flush=True)

    combo_results: list[tuple[float, float, float, float, float]] = []

    for ecr_a, idf_a, caatb_a in grid:
        ecr_pp   = EntityConfidenceReranker(alpha=ecr_a)
        idf_pp   = IDFKeywordBooster(alpha=idf_a)
        caatb_pp = ConfidenceAdaptiveTemporalBooster(alpha=caatb_a)

        r5_sum = ndcg5_sum = 0.0
        for raw_rows, query, qdate, correct in cached:
            rows = await ecr_pp.apply(list(raw_rows), query, question_date=qdate)
            rows = await idf_pp.apply(rows, query, question_date=qdate)
            rows = await caatb_pp.apply(rows, query, question_date=qdate)
            ranked     = results_to_session_ranking(rows)
            r5_sum    += recall_any_at_k(ranked, correct, 5)
            ndcg5_sum += ndcg_at_k(ranked, correct, 5)

        r5    = r5_sum    / n
        ndcg5 = ndcg5_sum / n
        combo_results.append((r5, ndcg5, ecr_a, idf_a, caatb_a))
        print(
            f"  {ecr_a:>6.1f}  {idf_a:>6.1f}  {caatb_a:>8.1f}    "
            f"{r5:>6.2%}  {ndcg5:>8.2%}",
            flush=True,
        )

    combo_results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_r5, best_ndcg5, best_ecr, best_idf, best_caatb = combo_results[0]

    print(f"\n[sweep] ── Best combo ──────────────────────────────────────", flush=True)
    print(f"  ECR α={best_ecr}  IDF α={best_idf}  CAATB α={best_caatb}", flush=True)
    print(f"  Dev R@5={best_r5:.2%}  NDCG@5={best_ndcg5:.2%}", flush=True)

    return best_ecr, best_idf, best_caatb, best_r5


# ── Aggregation ───────────────────────────────────────────────────────────────


def build_summary(rows: list[dict], subset: str, alphas: dict) -> dict:
    """Aggregate per-question rows into an overall + per-type summary dict.

    Parameters
    ----------
    rows:
        Per-question result rows produced by ``run_question``.
    subset:
        Name of the evaluated subset (``"dev"``, ``"held_out"``, or ``"all"``).
    alphas:
        Pipeline alpha values, e.g. ``{"ecr": 0.3, "idf": 0.6, "caatb": 0.2}``.

    Steps
    -----
    1. Group rows by ``question_type`` into a ``per_type`` mapping.
    2. Compute overall macro-average R@k and NDCG@k across all questions for
       every k in ``_KS``.
    3. Compute per-type R@5, R@10, NDCG@5 averages for each question type.
    4. Return a single dict with ``__summary__=True`` so the JSONL consumer can
       distinguish the summary row from per-question rows. The dict is written
       as the last line of the JSONL output file.
    """
    n = len(rows)
    per_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_type[r["question_type"]].append(r)

    summary: dict = {
        "__summary__": True,
        "subset":          subset,
        "n":               n,
        "embedding_model": OLLAMA_MODEL,
        "pipeline":        "ECR + IDF + CAATB",
        "alphas":          alphas,
    }
    for k in _KS:
        summary[f"recall_any@{k}"] = round(
            sum(r[f"recall_any@{k}"] for r in rows) / n, 4
        )
        summary[f"ndcg@{k}"] = round(
            sum(r[f"ndcg@{k}"] for r in rows) / n, 4
        )
    summary["per_type"] = {
        qtype: {
            "n":             len(type_rows),
            "recall_any@5":  round(sum(r["recall_any@5"]  for r in type_rows) / len(type_rows), 4),
            "recall_any@10": round(sum(r["recall_any@10"] for r in type_rows) / len(type_rows), 4),
            "ndcg@5":        round(sum(r["ndcg@5"]        for r in type_rows) / len(type_rows), 4),
        }
        for qtype, type_rows in sorted(per_type.items())
    }
    return summary


def print_summary(summary: dict, out_path: Path) -> None:
    """Print a formatted results table to stdout.

    Shows the output file path, pipeline alphas, overall metrics, and a
    per-question-type breakdown of R@5 and sample counts.

    Parameters
    ----------
    summary:
        Summary dict produced by ``build_summary``.
    out_path:
        Path to the JSONL results file, shown in the table header.
    """
    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  Results  →  {out_path}")
    print(
        f"  Pipeline    ECR α={summary['alphas']['ecr']}  "
        f"IDF α={summary['alphas']['idf']}  "
        f"CAATB α={summary['alphas']['caatb']}"
    )
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


# ── CLI ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    """CLI entry point supporting two evaluation modes.

    Sweep mode (``--sweep``)
    ------------------------
    Runs a 27-combination grid search over ECR × IDF × CAATB alphas on the
    dev split.  Uses two-phase caching: raw vector rows are computed once per
    question (Phase 1), then all alpha combos are applied offline without
    re-embedding (Phase 2).  Prints the best combo and the exact held-out
    command to reproduce.  Requires ``--subset dev``.

    Single evaluation mode (default)
    ---------------------------------
    Evaluates one fixed alpha configuration on any subset (dev / held_out /
    all).  For each question: index sessions → run pipeline → score → write
    row to JSONL.  Appends a ``__summary__`` row as the last JSONL line and
    prints a formatted summary table.  The output file is timestamped and
    tagged with the alpha values for traceability.
    """
    parser = argparse.ArgumentParser(
        description="Memweave × LongMemEval Benchmark — ECR + IDF + CAATB Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to longmemeval_s_cleaned.json",
    )
    parser.add_argument(
        "--split", required=True,
        help="Path to split JSON (50 dev / 450 held-out IDs)",
    )
    parser.add_argument(
        "--subset", default="dev", choices=["dev", "held_out", "all"],
        help="Question split to evaluate (default: dev)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help=(
            "Sweep 27 alpha combos on dev to find optimal hyperparameters. "
            "Requires --subset dev. Uses smart caching — raw rows are computed "
            "once; all combos are applied offline."
        ),
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3, metavar="ALPHA",
        help="ECR boost strength (default: 0.3)",
    )
    parser.add_argument(
        "--idf-alpha", type=float, default=0.6, metavar="IDF_ALPHA", dest="idf_alpha",
        help="IDF keyword boost strength (default: 0.6)",
    )
    parser.add_argument(
        "--caatb-alpha", type=float, default=0.2, metavar="CAATB_ALPHA", dest="caatb_alpha",
        help="CAATB temporal boost strength (default: 0.2)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Run only the first N questions (for quick iteration)",
    )
    parser.add_argument(
        "--out", default="benchmarks/results",
        help="Output directory for JSONL results (default: benchmarks/results)",
    )
    args = parser.parse_args()

    if args.sweep and args.subset != "dev":
        parser.error("--sweep requires --subset dev")

    check_ollama()
    print("Warming up Ollama model …", end=" ", flush=True)
    warmup_ollama()
    print("ready.\n", flush=True)

    data      = load_dataset(Path(args.dataset))
    split     = load_split(Path(args.split))
    questions = filter_questions(data, split, args.subset)
    if args.limit:
        questions = questions[: args.limit]

    # ── Sweep mode ────────────────────────────────────────────────────────────
    if args.sweep:
        best_ecr, best_idf, best_caatb, _ = await sweep_dev(questions)
        print(
            f"\n  Run held-out with these params:\n\n"
            f"    .venv/bin/python -u benchmarks/longmemeval_bench.py \\\n"
            f"        --dataset {args.dataset} \\\n"
            f"        --split   {args.split} \\\n"
            f"        --subset  held_out \\\n"
            f"        --alpha {best_ecr} --idf-alpha {best_idf} --caatb-alpha {best_caatb}",
            flush=True,
        )
        return

    # ── Single evaluation run ─────────────────────────────────────────────────
    alphas = {"ecr": args.alpha, "idf": args.idf_alpha, "caatb": args.caatb_alpha}
    ecr   = EntityConfidenceReranker(alpha=args.alpha)
    idf   = IDFKeywordBooster(alpha=args.idf_alpha)
    caatb = ConfidenceAdaptiveTemporalBooster(alpha=args.caatb_alpha)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    tag      = f"ecr{args.alpha}_idf{args.idf_alpha}_caatb{args.caatb_alpha}"
    out_path = out_dir / f"results_mw_{tag}_{args.subset}_{ts}.jsonl"

    print(
        f"Memweave × LongMemEval  |  "
        f"ECR α={args.alpha}  IDF α={args.idf_alpha}  CAATB α={args.caatb_alpha}  "
        f"subset={args.subset}  n={len(questions)}  model={OLLAMA_MODEL}"
    )
    print(f"Output → {out_path}\n", flush=True)

    rows: list[dict] = []
    with open(out_path, "w") as f:
        for i, entry in enumerate(questions, 1):
            row = await run_question(entry, ecr, idf, caatb)
            rows.append(row)

            hit = "✓" if row["recall_any@5"] == 1.0 else "✗"
            print(
                f"  [{i:>3}/{len(questions)}] {hit}  "
                f"{row['question_type']:<32}  "
                f"R@5={row['recall_any@5']:.0f}  "
                f"NDCG@5={row['ndcg@5']:.3f}  "
                f"({row['_duration_ms']} ms)",
                flush=True,
            )
            # strip internal timing key before writing to JSONL
            json_row = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(json_row) + "\n")
            f.flush()

        summary = build_summary(rows, args.subset, alphas)
        f.write(json.dumps(summary) + "\n")

    print_summary(summary, out_path)


if __name__ == "__main__":
    asyncio.run(main())
