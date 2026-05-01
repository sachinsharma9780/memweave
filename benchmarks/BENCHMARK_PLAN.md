# Memweave Benchmark Plan

## Goal

Establish credible, reproducible retrieval benchmarks for memweave on the
LongMemEval dataset — the de facto standard for agentic memory evaluation.
Results will be comparable to mempalace's published numbers because we use
identical conditions: same dataset, same split, same embedding model, same
primary metric.

We benchmark one dataset and one retrieval mode at a time, publishing results
incrementally.

---

## Reference Baseline (Mempalace)

| Mode | R@5 | Notes |
|---|---|---|
| Raw (semantic only) | 96.6% | Full 500 questions |
| Hybrid v1 (+ keyword boost) | 97.8% | Full 500 |
| Hybrid v2 (+ temporal boost) | **98.4%** | Clean 450 held-out — honest published number |
| Hybrid v4 + LLM rerank | 100% | Contaminated — tuned on specific failing questions |

Our north star for Phase 1 is matching the raw **96.6%** to validate that the
experimental setup is correct. Everything above that is upside.

---

## Milestone 1 — Environment & Dataset Setup

### 1.1 Install Dependencies

```bash
# Ollama (https://ollama.com) — local embedding server
ollama pull all-minilm:l6-v2

# Confirm it is running and the model responds
curl http://localhost:11434/api/embeddings \
  -d '{"model": "all-minilm:l6-v2", "prompt": "test"}' | python3 -m json.tool
```

Memweave uses LiteLLM under the hood, so no extra Python dependency is needed.
The model string to pass everywhere is `ollama/all-minilm:l6-v2`.

### 1.2 Download LongMemEval Dataset

```bash
mkdir -p benchmarks/data/longmemeval
curl -L -o benchmarks/data/longmemeval/longmemeval_s_cleaned.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
```

Expected file: `benchmarks/data/longmemeval/longmemeval_s_cleaned.json`

Each entry has these fields (verified against mempalace's benchmark script):
- `question_id`, `question`, `answer`, `question_type`, `question_date`
- `haystack_sessions` — list of sessions; each session is a list of `{"role": "user"|"assistant", "content": "..."}` turns
- `haystack_session_ids` — **separate top-level list** of session ID strings, parallel to `haystack_sessions`
- `haystack_dates` — **separate top-level list** of date strings, parallel to `haystack_sessions`
- `answer_session_ids` — list of ground-truth session IDs for scoring

### 1.3 Copy the Split File from Mempalace

We reuse mempalace's published split exactly so results are directly comparable.

```bash
cp /path/to/mempalace/benchmarks/lme_split_50_450.json \
   benchmarks/data/lme_split_50_450.json
```

Split details:
- `seed: 42`
- `dev`: 50 question IDs — use this for iteration and sanity checks
- `held_out`: 450 question IDs — the honest published number

> **Rule:** never tune anything by looking at held-out failures.
> All development and debugging happens against the dev split only.

### 1.4 Verify Dataset Shape

```python
import json

with open("benchmarks/data/longmemeval/longmemeval_s_cleaned.json") as f:
    data = json.load(f)

print(f"Total questions: {len(data)}")

from collections import Counter
types = Counter(e["question_type"] for e in data)
for t, n in sorted(types.items()):
    print(f"  {t}: {n}")

sample = data[0]
print(f"\nSample question_id:      {sample['question_id']}")
print(f"Haystack sessions:       {len(sample['haystack_sessions'])}")
print(f"Haystack session IDs:    {sample['haystack_session_ids'][:2]}")
print(f"Haystack dates:          {sample['haystack_dates'][:2]}")
print(f"Answer session IDs:      {sample['answer_session_ids']}")
print(f"Turns in first session:  {len(sample['haystack_sessions'][0])}")
```

Expected question types and approximate counts:

| Type | Count |
|---|---|
| multi-session | 133 |
| temporal-reasoning | 133 |
| knowledge-update | 78 |
| single-session-user | 70 |
| single-session-assistant | 56 |
| single-session-preference | 30 |

---

## Milestone 2 — Embedding Parity Check

Before running the full benchmark, confirm that the Ollama model produces
embeddings comparable to mempalace's ChromaDB default.

**Why this matters:** Both wrap `sentence-transformers/all-MiniLM-L6-v2` and
should produce identical 384-dimensional vectors. A quick sanity check rules
out version drift or configuration problems.

```python
import asyncio
from memweave.embedding.provider import LiteLLMEmbeddingProvider
from memweave.config import EmbeddingConfig

async def check():
    cfg = EmbeddingConfig(
        model="ollama/all-minilm:l6-v2",
        api_base="http://localhost:11434",
    )
    provider = LiteLLMEmbeddingProvider(cfg)
    vec = await provider.embed_query("What degree did I graduate with?")
    print(f"Dimensions : {len(vec)}")          # expect 384
    print(f"L2 norm    : {sum(x**2 for x in vec)**0.5:.4f}")  # expect ~1.0

asyncio.run(check())
```

Pass criteria: 384 dimensions, L2 norm ≈ 1.0.

---

## Milestone 3 — Benchmark Script Design

### 3.1 Per-Question Isolation Strategy

Mempalace uses `ChromaDB EphemeralClient()` — a fresh in-memory store per
question. Memweave uses SQLite on disk, so we create a **temporary workspace
directory per question** and delete it afterwards. This gives the same isolation
guarantee.

```
for each question:
    tmpdir = tempfile.mkdtemp(prefix="mw_bench_")
    workspace = tmpdir / "ws"
    └── memory/
        └── {session_id}.md   ← one file per haystack session
    .memweave/
        └── index.sqlite      ← created by MemWeave.index()

    run benchmark logic
    shutil.rmtree(tmpdir)
```

### 3.2 Session-to-File Formatting

Each haystack session is written as a single markdown file named
`{session_id}.md`. The filename is the session ID — this is what we map back to
after retrieval.

**Important:** Mempalace's raw mode indexes **user turns only** — it joins all
`role == "user"` content into a single document per session, discarding assistant
turns. We match this exactly for the raw baseline so the comparison is fair.

```
# Session: {session_id}
Date: {session_date}

{user_turn_1}

{user_turn_2}

{user_turn_3}
...
```

This is a deliberate choice for raw mode. For future hybrid modes we can test
full sessions (user + assistant) as a separate variant and report both.

### 3.3 Retrieval Pipeline (Raw Mode)

For raw mode we use `strategy="vector"` — pure cosine similarity, no BM25,
no MMR, no temporal decay. This is the direct equivalent of mempalace's `raw`
mode.

Key parameters:

| Parameter | Value | Reason |
|---|---|---|
| `strategy` | `"vector"` | semantic-only, matches mempalace raw |
| `max_results` | 50 | same candidate pool as mempalace |
| `min_score` | 0.0 | no threshold — let @K scoring handle filtering |
| `mmr.enabled` | False | off for raw mode |
| `temporal_decay.enabled` | False | off for raw mode |
| `sync.on_search` | False | we call `index()` explicitly; disable auto-sync to avoid redundant re-index |
| embedding model | `ollama/all-minilm:l6-v2` | same model as mempalace |

### 3.4 Chunk-to-Session Mapping

Memweave returns `SearchResult` objects with a `path` field — the file path of
the indexed chunk. Because each session is stored as `{session_id}.md`, we
recover the session ID by taking `Path(result.path).stem`.

When scoring Recall@K, de-duplicate by session ID first — multiple chunks from
the same session count as one hit.

```python
def results_to_session_ranking(results: list[SearchResult]) -> list[str]:
    seen = set()
    ranked_session_ids = []
    for r in results:
        sid = Path(r.path).stem
        if sid not in seen:
            seen.add(sid)
            ranked_session_ids.append(sid)
    return ranked_session_ids
```

### 3.5 Metrics

Exactly matching mempalace's metric definitions:

```python
from math import log2

def recall_any_at_k(ranked_sids: list[str], correct_sids: set[str], k: int) -> float:
    return float(any(sid in correct_sids for sid in ranked_sids[:k]))

def recall_all_at_k(ranked_sids: list[str], correct_sids: set[str], k: int) -> float:
    return float(all(sid in correct_sids for sid in ranked_sids[:k]))

def dcg(relevances: list[float], k: int) -> float:
    return sum(r / log2(i + 2) for i, r in enumerate(relevances[:k]))

def ndcg_at_k(ranked_sids: list[str], correct_sids: set[str], k: int) -> float:
    relevances = [1.0 if sid in correct_sids else 0.0 for sid in ranked_sids[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(relevances, k) / idcg if idcg > 0 else 0.0
```

Computed at K = 1, 3, 5, 10. Primary published metric: **Recall@5 (any)**.

### 3.6 Output Format

One JSONL line per question, structured to match mempalace's output format for
direct diff comparison:

```jsonl
{
  "question_id": "e47becba",
  "question_type": "single-session-user",
  "question": "What degree did I graduate with?",
  "answer": "Business Administration",
  "answer_session_ids": ["sharegpt_abc_0"],
  "ranked_session_ids": ["sharegpt_abc_0", "sharegpt_xyz_1", ...],
  "recall_any@1": 1.0,
  "recall_any@3": 1.0,
  "recall_any@5": 1.0,
  "recall_any@10": 1.0,
  "recall_all@5": 1.0,
  "ndcg@5": 1.0,
  "ndcg@10": 0.93,
  "num_sessions": 53,
  "duration_ms": 412
}
```

Final line of the file is an aggregate summary:

```jsonl
{"__summary__": true, "mode": "raw", "split": "dev", "n": 50, "recall_any@5": 0.94, ...}
```

### 3.7 Script Structure

```
benchmarks/
├── longmemeval_bench.py      ← main script (this milestone)
├── data/
│   ├── longmemeval/
│   │   └── longmemeval_s_cleaned.json
│   └── lme_split_50_450.json
└── results/
    └── results_mw_raw_vector_dev_<timestamp>.jsonl
```

Script CLI interface:

```bash
python benchmarks/longmemeval_bench.py \
  --dataset  benchmarks/data/longmemeval/longmemeval_s_cleaned.json \
  --split    benchmarks/data/lme_split_50_450.json \
  --mode     raw \
  --subset   dev \          # dev | held_out | all
  --limit    10 \           # optional: run first N questions only
  --out      benchmarks/results/
```

Internal flow:

```
main()
├── load_dataset()
├── load_split()
├── filter_questions(subset)
├── check_ollama_server()      ← fail fast if Ollama is not running
└── for each question:
    ├── write_sessions_to_tmpdir()
    ├── index_workspace()          ← MemWeave.index()
    ├── search_workspace()         ← MemWeave.search(strategy="vector")
    ├── map_results_to_sessions()  ← deduplicate by filename stem
    ├── compute_metrics()
    ├── append_to_jsonl()
    └── cleanup_tmpdir()
    
aggregate_and_print_summary()
```

---

## Milestone 4 — Dev Run & Validation (50 Questions)

Run on the dev split first. This is the iteration loop — safe to look at
failures here.

```bash
python benchmarks/longmemeval_bench.py \
  --dataset benchmarks/data/longmemeval/longmemeval_s_cleaned.json \
  --split   benchmarks/data/lme_split_50_450.json \
  --mode    raw \
  --subset  dev
```

**Pass criteria before proceeding to held-out:**

1. No crashes or uncaught exceptions across all 50 questions
2. Recall@5 on dev lands in the range **93–98%** — if outside this, something
   is wrong with the setup (embedding model not loaded, session files malformed,
   scoring logic bug)
3. Per-category breakdown looks plausible (temporal-reasoning questions should
   score lower than single-session questions, same as mempalace's pattern)
4. Spot-check 3 failed questions manually — read the question, open the top-5
   retrieved sessions, understand why the correct one was missed

**Timing expectation:** ~5–10 minutes for 50 questions on Apple Silicon.
Mempalace's in-memory ChromaDB is faster; our disk I/O adds overhead. Document
the wall-clock time in the result summary.

---

## Milestone 5 — Full Held-Out Run (450 Questions)

Only run this once the dev run passes. Never debug against held-out failures.

```bash
python benchmarks/longmemeval_bench.py \
  --dataset benchmarks/data/longmemeval/longmemeval_s_cleaned.json \
  --split   benchmarks/data/lme_split_50_450.json \
  --mode    raw \
  --subset  held_out
```

The number from this run is the **published baseline**. Compare directly to
mempalace's raw 96.6% on the same 500-question set (note: mempalace ran all 500,
we run the 450 held-out — slightly different denominators, both valid).

Result file name convention:
`results_mw_raw_vector_held_out_<YYYYMMDD>_<HHMM>.jsonl`

---

## Future Milestones (Planned, Not Yet Scheduled)

These will each follow the same dev-first → held-out pattern.

| Milestone | Mode | What Changes |
|---|---|---|
| 6 | Hybrid (default) | Add BM25 alongside vector (`strategy="hybrid"`) |
| 7 | Hybrid + MMR | Enable `mmr.enabled=True`, tune `lambda_param` on dev |
| 8 | Hybrid + LLM rerank | Claude Haiku final pass on top-10 |
| 9 | LoCoMo | Second dataset, same raw → hybrid progression |

---

## Integrity Rules

These apply to all milestones:

1. **No tuning against held-out.** All parameter choices are made on the dev
   split. The held-out run is a single, clean measurement.
2. **Same split file for every run.** Never regenerate the split. Use
   `lme_split_50_450.json` verbatim.
3. **Label results honestly.** If a mode was developed by inspecting failures,
   say so. Mempalace called their v4 "contaminated" — we should do the same.
4. **Commit result files.** All `.jsonl` result files are committed to the repo
   so results are reproducible and auditable.
5. **Pin the Ollama model version.** Record the exact Ollama version and model
   digest in the result summary so the run is reproducible.
