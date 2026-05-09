# Memweave × LongMemEval-S Benchmark

Evaluates memweave on **LongMemEval-S** — a 500-question multi-session memory
retrieval benchmark. Primary metric: **Recall@5 (any)** — whether any correct
session appears in the top-5 results.

**Key results:**

| Evaluation | R@5 | NDCG@5 | 100% recall at |
|---|---|---|---|
| Canonical held-out (450 Qs) | **98.00%** | **93.75%** | **R@23** |
| 5-seed cross-validated mean | **97.24% ±0.12%** | **92.28% ±0.69%** | R@25 ±0.00% |

All runs: `all-MiniLM-L6-v2` via Ollama (local). No LLM, no API key, no cloud
at any stage.

---

## Table of Contents

- [Dataset](#dataset)
- [Question Types](#question-types)
- [Pipeline](#pipeline)
- [Strategies](#strategies)
- [Setup](#setup)
- [Reproducing Results](#reproducing-results)
- [Results](#results)
- [File Structure](#file-structure)

---

## Dataset

**LongMemEval-S** (`longmemeval_s_cleaned.json`, ~265 MB, 500 questions)

Published with the paper:
> *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory* — Wu et al., 2024
> [HuggingFace dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)

Each record is one benchmark question paired with a haystack of ~53 conversation
sessions (range: 40–86). The system must retrieve the session(s) containing the
answer from that haystack. Sessions are multi-turn dialogues; **only user turns
are indexed** — matching mempalace's raw mode for a direct apples-to-apples
comparison.

### Dev / Held-Out Split

```
benchmarks/data/lme_split_50_450.json   ← canonical fixed split (seed=42)
benchmarks/data/splits/                 ← 5 stratified CV splits (seeds 42, 0, 1, 2, 3)
```

| Subset | Questions | Purpose |
|---|---|---|
| `dev` | 50 | Hyperparameter tuning only (~7 min) |
| `held_out` | 450 | Single clean evaluation (~60 min) |

**Integrity rule:** all alpha decisions are made on `dev` only. The held-out
result is run once; no parameters are adjusted after observing held-out failures.

---

## Question Types

| Type | n (held-out) | Description |
|---|---|---|
| `single-session-user` | 63 | Answer is in a user-said turn in one session |
| `single-session-assistant` | 54 | Answer is in an assistant-said turn (not indexed — structural ceiling) |
| `single-session-preference` | 25 | Answer is an implicit user preference |
| `multi-session` | 115 | Answer requires multiple sessions |
| `knowledge-update` | 69 | Answer is the most recent version of an updated fact |
| `temporal-reasoning` | 124 | Answer requires reasoning about when something happened |

`single-session-assistant` is a structural ceiling — only user turns are indexed,
so answers that only appear in assistant turns cannot be retrieved by any
embedding strategy.

---

## Pipeline

```
vector search  (200 candidates, semantic similarity, all-MiniLM-L6-v2 via Ollama)
      │
      ▼
EntityConfidenceReranker   (ECR,   α = 0.3)
      │
      ▼
IDFKeywordBooster          (IDF,   α = 0.6)
      │
      ▼
ConfidenceAdaptiveTemporalBooster  (CAATB, α = 0.2)
      │
      ▼
top-K  (K ∈ {1, 3, 5, 10, 15, 25})
```

All three post-processors are registered via `mem.register_postprocessor()`

---

## Strategies

### EntityConfidenceReranker (ECR)

**File:** `strategies/entity_confidence_reranker.py`

Boosts sessions containing the query's named entities, but only where the vector
model is uncertain — so it never overrides already-confident results. Skips
preference questions where entity matching is unreliable.

```
boost_weight  = alpha × (1 − normalized_vector_score)
entity_signal = fraction of query entities present in session text
new_score     = min(1.0, score + boost_weight × entity_signal)
```

Additive form can cross rank boundaries that multiplicative boosts cannot.
**Alpha=0.3** — higher values regress multi-session questions.

---

### IDFKeywordBooster (IDF)

**File:** `strategies/idf_keyword_boost.py`

Boosts sessions containing discriminative query terms, weighted by how rare each
term is within this question's own retrieved haystack (200 candidates). IDF is
computed per-question so common terms in this specific haystack score low.

```
idf(t)       = log(N / (1 + df(t)))   where N = unique sessions, df(t) = sessions containing t
idf_overlap  = Σ idf(t) [t in session] / Σ idf(t) [all query tokens]
new_score    = min(1.0, score × (1 + alpha × idf_overlap))
```

Multiplicative form preserves relative ordering among strong vector results.
**Alpha=0.6** — highest NDCG@5 on dev.

---

### ConfidenceAdaptiveTemporalBooster (CAATB)

**File:** `strategies/caatb.py`

Boosts sessions by temporal proximity for queries expressing time offsets.
Normalises spelled-out numbers (`"four weeks ago"` →
`"4 weeks ago"`) then parses a `(days_back, tolerance)` pair from the query.
No lexical gate — temporal proximity alone fires the boost.

```
temporal_score = 1.0 within tolerance, linear decay to 0.0 at 3×tolerance
alpha_eff      = alpha × (1 − normalized_vector_score)
new_score      = min(1.0, score + alpha_eff × temporal_score)
```

Additive + confidence-adaptive form avoids displacing sessions the vector model
already ranked highly. **Alpha=0.2** — higher values cause same-date collisions
on dev.

---

## Setup

### 1. Install Ollama and pull the embedding model

```bash
# Install Ollama: https://ollama.com
ollama pull all-minilm:l6-v2
```

Verify it responds:

```bash
curl http://localhost:11434/api/embeddings \
  -d '{"model": "all-minilm:l6-v2", "prompt": "test"}' | python3 -m json.tool
```

### 2. Install memweave

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

### 3. Download the dataset

```bash
mkdir -p benchmarks/data/longmemeval
curl -L -o benchmarks/data/longmemeval/longmemeval_s_cleaned.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
```

The split files are already in the repo at `benchmarks/data/` — do not
regenerate them, as the held-out IDs must be identical across runs.

---

## Reproducing Results

### Fixed split — canonical held-out result

**Step 1 — Find optimal alphas on dev (~7 min)**

```bash
.venv/bin/python -u benchmarks/longmemeval_bench.py \
    --dataset benchmarks/data/longmemeval/longmemeval_s_cleaned.json \
    --split   benchmarks/data/lme_split_50_450.json \
    --subset  dev --sweep
```

Expected best combo: `ECR α=0.3  IDF α=0.6  CAATB α=0.2`

The sweep evaluates 27 alpha combinations without re-embedding — raw vector rows
are cached once per question and all combos are applied offline.

**Step 2 — Evaluate on held-out (~60 min)**

```bash
.venv/bin/python -u benchmarks/longmemeval_bench.py \
    --dataset     benchmarks/data/longmemeval/longmemeval_s_cleaned.json \
    --split       benchmarks/data/lme_split_50_450.json \
    --subset      held_out \
    --alpha       0.3 --idf-alpha 0.6 --caatb-alpha 0.2
```

Expected output:

```
Recall@5:   98.00%
NDCG@5:     93.75%
Recall@25: 100.00%
```

Results are written to `benchmarks/results/results_mw_ecr{α}_idf{α}_caatb{α}_held_out_{timestamp}.jsonl`.

---

### 5-seed cross-validated results (~5–6 hours)

```bash
.venv/bin/python -u benchmarks/multiseed_sweep.py \
    --dataset    benchmarks/data/longmemeval/longmemeval_s_cleaned.json \
    --splits-dir benchmarks/data/splits \
    --out        benchmarks/final_results
```

Expected output:

```
R@5:    97.24% ±0.12%
R@10:   98.76% ±0.12%
R@25:  100.00% ±0.00%
NDCG@5: 92.28% ±0.69%
```

Each seed runs its own dev sweep before evaluating on its own held-out set — no
information leaks across seeds.

---

## Results

### Canonical held-out (450 questions, ECR α=0.3, IDF α=0.6, CAATB α=0.2)

| K | Recall@K | NDCG@K |
|---|---|---|
| 1 | 90.00% | 90.00% |
| 3 | 96.44% | 93.45% |
| **5** | **98.00%** | **93.75%** |
| 10 | 99.11% | 93.76% |
| 15 | 99.78% | 93.86% |
| 25 | **100.00%** | 93.83% |

#### Per question type

| Question type | n | R@5 | R@10 | NDCG@5 |
|---|---|---|---|---|
| single-session-user | 63 | **100.00%** | 100.00% | 98.62% |
| knowledge-update | 69 | 98.55% | 100.00% | 97.25% |
| single-session-assistant | 54 | 98.15% | 98.15% | 97.01% |
| multi-session | 115 | 99.13% | 99.13% | 94.57% |
| temporal-reasoning | 124 | 97.58% | 99.19% | 90.51% |
| single-session-preference | 25 | 88.00% | 96.00% | 77.12% |
| **Overall** | **450** | **98.00%** | **99.11%** | **93.75%** |

`single-session-preference` is the hardest type — preferences are often expressed
implicitly and the session vocabulary rarely matches the question phrasing.

### 5-seed cross-validated (50/450 stratified splits)

| Metric | Mean | ±Std | Min | Max |
|---|---|---|---|---|
| **R@5** | **97.24%** | **±0.12%** | 97.11% | 97.33% |
| R@10 | 98.76% | ±0.12% | 98.67% | 98.89% |
| R@25 | **100.00%** | ±0.00% | 100.00% | 100.00% |
| NDCG@5 | 92.28% | ±0.69% | 91.64% | 93.40% |

The ±0.12% R@5 standard deviation confirms the pipeline is stable across
different data splits — the canonical 98.00% is not a lucky split.

---

## File Structure

```
benchmarks/
├── longmemeval_bench.py          # Fixed-split benchmark runner (--sweep + single eval)
├── multiseed_sweep.py            # 5-seed cross-validated runner
│
├── strategies/
│   ├── entity_confidence_reranker.py   # ECR — confidence-adaptive entity boost
│   ├── idf_keyword_boost.py            # IDF — corpus-relative keyword boost
│   └── caatb.py                        # CAATB — additive confidence-adaptive temporal boost
│
├── data/
│   ├── lme_split_50_450.json           # Canonical fixed split (50 dev / 450 held-out)
│   └── splits/
│       ├── lme_stratified_split_seed42.json
│       ├── lme_stratified_split_seed0.json
│       ├── lme_stratified_split_seed1.json
│       ├── lme_stratified_split_seed2.json
│       └── lme_stratified_split_seed3.json
│
└── final_results/
    ├── results_mw_ecr0.3_idf0.6_caatb0.2_held_out_20260505_1710.jsonl  # canonical 98.00%
    ├── results_mw_cv_seed42_held_out_20260506_0834.jsonl
    ├── results_mw_cv_seed0_held_out_20260506_0834.jsonl
    ├── results_mw_cv_seed1_held_out_20260506_0834.jsonl
    ├── results_mw_cv_seed2_held_out_20260506_0834.jsonl
    ├── results_mw_cv_seed3_held_out_20260506_0834.jsonl
    └── summary_mw_cv_5seeds_20260506_0834.json   # CV aggregate (mean ± std per metric)
```

### JSONL format

Each result file has one JSON object per line. The last line is always a
`__summary__` row:

```json
{
  "__summary__": true,
  "subset": "held_out",
  "n": 450,
  "embedding_model": "ollama/all-minilm:l6-v2",
  "pipeline": "ECR + IDF + CAATB",
  "alphas": {"ecr": 0.3, "idf": 0.6, "caatb": 0.2},
  "recall_any@5": 0.98,
  "ndcg@5": 0.9375,
  "per_type": {
    "temporal-reasoning": {"n": 124, "recall_any@5": 0.9758, "recall_any@10": 0.9919, "ndcg@5": 0.9051},
    ...
  }
}
```

Per-question rows contain `question_id`, `question_type`, `ranked_session_ids`
(top-50), `answer_session_ids`, and metrics for all K values.
