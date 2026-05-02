# Memweave × LongMemEval — Benchmark Results

Retrieval benchmarks for memweave on the
[LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
dataset. Results are directly comparable to
[mempalace's published baseline](https://github.com/anhtranguyen-github/mempalace)
because we use identical conditions: same dataset, same train/test split
(`lme_split_50_450.json`, seed=42), same embedding model, and the same
primary metric (Recall@5 any).

**Integrity rule:** all development and parameter decisions were made against
the dev split only. The held-out result is a single clean measurement —
no tuning was done against held-out failures.

---

## Setup

| Parameter | Value |
|---|---|
| Dataset | LongMemEval-S (`longmemeval_s_cleaned.json`, 500 questions) |
| Split | 50 dev / 450 held-out — `lme_split_50_450.json`, seed=42 |
| Embedding model | `all-MiniLM-L6-v2` via Ollama (`ollama/all-minilm:l6-v2`) |
| Indexed content | User turns only (matches mempalace raw mode) |
| Retrieval strategy | Pure vector (cosine similarity) |
| min\_score | 0.0 (no threshold — let @K scoring handle filtering) |
| max\_results | 200 candidates before @K cutoff |
| MMR | Disabled |
| Temporal decay | Disabled |
| Per-question isolation | Fresh `tempdir` per question (equivalent to mempalace's `EphemeralClient`) |

---

## Headline Numbers

| Metric | Dev (50 Qs) | Held-Out (450 Qs) | Mempalace Raw (500 Qs) |
|---|---|---|---|
| **Recall@5 (any)** | **94.00%** | **94.67%** | 96.60% |
| Recall@10 (any) | 94.00% | 97.56% | — |
| Recall@1 (any) | 82.00% | 81.11% | — |
| Recall@3 (any) | 92.00% | 92.00% | — |
| Recall@5 (all) | 80.00% | 82.89% | — |
| NDCG@5 | 88.07% | 87.89% | — |
| NDCG@10 | 87.88% | 88.48% | — |
| Avg time / question | 8,579 ms | 8,335 ms | — |

> **Gap to mempalace raw baseline: −1.93 pp** (94.67% vs 96.6%).
> Mempalace applies no dataset-specific heuristics in raw mode either,
> so this is a fair apples-to-apples comparison.

---

## Per-Type Breakdown — Held-Out (450 Questions)

| Question Type | n | Recall@5 (any) | Recall@5 (all) | NDCG@5 |
|---|---|---|---|---|
| knowledge-update | 69 | **98.55%** | 92.75% | 93.67% |
| multi-session | 115 | 96.52% | 73.91% | 91.04% |
| single-session-assistant | 54 | 92.59% | 92.59% | 91.46% |
| single-session-preference | 25 | 92.00% | 92.00% | 75.19% |
| single-session-user | 63 | 92.06% | 92.06% | 86.27% |
| temporal-reasoning | 124 | 93.55% | 75.00% | 83.59% |
| **Overall** | **450** | **94.67%** | **82.89%** | **87.89%** |

## Per-Type Breakdown — Dev (50 Questions)

| Question Type | n | Recall@5 (any) | Recall@5 (all) | NDCG@5 |
|---|---|---|---|---|
| knowledge-update | 9 | 100.00% | 88.89% | 94.93% |
| multi-session | 18 | 100.00% | 77.78% | 93.60% |
| single-session-assistant | 2 | 50.00% | 50.00% | 50.00% |
| single-session-preference | 5 | 100.00% | 100.00% | 100.00% |
| single-session-user | 7 | 71.43% | 71.43% | 58.02% |
| temporal-reasoning | 9 | 100.00% | 77.78% | 95.34% |
| **Overall** | **50** | **94.00%** | **80.00%** | **88.07%** |

> The `single-session-assistant` and `single-session-user` outliers in dev
> (50%, 71%) are small-sample noise — both recover to ~92% in the 450-question
> held-out run.

---

## Key Observations

### 1. Dev predicts held-out almost exactly (94.0% → 94.67%)

The 0.67 pp difference is within noise for a 50-question sample. This
confirms the split is well-calibrated and there is no overfitting risk —
we made no parameter decisions based on held-out failures.

### 2. The gap to mempalace is a ranking problem, not a coverage problem

Recall@10 is **97.56%** — meaning the correct session is found in the top
10 for nearly all questions. The 1.93 pp gap to mempalace's Recall@5 is
almost entirely cases where the correct session lands at rank 6–10 rather
than missing entirely. This points directly at ranking quality, not
retrieval coverage.

**Implication:** hybrid search (BM25 + vector) and MMR should close most
of this gap without any dataset-specific tuning.

### 3. knowledge-update leads at 98.55%

Questions about changed facts or updated preferences are the easiest for
pure vector search. The most recent session containing the updated fact
dominates cosine similarity naturally — this is exactly the retrieval
pattern raw vector search is built for.

### 4. multi-session benefits from redundancy (96.52%)

Multi-session questions have multiple correct sessions. Recall@5 just
needs *any one* of them to land in the top 5 — the redundancy acts as a
retrieval buffer. Note that Recall@5 (all) drops to 73.91%, meaning
retrieving *every* correct session in top 5 is harder, which is expected.

### 5. temporal-reasoning is strong despite requiring time-based logic (93.55%)

These questions require temporal reasoning to *answer* (e.g., "what did I
say before my trip?"), but the *retrieval* task is still semantic matching.
The model doesn't need to understand dates to find the right session —
it just needs to match the topic. The 83.59% NDCG@5 (lowest of any type)
reflects that when the correct session is found, it sometimes ranks 3rd or
4th rather than 1st.

### 6. single-session types cluster at ~92% — the paraphrase gap

All three single-session types (user, assistant, preference) score around
92%. These share a structural challenge: there is exactly one correct
session and no redundancy. If the question phrasing is semantically distant
from how the user originally expressed the information, cosine similarity
alone may not rank it first.

Example failure pattern: question asks "What programming language do you
prefer?" but the session only contains "I've been writing Rust for two
years" — semantically related but not close in embedding space.

BM25 keyword matching (hybrid mode) is expected to help here by catching
exact token overlaps that vector search misses.

### 7. single-session-assistant recovers from the dev anomaly

The dev split had only 2 `single-session-assistant` questions and scored
50% — a meaningless sample. The held-out 54 questions score 92.59%, in
line with the other single-session types. This confirms the dev anomaly
was pure small-sample noise.

---

## Result Files

| File | Split | Mode | n | R@5 |
|---|---|---|---|---|
| `results/results_mw_raw_dev_20260501_1727.jsonl` | dev | raw | 50 | 94.00% |
| `results/results_mw_raw_held_out_20260501_1753.jsonl` | held-out | raw | 450 | 94.67% |

---

## Known Issues

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for issues discovered during this
benchmark run.

---

## Next Steps

| Milestone | Mode | Expected improvement |
|---|---|---|
| 6 | Hybrid (vector + BM25) | Close paraphrase gap in single-session types |
| 7 | Hybrid + MMR | Improve ranking quality (NDCG@5), reduce duplicate-session noise |
| 8 | Hybrid + LLM rerank | Final pass on hard cases |
