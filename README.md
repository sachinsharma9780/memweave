# memweave

**Agent memory you can read, search, and `git diff`.**

[![PyPI](https://img.shields.io/pypi/v/memweave)](https://pypi.org/project/memweave/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/memweave?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/memweave)
[![Python](https://img.shields.io/pypi/pyversions/memweave)](https://pypi.org/project/memweave/)
[![CI](https://github.com/sachinsharma9780/memweave/actions/workflows/ci.yml/badge.svg)](https://github.com/sachinsharma9780/memweave/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sachinsharma9780/memweave/branch/main/graph/badge.svg)](https://codecov.io/gh/sachinsharma9780/memweave)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

memweave is a zero-infrastructure, async-first Python library that gives AI agents persistent, searchable memory — stored as plain Markdown files and indexed by SQLite. No external services. No black-box databases. Every memory is a file you can open, edit, grep, and version-control.

---

## 💡 Why memweave?

- 📄 **Human-readable by design.** Memories live in plain `.md` files on disk. Open them in your editor, inspect them in your terminal, or `git diff` what your agent learned between runs.
- 🔍 **Hybrid search out of the box.** Combines BM25 keyword ranking (FTS5) with semantic vector search (sqlite-vec) and merges them — so "PostgreSQL JSONB" finds both exact matches and conceptually related content.
- ⚡ **Zero LLM calls on core operations.** Writing and searching memories never touches an LLM. Embeddings are cached by content hash — compute once, reuse forever.
- 🌐 **Works completely offline.** If your embedding API is down, memweave falls back to pure keyword search. It never crashes; it degrades gracefully.
- 💸 **Zero server cost, zero setup.** The entire memory store is a single SQLite file on disk — no vector database to provision, no cloud service to pay for, no Docker container to manage.
- 🔌 **Pluggable at every layer.** Swap in a custom search strategy, add a post-processing step, or bring your own embedding provider via a single protocol.
- 📅 **Memories age naturally.** Recent knowledge ranks above stale context automatically — no manual cleanup, no ever-growing noise. Foundational facts stay exempt.
- 🎯 **No redundant results.** MMR re-ranking ensures the top results cover different aspects of your query — not the same fact repeated from five slightly different chunks.

---

## 📋 Table of contents

- [Quickstart Guide](#-quickstart-guide)
- [How it works](#️-how-it-works)
- [Core concepts](#-core-concepts)
  - [Markdown as the source of truth](#markdown-as-the-source-of-truth)
  - [Evergreen vs dated files](#evergreen-vs-dated-files)
  - [Agent namespaces & source labels](#agent-namespaces--source-labels)
  - [Search pipeline](#search-pipeline)
  - [Temporal decay](#temporal-decay)
  - [MMR re-ranking](#mmr-re-ranking)
- [CLI](#-cli)
  - [index](#memweave-index)
  - [add](#memweave-add-file)
  - [files](#memweave-files)
  - [search](#memweave-search-query)
  - [stats](#memweave-stats)
- [Usage examples](#-usage-examples)
  - [Single agent with persistent memory](#single-agent-with-persistent-memory)
  - [Multi-agent with shared and isolated namespaces](#multi-agent-with-shared-and-isolated-namespaces)
  - [Memory flush](#memory-flush--persist-conversation-facts-before-context-compaction)
  - [Custom search strategy](#custom-search-strategy)
  - [File watcher](#file-watcher--auto-reindex-on-file-change)
  - [Inspect memory status](#inspect-memory-status)
  - [List indexed files](#list-indexed-files)
- [Configuring memweave](#-configuring-memweave)
  - [Embedding providers](#embedding-providers)
- [API reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#️-license)

---

## 🚀 Quickstart Guide

```bash
pip install memweave
```

Set an embedding provider (or skip to use keyword-only mode):

```bash
export OPENAI_API_KEY=sk-...
```

```python
import asyncio
from pathlib import Path
from memweave import MemWeave, MemoryConfig

async def main():
    async with MemWeave(MemoryConfig(workspace_dir=".")) as mem:
        # Write a memory file, then index it
        memory_file = Path("memory/preferences.md")
        memory_file.parent.mkdir(exist_ok=True)
        memory_file.write_text("The user prefers dark mode and concise answers.")
        await mem.add(memory_file)

        # Search across all memories.
        # min_score=0.0 ensures results surface in a small corpus;
        # in production the default 0.35 threshold filters low-confidence matches.
        results = await mem.search("What is the user preference?", min_score=0.0)
        for r in results:
            print(f"[{r.score:.2f}] {r.snippet}  ← {r.path}:{r.start_line}")

asyncio.run(main())
```

Memories are plain Markdown files in `memory/`. Inspect them any time:

```bash
cat memory/*.md
```

Each result includes a relevance score and the exact file and line it came from:

```
[0.35] The user prefers dark mode and concise answers.  ← memory/preferences.md:1
```

---

## ⚙️ How it works

memweave separates **storage** from **search**:

```
┌──────────────────────────────────────────────────────────────┐
│                 SOURCE OF TRUTH  (Markdown files)            │
│   memory/MEMORY.md        ← evergreen knowledge              │
│   memory/2026-03-21.md    ← daily logs                       │
│   memory/agents/coder/    ← agent-scoped namespace           │
└───────────────────────┬──────────────────────────────────────┘
                        │  chunking → hashing → embedding
┌───────────────────────▼──────────────────────────────────────┐
│                DERIVED INDEX  (SQLite)                       │
│   chunks          — text + metadata                          │
│   chunks_fts      — FTS5 full-text index  (BM25)             │
│   chunks_vec      — sqlite-vec SIMD index (cosine)           │
│   embedding_cache — hash → vector  (skip re-embedding)       │
│   files           — SHA-256 change detection                 │
└───────────────────────┬──────────────────────────────────────┘
                        │  hybrid merge → post-processing
                        ▼
              list[SearchResult]
```

**Write path** — `await mem.add(path)` takes any Markdown file you've written — dated, evergreen, agent-scoped, or session — chunks it, checks the embedding cache (hash lookup), calls the embedding API only on a miss, and inserts into both the FTS5 and vector tables. No LLM involved.

**Search path** — `await mem.search(query)` embeds the query, runs vector search and keyword search in parallel, merges scores (`0.7 × vector + 0.3 × BM25`), applies post-processors (threshold → temporal decay → MMR), and returns ranked results.

---

## 🧠 Core concepts

### Markdown as the source of truth

The SQLite index is a **derived cache** — always rebuildable from the Markdown files. This means:

- You can edit memories directly in your editor and re-index with `await mem.index()`.
- `git diff memory/` shows exactly what an agent learned between commits.
- Losing the database is not data loss. Losing the files is.

### Evergreen vs dated files

| File | Behaviour |
|------|-----------|
| `memory/MEMORY.md` | **Evergreen** — never decays, write-protected during `flush()` |
| `memory/2026-03-21.md` | **Dated** — subject to temporal decay (older memories rank lower) |
| `memory/researcher_agent/` | **Agent-scoped** — isolated namespace per agent |
| `memory/episodes/known-facts.md` | **Evergreen** — non-dated file in a subdirectory, always full score |
| `memory/sessions/2026-04-01.md` | **Dated** — subdirectory dated file, decays by filename date |

Evergreen files hold foundational facts that should always surface at full score. Dated files accumulate daily learning and fade naturally — recent memories rank higher.

### Agent namespaces & source labels

Every file gets a `source` label derived from its path — the **immediate subdirectory** under `memory/` becomes the label:

| File path | `source` |
|-----------|---------|
| `memory/notes.md` | `"memory"` |
| `memory/sessions/2026-04-03.md` | `"sessions"` |
| `memory/researcher_agent/findings.md` | `"researcher_agent"` |
| Outside `memory/` | `"external"` |

Pass `source_filter="researcher_agent"` to `search()` to scope results exclusively to that namespace. Only the first path component counts — `memory/researcher_agent/sub/x.md` has source `"researcher_agent"`, not `"sub"`.

### Search pipeline

Every `mem.search(query)` call moves through five fixed stages in order:

```
                        query
                          │
             ┌────────────┴────────────┐
             │                         │
     FTS5 BM25 (keyword)      sqlite-vec ANN (semantic)
     exact term matching       cosine similarity
             │                         │
             └────────────┬────────────┘
                          │  weighted merge
                          │  score = 0.7 × vector + 0.3 × BM25
                          │
                 ┌────────▼────────┐
                 │ ScoreThreshold  │  drop results below min_score (default 0.35)
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │ TemporalDecay   │  multiply score by exp(−λ × age_days)
                 │  (opt-in)       │  evergreen files exempt
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │  MMR Reranker   │  reorder for relevance + diversity
                 │  (opt-in)       │  λ × relevance − (1−λ) × similarity_to_selected
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │ Custom          │  your own PostProcessor(s)
                 │ processors      │  via mem.register_postprocessor()
                 └────────┬────────┘
                          │
                   list[SearchResult]
```

**Stage 1 — Hybrid merge.** Both backends run against the same query. FTS5 BM25 catches exact keyword matches (error codes, config values, proper names). sqlite-vec cosine catches semantically related content even when no keyword overlaps. Scores are normalised and merged: `0.7 × vector_score + 0.3 × bm25_score`. Weights are tunable via `HybridConfig`.

**Stage 2 — Score threshold.** Drops any result whose merged score is below `min_score` (default `0.35`). Acts as a noise gate — prevents low-confidence matches from entering the post-processing stages. Always active; override per-call with `mem.search(query, min_score=0.5)`.

**Stage 3 — Temporal decay** *(opt-in).* Multiplies each result's score by an exponential factor based on the age of its source file. Recent memories rank higher; old ones fade naturally. Evergreen files are exempt and always surface at full score. See [Temporal decay](#temporal-decay) below.

**Stage 4 — MMR re-ranking** *(opt-in).* Reorders the remaining results to balance relevance against diversity. Prevents the top results from being near-duplicates of each other. See [MMR re-ranking](#mmr-re-ranking) below.

**Stage 5 — Custom processors.** Any processors registered with `mem.register_postprocessor()` run last, in registration order. Each receives the output of the previous stage and can filter, reorder, or rescore freely.

---

### Temporal decay

Agents accumulate knowledge over time — but not all knowledge ages equally. A decision made yesterday should outrank one made six months ago when both are semantically relevant. Without decay, a stale debugging note from last quarter can surface above this morning's architecture decision simply because it embeds well.

Temporal decay solves this by multiplying each result's score by a factor that shrinks the older the source file is. The score is never zeroed out — old memories still surface, they just rank lower than recent ones.

**How the formula works:**

```
λ            = ln(2) / half_life_days
multiplier   = exp(−λ × age_days)
decayed_score = original_score × multiplier
```

At `age_days = 0` the multiplier is `1.0` — no change. At `age_days = half_life_days` it is exactly `0.5`. The curve is smooth and continuous, so a file that is two half-lives old scores at `0.25×`, three half-lives at `0.125×`, and so on.

With the default `half_life_days=30`:

| File age | Multiplier | Effect on a 0.80 score |
|----------|------------|------------------------|
| Today    | 1.00       | 0.80 (unchanged)       |
| 30 days  | 0.50       | 0.40                   |
| 60 days  | 0.25       | 0.20                   |
| 90 days  | 0.13       | 0.10                   |

**How age is determined — three file categories:**

| File | Age source | Decays? |
|------|------------|---------|
| `memory/MEMORY.md`, `memory/architecture.md` (any non-dated file directly under `memory/`) | — | **No** — evergreen, always full score |
| `memory/agents/notes.md` (non-dated file in any `memory/` subdirectory) | — | **No** — evergreen, same rule as root non-dated files |
| `memory/2026-03-21.md` (dated daily log) | Date parsed from filename | **Yes** |
| `memory/sessions/2026-03-21.md` (dated file in any `memory/` subdirectory) | Date parsed from filename | **Yes** — same rule as root dated files |

Evergreen files hold foundational facts — stack choices, hard constraints, permanent preferences — that should always surface at full score regardless of when they were written. Daily logs capture evolving context and fade naturally as new sessions add fresher knowledge.

**Enabling temporal decay:**

```python
from memweave import MemWeave
from memweave.config import MemoryConfig, QueryConfig, TemporalDecayConfig

config = MemoryConfig(
    query=QueryConfig(
        temporal_decay=TemporalDecayConfig(
            enabled=True,
            half_life_days=30.0,   # score halves every 30 days; tune to your workflow
        ),
    ),
)

async with MemWeave(config) as mem:
    results = await mem.search("database choice")
    # results from last week will rank above results from last quarter
    # results from memory/MEMORY.md are exempt and always surface at full score
```

Tune `half_life_days` to your workflow: `7` for fast-moving projects where week-old context is already stale, `90` for research or documentation repositories where knowledge stays relevant for months.

### MMR re-ranking

Without diversity control, the top results from a hybrid search are often near-duplicates — multiple chunks from the same file, or different phrasings of the same fact. An agent loading all of them into its context window wastes tokens and misses other relevant but different memories.

MMR (Maximal Marginal Relevance) reorders results after scoring to balance how relevant a result is against how similar it is to results already selected. At each step it picks the candidate that maximises:

```
MMR score = λ × relevance − (1−λ) × max_similarity_to_already_selected
```

Similarity is computed as Jaccard overlap between the token sets of the candidate and each already-selected result. This means two chunks that share many of the same words — even from different files — are treated as redundant, and the second one is pushed down in favour of something genuinely different.

**The `lambda_param` dial:**

| `lambda_param` | Behaviour |
|----------------|-----------|
| `1.0` | Pure relevance — identical to no MMR (no-op) |
| `0.7` | Default — strong relevance bias, light diversity push |
| `0.5` | Equal weight — relevance and diversity balanced |
| `0.0` | Pure diversity — maximally novel results, relevance ignored |

**Enabling MMR:**

```python
from memweave import MemWeave
from memweave.config import MemoryConfig, QueryConfig, MMRConfig

config = MemoryConfig(
    query=QueryConfig(
        mmr=MMRConfig(
            enabled=True,
            lambda_param=0.7,   # 0 = max diversity, 1 = max relevance
        ),
    ),
)

async with MemWeave(config) as mem:
    results = await mem.search("deployment steps")
    # top results will cover different aspects of deployment
    # rather than returning the same facts from multiple angles

    # override λ per-call without touching the config
    diverse = await mem.search("deployment steps", mmr_lambda=0.3)
```

MMR runs after temporal decay, so the diversity pass operates on already age-adjusted scores — the reranker sees a realistic picture of which results actually matter before deciding what is redundant.

---

## 💻 CLI

`pip install memweave` registers a `memweave` binary alongside the Python library. Every command is a thin shell over the same `MemWeave` public methods, so anything you can do from Python you can do from a terminal, a shell script, or a CI step — without writing a single line of Python.

This is particularly useful for:

- **Inspecting agent memory** without opening a Python REPL — browse what's indexed, check scores, read snippets directly in the terminal.
- **Shell scripts and CI pipelines** — index a workspace after a build, search for a known fact and fail the pipeline if it isn't there, or export results as JSON for downstream tools.
- **Agents that orchestrate subprocesses** — an LLM running a bash tool can call `memweave search` and parse the JSON output without embedding the library.

All commands accept `--workspace / -w` to point at any directory and `--embedding-model` to override the model. Every command that produces structured output accepts `--json` for machine-readable output.

---

### `memweave index`

Scan the workspace for `.md` files and embed any that have changed since the last run. Uses SHA-256 hashing to skip unchanged files — fast on large workspaces.

```bash
# Index a workspace
memweave index --workspace ./my_project --embedding-model text-embedding-3-small

# Force re-embed everything regardless of hash
memweave index --workspace ./my_project --embedding-model text-embedding-3-small --force
```

---

### `memweave add <file>`

Index a single file immediately. Useful after writing a new memory file and wanting it available in search right away, without scanning the whole workspace.

The `<file>` path is resolved from your **current working directory** (like any shell command), not from `--workspace`. So if your workspace is at `./my_project`, run from its parent:

```bash
# Run from the parent of my_project/
memweave add my_project/memory/2026-04-25.md --workspace ./my_project --embedding-model text-embedding-3-small

# Or cd into the workspace first, then the path is relative to CWD
cd my_project
memweave add memory/infrastructure.md --workspace . --embedding-model text-embedding-3-small

# Force re-index even if the file hasn't changed
memweave add my_project/memory/architecture.md --workspace ./my_project --embedding-model text-embedding-3-small --force
```

---

### `memweave files`

List every file currently tracked in the index with its source label, chunk count, and whether it is evergreen.

```bash
# Filter to a specific source namespace
memweave files --workspace ./my_project --source sessions

# Machine-readable output
memweave files --workspace ./my_project --json

# Table view
memweave files --workspace ./my_project
```

Example output:

```
Path                          Source    Chunks  Evergreen
memory/2026-04-25.md          memory        3   no
memory/architecture.md        memory        5   yes
memory/sessions/2026-04-24.md sessions      2   no
```

---

### `memweave search <query>`

Search the index and return ranked results with relevance scores, file paths, line ranges, source labels, and a content preview. The full search pipeline runs — hybrid (vector + keyword) by default, with optional MMR and temporal decay.

```bash
# Basic search
memweave search "PostgreSQL JSONB" --workspace ./my_project --embedding-model text-embedding-3-small

# Cap results and set a minimum score
memweave search "caching layer" --workspace ./my_project --max-results 3 --min-score 0.3

# Scope to one source namespace
memweave search "deployment steps" --workspace ./my_project --source-filter sessions --embedding-model text-embedding-3-small

# Use keyword-only search (no embedding API needed)
memweave search "Redis ElastiCache" --workspace ./my_project --strategy keyword

# Control diversity vs relevance with MMR (0 = diverse, 1 = relevant)
memweave search "database choice" --workspace ./my_project --mmr-lambda 0.3 --embedding-model text-embedding-3-small

# Apply temporal decay so older memories rank lower
memweave search "architecture decision" --workspace ./my_project --decay-half-life-days 14 --embedding-model text-embedding-3-small

# JSON output — pipe to jq, save to file, or pass to another tool
memweave search "database choice" --workspace ./my_project --embedding-model text-embedding-3-small --json
memweave search "database choice" --workspace ./my_project --embedding-model text-embedding-3-small --json | jq '.[0].snippet'
```

Example table output:

```
Score  Path                          Lines   Source   Preview
 0.91  memory/2026-04-25.md          1–8     memory   PostgreSQL 16 chosen for JSONB support.
 0.74  memory/infrastructure.md      4–11    memory   Production Redis runs on ElastiCache r6g.
 0.61  memory/sessions/findings.md   23–30   sessions Discussed moving from Memcached to Redis.
```

Example JSON output (`--json`):

```json
[
  {
    "path": "memory/2026-04-25.md",
    "start_line": 1,
    "end_line": 8,
    "score": 0.91,
    "snippet": "PostgreSQL 16 chosen for JSONB support.",
    "source": "memory",
    "vector_score": 0.91,
    "text_score": 0.70
  }
]
```

---

### `memweave stats`

Show a summary of the current index state — file and chunk counts, active search mode, embedding cache usage, and whether the index is stale. Prints a warning when files on disk have changed since the last `memweave index` run.

```bash
memweave stats --workspace ./my_project
memweave stats --workspace ./my_project --json
```

Example output:

```
──────────────────────────────────────
  Workspace:        /my_project
  DB path:          /my_project/.memweave/index.sqlite
  Search mode:      hybrid
  Provider:         litellm
  Model:            text-embedding-3-small

  Files:            12
  Chunks:           47
  Cache entries:    47
  Cache max:        unlimited
  Dirty:            no
  Watcher active:   no
  FTS available:    yes
  Vector available: yes
```

---

## 💻 Usage examples

### Single agent with persistent memory

```python
import asyncio
from pathlib import Path
from memweave import MemWeave, MemoryConfig

async def run_agent_session():
    config = MemoryConfig(workspace_dir="./my_project")

    async with MemWeave(config) as mem:
        # Write memory files, then index them
        memory_dir = Path("my_project/memory")
        memory_dir.mkdir(parents=True, exist_ok=True)

        (memory_dir / "stack.md").write_text("User's preferred stack: FastAPI + PostgreSQL + Redis.")
        (memory_dir / "guidelines.md").write_text("Avoid using global state in this codebase.")

        await mem.index()

        # Retrieve relevant context before responding
        context = await mem.search("database recommendations", min_score=0.0, max_results=2)
        for result in context:
            print(f"  [{result.score:.2f}] {result.snippet}  ({result.path}:{result.start_line})")

asyncio.run(run_agent_session())
```

### Multi-Agent with shared and isolated namespaces

Agents share one workspace but write to separate subdirectories under `memory/`. The subdirectory name becomes the `source` label — pass `source_filter="researcher_agent"` to scope a search exclusively to that agent's files.

```python
import asyncio
from pathlib import Path
from memweave import MemWeave, MemoryConfig

async def main():
    # Both agents share the same workspace root
    researcher = MemWeave(MemoryConfig(workspace_dir="./project"))
    writer = MemWeave(MemoryConfig(workspace_dir="./project"))

    async with researcher, writer:
        # Researcher writes space exploration findings to its own namespace
        memory_dir = Path("project/memory/researcher_agent")
        memory_dir.mkdir(parents=True, exist_ok=True)

        (memory_dir / "mars_habitat.md").write_text(
            "Mars surface pressure is ~0.6% of Earth's, requiring fully pressurised habitats. "
            "NASA's MOXIE experiment on Perseverance successfully produced oxygen from CO2 in 2021, "
            "validating in-situ resource utilisation (ISRU) as a viable strategy for long-duration missions."
        )
        (memory_dir / "artemis_mission.md").write_text(
            "Artemis III aims to land the first woman and next man near the lunar south pole. "
            "Permanently shadowed craters there hold water ice deposits confirmed by LCROSS in 2009. "
            "Ice can be electrolysed into hydrogen and oxygen, serving as both breathable air and rocket propellant."
        )
        (memory_dir / "deep_space_propulsion.md").write_text(
            "Ion drives expel charged xenon atoms at ~90,000 km/h, achieving far higher specific impulse "
            "than chemical rockets, though thrust is measured in millinewtons. NASA's Dawn spacecraft used "
            "ion propulsion to orbit both Vesta and Ceres — the first mission to orbit two extraterrestrial bodies."
        )

        await researcher.index()

        # Writer queries the researcher's findings — scoped to the researcher_agent source
        queries = [
            "how do astronauts get oxygen on Mars",
            "water ice on the Moon",
            "spacecraft propulsion beyond chemical rockets",
        ]

        for query in queries:
            print(f"\nQuery: {query!r}")
            results = await writer.search(query, source_filter="researcher_agent", min_score=0.0, max_results=1)
            for r in results:
                print(f"  [{r.score:.2f}] {r.snippet}  ({r.path}:{r.start_line})")

asyncio.run(main())
```

### Memory flush — persist conversation facts before context compaction

LLM context windows are finite. When a long conversation is compacted or a session ends, anything not written to memory is lost. `flush()` solves this by sending the conversation to an LLM with a structured extraction prompt — the model distils durable facts (decisions, preferences, constraints) and discards small talk. The extracted text is appended to the dated memory file (`memory/YYYY-MM-DD.md`) and immediately re-indexed, so it surfaces in future searches. If the LLM finds nothing worth storing it returns a silent sentinel and `flush()` returns `None` — nothing is written.

Requires an LLM API key (configured via `FlushConfig.model`, default `gpt-4o-mini`).

```python
import asyncio
from pathlib import Path
from memweave import MemWeave, MemoryConfig

WORKSPACE = Path(__file__).parent / "workspace"

conversation = [
    {"role": "user",      "content": "We just decided to use Valkey instead of Redis for caching."},
    {"role": "assistant", "content": "Got it. I'll note that Valkey is the new caching layer."},
    {"role": "user",      "content": "Also, we're targeting a 5ms p99 latency SLA for the cache."},
]

async def main():
    config = MemoryConfig(workspace_dir=WORKSPACE)

    async with MemWeave(config) as mem:
        # Extract durable facts from the conversation and write to workspace/memory/YYYY-MM-DD.md.
        # Returns the extracted text, or None if there was nothing worth storing.
        extracted = await mem.flush(conversation=conversation)
        if extracted:
            print(f"Stored:\n{extracted}\n")
        else:
            print("Nothing worth storing.\n")

        # Search the indexed knowledge immediately after flush
        results = await mem.search("Valkey caching latency", min_score=0.0)
        print(f"Search results ({len(results)} hits):")
        for r in results:
            print(f"  [{r.score:.3f}] {r.snippet.strip()}")

asyncio.run(main())
```

### Custom search strategy

The built-in `"hybrid"`, `"vector"`, and `"keyword"` strategies cover most cases, but sometimes you need ranking logic that none of them support — for example, boosting results from recently modified files, hard-pinning results from a specific file to the top, or implementing a completely different scoring algorithm. A custom strategy gives you direct access to the SQLite database, so you can write any query you like and return results in whatever order you want. memweave applies your results through the same post-processing pipeline (score threshold, MMR, temporal decay) as built-in strategies.

Register a strategy once with `mem.register_strategy(name, obj)`, then activate it per-call via `strategy=name`.

```python
import asyncio
import aiosqlite
from memweave import MemWeave, MemoryConfig
from memweave.search.strategy import RawSearchRow

class RecencyBoostStrategy:
    async def search(
        self,
        db: aiosqlite.Connection,
        query: str,
        query_vec: list[float] | None,
        model: str,
        limit: int,
        *,
        source_filter: str | None = None,
    ) -> list[RawSearchRow]:
        # Your custom ranking logic here — query `db` directly and return RawSearchRow objects
        ...

async def main():
    async with MemWeave(MemoryConfig(workspace_dir=".")) as mem:
        mem.register_strategy("recency", RecencyBoostStrategy())
        results = await mem.search("recent decisions", strategy="recency")

asyncio.run(main())
```

### File watcher — auto-reindex on file change

When running a long-lived agent, memory files can be edited externally — by another process, a human, or a separate agent writing to the same workspace. Without the watcher, those changes are invisible until the next explicit `await mem.index()` call. `start_watching()` launches a background task that monitors the `memory/` directory and re-indexes any `.md` file the moment it changes, so searches always reflect the latest content. Rapid successive writes are debounced (default 1500 ms) to avoid redundant re-indexing. The watcher stops automatically when the context manager exits.

Requires the `watchfiles` package (`pip install memweave[watch]`).

```python
import asyncio
from memweave import MemWeave

async def main():
    async with MemWeave() as mem:
        await mem.start_watching()   # starts background task, watches memory/
        # ... run your agent loop
        # any .md file edits are picked up and re-indexed automatically
        # watcher stops automatically on context manager exit

asyncio.run(main())
```

### Inspect memory status

`status()` gives a point-in-time snapshot of the store — how many files and chunks are indexed, which search mode is active (`hybrid`, `fts-only`, or `vector-only`), whether there are unindexed changes pending (`dirty`), and how many embeddings are cached. Useful for health checks, debugging, or surfacing store state in agent logs.

```python
async with MemWeave() as mem:
    status = await mem.status()
    print(f"Files:       {status.files}")
    print(f"Chunks:      {status.chunks}")
    print(f"Search mode: {status.search_mode}")   # hybrid | fts-only | vector-only
    print(f"Dirty:       {status.dirty}")         # unindexed changes pending
```

### List indexed files

`files()` returns metadata for every file currently tracked in the index — path, size, chunk count, source label, and whether the file is evergreen. Useful when an agent needs to audit what it has access to, detect stale files, or decide which namespace to write to next.

```python
async with MemWeave() as mem:
    for f in await mem.files():
        print(f"{f.path}  ({f.chunks} chunks, evergreen={f.is_evergreen}, source={f.source})")
```

---

## 🔧 Configuring memweave

All configuration is optional — sensible defaults work out of the box. Pass a `MemoryConfig` to override.

`MemoryConfig` is a single nested dataclass that groups every tunable knob into focused sub-configs. Each sub-config has its own defaults and can be overridden independently:

- **`EmbeddingConfig`** — which model to use for vectorising text, API key, batch size, timeout.
- **`ChunkingConfig`** — chunk size and overlap in tokens. Smaller chunks give more precise retrieval; larger chunks give more context per result.
- **`QueryConfig`** — default search strategy, max results, score threshold, and the settings for the three built-in post-processors (hybrid weights, MMR, temporal decay).
- **`CacheConfig`** — embedding cache toggle and optional LRU eviction cap to bound disk usage.
- **`SyncConfig`** — when to auto-reindex (before each search, on file change, or on a periodic interval).
- **`FlushConfig`** — the LLM model and system prompt used by `flush()` for fact extraction.

Every field can also be overridden per-call at search time (e.g. `min_score`, `max_results`, `strategy`) without touching the config.

```python
from memweave import MemWeave
from memweave.config import (
    MemoryConfig, EmbeddingConfig, QueryConfig,
    HybridConfig, MMRConfig, TemporalDecayConfig,
    SyncConfig, FlushConfig,
)

config = MemoryConfig(
    workspace_dir="./memory",            # where .md files live

    embedding=EmbeddingConfig(
        model="text-embedding-3-small",  # any LiteLLM-compatible model
        api_key="sk-...",                # or set via environment variable
        batch_size=100,
    ),

    query=QueryConfig(
        strategy="hybrid",               # "hybrid" | "vector" | "keyword"
        max_results=10,
        min_score=0.35,

        hybrid=HybridConfig(
            vector_weight=0.7,           # weight for semantic similarity
            text_weight=0.3,             # weight for BM25 keyword score
        ),

        mmr=MMRConfig(
            enabled=True,
            lambda_param=0.5,            # 0 = max diversity, 1 = max relevance
        ),

        temporal_decay=TemporalDecayConfig(
            enabled=True,
            half_life_days=30.0,         # score halves every 30 days
        ),
    ),

    sync=SyncConfig(
        on_search=True,                  # sync dirty files before each search
        watch=False,                     # enable file watcher
        watch_debounce_ms=500,
    ),

    flush=FlushConfig(
        enabled=True,
        model="gpt-4o-mini",             # LLM used for fact extraction
    ),
)

async with MemWeave(config) as mem:
    ...
```

### Embedding providers

memweave uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood — any LiteLLM-compatible embedding model works with zero code changes:

| Provider | Model example |
|----------|---------------|
| OpenAI | `text-embedding-3-small` |
| Gemini | `gemini/text-embedding-004` |
| Voyage AI | `voyage/voyage-3` |
| Mistral | `mistral/mistral-embed` |
| Ollama (local) | `ollama/nomic-embed-text` |
| Cohere | `cohere/embed-english-v3.0` |

**Ollama (no API key required):**

```python
from memweave.config import MemoryConfig, EmbeddingConfig

config = MemoryConfig(
    embedding=EmbeddingConfig(
        model="ollama/nomic-embed-text",
        api_base="http://localhost:11434",
    )
)
```

**Keyword-only mode (fully offline, no embeddings):**

```python
from memweave.config import MemoryConfig, QueryConfig

config = MemoryConfig(
    query=QueryConfig(strategy="keyword")
)
```

---

## 📖 API reference

### `MemWeave`

| Method | Description |
|--------|-------------|
| `await mem.add(path, *, force=False)` | Index a single Markdown file immediately |
| `await mem.index(*, force=False)` | (Re)index all Markdown files in the workspace |
| `await mem.search(query, *, max_results, min_score, strategy, source_filter)` | Search indexed memories |
| `await mem.flush(conversation, *, model=None, system_prompt=None)` | Extract and persist facts from a conversation via LLM |
| `await mem.status()` | Return `StoreStatus` (file count, chunk count, search mode, …) |
| `await mem.files()` | Return `list[FileInfo]` for all indexed files |
| `await mem.start_watching()` | Start background file watcher (auto-reindex on `.md` changes) |
| `await mem.close()` | Stop watcher and close database |
| `mem.register_strategy(name, strategy)` | Register a custom `SearchStrategy` |
| `mem.register_postprocessor(processor)` | Register a custom `PostProcessor` |

---

## 🤝 Contributing

Issues and pull requests are welcome. Please open an issue before starting large changes.

---

## 🙏 Acknowledgements

🦞 [OpenClaw](https://github.com/openclaw/openclaw) — the memory architecture that inspired memweave.

---

## ⚖️ License

MIT
