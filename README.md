# memweave

**Agent memory you can read, search, and `git diff`.**

[![PyPI](https://img.shields.io/pypi/v/memweave)](https://pypi.org/project/memweave/)
[![Python](https://img.shields.io/pypi/pyversions/memweave)](https://pypi.org/project/memweave/)
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

**Search path** — `await mem.search(query)` embeds the query, runs vector search and keyword search in parallel, merges scores (`0.7 × vector + 0.3 × BM25`), applies post-processors (threshold → MMR → temporal decay), and returns ranked results.

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
| `memory/episodes/event.md` | **Episodic** — named events, timestamped |

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

### Hybrid search

```
query: "which database should I use for JSON?"
         │
         ├─ FTS5 BM25  ──────── exact keywords → score A
         │
         └─ sqlite-vec cosine ─ semantic match  → score B
                    │
                    ▼
         merged = 0.7 × B + 0.3 × A
```

Post-processors run after merging:

- **Score threshold** — drops results below `min_score` (default `0.35`)
- **MMR re-ranking** — penalises redundant results, promotes diversity (disabled by default)
- **Temporal decay** — exponential score reduction by file age, evergreen exempt (disabled by default)

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
