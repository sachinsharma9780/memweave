"""
memweave/config.py — Configuration dataclasses with sensible defaults.

All configuration is optional: MemWeave(workspace_dir="./my_project") is valid.
Every parameter is a knob. Two-layer override:
  1. Config-level defaults (set once at initialization)
  2. Call-site overrides (override per-call via keyword arguments)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast


@dataclass
class EmbeddingConfig:
    """LiteLLM embedding provider settings.

    Controls which model is called when MemWeave needs to convert text into a
    vector (during ``index()`` and ``search()``). LiteLLM is used as the
    provider abstraction, so any of its 100+ supported models work here —
    OpenAI, Cohere, Azure, Bedrock, local Ollama, etc.

    Example — use a local Ollama model instead of OpenAI::

        EmbeddingConfig(
            model="ollama/nomic-embed-text",
            api_base="http://localhost:11434",
        )
    """

    model: str = "text-embedding-3-small"
    """Any LiteLLM-supported embedding model (100+ providers)."""

    api_base: str | None = None
    """Custom API base URL (e.g. local Ollama)."""

    api_key: str | None = None
    """Explicit API key. If None, reads from environment variable."""

    timeout: float = 60.0
    """Seconds per API call before timeout."""

    batch_size: int = 64
    """Max texts per batch embedding call."""

    def __post_init__(self) -> None:
        """Validate that numeric fields are within acceptable ranges.

        Raises:
            ValueError: If ``timeout`` or ``batch_size`` are <= 0.
        """
        if self.timeout <= 0:
            raise ValueError(f"EmbeddingConfig.timeout must be > 0, got {self.timeout}")
        if self.batch_size <= 0:
            raise ValueError(f"EmbeddingConfig.batch_size must be > 0, got {self.batch_size}")


@dataclass
class ChunkingConfig:
    """Markdown chunking parameters.

    Controls how source files are split into searchable pieces. Smaller chunks
    give more precise retrieval; larger chunks give more context per result.
    The ``overlap`` ensures that a sentence crossing a chunk boundary appears
    in both chunks, preventing retrieval gaps.

    Default (tokens=400, overlap=80) is suitable for most use cases with
    ``text-embedding-3-small`` (max 8191 tokens).
    """

    tokens: int = 400
    """Target chunk size in tokens."""

    overlap: int = 80
    """Overlap between consecutive chunks in tokens."""

    def __post_init__(self) -> None:
        """Validate chunk size and overlap constraints.

        Raises:
            ValueError: If ``tokens`` is <= 0, ``overlap`` is < 0, or
                        ``overlap >= tokens`` (overlap must be smaller than chunk).
        """
        if self.tokens <= 0:
            raise ValueError(f"ChunkingConfig.tokens must be > 0, got {self.tokens}")
        if self.overlap < 0:
            raise ValueError(f"ChunkingConfig.overlap must be >= 0, got {self.overlap}")
        if self.overlap >= self.tokens:
            raise ValueError(
                f"ChunkingConfig.overlap ({self.overlap}) must be < tokens ({self.tokens})"
            )

    @property
    def max_chars(self) -> int:
        """Character budget per chunk (approximate: 1 token ≈ 4 chars)."""
        return max(32, self.tokens * 4)

    @property
    def overlap_chars(self) -> int:
        """Character budget for overlap."""
        return max(0, self.overlap * 4)


@dataclass
class HybridConfig:
    """Hybrid search merge weights.

    Hybrid search combines two score components:
    - **Vector score**: semantic similarity between the query embedding and
      chunk embeddings (cosine similarity in the range [0, 1]).
    - **Text score**: keyword relevance via BM25 (normalized to [0, 1]).

    The final score is ``vector_weight * vector_score + text_weight * text_score``.
    The weights must sum to 1.0.

    ``candidate_multiplier`` controls the internal candidate pool size:
    if ``max_results=6``, the vector and text searches each return
    ``6 × candidate_multiplier = 24`` candidates, which are then merged and
    re-ranked before the final top-6 is returned.

    Example — pure vector search::

        HybridConfig(vector_weight=1.0, text_weight=0.0)

    Example — pure keyword search::

        HybridConfig(vector_weight=0.0, text_weight=1.0)
    """

    vector_weight: float = 0.7
    """Weight for vector (semantic) search score. Must sum to 1.0 with text_weight."""

    text_weight: float = 0.3
    """Weight for keyword (BM25) search score."""

    candidate_multiplier: int = 4
    """Internal candidate pool = top_k × candidate_multiplier."""

    def __post_init__(self) -> None:
        """Validate that weights are in [0, 1] and sum to 1.0.

        Raises:
            ValueError: If either weight is outside [0, 1], weights don't sum
                        to 1.0 (within 1e-6 tolerance), or multiplier <= 0.
        """
        if not 0.0 <= self.vector_weight <= 1.0:
            raise ValueError(
                f"HybridConfig.vector_weight must be in [0, 1], got {self.vector_weight}"
            )
        if not 0.0 <= self.text_weight <= 1.0:
            raise ValueError(f"HybridConfig.text_weight must be in [0, 1], got {self.text_weight}")
        if abs(self.vector_weight + self.text_weight - 1.0) > 1e-6:
            raise ValueError(
                f"HybridConfig.vector_weight + text_weight must equal 1.0, "
                f"got {self.vector_weight} + {self.text_weight} = {self.vector_weight + self.text_weight}"
            )
        if self.candidate_multiplier <= 0:
            raise ValueError(
                f"HybridConfig.candidate_multiplier must be > 0, got {self.candidate_multiplier}"
            )


@dataclass
class MMRConfig:
    """Maximal Marginal Relevance (MMR) reranking settings.

    MMR is a post-processing step that increases result diversity by
    penalizing chunks that are too similar to already-selected results.
    This prevents the top results from all being excerpts from the same
    paragraph.

    The algorithm (greedy selection):
    1. Start with the highest-scoring chunk.
    2. For each subsequent slot, score remaining candidates as:
       ``lambda * relevance - (1 - lambda) * max_similarity_to_selected``
    3. Select the candidate with the highest adjusted score.
    4. Repeat until ``max_results`` is filled.

    ``lambda_param`` controls the diversity/relevance trade-off:
    - ``1.0`` → pure relevance (same as no MMR)
    - ``0.0`` → pure diversity (maximally different results)
    - ``0.7`` (default) → slight diversity bias while keeping high relevance
    """

    enabled: bool = False
    """Enable MMR reranking after search."""

    lambda_param: float = 0.7
    """Trade-off between relevance and diversity. 0=diversity, 1=relevance."""

    def __post_init__(self) -> None:
        """Validate lambda_param is in [0, 1].

        Raises:
            ValueError: If ``lambda_param`` is outside [0, 1].
        """
        if not 0.0 <= self.lambda_param <= 1.0:
            raise ValueError(f"MMRConfig.lambda_param must be in [0, 1], got {self.lambda_param}")


@dataclass
class TemporalDecayConfig:
    """Exponential temporal decay for search scores.

    Applies a recency bias so that newer memories surface above older ones
    when relevance scores are otherwise equal. Decay follows the formula::

        decay_factor = e^(-λ × age_days)
        λ = ln(2) / half_life_days
        final_score = base_score * decay_factor

    For example, with ``half_life_days=30``:
    - A 0-day-old chunk retains 100% of its score.
    - A 30-day-old chunk retains 50% of its score.
    - A 60-day-old chunk retains 25% of its score.

    Evergreen files (MEMORY.md, non-dated reference files) are always exempt
    from decay regardless of age.
    """

    enabled: bool = False
    """Enable recency bias (newer memories score higher)."""

    half_life_days: float = 30.0
    """Half-life in days. Score halves every this many days. Evergreen files exempt."""

    def __post_init__(self) -> None:
        """Validate that half_life_days is positive.

        Raises:
            ValueError: If ``half_life_days`` is <= 0.
        """
        if self.half_life_days <= 0:
            raise ValueError(
                f"TemporalDecayConfig.half_life_days must be > 0, got {self.half_life_days}"
            )


@dataclass
class QueryConfig:
    """Search query defaults and post-processor settings.

    These are the defaults used when calling ``mem.search(query)`` without
    any overrides. Every field can be overridden per-call::

        # Use defaults
        results = await mem.search("deployment steps")

        # Override for this call only
        results = await mem.search(
            "deployment steps",
            max_results=10,
            min_score=0.5,
        )
    """

    strategy: str = "hybrid"
    """Default search strategy: 'hybrid' | 'vector' | 'keyword' | custom registered name."""

    max_results: int = 6
    """Max results returned per search."""

    min_score: float = 0.35
    """Filter threshold — results below this score are dropped."""

    snippet_max_chars: int = 700
    """Truncate result snippet to this many characters."""

    hybrid: HybridConfig = field(default_factory=HybridConfig)
    """Hybrid search weights."""

    mmr: MMRConfig = field(default_factory=MMRConfig)
    """MMR reranking settings."""

    temporal_decay: TemporalDecayConfig = field(default_factory=TemporalDecayConfig)
    """Temporal decay settings."""

    def __post_init__(self) -> None:
        """Validate numeric fields.

        Raises:
            ValueError: If ``max_results`` <= 0, ``min_score`` outside [0, 1],
                        or ``snippet_max_chars`` <= 0.
        """
        if self.max_results <= 0:
            raise ValueError(f"QueryConfig.max_results must be > 0, got {self.max_results}")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(f"QueryConfig.min_score must be in [0, 1], got {self.min_score}")
        if self.snippet_max_chars <= 0:
            raise ValueError(
                f"QueryConfig.snippet_max_chars must be > 0, got {self.snippet_max_chars}"
            )


@dataclass
class CacheConfig:
    """Embedding cache settings.

    The cache stores computed embedding vectors in SQLite, keyed by the SHA-256
    of the chunk text. On subsequent indexing runs, if the text hasn't changed,
    the stored vector is reused — no API call is made.

    Without the cache, every ``index()`` call would re-embed all chunks.
    With the cache, only new or changed chunks hit the embedding API.

    ``max_entries=None`` means unlimited. Set a limit to cap SQLite disk usage.
    When the limit is exceeded, the oldest entries (by ``updated_at``) are
    evicted (LRU eviction).
    """

    enabled: bool = True
    """Enable embedding cache (avoids re-embedding identical chunks)."""

    max_entries: int | None = None
    """LRU eviction cap. None = unlimited. Set to limit disk usage."""

    def __post_init__(self) -> None:
        """Validate max_entries if set.

        Raises:
            ValueError: If ``max_entries`` is set to a non-positive value.
        """
        if self.max_entries is not None and self.max_entries <= 0:
            raise ValueError(f"CacheConfig.max_entries must be > 0 or None, got {self.max_entries}")


@dataclass
class SyncConfig:
    """File sync and watch settings.

    Controls when MemWeave re-scans the workspace for changed markdown files.

    Three sync modes (can be combined):
    - **on_search** (default on): check for dirty files just before each
      ``search()`` call. Ensures results are always fresh without requiring
      manual ``index()`` calls.
    - **watch**: use ``watchfiles`` to monitor the workspace directory and
      re-index immediately when any ``.md`` file changes. Lower latency than
      ``on_search``, but requires ``MemWeave.start_watching()`` to be called.
    - **interval_minutes**: background periodic sync every N minutes.
      Useful for long-running agents that may miss watch events.
    """

    on_search: bool = True
    """Auto-sync before search if dirty files exist."""

    watch: bool = False
    """Enable file watcher (start_watching() must still be called explicitly)."""

    watch_debounce_ms: int = 1500
    """Debounce file watcher events (milliseconds)."""

    interval_minutes: int = 0
    """Periodic sync interval. 0 = disabled."""

    def __post_init__(self) -> None:
        """Validate debounce and interval values.

        Raises:
            ValueError: If ``watch_debounce_ms`` or ``interval_minutes`` are negative.
        """
        if self.watch_debounce_ms < 0:
            raise ValueError(
                f"SyncConfig.watch_debounce_ms must be >= 0, got {self.watch_debounce_ms}"
            )
        if self.interval_minutes < 0:
            raise ValueError(
                f"SyncConfig.interval_minutes must be >= 0, got {self.interval_minutes}"
            )


@dataclass
class FlushConfig:
    """Memory flush (LLM-driven fact extraction) settings.

    ``flush()`` is the only MemWeave operation that makes LLM API calls.
    It is called before context window compaction to extract durable facts
    from the current conversation and write them to a dated markdown file
    (``memory/YYYY-MM-DD.md``).

    The default ``system_prompt`` instructs the LLM to:
    - Append new entries to the current day's memory file (never overwrite).
    - Treat MEMORY.md and other bootstrap files as read-only.
    - Reply with ``@@SILENT_REPLY@@`` if there is nothing worth storing.

    To customize extraction behavior, override ``system_prompt``.
    """

    enabled: bool = True
    """Enable memory flush functionality."""

    model: str = "gpt-4o-mini"
    """LiteLLM model for fact extraction. Any supported model."""

    max_tokens: int = 1024
    """Max tokens in LLM extraction response."""

    temperature: float = 0.0
    """LLM temperature. 0.0 = deterministic (recommended for extraction)."""

    system_prompt: str = (
        "Pre-compaction memory flush.\n"
        "Store durable memories only in memory/YYYY-MM-DD.md (create memory/ if needed).\n"
        "Treat workspace bootstrap/reference files such as MEMORY.md as read-only during "
        "this flush; never overwrite, replace, or edit them.\n"
        "If memory/YYYY-MM-DD.md already exists, APPEND new content only and do not overwrite "
        "existing entries.\n"
        "Do NOT create timestamped variant files (e.g., YYYY-MM-DD-HHMM.md); always use the "
        "canonical YYYY-MM-DD.md filename.\n"
        "If nothing to store, reply with @@SILENT_REPLY@@."
    )
    """System prompt for LLM extraction.

    The literal token ``YYYY-MM-DD`` is replaced with today's ISO date before
    the LLM call, so the model always writes the correct date in headings.
    Custom prompts that include ``YYYY-MM-DD`` benefit from the same injection.
    Prompts without the token are passed through unchanged.
    """

    def __post_init__(self) -> None:
        """Validate max_tokens and temperature.

        Raises:
            ValueError: If ``max_tokens`` <= 0 or ``temperature`` outside [0, 2].
        """
        if self.max_tokens <= 0:
            raise ValueError(f"FlushConfig.max_tokens must be > 0, got {self.max_tokens}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"FlushConfig.temperature must be in [0, 2], got {self.temperature}")


@dataclass
class VectorConfig:
    """Vector search backend settings.

    Controls whether ANN vector search is used. When ``enabled=True`` (default),
    MemWeave attempts to load ``sqlite-vec`` and create the ``chunks_vec``
    virtual table. If loading fails, it gracefully falls back to FTS-only search.

    ``extension_path`` lets you point to a specific ``sqlite-vec`` binary
    (e.g. ``/usr/local/lib/vec0.so``). If ``None``, MemWeave tries common
    paths and the system ``LD_LIBRARY_PATH``.
    """

    enabled: bool = True
    """Enable vector search. If False, keyword-only mode."""

    extension_path: str | None = None
    """Path to sqlite-vec extension binary. None = auto-detect."""


@dataclass
class MemoryConfig:
    """Root configuration for MemWeave. Every field has a sensible default.

    The minimal useful configuration is just a ``workspace_dir``::

        config = MemoryConfig(workspace_dir="/my/project")

    Everything else resolves automatically:
    - SQLite database → ``workspace_dir/.memweave/index.sqlite``
    - Memory files  → ``workspace_dir/memory/*.md``
    - Embedding model → ``text-embedding-3-small`` (via LiteLLM/OpenAI)

    To change the embedding model::

        config = MemoryConfig(
            workspace_dir="/my/project",
            embedding=EmbeddingConfig(model="ollama/nomic-embed-text",
                                      api_base="http://localhost:11434"),
        )

    To serialize/deserialize (e.g. for config file persistence)::

        # Save
        data = config.to_dict()
        with open("memweave.json", "w") as f:
            json.dump(data, f)

        # Load
        with open("memweave.json") as f:
            data = json.load(f)
        config = MemoryConfig.from_dict(data)
    """

    workspace_dir: str | Path = "~/.memweave/default"
    """Root directory for memory files. Sub-directories: memory/, .memweave/."""

    db_path: str | Path | None = None
    """SQLite database path. None = workspace_dir/.memweave/index.sqlite."""

    timezone: str = "UTC"
    """Timezone for dated file naming (YYYY-MM-DD.md)."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    flush: FlushConfig = field(default_factory=FlushConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)

    progress: bool = True
    """Print human-readable progress lines to stdout during operations.
    Set to False to silence all progress output (e.g. when embedding memweave
    inside a server or test suite)."""

    extra_paths: list[str] = field(default_factory=list)
    """Additional directories to include in memory search."""

    bootstrap_files: list[str] = field(default_factory=lambda: ["MEMORY.md"])
    """Files pre-loaded as evergreen memory on init."""

    evergreen_patterns: list[str] = field(default_factory=lambda: ["MEMORY.md", "memory.md"])
    """File patterns exempt from temporal decay."""

    def __post_init__(self) -> None:
        """Normalize workspace_dir and db_path to resolved Path objects."""
        # Normalize paths
        self.workspace_dir = Path(self.workspace_dir).expanduser()
        if self.db_path is not None:
            self.db_path = Path(self.db_path).expanduser()

    @property
    def resolved_db_path(self) -> Path:
        """Return the effective SQLite database path.

        If ``db_path`` was set explicitly, return it. Otherwise, derive the
        default path as ``workspace_dir/.memweave/index.sqlite``.

        Returns:
            Absolute ``Path`` to the SQLite database file.
        """
        if self.db_path is not None:
            return Path(self.db_path)
        return Path(self.workspace_dir) / ".memweave" / "index.sqlite"

    @property
    def memory_dir(self) -> Path:
        """Return the path to the memory markdown files directory.

        Returns:
            ``workspace_dir/memory/`` as a ``Path``.
        """
        return Path(self.workspace_dir) / "memory"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this config to a plain nested dictionary.

        All ``Path`` values are converted to strings to ensure the result is
        JSON-serializable and can be passed back to ``from_dict()`` to
        reconstruct an equivalent config.

        Returns:
            Nested dict representation of all config fields.
        """
        import pathlib
        from dataclasses import asdict

        def _convert(obj: Any) -> Any:
            if isinstance(obj, pathlib.PurePath):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        return cast(dict[str, Any], _convert(asdict(self)))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryConfig:
        """Reconstruct a ``MemoryConfig`` from a plain nested dictionary.

        Handles reconstruction of all nested dataclasses (``EmbeddingConfig``,
        ``QueryConfig``, etc.) from their dict representations. Fields not
        present in ``data`` use the dataclass defaults.

        Steps:
        1. Extract and reconstruct each nested config object individually.
        2. Collect top-level scalar fields (``workspace_dir``, ``timezone``, etc.).
        3. Pass everything to ``cls(...)`` for final construction (which runs
           ``__post_init__`` validation).

        Args:
            data: Dict from ``to_dict()`` or a manually constructed equivalent.

        Returns:
            Fully constructed ``MemoryConfig`` with validated fields.

        Raises:
            ValueError: If any nested config has invalid field values
                        (e.g. ``HybridConfig`` weights don't sum to 1.0).
        """
        # Reconstruct nested dataclasses
        embedding = EmbeddingConfig(**data.get("embedding", {}))
        chunking = ChunkingConfig(**data.get("chunking", {}))
        hybrid = HybridConfig(**data.get("query", {}).get("hybrid", {}))
        mmr = MMRConfig(**data.get("query", {}).get("mmr", {}))
        decay = TemporalDecayConfig(**data.get("query", {}).get("temporal_decay", {}))
        query_data = {
            k: v
            for k, v in data.get("query", {}).items()
            if k not in ("hybrid", "mmr", "temporal_decay")
        }
        query = QueryConfig(hybrid=hybrid, mmr=mmr, temporal_decay=decay, **query_data)
        cache = CacheConfig(**data.get("cache", {}))
        sync = SyncConfig(**data.get("sync", {}))
        flush = FlushConfig(**data.get("flush", {}))
        vector = VectorConfig(**data.get("vector", {}))
        top_level = {
            k: v
            for k, v in data.items()
            if k not in ("embedding", "chunking", "query", "cache", "sync", "flush", "vector")
        }
        return cls(
            embedding=embedding,
            chunking=chunking,
            query=query,
            cache=cache,
            sync=sync,
            flush=flush,
            vector=vector,
            **top_level,
        )
