"""
memweave/types.py — Public data types returned to callers.

All result types are frozen (immutable) dataclasses with ``slots=True`` for
memory efficiency. Frozen dataclasses are hashable and safe to cache/pass
across threads.

Public types (returned by MemWeave methods):
    SearchResult  — one item in the list returned by ``mem.search()``
    IndexResult   — summary returned by ``mem.index()`` / ``mem.add()``
    FileInfo      — one item in the list returned by ``mem.files()``
    StoreStatus   — snapshot returned by ``mem.status()``

Internal type (used inside the search pipeline, not returned to users):
    ScoredChunk   — intermediate scored chunk before hydration to SearchResult
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result returned by ``MemWeave.search()``.

    Represents one text chunk from the memory store that matched the query.
    Results are sorted by ``score`` (descending) and filtered to those above
    ``QueryConfig.min_score``.

    Attributes:
        path:         Relative path of the source file, e.g. ``"memory/2026-03-21.md"``.
        start_line:   1-indexed line number where this chunk starts in the file.
        end_line:     1-indexed line number where this chunk ends in the file.
        score:        Combined relevance score in [0, 1]. Higher is more relevant.
                      Computed as ``vector_weight * vector_score + text_weight * text_score``
                      (before any post-processing).
        snippet:      Truncated chunk text, at most ``QueryConfig.snippet_max_chars`` chars.
        source:       Logical origin of the file: ``"memory"`` | ``"sessions"`` | custom.
        vector_score: Raw cosine similarity from vector search (before hybrid merge).
                      ``None`` if vector search was unavailable or not used.
        text_score:   Raw BM25 score from keyword search (before hybrid merge).
                      ``None`` if FTS search was unavailable or not used.

    Example::

        results = await mem.search("database migration steps")
        for r in results:
            print(f"[{r.score:.2f}] {r.path}:{r.start_line}-{r.end_line}")
            print(r.snippet)
    """

    path: str
    """Relative path of the source file, e.g. 'memory/2026-03-21.md'."""

    start_line: int
    """1-indexed line number where this chunk starts."""

    end_line: int
    """1-indexed line number where this chunk ends."""

    score: float
    """Combined relevance score in [0, 1]. Higher is more relevant."""

    snippet: str
    """Truncated chunk text (see QueryConfig.snippet_max_chars)."""

    source: str
    """Source type: 'memory' | 'sessions' | custom."""

    vector_score: float | None = None
    """Raw vector similarity score (for debugging). None if vector unavailable."""

    text_score: float | None = None
    """Raw BM25 keyword score (for debugging). None if FTS unavailable."""

    def __repr__(self) -> str:
        """Return a compact string representation for logging/debugging.

        Shows path, score (3 decimal places), and first 60 chars of snippet.

        Example::

            SearchResult(path='memory/2026-03-21.md', score=0.842, snippet='Deploy with...')
        """
        score_str = f"{self.score:.3f}"
        snippet_preview = self.snippet[:60].replace("\n", " ")
        if len(self.snippet) > 60:
            snippet_preview += "..."
        return f"SearchResult(path={self.path!r}, score={score_str}, snippet={snippet_preview!r})"


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Summary of an indexing operation returned by ``MemWeave.index()`` or ``add()``.

    Provides a complete audit trail of what happened during a sync:
    how many files were discovered, which were re-indexed vs. skipped
    (unchanged hash), which were removed from the index, how many
    embeddings were computed vs. reused from the cache, and total time.

    Attributes:
        files_scanned:        Total ``.md`` files discovered in the memory directory.
        files_indexed:        Files that were re-chunked and re-embedded
                              (either new or content hash changed).
        files_skipped:        Files skipped because their content hash matched
                              the stored hash (no changes detected).
        files_deleted:        Files removed from the index because they no longer
                              exist on disk.
        chunks_created:       New chunk rows inserted into SQLite ``chunks`` table.
        embeddings_cached:    Chunks whose embedding was reused from the cache
                              (no API call made).
        embeddings_computed:  Chunks whose embedding was freshly computed via API.
        duration_ms:          Total wall-clock time for the indexing operation
                              in milliseconds.

    Example::

        result = await mem.index()
        print(result)
        # IndexResult(files=2/10 indexed, chunks=14, embeddings=14 new/320 cached, duration=1842.3ms)
    """

    files_scanned: int
    """Total files discovered in memory directory."""

    files_indexed: int
    """Files re-indexed (hash changed or force=True)."""

    files_skipped: int
    """Files skipped (hash unchanged)."""

    files_deleted: int
    """Files removed from index (no longer on disk)."""

    chunks_created: int
    """New chunks inserted into SQLite."""

    embeddings_cached: int
    """Embeddings reused from cache (no API call)."""

    embeddings_computed: int
    """New embeddings computed via API."""

    duration_ms: float
    """Total indexing time in milliseconds."""

    def __repr__(self) -> str:
        """Return a compact summary string for logging.

        Example::

            IndexResult(files=2/10 indexed, chunks=14, embeddings=14 new/320 cached, duration=1842.3ms)
        """
        return (
            f"IndexResult(files={self.files_indexed}/{self.files_scanned} indexed, "
            f"chunks={self.chunks_created}, "
            f"embeddings={self.embeddings_computed} new/{self.embeddings_cached} cached, "
            f"duration={self.duration_ms:.1f}ms)"
        )


@dataclass(frozen=True, slots=True)
class FileInfo:
    """Metadata about a single tracked memory file.

    Returned as items in the list from ``MemWeave.files()``.

    Attributes:
        path:        Relative path, e.g. ``"memory/2026-03-21.md"``.
        size:        File size in bytes.
        hash:        SHA-256 hex digest of current file content.
        mtime:       Last modification timestamp (Unix epoch as float).
        chunks:      Number of indexed chunks from this file.
        is_evergreen: ``True`` for MEMORY.md and non-dated reference files
                     (these are exempt from temporal decay).
        source:      Logical origin label: ``"memory"`` | ``"sessions"`` | custom.
    """

    path: str
    """Relative path, e.g. 'memory/2026-03-21.md'."""

    size: int
    """File size in bytes."""

    hash: str
    """SHA-256 of file content."""

    mtime: float
    """Last modification time (unix timestamp)."""

    chunks: int
    """Number of indexed chunks."""

    is_evergreen: bool
    """True for MEMORY.md and non-dated files (no temporal decay)."""

    source: str
    """Source type: 'memory' | 'sessions' | custom."""


@dataclass(frozen=True, slots=True)
class StoreStatus:
    """Runtime status snapshot returned by ``MemWeave.status()``.

    Provides a read-only view of the current state of the memory store —
    useful for health checks, dashboards, and debugging.

    Attributes:
        files:             Total number of indexed files.
        chunks:            Total number of indexed chunks.
        dirty:             ``True`` if any files have changed since the last
                           ``index()`` call (unindexed changes pending).
        workspace_dir:     Absolute path to the workspace directory.
        db_path:           Absolute path to the SQLite database file.
        search_mode:       Active search capability:
                           ``"hybrid"`` | ``"fts-only"`` | ``"vector-only"`` | ``"unavailable"``.
        provider:          Embedding provider: ``"litellm"`` | ``"none"``.
        model:             Embedding model name, or ``None`` if not configured.
        fts_available:     ``True`` if FTS5 keyword search is available in this SQLite build.
        vector_available:  ``True`` if ``sqlite-vec`` extension is loaded and
                           ``chunks_vec`` table exists.
        cache_entries:     Number of cached embedding vectors.
        cache_max_entries: LRU cap for the embedding cache, or ``None`` if unlimited.
        watcher_active:    ``True`` if the file watcher background task is running.
    """

    files: int
    """Total indexed files."""

    chunks: int
    """Total indexed chunks."""

    dirty: bool
    """True if there are unindexed changes (files changed since last sync)."""

    workspace_dir: str
    """Absolute path to workspace directory."""

    db_path: str
    """Absolute path to SQLite database."""

    search_mode: str
    """Active search mode: 'hybrid' | 'fts-only' | 'vector-only' | 'unavailable'."""

    provider: str
    """Embedding provider name: 'litellm' | 'none'."""

    model: str | None
    """Embedding model name, or None if not configured."""

    fts_available: bool
    """True if FTS5 keyword search is available."""

    vector_available: bool
    """True if sqlite-vec vector search is available."""

    cache_entries: int
    """Number of cached embeddings."""

    cache_max_entries: int | None
    """LRU cap for cache, or None if unlimited."""

    watcher_active: bool
    """True if file watcher is running."""


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """Internal pipeline type — a chunk ID paired with its relevance score.

    Search strategies (``HybridSearch``, ``VectorSearch``, ``KeywordSearch``)
    return lists of ``ScoredChunk`` rather than full ``SearchResult`` objects.
    This decouples the scoring logic from the database hydration step:

    Pipeline flow::

        Strategy.search(query) → list[ScoredChunk]
            ↓  (post-processors: ScoreThreshold, MMR, TemporalDecay)
        list[ScoredChunk]  (filtered and reranked)
            ↓  (hydration: look up chunk text/path/lines in SQLite)
        list[SearchResult]  (returned to caller)

    The ``chunk_id`` is resolved to a full ``SearchResult`` by the hydration
    step after all post-processors have run. This avoids loading chunk text
    for candidates that will be filtered out.

    Attributes:
        chunk_id:    Internal SHA-256 chunk ID (from ``make_chunk_id()``).
        score:       Combined relevance score in [0, 1] after hybrid merge.
        vector_score: Vector similarity component before hybrid merge.
                      ``None`` if vector search was not used.
        text_score:  BM25 keyword component before hybrid merge.
                     ``None`` if FTS was not used.
    """

    chunk_id: str
    """Internal chunk ID (SHA-256 of source:path:start:end:hash:model)."""

    score: float
    """Relevance score in [0, 1]."""

    vector_score: float | None = None
    """Vector similarity component (before hybrid merge)."""

    text_score: float | None = None
    """Keyword BM25 component (before hybrid merge)."""
