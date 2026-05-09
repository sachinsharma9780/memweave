"""
memweave/store.py — MemWeave main store class.

Implements the core memory store logic.

``MemWeave`` is the single entry point for all library operations:

- ``search(query)``  — hybrid semantic + keyword search with optional MMR and
  temporal decay post-processing.
- ``index()``        — scan the workspace for changed ``.md`` files and
  re-embed any that differ from the stored SHA-256 hash.
- ``add(path)``      — index a single file immediately.
- ``flush()``        — extract durable facts from a conversation via LLM and
  append them to the dated memory file (delegates to
  ``memweave.flush.memory_flush``).
- ``status()``       — snapshot of store state (file/chunk counts, search mode,
  vector availability, watcher status).
- ``files()``        — list of all tracked files with metadata.
- ``close()``        — release the database connection and stop the watcher.

Lifecycle::

    mem = MemWeave(MemoryConfig(workspace_dir="/project"))
    async with mem:               # opens DB, runs schema migrations
        results = await mem.search("deployment steps")
        result  = await mem.index()

    # or manually:
    await mem.open()
    ...
    await mem.close()

Lazy initialization:
    ``_ensure_db()`` is called at the start of every public method. The first
    call opens the database, runs ``ensure_schema()``, optionally loads
    ``sqlite-vec``, and creates the ``SQLiteStore`` wrapper. Subsequent calls
    are a no-op.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiosqlite

from memweave._internal.hashing import (
    make_chunk_id,
    make_provider_key,
    sha256_text,
    truncate_snippet,
)
from memweave._progress import (
    EMOJI_ADD,
    EMOJI_CLOSE,
    EMOJI_DECAY,
    EMOJI_EMBED_API,
    EMOJI_EMBED_CACHE,
    EMOJI_FILES,
    EMOJI_FLUSH_DONE,
    EMOJI_FLUSH_EXTRACT,
    EMOJI_FLUSH_WRITE,
    EMOJI_INDEX_DONE,
    EMOJI_INDEX_FILE,
    EMOJI_INDEX_SCAN,
    EMOJI_MMR,
    EMOJI_OPEN,
    EMOJI_SEARCH,
    EMOJI_SEARCH_DONE,
    EMOJI_SEARCH_HYBRID,
    EMOJI_SEARCH_KW,
    EMOJI_SEARCH_VEC,
    EMOJI_STATUS,
    EMOJI_WARN,
    emit,
)
from memweave.chunking.markdown import chunk_markdown
from memweave.config import MemoryConfig
from memweave.embedding.cache import evict_cache_if_needed, get_cached_embeddings, store_embedding
from memweave.embedding.provider import EmbeddingProvider, LiteLLMEmbeddingProvider
from memweave.embedding.vectors import normalize_embedding
from memweave.exceptions import SearchError
from memweave.search import (
    HybridSearch,
    KeywordSearch,
    MMRReranker,
    PostProcessor,
    ScoreThreshold,
    SearchStrategy,
    TemporalDecayProcessor,
    VectorSearch,
)
from memweave.search.strategy import RawSearchRow
from memweave.storage.files import (
    build_file_entry,
    get_source_from_path,
    is_evergreen,
    list_memory_files,
    relative_path,
)
from memweave.storage.schema import ensure_schema, ensure_vector_table
from memweave.storage.sqlite_store import SQLiteStore
from memweave.types import FileInfo, IndexResult, SearchResult, StoreStatus

logger = logging.getLogger(__name__)

# Meta key for storing the provider/model fingerprint used to build the index.
# If it changes, the index is rebuilt from scratch (full re-embed).
_META_KEY = "memory_index_meta_v1"


class MemWeave:
    """Async-first memory store for AI agents.

    Wraps SQLite + sqlite-vec + FTS5 into a single high-level interface.
    Manages embedding, chunking, indexing, and search.

    Example — minimal usage::

        import asyncio
        from memweave import MemWeave, MemoryConfig

        async def main():
            config = MemoryConfig(workspace_dir="/project")
            async with MemWeave(config) as mem:
                await mem.index()
                results = await mem.search("deployment steps")
                for r in results:
                    print(r.snippet)

        asyncio.run(main())

    Example — custom embedding provider::

        class MyProvider:
            async def embed_query(self, text): ...
            async def embed_batch(self, texts): ...

        mem = MemWeave(config, embedding_provider=MyProvider())

    Attributes:
        config:             Active ``MemoryConfig``.
        embedding_provider: The ``EmbeddingProvider`` instance in use.
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Create a MemWeave instance.

        The database is NOT opened here. Call ``open()`` / ``__aenter__`` or
        any public async method (which calls ``_ensure_db()`` lazily).

        Args:
            config:             Configuration. If ``None``, uses
                                ``MemoryConfig()`` defaults.
            embedding_provider: Override the embedding provider. If ``None``,
                                ``LiteLLMEmbeddingProvider`` is used.
        """
        self.config: MemoryConfig = config or MemoryConfig()
        self.embedding_provider: EmbeddingProvider = (
            embedding_provider
            if embedding_provider is not None
            else LiteLLMEmbeddingProvider(self.config.embedding)
        )

        # Internal state — populated by _ensure_db()
        self._db: aiosqlite.Connection | None = None
        self._store: SQLiteStore | None = None
        self._vector_available: bool = False
        self._fts_available: bool = True  # FTS5 is always assumed available

        # Registered extensions
        self._strategies: dict[str, SearchStrategy] = {}
        self._postprocessors: list[PostProcessor] = []

        # Dirty flag — set when files change, cleared after index()
        self._dirty: bool = True

        # Optional watcher task (populated by start_watching())
        self._watcher_task: asyncio.Task[None] | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def _ensure_db(self) -> None:
        """Open the database and run schema migrations if not already done.

        Idempotent — subsequent calls return immediately.

        Steps on first call:
        1. Create the database directory if it doesn't exist.
        2. Open the aiosqlite connection.
        3. Run ``ensure_schema()`` (idempotent DDL).
        4. Try to load ``sqlite-vec`` and create ``chunks_vec`` virtual table.
        5. Wrap the connection in ``SQLiteStore``.
        """
        if self._db is not None:
            return

        db_path = self.config.resolved_db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(str(db_path))
        self._db.row_factory = aiosqlite.Row

        await ensure_schema(self._db)

        # Try to load sqlite-vec for ANN vector search.
        # enable_load_extension(True) must be called before load_extension()
        # — aiosqlite does not do this automatically.
        if self.config.vector.enabled:
            try:
                await self._db.enable_load_extension(True)
                ext_path = self.config.vector.extension_path
                if ext_path:
                    await self._db.load_extension(ext_path)
                else:
                    # Try loading by name — works when sqlite-vec is on PATH
                    try:
                        import sqlite_vec

                        await self._db.load_extension(sqlite_vec.loadable_path())
                    except Exception:
                        await self._db.load_extension("vec0")
                await self._db.enable_load_extension(False)  # re-lock after loading
                self._vector_available = True
            except Exception as exc:
                logger.debug("sqlite-vec not available, falling back to FTS-only: %s", exc)
                self._vector_available = False

        self._store = SQLiteStore(self._db)
        await self._startup_dirty_check()

    async def _startup_dirty_check(self) -> None:
        """Set _dirty=False if the files table matches what's currently on disk.

        Called once at the end of _ensure_db(). Compares the (path, mtime) rows
        in the files table against the workspace .md files. Sets _dirty=False only
        when the path sets are identical and no file on disk is newer than its
        stored mtime. Leaves _dirty=True on any mismatch or unexpected error.

        This lets fresh instances (e.g. every CLI invocation) reflect the true
        index state rather than always starting as dirty.
        """
        assert self._db is not None
        try:
            workspace = Path(self.config.workspace_dir)

            async with self._db.execute("SELECT path, mtime FROM files") as cursor:
                db_rows: dict[str, float] = {row["path"]: row["mtime"] async for row in cursor}

            disk_files = list_memory_files(workspace, self.config.extra_paths)
            disk_rows: dict[str, float] = {
                relative_path(f, workspace): f.stat().st_mtime for f in disk_files
            }

            if db_rows.keys() != disk_rows.keys():
                return

            for path, db_mtime in db_rows.items():
                if disk_rows[path] > db_mtime + 1e-3:
                    return

            self._dirty = False
        except Exception:
            pass  # leave _dirty=True on any unexpected error

    async def open(self) -> None:
        """Open the database connection (alias for ``_ensure_db``).

        Calling ``open()`` manually is optional — all public methods call
        ``_ensure_db()`` lazily. Use it when you want to guarantee the DB is
        ready before the first search/index call.
        """
        await self._ensure_db()
        p = self.config.progress
        mode = "hybrid" if self._vector_available else "fts-only"
        emit(p, EMOJI_OPEN, "open", f"db ready  —  search mode: {mode}")
        logger.info(
            "MemWeave opened: db=%s vector=%s",
            self.config.resolved_db_path,
            self._vector_available,
        )

    async def close(self) -> None:
        """Close the database connection and stop the file watcher.

        Safe to call multiple times. After ``close()``, the instance can be
        re-opened by calling any public method (which triggers ``_ensure_db()``).
        """
        if self._watcher_task is not None and not self._watcher_task.done():
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass
            self._watcher_task = None

        if self._db is not None:
            emit(self.config.progress, EMOJI_CLOSE, "close", "closing database")
            await self._db.close()
            self._db = None
            self._store = None
            self._vector_available = False

    async def __aenter__(self) -> MemWeave:
        """Open the DB on context manager entry."""
        await self._ensure_db()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the DB on context manager exit."""
        await self.close()

    # ── Extension points ──────────────────────────────────────────────────────

    def register_strategy(self, name: str, strategy: SearchStrategy) -> None:
        """Register a custom search strategy under a given name.

        The strategy can then be selected at search time via the ``strategy``
        keyword argument or by setting ``MemoryConfig.query.strategy``.

        Args:
            name:     Strategy identifier, e.g. ``"dense"`` or ``"bm25-only"``.
            strategy: Any object conforming to the ``SearchStrategy`` protocol.

        Example::

            class MySearch:
                async def search(self, db, query, query_vec, model, limit, *, source_filter=None):
                    ...

            mem.register_strategy("my-search", MySearch())
            results = await mem.search("query", strategy="my-search")
        """
        self._strategies[name] = strategy

    def register_postprocessor(self, processor: PostProcessor) -> None:
        """Append a post-processor to the pipeline.

        Post-processors run after the search strategy and before score
        truncation. They receive a ``list[RawSearchRow]`` and return a
        (possibly reordered/filtered) ``list[RawSearchRow]``.

        Built-in post-processors (``ScoreThreshold``, ``MMRReranker``,
        ``TemporalDecayProcessor``) are always applied automatically based on
        the active ``QueryConfig`` — this method is for additional custom
        processors.

        Args:
            processor: Any object conforming to the ``PostProcessor`` protocol.
        """
        self._postprocessors.append(processor)

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        min_score: float | None = None,
        strategy: str | None = None,
        source_filter: str | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Search the memory store for chunks relevant to ``query``.

        Runs a hybrid (vector + keyword) search by default. Applies
        ``ScoreThreshold``, ``MMRReranker`` (if enabled), and
        ``TemporalDecayProcessor`` (if enabled) as post-processors.

        If ``SyncConfig.on_search`` is True and the store is dirty (files
        changed since last index), a lightweight sync is run first.

        Args:
            query:         Search query string.
            max_results:   Override ``QueryConfig.max_results`` for this call.
                           Caps the number of results returned after all
                           post-processing.
            min_score:     Override ``QueryConfig.min_score`` for this call.
                           Results with a final score below this value are
                           dropped by ``ScoreThreshold``.
            strategy:      Override the search strategy for this call.
                           Built-in values: ``"hybrid"`` (default),
                           ``"vector"``, ``"keyword"``.
                           Pass a custom name registered via
                           :meth:`register_strategy`.
            source_filter: Restrict results to one source label, e.g.
                           ``"memory"`` (dated + evergreen files),
                           ``"sessions"`` (session transcripts), or a custom
                           subdirectory name.  ``None`` searches all sources.

        Keyword args (forwarded to post-processors via ``**kwargs``):
            mmr_lambda (float):
                Per-call lambda override for ``MMRReranker``.
                Range ``[0, 1]``: ``0`` = maximise diversity,
                ``1`` = pure relevance (no reranking).
                Only has effect when ``MMRConfig.enabled=True`` or the
                reranker is in the post-processor pipeline.
                Example: ``mem.search(q, mmr_lambda=0.5)``

            decay_half_life_days (float):
                Per-call half-life override for ``TemporalDecayProcessor``.
                Number of days until a score is halved.  Passing this
                implicitly enables decay for the call even if
                ``TemporalDecayConfig.enabled=False``.
                Example: ``mem.search(q, decay_half_life_days=14.0)``

        Returns:
            List of ``SearchResult`` objects, sorted by descending score.
            Empty list when no results match or no files are indexed.

        Raises:
            SearchError: On database or embedding failures.
        """
        await self._ensure_db()
        assert self._store is not None

        p = self.config.progress
        effective_strategy = strategy or self.config.query.strategy
        _strategy_emoji = {
            "hybrid": EMOJI_SEARCH_HYBRID,
            "vector": EMOJI_SEARCH_VEC,
            "keyword": EMOJI_SEARCH_KW,
        }.get(effective_strategy, EMOJI_SEARCH)
        emit(p, EMOJI_SEARCH, "search", f'{effective_strategy} {_strategy_emoji}  "{query}"')

        logger.info(
            "search: query=%r strategy=%s max_results=%s",
            query,
            strategy or self.config.query.strategy,
            max_results if max_results is not None else self.config.query.max_results,
        )

        # Auto-sync if dirty and on_search enabled
        if self._dirty and self.config.sync.on_search:
            try:
                await self._run_sync(force=False)
            except Exception as exc:
                logger.warning("Auto-sync before search failed: %s", exc)

        cfg = self.config.query
        effective_max = max_results if max_results is not None else cfg.max_results
        effective_min = min_score if min_score is not None else cfg.min_score

        try:
            # Embed the query (may be None if vector unavailable)
            query_vec: list[float] | None = None
            if self._vector_available:
                try:
                    query_vec = await self.embedding_provider.embed_query(query)
                except Exception as exc:
                    emit(
                        p,
                        EMOJI_WARN,
                        "search",
                        f"query embedding failed, falling back to FTS-only: {exc}",
                    )
                    logger.warning("Query embedding failed, using FTS-only: %s", exc)

            # Select and run the search strategy
            strategy_obj = self._get_strategy(effective_strategy)
            candidate_limit = effective_max * cfg.hybrid.candidate_multiplier
            raw_rows = await strategy_obj.search(
                self._db,  # type: ignore[arg-type]
                query,
                query_vec,
                self.config.embedding.model,
                candidate_limit,
                source_filter=source_filter,
            )

            # Apply post-processors
            rows = await self._apply_postprocessors(
                raw_rows,
                query,
                effective_min,
                **kwargs,
            )

            # Truncate to max_results
            rows = rows[:effective_max]

            # Convert to SearchResult
            snippet_max = cfg.snippet_max_chars
            results = [
                SearchResult(
                    path=r.path,
                    start_line=r.start_line,
                    end_line=r.end_line,
                    score=r.score,
                    snippet=truncate_snippet(r.text, snippet_max),
                    source=r.source,
                    vector_score=r.vector_score,
                    text_score=r.text_score,
                )
                for r in rows
            ]
            emit(p, EMOJI_SEARCH_DONE, "search", f"done  —  {len(results)} result(s)")
            logger.info("search: returned %d results", len(results))
            return results

        except Exception as exc:
            raise SearchError(f"Search failed for query={query!r}: {exc}") from exc

    def _get_strategy(self, name: str) -> SearchStrategy:
        """Return a search strategy by name, building built-ins on demand."""
        if name in self._strategies:
            return self._strategies[name]

        assert self._db is not None

        if name == "hybrid":
            if self._vector_available:
                return HybridSearch(
                    vector_weight=self.config.query.hybrid.vector_weight,
                    text_weight=self.config.query.hybrid.text_weight,
                )
            # Fall back to FTS-only when vector is unavailable
            return KeywordSearch()

        if name == "vector":
            if self._vector_available:
                return VectorSearch()
            return KeywordSearch()

        if name == "keyword":
            return KeywordSearch()

        raise SearchError(
            f"Unknown search strategy {name!r}. " "Register it with mem.register_strategy() first."
        )

    async def _apply_postprocessors(
        self,
        rows: list[RawSearchRow],
        query: str,
        min_score: float,
        **kwargs: Any,
    ) -> list[RawSearchRow]:
        """Apply the default post-processor pipeline plus any registered extras.

        Pipeline (in order):
        1. ``ScoreThreshold`` — filter rows below ``min_score``.
        2. ``TemporalDecayProcessor`` — if ``temporal_decay.enabled``.
        3. ``MMRReranker`` — if ``mmr.enabled``.
        4. Custom registered post-processors.
        """
        cfg = self.config.query
        p = self.config.progress

        # 1. Score threshold
        threshold = ScoreThreshold(min_score=min_score)
        rows = await threshold.apply(rows, query, **kwargs)

        # 2. Temporal decay — enabled by config OR by a per-call decay_half_life_days kwarg
        if cfg.temporal_decay.enabled or "decay_half_life_days" in kwargs:
            half_life = kwargs.get("decay_half_life_days", cfg.temporal_decay.half_life_days)
            emit(p, EMOJI_DECAY, "search", f"applying temporal decay  (half-life: {half_life}d)")
            decay = TemporalDecayProcessor(
                half_life_days=half_life,
                workspace_dir=self.config.workspace_dir,
            )
            rows = await decay.apply(rows, query, **kwargs)

        # 3. MMR reranker — enabled by config OR by a per-call mmr_lambda kwarg
        if cfg.mmr.enabled or "mmr_lambda" in kwargs:
            lam = kwargs.get("mmr_lambda", cfg.mmr.lambda_param)
            emit(p, EMOJI_MMR, "search", f"MMR reranking  (λ={lam})")
            mmr = MMRReranker(lam=lam)
            rows = await mmr.apply(rows, query, **kwargs)

        # 4. Custom processors
        for processor in self._postprocessors:
            rows = await processor.apply(rows, query, **kwargs)

        return rows

    # ── Indexing ──────────────────────────────────────────────────────────────

    async def index(self, *, force: bool = False) -> IndexResult:
        """Scan the workspace for changed files and re-index them.

        Uses SHA-256 hash comparison to detect changes — files whose content
        hash matches the stored hash are skipped.

        Steps:
        1. Discover all ``.md`` files under ``workspace_dir/memory/``.
        2. For each file, compute ``build_file_entry()`` (hash + mtime + size).
        3. Compare hash against what is stored in the ``files`` table.
        4. Re-index changed/new files: chunk → embed (cache-aware) → upsert.
        5. Delete stale index entries for files that no longer exist.
        6. Commit and clear the dirty flag.

        Args:
            force: Re-index all files regardless of hash change.

        Returns:
            ``IndexResult`` with counts of files scanned/indexed/skipped/deleted
            and embedding cache hit/miss statistics.

        Raises:
            StorageError: On database write failures.
        """
        await self._ensure_db()
        p = self.config.progress
        emit(p, EMOJI_INDEX_SCAN, "index", f"scanning workspace{' (force)' if force else ''}...")
        logger.info("index: scanning workspace %s (force=%s)", self.config.workspace_dir, force)
        result = await self._run_sync(force=force)
        emit(
            p,
            EMOJI_INDEX_DONE,
            "index",
            f"done  —  {result.files_indexed} indexed, {result.files_skipped} skipped, "
            f"{result.files_deleted} deleted  [{result.duration_ms:.0f}ms]",
        )
        logger.info(
            "index: done — %d files indexed, %d skipped, %d deleted, %d chunks",
            result.files_indexed,
            result.files_skipped,
            result.files_deleted,
            result.chunks_created,
        )
        return result

    async def add(self, path: str | Path, *, force: bool = False) -> IndexResult:
        """Index a single file immediately.

        Same as ``index()`` but operates on one file only. Useful for
        adding an in-memory-written file that isn't yet on disk or has
        just been written.

        Args:
            path:  Absolute or workspace-relative path to the ``.md`` file.
            force: Re-index even if the hash hasn't changed.

        Returns:
            ``IndexResult`` with counts for this single file.

        Raises:
            StorageError: On database write failures.
            FileNotFoundError: If ``path`` does not exist.
        """
        await self._ensure_db()
        assert self._store is not None

        workspace = Path(self.config.workspace_dir)
        abs_path = Path(path) if Path(path).is_absolute() else workspace / path
        if not abs_path.exists():
            raise FileNotFoundError(f"File not found: {abs_path}")

        p = self.config.progress
        emit(p, EMOJI_ADD, "add", f"{abs_path.name}{' (force)' if force else ''}")
        logger.info("add: indexing %s (force=%s)", abs_path, force)
        start = time.monotonic()
        result = await self._index_file(abs_path, workspace, force=force)
        await self._store.commit()
        self._dirty = False

        duration_ms = (time.monotonic() - start) * 1000
        index_result = IndexResult(
            files_scanned=1,
            files_indexed=result["indexed"],
            files_skipped=result["skipped"],
            files_deleted=0,
            chunks_created=result["chunks_created"],
            embeddings_cached=result["embeddings_cached"],
            embeddings_computed=result["embeddings_computed"],
            duration_ms=duration_ms,
        )
        emit(
            p,
            EMOJI_INDEX_DONE,
            "add",
            f"done  —  {index_result.chunks_created} chunk(s)  [{duration_ms:.0f}ms]",
        )
        logger.info("add: done — %d chunks created", index_result.chunks_created)
        return index_result

    async def _run_sync(self, *, force: bool = False) -> IndexResult:
        """Full sync: discover all files, re-index changed ones, prune stale."""
        assert self._store is not None

        p = self.config.progress
        workspace = Path(self.config.workspace_dir)
        start = time.monotonic()

        disk_files = list_memory_files(workspace, self.config.extra_paths)
        disk_paths = {relative_path(f, workspace) for f in disk_files}
        emit(p, EMOJI_INDEX_SCAN, "index", f"{len(disk_files)} file(s) found")

        # Detect provider/model changes that require full re-index
        if force or await self._provider_fingerprint_changed():
            force = True
            # Drop and recreate chunks_vec so its declared FLOAT[N] matches
            # the new model's output dimensions. Without this, inserting
            # new-dim vectors into an old-dim table fails silently.
            if self._vector_available and self._db is not None:
                try:
                    await self._db.execute("DROP TABLE IF EXISTS chunks_vec")
                    await self._db.commit()
                except Exception as exc:
                    logger.debug("Could not drop chunks_vec for rebuild: %s", exc)

        files_indexed = 0
        files_skipped = 0
        total_chunks = 0
        total_cached = 0
        total_computed = 0

        for i, abs_path in enumerate(disk_files, 1):
            rel = relative_path(abs_path, workspace)
            emit(p, EMOJI_INDEX_FILE, "index", f"{rel}  ({i}/{len(disk_files)})")
            r = await self._index_file(abs_path, workspace, force=force)
            files_indexed += r["indexed"]
            files_skipped += r["skipped"]
            total_chunks += r["chunks_created"]
            total_cached += r["embeddings_cached"]
            total_computed += r["embeddings_computed"]

        # Prune stale files (removed from disk but still in DB)
        stored_files = await self._store.list_files()
        stale = [f["path"] for f in stored_files if f["path"] not in disk_paths]
        for stale_path in stale:
            await self._store.delete_chunks_by_path(stale_path)
            await self._store.delete_fts_by_path(stale_path)
            await self._store.delete_file(stale_path)

        # Update provider fingerprint meta
        await self._save_provider_fingerprint()

        await self._store.commit()
        self._dirty = False

        duration_ms = (time.monotonic() - start) * 1000
        return IndexResult(
            files_scanned=len(disk_files),
            files_indexed=files_indexed,
            files_skipped=files_skipped,
            files_deleted=len(stale),
            chunks_created=total_chunks,
            embeddings_cached=total_cached,
            embeddings_computed=total_computed,
            duration_ms=duration_ms,
        )

    async def _index_file(
        self,
        abs_path: Path,
        workspace: Path,
        *,
        force: bool = False,
    ) -> dict[str, int]:
        """Index a single file. Returns dict with stats."""
        assert self._store is not None

        rel = relative_path(abs_path, workspace)
        source = get_source_from_path(abs_path, workspace)

        try:
            entry = build_file_entry(abs_path)
        except OSError as exc:
            logger.warning("Could not read file %s: %s", rel, exc)
            return {
                "indexed": 0,
                "skipped": 1,
                "chunks_created": 0,
                "embeddings_cached": 0,
                "embeddings_computed": 0,
            }

        new_hash = entry["hash"]

        # Check stored hash
        if not force:
            stored = await self._store.get_file(rel)
            if stored and stored["hash"] == new_hash:
                return {
                    "indexed": 0,
                    "skipped": 1,
                    "chunks_created": 0,
                    "embeddings_cached": 0,
                    "embeddings_computed": 0,
                }

        # Remove old chunks + FTS entries for this file
        await self._store.delete_chunks_by_path(rel)
        await self._store.delete_fts_by_path(rel)

        # Chunk the file
        content = abs_path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_markdown(
            content,
            chunk_tokens=self.config.chunking.tokens,
            chunk_overlap=self.config.chunking.overlap,
        )

        if not chunks:
            # Empty file — update file record with no chunks
            await self._store.upsert_file(
                path=rel,
                source=source,
                hash_=str(entry["hash"]),
                mtime=float(entry["mtime"]),
                size=int(entry["size"]),
            )
            return {
                "indexed": 1,
                "skipped": 0,
                "chunks_created": 0,
                "embeddings_cached": 0,
                "embeddings_computed": 0,
            }

        model = self.config.embedding.model
        provider_key = make_provider_key("litellm", model, self.config.embedding.api_base)

        # Build chunk IDs and text hashes
        chunk_texts = [c.text for c in chunks]
        text_hashes = [sha256_text(t) for t in chunk_texts]
        chunk_ids = [
            make_chunk_id(source, rel, c.start_line, c.end_line, th, model)
            for c, th in zip(chunks, text_hashes)
        ]

        # Bulk cache lookup
        cached_embeds: dict[str, list[float]] = {}
        embeddings_cached = 0
        embeddings_computed = 0

        if self.config.cache.enabled:
            cached_embeds = await get_cached_embeddings(
                self._store, text_hashes, model, provider_key
            )
            embeddings_cached = len(cached_embeds)

        # Embed cache misses
        miss_indices = [i for i, th in enumerate(text_hashes) if th not in cached_embeds]
        miss_texts = [chunk_texts[i] for i in miss_indices]

        p = self.config.progress
        if embeddings_cached:
            emit(p, EMOJI_EMBED_CACHE, "index", f"{embeddings_cached} chunk(s) from cache  [{rel}]")

        new_vecs: list[list[float]] = []
        if miss_texts:
            emit(
                p,
                EMOJI_EMBED_API,
                "index",
                f"embedding {len(miss_texts)} new chunk(s) via API  [{rel}]",
            )
            try:
                new_vecs = await self.embedding_provider.embed_batch(miss_texts)
                embeddings_computed = len(new_vecs)
            except Exception as exc:
                # Batch failed (e.g. one chunk exceeds model context length).
                # Fall back to per-chunk embedding so only the bad chunks lose
                # their vectors instead of the entire file's batch.
                emit(
                    p,
                    EMOJI_WARN,
                    "index",
                    f"batch embedding failed for {rel}, retrying per-chunk: {exc}",
                )
                logger.warning(
                    "Batch embedding failed for %s: %s. Retrying each chunk individually.", rel, exc
                )
                new_vecs = []
                for i, text in enumerate(miss_texts):
                    vec = await self._embed_with_halving(text, rel)
                    if vec:
                        new_vecs.append(vec)
                        embeddings_computed += 1
                    else:
                        new_vecs.append([])

        # Store new embeddings in cache
        if self.config.cache.enabled:
            for idx, vec in zip(miss_indices, new_vecs):
                if vec:
                    await store_embedding(
                        self._store,
                        text_hashes[idx],
                        vec,
                        model=model,
                        provider_key=provider_key,
                    )

        # Merge cached + new embeddings in original order
        miss_vec_map = dict(zip(miss_indices, new_vecs))
        all_vecs: list[list[float] | None] = []
        for i, th in enumerate(text_hashes):
            if th in cached_embeds:
                all_vecs.append(cached_embeds[th])
            elif i in miss_vec_map:
                v = miss_vec_map[i]
                all_vecs.append(v)
            else:
                all_vecs.append(None)

        # Upsert chunks + FTS
        chunk_vec: list[float] | None
        for chunk, chunk_id, text_hash, chunk_vec in zip(chunks, chunk_ids, text_hashes, all_vecs):
            await self._store.upsert_chunk(
                id_=chunk_id,
                path=rel,
                source=source,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                hash_=text_hash,
                model=model,
                text=chunk.text,
                embedding=chunk_vec,
            )
            await self._store.upsert_fts(
                text=chunk.text,
                chunk_id=chunk_id,
                path=rel,
                source=source,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                model=model,
            )

        # Upsert into chunks_vec if vector available.
        # Create the table on first use — we need the actual embedding dims
        # from the first real vector (not a config guess).
        if self._vector_available and self._db is not None:
            first_vec = next((v for v in all_vecs if v), None)
            if first_vec:
                await ensure_vector_table(self._db, len(first_vec))
            for chunk_id, chunk_vec in zip(chunk_ids, all_vecs):
                if chunk_vec:
                    await self._upsert_vec(chunk_id, chunk_vec)

        # Update file record
        await self._store.upsert_file(
            path=rel,
            source=source,
            hash_=str(entry["hash"]),
            mtime=float(entry["mtime"]),
            size=int(entry["size"]),
        )

        # Prune embedding cache if cap exceeded
        if self.config.cache.enabled and self.config.cache.max_entries is not None:
            await evict_cache_if_needed(self._store, model, self.config.cache.max_entries)

        return {
            "indexed": 1,
            "skipped": 0,
            "chunks_created": len(chunks),
            "embeddings_cached": embeddings_cached,
            "embeddings_computed": embeddings_computed,
        }

    async def _embed_with_halving(self, text: str, rel: str) -> list[float]:
        """Embed text, recursively halving if the model rejects it as too long.

        Splits at the nearest whitespace to the midpoint so sub-chunks are
        word-aligned. When both halves succeed their vectors are averaged and
        L2-normalised to produce a single representative vector for the chunk.
        Returns [] only if the text is too short to split further.
        """
        try:
            return await self.embedding_provider.embed_query(text)
        except Exception:
            if len(text) < 100:
                logger.warning(
                    "Chunk (chars=%d) in %s is too short to split further — storing without vector.",
                    len(text),
                    rel,
                )
                return []

            # Split near the midpoint at a word boundary
            mid = len(text) // 2
            split = text.rfind(" ", mid // 2, mid + mid // 2)
            if split == -1:
                split = mid

            left_vec = await self._embed_with_halving(text[:split].strip(), rel)
            right_vec = await self._embed_with_halving(text[split:].strip(), rel)

            if left_vec and right_vec:
                avg = [(lv + rv) / 2.0 for lv, rv in zip(left_vec, right_vec)]
                return normalize_embedding(avg)
            return left_vec or right_vec or []

    async def _upsert_vec(self, chunk_id: str, vec: list[float]) -> None:
        """Insert or replace a vector into chunks_vec (sqlite-vec virtual table)."""
        assert self._db is not None
        import struct

        vec_bytes = struct.pack(f"{len(vec)}f", *vec)
        try:
            await self._db.execute(
                "INSERT OR REPLACE INTO chunks_vec (id, embedding) VALUES (?, ?)",
                [chunk_id, vec_bytes],
            )
        except Exception as exc:
            logger.warning("Failed to upsert vec for %s: %s", chunk_id, exc)

    # ── Provider fingerprint ──────────────────────────────────────────────────

    async def _provider_fingerprint_changed(self) -> bool:
        """Return True if the stored provider/model fingerprint differs from config.

        Return True if the stored provider/model fingerprint differs from config.
        If the model or provider key changed, the entire index must be rebuilt.
        """
        assert self._store is not None
        stored = await self._store.get_meta(_META_KEY)
        if not stored:
            return False  # No stored meta → first run, not a forced re-index

        try:
            meta = json.loads(stored)
        except (json.JSONDecodeError, TypeError):
            return True  # Corrupt meta → rebuild

        cfg = self.config
        current_key = make_provider_key(
            "litellm",
            cfg.embedding.model,
            cfg.embedding.api_base,
        )

        return bool(
            meta.get("model") != cfg.embedding.model
            or meta.get("provider_key") != current_key
            or meta.get("chunk_tokens") != cfg.chunking.tokens
            or meta.get("chunk_overlap") != cfg.chunking.overlap
        )

    async def _save_provider_fingerprint(self) -> None:
        """Persist the current provider/model fingerprint to the meta table."""
        assert self._store is not None
        cfg = self.config
        provider_key = make_provider_key(
            "litellm",
            cfg.embedding.model,
            cfg.embedding.api_base,
        )
        meta = {
            "model": cfg.embedding.model,
            "provider_key": provider_key,
            "chunk_tokens": cfg.chunking.tokens,
            "chunk_overlap": cfg.chunking.overlap,
        }
        await self._store.set_meta(_META_KEY, json.dumps(meta))

    # ── Status & Files ────────────────────────────────────────────────────────

    async def status(self) -> StoreStatus:
        """Return a snapshot of the current store state.

        Provides counts of indexed files and chunks, vector/FTS availability,
        cache stats, and watcher status.

        Returns:
            ``StoreStatus`` snapshot.
        """
        await self._ensure_db()
        assert self._store is not None

        file_count = len(await self._store.list_files())
        chunk_count = await self._store.count_chunks()
        cache_entries = await self._store.count_cache_entries()

        if self._vector_available and self._fts_available:
            search_mode = "hybrid"
        elif self._vector_available:
            search_mode = "vector-only"
        elif self._fts_available:
            search_mode = "fts-only"
        else:
            search_mode = "unavailable"

        p = self.config.progress
        emit(
            p,
            EMOJI_STATUS,
            "status",
            f"{file_count} file(s), {chunk_count} chunk(s), "
            f"search: {search_mode}, cache: {cache_entries} entries",
        )

        return StoreStatus(
            files=file_count,
            chunks=chunk_count,
            dirty=self._dirty,
            workspace_dir=str(self.config.workspace_dir),
            db_path=str(self.config.resolved_db_path),
            search_mode=search_mode,
            provider="litellm",
            model=self.config.embedding.model,
            fts_available=self._fts_available,
            vector_available=self._vector_available,
            cache_entries=cache_entries,
            cache_max_entries=self.config.cache.max_entries,
            watcher_active=(self._watcher_task is not None and not self._watcher_task.done()),
        )

    async def files(self) -> list[FileInfo]:
        """Return metadata for all tracked memory files.

        Returns:
            List of ``FileInfo`` objects (one per tracked file).
        """
        await self._ensure_db()
        assert self._store is not None

        workspace = Path(self.config.workspace_dir)
        stored_files = await self._store.list_files()
        result: list[FileInfo] = []

        for f in stored_files:
            path_str = f["path"]
            # Count chunks for this file
            chunks = await self._store.get_chunks_by_path(path_str)
            chunk_count = len(chunks)

            # Determine evergreen status from the absolute path
            abs_path = workspace / path_str
            evergreen = is_evergreen(abs_path, self.config.evergreen_patterns)

            result.append(
                FileInfo(
                    path=path_str,
                    size=f["size"],
                    hash=f["hash"],
                    mtime=f["mtime"],
                    chunks=chunk_count,
                    is_evergreen=evergreen,
                    source=f["source"],
                )
            )

        emit(self.config.progress, EMOJI_FILES, "files", f"{len(result)} tracked file(s)")
        return result

    # ── Watcher ───────────────────────────────────────────────────────────────

    async def start_watching(self) -> None:
        """Start a background file watcher that re-indexes on changes.

        Requires ``watchfiles`` to be installed. If not installed, logs a
        warning and returns without starting the watcher.

        The watcher is debounced: rapid successive changes (within
        ``SyncConfig.watch_debounce_ms``) are coalesced into a single sync.

        To stop the watcher, call ``close()`` or cancel the task directly.
        """
        if self._watcher_task is not None and not self._watcher_task.done():
            return  # Already watching

        await self._ensure_db()

        try:
            from memweave.sync.watcher import MemoryWatcher
        except ImportError:
            logger.warning(
                "watchfiles is not installed — file watcher unavailable. "
                "Install it with: pip install memweave[watch]"
            )
            return

        watcher = MemoryWatcher(
            workspace_dir=Path(self.config.workspace_dir),
            on_change=self._on_watch_change,
            debounce_ms=self.config.sync.watch_debounce_ms,
        )
        self._watcher_task = asyncio.create_task(watcher.run(), name="memweave-watcher")

    async def _on_watch_change(self, changed_paths: set[Path]) -> None:
        """Called by the watcher when .md files change."""
        self._dirty = True
        logger.debug("File watcher detected changes: %s", changed_paths)
        try:
            await self._run_sync(force=False)
        except Exception as exc:
            logger.warning("Background sync after file change failed: %s", exc)

    # ── Flush ─────────────────────────────────────────────────────────────────

    async def flush(
        self,
        conversation: list[dict[str, str]],
        *,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> str | None:
        """Extract durable facts from a conversation and persist them.

        Calls an LLM (via LiteLLM) with the conversation history and a
        structured extraction system prompt. The extracted facts are appended
        to the dated memory file (``memory/YYYY-MM-DD.md``), then re-indexed.

        Args:
            conversation:  List of ``{"role": ..., "content": ...}`` messages.
            model:         Override ``FlushConfig.model`` for this call.
            system_prompt: Override ``FlushConfig.system_prompt`` for this call.

        Returns:
            The LLM's extracted text, or ``None`` if the LLM replied with
            ``@@SILENT_REPLY@@`` (nothing worth persisting).

        Raises:
            FlushError: On LLM API failure or file write error.
        """
        await self._ensure_db()

        if not self.config.flush.enabled:
            return None

        p = self.config.progress
        emit(p, EMOJI_FLUSH_EXTRACT, "flush", "extracting memories from conversation...")

        from memweave.flush.memory_flush import flush_conversation

        result = await flush_conversation(
            conversation=conversation,
            config=self.config,
            model=model,
            system_prompt=system_prompt,
        )

        if result is not None:
            # Re-index the updated memory file
            self._dirty = True
            workspace = Path(self.config.workspace_dir)
            memory_dir = workspace / "memory"
            import zoneinfo
            from datetime import date

            try:
                zoneinfo.ZoneInfo(self.config.timezone)
            except Exception:
                pass
            today = date.today()
            dated_file = memory_dir / f"{today.isoformat()}.md"
            emit(p, EMOJI_FLUSH_WRITE, "flush", f"writing to  {dated_file.relative_to(workspace)}")
            if dated_file.exists():
                try:
                    await self.add(dated_file, force=True)
                except Exception as exc:
                    emit(p, EMOJI_WARN, "flush", f"re-index after flush failed: {exc}")
                    logger.warning("Re-index after flush failed: %s", exc)
            emit(p, EMOJI_FLUSH_DONE, "flush", f"done  —  {len(result)} chars written")
        else:
            emit(p, EMOJI_FLUSH_DONE, "flush", "nothing to store  (@@SILENT_REPLY@@)")

        return result
