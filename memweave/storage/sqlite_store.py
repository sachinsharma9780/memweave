"""
memweave/storage/sqlite_store.py — Parameterized CRUD operations on all tables.

``SQLiteStore`` is the only module that issues SQL statements. All higher-level
code (indexer, search strategies, sync) goes through this class.

Design principles:
- **Async throughout**: every method is ``async``, uses ``aiosqlite`` internally.
- **Parameterized queries**: all user-supplied values are passed as ``?``
  placeholders — no string formatting into SQL, no SQL injection surface.
- **Errors wrapped**: every ``Exception`` is caught and re-raised as
  ``StorageError`` with a human-readable message that includes the offending
  key/path. This lets callers catch ``StorageError`` without importing
  ``aiosqlite``.
- **No lifecycle ownership**: ``SQLiteStore`` accepts an already-open
  ``aiosqlite.Connection`` and never calls ``.close()`` on it. The caller
  (typically ``MemWeave._open()``) owns the connection lifecycle.
- **Explicit commits**: write methods do NOT auto-commit. The caller batches
  multiple writes and calls ``store.commit()`` once at the end for performance.

Table coverage:
    meta            — ``set_meta``, ``get_meta``, ``get_all_meta``
    files           — ``upsert_file``, ``get_file``, ``delete_file``, ``list_files``
    chunks          — ``upsert_chunk``, ``get_chunk``, ``get_chunks_by_path``,
                      ``delete_chunks_by_path``, ``count_chunks``
    chunks_fts      — ``upsert_fts``, ``delete_fts_by_path``
    embedding_cache — ``upsert_embedding``, ``get_embedding``,
                      ``get_embeddings_bulk``, ``count_cache_entries``,
                      ``prune_cache``, ``clear_cache``
    transactions    — ``commit``, ``rollback``
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import aiosqlite

from memweave.exceptions import StorageError

logger = logging.getLogger(__name__)


class SQLiteStore:
    """Async CRUD abstraction over the MemWeave SQLite database.

    Accepts an open ``aiosqlite.Connection`` at construction time.
    Does NOT own the connection lifecycle — the caller opens and closes it.

    Typical usage inside the library::

        async with aiosqlite.connect(config.resolved_db_path) as db:
            store = SQLiteStore(db)
            await store.upsert_file(path="memory/2026-03-21.md", ...)
            await store.upsert_chunk(id_=chunk_id, ...)
            await store.commit()

    All write operations must be followed by ``await store.commit()`` (or
    ``await store.rollback()`` on failure) to persist changes.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        """Wrap an open aiosqlite connection.

        Args:
            db: An open ``aiosqlite.Connection``. The caller is responsible
                for closing it when done.
        """
        self._db = db

    # ── Meta table ───────────────────────────────────────────────────────────

    async def set_meta(self, key: str, value: str) -> None:
        """Insert or replace a key-value pair in the ``meta`` table.

        Used to store internal settings such as ``schema_version``,
        ``last_sync_at``, and provider fingerprints. Values are always strings;
        serialize numbers/dicts before calling this method.

        Args:
            key:   Unique string key, e.g. ``"last_sync_at"``.
            value: String value, e.g. ``"1742000000"``.

        Raises:
            StorageError: On database write failure.

        Example::

            await store.set_meta("last_sync_at", str(int(time.time())))
        """
        try:
            await self._db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [key, value],
            )
        except Exception as exc:
            raise StorageError(f"Failed to set meta key={key!r}: {exc}") from exc

    async def get_meta(self, key: str) -> str | None:
        """Return a value from the ``meta`` table, or ``None`` if not found.

        Args:
            key: Key to look up.

        Returns:
            String value, or ``None`` on a cache miss.

        Raises:
            StorageError: On database read failure.

        Example::

            ts = await store.get_meta("last_sync_at")
            if ts:
                last_sync = int(ts)
        """
        try:
            cursor = await self._db.execute("SELECT value FROM meta WHERE key = ?", [key])
            row = await cursor.fetchone()
            return row[0] if row else None
        except Exception as exc:
            raise StorageError(f"Failed to get meta key={key!r}: {exc}") from exc

    async def get_all_meta(self) -> dict[str, str]:
        """Return all key-value pairs from the ``meta`` table as a dict.

        Useful for serializing the full internal state for diagnostics or
        when migrating databases between schema versions.

        Returns:
            Dict mapping every key to its string value.

        Raises:
            StorageError: On database read failure.
        """
        try:
            cursor = await self._db.execute("SELECT key, value FROM meta")
            rows = await cursor.fetchall()
            return {row[0]: row[1] for row in rows}
        except Exception as exc:
            raise StorageError(f"Failed to get all meta: {exc}") from exc

    # ── Files table ──────────────────────────────────────────────────────────

    async def upsert_file(
        self,
        path: str,
        source: str,
        hash_: str,
        mtime: float,
        size: int,
    ) -> None:
        """Insert or replace a file record used for hash-based change detection.

        After indexing a file, the indexer calls this method to record the
        file's current content hash. On the next indexing run, if the stored
        hash matches the file on disk, the file is skipped.

        Args:
            path:   Relative file path, e.g. ``"memory/2026-03-21.md"``.
            source: Logical source label, e.g. ``"memory"`` or ``"sessions"``.
            hash_:  SHA-256 hex digest of the file's content.
            mtime:  File modification time (Unix timestamp as float).
            size:   File size in bytes.

        Raises:
            StorageError: On database write failure.

        Example::

            await store.upsert_file(
                path="memory/2026-03-21.md",
                source="memory",
                hash_="a3f5c9...",
                mtime=1742000000.0,
                size=4096,
            )
            await store.commit()
        """
        try:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO files (path, source, hash, mtime, size)
                VALUES (?, ?, ?, ?, ?)
                """,
                [path, source, hash_, mtime, size],
            )
        except Exception as exc:
            raise StorageError(f"Failed to upsert file path={path!r}: {exc}") from exc

    async def get_file(self, path: str) -> dict[str, Any] | None:
        """Return the stored file record as a dict, or ``None`` if not found.

        Args:
            path: Relative file path to look up.

        Returns:
            Dict with keys ``path``, ``source``, ``hash``, ``mtime``, ``size``,
            or ``None`` if the file is not tracked.

        Raises:
            StorageError: On database read failure.
        """
        try:
            cursor = await self._db.execute(
                "SELECT path, source, hash, mtime, size FROM files WHERE path = ?",
                [path],
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return {
                "path": row[0],
                "source": row[1],
                "hash": row[2],
                "mtime": row[3],
                "size": row[4],
            }
        except Exception as exc:
            raise StorageError(f"Failed to get file path={path!r}: {exc}") from exc

    async def delete_file(self, path: str) -> None:
        """Remove a file record from the ``files`` table.

        Called when a tracked file has been deleted from disk. Does NOT
        cascade to chunks — call ``delete_chunks_by_path`` and
        ``delete_fts_by_path`` separately before deleting the file record.

        Args:
            path: Relative file path to remove.

        Raises:
            StorageError: On database write failure.
        """
        try:
            await self._db.execute("DELETE FROM files WHERE path = ?", [path])
        except Exception as exc:
            raise StorageError(f"Failed to delete file path={path!r}: {exc}") from exc

    async def list_files(self, source: str | None = None) -> list[dict[str, Any]]:
        """Return all tracked file records, optionally filtered by source.

        Args:
            source: If provided, only return files with this source label
                    (e.g. ``"memory"`` or ``"sessions"``). ``None`` returns all.

        Returns:
            List of dicts, each with keys ``path``, ``source``, ``hash``,
            ``mtime``, ``size``.

        Raises:
            StorageError: On database read failure.

        Example::

            # List only session files
            session_files = await store.list_files(source="sessions")
        """
        try:
            if source is not None:
                cursor = await self._db.execute(
                    "SELECT path, source, hash, mtime, size FROM files WHERE source = ?",
                    [source],
                )
            else:
                cursor = await self._db.execute("SELECT path, source, hash, mtime, size FROM files")
            rows = await cursor.fetchall()
            return [
                {"path": r[0], "source": r[1], "hash": r[2], "mtime": r[3], "size": r[4]}
                for r in rows
            ]
        except Exception as exc:
            raise StorageError(f"Failed to list files: {exc}") from exc

    # ── Chunks table ─────────────────────────────────────────────────────────

    async def upsert_chunk(
        self,
        id_: str,
        path: str,
        source: str,
        start_line: int,
        end_line: int,
        hash_: str,
        model: str,
        text: str,
        embedding: list[float] | None,
    ) -> None:
        """Insert or replace a text chunk in the ``chunks`` table.

        The chunk ``id_`` is a deterministic SHA-256 of its metadata (see
        ``make_chunk_id``). If the same chunk is re-indexed (same file, same
        lines, same content hash), the existing row is replaced — no
        duplicates accumulate.

        Embeddings are stored as JSON strings to keep the schema simple and
        avoid binary blob handling. For large-scale deployments with millions
        of chunks, this may be replaced by ``chunks_vec`` (vector table).

        Args:
            id_:        Chunk ID from ``make_chunk_id()``.
            path:       Relative file path, e.g. ``"memory/2026-03-21.md"``.
            source:     Source label, e.g. ``"memory"``.
            start_line: 1-indexed first line of this chunk within the file.
            end_line:   1-indexed last line of this chunk within the file.
            hash_:      SHA-256 of the chunk's text content.
            model:      Embedding model used, e.g. ``"text-embedding-3-small"``.
            text:       Raw chunk text.
            embedding:  Embedding vector as a list of floats, or ``None`` if
                        the embedding has not been computed yet.

        Raises:
            StorageError: On database write failure.
        """
        embedding_json = json.dumps(embedding) if embedding is not None else None
        try:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO chunks
                    (id, path, source, start_line, end_line, hash, model, text, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    id_,
                    path,
                    source,
                    start_line,
                    end_line,
                    hash_,
                    model,
                    text,
                    embedding_json,
                    int(time.time()),
                ],
            )
        except Exception as exc:
            raise StorageError(f"Failed to upsert chunk id={id_!r}: {exc}") from exc

    async def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Return a single chunk record as a dict, or ``None`` if not found.

        Args:
            chunk_id: Chunk ID to look up (from ``make_chunk_id()``).

        Returns:
            Dict with keys: ``id``, ``path``, ``source``, ``start_line``,
            ``end_line``, ``hash``, ``model``, ``text``, ``embedding``
            (list of floats or ``None``), ``updated_at`` (Unix timestamp).

        Raises:
            StorageError: On database read failure.
        """
        try:
            cursor = await self._db.execute(
                """
                SELECT id, path, source, start_line, end_line, hash, model, text, embedding, updated_at
                FROM chunks WHERE id = ?
                """,
                [chunk_id],
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "path": row[1],
                "source": row[2],
                "start_line": row[3],
                "end_line": row[4],
                "hash": row[5],
                "model": row[6],
                "text": row[7],
                "embedding": json.loads(row[8]) if row[8] else None,
                "updated_at": row[9],
            }
        except Exception as exc:
            raise StorageError(f"Failed to get chunk id={chunk_id!r}: {exc}") from exc

    async def get_chunks_by_path(self, path: str) -> list[dict[str, Any]]:
        """Return all chunks for a given file, ordered by line number.

        Used during re-indexing to compare existing chunks against newly
        chunked content, and for building context windows in search results.

        Args:
            path: Relative file path.

        Returns:
            List of chunk dicts (same structure as ``get_chunk``), sorted by
            ``start_line`` ascending.

        Raises:
            StorageError: On database read failure.
        """
        try:
            cursor = await self._db.execute(
                """
                SELECT id, path, source, start_line, end_line, hash, model, text, embedding, updated_at
                FROM chunks WHERE path = ?
                ORDER BY start_line
                """,
                [path],
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "path": r[1],
                    "source": r[2],
                    "start_line": r[3],
                    "end_line": r[4],
                    "hash": r[5],
                    "model": r[6],
                    "text": r[7],
                    "embedding": json.loads(r[8]) if r[8] else None,
                    "updated_at": r[9],
                }
                for r in rows
            ]
        except Exception as exc:
            raise StorageError(f"Failed to get chunks for path={path!r}: {exc}") from exc

    async def delete_chunks_by_path(self, path: str) -> int:
        """Delete all chunks associated with a file.

        Called before re-indexing a changed file (old chunks are deleted,
        new chunks are inserted) and when a file is removed from disk.

        Note: This only removes rows from the ``chunks`` table. Call
        ``delete_fts_by_path`` separately to clean up the FTS index.

        Args:
            path: Relative file path whose chunks should be removed.

        Returns:
            Number of rows deleted (0 if the file had no indexed chunks).

        Raises:
            StorageError: On database write failure.
        """
        try:
            cursor = await self._db.execute("DELETE FROM chunks WHERE path = ?", [path])
            return cursor.rowcount or 0
        except Exception as exc:
            raise StorageError(f"Failed to delete chunks for path={path!r}: {exc}") from exc

    async def count_chunks(self) -> int:
        """Return the total number of indexed chunks across all files.

        Used by ``MemWeave.status()`` to populate ``StoreStatus.chunks``.

        Returns:
            Integer count of rows in the ``chunks`` table.

        Raises:
            StorageError: On database read failure.
        """
        try:
            cursor = await self._db.execute("SELECT COUNT(*) FROM chunks")
            row = await cursor.fetchone()
            return row[0] if row else 0
        except Exception as exc:
            raise StorageError(f"Failed to count chunks: {exc}") from exc

    # ── FTS table ────────────────────────────────────────────────────────────

    async def upsert_fts(
        self,
        text: str,
        chunk_id: str,
        path: str,
        source: str,
        start_line: int,
        end_line: int,
        model: str = "",
    ) -> None:
        """Insert a chunk into the FTS5 keyword search index.

        FTS5 virtual tables do not support ``ON CONFLICT DO UPDATE`` or
        ``INSERT OR REPLACE``. To simulate an upsert, this method:
        1. Deletes the existing FTS row (if any) for ``chunk_id``.
        2. Inserts a fresh row with the new text.

        This delete-then-insert pattern is safe because both operations run
        within the same transaction (committed by the caller).

        Args:
            text:       Full chunk text to index for keyword search.
            chunk_id:   Chunk ID linking this FTS row to the ``chunks`` table.
            path:       Relative file path (stored UNINDEXED for retrieval).
            source:     Source label (stored UNINDEXED).
            start_line: First line of the chunk (stored UNINDEXED).
            end_line:   Last line of the chunk (stored UNINDEXED).
            model:      Embedding model name (stored UNINDEXED, used for filtering).

        Raises:
            StorageError: On database write failure.
        """
        try:
            # Step 1: Delete existing FTS entry (FTS5 has no native upsert)
            await self._db.execute("DELETE FROM chunks_fts WHERE id = ?", [chunk_id])
            # Step 2: Insert fresh row with latest text
            await self._db.execute(
                """
                INSERT INTO chunks_fts (text, id, path, source, model, start_line, end_line)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [text, chunk_id, path, source, model, start_line, end_line],
            )
        except Exception as exc:
            raise StorageError(f"Failed to upsert FTS entry id={chunk_id!r}: {exc}") from exc

    async def delete_fts_by_path(self, path: str) -> None:
        """Delete all FTS5 index entries for a given file.

        Must be called alongside ``delete_chunks_by_path`` when removing or
        re-indexing a file. If this step is skipped, stale keyword search
        results pointing to non-existent chunks will appear.

        Args:
            path: Relative file path whose FTS entries should be removed.

        Raises:
            StorageError: On database write failure.
        """
        try:
            await self._db.execute("DELETE FROM chunks_fts WHERE path = ?", [path])
        except Exception as exc:
            raise StorageError(f"Failed to delete FTS entries for path={path!r}: {exc}") from exc

    # ── Embedding cache ──────────────────────────────────────────────────────

    async def upsert_embedding(
        self,
        provider: str,
        model: str,
        provider_key: str,
        hash_: str,
        embedding: list[float],
        dims: int,
    ) -> None:
        """Store a computed embedding vector in the cache.

        The cache key is ``(provider, model, provider_key, hash_)``. On the
        next indexing run, if the same chunk text (same hash) is encountered
        with the same model config, the stored vector is returned without
        calling the embedding API.

        Args:
            provider:     Provider name, e.g. ``"litellm"``.
            model:        Model name, e.g. ``"text-embedding-3-small"``.
            provider_key: SHA-256 fingerprint of the model config (from
                          ``make_provider_key()``).
            hash_:        SHA-256 of the chunk text (cache lookup key).
            embedding:    Embedding vector as a list of floats.
            dims:         Vector dimensionality (e.g. 1536). Stored for
                          validation on cache retrieval.

        Raises:
            StorageError: On database write failure.
        """
        embedding_json = json.dumps(embedding)
        try:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                    (provider, model, provider_key, hash, embedding, dims, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [provider, model, provider_key, hash_, embedding_json, dims, int(time.time())],
            )
        except Exception as exc:
            raise StorageError(f"Failed to upsert embedding hash={hash_!r}: {exc}") from exc

    async def get_embedding(
        self,
        provider: str,
        model: str,
        provider_key: str,
        hash_: str,
    ) -> list[float] | None:
        """Return a cached embedding vector, or ``None`` on a cache miss.

        Args:
            provider:     Provider name.
            model:        Model name.
            provider_key: Provider config fingerprint (from ``make_provider_key()``).
            hash_:        SHA-256 of the chunk text to look up.

        Returns:
            List of floats (the embedding vector), or ``None`` if not cached.

        Raises:
            StorageError: On database read failure.
        """
        try:
            cursor = await self._db.execute(
                """
                SELECT embedding FROM embedding_cache
                WHERE provider = ? AND model = ? AND provider_key = ? AND hash = ?
                """,
                [provider, model, provider_key, hash_],
            )
            row = await cursor.fetchone()
            return json.loads(row[0]) if row else None
        except Exception as exc:
            raise StorageError(f"Failed to get embedding hash={hash_!r}: {exc}") from exc

    async def get_embeddings_bulk(
        self,
        provider: str,
        model: str,
        provider_key: str,
        hashes: list[str],
    ) -> dict[str, list[float]]:
        """Batch-lookup cached embeddings for multiple chunk hashes.

        Much more efficient than calling ``get_embedding`` in a loop because
        it issues a single SQL query with an ``IN (...)`` clause instead of
        N separate queries.

        Steps:
        1. Build a parameterized ``IN (?, ?, ...)`` clause from ``hashes``.
        2. Execute one SELECT against ``embedding_cache``.
        3. Build and return a dict of ``{hash: vector}`` for all hits.
           Cache misses are simply absent from the returned dict.

        Args:
            provider:     Provider name.
            model:        Model name.
            provider_key: Provider config fingerprint.
            hashes:       List of chunk text SHA-256 hashes to look up.

        Returns:
            Dict mapping ``hash`` → embedding vector for all cache hits.
            If ``hashes`` is empty, returns ``{}``.

        Raises:
            StorageError: On database read failure.

        Example::

            cached = await store.get_embeddings_bulk(
                provider="litellm",
                model="text-embedding-3-small",
                provider_key=key,
                hashes=[sha256_text(chunk.text) for chunk in chunks],
            )
            # cached = {"a3f5c9...": [0.1, 0.2, ...], ...}
            missing = [c for c in chunks if sha256_text(c.text) not in cached]
            # Only embed the missing chunks
        """
        if not hashes:
            return {}
        placeholders = ",".join("?" * len(hashes))
        try:
            cursor = await self._db.execute(
                f"""
                SELECT hash, embedding FROM embedding_cache
                WHERE provider = ? AND model = ? AND provider_key = ?
                AND hash IN ({placeholders})
                """,
                [provider, model, provider_key, *hashes],
            )
            rows = await cursor.fetchall()
            return {row[0]: json.loads(row[1]) for row in rows}
        except Exception as exc:
            raise StorageError(f"Failed to bulk-get embeddings: {exc}") from exc

    async def count_cache_entries(self) -> int:
        """Return the total number of cached embedding vectors.

        Used by ``MemWeave.status()`` to populate ``StoreStatus.cache_entries``.

        Returns:
            Integer row count of the ``embedding_cache`` table.

        Raises:
            StorageError: On database read failure.
        """
        try:
            cursor = await self._db.execute("SELECT COUNT(*) FROM embedding_cache")
            row = await cursor.fetchone()
            return row[0] if row else 0
        except Exception as exc:
            raise StorageError(f"Failed to count cache: {exc}") from exc

    async def prune_cache(self, provider: str, model: str, max_entries: int) -> int:
        """LRU eviction: delete the oldest embeddings beyond ``max_entries``.

        When ``CacheConfig.max_entries`` is set, this method is called after
        each batch of embeddings is stored. It removes the rows with the
        smallest ``updated_at`` timestamps (least recently used) until the
        count for this ``(provider, model)`` pair is within the cap.

        Steps:
        1. Count current entries for this provider+model.
        2. If within cap, return 0 (nothing to do).
        3. Compute ``to_delete = current - max_entries``.
        4. Delete the ``to_delete`` oldest rows (ORDER BY updated_at ASC LIMIT ?).

        Note: Eviction is scoped to ``(provider, model)`` — different models
        share the same cache table but are evicted independently.

        Args:
            provider:    Provider name.
            model:       Model name.
            max_entries: Maximum entries to keep for this provider+model.

        Returns:
            Number of rows deleted (0 if no eviction was needed).

        Raises:
            StorageError: On database write failure.

        Example::

            # After storing new embeddings, prune to keep at most 10,000 entries
            deleted = await store.prune_cache("litellm", "text-embedding-3-small", 10000)
            if deleted:
                logger.info("Evicted %d old embeddings from cache", deleted)
        """
        try:
            # Step 1: Count current entries for this provider+model
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM embedding_cache WHERE provider = ? AND model = ?",
                [provider, model],
            )
            row = await cursor.fetchone()
            current = row[0] if row else 0

            # Step 2: Nothing to do if within cap
            if current <= max_entries:
                return 0

            # Step 3: Calculate how many to delete
            to_delete = current - max_entries

            # Step 4: Delete the oldest rows (LRU by updated_at)
            await self._db.execute(
                """
                DELETE FROM embedding_cache
                WHERE (provider, model, provider_key, hash) IN (
                    SELECT provider, model, provider_key, hash
                    FROM embedding_cache
                    WHERE provider = ? AND model = ?
                    ORDER BY updated_at ASC
                    LIMIT ?
                )
                """,
                [provider, model, to_delete],
            )
            return to_delete
        except Exception as exc:
            raise StorageError(f"Failed to prune cache: {exc}") from exc

    async def clear_cache(self) -> int:
        """Delete all embedding cache entries.

        Useful for forcing a full re-embed on the next indexing run (e.g. after
        changing embedding models). Returns the count of deleted rows for
        logging purposes.

        Returns:
            Number of rows deleted.

        Raises:
            StorageError: On database write/read failure.
        """
        try:
            cursor = await self._db.execute("SELECT COUNT(*) FROM embedding_cache")
            row = await cursor.fetchone()
            count = row[0] if row else 0
            await self._db.execute("DELETE FROM embedding_cache")
            return count
        except Exception as exc:
            raise StorageError(f"Failed to clear cache: {exc}") from exc

    # ── Transactions ─────────────────────────────────────────────────────────

    async def commit(self) -> None:
        """Commit the current transaction to disk.

        All write operations (``upsert_*``, ``delete_*``) are buffered in an
        implicit transaction until ``commit()`` is called. Call this after
        batching multiple writes to maximize write performance (fewer fsync
        calls).

        Raises:
            StorageError: On commit failure (e.g. disk full, locked database).
        """
        try:
            await self._db.commit()
        except Exception as exc:
            raise StorageError(f"Failed to commit: {exc}") from exc

    async def rollback(self) -> None:
        """Roll back the current transaction, discarding all buffered writes.

        Called in error handlers to undo a partially-completed batch. After
        rollback, the database is left in the state it was before the
        transaction started.

        Raises:
            StorageError: On rollback failure (rare; connection may be closed).
        """
        try:
            await self._db.rollback()
        except Exception as exc:
            raise StorageError(f"Failed to rollback: {exc}") from exc
