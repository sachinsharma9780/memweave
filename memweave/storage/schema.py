"""
memweave/storage/schema.py — SQLite schema creation and versioning.

Design principles:
- **Idempotent**: all statements use ``CREATE TABLE/INDEX IF NOT EXISTS``,
  so calling ``ensure_schema()`` multiple times is safe.
- **Derived index**: deleting ``index.sqlite`` and calling ``ensure_schema()``
  rebuilds the full structure from scratch; markdown files are the source of truth.
- **Deferred vector table**: ``chunks_vec`` requires the ``sqlite-vec`` extension
  to be loaded first. It is NOT created here — call ``ensure_vector_table()``
  separately after loading the extension.

Tables created by ``ensure_schema()``:

    meta              — key/value store for internal settings (schema version, etc.)
    files             — one row per tracked .md file, used for hash-based change detection
    chunks            — indexed text chunks with optional embedding vectors (JSON)
    chunks_fts        — FTS5 virtual table for keyword (BM25) search
    embedding_cache   — deduplication store: avoids re-calling the embedding API
                        for unchanged chunk text

Table created separately by ``ensure_vector_table()``:

    chunks_vec        — sqlite-vec virtual table for ANN vector search
"""

from __future__ import annotations

import logging

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# ── Table creation SQL ──────────────────────────────────────────────────────

_CREATE_META = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CREATE_FILES = """
CREATE TABLE IF NOT EXISTS files (
    path   TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'memory',
    hash   TEXT NOT NULL,
    mtime  REAL NOT NULL,
    size   INTEGER NOT NULL
);
"""

_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id         TEXT PRIMARY KEY,
    path       TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'memory',
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    hash       TEXT NOT NULL,
    model      TEXT NOT NULL,
    text       TEXT NOT NULL,
    embedding  TEXT,
    updated_at INTEGER NOT NULL
);
"""

# chunks_fts is a virtual FTS5 table — ``text`` is the only FTS-indexed column.
# All other columns are UNINDEXED: stored alongside each row but excluded from
# the full-text index, keeping the FTS index lean while still letting callers
# retrieve chunk metadata without a JOIN.
_CREATE_CHUNKS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    id         UNINDEXED,
    path       UNINDEXED,
    source     UNINDEXED,
    model      UNINDEXED,
    start_line UNINDEXED,
    end_line   UNINDEXED
);
"""

# embedding_cache deduplicates API calls across indexing runs.
# Primary key is (provider, model, provider_key, hash):
#   - provider/model:   human-readable for debugging and cache pruning
#   - provider_key:     SHA-256 fingerprint of the full model config (model +
#                       api_base), so different endpoints never share cache entries
#   - hash:             SHA-256 of the chunk's text content
# updated_at is used for LRU eviction in prune_cache().
_CREATE_EMBEDDING_CACHE = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    provider     TEXT NOT NULL,
    model        TEXT NOT NULL,
    provider_key TEXT NOT NULL,
    hash         TEXT NOT NULL,
    embedding    TEXT NOT NULL,
    dims         INTEGER,
    updated_at   INTEGER NOT NULL,
    PRIMARY KEY (provider, model, provider_key, hash)
);
"""

# ── Indices ──────────────────────────────────────────────────────────────────

_INDICES = [
    # LRU eviction: scan embedding_cache in insertion order
    "CREATE INDEX IF NOT EXISTS idx_embedding_cache_updated_at ON embedding_cache(updated_at);",
    # Chunk lookups by file path (used during file re-indexing and deletion)
    "CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);",
    # Source-filtered chunk queries (e.g. search only "memory" or "sessions")
    "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);",
    # Hash-based deduplication during indexing
    "CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash);",
]

# chunks_vec is created separately (requires sqlite-vec extension loaded first).
# Template is parameterized by ``dims`` because sqlite-vec requires the vector
# dimensionality to be declared at table creation time (e.g. FLOAT[1536]).
_CREATE_CHUNKS_VEC_TEMPLATE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    id        TEXT PRIMARY KEY,
    embedding FLOAT[{dims}]
);
"""


async def ensure_schema(db: aiosqlite.Connection) -> None:
    """Create all base tables and indices if they do not already exist.

    This function is **idempotent** — safe to call on every startup. It uses
    ``CREATE TABLE/INDEX IF NOT EXISTS`` throughout, so existing data is never
    touched.

    Steps performed:
        1. Execute a single ``executescript`` with all ``CREATE TABLE`` and
           ``CREATE INDEX`` statements (atomic w.r.t. schema changes).
        2. Insert the ``schema_version`` key into ``meta`` if not present.
        3. Commit the transaction.

    Does NOT create ``chunks_vec`` — that requires the ``sqlite-vec`` extension
    to be loaded first. Call ``ensure_vector_table()`` separately.

    Args:
        db: Open ``aiosqlite.Connection``. Must not be in WAL mode already
            (``executescript`` issues an implicit ``COMMIT`` first).

    Example::

        async with aiosqlite.connect(":memory:") as db:
            await ensure_schema(db)
            version = await get_schema_version(db)
            assert version == 1
    """
    async with db.executescript(
        "\n".join(
            [
                _CREATE_META,
                _CREATE_FILES,
                _CREATE_CHUNKS,
                _CREATE_CHUNKS_FTS,
                _CREATE_EMBEDDING_CACHE,
                *_INDICES,
            ]
        )
    ):
        pass

    await _ensure_schema_version(db)
    await db.commit()
    logger.debug("Schema ensured (version %d)", SCHEMA_VERSION)


async def ensure_vector_table(db: aiosqlite.Connection, dims: int) -> bool:
    """Create the ``chunks_vec`` virtual table for ANN vector search.

    This table uses the ``sqlite-vec`` extension (``vec0`` module). It must be
    called **after** the extension has been loaded into the connection via
    ``db.enable_load_extension(True)`` and ``db.load_extension(...)``.

    The ``dims`` argument must match the dimensionality of the embedding model
    in use (e.g. 1536 for ``text-embedding-3-small`` at full size, 256 for the
    truncated variant). Mismatching dimensions will cause insert errors later.

    Args:
        db:   Open ``aiosqlite.Connection`` with ``sqlite-vec`` already loaded.
        dims: Vector dimensionality. Must match the embedding model's output size.

    Returns:
        ``True`` if the table was created or already exists.
        ``False`` if creation failed (e.g. sqlite-vec not loaded, wrong dims).

    Example::

        await db.enable_load_extension(True)
        await db.load_extension("/path/to/vec0.so")
        ok = await ensure_vector_table(db, dims=1536)
        if not ok:
            # Fall back to FTS-only search
            ...
    """
    sql = _CREATE_CHUNKS_VEC_TEMPLATE.format(dims=dims)
    try:
        await db.execute(sql)
        await db.commit()
        logger.debug("Vector table ensured (dims=%d)", dims)
        return True
    except Exception as exc:
        logger.warning("Could not create vector table: %s", exc)
        return False


async def get_schema_version(db: aiosqlite.Connection) -> int:
    """Return the stored schema version from the ``meta`` table.

    Used during startup to detect whether a migration is needed. Returns ``0``
    if the table doesn't exist yet or the key is missing (fresh database).

    Args:
        db: Open ``aiosqlite.Connection``.

    Returns:
        Integer schema version (currently ``1``), or ``0`` if not set.
    """
    try:
        cursor = await db.execute("SELECT value FROM meta WHERE key = 'schema_version'")
        row = await cursor.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


async def _ensure_schema_version(db: aiosqlite.Connection) -> None:
    """Insert ``schema_version`` into ``meta`` if not already present.

    Uses ``INSERT OR IGNORE`` so existing versions are never overwritten.
    This is called internally by ``ensure_schema()`` after table creation.

    Args:
        db: Open ``aiosqlite.Connection``.
    """
    await db.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
        [str(SCHEMA_VERSION)],
    )
