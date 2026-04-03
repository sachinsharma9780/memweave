"""
memweave/exceptions.py — Custom exception hierarchy.

All MemWeave errors inherit from ``MemWeaveError``, allowing callers to catch
broadly (``except MemWeaveError``) or narrowly (``except EmbeddingError``)
depending on what they need to handle.

Hierarchy::

    MemWeaveError
    ├── ConfigError          — invalid/missing configuration at startup
    ├── StorageError         — SQLite or file I/O failure
    ├── IndexError           — chunking, embedding, or indexing failure
    │   └── EmbeddingError   — embedding API failure (provider/rate-limit/auth)
    ├── SearchError          — search pipeline failure
    │   └── StrategyError    — unknown or failed search strategy name
    └── FlushError           — LLM memory flush failure

Usage pattern for callers::

    try:
        results = await mem.search("deployment steps")
    except EmbeddingError as exc:
        # Embedding API is down — fall back to keyword-only or cached results
        logger.warning("Embedding unavailable: %s", exc)
    except SearchError as exc:
        # Any other search failure
        logger.error("Search failed: %s", exc)
"""

from __future__ import annotations


class MemWeaveError(Exception):
    """Base exception for all MemWeave errors.

    Catch this to handle any error from the library without importing
    every specific subclass.
    """


class ConfigError(MemWeaveError):
    """Raised for invalid or missing configuration.

    Examples:
    - ``workspace_dir`` does not exist and cannot be created.
    - Conflicting config values (e.g. ``hybrid`` weights don't sum to 1.0).
    - Unknown search strategy name passed to ``QueryConfig.strategy``.
    """


class StorageError(MemWeaveError):
    """Raised for SQLite or file I/O failures.

    Wraps exceptions from ``aiosqlite`` and ``aiofiles``. The error message
    always includes the offending key/path to aid debugging.

    Examples:
    - SQLite database is locked by another process.
    - Disk is full when writing to the database.
    - File permission denied when reading a markdown file.
    """


class IndexError(MemWeaveError):
    """Raised when chunking, embedding, or indexing fails.

    This is the parent class for all errors that occur during ``index()``
    or ``add()``. Catch ``IndexError`` to handle any indexing failure broadly,
    or catch ``EmbeddingError`` specifically for API-related issues.
    """


class SearchError(MemWeaveError):
    """Raised when the search pipeline fails.

    Examples:
    - The FTS5 or vector index is corrupted.
    - A post-processor raises an unexpected exception.
    - The database connection was closed before search completed.
    """


class StrategyError(SearchError):
    """Raised when a search strategy name is unknown or its execution fails.

    Examples:
    - ``strategy="graph"`` is requested but no graph strategy is registered.
    - A custom ``SearchStrategy`` plugin raises during ``search()``.
    """


class EmbeddingError(IndexError):
    """Raised for embedding API failures.

    Examples:
    - LiteLLM API call fails with an HTTP error (rate limit, auth, timeout).
    - The embedding model returns vectors of unexpected dimensionality.
    - The API returns an empty response for a non-empty input batch.
    """


class FlushError(MemWeaveError):
    """Raised when the LLM memory flush operation fails.

    ``flush()`` is the only MemWeave operation that calls an LLM. This error
    wraps failures from the LLM API call or from writing the extracted facts
    to disk.

    Examples:
    - LiteLLM ``acompletion()`` returns an error response.
    - The target daily memory file cannot be written (permissions, disk full).
    """
