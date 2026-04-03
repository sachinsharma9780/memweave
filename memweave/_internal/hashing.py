"""
memweave/_internal/hashing.py — Shared hashing and concurrency utilities.

Internal module — not part of the public API. All helpers here are pure
functions (deterministic, no I/O side effects) except sha256_file and
run_with_concurrency.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file on disk.

    Reads the file in 8 KB chunks to avoid loading large files fully into
    memory. This is the primary mechanism for change detection — if the hash
    of a file matches what is stored in SQLite, indexing is skipped.

    Args:
        path: Absolute path to the file to hash.

    Returns:
        64-character lowercase hex string, e.g.
        ``"a3f5c9..."``

    Example::

        hash_ = sha256_file(Path("/project/memory/2026-03-21.md"))
        # "a3f5c9e1b2d4..."
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Compute the SHA-256 hex digest of a UTF-8 string.

    Used for:
    - Embedding cache keys (hash of chunk text before API call)
    - Deriving chunk IDs via ``make_chunk_id``
    - Provider config keys via ``make_provider_key``

    Args:
        text: UTF-8 string to hash.

    Returns:
        64-character lowercase hex string.

    Example::

        key = sha256_text("The meeting notes from yesterday...")
        # "7e3b9a..."
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute the SHA-256 hex digest of raw bytes.

    Used when the data is already in binary form (e.g. serialized embeddings).

    Args:
        data: Raw bytes to hash.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(data).hexdigest()


def make_chunk_id(
    source: str,
    path: str,
    start_line: int,
    end_line: int,
    content_hash: str,
    model: str,
) -> str:
    """Generate a deterministic, stable chunk ID from its metadata.

    The ID is a SHA-256 of a colon-delimited composite key. Because it only
    depends on the chunk's coordinates and content hash (not wall time), the
    same chunk always produces the same ID across indexing runs. This enables
    SQLite upserts (``INSERT OR REPLACE``) without duplication.

    If the file content changes (new ``content_hash``), the chunk ID changes,
    so old chunks are replaced and new ones are inserted cleanly.

    Args:
        source:       Logical source label, e.g. ``"memory"`` or ``"sessions"``.
        path:         Relative file path, e.g. ``"memory/2026-03-21.md"``.
        start_line:   1-indexed first line of the chunk within the file.
        end_line:     1-indexed last line of the chunk within the file.
        content_hash: SHA-256 of the chunk's text content.
        model:        Embedding model name, e.g. ``"text-embedding-3-small"``.

    Returns:
        64-character lowercase hex string.

    Example::

        chunk_id = make_chunk_id(
            source="memory",
            path="memory/2026-03-21.md",
            start_line=10,
            end_line=25,
            content_hash="a3f5c9...",
            model="text-embedding-3-small",
        )
        # Stable across runs: same inputs → same ID
    """
    key = f"{source}:{path}:{start_line}:{end_line}:{content_hash}:{model}"
    return sha256_text(key)


def make_provider_key(
    provider: str,
    model: str,
    api_base: str | None,
) -> str:
    """Generate a provider config fingerprint for the embedding cache.

    The embedding cache is keyed by (provider, model, provider_key, text_hash).
    ``provider_key`` ensures that two different configurations of the same model
    are treated as separate cache namespaces and their vectors never collide.

    Args:
        provider:   Provider identifier, e.g. ``"litellm"``.
        model:      Model name, e.g. ``"text-embedding-3-small"``.
        api_base:   Custom API base URL, or ``None`` for the default endpoint.

    Returns:
        64-character lowercase hex string.

    Example::

        # Default endpoint
        key1 = make_provider_key("litellm", "text-embedding-3-small", None)

        # Local Ollama endpoint — different key, separate cache namespace
        key2 = make_provider_key("litellm", "nomic-embed-text", "http://localhost:11434")

        assert key1 != key2
    """
    key = f"{provider}:{model}:{api_base or ''}"
    return sha256_text(key)


async def run_with_concurrency(
    tasks: list[Callable[[], Awaitable[T]]],
    max_concurrent: int = 8,
) -> list[T]:
    """Run a list of async callables with a bounded concurrency limit.

    Uses ``asyncio.Semaphore`` to cap the number of tasks executing at the
    same time. This prevents flooding the embedding API with too many
    simultaneous requests while still benefiting from async I/O parallelism.

    Result ordering is preserved: ``results[i]`` corresponds to ``tasks[i]``.
    On the first exception, all remaining tasks are cancelled and the
    exception is re-raised (fail-fast semantics via ``asyncio.gather``).

    Args:
        tasks:          List of zero-argument async callables, e.g.
                        ``[lambda: embed(chunk) for chunk in batch]``.
        max_concurrent: Maximum number of tasks allowed to run simultaneously.
                        Default is 8, which works well for most embedding APIs.

    Returns:
        List of results in input order.

    Raises:
        Exception: First exception raised by any task (others are cancelled).

    Example::

        async def embed_one(text: str) -> list[float]:
            return await litellm.aembedding(model="...", input=text)

        # Embed 100 chunks, max 8 at a time
        results = await run_with_concurrency(
            [lambda t=chunk: embed_one(t) for chunk in chunks],
            max_concurrent=8,
        )
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run(task: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await task()

    return list(await asyncio.gather(*[_run(t) for t in tasks]))


def normalize_path(path: Path | str, base: Path) -> str:
    """Return a path as a forward-slash string relative to ``base``.

    Used to produce consistent, portable relative paths for storage in SQLite
    regardless of the OS path separator. If ``path`` is not under ``base``,
    the absolute POSIX path is returned as a fallback.

    Args:
        path: Absolute or relative path to normalize.
        base: Base directory to make the path relative to.

    Returns:
        Forward-slash relative path string, or absolute POSIX path if ``path``
        is not under ``base``.

    Examples::

        normalize_path("/project/memory/2026-03-21.md", Path("/project"))
        # → "memory/2026-03-21.md"

        normalize_path("memory/2026-03-21.md", Path("/project"))
        # → "memory/2026-03-21.md"   (already relative, returned as-is)

        normalize_path("/other/file.md", Path("/project"))
        # → "/other/file.md"         (fallback: not under base)
    """
    p = Path(path)
    if p.is_absolute():
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            return p.as_posix()
    return p.as_posix()


def truncate_snippet(text: str, max_chars: int) -> str:
    """Truncate text to at most ``max_chars`` characters, at a word boundary.

    If the text fits within the limit, it is returned unchanged. Otherwise,
    the text is cut at the last whitespace before the limit (to avoid cutting
    mid-word) and an ellipsis character ``…`` is appended. If no whitespace
    is found in the second half of the allowed range, the cut is made at
    exactly ``max_chars``.

    Args:
        text:      Input text to truncate.
        max_chars: Maximum number of characters in the output (including ``…``).

    Returns:
        Original text if short enough, otherwise truncated text ending in ``…``.

    Examples::

        truncate_snippet("Hello world", max_chars=20)
        # → "Hello world"   (no truncation needed)

        truncate_snippet("The quick brown fox jumps over the lazy dog", max_chars=20)
        # → "The quick brown fox…"
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to break at last whitespace to avoid cutting mid-word
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated + "…"


def batched(items: list[Any], size: int) -> list[list[Any]]:
    """Split a flat list into sub-lists of at most ``size`` items each.

    Used to chunk large lists of texts into batches for embedding API calls
    (respecting the provider's ``batch_size`` limit).

    Args:
        items: Any list to split.
        size:  Maximum items per batch. Must be >= 1.

    Returns:
        List of lists. The last batch may be smaller than ``size``.

    Example::

        batched([1, 2, 3, 4, 5], size=2)
        # → [[1, 2], [3, 4], [5]]

        batched([], size=10)
        # → []
    """
    return [items[i : i + size] for i in range(0, len(items), size)]
