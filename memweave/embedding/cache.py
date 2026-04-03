"""
memweave/embedding/cache.py — Embedding cache operations.

The embedding cache prevents redundant API calls. Every time a chunk is
indexed, its text is hashed (SHA-256) and checked against the cache. If the
same text was embedded before with the same model configuration, the stored
vector is reused — no API call is made.

This module is a thin wrapper around ``SQLiteStore``'s embedding-cache
methods. It adds:
- Bulk cache lookup (one SQL query for a whole batch).
- Separation of a batch into hits/misses.
- LRU eviction helper (delegates to ``SQLiteStore.prune_cache``).

Cache key design:
    The cache primary key is ``(provider, model, provider_key, text_hash)``.

    - ``provider``     — human-readable provider name (e.g. ``"litellm"``)
    - ``model``        — model name (e.g. ``"text-embedding-3-small"``)
    - ``provider_key`` — SHA-256 fingerprint of the full model config
                         (model + api_base), from ``make_provider_key()``.
                         Prevents two configs of the same model from sharing
                         cache entries.
    - ``text_hash``    — SHA-256 of the chunk text.

Example flow during indexing::

    # 1. Collect text hashes for all chunks in this file
    hashes = [sha256_text(chunk.text) for chunk in chunks]

    # 2. Bulk lookup: which hashes are already cached?
    hits = await get_cached_embeddings(store, hashes, provider, model, provider_key)
    # hits = {"a3f5c9...": [0.1, 0.2, ...], ...}  — only cached entries

    # 3. Identify misses
    misses = [c for c in chunks if sha256_text(c.text) not in hits]

    # 4. Embed only the missing chunks
    new_vecs = await provider.embed_batch([c.text for c in misses])

    # 5. Store new embeddings in cache
    for chunk, vec in zip(misses, new_vecs):
        await store_embedding(store, sha256_text(chunk.text), vec, ...)
"""

from __future__ import annotations

import logging

from memweave._internal.hashing import sha256_text
from memweave.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

# Fixed provider name stored alongside each cache entry
_PROVIDER_NAME = "litellm"


async def get_cached_embeddings(
    store: SQLiteStore,
    text_hashes: list[str],
    model: str,
    provider_key: str,
) -> dict[str, list[float]]:
    """Bulk-lookup cached embeddings for a list of text hashes.

    Issues a single SQL query for all requested hashes. Cache misses are
    simply absent from the returned dict — callers use set-difference to
    identify which texts need fresh embedding API calls.

    Args:
        store:        Open ``SQLiteStore`` instance.
        text_hashes:  List of SHA-256 hashes of chunk texts to look up.
        model:        Embedding model name (e.g. ``"text-embedding-3-small"``).
        provider_key: SHA-256 fingerprint of the model config, from
                      ``make_provider_key()``.

    Returns:
        Dict mapping ``{text_hash: embedding_vector}`` for all cache hits.
        Cache misses are absent. Empty dict if ``text_hashes`` is empty.

    Example::

        hits = await get_cached_embeddings(
            store,
            hashes=["a3f5c9...", "b7e2d1...", "c1a4f8..."],
            model="text-embedding-3-small",
            provider_key=make_provider_key("litellm", "text-embedding-3-small", None),
        )
        # → {"a3f5c9...": [0.1, 0.2, ...]}   (only "a3f5c9..." was cached)
    """
    if not text_hashes:
        return {}
    return await store.get_embeddings_bulk(
        provider=_PROVIDER_NAME,
        model=model,
        provider_key=provider_key,
        hashes=text_hashes,
    )


async def get_cached_embedding(
    store: SQLiteStore,
    text: str,
    model: str,
    provider_key: str,
) -> list[float] | None:
    """Look up the cached embedding for a single text string.

    Hashes the text internally — callers do not need to compute the hash.

    Args:
        store:        Open ``SQLiteStore`` instance.
        text:         Raw chunk text whose embedding to look up.
        model:        Embedding model name.
        provider_key: Provider config fingerprint.

    Returns:
        Embedding vector if cached, or ``None`` on a miss.

    Example::

        vec = await get_cached_embedding(
            store, "We chose PostgreSQL.", model, provider_key
        )
        if vec is None:
            vec = await provider.embed_query("We chose PostgreSQL.")
    """
    text_hash = sha256_text(text)
    return await store.get_embedding(
        provider=_PROVIDER_NAME,
        model=model,
        provider_key=provider_key,
        hash_=text_hash,
    )


async def store_embedding(
    store: SQLiteStore,
    text_hash: str,
    embedding: list[float],
    model: str,
    provider_key: str,
) -> None:
    """Persist a single embedding vector to the cache.

    Called after a fresh embedding API call, before committing the
    transaction. The ``text_hash`` key must be the SHA-256 of the
    corresponding chunk text.

    Args:
        store:        Open ``SQLiteStore`` instance.
        text_hash:    SHA-256 of the chunk text (cache lookup key).
        embedding:    L2-normalized embedding vector to store.
        model:        Embedding model name.
        provider_key: Provider config fingerprint.

    Example::

        text_hash = sha256_text(chunk.text)
        await store_embedding(store, text_hash, vec, model, provider_key)
        # Must call store.commit() afterwards to persist.
    """
    dims = len(embedding)
    await store.upsert_embedding(
        provider=_PROVIDER_NAME,
        model=model,
        provider_key=provider_key,
        hash_=text_hash,
        embedding=embedding,
        dims=dims,
    )


async def store_embeddings_bulk(
    store: SQLiteStore,
    text_hash_to_embedding: dict[str, list[float]],
    model: str,
    provider_key: str,
) -> int:
    """Persist multiple embeddings to the cache in one pass.

    More efficient than calling ``store_embedding`` in a loop because all
    upserts happen within the same open transaction (committed by the caller
    after this function returns).

    Args:
        store:                   Open ``SQLiteStore`` instance.
        text_hash_to_embedding:  Dict mapping ``{text_hash: vector}``.
        model:                   Embedding model name.
        provider_key:            Provider config fingerprint.

    Returns:
        Number of embeddings stored.

    Example::

        stored = await store_embeddings_bulk(
            store,
            {sha256_text(t): v for t, v in zip(texts, vecs)},
            model="text-embedding-3-small",
            provider_key=key,
        )
        await store.commit()
    """
    count = 0
    for text_hash, embedding in text_hash_to_embedding.items():
        dims = len(embedding)
        await store.upsert_embedding(
            provider=_PROVIDER_NAME,
            model=model,
            provider_key=provider_key,
            hash_=text_hash,
            embedding=embedding,
            dims=dims,
        )
        count += 1
    return count


async def evict_cache_if_needed(
    store: SQLiteStore,
    model: str,
    max_entries: int | None,
) -> int:
    """Run LRU eviction if ``max_entries`` is set and the cache exceeds the cap.

    Does nothing if ``max_entries`` is ``None`` (unlimited cache).

    Steps:
    1. Skip immediately if ``max_entries`` is ``None``.
    2. Delegate to ``store.prune_cache(provider, model, max_entries)``.
    3. Log the eviction count if > 0.

    Args:
        store:       Open ``SQLiteStore`` instance.
        model:       Embedding model name (eviction is scoped per model).
        max_entries: Maximum cache entries to keep. ``None`` = no eviction.

    Returns:
        Number of cache entries deleted (0 if no eviction needed or disabled).
    """
    if max_entries is None:
        return 0
    deleted = await store.prune_cache(
        provider=_PROVIDER_NAME,
        model=model,
        max_entries=max_entries,
    )
    if deleted:
        logger.debug("LRU eviction: removed %d cache entries (cap=%d)", deleted, max_entries)
    return deleted


def split_into_hits_and_misses(
    texts: list[str],
    cached: dict[str, list[float]],
) -> tuple[dict[int, list[float]], list[tuple[int, str]]]:
    """Partition a list of texts into cache hits and misses.

    Given a list of texts and a ``{hash: vector}`` dict from a bulk cache
    lookup, returns:
    - ``hits``: ``{original_index: vector}`` for texts whose hash was cached.
    - ``misses``: ``[(original_index, text)]`` for texts that need embedding.

    The ``original_index`` is the position of the text in the input list,
    which lets callers reconstruct the result in the original order after
    embedding the misses.

    Args:
        texts:  Full list of chunk texts to process.
        cached: Cache lookup result from ``get_cached_embeddings()``.

    Returns:
        Tuple of:
        - ``hits``:   ``{index: vector}`` for cached texts.
        - ``misses``: ``[(index, text)]`` for uncached texts.

    Example::

        texts = ["PostgreSQL chosen.", "Redis for caching.", "Nginx as proxy."]
        cached = {sha256_text("Redis for caching."): [0.1, 0.2, ...]}
        hits, misses = split_into_hits_and_misses(texts, cached)
        # hits   = {1: [0.1, 0.2, ...]}
        # misses = [(0, "PostgreSQL chosen."), (2, "Nginx as proxy.")]
    """
    from memweave._internal.hashing import sha256_text as _hash

    hits: dict[int, list[float]] = {}
    misses: list[tuple[int, str]] = []

    for i, text in enumerate(texts):
        h = _hash(text)
        if h in cached:
            hits[i] = cached[h]
        else:
            misses.append((i, text))

    return hits, misses


def merge_embeddings(
    hits: dict[int, list[float]],
    misses: list[tuple[int, str]],
    new_embeddings: list[list[float]],
) -> list[list[float]]:
    """Reconstruct the full ordered list of embeddings from hits + new embeds.

    After embedding the misses via the provider, this function merges the
    cached hits and the fresh embeddings back into input order.

    Args:
        hits:           ``{original_index: vector}`` from cache.
        misses:         ``[(original_index, text)]`` for uncached texts.
        new_embeddings: Vectors returned by the provider for each miss,
                        in the same order as ``misses``.

    Returns:
        Full list of embedding vectors in the original input order.

    Raises:
        ValueError: If ``len(new_embeddings) != len(misses)``.

    Example::

        # 3 input texts: index 1 was a cache hit, 0 and 2 were misses
        hits = {1: [0.1, 0.2]}
        misses = [(0, "PostgreSQL chosen."), (2, "Nginx as proxy.")]
        new_vecs = [[0.3, 0.4], [0.5, 0.6]]
        result = merge_embeddings(hits, misses, new_vecs)
        # result[0] = [0.3, 0.4]  ← embedded fresh
        # result[1] = [0.1, 0.2]  ← from cache
        # result[2] = [0.5, 0.6]  ← embedded fresh
    """
    if len(new_embeddings) != len(misses):
        raise ValueError(
            f"new_embeddings length ({len(new_embeddings)}) " f"!= misses length ({len(misses)})"
        )

    total = len(hits) + len(misses)
    result: list[list[float] | None] = [None] * total

    for idx, vec in hits.items():
        result[idx] = vec

    for (idx, _text), vec in zip(misses, new_embeddings):
        result[idx] = vec

    # All slots should be filled; raise if any are None (indicates logic error)
    out: list[list[float]] = []
    for i, v in enumerate(result):
        if v is None:
            raise ValueError(f"No embedding at index {i} after merge")
        out.append(v)
    return out
