"""
memweave/search/hybrid.py — Weighted merge of vector + keyword results.

Weighted merge of vector + keyword results.

The hybrid strategy runs both backends and combines their scores::

    combined_score = vector_weight * vector_score + text_weight * text_score

Missing scores (a chunk found only by one backend) are treated as 0 for
the missing component.
"""

from __future__ import annotations

import aiosqlite

from memweave.search.keyword import KeywordSearch
from memweave.search.strategy import RawSearchRow
from memweave.search.vector import VectorSearch


def merge_hybrid_results(
    vector_rows: list[RawSearchRow],
    keyword_rows: list[RawSearchRow],
    *,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    limit: int | None = None,
) -> list[RawSearchRow]:
    """Merge vector and keyword results into a single ranked list.

    Merge vector and keyword results into a single ranked list.

    Algorithm:
    1. Build a dict keyed by ``chunk_id`` from all vector results.
    2. Walk keyword results:
       - If the chunk already exists (found by both backends), add its
         ``text_score`` and recompute the combined score.
       - If it only exists in keyword results, create a new entry with
         ``vector_score = 0``.
    3. Compute ``combined_score = vector_weight * vs + text_weight * ts``
       for every entry.
    4. Sort by combined score descending.
    5. Truncate to ``limit`` if given.

    Snippet preference: when a chunk appears in both backends, the keyword
    snippet is preferred (FTS5 highlighted snippets tend to be more relevant
    to the query terms — FTS5 highlighted snippets tend to be more relevant.

    Args:
        vector_rows:   Results from :class:`~memweave.search.vector.VectorSearch`.
        keyword_rows:  Results from :class:`~memweave.search.keyword.KeywordSearch`.
        vector_weight: Weight applied to vector scores (default 0.7).
        text_weight:   Weight applied to keyword scores (default 0.3).
        limit:         If set, truncate output to this many rows.

    Returns:
        Merged, sorted list of :class:`~memweave.search.strategy.RawSearchRow`.
        Each row has both ``vector_score`` and ``text_score`` populated (0.0
        when the backend did not return that chunk).

    Examples::

        vec_rows = [RawSearchRow("id1", ..., score=0.9, vector_score=0.9)]
        kw_rows  = [RawSearchRow("id1", ..., score=0.6, text_score=0.6),
                    RawSearchRow("id2", ..., score=0.8, text_score=0.8)]
        merged = merge_hybrid_results(vec_rows, kw_rows)
        # id1 combined = 0.7*0.9 + 0.3*0.6 = 0.81
        # id2 combined = 0.7*0.0 + 0.3*0.8 = 0.24
        # → [id1 (0.81), id2 (0.24)]
    """
    # Keyed by chunk_id; value: (row, vector_score, text_score)
    merged: dict[str, tuple[RawSearchRow, float, float]] = {}

    for row in vector_rows:
        vs = row.vector_score if row.vector_score is not None else row.score
        merged[row.chunk_id] = (row, vs, 0.0)

    for row in keyword_rows:
        ts = row.text_score if row.text_score is not None else row.score
        if row.chunk_id in merged:
            existing_row, vs, _ = merged[row.chunk_id]
            # Prefer keyword snippet (matches query terms more directly)
            preferred_row = row if row.text else existing_row
            merged[row.chunk_id] = (preferred_row, vs, ts)
        else:
            merged[row.chunk_id] = (row, 0.0, ts)

    result: list[RawSearchRow] = []
    for row, vs, ts in merged.values():
        combined = vector_weight * vs + text_weight * ts
        result.append(
            RawSearchRow(
                chunk_id=row.chunk_id,
                path=row.path,
                source=row.source,
                start_line=row.start_line,
                end_line=row.end_line,
                text=row.text,
                score=combined,
                vector_score=vs,
                text_score=ts,
            )
        )

    result.sort(key=lambda r: r.score, reverse=True)
    if limit is not None:
        result = result[:limit]
    return result


class HybridSearch:
    """Combined vector + keyword search backend.

    Runs :class:`~memweave.search.vector.VectorSearch` and
    :class:`~memweave.search.keyword.KeywordSearch` in sequence and merges
    their results via :func:`merge_hybrid_results`.

    If vector search fails (e.g. sqlite-vec not loaded), the error propagates
    — there is no silent fallback.  Install ``memweave[vector]`` and ensure
    the extension is loaded before using this backend.

    Configuration::

        hs = HybridSearch(vector_weight=0.7, text_weight=0.3)
        rows = await hs.search(db, "PostgreSQL connection", query_vec, model, 10)

    The ``limit`` passed to ``search()`` is used as the *per-backend* fetch
    limit (``limit * candidate_multiplier``) before merging.  The final
    result list is then truncated to ``limit``.

    Attributes:
        vector_weight:        Weight for vector scores (default 0.7).
        text_weight:          Weight for keyword scores (default 0.3).
        candidate_multiplier: Internal pool multiplier (default 4).
                              Each backend fetches ``limit * multiplier``
                              candidates before the merge.
    """

    def __init__(
        self,
        *,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        candidate_multiplier: int = 4,
    ) -> None:
        self.vector_weight = vector_weight
        self.text_weight = text_weight
        self.candidate_multiplier = candidate_multiplier
        self._vector = VectorSearch()
        self._keyword = KeywordSearch()

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
        """Run hybrid search: vector + keyword, then merge.

        Steps:
        1. Fetch ``limit * candidate_multiplier`` rows from vector search.
        2. Fetch ``limit * candidate_multiplier`` rows from keyword search.
        3. Merge via :func:`merge_hybrid_results` with configured weights.
        4. Truncate to ``limit``.

        Args:
            db:            Open aiosqlite connection with sqlite-vec loaded.
            query:         Raw user query (used by keyword backend).
            query_vec:     L2-normalized query embedding (used by vector backend).
            model:         Embedding model name filter.
            limit:         Maximum rows in final result.
            source_filter: Optional source restriction.

        Returns:
            Merged list of :class:`~memweave.search.strategy.RawSearchRow`,
            at most ``limit`` entries, ordered by combined score descending.
        """
        pool = limit * self.candidate_multiplier

        vector_rows = await self._vector.search(
            db,
            query,
            query_vec,
            model,
            pool,
            source_filter=source_filter,
        )
        keyword_rows = await self._keyword.search(
            db,
            query,
            None,
            model,
            pool,
            source_filter=source_filter,
        )

        return merge_hybrid_results(
            vector_rows,
            keyword_rows,
            vector_weight=self.vector_weight,
            text_weight=self.text_weight,
            limit=limit,
        )
