"""
memweave/search/strategy.py — SearchStrategy protocol and shared result types.

Defines the duck-typed interface that all search backends (vector, keyword,
hybrid) conform to, plus the data types that flow between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import aiosqlite


@dataclass(frozen=True, slots=True)
class RawSearchRow:
    """A single row returned from a search backend before score merging.

    Both VectorSearch and KeywordSearch return lists of these. The hybrid
    layer merges them by ``chunk_id``.

    Attributes:
        chunk_id:    SHA-256 chunk identifier (primary key in ``chunks`` table).
        path:        Repo-relative file path, e.g. ``"memory/2026-03-01.md"``.
        source:      ``"memory"`` or ``"sessions"``.
        start_line:  1-indexed first line of the chunk in the source file.
        end_line:    1-indexed last line of the chunk.
        text:        Raw chunk text (used as snippet before truncation).
        score:       Relevance score in [0, 1].  Vector uses cosine similarity
                     converted from distance; keyword uses BM25-derived score.
        vector_score: Score contribution from vector search (None if keyword-only).
        text_score:   Score contribution from keyword search (None if vector-only).
    """

    chunk_id: str
    path: str
    source: str
    start_line: int
    end_line: int
    text: str
    score: float
    vector_score: float | None = None
    text_score: float | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Final user-facing search result after scoring and truncation.

    This is what ``MemoryStore.search()`` returns to callers.

    Attributes:
        path:         Repo-relative file path.
        start_line:   1-indexed first line of the chunk.
        end_line:     1-indexed last line of the chunk.
        score:        Combined relevance score in [0, 1].
        snippet:      Chunk text, truncated to ``snippet_max_chars``.
        source:       ``"memory"`` or ``"sessions"``.
        vector_score: Raw vector component (for debugging/inspection).
        text_score:   Raw keyword component (for debugging/inspection).
    """

    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    source: str
    vector_score: float | None = None
    text_score: float | None = None


@runtime_checkable
class SearchStrategy(Protocol):
    """Protocol for search backends.

    All backends (``VectorSearch``, ``KeywordSearch``, ``HybridSearch``) must
    implement this interface.  The protocol is ``@runtime_checkable`` so callers
    can use ``isinstance(obj, SearchStrategy)`` without inheritance.

    Example of a custom backend::

        class MySearch:
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
                ...

        assert isinstance(MySearch(), SearchStrategy)
    """

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
        """Run a search and return ranked rows.

        Args:
            db:            Open aiosqlite connection to the memweave SQLite DB.
            query:         Raw user query string (used by keyword / hybrid).
            query_vec:     L2-normalized embedding of ``query`` (used by vector
                           / hybrid).  May be ``None`` for keyword-only backends.
            model:         Embedding model name, e.g. ``"text-embedding-3-small"``.
                           Used to filter ``chunks`` rows to a consistent model.
            limit:         Maximum number of rows to return.
            source_filter: Optional source to restrict results to
                           (``"memory"`` or ``"sessions"``).  ``None`` = all
                           sources.

        Returns:
            List of :class:`RawSearchRow`, ordered by descending score.
            Empty list when no results match.
        """
        ...
