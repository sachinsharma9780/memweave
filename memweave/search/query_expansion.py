"""
memweave/search/query_expansion.py — Query expansion for FTS-only fallback mode.

Query expansion for FTS-only fallback mode.

When an embedding provider is unavailable, the search pipeline falls back to
FTS-only mode.  In that mode a plain AND query (as built by
:func:`~memweave.search.keyword.build_fts_query`) can be too strict — if even
one keyword is missing from a chunk, the chunk is excluded.  ``expand_query_for_fts``
builds a broader OR-based query that combines:
- The original (trimmed) query phrase for exact-match preference
- Individual keywords extracted via
  :func:`~memweave.search.keyword.extract_keywords`

The same :func:`~memweave.search.keyword.extract_keywords` and
:func:`~memweave.search.keyword.is_stop_word` functions used in keyword.py
are re-exported from here so callers that only need query expansion don't have
to import from two places.
"""

from __future__ import annotations

from typing import NamedTuple

# Re-export from keyword.py (single source of truth for stop words / tokenization)
from memweave.search.keyword import build_fts_query, extract_keywords, is_stop_word

__all__ = [
    "FtsQueryExpansion",
    "expand_query_for_fts",
    "extract_keywords",
    "is_stop_word",
    "build_fts_query",
]


class FtsQueryExpansion(NamedTuple):
    """Result of :func:`expand_query_for_fts`.

    Named-tuple with three fields: ``original``, ``keywords``, and ``expanded``.

    Attributes:
        original: The trimmed original query string.
        keywords: Meaningful keyword tokens extracted from ``original``
                  (stop words and short tokens removed).
        expanded: The final FTS5 MATCH expression:
                  ``"{original} OR kw1 OR kw2 OR ..."`` when keywords exist,
                  or just ``"{original}"`` when no meaningful keywords are found.
    """

    original: str
    keywords: list[str]
    expanded: str


def expand_query_for_fts(query: str) -> FtsQueryExpansion | None:
    """Build a broader FTS5 MATCH expression for FTS-only (no-embedding) mode.

    Produces an OR-joined query that includes:
    1. The trimmed original query string (for full-phrase / exact-match
       preference — FTS5 ranks chunks matching all terms higher).
    2. Each individual keyword extracted by
       :func:`~memweave.search.keyword.extract_keywords` as a bare OR term
       (stop words removed, so only meaningful tokens are added).

    Returns a :class:`FtsQueryExpansion` named-tuple with ``original``,
    ``keywords``, and ``expanded`` fields, or ``None`` if the query
    yields no FTS-tokenizable content (e.g. pure punctuation or empty string).

    Args:
        query: Raw user query string.

    Returns:
        :class:`FtsQueryExpansion` or ``None`` if the query has no tokens.

    Examples::

        expand_query_for_fts("which database did we pick?")
        # → FtsQueryExpansion(
        #       original="which database did we pick?",
        #       keywords=["database", "pick"],
        #       expanded='which database did we pick? OR database OR pick'
        #   )

        expand_query_for_fts("PostgreSQL connection pooling")
        # → FtsQueryExpansion(
        #       original="PostgreSQL connection pooling",
        #       keywords=["PostgreSQL", "connection", "pooling"],
        #       expanded='PostgreSQL connection pooling OR PostgreSQL OR connection OR pooling'
        #   )

        expand_query_for_fts("???")
        # → None
    """
    original = query.strip()

    # Guard: if no FTS-tokenizable content, return None rather than passing
    # an empty/broken string to SQLite MATCH.
    if not original or build_fts_query(original) is None:
        return None

    keywords = extract_keywords(original)

    # Build expanded query: original terms OR extracted keywords
    expanded = f"{original} OR {' OR '.join(keywords)}" if keywords else original

    return FtsQueryExpansion(original=original, keywords=keywords, expanded=expanded)
