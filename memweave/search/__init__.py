"""
memweave/search — Search strategy implementations and post-processors.

Public surface:
    SearchResult            — single ranked result from any search backend
    VectorSearch            — sqlite-vec cosine similarity search
    KeywordSearch           — SQLite FTS5 BM25 keyword search
    HybridSearch            — weighted merge of vector + keyword results
    build_fts_query         — sanitize raw query into FTS5 MATCH expression
    bm25_rank_to_score      — convert FTS5 BM25 rank to [0, 1] float
    extract_keywords        — tokenize query, strip stop words
    expand_query_for_fts    — build OR-expanded FTS query for no-embedding fallback
    PostProcessor           — protocol for result post-processors
    ScoreThreshold          — filter results below a minimum score
    MMRReranker             — Maximal Marginal Relevance diversity re-ranking
    TemporalDecayProcessor  — exponential score decay by file age
"""

from memweave.search.hybrid import HybridSearch, merge_hybrid_results
from memweave.search.keyword import (
    KeywordSearch,
    bm25_rank_to_score,
    build_fts_query,
    extract_keywords,
)
from memweave.search.mmr import MMRReranker
from memweave.search.postprocessor import PostProcessor, ScoreThreshold
from memweave.search.query_expansion import expand_query_for_fts
from memweave.search.strategy import RawSearchRow, SearchResult, SearchStrategy
from memweave.search.temporal_decay import TemporalDecayProcessor
from memweave.search.vector import VectorSearch

__all__ = [
    "SearchStrategy",
    "SearchResult",
    "RawSearchRow",
    "VectorSearch",
    "KeywordSearch",
    "HybridSearch",
    "build_fts_query",
    "bm25_rank_to_score",
    "extract_keywords",
    "merge_hybrid_results",
    "expand_query_for_fts",
    "PostProcessor",
    "ScoreThreshold",
    "MMRReranker",
    "TemporalDecayProcessor",
]
