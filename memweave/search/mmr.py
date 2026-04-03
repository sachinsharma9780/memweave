"""
memweave/search/mmr.py — Maximal Marginal Relevance (MMR) re-ranking.

Maximal Marginal Relevance (MMR) re-ranking.

MMR balances relevance with diversity: at each step, the algorithm selects
the candidate that maximises::

    MMR = λ × normalised_relevance − (1−λ) × max_jaccard_similarity_to_selected

Reference: Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking" (1998)
"""

from __future__ import annotations

import re

from memweave.search.strategy import RawSearchRow

# Token pattern: lowercase alphanumeric + underscore
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


# ── Pure algorithm helpers (no I/O) ──────────────────────────────────────────


def tokenize_for_mmr(text: str) -> frozenset[str]:
    """Extract lowercase alphanumeric+underscore tokens from text.

    Args:
        text: Arbitrary text (chunk content or snippet).

    Returns:
        Frozen set of lowercase tokens — frozenset for hashability.

    Example::

        tokenize_for_mmr("PostgreSQL connection pooling")
        # → frozenset({"postgresql", "connection", "pooling"})
    """
    return frozenset(_TOKEN_RE.findall(text.lower()))


def jaccard_similarity(set_a: frozenset[str], set_b: frozenset[str]) -> float:
    """Compute Jaccard similarity between two token sets.

    Formula: ``|A ∩ B| / |A ∪ B|``, with special cases:
    - Both empty → 1.0 (identical empty sets)
    - One empty  → 0.0 (no overlap possible)

    Args:
        set_a: Token set for the first document.
        set_b: Token set for the second document.

    Returns:
        Similarity in [0, 1].

    Examples::

        jaccard_similarity(frozenset({"a", "b"}), frozenset({"a", "b"}))
        # → 1.0

        jaccard_similarity(frozenset({"a"}), frozenset({"b"}))
        # → 0.0

        jaccard_similarity(frozenset({"a", "b"}), frozenset({"b", "c"}))
        # → 0.333...   (|{b}| / |{a,b,c}|)
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a) + len(set_b) - intersection
    return intersection / union if union > 0 else 0.0


def compute_mmr_score(relevance: float, max_similarity: float, lam: float) -> float:
    """Compute the MMR score for a single candidate.

    Formula: ``λ × relevance − (1−λ) × max_similarity``

    Args:
        relevance:      Normalised relevance score in [0, 1].
        max_similarity: Maximum Jaccard similarity to any already-selected item.
        lam:            Lambda parameter; 0 = max diversity, 1 = max relevance.

    Returns:
        MMR score (may be negative when diversity penalty dominates).
    """
    return lam * relevance - (1.0 - lam) * max_similarity


def mmr_rerank(
    rows: list[RawSearchRow],
    *,
    lam: float = 0.7,
) -> list[RawSearchRow]:
    """Re-rank search rows using Maximal Marginal Relevance.

    Algorithm:
    1. Pre-tokenize all row texts and cache the token sets.
    2. Normalise scores to [0, 1] so relevance and similarity are on the
       same scale.
    3. Iteratively select the candidate with the highest MMR score:
       - ``MMR = λ × norm_relevance − (1−λ) × max_jaccard_to_selected``
    4. Use original score as a tiebreaker (higher raw score wins).

    Early exits:
    - ``len(rows) <= 1`` → return unchanged.
    - ``lam == 1.0`` → pure relevance ordering, no diversity penalty.

    Args:
        rows: Input rows (any order; typically sorted by score descending).
        lam:  Lambda parameter in [0, 1].
              0 = maximise diversity, 1 = maximise relevance (λ=1 is no-op).
              Clamped to [0, 1].

    Returns:
        Re-ranked list of rows in MMR order.

    Examples::

        # λ=0.7 (default): 70% relevance, 30% diversity
        reranked = mmr_rerank(rows, lam=0.7)

        # Pure diversity (novel results first)
        reranked = mmr_rerank(rows, lam=0.0)
    """
    if len(rows) <= 1:
        return list(rows)

    clamped_lam = max(0.0, min(1.0, lam))

    # Pure relevance: skip diversity computation entirely
    if clamped_lam == 1.0:
        return sorted(rows, key=lambda r: r.score, reverse=True)

    # Pre-tokenize (use chunk_id as cache key, text as content)
    token_cache: dict[str, frozenset[str]] = {r.chunk_id: tokenize_for_mmr(r.text) for r in rows}

    # Normalise scores to [0, 1]
    max_score = max(r.score for r in rows)
    min_score = min(r.score for r in rows)
    score_range = max_score - min_score

    def normalise(score: float) -> float:
        return 1.0 if score_range == 0.0 else (score - min_score) / score_range

    selected: list[RawSearchRow] = []
    remaining: list[RawSearchRow] = list(rows)

    while remaining:
        best_row: RawSearchRow | None = None
        best_mmr = float("-inf")

        for candidate in remaining:
            norm_rel = normalise(candidate.score)
            # Max Jaccard similarity to any already-selected row
            cand_tokens = token_cache[candidate.chunk_id]
            max_sim = max(
                (jaccard_similarity(cand_tokens, token_cache[sel.chunk_id]) for sel in selected),
                default=0.0,
            )
            mmr = compute_mmr_score(norm_rel, max_sim, clamped_lam)

            if mmr > best_mmr or (
                mmr == best_mmr
                and candidate.score > (best_row.score if best_row else float("-inf"))
            ):
                best_mmr = mmr
                best_row = candidate

        if best_row is not None:
            selected.append(best_row)
            remaining.remove(best_row)
        else:
            break  # safety exit (should never happen)

    return selected


# ── PostProcessor wrapper ─────────────────────────────────────────────────────


class MMRReranker:
    """MMR diversity re-ranking post-processor.

    Wraps :func:`mmr_rerank` in the :class:`~memweave.search.postprocessor.PostProcessor`
    interface.

    Usage::

        reranker = MMRReranker(lam=0.7)
        rows = await reranker.apply(rows, query)

        # Override λ per-call:
        rows = await reranker.apply(rows, query, mmr_lambda=0.3)

    Attributes:
        lam: Default lambda parameter (0=diversity, 1=relevance). Default 0.7.
    """

    def __init__(self, *, lam: float = 0.7) -> None:
        self.lam = lam

    async def apply(
        self,
        rows: list[RawSearchRow],
        query: str,  # noqa: ARG002
        **kwargs: object,
    ) -> list[RawSearchRow]:
        """Apply MMR re-ranking.

        Args:
            rows:        Input rows.
            query:       Ignored (MMR uses row text, not query).
            mmr_lambda:  Per-call lambda override.

        Returns:
            Re-ranked rows in MMR order.
        """
        lam = float(kwargs.get("mmr_lambda", self.lam))  # type: ignore[arg-type]
        return mmr_rerank(rows, lam=lam)
