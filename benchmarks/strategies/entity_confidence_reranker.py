"""
Entity-Confidence Re-ranker (ECR).

A post-processor that re-ranks retrieved sessions by combining two signals:
named entity overlap with the query, and the embedding model's confidence
in each session.

Algorithm
---------
1. Extract discriminative terms from the query in priority order:
     a. Quoted phrases          e.g. "machine learning"
     b. Multi-word proper nouns e.g. Golden Gate Bridge
     c. All-caps acronyms       e.g. GPT, NASA
     d. Hyphenated compounds    e.g. follow-up, real-time
   If none are found, fall back to non-stopword tokens (3+ chars).
   Preference-type queries (containing words like "suggest", "recommend",
   "enjoy", "tips") are skipped entirely — entity boosting degrades them
   because their answers are expressed implicitly in session text rather
   than via specific named entities.

2. For each candidate session, compute:
     entity_signal  = fraction of query entities present in session text
     boost_weight   = alpha × (1 − normalized_vector_score)

   The boost weight is inversely proportional to the embedding's confidence:
   the top-ranked session (normalized score = 1.0) receives zero boost;
   lower-confidence sessions receive progressively more. This concentrates
   the correction on sessions that need re-ranking most.

3. Apply an additive adjustment and re-sort:
     new_score = min(1.0, score + boost_weight × entity_signal)

Sessions with no entity overlap are unchanged. The additive form ensures
that entity evidence can move a borderline session across a rank boundary
without distorting the relative ordering of already-strong results.

Tunable parameters
------------------
alpha : float  (default 0.3)
    Boost strength.  Tune on a dev split; recommended range 0.2–0.4.
    Can be overridden per search call via the ``ec_alpha`` kwarg.
"""

from __future__ import annotations

import dataclasses
import re

from memweave.search.strategy import RawSearchRow

# ── Stopwords ─────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall",
    "i", "me", "my", "we", "our", "you", "your", "he", "his", "she", "her",
    "it", "its", "they", "their", "them", "this", "that", "these", "those",
    "who", "which", "what", "when", "where", "how", "why",
    "tell", "told", "said", "say", "says", "think", "know", "get", "give",
    "make", "take", "come", "see", "look", "want", "need", "use", "used",
    "got", "made", "came", "went", "gave", "took", "saw",
    "also", "just", "now", "then", "about", "into", "some", "any", "all",
    "not", "no", "so", "up", "out", "new", "one", "two", "three",
    "like", "more", "other", "very", "still", "much",
})

# ── Preference question detection ─────────────────────────────────────────────

# Queries containing these tokens express answers implicitly in session text
# rather than via named entities — entity boosting degrades them, so they
# are passed through unchanged.
_PREFERENCE_TOKENS = frozenset({
    "prefer", "preference", "preferred", "prefers",
    "favourite", "favorite", "favourites", "favorites",
    "enjoy", "enjoying", "enjoyed",
    "recommend", "recommendation", "recommendations",
    "suggest", "suggestion", "suggestions",
    "tips", "advice",
    "interested", "interest",
})

# Multi-word phrases that also signal preference questions.
_PREFERENCE_PHRASES = (
    "what kind", "what type", "what sort",
    "what do i", "what is my", "what are my",
)


def _is_preference_question(query: str) -> bool:
    """Return True if the query is likely asking about a user preference.

    Preference questions (e.g. "What music do I enjoy?", "Can you recommend
    resources for video editing?") express their answers implicitly in sessions.
    Entity/keyword boosting does not help these — it uniformly lifts all sessions
    containing the generic category word ("music", "video") rather than the
    specific preference. Skipping the boost preserves the vector ranking, which
    handles these questions well at baseline.
    """
    lowered = query.lower()
    tokens = set(re.findall(r'\b[a-z]+\b', lowered))
    if tokens & _PREFERENCE_TOKENS:
        return True
    return any(phrase in lowered for phrase in _PREFERENCE_PHRASES)

# ── Entity extraction ─────────────────────────────────────────────────────────


def extract_entities(query: str) -> list[str]:
    """Extract named entities and compound nouns from the query string.

    Priority order (most to least specific):
      1. Quoted phrases              "machine learning"
      2. Multi-word proper nouns     Golden Gate Bridge
      3. All-caps acronyms           GPT, NASA
      4. Hyphenated compounds        follow-up, real-time

    Returns lowercase strings for case-insensitive matching against session text.
    Deduplicates while preserving priority order.
    """
    entities: list[str] = []

    # Quoted phrases — user explicitly specifies exact match
    entities.extend(m.lower() for m in re.findall(r'"([^"]+)"', query))

    # Multi-word capitalized sequences (proper noun phrases)
    entities.extend(
        m.lower()
        for m in re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b', query)
    )

    # All-caps acronyms (2+ chars)
    entities.extend(
        m.lower() for m in re.findall(r'\b([A-Z]{2,})\b', query)
    )

    # Hyphenated technical terms: Hardware-Aware, all-in-one
    entities.extend(
        m.lower()
        for m in re.findall(r'\b([A-Za-z]{2,}(?:-[A-Za-z]{2,})+)\b', query)
    )

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique


def fallback_keywords(query: str) -> list[str]:
    """Non-stopword tokens (3+ chars) as fallback when no entities are detected."""
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    return [t for t in tokens if t not in _STOPWORDS]


# ── Signal computation ────────────────────────────────────────────────────────


def _entity_signal(entities: list[str], text: str) -> float:
    """Fraction of query entities found in chunk text. Returns 0.0 if no entities."""
    if not entities:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for e in entities if e in text_lower)
    return hits / len(entities)


# ── Post-processor ────────────────────────────────────────────────────────────


class EntityConfidenceReranker:
    """Post-processor: adaptive confidence boost weighted by entity overlap.

    Parameters
    ----------
    alpha:
        Boost strength. Tune on the dev split; recommended range 0.2–0.4.
        Can be overridden per search call via ``ec_alpha=<value>`` kwarg.
    """

    def __init__(self, *, alpha: float = 0.3) -> None:
        self.alpha = alpha

    async def apply(
        self,
        rows: list[RawSearchRow],
        query: str,
        **kwargs: object,
    ) -> list[RawSearchRow]:
        """Re-rank rows using entity-confidence scoring.

        Steps:
          1. Extract entities from query (fall back to keyword tokens if none).
          2. Compute max vector score across all candidates for normalization.
          3. For each row: boost_weight = alpha * (1 - normalized_vector_score).
          4. For each row: entity_signal = fraction of query entities in row.text.
          5. adjusted_score = row.score + boost_weight * entity_signal.
          6. Re-sort by adjusted_score descending.

        Rows with no entity overlap are unchanged (boost = 0). The top vector
        result (normalized_score = 1.0) is always unchanged (boost_weight = 0).
        """
        if not rows:
            return rows

        alpha = float(kwargs.get("ec_alpha", self.alpha))

        # Preference questions answer via implicit phrasing, not named entities.
        # Entity boosting degrades them — skip it and let vector ranking stand.
        if _is_preference_question(query):
            return rows

        # Step 1 — entity extraction
        entities = extract_entities(query)
        if not entities:
            entities = fallback_keywords(query)
        if not entities:
            # Empty or all-stopword query — no signal available
            return rows

        # Step 2 — max vector score for normalization
        v_scores = [
            (r.vector_score if r.vector_score is not None else r.score)
            for r in rows
        ]
        max_vs = max(v_scores) if v_scores else 1.0
        if max_vs <= 0.0:
            max_vs = 1.0

        # Steps 3–5 — compute adjusted scores
        adjusted: list[RawSearchRow] = []
        for row, vs in zip(rows, v_scores):
            normalized = vs / max_vs
            boost_weight = alpha * (1.0 - normalized)
            signal = _entity_signal(entities, row.text)
            new_score = min(1.0, row.score + boost_weight * signal)
            adjusted.append(dataclasses.replace(row, score=new_score))

        # Step 6 — re-sort descending
        adjusted.sort(key=lambda r: r.score, reverse=True)
        return adjusted
