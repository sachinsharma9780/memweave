"""
Confidence-Adaptive Additive Temporal Boost (CAATB).

A post-processor that re-ranks retrieved sessions using two signals:
temporal proximity to the query's referenced date, and the embedding
model's confidence in each session.

Algorithm
---------
1. Parse a time reference from the query ("four weeks ago", "last month",
   "yesterday") after normalising written numbers to digits.  If none is
   found the ranker returns rows unchanged.

2. Compute a target date:
       target_date = question_date − days_back

3. Score each session by how close its date is to the target:
       temporal_score = 1.0                          if delta ≤ tolerance
                      = linear decay to 0.0          if tolerance < delta ≤ tolerance × 3
                      = 0.0                          beyond

4. Scale the boost by how uncertain the embedding already is about the
   session.  Sessions the vector model ranks highly (normalized score ≈ 1)
   receive near-zero boost; low-confidence sessions receive the full boost.
   This concentrates the lift where re-ranking is actually needed:
       alpha_effective = alpha × (1 − normalized_vector_score)

5. Apply an additive adjustment and re-sort:
       new_score = min(1.0, score + alpha_effective × temporal_score)

An additive (rather than multiplicative) form is important: it applies the
same absolute lift regardless of the session's starting score, which is
what is needed to move a borderline session across a fixed rank boundary.

Word-number normalisation handles spelled-out temporal phrases
("four weeks ago", "a couple of days") that digit-only patterns miss.

Tunable parameters
------------------
alpha : float  (default 0.2)
    Base boost magnitude.  Tune on a dev split; recommended range 0.1–0.3.
"""

from __future__ import annotations

import dataclasses
import re
from datetime import datetime, timedelta

from memweave.search.strategy import RawSearchRow

# ── Word-number normalization ─────────────────────────────────────────────────

_WORD_TO_NUM: list[tuple[str, int]] = sorted([
    ("a couple of", 2), ("a couple", 2), ("couple of", 2), ("couple", 2),
    ("a few", 3), ("few", 3),
    ("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5),
    ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9), ("ten", 10),
    ("eleven", 11), ("twelve", 12),
], key=lambda x: -len(x[0]))


def _normalize_numbers(text: str) -> str:
    for word, num in _WORD_TO_NUM:
        text = re.sub(r'\b' + re.escape(word) + r'\b', str(num), text, flags=re.IGNORECASE)
    return text


# ── Time offset extraction ────────────────────────────────────────────────────

_RAW_PATTERNS: list[tuple[str, object]] = [
    (r"(\d+)\s+days?\s+ago",   lambda m: (int(m.group(1)), 2)),
    (r"yesterday",             lambda m: (1, 1)),
    (r"(\d+)\s+weeks?\s+ago",  lambda m: (int(m.group(1)) * 7, 5)),
    (r"a\s+week\s+ago",        lambda m: (7, 3)),
    (r"last\s+week",           lambda m: (7, 3)),
    (r"(\d+)\s+months?\s+ago", lambda m: (int(m.group(1)) * 30, 10)),
    (r"a\s+month\s+ago",       lambda m: (30, 7)),
    (r"last\s+month",          lambda m: (30, 7)),
    (r"a\s+year\s+ago",        lambda m: (365, 30)),
    (r"last\s+year",           lambda m: (365, 30)),
    (r"recently",              lambda m: (14, 14)),
]
_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), fn) for p, fn in _RAW_PATTERNS]


def _parse_time_offset(query: str) -> tuple[int, int] | None:
    """Return (days_back, tolerance) after normalizing written numbers, or None."""
    normalized = _normalize_numbers(query)
    for pat, fn in _COMPILED_PATTERNS:
        m = pat.search(normalized)
        if m:
            return fn(m)
    return None


# ── Date parsing ──────────────────────────────────────────────────────────────

_SESSION_DATE_RE = re.compile(r"Date:\s*(\d{4})[/-](\d{2})[/-](\d{2})")
_QUESTION_DATE_RE = re.compile(r"(\d{4})[/-](\d{2})[/-](\d{2})")


def _parse_session_date(text: str) -> datetime | None:
    m = _SESSION_DATE_RE.search(text)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _parse_question_date(raw: str) -> datetime | None:
    m = _QUESTION_DATE_RE.search(raw)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


# ── Temporal score (smooth window) ────────────────────────────────────────────


def _temporal_score(delta_days: int, tolerance: int) -> float:
    """1.0 within tolerance, linear decay to 0.0 at tolerance×3, zero beyond."""
    if delta_days <= tolerance:
        return 1.0
    if delta_days <= tolerance * 3:
        return 1.0 - (delta_days - tolerance) / (tolerance * 2)
    return 0.0


# ── Booster ───────────────────────────────────────────────────────────────────


class ConfidenceAdaptiveTemporalBooster:
    """Post-processor: confidence-adaptive additive temporal boost.

    Parameters
    ----------
    alpha:
        Base boost magnitude.  Actual per-session boost is scaled by
        ``(1 - normalized_vector_score)``, so the effective maximum is
        ``alpha`` (only reached when vector confidence is zero).
        Tune on the dev split; recommended range 0.1–0.3.
    """

    def __init__(self, *, alpha: float = 0.4) -> None:
        self.alpha = alpha

    async def apply(
        self,
        rows: list[RawSearchRow],
        query: str,
        **kwargs: object,
    ) -> list[RawSearchRow]:
        """Re-rank rows using confidence-adaptive additive temporal boost.

        Steps:
          1. Normalize spelled-out numbers in query ("four weeks" → "4 weeks").
          2. Parse time offset → (days_back, tolerance); return unchanged if none.
          3. Compute target_date = question_date - days_back.
          4. Normalize vector scores across candidates (max = 1.0).
          5. For each row:
             a. Parse session date from "Date: YYYY/MM/DD" header.
             b. temporal_score: smooth window centred on target_date.
             c. alpha_effective = alpha × (1 - normalized_vector_score).
             d. new_score = min(1.0, score + alpha_effective × temporal_score).
          6. Re-sort descending.

        Returns rows unchanged when:
          - Query has no recognisable temporal reference.
          - question_date kwarg is absent or unparseable.
        """
        if not rows:
            return rows

        alpha = float(kwargs.get("caatb_alpha", self.alpha))

        # Step 1–2: parse temporal offset
        offset = _parse_time_offset(query)
        if offset is None:
            return rows
        days_back, tolerance = offset

        # Step 3: compute target date
        question_date = _parse_question_date(str(kwargs.get("question_date", "")))
        if question_date is None:
            return rows
        target_date = question_date - timedelta(days=days_back)

        # Step 4: normalize vector scores
        v_scores = [
            (r.vector_score if r.vector_score is not None else r.score)
            for r in rows
        ]
        max_vs = max(v_scores) if v_scores else 1.0
        if max_vs <= 0.0:
            max_vs = 1.0

        # Step 5: compute adjusted scores
        adjusted: list[RawSearchRow] = []
        for row, vs in zip(rows, v_scores):
            sess_date = _parse_session_date(row.text)
            if sess_date is not None:
                delta = abs((sess_date - target_date).days)
                t_score = _temporal_score(delta, tolerance)
            else:
                t_score = 0.0

            if t_score > 0.0:
                normalized_vs = vs / max_vs
                alpha_eff = alpha * (1.0 - normalized_vs)
                new_score = min(1.0, row.score + alpha_eff * t_score)
            else:
                new_score = row.score

            adjusted.append(dataclasses.replace(row, score=new_score))

        adjusted.sort(key=lambda r: r.score, reverse=True)
        return adjusted
