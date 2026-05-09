"""
IDF-Weighted Keyword Boost.

A post-processor that re-ranks retrieved sessions by boosting sessions whose
text contains rare, discriminative query terms — terms that appear in few
sessions get higher weight than common terms that appear everywhere.

Algorithm
---------
1. Tokenize the query: lowercase, drop stopwords and tokens under 3 chars.
   Optionally also extract bigrams (adjacent non-stopword pairs) for
   matching exact compound phrases.

2. Build a session-level corpus from the retrieved chunks.  Multiple chunks
   from the same session are merged so a term appearing in two chunks of the
   same session only increments its document frequency (df) once.

3. Compute IDF per query token:
       idf(t) = log(N / (1 + df(t)))
   where N = number of unique sessions in the corpus.
   Rare terms (small df) receive high IDF; ubiquitous terms receive low IDF.

4. Compute a normalised IDF overlap score per chunk:
       idf_overlap(chunk) = Σ idf(t)  for t in query_tokens if t ∈ chunk_tokens
                            ──────────────────────────────────────────────────
                            Σ idf(t)  for t in query_tokens

5. Apply a multiplicative adjustment and re-sort:
       new_score = min(1.0, score × (1 + alpha × idf_overlap))

A multiplicative form keeps the relative ordering of strong vector results
stable — a session with score 0.9 still dominates a session with 0.3 even
after the boost.

Tunable parameters
------------------
alpha : float  (default 0.6)
    Boost strength.  Higher values give keyword overlap more weight over the
    vector score.  Tune on a dev split; recommended range 0.4–0.8.
use_bigrams : bool  (default False)
    If True, adjacent non-stopword word pairs are added to the IDF vocabulary.
    Bigrams naturally have higher IDF and match exact compound phrases that
    unigrams miss.
"""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path

from memweave.search.strategy import RawSearchRow

# ── Stopwords (shared with ECR) ───────────────────────────────────────────────

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


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, filter stopwords and short tokens."""
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _extract_bigrams(text: str) -> list[str]:
    """Adjacent non-stopword word pairs as space-joined strings.

    Example: "Did I book a flight to New York last week?"
      unigrams after filtering: ["book", "flight", "new", "york", "last", "week"]
      bigrams: ["book flight", "flight new", "new york", "york last", "last week"]

    Bigrams are matched as exact substrings in session text, so "new york"
    only fires if those two words appear consecutively — not just anywhere.
    """
    non_stop = _tokenize(text)
    return [f"{non_stop[i]} {non_stop[i + 1]}" for i in range(len(non_stop) - 1)]


# ── Post-processor ────────────────────────────────────────────────────────────


class IDFKeywordBooster:
    """Post-processor: multiplicative IDF-weighted keyword overlap boost.

    Parameters
    ----------
    alpha:
        Boost strength.  Tune on the dev split; recommended range 0.4–0.8.
    use_bigrams:
        If True, adjacent non-stopword word pairs (bigrams) are added to the
        IDF vocabulary alongside unigrams.  Bigrams naturally have higher IDF
        and match exact compound phrases that unigrams miss.
    """

    def __init__(self, *, alpha: float = 0.6, use_bigrams: bool = False) -> None:
        self.alpha = alpha
        self.use_bigrams = use_bigrams

    async def apply(
        self,
        rows: list[RawSearchRow],
        query: str,
        **kwargs: object,
    ) -> list[RawSearchRow]:
        """Re-rank rows using IDF-weighted keyword overlap.

        Steps:
          1. Tokenize query into unigrams (and optionally bigrams).
          2. Build session-level corpus from retrieved rows.
          3. Compute IDF per token/bigram from corpus.
          4. Score each row by normalized IDF overlap.
          5. Multiply row score by (1 + alpha × idf_overlap), re-sort.
        """
        if not rows:
            return rows

        alpha = float(kwargs.get("idf_alpha", self.alpha))
        use_bigrams = bool(kwargs.get("idf_bigrams", self.use_bigrams))

        # Step 1 — build query vocabulary (unigrams + optional bigrams)
        unigrams = list(dict.fromkeys(_tokenize(query)))
        bigrams = list(dict.fromkeys(_extract_bigrams(query))) if use_bigrams else []
        query_vocab = list(dict.fromkeys(unigrams + bigrams))
        if not query_vocab:
            return rows

        # Step 2 — build session-level corpus
        session_texts: dict[str, str] = {}
        for row in rows:
            sid = Path(row.path).stem
            if sid not in session_texts:
                session_texts[sid] = row.text
            else:
                session_texts[sid] += " " + row.text

        N = len(session_texts)
        if N == 0:
            return rows

        # Precompute token sets and lowercased texts per session
        session_token_sets: dict[str, frozenset[str]] = {
            sid: frozenset(_tokenize(text))
            for sid, text in session_texts.items()
        }
        session_text_lower: dict[str, str] = {
            sid: text.lower() for sid, text in session_texts.items()
        }

        # Step 3 — compute df and IDF
        # Unigrams: check token set membership
        # Bigrams: check substring in lowercased text
        df: dict[str, int] = {}
        for token in unigrams:
            df[token] = sum(1 for tset in session_token_sets.values() if token in tset)
        for bigram in bigrams:
            df[bigram] = sum(1 for txt in session_text_lower.values() if bigram in txt)

        idf: dict[str, float] = {
            t: math.log(N / (1 + df[t])) for t in query_vocab
        }
        total_idf = sum(idf.values())
        if total_idf <= 0.0:
            return rows

        # Steps 4–5 — score each row and adjust
        adjusted: list[RawSearchRow] = []
        for row in rows:
            row_tokens = frozenset(_tokenize(row.text))
            row_lower = row.text.lower()

            unigram_overlap = sum(idf[t] for t in unigrams if t in row_tokens)
            bigram_overlap = sum(idf[b] for b in bigrams if b in row_lower)
            idf_overlap = (unigram_overlap + bigram_overlap) / total_idf

            new_score = min(1.0, row.score * (1.0 + alpha * idf_overlap))
            adjusted.append(dataclasses.replace(row, score=new_score))

        adjusted.sort(key=lambda r: r.score, reverse=True)
        return adjusted
