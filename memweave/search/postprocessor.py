"""
memweave/search/postprocessor.py — PostProcessor protocol and built-in processors.

Defines the duck-typed PostProcessor interface and provides three built-in
implementations:

- ``ScoreThreshold``  — filter results below a minimum score
- ``MMRReranker``     — Maximal Marginal Relevance diversity re-ranking
- ``TemporalDecayProcessor`` — exponential score decay by file age

Post-processors are applied sequentially (pipeline pattern).  Each processor
receives the output of the previous one.  Custom processors can be added to
the pipeline without subclassing — just implement the Protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from memweave.search.strategy import RawSearchRow


@runtime_checkable
class PostProcessor(Protocol):
    """Protocol for search result post-processors.

    Any object with an ``apply`` method matching this signature conforms to
    the protocol.  Use ``isinstance(obj, PostProcessor)`` to check at runtime.

    Example of a custom post-processor::

        class DedupByPath:
            async def apply(
                self,
                rows: list[RawSearchRow],
                query: str,
                **kwargs: object,
            ) -> list[RawSearchRow]:
                seen: set[str] = set()
                result = []
                for row in rows:
                    if row.path not in seen:
                        seen.add(row.path)
                        result.append(row)
                return result

        assert isinstance(DedupByPath(), PostProcessor)
    """

    async def apply(
        self,
        rows: list[RawSearchRow],
        query: str,
        **kwargs: object,
    ) -> list[RawSearchRow]:
        """Apply post-processing and return a (possibly modified) list.

        Args:
            rows:   Input rows from the previous stage (search or prior processor).
            query:  Original user query string (some processors use it for scoring).
            **kwargs: Optional per-call overrides passed through the pipeline.

        Returns:
            Processed list of :class:`~memweave.search.strategy.RawSearchRow`.
            May be a subset, reordered, or score-adjusted version of the input.
        """
        ...


class ScoreThreshold:
    """Filter out results whose score falls below a minimum threshold.

    This is the simplest post-processor: it removes any row whose
    ``score`` is strictly less than ``min_score``.

    Usage::

        processor = ScoreThreshold(min_score=0.35)
        rows = await processor.apply(rows, query)

        # Or override per-call:
        rows = await processor.apply(rows, query, min_score=0.5)

    Attributes:
        min_score: Default score threshold (inclusive lower bound). Default 0.35.
    """

    def __init__(self, *, min_score: float = 0.35) -> None:
        self.min_score = min_score

    async def apply(
        self,
        rows: list[RawSearchRow],
        query: str,  # noqa: ARG002
        **kwargs: object,
    ) -> list[RawSearchRow]:
        """Remove rows with score below ``min_score``.

        Args:
            rows:      Input rows.
            query:     Ignored.
            min_score: Per-call override for the threshold.

        Returns:
            Rows where ``row.score >= min_score``, in original order.
        """
        threshold = float(kwargs.get("min_score", self.min_score))  # type: ignore[arg-type]
        return [r for r in rows if r.score >= threshold]
