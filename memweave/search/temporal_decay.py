"""
memweave/search/temporal_decay.py — Exponential score decay by file age.

Exponential score decay by file age.

Dated memory files (``memory/YYYY-MM-DD.md``) have their scores multiplied by
an exponential decay factor based on how old the file is.  Evergreen files
(``MEMORY.md``, ``memory.md``, non-dated files under ``memory/``) are exempt
and their scores are left unchanged.

For undated, non-evergreen files (e.g. ``sessions/foo.md``) the file's mtime
is read from disk and used as the decay reference date.

Decay formula::

    λ = ln(2) / half_life_days
    multiplier = exp(−λ × age_days)
    decayed_score = original_score × multiplier

At ``age_days == 0`` the multiplier is 1.0 (no change).
At ``age_days == half_life_days`` the multiplier is 0.5.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timezone
from pathlib import Path

import aiofiles.os

from memweave.search.strategy import RawSearchRow

# Regex that matches dated memory paths: memory/YYYY-MM-DD.md
# Also matches one level of subdirectory: memory/sessions/YYYY-MM-DD.md
# Works with both ``memory/2026-03-28.md`` and ``./memory/sessions/2026-03-28.md``
_DATED_PATH_RE = re.compile(r"(?:^|\/)memory\/(?:[^/]+\/)?(\d{4})-(\d{2})-(\d{2})\.md$")

_DAY_SECONDS = 86_400.0


# ── Pure algorithm helpers (no I/O) ──────────────────────────────────────────


def to_decay_lambda(half_life_days: float) -> float:
    """Compute the exponential decay constant λ from a half-life.

    Formula: ``λ = ln(2) / half_life_days``

    Args:
        half_life_days: Number of days until a score halves. Must be > 0.

    Returns:
        Decay constant λ (≥ 0). Returns 0 for non-finite or non-positive inputs.

    Examples::

        to_decay_lambda(30.0)   # → 0.02310...  (ln2 / 30)
        to_decay_lambda(0)      # → 0.0
        to_decay_lambda(-1)     # → 0.0
    """
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        return 0.0
    return math.log(2) / half_life_days


def calculate_decay_multiplier(age_days: float, half_life_days: float) -> float:
    """Compute ``exp(−λ × age_days)`` — the score multiplier for a given age.

    Args:
        age_days:       Age of the file in days (negative values clamped to 0).
        half_life_days: Half-life for the decay curve.

    Returns:
        Multiplier in (0, 1].  Returns 1.0 when decay is disabled (λ=0) or
        age is 0.

    Examples::

        calculate_decay_multiplier(0, 30)    # → 1.0   (brand new)
        calculate_decay_multiplier(30, 30)   # → 0.5   (one half-life)
        calculate_decay_multiplier(60, 30)   # → 0.25  (two half-lives)
    """
    lam = to_decay_lambda(half_life_days)
    clamped_age = max(0.0, age_days)
    if lam <= 0 or not math.isfinite(clamped_age):
        return 1.0
    return math.exp(-lam * clamped_age)


def apply_decay_to_score(score: float, age_days: float, half_life_days: float) -> float:
    """Multiply a score by the temporal decay multiplier.

    Args:
        score:          Original relevance score.
        age_days:       Age of the file in days.
        half_life_days: Half-life for the decay curve.

    Returns:
        Decayed score (≤ original score).
    """
    return score * calculate_decay_multiplier(age_days, half_life_days)


def parse_date_from_path(file_path: str) -> date | None:
    """Extract ``YYYY-MM-DD`` date from a dated memory file path.

    Accepts paths like:
    - ``memory/2026-03-28.md``
    - ``./memory/2026-03-28.md``
    - ``/abs/path/memory/2026-03-28.md``

    Args:
        file_path: Relative or absolute file path.

    Returns:
        :class:`datetime.date` if the path matches the dated pattern and
        the date is valid.  ``None`` otherwise.

    Examples::

        parse_date_from_path("memory/2026-03-28.md")
        # → date(2026, 3, 28)

        parse_date_from_path("memory/MEMORY.md")
        # → None
    """
    match = _DATED_PATH_RE.search(file_path.replace("\\", "/"))
    if not match:
        return None
    try:
        year, month, day = int(match[1]), int(match[2]), int(match[3])
        return date(year, month, day)  # raises ValueError for invalid dates
    except ValueError:
        return None


def is_evergreen_path(file_path: str) -> bool:
    """Return True if a file path is considered "evergreen" (immune to decay).

    Evergreen paths:
    - ``MEMORY.md`` or ``memory.md`` (root bootstrap files)
    - Any non-dated file under ``memory/`` (topic/reference files)

    Dated files (``memory/YYYY-MM-DD.md``) are NOT evergreen.

    Args:
        file_path: Relative file path.

    Returns:
        ``True`` if the file should not be decayed.

    Examples::

        is_evergreen_path("MEMORY.md")               # → True
        is_evergreen_path("memory/architecture.md")  # → True
        is_evergreen_path("memory/2026-03-28.md")    # → False
        is_evergreen_path("sessions/foo.md")         # → False
    """
    normalized = file_path.replace("\\", "/").lstrip("./")
    if normalized in ("MEMORY.md", "memory.md"):
        return True
    if not normalized.startswith("memory/"):
        return False
    # Under memory/ but not dated → evergreen (applies at any depth)
    return _DATED_PATH_RE.search(file_path.replace("\\", "/")) is None


def age_in_days(file_date: date, now: date) -> float:
    """Compute the age of a file in fractional days.

    Args:
        file_date: The file's date (parsed from filename or mtime).
        now:       Current date.

    Returns:
        Age in days (≥ 0).  Returns 0 if ``file_date`` is in the future.
    """
    delta = now - file_date
    return max(0.0, float(delta.days))


# ── Async date extraction (with mtime fallback) ───────────────────────────────


async def _extract_date(
    file_path: str,
    workspace_dir: Path | None,
) -> date | None:
    """Extract the effective date for a file — from path first, then mtime.

    Steps:
    1. Parse date from filename (``memory/YYYY-MM-DD.md`` or
       ``memory/<subdir>/YYYY-MM-DD.md``). Return if found.
    2. If the file is evergreen → return ``None`` (no decay applies).
       All ``memory/`` files with non-dated filenames are evergreen at any depth.
    3. If ``workspace_dir`` is ``None`` → return ``None`` (cannot resolve path).
    4. Resolve the absolute path and ``stat()`` the file.
    5. Return the mtime as a :class:`datetime.date`, or ``None`` on error.

    Steps 3–5 (mtime fallback) are only reached for files outside ``memory/``
    (e.g. ``extra_paths`` external files with no date in their filename).
    All ``memory/``-managed files are fully resolved by steps 1–2.

    Args:
        file_path:     Relative or absolute file path.
        workspace_dir: Root workspace directory used to resolve relative paths.

    Returns:
        The reference date for decay, or ``None`` if the file should not decay.
    """
    # Step 1: date in filename
    from_path = parse_date_from_path(file_path)
    if from_path is not None:
        return from_path

    # Step 2: evergreen files never decay
    if is_evergreen_path(file_path):
        return None

    # Step 3: need workspace_dir to locate the file
    if workspace_dir is None:
        return None

    # Step 4 & 5: stat the file for mtime
    abs_path = Path(file_path) if Path(file_path).is_absolute() else workspace_dir / file_path
    try:
        stat_result = await aiofiles.os.stat(abs_path)
        mtime_date = datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc).date()
        return mtime_date
    except OSError:
        return None


async def apply_temporal_decay(
    rows: list[RawSearchRow],
    *,
    half_life_days: float = 30.0,
    now: date | None = None,
    workspace_dir: Path | str | None = None,
) -> list[RawSearchRow]:
    """Apply exponential temporal decay to a list of search rows.

    Steps for each row:
    1. Call :func:`_extract_date` to get a reference date:
       - Dated paths (``memory/YYYY-MM-DD.md``) → parsed from filename.
       - Evergreen paths → ``None`` (skip, score unchanged).
       - Undated non-evergreen (e.g. ``sessions/foo.md``) → file mtime via
         ``stat()`` (requires ``workspace_dir``).
    2. If no date → leave score unchanged.
    3. Compute ``age_days = (now - file_date).days``.
    4. Multiply ``row.score`` by the decay multiplier.

    A per-path cache avoids redundant ``stat()`` calls when multiple chunks
    come from the same file — using a per-path cache (``timestampPromiseCache`` pattern).

    Args:
        rows:           Input rows (score-sorted or unsorted).
        half_life_days: Half-life for the decay curve (default 30 days).
        now:            Reference date for age calculation.  Defaults to today
                        (UTC) when ``None``.
        workspace_dir:  Root workspace directory used to resolve relative paths
                        for the mtime fallback.  ``None`` disables mtime lookup.

    Returns:
        New list of :class:`~memweave.search.strategy.RawSearchRow` with
        decayed scores.  Rows without a resolvable date are returned unchanged.

    Examples::

        rows = [
            RawSearchRow("id1", "memory/2026-01-01.md", ..., score=1.0),
            RawSearchRow("id2", "MEMORY.md", ..., score=0.8),
            RawSearchRow("id3", "sessions/q1.md", ..., score=0.9),
        ]
        # With workspace_dir, sessions/q1.md is stat-ted for its mtime.
        decayed = await apply_temporal_decay(
            rows, half_life_days=30, workspace_dir=Path("/project")
        )
    """
    effective_now = now or date.today()
    resolved_workspace = Path(workspace_dir) if workspace_dir is not None else None

    # Per-path cache — avoids redundant stat() calls (timestampPromiseCache pattern).
    # Avoids multiple stat() calls when several chunks share the same file.
    date_cache: dict[str, date | None] = {}

    result: list[RawSearchRow] = []
    for row in rows:
        cache_key = row.path
        if cache_key not in date_cache:
            date_cache[cache_key] = await _extract_date(row.path, resolved_workspace)

        file_date = date_cache[cache_key]
        if file_date is None:
            result.append(row)
            continue

        days = age_in_days(file_date, effective_now)
        decayed_score = apply_decay_to_score(row.score, days, half_life_days)

        result.append(
            RawSearchRow(
                chunk_id=row.chunk_id,
                path=row.path,
                source=row.source,
                start_line=row.start_line,
                end_line=row.end_line,
                text=row.text,
                score=decayed_score,
                vector_score=row.vector_score,
                text_score=row.text_score,
            )
        )

    return sorted(result, key=lambda r: r.score, reverse=True)


# ── PostProcessor wrapper ─────────────────────────────────────────────────────


class TemporalDecayProcessor:
    """Temporal decay post-processor.

    Wraps :func:`apply_temporal_decay` in the
    :class:`~memweave.search.postprocessor.PostProcessor` interface.

    Usage::

        processor = TemporalDecayProcessor(half_life_days=30.0, workspace_dir=Path("."))
        rows = await processor.apply(rows, query)

        # Override half-life per-call:
        rows = await processor.apply(rows, query, decay_half_life_days=14.0)

    Attributes:
        half_life_days: Default half-life in days. Default 30.
        workspace_dir:  Root workspace directory for mtime fallback.
    """

    def __init__(
        self,
        *,
        half_life_days: float = 30.0,
        workspace_dir: Path | str | None = None,
    ) -> None:
        self.half_life_days = half_life_days
        self.workspace_dir = workspace_dir

    async def apply(
        self,
        rows: list[RawSearchRow],
        query: str,  # noqa: ARG002
        **kwargs: object,
    ) -> list[RawSearchRow]:
        """Apply temporal decay to scores.

        Args:
            rows:                 Input rows.
            query:                Ignored.
            decay_half_life_days: Per-call half-life override (float).
            workspace_dir:        Per-call workspace_dir override (str or Path).

        Returns:
            Rows with scores multiplied by decay factor.  Evergreen and
            undated files without a resolvable mtime are returned unchanged.
        """
        half_life = float(kwargs.get("decay_half_life_days", self.half_life_days))  # type: ignore[arg-type]
        workspace = kwargs.get("workspace_dir", self.workspace_dir)
        return await apply_temporal_decay(
            rows,
            half_life_days=half_life,
            workspace_dir=workspace,  # type: ignore[arg-type]
        )
