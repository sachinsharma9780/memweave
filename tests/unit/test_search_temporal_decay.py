"""
tests/unit/test_search_temporal_decay.py — Unit tests for search/temporal_decay.py

Tests cover:
- to_decay_lambda: formula, edge cases (zero, negative, non-finite)
- calculate_decay_multiplier: brand-new/half-life/two-half-lives, negative age clamp
- apply_decay_to_score: score × multiplier
- parse_date_from_path: dated patterns, evergreen paths, invalid dates
- is_evergreen_path: MEMORY.md, memory.md, memory/topic.md, memory/YYYY-MM-DD.md
- age_in_days: positive, zero, future (clamp to 0)
- _extract_date: path date, evergreen skip, mtime fallback, no workspace_dir
- apply_temporal_decay: async integration, mtime fallback, caching, mixed list
- TemporalDecayProcessor: async apply, per-call override, workspace_dir kwarg
"""

from __future__ import annotations

import math
import os
import time
from datetime import date
from pathlib import Path

import pytest

from memweave.search.strategy import RawSearchRow
from memweave.search.temporal_decay import (
    TemporalDecayProcessor,
    _extract_date,
    age_in_days,
    apply_decay_to_score,
    apply_temporal_decay,
    calculate_decay_multiplier,
    is_evergreen_path,
    parse_date_from_path,
    to_decay_lambda,
)


def _row(chunk_id: str, path: str, score: float) -> RawSearchRow:
    return RawSearchRow(
        chunk_id=chunk_id,
        path=path,
        source="memory",
        start_line=1,
        end_line=3,
        text=f"text for {chunk_id}",
        score=score,
    )


# ── to_decay_lambda ───────────────────────────────────────────────────────────


class TestToDecayLambda:
    def test_30_day_half_life(self):
        lam = to_decay_lambda(30.0)
        assert abs(lam - math.log(2) / 30.0) < 1e-12

    def test_zero_half_life_returns_0(self):
        assert to_decay_lambda(0) == 0.0

    def test_negative_half_life_returns_0(self):
        assert to_decay_lambda(-5.0) == 0.0

    def test_infinity_returns_0(self):
        assert to_decay_lambda(float("inf")) == 0.0

    def test_nan_returns_0(self):
        assert to_decay_lambda(float("nan")) == 0.0

    def test_positive_result(self):
        assert to_decay_lambda(14.0) > 0


# ── calculate_decay_multiplier ────────────────────────────────────────────────


class TestCalculateDecayMultiplier:
    def test_brand_new_multiplier_is_1(self):
        assert calculate_decay_multiplier(0, 30) == pytest.approx(1.0)

    def test_one_half_life_is_0_5(self):
        assert calculate_decay_multiplier(30, 30) == pytest.approx(0.5, rel=1e-6)

    def test_two_half_lives_is_0_25(self):
        assert calculate_decay_multiplier(60, 30) == pytest.approx(0.25, rel=1e-6)

    def test_negative_age_clamped_to_zero(self):
        """Future files (negative age) should behave like age=0."""
        assert calculate_decay_multiplier(-5, 30) == pytest.approx(1.0)

    def test_zero_half_life_no_decay(self):
        """lambda=0 → multiplier always 1.0."""
        assert calculate_decay_multiplier(100, 0) == pytest.approx(1.0)

    def test_infinite_age_approaches_zero(self):
        multiplier = calculate_decay_multiplier(10_000, 30)
        assert multiplier < 1e-50

    def test_multiplier_in_range(self):
        for age in [0, 10, 30, 60, 180]:
            m = calculate_decay_multiplier(age, 30)
            assert 0 < m <= 1.0


# ── apply_decay_to_score ──────────────────────────────────────────────────────


class TestApplyDecayToScore:
    def test_score_multiplied(self):
        multiplier = calculate_decay_multiplier(30, 30)  # ≈ 0.5
        result = apply_decay_to_score(1.0, 30, 30)
        assert abs(result - multiplier) < 1e-9

    def test_brand_new_unchanged(self):
        assert apply_decay_to_score(0.8, 0, 30) == pytest.approx(0.8)

    def test_zero_score_stays_zero(self):
        assert apply_decay_to_score(0.0, 60, 30) == pytest.approx(0.0)


# ── parse_date_from_path ──────────────────────────────────────────────────────


class TestParseDateFromPath:
    def test_simple_dated_path(self):
        d = parse_date_from_path("memory/2026-03-28.md")
        assert d == date(2026, 3, 28)

    def test_dotslash_prefix(self):
        d = parse_date_from_path("./memory/2026-01-01.md")
        assert d == date(2026, 1, 1)

    def test_absolute_path(self):
        d = parse_date_from_path("/home/user/project/memory/2025-12-31.md")
        assert d == date(2025, 12, 31)

    def test_windows_backslash(self):
        d = parse_date_from_path("memory\\2026-03-28.md")
        assert d == date(2026, 3, 28)

    def test_non_dated_memory_path(self):
        assert parse_date_from_path("memory/MEMORY.md") is None

    def test_root_memory_md(self):
        assert parse_date_from_path("MEMORY.md") is None

    def test_invalid_date_digits(self):
        """Month 99 is not a valid date."""
        assert parse_date_from_path("memory/2026-99-01.md") is None

    def test_empty_string(self):
        assert parse_date_from_path("") is None

    def test_session_file(self):
        assert parse_date_from_path("sessions/2026-03-28.md") is None

    def test_subdir_dated_path(self):
        assert parse_date_from_path("memory/sessions/2026-03-28.md") == date(2026, 3, 28)

    def test_subdir_dated_path_researcher(self):
        assert parse_date_from_path("memory/researcher/2026-04-01.md") == date(2026, 4, 1)


# ── is_evergreen_path ─────────────────────────────────────────────────────────


class TestIsEvergreenPath:
    def test_memory_md_uppercase(self):
        assert is_evergreen_path("MEMORY.md") is True

    def test_memory_md_lowercase(self):
        assert is_evergreen_path("memory.md") is True

    def test_dotslash_prefix_memory_md(self):
        assert is_evergreen_path("./MEMORY.md") is True

    def test_topic_file_under_memory(self):
        assert is_evergreen_path("memory/architecture.md") is True

    def test_nested_topic_under_memory(self):
        assert is_evergreen_path("memory/project_notes.md") is True

    def test_dated_file_not_evergreen(self):
        assert is_evergreen_path("memory/2026-03-28.md") is False

    def test_session_file_not_evergreen(self):
        assert is_evergreen_path("sessions/2026-03-28.md") is False

    def test_arbitrary_file_not_evergreen(self):
        assert is_evergreen_path("docs/readme.md") is False

    def test_sessions_subdir_nondated_is_evergreen(self):
        assert is_evergreen_path("memory/sessions/foo.md") is True

    def test_sessions_subdir_dated_not_evergreen(self):
        assert is_evergreen_path("memory/sessions/2026-03-29.md") is False

    def test_researcher_subdir_nondated_is_evergreen(self):
        assert is_evergreen_path("memory/researcher/analysis.md") is True

    def test_researcher_subdir_dated_not_evergreen(self):
        assert is_evergreen_path("memory/researcher/2026-04-01.md") is False

    def test_custom_subdir_nondated_is_evergreen(self):
        assert is_evergreen_path("memory/episodes/standup.md") is True


# ── age_in_days ───────────────────────────────────────────────────────────────


class TestAgeInDays:
    def test_positive_age(self):
        file_date = date(2026, 1, 1)
        now = date(2026, 3, 28)
        assert age_in_days(file_date, now) == pytest.approx(86.0)

    def test_same_day_is_zero(self):
        d = date(2026, 3, 28)
        assert age_in_days(d, d) == 0.0

    def test_future_file_clamped_to_zero(self):
        file_date = date(2026, 4, 1)
        now = date(2026, 3, 28)
        assert age_in_days(file_date, now) == 0.0


# ── _extract_date (async) ─────────────────────────────────────────────────────


class TestExtractDate:
    async def test_dated_path_returned_directly(self):
        """No stat call needed when path contains YYYY-MM-DD."""
        d = await _extract_date("memory/2026-03-28.md", None)
        assert d == date(2026, 3, 28)

    async def test_evergreen_returns_none(self):
        d = await _extract_date("MEMORY.md", None)
        assert d is None

    async def test_evergreen_topic_returns_none(self):
        d = await _extract_date("memory/architecture.md", None)
        assert d is None

    async def test_no_workspace_dir_returns_none(self):
        """Undated non-evergreen file without workspace_dir → None (no decay)."""
        d = await _extract_date("sessions/q1.md", None)
        assert d is None

    async def test_mtime_fallback_fresh_file(self, tmp_path: Path):
        """A freshly created session file should have mtime ≈ today."""
        (tmp_path / "sessions").mkdir()
        f = tmp_path / "sessions" / "q1.md"
        f.write_text("Session content")

        d = await _extract_date("sessions/q1.md", tmp_path)
        assert d == date.today()

    async def test_mtime_fallback_old_file(self, tmp_path: Path):
        """A session file with mtime 100 days ago should return that date."""
        (tmp_path / "sessions").mkdir()
        f = tmp_path / "sessions" / "old.md"
        f.write_text("Old session content")
        old_ts = time.time() - (100 * 86_400)
        os.utime(f, (old_ts, old_ts))

        d = await _extract_date("sessions/old.md", tmp_path)
        assert d is not None
        assert age_in_days(d, date.today()) == pytest.approx(100.0, abs=1.5)

    async def test_missing_file_returns_none(self, tmp_path: Path):
        """Non-existent file → OSError → return None gracefully."""
        d = await _extract_date("sessions/nonexistent.md", tmp_path)
        assert d is None

    async def test_absolute_path_resolved(self, tmp_path: Path):
        """Absolute file_path is used directly (not joined with workspace_dir)."""
        f = tmp_path / "standalone.md"
        f.write_text("content")
        d = await _extract_date(str(f), tmp_path)
        assert d is not None  # mtime of freshly created file


# ── apply_temporal_decay (async) ──────────────────────────────────────────────


class TestApplyTemporalDecay:
    async def test_evergreen_score_unchanged(self):
        rows = [_row("ev", "MEMORY.md", 0.8)]
        result = await apply_temporal_decay(rows, half_life_days=30, now=date(2026, 3, 28))
        assert result[0].score == pytest.approx(0.8)

    async def test_dated_score_decayed(self):
        # 2026-01-01 to 2026-03-28 = 86 days → big decay
        rows = [_row("old", "memory/2026-01-01.md", 1.0)]
        result = await apply_temporal_decay(rows, half_life_days=30, now=date(2026, 3, 28))
        expected = apply_decay_to_score(1.0, 86.0, 30.0)
        assert result[0].score == pytest.approx(expected, rel=1e-6)

    async def test_brand_new_dated_score_unchanged(self):
        today = date(2026, 3, 28)
        rows = [_row("new", "memory/2026-03-28.md", 0.9)]
        result = await apply_temporal_decay(rows, half_life_days=30, now=today)
        assert result[0].score == pytest.approx(0.9)

    async def test_undated_no_workspace_score_unchanged(self):
        """Undated non-evergreen file with no workspace_dir → unchanged."""
        rows = [_row("nd", "sessions/misc.md", 0.7)]
        result = await apply_temporal_decay(rows, half_life_days=30, now=date(2026, 3, 28))
        assert result[0].score == pytest.approx(0.7)

    async def test_session_file_decayed_via_mtime(self, tmp_path: Path):
        """Sessions file with old mtime is decayed using mtime fallback."""
        (tmp_path / "sessions").mkdir()
        f = tmp_path / "sessions" / "old-session.md"
        f.write_text("We discussed MongoDB")
        old_ts = time.time() - (100 * 86_400)
        os.utime(f, (old_ts, old_ts))

        rows = [_row("s1", "sessions/old-session.md", 1.0)]
        result = await apply_temporal_decay(rows, half_life_days=30, workspace_dir=tmp_path)
        # 100 days at half_life=30 → multiplier ≈ 0.096
        assert result[0].score < 0.15

    async def test_session_file_fresh_score_nearly_unchanged(self, tmp_path: Path):
        """Sessions file created today has mtime ≈ today → minimal decay."""
        (tmp_path / "sessions").mkdir()
        f = tmp_path / "sessions" / "fresh.md"
        f.write_text("Fresh session content")

        rows = [_row("s2", "sessions/fresh.md", 1.0)]
        result = await apply_temporal_decay(rows, half_life_days=30, workspace_dir=tmp_path)
        assert result[0].score > 0.9

    async def test_memory_sessions_nondated_is_evergreen(self):
        """Non-dated memory/sessions/ file is evergreen — score unchanged."""
        rows = [_row("s_nd", "memory/sessions/known-facts.md", 0.8)]
        result = await apply_temporal_decay(rows, half_life_days=30, now=date(2026, 3, 28))
        assert result[0].score == pytest.approx(0.8)

    async def test_memory_sessions_dated_file_decays_by_date(self):
        """Dated memory/sessions/YYYY-MM-DD.md decays by filename date, no mtime needed."""
        rows = [_row("s_d", "memory/sessions/2026-01-01.md", 1.0)]
        result = await apply_temporal_decay(rows, half_life_days=30, now=date(2026, 3, 28))
        expected = apply_decay_to_score(1.0, 86.0, 30.0)
        assert result[0].score == pytest.approx(expected, rel=1e-6)

    async def test_memory_researcher_dated_file_decays_by_date(self):
        """Dated memory/researcher/YYYY-MM-DD.md decays by filename date."""
        rows = [_row("r_d", "memory/researcher/2026-01-01.md", 1.0)]
        result = await apply_temporal_decay(rows, half_life_days=30, now=date(2026, 3, 28))
        expected = apply_decay_to_score(1.0, 86.0, 30.0)
        assert result[0].score == pytest.approx(expected, rel=1e-6)

    async def test_mtime_cache_deduplicates_stat_calls(self, tmp_path: Path, monkeypatch):
        """Multiple chunks from the same file path are stat-ted only once."""
        import aiofiles.os as aio_os

        stat_count = 0
        _real_stat = aio_os.stat

        async def counting_stat(path, **kwargs):
            nonlocal stat_count
            stat_count += 1
            return await _real_stat(path, **kwargs)

        monkeypatch.setattr(aio_os, "stat", counting_stat)

        (tmp_path / "sessions").mkdir()
        f = tmp_path / "sessions" / "shared.md"
        f.write_text("content")

        rows = [
            _row("r1", "sessions/shared.md", 0.9),
            _row("r2", "sessions/shared.md", 0.7),  # same path
            _row("r3", "sessions/shared.md", 0.5),  # same path again
        ]
        await apply_temporal_decay(rows, workspace_dir=tmp_path)
        assert stat_count == 1  # cached — only one stat call despite 3 rows

    async def test_mixed_list(self):
        today = date(2026, 3, 28)
        rows = [
            _row("ev", "MEMORY.md", 0.8),
            _row("old", "memory/2026-01-01.md", 1.0),
            _row("new", "memory/2026-03-28.md", 0.6),
        ]
        result = await apply_temporal_decay(rows, half_life_days=30, now=today)
        id_map = {r.chunk_id: r for r in result}

        assert id_map["ev"].score == pytest.approx(0.8)  # evergreen: unchanged
        assert id_map["old"].score < 0.5  # 86 days old: heavily decayed
        assert id_map["new"].score == pytest.approx(0.6)  # same-day: unchanged

    async def test_returns_new_list(self):
        rows = [_row("a", "memory/2026-01-01.md", 1.0)]
        result = await apply_temporal_decay(rows, half_life_days=30, now=date(2026, 3, 28))
        assert result is not rows

    async def test_empty_input(self):
        assert await apply_temporal_decay([]) == []

    async def test_non_decayed_row_same_other_fields(self):
        """Decayed rows keep all fields except score."""
        row = _row("r1", "memory/2026-01-01.md", 1.0)
        result = await apply_temporal_decay([row], half_life_days=30, now=date(2026, 3, 28))
        r = result[0]
        assert r.chunk_id == row.chunk_id
        assert r.path == row.path
        assert r.text == row.text
        assert r.score != row.score  # was decayed

    async def test_default_now_is_today(self):
        """Passing now=None should use today without raising."""
        rows = [_row("x", "memory/2020-01-01.md", 1.0)]
        result = await apply_temporal_decay(rows)
        # Score should be significantly decayed (file is years old)
        assert result[0].score < 0.1

    async def test_output_sorted_by_decayed_score(self):
        """Results must be returned sorted by decayed score descending.

        Input order has the oldest file first (highest raw score) — after decay
        it should rank last, with today's file ranking first.
        """
        today = date(2026, 3, 28)
        rows = [
            _row("old", "memory/2024-01-01.md", 0.9),  # ~820 days → near-zero after decay
            _row("recent", "memory/2026-03-01.md", 0.7),  # 27 days  → moderate decay
            _row("new", "memory/2026-03-28.md", 0.6),  # today     → no decay
        ]
        result = await apply_temporal_decay(rows, half_life_days=30, now=today)

        scores = [r.score for r in result]
        assert scores == sorted(
            scores, reverse=True
        ), f"Results must be sorted by decayed score descending, got: {scores}"
        # new (no decay) > recent (moderate) > old (near-zero)
        assert [r.chunk_id for r in result] == ["new", "recent", "old"]


# ── TemporalDecayProcessor (PostProcessor wrapper) ───────────────────────────


class TestTemporalDecayProcessor:
    async def test_apply_decays_scored_rows(self):
        processor = TemporalDecayProcessor(half_life_days=30.0)
        rows = [_row("old", "memory/2020-01-01.md", 1.0)]
        result = await processor.apply(rows, "query")
        assert result[0].score < 0.01

    async def test_per_call_half_life_override(self):
        processor = TemporalDecayProcessor(half_life_days=30.0)
        rows = [_row("old", "memory/2020-01-01.md", 1.0)]
        # Very long half-life → little decay
        result_long = await processor.apply(rows, "q", decay_half_life_days=100_000.0)
        assert result_long[0].score > 0.9

    async def test_workspace_dir_in_constructor(self, tmp_path: Path):
        """workspace_dir passed to constructor is used for mtime fallback."""
        (tmp_path / "sessions").mkdir()
        f = tmp_path / "sessions" / "s1.md"
        f.write_text("content")
        old_ts = time.time() - (90 * 86_400)
        os.utime(f, (old_ts, old_ts))

        processor = TemporalDecayProcessor(half_life_days=30.0, workspace_dir=tmp_path)
        rows = [_row("s1", "sessions/s1.md", 1.0)]
        result = await processor.apply(rows, "query")
        assert result[0].score < 0.15  # ~90 days decayed

    async def test_workspace_dir_per_call_override(self, tmp_path: Path):
        """workspace_dir can be overridden per-call via kwargs."""
        (tmp_path / "sessions").mkdir()
        f = tmp_path / "sessions" / "s2.md"
        f.write_text("content")
        old_ts = time.time() - (90 * 86_400)
        os.utime(f, (old_ts, old_ts))

        processor = TemporalDecayProcessor(half_life_days=30.0)  # no workspace in ctor
        rows = [_row("s2", "sessions/s2.md", 1.0)]
        # Without workspace_dir: no mtime → unchanged
        no_ws = await processor.apply(rows, "q")
        assert no_ws[0].score == pytest.approx(1.0)

        # With workspace_dir via kwarg: mtime applied → decayed
        with_ws = await processor.apply(rows, "q", workspace_dir=tmp_path)
        assert with_ws[0].score < 0.15

    async def test_empty_input(self):
        processor = TemporalDecayProcessor()
        result = await processor.apply([], "query")
        assert result == []

    async def test_default_half_life_stored(self):
        processor = TemporalDecayProcessor(half_life_days=14.0)
        assert processor.half_life_days == 14.0

    async def test_default_workspace_dir_stored(self, tmp_path: Path):
        processor = TemporalDecayProcessor(workspace_dir=tmp_path)
        assert processor.workspace_dir == tmp_path

    async def test_output_sorted_by_decayed_score(self):
        """Processor must return results sorted by decayed score descending."""
        today_path = f"memory/{date.today().isoformat()}.md"
        processor = TemporalDecayProcessor(half_life_days=30.0)
        rows = [
            _row("old", "memory/2020-01-01.md", 0.9),  # years old → near-zero
            _row("new", today_path, 0.6),  # today     → no decay
        ]
        result = await processor.apply(rows, "query")
        scores = [r.score for r in result]
        assert scores == sorted(
            scores, reverse=True
        ), f"Processor output must be sorted by decayed score, got: {scores}"
        assert result[0].chunk_id == "new"  # today → no decay → ranks first
        assert result[1].chunk_id == "old"  # years old → near-zero → ranks last
