"""
tests/integration/test_temporal_decay.py — Happy-path: temporal decay penalises old files.

Covers per-call decay_half_life_days kwarg:
- Without decay: identical-content files score within 0.05 of each other.
- With half_life=7d and a 60-day-old file: new file scores > old file score × 5.
- Longer half-life preserves more of the old file's score than shorter half-life.

Requires: live embedding API (--embedding-model flag).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv("./.env")

from memweave import EmbeddingConfig, MemWeave, MemoryConfig, QueryConfig  # noqa: E402

pytestmark = pytest.mark.integration

_OLD_DATE = "2026-02-01"  # ~60 days before 2026-04-01
_TODAY_DATE = "2026-04-01"
_SHARED_CONTENT = (
    "The deployment pipeline uses blue-green switching for zero-downtime releases.\n"
    "Health checks must pass for 30 seconds before traffic is shifted.\n"
)


@pytest.mark.asyncio
async def test_temporal_decay(workspace: Path, embedding_model: str) -> None:
    mem_dir = workspace / "memory"
    (mem_dir / f"{_OLD_DATE}.md").write_text(_SHARED_CONTENT)
    (mem_dir / f"{_TODAY_DATE}.md").write_text(_SHARED_CONTENT)

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
        query=QueryConfig(min_score=0.0, max_results=10),
    )

    query = "blue-green deployment zero-downtime health check"
    old_path = f"memory/{_OLD_DATE}.md"
    new_path = f"memory/{_TODAY_DATE}.md"

    async with MemWeave(config) as mem:
        await mem.index()

        # Without decay — identical content should score very similarly
        r_no_decay = await mem.search(query, min_score=0.0)
        scores = {r.path: r.score for r in r_no_decay}

        assert old_path in scores, f"{old_path} not found in results"
        assert new_path in scores, f"{new_path} not found in results"

        diff = abs(scores[old_path] - scores[new_path])
        assert (
            diff < 0.05
        ), f"Identical content should score similarly without decay, diff={diff:.4f}"

        # With aggressive decay (half_life=7d) — 60-day-old file should drop sharply
        # 2^(-60/7) ≈ 0.003 multiplier
        r_decay = await mem.search(query, min_score=0.0, decay_half_life_days=7.0)
        decay_scores = {r.path: r.score for r in r_decay}

        old_decay = decay_scores.get(old_path, 0.0)
        new_decay = decay_scores.get(new_path, scores[new_path])

        assert (
            new_decay > old_decay * 5
        ), f"With half_life=7d, new should score >5× old. new={new_decay:.4f} old={old_decay:.6f}"

        # Results must be sorted by decayed score descending
        decay_score_list = [r.score for r in r_decay]
        assert decay_score_list == sorted(decay_score_list, reverse=True), (
            f"mem.search() with decay must return results sorted by decayed score. "
            f"Got: {decay_score_list}"
        )

        # Longer half-life preserves more of the old score
        r_long = await mem.search(query, min_score=0.0, decay_half_life_days=365.0)
        long_scores = {r.path: r.score for r in r_long}
        old_long = long_scores.get(old_path, scores[old_path])

        assert old_long > old_decay, "Longer half-life should preserve more of the old file's score"


_SESSION_OLD_DATE = "2026-01-01"  # ~100 days before 2026-04-11
_SESSION_NEW_DATE = "2026-04-01"  # ~10 days before 2026-04-11
_SESSION_CONTENT = (
    "We confirmed the rollback plan: revert the migration and restart the service.\n"
    "The on-call engineer will monitor for 24 hours after deployment.\n"
)


@pytest.mark.asyncio
async def test_session_subdir_dated_decay(workspace: Path, embedding_model: str) -> None:
    """Dated files under memory/sessions/ decay by filename date — same rule as memory/ root."""
    sessions_dir = workspace / "memory" / "sessions"
    sessions_dir.mkdir()

    (sessions_dir / f"{_SESSION_OLD_DATE}.md").write_text(_SESSION_CONTENT)
    (sessions_dir / f"{_SESSION_NEW_DATE}.md").write_text(_SESSION_CONTENT)

    config = MemoryConfig(
        workspace_dir=workspace,
        embedding=EmbeddingConfig(model=embedding_model),
        query=QueryConfig(min_score=0.0, max_results=10),
    )

    query = "rollback plan on-call deployment"
    old_path = f"memory/sessions/{_SESSION_OLD_DATE}.md"
    new_path = f"memory/sessions/{_SESSION_NEW_DATE}.md"

    async with MemWeave(config) as mem:
        await mem.index()

        # Without decay: identical content → nearly identical scores
        r_no_decay = await mem.search(query, min_score=0.0)
        scores = {r.path: r.score for r in r_no_decay}
        assert old_path in scores, f"{old_path} not found in results"
        assert new_path in scores, f"{new_path} not found in results"
        diff = abs(scores[old_path] - scores[new_path])
        assert diff < 0.05, f"Identical content without decay should score similarly, diff={diff:.4f}"

        # With aggressive decay: ~100-day-old session file should drop sharply
        # half_life=7d → 2^(-100/7) ≈ 0.000054 multiplier
        r_decay = await mem.search(query, min_score=0.0, decay_half_life_days=7.0)
        decay_scores = {r.path: r.score for r in r_decay}

        old_decay = decay_scores.get(old_path, 0.0)
        new_decay = decay_scores.get(new_path, scores.get(new_path, 0.0))

        assert new_decay > old_decay * 5, (
            f"With half_life=7d, new session should score >5× old. "
            f"new={new_decay:.4f} old={old_decay:.6f}"
        )
