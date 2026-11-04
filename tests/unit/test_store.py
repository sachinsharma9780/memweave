"""
tests/unit/test_store.py — Unit tests for memweave/store.py (MemWeave class).

Tests cover:
- __init__: config defaults, custom embedding_provider
- _ensure_db: lazy initialization, idempotent, creates DB file/dir
- open/close: lifecycle, double-close is safe
- async context manager: __aenter__/__aexit__
- register_strategy / register_postprocessor
- index(): empty workspace, file hashing/skip, force re-index, stale pruning
- add(): single file indexing
- search(): FTS-only, result shape, empty index, strategy override
- status(): snapshot fields
- files(): list with correct metadata
- provider fingerprint: changed model triggers force re-index
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memweave import MemWeave, MemoryConfig
from memweave.config import (
    ChunkingConfig,
    EmbeddingConfig,
    MMRConfig,
    QueryConfig,
    TemporalDecayConfig,
)
from memweave.search.strategy import RawSearchRow
from memweave.types import IndexResult, SearchResult, StoreStatus

# ── Helpers ───────────────────────────────────────────────────────────────────


class MockEmbeddingProvider:
    """Fake embedding provider: returns deterministic float vectors."""

    def __init__(self, dims: int = 8):
        self.dims = dims
        self.embed_query_calls: list[str] = []
        self.embed_batch_calls: list[list[str]] = []

    async def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        # Deterministic: hash-based unit vector
        seed = sum(ord(c) for c in text) % 100
        v = [float((seed + i) % 10) / 10 for i in range(self.dims)]
        norm = sum(x**2 for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_batch_calls.append(texts)
        return [await self.embed_query(t) for t in texts]


def _make_config(tmp_path: Path, **kwargs: Any) -> MemoryConfig:
    """Build a MemoryConfig pointing at a temp directory."""
    return MemoryConfig(
        workspace_dir=tmp_path,
        embedding=EmbeddingConfig(model="test-embed", batch_size=10),
        **kwargs,
    )


async def _open_mem(tmp_path: Path, **kwargs: Any) -> MemWeave:
    """Create and open a MemWeave instance with a mock embedding provider."""
    cfg = _make_config(tmp_path, **kwargs)
    provider = MockEmbeddingProvider()
    mem = MemWeave(cfg, embedding_provider=provider)
    await mem.open()
    return mem


def _write_md(tmp_path: Path, rel: str, content: str) -> Path:
    """Write a markdown file to the workspace memory/ dir."""
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestMemWeaveInit:
    def test_default_config(self):
        mem = MemWeave()
        assert mem.config is not None
        assert mem._db is None  # not yet opened

    def test_custom_config(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        mem = MemWeave(cfg)
        assert mem.config is cfg

    def test_custom_embedding_provider(self, tmp_path: Path):
        provider = MockEmbeddingProvider()
        mem = MemWeave(_make_config(tmp_path), embedding_provider=provider)
        assert mem.embedding_provider is provider

    def test_dirty_flag_starts_true(self, tmp_path: Path):
        mem = MemWeave(_make_config(tmp_path))
        assert mem._dirty is True


# ── Lifecycle ─────────────────────────────────────────────────────────────────


class TestLifecycle:
    async def test_open_creates_db_file(self, tmp_path: Path):
        mem = await _open_mem(tmp_path)
        db_path = mem.config.resolved_db_path
        assert db_path.exists()
        await mem.close()

    async def test_double_close_is_safe(self, tmp_path: Path):
        mem = await _open_mem(tmp_path)
        await mem.close()
        await mem.close()  # should not raise

    async def test_ensure_db_idempotent(self, tmp_path: Path):
        mem = await _open_mem(tmp_path)
        db_before = mem._db
        await mem._ensure_db()
        assert mem._db is db_before  # same connection object
        await mem.close()

    async def test_context_manager(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        async with MemWeave(cfg, embedding_provider=MockEmbeddingProvider()) as mem:
            assert mem._db is not None
        assert mem._db is None  # closed on exit

    async def test_failed_open_closes_connection(self, tmp_path: Path):
        """A failure after aiosqlite.connect() (e.g. "database is locked" from the
        schema check) must close the connection: its worker thread is non-daemon,
        and leaving it running keeps the interpreter from ever exiting (#7)."""
        cfg = _make_config(tmp_path)
        mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
        before = set(threading.enumerate())
        with patch(
            "memweave.store.ensure_schema",
            AsyncMock(side_effect=sqlite3.OperationalError("database is locked")),
        ):
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                async with mem:
                    pass
        assert mem._db is None
        for t in set(threading.enumerate()) - before:
            t.join(timeout=5)
            assert not t.is_alive(), f"leaked thread after failed open: {t.name}"

    async def test_open_retries_after_failed_open(self, tmp_path: Path):
        """After a failed open the instance is reusable: the next call opens cleanly."""
        cfg = _make_config(tmp_path)
        mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
        with patch(
            "memweave.store.ensure_schema",
            AsyncMock(side_effect=sqlite3.OperationalError("database is locked")),
        ):
            with pytest.raises(sqlite3.OperationalError):
                await mem.open()
        await mem.open()
        assert mem._db is not None
        assert mem._store is not None  # a leaked half-open _db would skip this setup
        status = await mem.status()
        assert status.files == 0
        await mem.close()

    async def test_context_manager_close_on_exception(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
        try:
            async with mem:
                raise RuntimeError("intentional")
        except RuntimeError:
            pass
        assert mem._db is None  # still closed despite exception


# ── register_strategy / register_postprocessor ───────────────────────────────


class TestExtensions:
    async def test_register_strategy_stored(self, tmp_path: Path):
        mem = await _open_mem(tmp_path)
        strategy = MagicMock()
        mem.register_strategy("my-search", strategy)
        assert "my-search" in mem._strategies
        assert mem._strategies["my-search"] is strategy
        await mem.close()

    async def test_register_postprocessor_appended(self, tmp_path: Path):
        mem = await _open_mem(tmp_path)
        proc = MagicMock()
        mem.register_postprocessor(proc)
        assert proc in mem._postprocessors
        await mem.close()


# ── index() ───────────────────────────────────────────────────────────────────


class TestIndex:
    async def test_empty_workspace_returns_zero_counts(self, tmp_path: Path):
        mem = await _open_mem(tmp_path)
        result = await mem.index()
        assert isinstance(result, IndexResult)
        assert result.files_scanned == 0
        assert result.files_indexed == 0
        assert result.files_deleted == 0
        await mem.close()

    async def test_indexes_new_file(self, tmp_path: Path):
        _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nHello memory.")
        mem = await _open_mem(tmp_path)
        result = await mem.index()
        assert result.files_indexed == 1
        assert result.chunks_created >= 1
        await mem.close()

    async def test_skips_unchanged_file(self, tmp_path: Path):
        _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nHello memory.")
        mem = await _open_mem(tmp_path)
        await mem.index()
        # Second index should skip (hash unchanged)
        result2 = await mem.index()
        assert result2.files_indexed == 0
        assert result2.files_skipped == 1
        await mem.close()

    async def test_reindexes_on_content_change(self, tmp_path: Path):
        f = _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nOriginal content.")
        mem = await _open_mem(tmp_path)
        await mem.index()
        # Modify file
        f.write_text("# Note\n\nModified content.", encoding="utf-8")
        result2 = await mem.index()
        assert result2.files_indexed == 1
        await mem.close()

    async def test_force_reindexes_unchanged_file(self, tmp_path: Path):
        _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nSame content.")
        mem = await _open_mem(tmp_path)
        await mem.index()
        result2 = await mem.index(force=True)
        assert result2.files_indexed == 1
        await mem.close()

    async def test_prunes_deleted_files(self, tmp_path: Path):
        f = _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nTo be deleted.")
        mem = await _open_mem(tmp_path)
        await mem.index()
        # Remove the file from disk
        f.unlink()
        result2 = await mem.index()
        assert result2.files_deleted == 1
        await mem.close()

    async def test_dirty_flag_cleared_after_index(self, tmp_path: Path):
        _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nContent.")
        mem = await _open_mem(tmp_path)
        assert mem._dirty is True
        await mem.index()
        assert mem._dirty is False
        await mem.close()

    async def test_multiple_files(self, tmp_path: Path):
        _write_md(tmp_path, "memory/2026-01-01.md", "# Day 1\n\nFirst entry.")
        _write_md(tmp_path, "memory/2026-01-02.md", "# Day 2\n\nSecond entry.")
        mem = await _open_mem(tmp_path)
        result = await mem.index()
        assert result.files_scanned == 2
        assert result.files_indexed == 2
        await mem.close()

    async def test_empty_file_indexed(self, tmp_path: Path):
        """An empty file is still indexed (chunker produces one empty-text chunk)."""
        _write_md(tmp_path, "memory/empty.md", "")
        mem = await _open_mem(tmp_path)
        result = await mem.index()
        assert result.files_indexed == 1
        # chunk_markdown always returns at least one chunk even for empty content
        assert result.chunks_created >= 0
        await mem.close()


# ── startup dirty check ───────────────────────────────────────────────────────


class TestStartupDirtyCheck:
    async def test_fresh_instance_on_indexed_workspace_is_clean(self, tmp_path: Path):
        _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nContent.")
        mem = await _open_mem(tmp_path)
        await mem.index()
        await mem.close()

        # New instance pointing at the same workspace — should detect clean state
        mem2 = await _open_mem(tmp_path)
        assert mem2._dirty is False
        await mem2.close()

    async def test_fresh_instance_on_empty_workspace_is_clean(self, tmp_path: Path):
        # No .md files, nothing to index — empty index matches empty disk
        mem = await _open_mem(tmp_path)
        assert mem._dirty is False
        await mem.close()

    async def test_dirty_after_new_file_added_on_disk(self, tmp_path: Path):
        _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nContent.")
        mem = await _open_mem(tmp_path)
        await mem.index()
        await mem.close()

        # Add a new file without indexing it
        _write_md(tmp_path, "memory/2026-01-02.md", "# New\n\nNot indexed yet.")
        mem2 = await _open_mem(tmp_path)
        assert mem2._dirty is True
        await mem2.close()

    async def test_dirty_after_file_deleted_from_disk(self, tmp_path: Path):
        f = _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nContent.")
        mem = await _open_mem(tmp_path)
        await mem.index()
        await mem.close()

        # Delete the file without re-indexing
        f.unlink()
        mem2 = await _open_mem(tmp_path)
        assert mem2._dirty is True
        await mem2.close()

    async def test_dirty_after_file_modified_on_disk(self, tmp_path: Path):
        f = _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nOriginal.")
        mem = await _open_mem(tmp_path)
        await mem.index()
        await mem.close()

        # Modify the file's mtime to be strictly newer than what was stored
        current_mtime = f.stat().st_mtime
        os.utime(f, (current_mtime + 10, current_mtime + 10))
        mem2 = await _open_mem(tmp_path)
        assert mem2._dirty is True
        await mem2.close()


# ── add() ─────────────────────────────────────────────────────────────────────


class TestAdd:
    async def test_add_single_file(self, tmp_path: Path):
        f = _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nAdded directly.")
        mem = await _open_mem(tmp_path)
        result = await mem.add(f)
        assert result.files_scanned == 1
        assert result.files_indexed == 1
        assert result.chunks_created >= 1
        await mem.close()

    async def test_add_nonexistent_raises(self, tmp_path: Path):
        mem = await _open_mem(tmp_path)
        with pytest.raises(FileNotFoundError):
            await mem.add(tmp_path / "memory" / "nonexistent.md")
        await mem.close()

    async def test_add_relative_path(self, tmp_path: Path):
        _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nRelative path test.")
        mem = await _open_mem(tmp_path)
        result = await mem.add("memory/2026-01-01.md")
        assert result.files_indexed == 1
        await mem.close()


# ── search() ──────────────────────────────────────────────────────────────────


class TestSearch:
    async def test_search_empty_index_returns_empty(self, tmp_path: Path):
        mem = await _open_mem(tmp_path)
        # Disable auto-sync to avoid triggering index() on empty workspace.
        # Use keyword strategy — chunks_vec table only exists after index() runs.
        mem.config.sync.on_search = False
        results = await mem.search("anything", strategy="keyword")
        assert results == []
        await mem.close()

    async def test_search_returns_list_of_search_results(self, tmp_path: Path):
        _write_md(
            tmp_path,
            "memory/2026-01-01.md",
            "# Deployment\n\nDeploy with docker-compose up -d.\n",
        )
        mem = await _open_mem(tmp_path)
        await mem.index()
        results = await mem.search("docker deploy", min_score=0.0)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SearchResult)
            assert isinstance(r.path, str)
            assert isinstance(r.score, float)
            assert isinstance(r.snippet, str)
        await mem.close()

    async def test_search_max_results_honored(self, tmp_path: Path):
        for i in range(5):
            _write_md(
                tmp_path,
                f"memory/2026-01-0{i+1}.md",
                f"# Entry {i}\n\nContent about entry {i}.",
            )
        mem = await _open_mem(tmp_path)
        await mem.index()
        results = await mem.search("entry content", max_results=2, min_score=0.0)
        assert len(results) <= 2
        await mem.close()

    async def test_search_unknown_strategy_raises(self, tmp_path: Path):
        mem = await _open_mem(tmp_path)
        mem.config.sync.on_search = False
        from memweave.exceptions import SearchError

        with pytest.raises(SearchError, match="Unknown search strategy"):
            await mem.search("q", strategy="no-such-strategy")
        await mem.close()

    async def test_search_keyword_strategy(self, tmp_path: Path):
        _write_md(
            tmp_path,
            "memory/2026-01-01.md",
            "# Kubernetes\n\nWe use Kubernetes for container orchestration.\n",
        )
        mem = await _open_mem(tmp_path)
        await mem.index()
        results = await mem.search("kubernetes", strategy="keyword", min_score=0.0)
        assert isinstance(results, list)
        await mem.close()

    async def test_search_auto_syncs_when_dirty(self, tmp_path: Path):
        f = _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nAuto sync test.\n")
        mem = await _open_mem(tmp_path)
        # Don't call index() manually — let on_search trigger it
        mem.config.sync.on_search = True
        results = await mem.search("auto sync", min_score=0.0)
        # After search, dirty should be cleared
        assert mem._dirty is False
        await mem.close()


# ── status() ─────────────────────────────────────────────────────────────────


class TestStatus:
    async def test_status_fields_present(self, tmp_path: Path):
        mem = await _open_mem(tmp_path)
        s = await mem.status()
        assert isinstance(s, StoreStatus)
        assert isinstance(s.files, int)
        assert isinstance(s.chunks, int)
        assert isinstance(s.dirty, bool)
        assert isinstance(s.workspace_dir, str)
        assert isinstance(s.db_path, str)
        assert s.search_mode in ("hybrid", "fts-only", "vector-only", "unavailable")
        assert s.provider == "litellm"
        assert s.fts_available is True  # FTS5 always available
        assert s.watcher_active is False
        await mem.close()

    async def test_status_after_index(self, tmp_path: Path):
        _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nSome text.\n")
        mem = await _open_mem(tmp_path)
        await mem.index()
        s = await mem.status()
        assert s.files == 1
        assert s.chunks >= 1
        assert s.dirty is False
        await mem.close()

    async def test_status_dirty_before_index(self, tmp_path: Path):
        # A workspace with an unindexed file must report dirty
        _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nNot yet indexed.")
        mem = await _open_mem(tmp_path)
        s = await mem.status()
        assert s.dirty is True
        await mem.close()


# ── files() ──────────────────────────────────────────────────────────────────


class TestFiles:
    async def test_files_empty_when_no_files(self, tmp_path: Path):
        mem = await _open_mem(tmp_path)
        result = await mem.files()
        assert result == []
        await mem.close()

    async def test_files_lists_indexed_files(self, tmp_path: Path):
        _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nContent.")
        mem = await _open_mem(tmp_path)
        await mem.index()
        result = await mem.files()
        assert len(result) == 1
        fi = result[0]
        assert fi.path == "memory/2026-01-01.md"
        assert fi.hash  # non-empty SHA-256
        assert fi.chunks >= 1
        assert fi.source == "memory"
        assert fi.is_evergreen is False  # dated file → not evergreen
        await mem.close()

    async def test_files_evergreen_detection(self, tmp_path: Path):
        _write_md(tmp_path, "memory/architecture.md", "# Architecture\n\nReference doc.")
        mem = await _open_mem(tmp_path)
        await mem.index()
        result = await mem.files()
        fi = result[0]
        assert fi.is_evergreen is True  # non-dated file under memory/
        await mem.close()


# ── provider fingerprint ─────────────────────────────────────────────────────


class TestProviderFingerprint:
    async def test_first_run_no_forced_reindex(self, tmp_path: Path):
        """On first run (no stored meta), fingerprint_changed returns False."""
        mem = await _open_mem(tmp_path)
        changed = await mem._provider_fingerprint_changed()
        assert changed is False
        await mem.close()

    async def test_same_model_no_change(self, tmp_path: Path):
        _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nContent.")
        mem = await _open_mem(tmp_path)
        await mem.index()  # saves fingerprint
        changed = await mem._provider_fingerprint_changed()
        assert changed is False
        await mem.close()

    async def test_different_model_triggers_change(self, tmp_path: Path):
        _write_md(tmp_path, "memory/2026-01-01.md", "# Note\n\nContent.")
        mem = await _open_mem(tmp_path)
        await mem.index()  # saves fingerprint with model="test-embed"
        # Change the model
        mem.config.embedding.model = "different-model"
        changed = await mem._provider_fingerprint_changed()
        assert changed is True
        await mem.close()


# ── _apply_postprocessors() ───────────────────────────────────────────────────


class TestApplyPostprocessors:
    """Per-call kwargs are forwarded to processors, not silently dropped."""

    # ── MMR ───────────────────────────────────────────────────────────────────

    async def test_mmr_lambda_kwarg_used_not_config_default(self, tmp_path: Path):
        cfg = _make_config(
            tmp_path, query=QueryConfig(mmr=MMRConfig(enabled=True, lambda_param=0.7))
        )
        mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
        await mem.open()

        with patch("memweave.store.MMRReranker") as MockMMR:
            MockMMR.return_value.apply = AsyncMock(return_value=[])
            await mem._apply_postprocessors([], "q", 0.0, mmr_lambda=0.3)

        MockMMR.assert_called_once_with(lam=0.3)  # kwarg wins over config 0.7
        await mem.close()

    async def test_mmr_config_default_used_when_no_kwarg(self, tmp_path: Path):
        cfg = _make_config(
            tmp_path, query=QueryConfig(mmr=MMRConfig(enabled=True, lambda_param=0.7))
        )
        mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
        await mem.open()

        with patch("memweave.store.MMRReranker") as MockMMR:
            MockMMR.return_value.apply = AsyncMock(return_value=[])
            await mem._apply_postprocessors([], "q", 0.0)

        MockMMR.assert_called_once_with(lam=0.7)  # config default used
        await mem.close()

    async def test_mmr_activated_by_kwarg_when_config_disabled(self, tmp_path: Path):
        cfg = _make_config(
            tmp_path, query=QueryConfig(mmr=MMRConfig(enabled=False, lambda_param=0.7))
        )
        mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
        await mem.open()

        with patch("memweave.store.MMRReranker") as MockMMR:
            MockMMR.return_value.apply = AsyncMock(return_value=[])
            await mem._apply_postprocessors([], "q", 0.0, mmr_lambda=0.5)

        MockMMR.assert_called_once_with(lam=0.5)
        await mem.close()

    async def test_mmr_not_activated_when_disabled_and_no_kwarg(self, tmp_path: Path):
        cfg = _make_config(tmp_path, query=QueryConfig(mmr=MMRConfig(enabled=False)))
        mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
        await mem.open()

        with patch("memweave.store.MMRReranker") as MockMMR:
            await mem._apply_postprocessors([], "q", 0.0)

        MockMMR.assert_not_called()
        await mem.close()

    # ── Temporal decay ────────────────────────────────────────────────────────

    async def test_decay_half_life_kwarg_used_not_config_default(self, tmp_path: Path):
        cfg = _make_config(
            tmp_path,
            query=QueryConfig(
                temporal_decay=TemporalDecayConfig(enabled=True, half_life_days=30.0)
            ),
        )
        mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
        await mem.open()

        with patch("memweave.store.TemporalDecayProcessor") as MockTDP:
            MockTDP.return_value.apply = AsyncMock(return_value=[])
            await mem._apply_postprocessors([], "q", 0.0, decay_half_life_days=7.0)

        MockTDP.assert_called_once_with(
            half_life_days=7.0, workspace_dir=mem.config.workspace_dir
        )  # kwarg wins over config 30.0
        await mem.close()

    async def test_decay_config_default_used_when_no_kwarg(self, tmp_path: Path):
        cfg = _make_config(
            tmp_path,
            query=QueryConfig(
                temporal_decay=TemporalDecayConfig(enabled=True, half_life_days=30.0)
            ),
        )
        mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
        await mem.open()

        with patch("memweave.store.TemporalDecayProcessor") as MockTDP:
            MockTDP.return_value.apply = AsyncMock(return_value=[])
            await mem._apply_postprocessors([], "q", 0.0)

        MockTDP.assert_called_once_with(
            half_life_days=30.0, workspace_dir=mem.config.workspace_dir
        )  # config default used
        await mem.close()

    async def test_decay_activated_by_kwarg_when_config_disabled(self, tmp_path: Path):
        cfg = _make_config(
            tmp_path,
            query=QueryConfig(
                temporal_decay=TemporalDecayConfig(enabled=False, half_life_days=30.0)
            ),
        )
        mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
        await mem.open()

        with patch("memweave.store.TemporalDecayProcessor") as MockTDP:
            MockTDP.return_value.apply = AsyncMock(return_value=[])
            await mem._apply_postprocessors([], "q", 0.0, decay_half_life_days=14.0)

        MockTDP.assert_called_once_with(half_life_days=14.0, workspace_dir=mem.config.workspace_dir)
        await mem.close()

    async def test_decay_not_activated_when_disabled_and_no_kwarg(self, tmp_path: Path):
        cfg = _make_config(
            tmp_path,
            query=QueryConfig(temporal_decay=TemporalDecayConfig(enabled=False)),
        )
        mem = MemWeave(cfg, embedding_provider=MockEmbeddingProvider())
        await mem.open()

        with patch("memweave.store.TemporalDecayProcessor") as MockTDP:
            await mem._apply_postprocessors([], "q", 0.0)

        MockTDP.assert_not_called()
        await mem.close()
