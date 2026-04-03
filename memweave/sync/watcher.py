"""
memweave/sync/watcher.py — File system watcher for automatic re-indexing.

File system watcher for automatic re-indexing.

Watches the ``memory/`` directory inside the workspace for changes to ``.md``
files. When a change is detected, it calls the user-supplied ``on_change``
callback after a configurable debounce period.

Requirements:
    The ``watchfiles`` library must be installed::

        pip install watchfiles
        # or
        pip install memweave[watch]

    If ``watchfiles`` is not installed, importing this module raises
    ``ImportError`` — MemWeave handles this gracefully in
    ``MemWeave.start_watching()``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

try:
    import watchfiles

    _WATCHFILES_AVAILABLE = True
except ImportError:
    _WATCHFILES_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryWatcher:
    """Watches the memory directory and triggers re-indexing on changes.

    Uses ``watchfiles.awatch`` for cross-platform async file watching
    with minimal overhead.

    The watcher filters events to ``.md`` files only, and applies a debounce
    window (``debounce_ms``) to coalesce rapid successive changes (e.g. a
    text editor writing multiple times) into a single callback invocation.
    A debounce window coalesces rapid successive changes into a single callback.

    Usage::

        async def on_change(changed_paths: set[Path]) -> None:
            print(f"Changed: {changed_paths}")
            await mem.index()

        watcher = MemoryWatcher(
            workspace_dir=Path("/project"),
            on_change=on_change,
            debounce_ms=1500,
        )
        await watcher.run()   # blocks until cancelled

    Args:
        workspace_dir: Root workspace directory. The watcher monitors
                       ``workspace_dir/memory/`` recursively.
        on_change:     Async callback invoked with the set of changed
                       ``Path`` objects after each debounce window.
        debounce_ms:   Minimum ms between callback invocations. Defaults
                       to 1500 ms (the default debounce value).
    """

    def __init__(
        self,
        workspace_dir: Path,
        on_change: Callable[[set[Path]], Awaitable[None]],
        debounce_ms: int = 1500,
    ) -> None:
        if not _WATCHFILES_AVAILABLE:
            raise ImportError(
                "watchfiles is required for the file watcher. "
                "Install it with: pip install watchfiles"
            )
        self._workspace_dir = workspace_dir
        self._memory_dir = workspace_dir / "memory"
        self._on_change = on_change
        self._debounce_s = max(0.0, debounce_ms / 1000.0)

    async def run(self) -> None:
        """Watch the memory directory and invoke ``on_change`` on changes.

        Runs indefinitely until cancelled (via ``asyncio.CancelledError``).
        The ``MemWeave.start_watching()`` method wraps this in a background
        ``asyncio.Task``.

        Only monitors files with the ``.md`` extension. Ignores changes
        inside ``.memweave/`` (the SQLite database directory).
        """
        watch_path = self._memory_dir
        if not watch_path.exists():
            # Create the directory so watchfiles can watch it
            watch_path.mkdir(parents=True, exist_ok=True)

        logger.debug(
            "MemoryWatcher: watching %s (debounce=%sms)", watch_path, int(self._debounce_s * 1000)
        )

        try:
            async for changes in watchfiles.awatch(
                watch_path,
                debounce=int(self._debounce_s * 1000),
            ):
                changed_paths = self._filter_md_changes(changes)
                if changed_paths:
                    logger.debug("MemoryWatcher: detected changes in %s", changed_paths)
                    try:
                        await self._on_change(changed_paths)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning("MemoryWatcher: on_change callback raised: %s", exc)
        except asyncio.CancelledError:
            logger.debug("MemoryWatcher: stopped (cancelled)")
            raise

    def _filter_md_changes(self, changes: Any) -> set[Path]:
        """Return only .md file paths from a watchfiles change set.

        Ignores ``.memweave/`` internal files.

        Args:
            changes: Raw change set from ``watchfiles.awatch``. Each item is
                     a ``(ChangeType, path_str)`` tuple.

        Returns:
            Set of ``Path`` objects for changed ``.md`` files.
        """
        result: set[Path] = set()
        for _change_type, path_str in changes:
            p = Path(path_str)
            # Only track .md files, skip .memweave internals
            if p.suffix == ".md" and ".memweave" not in p.parts:
                result.add(p)
        return result
