"""
memweave/storage/files.py — Memory file discovery and metadata utilities.

This module handles everything related to finding and describing markdown files
on disk. It is a pure-function layer with no database dependencies — callers
pass the results to ``SQLiteStore`` for persistence.

Key concepts:
- **Discovery**: ``list_memory_files`` scans ``workspace_dir/memory/`` for ``.md``
  files, optionally including user-configured ``extra_paths``.
- **Change detection**: ``build_file_entry`` computes a SHA-256 hash of the file
  content. The indexer compares this hash against the stored hash to decide
  whether re-indexing is needed.
- **Source labelling**: every file gets a logical source string (``"memory"``,
  ``"sessions"``, ``"researcher"``, etc.) derived from its directory path. This
  lets search callers filter results by source.
- **Evergreen status**: files exempt from temporal decay (MEMORY.md,
  non-dated files) are identified by ``is_evergreen``.
"""

from __future__ import annotations

import re
from pathlib import Path

from memweave._internal.hashing import sha256_file

# Matches dated daily log files: ``YYYY-MM-DD.md``
# Files NOT matching this pattern are considered "evergreen" (no temporal decay).
_DATED_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")


def list_memory_files(workspace_dir: Path, extra_paths: list[str] | None = None) -> list[Path]:
    """Discover all ``.md`` files that MemWeave should index.

    Scans ``workspace_dir/memory/`` recursively for markdown files. The
    internal ``.memweave/`` directory (where the SQLite database lives) is
    always skipped to avoid indexing database artefacts.

    If ``extra_paths`` are configured, each entry is resolved relative to
    ``workspace_dir`` (unless it is already absolute) and appended to the
    result. Individual files and directories are both supported.

    Duplicate paths are deduplicated — a path present in both ``memory/`` and
    ``extra_paths`` appears only once.

    Args:
        workspace_dir: Root workspace directory. The function looks for a
                       ``memory/`` subdirectory inside this path.
        extra_paths:   Additional paths to include. Each entry may be:
                       - An absolute path to a ``.md`` file.
                       - A relative path (resolved against ``workspace_dir``).
                       - A directory (all ``.md`` files inside are included).

    Returns:
        Sorted list of absolute ``Path`` objects, one per discovered file.

    Example::

        files = list_memory_files(
            workspace_dir=Path("/project"),
            extra_paths=["docs/reference.md", "/shared/team-notes/"],
        )
        # Returns all .md in /project/memory/, plus docs/reference.md,
        # plus all .md files under /shared/team-notes/
    """
    memory_dir = workspace_dir / "memory"
    found: list[Path] = []

    if memory_dir.exists() and memory_dir.is_dir():
        for path in memory_dir.rglob("*.md"):
            # Skip .memweave internal directory (SQLite database, lock files, etc.)
            if ".memweave" in path.parts:
                continue
            if path.is_file():
                found.append(path)

    # Resolve and append extra_paths
    for extra in extra_paths or []:
        ep = Path(extra)
        if not ep.is_absolute():
            ep = workspace_dir / ep
        if ep.is_file() and ep.suffix == ".md":
            if ep not in found:
                found.append(ep)
        elif ep.is_dir():
            for path in ep.rglob("*.md"):
                if path.is_file() and path not in found:
                    found.append(path)

    return sorted(found)


def build_file_entry(path: Path) -> dict[str, str | float | int]:
    """Compute file metadata needed for change detection.

    Reads the file's SHA-256 content hash, modification time, and size.
    The indexer compares ``hash`` against the stored value in the ``files``
    table — if they match, the file is skipped (no re-chunking/embedding).

    Note: ``mtime`` and ``size`` are stored for informational purposes only;
    the hash is the authoritative change indicator.

    Args:
        path: Absolute path to an existing ``.md`` file.

    Returns:
        Dictionary with keys:
        - ``"hash"`` (str): SHA-256 hex digest of file content.
        - ``"mtime"`` (float): Last modification timestamp (Unix epoch).
        - ``"size"`` (int): File size in bytes.

    Raises:
        OSError: If the file does not exist or cannot be read.

    Example::

        entry = build_file_entry(Path("/project/memory/2026-03-21.md"))
        # {"hash": "a3f5c9...", "mtime": 1742000000.0, "size": 4096}
    """
    stat = path.stat()
    return {
        "hash": sha256_file(path),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def get_source_from_path(path: Path, workspace_dir: Path) -> str:
    """Determine the logical source label for a file.

    Source labels are used to tag chunks so search results can be filtered
    by origin (e.g. ``results = await mem.search(q, source="sessions")``).

    Derivation rules (applied in order):
    1. If the file is not under ``workspace_dir/memory/``, return ``"external"``.
    2. If the file is directly inside ``memory/`` (no sub-directory), return
       ``"memory"``.
    3. Otherwise, return the name of the immediate sub-directory, e.g.
       ``memory/sessions/foo.md`` → ``"sessions"``.

    Args:
        path:          Absolute path to the file.
        workspace_dir: Root workspace directory used to compute the relative path.

    Returns:
        Source label string.

    Examples::

        # memory/2026-03-21.md → directly under memory/
        get_source_from_path(Path("/proj/memory/2026-03-21.md"), Path("/proj"))
        # → "memory"

        # memory/sessions/s1.md → sub-directory "sessions"
        get_source_from_path(Path("/proj/memory/sessions/s1.md"), Path("/proj"))
        # → "sessions"

        # memory/researcher/analysis.md → sub-directory "researcher"
        get_source_from_path(Path("/proj/memory/researcher/analysis.md"), Path("/proj"))
        # → "researcher"

        # /other/file.md → outside workspace
        get_source_from_path(Path("/other/file.md"), Path("/proj"))
        # → "external"
    """
    try:
        rel = path.relative_to(workspace_dir / "memory")
    except ValueError:
        return "external"

    parts = rel.parts
    if len(parts) == 1:
        # File directly under memory/ (no sub-directory)
        return "memory"
    # First path component after memory/ is the source label
    return parts[0]


def is_memory_path(path: Path, workspace_dir: Path) -> bool:
    """Return ``True`` if ``path`` is located inside ``workspace_dir/memory/``.

    Used as a quick guard before computing source labels or evergreen status.

    Args:
        path:          Absolute path to check.
        workspace_dir: Root workspace directory.

    Returns:
        ``True`` if ``path`` is under ``workspace_dir/memory/``, else ``False``.
    """
    try:
        path.relative_to(workspace_dir / "memory")
        return True
    except ValueError:
        return False


def is_evergreen(path: Path, evergreen_patterns: list[str]) -> bool:
    """Return ``True`` if this file is exempt from temporal decay scoring.

    Temporal decay lowers the search score of older chunks so that recent
    memories surface first. Evergreen files are excluded from this decay —
    their scores are never penalized regardless of age.

    A file is evergreen if **any** of the following conditions hold:
    1. Its filename matches an entry in ``evergreen_patterns``
       (e.g. ``["MEMORY.md", "memory.md"]`` by default).
    2. Its filename does **not** match the dated file pattern ``YYYY-MM-DD.md``
       (reference docs, configuration files, etc. are implicitly evergreen).

    Only dated files like ``2026-03-21.md`` are subject to decay.

    Args:
        path:               Absolute path to the file being checked.
        evergreen_patterns: List of filenames that are always evergreen
                            (from ``MemoryConfig.evergreen_patterns``).

    Returns:
        ``True`` if the file should skip temporal decay, ``False`` otherwise.

    Examples::

        is_evergreen(Path("/proj/memory/MEMORY.md"), ["MEMORY.md"])
        # → True  (explicit pattern match)

        is_evergreen(Path("/proj/memory/architecture.md"), ["MEMORY.md"])
        # → True  (non-dated filename — reference doc)

        is_evergreen(Path("/proj/memory/2026-03-21.md"), ["MEMORY.md"])
        # → False (dated daily log — subject to decay)
    """
    filename = path.name

    # Step 1: Explicit evergreen patterns (e.g. "MEMORY.md")
    if filename in evergreen_patterns:
        return True

    # Step 2: Non-dated files are evergreen by convention.
    # Only YYYY-MM-DD.md files are considered time-bound.
    if not _DATED_PATTERN.match(filename):
        return True

    return False


def relative_path(path: Path, workspace_dir: Path) -> str:
    """Return ``path`` as a forward-slash string relative to ``workspace_dir``.

    Produces a portable relative path for storage in SQLite and display in
    search results. Falls back to the absolute POSIX path if ``path`` is not
    under ``workspace_dir`` (e.g. for external/extra_paths files).

    Args:
        path:          Absolute path to convert.
        workspace_dir: Base directory to make the path relative to.

    Returns:
        Relative POSIX path string, e.g. ``"memory/2026-03-21.md"``.
        Falls back to absolute POSIX path, e.g. ``"/shared/notes.md"``.

    Example::

        relative_path(Path("/proj/memory/2026-03-21.md"), Path("/proj"))
        # → "memory/2026-03-21.md"

        relative_path(Path("/other/notes.md"), Path("/proj"))
        # → "/other/notes.md"   (fallback)
    """
    try:
        return path.relative_to(workspace_dir).as_posix()
    except ValueError:
        return path.as_posix()
