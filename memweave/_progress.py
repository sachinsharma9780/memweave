"""
memweave/_progress.py — Human-readable progress output.

Prints a single line per notable event during library operations.
Controlled by MemoryConfig.progress (default True).

Format:
    <emoji> [memweave] <method>()  →  <message>

Usage inside store.py:
    from memweave._progress import emit

    emit(self.config.progress, "📂", "index", "scanning workspace...")
"""

from __future__ import annotations

import sys

# ── Emoji constants ───────────────────────────────────────────────────────────

# index()
EMOJI_INDEX_SCAN = "📂"  # scanning files on disk
EMOJI_INDEX_FILE = "🗂️"  # processing a single file
EMOJI_INDEX_DONE = "✅"  # index complete

# add()
EMOJI_ADD = "➕"  # adding a single file

# search()
EMOJI_SEARCH = "🔍"  # starting a search
EMOJI_SEARCH_KW = "🔤"  # keyword / FTS strategy
EMOJI_SEARCH_VEC = "🧬"  # vector strategy
EMOJI_SEARCH_HYBRID = "⚗️"  # hybrid strategy
EMOJI_SEARCH_DONE = "✅"  # search complete

# Post-processors
EMOJI_DECAY = "⏳"  # temporal decay applied
EMOJI_MMR = "🎯"  # MMR reranking applied

# Embedding
EMOJI_EMBED_API = "⚡"  # embedding via API call
EMOJI_EMBED_CACHE = "💡"  # embedding restored from cache

# flush()
EMOJI_FLUSH_EXTRACT = "🧠"  # extracting memories from conversation
EMOJI_FLUSH_WRITE = "💾"  # writing to memory file
EMOJI_FLUSH_DONE = "✅"  # flush complete

# status() / files()
EMOJI_STATUS = "📊"  # status snapshot
EMOJI_FILES = "📁"  # listing files

# open / close
EMOJI_OPEN = "🗂️"  # opening database
EMOJI_CLOSE = "🔒"  # closing database

# warnings / skipped
EMOJI_WARN = "⚠️"  # soft warning


# ── Emitter ───────────────────────────────────────────────────────────────────


def emit(enabled: bool, emoji: str, method: str, message: str) -> None:
    """Print one progress line to stdout.

    Args:
        enabled: When False this is a no-op (progress=False on config).
        emoji:   Single emoji character indicating the operation type.
        method:  The public API method name, e.g. ``"index"``.
        message: Free-form status text, e.g. ``"scanning 12 files..."``.
    """
    if not enabled:
        return
    print(f"{emoji} [memweave] {method}()  →  {message}", flush=True, file=sys.stdout)
