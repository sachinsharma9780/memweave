"""memweave — Async-first memory library for multi-agent AI systems."""

import logging

# Prevent "No handlers could be found" warnings for library users who don't
# configure logging themselves.
logging.getLogger("memweave").addHandler(logging.NullHandler())


def enable_logging(level: int = logging.DEBUG) -> None:
    """Enable memweave log output to stderr.

    Call once at application startup to see library logs::

        import memweave
        memweave.enable_logging()              # DEBUG and above
        memweave.enable_logging(logging.INFO)  # INFO and above only

    Args:
        level: Minimum log level to show (default: ``logging.DEBUG``).
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger("memweave")
    root.setLevel(level)
    if not root.handlers or all(isinstance(h, logging.NullHandler) for h in root.handlers):
        root.addHandler(handler)


# Logging must be configured before importing submodules so that handlers
# are in place before any module-level code in those submodules runs.
# ruff: noqa: E402
from importlib.metadata import version

__version__ = version("memweave")
from memweave.config import (
    CacheConfig,
    ChunkingConfig,
    EmbeddingConfig,
    FlushConfig,
    HybridConfig,
    MemoryConfig,
    MMRConfig,
    QueryConfig,
    SyncConfig,
    TemporalDecayConfig,
    VectorConfig,
)
from memweave.exceptions import (
    ConfigError,
    EmbeddingError,
    FlushError,
    IndexError,
    MemWeaveError,
    SearchError,
    StorageError,
    StrategyError,
)
from memweave.store import MemWeave
from memweave.types import (
    FileInfo,
    IndexResult,
    SearchResult,
    StoreStatus,
)

__all__ = [
    # Version
    "__version__",
    # Primary class
    "MemWeave",
    # Config
    "MemoryConfig",
    "EmbeddingConfig",
    "ChunkingConfig",
    "QueryConfig",
    "HybridConfig",
    "MMRConfig",
    "TemporalDecayConfig",
    "CacheConfig",
    "SyncConfig",
    "FlushConfig",
    "VectorConfig",
    # Result types
    "SearchResult",
    "IndexResult",
    "FileInfo",
    "StoreStatus",
    # Exceptions
    "MemWeaveError",
    "ConfigError",
    "StorageError",
    "IndexError",
    "SearchError",
    "StrategyError",
    "EmbeddingError",
    "FlushError",
    # Logging
    "enable_logging",
]
