"""
memweave/embedding/provider.py — Embedding provider protocol and LiteLLM adapter.

Defines the ``EmbeddingProvider`` Protocol so the rest of the library talks to
a stable interface, not directly to LiteLLM. Users can swap in any conforming
implementation (custom HTTP client, mock for testing, etc.) without changing
the indexing or search code.

``LiteLLMEmbeddingProvider`` is the production implementation. Key features:
- Supports 100+ models via LiteLLM (OpenAI, Cohere, Azure, Bedrock, Ollama, …).
- Batches texts automatically to respect ``EmbeddingConfig.batch_size``.
- Retries on transient errors (rate limit 429, server errors 5xx) with
  exponential back-off: 0.5 s → 2 s → 8 s (max 3 attempts).
- Fails fast on permanent errors (auth 401, bad request 400).
- Normalizes returned vectors to unit length for consistent cosine similarity.

Protocol:
    EmbeddingProvider
      ├── async embed_query(text: str) -> list[float]
      └── async embed_batch(texts: list[str]) -> list[list[float]]
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

from memweave.config import EmbeddingConfig
from memweave.embedding.vectors import normalize_embedding
from memweave.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# Maximum retry attempts for transient API failures
_MAX_RETRIES = 3
# Back-off delays between retries (seconds)
_BACKOFF_DELAYS = [0.5, 2.0, 8.0]
# HTTP status codes that are safe to retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Any object with ``embed_query`` and ``embed_batch`` methods conforms to
    this protocol, regardless of inheritance. This means:
    - ``isinstance(obj, EmbeddingProvider)`` works at runtime (due to
      ``@runtime_checkable``).
    - Users can pass a mock object in tests without subclassing.
    - Custom providers just need to implement the two methods.

    All returned vectors are expected to be L2-normalized to unit length.

    Example — custom provider::

        class MyProvider:
            async def embed_query(self, text: str) -> list[float]:
                vector = my_model.encode(text)
                return normalize_embedding(vector.tolist())

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                vectors = my_model.encode(texts)
                return [normalize_embedding(v.tolist()) for v in vectors]

        mem = MemWeave(config, embedding_provider=MyProvider())
    """

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query text to embed.

        Returns:
            L2-normalized embedding vector as a list of floats.

        Raises:
            EmbeddingError: On API failure.
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one or more API calls.

        The provider is responsible for batching internally (respecting
        ``batch_size``). Callers pass the full list; the provider handles
        chunking.

        Args:
            texts: List of strings to embed (may be empty).

        Returns:
            List of L2-normalized vectors, one per input text, in the same
            order. Empty list if ``texts`` is empty.

        Raises:
            EmbeddingError: On API failure.
        """
        ...


class LiteLLMEmbeddingProvider:
    """Production embedding provider backed by LiteLLM.

    LiteLLM wraps 100+ embedding providers under a single API:
    - OpenAI: ``"text-embedding-3-small"``
    - Cohere: ``"cohere/embed-english-v3.0"``
    - Azure: ``"azure/deployment-name"``
    - Ollama: ``"ollama/nomic-embed-text"``
    - Google: ``"gemini/text-embedding-004"``
    - … and many more.

    Retry behavior:
        - Retryable errors (429, 5xx): up to ``_MAX_RETRIES`` attempts with
          exponential back-off (0.5 s, 2 s, 8 s).
        - Non-retryable errors (401, 400): raises ``EmbeddingError`` immediately.

    Normalization:
        All returned vectors are L2-normalized to unit length. This ensures
        that cosine similarity == dot product, simplifying downstream math.

    Example::

        config = EmbeddingConfig(
            model="text-embedding-3-small",
            batch_size=64,
            timeout=60.0,
        )
        provider = LiteLLMEmbeddingProvider(config)
        vec = await provider.embed_query("What is the deployment process?")
        # vec is a unit-normalized list of floats (e.g. length 1536)
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialise the provider from an ``EmbeddingConfig``.

        Args:
            config: Embedding configuration (model, api_base, api_key,
                    timeout, batch_size).
        """
        self._config = config
        self._model = config.model
        self._api_base = config.api_base
        self._api_key = config.api_key
        self._timeout = config.timeout
        self._batch_size = config.batch_size

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Internally calls ``embed_batch([text])`` and returns the first result.

        Args:
            text: Query text. Should not be empty (most providers reject
                  empty strings).

        Returns:
            L2-normalized embedding vector.

        Raises:
            EmbeddingError: If the text is empty or the API call fails
                            after all retries.

        Example::

            vec = await provider.embed_query("database migration steps")
            # → [0.023, -0.112, 0.441, ...]  (unit-normalized, length 1536)
        """
        if not text.strip():
            raise EmbeddingError("embed_query received empty text")
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, respecting ``batch_size``.

        Steps:
        1. Return early if ``texts`` is empty.
        2. Split ``texts`` into batches of at most ``batch_size`` items.
        3. For each batch, call ``_embed_one_batch`` (with retry logic).
        4. Flatten and return results in input order.

        Args:
            texts: Texts to embed. Empty strings are passed through but
                   most providers will return zero vectors for them.

        Returns:
            List of L2-normalized vectors in the same order as ``texts``.

        Raises:
            EmbeddingError: If any batch fails after all retries.

        Example::

            vecs = await provider.embed_batch([
                "We chose PostgreSQL for JSONB support.",
                "Redis is used for caching sessions.",
            ])
            # → [[0.05, ...], [0.12, ...]]
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        # Split into batches of at most batch_size
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = await self._embed_one_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with retry logic.

        Retry policy:
        - Up to ``_MAX_RETRIES`` total attempts.
        - Retry on retryable errors (429, 500-504) with exponential back-off.
        - Fail immediately on auth/bad-request errors.

        Steps:
        1. Build kwargs from config (model, api_base, api_key, timeout, dimensions).
        2. Call ``litellm.aembedding(model=..., input=batch, **kwargs)``.
        3. Extract ``response.data[i]["embedding"]`` for each item.
        4. L2-normalize each vector.
        5. On failure, check if retryable and sleep before retry.

        Args:
            texts: Batch of texts (len <= batch_size).

        Returns:
            List of L2-normalized vectors.

        Raises:
            EmbeddingError: After all retries exhausted, or on non-retryable error.
        """
        import litellm

        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": texts,
            "timeout": self._timeout,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await litellm.aembedding(**kwargs)
                raw_vectors = [item["embedding"] for item in response.data]
                # L2-normalize all vectors before returning
                return [normalize_embedding(v) for v in raw_vectors]

            except Exception as exc:
                last_exc = exc
                # Determine if the error is retryable
                status = _get_status_code(exc)
                if status is not None and status not in _RETRYABLE_STATUS_CODES:
                    # Non-retryable (auth error, bad request, etc.)
                    raise EmbeddingError(
                        f"Embedding API error (non-retryable, status={status}): {exc}"
                    ) from exc

                if attempt < _MAX_RETRIES - 1:
                    delay = _BACKOFF_DELAYS[attempt]
                    logger.warning(
                        "Embedding attempt %d/%d failed (%s), retrying in %.1fs…",
                        attempt + 1,
                        _MAX_RETRIES,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Embedding failed after %d attempts: %s",
                        _MAX_RETRIES,
                        exc,
                    )

        raise EmbeddingError(
            f"Embedding failed after {_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    @property
    def model(self) -> str:
        """Return the configured embedding model name."""
        return self._model


def _get_status_code(exc: Exception) -> int | None:
    """Extract an HTTP status code from a LiteLLM exception if present.

    LiteLLM wraps provider errors in custom exception classes that store the
    HTTP status code in ``exc.status_code`` (an int attribute). This helper
    safely extracts it, returning ``None`` if not found.

    Args:
        exc: Any exception (LiteLLM or otherwise).

    Returns:
        HTTP status code int, or ``None`` if the exception has none.
    """
    return getattr(exc, "status_code", None)
