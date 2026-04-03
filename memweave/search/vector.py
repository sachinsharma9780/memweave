"""
memweave/search/vector.py — sqlite-vec cosine similarity search.

sqlite-vec cosine similarity search.

sqlite-vec is an optional dependency (``pip install memweave[vector]``).
If the extension is not loaded in the DB connection, this module raises
``VectorSearchUnavailableError`` at search time rather than silently
returning empty results.
"""

from __future__ import annotations

import struct

import aiosqlite

from memweave.search.strategy import RawSearchRow


class VectorSearchUnavailableError(RuntimeError):
    """Raised when sqlite-vec is not loaded in the database connection.

    Install the optional extra and load the extension before searching::

        pip install memweave[vector]

    Then ensure your DB setup calls::

        import sqlite_vec
        sqlite_vec.load(db)

    or passes a custom extension path in ``VectorConfig``.
    """


def _vec_to_blob(vec: list[float]) -> bytes:
    """Serialize a list of float32 values to a little-endian binary blob.

    sqlite-vec stores and queries embeddings as raw ``FLOAT[N]`` blobs
    (IEEE 754 float32, little-endian).  This matches the format produced by
    ``sqlite_vec.serialize_float32()`` in the Python sqlite-vec package.

    Args:
        vec: L2-normalized embedding vector.

    Returns:
        Packed binary blob (4 bytes per dimension).
    """
    return struct.pack(f"{len(vec)}f", *vec)


class VectorSearch:
    """sqlite-vec cosine similarity search backend.

    Uses the ``chunks_vec`` virtual table (created by ``ensure_schema()``)
    and sqlite-vec's ``vec_distance_cosine`` SQL function.

    The query::

        SELECT c.id, c.path, c.source, c.start_line, c.end_line, c.text,
               vec_distance_cosine(v.embedding, ?) AS dist
          FROM chunks_vec v
          JOIN chunks c ON c.id = v.id
         WHERE c.model = ?
           [AND c.source = ?]
         ORDER BY dist ASC
         LIMIT ?

    Distance is converted to similarity as ``score = 1 - dist`` (cosine
    distance of 0 = identical vectors → similarity 1.0).

    Usage::

        vs = VectorSearch()
        rows = await vs.search(db, "", query_vec, model, limit=20)

    .. note::
        ``query_vec`` **must** be L2-normalized before passing here.
        :class:`~memweave.embedding.provider.LiteLLMEmbeddingProvider`
        normalizes its output automatically.
    """

    async def search(
        self,
        db: aiosqlite.Connection,
        query: str,  # noqa: ARG002 — unused, satisfies SearchStrategy protocol
        query_vec: list[float] | None,
        model: str,
        limit: int,
        *,
        source_filter: str | None = None,
    ) -> list[RawSearchRow]:
        """Run a cosine similarity search using sqlite-vec.

        Steps:
        1. Serialize ``query_vec`` to a float32 blob for sqlite-vec.
        2. Execute the ``vec_distance_cosine`` query against ``chunks_vec``.
        3. Convert each distance to a similarity score: ``1 - dist``.
        4. Return rows ordered by descending similarity.

        Args:
            db:            Open aiosqlite connection with sqlite-vec loaded.
            query:         Ignored (vector search uses ``query_vec``).
            query_vec:     L2-normalized query embedding.  Must not be ``None``.
            model:         Embedding model name to filter chunks.
            limit:         Maximum rows to return.
            source_filter: ``"memory"`` or ``"sessions"`` to restrict results.

        Returns:
            List of :class:`~memweave.search.strategy.RawSearchRow`, ordered by
            descending similarity score.

        Raises:
            ValueError: If ``query_vec`` is ``None``.
            VectorSearchUnavailableError: If sqlite-vec is not loaded in ``db``.
        """
        if query_vec is None:
            raise ValueError("VectorSearch requires a query_vec (got None)")

        # Verify sqlite-vec is available by probing a known function
        try:
            await db.execute("SELECT vec_version()")
        except Exception as exc:
            raise VectorSearchUnavailableError(
                "sqlite-vec is not loaded. Install memweave[vector] and load "
                "the extension before searching."
            ) from exc

        blob = _vec_to_blob(query_vec)

        source_clause = " AND c.source = ?" if source_filter else ""
        params: list[object] = [blob, model]
        if source_filter:
            params.append(source_filter)
        params.append(limit)

        sql = (
            "SELECT c.id, c.path, c.source, c.start_line, c.end_line, c.text,"
            "       vec_distance_cosine(v.embedding, ?) AS dist"
            "  FROM chunks_vec v"
            "  JOIN chunks c ON c.id = v.id"
            " WHERE c.model = ?"
            f"{source_clause}"
            " ORDER BY dist ASC"
            " LIMIT ?"
        )

        rows: list[RawSearchRow] = []
        async with db.execute(sql, params) as cursor:
            async for row in cursor:
                chunk_id, path, source, start_line, end_line, text, dist = row
                score = 1.0 - float(dist)
                rows.append(
                    RawSearchRow(
                        chunk_id=chunk_id,
                        path=path,
                        source=source,
                        start_line=start_line,
                        end_line=end_line,
                        text=text,
                        score=score,
                        vector_score=score,
                    )
                )
        return rows
