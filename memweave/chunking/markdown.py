"""
memweave/chunking/markdown.py — Markdown chunking algorithm.

Markdown chunking algorithm implementation.

Algorithm overview:
    1. Split the document into lines (preserving line numbers for retrieval).
    2. For each line, if the line exceeds ``max_chars`` on its own, split it
       into fixed-size segments of ``max_chars`` characters each (sub-line
       splitting for very long lines).
    3. Accumulate segments into ``current`` buffer until adding the next
       segment would exceed ``max_chars``.
    4. When the buffer is full, ``flush`` it as a chunk (record start/end
       line numbers, join lines with ``\n``).
    5. ``carryOverlap``: instead of clearing the buffer completely, keep the
       last ``overlap_chars`` worth of lines so the next chunk starts with
       context from the previous chunk (backward overlap).
    6. After processing all lines, flush the remaining buffer.

Character budget:
    ``max_chars = max(32, tokens * 4)``
    ``overlap_chars = max(0, overlap * 4)``

    The ``* 4`` approximation comes from "1 token ≈ 4 characters" (English text).

Line accounting:
    - Lines are 1-indexed (line 1 = first line of file).
    - Sub-line segments all share the same ``lineNo``.
    - ``chunk.start_line`` = ``lineNo`` of the first entry in ``current``.
    - ``chunk.end_line``   = ``lineNo`` of the last entry in ``current``.

Result type:
    ``MarkdownChunk`` — named tuple with ``start_line``, ``end_line``, and
    ``text`` fields.

Example::

    chunks = chunk_markdown(
        "# Title\\n\\nFirst paragraph.\\n\\nSecond paragraph.",
        chunk_tokens=400,
        chunk_overlap=80,
    )
    for c in chunks:
        print(f"lines {c.start_line}-{c.end_line}: {c.text[:40]!r}")
"""

from __future__ import annotations

from typing import NamedTuple


class MarkdownChunk(NamedTuple):
    """A single chunk produced by ``chunk_markdown``.

    Attributes:
        start_line: 1-indexed first line of this chunk in the source file.
        end_line:   1-indexed last line of this chunk in the source file.
        text:       Raw chunk text (lines joined with ``\\n``).
    """

    start_line: int
    """1-indexed first line of this chunk in the source file."""

    end_line: int
    """1-indexed last line of this chunk in the source file."""

    text: str
    """Raw chunk text (lines joined with '\\n')."""


def chunk_markdown(
    content: str,
    chunk_tokens: int = 400,
    chunk_overlap: int = 80,
) -> list[MarkdownChunk]:
    """Split markdown text into overlapping chunks for embedding.

    This is a line-boundary chunker with backward overlap. It never splits
    in the middle of a line (except for lines that are longer than
    ``max_chars``, which are split into fixed-size segments).

    Args:
        content:       Full text of the markdown file.
        chunk_tokens:  Target chunk size in tokens. Converted to characters
                       via ``max(32, chunk_tokens * 4)``.
        chunk_overlap: Number of overlap tokens between consecutive chunks.
                       Converted to characters via ``max(0, chunk_overlap * 4)``.

    Returns:
        List of ``MarkdownChunk`` objects in document order.
        Returns an empty list if ``content`` is empty or whitespace-only.

    Algorithm:

    Step 1 — Character budgets::

        max_chars    = max(32, chunk_tokens * 4)
        overlap_chars = max(0, chunk_overlap * 4)

    Step 2 — Line iteration:
        For each line (1-indexed):
        - If the line is empty, treat it as a single empty segment.
        - Otherwise, split into segments of at most ``max_chars`` characters
          (to handle unusually long lines without overflowing a chunk).

    Step 3 — Buffer accumulation:
        For each segment:
        - If adding it would overflow ``max_chars`` AND the buffer is non-empty:
          * Flush the current buffer as a chunk.
          * Carry backward overlap into the new buffer.
        - Append the segment to the buffer.

    Step 4 — Final flush:
        After all lines, flush any remaining buffer content.

    Step 5 — Overlap carry-back (``carryOverlap``):
        Walk the current buffer from the end, accumulating lines until the
        accumulated character count reaches ``overlap_chars``. Keep only
        those lines as the seed for the next chunk.

    Example::

        # Short document — fits in a single chunk
        chunks = chunk_markdown("Hello world\\nSecond line.", chunk_tokens=100)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 2

        # Longer document — splits into multiple chunks with overlap
        long_text = "\\n".join(f"Line {i}" for i in range(1, 200))
        chunks = chunk_markdown(long_text, chunk_tokens=50, chunk_overlap=10)
        assert len(chunks) > 1
        # Adjacent chunks share lines (overlap)
        assert chunks[0].end_line >= chunks[1].start_line
    """
    lines = content.split("\n")
    if not lines:
        return []

    max_chars = max(32, chunk_tokens * 4)
    overlap_chars = max(0, chunk_overlap * 4)

    chunks: list[MarkdownChunk] = []

    # Buffer: list of (line_text, 1-indexed line number)
    current: list[tuple[str, int]] = []
    current_chars: int = 0

    def flush() -> None:
        """Emit the current buffer as a MarkdownChunk."""
        nonlocal current, current_chars
        if not current:
            return
        first_line_no = current[0][1]
        last_line_no = current[-1][1]
        text = "\n".join(entry[0] for entry in current)
        chunks.append(
            MarkdownChunk(
                start_line=first_line_no,
                end_line=last_line_no,
                text=text,
            )
        )

    def carry_overlap() -> None:
        """Retain the tail of the current buffer up to overlap_chars as seed for next chunk."""
        nonlocal current, current_chars
        if overlap_chars <= 0 or not current:
            current = []
            current_chars = 0
            return

        # Walk backwards accumulating lines until overlap budget is reached
        acc = 0
        kept: list[tuple[str, int]] = []
        for entry in reversed(current):
            acc += len(entry[0]) + 1  # +1 for the '\n' separator
            kept.insert(0, entry)
            if acc >= overlap_chars:
                break

        current = kept
        current_chars = sum(len(e[0]) + 1 for e in kept)

    for i, raw_line in enumerate(lines):
        line_no = i + 1  # 1-indexed

        # Sub-line segmentation: split very long lines into max_chars pieces
        if len(raw_line) == 0:
            segments = [""]
        else:
            segments = [
                raw_line[start : start + max_chars] for start in range(0, len(raw_line), max_chars)
            ]

        for segment in segments:
            # Each segment contributes (len + 1) chars to the buffer
            # (+1 accounts for the '\n' that join() will add between lines)
            line_size = len(segment) + 1

            # Flush when adding this segment would overflow the budget
            if current_chars + line_size > max_chars and current:
                flush()
                carry_overlap()

            current.append((segment, line_no))
            current_chars += line_size

    # Final flush — any remaining lines in the buffer
    flush()
    return chunks


def chunk_text(
    content: str,
    chunk_tokens: int = 400,
    chunk_overlap: int = 80,
) -> list[str]:
    """Convenience wrapper — return chunk texts only (no line numbers).

    Useful when line-number metadata is not needed (e.g. for ``mem.add()``
    when content is provided as a raw string without a source file).

    Args:
        content:       Text to chunk.
        chunk_tokens:  Target chunk size in tokens.
        chunk_overlap: Overlap size in tokens.

    Returns:
        List of chunk text strings.
    """
    return [c.text for c in chunk_markdown(content, chunk_tokens, chunk_overlap)]
