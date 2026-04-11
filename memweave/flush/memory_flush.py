"""
memweave/flush/memory_flush.py — LLM-driven fact extraction and memory flush.

LLM-driven fact extraction and memory flush logic.
hook. Called by ``MemWeave.flush()`` just before a context window compaction
event to extract durable facts from a conversation and persist them to the
dated memory file (``memory/YYYY-MM-DD.md``).

The flush pipeline:
1. Build the LLM request: system prompt + conversation messages.
2. Call the LLM via LiteLLM (non-streaming).
3. If the model replies with ``@@SILENT_REPLY@@``, there is nothing to store
   — return ``None`` immediately.
4. Otherwise, append the extracted content to ``memory/YYYY-MM-DD.md``
   (create the file if it doesn't exist; append-only, never overwrite).
5. Return the extracted text so the caller can re-index the file.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from memweave.config import MemoryConfig
from memweave.exceptions import FlushError

logger = logging.getLogger(__name__)

_SILENT_REPLY_SENTINEL = "@@SILENT_REPLY@@"


async def flush_conversation(
    conversation: list[dict[str, str]],
    config: MemoryConfig,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
) -> str | None:
    """Extract durable facts from a conversation and append to the dated memory file.

    Sends the conversation to an LLM with a structured extraction system prompt.
    If the model returns ``@@SILENT_REPLY@@``, returns ``None`` (nothing stored).
    Otherwise appends the extracted text to ``memory/YYYY-MM-DD.md`` and returns it.

    Args:
        conversation:  List of ``{"role": "user"|"assistant", "content": "..."}``
                       message dicts representing the conversation to compress.
        config:        Active ``MemoryConfig``.
        model:         Override ``FlushConfig.model`` for this call.
        system_prompt: Override ``FlushConfig.system_prompt`` for this call.

    Returns:
        Extracted text string, or ``None`` if the LLM found nothing to store.

    Raises:
        FlushError: On LLM API failure or filesystem write error.
    """
    flush_cfg = config.flush
    effective_model = model or flush_cfg.model
    effective_system = system_prompt or flush_cfg.system_prompt

    # Inject today's date before sending to the LLM so headings use the real date,
    # not an invented one.  Replaces the literal placeholder used in the default prompt.
    today = date.today()
    effective_system = effective_system.replace("YYYY-MM-DD", today.isoformat())

    messages = [
        {"role": "system", "content": effective_system},
        *conversation,
    ]

    try:
        import litellm

        response = await litellm.acompletion(
            model=effective_model,
            messages=messages,
            max_tokens=flush_cfg.max_tokens,
            temperature=flush_cfg.temperature,
        )
        extracted: str = response.choices[0].message.content or ""
    except Exception as exc:
        raise FlushError(f"LLM extraction failed (model={effective_model!r}): {exc}") from exc

    extracted = extracted.strip()

    if not extracted or _SILENT_REPLY_SENTINEL in extracted:
        logger.debug("Flush: LLM replied with silent sentinel — nothing to store.")
        return None

    # Determine the dated memory file path  (today already computed above)
    workspace = Path(config.workspace_dir)
    memory_dir = workspace / "memory"
    dated_file = memory_dir / f"{today.isoformat()}.md"

    try:
        memory_dir.mkdir(parents=True, exist_ok=True)
        if dated_file.exists():
            # Append only — never overwrite existing content
            existing = dated_file.read_text(encoding="utf-8")
            separator = "\n\n" if existing and not existing.endswith("\n\n") else ""
            dated_file.write_text(
                existing + separator + extracted + "\n",
                encoding="utf-8",
            )
        else:
            dated_file.write_text(extracted + "\n", encoding="utf-8")
    except OSError as exc:
        raise FlushError(f"Failed to write memory file {dated_file}: {exc}") from exc

    logger.info("Flush: wrote %d chars to %s", len(extracted), dated_file)
    return extracted
