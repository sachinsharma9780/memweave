"""
tests/unit/test_flush.py — Unit tests for flush/memory_flush.py.

Tests cover:
- flush_conversation: silent reply sentinel → returns None
- flush_conversation: normal extraction → appends to dated file
- flush_conversation: creates memory/ dir if missing
- flush_conversation: appends to existing file (never overwrites)
- flush_conversation: LLM error → raises FlushError
- flush_conversation: custom model/system_prompt override
- flush disabled config: MemWeave.flush returns None without calling LLM
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memweave import MemWeave, MemoryConfig
from memweave.config import FlushConfig
from memweave.exceptions import FlushError
from memweave.flush.memory_flush import flush_conversation


def _make_config(tmp_path: Path, *, flush_enabled: bool = True) -> MemoryConfig:
    return MemoryConfig(
        workspace_dir=tmp_path,
        flush=FlushConfig(enabled=flush_enabled),
    )


def _make_litellm_response(content: str) -> MagicMock:
    """Build a fake litellm response object."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ── flush_conversation ────────────────────────────────────────────────────────


class TestFlushConversation:
    async def test_silent_reply_returns_none(self, tmp_path: Path):
        """LLM returning @@SILENT_REPLY@@ → returns None, no file written."""
        config = _make_config(tmp_path)
        conversation = [{"role": "user", "content": "hello"}]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_litellm_response("@@SILENT_REPLY@@")
            result = await flush_conversation(conversation, config)

        assert result is None
        dated_file = tmp_path / "memory" / f"{date.today().isoformat()}.md"
        assert not dated_file.exists()

    async def test_extracted_content_written_to_file(self, tmp_path: Path):
        """LLM returning real content → written to dated memory file."""
        config = _make_config(tmp_path)
        conversation = [
            {"role": "user", "content": "We decided to use PostgreSQL."},
            {"role": "assistant", "content": "Good choice."},
        ]
        extracted = "Decision: use PostgreSQL for the main database."

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_litellm_response(extracted)
            result = await flush_conversation(conversation, config)

        assert result == extracted
        dated_file = tmp_path / "memory" / f"{date.today().isoformat()}.md"
        assert dated_file.exists()
        content = dated_file.read_text(encoding="utf-8")
        assert extracted in content

    async def test_creates_memory_dir_if_missing(self, tmp_path: Path):
        """memory/ directory is created if it doesn't exist."""
        config = _make_config(tmp_path)
        memory_dir = tmp_path / "memory"
        assert not memory_dir.exists()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_litellm_response("New fact: X.")
            await flush_conversation([], config)

        assert memory_dir.exists()

    async def test_appends_to_existing_file(self, tmp_path: Path):
        """Calling flush twice appends to the same dated file."""
        config = _make_config(tmp_path)
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        dated_file = memory_dir / f"{date.today().isoformat()}.md"
        dated_file.write_text("Existing content.\n", encoding="utf-8")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_litellm_response("New fact.")
            await flush_conversation([], config)

        content = dated_file.read_text(encoding="utf-8")
        assert "Existing content." in content
        assert "New fact." in content

    async def test_never_overwrites_existing_content(self, tmp_path: Path):
        """Existing file content must always be preserved."""
        config = _make_config(tmp_path)
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        dated_file = memory_dir / f"{date.today().isoformat()}.md"
        original = "Do NOT delete this line.\n"
        dated_file.write_text(original, encoding="utf-8")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_litellm_response("Appended content.")
            await flush_conversation([], config)

        content = dated_file.read_text(encoding="utf-8")
        assert "Do NOT delete this line." in content

    async def test_llm_error_raises_flush_error(self, tmp_path: Path):
        """LLM API failure → raises FlushError."""
        config = _make_config(tmp_path)

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("API timeout")
            with pytest.raises(FlushError, match="LLM extraction failed"):
                await flush_conversation([], config)

    async def test_custom_model_override(self, tmp_path: Path):
        """model= kwarg is passed to litellm.acompletion."""
        config = _make_config(tmp_path)

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_litellm_response("Fact.")
            await flush_conversation([], config, model="gpt-4o")

        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs.get("model") == "gpt-4o"

    async def test_custom_system_prompt_override(self, tmp_path: Path):
        """system_prompt= kwarg is used as the system message content."""
        config = _make_config(tmp_path)
        custom_prompt = "Extract only code decisions."

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_litellm_response("Code decision: X.")
            await flush_conversation([], config, system_prompt=custom_prompt)

        messages = mock_llm.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == custom_prompt

    async def test_empty_llm_response_returns_none(self, tmp_path: Path):
        """Empty string from LLM → returns None."""
        config = _make_config(tmp_path)

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_litellm_response("   ")
            result = await flush_conversation([], config)

        assert result is None

    # ── BUG-001: date injection ──────────────────────────────────────────────

    async def test_default_system_prompt_date_injected(self, tmp_path: Path):
        """Default system prompt: YYYY-MM-DD replaced with today before LLM call."""
        config = _make_config(tmp_path)

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_litellm_response("Fact.")
            await flush_conversation([], config)

        system_content = mock_llm.call_args[1]["messages"][0]["content"]
        assert date.today().isoformat() in system_content
        assert "YYYY-MM-DD" not in system_content

    async def test_custom_system_prompt_date_injected(self, tmp_path: Path):
        """Custom system_prompt= containing YYYY-MM-DD: token is still replaced."""
        config = _make_config(tmp_path)
        custom = "Store facts in memory/YYYY-MM-DD.md. If nothing, @@SILENT_REPLY@@."

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _make_litellm_response("Fact.")
            await flush_conversation([], config, system_prompt=custom)

        system_content = mock_llm.call_args[1]["messages"][0]["content"]
        assert date.today().isoformat() in system_content
        assert "YYYY-MM-DD" not in system_content


# ── MemWeave.flush() wrapper ─────────────────────────────────────────────────


class TestMemWeaveFlush:
    async def test_flush_disabled_returns_none(self, tmp_path: Path):
        """flush.enabled=False → MemWeave.flush() returns None without calling LLM."""
        from memweave.config import EmbeddingConfig

        cfg = MemoryConfig(
            workspace_dir=tmp_path,
            embedding=EmbeddingConfig(model="test-embed"),
            flush=FlushConfig(enabled=False),
        )

        class FakeProvider:
            async def embed_query(self, text):
                return [0.1] * 8

            async def embed_batch(self, texts):
                return [[0.1] * 8 for _ in texts]

        async with MemWeave(cfg, embedding_provider=FakeProvider()) as mem:
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
                result = await mem.flush([])
                mock_llm.assert_not_called()
            assert result is None
