"""Tests for authoritative conversation synchronization."""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import server_fastmcp
from conversation_memory import ConversationMemoryServer
from exceptions import ValidationError


def _record(*, content: str = "## User\n\nHello", title: str = "Llama.cpp: test") -> dict:
    return {
        "content": content,
        "conversation_type": "chat",
        "custom_fields": {
            "sync_source": "llama.cpp",
            "sync_source_id": "source-id",
        },
        "date": "2026-08-15T00:00:00",
        "session_id": "llama.cpp:source-id",
        "tags": ["source:llama.cpp"],
        "title": title,
    }


@pytest.mark.asyncio
async def test_sync_add_update_and_repeat_are_idempotent(tmp_path: Path) -> None:
    server = ConversationMemoryServer(str(tmp_path), use_data_dir=True)

    first = await server.sync_conversations([_record()])
    assert first == {
        "status": "success",
        "added": 1,
        "updated": 0,
        "unchanged": 0,
        "errors": [],
    }

    repeated = await server.sync_conversations([_record()])
    assert repeated["unchanged"] == 1
    assert len(list(server.conversations_path.rglob("conv_*.json"))) == 1

    changed = await server.sync_conversations(
        [_record(content="## User\n\nChanged", title="Llama.cpp: changed")]
    )
    assert changed["updated"] == 1
    stored_path = next(server.conversations_path.rglob("conv_*.json"))
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    assert stored["title"] == "Llama.cpp: changed"
    assert stored["content"] == "## User\n\nChanged"
    assert not stored["content"].startswith("[update ")


@pytest.mark.asyncio
async def test_concurrent_repeated_batches_do_not_create_duplicates(tmp_path: Path) -> None:
    server = ConversationMemoryServer(str(tmp_path), use_data_dir=True)

    first, second = await asyncio.gather(
        server.sync_conversations([_record()]),
        server.sync_conversations([_record()]),
    )

    assert sorted((first["added"], second["added"])) == [0, 1]
    assert sorted((first["unchanged"], second["unchanged"])) == [0, 1]
    assert len(list(server.conversations_path.rglob("conv_*.json"))) == 1


@pytest.mark.asyncio
async def test_sync_refuses_ambiguous_stored_session(tmp_path: Path) -> None:
    server = ConversationMemoryServer(str(tmp_path), use_data_dir=True)
    for title in ("First", "Second"):
        added = await server.add_conversation(
            "content",
            title,
            "2026-08-15T00:00:00",
            session_id="llama.cpp:source-id",
        )
        assert added["status"] == "success"

    result = await server.sync_conversations([_record()])
    assert result["status"] == "error"
    assert result["added"] == result["updated"] == result["unchanged"] == 0
    assert result["errors"][0]["session_id"] == "llama.cpp:source-id"
    assert "Multiple" in result["errors"][0]["message"]


def test_sync_batch_validation_preserves_authoritative_title() -> None:
    normalized = server_fastmcp._normalize_sync_batch(
        "llama.cpp",
        [
            {
                "source_id": "source-id",
                "title": "Llama.cpp: cache / identity?",
                "content": "content",
                "date": "2026-08-15T00:00:00Z",
            }
        ],
    )
    assert normalized[0]["session_id"] == "llama.cpp:source-id"
    assert normalized[0]["title"] == "Llama.cpp: cache / identity?"
    assert normalized[0]["tags"] == ["source:llama.cpp"]

    with pytest.raises(ValidationError, match="source must start"):
        server_fastmcp._normalize_sync_batch("bad source", [])
    with pytest.raises(ValidationError, match="source_id"):
        server_fastmcp._normalize_sync_batch(
            "llama.cpp", [{"source_id": "", "title": "title", "content": "content"}]
        )


class _FakeSyncMemory:
    def __init__(self) -> None:
        self.received: list[dict] | None = None

    async def sync_conversations(self, conversations: list[dict]) -> dict:
        self.received = conversations
        return {
            "status": "success",
            "added": 1,
            "updated": 0,
            "unchanged": 0,
            "errors": [],
        }


@pytest.mark.asyncio
async def test_fastmcp_sync_tool_returns_machine_readable_text() -> None:
    fake = _FakeSyncMemory()
    with patch.object(server_fastmcp, "memory_server", fake):
        response = await server_fastmcp.sync_conversations(
            "llama.cpp",
            [{"source_id": "source-id", "title": "Title", "content": "Content"}],
        )

    decoded = json.loads(response)
    assert decoded["status"] == "success"
    assert decoded["added"] == 1
    assert fake.received is not None
    assert fake.received[0]["session_id"] == "llama.cpp:source-id"

    invalid = await server_fastmcp.sync_conversations("bad source", [])
    assert json.loads(invalid)["status"] == "error"
