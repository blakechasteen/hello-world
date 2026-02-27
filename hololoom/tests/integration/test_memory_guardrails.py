from __future__ import annotations

import asyncio
from datetime import datetime

import pytest  # type: ignore[import]

from HoloLoom.alignment.safety_guardrails import SafetyGuardrails
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.memory.protocol import Memory, MemoryQuery


def run_sync(awaitable):
    """Execute an awaitable to completion for synchronous tests."""
    return asyncio.run(awaitable)


@pytest.mark.parametrize("memory_backend", [MemoryBackend.INMEMORY, MemoryBackend.HYBRID])
def test_guardrails_metadata_on_recall(memory_backend: MemoryBackend) -> None:
    guardrails = SafetyGuardrails()
    config = Config()
    config.memory_backend = memory_backend

    store = run_sync(create_memory_backend(config=config, guardrails=guardrails))

    memory = Memory(
        id=f"{memory_backend.value}-memory",
        text="Test memory about the MythRL shuttle architecture",
        timestamp=datetime.utcnow(),
        context={"entities": ["MythRL"]},
        metadata={"source": "unit-test"},
    )

    import asyncio

    run_sync(store.store(memory, user_id="tester"))

    query = MemoryQuery(text="What does MythRL store?", user_id="tester", limit=1)
    result = run_sync(store.recall(query, limit=1))

    guardrail_meta = result.metadata.get("guardrails") if result.metadata else None
    assert guardrail_meta is not None, "Guardrail metadata missing from recall result"
    assert guardrail_meta.get("risk_level") in {"safe", "low", "medium", "high", "critical"}

    actions = {request.action for request in guardrails.action_history}
    assert "memory_store" in actions
    assert "memory_recall" in actions


def test_guardrails_block_delete_without_approval() -> None:
    guardrails = SafetyGuardrails()
    config = Config()
    config.memory_backend = MemoryBackend.INMEMORY

    store = run_sync(create_memory_backend(config=config, guardrails=guardrails))

    memory = Memory(
        id="delete-check",
        text="Memory slated for deletion",
        timestamp=datetime.utcnow(),
        context={},
        metadata={},
    )

    import asyncio

    run_sync(store.store(memory))

    with pytest.raises(PermissionError):
        run_sync(store.delete(memory.id))

    assert any(request.action == "memory_delete" for request in guardrails.action_history)
