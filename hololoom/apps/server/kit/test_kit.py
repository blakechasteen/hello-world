"""
Agent Kit — proof it works.

Run: pytest hololoom/apps/server/kit/test_kit.py -v
"""
from __future__ import annotations

import pytest

from .draft import Agent, Draft, loop, named, think
from .introspect import NullMemory, consult, tabula_rasa
from .models import ChatRequest
from .testing import mock

# ============================================================================
# Helpers
# ============================================================================

def _req(text="hello", room_id="test"):
    return ChatRequest(text=text, room_id=room_id)


# ============================================================================
# Agent basics
# ============================================================================

@pytest.mark.asyncio
async def test_single_pass_agent():
    """Simplest agent: one backend, one think pass, no memory."""
    backend = mock(["The answer is 42."])
    agent = Agent(soul="You are a test agent.", memory=NullMemory(), backend=backend, name="test")

    resp = await agent(_req("What is the answer?"))

    assert resp.response == "The answer is 42."
    assert resp.model == "mock"
    assert resp.tokens > 0
    assert resp.duration_ms >= 0
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_agent_repr():
    """Agent prints cleanly."""
    agent = Agent(soul="Test.", memory=NullMemory(), backend=mock(), name="elle")
    assert repr(agent) == "Agent(elle, 0t, mock, 1p)"


@pytest.mark.asyncio
async def test_callable_soul():
    """Callable souls are evaluated per-request."""
    call_count = 0

    def dynamic_soul():
        nonlocal call_count
        call_count += 1
        return f"You are soul version {call_count}."

    backend = mock(["ok", "ok"])
    agent = Agent(soul=dynamic_soul, memory=NullMemory(), backend=backend, name="test")

    await agent(_req("first"))
    await agent(_req("second"))

    assert call_count == 2
    # Second call should have "version 2" in the system message
    last_system = backend.last_messages[0]["content"]
    assert "version 2" in last_system


# ============================================================================
# Tracing
# ============================================================================

@pytest.mark.asyncio
async def test_compose_traces_passes():
    """Every pass in the pipeline is traced with name and timing."""
    backend = mock(["Response."])
    agent = Agent(soul="Test.", memory=NullMemory(), backend=backend, name="test")

    resp = await agent(_req())
    trace = resp.metadata.get("trace", [])

    assert len(trace) == 1
    assert trace[0]["pass"] == "think(mock)"
    assert trace[0]["ms"] >= 0
    assert "error" not in trace[0]


@pytest.mark.asyncio
async def test_multi_pass_traced():
    """Multi-pass pipeline: each pass appears in trace."""
    fast = mock(["Draft."])
    quality = mock(["Refined."])

    agent = Agent(
        soul="Test.",
        memory=NullMemory(),
        backend=fast,
        pipeline=[think(fast), think(quality)],
        name="test",
    )

    resp = await agent(_req())
    trace = resp.metadata["trace"]

    assert len(trace) == 2
    assert trace[0]["pass"] == "think(mock)"
    assert trace[1]["pass"] == "think(mock)"
    assert resp.response == "Refined."


@pytest.mark.asyncio
async def test_named_pass_in_trace():
    """Custom passes can be named for tracing."""
    async def _shout(draft: Draft) -> Draft:
        draft.text = draft.text.upper()
        return draft
    shout = named("shout", _shout)

    backend = mock(["hello world"])
    agent = Agent(
        soul="Test.",
        memory=NullMemory(),
        backend=backend,
        pipeline=[think(backend), shout],
        name="test",
    )

    resp = await agent(_req())
    trace = resp.metadata["trace"]

    assert trace[1]["pass"] == "shout"
    assert resp.response == "HELLO WORLD"


@pytest.mark.asyncio
async def test_error_attaches_draft():
    """When a pass fails, the draft state is attached to the exception."""
    async def _explode(draft: Draft) -> Draft:
        raise ValueError("boom")
    explode = named("explode", _explode)

    backend = mock(["ok"])
    agent = Agent(
        soul="Test.",
        memory=NullMemory(),
        backend=backend,
        pipeline=[think(backend), explode],
        name="test",
    )

    with pytest.raises(Exception) as exc_info:
        await agent(_req())

    # The draft should be attached
    assert hasattr(exc_info.value, 'draft')
    assert exc_info.value.failed_pass == "explode"


# ============================================================================
# Introspection
# ============================================================================

@pytest.mark.asyncio
async def test_tabula_rasa_clones_without_memory():
    """tabula_rasa creates a clone with NullMemory."""
    backend = mock(["Original.", "Clone."])
    original = Agent(soul="You are Elle.", memory=NullMemory(), backend=backend, name="elle")

    fresh = tabula_rasa(original)

    assert fresh.name == "elle:tabula_rasa"
    assert fresh.soul == original.soul
    assert fresh.backend is original.backend
    assert isinstance(fresh.memory, NullMemory)


@pytest.mark.asyncio
async def test_consult_injects_response():
    """consult() wraps an agent as a Pass and injects its response."""
    elle_backend = mock(["Elle says: water the tomatoes."])
    elle = Agent(soul="You are Elle.", memory=NullMemory(), backend=elle_backend, name="elle")

    main_backend = mock(["Quick draft.", "Final: water the tomatoes daily."])
    agent = Agent(
        soul="You are a gardener.",
        memory=NullMemory(),
        backend=main_backend,
        pipeline=[think(main_backend), consult(elle), think(main_backend)],
        name="gardener",
    )

    resp = await agent(_req("How should I water tomatoes?"))

    # Consultation stored in metadata
    assert "elle" in resp.metadata.get("consultations", {})
    assert "water the tomatoes" in resp.metadata["consultations"]["elle"]

    # Trace shows all three passes
    trace = resp.metadata["trace"]
    assert len(trace) == 3
    assert trace[1]["pass"] == "consult(elle)"

    # Final response is from the last think
    assert resp.response == "Final: water the tomatoes daily."


# ============================================================================
# Skip flags
# ============================================================================

@pytest.mark.asyncio
async def test_skip_memory_flag():
    """Pipeline can suppress memory storage via metadata."""
    async def _skip(draft: Draft) -> Draft:
        draft.metadata["skip_memory"] = True
        return draft
    skip = named("skip_memory", _skip)

    # Use a real-ish memory to check nothing is stored
    stored = []

    class SpyMemory:
        max_turns = 10
        def get_messages(self, room_id): return []
        def add_turn(self, room_id, user_msg, assistant_msg):
            stored.append((room_id, user_msg, assistant_msg))
        def room_count(self): return 0
        def turn_count(self, room_id): return 0
        def total_turns(self): return 0

    backend = mock(["Response."])
    agent = Agent(
        soul="Test.",
        memory=SpyMemory(),
        backend=backend,
        pipeline=[think(backend), skip],
        name="test",
    )

    await agent(_req())
    assert len(stored) == 0  # Memory was skipped


# ============================================================================
# Loop convergence
# ============================================================================

@pytest.mark.asyncio
async def test_loop_converges():
    """loop() stops when passes produce stable output."""
    backend = mock(["First draft.", "Second draft.", "Second draft."])

    refine = loop(think(backend), max=3)
    agent = Agent(
        soul="Test.",
        memory=NullMemory(),
        backend=backend,
        pipeline=[refine],
        name="test",
    )

    resp = await agent(_req())
    # Should converge on second iteration (second and third responses match)
    assert backend.calls == 3  # think called inside loop
    assert resp.response == "Second draft."
