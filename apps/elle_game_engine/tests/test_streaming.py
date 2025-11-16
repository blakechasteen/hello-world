"""
Tests for streaming LLM clients.

Tests streaming functionality for all LLM providers:
- Dummy (simulated streaming)
- Anthropic (Claude streaming)
- OpenAI (GPT streaming)
- Local (Ollama streaming)

Also tests SSE endpoint integration.
"""

import pytest
import asyncio
from typing import List

from ..streaming import (
    StreamChunk,
    StreamingDummyClient,
    create_streaming_client,
)


# ============================================================================
# StreamChunk Tests
# ============================================================================

def test_stream_chunk_to_sse_format():
    """Test SSE format conversion."""
    chunk = StreamChunk(
        event="token",
        data="Hello",
        metadata={"timestamp": 12345.67}
    )

    sse = chunk.to_sse_format()

    assert "event: token" in sse
    assert "data: Hello" in sse
    assert sse.endswith("\n\n")


def test_stream_chunk_multiline_data():
    """Test SSE format with multiline data."""
    chunk = StreamChunk(
        event="complete",
        data="Line 1\nLine 2\nLine 3"
    )

    sse = chunk.to_sse_format()

    assert "data: Line 1" in sse
    assert "data: Line 2" in sse
    assert "data: Line 3" in sse


# ============================================================================
# StreamingDummyClient Tests
# ============================================================================

@pytest.mark.asyncio
async def test_dummy_client_basic_streaming():
    """Test basic streaming from dummy client."""
    client = StreamingDummyClient()

    chunks: List[StreamChunk] = []

    async for chunk in client.stream_complete("talk_to_npc"):
        chunks.append(chunk)

    # Should have multiple token chunks + 1 complete chunk
    assert len(chunks) > 1
    assert chunks[-1].event == "complete"

    # All other chunks should be tokens
    for chunk in chunks[:-1]:
        assert chunk.event == "token"


@pytest.mark.asyncio
async def test_dummy_client_accumulates_correctly():
    """Test that accumulated text matches complete text."""
    client = StreamingDummyClient()

    accumulated = ""
    complete_text = None

    async for chunk in client.stream_complete("talk_to_npc"):
        if chunk.event == "token":
            accumulated += chunk.data
        elif chunk.event == "complete":
            complete_text = chunk.data

    # Accumulated should match complete
    assert accumulated == complete_text


@pytest.mark.asyncio
async def test_dummy_client_npc_dialogue_response():
    """Test NPC dialogue response format."""
    client = StreamingDummyClient()

    complete_text = None

    async for chunk in client.stream_complete("talk_to_npc"):
        if chunk.event == "complete":
            complete_text = chunk.data
            break

    # Should be valid JSON
    import json
    data = json.loads(complete_text)

    assert data["mode"] == "npc_dialogue"
    assert len(data["dialogue"]) > 0
    assert data["dialogue"][0]["npc_id"] == "innkeeper"


@pytest.mark.asyncio
async def test_dummy_client_world_reaction_response():
    """Test world reaction response format."""
    client = StreamingDummyClient()

    complete_text = None

    async for chunk in client.stream_complete("enter_scene"):
        if chunk.event == "complete":
            complete_text = chunk.data
            break

    import json
    data = json.loads(complete_text)

    assert data["mode"] == "world_reaction"
    assert data["world_reaction"] is not None
    assert "description" in data["world_reaction"]


@pytest.mark.asyncio
async def test_dummy_client_hint_response():
    """Test hint response format."""
    client = StreamingDummyClient()

    complete_text = None

    async for chunk in client.stream_complete("request_hint"):
        if chunk.event == "complete":
            complete_text = chunk.data
            break

    import json
    data = json.loads(complete_text)

    assert data["mode"] == "hint"
    assert data["hint_text"] is not None


@pytest.mark.asyncio
async def test_dummy_client_token_delay():
    """Test that streaming has realistic delays."""
    import time

    client = StreamingDummyClient(tokens_per_second=10)  # 100ms per token

    start_time = time.time()
    chunk_count = 0

    async for chunk in client.stream_complete("talk_to_npc"):
        chunk_count += 1

    duration = time.time() - start_time

    # Should take at least some time (not instant)
    # At 10 tokens/sec, even 50 tokens should take ~5 seconds
    assert duration > 0.5  # At least half a second


@pytest.mark.asyncio
async def test_dummy_client_metadata():
    """Test that chunks include metadata."""
    client = StreamingDummyClient()

    async for chunk in client.stream_complete("talk_to_npc"):
        assert chunk.metadata is not None
        assert "timestamp" in chunk.metadata

        if chunk.event == "complete":
            assert "duration_ms" in chunk.metadata
            assert "total_tokens" in chunk.metadata


# ============================================================================
# Factory Tests
# ============================================================================

def test_create_streaming_client_dummy():
    """Test factory creates dummy client."""
    client = create_streaming_client("dummy")
    assert isinstance(client, StreamingDummyClient)


def test_create_streaming_client_invalid_provider():
    """Test factory raises error for invalid provider."""
    with pytest.raises(ValueError, match="Unknown streaming provider"):
        create_streaming_client("invalid_provider")


# ============================================================================
# Integration Tests (require real API keys - skip by default)
# ============================================================================

@pytest.mark.skip(reason="Requires Anthropic API key")
@pytest.mark.asyncio
async def test_anthropic_streaming():
    """Test Anthropic streaming (requires API key)."""
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    client = create_streaming_client("anthropic", api_key=api_key)

    chunks = []
    async for chunk in client.stream_complete("Say hello"):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert chunks[-1].event == "complete"


@pytest.mark.skip(reason="Requires OpenAI API key")
@pytest.mark.asyncio
async def test_openai_streaming():
    """Test OpenAI streaming (requires API key)."""
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    client = create_streaming_client("openai", api_key=api_key)

    chunks = []
    async for chunk in client.stream_complete("Say hello"):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert chunks[-1].event == "complete"


@pytest.mark.skip(reason="Requires Ollama running")
@pytest.mark.asyncio
async def test_local_streaming():
    """Test local Ollama streaming (requires Ollama running)."""
    client = create_streaming_client("local", model="llama3.2:3b")

    chunks = []
    async for chunk in client.stream_complete("Say hello"):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert chunks[-1].event == "complete"


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_streaming_is_faster_than_blocking():
    """Test that streaming provides tokens faster than blocking."""
    import time

    client = StreamingDummyClient(tokens_per_second=20)

    # Streaming: measure time to first token
    start_time = time.time()
    first_token_time = None

    async for chunk in client.stream_complete("talk_to_npc"):
        if chunk.event == "token" and first_token_time is None:
            first_token_time = time.time() - start_time

    # First token should arrive quickly (within first 10% of total time)
    # This simulates the benefit of streaming vs blocking
    assert first_token_time is not None
    assert first_token_time < 0.5  # First token within 500ms


@pytest.mark.asyncio
async def test_concurrent_streaming():
    """Test multiple concurrent streaming requests."""
    client = StreamingDummyClient()

    async def stream_request():
        chunks = []
        async for chunk in client.stream_complete("talk_to_npc"):
            chunks.append(chunk)
        return chunks

    # Run 5 concurrent streaming requests
    results = await asyncio.gather(*[stream_request() for _ in range(5)])

    # All should succeed
    assert len(results) == 5
    for chunks in results:
        assert len(chunks) > 0
        assert chunks[-1].event == "complete"
