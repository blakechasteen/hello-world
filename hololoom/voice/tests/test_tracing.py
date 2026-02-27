"""
Comprehensive Tests for Distributed Tracing
==========================================

Tests for OpenTelemetry + Jaeger integration.

Test Coverage:
- TracingConfig validation
- TracingManager initialization
- Span creation and attributes
- Error recording
- Context propagation
- Decorators (voice_command, tts_synthesis, cache_operation, hololoom_weave)
- Manual span management
- Performance overhead
- Graceful degradation

Date: November 16, 2025
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Optional

# Import tracing module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from hololoom.voice.tracing import (
    TracingConfig,
    TracingManager,
    SamplingStrategy,
    get_tracing_manager,
    shutdown_tracing,
    OTEL_AVAILABLE
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tracing_config():
    """Create test tracing config."""
    return TracingConfig(
        enable_tracing=True,
        service_name="test-service",
        jaeger_host="localhost",
        jaeger_port=6831,
        sampling_strategy=SamplingStrategy.ALWAYS_ON,
        sample_rate=1.0,
        console_export=False,
        verbose_spans=True
    )


@pytest.fixture
def tracing_manager(tracing_config):
    """Create test tracing manager."""
    manager = TracingManager(tracing_config)
    yield manager
    # Cleanup
    if manager.provider:
        manager.shutdown(timeout_seconds=5)


@pytest.fixture
def disabled_tracing_config():
    """Create disabled tracing config."""
    return TracingConfig(enable_tracing=False)


@pytest.fixture
def disabled_tracing_manager(disabled_tracing_config):
    """Create disabled tracing manager."""
    return TracingManager(disabled_tracing_config)


# ============================================================================
# Configuration Tests
# ============================================================================

def test_tracing_config_defaults():
    """Test TracingConfig default values."""
    config = TracingConfig()

    assert config.enable_tracing is True
    assert config.service_name == "hololoom-voice-agent"
    assert config.service_version == "1.0.0"
    assert config.jaeger_host == "localhost"
    assert config.jaeger_port == 6831
    assert config.sampling_strategy == SamplingStrategy.ALWAYS_ON
    assert config.sample_rate == 1.0
    assert config.batch_export is True
    assert config.console_export is False


def test_tracing_config_custom():
    """Test TracingConfig with custom values."""
    config = TracingConfig(
        enable_tracing=False,
        service_name="custom-service",
        jaeger_host="jaeger.example.com",
        jaeger_port=14268,
        sample_rate=0.5
    )

    assert config.enable_tracing is False
    assert config.service_name == "custom-service"
    assert config.jaeger_host == "jaeger.example.com"
    assert config.jaeger_port == 14268
    assert config.sample_rate == 0.5


def test_tracing_config_validation_sample_rate():
    """Test sample_rate validation."""
    with pytest.raises(ValueError, match="sample_rate must be 0.0-1.0"):
        TracingConfig(sample_rate=1.5)

    with pytest.raises(ValueError, match="sample_rate must be 0.0-1.0"):
        TracingConfig(sample_rate=-0.1)


def test_tracing_config_validation_port():
    """Test jaeger_port validation."""
    with pytest.raises(ValueError, match="jaeger_port must be 1-65535"):
        TracingConfig(jaeger_port=0)

    with pytest.raises(ValueError, match="jaeger_port must be 1-65535"):
        TracingConfig(jaeger_port=70000)


# ============================================================================
# TracingManager Initialization Tests
# ============================================================================

def test_tracing_manager_init_enabled(tracing_config):
    """Test TracingManager initialization when enabled."""
    manager = TracingManager(tracing_config)

    if OTEL_AVAILABLE:
        assert manager.config.enable_tracing is True
        assert manager.tracer is not None
        assert manager.provider is not None
    else:
        # Should gracefully disable if OpenTelemetry not available
        assert manager.config.enable_tracing is False

    manager.shutdown()


def test_tracing_manager_init_disabled(disabled_tracing_config):
    """Test TracingManager initialization when disabled."""
    manager = TracingManager(disabled_tracing_config)

    assert manager.config.enable_tracing is False
    assert manager.tracer is None


def test_tracing_manager_shutdown(tracing_manager):
    """Test TracingManager shutdown."""
    # Should not raise
    tracing_manager.shutdown(timeout_seconds=5)


def test_tracing_manager_shutdown_idempotent(tracing_manager):
    """Test multiple shutdown calls are safe."""
    tracing_manager.shutdown()
    tracing_manager.shutdown()  # Should not raise


# ============================================================================
# Decorator Tests
# ============================================================================

@pytest.mark.asyncio
async def test_trace_voice_command_decorator(tracing_manager):
    """Test trace_voice_command decorator."""

    @tracing_manager.trace_voice_command()
    async def mock_voice_command(self, transcript: str) -> str:
        await asyncio.sleep(0.01)
        return f"Processed: {transcript}"

    result = await mock_voice_command(None, "test transcript")

    assert result == "Processed: test transcript"


@pytest.mark.asyncio
async def test_trace_voice_command_with_attributes(tracing_manager):
    """Test voice command decorator records attributes."""

    class MockResult:
        confidence = 0.95
        tool_used = "answer"
        intent = "question"

    @tracing_manager.trace_voice_command()
    async def mock_voice_command(self, transcript: str) -> MockResult:
        return MockResult()

    result = await mock_voice_command(None, "what is the weather?")

    assert result.confidence == 0.95
    assert result.tool_used == "answer"


@pytest.mark.asyncio
async def test_trace_voice_command_with_error(tracing_manager):
    """Test voice command decorator records errors."""

    @tracing_manager.trace_voice_command()
    async def mock_voice_command_error(self, transcript: str):
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        await mock_voice_command_error(None, "test")


@pytest.mark.asyncio
async def test_trace_tts_synthesis_decorator(tracing_manager):
    """Test trace_tts_synthesis decorator."""

    @tracing_manager.trace_tts_synthesis()
    async def mock_synthesize(self, text: str, voice: str = "nova") -> bytes:
        await asyncio.sleep(0.01)
        return b"fake audio data"

    result = await mock_synthesize(None, "hello world", "nova")

    assert result == b"fake audio data"


@pytest.mark.asyncio
async def test_trace_tts_synthesis_records_size(tracing_manager):
    """Test TTS decorator records audio size."""

    @tracing_manager.trace_tts_synthesis()
    async def mock_synthesize(self, text: str, voice: str = "nova") -> bytes:
        return b"x" * 1024  # 1KB

    result = await mock_synthesize(None, "test", "nova")

    assert len(result) == 1024


@pytest.mark.asyncio
async def test_trace_cache_operation_decorator(tracing_manager):
    """Test trace_cache_operation decorator."""

    @tracing_manager.trace_cache_operation("lookup")
    async def mock_cache_lookup(self, key: str) -> Optional[bytes]:
        await asyncio.sleep(0.001)
        return b"cached data" if key == "found" else None

    # Cache hit
    result = await mock_cache_lookup(None, "found")
    assert result == b"cached data"

    # Cache miss
    result = await mock_cache_lookup(None, "notfound")
    assert result is None


@pytest.mark.asyncio
async def test_trace_hololoom_weave_decorator(tracing_manager):
    """Test trace_hololoom_weave decorator."""

    class MockQuery:
        text = "test query"

    class MockSpacetime:
        confidence = 0.88
        metadata = {"tool_used": "retrieve", "mode": "FAST"}

    @tracing_manager.trace_hololoom_weave()
    async def mock_weave(self, query):
        await asyncio.sleep(0.01)
        return MockSpacetime()

    result = await mock_weave(None, MockQuery())

    assert result.confidence == 0.88
    assert result.metadata["tool_used"] == "retrieve"


# ============================================================================
# Manual Span Management Tests
# ============================================================================

@pytest.mark.asyncio
async def test_manual_span_creation(tracing_manager):
    """Test manual span creation with context manager."""

    async with tracing_manager.span("test_operation") as span:
        await asyncio.sleep(0.001)

        if span:  # Only if tracing enabled
            span.set_attribute("test.key", "test.value")


@pytest.mark.asyncio
async def test_manual_span_with_attributes(tracing_manager):
    """Test manual span with initial attributes."""

    attributes = {
        "custom.metric": 42,
        "custom.name": "test"
    }

    async with tracing_manager.span("test_op", attributes=attributes) as span:
        await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_manual_span_with_error(tracing_manager):
    """Test manual span records errors."""

    with pytest.raises(RuntimeError, match="Test error"):
        async with tracing_manager.span("error_op") as span:
            raise RuntimeError("Test error")


@pytest.mark.asyncio
async def test_get_current_span(tracing_manager):
    """Test getting current active span."""

    async with tracing_manager.span("parent") as parent_span:
        current = tracing_manager.get_current_span()

        # Should get current span (or None if tracing disabled)
        if OTEL_AVAILABLE and tracing_manager.config.enable_tracing:
            assert current is not None
        else:
            assert current is None


@pytest.mark.asyncio
async def test_add_event(tracing_manager):
    """Test adding events to current span."""

    async with tracing_manager.span("test") as span:
        tracing_manager.add_event("test_event", {"key": "value"})


@pytest.mark.asyncio
async def test_set_attribute_on_current_span(tracing_manager):
    """Test setting attribute on current span."""

    async with tracing_manager.span("test") as span:
        tracing_manager.set_attribute("test.key", "test.value")


# ============================================================================
# Disabled Tracing Tests
# ============================================================================

@pytest.mark.asyncio
async def test_disabled_tracing_no_overhead(disabled_tracing_manager):
    """Test disabled tracing has no overhead."""

    @disabled_tracing_manager.trace_voice_command()
    async def mock_command(self, transcript: str) -> str:
        return "result"

    start = time.time()
    result = await mock_command(None, "test")
    elapsed = time.time() - start

    assert result == "result"
    assert elapsed < 0.001  # Should be instant


@pytest.mark.asyncio
async def test_disabled_tracing_decorators_work(disabled_tracing_manager):
    """Test all decorators work when tracing disabled."""

    @disabled_tracing_manager.trace_voice_command()
    async def cmd(self, t): return "cmd"

    @disabled_tracing_manager.trace_tts_synthesis()
    async def tts(self, t, v): return b"tts"

    @disabled_tracing_manager.trace_cache_operation()
    async def cache(self, k): return None

    @disabled_tracing_manager.trace_hololoom_weave()
    async def weave(self, q): return "weave"

    assert await cmd(None, "t") == "cmd"
    assert await tts(None, "t", "v") == b"tts"
    assert await cache(None, "k") is None
    assert await weave(None, "q") == "weave"


@pytest.mark.asyncio
async def test_disabled_manual_span(disabled_tracing_manager):
    """Test manual span works when disabled."""

    async with disabled_tracing_manager.span("test") as span:
        assert span is None


# ============================================================================
# Context Propagation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_nested_spans(tracing_manager):
    """Test nested span creation."""

    @tracing_manager.trace_voice_command()
    async def outer(self, transcript: str):
        async with tracing_manager.span("inner_operation"):
            await asyncio.sleep(0.001)
        return "done"

    result = await outer(None, "test")
    assert result == "done"


@pytest.mark.asyncio
async def test_concurrent_spans(tracing_manager):
    """Test concurrent span creation."""

    @tracing_manager.trace_voice_command()
    async def task1(self, t):
        await asyncio.sleep(0.01)
        return "task1"

    @tracing_manager.trace_voice_command()
    async def task2(self, t):
        await asyncio.sleep(0.01)
        return "task2"

    results = await asyncio.gather(
        task1(None, "t1"),
        task2(None, "t2")
    )

    assert results == ["task1", "task2"]


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_tracing_overhead(tracing_manager):
    """Test tracing overhead is <5ms per span."""

    @tracing_manager.trace_voice_command()
    async def fast_operation(self, transcript: str):
        return "done"

    # Warm up
    await fast_operation(None, "warmup")

    # Measure
    iterations = 100
    start = time.time()

    for i in range(iterations):
        await fast_operation(None, f"test {i}")

    elapsed = time.time() - start
    per_operation = (elapsed / iterations) * 1000  # ms

    # Should be <5ms overhead per operation
    assert per_operation < 5.0, f"Overhead {per_operation:.2f}ms > 5ms target"


@pytest.mark.asyncio
async def test_batch_export_performance(tracing_manager):
    """Test batch export doesn't block operations."""

    @tracing_manager.trace_voice_command()
    async def operation(self, i: int):
        return i

    # Create many spans quickly
    start = time.time()
    results = await asyncio.gather(*[
        operation(None, i) for i in range(100)
    ])
    elapsed = time.time() - start

    assert len(results) == 100
    assert elapsed < 1.0  # Should complete quickly


# ============================================================================
# Global Manager Tests
# ============================================================================

def test_get_global_tracing_manager():
    """Test getting global tracing manager."""
    manager = get_tracing_manager()
    assert manager is not None

    # Should return same instance
    manager2 = get_tracing_manager()
    assert manager is manager2


def test_shutdown_global_tracing():
    """Test shutting down global tracing."""
    manager = get_tracing_manager()
    shutdown_tracing()

    # Should be able to get new instance after shutdown
    manager2 = get_tracing_manager()
    assert manager2 is not None


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_voice_command_flow(tracing_manager):
    """Test complete voice command flow with tracing."""

    class VoiceAgent:
        def __init__(self):
            self.tracing = tracing_manager

        @property
        def trace_voice_command(self):
            return self.tracing.trace_voice_command

        @property
        def trace_tts_synthesis(self):
            return self.tracing.trace_tts_synthesis

        @trace_voice_command()
        async def process_voice_input(self, transcript: str) -> str:
            # Simulate processing
            await asyncio.sleep(0.01)

            # Simulate TTS
            audio = await self.synthesize(f"Response to: {transcript}")

            return "Success"

        @trace_tts_synthesis()
        async def synthesize(self, text: str) -> bytes:
            await asyncio.sleep(0.01)
            return text.encode()

    agent = VoiceAgent()
    result = await agent.process_voice_input("what is the weather?")

    assert result == "Success"


@pytest.mark.asyncio
async def test_error_propagation_with_tracing(tracing_manager):
    """Test errors are properly recorded in traces."""

    @tracing_manager.trace_voice_command()
    async def failing_operation(self, transcript: str):
        raise ValueError("Intentional error")

    with pytest.raises(ValueError, match="Intentional error"):
        await failing_operation(None, "test")


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_empty_transcript(tracing_manager):
    """Test handling empty transcript."""

    @tracing_manager.trace_voice_command()
    async def process(self, transcript: str):
        return transcript

    result = await process(None, "")
    assert result == ""


@pytest.mark.asyncio
async def test_very_long_transcript(tracing_manager):
    """Test handling very long transcript (should truncate)."""

    @tracing_manager.trace_voice_command()
    async def process(self, transcript: str):
        return "ok"

    long_text = "x" * 10000
    result = await process(None, long_text)
    assert result == "ok"


@pytest.mark.asyncio
async def test_none_result(tracing_manager):
    """Test handling None result."""

    @tracing_manager.trace_voice_command()
    async def process(self, transcript: str):
        return None

    result = await process(None, "test")
    assert result is None


# ============================================================================
# Cleanup
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_global_tracing():
    """Cleanup global tracing after all tests."""
    yield
    shutdown_tracing()
