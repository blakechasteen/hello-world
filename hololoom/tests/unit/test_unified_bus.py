"""
Tests for the Unified Bus — Phase 1.

Validates:
- Signal creation, ordering, serialization
- Emit + dispatch to target/kind/broadcast handlers
- Correlation chain propagation
- TTL expiration
- Dedup
- Request-response (emit-and-await)
- History and statistics
"""

import asyncio
import pytest
from hololoom.core.bus import (
    UnifiedBus,
    Signal,
    SignalKind,
    SignalPriority,
    TriggerMode,
    create_bus,
)


# ============================================================================
# Signal Tests
# ============================================================================

class TestSignal:
    def test_create_signal(self):
        s = Signal(kind=SignalKind.QUERY, source="test")
        assert s.kind == SignalKind.QUERY
        assert s.source == "test"
        assert s.priority == SignalPriority.NORMAL
        assert s.trigger_mode == TriggerMode.ACTIVE
        assert len(s.id) == 12

    def test_priority_ordering(self):
        critical = Signal(kind=SignalKind.ERROR, priority=SignalPriority.CRITICAL)
        normal = Signal(kind=SignalKind.QUERY, priority=SignalPriority.NORMAL)
        low = Signal(kind=SignalKind.HEARTBEAT, priority=SignalPriority.LOW)

        assert critical < normal
        assert normal < low
        assert critical < low

    def test_same_priority_orders_by_time(self):
        s1 = Signal(kind=SignalKind.QUERY, created_at="2026-01-01T00:00:00")
        s2 = Signal(kind=SignalKind.QUERY, created_at="2026-01-01T00:00:01")
        assert s1 < s2

    def test_ttl_not_expired(self):
        s = Signal(kind=SignalKind.QUERY, ttl_seconds=3600)
        assert not s.is_expired()

    def test_ttl_expired(self):
        s = Signal(
            kind=SignalKind.QUERY,
            ttl_seconds=0,
            created_at="2020-01-01T00:00:00",
        )
        assert s.is_expired()

    def test_no_ttl_never_expires(self):
        s = Signal(kind=SignalKind.QUERY, ttl_seconds=None)
        assert not s.is_expired()

    def test_derive_child_signal(self):
        parent = Signal(
            kind=SignalKind.QUERY,
            source="api",
            correlation_id="workflow-1",
        )
        child = parent.derive(
            kind=SignalKind.MCTS_DECISION,
            source="mcts",
            payload={"strategy": "verified_query"},
        )

        assert child.kind == SignalKind.MCTS_DECISION
        assert child.causation_id == parent.id
        assert child.correlation_id == "workflow-1"
        assert child.trigger_mode == TriggerMode.REACTIVE
        assert child.payload["strategy"] == "verified_query"

    def test_derive_starts_correlation_if_none(self):
        parent = Signal(kind=SignalKind.QUERY, correlation_id=None)
        child = parent.derive(kind=SignalKind.EXECUTION_START)
        assert child.correlation_id == parent.id

    def test_serialization_roundtrip(self):
        original = Signal(
            kind=SignalKind.STEP_COMPLETE,
            priority=SignalPriority.HIGH,
            source="chain_adapter",
            target="convergence",
            payload={"step_id": "verify", "confidence": 0.85},
            metadata={"chain": "verified_query"},
            correlation_id="corr-123",
            causation_id="cause-456",
            ttl_seconds=60,
            trigger_mode=TriggerMode.REACTIVE,
        )
        data = original.to_dict()
        restored = Signal.from_dict(data)

        assert restored.kind == original.kind
        assert restored.priority == original.priority
        assert restored.source == original.source
        assert restored.target == original.target
        assert restored.payload == original.payload
        assert restored.correlation_id == original.correlation_id
        assert restored.causation_id == original.causation_id
        assert restored.ttl_seconds == original.ttl_seconds
        assert restored.trigger_mode == original.trigger_mode

    def test_equality_and_hash(self):
        s1 = Signal(id="abc123", kind=SignalKind.QUERY)
        s2 = Signal(id="abc123", kind=SignalKind.ANSWER)
        s3 = Signal(id="def456", kind=SignalKind.QUERY)

        assert s1 == s2  # Same ID
        assert s1 != s3  # Different ID
        assert hash(s1) == hash(s2)


# ============================================================================
# UnifiedBus Tests
# ============================================================================

class TestUnifiedBus:
    @pytest.fixture
    def bus(self):
        return UnifiedBus()

    @pytest.mark.asyncio
    async def test_emit_returns_id(self, bus):
        signal = Signal(kind=SignalKind.QUERY)
        result = await bus.emit(signal)
        assert result == signal.id

    @pytest.mark.asyncio
    async def test_target_dispatch(self, bus):
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.subscribe("my_agent", handler)

        await bus.emit(Signal(kind=SignalKind.QUERY, target="my_agent"))
        await asyncio.sleep(0.01)  # Let dispatch complete

        assert len(received) == 1
        assert received[0].kind == SignalKind.QUERY

    @pytest.mark.asyncio
    async def test_target_miss(self, bus):
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.subscribe("agent_a", handler)

        await bus.emit(Signal(kind=SignalKind.QUERY, target="agent_b"))
        await asyncio.sleep(0.01)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_kind_dispatch(self, bus):
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.STEP_COMPLETE, handler)

        await bus.emit(Signal(kind=SignalKind.STEP_COMPLETE))
        await bus.emit(Signal(kind=SignalKind.QUERY))  # Should NOT match
        await asyncio.sleep(0.01)

        assert len(received) == 1
        assert received[0].kind == SignalKind.STEP_COMPLETE

    @pytest.mark.asyncio
    async def test_broadcast_dispatch(self, bus):
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.subscribe_broadcast(handler)

        await bus.emit(Signal(kind=SignalKind.QUERY))  # No target → broadcast
        await asyncio.sleep(0.01)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_broadcast_skipped_when_targeted(self, bus):
        broadcast_received = []
        target_received = []

        async def broadcast_handler(s: Signal):
            broadcast_received.append(s)
            return None

        async def target_handler(s: Signal):
            target_received.append(s)
            return None

        bus.subscribe_broadcast(broadcast_handler)
        bus.subscribe("specific", target_handler)

        # Targeted signal should NOT go to broadcast
        await bus.emit(Signal(kind=SignalKind.QUERY, target="specific"))
        await asyncio.sleep(0.01)

        assert len(target_received) == 1
        assert len(broadcast_received) == 0

    @pytest.mark.asyncio
    async def test_kind_fires_even_when_targeted(self, bus):
        """Kind handlers fire regardless of target (they're type-based)."""
        kind_received = []

        async def kind_handler(s: Signal):
            kind_received.append(s)
            return None

        bus.on(SignalKind.STEP_COMPLETE, kind_handler)

        await bus.emit(Signal(kind=SignalKind.STEP_COMPLETE, target="some_target"))
        await asyncio.sleep(0.01)

        assert len(kind_received) == 1

    @pytest.mark.asyncio
    async def test_handler_response_propagates_correlation(self, bus):
        """When a handler returns a Signal, it should inherit the correlation chain."""
        child_signals = []

        async def responder(s: Signal):
            return Signal(
                kind=SignalKind.MCTS_DECISION,
                source="mcts",
                payload={"strategy": "auto_refine"},
            )

        async def collector(s: Signal):
            child_signals.append(s)
            return None

        bus.on(SignalKind.QUERY, responder)
        bus.on(SignalKind.MCTS_DECISION, collector)

        parent = Signal(kind=SignalKind.QUERY, source="api")
        await bus.emit(parent)
        await asyncio.sleep(0.05)

        assert len(child_signals) == 1
        child = child_signals[0]
        assert child.correlation_id == parent.id
        assert child.causation_id == parent.id
        assert child.trigger_mode == TriggerMode.REACTIVE

    @pytest.mark.asyncio
    async def test_ttl_expired_signals_not_dispatched(self, bus):
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.QUERY, handler)

        await bus.emit(Signal(
            kind=SignalKind.QUERY,
            ttl_seconds=0,
            created_at="2020-01-01T00:00:00",
        ))
        await asyncio.sleep(0.01)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_dedup(self, bus):
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.QUERY, handler)

        signal = Signal(id="same-id", kind=SignalKind.QUERY)
        await bus.emit(signal)
        await bus.emit(signal)  # Duplicate
        await asyncio.sleep(0.01)

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_request_response(self, bus):
        """Test emit-and-await pattern."""

        async def echo_handler(s: Signal):
            return Signal(
                kind=SignalKind.EXECUTION_COMPLETE,
                payload={"echo": s.payload.get("text")},
            )

        bus.on(SignalKind.QUERY, echo_handler)

        response = await bus.request(
            Signal(kind=SignalKind.QUERY, payload={"text": "hello"}),
            response_kind=SignalKind.EXECUTION_COMPLETE,
            timeout=5.0,
        )

        assert response.kind == SignalKind.EXECUTION_COMPLETE
        assert response.payload["echo"] == "hello"

    @pytest.mark.asyncio
    async def test_request_timeout(self, bus):
        """Request with no responder should timeout."""
        with pytest.raises(asyncio.TimeoutError):
            await bus.request(
                Signal(kind=SignalKind.QUERY),
                response_kind=SignalKind.EXECUTION_COMPLETE,
                timeout=0.1,
            )

    @pytest.mark.asyncio
    async def test_history(self, bus):
        await bus.emit(Signal(kind=SignalKind.QUERY, source="a"))
        await bus.emit(Signal(kind=SignalKind.STEP_COMPLETE, source="b"))
        await bus.emit(Signal(kind=SignalKind.QUERY, source="c"))
        await asyncio.sleep(0.01)

        all_history = bus.get_history()
        assert len(all_history) == 3

        query_history = bus.get_history(kind=SignalKind.QUERY)
        assert len(query_history) == 2

        source_history = bus.get_history(source="b")
        assert len(source_history) == 1

    @pytest.mark.asyncio
    async def test_statistics(self, bus):
        await bus.emit(Signal(kind=SignalKind.QUERY))
        await bus.emit(Signal(kind=SignalKind.HEARTBEAT, priority=SignalPriority.LOW))
        await asyncio.sleep(0.01)

        stats = bus.get_statistics()
        assert stats["total_emitted"] == 2
        assert stats["by_kind"]["query"] == 1
        assert stats["by_kind"]["heartbeat"] == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus):
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.subscribe("target", handler)
        await bus.emit(Signal(kind=SignalKind.QUERY, target="target"))
        await asyncio.sleep(0.01)
        assert len(received) == 1

        bus.unsubscribe("target", handler)
        await bus.emit(Signal(id="new-id", kind=SignalKind.QUERY, target="target"))
        await asyncio.sleep(0.01)
        assert len(received) == 1  # No new signals

    @pytest.mark.asyncio
    async def test_off_kind(self, bus):
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.QUERY, handler)
        await bus.emit(Signal(kind=SignalKind.QUERY))
        await asyncio.sleep(0.01)
        assert len(received) == 1

        bus.off(SignalKind.QUERY, handler)
        await bus.emit(Signal(id="new", kind=SignalKind.QUERY))
        await asyncio.sleep(0.01)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_handler_error_does_not_break_dispatch(self, bus):
        """A failing handler should not prevent other handlers from running."""
        received = []

        async def bad_handler(s: Signal):
            raise ValueError("boom")

        async def good_handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.QUERY, bad_handler)
        bus.on(SignalKind.QUERY, good_handler)

        await bus.emit(Signal(kind=SignalKind.QUERY))
        await asyncio.sleep(0.01)

        assert len(received) == 1  # Good handler still fired

    @pytest.mark.asyncio
    async def test_clear_history(self, bus):
        await bus.emit(Signal(kind=SignalKind.QUERY))
        assert len(bus.get_history()) == 1

        cleared = bus.clear_history()
        assert cleared == 1
        assert len(bus.get_history()) == 0

    @pytest.mark.asyncio
    async def test_create_bus_factory(self):
        bus = create_bus(history_limit=500)
        assert isinstance(bus, UnifiedBus)
        assert bus._history_limit == 500

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with UnifiedBus() as bus:
            assert bus._running
        assert not bus._running

    @pytest.mark.asyncio
    async def test_multiple_kind_handlers(self, bus):
        """Multiple handlers on the same kind all fire."""
        results = []

        async def handler_a(s: Signal):
            results.append("a")
            return None

        async def handler_b(s: Signal):
            results.append("b")
            return None

        bus.on(SignalKind.QUERY, handler_a)
        bus.on(SignalKind.QUERY, handler_b)

        await bus.emit(Signal(kind=SignalKind.QUERY))
        await asyncio.sleep(0.01)

        assert sorted(results) == ["a", "b"]
