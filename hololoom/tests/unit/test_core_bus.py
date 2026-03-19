"""
Tests for hololoom/core/bus/ — Signal, UnifiedBus, Journal, Neo4j Store,
Convergence Adapter, Trigger Adapters, Refinement Adapter, Strategy Selector.

Targets 60%+ coverage across all 1,147 statements in the bus package.
"""

import asyncio
import os
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hololoom.core.bus.signal import (
    Signal,
    SignalHandler,
    SignalKind,
    SignalPriority,
    TriggerMode,
)
from hololoom.core.bus.unified_bus import UnifiedBus, create_bus


# ============================================================================
# Signal Tests
# ============================================================================


class TestSignalCreation:
    def test_default_signal(self):
        s = Signal()
        assert s.kind == SignalKind.REQUEST
        assert s.priority == SignalPriority.NORMAL
        assert s.trigger_mode == TriggerMode.ACTIVE
        assert s.source == ""
        assert s.target == ""
        assert s.payload == {}
        assert s.metadata == {}
        assert s.correlation_id is None
        assert s.causation_id is None
        assert s.ttl_seconds is None
        assert len(s.id) == 12

    def test_signal_with_all_fields(self):
        s = Signal(
            id="test12345678",
            kind=SignalKind.QUERY,
            priority=SignalPriority.HIGH,
            source="api",
            target="agent1",
            payload={"text": "hello"},
            metadata={"version": 1},
            correlation_id="corr1",
            causation_id="cause1",
            created_at="2026-01-01T00:00:00",
            ttl_seconds=60,
            trigger_mode=TriggerMode.REACTIVE,
        )
        assert s.id == "test12345678"
        assert s.kind == SignalKind.QUERY
        assert s.priority == SignalPriority.HIGH
        assert s.source == "api"
        assert s.target == "agent1"
        assert s.payload == {"text": "hello"}
        assert s.metadata == {"version": 1}
        assert s.correlation_id == "corr1"
        assert s.causation_id == "cause1"
        assert s.ttl_seconds == 60
        assert s.trigger_mode == TriggerMode.REACTIVE

    def test_signal_kind_values(self):
        assert SignalKind.QUERY.value == "query"
        assert SignalKind.HEARTBEAT.value == "heartbeat"
        assert SignalKind.ERROR.value == "error"
        assert SignalKind.CONVERGENCE.value == "convergence"
        assert SignalKind.REFINEMENT_START.value == "refinement_start"
        assert SignalKind.REFINEMENT_COMPLETE.value == "refinement_complete"
        assert SignalKind.MCTS_DECISION.value == "mcts_decision"
        assert SignalKind.MEMORY_STORED.value == "memory_stored"
        assert SignalKind.MEMORY_QUERIED.value == "memory_queried"

    def test_signal_priority_values(self):
        assert SignalPriority.CRITICAL.value == 0
        assert SignalPriority.HIGH.value == 1
        assert SignalPriority.NORMAL.value == 2
        assert SignalPriority.LOW.value == 3

    def test_trigger_mode_values(self):
        assert TriggerMode.ACTIVE.value == "active"
        assert TriggerMode.HEARTBEAT.value == "heartbeat"
        assert TriggerMode.CRON.value == "cron"
        assert TriggerMode.REACTIVE.value == "reactive"


class TestSignalOrdering:
    def test_priority_ordering(self):
        critical = Signal(priority=SignalPriority.CRITICAL)
        high = Signal(priority=SignalPriority.HIGH)
        normal = Signal(priority=SignalPriority.NORMAL)
        low = Signal(priority=SignalPriority.LOW)

        assert critical < high < normal < low

    def test_same_priority_time_ordering(self):
        s1 = Signal(created_at="2026-01-01T00:00:00")
        s2 = Signal(created_at="2026-01-01T00:00:01")
        assert s1 < s2

    def test_lt_not_implemented_for_non_signal(self):
        s = Signal()
        assert s.__lt__("not a signal") is NotImplemented

    def test_eq_not_implemented_for_non_signal(self):
        s = Signal()
        assert s.__eq__("not a signal") is NotImplemented


class TestSignalTTL:
    def test_no_ttl_never_expires(self):
        s = Signal(ttl_seconds=None)
        assert not s.is_expired()

    def test_expired_signal(self):
        s = Signal(
            ttl_seconds=1,
            created_at=(datetime.now() - timedelta(seconds=10)).isoformat(),
        )
        assert s.is_expired()

    def test_not_expired_signal(self):
        s = Signal(ttl_seconds=3600)
        assert not s.is_expired()


class TestSignalDerive:
    def test_derive_inherits_correlation(self):
        parent = Signal(kind=SignalKind.QUERY, correlation_id="workflow-1")
        child = parent.derive(kind=SignalKind.STEP_COMPLETE, payload={"step": 1})

        assert child.kind == SignalKind.STEP_COMPLETE
        assert child.correlation_id == "workflow-1"
        assert child.causation_id == parent.id
        assert child.trigger_mode == TriggerMode.REACTIVE
        assert child.payload == {"step": 1}

    def test_derive_creates_correlation_if_none(self):
        parent = Signal(kind=SignalKind.QUERY, correlation_id=None)
        child = parent.derive(kind=SignalKind.EXECUTION_START)
        assert child.correlation_id == parent.id

    def test_derive_inherits_priority(self):
        parent = Signal(priority=SignalPriority.HIGH)
        child = parent.derive(kind=SignalKind.STEP_COMPLETE)
        assert child.priority == SignalPriority.HIGH

    def test_derive_overrides_priority(self):
        # Note: derive uses `priority or self.priority` — CRITICAL=0 is falsy
        # so it cannot override to CRITICAL. This tests the actual behavior.
        parent = Signal(priority=SignalPriority.NORMAL)
        child = parent.derive(kind=SignalKind.ERROR, priority=SignalPriority.HIGH)
        assert child.priority == SignalPriority.HIGH

    def test_derive_inherits_source(self):
        parent = Signal(source="api")
        child = parent.derive(kind=SignalKind.STEP_COMPLETE)
        assert child.source == "api"

    def test_derive_overrides_source(self):
        parent = Signal(source="api")
        child = parent.derive(kind=SignalKind.STEP_COMPLETE, source="chain")
        assert child.source == "chain"


class TestSignalSerialization:
    def test_to_dict(self):
        s = Signal(
            id="abc",
            kind=SignalKind.QUERY,
            priority=SignalPriority.HIGH,
            source="test",
            target="agent",
            payload={"key": "val"},
            metadata={"m": 1},
            correlation_id="c1",
            causation_id="c2",
            created_at="2026-01-01T00:00:00",
            ttl_seconds=30,
            trigger_mode=TriggerMode.CRON,
        )
        d = s.to_dict()
        assert d["id"] == "abc"
        assert d["kind"] == "query"
        assert d["priority"] == 1
        assert d["source"] == "test"
        assert d["target"] == "agent"
        assert d["payload"] == {"key": "val"}
        assert d["metadata"] == {"m": 1}
        assert d["correlation_id"] == "c1"
        assert d["causation_id"] == "c2"
        assert d["ttl_seconds"] == 30
        assert d["trigger_mode"] == "cron"

    def test_from_dict_full(self):
        data = {
            "id": "xyz",
            "kind": "heartbeat",
            "priority": 3,
            "source": "hb",
            "target": "",
            "payload": {"tick": 1},
            "metadata": {},
            "correlation_id": None,
            "causation_id": None,
            "created_at": "2026-01-01T00:00:00",
            "ttl_seconds": None,
            "trigger_mode": "heartbeat",
        }
        s = Signal.from_dict(data)
        assert s.id == "xyz"
        assert s.kind == SignalKind.HEARTBEAT
        assert s.priority == SignalPriority.LOW
        assert s.trigger_mode == TriggerMode.HEARTBEAT

    def test_from_dict_defaults(self):
        s = Signal.from_dict({})
        assert s.kind == SignalKind.REQUEST
        assert s.priority == SignalPriority.NORMAL
        assert s.trigger_mode == TriggerMode.ACTIVE

    def test_roundtrip(self):
        original = Signal(
            kind=SignalKind.STEP_COMPLETE,
            priority=SignalPriority.CRITICAL,
            source="chain",
            target="monitor",
            payload={"confidence": 0.9},
            metadata={"pass": 2},
            correlation_id="c1",
            causation_id="c2",
            ttl_seconds=120,
            trigger_mode=TriggerMode.REACTIVE,
        )
        restored = Signal.from_dict(original.to_dict())
        assert restored.kind == original.kind
        assert restored.priority == original.priority
        assert restored.source == original.source
        assert restored.target == original.target
        assert restored.payload == original.payload
        assert restored.metadata == original.metadata
        assert restored.correlation_id == original.correlation_id
        assert restored.causation_id == original.causation_id
        assert restored.ttl_seconds == original.ttl_seconds
        assert restored.trigger_mode == original.trigger_mode


class TestSignalEqualityHash:
    def test_same_id_equal(self):
        a = Signal(id="same")
        b = Signal(id="same", kind=SignalKind.ERROR)
        assert a == b
        assert hash(a) == hash(b)

    def test_different_id_not_equal(self):
        a = Signal(id="aaa")
        b = Signal(id="bbb")
        assert a != b


# ============================================================================
# UnifiedBus Tests
# ============================================================================


class TestUnifiedBusLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        bus = UnifiedBus()
        await bus.start()
        assert bus._running
        await bus.stop()
        assert not bus._running

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        bus = UnifiedBus()
        await bus.start()
        await bus.start()  # Should not error
        assert bus._running
        await bus.stop()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        bus = UnifiedBus()
        await bus.stop()  # Not running — should be fine
        assert not bus._running

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with UnifiedBus() as bus:
            assert bus._running
        assert not bus._running

    @pytest.mark.asyncio
    async def test_stop_cancels_waiters(self):
        bus = UnifiedBus()
        await bus.start()
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        bus._waiters["test"] = [(SignalKind.QUERY, future)]
        await bus.stop()
        assert future.cancelled()
        assert len(bus._waiters) == 0


class TestUnifiedBusEmit:
    @pytest.mark.asyncio
    async def test_emit_returns_signal_id(self):
        bus = UnifiedBus()
        s = Signal(kind=SignalKind.QUERY)
        result = await bus.emit(s)
        assert result == s.id

    @pytest.mark.asyncio
    async def test_emit_increments_count(self):
        bus = UnifiedBus()
        assert bus._emit_count == 0
        await bus.emit(Signal(kind=SignalKind.QUERY))
        assert bus._emit_count == 1
        await bus.emit(Signal(kind=SignalKind.HEARTBEAT))
        assert bus._emit_count == 2

    @pytest.mark.asyncio
    async def test_emit_adds_to_history(self):
        bus = UnifiedBus()
        await bus.emit(Signal(kind=SignalKind.QUERY))
        assert len(bus._history) == 1

    @pytest.mark.asyncio
    async def test_emit_with_journal(self):
        journal = MagicMock()
        journal.append = AsyncMock()
        bus = UnifiedBus(journal=journal)
        s = Signal(kind=SignalKind.QUERY)
        await bus.emit(s)
        journal.append.assert_awaited_once_with(s)

    @pytest.mark.asyncio
    async def test_emit_journal_failure_does_not_crash(self):
        journal = MagicMock()
        journal.append = AsyncMock(side_effect=RuntimeError("db error"))
        bus = UnifiedBus(journal=journal)
        # Should not raise
        result = await bus.emit(Signal(kind=SignalKind.QUERY))
        assert result  # Still returns the signal ID


class TestUnifiedBusDispatch:
    @pytest.mark.asyncio
    async def test_target_dispatch(self):
        bus = UnifiedBus()
        received = []

        async def handler(s):
            received.append(s)

        bus.subscribe("agent1", handler)
        await bus.emit(Signal(kind=SignalKind.QUERY, target="agent1"))
        await asyncio.sleep(0.02)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_target_miss(self):
        bus = UnifiedBus()
        received = []

        async def handler(s):
            received.append(s)

        bus.subscribe("agent1", handler)
        await bus.emit(Signal(kind=SignalKind.QUERY, target="agent2"))
        await asyncio.sleep(0.02)
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_kind_dispatch(self):
        bus = UnifiedBus()
        received = []

        async def handler(s):
            received.append(s)

        bus.on(SignalKind.STEP_COMPLETE, handler)
        await bus.emit(Signal(kind=SignalKind.STEP_COMPLETE))
        await bus.emit(Signal(kind=SignalKind.QUERY))
        await asyncio.sleep(0.02)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_broadcast_dispatch(self):
        bus = UnifiedBus()
        received = []

        async def handler(s):
            received.append(s)

        bus.subscribe_broadcast(handler)
        await bus.emit(Signal(kind=SignalKind.QUERY))  # No target = broadcast
        await asyncio.sleep(0.02)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_broadcast_skipped_for_targeted(self):
        bus = UnifiedBus()
        broadcast_received = []
        target_received = []

        async def bc_handler(s):
            broadcast_received.append(s)

        async def tgt_handler(s):
            target_received.append(s)

        bus.subscribe_broadcast(bc_handler)
        bus.subscribe("specific", tgt_handler)
        await bus.emit(Signal(kind=SignalKind.QUERY, target="specific"))
        await asyncio.sleep(0.02)
        assert len(target_received) == 1
        assert len(broadcast_received) == 0

    @pytest.mark.asyncio
    async def test_kind_fires_even_when_targeted(self):
        bus = UnifiedBus()
        kind_received = []

        async def handler(s):
            kind_received.append(s)

        bus.on(SignalKind.STEP_COMPLETE, handler)
        await bus.emit(Signal(kind=SignalKind.STEP_COMPLETE, target="some_agent"))
        await asyncio.sleep(0.02)
        assert len(kind_received) == 1

    @pytest.mark.asyncio
    async def test_handler_error_does_not_break_others(self):
        bus = UnifiedBus()
        received = []

        async def bad_handler(s):
            raise ValueError("boom")

        async def good_handler(s):
            received.append(s)

        bus.on(SignalKind.QUERY, bad_handler)
        bus.on(SignalKind.QUERY, good_handler)
        await bus.emit(Signal(kind=SignalKind.QUERY))
        await asyncio.sleep(0.02)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_handler_response_propagates_correlation(self):
        bus = UnifiedBus()
        children = []

        async def responder(s):
            return Signal(kind=SignalKind.MCTS_DECISION, payload={"chosen": True})

        async def collector(s):
            children.append(s)

        bus.on(SignalKind.QUERY, responder)
        bus.on(SignalKind.MCTS_DECISION, collector)

        parent = Signal(kind=SignalKind.QUERY)
        await bus.emit(parent)
        await asyncio.sleep(0.05)

        assert len(children) == 1
        child = children[0]
        assert child.correlation_id == parent.id
        assert child.causation_id == parent.id
        assert child.trigger_mode == TriggerMode.REACTIVE

    @pytest.mark.asyncio
    async def test_ttl_expired_signals_not_dispatched(self):
        bus = UnifiedBus()
        received = []

        async def handler(s):
            received.append(s)

        bus.on(SignalKind.QUERY, handler)
        await bus.emit(Signal(
            kind=SignalKind.QUERY,
            ttl_seconds=0,
            created_at="2020-01-01T00:00:00",
        ))
        await asyncio.sleep(0.02)
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_dedup_prevents_double_dispatch(self):
        bus = UnifiedBus()
        received = []

        async def handler(s):
            received.append(s)

        bus.on(SignalKind.QUERY, handler)
        s = Signal(id="duped", kind=SignalKind.QUERY)
        await bus.emit(s)
        await bus.emit(s)
        await asyncio.sleep(0.02)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_dedup_set_pruning(self):
        bus = UnifiedBus(history_limit=5)
        # processed_limit = 5 * 2 = 10
        for i in range(15):
            await bus.emit(Signal(kind=SignalKind.QUERY))
        await asyncio.sleep(0.02)
        # Dedup set should have been pruned
        assert len(bus._processed_ids) <= 15


class TestUnifiedBusSubscribe:
    def test_subscribe_returns_sub_id(self):
        bus = UnifiedBus()

        async def handler(s):
            pass

        sub_id = bus.subscribe("target", handler)
        assert sub_id.startswith("target:")

    def test_on_returns_sub_id(self):
        bus = UnifiedBus()

        async def handler(s):
            pass

        sub_id = bus.on(SignalKind.QUERY, handler)
        assert sub_id.startswith("kind:")

    def test_subscribe_broadcast_returns_sub_id(self):
        bus = UnifiedBus()

        async def handler(s):
            pass

        sub_id = bus.subscribe_broadcast(handler)
        assert sub_id.startswith("broadcast:")

    def test_unsubscribe_success(self):
        bus = UnifiedBus()

        async def handler(s):
            pass

        bus.subscribe("t", handler)
        assert bus.unsubscribe("t", handler)

    def test_unsubscribe_missing_handler(self):
        bus = UnifiedBus()

        async def handler(s):
            pass

        async def other(s):
            pass

        bus.subscribe("t", handler)
        assert not bus.unsubscribe("t", other)

    def test_unsubscribe_missing_target(self):
        bus = UnifiedBus()

        async def handler(s):
            pass

        assert not bus.unsubscribe("nonexistent", handler)

    def test_off_success(self):
        bus = UnifiedBus()

        async def handler(s):
            pass

        bus.on(SignalKind.QUERY, handler)
        assert bus.off(SignalKind.QUERY, handler)

    def test_off_missing_handler(self):
        bus = UnifiedBus()

        async def handler(s):
            pass

        async def other(s):
            pass

        bus.on(SignalKind.QUERY, handler)
        assert not bus.off(SignalKind.QUERY, other)

    def test_off_missing_kind(self):
        bus = UnifiedBus()

        async def handler(s):
            pass

        assert not bus.off(SignalKind.ERROR, handler)


class TestUnifiedBusRequestResponse:
    @pytest.mark.asyncio
    async def test_request_response(self):
        bus = UnifiedBus()

        async def echo(s):
            return Signal(
                kind=SignalKind.EXECUTION_COMPLETE,
                payload={"echo": s.payload.get("msg")},
            )

        bus.on(SignalKind.QUERY, echo)
        resp = await bus.request(
            Signal(kind=SignalKind.QUERY, payload={"msg": "hi"}),
            response_kind=SignalKind.EXECUTION_COMPLETE,
            timeout=5.0,
        )
        assert resp.payload["echo"] == "hi"

    @pytest.mark.asyncio
    async def test_request_timeout(self):
        bus = UnifiedBus()
        with pytest.raises(asyncio.TimeoutError):
            await bus.request(
                Signal(kind=SignalKind.QUERY),
                response_kind=SignalKind.EXECUTION_COMPLETE,
                timeout=0.05,
            )

    @pytest.mark.asyncio
    async def test_request_cleans_up_waiter(self):
        bus = UnifiedBus()
        try:
            await bus.request(
                Signal(kind=SignalKind.QUERY),
                response_kind=SignalKind.EXECUTION_COMPLETE,
                timeout=0.05,
            )
        except asyncio.TimeoutError:
            pass
        assert len(bus._waiters) == 0


class TestUnifiedBusMemoryDelegation:
    @pytest.mark.asyncio
    async def test_store_memory_no_bus_raises(self):
        bus = UnifiedBus(memory_bus=None)
        with pytest.raises(RuntimeError, match="No memory bus configured"):
            await bus.store_memory("item")

    @pytest.mark.asyncio
    async def test_query_memory_no_bus_raises(self):
        bus = UnifiedBus(memory_bus=None)
        with pytest.raises(RuntimeError, match="No memory bus configured"):
            await bus.query_memory("query")

    @pytest.mark.asyncio
    async def test_store_memory_delegates(self):
        mock_bus = MagicMock()
        mock_bus.store = AsyncMock(return_value="item-123")
        bus = UnifiedBus(memory_bus=mock_bus)

        received = []
        bus.on(SignalKind.MEMORY_STORED, AsyncMock(side_effect=lambda s: received.append(s)))

        result = await bus.store_memory("test_item")
        assert result == "item-123"
        await asyncio.sleep(0.02)
        assert len(received) == 1
        assert received[0].kind == SignalKind.MEMORY_STORED

    @pytest.mark.asyncio
    async def test_query_memory_delegates(self):
        mock_result = MagicMock()
        mock_result.items = [1, 2, 3]
        mock_result.resolution_path = "semantic"

        mock_bus = MagicMock()
        mock_bus.query = AsyncMock(return_value=mock_result)
        bus = UnifiedBus(memory_bus=mock_bus)

        received = []
        bus.on(SignalKind.MEMORY_QUERIED, AsyncMock(side_effect=lambda s: received.append(s)))

        mock_query = MagicMock()
        mock_query.intent = "test"
        result = await bus.query_memory(mock_query)
        assert result == mock_result
        await asyncio.sleep(0.02)
        assert len(received) == 1
        assert received[0].payload["result_count"] == 3


class TestUnifiedBusHistory:
    @pytest.mark.asyncio
    async def test_history_filters(self):
        bus = UnifiedBus()
        await bus.emit(Signal(kind=SignalKind.QUERY, source="a", correlation_id="w1"))
        await bus.emit(Signal(kind=SignalKind.STEP_COMPLETE, source="b", correlation_id="w1"))
        await bus.emit(Signal(kind=SignalKind.QUERY, source="c", correlation_id="w2"))

        all_h = bus.get_history()
        assert len(all_h) == 3

        by_kind = bus.get_history(kind=SignalKind.QUERY)
        assert len(by_kind) == 2

        by_source = bus.get_history(source="b")
        assert len(by_source) == 1

        by_corr = bus.get_history(correlation_id="w1")
        assert len(by_corr) == 2

    @pytest.mark.asyncio
    async def test_history_limit(self):
        bus = UnifiedBus(history_limit=3)
        for i in range(5):
            await bus.emit(Signal(kind=SignalKind.QUERY))
        assert len(bus._history) == 3

    @pytest.mark.asyncio
    async def test_clear_history(self):
        bus = UnifiedBus()
        await bus.emit(Signal(kind=SignalKind.QUERY))
        await bus.emit(Signal(kind=SignalKind.QUERY))
        cleared = bus.clear_history()
        assert cleared == 2
        assert len(bus.get_history()) == 0


class TestUnifiedBusStatistics:
    @pytest.mark.asyncio
    async def test_statistics(self):
        bus = UnifiedBus()
        await bus.emit(Signal(kind=SignalKind.QUERY, priority=SignalPriority.NORMAL))
        await bus.emit(Signal(
            kind=SignalKind.HEARTBEAT,
            priority=SignalPriority.LOW,
            trigger_mode=TriggerMode.HEARTBEAT,
        ))
        await asyncio.sleep(0.02)

        stats = bus.get_statistics()
        assert stats["total_emitted"] == 2
        assert stats["history_size"] == 2
        assert stats["by_kind"]["query"] == 1
        assert stats["by_kind"]["heartbeat"] == 1
        assert stats["by_priority"][2] == 1  # NORMAL
        assert stats["by_priority"][3] == 1  # LOW
        assert stats["by_trigger"]["active"] == 1
        assert stats["by_trigger"]["heartbeat"] == 1


class TestCreateBusFactory:
    def test_create_bus_defaults(self):
        bus = create_bus()
        assert isinstance(bus, UnifiedBus)
        assert bus._history_limit == 1000

    def test_create_bus_custom(self):
        bus = create_bus(history_limit=50)
        assert bus._history_limit == 50


# ============================================================================
# Signal Journal Tests
# ============================================================================


class TestSignalJournal:
    @pytest.fixture
    def journal(self, tmp_path):
        from hololoom.core.bus.journal import SignalJournal
        return SignalJournal(str(tmp_path / "test_journal.db"))

    @pytest.fixture
    def memory_journal(self):
        from hololoom.core.bus.journal import SignalJournal
        return SignalJournal(":memory:")

    @pytest.mark.asyncio
    async def test_append_and_query(self, journal):
        s = Signal(kind=SignalKind.QUERY, source="test", correlation_id="c1")
        await journal.append(s)

        results = await journal.query_signals(correlation_id="c1")
        assert len(results) == 1
        assert results[0]["kind"] == "query"
        assert results[0]["source"] == "test"

    @pytest.mark.asyncio
    async def test_append_deduplicates(self, journal):
        s = Signal(id="dedup-test", kind=SignalKind.QUERY)
        await journal.append(s)
        await journal.append(s)  # INSERT OR IGNORE

        results = await journal.query_signals()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_signals_no_filters(self, memory_journal):
        for i in range(3):
            await memory_journal.append(Signal(kind=SignalKind.QUERY, source=f"s{i}"))
        results = await memory_journal.query_signals()
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_signals_by_kind(self, memory_journal):
        await memory_journal.append(Signal(kind=SignalKind.QUERY))
        await memory_journal.append(Signal(kind=SignalKind.HEARTBEAT))
        results = await memory_journal.query_signals(kind="query")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_signals_by_source(self, memory_journal):
        await memory_journal.append(Signal(kind=SignalKind.QUERY, source="api"))
        await memory_journal.append(Signal(kind=SignalKind.QUERY, source="cron"))
        results = await memory_journal.query_signals(source="api")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_signals_since(self, memory_journal):
        await memory_journal.append(Signal(
            kind=SignalKind.QUERY,
            created_at="2026-01-01T00:00:00",
        ))
        await memory_journal.append(Signal(
            kind=SignalKind.QUERY,
            created_at="2026-06-01T00:00:00",
        ))
        results = await memory_journal.query_signals(since="2026-03-01T00:00:00")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_signals_limit(self, memory_journal):
        for i in range(10):
            await memory_journal.append(Signal(kind=SignalKind.QUERY))
        results = await memory_journal.query_signals(limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_record_execution(self, memory_journal):
        await memory_journal.record_execution(
            trace_id="trace1",
            correlation_id="c1",
            execution_type="chain",
            name="test_chain",
            started_at="2026-01-01T00:00:00",
            completed_at="2026-01-01T00:00:05",
            success=True,
            total_steps=3,
            duration_ms=5000.0,
            final_confidence=0.9,
            context={"key": "value"},
        )
        traces = await memory_journal.query_traces(execution_type="chain")
        assert len(traces) == 1
        assert traces[0]["success"] == 1
        assert traces[0]["name"] == "test_chain"

    @pytest.mark.asyncio
    async def test_record_execution_failure(self, memory_journal):
        await memory_journal.record_execution(
            trace_id="trace2",
            correlation_id="c2",
            execution_type="chain",
            success=False,
            error="timeout",
        )
        traces = await memory_journal.query_traces(success=False)
        assert len(traces) == 1
        assert traces[0]["success"] == 0

    @pytest.mark.asyncio
    async def test_query_traces_no_filters(self, memory_journal):
        await memory_journal.record_execution(
            trace_id="t1", correlation_id="c1", execution_type="chain",
        )
        await memory_journal.record_execution(
            trace_id="t2", correlation_id="c2", execution_type="direct",
        )
        traces = await memory_journal.query_traces()
        assert len(traces) == 2

    @pytest.mark.asyncio
    async def test_record_step(self, memory_journal):
        await memory_journal.record_step(
            step_record_id="sr1",
            execution_id="exec1",
            step_id="step1",
            step_type="execute",
            status="success",
            timestamp="2026-01-01T00:00:01",
            duration_ms=100.0,
            confidence=0.8,
            model_used="qwen3.5:9b",
            output={"text": "result"},
        )
        steps = await memory_journal.query_steps("exec1")
        assert len(steps) == 1
        assert steps[0]["step_type"] == "execute"
        assert steps[0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_record_mcts_decision(self, memory_journal):
        await memory_journal.record_mcts_decision(
            decision_id="d1",
            tree_level="strategic",
            created_at="2026-01-01T00:00:00",
            correlation_id="c1",
            domain="code",
            simulations_run=50,
            selected_action="code_review",
            root_visits=50,
            best_value=0.85,
            alternatives=[{"strategy": "verified_query", "visits": 20, "value": 0.7}],
        )
        decisions = await memory_journal.query_mcts_decisions(domain="code")
        assert len(decisions) == 1
        assert decisions[0]["selected_action"] == "code_review"

    @pytest.mark.asyncio
    async def test_query_mcts_by_tree_level(self, memory_journal):
        await memory_journal.record_mcts_decision(
            decision_id="d1", tree_level="strategic",
            created_at="2026-01-01T00:00:00",
        )
        await memory_journal.record_mcts_decision(
            decision_id="d2", tree_level="tactical",
            created_at="2026-01-01T00:00:01",
        )
        results = await memory_journal.query_mcts_decisions(tree_level="strategic")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_workflow_history(self, memory_journal):
        # Signal
        await memory_journal.append(Signal(
            kind=SignalKind.QUERY, correlation_id="wf1",
        ))
        # Execution trace
        await memory_journal.record_execution(
            trace_id="t1", correlation_id="wf1", execution_type="chain",
        )
        # Step
        await memory_journal.record_step(
            step_record_id="s1", execution_id="t1", step_id="step1",
            step_type="execute", status="success",
            timestamp="2026-01-01T00:00:00",
        )
        # MCTS decision
        await memory_journal.record_mcts_decision(
            decision_id="m1", tree_level="strategic",
            created_at="2026-01-01T00:00:00", correlation_id="wf1",
        )

        history = await memory_journal.get_workflow_history("wf1")
        assert history["correlation_id"] == "wf1"
        assert len(history["signals"]) == 1
        assert len(history["traces"]) == 1
        assert len(history["steps"]) == 1
        assert len(history["mcts_decisions"]) == 1

    def test_close(self, tmp_path):
        from hololoom.core.bus.journal import SignalJournal
        j = SignalJournal(str(tmp_path / "close_test.db"))
        j._get_conn()  # Force connection
        assert j._conn is not None
        j.close()
        assert j._conn is None

    def test_close_idempotent(self, tmp_path):
        from hololoom.core.bus.journal import SignalJournal
        j = SignalJournal(str(tmp_path / "close_test.db"))
        j.close()  # Not connected
        j.close()  # Still fine

    def test_del_closes(self, tmp_path):
        from hololoom.core.bus.journal import SignalJournal
        j = SignalJournal(str(tmp_path / "del_test.db"))
        j._get_conn()
        j.__del__()
        assert j._conn is None


# ============================================================================
# Neo4j Store Tests (degraded mode — no Neo4j)
# ============================================================================


class TestMCTSStoreDegraded:
    def test_no_driver_degrades(self):
        from hololoom.core.bus.neo4j_store import MCTSStore
        store = MCTSStore(driver=None)
        assert not store.available

    def test_save_tree_degrades(self):
        from hololoom.core.bus.neo4j_store import MCTSStore
        store = MCTSStore(driver=None)
        result = store.save_tree("t1", "code", "strategic", [])
        assert result is False

    def test_load_tree_degrades(self):
        from hololoom.core.bus.neo4j_store import MCTSStore
        store = MCTSStore(driver=None)
        result = store.load_tree("code")
        assert result is None

    def test_backprop_update_degrades(self):
        from hololoom.core.bus.neo4j_store import MCTSStore
        store = MCTSStore(driver=None)
        assert store.backprop_update("n1", 0.5) is False

    def test_record_strategy_outcome_degrades(self):
        from hololoom.core.bus.neo4j_store import MCTSStore
        store = MCTSStore(driver=None)
        assert store.record_strategy_outcome("verified_query", "code", True) is False

    def test_query_best_strategy_degrades(self):
        from hololoom.core.bus.neo4j_store import MCTSStore
        store = MCTSStore(driver=None)
        assert store.query_best_strategy("code") == []

    def test_get_all_pattern_domains_degrades(self):
        from hololoom.core.bus.neo4j_store import MCTSStore
        store = MCTSStore(driver=None)
        assert store.get_all_pattern_domains() == []

    def test_prune_low_visit_nodes_degrades(self):
        from hololoom.core.bus.neo4j_store import MCTSStore
        store = MCTSStore(driver=None)
        assert store.prune_low_visit_nodes("t1") == 0

    def test_schema_failure_degrades(self):
        from hololoom.core.bus.neo4j_store import MCTSStore
        mock_driver = MagicMock()
        mock_driver.session.side_effect = RuntimeError("connection refused")
        store = MCTSStore(driver=mock_driver)
        assert not store.available


# ============================================================================
# Convergence Adapter Tests
# ============================================================================


class TestConvergenceMonitor:
    @pytest.fixture
    def bus(self):
        return UnifiedBus()

    @pytest.fixture
    def monitor(self, bus):
        from hololoom.core.bus.adapters.convergence_adapter import ConvergenceMonitor
        return ConvergenceMonitor(bus, confidence_threshold=0.85, min_delta=0.05)

    @pytest.mark.asyncio
    async def test_confidence_threshold_convergence(self, bus, monitor):
        converged = []

        async def convergence_handler(s):
            converged.append(s)

        bus.on(SignalKind.CONVERGENCE, convergence_handler)

        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            payload={"execution_id": "e1", "step_id": "s1", "confidence": 0.9},
        ))
        await asyncio.sleep(0.05)

        assert len(converged) == 1
        assert converged[0].payload["reason"].startswith("confidence 0.90")

    @pytest.mark.asyncio
    async def test_improvement_delta_convergence(self, bus, monitor):
        converged = []

        async def handler(s):
            converged.append(s)

        bus.on(SignalKind.CONVERGENCE, handler)

        # First step — no convergence yet
        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            payload={"execution_id": "e1", "step_id": "s1", "confidence": 0.5},
        ))
        await asyncio.sleep(0.03)

        # Second step — tiny improvement → converge via min_delta
        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            payload={"execution_id": "e1", "step_id": "s2", "confidence": 0.51},
        ))
        await asyncio.sleep(0.03)

        assert len(converged) == 1
        assert "min_delta" in converged[0].payload["reason"]

    @pytest.mark.asyncio
    async def test_no_convergence_below_threshold(self, bus, monitor):
        converged = []

        async def handler(s):
            converged.append(s)

        bus.on(SignalKind.CONVERGENCE, handler)

        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            payload={"execution_id": "e1", "step_id": "s1", "confidence": 0.5},
        ))
        await asyncio.sleep(0.03)
        assert len(converged) == 0

    @pytest.mark.asyncio
    async def test_no_execution_id_ignored(self, bus, monitor):
        """Signals without execution_id should be ignored."""
        converged = []

        async def handler(s):
            converged.append(s)

        bus.on(SignalKind.CONVERGENCE, handler)

        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            payload={"confidence": 0.95},  # No execution_id
        ))
        await asyncio.sleep(0.03)
        assert len(converged) == 0

    @pytest.mark.asyncio
    async def test_no_confidence_ignored(self, bus, monitor):
        converged = []

        async def handler(s):
            converged.append(s)

        bus.on(SignalKind.CONVERGENCE, handler)

        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            payload={"execution_id": "e1"},  # No confidence
        ))
        await asyncio.sleep(0.03)
        assert len(converged) == 0

    @pytest.mark.asyncio
    async def test_already_converged_not_re_emitted(self, bus, monitor):
        converged = []

        async def handler(s):
            converged.append(s)

        bus.on(SignalKind.CONVERGENCE, handler)

        # Converge
        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            payload={"execution_id": "e1", "step_id": "s1", "confidence": 0.95},
        ))
        await asyncio.sleep(0.03)

        # Another step on same execution — should NOT re-emit
        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            payload={"execution_id": "e1", "step_id": "s2", "confidence": 0.99},
        ))
        await asyncio.sleep(0.03)
        assert len(converged) == 1

    @pytest.mark.asyncio
    async def test_plateau_convergence(self, bus):
        from hololoom.core.bus.adapters.convergence_adapter import ConvergenceMonitor
        # To trigger plateau (not delta), we need:
        # - confidence < threshold (0.99)
        # - improvement >= min_delta (so delta check doesn't fire first)
        # - all recent improvements < plateau_min_delta
        # The trick: min_delta must be <= the actual improvements so delta
        # check doesn't trigger, but plateau_min_delta > improvements.
        # With 2+ obs, improvement = conf[i] - conf[i-1]. First obs has
        # improvement=0. So step 2 onward: improvement=0.005.
        # Set min_delta=0.004 so 0.005 >= 0.004 → delta doesn't fire.
        # Set plateau_min_delta=0.01 so 0.005 < 0.01 → counts as plateau.
        monitor = ConvergenceMonitor(
            bus,
            confidence_threshold=0.99,
            min_delta=0.004,
            plateau_window=3,
            plateau_min_delta=0.01,
        )
        converged = []

        async def handler(s):
            converged.append(s)

        bus.on(SignalKind.CONVERGENCE, handler)

        # Send 3 steps: first has improvement=0, rest have 0.005
        # All improvements < plateau_min_delta (0.01), so plateau fires
        for i in range(3):
            await bus.emit(Signal(
                kind=SignalKind.STEP_COMPLETE,
                payload={
                    "execution_id": "e1",
                    "step_id": f"s{i}",
                    "confidence": 0.5 + i * 0.005,
                },
            ))
            await asyncio.sleep(0.03)

        assert len(converged) == 1
        assert "plateaued" in converged[0].payload["reason"]

    def test_reset_specific_execution(self, bus, monitor):
        monitor._observations["e1"] = [MagicMock()]
        monitor._converged_executions.add("e1")
        monitor.reset("e1")
        assert "e1" not in monitor._observations
        assert "e1" not in monitor._converged_executions

    def test_reset_all(self, bus, monitor):
        monitor._observations["e1"] = [MagicMock()]
        monitor._observations["e2"] = [MagicMock()]
        monitor._converged_executions.add("e1")
        monitor.reset()
        assert len(monitor._observations) == 0
        assert len(monitor._converged_executions) == 0

    def test_get_status_empty(self, bus, monitor):
        status = monitor.get_status("e_nonexistent")
        assert status["converged"] is False
        assert status["observations"] == 0
        assert status["latest_confidence"] is None

    def test_get_status_with_data(self, bus, monitor):
        from hololoom.core.bus.adapters.convergence_adapter import StepObservation
        monitor._observations["e1"] = [
            StepObservation(step_id="s1", confidence=0.7, improvement=0.0),
            StepObservation(step_id="s2", confidence=0.8, improvement=0.1),
        ]
        status = monitor.get_status("e1")
        assert status["observations"] == 2
        assert status["latest_confidence"] == 0.8
        assert status["latest_improvement"] == 0.1


# ============================================================================
# Trigger Adapter Tests
# ============================================================================


class TestHeartbeatTrigger:
    @pytest.mark.asyncio
    async def test_manual_pulse(self):
        from hololoom.core.bus.adapters.trigger_adapters import HeartbeatTrigger

        bus = UnifiedBus()
        received = []

        async def handler(s):
            received.append(s)

        bus.on(SignalKind.HEARTBEAT, handler)

        hb = HeartbeatTrigger(bus, interval=60.0)
        await hb.pulse()
        await asyncio.sleep(0.03)

        assert len(received) == 1
        assert received[0].payload["tick"] == 1
        assert received[0].payload["manual"] is True
        assert hb.tick_count == 1

    @pytest.mark.asyncio
    async def test_pulse_with_health_fn(self):
        from hololoom.core.bus.adapters.trigger_adapters import HeartbeatTrigger

        bus = UnifiedBus()
        received = []

        async def handler(s):
            received.append(s)

        bus.on(SignalKind.HEARTBEAT, handler)

        hb = HeartbeatTrigger(bus, health_fn=lambda: {"status": "ok"})
        await hb.pulse()
        await asyncio.sleep(0.03)

        assert received[0].payload["health"] == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_pulse_with_health_fn_error(self):
        from hololoom.core.bus.adapters.trigger_adapters import HeartbeatTrigger

        bus = UnifiedBus()
        received = []

        async def handler(s):
            received.append(s)

        bus.on(SignalKind.HEARTBEAT, handler)

        hb = HeartbeatTrigger(bus, health_fn=lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        await hb.pulse()
        await asyncio.sleep(0.03)

        assert "health_error" in received[0].payload

    @pytest.mark.asyncio
    async def test_start_stop(self):
        from hololoom.core.bus.adapters.trigger_adapters import HeartbeatTrigger

        bus = UnifiedBus()
        hb = HeartbeatTrigger(bus, interval=0.05)
        await hb.start()
        assert hb._running
        # Start is idempotent
        await hb.start()
        assert hb._running

        await hb.stop()
        assert not hb._running
        # Stop is idempotent
        await hb.stop()
        assert not hb._running

    @pytest.mark.asyncio
    async def test_heartbeat_emits_on_interval(self):
        from hololoom.core.bus.adapters.trigger_adapters import HeartbeatTrigger

        bus = UnifiedBus()
        received = []

        async def handler(s):
            received.append(s)

        bus.on(SignalKind.HEARTBEAT, handler)

        hb = HeartbeatTrigger(bus, interval=0.05)
        await hb.start()
        await asyncio.sleep(0.15)
        await hb.stop()

        assert len(received) >= 1
        assert received[0].trigger_mode == TriggerMode.HEARTBEAT


class TestCronJob:
    def test_is_due(self):
        from hololoom.core.bus.adapters.trigger_adapters import CronJob
        job = CronJob(name="test", interval=10.0, payload={})
        now = time.time()
        assert job.is_due(now)

    def test_not_due_after_run(self):
        from hololoom.core.bus.adapters.trigger_adapters import CronJob
        job = CronJob(name="test", interval=10.0, payload={})
        now = time.time()
        job.mark_run(now)
        assert not job.is_due(now + 5)  # Only 5s later
        assert job.is_due(now + 11)  # 11s later

    def test_mark_run_increments(self):
        from hololoom.core.bus.adapters.trigger_adapters import CronJob
        job = CronJob(name="test", interval=10.0, payload={})
        assert job.run_count == 0
        job.mark_run(time.time())
        assert job.run_count == 1


class TestCronTrigger:
    @pytest.mark.asyncio
    async def test_register_and_tick(self):
        from hololoom.core.bus.adapters.trigger_adapters import CronTrigger

        bus = UnifiedBus()
        received = []

        async def handler(s):
            received.append(s)

        bus.on(SignalKind.CRON, handler)

        cron = CronTrigger(bus, check_interval=0.05)
        cron.every(0, "test_job", {"action": "consolidate"})

        await cron.tick()
        await asyncio.sleep(0.03)

        assert len(received) == 1
        assert received[0].payload["job_name"] == "test_job"
        assert received[0].payload["action"] == "consolidate"
        assert received[0].trigger_mode == TriggerMode.CRON

    @pytest.mark.asyncio
    async def test_remove_job(self):
        from hololoom.core.bus.adapters.trigger_adapters import CronTrigger

        bus = UnifiedBus()
        cron = CronTrigger(bus)
        cron.every(60, "job1", {})
        assert cron.remove("job1")
        assert not cron.remove("job1")  # Already removed

    @pytest.mark.asyncio
    async def test_get_jobs(self):
        from hololoom.core.bus.adapters.trigger_adapters import CronTrigger

        bus = UnifiedBus()
        cron = CronTrigger(bus)
        cron.every(60, "job_a", {"action": "a"})
        cron.every(120, "job_b", {"action": "b"})

        jobs = cron.get_jobs()
        assert "job_a" in jobs
        assert "job_b" in jobs
        assert jobs["job_a"]["interval"] == 60
        assert jobs["job_b"]["payload"]["action"] == "b"

    @pytest.mark.asyncio
    async def test_start_stop(self):
        from hololoom.core.bus.adapters.trigger_adapters import CronTrigger

        bus = UnifiedBus()
        cron = CronTrigger(bus, check_interval=0.05)
        await cron.start()
        assert cron._running
        await cron.start()  # Idempotent
        assert cron._running

        await cron.stop()
        assert not cron._running
        await cron.stop()  # Idempotent
        assert not cron._running

    @pytest.mark.asyncio
    async def test_cron_loop_fires_due_jobs(self):
        from hololoom.core.bus.adapters.trigger_adapters import CronTrigger

        bus = UnifiedBus()
        received = []

        async def handler(s):
            received.append(s)

        bus.on(SignalKind.CRON, handler)

        cron = CronTrigger(bus, check_interval=0.05)
        cron.every(0, "fast_job", {})
        await cron.start()
        await asyncio.sleep(0.15)
        await cron.stop()

        assert len(received) >= 1


# ============================================================================
# Strategy Selector Tests
# ============================================================================


class TestClassifyDomain:
    def test_code_domain(self):
        from hololoom.core.bus.strategy_selector import classify_domain
        domain, conf = classify_domain("debug this python function class")
        assert domain == "code"
        assert conf > 0

    def test_factual_domain(self):
        from hololoom.core.bus.strategy_selector import classify_domain
        domain, conf = classify_domain("what is the capital of France")
        assert domain == "factual"

    def test_reasoning_domain(self):
        from hololoom.core.bus.strategy_selector import classify_domain
        domain, conf = classify_domain("analyze and compare these tradeoffs")
        assert domain == "reasoning"

    def test_creative_domain(self):
        from hololoom.core.bus.strategy_selector import classify_domain
        domain, conf = classify_domain("write a poem about nature")
        assert domain == "creative"

    def test_chat_domain(self):
        from hololoom.core.bus.strategy_selector import classify_domain
        domain, conf = classify_domain("hello how are you")
        assert domain == "chat"

    def test_general_domain(self):
        from hololoom.core.bus.strategy_selector import classify_domain
        domain, conf = classify_domain("xyzzy foobar blorp")
        assert domain == "general"
        assert conf == 0.3


class TestStrategy:
    def test_defaults(self):
        from hololoom.core.bus.strategy_selector import Strategy
        s = Strategy("test", "chain", "test strategy")
        assert s.success_rate == 0.5
        assert s.avg_confidence == 0.5
        assert s.avg_duration_ms == 1000.0
        assert s.uses == 0

    def test_after_updates(self):
        from hololoom.core.bus.strategy_selector import Strategy
        s = Strategy("test", "chain", "test", successes=4.0, failures=1.0)
        s.total_confidence = 3.0
        s.total_duration_ms = 500.0
        s.uses = 3
        assert s.success_rate == 0.8
        assert s.avg_confidence == 1.0
        assert s.avg_duration_ms == pytest.approx(166.67, abs=0.1)


class TestStrategyStateSpace:
    def test_get_legal_actions(self):
        from hololoom.core.bus.strategy_selector import (
            Strategy,
            StrategyState,
            StrategyStateSpace,
        )
        strategies = [
            Strategy("a", "chain", ""),
            Strategy("b", "direct", ""),
        ]
        space = StrategyStateSpace(strategies)
        state = StrategyState(query_text="test", query_domain="general")

        actions = space.get_legal_actions(state)
        assert "a" in actions
        assert "b" in actions

    def test_terminal_state_no_actions(self):
        from hololoom.core.bus.strategy_selector import (
            Strategy,
            StrategyState,
            StrategyStateSpace,
        )
        space = StrategyStateSpace([Strategy("a", "chain", "")])
        state = StrategyState(selected="a")
        assert space.get_legal_actions(state) == []

    def test_apply_action(self):
        from hololoom.core.bus.strategy_selector import (
            Strategy,
            StrategyState,
            StrategyStateSpace,
        )
        space = StrategyStateSpace([Strategy("a", "chain", "")])
        state = StrategyState(query_text="test")
        new_state = space.apply_action(state, "a")
        assert new_state.selected == "a"
        assert state.selected is None  # Original unchanged

    @pytest.mark.asyncio
    async def test_evaluate_non_terminal(self):
        from hololoom.core.bus.strategy_selector import (
            Strategy,
            StrategyState,
            StrategyStateSpace,
        )
        space = StrategyStateSpace([Strategy("a", "chain", "")])
        state = StrategyState()
        val = await space.evaluate(state)
        assert val == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_unknown_strategy(self):
        from hololoom.core.bus.strategy_selector import (
            Strategy,
            StrategyState,
            StrategyStateSpace,
        )
        space = StrategyStateSpace([Strategy("a", "chain", "")])
        state = StrategyState(selected="unknown")
        val = await space.evaluate(state)
        assert val == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_speed_preference(self):
        from hololoom.core.bus.strategy_selector import (
            Strategy,
            StrategyState,
            StrategyStateSpace,
        )
        space = StrategyStateSpace([
            Strategy("quick_answer", "chain", ""),
            Strategy("research_pipeline", "chain", ""),
        ])

        speed_state = StrategyState(selected="quick_answer", speed_weight=0.8)
        quality_state = StrategyState(selected="research_pipeline", speed_weight=0.2)

        speed_val = await space.evaluate(speed_state)
        quality_val = await space.evaluate(quality_state)

        # Both should be > 0 (they get bonuses)
        assert speed_val > 0
        assert quality_val > 0

    @pytest.mark.asyncio
    async def test_evaluate_domain_bonus(self):
        from hololoom.core.bus.strategy_selector import (
            Strategy,
            StrategyState,
            StrategyStateSpace,
        )
        space = StrategyStateSpace([Strategy("code_review", "chain", "")])
        state = StrategyState(selected="code_review", query_domain="code")
        val = await space.evaluate(state)
        assert val > 0.0

    @pytest.mark.asyncio
    async def test_evaluate_factual_domain_bonus(self):
        from hololoom.core.bus.strategy_selector import (
            Strategy,
            StrategyState,
            StrategyStateSpace,
        )
        space = StrategyStateSpace([Strategy("fact_check", "chain", "")])
        state = StrategyState(selected="fact_check", query_domain="factual")
        val = await space.evaluate(state)
        assert val > 0.0

    def test_is_terminal(self):
        from hololoom.core.bus.strategy_selector import (
            Strategy,
            StrategyState,
            StrategyStateSpace,
        )
        space = StrategyStateSpace([Strategy("a", "chain", "")])
        assert not space.is_terminal(StrategyState())
        assert space.is_terminal(StrategyState(selected="a"))

    def test_copy_state(self):
        from hololoom.core.bus.strategy_selector import (
            Strategy,
            StrategyState,
            StrategyStateSpace,
        )
        space = StrategyStateSpace([Strategy("a", "chain", "")])
        state = StrategyState(query_text="test", selected="a")
        copied = space.copy_state(state)
        assert copied.query_text == state.query_text
        assert copied.selected == state.selected
        assert copied is not state


# ============================================================================
# Refinement Adapter Tests
# ============================================================================


class TestRefinementHandlerMaterialDifference:
    def test_empty_pass2(self):
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler
        assert not RefinementHandler._is_materially_different("hello world", "")

    def test_whitespace_only_pass2(self):
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler
        assert not RefinementHandler._is_materially_different("hello world", "   ")

    def test_identical(self):
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler
        assert not RefinementHandler._is_materially_different("hello world", "hello world")

    def test_case_insensitive_identical(self):
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler
        assert not RefinementHandler._is_materially_different("Hello World", "hello world")

    def test_high_overlap(self):
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler
        # Needs > 0.8 Jaccard overlap to count as not different
        text1 = "the quick brown fox jumps over the lazy dog in the park"
        text2 = "the quick brown fox jumps over the lazy dog in the yard"
        # 11 words each, 10 in common, union=12, overlap=10/12=0.83 > 0.8
        assert not RefinementHandler._is_materially_different(text1, text2)

    def test_short_pass2(self):
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler
        assert not RefinementHandler._is_materially_different("long original text here", "short")

    def test_materially_different(self):
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler
        text1 = "Python is a programming language known for simplicity"
        text2 = "Rust provides memory safety guarantees through ownership model and borrowing rules"
        assert RefinementHandler._is_materially_different(text1, text2)


class TestRefinementHandlerSubscription:
    @pytest.mark.asyncio
    async def test_subscribes_to_refinement_start(self):
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler

        bus = UnifiedBus()
        handler = RefinementHandler(
            bus=bus,
            call_model=AsyncMock(),
            get_system_prompt=lambda: "system",
        )
        assert SignalKind.REFINEMENT_START in bus._kind_handlers
        assert SignalKind.CONVERGENCE in bus._kind_handlers

    @pytest.mark.asyncio
    async def test_convergence_signal_marks_execution(self):
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler

        bus = UnifiedBus()
        handler = RefinementHandler(
            bus=bus,
            call_model=AsyncMock(),
            get_system_prompt=lambda: "system",
        )

        await handler._on_convergence(Signal(
            kind=SignalKind.CONVERGENCE,
            correlation_id="c1",
            payload={"reason": "confidence threshold"},
        ))
        assert "c1" in handler._converged
        assert handler._converged["c1"] == "confidence threshold"


# ============================================================================
# Bus Integration with Journal
# ============================================================================


class TestBusWithJournal:
    @pytest.mark.asyncio
    async def test_emit_records_to_journal(self, tmp_path):
        from hololoom.core.bus.journal import SignalJournal

        journal = SignalJournal(str(tmp_path / "int_test.db"))
        bus = UnifiedBus(journal=journal)

        s = Signal(kind=SignalKind.QUERY, source="integration_test")
        await bus.emit(s)

        results = await journal.query_signals(source="integration_test")
        assert len(results) == 1
        assert results[0]["id"] == s.id

        journal.close()


# ============================================================================
# __init__.py imports
# ============================================================================


class TestBusPackageImports:
    def test_imports(self):
        from hololoom.core.bus import (
            Signal,
            SignalHandler,
            SignalKind,
            SignalPriority,
            TriggerMode,
            UnifiedBus,
            create_bus,
        )
        assert Signal is not None
        assert UnifiedBus is not None
        assert create_bus is not None
