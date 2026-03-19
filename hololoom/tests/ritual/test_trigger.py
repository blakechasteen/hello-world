"""Tests for RitualTrigger — the bus ↔ registry bridge."""

import pytest

from hololoom.core.bus.signal import Signal, SignalKind, SignalPriority, TriggerMode
from hololoom.core.bus.unified_bus import UnifiedBus
from hololoom.core.bus.adapters.trigger_adapters import (
    RitualTrigger,
    RitualContextFactory,
)
from hololoom.core.ritual.types import (
    Ritual,
    RitualBudget,
    RitualResult,
    HookBinding,
    Priority,
    TokenBudget,
)
from hololoom.core.ritual.hooks import HookEvent
from hololoom.core.ritual.policies import FailurePolicy
from hololoom.core.ritual.registry import RitualRegistry


def _make_ritual(ritual_id="test_ritual", body=None):
    async def default_body(ctx):
        return RitualResult(success=True, produced={"TestBobbin": 1}, confidence_delta=0.01)

    return Ritual(
        id=ritual_id,
        version="1.0.0",
        domain="coding",
        intent="test ritual",
        author="system",
        consumes=[],
        produces=[],
        failure_policy=FailurePolicy.SKIP,
        budget=RitualBudget(max_tokens=100, max_seconds=5.0, priority_class="standard"),
        body=body or default_body,
    )


def _bind(ritual, trigger=HookEvent.TASK_RECEIVED, priority=Priority.BLOCKING, condition=None):
    return HookBinding(ritual=ritual, trigger=trigger, condition=condition, priority=priority)


def _make_signal(kind=SignalKind.QUERY, payload=None):
    return Signal(
        kind=kind,
        priority=SignalPriority.NORMAL,
        source="test",
        payload=payload or {"task_intent": "fix bug", "agent_id": "agent-1"},
        trigger_mode=TriggerMode.ACTIVE,
    )


class TestRitualContextFactory:
    def test_default_extraction(self):
        factory = RitualContextFactory()
        signal = _make_signal(payload={
            "agent_id": "coder-1",
            "agent_role": "coder",
            "domain": "coding",
            "task_intent": "refactor module",
            "session_id": "sess-42",
        })
        ctx = factory.from_signal(signal)
        assert ctx.agent_id == "coder-1"
        assert ctx.agent_role == "coder"
        assert ctx.domain == "coding"
        assert ctx.task_intent == "refactor module"
        assert ctx.session_id == "sess-42"

    def test_fallback_defaults(self):
        factory = RitualContextFactory()
        signal = Signal(kind=SignalKind.QUERY, payload={}, metadata={})
        ctx = factory.from_signal(signal)
        assert ctx.agent_id == "unknown"
        assert ctx.domain == "universal"
        assert ctx.task_complexity == "standard"
        assert ctx.token_budget.total == 5000

    def test_metadata_fallback(self):
        factory = RitualContextFactory()
        signal = Signal(
            kind=SignalKind.QUERY,
            payload={},
            metadata={"agent_id": "from-meta", "session_id": "meta-sess"},
        )
        ctx = factory.from_signal(signal)
        assert ctx.agent_id == "from-meta"
        assert ctx.session_id == "meta-sess"

    def test_task_type_from_signal_kind(self):
        factory = RitualContextFactory()
        signal = _make_signal(kind=SignalKind.EXECUTION_COMPLETE, payload={})
        ctx = factory.from_signal(signal)
        assert ctx.task_type == "execution_complete"


class TestRitualTriggerWiring:
    @pytest.mark.asyncio
    async def test_signal_fires_ritual(self):
        """QUERY signal → TASK_RECEIVED → ritual fires."""
        bus = UnifiedBus()
        registry = RitualRegistry()
        ritual = _make_ritual()
        registry.register(_bind(ritual, trigger=HookEvent.TASK_RECEIVED))

        trigger = RitualTrigger(bus, registry)
        await trigger.start()

        signal = _make_signal(kind=SignalKind.QUERY)
        result_signal = await trigger._handle(signal)

        assert result_signal is not None
        assert result_signal.kind == SignalKind.RITUAL
        assert result_signal.payload["hook"] == "TASK_RECEIVED"
        assert result_signal.payload["results"]["test_ritual"] is True
        assert trigger.dispatch_count == 1

    @pytest.mark.asyncio
    async def test_execution_complete_fires_task_completed(self):
        """EXECUTION_COMPLETE → TASK_COMPLETED."""
        bus = UnifiedBus()
        registry = RitualRegistry()
        ritual = _make_ritual(ritual_id="on_complete")
        registry.register(_bind(ritual, trigger=HookEvent.TASK_COMPLETED))

        trigger = RitualTrigger(bus, registry)
        await trigger.start()

        signal = _make_signal(kind=SignalKind.EXECUTION_COMPLETE)
        result_signal = await trigger._handle(signal)

        assert result_signal is not None
        assert result_signal.payload["hook"] == "TASK_COMPLETED"

    @pytest.mark.asyncio
    async def test_memory_stored_fires_memory_write(self):
        """MEMORY_STORED → MEMORY_WRITE (memory_write_guard)."""
        bus = UnifiedBus()
        registry = RitualRegistry()
        ritual = _make_ritual(ritual_id="mem_guard")
        registry.register(_bind(ritual, trigger=HookEvent.MEMORY_WRITE))

        trigger = RitualTrigger(bus, registry)
        await trigger.start()

        signal = _make_signal(kind=SignalKind.MEMORY_STORED)
        result_signal = await trigger._handle(signal)

        assert result_signal is not None
        assert result_signal.payload["hook"] == "MEMORY_WRITE"

    @pytest.mark.asyncio
    async def test_unmapped_signal_ignored(self):
        """Signals not in the map return None."""
        bus = UnifiedBus()
        registry = RitualRegistry()
        trigger = RitualTrigger(bus, registry)
        await trigger.start()

        signal = _make_signal(kind=SignalKind.BREAKTHROUGH)
        result_signal = await trigger._handle(signal)
        assert result_signal is None

    @pytest.mark.asyncio
    async def test_no_bindings_returns_none(self):
        """Signal maps to a HookEvent with no bindings → no RITUAL signal."""
        bus = UnifiedBus()
        registry = RitualRegistry()  # empty
        trigger = RitualTrigger(bus, registry)
        await trigger.start()

        signal = _make_signal(kind=SignalKind.QUERY)
        result_signal = await trigger._handle(signal)
        assert result_signal is None
        assert trigger.dispatch_count == 1  # still counts as a dispatch

    @pytest.mark.asyncio
    async def test_graceful_degradation_no_registry(self):
        """If registry is None, handle returns None without crashing."""
        bus = UnifiedBus()
        trigger = RitualTrigger(bus, registry=None)
        await trigger.start()

        signal = _make_signal(kind=SignalKind.QUERY)
        result_signal = await trigger._handle(signal)
        assert result_signal is None

    @pytest.mark.asyncio
    async def test_stopped_trigger_ignores_signals(self):
        """After stop(), signals are ignored."""
        bus = UnifiedBus()
        registry = RitualRegistry()
        ritual = _make_ritual()
        registry.register(_bind(ritual))

        trigger = RitualTrigger(bus, registry)
        await trigger.start()
        await trigger.stop()

        signal = _make_signal(kind=SignalKind.QUERY)
        result_signal = await trigger._handle(signal)
        assert result_signal is None

    @pytest.mark.asyncio
    async def test_derived_signal_has_correlation(self):
        """The RITUAL signal inherits the correlation chain from the source signal."""
        bus = UnifiedBus()
        registry = RitualRegistry()
        ritual = _make_ritual()
        registry.register(_bind(ritual))

        trigger = RitualTrigger(bus, registry)
        await trigger.start()

        signal = Signal(
            kind=SignalKind.QUERY,
            payload={"task_intent": "test"},
            correlation_id="workflow-123",
        )
        result_signal = await trigger._handle(signal)

        assert result_signal is not None
        assert result_signal.correlation_id == "workflow-123"
        assert result_signal.causation_id == signal.id

    @pytest.mark.asyncio
    async def test_ritual_failure_does_not_crash(self):
        """If a ritual raises, RitualTrigger logs and returns None."""
        async def failing_body(ctx):
            raise ValueError("boom")

        bus = UnifiedBus()
        registry = RitualRegistry()
        # HALT policy would raise RuntimeError from registry.fire()
        ritual = _make_ritual(body=failing_body)
        ritual = Ritual(
            id="halter",
            version="1.0.0",
            domain="coding",
            intent="test",
            author="system",
            consumes=[],
            produces=[],
            failure_policy=FailurePolicy.HALT,
            budget=RitualBudget(max_tokens=100, max_seconds=5.0, priority_class="standard"),
            body=failing_body,
        )
        registry.register(_bind(ritual))

        trigger = RitualTrigger(bus, registry)
        await trigger.start()

        signal = _make_signal(kind=SignalKind.QUERY)
        result_signal = await trigger._handle(signal)
        assert result_signal is None  # caught and logged, no crash
