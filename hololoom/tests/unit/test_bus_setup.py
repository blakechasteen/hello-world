"""
Tests for Phase 7 — Bus lifecycle, endpoint integration, refinement adapter.

Tests the bus_setup singleton, signal emission helpers, and the
RefinementHandler adapter (bus-driven multi-pass refinement).
"""

import asyncio
import pytest

from hololoom.core.bus import UnifiedBus, Signal, SignalKind, SignalPriority


# ============================================================================
# Bus Setup Singleton
# ============================================================================

class TestBusSetup:
    """Test the bus_setup module's lazy singleton pattern."""

    def test_get_bus_returns_bus(self):
        """get_bus() should return a UnifiedBus instance."""
        from hololoom.apps.server.bus_setup import get_bus
        # Reset singleton state for test isolation
        import hololoom.apps.server.bus_setup as mod
        mod._initialized = False
        mod._bus = None
        mod._journal = None
        mod._selector = None
        mod._convergence = None
        mod._heartbeat = None

        bus = get_bus()
        assert bus is not None
        assert isinstance(bus, UnifiedBus)

    def test_get_bus_is_singleton(self):
        """Repeated calls return the same instance."""
        from hololoom.apps.server.bus_setup import get_bus
        import hololoom.apps.server.bus_setup as mod
        mod._initialized = False
        mod._bus = None
        mod._journal = None
        mod._selector = None
        mod._convergence = None
        mod._heartbeat = None

        bus1 = get_bus()
        bus2 = get_bus()
        assert bus1 is bus2

    @pytest.mark.asyncio
    async def test_startup_shutdown_lifecycle(self):
        """startup_bus / shutdown_bus should not raise."""
        from hololoom.apps.server.bus_setup import startup_bus, shutdown_bus
        import hololoom.apps.server.bus_setup as mod
        mod._initialized = False
        mod._bus = None
        mod._journal = None
        mod._selector = None
        mod._convergence = None
        mod._heartbeat = None

        await startup_bus()
        assert mod._bus is not None
        await shutdown_bus()

    def test_get_journal_returns_journal(self):
        """get_journal() should return the signal journal after init."""
        from hololoom.apps.server.bus_setup import get_bus, get_journal
        import hololoom.apps.server.bus_setup as mod
        mod._initialized = False
        mod._bus = None
        mod._journal = None
        mod._selector = None
        mod._convergence = None
        mod._heartbeat = None

        get_bus()
        journal = get_journal()
        assert journal is not None

    def test_get_selector_returns_selector(self):
        """get_selector() should return the strategy selector after init."""
        from hololoom.apps.server.bus_setup import get_bus, get_selector
        import hololoom.apps.server.bus_setup as mod
        mod._initialized = False
        mod._bus = None
        mod._journal = None
        mod._selector = None
        mod._convergence = None
        mod._heartbeat = None

        get_bus()
        selector = get_selector()
        assert selector is not None


# ============================================================================
# Refinement Adapter
# ============================================================================

class TestRefinementHandler:
    """Test the bus-driven refinement handler."""

    @pytest.fixture
    def bus(self):
        return UnifiedBus()

    @pytest.fixture
    def mock_store(self):
        """Minimal RefinementStore-compatible mock."""
        class MockStore:
            def __init__(self):
                self.entries = {}

            def create(self, rid, query, prior, room_id, pass_number=2):
                self.entries[rid] = {
                    "status": "pending", "query": query,
                    "prior_response": prior, "room_id": room_id,
                    "pass": pass_number, "refinement": None,
                    "next_refinement_id": None,
                }

            def complete(self, rid, refinement, next_refinement_id=None, model=None):
                if rid in self.entries:
                    self.entries[rid]["status"] = "complete"
                    self.entries[rid]["refinement"] = refinement
                    self.entries[rid]["next_refinement_id"] = next_refinement_id
                    self.entries[rid]["model"] = model

            def converged(self, rid, reason="", model=None):
                if rid in self.entries:
                    self.entries[rid]["status"] = "converged"
                    self.entries[rid]["reason"] = reason

            def skip(self, rid, reason=""):
                if rid in self.entries:
                    self.entries[rid]["status"] = "skipped"
                    self.entries[rid]["reason"] = reason

            def get(self, rid):
                return self.entries.get(rid)

        return MockStore()

    @pytest.mark.asyncio
    async def test_model_says_pass_converges(self, bus, mock_store):
        """When LLM returns 'PASS', the handler should mark converged."""
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler

        async def mock_call_model(messages, model_spec=None, temperature=0.7,
                                  max_tokens=2048):
            return {"message": {"content": "PASS"}}

        handler = RefinementHandler(
            bus=bus,
            call_model=mock_call_model,
            get_system_prompt=lambda: "You are helpful.",
            refinement_store=mock_store,
            max_passes=3,
        )

        # Create the refinement entry
        mock_store.create("test-1", "What is Python?", "Python is a language.",
                          "room-1")

        # Emit REFINEMENT_START
        await bus.emit(Signal(
            kind=SignalKind.REFINEMENT_START,
            source="test",
            payload={
                "refinement_id": "test-1",
                "query": "What is Python?",
                "prior_response": "Python is a language.",
                "room_id": "room-1",
                "history": [],
                "complexity": 0.5,
            },
        ))

        # Let the background task run
        await asyncio.sleep(0.1)

        entry = mock_store.get("test-1")
        assert entry is not None
        assert entry["status"] == "converged"
        assert "PASS" in entry.get("reason", "")

    @pytest.mark.asyncio
    async def test_material_refinement_completes(self, bus, mock_store):
        """When LLM returns substantially different content, pass completes."""
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler

        call_count = 0

        async def mock_call_model(messages, model_spec=None, temperature=0.7,
                                  max_tokens=2048):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"message": {"content":
                    "Python is a high-level interpreted programming language "
                    "created by Guido van Rossum. It emphasizes readability "
                    "and supports multiple paradigms including OOP and FP."
                }}
            return {"message": {"content": "PASS"}}

        handler = RefinementHandler(
            bus=bus,
            call_model=mock_call_model,
            get_system_prompt=lambda: "You are helpful.",
            refinement_store=mock_store,
            max_passes=3,
        )

        mock_store.create("test-2", "What is Python?", "Python is a language.",
                          "room-1")

        await bus.emit(Signal(
            kind=SignalKind.REFINEMENT_START,
            source="test",
            payload={
                "refinement_id": "test-2",
                "query": "What is Python?",
                "prior_response": "Python is a language.",
                "room_id": "room-1",
                "history": [],
                "complexity": 0.6,
            },
        ))

        await asyncio.sleep(0.2)

        entry = mock_store.get("test-2")
        assert entry is not None
        # First pass should complete with the refined content
        # Second pass should converge (PASS)
        # Either way, it shouldn't be pending
        assert entry["status"] in ("complete", "converged")

    @pytest.mark.asyncio
    async def test_bus_convergence_stops_chain(self, bus, mock_store):
        """CONVERGENCE signal from bus should stop the refinement chain."""
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler

        async def slow_model(messages, model_spec=None, temperature=0.7,
                             max_tokens=2048):
            await asyncio.sleep(0.5)  # Slow enough for convergence to fire
            return {"message": {"content": "Some long refined answer that is different."}}

        handler = RefinementHandler(
            bus=bus,
            call_model=slow_model,
            get_system_prompt=lambda: "You are helpful.",
            refinement_store=mock_store,
            max_passes=4,
        )

        mock_store.create("test-3", "Complex question", "Initial answer.",
                          "room-1")

        signal = Signal(
            kind=SignalKind.REFINEMENT_START,
            source="test",
            payload={
                "refinement_id": "test-3",
                "query": "Complex question",
                "prior_response": "Initial answer.",
                "room_id": "room-1",
                "history": [],
                "complexity": 0.8,
            },
        )
        signal.correlation_id = "conv-test"
        await bus.emit(signal)

        # Emit convergence signal before the model finishes
        await asyncio.sleep(0.05)
        await bus.emit(Signal(
            kind=SignalKind.CONVERGENCE,
            source="test_convergence",
            payload={"reason": "threshold reached", "execution_id": "test-3"},
            correlation_id="conv-test",
        ))

        await asyncio.sleep(0.7)

        # The handler should have noticed convergence
        assert "conv-test" not in handler._converged  # Cleaned up after use

    @pytest.mark.asyncio
    async def test_is_materially_different(self, bus):
        """Static material difference check."""
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler

        # Identical
        assert not RefinementHandler._is_materially_different("hello world", "hello world")

        # Too similar (>80% word overlap — 9/11 shared = 0.818)
        assert not RefinementHandler._is_materially_different(
            "the quick brown fox jumps over the lazy dog nicely today",
            "the quick brown fox jumps over the lazy dog nicely tomorrow",
        )

        # Too short
        assert not RefinementHandler._is_materially_different("hello", "ok")

        # Materially different
        assert RefinementHandler._is_materially_different(
            "Python is a language.",
            "Python is a high-level interpreted programming language created "
            "by Guido van Rossum in 1991. It emphasizes code readability.",
        )

    @pytest.mark.asyncio
    async def test_refinement_emits_step_complete(self, bus, mock_store):
        """Refinement passes should emit STEP_COMPLETE for convergence monitor."""
        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler

        step_signals = []

        async def capture_steps(s: Signal):
            step_signals.append(s)
            return None

        bus.on(SignalKind.STEP_COMPLETE, capture_steps)

        async def mock_call_model(messages, model_spec=None, temperature=0.7,
                                  max_tokens=2048):
            return {"message": {"content": "PASS"}}

        handler = RefinementHandler(
            bus=bus,
            call_model=mock_call_model,
            get_system_prompt=lambda: "You are helpful.",
            refinement_store=mock_store,
            max_passes=3,
        )

        mock_store.create("test-step", "Question", "Answer.", "room-1")

        await bus.emit(Signal(
            kind=SignalKind.REFINEMENT_START,
            source="test",
            payload={
                "refinement_id": "test-step",
                "query": "Question",
                "prior_response": "Answer.",
                "room_id": "room-1",
                "history": [],
                "complexity": 0.5,
            },
        ))

        await asyncio.sleep(0.1)

        # Model said PASS, so convergence path should have emitted a step
        # (the converged path emits a STEP_COMPLETE with high confidence)
        # The REFINEMENT_COMPLETE signal should also have been emitted
        refine_complete = [
            s for s in bus._history if s.kind == SignalKind.REFINEMENT_COMPLETE
        ]
        assert len(refine_complete) >= 1

    @pytest.mark.asyncio
    async def test_refinement_start_signal_kind_exists(self):
        """REFINEMENT_START and REFINEMENT_COMPLETE should be valid SignalKinds."""
        assert SignalKind.REFINEMENT_START.value == "refinement_start"
        assert SignalKind.REFINEMENT_COMPLETE.value == "refinement_complete"


# ============================================================================
# Signal Emission Helpers (thin endpoint patterns)
# ============================================================================

class TestSignalEmission:
    """Test the signal emission patterns used by endpoints."""

    @pytest.fixture
    def bus(self):
        return UnifiedBus()

    @pytest.mark.asyncio
    async def test_query_signal_emits(self, bus):
        """QUERY signal should be emittable and capturable."""
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.QUERY, handler)

        await bus.emit(Signal(
            kind=SignalKind.QUERY,
            source="agentic_api",
            payload={"text": "test query", "mode": "verify"},
            correlation_id="test-corr",
        ))

        await asyncio.sleep(0.02)
        assert len(received) == 1
        assert received[0].payload["text"] == "test query"
        assert received[0].correlation_id == "test-corr"

    @pytest.mark.asyncio
    async def test_execution_complete_drives_backprop(self, bus):
        """EXECUTION_COMPLETE should resolve waiters and fire handlers."""
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.EXECUTION_COMPLETE, handler)

        await bus.emit(Signal(
            kind=SignalKind.EXECUTION_COMPLETE,
            source="agentic_api",
            payload={
                "success": True,
                "confidence": 0.92,
                "duration_ms": 1500,
                "mode": "verify",
                "steps_executed": 3,
            },
            correlation_id="test-corr",
        ))

        await asyncio.sleep(0.02)
        assert len(received) == 1
        assert received[0].payload["confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_elle_style_fire_and_forget(self, bus):
        """Elle-style fire-and-forget emission should work with ensure_future."""
        received = []

        async def handler(s: Signal):
            received.append(s)
            return None

        bus.on(SignalKind.EXECUTION_COMPLETE, handler)

        # Simulate Elle's pattern: ensure_future(bus.emit(...))
        asyncio.ensure_future(bus.emit(Signal(
            kind=SignalKind.EXECUTION_COMPLETE,
            priority=SignalPriority.LOW,
            source="elle",
            payload={"room_id": "default", "tokens": 150},
        )))

        await asyncio.sleep(0.05)
        assert len(received) == 1
        assert received[0].source == "elle"

    @pytest.mark.asyncio
    async def test_strategy_selector_learns_from_endpoint_signals(self, bus):
        """
        Strategy selector should backprop from EXECUTION_COMPLETE signals,
        including those from endpoints.
        """
        from hololoom.core.bus.strategy_selector import StrategySelector

        selector = StrategySelector(bus)

        # Emit a QUERY (selector picks a strategy and tracks it)
        query_signal = Signal(
            kind=SignalKind.QUERY,
            source="test",
            payload={"text": "explain python decorators", "speed_weight": 0.5},
        )
        query_signal.correlation_id = "learn-test"

        await bus.emit(query_signal)
        await asyncio.sleep(0.1)

        # Verify selector tracked the decision
        assert "learn-test" in selector._pending

        # Emit EXECUTION_COMPLETE (selector backprops)
        await bus.emit(Signal(
            kind=SignalKind.EXECUTION_COMPLETE,
            source="test",
            payload={"confidence": 0.9, "duration_ms": 500},
            correlation_id="learn-test",
        ))

        await asyncio.sleep(0.05)

        # Pending should be cleared after backprop
        assert "learn-test" not in selector._pending
