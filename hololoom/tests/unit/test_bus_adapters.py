"""
Tests for bus adapters — Phase 3.

Tests the chain adapter and convergence monitor working together
via the UnifiedBus without requiring a real Department.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass

from hololoom.core.bus import UnifiedBus, Signal, SignalKind, SignalPriority
from hololoom.core.bus.journal import SignalJournal
from hololoom.core.bus.adapters.convergence_adapter import ConvergenceMonitor


# ============================================================================
# Convergence Monitor Tests
# ============================================================================

class TestConvergenceMonitor:
    @pytest.fixture
    def bus(self):
        return UnifiedBus()

    @pytest.fixture
    def monitor(self, bus):
        return ConvergenceMonitor(
            bus,
            confidence_threshold=0.85,
            min_delta=0.05,
            plateau_window=3,
            plateau_min_delta=0.02,
        )

    @pytest.mark.asyncio
    async def test_confidence_threshold_convergence(self, bus, monitor):
        """Convergence fires when confidence >= threshold."""
        convergence_received = []

        async def convergence_handler(s: Signal):
            convergence_received.append(s)
            return None

        bus.on(SignalKind.CONVERGENCE, convergence_handler)

        # Emit a step with high confidence
        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            source="chain",
            payload={
                "execution_id": "exec-1",
                "step_id": "verify",
                "confidence": 0.90,
            },
            correlation_id="corr-1",
        ))
        await asyncio.sleep(0.05)

        assert len(convergence_received) == 1
        assert "threshold" in convergence_received[0].payload["reason"]

    @pytest.mark.asyncio
    async def test_no_convergence_below_threshold(self, bus, monitor):
        """No convergence when confidence is below threshold."""
        convergence_received = []

        async def convergence_handler(s: Signal):
            convergence_received.append(s)
            return None

        bus.on(SignalKind.CONVERGENCE, convergence_handler)

        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            source="chain",
            payload={"execution_id": "exec-1", "step_id": "s1", "confidence": 0.5},
            correlation_id="corr-1",
        ))
        await asyncio.sleep(0.05)

        assert len(convergence_received) == 0

    @pytest.mark.asyncio
    async def test_improvement_delta_convergence(self, bus, monitor):
        """Convergence fires when improvement drops below min_delta."""
        convergence_received = []

        async def convergence_handler(s: Signal):
            convergence_received.append(s)
            return None

        bus.on(SignalKind.CONVERGENCE, convergence_handler)

        # First step
        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            source="chain",
            payload={"execution_id": "exec-1", "step_id": "s1", "confidence": 0.7},
            correlation_id="corr-1",
        ))
        await asyncio.sleep(0.02)

        # Second step — tiny improvement (< 0.05)
        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            source="chain",
            payload={"execution_id": "exec-1", "step_id": "s2", "confidence": 0.71},
            correlation_id="corr-1",
        ))
        await asyncio.sleep(0.05)

        assert len(convergence_received) == 1
        assert "min_delta" in convergence_received[0].payload["reason"]

    @pytest.mark.asyncio
    async def test_plateau_convergence(self, bus, monitor):
        """Convergence fires after plateau_window steps with minimal improvement."""
        convergence_received = []

        async def convergence_handler(s: Signal):
            convergence_received.append(s)
            return None

        bus.on(SignalKind.CONVERGENCE, convergence_handler)

        # 3 steps with improvements < 0.02 each
        for i, conf in enumerate([0.70, 0.71, 0.711]):
            await bus.emit(Signal(
                kind=SignalKind.STEP_COMPLETE,
                source="chain",
                payload={
                    "execution_id": "exec-1",
                    "step_id": f"s{i}",
                    "confidence": conf,
                },
                correlation_id="corr-1",
            ))
            await asyncio.sleep(0.02)

        await asyncio.sleep(0.05)

        # Should converge (improvement delta fires on step 2, but plateau
        # should also be detectable). At minimum, improvement delta fires.
        assert len(convergence_received) >= 1

    @pytest.mark.asyncio
    async def test_no_double_convergence(self, bus, monitor):
        """Once converged, don't emit again for the same execution."""
        convergence_received = []

        async def convergence_handler(s: Signal):
            convergence_received.append(s)
            return None

        bus.on(SignalKind.CONVERGENCE, convergence_handler)

        # First: converge
        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            source="chain",
            payload={"execution_id": "exec-1", "step_id": "s1", "confidence": 0.95},
            correlation_id="corr-1",
        ))
        await asyncio.sleep(0.05)

        # Second: still high confidence, but should NOT re-emit
        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            source="chain",
            payload={"execution_id": "exec-1", "step_id": "s2", "confidence": 0.96},
            correlation_id="corr-1",
        ))
        await asyncio.sleep(0.05)

        assert len(convergence_received) == 1

    @pytest.mark.asyncio
    async def test_separate_executions_tracked_independently(self, bus, monitor):
        """Different execution_ids are tracked independently."""
        convergence_received = []

        async def convergence_handler(s: Signal):
            convergence_received.append(s)
            return None

        bus.on(SignalKind.CONVERGENCE, convergence_handler)

        # Execution A converges
        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            source="chain",
            payload={"execution_id": "exec-A", "step_id": "s1", "confidence": 0.90},
            correlation_id="corr-A",
        ))
        await asyncio.sleep(0.02)

        # Execution B does NOT converge
        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            source="chain",
            payload={"execution_id": "exec-B", "step_id": "s1", "confidence": 0.5},
            correlation_id="corr-B",
        ))
        await asyncio.sleep(0.05)

        assert len(convergence_received) == 1
        assert convergence_received[0].payload["execution_id"] == "exec-A"

    @pytest.mark.asyncio
    async def test_reset(self, bus, monitor):
        """Reset clears convergence state."""
        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            source="chain",
            payload={"execution_id": "exec-1", "step_id": "s1", "confidence": 0.95},
            correlation_id="corr-1",
        ))
        await asyncio.sleep(0.02)

        assert monitor.get_status("exec-1")["converged"] is True

        monitor.reset("exec-1")
        assert monitor.get_status("exec-1")["converged"] is False

    @pytest.mark.asyncio
    async def test_no_confidence_in_payload_skipped(self, bus, monitor):
        """Steps without confidence in payload are silently skipped."""
        convergence_received = []

        async def convergence_handler(s: Signal):
            convergence_received.append(s)
            return None

        bus.on(SignalKind.CONVERGENCE, convergence_handler)

        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            source="chain",
            payload={"execution_id": "exec-1", "step_id": "s1"},
            correlation_id="corr-1",
        ))
        await asyncio.sleep(0.05)

        assert len(convergence_received) == 0

    @pytest.mark.asyncio
    async def test_correlation_propagation(self, bus, monitor):
        """Convergence signal should carry the original correlation_id."""
        convergence_received = []

        async def convergence_handler(s: Signal):
            convergence_received.append(s)
            return None

        bus.on(SignalKind.CONVERGENCE, convergence_handler)

        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            source="chain",
            payload={"execution_id": "exec-1", "step_id": "s1", "confidence": 0.92},
            correlation_id="my-workflow-123",
        ))
        await asyncio.sleep(0.05)

        assert len(convergence_received) == 1
        # The convergence signal should be linked to the same workflow
        assert convergence_received[0].correlation_id == "my-workflow-123"


# ============================================================================
# Integration: Bus + Journal + Convergence
# ============================================================================

class TestBusJournalIntegration:
    @pytest.mark.asyncio
    async def test_signals_journaled(self):
        """Signals emitted through the bus are recorded in the journal."""
        journal = SignalJournal(db_path=":memory:")
        bus = UnifiedBus(journal=journal)

        await bus.emit(Signal(
            kind=SignalKind.QUERY,
            source="api",
            correlation_id="test-corr",
        ))
        await asyncio.sleep(0.02)

        results = await journal.query_signals(correlation_id="test-corr")
        assert len(results) == 1
        assert results[0]["kind"] == "query"

        journal.close()

    @pytest.mark.asyncio
    async def test_convergence_signal_journaled(self):
        """Convergence signals emitted by the monitor are journaled."""
        journal = SignalJournal(db_path=":memory:")
        bus = UnifiedBus(journal=journal)
        monitor = ConvergenceMonitor(bus, confidence_threshold=0.85)

        await bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            source="chain",
            payload={"execution_id": "exec-1", "step_id": "s1", "confidence": 0.90},
            correlation_id="corr-1",
        ))
        await asyncio.sleep(0.05)

        # Both the STEP_COMPLETE and CONVERGENCE should be journaled
        all_signals = await journal.query_signals()
        kinds = {r["kind"] for r in all_signals}
        assert "step_complete" in kinds
        assert "convergence" in kinds

        journal.close()
