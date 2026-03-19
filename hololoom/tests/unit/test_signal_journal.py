"""
Tests for SignalJournal — Phase 2.

Validates SQLite persistence for signals, execution traces, steps, and MCTS decisions.
"""

import pytest
from datetime import datetime

from hololoom.core.bus.signal import Signal, SignalKind, SignalPriority, TriggerMode
from hololoom.core.bus.journal import SignalJournal


@pytest.fixture
def journal():
    j = SignalJournal(db_path=":memory:")
    yield j
    j.close()


class TestSignalJournal:
    @pytest.mark.asyncio
    async def test_append_and_query(self, journal):
        signal = Signal(
            kind=SignalKind.QUERY,
            source="api",
            payload={"text": "hello"},
            correlation_id="corr-1",
        )
        await journal.append(signal)

        results = await journal.query_signals(correlation_id="corr-1")
        assert len(results) == 1
        assert results[0]["kind"] == "query"
        assert results[0]["source"] == "api"

    @pytest.mark.asyncio
    async def test_query_by_kind(self, journal):
        await journal.append(Signal(kind=SignalKind.QUERY, source="a"))
        await journal.append(Signal(kind=SignalKind.HEARTBEAT, source="b"))
        await journal.append(Signal(kind=SignalKind.QUERY, source="c"))

        results = await journal.query_signals(kind="query")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_by_source(self, journal):
        await journal.append(Signal(kind=SignalKind.QUERY, source="api"))
        await journal.append(Signal(kind=SignalKind.QUERY, source="mcts"))

        results = await journal.query_signals(source="mcts")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_duplicate_signal_ignored(self, journal):
        signal = Signal(id="dup-id", kind=SignalKind.QUERY)
        await journal.append(signal)
        await journal.append(signal)  # Duplicate

        results = await journal.query_signals()
        assert len(results) == 1


class TestExecutionTrace:
    @pytest.mark.asyncio
    async def test_record_and_query(self, journal):
        await journal.record_execution(
            trace_id="trace-1",
            correlation_id="corr-1",
            execution_type="chain",
            name="verified_query",
            started_at=datetime.now().isoformat(),
            success=True,
            total_steps=3,
            duration_ms=150.5,
            final_confidence=0.92,
        )

        traces = await journal.query_traces(execution_type="chain")
        assert len(traces) == 1
        assert traces[0]["name"] == "verified_query"
        assert traces[0]["success"] == 1
        assert traces[0]["final_confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_query_by_success(self, journal):
        await journal.record_execution(
            trace_id="t1", correlation_id="c1", execution_type="chain",
            started_at="2026-01-01", success=True,
        )
        await journal.record_execution(
            trace_id="t2", correlation_id="c2", execution_type="chain",
            started_at="2026-01-02", success=False, error="timeout",
        )

        success = await journal.query_traces(success=True)
        assert len(success) == 1
        assert success[0]["id"] == "t1"

        failures = await journal.query_traces(success=False)
        assert len(failures) == 1
        assert failures[0]["error"] == "timeout"


class TestStepResult:
    @pytest.mark.asyncio
    async def test_record_and_query_steps(self, journal):
        await journal.record_execution(
            trace_id="exec-1", correlation_id="c1", execution_type="chain",
            started_at="2026-01-01",
        )

        await journal.record_step(
            step_record_id="s1", execution_id="exec-1",
            step_id="execute", step_type="execute", status="success",
            timestamp="2026-01-01T00:00:01",
            duration_ms=50.0, confidence=0.75,
        )
        await journal.record_step(
            step_record_id="s2", execution_id="exec-1",
            step_id="verify", step_type="verify", status="success",
            timestamp="2026-01-01T00:00:02",
            duration_ms=30.0, confidence=0.92,
        )

        steps = await journal.query_steps("exec-1")
        assert len(steps) == 2
        assert steps[0]["step_id"] == "execute"
        assert steps[1]["step_id"] == "verify"
        assert steps[1]["confidence"] == 0.92


class TestMCTSDecision:
    @pytest.mark.asyncio
    async def test_record_and_query(self, journal):
        await journal.record_mcts_decision(
            decision_id="mcts-1",
            tree_level="strategic",
            created_at=datetime.now().isoformat(),
            domain="code",
            simulations_run=50,
            selected_action="research_pipeline",
            root_visits=50,
            best_value=0.82,
            alternatives=[
                {"action": "verified_query", "visits": 15, "value": 0.7},
                {"action": "simple_query", "visits": 5, "value": 0.4},
            ],
        )

        results = await journal.query_mcts_decisions(domain="code")
        assert len(results) == 1
        assert results[0]["selected_action"] == "research_pipeline"
        assert results[0]["simulations_run"] == 50

    @pytest.mark.asyncio
    async def test_query_by_level(self, journal):
        await journal.record_mcts_decision(
            decision_id="m1", tree_level="strategic",
            created_at="2026-01-01",
        )
        await journal.record_mcts_decision(
            decision_id="m2", tree_level="tactical",
            created_at="2026-01-01",
        )

        strategic = await journal.query_mcts_decisions(tree_level="strategic")
        assert len(strategic) == 1

        tactical = await journal.query_mcts_decisions(tree_level="tactical")
        assert len(tactical) == 1


class TestWorkflowHistory:
    @pytest.mark.asyncio
    async def test_full_workflow_history(self, journal):
        corr = "workflow-42"

        # Signals
        await journal.append(Signal(
            kind=SignalKind.QUERY, source="api", correlation_id=corr,
        ))
        await journal.append(Signal(
            kind=SignalKind.MCTS_DECISION, source="mcts", correlation_id=corr,
        ))
        await journal.append(Signal(
            kind=SignalKind.EXECUTION_COMPLETE, source="chain", correlation_id=corr,
        ))

        # Execution trace
        await journal.record_execution(
            trace_id="exec-42", correlation_id=corr, execution_type="chain",
            name="auto_refine", started_at="2026-01-01",
            success=True, total_steps=3, duration_ms=200.0,
        )

        # Steps
        await journal.record_step(
            step_record_id="s1", execution_id="exec-42",
            step_id="execute", step_type="execute", status="success",
            timestamp="2026-01-01T00:00:01",
        )

        # MCTS
        await journal.record_mcts_decision(
            decision_id="mcts-42", tree_level="strategic",
            correlation_id=corr, created_at="2026-01-01",
            selected_action="auto_refine",
        )

        # Query full history
        history = await journal.get_workflow_history(corr)

        assert len(history["signals"]) == 3
        assert len(history["traces"]) == 1
        assert len(history["steps"]) == 1
        assert len(history["mcts_decisions"]) == 1
        assert history["traces"][0]["name"] == "auto_refine"
        assert history["mcts_decisions"][0]["selected_action"] == "auto_refine"
