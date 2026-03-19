"""Tests for weaverlet.schemas — Budget, BudgetLedger, TaskLease, WorkingState."""

import time
import pytest

from hololoom.weaverlet.schemas import (
    Budget,
    BudgetExhaustedError,
    BudgetLedger,
    IterationOutput,
    MilestoneDelta,
    TaskLease,
    TaskResult,
    TerminationReason,
    WorkingState,
)


# ── Budget (frozen) ───────────────────────────────────────────────────────

class TestBudget:
    def test_defaults(self):
        b = Budget()
        assert b.max_iterations == 10
        assert b.max_tokens == 100_000
        assert b.max_wall_seconds == 600.0
        assert b.max_tool_calls == 50

    def test_frozen(self):
        b = Budget()
        with pytest.raises(AttributeError):
            b.max_iterations = 20

    def test_custom(self):
        b = Budget(max_iterations=1, max_tokens=500)
        assert b.max_iterations == 1
        assert b.max_tokens == 500


# ── BudgetLedger ──────────────────────────────────────────────────────────

class TestBudgetLedger:
    def test_charge_tokens(self):
        ledger = BudgetLedger(budget=Budget(max_tokens=100))
        ledger.charge_tokens(50)
        assert ledger.tokens_used == 50
        ledger.charge_tokens(30)
        assert ledger.tokens_used == 80

    def test_token_exhaustion(self):
        ledger = BudgetLedger(budget=Budget(max_tokens=100))
        with pytest.raises(BudgetExhaustedError, match="Token budget"):
            ledger.charge_tokens(150)

    def test_charge_tool_call(self):
        ledger = BudgetLedger(budget=Budget(max_tool_calls=3))
        ledger.charge_tool_call()
        ledger.charge_tool_call()
        ledger.charge_tool_call()
        with pytest.raises(BudgetExhaustedError, match="Tool call budget"):
            ledger.charge_tool_call()

    def test_charge_iteration(self):
        ledger = BudgetLedger(budget=Budget(max_iterations=2))
        ledger.charge_iteration()
        assert ledger.iterations_used == 1
        ledger.charge_iteration()
        assert ledger.is_exhausted()

    def test_is_exhausted_iterations(self):
        ledger = BudgetLedger(budget=Budget(max_iterations=1))
        assert not ledger.is_exhausted()
        ledger.charge_iteration()
        assert ledger.is_exhausted()

    def test_is_exhausted_wall_time(self):
        ledger = BudgetLedger(budget=Budget(max_wall_seconds=0.01))
        time.sleep(0.02)
        assert ledger.is_exhausted()

    def test_exhaustion_reason_tokens(self):
        ledger = BudgetLedger(budget=Budget(max_tokens=10))
        assert ledger.exhaustion_reason() is None
        ledger.tokens_used = 10
        reason = ledger.exhaustion_reason()
        assert reason is not None
        assert "tokens" in reason

    def test_remaining_fraction(self):
        ledger = BudgetLedger(budget=Budget(max_iterations=10, max_tokens=1000))
        ledger.charge_iteration()
        ledger.charge_tokens(500)
        frac = ledger.remaining_fraction()
        assert 0.0 < frac <= 0.9

    def test_remaining_dict(self):
        ledger = BudgetLedger(budget=Budget(max_iterations=4))
        ledger.charge_iteration()
        r = ledger.remaining()
        assert r["iterations"] == 0.75


# ── TaskLease ─────────────────────────────────────────────────────────────

class TestTaskLease:
    def test_frozen(self):
        lease = TaskLease(
            lease_id="x", goal="test", context_bundle={}, budget=Budget()
        )
        with pytest.raises(AttributeError):
            lease.goal = "changed"

    def test_factory(self):
        lease = TaskLease.create("do something", Budget(max_iterations=3))
        assert lease.goal == "do something"
        assert lease.budget.max_iterations == 3
        assert len(lease.lease_id) == 12

    def test_optional_fields(self):
        lease = TaskLease.create("goal", thread_id="t1", tapestry_id="tap1")
        assert lease.thread_id == "t1"
        assert lease.tapestry_id == "tap1"


# ── WorkingState ──────────────────────────────────────────────────────────

class TestWorkingState:
    def test_latest_score_empty(self):
        state = WorkingState()
        assert state.latest_score == 0.0

    def test_latest_score(self):
        state = WorkingState(score_history=[0.1, 0.3, 0.5])
        assert state.latest_score == 0.5

    def test_previous_score(self):
        state = WorkingState(score_history=[0.1, 0.3, 0.5])
        assert state.previous_score == 0.3

    def test_previous_score_single(self):
        state = WorkingState(score_history=[0.5])
        assert state.previous_score == 0.0


# ── IterationOutput (frozen) ─────────────────────────────────────────────

class TestIterationOutput:
    def test_frozen(self):
        out = IterationOutput(
            iteration_id="x",
            progress_score=0.5,
            updated_artifact=None,
            critique="ok",
            next_action="continue",
        )
        with pytest.raises(AttributeError):
            out.progress_score = 0.9


# ── TerminationReason ─────────────────────────────────────────────────────

class TestTerminationReason:
    def test_values(self):
        assert TerminationReason.GOAL_MET.value == "goal_met"
        assert TerminationReason.STAGNATION.value == "stagnation"
        assert TerminationReason.BUDGET_EXHAUSTED.value == "budget_exhausted"
