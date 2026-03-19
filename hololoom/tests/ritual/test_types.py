"""Tests for ritual type definitions."""

from datetime import datetime

from hololoom.core.ritual.types import (
    HookBinding,
    Priority,
    Ritual,
    RitualBudget,
    RitualContext,
    RitualOutcome,
    RitualRecord,
    RitualResult,
    TokenBudget,
)
from hololoom.core.ritual.hooks import HookEvent
from hololoom.core.ritual.policies import FailurePolicy


def _make_ctx(**overrides) -> RitualContext:
    defaults = dict(
        agent_id="agent-1",
        agent_role="coder",
        domain="coding",
        task_id="task-1",
        task_type="code_modify",
        task_intent="fix bug",
        task_complexity="standard",
        yarn=None,
        warp=None,
        confidence=0.8,
        confidence_history=[0.8],
        ritual_outputs={},
        ritual_log=[],
        token_budget=TokenBudget(total=1000, used=0),
        time_budget=60.0,
        escalation_path=[],
        session_id="session-1",
    )
    defaults.update(overrides)
    return RitualContext(**defaults)


class TestTokenBudget:
    def test_remaining(self):
        tb = TokenBudget(total=1000, used=300)
        assert tb.remaining == 700

    def test_remaining_zero(self):
        tb = TokenBudget(total=500, used=500)
        assert tb.remaining == 0

    def test_remaining_overspent(self):
        tb = TokenBudget(total=100, used=150)
        assert tb.remaining == -50


class TestPriority:
    def test_ordering(self):
        assert Priority.BLOCKING.value < Priority.PARALLEL.value
        assert Priority.PARALLEL.value < Priority.DEFERRED.value


class TestRitualResult:
    def test_success_result(self):
        r = RitualResult(
            success=True,
            produced={"ScoreBobbin": {"score": 9.8}},
            confidence_delta=0.03,
            notes="ok",
        )
        assert r.success
        assert r.produced["ScoreBobbin"]["score"] == 9.8
        assert r.error is None

    def test_failure_result(self):
        err = ValueError("bad input")
        r = RitualResult(
            success=False,
            produced={},
            confidence_delta=-0.1,
            error=err,
        )
        assert not r.success
        assert r.error is err


class TestRitual:
    def test_ritual_identity(self):
        async def body(ctx):
            return RitualResult(success=True, produced={}, confidence_delta=0.0)

        ritual = Ritual(
            id="test_ritual",
            version="1.0.0",
            domain="coding",
            intent="test",
            author="system",
            consumes=[],
            produces=["TestBobbin"],
            failure_policy=FailurePolicy.SKIP,
            budget=RitualBudget(max_tokens=100, max_seconds=5.0, priority_class="standard"),
            body=body,
        )
        assert ritual.id == "test_ritual"
        assert ritual.version == "1.0.0"
        assert ritual.compensation_ritual_id is None
        assert ritual.skip_condition is None


class TestHookBinding:
    def test_binding_separates_from_ritual(self):
        async def body(ctx):
            return RitualResult(success=True, produced={}, confidence_delta=0.0)

        ritual = Ritual(
            id="test_ritual",
            version="1.0.0",
            domain="coding",
            intent="test",
            author="system",
            consumes=[],
            produces=[],
            failure_policy=FailurePolicy.SKIP,
            budget=RitualBudget(max_tokens=100, max_seconds=5.0, priority_class="standard"),
            body=body,
        )
        binding = HookBinding(
            ritual=ritual,
            trigger=HookEvent.TASK_RECEIVED,
            condition=None,
            priority=Priority.BLOCKING,
        )
        assert binding.trigger == HookEvent.TASK_RECEIVED
        assert binding.ritual.id == "test_ritual"

    def test_rebinding(self):
        """A ritual can be rebound to a different trigger without changing identity."""
        async def body(ctx):
            return RitualResult(success=True, produced={}, confidence_delta=0.0)

        ritual = Ritual(
            id="rebindable",
            version="1.0.0",
            domain="universal",
            intent="test rebinding",
            author="system",
            consumes=[],
            produces=[],
            failure_policy=FailurePolicy.SKIP,
            budget=RitualBudget(max_tokens=100, max_seconds=5.0, priority_class="standard"),
            body=body,
        )
        b1 = HookBinding(ritual=ritual, trigger=HookEvent.TASK_RECEIVED, condition=None, priority=Priority.BLOCKING)
        b2 = HookBinding(ritual=ritual, trigger=HookEvent.SESSION_START, condition=None, priority=Priority.PARALLEL)

        assert b1.ritual is b2.ritual  # same identity
        assert b1.trigger != b2.trigger  # different triggers


class TestRitualOutcome:
    def test_outcome_fields(self):
        o = RitualOutcome(
            ritual_id="test",
            ritual_version="1.0.0",
            session_id="s1",
            agent_id="a1",
            task_context="fix bug",
            fired_at=datetime(2026, 3, 19, 12, 0),
            duration_ms=150,
            tokens_used=42,
            produced={"X": 1},
            downstream_task_success=None,
            confidence_delta=0.03,
            was_skipped=False,
            skip_reason=None,
            failure_policy_invoked=None,
        )
        assert o.downstream_task_success is None  # filled later
        assert o.duration_ms == 150

    def test_skipped_outcome(self):
        o = RitualOutcome(
            ritual_id="test",
            ritual_version="1.0.0",
            session_id="s1",
            agent_id="a1",
            task_context="trivial fix",
            fired_at=datetime(2026, 3, 19, 12, 0),
            duration_ms=0,
            tokens_used=0,
            produced={},
            downstream_task_success=None,
            confidence_delta=0.0,
            was_skipped=True,
            skip_reason="skip_condition evaluated True",
            failure_policy_invoked=None,
        )
        assert o.was_skipped
        assert o.skip_reason is not None


class TestRitualContext:
    def test_context_creation(self):
        ctx = _make_ctx()
        assert ctx.agent_id == "agent-1"
        assert ctx.token_budget.remaining == 1000

    def test_context_with_overrides(self):
        ctx = _make_ctx(domain="beekeeping", confidence=0.3)
        assert ctx.domain == "beekeeping"
        assert ctx.confidence == 0.3
