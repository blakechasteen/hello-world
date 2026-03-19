"""Tests for RitualRegistry — dispatch, skip_condition, DAG validation, failure policies."""

import asyncio
import pytest

from hololoom.core.ritual.types import (
    Ritual,
    RitualBudget,
    RitualContext,
    RitualResult,
    TokenBudget,
    HookBinding,
    Priority,
)
from hololoom.core.ritual.hooks import HookEvent
from hololoom.core.ritual.policies import FailurePolicy
from hololoom.core.ritual.registry import RitualRegistry, RitualDAGError, EscalationRequired


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
        token_budget=TokenBudget(total=5000, used=0),
        time_budget=60.0,
        escalation_path=[],
        session_id="session-1",
    )
    defaults.update(overrides)
    return RitualContext(**defaults)


def _make_ritual(
    ritual_id="test_ritual",
    consumes=None,
    produces=None,
    failure_policy=FailurePolicy.SKIP,
    budget_tokens=100,
    body=None,
    skip_condition=None,
    max_seconds=5.0,
    compensation_ritual_id=None,
):
    async def default_body(ctx):
        return RitualResult(success=True, produced={"TestBobbin": 1}, confidence_delta=0.01)

    return Ritual(
        id=ritual_id,
        version="1.0.0",
        domain="coding",
        intent="test ritual",
        author="system",
        consumes=consumes or [],
        produces=produces or [],
        failure_policy=failure_policy,
        budget=RitualBudget(max_tokens=budget_tokens, max_seconds=max_seconds, priority_class="standard"),
        body=body or default_body,
        skip_condition=skip_condition,
        compensation_ritual_id=compensation_ritual_id,
    )


def _bind(ritual, trigger=HookEvent.TASK_RECEIVED, priority=Priority.BLOCKING, condition=None):
    return HookBinding(ritual=ritual, trigger=trigger, condition=condition, priority=priority)


# =============================================================================
# DAG validation
# =============================================================================


class TestDAGValidation:
    def test_empty_registry_valid(self):
        reg = RitualRegistry()
        reg.validate_dag()  # no error

    def test_no_consumers_valid(self):
        reg = RitualRegistry()
        r = _make_ritual(produces=["X"])
        reg.register(_bind(r))
        reg.validate_dag()

    def test_satisfied_consumer(self):
        reg = RitualRegistry()
        producer = _make_ritual(ritual_id="producer", produces=["X"])
        consumer = _make_ritual(ritual_id="consumer", consumes=["X"])
        reg.register(_bind(producer))
        reg.register(_bind(consumer))
        reg.validate_dag()

    def test_unsatisfied_consumer_raises(self):
        reg = RitualRegistry()
        consumer = _make_ritual(ritual_id="consumer", consumes=["MissingBobbin"])
        reg.register(_bind(consumer))
        with pytest.raises(RitualDAGError, match="MissingBobbin"):
            reg.validate_dag()

    def test_cycle_detected(self):
        reg = RitualRegistry()
        a = _make_ritual(ritual_id="a", produces=["X"], consumes=["Y"])
        b = _make_ritual(ritual_id="b", produces=["Y"], consumes=["X"])
        reg.register(_bind(a))
        reg.register(_bind(b))
        with pytest.raises(RitualDAGError, match="Cycle"):
            reg.validate_dag()

    def test_diamond_no_cycle(self):
        """A → B and A → C, both consuming from A, is not a cycle."""
        reg = RitualRegistry()
        a = _make_ritual(ritual_id="a", produces=["X"])
        b = _make_ritual(ritual_id="b", consumes=["X"], produces=["Y"])
        c = _make_ritual(ritual_id="c", consumes=["X"], produces=["Z"])
        reg.register(_bind(a))
        reg.register(_bind(b))
        reg.register(_bind(c))
        reg.validate_dag()

    def test_chain_valid(self):
        """A → B → C is a valid DAG."""
        reg = RitualRegistry()
        a = _make_ritual(ritual_id="a", produces=["X"])
        b = _make_ritual(ritual_id="b", consumes=["X"], produces=["Y"])
        c = _make_ritual(ritual_id="c", consumes=["Y"])
        reg.register(_bind(a))
        reg.register(_bind(b))
        reg.register(_bind(c))
        reg.validate_dag()


# =============================================================================
# Dispatch
# =============================================================================


class TestDispatch:
    @pytest.mark.asyncio
    async def test_basic_dispatch(self):
        reg = RitualRegistry()
        ritual = _make_ritual()
        reg.register(_bind(ritual))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert "test_ritual" in results
        assert results["test_ritual"].success

    @pytest.mark.asyncio
    async def test_no_bindings_returns_empty(self):
        reg = RitualRegistry()
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.SESSION_START, ctx)
        assert results == {}

    @pytest.mark.asyncio
    async def test_condition_filters(self):
        reg = RitualRegistry()
        ritual = _make_ritual()
        binding = _bind(ritual, condition=lambda ctx: ctx.domain == "beekeeping")
        reg.register(binding)
        ctx = _make_ctx(domain="coding")
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert results == {}

    @pytest.mark.asyncio
    async def test_condition_passes(self):
        reg = RitualRegistry()
        ritual = _make_ritual()
        binding = _bind(ritual, condition=lambda ctx: ctx.domain == "coding")
        reg.register(binding)
        ctx = _make_ctx(domain="coding")
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert "test_ritual" in results

    @pytest.mark.asyncio
    async def test_blocking_runs_serially_feeds_outputs(self):
        call_order = []

        async def body_a(ctx):
            call_order.append("a")
            return RitualResult(success=True, produced={"A": 1}, confidence_delta=0.0)

        async def body_b(ctx):
            call_order.append("b")
            assert ctx.ritual_outputs.get("A") == 1  # fed from a
            return RitualResult(success=True, produced={"B": 2}, confidence_delta=0.0)

        reg = RitualRegistry()
        a = _make_ritual(ritual_id="a", body=body_a)
        b = _make_ritual(ritual_id="b", body=body_b)
        reg.register(_bind(a, priority=Priority.BLOCKING))
        reg.register(_bind(b, priority=Priority.BLOCKING))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert call_order == ["a", "b"]
        assert "A" in ctx.ritual_outputs
        assert "B" in ctx.ritual_outputs

    @pytest.mark.asyncio
    async def test_parallel_runs_concurrently(self):
        call_order = []

        async def body_a(ctx):
            call_order.append("a_start")
            await asyncio.sleep(0.01)
            call_order.append("a_end")
            return RitualResult(success=True, produced={"A": 1}, confidence_delta=0.0)

        async def body_b(ctx):
            call_order.append("b_start")
            await asyncio.sleep(0.01)
            call_order.append("b_end")
            return RitualResult(success=True, produced={"B": 2}, confidence_delta=0.0)

        reg = RitualRegistry()
        a = _make_ritual(ritual_id="a", body=body_a)
        b = _make_ritual(ritual_id="b", body=body_b)
        reg.register(_bind(a, priority=Priority.PARALLEL))
        reg.register(_bind(b, priority=Priority.PARALLEL))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        # Both started before either finished
        assert "a_start" in call_order[:2]
        assert "b_start" in call_order[:2]
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_outputs_available_to_subsequent(self):
        async def body_a(ctx):
            return RitualResult(success=True, produced={"HealthScore": 9.8}, confidence_delta=0.0)

        reg = RitualRegistry()
        a = _make_ritual(ritual_id="a", body=body_a)
        reg.register(_bind(a, priority=Priority.BLOCKING))
        ctx = _make_ctx()
        await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert ctx.ritual_outputs["HealthScore"] == 9.8


# =============================================================================
# Skip conditions
# =============================================================================


class TestSkipCondition:
    @pytest.mark.asyncio
    async def test_skip_when_true(self):
        ritual = _make_ritual(skip_condition=lambda ctx: True)
        reg = RitualRegistry()
        reg.register(_bind(ritual))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert results == {}  # skipped

    @pytest.mark.asyncio
    async def test_no_skip_when_false(self):
        ritual = _make_ritual(skip_condition=lambda ctx: False)
        reg = RitualRegistry()
        reg.register(_bind(ritual))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert "test_ritual" in results

    @pytest.mark.asyncio
    async def test_skip_based_on_context(self):
        ritual = _make_ritual(
            skip_condition=lambda ctx: ctx.task_complexity == "trivial"
        )
        reg = RitualRegistry()
        reg.register(_bind(ritual))

        ctx_trivial = _make_ctx(task_complexity="trivial")
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx_trivial)
        assert results == {}

        ctx_standard = _make_ctx(task_complexity="standard")
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx_standard)
        assert "test_ritual" in results


# =============================================================================
# Failure policies
# =============================================================================


class TestFailurePolicies:
    @pytest.mark.asyncio
    async def test_skip_policy(self):
        async def failing_body(ctx):
            raise ValueError("boom")

        ritual = _make_ritual(body=failing_body, failure_policy=FailurePolicy.SKIP)
        reg = RitualRegistry()
        reg.register(_bind(ritual))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert results == {}  # skipped, no error raised

    @pytest.mark.asyncio
    async def test_retry_once_succeeds_on_retry(self):
        call_count = 0

        async def flaky_body(ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first try fails")
            return RitualResult(success=True, produced={"R": 1}, confidence_delta=0.0)

        ritual = _make_ritual(body=flaky_body, failure_policy=FailurePolicy.RETRY_ONCE)
        reg = RitualRegistry()
        reg.register(_bind(ritual))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert "test_ritual" in results
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_once_fails_both_times(self):
        async def always_fails(ctx):
            raise ValueError("nope")

        ritual = _make_ritual(body=always_fails, failure_policy=FailurePolicy.RETRY_ONCE)
        reg = RitualRegistry()
        reg.register(_bind(ritual))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert results == {}  # falls back to skip

    @pytest.mark.asyncio
    async def test_halt_policy_raises(self):
        async def failing_body(ctx):
            raise ValueError("critical failure")

        ritual = _make_ritual(body=failing_body, failure_policy=FailurePolicy.HALT)
        reg = RitualRegistry()
        reg.register(_bind(ritual))
        ctx = _make_ctx()
        with pytest.raises(RuntimeError, match="halted pipeline"):
            await reg.fire(HookEvent.TASK_RECEIVED, ctx)

    @pytest.mark.asyncio
    async def test_escalate_policy_raises(self):
        async def failing_body(ctx):
            raise ValueError("need bigger model")

        ritual = _make_ritual(body=failing_body, failure_policy=FailurePolicy.ESCALATE)
        reg = RitualRegistry()
        reg.register(_bind(ritual))
        ctx = _make_ctx()
        with pytest.raises(EscalationRequired) as exc_info:
            await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert exc_info.value.ritual_id == "test_ritual"

    @pytest.mark.asyncio
    async def test_log_and_continue_policy(self):
        async def failing_body(ctx):
            raise ValueError("non-critical")

        ritual = _make_ritual(body=failing_body, failure_policy=FailurePolicy.LOG_AND_CONTINUE)
        reg = RitualRegistry()
        reg.register(_bind(ritual))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert results == {}  # continued

    @pytest.mark.asyncio
    async def test_compensate_policy(self):
        async def failing_body(ctx):
            raise ValueError("main failed")

        async def compensation_body(ctx):
            return RitualResult(success=True, produced={"Fallback": 1}, confidence_delta=0.0)

        comp = _make_ritual(ritual_id="comp", body=compensation_body)
        main = _make_ritual(
            ritual_id="main",
            body=failing_body,
            failure_policy=FailurePolicy.COMPENSATE,
            compensation_ritual_id="comp",
        )

        reg = RitualRegistry()
        reg.register(_bind(comp, trigger=HookEvent.SESSION_START))  # register comp so it's known
        reg.register(_bind(main))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert "main" in results
        assert results["main"].produced.get("Fallback") == 1

    @pytest.mark.asyncio
    async def test_compensate_falls_back_to_skip(self):
        """If compensation ritual doesn't exist, falls back to skip."""
        async def failing_body(ctx):
            raise ValueError("main failed")

        main = _make_ritual(
            body=failing_body,
            failure_policy=FailurePolicy.COMPENSATE,
            compensation_ritual_id="nonexistent",
        )

        reg = RitualRegistry()
        reg.register(_bind(main))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert results == {}


# =============================================================================
# Admission control integration
# =============================================================================


class TestAdmission:
    @pytest.mark.asyncio
    async def test_insufficient_budget_skips(self):
        ritual = _make_ritual(budget_tokens=1000)
        reg = RitualRegistry()
        reg.register(_bind(ritual))
        ctx = _make_ctx(token_budget=TokenBudget(total=100, used=50))
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert results == {}

    @pytest.mark.asyncio
    async def test_sufficient_budget_proceeds(self):
        ritual = _make_ritual(budget_tokens=100)
        reg = RitualRegistry()
        reg.register(_bind(ritual))
        ctx = _make_ctx(token_budget=TokenBudget(total=1000, used=0))
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert "test_ritual" in results


# =============================================================================
# Timeout
# =============================================================================


class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_triggers_failure_policy(self):
        async def slow_body(ctx):
            await asyncio.sleep(10)
            return RitualResult(success=True, produced={}, confidence_delta=0.0)

        ritual = _make_ritual(
            body=slow_body,
            max_seconds=0.1,
            failure_policy=FailurePolicy.SKIP,
        )
        reg = RitualRegistry()
        reg.register(_bind(ritual))
        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert results == {}  # timed out → skip
