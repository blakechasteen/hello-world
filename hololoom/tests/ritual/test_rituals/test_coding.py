"""Tests for pre_code_health_check ritual."""

import pytest

from hololoom.core.ritual.rituals.coding import (
    PRE_CODE_HEALTH_CHECK,
    PRE_CODE_HEALTH_CHECK_BINDING_CONFIG,
    _pre_code_health_check_body,
)
from hololoom.core.ritual.types import (
    RitualContext,
    RitualResult,
    TokenBudget,
    HookBinding,
    Priority,
)
from hololoom.core.ritual.hooks import HookEvent
from hololoom.core.ritual.policies import FailurePolicy
from hololoom.core.ritual.registry import RitualRegistry


def _make_ctx(**overrides) -> RitualContext:
    defaults = dict(
        agent_id="coder-1",
        agent_role="coder",
        domain="coding",
        task_id="task-1",
        task_type="code_modify",
        task_intent="refactor module",
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


class TestPreCodeHealthCheck:
    def test_ritual_identity(self):
        assert PRE_CODE_HEALTH_CHECK.id == "pre_code_health_check"
        assert PRE_CODE_HEALTH_CHECK.domain == "coding"
        assert PRE_CODE_HEALTH_CHECK.failure_policy == FailurePolicy.LOG_AND_CONTINUE
        assert "HealthScoreBobbin" in PRE_CODE_HEALTH_CHECK.produces

    def test_binding_config(self):
        assert PRE_CODE_HEALTH_CHECK_BINDING_CONFIG["trigger"] == HookEvent.TASK_RECEIVED
        assert PRE_CODE_HEALTH_CHECK_BINDING_CONFIG["priority"] == "BLOCKING"

    @pytest.mark.asyncio
    async def test_body_returns_health_score(self):
        ctx = _make_ctx()
        result = await _pre_code_health_check_body(ctx)
        assert result.success
        bobbin = result.produced["HealthScoreBobbin"]
        assert bobbin["score"] >= 9.5
        assert bobbin["status"] == "ok"
        assert result.confidence_delta > 0

    def test_skip_when_already_run(self):
        ctx = _make_ctx(ritual_outputs={"HealthScoreBobbin": {"score": 10.0}})
        assert PRE_CODE_HEALTH_CHECK.skip_condition(ctx) is True

    def test_skip_trivial_tasks(self):
        ctx = _make_ctx(task_complexity="trivial")
        assert PRE_CODE_HEALTH_CHECK.skip_condition(ctx) is True

    def test_skip_low_budget(self):
        ctx = _make_ctx(token_budget=TokenBudget(total=400, used=0))
        assert PRE_CODE_HEALTH_CHECK.skip_condition(ctx) is True

    def test_no_skip_standard_task(self):
        ctx = _make_ctx()
        assert PRE_CODE_HEALTH_CHECK.skip_condition(ctx) is False

    def test_binding_condition_filters_task_type(self):
        condition = PRE_CODE_HEALTH_CHECK_BINDING_CONFIG["condition"]
        ctx_modify = _make_ctx(task_type="code_modify")
        ctx_chat = _make_ctx(task_type="chat")
        assert condition(ctx_modify) is True
        assert condition(ctx_chat) is False


class TestPreCodeHealthCheckEndToEnd:
    @pytest.mark.asyncio
    async def test_full_dispatch(self):
        """End-to-end: register, dispatch, check outputs."""
        reg = RitualRegistry()
        binding = HookBinding(
            ritual=PRE_CODE_HEALTH_CHECK,
            trigger=HookEvent.TASK_RECEIVED,
            condition=PRE_CODE_HEALTH_CHECK_BINDING_CONFIG["condition"],
            priority=Priority.BLOCKING,
        )
        reg.register(binding)

        ctx = _make_ctx()
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert "pre_code_health_check" in results
        assert ctx.ritual_outputs["HealthScoreBobbin"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_skipped_on_trivial(self):
        reg = RitualRegistry()
        binding = HookBinding(
            ritual=PRE_CODE_HEALTH_CHECK,
            trigger=HookEvent.TASK_RECEIVED,
            condition=PRE_CODE_HEALTH_CHECK_BINDING_CONFIG["condition"],
            priority=Priority.BLOCKING,
        )
        reg.register(binding)

        ctx = _make_ctx(task_complexity="trivial")
        results = await reg.fire(HookEvent.TASK_RECEIVED, ctx)
        assert results == {}
