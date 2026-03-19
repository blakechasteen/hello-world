"""Tests for dawn/dusk session rituals."""

import pytest

from hololoom.core.ritual.rituals.session import (
    DAWN_PRIME,
    DAWN_PRIME_BINDING_CONFIG,
    DUSK_CONSOLIDATE,
    DUSK_CONSOLIDATE_BINDING_CONFIG,
    _dawn_prime_body,
    _dusk_consolidate_body,
)
from hololoom.core.ritual.types import RitualContext, TokenBudget, RitualRecord
from hololoom.core.ritual.hooks import HookEvent
from hololoom.core.ritual.policies import FailurePolicy


def _make_ctx(**overrides) -> RitualContext:
    defaults = dict(
        agent_id="agent-1",
        agent_role="coder",
        domain="coding",
        task_id="task-1",
        task_type="session",
        task_intent="start session",
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


class TestDawnPrime:
    def test_identity(self):
        assert DAWN_PRIME.id == "dawn_prime"
        assert DAWN_PRIME.domain == "universal"
        assert "SessionPrimeBobbin" in DAWN_PRIME.produces

    def test_binding_trigger(self):
        assert DAWN_PRIME_BINDING_CONFIG["trigger"] == HookEvent.SESSION_START

    @pytest.mark.asyncio
    async def test_body_returns_priming_context(self):
        ctx = _make_ctx()
        result = await _dawn_prime_body(ctx)
        assert result.success
        assert "SessionPrimeBobbin" in result.produced
        assert result.confidence_delta > 0


class TestDuskConsolidate:
    def test_identity(self):
        assert DUSK_CONSOLIDATE.id == "dusk_consolidate"
        assert DUSK_CONSOLIDATE.domain == "universal"
        assert "ConsolidationReportBobbin" in DUSK_CONSOLIDATE.produces

    def test_binding_trigger(self):
        assert DUSK_CONSOLIDATE_BINDING_CONFIG["trigger"] == HookEvent.SESSION_END

    def test_failure_policy(self):
        assert DUSK_CONSOLIDATE.failure_policy == FailurePolicy.LOG_AND_CONTINUE

    @pytest.mark.asyncio
    async def test_body_counts_ritual_log(self):
        from datetime import datetime, timezone
        logs = [
            RitualRecord(ritual_id="r1", fired_at=datetime.now(timezone.utc), was_skipped=False, result=None),
            RitualRecord(ritual_id="r2", fired_at=datetime.now(timezone.utc), was_skipped=True, result=None),
        ]
        ctx = _make_ctx(ritual_log=logs)
        result = await _dusk_consolidate_body(ctx)
        assert result.success
        report = result.produced["ConsolidationReportBobbin"]
        assert report["rituals_executed"] == 2
