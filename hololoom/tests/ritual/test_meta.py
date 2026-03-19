"""Tests for the meta-ritual: ritual_health_audit."""

import pytest

from hololoom.core.ritual.meta import RITUAL_HEALTH_AUDIT, _ritual_health_audit_body
from hololoom.core.ritual.types import RitualContext, TokenBudget
from hololoom.core.ritual.policies import FailurePolicy


def _make_ctx(**overrides) -> RitualContext:
    defaults = dict(
        agent_id="system",
        agent_role="masterweaver",
        domain="universal",
        task_id="audit-1",
        task_type="meta_audit",
        task_intent="evaluate ritual grammar health",
        task_complexity="standard",
        yarn=None,
        warp=None,
        confidence=0.9,
        confidence_history=[0.9],
        ritual_outputs={},
        ritual_log=[],
        token_budget=TokenBudget(total=5000, used=0),
        time_budget=120.0,
        escalation_path=[],
        session_id="session-1",
    )
    defaults.update(overrides)
    return RitualContext(**defaults)


class TestRitualHealthAudit:
    def test_ritual_identity(self):
        assert RITUAL_HEALTH_AUDIT.id == "ritual_health_audit"
        assert RITUAL_HEALTH_AUDIT.domain == "universal"
        assert RITUAL_HEALTH_AUDIT.failure_policy == FailurePolicy.LOG_AND_CONTINUE
        assert "RitualHealthReport" in RITUAL_HEALTH_AUDIT.produces

    def test_budget_is_best_effort(self):
        assert RITUAL_HEALTH_AUDIT.budget.priority_class == "best_effort"

    @pytest.mark.asyncio
    async def test_body_returns_report(self):
        ctx = _make_ctx()
        result = await _ritual_health_audit_body(ctx)
        assert result.success
        assert "RitualHealthReport" in result.produced
        report = result.produced["RitualHealthReport"]
        assert "high_skip_rituals" in report
        assert "failure_correlated" in report
        assert "dead_weight_outputs" in report
        assert "miscalibrated" in report
        assert report["recommendation"] == "ok"

    @pytest.mark.asyncio
    async def test_body_no_yarn_graceful(self):
        ctx = _make_ctx(yarn=None)
        result = await _ritual_health_audit_body(ctx)
        assert result.success  # doesn't crash without Yarn

    @pytest.mark.asyncio
    async def test_confidence_delta_zero(self):
        ctx = _make_ctx()
        result = await _ritual_health_audit_body(ctx)
        assert result.confidence_delta == 0.0
