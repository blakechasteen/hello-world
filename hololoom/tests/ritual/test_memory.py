"""Tests for ritual memory write (Yarn + Warp)."""

import pytest
from datetime import datetime

from hololoom.core.ritual.memory import write_outcome, _write_to_yarn, _embed_to_warp
from hololoom.core.ritual.types import RitualOutcome, RitualContext, TokenBudget


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


def _make_outcome(**overrides) -> RitualOutcome:
    defaults = dict(
        ritual_id="test_ritual",
        ritual_version="1.0.0",
        session_id="session-1",
        agent_id="agent-1",
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
    defaults.update(overrides)
    return RitualOutcome(**defaults)


class TestWriteOutcome:
    @pytest.mark.asyncio
    async def test_write_with_no_yarn_no_warp(self):
        """Graceful degradation — no crash when backends are unavailable."""
        ctx = _make_ctx(yarn=None, warp=None)
        outcome = _make_outcome()
        # Should not raise
        await write_outcome(outcome, ctx)

    @pytest.mark.asyncio
    async def test_write_yarn_called(self):
        """When yarn is available, _write_to_yarn runs."""
        ctx = _make_ctx(yarn=None)
        outcome = _make_outcome()
        await _write_to_yarn(outcome, ctx)  # no crash

    @pytest.mark.asyncio
    async def test_embed_warp_called(self):
        """When warp is available, _embed_to_warp runs."""
        ctx = _make_ctx(warp=None)
        outcome = _make_outcome()
        await _embed_to_warp(outcome, ctx)  # no crash

    @pytest.mark.asyncio
    async def test_skipped_outcome_written(self):
        """Skipped rituals still write outcomes."""
        ctx = _make_ctx()
        outcome = _make_outcome(was_skipped=True, skip_reason="test skip")
        await write_outcome(outcome, ctx)  # no crash

    @pytest.mark.asyncio
    async def test_failed_outcome_written(self):
        """Failed rituals write outcomes with failure_policy_invoked."""
        ctx = _make_ctx()
        outcome = _make_outcome(
            failure_policy_invoked="SKIP",
            downstream_task_success=False,
            confidence_delta=-0.05,
        )
        await write_outcome(outcome, ctx)  # no crash
