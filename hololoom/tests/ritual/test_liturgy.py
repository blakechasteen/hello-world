"""Tests for LiturgySpec — cross-agent coordination."""

import pytest

from hololoom.core.ritual.liturgy import LiturgySpec, LiturgyAborted
from hololoom.core.ritual.policies import LiturgyFailurePolicy
from hololoom.core.ritual.types import RitualContext, RitualResult, TokenBudget


def _make_ctx(**overrides) -> RitualContext:
    defaults = dict(
        agent_id="master",
        agent_role="masterweaver",
        domain="universal",
        task_id="task-1",
        task_type="digest",
        task_intent="weekly summary",
        task_complexity="standard",
        yarn=None,
        warp=None,
        confidence=0.8,
        confidence_history=[0.8],
        ritual_outputs={},
        ritual_log=[],
        token_budget=TokenBudget(total=5000, used=0),
        time_budget=120.0,
        escalation_path=[],
        session_id="session-1",
    )
    defaults.update(overrides)
    return RitualContext(**defaults)


def _simple_merge(results):
    merged = {}
    for role, r in results.items():
        merged.update(r.produced)
    return RitualResult(
        success=True,
        produced={"Merged": merged},
        confidence_delta=0.0,
    )


def _make_liturgy(quorum=0.5, policy=LiturgyFailurePolicy.PARTIAL_QUORUM):
    return LiturgySpec(
        name="test_liturgy",
        participants=["agent_a", "agent_b", "agent_c"],
        quorum=quorum,
        sync_point="TEMPORAL_WEEKLY",
        contributions={
            "agent_a": "SummaryA",
            "agent_b": "SummaryB",
            "agent_c": "SummaryC",
        },
        consensus_output="MergedSummary",
        failure_policy=policy,
        merge_fn=_simple_merge,
    )


class TestLiturgyQuorum:
    @pytest.mark.asyncio
    async def test_full_quorum(self):
        liturgy = _make_liturgy(quorum=1.0)
        results = {
            "agent_a": RitualResult(success=True, produced={"A": 1}, confidence_delta=0.0),
            "agent_b": RitualResult(success=True, produced={"B": 2}, confidence_delta=0.0),
            "agent_c": RitualResult(success=True, produced={"C": 3}, confidence_delta=0.0),
        }
        ctx = _make_ctx()
        merged = await liturgy.execute(ctx, results)
        assert merged is not None
        assert merged.produced["Merged"] == {"A": 1, "B": 2, "C": 3}

    @pytest.mark.asyncio
    async def test_partial_quorum_proceeds(self):
        liturgy = _make_liturgy(quorum=0.5, policy=LiturgyFailurePolicy.PARTIAL_QUORUM)
        results = {
            "agent_a": RitualResult(success=True, produced={"A": 1}, confidence_delta=0.0),
            "agent_b": RitualResult(success=True, produced={"B": 2}, confidence_delta=0.0),
        }
        ctx = _make_ctx()
        merged = await liturgy.execute(ctx, results)
        assert merged is not None

    @pytest.mark.asyncio
    async def test_quorum_fraction_is_fraction(self):
        """0.5 quorum with 3 participants = need at least 2."""
        liturgy = _make_liturgy(quorum=0.5, policy=LiturgyFailurePolicy.ABORT)
        one_result = {
            "agent_a": RitualResult(success=True, produced={"A": 1}, confidence_delta=0.0),
        }
        ctx = _make_ctx()
        # 1/3 = 0.33 < 0.5 → abort
        with pytest.raises(LiturgyAborted):
            await liturgy.execute(ctx, one_result)

    @pytest.mark.asyncio
    async def test_defer_returns_none(self):
        liturgy = _make_liturgy(quorum=0.8, policy=LiturgyFailurePolicy.DEFER)
        one_result = {
            "agent_a": RitualResult(success=True, produced={"A": 1}, confidence_delta=0.0),
        }
        ctx = _make_ctx()
        result = await liturgy.execute(ctx, one_result)
        assert result is None

    @pytest.mark.asyncio
    async def test_abort_raises(self):
        liturgy = _make_liturgy(quorum=1.0, policy=LiturgyFailurePolicy.ABORT)
        partial = {
            "agent_a": RitualResult(success=True, produced={"A": 1}, confidence_delta=0.0),
        }
        ctx = _make_ctx()
        with pytest.raises(LiturgyAborted, match="test_liturgy"):
            await liturgy.execute(ctx, partial)

    @pytest.mark.asyncio
    async def test_empty_participants_no_division_error(self):
        liturgy = LiturgySpec(
            name="empty",
            participants=[],
            quorum=0.5,
            sync_point="TEMPORAL_WEEKLY",
            contributions={},
            consensus_output="X",
            failure_policy=LiturgyFailurePolicy.PARTIAL_QUORUM,
            merge_fn=_simple_merge,
        )
        ctx = _make_ctx()
        result = await liturgy.execute(ctx, {})
        assert result is not None  # 0/max(0,1) = 0 < 0.5, but partial_quorum proceeds
