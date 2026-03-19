"""Tests for weaverlet.controller — TaskController lifecycle."""

import pytest
from typing import List, Tuple
from dataclasses import dataclass

from hololoom.core.fabric.spacetime import Artifact, ArtifactType
from hololoom.weaverlet.schemas import (
    Budget,
    BudgetLedger,
    MilestoneDelta,
    TaskLease,
    TaskResult,
    TerminationReason,
    WorkingState,
)
from hololoom.weaverlet.controller import TaskController
from hololoom.weaverlet.evaluation.stagnation import StagnationDetector
from hololoom.weaverlet.evaluation.milestone import MilestoneDetector


# ── Mock RefinementLoop ──────────────────────────────────────────────────

class MockLoop:
    """
    Simulates Plan→Act→Revise. Each call to act() bumps the artifact.
    Scores are fed externally via the evaluator.
    """

    def __init__(self):
        self.plan_count = 0
        self.act_count = 0
        self.revise_count = 0

    async def plan(self, state: WorkingState) -> str:
        self.plan_count += 1
        return f"plan_{self.plan_count}"

    async def act(self, state: WorkingState, plan: str) -> WorkingState:
        self.act_count += 1
        state.artifact = Artifact(
            name="artifact",
            type=ArtifactType.DOCUMENT,
            content=f"iteration_{self.act_count}",
        )
        return state

    async def revise(self, state: WorkingState, score: float, critique: str) -> WorkingState:
        self.revise_count += 1
        state.score_history.append(score)
        state.critique_history.append(critique)
        state.iteration += 1
        if state.ledger:
            state.ledger.charge_iteration()
        return state


# ── Mock Evaluator ───────────────────────────────────────────────────────

class MockEvaluator:
    """Returns scores from a pre-set sequence."""

    def __init__(self, scores: List[Tuple[float, str]]):
        self._scores = list(scores)
        self._index = 0

    async def evaluate(self, state, context=None):
        if self._index < len(self._scores):
            result = self._scores[self._index]
            self._index += 1
            return result
        return (0.5, "default")


# ── Mock ShuttleSync ─────────────────────────────────────────────────────

class MockShuttle:
    def __init__(self):
        self.milestones_sent = []
        self.finals_sent = []

    async def send_milestone(self, delta, state):
        self.milestones_sent.append(delta)

    async def send_final(self, result):
        self.finals_sent.append(result)


# ── Mock ActionItemTracker ───────────────────────────────────────────────

class MockTracker:
    def __init__(self):
        self.actions = []
        self.started = []
        self.completed = []
        self._next_id = 0

    def add_action(self, description, priority):
        self._next_id += 1
        action = type("Action", (), {"id": f"act_{self._next_id}"})()
        self.actions.append({"description": description, "priority": priority})
        return action

    def start_action(self, action_id):
        self.started.append(action_id)

    def complete_action(self, action_id, success, quality_score):
        self.completed.append({
            "id": action_id,
            "success": success,
            "quality_score": quality_score,
        })


# ── Tests ────────────────────────────────────────────────────────────────

class TestTaskControllerGoalMet:

    @pytest.mark.asyncio
    async def test_goal_met_terminates(self):
        """Score >= 0.9 triggers GOAL_MET."""
        loop = MockLoop()
        evaluator = MockEvaluator([
            (0.3, "getting started"),
            (0.6, "improving"),
            (0.95, "excellent"),
        ])
        controller = TaskController(loop, evaluator)
        lease = TaskLease.create("test goal", Budget(max_iterations=10))

        result = await controller.execute(lease)
        assert result.termination_reason == TerminationReason.GOAL_MET
        assert result.final_score >= 0.9
        assert result.iterations_completed == 3

    @pytest.mark.asyncio
    async def test_score_history_populated(self):
        evaluator = MockEvaluator([(0.4, "ok"), (0.95, "done")])
        controller = TaskController(MockLoop(), evaluator)
        lease = TaskLease.create("test", Budget(max_iterations=10))

        result = await controller.execute(lease)
        assert len(result.score_history) == 2
        assert result.score_history[-1] >= 0.9


class TestTaskControllerBudgetExhausted:

    @pytest.mark.asyncio
    async def test_iteration_budget(self):
        """Runs out of iterations → BUDGET_EXHAUSTED."""
        evaluator = MockEvaluator([
            (0.2, "low"),
            (0.3, "still low"),
            (0.4, "moderate"),
        ])
        controller = TaskController(MockLoop(), evaluator)
        lease = TaskLease.create("test", Budget(max_iterations=2))

        result = await controller.execute(lease)
        assert result.termination_reason == TerminationReason.BUDGET_EXHAUSTED
        assert result.iterations_completed == 2


class TestTaskControllerStagnation:

    @pytest.mark.asyncio
    async def test_stagnation_terminates(self):
        """Flat scores trigger STAGNATION."""
        evaluator = MockEvaluator([
            (0.5, "ok"),
            (0.5, "same"),
            (0.5, "still same"),
            (0.5, "stuck"),
        ])
        stag = StagnationDetector()  # epsilon=0.02, window=3
        controller = TaskController(
            MockLoop(), evaluator,
            stagnation_detector=stag,
        )
        lease = TaskLease.create("test", Budget(max_iterations=20))

        result = await controller.execute(lease)
        assert result.termination_reason == TerminationReason.STAGNATION


class TestTaskControllerMilestones:

    @pytest.mark.asyncio
    async def test_milestones_emitted(self):
        """Crossing 0.25 and 0.50 thresholds produces milestones."""
        evaluator = MockEvaluator([
            (0.3, "past 25%"),
            (0.6, "past 50%"),
            (0.95, "done"),
        ])
        controller = TaskController(MockLoop(), evaluator)
        lease = TaskLease.create("test", Budget(max_iterations=10))

        result = await controller.execute(lease)
        types = {m.milestone_type for m in result.milestones}
        assert "threshold_25" in types
        assert "threshold_50" in types

    @pytest.mark.asyncio
    async def test_milestones_sent_via_shuttle(self):
        evaluator = MockEvaluator([(0.3, "ok"), (0.95, "done")])
        shuttle = MockShuttle()
        controller = TaskController(MockLoop(), evaluator, shuttle=shuttle)
        lease = TaskLease.create("test", Budget(max_iterations=10))

        result = await controller.execute(lease)
        assert len(shuttle.milestones_sent) > 0
        assert len(shuttle.finals_sent) == 1


class TestTaskControllerShuttle:

    @pytest.mark.asyncio
    async def test_final_sent(self):
        evaluator = MockEvaluator([(0.95, "done")])
        shuttle = MockShuttle()
        controller = TaskController(MockLoop(), evaluator, shuttle=shuttle)
        lease = TaskLease.create("test", Budget(max_iterations=10))

        await controller.execute(lease)
        assert len(shuttle.finals_sent) == 1
        assert shuttle.finals_sent[0].termination_reason == TerminationReason.GOAL_MET

    @pytest.mark.asyncio
    async def test_no_shuttle_no_error(self):
        evaluator = MockEvaluator([(0.95, "done")])
        controller = TaskController(MockLoop(), evaluator, shuttle=None)
        lease = TaskLease.create("test", Budget(max_iterations=10))

        result = await controller.execute(lease)
        assert result.termination_reason == TerminationReason.GOAL_MET


class TestTaskControllerTracker:

    @pytest.mark.asyncio
    async def test_tracker_integration(self):
        evaluator = MockEvaluator([(0.95, "done")])
        tracker = MockTracker()
        controller = TaskController(MockLoop(), evaluator, tracker=tracker)
        lease = TaskLease.create("my goal", Budget(max_iterations=10))

        await controller.execute(lease)
        assert len(tracker.actions) == 1
        assert tracker.actions[0]["description"] == "my goal"
        assert len(tracker.started) == 1
        assert len(tracker.completed) == 1
        assert tracker.completed[0]["success"] is True
        assert tracker.completed[0]["quality_score"] >= 0.9

    @pytest.mark.asyncio
    async def test_tracker_failure(self):
        evaluator = MockEvaluator([(0.2, "low"), (0.2, "stuck")])
        tracker = MockTracker()
        stag = StagnationDetector()
        controller = TaskController(
            MockLoop(), evaluator,
            tracker=tracker,
            stagnation_detector=stag,
        )
        lease = TaskLease.create("hard task", Budget(max_iterations=2))

        await controller.execute(lease)
        assert len(tracker.completed) == 1
        assert tracker.completed[0]["success"] is False

    @pytest.mark.asyncio
    async def test_no_tracker_no_error(self):
        evaluator = MockEvaluator([(0.95, "done")])
        controller = TaskController(MockLoop(), evaluator, tracker=None)
        lease = TaskLease.create("test", Budget(max_iterations=10))

        result = await controller.execute(lease)
        assert result is not None


class TestTaskControllerCustomThreshold:

    @pytest.mark.asyncio
    async def test_custom_success_threshold(self):
        evaluator = MockEvaluator([(0.7, "good enough")])
        controller = TaskController(
            MockLoop(), evaluator, success_threshold=0.6
        )
        lease = TaskLease.create("test", Budget(max_iterations=10))

        result = await controller.execute(lease)
        assert result.termination_reason == TerminationReason.GOAL_MET


class TestTaskControllerResult:

    @pytest.mark.asyncio
    async def test_result_fields(self):
        evaluator = MockEvaluator([(0.95, "done")])
        controller = TaskController(MockLoop(), evaluator)
        lease = TaskLease.create("test goal", Budget(max_iterations=5))

        result = await controller.execute(lease)
        assert result.lease_id == lease.lease_id
        assert result.final_artifact is not None
        assert result.wall_time_seconds > 0
        assert result.total_tokens >= 0
        assert result.total_tool_calls >= 0
        assert isinstance(result.termination_reason, TerminationReason)

    @pytest.mark.asyncio
    async def test_single_pass_mode(self):
        """Budget(max_iterations=1) is single-pass mode."""
        evaluator = MockEvaluator([(0.4, "one shot")])
        controller = TaskController(MockLoop(), evaluator)
        lease = TaskLease.create("single pass", Budget(max_iterations=1))

        result = await controller.execute(lease)
        assert result.iterations_completed == 1
        assert result.termination_reason == TerminationReason.BUDGET_EXHAUSTED
