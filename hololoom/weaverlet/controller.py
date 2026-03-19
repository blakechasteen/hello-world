"""
Task Controller
================

Owns the Weaverlet lifecycle:

1. Accept a TaskLease
2. Create BudgetLedger + WorkingState
3. Drive the iteration loop (plan -> act -> evaluate -> revise)
4. Enforce termination conditions (goal met, budget, stagnation)
5. Emit milestones via ShuttleSync
6. Feed completion data back to ActionItemTracker (Thompson priors)
7. Return TaskResult

This replaces run_worker_loop() from domain_harness/worker.py.
Single-pass mode: Budget(max_iterations=1).
"""

from __future__ import annotations

import logging
import time

from hololoom.weaverlet.evaluation.evaluator import WeaverletEvaluator
from hololoom.weaverlet.evaluation.milestone import MilestoneDetector
from hololoom.weaverlet.evaluation.stagnation import StagnationDetector
from hololoom.weaverlet.loop import RefinementLoop
from hololoom.weaverlet.schemas import (
    BudgetExhaustedError,
    BudgetLedger,
    TaskLease,
    TaskResult,
    TerminationReason,
    WorkingState,
)
from hololoom.weaverlet.shuttle.sync import ShuttleSync

logger = logging.getLogger(__name__)

# Score at or above this threshold triggers GOAL_MET termination.
SUCCESS_THRESHOLD = 0.9


class TaskController:

    def __init__(
        self,
        loop: RefinementLoop,
        evaluator: WeaverletEvaluator,
        milestone_detector: MilestoneDetector | None = None,
        stagnation_detector: StagnationDetector | None = None,
        shuttle: ShuttleSync | None = None,
        tracker=None,  # ActionItemTracker (optional, avoids hard import)
        success_threshold: float = SUCCESS_THRESHOLD,
    ):
        self.loop = loop
        self.evaluator = evaluator
        self.milestone_detector = milestone_detector or MilestoneDetector()
        self.stagnation_detector = stagnation_detector or StagnationDetector()
        self.shuttle = shuttle
        self.tracker = tracker
        self.success_threshold = success_threshold

    async def execute(self, lease: TaskLease) -> TaskResult:
        """
        Main entry point. Runs the iterative loop until termination.

        Returns TaskResult with final artifact, score history, and reason.
        """
        wall_start = time.time()
        ledger = BudgetLedger(budget=lease.budget)
        state = WorkingState(goal=lease.goal, ledger=ledger)
        milestones = []

        # Track via ActionItemTracker if available
        action_id = None
        if self.tracker:
            action = self.tracker.add_action(
                description=lease.goal, priority=0.8
            )
            self.tracker.start_action(action.id)
            action_id = action.id

        logger.info(
            f"Weaverlet started: lease={lease.lease_id}, "
            f"goal={lease.goal[:60]}, budget={lease.budget}"
        )

        termination_reason = TerminationReason.BUDGET_EXHAUSTED  # default

        try:
            while True:
                # Check termination before each iteration
                reason = self._check_termination(state)
                if reason:
                    termination_reason = reason
                    break

                prev_score = state.latest_score

                # Phase 1: Plan
                plan = await self.loop.plan(state)

                # Phase 2: Act
                state = await self.loop.act(state, plan)

                # Phase 3: Evaluate
                score, critique = await self.evaluator.evaluate(state)

                # Phase 4: Revise
                state = await self.loop.revise(state, score, critique)

                # Check milestones
                new_milestones = self.milestone_detector.check(
                    prev_score, score, state.iteration, state
                )
                milestones.extend(new_milestones)
                for ms in new_milestones:
                    if self.shuttle:
                        await self.shuttle.send_milestone(ms, state)

                logger.info(
                    f"Iteration {state.iteration}: "
                    f"score={score:.3f}, tokens={ledger.tokens_used}, "
                    f"tools={ledger.tool_calls_used}"
                )

        except BudgetExhaustedError as e:
            logger.info(f"Budget exhausted mid-iteration: {e}")
            termination_reason = TerminationReason.BUDGET_EXHAUSTED

        wall_time = time.time() - wall_start

        result = TaskResult(
            lease_id=lease.lease_id,
            final_artifact=state.artifact,
            final_score=state.latest_score,
            score_history=state.score_history,
            iterations_completed=state.iteration,
            termination_reason=termination_reason,
            milestones=milestones,
            wall_time_seconds=wall_time,
            total_tokens=ledger.tokens_used,
            total_tool_calls=ledger.tool_calls_used,
        )

        # Send final result to daemon
        if self.shuttle:
            try:
                await self.shuttle.send_final(result)
            except Exception as e:
                logger.warning(f"Failed to send final result: {e}")

        # Feed back into Thompson priors
        if self.tracker and action_id:
            try:
                self.tracker.complete_action(
                    action_id,
                    success=(termination_reason == TerminationReason.GOAL_MET),
                    quality_score=state.latest_score,
                )
            except Exception as e:
                logger.warning(f"Failed to update tracker: {e}")

        logger.info(
            f"Weaverlet finished: reason={termination_reason.value}, "
            f"score={state.latest_score:.3f}, "
            f"iterations={state.iteration}, "
            f"wall={wall_time:.1f}s"
        )

        return result

    def _check_termination(self, state: WorkingState) -> TerminationReason | None:
        """Check all termination conditions. Returns reason or None."""
        # Goal met
        if state.latest_score >= self.success_threshold:
            return TerminationReason.GOAL_MET

        # Budget exhausted
        if state.ledger and state.ledger.is_exhausted():
            return TerminationReason.BUDGET_EXHAUSTED

        # Stagnation
        if self.stagnation_detector.is_stagnant(state.score_history):
            return TerminationReason.STAGNATION

        return None
