"""
Weaverlet Schemas
=================

Data contracts for the iterative worker system.

Design:
- Frozen types for records (TaskLease, IterationOutput, MilestoneDelta, Budget)
- Mutable types for evolving state (WorkingState, BudgetLedger)
- Enums for closed sets (TerminationReason)

All budget accounting flows through BudgetLedger — never mutate Budget directly.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from hololoom.core.fabric.spacetime import Artifact

# ── Budget ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Budget:
    """Immutable resource limits for a task."""
    max_iterations: int = 10
    max_tokens: int = 100_000
    max_wall_seconds: float = 600.0
    max_tool_calls: int = 50


class BudgetExhaustedError(Exception):
    """Raised when any budget dimension is spent."""
    pass


@dataclass
class BudgetLedger:
    """
    Mutable accounting against a frozen Budget.

    Every model call charges tokens, every tool call charges tool_calls,
    every iteration charges iterations. Wall time is checked passively.
    """
    budget: Budget
    iterations_used: int = 0
    tokens_used: int = 0
    tool_calls_used: int = 0
    wall_start: float = field(default_factory=time.time)

    def charge_tokens(self, n: int) -> None:
        self.tokens_used += n
        if self.tokens_used > self.budget.max_tokens:
            raise BudgetExhaustedError(
                f"Token budget exhausted: {self.tokens_used}/{self.budget.max_tokens}"
            )

    def charge_tool_call(self) -> None:
        self.tool_calls_used += 1
        if self.tool_calls_used > self.budget.max_tool_calls:
            raise BudgetExhaustedError(
                f"Tool call budget exhausted: {self.tool_calls_used}/{self.budget.max_tool_calls}"
            )

    def charge_iteration(self) -> None:
        self.iterations_used += 1

    def wall_elapsed(self) -> float:
        return time.time() - self.wall_start

    def is_exhausted(self) -> bool:
        if self.iterations_used >= self.budget.max_iterations:
            return True
        if self.tokens_used >= self.budget.max_tokens:
            return True
        if self.tool_calls_used >= self.budget.max_tool_calls:
            return True
        if self.wall_elapsed() >= self.budget.max_wall_seconds:
            return True
        return False

    def exhaustion_reason(self) -> str | None:
        if self.iterations_used >= self.budget.max_iterations:
            return f"iterations: {self.iterations_used}/{self.budget.max_iterations}"
        if self.tokens_used >= self.budget.max_tokens:
            return f"tokens: {self.tokens_used}/{self.budget.max_tokens}"
        if self.tool_calls_used >= self.budget.max_tool_calls:
            return f"tool_calls: {self.tool_calls_used}/{self.budget.max_tool_calls}"
        if self.wall_elapsed() >= self.budget.max_wall_seconds:
            return f"wall_time: {self.wall_elapsed():.1f}s/{self.budget.max_wall_seconds}s"
        return None

    def remaining(self) -> dict[str, float]:
        """Remaining budget as fractions (0.0-1.0) per dimension."""
        return {
            "iterations": max(0, 1 - self.iterations_used / self.budget.max_iterations),
            "tokens": max(0, 1 - self.tokens_used / self.budget.max_tokens),
            "tool_calls": max(0, 1 - self.tool_calls_used / self.budget.max_tool_calls),
            "wall_time": max(0, 1 - self.wall_elapsed() / self.budget.max_wall_seconds),
        }

    def remaining_fraction(self) -> float:
        """Minimum remaining fraction across all dimensions."""
        return min(self.remaining().values())


# ── Task Lease ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskLease:
    """
    Immutable task assignment. Issued once, never modified.

    Maps to the brief's TaskLease + replaces DomainWorker's WorkerConfig.
    """
    lease_id: str
    goal: str
    context_bundle: dict[str, Any]
    budget: Budget
    thread_id: str | None = None      # if tied to a Tapestry thread
    tapestry_id: str | None = None    # if tied to a Tapestry session
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        goal: str,
        budget: Budget | None = None,
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> TaskLease:
        """Factory with sensible defaults."""
        return cls(
            lease_id=uuid.uuid4().hex[:12],
            goal=goal,
            context_bundle=context or {},
            budget=budget or Budget(),
            **kwargs,
        )


# ── Working State ─────────────────────────────────────────────────────────

@dataclass
class WorkingState:
    """
    Mutable scratchpad that evolves across iterations.

    All loop state lives here — the loop itself is stateless.
    """
    iteration: int = 0
    goal: str = ""
    artifact: Artifact | None = None
    score_history: list[float] = field(default_factory=list)
    critique_history: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    ledger: BudgetLedger | None = None

    @property
    def latest_score(self) -> float:
        return self.score_history[-1] if self.score_history else 0.0

    @property
    def previous_score(self) -> float:
        if len(self.score_history) >= 2:
            return self.score_history[-2]
        return 0.0


# ── Iteration Output ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class IterationOutput:
    """Sealed record of one iteration's work."""
    iteration_id: str
    progress_score: float
    updated_artifact: Artifact | None
    critique: str
    next_action: str
    evidence_refs: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    tokens_used: int = 0
    tool_calls_used: int = 0


# ── Milestones ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MilestoneDelta:
    """Emitted when a progress threshold is crossed."""
    milestone_type: str     # "threshold_25", "first_artifact", etc.
    score_before: float
    score_after: float
    iteration: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Termination ───────────────────────────────────────────────────────────

class TerminationReason(Enum):
    GOAL_MET = "goal_met"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STAGNATION = "stagnation"
    BLOCKER = "blocker"
    CANCELLED = "cancelled"


# ── Task Result ───────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    """Final output from TaskController.execute()."""
    lease_id: str
    final_artifact: Artifact | None
    final_score: float
    score_history: list[float]
    iterations_completed: int
    termination_reason: TerminationReason
    milestones: list[MilestoneDelta] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    total_tokens: int = 0
    total_tool_calls: int = 0
