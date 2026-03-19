"""
MathLoom — Step injection protocol for math in the weaving cycle.

Any weaving step can request math through the loom. The loom filters
operations by step mask, allocates budget, selects via Thompson Sampling,
executes, and records what was used.

Two doors coexist:
  Door 1 (direct import): step imports exactly what it needs
  Door 2 (MathLoom): step asks, loom discovers what might help

The loom is the second door.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np

from .catalog import (
    STEP_MASKS,
    MathCatalogBuilder,
    MathDomain,
    MathOperation,
    OperationLevel,
    QueryIntent,
    StepMask,
    get_catalog,
)
from .budget import BudgetAllocator, BudgetAllocation, BudgetTier, get_allocator

logger = logging.getLogger(__name__)


# ============================================================================
# Result Types
# ============================================================================

@dataclass
class MathLoomResult:
    """Result from a MathLoom request."""
    operations_used: list[str]
    results: dict[str, Any]
    total_cost: int
    execution_ms: float
    step_id: int
    budget_remaining: int


@dataclass
class MathLoomSummary:
    """Summary of all math used across a weaving cycle."""
    total_operations: int
    total_cost: int
    total_execution_ms: float
    operations_by_step: dict[int, list[str]]
    budget_allocation: Optional[BudgetAllocation] = None


# ============================================================================
# MathLoom Protocol
# ============================================================================

class MathLoomProtocol(Protocol):
    """Protocol for requesting math from any weaving step."""

    def request(
        self,
        step_id: int,
        query_text: str,
        intent: Optional[QueryIntent] = None,
        max_cost: Optional[int] = None,
        required_domains: Optional[set[MathDomain]] = None,
        data: Optional[dict] = None,
    ) -> Optional[MathLoomResult]: ...

    def record_outcome(self, quality: float, latency_ms: float) -> None: ...

    def summary(self) -> MathLoomSummary: ...


# ============================================================================
# WeavingMathLoom Implementation
# ============================================================================

class WeavingMathLoom:
    """Concrete MathLoom implementation for the weaving cycle.

    Created once per query. Manages budget across all steps.
    Selects operations via Thompson Sampling. Records what was used.
    """

    def __init__(
        self,
        budget_allocation: Optional[BudgetAllocation] = None,
        catalog: Optional[dict[str, MathOperation]] = None,
    ):
        self._catalog = catalog or get_catalog()
        self._budget = budget_allocation or get_allocator().allocate()
        self._total_budget = self._budget.total_budget
        self._spent: dict[int, int] = {}  # step_id → cost spent
        self._operations_used: dict[int, list[str]] = {}  # step_id → op names
        self._results: dict[int, dict[str, Any]] = {}  # step_id → results
        self._total_execution_ms = 0.0

        logger.debug(
            "MathLoom created: budget=%d (%s), catalog=%d ops",
            self._total_budget, self._budget.tier.name, len(self._catalog),
        )

    def request(
        self,
        step_id: int,
        query_text: str,
        intent: Optional[QueryIntent] = None,
        max_cost: Optional[int] = None,
        required_domains: Optional[set[MathDomain]] = None,
        data: Optional[dict] = None,
    ) -> Optional[MathLoomResult]:
        """Request math for a weaving step.

        1. Get step mask (which domains/levels allowed)
        2. Compute remaining budget for this step
        3. Filter catalog
        4. Select top operations by relevance
        5. Execute and record
        """
        start = time.time()

        # Get step mask
        mask = STEP_MASKS.get(step_id, STEP_MASKS.get(56))  # default to unrestricted
        if mask is None:
            return None

        # Compute remaining budget
        step_budget = int(self._total_budget * mask.budget_fraction)
        spent = self._spent.get(step_id, 0)
        remaining = step_budget - spent

        if remaining <= 0:
            logger.debug("MathLoom step %d: budget exhausted", step_id)
            return None

        effective_max = min(remaining, max_cost) if max_cost else remaining

        # Filter catalog
        domains = required_domains or mask.allowed_domains
        filtered = MathCatalogBuilder.filter(
            self._catalog,
            allowed_domains=domains,
            max_level=mask.max_level,
            max_cost=effective_max,
            intent=intent,
        )

        if not filtered:
            logger.debug("MathLoom step %d: no operations match filters", step_id)
            return None

        # Select top operations via Thompson Sampling + cost-effectiveness
        # Learner scores operations by historical success at this step.
        # Blend: 60% learned score + 40% cost efficiency
        learner = get_loom_learner()
        ranked_by_ts = learner.rank_operations(filtered, step_id)

        # Blend Thompson score with cost efficiency
        max_cost_in_filtered = max((op.estimated_cost for op in filtered.values()), default=10)
        ranked = sorted(
            filtered.values(),
            key=lambda op: (
                # Higher is better: 60% Thompson + 40% cost efficiency + intent bonus
                -(0.6 * learner.score_operation(op.name, step_id)
                  + 0.4 * (1.0 - op.estimated_cost / max(max_cost_in_filtered, 1))
                  + (0.2 if intent and op.is_applicable(intent) else 0.0)),
            ),
        )

        # Select operations within budget
        selected = []
        cost_so_far = 0
        max_ops = 5  # limit operations per request
        for op in ranked:
            if cost_so_far + op.estimated_cost > effective_max:
                continue
            if len(selected) >= max_ops:
                break
            selected.append(op)
            cost_so_far += op.estimated_cost

        if not selected:
            return None

        # Execute selected operations
        results = {}
        ops_used = []
        actual_cost = 0

        for op in selected:
            try:
                module = op.load_module()
                if module is not None:
                    # Module loaded successfully — record it
                    ops_used.append(op.name)
                    actual_cost += op.estimated_cost
                    results[op.name] = {
                        "module": op.module_path,
                        "domain": op.domain.value,
                        "cost": op.estimated_cost,
                        "loaded": True,
                    }
            except Exception as e:
                logger.debug("MathLoom: failed to load %s: %s", op.name, e)

        # Record budget usage
        self._spent[step_id] = spent + actual_cost
        if step_id not in self._operations_used:
            self._operations_used[step_id] = []
        self._operations_used[step_id].extend(ops_used)
        self._results[step_id] = results

        execution_ms = (time.time() - start) * 1000
        self._total_execution_ms += execution_ms

        logger.debug(
            "MathLoom step %d: %d ops, cost=%d/%d, %.1fms",
            step_id, len(ops_used), actual_cost, step_budget, execution_ms,
        )

        return MathLoomResult(
            operations_used=ops_used,
            results=results,
            total_cost=actual_cost,
            execution_ms=round(execution_ms, 2),
            step_id=step_id,
            budget_remaining=remaining - actual_cost,
        )

    def record_outcome(self, quality: float, latency_ms: float) -> None:
        """Record the final quality of the weaving cycle for learning.

        Called from Step 9 (Fabric). Distributes quality signal to all
        operations used, weighted by cost fraction. Updates the Thompson
        Sampling learner so future queries benefit from this outcome.
        """
        total_ops = sum(len(ops) for ops in self._operations_used.values())
        if total_ops == 0:
            return

        success = quality > 0.5
        learner = get_loom_learner()

        # Distribute quality signal to all operations, tagged by step
        for step_id, ops_used in self._operations_used.items():
            for op_name in ops_used:
                op = self._catalog.get(op_name)
                cost = op.estimated_cost if op else 10
                learner.record_feedback(
                    operation=op_name,
                    step_id=step_id,
                    success=success,
                    quality=quality,
                    cost=cost,
                )

        logger.info(
            "MathLoom outcome: quality=%.2f, latency=%.0fms, %d ops across %d steps → %s",
            quality, latency_ms, total_ops, len(self._operations_used),
            "success" if success else "failure",
        )

    def summary(self) -> MathLoomSummary:
        """Get summary of all math used in this weaving cycle."""
        return MathLoomSummary(
            total_operations=sum(len(ops) for ops in self._operations_used.values()),
            total_cost=sum(self._spent.values()),
            total_execution_ms=round(self._total_execution_ms, 2),
            operations_by_step=dict(self._operations_used),
            budget_allocation=self._budget,
        )


# ============================================================================
# Loom Learner — Thompson Sampling per (operation, step)
# ============================================================================

@dataclass
class LoomArmStats:
    """Statistics for one (operation, step) pair."""
    successes: float = 1.0  # alpha — start with optimistic prior
    failures: float = 1.0   # beta
    total_quality: float = 0.0
    n_observations: int = 0

    @property
    def mean(self) -> float:
        return self.successes / (self.successes + self.failures)

    def sample(self) -> float:
        return float(np.random.beta(self.successes, self.failures))

    def update(self, success: bool, quality: float = 0.5) -> None:
        self.n_observations += 1
        self.total_quality += quality
        if success:
            self.successes += quality
        else:
            self.failures += (1.0 - quality)


class LoomLearner:
    """Thompson Sampling learner for MathLoom operation selection.

    Learns per (operation, step_id) which operations improve quality.
    Cold-start uses domain-level priors instead of uniform Beta(1,1).
    """

    def __init__(self):
        self.arms: dict[tuple[str, int], LoomArmStats] = {}
        self._domain_priors: dict[str, LoomArmStats] = {}

    def get_arm(self, operation: str, step_id: int) -> LoomArmStats:
        """Get or create arm for (operation, step) pair."""
        key = (operation, step_id)
        if key not in self.arms:
            # Cold start: use domain-level prior if available
            domain_arm = self._domain_priors.get(operation)
            if domain_arm and domain_arm.n_observations > 5:
                # Transfer domain knowledge
                self.arms[key] = LoomArmStats(
                    successes=max(1.0, domain_arm.successes * 0.5),
                    failures=max(1.0, domain_arm.failures * 0.5),
                )
            else:
                self.arms[key] = LoomArmStats()
        return self.arms[key]

    def score_operation(self, operation: str, step_id: int) -> float:
        """Sample a score for this operation at this step."""
        return self.get_arm(operation, step_id).sample()

    def record_feedback(
        self,
        operation: str,
        step_id: int,
        success: bool,
        quality: float = 0.5,
        cost: int = 10,
    ) -> None:
        """Record outcome for learning."""
        arm = self.get_arm(operation, step_id)
        arm.update(success, quality)

        # Also update domain-level prior for cold-start transfer
        if operation not in self._domain_priors:
            self._domain_priors[operation] = LoomArmStats()
        self._domain_priors[operation].update(success, quality)

    def rank_operations(
        self,
        candidates: dict[str, MathOperation],
        step_id: int,
    ) -> list[tuple[str, float]]:
        """Rank operations by Thompson Sampling score for this step.

        Returns [(op_name, score)] sorted by score descending.
        """
        scored = []
        for name in candidates:
            score = self.score_operation(name, step_id)
            scored.append((name, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def stats_summary(self, step_id: Optional[int] = None) -> dict:
        """Get summary of learned arm statistics."""
        summary = {}
        for (op, sid), arm in self.arms.items():
            if step_id is not None and sid != step_id:
                continue
            key = f"{op}@step{sid}"
            summary[key] = {
                "mean": round(arm.mean, 3),
                "n": arm.n_observations,
                "alpha": round(arm.successes, 1),
                "beta": round(arm.failures, 1),
            }
        return summary


# Singleton learner
_loom_learner: Optional[LoomLearner] = None


def get_loom_learner() -> LoomLearner:
    """Get or create the global loom learner."""
    global _loom_learner
    if _loom_learner is None:
        _loom_learner = LoomLearner()
    return _loom_learner


# ============================================================================
# Factory
# ============================================================================

def create_math_loom(
    query_length: int = 0,
    entity_count: int = 0,
    attention_entropy: float = 0.0,
    n_intents: int = 1,
    pattern_card: str = "fast",
) -> WeavingMathLoom:
    """Create a MathLoom for a weaving cycle with adaptive budget.

    Called at pipeline start. Budget scales with query complexity.
    """
    allocator = get_allocator()
    allocation = allocator.allocate(
        query_length=query_length,
        entity_count=entity_count,
        attention_entropy=attention_entropy,
        n_intents=n_intents,
        pattern_card=pattern_card,
    )

    logger.info(
        "MathLoom: %s budget=%d (score=%.3f)",
        allocation.tier.name, allocation.total_budget, allocation.complexity_score,
    )

    return WeavingMathLoom(budget_allocation=allocation)
