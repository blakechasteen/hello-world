"""
Decision Collapse Stage
=======================
Step 7 of the weaving cycle: Convergence Engine collapses to discrete tool selection.

This stage:
- Applies policy to features
- Uses convergence engine to collapse probabilities
- Selects final tool/action
"""

import logging
import time
from typing import Dict, Any, Optional, List
import numpy as np

from hololoom.core.protocols.types import Query, Features, Context
from hololoom.core.convergence.engine import ConvergenceEngine, CollapseStrategy
from hololoom.core.loom.command import PatternSpec
from hololoom.core.orchestrator.protocols import StageResult


class DecisionCollapseStage:
    """
    Decision Collapse Stage (Step 7: Convergence Engine).

    Collapses continuous probability distributions to discrete tool selections
    using various strategies (argmax, epsilon-greedy, Thompson sampling, etc).
    """

    def __init__(
        self,
        policy_engine: Any,
        convergence_engine: ConvergenceEngine,
        collapse_strategy: CollapseStrategy = CollapseStrategy.BAYESIAN_BLEND,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the decision collapse stage.

        Args:
            policy_engine: The policy engine for computing tool probabilities
            convergence_engine: The convergence engine for collapsing to discrete choice
            collapse_strategy: Strategy for collapse (default: Bayesian blend)
            logger: Logger instance
        """
        self.policy = policy_engine
        self.convergence_engine = convergence_engine
        self.collapse_strategy = collapse_strategy
        self.logger = logger or logging.getLogger(__name__)

    async def execute(
        self,
        query: Query,
        context: Dict[str, Any],
        pattern_spec: PatternSpec,
    ) -> StageResult:
        """
        Execute decision collapse.

        Args:
            query: The user query
            context: Shared context (should contain features, shards, etc)
            pattern_spec: Pattern specification

        Returns:
            StageResult with selected tool and decision metadata
        """
        start_time = time.time()

        try:
            # Extract necessary context
            features = context.get("features", Features())
            shards = context.get("shards", [])
            dot_plasma = context.get("dot_plasma", {})
            warp_result = context.get("warp_result", {})

            # Create context object for policy
            policy_context = Context(shards=shards[:5])  # Limit context size

            # Apply policy to get tool probabilities
            action_plan, thompson_probs, bandit_stats = self.policy.forward(
                features, policy_context
            )

            # Log probabilities for debugging
            tool_probs = action_plan.tool_probs.cpu().numpy()
            self.logger.info(f"  [7a] Policy probabilities: {tool_probs}")
            if thompson_probs is not None:
                self.logger.info(f"  [7a] Thompson probs: {thompson_probs}")

            # Prepare for convergence
            tool_names = self.policy.tools
            probabilities = {
                tool_names[i]: float(tool_probs[i])
                for i in range(len(tool_names))
            }

            # Apply convergence engine to collapse to discrete choice
            collapse_result = self.convergence_engine.collapse(
                probabilities=probabilities,
                strategy=self.collapse_strategy,
                thompson_probs=thompson_probs,
                exploration_rate=0.1,  # Could be configurable
                temperature=1.0,  # Could be configurable
            )

            selected_tool = collapse_result.selected
            selected_idx = tool_names.index(selected_tool)

            duration_ms = (time.time() - start_time) * 1000

            self.logger.info(
                f"  [7] Convergence Engine selected: {selected_tool} "
                f"(strategy={collapse_result.strategy.value})"
            )

            # Update bandit statistics with the selected tool
            if hasattr(self.policy, 'bandit'):
                confidence = float(tool_probs[selected_idx])
                self.policy.bandit.update(selected_idx, confidence)
                self.logger.info(
                    f"  [7b] Updated bandit for tool {selected_idx} "
                    f"with confidence {confidence:.3f}"
                )

            return StageResult(
                stage_name="decision_collapse",
                duration_ms=duration_ms,
                data={
                    "selected_tool": selected_tool,
                    "tool_probabilities": probabilities,
                    "collapse_result": collapse_result,
                    "action_plan": action_plan,
                    "bandit_stats": bandit_stats,
                },
                metadata={
                    "collapse_strategy": collapse_result.strategy.value,
                    "exploration_used": collapse_result.exploration_used,
                    "confidence": float(tool_probs[selected_idx]),
                }
            )

        except Exception as e:
            self.logger.error(f"Decision collapse failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            return StageResult(
                stage_name="decision_collapse",
                duration_ms=duration_ms,
                success=False,
                error=str(e),
                data={}
            )

    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        return "decision_collapse"

    def get_required_context_keys(self) -> List[str]:
        """Get required context keys."""
        return ["features"]  # Minimum requirement

    def get_output_context_keys(self) -> List[str]:
        """Get output context keys."""
        return [
            "selected_tool",
            "tool_probabilities",
            "collapse_result",
            "action_plan",
            "bandit_stats"
        ]