"""
Routing Orchestrator - Composable Backend Selection + Execution
==============================================================

Combines:
- Routing Strategy (WHICH backend)
- Execution Pattern (HOW to execute)

Both are modular, swappable, and A/B testable.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .protocol import (
    RoutingStrategy,
    RoutingDecision,
    RoutingOutcome,
    BackendType
)
from .execution_patterns import (
    ExecutionPattern,
    ExecutionPlan,
    ExecutionEngine,
    FeedForwardEngine,
    RecursiveEngine,
    StrangeLoopEngine,
    ChainingEngine,
    ParallelEngine,
    select_execution_pattern
)
from HoloLoom.memory.protocol import MemoryQuery, RetrievalResult


@dataclass
class RoutingConfig:
    """
    Configuration for routing orchestrator.

    Fully composable - swap any component:
    - routing_strategy: Which backend to select
    - execution_engines: How to execute (multiple options)
    - default_pattern: Fallback execution pattern
    """
    routing_strategy: RoutingStrategy
    execution_engines: Dict[ExecutionPattern, ExecutionEngine]
    default_pattern: ExecutionPattern = ExecutionPattern.FEED_FORWARD
    enable_auto_pattern_selection: bool = True


class RoutingOrchestrator:
    """
    Orchestrates routing + execution.

    Modular design:
    - Routing strategy is pluggable (rule-based, learned, hybrid)
    - Execution engines are pluggable (feed-forward, recursive, etc.)
    - Both can be A/B tested independently

    Usage:
        # Create with specific components
        orchestrator = RoutingOrchestrator(
            routing_strategy=LearnedRouter(),
            execution_engines={
                ExecutionPattern.FEED_FORWARD: FeedForwardEngine(),
                ExecutionPattern.RECURSIVE: RecursiveEngine(),
                ExecutionPattern.STRANGE_LOOP: StrangeLoopEngine(router),
            }
        )

        # Execute query
        result = await orchestrator.execute(query, backends)

        # Record outcome for learning
        orchestrator.record_outcome(outcome)
    """

    def __init__(
        self,
        routing_strategy: RoutingStrategy,
        execution_engines: Optional[Dict[ExecutionPattern, ExecutionEngine]] = None,
        default_pattern: ExecutionPattern = ExecutionPattern.FEED_FORWARD,
        enable_auto_pattern_selection: bool = True
    ):
        self.routing_strategy = routing_strategy
        self.execution_engines = execution_engines or self._default_engines()
        self.default_pattern = default_pattern
        self.enable_auto_pattern_selection = enable_auto_pattern_selection

        self.outcomes: List[RoutingOutcome] = []

    def _default_engines(self) -> Dict[ExecutionPattern, ExecutionEngine]:
        """Create default execution engines."""
        return {
            ExecutionPattern.FEED_FORWARD: FeedForwardEngine(),
            ExecutionPattern.RECURSIVE: RecursiveEngine(),
            ExecutionPattern.CHAIN: ChainingEngine(),
            ExecutionPattern.PARALLEL: ParallelEngine(),
        }

    async def execute(
        self,
        query: MemoryQuery,
        backends: Dict[BackendType, Any],
        execution_pattern: Optional[ExecutionPattern] = None
    ) -> RetrievalResult:
        """
        Execute query with routing + execution.

        Steps:
        1. Route query to backend (using routing strategy)
        2. Select execution pattern (auto or specified)
        3. Build execution plan
        4. Execute using appropriate engine

        Args:
            query: Query to execute
            backends: Available backend implementations
            execution_pattern: Optional override for execution pattern

        Returns:
            Results from execution
        """

        # Step 1: Route to backend
        available = list(backends.keys())
        routing_decision = self.routing_strategy.select_backend(
            query.text,
            available,
            context={'user_id': query.user_id}
        )

        # Step 2: Select execution pattern
        if execution_pattern is None and self.enable_auto_pattern_selection:
            execution_pattern = select_execution_pattern(
                query,
                routing_decision.confidence,
                available
            )
        elif execution_pattern is None:
            execution_pattern = self.default_pattern

        # Step 3: Build execution plan
        plan = self._build_plan(routing_decision, execution_pattern, available)

        # Step 4: Execute
        engine = self.execution_engines.get(execution_pattern)
        if engine is None:
            # Fallback to feed-forward
            engine = self.execution_engines[ExecutionPattern.FEED_FORWARD]

        result = await engine.execute(query, plan, backends)

        # Add routing metadata to result
        result.metadata['routing_decision'] = {
            'backend': routing_decision.backend_type.value,
            'confidence': routing_decision.confidence,
            'reasoning': routing_decision.reasoning
        }
        result.metadata['execution_pattern'] = execution_pattern.value

        return result

    def _build_plan(
        self,
        routing_decision: RoutingDecision,
        execution_pattern: ExecutionPattern,
        available_backends: List[BackendType]
    ) -> ExecutionPlan:
        """Build execution plan from routing decision."""

        # Secondary backends are the alternatives from routing
        secondaries = [
            b for b in routing_decision.alternatives
            if b in available_backends
        ]

        # Configure based on pattern
        if execution_pattern == ExecutionPattern.FEED_FORWARD:
            max_iterations = 1
        elif execution_pattern == ExecutionPattern.RECURSIVE:
            max_iterations = 3
        elif execution_pattern == ExecutionPattern.STRANGE_LOOP:
            max_iterations = 2
        else:
            max_iterations = 1

        return ExecutionPlan(
            pattern=execution_pattern,
            primary_backend=routing_decision.backend_type,
            secondary_backends=secondaries[:3],  # Max 3 secondaries
            max_iterations=max_iterations,
            confidence_threshold=0.7,
            metadata={'routing_confidence': routing_decision.confidence}
        )

    def record_outcome(self, outcome: RoutingOutcome):
        """Record outcome for learning."""
        self.outcomes.append(outcome)
        self.routing_strategy.record_outcome(outcome)

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            'routing_strategy': self.routing_strategy.get_statistics(),
            'total_executions': len(self.outcomes),
            'execution_patterns_available': list(self.execution_engines.keys()),
        }


# ============================================================================
# A/B Testing for Complete Orchestrator
# ============================================================================

class OrchestratorExperiment:
    """
    A/B test complete orchestrators (routing + execution).

    Can test:
    - Different routing strategies
    - Different execution patterns
    - Different combinations

    Example:
        experiment = OrchestratorExperiment()

        # Variant A: Rule-based + Feed-forward
        experiment.add_variant(
            "baseline",
            routing=RuleBasedRouter(),
            execution={ExecutionPattern.FEED_FORWARD: FeedForwardEngine()}
        )

        # Variant B: Learned + Recursive
        experiment.add_variant(
            "learned_recursive",
            routing=LearnedRouter(),
            execution={ExecutionPattern.RECURSIVE: RecursiveEngine()}
        )

        # Run experiment
        result = await experiment.execute(query, backends)
        experiment.record_outcome(outcome)

        # Get winner
        winner = experiment.get_winner()
    """

    def __init__(self):
        self.variants: Dict[str, RoutingOrchestrator] = {}
        self.variant_weights: Dict[str, float] = {}
        self.outcomes_by_variant: Dict[str, List[RoutingOutcome]] = {}

    def add_variant(
        self,
        name: str,
        routing: RoutingStrategy,
        execution: Dict[ExecutionPattern, ExecutionEngine],
        weight: float = 1.0
    ):
        """Add orchestrator variant to experiment."""
        orchestrator = RoutingOrchestrator(
            routing_strategy=routing,
            execution_engines=execution
        )

        self.variants[name] = orchestrator
        self.variant_weights[name] = weight
        self.outcomes_by_variant[name] = []

    async def execute(
        self,
        query: MemoryQuery,
        backends: Dict[BackendType, Any]
    ) -> RetrievalResult:
        """Execute using weighted random variant."""
        import random

        # Select variant
        variant_name = self._select_variant()
        orchestrator = self.variants[variant_name]

        # Execute
        result = await orchestrator.execute(query, backends)

        # Tag with variant
        result.metadata['experiment_variant'] = variant_name

        return result

    def _select_variant(self) -> str:
        """Weighted random variant selection."""
        import random

        total_weight = sum(self.variant_weights.values())
        rand = random.uniform(0, total_weight)

        cumulative = 0
        for variant, weight in self.variant_weights.items():
            cumulative += weight
            if rand <= cumulative:
                return variant

        return list(self.variants.keys())[0]

    def record_outcome(self, outcome: RoutingOutcome):
        """Record outcome to appropriate variant."""
        variant_name = outcome.decision.metadata.get('experiment_variant')

        if variant_name and variant_name in self.variants:
            self.outcomes_by_variant[variant_name].append(outcome)
            self.variants[variant_name].record_outcome(outcome)

    def get_winner(self) -> str:
        """Determine winning variant."""
        qualified = [
            (name, outcomes)
            for name, outcomes in self.outcomes_by_variant.items()
            if len(outcomes) >= 10
        ]

        if not qualified:
            return "insufficient_data"

        # Compute success rate for each
        def success_rate(outcomes):
            if not outcomes:
                return 0
            successes = sum(1 for o in outcomes if o.avg_relevance > 0.7)
            return successes / len(outcomes)

        ranked = sorted(
            qualified,
            key=lambda x: success_rate(x[1]),
            reverse=True
        )

        return ranked[0][0]

    def generate_report(self) -> Dict[str, Any]:
        """Generate experiment report."""
        def compute_metrics(outcomes):
            if not outcomes:
                return {
                    'success_rate': 0,
                    'avg_relevance': 0,
                    'avg_latency': 0,
                    'total': 0
                }

            successes = sum(1 for o in outcomes if o.avg_relevance > 0.7)

            return {
                'success_rate': successes / len(outcomes),
                'avg_relevance': sum(o.avg_relevance for o in outcomes) / len(outcomes),
                'avg_latency': sum(o.latency_ms for o in outcomes) / len(outcomes),
                'total': len(outcomes)
            }

        return {
            'winner': self.get_winner(),
            'variants': {
                name: compute_metrics(outcomes)
                for name, outcomes in self.outcomes_by_variant.items()
            }
        }


# ============================================================================
# Module Testing Helpers
# ============================================================================

def create_test_orchestrator(
    routing_type: str = "rule_based",
    execution_patterns: Optional[List[ExecutionPattern]] = None
) -> RoutingOrchestrator:
    """
    Factory for creating test orchestrators.

    Makes testing different combinations easy:

    Example:
        # Test rule-based + feed-forward
        orch1 = create_test_orchestrator("rule_based", [ExecutionPattern.FEED_FORWARD])

        # Test learned + recursive
        orch2 = create_test_orchestrator("learned", [ExecutionPattern.RECURSIVE])

        # Test with all patterns
        orch3 = create_test_orchestrator("rule_based")  # All patterns
    """
    from .rule_based import RuleBasedRouter
    from .learned import LearnedRouter

    # Select routing strategy
    if routing_type == "rule_based":
        routing = RuleBasedRouter()
    elif routing_type == "learned":
        routing = LearnedRouter()
    else:
        raise ValueError(f"Unknown routing type: {routing_type}")

    # Select execution engines
    if execution_patterns is None:
        # All patterns
        execution_engines = {
            ExecutionPattern.FEED_FORWARD: FeedForwardEngine(),
            ExecutionPattern.RECURSIVE: RecursiveEngine(),
            ExecutionPattern.STRANGE_LOOP: StrangeLoopEngine(routing),
            ExecutionPattern.CHAIN: ChainingEngine(),
            ExecutionPattern.PARALLEL: ParallelEngine(),
        }
    else:
        # Only specified patterns
        execution_engines = {}
        for pattern in execution_patterns:
            if pattern == ExecutionPattern.FEED_FORWARD:
                execution_engines[pattern] = FeedForwardEngine()
            elif pattern == ExecutionPattern.RECURSIVE:
                execution_engines[pattern] = RecursiveEngine()
            elif pattern == ExecutionPattern.STRANGE_LOOP:
                execution_engines[pattern] = StrangeLoopEngine(routing)
            elif pattern == ExecutionPattern.CHAIN:
                execution_engines[pattern] = ChainingEngine()
            elif pattern == ExecutionPattern.PARALLEL:
                execution_engines[pattern] = ParallelEngine()

    return RoutingOrchestrator(
        routing_strategy=routing,
        execution_engines=execution_engines
    )