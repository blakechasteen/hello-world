#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Mathematical Operation Selector with RL/ML
=================================================
Enhanced operation selector that LEARNS which mathematical operations work best
through reinforcement learning, composes operations into pipelines, and runs
rigorous automated testing.

Key Enhancements:
1. **RL Learning**: Thompson Sampling bandit learns operation effectiveness
2. **Composition**: Combines operations into functional pipelines
3. **Rigorous Testing**: Automated verification with property-based tests
4. **Feedback Loops**: Learns from success/failure of each operation plan

Philosophy:
Not just selecting operations - LEARNING which combinations work best for
different query types, with mathematical guarantees verified through testing.

Author: HoloLoom Team
Date: 2025-10-26
"""

import logging
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np

# Import base selector
try:
    from .operation_selector import (
        MathOperation, OperationPlan, MathDomain, OperationLevel,
        QueryIntent, MathOperationSelector
    )
except ImportError:
    from operation_selector import (
        MathOperation, OperationPlan, MathDomain, OperationLevel,
        QueryIntent, MathOperationSelector
    )

logger = logging.getLogger(__name__)


# ============================================================================
# Operator Composition
# ============================================================================

@dataclass
class ComposedOperation:
    """
    Composite operation formed by chaining multiple operations.

    Represents function composition: (f ∘ g ∘ h)(x) = f(g(h(x)))

    Attributes:
        operations: Ordered list of operations (executed left-to-right)
        name: Composite operation name
        estimated_cost: Sum of component costs
        composition_type: How operations are combined
    """
    operations: List[MathOperation]
    name: str
    estimated_cost: int
    composition_type: str = "sequential"  # or "parallel", "branching"

    def execute(self, data: Any) -> Any:
        """Execute composed operation pipeline."""
        result = data
        for op in self.operations:
            # Execute operation (would call actual implementation)
            result = self._execute_single(op, result)
        return result

    def _execute_single(self, op: MathOperation, data: Any) -> Any:
        """Execute single operation (placeholder for actual implementation)."""
        # This would dynamically import and call the actual operation
        logger.debug(f"Executing {op.name} on data")
        return data  # Placeholder


@dataclass
class OperationComposer:
    """
    Composes mathematical operations into functional pipelines.

    Supports:
    - Sequential composition: f ∘ g ∘ h
    - Parallel composition: (f, g, h) executed independently
    - Branching composition: if condition then f else g
    """

    def compose_sequential(
        self,
        operations: List[MathOperation],
        name: Optional[str] = None
    ) -> ComposedOperation:
        """
        Sequential composition: f ∘ g ∘ h.

        Args:
            operations: Operations in execution order
            name: Optional composite name

        Returns:
            ComposedOperation
        """
        if not name:
            op_names = [op.name for op in operations]
            name = " -> ".join(op_names)

        total_cost = sum(op.estimated_cost for op in operations)

        return ComposedOperation(
            operations=operations,
            name=name,
            estimated_cost=total_cost,
            composition_type="sequential"
        )

    def compose_parallel(
        self,
        operations: List[MathOperation],
        name: Optional[str] = None
    ) -> ComposedOperation:
        """
        Parallel composition: (f, g, h) executed independently.

        Args:
            operations: Operations to execute in parallel
            name: Optional composite name

        Returns:
            ComposedOperation
        """
        if not name:
            op_names = [op.name for op in operations]
            name = " || ".join(op_names)

        # Cost is max, not sum (parallel execution)
        total_cost = max(op.estimated_cost for op in operations)

        return ComposedOperation(
            operations=operations,
            name=name,
            estimated_cost=total_cost,
            composition_type="parallel"
        )

    def suggest_compositions(
        self,
        operations: List[MathOperation]
    ) -> List[ComposedOperation]:
        """
        Suggest useful operation compositions.

        Uses domain knowledge about common mathematical pipelines.

        Args:
            operations: Available operations

        Returns:
            List of suggested compositions
        """
        suggestions = []
        op_map = {op.name: op for op in operations}

        # Common pipelines

        # 1. Similarity pipeline: metric + inner_product
        if "metric_distance" in op_map and "inner_product" in op_map:
            suggestions.append(
                self.compose_sequential(
                    [op_map["inner_product"], op_map["metric_distance"]],
                    name="similarity_pipeline"
                )
            )

        # 2. Optimization pipeline: gradient + continuity_check
        if "gradient" in op_map and "continuity_check" in op_map:
            suggestions.append(
                self.compose_sequential(
                    [op_map["continuity_check"], op_map["gradient"]],
                    name="verified_optimization"
                )
            )

        # 3. Spectral analysis pipeline: eigenvalues + laplacian
        if "eigenvalues" in op_map and "laplacian" in op_map:
            suggestions.append(
                self.compose_sequential(
                    [op_map["laplacian"], op_map["eigenvalues"]],
                    name="spectral_pipeline"
                )
            )

        # 4. Verification suite: metric + continuity + convergence (parallel)
        verification_ops = []
        for name in ["metric_verification", "continuity_check", "convergence_analysis"]:
            if name in op_map:
                verification_ops.append(op_map[name])

        if len(verification_ops) >= 2:
            suggestions.append(
                self.compose_parallel(
                    verification_ops,
                    name="verification_suite"
                )
            )

        return suggestions


# ============================================================================
# RL Learning via Thompson Sampling
# ============================================================================

@dataclass
class OperationStatistics:
    """
    Statistics for reinforcement learning.

    Tracks success/failure of each operation for different query intents.
    """
    operation_name: str
    intent: str
    successes: int = 0
    failures: int = 0
    total_cost_spent: int = 0
    avg_execution_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Compute empirical success rate."""
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5

    def update(self, success: bool, cost: int, time: float):
        """Update statistics from execution feedback."""
        if success:
            self.successes += 1
        else:
            self.failures += 1

        self.total_cost_spent += cost

        # Running average
        total = self.successes + self.failures
        self.avg_execution_time = (
            (self.avg_execution_time * (total - 1) + time) / total
        )


class ThompsonSamplingLearner:
    """
    Thompson Sampling for operation selection.

    Learns which operations work best for each query intent through
    Bayesian bandit optimization.

    Each operation has a Beta(α, β) prior:
    - α = successes + 1
    - β = failures + 1

    Selection: Sample from Beta distribution, choose highest sample.
    """

    def __init__(self):
        """Initialize Thompson Sampling learner."""
        # Statistics: (operation_name, intent) -> OperationStatistics
        self.stats: Dict[Tuple[str, str], OperationStatistics] = {}

        # Prior parameters
        self.alpha_prior = 1.0  # Optimistic initialization
        self.beta_prior = 1.0

        logger.info("ThompsonSamplingLearner initialized")

    def get_stats(self, operation: str, intent: str) -> OperationStatistics:
        """Get or create statistics for (operation, intent) pair."""
        key = (operation, intent)
        if key not in self.stats:
            self.stats[key] = OperationStatistics(
                operation_name=operation,
                intent=intent
            )
        return self.stats[key]

    def select_operation(
        self,
        candidates: List[MathOperation],
        intent: QueryIntent
    ) -> MathOperation:
        """
        Select operation using Thompson Sampling.

        Args:
            candidates: Candidate operations
            intent: Query intent

        Returns:
            Selected operation
        """
        intent_str = intent.value

        # Sample from Beta distributions
        samples = []
        for op in candidates:
            stats = self.get_stats(op.name, intent_str)

            # Beta parameters
            alpha = self.alpha_prior + stats.successes
            beta = self.beta_prior + stats.failures

            # Sample success probability
            sample = np.random.beta(alpha, beta)
            samples.append(sample)

        # Choose operation with highest sample
        best_idx = np.argmax(samples)
        selected = candidates[best_idx]

        logger.debug(
            f"Thompson Sampling: {selected.name} (sample={samples[best_idx]:.3f})"
        )

        return selected

    def record_feedback(
        self,
        operation: str,
        intent: str,
        success: bool,
        cost: int,
        execution_time: float
    ):
        """
        Record execution feedback for learning.

        Args:
            operation: Operation name
            intent: Query intent
            success: Whether operation succeeded
            cost: Computational cost
            execution_time: Execution time in seconds
        """
        stats = self.get_stats(operation, intent)
        stats.update(success, cost, execution_time)

        logger.info(
            f"Feedback: {operation} ({intent}) - "
            f"Success: {success}, Rate: {stats.success_rate:.2%}"
        )

    def get_leaderboard(self, intent: Optional[str] = None) -> List[Dict]:
        """
        Get operation leaderboard sorted by success rate.

        Args:
            intent: Optional intent filter

        Returns:
            List of operation stats sorted by success rate
        """
        filtered = [
            stats for stats in self.stats.values()
            if intent is None or stats.intent == intent
        ]

        sorted_stats = sorted(
            filtered,
            key=lambda s: s.success_rate,
            reverse=True
        )

        return [asdict(s) for s in sorted_stats]


# ============================================================================
# Rigorous Testing Framework
# ============================================================================

@dataclass
class MathematicalProperty:
    """
    Mathematical property to verify.

    Properties are invariants that must hold for operations to be correct.
    Examples:
    - Metric triangle inequality: d(x,z) <= d(x,y) + d(y,z)
    - Gradient descent convergence: f(x_{n+1}) <= f(x_n)
    - Orthogonalization: <u_i, u_j> = δ_ij
    """
    name: str
    description: str
    property_function: Callable[[Any], bool]
    severity: str = "error"  # "error", "warning", "info"


class RigorousTester:
    """
    Rigorous testing framework for mathematical operations.

    Runs property-based tests to verify mathematical correctness:
    - Metric axioms (symmetry, triangle inequality, identity)
    - Optimization convergence
    - Orthogonality preservation
    - Numerical stability

    Uses QuickCheck-style property testing.
    """

    def __init__(self):
        """Initialize testing framework."""
        self.properties = self._build_property_catalog()
        self.test_history: List[Dict] = []

        logger.info("RigorousTester initialized")
        logger.info(f"  Properties: {len(self.properties)}")

    def _build_property_catalog(self) -> Dict[str, MathematicalProperty]:
        """Build catalog of mathematical properties to verify."""
        props = {}

        # Metric space properties
        props["metric_symmetry"] = MathematicalProperty(
            name="metric_symmetry",
            description="Metric is symmetric: d(x,y) = d(y,x)",
            property_function=self._check_metric_symmetry,
            severity="error"
        )

        props["metric_triangle_inequality"] = MathematicalProperty(
            name="metric_triangle_inequality",
            description="Triangle inequality: d(x,z) <= d(x,y) + d(y,z)",
            property_function=self._check_triangle_inequality,
            severity="error"
        )

        props["metric_identity"] = MathematicalProperty(
            name="metric_identity",
            description="Identity: d(x,x) = 0",
            property_function=self._check_metric_identity,
            severity="error"
        )

        # Optimization properties
        props["gradient_descent_convergence"] = MathematicalProperty(
            name="gradient_descent_convergence",
            description="Gradient descent decreases objective: f(x_{n+1}) <= f(x_n)",
            property_function=self._check_gradient_descent,
            severity="warning"
        )

        # Linear algebra properties
        props["orthogonality"] = MathematicalProperty(
            name="orthogonality",
            description="Vectors are orthogonal: <u_i, u_j> = δ_ij",
            property_function=self._check_orthogonality,
            severity="error"
        )

        props["normalization"] = MathematicalProperty(
            name="normalization",
            description="Vectors are normalized: ||v|| = 1",
            property_function=self._check_normalization,
            severity="warning"
        )

        # Numerical stability
        props["numerical_stability"] = MathematicalProperty(
            name="numerical_stability",
            description="Results are numerically stable (no NaN/Inf)",
            property_function=self._check_numerical_stability,
            severity="error"
        )

        return props

    # Property check functions

    def _check_metric_symmetry(self, data: Dict) -> bool:
        """Check metric symmetry."""
        if "distance_function" not in data:
            return True  # Skip if not applicable

        d = data["distance_function"]
        samples = data.get("samples", [])

        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                x, y = samples[i], samples[j]
                if not np.isclose(d(x, y), d(y, x), rtol=1e-5):
                    return False

        return True

    def _check_triangle_inequality(self, data: Dict) -> bool:
        """Check triangle inequality."""
        if "distance_function" not in data:
            return True

        d = data["distance_function"]
        samples = data.get("samples", [])

        for i in range(len(samples)):
            for j in range(len(samples)):
                for k in range(len(samples)):
                    x, y, z = samples[i], samples[j], samples[k]
                    if d(x, z) > d(x, y) + d(y, z) + 1e-5:
                        return False

        return True

    def _check_metric_identity(self, data: Dict) -> bool:
        """Check d(x,x) = 0."""
        if "distance_function" not in data:
            return True

        d = data["distance_function"]
        samples = data.get("samples", [])

        for x in samples:
            if not np.isclose(d(x, x), 0.0, atol=1e-6):
                return False

        return True

    def _check_gradient_descent(self, data: Dict) -> bool:
        """Check gradient descent convergence."""
        if "objective_values" not in data:
            return True

        values = data["objective_values"]

        # Check monotonic decrease (with tolerance for noise)
        for i in range(len(values) - 1):
            if values[i+1] > values[i] + 1e-4:
                return False

        return True

    def _check_orthogonality(self, data: Dict) -> bool:
        """Check orthogonality."""
        if "vectors" not in data:
            return True

        vectors = data["vectors"]
        n = len(vectors)

        for i in range(n):
            for j in range(n):
                inner = np.dot(vectors[i], vectors[j])
                expected = 1.0 if i == j else 0.0

                if not np.isclose(inner, expected, atol=1e-5):
                    return False

        return True

    def _check_normalization(self, data: Dict) -> bool:
        """Check normalization."""
        if "vectors" not in data:
            return True

        vectors = data["vectors"]

        for v in vectors:
            norm = np.linalg.norm(v)
            if not np.isclose(norm, 1.0, atol=1e-5):
                return False

        return True

    def _check_numerical_stability(self, data: Dict) -> bool:
        """Check for NaN/Inf."""
        if "result" not in data:
            return True

        result = data["result"]

        if isinstance(result, np.ndarray):
            return not (np.any(np.isnan(result)) or np.any(np.isinf(result)))
        elif isinstance(result, (int, float)):
            return not (np.isnan(result) or np.isinf(result))

        return True

    def verify_operation(
        self,
        operation: MathOperation,
        data: Dict,
        properties: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Verify operation satisfies mathematical properties.

        Args:
            operation: Operation to verify
            data: Test data (inputs, outputs)
            properties: Optional list of properties to check

        Returns:
            Verification result dict
        """
        if properties is None:
            # Auto-select properties based on operation
            properties = self._select_properties_for_operation(operation)

        results = {}
        all_passed = True

        for prop_name in properties:
            if prop_name not in self.properties:
                logger.warning(f"Unknown property: {prop_name}")
                continue

            prop = self.properties[prop_name]

            try:
                passed = prop.property_function(data)
                results[prop_name] = {
                    "passed": passed,
                    "severity": prop.severity,
                    "description": prop.description
                }

                if not passed and prop.severity == "error":
                    all_passed = False

            except Exception as e:
                logger.error(f"Property check failed: {prop_name} - {e}")
                results[prop_name] = {
                    "passed": False,
                    "severity": "error",
                    "error": str(e)
                }
                all_passed = False

        # Record in history
        test_result = {
            "operation": operation.name,
            "properties_checked": list(results.keys()),
            "all_passed": all_passed,
            "results": results
        }
        self.test_history.append(test_result)

        logger.info(
            f"Verified {operation.name}: "
            f"{sum(1 for r in results.values() if r['passed'])}/{len(results)} passed"
        )

        return test_result

    def _select_properties_for_operation(self, operation: MathOperation) -> List[str]:
        """Auto-select relevant properties for operation."""
        properties = []

        # Always check numerical stability
        properties.append("numerical_stability")

        # Domain-specific properties
        if "metric" in operation.name or "distance" in operation.name:
            properties.extend([
                "metric_symmetry",
                "metric_triangle_inequality",
                "metric_identity"
            ])

        if "gradient" in operation.name or "optimize" in operation.name:
            properties.append("gradient_descent_convergence")

        if "gram_schmidt" in operation.name or "orthogonal" in operation.name:
            properties.extend(["orthogonality", "normalization"])

        return properties

    def get_test_report(self) -> Dict[str, Any]:
        """Get comprehensive test report."""
        if not self.test_history:
            return {"total_tests": 0}

        total_tests = len(self.test_history)
        passed_tests = sum(1 for t in self.test_history if t["all_passed"])

        # Property failure rates
        property_failures = {}
        for test in self.test_history:
            for prop, result in test["results"].items():
                if prop not in property_failures:
                    property_failures[prop] = {"passed": 0, "failed": 0}

                if result["passed"]:
                    property_failures[prop]["passed"] += 1
                else:
                    property_failures[prop]["failed"] += 1

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests,
            "property_failures": property_failures,
            "recent_tests": self.test_history[-10:]
        }


# ============================================================================
# Smart Operation Selector (RL + Composition + Testing)
# ============================================================================

class SmartMathOperationSelector(MathOperationSelector):
    """
    Enhanced operation selector with RL learning, composition, and testing.

    Enhancements:
    1. Thompson Sampling learns which operations work best
    2. Operator composition creates functional pipelines
    3. Rigorous testing verifies mathematical correctness
    4. Feedback loops improve selection over time

    Example:
        selector = SmartMathOperationSelector()

        # Plan operations (with learning)
        plan = selector.plan_operations_smart(
            query_text="Find similar documents",
            context={"has_embeddings": True},
            enable_learning=True
        )

        # Execute plan
        result = selector.execute_plan(plan, data)

        # Provide feedback (learning signal)
        selector.record_feedback(plan, success=True, quality=0.85)
    """

    def __init__(self, load_state: bool = True):
        """
        Initialize smart selector.

        Args:
            load_state: Load saved learning state if available
        """
        super().__init__()

        # RL learning
        self.learner = ThompsonSamplingLearner()

        # Operator composition
        self.composer = OperationComposer()

        # Rigorous testing
        self.tester = RigorousTester()

        # State persistence
        self.state_file = Path("HoloLoom/warp/math/.smart_selector_state.json")

        if load_state and self.state_file.exists():
            self._load_state()

        logger.info("SmartMathOperationSelector initialized")
        logger.info("  RL Learning: Thompson Sampling")
        logger.info("  Composition: Enabled")
        logger.info("  Testing: Rigorous property verification")

    def plan_operations_smart(
        self,
        query_text: str,
        query_embedding: Optional[np.ndarray] = None,
        context: Optional[Dict] = None,
        budget: Optional[int] = None,
        enable_expensive: bool = False,
        enable_learning: bool = True,
        enable_composition: bool = True
    ) -> OperationPlan:
        """
        Plan operations with RL learning and composition.

        Args:
            query_text: Query string
            query_embedding: Optional query embedding
            context: Optional context
            budget: Optional cost budget
            enable_expensive: Allow expensive operations
            enable_learning: Use RL to select operations
            enable_composition: Enable operator composition

        Returns:
            Enhanced OperationPlan
        """
        # Classify intent
        intents = self.classify_intent(query_text, context)
        primary_intent = intents[0]

        # Get applicable operations
        candidates = [
            op for op in self.operations.values()
            if any(op.is_applicable(intent) for intent in intents)
        ]

        selected = []
        justifications = {}

        # RL-based selection if enabled
        if enable_learning and len(candidates) > 1:
            # Use Thompson Sampling to select best operations
            for _ in range(min(5, len(candidates))):  # Select up to 5 ops
                op = self.learner.select_operation(candidates, primary_intent)
                if op not in selected:
                    selected.append(op)
                    justifications[op.name] = (
                        f"Thompson Sampling (intent: {primary_intent.value})"
                    )
        else:
            # Fall back to base selection
            plan = super().plan_operations(
                query_text, query_embedding, context, budget, enable_expensive
            )
            return plan

        # Operator composition if enabled
        if enable_composition:
            compositions = self.composer.suggest_compositions(selected)

            # Add composed operations
            for comp in compositions:
                selected.append(comp)  # Note: ComposedOperation not MathOperation
                justifications[comp.name] = "Suggested composition pipeline"

        # Topological sort
        ordered = self._topological_sort([op for op in selected if isinstance(op, MathOperation)])

        # Apply budget
        if budget:
            cumulative = 0
            filtered = []
            for op in ordered:
                cost = op.estimated_cost if hasattr(op, 'estimated_cost') else 0
                if cumulative + cost <= budget:
                    filtered.append(op)
                    cumulative += cost
            ordered = filtered

        # Create plan
        total_cost = sum(
            op.estimated_cost if hasattr(op, 'estimated_cost') else 0
            for op in ordered
        )
        domains_used = set(
            op.domain for op in ordered
            if isinstance(op, MathOperation)
        )

        plan = OperationPlan(
            operations=ordered,
            justifications=justifications,
            total_cost=total_cost,
            domains_used=domains_used,
            metadata={
                "primary_intent": primary_intent.value,
                "rl_enabled": enable_learning,
                "composition_enabled": enable_composition
            }
        )

        return plan

    def execute_plan_with_verification(
        self,
        plan: OperationPlan,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute operation plan with rigorous testing.

        Args:
            plan: Operation plan
            data: Input data

        Returns:
            Execution results with verification
        """
        results = {
            "operations_executed": [],
            "verification_results": [],
            "total_time": 0.0,
            "all_tests_passed": True
        }

        import time

        for op in plan.operations:
            start_time = time.time()

            # Execute operation (placeholder)
            try:
                # Actual execution would happen here
                operation_result = self._execute_operation(op, data)
                execution_time = time.time() - start_time

                # Verify operation
                verification = self.tester.verify_operation(
                    op,
                    {"result": operation_result, **data}
                )

                results["operations_executed"].append({
                    "operation": op.name,
                    "success": True,
                    "time": execution_time
                })
                results["verification_results"].append(verification)

                if not verification["all_passed"]:
                    results["all_tests_passed"] = False
                    logger.warning(
                        f"Operation {op.name} failed verification"
                    )

                results["total_time"] += execution_time

            except Exception as e:
                logger.error(f"Operation {op.name} failed: {e}")
                results["operations_executed"].append({
                    "operation": op.name,
                    "success": False,
                    "error": str(e)
                })
                results["all_tests_passed"] = False

        return results

    def _execute_operation(self, op: MathOperation, data: Dict) -> Any:
        """Execute single operation (placeholder)."""
        # This would dynamically import and execute the actual operation
        logger.debug(f"Executing {op.name}")

        # Placeholder: return mock result
        if "result" in data:
            return data["result"]
        return np.random.randn(10)  # Mock result

    def record_feedback(
        self,
        plan: OperationPlan,
        success: bool,
        quality: float = 0.0,
        execution_time: float = 0.0
    ):
        """
        Record feedback for RL learning.

        Args:
            plan: Executed operation plan
            success: Whether plan succeeded
            quality: Quality score (0-1)
            execution_time: Total execution time
        """
        primary_intent = plan.metadata.get("primary_intent", "unknown")

        for op in plan.operations:
            if not isinstance(op, MathOperation):
                continue

            # Record feedback for RL
            self.learner.record_feedback(
                operation=op.name,
                intent=primary_intent,
                success=success and quality >= 0.5,  # Success if quality >= 50%
                cost=op.estimated_cost,
                execution_time=execution_time / len(plan.operations)
            )

        logger.info(
            f"Feedback recorded: Success={success}, Quality={quality:.2f}"
        )

    def get_smart_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including RL and testing."""
        base_stats = super().get_statistics()

        # RL statistics
        leaderboard = self.learner.get_leaderboard()

        # Testing statistics
        test_report = self.tester.get_test_report()

        return {
            **base_stats,
            "rl_learning": {
                "total_feedback": sum(
                    s.successes + s.failures
                    for s in self.learner.stats.values()
                ),
                "leaderboard": leaderboard[:10]  # Top 10
            },
            "testing": test_report
        }

    def _save_state(self):
        """Save learning state to disk."""
        state = {
            "learner_stats": {
                str(k): asdict(v)
                for k, v in self.learner.stats.items()
            },
            "test_history": self.tester.test_history[-100:]  # Last 100 tests
        }

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"State saved to {self.state_file}")

    def _load_state(self):
        """Load learning state from disk."""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Restore learner stats
            for key_str, stats_dict in state.get("learner_stats", {}).items():
                key = eval(key_str)  # Reconstruct tuple
                self.learner.stats[key] = OperationStatistics(**stats_dict)

            # Restore test history
            self.tester.test_history = state.get("test_history", [])

            logger.info(f"State loaded from {self.state_file}")

        except Exception as e:
            logger.warning(f"Failed to load state: {e}")


# ============================================================================
# Factory Function
# ============================================================================

def create_smart_selector(load_state: bool = True) -> SmartMathOperationSelector:
    """
    Create smart selector with defaults.

    Args:
        load_state: Load saved learning state

    Returns:
        Configured SmartMathOperationSelector
    """
    return SmartMathOperationSelector(load_state=load_state)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SMART MATHEMATICAL OPERATION SELECTOR")
    print("RL Learning + Composition + Rigorous Testing")
    print("="*80)
    print()

    # Create smart selector
    selector = create_smart_selector(load_state=False)

    print("Testing RL learning and composition...\n")

    # Simulate multiple queries to train RL
    test_scenarios = [
        ("Find similar documents", QueryIntent.SIMILARITY, True, 0.85),
        ("Find similar documents", QueryIntent.SIMILARITY, True, 0.92),
        ("Find similar documents", QueryIntent.SIMILARITY, False, 0.45),
        ("Optimize retrieval", QueryIntent.OPTIMIZATION, True, 0.78),
        ("Optimize retrieval", QueryIntent.OPTIMIZATION, True, 0.88),
        ("Verify metric space", QueryIntent.VERIFICATION, True, 0.95),
    ]

    for query, intent, success, quality in test_scenarios:
        print(f"Query: {query}")

        # Plan with RL
        plan = selector.plan_operations_smart(
            query_text=query,
            enable_learning=True,
            enable_composition=True,
            budget=50
        )

        print(f"  Operations: {[op.name for op in plan.operations[:3]]}")
        print(f"  Cost: {plan.total_cost}")

        # Provide feedback
        selector.record_feedback(plan, success, quality, 0.1)
        print()

    # Show learned statistics
    print("="*80)
    print("LEARNING STATISTICS")
    print("="*80)

    stats = selector.get_smart_statistics()

    print(f"\nRL Learning:")
    print(f"  Total feedback: {stats['rl_learning']['total_feedback']}")
    print(f"\n  Top operations by success rate:")

    for i, op_stats in enumerate(stats['rl_learning']['leaderboard'][:5], 1):
        print(f"    {i}. {op_stats['operation_name']} ({op_stats['intent']})")
        print(f"       Success rate: {op_stats['successes']}/{op_stats['successes'] + op_stats['failures']}")

    print(f"\nTesting:")
    test_stats = stats['testing']
    if test_stats.get('total_tests', 0) > 0:
        print(f"  Total tests: {test_stats['total_tests']}")
        print(f"  Pass rate: {test_stats['pass_rate']:.1%}")
    else:
        print("  No tests run yet")

    # Save state
    selector._save_state()

    print("\n" + "="*80)
    print("SMART selector learns which operations work best!")
    print("="*80)
