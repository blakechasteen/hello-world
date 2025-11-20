#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Pipeline Integration - Bridge to WeavingOrchestrator
==========================================================
Integrates the Smart Math Pipeline into the canonical WeavingOrchestrator.

This module provides a clean integration layer that:
1. Wires math pipeline into the weaving cycle (after ResonanceShed)
2. Supports incremental activation (Phase 1 → Phase 4)
3. Maintains graceful degradation (no crashes if disabled)
4. Follows "Reliable Systems: Safety First" philosophy

Architecture:
  WeavingOrchestrator
    → ResonanceShed (extract features)
    → MathPipelineIntegration (analyze features, select operations)
    → WarpSpace (enrich with math insights)
    → ConvergenceEngine (math-enriched decision)

Phase 1 (current): Basic operations (inner_product, metric_distance, norm)
Phase 2: RL learning (Thompson Sampling)
Phase 3: Composition + testing
Phase 4: Advanced operations (eigenvalues, Ricci flow, etc.)

Author: HoloLoom Team
Date: 2025-10-29
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Import math pipeline components
try:
    from HoloLoom.warp.math.operation_selector import (
        MathOperationSelector,
        QueryIntent,
        OperationPlan
    )
    from HoloLoom.warp.math.smart_operation_selector import (
        SmartMathOperationSelector
    )
    from HoloLoom.warp.math.meaning_synthesizer import (
        MeaningSynthesizer,
        MeaningResult
    )
    HAS_MATH_PIPELINE = True
except ImportError as e:
    logging.warning(f"Math pipeline not available: {e}")
    HAS_MATH_PIPELINE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Math Pipeline Result
# ============================================================================

@dataclass
class MathPipelineResult:
    """
    Result from math pipeline execution.

    Contains mathematical analysis, insights, and natural language explanations
    that enrich the weaving process.

    Attributes:
        summary: High-level summary (1-2 sentences)
        insights: List of key mathematical insights
        operations_used: Operations that were executed
        total_cost: Computational cost
        confidence: Confidence in results (0-1)
        meaning: Full MeaningResult with details
        execution_time_ms: Pipeline execution time
    """
    summary: str
    insights: List[str]
    operations_used: List[str]
    total_cost: int
    confidence: float
    meaning: Optional[Any] = None  # MeaningResult (avoid circular import)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Math Pipeline Integration
# ============================================================================

class MathPipelineIntegration:
    """
    Integration layer between WeavingOrchestrator and Smart Math Pipeline.

    This class manages the math pipeline lifecycle and provides a clean
    interface for the orchestrator to request mathematical analysis.

    Configuration Modes:
    - DISABLED: No math pipeline (default)
    - BASIC: Basic operations only (Phase 1)
    - RL: Basic + RL learning (Phase 2)
    - COMPOSED: RL + composition + testing (Phase 3)
    - ADVANCED: Full pipeline with expensive operations (Phase 4)

    Example:
        # Create integration (Phase 1: basic operations)
        integration = MathPipelineIntegration(
            enabled=True,
            budget=10,
            enable_rl=False,
            enable_composition=False
        )

        # Analyze query
        result = integration.analyze(
            query_text="Find similar documents",
            query_embedding=emb,
            context={"has_embeddings": True}
        )

        # Use insights in orchestrator
        print(result.summary)
        # "Found 3 similar items using 2 mathematical operations."
    """

    def __init__(
        self,
        enabled: bool = False,
        budget: int = 50,
        enable_expensive: bool = False,
        enable_rl: bool = False,
        enable_composition: bool = False,
        enable_testing: bool = False,
        output_style: str = "detailed",
        use_contextual_bandit: bool = False
    ):
        """
        Initialize math pipeline integration.

        Args:
            enabled: Enable math pipeline
            budget: Computational budget (cost limit)
            enable_expensive: Allow expensive operations (Ricci flow, etc.)
            enable_rl: Enable RL learning (Thompson Sampling)
            enable_composition: Enable operation composition
            enable_testing: Enable rigorous property-based testing
            output_style: "concise", "detailed", or "technical"
            use_contextual_bandit: Use 470-dim contextual bandit (advanced)
        """
        self.enabled = enabled and HAS_MATH_PIPELINE
        self.budget = budget
        self.enable_expensive = enable_expensive
        self.enable_rl = enable_rl
        self.enable_composition = enable_composition
        self.enable_testing = enable_testing
        self.output_style = output_style
        self.use_contextual_bandit = use_contextual_bandit

        # Initialize components if enabled
        if self.enabled:
            # Operation selector (smart if RL enabled, basic otherwise)
            if enable_rl:
                self.selector = SmartMathOperationSelector(
                    load_state=True,
                    use_contextual=use_contextual_bandit
                )
                logger.info("Math Pipeline: SMART mode (RL enabled)")
            else:
                self.selector = MathOperationSelector()
                logger.info("Math Pipeline: BASIC mode (no RL)")

            # Meaning synthesizer
            self.synthesizer = MeaningSynthesizer(
                use_data_understanding=True
            )

            logger.info("MathPipelineIntegration initialized")
            logger.info(f"  Budget: {budget}")
            logger.info(f"  Expensive ops: {enable_expensive}")
            logger.info(f"  Composition: {enable_composition}")
            logger.info(f"  Testing: {enable_testing}")
            logger.info(f"  Output: {output_style}")

        else:
            self.selector = None
            self.synthesizer = None

            if not HAS_MATH_PIPELINE:
                logger.warning("Math pipeline not available (module import failed)")
            else:
                logger.info("Math pipeline disabled")

        # Statistics
        self.stats = {
            "total_analyses": 0,
            "total_operations": 0,
            "total_cost": 0,
            "avg_confidence": 0.0,
            "operations_by_intent": {},
            "execution_times_ms": []
        }

    def analyze(
        self,
        query_text: str,
        query_embedding: Optional[np.ndarray] = None,
        context: Optional[Dict] = None
    ) -> Optional[MathPipelineResult]:
        """
        Analyze query using math pipeline.

        This is the main API called by WeavingOrchestrator after feature
        extraction (ResonanceShed).

        Args:
            query_text: Query string
            query_embedding: Optional query embedding
            context: Optional context (features, metadata)

        Returns:
            MathPipelineResult with analysis, or None if disabled
        """
        if not self.enabled:
            return None

        import time
        start_time = time.time()

        try:
            # Classify intent
            intents = self.selector.classify_intent(query_text, context)
            primary_intent = intents[0] if intents else QueryIntent.ANALYSIS

            logger.info(f"Math Pipeline analyzing: {query_text[:50]}...")
            logger.info(f"  Primary intent: {primary_intent.value}")

            # Plan operations
            if self.enable_rl and isinstance(self.selector, SmartMathOperationSelector):
                # Smart planning with RL
                plan = self.selector.plan_operations_smart(
                    query_text=query_text,
                    query_embedding=query_embedding,
                    context=context,
                    budget=self.budget,
                    enable_expensive=self.enable_expensive,
                    enable_learning=True,
                    enable_composition=self.enable_composition
                )
            else:
                # Basic planning (no RL)
                plan = self.selector.plan_operations(
                    query_text=query_text,
                    query_embedding=query_embedding,
                    context=context,
                    budget=self.budget,
                    enable_expensive=self.enable_expensive
                )

            logger.info(f"  Planned {len(plan.operations)} operations, cost {plan.total_cost}")

            # Execute operations
            if self.enable_testing and isinstance(self.selector, SmartMathOperationSelector):
                # Execute with rigorous testing
                execution_result = self.selector.execute_plan_with_verification(
                    plan=plan,
                    data=context or {}
                )
                logger.info(f"  Tests passed: {execution_result['all_tests_passed']}")
            else:
                # Execute without testing (Phase 1)
                execution_result = self._execute_plan_basic(plan, context or {})

            # Extract mathematical results
            math_results = {}
            for op_exec in execution_result.get("operations_executed", []):
                op_name = op_exec["operation"]
                # For Phase 1, generate mock results
                # In Phase 2+, this would be actual results from warp/math/ modules
                math_results[op_name] = self._generate_mock_results(op_name)

            # Synthesize meaning
            meaning = self.synthesizer.synthesize(
                results=math_results,
                intent=primary_intent,
                plan=plan,
                context=context
            )

            # Record feedback for RL (if enabled)
            if self.enable_rl and isinstance(self.selector, SmartMathOperationSelector):
                success = execution_result.get("all_tests_passed", True)
                self.selector.record_feedback(
                    plan=plan,
                    success=success,
                    quality=meaning.confidence,
                    execution_time=execution_result.get("total_time", 0.0),
                    query_text=query_text
                )

            # Build result
            execution_time_ms = (time.time() - start_time) * 1000

            result = MathPipelineResult(
                summary=meaning.summary,
                insights=meaning.key_insights,
                operations_used=[op.name for op in plan.operations],
                total_cost=plan.total_cost,
                confidence=meaning.confidence,
                meaning=meaning,
                execution_time_ms=execution_time_ms,
                metadata={
                    "intent": primary_intent.value,
                    "plan": plan,
                    "execution_result": execution_result
                }
            )

            # Update statistics
            self._update_stats(result, primary_intent)

            logger.info(f"  Math analysis complete in {execution_time_ms:.1f}ms")
            logger.info(f"  Confidence: {meaning.confidence:.2f}")

            return result

        except Exception as e:
            logger.error(f"Math pipeline error: {e}", exc_info=True)
            # Graceful degradation: return None instead of crashing
            return None

    def _execute_plan_basic(
        self,
        plan: OperationPlan,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute operation plan without rigorous testing (Phase 1).

        This is a simplified executor for Phase 1. In later phases, this
        would be replaced by SmartMathOperationSelector.execute_plan_with_verification.

        Args:
            plan: Operation plan
            data: Input data

        Returns:
            Execution results
        """
        results = {
            "operations_executed": [],
            "total_time": 0.0,
            "all_tests_passed": True  # No tests in Phase 1
        }

        import time

        for op in plan.operations:
            start_time = time.time()

            try:
                # Placeholder execution (Phase 1)
                # In Phase 2+, this would dynamically import from warp/math/
                execution_time = time.time() - start_time

                results["operations_executed"].append({
                    "operation": op.name,
                    "success": True,
                    "time": execution_time
                })

                results["total_time"] += execution_time

            except Exception as e:
                logger.error(f"Operation {op.name} failed: {e}")
                results["operations_executed"].append({
                    "operation": op.name,
                    "success": False,
                    "error": str(e)
                })

        return results

    def _generate_mock_results(self, operation: str) -> Dict[str, Any]:
        """
        Generate mock results for Phase 1 demo.

        In Phase 2+, this would be replaced by actual execution from warp/math/.

        Args:
            operation: Operation name

        Returns:
            Mock results dict
        """
        # Mock results that match meaning_synthesizer templates
        if operation == "inner_product":
            return {
                "similarities": [0.85, 0.72, 0.68, 0.55, 0.42],
                "success": True
            }
        elif operation == "metric_distance":
            return {
                "min_distance": 0.15,
                "max_distance": 1.85,
                "distances": [0.15, 0.28, 0.32, 0.45, 0.58],
                "success": True
            }
        elif operation == "norm":
            return {
                "norm": 1.0,
                "vector_magnitude": 1.0,
                "success": True
            }
        elif operation == "gradient":
            return {
                "gradient_norm": 0.045,
                "direction": [0.1, -0.2, 0.3],
                "success": True
            }
        elif operation == "eigenvalues":
            return {
                "max_eigenvalue": 0.85,
                "eigenvalues": [0.85, 0.42, 0.18],
                "success": True
            }
        else:
            return {"success": True}

    def _update_stats(self, result: MathPipelineResult, intent: QueryIntent):
        """Update internal statistics."""
        self.stats["total_analyses"] += 1
        self.stats["total_operations"] += len(result.operations_used)
        self.stats["total_cost"] += result.total_cost
        self.stats["execution_times_ms"].append(result.execution_time_ms)

        # Running average confidence
        n = self.stats["total_analyses"]
        prev_avg = self.stats["avg_confidence"]
        self.stats["avg_confidence"] = (
            (prev_avg * (n - 1) + result.confidence) / n
        )

        # Operations by intent
        intent_key = intent.value
        if intent_key not in self.stats["operations_by_intent"]:
            self.stats["operations_by_intent"][intent_key] = {
                "count": 0,
                "operations": {}
            }

        self.stats["operations_by_intent"][intent_key]["count"] += 1

        for op in result.operations_used:
            ops_dict = self.stats["operations_by_intent"][intent_key]["operations"]
            ops_dict[op] = ops_dict.get(op, 0) + 1

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get math pipeline statistics.

        Returns:
            Statistics dict
        """
        stats = dict(self.stats)

        # Add averages
        if self.stats["total_analyses"] > 0:
            stats["avg_operations_per_analysis"] = (
                self.stats["total_operations"] / self.stats["total_analyses"]
            )
            stats["avg_cost_per_analysis"] = (
                self.stats["total_cost"] / self.stats["total_analyses"]
            )

            if self.stats["execution_times_ms"]:
                times = self.stats["execution_times_ms"]
                stats["avg_execution_time_ms"] = np.mean(times)
                stats["p50_execution_time_ms"] = np.percentile(times, 50)
                stats["p95_execution_time_ms"] = np.percentile(times, 95)

        # Add selector statistics if available
        if self.enabled and self.selector:
            stats["selector_stats"] = self.selector.get_statistics()

            # RL statistics if smart selector
            if isinstance(self.selector, SmartMathOperationSelector):
                stats["rl_stats"] = self.selector.get_smart_statistics()

        return stats

    def save_state(self):
        """Save math pipeline state (RL statistics, etc.)."""
        if self.enabled and self.enable_rl:
            if isinstance(self.selector, SmartMathOperationSelector):
                self.selector._save_state()
                logger.info("Math pipeline state saved")

    def __repr__(self) -> str:
        """String representation."""
        if not self.enabled:
            return "MathPipelineIntegration(disabled)"

        mode = "SMART" if self.enable_rl else "BASIC"
        return (
            f"MathPipelineIntegration({mode}, "
            f"budget={self.budget}, "
            f"ops={self.stats['total_operations']}, "
            f"analyses={self.stats['total_analyses']})"
        )


# ============================================================================
# Factory Functions
# ============================================================================

def create_math_integration_lite() -> MathPipelineIntegration:
    """
    Create math integration in LITE mode (Phase 1).

    - Budget: 10 (basic operations only)
    - No RL learning
    - No composition
    - No testing

    Returns:
        MathPipelineIntegration configured for LITE mode
    """
    return MathPipelineIntegration(
        enabled=True,
        budget=10,
        enable_expensive=False,
        enable_rl=False,
        enable_composition=False,
        enable_testing=False,
        output_style="concise"
    )


def create_math_integration_fast() -> MathPipelineIntegration:
    """
    Create math integration in FAST mode (Phase 2).

    - Budget: 50 (basic + moderate operations)
    - RL learning enabled
    - No composition
    - No testing

    Returns:
        MathPipelineIntegration configured for FAST mode
    """
    return MathPipelineIntegration(
        enabled=True,
        budget=50,
        enable_expensive=False,
        enable_rl=True,
        enable_composition=False,
        enable_testing=False,
        output_style="detailed"
    )


def create_math_integration_full() -> MathPipelineIntegration:
    """
    Create math integration in FULL mode (Phase 3).

    - Budget: 100 (basic + moderate + advanced)
    - RL learning enabled
    - Composition enabled
    - Testing enabled

    Returns:
        MathPipelineIntegration configured for FULL mode
    """
    return MathPipelineIntegration(
        enabled=True,
        budget=100,
        enable_expensive=False,
        enable_rl=True,
        enable_composition=True,
        enable_testing=True,
        output_style="detailed"
    )


def create_math_integration_research() -> MathPipelineIntegration:
    """
    Create math integration in RESEARCH mode (Phase 4).

    - Budget: 999 (unlimited)
    - All features enabled
    - Expensive operations enabled (Ricci flow, etc.)
    - Contextual bandit (470-dim FGTS)

    Returns:
        MathPipelineIntegration configured for RESEARCH mode
    """
    return MathPipelineIntegration(
        enabled=True,
        budget=999,
        enable_expensive=True,
        enable_rl=True,
        enable_composition=True,
        enable_testing=True,
        output_style="technical",
        use_contextual_bandit=True
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MATH PIPELINE INTEGRATION - Phase 1 Demo")
    print("="*80)
    print()

    # Create integration (LITE mode - Phase 1)
    integration = create_math_integration_lite()

    print(f"Integration: {integration}")
    print()

    # Test queries
    test_queries = [
        ("Find documents similar to quantum computing", {"has_embeddings": True}),
        ("What is the distance between A and B?", {}),
        ("Optimize the retrieval algorithm", {"requires_optimization": True}),
    ]

    for query, context in test_queries:
        print("="*80)
        print(f"Query: {query}")
        print("="*80)

        # Analyze
        result = integration.analyze(
            query_text=query,
            query_embedding=np.random.randn(384),  # Mock embedding
            context=context
        )

        if result:
            print(f"\nSummary: {result.summary}")
            print(f"\nInsights:")
            for insight in result.insights:
                print(f"  • {insight}")
            print(f"\nOperations: {result.operations_used}")
            print(f"Cost: {result.total_cost}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"Time: {result.execution_time_ms:.1f}ms")
        else:
            print("(Math pipeline disabled or error)")

        print()

    # Show statistics
    print("="*80)
    print("STATISTICS")
    print("="*80)
    stats = integration.get_statistics()
    print(f"Total analyses: {stats['total_analyses']}")
    print(f"Total operations: {stats['total_operations']}")
    print(f"Avg operations/analysis: {stats.get('avg_operations_per_analysis', 0):.1f}")
    print(f"Avg confidence: {stats['avg_confidence']:.2%}")

    if "avg_execution_time_ms" in stats:
        print(f"Avg execution time: {stats['avg_execution_time_ms']:.1f}ms")

    print("\n" + "="*80)
    print("Math pipeline integration ready!")
    print("="*80)