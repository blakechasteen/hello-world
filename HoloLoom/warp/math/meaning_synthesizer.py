#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Meaning Synthesizer
=================================
The final layer that turns mathematical results into natural language meaning.

Complete Pipeline:
  Query (text)
    → Intent Classification
    → Smart Math Selection (RL)
    → Composed Operations
    → Rigorous Testing
    → Mathematical Results (numbers)
    → MEANING SYNTHESIS (numbers → words)
    → Natural Language Response

This is where the fancy math becomes human-readable explanations with
full computational provenance.

Philosophy:
Mathematics provides rigorous computation. Humans need meaningful language.
This synthesizer bridges the gap, turning eigenvalues into "stability analysis",
gradients into "optimization direction", and metrics into "similarity scores".

Author: HoloLoom Team
Date: 2025-10-26
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np

try:
    from .smart_operation_selector import SmartMathOperationSelector, OperationPlan
    from .operation_selector import QueryIntent, MathOperation
except ImportError:
    from smart_operation_selector import SmartMathOperationSelector, OperationPlan
    from operation_selector import QueryIntent, MathOperation

logger = logging.getLogger(__name__)


# ============================================================================
# Meaning Templates
# ============================================================================

@dataclass
class MeaningTemplate:
    """
    Template for converting mathematical results to natural language.

    Attributes:
        operation: Mathematical operation name
        template: Natural language template with placeholders
        formatter: Function to format results into template
    """
    operation: str
    template: str
    formatter: Any  # Callable[[Dict], str]


class MeaningTemplates:
    """
    Catalog of templates for each mathematical operation.

    Maps operations to natural language explanations.
    """

    @staticmethod
    def get_templates() -> Dict[str, MeaningTemplate]:
        """Get all meaning templates."""
        templates = {}

        # Similarity operations
        templates["inner_product"] = MeaningTemplate(
            operation="inner_product",
            template="Computed similarity scores using dot products. The top matches have similarity scores of {top_scores}.",
            formatter=lambda r: {"top_scores": ", ".join([f"{s:.2f}" for s in r.get("similarities", [])[:3]])}
        )

        templates["metric_distance"] = MeaningTemplate(
            operation="metric_distance",
            template="Calculated distances in the semantic space. The closest items are within {min_dist:.2f} units.",
            formatter=lambda r: {"min_dist": r.get("min_distance", 0.0)}
        )

        templates["hyperbolic_distance"] = MeaningTemplate(
            operation="hyperbolic_distance",
            template="Analyzed hierarchical structure using hyperbolic geometry. Found {n_clusters} distinct clusters in the Poincaré ball.",
            formatter=lambda r: {"n_clusters": r.get("clusters", 0)}
        )

        templates["kl_divergence"] = MeaningTemplate(
            operation="kl_divergence",
            template="Compared distributions using KL divergence. The distributions differ by {divergence:.3f} nats of information.",
            formatter=lambda r: {"divergence": r.get("kl_value", 0.0)}
        )

        # Optimization operations
        templates["gradient"] = MeaningTemplate(
            operation="gradient",
            template="Computed optimization direction. The gradient has magnitude {grad_norm:.3f}, pointing toward improvement.",
            formatter=lambda r: {"grad_norm": r.get("gradient_norm", 0.0)}
        )

        templates["gram_schmidt"] = MeaningTemplate(
            operation="gram_schmidt",
            template="Orthogonalized context for maximum diversity. Created {n_vectors} independent information sources.",
            formatter=lambda r: {"n_vectors": r.get("n_orthogonal", 0)}
        )

        templates["thompson_sampling"] = MeaningTemplate(
            operation="thompson_sampling",
            template="Balanced exploration and exploitation using Thompson Sampling. Selected option with {prob:.1%} estimated success probability.",
            formatter=lambda r: {"prob": r.get("sampled_probability", 0.0)}
        )

        # Analysis operations
        templates["eigenvalues"] = MeaningTemplate(
            operation="eigenvalues",
            template="Performed spectral analysis. The spectrum shows {stability} with dominant eigenvalue {max_eig:.3f}.",
            formatter=lambda r: {
                "stability": "stable behavior" if r.get("max_eigenvalue", 0) < 1 else "unstable dynamics",
                "max_eig": r.get("max_eigenvalue", 0.0)
            }
        )

        templates["laplacian"] = MeaningTemplate(
            operation="laplacian",
            template="Analyzed graph topology using the Laplacian. Found {n_components} connected components.",
            formatter=lambda r: {"n_components": r.get("components", 1)}
        )

        templates["fourier_transform"] = MeaningTemplate(
            operation="fourier_transform",
            template="Decomposed signal into frequency components. Dominant frequency at {freq:.2f} Hz accounts for {power:.1%} of energy.",
            formatter=lambda r: {"freq": r.get("dominant_freq", 0.0), "power": r.get("dominant_power", 0.0)}
        )

        templates["convergence_analysis"] = MeaningTemplate(
            operation="convergence_analysis",
            template="Analyzed convergence. The sequence is {converging} toward limit {limit:.3f}.",
            formatter=lambda r: {
                "converging": "converging" if r.get("is_converging", False) else "diverging",
                "limit": r.get("limit", 0.0)
            }
        )

        # Verification operations
        templates["metric_verification"] = MeaningTemplate(
            operation="metric_verification",
            template="Verified metric space axioms. The distance function is {valid} and the space is {complete}.",
            formatter=lambda r: {
                "valid": "valid" if r.get("is_valid_metric", False) else "invalid",
                "complete": "complete" if r.get("is_complete", False) else "incomplete"
            }
        )

        templates["continuity_check"] = MeaningTemplate(
            operation="continuity_check",
            template="Verified continuity. The function is {smooth} with Lipschitz constant {L:.2f}.",
            formatter=lambda r: {
                "smooth": "smooth" if r.get("is_smooth", False) else "not smooth",
                "L": r.get("lipschitz_constant", float('inf'))
            }
        )

        # Geometry operations
        templates["geodesic"] = MeaningTemplate(
            operation="geodesic",
            template="Found shortest path on the manifold. The geodesic has length {length:.3f}.",
            formatter=lambda r: {"length": r.get("geodesic_length", 0.0)}
        )

        templates["ricci_flow"] = MeaningTemplate(
            operation="ricci_flow",
            template="Applied Ricci flow for {iterations} iterations. Curvature smoothed from {initial:.3f} to {final:.3f}.",
            formatter=lambda r: {
                "iterations": r.get("iterations", 0),
                "initial": r.get("initial_curvature", 0.0),
                "final": r.get("final_curvature", 0.0)
            }
        )

        # Composition templates
        templates["similarity_pipeline"] = MeaningTemplate(
            operation="similarity_pipeline",
            template="Performed verified similarity search. Found {n_results} results with valid metric distances.",
            formatter=lambda r: {"n_results": r.get("n_results", 0)}
        )

        templates["verified_optimization"] = MeaningTemplate(
            operation="verified_optimization",
            template="Optimized with continuity guarantees. Improved objective by {improvement:.2%} with bounded sensitivity.",
            formatter=lambda r: {"improvement": r.get("improvement", 0.0)}
        )

        templates["spectral_pipeline"] = MeaningTemplate(
            operation="spectral_pipeline",
            template="Completed spectral analysis. Graph has {n_components} components with spectral gap {gap:.3f}.",
            formatter=lambda r: {"n_components": r.get("n_components", 0), "gap": r.get("spectral_gap", 0.0)}
        )

        return templates


# ============================================================================
# Meaning Synthesizer
# ============================================================================

@dataclass
class MeaningResult:
    """
    Natural language meaning extracted from mathematical results.

    Attributes:
        summary: High-level summary (1-2 sentences)
        details: Detailed explanations per operation
        key_insights: List of key insights
        recommendations: Optional recommendations
        confidence: Confidence in the meaning (0-1)
        provenance: Complete computational trace
    """
    summary: str
    details: List[str]
    key_insights: List[str]
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_text(self, style: str = "concise") -> str:
        """
        Convert to natural language text.

        Args:
            style: "concise", "detailed", or "technical"

        Returns:
            Natural language text
        """
        if style == "concise":
            return self.summary

        elif style == "detailed":
            parts = [self.summary, ""]

            if self.details:
                parts.append("Analysis:")
                for detail in self.details:
                    parts.append(f"  - {detail}")
                parts.append("")

            if self.key_insights:
                parts.append("Key Insights:")
                for insight in self.key_insights:
                    parts.append(f"  • {insight}")
                parts.append("")

            if self.recommendations:
                parts.append("Recommendations:")
                for rec in self.recommendations:
                    parts.append(f"  -> {rec}")

            return "\n".join(parts)

        elif style == "technical":
            parts = [self.summary, ""]

            parts.append("Mathematical Details:")
            for detail in self.details:
                parts.append(f"  {detail}")
            parts.append("")

            parts.append("Provenance:")
            for key, value in self.provenance.items():
                parts.append(f"  {key}: {value}")

            return "\n".join(parts)

        return self.summary


class MeaningSynthesizer:
    """
    Synthesizes natural language meaning from mathematical results.

    This is the final layer that turns numbers back into words, completing
    the full pipeline:

      Query (words)
        -> Math Selection (RL)
        -> Operations (numbers)
        -> Meaning Synthesis (numbers -> words)
        -> Response (words)

    Example:
        synthesizer = MeaningSynthesizer()

        # Mathematical results from operations
        results = {
            "inner_product": {"similarities": [0.85, 0.72, 0.68]},
            "metric_distance": {"min_distance": 0.15},
            "gradient": {"gradient_norm": 0.045}
        }

        # Synthesize meaning
        meaning = synthesizer.synthesize(
            results=results,
            intent=QueryIntent.SIMILARITY,
            plan=operation_plan
        )

        # Output: "Found 3 similar items with scores 0.85, 0.72, 0.68.
        #          The closest match is within 0.15 units..."
    """

    def __init__(self):
        """Initialize meaning synthesizer."""
        self.templates = MeaningTemplates.get_templates()
        self.synthesis_history: List[MeaningResult] = []

        logger.info("MeaningSynthesizer initialized")
        logger.info(f"  Templates: {len(self.templates)}")

    def synthesize(
        self,
        results: Dict[str, Any],
        intent: QueryIntent,
        plan: OperationPlan,
        context: Optional[Dict] = None
    ) -> MeaningResult:
        """
        Synthesize natural language meaning from mathematical results.

        Args:
            results: Dict mapping operation name to results
            intent: Query intent
            plan: Operation plan that was executed
            context: Optional additional context

        Returns:
            MeaningResult with natural language explanation
        """
        logger.info(f"Synthesizing meaning for {len(results)} operations")

        # Generate detail explanations
        details = []
        for op in plan.operations:
            op_name = op.name
            if op_name in results and op_name in self.templates:
                template = self.templates[op_name]
                try:
                    formatted = template.formatter(results[op_name])
                    explanation = template.template.format(**formatted)
                    details.append(explanation)
                except Exception as e:
                    logger.warning(f"Failed to format {op_name}: {e}")

        # Generate summary based on intent
        summary = self._generate_summary(intent, results, plan)

        # Extract key insights
        insights = self._extract_insights(results, intent)

        # Generate recommendations
        recommendations = self._generate_recommendations(results, intent, plan)

        # Compute confidence
        confidence = self._compute_confidence(results, plan)

        # Build provenance
        provenance = {
            "operations_executed": [op.name for op in plan.operations],
            "total_cost": plan.total_cost,
            "intent": intent.value,
            "operations_succeeded": len([r for r in results.values() if r.get("success", True)])
        }

        meaning = MeaningResult(
            summary=summary,
            details=details,
            key_insights=insights,
            recommendations=recommendations,
            confidence=confidence,
            provenance=provenance
        )

        self.synthesis_history.append(meaning)

        logger.info(f"Synthesized meaning: {len(details)} details, {len(insights)} insights")

        return meaning

    def _generate_summary(
        self,
        intent: QueryIntent,
        results: Dict[str, Any],
        plan: OperationPlan
    ) -> str:
        """Generate high-level summary based on intent."""

        if intent == QueryIntent.SIMILARITY:
            n_results = sum(
                len(r.get("similarities", []))
                for r in results.values()
            )
            return f"Found {n_results} similar items using {len(plan.operations)} mathematical operations."

        elif intent == QueryIntent.OPTIMIZATION:
            improvement = max(
                r.get("improvement", 0.0)
                for r in results.values()
            )
            return f"Optimized with {improvement:.1%} improvement using gradient-based methods."

        elif intent == QueryIntent.ANALYSIS:
            return f"Completed comprehensive analysis using {len(plan.operations)} mathematical techniques."

        elif intent == QueryIntent.VERIFICATION:
            passed = sum(
                1 for r in results.values()
                if r.get("is_valid", True) or r.get("all_passed", True)
            )
            return f"Verified {passed}/{len(results)} mathematical properties successfully."

        elif intent == QueryIntent.TRANSFORMATION:
            return f"Transformed representation through {len(plan.operations)} mathematical operations."

        elif intent == QueryIntent.DECISION:
            selected = next(
                (r.get("selected_option", "option") for r in results.values() if "selected_option" in r),
                "optimal choice"
            )
            return f"Decision analysis recommends: {selected}."

        else:
            return f"Executed {len(plan.operations)} mathematical operations successfully."

    def _extract_insights(
        self,
        results: Dict[str, Any],
        intent: QueryIntent
    ) -> List[str]:
        """Extract key insights from results."""
        insights = []

        # Similarity insights
        if "inner_product" in results:
            sims = results["inner_product"].get("similarities", [])
            if sims:
                avg_sim = np.mean(sims)
                if avg_sim > 0.8:
                    insights.append("Very high similarity detected - items are closely related")
                elif avg_sim < 0.3:
                    insights.append("Low similarity - items are quite different")

        # Optimization insights
        if "gradient" in results:
            grad_norm = results["gradient"].get("gradient_norm", 0.0)
            if grad_norm < 0.01:
                insights.append("Near optimal - gradient is very small")
            elif grad_norm > 1.0:
                insights.append("Far from optimal - large gradient indicates room for improvement")

        # Stability insights
        if "eigenvalues" in results:
            max_eig = results["eigenvalues"].get("max_eigenvalue", 0.0)
            if max_eig < 1.0:
                insights.append("System is stable - eigenvalues within unit circle")
            else:
                insights.append("System may be unstable - eigenvalue exceeds unity")

        # Convergence insights
        if "convergence_analysis" in results:
            is_converging = results["convergence_analysis"].get("is_converging", False)
            if is_converging:
                limit = results["convergence_analysis"].get("limit", 0.0)
                insights.append(f"Process is converging toward {limit:.3f}")
            else:
                insights.append("Process is not converging - may need different approach")

        return insights

    def _generate_recommendations(
        self,
        results: Dict[str, Any],
        intent: QueryIntent,
        plan: OperationPlan
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # High cost → recommend cheaper alternatives
        if plan.total_cost > 40:
            recommendations.append(
                f"Consider cheaper operations - current cost is {plan.total_cost}"
            )

        # Instability → recommend stabilization
        if "eigenvalues" in results:
            max_eig = results["eigenvalues"].get("max_eigenvalue", 0.0)
            if max_eig > 1.0:
                recommendations.append(
                    "Consider adding regularization to improve stability"
                )

        # Poor convergence → recommend adjustments
        if "convergence_analysis" in results:
            if not results["convergence_analysis"].get("is_converging", True):
                recommendations.append(
                    "Try adjusting learning rate or optimization parameters"
                )

        # Low similarity → recommend different approach
        if "inner_product" in results:
            sims = results["inner_product"].get("similarities", [])
            if sims and np.mean(sims) < 0.3:
                recommendations.append(
                    "Low similarity - consider alternative similarity metrics or data preprocessing"
                )

        return recommendations

    def _compute_confidence(
        self,
        results: Dict[str, Any],
        plan: OperationPlan
    ) -> float:
        """Compute confidence in the meaning."""
        # Base confidence
        confidence = 1.0

        # Reduce if any operations failed
        failed = sum(
            1 for r in results.values()
            if not r.get("success", True)
        )
        if failed > 0:
            confidence *= (1.0 - 0.1 * failed)

        # Reduce if verification failed
        verification_failed = sum(
            1 for r in results.values()
            if "all_passed" in r and not r["all_passed"]
        )
        if verification_failed > 0:
            confidence *= 0.8

        # Reduce if very high cost (may have skipped operations)
        if plan.total_cost < 10:
            confidence *= 0.95  # Minimal analysis

        return max(0.0, min(1.0, confidence))


# ============================================================================
# Complete Pipeline: Query → Math → Meaning
# ============================================================================

class CompleteMathMeaningPipeline:
    """
    End-to-end pipeline: Query → Math Selection → Execution → Meaning.

    This is the COMPLETE system that takes text input and produces
    natural language output with rigorous mathematical computation in between.

    Example:
        pipeline = CompleteMathMeaningPipeline()

        response = pipeline.process(
            query="Find documents similar to quantum computing",
            context={"has_embeddings": True}
        )

        print(response.to_text(style="detailed"))
        # Output:
        # "Found 5 similar items using 3 mathematical operations.
        #
        #  Analysis:
        #    - Computed similarity scores using dot products. Top scores: 0.85, 0.72, 0.68
        #    - Calculated distances in semantic space. Closest within 0.15 units
        #
        #  Key Insights:
        #    • Very high similarity detected - items are closely related
        #
        #  Recommendations:
        #    → Consider expanding search to related topics"
    """

    def __init__(self, load_state: bool = True):
        """
        Initialize complete pipeline.

        Args:
            load_state: Load saved RL state
        """
        self.selector = SmartMathOperationSelector(load_state=load_state)
        self.synthesizer = MeaningSynthesizer()

        logger.info("CompleteMathMeaningPipeline initialized")

    def process(
        self,
        query: str,
        context: Optional[Dict] = None,
        budget: int = 50,
        style: str = "detailed"
    ) -> MeaningResult:
        """
        Process query through complete pipeline.

        Args:
            query: User query text
            context: Optional context
            budget: Computational budget
            style: Output style ("concise", "detailed", "technical")

        Returns:
            MeaningResult with natural language response
        """
        logger.info(f"Processing query: {query}")

        # 1. Plan operations (RL selection)
        plan = self.selector.plan_operations_smart(
            query_text=query,
            context=context,
            budget=budget,
            enable_learning=True,
            enable_composition=True
        )

        logger.info(f"  Planned {len(plan.operations)} operations, cost {plan.total_cost}")

        # 2. Execute operations (with verification)
        execution_result = self.selector.execute_plan_with_verification(
            plan=plan,
            data=context or {}
        )

        logger.info(f"  Executed operations, tests passed: {execution_result['all_tests_passed']}")

        # 3. Extract results
        results = {}
        for op_exec in execution_result["operations_executed"]:
            op_name = op_exec["operation"]
            # Mock results for demo (would be actual results in production)
            results[op_name] = self._generate_mock_results(op_name)

        # 4. Synthesize meaning
        intent = self.selector.classify_intent(query, context)[0]
        meaning = self.synthesizer.synthesize(
            results=results,
            intent=intent,
            plan=plan,
            context=context
        )

        logger.info(f"  Synthesized meaning with confidence {meaning.confidence:.2f}")

        # 5. Record feedback for RL
        self.selector.record_feedback(
            plan=plan,
            success=execution_result["all_tests_passed"],
            quality=meaning.confidence,
            execution_time=execution_result["total_time"]
        )

        return meaning

    def _generate_mock_results(self, operation: str) -> Dict[str, Any]:
        """Generate mock results for demo (would be actual results in production)."""
        if operation == "inner_product":
            return {"similarities": [0.85, 0.72, 0.68, 0.55, 0.42], "success": True}
        elif operation == "metric_distance":
            return {"min_distance": 0.15, "max_distance": 1.85, "success": True}
        elif operation == "gradient":
            return {"gradient_norm": 0.045, "direction": [0.1, -0.2, 0.3], "success": True}
        elif operation == "eigenvalues":
            return {"max_eigenvalue": 0.85, "eigenvalues": [0.85, 0.42, 0.18], "success": True}
        elif operation == "convergence_analysis":
            return {"is_converging": True, "limit": 0.923, "success": True}
        else:
            return {"success": True}


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("COMPLETE PIPELINE: QUERY -> MATH -> MEANING")
    print("="*80)
    print()

    # Create complete pipeline
    pipeline = CompleteMathMeaningPipeline(load_state=False)

    # Test queries
    queries = [
        ("Find documents similar to quantum computing", QueryIntent.SIMILARITY),
        ("Optimize the retrieval algorithm", QueryIntent.OPTIMIZATION),
        ("Analyze convergence of the learning process", QueryIntent.ANALYSIS),
        ("Verify that the distance function is valid", QueryIntent.VERIFICATION),
    ]

    for query, expected_intent in queries:
        print("="*80)
        print(f"QUERY: {query}")
        print("="*80)
        print()

        # Process through complete pipeline
        meaning = pipeline.process(
            query=query,
            context={"has_embeddings": True},
            budget=50
        )

        # Show detailed output
        print(meaning.to_text(style="detailed"))
        print()
        print(f"Confidence: {meaning.confidence:.1%}")
        print(f"Operations: {meaning.provenance.get('operations_executed', [])}")
        print()

    # Show learning statistics
    print("="*80)
    print("LEARNING STATISTICS")
    print("="*80)
    stats = pipeline.selector.get_smart_statistics()
    print(f"Total feedback: {stats['rl_learning']['total_feedback']}")
    print(f"Synthesis history: {len(pipeline.synthesizer.synthesis_history)} meanings")

    print()
    print("="*80)
    print("COMPLETE: Query -> Math (RL + Testing) -> Meaning -> Words")
    print("="*80)
