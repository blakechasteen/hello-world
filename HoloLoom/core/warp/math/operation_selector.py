#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Operation Selector
================================
The intelligent meta-layer that chooses WHICH mathematical operations to apply
based on query characteristics, context, and computational goals.

This is the "tool that chooses which math to expose" - the decision layer
that sits between the weaving orchestrator and the 32 mathematical modules.

Philosophy:
Not all queries need Ricci flow or Galois theory. The selector analyzes the
query context and selects the minimal necessary mathematical machinery:

- "Find similar documents" → Metric spaces + inner products
- "Optimize retrieval" → Gradient descent + convex optimization
- "Analyze convergence" → Real analysis sequences + limits
- "Detect hierarchies" → Hyperbolic geometry + Poincaré ball
- "Ensure stability" → Spectral analysis + eigenvalues
- "Transform features" → Fourier analysis + wavelets

The selector provides JUSTIFICATION for each mathematical choice, creating
full computational provenance.

Author: HoloLoom Team
Date: 2025-10-26
"""

import logging
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Operation Categories
# ============================================================================

class MathDomain(Enum):
    """Mathematical domain categories."""
    ANALYSIS = "analysis"           # Real, complex, functional analysis
    ALGEBRA = "algebra"             # Abstract algebra, module theory
    GEOMETRY = "geometry"           # Differential, Riemannian, hyperbolic
    PROBABILITY = "probability"     # Measure theory, stochastic calculus
    OPTIMIZATION = "optimization"   # Gradient descent, convex optimization
    LOGIC = "logic"                 # Mathematical logic, computability
    DECISION = "decision"           # Game theory, information theory
    EXTENSIONS = "extensions"       # Specialized (combinatorics, curvature)


class OperationLevel(Enum):
    """Computational complexity level."""
    BASIC = "basic"         # O(n) - inner products, norms, distances
    MODERATE = "moderate"   # O(n²) - matrix operations, SVD
    ADVANCED = "advanced"   # O(n³) or iterative - eigenvalues, optimization
    EXPENSIVE = "expensive" # O(n⁴+) or very iterative - Ricci flow, MCTS


class QueryIntent(Enum):
    """High-level query intent categories."""
    SIMILARITY = "similarity"             # Find similar items
    OPTIMIZATION = "optimization"         # Improve/optimize something
    ANALYSIS = "analysis"                 # Analyze/understand structure
    GENERATION = "generation"             # Generate new content
    DECISION = "decision"                 # Make a choice
    VERIFICATION = "verification"         # Verify properties/correctness
    TRANSFORMATION = "transformation"     # Transform representation


# ============================================================================
# Operation Specifications
# ============================================================================

@dataclass
class MathOperation:
    """
    Specification for a mathematical operation.

    Attributes:
        name: Operation name
        domain: Mathematical domain
        level: Complexity level
        description: What the operation does
        use_cases: When to use this operation
        prerequisites: Required prior operations
        module_path: Python module path
        function_name: Function/class name
        estimated_cost: Relative computational cost (1-100)
    """
    name: str
    domain: MathDomain
    level: OperationLevel
    description: str
    use_cases: List[QueryIntent]
    prerequisites: List[str] = field(default_factory=list)
    module_path: str = ""
    function_name: str = ""
    estimated_cost: int = 1

    def is_applicable(self, intent: QueryIntent) -> bool:
        """Check if operation applies to given intent."""
        return intent in self.use_cases


@dataclass
class OperationPlan:
    """
    Complete plan of mathematical operations to apply.

    Attributes:
        operations: Ordered list of operations to execute
        justifications: Why each operation was chosen
        total_cost: Estimated total computational cost
        domains_used: Set of mathematical domains involved
        metadata: Additional planning metadata
    """
    operations: List[MathOperation]
    justifications: Dict[str, str]
    total_cost: int
    domains_used: Set[MathDomain]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_execution_order(self) -> List[str]:
        """Get operation names in execution order."""
        return [op.name for op in self.operations]

    def summarize(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Mathematical Operation Plan:",
            f"  Operations: {len(self.operations)}",
            f"  Domains: {', '.join(d.value for d in self.domains_used)}",
            f"  Est. Cost: {self.total_cost}",
            f"\nExecution Order:"
        ]
        for i, op in enumerate(self.operations, 1):
            lines.append(f"  {i}. {op.name} ({op.domain.value})")
            if op.name in self.justifications:
                lines.append(f"     -> {self.justifications[op.name]}")
        return "\n".join(lines)


# ============================================================================
# Mathematical Operation Selector
# ============================================================================

class MathOperationSelector:
    """
    Intelligent selector that chooses which mathematical operations to apply.

    This is the meta-layer that analyzes query context and selects the minimal
    necessary mathematical machinery from the 32 available modules.

    The selector provides:
    - Intent classification (what is the query trying to do?)
    - Operation selection (which math operations are needed?)
    - Execution planning (in what order?)
    - Cost estimation (how expensive will this be?)
    - Justification (why was each operation chosen?)

    Example:
        selector = MathOperationSelector()

        # Query: "Find documents similar to this one"
        plan = selector.plan_operations(
            query_text="Find similar documents",
            query_embedding=query_emb,
            context={"has_embeddings": True}
        )

        # Result:
        # 1. MetricSpace (verify distances are valid)
        # 2. InnerProduct (compute similarities)
        # 3. Ranking (sort by similarity)
    """

    def __init__(self):
        """Initialize operation selector with operation catalog."""
        self.operations = self._build_operation_catalog()
        self.planning_history: List[OperationPlan] = []
        self.intent_keywords = self._build_intent_keywords()

        logger.info("MathOperationSelector initialized")
        logger.info(f"  Operations available: {len(self.operations)}")
        logger.info(f"  Domains: {len(set(op.domain for op in self.operations.values()))}")

    def _build_operation_catalog(self) -> Dict[str, MathOperation]:
        """
        Build catalog of all available mathematical operations.

        Returns:
            Dict mapping operation name to MathOperation
        """
        ops = {}

        # ====================================================================
        # BASIC OPERATIONS (Layer 0: Linear Algebra)
        # ====================================================================

        ops["inner_product"] = MathOperation(
            name="inner_product",
            domain=MathDomain.ALGEBRA,
            level=OperationLevel.BASIC,
            description="Compute inner product (dot product) between vectors",
            use_cases=[QueryIntent.SIMILARITY, QueryIntent.ANALYSIS],
            module_path="HoloLoom.warp.math.algebra",
            function_name="inner_product",
            estimated_cost=1
        )

        ops["norm"] = MathOperation(
            name="norm",
            domain=MathDomain.ANALYSIS,
            level=OperationLevel.BASIC,
            description="Compute vector norm (magnitude)",
            use_cases=[QueryIntent.ANALYSIS, QueryIntent.VERIFICATION],
            module_path="HoloLoom.warp.math.analysis.functional_analysis",
            function_name="NormedSpace",
            estimated_cost=1
        )

        ops["metric_distance"] = MathOperation(
            name="metric_distance",
            domain=MathDomain.ANALYSIS,
            level=OperationLevel.BASIC,
            description="Compute metric distance between points",
            use_cases=[QueryIntent.SIMILARITY, QueryIntent.VERIFICATION],
            module_path="HoloLoom.warp.math.analysis.real_analysis",
            function_name="MetricSpace",
            estimated_cost=1
        )

        # ====================================================================
        # MODERATE OPERATIONS (Layer 1: Calculus & Optimization)
        # ====================================================================

        ops["gradient"] = MathOperation(
            name="gradient",
            domain=MathDomain.ANALYSIS,
            level=OperationLevel.MODERATE,
            description="Compute gradient for optimization",
            use_cases=[QueryIntent.OPTIMIZATION, QueryIntent.ANALYSIS],
            module_path="HoloLoom.warp.math.analysis.real_analysis",
            function_name="Differentiator.gradient",
            estimated_cost=5
        )

        ops["svd"] = MathOperation(
            name="svd",
            domain=MathDomain.ALGEBRA,
            level=OperationLevel.MODERATE,
            description="Singular Value Decomposition for dimensionality reduction",
            use_cases=[QueryIntent.TRANSFORMATION, QueryIntent.ANALYSIS],
            prerequisites=["inner_product"],
            module_path="numpy.linalg",
            function_name="svd",
            estimated_cost=10
        )

        ops["gram_schmidt"] = MathOperation(
            name="gram_schmidt",
            domain=MathDomain.ALGEBRA,
            level=OperationLevel.MODERATE,
            description="Gram-Schmidt orthogonalization for diverse context",
            use_cases=[QueryIntent.TRANSFORMATION, QueryIntent.OPTIMIZATION],
            prerequisites=["inner_product", "norm"],
            module_path="HoloLoom.warp.math.analysis.functional_analysis",
            function_name="HilbertSpace.gram_schmidt",
            estimated_cost=8
        )

        # ====================================================================
        # ADVANCED OPERATIONS (Layer 2: Spectral & Topology)
        # ====================================================================

        ops["eigenvalues"] = MathOperation(
            name="eigenvalues",
            domain=MathDomain.ALGEBRA,
            level=OperationLevel.ADVANCED,
            description="Compute eigenvalues for spectral analysis",
            use_cases=[QueryIntent.ANALYSIS, QueryIntent.VERIFICATION],
            prerequisites=["inner_product"],
            module_path="HoloLoom.warp.math.analysis.functional_analysis",
            function_name="SpectralAnalyzer",
            estimated_cost=15
        )

        ops["laplacian"] = MathOperation(
            name="laplacian",
            domain=MathDomain.GEOMETRY,
            level=OperationLevel.ADVANCED,
            description="Graph Laplacian for topological analysis",
            use_cases=[QueryIntent.ANALYSIS, QueryIntent.TRANSFORMATION],
            prerequisites=["metric_distance"],
            module_path="HoloLoom.warp.math.geometry",
            function_name="LaplacianOperator",
            estimated_cost=12
        )

        ops["fourier_transform"] = MathOperation(
            name="fourier_transform",
            domain=MathDomain.ANALYSIS,
            level=OperationLevel.ADVANCED,
            description="Fourier transform for frequency analysis",
            use_cases=[QueryIntent.ANALYSIS, QueryIntent.TRANSFORMATION],
            module_path="HoloLoom.warp.math.analysis.fourier_harmonic",
            function_name="FourierTransform",
            estimated_cost=10
        )

        # ====================================================================
        # GEOMETRY OPERATIONS
        # ====================================================================

        ops["hyperbolic_distance"] = MathOperation(
            name="hyperbolic_distance",
            domain=MathDomain.GEOMETRY,
            level=OperationLevel.MODERATE,
            description="Hyperbolic distance for hierarchical embeddings",
            use_cases=[QueryIntent.SIMILARITY, QueryIntent.ANALYSIS],
            module_path="HoloLoom.warp.math.extensions.hyperbolic_geometry",
            function_name="PoincareBall.distance",
            estimated_cost=5
        )

        ops["geodesic"] = MathOperation(
            name="geodesic",
            domain=MathDomain.GEOMETRY,
            level=OperationLevel.ADVANCED,
            description="Compute geodesic (shortest path on manifold)",
            use_cases=[QueryIntent.OPTIMIZATION, QueryIntent.ANALYSIS],
            prerequisites=["metric_distance"],
            module_path="HoloLoom.warp.math.geometry.riemannian_geometry",
            function_name="RiemannianManifold.geodesic",
            estimated_cost=20
        )

        # ====================================================================
        # PROBABILITY & DECISION OPERATIONS
        # ====================================================================

        ops["thompson_sampling"] = MathOperation(
            name="thompson_sampling",
            domain=MathDomain.DECISION,
            level=OperationLevel.MODERATE,
            description="Thompson Sampling for exploration/exploitation",
            use_cases=[QueryIntent.DECISION, QueryIntent.OPTIMIZATION],
            module_path="HoloLoom.convergence.engine",
            function_name="ThompsonBandit",
            estimated_cost=3
        )

        ops["entropy"] = MathOperation(
            name="entropy",
            domain=MathDomain.PROBABILITY,
            level=OperationLevel.BASIC,
            description="Compute Shannon entropy for information content",
            use_cases=[QueryIntent.ANALYSIS, QueryIntent.DECISION],
            module_path="HoloLoom.warp.math.decision.information_theory",
            function_name="InformationTheory.entropy",
            estimated_cost=2
        )

        ops["kl_divergence"] = MathOperation(
            name="kl_divergence",
            domain=MathDomain.PROBABILITY,
            level=OperationLevel.BASIC,
            description="KL divergence for distribution comparison",
            use_cases=[QueryIntent.SIMILARITY, QueryIntent.ANALYSIS],
            prerequisites=["entropy"],
            module_path="HoloLoom.warp.math.decision.information_theory",
            function_name="InformationTheory.kl_divergence",
            estimated_cost=3
        )

        # ====================================================================
        # EXPENSIVE OPERATIONS (Layer 3+: Advanced Mathematics)
        # ====================================================================

        ops["ricci_flow"] = MathOperation(
            name="ricci_flow",
            domain=MathDomain.GEOMETRY,
            level=OperationLevel.EXPENSIVE,
            description="Ricci flow for manifold smoothing",
            use_cases=[QueryIntent.TRANSFORMATION, QueryIntent.OPTIMIZATION],
            prerequisites=["geodesic", "laplacian"],
            module_path="HoloLoom.warp.math.extensions.advanced_curvature",
            function_name="RicciFlowAdvanced",
            estimated_cost=50
        )

        ops["spectral_clustering"] = MathOperation(
            name="spectral_clustering",
            domain=MathDomain.GEOMETRY,
            level=OperationLevel.EXPENSIVE,
            description="Spectral clustering using graph Laplacian",
            use_cases=[QueryIntent.ANALYSIS, QueryIntent.TRANSFORMATION],
            prerequisites=["laplacian", "eigenvalues"],
            module_path="HoloLoom.warp.math.geometry",
            function_name="SpectralClustering",
            estimated_cost=30
        )

        # ====================================================================
        # VERIFICATION OPERATIONS
        # ====================================================================

        ops["metric_verification"] = MathOperation(
            name="metric_verification",
            domain=MathDomain.ANALYSIS,
            level=OperationLevel.MODERATE,
            description="Verify metric space axioms (triangle inequality, etc.)",
            use_cases=[QueryIntent.VERIFICATION, QueryIntent.ANALYSIS],
            prerequisites=["metric_distance"],
            module_path="HoloLoom.warp.math.analysis.real_analysis",
            function_name="MetricSpace.is_metric",
            estimated_cost=8
        )

        ops["continuity_check"] = MathOperation(
            name="continuity_check",
            domain=MathDomain.ANALYSIS,
            level=OperationLevel.MODERATE,
            description="Verify function continuity (Lipschitz constant)",
            use_cases=[QueryIntent.VERIFICATION, QueryIntent.ANALYSIS],
            prerequisites=["metric_distance"],
            module_path="HoloLoom.warp.math.analysis.real_analysis",
            function_name="ContinuityChecker",
            estimated_cost=10
        )

        ops["convergence_analysis"] = MathOperation(
            name="convergence_analysis",
            domain=MathDomain.ANALYSIS,
            level=OperationLevel.MODERATE,
            description="Analyze sequence convergence",
            use_cases=[QueryIntent.VERIFICATION, QueryIntent.ANALYSIS],
            module_path="HoloLoom.warp.math.analysis.real_analysis",
            function_name="SequenceAnalyzer",
            estimated_cost=5
        )

        return ops

    def _build_intent_keywords(self) -> Dict[QueryIntent, List[str]]:
        """
        Build keyword mappings for intent classification.

        Returns:
            Dict mapping intent to keyword list
        """
        return {
            QueryIntent.SIMILARITY: [
                "similar", "find", "retrieve", "search", "match", "like",
                "closest", "nearest", "related", "compare"
            ],
            QueryIntent.OPTIMIZATION: [
                "optimize", "improve", "better", "best", "minimize", "maximize",
                "efficient", "faster", "enhance", "refine", "tune"
            ],
            QueryIntent.ANALYSIS: [
                "analyze", "understand", "explain", "why", "how", "structure",
                "pattern", "relationship", "property", "characteristic"
            ],
            QueryIntent.GENERATION: [
                "generate", "create", "produce", "synthesize", "build", "make",
                "construct", "compose", "write"
            ],
            QueryIntent.DECISION: [
                "choose", "select", "decide", "pick", "which", "should",
                "recommend", "suggest", "prefer"
            ],
            QueryIntent.VERIFICATION: [
                "verify", "check", "validate", "test", "prove", "confirm",
                "ensure", "guarantee", "correct", "valid"
            ],
            QueryIntent.TRANSFORMATION: [
                "transform", "convert", "change", "map", "project", "embed",
                "reduce", "expand", "encode", "decode"
            ]
        }

    def classify_intent(
        self,
        query_text: str,
        context: Optional[Dict] = None
    ) -> List[QueryIntent]:
        """
        Classify query intent using keyword matching.

        Args:
            query_text: Query string
            context: Optional context dict

        Returns:
            List of detected intents (ordered by confidence)
        """
        query_lower = query_text.lower()
        intent_scores = {intent: 0 for intent in QueryIntent}

        # Keyword matching
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    intent_scores[intent] += 1

        # Context-based hints
        if context:
            if context.get("has_embeddings"):
                intent_scores[QueryIntent.SIMILARITY] += 1
            if context.get("requires_optimization"):
                intent_scores[QueryIntent.OPTIMIZATION] += 1
            if context.get("needs_verification"):
                intent_scores[QueryIntent.VERIFICATION] += 1

        # Sort by score
        ranked = sorted(
            intent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return intents with score > 0
        detected = [intent for intent, score in ranked if score > 0]

        # Default to ANALYSIS if no intent detected
        if not detected:
            detected = [QueryIntent.ANALYSIS]

        return detected

    def plan_operations(
        self,
        query_text: str,
        query_embedding: Optional[np.ndarray] = None,
        context: Optional[Dict] = None,
        budget: Optional[int] = None,
        enable_expensive: bool = False
    ) -> OperationPlan:
        """
        Plan mathematical operations for query.

        This is the MAIN API - analyzes query and selects operations.

        Args:
            query_text: Query string
            query_embedding: Optional query embedding
            context: Optional context dict
            budget: Optional cost budget (operations with total cost > budget excluded)
            enable_expensive: Allow expensive operations (Ricci flow, etc.)

        Returns:
            OperationPlan with selected operations and justifications
        """
        logger.info(f"Planning operations for: {query_text[:50]}...")

        # Classify intent
        intents = self.classify_intent(query_text, context)
        primary_intent = intents[0] if intents else QueryIntent.ANALYSIS

        logger.info(f"  Detected intents: {[i.value for i in intents]}")
        logger.info(f"  Primary intent: {primary_intent.value}")

        # Select applicable operations
        selected = []
        justifications = {}

        for op in self.operations.values():
            # Check if operation applies to any detected intent
            if any(op.is_applicable(intent) for intent in intents):
                # Skip expensive operations unless enabled
                if op.level == OperationLevel.EXPENSIVE and not enable_expensive:
                    continue

                # Add operation
                selected.append(op)

                # Generate justification
                matching_intents = [i.value for i in intents if op.is_applicable(i)]
                justifications[op.name] = (
                    f"Applies to {', '.join(matching_intents)} intent(s)"
                )

        # Sort by prerequisites (topological sort)
        ordered = self._topological_sort(selected)

        # Apply budget constraint if provided
        if budget is not None:
            cumulative_cost = 0
            filtered = []
            for op in ordered:
                if cumulative_cost + op.estimated_cost <= budget:
                    filtered.append(op)
                    cumulative_cost += op.estimated_cost
                else:
                    logger.info(f"  Budget exceeded, skipping {op.name}")
            ordered = filtered

        # Calculate total cost
        total_cost = sum(op.estimated_cost for op in ordered)

        # Collect domains
        domains_used = set(op.domain for op in ordered)

        # Create plan
        plan = OperationPlan(
            operations=ordered,
            justifications=justifications,
            total_cost=total_cost,
            domains_used=domains_used,
            metadata={
                "primary_intent": primary_intent.value,
                "all_intents": [i.value for i in intents],
                "query": query_text[:100],
                "budget": budget,
                "budget_met": budget is None or total_cost <= budget
            }
        )

        # Record in history
        self.planning_history.append(plan)

        logger.info(f"  Selected {len(ordered)} operations")
        logger.info(f"  Total cost: {total_cost}")
        logger.info(f"  Domains: {[d.value for d in domains_used]}")

        return plan

    def _topological_sort(self, operations: List[MathOperation]) -> List[MathOperation]:
        """
        Sort operations respecting prerequisites (topological sort).

        Args:
            operations: Unsorted operations

        Returns:
            Topologically sorted operations
        """
        # Build dependency graph
        op_map = {op.name: op for op in operations}
        in_degree = {op.name: 0 for op in operations}

        for op in operations:
            for prereq in op.prerequisites:
                if prereq in op_map:
                    in_degree[op.name] += 1

        # Kahn's algorithm
        queue = [name for name, degree in in_degree.items() if degree == 0]
        sorted_ops = []

        while queue:
            name = queue.pop(0)
            sorted_ops.append(op_map[name])

            # Reduce in-degree for dependents
            for other_name, other_op in op_map.items():
                if name in other_op.prerequisites:
                    in_degree[other_name] -= 1
                    if in_degree[other_name] == 0:
                        queue.append(other_name)

        # Check for cycles
        if len(sorted_ops) != len(operations):
            logger.warning("Cycle detected in operation prerequisites, using original order")
            return operations

        return sorted_ops

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get selector statistics.

        Returns:
            Dict with usage statistics
        """
        if not self.planning_history:
            return {"total_plans": 0}

        # Aggregate statistics
        total_plans = len(self.planning_history)
        avg_ops = np.mean([len(p.operations) for p in self.planning_history])
        avg_cost = np.mean([p.total_cost for p in self.planning_history])

        # Most common operations
        op_counts = {}
        for plan in self.planning_history:
            for op in plan.operations:
                op_counts[op.name] = op_counts.get(op.name, 0) + 1

        most_common = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Intent distribution
        intent_counts = {}
        for plan in self.planning_history:
            primary = plan.metadata.get("primary_intent", "unknown")
            intent_counts[primary] = intent_counts.get(primary, 0) + 1

        return {
            "total_plans": total_plans,
            "avg_operations_per_plan": avg_ops,
            "avg_cost_per_plan": avg_cost,
            "most_common_operations": dict(most_common),
            "intent_distribution": intent_counts,
            "total_operations_available": len(self.operations)
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_selector() -> MathOperationSelector:
    """
    Create default MathOperationSelector.

    Returns:
        Configured selector
    """
    return MathOperationSelector()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MATHEMATICAL OPERATION SELECTOR")
    print("The Meta-Layer That Chooses Which Math To Expose")
    print("="*80)
    print()

    # Create selector
    selector = MathOperationSelector()

    print(f"Initialized with {len(selector.operations)} operations")
    print()

    # Example queries with different intents
    test_queries = [
        ("Find documents similar to this query", {}),
        ("Optimize the retrieval quality", {"requires_optimization": True}),
        ("Analyze the convergence of the learning process", {}),
        ("Verify that the metric space is valid", {"needs_verification": True}),
        ("Transform embeddings to hyperbolic space", {"has_embeddings": True}),
        ("What is the best tool to use?", {}),
    ]

    for idx, (query, context) in enumerate(test_queries, 1):
        print("="*80)
        print(f"QUERY {idx}: {query}")
        print("="*80)

        # Plan operations
        plan = selector.plan_operations(
            query_text=query,
            context=context,
            budget=50,  # Cost limit
            enable_expensive=False
        )

        # Show plan
        print(plan.summarize())
        print()

    # Show statistics
    print("="*80)
    print("SELECTOR STATISTICS")
    print("="*80)
    stats = selector.get_statistics()
    print(f"Total plans: {stats['total_plans']}")
    print(f"Avg operations/plan: {stats['avg_operations_per_plan']:.1f}")
    print(f"Avg cost/plan: {stats['avg_cost_per_plan']:.1f}")
    print(f"\nMost common operations:")
    for op_name, count in stats['most_common_operations'].items():
        print(f"  {op_name}: {count}")
    print(f"\nIntent distribution:")
    for intent, count in stats['intent_distribution'].items():
        print(f"  {intent}: {count}")

    print("\n" + "="*80)
    print("The selector chooses the minimal necessary mathematical machinery!")
    print("="*80)
