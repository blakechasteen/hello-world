"""
Intervention Engine - do() Operator Implementation

Implements Pearl's do-calculus for causal inference.

Key concepts:
- Intervention: Set variable to value, breaking incoming causal edges
- Graph surgery: Remove incoming edges to intervened variables
- Backdoor adjustment: Condition on confounders to identify causal effect
- Frontdoor adjustment: Use mediators when confounders are unobserved
"""

from typing import Dict, Set, Any, Optional, List
import numpy as np
from dataclasses import dataclass

from .dag import CausalDAG, CausalNode, NodeType
from .query import CausalQuery, CausalAnswer, QueryType


@dataclass
class InterventionResult:
    """
    Result of intervention.

    Attributes:
        intervened_variables: Variables that were set by do()
        intervened_values: Values they were set to
        mutilated_graph: Graph after intervention (edges removed)
        identifiable: Whether causal effect is identifiable
        identification_method: Method used (backdoor, frontdoor, etc.)
        adjustment_set: Variables to condition on
    """
    intervened_variables: Dict[str, Any]
    mutilated_graph: CausalDAG
    identifiable: bool
    identification_method: Optional[str] = None
    adjustment_set: Optional[Set[str]] = None
    explanation: str = ""


class InterventionEngine:
    """
    Implements do-operator and causal effect identification.

    Usage:
        engine = InterventionEngine(dag)

        # Simple intervention
        result = engine.do({"treatment": 1})

        # Query causal effect
        query = CausalQuery(
            query_type=QueryType.INTERVENTION,
            outcome="recovery",
            treatment="drug_A",
            treatment_value=1
        )
        answer = engine.query(query)
    """

    def __init__(self, dag: CausalDAG):
        """
        Initialize intervention engine.

        Args:
            dag: Causal DAG to operate on
        """
        self.dag = dag

    def do(self, interventions: Dict[str, Any]) -> InterventionResult:
        """
        Apply do-operator: Set variables to values, breaking incoming edges.

        This is "graph surgery" - we remove all incoming edges to intervened
        variables, making them exogenous (independent of their usual causes).

        Args:
            interventions: Variables to intervene on and their values
                          e.g., {"drug_A": 1, "exercise": 0}

        Returns:
            InterventionResult with mutilated graph
        """
        # Create copy of DAG
        mutilated = CausalDAG()

        # Copy all nodes
        for node in self.dag.nodes.values():
            mutilated.add_node(node)

        # Copy edges, but skip incoming edges to intervened variables
        for edge in self.dag.edges.values():
            if edge.target not in interventions:
                # Not intervening on target, keep edge
                mutilated.add_edge(edge)
            # else: Skip edge (graph surgery)

        # Mark intervened variables as INTERVENTION type
        for var in interventions:
            if var in mutilated.nodes:
                mutilated.nodes[var].node_type = NodeType.INTERVENTION

        return InterventionResult(
            intervened_variables=interventions,
            mutilated_graph=mutilated,
            identifiable=True,  # Intervention is always identifiable
            identification_method="do-calculus (graph surgery)",
            explanation=f"Intervened on {list(interventions.keys())}, removed incoming edges"
        )

    def identify_causal_effect(
        self,
        treatment: str,
        outcome: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> InterventionResult:
        """
        Identify causal effect of treatment on outcome.

        Tries multiple identification strategies:
        1. Backdoor adjustment (if backdoor criterion satisfied)
        2. Frontdoor adjustment (if frontdoor criterion satisfied)
        3. Full do-calculus (future: implement 3 rules of do-calculus)

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            evidence: Optional conditioning variables

        Returns:
            InterventionResult with identification strategy
        """
        if evidence is None:
            evidence = {}

        # Strategy 1: Try backdoor adjustment
        backdoor_result = self._try_backdoor_adjustment(treatment, outcome, evidence)
        if backdoor_result.identifiable:
            return backdoor_result

        # Strategy 2: Try frontdoor adjustment
        frontdoor_result = self._try_frontdoor_adjustment(treatment, outcome, evidence)
        if frontdoor_result.identifiable:
            return frontdoor_result

        # Strategy 3: Try instrumental variables (if available)
        # TODO: Implement IV identification

        # Not identifiable with current strategies
        return InterventionResult(
            intervened_variables={treatment: None},
            mutilated_graph=self.do({treatment: None}).mutilated_graph,
            identifiable=False,
            explanation=f"Causal effect of {treatment} on {outcome} is not identifiable "
                       f"using backdoor or frontdoor adjustment. May require do-calculus."
        )

    def _try_backdoor_adjustment(
        self,
        treatment: str,
        outcome: str,
        evidence: Dict[str, Any]
    ) -> InterventionResult:
        """
        Try to identify causal effect using backdoor criterion.

        Backdoor adjustment: P(Y|do(X=x)) = Σ_z P(Y|X=x,Z=z)P(Z=z)

        Where Z is a set that:
        1. Blocks all backdoor paths from X to Y
        2. Contains no descendants of X
        """
        # Find valid backdoor adjustment sets
        # Start with all ancestors of treatment and outcome (excluding descendants of treatment)
        treatment_descendants = self.dag.descendants(treatment)
        treatment_ancestors = self.dag.ancestors(treatment)
        outcome_ancestors = self.dag.ancestors(outcome)

        # Candidate adjustment sets are ancestors of both (minus treatment descendants)
        candidate_ancestors = (treatment_ancestors | outcome_ancestors) - treatment_descendants - {treatment, outcome}

        # Check minimal adjustment set (just confounders)
        confounders = self.dag.find_confounders(treatment, outcome)
        minimal_adjustment = confounders - treatment_descendants

        if self.dag.satisfies_backdoor_criterion(treatment, outcome, minimal_adjustment):
            # Create intervention result
            mutilated = self.do({treatment: None}).mutilated_graph

            return InterventionResult(
                intervened_variables={treatment: None},
                mutilated_graph=mutilated,
                identifiable=True,
                identification_method="backdoor adjustment",
                adjustment_set=minimal_adjustment,
                explanation=f"Causal effect identifiable via backdoor adjustment. "
                           f"Adjust for: {minimal_adjustment}"
            )

        # Try full ancestor set
        if self.dag.satisfies_backdoor_criterion(treatment, outcome, candidate_ancestors):
            mutilated = self.do({treatment: None}).mutilated_graph

            return InterventionResult(
                intervened_variables={treatment: None},
                mutilated_graph=mutilated,
                identifiable=True,
                identification_method="backdoor adjustment (full ancestors)",
                adjustment_set=candidate_ancestors,
                explanation=f"Causal effect identifiable via backdoor adjustment. "
                           f"Adjust for: {candidate_ancestors}"
            )

        # Backdoor adjustment not applicable
        return InterventionResult(
            intervened_variables={treatment: None},
            mutilated_graph=self.dag,
            identifiable=False,
            explanation="No valid backdoor adjustment set found"
        )

    def _try_frontdoor_adjustment(
        self,
        treatment: str,
        outcome: str,
        evidence: Dict[str, Any]
    ) -> InterventionResult:
        """
        Try to identify causal effect using frontdoor criterion.

        Frontdoor adjustment useful when there are unmeasured confounders.

        Frontdoor formula: P(Y|do(X=x)) = Σ_z P(Z=z|X=x) Σ_x' P(Y|X=x',Z=z)P(X=x')

        Where Z satisfies:
        1. Z intercepts all directed paths from X to Y
        2. No backdoor paths from X to Z
        3. X blocks all backdoor paths from Z to Y
        """
        # Find mediators (nodes on directed paths X → Y)
        mediators = self.dag.find_mediators(treatment, outcome)

        # Check frontdoor criterion for mediator set
        if mediators and self.dag.satisfies_frontdoor_criterion(treatment, outcome, mediators):
            mutilated = self.do({treatment: None}).mutilated_graph

            return InterventionResult(
                intervened_variables={treatment: None},
                mutilated_graph=mutilated,
                identifiable=True,
                identification_method="frontdoor adjustment",
                adjustment_set=mediators,
                explanation=f"Causal effect identifiable via frontdoor adjustment. "
                           f"Use mediators: {mediators}"
            )

        return InterventionResult(
            intervened_variables={treatment: None},
            mutilated_graph=self.dag,
            identifiable=False,
            explanation="No valid frontdoor adjustment set found"
        )

    def query(
        self,
        query: CausalQuery,
        data: Optional[np.ndarray] = None
    ) -> CausalAnswer:
        """
        Answer causal query using identification + estimation.

        Args:
            query: Causal query to answer
            data: Optional observational data for estimation

        Returns:
            CausalAnswer with result and explanation
        """
        if not query.is_interventional():
            raise ValueError(f"InterventionEngine handles interventional queries, got {query.query_type}")

        # Identify causal effect
        identification = self.identify_causal_effect(
            treatment=query.treatment,
            outcome=query.outcome,
            evidence=query.evidence
        )

        if not identification.identifiable:
            return CausalAnswer(
                query=query,
                result=np.nan,
                confidence=0.0,
                method="not identifiable",
                assumptions=[],
                explanation=identification.explanation
            )

        # If we have data, estimate effect
        if data is not None:
            result, confidence = self._estimate_effect(query, identification, data)
        else:
            # No data, just return identification result
            result = np.nan
            confidence = 0.0

        # Build assumptions list
        assumptions = []
        if identification.identification_method == "backdoor adjustment":
            assumptions.extend([
                f"Adjustment set {identification.adjustment_set} blocks all backdoor paths",
                "No unmeasured confounders beyond adjustment set",
                "Correct functional form",
                "Positivity (treatment assignment has positive probability)"
            ])
        elif identification.identification_method == "frontdoor adjustment":
            assumptions.extend([
                f"Mediators {identification.adjustment_set} intercept all directed paths",
                "No unmeasured confounders of treatment-mediator relation",
                "Treatment blocks all backdoor paths from mediators to outcome"
            ])

        return CausalAnswer(
            query=query,
            result=result,
            confidence=confidence,
            method=identification.identification_method,
            assumptions=assumptions,
            explanation=identification.explanation
        )

    def _estimate_effect(
        self,
        query: CausalQuery,
        identification: InterventionResult,
        data: np.ndarray
    ) -> tuple[float, float]:
        """
        Estimate causal effect from data given identification strategy.

        Args:
            query: Causal query
            identification: Identification result
            data: Observational data

        Returns:
            (effect_size, confidence)
        """
        # TODO: Implement actual estimation
        # For now, return placeholder

        # This would involve:
        # 1. Extract relevant columns from data
        # 2. Apply identification formula (backdoor/frontdoor)
        # 3. Compute effect size and confidence interval
        # 4. Return estimate

        # Placeholder: return random effect
        effect = np.random.randn() * 0.3
        confidence = 0.75

        return effect, confidence

    def compute_ate(
        self,
        treatment: str,
        outcome: str,
        treatment_value: Any,
        control_value: Any,
        data: Optional[np.ndarray] = None
    ) -> CausalAnswer:
        """
        Compute Average Treatment Effect (ATE).

        ATE = E[Y|do(X=treatment_value)] - E[Y|do(X=control_value)]

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_value: Treatment condition value
            control_value: Control condition value
            data: Observational data

        Returns:
            CausalAnswer with ATE
        """
        query = CausalQuery(
            query_type=QueryType.ATE,
            outcome=outcome,
            treatment=treatment,
            treatment_value=treatment_value,
            control_value=control_value
        )

        return self.query(query, data)

    def find_all_paths(
        self,
        treatment: str,
        outcome: str,
        path_type: str = "all"
    ) -> Dict[str, List[List[str]]]:
        """
        Find all causal paths between treatment and outcome.

        Args:
            treatment: Source variable
            outcome: Target variable
            path_type: "all", "directed", "backdoor", or "frontdoor"

        Returns:
            Dict mapping path type to list of paths
        """
        result = {}

        if path_type in {"all", "directed"}:
            result["directed"] = self.dag.get_paths(treatment, outcome)

        if path_type in {"all", "backdoor"}:
            result["backdoor"] = self.dag.backdoor_paths(treatment, outcome)

        if path_type in {"all", "frontdoor"}:
            result["frontdoor"] = self.dag.frontdoor_paths(treatment, outcome)

        return result

    def explain_identification(
        self,
        treatment: str,
        outcome: str
    ) -> str:
        """
        Explain why causal effect is (or isn't) identifiable.

        Returns human-readable explanation of identification strategy.
        """
        identification = self.identify_causal_effect(treatment, outcome)

        lines = []
        lines.append(f"Causal Effect: {treatment} → {outcome}")
        lines.append("=" * 60)

        if identification.identifiable:
            lines.append(f"✓ IDENTIFIABLE via {identification.identification_method}")
            lines.append("")
            lines.append(identification.explanation)

            if identification.adjustment_set:
                lines.append("")
                lines.append(f"Adjustment set: {identification.adjustment_set}")

            # Show paths
            paths = self.find_all_paths(treatment, outcome)
            lines.append("")
            lines.append("Causal paths:")
            if paths.get("directed"):
                lines.append(f"  Directed: {len(paths['directed'])} paths")
                for i, path in enumerate(paths["directed"][:3], 1):
                    lines.append(f"    {i}. {' → '.join(path)}")
            if paths.get("backdoor"):
                lines.append(f"  Backdoor: {len(paths['backdoor'])} paths (confounding)")
                for i, path in enumerate(paths["backdoor"][:3], 1):
                    lines.append(f"    {i}. {' - '.join(path)}")

        else:
            lines.append("✗ NOT IDENTIFIABLE")
            lines.append("")
            lines.append(identification.explanation)

        return "\n".join(lines)
