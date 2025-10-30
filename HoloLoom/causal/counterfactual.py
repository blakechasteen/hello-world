"""
Counterfactual Engine - Level 3 Causal Reasoning

Implements Pearl's twin network method for counterfactual inference.

Key concepts:
- Counterfactual: "What would Y be if X had been x?" (even though X=x' was observed)
- Twin networks: Run two worlds in parallel (factual + counterfactual)
- Abduction: Infer latent variables from observations
- Action: Intervene on counterfactual world
- Prediction: Compute outcome in counterfactual world

Three steps of counterfactual inference:
1. Abduction: P(U|E) - infer unobserved factors from evidence
2. Action: do(X=x) - intervene in counterfactual world
3. Prediction: P(Y|U, do(X=x)) - compute counterfactual outcome
"""

from typing import Dict, Any, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass

from .dag import CausalDAG, CausalNode, NodeType
from .query import CausalQuery, CausalAnswer, QueryType
from .intervention import InterventionEngine


@dataclass
class TwinNetwork:
    """
    Twin network for counterfactual reasoning.

    Contains two versions of each variable:
    - Factual world: X, Y (observed/actual values)
    - Counterfactual world: X', Y' (alternative values)

    Both worlds share the same exogenous variables U (hidden causes).
    """
    factual_dag: CausalDAG      # Original DAG
    counterfactual_dag: CausalDAG  # Mutilated DAG after intervention
    shared_exogenous: Set[str]  # Shared U variables
    factual_values: Dict[str, Any]  # Observed values
    counterfactual_values: Dict[str, Any]  # Alternative values
    abduced_exogenous: Dict[str, Any]  # Inferred U from observations


@dataclass
class CounterfactualResult:
    """
    Result of counterfactual inference.

    Attributes:
        factual_outcome: What actually happened (Y)
        counterfactual_outcome: What would have happened (Y')
        probability: P(Y'|X', E) - counterfactual probability
        necessity: Probability that X was necessary for Y
        sufficiency: Probability that X was sufficient for Y
        explanation: Human-readable explanation
    """
    factual_outcome: Any
    counterfactual_outcome: Any
    probability: float
    necessity: Optional[float] = None
    sufficiency: Optional[float] = None
    explanation: str = ""
    twin_network: Optional[TwinNetwork] = None


class CounterfactualEngine:
    """
    Implements counterfactual reasoning via twin networks.

    Usage:
        engine = CounterfactualEngine(dag)

        # "Would patient have recovered without treatment?"
        result = engine.counterfactual(
            intervention={"treatment": 0},  # What we change
            evidence={"treatment": 1, "recovery": 1},  # What we observed
            query="recovery"  # What we ask about
        )

        # Necessity: Was treatment necessary for recovery?
        necessity = engine.probability_of_necessity(
            treatment="drug_A",
            outcome="recovery",
            evidence={"drug_A": 1, "recovery": 1}
        )

        # Sufficiency: Is treatment sufficient for recovery?
        sufficiency = engine.probability_of_sufficiency(
            treatment="drug_A",
            outcome="recovery",
            evidence={"drug_A": 0, "recovery": 0}
        )
    """

    def __init__(self, dag: CausalDAG):
        """
        Initialize counterfactual engine.

        Args:
            dag: Causal DAG with structural equations
        """
        self.dag = dag
        self.intervention_engine = InterventionEngine(dag)

    def counterfactual(
        self,
        intervention: Dict[str, Any],
        evidence: Dict[str, Any],
        query: str
    ) -> CounterfactualResult:
        """
        Compute counterfactual: P(Y_x | E)

        "What would Y be if X had been x, given that we observed E?"

        Three-step process:
        1. Abduction: Infer U from evidence E
        2. Action: Apply intervention do(X=x)
        3. Prediction: Compute Y under intervention given U

        Args:
            intervention: Variables to intervene on {X: x}
            evidence: Observed values {X': x', Y': y'}
            query: Variable to query (Y)

        Returns:
            CounterfactualResult with outcome and probability
        """
        # Step 1: Abduction - Infer exogenous variables from evidence
        abduced_exogenous = self._abduction(evidence)

        # Step 2: Action - Apply intervention in counterfactual world
        intervention_result = self.intervention_engine.do(intervention)

        # Step 3: Prediction - Compute outcome given U and intervention
        counterfactual_value = self._prediction(
            query=query,
            exogenous=abduced_exogenous,
            dag=intervention_result.mutilated_graph
        )

        # Get factual value
        factual_value = evidence.get(query, None)

        # Compute probability (placeholder - would need structural equations)
        probability = self._compute_probability(
            query=query,
            value=counterfactual_value,
            exogenous=abduced_exogenous,
            dag=intervention_result.mutilated_graph
        )

        # Create twin network
        twin = TwinNetwork(
            factual_dag=self.dag,
            counterfactual_dag=intervention_result.mutilated_graph,
            shared_exogenous=set(abduced_exogenous.keys()),
            factual_values=evidence,
            counterfactual_values={**intervention, query: counterfactual_value},
            abduced_exogenous=abduced_exogenous
        )

        # Generate explanation
        explanation = self._explain_counterfactual(
            intervention=intervention,
            evidence=evidence,
            query=query,
            counterfactual_value=counterfactual_value,
            factual_value=factual_value
        )

        return CounterfactualResult(
            factual_outcome=factual_value,
            counterfactual_outcome=counterfactual_value,
            probability=probability,
            explanation=explanation,
            twin_network=twin
        )

    def _abduction(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abduction step: Infer exogenous variables U from evidence E.

        Given observations, work backwards to infer hidden causes.

        Args:
            evidence: Observed variable values

        Returns:
            Inferred exogenous variables
        """
        # TODO: Implement proper abduction
        # This requires:
        # 1. Structural equations for each variable
        # 2. Solving equations backwards to get U from observed values
        # 3. Handling noise/error terms

        # Placeholder: Create dummy exogenous variables
        exogenous = {}

        # For each observed variable, create corresponding U
        for var, value in evidence.items():
            # In real implementation, solve structural equation:
            # Y = f(parents, U_Y)  =>  U_Y = f_inverse(Y, parents)

            # Handle numeric vs categorical values
            if isinstance(value, (int, float)):
                # Numeric: add noise
                exogenous[f"U_{var}"] = value + np.random.randn() * 0.1
            else:
                # Categorical: store hash
                exogenous[f"U_{var}"] = hash(str(value)) % 1000 / 1000.0

        return exogenous

    def _prediction(
        self,
        query: str,
        exogenous: Dict[str, Any],
        dag: CausalDAG
    ) -> Any:
        """
        Prediction step: Compute outcome given U and mutilated DAG.

        Args:
            query: Variable to compute
            exogenous: Inferred exogenous variables
            dag: Mutilated DAG after intervention

        Returns:
            Predicted value of query variable
        """
        # TODO: Implement proper prediction
        # This requires:
        # 1. Topological ordering of variables
        # 2. Forward sampling through structural equations
        # 3. Using abduced U values

        # Placeholder: Return random value
        # In real implementation:
        # - Order nodes topologically
        # - For each node, compute value from parents + U
        # - Return value of query node

        return np.random.choice([0, 1])  # Binary outcome

    def _compute_probability(
        self,
        query: str,
        value: Any,
        exogenous: Dict[str, Any],
        dag: CausalDAG
    ) -> float:
        """
        Compute probability of counterfactual outcome.

        Args:
            query: Variable being queried
            value: Counterfactual value
            exogenous: Abduced exogenous variables
            dag: Mutilated DAG

        Returns:
            Probability (0-1)
        """
        # TODO: Implement proper probability computation
        # Would require marginalizing over uncertainty in U

        # Placeholder: Return high probability if deterministic, lower if noisy
        return 0.85

    def probability_of_necessity(
        self,
        treatment: str,
        outcome: str,
        evidence: Dict[str, Any]
    ) -> float:
        """
        Compute Probability of Necessity (PN).

        PN = P(Y_x=0 = 0 | X=1, Y=1)

        "Was treatment necessary for outcome?"
        Given that we had treatment (X=1) and outcome (Y=1),
        what's the probability outcome wouldn't have occurred without treatment?

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            evidence: Must include {treatment: 1, outcome: 1}

        Returns:
            Probability (0-1) that treatment was necessary
        """
        if evidence.get(treatment) != 1 or evidence.get(outcome) != 1:
            raise ValueError("PN requires evidence with treatment=1 and outcome=1")

        # Counterfactual: What if treatment had been 0?
        result = self.counterfactual(
            intervention={treatment: 0},
            evidence=evidence,
            query=outcome
        )

        # PN = P(Y_0 = 0 | X=1, Y=1)
        # If counterfactual outcome is 0, treatment was necessary
        if result.counterfactual_outcome == 0:
            necessity = result.probability
        else:
            necessity = 1.0 - result.probability

        return necessity

    def probability_of_sufficiency(
        self,
        treatment: str,
        outcome: str,
        evidence: Dict[str, Any]
    ) -> float:
        """
        Compute Probability of Sufficiency (PS).

        PS = P(Y_x=1 = 1 | X=0, Y=0)

        "Is treatment sufficient for outcome?"
        Given that we didn't have treatment (X=0) and no outcome (Y=0),
        what's the probability outcome would have occurred with treatment?

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            evidence: Must include {treatment: 0, outcome: 0}

        Returns:
            Probability (0-1) that treatment is sufficient
        """
        if evidence.get(treatment) != 0 or evidence.get(outcome) != 0:
            raise ValueError("PS requires evidence with treatment=0 and outcome=0")

        # Counterfactual: What if treatment had been 1?
        result = self.counterfactual(
            intervention={treatment: 1},
            evidence=evidence,
            query=outcome
        )

        # PS = P(Y_1 = 1 | X=0, Y=0)
        # If counterfactual outcome is 1, treatment is sufficient
        if result.counterfactual_outcome == 1:
            sufficiency = result.probability
        else:
            sufficiency = 1.0 - result.probability

        return sufficiency

    def probability_of_necessity_and_sufficiency(
        self,
        treatment: str,
        outcome: str
    ) -> float:
        """
        Compute Probability of Necessity and Sufficiency (PNS).

        PNS = P(Y_x=1 = 1 and Y_x=0 = 0)

        "Is treatment both necessary AND sufficient for outcome?"

        This is the strongest causal claim:
        - Outcome occurs if and only if treatment occurs
        - Treatment is both necessary and sufficient

        Returns:
            Probability (0-1)
        """
        # PNS requires computing joint probability over twin network
        # TODO: Implement full joint computation

        # Approximation: min(PN, PS) is lower bound
        # (actual PNS could be higher due to correlation)

        # For placeholder, return conservative estimate
        return 0.5

    def _explain_counterfactual(
        self,
        intervention: Dict[str, Any],
        evidence: Dict[str, Any],
        query: str,
        counterfactual_value: Any,
        factual_value: Any
    ) -> str:
        """Generate human-readable explanation of counterfactual."""
        lines = []

        lines.append("Counterfactual Analysis")
        lines.append("=" * 60)

        # What actually happened
        lines.append("\nFactual world (what actually happened):")
        for var, val in evidence.items():
            lines.append(f"  {var} = {val}")

        # What we're asking
        lines.append("\nCounterfactual question:")
        intervention_str = ", ".join(f"{k}={v}" for k, v in intervention.items())
        lines.append(f"  What if {intervention_str}?")

        # Answer
        lines.append("\nCounterfactual outcome:")
        lines.append(f"  {query} = {counterfactual_value}")

        # Comparison
        if factual_value is not None:
            lines.append("\nComparison:")
            if counterfactual_value == factual_value:
                lines.append(f"  Outcome unchanged: {query} = {factual_value} in both worlds")
            else:
                lines.append(f"  Outcome changed: {query} = {factual_value} → {counterfactual_value}")

        return "\n".join(lines)

    def query(
        self,
        query: CausalQuery,
        data: Optional[np.ndarray] = None
    ) -> CausalAnswer:
        """
        Answer counterfactual query.

        Args:
            query: Counterfactual query
            data: Optional data (not used for counterfactuals)

        Returns:
            CausalAnswer
        """
        if not query.is_counterfactual():
            raise ValueError(f"CounterfactualEngine handles counterfactual queries, got {query.query_type}")

        if query.query_type == QueryType.COUNTERFACTUAL:
            # Standard counterfactual
            result = self.counterfactual(
                intervention={query.treatment: query.treatment_value},
                evidence=query.evidence,
                query=query.outcome
            )

            return CausalAnswer(
                query=query,
                result=result.probability,
                confidence=0.7,  # Counterfactuals are inherently uncertain
                method="twin networks (3-step counterfactual)",
                assumptions=[
                    "Correct structural equations",
                    "Successful abduction of exogenous variables",
                    "Deterministic or well-specified noise model"
                ],
                explanation=result.explanation
            )

        elif query.query_type == QueryType.NECESSITY:
            # Probability of necessity
            necessity = self.probability_of_necessity(
                treatment=query.treatment,
                outcome=query.outcome,
                evidence=query.evidence
            )

            return CausalAnswer(
                query=query,
                result=necessity,
                confidence=0.7,
                method="probability of necessity",
                assumptions=[
                    f"{query.treatment}=1 and {query.outcome}=1 observed",
                    "Correct causal model"
                ],
                explanation=f"Probability that {query.treatment} was necessary for {query.outcome}: {necessity:.3f}"
            )

        elif query.query_type == QueryType.SUFFICIENCY:
            # Probability of sufficiency
            sufficiency = self.probability_of_sufficiency(
                treatment=query.treatment,
                outcome=query.outcome,
                evidence=query.evidence
            )

            return CausalAnswer(
                query=query,
                result=sufficiency,
                confidence=0.7,
                method="probability of sufficiency",
                assumptions=[
                    f"{query.treatment}=0 and {query.outcome}=0 observed",
                    "Correct causal model"
                ],
                explanation=f"Probability that {query.treatment} is sufficient for {query.outcome}: {sufficiency:.3f}"
            )

        else:
            raise ValueError(f"Unsupported counterfactual query type: {query.query_type}")

    def explain_counterfactual(
        self,
        treatment: str,
        outcome: str,
        evidence: Dict[str, Any],
        intervention_value: Any
    ) -> str:
        """
        Explain counterfactual reasoning in natural language.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            evidence: Observed values
            intervention_value: Counterfactual treatment value

        Returns:
            Human-readable explanation
        """
        result = self.counterfactual(
            intervention={treatment: intervention_value},
            evidence=evidence,
            query=outcome
        )

        return result.explanation
