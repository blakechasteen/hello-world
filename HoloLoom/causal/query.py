"""
Causal Query System

Defines queries for Pearl's three-level causal hierarchy:
1. Association (observational): P(Y|X)
2. Intervention (do-calculus): P(Y|do(X=x))
3. Counterfactual (twin networks): P(Y_x|X',Y')
"""

from dataclasses import dataclass
from typing import Dict, Set, Optional, Any
from enum import Enum


class QueryType(Enum):
    """Type of causal query."""
    # Level 1: Association (observational)
    CONDITIONAL = "conditional"         # P(Y|X) - conditional probability
    CORRELATION = "correlation"         # Correlation between variables
    ASSOCIATION = "association"         # General association queries

    # Level 2: Intervention (do-calculus)
    INTERVENTION = "intervention"       # P(Y|do(X=x)) - causal effect
    ATE = "ate"                        # Average Treatment Effect
    CATE = "cate"                      # Conditional ATE
    DIRECT_EFFECT = "direct_effect"    # Direct causal path only
    TOTAL_EFFECT = "total_effect"      # All causal paths

    # Level 3: Counterfactual (twin networks)
    COUNTERFACTUAL = "counterfactual"  # P(Y_x|X',Y') - "what if?"
    ETT = "ett"                        # Effect of Treatment on Treated
    NECESSITY = "necessity"            # Was X necessary for Y?
    SUFFICIENCY = "sufficiency"        # Was X sufficient for Y?


@dataclass
class CausalQuery:
    """
    Query for causal inference.

    Examples:
        # Level 1: What's the probability of recovery given treatment?
        CausalQuery(
            query_type=QueryType.CONDITIONAL,
            outcome="recovery",
            treatment="drug_A",
            evidence={"age": "adult"}
        )

        # Level 2: What's the causal effect of treatment on recovery?
        CausalQuery(
            query_type=QueryType.INTERVENTION,
            outcome="recovery",
            treatment="drug_A",
            treatment_value=1,
            evidence={"age": "adult"}
        )

        # Level 3: Would patient have recovered without treatment?
        CausalQuery(
            query_type=QueryType.COUNTERFACTUAL,
            outcome="recovery",
            treatment="drug_A",
            treatment_value=0,  # Counterfactual value
            evidence={"drug_A": 1, "recovery": 1}  # Observed values
        )
    """
    query_type: QueryType
    outcome: str                          # Variable of interest
    treatment: Optional[str] = None       # Intervention variable
    treatment_value: Optional[Any] = None # Value to set/intervene
    control_value: Optional[Any] = None   # Control value (for ATE)
    evidence: Dict[str, Any] = None       # Observed/conditioning variables
    confounders: Optional[Set[str]] = None  # Known confounders to adjust for
    mediators: Optional[Set[str]] = None    # Known mediators
    description: str = ""                 # Human-readable description

    def __post_init__(self):
        """Initialize evidence dict if None."""
        if self.evidence is None:
            self.evidence = {}
        if self.confounders is None:
            self.confounders = set()
        if self.mediators is None:
            self.mediators = set()

    def is_observational(self) -> bool:
        """Check if query is observational (Level 1)."""
        return self.query_type in {
            QueryType.CONDITIONAL,
            QueryType.CORRELATION,
            QueryType.ASSOCIATION
        }

    def is_interventional(self) -> bool:
        """Check if query requires do-calculus (Level 2)."""
        return self.query_type in {
            QueryType.INTERVENTION,
            QueryType.ATE,
            QueryType.CATE,
            QueryType.DIRECT_EFFECT,
            QueryType.TOTAL_EFFECT
        }

    def is_counterfactual(self) -> bool:
        """Check if query requires counterfactual reasoning (Level 3)."""
        return self.query_type in {
            QueryType.COUNTERFACTUAL,
            QueryType.ETT,
            QueryType.NECESSITY,
            QueryType.SUFFICIENCY
        }

    def get_level(self) -> int:
        """Get Pearl's causal hierarchy level (1, 2, or 3)."""
        if self.is_observational():
            return 1
        elif self.is_interventional():
            return 2
        else:
            return 3

    def to_natural_language(self) -> str:
        """
        Convert query to natural language.

        Examples:
            "What is the probability of recovery given drug_A?"
            "What is the causal effect of drug_A on recovery?"
            "Would recovery have occurred if drug_A was 0?"
        """
        if self.description:
            return self.description

        # Generate description based on query type
        if self.query_type == QueryType.CONDITIONAL:
            desc = f"What is P({self.outcome}"
            if self.treatment:
                desc += f"|{self.treatment}"
            if self.evidence:
                evidence_str = ", ".join(f"{k}={v}" for k, v in self.evidence.items())
                desc += f", {evidence_str}"
            desc += ")?"
            return desc

        elif self.query_type == QueryType.INTERVENTION:
            desc = f"What is the causal effect of {self.treatment}"
            if self.treatment_value is not None:
                desc += f"={self.treatment_value}"
            desc += f" on {self.outcome}?"
            if self.evidence:
                evidence_str = ", ".join(f"{k}={v}" for k, v in self.evidence.items())
                desc += f" (given {evidence_str})"
            return desc

        elif self.query_type == QueryType.ATE:
            desc = f"What is the average treatment effect of {self.treatment} on {self.outcome}?"
            if self.control_value is not None and self.treatment_value is not None:
                desc += f" (comparing {self.treatment}={self.treatment_value} vs {self.control_value})"
            return desc

        elif self.query_type == QueryType.COUNTERFACTUAL:
            desc = f"Would {self.outcome} have occurred if {self.treatment}={self.treatment_value}"
            if self.evidence:
                evidence_str = ", ".join(f"{k}={v}" for k, v in self.evidence.items())
                desc += f" (given observed: {evidence_str})"
            desc += "?"
            return desc

        elif self.query_type == QueryType.NECESSITY:
            desc = f"Was {self.treatment} necessary for {self.outcome}?"
            return desc

        elif self.query_type == QueryType.SUFFICIENCY:
            desc = f"Was {self.treatment} sufficient for {self.outcome}?"
            return desc

        else:
            return f"{self.query_type.value} query about {self.outcome}"

    def __repr__(self):
        return f"CausalQuery({self.query_type.value}: {self.to_natural_language()})"


@dataclass
class CausalAnswer:
    """
    Answer to causal query.

    Attributes:
        query: Original query
        result: Numerical result (probability, effect size, etc.)
        confidence: Confidence in answer (0-1)
        method: Method used to answer (e.g., "backdoor adjustment", "do-calculus")
        assumptions: Assumptions required for validity
        explanation: Human-readable explanation
        sensitivity: Sensitivity to unmeasured confounding
        metadata: Additional information
    """
    query: CausalQuery
    result: float
    confidence: float
    method: str
    assumptions: list[str] = None
    explanation: str = ""
    sensitivity: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize lists/dicts if None."""
        if self.assumptions is None:
            self.assumptions = []
        if self.metadata is None:
            self.metadata = {}

    def to_natural_language(self) -> str:
        """
        Convert answer to natural language.

        Example:
            "The causal effect of drug_A on recovery is 0.35 (35% increase).
             This estimate uses backdoor adjustment controlling for age.
             Confidence: 0.85 (high).
             Assumptions: no unmeasured confounding, correct model specification."
        """
        lines = []

        # Main result
        if self.query.query_type == QueryType.ATE:
            lines.append(f"The average treatment effect is {self.result:.3f}")
        elif self.query.query_type == QueryType.INTERVENTION:
            lines.append(f"The causal effect is {self.result:.3f}")
        elif self.query.is_counterfactual():
            lines.append(f"The counterfactual probability is {self.result:.3f}")
        else:
            lines.append(f"The result is {self.result:.3f}")

        # Method
        lines.append(f"Method: {self.method}")

        # Confidence
        confidence_level = "very low" if self.confidence < 0.3 else \
                          "low" if self.confidence < 0.5 else \
                          "medium" if self.confidence < 0.7 else \
                          "high" if self.confidence < 0.9 else "very high"
        lines.append(f"Confidence: {self.confidence:.2f} ({confidence_level})")

        # Assumptions
        if self.assumptions:
            lines.append("Assumptions:")
            for assumption in self.assumptions:
                lines.append(f"  - {assumption}")

        # Sensitivity
        if self.sensitivity is not None:
            lines.append(f"Sensitivity to unmeasured confounding: {self.sensitivity:.3f}")

        # Explanation
        if self.explanation:
            lines.append(f"\n{self.explanation}")

        return "\n".join(lines)

    def __repr__(self):
        return f"CausalAnswer(result={self.result:.3f}, confidence={self.confidence:.2f}, method={self.method})"
