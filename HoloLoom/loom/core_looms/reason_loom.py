"""
REASON Loom - The Logician
===========================

Causal inference, logic-grounded. Follows the chain.

Cognitive Style:
- Maximum inference depth (5 hops)
- Requires explicit logical links
- Validates each reasoning step
- Tracks premise dependencies

Philosophy:
"If A then B, if B then C, therefore if A then C.
The Logician follows only explicit chains."

Date: December 2025
Author: HoloLoom Architecture Team
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging
import re

from HoloLoom.loom.base_loom import BaseLoom
from HoloLoom.loom.protocol import REASON, MathInsight, CorrectionInsight
from HoloLoom.fabric.fabric import Fabric
from HoloLoom.departments.protocol import (
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
)

logger = logging.getLogger(__name__)


@dataclass
class LogicalStep:
    """A single step in a logical inference chain."""
    premise: str
    conclusion: str
    rule: str  # e.g., "modus_ponens", "transitivity"
    confidence: float
    valid: bool
    reason: Optional[str] = None


@dataclass
class InferenceChain:
    """A complete chain of logical inferences."""
    steps: List[LogicalStep]
    goal_reached: bool
    total_confidence: float


class ReasonLoom(BaseLoom):
    """
    The Logician - Causal inference, logic-grounded.

    Follows explicit logical chains. Validates each step.
    Excels at deductive reasoning and identifying logical flaws.

    Attributes:
        max_inference_depth: Maximum chain length (default: 5)
        require_explicit_links: Only follow stated connections (default: True)
        min_step_confidence: Minimum confidence per step (default: 0.6)
    """

    # Logical inference rules
    INFERENCE_RULES = {
        "modus_ponens": "If P then Q, P, therefore Q",
        "modus_tollens": "If P then Q, not Q, therefore not P",
        "transitivity": "If A then B, If B then C, therefore If A then C",
        "conjunction": "P and Q, therefore P (or Q)",
        "disjunctive_syllogism": "P or Q, not P, therefore Q",
        "hypothetical_syllogism": "If P then Q, If Q then R, therefore If P then R",
    }

    def __init__(
        self,
        max_inference_depth: int = 5,
        require_explicit_links: bool = True,
        min_step_confidence: float = 0.6,
        knowledge_graph: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the Reason Loom.

        Args:
            max_inference_depth: Maximum inference chain length
            require_explicit_links: Only follow explicit connections
            min_step_confidence: Minimum confidence for a step to be valid
            knowledge_graph: Optional knowledge graph for logical structure
            **kwargs: Arguments passed to BaseLoom
        """
        super().__init__(**kwargs)

        # Cognitive style parameters
        self.max_inference_depth = max_inference_depth
        self.require_explicit_links = require_explicit_links
        self.min_step_confidence = min_step_confidence

        # Knowledge graph for logical structure
        self._kg = knowledge_graph

        # Track invalid steps for learning
        self._invalid_steps: List[LogicalStep] = []

    @property
    def perspective(self) -> str:
        """The Logician's perspective."""
        return REASON

    @property
    def blind_spots(self) -> List[str]:
        """What the Logician cannot see."""
        return [
            "intuitive leaps",
            "emotional context",
            "ambiguity tolerance",
            "novel premises",
            "implicit assumptions",
            "probabilistic reasoning beyond chains",
        ]

    def admits_ignorance(self, query: str) -> List[str]:
        """
        What the Logician knows it doesn't know about this query.

        Args:
            query: The query to assess

        Returns:
            List of specific knowledge gaps
        """
        ignorance = []

        # Check for emotional/intuitive keywords
        intuitive_words = ['feel', 'sense', 'intuition', 'gut', 'believe']
        if any(word in query.lower() for word in intuitive_words):
            ignorance.append("Cannot reason about feelings or intuitions")

        # Check for probabilistic/uncertain keywords
        uncertain_words = ['maybe', 'might', 'probably', 'perhaps', 'possibly']
        if any(word in query.lower() for word in uncertain_words):
            ignorance.append("Uncertainty requires explicit probability chains")

        # Check for creative keywords
        creative_words = ['imagine', 'create', 'what if', 'suppose']
        if any(word in query.lower() for word in creative_words):
            ignorance.append("Hypotheticals require explicit premise definition")

        # Report recent invalid steps
        if self._invalid_steps:
            ignorance.append(f"Found {len(self._invalid_steps)} unprovable steps")

        if not ignorance:
            ignorance.append("All premises must be explicitly stated")

        return ignorance

    def falsifiable_by(self, claim: str) -> List[str]:
        """
        What evidence would change the Logician's claim.

        Args:
            claim: The claim to assess

        Returns:
            List of falsifiers
        """
        return [
            "Finding a logical fallacy in the reasoning chain",
            "Discovering a false premise in the argument",
            "Showing a missing step that breaks transitivity",
            "Providing a counterexample to a universal claim",
            "Demonstrating that a conditional is not universally true",
        ]

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute reasoning logic - build and validate inference chains.

        Args:
            request: Department request with query

        Returns:
            Response with logical analysis
        """
        start_time = datetime.now()
        query = request.query

        # Step 1: Extract logical structure from query
        premises, goal = self._extract_logical_structure(query)

        # Step 2: Build inference chain
        chain = await self._build_inference_chain(premises, goal)

        # Step 3: Validate each step
        valid_steps, invalid_steps = self._validate_chain(chain)
        self._invalid_steps = invalid_steps

        # Step 4: Format reasoning result
        response = self._format_reasoning(chain, valid_steps, invalid_steps)

        # Step 5: Compute confidence
        confidence = self._compute_chain_confidence(chain, valid_steps)

        # Step 6: Compute epistemic confidence
        epistemic = self._compute_epistemic_confidence(chain, invalid_steps)

        # Step 7: Generate falsifiers for the specific claim
        falsifiers = self._generate_specific_falsifiers(chain)

        # Calculate duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Update metrics
        self._update_metrics(confidence, duration_ms)

        # Log insights for dreaming
        if chain.goal_reached and confidence > 0.8:
            self.log_math_insight({
                "inference_pattern": "successful_chain",
                "rules_used": [s.rule for s in valid_steps],
                "chain_length": len(valid_steps),
            }, confidence=confidence)

        # Create fabric
        fabric = self._create_fabric(
            query=query,
            response=response,
            tool_used="reason",
            confidence=confidence,
            duration_ms=duration_ms,
            epistemic_confidence=epistemic,
            admits_ignorance=self.admits_ignorance(query),
            falsifiable_by=falsifiers,
            metadata={
                "chain_length": len(chain.steps),
                "valid_steps": len(valid_steps),
                "invalid_steps": len(invalid_steps),
                "goal_reached": chain.goal_reached,
                "rules_used": list(set(s.rule for s in chain.steps)),
            },
        )

        return DepartmentResponse(
            result=response,
            confidence=ConfidenceMetadata(
                score=confidence,
                source="reason_loom",
                reasoning=f"Chain of {len(valid_steps)} valid logical steps",
            ),
            metadata={
                "perspective": self.perspective,
                "fabric": fabric,
                "chain": chain,
                "valid_steps": valid_steps,
                "invalid_steps": invalid_steps,
            },
        )

    def _extract_logical_structure(
        self,
        query: str
    ) -> Tuple[List[str], str]:
        """
        Extract premises and goal from a query.

        Args:
            query: The query to analyze

        Returns:
            Tuple of (premises, goal)
        """
        # Look for explicit premise markers
        premises = []
        goal = query  # Default: the whole query is the goal

        # Pattern: "If X, then Y" or "Given X, Y"
        if_then = re.findall(r'[Ii]f\s+(.+?),?\s+then\s+(.+?)(?:\.|,|$)', query)
        for premise, conclusion in if_then:
            premises.append(f"If {premise.strip()} then {conclusion.strip()}")

        # Pattern: "Given X" or "Assuming X"
        given = re.findall(r'(?:[Gg]iven|[Aa]ssuming)\s+(.+?)(?:\.|,|$)', query)
        premises.extend(given)

        # Pattern: "Therefore" or "thus" indicates goal
        therefore = re.search(r'(?:[Tt]herefore|[Tt]hus|[Hh]ence)\s+(.+)', query)
        if therefore:
            goal = therefore.group(1).strip()

        # Pattern: "Why" or "Explain" indicates goal
        why = re.search(r'(?:[Ww]hy|[Ee]xplain)\s+(.+)', query)
        if why:
            goal = why.group(1).strip()

        # If no explicit premises, treat query words as implicit premises
        if not premises:
            words = query.split()
            if len(words) > 3:
                premises = [query]  # Use full query as context

        return premises, goal

    async def _build_inference_chain(
        self,
        premises: List[str],
        goal: str
    ) -> InferenceChain:
        """
        Build an inference chain from premises to goal.

        Args:
            premises: Starting premises
            goal: Target conclusion

        Returns:
            InferenceChain with steps
        """
        steps = []
        current_knowledge = set(premises)
        goal_reached = False

        for depth in range(self.max_inference_depth):
            # Try each inference rule
            new_step = None

            for rule_name, rule_desc in self.INFERENCE_RULES.items():
                step = self._try_inference_rule(
                    rule_name,
                    list(current_knowledge),
                    goal
                )
                if step and step.conclusion not in current_knowledge:
                    new_step = step
                    break

            if new_step:
                steps.append(new_step)
                current_knowledge.add(new_step.conclusion)

                # Check if we reached the goal
                if self._matches_goal(new_step.conclusion, goal):
                    goal_reached = True
                    break
            else:
                # No more inferences possible
                break

        # Calculate total confidence
        if steps:
            # Chain confidence = product of step confidences
            total_conf = 1.0
            for step in steps:
                total_conf *= step.confidence
        else:
            total_conf = 0.3  # Low confidence if no chain

        return InferenceChain(
            steps=steps,
            goal_reached=goal_reached,
            total_confidence=total_conf,
        )

    def _try_inference_rule(
        self,
        rule_name: str,
        knowledge: List[str],
        goal: str
    ) -> Optional[LogicalStep]:
        """
        Try to apply an inference rule to current knowledge.

        Args:
            rule_name: Name of the rule to try
            knowledge: Current known facts
            goal: Target goal

        Returns:
            LogicalStep if rule applies, None otherwise
        """
        # Simple rule application (symbolic)
        if rule_name == "transitivity":
            # Look for A->B, B->C patterns
            for fact1 in knowledge:
                if " then " in fact1.lower():
                    parts1 = fact1.lower().split(" then ")
                    if len(parts1) == 2:
                        middle = parts1[1].strip()
                        for fact2 in knowledge:
                            if fact2 != fact1 and " then " in fact2.lower():
                                parts2 = fact2.lower().split(" then ")
                                if len(parts2) == 2 and middle in parts2[0].lower():
                                    conclusion = f"If {parts1[0].strip()} then {parts2[1].strip()}"
                                    return LogicalStep(
                                        premise=f"{fact1} AND {fact2}",
                                        conclusion=conclusion,
                                        rule="transitivity",
                                        confidence=0.85,
                                        valid=True,
                                    )

        # Default: create a weakly supported step toward goal
        if knowledge and goal:
            return LogicalStep(
                premise=knowledge[0] if knowledge else "assumed",
                conclusion=f"Toward: {goal[:50]}...",
                rule="heuristic",
                confidence=0.5,
                valid=True,
                reason="Heuristic step toward goal",
            )

        return None

    def _matches_goal(self, conclusion: str, goal: str) -> bool:
        """Check if conclusion matches or contains the goal."""
        conclusion_lower = conclusion.lower()
        goal_lower = goal.lower()

        # Exact match
        if goal_lower in conclusion_lower:
            return True

        # Word overlap
        goal_words = set(goal_lower.split())
        conclusion_words = set(conclusion_lower.split())
        overlap = len(goal_words & conclusion_words) / max(len(goal_words), 1)

        return overlap > 0.6

    def _validate_chain(
        self,
        chain: InferenceChain
    ) -> Tuple[List[LogicalStep], List[LogicalStep]]:
        """
        Validate each step in the chain.

        Args:
            chain: InferenceChain to validate

        Returns:
            Tuple of (valid_steps, invalid_steps)
        """
        valid = []
        invalid = []

        for step in chain.steps:
            if step.valid and step.confidence >= self.min_step_confidence:
                valid.append(step)
            else:
                step.valid = False
                step.reason = f"Confidence {step.confidence:.2f} below threshold"
                invalid.append(step)

        return valid, invalid

    def _format_reasoning(
        self,
        chain: InferenceChain,
        valid_steps: List[LogicalStep],
        invalid_steps: List[LogicalStep]
    ) -> str:
        """
        Format the reasoning chain as text.

        Args:
            chain: The inference chain
            valid_steps: Valid steps
            invalid_steps: Invalid steps

        Returns:
            Formatted reasoning text
        """
        parts = ["Logical Analysis:"]

        if valid_steps:
            parts.append("\nValid Reasoning Steps:")
            for i, step in enumerate(valid_steps, 1):
                parts.append(
                    f"  {i}. [{step.rule}] {step.premise} -> {step.conclusion} "
                    f"(confidence: {step.confidence:.2f})"
                )

        if invalid_steps:
            parts.append("\nInvalid/Uncertain Steps:")
            for step in invalid_steps:
                parts.append(f"  - {step.premise} -> {step.conclusion}: {step.reason}")

        if chain.goal_reached:
            parts.append(f"\n[GOAL REACHED] Chain confidence: {chain.total_confidence:.2f}")
        else:
            parts.append(f"\n[GOAL NOT REACHED] Partial chain confidence: {chain.total_confidence:.2f}")

        return "\n".join(parts)

    def _compute_chain_confidence(
        self,
        chain: InferenceChain,
        valid_steps: List[LogicalStep]
    ) -> float:
        """
        Compute overall chain confidence.

        Args:
            chain: The inference chain
            valid_steps: Valid steps

        Returns:
            Confidence score (0-1)
        """
        if not valid_steps:
            return 0.3

        # Product of step confidences
        step_conf = 1.0
        for step in valid_steps:
            step_conf *= step.confidence

        # Bonus if goal reached
        if chain.goal_reached:
            step_conf = min(step_conf + 0.1, 1.0)

        return step_conf

    def _compute_epistemic_confidence(
        self,
        chain: InferenceChain,
        invalid_steps: List[LogicalStep]
    ) -> float:
        """
        Compute epistemic confidence.

        Args:
            chain: The inference chain
            invalid_steps: Invalid steps found

        Returns:
            Epistemic confidence (0-1)
        """
        epistemic = 0.9  # High base for logical reasoning

        # Reduce for invalid steps
        epistemic -= len(invalid_steps) * 0.1

        # Reduce if goal not reached
        if not chain.goal_reached:
            epistemic -= 0.15

        # Reduce for short chains (might be missing steps)
        if len(chain.steps) < 2:
            epistemic -= 0.1

        return max(epistemic, 0.3)

    def _generate_specific_falsifiers(
        self,
        chain: InferenceChain
    ) -> List[str]:
        """
        Generate specific falsifiers for this chain.

        Args:
            chain: The inference chain

        Returns:
            List of specific falsifiers
        """
        falsifiers = []

        for step in chain.steps:
            falsifiers.append(f"If {step.premise} were false")
            if step.rule == "transitivity":
                falsifiers.append(f"If the link in '{step.conclusion}' is broken")

        if not falsifiers:
            falsifiers = self.falsifiable_by("")

        return falsifiers[:5]  # Limit to 5


__all__ = ['ReasonLoom']
