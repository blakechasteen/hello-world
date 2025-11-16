"""
Debate Mode
===========
Multi-perspective reasoning for complex ethical decisions.

Implements debate-based decision making with multiple perspectives:
- Safety-first vs capability-first
- User autonomy vs societal impact
- Conservative vs progressive approaches

Based on research from:
- Irving et al. (2018): "AI Safety via Debate"
- Anthropic: Constitutional AI with multiple perspectives
- OpenAI: Deliberative alignment

**Implementation Date**: 2025-11-16
"""

import asyncio
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import re

logger = logging.getLogger("HoloLoom.alignment.debate")


class Perspective(Enum):
    """Debate perspective types.

    Each perspective prioritizes different values in decision-making.
    """
    SAFETY_FIRST = "safety_first"  # Prioritize safety over capability
    CAPABILITY_FIRST = "capability_first"  # Maximize capability
    USER_AUTONOMY = "user_autonomy"  # Respect user freedom
    SOCIETAL_IMPACT = "societal_impact"  # Consider broader impacts
    CONSERVATIVE = "conservative"  # Err on side of caution
    PROGRESSIVE = "progressive"  # Embrace innovation


@dataclass
class DebateArgument:
    """Single argument in debate.

    Represents reasoning from one perspective.
    """
    perspective: Perspective
    claim: str
    supporting_evidence: List[str]
    counter_arguments: List[str]
    confidence: float  # 0.0-1.0
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "perspective": self.perspective.value,
            "claim": self.claim,
            "supporting_evidence": self.supporting_evidence,
            "counter_arguments": self.counter_arguments,
            "confidence": self.confidence,
            "risk_assessment": self.risk_assessment,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DebateResult:
    """Result of multi-perspective debate.

    Contains all arguments, consensus (if reached), and recommended action.
    """
    question: str
    arguments: List[DebateArgument]
    consensus: Optional[str]
    consensus_confidence: float
    dissenting_perspectives: List[Perspective]
    recommended_action: str
    safety_score: float  # 0.0-1.0
    reasoning_trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "question": self.question,
            "arguments": [arg.to_dict() for arg in self.arguments],
            "consensus": self.consensus,
            "consensus_confidence": self.consensus_confidence,
            "dissenting_perspectives": [p.value for p in self.dissenting_perspectives],
            "recommended_action": self.recommended_action,
            "safety_score": self.safety_score,
            "reasoning_trace": self.reasoning_trace,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Question: {self.question}",
            f"Perspectives: {len(self.arguments)}",
            f"Consensus: {self.consensus or 'None reached'}",
            f"Confidence: {self.consensus_confidence:.2%}",
            f"Dissent: {len(self.dissenting_perspectives)} perspectives",
            f"Safety Score: {self.safety_score:.2%}",
            f"Action: {self.recommended_action}",
        ]
        return "\n".join(lines)


class DebateMode:
    """Multi-perspective reasoning for complex ethical decisions.

    Conducts debates between different value perspectives to reach
    well-reasoned decisions on complex or ambiguous questions.

    Example:
        debate = DebateMode()
        result = await debate.debate(
            question="Should system execute user's code?",
            context={"code": "import os; os.system('rm -rf /')"}
        )

        # Result contains arguments from all perspectives
        # SAFETY_FIRST: "No, catastrophic risk"
        # CAPABILITY_FIRST: "Yes, if user explicitly requested"
        # USER_AUTONOMY: "User has right, but warn first"

        # Consensus: "No, but offer safer alternative"
    """

    def __init__(
        self,
        perspectives: Optional[List[Perspective]] = None,
        min_consensus_confidence: float = 0.7,
        enable_counter_arguments: bool = True,
        max_debate_rounds: int = 3,
    ):
        """Initialize debate mode.

        Args:
            perspectives: List of perspectives to include (default: all)
            min_consensus_confidence: Minimum confidence for consensus
            enable_counter_arguments: Whether to generate counter-arguments
            max_debate_rounds: Maximum rounds of argumentation
        """
        self.perspectives = perspectives or [
            Perspective.SAFETY_FIRST,
            Perspective.CAPABILITY_FIRST,
            Perspective.USER_AUTONOMY,
            Perspective.SOCIETAL_IMPACT,
            Perspective.CONSERVATIVE,
            Perspective.PROGRESSIVE,
        ]
        self.min_consensus_confidence = min_consensus_confidence
        self.enable_counter_arguments = enable_counter_arguments
        self.max_debate_rounds = max_debate_rounds
        self.debate_history: List[DebateResult] = []

        logger.info(
            f"Initialized DebateMode with {len(self.perspectives)} perspectives, "
            f"min_consensus={min_consensus_confidence:.1%}"
        )

    async def debate(
        self,
        question: str,
        context: Dict[str, Any],
        required_perspectives: Optional[List[Perspective]] = None,
    ) -> DebateResult:
        """Conduct multi-perspective debate.

        Args:
            question: Question to debate
            context: Context dictionary with relevant information
            required_perspectives: Optional subset of perspectives to use

        Returns:
            DebateResult with arguments, consensus, and recommendation

        Example:
            result = await debate.debate(
                question="Should system modify user files?",
                context={
                    "action": "delete_old_logs",
                    "user_request": "clean up my system",
                    "risk_level": "medium"
                }
            )
        """
        logger.info(f"Starting debate on: {question}")
        reasoning_trace = [f"Question: {question}"]

        # Select perspectives
        active_perspectives = required_perspectives or self.perspectives
        reasoning_trace.append(f"Active perspectives: {[p.value for p in active_perspectives]}")

        # Round 1: Generate initial arguments
        arguments = []
        for perspective in active_perspectives:
            arg = await self._generate_argument(
                perspective,
                question,
                context,
                round_num=1,
            )
            arguments.append(arg)
            reasoning_trace.append(
                f"{perspective.value}: {arg.claim} (confidence={arg.confidence:.2f})"
            )

        # Additional rounds: Counter-arguments and refinement
        if self.enable_counter_arguments and self.max_debate_rounds > 1:
            for round_num in range(2, self.max_debate_rounds + 1):
                reasoning_trace.append(f"\n--- Round {round_num} ---")
                arguments = await self._debate_round(
                    arguments,
                    question,
                    context,
                    round_num,
                    reasoning_trace,
                )

        # Find consensus
        consensus, confidence = await self._find_consensus(arguments, reasoning_trace)

        # Identify dissent
        dissenting = self._identify_dissent(arguments, consensus)
        if dissenting:
            reasoning_trace.append(
                f"Dissenting perspectives: {[p.value for p in dissenting]}"
            )

        # Recommend action
        action = await self._recommend_action(
            arguments,
            consensus,
            dissenting,
            context,
            reasoning_trace,
        )

        # Calculate safety score
        safety_score = self._calculate_safety_score(arguments, action, context)
        reasoning_trace.append(f"Safety score: {safety_score:.2%}")

        result = DebateResult(
            question=question,
            arguments=arguments,
            consensus=consensus,
            consensus_confidence=confidence,
            dissenting_perspectives=dissenting,
            recommended_action=action,
            safety_score=safety_score,
            reasoning_trace=reasoning_trace,
            metadata={
                "perspectives_count": len(arguments),
                "debate_rounds": self.max_debate_rounds,
                "context": context,
            },
        )

        self.debate_history.append(result)
        logger.info(
            f"Debate complete: consensus_confidence={confidence:.2%}, "
            f"safety_score={safety_score:.2%}"
        )

        return result

    async def _generate_argument(
        self,
        perspective: Perspective,
        question: str,
        context: Dict[str, Any],
        round_num: int = 1,
        previous_arguments: Optional[List[DebateArgument]] = None,
    ) -> DebateArgument:
        """Generate argument from specific perspective.

        Uses perspective-specific reasoning to analyze question.
        """
        # Build perspective-specific reasoning
        claim, evidence, confidence = self._reason_from_perspective(
            perspective,
            question,
            context,
        )

        # Generate counter-arguments if enabled and round > 1
        counter_args = []
        if self.enable_counter_arguments and previous_arguments and round_num > 1:
            counter_args = self._generate_counter_arguments(
                perspective,
                claim,
                previous_arguments,
            )

        # Risk assessment
        risk_assessment = self._assess_risks(perspective, question, context)

        return DebateArgument(
            perspective=perspective,
            claim=claim,
            supporting_evidence=evidence,
            counter_arguments=counter_args,
            confidence=confidence,
            risk_assessment=risk_assessment,
            metadata={
                "round": round_num,
                "question": question,
            },
        )

    def _reason_from_perspective(
        self,
        perspective: Perspective,
        question: str,
        context: Dict[str, Any],
    ) -> Tuple[str, List[str], float]:
        """Generate reasoning from specific perspective.

        Returns: (claim, evidence, confidence)
        """
        action = context.get("action", "")
        risk_level = context.get("risk_level", "unknown")
        user_request = context.get("user_request", "")

        if perspective == Perspective.SAFETY_FIRST:
            # Prioritize safety above all
            if risk_level in ["high", "critical"]:
                return (
                    "Action should be blocked due to high safety risk",
                    [
                        f"Risk level is {risk_level}",
                        "Safety must be prioritized over capability",
                        "Potential for irreversible harm",
                    ],
                    0.9,
                )
            elif risk_level == "medium":
                return (
                    "Action requires additional safeguards before proceeding",
                    [
                        f"Risk level is {risk_level}",
                        "Need confirmation and reversibility measures",
                    ],
                    0.7,
                )
            else:
                return (
                    "Action appears safe to proceed",
                    ["Risk level is low", "Standard safety checks sufficient"],
                    0.8,
                )

        elif perspective == Perspective.CAPABILITY_FIRST:
            # Maximize system capability
            if user_request:
                return (
                    "Action should proceed to fulfill user's request",
                    [
                        f"User explicitly requested: {user_request}",
                        "System should be capable and helpful",
                        "Capability enables user goals",
                    ],
                    0.8,
                )
            else:
                return (
                    "Action unclear without explicit user request",
                    ["Need clear user intention", "Capability without purpose is risky"],
                    0.5,
                )

        elif perspective == Perspective.USER_AUTONOMY:
            # Respect user's freedom of choice
            if user_request:
                return (
                    "User has autonomy to make this decision",
                    [
                        "User's explicit request should be honored",
                        "Autonomy requires informed consent",
                        "System should enable, not restrict",
                    ],
                    0.8,
                )
            else:
                return (
                    "Cannot determine user's autonomous choice",
                    ["Need explicit user consent", "Implied consent insufficient"],
                    0.4,
                )

        elif perspective == Perspective.SOCIETAL_IMPACT:
            # Consider broader implications
            harmful_patterns = ["delete", "remove", "hack", "exploit", "abuse"]
            if any(pattern in action.lower() for pattern in harmful_patterns):
                return (
                    "Action has potential negative societal impact",
                    [
                        "Action pattern suggests potential harm",
                        "Broader implications beyond individual user",
                        "Sets precedent for acceptable behavior",
                    ],
                    0.7,
                )
            else:
                return (
                    "Action has neutral or positive societal impact",
                    [
                        "No obvious harmful patterns detected",
                        "Individual use case acceptable",
                    ],
                    0.6,
                )

        elif perspective == Perspective.CONSERVATIVE:
            # Err on side of caution
            if risk_level in ["medium", "high", "critical"]:
                return (
                    "Should not proceed without extensive validation",
                    [
                        "Conservative approach prioritizes caution",
                        f"Risk level {risk_level} warrants restraint",
                        "Better safe than sorry",
                    ],
                    0.8,
                )
            else:
                return (
                    "Can proceed with appropriate monitoring",
                    [
                        "Low risk allows careful progress",
                        "Conservative doesn't mean paralyzed",
                    ],
                    0.7,
                )

        elif perspective == Perspective.PROGRESSIVE:
            # Embrace innovation and progress
            if risk_level != "critical":
                return (
                    "Should proceed with appropriate safeguards",
                    [
                        "Innovation requires calculated risks",
                        "Progress comes from trying new things",
                        "Can mitigate risks while advancing",
                    ],
                    0.75,
                )
            else:
                return (
                    "Even progressive approach requires pause at critical risk",
                    [
                        "Critical risk exceeds progressive tolerance",
                        "Innovation requires sustainable approach",
                    ],
                    0.6,
                )

        # Fallback
        return (
            f"Perspective {perspective.value} suggests careful consideration",
            ["Analysis incomplete", "Need more context"],
            0.5,
        )

    def _generate_counter_arguments(
        self,
        perspective: Perspective,
        claim: str,
        other_arguments: List[DebateArgument],
    ) -> List[str]:
        """Generate counter-arguments against opposing claims."""
        counter_args = []

        for arg in other_arguments:
            if arg.perspective == perspective:
                continue  # Don't counter own argument

            # Generate perspective-specific counters
            if perspective == Perspective.SAFETY_FIRST:
                if "proceed" in arg.claim.lower():
                    counter_args.append(
                        f"{arg.perspective.value} underestimates safety risks"
                    )

            elif perspective == Perspective.CAPABILITY_FIRST:
                if "block" in arg.claim.lower():
                    counter_args.append(
                        f"{arg.perspective.value} over-restricts system capability"
                    )

            elif perspective == Perspective.USER_AUTONOMY:
                if "should not" in arg.claim.lower():
                    counter_args.append(
                        f"{arg.perspective.value} paternalistically restricts user choice"
                    )

        return counter_args[:3]  # Limit to top 3

    def _assess_risks(
        self,
        perspective: Perspective,
        question: str,
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Assess risks from perspective."""
        risk_level = context.get("risk_level", "unknown")

        # Risk mapping
        risk_map = {
            "safe": 0.1,
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95,
            "unknown": 0.5,
        }

        base_risk = risk_map.get(risk_level, 0.5)

        # Perspective-specific risk weights
        if perspective == Perspective.SAFETY_FIRST:
            safety_risk = base_risk * 1.2  # Amplify risk perception
        elif perspective == Perspective.PROGRESSIVE:
            safety_risk = base_risk * 0.8  # Attenuate risk perception
        else:
            safety_risk = base_risk

        return {
            "safety": min(1.0, safety_risk),
            "capability": 1.0 - safety_risk,  # Inverse relationship
            "autonomy": 0.5,  # Neutral
        }

    async def _debate_round(
        self,
        previous_arguments: List[DebateArgument],
        question: str,
        context: Dict[str, Any],
        round_num: int,
        reasoning_trace: List[str],
    ) -> List[DebateArgument]:
        """Conduct one round of debate with counter-arguments."""
        new_arguments = []

        for arg in previous_arguments:
            # Refine argument based on others' claims
            refined_arg = await self._generate_argument(
                arg.perspective,
                question,
                context,
                round_num=round_num,
                previous_arguments=previous_arguments,
            )
            new_arguments.append(refined_arg)

            reasoning_trace.append(
                f"{refined_arg.perspective.value}: {refined_arg.claim} "
                f"(counters: {len(refined_arg.counter_arguments)})"
            )

        return new_arguments

    async def _find_consensus(
        self,
        arguments: List[DebateArgument],
        reasoning_trace: List[str],
    ) -> Tuple[Optional[str], float]:
        """Find consensus among arguments.

        Returns: (consensus_statement, confidence)
        """
        reasoning_trace.append("\n--- Finding Consensus ---")

        # Extract claims
        claims = [arg.claim.lower() for arg in arguments]

        # Check for strong agreement
        proceed_count = sum(1 for c in claims if "proceed" in c or "should" in c and "not" not in c)
        block_count = sum(1 for c in claims if "block" in c or "should not" in c)
        caution_count = sum(1 for c in claims if "safeguard" in c or "caution" in c or "require" in c)

        total = len(arguments)

        reasoning_trace.append(
            f"Votes: proceed={proceed_count}, block={block_count}, caution={caution_count}"
        )

        # Determine consensus
        if block_count / total >= 0.6:
            return ("Action should be blocked", block_count / total)
        elif proceed_count / total >= 0.6:
            return ("Action should proceed", proceed_count / total)
        elif caution_count / total >= 0.4:
            return ("Action requires additional safeguards", caution_count / total)
        else:
            # No clear consensus
            return (None, 0.0)

    def _identify_dissent(
        self,
        arguments: List[DebateArgument],
        consensus: Optional[str],
    ) -> List[Perspective]:
        """Identify perspectives that dissent from consensus."""
        if not consensus:
            return []  # No consensus, so no dissent

        dissenting = []
        consensus_lower = consensus.lower()

        for arg in arguments:
            claim_lower = arg.claim.lower()

            # Check if claim contradicts consensus
            if "block" in consensus_lower and "proceed" in claim_lower:
                dissenting.append(arg.perspective)
            elif "proceed" in consensus_lower and "block" in claim_lower:
                dissenting.append(arg.perspective)
            elif "safeguard" in consensus_lower and ("block" in claim_lower or arg.confidence < 0.5):
                dissenting.append(arg.perspective)

        return dissenting

    async def _recommend_action(
        self,
        arguments: List[DebateArgument],
        consensus: Optional[str],
        dissenting: List[Perspective],
        context: Dict[str, Any],
        reasoning_trace: List[str],
    ) -> str:
        """Recommend action based on debate outcome."""
        reasoning_trace.append("\n--- Recommending Action ---")

        # If strong consensus and low dissent, follow it
        if consensus and len(dissenting) <= 1:
            action = consensus.replace("Action should", "").strip()
            reasoning_trace.append(f"Following consensus: {action}")
            return action.capitalize()

        # If no consensus or high dissent, be conservative
        if not consensus or len(dissenting) >= 3:
            reasoning_trace.append("No clear consensus - defaulting to conservative approach")
            return "Require human review before proceeding"

        # Middle ground: consensus exists but with some dissent
        risk_level = context.get("risk_level", "unknown")
        if risk_level in ["high", "critical"]:
            reasoning_trace.append("High risk with dissent - requiring review")
            return "Block action pending human review"
        else:
            reasoning_trace.append("Moderate risk with dissent - proceed with caution")
            return "Proceed with enhanced monitoring and safeguards"

    def _calculate_safety_score(
        self,
        arguments: List[DebateArgument],
        action: str,
        context: Dict[str, Any],
    ) -> float:
        """Calculate overall safety score (0.0-1.0).

        Higher score = safer action.
        """
        # Start with average confidence
        avg_confidence = sum(arg.confidence for arg in arguments) / len(arguments)

        # Adjust based on action type
        action_lower = action.lower()
        if "block" in action_lower or "review" in action_lower:
            safety_bonus = 0.2  # Conservative actions are safer
        elif "proceed" in action_lower and "safeguard" not in action_lower:
            safety_bonus = -0.1  # Proceeding without safeguards less safe
        else:
            safety_bonus = 0.0

        # Adjust based on risk assessments
        avg_safety_risk = sum(
            arg.risk_assessment.get("safety", 0.5)
            for arg in arguments
        ) / len(arguments)

        # Combine factors
        safety_score = (
            avg_confidence * 0.4 +
            (1.0 - avg_safety_risk) * 0.4 +
            (0.5 + safety_bonus) * 0.2
        )

        return max(0.0, min(1.0, safety_score))

    def get_debate_history(self) -> List[DebateResult]:
        """Get all debate results."""
        return self.debate_history

    def get_statistics(self) -> Dict[str, Any]:
        """Get debate statistics."""
        if not self.debate_history:
            return {
                "total_debates": 0,
                "avg_consensus_confidence": 0.0,
                "avg_safety_score": 0.0,
                "consensus_rate": 0.0,
            }

        total = len(self.debate_history)
        consensuses_reached = sum(
            1 for r in self.debate_history if r.consensus is not None
        )

        return {
            "total_debates": total,
            "avg_consensus_confidence": sum(
                r.consensus_confidence for r in self.debate_history
            ) / total,
            "avg_safety_score": sum(
                r.safety_score for r in self.debate_history
            ) / total,
            "consensus_rate": consensuses_reached / total,
            "avg_dissent": sum(
                len(r.dissenting_perspectives) for r in self.debate_history
            ) / total,
        }
