"""
Constitutional Self-Critique
============================
Two-phase Constitutional AI implementation for self-improving alignment.

Implements Anthropic's Constitutional AI approach:
- Phase 1 (SL-CAI): Supervised learning from self-critique
- Phase 2 (RLAIF): Reinforcement learning from AI feedback

Based on research from:
- Bai et al. (2022): "Constitutional AI: Harmlessness from AI Feedback"
- Anthropic: "Training a Helpful and Harmless Assistant"

Key Features:
- Configurable constitutional principles
- Multi-round self-critique and revision
- Principle violation detection
- RLAIF preference learning
- Continuous value alignment verification
"""

import logging
import asyncio
import random
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Callable, Awaitable, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime
import re
import hashlib

logger = logging.getLogger("HoloLoom.alignment.constitutional_critique")


# =============================================================================
# Phase 3 Refinement: Protocol Definitions for Extensibility (December 2025)
# =============================================================================


@runtime_checkable
class CritiqueEngineProtocol(Protocol):
    """
    Protocol for custom critique engine implementations.

    Enables plugging in different critique strategies:
    - LLM-based critique (production)
    - Heuristic-based critique (current default)
    - Hybrid critique (combining multiple signals)
    - Domain-specific critique (specialized for certain domains)

    December 2025: Added for Phase 3 extensibility.
    """

    def critique_response(
        self,
        prompt: str,
        response: str,
        principles: Optional[List["ConstitutionalPrinciple"]] = None,
    ) -> List["CritiqueResult"]:
        """
        Critique a response against constitutional principles.

        Args:
            prompt: Original user prompt
            response: Agent's response to critique
            principles: Specific principles to check, or None for all

        Returns:
            List of CritiqueResult for each principle checked
        """
        ...


@runtime_checkable
class RevisionEngineProtocol(Protocol):
    """
    Protocol for custom revision engine implementations.

    Enables different revision strategies:
    - LLM-based revision (production)
    - Template-based revision (fast, deterministic)
    - Iterative refinement (multiple passes)
    - Minimal intervention (smallest change that fixes violation)

    December 2025: Added for Phase 3 extensibility.
    """

    async def revise(
        self,
        prompt: str,
        response: str,
        critiques: List["CritiqueResult"],
    ) -> str:
        """
        Revise a response based on critique feedback.

        Args:
            prompt: Original user prompt
            response: Response to revise
            critiques: Critique results identifying issues

        Returns:
            Revised response addressing the critiques
        """
        ...


@runtime_checkable
class PrincipleProviderProtocol(Protocol):
    """
    Protocol for dynamic principle providers.

    Enables loading principles from various sources:
    - Static configuration (current default)
    - Database/remote service
    - User-specific principle sets
    - Context-dependent principles

    December 2025: Added for Phase 3 extensibility.
    """

    def get_principles(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> List["ConstitutionalPrinciple"]:
        """
        Get applicable principles for a given context.

        Args:
            context: Optional context for principle selection

        Returns:
            List of applicable constitutional principles
        """
        ...


@runtime_checkable
class PreferenceModelProtocol(Protocol):
    """
    Protocol for custom preference scoring models.

    Enables different preference learning approaches:
    - Rule-based scoring (current default)
    - Neural preference model
    - Human-in-the-loop preferences
    - Bradley-Terry model

    December 2025: Added for Phase 3 extensibility.
    """

    def compute_preference(
        self,
        response_a: str,
        response_b: str,
        prompt: str,
        critiques_a: List["CritiqueResult"],
        critiques_b: List["CritiqueResult"],
    ) -> Tuple[float, str]:
        """
        Compute preference between two responses.

        Args:
            response_a: First response
            response_b: Second response
            prompt: Original prompt
            critiques_a: Critiques of response A
            critiques_b: Critiques of response B

        Returns:
            Tuple of (score, reason) where score > 0.5 means A preferred
        """
        ...


# =============================================================================
# Phase 3 Refinement: Thompson Sampling for Adaptive Principle Weighting
# =============================================================================


@dataclass
class PrincipleWeightPrior:
    """
    Beta distribution prior for principle weight learning.

    Uses Thompson Sampling to adaptively adjust principle weights
    based on observed violations and their severity.

    December 2025: Added for adaptive principle importance learning.
    """
    alpha: float = 1.0  # Successes (principle was important for catching issues)
    beta: float = 1.0   # Failures (principle was not discriminative)
    base_weight: float = 1.0  # Original weight from principle definition

    def sample(self) -> float:
        """
        Sample from Beta posterior to get stochastic weight.

        Returns:
            Sampled weight multiplier (0.5 to 2.0 range)
        """
        sampled = random.betavariate(self.alpha, self.beta)
        # Map to [0.5, 2.0] range centered on base_weight
        return self.base_weight * (0.5 + sampled * 1.5)

    def expected_weight(self) -> float:
        """Get expected weight based on posterior mean."""
        expected = self.alpha / (self.alpha + self.beta)
        return self.base_weight * (0.5 + expected * 1.5)

    def update_success(self, severity_weight: float = 1.0):
        """
        Update prior after principle caught a violation.

        Args:
            severity_weight: Weight based on violation severity (0.0-1.0)
        """
        self.alpha += severity_weight

    def update_failure(self, weight: float = 0.1):
        """
        Update prior when principle was checked but found no issues.

        Args:
            weight: Small weight since no violation doesn't mean principle is unimportant
        """
        self.beta += weight


class AdaptivePrincipleWeighter:
    """
    Thompson Sampling-based adaptive principle weighting.

    Learns optimal principle weights based on:
    - Which principles catch real violations
    - Severity of violations caught
    - False positive rates

    December 2025: Added for intelligent principle prioritization.
    """

    def __init__(self, principles: List["ConstitutionalPrinciple"]):
        """
        Initialize with principle set.

        Args:
            principles: List of constitutional principles to track
        """
        self._priors: Dict[str, PrincipleWeightPrior] = {
            p.id: PrincipleWeightPrior(base_weight=p.weight)
            for p in principles
        }
        self._total_critiques = 0
        self._severity_weights = {
            "none": 0.0,
            "minor": 0.25,
            "moderate": 0.5,
            "severe": 0.75,
            "critical": 1.0,
        }

    def get_weighted_principles(
        self,
        principles: List["ConstitutionalPrinciple"],
        use_sampling: bool = True,
    ) -> List[Tuple["ConstitutionalPrinciple", float]]:
        """
        Get principles with adaptive weights.

        Args:
            principles: Principles to weight
            use_sampling: Whether to use Thompson Sampling (exploration)
                         or expected values (exploitation)

        Returns:
            List of (principle, weight) tuples sorted by weight
        """
        weighted = []
        for p in principles:
            prior = self._priors.get(p.id)
            if prior:
                weight = prior.sample() if use_sampling else prior.expected_weight()
            else:
                weight = p.weight
            weighted.append((p, weight))

        return sorted(weighted, key=lambda x: x[1], reverse=True)

    def update_from_critiques(self, critiques: List["CritiqueResult"]):
        """
        Update principle weights based on critique results.

        Args:
            critiques: Critique results from evaluation
        """
        self._total_critiques += 1

        for critique in critiques:
            prior = self._priors.get(critique.principle.id)
            if prior is None:
                continue

            if not critique.adheres:
                # Principle caught a violation - update success
                severity_weight = self._severity_weights.get(
                    critique.severity.value, 0.5
                )
                prior.update_success(severity_weight * critique.confidence)
            else:
                # Principle found no issues
                prior.update_failure(0.1)

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics for monitoring."""
        stats = {
            "total_critiques": self._total_critiques,
            "principle_weights": {},
        }

        for principle_id, prior in self._priors.items():
            stats["principle_weights"][principle_id] = {
                "alpha": prior.alpha,
                "beta": prior.beta,
                "expected_weight": prior.expected_weight(),
                "base_weight": prior.base_weight,
            }

        return stats


class PrincipleCategory(str, Enum):
    """
    Categories of constitutional principles.

    Organized by ethical domain for comprehensive coverage.
    """
    HARMLESSNESS = "harmlessness"      # Avoid harm to users and others
    HELPFULNESS = "helpfulness"        # Be genuinely helpful
    HONESTY = "honesty"                # Be truthful and transparent
    SAFETY = "safety"                  # Prioritize safety
    FAIRNESS = "fairness"              # Avoid bias and discrimination
    PRIVACY = "privacy"                # Respect user privacy
    AUTONOMY = "autonomy"              # Respect user autonomy
    LEGALITY = "legality"              # Comply with laws
    TRANSPARENCY = "transparency"      # Be clear about capabilities/limitations
    ALIGNMENT = "alignment"            # Stay aligned with human values


class ViolationSeverity(str, Enum):
    """
    Severity of principle violations.
    """
    NONE = "none"              # No violation
    MINOR = "minor"            # Small deviation, easy to fix
    MODERATE = "moderate"      # Significant issue requiring attention
    SEVERE = "severe"          # Major violation requiring intervention
    CRITICAL = "critical"      # Dangerous violation requiring immediate action


class CritiqueType(str, Enum):
    """
    Types of critique operations.
    """
    RED_TEAM = "red_team"              # Adversarial critique
    HARMLESSNESS = "harmlessness"      # Focus on avoiding harm
    HELPFULNESS = "helpfulness"        # Focus on being useful
    ETHICAL = "ethical"                # Broad ethical review
    FACTUAL = "factual"                # Accuracy check
    TONE = "tone"                      # Communication style


@dataclass
class ConstitutionalPrinciple:
    """
    A single constitutional principle.

    Each principle defines what the AI should or shouldn't do.
    """
    id: str
    category: PrincipleCategory
    name: str
    description: str
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    weight: float = 1.0  # Relative importance
    critique_prompt: Optional[str] = None  # Custom critique prompt

    def __post_init__(self):
        """Generate default critique prompt if not provided."""
        if self.critique_prompt is None:
            self.critique_prompt = (
                f"Evaluate whether the response adheres to the principle: "
                f"'{self.name}' - {self.description}"
            )


@dataclass
class CritiqueResult:
    """
    Result of critiquing a response against principles.
    """
    principle: ConstitutionalPrinciple
    adheres: bool
    severity: ViolationSeverity
    critique: str
    suggested_revision: Optional[str] = None
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "principle_id": self.principle.id,
            "principle_name": self.principle.name,
            "adheres": self.adheres,
            "severity": self.severity.value,
            "critique": self.critique,
            "suggested_revision": self.suggested_revision,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class RevisionResult:
    """
    Result of revising a response based on critique.
    """
    original_response: str
    revised_response: str
    critiques_addressed: List[CritiqueResult]
    revision_round: int
    improvement_score: float  # 0.0-1.0
    remaining_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "original_response": self.original_response,
            "revised_response": self.revised_response,
            "critiques_addressed": [c.to_dict() for c in self.critiques_addressed],
            "revision_round": self.revision_round,
            "improvement_score": self.improvement_score,
            "remaining_issues": self.remaining_issues,
        }


@dataclass
class PreferencePair:
    """
    A preference pair for RLAIF training.

    Captures which response is preferred and why.
    """
    prompt: str
    chosen_response: str
    rejected_response: str
    preference_strength: float  # 0.0-1.0
    principles_improved: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "prompt": self.prompt,
            "chosen_response": self.chosen_response,
            "rejected_response": self.rejected_response,
            "preference_strength": self.preference_strength,
            "principles_improved": self.principles_improved,
            "timestamp": self.timestamp.isoformat(),
        }


class Constitution:
    """
    The constitutional framework containing all principles.

    Provides default principles and allows customization.
    """

    # Default constitutional principles based on best practices
    DEFAULT_PRINCIPLES: List[ConstitutionalPrinciple] = [
        # Harmlessness principles
        ConstitutionalPrinciple(
            id="harm_prevention",
            category=PrincipleCategory.HARMLESSNESS,
            name="Harm Prevention",
            description="Avoid outputs that could cause physical, psychological, or financial harm",
            positive_examples=["I can't provide instructions for weapons"],
            negative_examples=["Here's how to make a dangerous substance"],
            weight=2.0
        ),
        ConstitutionalPrinciple(
            id="no_manipulation",
            category=PrincipleCategory.HARMLESSNESS,
            name="No Manipulation",
            description="Do not manipulate, deceive, or exploit users",
            positive_examples=["I want to be transparent about my limitations"],
            negative_examples=["You must do this or something bad will happen"],
            weight=1.5
        ),
        ConstitutionalPrinciple(
            id="child_safety",
            category=PrincipleCategory.HARMLESSNESS,
            name="Child Safety",
            description="Protect minors from harmful or inappropriate content",
            weight=2.5
        ),

        # Honesty principles
        ConstitutionalPrinciple(
            id="truthfulness",
            category=PrincipleCategory.HONESTY,
            name="Truthfulness",
            description="Provide accurate information; acknowledge uncertainty",
            positive_examples=["I'm not certain about this, but..."],
            negative_examples=["I definitely know that..."],
            weight=1.5
        ),
        ConstitutionalPrinciple(
            id="no_hallucination",
            category=PrincipleCategory.HONESTY,
            name="No Hallucination",
            description="Do not fabricate facts, citations, or capabilities",
            weight=1.5
        ),
        ConstitutionalPrinciple(
            id="transparent_limitations",
            category=PrincipleCategory.TRANSPARENCY,
            name="Acknowledge Limitations",
            description="Be clear about what you can and cannot do",
            weight=1.0
        ),

        # Helpfulness principles
        ConstitutionalPrinciple(
            id="genuine_assistance",
            category=PrincipleCategory.HELPFULNESS,
            name="Genuine Assistance",
            description="Provide actually useful help, not just superficial responses",
            weight=1.0
        ),
        ConstitutionalPrinciple(
            id="task_completion",
            category=PrincipleCategory.HELPFULNESS,
            name="Task Completion",
            description="Help users accomplish their legitimate goals",
            weight=1.0
        ),

        # Safety principles
        ConstitutionalPrinciple(
            id="safe_defaults",
            category=PrincipleCategory.SAFETY,
            name="Safe Defaults",
            description="When uncertain, choose the safer option",
            weight=1.5
        ),
        ConstitutionalPrinciple(
            id="corrigibility",
            category=PrincipleCategory.SAFETY,
            name="Corrigibility",
            description="Allow and facilitate correction and oversight",
            weight=1.5
        ),

        # Fairness principles
        ConstitutionalPrinciple(
            id="no_discrimination",
            category=PrincipleCategory.FAIRNESS,
            name="No Discrimination",
            description="Treat all users fairly regardless of demographics",
            weight=1.5
        ),
        ConstitutionalPrinciple(
            id="balanced_perspective",
            category=PrincipleCategory.FAIRNESS,
            name="Balanced Perspective",
            description="Present multiple viewpoints on controversial topics",
            weight=1.0
        ),

        # Privacy principles
        ConstitutionalPrinciple(
            id="respect_privacy",
            category=PrincipleCategory.PRIVACY,
            name="Respect Privacy",
            description="Do not request or expose personal information unnecessarily",
            weight=1.5
        ),

        # Autonomy principles
        ConstitutionalPrinciple(
            id="respect_autonomy",
            category=PrincipleCategory.AUTONOMY,
            name="Respect User Autonomy",
            description="Support user decision-making without undue influence",
            weight=1.0
        ),

        # Alignment principles
        ConstitutionalPrinciple(
            id="goal_transparency",
            category=PrincipleCategory.ALIGNMENT,
            name="Goal Transparency",
            description="Be clear about motivations and objectives",
            weight=1.5
        ),
        ConstitutionalPrinciple(
            id="value_alignment",
            category=PrincipleCategory.ALIGNMENT,
            name="Value Alignment",
            description="Act in accordance with human values and wellbeing",
            weight=2.0
        ),
    ]

    def __init__(self, principles: Optional[List[ConstitutionalPrinciple]] = None):
        """
        Initialize constitution with principles.

        Args:
            principles: Custom principles, or None for defaults
        """
        self.principles = principles or self.DEFAULT_PRINCIPLES.copy()
        self._principle_map = {p.id: p for p in self.principles}

    def get_principle(self, principle_id: str) -> Optional[ConstitutionalPrinciple]:
        """Get a principle by ID."""
        return self._principle_map.get(principle_id)

    def get_by_category(self, category: PrincipleCategory) -> List[ConstitutionalPrinciple]:
        """Get all principles in a category."""
        return [p for p in self.principles if p.category == category]

    def add_principle(self, principle: ConstitutionalPrinciple):
        """Add a new principle."""
        self.principles.append(principle)
        self._principle_map[principle.id] = principle

    def get_weighted_principles(self, min_weight: float = 0.0) -> List[ConstitutionalPrinciple]:
        """Get principles sorted by weight."""
        filtered = [p for p in self.principles if p.weight >= min_weight]
        return sorted(filtered, key=lambda p: p.weight, reverse=True)


class SelfCritique:
    """
    Phase 1: Supervised self-critique and revision.

    The agent critiques its own outputs against constitutional principles
    and generates improved versions.
    """

    def __init__(
        self,
        constitution: Optional[Constitution] = None,
        max_revision_rounds: int = 3,
        improvement_threshold: float = 0.1,
    ):
        """
        Initialize self-critique system.

        Args:
            constitution: Constitutional principles to evaluate against
            max_revision_rounds: Maximum revision iterations
            improvement_threshold: Minimum improvement to continue revising
        """
        self.constitution = constitution or Constitution()
        self.max_revision_rounds = max_revision_rounds
        self.improvement_threshold = improvement_threshold
        self._critique_history: List[CritiqueResult] = []
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def critique_response(
        self,
        prompt: str,
        response: str,
        principles: Optional[List[ConstitutionalPrinciple]] = None,
        critique_type: CritiqueType = CritiqueType.ETHICAL,
    ) -> List[CritiqueResult]:
        """
        Critique a response against constitutional principles.

        Args:
            prompt: Original user prompt
            response: Agent's response
            principles: Specific principles to check, or None for all
            critique_type: Type of critique to perform

        Returns:
            List of critique results
        """
        principles = principles or self.constitution.get_weighted_principles()
        critiques = []

        for principle in principles:
            critique = self._critique_against_principle(
                prompt, response, principle, critique_type
            )
            critiques.append(critique)
            self._critique_history.append(critique)

        return critiques

    def _critique_against_principle(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple,
        critique_type: CritiqueType,
    ) -> CritiqueResult:
        """
        Critique response against a single principle.

        Uses heuristic-based detection since we can't call LLM.
        In production, this would use LLM-based critique.
        """
        # Heuristic-based violation detection
        adheres, severity, evidence = self._detect_violation(
            response, principle, critique_type
        )

        # Generate critique explanation
        if adheres:
            critique = f"Response adheres to '{principle.name}': {principle.description}"
            suggested_revision = None
            confidence = 0.8
        else:
            critique = self._generate_critique(principle, evidence)
            suggested_revision = self._generate_revision_hint(principle, evidence)
            confidence = 0.7

        return CritiqueResult(
            principle=principle,
            adheres=adheres,
            severity=severity,
            critique=critique,
            suggested_revision=suggested_revision,
            confidence=confidence,
            evidence=evidence,
        )

    def _detect_violation(
        self,
        response: str,
        principle: ConstitutionalPrinciple,
        critique_type: CritiqueType,
    ) -> Tuple[bool, ViolationSeverity, List[str]]:
        """
        Detect if response violates a principle.

        Returns (adheres, severity, evidence)
        """
        response_lower = response.lower()
        evidence = []

        # Category-specific detection patterns
        if principle.category == PrincipleCategory.HARMLESSNESS:
            harm_patterns = [
                r"\bhow to (make|create|build) (weapons?|bombs?|explosives?)",
                r"\bhow to (hurt|harm|kill|attack)",
                r"instructions for (illegal|harmful|dangerous)",
                r"\b(self-harm|suicide) (methods?|instructions?)",
            ]
            for pattern in harm_patterns:
                if re.search(pattern, response_lower):
                    evidence.append(f"Pattern match: {pattern}")
                    return False, ViolationSeverity.CRITICAL, evidence

        elif principle.category == PrincipleCategory.HONESTY:
            # Check for hallucination indicators
            certainty_patterns = [
                r"I (definitely|absolutely|certainly) know",
                r"This is (absolutely|definitely|undeniably) true",
                r"There is no doubt that",
            ]
            for pattern in certainty_patterns:
                if re.search(pattern, response_lower):
                    evidence.append(f"Overconfident claim: {pattern}")
                    return False, ViolationSeverity.MODERATE, evidence

        elif principle.category == PrincipleCategory.TRANSPARENCY:
            # Check if limitations are acknowledged
            if len(response) > 500 and "I'm not" not in response and "I cannot" not in response:
                # Long response without limitations acknowledgment
                evidence.append("Long response without acknowledging limitations")
                return False, ViolationSeverity.MINOR, evidence

        elif principle.category == PrincipleCategory.FAIRNESS:
            # Check for discriminatory language
            bias_patterns = [
                r"\b(all|every) (men|women|blacks?|whites?) (are|always)",
                r"because (he's|she's|they're) a (man|woman)",
            ]
            for pattern in bias_patterns:
                if re.search(pattern, response_lower):
                    evidence.append(f"Potential bias: {pattern}")
                    return False, ViolationSeverity.SEVERE, evidence

        elif principle.category == PrincipleCategory.PRIVACY:
            # Check for privacy violations
            privacy_patterns = [
                r"(social security|ssn|credit card) number",
                r"(password|secret|private key)",
                r"(home address|phone number) of",
            ]
            for pattern in privacy_patterns:
                if re.search(pattern, response_lower):
                    evidence.append(f"Privacy concern: {pattern}")
                    return False, ViolationSeverity.SEVERE, evidence

        elif principle.category == PrincipleCategory.ALIGNMENT:
            # Check for goal transparency
            deceptive_patterns = [
                r"don't tell (anyone|them|the user)",
                r"hide this from",
                r"without (their|the user's) knowledge",
            ]
            for pattern in deceptive_patterns:
                if re.search(pattern, response_lower):
                    evidence.append(f"Deceptive pattern: {pattern}")
                    return False, ViolationSeverity.CRITICAL, evidence

        # Check against negative examples
        for neg_example in principle.negative_examples:
            if neg_example.lower() in response_lower:
                evidence.append(f"Matches negative example: {neg_example}")
                return False, ViolationSeverity.MODERATE, evidence

        return True, ViolationSeverity.NONE, evidence

    def _generate_critique(
        self,
        principle: ConstitutionalPrinciple,
        evidence: List[str],
    ) -> str:
        """Generate critique explanation."""
        critique = f"Response may violate '{principle.name}': {principle.description}\n"
        critique += "Evidence of potential violation:\n"
        for e in evidence:
            critique += f"  - {e}\n"
        return critique

    def _generate_revision_hint(
        self,
        principle: ConstitutionalPrinciple,
        evidence: List[str],
    ) -> str:
        """Generate hint for revising the response."""
        hints = {
            PrincipleCategory.HARMLESSNESS: "Remove or refuse potentially harmful content",
            PrincipleCategory.HONESTY: "Add uncertainty markers or verify claims",
            PrincipleCategory.TRANSPARENCY: "Acknowledge limitations and capabilities",
            PrincipleCategory.FAIRNESS: "Remove biased language and present balanced views",
            PrincipleCategory.PRIVACY: "Remove or redact personal information",
            PrincipleCategory.ALIGNMENT: "Be transparent about goals and intentions",
            PrincipleCategory.SAFETY: "Choose the safer option when uncertain",
            PrincipleCategory.HELPFULNESS: "Provide more actionable assistance",
            PrincipleCategory.AUTONOMY: "Support user decision-making without pressure",
            PrincipleCategory.LEGALITY: "Ensure compliance with applicable laws",
        }
        return hints.get(principle.category, "Review and revise the response")

    async def revise_response(
        self,
        prompt: str,
        response: str,
        critiques: List[CritiqueResult],
        revise_fn: Optional[Callable[[str, str, List[CritiqueResult]], Awaitable[str]]] = None,
    ) -> RevisionResult:
        """
        Revise response based on critique.

        Args:
            prompt: Original prompt
            response: Original response
            critiques: Critiques to address
            revise_fn: Optional async function to generate revision

        Returns:
            Revision result with improved response
        """
        # Filter to failing critiques
        failing = [c for c in critiques if not c.adheres]

        if not failing:
            return RevisionResult(
                original_response=response,
                revised_response=response,
                critiques_addressed=[],
                revision_round=0,
                improvement_score=1.0,
            )

        # Generate revision (heuristic if no revise_fn)
        if revise_fn:
            revised = await revise_fn(prompt, response, failing)
        else:
            revised = self._heuristic_revise(response, failing)

        # Calculate improvement
        new_critiques = self.critique_response(prompt, revised)
        new_failing = [c for c in new_critiques if not c.adheres]

        addressed = len(failing) - len(new_failing)
        improvement = addressed / len(failing) if failing else 1.0

        return RevisionResult(
            original_response=response,
            revised_response=revised,
            critiques_addressed=[c for c in critiques if c.adheres or c.principle.id in
                                 [nc.principle.id for nc in new_critiques if nc.adheres]],
            revision_round=1,
            improvement_score=improvement,
            remaining_issues=[c.critique for c in new_failing],
        )

    def _heuristic_revise(
        self,
        response: str,
        critiques: List[CritiqueResult],
    ) -> str:
        """Heuristically revise response based on critiques."""
        revised = response

        for critique in critiques:
            if critique.severity == ViolationSeverity.CRITICAL:
                # For critical violations, add refusal
                revised = (
                    "I'm not able to provide that information as it could be harmful. "
                    "Let me help you with something else instead.\n\n" +
                    "(Note: Original response contained potentially harmful content)"
                )
                break

            elif critique.severity == ViolationSeverity.SEVERE:
                # For severe violations, add strong disclaimer
                revised = (
                    "Note: This response has been flagged for review.\n\n" +
                    revised +
                    "\n\n(Content may require revision for policy compliance)"
                )

            elif critique.principle.category == PrincipleCategory.HONESTY:
                # Add uncertainty markers
                if "I definitely" in revised:
                    revised = revised.replace("I definitely", "Based on my knowledge, I believe")
                if "absolutely true" in revised.lower():
                    revised = revised.replace("absolutely true", "likely accurate")

            elif critique.principle.category == PrincipleCategory.TRANSPARENCY:
                # Add limitations acknowledgment
                revised += "\n\nNote: This response is based on my training data and may have limitations."

        return revised

    async def critique_and_revise(
        self,
        prompt: str,
        response: str,
        revise_fn: Optional[Callable[[str, str, List[CritiqueResult]], Awaitable[str]]] = None,
    ) -> Tuple[str, List[RevisionResult]]:
        """
        Full critique and revision loop.

        Returns final response and revision history.
        """
        current = response
        history = []

        for round_num in range(self.max_revision_rounds):
            critiques = self.critique_response(prompt, current)
            revision = await self.revise_response(prompt, current, critiques, revise_fn)
            revision.revision_round = round_num + 1
            history.append(revision)

            # Check if we've improved enough
            if revision.improvement_score < self.improvement_threshold:
                logger.info(f"Stopping revision: improvement {revision.improvement_score:.2f} below threshold")
                break

            # Check if no more issues
            if not revision.remaining_issues:
                logger.info("Stopping revision: all issues addressed")
                break

            current = revision.revised_response

        return current, history

    # =========================================================================
    # Phase 3 Refinement: Batch Processing (December 2025)
    # =========================================================================

    async def batch_critique(
        self,
        items: List[Tuple[str, str]],
        parallel: bool = True,
        max_concurrent: int = 10,
    ) -> List[List[CritiqueResult]]:
        """
        Batch critique multiple prompt-response pairs.

        December 2025: Added for efficient parallel processing.

        Args:
            items: List of (prompt, response) tuples
            parallel: Whether to process in parallel
            max_concurrent: Maximum concurrent operations

        Returns:
            List of critique results for each item
        """
        if not items:
            return []

        if not parallel:
            # Sequential processing
            return [
                self.critique_response(prompt, response)
                for prompt, response in items
            ]

        # Parallel processing with semaphore
        semaphore = asyncio.Semaphore(max_concurrent)

        async def critique_one(prompt: str, response: str) -> List[CritiqueResult]:
            async with semaphore:
                # critique_response is sync, wrap in executor for true parallelism
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self.critique_response, prompt, response, None, CritiqueType.ETHICAL
                )

        tasks = [critique_one(prompt, response) for prompt, response in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions - return empty critiques for failures
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Batch critique {i} failed: {result}")
                processed.append([])
            else:
                processed.append(result)

        return processed

    async def batch_critique_and_revise(
        self,
        items: List[Tuple[str, str]],
        parallel: bool = True,
        max_concurrent: int = 5,
    ) -> List[Tuple[str, List[RevisionResult]]]:
        """
        Batch critique and revise multiple prompt-response pairs.

        December 2025: Added for efficient parallel processing.

        Args:
            items: List of (prompt, response) tuples
            parallel: Whether to process in parallel
            max_concurrent: Maximum concurrent operations

        Returns:
            List of (final_response, revision_history) tuples
        """
        if not items:
            return []

        if not parallel:
            # Sequential processing
            results = []
            for prompt, response in items:
                final, history = await self.critique_and_revise(prompt, response)
                results.append((final, history))
            return results

        # Parallel processing with semaphore
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(prompt: str, response: str) -> Tuple[str, List[RevisionResult]]:
            async with semaphore:
                return await self.critique_and_revise(prompt, response)

        tasks = [process_one(prompt, response) for prompt, response in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Batch critique-revise {i} failed: {result}")
                # Return original response unchanged on failure
                processed.append((items[i][1], []))
            else:
                processed.append(result)

        return processed


class RLAIFTrainer:
    """
    Phase 2: Reinforcement Learning from AI Feedback.

    Collects preference pairs from self-critique and uses them
    for training a preference model (conceptual - actual RL training
    would require a training framework).
    """

    def __init__(
        self,
        constitution: Optional[Constitution] = None,
        self_critique: Optional[SelfCritique] = None,
    ):
        """
        Initialize RLAIF trainer.

        Args:
            constitution: Constitutional principles
            self_critique: Self-critique system for generating preferences
        """
        self.constitution = constitution or Constitution()
        self.self_critique = self_critique or SelfCritique(self.constitution)
        self.preference_pairs: List[PreferencePair] = []
        self._preference_buffer_size = 10000

    async def generate_preference_pair(
        self,
        prompt: str,
        original_response: str,
    ) -> Optional[PreferencePair]:
        """
        Generate a preference pair from critique and revision.

        Args:
            prompt: User prompt
            original_response: Original agent response

        Returns:
            Preference pair if revision improved response, None otherwise
        """
        final_response, history = await self.self_critique.critique_and_revise(
            prompt, original_response
        )

        if not history or final_response == original_response:
            return None

        # Calculate preference strength based on improvement
        total_improvement = sum(r.improvement_score for r in history) / len(history)

        # Get principles that were improved
        improved_principles = set()
        for revision in history:
            for critique in revision.critiques_addressed:
                improved_principles.add(critique.principle.name)

        pair = PreferencePair(
            prompt=prompt,
            chosen_response=final_response,
            rejected_response=original_response,
            preference_strength=total_improvement,
            principles_improved=list(improved_principles),
        )

        self.preference_pairs.append(pair)

        # Buffer management
        if len(self.preference_pairs) > self._preference_buffer_size:
            self.preference_pairs = self.preference_pairs[-self._preference_buffer_size:]

        return pair

    def get_training_batch(
        self,
        batch_size: int = 32,
        min_strength: float = 0.3,
    ) -> List[PreferencePair]:
        """
        Get a batch of preference pairs for training.

        Args:
            batch_size: Number of pairs to return
            min_strength: Minimum preference strength

        Returns:
            List of high-quality preference pairs
        """
        filtered = [p for p in self.preference_pairs if p.preference_strength >= min_strength]

        # Sort by strength (highest first)
        filtered.sort(key=lambda p: p.preference_strength, reverse=True)

        return filtered[:batch_size]

    def compute_preference_score(
        self,
        response_a: str,
        response_b: str,
        prompt: str,
    ) -> Tuple[float, str]:
        """
        Compute which response is preferred.

        Returns (score, reason) where score > 0.5 means A preferred.
        """
        critiques_a = self.self_critique.critique_response(prompt, response_a)
        critiques_b = self.self_critique.critique_response(prompt, response_b)

        # Count violations
        violations_a = sum(1 for c in critiques_a if not c.adheres)
        violations_b = sum(1 for c in critiques_b if not c.adheres)

        # Weight by severity
        severity_weights = {
            ViolationSeverity.NONE: 0,
            ViolationSeverity.MINOR: 1,
            ViolationSeverity.MODERATE: 2,
            ViolationSeverity.SEVERE: 4,
            ViolationSeverity.CRITICAL: 8,
        }

        weighted_a = sum(severity_weights[c.severity] for c in critiques_a if not c.adheres)
        weighted_b = sum(severity_weights[c.severity] for c in critiques_b if not c.adheres)

        # Score (higher = A preferred)
        if weighted_a == weighted_b:
            score = 0.5
            reason = "Responses are equivalent"
        elif weighted_a < weighted_b:
            score = 0.5 + (weighted_b - weighted_a) / (weighted_b + weighted_a + 1) * 0.5
            reason = f"Response A has fewer/less severe violations ({weighted_a} vs {weighted_b})"
        else:
            score = 0.5 - (weighted_a - weighted_b) / (weighted_a + weighted_b + 1) * 0.5
            reason = f"Response B has fewer/less severe violations ({weighted_b} vs {weighted_a})"

        return score, reason

    def export_preferences(self) -> List[Dict[str, Any]]:
        """Export preference pairs for external training."""
        return [p.to_dict() for p in self.preference_pairs]


class ValueAlignmentVerifier:
    """
    Continuous verification of value alignment.

    Monitors agent outputs for alignment drift and provides
    ongoing verification that the agent remains aligned.
    """

    def __init__(
        self,
        constitution: Optional[Constitution] = None,
        drift_threshold: float = 0.15,
        window_size: int = 100,
    ):
        """
        Initialize alignment verifier.

        Args:
            constitution: Constitutional principles
            drift_threshold: Threshold for detecting alignment drift
            window_size: Number of interactions to track
        """
        self.constitution = constitution or Constitution()
        self.self_critique = SelfCritique(self.constitution)
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self._interaction_history: List[Dict[str, Any]] = []
        self._baseline_adherence: Optional[Dict[str, float]] = None

    def record_interaction(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record an interaction and check alignment.

        Returns alignment metrics for this interaction.
        """
        critiques = self.self_critique.critique_response(prompt, response)

        # Calculate per-category adherence
        adherence = {}
        for category in PrincipleCategory:
            cat_critiques = [c for c in critiques if c.principle.category == category]
            if cat_critiques:
                adherence[category.value] = sum(1 for c in cat_critiques if c.adheres) / len(cat_critiques)

        # Calculate overall adherence
        overall = sum(1 for c in critiques if c.adheres) / len(critiques) if critiques else 1.0

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8],
            "response_hash": hashlib.md5(response.encode()).hexdigest()[:8],
            "overall_adherence": overall,
            "category_adherence": adherence,
            "violations": [c.to_dict() for c in critiques if not c.adheres],
            "metadata": metadata or {},
        }

        self._interaction_history.append(interaction)

        # Keep window size
        if len(self._interaction_history) > self.window_size:
            self._interaction_history = self._interaction_history[-self.window_size:]

        return interaction

    def set_baseline(self):
        """Set current adherence as baseline for drift detection."""
        if len(self._interaction_history) < 10:
            logger.warning("Insufficient history for baseline (need 10+)")
            return

        self._baseline_adherence = self._compute_window_adherence(
            self._interaction_history
        )
        logger.info(f"Baseline set: {self._baseline_adherence}")

    def _compute_window_adherence(
        self,
        interactions: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute average adherence over a window."""
        if not interactions:
            return {}

        adherence = {"overall": 0.0}
        for category in PrincipleCategory:
            adherence[category.value] = 0.0

        for interaction in interactions:
            adherence["overall"] += interaction["overall_adherence"]
            for cat, value in interaction.get("category_adherence", {}).items():
                adherence[cat] = adherence.get(cat, 0.0) + value

        n = len(interactions)
        return {k: v / n for k, v in adherence.items()}

    def detect_drift(self) -> Tuple[bool, float, str]:
        """
        Detect alignment drift from baseline.

        Returns (drift_detected, drift_magnitude, explanation)
        """
        if self._baseline_adherence is None:
            return False, 0.0, "No baseline set"

        if len(self._interaction_history) < 10:
            return False, 0.0, "Insufficient recent history"

        current = self._compute_window_adherence(
            self._interaction_history[-min(50, len(self._interaction_history)):]
        )

        # Compare to baseline
        drifts = []
        for key in self._baseline_adherence:
            baseline_val = self._baseline_adherence[key]
            current_val = current.get(key, 0.0)
            if baseline_val > 0:
                drift = (baseline_val - current_val) / baseline_val
                drifts.append((key, drift))

        max_drift = max(drifts, key=lambda x: x[1]) if drifts else ("none", 0.0)

        if max_drift[1] > self.drift_threshold:
            return True, max_drift[1], f"Drift detected in {max_drift[0]}: {max_drift[1]:.1%} decrease"

        return False, max_drift[1], f"No significant drift (max: {max_drift[1]:.1%} in {max_drift[0]})"

    def get_alignment_report(self) -> Dict[str, Any]:
        """Generate comprehensive alignment report."""
        if not self._interaction_history:
            return {"status": "no_data", "message": "No interactions recorded"}

        current = self._compute_window_adherence(self._interaction_history)
        drift_detected, drift_magnitude, drift_explanation = self.detect_drift()

        # Find most common violations
        violation_counts: Dict[str, int] = {}
        for interaction in self._interaction_history:
            for violation in interaction.get("violations", []):
                principle_id = violation.get("principle_id", "unknown")
                violation_counts[principle_id] = violation_counts.get(principle_id, 0) + 1

        top_violations = sorted(
            violation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            "status": "warning" if drift_detected else "healthy",
            "overall_adherence": current.get("overall", 0.0),
            "category_adherence": {k: v for k, v in current.items() if k != "overall"},
            "drift_detected": drift_detected,
            "drift_magnitude": drift_magnitude,
            "drift_explanation": drift_explanation,
            "total_interactions": len(self._interaction_history),
            "top_violations": top_violations,
            "baseline_set": self._baseline_adherence is not None,
        }


class ConstitutionalAI:
    """
    Complete Constitutional AI system.

    Combines self-critique, RLAIF training, and continuous verification
    into a unified alignment system.
    """

    def __init__(
        self,
        constitution: Optional[Constitution] = None,
        max_revision_rounds: int = 3,
        enable_rlaif: bool = True,
        enable_verification: bool = True,
    ):
        """
        Initialize Constitutional AI system.

        Args:
            constitution: Constitutional principles
            max_revision_rounds: Max critique-revise iterations
            enable_rlaif: Enable preference learning
            enable_verification: Enable continuous verification
        """
        self.constitution = constitution or Constitution()
        self.self_critique = SelfCritique(
            self.constitution,
            max_revision_rounds=max_revision_rounds
        )

        self.rlaif_trainer = RLAIFTrainer(
            self.constitution,
            self.self_critique
        ) if enable_rlaif else None

        self.verifier = ValueAlignmentVerifier(
            self.constitution
        ) if enable_verification else None

        self._stats = {
            "total_critiques": 0,
            "total_revisions": 0,
            "total_verifications": 0,
            "violations_detected": 0,
        }

    async def process_response(
        self,
        prompt: str,
        response: str,
        auto_revise: bool = True,
        record_for_training: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a response through the full CAI pipeline.

        Args:
            prompt: User prompt
            response: Agent response
            auto_revise: Automatically revise if issues found
            record_for_training: Record for RLAIF training

        Returns:
            Complete processing result
        """
        self._stats["total_critiques"] += 1

        # Phase 1: Critique
        critiques = self.self_critique.critique_response(prompt, response)
        violations = [c for c in critiques if not c.adheres]
        self._stats["violations_detected"] += len(violations)

        final_response = response
        revision_history = []

        # Auto-revise if needed
        if auto_revise and violations:
            self._stats["total_revisions"] += 1
            final_response, revision_history = await self.self_critique.critique_and_revise(
                prompt, response
            )

        # Record for RLAIF
        preference_pair = None
        if record_for_training and self.rlaif_trainer and violations:
            preference_pair = await self.rlaif_trainer.generate_preference_pair(
                prompt, response
            )

        # Verify alignment
        verification = None
        if self.verifier:
            self._stats["total_verifications"] += 1
            verification = self.verifier.record_interaction(
                prompt, final_response
            )

        return {
            "original_response": response,
            "final_response": final_response,
            "was_revised": final_response != response,
            "critiques": [c.to_dict() for c in critiques],
            "violations_found": len(violations),
            "revision_history": [r.to_dict() for r in revision_history],
            "preference_recorded": preference_pair is not None,
            "verification": verification,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = self._stats.copy()

        if self.rlaif_trainer:
            stats["preference_pairs_collected"] = len(self.rlaif_trainer.preference_pairs)

        if self.verifier:
            stats["alignment_report"] = self.verifier.get_alignment_report()

        return stats

    # =========================================================================
    # Phase 3 Refinement: Batch Processing (December 2025)
    # =========================================================================

    async def batch_process(
        self,
        items: List[Tuple[str, str]],
        auto_revise: bool = True,
        record_for_training: bool = True,
        parallel: bool = True,
        max_concurrent: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple prompt-response pairs through the CAI pipeline.

        December 2025: Added for efficient batch processing.

        Args:
            items: List of (prompt, response) tuples
            auto_revise: Automatically revise if issues found
            record_for_training: Record for RLAIF training
            parallel: Whether to process in parallel
            max_concurrent: Maximum concurrent operations

        Returns:
            List of processing results for each item
        """
        if not items:
            return []

        if not parallel:
            # Sequential processing
            results = []
            for prompt, response in items:
                result = await self.process_response(
                    prompt, response, auto_revise, record_for_training
                )
                results.append(result)
            return results

        # Parallel processing with semaphore
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(prompt: str, response: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.process_response(
                    prompt, response, auto_revise, record_for_training
                )

        tasks = [process_one(prompt, response) for prompt, response in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Batch process {i} failed: {result}")
                processed.append({
                    "original_response": items[i][1],
                    "final_response": items[i][1],
                    "was_revised": False,
                    "critiques": [],
                    "violations_found": 0,
                    "error": str(result),
                })
            else:
                processed.append(result)

        return processed

    async def batch_verify(
        self,
        items: List[Tuple[str, str]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batch verify alignment for multiple prompt-response pairs.

        December 2025: Added for efficient batch verification.

        Args:
            items: List of (prompt, response) tuples
            metadata: Optional list of metadata dicts for each item

        Returns:
            List of verification results for each item
        """
        if not self.verifier:
            return [{"error": "Verification not enabled"} for _ in items]

        results = []
        for i, (prompt, response) in enumerate(items):
            item_metadata = metadata[i] if metadata and i < len(metadata) else None
            result = self.verifier.record_interaction(prompt, response, item_metadata)
            results.append(result)

        return results


# Factory functions
def create_constitution(
    include_defaults: bool = True,
    custom_principles: Optional[List[ConstitutionalPrinciple]] = None,
) -> Constitution:
    """
    Create a constitution with optional customization.

    Args:
        include_defaults: Include default principles
        custom_principles: Additional custom principles

    Returns:
        Configured Constitution instance
    """
    principles = []

    if include_defaults:
        principles.extend(Constitution.DEFAULT_PRINCIPLES)

    if custom_principles:
        principles.extend(custom_principles)

    return Constitution(principles)


def create_constitutional_ai(
    constitution: Optional[Constitution] = None,
    max_revision_rounds: int = 3,
    enable_rlaif: bool = True,
    enable_verification: bool = True,
    enable_adaptive_weighting: bool = False,
    critique_engine: Optional[CritiqueEngineProtocol] = None,
    revision_engine: Optional[RevisionEngineProtocol] = None,
) -> ConstitutionalAI:
    """
    Create a complete Constitutional AI system.

    Args:
        constitution: Custom constitution or None for defaults
        max_revision_rounds: Maximum revision iterations
        enable_rlaif: Enable preference learning
        enable_verification: Enable continuous verification
        enable_adaptive_weighting: Enable Thompson Sampling principle weighting
            December 2025: Added for Phase 3 extensibility
        critique_engine: Custom critique engine implementing CritiqueEngineProtocol
            December 2025: Added for Phase 3 extensibility
        revision_engine: Custom revision engine implementing RevisionEngineProtocol
            December 2025: Added for Phase 3 extensibility

    Returns:
        Configured ConstitutionalAI instance

    Note:
        Custom critique_engine and revision_engine are stored on the returned
        instance but integration requires additional wiring in production.
        They provide extensibility points for future enhancements.
    """
    cai = ConstitutionalAI(
        constitution=constitution,
        max_revision_rounds=max_revision_rounds,
        enable_rlaif=enable_rlaif,
        enable_verification=enable_verification,
    )

    # Store extensibility references (December 2025)
    if enable_adaptive_weighting and cai.constitution:
        cai._adaptive_weighter = AdaptivePrincipleWeighter(
            cai.constitution.get_weighted_principles()
        )
    else:
        cai._adaptive_weighter = None

    # Store custom engines for extensibility
    cai._custom_critique_engine = critique_engine
    cai._custom_revision_engine = revision_engine

    return cai


# Quick check function
def quick_critique(
    response: str,
    prompt: str = "",
    categories: Optional[List[PrincipleCategory]] = None,
) -> Dict[str, Any]:
    """
    Quick critique of a response.

    Args:
        response: Response to critique
        prompt: Optional prompt context
        categories: Specific categories to check

    Returns:
        Quick critique summary
    """
    constitution = Constitution()
    critique = SelfCritique(constitution)

    if categories:
        principles = []
        for cat in categories:
            principles.extend(constitution.get_by_category(cat))
    else:
        principles = None

    results = critique.critique_response(prompt, response, principles)

    violations = [c for c in results if not c.adheres]

    return {
        "adheres": len(violations) == 0,
        "total_principles_checked": len(results),
        "violations_found": len(violations),
        "violation_summary": [
            {
                "principle": v.principle.name,
                "severity": v.severity.value,
                "critique": v.critique[:200],
            }
            for v in violations
        ],
        "overall_score": (len(results) - len(violations)) / len(results) if results else 1.0,
    }
