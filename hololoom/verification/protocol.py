"""
Chain of Verification Protocol Definitions.

This module defines the core protocols and data structures for the
Chain-of-Verification (CoVe) system, following Meta AI's methodology
for reducing hallucinations through systematic verification.

Key Research Foundation:
- TACL 2024 Survey: Self-correction fails without external feedback
- ACL Findings 2024: CoVe 4-step methodology significantly reduces hallucinations
- EMNLP 2024 (MiniCheck): Token-level uncertainty quantification

Created: 2025-12-05
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


class VerificationStatus(Enum):
    """Status of a verification check."""
    VERIFIED = "verified"           # Claim confirmed by verification
    CONTRADICTED = "contradicted"   # Claim contradicts verification evidence
    UNCERTAIN = "uncertain"         # Verification inconclusive
    NEEDS_HUMAN = "needs_human"     # Requires human review
    SKIPPED = "skipped"             # Verification skipped (low importance)


class ClaimType(Enum):
    """Type of verifiable claim."""
    FACTUAL = "factual"             # Verifiable facts (dates, names, definitions)
    PROCEDURAL = "procedural"       # Steps or processes
    CAUSAL = "causal"               # Cause-effect relationships
    DEFINITIONAL = "definitional"   # Definitions and explanations
    COMPARATIVE = "comparative"     # Comparisons between entities
    QUANTITATIVE = "quantitative"   # Numerical claims
    TEMPORAL = "temporal"           # Time-based claims
    UNKNOWN = "unknown"             # Unclassified claims


class DegradationLevel(Enum):
    """Graceful degradation levels for verification."""
    FULL = "full"           # Complete CoVe (4 steps) with LLM
    PARTIAL = "partial"     # CoVe with local/smaller LLM (Ollama)
    HEURISTIC = "heuristic" # Regex claims + similarity matching
    DISABLED = "disabled"   # Skip verification entirely


class ContradictionType(Enum):
    """Type of contradiction detected."""
    FACTUAL = "factual"         # Facts disagree
    LOGICAL = "logical"         # Logical inconsistency
    TEMPORAL = "temporal"       # Time-related contradiction
    QUANTITATIVE = "quantitative"  # Numbers disagree
    SEMANTIC = "semantic"       # Meaning conflicts
    NEGATION = "negation"       # Direct negation
    UNKNOWN = "unknown"         # Unclassified


@dataclass
class VerifiableClaim:
    """
    A single verifiable claim extracted from a response.

    Attributes:
        text: The claim text
        claim_type: Type of claim (factual, procedural, etc.)
        confidence: Initial confidence in the claim (0.0-1.0)
        source_span: Character positions in original response (start, end)
        importance: How important this claim is to the overall response (0.0-1.0)
        entities: Named entities mentioned in the claim
        metadata: Additional metadata about the claim
    """
    text: str
    claim_type: ClaimType = ClaimType.UNKNOWN
    confidence: float = 0.5
    source_span: tuple[int, int] = (0, 0)
    importance: float = 0.5
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate claim attributes."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"Importance must be 0.0-1.0, got {self.importance}")


@dataclass
class VerificationQuestion:
    """
    A question generated to verify a specific claim.

    Attributes:
        question: The verification question text
        claim: The claim this question is verifying
        question_type: Type of verification (fact-check, consistency, etc.)
        expected_answer_type: What kind of answer is expected
    """
    question: str
    claim: VerifiableClaim
    question_type: str = "fact_check"
    expected_answer_type: str = "boolean"  # boolean, entity, explanation


@dataclass
class VerificationAnswer:
    """
    An answer to a verification question, obtained independently.

    Attributes:
        question: The original question
        answer: The verification answer
        confidence: Confidence in this answer (0.0-1.0)
        sources: Sources used to derive this answer
        reasoning: Explanation of how answer was derived
    """
    question: VerificationQuestion
    answer: str
    confidence: float = 0.5
    sources: list[str] = field(default_factory=list)
    reasoning: str = ""

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")


@dataclass
class Contradiction:
    """
    A detected contradiction between a claim and verification evidence.

    Attributes:
        claim: The original claim
        evidence: The contradicting evidence
        contradiction_type: Type of contradiction
        severity: How severe the contradiction is (0.0-1.0)
        explanation: Human-readable explanation
    """
    claim: VerifiableClaim
    evidence: str
    severity: float = 0.5  # 0.0 = minor, 1.0 = critical
    explanation: str = ""
    contradiction_type: ContradictionType = ContradictionType.UNKNOWN

    def __post_init__(self):
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"Severity must be 0.0-1.0, got {self.severity}")
        # Allow string for backward compatibility
        if isinstance(self.contradiction_type, str):
            try:
                self.contradiction_type = ContradictionType(self.contradiction_type)
            except ValueError:
                self.contradiction_type = ContradictionType.UNKNOWN


@dataclass
class VerificationResult:
    """
    Complete result of the verification chain.

    Attributes:
        original_response: The initial response that was verified
        verified_response: The refined response after verification
        claims_extracted: All claims found in the response
        verification_questions: Questions generated for verification
        verification_answers: Answers to verification questions
        contradictions_found: Detected contradictions
        confidence_before: Confidence before verification
        confidence_after: Confidence after verification
        verification_passed: Whether overall verification passed
        refinement_applied: Whether response was refined
        status: Overall verification status
        iterations: Number of refinement iterations performed
        latency_ms: Total verification time in milliseconds
        degradation_level: What level of verification was performed
        metadata: Additional verification metadata
    """
    original_response: str
    verified_response: str
    claims_extracted: list[VerifiableClaim] = field(default_factory=list)
    verification_questions: list[VerificationQuestion] = field(default_factory=list)
    verification_answers: list[VerificationAnswer] = field(default_factory=list)
    contradictions_found: list[Contradiction] = field(default_factory=list)
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    verification_passed: bool = False
    refinement_applied: bool = False
    status: VerificationStatus = VerificationStatus.UNCERTAIN
    iterations: int = 0
    latency_ms: float = 0.0
    degradation_level: DegradationLevel = DegradationLevel.FULL
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate human-readable summary of verification result."""
        # Use ASCII markers instead of emojis for terminal compatibility
        status_marker = {
            VerificationStatus.VERIFIED: "[OK]",
            VerificationStatus.CONTRADICTED: "[FAIL]",
            VerificationStatus.UNCERTAIN: "[WARN]",
            VerificationStatus.NEEDS_HUMAN: "[HUMAN]",
            VerificationStatus.SKIPPED: "[SKIP]",
        }

        marker = status_marker.get(self.status, "[?]")

        lines = [
            f"{marker} Verification {self.status.value.upper()}",
            f"   Claims extracted: {len(self.claims_extracted)}",
            f"   Questions asked: {len(self.verification_questions)}",
            f"   Contradictions: {len(self.contradictions_found)}",
            f"   Confidence: {self.confidence_before:.2f} -> {self.confidence_after:.2f}",
            f"   Iterations: {self.iterations}",
            f"   Latency: {self.latency_ms:.1f}ms",
            f"   Level: {self.degradation_level.value}",
        ]

        if self.refinement_applied:
            lines.append("   Response was refined")

        return "\n".join(lines)


# =============================================================================
# Protocol Definitions (Extensibility)
# =============================================================================

class ClaimExtractorProtocol(Protocol):
    """
    Protocol for extracting verifiable claims from text.

    Implementations can use:
    - LLM-based extraction (most accurate)
    - Rule-based extraction (fast, no dependencies)
    - Hybrid approaches
    """

    async def extract(
        self,
        text: str,
        context: dict[str, Any]
    ) -> list[VerifiableClaim]:
        """
        Extract verifiable claims from text.

        Args:
            text: The response text to extract claims from
            context: Additional context (query, domain, etc.)

        Returns:
            List of extracted claims with importance scores
        """
        ...


class VerificationPlannerProtocol(Protocol):
    """
    Protocol for generating verification questions for claims.

    The planner creates targeted questions that can independently
    verify whether a claim is accurate.
    """

    async def plan(
        self,
        claim: VerifiableClaim,
        context: dict[str, Any]
    ) -> list[VerificationQuestion]:
        """
        Generate verification questions for a claim.

        Args:
            claim: The claim to generate questions for
            context: Additional context for question generation

        Returns:
            List of verification questions
        """
        ...


class IndependentVerifierProtocol(Protocol):
    """
    Protocol for answering verification questions INDEPENDENTLY.

    CRITICAL: Verification must be done WITHOUT seeing the original
    response to prevent self-confirmation bias. This is the key
    insight from the CoVe research.
    """

    async def verify(
        self,
        questions: list[VerificationQuestion],
        context: dict[str, Any]
    ) -> list[VerificationAnswer]:
        """
        Answer verification questions without seeing original response.

        Args:
            questions: Questions to answer
            context: Context for answering (knowledge base, external sources)
                     MUST NOT include the original response

        Returns:
            Answers to each verification question
        """
        ...


class ContradictionDetectorProtocol(Protocol):
    """
    Protocol for detecting contradictions between claims and evidence.

    Uses semantic analysis rather than simple keyword matching.
    """

    async def detect(
        self,
        claim: VerifiableClaim,
        answers: list[VerificationAnswer]
    ) -> list[Contradiction]:
        """
        Detect contradictions between a claim and verification answers.

        Args:
            claim: The original claim
            answers: Verification answers to compare against

        Returns:
            List of detected contradictions (empty if none)
        """
        ...


class ResultSynthesizerProtocol(Protocol):
    """
    Protocol for synthesizing a verified response from verification results.
    """

    async def synthesize(
        self,
        original_response: str,
        claims: list[VerifiableClaim],
        answers: list[VerificationAnswer],
        contradictions: list[Contradiction],
        context: dict[str, Any]
    ) -> tuple[str, float]:
        """
        Synthesize a verified response.

        Args:
            original_response: The original response
            claims: Extracted claims
            answers: Verification answers
            contradictions: Detected contradictions
            context: Additional context

        Returns:
            Tuple of (refined_response, new_confidence)
        """
        ...


class VerificationChainProtocol(Protocol):
    """
    Protocol for the complete verification chain.

    Implements the full 4-step CoVe methodology:
    1. Extract claims from baseline response
    2. Generate verification questions
    3. Answer questions independently
    4. Synthesize verified response
    """

    async def verify(
        self,
        query: str,
        response: str,
        confidence: float,
        context: dict[str, Any] | None = None,
        max_iterations: int = 3,
        confidence_threshold: float = 0.85,
    ) -> "VerificationResult":
        """
        Execute full verification chain.

        Args:
            query: Original query that generated the response
            response: Response to verify
            confidence: Initial confidence in response (0.0-1.0)
            context: Additional context for verification
            max_iterations: Maximum refinement iterations
            confidence_threshold: Confidence needed to pass verification

        Returns:
            VerificationResult with verified response and metadata
        """
        ...


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VerificationConfig:
    """Configuration for the verification chain."""

    # Thresholds
    confidence_threshold: float = 0.85      # Below this triggers verification
    contradiction_threshold: float = 0.7    # Above this marks as contradicted
    importance_threshold: float = 0.3       # Below this skips verification

    # Limits
    max_claims: int = 10                    # Max claims to extract
    max_questions_per_claim: int = 3        # Max questions per claim
    max_iterations: int = 3                 # Max refinement iterations

    # Performance
    parallel_verification: bool = True      # Verify claims in parallel
    enable_caching: bool = True             # Cache verification results
    cache_ttl_seconds: int = 3600           # Cache TTL (1 hour)

    # Degradation
    preferred_level: DegradationLevel = DegradationLevel.FULL
    fallback_on_error: bool = True          # Fall back to lower level on error

    # LLM settings
    llm_provider: str | None = None      # "anthropic", "openai", "ollama"
    llm_model: str | None = None         # Model name
    llm_temperature: float = 0.1            # Low temp for factual verification

    # Timeouts
    timeout_ms: float = 5000.0              # Total verification timeout
    claim_timeout_ms: float = 1000.0        # Per-claim timeout

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be 0.0-1.0")
        if not 0.0 <= self.contradiction_threshold <= 1.0:
            raise ValueError("contradiction_threshold must be 0.0-1.0")
        if self.max_claims < 1:
            raise ValueError("max_claims must be >= 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_config() -> VerificationConfig:
    """Create default verification configuration."""
    return VerificationConfig()


def create_fast_config() -> VerificationConfig:
    """Create fast verification configuration with lower thresholds."""
    return VerificationConfig(
        max_claims=5,
        max_questions_per_claim=2,
        max_iterations=1,
        timeout_ms=2000.0,
        preferred_level=DegradationLevel.HEURISTIC,
    )


def create_thorough_config() -> VerificationConfig:
    """Create thorough verification configuration for high-stakes responses."""
    return VerificationConfig(
        confidence_threshold=0.9,
        max_claims=20,
        max_questions_per_claim=5,
        max_iterations=5,
        timeout_ms=10000.0,
        preferred_level=DegradationLevel.FULL,
    )
