"""
Result Synthesizer for Chain of Verification.

Aggregates verification results and synthesizes a refined response.
Handles:
1. Keeping verified claims unchanged
2. Correcting contradicted claims with verified information
3. Flagging uncertain claims for human review
4. Calculating overall confidence

Created: 2025-12-05
"""

import logging
from typing import Any

from hololoom.verification.protocol import (
    Contradiction,
    DegradationLevel,
    ResultSynthesizerProtocol,
    VerifiableClaim,
    VerificationAnswer,
    VerificationConfig,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Claim Status Determination
# =============================================================================

def determine_claim_status(
    claim: VerifiableClaim,
    answers: list[VerificationAnswer],
    contradictions: list[Contradiction],
    config: VerificationConfig,
) -> tuple[VerificationStatus, float]:
    """
    Determine the verification status of a claim.

    Returns:
        Tuple of (status, confidence adjustment)
    """
    # Find answers and contradictions for this specific claim
    claim_answers = [a for a in answers if a.question.claim.text == claim.text]
    claim_contradictions = [c for c in contradictions if c.claim.text == claim.text]

    # Check for contradictions first
    if claim_contradictions:
        max_severity = max(c.severity for c in claim_contradictions)
        if max_severity >= config.contradiction_threshold:
            return (VerificationStatus.CONTRADICTED, -0.3)
        else:
            return (VerificationStatus.UNCERTAIN, -0.1)

    # Check answer confidence
    if claim_answers:
        avg_confidence = sum(a.confidence for a in claim_answers) / len(claim_answers)

        if avg_confidence >= 0.8:
            return (VerificationStatus.VERIFIED, 0.1)
        elif avg_confidence >= 0.5:
            return (VerificationStatus.VERIFIED, 0.0)
        elif avg_confidence >= 0.3:
            return (VerificationStatus.UNCERTAIN, -0.05)
        else:
            return (VerificationStatus.NEEDS_HUMAN, -0.15)

    # No answers - uncertain
    return (VerificationStatus.UNCERTAIN, 0.0)


def aggregate_claim_statuses(
    claims: list[VerifiableClaim],
    answers: list[VerificationAnswer],
    contradictions: list[Contradiction],
    config: VerificationConfig,
) -> dict[str, tuple[VerificationStatus, float]]:
    """Aggregate statuses for all claims."""
    statuses = {}
    for claim in claims:
        status, adjustment = determine_claim_status(
            claim, answers, contradictions, config
        )
        statuses[claim.text] = (status, adjustment)
    return statuses


def calculate_overall_status(
    claim_statuses: dict[str, tuple[VerificationStatus, float]],
) -> VerificationStatus:
    """Calculate overall verification status."""
    if not claim_statuses:
        return VerificationStatus.UNCERTAIN

    statuses = [s for s, _ in claim_statuses.values()]

    # If any claim is contradicted, overall is contradicted
    if VerificationStatus.CONTRADICTED in statuses:
        return VerificationStatus.CONTRADICTED

    # If any claim needs human review, overall needs review
    if VerificationStatus.NEEDS_HUMAN in statuses:
        return VerificationStatus.NEEDS_HUMAN

    # If most claims are verified, overall is verified
    verified_count = statuses.count(VerificationStatus.VERIFIED)
    if verified_count > len(statuses) / 2:
        return VerificationStatus.VERIFIED

    return VerificationStatus.UNCERTAIN


def calculate_new_confidence(
    original_confidence: float,
    claim_statuses: dict[str, tuple[VerificationStatus, float]],
) -> float:
    """Calculate new confidence based on verification results."""
    if not claim_statuses:
        return original_confidence

    # Sum all adjustments
    total_adjustment = sum(adj for _, adj in claim_statuses.values())

    # Apply with dampening
    new_confidence = original_confidence + (total_adjustment * 0.5)

    return max(0.0, min(1.0, new_confidence))


# =============================================================================
# Rule-Based Result Synthesizer (HEURISTIC Level)
# =============================================================================

class RuleBasedResultSynthesizer:
    """
    Rule-based result synthesizer using simple replacement.

    For each contradicted claim:
    1. Find the verification answer that contradicts it
    2. Replace the claim with corrected information
    3. Mark the correction in the response

    For verified/uncertain claims:
    - Keep them unchanged
    """

    def __init__(self, config: VerificationConfig | None = None):
        self.config = config or VerificationConfig()

    async def synthesize(
        self,
        original_response: str,
        claims: list[VerifiableClaim],
        answers: list[VerificationAnswer],
        contradictions: list[Contradiction],
        context: dict[str, Any]
    ) -> tuple[str, float]:
        """Synthesize refined response."""
        # Determine status for each claim
        claim_statuses = aggregate_claim_statuses(
            claims, answers, contradictions, self.config
        )

        # Calculate new confidence
        original_confidence = context.get('original_confidence', 0.5)
        new_confidence = calculate_new_confidence(original_confidence, claim_statuses)

        # Build refined response
        refined = self._refine_response(
            original_response, claims, answers, contradictions, claim_statuses
        )

        return (refined, new_confidence)

    def _refine_response(
        self,
        original: str,
        claims: list[VerifiableClaim],
        answers: list[VerificationAnswer],
        contradictions: list[Contradiction],
        claim_statuses: dict[str, tuple[VerificationStatus, float]],
    ) -> str:
        """Refine the response by correcting contradicted claims."""
        refined = original

        # Process contradicted claims
        for claim in claims:
            status, _ = claim_statuses.get(claim.text, (VerificationStatus.UNCERTAIN, 0.0))

            if status == VerificationStatus.CONTRADICTED:
                # Find the contradiction and correction
                correction = self._find_correction(claim, contradictions, answers)
                if correction:
                    refined = self._apply_correction(refined, claim, correction)

        return refined

    def _find_correction(
        self,
        claim: VerifiableClaim,
        contradictions: list[Contradiction],
        answers: list[VerificationAnswer],
    ) -> str | None:
        """Find correction for a contradicted claim."""
        # Find the contradiction
        for contradiction in contradictions:
            if contradiction.claim.text == claim.text:
                # Use the verification evidence as correction
                return contradiction.evidence

        # Fall back to verification answers
        for answer in answers:
            if answer.question.claim.text == claim.text and answer.confidence > 0.6:
                return answer.answer

        return None

    def _apply_correction(
        self,
        response: str,
        claim: VerifiableClaim,
        correction: str,
    ) -> str:
        """Apply correction to response."""
        # Simple replacement with marker
        corrected_text = f"[CORRECTED: {correction}]"

        # Try exact replacement
        if claim.text in response:
            return response.replace(claim.text, corrected_text)

        # Try span-based replacement
        start, end = claim.source_span
        if 0 <= start < end <= len(response):
            return response[:start] + corrected_text + response[end:]

        # Fall back to appending correction
        return f"{response}\n\n[CORRECTION: {claim.text} → {correction}]"


# =============================================================================
# LLM-Based Result Synthesizer (FULL Level)
# =============================================================================

SYNTHESIS_PROMPT = '''Revise the following response based on verification results.

Original Response:
{original_response}

Verification Results:
{verification_summary}

Contradictions Found:
{contradictions_summary}

Instructions:
1. Keep verified claims unchanged
2. Correct contradicted claims using the verification evidence
3. Soften or remove unsupported claims
4. Maintain the original tone and structure
5. Do not add new information beyond corrections

Provide the revised response:'''


class LLMResultSynthesizer:
    """
    LLM-based result synthesizer for intelligent revision.

    Uses language models to:
    - Intelligently integrate corrections
    - Maintain coherence and flow
    - Preserve tone while improving accuracy
    """

    def __init__(
        self,
        config: VerificationConfig | None = None,
        llm_client: Any | None = None,
    ):
        self.config = config or VerificationConfig()
        self.llm_client = llm_client
        self._fallback = RuleBasedResultSynthesizer(config)

    async def synthesize(
        self,
        original_response: str,
        claims: list[VerifiableClaim],
        answers: list[VerificationAnswer],
        contradictions: list[Contradiction],
        context: dict[str, Any]
    ) -> tuple[str, float]:
        """Synthesize refined response using LLM."""
        if self.llm_client is None:
            logger.warning("No LLM client available, falling back to rule-based synthesis")
            return await self._fallback.synthesize(
                original_response, claims, answers, contradictions, context
            )

        # Determine statuses and confidence
        claim_statuses = aggregate_claim_statuses(
            claims, answers, contradictions, self.config
        )
        original_confidence = context.get('original_confidence', 0.5)
        new_confidence = calculate_new_confidence(original_confidence, claim_statuses)

        # Check if we need to refine
        contradicted = any(
            s == VerificationStatus.CONTRADICTED
            for s, _ in claim_statuses.values()
        )

        if not contradicted:
            # No contradictions - return original with updated confidence
            return (original_response, new_confidence)

        try:
            # Build synthesis prompt
            verification_summary = self._build_verification_summary(claims, answers, claim_statuses)
            contradictions_summary = self._build_contradictions_summary(contradictions)

            prompt = SYNTHESIS_PROMPT.format(
                original_response=original_response,
                verification_summary=verification_summary,
                contradictions_summary=contradictions_summary,
            )

            # Call LLM
            refined = await self._call_llm(prompt)

            return (refined.strip(), new_confidence)

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}, falling back to rule-based")
            return await self._fallback.synthesize(
                original_response, claims, answers, contradictions, context
            )

    def _build_verification_summary(
        self,
        claims: list[VerifiableClaim],
        answers: list[VerificationAnswer],
        statuses: dict[str, tuple[VerificationStatus, float]],
    ) -> str:
        """Build summary of verification results."""
        lines = []

        for claim in claims:
            status, _ = statuses.get(claim.text, (VerificationStatus.UNCERTAIN, 0.0))
            status_emoji = {
                VerificationStatus.VERIFIED: "✓",
                VerificationStatus.CONTRADICTED: "✗",
                VerificationStatus.UNCERTAIN: "?",
                VerificationStatus.NEEDS_HUMAN: "⚠",
            }.get(status, "?")

            lines.append(f"{status_emoji} {claim.text[:100]}...")

            # Add relevant answer
            for answer in answers:
                if answer.question.claim.text == claim.text:
                    lines.append(f"  → {answer.answer[:100]}...")
                    break

        return "\n".join(lines)

    def _build_contradictions_summary(
        self,
        contradictions: list[Contradiction],
    ) -> str:
        """Build summary of contradictions."""
        if not contradictions:
            return "No contradictions found."

        lines = []
        for c in contradictions:
            lines.append(f"• Claim: {c.claim.text[:80]}...")
            lines.append(f"  Contradiction: {c.evidence[:80]}...")
            lines.append(f"  Severity: {c.severity:.1%}")
            lines.append(f"  Explanation: {c.explanation[:100]}...")
            lines.append("")

        return "\n".join(lines)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM client."""
        if hasattr(self.llm_client, 'generate'):
            return await self.llm_client.generate(prompt)
        elif hasattr(self.llm_client, 'complete'):
            return await self.llm_client.complete(prompt)
        elif callable(self.llm_client):
            result = self.llm_client(prompt)
            if hasattr(result, '__await__'):
                return await result
            return result
        else:
            raise ValueError(f"Unknown LLM client type: {type(self.llm_client)}")


# =============================================================================
# Confidence Recalibrator
# =============================================================================

class ConfidenceRecalibrator:
    """
    Recalibrates confidence based on verification outcomes.

    Uses Bayesian updating to adjust confidence based on:
    - Number of verified claims
    - Number of contradicted claims
    - Verification answer confidence
    - Historical accuracy (if available)
    """

    def __init__(self, config: VerificationConfig | None = None):
        self.config = config or VerificationConfig()
        self._history: list[dict[str, float]] = []

    def recalibrate(
        self,
        original_confidence: float,
        claims: list[VerifiableClaim],
        answers: list[VerificationAnswer],
        contradictions: list[Contradiction],
    ) -> float:
        """Recalibrate confidence using Bayesian approach."""
        if not claims:
            return original_confidence

        # Calculate evidence from verification
        n_verified = 0
        n_contradicted = 0
        answer_confidences = []

        for claim in claims:
            claim_answers = [a for a in answers if a.question.claim.text == claim.text]
            claim_contradictions = [c for c in contradictions if c.claim.text == claim.text]

            if claim_contradictions and any(c.severity > 0.7 for c in claim_contradictions):
                n_contradicted += 1
            elif claim_answers and any(a.confidence > 0.7 for a in claim_answers):
                n_verified += 1

            answer_confidences.extend([a.confidence for a in claim_answers])

        # Bayesian update
        # Prior: original confidence
        # Likelihood: based on verification outcomes
        prior = original_confidence

        # Calculate likelihood ratio
        if n_verified + n_contradicted > 0:
            verification_rate = n_verified / (n_verified + n_contradicted)
        else:
            verification_rate = 0.5

        # Average answer confidence
        if answer_confidences:
            avg_answer_conf = sum(answer_confidences) / len(answer_confidences)
        else:
            avg_answer_conf = 0.5

        # Combine evidence
        evidence = (verification_rate + avg_answer_conf) / 2

        # Bayesian update (simplified)
        posterior = (prior * evidence) / (
            prior * evidence + (1 - prior) * (1 - evidence)
        )

        # Store for history
        self._history.append({
            'prior': prior,
            'evidence': evidence,
            'posterior': posterior,
            'n_verified': n_verified,
            'n_contradicted': n_contradicted,
        })

        return posterior


# =============================================================================
# Factory Function
# =============================================================================

def create_result_synthesizer(
    config: VerificationConfig | None = None,
    llm_client: Any | None = None,
) -> ResultSynthesizerProtocol:
    """
    Create appropriate result synthesizer based on configuration.

    Args:
        config: Verification configuration
        llm_client: Optional LLM client for LLM-based synthesis

    Returns:
        ResultSynthesizer implementation
    """
    config = config or VerificationConfig()

    if config.preferred_level == DegradationLevel.HEURISTIC:
        return RuleBasedResultSynthesizer(config)

    if config.preferred_level == DegradationLevel.DISABLED:
        # Return a pass-through synthesizer
        class PassThroughSynthesizer:
            async def synthesize(self, original, claims, answers, contradictions, context):
                return (original, context.get('original_confidence', 0.5))
        return PassThroughSynthesizer()

    if llm_client is not None:
        return LLMResultSynthesizer(config, llm_client)

    return RuleBasedResultSynthesizer(config)
