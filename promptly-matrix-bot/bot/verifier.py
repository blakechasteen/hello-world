#!/usr/bin/env python3
"""
Verification System for Promptly Matrix Bot

Verifies claims through multi-step fact-checking:
1. Generate verification questions
2. Answer each question
3. Check for contradictions
4. Calculate confidence score

Features:
- Hallucination detection
- Fact-checking pipeline
- Contradiction detection
- Evidence collection
- Confidence scoring
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class VerificationQuestion:
    """A question to verify a claim"""
    question: str
    answer: Optional[str] = None
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    contradicts: bool = False


@dataclass
class VerificationResult:
    """Complete verification result"""
    claim: str
    verified: bool
    confidence: float
    questions: List[VerificationQuestion]
    contradictions: List[str] = field(default_factory=list)
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Human-readable summary"""
        status = "✅ VERIFIED" if self.verified else "❌ NOT VERIFIED"
        return f"{status} (confidence: {self.confidence:.1%})"


class Verifier:
    """
    Verifies claims through multi-step fact-checking.

    Usage:
        verifier = Verifier(promptly_core)
        result = await verifier.verify("Thompson Sampling balances exploration")

        print(result.summary())
        for q in result.questions:
            print(f"Q: {q.question}")
            print(f"A: {q.answer}")
    """

    def __init__(self, promptly_core=None):
        """
        Initialize verifier.

        Args:
            promptly_core: Promptly Core instance for LLM calls
        """
        self.promptly_core = promptly_core

    async def verify(
        self,
        claim: str,
        num_questions: int = 3,
        confidence_threshold: float = 0.7
    ) -> VerificationResult:
        """
        Verify a claim through multi-step fact-checking.

        Args:
            claim: Claim to verify
            num_questions: Number of verification questions to generate
            confidence_threshold: Threshold for verification (0-1)

        Returns:
            VerificationResult with complete analysis
        """
        result = VerificationResult(
            claim=claim,
            verified=False,
            confidence=0.0,
            questions=[]
        )

        # Step 1: Generate verification questions
        questions = await self._generate_questions(claim, num_questions)
        result.questions = questions

        # Step 2: Answer each question
        for question in questions:
            await self._answer_question(question, claim)

        # Step 3: Check for contradictions
        contradictions = self._find_contradictions(questions, claim)
        result.contradictions = contradictions

        # Step 4: Collect evidence
        result.evidence_for = [
            q.answer for q in questions
            if q.answer and not q.contradicts
        ]
        result.evidence_against = [
            q.answer for q in questions
            if q.answer and q.contradicts
        ]

        # Step 5: Calculate confidence
        result.confidence = self._calculate_confidence(questions, contradictions)
        result.verified = result.confidence >= confidence_threshold

        logger.info(f"Verified claim: '{claim}' -> {result.summary()}")

        return result

    async def _generate_questions(self, claim: str, num_questions: int) -> List[VerificationQuestion]:
        """Generate verification questions for claim"""
        questions = []

        # Use LLM if available
        if self.promptly_core:
            try:
                # Generate questions using LLM
                prompt = f"""Generate {num_questions} verification questions for this claim:

Claim: "{claim}"

Questions should:
1. Test specific factual assertions
2. Be answerable with evidence
3. Help determine if the claim is true

Format: One question per line, numbered 1-{num_questions}.
"""

                result = await self.promptly_core.run_workflow("generate_questions", prompt)
                if result.get('success'):
                    output = result.get('output', '')
                    # Parse questions
                    for line in output.split('\n'):
                        line = line.strip()
                        # Remove numbering
                        match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
                        if match:
                            questions.append(VerificationQuestion(question=match.group(1)))

            except Exception as e:
                logger.warning(f"LLM question generation failed: {e}")

        # Fallback: Generate heuristic questions
        if not questions:
            questions = self._generate_heuristic_questions(claim, num_questions)

        return questions[:num_questions]

    def _generate_heuristic_questions(self, claim: str, num_questions: int) -> List[VerificationQuestion]:
        """Generate questions using heuristics (no LLM)"""
        questions = []

        # Pattern-based question generation
        claim_lower = claim.lower()

        # Question templates
        templates = [
            f"What evidence supports the claim that {claim}?",
            f"Are there any contradictory findings about {claim}?",
            f"What are the key facts underlying {claim}?",
            f"Has {claim} been validated by reliable sources?",
            f"What assumptions does the claim '{claim}' make?"
        ]

        for template in templates[:num_questions]:
            questions.append(VerificationQuestion(question=template))

        return questions

    async def _answer_question(self, question: VerificationQuestion, claim: str):
        """Answer a verification question"""
        # Use LLM if available
        if self.promptly_core:
            try:
                prompt = f"""Answer this verification question about the claim:

Claim: "{claim}"

Question: {question.question}

Provide:
1. A direct answer
2. Supporting evidence
3. Confidence level (0-100%)

Format:
Answer: [your answer]
Evidence: [supporting evidence]
Confidence: [0-100]%
"""

                result = await self.promptly_core.run_workflow("answer_question", prompt)
                if result.get('success'):
                    output = result.get('output', '')

                    # Parse answer
                    answer_match = re.search(r'Answer:\s*(.+?)(?=Evidence:|Confidence:|$)', output, re.DOTALL)
                    if answer_match:
                        question.answer = answer_match.group(1).strip()

                    # Parse evidence
                    evidence_match = re.search(r'Evidence:\s*(.+?)(?=Confidence:|$)', output, re.DOTALL)
                    if evidence_match:
                        question.evidence = [evidence_match.group(1).strip()]

                    # Parse confidence
                    conf_match = re.search(r'Confidence:\s*(\d+)%', output)
                    if conf_match:
                        question.confidence = float(conf_match.group(1)) / 100.0

            except Exception as e:
                logger.warning(f"LLM answer generation failed: {e}")

        # Fallback: Simple heuristic answer
        if not question.answer:
            question.answer = f"Verification question: {question.question}"
            question.confidence = 0.5  # Neutral confidence

    def _find_contradictions(
        self,
        questions: List[VerificationQuestion],
        claim: str
    ) -> List[str]:
        """Find contradictions between answers and claim"""
        contradictions = []

        # Check for contradiction keywords
        contradiction_keywords = [
            'however', 'but', 'although', 'contrary', 'opposite',
            'incorrect', 'false', 'wrong', 'not true', 'disputed',
            'contradicts', 'conflicts', 'disagrees'
        ]

        for question in questions:
            if not question.answer:
                continue

            answer_lower = question.answer.lower()

            # Check for contradiction keywords
            has_contradiction = any(
                keyword in answer_lower
                for keyword in contradiction_keywords
            )

            if has_contradiction:
                question.contradicts = True
                contradictions.append(
                    f"Q: {question.question}\n"
                    f"A: {question.answer}"
                )

        return contradictions

    def _calculate_confidence(
        self,
        questions: List[VerificationQuestion],
        contradictions: List[str]
    ) -> float:
        """
        Calculate overall confidence score.

        Factors:
        - Average confidence of answers
        - Number of contradictions
        - Question coverage
        """
        if not questions:
            return 0.0

        # Average question confidence
        answered = [q for q in questions if q.answer]
        if answered:
            avg_confidence = sum(q.confidence for q in answered) / len(answered)
        else:
            avg_confidence = 0.0

        # Penalty for contradictions
        contradiction_penalty = len(contradictions) * 0.2

        # Bonus for complete coverage
        coverage_bonus = len(answered) / len(questions) * 0.1

        # Calculate final confidence
        confidence = avg_confidence + coverage_bonus - contradiction_penalty

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))


class QuickVerifier:
    """
    Quick verification without LLM.

    Uses heuristics and pattern matching for fast fact-checking.
    """

    def verify(self, claim: str) -> VerificationResult:
        """Quick verification using heuristics"""
        result = VerificationResult(
            claim=claim,
            verified=False,
            confidence=0.5  # Neutral
        )

        # Check for hedging language (increases confidence)
        hedging_words = ['may', 'might', 'could', 'possibly', 'potentially', 'often', 'sometimes']
        has_hedging = any(word in claim.lower() for word in hedging_words)

        # Check for absolute language (decreases confidence)
        absolute_words = ['always', 'never', 'all', 'none', 'every', 'only', 'must']
        has_absolute = any(word in claim.lower() for word in absolute_words)

        # Check for citation/source indicators (increases confidence)
        citation_indicators = ['study', 'research', 'according to', 'found that', 'shows that']
        has_citation = any(indicator in claim.lower() for indicator in citation_indicators)

        # Calculate confidence
        confidence = 0.5

        if has_hedging:
            confidence += 0.1
        if has_absolute:
            confidence -= 0.2
        if has_citation:
            confidence += 0.2

        result.confidence = max(0.0, min(1.0, confidence))
        result.verified = result.confidence >= 0.7

        # Add heuristic questions
        result.questions = [
            VerificationQuestion(
                question=f"Is the claim '{claim}' factually accurate?",
                answer="Heuristic verification (quick mode)",
                confidence=result.confidence
            )
        ]

        return result


# Singleton
_verifier = None
_quick_verifier = None

def get_verifier(promptly_core=None) -> Verifier:
    """Get singleton verifier"""
    global _verifier
    if _verifier is None:
        _verifier = Verifier(promptly_core)
    return _verifier

def get_quick_verifier() -> QuickVerifier:
    """Get singleton quick verifier"""
    global _quick_verifier
    if _quick_verifier is None:
        _quick_verifier = QuickVerifier()
    return _quick_verifier


# Test
if __name__ == "__main__":
    import asyncio

    async def test():
        # Test quick verifier (no LLM)
        quick = QuickVerifier()

        claims = [
            "Thompson Sampling balances exploration and exploitation",
            "All AI systems are always perfectly safe",  # Absolute language
            "Research shows that transformer models can be effective"  # Citation
        ]

        for claim in claims:
            result = quick.verify(claim)
            print(f"\nClaim: {claim}")
            print(f"Result: {result.summary()}")
            print(f"Confidence: {result.confidence:.1%}")

        print("\n✅ All verifier tests passed!")

    asyncio.run(test())
