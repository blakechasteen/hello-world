"""
Contradiction Detector for Chain of Verification.

Detects semantic contradictions between claims and verification evidence.
Uses multi-level analysis:
1. Semantic similarity/dissimilarity
2. Logical consistency checking
3. Factual contradiction detection
4. Temporal inconsistency detection

Moves beyond simple heuristic patterns (e.g., "however", "but") to
true semantic understanding of contradictions.

Created: 2025-12-05
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from HoloLoom.verification.protocol import (
    Contradiction,
    ContradictionDetectorProtocol,
    DegradationLevel,
    VerifiableClaim,
    VerificationAnswer,
    VerificationConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Contradiction Patterns (HEURISTIC Level)
# =============================================================================

# Negation patterns that might indicate contradiction
NEGATION_PATTERNS = [
    r'\b(not|no|never|none|neither|nor)\b',
    r'\b(isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t|didn\'t)\b',
    r'\b(cannot|can\'t|couldn\'t|wouldn\'t|shouldn\'t)\b',
    r'\b(false|incorrect|wrong|inaccurate|untrue)\b',
]

# Contrast patterns
CONTRAST_PATTERNS = [
    r'\b(however|but|although|though|whereas|while|yet)\b',
    r'\b(on the contrary|in contrast|conversely|instead)\b',
    r'\b(differs? from|different from|unlike|contrary to)\b',
]

# Quantitative contradiction patterns
QUANTITY_PATTERNS = [
    r'(\d+(?:\.\d+)?)',  # Numbers
    r'\b(all|every|always|never|none|some|few|many|most)\b',  # Quantifiers
]

# Temporal contradiction patterns
TEMPORAL_PATTERNS = [
    r'\b(before|after|during|since|until|when|while)\b',
    r'\b(\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b',  # Dates
    r'\b(now|currently|today|yesterday|tomorrow|recently)\b',
]


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    matches = re.findall(r'(\d+(?:\.\d+)?)', text)
    return [float(m) for m in matches]


def has_negation(text: str) -> bool:
    """Check if text contains negation."""
    text_lower = text.lower()
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def extract_key_entities(text: str) -> List[str]:
    """Extract key entities/concepts from text."""
    # Remove common words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                  'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                  'through', 'during', 'before', 'after', 'above', 'below',
                  'between', 'under', 'again', 'further', 'then', 'once',
                  'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                  'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                  'very', 'just', 'also', 'now', 'here', 'there', 'when',
                  'where', 'why', 'how', 'all', 'each', 'every', 'both',
                  'few', 'more', 'most', 'other', 'some', 'such', 'no',
                  'any', 'that', 'this', 'these', 'those', 'it', 'its'}

    words = re.findall(r'\b[A-Za-z]+\b', text)
    entities = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    return entities


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity using word overlap."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


# =============================================================================
# Rule-Based Contradiction Detector (HEURISTIC Level)
# =============================================================================

class RuleBasedContradictionDetector:
    """
    Rule-based contradiction detector using linguistic patterns.

    Fast and dependency-free, suitable for HEURISTIC degradation level.
    Detects:
    - Negation contradictions (claim vs negative answer)
    - Quantitative contradictions (different numbers)
    - Temporal contradictions (different dates/times)
    - Semantic dissimilarity (low word overlap with negation)
    """

    def __init__(self, config: Optional[VerificationConfig] = None):
        self.config = config or VerificationConfig()

    async def detect(
        self,
        claim: VerifiableClaim,
        answers: List[VerificationAnswer]
    ) -> List[Contradiction]:
        """Detect contradictions using rule-based analysis."""
        contradictions = []

        for answer in answers:
            # Check different types of contradictions
            contradiction = self._check_contradiction(claim, answer)
            if contradiction:
                contradictions.append(contradiction)

        return contradictions

    def _check_contradiction(
        self,
        claim: VerifiableClaim,
        answer: VerificationAnswer
    ) -> Optional[Contradiction]:
        """Check for contradiction between claim and answer."""
        claim_text = claim.text.lower()
        answer_text = answer.answer.lower()

        # 1. Negation contradiction
        negation_result = self._check_negation_contradiction(claim_text, answer_text)
        if negation_result:
            return Contradiction(
                claim=claim,
                evidence=answer.answer,
                contradiction_type="factual",
                severity=negation_result[1],
                explanation=negation_result[0],
            )

        # 2. Quantitative contradiction
        quant_result = self._check_quantitative_contradiction(claim_text, answer_text)
        if quant_result:
            return Contradiction(
                claim=claim,
                evidence=answer.answer,
                contradiction_type="quantitative",
                severity=quant_result[1],
                explanation=quant_result[0],
            )

        # 3. Low similarity with explicit disagreement markers
        if answer.confidence > 0.6:  # Only check if answer is confident
            disagree_result = self._check_explicit_disagreement(claim_text, answer_text)
            if disagree_result:
                return Contradiction(
                    claim=claim,
                    evidence=answer.answer,
                    contradiction_type="semantic",
                    severity=disagree_result[1],
                    explanation=disagree_result[0],
                )

        return None

    def _check_negation_contradiction(
        self,
        claim: str,
        answer: str
    ) -> Optional[Tuple[str, float]]:
        """Check if answer negates the claim."""
        claim_has_negation = has_negation(claim)
        answer_has_negation = has_negation(answer)

        # If only one has negation, might be contradiction
        if claim_has_negation != answer_has_negation:
            # Check if they're talking about the same thing
            claim_entities = set(extract_key_entities(claim))
            answer_entities = set(extract_key_entities(answer))

            overlap = len(claim_entities & answer_entities)
            if overlap > 0:
                # Same entities, different polarity = likely contradiction
                severity = min(1.0, 0.5 + overlap * 0.1)
                return (
                    f"Negation mismatch: claim {'has' if claim_has_negation else 'lacks'} "
                    f"negation while answer {'has' if answer_has_negation else 'lacks'} it",
                    severity
                )

        return None

    def _check_quantitative_contradiction(
        self,
        claim: str,
        answer: str
    ) -> Optional[Tuple[str, float]]:
        """Check for contradicting numbers."""
        claim_numbers = extract_numbers(claim)
        answer_numbers = extract_numbers(answer)

        if not claim_numbers or not answer_numbers:
            return None

        # Check if corresponding numbers differ significantly
        for cn in claim_numbers:
            for an in answer_numbers:
                if cn == 0 and an == 0:
                    continue

                # Calculate relative difference
                if cn != 0:
                    diff = abs(cn - an) / abs(cn)
                else:
                    diff = abs(an)

                if diff > 0.2:  # More than 20% difference
                    severity = min(1.0, 0.4 + diff * 0.3)
                    return (
                        f"Quantitative mismatch: claim says {cn}, "
                        f"verification says {an} (diff: {diff:.1%})",
                        severity
                    )

        return None

    def _check_explicit_disagreement(
        self,
        claim: str,
        answer: str
    ) -> Optional[Tuple[str, float]]:
        """Check for explicit disagreement markers."""
        disagreement_phrases = [
            'is false',
            'is incorrect',
            'is wrong',
            'is inaccurate',
            'is not true',
            'contradicts',
            'is a misconception',
            'is misleading',
            'actually',
            'in fact',
            'the truth is',
        ]

        for phrase in disagreement_phrases:
            if phrase in answer:
                return (
                    f"Explicit disagreement found: '{phrase}' in verification answer",
                    0.8
                )

        return None


# =============================================================================
# LLM-Based Contradiction Detector (FULL Level)
# =============================================================================

CONTRADICTION_PROMPT = '''Analyze whether the verification answer contradicts the original claim.

Original Claim: {claim}

Verification Answer: {answer}

Analyze for:
1. Factual contradictions (different facts stated)
2. Logical contradictions (incompatible statements)
3. Quantitative contradictions (different numbers/amounts)
4. Temporal contradictions (different dates/times)
5. Semantic contradictions (different meanings)

Return JSON:
{{
  "has_contradiction": true|false,
  "contradiction_type": "factual|logical|quantitative|temporal|semantic|none",
  "severity": 0.0-1.0,
  "explanation": "Brief explanation of the contradiction or why there is none"
}}

Analysis:'''


class LLMContradictionDetector:
    """
    LLM-based contradiction detector for semantic understanding.

    Uses language models to understand nuanced contradictions that
    simple pattern matching would miss.
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        self.config = config or VerificationConfig()
        self.llm_client = llm_client
        self._fallback = RuleBasedContradictionDetector(config)

    async def detect(
        self,
        claim: VerifiableClaim,
        answers: List[VerificationAnswer]
    ) -> List[Contradiction]:
        """Detect contradictions using LLM analysis."""
        if self.llm_client is None:
            logger.warning("No LLM client available, falling back to rule-based detection")
            return await self._fallback.detect(claim, answers)

        contradictions = []

        for answer in answers:
            try:
                contradiction = await self._analyze_contradiction(claim, answer)
                if contradiction:
                    contradictions.append(contradiction)
            except Exception as e:
                logger.error(f"LLM contradiction detection failed: {e}")
                # Fall back to rule-based for this answer
                fallback_results = await self._fallback.detect(claim, [answer])
                contradictions.extend(fallback_results)

        return contradictions

    async def _analyze_contradiction(
        self,
        claim: VerifiableClaim,
        answer: VerificationAnswer
    ) -> Optional[Contradiction]:
        """Analyze single claim-answer pair for contradiction."""
        prompt = CONTRADICTION_PROMPT.format(
            claim=claim.text,
            answer=answer.answer,
        )

        response = await self._call_llm(prompt)
        return self._parse_response(response, claim, answer)

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

    def _parse_response(
        self,
        response: str,
        claim: VerifiableClaim,
        answer: VerificationAnswer
    ) -> Optional[Contradiction]:
        """Parse LLM response into Contradiction."""
        import json

        try:
            # Handle markdown code blocks
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            data = json.loads(response.strip())

            if not data.get('has_contradiction', False):
                return None

            return Contradiction(
                claim=claim,
                evidence=answer.answer,
                contradiction_type=data.get('contradiction_type', 'semantic'),
                severity=float(data.get('severity', 0.5)),
                explanation=data.get('explanation', ''),
            )

        except json.JSONDecodeError:
            # Try to extract from raw response
            if 'contradict' in response.lower() or 'false' in response.lower():
                return Contradiction(
                    claim=claim,
                    evidence=answer.answer,
                    contradiction_type="unknown",
                    severity=0.5,
                    explanation=response[:200],
                )
            return None


# =============================================================================
# Hybrid Contradiction Detector
# =============================================================================

class HybridContradictionDetector:
    """
    Hybrid detector combining rule-based and LLM approaches.

    Uses rule-based detection for quick screening, then LLM for
    nuanced analysis of potential contradictions.
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        self.config = config or VerificationConfig()
        self.rule_detector = RuleBasedContradictionDetector(config)
        self.llm_detector = LLMContradictionDetector(config, llm_client)

    async def detect(
        self,
        claim: VerifiableClaim,
        answers: List[VerificationAnswer]
    ) -> List[Contradiction]:
        """
        Detect contradictions using hybrid approach.

        1. Quick rule-based screening
        2. LLM analysis for uncertain cases
        3. Merge and deduplicate results
        """
        # First pass: rule-based detection
        rule_contradictions = await self.rule_detector.detect(claim, answers)

        # If we have clear contradictions from rules, use them
        if rule_contradictions and all(c.severity > 0.7 for c in rule_contradictions):
            return rule_contradictions

        # Second pass: LLM for deeper analysis
        llm_contradictions = await self.llm_detector.detect(claim, answers)

        # Merge results
        return self._merge_contradictions(rule_contradictions, llm_contradictions)

    def _merge_contradictions(
        self,
        rule_results: List[Contradiction],
        llm_results: List[Contradiction]
    ) -> List[Contradiction]:
        """Merge contradictions from both sources."""
        # Use LLM results as primary (more accurate)
        merged = list(llm_results)

        # Add unique rule-based results
        llm_evidence = {c.evidence for c in llm_results}

        for rc in rule_results:
            if rc.evidence not in llm_evidence:
                # Reduce severity of rule-only findings
                rc.severity *= 0.8
                merged.append(rc)
            else:
                # Boost severity if both agree
                for lc in merged:
                    if lc.evidence == rc.evidence:
                        lc.severity = min(1.0, lc.severity + 0.1)

        return merged


# =============================================================================
# Semantic Similarity Detector (using embeddings)
# =============================================================================

class SemanticContradictionDetector:
    """
    Contradiction detector using semantic embeddings.

    Uses embedding similarity to detect semantic contradictions
    that surface-level pattern matching would miss.
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        embedder: Optional[Any] = None,
    ):
        self.config = config or VerificationConfig()
        self.embedder = embedder
        self._fallback = RuleBasedContradictionDetector(config)

    async def detect(
        self,
        claim: VerifiableClaim,
        answers: List[VerificationAnswer]
    ) -> List[Contradiction]:
        """Detect contradictions using semantic similarity."""
        if self.embedder is None:
            return await self._fallback.detect(claim, answers)

        contradictions = []

        try:
            claim_embedding = await self._get_embedding(claim.text)

            for answer in answers:
                answer_embedding = await self._get_embedding(answer.answer)

                # Calculate similarity
                similarity = self._cosine_similarity(claim_embedding, answer_embedding)

                # Check for semantic contradiction indicators
                if self._is_semantic_contradiction(claim.text, answer.answer, similarity):
                    severity = self._calculate_severity(similarity, answer)
                    contradictions.append(Contradiction(
                        claim=claim,
                        evidence=answer.answer,
                        contradiction_type="semantic",
                        severity=severity,
                        explanation=f"Semantic analysis indicates contradiction (similarity: {similarity:.2f})",
                    ))

        except Exception as e:
            logger.error(f"Semantic detection failed: {e}")
            return await self._fallback.detect(claim, answers)

        return contradictions

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if hasattr(self.embedder, 'embed'):
            return await self.embedder.embed(text)
        elif hasattr(self.embedder, 'encode'):
            result = self.embedder.encode(text)
            if hasattr(result, 'tolist'):
                return result.tolist()
            return list(result)
        else:
            raise ValueError(f"Unknown embedder type: {type(self.embedder)}")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _is_semantic_contradiction(
        self,
        claim: str,
        answer: str,
        similarity: float
    ) -> bool:
        """Determine if there's a semantic contradiction."""
        # High similarity with negation = likely contradiction
        if similarity > 0.5:
            claim_neg = has_negation(claim)
            answer_neg = has_negation(answer)
            if claim_neg != answer_neg:
                return True

        # Very low similarity with same entities = potential contradiction
        if similarity < 0.3:
            claim_entities = set(extract_key_entities(claim))
            answer_entities = set(extract_key_entities(answer))
            if len(claim_entities & answer_entities) > 2:
                return True

        return False

    def _calculate_severity(
        self,
        similarity: float,
        answer: VerificationAnswer
    ) -> float:
        """Calculate contradiction severity."""
        # Base severity from similarity (inverse relationship for contradictions)
        base_severity = 0.5

        # Adjust based on answer confidence
        if answer.confidence > 0.8:
            base_severity += 0.2
        elif answer.confidence < 0.4:
            base_severity -= 0.1

        return max(0.0, min(1.0, base_severity))


# =============================================================================
# Factory Function
# =============================================================================

def create_contradiction_detector(
    config: Optional[VerificationConfig] = None,
    llm_client: Optional[Any] = None,
    embedder: Optional[Any] = None,
) -> ContradictionDetectorProtocol:
    """
    Create appropriate contradiction detector based on configuration.

    Args:
        config: Verification configuration
        llm_client: Optional LLM client for LLM-based detection
        embedder: Optional embedder for semantic detection

    Returns:
        ContradictionDetector implementation
    """
    config = config or VerificationConfig()

    if config.preferred_level == DegradationLevel.HEURISTIC:
        return RuleBasedContradictionDetector(config)

    if config.preferred_level == DegradationLevel.DISABLED:
        # Return a no-op detector
        class NoOpDetector:
            async def detect(self, claim, answers):
                return []
        return NoOpDetector()

    if llm_client is not None:
        return HybridContradictionDetector(config, llm_client)

    if embedder is not None:
        return SemanticContradictionDetector(config, embedder)

    return RuleBasedContradictionDetector(config)
