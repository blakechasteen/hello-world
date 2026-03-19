"""
Claim Extractor for Chain of Verification.

Extracts verifiable claims from response text, identifying atomic
statements that can be independently verified.

Supports multiple extraction strategies:
- LLM-based: Most accurate, uses language models
- Rule-based: Fast, uses regex and linguistic patterns
- Hybrid: Combines both approaches

Created: 2025-12-05
"""

import logging
import re
from typing import Any

from hololoom.verification.protocol import (
    ClaimExtractorProtocol,
    ClaimType,
    DegradationLevel,
    VerifiableClaim,
    VerificationConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Claim Classification Patterns
# =============================================================================

# Patterns for identifying claim types
FACTUAL_PATTERNS = [
    r'\b(is|are|was|were)\b.*\b(the|a|an)\b',           # X is the Y
    r'\b(has|have|had)\b.*\b(been|become)\b',            # has been/become
    r'\b(named|called|known as)\b',                       # named/called
    r'\b(founded|created|invented|discovered)\b',         # origins
    r'\b(located|situated|based)\b',                      # location
]

PROCEDURAL_PATTERNS = [
    r'\b(first|then|next|finally|step)\b',               # sequence markers
    r'\b(to|in order to|so that)\b.*\b(must|should)\b',  # purpose + requirement
    r'\b(requires?|needs?|involves?)\b',                  # requirements
    r'\b(process|procedure|method|approach)\b',           # process words
]

CAUSAL_PATTERNS = [
    r'\b(because|since|due to|as a result)\b',           # causal connectors
    r'\b(causes?|leads? to|results? in)\b',              # causation verbs
    r'\b(therefore|thus|hence|consequently)\b',           # conclusions
    r'\b(if|when).*\b(then|will)\b',                     # conditionals
]

DEFINITIONAL_PATTERNS = [
    r'\b(is defined as|refers to|means)\b',              # definitions
    r'\b(is a type of|is a kind of|is an example of)\b', # classification
    r'^[A-Z][^.]*\b(is|are)\b[^.]*\.$',                  # X is Y sentences
]

COMPARATIVE_PATTERNS = [
    r'\b(more|less|better|worse|faster|slower)\b',       # comparatives
    r'\b(than|compared to|versus|vs\.?)\b',              # comparison markers
    r'\b(similar|different|same|unlike)\b',              # similarity
]

QUANTITATIVE_PATTERNS = [
    r'\b\d+(\.\d+)?%?\b',                                # numbers/percentages
    r'\b(million|billion|thousand|hundred)\b',            # large numbers
    r'\b(approximately|about|around|nearly)\b.*\d',      # approximations
]

TEMPORAL_PATTERNS = [
    r'\b(in \d{4}|on [A-Z][a-z]+ \d+)\b',               # dates
    r'\b(before|after|during|since|until)\b',            # temporal markers
    r'\b(recently|currently|now|today)\b',               # time references
]


# Sentence splitting pattern
SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Handle edge cases
    if not text.strip():
        return []

    # Split on sentence boundaries
    sentences = SENTENCE_PATTERN.split(text.strip())

    # Clean up sentences
    result = []
    for s in sentences:
        s = s.strip()
        if s and len(s) > 10:  # Filter very short fragments
            result.append(s)

    return result


def classify_claim_type(text: str) -> ClaimType:
    """Classify the type of a claim based on patterns."""
    text_lower = text.lower()

    # Check patterns in order of specificity
    for pattern in QUANTITATIVE_PATTERNS:
        if re.search(pattern, text_lower):
            return ClaimType.QUANTITATIVE

    for pattern in TEMPORAL_PATTERNS:
        if re.search(pattern, text_lower):
            return ClaimType.TEMPORAL

    for pattern in COMPARATIVE_PATTERNS:
        if re.search(pattern, text_lower):
            return ClaimType.COMPARATIVE

    for pattern in CAUSAL_PATTERNS:
        if re.search(pattern, text_lower):
            return ClaimType.CAUSAL

    for pattern in PROCEDURAL_PATTERNS:
        if re.search(pattern, text_lower):
            return ClaimType.PROCEDURAL

    for pattern in DEFINITIONAL_PATTERNS:
        if re.search(pattern, text_lower):
            return ClaimType.DEFINITIONAL

    for pattern in FACTUAL_PATTERNS:
        if re.search(pattern, text_lower):
            return ClaimType.FACTUAL

    return ClaimType.UNKNOWN


def calculate_importance(text: str, position: int, total_length: int) -> float:
    """
    Calculate importance score for a claim.

    Factors:
    - Position (earlier claims often more important)
    - Length (very short or very long claims less important)
    - Presence of key entities
    - Specificity indicators
    """
    score = 0.5  # Base score

    # Position factor: earlier claims slightly more important
    position_ratio = position / max(total_length, 1)
    score += 0.1 * (1.0 - position_ratio)

    # Length factor: prefer medium-length claims
    words = len(text.split())
    if 10 <= words <= 30:
        score += 0.1
    elif words < 5 or words > 50:
        score -= 0.1

    # Specificity: named entities, numbers, dates
    if re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text):  # Proper nouns
        score += 0.1
    if re.search(r'\b\d+\b', text):  # Numbers
        score += 0.05
    if re.search(r'\b(specifically|exactly|precisely)\b', text.lower()):
        score += 0.1

    # Vagueness penalty
    if re.search(r'\b(some|many|various|often|sometimes)\b', text.lower()):
        score -= 0.1

    return max(0.0, min(1.0, score))


def extract_entities(text: str) -> list[str]:
    """Extract named entities from text (simple pattern-based)."""
    entities = []

    # Proper nouns (capitalized words not at sentence start)
    proper_nouns = re.findall(r'(?<!^)(?<![.!?]\s)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    entities.extend(proper_nouns)

    # Technical terms (often CamelCase or with underscores)
    tech_terms = re.findall(r'\b([A-Z][a-z]+[A-Z][a-z]+)\b', text)
    entities.extend(tech_terms)

    # Quoted terms
    quoted = re.findall(r'"([^"]+)"', text)
    entities.extend(quoted)

    return list(set(entities))


# =============================================================================
# Rule-Based Claim Extractor (HEURISTIC Level)
# =============================================================================

class RuleBasedClaimExtractor:
    """
    Rule-based claim extractor using linguistic patterns.

    Fast and dependency-free, suitable for HEURISTIC degradation level.
    """

    def __init__(self, config: VerificationConfig | None = None):
        self.config = config or VerificationConfig()

    async def extract(
        self,
        text: str,
        context: dict[str, Any]
    ) -> list[VerifiableClaim]:
        """Extract claims using rule-based patterns."""
        claims = []

        # Split into sentences
        sentences = split_sentences(text)

        # Track position for importance calculation
        total_chars = len(text)
        current_pos = 0

        for sentence in sentences:
            # Skip non-declarative sentences
            if sentence.endswith('?'):
                current_pos += len(sentence)
                continue

            # Check if sentence contains verifiable content
            if self._is_verifiable(sentence):
                claim_type = classify_claim_type(sentence)
                importance = calculate_importance(sentence, current_pos, total_chars)
                entities = extract_entities(sentence)

                # Find span in original text
                try:
                    start = text.index(sentence)
                    end = start + len(sentence)
                except ValueError:
                    start, end = current_pos, current_pos + len(sentence)

                claim = VerifiableClaim(
                    text=sentence,
                    claim_type=claim_type,
                    confidence=0.5,  # Default confidence
                    source_span=(start, end),
                    importance=importance,
                    entities=entities,
                    metadata={"extraction_method": "rule_based"},
                )
                claims.append(claim)

            current_pos += len(sentence)

        # Sort by importance and limit
        claims.sort(key=lambda c: c.importance, reverse=True)
        return claims[:self.config.max_claims]

    def _is_verifiable(self, sentence: str) -> bool:
        """Check if a sentence contains verifiable content."""
        # Skip questions
        if sentence.strip().endswith('?'):
            return False

        # Skip commands/imperatives
        if re.match(r'^(Please|Let|Note|Remember|Consider)\b', sentence):
            return False

        # Skip meta-statements
        if re.search(r'\b(I think|I believe|In my opinion|It seems)\b', sentence):
            return False

        # Must have subject-verb structure
        if not re.search(r'\b(is|are|was|were|has|have|had|can|could|will|would)\b', sentence.lower()):
            return False

        return True


# =============================================================================
# LLM-Based Claim Extractor (FULL Level)
# =============================================================================

CLAIM_EXTRACTION_PROMPT = '''Extract verifiable claims from the following text.

For each claim, identify:
1. The exact claim text (verbatim from the source)
2. The claim type: factual, procedural, causal, definitional, comparative, quantitative, or temporal
3. Importance score (0.0-1.0): how central is this claim to the overall response?
4. Named entities mentioned

Return as JSON array:
[
  {
    "text": "exact claim text",
    "type": "factual|procedural|causal|definitional|comparative|quantitative|temporal",
    "importance": 0.8,
    "entities": ["Entity1", "Entity2"]
  }
]

Focus on:
- Claims that can be independently verified
- Specific facts rather than opinions
- Central claims, not peripheral details

Text to analyze:
---
{text}
---

Context (if relevant):
Query: {query}

Extract claims:'''


class LLMClaimExtractor:
    """
    LLM-based claim extractor for high-quality extraction.

    Uses language models to understand semantic structure and
    identify verifiable claims with better accuracy than rule-based.
    """

    def __init__(
        self,
        config: VerificationConfig | None = None,
        llm_client: Any | None = None,
    ):
        self.config = config or VerificationConfig()
        self.llm_client = llm_client
        self._fallback = RuleBasedClaimExtractor(config)

    async def extract(
        self,
        text: str,
        context: dict[str, Any]
    ) -> list[VerifiableClaim]:
        """Extract claims using LLM."""
        if self.llm_client is None:
            logger.warning("No LLM client available, falling back to rule-based extraction")
            return await self._fallback.extract(text, context)

        try:
            # Build prompt
            prompt = CLAIM_EXTRACTION_PROMPT.format(
                text=text,
                query=context.get("query", "N/A"),
            )

            # Call LLM
            response = await self._call_llm(prompt)

            # Parse response
            claims = self._parse_llm_response(response, text)

            if not claims:
                logger.warning("LLM returned no claims, falling back to rule-based")
                return await self._fallback.extract(text, context)

            # Limit to max claims
            claims.sort(key=lambda c: c.importance, reverse=True)
            return claims[:self.config.max_claims]

        except Exception as e:
            logger.error(f"LLM claim extraction failed: {e}, falling back to rule-based")
            return await self._fallback.extract(text, context)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM client."""
        # Support different LLM client interfaces
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

    def _parse_llm_response(
        self,
        response: str,
        original_text: str
    ) -> list[VerifiableClaim]:
        """Parse LLM response into claims."""
        import json

        claims = []

        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            data = json.loads(response.strip())

            if not isinstance(data, list):
                data = [data]

            for item in data:
                if not isinstance(item, dict) or 'text' not in item:
                    continue

                # Map type string to enum
                type_str = item.get('type', 'unknown').lower()
                claim_type = {
                    'factual': ClaimType.FACTUAL,
                    'procedural': ClaimType.PROCEDURAL,
                    'causal': ClaimType.CAUSAL,
                    'definitional': ClaimType.DEFINITIONAL,
                    'comparative': ClaimType.COMPARATIVE,
                    'quantitative': ClaimType.QUANTITATIVE,
                    'temporal': ClaimType.TEMPORAL,
                }.get(type_str, ClaimType.UNKNOWN)

                # Find span in original text
                claim_text = item['text']
                try:
                    start = original_text.index(claim_text)
                    end = start + len(claim_text)
                except ValueError:
                    start, end = 0, len(claim_text)

                claim = VerifiableClaim(
                    text=claim_text,
                    claim_type=claim_type,
                    confidence=0.5,
                    source_span=(start, end),
                    importance=float(item.get('importance', 0.5)),
                    entities=item.get('entities', []),
                    metadata={"extraction_method": "llm"},
                )
                claims.append(claim)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")

        return claims


# =============================================================================
# Hybrid Claim Extractor
# =============================================================================

class HybridClaimExtractor:
    """
    Hybrid claim extractor combining LLM and rule-based approaches.

    Uses LLM when available, falls back to rules, and can cross-validate
    claims from both sources.
    """

    def __init__(
        self,
        config: VerificationConfig | None = None,
        llm_client: Any | None = None,
    ):
        self.config = config or VerificationConfig()
        self.llm_extractor = LLMClaimExtractor(config, llm_client)
        self.rule_extractor = RuleBasedClaimExtractor(config)

    async def extract(
        self,
        text: str,
        context: dict[str, Any]
    ) -> list[VerifiableClaim]:
        """Extract claims using hybrid approach."""
        level = self.config.preferred_level

        if level == DegradationLevel.DISABLED:
            return []

        if level == DegradationLevel.HEURISTIC:
            return await self.rule_extractor.extract(text, context)

        # FULL or PARTIAL: try LLM first
        try:
            claims = await self.llm_extractor.extract(text, context)

            # Optionally cross-validate with rules
            if context.get("cross_validate", False):
                rule_claims = await self.rule_extractor.extract(text, context)
                claims = self._merge_claims(claims, rule_claims)

            return claims

        except Exception as e:
            if self.config.fallback_on_error:
                logger.warning(f"Hybrid extraction falling back to rules: {e}")
                return await self.rule_extractor.extract(text, context)
            raise

    def _merge_claims(
        self,
        llm_claims: list[VerifiableClaim],
        rule_claims: list[VerifiableClaim]
    ) -> list[VerifiableClaim]:
        """Merge claims from LLM and rule-based extraction."""
        # Use LLM claims as base
        merged = list(llm_claims)
        llm_texts = {c.text.lower().strip() for c in llm_claims}

        # Add unique rule-based claims
        for claim in rule_claims:
            if claim.text.lower().strip() not in llm_texts:
                # Boost importance if found by both methods
                claim.importance *= 0.8  # Slightly lower for rule-only
                merged.append(claim)
            else:
                # Boost confidence for claims found by both
                for llm_claim in llm_claims:
                    if llm_claim.text.lower().strip() == claim.text.lower().strip():
                        llm_claim.importance = min(1.0, llm_claim.importance + 0.1)

        # Sort by importance and limit
        merged.sort(key=lambda c: c.importance, reverse=True)
        return merged[:self.config.max_claims]


# =============================================================================
# Factory Function
# =============================================================================

def create_claim_extractor(
    config: VerificationConfig | None = None,
    llm_client: Any | None = None,
) -> ClaimExtractorProtocol:
    """
    Create appropriate claim extractor based on configuration.

    Args:
        config: Verification configuration
        llm_client: Optional LLM client for LLM-based extraction

    Returns:
        ClaimExtractor implementation
    """
    config = config or VerificationConfig()

    if config.preferred_level == DegradationLevel.HEURISTIC:
        return RuleBasedClaimExtractor(config)

    if llm_client is not None:
        return HybridClaimExtractor(config, llm_client)

    return RuleBasedClaimExtractor(config)
