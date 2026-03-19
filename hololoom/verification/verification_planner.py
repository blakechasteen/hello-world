"""
Verification Planner for Chain of Verification.

Generates targeted verification questions for each claim that can be
answered independently without seeing the original response.

The key insight from CoVe research: verification questions must be
specific enough to get definitive answers, but general enough that
they can be answered without knowing the original claim.

Created: 2025-12-05
"""

import logging
import re
from typing import Any

from hololoom.verification.protocol import (
    ClaimType,
    DegradationLevel,
    VerifiableClaim,
    VerificationConfig,
    VerificationPlannerProtocol,
    VerificationQuestion,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Question Templates by Claim Type
# =============================================================================

FACTUAL_TEMPLATES = [
    "What is {entity}?",
    "Is it true that {simplified_claim}?",
    "What are the key facts about {entity}?",
    "Verify: {simplified_claim}",
]

PROCEDURAL_TEMPLATES = [
    "What are the steps involved in {process}?",
    "How does {process} work?",
    "What is the correct procedure for {process}?",
    "What sequence of steps is required for {process}?",
]

CAUSAL_TEMPLATES = [
    "What causes {effect}?",
    "What are the effects of {cause}?",
    "Is there a causal relationship between {cause} and {effect}?",
    "What evidence supports that {cause} leads to {effect}?",
]

DEFINITIONAL_TEMPLATES = [
    "What is the definition of {term}?",
    "How is {term} defined in {domain}?",
    "What does {term} mean?",
    "What is {term}?",
]

COMPARATIVE_TEMPLATES = [
    "How does {entity1} compare to {entity2}?",
    "What are the differences between {entity1} and {entity2}?",
    "Is {entity1} better/faster/more than {entity2}?",
    "What are the relative strengths of {entity1} vs {entity2}?",
]

QUANTITATIVE_TEMPLATES = [
    "What is the value/amount of {metric}?",
    "How many/much {metric}?",
    "What are the numbers for {metric}?",
    "Verify the quantity: {simplified_claim}",
]

TEMPORAL_TEMPLATES = [
    "When did {event} occur?",
    "What is the timeline for {event}?",
    "What happened {time_reference}?",
    "Verify the date/time: {simplified_claim}",
]

DEFAULT_TEMPLATES = [
    "Is it accurate that {simplified_claim}?",
    "Verify the following statement: {simplified_claim}",
    "What evidence supports or contradicts: {simplified_claim}?",
]


# =============================================================================
# Entity and Component Extraction
# =============================================================================

def extract_main_entity(claim: VerifiableClaim) -> str:
    """Extract the main entity/subject from a claim."""
    if claim.entities:
        return claim.entities[0]

    text = claim.text

    # Try to find subject (before main verb)
    subject_match = re.match(r'^([A-Z][^,]*?)\s+(?:is|are|was|were|has|have)\b', text)
    if subject_match:
        return subject_match.group(1).strip()

    # Fall back to first noun phrase
    noun_match = re.match(r'^((?:The|A|An)\s+)?([A-Za-z\s]+?)\s+', text)
    if noun_match:
        return (noun_match.group(1) or '') + noun_match.group(2)

    # Last resort: first few words
    words = text.split()[:4]
    return ' '.join(words)


def extract_process(claim: VerifiableClaim) -> str:
    """Extract the process/procedure from a procedural claim."""
    text = claim.text.lower()

    # Look for process descriptions
    patterns = [
        r'(?:the process of|how to|procedure for)\s+([^.]+)',
        r'(?:steps? (?:for|to|in))\s+([^.]+)',
        r'(?:involves?|requires?)\s+([^.]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    return extract_main_entity(claim)


def extract_cause_effect(claim: VerifiableClaim) -> tuple:
    """Extract cause and effect from a causal claim."""
    text = claim.text

    # Common causal patterns
    patterns = [
        (r'(.+?)\s+(?:causes?|leads? to|results? in)\s+(.+)', 'cause_effect'),
        (r'(.+?)\s+(?:because|since|due to)\s+(.+)', 'effect_cause'),
        (r'(?:If|When)\s+(.+?),?\s+(?:then\s+)?(.+)', 'cause_effect'),
    ]

    for pattern, order in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if order == 'cause_effect':
                return (match.group(1).strip(), match.group(2).strip())
            else:
                return (match.group(2).strip(), match.group(1).strip())

    # Fall back to splitting on common words
    entity = extract_main_entity(claim)
    return (entity, "the observed outcome")


def extract_comparison_entities(claim: VerifiableClaim) -> tuple:
    """Extract entities being compared."""
    text = claim.text

    # Look for comparison patterns
    patterns = [
        r'(.+?)\s+(?:is|are)\s+(?:more|less|better|worse|faster|slower)\s+than\s+(.+)',
        r'(.+?)\s+(?:compared to|versus|vs\.?)\s+(.+)',
        r'(.+?)\s+(?:and|or)\s+(.+?)\s+(?:are|is)\s+(?:similar|different)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return (match.group(1).strip(), match.group(2).strip())

    # Fall back to first two entities
    if len(claim.entities) >= 2:
        return (claim.entities[0], claim.entities[1])

    entity = extract_main_entity(claim)
    return (entity, "alternatives")


def extract_term_and_domain(claim: VerifiableClaim) -> tuple:
    """Extract term being defined and its domain."""
    text = claim.text

    # Look for definition patterns
    patterns = [
        r'([A-Z][a-zA-Z\s]+?)\s+(?:is defined as|refers to|means)',
        r'([A-Z][a-zA-Z\s]+?)\s+is\s+(?:a|an|the)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            term = match.group(1).strip()
            # Try to identify domain from context
            domain = "this context"
            if claim.entities and len(claim.entities) > 1:
                domain = claim.entities[1]
            return (term, domain)

    return (extract_main_entity(claim), "general usage")


def extract_metric(claim: VerifiableClaim) -> str:
    """Extract the metric/quantity being measured."""
    text = claim.text

    # Look for numeric context
    match = re.search(r'(\d+(?:\.\d+)?%?)\s*([a-zA-Z\s]+)', text)
    if match:
        return match.group(2).strip()

    # Look for quantity words
    match = re.search(r'(?:number of|amount of|percentage of)\s+([^.]+)', text.lower())
    if match:
        return match.group(1).strip()

    return extract_main_entity(claim)


def extract_time_reference(claim: VerifiableClaim) -> str:
    """Extract time reference from temporal claim."""
    text = claim.text

    # Look for dates
    date_match = re.search(r'(?:in\s+)?(\d{4}|\d{1,2}/\d{1,2}/\d{2,4})', text)
    if date_match:
        return date_match.group(1)

    # Look for time words
    time_match = re.search(
        r'(before|after|during|since|until|recently|currently|now|today|yesterday|'
        r'last\s+\w+|next\s+\w+)',
        text.lower()
    )
    if time_match:
        return time_match.group(1)

    return "that time period"


def simplify_claim(claim: VerifiableClaim) -> str:
    """Simplify claim for use in verification questions."""
    text = claim.text

    # Remove hedging language
    text = re.sub(r'\b(I think|It seems|Perhaps|Maybe|Probably)\b', '', text, flags=re.IGNORECASE)

    # Remove unnecessary qualifiers
    text = re.sub(r'\b(very|extremely|significantly|quite|rather)\b', '', text, flags=re.IGNORECASE)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Ensure it ends with proper punctuation
    if not text.endswith(('.', '?', '!')):
        text += '.'

    return text


# =============================================================================
# Rule-Based Verification Planner (HEURISTIC Level)
# =============================================================================

class RuleBasedVerificationPlanner:
    """
    Rule-based planner using templates for different claim types.

    Fast and dependency-free, suitable for HEURISTIC degradation level.
    """

    def __init__(self, config: VerificationConfig | None = None):
        self.config = config or VerificationConfig()

    async def plan(
        self,
        claim: VerifiableClaim,
        context: dict[str, Any] | None = None
    ) -> list[VerificationQuestion]:
        """Generate verification questions using templates."""
        context = context or {}
        questions = []

        # Get templates for claim type
        templates = self._get_templates(claim.claim_type)

        # Fill templates with extracted components
        filled = self._fill_templates(templates, claim)

        # Create VerificationQuestion objects
        for i, question_text in enumerate(filled[:self.config.max_questions_per_claim]):
            question = VerificationQuestion(
                question=question_text,
                claim=claim,
                question_type=self._get_question_type(claim.claim_type),
                expected_answer_type=self._get_expected_answer_type(claim.claim_type),
            )
            questions.append(question)

        return questions

    def _get_templates(self, claim_type: ClaimType) -> list[str]:
        """Get question templates for a claim type."""
        type_templates = {
            ClaimType.FACTUAL: FACTUAL_TEMPLATES,
            ClaimType.PROCEDURAL: PROCEDURAL_TEMPLATES,
            ClaimType.CAUSAL: CAUSAL_TEMPLATES,
            ClaimType.DEFINITIONAL: DEFINITIONAL_TEMPLATES,
            ClaimType.COMPARATIVE: COMPARATIVE_TEMPLATES,
            ClaimType.QUANTITATIVE: QUANTITATIVE_TEMPLATES,
            ClaimType.TEMPORAL: TEMPORAL_TEMPLATES,
        }
        return type_templates.get(claim_type, DEFAULT_TEMPLATES)

    def _fill_templates(
        self,
        templates: list[str],
        claim: VerifiableClaim
    ) -> list[str]:
        """Fill template placeholders with extracted components."""
        filled = []

        # Extract components based on claim type
        entity = extract_main_entity(claim)
        simplified = simplify_claim(claim)

        # Type-specific extractions
        if claim.claim_type == ClaimType.PROCEDURAL:
            process = extract_process(claim)
            components = {'entity': entity, 'process': process, 'simplified_claim': simplified}
        elif claim.claim_type == ClaimType.CAUSAL:
            cause, effect = extract_cause_effect(claim)
            components = {'entity': entity, 'cause': cause, 'effect': effect, 'simplified_claim': simplified}
        elif claim.claim_type == ClaimType.COMPARATIVE:
            entity1, entity2 = extract_comparison_entities(claim)
            components = {'entity': entity, 'entity1': entity1, 'entity2': entity2, 'simplified_claim': simplified}
        elif claim.claim_type == ClaimType.DEFINITIONAL:
            term, domain = extract_term_and_domain(claim)
            components = {'entity': entity, 'term': term, 'domain': domain, 'simplified_claim': simplified}
        elif claim.claim_type == ClaimType.QUANTITATIVE:
            metric = extract_metric(claim)
            components = {'entity': entity, 'metric': metric, 'simplified_claim': simplified}
        elif claim.claim_type == ClaimType.TEMPORAL:
            time_ref = extract_time_reference(claim)
            components = {'entity': entity, 'event': entity, 'time_reference': time_ref, 'simplified_claim': simplified}
        else:
            components = {'entity': entity, 'simplified_claim': simplified}

        for template in templates:
            try:
                # Only fill placeholders that exist in components
                question = template
                for key, value in components.items():
                    question = question.replace('{' + key + '}', value)

                # Skip if template still has unfilled placeholders
                if '{' not in question:
                    filled.append(question)
            except Exception as e:
                logger.debug(f"Template fill failed: {e}")
                continue

        return filled

    def _get_question_type(self, claim_type: ClaimType) -> str:
        """Get question type for a claim type."""
        type_map = {
            ClaimType.FACTUAL: "fact_check",
            ClaimType.PROCEDURAL: "procedure_check",
            ClaimType.CAUSAL: "causation_check",
            ClaimType.DEFINITIONAL: "definition_check",
            ClaimType.COMPARATIVE: "comparison_check",
            ClaimType.QUANTITATIVE: "quantity_check",
            ClaimType.TEMPORAL: "temporal_check",
        }
        return type_map.get(claim_type, "general_check")

    def _get_expected_answer_type(self, claim_type: ClaimType) -> str:
        """Get expected answer type for a claim type."""
        if claim_type in (ClaimType.FACTUAL, ClaimType.CAUSAL):
            return "boolean"
        elif claim_type == ClaimType.QUANTITATIVE:
            return "numeric"
        elif claim_type == ClaimType.TEMPORAL:
            return "date"
        else:
            return "explanation"


# =============================================================================
# LLM-Based Verification Planner (FULL Level)
# =============================================================================

PLANNING_PROMPT = '''Generate verification questions for the following claim.

Claim: {claim_text}
Claim Type: {claim_type}

Generate {num_questions} specific questions that can be used to independently verify this claim.

Requirements:
1. Questions should be answerable WITHOUT seeing the original claim
2. Questions should target specific, verifiable facts
3. Questions should cover different aspects of verification
4. Answers should help determine if the claim is accurate

Return as JSON array:
[
  {
    "question": "The verification question",
    "question_type": "fact_check|consistency_check|source_check",
    "expected_answer_type": "boolean|entity|explanation|numeric"
  }
]

Generate questions:'''


class LLMVerificationPlanner:
    """
    LLM-based verification planner for high-quality question generation.

    Uses language models to generate targeted, context-aware verification
    questions that can independently verify claims.
    """

    def __init__(
        self,
        config: VerificationConfig | None = None,
        llm_client: Any | None = None,
    ):
        self.config = config or VerificationConfig()
        self.llm_client = llm_client
        self._fallback = RuleBasedVerificationPlanner(config)

    async def plan(
        self,
        claim: VerifiableClaim,
        context: dict[str, Any] | None = None
    ) -> list[VerificationQuestion]:
        """Generate verification questions using LLM."""
        context = context or {}
        if self.llm_client is None:
            logger.warning("No LLM client available, falling back to rule-based planning")
            return await self._fallback.plan(claim, context)

        try:
            # Build prompt
            prompt = PLANNING_PROMPT.format(
                claim_text=claim.text,
                claim_type=claim.claim_type.value,
                num_questions=self.config.max_questions_per_claim,
            )

            # Call LLM
            response = await self._call_llm(prompt)

            # Parse response
            questions = self._parse_llm_response(response, claim)

            if not questions:
                logger.warning("LLM returned no questions, falling back to rule-based")
                return await self._fallback.plan(claim, context)

            return questions[:self.config.max_questions_per_claim]

        except Exception as e:
            logger.error(f"LLM planning failed: {e}, falling back to rule-based")
            return await self._fallback.plan(claim, context)

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

    def _parse_llm_response(
        self,
        response: str,
        claim: VerifiableClaim
    ) -> list[VerificationQuestion]:
        """Parse LLM response into verification questions."""
        import json

        questions = []

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
                if not isinstance(item, dict) or 'question' not in item:
                    continue

                question = VerificationQuestion(
                    question=item['question'],
                    claim=claim,
                    question_type=item.get('question_type', 'fact_check'),
                    expected_answer_type=item.get('expected_answer_type', 'explanation'),
                )
                questions.append(question)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")

        return questions


# =============================================================================
# Factory Function
# =============================================================================

def create_verification_planner(
    config: VerificationConfig | None = None,
    llm_client: Any | None = None,
) -> VerificationPlannerProtocol:
    """
    Create appropriate verification planner based on configuration.

    Args:
        config: Verification configuration
        llm_client: Optional LLM client for LLM-based planning

    Returns:
        VerificationPlanner implementation
    """
    config = config or VerificationConfig()

    if config.preferred_level == DegradationLevel.HEURISTIC:
        return RuleBasedVerificationPlanner(config)

    if llm_client is not None:
        return LLMVerificationPlanner(config, llm_client)

    return RuleBasedVerificationPlanner(config)
