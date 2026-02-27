"""
Independent Verifier for Chain of Verification.

CRITICAL COMPONENT: This verifier answers questions WITHOUT seeing
the original response to prevent self-confirmation bias.

Key Research Finding (TACL 2024):
"LLMs are biased toward accepting incorrect solutions that resemble
their own reasoning. Independent verification prevents this bias."

The verifier uses:
1. Knowledge base lookups (HoloLoom memory)
2. Fresh LLM context (never sees original response)
3. External sources when available

Created: 2025-12-05
"""

import logging
from typing import Any, Dict, List, Optional

from hololoom.verification.protocol import (
    DegradationLevel,
    IndependentVerifierProtocol,
    VerificationAnswer,
    VerificationConfig,
    VerificationQuestion,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Verification Prompt (NEVER includes original response)
# =============================================================================

VERIFICATION_PROMPT = '''Answer the following verification question as accurately as possible.

Question: {question}

Requirements:
1. Provide a direct, factual answer
2. Include sources or reasoning for your answer
3. Express confidence in your answer (low/medium/high)
4. If uncertain, say so explicitly

Context information (if available):
{context_info}

Provide your answer as JSON:
{{
  "answer": "Your direct answer to the question",
  "confidence": "low|medium|high",
  "sources": ["source1", "source2"],
  "reasoning": "Brief explanation of how you arrived at this answer"
}}

Answer:'''


BATCH_VERIFICATION_PROMPT = '''Answer the following verification questions as accurately as possible.

Questions:
{questions}

Requirements:
1. Provide direct, factual answers for each question
2. Include sources or reasoning for each answer
3. Express confidence level for each (low/medium/high)
4. If uncertain about any, say so explicitly

Context information (if available):
{context_info}

Provide your answers as JSON array:
[
  {{
    "question_index": 0,
    "answer": "Your direct answer",
    "confidence": "low|medium|high",
    "sources": ["source1"],
    "reasoning": "Brief explanation"
  }},
  ...
]

Answers:'''


# =============================================================================
# Confidence Mapping
# =============================================================================

def confidence_str_to_float(confidence_str: str) -> float:
    """Convert string confidence to float."""
    mapping = {
        'low': 0.3,
        'medium': 0.6,
        'high': 0.9,
        'very low': 0.1,
        'very high': 0.95,
    }
    return mapping.get(confidence_str.lower(), 0.5)


# =============================================================================
# Knowledge Base Verifier (Uses HoloLoom Memory)
# =============================================================================

class KnowledgeBaseVerifier:
    """
    Verifier that uses HoloLoom's knowledge base for verification.

    Performs semantic search in the memory system to find relevant
    information for answering verification questions.
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        memory_backend: Optional[Any] = None,
    ):
        self.config = config or VerificationConfig()
        self.memory_backend = memory_backend

    async def verify(
        self,
        questions: List[VerificationQuestion],
        context: Dict[str, Any]
    ) -> List[VerificationAnswer]:
        """
        Answer verification questions using knowledge base.

        IMPORTANT: Context must NOT include the original response.
        """
        # Safety check: ensure original response is not in context
        if 'original_response' in context:
            logger.warning("Original response found in context - removing to prevent bias")
            context = {k: v for k, v in context.items() if k != 'original_response'}

        answers = []

        for question in questions:
            answer = await self._verify_single(question, context)
            answers.append(answer)

        return answers

    async def _verify_single(
        self,
        question: VerificationQuestion,
        context: Dict[str, Any]
    ) -> VerificationAnswer:
        """Answer a single verification question."""
        # Search knowledge base
        relevant_info = await self._search_knowledge_base(question.question)

        if relevant_info:
            # Formulate answer from knowledge base
            answer_text = self._synthesize_answer(question, relevant_info)
            confidence = self._assess_confidence(relevant_info)
            sources = [info.get('source', 'knowledge_base') for info in relevant_info]
        else:
            answer_text = "Unable to verify - no relevant information found in knowledge base"
            confidence = 0.2
            sources = []

        return VerificationAnswer(
            question=question,
            answer=answer_text,
            confidence=confidence,
            sources=sources,
            reasoning="Answer derived from knowledge base search",
        )

    async def _search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant information."""
        if self.memory_backend is None:
            return []

        try:
            # Try different memory backend interfaces
            if hasattr(self.memory_backend, 'recall'):
                results = await self.memory_backend.recall(query, k=5)
                return [{'text': r.content, 'source': 'memory', 'confidence': r.relevance}
                        for r in results]
            elif hasattr(self.memory_backend, 'search'):
                results = await self.memory_backend.search(query, limit=5)
                return [{'text': r.text, 'source': 'memory', 'confidence': r.score}
                        for r in results]
            else:
                logger.warning(f"Unknown memory backend type: {type(self.memory_backend)}")
                return []
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return []

    def _synthesize_answer(
        self,
        question: VerificationQuestion,
        info: List[Dict[str, Any]]
    ) -> str:
        """Synthesize an answer from knowledge base results."""
        if not info:
            return "No relevant information found"

        # Combine relevant texts
        texts = [item.get('text', '') for item in info if item.get('text')]
        if not texts:
            return "No relevant information found"

        # Simple synthesis: take most relevant text
        return texts[0]

    def _assess_confidence(self, info: List[Dict[str, Any]]) -> float:
        """Assess confidence based on knowledge base results."""
        if not info:
            return 0.2

        # Average confidence of results
        confidences = [item.get('confidence', 0.5) for item in info]
        avg_confidence = sum(confidences) / len(confidences)

        # Boost if multiple sources agree
        if len(info) > 2:
            avg_confidence = min(1.0, avg_confidence + 0.1)

        return avg_confidence

    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize context to remove original response.

        CRITICAL: Removes keys that could leak the original response
        into the verification process, preventing self-confirmation bias.
        """
        # Keys to remove (could contain original response)
        FORBIDDEN_KEYS = {
            'original_response',
            'response',
            'baseline_response',
            'initial_response',
        }

        sanitized = {}
        for key, value in context.items():
            if key.lower() in FORBIDDEN_KEYS:
                continue
            # Recursively sanitize nested dicts
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_context(value)
            else:
                sanitized[key] = value

        return sanitized


# =============================================================================
# LLM-Based Independent Verifier (FULL Level)
# =============================================================================

class LLMIndependentVerifier:
    """
    LLM-based verifier that answers questions in a fresh context.

    CRITICAL: The LLM context is completely separate from the original
    response generation to prevent self-confirmation bias.
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        llm_client: Optional[Any] = None,
        memory_backend: Optional[Any] = None,
    ):
        self.config = config or VerificationConfig()
        self.llm_client = llm_client
        self.memory_backend = memory_backend
        self._fallback = KnowledgeBaseVerifier(config, memory_backend)

    async def verify(
        self,
        questions: List[VerificationQuestion],
        context: Dict[str, Any]
    ) -> List[VerificationAnswer]:
        """
        Answer verification questions using LLM in fresh context.

        IMPORTANT: Context must NOT include the original response.
        This is enforced to prevent self-confirmation bias.
        """
        # CRITICAL: Remove any trace of original response
        safe_context = self._sanitize_context(context)

        if self.llm_client is None:
            logger.warning("No LLM client available, falling back to knowledge base")
            return await self._fallback.verify(questions, safe_context)

        try:
            # Batch verification for efficiency
            if len(questions) > 1:
                return await self._verify_batch(questions, safe_context)
            else:
                return [await self._verify_single(questions[0], safe_context)]

        except Exception as e:
            logger.error(f"LLM verification failed: {e}, falling back to knowledge base")
            return await self._fallback.verify(questions, safe_context)

    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove any trace of original response from context.

        This is CRITICAL for preventing self-confirmation bias.
        """
        # Keys that could contain original response
        forbidden_keys = {
            'original_response',
            'response',
            'generated_text',
            'output',
            'answer',
            'result',
            'baseline_response',
        }

        safe_context = {}
        for key, value in context.items():
            if key.lower() not in forbidden_keys:
                # Also check if value might be the response
                if isinstance(value, str) and len(value) > 200:
                    # Long strings might be responses - be cautious
                    logger.debug(f"Excluding potentially long response in context key: {key}")
                    continue
                safe_context[key] = value

        return safe_context

    async def _verify_single(
        self,
        question: VerificationQuestion,
        context: Dict[str, Any]
    ) -> VerificationAnswer:
        """Answer a single verification question."""
        # Gather context information (NOT including original response)
        context_info = await self._gather_context(question.question, context)

        # Build prompt
        prompt = VERIFICATION_PROMPT.format(
            question=question.question,
            context_info=context_info or "No additional context available",
        )

        # Call LLM
        response = await self._call_llm(prompt)

        # Parse response
        return self._parse_single_response(response, question)

    async def _verify_batch(
        self,
        questions: List[VerificationQuestion],
        context: Dict[str, Any]
    ) -> List[VerificationAnswer]:
        """Answer multiple verification questions in batch."""
        # Format questions
        questions_text = "\n".join([
            f"{i}. {q.question}"
            for i, q in enumerate(questions)
        ])

        # Gather context information
        context_info = await self._gather_context(questions[0].question, context)

        # Build prompt
        prompt = BATCH_VERIFICATION_PROMPT.format(
            questions=questions_text,
            context_info=context_info or "No additional context available",
        )

        # Call LLM
        response = await self._call_llm(prompt)

        # Parse response
        return self._parse_batch_response(response, questions)

    async def _gather_context(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Gather relevant context from knowledge base."""
        context_parts = []

        # Add any safe context provided
        if context.get('domain'):
            context_parts.append(f"Domain: {context['domain']}")

        if context.get('topic'):
            context_parts.append(f"Topic: {context['topic']}")

        # Search knowledge base
        if self.memory_backend:
            try:
                kb_verifier = KnowledgeBaseVerifier(self.config, self.memory_backend)
                results = await kb_verifier._search_knowledge_base(query)
                if results:
                    kb_info = "\n".join([r.get('text', '')[:200] for r in results[:3]])
                    context_parts.append(f"Relevant knowledge:\n{kb_info}")
            except Exception as e:
                logger.debug(f"Knowledge base context gathering failed: {e}")

        return "\n\n".join(context_parts) if context_parts else ""

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

    def _parse_single_response(
        self,
        response: str,
        question: VerificationQuestion
    ) -> VerificationAnswer:
        """Parse single question response."""
        import json

        try:
            # Handle markdown code blocks
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            data = json.loads(response.strip())

            return VerificationAnswer(
                question=question,
                answer=data.get('answer', response),
                confidence=confidence_str_to_float(data.get('confidence', 'medium')),
                sources=data.get('sources', []),
                reasoning=data.get('reasoning', ''),
            )

        except json.JSONDecodeError:
            # Fall back to raw response
            return VerificationAnswer(
                question=question,
                answer=response.strip(),
                confidence=0.5,
                sources=[],
                reasoning="Raw LLM response (JSON parsing failed)",
            )

    def _parse_batch_response(
        self,
        response: str,
        questions: List[VerificationQuestion]
    ) -> List[VerificationAnswer]:
        """Parse batch response."""
        import json

        answers = []

        try:
            # Handle markdown code blocks
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            data = json.loads(response.strip())

            if not isinstance(data, list):
                data = [data]

            # Map responses to questions
            for item in data:
                idx = item.get('question_index', len(answers))
                if idx < len(questions):
                    answer = VerificationAnswer(
                        question=questions[idx],
                        answer=item.get('answer', ''),
                        confidence=confidence_str_to_float(item.get('confidence', 'medium')),
                        sources=item.get('sources', []),
                        reasoning=item.get('reasoning', ''),
                    )
                    answers.append(answer)

        except json.JSONDecodeError:
            logger.warning("Batch response parsing failed, falling back to individual")
            # Fall back to creating basic answers
            pass

        # Ensure we have answers for all questions
        while len(answers) < len(questions):
            idx = len(answers)
            answers.append(VerificationAnswer(
                question=questions[idx],
                answer="Unable to parse verification response",
                confidence=0.2,
                sources=[],
                reasoning="Response parsing failed",
            ))

        return answers


# =============================================================================
# Hybrid Independent Verifier
# =============================================================================

class HybridIndependentVerifier:
    """
    Hybrid verifier combining LLM and knowledge base approaches.

    Uses LLM as primary source, cross-validates with knowledge base,
    and aggregates confidence across sources.
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        llm_client: Optional[Any] = None,
        memory_backend: Optional[Any] = None,
    ):
        self.config = config or VerificationConfig()
        self.llm_verifier = LLMIndependentVerifier(config, llm_client, memory_backend)
        self.kb_verifier = KnowledgeBaseVerifier(config, memory_backend)

    async def verify(
        self,
        questions: List[VerificationQuestion],
        context: Dict[str, Any]
    ) -> List[VerificationAnswer]:
        """
        Verify using hybrid approach.

        1. Get LLM answers
        2. Cross-validate with knowledge base
        3. Aggregate and adjust confidence
        """
        # Sanitize context
        safe_context = self.llm_verifier._sanitize_context(context)

        # Get LLM answers
        llm_answers = await self.llm_verifier.verify(questions, safe_context)

        # Get knowledge base answers for cross-validation
        if context.get('cross_validate', True):
            kb_answers = await self.kb_verifier.verify(questions, safe_context)
            return self._merge_answers(llm_answers, kb_answers)

        return llm_answers

    def _merge_answers(
        self,
        llm_answers: List[VerificationAnswer],
        kb_answers: List[VerificationAnswer]
    ) -> List[VerificationAnswer]:
        """Merge and cross-validate answers from different sources."""
        merged = []

        for llm_ans, kb_ans in zip(llm_answers, kb_answers):
            # Check if answers agree
            if self._answers_agree(llm_ans.answer, kb_ans.answer):
                # Boost confidence when sources agree
                new_confidence = min(1.0, (llm_ans.confidence + kb_ans.confidence) / 2 + 0.1)
                sources = list(set(llm_ans.sources + kb_ans.sources))
                reasoning = f"LLM and knowledge base agree. {llm_ans.reasoning}"
            else:
                # Reduce confidence when sources disagree
                new_confidence = max(0.1, (llm_ans.confidence + kb_ans.confidence) / 2 - 0.1)
                sources = llm_ans.sources + ['knowledge_base_disagreement']
                reasoning = f"LLM and knowledge base disagree. LLM: {llm_ans.answer}. KB: {kb_ans.answer}"

            merged_answer = VerificationAnswer(
                question=llm_ans.question,
                answer=llm_ans.answer,  # Prefer LLM answer
                confidence=new_confidence,
                sources=sources,
                reasoning=reasoning,
            )
            merged.append(merged_answer)

        return merged

    def _answers_agree(self, answer1: str, answer2: str) -> bool:
        """Check if two answers semantically agree."""
        # Simple check: look for common keywords or negation patterns
        a1_lower = answer1.lower()
        a2_lower = answer2.lower()

        # Check for explicit disagreement
        negation_words = ['not', 'no', 'false', 'incorrect', 'wrong', 'unable']
        a1_negative = any(word in a1_lower for word in negation_words)
        a2_negative = any(word in a2_lower for word in negation_words)

        if a1_negative != a2_negative:
            return False

        # Check for word overlap (simple heuristic)
        words1 = set(a1_lower.split())
        words2 = set(a2_lower.split())
        overlap = len(words1 & words2)
        total = len(words1 | words2)

        if total == 0:
            return True

        return overlap / total > 0.3  # At least 30% word overlap


# =============================================================================
# Factory Function
# =============================================================================

def create_independent_verifier(
    config: Optional[VerificationConfig] = None,
    llm_client: Optional[Any] = None,
    memory_backend: Optional[Any] = None,
) -> IndependentVerifierProtocol:
    """
    Create appropriate independent verifier based on configuration.

    Args:
        config: Verification configuration
        llm_client: Optional LLM client for LLM-based verification
        memory_backend: Optional memory backend for knowledge base lookups

    Returns:
        IndependentVerifier implementation
    """
    config = config or VerificationConfig()

    if config.preferred_level == DegradationLevel.HEURISTIC:
        return KnowledgeBaseVerifier(config, memory_backend)

    if config.preferred_level == DegradationLevel.DISABLED:
        # Return a no-op verifier
        class NoOpVerifier:
            async def verify(self, questions, context):
                return [
                    VerificationAnswer(
                        question=q,
                        answer="Verification disabled",
                        confidence=0.5,
                        sources=[],
                        reasoning="Verification is disabled",
                    )
                    for q in questions
                ]
        return NoOpVerifier()

    if llm_client is not None:
        return HybridIndependentVerifier(config, llm_client, memory_backend)

    return KnowledgeBaseVerifier(config, memory_backend)
