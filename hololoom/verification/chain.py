"""
Chain of Verification (CoVe) Orchestrator.

Implements Meta AI's 4-step CoVe methodology:
1. Baseline Response - Generate initial answer (already provided)
2. Verification Planning - Create targeted fact-checking questions
3. Independent Verification - Answer questions WITHOUT seeing original
4. Refined Response - Generate final answer using verified facts

Key Innovation: Independent verification prevents self-confirmation bias.

Created: 2025-12-05
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hololoom.verification.protocol import (
    Contradiction,
    DegradationLevel,
    VerifiableClaim,
    VerificationAnswer,
    VerificationChainProtocol,
    VerificationConfig,
    VerificationQuestion,
    VerificationResult,
    VerificationStatus,
)
from hololoom.verification.claim_extractor import create_claim_extractor
from hololoom.verification.verification_planner import create_verification_planner
from hololoom.verification.independent_verifier import create_independent_verifier
from hololoom.verification.contradiction_detector import create_contradiction_detector
from hololoom.verification.result_synthesizer import (
    create_result_synthesizer,
    calculate_overall_status,
    aggregate_claim_statuses,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Verification Cache
# =============================================================================

@dataclass
class CacheEntry:
    """Single cache entry."""
    result: VerificationResult
    timestamp: float
    hit_count: int = 0


class VerificationCache:
    """
    LRU cache for verification results.

    Caches by query+response hash to avoid re-verifying identical content.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []

    def _compute_key(self, query: str, response: str) -> str:
        """Compute cache key from query and response."""
        content = f"{query}||{response}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, query: str, response: str) -> Optional[VerificationResult]:
        """Get cached result if available and not expired."""
        key = self._compute_key(query, response)

        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check TTL
        if time.time() - entry.timestamp > self.ttl_seconds:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return None

        # Update access order
        entry.hit_count += 1
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        return entry.result

    def put(self, query: str, response: str, result: VerificationResult) -> None:
        """Store result in cache."""
        key = self._compute_key(query, response)

        # Evict if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]

        self._cache[key] = CacheEntry(
            result=result,
            timestamp=time.time(),
            hit_count=0
        )
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hit_count for e in self._cache.values())
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'total_hits': total_hits,
            'ttl_seconds': self.ttl_seconds,
        }


# =============================================================================
# Verification Metrics
# =============================================================================

@dataclass
class VerificationMetrics:
    """Metrics from verification chain execution."""
    total_time_ms: float = 0.0
    claim_extraction_ms: float = 0.0
    question_planning_ms: float = 0.0
    independent_verification_ms: float = 0.0
    contradiction_detection_ms: float = 0.0
    synthesis_ms: float = 0.0

    claims_extracted: int = 0
    questions_generated: int = 0
    answers_obtained: int = 0
    contradictions_found: int = 0

    cache_hit: bool = False
    degradation_level: DegradationLevel = DegradationLevel.FULL
    iterations_used: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timing': {
                'total_ms': self.total_time_ms,
                'claim_extraction_ms': self.claim_extraction_ms,
                'question_planning_ms': self.question_planning_ms,
                'verification_ms': self.independent_verification_ms,
                'contradiction_ms': self.contradiction_detection_ms,
                'synthesis_ms': self.synthesis_ms,
            },
            'counts': {
                'claims': self.claims_extracted,
                'questions': self.questions_generated,
                'answers': self.answers_obtained,
                'contradictions': self.contradictions_found,
            },
            'meta': {
                'cache_hit': self.cache_hit,
                'degradation_level': self.degradation_level.value,
                'iterations': self.iterations_used,
            }
        }


# =============================================================================
# Main Verification Chain
# =============================================================================

class VerificationChain:
    """
    Chain-of-Verification implementation following Meta AI's CoVe methodology.

    4-Step Process:
    1. Extract claims from baseline response
    2. Generate verification questions for each claim
    3. Answer questions INDEPENDENTLY (without seeing original response)
    4. Synthesize verified response, flag contradictions

    Key Features:
    - Independent verification prevents self-confirmation bias
    - Claim-level granularity enables precise correction
    - Iterative refinement for low-confidence results
    - Graceful degradation when LLM unavailable
    - Caching for repeated queries
    - Parallel verification for speed
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        llm_client: Optional[Any] = None,
        knowledge_base: Optional[Any] = None,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: float = 3600.0,
    ):
        """
        Initialize verification chain.

        Args:
            config: Verification configuration
            llm_client: Optional LLM client for full verification
            knowledge_base: Optional knowledge base for verification
            enable_cache: Whether to cache verification results
            cache_size: Maximum cache entries
            cache_ttl: Cache entry time-to-live in seconds
        """
        self.config = config or VerificationConfig()
        self.llm_client = llm_client
        self.knowledge_base = knowledge_base

        # Determine degradation level
        self._degradation_level = self._determine_degradation_level()

        # Create components
        self._claim_extractor = create_claim_extractor(
            self.config, llm_client
        )
        self._verification_planner = create_verification_planner(
            self.config, llm_client
        )
        self._independent_verifier = create_independent_verifier(
            self.config, llm_client, knowledge_base
        )
        self._contradiction_detector = create_contradiction_detector(
            self.config, llm_client
        )
        self._result_synthesizer = create_result_synthesizer(
            self.config, llm_client
        )

        # Cache
        self._cache_enabled = enable_cache
        self._cache = VerificationCache(
            max_size=cache_size,
            ttl_seconds=cache_ttl
        ) if enable_cache else None

        # Metrics
        self._total_verifications = 0
        self._cache_hits = 0

    def _determine_degradation_level(self) -> DegradationLevel:
        """Determine appropriate degradation level based on available resources."""
        if self.config.preferred_level == DegradationLevel.DISABLED:
            return DegradationLevel.DISABLED

        if self.llm_client is not None:
            return DegradationLevel.FULL

        if self.knowledge_base is not None:
            return DegradationLevel.PARTIAL

        return DegradationLevel.HEURISTIC

    async def verify(
        self,
        query: str,
        response: str,
        confidence: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 3,
        confidence_threshold: float = 0.85,
    ) -> VerificationResult:
        """
        Execute full verification chain.

        Args:
            query: Original query
            response: Response to verify
            confidence: Initial confidence estimate
            context: Additional context for verification
            max_iterations: Maximum refinement iterations
            confidence_threshold: Target confidence for early stopping

        Returns:
            VerificationResult with verified response and metadata
        """
        self._total_verifications += 1
        start_time = time.time()
        metrics = VerificationMetrics()
        metrics.degradation_level = self._degradation_level

        context = context or {}
        context['original_confidence'] = confidence
        context['query'] = query

        # Check cache
        if self._cache_enabled and self._cache:
            cached = self._cache.get(query, response)
            if cached is not None:
                self._cache_hits += 1
                metrics.cache_hit = True
                metrics.total_time_ms = (time.time() - start_time) * 1000
                # Return cached result with updated timing
                return cached

        # Check if verification is disabled
        if self._degradation_level == DegradationLevel.DISABLED:
            return self._create_passthrough_result(
                query, response, confidence, metrics, start_time
            )

        # Execute verification chain
        result = await self._execute_verification(
            query=query,
            response=response,
            confidence=confidence,
            context=context,
            max_iterations=max_iterations,
            confidence_threshold=confidence_threshold,
            metrics=metrics,
        )

        metrics.total_time_ms = (time.time() - start_time) * 1000
        result.metadata['metrics'] = metrics.to_dict()

        # Cache result
        if self._cache_enabled and self._cache:
            self._cache.put(query, response, result)

        return result

    async def _execute_verification(
        self,
        query: str,
        response: str,
        confidence: float,
        context: Dict[str, Any],
        max_iterations: int,
        confidence_threshold: float,
        metrics: VerificationMetrics,
    ) -> VerificationResult:
        """Execute the 4-step verification chain."""

        current_response = response
        current_confidence = confidence
        iteration = 0
        all_claims: List[VerifiableClaim] = []
        all_questions: List[VerificationQuestion] = []
        all_answers: List[VerificationAnswer] = []
        all_contradictions: List[Contradiction] = []

        while iteration < max_iterations:
            iteration += 1
            metrics.iterations_used = iteration

            # Step 1: Extract claims
            step_start = time.time()
            claims = await self._claim_extractor.extract(current_response, context)
            metrics.claim_extraction_ms += (time.time() - step_start) * 1000
            metrics.claims_extracted = len(claims)

            if not claims:
                logger.info("No verifiable claims extracted, returning original")
                break

            # Sample claims if too many (based on config)
            if len(claims) > self.config.max_claims:
                claims = self._sample_important_claims(claims, self.config.max_claims)

            all_claims = claims

            # Step 2: Generate verification questions
            step_start = time.time()
            questions = await self._generate_all_questions(claims, context)
            metrics.question_planning_ms += (time.time() - step_start) * 1000
            metrics.questions_generated = len(questions)

            all_questions = questions

            # Step 3: Independent verification (CRITICAL - no original response)
            step_start = time.time()
            sanitized_context = self._sanitize_context(context)
            answers = await self._independent_verifier.verify(questions, sanitized_context)
            metrics.independent_verification_ms += (time.time() - step_start) * 1000
            metrics.answers_obtained = len(answers)

            all_answers = answers

            # Step 4a: Detect contradictions
            step_start = time.time()
            contradictions = await self._detect_all_contradictions(claims, answers)
            metrics.contradiction_detection_ms += (time.time() - step_start) * 1000
            metrics.contradictions_found = len(contradictions)

            all_contradictions = contradictions

            # Step 4b: Synthesize refined response
            step_start = time.time()
            refined_response, new_confidence = await self._result_synthesizer.synthesize(
                original_response=current_response,
                claims=claims,
                answers=answers,
                contradictions=contradictions,
                context=context,
            )
            metrics.synthesis_ms += (time.time() - step_start) * 1000

            current_response = refined_response
            current_confidence = new_confidence

            # Check stopping conditions
            if new_confidence >= confidence_threshold:
                logger.info(f"Confidence threshold met: {new_confidence:.3f} >= {confidence_threshold}")
                break

            if not contradictions:
                logger.info("No contradictions found, stopping verification")
                break

            # Update context for next iteration
            context['previous_contradictions'] = contradictions
            context['iteration'] = iteration

        # Determine overall status
        claim_statuses = aggregate_claim_statuses(
            all_claims, all_answers, all_contradictions, self.config
        )
        overall_status = calculate_overall_status(claim_statuses)
        verification_passed = overall_status in [
            VerificationStatus.VERIFIED,
            VerificationStatus.UNCERTAIN
        ]

        return VerificationResult(
            original_response=response,
            verified_response=current_response,
            claims_extracted=all_claims,
            verification_questions=all_questions,
            verification_answers=all_answers,
            contradictions_found=all_contradictions,
            confidence_before=confidence,
            confidence_after=current_confidence,
            verification_passed=verification_passed,
            refinement_applied=current_response != response,
            status=overall_status,
            iterations=iteration,
            degradation_level=self._degradation_level,
            metadata={
                'query': query,
                'claim_statuses': {c.text: s.value for c, (s, _) in
                                  zip(all_claims, claim_statuses.values())},
            }
        )

    def _sample_important_claims(
        self,
        claims: List[VerifiableClaim],
        max_claims: int
    ) -> List[VerifiableClaim]:
        """Sample the most important claims."""
        if len(claims) <= max_claims:
            return claims

        # Sort by importance (descending)
        sorted_claims = sorted(claims, key=lambda c: c.importance, reverse=True)
        return sorted_claims[:max_claims]

    async def _generate_all_questions(
        self,
        claims: List[VerifiableClaim],
        context: Dict[str, Any] = None
    ) -> List[VerificationQuestion]:
        """Generate verification questions for all claims in parallel."""
        context = context or {}
        if self.config.parallel_verification:
            tasks = [
                self._verification_planner.plan(claim, context)
                for claim in claims
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            questions = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Question generation failed: {result}")
                else:
                    questions.extend(result)
            return questions
        else:
            questions = []
            for claim in claims:
                claim_questions = await self._verification_planner.plan(claim, context)
                questions.extend(claim_questions)
            return questions

    async def _detect_all_contradictions(
        self,
        claims: List[VerifiableClaim],
        answers: List[VerificationAnswer]
    ) -> List[Contradiction]:
        """Detect contradictions between claims and verification answers."""
        if self.config.parallel_verification:
            tasks = [
                self._contradiction_detector.detect(claim, answers)
                for claim in claims
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            contradictions = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Contradiction detection failed: {result}")
                else:
                    contradictions.extend(result)
            return contradictions
        else:
            contradictions = []
            for claim in claims:
                claim_contradictions = await self._contradiction_detector.detect(
                    claim, answers
                )
                contradictions.extend(claim_contradictions)
            return contradictions

    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove original response from context.

        CRITICAL: This is essential for independent verification.
        The verifier must not see the original response.
        """
        forbidden_keys = {
            'original_response', 'response', 'generated_text',
            'output', 'answer', 'result', 'baseline_response',
            'current_response', 'refined_response',
        }

        sanitized = {}
        for key, value in context.items():
            if key.lower() not in forbidden_keys:
                # Also sanitize nested dicts
                if isinstance(value, dict):
                    sanitized[key] = self._sanitize_context(value)
                else:
                    sanitized[key] = value

        return sanitized

    def _create_passthrough_result(
        self,
        query: str,
        response: str,
        confidence: float,
        metrics: VerificationMetrics,
        start_time: float,
    ) -> VerificationResult:
        """Create a passthrough result when verification is disabled."""
        metrics.total_time_ms = (time.time() - start_time) * 1000

        return VerificationResult(
            original_response=response,
            verified_response=response,
            claims_extracted=[],
            verification_questions=[],
            verification_answers=[],
            contradictions_found=[],
            confidence_before=confidence,
            confidence_after=confidence,
            verification_passed=True,
            refinement_applied=False,
            status=VerificationStatus.SKIPPED,
            iterations=0,
            degradation_level=DegradationLevel.DISABLED,
            metadata={
                'query': query,
                'metrics': metrics.to_dict(),
            }
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get chain statistics."""
        cache_stats = self._cache.get_stats() if self._cache else {}
        return {
            'total_verifications': self._total_verifications,
            'cache_hits': self._cache_hits,
            'cache_hit_rate': (
                self._cache_hits / self._total_verifications
                if self._total_verifications > 0 else 0.0
            ),
            'degradation_level': self._degradation_level.value,
            'cache': cache_stats,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

async def verify_response(
    query: str,
    response: str,
    confidence: float = 0.5,
    llm_client: Optional[Any] = None,
    knowledge_base: Optional[Any] = None,
    config: Optional[VerificationConfig] = None,
    max_iterations: int = 3,
) -> VerificationResult:
    """
    Convenience function to verify a single response.

    Args:
        query: Original query
        response: Response to verify
        confidence: Initial confidence estimate
        llm_client: Optional LLM client
        knowledge_base: Optional knowledge base
        config: Verification configuration
        max_iterations: Maximum refinement iterations

    Returns:
        VerificationResult with verified response

    Example:
        result = await verify_response(
            query="What is Thompson Sampling?",
            response="Thompson Sampling is a Bayesian approach...",
            confidence=0.7
        )

        if result.verification_passed:
            print(result.verified_response)
        else:
            print(f"Contradictions found: {len(result.contradictions_found)}")
    """
    chain = VerificationChain(
        config=config,
        llm_client=llm_client,
        knowledge_base=knowledge_base,
        enable_cache=False,  # No caching for one-off verification
    )

    return await chain.verify(
        query=query,
        response=response,
        confidence=confidence,
        max_iterations=max_iterations,
    )


def create_verification_chain(
    config: Optional[VerificationConfig] = None,
    llm_client: Optional[Any] = None,
    knowledge_base: Optional[Any] = None,
    enable_cache: bool = True,
) -> VerificationChain:
    """
    Factory function to create a verification chain.

    Args:
        config: Verification configuration
        llm_client: Optional LLM client for full verification
        knowledge_base: Optional knowledge base
        enable_cache: Whether to enable result caching

    Returns:
        Configured VerificationChain instance

    Example:
        # Full verification with LLM
        chain = create_verification_chain(llm_client=my_llm)

        # Heuristic-only verification (no LLM)
        chain = create_verification_chain()

        # With custom configuration
        config = VerificationConfig(
            max_claims=10,
            parallel_verification=True,
            contradiction_threshold=0.7
        )
        chain = create_verification_chain(config=config, llm_client=my_llm)
    """
    return VerificationChain(
        config=config,
        llm_client=llm_client,
        knowledge_base=knowledge_base,
        enable_cache=enable_cache,
    )


# =============================================================================
# Integration with HoloLoom Agentic System
# =============================================================================

class AgenticVerificationIntegration:
    """
    Integration layer for HoloLoom's agentic reasoning system.

    Enables verification in VERIFY reasoning mode.
    """

    def __init__(
        self,
        chain: Optional[VerificationChain] = None,
        config: Optional[VerificationConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        self.chain = chain or create_verification_chain(
            config=config,
            llm_client=llm_client,
        )

    async def verify_agentic_step(
        self,
        query: str,
        response: str,
        step_confidence: float,
        step_metadata: Dict[str, Any],
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Verify a single agentic reasoning step.

        Args:
            query: Query for this step
            response: Response generated for this step
            step_confidence: Confidence from this step
            step_metadata: Metadata from agentic step

        Returns:
            Tuple of (verified_response, new_confidence, verification_metadata)
        """
        result = await self.chain.verify(
            query=query,
            response=response,
            confidence=step_confidence,
            context=step_metadata,
            max_iterations=2,  # Limited iterations for per-step verification
        )

        verification_metadata = {
            'verification_passed': result.verification_passed,
            'contradictions_count': len(result.contradictions_found),
            'claims_verified': len([
                c for c, s in result.claim_statuses.items()
                if s == VerificationStatus.VERIFIED
            ]),
            'overall_status': result.overall_status.value,
        }

        return (
            result.verified_response,
            result.confidence_after,
            verification_metadata,
        )

    async def verify_research_synthesis(
        self,
        original_query: str,
        synthesis_response: str,
        sub_queries: List[str],
        sub_responses: List[str],
        synthesis_confidence: float,
    ) -> VerificationResult:
        """
        Verify a research mode synthesis that combines multiple sub-query results.

        Args:
            original_query: The original research query
            synthesis_response: The synthesized response
            sub_queries: List of sub-queries that were answered
            sub_responses: Verified responses from sub-queries
            synthesis_confidence: Confidence in the synthesis

        Returns:
            VerificationResult for the synthesis
        """
        context = {
            'sub_queries': sub_queries,
            'sub_responses': sub_responses,
            'is_synthesis': True,
        }

        return await self.chain.verify(
            query=original_query,
            response=synthesis_response,
            confidence=synthesis_confidence,
            context=context,
            max_iterations=3,
            confidence_threshold=0.85,
        )
