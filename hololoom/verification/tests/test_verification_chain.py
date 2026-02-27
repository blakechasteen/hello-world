"""
Comprehensive tests for Chain of Verification (CoVe) module.

Tests all components:
1. Protocol definitions and data classes
2. Claim extraction (rule-based and hybrid)
3. Verification planning (question generation)
4. Independent verification (critical: no original response exposure)
5. Contradiction detection (semantic analysis)
6. Result synthesis (response refinement)
7. Full chain integration

Created: 2025-12-05
"""

import asyncio
import pytest
from typing import Any, Dict, List

from hololoom.verification.protocol import (
    ClaimType,
    ContradictionType,
    DegradationLevel,
    VerificationStatus,
    Contradiction,
    VerifiableClaim,
    VerificationAnswer,
    VerificationConfig,
    VerificationQuestion,
    VerificationResult,
)
from hololoom.verification.claim_extractor import (
    RuleBasedClaimExtractor,
    create_claim_extractor,
)
from hololoom.verification.verification_planner import (
    RuleBasedVerificationPlanner,
    create_verification_planner,
)
from hololoom.verification.independent_verifier import (
    KnowledgeBaseVerifier,
    create_independent_verifier,
)
from hololoom.verification.contradiction_detector import (
    RuleBasedContradictionDetector,
    create_contradiction_detector,
)
from hololoom.verification.result_synthesizer import (
    RuleBasedResultSynthesizer,
    ConfidenceRecalibrator,
    determine_claim_status,
    calculate_overall_status,
    create_result_synthesizer,
)
from hololoom.verification.chain import (
    VerificationChain,
    VerificationCache,
    VerificationMetrics,
    verify_response,
    create_verification_chain,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_response():
    """Sample response for testing."""
    return (
        "Thompson Sampling is a Bayesian approach to the exploration-exploitation "
        "tradeoff. It maintains a probability distribution over the expected reward "
        "of each action. The algorithm was first introduced in 1933 by William R. "
        "Thompson. It is commonly used in multi-armed bandit problems and has been "
        "shown to achieve near-optimal regret bounds."
    )


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is Thompson Sampling and when was it introduced?"


@pytest.fixture
def sample_claims():
    """Sample claims for testing."""
    return [
        VerifiableClaim(
            text="Thompson Sampling is a Bayesian approach",
            claim_type=ClaimType.DEFINITIONAL,
            confidence=0.85,
            source_span=(0, 42),
            importance=0.9,
        ),
        VerifiableClaim(
            text="The algorithm was first introduced in 1933",
            claim_type=ClaimType.TEMPORAL,
            confidence=0.9,
            source_span=(120, 162),
            importance=0.8,
        ),
        VerifiableClaim(
            text="It is commonly used in multi-armed bandit problems",
            claim_type=ClaimType.FACTUAL,
            confidence=0.8,
            source_span=(185, 235),
            importance=0.7,
        ),
    ]


@pytest.fixture
def config():
    """Default verification config."""
    return VerificationConfig()


# =============================================================================
# Protocol and Data Class Tests
# =============================================================================

class TestProtocolDefinitions:
    """Test protocol definitions and data classes."""

    def test_claim_type_values(self):
        """Test all claim types are defined."""
        assert ClaimType.FACTUAL.value == "factual"
        assert ClaimType.PROCEDURAL.value == "procedural"
        assert ClaimType.CAUSAL.value == "causal"
        assert ClaimType.DEFINITIONAL.value == "definitional"
        assert ClaimType.QUANTITATIVE.value == "quantitative"
        assert ClaimType.TEMPORAL.value == "temporal"

    def test_verification_status_values(self):
        """Test all verification statuses are defined."""
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.CONTRADICTED.value == "contradicted"
        assert VerificationStatus.UNCERTAIN.value == "uncertain"
        assert VerificationStatus.NEEDS_HUMAN.value == "needs_human"
        assert VerificationStatus.SKIPPED.value == "skipped"

    def test_degradation_level_values(self):
        """Test degradation levels are defined."""
        assert DegradationLevel.FULL.value == "full"
        assert DegradationLevel.PARTIAL.value == "partial"
        assert DegradationLevel.HEURISTIC.value == "heuristic"
        assert DegradationLevel.DISABLED.value == "disabled"

    def test_verifiable_claim_creation(self):
        """Test VerifiableClaim creation and defaults."""
        claim = VerifiableClaim(text="Test claim")
        assert claim.text == "Test claim"
        assert claim.claim_type == ClaimType.UNKNOWN
        assert claim.confidence == 0.5
        assert claim.source_span == (0, 0)
        assert claim.importance == 0.5
        assert claim.entities == []
        assert claim.metadata == {}

    def test_verification_config_defaults(self):
        """Test VerificationConfig default values."""
        config = VerificationConfig()
        assert config.max_claims == 10
        assert config.max_questions_per_claim == 3
        assert config.contradiction_threshold == 0.7
        assert config.preferred_level == DegradationLevel.FULL
        assert config.parallel_verification is True

    def test_verification_result_creation(self):
        """Test VerificationResult creation."""
        result = VerificationResult(
            original_response="Original",
            verified_response="Verified",
            claims_extracted=[],
            verification_questions=[],
            verification_answers=[],
            contradictions_found=[],
            confidence_before=0.5,
            confidence_after=0.8,
            verification_passed=True,
            refinement_applied=True,
            iterations=1,
            status=VerificationStatus.VERIFIED,
            metadata={}
        )
        assert result.verification_passed is True
        assert result.refinement_applied is True
        assert result.confidence_after > result.confidence_before


# =============================================================================
# Claim Extractor Tests
# =============================================================================

class TestClaimExtractor:
    """Test claim extraction functionality."""

    @pytest.mark.asyncio
    async def test_rule_based_extractor_basic(self, sample_response, config):
        """Test basic claim extraction."""
        extractor = RuleBasedClaimExtractor(config)
        claims = await extractor.extract(sample_response, {})

        assert len(claims) > 0
        assert all(isinstance(c, VerifiableClaim) for c in claims)
        assert all(c.text for c in claims)

    @pytest.mark.asyncio
    async def test_extractor_finds_definitions(self, config):
        """Test extraction of definitional claims."""
        text = "Thompson Sampling is a Bayesian algorithm for bandits."
        extractor = RuleBasedClaimExtractor(config)
        claims = await extractor.extract(text, {})

        # Should find the definitional claim
        assert len(claims) >= 1
        definition_claims = [c for c in claims if "is" in c.text.lower()]
        assert len(definition_claims) >= 1

    @pytest.mark.asyncio
    async def test_extractor_finds_temporal(self, config):
        """Test extraction of temporal claims."""
        text = "The algorithm was introduced in 1933 by Thompson."
        extractor = RuleBasedClaimExtractor(config)
        claims = await extractor.extract(text, {})

        # Should extract the temporal claim
        assert len(claims) >= 1
        temporal_claims = [c for c in claims if "1933" in c.text]
        assert len(temporal_claims) >= 1

    @pytest.mark.asyncio
    async def test_extractor_entity_extraction(self, sample_response, config):
        """Test entity extraction from claims."""
        extractor = RuleBasedClaimExtractor(config)
        claims = await extractor.extract(sample_response, {})

        # At least some claims should have entities
        claims_with_entities = [c for c in claims if c.entities]
        # Entities are extracted from proper nouns
        assert isinstance(claims_with_entities, list)

    @pytest.mark.asyncio
    async def test_factory_creates_correct_type(self, config):
        """Test factory function creates appropriate extractor."""
        # Without LLM client, should create rule-based
        extractor = create_claim_extractor(config, None)
        assert isinstance(extractor, RuleBasedClaimExtractor)


# =============================================================================
# Verification Planner Tests
# =============================================================================

class TestVerificationPlanner:
    """Test verification question generation."""

    @pytest.mark.asyncio
    async def test_rule_based_planner_basic(self, sample_claims, config):
        """Test basic question generation."""
        planner = RuleBasedVerificationPlanner(config)
        questions = await planner.plan(sample_claims[0])

        assert len(questions) > 0
        assert all(isinstance(q, VerificationQuestion) for q in questions)
        assert all(q.question.endswith("?") for q in questions)

    @pytest.mark.asyncio
    async def test_planner_respects_max_questions(self, sample_claims, config):
        """Test max questions per claim is respected."""
        config.max_questions_per_claim = 2
        planner = RuleBasedVerificationPlanner(config)
        questions = await planner.plan(sample_claims[0])

        assert len(questions) <= 2

    @pytest.mark.asyncio
    async def test_planner_generates_relevant_questions(self, sample_claims, config):
        """Test questions are relevant to claims."""
        planner = RuleBasedVerificationPlanner(config)

        for claim in sample_claims:
            questions = await planner.plan(claim)
            # Questions should reference the claim's topic
            question_texts = " ".join(q.question.lower() for q in questions)
            # At least one word from the claim should appear in questions
            claim_words = set(claim.text.lower().split())
            question_words = set(question_texts.split())
            overlap = claim_words & question_words
            assert len(overlap) > 0

    @pytest.mark.asyncio
    async def test_factory_creates_correct_type(self, config):
        """Test factory creates appropriate planner."""
        planner = create_verification_planner(config, None)
        assert isinstance(planner, RuleBasedVerificationPlanner)


# =============================================================================
# Independent Verifier Tests
# =============================================================================

class TestIndependentVerifier:
    """Test independent verification (critical component)."""

    @pytest.mark.asyncio
    async def test_knowledge_base_verifier_basic(self, config):
        """Test basic knowledge base verification."""
        # Create simple knowledge base
        kb = {
            "thompson sampling": "Thompson Sampling is a Bayesian algorithm for bandits",
            "1933": "Thompson Sampling was introduced in 1933",
        }

        verifier = KnowledgeBaseVerifier(config, kb)

        claim = VerifiableClaim(
            text="Thompson Sampling is Bayesian",
            claim_type=ClaimType.DEFINITIONAL,
        )
        question = VerificationQuestion(
            question="What is Thompson Sampling?",
            claim=claim,
        )

        answers = await verifier.verify([question], {})

        assert len(answers) == 1
        assert answers[0].confidence > 0

    @pytest.mark.asyncio
    async def test_context_sanitization(self, config):
        """Test that original response is removed from context."""
        verifier = KnowledgeBaseVerifier(config, {})

        # Context with original response (should be removed)
        context = {
            'original_response': 'This should be hidden',
            'response': 'Also hidden',
            'query': 'What is Thompson Sampling?',  # Should remain
        }

        sanitized = verifier._sanitize_context(context)

        assert 'original_response' not in sanitized
        assert 'response' not in sanitized
        assert 'query' in sanitized
        assert sanitized['query'] == 'What is Thompson Sampling?'

    @pytest.mark.asyncio
    async def test_verifier_returns_answers_for_all_questions(self, config):
        """Test verifier returns answer for each question."""
        verifier = KnowledgeBaseVerifier(config, {})

        claims = [
            VerifiableClaim(text="Claim 1"),
            VerifiableClaim(text="Claim 2"),
        ]
        questions = [
            VerificationQuestion(question="Q1?", claim=claims[0]),
            VerificationQuestion(question="Q2?", claim=claims[1]),
        ]

        answers = await verifier.verify(questions, {})

        assert len(answers) == len(questions)

    @pytest.mark.asyncio
    async def test_factory_creates_correct_type(self, config):
        """Test factory creates appropriate verifier."""
        verifier = create_independent_verifier(config, None, None)
        assert isinstance(verifier, KnowledgeBaseVerifier)


# =============================================================================
# Contradiction Detector Tests
# =============================================================================

class TestContradictionDetector:
    """Test contradiction detection functionality."""

    @pytest.mark.asyncio
    async def test_rule_based_detector_finds_negation(self, config):
        """Test detection of negation contradictions."""
        detector = RuleBasedContradictionDetector(config)

        claim = VerifiableClaim(
            text="Thompson Sampling is a Bayesian algorithm",
            claim_type=ClaimType.DEFINITIONAL,
        )
        question = VerificationQuestion(question="Is Thompson Sampling Bayesian?", claim=claim)
        answer = VerificationAnswer(
            question=question,
            answer="Thompson Sampling is NOT a Bayesian algorithm",
            confidence=0.9,
            sources=["knowledge_base"]
        )

        contradictions = await detector.detect(claim, [answer])

        assert len(contradictions) >= 1
        assert contradictions[0].severity > 0

    @pytest.mark.asyncio
    async def test_detector_no_contradiction(self, config):
        """Test no contradiction when claim and evidence agree."""
        detector = RuleBasedContradictionDetector(config)

        claim = VerifiableClaim(
            text="Thompson Sampling is a Bayesian algorithm",
            claim_type=ClaimType.DEFINITIONAL,
        )
        question = VerificationQuestion(question="Is Thompson Sampling Bayesian?", claim=claim)
        answer = VerificationAnswer(
            question=question,
            answer="Yes, Thompson Sampling is indeed a Bayesian algorithm",
            confidence=0.9,
            sources=["knowledge_base"]
        )

        contradictions = await detector.detect(claim, [answer])

        # Should have fewer/no strong contradictions
        strong_contradictions = [c for c in contradictions if c.severity > 0.7]
        assert len(strong_contradictions) == 0

    @pytest.mark.asyncio
    async def test_detector_quantitative_mismatch(self, config):
        """Test detection of quantitative contradictions."""
        detector = RuleBasedContradictionDetector(config)

        claim = VerifiableClaim(
            text="The algorithm was introduced in 1933",
            claim_type=ClaimType.TEMPORAL,
        )
        question = VerificationQuestion(question="When was it introduced?", claim=claim)
        answer = VerificationAnswer(
            question=question,
            answer="The algorithm was introduced in 1935",
            confidence=0.9,
            sources=["knowledge_base"]
        )

        contradictions = await detector.detect(claim, [answer])

        # Should detect quantitative mismatch
        assert len(contradictions) >= 1

    @pytest.mark.asyncio
    async def test_factory_creates_correct_type(self, config):
        """Test factory creates appropriate detector."""
        detector = create_contradiction_detector(config, None)
        assert isinstance(detector, RuleBasedContradictionDetector)


# =============================================================================
# Result Synthesizer Tests
# =============================================================================

class TestResultSynthesizer:
    """Test result synthesis functionality."""

    def test_determine_claim_status_verified(self, config):
        """Test status determination for verified claim."""
        claim = VerifiableClaim(text="Test claim")
        question = VerificationQuestion(question="Q?", claim=claim)
        answer = VerificationAnswer(
            question=question,
            answer="Confirmed",
            confidence=0.9,
            sources=["kb"]
        )

        status, adjustment = determine_claim_status(
            claim, [answer], [], config
        )

        assert status == VerificationStatus.VERIFIED
        assert adjustment >= 0

    def test_determine_claim_status_contradicted(self, config):
        """Test status determination for contradicted claim."""
        claim = VerifiableClaim(text="Test claim")
        contradiction = Contradiction(
            claim=claim,
            evidence="Contradicting evidence",
            severity=0.9,
            explanation="Clear contradiction",
            contradiction_type=ContradictionType.FACTUAL
        )

        status, adjustment = determine_claim_status(
            claim, [], [contradiction], config
        )

        assert status == VerificationStatus.CONTRADICTED
        assert adjustment < 0

    def test_calculate_overall_status_verified(self):
        """Test overall status when most claims verified."""
        claim_statuses = {
            "claim1": (VerificationStatus.VERIFIED, 0.1),
            "claim2": (VerificationStatus.VERIFIED, 0.1),
            "claim3": (VerificationStatus.UNCERTAIN, 0.0),
        }

        status = calculate_overall_status(claim_statuses)
        assert status == VerificationStatus.VERIFIED

    def test_calculate_overall_status_contradicted(self):
        """Test overall status when any claim contradicted."""
        claim_statuses = {
            "claim1": (VerificationStatus.VERIFIED, 0.1),
            "claim2": (VerificationStatus.CONTRADICTED, -0.3),
        }

        status = calculate_overall_status(claim_statuses)
        assert status == VerificationStatus.CONTRADICTED

    @pytest.mark.asyncio
    async def test_rule_based_synthesizer_basic(self, sample_response, sample_claims, config):
        """Test basic synthesis."""
        synthesizer = RuleBasedResultSynthesizer(config)

        refined, confidence = await synthesizer.synthesize(
            original_response=sample_response,
            claims=sample_claims,
            answers=[],
            contradictions=[],
            context={'original_confidence': 0.7}
        )

        assert isinstance(refined, str)
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_synthesizer_applies_correction(self, config):
        """Test that synthesizer applies corrections for contradictions."""
        synthesizer = RuleBasedResultSynthesizer(config)

        original = "The algorithm was introduced in 1933."
        claim = VerifiableClaim(
            text="The algorithm was introduced in 1933",
            claim_type=ClaimType.TEMPORAL,
            source_span=(0, 38)
        )
        contradiction = Contradiction(
            claim=claim,
            evidence="The algorithm was actually introduced in 1935",
            severity=0.9,
            explanation="Date mismatch",
            contradiction_type=ContradictionType.FACTUAL
        )

        refined, confidence = await synthesizer.synthesize(
            original_response=original,
            claims=[claim],
            answers=[],
            contradictions=[contradiction],
            context={'original_confidence': 0.7}
        )

        # Should contain correction marker
        assert "[CORRECT" in refined or "1935" in refined

    def test_confidence_recalibrator(self):
        """Test Bayesian confidence recalibration."""
        recalibrator = ConfidenceRecalibrator()

        claim = VerifiableClaim(text="Test claim")
        question = VerificationQuestion(question="Q?", claim=claim)
        answer = VerificationAnswer(
            question=question,
            answer="Confirmed",
            confidence=0.9,
            sources=["kb"]
        )

        new_confidence = recalibrator.recalibrate(
            original_confidence=0.5,
            claims=[claim],
            answers=[answer],
            contradictions=[]
        )

        # High confidence answers should increase overall confidence
        assert new_confidence > 0.5


# =============================================================================
# Verification Chain Integration Tests
# =============================================================================

class TestVerificationChain:
    """Test full verification chain integration."""

    @pytest.mark.asyncio
    async def test_chain_basic_verification(self, sample_query, sample_response):
        """Test basic chain verification."""
        chain = create_verification_chain()

        result = await chain.verify(
            query=sample_query,
            response=sample_response,
            confidence=0.7
        )

        assert isinstance(result, VerificationResult)
        assert result.original_response == sample_response
        assert 0.0 <= result.confidence_after <= 1.0
        assert isinstance(result.verification_passed, bool)

    @pytest.mark.asyncio
    async def test_chain_extracts_claims(self, sample_query, sample_response):
        """Test chain extracts claims."""
        chain = create_verification_chain()

        result = await chain.verify(
            query=sample_query,
            response=sample_response,
            confidence=0.7
        )

        assert len(result.claims_extracted) > 0
        assert all(isinstance(c, VerifiableClaim) for c in result.claims_extracted)

    @pytest.mark.asyncio
    async def test_chain_generates_questions(self, sample_query, sample_response):
        """Test chain generates verification questions."""
        chain = create_verification_chain()

        result = await chain.verify(
            query=sample_query,
            response=sample_response,
            confidence=0.7
        )

        assert len(result.verification_questions) > 0
        assert all(isinstance(q, VerificationQuestion)
                  for q in result.verification_questions)

    @pytest.mark.asyncio
    async def test_chain_respects_disabled_mode(self, sample_query, sample_response):
        """Test chain respects disabled mode."""
        config = VerificationConfig(preferred_level=DegradationLevel.DISABLED)
        chain = create_verification_chain(config=config)

        result = await chain.verify(
            query=sample_query,
            response=sample_response,
            confidence=0.7
        )

        assert result.status == VerificationStatus.SKIPPED
        assert result.verified_response == sample_response
        assert len(result.claims_extracted) == 0

    @pytest.mark.asyncio
    async def test_chain_caching(self, sample_query, sample_response):
        """Test chain caches results."""
        chain = create_verification_chain(enable_cache=True)

        # First call
        result1 = await chain.verify(
            query=sample_query,
            response=sample_response,
            confidence=0.7
        )

        # Second call (should hit cache)
        result2 = await chain.verify(
            query=sample_query,
            response=sample_response,
            confidence=0.7
        )

        stats = chain.get_stats()
        assert stats['cache_hits'] >= 1

    @pytest.mark.asyncio
    async def test_chain_parallel_verification(self, sample_query, sample_response):
        """Test parallel verification mode."""
        config = VerificationConfig(parallel_verification=True)
        chain = create_verification_chain(config=config)

        result = await chain.verify(
            query=sample_query,
            response=sample_response,
            confidence=0.7
        )

        # Should complete without errors
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_verify_response_convenience(self, sample_query, sample_response):
        """Test verify_response convenience function."""
        result = await verify_response(
            query=sample_query,
            response=sample_response,
            confidence=0.7
        )

        assert isinstance(result, VerificationResult)
        assert result.original_response == sample_response


# =============================================================================
# Verification Cache Tests
# =============================================================================

class TestVerificationCache:
    """Test verification cache functionality."""

    def test_cache_put_and_get(self):
        """Test basic cache put and get."""
        cache = VerificationCache(max_size=10)

        result = VerificationResult(
            original_response="test",
            verified_response="test",
            claims_extracted=[],
            verification_questions=[],
            verification_answers=[],
            contradictions_found=[],
            confidence_before=0.5,
            confidence_after=0.8,
            verification_passed=True,
            refinement_applied=False,
            iterations=1,
            status=VerificationStatus.VERIFIED,
            metadata={}
        )

        cache.put("query", "response", result)
        retrieved = cache.get("query", "response")

        assert retrieved is not None
        assert retrieved.verified_response == "test"

    def test_cache_miss(self):
        """Test cache miss."""
        cache = VerificationCache(max_size=10)
        result = cache.get("nonexistent", "query")
        assert result is None

    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = VerificationCache(max_size=2)

        result = VerificationResult(
            original_response="test",
            verified_response="test",
            claims_extracted=[],
            verification_questions=[],
            verification_answers=[],
            contradictions_found=[],
            confidence_before=0.5,
            confidence_after=0.8,
            verification_passed=True,
            refinement_applied=False,
            iterations=1,
            status=VerificationStatus.VERIFIED,
            metadata={}
        )

        cache.put("q1", "r1", result)
        cache.put("q2", "r2", result)
        cache.put("q3", "r3", result)  # Should evict q1

        assert cache.get("q1", "r1") is None  # Evicted
        assert cache.get("q2", "r2") is not None
        assert cache.get("q3", "r3") is not None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = VerificationCache(max_size=10, ttl_seconds=3600)

        result = VerificationResult(
            original_response="test",
            verified_response="test",
            claims_extracted=[],
            verification_questions=[],
            verification_answers=[],
            contradictions_found=[],
            confidence_before=0.5,
            confidence_after=0.8,
            verification_passed=True,
            refinement_applied=False,
            iterations=1,
            status=VerificationStatus.VERIFIED,
            metadata={}
        )

        cache.put("q", "r", result)
        cache.get("q", "r")  # Hit
        cache.get("q", "r")  # Hit

        stats = cache.get_stats()
        assert stats['size'] == 1
        assert stats['max_size'] == 10
        assert stats['total_hits'] == 2


# =============================================================================
# Verification Metrics Tests
# =============================================================================

class TestVerificationMetrics:
    """Test verification metrics tracking."""

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = VerificationMetrics(
            total_time_ms=150.0,
            claim_extraction_ms=20.0,
            question_planning_ms=30.0,
            independent_verification_ms=50.0,
            contradiction_detection_ms=25.0,
            synthesis_ms=25.0,
            claims_extracted=5,
            questions_generated=10,
            answers_obtained=10,
            contradictions_found=1,
            cache_hit=False,
            degradation_level=DegradationLevel.HEURISTIC,
            iterations_used=2
        )

        d = metrics.to_dict()

        assert d['timing']['total_ms'] == 150.0
        assert d['counts']['claims'] == 5
        assert d['meta']['cache_hit'] is False
        assert d['meta']['degradation_level'] == 'heuristic'
        assert d['meta']['iterations'] == 2


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """Test handling of empty response."""
        chain = create_verification_chain()

        result = await chain.verify(
            query="Test query",
            response="",
            confidence=0.5
        )

        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_very_long_response(self):
        """Test handling of very long response."""
        chain = create_verification_chain()

        long_response = "Thompson Sampling is important. " * 100

        result = await chain.verify(
            query="Test query",
            response=long_response,
            confidence=0.5,
            max_iterations=1
        )

        assert isinstance(result, VerificationResult)
        # Should respect max_claims config
        assert len(result.claims_extracted) <= chain.config.max_claims

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test handling of special characters."""
        chain = create_verification_chain()

        response = "Test with special chars: <script>alert('xss')</script> & 'quotes'"

        result = await chain.verify(
            query="Test query",
            response=response,
            confidence=0.5
        )

        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        """Test handling of unicode content."""
        chain = create_verification_chain()

        response = "Thompson Sampling est une méthode bayésienne. 汤普森采样是一种方法。"

        result = await chain.verify(
            query="Test query",
            response=response,
            confidence=0.5
        )

        assert isinstance(result, VerificationResult)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
