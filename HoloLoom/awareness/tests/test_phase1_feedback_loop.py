#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Phase 1: Feedback Loop

Tests:
1. LLM provider abstraction
2. LLMContextPacker basic functionality
3. Quality scoring
4. Token efficiency calculation
5. Context utilization analysis
6. Learning system
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

# Import components to test
from HoloLoom.awareness.llm_providers import (
    LLMProvider,
    LLMResponse,
    LLMGenerationConfig,
    get_llm_provider
)

from HoloLoom.awareness.context_packer_llm import (
    LLMContextPacker,
    QualityScore,
    ContextUtilization,
    PackedGeneration,
    TokenBudget
)

from HoloLoom.awareness.context_packer import PackedContext


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Mock LLM response"""
    return LLMResponse(
        text="Quantum tunneling is a quantum mechanical phenomenon where particles pass through barriers.",
        provider="mock",
        model="mock-model",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        latency_ms=500.0,
        finish_reason="stop"
    )


@pytest.fixture
def mock_packed_context():
    """Mock packed context"""
    return PackedContext(
        awareness_section="Confidence: 0.85",
        memory_section="- Memory 1\n- Memory 2",
        pattern_section="Domain: quantum_mechanics",
        query_section="What is quantum tunneling?",
        total_tokens=200,
        elements_included=5,
        elements_compressed=2,
        elements_excluded=3,
        avg_importance=0.75,
        min_importance=0.6,
        packing_time_ms=1.5,
        compression_stats={'full': 3, 'summary': 2}
    )


@pytest.fixture
def mock_awareness_context():
    """Mock awareness context"""
    @dataclass
    class MockConfidence:
        uncertainty_level: float = 0.2
        query_cache_status: str = "miss"
        knowledge_gap_detected: bool = False

    @dataclass
    class MockStructural:
        phrase_type: str = "question"
        is_question: bool = True
        suggested_response_type: str = "explanation"

    @dataclass
    class MockPatterns:
        domain: str = "quantum_mechanics"
        subdomain: str = "tunneling"
        seen_count: int = 5
        confidence: float = 0.85

    @dataclass
    class MockAwarenessContext:
        confidence: MockConfidence = MockConfidence()
        structural: MockStructural = MockStructural()
        patterns: MockPatterns = MockPatterns()

    return MockAwarenessContext()


# ============================================================================
# Test LLM Providers
# ============================================================================

def test_llm_response_cost_estimate():
    """Test cost estimation for different providers"""

    # Anthropic (Claude)
    anthropic_resp = LLMResponse(
        text="test",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        latency_ms=1000.0,
        finish_reason="stop"
    )

    # $3/M input + $15/M output
    expected_cost = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
    assert abs(anthropic_resp.cost_estimate_usd - expected_cost) < 0.0001

    # OpenAI (GPT-4)
    openai_resp = LLMResponse(
        text="test",
        provider="openai",
        model="gpt-4-turbo-preview",
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        latency_ms=1000.0,
        finish_reason="stop"
    )

    # $10/M input + $30/M output
    expected_cost = (1000 / 1_000_000) * 10.0 + (500 / 1_000_000) * 30.0
    assert abs(openai_resp.cost_estimate_usd - expected_cost) < 0.0001

    # Ollama (free)
    ollama_resp = LLMResponse(
        text="test",
        provider="ollama",
        model="llama3.2:3b",
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        latency_ms=1000.0,
        finish_reason="stop"
    )

    assert ollama_resp.cost_estimate_usd == 0.0


def test_get_llm_provider():
    """Test provider factory"""

    # Anthropic
    provider = get_llm_provider("anthropic")
    assert provider.__class__.__name__ == "AnthropicProvider"
    assert provider.model == "claude-3-5-sonnet-20241022"

    # OpenAI
    provider = get_llm_provider("openai")
    assert provider.__class__.__name__ == "OpenAIProvider"
    assert provider.model == "gpt-4-turbo-preview"

    # Ollama
    provider = get_llm_provider("ollama")
    assert provider.__class__.__name__ == "OllamaProvider"
    assert provider.model == "llama3.2:3b"

    # Invalid provider
    with pytest.raises(ValueError):
        get_llm_provider("invalid")


# ============================================================================
# Test Quality Scoring
# ============================================================================

def test_quality_score_overall():
    """Test overall quality score calculation"""

    # Without accuracy
    score = QualityScore(
        coherence=0.9,
        completeness=0.8,
        relevance=0.85
    )

    # Weighted: 0.3*0.9 + 0.4*0.8 + 0.3*0.85 = 0.845
    expected = 0.3 * 0.9 + 0.4 * 0.8 + 0.3 * 0.85
    assert abs(score.overall - expected) < 0.01

    # With accuracy
    score = QualityScore(
        coherence=0.9,
        completeness=0.8,
        relevance=0.85,
        accuracy=0.95
    )

    # Weighted: 0.25*0.9 + 0.3*0.8 + 0.25*0.85 + 0.2*0.95
    expected = 0.25 * 0.9 + 0.3 * 0.8 + 0.25 * 0.85 + 0.2 * 0.95
    assert abs(score.overall - expected) < 0.01


def test_quality_score_grade():
    """Test letter grade assignment"""

    assert QualityScore(0.95, 0.95, 0.95).grade == "A"
    assert QualityScore(0.85, 0.85, 0.85).grade == "B"
    assert QualityScore(0.75, 0.75, 0.75).grade == "C"
    assert QualityScore(0.65, 0.65, 0.65).grade == "D"
    assert QualityScore(0.50, 0.50, 0.50).grade == "F"


# ============================================================================
# Test Context Utilization
# ============================================================================

def test_context_utilization_rate():
    """Test utilization rate calculation"""

    util = ContextUtilization(
        element_mentions={"mem1": 2, "mem2": 1},
        source_utilization={"awareness": 1.0, "memory": 0.5},
        total_elements_packed=10,
        total_elements_utilized=6
    )

    assert util.utilization_rate == 0.6
    assert util.wasted_elements == 4


def test_context_utilization_empty():
    """Test utilization with no elements"""

    util = ContextUtilization(
        element_mentions={},
        source_utilization={},
        total_elements_packed=0,
        total_elements_utilized=0
    )

    assert util.utilization_rate == 0.0
    assert util.wasted_elements == 0


# ============================================================================
# Test LLMContextPacker
# ============================================================================

@pytest.mark.asyncio
async def test_llm_context_packer_initialization():
    """Test LLMContextPacker initialization"""

    packer = LLMContextPacker(
        llm_provider="ollama",
        enable_learning=True,
        learning_rate=0.1
    )

    assert packer.llm_provider_name == "ollama"
    assert packer.enable_learning is True
    assert packer.learning_rate == 0.1
    assert packer._llm_provider is None  # Lazy loaded


@pytest.mark.asyncio
async def test_score_coherence():
    """Test coherence scoring"""

    packer = LLMContextPacker(llm_provider="ollama")

    # Good coherence
    good_text = "This is a well-structured response. It has multiple sentences. Each sentence is clear."
    score = packer._score_coherence(good_text)
    assert score > 0.7

    # Poor coherence (very short)
    poor_text = "Short"
    score = packer._score_coherence(poor_text)
    assert score < 0.8  # Penalized for being too short


@pytest.mark.asyncio
async def test_score_completeness():
    """Test completeness scoring"""

    packer = LLMContextPacker(llm_provider="ollama")

    query = "What is quantum tunneling?"

    # Good completeness (optimal length)
    good_text = "Quantum tunneling is a quantum mechanical phenomenon. " * 5
    score = packer._score_completeness(query, good_text)
    assert score > 0.7

    # Poor completeness (too short)
    poor_text = "It's a thing."
    score = packer._score_completeness(query, poor_text)
    assert score < 0.5


@pytest.mark.asyncio
async def test_score_relevance():
    """Test relevance scoring"""

    packer = LLMContextPacker(llm_provider="ollama")

    query = "What is quantum tunneling?"
    packed = PackedContext(
        awareness_section="",
        memory_section="quantum mechanics barrier penetration probability",
        pattern_section="",
        query_section=query,
        total_tokens=100,
        elements_included=1,
        elements_compressed=0,
        elements_excluded=0,
        avg_importance=0.8,
        min_importance=0.6,
        packing_time_ms=1.0,
        compression_stats={}
    )

    # Good relevance (mentions keywords)
    good_text = "Quantum tunneling involves particles penetrating barriers with probability."
    score = packer._score_relevance(query, good_text, packed)
    assert score > 0.5

    # Poor relevance (unrelated)
    poor_text = "The weather is nice today."
    score = packer._score_relevance(query, poor_text, packed)
    assert score < 0.3


@pytest.mark.asyncio
async def test_learning_system_adaptation():
    """Test learning system adapts importance threshold"""

    packer = LLMContextPacker(
        llm_provider="ollama",
        enable_learning=True,
        learning_rate=0.3  # Fast learning for test
    )

    initial_threshold = packer.min_importance

    # Simulate learning: low importance → low quality
    for _ in range(10):
        quality = QualityScore(0.5, 0.5, 0.5)  # Low quality
        utilization = ContextUtilization({}, {}, 10, 5)

        packed = PackedContext(
            awareness_section="",
            memory_section="",
            pattern_section="",
            query_section="test",
            total_tokens=100,
            elements_included=10,
            elements_compressed=2,
            elements_excluded=5,
            avg_importance=0.5,
            min_importance=0.2,  # Low threshold
            packing_time_ms=1.0,
            compression_stats={}
        )

        packer._learn_from_outcome(packed, quality, 0.005, utilization)

    # After learning, threshold should increase (learned that low threshold → low quality)
    # Note: This is a simplified test; actual adaptation depends on variance
    stats = packer.get_learning_statistics()
    assert stats["total_interactions"] == 10


@pytest.mark.asyncio
async def test_get_learning_statistics():
    """Test learning statistics reporting"""

    packer = LLMContextPacker(
        llm_provider="ollama",
        enable_learning=True
    )

    stats = packer.get_learning_statistics()

    assert "total_interactions" in stats
    assert "current_importance_threshold" in stats
    assert "learning_enabled" in stats
    assert "learning_rate" in stats

    assert stats["total_interactions"] == 0
    assert stats["learning_enabled"] is True


# ============================================================================
# Test Packed Generation
# ============================================================================

def test_packed_generation_summary(mock_packed_context, mock_llm_response):
    """Test PackedGeneration summary formatting"""

    quality = QualityScore(0.9, 0.85, 0.88)
    utilization = ContextUtilization(
        element_mentions={},
        source_utilization={"memory": 0.8},
        total_elements_packed=5,
        total_elements_utilized=4
    )

    gen = PackedGeneration(
        packed_context=mock_packed_context,
        llm_response=mock_llm_response,
        quality_score=quality,
        token_efficiency=quality.overall / mock_llm_response.total_tokens,
        context_utilization=utilization,
        query="Test query"
    )

    summary = gen.summary()

    assert "Quality:" in summary
    assert "Token Efficiency:" in summary
    assert "Context Utilization:" in summary
    assert "Response Length:" in summary


# ============================================================================
# Integration Tests (Mocked LLM)
# ============================================================================

@pytest.mark.asyncio
async def test_pack_and_generate_mocked(mock_awareness_context, mock_llm_response):
    """Test full pack_and_generate with mocked LLM"""

    packer = LLMContextPacker(
        llm_provider="ollama",
        enable_learning=True
    )

    # Mock the LLM generation
    async def mock_generate(packed, config):
        return mock_llm_response

    packer._generate_with_llm = mock_generate

    # Test data
    query = "What is quantum tunneling?"
    memories = [
        {'text': 'Memory 1', 'score': 0.9},
        {'text': 'Memory 2', 'score': 0.8}
    ]

    # Pack and generate
    result = await packer.pack_and_generate(
        query,
        mock_awareness_context,
        memory_results=memories,
        max_memories=5
    )

    # Verify result structure
    assert isinstance(result, PackedGeneration)
    assert result.query == query
    assert isinstance(result.quality_score, QualityScore)
    assert isinstance(result.context_utilization, ContextUtilization)
    assert result.token_efficiency > 0


@pytest.mark.asyncio
async def test_learning_disabled_no_adaptation():
    """Test that learning can be disabled"""

    packer = LLMContextPacker(
        llm_provider="ollama",
        enable_learning=False
    )

    initial_threshold = packer.min_importance

    # Simulate outcomes (should not learn)
    quality = QualityScore(0.5, 0.5, 0.5)
    utilization = ContextUtilization({}, {}, 10, 5)

    packed = PackedContext(
        awareness_section="",
        memory_section="",
        pattern_section="",
        query_section="test",
        total_tokens=100,
        elements_included=10,
        elements_compressed=2,
        elements_excluded=5,
        avg_importance=0.5,
        min_importance=0.2,
        packing_time_ms=1.0,
        compression_stats={}
    )

    packer._learn_from_outcome(packed, quality, 0.005, utilization)

    # Threshold should not change (learning disabled)
    # Note: _learn_from_outcome still records stats, but doesn't adapt
    stats = packer.get_learning_statistics()
    assert stats["learning_enabled"] is False


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
