"""
Unit tests for Production LLM Consolidator (Week 3).

Tests:
- LLM client initialization (OpenAI, Anthropic, Ollama, vLLM)
- Cost tracking and statistics
- Fact extraction (LLM + fallback)
- Entity extraction (LLM + fallback)
- Deduplication (LLM + fallback)
- Error handling and graceful degradation
- Multi-provider support

Based on LangMem research: "LLM extracts semantic facts from episodic memories"
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from HoloLoom.memory.llm_consolidator import (
    ProductionLLMConsolidator,
    LLMConfig,
    LLMClient,
    LLMUsage,
    calculate_cost,
    create_llm_config,
    create_production_consolidator
)
from HoloLoom.Documentation.types import MemoryShard


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_episodes():
    """Create sample episodic memories."""
    return [
        MemoryShard(
            id="ep1",
            text="Python is a high-level programming language. It supports multiple paradigms.",
            entities=["Python", "ProgrammingLanguage"],
            metadata={"timestamp": datetime.now().isoformat()}
        ),
        MemoryShard(
            id="ep2",
            text="TypeScript is a superset of JavaScript. It adds static typing.",
            entities=["TypeScript", "JavaScript"],
            metadata={"timestamp": datetime.now().isoformat()}
        ),
        MemoryShard(
            id="ep3",
            text="Python is widely used for data science. NumPy and Pandas are popular libraries.",
            entities=["Python", "NumPy", "Pandas"],
            metadata={"timestamp": datetime.now().isoformat()}
        )
    ]


# ============================================================================
# Test: Cost Calculation
# ============================================================================

def test_calculate_cost_gpt4():
    """Test cost calculation for GPT-4."""
    # GPT-4-turbo: $10/1M prompt tokens, $30/1M completion tokens
    cost = calculate_cost("gpt-4-turbo", prompt_tokens=1000, completion_tokens=500)

    expected = (1000 / 1_000_000) * 10.0 + (500 / 1_000_000) * 30.0
    assert abs(cost - expected) < 0.0001  # $0.025


def test_calculate_cost_gpt35():
    """Test cost calculation for GPT-3.5."""
    # GPT-3.5-turbo: $0.5/1M prompt tokens, $1.5/1M completion tokens
    cost = calculate_cost("gpt-3.5-turbo", prompt_tokens=10000, completion_tokens=5000)

    expected = (10000 / 1_000_000) * 0.5 + (5000 / 1_000_000) * 1.5
    assert abs(cost - expected) < 0.00001  # $0.0125


def test_calculate_cost_claude():
    """Test cost calculation for Claude."""
    # Claude-3-sonnet: $3/1M prompt tokens, $15/1M completion tokens
    cost = calculate_cost("claude-3-sonnet", prompt_tokens=2000, completion_tokens=1000)

    expected = (2000 / 1_000_000) * 3.0 + (1000 / 1_000_000) * 15.0
    assert abs(cost - expected) < 0.00001  # $0.021


def test_calculate_cost_ollama_free():
    """Test that Ollama is free (local)."""
    cost = calculate_cost("ollama", prompt_tokens=100000, completion_tokens=50000)
    assert cost == 0.0


# ============================================================================
# Test: LLM Config Factory
# ============================================================================

def test_create_llm_config_default():
    """Test LLM config creation with defaults."""
    config = create_llm_config(provider="openai")

    assert config.provider == "openai"
    assert config.model == "gpt-3.5-turbo"  # Default OpenAI model
    assert config.max_tokens == 500
    assert config.temperature == 0.3


def test_create_llm_config_custom_model():
    """Test LLM config with custom model."""
    config = create_llm_config(provider="openai", model="gpt-4-turbo")

    assert config.model == "gpt-4-turbo"


def test_create_llm_config_anthropic():
    """Test Anthropic config defaults."""
    config = create_llm_config(provider="anthropic")

    assert config.provider == "anthropic"
    assert config.model == "claude-3-haiku-20240307"  # Default Anthropic model


def test_create_llm_config_ollama():
    """Test Ollama config defaults."""
    config = create_llm_config(provider="ollama", base_url="http://localhost:11434")

    assert config.provider == "ollama"
    assert config.model == "llama2"
    assert config.base_url == "http://localhost:11434"


# ============================================================================
# Test: Production Consolidator Factory
# ============================================================================

def test_create_production_consolidator_none():
    """Test creating consolidator with no LLM (rule-based only)."""
    consolidator = create_production_consolidator(provider="none")

    assert consolidator.llm_client is None
    assert consolidator.enable_fallback is True


def test_create_production_consolidator_openai():
    """Test creating consolidator with OpenAI."""
    # Will fail to initialize without API key, but should create object
    consolidator = create_production_consolidator(
        provider="openai",
        enable_fallback=True
    )

    assert consolidator.config.provider == "openai"
    assert consolidator.enable_fallback is True


# ============================================================================
# Test: Rule-Based Fallback (No LLM)
# ============================================================================

@pytest.mark.asyncio
async def test_extract_facts_fallback(sample_episodes):
    """Test rule-based fact extraction (no LLM)."""
    consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)

    facts = await consolidator.extract_facts(sample_episodes)

    assert len(facts) > 0
    # Should extract unique sentences
    assert any("Python" in fact for fact in facts)
    assert any("TypeScript" in fact for fact in facts)


@pytest.mark.asyncio
async def test_extract_entities_fallback(sample_episodes):
    """Test rule-based entity extraction (no LLM)."""
    consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)

    edges = await consolidator.extract_entities(sample_episodes)

    assert len(edges) > 0
    # Should create edges between consecutive entities
    assert any(src == "Python" and dst == "ProgrammingLanguage" for src, dst, _ in edges)


@pytest.mark.asyncio
async def test_deduplicate_fallback():
    """Test rule-based deduplication (no LLM)."""
    consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)

    # Create duplicate memories
    memories = [
        MemoryShard(id="m1", text="Duplicate text", metadata={"timestamp": datetime.now().isoformat()}),
        MemoryShard(id="m2", text="Duplicate text", metadata={"timestamp": datetime.now().isoformat()}),
        MemoryShard(id="m3", text="Unique text", metadata={"timestamp": datetime.now().isoformat()})
    ]

    unique = await consolidator.deduplicate(memories)

    assert len(unique) == 2  # Only 2 unique texts
    assert "Unique text" in [m.text for m in unique]


@pytest.mark.asyncio
async def test_extract_facts_empty_episodes():
    """Test fact extraction with empty episode list."""
    consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)

    facts = await consolidator.extract_facts([])

    assert facts == []


# ============================================================================
# Test: LLM Client Mock (Simulated API Responses)
# ============================================================================

@pytest.mark.asyncio
async def test_extract_facts_with_mock_llm(sample_episodes):
    """Test fact extraction with mocked LLM response."""
    config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    consolidator = ProductionLLMConsolidator(config=config, enable_fallback=True)

    # Mock LLM response
    mock_completion = """
• Python is a high-level programming language
• TypeScript is a superset of JavaScript that adds static typing
• Python is widely used for data science with NumPy and Pandas
    """

    # Mock the LLM client
    consolidator.llm_client = MagicMock()
    consolidator.llm_client.client = MagicMock()  # Client exists (not None)
    consolidator.llm_client.complete = AsyncMock(return_value=mock_completion)

    facts = await consolidator.extract_facts(sample_episodes)

    # Should parse bullet points into facts
    assert len(facts) == 3
    assert "Python is a high-level programming language" in facts
    assert "TypeScript is a superset of JavaScript" in facts[1]


@pytest.mark.asyncio
async def test_extract_entities_with_mock_llm(sample_episodes):
    """Test entity extraction with mocked LLM response."""
    config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    consolidator = ProductionLLMConsolidator(config=config, enable_fallback=True)

    # Mock LLM response with relationship format
    mock_completion = """
• Python → IS_A → ProgrammingLanguage
• TypeScript → IS_A → ProgrammingLanguage
• TypeScript → EXTENDS → JavaScript
• Python → USES → NumPy
• Python → USES → Pandas
    """

    consolidator.llm_client = MagicMock()
    consolidator.llm_client.client = MagicMock()
    consolidator.llm_client.complete = AsyncMock(return_value=mock_completion)

    edges = await consolidator.extract_entities(sample_episodes)

    # Should parse relationships
    assert len(edges) == 5
    assert ("Python", "ProgrammingLanguage", "IS_A") in edges
    assert ("TypeScript", "JavaScript", "EXTENDS") in edges


@pytest.mark.asyncio
async def test_deduplicate_with_mock_llm():
    """Test deduplication with mocked LLM response."""
    config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    consolidator = ProductionLLMConsolidator(config=config, enable_fallback=True)

    memories = [
        MemoryShard(id="m1", text="Python is great", metadata={"timestamp": datetime.now().isoformat()}),
        MemoryShard(id="m2", text="Python is awesome", metadata={"timestamp": datetime.now().isoformat()}),
        MemoryShard(id="m3", text="TypeScript is useful", metadata={"timestamp": datetime.now().isoformat()})
    ]

    # Mock LLM response with unique IDs
    mock_completion = """
Unique memories to keep:
1. [ID: m1] Python is great (most informative)
2. [ID: m3] TypeScript is useful
    """

    consolidator.llm_client = MagicMock()
    consolidator.llm_client.client = MagicMock()
    consolidator.llm_client.complete = AsyncMock(return_value=mock_completion)

    unique = await consolidator.deduplicate(memories)

    # Should keep only m1 and m3
    assert len(unique) == 2
    unique_ids = {m.id for m in unique}
    assert "m1" in unique_ids
    assert "m3" in unique_ids
    assert "m2" not in unique_ids  # Duplicate removed


# ============================================================================
# Test: Error Handling and Graceful Degradation
# ============================================================================

@pytest.mark.asyncio
async def test_extract_facts_llm_error_fallback(sample_episodes):
    """Test that LLM errors trigger fallback to rule-based."""
    config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    consolidator = ProductionLLMConsolidator(config=config, enable_fallback=True)

    # Mock LLM to raise error
    consolidator.llm_client = MagicMock()
    consolidator.llm_client.client = MagicMock()
    consolidator.llm_client.complete = AsyncMock(side_effect=Exception("API error"))

    facts = await consolidator.extract_facts(sample_episodes)

    # Should fallback to rule-based
    assert len(facts) > 0
    assert any("Python" in fact for fact in facts)


@pytest.mark.asyncio
async def test_extract_facts_llm_none_fallback(sample_episodes):
    """Test that None LLM response triggers fallback."""
    config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    consolidator = ProductionLLMConsolidator(config=config, enable_fallback=True)

    # Mock LLM to return None
    consolidator.llm_client = MagicMock()
    consolidator.llm_client.client = MagicMock()
    consolidator.llm_client.complete = AsyncMock(return_value=None)

    facts = await consolidator.extract_facts(sample_episodes)

    # Should fallback to rule-based
    assert len(facts) > 0


@pytest.mark.asyncio
async def test_extract_facts_no_fallback():
    """Test that disabling fallback returns empty list on LLM failure."""
    config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    consolidator = ProductionLLMConsolidator(config=config, enable_fallback=False)

    consolidator.llm_client = MagicMock()
    consolidator.llm_client.client = MagicMock()
    consolidator.llm_client.complete = AsyncMock(return_value=None)

    facts = await consolidator.extract_facts([
        MemoryShard(id="ep1", text="Test", metadata={"timestamp": datetime.now().isoformat()})
    ])

    # Should return empty list (no fallback)
    assert facts == []


# ============================================================================
# Test: Statistics Tracking
# ============================================================================

def test_get_statistics_no_llm():
    """Test statistics with no LLM (rule-based only)."""
    consolidator = ProductionLLMConsolidator(config=None, enable_fallback=True)

    stats = consolidator.get_statistics()

    assert stats["provider"] == "none"
    assert stats["total_requests"] == 0
    assert stats["total_cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_get_statistics_with_usage():
    """Test statistics after LLM usage."""
    config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    consolidator = ProductionLLMConsolidator(config=config, enable_fallback=True)

    # Mock LLM client with usage tracking
    consolidator.llm_client = MagicMock()
    consolidator.llm_client.get_statistics = MagicMock(return_value={
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "total_requests": 5,
        "total_tokens": 15000,
        "total_cost_usd": 0.0225,
        "avg_tokens_per_request": 3000
    })

    stats = consolidator.get_statistics()

    assert stats["provider"] == "openai"
    assert stats["total_requests"] == 5
    assert stats["total_tokens"] == 15000
    assert stats["total_cost_usd"] == 0.0225


# ============================================================================
# Test: LLM Client (Direct Testing)
# ============================================================================

def test_llm_client_init_none():
    """Test LLM client with 'none' provider."""
    config = LLMConfig(provider="none", model="")
    client = LLMClient(config)

    assert client.client is None
    assert client.total_cost_usd == 0.0


@pytest.mark.asyncio
async def test_llm_client_complete_none():
    """Test LLM client completion with no provider."""
    config = LLMConfig(provider="none", model="")
    client = LLMClient(config)

    result = await client.complete("Test prompt")

    assert result is None  # No client available


def test_llm_client_usage_tracking():
    """Test LLM usage tracking."""
    config = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    client = LLMClient(config)

    # Manually record usage
    client._record_usage(
        prompt_tokens=1000,
        completion_tokens=500,
        cost_usd=0.00125
    )

    assert len(client.usage_history) == 1
    assert client.total_cost_usd == 0.00125

    stats = client.get_statistics()
    assert stats["total_requests"] == 1
    assert stats["total_tokens"] == 1500


# ============================================================================
# Test: Integration with Different Providers
# ============================================================================

@pytest.mark.asyncio
async def test_consolidator_with_different_providers():
    """Test consolidator creation with different providers."""
    # OpenAI
    consolidator_openai = create_production_consolidator(provider="openai")
    assert consolidator_openai.config.provider == "openai"

    # Anthropic
    consolidator_anthropic = create_production_consolidator(provider="anthropic")
    assert consolidator_anthropic.config.provider == "anthropic"

    # Ollama
    consolidator_ollama = create_production_consolidator(provider="ollama")
    assert consolidator_ollama.config.provider == "ollama"

    # None (rule-based)
    consolidator_none = create_production_consolidator(provider="none")
    assert consolidator_none.llm_client is None
