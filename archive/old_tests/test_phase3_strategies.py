"""
Tests for Phase 3 Strategies

Tests the seven new strategies implemented in Phase 3:
- deep (Deliberate Over-Instruction)
- scaffold (Zero-shot CoT Structure)
- prime (Reference Class Priming)
- teach (Few-shot Edge Cases)
- debate (Multi-Persona Debate)
- temp_sim (Temperature Simulation)
- meta_chain (Meta-Chain Prompting)
"""

import pytest
import sys
from pathlib import Path

# Add promptly_skills to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from HoloLoom.prompting.strategy import StrategyContext, StrategyCategory
from HoloLoom.prompting.registry import get_registry

# Import strategies directly
from promptly_skills.strategies.deep.strategy import DeepStrategy
from promptly_skills.strategies.scaffold.strategy import ScaffoldStrategy
from promptly_skills.strategies.prime.strategy import PrimeStrategy
from promptly_skills.strategies.teach.strategy import TeachStrategy
from promptly_skills.strategies.debate.strategy import DebateStrategy
from promptly_skills.strategies.temp_sim.strategy import TempSimStrategy
from promptly_skills.strategies.meta_chain.strategy import MetaChainStrategy


# Tests for DeepStrategy

def test_deep_properties():
    """Test deep strategy properties"""
    strategy = DeepStrategy()

    assert strategy.name == "deep"
    assert strategy.category == StrategyCategory.META_PROMPTING
    assert "Deliberate" in strategy.description


@pytest.mark.asyncio
async def test_deep_enhancement():
    """Test deep strategy enhances query"""
    strategy = DeepStrategy()
    context = StrategyContext(query="explain neural networks")

    result = await strategy.enhance(context)

    # Check for 7 sections
    assert "Section 1: Fundamentals" in result.enhanced_query
    assert "Section 2: Edge Cases" in result.enhanced_query
    assert "Section 3: Tradeoffs" in result.enhanced_query
    assert "Section 4: Alternatives" in result.enhanced_query
    assert "Section 5: Concrete Examples" in result.enhanced_query
    assert "Section 6: Common Pitfalls" in result.enhanced_query
    assert "Section 7: Best Practices" in result.enhanced_query

    # Check metadata
    assert result.metadata['strategy'] == 'deep'
    assert result.metadata['sections'] == 7
    assert result.confidence > 0.9


def test_deep_auto_detection_explicit():
    """Test deep auto-detects depth requests"""
    strategy = DeepStrategy()

    # High confidence for explicit depth
    context = StrategyContext(query="explain this in depth")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.8

    # Medium for explanation
    context = StrategyContext(query="explain transformers")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.4


def test_deep_auto_detection_brevity_penalty():
    """Test deep applies penalty for brevity keywords"""
    strategy = DeepStrategy()

    # Penalty for "quick"
    context = StrategyContext(query="quick overview of attention")
    confidence = strategy.can_apply(context)
    assert confidence < 0.5  # Should be low due to brevity


# Tests for ScaffoldStrategy

def test_scaffold_properties():
    """Test scaffold strategy properties"""
    strategy = ScaffoldStrategy()

    assert strategy.name == "scaffold"
    assert strategy.category == StrategyCategory.SELF_CORRECTION
    assert "Zero-shot" in strategy.description


@pytest.mark.asyncio
async def test_scaffold_enhancement():
    """Test scaffold strategy enhances query"""
    strategy = ScaffoldStrategy()
    context = StrategyContext(query="solve this problem step by step")

    result = await strategy.enhance(context)

    # Check for 6-step structure
    assert "Step 1: Understand" in result.enhanced_query
    assert "Step 2: Decompose" in result.enhanced_query
    assert "Step 3: Analyze" in result.enhanced_query
    assert "Step 4: Synthesize" in result.enhanced_query
    assert "Step 5: Validate" in result.enhanced_query
    assert "Step 6: Conclude" in result.enhanced_query

    # Check metadata
    assert result.metadata['strategy'] == 'scaffold'
    assert result.metadata['reasoning_steps'] == 6
    assert result.confidence >= 0.9


def test_scaffold_auto_detection():
    """Test scaffold auto-detects step-by-step requests"""
    strategy = ScaffoldStrategy()

    # High confidence for explicit
    context = StrategyContext(query="step by step solution")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.8

    # Medium for problem solving
    context = StrategyContext(query="solve this equation")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.4


# Tests for PrimeStrategy

def test_prime_properties():
    """Test prime strategy properties"""
    strategy = PrimeStrategy()

    assert strategy.name == "prime"
    assert strategy.category == StrategyCategory.META_PROMPTING
    assert "Reference" in strategy.description


@pytest.mark.asyncio
async def test_prime_enhancement():
    """Test prime strategy enhances query"""
    strategy = PrimeStrategy()
    context = StrategyContext(query="write production code")

    result = await strategy.enhance(context)

    # Check for quality standards
    assert "world-class" in result.enhanced_query.lower() or "World-Class" in result.enhanced_query
    assert "quality" in result.enhanced_query.lower() or "Quality" in result.enhanced_query
    assert "benchmark" in result.enhanced_query.lower() or "Benchmark" in result.enhanced_query

    # Check metadata
    assert result.metadata['strategy'] == 'prime'
    assert 'reference_classes' in result.metadata
    assert result.confidence >= 0.9


def test_prime_auto_detection():
    """Test prime auto-detects quality focus"""
    strategy = PrimeStrategy()

    # High for explicit quality
    context = StrategyContext(query="best practices for code")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.8

    # Medium for quality keywords
    context = StrategyContext(query="high quality implementation")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.5


# Tests for TeachStrategy

def test_teach_properties():
    """Test teach strategy properties"""
    strategy = TeachStrategy()

    assert strategy.name == "teach"
    assert strategy.category == StrategyCategory.META_PROMPTING
    assert "Few-shot" in strategy.description


@pytest.mark.asyncio
async def test_teach_enhancement():
    """Test teach strategy enhances query"""
    strategy = TeachStrategy()
    context = StrategyContext(query="show me examples of regex")

    result = await strategy.enhance(context)

    # Check for example structure
    assert "Example 1" in result.enhanced_query
    assert "Example 2" in result.enhanced_query
    assert "Example 3" in result.enhanced_query
    assert "Typical Case" in result.enhanced_query or "typical" in result.enhanced_query.lower()

    # Check metadata
    assert result.metadata['strategy'] == 'teach'
    assert result.metadata['min_examples'] == 3
    assert result.confidence >= 0.8


def test_teach_auto_detection():
    """Test teach auto-detects example requests"""
    strategy = TeachStrategy()

    # High for explicit examples
    context = StrategyContext(query="show examples of pattern matching")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.8

    # Medium for "examples" keyword
    context = StrategyContext(query="examples of edge cases")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.4


# Tests for DebateStrategy

def test_debate_properties():
    """Test debate strategy properties"""
    strategy = DebateStrategy()

    assert strategy.name == "debate"
    assert strategy.category == StrategyCategory.META_PROMPTING
    assert "Multi-Persona" in strategy.description


@pytest.mark.asyncio
async def test_debate_enhancement():
    """Test debate strategy enhances query"""
    strategy = DebateStrategy()
    context = StrategyContext(query="pros and cons of microservices")

    result = await strategy.enhance(context)

    # Check for personas
    assert "Persona 1" in result.enhanced_query or "Optimist" in result.enhanced_query
    assert "Persona 2" in result.enhanced_query or "Skeptic" in result.enhanced_query
    assert "Persona 3" in result.enhanced_query or "Pragmatist" in result.enhanced_query

    # Check metadata
    assert result.metadata['strategy'] == 'debate'
    assert result.metadata['personas'] == 3
    assert result.confidence >= 0.9


def test_debate_auto_detection():
    """Test debate auto-detects tradeoff requests"""
    strategy = DebateStrategy()

    # High for explicit debate
    context = StrategyContext(query="debate the pros and cons")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.8

    # Medium for perspectives
    context = StrategyContext(query="different perspectives on cloud")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.5


# Tests for TempSimStrategy

def test_temp_sim_properties():
    """Test temp_sim strategy properties"""
    strategy = TempSimStrategy()

    assert strategy.name == "temp_sim"
    assert strategy.category == StrategyCategory.META_PROMPTING
    assert "Temperature" in strategy.description


@pytest.mark.asyncio
async def test_temp_sim_enhancement():
    """Test temp_sim strategy enhances query"""
    strategy = TempSimStrategy()
    context = StrategyContext(query="will AI replace programmers?")

    result = await strategy.enhance(context)

    # Check for confidence levels
    assert "High Confidence" in result.enhanced_query or "high confidence" in result.enhanced_query.lower()
    assert "Medium Confidence" in result.enhanced_query or "medium confidence" in result.enhanced_query.lower()
    assert "Low Confidence" in result.enhanced_query or "low confidence" in result.enhanced_query.lower()

    # Check metadata
    assert result.metadata['strategy'] == 'temp_sim'
    assert 'confidence_levels' in result.metadata
    assert result.confidence >= 0.8


def test_temp_sim_auto_detection():
    """Test temp_sim auto-detects uncertainty"""
    strategy = TempSimStrategy()

    # High for explicit uncertainty
    context = StrategyContext(query="how confident are you about this prediction?")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.8

    # Medium for hedging words
    context = StrategyContext(query="might this approach work?")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.3


# Tests for MetaChainStrategy

def test_meta_chain_properties():
    """Test meta_chain strategy properties"""
    strategy = MetaChainStrategy()

    assert strategy.name == "meta_chain"
    assert strategy.category == StrategyCategory.META_PROMPTING
    assert "Meta-Chain" in strategy.description


@pytest.mark.asyncio
async def test_meta_chain_enhancement_learning():
    """Test meta_chain enhances learning query"""
    strategy = MetaChainStrategy()
    context = StrategyContext(query="explain neural networks thoroughly")

    result = await strategy.enhance(context)

    # Check for chained strategies
    assert "deep" in result.enhanced_query.lower()
    assert "teach" in result.enhanced_query.lower() or "example" in result.enhanced_query.lower()
    assert "verify" in result.enhanced_query.lower()

    # Check metadata
    assert result.metadata['strategy'] == 'meta_chain'
    assert 'chained_strategies' in result.metadata
    assert len(result.metadata['chained_strategies']) >= 2
    assert result.confidence >= 0.8


@pytest.mark.asyncio
async def test_meta_chain_enhancement_problem_solving():
    """Test meta_chain enhances problem-solving query"""
    strategy = MetaChainStrategy()
    context = StrategyContext(query="solve this calculus problem")

    result = await strategy.enhance(context)

    # Check for problem-solving chain
    assert "scaffold" in result.enhanced_query.lower() or "step" in result.enhanced_query.lower()

    # Check metadata
    assert result.metadata['strategy'] == 'meta_chain'
    assert result.metadata['query_type'] in ['problem_solving_query', 'Problem Solving Query']


def test_meta_chain_auto_detection():
    """Test meta_chain auto-detection"""
    strategy = MetaChainStrategy()

    # Medium-high for learning query
    context = StrategyContext(query="explain transformers")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.5

    # Medium-high for problem solving
    context = StrategyContext(query="solve this equation")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.5


# Integration Tests

def test_all_phase3_strategies_registered():
    """Test all Phase 3 strategies can be registered"""
    registry = get_registry()

    # Manually register strategies
    registry.register(DeepStrategy())
    registry.register(ScaffoldStrategy())
    registry.register(PrimeStrategy())
    registry.register(TeachStrategy())
    registry.register(DebateStrategy())
    registry.register(TempSimStrategy())
    registry.register(MetaChainStrategy())

    # Verify all registered
    assert 'deep' in registry
    assert 'scaffold' in registry
    assert 'prime' in registry
    assert 'teach' in registry
    assert 'debate' in registry
    assert 'temp_sim' in registry
    assert 'meta_chain' in registry


def test_phase3_strategies_composable():
    """Test Phase 3 strategies can be composed"""
    deep = DeepStrategy()
    teach = TeachStrategy()
    prime = PrimeStrategy()

    # Test composition with +
    composite1 = deep + teach
    assert composite1.name == "deep+teach"
    assert len(composite1) == 2

    # Test chaining
    composite2 = deep + teach + prime
    assert composite2.name == "deep+teach+prime"
    assert len(composite2) == 3


@pytest.mark.asyncio
async def test_phase3_strategies_produce_results():
    """Test all Phase 3 strategies produce valid results"""
    strategies = [
        DeepStrategy(),
        ScaffoldStrategy(),
        PrimeStrategy(),
        TeachStrategy(),
        DebateStrategy(),
        TempSimStrategy(),
        MetaChainStrategy()
    ]

    context = StrategyContext(query="test query")

    for strategy in strategies:
        result = await strategy.enhance(context)

        # All should produce results
        assert result.enhanced_query is not None
        assert len(result.enhanced_query) > len(context.query)
        assert result.metadata['strategy'] == strategy.name
        assert 0 <= result.confidence <= 1.0


def test_phase3_confidence_scoring_ranges():
    """Test all Phase 3 strategies return valid confidence scores"""
    strategies = [
        DeepStrategy(),
        ScaffoldStrategy(),
        PrimeStrategy(),
        TeachStrategy(),
        DebateStrategy(),
        TempSimStrategy(),
        MetaChainStrategy()
    ]

    contexts = [
        StrategyContext(query="explain in depth"),
        StrategyContext(query="step by step"),
        StrategyContext(query="best practices"),
        StrategyContext(query="show examples"),
        StrategyContext(query="pros and cons"),
        StrategyContext(query="uncertain about this"),
        StrategyContext(query="comprehensive analysis"),
        StrategyContext(query="simple question"),
    ]

    for strategy in strategies:
        for context in contexts:
            confidence = strategy.can_apply(context)

            # All confidence scores should be 0-1
            assert 0 <= confidence <= 1.0, (
                f"{strategy.name} returned invalid confidence {confidence} "
                f"for query '{context.query}'"
            )


def test_phase3_strategy_categories():
    """Test Phase 3 strategies have correct categories"""
    deep = DeepStrategy()
    scaffold = ScaffoldStrategy()
    prime = PrimeStrategy()
    teach = TeachStrategy()
    debate = DebateStrategy()
    temp_sim = TempSimStrategy()
    meta_chain = MetaChainStrategy()

    # Most are meta-prompting
    assert deep.category == StrategyCategory.META_PROMPTING
    assert prime.category == StrategyCategory.META_PROMPTING
    assert teach.category == StrategyCategory.META_PROMPTING
    assert debate.category == StrategyCategory.META_PROMPTING
    assert temp_sim.category == StrategyCategory.META_PROMPTING
    assert meta_chain.category == StrategyCategory.META_PROMPTING

    # Scaffold is self-correction (forces structured reasoning)
    assert scaffold.category == StrategyCategory.SELF_CORRECTION


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
