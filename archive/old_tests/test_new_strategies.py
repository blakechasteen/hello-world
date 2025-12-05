"""
Tests for Phase 2 Strategies

Tests the three new strategies implemented in Phase 2:
- challenge (Adversarial Prompting)
- optimize (Recursive Optimization)
- reverse (Reverse Prompting)
"""

import pytest
import sys
from pathlib import Path

# Add promptly_skills to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from HoloLoom.prompting.strategy import StrategyContext, StrategyCategory
from HoloLoom.prompting.registry import get_strategy, get_registry

# Import strategies directly
from promptly_skills.strategies.challenge.strategy import ChallengeStrategy
from promptly_skills.strategies.optimize.strategy import OptimizeStrategy
from promptly_skills.strategies.reverse.strategy import ReverseStrategy


# Tests for ChallengeStrategy

def test_challenge_properties():
    """Test challenge strategy properties"""
    strategy = ChallengeStrategy()

    assert strategy.name == "challenge"
    assert strategy.category == StrategyCategory.SELF_CORRECTION
    assert "Adversarial" in strategy.description


@pytest.mark.asyncio
async def test_challenge_enhancement():
    """Test challenge strategy enhances query"""
    strategy = ChallengeStrategy()
    context = StrategyContext(query="review security architecture")

    result = await strategy.enhance(context)

    # Check adversarial language
    assert "adversarial" in result.enhanced_query.lower()
    assert "attack" in result.enhanced_query.lower()
    assert "problem" in result.enhanced_query.lower()

    # Check metadata
    assert result.metadata['strategy'] == 'challenge'
    assert result.metadata['min_problems'] == 5
    assert result.confidence > 0.9


def test_challenge_auto_detection_security():
    """Test challenge auto-detects security contexts"""
    strategy = ChallengeStrategy()

    # High confidence for security
    context = StrategyContext(query="review security vulnerability")
    confidence = strategy.can_apply(context)
    assert confidence > 0.7

    # Very high for multiple keywords
    context = StrategyContext(query="penetration test authentication system")
    confidence = strategy.can_apply(context)
    assert confidence > 0.8


def test_challenge_auto_detection_general():
    """Test challenge has low confidence for non-security"""
    strategy = ChallengeStrategy()

    # Low confidence for general tasks
    context = StrategyContext(query="write documentation")
    confidence = strategy.can_apply(context)
    assert confidence < 0.5


def test_challenge_file_path_context():
    """Test challenge detects security file paths"""
    strategy = ChallengeStrategy()

    context = StrategyContext(
        query="review this code",
        file_path="/app/security/auth.py"
    )
    confidence = strategy.can_apply(context)

    # Should have higher confidence due to file path
    assert confidence > 0.5


# Tests for OptimizeStrategy

def test_optimize_properties():
    """Test optimize strategy properties"""
    strategy = OptimizeStrategy()

    assert strategy.name == "optimize"
    assert strategy.category == StrategyCategory.META_PROMPTING
    assert "Recursive" in strategy.description or "Optimization" in strategy.description


@pytest.mark.asyncio
async def test_optimize_enhancement():
    """Test optimize strategy enhances query"""
    strategy = OptimizeStrategy()
    context = StrategyContext(query="help me write code")

    result = await strategy.enhance(context)

    # Check iteration structure
    assert "Iteration 1" in result.enhanced_query
    assert "Iteration 2" in result.enhanced_query
    assert "Iteration 3" in result.enhanced_query
    assert "constraints" in result.enhanced_query.lower()
    assert "ambiguities" in result.enhanced_query.lower()

    # Check metadata
    assert result.metadata['strategy'] == 'optimize'
    assert result.metadata['iterations'] == 3
    assert result.confidence > 0.8


def test_optimize_auto_detection_explicit():
    """Test optimize auto-detects optimization requests"""
    strategy = OptimizeStrategy()

    # High confidence for explicit optimization
    context = StrategyContext(query="optimize this prompt")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.7

    # High for improvement
    context = StrategyContext(query="improve this query")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.7


def test_optimize_auto_detection_creation():
    """Test optimize has medium confidence for creation"""
    strategy = OptimizeStrategy()

    # Medium confidence for creation tasks
    context = StrategyContext(query="write a function")
    confidence = strategy.can_apply(context)
    assert 0.4 < confidence < 0.7


def test_optimize_long_query_penalty():
    """Test optimize applies penalty for very long queries"""
    strategy = OptimizeStrategy()

    # Long query (might already be well-structured)
    long_query = "optimize " + "x" * 200
    context = StrategyContext(query=long_query)
    confidence = strategy.can_apply(context)

    # Should still be positive but penalized
    assert confidence > 0
    assert confidence < 1.0


# Tests for ReverseStrategy

def test_reverse_properties():
    """Test reverse strategy properties"""
    strategy = ReverseStrategy()

    assert strategy.name == "reverse"
    assert strategy.category == StrategyCategory.META_PROMPTING
    assert "Reverse" in strategy.description


@pytest.mark.asyncio
async def test_reverse_enhancement():
    """Test reverse strategy enhances query"""
    strategy = ReverseStrategy()
    context = StrategyContext(query="help me understand SQL")

    result = await strategy.enhance(context)

    # Check reverse prompt structure
    assert "prompt engineer" in result.enhanced_query.lower()
    assert "design" in result.enhanced_query.lower()
    assert "ANALYSIS" in result.enhanced_query or "analysis" in result.enhanced_query.lower()

    # Check for 7-component framework mention
    assert "component" in result.enhanced_query.lower() or "role" in result.enhanced_query.lower()

    # Check metadata
    assert result.metadata['strategy'] == 'reverse'
    assert result.confidence > 0.9


def test_reverse_auto_detection_explicit():
    """Test reverse auto-detects prompt design requests"""
    strategy = ReverseStrategy()

    # Very high confidence for explicit prompt design
    context = StrategyContext(query="design a prompt for code review")
    confidence = strategy.can_apply(context)
    assert confidence > 0.8

    # High for meta questions
    context = StrategyContext(query="what's the best prompt for SQL optimization")
    confidence = strategy.can_apply(context)
    assert confidence > 0.7


def test_reverse_auto_detection_meta():
    """Test reverse detects meta-level questions"""
    strategy = ReverseStrategy()

    # Medium-high for meta questions
    context = StrategyContext(query="how should I ask about Python")
    confidence = strategy.can_apply(context)
    assert confidence >= 0.5


def test_reverse_auto_detection_general():
    """Test reverse has low confidence for direct questions"""
    strategy = ReverseStrategy()

    # Low confidence for direct, specific questions
    context = StrategyContext(query="what is the capital of France")
    confidence = strategy.can_apply(context)
    assert confidence < 0.5


# Integration tests

def test_all_strategies_registered():
    """Test all Phase 2 strategies can be registered"""
    registry = get_registry()

    # Manually register strategies
    registry.register(ChallengeStrategy())
    registry.register(OptimizeStrategy())
    registry.register(ReverseStrategy())

    # Verify all registered
    assert 'challenge' in registry
    assert 'optimize' in registry
    assert 'reverse' in registry


def test_strategies_composable():
    """Test Phase 2 strategies can be composed"""
    challenge = ChallengeStrategy()
    optimize = OptimizeStrategy()
    reverse = ReverseStrategy()

    # Test composition with +
    composite1 = challenge + optimize
    assert composite1.name == "challenge+optimize"
    assert len(composite1) == 2

    # Test chaining
    composite2 = reverse + optimize + challenge
    assert composite2.name == "reverse+optimize+challenge"
    assert len(composite2) == 3


@pytest.mark.asyncio
async def test_composite_execution():
    """Test composed strategies execute in order"""
    optimize = OptimizeStrategy()
    challenge = ChallengeStrategy()

    # Create composite
    composite = optimize + challenge

    context = StrategyContext(query="review security")
    result = await composite.enhance(context)

    # Should have pipeline metadata
    assert 'pipeline' in result.metadata
    assert result.metadata['pipeline'] == ['optimize', 'challenge']

    # Should have both strategies in output
    assert 'steps' in result.metadata
    assert len(result.metadata['steps']) == 2


def test_strategy_categories():
    """Test Phase 2 strategies have correct categories"""
    challenge = ChallengeStrategy()
    optimize = OptimizeStrategy()
    reverse = ReverseStrategy()

    # Challenge is self-correction
    assert challenge.category == StrategyCategory.SELF_CORRECTION

    # Optimize and reverse are meta-prompting
    assert optimize.category == StrategyCategory.META_PROMPTING
    assert reverse.category == StrategyCategory.META_PROMPTING


@pytest.mark.asyncio
async def test_all_strategies_produce_results():
    """Test all Phase 2 strategies produce valid results"""
    strategies = [
        ChallengeStrategy(),
        OptimizeStrategy(),
        ReverseStrategy()
    ]

    context = StrategyContext(query="test query")

    for strategy in strategies:
        result = await strategy.enhance(context)

        # All should produce results
        assert result.enhanced_query is not None
        assert len(result.enhanced_query) > len(context.query)
        assert result.metadata['strategy'] == strategy.name
        assert 0 <= result.confidence <= 1.0


def test_confidence_scoring_ranges():
    """Test all strategies return valid confidence scores"""
    strategies = [
        ChallengeStrategy(),
        OptimizeStrategy(),
        ReverseStrategy()
    ]

    contexts = [
        StrategyContext(query="security review"),
        StrategyContext(query="optimize prompt"),
        StrategyContext(query="design a prompt"),
        StrategyContext(query="write documentation"),
    ]

    for strategy in strategies:
        for context in contexts:
            confidence = strategy.can_apply(context)

            # All confidence scores should be 0-1
            assert 0 <= confidence <= 1.0, (
                f"{strategy.name} returned invalid confidence {confidence} "
                f"for query '{context.query}'"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
