"""
Tests for Promptly Strategy Framework

Tests core framework components:
- PromptingStrategy base interface
- StrategyRegistry auto-discovery
- CompositeStrategy chaining
- AutoDetector learning
"""

import pytest
import asyncio
from pathlib import Path

from HoloLoom.prompting.strategy import (
    PromptingStrategy,
    StrategyContext,
    StrategyResult,
    StrategyCategory,
    TemplateStrategy
)
from HoloLoom.prompting.registry import (
    StrategyRegistry,
    get_strategy,
    suggest_strategies
)
from HoloLoom.prompting.composite import (
    CompositeStrategy,
    parse_pipeline
)
from HoloLoom.prompting.auto_detect import AutoDetector


# Mock strategies for testing

class MockVerifyStrategy(PromptingStrategy):
    """Mock verify strategy for testing"""

    @property
    def name(self) -> str:
        return "mock_verify"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.SELF_CORRECTION

    @property
    def description(self) -> str:
        return "Mock verification strategy"

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        return StrategyResult(
            enhanced_query=f"VERIFY: {context.query}",
            metadata={'strategy': 'mock_verify'}
        )

    def can_apply(self, context: StrategyContext) -> float:
        if "analyze" in context.query.lower():
            return 0.9
        return 0.3


class MockChallengeStrategy(PromptingStrategy):
    """Mock challenge strategy for testing"""

    @property
    def name(self) -> str:
        return "mock_challenge"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.SELF_CORRECTION

    @property
    def description(self) -> str:
        return "Mock challenge strategy"

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        return StrategyResult(
            enhanced_query=f"CHALLENGE: {context.query}",
            metadata={'strategy': 'mock_challenge'},
            confidence=0.85
        )

    def can_apply(self, context: StrategyContext) -> float:
        if "security" in context.query.lower():
            return 0.95
        return 0.4


# Tests for StrategyContext

def test_strategy_context_creation():
    """Test StrategyContext can be created"""
    context = StrategyContext(query="test query")
    assert context.query == "test query"
    assert context.user_context is None
    assert context.selection is None


def test_strategy_context_with_query():
    """Test context.with_query creates new context"""
    context = StrategyContext(query="original", selection="code")
    new_context = context.with_query("new query")

    assert new_context.query == "new query"
    assert new_context.selection == "code"  # Preserved
    assert context.query == "original"  # Original unchanged


# Tests for StrategyResult

def test_strategy_result_creation():
    """Test StrategyResult can be created"""
    result = StrategyResult(
        enhanced_query="enhanced",
        metadata={'key': 'value'}
    )
    assert result.enhanced_query == "enhanced"
    assert result.metadata['key'] == 'value'
    assert result.confidence == 1.0  # Default


def test_strategy_result_with_metadata():
    """Test result.with_metadata adds metadata"""
    result = StrategyResult(enhanced_query="test", metadata={'a': 1})
    new_result = result.with_metadata(b=2, c=3)

    assert new_result.metadata == {'a': 1, 'b': 2, 'c': 3}
    assert result.metadata == {'a': 1}  # Original unchanged


# Tests for PromptingStrategy

@pytest.mark.asyncio
async def test_mock_strategy_enhance():
    """Test mock strategy can enhance query"""
    strategy = MockVerifyStrategy()
    context = StrategyContext(query="analyze this")

    result = await strategy.enhance(context)

    assert result.enhanced_query == "VERIFY: analyze this"
    assert result.metadata['strategy'] == 'mock_verify'


def test_mock_strategy_can_apply():
    """Test mock strategy can_apply scoring"""
    strategy = MockVerifyStrategy()

    # High confidence
    context_analyze = StrategyContext(query="analyze this")
    assert strategy.can_apply(context_analyze) == 0.9

    # Low confidence
    context_other = StrategyContext(query="write code")
    assert strategy.can_apply(context_other) == 0.3


def test_strategy_composition():
    """Test strategies can be composed with +"""
    verify = MockVerifyStrategy()
    challenge = MockChallengeStrategy()

    composite = verify + challenge

    assert isinstance(composite, CompositeStrategy)
    assert composite.name == "mock_verify+mock_challenge"


# Tests for TemplateStrategy

def test_template_strategy_creation():
    """Test TemplateStrategy can be created"""
    template = "Query: {query}, Selection: {selection}"
    strategy = TemplateStrategy(template, {'param': 'value'})

    assert strategy.template == template
    assert strategy.config['param'] == 'value'


@pytest.mark.asyncio
async def test_template_strategy_enhance():
    """Test TemplateStrategy formats template"""
    template = "Query: {query}, File: {file_path}"

    class MyTemplateStrategy(TemplateStrategy):
        @property
        def name(self) -> str:
            return "my_template"

        @property
        def category(self) -> StrategyCategory:
            return StrategyCategory.SCAFFOLDING

        @property
        def description(self) -> str:
            return "Test template"

        def can_apply(self, context: StrategyContext) -> float:
            return 0.5

    strategy = MyTemplateStrategy(template, {})
    context = StrategyContext(query="test query", file_path="test.py")

    result = await strategy.enhance(context)

    assert "Query: test query" in result.enhanced_query
    assert "File: test.py" in result.enhanced_query


# Tests for CompositeStrategy

@pytest.mark.asyncio
async def test_composite_strategy_execution():
    """Test CompositeStrategy runs strategies in sequence"""
    verify = MockVerifyStrategy()
    challenge = MockChallengeStrategy()
    composite = CompositeStrategy([verify, challenge])

    context = StrategyContext(query="original query")
    result = await composite.enhance(context)

    # Should apply both strategies
    assert "VERIFY" in result.enhanced_query
    assert "CHALLENGE" in result.enhanced_query

    # Should have pipeline metadata
    assert 'pipeline' in result.metadata
    assert result.metadata['pipeline'] == ['mock_verify', 'mock_challenge']


@pytest.mark.asyncio
async def test_composite_confidence_product():
    """Test CompositeStrategy confidence is product of all"""
    verify = MockVerifyStrategy()
    challenge = MockChallengeStrategy()
    composite = CompositeStrategy([verify, challenge])

    # Challenge returns confidence=0.85
    context = StrategyContext(query="test")
    result = await composite.enhance(context)

    # Confidence should be product (1.0 * 0.85 = 0.85)
    assert result.confidence == 0.85


def test_composite_can_apply_minimum():
    """Test CompositeStrategy can_apply is minimum"""
    verify = MockVerifyStrategy()
    challenge = MockChallengeStrategy()
    composite = CompositeStrategy([verify, challenge])

    # verify=0.3, challenge=0.4 for this query
    context = StrategyContext(query="write code")
    confidence = composite.can_apply(context)

    assert confidence == 0.3  # Minimum of the two


def test_parse_pipeline():
    """Test parse_pipeline creates composite from string"""
    # Register mock strategies
    from HoloLoom.prompting.registry import get_registry
    registry = get_registry()
    registry.register(MockVerifyStrategy())
    registry.register(MockChallengeStrategy())

    # Parse pipeline
    composite = parse_pipeline("mock_verify+mock_challenge")

    assert isinstance(composite, CompositeStrategy)
    assert composite.name == "mock_verify+mock_challenge"
    assert len(composite) == 2


# Tests for StrategyRegistry

def test_registry_creation():
    """Test StrategyRegistry can be created"""
    # Use a custom empty directory for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StrategyRegistry(strategies_dir=Path(tmpdir))
        assert len(registry._lazy_loaded) == 0


def test_registry_manual_registration():
    """Test strategies can be manually registered"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StrategyRegistry(strategies_dir=Path(tmpdir))
        strategy = MockVerifyStrategy()

        registry.register(strategy)

        assert 'mock_verify' in registry
        assert registry.get('mock_verify') == strategy


def test_registry_list_by_category():
    """Test registry can list by category"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StrategyRegistry(strategies_dir=Path(tmpdir))
        verify = MockVerifyStrategy()
        challenge = MockChallengeStrategy()

        registry.register(verify)
        registry.register(challenge)

        self_correction = registry.list_by_category(StrategyCategory.SELF_CORRECTION)
        assert len(self_correction) == 2


def test_registry_suggest():
    """Test registry can suggest strategies"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = StrategyRegistry(strategies_dir=Path(tmpdir))
        verify = MockVerifyStrategy()
        challenge = MockChallengeStrategy()

        registry.register(verify)
        registry.register(challenge)

        # For "analyze" query, verify should score higher
        context = StrategyContext(query="analyze security")
        suggestions = registry.suggest(context, top_k=2)

        assert len(suggestions) == 2
        assert suggestions[0][0] == 'mock_challenge'  # Security keyword
        assert suggestions[0][1] > 0.9


# Tests for AutoDetector

@pytest.mark.asyncio
async def test_auto_detector_creation(tmp_path):
    """Test AutoDetector can be created"""
    feedback_file = tmp_path / "feedback.json"
    detector = AutoDetector(feedback_file=feedback_file)

    assert len(detector.history) == 0
    assert detector.feedback_file == feedback_file


@pytest.mark.asyncio
async def test_auto_detector_without_learning(tmp_path):
    """Test detector returns base suggestions without learning"""
    feedback_file = tmp_path / "feedback.json"
    detector = AutoDetector(feedback_file=feedback_file)

    # Register mock strategies
    detector.registry.register(MockVerifyStrategy())
    detector.registry.register(MockChallengeStrategy())

    context = StrategyContext(query="analyze security")
    suggestions = await detector.detect(context, use_learning=False)

    assert len(suggestions) > 0
    assert all(0.0 <= conf <= 1.0 for _, conf in suggestions)


def test_auto_detector_record_feedback(tmp_path):
    """Test detector can record feedback"""
    feedback_file = tmp_path / "feedback.json"
    detector = AutoDetector(feedback_file=feedback_file)

    context = StrategyContext(query="analyze contract")
    detector.record_feedback(context, 'verify', was_helpful=True)

    assert len(detector.history) == 1
    assert detector.history[0].strategy_name == 'verify'
    assert detector.history[0].was_helpful is True


def test_auto_detector_save_load(tmp_path):
    """Test detector can save and load feedback"""
    feedback_file = tmp_path / "feedback.json"

    # Create detector and record feedback
    detector1 = AutoDetector(feedback_file=feedback_file)
    context = StrategyContext(query="test query")
    detector1.record_feedback(context, 'verify', was_helpful=True)

    # Create new detector (should load feedback)
    detector2 = AutoDetector(feedback_file=feedback_file)

    assert len(detector2.history) == 1
    assert detector2.history[0].strategy_name == 'verify'


def test_auto_detector_stats(tmp_path):
    """Test detector calculates statistics"""
    feedback_file = tmp_path / "feedback.json"
    detector = AutoDetector(feedback_file=feedback_file)

    # Record some feedback
    context = StrategyContext(query="test")
    detector.record_feedback(context, 'verify', was_helpful=True)
    detector.record_feedback(context, 'verify', was_helpful=True)
    detector.record_feedback(context, 'verify', was_helpful=False)
    detector.record_feedback(context, 'challenge', was_helpful=True)

    stats = detector.get_stats()

    assert stats['total_feedback'] == 4
    assert 'verify' in stats['strategy_success_rates']
    assert stats['strategy_success_rates']['verify'] == 2/3  # 2 out of 3
    assert stats['strategy_success_rates']['challenge'] == 1.0


# Integration test

@pytest.mark.asyncio
async def test_full_pipeline_integration():
    """Test full pipeline: registry → composite → execute"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create registry and register strategies
        registry = StrategyRegistry(strategies_dir=Path(tmpdir))
        verify = MockVerifyStrategy()
        challenge = MockChallengeStrategy()
        registry.register(verify)
        registry.register(challenge)

        # Create composite pipeline
        composite = verify + challenge

        # Execute pipeline
        context = StrategyContext(query="analyze security")
        result = await composite.enhance(context)

        # Verify result
        assert "VERIFY" in result.enhanced_query
        assert "CHALLENGE" in result.enhanced_query
        assert result.metadata['composite'] is True
        assert len(result.metadata['steps']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
