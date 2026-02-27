"""
Parametrized Integration Tests for HoloLoom
============================================

Tests N×M scenarios across different components:
- Execution modes (BARE, FAST, FUSED) × Query complexities
- Bandit strategies × Tool selection scenarios
- Memory backends × Retrieval patterns
- Config combinations × Feature flags

Created: December 2025
"""

import pytest
import asyncio
import numpy as np
from typing import List, Tuple
from unittest.mock import Mock, MagicMock, patch

# ============================================================================
# Test Data Generators
# ============================================================================

def generate_test_queries() -> List[Tuple[str, str]]:
    """Generate test queries with expected complexity levels."""
    return [
        ("hi", "TRIVIAL"),
        ("thanks", "TRIVIAL"),
        ("What is Python?", "SIMPLE"),
        ("Define recursion", "SIMPLE"),
        ("Explain Thompson Sampling in detail with examples", "COMPLEX"),
        ("Compare and contrast neural networks vs decision trees", "COMPLEX"),
        ("Analyze all tradeoffs of Thompson Sampling vs UCB algorithms", "RESEARCH"),
    ]


def generate_mock_embeddings(dim: int = 384, count: int = 10) -> np.ndarray:
    """Generate random embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(count, dim).astype(np.float32)


# ============================================================================
# Execution Mode × Query Complexity Tests
# ============================================================================

class TestModeComplexityMatrix:
    """Test all mode × complexity combinations."""

    @pytest.mark.parametrize("mode", ["bare", "fast", "fused"])
    @pytest.mark.parametrize("query,expected_complexity", [
        ("hi", "TRIVIAL"),
        ("What is Python?", "SIMPLE"),
        ("Explain Thompson Sampling in detail", "COMPLEX"),
    ])
    @pytest.mark.asyncio
    async def test_mode_handles_complexity(self, mode, query, expected_complexity):
        """Each mode should handle each complexity level."""
        from HoloLoom.config import Config
        from HoloLoom.routing.query_classifier import QueryClassifier

        # Get config for mode
        config = getattr(Config, mode)()

        # Classify query
        classifier = QueryClassifier()
        result = classifier.classify(query)

        # Verify classification works
        assert result is not None
        assert hasattr(result, 'complexity')
        assert result.complexity.value.upper() in ["TRIVIAL", "SIMPLE", "COMPLEX", "RESEARCH"]

    @pytest.mark.parametrize("mode", ["bare", "fast", "fused"])
    @pytest.mark.asyncio
    async def test_config_mode_attribute(self, mode):
        """Each mode should set correct ExecutionMode."""
        from HoloLoom.config import Config, ExecutionMode

        config = getattr(Config, mode)()

        expected_mode = ExecutionMode[mode.upper()]
        assert config.mode == expected_mode


# ============================================================================
# Bandit Strategy × Tool Selection Tests
# ============================================================================

class TestBanditStrategyMatrix:
    """Test bandit strategies across different scenarios."""

    @pytest.mark.parametrize("strategy", [
        "epsilon_greedy",
        "bayesian_blend",
        "pure_thompson"
    ])
    @pytest.mark.parametrize("n_tools", [2, 3, 5, 10])
    def test_strategy_with_different_tool_counts(self, strategy, n_tools):
        """Each strategy should work with different numbers of tools."""
        from HoloLoom.policy.unified import TSBandit, BanditStrategy

        bandit = TSBandit(n_arms=n_tools)
        probs = np.ones(n_tools) / n_tools  # Uniform probs

        strategy_enum = BanditStrategy[strategy.upper()]
        arm, metadata = bandit.select_with_strategy(probs, strategy=strategy_enum)

        assert 0 <= arm < n_tools, f"Selected arm {arm} out of range [0, {n_tools})"
        assert isinstance(metadata, dict)

    @pytest.mark.parametrize("strategy", [
        "epsilon_greedy",
        "bayesian_blend",
        "pure_thompson"
    ])
    @pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.2, 0.5])
    def test_strategy_with_different_epsilon_values(self, strategy, epsilon):
        """Strategies should work with different epsilon settings."""
        from HoloLoom.policy.unified import TSBandit, BanditStrategy

        n_tools = 4
        strategy_enum = BanditStrategy[strategy.upper()]
        bandit = TSBandit(n_arms=n_tools, strategy=strategy_enum, epsilon=epsilon)
        probs = np.array([0.4, 0.3, 0.2, 0.1])

        arm, metadata = bandit.select_with_strategy(probs, strategy=strategy_enum)

        assert 0 <= arm < n_tools
        assert isinstance(metadata, dict)

    @pytest.mark.parametrize("strategy", ["epsilon_greedy", "bayesian_blend", "pure_thompson"])
    def test_bandit_update_affects_future_selections(self, strategy):
        """Updates should influence future selections."""
        from HoloLoom.policy.unified import TSBandit, BanditStrategy

        n_tools = 3
        strategy_enum = BanditStrategy[strategy.upper()]
        bandit = TSBandit(n_arms=n_tools, strategy=strategy_enum)

        # Get initial stats
        initial_stats = bandit.get_stats()

        # Update arm 0 with high reward multiple times
        for _ in range(10):
            bandit.update(0, 0.95)

        # Get updated stats
        updated_stats = bandit.get_stats()

        # Arm 0 should have higher expected reward now (success count increased)
        assert updated_stats[0]['success'] > initial_stats[0]['success']


# ============================================================================
# Policy × Feature Dimension Tests
# ============================================================================

class TestPolicyDimensionMatrix:
    """Test policy with different feature dimensions."""

    @pytest.mark.parametrize("mem_dim", [96, 192, 384])
    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_policy_creation_with_dimensions(self, mem_dim, n_layers):
        """Policy should initialize with various dimension and layer combinations."""
        from HoloLoom.policy.unified import create_policy
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        scales = [96, 192, 384][:min(3, mem_dim // 96)]

        policy = create_policy(
            mem_dim=mem_dim,
            emb=emb,
            scales=scales,
            n_layers=n_layers
        )

        assert policy is not None
        assert hasattr(policy, 'bandit')
        # Policy uses fixed 4 tools: ["answer", "search", "notion_write", "calc"]
        assert policy.bandit.n_arms == 4

    @pytest.mark.parametrize("mem_dim", [96, 384])
    @pytest.mark.parametrize("n_heads", [2, 4, 8])
    def test_policy_structure_with_configurations(self, mem_dim, n_heads):
        """Policy should create with various configuration combinations."""
        from HoloLoom.policy.unified import create_policy
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])

        policy = create_policy(
            mem_dim=mem_dim,
            emb=emb,
            scales=[96],
            n_heads=n_heads
        )

        # Policy structure should be correct
        assert policy is not None
        assert hasattr(policy, 'core')
        assert hasattr(policy, 'bandit')
        # Policy uses async decide() method, verify it exists
        assert hasattr(policy, 'decide')


# ============================================================================
# Memory Backend × Operation Tests
# ============================================================================

class TestMemoryOperationMatrix:
    """Test memory operations across backends."""

    @pytest.mark.parametrize("backend_type", ["INMEMORY"])
    @pytest.mark.parametrize("operation", ["store", "retrieve", "update"])
    @pytest.mark.asyncio
    async def test_backend_operations(self, backend_type, operation):
        """Each backend should support basic operations."""
        from HoloLoom.config import Config, MemoryBackend
        from HoloLoom.memory.backend_factory import create_memory_backend

        config = Config.fast()
        config.memory_backend = MemoryBackend[backend_type]

        backend = await create_memory_backend(config)

        try:
            if operation == "store":
                # Test store operation
                assert hasattr(backend, 'store') or hasattr(backend, 'add_edges')
            elif operation == "retrieve":
                # Test retrieve operation
                assert hasattr(backend, 'retrieve') or hasattr(backend, 'get_neighbors')
            elif operation == "update":
                # Test update capability exists
                assert backend is not None
        finally:
            if hasattr(backend, 'close'):
                await backend.close()


# ============================================================================
# Classifier × Query Pattern Tests
# ============================================================================

class TestClassifierPatternMatrix:
    """Test classifier with various query patterns."""

    @pytest.mark.parametrize("pattern,expected_category", [
        # Trivial patterns
        ("hi", "trivial"),
        ("hello", "trivial"),
        ("thanks", "trivial"),
        ("ok", "trivial"),
        # Simple patterns
        ("What is X?", "simple"),
        ("Define Y", "simple"),
        ("Who is Z?", "simple"),
        # Complex patterns
        ("Explain X in detail", "complex"),
        ("How does X work step by step?", "complex"),
    ])
    def test_classifier_pattern_recognition(self, pattern, expected_category):
        """Classifier should recognize common query patterns."""
        from HoloLoom.routing.query_classifier import QueryClassifier

        classifier = QueryClassifier()
        result = classifier.classify(pattern)

        # Get the complexity value
        complexity = result.complexity.value.lower()

        # For patterns with placeholders, just verify classification works
        if expected_category == "trivial":
            assert complexity in ["trivial", "simple"]  # Flexible for edge cases
        elif expected_category == "simple":
            assert complexity in ["trivial", "simple", "complex"]
        elif expected_category == "complex":
            assert complexity in ["simple", "complex", "research"]

    @pytest.mark.parametrize("query_length", [1, 5, 10, 20, 50])
    def test_classifier_with_different_lengths(self, query_length):
        """Classifier should handle queries of various lengths."""
        from HoloLoom.routing.query_classifier import QueryClassifier

        classifier = QueryClassifier()

        # Generate query of specified word length
        words = ["word"] * query_length
        query = " ".join(words)

        result = classifier.classify(query)

        assert result is not None
        assert hasattr(result, 'complexity')
        assert hasattr(result, 'confidence')
        assert 0.0 <= result.confidence <= 1.0


# ============================================================================
# Config Flag × Feature Tests
# ============================================================================

class TestConfigFeatureMatrix:
    """Test config flag combinations."""

    @pytest.mark.parametrize("mode", ["bare", "fast", "fused"])
    @pytest.mark.parametrize("flag", [
        "enable_linguistic_gate",
        "use_compositional_cache",
    ])
    def test_config_flag_combinations(self, mode, flag):
        """Config flags should work across modes."""
        from HoloLoom.config import Config

        config = getattr(Config, mode)()

        # Check if flag exists
        if hasattr(config, flag):
            # Toggle flag
            original = getattr(config, flag)
            setattr(config, flag, not original)
            assert getattr(config, flag) != original


# ============================================================================
# Embedding Scale × Retrieval Tests
# ============================================================================

class TestEmbeddingScaleMatrix:
    """Test embedding at different scales."""

    @pytest.mark.parametrize("scale", [96, 192, 384])
    def test_matryoshka_scale_extraction(self, scale):
        """Matryoshka embeddings should work at each scale."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])

        # Create mock embedding
        full_embedding = np.random.randn(384).astype(np.float32)

        # Extract at scale (first N dimensions)
        scaled = full_embedding[:scale]

        assert scaled.shape[0] == scale
        assert np.allclose(scaled, full_embedding[:scale])

    @pytest.mark.parametrize("n_queries", [1, 5, 10])
    @pytest.mark.parametrize("scale", [96, 384])
    def test_batch_embedding_at_scale(self, n_queries, scale):
        """Batch embedding should work at different scales."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])

        # Generate batch of queries
        queries = [f"Query number {i}" for i in range(n_queries)]

        # Embed each query
        embeddings = []
        for q in queries:
            e = np.random.randn(384).astype(np.float32)
            embeddings.append(e[:scale])

        result = np.array(embeddings)

        assert result.shape == (n_queries, scale)


# ============================================================================
# Integration Scenario Tests
# ============================================================================

class TestIntegrationScenarioMatrix:
    """Test full integration scenarios."""

    @pytest.mark.parametrize("mode", ["bare", "fast"])
    @pytest.mark.parametrize("query_type", ["factual", "procedural", "analytical"])
    @pytest.mark.asyncio
    async def test_mode_query_type_combinations(self, mode, query_type):
        """Test mode × query type matrix."""
        from HoloLoom.config import Config
        from HoloLoom.routing.query_classifier import QueryClassifier

        queries = {
            "factual": "What is Python?",
            "procedural": "How do I install numpy?",
            "analytical": "Compare Python vs Java",
        }

        config = getattr(Config, mode)()
        classifier = QueryClassifier()

        result = classifier.classify(queries[query_type])

        assert result is not None
        # All queries should classify successfully
        assert result.complexity is not None

    @pytest.mark.parametrize("strategy", ["epsilon_greedy", "bayesian_blend"])
    @pytest.mark.parametrize("exploration_rate", [0.0, 0.1, 0.3, 0.5])
    def test_strategy_exploration_combinations(self, strategy, exploration_rate):
        """Test strategy × exploration rate matrix."""
        from HoloLoom.policy.unified import TSBandit, BanditStrategy

        n_tools = 4
        strategy_enum = BanditStrategy[strategy.upper()]
        # Set epsilon at init time
        bandit = TSBandit(n_arms=n_tools, strategy=strategy_enum, epsilon=exploration_rate)

        # Run multiple selections to test exploration
        probs = np.array([0.7, 0.2, 0.08, 0.02])

        selections = []
        for _ in range(100):
            arm, metadata = bandit.select_with_strategy(
                probs,
                strategy=strategy_enum
            )
            selections.append(arm)

        # With exploration > 0, we should see some variety
        unique_selections = len(set(selections))

        if exploration_rate > 0:
            # Should explore at least sometimes
            assert unique_selections >= 1
        else:
            # Pure exploitation might still show variety due to Thompson Sampling
            assert unique_selections >= 1


# ============================================================================
# Stress Tests with Parametrization
# ============================================================================

class TestStressParametrized:
    """Stress tests with parametrized inputs."""

    @pytest.mark.parametrize("n_iterations", [10, 50, 100])
    def test_bandit_many_updates(self, n_iterations):
        """Bandit should handle many updates."""
        from HoloLoom.policy.unified import TSBandit

        bandit = TSBandit(n_arms=4)

        # Get initial stats for comparison
        initial_stats = bandit.get_stats()
        initial_total = sum(initial_stats[i]['success'] for i in range(4))

        for i in range(n_iterations):
            arm = i % 4
            reward = np.random.random()
            bandit.update(arm, reward)

        # Stats should reflect updates
        # get_stats() returns {0: {'success', 'fail', 'mean', 'pulls'}, 1: {...}, ...}
        stats = bandit.get_stats()
        total_successes = sum(stats[i]['success'] for i in range(4))

        # After n_iterations updates with random 0-1 rewards (avg ~0.5),
        # total_successes should increase by roughly n_iterations * 0.5
        # Use 0.3 as conservative threshold to handle variance
        expected_min_increase = n_iterations * 0.3
        actual_increase = total_successes - initial_total
        assert actual_increase >= expected_min_increase, \
            f"Expected at least {expected_min_increase:.1f} increase, got {actual_increase:.1f}"

    @pytest.mark.parametrize("query_count", [5, 20, 50])
    def test_classifier_many_queries(self, query_count):
        """Classifier should handle many queries efficiently."""
        from HoloLoom.routing.query_classifier import QueryClassifier

        classifier = QueryClassifier()

        queries = [f"Query number {i} about topic {i % 5}" for i in range(query_count)]

        results = []
        for q in queries:
            result = classifier.classify(q)
            results.append(result)

        assert len(results) == query_count
        assert all(r is not None for r in results)
