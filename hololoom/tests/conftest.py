"""
Shared pytest fixtures and configuration for HoloLoom test suite.

This module provides:
- Reproducible random seeds
- Common test configurations
- Mock fixtures for external dependencies
- Performance budget enforcement
- Test data factories

Usage:
    Fixtures are automatically discovered by pytest and available to all tests.
"""

import pytest
import asyncio
import numpy as np
import os
import warnings
from typing import List
from unittest.mock import Mock, patch, MagicMock

# Import HoloLoom types and configs
from HoloLoom.config import Config, ExecutionMode
from HoloLoom.protocols.types import Query, MemoryShard, Context, Features


# =============================================================================
# pytest-asyncio Configuration
# =============================================================================

# Configure pytest-asyncio to use "auto" mode for async test discovery
pytest_plugins = ('pytest_asyncio',)

def pytest_configure(config):
    """Configure pytest-asyncio mode."""
    config.option.asyncio_mode = "auto"


# =============================================================================
# Reproducibility
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    """Set reproducible random seeds for all tests."""
    np.random.seed(42)
    import random
    random.seed(42)

    # Also set torch seed if available
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass

    yield


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment variables between tests."""
    # Save original environment
    original_env = os.environ.copy()

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def bare_config():
    """Minimal processing configuration for fast tests."""
    return Config.bare()


@pytest.fixture
def fast_config():
    """Balanced configuration for standard tests."""
    return Config.fast()


@pytest.fixture
def fused_config():
    """Full processing configuration for comprehensive tests."""
    return Config.fused()


@pytest.fixture(params=[ExecutionMode.BARE, ExecutionMode.FAST, ExecutionMode.FUSED])
def all_configs(request):
    """Parametrize tests across all execution modes."""
    mode = request.param
    if mode == ExecutionMode.BARE:
        return Config.bare()
    elif mode == ExecutionMode.FAST:
        return Config.fast()
    else:
        return Config.fused()


# =============================================================================
# Test Data Factories
# =============================================================================

@pytest.fixture
def test_query():
    """Simple test query."""
    return Query(text="What is Thompson Sampling?")


@pytest.fixture
def test_queries():
    """Multiple test queries for batch testing."""
    return [
        Query(text="What is Thompson Sampling?"),
        Query(text="How does reinforcement learning work?"),
        Query(text="Explain the explore-exploit tradeoff"),
    ]


@pytest.fixture
def test_shards():
    """Create test memory shards."""
    return [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            episode="test_episode",
            metadata={"source": "test"}
        ),
        MemoryShard(
            id="shard_2",
            text="Reinforcement learning involves learning from rewards and penalties.",
            episode="test_episode",
            metadata={"source": "test"}
        ),
        MemoryShard(
            id="shard_3",
            text="The explore-exploit tradeoff balances trying new actions versus using known good actions.",
            episode="test_episode",
            metadata={"source": "test"}
        ),
    ]


@pytest.fixture
def test_context(test_shards, test_query):
    """Create test context."""
    return Context(
        shards=test_shards,
        hits=[0, 1, 2],
        shard_texts=[s.text for s in test_shards],
        query=test_query,
        features=None  # Will be populated by tests
    )


@pytest.fixture
def test_features():
    """Create test features (DotPlasma)."""
    return Features(
        psi=[0.1] * 96,  # 96-dimensional embedding
        motifs=["sampling", "bayesian"],
        metrics={"spectral": {"eigenvalue_mean": 0.5}},
        metadata={"scale": 96}
    )


# =============================================================================
# Mock Fixtures for External Dependencies
# =============================================================================

@pytest.fixture
def mock_neo4j():
    """Mock Neo4j database driver."""
    with patch('neo4j.GraphDatabase.driver') as mock_driver:
        # Configure mock driver
        mock_session = MagicMock()
        mock_driver.return_value.session.return_value = mock_session
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None

        yield mock_driver


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant vector database client."""
    with patch('qdrant_client.QdrantClient') as mock_client:
        # Configure mock client
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        yield mock_client


@pytest.fixture
def mock_ollama():
    """Mock Ollama LLM client."""
    with patch('ollama.generate') as mock_generate:
        # Configure mock response
        mock_generate.return_value = "This is a mocked LLM response."

        yield mock_generate


@pytest.fixture
def mock_all_external_deps(mock_neo4j, mock_qdrant, mock_ollama):
    """Mock all external dependencies at once."""
    return {
        'neo4j': mock_neo4j,
        'qdrant': mock_qdrant,
        'ollama': mock_ollama
    }


# =============================================================================
# Performance Budget Enforcement
# =============================================================================

@pytest.fixture(autouse=True)
def performance_budget(request):
    """
    Enforce performance budgets based on test tier.

    - Unit tests: <500ms per test
    - Integration tests: <2s per test
    - E2E tests: <30s per test
    """
    import time

    start = time.time()
    yield
    elapsed = time.time() - start

    # Determine budget based on test location
    test_path = str(request.fspath)

    if "unit" in test_path:
        budget = 0.5  # 500ms
        tier = "unit"
    elif "integration" in test_path:
        budget = 2.0  # 2 seconds
        tier = "integration"
    elif "e2e" in test_path:
        budget = 30.0  # 30 seconds
        tier = "e2e"
    else:
        budget = 5.0  # Default: 5 seconds
        tier = "default"

    if elapsed > budget:
        warnings.warn(
            f"{tier.upper()} test {request.node.name} took {elapsed:.2f}s (budget: {budget}s)",
            UserWarning
        )


# =============================================================================
# Async Helpers
# =============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Embedding Fixtures (Session-scoped for performance)
# =============================================================================

@pytest.fixture(scope="session")
def cached_embeddings():
    """
    Session-scoped embedding model to avoid loading for every test.

    This fixture significantly speeds up tests that need embeddings.
    Loading embeddings once per session instead of per test reduces
    test time from 3.3s to <0.5s per test.
    """
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

    # MatryoshkaEmbeddings is a dataclass with sizes parameter
    return MatryoshkaEmbeddings(sizes=[96, 192, 384])


# =============================================================================
# Temporary Directory
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """
    Provide a temporary directory for tests.

    Automatically cleaned up after test completion.
    """
    return tmp_path


# =============================================================================
# Test Utilities
# =============================================================================

@pytest.fixture
def assert_almost_equal():
    """Utility for comparing floating point values."""
    def _assert_almost_equal(a, b, tolerance=1e-6):
        assert abs(a - b) < tolerance, f"{a} != {b} (tolerance={tolerance})"
    return _assert_almost_equal


@pytest.fixture
def assert_valid_confidence():
    """Utility for validating confidence scores."""
    def _assert_valid_confidence(confidence, message=""):
        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} not in [0,1]. {message}"
        assert isinstance(confidence, (int, float)), f"Confidence {confidence} not numeric. {message}"
    return _assert_valid_confidence


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "requires_docker: marks tests that require Docker services")
    config.addinivalue_line("markers", "requires_llm: marks tests that require Ollama/LLM")
