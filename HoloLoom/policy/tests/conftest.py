"""
Shared Fixtures for Policy Engine Tests
=======================================
Provides common test fixtures for neural network components,
Thompson Sampling bandit, and policy integration tests.

Author: HoloLoom Team
Date: January 2026
"""

import os
import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, AsyncMock

# Disable guardrails for testing
os.environ["DISABLE_GUARDRAILS"] = "1"


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Provide the best available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    """Force CPU device for deterministic tests."""
    return torch.device("cpu")


# ============================================================================
# Mock Classes for Testing
# ============================================================================

class MockMatryoshkaEmbeddings:
    """Mock embeddings for testing without actual model loading.

    Uses deterministic embeddings by caching results per text/size combination.
    This ensures the same input text always produces the same embedding,
    which is essential for testing deterministic behavior (e.g., epsilon=0).
    """

    def __init__(self, sizes: List[int] = None):
        self.sizes = sizes or [96, 192, 384]
        self._cache: Dict[tuple, np.ndarray] = {}  # (text, size) -> embedding

    def encode_scales(self, texts: List[str], size: int = 384) -> np.ndarray:
        """Return deterministic embeddings of the requested size.

        Caches embeddings per text/size so the same input always returns
        the same output, enabling reproducible test behavior.
        """
        results = []
        for text in texts:
            cache_key = (text, size)
            if cache_key not in self._cache:
                # Use hash of text as seed for reproducibility
                seed = hash(text) % (2**32)
                rng = np.random.RandomState(seed)
                self._cache[cache_key] = rng.randn(size).astype(np.float32)
            results.append(self._cache[cache_key])
        return np.array(results)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using maximum scale."""
        return self.encode_scales(texts, max(self.sizes))


@dataclass
class MockContext:
    """Mock context for policy testing."""
    shard_texts: List[str] = field(default_factory=list)
    hits: List[Any] = field(default_factory=list)
    relevance: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)
    query: Any = None


@dataclass
class MockFeatures:
    """Mock features for policy testing."""
    # psi must match mem_dim (384) because psi_proj = nn.Linear(mem_dim, 8)
    psi: np.ndarray = field(default_factory=lambda: np.random.randn(384).astype(np.float32))
    motifs: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=lambda: {"coherence": 0.8})
    confidence: float = 0.85


@dataclass
class MockShard:
    """Mock shard for context testing."""
    episode: str = "episode_1"
    text: str = "Test shard content"


# ============================================================================
# Neural Network Fixtures
# ============================================================================

@pytest.fixture
def d_model():
    """Standard model dimension for testing."""
    return 384


@pytest.fixture
def n_heads():
    """Standard number of attention heads."""
    return 4


@pytest.fixture
def n_motifs():
    """Standard number of motif patterns."""
    return 8


@pytest.fixture
def n_adapters():
    """Standard number of LoRA adapters."""
    return 4


@pytest.fixture
def n_tools():
    """Standard number of tools."""
    return 4


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 32


@pytest.fixture
def seq_length():
    """Standard sequence length for testing."""
    return 16


@pytest.fixture
def random_input(batch_size, d_model, device):
    """Generate random input tensor."""
    return torch.randn(batch_size, d_model).to(device)


@pytest.fixture
def random_seq_input(batch_size, seq_length, d_model, device):
    """Generate random sequential input tensor."""
    return torch.randn(batch_size, seq_length, d_model).to(device)


@pytest.fixture
def random_gates(batch_size, n_heads, device):
    """Generate random gate values for attention."""
    return torch.rand(batch_size, n_heads).to(device)


@pytest.fixture
def random_motif_ctrl(batch_size, n_motifs, device):
    """Generate random motif control vector."""
    return torch.rand(batch_size, n_motifs).to(device)


# ============================================================================
# Thompson Sampling Fixtures
# ============================================================================

@pytest.fixture
def ts_bandit():
    """Create a fresh Thompson Sampling bandit."""
    from HoloLoom.policy.thompson_sampling import TSBandit
    from HoloLoom.protocols.types import BanditStrategy
    return TSBandit(n_arms=4, strategy=BanditStrategy.EPSILON_GREEDY, epsilon=0.1)


@pytest.fixture
def epsilon_greedy_bandit():
    """Create epsilon-greedy bandit."""
    from HoloLoom.policy.thompson_sampling import TSBandit
    from HoloLoom.protocols.types import BanditStrategy
    return TSBandit(n_arms=4, strategy=BanditStrategy.EPSILON_GREEDY, epsilon=0.1)


@pytest.fixture
def bayesian_blend_bandit():
    """Create Bayesian blend bandit."""
    from HoloLoom.policy.thompson_sampling import TSBandit
    from HoloLoom.protocols.types import BanditStrategy
    return TSBandit(n_arms=4, strategy=BanditStrategy.BAYESIAN_BLEND, epsilon=0.1)


@pytest.fixture
def pure_thompson_bandit():
    """Create pure Thompson Sampling bandit."""
    from HoloLoom.policy.thompson_sampling import TSBandit
    from HoloLoom.protocols.types import BanditStrategy
    return TSBandit(n_arms=4, strategy=BanditStrategy.PURE_THOMPSON, epsilon=0.1)


@pytest.fixture
def trained_bandit():
    """Create a bandit with some training history."""
    from HoloLoom.policy.thompson_sampling import TSBandit
    from HoloLoom.protocols.types import BanditStrategy
    bandit = TSBandit(n_arms=4, strategy=BanditStrategy.BAYESIAN_BLEND, epsilon=0.1)
    # Add some training history
    for _ in range(10):
        bandit.update(0, 0.8)  # arm 0 is good
        bandit.update(1, 0.3)  # arm 1 is mediocre
        bandit.update(2, 0.1)  # arm 2 is poor
        bandit.update(3, -0.2)  # arm 3 has negative feedback
    return bandit


# ============================================================================
# Policy Fixtures
# ============================================================================

@pytest.fixture
def mock_embeddings():
    """Provide mock embeddings."""
    return MockMatryoshkaEmbeddings(sizes=[96, 192, 384])


@pytest.fixture
def mock_context():
    """Provide mock context with sample data."""
    ctx = MockContext(
        shard_texts=["Sample context text 1", "Sample context text 2"],
        hits=[(MockShard(episode="ep1"), 0.8), (MockShard(episode="ep2"), 0.7)],
        relevance=0.75,
        metadata={"user_id": "test_user", "session_id": "test_session"}
    )
    return ctx


@pytest.fixture
def mock_features():
    """Provide mock features."""
    # psi must be 384-dimensional to match mem_dim in NeuralCore
    psi = np.random.randn(384).astype(np.float32)
    psi[:6] = [0.1, 0.5, 0.2, 0.3, 0.4, 0.6]  # Set first 6 values for consistency
    return MockFeatures(
        psi=psi,
        motifs=["question→answer", "cause→effect"],  # Use Unicode arrows
        metrics={"coherence": 0.8, "fiedler": 0.3},
        confidence=0.85
    )


@pytest.fixture
def empty_context():
    """Provide empty context for edge case testing."""
    return MockContext(
        shard_texts=[],
        hits=[],
        relevance=0.0,
        metadata={}
    )


@pytest.fixture
def scales():
    """Standard embedding scales."""
    return [96, 192, 384]


# ============================================================================
# Neural Core Fixtures
# ============================================================================

@pytest.fixture
def custom_mha(d_model, n_heads):
    """Create CustomMHA module."""
    from HoloLoom.policy.unified import CustomMHA
    return CustomMHA(d_model=d_model, n_heads=n_heads)


@pytest.fixture
def cross_attention(d_model, n_heads):
    """Create CrossAttention module."""
    from HoloLoom.policy.unified import CrossAttention
    return CrossAttention(d_model=d_model, n_heads=n_heads)


@pytest.fixture
def motif_gated_mha(d_model, n_heads, n_motifs):
    """Create MotifGatedMHA module."""
    from HoloLoom.policy.unified import MotifGatedMHA
    return MotifGatedMHA(d_model=d_model, n_heads=n_heads, n_motifs=n_motifs)


@pytest.fixture
def lora_ffn(d_model, n_adapters):
    """Create LoRALikeFFN module."""
    from HoloLoom.policy.unified import LoRALikeFFN
    return LoRALikeFFN(d_model=d_model, d_ff=4*d_model, r=8, n_adapters=n_adapters)


@pytest.fixture
def transformer_block(d_model, n_heads, n_motifs, n_adapters):
    """Create TinyTransformerBlock module."""
    from HoloLoom.policy.unified import TinyTransformerBlock
    return TinyTransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        n_motifs=n_motifs,
        n_adapters=n_adapters
    )


@pytest.fixture
def neural_core(d_model, n_heads, n_motifs, n_adapters, n_tools):
    """Create NeuralCore module."""
    from HoloLoom.policy.unified import NeuralCore
    return NeuralCore(
        d_model=d_model,
        n_layers=2,
        n_heads=n_heads,
        n_motifs=n_motifs,
        n_adapters=n_adapters,
        n_tools=n_tools
    )


# ============================================================================
# Test Helper Fixtures
# ============================================================================

@pytest.fixture
def assert_tensor_shape():
    """Factory fixture for tensor shape assertions."""
    def _assert(tensor, expected_shape, name="tensor"):
        assert tensor.shape == expected_shape, \
            f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"
    return _assert


@pytest.fixture
def assert_gradients_exist():
    """Factory fixture for gradient existence assertions."""
    def _assert(module, input_tensor):
        # Forward pass
        output = module(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        # Backward pass
        loss = output.sum()
        loss.backward()
        # Check gradients
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    return _assert


@pytest.fixture
def measure_time():
    """Factory fixture for timing measurements."""
    import time
    def _measure(func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed * 1000  # Return result and time in ms
    return _measure


# ============================================================================
# Async Test Helpers
# ============================================================================

@pytest.fixture
def run_async():
    """Helper to run async functions in tests."""
    import asyncio
    def _run(coro):
        return asyncio.get_event_loop().run_until_complete(coro)
    return _run


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Cleanup CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def seed_random():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    yield
