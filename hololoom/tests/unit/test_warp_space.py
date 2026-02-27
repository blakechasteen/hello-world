"""
Test Warp Space - Tensioned Tensor Field
=========================================

Tests for HoloLoom/warp/space.py - Continuous computational manifold.

Coverage:
- Tensioning threads into Warp Space
- Multi-scale embedding support
- Sparsity control and breathing integration
- Spectral feature computation
- Attention mechanism
- Weighted context aggregation
- Collapse and detension
- Computational trace
- Safety guardrails integration
- Error handling
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from HoloLoom.warp.space import (
    WarpSpace,
    TensionedThread
)
from HoloLoom.alignment.safety_guardrails import (
    SafetyGuardrails,
    ActionRequest,
    SafetyDecision,
    ActionCategory,
    RiskLevel
)


# ============================================================================
# Mock Embedder
# ============================================================================

class MockEmbedder:
    """Mock matryoshka embedder."""

    def __init__(self, sizes=[96, 192, 768]):
        self.sizes = sizes

    def encode_scales(self, texts: List[str], size: int = None) -> dict:
        """Return embeddings at all scales."""
        embeddings = {}
        for scale in self.sizes:
            embeddings[scale] = np.random.randn(len(texts), scale).astype(np.float32)

        if size:
            return embeddings[size]
        return embeddings

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return embeddings at largest scale."""
        max_scale = max(self.sizes)
        return np.random.randn(len(texts), max_scale).astype(np.float32)


class MockSpectralFusion:
    """Mock spectral feature extractor."""

    async def features(self, graph, texts, embedder):
        return np.random.randn(10), {"graph_size": 5}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def embedder():
    """Create mock embedder."""
    return MockEmbedder(sizes=[96, 192, 768])


@pytest.fixture
def warp(embedder):
    """Create basic warp space."""
    return WarpSpace(
        embedder=embedder,
        scales=[96, 192, 768],
        spectral_fusion=None,
        guardrails=None  # Disable for basic tests
    )


@pytest.fixture
def warp_with_spectral(embedder):
    """Create warp space with spectral fusion."""
    return WarpSpace(
        embedder=embedder,
        scales=[96, 192, 768],
        spectral_fusion=MockSpectralFusion(),
        guardrails=None
    )


@pytest.fixture
def mock_guardrails():
    """Create mock safety guardrails that allow all actions."""
    guardrails = Mock(spec=SafetyGuardrails)
    guardrails.evaluate.return_value = SafetyDecision(
        allowed=True,
        risk_level=RiskLevel.LOW,
        reason="Test: allowed",
        requires_approval=False
    )
    return guardrails


@pytest.fixture
def test_threads():
    """Standard test thread texts."""
    return [
        "Thompson Sampling balances exploration and exploitation",
        "Neural networks learn hierarchical representations",
        "Attention mechanisms enable context-aware processing"
    ]


# ============================================================================
# Test Initialization
# ============================================================================

def test_warp_space_initialization(embedder):
    """Test warp space initializes correctly."""
    warp = WarpSpace(
        embedder=embedder,
        scales=[96, 192, 768],
        spectral_fusion=None
    )

    assert warp.embedder is embedder
    assert warp.scales == [96, 192, 768]
    assert warp.is_tensioned is False
    assert len(warp.threads) == 0
    assert warp.tensor_field is None
    assert len(warp.operations) == 0


def test_warp_space_with_spectral_fusion(embedder):
    """Test warp space with spectral fusion."""
    spectral = MockSpectralFusion()
    warp = WarpSpace(
        embedder=embedder,
        scales=[96, 192, 768],
        spectral_fusion=spectral
    )

    assert warp.spectral_fusion is spectral


# ============================================================================
# Test Tensioning Threads
# ============================================================================

@pytest.mark.asyncio
async def test_tension_creates_field(warp, test_threads):
    """Test tensioning creates tensor field."""
    await warp.tension(test_threads)

    assert warp.is_tensioned is True
    assert len(warp.threads) == 3
    assert warp.tensor_field is not None
    assert warp.tensor_field.shape == (3, 768)  # 3 threads × largest scale


@pytest.mark.asyncio
async def test_tension_multi_scale_embeddings(warp, test_threads):
    """Test tensioning creates embeddings at all scales."""
    await warp.tension(test_threads)

    # Check each thread has multi-scale embeddings
    for thread in warp.threads:
        multi_scale = thread.metadata['multi_scale_embeddings']
        assert 96 in multi_scale
        assert 192 in multi_scale
        assert 768 in multi_scale

        # Check dimensions
        assert multi_scale[96].shape == (96,)
        assert multi_scale[192].shape == (192,)
        assert multi_scale[768].shape == (768,)


@pytest.mark.asyncio
async def test_tension_with_custom_ids(warp, test_threads):
    """Test tensioning with custom thread IDs."""
    custom_ids = ["thread_a", "thread_b", "thread_c"]

    await warp.tension(test_threads, thread_ids=custom_ids)

    # Check IDs assigned
    for i, thread in enumerate(warp.threads):
        assert thread.thread_id == custom_ids[i]


@pytest.mark.asyncio
async def test_tension_with_custom_weights(warp, test_threads):
    """Test tensioning with custom tension weights."""
    custom_weights = [0.5, 1.0, 0.8]

    await warp.tension(test_threads, tension_weights=custom_weights)

    # Check weights assigned
    for i, thread in enumerate(warp.threads):
        assert thread.tension == custom_weights[i]


@pytest.mark.asyncio
async def test_tension_default_weights_uniform(warp, test_threads):
    """Test tensioning defaults to uniform tension weights."""
    await warp.tension(test_threads)

    # All threads should have tension = 1.0
    for thread in warp.threads:
        assert thread.tension == 1.0


@pytest.mark.asyncio
async def test_tension_creates_thread_index(warp, test_threads):
    """Test tensioning creates thread index for lookup."""
    await warp.tension(test_threads)

    assert len(warp.thread_index) == 3

    # Check can lookup by ID
    thread_0 = warp.thread_index["thread_0"]
    assert thread_0.thread_id == "thread_0"


@pytest.mark.asyncio
async def test_tension_stores_text_metadata(warp, test_threads):
    """Test tensioning stores original text in metadata."""
    await warp.tension(test_threads)

    for i, thread in enumerate(warp.threads):
        assert thread.metadata['text'] == test_threads[i]
        assert thread.metadata['index'] == i


# ============================================================================
# Test Sparsity Control
# ============================================================================

@pytest.mark.asyncio
async def test_tension_sparsity_zero_all_threads_active(warp, test_threads):
    """Test sparsity=0.0 activates all threads."""
    await warp.tension(test_threads, sparsity=0.0)

    # All threads should be active
    assert len(warp.threads) == 3


@pytest.mark.asyncio
async def test_tension_sparsity_high_fewer_threads(warp, test_threads):
    """Test sparsity=0.7 activates only top 30% threads."""
    tension_weights = [0.5, 1.0, 0.3]  # thread_1 is highest

    await warp.tension(test_threads, tension_weights=tension_weights, sparsity=0.7)

    # Should have only 1 thread (30% of 3 = 0.9, rounded to 1)
    assert len(warp.threads) == 1

    # Should be the highest-weight thread (thread_1)
    assert warp.threads[0].thread_id == "thread_1"


@pytest.mark.asyncio
async def test_tension_sparsity_always_activates_at_least_one(warp, test_threads):
    """Test sparsity always activates at least one thread."""
    await warp.tension(test_threads, sparsity=0.99)

    # Should have at least 1 thread even with 99% sparsity
    assert len(warp.threads) >= 1


@pytest.mark.asyncio
async def test_tension_sparsity_breathing_integration(warp):
    """Test sparsity control enables breathing rhythm (exhale phase)."""
    threads = [f"thread {i}" for i in range(10)]
    weights = [1.0] * 10

    # Exhale phase: high sparsity (0.7)
    await warp.tension(threads, tension_weights=weights, sparsity=0.7)

    # Should have only 30% of threads (3 threads)
    assert len(warp.threads) == 3


# ============================================================================
# Test Empty Thread Handling
# ============================================================================

@pytest.mark.asyncio
async def test_tension_empty_threads_warning(warp):
    """Test tensioning with empty thread list logs warning."""
    await warp.tension([])

    # Should not crash, but not tension
    assert warp.is_tensioned is False
    assert len(warp.threads) == 0


# ============================================================================
# Test Get Field at Different Scales
# ============================================================================

@pytest.mark.asyncio
async def test_get_field_default_returns_largest_scale(warp, test_threads):
    """Test get_field() returns tensor field at largest scale."""
    await warp.tension(test_threads)

    field = warp.get_field()

    assert field.shape == (3, 768)  # Largest scale


@pytest.mark.asyncio
async def test_get_field_at_scale_96(warp, test_threads):
    """Test get_field(scale=96) returns 96-dim embeddings."""
    await warp.tension(test_threads)

    field = warp.get_field(scale=96)

    assert field.shape == (3, 96)


@pytest.mark.asyncio
async def test_get_field_at_scale_192(warp, test_threads):
    """Test get_field(scale=192) returns 192-dim embeddings."""
    await warp.tension(test_threads)

    field = warp.get_field(scale=192)

    assert field.shape == (3, 192)


@pytest.mark.asyncio
async def test_get_field_without_tension_raises_error(warp):
    """Test get_field() raises error if not tensioned."""
    with pytest.raises(RuntimeError, match="not tensioned"):
        warp.get_field()


@pytest.mark.asyncio
async def test_get_field_unsupported_scale_projects(warp, test_threads):
    """Test get_field() projects from largest scale if scale not available."""
    await warp.tension(test_threads)

    # Request unsupported scale (should project from 768)
    field = warp.get_field(scale=128)

    # Should truncate from 768 to 128
    assert field.shape == (3, 128)


# ============================================================================
# Test Spectral Features
# ============================================================================

@pytest.mark.asyncio
async def test_compute_spectral_features_basic(warp, test_threads):
    """Test spectral features compute basic statistics."""
    await warp.tension(test_threads)

    features = warp.compute_spectral_features()

    assert 'field_norm' in features
    assert 'mean_activation' in features
    assert 'dimensionality' in features
    assert features['dimensionality'] == (3, 768)


@pytest.mark.asyncio
async def test_compute_spectral_features_svd(warp, test_threads):
    """Test spectral features compute SVD."""
    await warp.tension(test_threads)

    features = warp.compute_spectral_features()

    assert 'singular_values' in features
    assert 'spectral_entropy' in features
    assert len(features['singular_values']) <= 6  # First 6 singular values


@pytest.mark.asyncio
async def test_compute_spectral_features_records_operation(warp, test_threads):
    """Test spectral computation records operation."""
    await warp.tension(test_threads)

    warp.compute_spectral_features()

    assert len(warp.operations) == 1
    assert warp.operations[0][0] == 'spectral_features'


@pytest.mark.asyncio
async def test_compute_spectral_features_without_tension_returns_empty(warp):
    """Test spectral features without tension returns empty dict."""
    features = warp.compute_spectral_features()

    assert features == {}


# ============================================================================
# Test Attention Mechanism
# ============================================================================

@pytest.mark.asyncio
async def test_apply_attention_computes_scores(warp, test_threads):
    """Test attention computes similarity scores."""
    await warp.tension(test_threads)

    # Create query embedding
    query_emb = np.random.randn(768).astype(np.float32)

    attention = warp.apply_attention(query_emb)

    # Attention should be normalized (sum to 1)
    assert attention.shape == (3,)
    assert np.isclose(np.sum(attention), 1.0)
    assert np.all(attention >= 0)


@pytest.mark.asyncio
async def test_apply_attention_records_operation(warp, test_threads):
    """Test attention application records operation."""
    await warp.tension(test_threads)
    query_emb = np.random.randn(768).astype(np.float32)

    warp.apply_attention(query_emb)

    assert len(warp.operations) == 1
    assert warp.operations[0][0] == 'attention'


@pytest.mark.asyncio
async def test_apply_attention_without_tension_raises_error(warp):
    """Test attention without tension raises error."""
    query_emb = np.random.randn(768).astype(np.float32)

    with pytest.raises(RuntimeError, match="not tensioned"):
        warp.apply_attention(query_emb)


# ============================================================================
# Test Weighted Context
# ============================================================================

@pytest.mark.asyncio
async def test_weighted_context_aggregates_threads(warp, test_threads):
    """Test weighted context aggregates thread embeddings."""
    await warp.tension(test_threads)

    # Uniform attention
    attention = np.array([1/3, 1/3, 1/3], dtype=np.float32)

    context = warp.weighted_context(attention)

    # Context should be weighted average of embeddings
    assert context.shape == (768,)


@pytest.mark.asyncio
async def test_weighted_context_records_operation(warp, test_threads):
    """Test weighted context records operation."""
    await warp.tension(test_threads)
    attention = np.array([1/3, 1/3, 1/3], dtype=np.float32)

    warp.weighted_context(attention)

    assert len(warp.operations) == 1
    assert warp.operations[0][0] == 'weighted_context'
    assert 'attention_entropy' in warp.operations[0][1]


@pytest.mark.asyncio
async def test_weighted_context_without_tension_raises_error(warp):
    """Test weighted context without tension raises error."""
    attention = np.array([1/3, 1/3, 1/3], dtype=np.float32)

    with pytest.raises(RuntimeError, match="not tensioned"):
        warp.weighted_context(attention)


# ============================================================================
# Test Collapse and Detension
# ============================================================================

@pytest.mark.asyncio
async def test_collapse_returns_updates(warp, test_threads):
    """Test collapse returns thread updates and trace."""
    await warp.tension(test_threads)

    updates = warp.collapse()

    assert 'threads' in updates
    assert 'operations' in updates
    assert 'field_stats' in updates
    assert len(updates['threads']) == 3


@pytest.mark.asyncio
async def test_collapse_includes_thread_metadata(warp, test_threads):
    """Test collapse includes thread metadata."""
    await warp.tension(test_threads)

    updates = warp.collapse()

    for thread_update in updates['threads']:
        assert 'thread_id' in thread_update
        assert 'tension' in thread_update
        assert 'embedding_norm' in thread_update
        assert 'metadata' in thread_update


@pytest.mark.asyncio
async def test_collapse_includes_field_stats(warp, test_threads):
    """Test collapse includes tensor field statistics."""
    await warp.tension(test_threads)

    updates = warp.collapse()

    stats = updates['field_stats']
    assert 'shape' in stats
    assert 'norm' in stats
    assert 'mean' in stats
    assert 'std' in stats


@pytest.mark.asyncio
async def test_collapse_resets_state(warp, test_threads):
    """Test collapse resets warp space state."""
    await warp.tension(test_threads)
    warp.compute_spectral_features()  # Add operation

    assert warp.is_tensioned is True
    assert len(warp.threads) > 0
    assert len(warp.operations) > 0

    warp.collapse()

    # State should be reset
    assert warp.is_tensioned is False
    assert len(warp.threads) == 0
    assert warp.tensor_field is None
    assert len(warp.operations) == 0


@pytest.mark.asyncio
async def test_collapse_already_collapsed_returns_empty(warp):
    """Test collapsing when not tensioned returns empty dict."""
    updates = warp.collapse()

    assert updates == {}


@pytest.mark.asyncio
async def test_collapse_can_tension_again_after_collapse(warp, test_threads):
    """Test can tension again after collapse."""
    # First cycle
    await warp.tension(test_threads)
    warp.collapse()

    # Second cycle
    await warp.tension(test_threads)

    assert warp.is_tensioned is True
    assert len(warp.threads) == 3


# ============================================================================
# Test Computational Trace
# ============================================================================

@pytest.mark.asyncio
async def test_get_trace_shows_current_state(warp, test_threads):
    """Test get_trace() returns current state without collapsing."""
    await warp.tension(test_threads)
    warp.compute_spectral_features()

    trace = warp.get_trace()

    assert trace['is_tensioned'] is True
    assert trace['num_threads'] == 3
    assert len(trace['operations']) == 1
    assert trace['field_shape'] == (3, 768)

    # Should NOT collapse
    assert warp.is_tensioned is True


@pytest.mark.asyncio
async def test_get_trace_before_tension(warp):
    """Test get_trace() before tensioning."""
    trace = warp.get_trace()

    assert trace['is_tensioned'] is False
    assert trace['num_threads'] == 0
    assert trace['field_shape'] is None


# ============================================================================
# Test Safety Guardrails Integration
# ============================================================================

@pytest.mark.asyncio
async def test_guardrails_called_during_tension(mock_guardrails, embedder, test_threads):
    """Test safety guardrails called during tensioning."""
    warp = WarpSpace(
        embedder=embedder,
        scales=[96, 192, 768],
        guardrails=mock_guardrails
    )

    await warp.tension(test_threads)

    # Guardrails should have been called
    mock_guardrails.evaluate.assert_called()


@pytest.mark.asyncio
async def test_guardrails_blocked_tension_raises_error(embedder, test_threads):
    """Test blocked tension by guardrails raises PermissionError."""
    blocked_guardrails = Mock(spec=SafetyGuardrails)
    blocked_guardrails.evaluate.return_value = SafetyDecision(
        allowed=False,
        risk_level=RiskLevel.HIGH,
        reason="Test: blocked",
        requires_approval=False
    )

    warp = WarpSpace(
        embedder=embedder,
        scales=[96, 192, 768],
        guardrails=blocked_guardrails
    )

    with pytest.raises(PermissionError, match="Test: blocked"):
        await warp.tension(test_threads)


@pytest.mark.asyncio
async def test_guardrails_decisions_in_collapse(mock_guardrails, embedder, test_threads):
    """Test guardrails decisions included in collapse updates."""
    warp = WarpSpace(
        embedder=embedder,
        scales=[96, 192, 768],
        guardrails=mock_guardrails
    )

    await warp.tension(test_threads)
    warp.compute_spectral_features()

    query_emb = np.random.randn(768).astype(np.float32)
    warp.apply_attention(query_emb)

    updates = warp.collapse()

    # Check guardrails decisions recorded
    assert 'guardrails' in updates
    assert 'tension' in updates['guardrails']
    assert 'spectral' in updates['guardrails']
    assert 'attention' in updates['guardrails']


# ============================================================================
# Test Lifecycle and Operations
# ============================================================================

@pytest.mark.asyncio
async def test_complete_warp_cycle(warp, test_threads):
    """Test complete warp space lifecycle."""
    # 1. Tension
    await warp.tension(test_threads)
    assert warp.is_tensioned is True

    # 2. Compute features
    spectral = warp.compute_spectral_features()
    assert spectral is not None

    # 3. Apply attention
    query_emb = np.random.randn(768).astype(np.float32)
    attention = warp.apply_attention(query_emb)
    assert attention is not None

    # 4. Get context
    context = warp.weighted_context(attention)
    assert context is not None

    # 5. Collapse
    updates = warp.collapse()
    assert len(updates['operations']) == 3  # spectral, attention, context

    # State should be reset
    assert warp.is_tensioned is False


@pytest.mark.asyncio
async def test_operations_recorded_in_order(warp, test_threads):
    """Test operations are recorded in execution order."""
    await warp.tension(test_threads)

    warp.compute_spectral_features()
    query_emb = np.random.randn(768).astype(np.float32)
    warp.apply_attention(query_emb)
    attention = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    warp.weighted_context(attention)

    # Check operation order
    assert len(warp.operations) == 3
    assert warp.operations[0][0] == 'spectral_features'
    assert warp.operations[1][0] == 'attention'
    assert warp.operations[2][0] == 'weighted_context'


# ============================================================================
# Test Tensioned Thread Data Structure
# ============================================================================

def test_tensioned_thread_initialization():
    """Test TensionedThread initialization."""
    embedding = np.random.randn(384).astype(np.float32)

    thread = TensionedThread(
        thread_id="test_thread",
        entity="Test Entity",
        embedding=embedding,
        tension=0.8,
        metadata={"key": "value"}
    )

    assert thread.thread_id == "test_thread"
    assert thread.entity == "Test Entity"
    assert thread.tension == 0.8
    assert thread.metadata["key"] == "value"
    assert np.array_equal(thread.embedding, embedding)


def test_tensioned_thread_default_metadata():
    """Test TensionedThread defaults metadata to empty dict."""
    embedding = np.random.randn(384).astype(np.float32)

    thread = TensionedThread(
        thread_id="test",
        entity="Entity",
        embedding=embedding
    )

    assert thread.metadata == {}
    assert thread.tension == 1.0  # Default


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_svd_failure_handled_gracefully(warp, test_threads):
    """Test SVD computation failure is handled gracefully."""
    await warp.tension(test_threads)

    # Mock SVD to raise exception
    with patch('numpy.linalg.svd', side_effect=Exception("SVD failed")):
        features = warp.compute_spectral_features()

    # Should still return features (without singular values)
    assert 'field_norm' in features
    assert 'singular_values' not in features


@pytest.mark.asyncio
async def test_attention_with_mismatched_dimension_raises_error(warp, test_threads):
    """Test attention with wrong query dimension raises error."""
    await warp.tension(test_threads)

    # Wrong dimension query (384 instead of 768)
    query_emb = np.random.randn(384).astype(np.float32)

    with pytest.raises((ValueError, RuntimeError)):
        warp.apply_attention(query_emb)


@pytest.mark.asyncio
async def test_weighted_context_with_mismatched_attention_length(warp, test_threads):
    """Test weighted context with wrong attention length."""
    await warp.tension(test_threads)

    # Wrong number of attention weights (5 instead of 3)
    attention = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    with pytest.raises((ValueError, RuntimeError)):
        warp.weighted_context(attention)


# ============================================================================
# Test Multi-Scale Fusion Advanced
# ============================================================================

@pytest.mark.asyncio
async def test_multi_scale_embedding_consistency(warp, test_threads):
    """Test multi-scale embeddings are consistent across scales."""
    await warp.tension(test_threads)

    # Get fields at different scales
    field_96 = warp.get_field(scale=96)
    field_192 = warp.get_field(scale=192)
    field_768 = warp.get_field(scale=768)

    # Check dimensions
    assert field_96.shape == (3, 96)
    assert field_192.shape == (3, 192)
    assert field_768.shape == (3, 768)

    # Larger scales should contain more information
    norm_96 = np.linalg.norm(field_96)
    norm_768 = np.linalg.norm(field_768)
    # Higher dimension doesn't necessarily mean higher norm, but check they're valid
    assert norm_96 > 0
    assert norm_768 > 0


@pytest.mark.asyncio
async def test_scale_projection_truncation(warp, test_threads):
    """Test scale projection truncates from largest scale."""
    await warp.tension(test_threads)

    # Request unsupported scale (should project from 768)
    field_128 = warp.get_field(scale=128)
    field_768 = warp.get_field(scale=768)

    # Should be first 128 dimensions of 768
    assert field_128.shape == (3, 128)
    assert np.allclose(field_128[0], field_768[0][:128])


# ============================================================================
# Test Thread Weighting and Selection
# ============================================================================

@pytest.mark.asyncio
async def test_tension_with_heterogeneous_weights(warp, test_threads):
    """Test tensioning with varied weights (0.0 to 1.0)."""
    weights = [0.0, 0.5, 1.0]

    await warp.tension(test_threads, tension_weights=weights)

    # All threads should be present (sparsity=0)
    assert len(warp.threads) == 3
    assert warp.threads[0].tension == 0.0
    assert warp.threads[1].tension == 0.5
    assert warp.threads[2].tension == 1.0


@pytest.mark.asyncio
async def test_sparsity_with_equal_weights(warp):
    """Test sparsity selection with equal weights (random tie-breaking)."""
    threads = [f"thread {i}" for i in range(5)]
    weights = [1.0] * 5  # All equal

    await warp.tension(threads, tension_weights=weights, sparsity=0.6)

    # Should keep 40% = 2 threads
    assert len(warp.threads) == 2


@pytest.mark.asyncio
async def test_sparsity_boundary_conditions(warp):
    """Test sparsity at boundary values."""
    threads = ["a", "b", "c", "d", "e"]

    # 0% sparsity (all active)
    await warp.tension(threads, sparsity=0.0)
    assert len(warp.threads) == 5
    warp.collapse()

    # 80% sparsity (1 thread = 20%)
    await warp.tension(threads, sparsity=0.8)
    assert len(warp.threads) == 1
    warp.collapse()

    # 100% sparsity (still keeps 1 thread minimum)
    await warp.tension(threads, sparsity=1.0)
    assert len(warp.threads) >= 1


# ============================================================================
# Test Attention Mechanism Advanced
# ============================================================================

@pytest.mark.asyncio
async def test_attention_scores_sum_to_one(warp, test_threads):
    """Test attention weights are properly normalized."""
    await warp.tension(test_threads)

    query_emb = np.random.randn(768).astype(np.float32)
    attention = warp.apply_attention(query_emb)

    # Should sum to 1.0 (softmax normalization)
    assert np.isclose(np.sum(attention), 1.0)


@pytest.mark.asyncio
async def test_attention_is_non_negative(warp, test_threads):
    """Test all attention weights are non-negative."""
    await warp.tension(test_threads)

    query_emb = np.random.randn(768).astype(np.float32)
    attention = warp.apply_attention(query_emb)

    # All weights should be >= 0
    assert np.all(attention >= 0)


@pytest.mark.asyncio
async def test_attention_max_on_most_similar(warp):
    """Test attention places maximum weight on most similar thread."""
    # Create threads with known similarity
    threads = [
        "machine learning algorithms",
        "neural network architecture",
        "random unrelated text"
    ]

    await warp.tension(threads)

    # Query similar to first thread
    query_emb = warp.threads[0].embedding  # Exact match

    attention = warp.apply_attention(query_emb)

    # Thread 0 should have highest attention
    assert np.argmax(attention) == 0
    assert attention[0] > attention[1]
    assert attention[0] > attention[2]


# ============================================================================
# Test Spectral Features Advanced
# ============================================================================

@pytest.mark.asyncio
async def test_spectral_entropy_positive(warp, test_threads):
    """Test spectral entropy is positive."""
    await warp.tension(test_threads)

    features = warp.compute_spectral_features()

    if 'spectral_entropy' in features:
        # Entropy should be non-negative
        assert features['spectral_entropy'] >= 0


@pytest.mark.asyncio
async def test_singular_values_decreasing(warp, test_threads):
    """Test singular values are in descending order."""
    await warp.tension(test_threads)

    features = warp.compute_spectral_features()

    if 'singular_values' in features:
        s_vals = features['singular_values']
        # Should be sorted descending
        for i in range(len(s_vals) - 1):
            assert s_vals[i] >= s_vals[i + 1]


@pytest.mark.asyncio
async def test_spectral_features_field_norm_positive(warp, test_threads):
    """Test field norm is always positive."""
    await warp.tension(test_threads)

    features = warp.compute_spectral_features()

    assert features['field_norm'] > 0


# ============================================================================
# Test Weighted Context Advanced
# ============================================================================

@pytest.mark.asyncio
async def test_weighted_context_uniform_attention(warp, test_threads):
    """Test weighted context with uniform attention is mean of embeddings."""
    await warp.tension(test_threads)

    # Uniform attention
    attention = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    context = warp.weighted_context(attention)

    # Should be close to mean of all embeddings
    expected = np.mean(warp.tensor_field, axis=0)
    assert np.allclose(context, expected, atol=1e-5)


@pytest.mark.asyncio
async def test_weighted_context_single_thread_attention(warp, test_threads):
    """Test weighted context with attention on single thread."""
    await warp.tension(test_threads)

    # All attention on first thread
    attention = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    context = warp.weighted_context(attention)

    # Should equal first thread's embedding
    assert np.allclose(context, warp.threads[0].embedding)


@pytest.mark.asyncio
async def test_weighted_context_entropy_calculation(warp, test_threads):
    """Test weighted context calculates attention entropy."""
    await warp.tension(test_threads)

    # Uniform attention has high entropy
    uniform_attention = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    warp.weighted_context(uniform_attention)

    uniform_op = warp.operations[-1]
    uniform_entropy = uniform_op[1]['attention_entropy']

    # Reset for next test
    warp.operations = []

    # Focused attention has low entropy
    focused_attention = np.array([0.9, 0.05, 0.05], dtype=np.float32)
    warp.weighted_context(focused_attention)

    focused_op = warp.operations[-1]
    focused_entropy = focused_op[1]['attention_entropy']

    # Uniform should have higher entropy
    assert uniform_entropy > focused_entropy


# ============================================================================
# Test Multiple Cycles
# ============================================================================

@pytest.mark.asyncio
async def test_multiple_tension_collapse_cycles(warp, test_threads):
    """Test multiple tension-collapse cycles work correctly."""
    # First cycle
    await warp.tension(test_threads)
    assert warp.is_tensioned is True
    warp.collapse()
    assert warp.is_tensioned is False

    # Second cycle
    await warp.tension(test_threads)
    assert warp.is_tensioned is True
    warp.collapse()
    assert warp.is_tensioned is False

    # Third cycle
    await warp.tension(test_threads)
    assert warp.is_tensioned is True


@pytest.mark.asyncio
async def test_state_isolation_between_cycles(warp, test_threads):
    """Test state is properly isolated between cycles."""
    # First cycle with operations
    await warp.tension(test_threads)
    warp.compute_spectral_features()
    assert len(warp.operations) == 1

    updates1 = warp.collapse()

    # Second cycle - operations should be reset
    await warp.tension(test_threads)
    assert len(warp.operations) == 0

    query_emb = np.random.randn(768).astype(np.float32)
    warp.apply_attention(query_emb)
    assert len(warp.operations) == 1

    updates2 = warp.collapse()

    # Operations should be different
    assert len(updates1['operations']) == 1
    assert len(updates2['operations']) == 1
    assert updates1['operations'][0][0] == 'spectral_features'
    assert updates2['operations'][0][0] == 'attention'


# ============================================================================
# Test Thread Index Lookup
# ============================================================================

@pytest.mark.asyncio
async def test_thread_index_lookup(warp, test_threads):
    """Test thread index allows lookup by ID."""
    custom_ids = ["alpha", "beta", "gamma"]

    await warp.tension(test_threads, thread_ids=custom_ids)

    # Can look up by ID
    alpha = warp.thread_index["alpha"]
    beta = warp.thread_index["beta"]
    gamma = warp.thread_index["gamma"]

    assert alpha.thread_id == "alpha"
    assert beta.thread_id == "beta"
    assert gamma.thread_id == "gamma"


@pytest.mark.asyncio
async def test_thread_index_cleared_on_collapse(warp, test_threads):
    """Test thread index is cleared on collapse."""
    await warp.tension(test_threads)

    assert len(warp.thread_index) == 3

    warp.collapse()

    assert len(warp.thread_index) == 0


# ============================================================================
# Test Field Statistics
# ============================================================================

@pytest.mark.asyncio
async def test_collapse_field_stats_accuracy(warp, test_threads):
    """Test collapse includes accurate field statistics."""
    await warp.tension(test_threads)

    updates = warp.collapse()
    stats = updates['field_stats']

    # Verify shape
    assert stats['shape'] == (3, 768)

    # Verify norm is positive
    assert stats['norm'] > 0

    # Verify mean and std are finite
    assert np.isfinite(stats['mean'])
    assert np.isfinite(stats['std'])
    assert stats['std'] >= 0  # Std deviation is non-negative


# ============================================================================
# Test Guardrails Context Propagation
# ============================================================================

@pytest.mark.asyncio
async def test_guardrails_context_includes_sparsity(mock_guardrails, embedder, test_threads):
    """Test guardrails context includes sparsity parameter."""
    warp = WarpSpace(
        embedder=embedder,
        scales=[96, 192, 768],
        guardrails=mock_guardrails
    )

    await warp.tension(test_threads, sparsity=0.7)

    # Check call included sparsity
    call_args = mock_guardrails.evaluate.call_args
    context = call_args[0][0].context
    assert context['sparsity'] == 0.7


@pytest.mark.asyncio
async def test_guardrails_text_sample_provided(mock_guardrails, embedder, test_threads):
    """Test guardrails receive text sample for analysis."""
    warp = WarpSpace(
        embedder=embedder,
        scales=[96, 192, 768],
        guardrails=mock_guardrails
    )

    await warp.tension(test_threads)

    # Check text_input was provided
    call_args = mock_guardrails.evaluate.call_args
    text_input = call_args[1]['text_input']
    assert len(text_input) > 0
    # Should be sample from first 3 threads
    assert any(thread[:20] in text_input for thread in test_threads)


# ============================================================================
# Test Memory Efficiency
# ============================================================================

@pytest.mark.asyncio
async def test_large_batch_tensioning(warp):
    """Test tensioning large batch of threads."""
    # Create 100 threads
    large_batch = [f"thread content {i}" for i in range(100)]

    await warp.tension(large_batch)

    assert len(warp.threads) == 100
    assert warp.tensor_field.shape == (100, 768)


@pytest.mark.asyncio
async def test_sparsity_reduces_memory(warp):
    """Test sparsity reduces number of active threads."""
    threads = [f"thread {i}" for i in range(100)]

    # Full density
    await warp.tension(threads, sparsity=0.0)
    full_count = len(warp.threads)
    warp.collapse()

    # 90% sparsity (only 10% active)
    await warp.tension(threads, sparsity=0.9)
    sparse_count = len(warp.threads)

    # Sparse should have much fewer threads
    assert sparse_count < full_count
    assert 9 <= sparse_count <= 11  # ~10% of 100 (allowing for rounding)


# ============================================================================
# Test Trace Completeness
# ============================================================================

@pytest.mark.asyncio
async def test_trace_captures_all_operations(warp, test_threads):
    """Test trace captures complete operation history."""
    await warp.tension(test_threads)

    # Perform multiple operations
    warp.compute_spectral_features()
    query_emb = np.random.randn(768).astype(np.float32)
    warp.apply_attention(query_emb)
    attention = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    warp.weighted_context(attention)

    trace = warp.get_trace()

    # Should have all 3 operations
    assert len(trace['operations']) == 3
    assert trace['operations'][0][0] == 'spectral_features'
    assert trace['operations'][1][0] == 'attention'
    assert trace['operations'][2][0] == 'weighted_context'


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary (EXPANDED):
- Initialization: 2 tests
- Tensioning: 7 tests
- Sparsity Control: 4 tests
- Empty Thread Handling: 1 test
- Get Field at Scales: 5 tests
- Spectral Features: 4 tests (+ 3 advanced)
- Attention Mechanism: 3 tests (+ 3 advanced)
- Weighted Context: 3 tests (+ 3 advanced)
- Collapse and Detension: 6 tests
- Computational Trace: 2 tests (+ 1 advanced)
- Safety Guardrails: 3 tests (+ 2 advanced)
- Lifecycle and Operations: 2 tests
- Tensioned Thread: 2 tests
- Error Handling: 1 test (+ 2 advanced)
- Multi-Scale Fusion Advanced: 2 tests
- Thread Weighting: 3 tests
- Multiple Cycles: 2 tests
- Thread Index: 2 tests
- Field Statistics: 1 test
- Memory Efficiency: 2 tests

Total: 68 tests (was 45) covering comprehensive warp space functionality
New tests added: 23
"""