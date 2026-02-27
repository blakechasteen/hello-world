"""
Integration test for WarpSpace full lifecycle.

Tests the complete tension → compute → collapse cycle.
"""
import pytest
import numpy as np
from hololoom.warp.space import WarpSpace
from hololoom.embedding.spectral import MatryoshkaEmbeddings


@pytest.mark.asyncio
async def test_warp_space_full_lifecycle():
    """Test complete WarpSpace lifecycle: tension → compute → collapse"""

    # Setup
    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    # Sample thread texts
    thread_texts = [
        "Thompson Sampling balances exploration and exploitation",
        "Neural networks learn hierarchical representations",
        "Attention mechanisms enable context-aware processing"
    ]

    # 1. TENSION - Pull threads into Warp Space
    await warp.tension(thread_texts)

    assert warp.is_tensioned, "WarpSpace should be tensioned"
    assert len(warp.threads) == 3, "Should have 3 threads"
    assert warp.tensor_field is not None, "Tensor field should exist"
    assert warp.tensor_field.shape == (3, 384), "Field should be (3, 384)"

    # 2. COMPUTE - Perform tensor operations
    query_text = ["What is Thompson Sampling?"]
    query_emb = embedder.encode_scales(query_text, size=384)

    compute_results = warp.compute(query_emb[0], compute_spectral=True)

    # Verify compute results
    assert 'spectral_features' in compute_results, "Should have spectral features"
    assert 'attention_weights' in compute_results, "Should have attention weights"
    assert 'context_vector' in compute_results, "Should have context vector"
    assert 'attention_entropy' in compute_results, "Should have attention entropy"
    assert 'metadata' in compute_results, "Should have metadata"

    # Verify attention weights shape and sum to 1
    attention = compute_results['attention_weights']
    assert len(attention) == 3, "Attention should have 3 weights"
    assert np.abs(np.sum(attention) - 1.0) < 1e-6, "Attention should sum to 1.0"

    # Verify context vector shape
    context = compute_results['context_vector']
    assert len(context) == 384, "Context should be 384-dimensional"

    # Verify spectral features exist
    spectral = compute_results['spectral_features']
    assert spectral is not None, "Spectral features should exist"
    assert 'field_norm' in spectral, "Should have field norm"
    assert 'spectral_entropy' in spectral, "Should have spectral entropy"

    # 3. COLLAPSE - Detension and get updates
    updates = warp.collapse()

    assert not warp.is_tensioned, "WarpSpace should be detensioned"
    assert len(warp.threads) == 0, "Threads should be cleared"
    assert warp.tensor_field is None, "Tensor field should be cleared"

    # Verify updates structure
    assert 'threads' in updates, "Should have thread updates"
    assert 'operations' in updates, "Should have operations log"
    assert 'field_stats' in updates, "Should have field stats"

    assert len(updates['threads']) == 3, "Should have 3 thread updates"
    assert len(updates['operations']) > 0, "Should have recorded operations"


@pytest.mark.asyncio
async def test_warp_space_compute_without_tension():
    """Test that compute() raises error if not tensioned"""

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    query_emb = np.random.randn(384)

    with pytest.raises(RuntimeError, match="not tensioned"):
        warp.compute(query_emb)


@pytest.mark.asyncio
async def test_warp_space_sparsity():
    """Test WarpSpace with sparsity enforcement"""

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    thread_texts = [
        "Thread 1",
        "Thread 2",
        "Thread 3",
        "Thread 4",
        "Thread 5"
    ]

    # Tension with 60% sparsity (only 40% threads active)
    await warp.tension(thread_texts, sparsity=0.6)

    # Should have only 2 threads (40% of 5 = 2)
    assert len(warp.threads) <= 2, f"With 60% sparsity, should have ≤2 threads, got {len(warp.threads)}"
    assert warp.is_tensioned, "Should be tensioned"


@pytest.mark.asyncio
async def test_warp_space_multi_scale():
    """Test WarpSpace with multiple embedding scales"""

    embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    warp = WarpSpace(embedder, scales=[96, 192, 384])

    thread_texts = ["Sample thread"]

    await warp.tension(thread_texts)

    # Get field at different scales
    field_96 = warp.get_field(scale=96)
    field_192 = warp.get_field(scale=192)
    field_384 = warp.get_field(scale=384)

    assert field_96.shape == (1, 96), "96-scale field should be (1, 96)"
    assert field_192.shape == (1, 192), "192-scale field should be (1, 192)"
    assert field_384.shape == (1, 384), "384-scale field should be (1, 384)"

    warp.collapse()


if __name__ == "__main__":
    import asyncio

    async def run_tests():
        print("="*80)
        print("WarpSpace Lifecycle Integration Tests")
        print("="*80 + "\n")

        print("Test 1: Full lifecycle (tension → compute → collapse)")
        await test_warp_space_full_lifecycle()
        print("✓ PASSED\n")

        print("Test 2: Compute without tension (error handling)")
        await test_warp_space_compute_without_tension()
        print("✓ PASSED\n")

        print("Test 3: Sparsity enforcement")
        await test_warp_space_sparsity()
        print("✓ PASSED\n")

        print("Test 4: Multi-scale operations")
        await test_warp_space_multi_scale()
        print("✓ PASSED\n")

        print("="*80)
        print("All WarpSpace lifecycle tests passed!")
        print("="*80)

    asyncio.run(run_tests())
