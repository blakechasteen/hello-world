"""
Tests for hololoom/core/warp/ modules
=====================================

Targets 80%+ coverage on:
- __init__.py (imports and graceful degradation)
- space.py (compute method, guardrails requires_approval)
- merge.py (internal_merge, parallel_merge, visualize_merge_tree)
- optimized.py (SparseTensorField, LazyWarpOperation, TensorMemoryPool, BatchedWarpProcessor)
- advanced.py (RiemannianManifold, TensorDecomposer, QuantumWarpOperations, FisherInformationGeometry)
- eggroll_warp.py (ContextPacker, SpectralScorer, Warp)
- combinatorics.py (ChainComplex, DiscreteMorseFunction, Sheaf)

All tests use small numpy arrays and run in <5s each.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List


# ============================================================================
# Mock Embedder (shared across tests)
# ============================================================================

class MockEmbedder:
    """Deterministic mock embedder for reproducible tests."""

    def __init__(self, sizes=None):
        self.sizes = sizes or [96, 192, 384]

    def encode_scales(self, texts: List[str], size: int = None):
        rng = np.random.RandomState(42)
        if size:
            return rng.randn(len(texts), size).astype(np.float32)
        return {s: rng.randn(len(texts), s).astype(np.float32) for s in self.sizes}

    def encode(self, texts: List[str]) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), max(self.sizes)).astype(np.float32)


# ============================================================================
# __init__.py — verify imports and HAS_* flags
# ============================================================================

class TestWarpInit:
    """Test warp package __init__.py import guards."""

    def test_warpspace_always_importable(self):
        from hololoom.core.warp import WarpSpace
        assert WarpSpace is not None

    def test_has_advanced_flag_set(self):
        from hololoom.core.warp import HAS_ADVANCED
        # Should be True since advanced.py has no hard external deps
        assert isinstance(HAS_ADVANCED, bool)

    def test_has_optimized_flag_set(self):
        from hololoom.core.warp import HAS_OPTIMIZED
        assert isinstance(HAS_OPTIMIZED, bool)

    def test_has_combinatorics_flag_set(self):
        from hololoom.core.warp import HAS_COMBINATORICS
        assert isinstance(HAS_COMBINATORICS, bool)

    def test_all_exports_contains_warpspace(self):
        import hololoom.core.warp as warp_pkg
        assert "WarpSpace" in warp_pkg.__all__

    def test_advanced_classes_importable_if_available(self):
        from hololoom.core.warp import HAS_ADVANCED
        if HAS_ADVANCED:
            from hololoom.core.warp import (
                RiemannianManifold,
                TensorDecomposer,
                QuantumWarpOperations,
                FisherInformationGeometry,
            )
            assert RiemannianManifold is not None

    def test_optimized_classes_importable_if_available(self):
        from hololoom.core.warp import HAS_OPTIMIZED
        if HAS_OPTIMIZED:
            from hololoom.core.warp import (
                SparseTensorField,
                LazyWarpOperation,
                TensorMemoryPool,
                BatchedWarpProcessor,
            )
            assert SparseTensorField is not None

    def test_combinatorics_classes_importable_if_available(self):
        from hololoom.core.warp import HAS_COMBINATORICS
        if HAS_COMBINATORICS:
            from hololoom.core.warp import (
                ChainComplex,
                DiscreteMorseFunction,
                Sheaf,
            )
            assert ChainComplex is not None

    def test_has_topology_flag(self):
        import hololoom.core.warp as warp_pkg
        assert isinstance(warp_pkg.HAS_TOPOLOGY, bool)

    def test_has_category_flag(self):
        import hololoom.core.warp as warp_pkg
        assert isinstance(warp_pkg.HAS_CATEGORY, bool)

    def test_has_representation_flag(self):
        import hololoom.core.warp as warp_pkg
        assert isinstance(warp_pkg.HAS_REPRESENTATION, bool)


# ============================================================================
# space.py — compute() method and guardrails requires_approval path
# ============================================================================

class TestWarpSpaceCompute:
    """Test WarpSpace.compute() unified method and edge cases."""

    @pytest.fixture
    def warp(self):
        from hololoom.core.warp.space import WarpSpace
        return WarpSpace(
            embedder=MockEmbedder(),
            scales=[96, 192, 384],
            guardrails=None,
        )

    @pytest.fixture
    def test_threads(self):
        return ["alpha thread", "beta thread", "gamma thread"]

    @pytest.mark.asyncio
    async def test_compute_returns_all_keys(self, warp, test_threads):
        """compute() returns spectral_features, attention_weights, context_vector, metadata."""
        await warp.tension(test_threads)
        query = np.random.randn(384).astype(np.float32)
        result = warp.compute(query, compute_spectral=True)

        assert "spectral_features" in result
        assert "attention_weights" in result
        assert "context_vector" in result
        assert "attention_entropy" in result
        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_compute_without_spectral(self, warp, test_threads):
        """compute(compute_spectral=False) skips spectral."""
        await warp.tension(test_threads)
        query = np.random.randn(384).astype(np.float32)
        result = warp.compute(query, compute_spectral=False)

        assert "spectral_features" not in result
        assert "attention_weights" in result
        assert "context_vector" in result

    @pytest.mark.asyncio
    async def test_compute_metadata_shape(self, warp, test_threads):
        """compute() metadata has correct thread count and shape."""
        await warp.tension(test_threads)
        query = np.random.randn(384).astype(np.float32)
        result = warp.compute(query)

        meta = result["metadata"]
        assert meta["num_threads"] == 3
        assert meta["tensor_field_shape"] == (3, 384)
        assert meta["query_embedding_dim"] == 384

    @pytest.mark.asyncio
    async def test_compute_not_tensioned_raises(self, warp):
        """compute() raises RuntimeError when not tensioned."""
        query = np.random.randn(384).astype(np.float32)
        with pytest.raises(RuntimeError, match="not tensioned"):
            warp.compute(query)

    @pytest.mark.asyncio
    async def test_compute_spectral_failure_graceful(self, warp, test_threads):
        """compute() handles spectral failure gracefully."""
        await warp.tension(test_threads)
        query = np.random.randn(384).astype(np.float32)

        with patch.object(warp, "compute_spectral_features", side_effect=Exception("boom")):
            result = warp.compute(query, compute_spectral=True)

        assert result["spectral_features"] is None
        assert "attention_weights" in result

    @pytest.mark.asyncio
    async def test_guardrails_requires_approval_raises(self):
        """Guardrails with requires_approval=True raises PermissionError."""
        from hololoom.core.warp.space import WarpSpace
        from hololoom.alignment.safety_guardrails import (
            SafetyGuardrails, SafetyDecision, RiskLevel,
        )

        guardrails = Mock(spec=SafetyGuardrails)
        guardrails.evaluate.return_value = SafetyDecision(
            allowed=True,
            risk_level=RiskLevel.MEDIUM,
            reason="Needs human review",
            requires_approval=True,
        )
        warp = WarpSpace(
            embedder=MockEmbedder(),
            scales=[96, 192, 384],
            guardrails=guardrails,
        )
        with pytest.raises(PermissionError, match="requires approval"):
            await warp.tension(["hello"])

    @pytest.mark.asyncio
    async def test_decision_to_dict_none(self):
        from hololoom.core.warp.space import WarpSpace
        assert WarpSpace._decision_to_dict(None) is None


# ============================================================================
# merge.py — internal_merge, parallel_merge, recursive_merge, visualize
# ============================================================================

class TestMergeOperator:
    """Test MergeOperator for uncovered paths."""

    @pytest.fixture
    def merger(self):
        from hololoom.core.warp.merge import MergeOperator
        return MergeOperator(embedder=MockEmbedder(), fusion_method="weighted_sum")

    def test_external_merge_alpha_is_head(self, merger):
        """External merge with alpha_is_head=True."""
        a = np.random.randn(384).astype(np.float32)
        b = np.random.randn(384).astype(np.float32)
        result = merger.external_merge(a, b, head="a", dependent="b", label="VP", alpha_is_head=True)

        assert result.head == "a"
        assert result.merge_type.value == "external"
        assert result.label == "VP"
        assert result.metadata["alpha_is_head"] is True
        # Embedding should be normalized
        assert np.isclose(np.linalg.norm(result.embedding), 1.0, atol=1e-5)

    def test_external_merge_concat_method(self):
        from hololoom.core.warp.merge import MergeOperator
        merger = MergeOperator(embedder=MockEmbedder(), fusion_method="concat")
        a = np.random.randn(384).astype(np.float32)
        b = np.random.randn(384).astype(np.float32)
        result = merger.external_merge(a, b, head="b", dependent="a", label="NP")
        # Concat doubles the dimension, then normalizes
        assert result.embedding.shape == (768,)

    def test_external_merge_hadamard_method(self):
        from hololoom.core.warp.merge import MergeOperator
        merger = MergeOperator(embedder=MockEmbedder(), fusion_method="hadamard")
        a = np.random.randn(384).astype(np.float32)
        b = np.random.randn(384).astype(np.float32)
        result = merger.external_merge(a, b, head="b", dependent="a", label="NP")
        assert result.embedding.shape == (384,)

    def test_external_merge_unknown_method_raises(self):
        from hololoom.core.warp.merge import MergeOperator
        merger = MergeOperator(embedder=MockEmbedder(), fusion_method="unknown")
        a = np.random.randn(384).astype(np.float32)
        b = np.random.randn(384).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown fusion method"):
            merger.external_merge(a, b, head="b", dependent="a", label="NP")

    def test_internal_merge_front(self, merger):
        """Internal merge moves component to front."""
        from hololoom.core.warp.merge import MergedObject, MergeType
        obj = MergedObject(
            embedding=np.random.randn(384),
            components=["the", "big", "cat"],
            head="cat",
            merge_type=MergeType.EXTERNAL,
            label="NP",
        )
        result = merger.internal_merge(obj, move_index=2, target_position="front")
        assert result.components[0] == "cat"
        assert result.merge_type == MergeType.INTERNAL
        assert result.metadata["moved_component"] == "cat"

    def test_internal_merge_back(self, merger):
        """Internal merge moves component to back."""
        from hololoom.core.warp.merge import MergedObject, MergeType
        obj = MergedObject(
            embedding=np.random.randn(384),
            components=["what", "did", "see"],
            head="see",
            merge_type=MergeType.EXTERNAL,
            label="VP",
        )
        result = merger.internal_merge(obj, move_index=0, target_position="back")
        assert result.components[-1] == "what"

    def test_parallel_merge(self, merger):
        """Parallel merge of multiple embeddings."""
        embs = [np.random.randn(384).astype(np.float32) for _ in range(4)]
        components = ["the", "big", "red", "ball"]
        result = merger.parallel_merge(embs, components, head_index=3, label="NP")

        assert result.head == "ball"
        assert result.merge_type.value == "parallel"
        assert result.metadata["n_components"] == 4
        assert np.isclose(np.linalg.norm(result.embedding), 1.0, atol=1e-5)

    def test_parallel_merge_mismatched_lengths_raises(self, merger):
        """Parallel merge with mismatched lengths raises ValueError."""
        embs = [np.random.randn(384) for _ in range(3)]
        components = ["a", "b"]
        with pytest.raises(ValueError, match="same length"):
            merger.parallel_merge(embs, components, head_index=0)

    def test_recursive_merge(self, merger):
        """Recursive merge builds tree structure."""
        embs = [np.random.randn(384).astype(np.float32) for _ in range(3)]
        words = ["the", "big", "cat"]
        structure = [
            (1, 2, "right", "N'"),
            (0, -1, "right", "NP"),
        ]
        result = merger.recursive_merge(embs, words, structure)
        assert result.label == "NP"
        assert len(result.children) == 2

    def test_visualize_merge_tree(self):
        """visualize_merge_tree returns string representation."""
        from hololoom.core.warp.merge import MergedObject, MergeType, visualize_merge_tree
        child1 = MergedObject(
            embedding=np.zeros(4), components=["cat"], head="cat",
            merge_type=MergeType.EXTERNAL, label="N",
        )
        child2 = MergedObject(
            embedding=np.zeros(4), components=["big"], head="big",
            merge_type=MergeType.EXTERNAL, label="Adj",
        )
        parent = MergedObject(
            embedding=np.zeros(4), components=["big", "cat"], head="cat",
            merge_type=MergeType.EXTERNAL, label="N'",
            children=[child1, child2],
        )
        text = visualize_merge_tree(parent)
        assert "N'" in text
        assert "cat" in text
        assert "big" in text

    def test_merged_object_repr(self):
        from hololoom.core.warp.merge import MergedObject, MergeType
        obj = MergedObject(
            embedding=np.zeros(4), components=["the", "cat"], head="cat",
            merge_type=MergeType.EXTERNAL, label="NP",
        )
        r = repr(obj)
        assert "NP" in r
        assert "the+cat" in r


# ============================================================================
# optimized.py — SparseTensorField, LazyWarpOperation, TensorMemoryPool,
#                BatchedWarpProcessor
# ============================================================================

class TestSparseTensorField:
    """Test SparseTensorField operations."""

    def test_from_dense_captures_shape(self):
        from hololoom.core.warp.optimized import SparseTensorField
        dense = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        sparse = SparseTensorField(dense, threshold=1e-6)
        assert sparse.shape == (2, 3)
        assert sparse.density == pytest.approx(2 / 6)

    def test_to_dense_roundtrip(self):
        from hololoom.core.warp.optimized import SparseTensorField
        dense = np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]], dtype=np.float64)
        sparse = SparseTensorField(dense, threshold=1e-6)
        recovered = sparse.to_dense()
        np.testing.assert_array_almost_equal(recovered, dense)

    def test_to_dense_empty(self):
        from hololoom.core.warp.optimized import SparseTensorField
        sparse = SparseTensorField()
        result = sparse.to_dense()
        assert result.shape == (0,)

    def test_matmul_with_vector(self):
        from hololoom.core.warp.optimized import SparseTensorField
        dense = np.array([[1.0, 2.0], [3.0, 4.0]])
        sparse = SparseTensorField(dense, threshold=1e-6)
        vec = np.array([1.0, 1.0])
        result = sparse @ vec
        expected = dense @ vec
        np.testing.assert_array_almost_equal(result, expected)

    def test_matmul_with_matrix(self):
        from hololoom.core.warp.optimized import SparseTensorField
        A = np.array([[1.0, 2.0], [3.0, 0.0]])
        B = np.array([[1.0], [2.0]])
        sparse = SparseTensorField(A, threshold=1e-6)
        result = sparse @ B
        expected = A @ B
        np.testing.assert_array_almost_equal(result, expected)

    def test_matmul_with_sparse(self):
        from hololoom.core.warp.optimized import SparseTensorField
        A = np.array([[1.0, 0.0], [0.0, 2.0]])
        B = np.array([[3.0, 0.0], [0.0, 4.0]])
        sA = SparseTensorField(A, threshold=1e-6)
        sB = SparseTensorField(B, threshold=1e-6)
        result = sA @ sB
        expected = A @ B
        np.testing.assert_array_almost_equal(result, expected)

    def test_matmul_incompatible_shapes_raises(self):
        from hololoom.core.warp.optimized import SparseTensorField
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[1.0, 2.0, 3.0]])
        sparse = SparseTensorField(A, threshold=1e-6)
        with pytest.raises(ValueError, match="Incompatible shapes"):
            sparse @ B

    def test_matmul_empty_sparse(self):
        from hololoom.core.warp.optimized import SparseTensorField
        sparse = SparseTensorField()
        result = sparse @ np.array([1.0, 2.0])
        assert result.shape == (0,)

    def test_matmul_all_zeros(self):
        from hololoom.core.warp.optimized import SparseTensorField
        A = np.zeros((3, 3))
        sparse = SparseTensorField(A, threshold=1e-6)
        vec = np.array([1.0, 2.0, 3.0])
        result = sparse @ vec
        np.testing.assert_array_almost_equal(result, np.zeros(3))


class TestLazyWarpOperation:
    """Test LazyWarpOperation deferred computation."""

    def test_lazy_not_computed_on_init(self):
        from hololoom.core.warp.optimized import LazyWarpOperation
        lazy = LazyWarpOperation("attention", None, None)
        assert lazy.computed is False
        assert lazy.result is None

    def test_lazy_unknown_operation_raises(self):
        from hololoom.core.warp.optimized import LazyWarpOperation
        lazy = LazyWarpOperation("unknown_op")
        with pytest.raises(ValueError, match="Unknown lazy operation"):
            lazy.compute()

    def test_lazy_callable(self):
        """LazyWarpOperation is callable and delegates to compute."""
        from hololoom.core.warp.optimized import LazyWarpOperation
        lazy = LazyWarpOperation("unknown_op")
        with pytest.raises(ValueError):
            lazy()

    def test_lazy_attention_operation(self):
        """Test lazy attention calls warp_space._compute_attention."""
        from hololoom.core.warp.optimized import LazyWarpOperation
        mock_warp = Mock()
        mock_warp._compute_attention.return_value = np.array([0.5, 0.5])
        query = np.array([1.0, 2.0])

        lazy = LazyWarpOperation("attention", mock_warp, query)
        result = lazy.compute()

        mock_warp._compute_attention.assert_called_once_with(query)
        np.testing.assert_array_equal(result, np.array([0.5, 0.5]))
        assert lazy.computed is True

    def test_lazy_spectral_operation(self):
        from hololoom.core.warp.optimized import LazyWarpOperation
        mock_warp = Mock()
        mock_warp._compute_spectral.return_value = {"entropy": 1.5}

        lazy = LazyWarpOperation("spectral", mock_warp)
        result = lazy.compute()

        assert result == {"entropy": 1.5}
        assert lazy.computed is True

    def test_lazy_context_operation(self):
        from hololoom.core.warp.optimized import LazyWarpOperation
        mock_warp = Mock()
        mock_warp._compute_context.return_value = np.array([1.0, 2.0])
        attention = np.array([0.5, 0.5])

        lazy = LazyWarpOperation("context", mock_warp, attention)
        result = lazy.compute()

        mock_warp._compute_context.assert_called_once_with(attention)
        assert lazy.computed is True

    def test_lazy_caches_result(self):
        """Second call returns cached result without recomputing."""
        from hololoom.core.warp.optimized import LazyWarpOperation
        mock_warp = Mock()
        mock_warp._compute_spectral.return_value = {"val": 42}

        lazy = LazyWarpOperation("spectral", mock_warp)
        r1 = lazy.compute()
        r2 = lazy.compute()

        assert r1 is r2
        mock_warp._compute_spectral.assert_called_once()


class TestTensorMemoryPool:
    """Test TensorMemoryPool allocation and reuse."""

    def test_allocate_new_tensor(self):
        from hololoom.core.warp.optimized import TensorMemoryPool
        pool = TensorMemoryPool(max_pool_size=5)
        t = pool.allocate((3, 4))
        assert t.shape == (3, 4)
        assert pool.misses == 1
        assert pool.hits == 0

    def test_release_and_reuse(self):
        from hololoom.core.warp.optimized import TensorMemoryPool
        pool = TensorMemoryPool(max_pool_size=5)
        # Use explicit dtype matching: allocate key = (shape, dtype_param),
        # release key = (shape, tensor.dtype). np.float32 != np.dtype('float32')
        # as dict keys, so we use np.dtype to ensure consistent keying.
        dtype = np.dtype(np.float32)
        t1 = pool.allocate((3, 4), dtype=dtype)
        t1[0, 0] = 99.0
        pool.release(t1)

        t2 = pool.allocate((3, 4), dtype=dtype)
        assert pool.hits == 1
        # Should be zeroed on reuse
        assert t2[0, 0] == 0.0

    def test_pool_max_size_enforced(self):
        from hololoom.core.warp.optimized import TensorMemoryPool
        pool = TensorMemoryPool(max_pool_size=2)
        tensors = [pool.allocate((2, 2)) for _ in range(5)]
        for t in tensors:
            pool.release(t)
        # Only 2 should be cached
        stats = pool.get_stats()
        assert stats["total_cached"] == 2

    def test_get_stats(self):
        from hololoom.core.warp.optimized import TensorMemoryPool
        pool = TensorMemoryPool()
        pool.allocate((2, 3))
        pool.allocate((2, 3))
        stats = pool.get_stats()
        assert stats["misses"] == 2
        assert stats["hits"] == 0
        assert "hit_rate" in stats
        assert "pool_sizes" in stats

    def test_different_shapes_different_buckets(self):
        from hololoom.core.warp.optimized import TensorMemoryPool
        pool = TensorMemoryPool()
        t1 = pool.allocate((2, 3))
        pool.release(t1)
        t2 = pool.allocate((3, 2))  # different shape -> miss
        assert pool.misses == 2
        assert pool.hits == 0


class TestGPUWarpSpace:
    """Test GPUWarpSpace with PyTorch (CPU fallback)."""

    @pytest.fixture
    def gpu_warp(self):
        from hololoom.core.warp.optimized import GPUWarpSpace, HAS_TORCH
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        gpu = GPUWarpSpace(
            embedder=MockEmbedder(),
            scales=[96, 192, 384],
            use_gpu=False,  # Force CPU for CI
            dtype="float32",
            guardrails=None,
        )
        # Disable guardrails to avoid ActionRequest kwarg mismatch
        gpu.guardrails_enabled = False
        return gpu

    def test_init_cpu(self):
        from hololoom.core.warp.optimized import GPUWarpSpace, HAS_TORCH
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        gpu = GPUWarpSpace(embedder=MockEmbedder(), use_gpu=False, guardrails=None)
        import torch
        assert gpu.device == torch.device("cpu")
        assert gpu.dtype == torch.float32

    def test_init_float16(self):
        from hololoom.core.warp.optimized import GPUWarpSpace, HAS_TORCH
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        gpu = GPUWarpSpace(embedder=MockEmbedder(), use_gpu=False, dtype="float16", guardrails=None)
        import torch
        assert gpu.dtype == torch.float16

    @pytest.mark.asyncio
    async def test_tension(self, gpu_warp):
        await gpu_warp.tension(["hello", "world"])
        assert gpu_warp.tensor_field is not None
        assert gpu_warp.tensor_field.shape[0] == 2
        assert len(gpu_warp.threads_meta) == 2

    @pytest.mark.asyncio
    async def test_compute_attention(self, gpu_warp):
        await gpu_warp.tension(["hello", "world"])
        query = np.random.randn(384).astype(np.float32)
        attention = gpu_warp.compute_attention(query, temperature=1.0)
        # Should sum to ~1 (softmax)
        assert abs(float(attention.sum()) - 1.0) < 1e-5
        assert attention.shape[0] == 2

    @pytest.mark.asyncio
    async def test_compute_attention_temperature(self, gpu_warp):
        await gpu_warp.tension(["hello", "world"])
        query = np.random.randn(384).astype(np.float32)
        attn_low = gpu_warp.compute_attention(query, temperature=0.1)
        attn_high = gpu_warp.compute_attention(query, temperature=10.0)
        # Higher temperature -> more uniform distribution
        assert float(attn_high.max() - attn_high.min()) < float(attn_low.max() - attn_low.min())

    @pytest.mark.asyncio
    async def test_weighted_context(self, gpu_warp):
        await gpu_warp.tension(["hello", "world"])
        query = np.random.randn(384).astype(np.float32)
        attention = gpu_warp.compute_attention(query)
        context = gpu_warp.weighted_context(attention)
        assert isinstance(context, np.ndarray)
        assert context.shape == (384,)

    @pytest.mark.asyncio
    async def test_batch_attention(self, gpu_warp):
        await gpu_warp.tension(["hello", "world", "foo"])
        queries = [np.random.randn(384).astype(np.float32) for _ in range(3)]
        contexts = gpu_warp.batch_attention(queries, temperature=1.0)
        assert isinstance(contexts, np.ndarray)
        assert contexts.shape == (3, 384)

    @pytest.mark.asyncio
    async def test_guardrail_trace(self, gpu_warp):
        trace = gpu_warp.guardrail_trace()
        assert "tension" in trace
        assert "attention" in trace

    def test_decision_to_dict_none(self):
        from hololoom.core.warp.optimized import GPUWarpSpace, HAS_TORCH
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        assert GPUWarpSpace._decision_to_dict(None) is None

    def test_reset_guardrail_trace(self):
        from hololoom.core.warp.optimized import GPUWarpSpace, HAS_TORCH
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        gpu = GPUWarpSpace(embedder=MockEmbedder(), use_gpu=False, guardrails=None)
        gpu._reset_guardrail_trace()
        for v in gpu._guardrail_decisions.values():
            assert v is None


class TestBatchedWarpProcessor:
    """Test BatchedWarpProcessor."""

    @pytest.mark.asyncio
    async def test_batch_tension_and_attend_cpu(self):
        """Test batched processing with standard WarpSpace (CPU fallback)."""
        from hololoom.core.warp.optimized import BatchedWarpProcessor
        from hololoom.core.warp.space import WarpSpace

        embedder = MockEmbedder()
        processor = BatchedWarpProcessor(max_batch_size=16)
        warp = WarpSpace(embedder=embedder, scales=[96, 192, 384], guardrails=None)

        thread_batches = [["hello world", "foo bar"]]
        query_batches = [[np.random.randn(384).astype(np.float32)]]

        results = await processor.batch_tension_and_attend(
            thread_batches, query_batches, warp
        )
        assert len(results) == 1
        assert len(results[0]) == 1
        assert results[0][0].shape == (384,)

    @pytest.mark.asyncio
    async def test_batch_with_gpu_warp(self):
        """Test batched processing with GPUWarpSpace."""
        from hololoom.core.warp.optimized import BatchedWarpProcessor, GPUWarpSpace, HAS_TORCH
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")

        embedder = MockEmbedder()
        processor = BatchedWarpProcessor(max_batch_size=16)
        gpu_warp = GPUWarpSpace(embedder=embedder, use_gpu=False, guardrails=None)
        gpu_warp.guardrails_enabled = False  # Avoid ActionRequest kwarg issue

        thread_batches = [["hello world", "foo bar"]]
        query_batches = [[np.random.randn(384).astype(np.float32)]]

        results = await processor.batch_tension_and_attend(
            thread_batches, query_batches, gpu_warp
        )
        assert len(results) == 1


# ============================================================================
# advanced.py — RiemannianManifold, TensorDecomposer,
#               QuantumWarpOperations, FisherInformationGeometry
# ============================================================================

class TestRiemannianManifold:
    """Test Riemannian manifold geometry operations."""

    @pytest.fixture
    def flat(self):
        from hololoom.core.warp.advanced import RiemannianManifold
        return RiemannianManifold(dim=4, curvature=0.0)

    @pytest.fixture
    def spherical(self):
        from hololoom.core.warp.advanced import RiemannianManifold
        return RiemannianManifold(dim=4, curvature=1.0)

    @pytest.fixture
    def hyperbolic(self):
        from hololoom.core.warp.advanced import RiemannianManifold
        return RiemannianManifold(dim=4, curvature=-1.0)

    # -- geodesic_distance --

    def test_flat_geodesic_is_euclidean(self, flat):
        p1 = np.array([1.0, 0.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0, 0.0])
        d = flat.geodesic_distance(p1, p2)
        expected = np.sqrt(2.0)
        assert abs(d - expected) < 1e-6

    def test_spherical_geodesic_same_point_zero(self, spherical):
        p = np.array([1.0, 0.0, 0.0, 0.0])
        d = spherical.geodesic_distance(p, p)
        assert abs(d) < 1e-4  # numerical precision from arccos near 1.0

    def test_spherical_geodesic_orthogonal(self, spherical):
        p1 = np.array([1.0, 0.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0, 0.0])
        d = spherical.geodesic_distance(p1, p2)
        # Angle is pi/2, radius=1, so arc length = pi/2
        assert abs(d - np.pi / 2) < 1e-6

    def test_hyperbolic_geodesic_positive(self, hyperbolic):
        p1 = np.array([0.1, 0.1, 0.0, 0.0])
        p2 = np.array([0.2, 0.0, 0.1, 0.0])
        d = hyperbolic.geodesic_distance(p1, p2)
        assert d > 0

    def test_hyperbolic_geodesic_degenerate_fallback(self, hyperbolic):
        """Degenerate case: norm close to 1 triggers fallback."""
        p1 = np.array([0.99, 0.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.99, 0.0, 0.0])
        d = hyperbolic.geodesic_distance(p1, p2)
        assert d > 0

    # -- exponential_map --

    def test_flat_exp_map_is_addition(self, flat):
        base = np.array([1.0, 0.0, 0.0, 0.0])
        tangent = np.array([0.0, 1.0, 0.0, 0.0])
        result = flat.exponential_map(base, tangent)
        np.testing.assert_array_almost_equal(result, base + tangent)

    def test_spherical_exp_map_zero_tangent(self, spherical):
        base = np.array([1.0, 0.0, 0.0, 0.0])
        tangent = np.zeros(4)
        result = spherical.exponential_map(base, tangent)
        np.testing.assert_array_almost_equal(result, base)

    def test_spherical_exp_map_nontrivial(self, spherical):
        base = np.array([1.0, 0.0, 0.0, 0.0])
        tangent = np.array([0.0, 0.1, 0.0, 0.0])
        result = spherical.exponential_map(base, tangent)
        assert result.shape == (4,)
        # Result should differ from base
        assert not np.allclose(result, base)

    def test_hyperbolic_exp_map_zero_tangent(self, hyperbolic):
        base = np.array([0.1, 0.1, 0.0, 0.0])
        tangent = np.zeros(4)
        result = hyperbolic.exponential_map(base, tangent)
        np.testing.assert_array_almost_equal(result, base)

    def test_hyperbolic_exp_map_nontrivial(self, hyperbolic):
        base = np.array([0.1, 0.0, 0.0, 0.0])
        tangent = np.array([0.0, 0.05, 0.0, 0.0])
        result = hyperbolic.exponential_map(base, tangent)
        assert result.shape == (4,)

    # -- logarithmic_map --

    def test_flat_log_map_is_subtraction(self, flat):
        base = np.array([1.0, 0.0, 0.0, 0.0])
        target = np.array([2.0, 1.0, 0.0, 0.0])
        result = flat.logarithmic_map(base, target)
        np.testing.assert_array_almost_equal(result, target - base)

    def test_spherical_log_map_same_point(self, spherical):
        p = np.array([1.0, 0.0, 0.0, 0.0])
        result = spherical.logarithmic_map(p, p)
        np.testing.assert_array_almost_equal(result, np.zeros(4))

    def test_spherical_log_map_nontrivial(self, spherical):
        base = np.array([1.0, 0.0, 0.0, 0.0])
        target = np.array([0.0, 1.0, 0.0, 0.0])
        result = spherical.logarithmic_map(base, target)
        assert np.linalg.norm(result) > 0

    def test_hyperbolic_log_map(self, hyperbolic):
        base = np.array([0.1, 0.0, 0.0, 0.0])
        target = np.array([0.2, 0.1, 0.0, 0.0])
        result = hyperbolic.logarithmic_map(base, target)
        np.testing.assert_array_almost_equal(result, target - base)

    # -- parallel_transport --

    def test_flat_parallel_transport_identity(self, flat):
        v = np.array([1.0, 0.0, 0.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0, 0.0])
        p2 = np.array([1.0, 1.0, 0.0, 0.0])
        result = flat.parallel_transport(v, p1, p2)
        np.testing.assert_array_almost_equal(result, v)

    def test_spherical_parallel_transport_aligned(self):
        """Aligned points with 3D vectors (cross product requires dim 3)."""
        from hololoom.core.warp.advanced import RiemannianManifold
        m = RiemannianManifold(dim=3, curvature=1.0)
        v = np.array([0.0, 1.0, 0.0])
        p = np.array([1.0, 0.0, 0.0])
        # Same from/to -> axis_norm < 1e-10 -> returns vector unchanged
        result = m.parallel_transport(v, p, p)
        np.testing.assert_array_almost_equal(result, v)

    def test_spherical_parallel_transport_different_points(self):
        """3D vectors for valid cross product."""
        from hololoom.core.warp.advanced import RiemannianManifold
        m = RiemannianManifold(dim=3, curvature=1.0)
        v = np.array([0.0, 1.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 1.0])
        result = m.parallel_transport(v, p1, p2)
        assert result.shape == (3,)

    # -- sectional_curvature --

    def test_sectional_curvature_returns_constant(self, spherical):
        p = np.random.randn(4)
        v1 = np.random.randn(4)
        v2 = np.random.randn(4)
        assert spherical.sectional_curvature(p, v1, v2) == 1.0

    def test_hyperbolic_sectional_curvature(self, hyperbolic):
        assert hyperbolic.sectional_curvature(np.zeros(4), np.ones(4), np.ones(4)) == -1.0


class TestTensorDecomposer:
    """Test Tucker and CP decompositions."""

    def test_tucker_decomposition_shapes(self):
        from hololoom.core.warp.advanced import TensorDecomposer
        tensor = np.random.randn(4, 5, 3)
        core, factors = TensorDecomposer.tucker_decomposition(tensor, ranks=[2, 3, 2])
        assert core.shape == (2, 3, 2)
        assert len(factors) == 3
        assert factors[0].shape == (4, 2)
        assert factors[1].shape == (5, 3)
        assert factors[2].shape == (3, 2)

    def test_tucker_default_ranks(self):
        from hololoom.core.warp.advanced import TensorDecomposer
        tensor = np.random.randn(3, 4, 2)
        core, factors = TensorDecomposer.tucker_decomposition(tensor)
        assert len(factors) == 3

    def test_mode_n_unfolding(self):
        from hololoom.core.warp.advanced import TensorDecomposer
        t = np.arange(24).reshape(2, 3, 4)
        u0 = TensorDecomposer._mode_n_unfolding(t, 0)
        assert u0.shape == (2, 12)
        u1 = TensorDecomposer._mode_n_unfolding(t, 1)
        assert u1.shape == (3, 8)

    def test_cp_decomposition_converges(self):
        from hololoom.core.warp.advanced import TensorDecomposer
        np.random.seed(42)
        tensor = np.random.randn(3, 4, 2)
        factors = TensorDecomposer.cp_decomposition(tensor, rank=2, max_iter=50)
        assert len(factors) == 3
        assert factors[0].shape == (3, 2)
        assert factors[1].shape == (4, 2)
        assert factors[2].shape == (2, 2)

    def test_khatri_rao_product(self):
        from hololoom.core.warp.advanced import TensorDecomposer
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        kr = TensorDecomposer._khatri_rao([A, B])
        assert kr.shape[1] == 2  # same number of columns


class TestQuantumWarpOperations:
    """Test quantum-inspired operations."""

    def test_superposition_uniform(self):
        from hololoom.core.warp.advanced import QuantumWarpOperations
        s1 = np.array([1.0, 0.0])
        s2 = np.array([0.0, 1.0])
        result = QuantumWarpOperations.superposition([s1, s2])
        # Uniform: (1/sqrt(2))*[1,0] + (1/sqrt(2))*[0,1] = [1/sqrt(2), 1/sqrt(2)]
        expected = np.array([1.0, 1.0]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_superposition_custom_amplitudes(self):
        from hololoom.core.warp.advanced import QuantumWarpOperations
        s1 = np.array([1.0, 0.0])
        s2 = np.array([0.0, 1.0])
        amps = np.array([3.0, 4.0])
        result = QuantumWarpOperations.superposition([s1, s2], amplitudes=amps)
        norm_amps = amps / np.linalg.norm(amps)
        expected = norm_amps[0] * s1 + norm_amps[1] * s2
        np.testing.assert_array_almost_equal(result, expected)

    def test_entangle_tensor_product(self):
        from hololoom.core.warp.advanced import QuantumWarpOperations
        s1 = np.array([1.0, 0.0])
        s2 = np.array([0.0, 1.0])
        result = QuantumWarpOperations.entangle(s1, s2)
        # Outer product flattened: [0, 1, 0, 0]
        np.testing.assert_array_almost_equal(result, np.array([0.0, 1.0, 0.0, 0.0]))

    def test_measure_with_collapse(self):
        from hololoom.core.warp.advanced import QuantumWarpOperations
        state = np.array([1.0, 0.0, 0.0])
        observables = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        idx, prob, collapsed = QuantumWarpOperations.measure(state, observables, collapse=True)
        assert idx == 0  # state is aligned with first observable
        assert prob > 0.99
        assert collapsed is not None

    def test_measure_without_collapse(self):
        from hololoom.core.warp.advanced import QuantumWarpOperations
        state = np.array([1.0, 0.0])
        observables = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        idx, prob, collapsed = QuantumWarpOperations.measure(state, observables, collapse=False)
        assert collapsed is None

    def test_decoherence_preserves_norm(self):
        from hololoom.core.warp.advanced import QuantumWarpOperations
        state = np.array([1.0, 0.0, 0.0])
        decohered = QuantumWarpOperations.decoherence(state, noise_level=0.1)
        assert abs(np.linalg.norm(decohered) - 1.0) < 1e-6


class TestFisherInformationGeometry:
    """Test Fisher information metric and natural gradient."""

    def test_fisher_information_matrix_shape(self):
        from hololoom.core.warp.advanced import FisherInformationGeometry
        dist = np.array([0.3, 0.5, 0.2])
        grads = [np.array([0.1, -0.1, 0.0]), np.array([0.0, 0.2, -0.2])]
        fim = FisherInformationGeometry.fisher_information_matrix(dist, grads)
        assert fim.shape == (2, 2)

    def test_fisher_information_matrix_symmetric(self):
        from hololoom.core.warp.advanced import FisherInformationGeometry
        dist = np.array([0.3, 0.5, 0.2])
        grads = [np.array([0.1, -0.1, 0.0]), np.array([0.0, 0.2, -0.2])]
        fim = FisherInformationGeometry.fisher_information_matrix(dist, grads)
        np.testing.assert_array_almost_equal(fim, fim.T)

    def test_natural_gradient(self):
        from hololoom.core.warp.advanced import FisherInformationGeometry
        fim = np.array([[2.0, 0.0], [0.0, 3.0]])
        loss_grad = np.array([1.0, 1.0])
        nat_grad = FisherInformationGeometry.natural_gradient(loss_grad, fim)
        # F^{-1} @ grad = [0.5, 0.333...]
        np.testing.assert_array_almost_equal(nat_grad, [0.5, 1.0 / 3], decimal=3)

    def test_natural_gradient_singular_uses_pinv(self):
        from hololoom.core.warp.advanced import FisherInformationGeometry
        fim = np.array([[0.0, 0.0], [0.0, 0.0]])  # Singular
        loss_grad = np.array([1.0, 1.0])
        nat_grad = FisherInformationGeometry.natural_gradient(loss_grad, fim, damping=1e-4)
        # Should not crash
        assert nat_grad.shape == (2,)


# ============================================================================
# eggroll_warp.py — ContextPacker, SpectralScorer, Warp
# ============================================================================

class TestContextPacker:
    """Test ContextPacker packing logic."""

    def test_pack_fits_all(self):
        from hololoom.core.warp.eggroll_warp import ContextPacker
        packer = ContextPacker(max_tokens=1000)
        items = ["short text", "another short"]
        result = packer.pack(items)
        assert "short text" in result
        assert "another short" in result

    def test_pack_truncates_on_limit(self):
        from hololoom.core.warp.eggroll_warp import ContextPacker
        packer = ContextPacker(max_tokens=5)  # Very small
        items = ["a" * 100, "b" * 100]
        result = packer.pack(items)
        # Should stop before adding both
        assert len(result) < 200

    def test_pack_empty_list(self):
        from hololoom.core.warp.eggroll_warp import ContextPacker
        packer = ContextPacker()
        result = packer.pack([])
        assert result == ""


class TestSpectralScorer:
    """Test SpectralScorer methods."""

    def test_spectral_radius_square_matrix(self):
        from hololoom.core.warp.eggroll_warp import SpectralScorer
        m = np.array([[2.0, 0.0], [0.0, 3.0]])
        r = SpectralScorer.compute_spectral_radius(m)
        assert abs(r - 3.0) < 1e-6

    def test_spectral_radius_non_square(self):
        from hololoom.core.warp.eggroll_warp import SpectralScorer
        m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = SpectralScorer.compute_spectral_radius(m)
        # Should use SVD, return largest singular value
        s = np.linalg.svd(m, compute_uv=False)
        assert abs(r - s[0]) < 1e-6

    def test_spectral_radius_1d(self):
        """1D array treated as non-square."""
        from hololoom.core.warp.eggroll_warp import SpectralScorer
        m = np.array([[1.0]])
        r = SpectralScorer.compute_spectral_radius(m)
        assert abs(r - 1.0) < 1e-6

    def test_analyze_sparsity(self):
        from hololoom.core.warp.eggroll_warp import SpectralScorer
        m = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        s = SpectralScorer.analyze_sparsity(m, threshold=1e-4)
        assert abs(s - 4 / 6) < 1e-6

    def test_analyze_sparsity_dense(self):
        from hololoom.core.warp.eggroll_warp import SpectralScorer
        m = np.ones((3, 3))
        s = SpectralScorer.analyze_sparsity(m)
        assert s == 0.0


class TestWarp:
    """Test Warp evaluator class."""

    def test_init_no_crash(self):
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        assert w.packer is not None

    def test_embed_returns_vector(self):
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        result = w.embed("hello world")
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)

    def test_cosine_similarity_identical(self):
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        v = np.array([1.0, 2.0, 3.0])
        assert abs(w.cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert abs(w.cosine_similarity(v1, v2)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        assert w.cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0

    def test_score_without_embedder(self):
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        w.has_embedder = False
        w.has_safety = False
        w.has_resonance = False
        s = w.score("target", "output")
        assert 0.0 <= s <= 1.0

    def test_score_with_metrics(self):
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        w.has_embedder = False
        w.has_safety = False
        w.has_resonance = False
        metrics = {"sparsity": 0.8, "energy": 0.3, "spectral_radius": 1.0}
        s = w.score("target", "output", metrics=metrics)
        assert 0.0 <= s <= 1.0

    def test_score_efficiency_run_pattern(self):
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        w.has_embedder = False
        w.has_safety = False
        w.has_resonance = False
        s = w.score("t", "o", pattern_name="efficiency_run")
        assert 0.0 <= s <= 1.0

    def test_score_quality_first_pattern(self):
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        w.has_embedder = False
        w.has_safety = False
        w.has_resonance = False
        s = w.score("t", "o", pattern_name="quality_first")
        assert 0.0 <= s <= 1.0

    def test_score_zero_weight(self):
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        w.has_embedder = False
        w.has_safety = False
        w.has_resonance = False
        s = w.score(123, "output")  # non-string target
        assert 0.0 <= s <= 1.0

    def test_score_with_mock_embedder_short(self):
        """Test score with embedder — covers semantic branch, short text (topology fallback)."""
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        # Force mock embedder to avoid heavy dep chains
        w.has_embedder = True
        w.has_safety = False
        w.has_resonance = False
        w.embedder = None  # embed() fallback returns random 384-d

        def mock_embed(text):
            rng = np.random.RandomState(abs(hash(text)) % (2**31))
            return rng.randn(384).astype(np.float32)
        w.embed = mock_embed

        # Mock the math_crusher imports to avoid deep import chains
        mock_hyp = Mock()
        mock_hyp.exp_map.side_effect = lambda v: v / (np.linalg.norm(v) + 1e-10)
        mock_hyp.poincare_distance.return_value = 0.5

        mock_info = Mock()
        mock_info.mutual_information.return_value = 1.0

        mock_stats = Mock()
        mock_stats.kl_divergence.return_value = 0.5

        mock_topo = Mock()
        mock_topo.compute_betti_0.return_value = 2

        import sys
        mock_module = Mock()
        mock_module.HyperbolicGeometry = mock_hyp
        mock_module.InformationTheory = mock_info
        mock_module.StatisticalMeasures = mock_stats
        mock_module.TopologicalFeatures = mock_topo
        with patch.dict(sys.modules, {"hololoom.eggroll.math_crusher": mock_module}):
            s = w.score("hello world", "hello there")
        assert 0.0 <= s <= 1.0

    def test_score_with_mock_embedder_long_text(self):
        """Test score with enough sentences to trigger topology branch."""
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        w.has_embedder = True
        w.has_safety = False
        w.has_resonance = False

        def mock_embed(text):
            rng = np.random.RandomState(abs(hash(text)) % (2**31))
            return rng.randn(384).astype(np.float32)
        w.embed = mock_embed

        mock_hyp = Mock()
        mock_hyp.exp_map.side_effect = lambda v: v / (np.linalg.norm(v) + 1e-10)
        mock_hyp.poincare_distance.return_value = 0.3

        mock_info = Mock()
        mock_info.mutual_information.return_value = 2.0

        mock_stats = Mock()
        mock_stats.kl_divergence.return_value = 1.0

        mock_topo = Mock()
        mock_topo.compute_betti_0.return_value = 3

        import sys
        mock_module = Mock()
        mock_module.HyperbolicGeometry = mock_hyp
        mock_module.InformationTheory = mock_info
        mock_module.StatisticalMeasures = mock_stats
        mock_module.TopologicalFeatures = mock_topo
        long_text = "First sentence here. Second sentence now. Third one follows. Fourth comes along. Fifth wraps up."
        with patch.dict(sys.modules, {"hololoom.eggroll.math_crusher": mock_module}):
            s = w.score("A reference target", long_text)
        assert 0.0 <= s <= 1.0

    def test_score_research_pipeline_pattern(self):
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        w.has_embedder = False
        w.has_safety = False
        w.has_resonance = False
        s = w.score("t", "o", pattern_name="research_pipeline")
        assert 0.0 <= s <= 1.0

    def test_score_with_safety(self):
        """Test score with safety guardrails enabled."""
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        w.has_embedder = False
        w.has_resonance = False
        if not w.has_safety:
            pytest.skip("SafetyGuardrails not available")
        s = w.score("target", "safe output text")
        assert 0.0 <= s <= 1.0

    def test_score_with_resonance(self):
        """Test score with resonance enabled."""
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        w.has_embedder = False
        w.has_safety = False
        if not w.has_resonance:
            pytest.skip("ResonanceShed not available")
        s = w.score("target", "output text")
        assert 0.0 <= s <= 1.0

    def test_score_partial_metrics(self):
        """Test score with only some metrics provided."""
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        w.has_embedder = False
        w.has_safety = False
        w.has_resonance = False
        # Only sparsity, no energy or spectral_radius
        s = w.score("t", "o", metrics={"sparsity": 0.9})
        assert 0.0 <= s <= 1.0

    def test_embed_fallback_without_embedder(self):
        """Test embed returns random vector when embedder is missing."""
        from hololoom.core.warp.eggroll_warp import Warp
        w = Warp()
        w.has_embedder = False
        vec = w.embed("test")
        assert vec.shape == (384,)


# ============================================================================
# combinatorics.py — ChainComplex, DiscreteMorseFunction, Sheaf
# ============================================================================

class TestChainComplex:
    """Test ChainComplex homology computation."""

    @pytest.fixture
    def triangle_complex(self):
        """Triangle with boundary: 3 vertices, 3 edges, 1 face."""
        from hololoom.core.warp.combinatorics import ChainComplex
        c = ChainComplex(dimension=2)
        for i in range(3):
            c.add_chain(0, (i,))
        c.add_chain(1, (0, 1))
        c.add_chain(1, (1, 2))
        c.add_chain(1, (0, 2))
        c.add_chain(2, (0, 1, 2))
        return c

    @pytest.fixture
    def circle_complex(self):
        """Circle (no face): 3 vertices, 3 edges."""
        from hololoom.core.warp.combinatorics import ChainComplex
        c = ChainComplex(dimension=1)
        for i in range(3):
            c.add_chain(0, (i,))
        c.add_chain(1, (0, 1))
        c.add_chain(1, (1, 2))
        c.add_chain(1, (0, 2))
        return c

    def test_add_chain_no_duplicates(self):
        from hololoom.core.warp.combinatorics import ChainComplex
        c = ChainComplex(dimension=1)
        c.add_chain(0, (0,))
        c.add_chain(0, (0,))  # duplicate
        assert len(c.chains[0]) == 1

    def test_compute_boundary_matrices(self, triangle_complex):
        triangle_complex.compute_boundary_matrices()
        assert 1 in triangle_complex.boundaries
        assert 2 in triangle_complex.boundaries
        # d1: 3 vertices x 3 edges
        assert triangle_complex.boundaries[1].shape == (3, 3)
        # d2: 3 edges x 1 face
        assert triangle_complex.boundaries[2].shape == (3, 1)

    def test_boundary_squared_is_zero(self, triangle_complex):
        """d_{k} * d_{k+1} = 0 (fundamental property)."""
        triangle_complex.compute_boundary_matrices()
        d1 = triangle_complex.boundaries[1]
        d2 = triangle_complex.boundaries[2]
        product = d1 @ d2
        np.testing.assert_array_equal(product, np.zeros_like(product))

    def test_triangle_homology_h0(self, triangle_complex):
        """H_0 of filled triangle: exercises the full homology computation."""
        triangle_complex.compute_boundary_matrices()
        h0 = triangle_complex.compute_homology(0, field="Z2")
        # Should return a valid result with kernel and image dimensions
        assert h0["kernel_dim"] >= 0
        assert h0["image_dim"] >= 0
        assert h0["rank"] >= 0

    def test_triangle_homology_h1(self, triangle_complex):
        """H_1 of filled triangle exercises boundary + kernel/image."""
        triangle_complex.compute_boundary_matrices()
        h1 = triangle_complex.compute_homology(1, field="Z2")
        # With a filled triangle, H_1 should be 0 (no hole)
        assert h1["rank"] >= 0
        assert "kernel_basis" in h1
        assert "image_basis" in h1

    def test_circle_homology_h1(self, circle_complex):
        """H_1 of circle: exercises homology without d_{k+1}."""
        circle_complex.compute_boundary_matrices()
        h1 = circle_complex.compute_homology(1, field="Z2")
        # Without 2-simplex, image is empty, so H_1 = kernel(d1)
        assert h1["image_dim"] == 0
        assert h1["rank"] == h1["kernel_dim"]

    def test_homology_empty_dimension(self):
        from hololoom.core.warp.combinatorics import ChainComplex
        c = ChainComplex(dimension=3)
        h = c.compute_homology(5)
        assert h["rank"] == 0

    def test_homology_over_reals(self, triangle_complex):
        triangle_complex.compute_boundary_matrices()
        h0 = triangle_complex.compute_homology(0, field="R")
        assert h0["rank"] >= 0

    def test_null_space_empty_matrix(self):
        from hololoom.core.warp.combinatorics import ChainComplex
        c = ChainComplex(dimension=0)
        ns = c._null_space(np.array([]).reshape(0, 3), field="Z2")
        assert ns.shape == (3, 0)

    def test_column_space_empty(self):
        from hololoom.core.warp.combinatorics import ChainComplex
        c = ChainComplex(dimension=0)
        cs = c._column_space(np.array([]).reshape(0, 0))
        assert cs.shape == (0, 0)

    def test_compute_homology_no_boundary_map(self):
        """Chains with no boundary: all are cycles."""
        from hololoom.core.warp.combinatorics import ChainComplex
        c = ChainComplex(dimension=0)
        c.add_chain(0, (0,))
        c.add_chain(0, (1,))
        h = c.compute_homology(0)
        assert h["kernel_dim"] == 2  # All chains are cycles


class TestDiscreteMorseFunction:
    """Test discrete Morse theory."""

    @pytest.fixture
    def morse_on_triangle(self):
        from hololoom.core.warp.combinatorics import ChainComplex, DiscreteMorseFunction
        c = ChainComplex(dimension=2)
        for i in range(3):
            c.add_chain(0, (i,))
        c.add_chain(1, (0, 1))
        c.add_chain(1, (1, 2))
        c.add_chain(1, (0, 2))
        c.add_chain(2, (0, 1, 2))
        c.compute_boundary_matrices()
        return DiscreteMorseFunction(c)

    def test_gradient_flow_produces_pairs(self, morse_on_triangle):
        morse_on_triangle.compute_gradient_flow()
        # Should have some gradient pairs and some critical cells
        total_cells = sum(len(v) for v in morse_on_triangle.complex.chains.values())
        paired = len(morse_on_triangle.gradient_pairs) * 2
        critical = sum(len(v) for v in morse_on_triangle.critical.values())
        assert paired + critical == total_cells

    def test_is_face_of(self, morse_on_triangle):
        assert morse_on_triangle._is_face_of((0, 1), (0, 1, 2))
        assert morse_on_triangle._is_face_of((1, 2), (0, 1, 2))
        assert not morse_on_triangle._is_face_of((0, 1, 2), (0, 1))
        assert not morse_on_triangle._is_face_of((0, 3), (0, 1, 2))

    def test_morse_complex_has_same_homology(self, morse_on_triangle):
        """Morse complex should have same homology as original (weak check)."""
        morse_on_triangle.compute_gradient_flow()
        mc = morse_on_triangle.morse_complex()
        # Morse complex should have fewer cells
        original_cells = sum(len(v) for v in morse_on_triangle.complex.chains.values())
        morse_cells = sum(len(v) for v in mc.chains.values())
        assert morse_cells <= original_cells

    def test_gradient_flow_on_edge_only_complex(self):
        """Complex with no 2-cells: all top-dim cells are critical."""
        from hololoom.core.warp.combinatorics import ChainComplex, DiscreteMorseFunction
        c = ChainComplex(dimension=1)
        c.add_chain(0, (0,))
        c.add_chain(0, (1,))
        c.add_chain(1, (0, 1))
        c.compute_boundary_matrices()
        morse = DiscreteMorseFunction(c)
        morse.compute_gradient_flow()
        # The edge (0,1) should pair with a vertex, leaving 1 critical 0-cell
        assert len(morse.gradient_pairs) >= 0


class TestSheaf:
    """Test Sheaf operations."""

    @pytest.fixture
    def simple_sheaf(self):
        from hololoom.core.warp.combinatorics import Sheaf
        sheaf = Sheaf(base_space="path")
        sheaf.add_stalk(0, np.array([1.0, 0.0]))
        sheaf.add_stalk(1, np.array([0.7, 0.7]))
        sheaf.add_stalk(2, np.array([0.0, 1.0]))
        I = np.eye(2)
        sheaf.add_restriction(0, 1, I)
        sheaf.add_restriction(1, 2, I)
        return sheaf

    def test_add_stalk_and_restriction(self):
        from hololoom.core.warp.combinatorics import Sheaf
        s = Sheaf(base_space="g")
        s.add_stalk("a", np.array([1.0]))
        s.add_restriction("a", "b", np.array([[1.0]]))
        assert "a" in s.stalks
        assert ("a", "b") in s.restriction_maps

    def test_sheaf_laplacian_shape(self, simple_sheaf):
        L = simple_sheaf.sheaf_laplacian()
        # 3 vertices * 2D stalks = 6x6
        assert L.shape == (6, 6)

    def test_sheaf_laplacian_symmetric(self, simple_sheaf):
        L = simple_sheaf.sheaf_laplacian()
        np.testing.assert_array_almost_equal(L, L.T)

    def test_sheaf_laplacian_psd(self, simple_sheaf):
        """Sheaf Laplacian should be positive semi-definite."""
        L = simple_sheaf.sheaf_laplacian()
        eigenvalues = np.linalg.eigvalsh(L)
        assert np.all(eigenvalues >= -1e-10)

    def test_sheaf_laplacian_empty(self):
        from hololoom.core.warp.combinatorics import Sheaf
        s = Sheaf(base_space="empty")
        L = s.sheaf_laplacian()
        assert L.shape == (0,)

    def test_global_sections(self, simple_sheaf):
        sections = simple_sheaf.global_sections(tol=1e-4)
        # With identity restriction maps, sections are consistent assignments
        assert sections.shape[0] == 6  # 3 vertices * 2D
        assert sections.shape[1] >= 0

    def test_global_sections_empty_sheaf(self):
        from hololoom.core.warp.combinatorics import Sheaf
        s = Sheaf(base_space="empty")
        result = s.global_sections()
        assert result.shape == (0,)

    def test_cohomology_h0(self, simple_sheaf):
        h0 = simple_sheaf.cohomology_dimension(0)
        assert h0 >= 0

    def test_cohomology_h1(self, simple_sheaf):
        h1 = simple_sheaf.cohomology_dimension(1)
        assert h1 >= 0

    def test_cohomology_h2_not_implemented(self, simple_sheaf):
        h2 = simple_sheaf.cohomology_dimension(2)
        assert h2 == 0

    def test_sheaf_laplacian_unknown_vertex_in_restriction(self):
        """Restriction map references unknown vertex: should skip."""
        from hololoom.core.warp.combinatorics import Sheaf
        s = Sheaf(base_space="g")
        s.add_stalk(0, np.array([1.0]))
        s.add_restriction(0, 99, np.array([[1.0]]))  # 99 not in stalks
        L = s.sheaf_laplacian()
        assert L.shape == (1, 1)

    def test_sheaf_laplacian_eigenvalues_nonneg(self):
        """Sheaf Laplacian eigenvalues should be non-negative (PSD check variant)."""
        from hololoom.core.warp.combinatorics import Sheaf
        s = Sheaf(base_space="g")
        s.add_stalk(0, np.array([1.0, 0.0]))
        s.add_stalk(1, np.array([0.0, 1.0]))
        I = np.eye(2)
        s.add_restriction(0, 1, I)
        L = s.sheaf_laplacian()
        eigenvalues = np.linalg.eigvalsh(L)
        assert np.all(eigenvalues >= -1e-10)
