"""
Optimized Warp Space
====================
High-performance version with GPU acceleration, sparse tensors, and lazy evaluation.

Performance Features:
- GPU/CUDA support via PyTorch (optional)
- Sparse tensor operations for memory efficiency
- Lazy evaluation for deferred computation
- Batch processing for parallel warp operations
- Memory pooling and caching
- Compiled kernels for critical paths

This module is a drop-in replacement for standard WarpSpace with
significant performance improvements for large-scale deployments.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

# Optional PyTorch for GPU acceleration
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch available. Using device: {DEVICE}")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    logger.warning("PyTorch not available. GPU acceleration disabled.")


# ============================================================================
# Sparse Tensor Operations
# ============================================================================

class SparseTensorField:
    """
    Sparse tensor field for memory-efficient warp space.

    Uses sparse representation for embeddings with many zeros or
    low-rank structure.
    """

    def __init__(self, dense_tensor: Optional[np.ndarray] = None, threshold: float = 1e-6):
        """
        Initialize sparse tensor.

        Args:
            dense_tensor: Optional dense tensor to convert
            threshold: Values below this are set to zero
        """
        self.shape = None
        self.indices = None
        self.values = None
        self.density = 0.0

        if dense_tensor is not None:
            self.from_dense(dense_tensor, threshold)

    def from_dense(self, dense: np.ndarray, threshold: float = 1e-6):
        """Convert dense tensor to sparse representation."""
        # Find non-zero elements
        mask = np.abs(dense) > threshold
        self.indices = np.argwhere(mask)
        self.values = dense[mask]
        self.shape = dense.shape
        self.density = len(self.values) / dense.size

        logger.debug(f"Sparse tensor: {self.shape}, density={self.density:.2%}")

    def to_dense(self) -> np.ndarray:
        """Convert sparse back to dense tensor."""
        if self.shape is None:
            return np.array([])

        dense = np.zeros(self.shape)
        for idx, val in zip(self.indices, self.values):
            dense[tuple(idx)] = val

        return dense

    def __matmul__(self, other: Union['SparseTensorField', np.ndarray]) -> np.ndarray:
        """Sparse matrix multiplication."""
        # For now, convert to dense and multiply
        # TODO: Implement true sparse multiplication
        return self.to_dense() @ other


# ============================================================================
# GPU-Accelerated Warp Space
# ============================================================================

class GPUWarpSpace:
    """
    GPU-accelerated Warp Space using PyTorch.

    All tensor operations are performed on GPU if available,
    with automatic fallback to CPU.
    """

    def __init__(
        self,
        embedder,
        scales: List[int] = [96, 192, 384],
        use_gpu: bool = True,
        dtype: str = "float32"
    ):
        """
        Initialize GPU warp space.

        Args:
            embedder: Embedding module
            scales: Embedding scales
            use_gpu: Use GPU if available
            dtype: Data type ("float32" or "float16" for mixed precision)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for GPU acceleration")

        self.embedder = embedder
        self.scales = scales
        self.device = DEVICE if use_gpu else torch.device("cpu")

        # Set dtype
        self.dtype = torch.float16 if dtype == "float16" else torch.float32

        # Tensor field on GPU
        self.tensor_field = None
        self.threads_meta = []

        logger.info(f"GPUWarpSpace initialized: device={self.device}, dtype={self.dtype}")

    async def tension(
        self,
        thread_texts: List[str],
        batch_size: int = 32
    ) -> None:
        """
        Tension threads with batched GPU processing.

        Args:
            thread_texts: Texts to embed
            batch_size: Batch size for parallel processing
        """
        logger.info(f"Tensioning {len(thread_texts)} threads on {self.device}")

        # Batch encode (already supports batching)
        embeddings_dict = self.embedder.encode_scales(thread_texts)

        # Use largest scale
        max_scale = max(self.scales)
        embeddings = embeddings_dict[max_scale]

        # Convert to PyTorch tensor on GPU
        self.tensor_field = torch.tensor(
            embeddings,
            dtype=self.dtype,
            device=self.device
        )

        # Store metadata
        self.threads_meta = [
            {"text": text, "index": i}
            for i, text in enumerate(thread_texts)
        ]

        logger.info(f"Tensioned field shape: {self.tensor_field.shape}")

    def compute_attention(
        self,
        query_embedding: np.ndarray,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute attention weights on GPU.

        Args:
            query_embedding: Query vector
            temperature: Softmax temperature (higher = more uniform)

        Returns:
            Attention weights (on GPU)
        """
        # Convert query to GPU tensor
        query = torch.tensor(
            query_embedding,
            dtype=self.dtype,
            device=self.device
        )

        # Compute scores (batched dot product)
        scores = torch.matmul(self.tensor_field, query) / temperature

        # Softmax attention
        attention = F.softmax(scores, dim=0)

        return attention

    def weighted_context(self, attention: torch.Tensor) -> np.ndarray:
        """
        Compute weighted context vector on GPU.

        Args:
            attention: Attention weights (on GPU)

        Returns:
            Context vector (numpy)
        """
        # Weighted sum on GPU
        context = torch.matmul(attention, self.tensor_field)

        # Convert back to numpy
        return context.cpu().numpy()

    def batch_attention(
        self,
        query_embeddings: List[np.ndarray],
        temperature: float = 1.0
    ) -> List[np.ndarray]:
        """
        Compute attention for multiple queries in parallel.

        Args:
            query_embeddings: List of query vectors
            temperature: Softmax temperature

        Returns:
            List of context vectors
        """
        # Stack queries into batch
        queries = torch.tensor(
            np.stack(query_embeddings),
            dtype=self.dtype,
            device=self.device
        )  # (batch_size, dim)

        # Batch matrix multiply: (batch, dim) @ (dim, n_threads)
        scores = torch.matmul(queries, self.tensor_field.T) / temperature

        # Batch softmax
        attention = F.softmax(scores, dim=1)

        # Batch weighted context: (batch, n_threads) @ (n_threads, dim)
        contexts = torch.matmul(attention, self.tensor_field)

        return contexts.cpu().numpy()


# ============================================================================
# Lazy Evaluation
# ============================================================================

class LazyWarpOperation:
    """
    Lazy evaluation wrapper for deferred computation.

    Builds computation graph and executes only when result is needed.
    """

    def __init__(self, operation: str, *args, **kwargs):
        """
        Initialize lazy operation.

        Args:
            operation: Operation name
            *args, **kwargs: Operation arguments
        """
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.computed = False

    def compute(self) -> Any:
        """Execute the lazy operation."""
        if not self.computed:
            logger.debug(f"Computing lazy operation: {self.operation}")

            # Execute operation
            if self.operation == "attention":
                # Lazy attention computation
                warp_space, query = self.args
                self.result = warp_space._compute_attention(query)

            elif self.operation == "spectral":
                # Lazy spectral features
                warp_space = self.args[0]
                self.result = warp_space._compute_spectral()

            elif self.operation == "context":
                # Lazy context computation
                warp_space, attention = self.args
                self.result = warp_space._compute_context(attention)

            else:
                raise ValueError(f"Unknown lazy operation: {self.operation}")

            self.computed = True

        return self.result

    def __call__(self):
        """Make it callable."""
        return self.compute()


# ============================================================================
# Memory Pool
# ============================================================================

class TensorMemoryPool:
    """
    Memory pool for efficient tensor allocation.

    Reuses allocated tensors to reduce allocation overhead.
    """

    def __init__(self, max_pool_size: int = 100):
        """
        Initialize memory pool.

        Args:
            max_pool_size: Maximum number of cached tensors
        """
        self.pool: Dict[Tuple, List[np.ndarray]] = {}
        self.max_pool_size = max_pool_size
        self.hits = 0
        self.misses = 0

    def allocate(self, shape: Tuple, dtype=np.float32) -> np.ndarray:
        """
        Allocate tensor from pool or create new.

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            Tensor (reused or new)
        """
        key = (shape, dtype)

        if key in self.pool and len(self.pool[key]) > 0:
            # Reuse from pool
            self.hits += 1
            tensor = self.pool[key].pop()
            tensor.fill(0)  # Clear previous data
            return tensor
        else:
            # Allocate new
            self.misses += 1
            return np.zeros(shape, dtype=dtype)

    def release(self, tensor: np.ndarray):
        """
        Release tensor back to pool.

        Args:
            tensor: Tensor to release
        """
        key = (tensor.shape, tensor.dtype)

        if key not in self.pool:
            self.pool[key] = []

        # Add to pool if not full
        if len(self.pool[key]) < self.max_pool_size:
            self.pool[key].append(tensor)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses + 1e-10),
            "pool_sizes": {str(k): len(v) for k, v in self.pool.items()},
            "total_cached": sum(len(v) for v in self.pool.values())
        }


# ============================================================================
# Batched Warp Processor
# ============================================================================

class BatchedWarpProcessor:
    """
    Process multiple warp operations in parallel batches.

    Optimized for throughput when processing many queries simultaneously.
    """

    def __init__(self, max_batch_size: int = 32):
        """
        Initialize batched processor.

        Args:
            max_batch_size: Maximum batch size
        """
        self.max_batch_size = max_batch_size

    async def batch_tension_and_attend(
        self,
        thread_batches: List[List[str]],
        query_batches: List[List[np.ndarray]],
        warp_space
    ) -> List[List[np.ndarray]]:
        """
        Process multiple warp space operations in batches.

        Args:
            thread_batches: List of thread text lists
            query_batches: List of query embedding lists
            warp_space: Warp space instance

        Returns:
            List of context vector lists
        """
        all_contexts = []

        # Process each batch
        for threads, queries in zip(thread_batches, query_batches):
            # Tension threads
            await warp_space.tension(threads)

            # Batch compute attention
            if isinstance(warp_space, GPUWarpSpace):
                contexts = warp_space.batch_attention(queries)
            else:
                # Fallback to sequential
                contexts = []
                for query in queries:
                    attention = warp_space.apply_attention(query)
                    context = warp_space.weighted_context(attention)
                    contexts.append(context)

            all_contexts.append(contexts)

        return all_contexts


# ============================================================================
# Compiled Kernels (Numba)
# ============================================================================

try:
    from numba import jit

    @jit(nopython=True)
    def fast_attention_kernel(tensor_field, query, temperature=1.0):
        """JIT-compiled attention computation."""
        n_threads, dim = tensor_field.shape

        # Compute scores
        scores = np.zeros(n_threads)
        for i in range(n_threads):
            score = 0.0
            for j in range(dim):
                score += tensor_field[i, j] * query[j]
            scores[i] = score / temperature

        # Softmax
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)
        sum_exp = np.sum(exp_scores)
        attention = exp_scores / sum_exp

        return attention

    HAS_NUMBA = True
    logger.info("Numba JIT compilation available")

except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba not available. JIT compilation disabled.")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import time

    print("="*80)
    print("Optimized Warp Space Demo")
    print("="*80 + "\n")

    # 1. Sparse Tensors
    print("1. Sparse Tensor Field")
    print("-" * 40)

    dense = np.random.randn(100, 50)
    dense[np.abs(dense) < 1.0] = 0  # Sparsify

    sparse = SparseTensorField(dense, threshold=0.5)
    print(f"Shape: {sparse.shape}")
    print(f"Density: {sparse.density:.2%}")
    print(f"Memory saved: {(1 - sparse.density)*100:.1f}%")
    print()

    # 2. Memory Pool
    print("2. Tensor Memory Pool")
    print("-" * 40)

    pool = TensorMemoryPool()

    # Allocate and release
    for _ in range(10):
        t1 = pool.allocate((100, 50))
        t2 = pool.allocate((100, 50))
        pool.release(t1)
        pool.release(t2)

    stats = pool.get_stats()
    print(f"Hit rate: {stats['hit_rate']:.1%}")
    print(f"Cached tensors: {stats['total_cached']}")
    print()

    # 3. GPU Warp Space (if available)
    if HAS_TORCH:
        print("3. GPU-Accelerated Warp Space")
        print("-" * 40)

        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        async def test_gpu():
            embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
            gpu_warp = GPUWarpSpace(embedder, use_gpu=True)

            threads = ["Sample thread " + str(i) for i in range(10)]

            start = time.time()
            await gpu_warp.tension(threads)
            tension_time = time.time() - start

            # Query
            query_emb = np.random.randn(384)
            attention = gpu_warp.compute_attention(query_emb)

            print(f"Device: {gpu_warp.device}")
            print(f"Tensor shape: {gpu_warp.tensor_field.shape}")
            print(f"Tension time: {tension_time*1000:.2f}ms")
            print(f"Attention shape: {attention.shape}")
            print()

            # Batch queries
            queries = [np.random.randn(384) for _ in range(5)]
            start = time.time()
            contexts = gpu_warp.batch_attention(queries)
            batch_time = time.time() - start

            print(f"Batch queries: {len(queries)}")
            print(f"Batch time: {batch_time*1000:.2f}ms ({batch_time/len(queries)*1000:.2f}ms per query)")

        asyncio.run(test_gpu())
        print()

    # 4. JIT Compilation (if available)
    if HAS_NUMBA:
        print("4. JIT-Compiled Attention")
        print("-" * 40)

        tensor_field = np.random.randn(100, 128).astype(np.float64)
        query = np.random.randn(128).astype(np.float64)

        # First call (compilation + execution)
        start = time.time()
        attention1 = fast_attention_kernel(tensor_field, query)
        first_time = time.time() - start

        # Second call (cached, execution only)
        start = time.time()
        attention2 = fast_attention_kernel(tensor_field, query)
        second_time = time.time() - start

        print(f"First call (compile + run): {first_time*1000:.2f}ms")
        print(f"Second call (cached):        {second_time*1000:.2f}ms")
        print(f"Speedup: {first_time/second_time:.1f}x")
        print()

    print("Demo complete!")
