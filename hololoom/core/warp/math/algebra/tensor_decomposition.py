"""
Tensor Decomposition & Tensor Networks
========================================

Scope slots: LA4.7 (Tucker decomposition), LA4.8 (Tensor Train),
LA4.9 (Tensor Network contraction).

Pure numpy implementations of multilinear algebra:
- Tucker decomposition via Higher-Order SVD (HOSVD)
- Tensor Train decomposition via TT-SVD
- Tensor network contraction via Einstein-style index summation

Author: Claude Code
Date: March 2026
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# TUCKER DECOMPOSITION — HOSVD (LA4.7)
# ============================================================================

@dataclass
class TuckerResult:
    """Result of Tucker decomposition.

    A tensor T ≈ core ×₁ factors[0] ×₂ factors[1] × ... ×_N factors[N-1]

    Attributes:
        core: Core tensor of shape (r₁, r₂, ..., r_N).
        factors: List of factor matrices, factors[k] has shape (n_k, r_k).
        relative_error: Frobenius-norm relative error of approximation.
    """
    core: np.ndarray
    factors: list[np.ndarray]
    relative_error: float


class TuckerDecomposition:
    """Tucker decomposition via Higher-Order SVD (HOSVD).

    Decomposes an N-dimensional tensor into a core tensor multiplied
    by a factor matrix along each mode. The HOSVD computes each factor
    by SVD of the mode-k unfolding.

    The Tucker decomposition generalizes the matrix SVD to tensors:
        T ≈ G ×₁ U₁ ×₂ U₂ × ... ×_N U_N

    where G is the (typically smaller) core tensor and U_k are
    orthogonal factor matrices.

    Example:
        >>> tensor = np.random.randn(10, 8, 6)
        >>> tucker = TuckerDecomposition()
        >>> result = tucker.decompose(tensor, ranks=(3, 3, 2))
        >>> result.core.shape
        (3, 3, 2)
    """

    @staticmethod
    def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
        """Mode-k unfolding (matricization) of a tensor.

        Rearranges a tensor into a matrix by mapping mode k to rows
        and all other modes to columns (in order).

        Args:
            tensor: N-dimensional array.
            mode: Mode to unfold along (0-indexed).

        Returns:
            2D array of shape (n_k, prod(n_j for j != k)).
        """
        return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

    @staticmethod
    def fold(matrix: np.ndarray, mode: int, shape: tuple[int, ...]) -> np.ndarray:
        """Refold a matrix into a tensor (inverse of unfold).

        Args:
            matrix: Mode-k unfolded matrix.
            mode: Original mode index.
            shape: Target tensor shape.

        Returns:
            N-dimensional tensor.
        """
        new_shape = [shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode]
        tensor = matrix.reshape(new_shape)
        return np.moveaxis(tensor, 0, mode)

    @staticmethod
    def mode_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
        """n-mode product: tensor ×_mode matrix.

        Multiplies a tensor by a matrix along the specified mode.

        Args:
            tensor: N-dimensional array.
            matrix: 2D array of shape (J, I_mode).
            mode: Mode along which to multiply.

        Returns:
            Tensor with mode dimension changed from I_mode to J.
        """
        unfolded = TuckerDecomposition.unfold(tensor, mode)
        result_unfolded = matrix @ unfolded
        new_shape = list(tensor.shape)
        new_shape[mode] = matrix.shape[0]
        return TuckerDecomposition.fold(result_unfolded, mode, tuple(new_shape))

    def decompose(self, tensor: np.ndarray, ranks: tuple[int, ...]) -> TuckerResult:
        """Compute Tucker decomposition via HOSVD.

        Algorithm:
        1. For each mode k, compute SVD of the mode-k unfolding
        2. Truncate to rank r_k, yielding factor U_k
        3. Core = tensor ×₁ U₁ᵀ ×₂ U₂ᵀ × ... ×_N U_Nᵀ

        Args:
            tensor: N-dimensional array of shape (n₁, n₂, ..., n_N).
            ranks: Tuple of target ranks (r₁, r₂, ..., r_N).

        Returns:
            TuckerResult with core tensor, factor matrices, and error.
        """
        ndim = tensor.ndim
        if len(ranks) != ndim:
            raise ValueError(f"ranks must have {ndim} entries, got {len(ranks)}")

        factors = []
        norm_orig = np.linalg.norm(tensor)

        for mode in range(ndim):
            unfolded = self.unfold(tensor, mode)
            U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
            r = min(ranks[mode], U.shape[1])
            factors.append(U[:, :r])

        # Compute core: project tensor onto factor subspaces
        core = tensor.copy()
        for mode in range(ndim):
            core = self.mode_product(core, factors[mode].T, mode)

        # Reconstruct for error estimate
        reconstructed = core.copy()
        for mode in range(ndim):
            reconstructed = self.mode_product(reconstructed, factors[mode], mode)

        error = np.linalg.norm(tensor - reconstructed)
        rel_error = error / norm_orig if norm_orig > 1e-15 else 0.0

        return TuckerResult(core=core, factors=factors, relative_error=rel_error)

    @staticmethod
    def reconstruct(result: TuckerResult) -> np.ndarray:
        """Reconstruct tensor from Tucker decomposition."""
        tensor = result.core.copy()
        for mode, factor in enumerate(result.factors):
            tensor = TuckerDecomposition.mode_product(tensor, factor, mode)
        return tensor


# ============================================================================
# TENSOR TRAIN DECOMPOSITION — TT-SVD (LA4.8)
# ============================================================================

@dataclass
class TensorTrainResult:
    """Result of Tensor Train decomposition.

    T(i₁, i₂, ..., i_N) = G₁(i₁) · G₂(i₂) · ... · G_N(i_N)

    Each G_k(i_k) is a matrix of shape (r_{k-1}, r_k), so the product
    yields a scalar. r₀ = r_N = 1.

    Attributes:
        cores: List of TT cores. cores[k] has shape (r_{k-1}, n_k, r_k).
        ranks: TT ranks (r₀, r₁, ..., r_N) with r₀ = r_N = 1.
        relative_error: Frobenius-norm relative error.
    """
    cores: list[np.ndarray]
    ranks: list[int]
    relative_error: float


class TensorTrain:
    """Tensor Train (TT) decomposition via TT-SVD.

    Decomposes an N-dimensional tensor into a train of 3D cores:
        T(i₁, ..., i_N) = G₁(:, i₁, :) · G₂(:, i₂, :) · ... · G_N(:, i_N, :)

    The TT format has O(Σ n_k r²) storage instead of O(Π n_k).
    TT-SVD is the standard algorithm using sequential SVDs from left to right.

    Example:
        >>> tensor = np.random.randn(4, 5, 6, 3)
        >>> tt = TensorTrain()
        >>> result = tt.decompose(tensor, max_rank=3)
        >>> len(result.cores)
        4
    """

    def decompose(
        self,
        tensor: np.ndarray,
        ranks: tuple[int, ...] | None = None,
        max_rank: int | None = None,
        tol: float = 1e-12,
    ) -> TensorTrainResult:
        """Compute TT decomposition via TT-SVD.

        Algorithm (Oseledets 2011):
        1. Reshape and unfold from the left
        2. SVD, truncate, store left singular vectors as core
        3. Pass remainder to next step

        Args:
            tensor: N-dimensional array.
            ranks: Explicit TT ranks (overrides max_rank and tol).
            max_rank: Maximum allowed TT rank per bond.
            tol: Relative truncation tolerance for SVD.

        Returns:
            TensorTrainResult with cores and ranks.
        """
        shape = tensor.shape
        ndim = tensor.ndim
        norm_orig = np.linalg.norm(tensor)

        if ndim < 2:
            raise ValueError("Tensor must have at least 2 dimensions")

        cores = []
        tt_ranks = [1]
        C = tensor.reshape(shape[0], -1)  # Current remainder

        delta = tol * norm_orig / np.sqrt(ndim - 1) if norm_orig > 0 else 0

        for k in range(ndim - 1):
            nk = shape[k]
            r_prev = tt_ranks[-1]

            C = C.reshape(r_prev * nk, -1)
            U, S, Vt = np.linalg.svd(C, full_matrices=False)

            # Determine rank truncation
            if ranks is not None:
                r = min(ranks[k], len(S))
            elif max_rank is not None:
                # Truncate by max_rank and tolerance
                r = min(max_rank, len(S))
                if delta > 0:
                    cumsum = np.cumsum(S[::-1] ** 2)[::-1]
                    for rr in range(1, len(S) + 1):
                        if rr >= len(cumsum) or np.sqrt(cumsum[rr]) <= delta:
                            r = min(r, rr)
                            break
            else:
                # Truncate by tolerance only
                r = len(S)
                if delta > 0:
                    cumsum = np.cumsum(S[::-1] ** 2)[::-1]
                    for rr in range(1, len(S) + 1):
                        if rr >= len(cumsum) or np.sqrt(cumsum[rr]) <= delta:
                            r = rr
                            break

            r = max(1, r)

            # Store core
            core = U[:, :r].reshape(r_prev, nk, r)
            cores.append(core)
            tt_ranks.append(r)

            # Remainder for next step
            C = np.diag(S[:r]) @ Vt[:r, :]

        # Last core
        cores.append(C.reshape(tt_ranks[-1], shape[-1], 1))
        tt_ranks.append(1)

        # Compute error
        reconstructed = self.reconstruct(cores, shape)
        error = np.linalg.norm(tensor - reconstructed)
        rel_error = error / norm_orig if norm_orig > 1e-15 else 0.0

        return TensorTrainResult(
            cores=cores, ranks=tt_ranks, relative_error=rel_error
        )

    @staticmethod
    def reconstruct(cores: list[np.ndarray], shape: tuple[int, ...] | None = None) -> np.ndarray:
        """Reconstruct full tensor from TT cores.

        Args:
            cores: List of TT cores.
            shape: Original tensor shape (inferred if None).

        Returns:
            Reconstructed N-dimensional tensor.
        """
        if shape is None:
            shape = tuple(c.shape[1] for c in cores)

        ndim = len(cores)
        n_total = int(np.prod(shape))

        # Evaluate all entries via matrix chain multiplication
        result = np.zeros(n_total)
        indices = np.array(np.unravel_index(np.arange(n_total), shape)).T

        for flat_idx, multi_idx in enumerate(indices):
            val = cores[0][:, multi_idx[0], :]
            for k in range(1, ndim):
                val = val @ cores[k][:, multi_idx[k], :]
            result[flat_idx] = val.item()

        return result.reshape(shape)

    @staticmethod
    def tt_dot(cores_a: list[np.ndarray], cores_b: list[np.ndarray]) -> float:
        """Compute inner product <A, B> in TT format without reconstruction.

        Uses the transfer matrix approach: O(N d r⁴) instead of O(dᴺ).

        Args:
            cores_a, cores_b: TT cores of two tensors.

        Returns:
            Inner product (Frobenius).
        """
        ndim = len(cores_a)
        # Transfer matrix: start with 1x1 identity
        transfer = np.array([[1.0]])

        for k in range(ndim):
            ra_prev, nk, ra_next = cores_a[k].shape
            rb_prev, _, rb_next = cores_b[k].shape

            new_transfer = np.zeros((ra_next, rb_next))
            for ik in range(nk):
                Ak = cores_a[k][:, ik, :]  # (ra_prev, ra_next)
                Bk = cores_b[k][:, ik, :]  # (rb_prev, rb_next)
                new_transfer += Ak.T @ transfer @ Bk

            transfer = new_transfer

        return float(transfer.item())


# ============================================================================
# TENSOR NETWORK CONTRACTION (LA4.9)
# ============================================================================

class TensorNetwork:
    """Tensor network contraction via Einstein-style index specification.

    A tensor network is a collection of tensors connected by shared indices.
    Contraction sums over all shared (internal) indices, producing an
    output tensor indexed by the remaining (external) indices.

    This is the generalization of matrix multiplication, trace, and
    Einstein summation to arbitrary networks of tensors.

    Example:
        >>> A = np.random.randn(3, 4)
        >>> B = np.random.randn(4, 5)
        >>> C = np.random.randn(5, 3)
        >>> tn = TensorNetwork()
        >>> # A_{ij} B_{jk} C_{ki} = scalar (trace of product)
        >>> result = tn.contract(
        ...     [A, B, C],
        ...     [('i', 'j'), ('j', 'k'), ('k', 'i')],
        ...     output_indices=()
        ... )
    """

    def __init__(self):
        self._tensors: list[tuple[np.ndarray, tuple[str, ...]]] = []

    def add_tensor(self, tensor: np.ndarray, indices: tuple[str, ...]) -> int:
        """Add a tensor with named indices to the network.

        Args:
            tensor: N-dimensional array.
            indices: Tuple of index names, one per dimension.

        Returns:
            Tensor index in the network.
        """
        if len(indices) != tensor.ndim:
            raise ValueError(f"Expected {tensor.ndim} index names, got {len(indices)}")
        self._tensors.append((tensor, indices))
        return len(self._tensors) - 1

    def contract(
        self,
        tensors: list[np.ndarray] | None = None,
        indices: list[tuple[str, ...]] | None = None,
        output_indices: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        """Contract the tensor network.

        Sums over all indices that appear exactly twice (contracted).
        Indices appearing once are output (free) indices.

        Args:
            tensors: List of tensors (uses stored tensors if None).
            indices: Index labels for each tensor.
            output_indices: Explicit output index ordering (auto-detected if None).

        Returns:
            Contracted result tensor.
        """
        if tensors is not None and indices is not None:
            tensor_list = list(zip(tensors, indices))
        else:
            tensor_list = list(self._tensors)

        if not tensor_list:
            raise ValueError("No tensors to contract")

        # Build einsum string
        subscript = self._build_einsum(tensor_list, output_indices)
        tensor_arrays = [t for t, _ in tensor_list]

        return np.einsum(subscript, *tensor_arrays)

    def _build_einsum(
        self,
        tensor_list: list[tuple[np.ndarray, tuple[str, ...]]],
        output_indices: tuple[str, ...] | None,
    ) -> str:
        """Build numpy einsum subscript string from named indices."""
        # Map named indices to single characters
        all_names = set()
        for _, idx in tensor_list:
            all_names.update(idx)

        name_to_char: dict[str, str] = {}
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i, name in enumerate(sorted(all_names)):
            if i >= len(chars):
                raise ValueError(f"Too many unique indices (max {len(chars)})")
            name_to_char[name] = chars[i]

        # Count occurrences to find free vs contracted
        count: dict[str, int] = {}
        for _, idx in tensor_list:
            for name in idx:
                count[name] = count.get(name, 0) + 1

        # Build input subscripts
        input_parts = []
        for _, idx in tensor_list:
            input_parts.append(''.join(name_to_char[n] for n in idx))

        input_str = ','.join(input_parts)

        # Build output subscript
        if output_indices is not None:
            output_str = ''.join(name_to_char[n] for n in output_indices)
        else:
            # Free indices: appear exactly once, in order of first appearance
            free = []
            seen = set()
            for _, idx in tensor_list:
                for name in idx:
                    if name not in seen:
                        seen.add(name)
                        if count[name] == 1:
                            free.append(name)
            output_str = ''.join(name_to_char[n] for n in free)

        return f"{input_str}->{output_str}"

    @staticmethod
    def pairwise_contract(
        A: np.ndarray, A_idx: tuple[str, ...],
        B: np.ndarray, B_idx: tuple[str, ...],
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Contract two tensors along shared indices.

        Convenience method for pairwise contraction. Returns the result
        tensor and its remaining index labels.

        Args:
            A, B: Tensors.
            A_idx, B_idx: Index labels.

        Returns:
            (result_tensor, result_indices).
        """
        shared = set(A_idx) & set(B_idx)
        free_a = [n for n in A_idx if n not in shared]
        free_b = [n for n in B_idx if n not in shared]
        output = tuple(free_a + free_b)

        tn = TensorNetwork()
        result = tn.contract([A, B], [A_idx, B_idx], output)
        return result, output

    @staticmethod
    def trace(tensor: np.ndarray, axis1: int = 0, axis2: int = 1) -> np.ndarray:
        """Trace over two axes of a tensor.

        Args:
            tensor: Array with at least 2 dimensions.
            axis1, axis2: Axes to trace over (must have same size).

        Returns:
            Tensor with those two axes removed.
        """
        return np.trace(tensor, axis1=axis1, axis2=axis2)

    @staticmethod
    def mps_contraction(cores: list[np.ndarray]) -> np.ndarray:
        """Contract a Matrix Product State (equivalent to TT reconstruction).

        Args:
            cores: List of MPS/TT cores, each of shape (r_{k-1}, d_k, r_k).

        Returns:
            Full tensor of shape (d_1, d_2, ..., d_N).
        """
        return TensorTrain.reconstruct(cores)
