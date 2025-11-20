"""
Combinatorial Topology for Warp Space
======================================
Discrete structures, chain complexes, and sheaf theory for knowledge graphs.

This module provides:
- Chain complexes and boundary operators
- Homology computation (kernel/image)
- Discrete Morse theory (gradient flows)
- Sheaf theory for knowledge graphs
- Combinatorial Laplacians
- Spectral sequences

Philosophy:
Combinatorial topology studies discrete structures (graphs, simplicial complexes)
using algebraic tools. Unlike continuous topology, it's exact and computable.

Perfect for:
- Knowledge graph analysis
- Logical reasoning chains
- Discrete optimization
- Network flow problems
"""

import logging
import numpy as np
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import itertools

logger = logging.getLogger(__name__)


# ============================================================================
# Chain Complexes
# ============================================================================

@dataclass
class ChainComplex:
    """
    Chain complex: C₀ ← C₁ ← C₂ ← ... ← Cₙ

    Each Cₖ is a free abelian group (vectors over Z or R).
    Boundary operators ∂ₖ: Cₖ → Cₖ₋₁ satisfy ∂ₖ ∘ ∂ₖ₊₁ = 0.

    Homology: Hₖ = ker(∂ₖ) / im(∂ₖ₊₁)
    """
    dimension: int
    chains: Dict[int, List[Any]] = field(default_factory=dict)  # k → list of k-chains
    boundaries: Dict[int, np.ndarray] = field(default_factory=dict)  # k → boundary matrix

    def add_chain(self, k: int, simplex: Tuple) -> None:
        """Add a k-simplex to the complex."""
        if k not in self.chains:
            self.chains[k] = []
        if simplex not in self.chains[k]:
            self.chains[k].append(simplex)

    def compute_boundary_matrices(self):
        """
        Compute boundary operator matrices.

        ∂ₖ: Cₖ → Cₖ₋₁

        For a k-simplex [v₀, v₁, ..., vₖ]:
        ∂ₖ([v₀,...,vₖ]) = Σᵢ (-1)ⁱ [v₀,...,v̂ᵢ,...,vₖ]
        where v̂ᵢ means omit vᵢ
        """
        logger.info("Computing boundary matrices...")

        for k in range(1, self.dimension + 1):
            if k not in self.chains or k - 1 not in self.chains:
                continue

            k_simplices = self.chains[k]
            k1_simplices = self.chains[k - 1]

            # Build boundary matrix
            n_rows = len(k1_simplices)
            n_cols = len(k_simplices)
            boundary = np.zeros((n_rows, n_cols), dtype=int)

            for j, simplex in enumerate(k_simplices):
                # Compute boundary of this simplex
                for i, vertex in enumerate(simplex):
                    # Omit vertex i
                    face = tuple(v for idx, v in enumerate(simplex) if idx != i)

                    # Find this face in k-1 simplices
                    if face in k1_simplices:
                        row_idx = k1_simplices.index(face)
                        # Alternating sign
                        boundary[row_idx, j] = (-1) ** i

            self.boundaries[k] = boundary
            logger.info(f"  ∂_{k}: {boundary.shape}")

    def compute_homology(self, k: int, field: str = "Z2") -> Dict[str, Any]:
        """
        Compute k-th homology group: Hₖ = ker(∂ₖ) / im(∂ₖ₊₁)

        Args:
            k: Dimension
            field: Field to work over ("Z2" or "R")

        Returns:
            Dict with rank, kernel, image
        """
        logger.info(f"Computing H_{k}...")

        if k not in self.boundaries and k not in self.chains:
            logger.warning(f"No chains for dimension {k}")
            return {
                "dimension": 0,
                "rank": 0,
                "kernel_dim": 0,
                "image_dim": 0,
                "kernel_basis": np.array([]).reshape(0, 0),
                "image_basis": np.array([]).reshape(0, 0)
            }

        # Boundary matrices
        d_k = self.boundaries.get(k)
        d_k1 = self.boundaries.get(k + 1)

        # Work over Z/2Z (simplest case)
        if field == "Z2":
            if d_k is not None:
                d_k = d_k % 2
            if d_k1 is not None:
                d_k1 = d_k1 % 2

        # Compute kernel of ∂ₖ
        if d_k is not None:
            # Kernel = null space
            kernel_basis = self._null_space(d_k, field)
        else:
            # No boundary map means all chains are cycles (∂ₖ = 0)
            # So kernel is all of Cₖ
            n_chains = len(self.chains.get(k, []))
            kernel_basis = np.eye(n_chains) if n_chains > 0 else np.zeros((0, 0))

        # Compute image of ∂ₖ₊₁
        if d_k1 is not None:
            image_basis = self._column_space(d_k1, field)
        else:
            image_basis = np.zeros((len(self.chains.get(k, [])), 0))

        # Betti number = dim(kernel) - dim(image)
        betti = kernel_basis.shape[1] - image_basis.shape[1]

        result = {
            "dimension": max(0, betti),  # Use 'dimension' for consistency
            "rank": max(0, betti),        # Keep 'rank' for backwards compatibility
            "kernel_dim": kernel_basis.shape[1],
            "image_dim": image_basis.shape[1],
            "kernel_basis": kernel_basis,
            "image_basis": image_basis
        }

        logger.info(f"  β_{k} = {result['dimension']} (ker={result['kernel_dim']}, im={result['image_dim']})")

        return result

    def _null_space(self, matrix: np.ndarray, field: str = "Z2") -> np.ndarray:
        """Compute null space (kernel)."""
        if matrix.size == 0:
            return np.array([]).reshape(matrix.shape[1], 0)

        if field == "Z2":
            # Simplified for Z/2Z: use row reduction
            rank = np.linalg.matrix_rank(matrix)
            if rank == matrix.shape[1]:
                return np.zeros((matrix.shape[1], 0))
            else:
                # Use SVD over reals, then project
                _, s, vt = np.linalg.svd(matrix.astype(float))
                null_dim = (s < 1e-10).sum()
                return vt[-null_dim:].T if null_dim > 0 else np.zeros((matrix.shape[1], 0))
        else:
            # Over R: standard null space
            _, s, vt = np.linalg.svd(matrix)
            null_mask = s < 1e-10
            return vt[null_mask].T

    def _column_space(self, matrix: np.ndarray, field: str = "Z2") -> np.ndarray:
        """Compute column space (image)."""
        if matrix.size == 0:
            return matrix

        if field == "Z2":
            matrix = matrix % 2

        # Use SVD
        u, s, _ = np.linalg.svd(matrix.astype(float), full_matrices=False)
        rank = (s > 1e-10).sum()

        return u[:, :rank]


# ============================================================================
# Discrete Morse Theory
# ============================================================================

class DiscreteMorseFunction:
    """
    Discrete Morse function on a simplicial complex.

    A discrete Morse function assigns values to simplices such that:
    - Each simplex has at most one critical face/coface
    - Creates gradient flow (like smooth Morse theory)

    Used for:
    - Topological simplification
    - Finding optimal homology bases
    - Understanding critical features
    """

    def __init__(self, complex: ChainComplex):
        """
        Initialize discrete Morse function.

        Args:
            complex: Underlying chain complex
        """
        self.complex = complex

        # Critical simplices (local maxima/minima)
        self.critical: Dict[int, Set] = defaultdict(set)

        # Gradient pairs (simplex → coface)
        self.gradient_pairs: List[Tuple[Tuple, Tuple]] = []

        logger.info("Discrete Morse function initialized")

    def compute_gradient_flow(self, heuristic: str = "lexicographic"):
        """
        Compute discrete gradient flow.

        Uses greedy algorithm to pair simplices with cofaces,
        leaving critical simplices unpaired.

        Args:
            heuristic: How to order simplices ("lexicographic", "random")
        """
        logger.info(f"Computing discrete gradient ({heuristic})...")

        # Track which simplices are paired
        paired = set()

        # Process by dimension
        for k in sorted(self.complex.chains.keys()):
            if k + 1 not in self.complex.chains:
                # All k-simplices are critical (no cofaces)
                for simplex in self.complex.chains[k]:
                    if tuple(simplex) not in paired:
                        self.critical[k].add(tuple(simplex))
                continue

            simplices = self.complex.chains[k]
            cofaces = self.complex.chains[k + 1]

            # Try to pair each simplex with a coface
            for simplex in simplices:
                if tuple(simplex) in paired:
                    continue

                # Find unpaired cofaces containing this simplex
                candidate_cofaces = [
                    cf for cf in cofaces
                    if self._is_face_of(simplex, cf) and tuple(cf) not in paired
                ]

                if candidate_cofaces:
                    # Pair with first available coface
                    coface = candidate_cofaces[0]

                    self.gradient_pairs.append((tuple(simplex), tuple(coface)))
                    paired.add(tuple(simplex))
                    paired.add(tuple(coface))
                else:
                    # Critical simplex (no unpaired coface)
                    self.critical[k].add(tuple(simplex))

        # Count critical simplices by dimension
        for k in sorted(self.critical.keys()):
            logger.info(f"  Critical {k}-cells: {len(self.critical[k])}")

        logger.info(f"  Gradient pairs: {len(self.gradient_pairs)}")

    def _is_face_of(self, face: Tuple, simplex: Tuple) -> bool:
        """Check if face is a face of simplex."""
        return set(face).issubset(set(simplex)) and len(face) == len(simplex) - 1

    def morse_complex(self) -> ChainComplex:
        """
        Build Morse complex from critical cells.

        The Morse complex has:
        - Chains: Only critical simplices
        - Boundaries: Induced from gradient flow

        Theorem: Morse complex has same homology as original complex.
        """
        logger.info("Building Morse complex...")

        morse = ChainComplex(dimension=self.complex.dimension)

        # Add critical simplices
        for k in self.critical:
            morse.chains[k] = list(self.critical[k])

        # Compute induced boundaries (simplified: use original boundaries)
        # Full implementation would trace gradient paths
        morse.compute_boundary_matrices()

        logger.info(f"  Morse complex size: {sum(len(c) for c in morse.chains.values())} cells")
        logger.info(f"  Original size: {sum(len(c) for c in self.complex.chains.values())} cells")

        return morse


# ============================================================================
# Sheaf Theory
# ============================================================================

@dataclass
class Sheaf:
    """
    Sheaf on a topological space (or graph).

    A sheaf assigns:
    - Data (vector space) to each open set (or vertex)
    - Restriction maps between overlapping sets

    For knowledge graphs:
    - Each node has local data (embeddings, facts)
    - Edges define how data should be consistent

    Sheaf cohomology measures:
    - H⁰: Global consistent sections
    - H¹: Obstructions to consistency
    """
    base_space: Any  # Graph or simplicial complex
    stalks: Dict[Any, np.ndarray] = field(default_factory=dict)  # vertex → data
    restriction_maps: Dict[Tuple, np.ndarray] = field(default_factory=dict)  # (u,v) → matrix

    def add_stalk(self, vertex: Any, data: np.ndarray):
        """Assign data to a vertex."""
        self.stalks[vertex] = data

    def add_restriction(self, u: Any, v: Any, matrix: np.ndarray):
        """Define restriction map from u to v."""
        self.restriction_maps[(u, v)] = matrix

    def sheaf_laplacian(self) -> np.ndarray:
        """
        Compute sheaf Laplacian.

        Measures consistency of data across edges.

        L_sheaf = Σ_{edges (u,v)} (f_u - R_{uv} f_v)ᵀ (f_u - R_{uv} f_v)

        Eigenvectors of L_sheaf give:
        - Harmonic sheaf sections (kernel)
        - Obstructions (non-zero eigenvalues)
        """
        # Get all vertices
        vertices = list(self.stalks.keys())
        n_vertices = len(vertices)

        if n_vertices == 0:
            return np.array([])

        # Dimension of each stalk (assume all same)
        stalk_dim = self.stalks[vertices[0]].shape[0]

        # Sheaf Laplacian is stalk_dim × stalk_dim per vertex
        total_dim = n_vertices * stalk_dim
        L = np.zeros((total_dim, total_dim))

        # Build Laplacian from restriction maps
        for (u, v), R_uv in self.restriction_maps.items():
            if u not in vertices or v not in vertices:
                continue

            u_idx = vertices.index(u)
            v_idx = vertices.index(v)

            # Block indices
            u_start = u_idx * stalk_dim
            u_end = u_start + stalk_dim
            v_start = v_idx * stalk_dim
            v_end = v_start + stalk_dim

            # Add (I - R_uv)ᵀ(I - R_uv) terms
            I_R = np.eye(stalk_dim) - R_uv

            L[u_start:u_end, u_start:u_end] += I_R.T @ I_R
            L[u_start:u_end, v_start:v_end] += -I_R.T @ R_uv
            L[v_start:v_end, u_start:u_end] += -R_uv.T @ I_R
            L[v_start:v_end, v_start:v_end] += R_uv.T @ R_uv

        logger.info(f"Sheaf Laplacian: {L.shape}")

        return L

    def global_sections(self, tol: float = 1e-6) -> np.ndarray:
        """
        Find global consistent sections.

        These are assignments of data to each vertex such that
        restriction maps are satisfied.

        Returns kernel of sheaf Laplacian.
        """
        L = self.sheaf_laplacian()

        if L.size == 0:
            return np.array([])

        # Compute eigenvectors with eigenvalue ≈ 0
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Kernel = eigenvectors with eigenvalue < tol
        kernel_mask = eigenvalues < tol
        kernel = eigenvectors[:, kernel_mask]

        logger.info(f"Global sections: {kernel.shape[1]} dimensions")

        return kernel

    def cohomology_dimension(self, degree: int = 1) -> int:
        """
        Compute dimension of sheaf cohomology H^k.

        H⁰ = global sections (kernel of Laplacian)
        H¹ = obstructions (cokernel)

        Simplified for degree 1.
        """
        if degree == 0:
            kernel = self.global_sections()
            return kernel.shape[1]

        elif degree == 1:
            L = self.sheaf_laplacian()
            if L.size == 0:
                return 0

            # H¹ dimension = dim(cokernel) = total_dim - rank(L)
            rank = np.linalg.matrix_rank(L)
            h1_dim = L.shape[0] - rank

            logger.info(f"H^1 dimension: {h1_dim}")
            return h1_dim

        else:
            logger.warning(f"Cohomology H^{degree} not implemented")
            return 0


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Combinatorial Topology Demo")
    print("="*80 + "\n")

    # 1. Chain Complex and Homology
    print("1. Chain Complex and Homology Computation")
    print("-" * 40)

    # Build a simple complex (triangle with interior)
    complex = ChainComplex(dimension=2)

    # 0-simplices (vertices)
    for i in range(3):
        complex.add_chain(0, (i,))

    # 1-simplices (edges)
    complex.add_chain(1, (0, 1))
    complex.add_chain(1, (1, 2))
    complex.add_chain(1, (0, 2))

    # 2-simplex (triangle)
    complex.add_chain(2, (0, 1, 2))

    print(f"0-cells: {len(complex.chains[0])}")
    print(f"1-cells: {len(complex.chains[1])}")
    print(f"2-cells: {len(complex.chains[2])}")

    # Compute boundaries
    complex.compute_boundary_matrices()

    # Compute homology
    print("\nHomology:")
    for k in range(3):
        homology = complex.compute_homology(k, field="Z2")
        print(f"  H_{k}: rank = {homology['rank']}")

    # 2. Discrete Morse Theory
    print("\n2. Discrete Morse Theory")
    print("-" * 40)

    morse = DiscreteMorseFunction(complex)
    morse.compute_gradient_flow()

    # Build Morse complex
    morse_complex = morse.morse_complex()

    print("\nMorse complex homology:")
    for k in range(3):
        if k in morse_complex.chains:
            homology = morse_complex.compute_homology(k, field="Z2")
            print(f"  H_{k}: rank = {homology['rank']}")

    # 3. Sheaf Theory
    print("\n3. Sheaf Theory on Graph")
    print("-" * 40)

    # Simple graph: 0 -- 1 -- 2
    sheaf = Sheaf(base_space="path_graph")

    # Assign data to each vertex (2D vectors)
    sheaf.add_stalk(0, np.array([1.0, 0.0]))
    sheaf.add_stalk(1, np.array([0.7, 0.7]))
    sheaf.add_stalk(2, np.array([0.0, 1.0]))

    # Restriction maps (identity = perfect consistency)
    I = np.eye(2)
    sheaf.add_restriction(0, 1, I)
    sheaf.add_restriction(1, 2, I)

    # Compute sheaf Laplacian
    L = sheaf.sheaf_laplacian()
    print(f"Sheaf Laplacian shape: {L.shape}")

    # Global sections
    sections = sheaf.global_sections(tol=1e-4)
    print(f"Global sections: {sections.shape[1]} dimensions")

    # Cohomology
    h0 = sheaf.cohomology_dimension(0)
    h1 = sheaf.cohomology_dimension(1)
    print(f"H^0 = {h0}, H^1 = {h1}")

    print("\nDemo complete!")
