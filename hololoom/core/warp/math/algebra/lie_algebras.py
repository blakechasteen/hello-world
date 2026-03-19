"""
Lie Algebras for HoloLoom Warp Drive
=====================================

Lie algebras, root systems, Dynkin diagrams, and universal enveloping algebras.

Lie Algebra (𝔤, [·,·]):
  - Vector space with bilinear bracket [X, Y]
  - Antisymmetry: [X, Y] = -[Y, X]
  - Jacobi identity: [X, [Y, Z]] + [Y, [Z, X]] + [Z, [X, Y]] = 0

Key invariants:
  - Killing form: B(X, Y) = tr(ad(X) ∘ ad(Y))
  - Cartan's criterion: 𝔤 semisimple ⟺ Killing form non-degenerate
  - Root system: eigenvalues of Cartan subalgebra action
  - Dynkin diagram: encodes the Cartan matrix → classifies simple Lie algebras

Classification (simple Lie algebras over ℂ):
  Classical: A_n (sl(n+1)), B_n (so(2n+1)), C_n (sp(2n)), D_n (so(2n))
  Exceptional: E_6, E_7, E_8, F_4, G_2

Author: HoloLoom Team
Date: 2026-03-19
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# LIE ALGEBRA
# ============================================================================

class LieAlgebra:
    """
    Finite-dimensional Lie algebra over ℝ (or ℂ via complex arrays).

    Represented by structure constants: [eᵢ, eⱼ] = Σₖ cᵢⱼₖ eₖ
    where {e₁, ..., eₙ} is a basis.
    """

    def __init__(self, structure_constants: np.ndarray, name: str = "𝔤"):
        """
        Args:
            structure_constants: Array of shape (n, n, n) where
                c[i,j,k] is the coefficient of eₖ in [eᵢ, eⱼ].
            name: Algebra name.
        """
        self.dim = structure_constants.shape[0]
        self.c = structure_constants.astype(float)
        self.name = name

        assert structure_constants.shape == (self.dim, self.dim, self.dim), \
            f"Structure constants must be ({self.dim}, {self.dim}, {self.dim})"

        logger.info(f"LieAlgebra {name}, dim={self.dim}")

    def bracket(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Lie bracket [X, Y] = Σᵢⱼₖ Xᵢ Yⱼ cᵢⱼₖ eₖ.

        Args:
            X, Y: Vectors in ℝⁿ (coefficients in the basis).
        Returns:
            [X, Y] as vector in ℝⁿ.
        """
        # [X, Y]_k = sum_{i,j} X_i Y_j c_{ijk}
        return np.einsum('i,j,ijk->k', X, Y, self.c)

    def adjoint(self, X: np.ndarray) -> np.ndarray:
        """
        Adjoint representation ad(X): 𝔤 → 𝔤, Y ↦ [X, Y].

        Returns the matrix of ad(X) in the standard basis.
        Shape: (dim, dim), where ad(X)[k, j] = Σᵢ Xᵢ cᵢⱼₖ.
        """
        # ad(X)_{kj} = sum_i X_i c_{ijk}
        return np.einsum('i,ijk->kj', X, self.c)

    def killing_form(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Killing form B(X, Y) = tr(ad(X) ∘ ad(Y)).

        The fundamental bilinear form of the Lie algebra.
        """
        adX = self.adjoint(X)
        adY = self.adjoint(Y)
        return np.trace(adX @ adY)

    def killing_matrix(self) -> np.ndarray:
        """
        Killing form as a matrix: B[i,j] = B(eᵢ, eⱼ).
        """
        B = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            ei = np.zeros(self.dim)
            ei[i] = 1.0
            for j in range(i, self.dim):
                ej = np.zeros(self.dim)
                ej[j] = 1.0
                B[i, j] = self.killing_form(ei, ej)
                B[j, i] = B[i, j]
        return B

    def is_semisimple(self) -> bool:
        """
        Cartan's criterion: 𝔤 is semisimple iff the Killing form
        is non-degenerate (det B ≠ 0).
        """
        B = self.killing_matrix()
        return abs(np.linalg.det(B)) > 1e-10

    def is_nilpotent(self) -> bool:
        """
        𝔤 is nilpotent iff the lower central series terminates at 0.

        𝔤⁰ = 𝔤, 𝔤ⁱ⁺¹ = [𝔤, 𝔤ⁱ]
        Nilpotent if 𝔤ⁿ = 0 for some n.

        Engel's theorem: 𝔤 is nilpotent iff ad(X) is nilpotent for all X.
        We check: ad(eᵢ)^dim = 0 for all basis elements.
        """
        for i in range(self.dim):
            ei = np.zeros(self.dim)
            ei[i] = 1.0
            ad = self.adjoint(ei)
            power = np.eye(self.dim)
            for _ in range(self.dim):
                power = power @ ad
            if np.max(np.abs(power)) > 1e-10:
                return False
        return True

    def is_solvable(self) -> bool:
        """
        𝔤 is solvable iff the derived series terminates at 0.

        𝔤⁽⁰⁾ = 𝔤, 𝔤⁽ⁱ⁺¹⁾ = [𝔤⁽ⁱ⁾, 𝔤⁽ⁱ⁾]
        Solvable if 𝔤⁽ⁿ⁾ = 0 for some n.

        Cartan's criterion: 𝔤 solvable iff B(X, Y) = 0 for all X ∈ [𝔤, 𝔤], Y ∈ 𝔤.
        """
        # Compute [𝔤, 𝔤] as a subspace
        brackets = []
        for i in range(self.dim):
            ei = np.zeros(self.dim)
            ei[i] = 1.0
            for j in range(i + 1, self.dim):
                ej = np.zeros(self.dim)
                ej[j] = 1.0
                b = self.bracket(ei, ej)
                if np.max(np.abs(b)) > 1e-12:
                    brackets.append(b)

        if len(brackets) == 0:
            return True  # Abelian is solvable

        # Check Cartan's criterion on derived subalgebra
        bracket_matrix = np.array(brackets)
        B = self.killing_matrix()
        for bvec in bracket_matrix:
            for j in range(self.dim):
                ej = np.zeros(self.dim)
                ej[j] = 1.0
                if abs(self.killing_form(bvec, ej)) > 1e-10:
                    return False
        return True

    def verify_jacobi(self) -> bool:
        """
        Verify Jacobi identity on basis elements:
        [eᵢ, [eⱼ, eₖ]] + [eⱼ, [eₖ, eᵢ]] + [eₖ, [eᵢ, eⱼ]] = 0
        """
        for i in range(self.dim):
            ei = np.zeros(self.dim); ei[i] = 1.0
            for j in range(self.dim):
                ej = np.zeros(self.dim); ej[j] = 1.0
                for k in range(self.dim):
                    ek = np.zeros(self.dim); ek[k] = 1.0
                    term1 = self.bracket(ei, self.bracket(ej, ek))
                    term2 = self.bracket(ej, self.bracket(ek, ei))
                    term3 = self.bracket(ek, self.bracket(ei, ej))
                    if np.max(np.abs(term1 + term2 + term3)) > 1e-10:
                        return False
        return True

    def verify_antisymmetry(self) -> bool:
        """Check [X, Y] = -[Y, X] on basis."""
        for i in range(self.dim):
            ei = np.zeros(self.dim); ei[i] = 1.0
            for j in range(self.dim):
                ej = np.zeros(self.dim); ej[j] = 1.0
                if np.max(np.abs(self.bracket(ei, ej) + self.bracket(ej, ei))) > 1e-10:
                    return False
        return True

    @staticmethod
    def sl(n: int) -> 'LieAlgebra':
        """
        𝔰𝔩(n, ℝ): traceless n×n matrices.

        Dimension: n² - 1.
        Bracket: [X, Y] = XY - YX (matrix commutator).
        """
        dim = n * n - 1
        # Basis: E_{ij} for i≠j, and H_i = E_{ii} - E_{i+1,i+1} for i=1..n-1
        basis = []

        # Off-diagonal matrices E_{ij}
        for i in range(n):
            for j in range(n):
                if i != j:
                    E = np.zeros((n, n))
                    E[i, j] = 1.0
                    basis.append(E)

        # Diagonal traceless: H_i = E_{ii} - E_{i+1,i+1}
        for i in range(n - 1):
            H = np.zeros((n, n))
            H[i, i] = 1.0
            H[i + 1, i + 1] = -1.0
            basis.append(H)

        assert len(basis) == dim

        # Structure constants from [basis[i], basis[j]] = sum_k c[i,j,k] basis[k]
        c = np.zeros((dim, dim, dim))
        for i in range(dim):
            for j in range(dim):
                bracket = basis[i] @ basis[j] - basis[j] @ basis[i]
                # Express in basis (least squares for numerical stability)
                basis_matrix = np.array([b.flatten() for b in basis]).T
                coeffs, _, _, _ = np.linalg.lstsq(basis_matrix, bracket.flatten(), rcond=None)
                c[i, j, :] = coeffs

        return LieAlgebra(c, name=f"sl({n})")

    @staticmethod
    def so(n: int) -> 'LieAlgebra':
        """
        𝔰𝔬(n, ℝ): skew-symmetric n×n matrices.

        Dimension: n(n-1)/2.
        """
        dim = n * (n - 1) // 2
        basis = []

        for i in range(n):
            for j in range(i + 1, n):
                E = np.zeros((n, n))
                E[i, j] = 1.0
                E[j, i] = -1.0
                basis.append(E)

        c = np.zeros((dim, dim, dim))
        basis_matrix = np.array([b.flatten() for b in basis]).T

        for i in range(dim):
            for j in range(dim):
                bracket = basis[i] @ basis[j] - basis[j] @ basis[i]
                coeffs, _, _, _ = np.linalg.lstsq(basis_matrix, bracket.flatten(), rcond=None)
                c[i, j, :] = coeffs

        return LieAlgebra(c, name=f"so({n})")


# ============================================================================
# ROOT SYSTEM
# ============================================================================

class RootSystem:
    """
    Root system of a semisimple Lie algebra, encoded by its Cartan matrix.

    Cartan matrix A: a_{ij} = 2(αᵢ, αⱼ)/(αⱼ, αⱼ)
    where α₁, ..., αᵣ are simple roots.

    Properties:
    - a_{ii} = 2
    - a_{ij} ≤ 0 for i ≠ j
    - a_{ij} = 0 ⟺ a_{ji} = 0
    """

    def __init__(self, cartan_matrix: np.ndarray):
        self.A = np.array(cartan_matrix, dtype=float)
        self.rank = self.A.shape[0]

        assert self.A.shape == (self.rank, self.rank), "Cartan matrix must be square"
        for i in range(self.rank):
            assert self.A[i, i] == 2, f"Diagonal must be 2, got {self.A[i, i]}"

        logger.info(f"RootSystem rank={self.rank}")

    def simple_roots(self) -> np.ndarray:
        """
        Simple roots in the standard realization.

        Uses the Cartan matrix to construct simple roots in ℝʳ
        with inner products matching (αᵢ, αⱼ) = A[i,j] * (αⱼ, αⱼ)/2.

        For simplicity, we realize in ℝʳ via Cholesky-like decomposition
        of the symmetrized Cartan matrix.
        """
        # Symmetrize: B[i,j] = A[i,j] (using unit root lengths for simplicity)
        D = np.diag(np.ones(self.rank))
        B = D @ self.A  # Symmetrized form

        # Make symmetric
        S = (B + B.T) / 2.0

        # Try Cholesky (works if positive definite)
        try:
            L = np.linalg.cholesky(S)
            return L
        except np.linalg.LinAlgError:
            # Fallback: use eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(S)
            eigvals = np.maximum(eigvals, 0)
            return eigvecs @ np.diag(np.sqrt(eigvals))

    def positive_roots(self) -> list[np.ndarray]:
        """
        Compute all positive roots by adding simple roots.

        Algorithm: start with simple roots, repeatedly add simple roots
        to existing positive roots, checking if result is a root
        (via the string criterion: αᵢ-string through β).
        """
        simple = [np.zeros(self.rank) for _ in range(self.rank)]
        for i in range(self.rank):
            simple[i][i] = 1.0

        positive = [s.copy() for s in simple]
        queue = list(range(len(positive)))
        seen = {tuple(s) for s in positive}
        max_roots = self.rank * self.rank * 4  # Safety bound

        idx = 0
        while idx < len(queue) and len(positive) < max_roots:
            beta = positive[queue[idx]]
            idx += 1

            for i in range(self.rank):
                alpha_i = simple[i]
                # p - q = <β, αᵢ∨> where p = max k s.t. β - k·αᵢ is root
                # q = max k s.t. β + k·αᵢ is root
                # If <β, αᵢ∨> > 0, then β + αᵢ might be a root
                inner = int(round(self.A[i, :] @ beta))

                if inner < 0:
                    # β + αᵢ might be a root
                    new_root = beta + alpha_i
                    key = tuple(new_root)
                    if key not in seen and all(c >= 0 for c in new_root):
                        seen.add(key)
                        positive.append(new_root)
                        queue.append(len(positive) - 1)

        return positive

    def weyl_group(self) -> list[np.ndarray]:
        """
        Generate the Weyl group as matrices acting on root space.

        Weyl group W = ⟨s₁, ..., sᵣ⟩ where sᵢ is reflection through αᵢ.
        sᵢ(β) = β - ⟨β, αᵢ∨⟩ αᵢ = β - (A[i,:] · β) eᵢ
        """
        # Simple reflections
        reflections = []
        for i in range(self.rank):
            S = np.eye(self.rank)
            for j in range(self.rank):
                S[j, j] -= self.A[i, j] if j == i else 0
                if j != i:
                    S[j, i] = -self.A[i, j]  # Actually: s_i(e_j) = e_j - A[i,j] e_i
            # Correct: s_i(v) = v - (A[i,:] @ v) * e_i
            reflections.append(i)

        def reflect(i: int, v: np.ndarray) -> np.ndarray:
            coeff = self.A[i, :] @ v
            result = v.copy()
            result[i] -= coeff
            return result

        def reflect_matrix(i: int) -> np.ndarray:
            S = np.eye(self.rank)
            for j in range(self.rank):
                ej = np.zeros(self.rank)
                ej[j] = 1.0
                S[:, j] = reflect(i, ej)
            return S

        # Generate group by closure under multiplication
        generators = [reflect_matrix(i) for i in range(self.rank)]
        group = [np.eye(self.rank)]
        seen = {tuple(np.eye(self.rank).flatten())}
        queue = list(generators)

        max_order = 100000  # Safety

        while queue and len(group) < max_order:
            w = queue.pop(0)
            w_key = tuple(np.round(w, 10).flatten())
            if w_key in seen:
                continue
            seen.add(w_key)
            group.append(w)

            for g in generators:
                product = w @ g
                p_key = tuple(np.round(product, 10).flatten())
                if p_key not in seen:
                    queue.append(product)

        logger.info(f"Weyl group has order {len(group)}")
        return group

    @staticmethod
    def A(n: int) -> 'RootSystem':
        """Type Aₙ: sl(n+1). Cartan matrix with 2 on diagonal, -1 on off-diagonal."""
        A = 2 * np.eye(n)
        for i in range(n - 1):
            A[i, i + 1] = -1
            A[i + 1, i] = -1
        return RootSystem(A)

    @staticmethod
    def B(n: int) -> 'RootSystem':
        """Type Bₙ: so(2n+1)."""
        A = 2 * np.eye(n)
        for i in range(n - 2):
            A[i, i + 1] = -1
            A[i + 1, i] = -1
        if n >= 2:
            A[n - 2, n - 1] = -2
            A[n - 1, n - 2] = -1
        return RootSystem(A)

    @staticmethod
    def D(n: int) -> 'RootSystem':
        """Type Dₙ: so(2n)."""
        assert n >= 3, "D_n requires n >= 3"
        A = 2 * np.eye(n)
        for i in range(n - 2):
            A[i, i + 1] = -1
            A[i + 1, i] = -1
        # Last node branches
        A[n - 3, n - 1] = -1
        A[n - 1, n - 3] = -1
        return RootSystem(A)


# ============================================================================
# DYNKIN DIAGRAM
# ============================================================================

class DynkinDiagram:
    """
    Dynkin diagram: graph encoding of a Cartan matrix.

    Nodes = simple roots. Edges encode angles between roots.
    - No edge: orthogonal (a_{ij} = 0)
    - Single edge: 120° (a_{ij} a_{ji} = 1)
    - Double edge: 135° (a_{ij} a_{ji} = 2), arrow → shorter root
    - Triple edge: 150° (a_{ij} a_{ji} = 3), arrow → shorter root
    """

    def __init__(self, cartan_matrix: np.ndarray):
        self.A = np.array(cartan_matrix, dtype=int)
        self.rank = self.A.shape[0]

    @staticmethod
    def from_cartan_matrix(A: np.ndarray) -> 'DynkinDiagram':
        return DynkinDiagram(A)

    def edges(self) -> list[tuple[int, int, int]]:
        """
        Return edges as (i, j, bond_order) where bond_order = a_{ij} * a_{ji}.
        """
        result = []
        for i in range(self.rank):
            for j in range(i + 1, self.rank):
                bond = self.A[i, j] * self.A[j, i]
                if bond != 0:
                    result.append((i, j, bond))
        return result

    def classify(self) -> str:
        """
        Classify the Dynkin diagram.

        Returns one of: A_n, B_n, C_n, D_n, E_6, E_7, E_8, F_4, G_2,
        or "unknown" for non-standard Cartan matrices.
        """
        n = self.rank
        edges = self.edges()

        if n == 1:
            return "A_1"

        if n == 2:
            if len(edges) == 0:
                return "A_1 x A_1"
            bond = edges[0][2]
            if bond == 1:
                return "A_2"
            elif bond == 2:
                if self.A[0, 1] == -2:
                    return "B_2"
                else:
                    return "C_2"
            elif bond == 3:
                return "G_2"

        # Check if it's a simply-laced diagram (all bonds = 1)
        all_single = all(b == 1 for _, _, b in edges)
        is_linear = self._is_linear_graph(edges)

        if all_single and is_linear and len(edges) == n - 1:
            return f"A_{n}"

        if all_single and len(edges) == n - 1 and not is_linear:
            # Check for D_n (one branch point of degree 3)
            degrees = self._node_degrees(edges)
            branch_points = [v for v, d in degrees.items() if d >= 3]
            if len(branch_points) == 1 and degrees[branch_points[0]] == 3:
                return f"D_{n}"

            # Check for E_6, E_7, E_8
            if n in (6, 7, 8) and len(branch_points) == 1:
                return f"E_{n}"

        # Non-simply-laced
        if not all_single and len(edges) == n - 1:
            double_edges = [(i, j) for i, j, b in edges if b == 2]
            if len(double_edges) == 1:
                if is_linear:
                    i, j = double_edges[0]
                    # B_n: double bond at end, arrow toward short root (last)
                    if j == n - 1 or i == 0:
                        if self.A[i, j] == -2:
                            return f"C_{n}"
                        else:
                            return f"B_{n}"
                    elif n == 4:
                        return "F_4"

        return f"unknown_rank_{n}"

    def _is_linear_graph(self, edges: list[tuple[int, int, int]]) -> bool:
        """Check if edges form a path (linear graph)."""
        degrees = self._node_degrees(edges)
        endpoints = [v for v, d in degrees.items() if d == 1]
        return len(endpoints) == 2 and all(d <= 2 for d in degrees.values())

    def _node_degrees(self, edges: list[tuple[int, int, int]]) -> dict[int, int]:
        degrees: dict[int, int] = {}
        for i, j, _ in edges:
            degrees[i] = degrees.get(i, 0) + 1
            degrees[j] = degrees.get(j, 0) + 1
        return degrees


# ============================================================================
# UNIVERSAL ENVELOPING ALGEBRA
# ============================================================================

class UniversalEnveloping:
    """
    Universal enveloping algebra U(𝔤) of a Lie algebra 𝔤.

    U(𝔤) = T(𝔤) / (X⊗Y - Y⊗X - [X,Y])
    where T(𝔤) is the tensor algebra.

    PBW Theorem: {e₁^{a₁} e₂^{a₂} ⋯ eₙ^{aₙ} : aᵢ ≥ 0}
    forms a basis for U(𝔤) (ordered monomials).
    """

    def __init__(self, lie_algebra: LieAlgebra):
        self.g = lie_algebra
        self.dim = lie_algebra.dim
        logger.info(f"UniversalEnveloping of {lie_algebra.name}")

    def PBW_basis(self, degree: int) -> list[tuple[int, ...]]:
        """
        Poincaré-Birkhoff-Witt basis monomials up to given degree.

        Each monomial is a tuple (a₁, a₂, ..., aₙ) where aᵢ ≥ 0
        and Σ aᵢ ≤ degree. Represents e₁^{a₁} e₂^{a₂} ⋯ eₙ^{aₙ}.

        Returns:
            List of exponent tuples.
        """
        monomials = []
        self._generate_monomials(0, degree, [], monomials)
        return monomials

    def _generate_monomials(self, idx: int, remaining: int,
                            current: list[int], result: list[tuple[int, ...]]):
        if idx == self.dim:
            result.append(tuple(current))
            return

        for a in range(remaining + 1):
            self._generate_monomials(idx + 1, remaining - a, current + [a], result)

    def PBW_dimension(self, degree: int) -> int:
        """
        Number of PBW basis elements up to given degree.

        This is C(n + d, d) where n = dim(𝔤), d = degree.
        """
        from math import comb
        return comb(self.dim + degree, degree)

    def multiply_basis(self, mono1: tuple[int, ...],
                       mono2: tuple[int, ...]) -> dict[tuple[int, ...], float]:
        """
        Multiply two PBW monomials, reordering via commutation relations.

        e_j e_i = e_i e_j + [e_j, e_i]  (for j > i)

        Returns a dictionary mapping PBW monomials to coefficients.
        """
        # Convert to a sequence of generator indices
        word1 = []
        for i, a in enumerate(mono1):
            word1.extend([i] * a)

        word2 = []
        for i, a in enumerate(mono2):
            word2.extend([i] * a)

        # Start with the concatenated word and bubble-sort to PBW order
        # Each swap introduces bracket terms
        result: dict[tuple[int, ...], float] = {}
        self._reorder_word(word1 + word2, 1.0, result)
        return result

    def _reorder_word(self, word: list[int], coeff: float,
                      result: dict[tuple[int, ...], float],
                      depth: int = 0):
        """Recursively reorder a word into PBW form using commutation."""
        if depth > 50 or abs(coeff) < 1e-15:
            return

        # Find first out-of-order pair
        for pos in range(len(word) - 1):
            if word[pos] > word[pos + 1]:
                i, j = word[pos + 1], word[pos]  # i < j

                # Swap: e_j e_i = e_i e_j + [e_j, e_i]
                swapped = word[:pos] + [i, j] + word[pos + 2:]
                self._reorder_word(swapped, coeff, result, depth + 1)

                # Bracket term: [e_j, e_i] = sum_k c[j,i,k] e_k
                ei = np.zeros(self.dim); ei[i] = 1.0
                ej = np.zeros(self.dim); ej[j] = 1.0
                bracket = self.g.bracket(ej, ei)

                for k in range(self.dim):
                    if abs(bracket[k]) > 1e-15:
                        new_word = word[:pos] + [k] + word[pos + 2:]
                        self._reorder_word(new_word, coeff * bracket[k], result, depth + 1)
                return

        # Word is in order — convert to monomial
        mono = [0] * self.dim
        for idx in word:
            mono[idx] += 1
        key = tuple(mono)
        result[key] = result.get(key, 0.0) + coeff
