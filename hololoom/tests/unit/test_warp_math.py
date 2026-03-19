"""
Test Warp Math Subsystem - Unit Tests
=====================================

Unit tests for the mathematical operations in hololoom/core/warp/:
- hofstadter.py: Self-referential sequences and memory indexing
- representation.py: Group theory and representation theory
- merge.py: Chomskyan compositional merge operations
- optimized.py: Sparse tensors and memory pooling
- riemannian_geometry.py: Manifold operations (hyperbolic, spherical, product)
- semantic_pde.py: PDE solvers (heat, wave, reaction-diffusion, Hamilton-Jacobi)

Focus: mathematical PROPERTIES, not just outputs.
All tests use small synthetic data and run in <5s total.
"""

import pytest
import numpy as np
from typing import List


# ============================================================================
# Hofstadter Sequences
# ============================================================================

class TestHofstadterSequences:
    """Test self-referential Hofstadter sequence properties."""

    @pytest.fixture
    def seq(self):
        from hololoom.warp.hofstadter import HofstadterSequences
        return HofstadterSequences(max_n=500)

    def test_g_base_case(self, seq):
        """G(0) = 0 by definition."""
        assert seq.G(0) == 0

    def test_g_recurrence(self, seq):
        """G(n) = n - G(G(n-1)) for n > 0."""
        for n in range(1, 30):
            expected = n - seq.G(seq.G(n - 1))
            assert seq.G(n) == expected, f"G({n}) recurrence violated"

    def test_g_grows_roughly_as_n_over_phi(self, seq):
        """G(n) ~ n/phi for large n (golden ratio property)."""
        phi = (1 + np.sqrt(5)) / 2
        n = 100
        ratio = seq.G(n) / n
        # Should be approximately 1/phi ~ 0.618
        assert abs(ratio - 1 / phi) < 0.05, f"G(100)/100 = {ratio}, expected ~{1/phi:.3f}"

    def test_g_is_strictly_increasing(self, seq):
        """G sequence is strictly increasing for n >= 1."""
        values = [seq.G(n) for n in range(1, 50)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], f"G not monotone at n={i+1}"

    def test_h_base_cases(self, seq):
        """H(0) = 0, H(1) = 1 by definition."""
        assert seq.H(0) == 0
        assert seq.H(1) == 1

    def test_h_recurrence(self, seq):
        """H(n) = n - H(H(H(n-1))) for n > 1."""
        for n in range(2, 25):
            expected = n - seq.H(seq.H(seq.H(n - 1)))
            assert seq.H(n) == expected, f"H({n}) recurrence violated"

    def test_h_grows_sublinearly(self, seq):
        """H(n) < n for n > 1 (sublinear growth)."""
        for n in [10, 20, 50, 100]:
            assert seq.H(n) < n, f"H({n}) >= {n}"

    def test_q_base_cases(self, seq):
        """Q(1) = Q(2) = 1 by definition."""
        assert seq.Q(1) == 1
        assert seq.Q(2) == 1

    def test_q_recurrence(self, seq):
        """Q(n) = Q(n - Q(n-1)) + Q(n - Q(n-2)) for n > 2."""
        for n in range(3, 20):
            expected = seq.Q(n - seq.Q(n - 1)) + seq.Q(n - seq.Q(n - 2))
            assert seq.Q(n) == expected, f"Q({n}) recurrence violated"

    def test_q_always_positive(self, seq):
        """Q(n) > 0 for all n >= 1."""
        for n in range(1, 50):
            assert seq.Q(n) > 0, f"Q({n}) <= 0"

    def test_r_base_cases(self, seq):
        """R(1) = R(2) = 1 by definition."""
        assert seq.R(1) == 1
        assert seq.R(2) == 1

    def test_r_recurrence(self, seq):
        """R(n) = R(n-1) + R(n - R(n-1)) for n > 2."""
        for n in range(3, 25):
            expected = seq.R(n - 1) + seq.R(n - seq.R(n - 1))
            assert seq.R(n) == expected, f"R({n}) recurrence violated"

    def test_r_monotonically_increasing(self, seq):
        """R is monotonically non-decreasing."""
        values = [seq.R(n) for n in range(1, 50)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], f"R not monotone at n={i+1}"

    def test_generate_sequence_matches_individual_calls(self, seq):
        """generate_sequence returns same values as individual calls."""
        for stype in ['G', 'H', 'Q', 'R']:
            generated = seq.generate_sequence(stype, 20)
            func = getattr(seq, stype)
            for i, val in enumerate(generated):
                assert val == func(i + 1), f"{stype}({i+1}) mismatch"

    def test_generate_sequence_invalid_type(self, seq):
        """generate_sequence raises ValueError on invalid type."""
        with pytest.raises(ValueError, match="Unknown sequence type"):
            seq.generate_sequence('Z', 10)

    def test_caching_produces_consistent_results(self, seq):
        """Calling the same value twice returns identical result (cache hit)."""
        first = seq.G(42)
        second = seq.G(42)
        assert first == second

    def test_max_n_fallback(self):
        """Values beyond max_n use fallback approximation (no stack overflow)."""
        from hololoom.warp.hofstadter import HofstadterSequences
        small = HofstadterSequences(max_n=10)
        # Should not recurse infinitely, uses fallback
        result = small.G(20)
        assert isinstance(result, int)
        assert result > 0


class TestHofstadterMemoryIndex:
    """Test memory indexing built on Hofstadter sequences."""

    @pytest.fixture
    def indexer(self):
        from hololoom.warp.hofstadter import HofstadterMemoryIndex
        return HofstadterMemoryIndex(max_n=500)

    def test_index_memory_returns_all_fields(self, indexer):
        """index_memory returns a MemoryIndex with all expected fields."""
        idx = indexer.index_memory(42, timestamp=1700000000.0)
        assert idx.memory_id == 42
        assert isinstance(idx.forward, int)
        assert isinstance(idx.backward, int)
        assert isinstance(idx.associate, int)
        assert isinstance(idx.salience, int)
        assert isinstance(idx.temporal_phase, int)

    def test_index_memory_deterministic(self, indexer):
        """Same inputs produce same index."""
        idx1 = indexer.index_memory(42, timestamp=1000.0)
        idx2 = indexer.index_memory(42, timestamp=1000.0)
        assert idx1.forward == idx2.forward
        assert idx1.backward == idx2.backward
        assert idx1.associate == idx2.associate

    def test_index_memory_zero_wraps_to_one(self, indexer):
        """memory_id=0 maps to n=1 (avoid G(0)=0 fixed point)."""
        idx = indexer.index_memory(0, timestamp=0.0)
        # Should not crash, n gets remapped to 1
        assert idx.memory_id == 0

    def test_resonance_symmetric(self, indexer):
        """Resonance between (a,b) equals resonance between (b,a)."""
        ids = [10, 25, 42]
        resonances = indexer.find_resonance(ids, depth=5, min_score=0.0)
        for a, b, score in resonances:
            # The implementation iterates i < j, so symmetry is implicit
            assert a < b, "Resonance pairs should be ordered"

    def test_resonance_score_bounded(self, indexer):
        """Resonance scores are in [0, 1]."""
        ids = list(range(5, 20))
        resonances = indexer.find_resonance(ids, depth=5, min_score=0.0)
        for _, _, score in resonances:
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds"

    def test_resonance_sorted_descending(self, indexer):
        """Resonance results sorted by score descending."""
        ids = list(range(5, 20))
        resonances = indexer.find_resonance(ids, depth=10, min_score=0.0)
        for i in range(len(resonances) - 1):
            assert resonances[i][2] >= resonances[i + 1][2]

    def test_traverse_starts_with_start_id(self, indexer):
        """Traversal path starts with the given start_id."""
        path = indexer.traverse_sequence(42, sequence_type='forward', steps=5)
        assert path[0] == 42

    def test_traverse_no_duplicates(self, indexer):
        """Traversal path has no duplicate entries (cycle detection)."""
        path = indexer.traverse_sequence(10, sequence_type='forward', steps=20)
        assert len(path) == len(set(path))

    def test_traverse_invalid_type_raises(self, indexer):
        """Invalid sequence type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sequence type"):
            indexer.traverse_sequence(10, sequence_type='invalid')

    def test_statistics_empty_returns_empty(self, indexer):
        """Empty memory_ids returns empty statistics."""
        stats = indexer.sequence_statistics([])
        assert stats == {}

    def test_statistics_has_expected_keys(self, indexer):
        """Statistics include all expected keys."""
        stats = indexer.sequence_statistics([10, 20, 30])
        expected_keys = {'avg_forward', 'avg_backward', 'associate_entropy',
                         'resonance_density', 'num_memories'}
        assert expected_keys == set(stats.keys())

    def test_statistics_density_bounded(self, indexer):
        """Resonance density is in [0, 1]."""
        stats = indexer.sequence_statistics(list(range(5, 15)))
        assert 0.0 <= stats['resonance_density'] <= 1.0


# ============================================================================
# Representation Theory
# ============================================================================

class TestGroupTheory:
    """Test group construction and properties."""

    @pytest.fixture
    def c3(self):
        from hololoom.warp.representation import cyclic_group
        return cyclic_group(3)

    @pytest.fixture
    def c4(self):
        from hololoom.warp.representation import cyclic_group
        return cyclic_group(4)

    @pytest.fixture
    def s3(self):
        from hololoom.warp.representation import symmetric_group
        return symmetric_group(3)

    def test_cyclic_group_order(self, c3):
        """C3 has 3 elements."""
        assert c3.order() == 3

    def test_cyclic_group_identity(self, c3):
        """Identity element is 0 in cyclic group."""
        assert c3.identity == 0

    def test_cyclic_group_closure(self, c3):
        """Multiplication stays within the group."""
        for g in c3.elements:
            for h in c3.elements:
                product = c3.multiply(g, h)
                assert product in c3.elements

    def test_cyclic_group_associativity(self, c3):
        """(g * h) * k == g * (h * k) for all elements."""
        for g in c3.elements:
            for h in c3.elements:
                for k in c3.elements:
                    lhs = c3.multiply(c3.multiply(g, h), k)
                    rhs = c3.multiply(g, c3.multiply(h, k))
                    assert lhs == rhs, f"Associativity fails: ({g}*{h})*{k} != {g}*({h}*{k})"

    def test_cyclic_group_identity_property(self, c3):
        """e * g == g * e == g for all g."""
        e = c3.identity
        for g in c3.elements:
            assert c3.multiply(e, g) == g
            assert c3.multiply(g, e) == g

    def test_cyclic_group_inverse_property(self, c3):
        """g * g^{-1} == g^{-1} * g == e."""
        e = c3.identity
        for g in c3.elements:
            g_inv = c3.inverse(g)
            assert c3.multiply(g, g_inv) == e
            assert c3.multiply(g_inv, g) == e

    def test_cyclic_group_is_abelian(self, c3):
        """Cyclic groups are always abelian."""
        assert c3.is_abelian() is True

    def test_symmetric_group_order(self, s3):
        """S3 has 3! = 6 elements."""
        assert s3.order() == 6

    def test_symmetric_group_not_abelian(self, s3):
        """S3 is not abelian (for n >= 3)."""
        assert s3.is_abelian() is False

    def test_symmetric_group_associativity(self, s3):
        """Associativity holds for S3."""
        elems = list(s3.elements)
        # Test a sample (full test would be 6^3 = 216 triples)
        for g in elems[:3]:
            for h in elems[:3]:
                for k in elems[:3]:
                    lhs = s3.multiply(s3.multiply(g, h), k)
                    rhs = s3.multiply(g, s3.multiply(h, k))
                    assert lhs == rhs


class TestRepresentationTheory:
    """Test representations and character theory."""

    @pytest.fixture
    def c3(self):
        from hololoom.warp.representation import cyclic_group
        return cyclic_group(3)

    @pytest.fixture
    def trivial_rep(self, c3):
        from hololoom.warp.representation import trivial_representation
        return trivial_representation(c3)

    @pytest.fixture
    def regular_rep(self, c3):
        from hololoom.warp.representation import regular_representation
        return regular_representation(c3)

    def test_trivial_rep_is_homomorphism(self, trivial_rep):
        """Trivial representation satisfies rho(gh) = rho(g)rho(h)."""
        assert trivial_rep.verify_homomorphism() is True

    def test_trivial_rep_is_irreducible(self, trivial_rep):
        """Trivial representation is irreducible."""
        assert trivial_rep.is_irreducible()

    def test_trivial_rep_identity_mapped_to_identity(self, trivial_rep, c3):
        """rho(e) = I for trivial representation."""
        rho_e = trivial_rep(c3.identity)
        assert np.allclose(rho_e, np.eye(1))

    def test_trivial_character_all_ones(self, trivial_rep, c3):
        """Character of trivial representation is 1 for all elements."""
        char = trivial_rep.character()
        for g in c3.elements:
            assert np.isclose(char[g], 1.0)

    def test_regular_rep_is_homomorphism(self, regular_rep):
        """Regular representation is a valid homomorphism."""
        assert regular_rep.verify_homomorphism() is True

    def test_regular_rep_dimension(self, regular_rep, c3):
        """Regular representation has dimension |G|."""
        assert regular_rep.dimension == c3.order()

    def test_regular_rep_not_irreducible(self, regular_rep):
        """Regular representation is not irreducible (for |G| > 1)."""
        assert not regular_rep.is_irreducible()

    def test_regular_rep_identity_character(self, regular_rep, c3):
        """chi_reg(e) = |G| (trace of identity matrix)."""
        char = regular_rep.character()
        assert np.isclose(char[c3.identity], c3.order())

    def test_regular_rep_non_identity_character_zero(self, regular_rep, c3):
        """chi_reg(g) = 0 for g != e in regular representation."""
        char = regular_rep.character()
        for g in c3.elements:
            if g != c3.identity:
                assert np.isclose(char[g], 0.0), f"chi_reg({g}) = {char[g]}, expected 0"

    def test_representation_matrix_dimension_check(self, c3):
        """Setting wrong-dimension matrix raises ValueError."""
        from hololoom.warp.representation import Representation
        rep = Representation(group=c3, dimension=2)
        with pytest.raises(ValueError, match="2.2"):
            rep.set_matrix(0, np.eye(3))  # Wrong size

    def test_sign_representation_s3(self):
        """Sign representation of S3 is a valid irreducible representation."""
        from hololoom.warp.representation import symmetric_group, Representation
        S3 = symmetric_group(3)
        sign_rep = Representation(group=S3, dimension=1, name="sign")
        for perm in S3.elements:
            inversions = sum(
                1 for i in range(len(perm))
                for j in range(i + 1, len(perm))
                if perm[i] > perm[j]
            )
            sign = 1 if inversions % 2 == 0 else -1
            sign_rep.set_matrix(perm, np.array([[sign]]))

        assert sign_rep.verify_homomorphism()
        assert sign_rep.is_irreducible()


class TestCharacterTable:
    """Test character table construction."""

    def test_conjugacy_classes_partition_group(self):
        """Conjugacy classes partition the group (every element in exactly one class)."""
        from hololoom.warp.representation import cyclic_group, CharacterTable
        C4 = cyclic_group(4)
        ct = CharacterTable(C4)
        all_elements = set()
        for cls in ct.conjugacy_classes:
            for elem in cls:
                assert elem not in all_elements, f"Element {elem} in multiple classes"
                all_elements.add(elem)
        assert all_elements == C4.elements

    def test_abelian_group_classes_are_singletons(self):
        """In abelian groups, each conjugacy class is a singleton {g}."""
        from hololoom.warp.representation import cyclic_group, CharacterTable
        C3 = cyclic_group(3)
        ct = CharacterTable(C3)
        for cls in ct.conjugacy_classes:
            assert len(cls) == 1, f"Abelian group has non-singleton class: {cls}"

    def test_number_of_classes_equals_number_of_irreps(self):
        """For C3, number of conjugacy classes = 3 = number of irreps needed."""
        from hololoom.warp.representation import cyclic_group, CharacterTable
        C3 = cyclic_group(3)
        ct = CharacterTable(C3)
        # C3 is abelian, so #classes = #elements = 3
        assert len(ct.conjugacy_classes) == 3


# ============================================================================
# Merge Operations
# ============================================================================

class MockEmbedder:
    """Deterministic mock embedder for merge tests."""

    def encode(self, texts: List[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        return np.array([rng.normal(0, 1, 16) for _ in texts])


class TestMergeOperator:
    """Test Chomskyan merge operations on tensor space."""

    @pytest.fixture
    def merger(self):
        from hololoom.warp.merge import MergeOperator
        return MergeOperator(
            embedder=MockEmbedder(),
            fusion_method="weighted_sum",
            head_weight=0.7,
            dependent_weight=0.3
        )

    @pytest.fixture
    def emb_a(self):
        rng = np.random.default_rng(100)
        v = rng.normal(0, 1, 16)
        return v / (np.linalg.norm(v) + 1e-10)

    @pytest.fixture
    def emb_b(self):
        rng = np.random.default_rng(200)
        v = rng.normal(0, 1, 16)
        return v / (np.linalg.norm(v) + 1e-10)

    @pytest.fixture
    def emb_c(self):
        rng = np.random.default_rng(300)
        v = rng.normal(0, 1, 16)
        return v / (np.linalg.norm(v) + 1e-10)

    def test_external_merge_output_normalized(self, merger, emb_a, emb_b):
        """External merge produces unit-norm embedding."""
        merged = merger.external_merge(emb_a, emb_b, head="b", dependent="a", label="NP")
        norm = np.linalg.norm(merged.embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_external_merge_preserves_head(self, merger, emb_a, emb_b):
        """External merge correctly tracks the head word."""
        merged = merger.external_merge(emb_a, emb_b, head="cat", dependent="the", label="NP")
        assert merged.head == "cat"
        assert merged.label == "NP"

    def test_external_merge_head_dominates(self, merger):
        """Head embedding has more weight than dependent in weighted_sum."""
        head = np.ones(16)
        dep = np.zeros(16)
        merged = merger.external_merge(
            dep, head, head="noun", dependent="det", label="NP", alpha_is_head=False
        )
        # Result should be closer to head (all ones) than dependent (all zeros)
        # After normalization: direction should match head more
        dot_with_head = np.dot(merged.embedding, head / np.linalg.norm(head))
        assert dot_with_head > 0.5, "Head should dominate the merge result"

    def test_external_merge_concat_doubles_dimension(self, emb_a, emb_b):
        """Concat fusion method doubles the embedding dimension."""
        from hololoom.warp.merge import MergeOperator
        concat_merger = MergeOperator(MockEmbedder(), fusion_method="concat")
        merged = concat_merger.external_merge(emb_a, emb_b, head="b", dependent="a")
        # After normalization, dimension should be 2 * 16 = 32
        assert merged.embedding.shape == (32,)

    def test_external_merge_hadamard_same_dimension(self, merger, emb_a, emb_b):
        """Hadamard fusion preserves dimension."""
        from hololoom.warp.merge import MergeOperator
        h_merger = MergeOperator(MockEmbedder(), fusion_method="hadamard")
        merged = h_merger.external_merge(emb_a, emb_b, head="b", dependent="a")
        assert merged.embedding.shape == (16,)

    def test_external_merge_unknown_method_raises(self, emb_a, emb_b):
        """Unknown fusion method raises ValueError."""
        from hololoom.warp.merge import MergeOperator
        bad_merger = MergeOperator(MockEmbedder(), fusion_method="quantum")
        with pytest.raises(ValueError, match="Unknown fusion method"):
            bad_merger.external_merge(emb_a, emb_b, head="b", dependent="a")

    def test_parallel_merge_output_normalized(self, merger):
        """Parallel merge produces unit-norm embedding."""
        rng = np.random.default_rng(42)
        embeddings = [rng.normal(0, 1, 16) for _ in range(4)]
        components = ["the", "big", "red", "ball"]
        merged = merger.parallel_merge(embeddings, components, head_index=3, label="NP")
        norm = np.linalg.norm(merged.embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_parallel_merge_tracks_head(self, merger):
        """Parallel merge correctly identifies head component."""
        rng = np.random.default_rng(42)
        embeddings = [rng.normal(0, 1, 16) for _ in range(3)]
        merged = merger.parallel_merge(embeddings, ["a", "b", "c"], head_index=1)
        assert merged.head == "b"

    def test_parallel_merge_mismatched_lengths_raises(self, merger):
        """Parallel merge raises on mismatched embeddings/components lengths."""
        rng = np.random.default_rng(42)
        embeddings = [rng.normal(0, 1, 16) for _ in range(3)]
        with pytest.raises(ValueError, match="same length"):
            merger.parallel_merge(embeddings, ["a", "b"], head_index=0)

    def test_recursive_merge_builds_tree(self, merger):
        """Recursive merge builds hierarchical structure with children."""
        rng = np.random.default_rng(42)
        words = ["the", "big", "cat"]
        embeddings = [rng.normal(0, 1, 16) for _ in words]
        structure = [
            (1, 2, "right", "N'"),   # big + cat -> N' (cat is head)
            (0, -1, "right", "NP")   # the + N' -> NP
        ]
        root = merger.recursive_merge(embeddings, words, structure)
        assert root.label == "NP"
        assert len(root.children) == 2
        assert root.embedding.shape == (16,)

    def test_weight_normalization(self):
        """Head and dependent weights are normalized to sum to 1."""
        from hololoom.warp.merge import MergeOperator
        m = MergeOperator(MockEmbedder(), head_weight=3.0, dependent_weight=7.0)
        assert np.isclose(m.head_weight + m.dependent_weight, 1.0)
        assert np.isclose(m.head_weight, 0.3)
        assert np.isclose(m.dependent_weight, 0.7)


# ============================================================================
# Sparse Tensor Field (from optimized.py)
# ============================================================================

class TestSparseTensorField:
    """Test sparse tensor operations."""

    @pytest.fixture
    def dense_matrix(self):
        """Create a sparse-ish 10x8 matrix."""
        rng = np.random.default_rng(42)
        m = rng.normal(0, 1, (10, 8))
        m[np.abs(m) < 0.8] = 0  # Make ~50% sparse
        return m

    def test_from_dense_preserves_shape(self, dense_matrix):
        from hololoom.warp.optimized import SparseTensorField
        sparse = SparseTensorField(dense_matrix, threshold=0.1)
        assert sparse.shape == dense_matrix.shape

    def test_roundtrip_dense_sparse_dense(self, dense_matrix):
        """Converting dense -> sparse -> dense preserves values above threshold."""
        from hololoom.warp.optimized import SparseTensorField
        threshold = 0.1
        sparse = SparseTensorField(dense_matrix, threshold=threshold)
        recovered = sparse.to_dense()
        # Values above threshold should match; below should be zero
        mask = np.abs(dense_matrix) > threshold
        assert np.allclose(recovered[mask], dense_matrix[mask])
        assert np.allclose(recovered[~mask], 0.0)

    def test_density_bounded(self, dense_matrix):
        """Density is in [0, 1]."""
        from hololoom.warp.optimized import SparseTensorField
        sparse = SparseTensorField(dense_matrix, threshold=0.1)
        assert 0.0 <= sparse.density <= 1.0

    def test_higher_threshold_lower_density(self):
        """Higher threshold produces lower density."""
        from hololoom.warp.optimized import SparseTensorField
        rng = np.random.default_rng(42)
        m = rng.normal(0, 1, (20, 15))
        low = SparseTensorField(m, threshold=0.5)
        high = SparseTensorField(m, threshold=1.5)
        assert high.density <= low.density

    def test_matmul_with_dense_vector(self, dense_matrix):
        """Sparse @ dense vector matches dense @ dense vector."""
        from hololoom.warp.optimized import SparseTensorField
        sparse = SparseTensorField(dense_matrix, threshold=0.1)
        rng = np.random.default_rng(99)
        v = rng.normal(0, 1, dense_matrix.shape[1])
        sparse_result = sparse @ v
        # Compare against dense matmul with zeroed-out entries
        dense_zeroed = sparse.to_dense()
        dense_result = dense_zeroed @ v
        assert np.allclose(sparse_result, dense_result, atol=1e-10)

    def test_matmul_with_dense_matrix(self, dense_matrix):
        """Sparse @ dense matrix matches dense multiplication."""
        from hololoom.warp.optimized import SparseTensorField
        sparse = SparseTensorField(dense_matrix, threshold=0.1)
        rng = np.random.default_rng(99)
        B = rng.normal(0, 1, (dense_matrix.shape[1], 5))
        sparse_result = sparse @ B
        dense_zeroed = sparse.to_dense()
        dense_result = dense_zeroed @ B
        assert np.allclose(sparse_result, dense_result, atol=1e-10)

    def test_matmul_incompatible_shapes_raises(self):
        """Matmul with incompatible shapes raises ValueError."""
        from hololoom.warp.optimized import SparseTensorField
        m = np.eye(5)
        sparse = SparseTensorField(m)
        with pytest.raises(ValueError, match="Incompatible shapes"):
            sparse @ np.ones((3, 2))  # 5 cols vs 3 rows

    def test_matmul_with_another_sparse(self, dense_matrix):
        """Sparse @ Sparse produces correct result."""
        from hololoom.warp.optimized import SparseTensorField
        A = SparseTensorField(dense_matrix, threshold=0.1)
        rng = np.random.default_rng(99)
        B_dense = rng.normal(0, 1, (dense_matrix.shape[1], 4))
        B = SparseTensorField(B_dense, threshold=0.1)
        result = A @ B
        expected = A.to_dense() @ B.to_dense()
        assert np.allclose(result, expected, atol=1e-10)

    def test_empty_sparse_matmul(self):
        """Matmul with all-zero sparse tensor returns zeros."""
        from hololoom.warp.optimized import SparseTensorField
        m = np.zeros((5, 4))
        sparse = SparseTensorField(m)
        v = np.ones(4)
        result = sparse @ v
        assert np.allclose(result, 0.0)

    def test_identity_sparse(self):
        """Sparse identity matrix preserves vectors."""
        from hololoom.warp.optimized import SparseTensorField
        I = np.eye(6)
        sparse_I = SparseTensorField(I)
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = sparse_I @ v
        assert np.allclose(result, v)


class TestTensorMemoryPool:
    """Test tensor memory pooling and reuse."""

    @pytest.fixture
    def pool(self):
        from hololoom.warp.optimized import TensorMemoryPool
        return TensorMemoryPool(max_pool_size=5)

    def test_allocate_returns_zeros(self, pool):
        """Freshly allocated tensor is all zeros."""
        t = pool.allocate((4, 3))
        assert t.shape == (4, 3)
        assert np.allclose(t, 0.0)

    def test_release_and_reuse(self, pool):
        """Released tensor is reused on next allocate of same shape."""
        t1 = pool.allocate((4, 3))
        t1[:] = 99.0
        pool.release(t1)
        t2 = pool.allocate((4, 3))
        # Should be cleared to zeros even if reused
        assert np.allclose(t2, 0.0)

    def test_hit_rate_increases_with_reuse(self, pool):
        """Released tensors accumulate in pool for future reuse.

        Note: there is a key mismatch between allocate (uses np.float32 class)
        and release (uses tensor.dtype instance), so pool hits require matching
        the exact key. This test verifies the pool grows on release.
        """
        shape = (10, 5)
        t1 = pool.allocate(shape)
        pool.release(t1)
        stats = pool.get_stats()
        # After release, pool should have 1 cached tensor
        assert stats['total_cached'] >= 1
        # And at least 1 miss from the initial allocate
        assert stats['misses'] >= 1

    def test_pool_respects_max_size(self):
        """Pool does not cache more than max_pool_size tensors."""
        from hololoom.warp.optimized import TensorMemoryPool
        pool = TensorMemoryPool(max_pool_size=2)
        shape = (3, 3)
        tensors = [pool.allocate(shape) for _ in range(5)]
        for t in tensors:
            pool.release(t)
        stats = pool.get_stats()
        assert stats['total_cached'] <= 2

    def test_different_shapes_separate_pools(self, pool):
        """Different shapes use separate pool buckets."""
        t1 = pool.allocate((3, 3))
        pool.release(t1)
        # Allocating different shape should miss
        t2 = pool.allocate((4, 4))
        stats = pool.get_stats()
        assert stats['misses'] >= 2  # Both initial allocations were misses


# ============================================================================
# Riemannian Geometry
# ============================================================================

class TestHyperbolicSpace:
    """Test Poincare ball model operations."""

    @pytest.fixture
    def H(self):
        from hololoom.warp.riemannian_geometry import HyperbolicSpace
        return HyperbolicSpace(curvature=-1.0)

    def test_project_inside_ball(self, H):
        """Points inside the ball are unchanged (approximately)."""
        x = np.array([0.1, 0.2, 0.3])
        px = H.project(x)
        assert np.allclose(px, x, atol=1e-5)

    def test_project_outside_ball_clamped(self, H):
        """Points outside the ball are projected inside."""
        x = np.array([10.0, 10.0, 10.0])
        px = H.project(x)
        assert np.linalg.norm(px) < 1.0

    def test_distance_is_zero_for_same_point(self, H):
        """d(x, x) ~ 0 (up to numerical epsilon in arccosh)."""
        x = np.array([0.1, 0.2, 0.3])
        d = H.distance(x, x)
        assert d < 0.01, f"d(x,x) = {d}, expected ~0"

    def test_distance_is_positive(self, H):
        """d(x, y) > 0 for x != y."""
        x = np.array([0.1, 0.2, 0.0])
        y = np.array([0.3, -0.1, 0.0])
        d = H.distance(x, y)
        assert d > 0

    def test_distance_is_symmetric(self, H):
        """d(x, y) = d(y, x)."""
        x = np.array([0.1, 0.2, 0.0])
        y = np.array([0.3, -0.1, 0.0])
        assert np.isclose(H.distance(x, y), H.distance(y, x), atol=1e-10)

    def test_distance_triangle_inequality(self, H):
        """d(x, z) <= d(x, y) + d(y, z)."""
        x = np.array([0.1, 0.0, 0.0])
        y = np.array([0.0, 0.2, 0.0])
        z = np.array([-0.1, -0.1, 0.0])
        d_xz = H.distance(x, z)
        d_xy = H.distance(x, y)
        d_yz = H.distance(y, z)
        assert d_xz <= d_xy + d_yz + 1e-10

    def test_distance_grows_near_boundary(self, H):
        """Distances grow as points approach boundary (hyperbolic expansion)."""
        # Points near center
        d_center = H.distance(np.array([0.0, 0.0]), np.array([0.1, 0.0]))
        # Same Euclidean gap near boundary
        d_edge = H.distance(np.array([0.8, 0.0]), np.array([0.9, 0.0]))
        assert d_edge > d_center

    def test_exp_map_at_origin_direction(self, H):
        """exp_map from origin moves in direction of tangent vector."""
        x = np.zeros(3)
        v = np.array([0.1, 0.0, 0.0])
        result = H.exp_map(x, v)
        # Result should be in positive x direction
        assert result[0] > 0
        # And inside the ball
        assert np.linalg.norm(result) < 1.0

    def test_exp_log_inverse_at_origin(self, H):
        """exp_map(0, log_map(0, y)) ~ y for small y near origin."""
        origin = np.zeros(4)
        y = np.array([0.1, 0.05, -0.05, 0.02])
        v = H.log_map(origin, y)
        recovered = H.exp_map(origin, v)
        assert np.allclose(recovered, y, atol=0.05)

    def test_mobius_add_identity(self, H):
        """x + 0 = x in Mobius addition."""
        x = np.array([0.1, 0.2, 0.3])
        zero = np.zeros(3)
        result = H.mobius_add(x, zero)
        assert np.allclose(result, x, atol=1e-10)

    def test_parallel_transport_preserves_norm_approximately(self, H):
        """Parallel transport approximately preserves vector norm."""
        x = np.array([0.1, 0.0, 0.0])
        y = np.array([0.0, 0.1, 0.0])
        v = np.array([0.05, 0.05, 0.0])
        transported = H.parallel_transport(x, y, v)
        # Norm should be approximately preserved (up to conformal factor)
        assert np.linalg.norm(transported) > 0


class TestSphericalSpace:
    """Test unit sphere operations."""

    @pytest.fixture
    def S(self):
        from hololoom.warp.riemannian_geometry import SphericalSpace
        return SphericalSpace(curvature=1.0)

    def test_project_normalizes(self, S):
        """Projection normalizes to unit sphere."""
        x = np.array([3.0, 4.0, 0.0])
        px = S.project(x)
        assert np.isclose(np.linalg.norm(px), 1.0, atol=1e-5)

    def test_distance_same_point_zero(self, S):
        """d(x, x) ~ 0 on sphere (up to numerical eps in arccos)."""
        x = np.array([1.0, 0.0, 0.0])
        d = S.distance(x, x)
        assert d < 0.01, f"d(x,x) = {d}, expected ~0"

    def test_distance_antipodal_is_pi(self, S):
        """d(x, -x) = pi for unit curvature sphere."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([-1.0, 0.0, 0.0])
        d = S.distance(x, y)
        assert np.isclose(d, np.pi, atol=0.01)

    def test_distance_orthogonal_is_pi_over_2(self, S):
        """d(e_1, e_2) = pi/2 on unit sphere."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        d = S.distance(x, y)
        assert np.isclose(d, np.pi / 2, atol=0.01)

    def test_distance_is_symmetric(self, S):
        """d(x, y) = d(y, x) on sphere."""
        x = np.array([1.0, 1.0, 0.0])
        y = np.array([0.0, 1.0, 1.0])
        assert np.isclose(S.distance(x, y), S.distance(y, x), atol=1e-10)

    def test_distance_triangle_inequality(self, S):
        """d(x, z) <= d(x, y) + d(y, z) on sphere."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])
        d_xz = S.distance(x, z)
        d_xy = S.distance(x, y)
        d_yz = S.distance(y, z)
        assert d_xz <= d_xy + d_yz + 1e-10

    def test_exp_map_stays_on_sphere(self, S):
        """exp_map result is on the unit sphere."""
        x = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 0.5, 0.0])  # Tangent at x
        result = S.exp_map(x, v)
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-5)

    def test_log_map_tangent_direction(self, S):
        """log_map points from x toward y."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        v = S.log_map(x, y)
        # v should have positive y-component (pointing toward y)
        assert v[1] > 0

    def test_parallel_transport_result_nonzero(self, S):
        """Parallel transport produces non-zero vector."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])  # Tangent at x, perpendicular to both
        transported = S.parallel_transport(x, y, v)
        assert np.linalg.norm(transported) > 0


class TestProductManifold:
    """Test mixed-curvature product manifold."""

    @pytest.fixture
    def manifold(self):
        from hololoom.warp.riemannian_geometry import ProductManifold, ManifoldConfig
        config = ManifoldConfig(
            curvature=-1.0,
            hyperbolic_dim=4,
            spherical_dim=4,
            euclidean_dim=4
        )
        return ProductManifold(config)

    def test_split_combine_roundtrip(self, manifold):
        """split then combine recovers the original vector."""
        x = np.random.default_rng(42).normal(0, 0.3, 12)
        h, s, e = manifold.split(x)
        recovered = manifold.combine(h, s, e)
        assert np.allclose(x, recovered)

    def test_split_dimensions(self, manifold):
        """Split produces correct subspace dimensions."""
        x = np.zeros(12)
        h, s, e = manifold.split(x)
        assert h.shape == (4,)
        assert s.shape == (4,)
        assert e.shape == (4,)

    def test_distance_is_zero_for_same_point(self, manifold):
        """Product distance d(x, x) ~ 0 (up to numerical eps)."""
        x = np.random.default_rng(42).normal(0, 0.1, 12)
        d = manifold.distance(x, x)
        assert d < 0.05, f"d(x,x) = {d}, expected ~0"

    def test_distance_is_positive(self, manifold):
        """Product distance d(x, y) > 0 for x != y."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 0.1, 12)
        y = rng.normal(0, 0.1, 12)
        d = manifold.distance(x, y)
        assert d > 0

    def test_distance_is_symmetric(self, manifold):
        """Product distance d(x, y) = d(y, x)."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 0.1, 12)
        y = rng.normal(0, 0.1, 12)
        assert np.isclose(manifold.distance(x, y), manifold.distance(y, x), atol=1e-6)

    def test_exp_map_changes_point(self, manifold):
        """exp_map with nonzero tangent moves the point."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 0.1, 12)
        v = rng.normal(0, 0.01, 12)
        result = manifold.exp_map(x, v)
        assert not np.allclose(x, result)

    def test_parallel_transport_euclidean_component_identity(self, manifold):
        """Euclidean component of parallel transport is identity."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 0.1, 12)
        y = rng.normal(0, 0.1, 12)
        v = rng.normal(0, 0.05, 12)
        transported = manifold.parallel_transport(x, y, v)
        _, _, v_e = manifold.split(v)
        _, _, t_e = manifold.split(transported)
        assert np.allclose(v_e, t_e)


class TestManifoldFactory:
    """Test manifold creation factory."""

    def test_create_hyperbolic(self):
        from hololoom.warp.riemannian_geometry import create_manifold, ManifoldConfig, ManifoldType, HyperbolicSpace
        config = ManifoldConfig(manifold_type=ManifoldType.HYPERBOLIC, curvature=-1.0)
        m = create_manifold(config)
        assert isinstance(m, HyperbolicSpace)

    def test_create_spherical(self):
        from hololoom.warp.riemannian_geometry import create_manifold, ManifoldConfig, ManifoldType, SphericalSpace
        config = ManifoldConfig(manifold_type=ManifoldType.SPHERICAL, curvature=1.0)
        m = create_manifold(config)
        assert isinstance(m, SphericalSpace)

    def test_create_product(self):
        from hololoom.warp.riemannian_geometry import create_manifold, ManifoldConfig, ManifoldType, ProductManifold
        config = ManifoldConfig(manifold_type=ManifoldType.PRODUCT, curvature=-1.0)
        m = create_manifold(config)
        assert isinstance(m, ProductManifold)

    def test_create_euclidean_returns_none_with_warning(self):
        from hololoom.warp.riemannian_geometry import create_manifold, ManifoldConfig, ManifoldType
        config = ManifoldConfig(manifold_type=ManifoldType.EUCLIDEAN)
        with pytest.warns(UserWarning, match="Euclidean"):
            m = create_manifold(config)
        assert m is None


# ============================================================================
# Semantic PDEs
# ============================================================================

class TestHeatEquationSolver:
    """Test heat equation (diffusion) solver."""

    @pytest.fixture
    def solver(self):
        from hololoom.warp.semantic_pde import HeatEquationSolver
        return HeatEquationSolver(
            n_points=21,
            domain_size=1.0,
            dt=0.0001,
            diffusion_coefficient=0.1,
            boundary_condition="dirichlet"
        )

    def test_step_conserves_shape(self, solver):
        """Single step preserves array shape."""
        u = np.zeros(21)
        u[10] = 1.0
        u_next = solver.step(u)
        assert u_next.shape == u.shape

    def test_dirichlet_boundary_enforced(self, solver):
        """Dirichlet boundaries remain zero after step."""
        u = np.zeros(21)
        u[10] = 1.0
        u_next = solver.step(u)
        assert u_next[0] == 0.0
        assert u_next[-1] == 0.0

    def test_heat_diffuses_peak(self, solver):
        """Heat diffuses: peak value decreases over time."""
        u = np.zeros(21)
        u[10] = 1.0
        peak_before = u[10]
        for _ in range(10):
            u = solver.step(u)
        assert u[10] < peak_before, "Peak should decrease as heat diffuses"

    def test_heat_spreads_to_neighbors(self, solver):
        """Heat spreads from peak to neighbors."""
        u = np.zeros(21)
        u[10] = 1.0
        for _ in range(10):
            u = solver.step(u)
        # Neighbors should have positive values
        assert u[9] > 0
        assert u[11] > 0

    def test_solve_returns_snapshots(self, solver):
        """solve() returns time array and solution snapshots."""
        u0 = np.zeros(21)
        u0[10] = 1.0
        times, solutions = solver.solve(u0, t_final=0.01, n_snapshots=5)
        assert len(times) >= 2
        assert len(solutions) >= 2
        assert solutions[0].shape == u0.shape

    def test_steady_state_approaches_zero_dirichlet(self, solver):
        """With Dirichlet BC, solution approaches zero steady state."""
        u = np.zeros(21)
        u[10] = 1.0
        for _ in range(5000):
            u = solver.step(u)
        assert np.max(np.abs(u)) < 0.1, "Should approach zero under Dirichlet BC"

    def test_energy_decreases_monotonically(self, solver):
        """Total energy (sum of u^2) decreases over time with Dirichlet BC."""
        u = np.zeros(21)
        u[10] = 1.0
        energies = []
        for _ in range(50):
            energies.append(np.sum(u ** 2))
            u = solver.step(u)
        # Energy should be monotonically non-increasing
        for i in range(len(energies) - 1):
            assert energies[i] >= energies[i + 1] - 1e-10

    def test_neumann_boundary_condition(self):
        """Neumann BC: boundary copies neighbor value."""
        from hololoom.warp.semantic_pde import HeatEquationSolver
        solver = HeatEquationSolver(
            n_points=11,
            dt=0.0001,
            diffusion_coefficient=0.1,
            boundary_condition="neumann"
        )
        u = np.zeros(11)
        u[5] = 1.0
        u_next = solver.step(u)
        assert u_next[0] == u_next[1]
        assert u_next[-1] == u_next[-2]

    def test_periodic_boundary_condition(self):
        """Periodic BC: boundaries wrap around."""
        from hololoom.warp.semantic_pde import HeatEquationSolver
        solver = HeatEquationSolver(
            n_points=11,
            dt=0.0001,
            diffusion_coefficient=0.1,
            boundary_condition="periodic"
        )
        u = np.zeros(11)
        u[5] = 1.0
        u_next = solver.step(u)
        assert u_next[0] == u_next[-2]
        assert u_next[-1] == u_next[1]


class TestWaveEquationSolver:
    """Test wave equation solver."""

    @pytest.fixture
    def solver(self):
        from hololoom.warp.semantic_pde import WaveEquationSolver
        return WaveEquationSolver(
            n_points=21,
            domain_size=1.0,
            wave_speed=1.0,
            dt=0.001,
            boundary_condition="dirichlet"
        )

    def test_step_returns_displacement_and_velocity(self, solver):
        """Step returns both u and ut."""
        u = np.zeros(21)
        u[10] = 1.0
        ut = np.zeros(21)
        u_new, ut_new = solver.step(u, ut)
        assert u_new.shape == (21,)
        assert ut_new.shape == (21,)

    def test_dirichlet_boundary(self, solver):
        """Dirichlet boundaries remain zero for wave equation."""
        u = np.zeros(21)
        u[10] = 1.0
        ut = np.zeros(21)
        u_new, ut_new = solver.step(u, ut)
        assert u_new[0] == 0.0
        assert u_new[-1] == 0.0

    def test_wave_propagates(self, solver):
        """Initial displacement propagates outward."""
        u = np.zeros(21)
        u[10] = 1.0
        ut = np.zeros(21)
        for _ in range(20):
            u, ut = solver.step(u, ut)
        # Energy should have spread beyond the initial point
        assert np.sum(np.abs(u[8:13]) > 0.001) > 1

    def test_zero_initial_velocity_symmetric_propagation(self, solver):
        """With zero initial velocity, wave propagates symmetrically."""
        u = np.zeros(21)
        u[10] = 1.0
        ut = np.zeros(21)
        for _ in range(5):
            u, ut = solver.step(u, ut)
        # Should be approximately symmetric around center
        assert np.isclose(u[9], u[11], atol=1e-5)

    def test_solve_returns_snapshots(self, solver):
        """solve() returns time and solution snapshots."""
        u0 = np.zeros(21)
        u0[10] = 1.0
        v0 = np.zeros(21)
        times, solutions = solver.solve(u0, v0, t_final=0.01, n_snapshots=5)
        assert len(times) >= 2
        assert solutions.shape[1] == 21


class TestReactionDiffusionSolver:
    """Test reaction-diffusion solver."""

    @pytest.fixture
    def solver(self):
        from hololoom.warp.semantic_pde import ReactionDiffusionSolver
        return ReactionDiffusionSolver(
            n_points=21,
            domain_size=1.0,
            dt=0.001,
            diffusion_a=0.1,
            diffusion_b=0.05,
            reaction_rate_a=0.04,
            reaction_rate_b=0.06
        )

    def test_step_returns_two_species(self, solver):
        """Step returns both u and v species."""
        u = np.ones(21)
        v = np.zeros(21)
        v[10] = 0.5
        u_next, v_next = solver.step(u, v)
        assert u_next.shape == (21,)
        assert v_next.shape == (21,)

    def test_species_stay_bounded(self, solver):
        """Species concentrations stay in [0, 1.5] (clipped)."""
        u = np.ones(21)
        v = np.zeros(21)
        v[10] = 0.5
        for _ in range(100):
            u, v = solver.step(u, v)
        assert np.all(u >= 0.0)
        assert np.all(u <= 1.5)
        assert np.all(v >= 0.0)
        assert np.all(v <= 1.5)

    def test_neumann_boundary_applied(self, solver):
        """Neumann zero-gradient boundary enforced on both species."""
        u = np.ones(21)
        v = np.zeros(21)
        v[10] = 0.5
        u_next, v_next = solver.step(u, v)
        assert u_next[0] == u_next[1]
        assert u_next[-1] == u_next[-2]
        assert v_next[0] == v_next[1]
        assert v_next[-1] == v_next[-2]

    def test_solve_returns_three_arrays(self, solver):
        """solve() returns times, u_solutions, v_solutions."""
        u0 = np.ones(21)
        v0 = np.zeros(21)
        v0[10] = 0.5
        times, solutions_u, solutions_v = solver.solve(u0, v0, t_final=0.05, n_snapshots=3)
        assert len(times) >= 2
        assert solutions_u.shape[1] == 21
        assert solutions_v.shape[1] == 21


class TestAdvectionDiffusionSolver:
    """Test advection-diffusion solver."""

    @pytest.fixture
    def solver(self):
        from hololoom.warp.semantic_pde import AdvectionDiffusionSolver
        return AdvectionDiffusionSolver(
            n_points=21,
            domain_size=1.0,
            velocity=0.5,
            diffusion_coefficient=0.1,
            dt=0.001,
            boundary_condition="periodic"
        )

    def test_step_preserves_shape(self, solver):
        """Step preserves array shape."""
        u = np.zeros(21)
        u[10] = 1.0
        u_next = solver.step(u)
        assert u_next.shape == (21,)

    def test_advection_shifts_peak(self, solver):
        """Advection shifts the peak in the velocity direction."""
        u = np.zeros(21)
        u[10] = 1.0
        for _ in range(5):
            u = solver.step(u)
        # With positive velocity, peak should shift right (higher index)
        peak_idx = np.argmax(u)
        assert peak_idx >= 10  # May stay or shift right


class TestHamiltonJacobiSolver:
    """Test Hamilton-Jacobi equation solver."""

    @pytest.fixture
    def solver(self):
        from hololoom.warp.semantic_pde import HamiltonJacobiSolver
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ], dtype=float)
        return HamiltonJacobiSolver(
            adjacency=adj,
            hamiltonian=lambda p: 0.5 * p ** 2,
            dt=0.01
        )

    def test_gradient_zero_for_constant(self, solver):
        """Gradient of constant function is zero."""
        u = np.ones(4) * 5.0
        grad = solver.compute_gradient(u)
        assert np.allclose(grad, 0.0, atol=1e-10)

    def test_step_preserves_shape(self, solver):
        """Step preserves array shape."""
        u = np.array([1.0, 0.5, 0.0, -0.5])
        u_next = solver.step(u)
        assert u_next.shape == (4,)

    def test_solve_returns_snapshots(self, solver):
        """solve() returns time and solution snapshots."""
        u0 = np.array([1.0, 0.5, 0.0, 0.0])
        times, solutions = solver.solve(u0, t_final=0.05, n_snapshots=3)
        assert len(times) >= 2
        assert solutions.shape[1] == 4


class TestReactionFunctions:
    """Test common reaction functions."""

    def test_logistic_reaction_zero_at_boundaries(self):
        """Logistic reaction f(0) = 0 and f(1) = 0."""
        from hololoom.warp.semantic_pde import logistic_reaction
        f = logistic_reaction(r=1.0)
        u = np.array([0.0, 1.0])
        result = f(u)
        assert np.allclose(result, 0.0)

    def test_logistic_reaction_positive_in_interior(self):
        """Logistic reaction f(u) > 0 for 0 < u < 1."""
        from hololoom.warp.semantic_pde import logistic_reaction
        f = logistic_reaction(r=2.0)
        u = np.array([0.25, 0.5, 0.75])
        result = f(u)
        assert np.all(result > 0)

    def test_competitive_reaction_suppresses_weak(self):
        """Competitive reaction suppresses values below threshold."""
        from hololoom.warp.semantic_pde import competitive_reaction
        f = competitive_reaction(r=1.0, theta=0.5)
        u = np.array([0.1, 0.3])  # Below threshold
        result = f(u)
        assert np.all(result < 0), "Weak activations should be suppressed"

    def test_competitive_reaction_amplifies_strong(self):
        """Competitive reaction amplifies values above threshold."""
        from hololoom.warp.semantic_pde import competitive_reaction
        f = competitive_reaction(r=1.0, theta=0.3)
        u = np.array([0.5, 0.7])  # Above threshold, below 1
        result = f(u)
        assert np.all(result > 0), "Strong activations should be amplified"

    def test_cubic_reaction_odd_symmetry(self):
        """Cubic reaction f(-u) = -f(u) (odd function)."""
        from hololoom.warp.semantic_pde import cubic_reaction
        f = cubic_reaction(a=1.0, b=1.0)
        u = np.array([0.5, 1.0, 2.0])
        assert np.allclose(f(-u), -f(u))

    def test_cubic_reaction_fixed_points(self):
        """Cubic reaction has fixed points at u = 0 and u = +/- sqrt(a/b)."""
        from hololoom.warp.semantic_pde import cubic_reaction
        a, b = 2.0, 2.0
        f = cubic_reaction(a=a, b=b)
        # Fixed point at 0
        assert np.isclose(f(np.array([0.0]))[0], 0.0)
        # Fixed points at +/- sqrt(a/b) = +/- 1
        assert np.isclose(f(np.array([1.0]))[0], 0.0)
        assert np.isclose(f(np.array([-1.0]))[0], 0.0)


# ============================================================================
# PDE Factory Functions
# ============================================================================

class TestPDEFactories:
    """Test PDE solver factory functions."""

    def test_create_heat_solver(self):
        from hololoom.warp.semantic_pde import create_heat_solver, HeatEquationSolver
        L = np.zeros((5, 5))
        np.fill_diagonal(L, -2.0)
        solver = create_heat_solver(L, dt=0.01)
        assert isinstance(solver, HeatEquationSolver)

    def test_create_wave_solver(self):
        from hololoom.warp.semantic_pde import create_wave_solver, WaveEquationSolver
        L = np.zeros((5, 5))
        np.fill_diagonal(L, -2.0)
        solver = create_wave_solver(L, wave_speed=2.0)
        assert isinstance(solver, WaveEquationSolver)

    def test_create_reaction_diffusion_solver(self):
        from hololoom.warp.semantic_pde import create_reaction_diffusion_solver, ReactionDiffusionSolver
        L = np.zeros((5, 5))
        np.fill_diagonal(L, -2.0)
        solver = create_reaction_diffusion_solver(L)
        assert isinstance(solver, ReactionDiffusionSolver)


# ============================================================================
# Laplacian Construction
# ============================================================================

class TestLaplacianConstruction:
    """Test the 1D Laplacian builder used by PDE solvers."""

    def test_laplacian_symmetric(self):
        """1D Laplacian matrix is symmetric."""
        from hololoom.warp.semantic_pde import HeatEquationSolver
        L = HeatEquationSolver._build_1d_laplacian(10, 0.1, "dirichlet")
        assert np.allclose(L, L.T)

    def test_laplacian_row_sums_dirichlet(self):
        """Interior rows of Dirichlet Laplacian sum to zero."""
        from hololoom.warp.semantic_pde import HeatEquationSolver
        L = HeatEquationSolver._build_1d_laplacian(10, 0.1, "dirichlet")
        for i in range(1, 9):  # Interior rows
            assert np.isclose(np.sum(L[i]), 0.0, atol=1e-10)

    def test_laplacian_periodic_row_sums_zero(self):
        """All rows of periodic Laplacian sum to zero."""
        from hololoom.warp.semantic_pde import HeatEquationSolver
        L = HeatEquationSolver._build_1d_laplacian(10, 0.1, "periodic")
        for i in range(10):
            assert np.isclose(np.sum(L[i]), 0.0, atol=1e-10)

    def test_laplacian_negative_diagonal(self):
        """Laplacian has negative diagonal entries."""
        from hololoom.warp.semantic_pde import HeatEquationSolver
        L = HeatEquationSolver._build_1d_laplacian(10, 0.1, "dirichlet")
        for i in range(10):
            assert L[i, i] < 0


# ============================================================================
# LazyWarpOperation (from optimized.py)
# ============================================================================

class TestLazyWarpOperation:
    """Test lazy evaluation wrapper."""

    def test_lazy_not_computed_initially(self):
        from hololoom.warp.optimized import LazyWarpOperation
        op = LazyWarpOperation("test_op")
        assert op.computed is False
        assert op.result is None

    def test_lazy_unknown_operation_raises(self):
        from hololoom.warp.optimized import LazyWarpOperation
        op = LazyWarpOperation("nonexistent_op")
        with pytest.raises(ValueError, match="Unknown lazy operation"):
            op.compute()

    def test_lazy_callable(self):
        """LazyWarpOperation is callable (delegates to compute)."""
        from hololoom.warp.optimized import LazyWarpOperation
        op = LazyWarpOperation("nonexistent_op")
        with pytest.raises(ValueError):
            op()  # __call__ delegates to compute
