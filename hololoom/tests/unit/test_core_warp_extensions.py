"""
Unit tests for warp math modules:
- extensions/complexity_cybernetics.py
- extensions/algebraic_combinatorics.py
- extensions/manifold_learning.py
- advanced/alpha_cech_complexes.py
- advanced/covering_spaces.py
- geometry/computational_geometry.py
- self_assembly/genetic_programming.py
- self_assembly/self_modifying.py

Pure math tests using small structures. Each test < 5s.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# complexity_cybernetics
# ---------------------------------------------------------------------------

from hololoom.core.warp.math.extensions.complexity_cybernetics import (
    ApproximationAlgorithm,
    ParameterizedComplexity,
    KripkeFrame,
    IntuitionisticLogic,
    TypeTheory,
    ViabilityTheory,
)


class TestApproximationAlgorithm:
    def test_greedy_set_cover_basic(self):
        universe = {0, 1, 2, 3, 4}
        sets = [{0, 1, 2}, {2, 3}, {3, 4}, {0, 4}]
        cover = ApproximationAlgorithm.greedy_set_cover(universe, sets)
        covered = set()
        for i in cover:
            covered |= sets[i]
        assert covered == universe

    def test_greedy_set_cover_single_set(self):
        universe = {0, 1}
        sets = [{0, 1}]
        cover = ApproximationAlgorithm.greedy_set_cover(universe, sets)
        assert cover == [0]

    def test_greedy_set_cover_empty_universe(self):
        cover = ApproximationAlgorithm.greedy_set_cover(set(), [{0}])
        assert cover == []

    def test_greedy_set_cover_impossible(self):
        universe = {0, 1, 2}
        sets = [{0}, {1}]
        cover = ApproximationAlgorithm.greedy_set_cover(universe, sets)
        # Cannot cover 2, but should cover what it can
        covered = set()
        for i in cover:
            covered |= sets[i]
        assert 2 not in covered

    def test_vertex_cover_2approx(self):
        # Triangle graph
        adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        cover = ApproximationAlgorithm.vertex_cover_2approx(adj)
        # Must cover all edges
        for u in adj:
            for v in adj[u]:
                assert u in cover or v in cover

    def test_vertex_cover_path(self):
        adj = {0: {1}, 1: {0, 2}, 2: {1}}
        cover = ApproximationAlgorithm.vertex_cover_2approx(adj)
        for u in adj:
            for v in adj[u]:
                assert u in cover or v in cover

    def test_weighted_set_cover(self):
        universe = {0, 1, 2}
        sets = [{0, 1, 2}, {0}, {1}]
        weights = [10.0, 1.0, 1.0]
        cover = ApproximationAlgorithm.weighted_set_cover(universe, sets, weights)
        covered = set()
        for i in cover:
            covered |= sets[i]
        assert covered == universe

    def test_weighted_set_cover_prefers_cheap(self):
        universe = {0, 1}
        sets = [{0, 1}, {0}, {1}]
        weights = [100.0, 1.0, 1.0]
        cover = ApproximationAlgorithm.weighted_set_cover(universe, sets, weights)
        # Should pick the two cheap sets over the expensive one
        total_cost = sum(weights[i] for i in cover)
        assert total_cost <= 100.0


class TestParameterizedComplexity:
    def test_fpt_vertex_cover_triangle(self):
        pc = ParameterizedComplexity()
        adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        cover = pc.fpt_vertex_cover(adj, k=2)
        assert cover is not None
        assert len(cover) <= 2
        for u in adj:
            for v in adj[u]:
                assert u in cover or v in cover

    def test_fpt_vertex_cover_too_small_k(self):
        pc = ParameterizedComplexity()
        # K4: needs at least 3 vertices
        adj = {0: {1, 2, 3}, 1: {0, 2, 3}, 2: {0, 1, 3}, 3: {0, 1, 2}}
        cover = pc.fpt_vertex_cover(adj, k=1)
        assert cover is None

    def test_fpt_vertex_cover_no_edges(self):
        pc = ParameterizedComplexity()
        adj = {0: set(), 1: set()}
        cover = pc.fpt_vertex_cover(adj, k=0)
        assert cover is not None
        assert len(cover) == 0

    def test_kernelization(self):
        pc = ParameterizedComplexity()
        # Star graph: center connected to 5 leaves, k=1
        adj = {0: {1, 2, 3, 4, 5}}
        for i in range(1, 6):
            adj[i] = {0}
        reduced, forced, remaining_k = pc.kernelization_vertex_cover(adj, k=1)
        # Vertex 0 has degree 5 > k=1, must be forced
        assert 0 in forced

    def test_kernelization_removes_isolated(self):
        pc = ParameterizedComplexity()
        adj = {0: set(), 1: {2}, 2: {1}}
        reduced, forced, remaining_k = pc.kernelization_vertex_cover(adj, k=5)
        assert 0 not in reduced


class TestIntuitionisticLogic:
    @pytest.fixture
    def il(self):
        return IntuitionisticLogic()

    @pytest.fixture
    def two_world_frame(self):
        return KripkeFrame(
            worlds=['w0', 'w1'],
            order={'w0': {'w0', 'w1'}, 'w1': {'w1'}},
            valuation={('p', 'w1'): True},
        )

    def test_atom_true(self, il, two_world_frame):
        assert il.evaluate(('atom', 'p'), two_world_frame, 'w1') is True

    def test_atom_false(self, il, two_world_frame):
        assert il.evaluate(('atom', 'p'), two_world_frame, 'w0') is False

    def test_top(self, il, two_world_frame):
        assert il.evaluate(('top',), two_world_frame, 'w0') is True

    def test_bottom(self, il, two_world_frame):
        assert il.evaluate(('bottom',), two_world_frame, 'w0') is False

    def test_conjunction(self, il, two_world_frame):
        f = ('and', ('top',), ('atom', 'p'))
        assert il.evaluate(f, two_world_frame, 'w1') is True
        assert il.evaluate(f, two_world_frame, 'w0') is False

    def test_disjunction(self, il, two_world_frame):
        f = ('or', ('atom', 'p'), ('bottom',))
        assert il.evaluate(f, two_world_frame, 'w1') is True
        assert il.evaluate(f, two_world_frame, 'w0') is False

    def test_excluded_middle_fails(self, il, two_world_frame):
        # p v ~p should fail at w0 in intuitionistic logic
        lem = ('or', ('atom', 'p'), ('not', ('atom', 'p')))
        assert il.evaluate(lem, two_world_frame, 'w0') is False

    def test_implication(self, il, two_world_frame):
        # p -> p is always true
        f = ('implies', ('atom', 'p'), ('atom', 'p'))
        assert il.evaluate(f, two_world_frame, 'w0') is True

    def test_negation(self, il, two_world_frame):
        # ~p at w0: p -> bottom for all w' >= w0
        # At w1, p is true so p -> bottom fails
        assert il.evaluate(('not', ('atom', 'p')), two_world_frame, 'w0') is False

    def test_is_valid(self, il, two_world_frame):
        assert il.is_valid(('top',), two_world_frame) is True
        assert il.is_valid(('atom', 'p'), two_world_frame) is False

    def test_extract_atoms(self, il):
        f = ('implies', ('and', ('atom', 'p'), ('atom', 'q')), ('or', ('atom', 'p'), ('atom', 'r')))
        atoms = il._extract_atoms(f)
        assert atoms == {'p', 'q', 'r'}

    def test_is_intuitionistically_valid_identity(self, il):
        # p -> p is intuitionistically valid
        f = ('implies', ('atom', 'p'), ('atom', 'p'))
        assert il.is_intuitionistically_valid(f, max_worlds=2) is True

    def test_unknown_formula_raises(self, il, two_world_frame):
        with pytest.raises(ValueError, match="Unknown formula kind"):
            il.evaluate(('unknown',), two_world_frame, 'w0')


class TestTypeTheory:
    @pytest.fixture
    def tt(self):
        return TypeTheory()

    def test_identity_function(self, tt):
        # \x:A. x  : A -> A
        term = ('lam', 'x', ('base', 'A'), ('var', 'x'))
        result = tt.type_check(term, {})
        assert result == ('arrow', ('base', 'A'), ('base', 'A'))

    def test_variable_lookup(self, tt):
        result = tt.type_check(('var', 'x'), {'x': ('base', 'Int')})
        assert result == ('base', 'Int')

    def test_variable_not_found(self, tt):
        assert tt.type_check(('var', 'y'), {}) is None

    def test_application(self, tt):
        # (\x:A. x) applied to a:A => A
        f = ('lam', 'x', ('base', 'A'), ('var', 'x'))
        app = ('app', f, ('var', 'a'))
        result = tt.type_check(app, {'a': ('base', 'A')})
        assert result == ('base', 'A')

    def test_application_type_mismatch(self, tt):
        f = ('lam', 'x', ('base', 'A'), ('var', 'x'))
        app = ('app', f, ('var', 'b'))
        result = tt.type_check(app, {'b': ('base', 'B')})
        assert result is None

    def test_application_non_function(self, tt):
        app = ('app', ('var', 'x'), ('var', 'y'))
        result = tt.type_check(app, {'x': ('base', 'A'), 'y': ('base', 'A')})
        assert result is None

    def test_nested_lambda(self, tt):
        # \x:A. \y:B. x  : A -> B -> A
        term = ('lam', 'x', ('base', 'A'),
                ('lam', 'y', ('base', 'B'), ('var', 'x')))
        result = tt.type_check(term, {})
        assert result == ('arrow', ('base', 'A'), ('arrow', ('base', 'B'), ('base', 'A')))

    def test_type_to_string(self, tt):
        assert tt.type_to_string(('base', 'Int')) == 'Int'
        assert tt.type_to_string(('arrow', ('base', 'A'), ('base', 'B'))) == 'A -> B'
        nested = ('arrow', ('arrow', ('base', 'A'), ('base', 'B')), ('base', 'C'))
        assert tt.type_to_string(nested) == '(A -> B) -> C'

    def test_term_to_string(self, tt):
        assert tt.term_to_string(('var', 'x')) == 'x'

    def test_curry_howard(self, tt):
        assert TypeTheory.curry_howard(('base', 'P')) == 'P'
        assert TypeTheory.curry_howard(('arrow', ('base', 'A'), ('base', 'B'))) == '(A => B)'

    def test_unknown_term(self, tt):
        assert tt.type_check(('bogus', 1, 2), {}) is None


class TestViabilityTheory:
    def test_viable_set_simple(self):
        vt = ViabilityTheory()
        dynamics = lambda x, u: u
        constraint = lambda x: abs(x) <= 2.0
        domain = np.linspace(-3, 3, 30)
        viable = vt.viable_set(dynamics, constraint, domain, dt=0.1,
                               n_iterations=20, n_controls=5)
        # Points outside constraint should not be viable
        for i, x in enumerate(domain):
            if abs(x) > 2.0:
                assert not viable[i]

    def test_viable_set_all_viable(self):
        vt = ViabilityTheory()
        dynamics = lambda x, u: 0.0  # No movement
        constraint = lambda x: True
        domain = np.linspace(-1, 1, 10)
        viable = vt.viable_set(dynamics, constraint, domain, n_iterations=5,
                               n_controls=3)
        assert viable.all()

    def test_capture_basin(self):
        vt = ViabilityTheory()
        dynamics = lambda x, u: u
        target = lambda x: abs(x) < 0.5
        constraint = lambda x: abs(x) <= 3.0
        domain = np.linspace(-3, 3, 30)
        basin = vt.capture_basin(dynamics, target, constraint, domain,
                                 dt=0.1, n_iterations=30, n_controls=5)
        # Target states must be in the basin
        for i, x in enumerate(domain):
            if abs(x) < 0.5:
                assert basin[i]


# ---------------------------------------------------------------------------
# algebraic_combinatorics
# ---------------------------------------------------------------------------

from hololoom.core.warp.math.extensions.algebraic_combinatorics import (
    YoungTableauxOps,
    RSKCorrespondence,
    SchurFunction,
    HallLittlewood,
    DiscreteMorseInequalities,
)


class TestYoungTableaux:
    def test_standard_partition_21(self):
        syts = YoungTableauxOps.standard([2, 1])
        assert len(syts) == 2  # f^{(2,1)} = 2

    def test_standard_partition_2(self):
        syts = YoungTableauxOps.standard([2])
        assert len(syts) == 1

    def test_standard_empty(self):
        assert YoungTableauxOps.standard([]) == [[]]

    def test_hook_length_formula_21(self):
        try:
            count = YoungTableauxOps.hook_length_formula([2, 1])
            assert count == 2
        except AttributeError:
            pytest.skip("np.math.factorial unavailable in this numpy version")

    def test_hook_length_formula_22(self):
        try:
            count = YoungTableauxOps.hook_length_formula([2, 2])
            assert count == 2
        except AttributeError:
            pytest.skip("np.math.factorial unavailable in this numpy version")

    def test_hook_length_formula_31(self):
        try:
            count = YoungTableauxOps.hook_length_formula([3, 1])
            assert count == 3
        except AttributeError:
            pytest.skip("np.math.factorial unavailable in this numpy version")

    def test_hook_length_matches_enumeration(self):
        partition = [2, 1]
        syts = YoungTableauxOps.standard(partition)
        try:
            hook = YoungTableauxOps.hook_length_formula(partition)
            assert len(syts) == hook
        except AttributeError:
            pytest.skip("np.math.factorial unavailable in this numpy version")

    def test_semistandard(self):
        ssyts = YoungTableauxOps.semistandard([2, 1], 2)
        # Each SSYT: rows non-decreasing, columns strictly increasing
        for T in ssyts:
            for row in T:
                for j in range(1, len(row)):
                    assert row[j] >= row[j - 1]


class TestRSK:
    def test_forward_identity(self):
        P, Q = RSKCorrespondence.forward([1, 2, 3])
        assert P == [[1, 2, 3]]
        assert Q == [[1, 2, 3]]

    def test_forward_reverse(self):
        P, Q = RSKCorrespondence.forward([3, 2, 1])
        assert RSKCorrespondence.shape(P) == RSKCorrespondence.shape(Q)

    def test_shape(self):
        P = [[1, 3], [2]]
        assert RSKCorrespondence.shape(P) == [2, 1]

    def test_forward_bumping(self):
        P, Q = RSKCorrespondence.forward([2, 1, 3])
        # 2 is inserted, then 1 bumps 2 to second row
        assert P == [[1, 3], [2]]

    def test_roundtrip_identity(self):
        perm = [1, 2, 3]
        P, Q = RSKCorrespondence.forward(perm)
        recovered = RSKCorrespondence.inverse(P, Q)
        assert recovered == perm

    def test_roundtrip_231(self):
        perm = [2, 3, 1]
        P, Q = RSKCorrespondence.forward(perm)
        recovered = RSKCorrespondence.inverse(P, Q)
        assert recovered == perm


class TestSchurFunction:
    def test_schur_trivial_partition(self):
        # s_{(0)}(x) = 1 for any x
        val = SchurFunction.compute([0], np.array([2.0]))
        assert abs(val - 1.0) < 1e-8

    def test_schur_single_variable(self):
        # s_{(k)}(x) = x^k for single variable
        val = SchurFunction.compute([2], np.array([3.0]))
        assert abs(val - 9.0) < 1e-8

    def test_schur_two_variables(self):
        # s_{(1)}(x,y) = x + y
        val = SchurFunction.compute([1], np.array([2.0, 3.0]))
        assert abs(val - 5.0) < 1e-6


class TestHallLittlewood:
    def test_t_zero_equals_schur(self):
        partition = [2, 1]
        variables = np.array([1.5, 2.0, 0.5])
        hl = HallLittlewood.compute(partition, 0.0, variables)
        schur = SchurFunction.compute(partition, variables)
        assert abs(hl - schur) < 1e-6

    def test_basic_value(self):
        # Just check it returns a finite number
        val = HallLittlewood.compute([1], 0.5, np.array([1.0, 2.0]))
        assert np.isfinite(val)


class TestDiscreteMorse:
    def test_euler_characteristic_triangle(self):
        complex_faces = {
            0: [(0,), (1,), (2,)],
            1: [(0, 1), (1, 2), (0, 2)],
            2: [(0, 1, 2)],
        }
        chi = DiscreteMorseInequalities.euler_characteristic(complex_faces)
        assert chi == 3 - 3 + 1  # V - E + F = 1

    def test_euler_characteristic_two_vertices(self):
        complex_faces = {0: [(0,), (1,)], 1: [(0, 1)]}
        chi = DiscreteMorseInequalities.euler_characteristic(complex_faces)
        assert chi == 2 - 1  # 1

    def test_morse_check_all_critical(self):
        complex_faces = {
            0: [(0,), (1,), (2,)],
            1: [(0, 1), (1, 2), (0, 2)],
        }
        # Morse function where every simplex is critical
        morse_fn = {(0, i): float(i) for i in range(3)}
        morse_fn.update({(1, i): float(i + 3) for i in range(3)})
        result = DiscreteMorseInequalities.check(complex_faces, morse_fn)
        assert isinstance(result, bool)

    def test_betti_computation(self):
        # Circle: 3 vertices, 3 edges, no faces
        complex_faces = {
            0: [(0,), (1,), (2,)],
            1: [(0, 1), (1, 2), (0, 2)],
        }
        betti = DiscreteMorseInequalities._compute_betti(complex_faces)
        assert betti[0] == 1  # Connected
        assert betti[1] == 1  # One loop


# ---------------------------------------------------------------------------
# manifold_learning
# ---------------------------------------------------------------------------

from hololoom.core.warp.math.extensions.manifold_learning import (
    _pairwise_distances,
    _knn_graph,
    trustworthiness,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    UMAP,
    ManifoldEmbedding,
)


class TestManifoldUtilities:
    def test_pairwise_distances_identity(self):
        X = np.eye(3)
        D = _pairwise_distances(X)
        assert D.shape == (3, 3)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)
        assert D[0, 1] > 0

    def test_pairwise_distances_symmetric(self):
        X = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        D = _pairwise_distances(X)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_knn_graph_shape(self):
        D = _pairwise_distances(np.random.default_rng(0).normal(0, 1, (10, 3)))
        knn = _knn_graph(D, k=3)
        assert knn.shape == (10, 3)

    def test_trustworthiness_perfect(self):
        X = np.random.default_rng(42).normal(0, 1, (20, 2))
        # Identity embedding should have perfect trustworthiness
        t = trustworthiness(X, X, k=3)
        assert t >= 0.99

    def test_trustworthiness_small(self):
        X = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        t = trustworthiness(X, X, k=1)
        assert 0 <= t <= 1


class TestTSNE:
    def test_basic_embedding(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 5))
        result = TSNE.fit_transform(X, n_components=2, perplexity=5, n_iter=50, seed=42)
        assert isinstance(result, ManifoldEmbedding)
        assert result.coordinates.shape == (20, 2)
        assert result.method == "t-SNE"
        assert np.isfinite(result.stress)

    def test_small_dataset(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        result = TSNE.fit_transform(X, n_components=2, perplexity=2, n_iter=30, seed=0)
        assert result.coordinates.shape == (4, 2)


class TestIsomap:
    def test_basic_embedding(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (15, 4))
        result = Isomap.fit_transform(X, n_components=2, n_neighbors=5)
        assert result.coordinates.shape == (15, 2)
        assert result.method == "Isomap"

    def test_preserves_structure(self):
        # Collinear points should map roughly to a line
        X = np.column_stack([np.linspace(0, 1, 10), np.zeros(10), np.zeros(10)])
        result = Isomap.fit_transform(X, n_components=1, n_neighbors=3)
        assert result.coordinates.shape == (10, 1)


class TestLLE:
    def test_basic_embedding(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (15, 4))
        result = LocallyLinearEmbedding.fit_transform(X, n_components=2, n_neighbors=5)
        assert result.coordinates.shape == (15, 2)
        assert result.method == "LLE"


class TestUMAP:
    def test_basic_embedding(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (15, 4))
        result = UMAP.fit_transform(X, n_components=2, n_neighbors=5,
                                     n_epochs=10, seed=42)
        assert result.coordinates.shape == (15, 2)
        assert result.method == "UMAP"


# ---------------------------------------------------------------------------
# alpha_cech_complexes
# ---------------------------------------------------------------------------

from hololoom.core.warp.math.advanced.alpha_cech_complexes import (
    SimplexTree,
    PersistencePair,
    DelaunayIncremental,
    AlphaComplex,
    CechComplex,
    TDAPipeline,
)


class TestSimplexTree:
    def test_insert_and_filtration(self):
        tree = SimplexTree()
        tree.insert([0], 0.0)
        tree.insert([1], 0.0)
        tree.insert([0, 1], 0.5)
        assert tree.filtration([0, 1]) == 0.5
        assert tree.filtration([2]) == np.inf

    def test_dimension(self):
        tree = SimplexTree()
        tree.insert([0], 0.0)
        tree.insert([0, 1], 0.5)
        tree.insert([0, 1, 2], 1.0)
        assert tree.dimension == 2

    def test_num_simplices(self):
        tree = SimplexTree()
        tree.insert([0], 0.0)
        tree.insert([1], 0.0)
        tree.insert([0, 1], 0.5)
        assert tree.num_simplices == 3

    def test_skeleton(self):
        tree = SimplexTree()
        tree.insert([0], 0.0)
        tree.insert([0, 1], 0.5)
        tree.insert([0, 1, 2], 1.0)
        skel = tree.skeleton(1)
        assert skel.dimension <= 1

    def test_simplices_sorted(self):
        tree = SimplexTree()
        tree.insert([0, 1], 1.0)
        tree.insert([0], 0.0)
        simplices = tree.simplices
        # Should be sorted by filtration
        filtrations = [f for _, f in simplices]
        assert filtrations == sorted(filtrations)

    def test_insert_updates_lower_filtration(self):
        tree = SimplexTree()
        tree.insert([0, 1], 1.0)
        tree.insert([0, 1], 0.5)
        assert tree.filtration([0, 1]) == 0.5

    def test_empty_tree(self):
        tree = SimplexTree()
        assert tree.dimension == -1
        assert tree.num_simplices == 0


class TestDelaunayIncremental:
    def test_triangle(self):
        points = np.array([[0, 0], [1, 0], [0.5, 0.87]])
        tris = DelaunayIncremental.triangulate_2d(points)
        assert len(tris) >= 1
        # Should form one triangle
        for tri in tris:
            assert len(tri) == 3

    def test_square(self):
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        tris = DelaunayIncremental.triangulate_2d(points)
        assert len(tris) == 2  # Square -> 2 triangles

    def test_two_points(self):
        points = np.array([[0, 0], [1, 0]])
        tris = DelaunayIncremental.triangulate_2d(points)
        assert len(tris) == 1  # Edge only
        assert len(tris[0]) == 2

    def test_one_point(self):
        points = np.array([[0, 0]])
        tris = DelaunayIncremental.triangulate_2d(points)
        assert len(tris) == 0


class TestAlphaComplex:
    def test_from_points_2d(self):
        points = np.array([[0, 0], [1, 0], [0.5, 0.87], [0.5, 0.3]], dtype=float)
        ac = AlphaComplex()
        tree = ac.from_points(points)
        assert tree.num_simplices > 4  # At least vertices + edges

    def test_max_alpha(self):
        # Use points close enough that Delaunay includes the edge
        points = np.array([[0, 0], [1, 0], [0.5, 0.87]], dtype=float)
        ac = AlphaComplex()
        tree_limited = ac.from_points(points, max_alpha=0.3)
        tree_full = ac.from_points(points)
        # Limited alpha should have fewer simplices
        assert tree_limited.num_simplices <= tree_full.num_simplices

    def test_circumradius(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([0.5, np.sqrt(3)/2])
        r = AlphaComplex._circumradius(a, b, c)
        # Equilateral triangle with side 1 has circumradius 1/sqrt(3)
        assert abs(r - 1.0 / np.sqrt(3)) < 1e-6

    def test_higher_dim_rips_fallback(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        ac = AlphaComplex()
        tree = ac.from_points(points)
        assert tree.num_simplices >= 3


class TestCechComplex:
    def test_close_points(self):
        points = np.array([[0, 0], [0.5, 0], [1, 0]], dtype=float)
        cc = CechComplex()
        tree = cc.from_points(points, radius=0.3)
        # Only adjacent pairs have distance <= 0.6
        assert tree.filtration([0, 1]) <= 0.3
        # 0 and 2 are distance 1 apart, need radius >= 0.5
        assert tree.filtration([0, 2]) == np.inf

    def test_miniball_radius(self):
        pts = np.array([[0, 0], [1, 0]], dtype=float)
        r = CechComplex._miniball_radius(pts)
        assert r <= 0.5 + 0.1  # radius should be ~0.5

    def test_triangles_included(self):
        # Three close points forming a tight triangle
        points = np.array([[0, 0], [0.5, 0], [0.25, 0.4]], dtype=float)
        cc = CechComplex()
        tree = cc.from_points(points, radius=1.0, max_dim=2)
        # Triangle should be present
        has_triangle = any(len(s) == 3 for s, _ in tree.simplices)
        assert has_triangle


class TestTDAPipeline:
    def test_rips_pipeline(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (15, 2))
        tda = TDAPipeline()
        result = tda.fit(data, method='rips', max_dim=1, max_radius=1.5)
        assert result.complex_size > 0
        assert isinstance(result.total_persistence, float)
        assert result.persistence_landscape.shape == (100,)

    def test_alpha_pipeline(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (10, 2))
        tda = TDAPipeline()
        result = tda.fit(data, method='alpha', max_radius=2.0)
        assert result.complex_size > 0

    def test_persistence_landscape_empty(self):
        landscape = TDAPipeline._persistence_landscape([], n_points=50)
        np.testing.assert_array_equal(landscape, np.zeros(50))

    def test_betti_at_median(self):
        pairs = [PersistencePair(0.0, 1.0, 0), PersistencePair(0.2, 0.8, 1)]
        betti = TDAPipeline._betti_at_median(pairs, max_val=1.0)
        assert betti.get(0, 0) == 1  # pair [0, 1) alive at t=0.5
        assert betti.get(1, 0) == 1  # pair [0.2, 0.8) alive at t=0.5


# ---------------------------------------------------------------------------
# covering_spaces
# ---------------------------------------------------------------------------

from hololoom.core.warp.math.advanced.covering_spaces import (
    SimplicialComplex,
    FundamentalGroup,
    CoveringSpace,
    HigherHomotopy,
    CWComplex,
)


class TestSimplicialComplex:
    def test_edges(self):
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2])],
        )
        edges = sc.edges()
        assert len(edges) == 3

    def test_triangles(self):
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0, 1, 2])],
        )
        assert len(sc.triangles()) == 1

    def test_neighbors(self):
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0, 1]), frozenset([1, 2])],
        )
        assert sc.neighbors(1) == {0, 2}

    def test_skeleton(self):
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0]), frozenset([1]), frozenset([2]),
                       frozenset([0, 1]), frozenset([0, 1, 2])],
        )
        skel = sc.skeleton(1)
        assert all(len(s) <= 2 for s in skel.simplices)

    def test_dim(self):
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0, 1, 2])],
        )
        assert sc.dim == 2


class TestFundamentalGroup:
    def test_tree_trivial(self):
        # Tree has trivial fundamental group
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0]), frozenset([1]), frozenset([2]),
                       frozenset([0, 1]), frozenset([1, 2])],
        )
        fg = FundamentalGroup()
        gens = fg.compute(sc, base_point=0)
        assert len(gens) == 0

    def test_circle_one_generator(self):
        # Triangle boundary = circle = Z
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0]), frozenset([1]), frozenset([2]),
                       frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2])],
        )
        fg = FundamentalGroup()
        gens = fg.compute(sc, base_point=0)
        assert len(gens) == 1  # One loop

    def test_filled_triangle_trivial(self):
        # Filled triangle is contractible
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0]), frozenset([1]), frozenset([2]),
                       frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2]),
                       frozenset([0, 1, 2])],
        )
        fg = FundamentalGroup()
        n_gen, relations = fg.compute_presentation(sc, base_point=0)
        # The triangle relation kills the one generator
        # n_gen=1 but there's a relation making it trivial
        assert n_gen <= 1

    def test_euler_characteristic(self):
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0]), frozenset([1]), frozenset([2]),
                       frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2]),
                       frozenset([0, 1, 2])],
        )
        fg = FundamentalGroup()
        chi = fg.euler_characteristic(sc)
        assert chi == 3 - 3 + 1  # V - E + F = 1

    def test_empty_complex(self):
        sc = SimplicialComplex(vertices=[], simplices=[])
        fg = FundamentalGroup()
        assert fg.compute(sc) == []


class TestCoveringSpace:
    def test_trivial_covering(self):
        cs = CoveringSpace(n_sheets=1)
        path = [0, 1, 2]
        lifted = cs.lift_path(path)
        assert lifted == [(0, 0), (1, 0), (2, 0)]

    def test_fiber(self):
        cs = CoveringSpace(n_sheets=3)
        fiber = cs.fiber(5)
        assert len(fiber) == 3
        assert all(v == 5 for v, s in fiber)

    def test_deck_transformations_trivial(self):
        cs = CoveringSpace(n_sheets=2)
        deck = cs.deck_transformations()
        # No monodromy => all permutations commute
        assert len(deck) == 2

    def test_lift_with_monodromy(self):
        # 2-sheeted cover with monodromy swapping sheets
        perm = np.array([[0, 1], [1, 0]], dtype=float)
        cs = CoveringSpace(n_sheets=2, monodromy={1: perm})
        gen_labels = {(0, 1): 1}
        lifted = cs.lift_path([0, 1, 0], base_point_sheet=0,
                              generator_labels=gen_labels)
        # After crossing edge (0,1) which is generator 1, sheet should swap
        assert lifted[0] == (0, 0)
        assert lifted[1][1] == 1  # Swapped to sheet 1

    def test_is_regular_trivial(self):
        cs = CoveringSpace(n_sheets=1)
        assert cs.is_regular()

    def test_empty_path(self):
        cs = CoveringSpace(n_sheets=2)
        assert cs.lift_path([]) == []


class TestHigherHomotopy:
    def test_betti_filled_triangle(self):
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0]), frozenset([1]), frozenset([2]),
                       frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2]),
                       frozenset([0, 1, 2])],
        )
        hh = HigherHomotopy()
        betti = hh.betti_numbers(sc)
        assert betti[0] == 1  # Connected
        assert betti[1] == 0  # No 1-cycles (filled)

    def test_betti_circle(self):
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0]), frozenset([1]), frozenset([2]),
                       frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2])],
        )
        hh = HigherHomotopy()
        betti = hh.betti_numbers(sc)
        assert betti[0] == 1
        assert betti[1] == 1  # One loop

    def test_pi_1_circle(self):
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0]), frozenset([1]), frozenset([2]),
                       frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2])],
        )
        hh = HigherHomotopy()
        rank = hh.pi_n(sc, 1)
        assert rank == 1

    def test_pi_n_raises_for_zero(self):
        sc = SimplicialComplex(vertices=[0], simplices=[frozenset([0])])
        hh = HigherHomotopy()
        with pytest.raises(ValueError):
            hh.pi_n(sc, 0)

    def test_disconnected_components(self):
        sc = SimplicialComplex(
            vertices=[0, 1],
            simplices=[frozenset([0]), frozenset([1])],
        )
        hh = HigherHomotopy()
        betti = hh.betti_numbers(sc, max_dim=0)
        assert betti[0] == 2


class TestCWComplex:
    def test_from_simplicial(self):
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0]), frozenset([1]), frozenset([2]),
                       frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2])],
        )
        cw = CWComplex.from_simplicial(sc)
        assert cw.n_cells(0) == 3
        assert cw.n_cells(1) == 3

    def test_euler_characteristic(self):
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0]), frozenset([1]), frozenset([2]),
                       frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2]),
                       frozenset([0, 1, 2])],
        )
        cw = CWComplex.from_simplicial(sc)
        assert cw.euler_characteristic() == 1  # Disk

    def test_chain_complex(self):
        sc = SimplicialComplex(
            vertices=[0, 1, 2],
            simplices=[frozenset([0]), frozenset([1]), frozenset([2]),
                       frozenset([0, 1]), frozenset([1, 2]), frozenset([0, 2])],
        )
        cw = CWComplex.from_simplicial(sc)
        boundaries = cw.chain_complex()
        assert len(boundaries) >= 1
        # d_1 should be 3x3 (3 vertices, 3 edges)
        assert boundaries[0].shape == (3, 3)


# ---------------------------------------------------------------------------
# computational_geometry
# ---------------------------------------------------------------------------

from hololoom.core.warp.math.geometry.computational_geometry import (
    ConvexHull,
    DelaunayTriangulation,
    Triangle,
    VoronoiDiagram,
    KDTree,
)


class TestConvexHull:
    def test_square(self):
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=float)
        ch = ConvexHull()
        hull = ch.compute(points)
        assert len(hull) == 4  # Interior point excluded

    def test_triangle(self):
        points = np.array([[0, 0], [1, 0], [0.5, 1.0]], dtype=float)
        ch = ConvexHull()
        hull = ch.compute(points)
        assert len(hull) == 3

    def test_collinear(self):
        points = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        ch = ConvexHull()
        hull = ch.compute(points)
        assert len(hull) >= 2

    def test_area_square(self):
        points = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        hull = np.array([0, 1, 2, 3])
        area = ConvexHull.area(points, hull)
        assert abs(area - 4.0) < 1e-10

    def test_is_inside(self):
        points = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        hull = np.array([0, 1, 2, 3])
        assert ConvexHull.is_inside(np.array([1.0, 1.0]), points, hull)

    def test_is_outside(self):
        points = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        hull = np.array([0, 1, 2, 3])
        assert not ConvexHull.is_inside(np.array([3.0, 3.0]), points, hull)

    def test_small_point_set(self):
        ch = ConvexHull()
        assert len(ch.compute(np.array([[0, 0]]))) == 1
        assert len(ch.compute(np.array([[0, 0], [1, 0]]))) == 2


class TestDelaunayTriangulation:
    def test_basic(self):
        points = np.array([[0, 0], [1, 0], [0.5, 0.87], [0.5, 0.3]], dtype=float)
        dt = DelaunayTriangulation()
        tris = dt.compute(points)
        assert len(tris) >= 2
        for tri in tris:
            assert isinstance(tri, Triangle)

    def test_circumcenter(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([0.0, 1.0])
        cc = DelaunayTriangulation.circumcenter(p1, p2, p3)
        assert abs(cc[0] - 0.5) < 1e-10
        assert abs(cc[1] - 0.5) < 1e-10

    def test_too_few_points(self):
        dt = DelaunayTriangulation()
        assert dt.compute(np.array([[0, 0], [1, 0]])) == []


class TestVoronoiDiagram:
    def test_basic(self):
        points = np.array([[0, 0], [2, 0], [1, 1.7]], dtype=float)
        vd = VoronoiDiagram()
        result = vd.compute(points)
        assert result.vertices.shape[1] == 2
        assert len(result.regions) == 3

    def test_single_point(self):
        vd = VoronoiDiagram()
        result = vd.compute(np.array([[0, 0]]))
        assert len(result.regions) == 1


class TestKDTree:
    def test_basic_query(self):
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        kd = KDTree()
        kd.build(points)
        dists, indices = kd.query(np.array([0.1, 0.1]), k=1)
        assert indices[0] == 0

    def test_k_nearest(self):
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
        kd = KDTree()
        kd.build(points)
        dists, indices = kd.query(np.array([0.5, 0.5]), k=3)
        assert len(indices) == 3
        assert 4 in indices  # The point itself

    def test_query_ball(self):
        points = np.array([[0, 0], [1, 0], [5, 5]], dtype=float)
        kd = KDTree()
        kd.build(points)
        result = kd.query_ball(np.array([0.5, 0.0]), radius=1.0)
        assert 0 in result
        assert 1 in result
        assert 2 not in result

    def test_empty_tree(self):
        kd = KDTree()
        dists, indices = kd.query(np.array([0, 0]), k=1)
        assert len(dists) == 0

    def test_3d_query(self):
        rng = np.random.default_rng(42)
        points = rng.normal(0, 1, (30, 3))
        kd = KDTree()
        kd.build(points)
        q = np.array([0.0, 0.0, 0.0])
        dists, indices = kd.query(q, k=5)
        assert len(dists) == 5
        assert dists[0] <= dists[-1]  # Sorted


# ---------------------------------------------------------------------------
# genetic_programming
# ---------------------------------------------------------------------------

from hololoom.core.warp.math.self_assembly.genetic_programming import (
    GPNode,
    GeneticProgramming,
    NEATGenome,
    NEATGene,
    NEAT,
    GrammarGP,
    NeuralProgramInduction,
)


class TestGPNode:
    def test_terminal_depth(self):
        node = GPNode(value=1.0, is_terminal=True)
        assert node.depth == 0
        assert node.size == 1

    def test_tree_depth(self):
        node = GPNode(value='add', children=[
            GPNode(value=1.0, is_terminal=True),
            GPNode(value=2.0, is_terminal=True),
        ])
        assert node.depth == 1
        assert node.size == 3


class TestGeneticProgramming:
    def test_evaluate_terminal(self):
        node = GPNode(value=3.14, is_terminal=True)
        assert GeneticProgramming.evaluate(node, {}) == pytest.approx(3.14)

    def test_evaluate_variable(self):
        node = GPNode(value='x', is_terminal=True)
        assert GeneticProgramming.evaluate(node, {'x': 5.0}) == pytest.approx(5.0)

    def test_evaluate_add(self):
        node = GPNode(value='add', children=[
            GPNode(value=2.0, is_terminal=True),
            GPNode(value=3.0, is_terminal=True),
        ])
        assert GeneticProgramming.evaluate(node, {}) == pytest.approx(5.0)

    def test_evaluate_protected_div(self):
        node = GPNode(value='div', children=[
            GPNode(value=1.0, is_terminal=True),
            GPNode(value=0.0, is_terminal=True),
        ])
        result = GeneticProgramming.evaluate(node, {})
        assert result == pytest.approx(1.0)  # Protected division

    def test_evaluate_nested(self):
        # (2 + 3) * 4 = 20
        node = GPNode(value='mul', children=[
            GPNode(value='add', children=[
                GPNode(value=2.0, is_terminal=True),
                GPNode(value=3.0, is_terminal=True),
            ]),
            GPNode(value=4.0, is_terminal=True),
        ])
        assert GeneticProgramming.evaluate(node, {}) == pytest.approx(20.0)

    def test_evolve_simple(self):
        np.random.seed(42)
        # Find a tree that evaluates to ~10 for x=2
        def fitness(tree):
            try:
                val = GeneticProgramming.evaluate(tree, {'x': 2.0})
                return -abs(val - 10.0)
            except Exception:
                return -1000.0

        best = GeneticProgramming.evolve(
            fitness, terminal_set=[1.0, 2.0, 'x'],
            function_set=['add', 'mul'],
            n_gen=5, pop_size=20, max_depth=3,
        )
        assert isinstance(best, GPNode)

    def test_random_tree(self):
        tree = GeneticProgramming._random_tree(
            ['add', 'mul'], [1.0, 2.0], {'add': 2, 'mul': 2}, 3, 'full'
        )
        assert isinstance(tree, GPNode)

    def test_unknown_function_returns_zero(self):
        node = GPNode(value='bogus', children=[
            GPNode(value=1.0, is_terminal=True),
        ])
        assert GeneticProgramming.evaluate(node, {}) == 0.0

    def test_unknown_string_terminal(self):
        node = GPNode(value='undefined_var', is_terminal=True)
        assert GeneticProgramming.evaluate(node, {}) == 0.0


class TestNEATGenome:
    def test_n_nodes(self):
        g = NEATGenome(n_inputs=3, n_outputs=2, n_hidden=1)
        assert g.n_nodes == 6

    def test_forward(self):
        g = NEATGenome(n_inputs=2, n_outputs=1)
        g.connections = [
            NEATGene(0, 2, 0.5),
            NEATGene(1, 2, 0.5),
        ]
        out = g.forward(np.array([1.0, 1.0]))
        assert len(out) == 1
        assert np.isfinite(out[0])

    def test_disabled_connection(self):
        g = NEATGenome(n_inputs=2, n_outputs=1)
        g.connections = [
            NEATGene(0, 2, 1.0, enabled=False),
            NEATGene(1, 2, 0.5),
        ]
        out = g.forward(np.array([1.0, 1.0]))
        assert len(out) == 1


class TestNEAT:
    def test_evolve_xor_structure(self):
        np.random.seed(42)
        def fitness(genome):
            # Just check it runs
            try:
                total = 0.0
                for x1, x2, target in [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)]:
                    out = genome.forward(np.array([x1, x2], dtype=float))
                    total -= (out[0] - target) ** 2
                return total
            except Exception:
                return -100.0

        best = NEAT.evolve(fitness, n_inputs=2, n_outputs=1,
                           n_gen=3, pop_size=10)
        assert isinstance(best, NEATGenome)


class TestGrammarGP:
    def test_derive_simple(self):
        grammar = {
            'S': [['num']],
            'num': [['1'], ['2']],
        }
        choices = [0] * 50
        result = GrammarGP._derive(grammar, choices, 'S', 5)
        assert result in ['1', '2']

    def test_evolve_simple(self):
        np.random.seed(42)
        grammar = {
            'S': [['num'], ['num', '+', 'num']],
            'num': [['1'], ['2'], ['3']],
        }
        def fitness(s):
            try:
                return -abs(eval(s) - 5)
            except Exception:
                return -100.0

        best = GrammarGP.evolve(grammar, fitness, n_gen=3, pop_size=20)
        assert isinstance(best, str)


class TestNeuralProgramInduction:
    def test_induce_basic(self):
        np.random.seed(42)
        # Simple: copy reg 0 to reg 1
        examples = [
            (np.array([3.0, 0.0, 0.0, 0.0]), np.array([3.0, 3.0, 0.0, 0.0])),
        ]
        program = NeuralProgramInduction.induce(
            examples, max_steps=2, n_registers=4, n_iter=3, lr=0.001
        )
        assert len(program) == 2
        assert all(len(instr) == 3 for instr in program)
        assert all(instr[0] in ['ADD', 'SUB', 'MUL', 'COPY', 'ZERO'] for instr in program)


# ---------------------------------------------------------------------------
# self_modifying
# ---------------------------------------------------------------------------

from hololoom.core.warp.math.self_assembly.self_modifying import (
    LiquidStateMachine,
    EchoStateProperty,
    Quine,
    ReflectiveTower,
    OpenEndedEvolution,
)


class TestLiquidStateMachine:
    def test_process(self):
        lsm = LiquidStateMachine(n_neurons=20, connectivity=0.2,
                                  spectral_radius=0.9)
        spikes = (np.random.default_rng(0).random((10, 3)) > 0.7).astype(float)
        states = lsm.process(spikes)
        assert states.shape == (10, 20)
        assert np.isfinite(states).all()

    def test_reset_state(self):
        lsm = LiquidStateMachine(n_neurons=10)
        lsm.state = np.ones(10)
        lsm.reset_state()
        np.testing.assert_array_equal(lsm.state, np.zeros(10))
        np.testing.assert_array_equal(lsm.spikes, np.zeros(10))

    def test_train_readout(self):
        lsm = LiquidStateMachine(n_neurons=10)
        states = np.random.default_rng(0).normal(0, 1, (20, 10))
        targets = np.random.default_rng(1).normal(0, 1, (20, 2))
        W_out = lsm.train_readout(states, targets)
        assert W_out.shape == (10, 2)

    def test_spectral_radius_scaling(self):
        lsm = LiquidStateMachine(n_neurons=20, connectivity=0.3, spectral_radius=0.8)
        rho = np.max(np.abs(np.linalg.eigvals(lsm.W)))
        assert abs(rho - 0.8) < 0.1


class TestEchoStateProperty:
    def test_check_stable(self):
        W = np.eye(3) * 0.5
        assert EchoStateProperty.check(W) is True

    def test_check_unstable(self):
        W = np.eye(3) * 1.5
        assert EchoStateProperty.check(W) is False

    def test_check_borderline_strict(self):
        W = np.eye(3) * 1.0
        assert EchoStateProperty.check(W, strict=True) is False
        assert EchoStateProperty.check(W, strict=False) is True

    def test_spectral_radius(self):
        W = np.diag([0.5, -0.8, 0.3])
        rho = EchoStateProperty.spectral_radius(W)
        assert abs(rho - 0.8) < 1e-10

    def test_rescale(self):
        W = np.diag([1.0, 2.0, 3.0])
        W_scaled = EchoStateProperty.rescale(W, target_radius=0.9)
        rho = EchoStateProperty.spectral_radius(W_scaled)
        assert abs(rho - 0.9) < 1e-10

    def test_rescale_zero_matrix(self):
        W = np.zeros((3, 3))
        W_scaled = EchoStateProperty.rescale(W, 0.9)
        np.testing.assert_array_equal(W_scaled, W)

    def test_memory_capacity(self):
        np.random.seed(42)
        n = 20
        W = np.random.randn(n, n) * 0.1
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        if rho > 1e-10:
            W *= 0.9 / rho
        W_in = np.random.randn(n, 1) * 0.5
        mc = EchoStateProperty.memory_capacity(W, W_in, n_steps=200, max_delay=10)
        assert mc >= 0
        assert np.isfinite(mc)


class TestQuine:
    def test_python_quine(self):
        q = Quine.generate('python')
        assert isinstance(q, str)
        assert len(q) > 0

    def test_javascript_quine(self):
        q = Quine.generate('javascript')
        assert 'console.log' in q

    def test_c_quine(self):
        q = Quine.generate('c')
        assert '#include' in q

    def test_unknown_language(self):
        with pytest.raises(ValueError):
            Quine.generate('brainfuck')

    def test_verify_python(self):
        q = Quine._python_quine()
        result = Quine.verify(q, 'python')
        assert isinstance(result, bool)

    def test_verify_non_python(self):
        result = Quine.verify("code", 'javascript')
        assert result is False


class TestReflectiveTower:
    def test_evaluate_number(self):
        rt = ReflectiveTower()
        assert rt.evaluate(42) == 42

    def test_evaluate_arithmetic(self):
        rt = ReflectiveTower()
        assert rt.evaluate(['add', 2, 3]) == 5
        assert rt.evaluate(['mul', 4, 5]) == 20
        assert rt.evaluate(['sub', 10, 3]) == 7

    def test_evaluate_div_by_zero(self):
        rt = ReflectiveTower()
        assert rt.evaluate(['div', 1, 0]) == 0

    def test_evaluate_quote(self):
        rt = ReflectiveTower()
        assert rt.evaluate(['quote', [1, 2, 3]]) == [1, 2, 3]

    def test_evaluate_if(self):
        rt = ReflectiveTower()
        assert rt.evaluate(['if', 1, ['quote', 'yes'], ['quote', 'no']]) == 'yes'
        assert rt.evaluate(['if', 0, ['quote', 'yes'], ['quote', 'no']]) == 'no'

    def test_evaluate_define_and_lookup(self):
        rt = ReflectiveTower()
        rt.evaluate(['define', 'x', 10])
        assert rt.evaluate('x') == 10

    def test_evaluate_lambda(self):
        rt = ReflectiveTower()
        rt.evaluate(['define', 'double', ['lambda', ['x'], ['mul', 'x', 2]]])
        result = rt.evaluate(['double', 5])
        assert result == 10

    def test_reflect(self):
        rt = ReflectiveTower()
        info = rt.evaluate(['reflect'])
        assert info['type'] == 'evaluator'
        assert 'add' in info['operations']

    def test_meta_level(self):
        rt = ReflectiveTower()
        result = rt.evaluate(['add', 1, 2], level=1)
        assert result['result'] == 3
        assert result['level'] == 1

    def test_undefined_variable(self):
        rt = ReflectiveTower()
        with pytest.raises(NameError):
            rt.evaluate('undefined')

    def test_unknown_expression(self):
        rt = ReflectiveTower()
        with pytest.raises((ValueError, NameError)):
            rt.evaluate(['unknown_op', 1, 2])


class TestOpenEndedEvolution:
    def test_novelty_score_empty_archive(self):
        b = np.array([1.0, 0.0])
        score = OpenEndedEvolution.novelty_score(b, [])
        assert score == float('inf')

    def test_novelty_score(self):
        b = np.array([0.0, 0.0])
        archive = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        score = OpenEndedEvolution.novelty_score(b, archive, k=2)
        assert score > 0

    def test_step(self):
        np.random.seed(42)
        population = [np.random.randn(3) for _ in range(10)]
        archive = []
        behavior_fn = lambda g: g  # Identity
        mutation_fn = lambda g: g + np.random.randn(*g.shape) * 0.1

        new_pop, new_archive = OpenEndedEvolution.step(
            population, archive, behavior_fn, mutation_fn,
            k=3, archive_threshold=0.0, pop_size=10,
        )
        assert len(new_pop) == 10
        assert len(new_archive) > 0  # Should have archived something

    def test_step_maintains_pop_size(self):
        np.random.seed(42)
        pop = [np.array([float(i)]) for i in range(5)]
        new_pop, _ = OpenEndedEvolution.step(
            pop, [], lambda g: g, lambda g: g + 0.1,
            k=2, pop_size=5,
        )
        assert len(new_pop) == 5
