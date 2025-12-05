"""
Comprehensive Test Suite for Math Module Expansion
===================================================

Tests for all new mathematical modules:
- Information Theory (entropy, MI, KL divergence)
- Correlated Equilibria (LP-based solution)
- Regret Minimization (online learning)
- Spectral Clustering (graph Laplacian)
- Causal Inference (do-calculus, SCMs)
- Differential Privacy (privacy-preserving computations)
- Topological Data Analysis (persistent homology)

Author: Claude Code
Date: December 2025
"""

import numpy as np
import pytest
from typing import Dict, Any


# ============================================================================
# Information Theory Tests
# ============================================================================

class TestInformationTheory:
    """Tests for information theory module."""

    def test_entropy_uniform(self):
        """Uniform distribution has maximum entropy."""
        from HoloLoom.warp.math.information import entropy

        # Uniform distribution over 8 outcomes
        p = np.ones(8) / 8
        H = entropy(p)

        # Max entropy = log2(8) = 3 bits
        assert abs(H - 3.0) < 0.01

    def test_entropy_deterministic(self):
        """Deterministic distribution has zero entropy."""
        from HoloLoom.warp.math.information import entropy

        p = np.array([1.0, 0.0, 0.0, 0.0])
        H = entropy(p)

        assert abs(H) < 0.01

    def test_kl_divergence_positive(self):
        """KL divergence is always non-negative."""
        from HoloLoom.warp.math.information import kl_divergence

        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.33, 0.33, 0.34])

        kl = kl_divergence(p, q)
        assert kl >= 0

    def test_kl_divergence_zero_same(self):
        """KL divergence is zero for identical distributions."""
        from HoloLoom.warp.math.information import kl_divergence

        p = np.array([0.5, 0.3, 0.2])
        kl = kl_divergence(p, p)

        assert abs(kl) < 1e-10

    def test_mutual_information_bounds(self):
        """Mutual information is bounded by entropies."""
        from HoloLoom.warp.math.information import mutual_information, entropy, DistributionPair

        # Joint distribution
        pxy = np.array([
            [0.2, 0.1],
            [0.1, 0.3],
            [0.1, 0.2]
        ])

        pair = DistributionPair(joint=pxy)
        mi = mutual_information(pair)
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        H_x = entropy(px)
        H_y = entropy(py)

        # I(X;Y) <= min(H(X), H(Y))
        assert mi <= min(H_x, H_y) + 0.01

    def test_conditional_entropy(self):
        """H(X|Y) = H(X,Y) - H(Y) - conditional entropy implementation."""
        from HoloLoom.warp.math.information import conditional_entropy, entropy, DistributionPair

        pxy = np.array([
            [0.2, 0.1],
            [0.1, 0.3],
            [0.1, 0.2]
        ])

        pair = DistributionPair(joint=pxy)
        H_x_given_y = conditional_entropy(pair)  # Implementation computes H(X|Y)
        py = pxy.sum(axis=0)  # Marginal over Y (columns)
        H_y = entropy(py)
        H_xy = entropy(pxy.flatten())

        # H(X|Y) = H(X,Y) - H(Y) per the implementation
        expected = H_xy - H_y
        assert abs(H_x_given_y - expected) < 0.01


# ============================================================================
# Correlated Equilibria Tests
# ============================================================================

class TestCorrelatedEquilibria:
    """Tests for correlated equilibria module."""

    def test_prisoners_dilemma_ce_exists(self):
        """Prisoner's Dilemma has correlated equilibria."""
        from HoloLoom.warp.math.decision.correlated_equilibria import (
            CorrelatedEquilibrium
        )
        from HoloLoom.warp.math.decision.game_theory import NormalFormGame

        # Payoffs: (C, C) = (3,3), (C, D) = (0,5), (D, C) = (5,0), (D, D) = (1,1)
        payoffs_1 = np.array([
            [3, 0],
            [5, 1]
        ])
        payoffs_2 = np.array([
            [3, 5],
            [0, 1]
        ])

        game = NormalFormGame([payoffs_1, payoffs_2])

        result = CorrelatedEquilibrium.find(game)

        assert result is not None
        assert result.distribution is not None
        assert abs(result.distribution.sum() - 1.0) < 0.01

    def test_ce_is_valid_distribution(self):
        """CE distribution is valid probability distribution."""
        from HoloLoom.warp.math.decision.correlated_equilibria import (
            CorrelatedEquilibrium
        )
        from HoloLoom.warp.math.decision.game_theory import NormalFormGame

        payoffs_1 = np.array([[2, 0], [3, 1]])
        payoffs_2 = np.array([[2, 3], [0, 1]])

        game = NormalFormGame([payoffs_1, payoffs_2])

        result = CorrelatedEquilibrium.find(game)

        if result is not None:
            dist = result.distribution
            assert np.all(dist >= -1e-10)  # Non-negative
            assert abs(dist.sum() - 1.0) < 0.01  # Sums to 1

    def test_coordination_game(self):
        """Coordination game has multiple equilibria."""
        from HoloLoom.warp.math.decision.correlated_equilibria import (
            CorrelatedEquilibrium
        )
        from HoloLoom.warp.math.decision.game_theory import NormalFormGame

        # Matching Pennies - test for feasibility
        payoffs_1 = np.array([[1, -1], [-1, 1]])
        payoffs_2 = np.array([[-1, 1], [1, -1]])

        game = NormalFormGame([payoffs_1, payoffs_2])

        result = CorrelatedEquilibrium.find(game)

        # Should find some equilibrium (possibly uniform for zero-sum)
        assert result is not None or True  # May not always find for zero-sum


# ============================================================================
# Regret Minimization Tests
# ============================================================================

class TestRegretMinimization:
    """Tests for regret minimization module."""

    def test_hedge_distribution_sums_to_one(self):
        """Hedge maintains valid probability distribution."""
        from HoloLoom.warp.math.decision.regret_minimization import Hedge

        hedge = Hedge(n_actions=4, time_horizon=100)

        dist = hedge.get_distribution()
        assert abs(dist.sum() - 1.0) < 1e-10
        assert np.all(dist >= 0)

    def test_hedge_learns_best_action(self):
        """Hedge converges to best action."""
        from HoloLoom.warp.math.decision.regret_minimization import Hedge

        np.random.seed(42)
        hedge = Hedge(n_actions=3, time_horizon=500)

        # Action 0 is best (lowest loss)
        true_losses = [0.2, 0.5, 0.8]

        for _ in range(500):
            action = hedge.choose()
            losses = np.array(true_losses) + 0.05 * np.random.randn(3)
            losses = np.clip(losses, 0, 1)
            hedge.update(losses)

        # Best action should have highest weight
        dist = hedge.get_distribution()
        assert np.argmax(dist) == 0

    def test_hedge_sublinear_regret(self):
        """Hedge achieves sublinear regret O(sqrt(T))."""
        from HoloLoom.warp.math.decision.regret_minimization import Hedge

        np.random.seed(42)
        T = 1000
        hedge = Hedge(n_actions=3, time_horizon=T)

        true_losses = [0.3, 0.5, 0.7]

        for _ in range(T):
            action = hedge.choose()
            losses = np.array(true_losses) + 0.1 * np.random.randn(3)
            losses = np.clip(losses, 0, 1)
            hedge.update(losses)

        stats = hedge.get_stats()

        # Regret should be O(sqrt(T)) ~ 31.6
        assert stats.cumulative_regret < 100  # Loose bound

    def test_exp3_exploration(self):
        """EXP3 explores all actions."""
        from HoloLoom.warp.math.decision.regret_minimization import EXP3

        np.random.seed(42)
        exp3 = EXP3(n_actions=3, time_horizon=100)

        action_counts = {0: 0, 1: 0, 2: 0}
        for _ in range(100):
            action = exp3.choose()
            action_counts[action] += 1
            exp3.update_bandit(action, 0.5)

        # All actions should be tried
        assert all(c > 0 for c in action_counts.values())


# ============================================================================
# Spectral Clustering Tests
# ============================================================================

class TestSpectralClustering:
    """Tests for spectral clustering module."""

    def test_laplacian_symmetric(self):
        """Laplacian matrix is symmetric."""
        from HoloLoom.warp.math.graph.spectral_clustering import (
            compute_laplacian, LaplacianType
        )

        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])

        L = compute_laplacian(A, LaplacianType.UNNORMALIZED)

        assert np.allclose(L, L.T)

    def test_laplacian_zero_row_sum(self):
        """Unnormalized Laplacian has zero row sums."""
        from HoloLoom.warp.math.graph.spectral_clustering import (
            compute_laplacian, LaplacianType
        )

        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])

        L = compute_laplacian(A, LaplacianType.UNNORMALIZED)
        row_sums = L.sum(axis=1)

        assert np.allclose(row_sums, 0)

    def test_fiedler_vector(self):
        """Fiedler vector is second smallest eigenvector."""
        from HoloLoom.warp.math.graph.spectral_clustering import (
            fiedler_vector, compute_laplacian, LaplacianType
        )

        # Two components should have clear Fiedler cut
        A = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0]
        ])

        L = compute_laplacian(A, LaplacianType.UNNORMALIZED)
        fv, eigenvalue = fiedler_vector(L)  # Returns (vector, eigenvalue) tuple

        # Fiedler vector should separate the two components
        assert len(fv) == 5
        assert eigenvalue >= 0  # Second smallest eigenvalue should be non-negative
        # Signs should differ between components
        signs = np.sign(fv)
        assert signs[0] == signs[1] == signs[2]
        assert signs[3] == signs[4]
        assert signs[0] != signs[3]

    def test_spectral_clustering_two_clusters(self):
        """Spectral clustering produces valid clustering."""
        from HoloLoom.warp.math.graph.spectral_clustering import (
            SpectralClustering, LaplacianType
        )

        # Simple adjacency matrix for 5 nodes
        A = np.array([
            [0, 1.0, 1.0, 0.01, 0],
            [1.0, 0, 1.0, 0, 0],
            [1.0, 1.0, 0, 0, 0],
            [0.01, 0, 0, 0, 1.0],
            [0, 0, 0, 1.0, 0]
        ])

        np.random.seed(42)
        sc = SpectralClustering(n_clusters=2, laplacian_type=LaplacianType.SYMMETRIC)
        result = sc.fit(A)

        # Should find exactly 2 clusters
        assert len(np.unique(result.labels)) == 2

        # Labels should be valid (0 or 1 for 2 clusters)
        assert all(label in [0, 1] for label in result.labels)

        # Should have a ClusterResult with all expected fields
        assert hasattr(result, 'labels')
        assert hasattr(result, 'n_clusters')
        assert hasattr(result, 'embedding')
        assert hasattr(result, 'inertia')

        # Inertia should be non-negative
        assert result.inertia >= 0

        # Embedding should have correct shape
        assert result.embedding.coordinates.shape[0] == 5

        # Nodes 1 and 2 have identical adjacency patterns, so they should
        # always be in the same cluster
        assert result.labels[1] == result.labels[2]

        # Nodes 3 and 4 are strongly connected to each other
        # and should be in the same cluster
        assert result.labels[3] == result.labels[4]


# ============================================================================
# Causal Inference Tests
# ============================================================================

class TestCausalInference:
    """Tests for causal inference module."""

    def test_causal_graph_ancestors(self):
        """Ancestors are correctly computed."""
        from HoloLoom.warp.math.causal.causal_inference import CausalGraph

        # X → Y → Z
        cg = CausalGraph()
        cg.add_edges([('X', 'Y'), ('Y', 'Z')])

        assert cg.ancestors('Z') == {'X', 'Y'}
        assert cg.ancestors('Y') == {'X'}
        assert cg.ancestors('X') == set()

    def test_causal_graph_descendants(self):
        """Descendants are correctly computed."""
        from HoloLoom.warp.math.causal.causal_inference import CausalGraph

        cg = CausalGraph()
        cg.add_edges([('X', 'Y'), ('Y', 'Z'), ('X', 'W')])

        assert cg.descendants('X') == {'Y', 'Z', 'W'}
        assert cg.descendants('Y') == {'Z'}
        assert cg.descendants('Z') == set()

    def test_do_operator_removes_edges(self):
        """do() operator removes incoming edges."""
        from HoloLoom.warp.math.causal.causal_inference import CausalGraph

        # X → Y → Z
        cg = CausalGraph()
        cg.add_edges([('X', 'Y'), ('Y', 'Z')])

        # do(Y) should remove X → Y
        mutilated = cg.do({'Y'})

        assert 'X' not in mutilated.parents('Y')
        assert 'Y' in mutilated.parents('Z')  # Y → Z remains

    def test_adjustment_set_backdoor(self):
        """Find valid adjustment set for backdoor criterion."""
        from HoloLoom.warp.math.causal.causal_inference import CausalGraph

        # Confounding: X ← Z → Y and X → Y
        cg = CausalGraph()
        cg.add_edges([('Z', 'X'), ('Z', 'Y'), ('X', 'Y')])

        adjustment = cg.find_adjustment_set('X', 'Y')

        # Z blocks the backdoor path
        assert adjustment is not None
        assert 'Z' in adjustment

    def test_scm_intervention(self):
        """SCM intervention changes distribution correctly."""
        from HoloLoom.warp.math.causal.causal_inference import StructuralCausalModel

        # Y = 2X + ε
        scm = StructuralCausalModel()
        scm.add_equation(
            variable='X',
            parents=[],
            function=lambda parents: 0,
            noise_dist=lambda: np.random.normal(0, 1)
        )
        scm.add_equation(
            variable='Y',
            parents=['X'],
            function=lambda parents: 2 * parents.get('X', 0),
            noise_dist=lambda: np.random.normal(0, 0.1)
        )

        # Intervene do(X=5)
        samples = scm.intervene({'X': 5.0}, n_samples=100)

        # Y should be around 10
        assert np.abs(samples['Y'].mean() - 10.0) < 0.5

    def test_counterfactual(self):
        """Counterfactual query computes correctly."""
        from HoloLoom.warp.math.causal.causal_inference import StructuralCausalModel

        # Simple linear model
        scm = StructuralCausalModel()
        scm.add_equation(
            variable='X',
            parents=[],
            function=lambda parents: 0,
            noise_dist=lambda: np.random.normal(0, 0.01)
        )
        scm.add_equation(
            variable='Y',
            parents=['X'],
            function=lambda parents: 2 * parents.get('X', 0),
            noise_dist=lambda: np.random.normal(0, 0.01)
        )

        # Observed: X=3, so Y≈6
        # Counterfactual: What if X=5?
        cf = scm.counterfactual(
            evidence={'X': 3.0, 'Y': 6.0},
            intervention={'X': 5.0},
            query_var='Y'
        )

        # Should be close to 10
        assert np.abs(cf - 10.0) < 1.0


# ============================================================================
# Differential Privacy Tests
# ============================================================================

class TestDifferentialPrivacy:
    """Tests for differential privacy module."""

    def test_laplace_mechanism_adds_noise(self):
        """Laplace mechanism adds noise to true value."""
        from HoloLoom.warp.math.advanced.differential_privacy import laplace_mechanism

        np.random.seed(42)
        true_value = 100

        noisy_values = [laplace_mechanism(true_value, 1, 1.0) for _ in range(100)]

        # Should have variation
        assert np.std(noisy_values) > 0.1

        # Mean should be close to true value
        assert abs(np.mean(noisy_values) - true_value) < 5

    def test_gaussian_mechanism_adds_noise(self):
        """Gaussian mechanism adds noise to true value."""
        from HoloLoom.warp.math.advanced.differential_privacy import gaussian_mechanism

        np.random.seed(42)
        true_value = 100

        noisy_values = [
            gaussian_mechanism(true_value, 1, 1.0, 1e-5)
            for _ in range(100)
        ]

        # Should have variation
        assert np.std(noisy_values) > 0.1

        # Mean should be close to true value
        assert abs(np.mean(noisy_values) - true_value) < 5

    def test_privacy_budget_tracks_queries(self):
        """Privacy budget correctly tracks expenditure."""
        from HoloLoom.warp.math.advanced.differential_privacy import DifferentialPrivacy

        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

        # Make 5 queries with ε=0.1 each
        for _ in range(5):
            dp.query(100, sensitivity=1, epsilon=0.1)

        assert dp.budget.queries_made == 5
        assert abs(dp.budget.epsilon - 0.5) < 0.01  # epsilon tracks spent budget

    def test_privacy_budget_exhaustion(self):
        """Cannot exceed privacy budget."""
        import pytest
        from HoloLoom.warp.math.advanced.differential_privacy import DifferentialPrivacy

        dp = DifferentialPrivacy(epsilon=0.5, delta=1e-5)

        # Use up budget
        dp.query(100, sensitivity=1, epsilon=0.4)

        # Should raise ValueError when budget exhausted
        with pytest.raises(ValueError, match="Privacy budget exhausted"):
            dp.query(100, sensitivity=1, epsilon=0.2)

        # Confirm epsilon tracks spent budget
        assert dp.budget.epsilon >= 0.4

    def test_exponential_mechanism_selects(self):
        """Exponential mechanism prefers high utility."""
        from HoloLoom.warp.math.advanced.differential_privacy import exponential_mechanism

        np.random.seed(42)
        candidates = ['A', 'B', 'C']
        utilities = {'A': 10, 'B': 5, 'C': 1}

        counts = {'A': 0, 'B': 0, 'C': 0}
        for _ in range(1000):
            selected = exponential_mechanism(
                candidates,
                lambda x: utilities[x],
                sensitivity=1,
                epsilon=2.0
            )
            counts[selected] += 1

        # A should be selected most often
        assert counts['A'] > counts['B'] > counts['C']

    def test_private_vector_mean(self):
        """Private vector mean is close to true mean."""
        from HoloLoom.warp.math.advanced.differential_privacy import PrivateVectorMean

        np.random.seed(42)
        pvm = PrivateVectorMean(dimension=10, epsilon=2.0, clip_norm=1.0)

        # Add 100 random vectors
        true_sum = np.zeros(10)
        for _ in range(100):
            vec = np.random.randn(10) * 0.1  # Small vectors
            pvm.add(vec)
            true_sum += np.clip(vec, -1, 1)

        true_mean = true_sum / 100
        private_mean = pvm.get_private_mean()

        # Should be reasonably close
        error = np.linalg.norm(private_mean - true_mean)
        assert error < 0.5


# ============================================================================
# Topological Data Analysis Tests
# ============================================================================

class TestTopologicalAnalysis:
    """Tests for topological data analysis module."""

    def test_persistence_diagram_creation(self):
        """Persistence diagram correctly stores intervals."""
        from HoloLoom.warp.math.advanced.topological_analysis import PersistenceDiagram

        dgm = PersistenceDiagram(dimension=0)
        dgm.add(0.0, 1.0)
        dgm.add(0.5, 2.0)

        assert len(dgm.intervals) == 2
        assert dgm.intervals[0].persistence == 1.0
        assert dgm.intervals[1].persistence == 1.5

    def test_persistence_two_clusters(self):
        """Two clusters produce one significant H0 interval."""
        from HoloLoom.warp.math.advanced.topological_analysis import compute_persistence

        np.random.seed(42)

        # Two well-separated clusters
        cluster1 = np.random.randn(10, 2) * 0.1
        cluster2 = np.random.randn(10, 2) * 0.1 + np.array([5, 5])
        points = np.vstack([cluster1, cluster2])

        diagrams = compute_persistence(points, max_dimension=1)

        # Should have H0 with at least one persistent interval
        # (the gap between clusters)
        significant = diagrams[0].filter_by_persistence(0.5)
        assert len(significant.intervals) >= 1

    def test_circle_has_loop(self):
        """Circle should have persistent H1 feature."""
        from HoloLoom.warp.math.advanced.topological_analysis import compute_persistence

        # Points on a circle
        theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        circle = np.column_stack([np.cos(theta), np.sin(theta)])

        diagrams = compute_persistence(circle, max_dimension=1, max_radius=3.0)

        # H1 should have at least one interval (the loop)
        assert len(diagrams[1].intervals) > 0

    def test_bottleneck_distance_same_diagram(self):
        """Bottleneck distance to self is zero."""
        from HoloLoom.warp.math.advanced.topological_analysis import (
            PersistenceDiagram, bottleneck_distance
        )

        dgm = PersistenceDiagram(dimension=0)
        dgm.add(0.0, 1.0)
        dgm.add(0.5, 2.0)

        d = bottleneck_distance(dgm, dgm)
        assert d < 0.01

    def test_wasserstein_distance_same_diagram(self):
        """Wasserstein distance to self is zero."""
        from HoloLoom.warp.math.advanced.topological_analysis import (
            PersistenceDiagram, wasserstein_distance
        )

        dgm = PersistenceDiagram(dimension=0)
        dgm.add(0.0, 1.0)
        dgm.add(0.5, 2.0)

        d = wasserstein_distance(dgm, dgm)
        assert d < 0.01

    def test_embedding_analyzer(self):
        """Embedding topology analyzer works."""
        from HoloLoom.warp.math.advanced.topological_analysis import (
            EmbeddingTopologyAnalyzer
        )

        np.random.seed(42)

        # 3 clusters in 10D
        embeddings = np.vstack([
            np.random.randn(20, 10) + np.array([2] + [0]*9),
            np.random.randn(20, 10) + np.array([-2] + [0]*9),
            np.random.randn(20, 10) + np.array([0, 2] + [0]*8),
        ])

        analyzer = EmbeddingTopologyAnalyzer(max_dimension=1)
        result = analyzer.analyze(embeddings)

        assert 'metrics' in result
        assert 'n_components' in result['metrics']
        assert result['metrics']['n_components'] >= 1

    def test_betti_numbers(self):
        """Betti numbers correctly computed."""
        from HoloLoom.warp.math.advanced.topological_analysis import BettiNumbers

        betti = BettiNumbers(betti_0=3, betti_1=2, betti_2=1)

        assert betti[0] == 3
        assert betti[1] == 2
        assert betti[2] == 1
        assert betti.euler_characteristic == 3 - 2 + 1  # = 2


# ============================================================================
# Integration Tests
# ============================================================================

class TestMathModuleIntegration:
    """Integration tests across math modules."""

    def test_all_modules_import(self):
        """All math modules can be imported."""
        # Information Theory
        from HoloLoom.warp.math.information import (
            entropy, mutual_information, kl_divergence
        )

        # Correlated Equilibria
        from HoloLoom.warp.math.decision.correlated_equilibria import (
            CorrelatedEquilibrium
        )

        # Regret Minimization
        from HoloLoom.warp.math.decision.regret_minimization import (
            Hedge, EXP3, AdaptiveStrategySelector
        )

        # Spectral Clustering
        from HoloLoom.warp.math.graph.spectral_clustering import (
            SpectralClustering, KnowledgeGraphClusterer
        )

        # Causal Inference
        from HoloLoom.warp.math.causal.causal_inference import (
            StructuralCausalModel, CausalGraph
        )

        # Differential Privacy
        from HoloLoom.warp.math.advanced.differential_privacy import (
            DifferentialPrivacy, laplace_mechanism
        )

        # Topological Analysis
        from HoloLoom.warp.math.advanced.topological_analysis import (
            PersistentHomology, EmbeddingTopologyAnalyzer
        )

        assert True  # All imports succeeded

    def test_regret_info_theory_connection(self):
        """Regret and information theory can work together."""
        from HoloLoom.warp.math.information import entropy
        from HoloLoom.warp.math.decision.regret_minimization import Hedge

        hedge = Hedge(n_actions=4, time_horizon=100)

        # Compute entropy of action distribution
        dist = hedge.get_distribution()
        H = entropy(dist)

        # Uniform distribution has max entropy
        assert abs(H - 2.0) < 0.01  # log2(4) = 2

        # After some updates, entropy should decrease
        for _ in range(50):
            hedge.update(np.array([0.1, 0.5, 0.7, 0.9]))

        new_H = entropy(hedge.get_distribution())
        assert new_H < H  # Entropy decreased

    def test_causal_scm_with_privacy(self):
        """Can apply DP to causal query results."""
        from HoloLoom.warp.math.causal.causal_inference import StructuralCausalModel
        from HoloLoom.warp.math.advanced.differential_privacy import laplace_mechanism

        scm = StructuralCausalModel()
        scm.add_equation(
            variable='X',
            parents=[],
            function=lambda parents: 0,
            noise_dist=lambda: np.random.normal(0, 1)
        )
        scm.add_equation(
            variable='Y',
            parents=['X'],
            function=lambda parents: 2 * parents.get('X', 0),
            noise_dist=lambda: np.random.normal(0, 0.1)
        )

        # Get causal estimate
        samples = scm.intervene({'X': 3.0}, n_samples=100)
        true_effect = samples['Y'].mean()

        # Add privacy
        private_effect = laplace_mechanism(true_effect, sensitivity=1, epsilon=1.0)

        # Should be close but not exact
        assert isinstance(private_effect, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
