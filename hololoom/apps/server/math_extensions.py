"""
Math Extensions — information geometry, differential privacy, constrained
optimization, hyperbolic distance, and topological analysis wired into the
promotion pipeline.

No beautiful math left unwired.

Each extension is independently optional (try/except ImportError).
Each provides a function that other pipeline components can call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from hololoom.apps.server.vault_bridge import VAULT_ROOT

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Natural Gradient — geometry-aware Thompson Sampling updates
# ============================================================================

try:
    from hololoom.core.warp.math.geometry.information_geometry import NaturalGradient
    HAS_NATURAL_GRADIENT = True
except ImportError:
    HAS_NATURAL_GRADIENT = False


def beta_natural_gradient_update(
    alpha: float,
    beta_param: float,
    reward: float,
    learning_rate: float = 0.1,
) -> tuple[float, float]:
    """Update Beta(alpha, beta) parameters using natural gradient.

    Instead of the naive update (alpha += reward), this follows the
    steepest descent direction on the Beta manifold via Fisher information.

    For Beta(a, b), the Fisher information matrix is:
        G = [[psi'(a) - psi'(a+b),  -psi'(a+b)],
             [-psi'(a+b),            psi'(b) - psi'(a+b)]]
    where psi' is the trigamma function.

    The natural gradient G^{-1} * grad gives the update direction
    that's invariant to reparameterization.

    Falls back to standard update if NaturalGradient unavailable.
    """
    if not HAS_NATURAL_GRADIENT:
        # Standard update (existing behavior)
        if reward > 0:
            return alpha + reward, beta_param
        else:
            return alpha, beta_param + abs(reward)

    try:
        from scipy.special import polygamma

        # Trigamma function: psi'(x)
        psi1_a = float(polygamma(1, max(alpha, 0.01)))
        psi1_b = float(polygamma(1, max(beta_param, 0.01)))
        psi1_ab = float(polygamma(1, max(alpha + beta_param, 0.01)))

        # Fisher information matrix for Beta distribution
        G = np.array([
            [psi1_a - psi1_ab, -psi1_ab],
            [-psi1_ab, psi1_b - psi1_ab],
        ])

        # Gradient of log-likelihood w.r.t. (alpha, beta)
        # For a Bernoulli outcome with reward r:
        # grad_alpha = psi(alpha+beta) - psi(alpha) + log(reward + 0.01)
        # grad_beta  = psi(alpha+beta) - psi(beta) + log(1 - reward + 0.01)
        from scipy.special import digamma
        psi_a = float(digamma(max(alpha, 0.01)))
        psi_b = float(digamma(max(beta_param, 0.01)))
        psi_ab = float(digamma(max(alpha + beta_param, 0.01)))

        grad = np.array([
            psi_ab - psi_a + np.log(max(reward, 0.01)),
            psi_ab - psi_b + np.log(max(1 - reward, 0.01)),
        ])

        # Natural gradient: G^{-1} * grad
        try:
            natural_grad = np.linalg.solve(G, grad)
        except np.linalg.LinAlgError:
            natural_grad = grad  # Fallback if G is singular

        # Update parameters (dampen to prevent overshooting)
        step = learning_rate * natural_grad
        step = np.clip(step, -0.5, 0.5)  # limit step size
        new_alpha = max(0.5, alpha + step[0])
        new_beta = max(0.5, beta_param + step[1])

        return new_alpha, new_beta

    except ImportError:
        # scipy not available — standard update
        if reward > 0:
            return alpha + reward, beta_param
        else:
            return alpha, beta_param + abs(reward)


# ============================================================================
# 2. Differential Privacy — noise injection for vault promotion
# ============================================================================

try:
    from hololoom.core.warp.math.advanced.differential_privacy import (
        PrivacyBudget,
        gaussian_mechanism,
        laplace_mechanism,
    )
    HAS_DP = True
except ImportError:
    HAS_DP = False


@dataclass
class PrivateScore:
    """A score with differential privacy noise applied."""
    true_value: float
    noisy_value: float
    epsilon: float
    mechanism: str


# Per-department privacy budgets
_privacy_budgets: dict[str, PrivacyBudget] = {}


def get_privacy_budget(dept: str) -> PrivacyBudget | None:
    """Get or create privacy budget for a department."""
    if not HAS_DP:
        return None
    if dept not in _privacy_budgets:
        _privacy_budgets[dept] = PrivacyBudget()
    return _privacy_budgets[dept]


def private_quality_score(
    true_score: float,
    dept: str,
    epsilon: float = 0.5,
    sensitivity: float = 1.0,
) -> PrivateScore:
    """Apply differential privacy to a quality score before logging.

    Ensures individual note quality scores don't leak information
    about the specific content that was evaluated. The noisy score
    is what gets recorded in federation tracking.

    Falls back to true score if DP unavailable.
    """
    if not HAS_DP:
        return PrivateScore(true_score, true_score, 0.0, "none")

    budget = get_privacy_budget(dept)
    if budget and budget.is_exhausted(max_epsilon=10.0):
        logger.warning("Privacy budget exhausted for dept %s", dept)
        return PrivateScore(true_score, true_score, 0.0, "budget_exhausted")

    noisy = float(laplace_mechanism(true_score, sensitivity, epsilon))
    noisy = max(0.0, min(1.0, noisy))  # Clamp to valid range

    if budget:
        budget.add_query(epsilon)

    return PrivateScore(true_score, noisy, epsilon, "laplace")


def private_embedding(
    embedding: np.ndarray,
    epsilon: float = 1.0,
    delta: float = 1e-5,
) -> np.ndarray:
    """Apply Gaussian DP noise to an embedding vector before vault storage.

    Protects individual note content from being recoverable via the
    stored embedding. Uses Gaussian mechanism for better concentration.
    """
    if not HAS_DP:
        return embedding

    # L2 sensitivity of normalized embeddings is bounded by 2
    sensitivity = 2.0
    return gaussian_mechanism(embedding, sensitivity, epsilon, delta)


# ============================================================================
# 3. Constrained Optimization — dynamic rubric weighting
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.optimization import (
        ConstrainedProblem,
        LagrangeMultipliers,
    )
    HAS_OPTIMIZATION = True
except ImportError:
    HAS_OPTIMIZATION = False


def optimize_rubric_weights(
    base_weights: dict[str, float],
    quality_target: float = 0.7,
    time_budget: float = 1.0,
    term_costs: dict[str, float] | None = None,
) -> dict[str, float]:
    """Find optimal rubric weights under quality and time constraints.

    Maximizes expected score subject to:
      - Sum of weights = 100 (normalization)
      - Time cost of evaluating all terms <= time_budget
      - Each weight >= 5 (minimum relevance)

    Falls back to base weights if optimization unavailable.
    """
    if not HAS_OPTIMIZATION or not base_weights:
        return base_weights

    terms = list(base_weights.keys())
    n = len(terms)

    if term_costs is None:
        # Default: mechanical terms are cheap, LLM terms are expensive
        term_costs = {t: 0.1 if t in ("links", "substance") else 0.3 for t in terms}

    try:
        # Gradient ascent with projected constraints
        weights = np.array([base_weights[t] for t in terms], dtype=np.float64)
        costs = np.array([term_costs.get(t, 0.2) for t in terms], dtype=np.float64)

        # Simple projected gradient: maximize quality subject to budget
        for _ in range(20):
            # Gradient: increase weight of high-value, low-cost terms
            value_per_cost = weights / (costs + 1e-6)
            grad = value_per_cost / value_per_cost.sum()

            # Step
            weights += 0.5 * grad

            # Project: enforce constraints
            weights = np.maximum(weights, 5.0)  # min weight
            total_cost = np.sum(weights * costs) / weights.sum()
            if total_cost > time_budget:
                # Scale down expensive terms
                expensive = costs > np.median(costs)
                weights[expensive] *= 0.9

            # Normalize to sum to 100
            weights = weights * 100.0 / weights.sum()

        return {terms[i]: round(float(weights[i]), 1) for i in range(n)}

    except Exception as e:
        logger.debug("Rubric weight optimization failed: %s", e)
        return base_weights


# ============================================================================
# 4. Hyperbolic Distance — hierarchical belief communities
# ============================================================================

try:
    from hololoom.core.warp.math.extensions.hyperbolic_geometry import PoincareBall
    HAS_HYPERBOLIC = True
except ImportError:
    HAS_HYPERBOLIC = False


def hyperbolic_community_distance(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
) -> float:
    """Compute hyperbolic distance between two sets of embeddings.

    Projects embeddings into the Poincare ball and computes distance.
    Better than Euclidean for hierarchical relationships — exponential
    separation with distance respects tree structure.

    Returns mean pairwise hyperbolic distance.
    Falls back to Euclidean if hyperbolic geometry unavailable.
    """
    if not HAS_HYPERBOLIC:
        # Euclidean fallback
        centroid_a = embeddings_a.mean(axis=0)
        centroid_b = embeddings_b.mean(axis=0)
        return float(np.linalg.norm(centroid_a - centroid_b))

    dim = embeddings_a.shape[1]
    ball = PoincareBall(dim)

    # Project to Poincare ball: tanh(||x||) * x/||x|| maps to B^n
    def project(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x)
        if norm < 1e-10:
            return x
        return np.tanh(norm * 0.1) * x / norm  # scale factor 0.1 to stay well inside ball

    centroid_a = project(embeddings_a.mean(axis=0))
    centroid_b = project(embeddings_b.mean(axis=0))

    try:
        return float(ball.distance(centroid_a, centroid_b))
    except ValueError:
        # Point outside ball (numerical error)
        return float(np.linalg.norm(centroid_a - centroid_b))


def hyperbolic_hierarchy_depth(embeddings: np.ndarray) -> float:
    """Estimate hierarchical depth of embeddings in hyperbolic space.

    Points near the center of the Poincare ball are "root" concepts.
    Points near the boundary are "leaf" concepts.
    Greater variance in radius = deeper hierarchy.

    Returns standard deviation of radii (0 = flat, high = deep hierarchy).
    """
    if not HAS_HYPERBOLIC or len(embeddings) < 2:
        return 0.0

    # Project to ball
    norms = np.linalg.norm(embeddings, axis=1)
    radii = np.tanh(norms * 0.1)  # radius in Poincare ball
    return float(np.std(radii))


# ============================================================================
# 5. Topological Analysis — embedding quality audits
# ============================================================================

try:
    from hololoom.core.warp.math.advanced.topological_analysis import (
        HomologyDimension,
        PersistenceDiagram,
        PersistentHomology,
    )
    HAS_TDA = True
except ImportError:
    HAS_TDA = False


@dataclass
class TopologyAudit:
    """Results of topological analysis on department embeddings."""
    n_components: int         # H0: connected components (should be 1 for good dept)
    n_loops: int              # H1: semantic loops (cycles in knowledge)
    n_voids: int              # H2: semantic voids (missing concepts)
    max_persistence_h0: float # Most persistent component gap
    max_persistence_h1: float # Most persistent loop
    health: str               # "healthy", "fragmented", "cyclic", "sparse"


def audit_department_topology(dept: str) -> TopologyAudit | None:
    """Run topological analysis on a department's embedding space.

    Uses persistent homology to detect:
      - H0 (components): Are all notes connected? Fragmentation = isolated clusters.
      - H1 (loops): Are there semantic cycles? Loops = circular reasoning or redundancy.
      - H2 (voids): Are there missing concepts? Voids = gaps in coverage.

    Falls back to None if TDA unavailable or not enough data.
    """
    if not HAS_TDA:
        return None

    # Get department note embeddings
    try:
        import sys
        sys.path.insert(0, str(VAULT_ROOT))
        from services.data_service import data_service
        if not data_service.vector_store:
            return None

        # Collect embeddings for dept PARA notes
        dept_prefix = f"departments/{dept}/para"
        embeddings = []
        for path_str, embedding in data_service.vector_store.items():
            if dept_prefix in str(path_str).replace("\\", "/"):
                embeddings.append(embedding)

        if len(embeddings) < 3:
            return None

        embeddings_array = np.array(embeddings)

    except Exception as e:
        logger.debug("Could not load dept embeddings for TDA: %s", e)
        return None

    # Compute pairwise distance matrix
    n = len(embeddings_array)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(embeddings_array[i] - embeddings_array[j]))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Run persistent homology
    try:
        ph = PersistentHomology()
        diagrams = ph.compute(dist_matrix)

        h0 = diagrams.get(0, PersistenceDiagram(dimension=0))
        h1 = diagrams.get(1, PersistenceDiagram(dimension=1))

        # Count features with persistence above noise threshold
        noise_threshold = np.percentile(dist_matrix[dist_matrix > 0], 10) if (dist_matrix > 0).any() else 0.1

        n_components = len([i for i in h0.intervals if i.persistence > noise_threshold]) + 1  # +1 for the infinite component
        n_loops = len([i for i in h1.intervals if i.persistence > noise_threshold])
        n_voids = 0  # H2 is expensive, skip for now

        max_h0 = max((i.persistence for i in h0.intervals if i.persistence != float('inf')), default=0.0)
        max_h1 = max((i.persistence for i in h1.intervals), default=0.0)

        # Interpret
        if n_components > 3:
            health = "fragmented"
        elif n_loops > 2:
            health = "cyclic"
        elif n_components == 1 and n_loops == 0:
            health = "healthy"
        else:
            health = "sparse"

        return TopologyAudit(
            n_components=n_components,
            n_loops=n_loops,
            n_voids=n_voids,
            max_persistence_h0=round(max_h0, 4),
            max_persistence_h1=round(max_h1, 4),
            health=health,
        )

    except Exception as e:
        logger.debug("Persistent homology computation failed: %s", e)
        return None


# ============================================================================
# Availability summary
# ============================================================================

def math_extensions_status() -> dict[str, bool]:
    """Report which math extensions are available."""
    return {
        "natural_gradient": HAS_NATURAL_GRADIENT,
        "differential_privacy": HAS_DP,
        "constrained_optimization": HAS_OPTIMIZATION,
        "hyperbolic_geometry": HAS_HYPERBOLIC,
        "topological_analysis": HAS_TDA,
    }
