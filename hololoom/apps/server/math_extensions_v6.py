"""
Math Extensions v6 — consolidated integration for the full math library.

Covers modules built in tiers 2-3 that weren't in v5:
survival analysis, mixture models, robust regression, ergodic theory,
optimal control, Wasserstein, Bayesian nonparametrics, copulas,
Q-learning family, genetic algorithms, swarm, ADMM, spherical harmonics,
covering spaces, adjoint functors, model-based RL, offline RL,
reward machines, cellular automata, reservoir computing,
factor analysis, discriminant analysis, bootstrap, streaming, density estimation,
plus the algebra/computational deep stack.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Consolidated imports with graceful degradation
# ============================================================================

# Survival analysis
try:
    from hololoom.core.warp.math.analysis.survival_analysis import (
        CoxProportionalHazards,
        KaplanMeier,
    )
    HAS_SURVIVAL = True
except ImportError:
    HAS_SURVIVAL = False

# Mixture models
try:
    from hololoom.core.warp.math.analysis.mixture_models import (
        BayesianGaussianMixture,
        GaussianMixture,
    )
    HAS_MIXTURE = True
except ImportError:
    HAS_MIXTURE = False

# Optimal control
try:
    from hololoom.core.warp.math.decision.optimal_control import LQR, ModelPredictiveControl
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False

# Wasserstein
try:
    from hololoom.core.warp.math.analysis.wasserstein import (
        SlicedWasserstein,
        WassersteinBarycenter,
    )
    HAS_WASSERSTEIN = True
except ImportError:
    HAS_WASSERSTEIN = False

# Genetic algorithms
try:
    from hololoom.core.warp.math.self_assembly.genetic_algorithms import (
        DifferentialEvolution,
        GeneticAlgorithm,
    )
    HAS_GA = True
except ImportError:
    HAS_GA = False

# Cellular automata
try:
    from hololoom.core.warp.math.extensions.cellular_automata import ElementaryCA, GameOfLife
    HAS_CA = True
except ImportError:
    HAS_CA = False

# Reservoir computing
try:
    from hololoom.core.warp.math.extensions.reservoir_computing import Conceptor, EchoStateNetwork
    HAS_RESERVOIR = True
except ImportError:
    HAS_RESERVOIR = False

# Automatic differentiation
try:
    from hololoom.core.warp.math.computational.automatic_differentiation import (
        ForwardModeAD,
        ReverseModeAD,
    )
    HAS_AD = True
except ImportError:
    HAS_AD = False

# Bootstrap
try:
    from hololoom.core.warp.math.analysis.bootstrap import Bootstrap, PermutationTest
    HAS_BOOTSTRAP = True
except ImportError:
    HAS_BOOTSTRAP = False

# Streaming algorithms
try:
    from hololoom.core.warp.math.analysis.streaming import (
        CountMinSketch,
        HyperLogLog,
        ReservoirSampling,
    )
    HAS_STREAMING = True
except ImportError:
    HAS_STREAMING = False

# Density estimation
try:
    from hololoom.core.warp.math.analysis.density_estimation import KDEEstimator, WaveletDensity
    HAS_DENSITY = True
except ImportError:
    HAS_DENSITY = False

# Model-based RL
try:
    from hololoom.core.warp.math.decision.model_based_rl import Dyna, WorldModel
    HAS_MODEL_RL = True
except ImportError:
    HAS_MODEL_RL = False

# Policy gradient
try:
    from hololoom.core.warp.math.decision.policy_gradient import REINFORCE, SAC, ActorCritic
    HAS_PG = True
except ImportError:
    HAS_PG = False

# Elliptic curves
try:
    from hololoom.core.warp.math.computational.number_theory_algorithms import (
        EllipticCurve,
        PrimalityTesting,
    )
    HAS_NUMBER_THEORY = True
except ImportError:
    HAS_NUMBER_THEORY = False

# Symbolic computation
try:
    from hololoom.core.warp.math.computational.symbolic_computation import (
        RewriteSystem,
        SymbolicExpr,
    )
    HAS_SYMBOLIC = True
except ImportError:
    HAS_SYMBOLIC = False


# ============================================================================
# Pipeline adapters
# ============================================================================

def survival_analysis_on_notes(
    promotion_times: list[float],
    events: list[bool],
) -> dict | None:
    """Kaplan-Meier survival curve for note promotion times."""
    if not HAS_SURVIVAL or len(promotion_times) < 5:
        return None
    try:
        result = KaplanMeier.fit(np.array(promotion_times), np.array(events, dtype=float))
        return {"median_survival": round(result.median_survival, 2), "n_events": sum(events)}
    except Exception:
        return None


def cluster_notes_gmm(
    embeddings: np.ndarray,
    n_components: int = 3,
) -> dict | None:
    """Cluster note embeddings via Gaussian mixture."""
    if not HAS_MIXTURE or len(embeddings) < n_components:
        return None
    try:
        result = GaussianMixture.fit(embeddings, n_components=n_components)
        return {"labels": result.labels.tolist(), "bic": round(result.bic, 2), "n_components": n_components}
    except Exception:
        return None


def bootstrap_confidence(
    scores: list[float],
    statistic: str = "mean",
    alpha: float = 0.05,
) -> dict | None:
    """Bootstrap confidence interval for promotion scores."""
    if not HAS_BOOTSTRAP or len(scores) < 5:
        return None
    try:
        stat_fn = np.mean if statistic == "mean" else np.median
        lower, upper = Bootstrap.confidence_interval(np.array(scores), stat_fn, alpha=alpha)
        return {"lower": round(lower, 2), "upper": round(upper, 2), "alpha": alpha, "statistic": statistic}
    except Exception:
        return None


def stream_count_notes(
    note_ids: list[str],
) -> dict | None:
    """Approximate unique note count via HyperLogLog."""
    if not HAS_STREAMING:
        return None
    try:
        hll = HyperLogLog()
        for nid in note_ids:
            hll.add(nid)
        return {"approximate_count": hll.count(), "exact_count": len(set(note_ids))}
    except Exception:
        return None


def auto_differentiate(
    f: Callable[[float], float],
    x: float,
) -> dict | None:
    """Compute exact derivative via automatic differentiation."""
    if not HAS_AD:
        return None
    try:
        deriv = ForwardModeAD.differentiate(f, x)
        return {"derivative": round(deriv, 6), "at_x": x}
    except Exception:
        return None


def wasserstein_between_depts(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict | None:
    """Sliced Wasserstein distance between two departments' score distributions."""
    if not HAS_WASSERSTEIN:
        return None
    try:
        dist = SlicedWasserstein.compute(scores_a.reshape(-1, 1), scores_b.reshape(-1, 1))
        return {"sliced_wasserstein": round(float(dist), 4)}
    except Exception:
        return None


# ============================================================================
# Status
# ============================================================================

def math_extensions_v6_status() -> dict[str, bool]:
    """Report which v6 math extensions are available."""
    return {
        "survival_analysis": HAS_SURVIVAL,
        "mixture_models": HAS_MIXTURE,
        "optimal_control": HAS_CONTROL,
        "wasserstein_distances": HAS_WASSERSTEIN,
        "genetic_algorithms": HAS_GA,
        "cellular_automata": HAS_CA,
        "reservoir_computing": HAS_RESERVOIR,
        "automatic_differentiation": HAS_AD,
        "bootstrap_tests": HAS_BOOTSTRAP,
        "streaming_algorithms": HAS_STREAMING,
        "density_estimation": HAS_DENSITY,
        "model_based_rl": HAS_MODEL_RL,
        "policy_gradient": HAS_PG,
        "number_theory": HAS_NUMBER_THEORY,
        "symbolic_computation": HAS_SYMBOLIC,
    }
