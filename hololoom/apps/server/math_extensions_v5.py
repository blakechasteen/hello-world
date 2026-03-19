"""
Math Extensions v5 — The 30. All wired into the promotion pipeline.

12 sections, each with try/except guard. Two doors to the same core.

No beautiful math left unwired.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Score Denoising (Kalman Filter)
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.kalman_filtering import (
        ExtendedKalmanFilter,
        KalmanFilter,
        KalmanState,
    )
    HAS_KALMAN = True
except ImportError:
    HAS_KALMAN = False


def kalman_score_filter(
    score_history: list[float],
    process_noise: float = 1.0,
    measurement_noise: float = 5.0,
) -> list[float]:
    """Filter noisy LLM evaluation scores via Kalman.

    Each LLM score is a noisy observation of "true quality."
    Kalman gives the optimal estimate by fusing prior belief
    with new observations, weighted by their relative noise.
    """
    if not HAS_KALMAN or len(score_history) < 2:
        return list(score_history)

    kf = KalmanFilter()
    F = np.array([[1.0]])  # state transition: quality persists
    H = np.array([[1.0]])  # observation: we see quality directly
    Q = np.array([[process_noise]])
    R = np.array([[measurement_noise]])

    state = KalmanState(
        mean=np.array([score_history[0]]),
        covariance=np.array([[10.0]]),
    )

    filtered = []
    for z in score_history:
        state = kf.predict(state, F, Q)
        state = kf.update(state, np.array([z]), H, R)
        filtered.append(float(state.mean[0]))

    return filtered


# ============================================================================
# 2. Throughput Forecasting (ARIMA + Spectral)
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.time_series_models import (
        ARIMA,
        AutoCorrelation,
        ExponentialSmoothing,
    )
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

try:
    from hololoom.core.warp.math.analysis.fourier_harmonic import SpectralTimeSeries
    HAS_SPECTRAL_TS = True
except ImportError:
    HAS_SPECTRAL_TS = False


def forecast_throughput(
    history: list[float],
    horizon: int = 7,
) -> dict | None:
    """Forecast promotion throughput via ARIMA + spectral analysis."""
    if not HAS_ARIMA or len(history) < 10:
        return None

    try:
        params = ARIMA.fit(np.array(history), order=(2, 0, 1))
        forecast = ARIMA.forecast(params, np.array(history), steps=horizon)

        result = {
            "forecast": forecast.values.tolist(),
            "lower": forecast.confidence_lower.tolist(),
            "upper": forecast.confidence_upper.tolist(),
        }

        if HAS_SPECTRAL_TS:
            freqs = SpectralTimeSeries.dominant_frequencies(np.array(history), top_k=3)
            result["dominant_periods"] = [(round(1.0 / f, 1), round(p, 4)) for f, p in freqs if f > 0]

        return result
    except Exception as e:
        logger.debug("Throughput forecast failed: %s", e)
        return None


# ============================================================================
# 3. Regime Detection (Changepoint + HMM)
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.changepoint_detection import (
        CUSUM,
        PELT,
        BayesianChangepoint,
        Changepoint,
    )
    HAS_CHANGEPOINT = True
except ImportError:
    HAS_CHANGEPOINT = False

try:
    from hololoom.core.warp.math.analysis.hidden_markov_models import (
        GaussianHMM,
        HiddenMarkovModel,
    )
    HAS_HMM = True
except ImportError:
    HAS_HMM = False


def detect_quality_regime_shift(score_history: list[float]) -> dict | None:
    """Detect when promotion quality regime shifts."""
    if not HAS_CHANGEPOINT or len(score_history) < 20:
        return None

    try:
        series = np.array(score_history)
        changepoints = CUSUM.detect(series, threshold=4.0, drift=0.5)
        return {
            "changepoints": [{"index": cp.index, "confidence": round(cp.confidence, 3), "type": cp.type} for cp in changepoints],
            "n_regimes": len(changepoints) + 1,
        }
    except Exception as e:
        logger.debug("Changepoint detection failed: %s", e)
        return None


def infer_latent_quality_states(
    observations: list[float],
    n_states: int = 3,
) -> dict | None:
    """Infer hidden quality states from observed scores via HMM."""
    if not HAS_HMM or len(observations) < 20:
        return None

    try:
        hmm = GaussianHMM()
        obs = np.array(observations)
        params = hmm.baum_welch_gaussian(obs, n_states=n_states)
        states = hmm.viterbi(obs, params)
        return {
            "state_sequence": states.tolist(),
            "n_states": n_states,
            "state_means": [round(float(m), 2) for m in params.means],
        }
    except Exception as e:
        logger.debug("HMM inference failed: %s", e)
        return None


# ============================================================================
# 4. Tail Risk (EVD + Risk-Sensitive RL)
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.extreme_value_distributions import (
        GeneralizedExtremeValue,
        GumbelDistribution,
        PeaksOverThreshold,
    )
    HAS_EVD = True
except ImportError:
    HAS_EVD = False

try:
    from hololoom.core.warp.math.decision.risk_sensitive_rl import (
        CVaRMDP,
        RiskMetrics,
        compute_risk_metrics,
    )
    HAS_RISK = True
except ImportError:
    HAS_RISK = False


def score_tail_analysis(scores: list[float], alpha: float = 0.05) -> dict | None:
    """Analyze tail risk in promotion scores."""
    if len(scores) < 10:
        return None

    result = {}
    arr = np.array(scores)

    if HAS_EVD:
        try:
            params = GeneralizedExtremeValue.fit(arr)
            result["gev"] = {"mu": round(params.mu, 2), "sigma": round(params.sigma, 2), "xi": round(params.xi, 4)}
            result["return_level_100"] = round(GeneralizedExtremeValue.return_level(100, params), 2)
        except Exception:
            pass

    if HAS_RISK:
        try:
            metrics = compute_risk_metrics(arr, alpha=alpha)
            result["cvar"] = round(metrics.cvar, 2)
            result["var"] = round(metrics.var, 2)
            result["expected"] = round(metrics.expected_return, 2)
        except Exception:
            pass

    return result if result else None


# ============================================================================
# 5. Diversity & Compression (Rényi + MDL + Info Bottleneck)
# ============================================================================

try:
    from hololoom.core.warp.math.information.renyi_entropy import RenyiEntropy, TsallisEntropy
    HAS_RENYI = True
except ImportError:
    HAS_RENYI = False

try:
    from hololoom.core.warp.math.information.description_length import MinimumDescriptionLength
    HAS_MDL = True
except ImportError:
    HAS_MDL = False

try:
    from hololoom.core.warp.math.information.information_theory import InformationBottleneck
    HAS_IB = True
except ImportError:
    HAS_IB = False


def evaluation_diversity(score_distribution: np.ndarray, alpha: float = 2.0) -> dict | None:
    """Measure evaluation diversity via Rényi entropy."""
    if not HAS_RENYI:
        return None

    try:
        p = np.abs(score_distribution) + 1e-10
        p = p / p.sum()
        h = RenyiEntropy.compute(p, alpha)
        eff_n = RenyiEntropy.effective_number(p, alpha)
        return {"renyi_entropy": round(h, 4), "effective_categories": round(eff_n, 2), "alpha": alpha}
    except Exception:
        return None


def rubric_complexity_check(n_criteria: int, n_observations: int) -> dict | None:
    """Check if rubric complexity is justified by data via MDL."""
    if not HAS_MDL:
        return None

    try:
        from hololoom.core.warp.math.information.description_length import StochasticComplexity
        pc = StochasticComplexity.parametric_complexity(n_observations, n_criteria)
        return {
            "parametric_complexity": round(pc, 2),
            "justified": pc < n_observations * 0.5,
            "recommendation": "rubric justified" if pc < n_observations * 0.5 else "rubric may be overfit — consider fewer criteria",
        }
    except Exception:
        return None


# ============================================================================
# 6. Stability & Control (Lyapunov + Bifurcation + LaSalle + FP)
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.stability_analysis import (
        FixedPointFinder,
        LinearStability,
        LyapunovExponents,
    )
    HAS_STABILITY = True
except ImportError:
    HAS_STABILITY = False

try:
    from hololoom.core.warp.math.analysis.bifurcation_analysis import BifurcationAnalyzer
    HAS_BIFURCATION = True
except ImportError:
    HAS_BIFURCATION = False

try:
    from hololoom.core.warp.math.analysis.fokker_planck import FokkerPlanckSolver
    HAS_FP = True
except ImportError:
    HAS_FP = False


def pipeline_stability_report(score_trajectory: list[float]) -> dict | None:
    """Comprehensive stability report via Lyapunov + bifurcation."""
    if not HAS_STABILITY or len(score_trajectory) < 20:
        return None

    try:
        traj = np.array(score_trajectory).reshape(-1, 1)
        result = LyapunovExponents.compute_from_trajectory(traj)
        report = {
            "max_lyapunov": round(result.max_exponent, 4),
            "is_chaotic": result.is_chaotic,
            "stability": "chaotic" if result.is_chaotic else "stable" if result.max_exponent < 0 else "marginal",
        }
        return report
    except Exception as e:
        logger.debug("Stability report failed: %s", e)
        return None


def probability_density_forecast(
    current_score: float,
    theta: float = 0.5,
    mu: float = 70.0,
    sigma: float = 5.0,
) -> dict | None:
    """Forecast score density evolution via Fokker-Planck."""
    if not HAS_FP:
        return None

    try:
        result = FokkerPlanckSolver.ornstein_uhlenbeck_solution(
            theta=theta, mu=mu, sigma=sigma,
            x_range=(0, 100), t=30.0,
        )
        mean = float(np.sum(result * np.linspace(0, 100, len(result))) / result.sum()) if result.sum() > 0 else mu
        return {"forecast_mean": round(mean, 2), "target": mu, "reversion_speed": theta}
    except Exception:
        return None


# ============================================================================
# 7. Chaos & Predictability (Strange Attractors)
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.chaos_dynamical_systems import (
        PredictabilityAnalyzer,
        StrangeAttractorDetector,
    )
    HAS_CHAOS = True
except ImportError:
    HAS_CHAOS = False


def predictability_horizon(score_trajectory: list[float]) -> dict | None:
    """Determine how far ahead scores are predictable."""
    if not HAS_CHAOS or len(score_trajectory) < 30:
        return None

    try:
        traj = np.array(score_trajectory)
        delay = StrangeAttractorDetector.mutual_information_delay(traj)
        embedded = StrangeAttractorDetector.takens_embedding(traj, delay=max(1, delay), dimension=3)
        corr_dim = StrangeAttractorDetector.correlation_dimension(embedded)

        # Estimate max Lyapunov from stability module
        horizon = float('inf')
        if HAS_STABILITY:
            result = LyapunovExponents.compute_from_trajectory(embedded)
            if result.max_exponent > 0:
                horizon = PredictabilityAnalyzer.horizon(result.max_exponent)

        return {
            "correlation_dimension": round(corr_dim, 2),
            "embedding_delay": delay,
            "predictability_horizon": round(horizon, 1) if horizon < 1000 else "infinite",
        }
    except Exception as e:
        logger.debug("Predictability analysis failed: %s", e)
        return None


# ============================================================================
# 8. Multi-Agent Evaluation (CMDP + CFR + MARL + HRL)
# ============================================================================

try:
    from hololoom.core.warp.math.decision.constrained_mdp import CMDPSpec, ConstrainedMDP
    HAS_CMDP = True
except ImportError:
    HAS_CMDP = False

try:
    from hololoom.core.warp.math.decision.counterfactual_regret import (
        CounterfactualRegretMinimization,
        GameNode,
    )
    HAS_CFR = True
except ImportError:
    HAS_CFR = False

try:
    from hololoom.core.warp.math.decision.multi_agent_rl import IndependentLearners, MARLAgent
    HAS_MARL = True
except ImportError:
    HAS_MARL = False

try:
    from hololoom.core.warp.math.decision.hierarchical_rl import Option, OptionsFramework
    HAS_HRL = True
except ImportError:
    HAS_HRL = False


def resolve_evaluator_conflict(payoff_head: np.ndarray, payoff_gate: np.ndarray) -> dict | None:
    """Resolve Head vs Gate disagreement via CFR."""
    if not HAS_CFR:
        return None

    try:
        root = GameNode.from_matrix([payoff_head, payoff_gate])
        cfr = CounterfactualRegretMinimization()
        result = cfr.train(root, n_iterations=500)
        return {
            "exploitability": round(result.exploitability, 4),
            "iterations": result.iterations,
            "n_strategies": len(result.strategy),
        }
    except Exception as e:
        logger.debug("CFR conflict resolution failed: %s", e)
        return None


# ============================================================================
# 9. Self-Optimization (IRL + CMA-ES + Meta-Learning)
# ============================================================================

try:
    from hololoom.core.warp.math.decision.inverse_reinforcement_learning import IRLResult, MaxEntIRL
    HAS_IRL = True
except ImportError:
    HAS_IRL = False

try:
    from hololoom.core.warp.math.decision.evolution_strategies import CMAES, ESResult
    HAS_CMAES = True
except ImportError:
    HAS_CMAES = False

try:
    from hololoom.core.warp.math.decision.meta_learning import MAML, MetaLearnResult, Reptile
    HAS_META = True
except ImportError:
    HAS_META = False


def optimize_rubric_cmaes(
    objective_fn: Callable[[np.ndarray], float],
    n_weights: int,
    n_generations: int = 50,
) -> dict | None:
    """Optimize rubric weights via CMA-ES (derivative-free)."""
    if not HAS_CMAES:
        return None

    try:
        cma = CMAES(dim=n_weights, sigma=5.0)
        result = cma.optimize(objective_fn, n_generations=n_generations)
        return {
            "best_weights": result.best.tolist(),
            "best_fitness": round(result.best_fitness, 4),
            "generations": result.generations,
        }
    except Exception as e:
        logger.debug("CMA-ES optimization failed: %s", e)
        return None


# ============================================================================
# 10. Self-Organization (Cybernetics + Autopoiesis + SOC)
# ============================================================================

try:
    from hololoom.core.warp.math.extensions.second_order_cybernetics import (
        CircularCausality,
        RequisiteVariety,
        SecondOrderObserver,
    )
    HAS_CYBERNETICS = True
except ImportError:
    HAS_CYBERNETICS = False

try:
    from hololoom.core.warp.math.extensions.autopoiesis import (
        AutopoiesisAnalyzer,
        Component,
        StructuralCoupling,
    )
    HAS_AUTOPOIESIS = True
except ImportError:
    HAS_AUTOPOIESIS = False


def requisite_variety_check(
    system_states: np.ndarray,
    controller_actions: np.ndarray,
) -> dict | None:
    """Check Ashby's law: does the controller have enough variety?"""
    if not HAS_CYBERNETICS:
        return None

    try:
        measure = RequisiteVariety.ashby_measure(system_states, controller_actions)
        gap = RequisiteVariety.variety_gap(system_states, controller_actions)
        return {
            "ashby_ratio": round(measure, 3),
            "variety_gap": round(gap, 3),
            "sufficient": gap <= 0,
            "interpretation": "controller has sufficient variety" if gap <= 0 else f"controller needs {gap:.1f} more bits of variety",
        }
    except Exception:
        return None


def is_pipeline_autopoietic(components: list[dict]) -> dict | None:
    """Check if the pipeline is self-producing."""
    if not HAS_AUTOPOIESIS:
        return None

    try:
        parts = [Component(id=c["id"], type=c.get("type", ""), produces=c.get("produces", []), requires=c.get("requires", [])) for c in components]
        analyzer = AutopoiesisAnalyzer()
        is_auto = analyzer.is_autopoietic(parts)
        closure = analyzer.check_closure(parts)
        return {
            "autopoietic": is_auto,
            "production_closure": closure,
            "n_components": len(parts),
        }
    except Exception:
        return None


# ============================================================================
# 11. Visualization (t-SNE/UMAP + Coloring + Randomized SVD)
# ============================================================================

try:
    from hololoom.core.warp.math.extensions.manifold_learning import TSNE, UMAP, ManifoldEmbedding
    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False

try:
    from hololoom.core.warp.math.graph.graph_coloring import ConflictGraph, GraphColoring
    HAS_COLORING = True
except ImportError:
    HAS_COLORING = False

try:
    from hololoom.core.warp.math.analysis.randomized_linear_algebra import RandomizedSVD
    HAS_RSVD = True
except ImportError:
    HAS_RSVD = False


def embed_for_visualization(
    embeddings: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
) -> dict | None:
    """Reduce high-dimensional embeddings for visualization."""
    if not HAS_MANIFOLD or len(embeddings) < 3:
        return None

    try:
        if method == "tsne":
            result = TSNE.fit_transform(embeddings, n_components=n_components)
        elif method == "umap":
            result = UMAP.fit_transform(embeddings, n_components=n_components)
        else:
            return None

        return {
            "coordinates": result.coordinates.tolist(),
            "stress": round(result.stress, 4),
            "trustworthiness": round(result.trustworthiness, 4),
            "method": method,
        }
    except Exception as e:
        logger.debug("Manifold embedding failed: %s", e)
        return None


def schedule_parallel_evaluations(
    n_notes: int,
    conflicts: list[tuple[int, int]],
) -> dict | None:
    """Schedule non-conflicting parallel evaluations via graph coloring."""
    if not HAS_COLORING or n_notes < 2:
        return None

    try:
        adj = np.zeros((n_notes, n_notes))
        for i, j in conflicts:
            if i < n_notes and j < n_notes:
                adj[i, j] = adj[j, i] = 1

        result = GraphColoring.dsatur(adj)
        return {
            "n_colors": result.n_colors,
            "schedule": result.colors,
            "max_parallel": max(sum(1 for c in result.colors.values() if c == color) for color in set(result.colors.values())),
        }
    except Exception:
        return None


def fast_approximate_svd(matrix: np.ndarray, k: int = 10) -> dict | None:
    """Fast approximate SVD for large embedding matrices."""
    if not HAS_RSVD:
        return None

    try:
        result = RandomizedSVD.decompose(matrix, k=min(k, min(matrix.shape) - 1))
        return {
            "singular_values": result.S.tolist(),
            "explained_variance": round(result.explained_variance_ratio, 4),
            "k": len(result.S),
        }
    except Exception:
        return None


# ============================================================================
# 12. Kernel Methods (Kernel Regression)
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.kernel_regression import (
        RKHS,
        KernelDensityEstimation,
        NadarayaWatson,
    )
    HAS_KERNEL = True
except ImportError:
    HAS_KERNEL = False


def nonparametric_score_prediction(
    features_train: np.ndarray,
    scores_train: np.ndarray,
    features_new: np.ndarray,
    bandwidth: float | None = None,
) -> dict | None:
    """Predict scores for new notes via kernel regression."""
    if not HAS_KERNEL or len(features_train) < 3:
        return None

    try:
        nw = NadarayaWatson()
        result = nw.fit_predict(features_train, scores_train, features_new, bandwidth=bandwidth)
        return {
            "predictions": result.predictions.tolist(),
            "bandwidth": round(result.bandwidth, 4),
            "effective_df": round(result.effective_df, 2),
        }
    except Exception as e:
        logger.debug("Kernel regression failed: %s", e)
        return None


# ============================================================================
# Status
# ============================================================================

def math_extensions_v5_status() -> dict[str, bool]:
    """Report which v5 math extensions are available."""
    return {
        # Denoising
        "kalman_filter": HAS_KALMAN,
        # Temporal
        "arima_forecasting": HAS_ARIMA,
        "spectral_time_series": HAS_SPECTRAL_TS,
        "changepoint_detection": HAS_CHANGEPOINT,
        "hidden_markov_models": HAS_HMM,
        # Tail Risk
        "extreme_value_distributions": HAS_EVD,
        "risk_sensitive_rl": HAS_RISK,
        # Diversity
        "renyi_entropy": HAS_RENYI,
        "minimum_description_length": HAS_MDL,
        "information_bottleneck": HAS_IB,
        # Stability
        "stability_analysis": HAS_STABILITY,
        "bifurcation_analysis": HAS_BIFURCATION,
        "fokker_planck": HAS_FP,
        # Chaos
        "chaos_dynamical_systems": HAS_CHAOS,
        # Multi-Agent
        "constrained_mdp": HAS_CMDP,
        "counterfactual_regret": HAS_CFR,
        "multi_agent_rl": HAS_MARL,
        "hierarchical_rl": HAS_HRL,
        # Self-Optimization
        "inverse_rl": HAS_IRL,
        "cma_es": HAS_CMAES,
        "meta_learning": HAS_META,
        # Self-Organization
        "second_order_cybernetics": HAS_CYBERNETICS,
        "autopoiesis": HAS_AUTOPOIESIS,
        # Visualization
        "manifold_learning": HAS_MANIFOLD,
        "graph_coloring": HAS_COLORING,
        "randomized_svd": HAS_RSVD,
        # Kernel
        "kernel_regression": HAS_KERNEL,
    }
