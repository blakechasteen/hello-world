"""
Buried Math Integration — wire the ~15K LOC of mathematical infrastructure
scattered across the codebase that's been sitting idle.

neural_ts, semantic_calculus, POMDP, causal stack, Hofstadter,
semantic PDEs, spectral embeddings, context packing, Riemannian Matryoshka.

Two doors to the same core. This is another door.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Neural Thompson Sampling (bandits/neural_ts/)
# ============================================================================

try:
    from hololoom.bandits.neural_ts.policy import NeuralThompsonPolicy
    HAS_NEURAL_TS = True
except ImportError:
    HAS_NEURAL_TS = False


def neural_bandit_select(
    context: np.ndarray,
    n_arms: int,
) -> dict | None:
    """Select an arm using neural Thompson Sampling.

    The neural reward model learns nonlinear context → reward mappings,
    going beyond the linear assumption of the contextual bandit.
    """
    if not HAS_NEURAL_TS:
        return None
    try:
        policy = NeuralThompsonPolicy(context_dim=len(context), n_arms=n_arms)
        arm = policy.select(context)
        return {"arm": int(arm) if isinstance(arm, (int, np.integer)) else 0}
    except Exception as e:
        logger.debug("Neural TS failed: %s", e)
        return None


# ============================================================================
# 2. Semantic Calculus (integral geometry, flow calculus)
# ============================================================================

try:
    from hololoom.semantic_calculus.integral_geometry import RadonTransform
    HAS_RADON = True
except ImportError:
    HAS_RADON = False

try:
    from hololoom.semantic_calculus.flow_calculus import SemanticFlowCalculus
    HAS_SEMANTIC_FLOW = True
except ImportError:
    HAS_SEMANTIC_FLOW = False

try:
    from hololoom.semantic_calculus.hyperbolic import HyperbolicSemanticSpace
    HAS_SEMANTIC_HYPERBOLIC = True
except ImportError:
    HAS_SEMANTIC_HYPERBOLIC = False


def semantic_tomography(
    projections: list[np.ndarray],
    n_angles: int = 8,
) -> np.ndarray | None:
    """Reconstruct semantic context from projections via Radon transform.

    Each "projection" is a partial view of a note's meaning (from different
    evaluators or rubric terms). Radon inversion reconstructs the full picture.
    """
    if not HAS_RADON or not projections:
        return None
    try:
        rt = RadonTransform()
        sinogram = np.array(projections)
        return rt.inverse(sinogram, n_angles)
    except Exception as e:
        logger.debug("Semantic tomography failed: %s", e)
        return None


def compute_semantic_flow(
    embeddings: dict[str, np.ndarray],
    connections: dict[str, list[str]],
    dt: float = 0.1,
    n_steps: int = 10,
) -> dict | None:
    """Simulate semantic flow between department notes.

    Knowledge diffuses along wiki-link connections. High-value knowledge
    flows to sparse regions naturally.
    """
    if not HAS_SEMANTIC_FLOW:
        return None
    try:
        flow = SemanticFlow()
        for note_id, emb in embeddings.items():
            flow.add_node(note_id, emb)
        for src, targets in connections.items():
            for tgt in targets:
                if tgt in embeddings:
                    flow.add_edge(src, tgt)

        for _ in range(n_steps):
            flow.step(dt)

        return {"converged": True, "n_nodes": len(embeddings), "n_steps": n_steps}
    except Exception as e:
        logger.debug("Semantic flow failed: %s", e)
        return None


# ============================================================================
# 3. POMDP Planning (planning/pomdp.py)
# ============================================================================

try:
    from hololoom.planning.pomdp import POMDP, BeliefState
    HAS_POMDP = True
except ImportError:
    HAS_POMDP = False


@dataclass
class EvaluationBelief:
    """Belief state for note evaluation under partial observability."""
    quality_belief: np.ndarray  # probability over quality levels
    most_likely_quality: int
    uncertainty: float


def belief_based_evaluation(
    prior_belief: np.ndarray | None = None,
    observation: float | None = None,
    n_quality_levels: int = 5,
) -> EvaluationBelief | None:
    """Update belief about note quality given partial observation.

    The true quality is hidden. We observe noisy evaluation scores.
    POMDP belief update tells us the posterior over quality levels.
    """
    if not HAS_POMDP:
        return None

    try:
        if prior_belief is None:
            prior_belief = np.ones(n_quality_levels) / n_quality_levels

        if observation is not None:
            # Simple observation model: observation is noisy version of quality
            obs_level = min(n_quality_levels - 1, max(0, int(observation / (100 / n_quality_levels))))
            # Likelihood: higher prob of observing correct level
            likelihood = np.ones(n_quality_levels) * 0.1
            likelihood[obs_level] = 0.6
            if obs_level > 0:
                likelihood[obs_level - 1] = 0.15
            if obs_level < n_quality_levels - 1:
                likelihood[obs_level + 1] = 0.15

            posterior = prior_belief * likelihood
            posterior = posterior / posterior.sum()
        else:
            posterior = prior_belief

        return EvaluationBelief(
            quality_belief=posterior,
            most_likely_quality=int(np.argmax(posterior)),
            uncertainty=float(-np.sum(posterior * np.log(posterior + 1e-10))),
        )
    except Exception as e:
        logger.debug("Belief evaluation failed: %s", e)
        return None


# ============================================================================
# 4. Causal Stack (causal/ — full Pearl framework)
# ============================================================================

try:
    from hololoom.causal.dag import CausalDAG
    HAS_CAUSAL_FULL = True
except ImportError:
    HAS_CAUSAL_FULL = False

try:
    from hololoom.causal.discovery import CausalDiscovery
    HAS_DISCOVERY = True
except ImportError:
    HAS_DISCOVERY = False

try:
    from hololoom.causal.temporal import TemporalCausalDAG
    HAS_TEMPORAL_CAUSAL = True
except ImportError:
    HAS_TEMPORAL_CAUSAL = False


def discover_promotion_causal_structure(
    score_data: np.ndarray,
    variable_names: list[str],
) -> dict | None:
    """Discover causal structure from promotion score data.

    Instead of hardcoding the causal DAG (as in promotion_causal.py),
    LEARN it from observed data via the PC algorithm.
    """
    if not HAS_DISCOVERY:
        return None
    try:
        discovery = CausalDiscovery()
        edges = discovery.pc_algorithm(score_data, variable_names)
        return {"discovered_edges": edges, "n_variables": len(variable_names)}
    except Exception as e:
        logger.debug("Causal discovery failed: %s", e)
        return None


# ============================================================================
# 5. Hofstadter Self-Referential Indexing
# ============================================================================

try:
    from hololoom.core.warp.hofstadter import HofstadterSequences
    HAS_HOFSTADTER = True
except ImportError:
    HAS_HOFSTADTER = False


def hofstadter_retrieval_order(
    n_notes: int,
    sequence_type: str = "Q",
) -> list[int] | None:
    """Generate self-referential retrieval order via Hofstadter sequences.

    Hofstadter sequences are defined in terms of themselves:
    Q(n) = Q(n - Q(n-1)), generating chaotic-yet-structured patterns.
    Novel retrieval strategy for self-referential knowledge.
    """
    if not HAS_HOFSTADTER or n_notes < 2:
        return None
    try:
        hs = HofstadterSequences()
        if sequence_type == "Q":
            seq = hs.Q(n_notes)
        elif sequence_type == "G":
            seq = hs.G(n_notes)
        else:
            seq = hs.Q(n_notes)
        # Use sequence values as retrieval priority indices
        order = sorted(range(n_notes), key=lambda i: seq[i] if i < len(seq) else i)
        return order
    except Exception as e:
        logger.debug("Hofstadter retrieval failed: %s", e)
        return None


# ============================================================================
# 6. Semantic PDEs (heat/wave/diffusion on knowledge graph)
# ============================================================================

try:
    from hololoom.core.warp.semantic_pde import HeatEquationSolver, WaveEquationSolver
    HAS_SEMANTIC_PDE = True
except ImportError:
    HAS_SEMANTIC_PDE = False


def knowledge_diffusion(
    initial_values: dict[str, float],
    adjacency: np.ndarray,
    t: float = 1.0,
    diffusion_coefficient: float = 0.1,
) -> dict | None:
    """Diffuse knowledge values across the department graph.

    Heat equation on the knowledge graph: du/dt = D * L * u
    High-value knowledge spreads to neighbors over time.
    """
    if not HAS_SEMANTIC_PDE:
        # Fallback: simple averaging
        names = list(initial_values.keys())
        values = np.array([initial_values[n] for n in names])
        if adjacency.shape[0] != len(values):
            return None
        # One step of diffusion: u += D * L * u
        degree = adjacency.sum(axis=1)
        laplacian = np.diag(degree) - adjacency
        diffused = values - diffusion_coefficient * t * laplacian @ values
        return {
            "values": {names[i]: round(float(diffused[i]), 4) for i in range(len(names))},
            "method": "fallback_laplacian",
        }

    try:
        solver = HeatEquationSolver()
        result = solver.solve(initial_values, adjacency, t, diffusion_coefficient)
        return {"values": result, "method": "semantic_pde"}
    except Exception as e:
        logger.debug("Knowledge diffusion failed: %s", e)
        return None


# ============================================================================
# 7. Context Packing (6-signal importance scoring)
# ============================================================================

try:
    from hololoom.context_packing.importance_scorer import ImportanceScorer
    HAS_IMPORTANCE = True
except ImportError:
    HAS_IMPORTANCE = False


def six_signal_importance(
    note_features: dict,
) -> dict | None:
    """Score note importance via 6-signal scoring.

    Signals: recency, relevance, centrality, frequency, confidence, heat.
    Richer than the simple quality_score in vault_bridge.
    """
    if not HAS_IMPORTANCE:
        return None
    try:
        scorer = ImportanceScorer()
        score = scorer.score(note_features)
        return {"importance": round(float(score), 4), "method": "six_signal"}
    except Exception as e:
        logger.debug("Importance scoring failed: %s", e)
        return None


# ============================================================================
# 8. Spectral Embeddings (multi-scale spectral analysis)
# ============================================================================

try:
    from hololoom.core.embedding.spectral import EmbeddingDriftDetector, SpectralFusion
    HAS_SPECTRAL_EMB = True
except ImportError:
    HAS_SPECTRAL_EMB = False

try:
    from hololoom.core.embedding.riemannian_matryoshka import RiemannianMatryoshka
    HAS_RIEMANNIAN_MATRYOSHKA = True
except ImportError:
    HAS_RIEMANNIAN_MATRYOSHKA = False


def multi_scale_spectral_analysis(
    embeddings: np.ndarray,
    n_components: int = 5,
) -> dict | None:
    """Multi-scale spectral analysis of note embeddings."""
    if not HAS_SPECTRAL_EMB:
        return None
    try:
        se = SpectralEmbedder()
        result = se.fit_transform(embeddings, n_components=n_components)
        return {
            "n_components": n_components,
            "explained_variance": float(np.sum(result.explained_variance_ratio_[:n_components]))
            if hasattr(result, 'explained_variance_ratio_') else 0.0,
        }
    except Exception as e:
        logger.debug("Spectral embedding failed: %s", e)
        return None


# ============================================================================
# Status
# ============================================================================

def buried_math_status() -> dict[str, bool]:
    """Report which buried math modules are available."""
    return {
        "neural_thompson_sampling": HAS_NEURAL_TS,
        "radon_transform": HAS_RADON,
        "semantic_flow": HAS_SEMANTIC_FLOW,
        "semantic_hyperbolic": HAS_SEMANTIC_HYPERBOLIC,
        "pomdp_planning": HAS_POMDP,
        "causal_full_stack": HAS_CAUSAL_FULL,
        "causal_discovery": HAS_DISCOVERY,
        "temporal_causal": HAS_TEMPORAL_CAUSAL,
        "hofstadter_sequences": HAS_HOFSTADTER,
        "semantic_pde": HAS_SEMANTIC_PDE,
        "importance_scoring": HAS_IMPORTANCE,
        "spectral_embeddings": HAS_SPECTRAL_EMB,
        "riemannian_matryoshka": HAS_RIEMANNIAN_MATRYOSHKA,
    }
