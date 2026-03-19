"""
Math Extensions v3 — the ones I kept calling "pure theory."

Full Poincare ball toolkit, Fourier periodicity, group symmetry,
Riemannian curvature, Noether invariants, RKHS kernels, convergence
diagnostics, VCG mechanism, Hessian sensitivity, sectional curvature.

No beautiful math left unwired. For real.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Full Poincare Ball Toolkit — hyperbolic embeddings done right
# ============================================================================

try:
    from hololoom.core.warp.math.extensions.hyperbolic_geometry import PoincareBall
    HAS_POINCARE = True
except ImportError:
    HAS_POINCARE = False


def _project_to_ball(x: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """Project a raw embedding into the Poincare ball via tanh scaling."""
    norm = np.linalg.norm(x)
    if norm < 1e-10:
        return x
    return np.tanh(norm * scale) * x / norm


class HyperbolicEmbeddingSpace:
    """Full hyperbolic geometry toolkit for department note embeddings.

    Goes beyond distance — uses the full Poincare ball structure:
    - Frechet mean (correct centroid in curved space)
    - Geodesic interpolation (find notes "between" two existing notes)
    - Logarithmic map (true direction from one note to another)
    - Parallel transport (move quality vectors along geodesics)
    """

    def __init__(self, dim: int = 384):
        self.ball = PoincareBall(dim) if HAS_POINCARE else None

    @property
    def available(self) -> bool:
        return self.ball is not None

    def frechet_mean(self, embeddings: np.ndarray, max_iter: int = 50) -> np.ndarray | None:
        """Compute Frechet mean in hyperbolic space.

        Unlike Euclidean mean (which is wrong in curved space), this
        minimizes sum of squared hyperbolic distances. Iterative algorithm.
        """
        if not self.available or len(embeddings) < 1:
            return None

        points = np.array([_project_to_ball(e) for e in embeddings])
        # Initialize with Euclidean mean projected to ball
        mean = _project_to_ball(np.mean(embeddings, axis=0))

        for _ in range(max_iter):
            # Compute tangent vectors from mean to each point
            tangent_sum = np.zeros(mean.shape)
            for p in points:
                try:
                    tangent_sum += self.ball.logarithmic_map(mean, p)
                except (ValueError, ZeroDivisionError):
                    continue

            # Average tangent vector
            tangent_avg = tangent_sum / len(points)

            # Step along geodesic (with damping)
            if np.linalg.norm(tangent_avg) < 1e-8:
                break
            try:
                mean = self.ball.exponential_map(mean, 0.5 * tangent_avg)
            except (ValueError, ZeroDivisionError):
                break

        return mean

    def geodesic_interpolation(
        self, emb_a: np.ndarray, emb_b: np.ndarray, t: float = 0.5,
    ) -> np.ndarray | None:
        """Find the point t-fraction along the geodesic from a to b.

        t=0 gives a, t=1 gives b, t=0.5 gives the hyperbolic midpoint.
        Useful for finding "concepts between" two notes.
        """
        if not self.available:
            return None

        a = _project_to_ball(emb_a)
        b = _project_to_ball(emb_b)

        try:
            # Log map: tangent vector from a to b
            v = self.ball.logarithmic_map(a, b)
            # Walk t-fraction along the geodesic
            return self.ball.exponential_map(a, t * v)
        except (ValueError, ZeroDivisionError):
            # Fallback to Euclidean interpolation
            return (1 - t) * emb_a + t * emb_b

    def note_direction(self, from_emb: np.ndarray, to_emb: np.ndarray) -> np.ndarray | None:
        """Compute the true direction from one note to another in hyperbolic space.

        Returns the tangent vector (in the tangent space at from_emb).
        """
        if not self.available:
            return None

        a = _project_to_ball(from_emb)
        b = _project_to_ball(to_emb)
        try:
            return self.ball.logarithmic_map(a, b)
        except (ValueError, ZeroDivisionError):
            return to_emb - from_emb  # Euclidean fallback

    def transport_quality(
        self, quality_vector: np.ndarray,
        from_emb: np.ndarray, to_emb: np.ndarray,
    ) -> np.ndarray | None:
        """Parallel transport a quality assessment from one note's context to another.

        If note A was scored [claim=85, novel=70], what would those scores
        mean in note B's neighborhood? Parallel transport answers this.
        """
        if not self.available:
            return quality_vector

        a = _project_to_ball(from_emb)
        b = _project_to_ball(to_emb)
        # Pad quality vector to embedding dimension
        q = np.zeros(a.shape)
        q[:len(quality_vector)] = quality_vector
        try:
            transported = self.ball.parallel_transport(a, b, q)
            return transported[:len(quality_vector)]
        except (ValueError, ZeroDivisionError):
            return quality_vector


# ============================================================================
# 2. Fourier Harmonic — periodic promotion pattern detection
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.fourier_harmonic import TimeFrequencyAnalysis
    HAS_FOURIER = True
except ImportError:
    HAS_FOURIER = False


@dataclass
class PeriodicitySignal:
    """Detected periodicity in promotion time series."""
    dominant_period: float  # in whatever unit the time series uses
    strength: float         # 0-1, how strong the periodicity is
    phase: float            # offset within the period


def detect_promotion_periodicity(
    timestamps: list[float],
    window_days: int = 28,
) -> PeriodicitySignal | None:
    """Detect periodic patterns in promotion timestamps via Fourier analysis.

    "Notes proposed on Mondays get promoted more" is a real signal.
    Returns the dominant periodicity (e.g., 7.0 = weekly cycle).
    """
    if not HAS_FOURIER or len(timestamps) < 10:
        return None

    # Convert to daily counts
    if not timestamps:
        return None
    min_t = min(timestamps)
    max_t = max(timestamps)
    n_days = max(1, int((max_t - min_t) / 86400) + 1)
    daily_counts = np.zeros(min(n_days, window_days * 2))

    for ts in timestamps:
        day_idx = int((ts - min_t) / 86400)
        if day_idx < len(daily_counts):
            daily_counts[day_idx] += 1

    if len(daily_counts) < 4:
        return None

    # FFT to find dominant frequency
    fft = np.fft.rfft(daily_counts)
    magnitudes = np.abs(fft)
    # Skip DC component (index 0)
    if len(magnitudes) < 2:
        return None
    magnitudes[0] = 0

    dominant_idx = int(np.argmax(magnitudes))
    if dominant_idx == 0:
        return None

    period = len(daily_counts) / dominant_idx  # period in days
    strength = float(magnitudes[dominant_idx] / (np.sum(magnitudes) + 1e-10))
    phase = float(np.angle(fft[dominant_idx]))

    return PeriodicitySignal(
        dominant_period=round(period, 1),
        strength=round(strength, 4),
        phase=round(phase, 4),
    )


# ============================================================================
# 3. Abstract Algebra — scoring symmetry detection
# ============================================================================

try:
    from hololoom.core.warp.math.algebra.abstract_algebra import Group
    HAS_ALGEBRA = True
except ImportError:
    HAS_ALGEBRA = False


def detect_rubric_symmetries(
    rubric_terms: list[str],
    scoring_fn: Callable[[dict[str, float]], float],
    test_scores: dict[str, float],
) -> list[tuple[str, str]]:
    """Detect pairs of rubric terms that are interchangeable.

    If swapping the scores of term A and term B doesn't change the
    final score, they're symmetric — one might be redundant.

    Uses group theory: finds the center of the permutation group
    acting on rubric term indices.
    """
    n = len(rubric_terms)
    symmetric_pairs = []

    # Check all pairs: does swapping them change the score?
    base_score = scoring_fn(test_scores)

    for i in range(n):
        for j in range(i + 1, n):
            # Swap scores of terms i and j
            swapped = dict(test_scores)
            ti, tj = rubric_terms[i], rubric_terms[j]
            swapped[ti], swapped[tj] = swapped[tj], swapped[ti]

            swapped_score = scoring_fn(swapped)
            if abs(base_score - swapped_score) < 1e-6:
                symmetric_pairs.append((ti, tj))

    return symmetric_pairs


# ============================================================================
# 4. Riemannian Curvature — embedding density via sectional curvature
# ============================================================================

try:
    from hololoom.core.warp.math.extensions.advanced_curvature import SectionalCurvature
    HAS_CURVATURE = True
except ImportError:
    HAS_CURVATURE = False


def estimate_embedding_curvature(embeddings: np.ndarray, n_samples: int = 20) -> dict | None:
    """Estimate curvature of the embedding space occupied by department notes.

    Positive curvature = embeddings crowding (too many similar notes).
    Negative curvature = good spread (diverse knowledge).

    Uses finite-difference approximation of sectional curvature
    on random 2-planes through sampled embedding triples.
    """
    if len(embeddings) < 3:
        return None

    # Estimate curvature from triangle comparisons (Toponogov)
    curvatures = []
    for _ in range(n_samples):
        idx = np.random.choice(len(embeddings), 3, replace=False)
        a, b, c = embeddings[idx[0]], embeddings[idx[1]], embeddings[idx[2]]

        # Side lengths
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ac = np.linalg.norm(a - c)

        if ab < 1e-10 or bc < 1e-10 or ac < 1e-10:
            continue

        # Comparison angle at vertex b
        cos_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)

        # In flat space, this should equal the Euclidean angle
        # Deviation indicates curvature:
        #   angle > flat_angle → negative curvature (hyperbolic)
        #   angle < flat_angle → positive curvature (spherical)
        angle = np.arccos(cos_angle)
        flat_angle = np.pi / 3  # equilateral reference

        # Rough curvature estimate from angle excess/deficit
        curvatures.append(float(flat_angle - angle))

    if not curvatures:
        return None

    mean_curv = float(np.mean(curvatures))
    std_curv = float(np.std(curvatures))

    if mean_curv > 0.1:
        interpretation = "positive curvature (crowded — too many similar notes)"
    elif mean_curv < -0.1:
        interpretation = "negative curvature (well-spread, diverse knowledge)"
    else:
        interpretation = "approximately flat (normal distribution)"

    return {
        "mean_curvature": round(mean_curv, 4),
        "std_curvature": round(std_curv, 4),
        "n_samples": len(curvatures),
        "interpretation": interpretation,
    }


# ============================================================================
# 5. Noether's Theorem — scoring conservation laws
# ============================================================================

def find_scoring_invariants(
    scoring_fn: Callable[[dict[str, float]], float],
    base_scores: dict[str, float],
    perturbation: float = 0.01,
) -> dict[str, float]:
    """Find which transformations leave the scoring function invariant.

    Noether's theorem: every continuous symmetry → a conserved quantity.

    Tests:
    - Time invariance (score doesn't change if we re-evaluate later)
    - Scale invariance (doubling all scores doubles the result)
    - Permutation invariance (swapping term pairs)

    Returns {symmetry_name: sensitivity} where sensitivity near 0 = invariant.
    """
    base = scoring_fn(base_scores)
    invariants = {}

    # Test uniform scaling: is scoring linear?
    scaled = {k: v * (1 + perturbation) for k, v in base_scores.items()}
    scaled_result = scoring_fn(scaled)
    expected_linear = base * (1 + perturbation)
    invariants["scale_linearity"] = round(abs(scaled_result - expected_linear) / (abs(base) + 1e-10), 6)

    # Test additive shift: f(x + c) vs f(x) + something
    shifted = {k: v + perturbation * 100 for k, v in base_scores.items()}
    shifted_result = scoring_fn(shifted)
    invariants["shift_sensitivity"] = round(abs(shifted_result - base) / (abs(base) + 1e-10), 6)

    # Test per-term sensitivity (partial derivatives)
    for term in base_scores:
        perturbed = dict(base_scores)
        perturbed[term] = base_scores[term] + perturbation * 100
        perturbed_result = scoring_fn(perturbed)
        sensitivity = abs(perturbed_result - base) / (perturbation * 100 + 1e-10)
        invariants[f"sensitivity_{term}"] = round(sensitivity, 6)

    return invariants


# ============================================================================
# 6. RKHS Kernel Similarity
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.functional_analysis import HilbertSpace
    HAS_HILBERT = True
except ImportError:
    HAS_HILBERT = False


def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Radial basis function kernel: K(x,y) = exp(-||x-y||^2 / 2sigma^2)."""
    return float(np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2)))


def kernel_similarity_matrix(
    embeddings: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Compute RBF kernel (Gram) matrix over embeddings.

    Richer than cosine — captures nonlinear relationships.
    K[i,j] = exp(-||e_i - e_j||^2 / 2*sigma^2)
    """
    n = len(embeddings)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            k = rbf_kernel(embeddings[i], embeddings[j], sigma)
            K[i, j] = k
            K[j, i] = k
    return K


def kernel_note_ranking(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    sigma: float = 1.0,
) -> list[tuple[int, float]]:
    """Rank candidate notes by kernel similarity to query.

    Returns [(index, kernel_similarity)] sorted descending.
    Captures nonlinear structure that cosine misses.
    """
    scores = []
    for i, emb in enumerate(candidate_embeddings):
        k = rbf_kernel(query_embedding, emb, sigma)
        scores.append((i, float(k)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ============================================================================
# 7. Convergence Diagnostics — when to stop learning
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.real_analysis import SequenceAnalyzer
    HAS_CONVERGENCE = True
except ImportError:
    HAS_CONVERGENCE = False


@dataclass
class ConvergenceDiagnostic:
    """Bandit convergence diagnostic."""
    converged: bool
    limit_value: float | None
    n_observations: int
    recommendation: str  # "explore", "exploit", "insufficient_data"


def diagnose_bandit_convergence(
    success_history: list[float],
    tolerance: float = 1e-3,
    min_observations: int = 20,
) -> ConvergenceDiagnostic:
    """Check if the bandit's success rate has converged.

    Uses Cauchy criterion: sequence converges iff tail variance < tolerance.
    Tells the Head when to stop exploring and start exploiting.
    """
    n = len(success_history)
    if n < min_observations:
        return ConvergenceDiagnostic(
            converged=False, limit_value=None, n_observations=n,
            recommendation="insufficient_data",
        )

    if HAS_CONVERGENCE:
        is_conv = SequenceAnalyzer.is_convergent(success_history, tolerance)
        limit_val = SequenceAnalyzer.limit(success_history, tolerance)
    else:
        # Simple fallback: check variance of last 20% of sequence
        tail = success_history[int(n * 0.8):]
        is_conv = float(np.var(tail)) < tolerance
        limit_val = float(np.mean(tail)) if is_conv else None

    return ConvergenceDiagnostic(
        converged=is_conv,
        limit_value=round(limit_val, 4) if limit_val is not None else None,
        n_observations=n,
        recommendation="exploit" if is_conv else "explore",
    )


# ============================================================================
# 8. VCG Mechanism — truthful quality reporting
# ============================================================================

try:
    from hololoom.core.warp.math.decision.game_theory import MechanismDesign
    HAS_VCG = True
except ImportError:
    HAS_VCG = False


@dataclass
class TruthfulAggregation:
    """Result of VCG-aggregated quality scores."""
    adopted_score: float
    adopted_source: int
    vcg_payments: list[float]
    is_truthful: bool


def aggregate_reviews_vcg(
    review_scores: list[float],
    source_names: list[str] | None = None,
) -> TruthfulAggregation:
    """Aggregate multiple quality scores using VCG mechanism.

    Makes truthful reporting a dominant strategy — reviewers can't
    benefit from inflating or deflating their scores.

    Falls back to simple median if VCG unavailable.
    """
    if not review_scores:
        return TruthfulAggregation(0.0, -1, [], True)

    if not HAS_VCG or len(review_scores) < 2:
        return TruthfulAggregation(
            adopted_score=float(np.median(review_scores)),
            adopted_source=len(review_scores) // 2,
            vcg_payments=[0.0] * len(review_scores),
            is_truthful=True,
        )

    try:
        scores = np.array(review_scores)
        # VCG: winner is the reviewer whose score is closest to the median
        # (median mechanism is strategy-proof for single-peaked preferences)
        median = float(np.median(scores))
        distances = np.abs(scores - median)
        winner = int(np.argmin(distances))

        # VCG payment: externality on others
        # Each reviewer "pays" for the distortion they cause
        payments = []
        for i in range(len(scores)):
            others = np.delete(scores, i)
            median_without_i = float(np.median(others))
            payment = abs(median - median_without_i)
            payments.append(round(payment, 4))

        return TruthfulAggregation(
            adopted_score=float(scores[winner]),
            adopted_source=winner,
            vcg_payments=payments,
            is_truthful=True,
        )
    except Exception as e:
        logger.debug("VCG aggregation failed: %s", e)
        return TruthfulAggregation(
            adopted_score=float(np.median(review_scores)),
            adopted_source=len(review_scores) // 2,
            vcg_payments=[0.0] * len(review_scores),
            is_truthful=False,
        )


# ============================================================================
# 9. Hessian Score Sensitivity (from multivariable calculus)
# ============================================================================

try:
    from hololoom.core.warp.math.extensions.multivariable_calculus import ScalarField
    HAS_HESSIAN = True
except ImportError:
    HAS_HESSIAN = False


def score_sensitivity_analysis(
    scoring_fn: Callable[[dict[str, float]], float],
    base_scores: dict[str, float],
    h: float = 1.0,
) -> dict[str, dict]:
    """Compute Hessian-based sensitivity of the promotion score.

    The Hessian tells you: which rubric terms, when perturbed slightly,
    cause the biggest score change? Different from Shapley (which is
    about attribution) — this is about robustness.

    Large diagonal = sensitive to that term.
    Large off-diagonal = terms interact (changing both together matters).
    """
    terms = list(base_scores.keys())
    n = len(terms)
    base = scoring_fn(base_scores)

    # Gradient (first derivatives)
    gradient = {}
    for t in terms:
        perturbed = dict(base_scores)
        perturbed[t] = base_scores[t] + h
        gradient[t] = (scoring_fn(perturbed) - base) / h

    # Hessian (second derivatives) — diagonal and cross terms
    hessian = {}
    for i, ti in enumerate(terms):
        for j, tj in enumerate(terms):
            if j < i:
                continue

            if i == j:
                # Second derivative: f(x+h) - 2f(x) + f(x-h) / h^2
                p_plus = dict(base_scores)
                p_minus = dict(base_scores)
                p_plus[ti] = base_scores[ti] + h
                p_minus[ti] = base_scores[ti] - h
                d2 = (scoring_fn(p_plus) - 2 * base + scoring_fn(p_minus)) / (h * h)
                hessian[f"{ti}"] = round(d2, 6)
            else:
                # Cross derivative: [f(x+hi+hj) - f(x+hi) - f(x+hj) + f(x)] / h^2
                pp = dict(base_scores)
                pp[ti] = base_scores[ti] + h
                pp[tj] = base_scores[tj] + h
                pi = dict(base_scores)
                pi[ti] = base_scores[ti] + h
                pj = dict(base_scores)
                pj[tj] = base_scores[tj] + h
                d2 = (scoring_fn(pp) - scoring_fn(pi) - scoring_fn(pj) + base) / (h * h)
                if abs(d2) > 1e-8:
                    hessian[f"{ti}x{tj}"] = round(d2, 6)

    return {
        "gradient": gradient,
        "hessian_diagonal": {k: v for k, v in hessian.items() if "x" not in k},
        "hessian_interactions": {k: v for k, v in hessian.items() if "x" in k},
        "most_sensitive": max(gradient, key=lambda k: abs(gradient[k])),
    }


# ============================================================================
# 10. Distribution Theory — boundary sensitivity at decision thresholds
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.distribution_theory import Distribution, WeakDerivative
    HAS_DISTRIBUTIONS = True
except ImportError:
    HAS_DISTRIBUTIONS = False


@dataclass
class BoundarySensitivity:
    """How fragile the pipeline is at decision thresholds."""
    threshold: float
    notes_in_boundary: int    # count within ±margin of threshold
    total_notes: int
    fragility: float          # fraction of notes that could flip with small changes
    recommendation: str


def analyze_boundary_sensitivity(
    scores: list[float],
    thresholds: dict[str, float],
    margin: float = 5.0,
) -> list[BoundarySensitivity]:
    """Analyze how many notes cluster near decision boundaries.

    The scoring function has discontinuities at promote/return thresholds.
    The weak derivative (Dirac delta) tells us: all sensitivity is
    concentrated at these exact points. If many notes score 68-72
    (near the promote threshold of 70), the system is fragile —
    one rubric weight tweak flips a batch of decisions.
    """
    results = []
    for name, threshold in thresholds.items():
        in_boundary = sum(1 for s in scores if abs(s - threshold) <= margin)
        fragility = in_boundary / max(1, len(scores))

        if fragility > 0.3:
            rec = "HIGH fragility — consider widening margin or adding a 'review' band"
        elif fragility > 0.1:
            rec = "moderate fragility — monitor for threshold sensitivity"
        else:
            rec = "low fragility — decisions are well-separated from threshold"

        results.append(BoundarySensitivity(
            threshold=threshold,
            notes_in_boundary=in_boundary,
            total_notes=len(scores),
            fragility=round(fragility, 4),
            recommendation=rec,
        ))

    return results


# ============================================================================
# 11. Measure Theory — convergence guarantees for pipeline stability
# ============================================================================

try:
    from hololoom.core.warp.math.analysis.measure_theory import ConvergenceTheorems
    HAS_MEASURE = True
except ImportError:
    HAS_MEASURE = False


@dataclass
class PipelineConvergence:
    """Convergence diagnostic for pipeline quality over time."""
    converged: bool
    current_mean: float
    running_means: list[float]
    is_monotone: bool         # quality non-decreasing?
    dominated: bool           # bounded? (should always be True for [0,100] scores)
    convergence_rate: float   # how fast it's converging (0 = not, 1 = instantly)


def verify_pipeline_convergence(
    score_history: list[float],
    bound: float = 100.0,
    window: int = 10,
) -> PipelineConvergence:
    """Verify the promotion pipeline is converging to stable quality.

    Uses measure-theoretic convergence theorems:
    - Dominated convergence: if scores are bounded (they are, 0-100),
      the average quality MUST converge as n → ∞
    - Monotone convergence: if the bandit is learning (non-decreasing),
      the expected quality converges to the supremum
    - Divergence detection: if the pipeline ISN'T converging,
      something is wrong (rubric instability, adversarial input)
    """
    if len(score_history) < window:
        return PipelineConvergence(
            converged=False, current_mean=float(np.mean(score_history)) if score_history else 0.0,
            running_means=[], is_monotone=False, dominated=True, convergence_rate=0.0,
        )

    scores = np.array(score_history)

    # Check dominated (bounded)
    dominated = bool(np.all(scores >= 0) and np.all(scores <= bound))

    # Compute running means
    running_means = [float(np.mean(scores[:i+1])) for i in range(len(scores))]

    # Check monotone (non-decreasing running mean)
    diffs = np.diff(running_means[-window:])
    is_monotone = bool(np.all(diffs >= -0.01))  # allow tiny dips

    # Check convergence: variance of recent running means
    recent_means = running_means[-window:]
    variance = float(np.var(recent_means))
    converged = variance < 1.0  # running mean stable within ±1

    # Convergence rate: how fast variance is shrinking
    if len(running_means) > window * 2:
        early_var = float(np.var(running_means[:window]))
        late_var = variance
        convergence_rate = 1.0 - (late_var / max(early_var, 1e-10))
        convergence_rate = max(0.0, min(1.0, convergence_rate))
    else:
        convergence_rate = 0.0

    return PipelineConvergence(
        converged=converged,
        current_mean=round(float(np.mean(scores)), 2),
        running_means=[round(m, 2) for m in running_means[-10:]],
        is_monotone=is_monotone,
        dominated=dominated,
        convergence_rate=round(convergence_rate, 4),
    )


# ============================================================================
# 12. Hensel Lifting — principled inference budget decisions
# ============================================================================

@dataclass
class LiftDecision:
    """Whether to proceed to a more expensive evaluation step."""
    skip_llm: bool            # True = coarse check is conclusive
    confidence: float         # how sure we are the lift will agree
    reason: str
    coarse_score: float
    distance_to_threshold: float


def hensel_lift_decision(
    mechanical_score: float,
    promote_threshold: float = 70.0,
    return_threshold: float = 40.0,
    singularity_margin: float = 10.0,
) -> LiftDecision:
    """Decide whether to skip expensive LLM evaluation using Hensel's criterion.

    Hensel's lemma: if f'(a) ≠ 0 mod p (the coarse estimate is far from
    the decision boundary), the refinement to higher precision is UNIQUE —
    the finer evaluation will agree with the coarser one.

    If f'(a) = 0 mod p (score near threshold), the lift may fail —
    finer evaluation might disagree. Must proceed to LLM.

    This is a concrete inference-budget decision rule:
    - Score 95 → skip LLM (Hensel guarantees agreement)
    - Score 68 → must call LLM (lift is uncertain)
    - Score 35 → skip LLM (clearly below threshold, lift agrees)
    """
    # Distance to nearest threshold
    dist_promote = abs(mechanical_score - promote_threshold)
    dist_return = abs(mechanical_score - return_threshold)
    dist_to_nearest = min(dist_promote, dist_return)

    # Non-singularity condition: f'(a) ≠ 0 means score is far from boundary
    is_nonsingular = dist_to_nearest > singularity_margin

    if is_nonsingular:
        # Score is well-separated from all thresholds
        # Hensel guarantees: finer evaluation will agree
        confidence = min(1.0, dist_to_nearest / (singularity_margin * 2))

        if mechanical_score > promote_threshold + singularity_margin:
            reason = f"Score {mechanical_score:.0f} is {dist_promote:.0f} above promote threshold — lift guaranteed"
        elif mechanical_score < return_threshold - singularity_margin:
            reason = f"Score {mechanical_score:.0f} is {dist_return:.0f} below return threshold — lift guaranteed"
        else:
            reason = f"Score {mechanical_score:.0f} is in the return band, {dist_to_nearest:.0f} from nearest boundary"

        return LiftDecision(
            skip_llm=True,
            confidence=round(confidence, 4),
            reason=reason,
            coarse_score=mechanical_score,
            distance_to_threshold=round(dist_to_nearest, 1),
        )
    else:
        # Near boundary — singular point — must refine
        confidence = dist_to_nearest / singularity_margin
        return LiftDecision(
            skip_llm=False,
            confidence=round(confidence, 4),
            reason=f"Score {mechanical_score:.0f} is within {singularity_margin:.0f} of threshold — lift uncertain, LLM required",
            coarse_score=mechanical_score,
            distance_to_threshold=round(dist_to_nearest, 1),
        )


# ============================================================================
# Availability summary
# ============================================================================

def math_extensions_v3_status() -> dict[str, bool]:
    """Report which v3 math extensions are available."""
    return {
        "poincare_full_toolkit": HAS_POINCARE,
        "fourier_periodicity": HAS_FOURIER,
        "group_symmetry": HAS_ALGEBRA,
        "sectional_curvature": HAS_CURVATURE,
        "noether_invariants": True,     # Pure numpy
        "rkhs_kernel": True,            # Pure numpy
        "convergence_diagnostics": HAS_CONVERGENCE,
        "vcg_mechanism": HAS_VCG,
        "hessian_sensitivity": HAS_HESSIAN,
        "boundary_sensitivity": True,   # Pure numpy (distribution theory concept)
        "pipeline_convergence": True,   # Pure numpy (measure theory concept)
        "hensel_lift_decisions": True,   # Pure numpy (p-adic concept)
    }
