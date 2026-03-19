"""
Renyi Entropy — Generalized Information Measures
=================================================

A family of entropy measures parameterized by order alpha, unifying
Shannon, Hartley, collision, and min-entropy as special cases.

Core Concepts:
- Renyi Entropy: H_alpha = (1/(1-alpha)) * log(sum(p_i^alpha))
- Tsallis Entropy: Non-extensive generalization S_q = (1/(q-1))(1 - sum(p_i^q))
- Alpha-Divergence: Generalized divergence family connecting KL, Hellinger, chi^2
- Diversity Profile: Continuous mapping alpha -> H_alpha revealing distribution structure

Mathematical Foundation:
Special cases of Renyi entropy:
    alpha=0:   Hartley entropy = log(|support|)
    alpha->1:  Shannon entropy = -sum(p_i log(p_i))  (by L'Hopital)
    alpha=2:   Collision entropy = -log(sum(p_i^2))
    alpha->inf: Min-entropy = -log(max(p_i))

Effective number: exp(H_alpha) = number of "equally likely" outcomes
    giving the same entropy value

Applications to Warp Space:
- Diversity measurement for Thompson Sampling arm populations
- Uncertainty quantification across embedding dimensions
- Information-theoretic convergence criteria for weaving cycles
- Robustness analysis via min-entropy (worst-case uncertainty)

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    from scipy.special import rel_entr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# HELPERS
# ============================================================================

def _validate_distribution(p: np.ndarray, name: str = "p") -> np.ndarray:
    """Validate and normalize a probability distribution."""
    p = np.asarray(p, dtype=np.float64).ravel()
    if np.any(p < 0):
        raise ValueError(f"Distribution {name} has negative values")
    total = p.sum()
    if total < 1e-15:
        raise ValueError(f"Distribution {name} sums to zero")
    if abs(total - 1.0) > 1e-6:
        logger.debug(f"Distribution {name} sums to {total}, renormalizing")
        p = p / total
    return p


def _safe_log(x: np.ndarray, base: float = np.e) -> np.ndarray:
    """Logarithm with protection against log(0)."""
    return np.where(x > 0, np.log(x) / np.log(base), 0.0)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RenyiResult:
    """Result of a Renyi entropy computation.

    Attributes:
        entropy: The computed Renyi entropy value.
        alpha: The order parameter used.
        effective_n: Effective number of categories exp(H_alpha).
    """
    entropy: float
    alpha: float
    effective_n: float

    @property
    def normalized(self) -> float:
        """Entropy normalized to [0, 1] by dividing by log of support size.

        Only meaningful if effective_n reflects the support size.
        Returns the entropy divided by log(effective_n) at alpha=0 (Hartley).
        """
        if self.effective_n <= 1:
            return 0.0
        # This gives a rough normalization; for exact normalization
        # you'd need the support size separately
        max_entropy = np.log(self.effective_n)
        if max_entropy < 1e-15:
            return 0.0
        return min(1.0, self.entropy / max_entropy)


# ============================================================================
# RENYI ENTROPY
# ============================================================================

class RenyiEntropy:
    """
    Renyi entropy and related information measures.

    The Renyi entropy of order alpha for a discrete distribution p is:
        H_alpha(p) = (1 / (1 - alpha)) * log(sum_i p_i^alpha)

    It forms a non-increasing family in alpha, with special cases:
        alpha=0:    Hartley entropy (log of support size)
        alpha->1:   Shannon entropy (continuous limit)
        alpha=2:    Collision entropy (-log of collision probability)
        alpha->inf: Min-entropy (-log of max probability)
    """

    @staticmethod
    def compute(
        p: np.ndarray,
        alpha: float,
        base: float = np.e,
    ) -> float:
        """
        Compute Renyi entropy of order alpha.

        H_alpha(p) = (1 / (1 - alpha)) * log_base(sum(p_i^alpha))

        Special cases are handled analytically to avoid numerical issues.

        Args:
            p: Probability distribution (non-negative, sums to 1).
            alpha: Order parameter (>= 0). Non-negative real number.
            base: Logarithm base (default: natural log).

        Returns:
            Renyi entropy value.

        Raises:
            ValueError: If alpha < 0 or distribution is invalid.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

        p = _validate_distribution(p)

        # Special case: alpha = 0 (Hartley entropy)
        if alpha == 0:
            support_size = np.sum(p > 0)
            return float(np.log(support_size) / np.log(base)) if support_size > 0 else 0.0

        # Special case: alpha -> 1 (Shannon entropy, via L'Hopital's rule)
        if abs(alpha - 1.0) < 1e-12:
            return float(-np.sum(p * _safe_log(p, base)))

        # Special case: alpha -> infinity (min-entropy)
        if alpha > 1e6:
            return RenyiEntropy.min_entropy(p, base)

        # General case
        power_sum = np.sum(p ** alpha)
        if power_sum <= 0:
            return 0.0

        return float((1.0 / (1.0 - alpha)) * np.log(power_sum) / np.log(base))

    @staticmethod
    def spectrum(
        p: np.ndarray,
        alphas: np.ndarray | None = None,
        base: float = np.e,
    ) -> list[RenyiResult]:
        """
        Compute Renyi entropy at multiple alpha values.

        The spectrum reveals the structure of the distribution:
        - Flat distributions have constant spectrum
        - Peaked distributions show steep decline with alpha
        - The gap between alpha=1 and alpha=2 indicates tail weight

        Args:
            p: Probability distribution.
            alphas: Array of alpha values to evaluate. If None, uses
                   a default grid [0, 0.5, 1, 1.5, 2, 3, 5, 10, inf].
            base: Logarithm base.

        Returns:
            List of RenyiResult for each alpha value.
        """
        p = _validate_distribution(p)

        if alphas is None:
            alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf])
        else:
            alphas = np.asarray(alphas, dtype=np.float64)

        results = []
        for a in alphas:
            if np.isinf(a):
                h = RenyiEntropy.min_entropy(p, base)
                eff_n = RenyiEntropy.effective_number(p, np.inf)
            else:
                h = RenyiEntropy.compute(p, a, base)
                eff_n = RenyiEntropy.effective_number(p, a)

            results.append(RenyiResult(entropy=h, alpha=float(a), effective_n=eff_n))

        return results

    @staticmethod
    def min_entropy(p: np.ndarray, base: float = np.e) -> float:
        """
        Min-entropy: Renyi entropy as alpha -> infinity.

        H_inf(p) = -log(max(p_i))

        The most conservative entropy measure. Gives the worst-case
        uncertainty — the probability of guessing the most likely outcome.

        Args:
            p: Probability distribution.
            base: Logarithm base.

        Returns:
            Min-entropy value.
        """
        p = _validate_distribution(p)
        p_max = float(np.max(p))
        if p_max <= 0:
            return 0.0
        return float(-np.log(p_max) / np.log(base))

    @staticmethod
    def collision_entropy(p: np.ndarray, base: float = np.e) -> float:
        """
        Collision entropy: Renyi entropy at alpha = 2.

        H_2(p) = -log(sum(p_i^2))

        Related to the probability that two independent draws from p
        yield the same outcome. Equivalent to -log(Simpson's index).

        Args:
            p: Probability distribution.
            base: Logarithm base.

        Returns:
            Collision entropy value.
        """
        p = _validate_distribution(p)
        collision_prob = float(np.sum(p ** 2))
        if collision_prob <= 0:
            return 0.0
        return float(-np.log(collision_prob) / np.log(base))

    @staticmethod
    def effective_number(p: np.ndarray, alpha: float) -> float:
        """
        Effective number of categories: exp(H_alpha).

        Represents the number of equally-likely outcomes that would
        produce the same entropy value. Also known as the "true diversity"
        of order alpha in ecology (Hill numbers).

        Args:
            p: Probability distribution.
            alpha: Order parameter.

        Returns:
            Effective number of categories.
        """
        p = _validate_distribution(p)

        if np.isinf(alpha):
            # exp(-log(max(p))) = 1/max(p)
            p_max = np.max(p)
            return 1.0 / p_max if p_max > 0 else 0.0

        h = RenyiEntropy.compute(p, alpha, base=np.e)
        return float(np.exp(h))

    @staticmethod
    def diversity_profile(p: np.ndarray, base: float = np.e) -> Callable[[float], float]:
        """
        Return a continuous function mapping alpha -> H_alpha(p).

        The diversity profile reveals the full information content of
        a distribution at all scales. Steep profiles indicate uneven
        distributions; flat profiles indicate near-uniform distributions.

        Args:
            p: Probability distribution.
            base: Logarithm base.

        Returns:
            Callable that takes alpha and returns H_alpha(p).
        """
        p = _validate_distribution(p)
        # Capture p in closure
        p_frozen = p.copy()

        def profile(alpha: float) -> float:
            if np.isinf(alpha):
                return RenyiEntropy.min_entropy(p_frozen, base)
            return RenyiEntropy.compute(p_frozen, alpha, base)

        return profile

    @staticmethod
    def renyi_divergence(
        p: np.ndarray,
        q: np.ndarray,
        alpha: float,
        base: float = np.e,
    ) -> float:
        """
        Renyi divergence of order alpha between distributions p and q.

        D_alpha(p || q) = (1/(alpha-1)) * log(sum(p_i^alpha * q_i^(1-alpha)))

        Generalizes KL divergence (alpha -> 1).

        Args:
            p: First distribution.
            q: Second distribution.
            alpha: Order parameter (> 0, != 1).
            base: Logarithm base.

        Returns:
            Renyi divergence value.
        """
        p = _validate_distribution(p, "p")
        q = _validate_distribution(q, "q")

        if len(p) != len(q):
            raise ValueError(f"Distributions must have same length: {len(p)} vs {len(q)}")

        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

        # Special case: alpha -> 1 (KL divergence)
        if abs(alpha - 1.0) < 1e-12:
            # KL(p || q) = sum(p * log(p/q))
            mask = p > 0
            if np.any(q[mask] <= 0):
                return float('inf')
            return float(np.sum(p[mask] * np.log(p[mask] / q[mask]) / np.log(base)))

        # General case
        mask = (p > 0) | (q > 0)
        # Handle cases where q=0 but p>0
        if np.any((p > 0) & (q <= 0)):
            if alpha > 1:
                return float('inf')
            # For alpha < 1, the term p^alpha * q^(1-alpha) with q=0
            # goes to 0 if alpha < 1 and p > 0
            # Actually: 0^(1-alpha) = 0 for alpha < 1, so term = 0

        # Compute sum(p^alpha * q^(1-alpha))
        terms = np.zeros_like(p)
        valid = (p > 0) & (q > 0)
        terms[valid] = (p[valid] ** alpha) * (q[valid] ** (1.0 - alpha))

        total = np.sum(terms)
        if total <= 0:
            return float('inf')

        return float((1.0 / (alpha - 1.0)) * np.log(total) / np.log(base))


# ============================================================================
# TSALLIS ENTROPY
# ============================================================================

class TsallisEntropy:
    """
    Tsallis entropy — a non-extensive generalization of Shannon entropy.

    S_q(p) = (1/(q-1)) * (1 - sum(p_i^q))

    Unlike Renyi entropy, Tsallis entropy is non-additive for independent
    systems: S_q(A+B) = S_q(A) + S_q(B) + (1-q)*S_q(A)*S_q(B).

    This non-extensivity makes it suitable for systems with long-range
    correlations or fractal phase spaces.
    """

    @staticmethod
    def compute(
        p: np.ndarray,
        q: float,
    ) -> float:
        """
        Compute Tsallis entropy of order q.

        S_q(p) = (1/(q-1)) * (1 - sum(p_i^q))

        Special case q -> 1 recovers Shannon entropy.

        Args:
            p: Probability distribution.
            q: Entropic index (q > 0).

        Returns:
            Tsallis entropy value.
        """
        if q <= 0:
            raise ValueError(f"q must be > 0, got {q}")

        p = _validate_distribution(p)

        # Special case: q -> 1 (Shannon entropy)
        if abs(q - 1.0) < 1e-12:
            return float(-np.sum(p * _safe_log(p)))

        power_sum = np.sum(p ** q)
        return float((1.0 / (q - 1.0)) * (1.0 - power_sum))

    @staticmethod
    def tsallis_divergence(
        p: np.ndarray,
        q_dist: np.ndarray,
        q_param: float,
    ) -> float:
        """
        Tsallis divergence (generalized KL divergence).

        D_q(p || r) = -(1/(1-q)) * (1 - sum(p_i^q * r_i^(1-q)))

        where r is the reference distribution (q_dist argument).

        Args:
            p: First distribution.
            q_dist: Reference distribution.
            q_param: Entropic index.

        Returns:
            Tsallis divergence value.
        """
        if q_param <= 0:
            raise ValueError(f"q_param must be > 0, got {q_param}")

        p = _validate_distribution(p, "p")
        q_dist = _validate_distribution(q_dist, "q")

        if len(p) != len(q_dist):
            raise ValueError(f"Distributions must have same length: {len(p)} vs {len(q_dist)}")

        # Special case: q -> 1 (KL divergence)
        if abs(q_param - 1.0) < 1e-12:
            mask = p > 0
            if np.any(q_dist[mask] <= 0):
                return float('inf')
            return float(np.sum(p[mask] * np.log(p[mask] / q_dist[mask])))

        # General case
        valid = (p > 0) & (q_dist > 0)
        terms = np.zeros_like(p)
        terms[valid] = (p[valid] ** q_param) * (q_dist[valid] ** (1.0 - q_param))

        total = np.sum(terms)
        return float(-(1.0 / (1.0 - q_param)) * (1.0 - total))

    @staticmethod
    def q_logarithm(x: float, q: float) -> float:
        """
        q-logarithm: ln_q(x) = (x^(1-q) - 1) / (1 - q).

        Generalization of natural log. ln_1(x) = ln(x).
        Used in Tsallis statistical mechanics.

        Args:
            x: Positive real number.
            q: Entropic index.

        Returns:
            q-logarithm value.
        """
        if x <= 0:
            raise ValueError(f"x must be > 0, got {x}")
        if abs(q - 1.0) < 1e-12:
            return float(np.log(x))
        return float((x ** (1.0 - q) - 1.0) / (1.0 - q))

    @staticmethod
    def q_exponential(x: float, q: float) -> float:
        """
        q-exponential: exp_q(x) = [1 + (1-q)x]^(1/(1-q)) if 1+(1-q)x > 0.

        Inverse of q-logarithm. exp_1(x) = exp(x).

        Args:
            x: Real number.
            q: Entropic index.

        Returns:
            q-exponential value.
        """
        if abs(q - 1.0) < 1e-12:
            return float(np.exp(x))
        base = 1.0 + (1.0 - q) * x
        if base <= 0:
            return 0.0
        return float(base ** (1.0 / (1.0 - q)))


# ============================================================================
# ALPHA-DIVERGENCE & f-DIVERGENCE
# ============================================================================

class AlphaDivergence:
    """
    Alpha-divergence and f-divergence families.

    The alpha-divergence unifies many standard divergence measures:
        alpha = 1:   KL(p || q)
        alpha = -1:  KL(q || p)
        alpha = 0:   2 * (1 - Bhattacharyya coefficient)
        alpha = 0.5: 4 * Hellinger^2

    The f-divergence is the most general divergence of the form:
        D_f(p || q) = sum_i q_i * f(p_i / q_i)
    """

    @staticmethod
    def compute(
        p: np.ndarray,
        q: np.ndarray,
        alpha: float,
    ) -> float:
        """
        Compute alpha-divergence between p and q.

        D_alpha(p || q) = (4 / (1 - alpha^2)) * (1 - sum(p_i^((1+alpha)/2) * q_i^((1-alpha)/2)))

        This is the Amari alpha-divergence, symmetrized form.

        Args:
            p: First distribution.
            q: Second distribution.
            alpha: Divergence parameter (real number, != +/-1).

        Returns:
            Alpha-divergence value (non-negative).
        """
        p = _validate_distribution(p, "p")
        q = _validate_distribution(q, "q")

        if len(p) != len(q):
            raise ValueError(f"Distributions must have same length: {len(p)} vs {len(q)}")

        # Special case: alpha = 1 (KL divergence p||q)
        if abs(alpha - 1.0) < 1e-12:
            mask = p > 0
            if np.any(q[mask] <= 0):
                return float('inf')
            return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

        # Special case: alpha = -1 (KL divergence q||p)
        if abs(alpha + 1.0) < 1e-12:
            mask = q > 0
            if np.any(p[mask] <= 0):
                return float('inf')
            return float(np.sum(q[mask] * np.log(q[mask] / p[mask])))

        # General case
        exp_p = (1.0 + alpha) / 2.0
        exp_q = (1.0 - alpha) / 2.0

        valid = (p > 0) & (q > 0)
        terms = np.zeros_like(p)
        terms[valid] = (p[valid] ** exp_p) * (q[valid] ** exp_q)

        integral = np.sum(terms)
        coeff = 4.0 / (1.0 - alpha ** 2)

        return float(coeff * (1.0 - integral))

    @staticmethod
    def f_divergence(
        p: np.ndarray,
        q: np.ndarray,
        f_fn: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """
        Compute generic f-divergence.

        D_f(p || q) = sum_i q_i * f(p_i / q_i)

        where f is a convex function with f(1) = 0.

        Common choices of f:
            f(t) = t*log(t)           -> KL divergence
            f(t) = -log(t)            -> Reverse KL
            f(t) = (sqrt(t) - 1)^2    -> Squared Hellinger
            f(t) = |t - 1|            -> Total variation
            f(t) = (t - 1)^2          -> Chi-squared

        Args:
            p: First distribution.
            q: Second distribution.
            f_fn: Convex function f with f(1) = 0. Applied element-wise
                  to numpy arrays.

        Returns:
            f-divergence value.
        """
        p = _validate_distribution(p, "p")
        q = _validate_distribution(q, "q")

        if len(p) != len(q):
            raise ValueError(f"Distributions must have same length: {len(p)} vs {len(q)}")

        # Compute ratios where q > 0
        mask = q > 0
        result = 0.0

        if np.any(mask):
            ratios = np.zeros_like(p)
            ratios[mask] = p[mask] / q[mask]
            f_values = f_fn(ratios)
            result = float(np.sum(q[mask] * f_values[mask]))

        # Handle q_i = 0, p_i > 0 (contributes f(inf) * 0 or infinity)
        # Convention: 0 * f(p/0) = p * lim_{t->inf} f(t)/t (perspective function)
        # For most practical f, this is either 0 or infinity
        # We skip this and log a warning if it occurs
        if np.any((q <= 0) & (p > 0)):
            logger.warning("f-divergence: q=0 where p>0, result may be inaccurate")

        return result

    @staticmethod
    def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        Hellinger distance between two distributions.

        H(p, q) = (1/sqrt(2)) * sqrt(sum((sqrt(p_i) - sqrt(q_i))^2))

        Always in [0, 1]. Related to Bhattacharyya coefficient:
        H^2 = 1 - BC where BC = sum(sqrt(p_i * q_i)).

        Args:
            p: First distribution.
            q: Second distribution.

        Returns:
            Hellinger distance in [0, 1].
        """
        p = _validate_distribution(p, "p")
        q = _validate_distribution(q, "q")

        if len(p) != len(q):
            raise ValueError("Distributions must have same length")

        diff = np.sqrt(p) - np.sqrt(q)
        return float(np.sqrt(np.sum(diff ** 2)) / np.sqrt(2.0))

    @staticmethod
    def bhattacharyya_coefficient(p: np.ndarray, q: np.ndarray) -> float:
        """
        Bhattacharyya coefficient: BC(p, q) = sum(sqrt(p_i * q_i)).

        Measures overlap between two distributions. BC = 1 for identical
        distributions, BC = 0 for non-overlapping distributions.

        Args:
            p: First distribution.
            q: Second distribution.

        Returns:
            Bhattacharyya coefficient in [0, 1].
        """
        p = _validate_distribution(p, "p")
        q = _validate_distribution(q, "q")

        if len(p) != len(q):
            raise ValueError("Distributions must have same length")

        return float(np.sum(np.sqrt(p * q)))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def renyi_entropy(p: np.ndarray, alpha: float, base: float = np.e) -> float:
    """Quick interface for Renyi entropy computation."""
    return RenyiEntropy.compute(p, alpha, base)


def tsallis_entropy(p: np.ndarray, q: float) -> float:
    """Quick interface for Tsallis entropy computation."""
    return TsallisEntropy.compute(p, q)


def min_entropy(p: np.ndarray, base: float = np.e) -> float:
    """Quick interface for min-entropy."""
    return RenyiEntropy.min_entropy(p, base)


def collision_entropy(p: np.ndarray, base: float = np.e) -> float:
    """Quick interface for collision entropy (alpha=2)."""
    return RenyiEntropy.collision_entropy(p, base)
