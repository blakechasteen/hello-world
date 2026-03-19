"""
Information Theory - Entropy, Mutual Information, Divergences
=============================================================

Mathematical foundations of information and uncertainty.

Classes:
    InformationMetrics: Unified interface for information measures
    DistributionPair: Joint/marginal distribution container

Functions:
    entropy: Shannon entropy H(X)
    cross_entropy: H(P, Q)
    kl_divergence: Kullback-Leibler D_KL(P || Q)
    mutual_information: I(X; Y)
    jensen_shannon: Symmetric divergence JS(P || Q)
    conditional_entropy: H(X | Y)
    information_gain: I(X; Y) = H(X) - H(X|Y)

Applications:
    - Memory deduplication (similar memories have low MI)
    - Embedding compression (maximize MI, minimize entropy)
    - Feature selection (maximize information gain)
    - Distribution matching (minimize KL/JS divergence)

Author: Claude Code
Date: December 2025 (Math Module Expansion)
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class DivergenceType(Enum):
    """Types of statistical divergences."""
    KL = "kullback_leibler"
    JS = "jensen_shannon"
    HELLINGER = "hellinger"
    TOTAL_VARIATION = "total_variation"


@dataclass
class DistributionPair:
    """
    Container for joint and marginal distributions.

    Attributes:
        joint: P(X, Y) joint distribution matrix
        marginal_x: P(X) marginal distribution
        marginal_y: P(Y) marginal distribution
    """
    joint: np.ndarray
    marginal_x: np.ndarray | None = None
    marginal_y: np.ndarray | None = None

    def __post_init__(self):
        """Compute marginals if not provided."""
        if self.marginal_x is None:
            self.marginal_x = self.joint.sum(axis=1)
        if self.marginal_y is None:
            self.marginal_y = self.joint.sum(axis=0)

    def validate(self) -> bool:
        """Check that distributions are valid (sum to 1, non-negative)."""
        eps = 1e-9
        joint_ok = abs(self.joint.sum() - 1.0) < eps and np.all(self.joint >= 0)
        mx_ok = abs(self.marginal_x.sum() - 1.0) < eps
        my_ok = abs(self.marginal_y.sum() - 1.0) < eps
        return joint_ok and mx_ok and my_ok


def _safe_log(x: np.ndarray, base: float = np.e) -> np.ndarray:
    """Logarithm with protection against log(0)."""
    return np.where(x > 0, np.log(x) / np.log(base), 0.0)


def entropy(p: np.ndarray, base: float = 2.0) -> float:
    """
    Shannon entropy: H(X) = -∑ p(x) log p(x).

    Measures uncertainty/information content of distribution.

    Args:
        p: Probability distribution (must sum to 1)
        base: Logarithm base (2 for bits, e for nats)

    Returns:
        Entropy in specified units (bits if base=2)

    Example:
        >>> entropy(np.array([0.5, 0.5]))  # Fair coin
        1.0
        >>> entropy(np.array([1.0, 0.0]))  # Certain outcome
        0.0
    """
    p = np.asarray(p).flatten()
    p = p[p > 0]  # Remove zeros to avoid log(0)
    return -np.sum(p * _safe_log(p, base))


def cross_entropy(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """
    Cross-entropy: H(P, Q) = -∑ p(x) log q(x).

    Measures average bits needed to encode P using code optimized for Q.

    Args:
        p: True distribution
        q: Model distribution
        base: Logarithm base

    Returns:
        Cross-entropy (always >= entropy(p))
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()

    # Avoid log(0) by adding small epsilon where q=0 but p>0
    q_safe = np.where((p > 0) & (q == 0), 1e-10, q)

    return -np.sum(p * _safe_log(q_safe, base))


def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """
    Kullback-Leibler divergence: D_KL(P || Q) = ∑ p(x) log(p(x)/q(x)).

    Measures how P differs from reference Q. NOT symmetric.

    Args:
        p: True distribution
        q: Reference/model distribution
        base: Logarithm base

    Returns:
        KL divergence (always >= 0, = 0 iff P = Q)

    Properties:
        - D_KL(P || Q) >= 0 (Gibbs' inequality)
        - D_KL(P || Q) = 0 iff P = Q
        - D_KL(P || Q) != D_KL(Q || P) in general
    """
    return cross_entropy(p, q, base) - entropy(p, base)


def jensen_shannon(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """
    Jensen-Shannon divergence: JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M).

    Symmetric version of KL divergence where M = 0.5 * (P + Q).

    Args:
        p: First distribution
        q: Second distribution
        base: Logarithm base

    Returns:
        JS divergence in [0, 1] for base=2

    Properties:
        - 0 <= JS(P || Q) <= 1 (for base=2)
        - JS(P || Q) = JS(Q || P) (symmetric)
        - sqrt(JS) is a metric
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, base) + 0.5 * kl_divergence(q, m, base)


def mutual_information(pair: DistributionPair, base: float = 2.0) -> float:
    """
    Mutual information: I(X; Y) = H(X) + H(Y) - H(X, Y).

    Measures information shared between X and Y.

    Args:
        pair: DistributionPair with joint and marginal distributions
        base: Logarithm base

    Returns:
        Mutual information (always >= 0)

    Properties:
        - I(X; Y) >= 0
        - I(X; Y) = 0 iff X and Y are independent
        - I(X; Y) = I(Y; X) (symmetric)
        - I(X; Y) <= min(H(X), H(Y))

    Example:
        >>> # Independent variables
        >>> joint = np.array([[0.25, 0.25], [0.25, 0.25]])
        >>> pair = DistributionPair(joint)
        >>> mutual_information(pair)
        0.0
    """
    h_x = entropy(pair.marginal_x, base)
    h_y = entropy(pair.marginal_y, base)
    h_xy = entropy(pair.joint.flatten(), base)
    return h_x + h_y - h_xy


def conditional_entropy(pair: DistributionPair, base: float = 2.0) -> float:
    """
    Conditional entropy: H(X | Y) = H(X, Y) - H(Y).

    Remaining uncertainty in X after knowing Y.

    Args:
        pair: DistributionPair with joint distribution
        base: Logarithm base

    Returns:
        Conditional entropy H(X|Y)

    Properties:
        - 0 <= H(X|Y) <= H(X)
        - H(X|Y) = 0 iff X is deterministic given Y
        - H(X|Y) = H(X) iff X and Y are independent
    """
    h_xy = entropy(pair.joint.flatten(), base)
    h_y = entropy(pair.marginal_y, base)
    return h_xy - h_y


def information_gain(pair: DistributionPair, base: float = 2.0) -> float:
    """
    Information gain: IG(X; Y) = H(X) - H(X | Y) = I(X; Y).

    Reduction in entropy of X due to knowledge of Y.
    Commonly used in decision trees.

    Args:
        pair: DistributionPair
        base: Logarithm base

    Returns:
        Information gain (equals mutual information)
    """
    return mutual_information(pair, base)


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Hellinger distance: H(P, Q) = (1/√2) * ||√P - √Q||_2.

    Bounded metric between distributions.

    Returns:
        Hellinger distance in [0, 1]
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """
    Total variation distance: TV(P, Q) = 0.5 * ||P - Q||_1.

    Maximum difference in probability of any event.

    Returns:
        TV distance in [0, 1]
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return 0.5 * np.sum(np.abs(p - q))


class InformationMetrics:
    """
    Unified interface for information-theoretic computations.

    Provides batch operations and caching for efficiency.

    Example:
        >>> metrics = InformationMetrics(base=2)
        >>> # Memory deduplication: find similar memories
        >>> similarities = metrics.batch_js_divergence(embeddings)
        >>> duplicates = np.where(similarities < threshold)
    """

    def __init__(self, base: float = 2.0, epsilon: float = 1e-10):
        """
        Args:
            base: Logarithm base (2 for bits, e for nats)
            epsilon: Small value for numerical stability
        """
        self.base = base
        self.epsilon = epsilon

    def entropy(self, p: np.ndarray) -> float:
        """Compute entropy."""
        return entropy(p, self.base)

    def batch_entropy(self, distributions: np.ndarray) -> np.ndarray:
        """
        Compute entropy for multiple distributions.

        Args:
            distributions: Shape (n_distributions, n_outcomes)

        Returns:
            Entropies array of shape (n_distributions,)
        """
        # Vectorized entropy computation
        p = np.clip(distributions, self.epsilon, 1.0)
        log_p = np.log(p) / np.log(self.base)
        return -np.sum(distributions * log_p, axis=1)

    def batch_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Compute KL divergence for multiple distribution pairs.

        Args:
            p: Shape (n_pairs, n_outcomes)
            q: Shape (n_pairs, n_outcomes)

        Returns:
            KL divergences of shape (n_pairs,)
        """
        q_safe = np.clip(q, self.epsilon, 1.0)
        p_safe = np.clip(p, self.epsilon, 1.0)
        log_ratio = np.log(p_safe / q_safe) / np.log(self.base)
        return np.sum(p * log_ratio, axis=1)

    def batch_js_divergence(self, distributions: np.ndarray) -> np.ndarray:
        """
        Compute pairwise JS divergence matrix.

        Args:
            distributions: Shape (n, d) where n is number of distributions

        Returns:
            Symmetric distance matrix of shape (n, n)
        """
        n = distributions.shape[0]
        js_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                js = jensen_shannon(distributions[i], distributions[j], self.base)
                js_matrix[i, j] = js
                js_matrix[j, i] = js

        return js_matrix

    def estimate_mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Estimate mutual information from samples via histogram.

        Args:
            x: Samples from X (n_samples,)
            y: Samples from Y (n_samples,)
            n_bins: Number of histogram bins

        Returns:
            Estimated I(X; Y)
        """
        # Create 2D histogram for joint distribution
        joint_hist, _, _ = np.histogram2d(x, y, bins=n_bins)
        joint_prob = joint_hist / joint_hist.sum()

        # Create marginals
        marginal_x = joint_prob.sum(axis=1)
        marginal_y = joint_prob.sum(axis=0)

        pair = DistributionPair(joint_prob, marginal_x, marginal_y)
        return mutual_information(pair, self.base)

    def compression_ratio(self, original: np.ndarray, compressed: np.ndarray) -> float:
        """
        Compute compression ratio using entropy.

        Args:
            original: Original data (as probability distribution)
            compressed: Compressed representation

        Returns:
            Ratio H(compressed) / H(original)
        """
        h_original = self.entropy(original)
        h_compressed = self.entropy(compressed)
        return h_compressed / h_original if h_original > 0 else 1.0

    def redundancy(self, p: np.ndarray) -> float:
        """
        Compute redundancy: 1 - H(X) / H_max.

        Measures how far from maximum entropy (uniform).

        Returns:
            Redundancy in [0, 1]
        """
        n = len(p)
        h_max = np.log(n) / np.log(self.base)  # Entropy of uniform
        h = self.entropy(p)
        return 1.0 - h / h_max if h_max > 0 else 0.0


# Integration with HoloLoom memory system
def memory_similarity_score(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute information-theoretic similarity between memory embeddings.

    Uses Jensen-Shannon divergence on softmax-normalized embeddings.

    Args:
        embedding1: First memory embedding
        embedding2: Second memory embedding

    Returns:
        Similarity score in [0, 1] (1 = identical)
    """
    # Convert to probability distributions via softmax
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    p = softmax(embedding1)
    q = softmax(embedding2)

    # JS divergence is bounded [0, 1] for base 2
    js = jensen_shannon(p, q, base=2.0)

    # Convert to similarity (1 - divergence)
    return 1.0 - js


def find_duplicate_memories(
    embeddings: np.ndarray,
    threshold: float = 0.9
) -> list[tuple[int, int, float]]:
    """
    Find duplicate/near-duplicate memories using information theory.

    Args:
        embeddings: Memory embeddings of shape (n_memories, embedding_dim)
        threshold: Similarity threshold for duplicates

    Returns:
        List of (idx1, idx2, similarity) for duplicates
    """
    n = embeddings.shape[0]
    duplicates = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = memory_similarity_score(embeddings[i], embeddings[j])
            if sim >= threshold:
                duplicates.append((i, j, sim))

    return duplicates


# ============================================================================
# INFORMATION BOTTLENECK
# ============================================================================


class InformationBottleneck:
    """Information Bottleneck method — optimal lossy compression.

    Find compressed representation T of X that preserves maximum
    information about Y: max I(T;Y) subject to I(T;X) <= R.

    The IB Lagrangian:  L[p(t|x)] = I(T;X) - beta * I(T;Y)

    At beta=0, T is trivial (one cluster). As beta -> inf, T preserves
    all information about Y. The IB curve traces optimal tradeoffs.

    References:
        Tishby, Pereira & Bialek (1999) "The Information Bottleneck Method"
    """

    @staticmethod
    def compress(
        joint_XY: np.ndarray,
        beta: float,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> tuple[np.ndarray, float, float]:
        """IB compression via iterative Blahut-Arimoto style algorithm.

        Finds p(t|x) that minimizes I(T;X) - beta * I(T;Y).

        Args:
            joint_XY: Joint distribution P(X,Y), shape (|X|, |Y|).
            beta: Tradeoff parameter (high = preserve more info about Y).
            n_clusters: Number of compressed states |T|.
            max_iter: Maximum iterations.
            tol: Convergence tolerance on I(T;Y).

        Returns:
            (assignment_probs, I_TX, I_TY) where assignment_probs is
            p(t|x) of shape (|X|, |T|).
        """
        joint_XY = np.asarray(joint_XY, dtype=np.float64)
        n_x, n_y = joint_XY.shape
        eps = 1e-15

        # Marginals
        p_x = joint_XY.sum(axis=1)  # (|X|,)
        p_y = joint_XY.sum(axis=0)  # (|Y|,)

        # p(y|x) = p(x,y) / p(x)
        p_y_given_x = joint_XY / (p_x[:, None] + eps)  # (|X|, |Y|)

        # Initialize p(t|x) randomly (soft assignment)
        rng = np.random.default_rng(42)
        p_t_given_x = rng.dirichlet(np.ones(n_clusters), size=n_x)  # (|X|, |T|)

        prev_i_ty = -1.0

        for _ in range(max_iter):
            # p(t) = sum_x p(t|x) * p(x)
            p_t = p_t_given_x.T @ p_x  # (|T|,)
            p_t = np.maximum(p_t, eps)

            # p(y|t) = sum_x p(y|x) * p(t|x) * p(x) / p(t)
            # Numerator: (|T|, |Y|) = (|T|, |X|) @ (|X|, |Y|) element-weighted by p(x)
            weighted = p_t_given_x.T * p_x[None, :]  # (|T|, |X|)
            p_y_given_t = (weighted @ p_y_given_x) / (p_t[:, None] + eps)  # (|T|, |Y|)
            p_y_given_t = np.maximum(p_y_given_t, eps)
            p_y_given_t /= p_y_given_t.sum(axis=1, keepdims=True)

            # Update p(t|x) via IB self-consistent equation:
            # p(t|x) ∝ p(t) * exp(-beta * D_KL[p(y|x) || p(y|t)])
            log_ratio = np.zeros((n_x, n_clusters))
            for t in range(n_clusters):
                # D_KL(p(y|x) || p(y|t)) for each x
                ratio = p_y_given_x / (p_y_given_t[t][None, :] + eps)
                kl = np.sum(
                    p_y_given_x * np.log(np.maximum(ratio, eps)), axis=1
                )
                log_ratio[:, t] = np.log(p_t[t] + eps) - beta * kl

            # Normalize (softmax for numerical stability)
            log_ratio -= log_ratio.max(axis=1, keepdims=True)
            p_t_given_x = np.exp(log_ratio)
            p_t_given_x /= p_t_given_x.sum(axis=1, keepdims=True)

            # Compute I(T;Y) for convergence check
            p_t = p_t_given_x.T @ p_x
            p_t = np.maximum(p_t, eps)
            weighted = p_t_given_x.T * p_x[None, :]
            p_y_given_t = (weighted @ p_y_given_x) / (p_t[:, None] + eps)
            p_y_given_t = np.maximum(p_y_given_t, eps)
            p_y_given_t /= p_y_given_t.sum(axis=1, keepdims=True)

            # I(T;Y) = sum_t p(t) * D_KL(p(y|t) || p(y))
            i_ty = 0.0
            for t in range(n_clusters):
                if p_t[t] > eps:
                    kl_t = np.sum(
                        p_y_given_t[t]
                        * np.log(p_y_given_t[t] / (p_y + eps) + eps)
                    )
                    i_ty += p_t[t] * kl_t

            if abs(i_ty - prev_i_ty) < tol:
                break
            prev_i_ty = i_ty

        # Compute I(T;X) = sum_x p(x) * D_KL(p(t|x) || p(t))
        i_tx = 0.0
        for x in range(n_x):
            if p_x[x] > eps:
                kl_x = np.sum(
                    p_t_given_x[x]
                    * np.log(p_t_given_x[x] / (p_t + eps) + eps)
                )
                i_tx += p_x[x] * kl_x

        return p_t_given_x, float(i_tx), float(i_ty)

    @staticmethod
    def ib_curve(
        joint_XY: np.ndarray,
        betas: np.ndarray | None = None,
        n_clusters: int = 2,
    ) -> list[tuple[float, float, float]]:
        """Compute the Information Bottleneck curve: I(T;X) vs I(T;Y).

        Sweeps beta from low (maximum compression) to high (maximum
        preservation) and records the tradeoff at each point.

        Args:
            joint_XY: Joint distribution P(X,Y), shape (|X|, |Y|).
            betas: Array of beta values to evaluate. If None, uses
                logspace from 0.1 to 100 with 20 points.
            n_clusters: Number of compressed states |T|.

        Returns:
            List of (I_TX, I_TY, beta) tuples, one per beta value.
        """
        if betas is None:
            betas = np.logspace(-1, 2, 20)

        curve = []
        for b in betas:
            _, i_tx, i_ty = InformationBottleneck.compress(
                joint_XY, beta=b, n_clusters=n_clusters
            )
            curve.append((i_tx, i_ty, float(b)))

        return curve

    @staticmethod
    def deterministic_ib(
        joint_XY: np.ndarray,
        n_clusters: int,
        max_iter: int = 100,
    ) -> np.ndarray:
        """Hard clustering version of the Information Bottleneck.

        Each x is assigned to exactly one cluster t, maximizing I(T;Y).

        Args:
            joint_XY: Joint distribution P(X,Y), shape (|X|, |Y|).
            n_clusters: Number of clusters.
            max_iter: Maximum iterations.

        Returns:
            Cluster assignments array of shape (|X|,), values in [0, n_clusters).
        """
        joint_XY = np.asarray(joint_XY, dtype=np.float64)
        n_x, n_y = joint_XY.shape
        eps = 1e-15

        p_x = joint_XY.sum(axis=1)
        p_y_given_x = joint_XY / (p_x[:, None] + eps)

        # Initialize with random assignment
        rng = np.random.default_rng(42)
        assignments = rng.integers(0, n_clusters, size=n_x)

        for _ in range(max_iter):
            # Compute p(y|t) for each cluster
            p_y_given_t = np.zeros((n_clusters, n_y))
            p_t = np.zeros(n_clusters)
            for t in range(n_clusters):
                mask = assignments == t
                if mask.any():
                    p_t[t] = p_x[mask].sum()
                    p_y_given_t[t] = (joint_XY[mask].sum(axis=0)) / (p_t[t] + eps)
            p_y_given_t = np.maximum(p_y_given_t, eps)
            p_y_given_t /= p_y_given_t.sum(axis=1, keepdims=True)
            p_t = np.maximum(p_t, eps)

            # Reassign each x to the cluster with smallest KL(p(y|x) || p(y|t))
            new_assignments = np.zeros(n_x, dtype=int)
            for x in range(n_x):
                best_t = 0
                best_kl = np.inf
                for t in range(n_clusters):
                    kl = np.sum(
                        p_y_given_x[x]
                        * np.log(p_y_given_x[x] / (p_y_given_t[t] + eps) + eps)
                    )
                    if kl < best_kl:
                        best_kl = kl
                        best_t = t
                new_assignments[x] = best_t

            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments

        return assignments

    @staticmethod
    def sufficient_statistic_test(
        joint_XY: np.ndarray,
        T_given_X: np.ndarray,
        tol: float = 1e-4,
    ) -> tuple[bool, float]:
        """Test if T is approximately sufficient for Y given X.

        A statistic T(X) is sufficient for Y if I(X;Y|T) approx 0,
        meaning T captures all the information X has about Y.

        I(X;Y|T) = I(X;Y) - I(T;Y)

        Args:
            joint_XY: Joint distribution P(X,Y), shape (|X|, |Y|).
            T_given_X: Conditional p(t|x), shape (|X|, |T|).
            tol: Tolerance for "approximately zero".

        Returns:
            (is_sufficient, residual_info) where residual_info = I(X;Y|T).
        """
        joint_XY = np.asarray(joint_XY, dtype=np.float64)
        T_given_X = np.asarray(T_given_X, dtype=np.float64)
        eps = 1e-15

        n_x, n_y = joint_XY.shape
        n_t = T_given_X.shape[1]

        p_x = joint_XY.sum(axis=1)
        p_y = joint_XY.sum(axis=0)

        # I(X;Y) via mutual_information
        pair_xy = DistributionPair(joint_XY)
        i_xy = mutual_information(pair_xy, base=np.e)

        # I(T;Y): build joint P(T,Y)
        # p(t,y) = sum_x p(t|x) * p(x,y)
        p_ty = T_given_X.T @ joint_XY  # (|T|, |Y|)
        p_ty_sum = p_ty.sum()
        if p_ty_sum > eps:
            p_ty /= p_ty_sum

        pair_ty = DistributionPair(p_ty)
        i_ty = mutual_information(pair_ty, base=np.e)

        residual = max(0.0, i_xy - i_ty)
        return residual < tol, float(residual)


if __name__ == "__main__":
    print("Information Theory Module Demo")
    print("=" * 50)

    # Basic entropy examples
    fair_coin = np.array([0.5, 0.5])
    biased_coin = np.array([0.9, 0.1])

    print(f"\nFair coin entropy: {entropy(fair_coin):.4f} bits")
    print(f"Biased coin (90/10) entropy: {entropy(biased_coin):.4f} bits")

    # KL divergence
    print(f"\nKL(biased || fair): {kl_divergence(biased_coin, fair_coin):.4f}")
    print(f"KL(fair || biased): {kl_divergence(fair_coin, biased_coin):.4f}")

    # JS divergence (symmetric)
    print(f"JS(biased, fair): {jensen_shannon(biased_coin, fair_coin):.4f}")

    # Mutual information
    independent = np.array([[0.25, 0.25], [0.25, 0.25]])
    dependent = np.array([[0.5, 0.0], [0.0, 0.5]])

    pair_indep = DistributionPair(independent)
    pair_dep = DistributionPair(dependent)

    print(f"\nIndependent MI: {mutual_information(pair_indep):.4f}")
    print(f"Perfectly dependent MI: {mutual_information(pair_dep):.4f}")

    print("\n" + "=" * 50)
    print("Information theory module ready!")
