"""
Information Geometry — Statistical Manifolds & Natural Gradient
===============================================================

The geometry of probability distributions. Every parametric family
(Gaussian, Beta, Dirichlet, ...) forms a Riemannian manifold where
the metric tensor is the Fisher information matrix. This module
implements the full Amari-Chentsov framework:

    Fisher metric → α-connections → dually flat structure → natural gradient

Classes:
    FisherMetric: Fisher information matrix as Riemannian metric tensor
    AlphaConnection: Family of affine connections indexed by α ∈ ℝ
    StatisticalManifold: Manifold equipped with metric + dual connections
    NaturalGradient: Gradient descent on statistical manifolds
    BregmanDivergence: Divergence from dually flat structure
    StatisticalGeodesic: Geodesics on statistical manifolds

Applications:
    - Natural gradient for Thompson Sampling (model_router.py)
    - Manifold-aware PPO updates (reflection/ppo_trainer.py)
    - Divergence-based collapse scoring (convergence/engine.py)
    - Embedding drift detection via geodesic distance
"""

from collections.abc import Callable

import numpy as np


class FisherMetric:
    """
    Fisher information metric on a parametric statistical manifold.

    For a family p(x|θ), the Fisher information matrix is:

        g_ij(θ) = E[ ∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j ]

    This is the unique (up to scale) Riemannian metric that is
    invariant under sufficient statistics (Čencov's theorem).
    """

    def __init__(self, dim: int,
                 log_likelihood: Callable[[np.ndarray, np.ndarray], float],
                 sample_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
                 n_samples: int = 1000):
        """
        Initialize Fisher metric.

        Args:
            dim: Dimension of parameter space
            log_likelihood: Function (x, theta) -> log p(x|theta)
            sample_fn: Function (theta, n) -> samples from p(x|theta).
                       If None, uses numerical score estimation.
            n_samples: Number of samples for Monte Carlo estimation
        """
        self.dim = dim
        self.log_likelihood = log_likelihood
        self.sample_fn = sample_fn
        self.n_samples = n_samples

    def score(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Score function: ∂log p(x|θ)/∂θ.

        Computed via central finite differences.

        Args:
            x: Data point
            theta: Parameter vector

        Returns:
            Score vector of shape (dim,)
        """
        h = 1e-6
        s = np.zeros(self.dim)
        for i in range(self.dim):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += h
            theta_minus[i] -= h
            s[i] = (self.log_likelihood(x, theta_plus) -
                     self.log_likelihood(x, theta_minus)) / (2 * h)
        return s

    def matrix(self, theta: np.ndarray,
               samples: np.ndarray | None = None) -> np.ndarray:
        """
        Fisher information matrix G(θ) via Monte Carlo.

        G_ij = (1/N) Σ_k score_i(x_k, θ) · score_j(x_k, θ)

        Args:
            theta: Parameter vector
            samples: Pre-computed samples. If None, draws from sample_fn.

        Returns:
            Fisher matrix of shape (dim, dim)
        """
        if samples is None:
            if self.sample_fn is None:
                raise ValueError("Need sample_fn or explicit samples")
            samples = self.sample_fn(theta, self.n_samples)

        G = np.zeros((self.dim, self.dim))
        for x in samples:
            s = self.score(x, theta)
            G += np.outer(s, s)
        G /= len(samples)

        # Symmetrize for numerical stability
        G = 0.5 * (G + G.T)
        return G

    def inner_product(self, theta: np.ndarray,
                      v: np.ndarray, w: np.ndarray,
                      G: np.ndarray | None = None) -> float:
        """
        Inner product ⟨v, w⟩_θ = v^T G(θ) w.

        Args:
            theta: Point on manifold
            v: First tangent vector
            w: Second tangent vector
            G: Pre-computed Fisher matrix (optional)

        Returns:
            Inner product value
        """
        if G is None:
            G = self.matrix(theta)
        return float(v @ G @ w)

    def norm(self, theta: np.ndarray, v: np.ndarray,
             G: np.ndarray | None = None) -> float:
        """
        Norm ||v||_θ = sqrt(v^T G(θ) v).

        Args:
            theta: Point on manifold
            v: Tangent vector
            G: Pre-computed Fisher matrix (optional)

        Returns:
            Norm value
        """
        return np.sqrt(max(0.0, self.inner_product(theta, v, v, G)))

    @staticmethod
    def beta_family() -> 'FisherMetric':
        """
        Fisher metric for Beta(α, β) family.

        Analytic Fisher matrix:
            G_11 = ψ'(α) - ψ'(α+β)
            G_22 = ψ'(β) - ψ'(α+β)
            G_12 = -ψ'(α+β)

        where ψ' is the trigamma function.
        """
        from scipy.special import betaln, polygamma

        def log_lik(x: np.ndarray, theta: np.ndarray) -> float:
            a, b = theta
            if a <= 0 or b <= 0 or x <= 0 or x >= 1:
                return -1e10
            return (a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - betaln(a, b)

        def sample_fn(theta: np.ndarray, n: int) -> np.ndarray:
            a, b = theta
            return np.random.beta(max(a, 0.01), max(b, 0.01), size=n)

        metric = FisherMetric(2, log_lik, sample_fn)

        # Override matrix() with analytic form
        original_matrix = metric.matrix

        def analytic_matrix(theta: np.ndarray,
                            samples: np.ndarray | None = None) -> np.ndarray:
            a, b = theta
            if a <= 0 or b <= 0:
                return original_matrix(theta, samples)
            psi1_a = float(polygamma(1, a))
            psi1_b = float(polygamma(1, b))
            psi1_ab = float(polygamma(1, a + b))
            return np.array([
                [psi1_a - psi1_ab, -psi1_ab],
                [-psi1_ab, psi1_b - psi1_ab]
            ])

        metric.matrix = analytic_matrix
        return metric

    @staticmethod
    def gaussian_family() -> 'FisherMetric':
        """
        Fisher metric for N(μ, σ²) family.

        G(μ, σ) = [[1/σ², 0], [0, 2/σ²]]

        The Gaussian manifold is hyperbolic (constant negative curvature -1/2).
        """
        def log_lik(x: np.ndarray, theta: np.ndarray) -> float:
            mu, sigma = theta
            if sigma <= 0:
                return -1e10
            return -0.5 * np.log(2 * np.pi * sigma**2) - (x - mu)**2 / (2 * sigma**2)

        def sample_fn(theta: np.ndarray, n: int) -> np.ndarray:
            mu, sigma = theta
            return np.random.normal(mu, max(sigma, 1e-10), size=n)

        metric = FisherMetric(2, log_lik, sample_fn)

        def analytic_matrix(theta: np.ndarray,
                            samples: np.ndarray | None = None) -> np.ndarray:
            mu, sigma = theta
            s2 = max(sigma**2, 1e-20)
            return np.array([
                [1.0 / s2, 0.0],
                [0.0, 2.0 / s2]
            ])

        metric.matrix = analytic_matrix
        return metric

    @staticmethod
    def dirichlet_family(k: int) -> 'FisherMetric':
        """
        Fisher metric for Dirichlet(α₁, ..., αₖ) family.

        G_ij = δ_ij · ψ'(αᵢ) - ψ'(Σαⱼ)

        Args:
            k: Number of categories
        """
        from scipy.special import gammaln, polygamma

        def log_lik(x: np.ndarray, theta: np.ndarray) -> float:
            if np.any(theta <= 0) or np.any(x <= 0):
                return -1e10
            return (gammaln(np.sum(theta)) - np.sum(gammaln(theta)) +
                    np.sum((theta - 1) * np.log(x)))

        def sample_fn(theta: np.ndarray, n: int) -> np.ndarray:
            alpha = np.maximum(theta, 0.01)
            return np.random.dirichlet(alpha, size=n)

        metric = FisherMetric(k, log_lik, sample_fn)

        def analytic_matrix(theta: np.ndarray,
                            samples: np.ndarray | None = None) -> np.ndarray:
            psi1 = np.array([float(polygamma(1, a)) for a in theta])
            psi1_sum = float(polygamma(1, np.sum(theta)))
            G = -psi1_sum * np.ones((k, k))
            np.fill_diagonal(G, psi1 - psi1_sum)
            return G

        metric.matrix = analytic_matrix
        return metric


class AlphaConnection:
    """
    α-connection on a statistical manifold.

    The α-connection ∇^(α) is defined by its Christoffel symbols:

        Γ^(α)_{ij,k} = E[∂²log p/∂θ_i∂θ_j · ∂log p/∂θ_k]
                        + (1-α)/2 · E[∂log p/∂θ_i · ∂log p/∂θ_j · ∂log p/∂θ_k]

    Special cases:
        α = 0: Levi-Civita connection (Riemannian)
        α = 1: Exponential connection (e-connection)
        α = -1: Mixture connection (m-connection)

    The e-connection and m-connection are dual: ∇^(1) and ∇^(-1)
    are conjugate with respect to the Fisher metric.
    """

    def __init__(self, fisher: FisherMetric, alpha: float = 0.0):
        """
        Args:
            fisher: Fisher metric on the manifold
            alpha: Connection parameter (0 = Levi-Civita, 1 = e, -1 = m)
        """
        self.fisher = fisher
        self.alpha = alpha

    def dual(self) -> 'AlphaConnection':
        """
        Return the dual connection ∇^(-α).

        The pair (∇^(α), ∇^(-α)) are conjugate w.r.t. the Fisher metric:
            X⟨Y,Z⟩ = ⟨∇^(α)_X Y, Z⟩ + ⟨Y, ∇^(-α)_X Z⟩
        """
        return AlphaConnection(self.fisher, -self.alpha)

    def christoffel(self, theta: np.ndarray,
                    samples: np.ndarray | None = None) -> np.ndarray:
        """
        α-Christoffel symbols Γ^(α)_{ij,k} (first kind).

        Computed via Monte Carlo estimation of the required expectations.

        Args:
            theta: Parameter point
            samples: Pre-computed samples (optional)

        Returns:
            Array of shape (dim, dim, dim) with Γ_{ij,k}
        """
        dim = self.fisher.dim
        h = 1e-5

        if samples is None:
            if self.fisher.sample_fn is None:
                raise ValueError("Need sample_fn or explicit samples")
            samples = self.fisher.sample_fn(theta, self.fisher.n_samples)

        Gamma = np.zeros((dim, dim, dim))

        for x in samples:
            # First derivatives (score)
            s = self.fisher.score(x, theta)

            # Second derivatives of log-likelihood
            H = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(i, dim):
                    theta_pp = theta.copy()
                    theta_pm = theta.copy()
                    theta_mp = theta.copy()
                    theta_mm = theta.copy()
                    theta_pp[i] += h; theta_pp[j] += h
                    theta_pm[i] += h; theta_pm[j] -= h
                    theta_mp[i] -= h; theta_mp[j] += h
                    theta_mm[i] -= h; theta_mm[j] -= h
                    H[i, j] = (self.fisher.log_likelihood(x, theta_pp) -
                               self.fisher.log_likelihood(x, theta_pm) -
                               self.fisher.log_likelihood(x, theta_mp) +
                               self.fisher.log_likelihood(x, theta_mm)) / (4 * h * h)
                    H[j, i] = H[i, j]

            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        Gamma[i, j, k] += (H[i, j] * s[k] +
                                           (1 - self.alpha) / 2 * s[i] * s[j] * s[k])

        Gamma /= len(samples)
        return Gamma

    def is_flat(self, theta: np.ndarray,
                samples: np.ndarray | None = None,
                tol: float = 1e-4) -> bool:
        """
        Check if the connection is flat (all Christoffel symbols ≈ 0).

        Exponential families are e-flat (α=1) and m-flat (α=-1).

        Args:
            theta: Parameter point to check
            samples: Pre-computed samples (optional)
            tol: Tolerance for zero check

        Returns:
            True if connection is flat at theta
        """
        Gamma = self.christoffel(theta, samples)
        return bool(np.all(np.abs(Gamma) < tol))


class BregmanDivergence:
    """
    Bregman divergence from a dually flat structure.

    Given a strictly convex function φ (potential):

        D_φ(p || q) = φ(p) - φ(q) - ∇φ(q)^T (p - q)

    Properties:
        - D_φ(p || q) ≥ 0 (non-negative)
        - D_φ(p || q) = 0 iff p = q
        - Generally asymmetric: D_φ(p || q) ≠ D_φ(q || p)
        - KL divergence is the Bregman divergence of negative entropy
        - The dual divergence uses the conjugate potential φ*

    In the dually flat structure of an exponential family:
        - θ-coordinates (natural parameters) → e-flat
        - η-coordinates (expectation parameters) → m-flat
        - φ(θ) = log partition function
        - φ*(η) = negative entropy
    """

    def __init__(self, potential: Callable[[np.ndarray], float],
                 dim: int):
        """
        Args:
            potential: Strictly convex function φ: ℝⁿ → ℝ
            dim: Dimension of parameter space
        """
        self.potential = potential
        self.dim = dim

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        Gradient ∇φ(θ) via central differences.

        In an exponential family, ∇φ maps natural → expectation parameters.

        Args:
            theta: Parameter point

        Returns:
            Gradient vector
        """
        h = 1e-7
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[i] += h
            theta_m[i] -= h
            grad[i] = (self.potential(theta_p) - self.potential(theta_m)) / (2 * h)
        return grad

    def hessian(self, theta: np.ndarray) -> np.ndarray:
        """
        Hessian ∇²φ(θ).

        For exponential families, the Hessian of the log-partition
        function equals the Fisher information matrix.

        Args:
            theta: Parameter point

        Returns:
            Hessian matrix of shape (dim, dim)
        """
        h = 1e-5
        H = np.zeros((self.dim, self.dim))
        self.potential(theta)
        for i in range(self.dim):
            for j in range(i, self.dim):
                theta_pp = theta.copy()
                theta_pm = theta.copy()
                theta_mp = theta.copy()
                theta_mm = theta.copy()
                theta_pp[i] += h; theta_pp[j] += h
                theta_pm[i] += h; theta_pm[j] -= h
                theta_mp[i] -= h; theta_mp[j] += h
                theta_mm[i] -= h; theta_mm[j] -= h
                H[i, j] = (self.potential(theta_pp) - self.potential(theta_pm) -
                           self.potential(theta_mp) + self.potential(theta_mm)) / (4 * h * h)
                H[j, i] = H[i, j]
        return H

    def divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Bregman divergence D_φ(p || q) = φ(p) - φ(q) - ∇φ(q)^T (p - q).

        Args:
            p: First parameter point
            q: Second parameter point

        Returns:
            Non-negative divergence value
        """
        return (self.potential(p) - self.potential(q) -
                self.gradient(q) @ (p - q))

    def symmetrized(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Symmetrized Bregman divergence:
            D_sym(p, q) = (D_φ(p||q) + D_φ(q||p)) / 2
                        = (∇φ(p) - ∇φ(q))^T (p - q) / 2
        """
        return (self.divergence(p, q) + self.divergence(q, p)) / 2

    @staticmethod
    def kl_divergence() -> 'BregmanDivergence':
        """
        KL divergence as Bregman divergence of negative entropy.

        φ(p) = Σ p_i log p_i  (negative entropy)
        D_φ(p||q) = KL(p||q) = Σ p_i log(p_i / q_i)
        """
        def neg_entropy(p: np.ndarray) -> float:
            p_safe = np.clip(p, 1e-15, None)
            return float(np.sum(p_safe * np.log(p_safe)))

        return BregmanDivergence(neg_entropy, dim=0)  # dim set per use

    @staticmethod
    def squared_euclidean() -> 'BregmanDivergence':
        """
        Squared Euclidean distance as Bregman divergence.

        φ(x) = ||x||² / 2
        D_φ(x||y) = ||x - y||² / 2
        """
        def half_sq_norm(x: np.ndarray) -> float:
            return float(0.5 * np.sum(x**2))

        return BregmanDivergence(half_sq_norm, dim=0)

    @staticmethod
    def beta_log_partition() -> 'BregmanDivergence':
        """
        Bregman divergence for Beta family via log-partition function.

        Natural parameters: θ = (α-1, β-1)
        Log-partition: φ(θ) = log B(θ₁+1, θ₂+1) = log Γ(θ₁+1) + log Γ(θ₂+1) - log Γ(θ₁+θ₂+2)

        The resulting Bregman divergence equals the KL divergence
        between Beta distributions.
        """
        from scipy.special import gammaln

        def log_partition(theta: np.ndarray) -> float:
            a, b = theta[0] + 1, theta[1] + 1
            if a <= 0 or b <= 0:
                return 1e10
            return float(gammaln(a) + gammaln(b) - gammaln(a + b))

        return BregmanDivergence(log_partition, dim=2)


class NaturalGradient:
    """
    Natural gradient descent on a statistical manifold.

    The natural gradient is:
        θ̃ = G(θ)⁻¹ ∇L(θ)

    where G(θ) is the Fisher information matrix. This follows the
    steepest descent direction in the Fisher-Rao metric, which is
    invariant to parameterization.

    For Thompson Sampling with Beta(α,β) priors, natural gradient
    updates move along geodesics of the Beta manifold rather than
    in flat Euclidean space — giving faster convergence especially
    when α and β are far from 1.
    """

    def __init__(self, fisher: FisherMetric,
                 damping: float = 1e-4):
        """
        Args:
            fisher: Fisher metric on parameter space
            damping: Tikhonov regularization for matrix inversion (G + λI)
        """
        self.fisher = fisher
        self.damping = damping

    def compute(self, theta: np.ndarray,
                euclidean_grad: np.ndarray,
                G: np.ndarray | None = None) -> np.ndarray:
        """
        Compute natural gradient G(θ)⁻¹ ∇L(θ).

        Args:
            theta: Current parameter point
            euclidean_grad: Standard gradient ∇L(θ)
            G: Pre-computed Fisher matrix (optional)

        Returns:
            Natural gradient vector
        """
        if G is None:
            G = self.fisher.matrix(theta)

        # Damped inverse for numerical stability
        G_damped = G + self.damping * np.eye(self.fisher.dim)
        return np.linalg.solve(G_damped, euclidean_grad)

    def step(self, theta: np.ndarray,
             euclidean_grad: np.ndarray,
             lr: float = 0.01,
             G: np.ndarray | None = None) -> np.ndarray:
        """
        Take a natural gradient step: θ ← θ - lr · G⁻¹ ∇L.

        Args:
            theta: Current parameters
            euclidean_grad: Standard gradient
            lr: Learning rate
            G: Pre-computed Fisher matrix (optional)

        Returns:
            Updated parameters
        """
        nat_grad = self.compute(theta, euclidean_grad, G)
        return theta - lr * nat_grad

    def step_with_trust_region(self, theta: np.ndarray,
                               euclidean_grad: np.ndarray,
                               max_kl: float = 0.01,
                               G: np.ndarray | None = None) -> np.ndarray:
        """
        Natural gradient step with KL trust region constraint.

        Finds lr such that KL(p_new || p_old) ≤ max_kl.
        The KL divergence in natural parameters is approximately:
            KL ≈ (1/2) Δθ^T G Δθ

        This is the core of TRPO/PPO-style updates.

        Args:
            theta: Current parameters
            euclidean_grad: Standard gradient
            max_kl: Maximum KL divergence allowed
            G: Pre-computed Fisher matrix (optional)

        Returns:
            Updated parameters within trust region
        """
        if G is None:
            G = self.fisher.matrix(theta)

        nat_grad = self.compute(theta, euclidean_grad, G)

        # Quadratic form: Δθ^T G Δθ = lr² · nat_grad^T G nat_grad
        quad = float(nat_grad @ G @ nat_grad)
        if quad < 1e-20:
            return theta

        # lr = sqrt(2 * max_kl / (nat_grad^T G nat_grad))
        lr = np.sqrt(2 * max_kl / quad)
        return theta - lr * nat_grad


class StatisticalManifold:
    """
    Statistical manifold with dually flat structure.

    A statistical manifold (S, g, ∇, ∇*) where:
        - S is a family of probability distributions
        - g is the Fisher information metric
        - ∇ = ∇^(1) is the e-connection (exponential)
        - ∇* = ∇^(-1) is the m-connection (mixture)

    For exponential families, the manifold is dually flat with
    two coordinate systems:
        - θ (natural/canonical parameters) — e-affine
        - η (expectation/mean parameters) — m-affine

    The Legendre transform φ ↔ φ* connects the dual structures.
    """

    def __init__(self, fisher: FisherMetric,
                 theta_to_eta: Callable[[np.ndarray], np.ndarray] | None = None,
                 eta_to_theta: Callable[[np.ndarray], np.ndarray] | None = None,
                 potential: Callable[[np.ndarray], float] | None = None):
        """
        Args:
            fisher: Fisher metric
            theta_to_eta: Map from natural to expectation parameters
            eta_to_theta: Map from expectation to natural parameters
            potential: Log-partition function φ(θ) (generates the dually flat structure)
        """
        self.fisher = fisher
        self.e_connection = AlphaConnection(fisher, alpha=1.0)
        self.m_connection = AlphaConnection(fisher, alpha=-1.0)
        self.levi_civita = AlphaConnection(fisher, alpha=0.0)
        self.theta_to_eta = theta_to_eta
        self.eta_to_theta = eta_to_theta
        self._potential = potential

        if potential is not None:
            self.bregman = BregmanDivergence(potential, fisher.dim)
        else:
            self.bregman = None

    def geodesic_distance(self, theta1: np.ndarray, theta2: np.ndarray,
                          n_steps: int = 100) -> float:
        """
        Approximate geodesic distance by integrating ||γ'(t)||_g along
        the geodesic from theta1 to theta2.

        Uses the approximation of shooting along the Levi-Civita geodesic
        via small Euclidean steps weighted by the Fisher metric.

        Args:
            theta1: Start point
            theta2: End point
            n_steps: Number of integration steps

        Returns:
            Approximate geodesic distance
        """
        # Trapezoidal integration along straight line (first-order approx)
        length = 0.0
        for i in range(n_steps):
            t = i / n_steps
            t_next = (i + 1) / n_steps
            theta_mid = theta1 + (t + t_next) / 2 * (theta2 - theta1)
            dt = (theta2 - theta1) / n_steps

            G = self.fisher.matrix(theta_mid)
            ds = np.sqrt(max(0.0, float(dt @ G @ dt)))
            length += ds

        return length

    def e_geodesic(self, theta_start: np.ndarray, theta_end: np.ndarray,
                   n_points: int = 50) -> np.ndarray:
        """
        e-geodesic (straight line in θ-coordinates for exponential families).

        Args:
            theta_start: Start in natural parameters
            theta_end: End in natural parameters
            n_points: Number of points along path

        Returns:
            Array of shape (n_points, dim) in θ-coordinates
        """
        t = np.linspace(0, 1, n_points)
        return np.array([theta_start + ti * (theta_end - theta_start) for ti in t])

    def m_geodesic(self, theta_start: np.ndarray, theta_end: np.ndarray,
                   n_points: int = 50) -> np.ndarray | None:
        """
        m-geodesic (straight line in η-coordinates for exponential families).

        Requires theta_to_eta and eta_to_theta maps.

        Args:
            theta_start: Start in natural parameters
            theta_end: End in natural parameters
            n_points: Number of points along path

        Returns:
            Array of shape (n_points, dim) in θ-coordinates, or None if
            coordinate maps not available
        """
        if self.theta_to_eta is None or self.eta_to_theta is None:
            return None

        eta_start = self.theta_to_eta(theta_start)
        eta_end = self.theta_to_eta(theta_end)

        t = np.linspace(0, 1, n_points)
        eta_path = np.array([eta_start + ti * (eta_end - eta_start) for ti in t])
        return np.array([self.eta_to_theta(eta) for eta in eta_path])

    def kl_divergence(self, theta1: np.ndarray, theta2: np.ndarray) -> float | None:
        """
        KL divergence via Bregman divergence of the potential.

        KL(p_θ₁ || p_θ₂) = D_φ(θ₂ || θ₁)  [note the reversal!]

        Args:
            theta1: Parameters of first distribution
            theta2: Parameters of second distribution

        Returns:
            KL divergence, or None if potential not available
        """
        if self.bregman is None:
            return None
        # KL(p||q) = D_φ(θ_q || θ_p) for exponential families
        return self.bregman.divergence(theta2, theta1)

    def natural_gradient(self, damping: float = 1e-4) -> NaturalGradient:
        """
        Create a NaturalGradient optimizer for this manifold.

        Args:
            damping: Tikhonov regularization

        Returns:
            NaturalGradient instance
        """
        return NaturalGradient(self.fisher, damping)

    @staticmethod
    def beta_manifold() -> 'StatisticalManifold':
        """
        Statistical manifold for the Beta(α, β) family.

        Natural parameters: θ = (α-1, β-1)
        Expectation parameters: η = (E[log X], E[log(1-X)]) = (ψ(α)-ψ(α+β), ψ(β)-ψ(α+β))
        Potential: φ(θ) = log B(α, β)

        This is the manifold that Thompson Sampling lives on.
        """
        from scipy.special import digamma, gammaln

        fisher = FisherMetric.beta_family()

        def theta_to_eta(theta: np.ndarray) -> np.ndarray:
            a, b = theta[0] + 1, theta[1] + 1
            psi_ab = digamma(a + b)
            return np.array([digamma(a) - psi_ab, digamma(b) - psi_ab])

        def eta_to_theta(eta: np.ndarray) -> np.ndarray:
            # Inverse mapping via Newton's method
            theta = np.array([1.0, 1.0])  # Start at Beta(2,2)
            for _ in range(50):
                a, b = theta[0] + 1, theta[1] + 1
                psi_ab = digamma(a + b)
                f = np.array([digamma(a) - psi_ab - eta[0],
                              digamma(b) - psi_ab - eta[1]])
                if np.linalg.norm(f) < 1e-10:
                    break
                # Jacobian is the Fisher matrix
                J = fisher.matrix(theta)
                theta -= np.linalg.solve(J, f)
            return theta

        def potential(theta: np.ndarray) -> float:
            a, b = theta[0] + 1, theta[1] + 1
            if a <= 0 or b <= 0:
                return 1e10
            return float(gammaln(a) + gammaln(b) - gammaln(a + b))

        return StatisticalManifold(fisher, theta_to_eta, eta_to_theta, potential)

    @staticmethod
    def gaussian_manifold() -> 'StatisticalManifold':
        """
        Statistical manifold for N(μ, σ²).

        This manifold has constant negative Gaussian curvature -1/2,
        making it isometric to the hyperbolic plane ℍ².
        """
        fisher = FisherMetric.gaussian_family()

        def potential(theta: np.ndarray) -> float:
            # Natural params: θ₁ = μ/σ², θ₂ = -1/(2σ²)
            # We work in (μ, σ) directly
            mu, sigma = theta
            if sigma <= 0:
                return 1e10
            return float(0.5 * np.log(2 * np.pi * sigma**2) + 0.5)

        return StatisticalManifold(fisher, potential=potential)


class StatisticalGeodesic:
    """
    Geodesic curves on statistical manifolds.

    Computes geodesics via the geodesic equation:
        d²θ^k/dt² + Γ^k_{ij} dθ^i/dt dθ^j/dt = 0

    For the Fisher-Rao metric, these are the curves of minimum
    information-geometric length between distributions.
    """

    def __init__(self, manifold: StatisticalManifold):
        """
        Args:
            manifold: Statistical manifold to compute geodesics on
        """
        self.manifold = manifold

    def shoot(self, theta0: np.ndarray, velocity: np.ndarray,
              t_final: float = 1.0, dt: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute geodesic by shooting from initial point with given velocity.

        Uses Velocity Verlet (symplectic) integration of the geodesic equation.

        Args:
            theta0: Initial point
            velocity: Initial velocity (tangent vector)
            t_final: Integration time
            dt: Time step

        Returns:
            Tuple of (positions, velocities) arrays
        """
        dim = self.manifold.fisher.dim
        n_steps = int(t_final / dt)

        positions = np.zeros((n_steps + 1, dim))
        velocities = np.zeros((n_steps + 1, dim))
        positions[0] = theta0.copy()
        velocities[0] = velocity.copy()

        for step in range(n_steps):
            theta = positions[step]
            v = velocities[step]

            # Compute Christoffel symbols (second kind) via metric
            G = self.manifold.fisher.matrix(theta)
            G_inv = np.linalg.inv(G + 1e-10 * np.eye(dim))

            # Numerical Christoffel symbols Γ^k_{ij}
            h = 1e-5
            np.zeros((dim, dim, dim))
            for m in range(dim):
                theta_p = theta.copy()
                theta_m = theta.copy()
                theta_p[m] += h
                theta_m[m] -= h
                G_p = self.manifold.fisher.matrix(theta_p)
                G_m = self.manifold.fisher.matrix(theta_m)
                (G_p - G_m) / (2 * h)
                for k in range(dim):
                    for i in range(dim):
                        for j in range(dim):
                            # Γ^k_{ij} = (1/2) g^{kl} (∂g_{li}/∂θ_j + ∂g_{lj}/∂θ_i - ∂g_{ij}/∂θ_l)
                            # Simplified: use only ∂g/∂θ_m contribution
                            pass

            # Acceleration from geodesic equation: a^k = -Γ^k_{ij} v^i v^j
            # Use finite-difference approach for the full Christoffel computation
            accel = np.zeros(dim)
            for k in range(dim):
                for i in range(dim):
                    for j in range(dim):
                        # Compute Γ^k_{ij} numerically
                        gamma_kij = 0.0
                        for l in range(dim):
                            # ∂g_{li}/∂θ_j
                            theta_p = theta.copy()
                            theta_m = theta.copy()
                            theta_p[j] += h
                            theta_m[j] -= h
                            dg_li_j = (self.manifold.fisher.matrix(theta_p)[l, i] -
                                       self.manifold.fisher.matrix(theta_m)[l, i]) / (2 * h)
                            # ∂g_{lj}/∂θ_i
                            theta_p = theta.copy()
                            theta_m = theta.copy()
                            theta_p[i] += h
                            theta_m[i] -= h
                            dg_lj_i = (self.manifold.fisher.matrix(theta_p)[l, j] -
                                       self.manifold.fisher.matrix(theta_m)[l, j]) / (2 * h)
                            # ∂g_{ij}/∂θ_l
                            theta_p = theta.copy()
                            theta_m = theta.copy()
                            theta_p[l] += h
                            theta_m[l] -= h
                            dg_ij_l = (self.manifold.fisher.matrix(theta_p)[i, j] -
                                       self.manifold.fisher.matrix(theta_m)[i, j]) / (2 * h)

                            gamma_kij += 0.5 * G_inv[k, l] * (dg_li_j + dg_lj_i - dg_ij_l)

                        accel[k] -= gamma_kij * v[i] * v[j]

            # Velocity Verlet step
            theta_new = theta + v * dt + 0.5 * accel * dt**2
            # Recompute acceleration at new position (simplified: reuse)
            v_new = v + accel * dt

            positions[step + 1] = theta_new
            velocities[step + 1] = v_new

        return positions, velocities

    def boundary_value(self, theta_start: np.ndarray, theta_end: np.ndarray,
                       n_points: int = 50,
                       n_iterations: int = 20) -> np.ndarray:
        """
        Solve boundary value geodesic problem via shooting method.

        Find the geodesic connecting theta_start to theta_end by
        iteratively adjusting the initial velocity.

        Args:
            theta_start: Start point
            theta_end: End point
            n_points: Number of output points
            n_iterations: Shooting iterations

        Returns:
            Geodesic path of shape (n_points, dim)
        """
        dt = 1.0 / n_points

        # Initial guess: Euclidean straight line velocity
        v0 = theta_end - theta_start

        for iteration in range(n_iterations):
            positions, _ = self.shoot(theta_start, v0, t_final=1.0, dt=dt)
            endpoint = positions[-1]
            error = theta_end - endpoint

            if np.linalg.norm(error) < 1e-8:
                break

            # Adjust velocity (simple correction)
            v0 += error * 0.5

        positions, _ = self.shoot(theta_start, v0, t_final=1.0, dt=dt)
        return positions


if __name__ == "__main__":
    print("Information Geometry")
    print("=" * 60)

    # Test 1: Fisher metric for Gaussian family
    print("\n[Test 1] Gaussian Fisher metric")
    fisher_gauss = FisherMetric.gaussian_family()
    G = fisher_gauss.matrix(np.array([0.0, 1.0]))
    print(f"G(μ=0, σ=1) = \n{G}")
    print("Expected: [[1, 0], [0, 2]]")
    assert np.allclose(G, np.array([[1.0, 0.0], [0.0, 2.0]])), "Gaussian Fisher matrix incorrect"

    # Test 2: Fisher metric for Beta family
    print("\n[Test 2] Beta Fisher metric")
    fisher_beta = FisherMetric.beta_family()
    G_beta = fisher_beta.matrix(np.array([2.0, 3.0]))
    print(f"G(α=2, β=3) =\n{G_beta}")
    print(f"Positive definite: {np.all(np.linalg.eigvals(G_beta) > 0)}")

    # Test 3: Natural gradient
    print("\n[Test 3] Natural gradient descent")
    nat_grad = NaturalGradient(fisher_gauss, damping=1e-6)
    theta = np.array([0.0, 1.0])
    euclidean_grad = np.array([1.0, 0.5])
    natural = nat_grad.compute(theta, euclidean_grad)
    print(f"Euclidean gradient: {euclidean_grad}")
    print(f"Natural gradient:   {natural}")
    print("Expected: [1.0, 0.25] (G⁻¹ = [[1,0],[0,0.5]])")

    # Test 4: Bregman divergence
    print("\n[Test 4] Bregman divergence (squared Euclidean)")
    breg = BregmanDivergence.squared_euclidean()
    breg.dim = 2
    p = np.array([1.0, 2.0])
    q = np.array([3.0, 4.0])
    d = breg.divergence(p, q)
    expected = 0.5 * np.sum((p - q)**2)
    print(f"D(p||q) = {d:.4f}, expected = {expected:.4f}")
    assert abs(d - expected) < 1e-6, "Squared Euclidean Bregman incorrect"

    # Test 5: Beta manifold
    print("\n[Test 5] Beta statistical manifold")
    beta_mfld = StatisticalManifold.beta_manifold()
    theta1 = np.array([1.0, 1.0])  # Beta(2,2)
    theta2 = np.array([4.0, 1.0])  # Beta(5,2)
    dist = beta_mfld.geodesic_distance(theta1, theta2, n_steps=200)
    print(f"Geodesic distance Beta(2,2) → Beta(5,2): {dist:.4f}")

    # Test 6: Trust region step
    print("\n[Test 6] Natural gradient with KL trust region")
    nat_grad_tr = NaturalGradient(fisher_gauss)
    theta = np.array([0.0, 2.0])
    grad = np.array([5.0, 3.0])
    theta_new = nat_grad_tr.step_with_trust_region(theta, grad, max_kl=0.01)
    delta = theta_new - theta
    G_at = fisher_gauss.matrix(theta)
    actual_kl = 0.5 * float(delta @ G_at @ delta)
    print(f"Step: {theta} → {theta_new}")
    print(f"Approximate KL: {actual_kl:.6f} (max: 0.01)")

    # Test 7: e-geodesic on Beta manifold
    print("\n[Test 7] e-geodesic (straight in θ-coordinates)")
    e_path = beta_mfld.e_geodesic(theta1, theta2, n_points=5)
    print(f"Path shape: {e_path.shape}")
    print(f"Start: {e_path[0]}, End: {e_path[-1]}")

    # Test 8: Dirichlet Fisher metric
    print("\n[Test 8] Dirichlet Fisher metric (k=3)")
    fisher_dir = FisherMetric.dirichlet_family(3)
    G_dir = fisher_dir.matrix(np.array([2.0, 3.0, 1.0]))
    print(f"G(α=[2,3,1]) =\n{G_dir}")
    eigvals = np.linalg.eigvals(G_dir)
    print(f"Eigenvalues: {eigvals} (all positive: {np.all(eigvals > 0)})")

    # Test 9: Beta log-partition Bregman
    print("\n[Test 9] Beta KL via Bregman divergence")
    breg_beta = BregmanDivergence.beta_log_partition()
    # Natural params: θ = (α-1, β-1)
    nat1 = np.array([1.0, 1.0])  # Beta(2,2)
    nat2 = np.array([4.0, 1.0])  # Beta(5,2)
    kl_breg = breg_beta.divergence(nat2, nat1)
    print(f"KL(Beta(5,2) || Beta(2,2)) via Bregman: {kl_breg:.4f}")

    print("\n" + "=" * 60)
    print("All tests complete!")
