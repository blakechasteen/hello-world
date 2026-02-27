"""
Probability Theory for HoloLoom Warp Drive
==========================================

Measure-theoretic probability and statistical inference.

Core Concepts:
- Probability Spaces: (Ω, F, P) where P is a probability measure
- Random Variables: Measurable functions X: Ω → ℝ
- Distributions: CDF, PDF, common distributions
- Limit Theorems: Law of Large Numbers, Central Limit Theorem
- Statistical Inference: MLE, Bayesian inference, hypothesis testing

Mathematical Foundation:
Probability measure: P(Ω) = 1, P(∅) = 0
Random variable: X^{-1}(B) ∈ F for all Borel sets B
Expectation: E[X] = ∫ X dP
Law of Large Numbers: (1/n) Σ X_i → E[X] a.s.
Central Limit Theorem: √n (X̄ - μ) → N(0, σ²)

Applications to Warp Space:
- Uncertainty quantification in embeddings
- Probabilistic knowledge graphs
- Bayesian inference for decision making
- Statistical learning theory foundations

Author: HoloLoom Team
Date: 2025-10-26
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# PROBABILITY SPACES
# ============================================================================

class ProbabilitySpace:
    """
    Probability space (Ω, F, P).

    - Ω: Sample space
    - F: Sigma-algebra (event space)
    - P: Probability measure (P(Ω) = 1)
    """

    def __init__(self, outcomes: List, probabilities: Optional[List[float]] = None):
        """
        Initialize discrete probability space.

        Args:
            outcomes: List of possible outcomes
            probabilities: Probability of each outcome (uniform if None)
        """
        self.outcomes = outcomes
        n = len(outcomes)

        if probabilities is None:
            probabilities = [1.0 / n] * n

        # Verify probabilities sum to 1
        total = sum(probabilities)
        if abs(total - 1.0) > 1e-10:
            raise ValueError(f"Probabilities must sum to 1, got {total}")

        self.probabilities = {outcome: p for outcome, p in zip(outcomes, probabilities)}
        logger.info(f"Probability space with {n} outcomes")

    def probability(self, event: Union[List, set]) -> float:
        """
        Compute P(event) for a subset of outcomes.
        """
        prob = 0.0
        for outcome in event:
            if outcome in self.probabilities:
                prob += self.probabilities[outcome]
        return prob

    def sample(self, size: int = 1) -> List:
        """
        Sample from the probability space.
        """
        return np.random.choice(
            self.outcomes,
            size=size,
            p=list(self.probabilities.values())
        ).tolist()


# ============================================================================
# RANDOM VARIABLES
# ============================================================================

@dataclass
class Distribution:
    """
    Probability distribution specification.
    """
    name: str
    pdf: Optional[Callable[[float], float]] = None
    cdf: Optional[Callable[[float], float]] = None
    mean: Optional[float] = None
    variance: Optional[float] = None
    support: Optional[Tuple[float, float]] = None


class RandomVariable:
    """
    Random variable: measurable function X: Ω → ℝ.

    In discrete case, specified by probability mass function.
    In continuous case, specified by probability density function.
    """

    def __init__(
        self,
        distribution: Distribution,
        discrete: bool = False,
        values: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None
    ):
        """
        Initialize random variable.

        Args:
            distribution: Distribution specification
            discrete: True for discrete RV, False for continuous
            values: Discrete values (if discrete=True)
            probabilities: Probabilities for each value (if discrete=True)
        """
        self.distribution = distribution
        self.discrete = discrete

        if discrete:
            if values is None or probabilities is None:
                raise ValueError("Discrete RV requires values and probabilities")
            self.values = values
            self.probabilities = probabilities
        else:
            self.values = None
            self.probabilities = None

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate samples from the distribution.
        """
        if self.discrete:
            return np.random.choice(self.values, size=size, p=self.probabilities)
        else:
            # Use distribution name to sample
            if self.distribution.name == "normal":
                mean = self.distribution.mean or 0
                std = np.sqrt(self.distribution.variance or 1)
                return np.random.normal(mean, std, size=size)
            elif self.distribution.name == "exponential":
                rate = 1.0 / self.distribution.mean if self.distribution.mean else 1.0
                return np.random.exponential(1.0 / rate, size=size)
            elif self.distribution.name == "uniform":
                a, b = self.distribution.support or (0, 1)
                return np.random.uniform(a, b, size=size)
            else:
                raise NotImplementedError(f"Sampling for {self.distribution.name} not implemented")

    def expectation(self, g: Optional[Callable[[float], float]] = None) -> float:
        """
        Compute E[g(X)].

        If g is None, computes E[X].
        """
        if g is None:
            g = lambda x: x

        if self.discrete:
            return np.sum([g(v) * p for v, p in zip(self.values, self.probabilities)])
        else:
            # Use Monte Carlo estimation
            samples = self.sample(size=10000)
            return np.mean([g(x) for x in samples])

    def variance(self) -> float:
        """
        Compute Var(X) = E[(X - E[X])²].
        """
        if self.distribution.variance is not None:
            return self.distribution.variance

        mean = self.expectation()
        return self.expectation(lambda x: (x - mean) ** 2)


# ============================================================================
# COMMON DISTRIBUTIONS
# ============================================================================

class CommonDistributions:
    """
    Standard probability distributions.
    """

    @staticmethod
    def normal(mu: float = 0, sigma2: float = 1) -> RandomVariable:
        """
        Normal (Gaussian) distribution N(μ, σ²).

        PDF: f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
        """
        dist = Distribution(
            name="normal",
            pdf=lambda x: (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-(x - mu)**2 / (2 * sigma2)),
            cdf=lambda x: stats.norm.cdf(x, loc=mu, scale=np.sqrt(sigma2)),
            mean=mu,
            variance=sigma2,
            support=(-np.inf, np.inf)
        )
        return RandomVariable(dist, discrete=False)

    @staticmethod
    def exponential(rate: float = 1.0) -> RandomVariable:
        """
        Exponential distribution Exp(λ).

        PDF: f(x) = λ exp(-λx) for x ≥ 0
        Mean: 1/λ
        """
        dist = Distribution(
            name="exponential",
            pdf=lambda x: rate * np.exp(-rate * x) if x >= 0 else 0,
            cdf=lambda x: 1 - np.exp(-rate * x) if x >= 0 else 0,
            mean=1.0 / rate,
            variance=1.0 / rate**2,
            support=(0, np.inf)
        )
        return RandomVariable(dist, discrete=False)

    @staticmethod
    def uniform(a: float = 0, b: float = 1) -> RandomVariable:
        """
        Uniform distribution U(a, b).

        PDF: f(x) = 1/(b-a) for x ∈ [a, b]
        """
        dist = Distribution(
            name="uniform",
            pdf=lambda x: 1.0 / (b - a) if a <= x <= b else 0,
            cdf=lambda x: 0 if x < a else (1 if x > b else (x - a) / (b - a)),
            mean=(a + b) / 2,
            variance=(b - a)**2 / 12,
            support=(a, b)
        )
        return RandomVariable(dist, discrete=False)

    @staticmethod
    def bernoulli(p: float) -> RandomVariable:
        """
        Bernoulli distribution Ber(p).

        P(X = 1) = p, P(X = 0) = 1 - p
        """
        dist = Distribution(
            name="bernoulli",
            mean=p,
            variance=p * (1 - p),
            support=(0, 1)
        )
        return RandomVariable(dist, discrete=True, values=np.array([0, 1]), probabilities=np.array([1-p, p]))

    @staticmethod
    def binomial(n: int, p: float) -> RandomVariable:
        """
        Binomial distribution Bin(n, p).

        Number of successes in n independent Bernoulli(p) trials.
        """
        values = np.arange(n + 1)
        probabilities = stats.binom.pmf(values, n, p)

        dist = Distribution(
            name="binomial",
            mean=n * p,
            variance=n * p * (1 - p),
            support=(0, n)
        )
        return RandomVariable(dist, discrete=True, values=values, probabilities=probabilities)

    @staticmethod
    def poisson(lam: float) -> RandomVariable:
        """
        Poisson distribution Poi(λ).

        Models count of events in fixed interval.
        P(X = k) = (λ^k / k!) exp(-λ)
        """
        # Truncate at reasonable upper bound
        max_k = int(lam + 10 * np.sqrt(lam))
        values = np.arange(max_k + 1)
        probabilities = stats.poisson.pmf(values, lam)

        dist = Distribution(
            name="poisson",
            mean=lam,
            variance=lam,
            support=(0, np.inf)
        )
        return RandomVariable(dist, discrete=True, values=values, probabilities=probabilities)


# ============================================================================
# LIMIT THEOREMS
# ============================================================================

class LimitTheorems:
    """
    Classical limit theorems of probability.
    """

    @staticmethod
    def verify_weak_law_large_numbers(
        rv: RandomVariable,
        n_samples: int = 1000,
        n_trials: int = 100
    ) -> Tuple[bool, float]:
        """
        Verify Weak Law of Large Numbers empirically.

        (1/n) Σ X_i → E[X] in probability

        Returns (converged, final_mean)
        """
        true_mean = rv.expectation()

        # Run multiple trials
        final_means = []
        for _ in range(n_trials):
            samples = rv.sample(size=n_samples)
            sample_mean = np.mean(samples)
            final_means.append(sample_mean)

        avg_final_mean = np.mean(final_means)
        converged = abs(avg_final_mean - true_mean) < 0.1

        logger.info(f"WLLN: True mean={true_mean:.4f}, Empirical={avg_final_mean:.4f}")

        return converged, avg_final_mean

    @staticmethod
    def verify_central_limit_theorem(
        rv: RandomVariable,
        n_samples: int = 100,
        n_trials: int = 1000
    ) -> Tuple[bool, np.ndarray]:
        """
        Verify Central Limit Theorem empirically.

        √n (X̄ - μ) / σ → N(0, 1)

        Returns (is_normal, normalized_means)
        """
        mu = rv.expectation()
        sigma2 = rv.variance()
        sigma = np.sqrt(sigma2)

        normalized_means = []
        for _ in range(n_trials):
            samples = rv.sample(size=n_samples)
            sample_mean = np.mean(samples)
            normalized = np.sqrt(n_samples) * (sample_mean - mu) / sigma
            normalized_means.append(normalized)

        normalized_means = np.array(normalized_means)

        # Test normality using Kolmogorov-Smirnov test
        _, p_value = stats.kstest(normalized_means, 'norm')
        is_normal = p_value > 0.05  # Accept normality at 5% significance

        logger.info(f"CLT: K-S p-value={p_value:.4f}, is_normal={is_normal}")

        return is_normal, normalized_means


# ============================================================================
# STATISTICAL INFERENCE
# ============================================================================

class MaximumLikelihoodEstimation:
    """
    Maximum Likelihood Estimation (MLE).

    Find parameters θ that maximize L(θ) = P(data | θ).
    """

    @staticmethod
    def mle_normal(data: np.ndarray) -> Tuple[float, float]:
        """
        MLE for normal distribution parameters.

        Returns (μ_hat, σ²_hat)
        """
        mu_hat = np.mean(data)
        sigma2_hat = np.var(data, ddof=0)  # MLE uses n, not n-1

        return mu_hat, sigma2_hat

    @staticmethod
    def mle_exponential(data: np.ndarray) -> float:
        """
        MLE for exponential distribution rate parameter.

        Returns λ_hat = 1 / mean
        """
        return 1.0 / np.mean(data)

    @staticmethod
    def mle_bernoulli(data: np.ndarray) -> float:
        """
        MLE for Bernoulli parameter p.

        Returns p_hat = (# of 1s) / n
        """
        return np.mean(data)


class BayesianInference:
    """
    Bayesian inference: posterior ∝ likelihood × prior.
    """

    @staticmethod
    def beta_binomial_posterior(
        successes: int,
        trials: int,
        prior_alpha: float = 1,
        prior_beta: float = 1
    ) -> Tuple[float, float, Callable[[float], float]]:
        """
        Beta-Binomial conjugate pair.

        Prior: p ~ Beta(α, β)
        Likelihood: k | p ~ Binomial(n, p)
        Posterior: p | k ~ Beta(α + k, β + n - k)

        Returns (posterior_mean, posterior_variance, posterior_pdf)
        """
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + (trials - successes)

        mean = posterior_alpha / (posterior_alpha + posterior_beta)
        total = posterior_alpha + posterior_beta
        variance = (posterior_alpha * posterior_beta) / (total**2 * (total + 1))

        posterior_pdf = lambda p: stats.beta.pdf(p, posterior_alpha, posterior_beta)

        logger.info(f"Posterior: Beta({posterior_alpha}, {posterior_beta})")
        logger.info(f"Posterior mean: {mean:.4f}, variance: {variance:.6f}")

        return mean, variance, posterior_pdf

    @staticmethod
    def normal_normal_posterior(
        data: np.ndarray,
        prior_mu: float,
        prior_sigma2: float,
        likelihood_sigma2: float
    ) -> Tuple[float, float]:
        """
        Normal-Normal conjugate pair (known variance).

        Prior: μ ~ N(μ₀, σ₀²)
        Likelihood: X | μ ~ N(μ, σ²)
        Posterior: μ | X ~ N(μ_n, σ_n²)

        Returns (posterior_mu, posterior_sigma2)
        """
        n = len(data)
        x_bar = np.mean(data)

        # Posterior precision = prior precision + n × likelihood precision
        precision_prior = 1 / prior_sigma2
        precision_likelihood = 1 / likelihood_sigma2

        posterior_precision = precision_prior + n * precision_likelihood
        posterior_sigma2 = 1 / posterior_precision

        # Posterior mean is precision-weighted average
        posterior_mu = posterior_sigma2 * (
            precision_prior * prior_mu + n * precision_likelihood * x_bar
        )

        logger.info(f"Posterior: N({posterior_mu:.4f}, {posterior_sigma2:.6f})")

        return posterior_mu, posterior_sigma2


class HypothesisTesting:
    """
    Classical hypothesis testing.
    """

    @staticmethod
    def t_test_one_sample(
        data: np.ndarray,
        null_mean: float,
        alpha: float = 0.05
    ) -> Tuple[bool, float, float]:
        """
        One-sample t-test.

        H₀: μ = μ₀
        H₁: μ ≠ μ₀

        Returns (reject_null, t_statistic, p_value)
        """
        t_stat, p_value = stats.ttest_1samp(data, null_mean)
        reject = p_value < alpha

        logger.info(f"t-test: t={t_stat:.4f}, p={p_value:.4f}, reject={reject}")

        return reject, t_stat, p_value

    @staticmethod
    def t_test_two_sample(
        data1: np.ndarray,
        data2: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[bool, float, float]:
        """
        Two-sample t-test (independent samples).

        H₀: μ₁ = μ₂
        H₁: μ₁ ≠ μ₂

        Returns (reject_null, t_statistic, p_value)
        """
        t_stat, p_value = stats.ttest_ind(data1, data2)
        reject = p_value < alpha

        logger.info(f"Two-sample t-test: t={t_stat:.4f}, p={p_value:.4f}, reject={reject}")

        return reject, t_stat, p_value

    @staticmethod
    def chi_square_goodness_of_fit(
        observed: np.ndarray,
        expected: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[bool, float, float]:
        """
        Chi-square goodness of fit test.

        H₀: Observed follows expected distribution
        H₁: Does not follow expected distribution

        Returns (reject_null, chi2_statistic, p_value)
        """
        chi2_stat, p_value = stats.chisquare(observed, expected)
        reject = p_value < alpha

        logger.info(f"Chi-square: χ²={chi2_stat:.4f}, p={p_value:.4f}, reject={reject}")

        return reject, chi2_stat, p_value


# ============================================================================
# MARKOV CHAINS
# ============================================================================

class MarkovChain:
    """
    Discrete-time Markov chain.

    State space S, transition matrix P where P[i,j] = P(X_{n+1}=j | X_n=i)
    """

    def __init__(self, transition_matrix: np.ndarray, states: Optional[List] = None):
        """
        Initialize Markov chain.

        Args:
            transition_matrix: Transition probability matrix P
            states: State labels (optional)
        """
        self.P = transition_matrix
        n = transition_matrix.shape[0]

        # Verify P is stochastic (rows sum to 1)
        row_sums = np.sum(self.P, axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Transition matrix rows must sum to 1")

        if states is None:
            states = list(range(n))
        self.states = states

        logger.info(f"Markov chain with {n} states")

    def simulate(self, initial_state: int, n_steps: int) -> List[int]:
        """
        Simulate Markov chain for n_steps.

        Returns list of states visited.
        """
        path = [initial_state]
        current = initial_state

        for _ in range(n_steps):
            # Sample next state according to P[current, :]
            next_state = np.random.choice(len(self.states), p=self.P[current, :])
            path.append(next_state)
            current = next_state

        return path

    def stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution π.

        Solve πP = π, Σ π_i = 1
        """
        n = len(self.states)

        # Solve (P^T - I) π = 0 with constraint Σ π_i = 1
        # Use eigenvalue method: find eigenvector for eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)

        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])

        # Normalize
        stationary = stationary / np.sum(stationary)

        return stationary

    def is_irreducible(self) -> bool:
        """
        Check if chain is irreducible.

        Irreducible if all states communicate (can reach any state from any other).
        """
        n = len(self.states)

        # Compute P^n for large n
        P_power = np.linalg.matrix_power(self.P, n * 2)

        # Check if all entries are positive
        return np.all(P_power > 0)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ProbabilitySpace',
    'RandomVariable',
    'Distribution',
    'CommonDistributions',
    'LimitTheorems',
    'MaximumLikelihoodEstimation',
    'BayesianInference',
    'HypothesisTesting',
    'MarkovChain'
]
