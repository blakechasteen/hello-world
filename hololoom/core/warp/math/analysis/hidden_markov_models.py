"""
Hidden Markov Models — Latent State Inference from Noisy Observations
=====================================================================

Discrete and Gaussian HMMs with full inference suite.

Core Concepts:
- Hidden states evolve via Markov transitions (unobserved)
- Observations are emitted from states via emission distributions
- Forward-Backward: compute marginal likelihoods and posteriors
- Viterbi: most likely state sequence via dynamic programming
- Baum-Welch: EM algorithm for parameter learning

Mathematical Foundation:
HMM is defined by λ = (A, B, π):
  A[i,j] = P(z_{t+1}=j | z_t=i)     transition matrix
  B[j,k] = P(x_t=k | z_t=j)          emission matrix (discrete)
  π[i]   = P(z_1=i)                   initial state distribution

Forward: α_t(j) = P(x_1...x_t, z_t=j | λ)
Backward: β_t(i) = P(x_{t+1}...x_T | z_t=i, λ)
Posterior: γ_t(i) = P(z_t=i | X, λ) = α_t(i)β_t(i) / P(X|λ)

Builds on MarkovChain from probability_theory.py.

Applications to Warp Space:
- Latent topic tracking in conversational memory
- Regime detection in scoring term dynamics
- Anomaly detection via likelihood monitoring
- Skill proficiency modeling in math curriculum

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class HMMParams:
    """Parameters of a discrete Hidden Markov Model.

    Attributes:
        transition: State transition matrix A, shape (N, N).
                    A[i,j] = P(z_{t+1}=j | z_t=i). Rows sum to 1.
        emission: Emission probability matrix B, shape (N, M).
                  B[j,k] = P(x_t=k | z_t=j). Rows sum to 1.
        initial: Initial state distribution π, shape (N,). Sums to 1.
        n_states: Number of hidden states N.
        n_obs: Number of distinct observation symbols M.
    """
    transition: np.ndarray
    emission: np.ndarray
    initial: np.ndarray
    n_states: int
    n_obs: int

    def __post_init__(self) -> None:
        self.transition = np.asarray(self.transition, dtype=np.float64)
        self.emission = np.asarray(self.emission, dtype=np.float64)
        self.initial = np.asarray(self.initial, dtype=np.float64)

    def validate(self) -> None:
        """Check that all probability constraints hold."""
        assert self.transition.shape == (self.n_states, self.n_states), \
            f"Transition shape {self.transition.shape} != ({self.n_states}, {self.n_states})"
        assert self.emission.shape == (self.n_states, self.n_obs), \
            f"Emission shape {self.emission.shape} != ({self.n_states}, {self.n_obs})"
        assert self.initial.shape == (self.n_states,), \
            f"Initial shape {self.initial.shape} != ({self.n_states},)"
        assert np.allclose(self.transition.sum(axis=1), 1.0, atol=1e-6), \
            "Transition rows must sum to 1"
        assert np.allclose(self.emission.sum(axis=1), 1.0, atol=1e-6), \
            "Emission rows must sum to 1"
        assert np.abs(self.initial.sum() - 1.0) < 1e-6, \
            "Initial distribution must sum to 1"


@dataclass
class GaussianHMMParams:
    """Parameters of a Gaussian Hidden Markov Model.

    Emission for state k: x_t | z_t=k ~ N(means[k], covariances[k]).

    Attributes:
        transition: State transition matrix A, shape (N, N).
        means: Emission means, shape (N,) for 1D or (N, D) for multivariate.
        covariances: Emission variances (1D) or covariance matrices.
                     Shape (N,) for 1D, (N, D, D) for multivariate.
        initial: Initial state distribution π, shape (N,).
        n_states: Number of hidden states.
    """
    transition: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    initial: np.ndarray
    n_states: int

    def __post_init__(self) -> None:
        self.transition = np.asarray(self.transition, dtype=np.float64)
        self.means = np.asarray(self.means, dtype=np.float64)
        self.covariances = np.asarray(self.covariances, dtype=np.float64)
        self.initial = np.asarray(self.initial, dtype=np.float64)


@dataclass
class HMMResult:
    """Result of HMM inference.

    Attributes:
        log_likelihood: Log probability of the observation sequence log P(X|λ).
        state_sequence: Most likely state sequence from Viterbi, shape (T,).
        posterior: Posterior state probabilities γ_t(i), shape (T, N).
    """
    log_likelihood: float
    state_sequence: np.ndarray
    posterior: np.ndarray


# ============================================================================
# DISCRETE HMM
# ============================================================================

class HiddenMarkovModel:
    """
    Hidden Markov Model with discrete emissions.

    Full inference suite: forward, backward, Viterbi, Baum-Welch, posterior
    decoding. All computations use log-space scaling for numerical stability.
    """

    @staticmethod
    def forward(
        obs: np.ndarray,
        params: HMMParams
    ) -> tuple[np.ndarray, float]:
        """
        Forward algorithm — compute α_t(j) and log-likelihood.

        α_t(j) = P(x_1,...,x_t, z_t=j | λ)

        Uses log-space scaling to prevent underflow on long sequences.
        At each step t, α_t is normalized by c_t = Σ_j α_t(j), and
        log P(X|λ) = Σ_t log(c_t).

        Args:
            obs: Observation sequence, shape (T,), integer-valued in [0, M).
            params: HMM parameters.

        Returns:
            (alpha, log_likelihood) where alpha has shape (T, N) and contains
            the scaled forward variables.
        """
        obs = np.asarray(obs, dtype=np.int64)
        T = len(obs)
        N = params.n_states
        A = params.transition
        B = params.emission
        pi = params.initial

        alpha = np.zeros((T, N), dtype=np.float64)
        scale = np.zeros(T, dtype=np.float64)

        # Initialization: α_1(j) = π_j * B_j(x_1)
        alpha[0] = pi * B[:, obs[0]]
        scale[0] = alpha[0].sum()
        if scale[0] > 0:
            alpha[0] /= scale[0]
        else:
            # All-zero emission — degenerate case
            alpha[0] = 1.0 / N
            scale[0] = 1e-300

        # Induction: α_t(j) = [Σ_i α_{t-1}(i) * A_{ij}] * B_j(x_t)
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.dot(alpha[t - 1], A[:, j]) * B[j, obs[t]]
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]
            else:
                alpha[t] = 1.0 / N
                scale[t] = 1e-300

        log_likelihood = np.sum(np.log(scale))

        return alpha, log_likelihood

    @staticmethod
    def backward(
        obs: np.ndarray,
        params: HMMParams,
        scale: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Backward algorithm — compute β_t(i).

        β_t(i) = P(x_{t+1},...,x_T | z_t=i, λ)

        Uses the same scaling factors as forward for consistency.

        Args:
            obs: Observation sequence, shape (T,).
            params: HMM parameters.
            scale: Scaling factors from forward pass. If None, computes them.

        Returns:
            beta: Scaled backward variables, shape (T, N).
        """
        obs = np.asarray(obs, dtype=np.int64)
        T = len(obs)
        N = params.n_states
        A = params.transition
        B = params.emission

        # Compute scale factors if not provided
        if scale is None:
            alpha = np.zeros((T, N), dtype=np.float64)
            scale = np.zeros(T, dtype=np.float64)
            alpha[0] = params.initial * B[:, obs[0]]
            scale[0] = max(alpha[0].sum(), 1e-300)
            alpha[0] /= scale[0]
            for t in range(1, T):
                for j in range(N):
                    alpha[t, j] = np.dot(alpha[t - 1], A[:, j]) * B[j, obs[t]]
                scale[t] = max(alpha[t].sum(), 1e-300)
                alpha[t] /= scale[t]

        beta = np.zeros((T, N), dtype=np.float64)

        # Initialization: β_T(i) = 1
        beta[T - 1] = 1.0

        # Induction: β_t(i) = Σ_j A_{ij} * B_j(x_{t+1}) * β_{t+1}(j)
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t, i] = np.dot(A[i, :] * B[:, obs[t + 1]], beta[t + 1])
            if scale[t + 1] > 0:
                beta[t] /= scale[t + 1]

        return beta

    @staticmethod
    def viterbi(
        obs: np.ndarray,
        params: HMMParams
    ) -> np.ndarray:
        """
        Viterbi algorithm — most likely state sequence.

        Finds z* = argmax_{z_1...z_T} P(z_1...z_T | x_1...x_T, λ)
        via dynamic programming in log space.

        Args:
            obs: Observation sequence, shape (T,).
            params: HMM parameters.

        Returns:
            Optimal state sequence, shape (T,), integer-valued.
        """
        obs = np.asarray(obs, dtype=np.int64)
        T = len(obs)
        N = params.n_states
        A = params.transition
        B = params.emission
        pi = params.initial

        # Work in log space to prevent underflow
        log_A = np.log(np.maximum(A, 1e-300))
        log_B = np.log(np.maximum(B, 1e-300))
        log_pi = np.log(np.maximum(pi, 1e-300))

        # δ_t(j) = max log-probability of best path ending in state j at time t
        delta = np.zeros((T, N), dtype=np.float64)
        psi = np.zeros((T, N), dtype=np.int64)

        # Initialization
        delta[0] = log_pi + log_B[:, obs[0]]
        psi[0] = 0

        # Recursion
        for t in range(1, T):
            for j in range(N):
                candidates = delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_B[j, obs[t]]

        # Backtracking
        states = np.zeros(T, dtype=np.int64)
        states[T - 1] = np.argmax(delta[T - 1])

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    @staticmethod
    def posterior_decode(
        obs: np.ndarray,
        params: HMMParams
    ) -> np.ndarray:
        """
        Posterior decoding — compute γ_t(i) = P(z_t=i | X, λ).

        Unlike Viterbi (globally optimal path), posterior decoding picks
        the most probable state at each time independently.

        Args:
            obs: Observation sequence, shape (T,).
            params: HMM parameters.

        Returns:
            Posterior probabilities γ, shape (T, N).
        """
        obs = np.asarray(obs, dtype=np.int64)
        T = len(obs)
        N = params.n_states

        alpha, _ = HiddenMarkovModel.forward(obs, params)

        # Compute scale factors for backward pass
        A = params.transition
        B = params.emission
        scale = np.zeros(T, dtype=np.float64)
        alpha_unscaled = np.zeros((T, N), dtype=np.float64)
        alpha_unscaled[0] = params.initial * B[:, obs[0]]
        scale[0] = max(alpha_unscaled[0].sum(), 1e-300)
        alpha_unscaled[0] /= scale[0]
        for t in range(1, T):
            for j in range(N):
                alpha_unscaled[t, j] = np.dot(alpha_unscaled[t - 1], A[:, j]) * B[j, obs[t]]
            scale[t] = max(alpha_unscaled[t].sum(), 1e-300)
            alpha_unscaled[t] /= scale[t]

        beta = HiddenMarkovModel.backward(obs, params, scale)

        # γ_t(i) = α_t(i) * β_t(i) / Σ_j α_t(j) * β_t(j)
        gamma = alpha * beta
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)
        gamma /= row_sums

        return gamma

    @staticmethod
    def baum_welch(
        obs: np.ndarray,
        n_states: int,
        n_obs: int | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: int | None = None
    ) -> HMMParams:
        """
        Baum-Welch (EM) algorithm — learn HMM parameters from observations.

        Iterates:
          E-step: compute expected sufficient statistics via forward-backward
          M-step: re-estimate A, B, π from expected counts

        Guaranteed to increase (or maintain) log-likelihood at each iteration.
        Converges to a local maximum.

        Args:
            obs: Observation sequence, shape (T,), integer-valued.
            n_states: Number of hidden states N.
            n_obs: Number of distinct observation symbols M.
                   If None, inferred as max(obs) + 1.
            max_iter: Maximum EM iterations.
            tol: Convergence tolerance on log-likelihood change.
            seed: Random seed for initialization.

        Returns:
            Learned HMMParams.
        """
        obs = np.asarray(obs, dtype=np.int64)
        T = len(obs)
        if n_obs is None:
            n_obs = int(obs.max()) + 1
        N = n_states
        M = n_obs

        rng = np.random.RandomState(seed)

        # Random initialization (Dirichlet-like)
        A = rng.dirichlet(np.ones(N), size=N)
        B = rng.dirichlet(np.ones(M), size=N)
        pi = rng.dirichlet(np.ones(N))

        params = HMMParams(
            transition=A, emission=B, initial=pi,
            n_states=N, n_obs=M
        )

        prev_ll = -np.inf

        for iteration in range(max_iter):
            # E-step: forward-backward
            alpha, log_lik = HiddenMarkovModel.forward(obs, params)

            # Compute scale for backward
            scale = np.zeros(T, dtype=np.float64)
            a_tmp = np.zeros((T, N), dtype=np.float64)
            a_tmp[0] = params.initial * params.emission[:, obs[0]]
            scale[0] = max(a_tmp[0].sum(), 1e-300)
            a_tmp[0] /= scale[0]
            for t in range(1, T):
                for j in range(N):
                    a_tmp[t, j] = np.dot(a_tmp[t - 1], params.transition[:, j]) * params.emission[j, obs[t]]
                scale[t] = max(a_tmp[t].sum(), 1e-300)
                a_tmp[t] /= scale[t]

            beta = HiddenMarkovModel.backward(obs, params, scale)

            # Compute gamma: γ_t(i) = P(z_t=i | X, λ)
            gamma = a_tmp * beta
            gamma_sums = gamma.sum(axis=1, keepdims=True)
            gamma_sums = np.maximum(gamma_sums, 1e-300)
            gamma /= gamma_sums

            # Compute xi: ξ_t(i,j) = P(z_t=i, z_{t+1}=j | X, λ)
            xi = np.zeros((T - 1, N, N), dtype=np.float64)
            for t in range(T - 1):
                for i in range(N):
                    for j in range(N):
                        xi[t, i, j] = (
                            a_tmp[t, i]
                            * params.transition[i, j]
                            * params.emission[j, obs[t + 1]]
                            * beta[t + 1, j]
                        )
                xi_sum = xi[t].sum()
                if xi_sum > 0:
                    xi[t] /= xi_sum

            # M-step: re-estimate parameters
            # π_i = γ_1(i)
            pi_new = gamma[0].copy()
            pi_new = np.maximum(pi_new, 1e-300)
            pi_new /= pi_new.sum()

            # A_{ij} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
            A_new = xi.sum(axis=0)  # (N, N)
            A_denom = gamma[:-1].sum(axis=0)  # (N,)
            A_denom = np.maximum(A_denom, 1e-300)
            A_new /= A_denom[:, np.newaxis]
            # Ensure rows sum to 1
            A_row_sums = A_new.sum(axis=1, keepdims=True)
            A_row_sums = np.maximum(A_row_sums, 1e-300)
            A_new /= A_row_sums

            # B_{jk} = Σ_{t: x_t=k} γ_t(j) / Σ_t γ_t(j)
            B_new = np.zeros((N, M), dtype=np.float64)
            gamma_sum_all = gamma.sum(axis=0)  # (N,)
            gamma_sum_all = np.maximum(gamma_sum_all, 1e-300)
            for k in range(M):
                mask = (obs == k)
                B_new[:, k] = gamma[mask].sum(axis=0)
            B_new /= gamma_sum_all[:, np.newaxis]
            # Ensure rows sum to 1
            B_row_sums = B_new.sum(axis=1, keepdims=True)
            B_row_sums = np.maximum(B_row_sums, 1e-300)
            B_new /= B_row_sums

            params = HMMParams(
                transition=A_new, emission=B_new, initial=pi_new,
                n_states=N, n_obs=M
            )

            # Check convergence
            if abs(log_lik - prev_ll) < tol:
                logger.info(f"Baum-Welch converged at iteration {iteration + 1}, "
                            f"log-likelihood={log_lik:.4f}")
                break
            prev_ll = log_lik

        else:
            logger.info(f"Baum-Welch reached max_iter={max_iter}, "
                        f"log-likelihood={log_lik:.4f}")

        return params

    @staticmethod
    def simulate(
        params: HMMParams,
        T: int,
        seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate from an HMM.

        Generate a sequence of hidden states and observations.

        Args:
            params: HMM parameters.
            T: Length of sequence to generate.
            seed: Random seed.

        Returns:
            (states, observations) each of shape (T,), integer-valued.
        """
        rng = np.random.RandomState(seed)
        N = params.n_states
        M = params.n_obs

        states = np.zeros(T, dtype=np.int64)
        observations = np.zeros(T, dtype=np.int64)

        # Sample initial state
        states[0] = rng.choice(N, p=params.initial)
        observations[0] = rng.choice(M, p=params.emission[states[0]])

        for t in range(1, T):
            states[t] = rng.choice(N, p=params.transition[states[t - 1]])
            observations[t] = rng.choice(M, p=params.emission[states[t]])

        return states, observations

    @staticmethod
    def infer(
        obs: np.ndarray,
        params: HMMParams
    ) -> HMMResult:
        """
        Full inference: log-likelihood, Viterbi path, and posterior.

        Convenience method that runs forward, Viterbi, and posterior_decode.

        Args:
            obs: Observation sequence.
            params: HMM parameters.

        Returns:
            HMMResult with all inference outputs.
        """
        _, log_lik = HiddenMarkovModel.forward(obs, params)
        state_seq = HiddenMarkovModel.viterbi(obs, params)
        posterior = HiddenMarkovModel.posterior_decode(obs, params)

        return HMMResult(
            log_likelihood=log_lik,
            state_sequence=state_seq,
            posterior=posterior,
        )


# ============================================================================
# GAUSSIAN HMM
# ============================================================================

class GaussianHMM:
    """
    Hidden Markov Model with Gaussian emissions.

    Each hidden state k emits observations from N(μ_k, σ²_k) (1D)
    or N(μ_k, Σ_k) (multivariate). More flexible than discrete HMMs
    for continuous observation spaces.
    """

    @staticmethod
    def _gaussian_log_pdf(x: float, mean: float, var: float) -> float:
        """Log probability density of univariate Gaussian."""
        return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)

    @staticmethod
    def _gaussian_pdf(x: float, mean: float, var: float) -> float:
        """Probability density of univariate Gaussian."""
        return np.exp(GaussianHMM._gaussian_log_pdf(x, mean, var))

    @staticmethod
    def forward(
        obs: np.ndarray,
        params: GaussianHMMParams
    ) -> tuple[np.ndarray, float]:
        """
        Forward algorithm for Gaussian HMM.

        Args:
            obs: Continuous observations, shape (T,).
            params: Gaussian HMM parameters.

        Returns:
            (alpha, log_likelihood) with alpha shape (T, N).
        """
        obs = np.asarray(obs, dtype=np.float64)
        T = len(obs)
        N = params.n_states
        A = params.transition
        means = params.means
        covs = params.covariances
        pi = params.initial

        alpha = np.zeros((T, N), dtype=np.float64)
        scale = np.zeros(T, dtype=np.float64)

        # Initialization
        for j in range(N):
            alpha[0, j] = pi[j] * GaussianHMM._gaussian_pdf(obs[0], means[j], covs[j])
        scale[0] = max(alpha[0].sum(), 1e-300)
        alpha[0] /= scale[0]

        # Induction
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = (
                    np.dot(alpha[t - 1], A[:, j])
                    * GaussianHMM._gaussian_pdf(obs[t], means[j], covs[j])
                )
            scale[t] = max(alpha[t].sum(), 1e-300)
            alpha[t] /= scale[t]

        log_likelihood = np.sum(np.log(scale))
        return alpha, log_likelihood

    @staticmethod
    def backward(
        obs: np.ndarray,
        params: GaussianHMMParams,
        scale: np.ndarray
    ) -> np.ndarray:
        """
        Backward algorithm for Gaussian HMM.

        Args:
            obs: Continuous observations, shape (T,).
            params: Gaussian HMM parameters.
            scale: Scaling factors from forward pass.

        Returns:
            beta: Scaled backward variables, shape (T, N).
        """
        obs = np.asarray(obs, dtype=np.float64)
        T = len(obs)
        N = params.n_states
        A = params.transition
        means = params.means
        covs = params.covariances

        beta = np.zeros((T, N), dtype=np.float64)
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            for i in range(N):
                for j in range(N):
                    beta[t, i] += (
                        A[i, j]
                        * GaussianHMM._gaussian_pdf(obs[t + 1], means[j], covs[j])
                        * beta[t + 1, j]
                    )
            if scale[t + 1] > 0:
                beta[t] /= scale[t + 1]

        return beta

    @staticmethod
    def viterbi(
        obs: np.ndarray,
        params: GaussianHMMParams
    ) -> np.ndarray:
        """
        Viterbi algorithm for Gaussian HMM.

        Args:
            obs: Continuous observations, shape (T,).
            params: Gaussian HMM parameters.

        Returns:
            Most likely state sequence, shape (T,).
        """
        obs = np.asarray(obs, dtype=np.float64)
        T = len(obs)
        N = params.n_states
        A = params.transition
        means = params.means
        covs = params.covariances
        pi = params.initial

        log_A = np.log(np.maximum(A, 1e-300))
        log_pi = np.log(np.maximum(pi, 1e-300))

        delta = np.zeros((T, N), dtype=np.float64)
        psi = np.zeros((T, N), dtype=np.int64)

        # Initialization
        for j in range(N):
            delta[0, j] = log_pi[j] + GaussianHMM._gaussian_log_pdf(obs[0], means[j], covs[j])

        # Recursion
        for t in range(1, T):
            for j in range(N):
                candidates = delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = (
                    candidates[psi[t, j]]
                    + GaussianHMM._gaussian_log_pdf(obs[t], means[j], covs[j])
                )

        # Backtracking
        states = np.zeros(T, dtype=np.int64)
        states[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    @staticmethod
    def baum_welch_gaussian(
        obs: np.ndarray,
        n_states: int,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: int | None = None
    ) -> GaussianHMMParams:
        """
        Baum-Welch (EM) for Gaussian HMM — learn parameters from data.

        E-step: forward-backward with Gaussian emissions.
        M-step: re-estimate means and variances from posterior-weighted data.

        Args:
            obs: Continuous observations, shape (T,).
            n_states: Number of hidden states.
            max_iter: Maximum EM iterations.
            tol: Convergence tolerance on log-likelihood.
            seed: Random seed for initialization.

        Returns:
            Learned GaussianHMMParams.
        """
        obs = np.asarray(obs, dtype=np.float64)
        T = len(obs)
        N = n_states

        rng = np.random.RandomState(seed)

        # Initialize with k-means-like heuristic
        sorted_obs = np.sort(obs)
        # Split observations into N roughly equal bins for initial means
        bin_size = T // N
        means = np.array([
            sorted_obs[min(i * bin_size + bin_size // 2, T - 1)]
            for i in range(N)
        ])
        # Add jitter
        means += rng.randn(N) * np.std(obs) * 0.1

        covs = np.full(N, np.var(obs) + 1e-6)
        A = rng.dirichlet(np.ones(N), size=N)
        pi = rng.dirichlet(np.ones(N))

        params = GaussianHMMParams(
            transition=A, means=means, covariances=covs,
            initial=pi, n_states=N
        )

        prev_ll = -np.inf

        for iteration in range(max_iter):
            # E-step
            alpha, log_lik = GaussianHMM.forward(obs, params)

            # Reconstruct scale factors
            scale = np.zeros(T, dtype=np.float64)
            a_tmp = np.zeros((T, N), dtype=np.float64)
            for j in range(N):
                a_tmp[0, j] = params.initial[j] * GaussianHMM._gaussian_pdf(
                    obs[0], params.means[j], params.covariances[j]
                )
            scale[0] = max(a_tmp[0].sum(), 1e-300)
            a_tmp[0] /= scale[0]
            for t in range(1, T):
                for j in range(N):
                    a_tmp[t, j] = (
                        np.dot(a_tmp[t - 1], params.transition[:, j])
                        * GaussianHMM._gaussian_pdf(obs[t], params.means[j], params.covariances[j])
                    )
                scale[t] = max(a_tmp[t].sum(), 1e-300)
                a_tmp[t] /= scale[t]

            beta = GaussianHMM.backward(obs, params, scale)

            # Gamma
            gamma = a_tmp * beta
            gamma_sums = gamma.sum(axis=1, keepdims=True)
            gamma_sums = np.maximum(gamma_sums, 1e-300)
            gamma /= gamma_sums

            # Xi
            xi = np.zeros((T - 1, N, N), dtype=np.float64)
            for t in range(T - 1):
                for i in range(N):
                    for j in range(N):
                        xi[t, i, j] = (
                            a_tmp[t, i]
                            * params.transition[i, j]
                            * GaussianHMM._gaussian_pdf(obs[t + 1], params.means[j], params.covariances[j])
                            * beta[t + 1, j]
                        )
                xi_sum = xi[t].sum()
                if xi_sum > 0:
                    xi[t] /= xi_sum

            # M-step
            gamma_sum_all = gamma.sum(axis=0)  # (N,)
            gamma_sum_all = np.maximum(gamma_sum_all, 1e-300)

            # Update initial
            pi_new = gamma[0].copy()
            pi_new = np.maximum(pi_new, 1e-300)
            pi_new /= pi_new.sum()

            # Update transition
            A_new = xi.sum(axis=0)
            A_denom = gamma[:-1].sum(axis=0)
            A_denom = np.maximum(A_denom, 1e-300)
            A_new /= A_denom[:, np.newaxis]
            A_row_sums = A_new.sum(axis=1, keepdims=True)
            A_row_sums = np.maximum(A_row_sums, 1e-300)
            A_new /= A_row_sums

            # Update means: μ_k = Σ_t γ_t(k) x_t / Σ_t γ_t(k)
            means_new = np.zeros(N, dtype=np.float64)
            for k in range(N):
                means_new[k] = np.dot(gamma[:, k], obs) / gamma_sum_all[k]

            # Update variances: σ²_k = Σ_t γ_t(k)(x_t - μ_k)² / Σ_t γ_t(k)
            covs_new = np.zeros(N, dtype=np.float64)
            for k in range(N):
                diff = obs - means_new[k]
                covs_new[k] = np.dot(gamma[:, k], diff ** 2) / gamma_sum_all[k]
                covs_new[k] = max(covs_new[k], 1e-6)  # Floor for numerical stability

            params = GaussianHMMParams(
                transition=A_new, means=means_new, covariances=covs_new,
                initial=pi_new, n_states=N
            )

            if abs(log_lik - prev_ll) < tol:
                logger.info(f"Gaussian Baum-Welch converged at iteration {iteration + 1}, "
                            f"log-likelihood={log_lik:.4f}")
                break
            prev_ll = log_lik
        else:
            logger.info(f"Gaussian Baum-Welch reached max_iter={max_iter}, "
                        f"log-likelihood={log_lik:.4f}")

        return params

    @staticmethod
    def simulate(
        params: GaussianHMMParams,
        T: int,
        seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate from a Gaussian HMM.

        Args:
            params: Gaussian HMM parameters.
            T: Sequence length.
            seed: Random seed.

        Returns:
            (states, observations) where states is int array (T,)
            and observations is float array (T,).
        """
        rng = np.random.RandomState(seed)
        N = params.n_states

        states = np.zeros(T, dtype=np.int64)
        observations = np.zeros(T, dtype=np.float64)

        states[0] = rng.choice(N, p=params.initial)
        observations[0] = rng.normal(params.means[states[0]], np.sqrt(params.covariances[states[0]]))

        for t in range(1, T):
            states[t] = rng.choice(N, p=params.transition[states[t - 1]])
            observations[t] = rng.normal(
                params.means[states[t]],
                np.sqrt(params.covariances[states[t]])
            )

        return states, observations


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'HMMParams',
    'GaussianHMMParams',
    'HMMResult',
    'HiddenMarkovModel',
    'GaussianHMM',
]
