"""
Reservoir Computing — SA5.1-5.3
================================

Recurrent neural computation where only the readout is trained.
The reservoir (recurrent layer) has fixed random weights chosen
to operate near the edge of chaos (spectral radius ~ 1).

Classes:
    EchoStateNetwork: Random recurrent reservoir with linear readout
    Conceptor: Pattern management via soft Boolean operations on reservoir states

Key Concepts:
    - Echo state property: reservoir state is a function of input history
    - Spectral radius: controls memory vs. nonlinearity tradeoff
    - Readout training: ridge regression (only trainable layer)
    - Conceptors: learned filters that select reservoir dynamics patterns

Applications:
    - Time series prediction with minimal training
    - Pattern recognition in HoloLoom memory streams
    - Compositional pattern management (AND/OR/NOT on learned patterns)

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class EchoStateNetwork:
    """
    Echo State Network (ESN) — reservoir computing architecture.

    Architecture:
        x(t) = tanh(W_in * u(t) + W * x(t-1) + W_fb * y(t-1))
        y(t) = W_out * [x(t); u(t)]  (readout, trained by ridge regression)

    The reservoir W is a sparse random matrix with controlled spectral
    radius. Only W_out is learned; W_in and W are fixed after initialization.

    Attributes:
        n_input: Input dimension
        n_reservoir: Reservoir size (number of recurrent units)
        n_output: Output dimension
        spectral_radius: Largest eigenvalue magnitude of W (controls memory)
        W_in: Input-to-reservoir weights, shape (n_reservoir, n_input)
        W: Reservoir recurrent weights, shape (n_reservoir, n_reservoir)
        W_out: Readout weights, shape (n_output, n_reservoir + n_input)
    """

    def __init__(
        self,
        n_input: int,
        n_reservoir: int,
        n_output: int,
        spectral_radius: float = 0.95,
        input_scaling: float = 1.0,
        sparsity: float = 0.9,
        noise: float = 0.0,
        leak_rate: float = 1.0,
        seed: int | None = None,
    ):
        """
        Initialize ESN with random reservoir.

        Args:
            n_input: Input dimension
            n_reservoir: Number of reservoir units
            n_output: Output dimension
            spectral_radius: Target spectral radius of W (< 1 for ESP)
            input_scaling: Scaling factor for W_in
            sparsity: Fraction of zero entries in W
            noise: Noise added to reservoir state updates
            leak_rate: Leaky integration rate (1 = no leaking)
            seed: Random seed for reproducibility
        """
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        self.spectral_radius = spectral_radius
        self.noise = noise
        self.leak_rate = leak_rate

        rng = np.random.RandomState(seed)

        # Input weights
        self.W_in = rng.randn(n_reservoir, n_input) * input_scaling

        # Reservoir weights (sparse random, scaled to spectral radius)
        W = rng.randn(n_reservoir, n_reservoir)
        mask = rng.random((n_reservoir, n_reservoir)) > sparsity
        W *= mask

        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig > 0:
            W *= spectral_radius / max_eig
        self.W = W

        # Readout weights (learned)
        self.W_out = np.zeros((n_output, n_reservoir + n_input))

        # Internal state
        self._state = np.zeros(n_reservoir)

    def reset_state(self):
        """Reset reservoir state to zero."""
        self._state = np.zeros(self.n_reservoir)

    def _update(self, u: np.ndarray) -> np.ndarray:
        """
        Single reservoir update step.

        Args:
            u: Input vector, shape (n_input,)

        Returns:
            Updated reservoir state, shape (n_reservoir,)
        """
        pre_activation = self.W_in @ u + self.W @ self._state
        new_state = np.tanh(pre_activation)

        if self.noise > 0:
            new_state += self.noise * np.random.randn(self.n_reservoir)

        # Leaky integration
        self._state = (1 - self.leak_rate) * self._state + self.leak_rate * new_state
        return self._state.copy()

    def _collect_states(self, X: np.ndarray, washout: int = 0) -> np.ndarray:
        """
        Drive reservoir with input sequence and collect states.

        Args:
            X: Input sequence, shape (n_steps, n_input)
            washout: Number of initial steps to discard (transient)

        Returns:
            Reservoir states, shape (n_steps - washout, n_reservoir + n_input)
        """
        n_steps = X.shape[0]
        states = np.zeros((n_steps, self.n_reservoir + self.n_input))

        self.reset_state()

        for t in range(n_steps):
            x = self._update(X[t])
            states[t] = np.concatenate([x, X[t]])

        return states[washout:]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        washout: int = 100,
        ridge_alpha: float = 1e-6,
    ) -> np.ndarray:
        """
        Train the readout layer via ridge regression.

        Args:
            X: Input sequence, shape (n_steps, n_input)
            y: Target output, shape (n_steps, n_output)
            washout: Initial transient steps to discard
            ridge_alpha: Ridge regression regularization

        Returns:
            Readout weight matrix W_out, shape (n_output, n_reservoir + n_input)
        """
        states = self._collect_states(X, washout)
        targets = y[washout:]

        # Ridge regression: W_out = Y * S^T * (S * S^T + alpha * I)^{-1}
        S = states.T  # (n_features, n_samples)
        Y = targets.T  # (n_output, n_samples)

        reg = ridge_alpha * np.eye(S.shape[0])
        self.W_out = Y @ S.T @ np.linalg.inv(S @ S.T + reg)

        return self.W_out

    def predict(self, X: np.ndarray, washout: int = 0) -> np.ndarray:
        """
        Generate predictions for input sequence.

        Args:
            X: Input sequence, shape (n_steps, n_input)
            washout: Initial steps to discard

        Returns:
            Predictions, shape (n_steps - washout, n_output)
        """
        states = self._collect_states(X, washout)
        return (self.W_out @ states.T).T

    def generate(self, n_steps: int, seed_input: np.ndarray | None = None) -> np.ndarray:
        """
        Free-run generation (autonomous mode).

        Feeds predictions back as input for generative mode.

        Args:
            n_steps: Number of steps to generate
            seed_input: Optional input to initialize the reservoir

        Returns:
            Generated sequence, shape (n_steps, n_output)
        """
        if seed_input is not None:
            self._collect_states(seed_input)

        output = np.zeros((n_steps, self.n_output))

        for t in range(n_steps):
            # Use previous output as input (or zero for first step)
            if t == 0 and seed_input is not None:
                u = seed_input[-1][:self.n_input] if seed_input.shape[1] >= self.n_input else np.zeros(self.n_input)
            elif t > 0:
                u = output[t - 1][:self.n_input] if self.n_output >= self.n_input else np.zeros(self.n_input)
                u_pad = np.zeros(self.n_input)
                u_pad[:min(self.n_output, self.n_input)] = output[t - 1][:min(self.n_output, self.n_input)]
                u = u_pad
            else:
                u = np.zeros(self.n_input)

            x = self._update(u)
            state = np.concatenate([x, u])
            output[t] = self.W_out @ state

        return output

    def memory_capacity(self, n_steps: int = 1000, max_delay: int = 50) -> float:
        """
        Measure the reservoir's short-term memory capacity.

        MC = sum_{k=1}^{K} r^2(y_k, u_{t-k}) where r is correlation.
        Theoretical maximum equals reservoir size.

        Args:
            n_steps: Length of test sequence
            max_delay: Maximum delay to test

        Returns:
            Total memory capacity (sum of squared correlations)
        """
        # Random input sequence
        u = np.random.randn(n_steps, self.n_input)

        # Collect states
        self.reset_state()
        states = np.zeros((n_steps, self.n_reservoir))
        for t in range(n_steps):
            states[t] = self._update(u[t])

        # For each delay k, train readout to predict u(t-k)
        mc = 0.0
        washout = max_delay + 100

        for k in range(1, max_delay + 1):
            S = states[washout:].T
            target = u[washout - k:-k, 0] if k > 0 else u[washout:, 0]

            if len(target) != S.shape[1]:
                target = target[:S.shape[1]]

            # Ridge regression
            reg = 1e-6 * np.eye(S.shape[0])
            w = target @ S.T @ np.linalg.inv(S @ S.T + reg)
            prediction = w @ S

            # Squared correlation
            if np.std(prediction) > 1e-10 and np.std(target) > 1e-10:
                corr = np.corrcoef(prediction, target)[0, 1]
                mc += corr ** 2

        return mc


class Conceptor:
    """
    Conceptor matrices for reservoir pattern management.

    A conceptor C is a soft projection matrix that captures the
    "ellipsoid of typical states" when the reservoir is driven by
    a particular input pattern. It acts as a learned filter.

    Given reservoir states R from pattern p:
        C_p = R R^T (R R^T + alpha^{-2} I)^{-1}

    where alpha (aperture) controls the specificity of the filter.

    Boolean operations on conceptors enable compositional pattern management:
        AND(C1, C2): states common to both patterns
        OR(C1, C2): states in either pattern
        NOT(C): states NOT in the pattern
    """

    def __init__(self, n_reservoir: int, aperture: float = 10.0):
        """
        Initialize conceptor system.

        Args:
            n_reservoir: Reservoir dimension
            aperture: Aperture parameter alpha (higher = more inclusive)
        """
        self.n_reservoir = n_reservoir
        self.aperture = aperture
        self.matrix: np.ndarray | None = None

    def learn(self, reservoir_states: np.ndarray) -> np.ndarray:
        """
        Learn a conceptor from reservoir state samples.

        C = R R^T (R R^T + alpha^{-2} I)^{-1}

        Args:
            reservoir_states: States collected while driving with pattern,
                              shape (n_samples, n_reservoir)

        Returns:
            Conceptor matrix, shape (n_reservoir, n_reservoir)
        """
        R = reservoir_states.T  # (n_reservoir, n_samples)
        n = reservoir_states.shape[0]

        # State correlation matrix
        corr = (R @ R.T) / n

        # Conceptor via regularized inverse
        reg = (1.0 / self.aperture ** 2) * np.eye(self.n_reservoir)
        self.matrix = corr @ np.linalg.inv(corr + reg)

        return self.matrix

    @staticmethod
    def AND(C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
        """
        Soft AND of two conceptors.

        C1 AND C2 = (C1^{-1} + C2^{-1} - I)^{-1}

        States must be typical for BOTH patterns.

        Args:
            C1, C2: Conceptor matrices

        Returns:
            AND conceptor matrix
        """
        n = C1.shape[0]
        I = np.eye(n)
        eps = 1e-8 * I

        # Regularized pseudo-inverse
        C1_inv = np.linalg.inv(C1 + eps)
        C2_inv = np.linalg.inv(C2 + eps)

        result = np.linalg.inv(C1_inv + C2_inv - I + eps)

        # Clip eigenvalues to [0, 1]
        eigvals, eigvecs = np.linalg.eigh(result)
        eigvals = np.clip(eigvals, 0, 1)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    @staticmethod
    def OR(C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
        """
        Soft OR of two conceptors.

        C1 OR C2 = I - (I - C1)^{-1} + (I - C2)^{-1} - I)^{-1}
                  = NOT(NOT(C1) AND NOT(C2))

        States typical for EITHER pattern.

        Args:
            C1, C2: Conceptor matrices

        Returns:
            OR conceptor matrix
        """
        not_C1 = Conceptor.NOT(C1)
        not_C2 = Conceptor.NOT(C2)
        return Conceptor.NOT(Conceptor.AND(not_C1, not_C2))

    @staticmethod
    def NOT(C: np.ndarray) -> np.ndarray:
        """
        Soft NOT of a conceptor.

        NOT(C) = I - C

        States NOT typical for the pattern.

        Args:
            C: Conceptor matrix

        Returns:
            NOT conceptor matrix
        """
        return np.eye(C.shape[0]) - C

    @staticmethod
    def similarity(C1: np.ndarray, C2: np.ndarray) -> float:
        """
        Similarity between two conceptors via generalized cosine.

        sim(C1, C2) = ||C1^{1/2} C2^{1/2}||_F^2 / (||C1||_F * ||C2||_F)

        Args:
            C1, C2: Conceptor matrices

        Returns:
            Similarity in [0, 1]
        """
        norm1 = np.linalg.norm(C1, 'fro')
        norm2 = np.linalg.norm(C2, 'fro')
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        # Frobenius inner product
        inner = np.trace(C1.T @ C2)
        return abs(inner) / (norm1 * norm2)

    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply conceptor as a soft filter on reservoir state.

        Args:
            state: Reservoir state, shape (n_reservoir,)

        Returns:
            Filtered state C @ x
        """
        if self.matrix is None:
            raise ValueError("Conceptor not yet learned. Call learn() first.")
        return self.matrix @ state

    @staticmethod
    def quota(C: np.ndarray) -> float:
        """
        Conceptor quota: fraction of reservoir space captured.

        quota(C) = trace(C) / n = mean of singular values.
        Indicates how much of the reservoir the pattern uses.

        Args:
            C: Conceptor matrix

        Returns:
            Quota in [0, 1]
        """
        return np.trace(C) / C.shape[0]

    @staticmethod
    def aperture_adapt(C: np.ndarray, gamma: float) -> np.ndarray:
        """
        Change the aperture of a conceptor by factor gamma.

        phi(C, gamma) = C (C + gamma^{-2}(I - C))^{-1}

        gamma > 1: more inclusive (broader pattern match)
        gamma < 1: more exclusive (narrower pattern match)

        Args:
            C: Conceptor matrix
            gamma: Aperture adaptation factor

        Returns:
            Adapted conceptor matrix
        """
        n = C.shape[0]
        I = np.eye(n)
        eps = 1e-8 * I
        inner = C + (1.0 / gamma ** 2) * (I - C) + eps
        result = C @ np.linalg.inv(inner)

        # Clip eigenvalues
        eigvals, eigvecs = np.linalg.eigh(result)
        eigvals = np.clip(eigvals, 0, 1)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
