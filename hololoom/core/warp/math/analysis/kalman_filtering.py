"""
Kalman Filtering — Optimal State Estimation Under Noise
=======================================================

Linear and nonlinear state estimation for dynamical systems observed through
noisy measurements.

Core Concepts:
- Kalman Filter: Optimal linear estimator (minimizes MSE)
- Extended Kalman Filter: Linearization via Jacobians for nonlinear systems
- Unscented Kalman Filter: Sigma-point propagation (no Jacobians needed)
- Rauch-Tung-Striebel Smoother: Backward pass for offline estimation
- Discrete Algebraic Riccati Equation: Steady-state gain

Mathematical Foundation:
Predict:  x̂⁻ = F x̂,  P⁻ = F P Fᵀ + Q
Update:   K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹
          x̂ = x̂⁻ + K (z - H x̂⁻)
          P = (I - K H) P⁻

Innovation:  ν = z - H x̂⁻
Innovation covariance:  S = H P⁻ Hᵀ + R
Log-likelihood:  -½ [n ln(2π) + ln|S| + νᵀ S⁻¹ ν]

Applications to Warp Space:
- Tracking latent states in knowledge graphs
- Smoothing noisy embedding trajectories
- Online estimation of model confidence
- Sensor fusion for multi-modal inputs

Author: HoloLoom Team
Date: 2025-10-26
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import linalg as scipy_linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.debug("scipy not available; DARE solver will use iterative fallback")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class KalmanState:
    """
    State of a Kalman filter at a single time step.

    Attributes:
        mean: State estimate vector (n,)
        covariance: Error covariance matrix (n, n)
        log_likelihood: Log-likelihood of the observation that produced this state.
                        Cumulative when summed over a sequence.
    """
    mean: np.ndarray
    covariance: np.ndarray
    log_likelihood: float = 0.0

    @property
    def dim(self) -> int:
        """State dimension."""
        return self.mean.shape[0]

    def mahalanobis(self, x: np.ndarray) -> float:
        """Mahalanobis distance of x from the state estimate."""
        diff = x - self.mean
        P_inv = np.linalg.inv(self.covariance)
        return float(np.sqrt(diff @ P_inv @ diff))

    def confidence_ellipse(self, n_std: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Eigenvalues and eigenvectors of the covariance scaled by n_std.
        Useful for 2D visualization of uncertainty.

        Returns:
            (eigenvalues scaled by n_std, eigenvectors)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        return n_std * np.sqrt(np.maximum(eigenvalues, 0.0)), eigenvectors


# ============================================================================
# LINEAR KALMAN FILTER
# ============================================================================

class KalmanFilter:
    """
    Standard (linear) Kalman filter.

    Implements the predict-update cycle for linear Gaussian state-space models:
        x_{k+1} = F x_k + w_k,   w_k ~ N(0, Q)
        z_k     = H x_k + v_k,   v_k ~ N(0, R)

    All methods are static — no mutable state. Thread-safe.
    """

    @staticmethod
    def predict(state: KalmanState, F: np.ndarray, Q: np.ndarray,
                B: np.ndarray | None = None,
                u: np.ndarray | None = None) -> KalmanState:
        """
        Predict step: propagate state through dynamics.

        Args:
            state: Current state estimate
            F: State transition matrix (n, n)
            Q: Process noise covariance (n, n)
            B: Optional control input matrix (n, m)
            u: Optional control vector (m,)

        Returns:
            Predicted state (prior)
        """
        mean_pred = F @ state.mean
        if B is not None and u is not None:
            mean_pred = mean_pred + B @ u

        cov_pred = F @ state.covariance @ F.T + Q
        # Enforce symmetry (numerical drift)
        cov_pred = 0.5 * (cov_pred + cov_pred.T)

        return KalmanState(
            mean=mean_pred,
            covariance=cov_pred,
            log_likelihood=state.log_likelihood
        )

    @staticmethod
    def update(state: KalmanState, z: np.ndarray,
               H: np.ndarray, R: np.ndarray) -> KalmanState:
        """
        Update step: incorporate an observation.

        Args:
            state: Predicted (prior) state
            z: Observation vector (m,)
            H: Observation model matrix (m, n)
            R: Measurement noise covariance (m, m)

        Returns:
            Updated (posterior) state with log-likelihood contribution
        """
        # Innovation
        innovation = z - H @ state.mean
        # Innovation covariance
        S = H @ state.covariance @ H.T + R

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance; using pseudoinverse")
            S_inv = np.linalg.pinv(S)

        K = state.covariance @ H.T @ S_inv

        # Updated state
        mean_upd = state.mean + K @ innovation

        # Joseph form for numerical stability: P = (I - KH)P(I - KH)' + KRK'
        n = state.dim
        I_KH = np.eye(n) - K @ H
        cov_upd = I_KH @ state.covariance @ I_KH.T + K @ R @ K.T
        cov_upd = 0.5 * (cov_upd + cov_upd.T)

        # Log-likelihood contribution
        m = z.shape[0]
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            ll_contrib = -1e10  # degenerate
        else:
            ll_contrib = -0.5 * (m * np.log(2 * np.pi) + logdet +
                                 float(innovation @ S_inv @ innovation))

        return KalmanState(
            mean=mean_upd,
            covariance=cov_upd,
            log_likelihood=state.log_likelihood + ll_contrib
        )

    @staticmethod
    def smooth(states_predicted: list[KalmanState],
               states_filtered: list[KalmanState],
               F: np.ndarray) -> list[KalmanState]:
        """
        Rauch-Tung-Striebel smoother — backward pass over filtered states.

        Uses both the predicted (prior) and filtered (posterior) states from
        the forward pass to compute smoothed estimates that incorporate all
        observations.

        Args:
            states_predicted: Predicted states from forward pass (length T)
            states_filtered: Filtered states from forward pass (length T)
            F: State transition matrix (assumed constant)

        Returns:
            Smoothed states (length T)
        """
        T = len(states_filtered)
        if T == 0:
            return []
        if T == 1:
            return [KalmanState(
                mean=states_filtered[0].mean.copy(),
                covariance=states_filtered[0].covariance.copy(),
                log_likelihood=states_filtered[0].log_likelihood
            )]

        smoothed = [None] * T
        smoothed[T - 1] = KalmanState(
            mean=states_filtered[T - 1].mean.copy(),
            covariance=states_filtered[T - 1].covariance.copy(),
            log_likelihood=states_filtered[T - 1].log_likelihood
        )

        for k in range(T - 2, -1, -1):
            P_filt = states_filtered[k].covariance
            P_pred = states_predicted[k + 1].covariance

            try:
                P_pred_inv = np.linalg.inv(P_pred)
            except np.linalg.LinAlgError:
                P_pred_inv = np.linalg.pinv(P_pred)

            # Smoother gain
            G = P_filt @ F.T @ P_pred_inv

            mean_smooth = (states_filtered[k].mean +
                           G @ (smoothed[k + 1].mean - states_predicted[k + 1].mean))

            cov_smooth = (P_filt +
                          G @ (smoothed[k + 1].covariance - P_pred) @ G.T)
            cov_smooth = 0.5 * (cov_smooth + cov_smooth.T)

            smoothed[k] = KalmanState(
                mean=mean_smooth,
                covariance=cov_smooth,
                log_likelihood=states_filtered[k].log_likelihood
            )

        return smoothed

    @staticmethod
    def filter_sequence(observations: list[np.ndarray],
                        F: np.ndarray, H: np.ndarray,
                        Q: np.ndarray, R: np.ndarray,
                        initial: KalmanState,
                        B: np.ndarray | None = None,
                        controls: list[np.ndarray] | None = None
                        ) -> tuple[list[KalmanState], list[KalmanState]]:
        """
        Run the full forward filter over a sequence of observations.

        Args:
            observations: List of observation vectors
            F: State transition matrix
            H: Observation model matrix
            Q: Process noise covariance
            R: Measurement noise covariance
            initial: Initial state estimate
            B: Optional control input matrix
            controls: Optional list of control vectors (same length as observations)

        Returns:
            (filtered_states, predicted_states) — both length T
        """
        predicted = []
        filtered = []
        state = initial

        for k, z in enumerate(observations):
            u = controls[k] if controls is not None else None
            pred = KalmanFilter.predict(state, F, Q, B, u)
            predicted.append(pred)

            filt = KalmanFilter.update(pred, z, H, R)
            filtered.append(filt)
            state = filt

        return filtered, predicted

    @staticmethod
    def innovation_sequence(observations: list[np.ndarray],
                            states_predicted: list[KalmanState],
                            H: np.ndarray, R: np.ndarray
                            ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Compute the innovation sequence and covariances.

        Useful for model validation — innovations should be white noise
        under a correctly specified model.

        Returns:
            List of (innovation_vector, innovation_covariance) tuples
        """
        innovations = []
        for z, pred in zip(observations, states_predicted):
            nu = z - H @ pred.mean
            S = H @ pred.covariance @ H.T + R
            innovations.append((nu, S))
        return innovations


# ============================================================================
# EXTENDED KALMAN FILTER
# ============================================================================

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state-space models.

    Linearizes the dynamics and observation functions via Jacobians:
        x_{k+1} = f(x_k) + w_k
        z_k     = h(x_k) + v_k

    Jacobians F = ∂f/∂x and H = ∂h/∂x are evaluated at the current estimate.
    """

    @staticmethod
    def predict(state: KalmanState,
                f: Callable[[np.ndarray], np.ndarray],
                F_jacobian: Callable[[np.ndarray], np.ndarray],
                Q: np.ndarray) -> KalmanState:
        """
        EKF predict step with nonlinear dynamics.

        Args:
            state: Current state estimate
            f: Nonlinear state transition function x -> f(x)
            F_jacobian: Function returning the Jacobian ∂f/∂x at a given state
            Q: Process noise covariance

        Returns:
            Predicted state
        """
        mean_pred = f(state.mean)
        F = F_jacobian(state.mean)
        cov_pred = F @ state.covariance @ F.T + Q
        cov_pred = 0.5 * (cov_pred + cov_pred.T)

        return KalmanState(
            mean=mean_pred,
            covariance=cov_pred,
            log_likelihood=state.log_likelihood
        )

    @staticmethod
    def update(state: KalmanState,
               z: np.ndarray,
               h: Callable[[np.ndarray], np.ndarray],
               H_jacobian: Callable[[np.ndarray], np.ndarray],
               R: np.ndarray) -> KalmanState:
        """
        EKF update step with nonlinear observation.

        Args:
            state: Predicted state
            z: Observation vector
            h: Nonlinear observation function x -> h(x)
            H_jacobian: Function returning ∂h/∂x at a given state
            R: Measurement noise covariance

        Returns:
            Updated state
        """
        z_pred = h(state.mean)
        H = H_jacobian(state.mean)

        innovation = z - z_pred
        S = H @ state.covariance @ H.T + R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = state.covariance @ H.T @ S_inv

        mean_upd = state.mean + K @ innovation

        n = state.dim
        I_KH = np.eye(n) - K @ H
        cov_upd = I_KH @ state.covariance @ I_KH.T + K @ R @ K.T
        cov_upd = 0.5 * (cov_upd + cov_upd.T)

        m = z.shape[0]
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            ll_contrib = -1e10
        else:
            ll_contrib = -0.5 * (m * np.log(2 * np.pi) + logdet +
                                 float(innovation @ S_inv @ innovation))

        return KalmanState(
            mean=mean_upd,
            covariance=cov_upd,
            log_likelihood=state.log_likelihood + ll_contrib
        )

    @staticmethod
    def filter_sequence(observations: list[np.ndarray],
                        f: Callable[[np.ndarray], np.ndarray],
                        h: Callable[[np.ndarray], np.ndarray],
                        F_jacobian: Callable[[np.ndarray], np.ndarray],
                        H_jacobian: Callable[[np.ndarray], np.ndarray],
                        Q: np.ndarray, R: np.ndarray,
                        initial: KalmanState
                        ) -> tuple[list[KalmanState], list[KalmanState]]:
        """
        Run EKF over a sequence.

        Returns:
            (filtered_states, predicted_states)
        """
        predicted = []
        filtered = []
        state = initial

        for z in observations:
            pred = ExtendedKalmanFilter.predict(state, f, F_jacobian, Q)
            predicted.append(pred)

            filt = ExtendedKalmanFilter.update(pred, z, h, H_jacobian, R)
            filtered.append(filt)
            state = filt

        return filtered, predicted


# ============================================================================
# UNSCENTED KALMAN FILTER
# ============================================================================

class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter — sigma-point propagation for nonlinear systems.

    Instead of linearizing, the UKF propagates a carefully chosen set of
    sigma points through the nonlinear functions and reconstructs statistics
    from the transformed points. Captures mean and covariance to 2nd order
    (vs. 1st order for EKF) without requiring Jacobians.

    Sigma point scheme: 2n+1 points for n-dimensional state.
    """

    @staticmethod
    def generate_sigma_points(state: KalmanState,
                              alpha: float = 1e-3,
                              beta: float = 2.0,
                              kappa: float = 0.0
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate sigma points and weights using the scaled unscented transform.

        Args:
            state: Current state (mean and covariance)
            alpha: Spread of sigma points (small positive, e.g. 1e-3)
            beta: Prior distribution parameter (2 is optimal for Gaussian)
            kappa: Secondary scaling (typically 0 or 3-n)

        Returns:
            (sigma_points, weights_mean, weights_cov)
            sigma_points: (2n+1, n) array
            weights_mean: (2n+1,) weights for mean reconstruction
            weights_cov: (2n+1,) weights for covariance reconstruction
        """
        n = state.dim
        lam = alpha ** 2 * (n + kappa) - n

        # Square root of (n + lambda) * P
        scale = n + lam
        try:
            sqrt_P = np.linalg.cholesky(scale * state.covariance)
        except np.linalg.LinAlgError:
            # Covariance not positive definite; regularize
            eigvals = np.linalg.eigvalsh(state.covariance)
            eps = max(1e-10, -2.0 * min(eigvals))
            P_reg = state.covariance + eps * np.eye(n)
            sqrt_P = np.linalg.cholesky(scale * P_reg)

        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = state.mean

        for i in range(n):
            sigma_points[i + 1] = state.mean + sqrt_P[i]
            sigma_points[n + i + 1] = state.mean - sqrt_P[i]

        # Weights
        w_mean = np.zeros(2 * n + 1)
        w_cov = np.zeros(2 * n + 1)

        w_mean[0] = lam / (n + lam)
        w_cov[0] = lam / (n + lam) + (1 - alpha ** 2 + beta)

        w_rest = 1.0 / (2.0 * (n + lam))
        w_mean[1:] = w_rest
        w_cov[1:] = w_rest

        return sigma_points, w_mean, w_cov

    @staticmethod
    def predict(state: KalmanState,
                f: Callable[[np.ndarray], np.ndarray],
                Q: np.ndarray,
                alpha: float = 1e-3,
                beta: float = 2.0,
                kappa: float = 0.0) -> KalmanState:
        """
        UKF predict step.

        Propagates sigma points through the nonlinear dynamics f,
        then reconstructs predicted mean and covariance.

        Args:
            state: Current state
            f: Nonlinear transition function
            Q: Process noise covariance
            alpha, beta, kappa: Sigma point parameters

        Returns:
            Predicted state
        """
        sigma_pts, w_m, w_c = UnscentedKalmanFilter.generate_sigma_points(
            state, alpha, beta, kappa
        )

        n_sigma = sigma_pts.shape[0]

        # Propagate sigma points
        transformed = np.array([f(sigma_pts[i]) for i in range(n_sigma)])

        # Reconstruct mean
        mean_pred = np.sum(w_m[:, np.newaxis] * transformed, axis=0)

        # Reconstruct covariance
        n = mean_pred.shape[0]
        cov_pred = np.zeros((n, n))
        for i in range(n_sigma):
            diff = transformed[i] - mean_pred
            cov_pred += w_c[i] * np.outer(diff, diff)

        cov_pred += Q
        cov_pred = 0.5 * (cov_pred + cov_pred.T)

        return KalmanState(
            mean=mean_pred,
            covariance=cov_pred,
            log_likelihood=state.log_likelihood
        )

    @staticmethod
    def update(state: KalmanState,
               z: np.ndarray,
               h: Callable[[np.ndarray], np.ndarray],
               R: np.ndarray,
               alpha: float = 1e-3,
               beta: float = 2.0,
               kappa: float = 0.0) -> KalmanState:
        """
        UKF update step.

        Propagates sigma points through the observation function h,
        computes cross-covariance, and applies the Kalman update.

        Args:
            state: Predicted state
            z: Observation vector
            h: Nonlinear observation function
            R: Measurement noise covariance
            alpha, beta, kappa: Sigma point parameters

        Returns:
            Updated state
        """
        sigma_pts, w_m, w_c = UnscentedKalmanFilter.generate_sigma_points(
            state, alpha, beta, kappa
        )

        n_sigma = sigma_pts.shape[0]

        # Propagate through observation function
        z_sigma = np.array([h(sigma_pts[i]) for i in range(n_sigma)])

        # Predicted observation mean
        z_pred = np.sum(w_m[:, np.newaxis] * z_sigma, axis=0)

        # Innovation covariance
        m = z.shape[0]
        S = np.zeros((m, m))
        for i in range(n_sigma):
            dz = z_sigma[i] - z_pred
            S += w_c[i] * np.outer(dz, dz)
        S += R

        # Cross-covariance between state and observation
        n = state.dim
        P_xz = np.zeros((n, m))
        for i in range(n_sigma):
            dx = sigma_pts[i] - state.mean
            dz = z_sigma[i] - z_pred
            P_xz += w_c[i] * np.outer(dx, dz)

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = P_xz @ S_inv

        # Update
        innovation = z - z_pred
        mean_upd = state.mean + K @ innovation

        cov_upd = state.covariance - K @ S @ K.T
        cov_upd = 0.5 * (cov_upd + cov_upd.T)

        # Log-likelihood
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            ll_contrib = -1e10
        else:
            ll_contrib = -0.5 * (m * np.log(2 * np.pi) + logdet +
                                 float(innovation @ S_inv @ innovation))

        return KalmanState(
            mean=mean_upd,
            covariance=cov_upd,
            log_likelihood=state.log_likelihood + ll_contrib
        )

    @staticmethod
    def filter_sequence(observations: list[np.ndarray],
                        f: Callable[[np.ndarray], np.ndarray],
                        h: Callable[[np.ndarray], np.ndarray],
                        Q: np.ndarray, R: np.ndarray,
                        initial: KalmanState,
                        alpha: float = 1e-3,
                        beta: float = 2.0,
                        kappa: float = 0.0) -> list[KalmanState]:
        """
        Run full UKF over a sequence.

        Returns:
            List of filtered states
        """
        filtered = []
        state = initial

        for z in observations:
            pred = UnscentedKalmanFilter.predict(state, f, Q, alpha, beta, kappa)
            filt = UnscentedKalmanFilter.update(pred, z, h, R, alpha, beta, kappa)
            filtered.append(filt)
            state = filt

        return filtered


# ============================================================================
# STEADY-STATE GAIN (DARE)
# ============================================================================

def steady_state_gain(F: np.ndarray, H: np.ndarray,
                      Q: np.ndarray, R: np.ndarray,
                      max_iter: int = 1000,
                      tol: float = 1e-12) -> np.ndarray:
    """
    Compute the steady-state Kalman gain by solving the Discrete Algebraic
    Riccati Equation (DARE).

    The DARE finds the fixed-point covariance P∞ satisfying:
        P = F P Fᵀ + Q - F P Hᵀ (H P Hᵀ + R)⁻¹ H P Fᵀ

    Then K∞ = P∞ Hᵀ (H P∞ Hᵀ + R)⁻¹.

    Uses scipy.linalg.solve_discrete_are when available, otherwise iterates
    the Riccati recursion to convergence.

    Args:
        F: State transition matrix (n, n)
        H: Observation matrix (m, n)
        Q: Process noise covariance (n, n)
        R: Measurement noise covariance (m, m)
        max_iter: Maximum iterations for iterative solver
        tol: Convergence tolerance on ||P_{k+1} - P_k||

    Returns:
        Steady-state Kalman gain K∞ (n, m)
    """
    n = F.shape[0]

    if HAS_SCIPY:
        try:
            # scipy solves: Aᵀ X A - X - Aᵀ X B (Bᵀ X B + R)⁻¹ Bᵀ X A + Q = 0
            # Map our problem: A=F', B=H', Q=Q, R=R
            P = scipy_linalg.solve_discrete_are(F.T, H.T, Q, R)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            return K
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"scipy DARE solver failed ({e}); falling back to iteration")

    # Iterative fallback: run the Riccati recursion until convergence
    P = np.eye(n)
    for i in range(max_iter):
        S = H @ P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = P @ H.T @ S_inv
        P_new = F @ (P - K @ H @ P) @ F.T + Q
        P_new = 0.5 * (P_new + P_new.T)

        delta = np.linalg.norm(P_new - P, ord='fro')
        P = P_new
        if delta < tol:
            logger.debug(f"DARE converged in {i + 1} iterations (delta={delta:.2e})")
            break
    else:
        logger.warning(f"DARE did not converge after {max_iter} iterations "
                       f"(delta={delta:.2e})")

    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    return K


# ============================================================================
# UTILITIES
# ============================================================================

def normalized_innovation_squared(innovations: list[tuple[np.ndarray, np.ndarray]]
                                  ) -> np.ndarray:
    """
    Compute the Normalized Innovation Squared (NIS) test statistic.

    NIS = νᵀ S⁻¹ ν

    Under a correctly specified model, NIS follows a chi-squared distribution
    with m degrees of freedom (m = observation dimension). Useful for
    consistency checking.

    Args:
        innovations: List of (innovation, innovation_covariance) from
                     KalmanFilter.innovation_sequence

    Returns:
        Array of NIS values, one per time step
    """
    nis_values = []
    for nu, S in innovations:
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        nis = float(nu @ S_inv @ nu)
        nis_values.append(nis)
    return np.array(nis_values)


def filter_and_smooth(observations: list[np.ndarray],
                      F: np.ndarray, H: np.ndarray,
                      Q: np.ndarray, R: np.ndarray,
                      initial: KalmanState
                      ) -> tuple[list[KalmanState], list[KalmanState], list[KalmanState]]:
    """
    Convenience: run forward filter then RTS smoother.

    Returns:
        (filtered, predicted, smoothed) state sequences
    """
    filtered, predicted = KalmanFilter.filter_sequence(
        observations, F, H, Q, R, initial
    )
    smoothed = KalmanFilter.smooth(predicted, filtered, F)
    return filtered, predicted, smoothed
