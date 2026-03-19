"""
Second-Order Cybernetics — Observer-Dependent Systems and Self-Reference
========================================================================

Implements Heinz von Foerster's second-order cybernetics: systems where
the observer is part of the system being observed. Key concepts:

    - Observer-dependent observation (filtered perception via learned model)
    - Eigenform computation (fixed points of observation: x* where observe(f(x*)) = x*)
    - Ashby's Law of Requisite Variety (controller must match system complexity)
    - Circular causality (mutual information between observer and system trajectories)

Mathematical Foundation:
    Observation:        o = M @ x  (linear model applied to system state)
    Model update:       M += lr * (actual - observed) @ x^T  (gradient correction)
    Eigenform:          x* = lim_{n->inf} observe(f(observe(f(...(x0)))))
    Variety:            V(S) = log2(|distinguishable states|)
    Circular causality: I(X;Y) via binned mutual information

Pure numpy, no framework dependencies.

Author: Claude Code (with HoloLoom architecture by Blake)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Observer:
    """An observer with a learned perceptual model."""
    id: str
    model: np.ndarray                          # Observation matrix M (obs_dim x sys_dim)
    observations: list[np.ndarray] = field(default_factory=list)


@dataclass
class CyberneticState:
    """Full state of a second-order cybernetic system."""
    system: np.ndarray                         # System state vector
    observers: list[Observer]                  # Participating observers
    requisite_variety: float                   # Ashby measure
    circular_causality: float                  # Mutual information measure


# ============================================================================
# Second-Order Observer
# ============================================================================

class SecondOrderObserver:
    """
    Observer that filters reality through a learned model and updates
    that model from prediction error. The observer changes what it
    observes by the act of observing (model shapes perception).

    The observation model M is a linear map from system space to
    observation space. The observer never sees the system directly —
    only M @ system_state.
    """

    def __init__(self, dim: int, obs_dim: int | None = None, learning_rate: float = 0.1):
        """
        Initialize second-order observer.

        Args:
            dim: System state dimensionality
            obs_dim: Observation dimensionality (defaults to dim)
            learning_rate: Model update step size
        """
        self.dim = dim
        self.obs_dim = obs_dim if obs_dim is not None else dim
        self.learning_rate = learning_rate

        # Initialize model as slightly perturbed identity (or rectangular slice)
        if self.obs_dim == dim:
            self.model = np.eye(dim) + 0.01 * np.random.randn(dim, dim)
        else:
            self.model = np.random.randn(self.obs_dim, dim) * 0.1

        self._observations: list[np.ndarray] = []

    def observe(self, system_state: np.ndarray) -> np.ndarray:
        """
        Apply perceptual model to system state.

        The observer doesn't see reality — it sees M @ reality.
        Each observation is recorded for trajectory analysis.

        Args:
            system_state: True system state vector (dim,)

        Returns:
            Filtered observation (obs_dim,)
        """
        state = np.asarray(system_state, dtype=np.float64)
        observation = self.model @ state
        self._observations.append(observation.copy())
        return observation

    def update_model(self, observation: np.ndarray, actual: np.ndarray) -> None:
        """
        Learn from observation error via gradient correction.

        Updates M to reduce the discrepancy between what the observer
        predicted (observation) and what actually happened (actual).

        The update rule is a rank-1 outer product correction:
            M += lr * (actual - observation) @ x_implied^T

        where x_implied is the pseudo-inverse reconstruction of the
        state that would have produced the observation.

        Args:
            observation: What the observer saw (obs_dim,)
            actual: What actually happened (obs_dim,)
        """
        observation = np.asarray(observation, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)

        error = actual - observation

        # Reconstruct implied state via pseudo-inverse
        # x_implied = M^+ @ observation
        try:
            x_implied = np.linalg.lstsq(self.model, observation, rcond=None)[0]
        except np.linalg.LinAlgError:
            x_implied = np.zeros(self.dim)

        # Rank-1 gradient update: dM = lr * error (x) x_implied^T
        x_norm_sq = np.dot(x_implied, x_implied)
        if x_norm_sq > 1e-12:
            self.model += self.learning_rate * np.outer(error, x_implied) / x_norm_sq

    def eigenform(
        self,
        f_system: Callable[[np.ndarray], np.ndarray],
        n_iter: int = 100,
        tol: float = 1e-8,
        x0: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Find the eigenform — a fixed point of iterated observation.

        An eigenform x* satisfies: observe(f(x*)) = x*

        This is von Foerster's key insight: stable perceptions are
        fixed points of the observation-action loop. The world the
        observer "sees" is the eigenform of their interaction with it.

        The iteration is: x_{n+1} = observe(f(x_n))
        We seek convergence: ||x_{n+1} - x_n|| < tol

        Args:
            f_system: System dynamics function (state -> next_state)
            n_iter: Maximum iterations
            tol: Convergence tolerance
            x0: Initial state (random if None)

        Returns:
            Eigenform x* (obs_dim,) — the fixed point of observation
        """
        if x0 is not None:
            x = np.asarray(x0, dtype=np.float64).copy()
        else:
            x = np.random.randn(self.obs_dim) * 0.1

        for _ in range(n_iter):
            # Need to lift x from observation space back to system space
            # for f_system to act on it, then project back down
            if self.obs_dim == self.dim:
                x_system = x
            else:
                # Pseudo-inverse lift: find system state consistent with observation
                x_system = np.linalg.lstsq(self.model, x, rcond=None)[0]

            evolved = f_system(x_system)
            x_new = self.observe(evolved)

            if np.linalg.norm(x_new - x) < tol:
                return x_new

            x = x_new

        return x

    def get_observer(self, observer_id: str = "default") -> Observer:
        """Export current state as an Observer dataclass."""
        return Observer(
            id=observer_id,
            model=self.model.copy(),
            observations=[obs.copy() for obs in self._observations],
        )

    @property
    def observation_trajectory(self) -> np.ndarray:
        """Return full observation history as (n_obs, obs_dim) array."""
        if not self._observations:
            return np.empty((0, self.obs_dim))
        return np.array(self._observations)


# ============================================================================
# Requisite Variety (Ashby's Law)
# ============================================================================

class RequisiteVariety:
    """
    Ashby's Law of Requisite Variety: a controller can only control
    a system if its variety (number of distinguishable actions) is
    at least as large as the system's variety (number of distinguishable
    states).

    V(controller) >= V(system)  is required for regulation.

    Variety is measured as log2 of the number of distinguishable states,
    where "distinguishable" means separated by at least a resolution
    threshold in state space.
    """

    def __init__(self, resolution: float = 0.01):
        """
        Args:
            resolution: Minimum distance to consider two states distinguishable
        """
        self.resolution = resolution

    @staticmethod
    def variety(states: np.ndarray, resolution: float = 0.01) -> float:
        """
        Compute variety as log2 of distinct states.

        Two states are "distinct" if their Euclidean distance exceeds
        the resolution threshold. Uses a greedy clustering approach
        to count distinguishable states.

        Args:
            states: Array of states (n_states, dim) or (n_states,)
            resolution: Minimum separation for distinctness

        Returns:
            Variety V = log2(n_distinct)
        """
        states = np.atleast_2d(states)
        if len(states) == 0:
            return 0.0

        # Greedy distinct-state counting
        # Each new state is "distinct" if it's farther than resolution
        # from all previously seen distinct states
        distinct = [states[0]]
        for s in states[1:]:
            distances = np.array([np.linalg.norm(s - d) for d in distinct])
            if np.all(distances > resolution):
                distinct.append(s)

        n_distinct = len(distinct)
        if n_distinct <= 1:
            return 0.0

        return np.log2(n_distinct)

    def ashby_measure(
        self,
        system_states: np.ndarray,
        controller_actions: np.ndarray,
    ) -> float:
        """
        Compute Ashby's variety ratio: V(controller) / V(system).

        A ratio >= 1.0 means the controller has sufficient variety.
        A ratio < 1.0 means the controller cannot fully regulate the system.

        Args:
            system_states: Observed system states (n, dim)
            controller_actions: Controller's action repertoire (m, dim)

        Returns:
            Variety ratio (>= 1.0 means sufficient)
        """
        v_system = self.variety(system_states, self.resolution)
        v_controller = self.variety(controller_actions, self.resolution)

        if v_system < 1e-12:
            return float('inf')  # Trivial system — any controller suffices

        return v_controller / v_system

    def variety_gap(
        self,
        system_states: np.ndarray,
        controller_actions: np.ndarray,
    ) -> float:
        """
        How much variety the controller lacks to regulate the system.

        Gap = max(0, V(system) - V(controller))

        A gap of 0 means the controller is sufficient.
        A positive gap means the controller needs more distinct actions.

        Args:
            system_states: Observed system states (n, dim)
            controller_actions: Controller's action repertoire (m, dim)

        Returns:
            Variety gap in bits (0.0 = sufficient)
        """
        v_system = self.variety(system_states, self.resolution)
        v_controller = self.variety(controller_actions, self.resolution)

        return max(0.0, v_system - v_controller)


# ============================================================================
# Circular Causality
# ============================================================================

class CircularCausality:
    """
    Measure circular causality between observer and system.

    In second-order cybernetics, the observer affects the system
    and the system affects the observer — creating a causal loop.
    We measure this via mutual information between their trajectories.

    If I(system; observer) is high, each is informative about the other,
    indicating strong circular coupling.
    """

    def __init__(self, n_bins: int = 20):
        """
        Args:
            n_bins: Number of histogram bins for mutual information estimation
        """
        self.n_bins = n_bins

    def measure(
        self,
        system_trajectory: np.ndarray,
        observer_trajectory: np.ndarray,
    ) -> float:
        """
        Estimate mutual information between system and observer trajectories.

        Uses binned histogram estimation of I(X;Y) = H(X) + H(Y) - H(X,Y).

        For multi-dimensional trajectories, we project each to its first
        principal component before binning.

        Args:
            system_trajectory: System states over time (T, dim_s) or (T,)
            observer_trajectory: Observer states over time (T, dim_o) or (T,)

        Returns:
            Mutual information in nats (>= 0). Higher = stronger circular causality.
        """
        sys_t = np.atleast_2d(system_trajectory)
        obs_t = np.atleast_2d(observer_trajectory)

        # Align lengths
        T = min(len(sys_t), len(obs_t))
        if T < 2:
            return 0.0

        sys_t = sys_t[:T]
        obs_t = obs_t[:T]

        # Project to 1D via first principal component if multi-dim
        x = self._project_1d(sys_t)
        y = self._project_1d(obs_t)

        # Compute MI via binned histograms
        return self._binned_mi(x, y)

    def is_self_referential(
        self,
        system_fn: Callable[[np.ndarray], np.ndarray],
        observer_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        n_iter: int = 50,
        convergence_tol: float = 1e-6,
    ) -> bool:
        """
        Test whether the observer-system loop converges to a fixed point.

        Iterates:
            x_{n+1} = system_fn(observer_fn(x_n))

        If it converges, the loop is self-referential — a stable
        eigenform exists where the observer and system mutually
        determine each other.

        Args:
            system_fn: System dynamics (state -> state)
            observer_fn: Observer filter (state -> observation)
            x0: Initial state
            n_iter: Maximum iterations
            convergence_tol: Convergence threshold

        Returns:
            True if the loop converges to a fixed point
        """
        x = np.asarray(x0, dtype=np.float64).copy()

        for _ in range(n_iter):
            observed = observer_fn(x)
            x_new = system_fn(observed)

            if np.linalg.norm(x_new - x) < convergence_tol:
                return True

            x = x_new

        return False

    def _project_1d(self, data: np.ndarray) -> np.ndarray:
        """Project multi-dimensional data to 1D via first principal component."""
        if data.shape[1] == 1:
            return data[:, 0]

        centered = data - data.mean(axis=0)
        cov = centered.T @ centered / max(1, len(centered) - 1)

        # First eigenvector (largest eigenvalue)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        pc1 = eigenvectors[:, -1]  # Largest eigenvalue is last

        return centered @ pc1

    def _binned_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mutual information via binned histogram.

        I(X;Y) = H(X) + H(Y) - H(X,Y)
        where H is the Shannon entropy estimated from histograms.
        """
        # Create 2D histogram
        hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=self.n_bins)

        # Normalize to joint probability
        p_xy = hist_xy / hist_xy.sum()

        # Marginals
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)

        # Entropies (in nats)
        h_x = self._entropy(p_x)
        h_y = self._entropy(p_y)
        h_xy = self._entropy(p_xy.ravel())

        # MI = H(X) + H(Y) - H(X,Y), clamped to >= 0
        mi = h_x + h_y - h_xy
        return max(0.0, mi)

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        """Shannon entropy H = -sum(p * ln(p)) for non-zero entries."""
        nonzero = p[p > 0]
        if len(nonzero) == 0:
            return 0.0
        return -np.sum(nonzero * np.log(nonzero))


# ============================================================================
# Convenience: Build Full Cybernetic State
# ============================================================================

def analyze_cybernetic_system(
    system_states: np.ndarray,
    observers: list[SecondOrderObserver],
    controller_actions: np.ndarray | None = None,
) -> CyberneticState:
    """
    Build a complete CyberneticState snapshot.

    Runs each observer on the current system state, measures requisite
    variety (if controller actions provided), and computes circular
    causality between the system and each observer's trajectory.

    Args:
        system_states: System trajectory (T, dim)
        observers: List of SecondOrderObserver instances
        controller_actions: Optional controller action repertoire (m, dim)

    Returns:
        CyberneticState with all measurements
    """
    system_states = np.atleast_2d(system_states)
    current_state = system_states[-1]

    # Build Observer dataclasses
    observer_data = []
    for i, obs in enumerate(observers):
        # Observe current state
        obs.observe(current_state)
        observer_data.append(obs.get_observer(f"observer_{i}"))

    # Requisite variety
    rv = RequisiteVariety()
    if controller_actions is not None:
        req_variety = rv.ashby_measure(system_states, controller_actions)
    else:
        req_variety = 0.0

    # Circular causality (average across observers)
    cc = CircularCausality()
    causality_scores = []
    for obs in observers:
        obs_traj = obs.observation_trajectory
        if len(obs_traj) >= 2 and len(system_states) >= 2:
            mi = cc.measure(system_states, obs_traj)
            causality_scores.append(mi)

    avg_causality = float(np.mean(causality_scores)) if causality_scores else 0.0

    return CyberneticState(
        system=current_state.copy(),
        observers=observer_data,
        requisite_variety=req_variety,
        circular_causality=avg_causality,
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "Observer",
    "CyberneticState",
    "SecondOrderObserver",
    "RequisiteVariety",
    "CircularCausality",
    "analyze_cybernetic_system",
]
