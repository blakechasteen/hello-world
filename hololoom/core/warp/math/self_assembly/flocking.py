"""
Flocking & Consensus — Boids, Average Consensus, Byzantine Fault Tolerance
===========================================================================

Collective behavior and agreement protocols: from Reynolds flocking
rules to distributed consensus algorithms with fault tolerance.

Core Concepts:
- Boids: Separation + Alignment + Cohesion → emergent flocking
- Consensus Protocol: Agents converge to a shared value via local averaging
- Byzantine Fault Tolerance: Agreement despite f < n/3 malicious nodes

Mathematical Foundation:
Boids: v_i ← v_i + w_s*s_i + w_a*a_i + w_c*c_i (separation, alignment, cohesion)
Consensus: x_i(t+1) = Σ_j w_ij x_j(t), converges if graph is connected
BFT: Agreement possible iff n ≥ 3f + 1 (Lamport, Shostak, Pease 1982)

Applications to Warp Space:
- Flocking for coordinated embedding movement in latent space
- Consensus for distributed scoring term agreement across memory systems
- BFT for robust aggregation with unreliable data sources
- Collective intelligence emergence from simple local rules

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
class FlockState:
    """State of a flock/swarm simulation.

    Attributes:
        positions: Agent positions (n_agents, d).
        velocities: Agent velocities (n_agents, d).
        alignment_order: Alignment order parameter Φ ∈ [0, 1].
        center_of_mass: Flock centroid (d,).
    """
    positions: np.ndarray
    velocities: np.ndarray
    alignment_order: float
    center_of_mass: np.ndarray


@dataclass
class ConsensusResult:
    """Result of consensus protocol.

    Attributes:
        values: Final values (n,).
        consensus_value: Agreed-upon value (mean of final values).
        converged: Whether convergence was achieved.
        iterations: Number of iterations taken.
        history: Value history (n_iterations, n).
    """
    values: np.ndarray
    consensus_value: float
    converged: bool
    iterations: int
    history: np.ndarray | None = None


# ============================================================================
# BOIDS (REYNOLDS FLOCKING)
# ============================================================================

class Boids:
    """Reynolds Boids — emergent flocking from three simple rules.

    1. Separation: Steer away from nearby neighbors (avoid collision)
    2. Alignment: Steer toward average heading of nearby neighbors
    3. Cohesion: Steer toward centroid of nearby neighbors

    Each rule operates within its own interaction radius.
    Emergent behavior: schooling, flocking, herding.

    Example:
        >>> boids = Boids()
        >>> n = 100
        >>> pos = np.random.randn(n, 2) * 10
        >>> vel = np.random.randn(n, 2)
        >>> for _ in range(200):
        ...     state = boids.step(pos, vel)
        ...     pos, vel = state.positions, state.velocities
    """

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        separation_r: float = 2.0,
        alignment_r: float = 5.0,
        cohesion_r: float = 5.0,
        separation_weight: float = 1.5,
        alignment_weight: float = 1.0,
        cohesion_weight: float = 1.0,
        max_speed: float = 2.0,
        max_force: float = 0.5,
        dt: float = 1.0,
    ) -> FlockState:
        """Execute one simulation step.

        Args:
            positions: Current positions (n, d).
            velocities: Current velocities (n, d).
            separation_r: Separation interaction radius.
            alignment_r: Alignment interaction radius.
            cohesion_r: Cohesion interaction radius.
            separation_weight: Weight for separation force.
            alignment_weight: Weight for alignment force.
            cohesion_weight: Weight for cohesion force.
            max_speed: Maximum agent speed.
            max_force: Maximum steering force magnitude.
            dt: Time step.

        Returns:
            FlockState with updated positions and velocities.
        """
        positions = np.asarray(positions, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        n, d = positions.shape

        # Pairwise distance matrix
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)

        new_vel = velocities.copy()

        for i in range(n):
            force = np.zeros(d)

            # Separation: avoid neighbors within separation_r
            sep_mask = (dists[i] < separation_r) & (dists[i] > 0)
            if np.any(sep_mask):
                away = -diff[i][sep_mask]
                # Weight inversely by distance
                weights = 1.0 / (dists[i][sep_mask, np.newaxis] + 1e-6)
                sep_force = np.sum(away * weights, axis=0)
                sep_force = self._limit(sep_force, max_force)
                force += separation_weight * sep_force

            # Alignment: match velocity of neighbors within alignment_r
            align_mask = (dists[i] < alignment_r) & (dists[i] > 0)
            if np.any(align_mask):
                avg_vel = np.mean(velocities[align_mask], axis=0)
                align_force = avg_vel - velocities[i]
                align_force = self._limit(align_force, max_force)
                force += alignment_weight * align_force

            # Cohesion: steer toward centroid of neighbors within cohesion_r
            coh_mask = (dists[i] < cohesion_r) & (dists[i] > 0)
            if np.any(coh_mask):
                centroid = np.mean(positions[coh_mask], axis=0)
                toward = centroid - positions[i]
                coh_force = self._limit(toward, max_force)
                force += cohesion_weight * coh_force

            # Apply force
            new_vel[i] += force * dt

            # Speed limit
            speed = np.linalg.norm(new_vel[i])
            if speed > max_speed:
                new_vel[i] = new_vel[i] / speed * max_speed

        new_pos = positions + new_vel * dt

        # Compute alignment order parameter: Φ = |Σv_i| / Σ|v_i|
        speeds = np.linalg.norm(new_vel, axis=1)
        total_speed = np.sum(speeds)
        if total_speed > 0:
            alignment_order = float(np.linalg.norm(np.sum(new_vel, axis=0)) / total_speed)
        else:
            alignment_order = 0.0

        return FlockState(
            positions=new_pos,
            velocities=new_vel,
            alignment_order=alignment_order,
            center_of_mass=np.mean(new_pos, axis=0),
        )

    @staticmethod
    def _limit(vec: np.ndarray, max_mag: float) -> np.ndarray:
        """Limit vector magnitude."""
        mag = np.linalg.norm(vec)
        if mag > max_mag:
            return vec / mag * max_mag
        return vec

    def simulate(
        self,
        n_agents: int = 50,
        dim: int = 2,
        n_steps: int = 200,
        **kwargs,
    ) -> list[FlockState]:
        """Run full simulation from random initial conditions.

        Args:
            n_agents: Number of boids.
            dim: Spatial dimension.
            n_steps: Number of time steps.
            **kwargs: Passed to step().

        Returns:
            List of FlockState for each time step.
        """
        positions = np.random.randn(n_agents, dim) * 10.0
        velocities = np.random.randn(n_agents, dim) * 0.5
        history = []

        for _ in range(n_steps):
            state = self.step(positions, velocities, **kwargs)
            positions = state.positions
            velocities = state.velocities
            history.append(state)

        return history


# ============================================================================
# AVERAGE CONSENSUS PROTOCOL
# ============================================================================

class ConsensusProtocol:
    """Linear average consensus protocol.

    x_i(t+1) = Σ_j w_ij x_j(t)

    Agents update their values as weighted averages of neighbors.
    Converges to the average of initial values if the weight matrix
    is doubly stochastic and the communication graph is connected.

    Example:
        >>> cp = ConsensusProtocol()
        >>> values = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        >>> # Ring graph adjacency
        >>> adj = np.array([[0,1,0,0,1],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[1,0,0,1,0]])
        >>> result = cp.reach_consensus(values, adj)
    """

    def reach_consensus(
        self,
        values: np.ndarray,
        adjacency: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-8,
        method: str = 'metropolis',
    ) -> ConsensusResult:
        """Run consensus protocol until convergence.

        Args:
            values: Initial values (n,).
            adjacency: Binary adjacency matrix (n, n). Symmetric.
            max_iter: Maximum iterations.
            tol: Convergence tolerance (max deviation from mean).
            method: Weight construction method ('metropolis', 'laplacian', 'max_degree').

        Returns:
            ConsensusResult.
        """
        values = np.asarray(values, dtype=float).copy()
        adjacency = np.asarray(adjacency, dtype=float)
        n = len(values)

        # Build doubly stochastic weight matrix
        W = self._build_weights(adjacency, method)

        history = [values.copy()]

        for it in range(max_iter):
            values = W @ values
            history.append(values.copy())

            # Check convergence
            if np.max(np.abs(values - np.mean(values))) < tol:
                return ConsensusResult(
                    values=values,
                    consensus_value=float(np.mean(values)),
                    converged=True,
                    iterations=it + 1,
                    history=np.array(history),
                )

        return ConsensusResult(
            values=values,
            consensus_value=float(np.mean(values)),
            converged=False,
            iterations=max_iter,
            history=np.array(history),
        )

    @staticmethod
    def _build_weights(adjacency: np.ndarray, method: str) -> np.ndarray:
        """Build doubly stochastic weight matrix from adjacency."""
        n = len(adjacency)

        if method == 'metropolis':
            # Metropolis-Hastings weights
            W = np.zeros((n, n))
            degrees = adjacency.sum(axis=1)
            for i in range(n):
                for j in range(n):
                    if i != j and adjacency[i, j] > 0:
                        W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
                W[i, i] = 1.0 - np.sum(W[i, :])
        elif method == 'laplacian':
            # Laplacian-based: W = I - εL
            degrees = adjacency.sum(axis=1)
            L = np.diag(degrees) - adjacency
            max_deg = np.max(degrees) if np.max(degrees) > 0 else 1.0
            epsilon = 1.0 / (max_deg + 1.0)
            W = np.eye(n) - epsilon * L
        else:  # max_degree
            degrees = adjacency.sum(axis=1)
            max_deg = np.max(degrees) if np.max(degrees) > 0 else 1.0
            W = np.eye(n) + adjacency / (max_deg + 1.0)
            W = W / W.sum(axis=1, keepdims=True)

        return W

    def convergence_rate(self, adjacency: np.ndarray, method: str = 'metropolis') -> float:
        """Compute asymptotic convergence rate.

        Rate = |λ₂(W)| where λ₂ is the second largest eigenvalue magnitude.
        Smaller is faster convergence.

        Args:
            adjacency: Graph adjacency matrix.
            method: Weight construction method.

        Returns:
            Convergence rate ∈ [0, 1).
        """
        W = self._build_weights(adjacency, method)
        eigenvalues = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]
        if len(eigenvalues) < 2:
            return 0.0
        return float(eigenvalues[1])


# ============================================================================
# BYZANTINE FAULT TOLERANCE
# ============================================================================

class ByzantineFaultTolerance:
    """Byzantine fault-tolerant consensus.

    Achieves agreement among n nodes even when up to f < n/3 nodes
    are Byzantine (can send arbitrary messages). Uses trimmed mean
    approach as a practical approximation.

    The fundamental bound: consensus is impossible for n ≤ 3f
    (Lamport, Shostak, Pease 1982).

    Example:
        >>> bft = ByzantineFaultTolerance()
        >>> values = np.array([1.0, 2.0, 1.5, 100.0, 1.8])  # Node 3 is Byzantine
        >>> result = bft.consensus(values, faulty_nodes={3}, n_rounds=5)
    """

    def consensus(
        self,
        values: np.ndarray,
        faulty_nodes: set,
        n_rounds: int = 10,
        tol: float = 1e-6,
    ) -> float:
        """Run Byzantine fault-tolerant consensus.

        Uses the trimmed mean approach: in each round, each honest node
        removes the f highest and f lowest values from its view and
        averages the rest.

        Args:
            values: Initial values for all nodes (n,), including faulty.
            faulty_nodes: Set of faulty node indices.
            n_rounds: Number of consensus rounds.
            tol: Convergence tolerance.

        Returns:
            Consensus value among honest nodes.
        """
        values = np.asarray(values, dtype=float).copy()
        n = len(values)
        f = len(faulty_nodes)

        if n <= 3 * f:
            logger.warning(f"BFT impossible: n={n} ≤ 3f={3*f}")
            # Still try — will get approximate result
            honest = [i for i in range(n) if i not in faulty_nodes]
            return float(np.mean(values[honest])) if honest else float(np.mean(values))

        honest_nodes = [i for i in range(n) if i not in faulty_nodes]

        for round_num in range(n_rounds):
            new_values = values.copy()

            for i in honest_nodes:
                # Each honest node sees all values (broadcast model)
                all_vals = values.copy()

                # Faulty nodes may send arbitrary values
                # (In this simulation, they keep their original values,
                # but in practice they could send anything)

                # Trimmed mean: remove f highest and f lowest
                sorted_vals = np.sort(all_vals)
                if 2 * f < n:
                    trimmed = sorted_vals[f: n - f]
                    new_values[i] = np.mean(trimmed)
                else:
                    new_values[i] = np.median(all_vals)

            # Check convergence among honest nodes
            honest_vals = new_values[honest_nodes]
            if np.max(honest_vals) - np.min(honest_vals) < tol:
                values = new_values
                break

            values = new_values

        # Return mean of honest node values
        return float(np.mean(values[honest_nodes]))

    @staticmethod
    def resilience_bound(n: int) -> int:
        """Maximum number of faulty nodes tolerable.

        Byzantine consensus requires n ≥ 3f + 1, so f ≤ (n-1)/3.

        Args:
            n: Total number of nodes.

        Returns:
            Maximum f (number of faulty nodes).
        """
        return (n - 1) // 3

    def approximate_consensus(
        self,
        values: np.ndarray,
        adjacency: np.ndarray,
        f: int,
        n_rounds: int = 20,
    ) -> ConsensusResult:
        """Approximate Byzantine consensus on a graph.

        Each node only communicates with neighbors. Uses the
        W-MSR (Weighted Mean-Subsequence-Reduced) algorithm.

        Args:
            values: Initial values (n,).
            adjacency: Graph adjacency matrix.
            f: Number of faulty nodes to tolerate.
            n_rounds: Number of rounds.

        Returns:
            ConsensusResult.
        """
        values = np.asarray(values, dtype=float).copy()
        n = len(values)
        history = [values.copy()]

        for _ in range(n_rounds):
            new_values = values.copy()

            for i in range(n):
                # Get neighbor values
                neighbors = np.where(adjacency[i] > 0)[0]
                neighbor_vals = values[neighbors]

                if len(neighbor_vals) <= 2 * f:
                    # Not enough neighbors for filtering
                    new_values[i] = values[i]
                    continue

                # Remove f highest and f lowest among neighbors + self
                all_vals = np.append(neighbor_vals, values[i])
                sorted_vals = np.sort(all_vals)

                if 2 * f < len(sorted_vals):
                    trimmed = sorted_vals[f: len(sorted_vals) - f]
                    new_values[i] = np.mean(trimmed)
                else:
                    new_values[i] = np.median(all_vals)

            values = new_values
            history.append(values.copy())

        return ConsensusResult(
            values=values,
            consensus_value=float(np.mean(values)),
            converged=bool(np.max(values) - np.min(values) < 0.01),
            iterations=n_rounds,
            history=np.array(history),
        )
