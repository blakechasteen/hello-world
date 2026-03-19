"""
Swarm Intelligence — PSO and Ant Colony Optimization
=====================================================

Decentralized optimization via collective behavior of simple agents.
No central controller — intelligence emerges from local interactions.

Core Concepts:
- PSO: Particles fly through search space, attracted to personal and global best
- ACO: Ants deposit pheromone on good paths, amplifying successful routes
- Stigmergy: Indirect coordination through environmental modification

Mathematical Foundation:
PSO velocity: v_i ← w*v_i + c₁r₁(pbest_i - x_i) + c₂r₂(gbest - x_i)
PSO position: x_i ← x_i + v_i
ACO pheromone: τ_ij ← (1-ρ)τ_ij + Σ_k Δτ_ij^k

Applications to Warp Space:
- Global optimization of weaving cycle parameters
- Combinatorial optimization for memory retrieval ordering
- Emergent coordination in multi-agent Thompson Sampling

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# PARTICLE SWARM OPTIMIZATION
# ============================================================================

class ParticleSwarmOptimization:
    """Particle Swarm Optimization (PSO).

    Each particle maintains position, velocity, and personal best.
    The swarm converges toward the global best via social and
    cognitive acceleration terms.

    v_i ← w*v_i + c₁*r₁*(pbest_i - x_i) + c₂*r₂*(gbest - x_i)
    x_i ← x_i + v_i

    Example:
        >>> pso = ParticleSwarmOptimization()
        >>> def sphere(x): return -np.sum(x**2)  # maximize
        >>> pos, val = pso.optimize(sphere, bounds=[(-5, 5)]*3,
        ...                         n_particles=30, n_iterations=100)
        >>> np.allclose(pos, 0, atol=0.5)
        True
    """

    def optimize(
        self,
        f: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        n_particles: int = 30,
        n_iterations: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        w_decay: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run PSO optimization.

        Args:
            f: Objective function (maximize)
            bounds: [(lo, hi)] for each dimension
            n_particles: Number of particles
            n_iterations: Number of iterations
            w: Inertia weight
            c1: Cognitive acceleration (toward personal best)
            c2: Social acceleration (toward global best)
            w_decay: If set, linearly decay w to this value

        Returns:
            (best_position, best_value)
        """
        bounds = np.array(bounds, dtype=float)
        lo = bounds[:, 0]
        hi = bounds[:, 1]
        d = len(bounds)

        # Initialize particles
        positions = np.random.uniform(lo, hi, size=(n_particles, d))
        velocities = np.random.uniform(
            -(hi - lo) * 0.1, (hi - lo) * 0.1, size=(n_particles, d)
        )

        # Evaluate initial fitness
        fitness = np.array([f(x) for x in positions])

        # Personal bests
        pbest_pos = positions.copy()
        pbest_val = fitness.copy()

        # Global best
        gbest_idx = np.argmax(fitness)
        gbest_pos = positions[gbest_idx].copy()
        gbest_val = fitness[gbest_idx]

        w_start = w
        w_end = w_decay if w_decay is not None else w

        for iteration in range(n_iterations):
            # Linearly decay inertia
            w_current = w_start - (w_start - w_end) * iteration / max(n_iterations - 1, 1)

            for i in range(n_particles):
                r1 = np.random.random(d)
                r2 = np.random.random(d)

                # Update velocity
                cognitive = c1 * r1 * (pbest_pos[i] - positions[i])
                social = c2 * r2 * (gbest_pos - positions[i])
                velocities[i] = w_current * velocities[i] + cognitive + social

                # Clamp velocity
                v_max = (hi - lo) * 0.2
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lo, hi)

                # Evaluate
                fit = f(positions[i])
                fitness[i] = fit

                # Update personal best
                if fit > pbest_val[i]:
                    pbest_val[i] = fit
                    pbest_pos[i] = positions[i].copy()

                    # Update global best
                    if fit > gbest_val:
                        gbest_val = fit
                        gbest_pos = positions[i].copy()

        return gbest_pos, gbest_val


# ============================================================================
# ANT COLONY OPTIMIZATION (for TSP)
# ============================================================================

class AntColonyOptimization:
    """Ant Colony Optimization for the Traveling Salesman Problem.

    Ants construct tours probabilistically based on pheromone trails
    and heuristic (distance) information. Pheromone is deposited on
    good tours and evaporates over time.

    P(next = j) ∝ τ_ij^α * η_ij^β  where η = 1/distance

    Example:
        >>> aco = AntColonyOptimization()
        >>> n = 10
        >>> coords = np.random.rand(n, 2)
        >>> dist = np.zeros((n, n))
        >>> for i in range(n):
        ...     for j in range(n):
        ...         dist[i, j] = np.linalg.norm(coords[i] - coords[j])
        >>> path, cost = aco.solve_tsp(dist, n_ants=20, n_iterations=50)
        >>> len(path) == n
        True
    """

    def solve_tsp(
        self,
        distance_matrix: np.ndarray,
        n_ants: int = 20,
        n_iterations: int = 100,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.5,
        Q: float = 1.0,
    ) -> tuple[list[int], float]:
        """Solve TSP using ant colony optimization.

        Args:
            distance_matrix: Distance between cities (n, n)
            n_ants: Number of ants per iteration
            n_iterations: Number of iterations
            alpha: Pheromone importance exponent
            beta: Heuristic (distance) importance exponent
            evaporation: Pheromone evaporation rate (0 to 1)
            Q: Pheromone deposit constant

        Returns:
            (best_path, best_cost)
        """
        D = np.asarray(distance_matrix, dtype=float)
        n = D.shape[0]

        # Initialize pheromone
        tau = np.ones((n, n)) / n

        # Heuristic: inverse distance
        eta = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and D[i, j] > 0:
                    eta[i, j] = 1.0 / D[i, j]

        best_path = list(range(n))
        best_cost = self._tour_cost(best_path, D)

        for iteration in range(n_iterations):
            paths = []
            costs = []

            for ant in range(n_ants):
                path = self._construct_tour(tau, eta, alpha, beta, n)
                cost = self._tour_cost(path, D)
                paths.append(path)
                costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_path = path[:]

            # Update pheromone
            tau = stigmergy_update(tau, paths, costs, evaporation, Q)

        return best_path, best_cost

    @staticmethod
    def _construct_tour(
        tau: np.ndarray,
        eta: np.ndarray,
        alpha: float,
        beta: float,
        n: int,
    ) -> list[int]:
        """Construct a single ant's tour."""
        start = np.random.randint(n)
        visited = {start}
        path = [start]

        current = start
        for _ in range(n - 1):
            # Compute probabilities for unvisited cities
            probs = np.zeros(n)
            for j in range(n):
                if j not in visited:
                    probs[j] = (tau[current, j] ** alpha) * (eta[current, j] ** beta)

            total = probs.sum()
            if total < 1e-15:
                # All probabilities zero — pick random unvisited
                unvisited = [j for j in range(n) if j not in visited]
                next_city = np.random.choice(unvisited)
            else:
                probs /= total
                next_city = int(np.random.choice(n, p=probs))

            path.append(next_city)
            visited.add(next_city)
            current = next_city

        return path

    @staticmethod
    def _tour_cost(path: list[int], D: np.ndarray) -> float:
        """Compute total tour cost (including return to start)."""
        cost = 0.0
        for i in range(len(path)):
            cost += D[path[i], path[(i + 1) % len(path)]]
        return cost


# ============================================================================
# STIGMERGY UPDATE
# ============================================================================

def stigmergy_update(
    pheromone_matrix: np.ndarray,
    paths: list[list[int]],
    costs: list[float],
    evaporation: float = 0.5,
    Q: float = 1.0,
) -> np.ndarray:
    """Update pheromone matrix based on ant paths (stigmergy).

    τ_ij ← (1 - ρ) τ_ij + Σ_k Δτ_ij^k
    where Δτ_ij^k = Q / cost_k if ant k used edge (i,j)

    Args:
        pheromone_matrix: Current pheromone levels (n, n)
        paths: List of tours (each a list of city indices)
        costs: Cost of each tour
        evaporation: Evaporation rate ρ ∈ [0, 1]
        Q: Pheromone deposit constant

    Returns:
        Updated pheromone matrix
    """
    tau = pheromone_matrix.copy()

    # Evaporation
    tau *= (1 - evaporation)

    # Deposit
    for path, cost in zip(paths, costs):
        if cost < 1e-15:
            continue
        deposit = Q / cost
        for i in range(len(path)):
            j = path[(i + 1) % len(path)]
            tau[path[i], j] += deposit
            tau[j, path[i]] += deposit  # Symmetric

    # Minimum pheromone to prevent starvation
    tau = np.maximum(tau, 1e-10)

    return tau
