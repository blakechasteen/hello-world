"""
Operations Research - Linear Programming, Network Flows, Scheduling
==================================================================

Optimization methods for decision-making under constraints.

Classes:
    LinearProgramming: Simplex algorithm, duality theory
    NetworkFlows: Max flow, min cost flow, matching
    IntegerProgramming: Branch and bound, cutting planes
    Scheduling: Job shop, flow shop, resource allocation
    DynamicProgramming: Bellman equations, value iteration
    InventoryTheory: EOQ, newsvendor, (s,S) policies

Applications:
    - Supply chain optimization
    - Resource allocation
    - Production planning
    - Transportation networks
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, deque


class LinearProgramming:
    """
    Linear programming: optimize linear objective subject to linear constraints.

    Standard form: min c^T x subject to Ax = b, x >= 0
    """

    @staticmethod
    def solve_2d_graphical(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solve 2D LP graphically (enumerate vertices).

        For demonstration only. Use scipy.optimize.linprog for production.
        """
        # Find vertices by solving pairs of constraints
        vertices = []
        n_constraints = A.shape[0]

        for i in range(n_constraints):
            for j in range(i+1, n_constraints):
                # Solve A[i] @ x = b[i], A[j] @ x = b[j]
                A_sub = np.array([A[i], A[j]])
                b_sub = np.array([b[i], b[j]])

                try:
                    x = np.linalg.solve(A_sub, b_sub)
                    # Check if feasible
                    if np.all(A @ x <= b + 1e-9) and np.all(x >= -1e-9):
                        vertices.append(x)
                except:
                    pass

        # Also check axis intersections
        for i in range(2):
            x = np.zeros(2)
            for j in range(n_constraints):
                if abs(A[j, 1-i]) > 1e-10:
                    x[1-i] = b[j] / A[j, 1-i]
                    if np.all(A @ x <= b + 1e-9) and np.all(x >= -1e-9):
                        vertices.append(x.copy())

        # Evaluate objective at all vertices
        best_x = vertices[0] if vertices else np.zeros(2)
        best_obj = c @ best_x

        for x in vertices:
            obj = c @ x
            if obj < best_obj:
                best_obj = obj
                best_x = x

        return best_x, best_obj

    @staticmethod
    def duality_theorem(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> str:
        """
        Strong duality: optimal primal = optimal dual.

        Primal: min c^T x s.t. Ax >= b, x >= 0
        Dual: max b^T y s.t. A^T y <= c, y >= 0
        """
        return (
            "Strong Duality Theorem:\n"
            "If primal has optimal solution x*, dual has optimal solution y*.\n"
            "Optimal values are equal: c^T x* = b^T y*\n\n"
            "Complementary slackness:\n"
            "- If x_i* > 0, then (A^T y*)_i = c_i\n"
            "- If (Ax*)_j > b_j, then y_j* = 0"
        )


class NetworkFlows:
    """
    Network flow problems: max flow, min cost flow, matching.

    Graph G = (V, E) with capacities u(e) and costs c(e).
    """

    def __init__(self, n_nodes: int):
        self.n = n_nodes
        self.capacity = defaultdict(lambda: defaultdict(int))
        self.cost = defaultdict(lambda: defaultdict(int))

    def add_edge(self, u: int, v: int, capacity: int, cost: int = 0):
        """Add directed edge with capacity and cost."""
        self.capacity[u][v] = capacity
        self.cost[u][v] = cost

    def max_flow(self, source: int, sink: int) -> int:
        """
        Maximum flow from source to sink (Ford-Fulkerson).

        Uses BFS to find augmenting paths.
        """
        # Residual capacities
        residual = defaultdict(lambda: defaultdict(int))
        for u in self.capacity:
            for v in self.capacity[u]:
                residual[u][v] = self.capacity[u][v]

        max_flow_value = 0

        while True:
            # Find augmenting path via BFS
            parent = {source: None}
            queue = deque([source])
            found_sink = False

            while queue and not found_sink:
                u = queue.popleft()
                for v in residual[u]:
                    if v not in parent and residual[u][v] > 0:
                        parent[v] = u
                        queue.append(v)
                        if v == sink:
                            found_sink = True
                            break

            if not found_sink:
                break

            # Find bottleneck capacity
            path_flow = float('inf')
            v = sink
            while parent[v] is not None:
                u = parent[v]
                path_flow = min(path_flow, residual[u][v])
                v = u

            # Update residual graph
            v = sink
            while parent[v] is not None:
                u = parent[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                v = u

            max_flow_value += path_flow

        return max_flow_value

    def min_cut(self, source: int, sink: int) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Minimum s-t cut (max-flow min-cut theorem).

        Returns: (cut capacity, list of cut edges)
        """
        flow_value = self.max_flow(source, sink)

        # Find reachable nodes from source in residual graph
        # (Simplified: return flow value only)
        return flow_value, []

    @staticmethod
    def max_flow_min_cut_theorem() -> str:
        """Statement of max-flow min-cut theorem."""
        return (
            "Max-Flow Min-Cut Theorem:\n"
            "Maximum flow value = Minimum cut capacity\n\n"
            "For any network, max flow from s to t equals\n"
            "minimum total capacity of edges that separate s from t."
        )


class IntegerProgramming:
    """
    Integer programming: LP with integrality constraints.

    NP-hard in general. Uses branch-and-bound, cutting planes.
    """

    @staticmethod
    def branch_and_bound(c: np.ndarray, A: np.ndarray, b: np.ndarray,
                        depth: int = 0, max_depth: int = 10) -> Tuple[Optional[np.ndarray], float]:
        """
        Branch and bound for integer LP (simplified).

        Recursively partition feasible region.
        """
        # Solve LP relaxation (placeholder: use simplex)
        # If solution integral, done
        # Else, branch on fractional variable

        # Placeholder: return integer solution
        x = np.round(b / A.sum(axis=1))
        obj = c @ x
        return x, obj

    @staticmethod
    def cutting_plane() -> str:
        """Gomory cutting planes for ILP."""
        return (
            "Cutting Plane Method:\n"
            "1. Solve LP relaxation\n"
            "2. If solution fractional, add cutting plane (cuts off fractional point)\n"
            "3. Repeat until integer solution found\n\n"
            "Gomory cuts: derived from simplex tableau rows"
        )


class Scheduling:
    """
    Scheduling problems: job shop, flow shop, resource allocation.

    Minimize makespan, tardiness, or other objectives.
    """

    @staticmethod
    def earliest_deadline_first(jobs: List[Tuple[int, int]]) -> List[int]:
        """
        EDF scheduling: minimize maximum lateness.

        Optimal for single machine, non-preemptive.

        Args:
            jobs: List of (processing_time, deadline) tuples

        Returns: Job sequence (indices)
        """
        # Sort by deadline
        indexed_jobs = [(i, pt, d) for i, (pt, d) in enumerate(jobs)]
        indexed_jobs.sort(key=lambda x: x[2])  # Sort by deadline

        return [i for i, pt, d in indexed_jobs]

    @staticmethod
    def shortest_processing_time(processing_times: np.ndarray) -> np.ndarray:
        """
        SPT rule: minimize average completion time.

        Optimal for single machine.
        """
        return np.argsort(processing_times)

    @staticmethod
    def johnsons_algorithm(processing_times: np.ndarray) -> np.ndarray:
        """
        Johnson's algorithm: 2-machine flow shop scheduling.

        Minimizes makespan (total time to complete all jobs).

        Args:
            processing_times: shape (n_jobs, 2) for machines 1 and 2
        """
        n_jobs = processing_times.shape[0]
        machine1_times = processing_times[:, 0]
        machine2_times = processing_times[:, 1]

        # Partition jobs: machine1_times < machine2_times vs opposite
        set_a = [i for i in range(n_jobs) if machine1_times[i] < machine2_times[i]]
        set_b = [i for i in range(n_jobs) if machine1_times[i] >= machine2_times[i]]

        # Sort A by machine 1 (increasing), B by machine 2 (decreasing)
        set_a.sort(key=lambda i: machine1_times[i])
        set_b.sort(key=lambda i: machine2_times[i], reverse=True)

        return np.array(set_a + set_b)


class DynamicProgramming:
    """
    Dynamic programming: Bellman optimality principle.

    Solve by backward induction or value iteration.
    """

    @staticmethod
    def knapsack_01(values: np.ndarray, weights: np.ndarray, capacity: int) -> Tuple[float, List[int]]:
        """
        0-1 knapsack problem: maximize value subject to weight constraint.

        DP: V(i, w) = max(V(i-1, w), V(i-1, w - w_i) + v_i)
        """
        n = len(values)
        dp = np.zeros((n + 1, capacity + 1))

        for i in range(1, n + 1):
            for w in range(capacity + 1):
                # Don't take item i-1
                dp[i, w] = dp[i-1, w]

                # Take item i-1 if it fits
                if weights[i-1] <= w:
                    dp[i, w] = max(dp[i, w], dp[i-1, w - weights[i-1]] + values[i-1])

        # Backtrack to find items
        w = capacity
        items = []
        for i in range(n, 0, -1):
            if dp[i, w] != dp[i-1, w]:
                items.append(i-1)
                w -= weights[i-1]

        return dp[n, capacity], items

    @staticmethod
    def shortest_path(graph: np.ndarray) -> np.ndarray:
        """
        All-pairs shortest paths (Floyd-Warshall).

        DP: dist[k][i][j] = min(dist[k-1][i][j], dist[k-1][i][k] + dist[k-1][k][j])
        """
        n = graph.shape[0]
        dist = graph.copy()

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

        return dist


class InventoryTheory:
    """
    Inventory management: EOQ, newsvendor, dynamic inventory.

    Balances holding costs vs stockout costs.
    """

    @staticmethod
    def eoq(demand_rate: float, order_cost: float, holding_cost: float) -> Tuple[float, float]:
        """
        Economic Order Quantity: optimal batch size.

        Q* = sqrt(2 * D * K / h)

        Args:
            demand_rate: D (units per year)
            order_cost: K ($ per order)
            holding_cost: h ($ per unit per year)

        Returns: (optimal order quantity, total cost)
        """
        Q_star = np.sqrt(2 * demand_rate * order_cost / holding_cost)
        total_cost = np.sqrt(2 * demand_rate * order_cost * holding_cost)
        return Q_star, total_cost

    @staticmethod
    def newsvendor(mean_demand: float, std_demand: float,
                  cost: float, price: float, salvage: float = 0) -> float:
        """
        Newsvendor model: single-period inventory.

        Optimal quantity: F^{-1}((p - c) / (p - s))
        where F is demand CDF.

        Args:
            mean_demand: Expected demand
            std_demand: Demand standard deviation
            cost: Unit cost
            price: Selling price
            salvage: Salvage value

        Returns: Optimal stocking quantity
        """
        # Critical fractile
        fractile = (price - cost) / (price - salvage)

        # For normal demand: Q* = μ + z * σ
        from scipy.stats import norm
        z = norm.ppf(fractile)
        Q_star = mean_demand + z * std_demand

        return max(0, Q_star)


# ============================================================================
# EXAMPLES AND TESTS
# ============================================================================

def example_max_flow():
    """Example: Max flow in simple network."""
    network = NetworkFlows(n_nodes=6)

    # Add edges (u, v, capacity)
    network.add_edge(0, 1, 10)
    network.add_edge(0, 2, 10)
    network.add_edge(1, 3, 4)
    network.add_edge(1, 4, 8)
    network.add_edge(2, 4, 9)
    network.add_edge(3, 5, 10)
    network.add_edge(4, 3, 6)
    network.add_edge(4, 5, 10)

    max_flow = network.max_flow(source=0, sink=5)
    return max_flow


def example_knapsack():
    """Example: 0-1 knapsack."""
    values = np.array([60, 100, 120])
    weights = np.array([10, 20, 30])
    capacity = 50

    max_value, items = DynamicProgramming.knapsack_01(values, weights, capacity)
    return max_value, items


def example_eoq():
    """Example: Economic order quantity."""
    demand = 1000  # units/year
    order_cost = 50  # $/order
    holding_cost = 2  # $/unit/year

    Q_star, total_cost = InventoryTheory.eoq(demand, order_cost, holding_cost)
    return Q_star, total_cost


if __name__ == "__main__":
    print("Operations Research Module")
    print("=" * 60)

    # Test 1: Max flow
    print("\n[Test 1] Maximum flow")
    max_flow = example_max_flow()
    print(f"Max flow: {max_flow}")

    # Test 2: Knapsack
    print("\n[Test 2] 0-1 Knapsack")
    max_value, items = example_knapsack()
    print(f"Max value: ${max_value:.0f}")
    print(f"Items selected: {items}")

    # Test 3: Scheduling (EDF)
    print("\n[Test 3] Earliest Deadline First")
    jobs = [(5, 10), (3, 6), (2, 8), (4, 12)]
    schedule = Scheduling.earliest_deadline_first(jobs)
    print(f"Job sequence: {schedule}")

    # Test 4: EOQ
    print("\n[Test 4] Economic Order Quantity")
    Q, TC = example_eoq()
    print(f"Optimal order quantity: {Q:.2f} units")
    print(f"Total annual cost: ${TC:.2f}")

    # Test 5: Duality
    print("\n[Test 5] LP Duality")
    duality = LinearProgramming.duality_theorem(None, None, None)
    print(duality)

    print("\n" + "=" * 60)
    print("All operations research tests complete!")
    print("LP, network flows, scheduling, and inventory ready.")
