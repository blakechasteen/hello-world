"""
Math Modules Performance Benchmark
==================================

Benchmarks HoloLoom's math modules across different input sizes.

Metrics Per Algorithm:
- Latency: p50, p95, p99 (100 iterations each)
- Throughput: ops/second
- Scaling: O(n) verification with increasing input sizes

Run: PYTHONPATH=. python demos/benchmark_math_modules.py
"""

import numpy as np
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import networkx as nx

# Math modules
from hololoom.warp.math.information import entropy, mutual_information, DistributionPair
from hololoom.warp.math.graph import (
    compute_laplacian, fiedler_vector, algebraic_connectivity,
    SpectralClustering
)
from hololoom.warp.math.decision.game_theory import NormalFormGame, NashEquilibrium
from hololoom.warp.math.causal import CausalGraph
from hololoom.warp.math.decision.regret_minimization import Hedge


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    input_size: int
    iterations: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    ops_per_sec: float


def percentile(values: List[float], p: float) -> float:
    """Compute percentile of values."""
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def benchmark_function(fn, iterations: int = 100) -> Dict[str, float]:
    """Run a function multiple times and compute timing statistics."""
    times_ms = []

    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)

    total_time = sum(times_ms)

    return {
        "p50": percentile(times_ms, 50),
        "p95": percentile(times_ms, 95),
        "p99": percentile(times_ms, 99),
        "ops_per_sec": iterations / (total_time / 1000) if total_time > 0 else float('inf')
    }


def print_header(title: str):
    """Print a formatted header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(size: int, stats: Dict[str, float]):
    """Print benchmark result row."""
    print(f"  Size {size:5d}: p50={stats['p50']:.3f}ms, "
          f"p95={stats['p95']:.3f}ms, p99={stats['p99']:.3f}ms, "
          f"ops/s={stats['ops_per_sec']:.0f}")


# =============================================================================
# Benchmark 1: Information Theory
# =============================================================================

def benchmark_information_theory():
    """Benchmark entropy and mutual information calculations."""
    print_header("Benchmark 1: Information Theory")

    sizes = [10, 100, 1000, 10000]
    results = []

    print("\n  --- Shannon Entropy ---")
    for size in sizes:
        # Generate random probability distribution
        probs = np.random.dirichlet(np.ones(size))

        stats = benchmark_function(lambda: entropy(probs, base=2.0), iterations=100)
        print_result(size, stats)
        results.append(BenchmarkResult(
            name="entropy", input_size=size, iterations=100,
            p50_ms=stats["p50"], p95_ms=stats["p95"],
            p99_ms=stats["p99"], ops_per_sec=stats["ops_per_sec"]
        ))

    print("\n  --- Mutual Information ---")
    mi_sizes = [5, 10, 20, 50]
    for size in mi_sizes:
        # Generate random joint distribution
        joint = np.random.dirichlet(np.ones(size * size)).reshape(size, size)
        pair = DistributionPair(joint=joint)

        stats = benchmark_function(
            lambda p=pair: mutual_information(p, base=2.0),
            iterations=100
        )
        print_result(size, stats)
        results.append(BenchmarkResult(
            name="mutual_information", input_size=size, iterations=100,
            p50_ms=stats["p50"], p95_ms=stats["p95"],
            p99_ms=stats["p99"], ops_per_sec=stats["ops_per_sec"]
        ))

    return results


# =============================================================================
# Benchmark 2: Spectral Clustering
# =============================================================================

def benchmark_spectral_clustering():
    """Benchmark graph spectral operations."""
    print_header("Benchmark 2: Spectral Clustering")

    # Use smaller sizes since spectral methods are O(n³)
    sizes = [20, 50, 100, 200]
    results = []

    print("\n  --- Graph Laplacian ---")
    for n_nodes in sizes:
        # Create Erdos-Renyi random graph
        G = nx.erdos_renyi_graph(n_nodes, 0.1)

        stats = benchmark_function(
            lambda g=G: compute_laplacian(g, data_type="networkx"),
            iterations=20  # Fewer iterations (expensive)
        )
        print_result(n_nodes, stats)
        results.append(BenchmarkResult(
            name="laplacian", input_size=n_nodes, iterations=20,
            p50_ms=stats["p50"], p95_ms=stats["p95"],
            p99_ms=stats["p99"], ops_per_sec=stats["ops_per_sec"]
        ))

    print("\n  --- Algebraic Connectivity ---")
    for n_nodes in sizes:
        G = nx.erdos_renyi_graph(n_nodes, 0.1)
        L = compute_laplacian(G, data_type="networkx")

        stats = benchmark_function(
            lambda l=L: algebraic_connectivity(l),
            iterations=20
        )
        print_result(n_nodes, stats)
        results.append(BenchmarkResult(
            name="algebraic_connectivity", input_size=n_nodes, iterations=20,
            p50_ms=stats["p50"], p95_ms=stats["p95"],
            p99_ms=stats["p99"], ops_per_sec=stats["ops_per_sec"]
        ))

    print("\n  --- Spectral Clustering (k=3) ---")
    for n_nodes in sizes:
        G = nx.erdos_renyi_graph(n_nodes, 0.2)
        A = nx.to_numpy_array(G)
        clustering = SpectralClustering(n_clusters=3)

        stats = benchmark_function(
            lambda a=A, c=clustering: c.fit(a),
            iterations=10  # Very expensive
        )
        print_result(n_nodes, stats)
        results.append(BenchmarkResult(
            name="spectral_clustering", input_size=n_nodes, iterations=10,
            p50_ms=stats["p50"], p95_ms=stats["p95"],
            p99_ms=stats["p99"], ops_per_sec=stats["ops_per_sec"]
        ))

    return results


# =============================================================================
# Benchmark 3: Game Theory
# =============================================================================

def benchmark_game_theory():
    """Benchmark game theory operations."""
    print_header("Benchmark 3: Game Theory")

    sizes = [2, 3, 5, 10]
    results = []

    print("\n  --- Pure Nash Equilibrium Search ---")
    nash = NashEquilibrium()

    for n in sizes:
        # Random payoff matrix
        payoff = np.random.rand(n, n)
        game = NormalFormGame([payoff, -payoff])  # Zero-sum game

        stats = benchmark_function(
            lambda g=game: nash.find_pure(g),
            iterations=50
        )
        print(f"  {n}x{n} game: p50={stats['p50']:.3f}ms, "
              f"p95={stats['p95']:.3f}ms, ops/s={stats['ops_per_sec']:.0f}")
        results.append(BenchmarkResult(
            name="pure_nash", input_size=n, iterations=50,
            p50_ms=stats["p50"], p95_ms=stats["p95"],
            p99_ms=stats["p99"], ops_per_sec=stats["ops_per_sec"]
        ))

    print("\n  --- Mixed Nash Equilibrium (2-player) ---")
    # Only 2x2 games have closed-form solution
    payoff = np.random.rand(2, 2)
    game = NormalFormGame([payoff, -payoff])

    stats = benchmark_function(
        lambda: nash.find_mixed_2player(game),
        iterations=100
    )
    print(f"  2x2 game: p50={stats['p50']:.3f}ms, "
          f"p95={stats['p95']:.3f}ms, ops/s={stats['ops_per_sec']:.0f}")
    results.append(BenchmarkResult(
        name="mixed_nash_2player", input_size=2, iterations=100,
        p50_ms=stats["p50"], p95_ms=stats["p95"],
        p99_ms=stats["p99"], ops_per_sec=stats["ops_per_sec"]
    ))

    return results


# =============================================================================
# Benchmark 4: Causal Inference
# =============================================================================

def benchmark_causal_inference():
    """Benchmark causal inference operations."""
    print_header("Benchmark 4: Causal Inference (d-separation)")

    sizes = [10, 50, 100, 200]
    results = []

    print("\n  --- D-Separation Test ---")
    for n_nodes in sizes:
        # Build chain graph: X0 -> X1 -> X2 -> ... -> Xn
        cg = CausalGraph()
        for i in range(n_nodes - 1):
            cg.add_edge(f"X{i}", f"X{i+1}")

        # Test d-separation from start to end, conditioning on middle
        x = {f"X0"}
        y = {f"X{n_nodes - 1}"}
        z = {f"X{n_nodes // 2}"}

        stats = benchmark_function(
            lambda cg=cg, x=x, y=y, z=z: cg.d_separated(x, y, z),
            iterations=100
        )
        print_result(n_nodes, stats)
        results.append(BenchmarkResult(
            name="d_separation", input_size=n_nodes, iterations=100,
            p50_ms=stats["p50"], p95_ms=stats["p95"],
            p99_ms=stats["p99"], ops_per_sec=stats["ops_per_sec"]
        ))

    print("\n  --- Causal Graph Construction ---")
    for n_nodes in sizes:
        edges = [(f"X{i}", f"X{i+1}") for i in range(n_nodes - 1)]

        def build_graph(edges=edges):
            cg = CausalGraph()
            for src, dst in edges:
                cg.add_edge(src, dst)
            return cg

        stats = benchmark_function(build_graph, iterations=50)
        print_result(n_nodes, stats)
        results.append(BenchmarkResult(
            name="causal_graph_construction", input_size=n_nodes, iterations=50,
            p50_ms=stats["p50"], p95_ms=stats["p95"],
            p99_ms=stats["p99"], ops_per_sec=stats["ops_per_sec"]
        ))

    return results


# =============================================================================
# Benchmark 5: Regret Minimization
# =============================================================================

def benchmark_regret_minimization():
    """Benchmark regret minimization algorithms."""
    print_header("Benchmark 5: Regret Minimization (Hedge)")

    n_actions_list = [4, 10, 50, 100]
    results = []

    print("\n  --- Hedge Update (per round) ---")
    for n_actions in n_actions_list:
        hedge = Hedge(n_actions, time_horizon=1000)
        losses = np.random.rand(n_actions)

        # Pre-choose to avoid including choice time
        hedge.choose()

        stats = benchmark_function(
            lambda h=hedge, l=losses: h.update(l.copy()),
            iterations=100
        )
        print(f"  Actions {n_actions:4d}: p50={stats['p50']:.3f}ms, "
              f"p95={stats['p95']:.3f}ms, ops/s={stats['ops_per_sec']:.0f}")
        results.append(BenchmarkResult(
            name="hedge_update", input_size=n_actions, iterations=100,
            p50_ms=stats["p50"], p95_ms=stats["p95"],
            p99_ms=stats["p99"], ops_per_sec=stats["ops_per_sec"]
        ))

    print("\n  --- Hedge Full Round (choose + update) ---")
    for n_actions in n_actions_list:
        hedge = Hedge(n_actions, time_horizon=1000)

        def full_round():
            action = hedge.choose()
            losses = np.random.rand(n_actions)
            hedge.update(losses)
            return action

        stats = benchmark_function(full_round, iterations=100)
        print(f"  Actions {n_actions:4d}: p50={stats['p50']:.3f}ms, "
              f"p95={stats['p95']:.3f}ms, ops/s={stats['ops_per_sec']:.0f}")
        results.append(BenchmarkResult(
            name="hedge_full_round", input_size=n_actions, iterations=100,
            p50_ms=stats["p50"], p95_ms=stats["p95"],
            p99_ms=stats["p99"], ops_per_sec=stats["ops_per_sec"]
        ))

    return results


# =============================================================================
# Summary
# =============================================================================

def print_summary(all_results: List[BenchmarkResult]):
    """Print summary table of all benchmarks."""
    print_header("SUMMARY")

    print("\n  | Module                 | Typical p50  | Scaling     |")
    print("  |------------------------|--------------|-------------|")

    # Group by module and compute typical p50
    modules = {
        "Information Theory": ["entropy", "mutual_information"],
        "Spectral Clustering": ["laplacian", "algebraic_connectivity", "spectral_clustering"],
        "Game Theory": ["pure_nash", "mixed_nash_2player"],
        "Causal Inference": ["d_separation", "causal_graph_construction"],
        "Regret Minimization": ["hedge_update", "hedge_full_round"]
    }

    scalings = {
        "Information Theory": "O(n)",
        "Spectral Clustering": "O(n³)",
        "Game Theory": "O(n²)",
        "Causal Inference": "O(n+e)",
        "Regret Minimization": "O(n)"
    }

    for module, ops in modules.items():
        # Find results for this module
        module_results = [r for r in all_results if r.name in ops]
        if module_results:
            # Use median input size result
            mid_idx = len(module_results) // 2
            typical_p50 = module_results[mid_idx].p50_ms
            scaling = scalings.get(module, "O(?)")
            print(f"  | {module:22s} | {typical_p50:>10.2f}ms | {scaling:11s} |")

    print()


def main():
    """Run all benchmarks."""
    print()
    print("=" * 60)
    print("  HOLOLOOM MATH MODULES BENCHMARK")
    print("=" * 60)

    start_time = time.perf_counter()

    all_results = []

    # Run benchmarks
    all_results.extend(benchmark_information_theory())
    all_results.extend(benchmark_spectral_clustering())
    all_results.extend(benchmark_game_theory())
    all_results.extend(benchmark_causal_inference())
    all_results.extend(benchmark_regret_minimization())

    # Print summary
    print_summary(all_results)

    total_time = (time.perf_counter() - start_time) * 1000

    print(f"  Total benchmark time: {total_time:.1f}ms")
    print()


if __name__ == "__main__":
    main()
