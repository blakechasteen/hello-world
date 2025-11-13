"""
Resonance Shed Performance Benchmark
=====================================

Benchmarks ResonanceShed performance across all fusion modes.

Performance Targets:
- BARE mode (minimal): <50ms
- FAST mode (hybrid): <100ms
- FUSED mode (all features): <150ms

Fusion Modes Tested:
- weighted_sum (default)
- attention (cross-attention)
- concat (concatenation)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
import numpy as np
from typing import List
from dataclasses import dataclass

# Mock components for benchmarking
from HoloLoom.documentation.types import Motif


class MockMotifDetector:
    """Lightweight mock motif detector"""
    async def detect(self, text: str) -> List[Motif]:
        # Simulate fast pattern matching
        patterns = []
        if "thompson" in text.lower():
            patterns.append(Motif("THOMPSON", (0, 8), 0.9))
        if "sampling" in text.lower():
            patterns.append(Motif("SAMPLING", (0, 8), 0.85))
        return patterns


class MockEmbedder:
    """Lightweight mock embedder"""
    def __init__(self, dim=384):
        self.dim = dim
        self._cache = {}

    def encode(self, texts: List[str]) -> np.ndarray:
        # Fast deterministic embeddings
        vecs = []
        for text in texts:
            if text in self._cache:
                vecs.append(self._cache[text])
            else:
                seed = abs(hash(text)) % (2**32)
                rng = np.random.default_rng(seed)
                vec = rng.normal(0, 1, self.dim)
                vec = vec / (np.linalg.norm(vec) + 1e-9)
                self._cache[text] = vec
                vecs.append(vec)
        return np.vstack(vecs)

    def encode_scales(self, texts: List[str], size: int):
        base = self.encode(texts)
        return base[:, :size]


class MockSpectralFusion:
    """Lightweight mock spectral fusion"""
    async def features(self, kg_sub, shard_texts, emb):
        # Fast spectral features
        spectral = np.array([0.1, 0.2, 0.3, 0.4])
        metrics = {"fiedler": 0.2, "topic_var": 0.6, "coherence": 0.8}
        return spectral, metrics


class MockSemanticCalculus:
    """Lightweight mock semantic calculus"""
    def extract_features(self, text: str):
        n_words = len(text.split())
        return {
            "n_states": n_words,
            "avg_velocity": 0.5,
            "avg_acceleration": 0.1,
            "total_distance": float(n_words * 0.5),
            "curvature": [0.1] * n_words
        }


@dataclass
class BenchmarkResult:
    """Performance benchmark result"""
    mode: str
    fusion: str
    duration_ms: float
    thread_count: int
    psi_dim: int
    iterations: int


async def benchmark_resonance_shed():
    """Run comprehensive performance benchmarks"""
    print("=" * 80)
    print("Resonance Shed Performance Benchmark")
    print("=" * 80)
    print()

    from HoloLoom.resonance.shed import ResonanceShed

    # Test configurations
    configs = [
        # BARE mode (minimal)
        {
            "name": "BARE",
            "motif_detector": None,
            "embedder": MockEmbedder(96),  # Small embedding
            "spectral_fusion": None,
            "semantic_calculus": None,
            "target": 50.0  # ms
        },
        # FAST mode (hybrid)
        {
            "name": "FAST",
            "motif_detector": MockMotifDetector(),
            "embedder": MockEmbedder(192),  # Medium embedding
            "spectral_fusion": None,
            "semantic_calculus": None,
            "target": 100.0  # ms
        },
        # FUSED mode (all features)
        {
            "name": "FUSED",
            "motif_detector": MockMotifDetector(),
            "embedder": MockEmbedder(384),  # Large embedding
            "spectral_fusion": MockSpectralFusion(),
            "semantic_calculus": MockSemanticCalculus(),
            "target": 150.0  # ms
        }
    ]

    fusion_modes = ["weighted_sum", "attention", "concat"]

    test_texts = [
        "Thompson Sampling balances exploration and exploitation",
        "What is the optimal policy for multi-armed bandits?",
        "Explain the difference between UCB and Thompson Sampling",
        "How does Bayesian optimization work in practice?",
        "Describe the exploration-exploitation tradeoff"
    ]

    results = []

    for config in configs:
        mode_name = config["name"]
        target_ms = config["target"]

        print(f"\n{mode_name} Mode (target: <{target_ms}ms)")
        print("-" * 80)

        for fusion_mode in fusion_modes:
            # Create shed
            shed = ResonanceShed(
                motif_detector=config["motif_detector"],
                embedder=config["embedder"],
                spectral_fusion=config["spectral_fusion"],
                semantic_calculus=config["semantic_calculus"],
                interference_mode=fusion_mode,
                guardrails=None  # Disable for benchmarking
            )

            # Warm-up (1 iteration)
            await shed.weave(test_texts[0])

            # Benchmark (multiple iterations)
            iterations = 10
            durations = []

            for text in test_texts:
                start = time.perf_counter()
                plasma = await shed.weave(text)
                duration = (time.perf_counter() - start) * 1000  # ms
                durations.append(duration)

            # Statistics
            avg_duration = np.mean(durations)
            std_duration = np.std(durations)
            min_duration = np.min(durations)
            max_duration = np.max(durations)

            # Get dimensions
            final_plasma = await shed.weave(test_texts[0])
            thread_count = len(final_plasma.get("threads", []))
            psi_dim = len(final_plasma.get("psi", []))

            # Record result
            result = BenchmarkResult(
                mode=mode_name,
                fusion=fusion_mode,
                duration_ms=avg_duration,
                thread_count=thread_count,
                psi_dim=psi_dim,
                iterations=iterations
            )
            results.append(result)

            # Check target
            status = "PASS" if avg_duration < target_ms else "FAIL"

            # Print results
            print(f"  {fusion_mode:15s}: {avg_duration:6.2f}ms +- {std_duration:5.2f}ms "
                  f"(min={min_duration:5.2f}ms, max={max_duration:5.2f}ms) "
                  f"[threads={thread_count}, dim={psi_dim}] {status}")

    # Summary table
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print()
    print(f"{'Mode':<10} {'Fusion':<15} {'Avg (ms)':<12} {'Threads':<10} {'Dim':<8} {'Target':<10} {'Pass'}")
    print("-" * 80)

    targets = {"BARE": 50.0, "FAST": 100.0, "FUSED": 150.0}

    for result in results:
        target = targets[result.mode]
        status = "PASS" if result.duration_ms < target else "FAIL"
        print(f"{result.mode:<10} {result.fusion:<15} "
              f"{result.duration_ms:<12.2f} "
              f"{result.thread_count:<10} {result.psi_dim:<8} "
              f"<{target:.0f}ms{'':<5} {status}")

    print()

    # Overall statistics
    all_durations = [r.duration_ms for r in results]
    print(f"Overall Average: {np.mean(all_durations):.2f}ms")
    print(f"Overall Std Dev: {np.std(all_durations):.2f}ms")
    print(f"Fastest: {np.min(all_durations):.2f}ms ({min((r for r in results), key=lambda r: r.duration_ms).mode} / {min((r for r in results), key=lambda r: r.duration_ms).fusion})")
    print(f"Slowest: {np.max(all_durations):.2f}ms ({max((r for r in results), key=lambda r: r.duration_ms).mode} / {max((r for r in results), key=lambda r: r.duration_ms).fusion})")

    # Pass/fail count
    passed = sum(1 for r in results if r.duration_ms < targets[r.mode])
    total = len(results)
    print(f"\nPassed: {passed}/{total} ({100*passed/total:.1f}%)")

    print("\nPASS Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(benchmark_resonance_shed())
