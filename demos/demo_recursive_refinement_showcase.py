#!/usr/bin/env python3
"""
HoloLoom Recursive Refinement Showcase
========================================
Demonstrates multi-strategy self-improvement where initial responses are
iteratively refined through quality-focused passes.

Philosophy: "Great answers aren't written, they're refined."

Strategies:
  VERIFY    - Accuracy -> Completeness -> Consistency  (3-pass)
  ELEGANCE  - Clarity  -> Simplicity   -> Beauty       (3-pass)
  CRITIQUE  - Self-critical analysis                    (1-pass)
  HOFSTADTER - Recursive self-reference                 (iterative)
  AUTO      - Learns which strategy fits each query     (adaptive)

Run:
    python demos/demo_recursive_refinement_showcase.py

No heavy dependencies required (numpy only).
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

class RefinementStrategy(Enum):
    VERIFY = "verify"
    ELEGANCE = "elegance"
    CRITIQUE = "critique"
    HOFSTADTER = "hofstadter"
    AUTO = "auto"


@dataclass
class QualityMetrics:
    """Composite quality measurement for a single refinement pass."""

    confidence: float         # 0-1 core accuracy
    context_richness: float   # 0-1 how much context was used
    response_completeness: float  # 0-1 how thorough the answer is
    coherence: float = 0.0    # 0-1 logical consistency
    clarity: float = 0.0      # 0-1 readability

    def score(self) -> float:
        """Weighted composite: accuracy dominates, context and completeness support."""
        return (
            0.40 * self.confidence
            + 0.20 * self.context_richness
            + 0.15 * self.response_completeness
            + 0.15 * self.coherence
            + 0.10 * self.clarity
        )

    def summary_line(self) -> str:
        return (
            f"conf={self.confidence:.3f}  ctx={self.context_richness:.3f}  "
            f"comp={self.response_completeness:.3f}  coh={self.coherence:.3f}  "
            f"clr={self.clarity:.3f}  => {self.score():.4f}"
        )


@dataclass
class RefinementResult:
    strategy: RefinementStrategy
    trajectory: List[QualityMetrics]
    iterations: int
    improved: bool
    stop_reason: str

    @property
    def initial_score(self) -> float:
        return self.trajectory[0].score() if self.trajectory else 0.0

    @property
    def final_score(self) -> float:
        return self.trajectory[-1].score() if self.trajectory else 0.0

    @property
    def improvement(self) -> float:
        if self.initial_score == 0:
            return 0.0
        return (self.final_score - self.initial_score) / self.initial_score


# ---------------------------------------------------------------------------
# Refinement engine
# ---------------------------------------------------------------------------

class RefinementEngine:
    """Simulates multi-pass refinement with strategy-specific improvement curves."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _simulate_pass(
        self,
        prev: QualityMetrics,
        strategy: RefinementStrategy,
        pass_num: int,
        pass_focus: str,
    ) -> QualityMetrics:
        """Simulate one refinement pass with strategy-specific dynamics."""

        # Each strategy improves different signals at different rates
        improvement_profiles = {
            RefinementStrategy.VERIFY: {
                0: {"confidence": 0.15, "coherence": 0.08},      # Accuracy check
                1: {"response_completeness": 0.12, "context_richness": 0.10},  # Completeness
                2: {"coherence": 0.15, "confidence": 0.05},      # Consistency
            },
            RefinementStrategy.ELEGANCE: {
                0: {"clarity": 0.20, "coherence": 0.05},         # Clarity pass
                1: {"clarity": 0.10, "response_completeness": -0.05},  # Simplicity (may trim)
                2: {"clarity": 0.08, "coherence": 0.12, "confidence": 0.03},  # Beauty
            },
            RefinementStrategy.CRITIQUE: {
                0: {"confidence": 0.10, "coherence": 0.12, "context_richness": 0.08},
            },
            RefinementStrategy.HOFSTADTER: {
                0: {"confidence": 0.08, "coherence": 0.06, "clarity": 0.04},
                1: {"confidence": 0.06, "coherence": 0.10, "clarity": 0.06},
                2: {"confidence": 0.04, "coherence": 0.08, "clarity": 0.08},
                3: {"confidence": 0.03, "coherence": 0.06, "clarity": 0.10},
            },
        }

        profile = improvement_profiles.get(strategy, {})
        deltas = profile.get(pass_num, profile.get(0, {}))

        # Apply improvements with diminishing returns and noise
        noise = self.rng.normal(0, 0.02)

        new_conf = min(1.0, max(0.0, prev.confidence + deltas.get("confidence", 0) + noise))
        new_ctx = min(1.0, max(0.0, prev.context_richness + deltas.get("context_richness", 0) + noise))
        new_comp = min(1.0, max(0.0, prev.response_completeness + deltas.get("response_completeness", 0) + noise))
        new_coh = min(1.0, max(0.0, prev.coherence + deltas.get("coherence", 0) + noise))
        new_clr = min(1.0, max(0.0, prev.clarity + deltas.get("clarity", 0) + noise))

        return QualityMetrics(
            confidence=new_conf,
            context_richness=new_ctx,
            response_completeness=new_comp,
            coherence=new_coh,
            clarity=new_clr,
        )

    def refine(
        self,
        initial: QualityMetrics,
        strategy: RefinementStrategy,
        max_iterations: int = 3,
        quality_threshold: float = 0.85,
    ) -> RefinementResult:
        """Run multi-pass refinement with a given strategy."""

        pass_labels = {
            RefinementStrategy.VERIFY: ["Accuracy", "Completeness", "Consistency"],
            RefinementStrategy.ELEGANCE: ["Clarity", "Simplicity", "Beauty"],
            RefinementStrategy.CRITIQUE: ["Self-critique"],
            RefinementStrategy.HOFSTADTER: ["Meta-1", "Meta-2", "Meta-3", "Meta-4"],
        }

        labels = pass_labels.get(strategy, ["Pass"] * max_iterations)
        trajectory = [initial]
        current = initial
        stop_reason = "max_iterations"

        for i in range(min(max_iterations, len(labels))):
            focus = labels[i] if i < len(labels) else f"Pass-{i+1}"
            new = self._simulate_pass(current, strategy, i, focus)
            trajectory.append(new)

            if new.score() >= quality_threshold:
                stop_reason = f"threshold_reached ({quality_threshold:.2f})"
                current = new
                break

            # Convergence detection
            if abs(new.score() - current.score()) < 0.005:
                stop_reason = "converged"
                current = new
                break

            current = new

        return RefinementResult(
            strategy=strategy,
            trajectory=trajectory,
            iterations=len(trajectory) - 1,
            improved=trajectory[-1].score() > trajectory[0].score(),
            stop_reason=stop_reason,
        )


# ---------------------------------------------------------------------------
# Demo 1: VERIFY Strategy - Truth-Seeking
# ---------------------------------------------------------------------------

def demo_verify_strategy(engine: RefinementEngine):
    """
    VERIFY: Three-pass truth-seeking refinement.
      Pass 1: Check factual accuracy
      Pass 2: Fill gaps in completeness
      Pass 3: Ensure internal consistency
    """
    print("=" * 64)
    print("Demo 1: VERIFY Strategy - Truth-Seeking")
    print("=" * 64)
    print()
    print("  Three passes focused on correctness:")
    print("    Pass 1: Accuracy   - Are the facts right?")
    print("    Pass 2: Completeness - Is anything missing?")
    print("    Pass 3: Consistency - Do all parts agree?")
    print()

    initial = QualityMetrics(
        confidence=0.55, context_richness=0.40,
        response_completeness=0.35, coherence=0.50, clarity=0.60,
    )

    result = engine.refine(initial, RefinementStrategy.VERIFY, max_iterations=3)
    _display_trajectory(result, ["Initial", "Accuracy", "Completeness", "Consistency"])


# ---------------------------------------------------------------------------
# Demo 2: ELEGANCE Strategy - Clarity Pursuit
# ---------------------------------------------------------------------------

def demo_elegance_strategy(engine: RefinementEngine):
    """
    ELEGANCE: Three-pass clarity refinement.
      Pass 1: Improve readability and clarity
      Pass 2: Simplify (may reduce length)
      Pass 3: Polish for beauty and flow
    """
    print("=" * 64)
    print("Demo 2: ELEGANCE Strategy - Clarity Pursuit")
    print("=" * 64)
    print()
    print("  Three passes focused on communication:")
    print("    Pass 1: Clarity    - Is it easy to understand?")
    print("    Pass 2: Simplicity - Can anything be removed?")
    print("    Pass 3: Beauty     - Does it flow naturally?")
    print()

    initial = QualityMetrics(
        confidence=0.70, context_richness=0.60,
        response_completeness=0.65, coherence=0.45, clarity=0.35,
    )

    result = engine.refine(initial, RefinementStrategy.ELEGANCE, max_iterations=3)
    _display_trajectory(result, ["Initial", "Clarity", "Simplicity", "Beauty"])


# ---------------------------------------------------------------------------
# Demo 3: Strategy Comparison
# ---------------------------------------------------------------------------

def demo_strategy_comparison(engine: RefinementEngine):
    """
    Compare all four strategies on the same initial response.
    Shows how each strategy targets different quality dimensions.
    """
    print("=" * 64)
    print("Demo 3: Strategy Comparison")
    print("=" * 64)
    print()

    initial = QualityMetrics(
        confidence=0.50, context_richness=0.45,
        response_completeness=0.40, coherence=0.42, clarity=0.38,
    )

    print(f"  Same starting point: score = {initial.score():.4f}")
    print(f"    {initial.summary_line()}")
    print()

    strategies = [
        RefinementStrategy.VERIFY,
        RefinementStrategy.ELEGANCE,
        RefinementStrategy.CRITIQUE,
        RefinementStrategy.HOFSTADTER,
    ]

    results: Dict[str, RefinementResult] = {}

    print(f"  {'Strategy':<14} {'Initial':>8} {'Final':>8} {'Change':>8} {'Iters':>6}  Stop Reason")
    print(f"  {'─' * 14} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 6}  {'─' * 25}")

    for strategy in strategies:
        result = engine.refine(initial, strategy, max_iterations=4)
        results[strategy.value] = result
        change = result.final_score - result.initial_score
        sign = "+" if change >= 0 else ""
        print(
            f"  {strategy.value:<14} {result.initial_score:>8.4f} {result.final_score:>8.4f} "
            f"{sign}{change:>7.4f} {result.iterations:>6}  {result.stop_reason}"
        )

    # Show dimension improvements per strategy
    print()
    print("  Per-dimension improvement (final - initial):")
    print()
    print(f"  {'Strategy':<14} {'Conf':>7} {'Context':>7} {'Compl':>7} {'Coher':>7} {'Clarity':>7}")
    print(f"  {'─' * 14} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 7}")

    for strategy in strategies:
        result = results[strategy.value]
        init = result.trajectory[0]
        final = result.trajectory[-1]
        print(
            f"  {strategy.value:<14} "
            f"{final.confidence - init.confidence:>+7.3f} "
            f"{final.context_richness - init.context_richness:>+7.3f} "
            f"{final.response_completeness - init.response_completeness:>+7.3f} "
            f"{final.coherence - init.coherence:>+7.3f} "
            f"{final.clarity - init.clarity:>+7.3f}"
        )

    print()

    # Visual comparison
    print("  Quality improvement visualised:")
    print()
    best_final = max(r.final_score for r in results.values())
    for strategy in strategies:
        result = results[strategy.value]
        init_bar = int(result.initial_score * 40)
        final_bar = int(result.final_score * 40)
        gained = final_bar - init_bar

        bar = "░" * init_bar + "█" * gained + "·" * (40 - final_bar)
        marker = " <-- best" if result.final_score == best_final else ""
        print(f"    {strategy.value:<12} {bar} {result.final_score:.3f}{marker}")

    print()
    print("    Legend: ░ = initial quality, █ = improvement, · = headroom")
    print()


# ---------------------------------------------------------------------------
# Demo 4: Convergence and Early Stopping
# ---------------------------------------------------------------------------

def demo_convergence(engine: RefinementEngine):
    """
    Shows how refinement converges — quality gains diminish with each
    pass, and the system stops when improvement plateaus.
    """
    print("=" * 64)
    print("Demo 4: Convergence and Early Stopping")
    print("=" * 64)
    print()
    print("  Refinement follows diminishing returns:")
    print("  each pass yields smaller improvements until convergence.")
    print()

    # Run with many iterations to see convergence
    initial = QualityMetrics(
        confidence=0.45, context_richness=0.40,
        response_completeness=0.35, coherence=0.40, clarity=0.30,
    )

    # Manual multi-pass to show convergence curve
    trajectory = [initial]
    current = initial
    for i in range(8):
        # Diminishing improvements
        decay = 0.6 ** i
        noise = engine.rng.normal(0, 0.01)
        improvement = 0.08 * decay + noise
        current = QualityMetrics(
            confidence=min(1.0, current.confidence + improvement * 0.4),
            context_richness=min(1.0, current.context_richness + improvement * 0.2),
            response_completeness=min(1.0, current.response_completeness + improvement * 0.15),
            coherence=min(1.0, current.coherence + improvement * 0.15),
            clarity=min(1.0, current.clarity + improvement * 0.1),
        )
        trajectory.append(current)

    print(f"  {'Pass':<6} {'Score':>8} {'Delta':>8} {'Visualisation'}")
    print(f"  {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 40}")

    for i, metrics in enumerate(trajectory):
        score = metrics.score()
        delta = score - trajectory[i - 1].score() if i > 0 else 0.0
        bar_len = int(score * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        label = "initial" if i == 0 else f"pass {i}"
        converged = " <- converged" if i > 0 and abs(delta) < 0.005 else ""
        print(f"  {label:<6} {score:>8.4f} {delta:>+8.4f} {bar}{converged}")

    print()
    print("  Key insight: most improvement happens in the first 2-3 passes.")
    print("  Continuing beyond convergence wastes compute without quality gain.")
    print()


# ---------------------------------------------------------------------------
# Demo 5: AUTO Strategy Selection
# ---------------------------------------------------------------------------

def demo_auto_strategy(engine: RefinementEngine):
    """
    AUTO mode learns which strategy works best for different query types.
    Uses Thompson Sampling to balance exploration of strategies with
    exploitation of known-good ones.
    """
    print("=" * 64)
    print("Demo 5: AUTO Strategy Selection (Thompson Sampling)")
    print("=" * 64)
    print()
    print("  AUTO learns optimal strategy per query type using")
    print("  Thompson Sampling over Beta(alpha, beta) priors.")
    print()

    strategies = [
        RefinementStrategy.VERIFY,
        RefinementStrategy.ELEGANCE,
        RefinementStrategy.CRITIQUE,
        RefinementStrategy.HOFSTADTER,
    ]

    # Thompson Sampling priors per (query_type, strategy)
    query_types = ["factual", "analytical", "creative", "technical"]
    # Ground truth: which strategy actually works best per query type
    ground_truth = {
        "factual": RefinementStrategy.VERIFY,
        "analytical": RefinementStrategy.CRITIQUE,
        "creative": RefinementStrategy.ELEGANCE,
        "technical": RefinementStrategy.HOFSTADTER,
    }

    alpha = {(qt, s): 1.0 for qt in query_types for s in strategies}
    beta = {(qt, s): 1.0 for qt in query_types for s in strategies}

    n_rounds = 60
    correct_selections = 0

    print(f"  Running {n_rounds} rounds of adaptive strategy selection...")
    print()

    rng = np.random.default_rng(42)

    for round_num in range(n_rounds):
        qt = query_types[round_num % len(query_types)]

        # Thompson sample: pick strategy with highest sampled value
        samples = {}
        for s in strategies:
            samples[s] = rng.beta(alpha[(qt, s)], beta[(qt, s)])

        chosen = max(samples, key=lambda s: samples[s])
        is_optimal = chosen == ground_truth[qt]

        # Simulate outcome: optimal strategy succeeds 80%, others 40%
        success_prob = 0.80 if is_optimal else 0.40
        success = rng.random() < success_prob

        # Update priors
        if success:
            alpha[(qt, chosen)] += 1.0
            correct_selections += 1 if is_optimal else 0
        else:
            beta[(qt, chosen)] += 1.0

    # Display learned preferences
    print(f"  Learned strategy preferences after {n_rounds} rounds:")
    print()
    print(f"  {'Query Type':<14}", end="")
    for s in strategies:
        print(f" {s.value:>12}", end="")
    print("     Best")
    print(f"  {'─' * 14}", end="")
    for _ in strategies:
        print(f" {'─' * 12}", end="")
    print(f"  {'─' * 12}")

    for qt in query_types:
        print(f"  {qt:<14}", end="")
        best_expected = 0
        best_strat = None
        for s in strategies:
            a, b = alpha[(qt, s)], beta[(qt, s)]
            expected = a / (a + b)
            if expected > best_expected:
                best_expected = expected
                best_strat = s
            print(f" {expected:>12.3f}", end="")

        correct = best_strat == ground_truth[qt]
        marker = "correct" if correct else "learning..."
        print(f"  {best_strat.value if best_strat else '?':<10} ({marker})")

    print()

    # Show convergence of one query type
    qt_show = "factual"
    print(f"  Expected reward trajectory for '{qt_show}' queries:")
    print()
    for s in strategies:
        a, b = alpha[(qt_show, s)], beta[(qt_show, s)]
        expected = a / (a + b)
        bar = "█" * int(expected * 40) + "░" * (40 - int(expected * 40))
        optimal = " <-- ground truth" if s == ground_truth[qt_show] else ""
        print(f"    {s.value:<12} E[r]={expected:.3f}  {bar}{optimal}")

    print()


# ---------------------------------------------------------------------------
# Demo 6: Quality Trajectory Landscape
# ---------------------------------------------------------------------------

def demo_quality_landscape(engine: RefinementEngine):
    """
    Visualise the 5-dimensional quality space as a parallel coordinates
    display, showing how each strategy navigates the landscape.
    """
    print("=" * 64)
    print("Demo 6: Quality Trajectory Landscape")
    print("=" * 64)
    print()
    print("  5 quality dimensions form the refinement landscape.")
    print("  Each strategy charts a different path through this space.")
    print()

    initial = QualityMetrics(
        confidence=0.50, context_richness=0.45,
        response_completeness=0.40, coherence=0.42, clarity=0.38,
    )

    strategies = [
        RefinementStrategy.VERIFY,
        RefinementStrategy.ELEGANCE,
    ]

    dimension_names = ["Confidence", "Context", "Completeness", "Coherence", "Clarity"]

    for strategy in strategies:
        result = engine.refine(initial, strategy, max_iterations=3)

        print(f"  Strategy: {strategy.value.upper()}")
        print(f"  {'Pass':<14}", end="")
        for dim in dimension_names:
            print(f" {dim:>12}", end="")
        print(f" {'Score':>8}")
        print(f"  {'─' * 14}", end="")
        for _ in dimension_names:
            print(f" {'─' * 12}", end="")
        print(f" {'─' * 8}")

        labels = {
            RefinementStrategy.VERIFY: ["Initial", "Accuracy", "Complete", "Consist"],
            RefinementStrategy.ELEGANCE: ["Initial", "Clarity", "Simplify", "Beauty"],
        }

        step_labels = labels.get(strategy, ["Initial"] + [f"Pass {i}" for i in range(1, 5)])

        for i, metrics in enumerate(result.trajectory):
            label = step_labels[i] if i < len(step_labels) else f"Pass {i}"
            vals = [metrics.confidence, metrics.context_richness,
                    metrics.response_completeness, metrics.coherence, metrics.clarity]
            print(f"  {label:<14}", end="")
            for v in vals:
                bar = "▓" * int(v * 10) + "░" * (10 - int(v * 10))
                print(f" {bar} {v:.2f}", end="")
            print(f" {metrics.score():>8.4f}")

        print()

    # Final comparison
    print("  Radar summary (final state):")
    print()
    for strategy in strategies:
        result = engine.refine(initial, strategy, max_iterations=3)
        final = result.trajectory[-1]
        vals = [final.confidence, final.context_richness,
                final.response_completeness, final.coherence, final.clarity]
        print(f"    {strategy.value.upper():<12}", end="")
        for dim, v in zip(dimension_names, vals):
            bar = "█" * int(v * 15)
            print(f"  {dim[:4]}:{bar}", end="")
        print(f"  total={final.score():.3f}")
    print()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _display_trajectory(result: RefinementResult, step_labels: List[str]):
    """Pretty-print a refinement trajectory."""

    print(f"  Trajectory ({result.iterations} passes, stop: {result.stop_reason}):")
    print()
    print(f"  {'Step':<14} {'Conf':>6} {'Ctx':>6} {'Comp':>6} {'Coh':>6} {'Clr':>6}  {'Score':>7}  Visual")
    print(f"  {'─' * 14} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 6}  {'─' * 7}  {'─' * 25}")

    for i, metrics in enumerate(result.trajectory):
        label = step_labels[i] if i < len(step_labels) else f"Pass {i}"
        score = metrics.score()
        bar_len = int(score * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(
            f"  {label:<14} {metrics.confidence:>6.3f} {metrics.context_richness:>6.3f} "
            f"{metrics.response_completeness:>6.3f} {metrics.coherence:>6.3f} "
            f"{metrics.clarity:>6.3f}  {score:>7.4f}  {bar}"
        )

    change = result.final_score - result.initial_score
    pct = result.improvement * 100
    print()
    print(f"  Improvement: {result.initial_score:.4f} -> {result.final_score:.4f} ({pct:+.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("  ╔════════════════════════════════════════════════════════════╗")
    print("  ║       HoloLoom Recursive Refinement Showcase              ║")
    print("  ║       \"Great answers aren't written, they're refined.\"    ║")
    print("  ╚════════════════════════════════════════════════════════════╝")
    print()

    engine = RefinementEngine(seed=42)

    demo_verify_strategy(engine)
    demo_elegance_strategy(engine)
    demo_strategy_comparison(engine)
    demo_convergence(engine)
    demo_auto_strategy(engine)
    demo_quality_landscape(engine)

    print("  " + "═" * 62)
    print("  All demonstrations complete.")
    print()


if __name__ == "__main__":
    main()
