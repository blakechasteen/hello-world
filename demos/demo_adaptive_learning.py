"""
Demo: Adaptive Learning for Context Packing - Phase 6.4
========================================================

Demonstrates Thompson Sampling for optimal MI budget allocation.

Features:
- Thompson Sampler basics (Beta distribution posteriors)
- AdaptiveBudgetLearner for multi-complexity learning
- Learning from query outcomes
- Budget recommendations with reasoning
- State persistence (save/load)
- Convenience functions for global usage

Author: Claude Code
Date: 2025-12-09
"""

import asyncio
import tempfile
import os
from typing import List, Tuple

# Phase 6.4 imports
from hololoom.context_packing.learning import (
    QueryComplexity,
    BudgetOutcome,
    BudgetRecommendation,
    ThompsonSampler,
    AdaptiveBudgetLearner,
    get_learner,
    get_adaptive_budget,
    record_outcome,
    get_learning_statistics,
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


# ============================================================================
# Demo 1: Thompson Sampler Basics
# ============================================================================

def demo_thompson_sampler():
    """Demonstrate basic Thompson Sampler usage."""
    print_header("Demo 1: Thompson Sampler Basics")

    print("""
Thompson Sampling uses a Beta distribution to model uncertainty about
success rates. It balances exploration (trying uncertain options) with
exploitation (using what works).

Beta(alpha, beta) where:
- alpha = "pseudo-successes" (prior + observed)
- beta = "pseudo-failures" (prior + observed)
- Expected value = alpha / (alpha + beta)
""")

    # Create sampler with base budget of 5.0 bits
    sampler = ThompsonSampler(base_budget=5.0)

    print_subheader("Initial State (Uniform Prior)")
    stats = sampler.get_statistics()
    print(f"  Alpha: {stats['alpha']:.2f}")
    print(f"  Beta: {stats['beta']:.2f}")
    print(f"  Expected Quality: {stats['expected_quality']:.2%}")
    print(f"  Confidence: {stats['confidence']:.2%}")
    print(f"  Base Budget: {stats['base_budget']:.1f} bits")

    # Sample a few budgets
    print_subheader("Sampling Budgets (exploration)")
    samples = [sampler.sample() for _ in range(5)]
    print(f"  Samples: {[f'{s:.2f}' for s in samples]}")
    print(f"  Range shows exploration around base budget")

    # Simulate successful outcomes
    print_subheader("Learning from Successful Outcomes")
    for i in range(5):
        sampler.update(BudgetOutcome(
            complexity=QueryComplexity.MODERATE,
            budget_used=5.0,
            quality_score=0.85 + i * 0.02,  # High quality
            confidence=0.9,
        ))

    stats = sampler.get_statistics()
    print(f"  After 5 successes:")
    print(f"    Alpha: {stats['alpha']:.2f} (increased)")
    print(f"    Beta: {stats['beta']:.2f} (unchanged)")
    print(f"    Expected Quality: {stats['expected_quality']:.2%} (higher)")
    print(f"    Success Rate: {stats['success_rate']:.0%}")

    # Simulate some failures
    print_subheader("Learning from Failed Outcomes")
    for _ in range(3):
        sampler.update(BudgetOutcome(
            complexity=QueryComplexity.MODERATE,
            budget_used=5.0,
            quality_score=0.5,  # Below threshold (0.7)
            confidence=0.6,
        ))

    stats = sampler.get_statistics()
    print(f"  After 3 failures:")
    print(f"    Alpha: {stats['alpha']:.2f}")
    print(f"    Beta: {stats['beta']:.2f} (increased)")
    print(f"    Expected Quality: {stats['expected_quality']:.2%} (lower)")
    print(f"    Success Rate: {stats['success_rate']:.0%}")
    print(f"    Total Queries: {stats['total_queries']}")


# ============================================================================
# Demo 2: Adaptive Budget Learner
# ============================================================================

def demo_adaptive_learner():
    """Demonstrate AdaptiveBudgetLearner across complexity levels."""
    print_header("Demo 2: Adaptive Budget Learner")

    print("""
The AdaptiveBudgetLearner maintains separate Thompson Samplers for each
query complexity level, learning optimal MI budgets independently.

Default Budgets:
  TRIVIAL:  2.0 bits (greetings, simple confirmations)
  SIMPLE:   3.0 bits (factual lookups)
  MODERATE: 5.0 bits (explanations)
  COMPLEX:  8.0 bits (comparisons, analysis)
  RESEARCH: 15.0 bits (comprehensive exploration)
""")

    learner = AdaptiveBudgetLearner()

    print_subheader("Initial Budget Recommendations")
    for complexity in QueryComplexity:
        budget = learner.get_budget(complexity)
        print(f"  {complexity.value:10s}: {budget:.2f} bits")

    # Simulate training for MODERATE complexity
    print_subheader("Training MODERATE Complexity (20 queries)")
    for i in range(20):
        budget = learner.get_budget(QueryComplexity.MODERATE)
        # Simulate: higher budgets tend to give better quality
        quality = min(1.0, 0.6 + (budget / 20.0))
        learner.update(
            complexity=QueryComplexity.MODERATE,
            budget_used=budget,
            quality_score=quality,
            confidence=0.85,
        )

    # Get detailed recommendation
    rec = learner.get_recommendation(QueryComplexity.MODERATE)
    print(f"\n  Recommendation for MODERATE:")
    print(f"    Budget: {rec.recommended_budget:.2f} bits")
    print(f"    Expected Quality: {rec.expected_quality:.2%}")
    print(f"    Confidence: {rec.confidence:.2%}")
    print(f"    Reasoning: {rec.reasoning}")

    print_subheader("Alternative Budgets")
    print("    Budget    Expected Quality")
    for budget, quality in rec.alternatives[:5]:
        print(f"    {budget:5.1f}     {quality:.2%}")

    # Show statistics
    print_subheader("Learning Statistics")
    stats = learner.get_statistics()
    print(f"  Total Updates: {stats['total_updates']}")
    print(f"  Recent Avg Quality: {stats['recent_avg_quality']:.2%}")
    print(f"  Recent Success Rate: {stats['recent_success_rate']:.0%}")

    print("\n  By Complexity Level:")
    for complexity, comp_stats in stats['by_complexity'].items():
        if comp_stats['total_queries'] > 0:
            print(f"    {complexity:10s}: {comp_stats['total_queries']} queries, "
                  f"{comp_stats['success_rate']:.0%} success")


# ============================================================================
# Demo 3: User Feedback Weighting
# ============================================================================

def demo_feedback_weighting():
    """Demonstrate how user feedback is weighted more heavily."""
    print_header("Demo 3: User Feedback Weighting")

    print("""
When user feedback is available, it's weighted more heavily than
model confidence alone:

  effective_quality = 0.7 * feedback + 0.3 * model_quality

This ensures the system learns from actual user satisfaction,
not just model predictions.
""")

    sampler = ThompsonSampler(base_budget=5.0)

    # Case 1: Model says poor, user says great
    print_subheader("Case 1: Model Poor (0.5), User Great (0.95)")
    sampler.update(BudgetOutcome(
        complexity=QueryComplexity.MODERATE,
        budget_used=5.0,
        quality_score=0.5,   # Model thinks it's poor
        confidence=0.9,
        feedback=0.95,       # User thinks it's great!
    ))

    effective = 0.7 * 0.95 + 0.3 * 0.5
    print(f"  Effective Quality: {effective:.2f}")
    print(f"  Above threshold (0.7)? {'Yes' if effective >= 0.7 else 'No'}")
    print(f"  Result: Counted as SUCCESS")

    stats = sampler.get_statistics()
    print(f"  Total Successes: {stats['total_successes']}")

    # Case 2: Model says great, user says poor
    print_subheader("Case 2: Model Great (0.9), User Poor (0.3)")
    sampler.update(BudgetOutcome(
        complexity=QueryComplexity.MODERATE,
        budget_used=5.0,
        quality_score=0.9,   # Model thinks it's great
        confidence=0.9,
        feedback=0.3,        # User thinks it's poor!
    ))

    effective = 0.7 * 0.3 + 0.3 * 0.9
    print(f"  Effective Quality: {effective:.2f}")
    print(f"  Above threshold (0.7)? {'Yes' if effective >= 0.7 else 'No'}")
    print(f"  Result: Counted as FAILURE")

    stats = sampler.get_statistics()
    print(f"  Total Queries: {stats['total_queries']}")
    print(f"  Total Successes: {stats['total_successes']}")
    print(f"  Success Rate: {stats['success_rate']:.0%}")


# ============================================================================
# Demo 4: State Persistence
# ============================================================================

def demo_persistence():
    """Demonstrate saving and loading learned state."""
    print_header("Demo 4: State Persistence")

    print("""
Learned parameters can be saved to disk and restored later,
preserving learning across application restarts.
""")

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "learning_state.json")

        # Create and train a learner
        print_subheader("Training Original Learner")
        learner1 = AdaptiveBudgetLearner()

        for _ in range(30):
            for complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]:
                learner1.update(
                    complexity=complexity,
                    budget_used=learner1.get_budget(complexity),
                    quality_score=0.85,
                    confidence=0.9,
                )

        stats1 = learner1.get_statistics()
        print(f"  Total Updates: {stats1['total_updates']}")
        print(f"  SIMPLE alpha: {learner1.samplers[QueryComplexity.SIMPLE].alpha:.2f}")
        print(f"  MODERATE alpha: {learner1.samplers[QueryComplexity.MODERATE].alpha:.2f}")

        # Save state
        print_subheader("Saving State")
        learner1.save(state_path)
        print(f"  Saved to: {state_path}")
        print(f"  File size: {os.path.getsize(state_path)} bytes")

        # Create new learner and restore
        print_subheader("Restoring State to New Learner")
        learner2 = AdaptiveBudgetLearner()
        learner2.load(state_path)

        stats2 = learner2.get_statistics()
        print(f"  Total Updates: {stats2['total_updates']} (restored)")
        print(f"  SIMPLE alpha: {learner2.samplers[QueryComplexity.SIMPLE].alpha:.2f}")
        print(f"  MODERATE alpha: {learner2.samplers[QueryComplexity.MODERATE].alpha:.2f}")

        # Verify restoration
        match = stats1['total_updates'] == stats2['total_updates']
        print(f"\n  State Restored Correctly: {'Yes' if match else 'No'}")


# ============================================================================
# Demo 5: Convenience Functions
# ============================================================================

def demo_convenience_functions():
    """Demonstrate global convenience functions."""
    print_header("Demo 5: Convenience Functions")

    print("""
For simple use cases, convenience functions provide access to a
global learner without explicit instantiation:

  get_learner() - Get/create global AdaptiveBudgetLearner
  get_adaptive_budget(complexity) - Get budget for complexity string
  record_outcome(...) - Record query outcome for learning
  get_learning_statistics() - Get global learner statistics
""")

    print_subheader("Getting Adaptive Budgets")
    for complexity_str in ["trivial", "simple", "moderate", "complex", "research"]:
        budget = get_adaptive_budget(complexity_str)
        print(f"  {complexity_str:10s}: {budget:.2f} bits")

    print_subheader("Recording Outcomes")
    # Record some outcomes
    record_outcome(
        complexity="moderate",
        budget_used=5.0,
        quality_score=0.88,
        confidence=0.9,
    )
    record_outcome(
        complexity="simple",
        budget_used=3.0,
        quality_score=0.92,
        confidence=0.95,
    )
    record_outcome(
        complexity="complex",
        budget_used=8.0,
        quality_score=0.75,
        confidence=0.8,
    )
    print("  Recorded 3 outcomes")

    print_subheader("Global Statistics")
    stats = get_learning_statistics()
    print(f"  Total Updates: {stats['total_updates']}")
    print(f"  Quality Threshold: {stats['quality_threshold']}")
    print(f"  Exploration Rate: {stats['exploration_rate']}")


# ============================================================================
# Demo 6: Full Learning Simulation
# ============================================================================

def demo_full_simulation():
    """Simulate a full learning scenario over many queries."""
    print_header("Demo 6: Full Learning Simulation")

    print("""
Simulating 100 queries with varying complexity and outcomes.
The learner should adapt budgets based on observed success rates.
""")

    learner = AdaptiveBudgetLearner(
        quality_threshold=0.7,
        enable_exploration=True,
        exploration_rate=0.1,
    )

    # Simulation parameters
    complexity_distribution = [
        (QueryComplexity.TRIVIAL, 0.2),
        (QueryComplexity.SIMPLE, 0.3),
        (QueryComplexity.MODERATE, 0.3),
        (QueryComplexity.COMPLEX, 0.15),
        (QueryComplexity.RESEARCH, 0.05),
    ]

    # Quality model: base_quality increases with budget (diminishing returns)
    def simulate_quality(complexity: QueryComplexity, budget: float) -> float:
        base = {
            QueryComplexity.TRIVIAL: 0.9,    # Easy queries succeed often
            QueryComplexity.SIMPLE: 0.85,
            QueryComplexity.MODERATE: 0.75,
            QueryComplexity.COMPLEX: 0.65,
            QueryComplexity.RESEARCH: 0.55,
        }[complexity]

        # Budget boost (diminishing returns)
        budget_boost = min(0.3, budget / 30.0)
        return min(1.0, base + budget_boost)

    print_subheader("Running Simulation")

    import random
    random.seed(42)  # Reproducibility

    for i in range(100):
        # Select complexity based on distribution
        r = random.random()
        cumulative = 0
        for complexity, prob in complexity_distribution:
            cumulative += prob
            if r <= cumulative:
                break

        # Get adaptive budget
        budget = learner.get_budget(complexity)

        # Simulate quality
        quality = simulate_quality(complexity, budget)
        # Add noise
        quality += random.gauss(0, 0.05)
        quality = max(0.0, min(1.0, quality))

        # Update learner
        learner.update(
            complexity=complexity,
            budget_used=budget,
            quality_score=quality,
            confidence=0.9,
        )

        if (i + 1) % 25 == 0:
            print(f"  Completed {i + 1} queries...")

    print_subheader("Final Results")
    stats = learner.get_statistics()
    print(f"  Total Queries: {stats['total_updates']}")
    print(f"  Recent Avg Quality: {stats['recent_avg_quality']:.2%}")
    print(f"  Recent Success Rate: {stats['recent_success_rate']:.0%}")

    print("\n  Learned Budgets vs Default:")
    print("    Complexity    Default    Learned    Change")
    for complexity in QueryComplexity:
        default = learner.DEFAULT_BUDGETS[complexity]
        rec = learner.get_recommendation(complexity)
        change = rec.recommended_budget - default
        sign = "+" if change >= 0 else ""
        print(f"    {complexity.value:10s}  {default:6.1f}    {rec.recommended_budget:6.2f}    {sign}{change:.2f}")

    print("\n  Per-Complexity Performance:")
    for complexity in QueryComplexity:
        comp_stats = stats['by_complexity'][complexity.value]
        if comp_stats['total_queries'] > 0:
            print(f"    {complexity.value:10s}: {comp_stats['total_queries']:3d} queries, "
                  f"E[Q]={comp_stats['expected_quality']:.2f}, "
                  f"success={comp_stats['success_rate']:.0%}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#  PHASE 6.4: ADAPTIVE LEARNING FOR CONTEXT PACKING        #")
    print("#" + " " * 58 + "#")
    print("#" * 60)

    print("""
This demo showcases Thompson Sampling for learning optimal MI budgets
from query outcomes. The system:

1. Maintains Beta(alpha, beta) posteriors per complexity level
2. Samples budgets that balance exploration/exploitation
3. Updates posteriors based on success/failure outcomes
4. Weighs user feedback more heavily than model predictions
5. Persists learned state across sessions

Press Enter to continue through each demo...
""")

    demos = [
        demo_thompson_sampler,
        demo_adaptive_learner,
        demo_feedback_weighting,
        demo_persistence,
        demo_convenience_functions,
        demo_full_simulation,
    ]

    for demo in demos:
        demo()
        print("\n" + "-" * 60)

    print_header("Demo Complete!")
    print("""
Phase 6.4 provides:
- Thompson Sampling for principled exploration/exploitation
- Per-complexity learning of optimal MI budgets
- User feedback weighting (70% feedback, 30% model)
- State persistence for continuous learning
- Simple convenience API for integration

Integration Example:
    from hololoom.context_packing import get_adaptive_budget, record_outcome

    # Get budget for query complexity
    budget = get_adaptive_budget("moderate")

    # After query execution...
    record_outcome(
        complexity="moderate",
        budget_used=budget,
        quality_score=0.85,
        feedback=0.9  # Optional user feedback
    )
""")


if __name__ == "__main__":
    main()
