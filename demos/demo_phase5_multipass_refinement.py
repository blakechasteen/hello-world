#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Phase 5 Multi-Pass Refinement

Demonstrates iterative quality improvement through multiple refinement passes.

Shows:
1. Quality improvement trajectory (0.65 → 0.75 → 0.88 → 0.92)
2. Different refinement strategies (ADAPTIVE, DEPTH_FIRST, BREADTH_FIRST)
3. Stopping criteria in action (quality threshold, max passes, diminishing returns)
4. Pass-by-pass statistics and insights

Expected Output:
- Initial quality: ~0.65 (needs refinement)
- After refinement: ~0.92 (3-4 passes)
- Total improvement: +27 quality points (+41%)
- Demonstrates how refinement adapts strategy based on weakest dimension
"""

import asyncio
from unittest.mock import Mock
from dataclasses import dataclass
from typing import Optional, List

# Import Phase 5 components
try:
    from HoloLoom.awareness.refinement_engine import (
        RefinementEngine,
        RefinementStrategy,
        StoppingCriterion,
        RefinementResult
    )
    REFINEMENT_AVAILABLE = True
except ImportError:
    print("❌ Phase 5 not available. Install HoloLoom awareness module.")
    exit(1)


# ============================================================================
# Mock LLM Context Packer (for demo purposes)
# ============================================================================

@dataclass
class MockQualityScore:
    """Mock quality score"""
    coherence: float
    completeness: float
    relevance: float
    overall: float


@dataclass
class MockPackedGeneration:
    """Mock packed generation result"""
    response: str
    quality_score: MockQualityScore


class DemoLLMContextPacker:
    """
    Demo packer that simulates improving quality across passes.

    Simulates realistic quality improvement:
    - Pass 0 (initial): 0.65 quality (low completeness)
    - Pass 1: 0.75 quality (improved completeness, still weak coherence)
    - Pass 2: 0.88 quality (improved coherence, good quality)
    - Pass 3: 0.92 quality (excellent quality, threshold reached)
    """

    def __init__(self):
        self.call_count = 0
        self.enable_compression = True
        self.min_importance = 0.7
        self.budget = Mock(total=5000)

        # Quality progression: improving over passes
        self.quality_progression = [
            {
                "overall": 0.65,
                "coherence": 0.70,
                "completeness": 0.55,  # Weakest dimension
                "relevance": 0.70,
                "response": (
                    "Thompson Sampling is a technique. "
                    "It uses probability distributions. "
                    "Good for exploration-exploitation."
                )
            },
            {
                "overall": 0.75,
                "coherence": 0.68,  # Still weak
                "completeness": 0.80,  # Improved
                "relevance": 0.77,
                "response": (
                    "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
                    "It maintains probability distributions over reward parameters for each arm. "
                    "By sampling from these distributions, it naturally balances exploration and "
                    "exploitation. More uncertain arms get sampled more often."
                )
            },
            {
                "overall": 0.88,
                "coherence": 0.90,  # Much improved
                "completeness": 0.85,
                "relevance": 0.89,
                "response": (
                    "Thompson Sampling is a Bayesian algorithm for the multi-armed bandit problem "
                    "that elegantly solves the exploration-exploitation tradeoff. It works by:\n"
                    "1. Maintaining Beta(α, β) distributions for each arm (α = successes, β = failures)\n"
                    "2. Sampling a reward probability θᵢ from each distribution\n"
                    "3. Selecting the arm with highest sampled θᵢ\n"
                    "4. Updating the distribution based on observed reward\n\n"
                    "This provides optimal regret bounds while being computationally efficient. "
                    "Arms with high uncertainty (flat distributions) get sampled more often, "
                    "naturally implementing exploration. As evidence accumulates, distributions "
                    "narrow and the algorithm exploits the best arm."
                )
            },
            {
                "overall": 0.92,
                "coherence": 0.93,
                "completeness": 0.90,
                "relevance": 0.93,
                "response": (
                    "Thompson Sampling is a Bayesian algorithm for the multi-armed bandit problem "
                    "that elegantly solves the exploration-exploitation tradeoff through probability matching.\n\n"
                    "**Algorithm:**\n"
                    "1. Maintain Beta(αᵢ, βᵢ) distributions for each arm i\n"
                    "2. Sample reward probability θᵢ ~ Beta(αᵢ, βᵢ) for each arm\n"
                    "3. Select arm argmax_i θᵢ\n"
                    "4. Observe reward r ∈ {0,1}\n"
                    "5. Update: αᵢ ← αᵢ + r, βᵢ ← βᵢ + (1-r)\n\n"
                    "**Key Properties:**\n"
                    "- Optimal regret: O(√(KT log T)) where K=arms, T=rounds\n"
                    "- Natural exploration: Uncertain arms have flat distributions → higher sampling probability\n"
                    "- Exploitation: As evidence accumulates, distributions narrow around true values\n"
                    "- Computationally efficient: O(K) per round\n\n"
                    "**Advantages over ε-greedy:**\n"
                    "- Adapts exploration rate automatically (no ε tuning)\n"
                    "- Better empirical performance in practice\n"
                    "- Theoretically grounded in Bayesian decision theory\n\n"
                    "**Applications:** A/B testing, clinical trials, recommendation systems, "
                    "hyperparameter optimization, reinforcement learning."
                )
            }
        ]

    async def pack_and_generate(
        self,
        query: str,
        awareness_context,
        memory_results: Optional[List] = None,
        max_memories: int = 10
    ) -> MockPackedGeneration:
        """Mock pack and generate with improving quality"""
        # Get quality for this pass
        quality_data = self.quality_progression[
            min(self.call_count, len(self.quality_progression) - 1)
        ]

        self.call_count += 1

        # Create quality score
        quality_score = MockQualityScore(
            coherence=quality_data["coherence"],
            completeness=quality_data["completeness"],
            relevance=quality_data["relevance"],
            overall=quality_data["overall"]
        )

        # Simulate some latency
        await asyncio.sleep(0.05)  # 50ms

        return MockPackedGeneration(
            response=quality_data["response"],
            quality_score=quality_score
        )


# ============================================================================
# Demo Functions
# ============================================================================

def print_header(title: str):
    """Print section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_quality_score(score: MockQualityScore, label: str = "Quality"):
    """Print formatted quality score"""
    print(f"{label}:")
    print(f"  Overall:      {score.overall:.2f}")
    print(f"  Coherence:    {score.coherence:.2f}")
    print(f"  Completeness: {score.completeness:.2f}")
    print(f"  Relevance:    {score.relevance:.2f}")


def print_refinement_pass(pass_data, pass_num: int):
    """Print refinement pass details"""
    print(f"\n{'─' * 80}")
    print(f"Pass {pass_num}: {pass_data.strategy_used.value.upper()}")
    print(f"{'─' * 80}")
    print(f"Quality: {pass_data.quality_before:.2f} → {pass_data.quality_after:.2f} "
          f"({pass_data.quality_improvement:+.2f}, {(pass_data.quality_improvement/pass_data.quality_before)*100:+.1f}%)")
    print(f"Latency: {pass_data.latency_ms:.1f}ms")


async def demo_basic_refinement():
    """Demo 1: Basic refinement showing quality improvement"""
    print_header("Demo 1: Basic Multi-Pass Refinement")

    # Create packer and refiner
    packer = DemoLLMContextPacker()
    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.90,  # Target quality
        max_passes=4,
        strategy=RefinementStrategy.ADAPTIVE
    )

    # Execute refinement
    print("Refining query: 'Explain Thompson Sampling'")
    print("Target quality: 0.90")
    print("Max passes: 4")
    print("Strategy: ADAPTIVE (auto-select based on weakness)\n")

    result = await refiner.refine(
        query="Explain Thompson Sampling",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Print initial quality
    print("📊 Initial Generation (Pass 0)")
    print_quality_score(result.initial_generation.quality_score, "Quality")
    print(f"\n💡 Weakest dimension: completeness (0.55)")
    print(f"   → Refiner will use BREADTH_FIRST strategy to expand coverage")

    # Print each refinement pass
    for i, pass_data in enumerate(result.passes, 1):
        print_refinement_pass(pass_data, i)

    # Print final result
    print(f"\n{'=' * 80}")
    print("📈 Final Result")
    print(f"{'=' * 80}")
    print(f"Passes executed: {result.passes_executed}")
    print(f"Best pass: {result.best_pass_number}")
    print(f"Stopping reason: {result.stopping_criterion.value}")
    print(f"\nQuality improvement: {result.initial_quality:.2f} → {result.final_quality:.2f} "
          f"({result.total_improvement:+.2f}, {(result.total_improvement/result.initial_quality)*100:+.1f}%)")
    print(f"Total time: {result.total_latency_ms:.0f}ms "
          f"(avg {result.avg_latency_per_pass_ms:.0f}ms/pass)")

    # Show final response excerpt
    print(f"\n📝 Final Response (excerpt):")
    final_response = result.final_generation.response
    excerpt = final_response[:200] + "..." if len(final_response) > 200 else final_response
    print(f"   {excerpt}")


async def demo_strategy_comparison():
    """Demo 2: Compare different refinement strategies"""
    print_header("Demo 2: Strategy Comparison")

    strategies = [
        (RefinementStrategy.DEPTH_FIRST, "Deep dive with more context"),
        (RefinementStrategy.BREADTH_FIRST, "Expand coverage broadly"),
        (RefinementStrategy.FOCUSED, "Focus on weakest dimension"),
        (RefinementStrategy.ADAPTIVE, "Auto-select based on weakness")
    ]

    results = []

    for strategy, description in strategies:
        packer = DemoLLMContextPacker()
        refiner = RefinementEngine(
            packer=packer,
            quality_threshold=0.90,
            max_passes=3,
            strategy=strategy
        )

        result = await refiner.refine(
            query="Explain Thompson Sampling",
            awareness_ctx=Mock(),
            memory_results=[],
            max_memories=10
        )

        results.append((strategy, description, result))

    # Print comparison table
    print(f"{'Strategy':<20} {'Description':<35} {'Passes':<8} {'Quality Δ':<12} {'Final'}")
    print(f"{'-' * 95}")

    for strategy, description, result in results:
        quality_delta = result.total_improvement
        passes = result.passes_executed

        print(f"{strategy.value:<20} {description:<35} {passes:<8} "
              f"{quality_delta:+.2f} ({(quality_delta/result.initial_quality)*100:+.0f}%)   {result.final_quality:.2f}")

    print(f"\n💡 Insight: ADAPTIVE strategy chooses BREADTH_FIRST for low completeness,")
    print(f"   then switches to DEPTH_FIRST for subsequent passes as needed.")


async def demo_stopping_criteria():
    """Demo 3: Different stopping criteria"""
    print_header("Demo 3: Stopping Criteria")

    scenarios = [
        {
            "name": "Quality Threshold Reached",
            "threshold": 0.85,
            "max_passes": 5,
            "expected": "Stops at pass 2 (quality 0.88 ≥ 0.85)"
        },
        {
            "name": "Max Passes Limit",
            "threshold": 0.99,  # Won't be reached
            "max_passes": 2,
            "expected": "Stops after 2 passes (limit reached)"
        },
        {
            "name": "High Initial Quality",
            "threshold": 0.70,
            "max_passes": 5,
            "expected": "Skips refinement (quality 0.65 already acceptable)"
        }
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print(f"{'─' * 60}")
        print(f"Configuration: threshold={scenario['threshold']:.2f}, max_passes={scenario['max_passes']}")

        # Special handling for scenario 3
        if scenario['name'] == "High Initial Quality":
            # Use a packer that starts with high quality
            class HighQualityPacker(DemoLLMContextPacker):
                def __init__(self):
                    super().__init__()
                    self.quality_progression[0]["overall"] = 0.95

            packer = HighQualityPacker()
        else:
            packer = DemoLLMContextPacker()

        refiner = RefinementEngine(
            packer=packer,
            quality_threshold=scenario['threshold'],
            max_passes=scenario['max_passes'],
            strategy=RefinementStrategy.ADAPTIVE
        )

        result = await refiner.refine(
            query="Explain Thompson Sampling",
            awareness_ctx=Mock(),
            memory_results=[],
            max_memories=10
        )

        print(f"Result: {result.passes_executed} passes executed")
        print(f"Stopped: {result.stopping_criterion.value}")
        print(f"Quality: {result.initial_quality:.2f} → {result.final_quality:.2f}")
        print(f"Expected: {scenario['expected']}")

        # Verify expectation
        if "Stops at pass" in scenario['expected']:
            expected_passes = int(scenario['expected'].split("pass ")[1][0])
            status = "✅" if result.passes_executed == expected_passes else "❌"
        elif "Skips refinement" in scenario['expected']:
            status = "✅" if result.passes_executed == 0 else "❌"
        elif "limit reached" in scenario['expected']:
            status = "✅" if result.stopping_criterion == StoppingCriterion.MAX_PASSES else "❌"
        else:
            status = "?"

        print(f"Status: {status}")


async def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("  Phase 5: Multi-Pass Refinement - Demo")
    print("=" * 80)
    print("\nDemonstrates iterative quality improvement through multiple refinement passes.")
    print("Shows strategy selection, stopping criteria, and quality trajectory tracking.")

    # Run demos
    await demo_basic_refinement()
    await demo_strategy_comparison()
    await demo_stopping_criteria()

    # Summary
    print_header("Summary")
    print("✅ Phase 5 multi-pass refinement enables:")
    print("   • Iterative quality improvement (3-4 passes typical)")
    print("   • Adaptive strategy selection based on weakest dimension")
    print("   • Intelligent stopping criteria (quality threshold, max passes, diminishing returns)")
    print("   • Complete pass-by-pass provenance for debugging")
    print("\n📊 Typical improvement: +20-30 quality points (+30-45%)")
    print("⏱️  Typical overhead: 150-300ms for 2-4 passes")
    print("🎯 Use case: Low-confidence queries that need deeper reasoning\n")


if __name__ == "__main__":
    asyncio.run(main())
