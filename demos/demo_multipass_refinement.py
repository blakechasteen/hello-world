"""
Multi-Pass Refinement Demo
===========================
Demonstrates ELEGANCE and VERIFY strategies with multiple passes,
showing how iterative refinement improves different quality dimensions.

Each strategy takes multiple passes focusing on specific aspects:

ELEGANCE (3-pass):
  Pass 1: Clarity - make it understandable
  Pass 2: Simplicity - make it concise
  Pass 3: Beauty - make it elegant

VERIFY (3-pass):
  Pass 1: Accuracy verification
  Pass 2: Completeness check
  Pass 3: Consistency validation

This demonstrates the philosophy: "Great answers aren't written, they're refined."
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from HoloLoom.documentation.types import Query, MemoryShard, Spacetime, WeavingTrace
from HoloLoom.config import Config
from HoloLoom.recursive import AdvancedRefiner, RefinementStrategy
from HoloLoom.weaving_orchestrator import WeavingOrchestrator


def create_demo_shards() -> list[MemoryShard]:
    """Create demonstration knowledge shards"""
    return [
        MemoryShard(
            id="recursion",
            content="Recursion is when a function calls itself. It needs a base case to stop.",
            metadata={"topic": "programming", "complexity": "intermediate"}
        ),
        MemoryShard(
            id="recursion_detailed",
            content="Recursive algorithms work by breaking problems into smaller subproblems. "
                   "Each recursive call handles a simpler version until reaching the base case. "
                   "Common examples: factorial, fibonacci, tree traversal.",
            metadata={"topic": "programming", "complexity": "advanced"}
        ),
        MemoryShard(
            id="recursion_pitfalls",
            content="Recursion pitfalls: stack overflow from missing base case, inefficiency "
                   "from repeated computations (can be solved with memoization), and harder "
                   "debugging compared to iterative solutions.",
            metadata={"topic": "programming", "complexity": "advanced"}
        ),
    ]


def create_mock_spacetime(query_text: str, response: str, confidence: float = 0.65) -> Spacetime:
    """Create a mock spacetime for demo purposes"""
    return Spacetime(
        query_text=query_text,
        response=response,
        tool_used="answer",
        confidence=confidence,
        trace=WeavingTrace(
            tool_selected="answer",
            tool_confidence=confidence,
            threads_activated=["recursion"],
            motifs_detected=["recursion"],
            policy_adapter="default"
        ),
        complexity="FAST",
        metadata={}
    )


async def demo_elegance_refinement():
    """Demo: ELEGANCE strategy with 3 passes"""
    print("=" * 80)
    print("DEMO: ELEGANCE Refinement - Multi-Pass Polish")
    print("=" * 80)
    print()

    config = Config.fast()
    shards = create_demo_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        refiner = AdvancedRefiner(
            orchestrator=orchestrator,
            enable_learning=True
        )

        # Intentionally verbose/unclear response to refine
        query = Query(text="Explain recursion")
        initial_response = (
            "Recursion is basically when you have a function and that function, "
            "well, it calls itself, which might sound confusing but it's actually "
            "a programming technique where the function invokes itself and you need "
            "to make sure you have a base case otherwise it'll just keep calling "
            "itself forever and ever and that's bad because it causes a stack overflow "
            "which is an error that happens when you run out of memory on the call stack."
        )

        initial_spacetime = create_mock_spacetime(
            query_text=query.text,
            response=initial_response,
            confidence=0.65  # Low confidence triggers refinement
        )

        print(f"Query: {query.text}")
        print()
        print("Initial Response (Verbose/Unclear):")
        print("-" * 80)
        print(initial_response)
        print()
        print(f"Initial Confidence: {initial_spacetime.trace.tool_confidence:.2f}")
        print()

        # Refine with ELEGANCE strategy (3 passes)
        print("Applying ELEGANCE refinement (3 passes)...")
        print()

        result = await refiner.refine(
            query=query,
            initial_spacetime=initial_spacetime,
            strategy=RefinementStrategy.ELEGANCE,
            max_iterations=3,
            quality_threshold=0.95
        )

        # Show trajectory
        print("Quality Trajectory:")
        print("-" * 80)
        for i, metrics in enumerate(result.trajectory):
            print(f"Pass {i}: Quality = {metrics.score():.3f}, "
                  f"Confidence = {metrics.confidence:.2f}, "
                  f"Length = {metrics.response_length}")

        print()
        print("Refinement Summary:")
        print("-" * 80)
        print(result.summary())
        print()

        print("Final Response (After Elegance Refinement):")
        print("-" * 80)
        print(result.final_spacetime.response[:500])
        print()


async def demo_verify_refinement():
    """Demo: VERIFY strategy with 3 passes"""
    print("=" * 80)
    print("DEMO: VERIFY Refinement - Multi-Pass Verification")
    print("=" * 80)
    print()

    config = Config.fast()
    shards = create_demo_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        refiner = AdvancedRefiner(
            orchestrator=orchestrator,
            enable_learning=True
        )

        # Response with potential accuracy issues
        query = Query(text="What are the key aspects of recursion?")
        initial_response = (
            "Recursion is a fundamental programming concept where a function calls itself. "
            "The main requirement is having a base case to prevent infinite loops."
        )

        initial_spacetime = create_mock_spacetime(
            query_text=query.text,
            response=initial_response,
            confidence=0.70  # Medium confidence
        )

        print(f"Query: {query.text}")
        print()
        print("Initial Response (Needs Verification):")
        print("-" * 80)
        print(initial_response)
        print()
        print(f"Initial Confidence: {initial_spacetime.trace.tool_confidence:.2f}")
        print()

        # Refine with VERIFY strategy (3 passes)
        print("Applying VERIFY refinement (3 passes)...")
        print("  Pass 1: Accuracy verification")
        print("  Pass 2: Completeness check")
        print("  Pass 3: Consistency validation")
        print()

        result = await refiner.refine(
            query=query,
            initial_spacetime=initial_spacetime,
            strategy=RefinementStrategy.VERIFY,
            max_iterations=3,
            quality_threshold=0.92
        )

        # Show trajectory
        print("Quality Trajectory:")
        print("-" * 80)
        for i, metrics in enumerate(result.trajectory):
            verification_focus = ["Accuracy", "Completeness", "Consistency"][i] if i < 3 else "Accuracy"
            print(f"Pass {i} ({verification_focus}): Quality = {metrics.score():.3f}, "
                  f"Confidence = {metrics.confidence:.2f}")

        print()
        print("Refinement Summary:")
        print("-" * 80)
        print(result.summary())
        print()

        print("Final Response (After Verification):")
        print("-" * 80)
        print(result.final_spacetime.response[:500])
        print()


async def demo_elegance_vs_verify():
    """Demo: Compare ELEGANCE and VERIFY strategies"""
    print("=" * 80)
    print("DEMO: ELEGANCE vs VERIFY - Strategy Comparison")
    print("=" * 80)
    print()

    config = Config.fast()
    shards = create_demo_shards()

    query = Query(text="Explain the trade-offs of recursive vs iterative algorithms")
    initial_response = (
        "Recursion can be more elegant but uses more memory. "
        "Iteration is faster but might be harder to understand for some problems."
    )

    initial_spacetime = create_mock_spacetime(
        query_text=query.text,
        response=initial_response,
        confidence=0.68
    )

    print(f"Query: {query.text}")
    print()
    print("Initial Response:")
    print("-" * 80)
    print(initial_response)
    print(f"Initial Quality: {0.68:.2f}")
    print()

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        refiner = AdvancedRefiner(
            orchestrator=orchestrator,
            enable_learning=True
        )

        # Try ELEGANCE
        print("Refining with ELEGANCE strategy...")
        elegance_result = await refiner.refine(
            query=query,
            initial_spacetime=initial_spacetime,
            strategy=RefinementStrategy.ELEGANCE,
            max_iterations=3
        )

        print(f"  Final Quality: {elegance_result.trajectory[-1].score():.3f}")
        print(f"  Improvement: {elegance_result.improvement_rate:.3f}/iteration")
        print()

        # Try VERIFY
        print("Refining with VERIFY strategy...")
        verify_result = await refiner.refine(
            query=query,
            initial_spacetime=initial_spacetime,
            strategy=RefinementStrategy.VERIFY,
            max_iterations=3
        )

        print(f"  Final Quality: {verify_result.trajectory[-1].score():.3f}")
        print(f"  Improvement: {verify_result.improvement_rate:.3f}/iteration")
        print()

        # Compare
        print("Strategy Comparison:")
        print("-" * 80)
        print(f"ELEGANCE:")
        print(f"  Focus: Clarity → Simplicity → Beauty")
        print(f"  Improvement: {elegance_result.improvement_rate:.3f}/iteration")
        print()
        print(f"VERIFY:")
        print(f"  Focus: Accuracy → Completeness → Consistency")
        print(f"  Improvement: {verify_result.improvement_rate:.3f}/iteration")
        print()

        better_strategy = "ELEGANCE" if elegance_result.improvement_rate > verify_result.improvement_rate else "VERIFY"
        print(f"Better for this query: {better_strategy}")
        print()


async def demo_learning_from_refinements():
    """Demo: System learns which strategies work best"""
    print("=" * 80)
    print("DEMO: Learning From Refinement Patterns")
    print("=" * 80)
    print()

    config = Config.fast()
    shards = create_demo_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        refiner = AdvancedRefiner(
            orchestrator=orchestrator,
            enable_learning=True
        )

        # Process multiple queries with different strategies
        test_cases = [
            (
                "Explain recursion simply",
                "Recursion is when a function calls itself with a base case to stop.",
                RefinementStrategy.ELEGANCE
            ),
            (
                "What are the performance implications of recursion?",
                "Recursion can be slower and use more memory than iteration.",
                RefinementStrategy.VERIFY
            ),
            (
                "How do you implement a recursive function?",
                "Define the base case, then call the function with a simpler input.",
                RefinementStrategy.ELEGANCE
            ),
        ]

        print("Processing test cases to learn refinement patterns...")
        print()

        for i, (query_text, response, strategy) in enumerate(test_cases, 1):
            query = Query(text=query_text)
            spacetime = create_mock_spacetime(query_text, response, confidence=0.70)

            result = await refiner.refine(
                query=query,
                initial_spacetime=spacetime,
                strategy=strategy,
                max_iterations=2
            )

            print(f"{i}. {query_text[:50]}...")
            print(f"   Strategy: {strategy.value}")
            print(f"   Improvement: {result.improvement_rate:.3f}")
            print()

        # Show learned patterns
        print("Learned Refinement Patterns:")
        print("-" * 80)

        stats = refiner.get_strategy_statistics()
        for strategy_name, perf in stats.items():
            if perf['uses'] > 0:
                print(f"{strategy_name.upper()}:")
                print(f"  Uses: {perf['uses']}")
                print(f"  Avg Improvement: {perf['avg_improvement']:.3f}")
                print(f"  Success Rate: {perf['success_rate']:.1%}")
                print()

        print("System learns which strategies work best for which queries!")
        print()


async def main():
    """Run all demos"""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "         MULTI-PASS REFINEMENT DEMONSTRATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  ELEGANCE: Clarity → Simplicity → Beauty".center(78) + "║")
    print("║" + "  VERIFY: Accuracy → Completeness → Consistency".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    print('"Great answers aren\'t written, they\'re refined."')
    print()

    demos = [
        ("ELEGANCE Refinement", demo_elegance_refinement),
        ("VERIFY Refinement", demo_verify_refinement),
        ("Strategy Comparison", demo_elegance_vs_verify),
        ("Learning From Refinements", demo_learning_from_refinements),
    ]

    for name, demo_fn in demos:
        try:
            await demo_fn()
            print(f"✓ {name} complete")
            print()
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            print()

    print()
    print("=" * 80)
    print("MULTI-PASS REFINEMENT SUMMARY")
    print("=" * 80)
    print()
    print("Key Concepts:")
    print("  • ELEGANCE: Iterative polish for clarity, simplicity, beauty")
    print("  • VERIFY: Multi-pass verification for accuracy, completeness, consistency")
    print("  • Each pass focuses on a specific quality dimension")
    print("  • System learns which strategies work best")
    print("  • Quality trajectory shows incremental improvement")
    print()
    print("Philosophy: Great answers require multiple passes,")
    print("            each refining a different aspect of quality.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
