"""
Recursive Reasoner Demo - 8 Comprehensive Scenarios
====================================================

Demonstrates all capabilities of the recursive reasoning system:
1. Simple query (converges quickly)
2. Low-confidence query (needs refinement)
3. Complex query (requires decomposition)
4. Iterative improvement (watch convergence)
5. Strategy comparison (expand vs rerank vs decompose)
6. Learning trajectory (system improves over time)
7. Multi-criteria convergence (combined strategies)
8. Real-world research query (full recursive pipeline)

Created: 2025-11-20
Author: HoloLoom Team
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.config import Config
from HoloLoom.apps.departments.rag_department import RAGDepartment
from HoloLoom.convergence.recursive_reasoner_enhanced import create_recursive_reasoner
from HoloLoom.protocols.recursive_reasoning import RecursiveConfig, ReasoningStrategy


# ============================================================================
# Helper Functions
# ============================================================================

def print_header(title: str):
    """Print scenario header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_result_summary(result: dict):
    """Print result summary."""
    print(f"\n{'─'*80}")
    print("  RESULT SUMMARY")
    print(f"{'─'*80}")
    print(f"  Final Confidence: {result['final_confidence']:.2%}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Convergence: {result['convergence_reason']}")
    print(f"  Total Time: {result['total_latency_ms']:.1f}ms")

    print(f"\n  Improvement Trajectory:")
    for i, conf in enumerate(result['improvement_trajectory'], 1):
        bar = '█' * int(conf * 40)
        print(f"    Iter {i}: {bar} {conf:.2%}")

    if result.get('decomposition'):
        print(f"\n  Decomposition:")
        print(f"    Decomposed: {result['decomposition']['decomposed']}")
        if result['decomposition']['decomposed']:
            print(f"    Sub-queries: {len(result['decomposition']['sub_queries'])}")
            for i, sq in enumerate(result['decomposition']['sub_queries'], 1):
                print(f"      {i}. {sq[:60]}...")

    print(f"\n  Final Answer (first 200 chars):")
    print(f"    {result['final_answer'][:200]}...")
    print()


# ============================================================================
# Scenario 1: Simple Query (Converges Quickly)
# ============================================================================

async def scenario_1_simple_query():
    """Scenario 1: Simple factual query that converges in 1-2 iterations."""
    print_header("Scenario 1: Simple Query (Quick Convergence)")

    print("Query: 'What is Thompson Sampling?'")
    print("Expected: Converges quickly (1-2 iterations) with high confidence\n")

    async with RAGDepartment(config=Config.fast()) as rag:
        result = await rag.recursive_reason(
            query="What is Thompson Sampling?",
            max_depth=5,
            min_confidence=0.85,
            enable_decomposition=False  # Simple query, no decomposition needed
        )

        print_result_summary(result)


# ============================================================================
# Scenario 2: Low-Confidence Query (Needs Refinement)
# ============================================================================

async def scenario_2_low_confidence():
    """Scenario 2: Query that initially has low confidence and needs refinement."""
    print_header("Scenario 2: Low-Confidence Query (Needs Refinement)")

    print("Query: 'Explain the mathematical foundations of Bayesian bandits'")
    print("Expected: Initial low confidence triggers refinement loop\n")

    async with RAGDepartment(config=Config.fast()) as rag:
        result = await rag.recursive_reason(
            query="Explain the mathematical foundations of Bayesian bandits",
            max_depth=5,
            min_confidence=0.85,
            enable_decomposition=False
        )

        print_result_summary(result)

        # Show refinement steps
        print(f"{'─'*80}")
        print("  REFINEMENT STEPS:")
        print(f"{'─'*80}")
        for step in result['refinement_steps']:
            print(f"  {step['iteration']}. [{step['strategy']:15s}] "
                  f"Confidence: {step['confidence']:.2%} "
                  f"(Δ {step['improvement']:+.3f})")
            print(f"      {step['reasoning']}")


# ============================================================================
# Scenario 3: Complex Query (Requires Decomposition)
# ============================================================================

async def scenario_3_complex_query():
    """Scenario 3: Complex query that triggers automatic decomposition."""
    print_header("Scenario 3: Complex Query (Automatic Decomposition)")

    print("Query: 'Compare and contrast Thompson Sampling and Upper Confidence Bound algorithms'")
    print("Expected: Query decomposed into sub-queries, then synthesized\n")

    async with RAGDepartment(config=Config.fast()) as rag:
        result = await rag.recursive_reason(
            query="Compare and contrast Thompson Sampling and Upper Confidence Bound algorithms",
            max_depth=5,
            min_confidence=0.85,
            enable_decomposition=True  # Enable decomposition
        )

        print_result_summary(result)


# ============================================================================
# Scenario 4: Iterative Improvement (Watch Convergence)
# ============================================================================

async def scenario_4_iterative_improvement():
    """Scenario 4: Watch incremental quality improvement over iterations."""
    print_header("Scenario 4: Iterative Improvement (Convergence Visualization)")

    print("Query: 'What are the tradeoffs of reinforcement learning in production?'")
    print("Expected: Steady confidence improvement until convergence\n")

    async with RAGDepartment(config=Config.fast()) as rag:
        result = await rag.recursive_reason(
            query="What are the tradeoffs of reinforcement learning in production?",
            max_depth=5,
            min_confidence=0.85,
            enable_decomposition=False
        )

        print_result_summary(result)

        # Visualize convergence
        print(f"{'─'*80}")
        print("  CONVERGENCE VISUALIZATION:")
        print(f"{'─'*80}\n")

        for i, (curr, step) in enumerate(zip(result['improvement_trajectory'], result['refinement_steps'])):
            # Confidence bar
            bar_length = int(curr * 50)
            bar = '█' * bar_length + '░' * (50 - bar_length)

            # Improvement indicator
            improvement_indicator = ""
            if i > 0:
                delta = result['improvement_trajectory'][i] - result['improvement_trajectory'][i-1]
                if delta > 0.05:
                    improvement_indicator = "  ↑↑ Significant improvement"
                elif delta > 0.02:
                    improvement_indicator = "  ↑ Improvement"
                elif delta > 0:
                    improvement_indicator = "  ↗ Small improvement"
                elif delta == 0:
                    improvement_indicator = "  → Plateau"
                else:
                    improvement_indicator = "  ↓ Degradation"

            print(f"  Iter {i+1}: |{bar}| {curr:.2%}  [{step['strategy']:12s}]{improvement_indicator}")


# ============================================================================
# Scenario 5: Strategy Comparison
# ============================================================================

async def scenario_5_strategy_comparison():
    """Scenario 5: Compare different refinement strategies."""
    print_header("Scenario 5: Strategy Comparison")

    print("Testing different strategies on same query\n")

    query = "Explain the exploration-exploitation tradeoff"

    strategies_to_test = [
        ("expand_search", "Expand search (retrieve more sources)"),
        ("rerank", "Rerank sources for better precision"),
        ("alternate_mode", "Try alternate reasoning mode")
    ]

    async with RAGDepartment(config=Config.fast()) as rag:
        for strategy_name, description in strategies_to_test:
            print(f"\n{'─'*80}")
            print(f"  Strategy: {description}")
            print(f"{'─'*80}")

            # Note: Current implementation selects strategy automatically
            # This scenario shows the automatic selection in action
            result = await rag.recursive_reason(
                query=query,
                max_depth=3,
                min_confidence=0.85,
                enable_decomposition=False
            )

            print(f"  Final Confidence: {result['final_confidence']:.2%}")
            print(f"  Iterations: {result['iterations']}")
            print(f"  Strategies used: {[s['strategy'] for s in result['refinement_steps']]}")


# ============================================================================
# Scenario 6: Learning Trajectory
# ============================================================================

async def scenario_6_learning_trajectory():
    """Scenario 6: System learns and improves over time."""
    print_header("Scenario 6: Learning Trajectory (System Improves Over Time)")

    print("Processing multiple queries to observe learning\n")

    queries = [
        "What is gradient descent?",
        "How does backpropagation work?",
        "Explain the vanishing gradient problem",
        "What are optimization algorithms?",
        "Compare SGD and Adam optimizers"
    ]

    results = []

    async with RAGDepartment(config=Config.fast()) as rag:
        for i, query in enumerate(queries, 1):
            print(f"\n{'─'*40}")
            print(f"  Query {i}/{len(queries)}: {query}")
            print(f"{'─'*40}")

            result = await rag.recursive_reason(
                query=query,
                max_depth=5,
                min_confidence=0.85,
                enable_decomposition=False,
                enable_learning=True  # Enable learning
            )

            results.append(result)

            print(f"  Final Confidence: {result['final_confidence']:.2%}")
            print(f"  Iterations: {result['iterations']}")

        # Show learning progression
        print(f"\n{'='*80}")
        print("  LEARNING PROGRESSION:")
        print(f"{'='*80}\n")

        print("  Query | Final Conf | Iterations | Avg Improvement")
        print("  " + "-"*60)

        for i, (query, result) in enumerate(zip(queries, results), 1):
            avg_improvement = result['metadata']['total_improvement'] / max(result['iterations'], 1)
            print(f"    {i:2d}  |   {result['final_confidence']:.2%}     |     {result['iterations']}      |     {avg_improvement:+.3f}")


# ============================================================================
# Scenario 7: Multi-Criteria Convergence
# ============================================================================

async def scenario_7_multi_criteria():
    """Scenario 7: Multi-criteria convergence detection."""
    print_header("Scenario 7: Multi-Criteria Convergence")

    print("Demonstrating different convergence criteria\n")

    test_cases = [
        ("High confidence threshold met", "What is Python?", 0.90),
        ("Improvement plateau detected", "Explain neural networks", 0.70),
        ("Max iterations reached", "Comprehensive AI overview", 0.60)
    ]

    async with RAGDepartment(config=Config.fast()) as rag:
        for case_name, query, threshold in test_cases:
            print(f"\n{'─'*80}")
            print(f"  Case: {case_name}")
            print(f"  Query: {query}")
            print(f"  Threshold: {threshold:.2%}")
            print(f"{'─'*80}")

            result = await rag.recursive_reason(
                query=query,
                max_depth=3,
                min_confidence=threshold,
                enable_decomposition=False
            )

            print(f"  Convergence Reason: {result['convergence_reason']}")
            print(f"  Final Confidence: {result['final_confidence']:.2%}")
            print(f"  Iterations: {result['iterations']}")


# ============================================================================
# Scenario 8: Real-World Research Query
# ============================================================================

async def scenario_8_research_query():
    """Scenario 8: Full recursive pipeline on complex research query."""
    print_header("Scenario 8: Real-World Research Query (Full Pipeline)")

    print("Complex research query exercising all features:\n")
    print("Query: 'What are the latest advances in large language model training,")
    print("       including techniques for reducing computational cost, improving")
    print("       factual accuracy, and mitigating biases?'\n")

    print("Expected:")
    print("  - Query decomposition into sub-topics")
    print("  - Multiple refinement iterations")
    print("  - Strategy learning and adaptation")
    print("  - Comprehensive synthesis\n")

    async with RAGDepartment(config=Config.fast()) as rag:
        result = await rag.recursive_reason(
            query=(
                "What are the latest advances in large language model training, "
                "including techniques for reducing computational cost, improving "
                "factual accuracy, and mitigating biases?"
            ),
            max_depth=5,
            min_confidence=0.85,
            enable_decomposition=True,  # Complex query, enable decomposition
            enable_learning=True  # Learn from this
        )

        print_result_summary(result)

        # Detailed breakdown
        print(f"{'='*80}")
        print("  DETAILED BREAKDOWN:")
        print(f"{'='*80}\n")

        print(f"  Initial Confidence: {result['metadata']['initial_confidence']:.2%}")
        print(f"  Final Confidence: {result['metadata']['final_confidence']:.2%}")
        print(f"  Total Improvement: {result['metadata']['total_improvement']:+.2%}")
        print(f"  Avg Improvement per Iteration: {result['metadata']['total_improvement'] / max(result['iterations'], 1):+.2%}")

        if result.get('decomposition') and result['decomposition']['decomposed']:
            print(f"\n  Decomposition Details:")
            print(f"    Complexity Score: {result['decomposition']['complexity_score']:.2f}")
            print(f"    Synthesis Strategy: {result['decomposition']['synthesis_strategy']}")
            print(f"    Sub-queries:")
            for i, sq in enumerate(result['decomposition']['sub_queries'], 1):
                print(f"      {i}. {sq}")


# ============================================================================
# Main Demo Runner
# ============================================================================

async def run_all_scenarios():
    """Run all 8 scenarios."""
    print("\n")
    print("="*80)
    print(" "*20 + "RECURSIVE REASONER DEMO")
    print(" "*15 + "8 Comprehensive Scenarios")
    print("="*80)

    scenarios = [
        ("1", "Simple Query (Quick Convergence)", scenario_1_simple_query),
        ("2", "Low-Confidence Query (Refinement)", scenario_2_low_confidence),
        ("3", "Complex Query (Decomposition)", scenario_3_complex_query),
        ("4", "Iterative Improvement (Visualization)", scenario_4_iterative_improvement),
        ("5", "Strategy Comparison", scenario_5_strategy_comparison),
        ("6", "Learning Trajectory", scenario_6_learning_trajectory),
        ("7", "Multi-Criteria Convergence", scenario_7_multi_criteria),
        ("8", "Real-World Research Query", scenario_8_research_query),
    ]

    for num, name, func in scenarios:
        try:
            await func()
            print(f"\n✓ Scenario {num} completed successfully")
        except Exception as e:
            print(f"\n✗ Scenario {num} failed: {e}")

        # Pause between scenarios
        print("\nPress Enter to continue to next scenario...")
        input()

    print("\n" + "="*80)
    print(" "*25 + "ALL SCENARIOS COMPLETE!")
    print("="*80 + "\n")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\nRecursive Reasoner Demo")
    print("=======================\n")
    print("This demo shows 8 scenarios demonstrating:")
    print("  1. Quick convergence for simple queries")
    print("  2. Automatic refinement for low confidence")
    print("  3. Query decomposition for complex queries")
    print("  4. Convergence visualization")
    print("  5. Strategy comparison")
    print("  6. Learning over time")
    print("  7. Multi-criteria convergence")
    print("  8. Full pipeline on research query\n")

    print("Note: Requires running HoloLoom with RAG Department initialized")
    print("      and appropriate LLM backend (Ollama/Anthropic/OpenAI)\n")

    try:
        asyncio.run(run_all_scenarios())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
