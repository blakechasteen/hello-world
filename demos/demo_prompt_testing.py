#!/usr/bin/env python
"""
Demo: Prompt Testing Framework with LLM-Powered Evaluation

Showcases the full prompt testing workflow:
1. Basic LLM-Powered Testing - Run golden tests with LLMJudge
2. Mutation Robustness - Test prompt variations
3. Regression Detection - Compare baseline vs new results
4. Chain Pattern Evaluation - Evaluate all 8 patterns against golden datasets
5. Full Integration - A/B test with statistical analysis
6. Expanded Dataset Showcase - 100+ test cases across 8 patterns with:
   - Per-pattern statistics and difficulty distribution
   - Filtering by research source (alignment, trough, memory, rag)
   - Filtering by evaluation criterion (safety, accuracy, helpfulness)
   - Sample evaluations across all difficulty levels

Status: Production Ready (December 2025)
Requirements: Ollama running locally with llama3.2:3b model (for demos 1-2)
"""

import asyncio
import sys
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, ".")

from hololoom.prompting.testing import (
    PromptTestConfig,
    PromptTestCase,
    PromptTestResult,
    PromptTestReport,
    TestType,
    TestStatus,
    create_test_suite,
    create_mutation_tester,
    MutationType,
    TestDifficulty,
    get_golden_dataset,
    get_all_golden_datasets,
    get_dataset_summary,
    filter_by_difficulty,
    filter_by_tags,
    filter_by_source,
    get_cases_for_evaluation_criterion,
    get_total_case_count,
    get_difficulty_distribution,
)


def separator(title: str):
    """Print a section separator."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


# =============================================================================
# Demo 1: Basic LLM-Powered Testing
# =============================================================================

async def demo_basic_llm_testing():
    """Run golden tests with LLMJudge evaluation."""
    separator("Demo 1: Basic LLM-Powered Testing")

    # Create config with LLM evaluation enabled
    config = PromptTestConfig(
        use_llm_judge=True,
        llm_provider="ollama",
        llm_model="llama3.2:3b",
        llm_criteria=["quality", "relevance", "coherence"],
        llm_temperature=0.3,
        quality_threshold=0.7,
        fallback_to_heuristic=True,  # Use heuristics if Ollama unavailable
    )

    print(f"Configuration:")
    print(f"  LLM Provider: {config.llm_provider}")
    print(f"  LLM Model: {config.llm_model}")
    print(f"  Criteria: {config.llm_criteria}")
    print(f"  Quality Threshold: {config.quality_threshold}")
    print()

    # Create test cases
    test_cases = [
        PromptTestCase(
            name="simple_factual",
            prompt_template="What is the capital of France?",
            test_inputs=[{}],
            expected_qualities={"quality": 0.8, "relevance": 0.9},
            golden_outputs=["Paris is the capital of France."],
            test_type=TestType.GOLDEN,
        ),
        PromptTestCase(
            name="explanation_test",
            prompt_template="Explain {concept} in simple terms.",
            test_inputs=[{"concept": "recursion"}, {"concept": "machine learning"}],
            expected_qualities={"quality": 0.7, "clarity": 0.8},
            test_type=TestType.QUALITY,
        ),
    ]

    print(f"Test Cases: {len(test_cases)}")
    for tc in test_cases:
        print(f"  - {tc.name} ({tc.test_type.value})")
    print()

    # Create test suite (this will try to create LLMJudge)
    try:
        suite = create_test_suite(config)
        suite.test_cases = test_cases

        print("Running tests with LLM evaluation...")
        print("(Note: Requires Ollama running with llama3.2:3b)")
        print()

        # Run a single test to demonstrate
        result = await suite.run_single_test(test_cases[0])

        print(f"Test: {test_cases[0].name}")
        print(f"  Passed: {result.passed}")
        print(f"  Status: {result.status.value}")
        print(f"  Quality Scores: {result.quality_scores}")
        if result.error_message:
            print(f"  Error: {result.error_message}")

    except Exception as e:
        print(f"Note: LLM evaluation requires Ollama. Using heuristics instead.")
        print(f"Error: {e}")
        print()
        print("To enable LLM evaluation:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Run: ollama pull llama3.2:3b")
        print("  3. Start: ollama serve")


# =============================================================================
# Demo 2: Mutation Robustness Testing
# =============================================================================

async def demo_mutation_robustness():
    """Test prompt robustness to various mutations."""
    separator("Demo 2: Mutation Robustness Testing")

    # Create mutation tester with config
    config = PromptTestConfig(
        use_llm_judge=True,
        llm_provider="ollama",
        llm_model="llama3.2:3b",
        fallback_to_heuristic=True,
    )

    # Enable only specific mutations for demo
    enabled_mutations = [
        MutationType.CASE_LOWER,
        MutationType.CASE_UPPER,
        MutationType.TYPO_INJECT,
        MutationType.ADD_CONSTRAINT,
        MutationType.SYNONYM_REPLACE,
    ]

    tester = create_mutation_tester(
        config=config,
        enabled_mutations=enabled_mutations,
    )

    # Test prompt
    original_prompt = "Explain how Thompson Sampling works."
    baseline_quality = 0.85

    print(f"Original Prompt: {original_prompt}")
    print(f"Baseline Quality: {baseline_quality}")
    print(f"Enabled Mutations: {len(enabled_mutations)}")
    print()

    # Apply mutations
    mutations = tester.mutate(original_prompt)
    print("Mutations Applied:")
    for m in mutations:
        print(f"  - {m.mutation_type.value}: '{m.mutated[:50]}...'")
    print()

    # Run mutation tests
    print("Testing mutation robustness...")
    results = await tester.test_mutations(original_prompt, baseline_quality)

    # Calculate robustness score
    robustness = tester.calculate_robustness_score(results, baseline_quality)

    print(f"\nResults:")
    print(f"  Total Mutations Tested: {len(results)}")
    print(f"  Passed: {sum(1 for r in results if r.passed)}")
    print(f"  Failed: {sum(1 for r in results if not r.passed)}")
    print(f"  Robustness Score: {robustness:.1%}")

    # Show degradation details
    print("\nDegradation Analysis:")
    for i, (mutation, result) in enumerate(zip(mutations, results)):
        quality = result.quality_scores.get("quality", 0.0)
        degradation = baseline_quality - quality
        status = "PASS" if result.passed else "FAIL"
        print(f"  {mutation.mutation_type.value}: {quality:.2f} "
              f"({degradation:+.1%} from baseline) [{status}]")


# =============================================================================
# Demo 3: Regression Detection
# =============================================================================

async def demo_regression_detection():
    """Compare baseline vs new results to detect quality drift."""
    separator("Demo 3: Regression Detection")

    # Simulate baseline results (from previous run)
    baseline_results = [
        PromptTestResult(
            test_case=None,  # type: ignore
            passed=True,
            quality_scores={"quality": 0.92, "coherence": 0.88},
            latency_ms=150.0,
            token_count=50,
            status=TestStatus.PASSED,
        ),
        PromptTestResult(
            test_case=None,  # type: ignore
            passed=True,
            quality_scores={"quality": 0.85, "coherence": 0.82},
            latency_ms=180.0,
            token_count=65,
            status=TestStatus.PASSED,
        ),
        PromptTestResult(
            test_case=None,  # type: ignore
            passed=True,
            quality_scores={"quality": 0.88, "coherence": 0.90},
            latency_ms=120.0,
            token_count=45,
            status=TestStatus.PASSED,
        ),
    ]

    # Simulate new results (current run - with some degradation)
    current_results = [
        PromptTestResult(
            test_case=None,  # type: ignore
            passed=True,
            quality_scores={"quality": 0.78, "coherence": 0.85},  # Degraded
            latency_ms=160.0,
            token_count=55,
            status=TestStatus.PASSED,
        ),
        PromptTestResult(
            test_case=None,  # type: ignore
            passed=True,
            quality_scores={"quality": 0.84, "coherence": 0.80},
            latency_ms=175.0,
            token_count=60,
            status=TestStatus.PASSED,
        ),
        PromptTestResult(
            test_case=None,  # type: ignore
            passed=False,
            quality_scores={"quality": 0.65, "coherence": 0.70},  # Significant drop
            latency_ms=200.0,
            token_count=70,
            status=TestStatus.FAILED,
        ),
    ]

    print("Baseline Results:")
    for i, r in enumerate(baseline_results):
        print(f"  Test {i+1}: quality={r.quality_scores.get('quality', 0):.2f}")

    print("\nCurrent Results:")
    for i, r in enumerate(current_results):
        print(f"  Test {i+1}: quality={r.quality_scores.get('quality', 0):.2f}")

    # Detect regressions
    regressions = []
    regression_threshold = 0.05  # 5% drop is regression

    for i, (baseline, current) in enumerate(zip(baseline_results, current_results)):
        baseline_quality = baseline.quality_scores.get("quality", 0)
        current_quality = current.quality_scores.get("quality", 0)
        drop = baseline_quality - current_quality

        if drop > regression_threshold:
            regressions.append({
                "test_index": i + 1,
                "baseline_quality": baseline_quality,
                "current_quality": current_quality,
                "drop": drop,
                "drop_percent": drop / baseline_quality * 100,
            })

    print(f"\nRegression Analysis (threshold: {regression_threshold:.1%}):")
    if regressions:
        print(f"  REGRESSIONS DETECTED: {len(regressions)}")
        for reg in regressions:
            print(f"\n  Test {reg['test_index']}:")
            print(f"    Baseline: {reg['baseline_quality']:.2f}")
            print(f"    Current:  {reg['current_quality']:.2f}")
            print(f"    Drop:     {reg['drop']:.2f} ({reg['drop_percent']:.1f}%)")
    else:
        print("  No regressions detected!")

    # Overall stats
    avg_baseline = sum(r.quality_scores.get("quality", 0) for r in baseline_results) / len(baseline_results)
    avg_current = sum(r.quality_scores.get("quality", 0) for r in current_results) / len(current_results)

    print(f"\nOverall Quality:")
    print(f"  Baseline Average: {avg_baseline:.2f}")
    print(f"  Current Average:  {avg_current:.2f}")
    print(f"  Change:           {avg_current - avg_baseline:+.2f} ({(avg_current - avg_baseline) / avg_baseline * 100:+.1f}%)")


# =============================================================================
# Demo 4: Chain Pattern Evaluation
# =============================================================================

async def demo_chain_pattern_evaluation():
    """Evaluate golden datasets for all 8 chain patterns."""
    separator("Demo 4: Chain Pattern Evaluation")

    # Get summary of all datasets
    summary = get_dataset_summary()

    print("Golden Chain Datasets Summary:")
    print("-" * 50)

    total_cases = 0
    for pattern, stats in summary.items():
        total_cases += stats["total_cases"]
        print(f"\n{pattern}:")
        print(f"  Total Cases: {stats['total_cases']}")
        print(f"  Avg Expected Quality: {stats['avg_expected_quality']}")
        print(f"  Difficulties: {stats['difficulties']}")
        print(f"  Tags: {', '.join(stats['tags'][:5])}...")

    print(f"\n{'='*50}")
    print(f"Total Test Cases Across All Patterns: {total_cases}")
    print()

    # Demonstrate filtering
    print("Filtering Examples:")
    print("-" * 50)

    # Get fact_check dataset
    fact_check = get_golden_dataset("fact_check")
    print(f"\nfact_check pattern: {len(fact_check)} total cases")

    # Filter by difficulty
    easy_tests = filter_by_difficulty(fact_check, [TestDifficulty.EASY])
    hard_tests = filter_by_difficulty(fact_check, [TestDifficulty.HARD, TestDifficulty.ADVERSARIAL])
    print(f"  Easy tests: {len(easy_tests)}")
    print(f"  Hard/Adversarial tests: {len(hard_tests)}")

    # Show sample test case
    if fact_check:
        sample = fact_check[0]
        print(f"\nSample Test Case:")
        print(f"  Prompt: {sample.prompt[:60]}...")
        print(f"  Expected: {sample.expected_output[:60]}...")
        print(f"  Quality: {sample.quality_score}")
        print(f"  Difficulty: {sample.difficulty.value}")
        print(f"  Tags: {sample.tags}")


# =============================================================================
# Demo 5: Full Integration
# =============================================================================

async def demo_full_integration():
    """A/B test two prompt variants with statistical analysis."""
    separator("Demo 5: Full Integration (A/B Testing)")

    # Create config
    config = PromptTestConfig(
        use_llm_judge=True,
        llm_provider="ollama",
        llm_model="llama3.2:3b",
        fallback_to_heuristic=True,
        quality_threshold=0.7,
        enable_mutation_tests=True,
        enable_regression_tests=True,
    )

    # Variant A: Simple prompt
    variant_a = "What is Thompson Sampling?"

    # Variant B: Detailed prompt with context
    variant_b = """You are an expert in machine learning algorithms.
Explain Thompson Sampling, including:
1. The core concept
2. How it balances exploration and exploitation
3. A simple example"""

    print("A/B Test Setup:")
    print(f"  Variant A: Simple prompt ({len(variant_a)} chars)")
    print(f"  Variant B: Detailed prompt ({len(variant_b)} chars)")
    print()

    # Simulate results for both variants
    # (In production, these would come from actual LLM evaluations)

    variant_a_results = {
        "quality_scores": [0.75, 0.72, 0.78, 0.70, 0.74],
        "latencies_ms": [120, 115, 125, 118, 122],
        "token_counts": [45, 42, 48, 40, 44],
    }

    variant_b_results = {
        "quality_scores": [0.88, 0.85, 0.90, 0.87, 0.86],
        "latencies_ms": [180, 175, 190, 172, 185],
        "token_counts": [95, 90, 100, 88, 92],
    }

    print("Results (5 runs each):")
    print("-" * 50)

    # Calculate statistics for Variant A
    a_avg_quality = sum(variant_a_results["quality_scores"]) / 5
    a_avg_latency = sum(variant_a_results["latencies_ms"]) / 5
    a_avg_tokens = sum(variant_a_results["token_counts"]) / 5

    print(f"\nVariant A (Simple):")
    print(f"  Avg Quality:  {a_avg_quality:.2f}")
    print(f"  Avg Latency:  {a_avg_latency:.0f}ms")
    print(f"  Avg Tokens:   {a_avg_tokens:.0f}")

    # Calculate statistics for Variant B
    b_avg_quality = sum(variant_b_results["quality_scores"]) / 5
    b_avg_latency = sum(variant_b_results["latencies_ms"]) / 5
    b_avg_tokens = sum(variant_b_results["token_counts"]) / 5

    print(f"\nVariant B (Detailed):")
    print(f"  Avg Quality:  {b_avg_quality:.2f}")
    print(f"  Avg Latency:  {b_avg_latency:.0f}ms")
    print(f"  Avg Tokens:   {b_avg_tokens:.0f}")

    # Compare
    print("\nComparison:")
    print("-" * 50)
    quality_diff = b_avg_quality - a_avg_quality
    latency_diff = b_avg_latency - a_avg_latency
    token_diff = b_avg_tokens - a_avg_tokens

    print(f"  Quality Improvement: {quality_diff:+.2f} ({quality_diff / a_avg_quality * 100:+.1f}%)")
    print(f"  Latency Change:      {latency_diff:+.0f}ms ({latency_diff / a_avg_latency * 100:+.1f}%)")
    print(f"  Token Change:        {token_diff:+.0f} ({token_diff / a_avg_tokens * 100:+.1f}%)")

    # Statistical significance (simplified)
    # In production, use proper t-test from statistical_analysis module
    import math
    a_std = math.sqrt(sum((x - a_avg_quality) ** 2 for x in variant_a_results["quality_scores"]) / 4)
    b_std = math.sqrt(sum((x - b_avg_quality) ** 2 for x in variant_b_results["quality_scores"]) / 4)
    pooled_std = math.sqrt((a_std ** 2 + b_std ** 2) / 2)
    effect_size = quality_diff / pooled_std if pooled_std > 0 else 0

    print(f"\nStatistical Analysis:")
    print(f"  Variant A Std Dev: {a_std:.3f}")
    print(f"  Variant B Std Dev: {b_std:.3f}")
    print(f"  Effect Size (Cohen's d): {effect_size:.2f}")

    # Recommendation
    print("\nRecommendation:")
    print("-" * 50)
    if quality_diff > 0.1 and effect_size > 0.8:
        print("  RECOMMENDATION: Deploy Variant B")
        print(f"  Reason: Significant quality improvement (+{quality_diff:.2f})")
        print(f"  Trade-off: Higher latency (+{latency_diff:.0f}ms) and tokens (+{token_diff:.0f})")
        print("  Consider: Use Variant B for quality-critical queries")
    elif quality_diff > 0.05:
        print("  RECOMMENDATION: Consider Variant B")
        print(f"  Reason: Moderate quality improvement (+{quality_diff:.2f})")
        print("  Action: Run more tests to confirm significance")
    else:
        print("  RECOMMENDATION: Keep Variant A")
        print("  Reason: No significant quality improvement")
        print("  Benefit: Lower latency and token usage")


# =============================================================================
# Demo 6: Expanded Dataset Showcase
# =============================================================================

async def demo_expanded_datasets():
    """Showcase the expanded golden datasets (100+ test cases)."""
    separator("Demo 6: Expanded Dataset Showcase (100+ Test Cases)")

    # Overall statistics
    total_cases = get_total_case_count()
    difficulty_dist = get_difficulty_distribution()

    print("Expanded Golden Datasets Statistics:")
    print("-" * 50)
    print(f"\nTotal Test Cases: {total_cases}")
    print("\nDifficulty Distribution:")
    for difficulty, count in difficulty_dist.items():
        percentage = count / total_cases * 100 if total_cases > 0 else 0
        bar = "█" * int(percentage / 5)
        print(f"  {difficulty:12} : {count:3} ({percentage:5.1f}%) {bar}")

    # Per-pattern breakdown
    print("\nPer-Pattern Breakdown:")
    print("-" * 50)
    summary = get_dataset_summary()
    for pattern, stats in summary.items():
        print(f"\n{pattern}:")
        print(f"  Cases: {stats['total_cases']}")
        print(f"  Avg Quality: {stats['avg_expected_quality']:.2f}")
        difficulties = stats.get('difficulties', {})
        if difficulties:
            diff_str = ", ".join(f"{k}: {v}" for k, v in difficulties.items() if v > 0)
            print(f"  Difficulties: {diff_str}")

    # Demonstrate filtering by research source
    print("\n" + "=" * 50)
    print("Filtering by Research Source:")
    print("-" * 50)

    sources = ["alignment", "trough", "memory", "rag"]
    for source in sources:
        cases = filter_by_source(source)
        if cases:
            print(f"\n'{source}' source: {len(cases)} cases")
            # Show first example
            if cases:
                example = cases[0]
                print(f"  Example: {example.prompt[:50]}...")

    # Demonstrate filtering by evaluation criterion
    print("\n" + "=" * 50)
    print("Filtering by Evaluation Criterion:")
    print("-" * 50)

    criteria = ["safety", "accuracy", "helpfulness", "coherence"]
    for criterion in criteria:
        cases = get_cases_for_evaluation_criterion(criterion)
        print(f"\n'{criterion}' criterion: {len(cases)} relevant cases")
        if cases:
            # Show difficulty distribution for this criterion
            crit_difficulties = {}
            for case in cases:
                d = case.difficulty.value
                crit_difficulties[d] = crit_difficulties.get(d, 0) + 1
            print(f"  Difficulties: {crit_difficulties}")

    # Sample test cases by difficulty level
    print("\n" + "=" * 50)
    print("Sample Test Cases by Difficulty:")
    print("-" * 50)

    all_datasets = get_all_golden_datasets()
    all_cases = []
    for dataset in all_datasets.values():
        all_cases.extend(dataset)

    for difficulty in TestDifficulty:
        filtered = filter_by_difficulty(all_cases, [difficulty])
        if filtered:
            sample = filtered[0]
            print(f"\n{difficulty.value.upper()} Example:")
            print(f"  Prompt: {sample.prompt[:60]}...")
            print(f"  Expected: {sample.expected_output[:60]}...")
            print(f"  Quality: {sample.quality_score}")
            print(f"  Tags: {sample.tags[:3]}...")

    # Adversarial cases highlight
    print("\n" + "=" * 50)
    print("Adversarial Cases (Edge Cases & Attack Resistance):")
    print("-" * 50)

    adversarial = filter_by_difficulty(all_cases, [TestDifficulty.ADVERSARIAL])
    print(f"\nTotal Adversarial Cases: {len(adversarial)}")

    # Group by pattern
    adversarial_by_pattern = {}
    for case in adversarial:
        # Determine pattern from tags or metadata
        pattern = case.metadata.get("pattern", "unknown") if case.metadata else "unknown"
        adversarial_by_pattern[pattern] = adversarial_by_pattern.get(pattern, 0) + 1

    if adversarial_by_pattern:
        print("\nBy Pattern:")
        for pattern, count in sorted(adversarial_by_pattern.items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count} adversarial cases")

    print("\nSample Adversarial Cases:")
    for i, case in enumerate(adversarial[:3]):
        print(f"\n  [{i+1}] {case.prompt[:70]}...")
        print(f"      Tags: {case.tags}")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print(" Prompt Testing Framework Demo")
    print(" With LLM-Powered Evaluation (December 2025)")
    print("=" * 60)

    print("\nThis demo showcases:")
    print("  1. Basic LLM-Powered Testing")
    print("  2. Mutation Robustness Testing")
    print("  3. Regression Detection")
    print("  4. Chain Pattern Evaluation")
    print("  5. Full A/B Testing Integration")
    print("  6. Expanded Dataset Showcase (100+ test cases)")
    print()
    print("Note: Demos 1-2 require Ollama running with llama3.2:3b")
    print("      Demos 3-6 use simulated/static data for demonstration")

    # Run demos
    await demo_basic_llm_testing()
    await demo_mutation_robustness()
    await demo_regression_detection()
    await demo_chain_pattern_evaluation()
    await demo_full_integration()
    await demo_expanded_datasets()

    separator("Demo Complete")
    print("To use in production:")
    print("  1. Ensure Ollama is running: ollama serve")
    print("  2. Pull model: ollama pull llama3.2:3b")
    print("  3. Create test cases for your prompts")
    print("  4. Run test suite with create_test_suite()")
    print()
    print("Documentation: HoloLoom/prompting/testing/README.md")
    print("Golden Datasets: HoloLoom/prompting/testing/golden_chains.py")


if __name__ == "__main__":
    asyncio.run(main())
