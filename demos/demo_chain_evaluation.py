"""
Demo: Chain Evaluation with A/B Testing
=========================================

Demonstrates the HoloLoom chain evaluation framework with:
- LLM Judge (Ollama backend) for quality scoring
- Chain comparison using different patterns
- A/B testing integration for statistical validation
- Multiple evaluation presets

Usage:
    PYTHONPATH=. python demos/demo_chain_evaluation.py

Requirements:
    - Ollama running locally (ollama serve)
    - Model: llama3.2:3b (or configure OLLAMA_MODEL env var)

December 2025
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.chaining import (
    # Core
    Chain,
    ChainStep,
    StepType,
    ChainPatterns,
    # Evaluation
    LLMJudge,
    ChainEvaluator,
    TestCase,
    JudgeConfig,
    JudgeCriteria,
    EvalPresets,
    create_judge,
    create_evaluator,
)
from HoloLoom.prompting.analytics.ab_testing import ABTest, create_ab_test


# ============================================================================
# Demo 1: Basic LLM Judge Usage
# ============================================================================

async def demo_basic_judge():
    """Demonstrate basic LLM Judge evaluation."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic LLM Judge Usage")
    print("=" * 60)

    # Create judge with Ollama backend
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    judge = create_judge(model_provider="ollama", model_name=model)

    print(f"\nUsing model: {model}")

    # Define a task and output to evaluate
    task = "Explain what Thompson Sampling is in simple terms"

    # Two different outputs to compare
    output_good = """
    Thompson Sampling is a technique for making decisions when you're uncertain.

    Imagine you're at a casino with multiple slot machines. You don't know which
    machine pays out the most. Thompson Sampling helps you balance between:
    1. Trying new machines (exploration)
    2. Playing machines that have paid well before (exploitation)

    It works by maintaining a probability estimate for each option and randomly
    sampling from these probabilities to decide what to try next. Over time,
    it naturally focuses on the best options while still occasionally exploring.
    """

    output_poor = """
    Thompson Sampling is an algorithm. It uses Bayesian methods. Beta distributions
    are involved. You sample from posteriors. It's used in bandits.
    """

    # Evaluate with quality preset
    config = EvalPresets.quality_eval()

    print("\n--- Evaluating Good Output ---")
    try:
        result_good = await judge.evaluate(task, output_good, config)
        print(f"Overall Score: {result_good.overall_score:.2f}")
        print(f"Passed: {result_good.passed}")
        print("Criterion Scores:")
        for score in result_good.scores:
            print(f"  {score.criterion.value}: {score.score:.2f} - {score.reasoning[:50]}...")
    except Exception as e:
        print(f"Evaluation failed (is Ollama running?): {e}")
        result_good = None

    print("\n--- Evaluating Poor Output ---")
    try:
        result_poor = await judge.evaluate(task, output_poor, config)
        print(f"Overall Score: {result_poor.overall_score:.2f}")
        print(f"Passed: {result_poor.passed}")
        print("Criterion Scores:")
        for score in result_poor.scores:
            print(f"  {score.criterion.value}: {score.score:.2f} - {score.reasoning[:50]}...")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        result_poor = None

    # Compare outputs
    if result_good and result_poor:
        print("\n--- Comparison ---")
        try:
            comparison = await judge.compare_outputs(task, output_good, output_poor, config)
            print(f"Winner: Output {'A (Good)' if comparison['winner'] == 'A' else 'B (Poor)'}")
            print(f"Reasoning: {comparison['reasoning'][:100]}...")
        except Exception as e:
            print(f"Comparison failed: {e}")

    return judge


# ============================================================================
# Demo 2: Chain Evaluation
# ============================================================================

async def demo_chain_evaluation(judge: LLMJudge):
    """Demonstrate chain evaluation with test cases."""
    print("\n" + "=" * 60)
    print("Demo 2: Chain Evaluation")
    print("=" * 60)

    # Create a simple chain for evaluation
    chain = ChainPatterns.verified_query()
    print(f"\nChain: {chain.name}")
    print(f"Steps: {list(chain.steps.keys())}")

    # Create test cases
    test_cases = [
        TestCase(
            input_data="What is machine learning?",
            expected_output="Machine learning is a subset of AI that enables systems to learn from data",
            metadata={"difficulty": "easy"}
        ),
        TestCase(
            input_data="Explain neural networks",
            expected_output="Neural networks are computing systems inspired by biological neural networks",
            metadata={"difficulty": "medium"}
        ),
        TestCase(
            input_data="What are the tradeoffs of Thompson Sampling vs UCB?",
            expected_output="Thompson Sampling is Bayesian and stochastic, UCB is frequentist and deterministic",
            metadata={"difficulty": "hard"}
        ),
    ]

    print(f"\nTest cases: {len(test_cases)}")
    for i, tc in enumerate(test_cases, 1):
        print(f"  {i}. {tc.input_data[:40]}... (difficulty: {tc.metadata.get('difficulty')})")

    # Create A/B test for tracking
    ab_test = create_ab_test(
        name="chain_eval_demo",
        control_description="Baseline chain performance",
        treatment_description="Chain performance metrics"
    )

    # Create evaluator
    evaluator = create_evaluator(ab_test, judge)

    # Evaluate chain (note: this simulates execution since we don't have a real orchestrator)
    config = EvalPresets.chain_eval()

    print("\n--- Evaluating Chain ---")
    print("(Simulating chain execution with mock outputs)")

    # For demo purposes, we'll evaluate with mock outputs
    # In production, ChainOrchestrator would execute the chain
    mock_outputs = [
        "Machine learning is a field of AI where computers learn patterns from data to make predictions.",
        "Neural networks are computational models with interconnected nodes that process information in layers.",
        "Thompson Sampling is probabilistic and explores naturally, while UCB is deterministic with explicit exploration terms.",
    ]

    total_score = 0.0
    for i, (tc, mock_output) in enumerate(zip(test_cases, mock_outputs), 1):
        print(f"\nTest Case {i}: {tc.input_data[:30]}...")
        try:
            result = await judge.evaluate(tc.input_data, mock_output, config)
            total_score += result.overall_score
            print(f"  Score: {result.overall_score:.2f}")
            print(f"  Passed: {result.passed}")

            # Log to A/B test
            ab_test.log_result(
                user_id=f"test_case_{i}",
                group=ab_test.assign_group(f"test_case_{i}"),
                quality_score=result.overall_score,
                execution_time_ms=100.0,  # Simulated
                metadata={"test_case": tc.input_data[:30]}
            )
        except Exception as e:
            print(f"  Evaluation failed: {e}")

    avg_score = total_score / len(test_cases) if test_cases else 0
    print(f"\n--- Summary ---")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Test Cases: {len(test_cases)}")

    return ab_test


# ============================================================================
# Demo 3: A/B Testing Chain Variants
# ============================================================================

async def demo_ab_testing(judge: LLMJudge):
    """Demonstrate A/B testing between chain variants."""
    print("\n" + "=" * 60)
    print("Demo 3: A/B Testing Chain Variants")
    print("=" * 60)

    # Create two chain variants
    chain_v1 = ChainPatterns.fact_check()
    chain_v2 = ChainPatterns.rag_optimized()

    print(f"\nChain A (Control): {chain_v1.name}")
    print(f"  Steps: {list(chain_v1.steps.keys())}")
    print(f"\nChain B (Treatment): {chain_v2.name}")
    print(f"  Steps: {list(chain_v2.steps.keys())}")

    # Create A/B test
    ab_test = create_ab_test(
        name="fact_check_vs_rag",
        control_description="Fact-checking chain with source verification",
        treatment_description="RAG-optimized chain with reranking",
        traffic_split=0.5,
        minimum_sample_size=5  # Lower for demo
    )

    # Simulate running both chains on test queries
    test_queries = [
        "Is Python the most popular programming language?",
        "Does Thompson Sampling outperform UCB in all scenarios?",
        "Is deep learning a subset of machine learning?",
        "Are transformers the best architecture for NLP?",
        "Does gradient descent always find the global minimum?",
        "Is reinforcement learning used in game playing?",
        "Are neural networks inspired by biological neurons?",
        "Does batch normalization always improve training?",
        "Is BERT a decoder-only transformer?",
        "Does dropout prevent overfitting?",
    ]

    print(f"\nTest queries: {len(test_queries)}")
    print("\n--- Running A/B Test ---")

    import random
    random.seed(42)  # Reproducible results

    for i, query in enumerate(test_queries, 1):
        user_id = f"user_{i}"
        group = ab_test.assign_group(user_id)

        # Simulate different quality based on chain type
        # In production, actual chain execution would produce these scores
        if group.value == "control":
            # Fact-check chain: good at verification but slower
            quality = 0.75 + random.uniform(0.05, 0.20)
            time_ms = 500 + random.uniform(0, 200)
        else:
            # RAG-optimized: faster but variable quality
            quality = 0.70 + random.uniform(0.10, 0.25)
            time_ms = 300 + random.uniform(0, 150)

        ab_test.log_result(
            user_id=user_id,
            group=group,
            quality_score=min(quality, 1.0),
            execution_time_ms=time_ms,
            metadata={"query": query[:30]}
        )

        print(f"  Query {i}: {group.value} | quality={quality:.2f} | time={time_ms:.0f}ms")

    # Analyze results
    print("\n--- A/B Test Analysis ---")
    summary = ab_test.get_summary()
    print(f"Test: {summary['name']}")
    print(f"Control: {summary['sample_sizes']['control']} samples")
    print(f"Treatment: {summary['sample_sizes']['treatment']} samples")
    print(f"Ready for analysis: {summary['ready_for_analysis']}")

    if summary['ready_for_analysis']:
        analysis = ab_test.analyze()
        print(f"\nStatistical Analysis:")
        print(f"  Significant: {analysis['is_significant']}")
        print(f"  Treatment Better: {analysis['treatment_better']}")
        print(f"  Deployment Decision: {analysis['deployment_decision']}")
        print(f"\nStatistics:")
        print(f"  Control Mean Quality: {analysis['statistics']['control']['mean_quality']:.3f}")
        print(f"  Treatment Mean Quality: {analysis['statistics']['treatment']['mean_quality']:.3f}")
        print(f"  Quality Improvement: {analysis['statistics']['difference']['quality_improvement_percent']:+.1f}%")
        print(f"  Cohen's d: {analysis['statistics']['difference']['cohens_d']:.3f}")
        print(f"  Time Increase: {analysis['statistics']['difference']['time_increase_percent']:+.1f}%")
        print(f"\nInterpretation: {analysis['interpretation']}")
    else:
        print("\nNeed more samples for statistical analysis.")

    return ab_test


# ============================================================================
# Demo 4: Evaluation Presets
# ============================================================================

async def demo_eval_presets():
    """Demonstrate different evaluation presets."""
    print("\n" + "=" * 60)
    print("Demo 4: Evaluation Presets")
    print("=" * 60)

    presets = [
        ("Quality Eval", EvalPresets.quality_eval()),
        ("Safety Eval", EvalPresets.safety_eval()),
        ("RAG Eval", EvalPresets.rag_eval()),
        ("Chain Eval", EvalPresets.chain_eval()),
        ("Comprehensive", EvalPresets.comprehensive_eval()),
    ]

    print("\nAvailable Presets:")
    for name, config in presets:
        criteria_names = [c.value for c in config.criteria]
        print(f"\n{name}:")
        print(f"  Criteria: {', '.join(criteria_names)}")
        print(f"  Threshold: {config.threshold:.2f}")
        if config.weights:
            weighted = [f"{c.value}:{w:.1f}" for c, w in zip(config.criteria, config.weights)]
            print(f"  Weights: {', '.join(weighted)}")

    print("\n--- Creating Custom Preset ---")
    custom_config = JudgeConfig(
        criteria=[JudgeCriteria.ACCURACY, JudgeCriteria.CONCISENESS],
        threshold=0.8,
        weights=[0.7, 0.3],
        model_provider="ollama",
        model_name="llama3.2:3b",
        custom_rubrics={
            JudgeCriteria.ACCURACY: "Is the response factually correct? Score 0-1.",
            JudgeCriteria.CONCISENESS: "Is the response brief and to the point? Score 0-1.",
        }
    )
    print(f"Custom config created with {len(custom_config.criteria)} criteria")
    print(f"  Criteria: {[c.value for c in custom_config.criteria]}")
    print(f"  Threshold: {custom_config.threshold}")


# ============================================================================
# Demo 5: Full Pipeline
# ============================================================================

async def demo_full_pipeline():
    """Demonstrate full evaluation pipeline."""
    print("\n" + "=" * 60)
    print("Demo 5: Full Evaluation Pipeline")
    print("=" * 60)

    print("\nPipeline Steps:")
    print("  1. Create chains to compare")
    print("  2. Create A/B test for tracking")
    print("  3. Create LLM Judge for evaluation")
    print("  4. Create evaluator combining test + judge")
    print("  5. Run test cases through both chains")
    print("  6. Analyze results with statistical tests")
    print("  7. Make deployment decision")

    # Create chains
    chain_simple = ChainPatterns.verified_query()
    chain_thorough = ChainPatterns.hallucination_guard()

    print(f"\n--- Chains ---")
    print(f"Simple: {chain_simple.name} ({len(chain_simple.steps)} steps)")
    print(f"Thorough: {chain_thorough.name} ({len(chain_thorough.steps)} steps)")

    # Create A/B test
    ab_test = create_ab_test(
        name="simple_vs_thorough",
        control_description="Simple verified query",
        treatment_description="Thorough hallucination guard",
        traffic_split=0.5,
        minimum_sample_size=3
    )

    # Create judge
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    judge = create_judge(model_provider="ollama", model_name=model)

    # Create evaluator
    evaluator = create_evaluator(ab_test, judge)

    # Test cases
    test_cases = [
        TestCase("What is Python?", "A programming language"),
        TestCase("Define AI", "Artificial Intelligence"),
        TestCase("Explain ML", "Machine Learning"),
        TestCase("What is NLP?", "Natural Language Processing"),
        TestCase("Define deep learning", "A subset of ML using neural networks"),
        TestCase("What is reinforcement learning?", "Learning through rewards"),
    ]

    print(f"\n--- Test Cases: {len(test_cases)} ---")

    # Simulate evaluation
    import random
    random.seed(123)

    print("\n--- Running Evaluation ---")
    for tc in test_cases:
        user_id = tc.input_data[:10]
        group = ab_test.assign_group(user_id)

        # Simulate quality (thorough should be slightly better but slower)
        if group.value == "control":
            quality = 0.70 + random.uniform(0.05, 0.15)
            time_ms = 200 + random.uniform(0, 100)
        else:
            quality = 0.75 + random.uniform(0.05, 0.15)
            time_ms = 400 + random.uniform(0, 150)

        ab_test.log_result(user_id, group, quality, time_ms)
        print(f"  {tc.input_data[:20]}: {group.value} | {quality:.2f}")

    # Final analysis
    print("\n--- Final Analysis ---")
    if ab_test.get_summary()['ready_for_analysis']:
        analysis = ab_test.analyze()
        print(f"Deployment Decision: {analysis['deployment_decision'].upper()}")
        print(f"Interpretation: {analysis['interpretation']}")
    else:
        print("Need more samples (increase test cases)")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         HoloLoom Chain Evaluation Demo - December 2025       ║
╠══════════════════════════════════════════════════════════════╣
║  This demo shows the chain evaluation framework with:        ║
║  - LLM Judge (Ollama) for quality scoring                    ║
║  - A/B testing for statistical validation                    ║
║  - Multiple evaluation presets                               ║
║  - Chain comparison workflows                                ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Check Ollama availability
    print("Checking Ollama availability...")
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("  Ollama is available")
            model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
            if model in result.stdout:
                print(f"  Model '{model}' is available")
            else:
                print(f"  Warning: Model '{model}' not found, demos may fail")
        else:
            print("  Warning: Ollama not running, LLM demos will be skipped")
    except Exception as e:
        print(f"  Warning: Could not check Ollama: {e}")
        print("  LLM evaluation demos may fail")

    # Run demos
    try:
        # Demo 1: Basic Judge
        judge = await demo_basic_judge()

        # Demo 2: Chain Evaluation
        await demo_chain_evaluation(judge)

        # Demo 3: A/B Testing
        await demo_ab_testing(judge)

        # Demo 4: Presets
        await demo_eval_presets()

        # Demo 5: Full Pipeline
        await demo_full_pipeline()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
