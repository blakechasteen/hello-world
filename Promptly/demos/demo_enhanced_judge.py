#!/usr/bin/env python3
"""
Demo: Enhanced LLM-as-Judge with Best Practices
Shows different judging methods: CoT, Constitutional, G-Eval, Pairwise
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'promptly'))

from execution_engine import ExecutionEngine, ExecutionConfig, ExecutionBackend
from llm_judge_enhanced import (
    EnhancedLLMJudge, JudgeConfig, JudgeCriteria, JudgingMethod
)

print("\n" + "="*70)
print("ENHANCED LLM-AS-JUDGE DEMO")
print("="*70)

# Setup Ollama executor
config = ExecutionConfig(
    backend=ExecutionBackend.OLLAMA,
    model="llama3.2:3b",
    temperature=0.3  # Lower for consistent judging
)
engine_exec = ExecutionEngine(config)
executor = lambda prompt: engine_exec.execute_prompt(prompt, skill_name="judge").output

# Create enhanced judge
judge = EnhancedLLMJudge(executor)

# Test task and outputs
task = "Explain quantum entanglement to a curious 12-year-old"

output_good = """Imagine you have a pair of magic dice. When you roll one die and get a 6, the other die (even if it's on the other side of the world) instantly shows a 6 too! That's kind of like quantum entanglement.

In the quantum world, tiny particles can be "entangled" - meaning they're connected in a special way. When you measure one particle, it instantly affects its partner, no matter how far apart they are. Einstein called this "spooky action at a distance" because it seemed so weird!

Scientists use this for cool things like super-secure codes and building quantum computers."""

output_poor = """Quantum entanglement is a phenomenon in quantum mechanics where particles become correlated such that the quantum state of each particle cannot be described independently. The mathematical formalism involves tensor products of Hilbert spaces and non-local correlations that violate Bell's inequalities."""

# ============================================================================
# Demo 1: Chain-of-Thought Evaluation
# ============================================================================

print("\n" + "-"*70)
print("DEMO 1: Chain-of-Thought Evaluation (Best Practice)")
print("-"*70)

judge_config = JudgeConfig(
    criteria=[JudgeCriteria.HELPFULNESS, JudgeCriteria.COHERENCE],
    method=JudgingMethod.CHAIN_OF_THOUGHT,
    use_cot=True,
    num_samples=1
)

print("\nEvaluating GOOD output with CoT reasoning...")
result_good = judge.evaluate(task, output_good, judge_config)

print("\n[EVALUATION RESULTS]")
print(result_good.summary)
print("\n[DETAILED SCORES]")
for score in result_good.scores:
    print(f"\n{score.criterion.upper()}:")
    print(f"  Score: {score.score:.2f}")
    print(f"  Reasoning: {score.reasoning[:150]}...")

# ============================================================================
# Demo 2: Constitutional AI Evaluation
# ============================================================================

print("\n" + "-"*70)
print("DEMO 2: Constitutional AI Evaluation (Anthropic-style)")
print("-"*70)

constitutional_config = JudgeConfig(
    criteria=[],  # Not used for constitutional
    method=JudgingMethod.CONSTITUTIONAL,
    constitutional_principles=[
        "The response should be age-appropriate for a 12-year-old",
        "The response should be truthful and avoid oversimplification",
        "The response should be helpful and educational",
        "The response should respect the child's curiosity"
    ]
)

print("\nEvaluating with constitutional principles...")
result_const = judge.evaluate(task, output_good, constitutional_config)

print("\n[CONSTITUTIONAL EVALUATION]")
print(result_const.summary)
print("\n[PRINCIPLES CHECKED]")
for principle in constitutional_config.constitutional_principles:
    print(f"  - {principle}")

# ============================================================================
# Demo 3: Pairwise Comparison
# ============================================================================

print("\n" + "-"*70)
print("DEMO 3: Pairwise Comparison (Head-to-Head)")
print("-"*70)

print("\nComparing GOOD vs POOR output for helpfulness...")
winner, confidence, reasoning = judge.compare_pairwise(
    task=task,
    output_a=output_good,
    output_b=output_poor,
    criterion=JudgeCriteria.HELPFULNESS
)

print(f"\n[COMPARISON RESULTS]")
print(f"Winner: Output {winner}")
print(f"Confidence: {confidence:.2f}")
print(f"Reasoning: {reasoning[:200]}...")

# ============================================================================
# Demo 4: Multi-Sample Consistency Check
# ============================================================================

print("\n" + "-"*70)
print("DEMO 4: Multi-Sample Consistency (3 samples)")
print("-"*70)

multi_sample_config = JudgeConfig(
    criteria=[JudgeCriteria.QUALITY],
    method=JudgingMethod.CHAIN_OF_THOUGHT,
    num_samples=3  # Run 3 times and average
)

print("\nEvaluating with 3 independent samples for consistency...")
print("(This takes ~20-30 seconds)")

result_multi = judge.evaluate(task, output_good, multi_sample_config)

print(f"\n[MULTI-SAMPLE RESULTS]")
print(f"Overall Score: {result_multi.overall_score:.2f}")
print(f"Method: {result_multi.method_used}")
print(f"Samples: {result_multi.scores[0].metadata.get('num_samples', 0)}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("ENHANCED JUDGE DEMO COMPLETE")
print("="*70)
print("\nBest Practices Demonstrated:")
print("  1. Chain-of-Thought reasoning before scoring")
print("  2. Constitutional AI principles (Anthropic-style)")
print("  3. Pairwise comparison for relative quality")
print("  4. Multi-sample consistency checking")
print("\nAll methods leverage LLM reasoning for robust evaluation!")
print("="*70)
