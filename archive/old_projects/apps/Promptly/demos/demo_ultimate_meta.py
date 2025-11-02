#!/usr/bin/env python3
"""
Demo 10: The Ultimate Meta Test
Combines Hofstadter loops + Iterative refinement
Question: "Can recursive self-improvement lead to artificial general intelligence?"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'promptly'))

from execution_engine import ExecutionEngine, ExecutionConfig, ExecutionBackend
from recursive_loops import RecursiveEngine, LoopConfig, LoopType

print("\n" + "="*70)
print("DEMO 10: THE ULTIMATE META TEST")
print("="*70)
print("\nQuestion: Can recursive self-improvement lead to AGI?")
print("\nThis demo combines TWO recursive systems:")
print("  1. Hofstadter strange loops (4 meta-levels)")
print("  2. Iterative refinement (3 iterations)")
print("\nTotal time: ~60-90 seconds")
print("="*70)

# Setup Ollama
config = ExecutionConfig(
    backend=ExecutionBackend.OLLAMA,
    model="llama3.2:3b",
    temperature=0.7
)
engine_exec = ExecutionEngine(config)
executor = lambda prompt: engine_exec.execute_prompt(prompt, skill_name="demo").output

# Create recursive engine
engine = RecursiveEngine(executor)

# ============================================================================
# PHASE 1: Hofstadter Strange Loop (Deep Meta-Level Thinking)
# ============================================================================

print("\n" + "-"*70)
print("PHASE 1: Hofstadter Strange Loop - Deep Meta-Level Thinking")
print("-"*70)

hofstadter_config = LoopConfig(
    loop_type=LoopType.HOFSTADTER,
    max_iterations=4,
    enable_scratchpad=True
)

print("\nRunning 4 meta-levels of recursive thinking...")
print("(30-40 seconds)\n")

phase1_result = engine.execute_hofstadter_loop(
    task="Can recursive self-improvement lead to artificial general intelligence?",
    config=hofstadter_config
)

print("[PHASE 1 COMPLETE]")
print(f"- Meta-levels explored: {phase1_result.iterations}")
print(f"- Stop reason: {phase1_result.stop_reason}")

# Show meta-level progression
if 'levels' in phase1_result.metadata:
    print("\n[META-LEVEL OUTPUTS]")
    for i, level_output in enumerate(phase1_result.metadata['levels'], 1):
        preview = level_output[:150].replace('\n', ' ')
        print(f"  Level {i}: {preview}...")

print("\n[PHASE 1 SYNTHESIS]")
synthesis_preview = phase1_result.final_output[:300].replace('\n', ' ')
print(f"{synthesis_preview}...\n")

# ============================================================================
# PHASE 2: Iterative Refinement (Polish the Conclusion)
# ============================================================================

print("-"*70)
print("PHASE 2: Iterative Refinement - Polish the Conclusion")
print("-"*70)

refine_config = LoopConfig(
    loop_type=LoopType.REFINE,
    max_iterations=3,
    quality_threshold=0.85,
    min_improvement=0.05,
    enable_scratchpad=True
)

print("\nRefining the synthesis for clarity, depth, and insight...")
print("(Will stop at quality >= 0.85 or no improvement)")
print("(20-30 seconds)\n")

phase2_result = engine.execute_refine_loop(
    task="Polish this analysis for maximum clarity, philosophical depth, and insight about recursive self-improvement and AGI",
    initial_output=phase1_result.final_output,
    config=refine_config
)

print("[PHASE 2 COMPLETE]")
print(f"- Refinement iterations: {phase2_result.iterations}")
print(f"- Stop reason: {phase2_result.stop_reason}")

# Show quality progression
if phase2_result.improvement_history:
    print("\n[QUALITY PROGRESSION]")
    for i, score in enumerate(phase2_result.improvement_history, 1):
        bar = "#" * int(score * 50)
        print(f"  Iteration {i}: {score:.2f} {bar}")

# ============================================================================
# FINAL OUTPUT
# ============================================================================

print("\n" + "="*70)
print("FINAL REFINED OUTPUT")
print("(After 4 meta-levels + " + str(phase2_result.iterations) + " refinements)")
print("="*70)
print()
print(phase2_result.final_output)
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*70)
print("ULTIMATE META TEST COMPLETE!")
print("="*70)
print(f"\nPhase 1 (Hofstadter): {phase1_result.iterations} meta-levels")
print(f"Phase 2 (Refinement): {phase2_result.iterations} iterations")
print(f"Total recursive operations: {phase1_result.iterations + phase2_result.iterations}")
print("\nTwo recursive systems successfully combined!")
print("Deep philosophical insight + iterative polish = Ultimate answer")
print("="*70)

# Show scratchpad preview if available
if phase2_result.scratchpad and phase2_result.scratchpad.entries:
    print("\n[REFINEMENT THOUGHT PROCESS - Preview]")
    for entry in phase2_result.scratchpad.entries[:2]:  # Show first 2
        print(f"\nIteration {entry.iteration}:")
        print(f"  Thought: {entry.thought[:100]}...")
        if entry.score:
            print(f"  Score: {entry.score:.2f}")
