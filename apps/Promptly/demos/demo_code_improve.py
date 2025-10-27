#!/usr/bin/env python3
"""
Demo: Iterative Code Improvement
Watch code evolve from simple to production-ready through self-critique
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'promptly'))

from execution_engine import ExecutionEngine, ExecutionConfig, ExecutionBackend
from recursive_loops import RecursiveEngine, LoopConfig, LoopType

print("\n" + "="*70)
print("DEMO: Iterative Code Improvement")
print("(Watch code evolve through self-critique and refinement)")
print("="*70)

initial_code = """def f(n):
    if n < 2: return n
    return f(n-1) + f(n-2)"""

print("\n[INITIAL CODE]")
print(initial_code)
print("\nTASK: Transform into production-ready code with:")
print("  - Type hints")
print("  - Docstrings")
print("  - Memoization")
print("  - Error handling")
print("  - Performance optimization")

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

# Configure refinement loop
loop_config = LoopConfig(
    loop_type=LoopType.REFINE,
    max_iterations=4,
    quality_threshold=0.85,
    min_improvement=0.05,
    enable_scratchpad=True
)

print("\nRunning iterative refinement loop...")
print("(Will stop when quality >= 0.85 or no improvement)")
print("\nProcessing...\n")

# Execute!
result = engine.execute_refine_loop(
    task="Transform this into production-ready Fibonacci implementation with type hints, docstrings, memoization, and comprehensive error handling",
    initial_output=initial_code,
    config=loop_config
)

# Show full report
print("="*70)
print("REFINEMENT REPORT")
print("="*70)
print(result.to_report())

print("\n" + "="*70)
print("CODE EVOLUTION COMPLETE!")
print(f"Improved over {result.iterations} iterations")
print(f"Stop reason: {result.stop_reason}")
print("="*70)

# Show quality progression
if result.improvement_history:
    print("\n[QUALITY SCORES]")
    for i, score in enumerate(result.improvement_history, 1):
        bar = "#" * int(score * 50)
        print(f"Iteration {i}: {score:.2f} {bar}")
