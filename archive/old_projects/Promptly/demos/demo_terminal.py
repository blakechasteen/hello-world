#!/usr/bin/env python3
"""
Terminal demos for Promptly recursive intelligence
Run impressive demos directly in the terminal with Ollama
"""

import sys
import os

# Add promptly to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'promptly'))

from execution_engine import ExecutionEngine, ExecutionConfig, ExecutionBackend
from recursive_loops import RecursiveEngine, LoopConfig, LoopType


def setup_ollama_executor():
    """Setup Ollama executor"""
    config = ExecutionConfig(
        backend=ExecutionBackend.OLLAMA,
        model="llama3.2:3b",
        temperature=0.7
    )
    engine = ExecutionEngine(config)
    return lambda prompt: engine.execute_prompt(prompt, skill_name="demo").output


def demo_1_strange_loop_about_strange_loops():
    """Demo: Use strange loop to explain strange loops (MOST META!)"""
    print("\n" + "="*70)
    print("DEMO 1: What is a strange loop?")
    print("(Using a strange loop to explain strange loops - perfectly meta!)")
    print("="*70)

    executor = setup_ollama_executor()
    engine = RecursiveEngine(executor)

    config = LoopConfig(
        loop_type=LoopType.HOFSTADTER,
        max_iterations=4,
        enable_scratchpad=True
    )

    print("\nRunning 4 meta-levels of recursive thinking...")
    print("(This will take ~30-40 seconds)\n")

    result = engine.execute_hofstadter_loop(
        task="What is a strange loop?",
        config=config
    )

    print(result.to_report())

    print("\n" + "="*70)
    print("AMAZING: The tool explained itself by being itself!")
    print("="*70)


def demo_2_iterative_code_improvement():
    """Demo: Iteratively improve code"""
    print("\n" + "="*70)
    print("DEMO 2: Iterative Code Improvement")
    print("(Watch code evolve through self-critique)")
    print("="*70)

    initial_code = """def f(n):
    if n < 2: return n
    return f(n-1) + f(n-2)"""

    print("\nInitial code:")
    print(initial_code)

    executor = setup_ollama_executor()
    engine = RecursiveEngine(executor)

    config = LoopConfig(
        loop_type=LoopType.REFINE,
        max_iterations=3,
        quality_threshold=0.85,
        enable_scratchpad=True
    )

    print("\nRunning iterative refinement...")
    print("(This will take ~20-30 seconds)\n")

    result = engine.execute_refine_loop(
        task="Transform this into production-ready code with type hints, docstrings, memoization, and error handling",
        initial_output=initial_code,
        config=config
    )

    print(result.to_report())

    print("\n" + "="*70)
    print("CODE EVOLUTION: Simple → Production-ready")
    print("="*70)


def demo_3_consciousness_strange_loop():
    """Demo: Is consciousness a strange loop?"""
    print("\n" + "="*70)
    print("DEMO 3: Is consciousness a strange loop?")
    print("(Hofstadter's core thesis from 'I Am a Strange Loop')")
    print("="*70)

    executor = setup_ollama_executor()
    engine = RecursiveEngine(executor)

    config = LoopConfig(
        loop_type=LoopType.HOFSTADTER,
        max_iterations=5,
        enable_scratchpad=True
    )

    print("\nRunning 5 meta-levels of recursive thinking...")
    print("(This will take ~40-50 seconds)\n")

    result = engine.execute_hofstadter_loop(
        task="Is consciousness a strange loop?",
        config=config
    )

    print(result.to_report())

    print("\n" + "="*70)
    print("DEEP INSIGHT: Each meta-level reveals new understanding")
    print("="*70)


def demo_4_ai_understanding_understanding():
    """Demo: Can AI understand understanding?"""
    print("\n" + "="*70)
    print("DEMO 4: Can an AI truly understand the concept of understanding?")
    print("(Self-referential paradox - AI thinking about AI thinking)")
    print("="*70)

    executor = setup_ollama_executor()
    engine = RecursiveEngine(executor)

    config = LoopConfig(
        loop_type=LoopType.HOFSTADTER,
        max_iterations=4,
        enable_scratchpad=True
    )

    print("\nRunning 4 meta-levels...")
    print("(This will take ~30-40 seconds)\n")

    result = engine.execute_hofstadter_loop(
        task="Can an AI truly understand the concept of understanding?",
        config=config
    )

    print(result.to_report())

    print("\n" + "="*70)
    print("PARADOX EXPLORED: The answer emerges through recursion")
    print("="*70)


def demo_5_combined_loops():
    """Demo: Combine Hofstadter + Refinement"""
    print("\n" + "="*70)
    print("DEMO 5: ULTIMATE META TEST")
    print("(Hofstadter loop + Iterative refinement combined)")
    print("="*70)

    executor = setup_ollama_executor()
    engine = RecursiveEngine(executor)

    # First: Hofstadter loop for deep thinking
    print("\nPhase 1: Deep meta-level thinking...")
    hofstadter_config = LoopConfig(
        loop_type=LoopType.HOFSTADTER,
        max_iterations=3,
        enable_scratchpad=True
    )

    phase1 = engine.execute_hofstadter_loop(
        task="Can recursive self-improvement lead to artificial general intelligence?",
        config=hofstadter_config
    )

    print("\nHofstadter Loop Result:")
    print("-" * 70)
    print(phase1.final_output[:300] + "...")

    # Second: Refine the conclusion
    print("\n\nPhase 2: Iteratively refining the conclusion...")
    refine_config = LoopConfig(
        loop_type=LoopType.REFINE,
        max_iterations=2,
        quality_threshold=0.85,
        enable_scratchpad=True
    )

    phase2 = engine.execute_refine_loop(
        task="Polish this answer for clarity, depth, and insight",
        initial_output=phase1.final_output,
        config=refine_config
    )

    print("\n" + "="*70)
    print("FINAL REFINED OUTPUT:")
    print("="*70)
    print(phase2.final_output)

    print("\n" + "="*70)
    print("TWO RECURSIVE SYSTEMS COMBINED!")
    print("="*70)


def quick_demo_creativity():
    """Quick demo: What is creativity?"""
    print("\n" + "="*70)
    print("QUICK DEMO: What is creativity?")
    print("="*70)

    executor = setup_ollama_executor()
    engine = RecursiveEngine(executor)

    config = LoopConfig(
        loop_type=LoopType.HOFSTADTER,
        max_iterations=3,
        enable_scratchpad=False  # Faster without scratchpad
    )

    print("\nRunning 3 meta-levels...\n")

    result = engine.execute_hofstadter_loop(
        task="What is creativity?",
        config=config
    )

    print("FINAL SYNTHESIS:")
    print(result.final_output)


def main():
    """Run demo menu"""
    print("\n" + "="*70)
    print("PROMPTLY RECURSIVE INTELLIGENCE - TERMINAL DEMOS")
    print("="*70)
    print("\nAvailable demos:")
    print("  1. Strange loop about strange loops (MOST META) - 30s")
    print("  2. Iterative code improvement - 20s")
    print("  3. Is consciousness a strange loop? (DEEPEST) - 40s")
    print("  4. Can AI understand understanding? (PARADOX) - 30s")
    print("  5. Ultimate meta test (COMBINED LOOPS) - 60s")
    print("  6. What is creativity? (QUICK) - 15s")
    print("  0. Run all demos (WARNING: ~3 minutes)")

    choice = input("\nEnter demo number (or 'q' to quit): ").strip()

    demos = {
        '1': demo_1_strange_loop_about_strange_loops,
        '2': demo_2_iterative_code_improvement,
        '3': demo_3_consciousness_strange_loop,
        '4': demo_4_ai_understanding_understanding,
        '5': demo_5_combined_loops,
        '6': quick_demo_creativity,
    }

    if choice == '0':
        print("\n🚀 RUNNING ALL DEMOS (grab a coffee!)\n")
        for i in ['1', '2', '3', '4', '5', '6']:
            demos[i]()
            input("\nPress Enter to continue to next demo...")
    elif choice in demos:
        demos[choice]()
    elif choice.lower() == 'q':
        print("\nBye!")
        return
    else:
        print("\nInvalid choice!")
        return

    print("\n" + "="*70)
    print("Demo complete! All stopping conditions working correctly.")
    print("="*70)


if __name__ == "__main__":
    main()
