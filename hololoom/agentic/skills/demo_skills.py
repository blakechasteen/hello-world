#!/usr/bin/env python3
"""
Simple Skills Demo
==================
Demonstrates basic usage of the HoloLoom Skills System.

This demo shows how to:
1. List available skills
2. Execute a skill with parameters
3. Access execution results

Usage:
    PYTHONPATH=. python hololoom/agentic/skills/demo_skills.py
"""

import asyncio
import sys
from pathlib import Path

# Add repository root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hololoom.agentic import execute_skill, list_available_skills
from hololoom.config import Config


async def main():
    """Run simple skills demo."""
    print("="*80)
    print("HoloLoom Skills System - Demo")
    print("="*80)
    print()

    # Step 1: List available skills
    print("Step 1: Listing available skills...")
    print()

    try:
        skills_by_category = await list_available_skills()

        for category, skill_names in skills_by_category.items():
            print(f"{category.upper()}:")
            for skill_name in skill_names:
                print(f"  - {skill_name}")
            print()

        total_skills = sum(len(names) for names in skills_by_category.values())
        print(f"Total: {total_skills} skills available")
        print()

    except Exception as e:
        print(f"ERROR: Could not list skills: {e}")
        print()

    # Step 2: Execute a simple skill
    print("="*80)
    print("Step 2: Executing 'code_reviewer' skill...")
    print("="*80)
    print()

    sample_code = """
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price']
    return total
"""

    print(f"Sample code to review:")
    print(sample_code)
    print()

    try:
        result = await execute_skill(
            skill_name="code_reviewer",
            parameters={
                "code": sample_code,
                "language": "python"
            },
            config=Config.bare()  # Fastest mode for demo
        )

        print("="*80)
        print("Execution Result:")
        print("="*80)
        print()

        if result.success:
            print(f"SUCCESS!")
            print()
            print(f"Output:")
            print("-"*80)
            print(result.output)
            print("-"*80)
            print()
            print(f"Metadata:")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Strategy: {result.strategy_used}")
            print(f"  Execution time: {result.execution_time_ms:.1f}ms")
            print()
        else:
            print(f"FAILED: {result.error}")
            print()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()

    print("="*80)
    print("Demo Complete!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
