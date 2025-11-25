#!/usr/bin/env python3
"""
Code Reviewer Skill Demo
=========================
Demonstrates using the code-reviewer skill from HoloLoom's Skills System.

This demo shows:
1. Basic code review
2. Review with different code quality levels
3. Accessing detailed results
4. Performance characteristics

Run: PYTHONPATH=. python demos/demo_code_reviewer.py
"""

import asyncio
import time
from pathlib import Path

from HoloLoom.agentic import execute_skill, list_available_skills
from HoloLoom.config import Config


async def demo_basic_review():
    """Demonstrate basic code review functionality."""
    print("=" * 80)
    print("DEMO 1: Basic Code Review")
    print("=" * 80)

    sample_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total / len(numbers)
"""

    print("\nCode to review:")
    print(sample_code)

    start = time.time()
    result = await execute_skill(
        skill_name="code-reviewer",
        parameters={
            "code": sample_code,
            "language": "python"
        },
        config=Config.fast()
    )
    duration = time.time() - start

    print("\n" + "=" * 80)
    print("REVIEW RESULTS:")
    print("=" * 80)
    print(result.output)

    print("\n" + "=" * 80)
    print("METADATA:")
    print("=" * 80)
    print(f"✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")
    print(f"✓ Iterations: {result.iterations}")
    print(f"✓ Execution Time: {duration:.2f}s ({result.execution_time_ms:.1f}ms)")


async def demo_code_with_issues():
    """Demonstrate reviewing code with common issues."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Code with Multiple Issues")
    print("=" * 80)

    problematic_code = """
def ProcessData(data):
    result = []
    for i in range(0, len(data)):
        if data[i] != None:
            result.append(data[i] * 2)
    return result

# TODO: Add error handling
"""

    print("\nCode with issues to review:")
    print(problematic_code)

    result = await execute_skill(
        skill_name="code-reviewer",
        parameters={
            "code": problematic_code,
            "language": "python"
        },
        config=Config.fused()  # Use FUSED for thorough review
    )

    print("\n" + "=" * 80)
    print("REVIEW RESULTS:")
    print("=" * 80)
    print(result.output)

    print("\n" + "=" * 80)
    print("METADATA:")
    print("=" * 80)
    print(f"✓ Confidence: {result.confidence:.2f}")
    print(f"✓ Iterations: {result.iterations} (FUSED mode)")


async def demo_real_world_code():
    """Demonstrate reviewing real-world-like code."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Real-World API Endpoint")
    print("=" * 80)

    api_code = """
async def create_user(request: Request):
    '''Create a new user account.'''
    data = await request.json()

    # Validate email
    if '@' not in data['email']:
        return {'error': 'Invalid email'}

    # Check if user exists
    existing = await db.query(
        f"SELECT * FROM users WHERE email = '{data['email']}'"
    )

    if existing:
        return {'error': 'User exists'}

    # Create user
    user_id = await db.execute(
        f"INSERT INTO users (email, password) VALUES ('{data['email']}', '{data['password']}')"
    )

    return {'user_id': user_id}
"""

    print("\nAPI endpoint code to review:")
    print(api_code)

    result = await execute_skill(
        skill_name="code-reviewer",
        parameters={
            "code": api_code,
            "language": "python"
        },
        config=Config.fused()
    )

    print("\n" + "=" * 80)
    print("REVIEW RESULTS:")
    print("=" * 80)
    print(result.output)

    print("\n" + "=" * 80)
    print("Expected Issues Found:")
    print("=" * 80)
    print("✓ SQL Injection vulnerability (f-strings in queries)")
    print("✓ Plain text password storage")
    print("✓ Missing input validation")
    print("✓ No error handling")
    print("✓ Missing authentication check")


async def demo_compare_modes():
    """Compare code review across different HoloLoom modes."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Performance Comparison (BARE vs FAST vs FUSED)")
    print("=" * 80)

    sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

    configs = [
        ("BARE", Config.bare()),
        ("FAST", Config.fast()),
        ("FUSED", Config.fused())
    ]

    results = []

    for mode_name, config in configs:
        print(f"\nRunning in {mode_name} mode...")
        start = time.time()

        result = await execute_skill(
            skill_name="code-reviewer",
            parameters={
                "code": sample_code,
                "language": "python"
            },
            config=config
        )

        duration = time.time() - start
        results.append((mode_name, result, duration))

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON:")
    print("=" * 80)
    print(f"{'Mode':<10} {'Confidence':<12} {'Iterations':<12} {'Time (ms)':<12}")
    print("-" * 80)

    for mode_name, result, duration in results:
        print(f"{mode_name:<10} {result.confidence:<12.2f} {result.iterations:<12} {duration * 1000:<12.1f}")


async def demo_skill_discovery():
    """Demonstrate discovering available skills."""
    print("\n\n" + "=" * 80)
    print("DEMO 5: Skill Discovery")
    print("=" * 80)

    skills = await list_available_skills()

    print("\nAll Available Skills by Category:")
    print("=" * 80)

    for category, skill_list in sorted(skills.items()):
        print(f"\n{category.upper()}:")
        for skill in skill_list:
            print(f"  - {skill}")

    total = sum(len(skill_list) for skill_list in skills.values())
    print(f"\nTotal: {total} skills available")


async def main():
    """Run all demos."""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CODE REVIEWER SKILL DEMO" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        # Run all demos in sequence
        await demo_basic_review()
        await demo_code_with_issues()
        await demo_real_world_code()
        await demo_compare_modes()
        await demo_skill_discovery()

        print("\n\n" + "=" * 80)
        print("ALL DEMOS COMPLETE!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("✓ code-reviewer skill provides comprehensive code analysis")
        print("✓ Reviews cover correctness, efficiency, readability, maintainability")
        print("✓ FUSED mode provides most thorough reviews (~300ms)")
        print("✓ FAST mode provides good balance (~150ms)")
        print("✓ BARE mode provides quick reviews (~50ms)")
        print("✓ Multi-pass refinement improves quality automatically")
        print("\nNext Steps:")
        print("• Try other skills: bug-detective, refactoring-expert, test-generator")
        print("• Create custom skills with SKILLS_USAGE.md guide")
        print("• Integrate skills into your development workflow")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
