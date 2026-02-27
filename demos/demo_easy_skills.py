"""
Demo: Easy Skill Launching for HoloLoom
========================================

**Created**: December 30, 2025
**Purpose**: Demonstrate the one-line skill launching API

This demo shows:
1. Creating skills with @skill decorator
2. Launching skills with one line
3. Parallel and chain execution
4. Skill discovery
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.skills import (
    # Easy skill creation
    skill,
    code_skill,
    domain_skill,

    # Easy launching
    launch_skill,
    launch,
    launch_parallel,
    launch_chain,

    # Discovery
    list_skills,
    get_skill_info,
    get_launch_stats,

    # Builder API
    SkillBuilder,
)


# ============================================================================
# Demo 1: Create Skills with Decorators
# ============================================================================

@skill(name="hello_world", category="domain", tags=["demo", "greeting"])
async def greet(name: str = "World") -> dict:
    """A simple greeting skill."""
    return {
        "greeting": f"Hello, {name}!",
        "skill": "hello_world"
    }


@code_skill(name="line_counter", tags=["code", "analysis"])
async def count_lines(code: str) -> dict:
    """Count lines in code."""
    lines = code.strip().split('\n')
    return {
        "total_lines": len(lines),
        "non_empty_lines": len([l for l in lines if l.strip()]),
        "code_preview": lines[0] if lines else ""
    }


@domain_skill(name="sentiment_analyzer", tags=["nlp", "sentiment"])
def analyze_sentiment(text: str) -> dict:
    """Simple sentiment analysis (sync function)."""
    positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'awesome']
    negative_words = ['bad', 'terrible', 'hate', 'awful', 'poor', 'sad']

    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    if pos_count > neg_count:
        sentiment = "positive"
        score = min(1.0, 0.5 + pos_count * 0.1)
    elif neg_count > pos_count:
        sentiment = "negative"
        score = max(0.0, 0.5 - neg_count * 0.1)
    else:
        sentiment = "neutral"
        score = 0.5

    return {
        "sentiment": sentiment,
        "score": score,
        "positive_matches": pos_count,
        "negative_matches": neg_count
    }


@skill(name="math_calculator", category="system")
async def calculate(expression: str) -> dict:
    """Safely evaluate a math expression."""
    try:
        # Only allow safe operations
        allowed = set('0123456789+-*/().% ')
        if not all(c in allowed for c in expression):
            return {"error": "Invalid characters in expression", "result": None}

        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "result": None}


# ============================================================================
# Demo 2: Create Skills with Builder API
# ============================================================================

def word_counter(text: str) -> dict:
    """Count words in text."""
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "unique_words": len(set(words))
    }


# Build and register
word_count_skill = (SkillBuilder()
    .name("word_counter")
    .category("domain")
    .description("Count words in text")
    .tags(["text", "analysis"])
    .handler(word_counter)
    .build())


# ============================================================================
# Demo Execution
# ============================================================================

async def demo_1_basic_launching():
    """Demo 1: Basic skill launching."""
    print("\n" + "="*70)
    print("Demo 1: Basic Skill Launching")
    print("="*70)

    # Launch greeting skill
    print("\n1. Launch greeting skill:")
    result = await launch_skill("hello_world", name="HoloLoom User")
    print(f"   Success: {result.success}")
    print(f"   Result: {result.result}")
    print(f"   Time: {result.execution_time_ms:.1f}ms")

    # Launch code skill
    print("\n2. Launch code analysis skill:")
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""
    result = await launch("line_counter", code=code)
    print(f"   Success: {result.success}")
    print(f"   Result: {result.result}")

    # Launch sentiment skill (sync function)
    print("\n3. Launch sentiment skill (sync function wrapped):")
    result = await launch("sentiment_analyzer", text="This is an awesome and great demo!")
    print(f"   Success: {result.success}")
    print(f"   Sentiment: {result.result}")

    # Launch math skill
    print("\n4. Launch math skill:")
    result = await launch("math_calculator", expression="2 + 3 * 4")
    print(f"   Success: {result.success}")
    print(f"   Result: {result.result}")


async def demo_2_parallel_execution():
    """Demo 2: Parallel skill execution."""
    print("\n" + "="*70)
    print("Demo 2: Parallel Skill Execution")
    print("="*70)

    # Run multiple skills in parallel
    print("\n Running 3 skills in parallel...")

    results = await launch_parallel(
        {"skill_name": "hello_world", "parameters": {"name": "Alice"}},
        {"skill_name": "hello_world", "parameters": {"name": "Bob"}},
        {"skill_name": "sentiment_analyzer", "parameters": {"text": "Great day!"}}
    )

    for i, result in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"   Skill: {result.skill_name}")
        print(f"   Success: {result.success}")
        print(f"   Result: {result.result}")


async def demo_3_chain_execution():
    """Demo 3: Chain skill execution."""
    print("\n" + "="*70)
    print("Demo 3: Chain Skill Execution")
    print("="*70)

    # Run skills sequentially
    print("\n Running skills in chain...")

    results = await launch_chain(
        {"skill_name": "hello_world", "parameters": {"name": "Step 1"}},
        {"skill_name": "math_calculator", "parameters": {"expression": "1 + 1"}},
        {"skill_name": "word_counter", "parameters": {"text": "Chain execution complete"}},
        stop_on_failure=True
    )

    for i, result in enumerate(results, 1):
        print(f"\n   Step {i} ({result.skill_name}):")
        print(f"   Success: {result.success}")
        print(f"   Result: {result.result}")


async def demo_4_skill_discovery():
    """Demo 4: Skill discovery."""
    print("\n" + "="*70)
    print("Demo 4: Skill Discovery")
    print("="*70)

    # List all skills
    print("\n1. All registered skills:")
    skills = list_skills()
    for skill_name in skills:
        print(f"   - {skill_name}")

    # Get skill info
    print("\n2. Skill info for 'hello_world':")
    info = get_skill_info("hello_world")
    if info:
        print(f"   Name: {info['name']}")
        print(f"   Version: {info['version']}")
        print(f"   Category: {info['category']}")
        print(f"   Description: {info['description']}")
        print(f"   Tags: {info['tags']}")

    # Get launch stats
    print("\n3. Launch statistics:")
    stats = get_launch_stats()
    print(f"   Total executions: {stats['total_executions']}")
    print(f"   Avg time: {stats['avg_time_ms']:.1f}ms")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")


async def demo_5_error_handling():
    """Demo 5: Error handling."""
    print("\n" + "="*70)
    print("Demo 5: Error Handling")
    print("="*70)

    # Try to launch non-existent skill
    print("\n1. Launch non-existent skill:")
    result = await launch("nonexistent_skill", param="value")
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")

    # Launch with invalid expression
    print("\n2. Launch with invalid input:")
    result = await launch("math_calculator", expression="import os")
    print(f"   Success: {result.success}")
    print(f"   Result: {result.result}")


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("HoloLoom Easy Skill Launching Demo")
    print("="*70)

    await demo_1_basic_launching()
    await demo_2_parallel_execution()
    await demo_3_chain_execution()
    await demo_4_skill_discovery()
    await demo_5_error_handling()

    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70)
    print("\nUsage Summary:")
    print("  from hololoom.skills import launch_skill, skill")
    print("  ")
    print("  @skill(name='my_skill')")
    print("  async def my_skill(param: str) -> dict:")
    print("      return {'result': param}")
    print("  ")
    print("  result = await launch_skill('my_skill', param='value')")
    print("="*70)


if __name__ == '__main__':
    asyncio.run(main())
