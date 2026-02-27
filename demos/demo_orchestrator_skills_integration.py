"""
Demo: Skills Integration with HoloLoom Orchestrator
===================================================

Demonstrates the full integration of the skills system with the
HoloLoom weaving orchestrator.

Shows:
1. Automatic skill discovery and tool registration
2. Skill execution through orchestrator tool executor
3. Skills + core tools working together
4. Metrics tracking across the full pipeline

Created: 2025-11-25
"""

import asyncio
import logging
from hololoom.tools.executor import ToolExecutor
from hololoom.tools.skills_bridge import get_skills_bridge
from hololoom.protocols.types import Query, Context

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def print_section(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60 + '\n')


async def demo_skill_discovery():
    """Demo 1: Skill Discovery and Registration"""
    print_section("Demo 1: Skill Discovery")

    bridge = get_skills_bridge()

    # Show all registered skills
    skills = bridge.get_registered_skills()
    print(f"Total skills registered: {len(skills)}\n")

    # Group by category
    by_category = bridge.list_skills_by_category()
    for category, skill_names in sorted(by_category.items()):
        print(f"{category.upper()}:")
        for skill_name in skill_names:
            print(f"  - {skill_name}")
        print()


async def demo_executor_integration():
    """Demo 2: Executor Integration"""
    print_section("Demo 2: Executor Integration")

    executor = ToolExecutor(enable_skills=True)

    print(f"Core tools: {executor.core_tools}")
    print(f"Total tools available: {len(executor.tools)}")
    print(f"\nAll tools:")
    for i, tool in enumerate(executor.tools, 1):
        marker = "[CORE]" if tool in executor.core_tools else "[SKILL]"
        print(f"  {i}. {tool} {marker}")


async def demo_skill_execution():
    """Demo 3: Skill Execution Through Orchestrator"""
    print_section("Demo 3: Skill Execution")

    executor = ToolExecutor(enable_skills=True)

    # Create mock query and context
    query = Query(text="Generate a simple diagram")
    context = Context(memories=[], metadata={})

    # Execute graphviz skill (if available and Graphviz installed)
    if "graphviz" in executor.tools:
        print("Executing graphviz skill through orchestrator...\n")

        # Note: This will fail with "execute" operation since graphviz
        # expects specific operations like "render_dot"
        # In production, we'd enhance _handle_skill to parse the query
        # and determine the correct operation

        result = await executor.execute("graphviz", query, context)

        print(f"Tool: {result.get('tool')}")
        print(f"Status: {result.get('status')}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print(f"Cached: {result.get('cached', False)}")

        if result.get('status') == 'error':
            print(f"Error: {result.get('error')}")
            print("\nNote: This error is expected - the generic handler")
            print("      needs enhancement to map queries to skill operations.")
        else:
            print(f"Result: {result.get('result')}")


async def demo_core_and_skills():
    """Demo 4: Core Tools + Skills Working Together"""
    print_section("Demo 4: Core Tools + Skills Together")

    executor = ToolExecutor(enable_skills=True)

    # Create test query and context
    query = Query(text="Explain this code")
    context = Context(memories=["def hello(): print('world')"], metadata={})

    # Execute core "answer" tool
    print("1. Executing core 'answer' tool...\n")
    answer_result = await executor.execute("answer", query, context)
    print(f"   Tool: {answer_result['tool']}")
    print(f"   Confidence: {answer_result['confidence']:.2f}")
    print(f"   Result preview: {answer_result['result'][:100]}...")

    # Execute skill tool (if any available)
    if "file_search" in executor.tools:
        print("\n2. Executing 'file_search' skill...\n")

        # This will also fail with generic operation, but shows the dispatch
        search_result = await executor.execute("file_search", query, context)
        print(f"   Tool: {search_result['tool']}")
        print(f"   Status: {search_result['status']}")
        print(f"   Confidence: {search_result['confidence']:.2f}")


async def demo_metrics():
    """Demo 5: Metrics Across Integration"""
    print_section("Demo 5: Metrics Integration")

    bridge = get_skills_bridge()

    # Get current metrics summary
    metrics = bridge.get_metrics_summary()

    print("Skills Metrics Summary:")
    print(f"  Total executions: {metrics.get('total_executions', 0)}")

    if metrics.get('total_executions', 0) > 0:
        print(f"  Success rate: {metrics.get('success_rate', 0):.1%}")
        print(f"  Avg time: {metrics.get('avg_time_ms', 0):.1f}ms")
        print(f"  Cache hit rate: {metrics.get('cache_hit_rate', 0):.1%}")
    else:
        print("  (No skill executions yet)")


async def demo_skill_info():
    """Demo 6: Skill Information"""
    print_section("Demo 6: Skill Information")

    bridge = get_skills_bridge()
    skills = bridge.get_registered_skills()

    # Show info for first skill
    if skills:
        skill_name = list(skills.keys())[0]
        info = bridge.get_skill_info(skill_name)

        print(f"Skill: {info['name']}")
        print(f"Version: {info['version']}")
        print(f"Author: {info['author']}")
        print(f"Category: {info['category']}")
        print(f"Tags: {', '.join(info['tags'])}")
        print(f"\nDescription: {info['description_short']}")
        print(f"\nRequirements:")
        print(f"  Code execution: {info['requires_code_execution']}")
        print(f"  Filesystem read: {info['requires_filesystem_read']}")
        print(f"  Filesystem write: {info['requires_filesystem_write']}")
        print(f"\nDependencies: {', '.join(info['external_dependencies'])}")
        print(f"Expected latency: {info['expected_latency_ms']}")
        print(f"Token usage: {info['token_usage']}")


async def main():
    """Run all demos."""
    print("="*60)
    print("  HoloLoom Skills + Orchestrator Integration Demo")
    print("="*60)

    try:
        await demo_skill_discovery()
        await demo_executor_integration()
        await demo_skill_execution()
        await demo_core_and_skills()
        await demo_metrics()
        await demo_skill_info()

        print_section("All demos completed!")

        print("\nKey Takeaways:")
        print("  1. Skills automatically discovered and registered")
        print("  2. Executor seamlessly integrates skills as tools")
        print("  3. Core tools + skills work together")
        print("  4. Metrics track performance across both")
        print("\nNext Steps:")
        print("  - Enhance _handle_skill to parse queries into operations")
        print("  - Add NLU to extract structured parameters from text")
        print("  - Integrate with full weaving orchestrator pipeline")

    except Exception as e:
        print(f"\n[X] Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
