"""
Demo: Advanced Skill Features (Phase 3)
========================================

Demonstrates:
1. SkillCache - Intelligent result caching with LRU eviction
2. SkillExecutor - Parallel execution with caching
3. SkillWorkflow - Skill composition and chaining
4. SkillMetricsTracker - Performance monitoring

This shows the full power of the Phase 3 skill system.
"""

import asyncio
import json
from HoloLoom.skills.base import SkillInput, get_registry
from HoloLoom.skills.cache import SkillCache
from HoloLoom.skills.executor import SkillExecutor
from HoloLoom.skills.composition import (
    SkillWorkflow,
    execute_step,
    parallel_step,
    conditional_step,
    transform_step
)
from HoloLoom.skills.metrics import get_tracker

# Import some skills (they register themselves)
from HoloLoom.skills.creative import graphviz_skill
from HoloLoom.skills.testing import pytest_runner


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def demo_cache():
    """Demo 1: Intelligent caching with LRU eviction."""
    print_section("Demo 1: Skill Caching")

    cache = SkillCache(max_size=3, default_ttl=3600.0)

    # Simulate some skill results
    from HoloLoom.skills.base import SkillOutput, SkillStatus

    # Cache 3 results
    print(" Caching 3 results...")
    for i in range(3):
        result = SkillOutput(
            status=SkillStatus.SUCCESS,
            result=f"Result {i}",
            message=f"Success {i}",
            execution_time_ms=100.0
        )
        cache.put(f"skill_{i}", "test", {"value": i}, result)

    stats = cache.get_stats()
    print(f"Cache size: {stats['size']}/{stats['max_size']}")

    # Access result 0 and 1 (make them recently used)
    print("\n Accessing results 0 and 1 (LRU tracking)...")
    cache.get("skill_0", "test", {"value": 0})
    cache.get("skill_1", "test", {"value": 1})

    # Add 4th result - should evict skill_2 (least recently used)
    print("\n[+] Adding 4th result (triggers LRU eviction)...")
    result = SkillOutput(
        status=SkillStatus.SUCCESS,
        result="Result 3",
        message="Success 3",
        execution_time_ms=100.0
    )
    cache.put("skill_3", "test", {"value": 3}, result)

    stats = cache.get_stats()
    print(f"Cache size: {stats['size']}/{stats['max_size']}")
    print(f"Evictions: {stats['evictions']}")

    # Verify skill_2 was evicted
    print("\n[v] Verifying LRU eviction...")
    result_0 = cache.get("skill_0", "test", {"value": 0})
    result_2 = cache.get("skill_2", "test", {"value": 2})  # Should be None (evicted)
    result_3 = cache.get("skill_3", "test", {"value": 3})

    print(f"skill_0 in cache: {result_0 is not None}")
    print(f"skill_2 in cache: {result_2 is not None} (evicted OK)")
    print(f"skill_3 in cache: {result_3 is not None}")

    stats = cache.get_stats()
    print(f"\nFinal cache stats:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")


async def demo_executor():
    """Demo 2: Parallel skill execution with caching."""
    print_section("Demo 2: Parallel Execution")

    executor = SkillExecutor()

    # Define multiple executions (graphviz renders)
    executions = [
        {
            "skill_name": "graphviz",
            "operation": "render",
            "parameters": {
                "dot_source": "digraph { A -> B; B -> C; }",
                "format": "png",
                "output_path": "/tmp/graph1.png"
            }
        },
        {
            "skill_name": "graphviz",
            "operation": "render",
            "parameters": {
                "dot_source": "digraph { X -> Y; Y -> Z; }",
                "format": "png",
                "output_path": "/tmp/graph2.png"
            }
        },
        {
            "skill_name": "graphviz",
            "operation": "render",
            "parameters": {
                "dot_source": "digraph { 1 -> 2; 2 -> 3; }",
                "format": "png",
                "output_path": "/tmp/graph3.png"
            }
        }
    ]

    # First run - all cold (no cache)
    print(" First run (cold cache)...")
    result1 = await executor.execute_parallel(executions)

    print(f"Results: {result1.success_count}/{len(result1.results)} successful")
    print(f"Total time: {result1.total_time_ms:.1f}ms")
    print(f"Cache hit rate: {result1.cache_hit_rate:.1%}")

    # Second run - all hot (cached)
    print("\n Second run (warm cache)...")
    result2 = await executor.execute_parallel(executions)

    print(f"Results: {result2.success_count}/{len(result2.results)} successful")
    print(f"Total time: {result2.total_time_ms:.1f}ms")
    print(f"Cache hit rate: {result2.cache_hit_rate:.1%}")

    speedup = result1.total_time_ms / result2.total_time_ms if result2.total_time_ms > 0 else 0
    print(f"\n[lightning] Speedup: {speedup:.1f}x faster with caching!")

    # Executor statistics
    print("\n Executor statistics:")
    stats = executor.get_statistics()
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Average time: {stats['avg_time_ms']:.1f}ms")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")


async def demo_workflow():
    """Demo 3: Skill composition with workflows."""
    print_section("Demo 3: Skill Workflows")

    executor = SkillExecutor()

    # Define a workflow: test → visualization
    workflow = SkillWorkflow(
        name="test_and_visualize",
        steps=[
            # Step 1: Run pytest
            execute_step(
                name="test_result",
                skill_name="pytest",
                operation="run_tests",
                parameters={
                    "test_path": "HoloLoom/tests",
                    "markers": []
                }
            ),

            # Step 2: Transform result into graph data
            transform_step(
                name="prepare_graph",
                transform_func=lambda ctx: {
                    "graph_data": {
                        "nodes": ["Tests", "Passed", "Failed"],
                        "edges": [("Tests", "Passed"), ("Tests", "Failed")]
                    }
                }
            ),

            # Step 3: Conditional - only create graph if we have data
            conditional_step(
                name="create_visualization",
                condition=lambda ctx: "graph_data" in ctx,
                if_true=[
                    execute_step(
                        name="graph_image",
                        skill_name="graphviz",
                        operation="render",
                        parameters={
                            "dot_source": "digraph { A -> B; B -> C; }",
                            "format": "png",
                            "output_path": "/tmp/workflow_graph.png"
                        }
                    )
                ],
                if_false=[]
            )
        ],
        executor=executor
    )

    # Execute workflow
    print(" Executing workflow: search_and_visualize")
    result = await workflow.execute()

    print(f"\nWorkflow status: {'[v] Success' if result.success else '[X] Failed'}")
    print(f"Steps executed: {result.steps_executed}")
    print(f"Total time: {result.total_time_ms:.1f}ms")

    if result.errors:
        print(f"Errors: {result.errors}")

    print(f"\n Workflow outputs:")
    for key, value in result.outputs.items():
        if isinstance(value, dict):
            print(f"  {key}: {json.dumps(value, indent=2)[:100]}...")
        else:
            print(f"  {key}: {str(value)[:100]}...")


async def demo_parallel_workflow():
    """Demo 4: Parallel execution in workflows."""
    print_section("Demo 4: Parallel Workflow Steps")

    executor = SkillExecutor()

    # Workflow with parallel steps - multiple graph renders
    workflow = SkillWorkflow(
        name="parallel_renders",
        steps=[
            # Execute 3 renders in parallel
            parallel_step(
                name="parallel_renders",
                steps=[
                    execute_step(
                        name="render_ml",
                        skill_name="graphviz",
                        operation="render",
                        parameters={
                            "dot_source": "digraph { ML -> DL; DL -> NLP; }",
                            "format": "png",
                            "output_path": "/tmp/parallel1.png"
                        }
                    ),
                    execute_step(
                        name="render_dl",
                        skill_name="graphviz",
                        operation="render",
                        parameters={
                            "dot_source": "digraph { Input -> Hidden; Hidden -> Output; }",
                            "format": "png",
                            "output_path": "/tmp/parallel2.png"
                        }
                    ),
                    execute_step(
                        name="render_rl",
                        skill_name="graphviz",
                        operation="render",
                        parameters={
                            "dot_source": "digraph { State -> Action; Action -> Reward; }",
                            "format": "png",
                            "output_path": "/tmp/parallel3.png"
                        }
                    )
                ]
            )
        ],
        executor=executor
    )

    print(" Executing parallel workflow...")
    result = await workflow.execute()

    print(f"\nWorkflow status: {'[v] Success' if result.success else '[X] Failed'}")
    print(f"Steps executed: {result.steps_executed}")
    print(f"Total time: {result.total_time_ms:.1f}ms")

    # Show parallel execution details
    for step_result in result.step_results:
        if step_result.get("step_type") == "parallel":
            print(f"\n Parallel execution stats:")
            print(f"  Parallel tasks: {step_result.get('parallel_count', 0)}")
            print(f"  Successes: {step_result.get('success_count', 0)}")
            print(f"  Failures: {step_result.get('failure_count', 0)}")
            print(f"  Cache hit rate: {step_result.get('cache_hit_rate', 0):.1%}")


async def demo_metrics():
    """Demo 5: Performance metrics tracking."""
    print_section("Demo 5: Metrics Tracking")

    # Get global metrics tracker
    tracker = get_tracker()
    tracker.reset()  # Start fresh

    # Execute some skills to generate metrics
    executor = SkillExecutor()

    print(" Executing skills to generate metrics...")

    # Run graphviz multiple times
    for i in range(5):
        await executor.execute_single(
            skill_name="graphviz",
            operation="render",
            parameters={
                "dot_source": f"digraph {{ Node{i} -> Node{i+1}; }}",
                "format": "png",
                "output_path": f"/tmp/metrics_{i}.png"
            }
        )

    # Get skill metrics
    print("\n Skill Metrics:")
    metrics = tracker.get_skill_metrics("graphviz")
    if metrics:
        print(f"  Skill: {metrics.skill_name}")
        print(f"  Total executions: {metrics.total_executions}")
        print(f"  Success rate: {metrics.success_rate:.1%}")
        print(f"  Avg time: {metrics.avg_time_ms:.1f}ms")
        print(f"  Median time: {metrics.median_time_ms:.1f}ms")
        print(f"  Min time: {metrics.min_time_ms:.1f}ms")
        print(f"  Max time: {metrics.max_time_ms:.1f}ms")
        print(f"  Cache hit rate: {metrics.cache_hit_rate:.1%}")

    # Get summary
    print("\n Summary Metrics:")
    summary = tracker.get_summary()
    print(f"  Total skills: {summary['total_skills']}")
    print(f"  Total executions: {summary['total_executions']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Avg time: {summary['avg_time_ms']:.1f}ms")
    print(f"  Cache hit rate: {summary['cache_hit_rate']:.1%}")

    # Export metrics as JSON
    print("\n Exporting metrics to JSON...")
    export = tracker.export_json()
    print(f"  Exported {len(export['skills'])} skill metrics")
    print(f"  Timestamp: {export['timestamp']}")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("  Advanced Skills Demo - Phase 3 Features")
    print("="*60)
    print("\nDemonstrating:")
    print("  1. Intelligent caching with LRU eviction")
    print("  2. Parallel skill execution")
    print("  3. Skill composition (workflows)")
    print("  4. Parallel workflow steps")
    print("  5. Performance metrics tracking")

    try:
        await demo_cache()
        await demo_executor()
        await demo_workflow()
        await demo_parallel_workflow()
        await demo_metrics()

        print("\n" + "="*60)
        print("  [v] All Phase 3 demos completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n[X] Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
