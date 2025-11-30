"""
Demo: Skills Metrics Dashboard
================================

Demonstrates the skills metrics dashboard with realistic usage patterns.

Shows:
1. Multiple skill executions with varying success/failure
2. Cache hits/misses
3. Performance variations
4. Dashboard generation from metrics

Created: 2025-11-25
"""

import asyncio
import time
import sys
from pathlib import Path

# Direct import to avoid visualization __init__.py issues
sys.path.insert(0, str(Path(__file__).parent.parent))
from HoloLoom.skills.executor import SkillExecutor
from HoloLoom.skills.metrics import get_tracker

# Import dashboard renderer directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "skills_dashboard",
    Path(__file__).parent.parent / "HoloLoom" / "visualization" / "skills_dashboard.py"
)
skills_dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skills_dashboard)
render_dashboard = skills_dashboard.render_dashboard


def print_section(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60 + '\n')


async def simulate_skill_executions():
    """Simulate realistic skill execution patterns."""
    print_section("Simulating Skill Executions")

    executor = SkillExecutor()

    # Simulate various execution patterns
    test_cases = [
        # Graphviz - mostly successful
        ("graphviz", "render_dot", {"dot_source": "digraph { A -> B; }", "output": "/tmp/test1.png"}, True),
        ("graphviz", "render_dot", {"dot_source": "digraph { A -> B; }", "output": "/tmp/test1.png"}, True),  # Cache hit
        ("graphviz", "render_dot", {"dot_source": "digraph { C -> D; }", "output": "/tmp/test2.png"}, True),
        ("graphviz", "render_neato", {"dot_source": "graph { A -- B; }", "output": "/tmp/test3.png"}, True),

        # PyTest - some failures
        ("pytest_runner", "run_tests", {"path": "/nonexistent"}, False),  # Will fail
        ("pytest_runner", "run_tests", {"path": "/nonexistent"}, False),  # Will fail again
        ("pytest_runner", "discover_tests", {"path": "."}, True),

        # LSP Integration - mix of success/failure
        ("lsp_integration", "analyze_code", {"file": "test.py", "language": "python"}, True),
        ("lsp_integration", "analyze_code", {"file": "test.py", "language": "python"}, True),  # Cache hit
        ("lsp_integration", "get_completions", {"file": "test.py", "position": {"line": 1, "col": 5}}, True),
        ("lsp_integration", "analyze_code", {"file": "invalid.py", "language": "python"}, False),  # Fail

        # File search - fast operations
        ("file_search", "search", {"pattern": "test", "path": "."}, True),
        ("file_search", "search", {"pattern": "test", "path": "."}, True),  # Cache hit
        ("file_search", "list_files", {"path": ".", "file_type": "py"}, True),

        # Image processing - varied timing
        ("image_processing", "get_info", {"input": "/tmp/test.png"}, False),  # File doesn't exist
        ("image_processing", "resize", {"input": "/tmp/test.png", "width": 100}, False),  # File doesn't exist

        # Data format - successful conversions
        ("data_format", "validate_json", {"data": '{"key": "value"}'}, True),
        ("data_format", "validate_json", {"data": '{"key": "value"}'}, True),  # Cache hit
        ("data_format", "pretty_print", {"data": '{"a":1,"b":2}'}, True),
        ("data_format", "convert", {"data": '{"x": 1}', "to_format": "yaml"}, True),

        # Slack bot - simulated API calls
        ("slack_bot", "send_message", {"channel": "test", "text": "Hello"}, True),
        ("slack_bot", "list_channels", {}, True),
        ("slack_bot", "list_channels", {}, True),  # Cache hit
    ]

    print(f"Executing {len(test_cases)} skill operations...\n")

    for i, (skill, operation, params, _) in enumerate(test_cases, 1):
        try:
            result = await executor.execute_single(
                skill_name=skill,
                operation=operation,
                parameters=params,
                use_cache=True
            )

            status = "[OK]" if result.output.status.value == "success" else "[FAIL]"
            cached = "[CACHED]" if result.cache_hit else ""
            print(f"{i:2}. {status} {skill}.{operation} {cached}")

            # Small delay to simulate real usage
            await asyncio.sleep(0.01)

        except Exception as e:
            print(f"{i:2}. [FAIL] {skill}.{operation} - Error: {e}")

    print(f"\n[OK] Completed {len(test_cases)} executions")


async def generate_dashboard():
    """Generate and save the metrics dashboard."""
    print_section("Generating Metrics Dashboard")

    tracker = get_tracker()

    # Export metrics as JSON
    metrics_json = tracker.export_json()

    print("Metrics Summary:")
    summary = metrics_json['summary']
    print(f"  Total Skills: {summary['total_skills']}")
    print(f"  Total Executions: {summary['total_executions']}")
    print(f"  Success Rate: {summary['success_rate']*100:.1f}%")
    print(f"  Avg Time: {summary['avg_time_ms']:.1f}ms")
    print(f"  Cache Hit Rate: {summary['cache_hit_rate']*100:.1f}%")

    # Generate HTML dashboard
    html = render_dashboard(
        metrics_json,
        title="HoloLoom Skills Performance Dashboard"
    )

    # Save to file
    output_path = "demos/output/skills_dashboard.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n[OK] Dashboard saved to: {output_path}")
    print(f"\n  Open in browser to view interactive dashboard")


async def show_detailed_metrics():
    """Show detailed metrics breakdown."""
    print_section("Detailed Metrics Breakdown")

    tracker = get_tracker()

    # Top skills by executions
    top_skills = tracker.get_top_skills(n=5, by="executions")
    print("Top 5 Most Used Skills:")
    for i, metrics in enumerate(top_skills, 1):
        print(f"  {i}. {metrics.skill_name}: {metrics.total_executions} executions")

    # Slow skills
    slow_skills = tracker.get_slow_skills(threshold_ms=100.0)
    if slow_skills:
        print(f"\nSlow Skills (>100ms avg):")
        for metrics in slow_skills:
            print(f"  - {metrics.skill_name}: {metrics.avg_time_ms:.1f}ms avg")
    else:
        print(f"\n[OK] No slow skills detected")

    # Failing skills
    failing_skills = tracker.get_failing_skills(min_failure_rate=0.2)
    if failing_skills:
        print(f"\nFailing Skills (>20% failure rate):")
        for metrics in failing_skills:
            print(f"  - {metrics.skill_name}: {metrics.success_rate*100:.1f}% success")
    else:
        print(f"\n[OK] No failing skills detected")

    # Cache analysis
    all_metrics = tracker.get_all_metrics()
    high_cache_skills = [
        (name, m) for name, m in all_metrics.items()
        if m.cache_hit_rate > 0.5 and (m.cache_hits + m.cache_misses) >= 3
    ]
    if high_cache_skills:
        print(f"\nHigh Cache Efficiency (>50% hit rate):")
        for name, metrics in high_cache_skills:
            print(f"  - {name}: {metrics.cache_hit_rate*100:.1f}% hit rate")


async def main():
    """Run complete dashboard demo."""
    print("="*60)
    print("  Skills Metrics Dashboard Demo")
    print("="*60)

    try:
        # Simulate skill usage
        await simulate_skill_executions()

        # Show detailed metrics
        await show_detailed_metrics()

        # Generate dashboard
        await generate_dashboard()

        print_section("Demo Complete!")
        print("Next steps:")
        print("  1. Open demos/output/skills_dashboard.html in browser")
        print("  2. Explore metrics visualizations")
        print("  3. Identify performance bottlenecks")
        print("  4. Use metrics to optimize skill usage")

    except Exception as e:
        print(f"\n[ERROR] Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
