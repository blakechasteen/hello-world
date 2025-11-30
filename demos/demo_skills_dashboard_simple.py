"""
Simple Skills Metrics Dashboard Demo
=======================================

Quick demo that generates the dashboard with minimal executions.

Created: 2025-11-25
"""

import asyncio
from pathlib import Path
import sys

# Direct import to avoid visualization __init__.py issues
sys.path.insert(0, str(Path(__file__).parent.parent))
from HoloLoom.skills.executor import SkillExecutor
from HoloLoom.skills.metrics import get_tracker
from HoloLoom.skills.base import SkillStatus

# Import dashboard renderer directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "skills_dashboard",
    Path(__file__).parent.parent / "HoloLoom" / "visualization" / "skills_dashboard.py"
)
skills_dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skills_dashboard)
render_dashboard = skills_dashboard.render_dashboard


async def main():
    """Quick dashboard generation demo."""
    print("Skills Metrics Dashboard - Simple Demo")
    print("=" * 60)

    # Create executor
    executor = SkillExecutor()

    # Execute a few skills to generate some metrics
    print("\nExecuting sample skills...")

    test_cases = [
        ("graphviz", "render_dot", {"dot_source": "digraph { A -> B; }", "output": "/tmp/test.png"}),
        ("graphviz", "render_dot", {"dot_source": "digraph { A -> B; }", "output": "/tmp/test.png"}),  # Cache hit
        ("data_format", "validate_json", {"data": '{"key": "value"}'}),
        ("data_format", "pretty_print", {"data": '{"a":1}'}),
    ]

    import time
    tracker = get_tracker()

    for skill, operation, params in test_cases:
        start = time.time()
        try:
            result = await executor.execute_single(
                skill_name=skill,
                operation=operation,
                parameters=params,
                use_cache=True
            )
            execution_time = (time.time() - start) * 1000  # Convert to ms

            # Determine status enum
            if result.output.status == SkillStatus.SUCCESS:
                status_enum = SkillStatus.SUCCESS
            elif result.output.status == SkillStatus.TIMEOUT:
                status_enum = SkillStatus.TIMEOUT
            else:
                status_enum = SkillStatus.ERROR

            cached = result.cache_hit

            # Record to metrics tracker with correct parameters
            print(f"    Recording: skill={skill}, operation={operation}, status={status_enum}, time={execution_time:.1f}ms, cached={cached}")
            tracker.record_execution(skill, operation, status_enum, execution_time, cached)

            status_label = "[OK]" if status_enum == SkillStatus.SUCCESS else "[FAIL]"
            cache_label = "[CACHED]" if cached else ""
            print(f"  {status_label} {skill}.{operation} {cache_label}")
        except Exception as e:
            execution_time = (time.time() - start) * 1000
            tracker.record_execution(skill, operation, SkillStatus.ERROR, execution_time, False)
            print(f"  [FAIL] {skill}.{operation} - {e}")

    # Get metrics
    tracker = get_tracker()
    metrics_json = tracker.export_json()

    print("\nMetrics Summary:")
    summary = metrics_json['summary']
    print(f"  Skills: {summary.get('total_skills', 0)}")
    print(f"  Executions: {summary.get('total_executions', 0)}")

    if summary.get('total_executions', 0) > 0:
        print(f"  Success Rate: {summary.get('success_rate', 0)*100:.1f}%")
        print(f"  Cache Hit Rate: {summary.get('cache_hit_rate', 0)*100:.1f}%")
    else:
        print("  (No executions recorded)")

    # Generate dashboard
    print("\nGenerating dashboard...")
    html = render_dashboard(metrics_json, title="Skills Performance Dashboard")

    # Save to file
    output_path = "demos/output/skills_dashboard.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n[OK] Dashboard saved to: {output_path}")
    print(f"\nOpen {output_path} in your browser to view the dashboard!")


if __name__ == "__main__":
    asyncio.run(main())
