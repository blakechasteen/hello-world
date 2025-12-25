#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow Evaluation Demo
========================

Demonstrates the workflow evaluation framework by running benchmarks
on the example workflows.

Usage:
    python demo_workflow_eval.py

Created: 2025-12-23
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

from workflow_evaluation import (
    WorkflowEvaluator,
    WorkflowVersionControl,
    ABTestRunner,
    ABTestConfig,
    BenchmarkDataset,
    TestCase
)


async def load_example_workflows():
    """Load all example workflows from the advanced directory."""
    workflows = {}
    examples_dir = Path(__file__).parent / "example_workflows" / "advanced"

    for workflow_path in examples_dir.glob("*.json"):
        with open(workflow_path) as f:
            workflow = json.load(f)
            workflows[workflow_path.stem] = workflow

    return workflows


async def evaluate_single_workflow(evaluator, workflow, test_cases, workflow_name):
    """Evaluate a single workflow and return results."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {workflow.get('name', workflow_name)}")
    print(f"{'='*60}")
    print(f"Description: {workflow.get('description', 'N/A')[:100]}...")
    print(f"Nodes: {len(workflow.get('nodes', []))}")
    print(f"Connections: {len(workflow.get('connections', []))}")
    print(f"Test cases: {len(test_cases)}")

    report = await evaluator.evaluate_workflow(workflow, test_cases, parallel=True)

    print(f"\n[Results]")
    print(f"   Success Rate: {report.success_rate:.1%}")
    print(f"   Avg Latency: {report.avg_latency_ms:.1f}ms")
    print(f"   P95 Latency: {report.p95_latency_ms:.1f}ms")
    print(f"   Avg Confidence: {report.avg_confidence:.3f}")
    print(f"   Cache Hit Rate: {report.cache_hit_rate:.1%}")

    return report


async def run_ab_test(runner, workflow_a, workflow_b, test_cases):
    """Run A/B test between two workflows."""
    print(f"\n{'='*60}")
    print(f"A/B Test: {workflow_a.get('name')} vs {workflow_b.get('name')}")
    print(f"{'='*60}")

    config = ABTestConfig(
        name=f"AB Test: {workflow_a.get('name')} vs {workflow_b.get('name')}",
        workflow_a=workflow_a,
        workflow_b=workflow_b,
        test_cases=test_cases,
        traffic_split=0.5,
        min_samples=5
    )

    result = await runner.run_test(config)

    print(f"\n[A/B Test Results]")
    print(f"   Samples A: {result.samples_a}, Samples B: {result.samples_b}")
    print(f"   Latency A: {result.latency_a_avg:.1f}ms, B: {result.latency_b_avg:.1f}ms ({result.latency_improvement:+.1f}%)")
    print(f"   Confidence A: {result.confidence_a_avg:.3f}, B: {result.confidence_b_avg:.3f} ({result.confidence_improvement:+.1f}%)")
    print(f"   Success A: {result.success_rate_a:.1%}, B: {result.success_rate_b:.1%}")
    print(f"   Statistically Significant: {result.is_significant} (p={result.p_value:.4f})")
    print(f"   Winner: {result.winner or 'No clear winner'}")
    print(f"\n   >> Recommendation: {result.recommendation}")

    return result


async def demo_version_control(workflows):
    """Demonstrate version control features."""
    print(f"\n{'='*60}")
    print("Version Control Demo")
    print(f"{'='*60}")

    vcs = WorkflowVersionControl("./workflow_versions")

    # Commit all example workflows as v1
    versions = {}
    for name, workflow in workflows.items():
        workflow['id'] = name
        v = vcs.commit(
            workflow,
            f"Initial version of {workflow.get('name', name)}",
            author="demo"
        )
        versions[name] = v
        print(f"[+] Committed: {v.version_id}")

    # Create experiment branch for one workflow
    if 'ensemble_decision_maker' in workflows:
        vcs.create_branch('ensemble_decision_maker', 'experiment')
        print(f"[branch] Created 'experiment' for ensemble_decision_maker")

        # Modify and commit to experiment branch
        modified_workflow = json.loads(json.dumps(workflows['ensemble_decision_maker']))
        modified_workflow['nodes'].append({
            "id": "extra_validator",
            "agentType": "safety",
            "x": 1300,
            "y": 500,
            "config": {"validation_level": "strict"}
        })

        v2 = vcs.commit(
            modified_workflow,
            "Add extra safety validation",
            author="demo",
            branch="experiment"
        )
        print(f"[+] Committed to experiment branch: {v2.version_id}")

        # Show diff
        diff = vcs.diff(versions['ensemble_decision_maker'].version_id, v2.version_id)
        print(f"\n[Diff] main -> experiment:")
        print(f"   Added nodes: {len(diff['nodes']['added'])}")
        print(f"   Removed nodes: {len(diff['nodes']['removed'])}")
        print(f"   Modified nodes: {len(diff['nodes']['modified'])}")

    # List all branches
    for name in workflows:
        branches = vcs.list_branches(name)
        print(f"\n[repo] {name}: branches = {branches}")

    return vcs


async def main():
    """Run the complete evaluation demo."""
    print("=== HoloLoom Workflow Evaluation Demo ===")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Load example workflows
    workflows = await load_example_workflows()
    print(f"\nLoaded {len(workflows)} example workflows:")
    for name, workflow in workflows.items():
        print(f"   - {name}: {workflow.get('name', 'Unnamed')}")

    # Create evaluator
    evaluator = WorkflowEvaluator()

    # Get benchmark test cases
    factual_tests = BenchmarkDataset.factual_qa(10)
    complex_tests = BenchmarkDataset.complex_reasoning(5)
    adversarial_tests = BenchmarkDataset.adversarial(5)

    all_tests = factual_tests + complex_tests + adversarial_tests

    print(f"\n[Benchmark Dataset]")
    print(f"   Factual Q&A: {len(factual_tests)} tests")
    print(f"   Complex Reasoning: {len(complex_tests)} tests")
    print(f"   Adversarial: {len(adversarial_tests)} tests")
    print(f"   Total: {len(all_tests)} tests")

    # Evaluate each workflow
    reports = {}
    for name, workflow in workflows.items():
        report = await evaluate_single_workflow(evaluator, workflow, all_tests, name)
        reports[name] = report

    # Summary comparison
    print(f"\n{'='*60}")
    print("Summary Comparison")
    print(f"{'='*60}")
    print(f"\n{'Workflow':<35} {'Success':<10} {'Latency':<12} {'Confidence':<12}")
    print("-" * 70)

    for name, report in sorted(reports.items(), key=lambda x: -x[1].success_rate):
        print(f"{report.workflow_name[:34]:<35} {report.success_rate:>8.1%} {report.avg_latency_ms:>10.1f}ms {report.avg_confidence:>10.3f}")

    # Find best performer
    best = max(reports.items(), key=lambda x: x[1].avg_confidence)
    print(f"\n[*] Best Performer (by confidence): {best[0]}")

    # Run A/B test if we have multiple workflows
    if len(workflows) >= 2:
        workflow_names = list(workflows.keys())
        ab_runner = ABTestRunner(evaluator)
        ab_result = await run_ab_test(
            ab_runner,
            workflows[workflow_names[0]],
            workflows[workflow_names[1]],
            all_tests
        )

    # Demo version control
    vcs = await demo_version_control(workflows)

    # Attach evaluation to versions
    print(f"\n{'='*60}")
    print("Attaching Evaluations to Versions")
    print(f"{'='*60}")

    for name, report in reports.items():
        history = vcs.get_history(name, limit=1)
        if history:
            vcs.attach_evaluation(history[0].version_id, report)
            vcs.tag(history[0].version_id, f"eval-{report.success_rate:.0%}")
            print(f"[+] Attached evaluation to {name} (tagged: eval-{report.success_rate:.0%})")

    # Save combined report
    report_path = Path("./eval_reports")
    report_path.mkdir(exist_ok=True)

    combined_report = {
        "timestamp": datetime.now().isoformat(),
        "workflows_evaluated": len(workflows),
        "total_tests": len(all_tests),
        "reports": {name: report.to_dict() for name, report in reports.items()},
        "benchmark_summary": {
            "best_performer": best[0],
            "best_success_rate": best[1].success_rate,
            "best_confidence": best[1].avg_confidence
        }
    }

    report_file = report_path / f"combined_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(combined_report, f, indent=2)

    print(f"\n[saved] Combined report: {report_file}")

    # Generate markdown summaries
    for name, report in reports.items():
        md_file = report_path / f"{name}_report.md"
        with open(md_file, 'w') as f:
            f.write(report.to_markdown())
        print(f"[saved] Markdown report: {md_file}")

    print(f"\n{'='*60}")
    print("Evaluation Demo Complete!")
    print(f"{'='*60}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == '__main__':
    asyncio.run(main())
