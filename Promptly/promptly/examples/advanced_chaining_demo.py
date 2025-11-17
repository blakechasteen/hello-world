#!/usr/bin/env python3
"""
Advanced Chaining Demo

Demonstrates all advanced chaining capabilities:
1. Load and execute 10 advanced chain examples
2. Visualize chains with interactive HTML
3. Debug chains with breakpoints
4. Optimize chains for performance
5. Compose chains together
6. Monitor chain execution

Usage:
    python advanced_chaining_demo.py [demo_number]

    If no demo_number provided, runs all demos.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chain_dsl import ChainDSL
from chain_viz_advanced import visualize_chain_advanced, AdvancedChainVisualizer
from chain_debugger import create_debugger, add_conditional_breakpoint, DebugAction
from chain_optimizer import optimize_chain, ChainOptimizer
from chain_composer import ChainComposer
from chain_monitor import get_global_monitor
from chain_tracing import create_tracer, TraceLevel

# Demo configuration
CHAINS_DIR = Path(__file__).parent.parent / "examples" / "advanced_chains"
OUTPUT_DIR = Path(__file__).parent / "demo_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def print_header(title: str) -> None:
    """Print demo section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_step(step: str) -> None:
    """Print demo step"""
    print(f"➤ {step}")


def demo_1_load_and_visualize_chains():
    """Demo 1: Load all advanced chains and create visualizations"""
    print_header("Demo 1: Load and Visualize Advanced Chains")

    chains = [
        "research_pipeline.yaml",
        "adaptive_content.yaml",
        "code_review_chain.yaml",
        "customer_support.yaml",
        "data_enrichment.yaml",
        "consensus_chain.yaml",
        "refinement_loop.yaml",
        "dynamic_router.yaml",
        "hierarchical_planning.yaml",
        "learning_chain.yaml"
    ]

    dsl = ChainDSL()

    for chain_file in chains:
        chain_path = CHAINS_DIR / chain_file
        if not chain_path.exists():
            print(f"⚠ Chain not found: {chain_file}")
            continue

        print_step(f"Loading {chain_file}")

        try:
            # Load chain
            chain_def = dsl.load_chain(str(chain_path))
            print(f"  ✓ Loaded: {chain_def['name']}")
            print(f"    Description: {chain_def.get('description', 'N/A')}")
            print(f"    Steps: {len(chain_def.get('steps', []))}")

            # Validate
            validation = dsl.validate_chain(chain_def)
            if not validation['valid']:
                print(f"  ✗ Validation errors: {validation['errors']}")
                continue

            print(f"  ✓ Validation passed")

            # Create visualization
            visualizer = AdvancedChainVisualizer()
            visualizer.build_graph_from_chain(chain_def)

            # Save multiple formats
            output_base = OUTPUT_DIR / chain_def['name']

            # Mermaid
            mermaid = visualizer.to_mermaid()
            (output_base.with_suffix('.mermaid')).write_text(mermaid)

            # Graphviz
            graphviz = visualizer.to_graphviz()
            (output_base.with_suffix('.dot')).write_text(graphviz)

            # ASCII
            ascii_viz = visualizer.to_ascii()
            (output_base.with_suffix('.txt')).write_text(ascii_viz)

            # JSON
            json_viz = json.dumps(visualizer.to_json(), indent=2)
            (output_base.with_suffix('.json')).write_text(json_viz)

            # Interactive HTML (without execution trace for now)
            html = visualizer.to_interactive_html(f"{chain_def['name']} Visualization")
            html_path = output_base.with_suffix('.html')
            html_path.write_text(html)

            print(f"  ✓ Visualizations saved to: {output_base}.*")
            print(f"    - HTML: {html_path}")
            print(f"    - Mermaid, Graphviz, ASCII, JSON")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n✓ Demo 1 completed!")
    print(f"  Output directory: {OUTPUT_DIR}")


def demo_2_chain_debugging():
    """Demo 2: Debug a chain with breakpoints"""
    print_header("Demo 2: Chain Debugging with Breakpoints")

    # Load research pipeline
    dsl = ChainDSL()
    chain_path = CHAINS_DIR / "refinement_loop.yaml"

    if not chain_path.exists():
        print("⚠ Chain not found, skipping demo")
        return

    chain_def = dsl.load_chain(str(chain_path))
    print_step(f"Loaded chain: {chain_def['name']}")

    # Create debugger
    debugger = create_debugger(chain_def)
    print_step("Created debugger")

    # Add breakpoints
    debugger.add_breakpoint("generate_initial_version")
    print("  Added breakpoint: generate_initial_version")

    # Conditional breakpoint
    add_conditional_breakpoint(
        debugger,
        "refinement_loop",
        "quality_score < 0.85"
    )
    print("  Added conditional breakpoint: refinement_loop (quality_score < 0.85)")

    # Hit count breakpoint
    debugger.add_breakpoint("extract_final_version", hit_condition="== 1")
    print("  Added hit count breakpoint: extract_final_version (first hit only)")

    # List all breakpoints
    print_step("Configured breakpoints:")
    for bp_info in debugger.list_breakpoints():
        print(f"  - {bp_info['step']}")
        print(f"    Enabled: {bp_info['enabled']}")
        print(f"    Has condition: {bp_info['has_condition']}")
        if bp_info['hit_condition']:
            print(f"    Hit condition: {bp_info['hit_condition']}")

    # Set breakpoint callback
    breakpoint_hit_count = [0]

    def on_breakpoint(bp, frame):
        breakpoint_hit_count[0] += 1
        print(f"\n  🔴 Breakpoint #{breakpoint_hit_count[0]}: {bp.step_name}")

        # Inspect variables
        variables = debugger.inspect_variables(frame)
        print(f"    Variables: {list(variables.keys())}")

        # Call stack
        stack = debugger.get_call_stack()
        print(f"    Call stack depth: {len(stack)}")

        # Auto-continue for demo
        return DebugAction.CONTINUE

    debugger.on_breakpoint_hit = on_breakpoint

    print_step("Breakpoint callback configured")

    # Get execution summary
    print_step("Debugger ready")
    print(f"  Total breakpoints: {len(debugger.breakpoints)}")

    # Export debug configuration
    config_path = OUTPUT_DIR / "debug_config.json"
    config_data = {
        "chain": chain_def['name'],
        "breakpoints": debugger.list_breakpoints()
    }
    config_path.write_text(json.dumps(config_data, indent=2))
    print(f"  Debug configuration saved: {config_path}")

    print("\n✓ Demo 2 completed!")


def demo_3_chain_optimization():
    """Demo 3: Analyze and optimize chains"""
    print_header("Demo 3: Chain Optimization Analysis")

    # Load a chain
    dsl = ChainDSL()
    chain_path = CHAINS_DIR / "research_pipeline.yaml"

    if not chain_path.exists():
        print("⚠ Chain not found, skipping demo")
        return

    chain_def = dsl.load_chain(str(chain_path))
    print_step(f"Loaded chain: {chain_def['name']}")

    # Create optimizer
    optimizer = ChainOptimizer()
    print_step("Analyzing chain for optimization opportunities...")

    # Analyze (without execution trace for structural analysis)
    suggestions = optimizer.analyze(chain_def)

    print(f"\n  Found {len(suggestions)} optimization opportunities\n")

    # Group by category
    by_category = {}
    for suggestion in suggestions:
        if suggestion.category not in by_category:
            by_category[suggestion.category] = []
        by_category[suggestion.category].append(suggestion)

    # Print by category
    for category, suggs in by_category.items():
        print(f"  {category.upper()}: {len(suggs)} suggestions")
        for i, sugg in enumerate(suggs[:2], 1):  # Show first 2
            print(f"    {i}. {sugg.title}")
            print(f"       Impact: {sugg.impact} | Effort: {sugg.effort}")
            print(f"       Improvement: {sugg.estimated_improvement}")

    # Generate full report
    print_step("Generating optimization report...")
    report = optimizer.generate_report()
    report_path = OUTPUT_DIR / f"{chain_def['name']}_optimization_report.md"
    report_path.write_text(report)
    print(f"  ✓ Report saved: {report_path}")

    # Export JSON
    json_path = OUTPUT_DIR / f"{chain_def['name']}_optimizations.json"
    json_path.write_text(optimizer.export_json())
    print(f"  ✓ JSON export: {json_path}")

    print("\n✓ Demo 3 completed!")


def demo_4_chain_composition():
    """Demo 4: Compose multiple chains together"""
    print_header("Demo 4: Chain Composition")

    dsl = ChainDSL()
    composer = ChainComposer()

    # Load some chains
    print_step("Loading chains for composition...")

    chains_to_load = ["research_pipeline", "adaptive_content", "consensus_chain"]
    loaded_chains = {}

    for chain_name in chains_to_load:
        chain_path = CHAINS_DIR / f"{chain_name}.yaml"
        if chain_path.exists():
            chain_def = dsl.load_chain(str(chain_path))
            composer.register_chain(chain_def)
            loaded_chains[chain_name] = chain_def
            print(f"  ✓ Loaded: {chain_name}")

    # Demo 1: Sequential composition
    if len(loaded_chains) >= 2:
        print_step("Sequential Composition: Research → Adaptive Content")

        try:
            composed = composer.compose_sequential(
                chain_names=["research_pipeline", "adaptive_content"],
                name="research_and_content",
                share_context=True
            )

            print(f"  ✓ Created: {composed['name']}")
            print(f"    Type: {composed['composition_type']}")
            print(f"    Composed from: {composed['composed_from']}")
            print(f"    Total steps: {len(composed['steps'])}")

            # Save composed chain
            output_path = OUTPUT_DIR / "composed_sequential.yaml"
            import yaml
            output_path.write_text(yaml.dump(composed, default_flow_style=False))
            print(f"  ✓ Saved: {output_path}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Demo 2: Parallel composition
    if len(loaded_chains) >= 3:
        print_step("Parallel Composition: Multiple Consensus")

        try:
            composed = composer.compose_parallel(
                chain_names=list(loaded_chains.keys())[:3],
                name="parallel_consensus",
                aggregation="all"
            )

            print(f"  ✓ Created: {composed['name']}")
            print(f"    Type: {composed['composition_type']}")
            print(f"    Chains in parallel: {len(composed['composed_from'])}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Demo 3: Template chain
    if "research_pipeline" in loaded_chains:
        print_step("Template Chain: Parameterized Research")

        try:
            template = composer.create_template_chain(
                base_chain="research_pipeline",
                name="research_template",
                parameters={
                    "max_subquestions": 5,
                    "research_timeout": 60.0,
                    "citation_format": "apa"
                }
            )

            print(f"  ✓ Created template: {template['name']}")
            print(f"    Base: {template['template_base']}")
            print(f"    Parameters: {list(template['parameters'].keys())}")

            # Instantiate template
            instance = composer.instantiate_template(
                template_name="research_template",
                instance_name="academic_research",
                parameter_values={
                    "max_subquestions": 3,
                    "research_timeout": 30.0,
                    "citation_format": "mla"
                }
            )

            print(f"  ✓ Instantiated: {instance['name']}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # List all compositions
    print_step("All Compositions:")
    compositions = composer.list_compositions()
    for comp in compositions:
        print(f"  - {comp['name']} ({comp['type']})")
        print(f"    From: {comp['composed_from']}")

    print("\n✓ Demo 4 completed!")


def demo_5_chain_monitoring():
    """Demo 5: Monitor chain execution"""
    print_header("Demo 5: Chain Monitoring & Metrics")

    monitor = get_global_monitor()

    print_step("Simulating chain executions...")

    # Simulate various executions
    chains = [
        ("research_pipeline", 42.3, 0.034, True, None),
        ("research_pipeline", 38.7, 0.029, True, None),
        ("research_pipeline", 125.2, 0.082, False, "Timeout"),
        ("adaptive_content", 67.4, 0.051, True, None),
        ("adaptive_content", 71.2, 0.056, True, None),
        ("code_review_chain", 89.3, 0.073, True, None),
        ("code_review_chain", 156.8, 0.124, False, "Rate limit"),
        ("customer_support", 12.4, 0.008, True, None),
        ("customer_support", 11.9, 0.007, True, None),
    ]

    for chain_name, duration, cost, success, error in chains:
        monitor.record_execution(chain_name, duration, cost, success, error)
        time.sleep(0.1)  # Small delay

    print(f"  ✓ Recorded {len(chains)} executions")

    # Get metrics
    print_step("Chain Metrics:")
    for chain_name in ["research_pipeline", "adaptive_content", "code_review_chain", "customer_support"]:
        metrics = monitor.get_metrics(chain_name)
        if metrics:
            print(f"\n  {chain_name}:")
            print(f"    Executions: {metrics['total_executions']}")
            print(f"    Success rate: {metrics['success_rate']*100:.1f}%")
            print(f"    Avg duration: {metrics['average_duration']:.2f}s")
            print(f"    Avg cost: ${metrics['average_cost']:.4f}")

    # Dashboard data
    print_step("Dashboard Summary:")
    dashboard = monitor.get_dashboard_data()
    summary = dashboard['summary']
    print(f"  Total chains: {summary['total_chains']}")
    print(f"  Total executions: {summary['total_executions']}")
    print(f"  Overall success rate: {summary['overall_success_rate']*100:.1f}%")
    print(f"  Total cost: ${summary['total_cost']:.4f}")
    print(f"  Active alerts: {summary['active_alerts']}")

    # Cost breakdown
    print_step("Top 3 Chains by Cost:")
    for i, (chain, cost) in enumerate(list(dashboard['cost_breakdown'].items())[:3], 1):
        print(f"  {i}. {chain}: ${cost:.4f}")

    # Generate HTML dashboard
    print_step("Generating HTML dashboard...")
    html = monitor.generate_html_dashboard()
    dashboard_path = OUTPUT_DIR / "monitoring_dashboard.html"
    dashboard_path.write_text(html)
    print(f"  ✓ Dashboard saved: {dashboard_path}")

    # Export metrics
    json_path = OUTPUT_DIR / "metrics.json"
    json_path.write_text(monitor.export_json())
    print(f"  ✓ Metrics JSON: {json_path}")

    print("\n✓ Demo 5 completed!")


def demo_6_end_to_end_workflow():
    """Demo 6: Complete workflow with all tools"""
    print_header("Demo 6: End-to-End Workflow")

    print_step("Load chain")
    dsl = ChainDSL()
    chain_path = CHAINS_DIR / "research_pipeline.yaml"

    if not chain_path.exists():
        print("⚠ Chain not found, skipping demo")
        return

    chain_def = dsl.load_chain(str(chain_path))
    print(f"  ✓ Loaded: {chain_def['name']}")

    # Step 1: Visualize
    print_step("1. Create visualization")
    viz = AdvancedChainVisualizer()
    viz.build_graph_from_chain(chain_def)
    html = viz.to_interactive_html("Research Pipeline Workflow")
    viz_path = OUTPUT_DIR / "workflow_visualization.html"
    viz_path.write_text(html)
    print(f"  ✓ Visualization: {viz_path}")

    # Step 2: Optimize
    print_step("2. Analyze for optimizations")
    optimizer = ChainOptimizer()
    suggestions = optimizer.analyze(chain_def)
    print(f"  ✓ Found {len(suggestions)} optimization opportunities")

    # Step 3: Create debugger
    print_step("3. Setup debugging")
    debugger = create_debugger(chain_def)
    debugger.add_breakpoint("parallel_research")
    print(f"  ✓ Debugger ready with {len(debugger.breakpoints)} breakpoints")

    # Step 4: Setup monitoring
    print_step("4. Setup monitoring")
    monitor = get_global_monitor()

    def alert_handler(alert):
        print(f"    🚨 Alert: {alert.severity} - {alert.message}")

    monitor.register_alert_callback(alert_handler)
    print("  ✓ Monitoring configured")

    # Step 5: Create summary report
    print_step("5. Generate comprehensive report")

    report = f"""# Workflow Analysis Report

## Chain: {chain_def['name']}

### Description
{chain_def.get('description', 'N/A')}

### Structure
- Version: {chain_def.get('version', '1.0')}
- Total steps: {len(chain_def.get('steps', []))}
- Variables: {len(chain_def.get('variables', {}))}

### Optimization Opportunities
{len(suggestions)} suggestions found:
"""

    # Add optimization categories
    by_category = {}
    for sugg in suggestions:
        if sugg.category not in by_category:
            by_category[sugg.category] = []
        by_category[sugg.category].append(sugg)

    for category, suggs in by_category.items():
        report += f"\n#### {category.upper()}\n"
        for sugg in suggs[:3]:
            report += f"- {sugg.title} (Impact: {sugg.impact}, Effort: {sugg.effort})\n"

    report += f"""
### Debugging Setup
- Breakpoints configured: {len(debugger.breakpoints)}
- Debug trace level: Standard

### Monitoring
- Metrics tracked: Execution count, duration, cost, success rate
- Alerts configured: Yes
- Dashboard generated: Yes

### Outputs
- Visualization: workflow_visualization.html
- Optimization report: Available
- Debug configuration: Ready
- Monitoring dashboard: monitoring_dashboard.html

## Next Steps

1. Review optimization suggestions
2. Apply high-impact, low-effort optimizations
3. Test with debugging enabled
4. Monitor production execution
5. Iterate based on metrics
"""

    report_path = OUTPUT_DIR / "workflow_analysis_report.md"
    report_path.write_text(report)
    print(f"  ✓ Report saved: {report_path}")

    print("\n✓ Demo 6 completed!")
    print(f"\n📁 All outputs in: {OUTPUT_DIR}")


def demo_7_performance_comparison():
    """Demo 7: Compare chain performance"""
    print_header("Demo 7: Performance Comparison")

    print_step("Performance comparison of different chain patterns...")

    # Simulate different execution patterns
    patterns = {
        "sequential_baseline": {
            "description": "Sequential execution (baseline)",
            "executions": [
                (45.2, 0.035),
                (43.8, 0.034),
                (46.1, 0.036),
            ]
        },
        "parallel_optimized": {
            "description": "With parallelization",
            "executions": [
                (15.3, 0.037),  # Faster but slightly more expensive
                (14.9, 0.036),
                (15.7, 0.038),
            ]
        },
        "cached_version": {
            "description": "With caching enabled",
            "executions": [
                (8.2, 0.015),  # Much faster and cheaper (cache hits)
                (42.1, 0.034),  # Cache miss
                (7.9, 0.014),  # Cache hit
            ]
        }
    }

    monitor = get_global_monitor()

    # Record all executions
    for pattern_name, pattern_data in patterns.items():
        for duration, cost in pattern_data["executions"]:
            monitor.record_execution(pattern_name, duration, cost, True)

    # Compare results
    print_step("Performance Comparison:\n")

    comparison_data = []
    for pattern_name, pattern_data in patterns.items():
        metrics = monitor.get_metrics(pattern_name)
        comparison_data.append({
            "pattern": pattern_name,
            "description": pattern_data["description"],
            "avg_duration": metrics["average_duration"],
            "avg_cost": metrics["average_cost"],
            "executions": metrics["total_executions"]
        })

    # Print table
    print("  Pattern                    | Avg Duration | Avg Cost  | Speedup")
    print("  " + "-" * 70)

    baseline_duration = comparison_data[0]["avg_duration"]

    for data in comparison_data:
        speedup = baseline_duration / data["avg_duration"]
        print(f"  {data['description']:25} | {data['avg_duration']:8.2f}s   | ${data['avg_cost']:.4f} | {speedup:5.2f}x")

    print("\n  Insights:")
    print(f"  - Parallelization: {baseline_duration / comparison_data[1]['avg_duration']:.2f}x faster")
    print(f"  - Caching: {baseline_duration / comparison_data[2]['avg_duration']:.2f}x faster (with hits)")
    print(f"  - Cost reduction with caching: {(1 - comparison_data[2]['avg_cost'] / comparison_data[0]['avg_cost']) * 100:.1f}%")

    # Save comparison report
    report_path = OUTPUT_DIR / "performance_comparison.json"
    report_path.write_text(json.dumps(comparison_data, indent=2))
    print(f"\n  ✓ Comparison saved: {report_path}")

    print("\n✓ Demo 7 completed!")


def main():
    """Main demo runner"""
    print("\n" + "=" * 80)
    print("  Advanced Prompt Chaining Demo")
    print("  Showcasing all advanced features")
    print("=" * 80)

    # Check if specific demo requested
    if len(sys.argv) > 1:
        demo_num = sys.argv[1]
        demos = {
            "1": demo_1_load_and_visualize_chains,
            "2": demo_2_chain_debugging,
            "3": demo_3_chain_optimization,
            "4": demo_4_chain_composition,
            "5": demo_5_chain_monitoring,
            "6": demo_6_end_to_end_workflow,
            "7": demo_7_performance_comparison
        }

        if demo_num in demos:
            demos[demo_num]()
        else:
            print(f"Unknown demo: {demo_num}")
            print("Available demos: 1-7")
            return
    else:
        # Run all demos
        demos = [
            demo_1_load_and_visualize_chains,
            demo_2_chain_debugging,
            demo_3_chain_optimization,
            demo_4_chain_composition,
            demo_5_chain_monitoring,
            demo_6_end_to_end_workflow,
            demo_7_performance_comparison
        ]

        for i, demo in enumerate(demos, 1):
            try:
                demo()
                time.sleep(1)  # Brief pause between demos
            except Exception as e:
                print(f"\n✗ Demo {i} failed: {e}")
                import traceback
                traceback.print_exc()

    # Final summary
    print("\n" + "=" * 80)
    print("  Demo Complete!")
    print("=" * 80)
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("\n  Generated files:")
    for file in sorted(OUTPUT_DIR.iterdir()):
        print(f"    - {file.name}")

    print("\n  Next steps:")
    print("    1. Open HTML visualizations in browser")
    print("    2. Review optimization reports")
    print("    3. Explore monitoring dashboard")
    print("    4. Try modifying the chains")
    print("\n  Documentation:")
    print("    - ADVANCED_CHAINING_GUIDE.md")
    print("    - CHAIN_PATTERNS.md")
    print("    - TROUBLESHOOTING_CHAINS.md")
    print()


if __name__ == "__main__":
    main()
