#!/usr/bin/env python3
"""Demo script showing CARTS attack trajectory visualization.

This script demonstrates the visualization foundation with realistic
red team attack data, anomaly detection, and strategy comparison.

Run from repository root:
    PYTHONPATH=. python HoloLoom/redteam/visualization/demo_attack_trajectory.py
"""

from hololoom.redteam.visualization import (
    AttackPoint,
    AttackTrajectoryRenderer,
    render_attack_trajectory,
)
import json
from pathlib import Path


def demo_1_simple_rendering():
    """Demo 1: Simplest usage with convenience function."""
    print("\n" + "=" * 70)
    print("DEMO 1: Simple Rendering (Convenience Function)")
    print("=" * 70)

    strategies = ["prompt_injection", "jailbreak", "context_overflow"]
    success_rates = [0.65, 0.42, 0.28]
    attack_counts = [10, 10, 10]
    bypass_counts = [6, 4, 3]

    html = render_attack_trajectory(
        strategies,
        success_rates,
        attack_counts,
        bypass_counts,
        title="Red Team Attack Analysis",
        subtitle="Phase 1: Initial attack vectors",
    )

    # Save output
    output_path = Path(__file__).parent / "demo_output_1_simple.html"
    output_path.write_text(html)
    print(f"✓ Saved to: {output_path}")
    print(f"  File size: {len(html):,} bytes")
    print(f"  Open in browser to view")


def demo_2_detailed_points():
    """Demo 2: Detailed attack points with metadata and realistic campaign."""
    print("\n" + "=" * 70)
    print("DEMO 2: Detailed Points (Realistic Campaign)")
    print("=" * 70)

    # Realistic attack campaign data
    points = [
        # Batch 1: Initial probing
        AttackPoint(
            index=0,
            strategy="prompt_injection",
            success_rate=0.60,
            attack_count=10,
            bypass_count=6,
            avg_severity=0.75,
            metadata={"batch": 1, "model": "gpt-4", "phase": "probing"},
        ),
        AttackPoint(
            index=1,
            strategy="prompt_injection",
            success_rate=0.65,
            attack_count=10,
            bypass_count=6,
            avg_severity=0.78,
            metadata={"batch": 2, "model": "gpt-4", "phase": "refinement"},
        ),
        AttackPoint(
            index=2,
            strategy="prompt_injection",
            success_rate=0.72,
            attack_count=10,
            bypass_count=7,
            avg_severity=0.82,
            metadata={"batch": 3, "model": "gpt-4", "phase": "breakthrough"},
        ),
        # Strategy shift: Different approach
        AttackPoint(
            index=3,
            strategy="jailbreak",
            success_rate=0.40,
            attack_count=10,
            bypass_count=4,
            avg_severity=0.65,
            metadata={"batch": 1, "model": "gpt-4", "phase": "shift"},
        ),
        AttackPoint(
            index=4,
            strategy="jailbreak",
            success_rate=0.42,
            attack_count=10,
            bypass_count=4,
            avg_severity=0.68,
            metadata={"batch": 2, "model": "gpt-4", "phase": "refinement"},
        ),
        # Final strategy: Context overflow
        AttackPoint(
            index=5,
            strategy="context_overflow",
            success_rate=0.28,
            attack_count=10,
            bypass_count=3,
            avg_severity=0.55,
            metadata={"batch": 1, "model": "gpt-4", "phase": "exploration"},
        ),
        AttackPoint(
            index=6,
            strategy="context_overflow",
            success_rate=0.26,
            attack_count=10,
            bypass_count=2,
            avg_severity=0.52,
            metadata={"batch": 2, "model": "gpt-4", "phase": "degradation"},
        ),
        # Defender improvements (degradation = good news)
        AttackPoint(
            index=7,
            strategy="prompt_injection",
            success_rate=0.55,
            attack_count=10,
            bypass_count=5,
            avg_severity=0.70,
            metadata={"batch": 4, "model": "gpt-4-updated", "phase": "defenses"},
        ),
    ]

    renderer = AttackTrajectoryRenderer(
        detect_anomalies=True, show_strategy_breakdown=True
    )

    html = renderer.render(
        points,
        title="CARTS Attack Campaign: Multi-Strategy Red Team",
        subtitle="8-batch campaign across 3 attack vectors",
    )

    output_path = Path(__file__).parent / "demo_output_2_detailed.html"
    output_path.write_text(html)
    print(f"✓ Saved to: {output_path}")
    print(f"  Data points: {len(points)}")
    print(f"  Strategies: {len(set(p.strategy for p in points))}")
    print(f"  File size: {len(html):,} bytes")
    print(f"  Anomalies detected: Yes (see diamonds on chart)")


def demo_3_no_anomalies():
    """Demo 3: Same data but without anomaly detection."""
    print("\n" + "=" * 70)
    print("DEMO 3: Clean Visualization (No Anomaly Markers)")
    print("=" * 70)

    strategies = [
        "prompt_injection",
        "prompt_injection",
        "prompt_injection",
        "jailbreak",
        "jailbreak",
        "context_overflow",
        "context_overflow",
        "prompt_injection",
    ]

    success_rates = [0.60, 0.65, 0.72, 0.40, 0.42, 0.28, 0.26, 0.55]
    attack_counts = [10] * 8
    bypass_counts = [6, 6, 7, 4, 4, 3, 2, 5]

    renderer = AttackTrajectoryRenderer(
        detect_anomalies=False,  # Disable anomalies
        show_strategy_breakdown=True,
    )

    # Create points manually for comparison
    points = [
        AttackPoint(
            index=i,
            strategy=strategies[i],
            success_rate=success_rates[i],
            attack_count=attack_counts[i],
            bypass_count=bypass_counts[i],
            avg_severity=bypass_counts[i] / attack_counts[i],
        )
        for i in range(len(strategies))
    ]

    html = renderer.render(
        points,
        title="Clean Attack Trajectory (No Anomalies)",
        subtitle="Same data, different visualization",
    )

    output_path = Path(__file__).parent / "demo_output_3_clean.html"
    output_path.write_text(html)
    print(f"✓ Saved to: {output_path}")
    print(f"  Anomaly detection: DISABLED")
    print(f"  File size: {len(html):,} bytes")


def demo_4_model_comparison():
    """Demo 4: Comparing attack effectiveness across models."""
    print("\n" + "=" * 70)
    print("DEMO 4: Multi-Model Comparison")
    print("=" * 70)

    # Different models show different vulnerabilities
    models = {
        "GPT-4": [0.65, 0.68, 0.72, 0.70],  # Generally more vulnerable
        "Claude-3": [0.42, 0.40, 0.38, 0.36],  # Better defenses
        "Gemini": [0.28, 0.26, 0.24, 0.22],  # Best defenses
    }

    points = []
    index = 0

    for model, rates in models.items():
        for i, rate in enumerate(rates):
            point = AttackPoint(
                index=index,
                strategy=f"{model} (batch {i+1})",
                success_rate=rate,
                attack_count=20,
                bypass_count=int(rate * 20),
                avg_severity=rate * 0.85,  # Severity correlates with rate
                metadata={"model": model, "batch": i + 1},
            )
            points.append(point)
            index += 1

    renderer = AttackTrajectoryRenderer(
        detect_anomalies=True, show_strategy_breakdown=True
    )

    html = renderer.render(
        points,
        title="Red Team Comparative Analysis: Model Vulnerability",
        subtitle="Attack effectiveness across 3 LLM providers",
    )

    output_path = Path(__file__).parent / "demo_output_4_models.html"
    output_path.write_text(html)
    print(f"✓ Saved to: {output_path}")
    print(f"  Models compared: {len(models)}")
    print(f"  Data points: {len(points)}")
    print(f"  Key insight: GPT-4 most vulnerable, Gemini most robust")


def demo_5_metrics_summary():
    """Demo 5: Show metrics computed from data."""
    print("\n" + "=" * 70)
    print("DEMO 5: Metrics Summary")
    print("=" * 70)

    strategies = ["prompt_injection", "prompt_injection", "jailbreak", "jailbreak"]
    success_rates = [0.60, 0.72, 0.40, 0.35]
    attack_counts = [10, 10, 10, 10]
    bypass_counts = [6, 7, 4, 3]

    renderer = AttackTrajectoryRenderer()

    # Create points for metrics computation
    points = [
        AttackPoint(
            index=i,
            strategy=strategies[i],
            success_rate=success_rates[i],
            attack_count=attack_counts[i],
            bypass_count=bypass_counts[i],
            avg_severity=bypass_counts[i] / attack_counts[i],
        )
        for i in range(len(strategies))
    ]

    # Get metrics
    metrics = renderer._compute_metrics(points)
    by_strategy = renderer._group_by_strategy(points)
    strategy_metrics = renderer._compute_strategy_metrics(by_strategy)

    print("\nOverall Metrics:")
    print(f"  Overall Bypass Rate: {metrics['overall_bypass_rate']:.1%}")
    print(f"  Total Attacks: {metrics['total_attacks']}")
    print(f"  Successful Bypasses: {metrics['total_bypasses']}")
    print(f"  Average Severity: {metrics['avg_severity']:.2f}")
    print(f"  Peak Success Rate: {metrics['max_success_rate']:.1%}")
    print(f"  Min Success Rate: {metrics['min_success_rate']:.1%}")
    print(f"  Success Volatility: {metrics['success_volatility']:.3f}")
    print(f"  Trend: {metrics['bypass_trend']}")

    print("\nPer-Strategy Metrics:")
    for strategy, metrics in strategy_metrics.items():
        print(f"\n  {strategy}:")
        print(f"    Total Attacks: {metrics.total_attacks}")
        print(f"    Bypass Rate: {metrics.bypass_rate:.1%}")
        print(f"    Avg Severity: {metrics.avg_severity:.2f}")
        print(f"    Trend: {metrics.trend}")
        print(f"    Data points: {metrics.points_count}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("CARTS Attack Trajectory Visualization Demo")
    print("=" * 70)
    print("\nThis script demonstrates the visualization foundation with")
    print("realistic red team attack data and comprehensive analytics.")

    try:
        demo_1_simple_rendering()
        demo_2_detailed_points()
        demo_3_no_anomalies()
        demo_4_model_comparison()
        demo_5_metrics_summary()

        print("\n" + "=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)
        print("\nGenerated files:")
        demo_dir = Path(__file__).parent
        for html_file in sorted(demo_dir.glob("demo_output_*.html")):
            size_kb = html_file.stat().st_size / 1024
            print(f"  ✓ {html_file.name} ({size_kb:.1f} KB)")

        print("\nTo view visualizations:")
        print("  1. Open the .html files in a web browser")
        print("  2. Each file is self-contained (no external dependencies)")
        print("  3. Charts show attack effectiveness over time")
        print("  4. Anomalies marked with colored diamonds")
        print("  5. Strategy comparison shown in bottom table")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
