#!/usr/bin/env python3
"""
HoloLoom Visualization Gallery
==============================
Demonstrates Tufte-style visualizations without heavy dependencies.
Created: 2026-01-28

"Above all else show the data." — Edward Tufte

Features demonstrated:
1. Sparklines - Word-sized graphics showing trends
2. Small Multiples - Side-by-side comparisons
3. Data Density Tables - Maximum information per square inch
4. Stage Waterfall - Pipeline timing visualization
5. Confidence Trajectory - Tracking confidence over time

Philosophy: High data-ink ratio
    - Maximize information, minimize decoration
    - Show meaning first
    - Zero external dependencies (pure HTML/CSS/SVG)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path


def section(title: str, subtitle: str = ""):
    """Print an elegant section header."""
    width = 65
    print(f"\n{'─' * width}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'─' * width}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMO 1: Tufte Sparklines
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def demo_sparklines():
    """Demonstrate word-sized sparkline graphics."""
    section(
        "1. Tufte Sparklines",
        "Word-sized graphics showing trends inline with text"
    )

    def render_sparkline(values, width=100, height=20, color="#3498db"):
        """Render a sparkline as SVG."""
        if not values or len(values) < 2:
            return f'<svg width="{width}" height="{height}"></svg>'

        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val or 1

        # Normalize values to SVG coordinates
        points = []
        for i, v in enumerate(values):
            x = (i / (len(values) - 1)) * (width - 4) + 2
            y = height - 2 - ((v - min_val) / val_range) * (height - 4)
            points.append(f"{x:.1f},{y:.1f}")

        polyline = f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>'

        # Add endpoint markers
        first_x, first_y = points[0].split(",")
        last_x, last_y = points[-1].split(",")
        markers = f'''
            <circle cx="{first_x}" cy="{first_y}" r="2" fill="#95a5a6"/>
            <circle cx="{last_x}" cy="{last_y}" r="2" fill="{color}"/>
        '''

        return f'<svg width="{width}" height="{height}" style="vertical-align: middle;">{polyline}{markers}</svg>'

    # Generate sample data
    random.seed(42)

    metrics = {
        "Query Latency": [145, 132, 128, 125, 138, 122, 119, 115, 118, 112],
        "Confidence": [0.82, 0.85, 0.79, 0.88, 0.91, 0.87, 0.93, 0.89, 0.92, 0.94],
        "Memory Usage": [456, 462, 471, 468, 489, 495, 502, 498, 512, 508],
        "Cache Hit Rate": [0.65, 0.72, 0.75, 0.78, 0.82, 0.85, 0.88, 0.86, 0.91, 0.89],
    }

    print("  Sparklines show trends inline with metrics:")
    print()
    print(f"    {'Metric':<20} {'Current':>10} {'Trend':>15} {'Change':>10}")
    print(f"    {'─' * 20} {'─' * 10} {'─' * 15} {'─' * 10}")

    for name, values in metrics.items():
        current = values[-1]
        change = current - values[0]
        change_pct = (change / values[0]) * 100 if values[0] else 0
        trend = "▲" if change > 0 else "▼" if change < 0 else "─"

        # Format based on metric type
        if "Rate" in name or "Confidence" in name:
            current_str = f"{current:.1%}"
            change_str = f"{trend} {abs(change_pct):.1f}%"
        elif "Latency" in name:
            current_str = f"{current}ms"
            change_str = f"{trend} {abs(change):.0f}ms"
        else:
            current_str = f"{current}MB"
            change_str = f"{trend} {abs(change):.0f}MB"

        # Console-friendly sparkline
        spark_chars = []
        min_v, max_v = min(values), max(values)
        for v in values:
            norm = (v - min_v) / (max_v - min_v) if max_v != min_v else 0.5
            spark_chars.append("▁▂▃▄▅▆▇█"[int(norm * 7)])

        print(f"    {name:<20} {current_str:>10} {''.join(spark_chars):>15} {change_str:>10}")

    print()
    print("  Key Principle: Sparklines convey trends in minimal space,")
    print("  allowing data to appear inline with explanatory text.")

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMO 2: Small Multiples
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def demo_small_multiples():
    """Demonstrate small multiples for comparison."""
    section(
        "2. Small Multiples",
        "Side-by-side comparison with consistent scales"
    )

    random.seed(42)

    # Query performance across different modes
    queries = [
        {"name": "Simple Query", "latency": 45, "confidence": 0.92, "threads": 3, "cached": True},
        {"name": "Complex Query", "latency": 156, "confidence": 0.87, "threads": 8, "cached": False},
        {"name": "Research Query", "latency": 342, "confidence": 0.94, "threads": 12, "cached": False},
        {"name": "Repeat Query", "latency": 2, "confidence": 0.95, "threads": 0, "cached": True},
    ]

    # Find max for consistent scales
    max_latency = max(q["latency"] for q in queries)
    max_threads = max(q["threads"] for q in queries)

    print("  Small Multiples: Comparing query performance")
    print()
    print("  ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐")

    # Names row
    name_row = "  │"
    for q in queries:
        name = q["name"][:15]
        name_row += f" {name:^15} │"
    print(name_row)

    print("  ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤")

    # Latency row with bars
    lat_row = "  │"
    for q in queries:
        bar_len = int((q["latency"] / max_latency) * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        lat_row += f" {bar} {q['latency']:>3}ms│"
    print(lat_row)

    # Confidence row
    conf_row = "  │"
    for q in queries:
        bar_len = int(q["confidence"] * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        conf_row += f" {bar} {q['confidence']:.0%} │"
    print(conf_row)

    # Threads row
    thread_row = "  │"
    for q in queries:
        if max_threads > 0:
            bar_len = int((q["threads"] / max_threads) * 10)
        else:
            bar_len = 0
        bar = "█" * bar_len + "░" * (10 - bar_len)
        thread_row += f" {bar} {q['threads']:>2}th │"
    print(thread_row)

    # Cached row
    cache_row = "  │"
    for q in queries:
        cached = "✓ cached" if q["cached"] else "✗ fresh "
        cache_row += f" {cached:^15} │"
    print(cache_row)

    print("  └─────────────────┴─────────────────┴─────────────────┴─────────────────┘")
    print()

    # Highlight best/worst
    best_latency = min(queries, key=lambda q: q["latency"])
    best_conf = max(queries, key=lambda q: q["confidence"])

    print(f"  ★ Best latency: {best_latency['name']} ({best_latency['latency']}ms)")
    print(f"  ★ Best confidence: {best_conf['name']} ({best_conf['confidence']:.0%})")
    print()
    print("  Key Principle: Small multiples enable fair comparison")
    print("  by using consistent scales across all visualizations.")

    return queries


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMO 3: Data Density Table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def demo_density_table():
    """Demonstrate high-density data tables."""
    section(
        "3. Data Density Tables",
        "Maximum information per square inch"
    )

    random.seed(42)

    # Pipeline stage timing data
    stages = [
        {"name": "Pattern Selection", "duration": 12.5, "trend": [15, 14, 13, 12, 12.5], "delta": -2.5},
        {"name": "Chrono Trigger", "duration": 8.3, "trend": [9, 8.5, 8.2, 8.1, 8.3], "delta": -0.7},
        {"name": "Thread Selection", "duration": 25.7, "trend": [30, 28, 26, 25, 25.7], "delta": -4.3},
        {"name": "Feature Extract", "duration": 45.2, "trend": [50, 48, 46, 45, 45.2], "delta": -4.8},
        {"name": "Warp Space", "duration": 18.9, "trend": [20, 19.5, 19, 18.8, 18.9], "delta": -1.1},
        {"name": "Convergence", "duration": 15.4, "trend": [16, 15.8, 15.5, 15.3, 15.4], "delta": -0.6},
        {"name": "Tool Execution", "duration": 22.1, "trend": [25, 24, 23, 22, 22.1], "delta": -2.9},
        {"name": "Spacetime Fabric", "duration": 6.8, "trend": [8, 7.5, 7.2, 7, 6.8], "delta": -1.2},
    ]

    total_duration = sum(s["duration"] for s in stages)

    print("  High-Density Pipeline Timing Table:")
    print()
    print(f"  {'Stage':<18} {'Time':>8} {'%':>6} {'Trend':>10} {'Δ':>8} {'Status'}")
    print(f"  {'─' * 18} {'─' * 8} {'─' * 6} {'─' * 10} {'─' * 8} {'─' * 8}")

    for stage in stages:
        pct = (stage["duration"] / total_duration) * 100

        # Console sparkline
        trend = stage["trend"]
        min_t, max_t = min(trend), max(trend)
        spark = ""
        for t in trend:
            norm = (t - min_t) / (max_t - min_t) if max_t != min_t else 0.5
            spark += "▁▂▃▄▅▆▇█"[int(norm * 7)]

        # Delta indicator
        delta = stage["delta"]
        if delta < 0:
            delta_str = f"▼{abs(delta):.1f}"
        elif delta > 0:
            delta_str = f"▲{delta:.1f}"
        else:
            delta_str = "─"

        # Status (bottleneck detection)
        if pct > 25:
            status = "⚠ SLOW"
        elif pct > 15:
            status = "◐"
        else:
            status = "✓"

        print(f"  {stage['name']:<18} {stage['duration']:>6.1f}ms {pct:>5.1f}% {spark:>10} {delta_str:>8} {status}")

    print(f"  {'─' * 18} {'─' * 8} {'─' * 6} {'─' * 10} {'─' * 8} {'─' * 8}")
    print(f"  {'TOTAL':<18} {total_duration:>6.1f}ms {100.0:>5.1f}%")
    print()
    print("  Key Principle: Dense tables pack maximum information")
    print("  using inline sparklines, delta indicators, and status flags.")

    return stages


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMO 4: Stage Waterfall Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def demo_waterfall():
    """Demonstrate pipeline waterfall visualization."""
    section(
        "4. Stage Waterfall Chart",
        "Sequential pipeline timing with bottleneck detection"
    )

    stages = [
        ("Loom Command", 12.5, "success"),
        ("Chrono Trigger", 8.3, "success"),
        ("Thread Select", 25.7, "success"),
        ("Resonance Shed", 45.2, "warning"),  # Bottleneck!
        ("Warp Space", 18.9, "success"),
        ("Convergence", 15.4, "success"),
        ("Tool Execute", 22.1, "success"),
        ("Spacetime", 6.8, "success"),
    ]

    total = sum(s[1] for s in stages)
    max_duration = max(s[1] for s in stages)

    print("  9-Step Weaving Cycle Waterfall:")
    print()
    print(f"  {'Stage':<15} {'Time':>8} {'Bar (scale: {:.0f}ms)':^40} Status".format(max_duration))
    print(f"  {'─' * 15} {'─' * 8} {'─' * 40} {'─' * 8}")

    cumulative = 0
    for name, duration, status in stages:
        # Calculate bar position and length
        bar_width = 35
        bar_len = int((duration / max_duration) * bar_width)
        offset = int((cumulative / total) * bar_width)

        # Create visual bar
        pre_space = " " * min(offset, bar_width - bar_len)
        bar = "█" * bar_len

        # Status indicator
        if status == "warning":
            status_str = "⚠ SLOW"
            bar = "▓" * bar_len  # Different pattern for bottleneck
        elif status == "error":
            status_str = "✗ ERR"
        else:
            status_str = "✓"

        print(f"  {name:<15} {duration:>6.1f}ms │{pre_space}{bar:<35}│ {status_str}")
        cumulative += duration

    print(f"  {'─' * 15} {'─' * 8} {'─' * 40} {'─' * 8}")
    print(f"  {'TOTAL':<15} {total:>6.1f}ms")
    print()

    # Identify bottleneck
    bottleneck = max(stages, key=lambda s: s[1])
    bottleneck_pct = (bottleneck[1] / total) * 100

    print(f"  ⚠ Bottleneck: {bottleneck[0]} ({bottleneck[1]:.1f}ms, {bottleneck_pct:.0f}% of total)")
    print()
    print("  Key Principle: Waterfall charts reveal sequential dependencies")
    print("  and highlight bottlenecks that limit overall performance.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMO 5: Confidence Trajectory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def demo_confidence_trajectory():
    """Demonstrate confidence tracking over queries."""
    section(
        "5. Confidence Trajectory",
        "Tracking confidence over time with anomaly detection"
    )

    random.seed(42)

    # Simulated query results
    queries = []
    base_conf = 0.85
    for i in range(15):
        # Some variation, with an anomaly at query 8
        if i == 7:
            conf = 0.45  # Anomaly!
        else:
            conf = base_conf + random.uniform(-0.1, 0.1)
            conf = max(0.5, min(0.98, conf))

        cached = random.random() > 0.4
        queries.append({
            "query": f"Query {i+1}",
            "confidence": conf,
            "cached": cached,
            "timestamp": datetime.now() - timedelta(minutes=(15-i)*5)
        })

    print("  Confidence over 15 queries:")
    print()
    print("  100% ┤")

    # ASCII chart
    height = 10
    for row in range(height, 0, -1):
        threshold = row / height
        line = f"  {int(threshold*100):>3}% │"
        for q in queries:
            conf = q["confidence"]
            if conf >= threshold:
                if q["cached"]:
                    line += "◆"  # Cached
                else:
                    line += "●"  # Fresh
            else:
                line += " "
        print(line)

    print("    0% └" + "─" * len(queries))
    print("        " + "".join([str(i+1)[0] if (i+1) < 10 else str((i+1)//10) for i in range(len(queries))]))
    print("        " + "".join([" " if (i+1) < 10 else str((i+1)%10) for i in range(len(queries))]))
    print()

    # Legend
    print("  Legend: ● Fresh query  ◆ Cached query")
    print()

    # Anomaly detection
    threshold = 0.6
    anomalies = [q for q in queries if q["confidence"] < threshold]

    if anomalies:
        print(f"  ⚠ Anomalies Detected: {len(anomalies)} queries below {threshold:.0%} threshold")
        for a in anomalies:
            print(f"    • {a['query']}: {a['confidence']:.1%} confidence")
    else:
        print("  ✓ No anomalies detected")

    # Statistics
    confs = [q["confidence"] for q in queries]
    avg_conf = sum(confs) / len(confs)
    cache_hits = sum(1 for q in queries if q["cached"])

    print()
    print(f"  Statistics:")
    print(f"    Mean confidence: {avg_conf:.1%}")
    print(f"    Cache hit rate: {cache_hits/len(queries):.1%}")
    print()
    print("  Key Principle: Trajectory charts reveal patterns over time,")
    print("  enabling early detection of performance degradation.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMO 6: Generate HTML Gallery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_html_gallery():
    """Generate a complete HTML visualization gallery."""
    section(
        "6. HTML Gallery Generation",
        "Self-contained HTML output for sharing"
    )

    random.seed(42)

    # Generate sample data for visualizations
    metrics = {
        "Query Latency (ms)": [145, 132, 128, 125, 138, 122, 119, 115, 118, 112],
        "Confidence (%)": [82, 85, 79, 88, 91, 87, 93, 89, 92, 94],
        "Memory (MB)": [456, 462, 471, 468, 489, 495, 502, 498, 512, 508],
    }

    html = '''<!DOCTYPE html>
<html>
<head>
    <title>HoloLoom Visualization Gallery</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem; background: #fafafa; color: #333; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; }
        h2 { color: #34495e; margin-top: 2rem; }
        .card { background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 1rem 0; }
        .metric { display: flex; align-items: center; gap: 1rem; padding: 0.5rem 0; border-bottom: 1px solid #eee; }
        .metric-name { width: 150px; font-weight: 500; }
        .metric-value { width: 80px; text-align: right; font-family: monospace; }
        .sparkline { flex: 1; }
        table { border-collapse: collapse; width: 100%; }
        th, td { text-align: left; padding: 0.5rem; border-bottom: 1px solid #ddd; }
        th { background: #f5f5f5; font-weight: 500; }
        .status-ok { color: #27ae60; }
        .status-warn { color: #f39c12; }
        .footer { text-align: center; color: #666; margin-top: 2rem; font-size: 0.9rem; }
    </style>
</head>
<body>
    <h1>HoloLoom Visualization Gallery</h1>
    <p>Tufte-style visualizations: "Above all else show the data."</p>

    <div class="card">
        <h2>Sparklines with Metrics</h2>
'''

    # Add sparkline metrics
    for name, values in metrics.items():
        current = values[-1]
        change = current - values[0]
        trend_dir = "▲" if change > 0 else "▼" if change < 0 else "─"

        # SVG sparkline
        min_val, max_val = min(values), max(values)
        val_range = max_val - min_val or 1
        width, height = 150, 25

        points = []
        for i, v in enumerate(values):
            x = (i / (len(values) - 1)) * (width - 4) + 2
            y = height - 2 - ((v - min_val) / val_range) * (height - 4)
            points.append(f"{x:.1f},{y:.1f}")

        svg = f'''<svg width="{width}" height="{height}">
            <polyline points="{' '.join(points)}" fill="none" stroke="#3498db" stroke-width="1.5"/>
            <circle cx="{points[0].split(',')[0]}" cy="{points[0].split(',')[1]}" r="2" fill="#95a5a6"/>
            <circle cx="{points[-1].split(',')[0]}" cy="{points[-1].split(',')[1]}" r="2" fill="#3498db"/>
        </svg>'''

        html += f'''
        <div class="metric">
            <span class="metric-name">{name}</span>
            <span class="metric-value">{current}</span>
            <span class="sparkline">{svg}</span>
            <span>{trend_dir} {abs(change):.0f}</span>
        </div>'''

    html += '''
    </div>

    <div class="card">
        <h2>Pipeline Stage Timing</h2>
        <table>
            <tr><th>Stage</th><th>Duration</th><th>%</th><th>Status</th></tr>
'''

    # Add pipeline stages
    stages = [
        ("Pattern Selection", 12.5),
        ("Thread Selection", 25.7),
        ("Feature Extraction", 45.2),
        ("Convergence", 15.4),
        ("Tool Execution", 22.1),
    ]
    total = sum(s[1] for s in stages)

    for name, duration in stages:
        pct = (duration / total) * 100
        status = "status-warn" if pct > 30 else "status-ok"
        status_text = "⚠ Bottleneck" if pct > 30 else "✓"
        html += f'''            <tr>
                <td>{name}</td>
                <td>{duration:.1f}ms</td>
                <td>{pct:.1f}%</td>
                <td class="{status}">{status_text}</td>
            </tr>
'''

    html += f'''            <tr style="font-weight: bold;">
                <td>Total</td>
                <td>{total:.1f}ms</td>
                <td>100%</td>
                <td></td>
            </tr>
        </table>
    </div>

    <div class="footer">
        <p>Generated by HoloLoom Visualization Gallery • {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        <p>Following Edward Tufte's principles: High data-ink ratio, zero chartjunk</p>
    </div>
</body>
</html>'''

    # Save HTML file
    output_dir = Path("demos/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "visualization_gallery.html"
    output_path.write_text(html)

    print(f"  ✓ Generated HTML gallery: {output_path}")
    print(f"  ✓ File size: {len(html):,} bytes")
    print(f"  ✓ Zero external dependencies (pure HTML/CSS/SVG)")
    print()
    print("  To view: Open demos/output/visualization_gallery.html in a browser")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║                                                               ║")
    print("║        HoloLoom Visualization Gallery                         ║")
    print("║        ──────────────────────────────                         ║")
    print("║        Tufte-Style Data Visualization                         ║")
    print("║                                                               ║")
    print("║        'Above all else show the data.'                        ║")
    print("║                          — Edward Tufte                       ║")
    print("║                                                               ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    t0 = time.time()

    try:
        # Demo 1: Sparklines
        demo_sparklines()

        # Demo 2: Small Multiples
        demo_small_multiples()

        # Demo 3: Data Density
        demo_density_table()

        # Demo 4: Waterfall
        demo_waterfall()

        # Demo 5: Confidence Trajectory
        demo_confidence_trajectory()

        # Demo 6: Generate HTML
        generate_html_gallery()

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.time() - t0

    # Summary
    section("Summary")
    print("  6 Tufte-style visualization demonstrations completed")
    print(f"  Execution time: {elapsed:.2f}s")
    print()
    print("  Key Takeaways:")
    print("    • Sparklines show trends in minimal space")
    print("    • Small multiples enable fair comparisons")
    print("    • Data density tables maximize information")
    print("    • Waterfalls reveal sequential dependencies")
    print("    • Trajectories detect anomalies over time")
    print()
    print("  HoloLoom's visualization system: 8,000+ lines implementing")
    print("  Tufte principles with zero external dependencies.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
