#!/usr/bin/env python3
"""
Intelligent Widget Builder Demo
================================
Shows the WidgetBuilder automatically constructing dashboards from raw data.

"Give me data, I'll build the dashboard" - No manual configuration needed!

This demo shows:
1. Bee winter survival (time-series, correlation, comparison)
2. Sales performance (categorical, trends)
3. Server metrics (outlier detection, distribution)

The WidgetBuilder automatically:
- Detects data types and patterns
- Selects optimal visualization types
- Generates intelligence insights
- Constructs complete dashboards

Author: Claude Code
Date: October 29, 2025
"""

import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.visualization import WidgetBuilder
from HoloLoom.visualization.html_renderer import HTMLRenderer


def demo_bee_survival():
    """
    Demo 1: Bee Winter Survival
    Auto-detects: time-series, correlation, trends
    """
    print("\n" + "="*80)
    print("DEMO 1: BEE WINTER SURVIVAL TRACKING")
    print("="*80)

    # Raw data (no pre-configuration!)
    data = {
        'month': ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'],
        'supplemental_feeding': [98, 96, 94, 91, 88, 85],
        'insulation_only': [97, 93, 88, 82, 76, 70],
        'control': [95, 88, 79, 68, 55, 42],
        'combined_treatment': [99, 98, 97, 95, 93, 91],
        'avg_temperature': [-2, -5, -8, -12, -10, -3]
    }

    print("\nInput data:")
    print(f"  - {len(data)} columns")
    print(f"  - {len(data['month'])} rows")
    print(f"  - Columns: {list(data.keys())}")

    # Build dashboard automatically
    builder = WidgetBuilder()
    dashboard = builder.build_from_data(
        data=data,
        title="Bee Colony Winter Survival Analysis",
        max_panels=10
    )

    # Render to HTML
    renderer = HTMLRenderer()
    html = renderer.render(dashboard)

    # Save
    output_dir = Path(__file__).parent / 'dashboards'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'widget_bee_survival.html'
    output_path.write_text(html, encoding='utf-8')

    print(f"\n[+] Dashboard auto-generated: {len(dashboard.panels)} panels")
    print(f"[+] File saved: {output_path}")
    print(f"[+] Size: {len(html)/1024:.1f} KB")

    return output_path


def demo_sales_performance():
    """
    Demo 2: Sales Performance
    Auto-detects: categorical comparison, trends, distribution
    """
    print("\n" + "="*80)
    print("DEMO 2: SALES PERFORMANCE BY REGION")
    print("="*80)

    # Raw data - sales across regions
    data = {
        'region': ['North', 'South', 'East', 'West', 'Central'],
        'q1_sales': [245000, 198000, 312000, 267000, 189000],
        'q2_sales': [267000, 215000, 334000, 289000, 201000],
        'q3_sales': [289000, 234000, 356000, 312000, 218000],
        'q4_sales': [334000, 267000, 398000, 345000, 245000],
        'growth_rate': [36.3, 34.8, 27.6, 29.2, 29.6]
    }

    print("\nInput data:")
    print(f"  - {len(data)} columns")
    print(f"  - {len(data['region'])} regions")
    print(f"  - Quarterly sales data")

    builder = WidgetBuilder()
    dashboard = builder.build_from_data(
        data=data,
        title="Regional Sales Performance Analysis",
        max_panels=10
    )

    renderer = HTMLRenderer()
    html = renderer.render(dashboard)

    output_dir = Path(__file__).parent / 'dashboards'
    output_path = output_dir / 'widget_sales.html'
    output_path.write_text(html, encoding='utf-8')

    print(f"\n[+] Dashboard auto-generated: {len(dashboard.panels)} panels")
    print(f"[+] File saved: {output_path}")
    print(f"[+] Size: {len(html)/1024:.1f} KB")

    return output_path


def demo_server_metrics():
    """
    Demo 3: Server Performance Metrics
    Auto-detects: outliers, distribution, time-series
    """
    print("\n" + "="*80)
    print("DEMO 3: SERVER PERFORMANCE MONITORING")
    print("="*80)

    # Raw data - server metrics with anomaly
    data = {
        'hour': list(range(24)),
        'cpu_usage': [45, 48, 52, 49, 51, 47, 46, 55, 58, 62, 67, 71,
                     95, 68, 64, 59, 54, 51, 48, 46, 44, 42, 40, 43],  # Spike at hour 12
        'memory_usage': [62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 74, 75,
                        77, 76, 75, 73, 71, 69, 67, 65, 63, 61, 60, 61],
        'requests_per_sec': [1200, 1250, 1300, 1280, 1320, 1290, 1310, 1400,
                           1450, 1520, 1580, 1620, 2890, 1560, 1480, 1420,
                           1380, 1340, 1290, 1240, 1200, 1180, 1160, 1190]  # Spike at hour 12
    }

    print("\nInput data:")
    print(f"  - {len(data)} metrics")
    print(f"  - {len(data['hour'])} hourly readings")
    print(f"  - Metrics: CPU, memory, requests/sec")

    builder = WidgetBuilder()
    dashboard = builder.build_from_data(
        data=data,
        title="Server Performance Monitoring - 24 Hour View",
        max_panels=10
    )

    renderer = HTMLRenderer()
    html = renderer.render(dashboard)

    output_dir = Path(__file__).parent / 'dashboards'
    output_path = output_dir / 'widget_server.html'
    output_path.write_text(html, encoding='utf-8')

    print(f"\n[+] Dashboard auto-generated: {len(dashboard.panels)} panels")
    print(f"[+] File saved: {output_path}")
    print(f"[+] Size: {len(html)/1024:.1f} KB")

    return output_path


def main():
    """Run all widget builder demos."""

    print("\n" + "="*80)
    print("INTELLIGENT WIDGET BUILDER DEMO")
    print("="*80)
    print("""
    The WidgetBuilder automatically:
    1. Analyzes data types (numeric, categorical, temporal)
    2. Detects patterns (trends, correlations, outliers)
    3. Selects optimal visualizations
    4. Generates intelligence insights
    5. Constructs complete dashboards

    NO MANUAL CONFIGURATION NEEDED!
    """)

    # Run demos
    bee_path = demo_bee_survival()
    sales_path = demo_sales_performance()
    server_path = demo_server_metrics()

    # Summary
    print("\n" + "="*80)
    print("ALL DASHBOARDS AUTO-GENERATED")
    print("="*80)
    print(f"""
    Generated 3 dashboards from raw data:

    1. Bee Survival:     file://{bee_path.absolute()}
       - Time-series line charts
       - Correlation scatter plots
       - Trend insights

    2. Sales Performance: file://{sales_path.absolute()}
       - Categorical bar charts
       - Growth metrics
       - Regional comparisons

    3. Server Metrics:    file://{server_path.absolute()}
       - Multi-series line charts
       - Outlier detection
       - Performance insights

    Each dashboard was built automatically from just:
      - Dictionary of column_name -> values
      - Title string
      - No visualization configuration!

    The WidgetBuilder detected:
      - Data types (numeric, categorical, temporal)
      - Patterns (trends, correlations, outliers, time-series)
      - Optimal chart types (line, scatter, bar, metric)
      - Intelligence insights (with confidence scores)

    Open the HTML files to explore the auto-generated dashboards!
    """)

    print("="*80)


if __name__ == '__main__':
    main()
