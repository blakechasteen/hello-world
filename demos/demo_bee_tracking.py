#!/usr/bin/env python3
"""
Bee Winter Survival Tracking Dashboard Demo
============================================
Demonstrates new visualization types:
- Scatter plots (correlation analysis)
- Line charts (time-series trends)
- Bar charts (categorical comparisons)
- Insight cards (auto-detected intelligence)

Example query: "How are my bees doing with winter survival treatments?"

Author: Claude Code
Date: October 29, 2025
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.visualization import (
    Dashboard, Panel, PanelType, PanelSize, LayoutType
)
from HoloLoom.visualization.html_renderer import HTMLRenderer


# Mock Spacetime for demo
@dataclass
class MockSpacetime:
    query_text: str = "How are my bees doing with winter survival treatments?"
    response: str = "Analysis of bee colony winter survival data"
    tool_used: str = "data_analysis"
    confidence: float = 0.92
    trace: Any = None
    metadata: Dict[str, Any] = None

    def to_dict(self):
        return {
            'query': self.query_text,
            'response': self.response,
            'tool': self.tool_used,
            'confidence': self.confidence
        }


@dataclass
class MockTrace:
    duration_ms: float = 345.7
    stages: list = None


def create_bee_survival_dashboard():
    """
    Create comprehensive bee tracking dashboard with new visualization types.

    Story: Beekeeper is tracking winter survival rates across different
    treatment groups and environmental conditions.
    """

    print("\n" + "="*80)
    print("BEE WINTER SURVIVAL TRACKING DASHBOARD")
    print("="*80)

    # Mock data - realistic bee survival tracking
    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']

    # Treatment groups
    treatments = {
        'Supplemental Feeding': {
            'survival': [98, 96, 94, 91, 88, 85],
            'color': '#10b981'  # green
        },
        'Insulation Only': {
            'survival': [97, 93, 88, 82, 76, 70],
            'color': '#f59e0b'  # yellow
        },
        'Control (No Treatment)': {
            'survival': [95, 88, 79, 68, 55, 42],
            'color': '#ef4444'  # red
        },
        'Combined Treatment': {
            'survival': [99, 98, 97, 95, 93, 91],
            'color': '#8b5cf6'  # purple
        }
    }

    # Temperature data
    avg_temps = [-2, -5, -8, -12, -10, -3]  # Celsius

    # Create spacetime mock
    spacetime = MockSpacetime()
    spacetime.trace = MockTrace()
    spacetime.metadata = {
        'complexity': 'FULL',
        'panel_count': 0  # Will be set
    }

    panels = []

    # ========================================================================
    # 1. KEY METRICS
    # ========================================================================

    panels.append(Panel(
        id="metric_best_survival",
        type=PanelType.METRIC,
        title="Best Treatment",
        subtitle="Combined approach",
        data={
            'value': 91,
            'formatted': '91%',
            'label': 'Survival Rate (Mar)',
            'color': 'green',
            'unit': '%'
        },
        size=PanelSize.SMALL
    ))

    panels.append(Panel(
        id="metric_colonies",
        type=PanelType.METRIC,
        title="Colonies Tracked",
        data={
            'value': 24,
            'formatted': '24',
            'label': 'Total Colonies',
            'color': 'blue'
        },
        size=PanelSize.SMALL
    ))

    panels.append(Panel(
        id="metric_avg_temp",
        type=PanelType.METRIC,
        title="Avg Winter Temp",
        data={
            'value': -6.7,
            'formatted': '-6.7°C',
            'label': 'Oct-Mar Average',
            'color': 'purple'
        },
        size=PanelSize.SMALL
    ))

    # ========================================================================
    # 2. LINE CHART - Survival Trends Over Time
    # ========================================================================

    print("\n[1/7] Creating survival trends line chart...")

    traces_data = []
    for treatment_name, treatment_data in treatments.items():
        traces_data.append({
            'name': treatment_name,
            'x': months,
            'y': treatment_data['survival'],
            'color': treatment_data['color']
        })

    panels.append(Panel(
        id="line_survival_trends",
        type=PanelType.LINE,
        title="Colony Survival Rates Over Winter",
        subtitle="Percentage of colonies surviving by treatment group",
        data={
            'traces': traces_data,
            'x_label': 'Month',
            'y_label': 'Survival Rate (%)',
            'show_points': True
        },
        size=PanelSize.FULL_WIDTH
    ))

    # ========================================================================
    # 3. SCATTER PLOT - Temperature vs Survival Correlation
    # ========================================================================

    print("[2/7] Creating temperature correlation scatter plot...")

    # Generate data points (temperature vs survival for each month)
    scatter_x = []  # Temperature
    scatter_y = []  # Average survival across all treatments
    scatter_labels = []
    scatter_colors = []

    for i, month in enumerate(months):
        temp = avg_temps[i]
        avg_survival = sum(t['survival'][i] for t in treatments.values()) / len(treatments)

        scatter_x.append(temp)
        scatter_y.append(avg_survival)
        scatter_labels.append(f"{month}: {temp}°C, {avg_survival:.1f}% survival")

        # Color by temperature (blue = cold, red = warm)
        if temp < -10:
            scatter_colors.append('#3b82f6')  # blue
        elif temp < -5:
            scatter_colors.append('#8b5cf6')  # purple
        else:
            scatter_colors.append('#f59e0b')  # orange

    # Calculate correlation
    import math
    n = len(scatter_x)
    sum_x = sum(scatter_x)
    sum_y = sum(scatter_y)
    sum_xy = sum(x * y for x, y in zip(scatter_x, scatter_y))
    sum_x2 = sum(x * x for x in scatter_x)
    sum_y2 = sum(y * y for y in scatter_y)

    correlation = (n * sum_xy - sum_x * sum_y) / math.sqrt(
        (n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)
    )

    panels.append(Panel(
        id="scatter_temp_survival",
        type=PanelType.SCATTER,
        title="Temperature Impact on Survival",
        subtitle="Correlation between average temperature and colony survival",
        data={
            'x': scatter_x,
            'y': scatter_y,
            'labels': scatter_labels,
            'x_label': 'Average Temperature (°C)',
            'y_label': 'Average Survival Rate (%)',
            'colors': scatter_colors,
            'sizes': [12] * len(scatter_x),
            'correlation': correlation
        },
        size=PanelSize.LARGE
    ))

    # ========================================================================
    # 4. BAR CHART - Treatment Effectiveness Comparison (Final Month)
    # ========================================================================

    print("[3/7] Creating treatment comparison bar chart...")

    final_survivals = [t['survival'][-1] for t in treatments.values()]
    treatment_names = list(treatments.keys())
    bar_colors = [t['color'] for t in treatments.values()]

    panels.append(Panel(
        id="bar_treatments",
        type=PanelType.BAR,
        title="Treatment Effectiveness (March Results)",
        subtitle="Final survival rates by treatment approach",
        data={
            'categories': treatment_names,
            'values': final_survivals,
            'orientation': 'h',  # Horizontal for long labels
            'x_label': 'Survival Rate (%)',
            'y_label': 'Treatment',
            'colors': bar_colors
        },
        size=PanelSize.LARGE
    ))

    # ========================================================================
    # 5. INSIGHT CARD - Trend Detection
    # ========================================================================

    print("[4/7] Generating trend insight...")

    panels.append(Panel(
        id="insight_trend",
        type=PanelType.INSIGHT,
        title="Winter Mortality Accelerates in January",
        data={
            'type': 'trend',
            'message': 'All treatment groups show steepest mortality decline between December and January, coinciding with the coldest temperatures. Combined treatment group maintains >90% survival throughout.',
            'confidence': 0.89,
            'details': {
                'Critical Period': 'Dec-Jan',
                'Avg Temp': '-12°C',
                'Mortality Increase': '2-3x normal rate',
                'Best Practice': 'Increase feeding in late December'
            }
        },
        size=PanelSize.MEDIUM
    ))

    # ========================================================================
    # 6. INSIGHT CARD - Correlation Discovery
    # ========================================================================

    print("[5/7] Generating correlation insight...")

    panels.append(Panel(
        id="insight_correlation",
        type=PanelType.INSIGHT,
        title="Strong Temperature-Survival Correlation",
        data={
            'type': 'correlation',
            'message': f'Temperature strongly correlates with survival (r={correlation:.3f}). Each 1°C drop corresponds to ~2.3% decrease in survival rate on average.',
            'confidence': 0.92,
            'details': {
                'Correlation': f'{correlation:.3f}',
                'Significance': 'p < 0.01',
                'Effect Size': '2.3% per °C',
                'Recommendation': 'Monitor temps closely'
            }
        },
        size=PanelSize.MEDIUM
    ))

    # ========================================================================
    # 7. INSIGHT CARD - Recommendation
    # ========================================================================

    print("[6/7] Generating recommendations...")

    panels.append(Panel(
        id="insight_recommendation",
        type=PanelType.INSIGHT,
        title="Combined Treatment Shows Superior Results",
        data={
            'type': 'recommendation',
            'message': 'Combining supplemental feeding with insulation yields 21% higher survival than insulation alone and 49% higher than control. Cost-benefit analysis suggests this is optimal.',
            'confidence': 0.95,
            'details': {
                'Improvement vs Control': '+49 percentage points',
                'Improvement vs Insulation': '+21 percentage points',
                'Cost per Colony': '$45 (feeding) + $30 (insulation)',
                'ROI': 'High (colony replacement: $200+)'
            }
        },
        size=PanelSize.MEDIUM
    ))

    # ========================================================================
    # Create Dashboard
    # ========================================================================

    print("[7/7] Assembling dashboard...")

    spacetime.metadata['panel_count'] = len(panels)

    dashboard = Dashboard(
        title="Bee Colony Winter Survival Analysis 2024-2025",
        layout=LayoutType.RESEARCH,  # 3-column grid for comprehensive view
        panels=panels,
        spacetime=spacetime,
        metadata={
            'complexity': 'FULL',
            'panel_count': len(panels),
            'generated_at': datetime.now().isoformat(),
            'data_period': 'October 2024 - March 2025',
            'colonies_tracked': 24
        }
    )

    print(f"\n[+] Dashboard created: {len(panels)} panels")
    print(f"    - {sum(1 for p in panels if p.type == PanelType.METRIC)} metrics")
    print(f"    - {sum(1 for p in panels if p.type == PanelType.LINE)} line charts")
    print(f"    - {sum(1 for p in panels if p.type == PanelType.SCATTER)} scatter plots")
    print(f"    - {sum(1 for p in panels if p.type == PanelType.BAR)} bar charts")
    print(f"    - {sum(1 for p in panels if p.type == PanelType.INSIGHT)} insights")

    return dashboard


def main():
    """Run the bee tracking demo."""

    # Create output directory
    output_dir = Path(__file__).parent / 'dashboards'
    output_dir.mkdir(exist_ok=True)

    # Generate dashboard
    dashboard = create_bee_survival_dashboard()

    # Render to HTML
    print("\n" + "="*80)
    print("RENDERING DASHBOARD")
    print("="*80)

    renderer = HTMLRenderer()
    html = renderer.render(dashboard)

    # Save to file
    output_path = output_dir / 'demo_bee_tracking.html'
    output_path.write_text(html, encoding='utf-8')

    file_size_kb = len(html) / 1024

    print(f"\n[+] Dashboard saved: {output_path}")
    print(f"[+] File size: {file_size_kb:.1f} KB")

    print("\n" + "="*80)
    print("DASHBOARD FEATURES")
    print("="*80)
    print("""
    Visualization Types Demonstrated:

    1. METRICS (3 cards)
       - Best survival rate
       - Total colonies tracked
       - Average winter temperature

    2. LINE CHART (multi-series)
       - 4 treatment groups over 6 months
       - Interactive hover tooltips
       - Zoom and pan enabled

    3. SCATTER PLOT (correlation analysis)
       - Temperature vs survival rate
       - Auto-calculated correlation coefficient
       - Color-coded by temperature

    4. BAR CHART (categorical comparison)
       - Horizontal orientation (long labels)
       - Final month survival by treatment
       - Color-coded by treatment group

    5. INSIGHT CARDS (3 intelligence findings)
       - Trend: Winter mortality patterns
       - Correlation: Temperature impact
       - Recommendation: Best treatment approach

    Interactive Features:
    - Click expand buttons to toggle panels
    - Click panels for drill-down details
    - Hover over charts for data points
    - Drag to zoom on Plotly charts
    - Settings button for preferences
    - Dark mode support

    Open in browser to explore!
    """)

    print(f"\nQuick open:\n  file://{output_path.absolute()}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
