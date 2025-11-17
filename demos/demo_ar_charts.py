"""Demo: AR Data Visualization Charts

Demonstrates all 6 chart types rendered in AR space:
- Bar charts (categorical data)
- Line charts (time series)
- Pie charts (distribution)
- Gauges (progress/status)
- Histograms (frequency distribution)
- Scatter plots (correlation)

Real-time data updates with LiveChartUpdater.

Implemented: 2025-11-17
Run: PYTHONPATH=. python demos/demo_ar_charts.py
"""

import asyncio
import numpy as np
from typing import List

from HoloLoom.visualization.ar_charts import (
    ARChart,
    ChartType,
    ChartConfig,
    ARChartRenderer,
    GaugeData,
    GaugeStyle,
    LiveChartUpdater,
)


async def create_dummy_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a dummy camera frame."""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 200
    return frame


async def demo_bar_chart():
    """Demo: Bar chart for quarterly sales."""
    print("\n1. BAR CHART - Quarterly Sales")
    print("-" * 50)

    chart = ARChart(
        chart_type=ChartType.BAR,
        data=[150000, 180000, 165000, 220000],
        labels=["Q1", "Q2", "Q3", "Q4"],
        position=(0, 0, 1),
        config=ChartConfig(
            title="2025 Sales Report",
            width=350,
            height=250,
            show_legend=True,
            show_values=True,
        ),
    )

    print(f"Chart type: {chart.chart_type.value}")
    print(f"Data: {chart.data}")
    print(f"Labels: {chart.labels}")
    print(f"Position: {chart.position}")

    # Render
    renderer = ARChartRenderer()
    frame = await create_dummy_frame()
    result = await renderer.render_chart(chart, frame, (0, 0, 0))

    print(f"Rendered successfully: {result.shape}")
    print(f"Data range: ${min(chart.data):,} - ${max(chart.data):,}")
    print(f"Growth: {((chart.data[-1] - chart.data[0]) / chart.data[0] * 100):.1f}%")


async def demo_line_chart():
    """Demo: Line chart for trend analysis."""
    print("\n2. LINE CHART - Trend Analysis")
    print("-" * 50)

    # Simulated daily temperature readings
    data = [22.5, 23.1, 22.8, 24.2, 25.1, 24.9, 23.7]

    chart = ARChart(
        chart_type=ChartType.LINE,
        data=data,
        labels=[f"Day {i+1}" for i in range(len(data))],
        position=(0, 0, 1),
        config=ChartConfig(
            title="Temperature Trend (°C)",
            width=350,
            height=250,
            show_grid=True,
            show_values=True,
        ),
    )

    print(f"Chart type: {chart.chart_type.value}")
    print(f"Data points: {len(chart.data)}")
    print(f"Min temperature: {min(chart.data):.1f}°C")
    print(f"Max temperature: {max(chart.data):.1f}°C")
    print(f"Average: {np.mean(chart.data):.1f}°C")

    # Render
    renderer = ARChartRenderer()
    frame = await create_dummy_frame()
    result = await renderer.render_chart(chart, frame, (0, 0, 0))

    print(f"Rendered successfully: {result.shape}")
    print(f"Trend: {'📈 Upward' if chart.data[-1] > chart.data[0] else '📉 Downward'}")


async def demo_pie_chart():
    """Demo: Pie chart for market share."""
    print("\n3. PIE CHART - Market Share Distribution")
    print("-" * 50)

    companies = ["Company A", "Company B", "Company C", "Company D"]
    market_share = [35, 28, 22, 15]

    chart = ARChart(
        chart_type=ChartType.PIE,
        data=market_share,
        labels=companies,
        position=(0, 0, 1),
        config=ChartConfig(
            title="Market Share 2025",
            width=300,
            height=300,
            show_legend=True,
        ),
    )

    print(f"Chart type: {chart.chart_type.value}")
    print(f"Segments: {len(chart.labels)}")
    print("Distribution:")
    for label, value in zip(chart.labels, chart.data):
        pct = value / sum(chart.data) * 100
        print(f"  {label}: {value}% ({pct:.1f}%)")

    # Render
    renderer = ARChartRenderer()
    frame = await create_dummy_frame()
    result = await renderer.render_chart(chart, frame, (0, 0, 0))

    print(f"Rendered successfully: {result.shape}")
    print(f"Total: {sum(chart.data)}%")


async def demo_gauge_chart():
    """Demo: Gauge chart for system health."""
    print("\n4. GAUGE CHART - System Health")
    print("-" * 50)

    gauge_data = GaugeData(
        value=78.5,
        min_value=0.0,
        max_value=100.0,
        unit="%",
        style=GaugeStyle.CIRCULAR,
    )

    chart = ARChart(
        chart_type=ChartType.GAUGE,
        data=[gauge_data],
        position=(0, 0, 1),
        config=ChartConfig(
            title="System Health",
            width=300,
            height=300,
        ),
    )

    print(f"Chart type: {chart.chart_type.value}")
    print(f"Value: {gauge_data.value}{gauge_data.unit}")
    print(f"Range: {gauge_data.min_value} - {gauge_data.max_value}")
    print(f"Style: {gauge_data.style.value}")

    # Determine health status
    if gauge_data.value >= 80:
        status = "✓ Excellent"
    elif gauge_data.value >= 60:
        status = "~ Good"
    else:
        status = "✗ Warning"

    print(f"Status: {status}")

    # Render
    renderer = ARChartRenderer()
    frame = await create_dummy_frame()
    result = await renderer.render_chart(chart, frame, (0, 0, 0))

    print(f"Rendered successfully: {result.shape}")


async def demo_histogram():
    """Demo: Histogram for age distribution."""
    print("\n5. HISTOGRAM - Age Distribution")
    print("-" * 50)

    # Age group frequencies (18-25, 26-35, 36-45, 46-55, 56-65, 65+)
    frequencies = [15, 28, 32, 24, 18, 12]
    age_groups = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]

    chart = ARChart(
        chart_type=ChartType.HISTOGRAM,
        data=frequencies,
        labels=age_groups,
        position=(0, 0, 1),
        config=ChartConfig(
            title="Population by Age Group",
            width=350,
            height=250,
            show_values=True,
        ),
    )

    print(f"Chart type: {chart.chart_type.value}")
    print(f"Age groups: {len(chart.labels)}")
    print("Distribution:")
    total = sum(frequencies)
    for label, freq in zip(age_groups, frequencies):
        pct = freq / total * 100
        print(f"  {label}: {freq} ({pct:.1f}%)")

    # Render
    renderer = ARChartRenderer()
    frame = await create_dummy_frame()
    result = await renderer.render_chart(chart, frame, (0, 0, 0))

    print(f"Rendered successfully: {result.shape}")
    print(f"Total population: {total}")
    print(f"Most common group: {age_groups[frequencies.index(max(frequencies))]}")


async def demo_scatter_chart():
    """Demo: Scatter plot for correlation analysis."""
    print("\n6. SCATTER PLOT - Correlation Analysis")
    print("-" * 50)

    # Study hours vs exam scores
    data = [
        (2, 45),
        (3, 52),
        (4, 58),
        (5, 65),
        (6, 70),
        (7, 78),
        (8, 82),
        (9, 88),
        (10, 92),
    ]

    chart = ARChart(
        chart_type=ChartType.SCATTER,
        data=data,
        position=(0, 0, 1),
        config=ChartConfig(
            title="Study Hours vs Exam Score",
            width=350,
            height=250,
            show_grid=True,
        ),
    )

    print(f"Chart type: {chart.chart_type.value}")
    print(f"Data points: {len(data)}")
    print("Sample data:")
    for x, y in data[::2]:  # Show every 2nd point
        print(f"  ({x} hours, {y} score)")

    # Calculate correlation
    x_vals = [p[0] for p in data]
    y_vals = [p[1] for p in data]
    correlation = np.corrcoef(x_vals, y_vals)[0, 1]

    print(f"Correlation: {correlation:.3f} (strong positive)")

    # Render
    renderer = ARChartRenderer()
    frame = await create_dummy_frame()
    result = await renderer.render_chart(chart, frame, (0, 0, 0))

    print(f"Rendered successfully: {result.shape}")


async def demo_live_chart_updater():
    """Demo: Live chart updates with real-time data."""
    print("\n7. LIVE CHART UPDATER - Real-time Data")
    print("-" * 50)

    updater = LiveChartUpdater()

    # Register initial chart
    initial_data = [100, 120, 115, 140, 150]
    chart = ARChart(
        chart_type=ChartType.BAR,
        data=initial_data,
        labels=["Mon", "Tue", "Wed", "Thu", "Fri"],
    )

    await updater.register_chart("daily_sales", chart)
    print("Registered chart: daily_sales")
    print(f"Initial data: {initial_data}")

    # Simulate data updates
    print("\nSimulating live updates...")
    update_sequence = [
        [105, 125, 118, 145, 155],
        [110, 130, 120, 150, 160],
        [115, 135, 125, 155, 165],
    ]

    for i, new_data in enumerate(update_sequence, 1):
        await updater.update_data("daily_sales", new_data)
        retrieved = await updater.get_chart("daily_sales")

        avg = np.mean(new_data)
        max_val = max(new_data)
        print(f"  Update {i}: avg=${avg:.0f}, peak=${max_val}")

    print("Live update demo complete")


async def main():
    """Main demo function."""
    print("=" * 70)
    print("AR DATA VISUALIZATION CHARTS DEMO")
    print("=" * 70)
    print()

    print("This demo showcases 6 chart types in AR:")
    print("  1. Bar charts - Categorical data comparison")
    print("  2. Line charts - Time series trends")
    print("  3. Pie charts - Compositional data")
    print("  4. Gauges - Status/progress indicators")
    print("  5. Histograms - Frequency distributions")
    print("  6. Scatter plots - Correlation analysis")
    print()

    # Run all demos
    try:
        await demo_bar_chart()
        await demo_line_chart()
        await demo_pie_chart()
        await demo_gauge_chart()
        await demo_histogram()
        await demo_scatter_chart()
        await demo_live_chart_updater()
    except Exception as e:
        print(f"Error during demo: {e}")
        return

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print("  ✓ Demonstrated 6 chart types")
    print("  ✓ Showed real-world data examples")
    print("  ✓ Demonstrated live chart updates")
    print()
    print("Key Features:")
    print("  ✓ Multiple chart types with consistent API")
    print("  ✓ Flexible configuration (titles, legends, grids)")
    print("  ✓ Automatic data normalization")
    print("  ✓ Real-time data updates via LiveChartUpdater")
    print("  ✓ Color support for visual distinction")
    print("  ✓ Value labeling for data clarity")
    print()


if __name__ == "__main__":
    print()
    asyncio.run(main())
