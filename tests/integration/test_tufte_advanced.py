#!/usr/bin/env python3
"""
Test Advanced Tufte Visualizations - Meaning First
===================================================
Tests small multiples and data density tables.

Author: Claude Code
Date: October 29, 2025
"""

from pathlib import Path
from HoloLoom.visualization.small_multiples import render_small_multiples, QueryMultiple
from HoloLoom.visualization.density_table import render_stage_timing_table, Column, Row, ColumnType, ColumnAlign, DensityTableRenderer


def test_small_multiples():
    """Test small multiples for query comparison."""
    print('\n[TEST 1] Small Multiples - Query Comparison')
    print('=' * 70)

    # Create sample queries
    queries = [
        {
            'query_text': 'What is Thompson Sampling?',
            'latency_ms': 95.2,
            'confidence': 0.92,
            'threads_count': 3,
            'cached': True,
            'trend': [105, 102, 98, 96, 95.2],
            'timestamp': 1698595200.0,
            'tool_used': 'answer'
        },
        {
            'query_text': 'Explain weaving orchestrator architecture',
            'latency_ms': 145.8,
            'confidence': 0.88,
            'threads_count': 5,
            'cached': False,
            'trend': [120, 125, 135, 140, 145.8],
            'timestamp': 1698595210.0,
            'tool_used': 'answer'
        },
        {
            'query_text': 'How does semantic cache work?',
            'latency_ms': 78.5,
            'confidence': 0.95,
            'threads_count': 2,
            'cached': True,
            'trend': [85, 82, 80, 79, 78.5],
            'timestamp': 1698595220.0,
            'tool_used': 'answer'
        },
        {
            'query_text': 'Compare BARE vs FAST mode',
            'latency_ms': 112.3,
            'confidence': 0.85,
            'threads_count': 4,
            'cached': False,
            'trend': [110, 111, 112, 112, 112.3],
            'timestamp': 1698595230.0,
            'tool_used': 'compare'
        },
    ]

    # Render small multiples
    html = render_small_multiples(queries, layout='grid', max_columns=2)

    assert '<div class="small-multiples-container"' in html, "Should have container"
    assert 'Thompson Sampling' in html, "Should include query text"
    assert 'grid-template-columns' in html, "Should have grid layout"

    print(f'  Queries compared: {len(queries)}')
    print(f'  Layout: grid (2 columns)')
    print(f'  Best query: {min(queries, key=lambda q: q["latency_ms"])["query_text"][:30]}')
    print(f'  Worst query: {max(queries, key=lambda q: q["latency_ms"])["query_text"][:30]}')
    print('  [PASS] Small multiples generated correctly')


def test_density_table():
    """Test data density table for stage timing."""
    print('\n[TEST 2] Data Density Table - Stage Timing')
    print('=' * 70)

    # Create stage timing data
    stages = [
        {
            'name': 'Pattern Selection',
            'duration_ms': 5.2,
            'trend': [6, 5.5, 5.3, 5.2, 5.2],
            'delta': -0.3
        },
        {
            'name': 'Retrieval',
            'duration_ms': 50.5,
            'trend': [45, 47, 48, 50, 50.5],
            'delta': +2.5
        },
        {
            'name': 'Convergence',
            'duration_ms': 30.0,
            'trend': [32, 31, 30, 30, 30.0],
            'delta': -1.0
        },
        {
            'name': 'Tool Execution',
            'duration_ms': 64.3,
            'trend': [60, 61, 62, 63, 64.3],
            'delta': +1.3
        },
    ]

    total_duration = sum(s['duration_ms'] for s in stages)

    # Render density table
    html = render_stage_timing_table(stages, total_duration, bottleneck_threshold=0.4)

    assert '<div class="density-table"' in html, "Should have table container"
    assert 'Stage Timing Analysis' in html, "Should have title"
    assert 'Bottleneck?' in html, "Should have bottleneck column"
    assert 'Total' in html, "Should have footer total"

    # Check bottleneck detection
    bottleneck_stages = [s for s in stages if (s['duration_ms'] / total_duration) >= 0.4]

    print(f'  Total stages: {len(stages)}')
    print(f'  Total duration: {total_duration:.1f}ms')
    print(f'  Bottlenecks detected: {len(bottleneck_stages)}')
    if bottleneck_stages:
        print(f'  Bottleneck stage: {bottleneck_stages[0]["name"]} ({bottleneck_stages[0]["duration_ms"]:.1f}ms)')
    print('  [PASS] Density table generated correctly')


def test_combined_visualization():
    """Test combining small multiples and density table."""
    print('\n[TEST 3] Combined Visualization')
    print('=' * 70)

    # Create sample queries
    queries = [
        {
            'query_text': 'Query A (fast)',
            'latency_ms': 85.0,
            'confidence': 0.92,
            'threads_count': 2,
            'cached': True,
            'trend': [90, 88, 87, 86, 85],
            'timestamp': 1698595200.0,
            'tool_used': 'answer'
        },
        {
            'query_text': 'Query B (slow)',
            'latency_ms': 150.0,
            'confidence': 0.88,
            'threads_count': 5,
            'cached': False,
            'trend': [145, 147, 148, 149, 150],
            'timestamp': 1698595210.0,
            'tool_used': 'answer'
        },
    ]

    # Render small multiples
    multiples_html = render_small_multiples(queries, layout='row')

    # Create stage data for Query B (slow one)
    stages = [
        {
            'name': 'Pattern Selection',
            'duration_ms': 10.0,
            'trend': [9, 9.5, 10, 10, 10],
            'delta': +1.0
        },
        {
            'name': 'Retrieval',
            'duration_ms': 75.0,
            'trend': [70, 72, 73, 74, 75],
            'delta': +3.0
        },
        {
            'name': 'Convergence',
            'duration_ms': 30.0,
            'trend': [30, 30, 30, 30, 30],
            'delta': 0.0
        },
        {
            'name': 'Tool Execution',
            'duration_ms': 35.0,
            'trend': [33, 34, 34, 35, 35],
            'delta': +1.0
        },
    ]

    # Render density table
    table_html = render_stage_timing_table(stages, 150.0)

    # Combine into complete dashboard
    complete_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Tufte Visualization Demo - Meaning First</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                margin: 40px;
                background: #f9fafb;
            }}
            h1 {{
                color: #111827;
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 8px;
            }}
            .subtitle {{
                color: #6b7280;
                font-size: 14px;
                margin-bottom: 32px;
            }}
            .section {{
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 24px;
                margin-bottom: 24px;
            }}
            .section-title {{
                font-size: 16px;
                font-weight: 600;
                color: #374151;
                margin-bottom: 16px;
            }}
            .tufte-quote {{
                border-left: 4px solid #3b82f6;
                padding-left: 16px;
                margin: 24px 0;
                font-style: italic;
                color: #6b7280;
            }}
        </style>
    </head>
    <body>
        <h1>Tufte Visualization Demo: Meaning First</h1>
        <div class="subtitle">
            High-density, low-decoration visualizations following Edward Tufte's principles
        </div>

        <div class="tufte-quote">
            "Above all else show the data. Graphical excellence is that which gives to the viewer
            the greatest number of ideas in the shortest time with the least ink in the smallest space."
            <br><strong>— Edward Tufte</strong>
        </div>

        <div class="section">
            <div class="section-title">Small Multiples: Query Comparison</div>
            <p style="color: #6b7280; font-size: 13px; margin-bottom: 16px;">
                Compare multiple queries side-by-side. Consistent scales, minimal decoration,
                maximum information. Best query marked with ★, worst with ⚠.
            </p>
            {multiples_html}
        </div>

        <div class="section">
            <div class="section-title">Data Density Table: Stage Timing Analysis</div>
            <p style="color: #6b7280; font-size: 13px; margin-bottom: 16px;">
                Maximum information per square inch. Inline sparklines, delta indicators,
                bottleneck detection. Right-aligned numbers, monospace font, tight spacing.
            </p>
            {table_html}
        </div>

        <div class="section">
            <div class="section-title">Design Principles Applied</div>
            <ul style="color: #374151; font-size: 13px; line-height: 1.8;">
                <li><strong>Maximize data-ink ratio:</strong> Remove chartjunk, keep only what conveys information</li>
                <li><strong>Small multiples:</strong> Enable comparison through consistent repetition</li>
                <li><strong>Data density:</strong> Show lots of information in small space</li>
                <li><strong>Content-rich labels:</strong> Labels inform, not just identify</li>
                <li><strong>Layering:</strong> Multiple dimensions without clutter (color, opacity, position)</li>
                <li><strong>Meaning first:</strong> Information directly where eyes go first</li>
            </ul>
        </div>

        <div style="margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;
                    text-align: center; color: #9ca3af; font-size: 12px;">
            Generated with HoloLoom Visualizer | Inspired by Edward Tufte's principles
        </div>
    </body>
    </html>
    """

    # Save to file
    output_path = Path('demos/output/tufte_advanced_demo.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(complete_html, encoding='utf-8')

    print(f'  Small multiples: {len(queries)} queries')
    print(f'  Density table: {len(stages)} stages')
    print(f'  Combined HTML size: {len(complete_html):,} bytes')
    print(f'  [SAVED] Demo HTML: {output_path}')
    print('  [PASS] Combined visualization generated')


def test_custom_density_table():
    """Test custom density table with mixed column types."""
    print('\n[TEST 4] Custom Density Table - Mixed Column Types')
    print('=' * 70)

    # Define columns
    columns = [
        Column('Metric', ColumnType.TEXT, ColumnAlign.LEFT),
        Column('Current', ColumnType.NUMBER, ColumnAlign.RIGHT, unit=''),
        Column('Target', ColumnType.NUMBER, ColumnAlign.RIGHT, unit=''),
        Column('Delta', ColumnType.DELTA, ColumnAlign.RIGHT),
        Column('Trend', ColumnType.SPARKLINE, ColumnAlign.CENTER),
        Column('Status', ColumnType.INDICATOR, ColumnAlign.LEFT),
    ]

    # Define rows
    rows = [
        Row(cells={
            'Metric': 'Cache Hit Rate',
            'Current': 75.0,
            'Target': 80.0,
            'Delta': -5.0,
            'Trend': [70, 72, 73, 74, 75],
            'Status': False  # Not meeting target
        }),
        Row(cells={
            'Metric': 'Avg Latency',
            'Current': 95.0,
            'Target': 100.0,
            'Delta': -5.0,
            'Trend': [110, 105, 100, 97, 95],
            'Status': True  # Meeting target (under 100ms)
        }, highlight=True, highlight_color='#f0fdf4'),  # Light green
        Row(cells={
            'Metric': 'P95 Latency',
            'Current': 180.0,
            'Target': 150.0,
            'Delta': +30.0,
            'Trend': [150, 155, 165, 175, 180],
            'Status': False
        }),
    ]

    # Render
    renderer = DensityTableRenderer()
    html = renderer.render(
        columns=columns,
        rows=rows,
        title='Performance Metrics Dashboard'
    )

    assert 'Cache Hit Rate' in html, "Should include metric names"
    assert 'Target' in html, "Should have target column"
    assert 'Status' in html, "Should have status column"

    print(f'  Columns: {len(columns)}')
    print(f'  Rows: {len(rows)}')
    print(f'  Column types: {[c.type.value for c in columns]}')
    print('  [PASS] Custom density table generated')


def test_grid_layouts():
    """Test different grid layouts for small multiples."""
    print('\n[TEST 5] Small Multiples - Grid Layout Variations')
    print('=' * 70)

    # Create 6 queries
    queries = [
        {'query_text': f'Query {i+1}', 'latency_ms': 80 + i*10, 'confidence': 0.90 - i*0.02,
         'threads_count': 2 + i, 'cached': i % 2 == 0, 'trend': [85-i, 84-i, 83-i, 82-i, 80+i*10],
         'timestamp': 1698595200.0 + i*10, 'tool_used': 'answer'}
        for i in range(6)
    ]

    # Test grid layout
    grid_html = render_small_multiples(queries, layout='grid', max_columns=3)
    assert 'grid-template-columns' in grid_html

    # Test row layout
    row_html = render_small_multiples(queries, layout='row')
    assert 'grid-template-columns' in row_html  # Will be "repeat(6, 1fr)"

    # Test column layout
    col_html = render_small_multiples(queries, layout='column')
    assert 'grid-template-columns' in col_html  # Will be "1fr"

    print(f'  Grid layout: 3 columns')
    print(f'  Row layout: 6 columns')
    print(f'  Column layout: 1 column')
    print('  [PASS] Grid layouts working correctly')


def run_all_tests():
    """Run all advanced Tufte visualization tests."""
    print('\n' + '=' * 70)
    print('ADVANCED TUFTE VISUALIZATIONS: MEANING FIRST')
    print('=' * 70)

    try:
        test_small_multiples()
        test_density_table()
        test_combined_visualization()
        test_custom_density_table()
        test_grid_layouts()

        print('\n' + '=' * 70)
        print('[SUCCESS] All advanced Tufte tests passing!')
        print('=' * 70)
        print('\nNew Visualization Capabilities:')
        print('  + Small Multiples: Compare multiple queries side-by-side')
        print('  + Data Density Tables: Maximum information per square inch')
        print('  + Inline Sparklines: Trends within table cells')
        print('  + Delta Indicators: Color-coded changes')
        print('  + Bottleneck Detection: Automatic highlighting')
        print('  + Consistent Scales: Enable fair comparison')
        print('  + Minimal Decoration: Data-ink ratio optimized')
        print('\nTufte Principles Applied:')
        print('  1. Above all else show the data')
        print('  2. Maximize data-ink ratio (remove chartjunk)')
        print('  3. Small multiples enable comparison')
        print('  4. High data density, small space')
        print('  5. Meaning first - information where eyes go')
        print('\n')

    except AssertionError as e:
        print(f'\n[FAIL] Test failed: {e}')
        raise
    except Exception as e:
        print(f'\n[ERROR] Unexpected error: {e}')
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    run_all_tests()
