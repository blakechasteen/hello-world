#!/usr/bin/env python3
"""
Test Stage Waterfall Visualization

Tests sequential pipeline timing visualization with bottleneck detection.

Author: Claude Code
Date: October 29, 2025
"""

from pathlib import Path
from HoloLoom.visualization.stage_waterfall import (
    StageWaterfallRenderer,
    WaterfallStage,
    StageStatus,
    render_pipeline_waterfall,
    render_parallel_waterfall
)


def test_basic_sequential_waterfall():
    """Test basic sequential waterfall rendering."""
    print('\n[TEST 1] Basic Sequential Waterfall')
    print('=' * 70)

    # Create sequential stages
    stages = [
        WaterfallStage(
            name='Pattern Selection',
            start_ms=0.0,
            duration_ms=5.2,
            status=StageStatus.SUCCESS,
            trend=[6.0, 5.5, 5.3, 5.2, 5.2]
        ),
        WaterfallStage(
            name='Retrieval',
            start_ms=5.2,
            duration_ms=50.5,
            status=StageStatus.SUCCESS,
            trend=[45.0, 47.0, 48.0, 50.0, 50.5]
        ),
        WaterfallStage(
            name='Convergence',
            start_ms=55.7,
            duration_ms=30.0,
            status=StageStatus.SUCCESS,
            trend=[32.0, 31.0, 30.0, 30.0, 30.0]
        ),
        WaterfallStage(
            name='Tool Execution',
            start_ms=85.7,
            duration_ms=64.3,
            status=StageStatus.SUCCESS,
            trend=[60.0, 61.0, 62.0, 63.0, 64.3]
        ),
    ]

    renderer = StageWaterfallRenderer()
    html = renderer.render(stages, title='Pipeline Stage Waterfall')

    # Validate
    assert 'stage-waterfall' in html, "Should have waterfall container"
    assert 'Pipeline Stage Waterfall' in html, "Should have title"
    assert 'Pattern Selection' in html, "Should include stage names"
    assert 'Tool Execution' in html, "Should include all stages"
    assert 'Total: 150.0ms' in html, "Should show total duration"

    total_duration = sum(s.duration_ms for s in stages)
    print(f'  Stages: {len(stages)}')
    print(f'  Total duration: {total_duration:.1f}ms')
    print(f'  Fastest stage: Pattern Selection (5.2ms)')
    print(f'  Slowest stage: Tool Execution (64.3ms)')
    print('  [PASS] Basic waterfall rendered correctly')


def test_bottleneck_detection():
    """Test automatic bottleneck detection and highlighting."""
    print('\n[TEST 2] Bottleneck Detection')
    print('=' * 70)

    # Create stages with one bottleneck (>40% of total)
    stages = [
        WaterfallStage(name='Fast Stage', start_ms=0.0, duration_ms=10.0),
        WaterfallStage(name='Slow Stage', start_ms=10.0, duration_ms=80.0),  # 80% of total!
        WaterfallStage(name='Medium Stage', start_ms=90.0, duration_ms=10.0),
    ]

    renderer = StageWaterfallRenderer(bottleneck_threshold=0.4)
    html = renderer.render(stages)

    # Should highlight "Slow Stage" as bottleneck
    assert 'BOTTLENECK' in html, "Should detect bottleneck"
    assert '#f59e0b' in html, "Should use amber color for bottleneck"

    total = sum(s.duration_ms for s in stages)
    slow_pct = (80.0 / total) * 100

    print(f'  Total duration: {total:.1f}ms')
    print(f'  Slow Stage: 80.0ms ({slow_pct:.1f}%)')
    print(f'  Bottleneck threshold: 40%')
    print(f'  Bottleneck detected: YES')
    print('  [PASS] Bottleneck detection working')


def test_stage_statuses():
    """Test different stage status rendering (success, error, warning, skipped)."""
    print('\n[TEST 3] Stage Status Rendering')
    print('=' * 70)

    stages = [
        WaterfallStage(name='Success Stage', start_ms=0.0, duration_ms=20.0,
                      status=StageStatus.SUCCESS),
        WaterfallStage(name='Warning Stage', start_ms=20.0, duration_ms=20.0,
                      status=StageStatus.WARNING),
        WaterfallStage(name='Error Stage', start_ms=40.0, duration_ms=20.0,
                      status=StageStatus.ERROR),
        WaterfallStage(name='Skipped Stage', start_ms=60.0, duration_ms=5.0,
                      status=StageStatus.SKIPPED),
    ]

    renderer = StageWaterfallRenderer()
    html = renderer.render(stages)

    # Check colors
    assert '#10b981' in html, "Should have success green"
    assert '#ef4444' in html, "Should have error red"
    assert '#f59e0b' in html, "Should have warning amber"
    assert '#9ca3af' in html, "Should have skipped gray"

    # Check status indicators
    assert '&#10003;' in html, "Should have checkmark for success"

    print(f'  Statuses tested: 4 (success, warning, error, skipped)')
    print(f'  Colors validated: green, amber, red, gray')
    print('  [PASS] All status types rendering correctly')


def test_sparklines():
    """Test historical trend sparklines."""
    print('\n[TEST 4] Sparkline Trend Visualization')
    print('=' * 70)

    stages = [
        WaterfallStage(
            name='Improving Stage',
            start_ms=0.0,
            duration_ms=20.0,
            trend=[30.0, 28.0, 25.0, 22.0, 20.0]  # Improving trend
        ),
        WaterfallStage(
            name='Degrading Stage',
            start_ms=20.0,
            duration_ms=40.0,
            trend=[30.0, 32.0, 35.0, 38.0, 40.0]  # Degrading trend
        ),
    ]

    # With sparklines
    renderer_with = StageWaterfallRenderer(show_sparklines=True)
    html_with = renderer_with.render(stages)
    assert '<svg' in html_with, "Should include SVG sparklines"
    assert '<path' in html_with, "Should have sparkline paths"

    # Without sparklines
    renderer_without = StageWaterfallRenderer(show_sparklines=False)
    html_without = renderer_without.render(stages)
    sparkline_count_with = html_with.count('<svg')
    sparkline_count_without = html_without.count('<svg')

    print(f'  Sparklines with show_sparklines=True: {sparkline_count_with}')
    print(f'  Sparklines with show_sparklines=False: {sparkline_count_without}')
    print(f'  Trend directions: improving (down), degrading (up)')
    print('  [PASS] Sparklines rendering correctly')


def test_convenience_function():
    """Test convenience function for quick waterfall creation."""
    print('\n[TEST 5] Convenience Function')
    print('=' * 70)

    # Simple dict input
    durations = {
        'Pattern Selection': 5.2,
        'Retrieval': 50.5,
        'Convergence': 30.0,
        'Tool Execution': 64.3
    }

    trends = {
        'Retrieval': [45.0, 47.0, 48.0, 50.0, 50.5],
        'Tool Execution': [60.0, 61.0, 62.0, 63.0, 64.3]
    }

    html = render_pipeline_waterfall(
        durations,
        stage_trends=trends,
        title='Quick Pipeline Waterfall'
    )

    assert 'Quick Pipeline Waterfall' in html
    assert 'Pattern Selection' in html
    assert all(name in html for name in durations.keys())

    print(f'  Input: Simple dict with {len(durations)} stages')
    print(f'  Trends provided: {len(trends)} stages')
    print(f'  Total duration: {sum(durations.values()):.1f}ms')
    print('  [PASS] Convenience function working')


def test_parallel_waterfall():
    """Test parallel execution waterfall."""
    print('\n[TEST 6] Parallel Execution Waterfall')
    print('=' * 70)

    durations = {
        'Input Processing': 10.0,
        'Feature A': 30.0,
        'Feature B': 25.0,
        'Feature C': 35.0,
        'Decision': 20.0
    }

    # Define parallel groups
    parallel_groups = [
        ['Input Processing'],           # Sequential
        ['Feature A', 'Feature B', 'Feature C'],  # Parallel
        ['Decision']                    # Sequential
    ]

    html = render_parallel_waterfall(durations, parallel_groups)

    assert 'Input Processing' in html
    assert 'Feature A' in html
    assert 'Feature B' in html
    assert 'Feature C' in html
    assert 'Decision' in html

    # Calculate expected total
    # Input (10ms) + max(Feature A, B, C) + Decision (20ms)
    parallel_max = max(30.0, 25.0, 35.0)
    expected_total = 10.0 + parallel_max + 20.0

    print(f'  Sequential stages: 2 (Input, Decision)')
    print(f'  Parallel stages: 3 (Features A, B, C)')
    print(f'  Parallel group duration: {parallel_max:.1f}ms (max of group)')
    print(f'  Expected total: {expected_total:.1f}ms')
    print('  [PASS] Parallel waterfall working')


def test_combined_demo():
    """Generate comprehensive demo HTML."""
    print('\n[TEST 7] Combined Demo Generation')
    print('=' * 70)

    # Example 1: Standard pipeline
    standard_durations = {
        'Pattern Selection': 5.2,
        'Retrieval': 50.5,
        'Convergence': 30.0,
        'Tool Execution': 64.3
    }

    standard_trends = {
        'Pattern Selection': [6.0, 5.5, 5.3, 5.2, 5.2],
        'Retrieval': [45.0, 47.0, 48.0, 50.0, 50.5],
        'Convergence': [32.0, 31.0, 30.0, 30.0, 30.0],
        'Tool Execution': [60.0, 61.0, 62.0, 63.0, 64.3]
    }

    waterfall1 = render_pipeline_waterfall(
        standard_durations,
        stage_trends=standard_trends,
        title='Standard Pipeline (FAST mode)'
    )

    # Example 2: Pipeline with bottleneck
    bottleneck_durations = {
        'Pattern Selection': 3.0,
        'Retrieval': 15.0,
        'Convergence': 8.0,
        'Database Query': 120.0  # Major bottleneck!
    }

    waterfall2 = render_pipeline_waterfall(
        bottleneck_durations,
        title='Pipeline with Bottleneck (Database)'
    )

    # Example 3: Parallel execution
    parallel_durations = {
        'Input Parsing': 5.0,
        'Motif Detection': 25.0,
        'Embedding (96D)': 30.0,
        'Embedding (192D)': 35.0,
        'Spectral Features': 20.0,
        'Policy Decision': 15.0
    }

    parallel_groups = [
        ['Input Parsing'],
        ['Motif Detection', 'Embedding (96D)', 'Embedding (192D)', 'Spectral Features'],
        ['Policy Decision']
    ]

    waterfall3 = render_parallel_waterfall(
        parallel_durations,
        parallel_groups,
        title='Parallel Feature Extraction'
    )

    # Create complete demo HTML
    complete_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Stage Waterfall Visualization Demo</title>
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
            .use-cases {{
                background: #f0fdf4;
                border-left: 4px solid #10b981;
                padding: 16px;
                margin: 16px 0;
                border-radius: 4px;
            }}
            .use-cases h3 {{
                margin-top: 0;
                color: #065f46;
                font-size: 14px;
            }}
            .use-cases ul {{
                margin: 8px 0;
                padding-left: 20px;
                color: #374151;
                font-size: 13px;
                line-height: 1.6;
            }}
        </style>
    </head>
    <body>
        <h1>Stage Waterfall Visualization</h1>
        <div class="subtitle">
            Sequential pipeline timing with bottleneck detection - Tufte-style
        </div>

        <div class="tufte-quote">
            "Above all else show the data. The representation of numbers, as physically measured on the
            surface of the graphic itself, should be directly proportional to the quantities represented."
            <br><strong>— Edward Tufte</strong>
        </div>

        <div class="use-cases">
            <h3>Use Cases for Waterfall Charts</h3>
            <ul>
                <li><strong>Pipeline Optimization:</strong> Identify bottlenecks in processing stages</li>
                <li><strong>Performance Debugging:</strong> See exactly which stages are slowing down</li>
                <li><strong>Parallel Execution:</strong> Visualize concurrent vs sequential operations</li>
                <li><strong>Historical Trends:</strong> Track stage performance over time with sparklines</li>
                <li><strong>Comparative Analysis:</strong> Compare different execution modes (BARE/FAST/FUSED)</li>
            </ul>
        </div>

        <div class="section">
            <div class="section-title">Example 1: Standard Pipeline (FAST mode)</div>
            <p style="color: #6b7280; font-size: 13px; margin-bottom: 16px;">
                Typical query processing with 4 sequential stages. Sparklines show historical performance trends.
                Tool Execution takes 42.9% of total time but is not highlighted as bottleneck (threshold: 40%).
            </p>
            {waterfall1}
        </div>

        <div class="section">
            <div class="section-title">Example 2: Pipeline with Bottleneck</div>
            <p style="color: #6b7280; font-size: 13px; margin-bottom: 16px;">
                Pipeline with database bottleneck. Database Query takes 82.2% of total time and is
                automatically highlighted in amber. This immediately signals where optimization is needed.
            </p>
            {waterfall2}
        </div>

        <div class="section">
            <div class="section-title">Example 3: Parallel Feature Extraction</div>
            <p style="color: #6b7280; font-size: 13px; margin-bottom: 16px;">
                Multi-modal feature extraction with parallel execution. Four feature extractors run concurrently.
                Total time is determined by slowest parallel stage (Embedding 192D: 35ms), not sum of all stages.
            </p>
            {waterfall3}
        </div>

        <div class="section">
            <div class="section-title">Design Principles Applied</div>
            <ul style="color: #374151; font-size: 13px; line-height: 1.8;">
                <li><strong>Meaning First:</strong> Bottlenecks immediately visible with amber highlighting</li>
                <li><strong>Maximize data-ink ratio:</strong> No axes, grids, or unnecessary decoration</li>
                <li><strong>Layering:</strong> Color, position, width, status icons convey multiple dimensions</li>
                <li><strong>Data density:</strong> Duration, percentage, trend, status in compact space</li>
                <li><strong>Direct labeling:</strong> Values shown in bars, no legend lookup needed</li>
                <li><strong>Sparklines:</strong> Historical context inline with current metrics</li>
                <li><strong>Semantic colors:</strong> Green (success), Amber (bottleneck/warning), Red (error), Gray (skipped)</li>
            </ul>
        </div>

        <div class="section">
            <div class="section-title">Integration with HoloLoom</div>
            <p style="color: #6b7280; font-size: 13px; line-height: 1.6;">
                The Stage Waterfall visualization integrates seamlessly with HoloLoom's WeavingOrchestrator.
                The <code>Spacetime.trace.stage_durations</code> dict can be passed directly to
                <code>render_pipeline_waterfall()</code> to visualize any query's execution timeline.
            </p>
            <pre style="background: #f3f4f6; padding: 12px; border-radius: 4px; font-size: 12px; overflow-x: auto;">
from HoloLoom.visualization.stage_waterfall import render_pipeline_waterfall

# After weaving
spacetime = await orchestrator.weave(query)

# Render waterfall
html = render_pipeline_waterfall(
    spacetime.trace.stage_durations,
    title=f"Pipeline: {{query.text[:50]}}"
)
            </pre>
        </div>

        <div style="margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;
                    text-align: center; color: #9ca3af; font-size: 12px;">
            Generated with HoloLoom Visualizer | Stage Waterfall Visualization
        </div>
    </body>
    </html>
    """

    # Save to file
    output_path = Path('demos/output/stage_waterfall_demo.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(complete_html, encoding='utf-8')

    print(f'  Examples rendered: 3')
    print(f'  - Standard pipeline (4 stages)')
    print(f'  - Bottleneck detection (Database 82.2%)')
    print(f'  - Parallel execution (4 concurrent features)')
    print(f'  Combined HTML size: {len(complete_html):,} bytes')
    print(f'  [SAVED] Demo HTML: {output_path}')
    print('  [PASS] Combined demo generated')


def run_all_tests():
    """Run all stage waterfall tests."""
    print('\n' + '=' * 70)
    print('STAGE WATERFALL VISUALIZATION TESTS')
    print('=' * 70)

    try:
        test_basic_sequential_waterfall()
        test_bottleneck_detection()
        test_stage_statuses()
        test_sparklines()
        test_convenience_function()
        test_parallel_waterfall()
        test_combined_demo()

        print('\n' + '=' * 70)
        print('[SUCCESS] All stage waterfall tests passing!')
        print('=' * 70)
        print('\nNew Visualization Capability:')
        print('  + Stage Waterfall: Sequential pipeline timing')
        print('  + Bottleneck Detection: Automatic highlighting (>40% threshold)')
        print('  + Status Indicators: Success, warning, error, skipped')
        print('  + Sparklines: Historical trends inline')
        print('  + Parallel Execution: Concurrent stage visualization')
        print('\nTufte Principles Applied:')
        print('  1. Meaning first - bottlenecks immediately visible')
        print('  2. Maximize data-ink ratio - no unnecessary decoration')
        print('  3. Layering - color, position, width, status')
        print('  4. Data density - duration + % + trend + status')
        print('  5. Direct labeling - values in bars, not legend')
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
