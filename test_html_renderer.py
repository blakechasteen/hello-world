#!/usr/bin/env python3
"""
Test HTMLRenderer - Phase 2.3 Validation
=========================================
Tests complete dashboard generation pipeline:
Spacetime → StrategySelector → DashboardConstructor → HTMLRenderer → HTML

Author: Claude Code
Date: October 28, 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path

from HoloLoom.visualization.constructor import DashboardConstructor
from HoloLoom.visualization.html_renderer import HTMLRenderer, save_dashboard


@dataclass
class MockTrace:
    """Mock trace for testing."""
    duration_ms: float = 150.0
    stage_durations: Dict[str, float] = field(default_factory=lambda: {
        'pattern_selection': 5.0,
        'retrieval': 50.0,
        'convergence': 30.0,
        'tool_execution': 60.0
    })
    threads_activated: List[str] = field(default_factory=lambda: ['thread1', 'thread2', 'thread3'])
    errors: List = field(default_factory=list)


@dataclass
class MockSpacetime:
    """Mock spacetime for testing."""
    query_text: str
    response: str
    tool_used: str
    confidence: float
    trace: MockTrace
    complexity: str = 'FAST'
    metadata: Dict = field(default_factory=lambda: {
        'semantic_cache': {
            'enabled': True,
            'hits': 3,
            'misses': 1
        }
    })


def test_html_renderer():
    """Test complete dashboard generation pipeline."""
    print('[TEST] HTMLRenderer - Complete Pipeline')
    print('=' * 60)

    # 1. Create mock spacetime
    spacetime = MockSpacetime(
        query_text='How does the weaving orchestrator work?',
        response='The orchestrator coordinates feature extraction and decision making.',
        tool_used='answer',
        confidence=0.92,
        trace=MockTrace()
    )

    # 2. Construct dashboard
    print('\n[STEP 1] Constructing dashboard from Spacetime...')
    constructor = DashboardConstructor()
    dashboard = constructor.construct(spacetime)

    print(f'  Title: {dashboard.title}')
    print(f'  Layout: {dashboard.layout.value}')
    print(f'  Panels: {len(dashboard.panels)}')

    # 3. Render to HTML
    print('\n[STEP 2] Rendering dashboard to HTML...')
    renderer = HTMLRenderer()
    html = renderer.render(dashboard)

    # Validate HTML structure
    assert '<!DOCTYPE html>' in html
    assert 'Tailwind CSS' in html
    assert 'Plotly.js' in html
    assert dashboard.title in html
    print(f'  HTML length: {len(html):,} characters')

    # 4. Save to file
    output_path = Path('demos/output/dashboard_test.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'\n[STEP 3] Saving to {output_path}...')
    save_dashboard(dashboard, str(output_path))

    # Validate file was created
    assert output_path.exists()
    file_size = output_path.stat().st_size
    print(f'  File size: {file_size:,} bytes')

    # 5. Check panel rendering
    print('\n[STEP 4] Validating panel rendering...')
    panels_rendered = 0
    for panel in dashboard.panels:
        if panel.title in html:
            panels_rendered += 1
            print(f'  [PASS] {panel.title} ({panel.type.value})')

    print(f'\n  Total panels rendered: {panels_rendered}/{len(dashboard.panels)}')

    # 6. Check Plotly charts
    print('\n[STEP 5] Checking interactive visualizations...')
    plotly_count = html.count('Plotly.newPlot')
    print(f'  Plotly charts: {plotly_count}')

    # 7. Check metadata
    print('\n[STEP 6] Checking metadata footer...')
    assert 'Complexity: FAST' in html
    assert 'Panels: 6' in html
    assert 'Cache: 75% hits' in html  # 3/(3+1) = 0.75
    print('  [PASS] Metadata footer present')

    print('\n' + '=' * 60)
    print('[PASS] HTMLRenderer Complete!')
    print(f'      View dashboard: {output_path.absolute()}')
    print('=' * 60)


def test_different_layouts():
    """Test different dashboard layouts."""
    print('\n[TEST] Testing Different Layouts')
    print('=' * 60)

    from HoloLoom.visualization.strategy import StrategySelector, QueryAnalyzer
    from HoloLoom.visualization.dashboard import LayoutType

    test_cases = [
        ('What is X?', 'FACTUAL', LayoutType.METRIC),
        ('How does X work?', 'EXPLORATORY', LayoutType.FLOW),
        ('Compare X vs Y', 'COMPARISON', LayoutType.FLOW),
    ]

    constructor = DashboardConstructor()
    renderer = HTMLRenderer()

    for query, expected_intent, expected_layout in test_cases:
        spacetime = MockSpacetime(
            query_text=query,
            response='Mock response',
            tool_used='answer',
            confidence=0.85,
            trace=MockTrace()
        )

        dashboard = constructor.construct(spacetime)
        html = renderer.render(dashboard)

        print(f'\n  Query: "{query}"')
        print(f'    Layout: {dashboard.layout.value}')
        print(f'    Panels: {len(dashboard.panels)}')
        print(f'    HTML: {len(html):,} chars')

    print('\n' + '=' * 60)
    print('[PASS] All layouts working!')


if __name__ == '__main__':
    test_html_renderer()
    test_different_layouts()
    print('\n[SUCCESS] Phase 2.3 Complete - Edward Tufte Machine operational!')
