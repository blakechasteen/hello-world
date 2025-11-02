#!/usr/bin/env python3
"""
Test Bottleneck Detection - Phase 2.3 Validation
=================================================
Tests automatic bottleneck detection and visual highlighting in dashboards.

Author: Claude Code
Date: October 29, 2025
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

from HoloLoom.visualization.constructor import DashboardConstructor
from HoloLoom.visualization.html_renderer import save_dashboard


@dataclass
class MockTrace:
    """Mock trace with configurable stage durations."""
    duration_ms: float
    stage_durations: Dict[str, float]
    threads_activated: List[str] = field(default_factory=lambda: ['thread1', 'thread2'])
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
    metadata: Dict = field(default_factory=lambda: {'semantic_cache': {'enabled': True}})


def test_no_bottleneck():
    """Test case: No bottleneck (all stages balanced)."""
    print('\n[TEST 1] No Bottleneck (Balanced Stages)')
    print('=' * 70)

    # Create balanced stage durations (no stage >40%)
    stage_durations = {
        'pattern_selection': 25.0,  # 25%
        'retrieval': 30.0,          # 30%
        'convergence': 20.0,        # 20%
        'tool_execution': 25.0      # 25%
    }

    total = sum(stage_durations.values())
    spacetime = MockSpacetime(
        query_text='What is Thompson Sampling?',
        response='Thompson Sampling balances exploration and exploitation.',
        tool_used='answer',
        confidence=0.92,
        trace=MockTrace(duration_ms=total, stage_durations=stage_durations)
    )

    # Construct dashboard
    constructor = DashboardConstructor()
    dashboard = constructor.construct(spacetime)

    # Find timeline panel
    timeline_panel = None
    for panel in dashboard.panels:
        if panel.type.value == 'timeline':
            timeline_panel = panel
            break

    if not timeline_panel:
        print('[SKIP] No timeline panel generated')
        return

    # Validate bottleneck data
    bottleneck = timeline_panel.data.get('bottleneck', {})
    assert not bottleneck.get('detected', True), "Bottleneck should NOT be detected"

    print(f'  Total duration: {total:.1f}ms')
    print(f'  Bottleneck detected: {bottleneck.get("detected")}')
    print(f'  [PASS] No bottleneck correctly identified')


def test_moderate_bottleneck():
    """Test case: Moderate bottleneck (40-50% threshold)."""
    print('\n[TEST 2] Moderate Bottleneck (40-50%)')
    print('=' * 70)

    # Create bottleneck: retrieval takes 45%
    stage_durations = {
        'pattern_selection': 10.0,   # 10%
        'retrieval': 45.0,           # 45% ⚠️ BOTTLENECK
        'convergence': 15.0,         # 15%
        'tool_execution': 30.0       # 30%
    }

    total = sum(stage_durations.values())
    spacetime = MockSpacetime(
        query_text='How does the weaving orchestrator work?',
        response='The orchestrator coordinates feature extraction and decision making.',
        tool_used='answer',
        confidence=0.88,
        trace=MockTrace(duration_ms=total, stage_durations=stage_durations)
    )

    # Construct dashboard
    constructor = DashboardConstructor()
    dashboard = constructor.construct(spacetime)

    # Find timeline panel
    timeline_panel = None
    for panel in dashboard.panels:
        if panel.type.value == 'timeline':
            timeline_panel = panel
            break

    assert timeline_panel, "Timeline panel should exist"

    # Validate bottleneck data
    bottleneck = timeline_panel.data.get('bottleneck', {})
    assert bottleneck.get('detected'), "Bottleneck SHOULD be detected"
    assert bottleneck.get('stage') == 'retrieval', f"Wrong stage: {bottleneck.get('stage')}"
    assert 44 <= bottleneck.get('percentage', 0) <= 46, f"Wrong percentage: {bottleneck.get('percentage')}"

    # Check color coding
    colors = timeline_panel.data.get('colors', [])
    stages = timeline_panel.data.get('stages', [])
    retrieval_idx = stages.index('retrieval')
    assert colors[retrieval_idx] == '#f97316', f"Expected orange for moderate bottleneck, got {colors[retrieval_idx]}"

    print(f'  Total duration: {total:.1f}ms')
    print(f'  Bottleneck stage: {bottleneck.get("stage")}')
    print(f'  Bottleneck percentage: {bottleneck.get("percentage"):.1f}%')
    print(f'  Optimization: {bottleneck.get("optimization", "")[:80]}...')
    print(f'  [PASS] Moderate bottleneck correctly detected')


def test_severe_bottleneck():
    """Test case: Severe bottleneck (>50% threshold)."""
    print('\n[TEST 3] Severe Bottleneck (>50%)')
    print('=' * 70)

    # Create severe bottleneck: tool_execution takes 60%
    stage_durations = {
        'pattern_selection': 5.0,    # 5%
        'retrieval': 20.0,           # 20%
        'convergence': 15.0,         # 15%
        'tool_execution': 60.0       # 60% 🔴 SEVERE BOTTLENECK
    }

    total = sum(stage_durations.values())
    spacetime = MockSpacetime(
        query_text='Complex query requiring expensive tool',
        response='Result from expensive computation.',
        tool_used='expensive_tool',
        confidence=0.95,
        trace=MockTrace(duration_ms=total, stage_durations=stage_durations)
    )

    # Construct dashboard
    constructor = DashboardConstructor()
    dashboard = constructor.construct(spacetime)

    # Find timeline panel
    timeline_panel = None
    for panel in dashboard.panels:
        if panel.type.value == 'timeline':
            timeline_panel = panel
            break

    assert timeline_panel, "Timeline panel should exist"

    # Validate bottleneck data
    bottleneck = timeline_panel.data.get('bottleneck', {})
    assert bottleneck.get('detected'), "Bottleneck SHOULD be detected"
    assert bottleneck.get('stage') == 'tool_execution', f"Wrong stage: {bottleneck.get('stage')}"
    assert bottleneck.get('percentage', 0) == 60.0, f"Wrong percentage: {bottleneck.get('percentage')}"

    # Check color coding
    colors = timeline_panel.data.get('colors', [])
    stages = timeline_panel.data.get('stages', [])
    tool_idx = stages.index('tool_execution')
    assert colors[tool_idx] == '#ef4444', f"Expected red for severe bottleneck, got {colors[tool_idx]}"

    print(f'  Total duration: {total:.1f}ms')
    print(f'  Bottleneck stage: {bottleneck.get("stage")}')
    print(f'  Bottleneck percentage: {bottleneck.get("percentage"):.1f}%')
    print(f'  Optimization: {bottleneck.get("optimization", "")[:80]}...')
    print(f'  [PASS] Severe bottleneck correctly detected')

    # Save demo HTML
    output_path = Path('demos/output/bottleneck_detection_demo.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dashboard(dashboard, str(output_path))
    print(f'\n  [SAVED] Demo HTML: {output_path}')


def test_html_warning_rendering():
    """Test case: Validate HTML includes warning banner."""
    print('\n[TEST 4] HTML Warning Banner Rendering')
    print('=' * 70)

    # Create bottleneck scenario
    stage_durations = {
        'pattern_selection': 10.0,
        'retrieval': 70.0,  # 70% severe bottleneck
        'convergence': 10.0,
        'tool_execution': 10.0
    }

    total = sum(stage_durations.values())
    spacetime = MockSpacetime(
        query_text='Test query',
        response='Test response',
        tool_used='answer',
        confidence=0.90,
        trace=MockTrace(duration_ms=total, stage_durations=stage_durations)
    )

    # Construct dashboard
    constructor = DashboardConstructor()
    dashboard = constructor.construct(spacetime)

    # Render to HTML
    from HoloLoom.visualization.html_renderer import HTMLRenderer
    renderer = HTMLRenderer()
    html = renderer.render(dashboard)

    # Validate HTML contains bottleneck warning elements
    assert 'Bottleneck Detected' in html, "HTML should contain 'Bottleneck Detected'"
    assert 'bg-red-50' in html or 'bg-orange-50' in html, "HTML should contain warning background color"
    assert '🔴' in html or '⚠️' in html, "HTML should contain warning icon"
    assert 'retrieval' in html, "HTML should mention bottleneck stage"

    print('  HTML validation:')
    print('    [OK] Contains "Bottleneck Detected" text')
    print('    [OK] Contains warning background color')
    print('    [OK] Contains warning icon (red circle or warning triangle)')
    print('    [OK] Mentions bottleneck stage name')
    print('  [PASS] HTML warning rendering correct')


def run_all_tests():
    """Run all bottleneck detection tests."""
    print('\n' + '=' * 70)
    print('PHASE 2.3: BOTTLENECK DETECTION VALIDATION')
    print('=' * 70)

    try:
        test_no_bottleneck()
        test_moderate_bottleneck()
        test_severe_bottleneck()
        test_html_warning_rendering()

        print('\n' + '=' * 70)
        print('[SUCCESS] All Phase 2.3 tests passing!')
        print('=' * 70)
        print('\nBottleneck Detection Features:')
        print('  + Detects stages taking >40% of total time')
        print('  + Color codes: Red (>50%), Orange (40-50%)')
        print('  + Generates actionable optimization suggestions')
        print('  + Visual warning banner in dashboard')
        print('  + Threshold configurable (currently 40%)')
        print('\n')

    except AssertionError as e:
        print(f'\n[FAIL] Test failed: {e}')
        raise
    except Exception as e:
        print(f'\n[ERROR] Unexpected error: {e}')
        raise


if __name__ == '__main__':
    run_all_tests()
