#!/usr/bin/env python3
"""
Edward Tufte Machine - Complete Demo
=====================================
Demonstrates complete dashboard generation pipeline:

Spacetime → StrategySelector → DashboardConstructor → HTMLRenderer → Beautiful HTML

This shows how HoloLoom automatically generates beautiful, interactive
dashboards following Edward Tufte's data visualization principles.

Author: Claude Code with HoloLoom architecture
Date: October 28, 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path

from HoloLoom.visualization import (
    DashboardConstructor,
    save_dashboard
)


@dataclass
class MockTrace:
    """Mock trace for demo."""
    duration_ms: float = 287.5
    stage_durations: Dict[str, float] = field(default_factory=lambda: {
        'pattern_selection': 12.3,
        'feature_extraction': 45.2,
        'retrieval': 89.7,
        'convergence': 67.1,
        'tool_execution': 73.2
    })
    threads_activated: List[str] = field(default_factory=lambda: [
        'weaving_fundamentals',
        'hololoom_architecture',
        'matryoshka_embeddings',
        'semantic_cache',
        'edward_tufte_principles'
    ])
    errors: List = field(default_factory=list)


@dataclass
class MockSpacetime:
    """Mock spacetime for demo."""
    query_text: str
    response: str
    tool_used: str
    confidence: float
    trace: MockTrace
    complexity: str = 'FULL'
    metadata: Dict = field(default_factory=lambda: {
        'semantic_cache': {
            'enabled': True,
            'hits': 8,
            'misses': 2,
            'hot_tier': 3,
            'warm_tier': 5
        }
    })


def demo_exploratory_query():
    """Demo: Exploratory query generates mechanism panels."""
    print('\n' + '=' * 70)
    print('DEMO 1: Exploratory Query - "How does X work?"')
    print('=' * 70)

    spacetime = MockSpacetime(
        query_text='How does the semantic cache accelerate HoloLoom queries?',
        response='''The semantic cache provides 3-10× speedup through three-tier architecture:
        1. Hot tier: Pre-loaded common patterns (<0.001ms)
        2. Warm tier: LRU cache of recent queries (<0.001ms)
        3. Cold path: Full computation with caching (~18ms)''',
        tool_used='explain_mechanism',
        confidence=0.94,
        trace=MockTrace()
    )

    constructor = DashboardConstructor()
    dashboard = constructor.construct(spacetime)

    print(f'\nGenerated Dashboard:')
    print(f'  Title: {dashboard.title}')
    print(f'  Layout: {dashboard.layout.value}')
    print(f'  Panels: {len(dashboard.panels)}')
    print('\n  Panel Breakdown:')
    for i, panel in enumerate(dashboard.panels, 1):
        print(f'    {i}. {panel.title} ({panel.type.value}, size={panel.size.value})')

    # Save
    output_path = Path('demos/output/exploratory_dashboard.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dashboard(dashboard, str(output_path))
    print(f'\n  Saved: {output_path.absolute()}')


def demo_optimization_query():
    """Demo: Optimization query generates performance panels."""
    print('\n' + '=' * 70)
    print('DEMO 2: Optimization Query - "How to improve X?"')
    print('=' * 70)

    spacetime = MockSpacetime(
        query_text='How can I optimize HoloLoom query performance?',
        response='''To optimize performance:
        1. Enable semantic cache (3-10× speedup)
        2. Use FAST mode for standard queries (<150ms)
        3. Batch similar queries to improve cache hit rate
        4. Monitor bottleneck stages in execution timeline''',
        tool_used='optimize',
        confidence=0.89,
        trace=MockTrace(
            duration_ms=156.8,
            stage_durations={
                'pattern_selection': 8.2,
                'feature_extraction': 28.4,
                'retrieval': 92.3,  # Bottleneck!
                'convergence': 18.6,
                'tool_execution': 9.3
            }
        )
    )

    constructor = DashboardConstructor()
    dashboard = constructor.construct(spacetime)

    print(f'\nGenerated Dashboard:')
    print(f'  Title: {dashboard.title}')
    print(f'  Layout: {dashboard.layout.value}')
    print(f'  Panels: {len(dashboard.panels)}')
    print('\n  Panel Breakdown:')
    for i, panel in enumerate(dashboard.panels, 1):
        print(f'    {i}. {panel.title} ({panel.type.value}, size={panel.size.value})')
        if 'bottleneck' in panel.title.lower():
            print(f'       → Identifies: retrieval stage (92.3ms, 59%)')

    # Save
    output_path = Path('demos/output/optimization_dashboard.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dashboard(dashboard, str(output_path))
    print(f'\n  Saved: {output_path.absolute()}')


def demo_factual_query():
    """Demo: Factual query generates minimal panels."""
    print('\n' + '=' * 70)
    print('DEMO 3: Factual Query - "What is X?"')
    print('=' * 70)

    spacetime = MockSpacetime(
        query_text='What is Thompson Sampling?',
        response='''Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.
        It balances exploration and exploitation by sampling from posterior distributions.''',
        tool_used='define',
        confidence=0.97,
        trace=MockTrace(
            duration_ms=42.1,
            stage_durations={
                'pattern_selection': 3.1,
                'retrieval': 28.7,
                'execution': 10.3
            },
            threads_activated=['thompson_sampling_basics', 'bayesian_statistics']
        )
    )

    constructor = DashboardConstructor()
    dashboard = constructor.construct(spacetime)

    print(f'\nGenerated Dashboard:')
    print(f'  Title: {dashboard.title}')
    print(f'  Layout: {dashboard.layout.value} (simpler layout for factual queries)')
    print(f'  Panels: {len(dashboard.panels)} (fewer panels, just the essentials)')
    print('\n  Panel Breakdown:')
    for i, panel in enumerate(dashboard.panels, 1):
        print(f'    {i}. {panel.title} ({panel.type.value}, size={panel.size.value})')

    # Save
    output_path = Path('demos/output/factual_dashboard.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dashboard(dashboard, str(output_path))
    print(f'\n  Saved: {output_path.absolute()}')


def print_summary():
    """Print summary of Edward Tufte Machine capabilities."""
    print('\n' + '=' * 70)
    print('EDWARD TUFTE MACHINE - CAPABILITIES SUMMARY')
    print('=' * 70)
    print('''
Edward Tufte Principles Applied:
  ✓ Maximize data-ink ratio (minimal decoration)
  ✓ Show causality through timeline visualizations
  ✓ Enable micro/macro readings (detail on hover)
  ✓ Create narrative flow (panels tell a story)
  ✓ Visual integrity (semantic color coding)

Intent-Based Panel Selection:
  • FACTUAL queries → Minimal panels (definitions, metrics)
  • EXPLORATORY queries → Mechanism panels (timelines, graphs)
  • COMPARISON queries → Side-by-side visualizations
  • DEBUGGING queries → Error traces, bottleneck analysis
  • OPTIMIZATION queries → Performance metrics, bottlenecks

Technologies Used:
  • Tailwind CSS (responsive layouts, no build step)
  • Plotly.js (interactive charts)
  • Semantic color coding (green=good, yellow=ok, red=warning)

Output:
  • Standalone HTML files (no server required)
  • Responsive layouts (mobile + desktop)
  • Interactive visualizations (hover for details)
  • Beautiful typography (Edward Tufte-inspired)
    ''')
    print('=' * 70)
    print('\nView Dashboards:')
    print('  1. demos/output/exploratory_dashboard.html')
    print('  2. demos/output/optimization_dashboard.html')
    print('  3. demos/output/factual_dashboard.html')
    print('=' * 70)


if __name__ == '__main__':
    print('\nEdward Tufte Machine - Complete Demo')
    print('Automatic dashboard generation following data visualization best practices\n')

    demo_exploratory_query()
    demo_optimization_query()
    demo_factual_query()
    print_summary()

    print('\n✅ All demos complete!')
    print('   Phase 2 (Edward Tufte Machine) is fully operational.\n')
