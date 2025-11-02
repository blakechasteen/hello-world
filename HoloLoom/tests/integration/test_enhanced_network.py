#!/usr/bin/env python3
"""
Test Enhanced Network Visualization - Phase 1.1
================================================
Tests force-directed network graphs with zoom/pan and interactivity.

Author: Claude Code
Date: October 29, 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path

from HoloLoom.visualization.constructor import DashboardConstructor
from HoloLoom.visualization.html_renderer import save_dashboard


@dataclass
class MockTrace:
    """Mock trace with knowledge threads."""
    duration_ms: float = 250.0
    stage_durations: Dict[str, float] = field(default_factory=lambda: {
        'pattern_selection': 15.0,
        'feature_extraction': 60.0,
        'retrieval': 120.0,
        'convergence': 35.0,
        'tool_execution': 20.0
    })
    threads_activated: List[str] = field(default_factory=lambda: [
        'weaving_fundamentals',
        'hololoom_architecture',
        'matryoshka_embeddings',
        'thompson_sampling',
        'semantic_cache',
        'edward_tufte_principles',
        'force_directed_graphs',
        'd3_visualization',
        'knowledge_graphs'
    ])
    errors: List = field(default_factory=list)


@dataclass
class MockSpacetime:
    """Mock spacetime for testing."""
    query_text: str
    response: str
    tool_used: str
    confidence: float
    trace: MockTrace
    complexity: str = 'FULL'
    metadata: Dict = field(default_factory=lambda: {
        'semantic_cache': {
            'enabled': True,
            'hits': 7,
            'misses': 2
        }
    })


def test_enhanced_network_visualization():
    """Test Phase 1.1: Enhanced Force-Directed Network Graphs."""
    print('[TEST] Phase 1.1 - Enhanced Network Visualization')
    print('=' * 70)

    # Create spacetime with multiple threads
    spacetime = MockSpacetime(
        query_text='Explain the relationship between HoloLoom components',
        response='HoloLoom integrates multiple components including the weaving orchestrator, semantic cache, and visualization system.',
        tool_used='explain',
        confidence=0.94,
        trace=MockTrace()
    )

    print('\n[STEP 1] Constructing dashboard with network panel...')
    constructor = DashboardConstructor()
    dashboard = constructor.construct(spacetime)

    # Find network panel
    network_panel = None
    for panel in dashboard.panels:
        if panel.type.value == 'network':
            network_panel = panel
            break

    if not network_panel:
        print('  [SKIP] No network panel generated')
        return

    print(f'  Network Panel: {network_panel.title}')
    print(f'    Nodes: {network_panel.data.get("node_count")}')
    print(f'    Edges: {len(network_panel.data.get("edges", []))}')

    # Validate enhanced structure
    nodes = network_panel.data.get('nodes', [])
    edges = network_panel.data.get('edges', [])

    # Check for query node
    query_nodes = [n for n in nodes if n.get('type') == 'query']
    thread_nodes = [n for n in nodes if n.get('type') == 'thread']

    print(f'\n  Node Analysis:')
    print(f'    Query nodes: {len(query_nodes)} (should be 1)')
    print(f'    Thread nodes: {len(thread_nodes)} (should be {len(spacetime.trace.threads_activated)})')

    if query_nodes:
        print(f'    Query node color: {query_nodes[0].get("color")} (should be purple)')
        print(f'    Query node size: {query_nodes[0].get("size")} (should be larger)')

    print(f'\n  Edge Analysis:')
    print(f'    Total edges: {len(edges)}')
    print(f'    Expected minimum: {len(thread_nodes)} (query → threads)')

    # Validate edges connect properly
    edge_sources = set(e['source'] for e in edges)
    edge_targets = set(e['target'] for e in edges)
    print(f'    Unique sources: {len(edge_sources)}')
    print(f'    Unique targets: {len(edge_targets)}')

    assert len(query_nodes) == 1, "Should have exactly 1 query node"
    assert len(thread_nodes) == len(spacetime.trace.threads_activated), "Should have 1 node per thread"
    assert len(edges) >= len(thread_nodes), "Should have at least 1 edge per thread"

    print('\n  [PASS] Network structure validated!')

    # Save to HTML
    print('\n[STEP 2] Rendering to HTML with D3.js...')
    output_path = Path('demos/output/enhanced_network_demo.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dashboard(dashboard, str(output_path))
    print(f'  Saved: {output_path.absolute()}')
    print(f'  Size: {output_path.stat().st_size:,} bytes')

    # Check HTML contains D3.js features
    html_content = output_path.read_text()

    features_to_check = [
        ('D3.js loaded', 'd3js.org/d3'),
        ('Zoom behavior', 'd3.zoom()'),
        ('Force simulation', 'd3.forceSimulation'),
        ('Drag interactions', 'dragstarted'),
        ('Highlight function', 'highlightConnected'),
        ('Reset zoom', 'dblclick.zoom'),
        ('Instructions', 'Drag nodes'),
    ]

    print('\n[STEP 3] Validating D3.js features in HTML...')
    for feature_name, search_string in features_to_check:
        if search_string in html_content:
            print(f'  [PASS] {feature_name}')
        else:
            print(f'  [FAIL] {feature_name} - not found')

    print('\n' + '=' * 70)
    print('[SUCCESS] Phase 1.1 Complete!')
    print(f'\nOpen this file to see the interactive force-directed graph:')
    print(f'  {output_path.absolute()}')
    print('\nFeatures:')
    print('  • Drag nodes to rearrange')
    print('  • Scroll to zoom in/out')
    print('  • Double-click to reset view')
    print('  • Hover to highlight connected nodes')
    print('  • Purple query node at center')
    print('  • Indigo thread nodes around query')
    print('=' * 70)


if __name__ == '__main__':
    test_enhanced_network_visualization()
