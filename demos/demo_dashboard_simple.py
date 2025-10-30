#!/usr/bin/env python3
"""
Simple Dashboard Demo - No Heavy Dependencies
==============================================
Demonstrates dashboard features without full orchestrator overhead.

This demo creates dashboards directly from mock data to showcase:
- Interactive panels
- Network graphs
- User preferences
- localStorage persistence
- Drill-down features

Author: Claude Code
Date: October 29, 2025
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any

from HoloLoom.visualization import (
    Dashboard, Panel, PanelType, PanelSize, LayoutType,
    HTMLRenderer, DashboardConstructor, UserPreferences
)


# ============================================================================
# Mock Spacetime (lightweight, no orchestrator needed)
# ============================================================================

@dataclass
class MockTrace:
    start_time: datetime
    end_time: datetime
    duration_ms: float
    stage_durations: Dict[str, float] = field(default_factory=dict)
    threads_activated: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class MockSpacetime:
    query_text: str
    response: str
    tool_used: str
    confidence: float
    trace: MockTrace
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Demo 1: Basic Dashboard with Interactivity
# ============================================================================

def demo_basic_dashboard():
    """Create a basic dashboard showing core features."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Interactive Dashboard")
    print("="*80)

    # Create panels
    panels = [
        Panel(
            id="metric_1",
            type=PanelType.METRIC,
            title="Confidence Score",
            data={'value': 0.94, 'formatted': '94%', 'color': 'green'},
            size=PanelSize.SMALL
        ),
        Panel(
            id="metric_2",
            type=PanelType.METRIC,
            title="Duration",
            data={'value': 234.5, 'formatted': '234.5ms', 'color': 'blue'},
            size=PanelSize.SMALL
        ),
        Panel(
            id="text_1",
            type=PanelType.TEXT,
            title="Response",
            subtitle="Click to expand for full details",
            data={'content': """Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.

Key features:
- Balances exploration and exploitation naturally
- Uses posterior distributions for arm selection
- Achieves optimal regret bounds
- Widely used in reinforcement learning

HoloLoom uses Thompson Sampling in the UnifiedPolicy for tool selection."""},
            size=PanelSize.LARGE
        ),
    ]

    # Create mock spacetime
    spacetime = MockSpacetime(
        query_text="What is Thompson Sampling?",
        response="Thompson Sampling explanation...",
        tool_used="answer",
        confidence=0.94,
        trace=MockTrace(datetime.now(), datetime.now(), 234.5),
        metadata={'complexity': 'FAST'}
    )

    # Create dashboard
    dashboard = Dashboard(
        title="Query Response: What is Thompson Sampling?",
        layout=LayoutType.METRIC,
        panels=panels,
        spacetime=spacetime,
        metadata={'generated_at': datetime.now().isoformat()}
    )

    # Render to HTML
    renderer = HTMLRenderer()
    html = renderer.render(dashboard)

    # Save
    output_path = Path("dashboards/demo_basic.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')

    print(f"\n[+] Dashboard created: {len(panels)} panels")
    print(f"[+] File saved: {output_path}")
    print(f"[+] File size: {len(html):,} bytes")
    print("\nFeatures to try:")
    print("  - Click the expand button (down-arrow) on text panel")
    print("  - Click the settings button (bottom-right)")
    print("  - Try toggling dark mode in preferences")
    print("  - Reload page - settings persist via localStorage!")


# ============================================================================
# Demo 2: Network Graph Visualization
# ============================================================================

def demo_network_graph():
    """Create dashboard with interactive D3.js network graph."""
    print("\n" + "="*80)
    print("DEMO 2: Interactive Network Graph (D3.js)")
    print("="*80)

    # Create network data
    network_data = {
        'nodes': [
            {'id': 'query', 'label': 'Query', 'size': 20, 'color': '#8b5cf6'},
            {'id': 'motif', 'label': 'Motif Detector', 'size': 15, 'color': '#6366f1'},
            {'id': 'embedding', 'label': 'Embeddings', 'size': 18, 'color': '#10b981'},
            {'id': 'spectral', 'label': 'Spectral Features', 'size': 12, 'color': '#f59e0b'},
            {'id': 'graph', 'label': 'Knowledge Graph', 'size': 16, 'color': '#ef4444'},
            {'id': 'policy', 'label': 'Policy Engine', 'size': 20, 'color': '#ec4899'},
            {'id': 'response', 'label': 'Response', 'size': 18, 'color': '#14b8a6'},
        ],
        'edges': [
            {'source': 'query', 'target': 'motif'},
            {'source': 'query', 'target': 'embedding'},
            {'source': 'motif', 'target': 'policy'},
            {'source': 'embedding', 'target': 'policy'},
            {'source': 'embedding', 'target': 'spectral'},
            {'source': 'graph', 'target': 'embedding'},
            {'source': 'spectral', 'target': 'policy'},
            {'source': 'policy', 'target': 'response'},
        ],
        'node_count': 7
    }

    panels = [
        Panel(
            id="network_1",
            type=PanelType.NETWORK,
            title="HoloLoom Processing Pipeline",
            subtitle="Drag nodes to rearrange - Double-click for details",
            data=network_data,
            size=PanelSize.FULL_WIDTH
        ),
        Panel(
            id="metric_threads",
            type=PanelType.METRIC,
            title="Active Threads",
            data={'value': 7, 'formatted': '7', 'color': 'purple'},
            size=PanelSize.SMALL
        ),
    ]

    spacetime = MockSpacetime(
        query_text="Show processing pipeline",
        response="Pipeline visualization",
        tool_used="visualize",
        confidence=0.98,
        trace=MockTrace(
            datetime.now(),
            datetime.now(),
            156.2,
            threads_activated=['query', 'motif', 'embedding', 'spectral', 'graph', 'policy', 'response']
        )
    )

    dashboard = Dashboard(
        title="HoloLoom Processing Pipeline Visualization",
        layout=LayoutType.FLOW,
        panels=panels,
        spacetime=spacetime
    )

    renderer = HTMLRenderer()
    html = renderer.render(dashboard)

    output_path = Path("dashboards/demo_network.html")
    output_path.write_text(html, encoding='utf-8')

    print(f"\n[+] Network graph created: {network_data['node_count']} nodes, {len(network_data['edges'])} edges")
    print(f"[+] File saved: {output_path}")
    print("\nFeatures to try:")
    print("  - Drag nodes to rearrange the graph")
    print("  - Watch the physics simulation settle")
    print("  - Hover over nodes for tooltips")
    print("  - Click panel for drill-down view")


# ============================================================================
# Demo 3: Timeline with Stage Analysis
# ============================================================================

def demo_timeline():
    """Create dashboard with interactive timeline chart."""
    print("\n" + "="*80)
    print("DEMO 3: Execution Timeline Analysis")
    print("="*80)

    stage_durations = {
        'features': 123.4,
        'retrieval': 234.7,
        'decision': 87.3,
        'execution': 45.2
    }

    panels = [
        Panel(
            id="timeline_1",
            type=PanelType.TIMELINE,
            title="Execution Timeline",
            subtitle="Stage-by-stage breakdown",
            data={'stages': stage_durations},
            size=PanelSize.FULL_WIDTH
        ),
        Panel(
            id="metric_total",
            type=PanelType.METRIC,
            title="Total Duration",
            data={'value': sum(stage_durations.values()), 'formatted': f"{sum(stage_durations.values()):.1f}ms", 'color': 'blue'},
            size=PanelSize.SMALL
        ),
        Panel(
            id="metric_bottleneck",
            type=PanelType.METRIC,
            title="Slowest Stage",
            data={'value': max(stage_durations.values()), 'formatted': 'retrieval (234.7ms)', 'color': 'orange'},
            size=PanelSize.MEDIUM
        ),
    ]

    spacetime = MockSpacetime(
        query_text="Analyze execution performance",
        response="Timeline analysis complete",
        tool_used="analyze",
        confidence=0.91,
        trace=MockTrace(
            datetime.now(),
            datetime.now(),
            sum(stage_durations.values()),
            stage_durations=stage_durations
        )
    )

    dashboard = Dashboard(
        title="Execution Timeline Analysis",
        layout=LayoutType.FLOW,
        panels=panels,
        spacetime=spacetime
    )

    renderer = HTMLRenderer()
    html = renderer.render(dashboard)

    output_path = Path("dashboards/demo_timeline.html")
    output_path.write_text(html, encoding='utf-8')

    print(f"\n[+] Timeline created: {len(stage_durations)} stages")
    print(f"[+] Total duration: {sum(stage_durations.values()):.1f}ms")
    print(f"[+] Bottleneck: retrieval ({stage_durations['retrieval']:.1f}ms)")
    print(f"[+] File saved: {output_path}")
    print("\nFeatures to try:")
    print("  - Interactive Plotly chart (zoom, pan)")
    print("  - Hover over bars for exact values")
    print("  - Click to drill down into stage details")


# ============================================================================
# Demo 4: User Preferences
# ============================================================================

def demo_preferences():
    """Demonstrate user preferences customization."""
    print("\n" + "="*80)
    print("DEMO 4: User Preferences Customization")
    print("="*80)

    # Create preferences
    prefs = UserPreferences(
        preferred_panels=[PanelType.TIMELINE, PanelType.NETWORK],
        color_scheme='dark',
        detail_level='detailed',
        max_panels=6,
        enable_animations=True,
        auto_expand_errors=True
    )

    print("\nUser Preferences:")
    print(f"  - Color scheme: {prefs.color_scheme}")
    print(f"  - Detail level: {prefs.detail_level}")
    print(f"  - Max panels: {prefs.max_panels}")
    print(f"  - Animations: {prefs.enable_animations}")
    print(f"  - Auto-expand errors: {prefs.auto_expand_errors}")
    print(f"  - Preferred panels: {[p.value for p in prefs.preferred_panels]}")

    # Serialize to JSON (what gets saved to localStorage)
    prefs_json = prefs.to_dict()

    print(f"\nSerialized to localStorage:")
    import json
    print(json.dumps(prefs_json, indent=2))

    # Create dashboard with preferences applied
    spacetime = MockSpacetime(
        query_text="Test with custom preferences",
        response="Preferences applied",
        tool_used="test",
        confidence=0.95,
        trace=MockTrace(datetime.now(), datetime.now(), 100.0)
    )

    # Use DashboardConstructor with preferences
    from HoloLoom.visualization.strategy import StrategySelector

    selector = StrategySelector(user_prefs=prefs)

    print("\n[+] Preferences created and serialized")
    print("[+] Ready for localStorage persistence")
    print("\nTo use in browser:")
    print("  1. Open any generated dashboard")
    print("  2. Click settings button (bottom-right)")
    print("  3. Adjust preferences")
    print("  4. Preferences auto-save to localStorage")
    print("  5. Reload page - preferences persist!")


# ============================================================================
# Main
# ============================================================================

def main():
    print("""
    ================================================================================
                  Simple Dashboard Demo - No Heavy Dependencies
                        "Show Off the Features Fast"
    ================================================================================

    This demo creates dashboards FAST - no orchestrator initialization!

    Generating 4 interactive dashboards:
      1. Basic interactive dashboard (expand/collapse, preferences)
      2. D3.js network graph (drag nodes, physics simulation)
      3. Plotly timeline (zoom, pan, hover)
      4. User preferences demo (localStorage, customization)
    """)

    try:
        demo_basic_dashboard()
        demo_network_graph()
        demo_timeline()
        demo_preferences()

        print("\n" + "="*80)
        print("ALL DEMOS COMPLETE!")
        print("="*80)

        dashboards_dir = Path("dashboards")
        files = list(dashboards_dir.glob("demo_*.html"))

        print(f"\nGenerated {len(files)} dashboards:")
        for f in sorted(files):
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name:25} ({size_kb:6.1f} KB)")

        print("\n" + "="*80)
        print("OPEN IN BROWSER TO SEE:")
        print("="*80)
        print("  [+] Click expand buttons to toggle panels")
        print("  [+] Click panels for drill-down modals")
        print("  [+] Drag network graph nodes")
        print("  [+] Zoom/pan Plotly charts")
        print("  [+] Click settings ([settings]) for preferences")
        print("  [+] Toggle dark mode")
        print("  [+] Reload - state persists via localStorage!")

        print("\nQuick open:")
        for f in sorted(files):
            print(f"  file://{f.absolute()}")

        # Offer to open
        print("\n" + "="*80)
        response = input("Open dashboards in browser? (y/n): ").strip().lower()
        if response == 'y':
            import webbrowser
            for f in sorted(files):
                webbrowser.open(f'file://{f.absolute()}')
            print("[+] Opened in browser!")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
