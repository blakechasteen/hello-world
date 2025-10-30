#!/usr/bin/env python3
"""
Integrated Dashboard Demo - Full HoloLoom + Dashboard Integration
==================================================================
Demonstrates complete integration of self-constructing dashboards
with the WeavingOrchestrator.

Features demonstrated:
1. weave_and_visualize() - One-shot API
2. DashboardOrchestrator - Extended orchestrator with dashboard generation
3. Network graph visualization with D3.js force-directed layout
4. Auto-opening in browser for interactive exploration

Author: Claude Code
Date: October 29, 2025
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any

from HoloLoom.config import Config
from HoloLoom.documentation.types import MemoryShard, Query


# ============================================================================
# Mock Data for Demo (would come from actual HoloLoom in production)
# ============================================================================

def create_test_shards():
    """Create test memory shards."""
    return [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian approach to multi-armed bandits",
            episode="knowledge_base",
            entities=["Thompson Sampling", "Bayesian", "bandits"],
            motifs=["reinforcement_learning"],
            metadata={"topic": "reinforcement_learning"}
        ),
        MemoryShard(
            id="shard_2",
            text="The WeavingOrchestrator implements 9-step weaving cycle",
            episode="architecture_docs",
            entities=["WeavingOrchestrator", "weaving cycle"],
            motifs=["architecture"],
            metadata={"topic": "architecture"}
        ),
        MemoryShard(
            id="shard_3",
            text="Matryoshka embeddings enable multi-scale retrieval",
            episode="technical_docs",
            entities=["Matryoshka", "embeddings", "retrieval"],
            motifs=["embeddings"],
            metadata={"topic": "embeddings"}
        )
    ]


async def demo_one_shot_api():
    """Demo: weave_and_visualize() one-shot API."""
    print("\n" + "="*80)
    print("DEMO 1: One-Shot API (weave_and_visualize)")
    print("="*80)
    print("\nThis is the simplest API - just pass a query and get a dashboard!\n")

    from HoloLoom.visualization import weave_and_visualize

    # Create config and shards
    cfg = Config.fast()
    shards = create_test_shards()

    # One-shot: weave + visualize
    print("Executing: weave_and_visualize('What is Thompson Sampling?', ...)")

    try:
        result = await weave_and_visualize(
            "What is Thompson Sampling?",
            cfg=cfg,
            shards=shards,
            save_path="dashboards/demo_one_shot.html",
            open_browser=False  # Set to True to open in browser
        )

        print(f"\n[+] Weaving complete:")
        print(f"    Query: {result.spacetime.query_text}")
        print(f"    Tool: {result.spacetime.tool_used}")
        print(f"    Confidence: {result.spacetime.confidence:.2f}")
        print(f"    Duration: {result.spacetime.trace.duration_ms:.1f}ms")

        print(f"\n[+] Dashboard generated:")
        print(f"    Title: {result.dashboard.title}")
        print(f"    Panels: {len(result.dashboard.panels)}")
        print(f"    Layout: {result.dashboard.layout.value}")
        print(f"    Saved to: {result.file_path}")

        print("\n[OK] One-shot API demo complete!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


async def demo_dashboard_orchestrator():
    """Demo: DashboardOrchestrator with network visualization."""
    print("\n" + "="*80)
    print("DEMO 2: DashboardOrchestrator with Network Visualization")
    print("="*80)
    print("\nShows extended orchestrator with dashboard capabilities\n")

    from HoloLoom.visualization import DashboardOrchestrator

    # Create config and shards
    cfg = Config.fast()
    shards = create_test_shards()

    # Initialize dashboard-enabled orchestrator
    orch = DashboardOrchestrator(
        cfg=cfg,
        shards=shards,
        enable_dashboard_generation=True,
        dashboard_theme='light'
    )

    print("[+] DashboardOrchestrator initialized")
    print("    Features: Auto-dashboard generation enabled")
    print("    Theme: Light mode")

    # Weave with dashboard
    query = Query(text="How does the weaving orchestrator work?")

    print(f"\n[+] Weaving with auto-dashboard...")
    print(f"    Query: {query.text}")

    try:
        result = await orch.weave_with_dashboard(
            query,
            save_path="dashboards/demo_orchestrator.html",
            open_browser=False
        )

        print(f"\n[+] Weaving complete:")
        print(f"    Duration: {result.spacetime.trace.duration_ms:.1f}ms")
        print(f"    Stages: {list(result.spacetime.trace.stage_durations.keys())}")

        print(f"\n[+] Dashboard generated:")
        print(f"    Title: {result.dashboard.title}")
        print(f"    Panels: {[p.type.value for p in result.dashboard.panels]}")
        print(f"    HTML size: {len(result.html)} chars")
        print(f"    File: {result.file_path}")

        print("\n[OK] DashboardOrchestrator demo complete!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


async def demo_network_graph():
    """Demo: Network graph visualization with D3.js."""
    print("\n" + "="*80)
    print("DEMO 3: Interactive Network Graph (D3.js)")
    print("="*80)
    print("\nGenerates dashboard with force-directed graph visualization\n")

    from HoloLoom.visualization import DashboardConstructor, HTMLRenderer
    from HoloLoom.visualization.dashboard import (
        Dashboard, Panel, PanelType, PanelSize, LayoutType
    )

    # Create mock spacetime with network data
    @dataclass
    class MockTrace:
        start_time: datetime
        end_time: datetime
        duration_ms: float
        stage_durations: Dict[str, float] = field(default_factory=dict)
        threads_activated: List[str] = field(default_factory=list)

    @dataclass
    class MockSpacetime:
        query_text: str
        response: str
        tool_used: str
        confidence: float
        trace: MockTrace
        metadata: Dict[str, Any] = field(default_factory=dict)

    # Create network data (nodes + edges)
    network_data = {
        'nodes': [
            {'id': 'motif', 'label': 'Motif Detector', 'size': 15, 'color': '#6366f1'},
            {'id': 'embedding', 'label': 'Embeddings', 'size': 18, 'color': '#10b981'},
            {'id': 'spectral', 'label': 'Spectral', 'size': 12, 'color': '#f59e0b'},
            {'id': 'graph', 'label': 'Graph Store', 'size': 20, 'color': '#ef4444'},
            {'id': 'policy', 'label': 'Policy Engine', 'size': 16, 'color': '#8b5cf6'},
        ],
        'edges': [
            {'source': 'motif', 'target': 'policy'},
            {'source': 'embedding', 'target': 'policy'},
            {'source': 'spectral', 'target': 'policy'},
            {'source': 'graph', 'target': 'embedding'},
            {'source': 'graph', 'target': 'spectral'},
        ],
        'node_count': 5
    }

    spacetime = MockSpacetime(
        query_text="Show me the thread activation network",
        response="Network visualization of activated threads",
        tool_used="visualize",
        confidence=0.95,
        trace=MockTrace(
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=234.5,
            stage_durations={
                "features": 78.2,
                "retrieval": 89.1,
                "decision": 45.3,
                "execution": 21.9
            },
            threads_activated=['motif', 'embedding', 'spectral', 'graph', 'policy']
        ),
        metadata={"complexity": "FAST"}
    )

    # Create dashboard with network panel
    network_panel = Panel(
        id="network_1",
        type=PanelType.NETWORK,
        title="Thread Activation Network",
        subtitle="Interactive force-directed graph (drag nodes to explore)",
        data=network_data,
        size=PanelSize.FULL_WIDTH
    )

    dashboard = Dashboard(
        title="Network Visualization Demo",
        layout=LayoutType.FLOW,
        panels=[network_panel],
        spacetime=spacetime,
        metadata={"complexity": "FAST", "generated_at": datetime.now().isoformat()}
    )

    # Render to HTML
    renderer = HTMLRenderer()
    html = renderer.render(dashboard)

    # Save
    output_path = Path("dashboards/demo_network_graph.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')

    print(f"[+] Network graph dashboard generated:")
    print(f"    Nodes: {len(network_data['nodes'])}")
    print(f"    Edges: {len(network_data['edges'])}")
    print(f"    Visualization: D3.js force-directed graph")
    print(f"    Interactive: Drag nodes to rearrange")
    print(f"    Saved to: {output_path}")

    print("\n[OK] Network graph demo complete!")


async def main():
    """Run all demos."""
    print("""
    ================================================================================
                          Integrated Dashboard System Demo
                        "Wolfram Alpha for HoloLoom Queries"
    ================================================================================

    This demo shows the complete integration of self-constructing dashboards
    with the HoloLoom WeavingOrchestrator.

    Three demos:
    1. One-Shot API - Simplest usage (weave_and_visualize)
    2. Dashboard Orchestrator - Extended orchestrator with auto-dashboards
    3. Network Graph - Interactive D3.js force-directed visualization
    """)

    try:
        await demo_one_shot_api()
        await demo_dashboard_orchestrator()
        await demo_network_graph()

        print("\n" + "="*80)
        print("ALL DEMOS COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("  - dashboards/demo_one_shot.html")
        print("  - dashboards/demo_orchestrator.html")
        print("  - dashboards/demo_network_graph.html")
        print("\nOpen these files in your browser to see:")
        print("  [+] Auto-generated optimal panels")
        print("  [+] Interactive Plotly timeline charts")
        print("  [+] D3.js force-directed network graphs")
        print("  [+] Responsive Tailwind CSS layouts")
        print("\nThe 'Wolfram Alpha Machine' is fully operational!")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
