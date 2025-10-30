#!/usr/bin/env python3
"""
Self-Constructing Dashboard Demo
=================================
Full integration: Query -> HoloLoom -> Spacetime -> Auto-Dashboard

This demonstrates the "Wolfram Alpha Machine" in action:
  1. User asks a question
  2. HoloLoom processes it
  3. System auto-generates optimal visualization
  4. Dashboard opens in browser
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any
import webbrowser

from HoloLoom.visualization import DashboardConstructor, DashboardRenderer


# Mock Spacetime for demo (replace with actual HoloLoom integration)
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


def create_sample_queries():
    """Create sample queries demonstrating different dashboard types."""

    queries = {
        "factual": MockSpacetime(
            query_text="What is Thompson Sampling?",
            response="""Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.
            It balances exploration and exploitation by sampling from posterior distributions of rewards.

            Key features:
            - Bayesian inference for arm selection
            - Natural exploration via sampling
            - Optimal regret bounds
            - Used in HoloLoom's UnifiedPolicy for tool selection""",
            tool_used="answer",
            confidence=0.94,
            trace=MockTrace(
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_ms=143.2,
                stage_durations={
                    "features": 35.1,
                    "retrieval": 52.3,
                    "decision": 41.8,
                    "execution": 14.0
                },
                threads_activated=["embedding", "motif"]
            ),
            metadata={"complexity": "FAST"}
        ),

        "exploratory": MockSpacetime(
            query_text="How does the weaving orchestrator work?",
            response="""The WeavingOrchestrator implements a 9-step cycle:

            1. Loom Command - Pattern card selection
            2. Chrono Trigger - Temporal window creation
            3. Yarn Graph - Thread selection from memory
            4. Resonance Shed - Feature extraction, DotPlasma creation
            5. Warp Space - Continuous manifold tensioning
            6. Convergence Engine - Discrete decision collapse
            7. Tool Execution - Action with results
            8. Spacetime Fabric - Provenance and trace
            9. Reflection Buffer - Learning from outcome

            This architecture enables seamless symbolic <-> continuous transitions.""",
            tool_used="explain",
            confidence=0.89,
            trace=MockTrace(
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_ms=512.7,
                stage_durations={
                    "features": 145.3,
                    "retrieval": 198.2,
                    "decision": 112.5,
                    "execution": 56.7
                },
                threads_activated=["motif", "embedding", "spectral", "graph", "semantic"]
            ),
            metadata={"complexity": "FULL"}
        ),

        "debugging": MockSpacetime(
            query_text="Why did the Neo4j connection fail?",
            response="""Connection failure analysis:

            1. Neo4j service unreachable on bolt://localhost:7687
            2. Docker container may not be running
            3. Auto-fallback to InMemory backend triggered
            4. Query completed successfully with degraded performance

            Recommendation: Start Docker services with 'docker-compose up -d'""",
            tool_used="debug",
            confidence=0.82,
            trace=MockTrace(
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_ms=1247.3,
                stage_durations={
                    "features": 89.2,
                    "retrieval": 892.5,  # Slow due to timeout + retry
                    "decision": 198.1,
                    "execution": 67.5
                },
                threads_activated=["graph", "memory"],
                errors=[
                    "ConnectionRefused: Neo4j unreachable at bolt://localhost:7687",
                    "Timeout: Connection attempt exceeded 5000ms",
                    "FallbackTriggered: Using InMemory backend"
                ]
            ),
            metadata={"complexity": "FAST"}
        ),

        "optimization": MockSpacetime(
            query_text="How can I speed up retrieval?",
            response="""Performance optimization recommendations:

            1. Enable vector caching (40% speedup)
            2. Use FAST mode instead of FULL for simple queries
            3. Reduce embedding dimensions: 384 -> 192 for non-critical paths
            4. Enable Neo4j connection pooling
            5. Use matryoshka importance gating for recursive crawling

            Expected impact: 150ms -> 60ms average query time""",
            tool_used="optimize",
            confidence=0.91,
            trace=MockTrace(
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_ms=287.9,
                stage_durations={
                    "features": 78.3,
                    "retrieval": 125.7,  # Current bottleneck
                    "decision": 56.2,
                    "execution": 27.7
                },
                threads_activated=["embedding", "cache", "profile"]
            ),
            metadata={"complexity": "FAST"}
        )
    }

    return queries


async def generate_dashboard(query_type: str, spacetime: MockSpacetime, output_dir: Path):
    """Generate and save dashboard for a query."""

    print(f"\n{'='*80}")
    print(f"Query Type: {query_type.upper()}")
    print(f"{'='*80}")
    print(f"Query: {spacetime.query_text}")
    print(f"Confidence: {spacetime.confidence:.2f}")
    print(f"Duration: {spacetime.trace.duration_ms:.1f}ms")
    print(f"Threads: {', '.join(spacetime.trace.threads_activated)}")
    if spacetime.trace.errors:
        print(f"Errors: {len(spacetime.trace.errors)}")

    # Construct dashboard
    constructor = DashboardConstructor()
    dashboard = constructor.construct(spacetime)

    print(f"\nDashboard Generated:")
    print(f"  Title: {dashboard.title}")
    print(f"  Layout: {dashboard.layout.value}")
    print(f"  Panels: {len(dashboard.panels)}")
    print(f"  Panel types: {[p.type.value for p in dashboard.panels]}")

    # Render to HTML
    renderer = DashboardRenderer()
    html = renderer.render(dashboard)

    # Save
    output_path = output_dir / f"dashboard_{query_type}.html"
    output_path.write_text(html, encoding='utf-8')
    print(f"  Saved: {output_path}")

    return output_path


async def main():
    """Run the demo."""

    print("""
    ================================================================================
                      Self-Constructing Dashboard Demo
                         "The Wolfram Alpha Machine"
    ================================================================================

    This demo shows how HoloLoom automatically generates optimal visualizations
    based on query intent and data availability.

    Generating 4 dashboards:
      1. FACTUAL     - "What is X?" -> Metric + Text
      2. EXPLORATORY - "How does X work?" -> Timeline + Network
      3. DEBUGGING   - "Why did X fail?" -> Errors + Timeline
      4. OPTIMIZATION - "How to speed up X?" -> Bottleneck analysis
    """)

    # Create output directory
    output_dir = Path(__file__).parent / "dashboards"
    output_dir.mkdir(exist_ok=True)

    # Generate dashboards for all query types
    queries = create_sample_queries()
    paths = []

    for query_type, spacetime in queries.items():
        path = await generate_dashboard(query_type, spacetime, output_dir)
        paths.append(path)
        await asyncio.sleep(0.1)  # Small delay for visual clarity

    # Summary
    print(f"\n{'='*80}")
    print("COMPLETE - All Dashboards Generated")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    for path in paths:
        print(f"  - {path.name}")

    # Offer to open in browser
    print(f"\n{'='*80}")
    response = input("Open dashboards in browser? (y/n): ").strip().lower()
    if response == 'y':
        print("\nOpening dashboards...")
        for i, path in enumerate(paths):
            webbrowser.open(f'file://{path.absolute()}')
            if i < len(paths) - 1:
                await asyncio.sleep(0.5)  # Stagger browser tabs

    print("\nDemo complete!")
    print("\nKey Features Demonstrated:")
    print("  [+] Intent detection (factual, exploratory, debugging, optimization)")
    print("  [+] Data-driven panel selection")
    print("  [+] Timeline visualization with stage colors")
    print("  [+] Error highlighting")
    print("  [+] Adaptive complexity (FAST/FULL)")
    print("  [+] Responsive HTML + Plotly charts")
    print("\nThe 'Edward Tufte Machine' is operational!")


if __name__ == "__main__":
    asyncio.run(main())
