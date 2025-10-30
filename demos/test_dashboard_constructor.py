#!/usr/bin/env python3
"""
Test: DashboardConstructor End-to-End
======================================
Verifies complete self-constructing dashboard flow:
  Spacetime -> StrategySelector -> Dashboard -> HTML
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any

try:
    from HoloLoom.visualization import DashboardConstructor, DashboardRenderer
    print("OK Imports successful\n")
except Exception as e:
    print(f"Import error: {e}\n")


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


def test_end_to_end():
    """Test complete flow: Spacetime -> Dashboard -> HTML."""
    print("[Test] End-to-End Dashboard Generation")

    spacetime = MockSpacetime(
        query_text="How does the weaving orchestrator work?",
        response="The weaving orchestrator coordinates feature extraction...",
        tool_used="explain",
        confidence=0.87,
        trace=MockTrace(
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=432.1,
            stage_durations={
                "features": 120.5,
                "retrieval": 98.3,
                "decision": 156.2,
                "execution": 57.1
            },
            threads_activated=["motif", "embedding", "spectral", "graph"]
        ),
        metadata={}
    )

    constructor = DashboardConstructor()
    dashboard = constructor.construct(spacetime)

    print(f"  Title: {dashboard.title}")
    print(f"  Panels: {len(dashboard.panels)}")
    print(f"  Types: {[p.type.value for p in dashboard.panels]}")

    renderer = DashboardRenderer()
    html = renderer.render(dashboard)

    assert "<html" in html
    assert "Plotly.newPlot" in html
    assert len(html) > 1000

    output_path = Path(__file__).parent / "output_dashboard.html"
    output_path.write_text(html)
    print(f"  Saved: {output_path}")
    print("  OK End-to-end flow working\n")


def main():
    try:
        print("="*80)
        print("Dashboard Constructor Test")
        print("="*80)
        print()
        test_end_to_end()
        print("="*80)
        print("OK Test passed!")
        print("="*80)
        return 0
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())