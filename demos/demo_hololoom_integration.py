#!/usr/bin/env python3
"""
HoloLoom Complete Integration Demo
===================================
Shows how auto-visualization integrates with the full HoloLoom stack.

The complete flow:
  Query -> WeavingOrchestrator -> Spacetime -> auto() -> Dashboard

Zero configuration at every step.

Author: Claude Code
Date: October 29, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_future_integration():
    """
    Shows how auto-visualization will integrate with WeavingOrchestrator.

    NOTE: This is a design demo - shows the intended API.
    Actual integration requires WeavingOrchestrator modifications.
    """

    print("\n" + "="*80)
    print("FUTURE: COMPLETE HOLOLOOM INTEGRATION")
    print("="*80)

    print("""
    The Vision: Query -> Dashboard in one line

    CURRENT (manual):
        orchestrator = WeavingOrchestrator(cfg=config)
        spacetime = await orchestrator.weave(query)
        # ... manually extract data ...
        # ... manually create dashboard ...

    FUTURE (automatic):
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.visualization import auto

        orchestrator = WeavingOrchestrator(cfg=config)
        spacetime = await orchestrator.weave(query)

        # One line - auto-extracts and visualizes
        auto(spacetime, save_path='result.html')

    Even Better (built-in):
        # WeavingOrchestrator auto-generates dashboard by default
        spacetime = await orchestrator.weave(query, auto_dashboard=True)
        # -> spacetime.dashboard automatically populated
        # -> dashboard.html automatically saved

    Ruthlessly integrated.
    """)


def demo_simulated_integration():
    """
    Simulate the integration with mock Spacetime objects.

    Shows what the experience will be like.
    """

    print("\n" + "="*80)
    print("SIMULATED INTEGRATION")
    print("="*80)

    from dataclasses import dataclass, field
    from typing import Dict, Any
    from HoloLoom.visualization import auto

    # Simulate a Spacetime result from a query
    @dataclass
    class MockTrace:
        duration_ms: float = 345.2
        stages: list = field(default_factory=lambda: [
            {'name': 'Feature Extraction', 'duration_ms': 78.3},
            {'name': 'Memory Retrieval', 'duration_ms': 145.6},
            {'name': 'Policy Decision', 'duration_ms': 89.2},
            {'name': 'Response Generation', 'duration_ms': 32.1}
        ])

    @dataclass
    class MockSpacetime:
        query_text: str = "What are the winter survival rates for my bee colonies?"
        response: str = "Analysis shows 85% avg survival with combined treatment"
        tool_used: str = "data_analysis"
        confidence: float = 0.94
        trace: Any = field(default_factory=MockTrace)
        metadata: Dict[str, Any] = field(default_factory=lambda: {
            'analysis_data': {
                'month': ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'],
                'supplemental_feeding': [98, 96, 94, 91, 88, 85],
                'insulation_only': [97, 93, 88, 82, 76, 70],
                'control': [95, 88, 79, 68, 55, 42],
                'combined_treatment': [99, 98, 97, 95, 93, 91],
                'avg_temperature': [-2, -5, -8, -12, -10, -3]
            }
        })

        def to_dict(self):
            return {
                'query': self.query_text,
                'response': self.response,
                'confidence': self.confidence
            }

    # Simulate the query result
    print("\n1. User asks query:")
    print('   "What are the winter survival rates for my bee colonies?"')

    print("\n2. WeavingOrchestrator processes query...")
    spacetime = MockSpacetime()

    print(f"\n3. Spacetime generated:")
    print(f"   - Query: {spacetime.query_text}")
    print(f"   - Confidence: {spacetime.confidence:.2%}")
    print(f"   - Duration: {spacetime.trace.duration_ms}ms")
    print(f"   - Has analysis data: {bool(spacetime.metadata.get('analysis_data'))}")

    print("\n4. Auto-visualize (ONE LINE):")
    print("   auto(spacetime, save_path='dashboards/hololoom_result.html')")

    # Generate dashboard automatically
    dashboard = auto(spacetime, save_path='dashboards/hololoom_result.html')

    print(f"\n5. Dashboard generated:")
    print(f"   - Title: {dashboard.title}")
    print(f"   - Panels: {len(dashboard.panels)}")
    print(f"   - Layout: {dashboard.layout.value}")
    print(f"   - Patterns: {dashboard.metadata.get('patterns_detected', [])}")

    print("\n6. User opens dashboard in browser and sees:")
    print("   - Line charts showing survival trends over 6 months")
    print("   - Scatter plots showing temperature correlation")
    print("   - Insight cards with auto-detected patterns")
    print("   - All interactive with zoom, pan, dark mode")

    print("\n[+] Complete flow: Query -> Spacetime -> Dashboard (automatic)")

    return dashboard


def demo_memory_visualization():
    """
    Shows memory graph -> network visualization.
    """

    print("\n" + "="*80)
    print("MEMORY GRAPH VISUALIZATION")
    print("="*80)

    print("""
    Integration with HoloLoom memory backends:

    from HoloLoom.memory.backend_factory import create_memory_backend
    from HoloLoom.visualization import auto

    # Create memory backend (NetworkX, Neo4j, etc.)
    memory = await create_memory_backend(config)

    # ... add some entities and relations ...
    memory.add_entity('bee_colony_1', type='colony')
    memory.add_entity('treatment_A', type='treatment')
    memory.add_relation('bee_colony_1', 'treatment_A', 'uses')

    # Visualize entire knowledge graph (ONE LINE)
    auto(memory, save_path='knowledge_graph.html')

    Result: Interactive network visualization with:
    - D3.js force-directed layout
    - Draggable nodes
    - Relationship labels
    - Entity metadata on hover
    - Auto-generated metrics (node count, edge count)

    Zero configuration. Just: auto(memory)
    """)

    # Simulate with mock network data
    from HoloLoom.visualization.auto import _build_network_dashboard

    network_data = {
        'network': {
            'nodes': [
                {'id': 'colony1', 'label': 'Colony Alpha', 'type': 'colony', 'size': 20},
                {'id': 'colony2', 'label': 'Colony Beta', 'type': 'colony', 'size': 18},
                {'id': 'treatment_a', 'label': 'Supplemental Feeding', 'type': 'treatment', 'size': 15},
                {'id': 'treatment_b', 'label': 'Insulation', 'type': 'treatment', 'size': 15},
                {'id': 'metric_survival', 'label': 'Survival Rate', 'type': 'metric', 'size': 12},
            ],
            'edges': [
                {'source': 'colony1', 'target': 'treatment_a', 'label': 'uses'},
                {'source': 'colony2', 'target': 'treatment_b', 'label': 'uses'},
                {'source': 'treatment_a', 'target': 'metric_survival', 'label': 'affects'},
                {'source': 'treatment_b', 'target': 'metric_survival', 'label': 'affects'},
            ]
        }
    }

    dashboard = _build_network_dashboard(network_data, "Bee Treatment Knowledge Graph")

    from HoloLoom.visualization import save
    save(dashboard, 'dashboards/memory_network.html')

    print("\n[+] Memory graph visualized: dashboards/memory_network.html")
    print(f"[+] Network: {len(network_data['network']['nodes'])} nodes, {len(network_data['network']['edges'])} edges")

    return dashboard


def demo_reflection_buffer_analysis():
    """
    Future: Visualize learning from reflection buffer.
    """

    print("\n" + "="*80)
    print("REFLECTION BUFFER ANALYSIS (FUTURE)")
    print("="*80)

    print("""
    Integration with HoloLoom learning system:

    from HoloLoom.reflection.buffer import ReflectionBuffer
    from HoloLoom.visualization import auto

    # Reflection buffer tracks query performance over time
    buffer = ReflectionBuffer()

    # After many queries, analyze learning
    performance_data = {
        'query_id': list(range(100)),
        'duration_ms': [...],  # Response times
        'confidence': [...],   # Confidence scores
        'user_rating': [...],  # Feedback ratings
        'tool_used': [...],    # Which tool was selected
    }

    # Visualize learning trends (ONE LINE)
    auto(performance_data, title="HoloLoom Learning Analysis")

    Auto-detects:
    - Time-series trends (are we getting faster?)
    - Correlation (confidence vs user rating)
    - Outliers (which queries struggled?)
    - Tool usage patterns (which tools most effective?)

    Intelligence insights:
    - "Response time improving 15% over 100 queries"
    - "Strong correlation (r=0.87) between confidence and ratings"
    - "Outlier detected: Query #47 took 2.3s (3.2 sigma)"
    - "Recommendation: thompson_sampling most effective (92% success)"

    All automatic from reflection buffer data.
    """)

    print("\n[+] Future enhancement: auto-learning dashboard from reflection buffer")


def main():
    """Run all integration demos."""

    print("\n" + "="*80)
    print("HOLOLOOM COMPLETE INTEGRATION")
    print("="*80)
    print("""
    Ruthless integration of auto-visualization into HoloLoom ecosystem:

    1. WeavingOrchestrator -> auto(spacetime)
    2. Memory backends -> auto(memory_graph)
    3. Reflection buffer -> auto(learning_data)

    All automatic. All one line. All integrated.
    """)

    # Run demos
    demo_future_integration()
    spacetime_dashboard = demo_simulated_integration()
    memory_dashboard = demo_memory_visualization()
    demo_reflection_buffer_analysis()

    # Summary
    print("\n" + "="*80)
    print("INTEGRATION SUMMARY")
    print("="*80)
    print("""
    Demonstrated:
    1. Spacetime -> Dashboard (automatic data extraction)
    2. Memory graph -> Network viz (automatic network generation)
    3. Reflection buffer -> Learning dashboard (future)

    Generated files:
    - dashboards/hololoom_result.html  (Spacetime visualization)
    - dashboards/memory_network.html   (Knowledge graph network)

    API Simplicity:
    - auto(spacetime) - visualize query results
    - auto(memory)    - visualize knowledge graph
    - auto(data)      - visualize any data

    One function. Everything integrated. Zero configuration.

    Next step: Integrate auto() directly into WeavingOrchestrator
    so every query automatically generates a dashboard.

    Vision:
        spacetime = await orchestrator.weave(query)
        # spacetime.dashboard automatically populated
        # dashboard.html automatically saved

    Ruthless elegance at every layer.
    """)

    print("="*80)


if __name__ == '__main__':
    main()
