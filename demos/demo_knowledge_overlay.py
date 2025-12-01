#!/usr/bin/env python3
"""
Demo: Knowledge Overlay System
================================

Interactive demo showing HoloLoom's Phase 4 AR knowledge overlay visualization:
- Create knowledge overlays from HoloLoom memory
- Apply different layout algorithms
- Memory Palace room creation
- Visibility mode switching

Part of HoloLoom Phase 4: Spatial Computing Enhancement (November 2025)
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Knowledge overlay imports
from HoloLoom.spatial.knowledge_overlay import (
    KnowledgeOverlayManager,
    KnowledgeNodeOverlay,
    KnowledgeEdgeOverlay,
    OverlayCluster,
    MemoryPalaceRoom,
    LayoutEngine,
    OverlayStyle,
    EdgeStyle,
    LayoutAlgorithm,
    VisibilityMode,
    create_knowledge_overlay_manager
)
from HoloLoom.spatial.math_types import Vector3, Color


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_subheader(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def demo_overlay_creation():
    """Demo 1: Create knowledge overlays."""
    print_header("Demo 1: Knowledge Overlay Creation")

    # Create manager using factory
    manager = create_knowledge_overlay_manager()
    print("✓ Created KnowledgeOverlayManager")

    # Create individual overlays
    print_subheader("Creating Individual Overlays")

    overlays = [
        ("thompson_sampling", "Thompson Sampling",
         "Bayesian approach to exploration/exploitation",
         ["machine_learning", "optimization"], 0.95, 0.9),
        ("bayesian_methods", "Bayesian Methods",
         "Probabilistic inference using Bayes' theorem",
         ["machine_learning", "statistics"], 0.90, 0.85),
        ("exploration", "Exploration-Exploitation",
         "Trade-off between exploring new options and exploiting known good ones",
         ["machine_learning", "decision_making"], 0.85, 0.75),
        ("neural_networks", "Neural Networks",
         "Interconnected layers of artificial neurons",
         ["deep_learning", "machine_learning"], 0.92, 0.8),
        ("reinforcement_learning", "Reinforcement Learning",
         "Learning through interaction with environment",
         ["machine_learning", "decision_making"], 0.88, 0.7),
    ]

    for node_id, title, summary, tags, confidence, importance in overlays:
        overlay = manager.create_overlay(
            node_id=node_id,
            title=title,
            summary=summary,
            tags=tags,
            confidence=confidence,
            importance=importance
        )
        print(f"  ✓ Created: {title}")
        print(f"    ID: {overlay.overlay_id}")
        print(f"    Style: {overlay.style.name}")
        print(f"    Tags: {', '.join(tags)}")

    print(f"\n  Total overlays: {len(manager.overlays)}")
    return manager


def demo_edge_creation(manager: KnowledgeOverlayManager):
    """Demo 2: Create relationship edges."""
    print_header("Demo 2: Relationship Edges")

    # Get overlay IDs
    overlays_by_title = {o.title: o.overlay_id for o in manager.overlays.values()}

    # Define relationships
    relationships = [
        ("Thompson Sampling", "Bayesian Methods", "USES"),
        ("Thompson Sampling", "Exploration-Exploitation", "IS_A"),
        ("Reinforcement Learning", "Exploration-Exploitation", "USES"),
        ("Neural Networks", "Reinforcement Learning", "USES"),
        ("Bayesian Methods", "Neural Networks", "RELATED"),
    ]

    print_subheader("Creating Edges")
    for source, target, rel_type in relationships:
        source_id = overlays_by_title.get(source)
        target_id = overlays_by_title.get(target)

        if source_id and target_id:
            edge = manager.create_edge(source_id, target_id, rel_type)
            print(f"  ✓ {source} --[{rel_type}]--> {target}")
            print(f"    Edge ID: {edge.edge_id}")
            print(f"    Color: ({edge.color.r:.1f}, {edge.color.g:.1f}, {edge.color.b:.1f})")

    print(f"\n  Total edges: {len(manager.edges)}")


def demo_layout_algorithms(manager: KnowledgeOverlayManager):
    """Demo 3: Different layout algorithms."""
    print_header("Demo 3: Layout Algorithms")

    algorithms = [
        LayoutAlgorithm.FORCE_DIRECTED,
        LayoutAlgorithm.RADIAL,
        LayoutAlgorithm.HIERARCHICAL,
        LayoutAlgorithm.CIRCULAR,
        LayoutAlgorithm.GRID,
        LayoutAlgorithm.CLUSTER,
    ]

    for alg in algorithms:
        print_subheader(f"{alg.name} Layout")

        # Apply layout
        manager.apply_layout(
            algorithm=alg,
            center=Vector3(0, 1.5, 0),
            spread=3.0
        )

        # Show positions
        print(f"  Positions after {alg.name}:")
        for overlay in list(manager.overlays.values())[:3]:  # Show first 3
            p = overlay.position
            print(f"    • {overlay.title[:20]}: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")
        print(f"    ... and {len(manager.overlays) - 3} more")


def demo_visibility_modes(manager: KnowledgeOverlayManager):
    """Demo 4: Visibility modes."""
    print_header("Demo 4: Visibility Modes")

    # Show different visibility modes
    print_subheader("Available Visibility Modes")
    for mode in VisibilityMode:
        print(f"  • {mode.name}: ", end="")
        descriptions = {
            VisibilityMode.ALWAYS: "Always visible",
            VisibilityMode.PROXIMITY: "Visible when nearby",
            VisibilityMode.GAZE: "Visible when looked at",
            VisibilityMode.GESTURE: "Visible on gesture",
            VisibilityMode.CONTEXT: "Context-aware visibility",
            VisibilityMode.QUERY: "Visible from query results",
        }
        print(descriptions.get(mode, ""))

    # Demo proximity-based visibility
    print_subheader("Proximity-Based Visibility Demo")

    # Set all overlays to proximity mode
    for overlay in manager.overlays.values():
        overlay.visibility_mode = VisibilityMode.PROXIMITY
        overlay.visibility_distance = 5.0

    # Test with different viewer positions
    viewer_positions = [
        (Vector3(0, 0, 0), "Origin"),
        (Vector3(10, 0, 0), "Far away (10m)"),
        (Vector3(2, 1.5, 0), "Close (2m)"),
    ]

    for pos, desc in viewer_positions:
        manager.update_visibility(pos)
        visible_count = sum(1 for o in manager.overlays.values() if o.is_visible)
        print(f"  Viewer at {desc}: {visible_count}/{len(manager.overlays)} visible")


def demo_overlay_interaction(manager: KnowledgeOverlayManager):
    """Demo 5: Overlay interaction."""
    print_header("Demo 5: Overlay Interaction")

    # Get first overlay for demo
    overlay = list(manager.overlays.values())[0]
    print(f"  Demo overlay: {overlay.title}")

    print_subheader("Selection")
    initial_access = overlay.access_count
    manager.select_overlay(overlay.overlay_id)
    print(f"  ✓ Selected: {overlay.is_selected}")
    print(f"  Access count: {initial_access} → {overlay.access_count}")

    print_subheader("Expand/Collapse")
    manager.expand_overlay(overlay.overlay_id)
    print(f"  ✓ Expanded: {overlay.is_expanded}")

    manager.collapse_overlay(overlay.overlay_id)
    print(f"  ✓ Collapsed: {not overlay.is_expanded}")

    print_subheader("Pin/Unpin")
    manager.pin_overlay(overlay.overlay_id)
    print(f"  ✓ Pinned: {overlay.is_pinned}")

    # Visibility when pinned (should always be visible)
    manager.update_visibility(Vector3(100, 0, 0))  # Far away
    print(f"  Visible when far (pinned): {overlay.is_visible}")

    manager.unpin_overlay(overlay.overlay_id)


def demo_clustering(manager: KnowledgeOverlayManager):
    """Demo 6: Auto-clustering by tags."""
    print_header("Demo 6: Auto-Clustering")

    # Auto cluster by tags
    clusters = manager.auto_cluster(min_cluster_size=2)

    print(f"✓ Created {len(clusters)} clusters automatically")
    print_subheader("Clusters by Tag")

    for cluster in clusters:
        print(f"\n  Cluster: {cluster.name}")
        print(f"    ID: {cluster.cluster_id}")
        print(f"    Center: ({cluster.center.x:.1f}, {cluster.center.y:.1f}, {cluster.center.z:.1f})")
        print(f"    Overlays: {len(cluster.overlays)}")
        for oid in cluster.overlays[:3]:
            if oid in manager.overlays:
                print(f"      • {manager.overlays[oid].title}")


def demo_memory_palace(manager: KnowledgeOverlayManager):
    """Demo 7: Memory Palace creation."""
    print_header("Demo 7: Memory Palace")

    # Define room configurations
    room_configs = [
        {
            "name": "Machine Learning Library",
            "theme": "library",
            "position": {"x": 0, "y": 0, "z": 0},
            "size": {"x": 12, "y": 4, "z": 12},
            "overlay_tags": ["machine_learning"],
            "connected_rooms": ["Deep Learning Lab"]
        },
        {
            "name": "Deep Learning Lab",
            "theme": "lab",
            "position": {"x": 15, "y": 0, "z": 0},
            "size": {"x": 10, "y": 4, "z": 10},
            "overlay_tags": ["deep_learning"],
            "connected_rooms": ["Machine Learning Library", "Statistics Garden"]
        },
        {
            "name": "Statistics Garden",
            "theme": "garden",
            "position": {"x": 30, "y": 0, "z": 0},
            "size": {"x": 8, "y": 4, "z": 8},
            "overlay_tags": ["statistics"],
            "connected_rooms": ["Deep Learning Lab"]
        },
        {
            "name": "Decision Making Arena",
            "theme": "arena",
            "position": {"x": 0, "y": 0, "z": 15},
            "size": {"x": 10, "y": 4, "z": 10},
            "overlay_tags": ["decision_making"],
            "connected_rooms": ["Machine Learning Library"]
        }
    ]

    print_subheader("Building Memory Palace")
    rooms = manager.build_memory_palace(room_configs)

    print(f"✓ Created {len(rooms)} rooms")

    for room_id, room in rooms.items():
        print(f"\n  Room: {room.name}")
        print(f"    Theme: {room.theme}")
        print(f"    Center: ({room.center.x:.0f}, {room.center.y:.0f}, {room.center.z:.0f})")
        print(f"    Size: {room.size.x:.0f}x{room.size.y:.0f}x{room.size.z:.0f}m")
        print(f"    Overlays: {len(room.overlays)}")
        print(f"    Connected to: {len(room.connected_rooms)} rooms")

        # List overlays in room
        for oid in room.overlays[:2]:
            if oid in manager.overlays:
                overlay = manager.overlays[oid]
                p = overlay.position
                print(f"      • {overlay.title} at ({p.x:.1f}, {p.y:.1f}, {p.z:.1f})")


def demo_bulk_creation(manager: KnowledgeOverlayManager):
    """Demo 8: Bulk creation from knowledge graph data."""
    print_header("Demo 8: Bulk Creation from Knowledge Data")

    # Simulate HoloLoom knowledge graph data
    knowledge_nodes = [
        {"id": "bert", "title": "BERT", "summary": "Bidirectional Encoder Representations",
         "tags": ["nlp", "transformer"], "confidence": 0.95, "importance": 0.85},
        {"id": "gpt", "title": "GPT", "summary": "Generative Pre-trained Transformer",
         "tags": ["nlp", "transformer"], "confidence": 0.92, "importance": 0.90},
        {"id": "transformer", "title": "Transformer Architecture",
         "summary": "Attention-based neural network architecture",
         "tags": ["nlp", "deep_learning"], "confidence": 0.98, "importance": 0.95},
        {"id": "attention", "title": "Self-Attention",
         "summary": "Mechanism to weigh input importance",
         "tags": ["transformer", "mechanism"], "confidence": 0.90, "importance": 0.80},
    ]

    relationships = [
        {"source_id": "bert", "target_id": "transformer", "relationship_type": "USES"},
        {"source_id": "gpt", "target_id": "transformer", "relationship_type": "USES"},
        {"source_id": "transformer", "target_id": "attention", "relationship_type": "USES"},
    ]

    print_subheader("Creating Overlays from Knowledge Data")
    overlays, edges = manager.create_overlays_from_knowledge(knowledge_nodes, relationships)

    print(f"✓ Created {len(overlays)} overlays and {len(edges)} edges from knowledge data")

    for overlay in overlays:
        p = overlay.position
        print(f"  • {overlay.title}: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")


def demo_state_export(manager: KnowledgeOverlayManager):
    """Demo 9: Export state for WebXR client."""
    print_header("Demo 9: State Export")

    state = manager.to_state()

    print_subheader("Exported State Summary")
    print(f"  Overlays: {state['overlay_count']}")
    print(f"  Edges: {state['edge_count']}")
    print(f"  Clusters: {len(state['clusters'])}")
    print(f"  Memory Palace Rooms: {len(state['rooms'])}")
    print(f"  Current Layout: {state['current_layout']}")

    # Show sample overlay state
    if state['overlays']:
        sample_id = list(state['overlays'].keys())[0]
        sample = state['overlays'][sample_id]
        print_subheader("Sample Overlay State (for WebXR)")
        print(f"  overlay_id: {sample['overlay_id']}")
        print(f"  title: {sample['title']}")
        print(f"  position: {sample['position']}")
        print(f"  style: {sample['style']}")
        print(f"  primary_color: {sample['primary_color']}")
        print(f"  is_visible: {sample['is_visible']}")


async def demo_query_results(manager: KnowledgeOverlayManager):
    """Demo 10: Display query results."""
    print_header("Demo 10: Query Results Display")

    # Simulate query results
    query = "What is Thompson Sampling?"
    results = [
        {"id": "q1", "title": "Thompson Sampling Basics",
         "summary": "Introduction to Thompson Sampling algorithm",
         "tags": ["basics"], "confidence": 0.95, "importance": 0.9},
        {"id": "q2", "title": "Bayesian Bandits",
         "summary": "Using Bayesian methods for multi-armed bandits",
         "tags": ["bandits"], "confidence": 0.88, "importance": 0.8},
        {"id": "q3", "title": "Exploration Strategies",
         "summary": "Comparing exploration strategies",
         "tags": ["strategies"], "confidence": 0.82, "importance": 0.7},
    ]

    print(f"  Query: '{query}'")
    print(f"  Results: {len(results)} items")

    # Clear existing and show results
    viewer_pos = Vector3(0, 1.5, 0)
    await manager.show_query_results(query, results, viewer_pos)

    print_subheader("Query Result Overlays")
    for overlay in manager.overlays.values():
        p = overlay.position
        print(f"  • {overlay.title}: ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")


async def main():
    """Run the complete knowledge overlay demo."""
    print("\n" + "=" * 60)
    print("  HoloLoom Phase 4: Knowledge Overlay System Demo")
    print("  November 2025")
    print("=" * 60)

    try:
        # Demo 1: Create overlays
        manager = demo_overlay_creation()

        # Demo 2: Create edges
        demo_edge_creation(manager)

        # Demo 3: Layout algorithms
        demo_layout_algorithms(manager)

        # Demo 4: Visibility modes
        demo_visibility_modes(manager)

        # Demo 5: Interaction
        demo_overlay_interaction(manager)

        # Demo 6: Clustering
        demo_clustering(manager)

        # Demo 7: Memory Palace
        demo_memory_palace(manager)

        # Demo 8: Bulk creation
        demo_bulk_creation(manager)

        # Demo 9: State export
        demo_state_export(manager)

        # Demo 10: Query results
        await demo_query_results(manager)

        print("\n" + "=" * 60)
        print("  Demo Complete!")
        print("=" * 60)
        print("\nAll features demonstrated:")
        print("  ✓ Individual overlay creation")
        print("  ✓ Relationship edge creation")
        print("  ✓ Layout algorithms (6 types)")
        print("  ✓ Visibility modes (6 types)")
        print("  ✓ Overlay interaction (select, expand, pin)")
        print("  ✓ Auto-clustering by tags")
        print("  ✓ Memory Palace room creation")
        print("  ✓ Bulk creation from knowledge data")
        print("  ✓ State export for WebXR")
        print("  ✓ Query results display")

    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
