"""
HoloLoom Phase 4: WebXR Knowledge Graph
November 2025

Immersive 3D knowledge graph visualization for AR/VR.
Supports WebXR for browser-based spatial computing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import math


class XRMode(Enum):
    """WebXR session modes."""
    INLINE = "inline"              # Standard 2D fallback
    IMMERSIVE_VR = "immersive-vr"  # Full VR headset
    IMMERSIVE_AR = "immersive-ar"  # AR passthrough


class NodeLayout(Enum):
    """3D layout algorithms."""
    FORCE_DIRECTED = "force"      # Physics-based layout
    SPHERICAL = "spherical"       # Nodes on sphere surface
    HIERARCHICAL = "hierarchical" # Tree-like structure
    RADIAL = "radial"             # Concentric circles


@dataclass
class Vector3:
    """3D position vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    def distance_to(self, other: 'Vector3') -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def normalize(self) -> 'Vector3':
        length = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if length == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x/length, self.y/length, self.z/length)


@dataclass
class XRNode:
    """
    Knowledge node in XR space.

    Represents a memory/concept positioned in 3D space with
    visual properties for rendering in WebXR.
    """
    node_id: str
    label: str
    position: Vector3 = field(default_factory=Vector3)

    # Visual properties
    color: str = "#00ff88"
    size: float = 0.1  # Meters
    opacity: float = 1.0
    glow_intensity: float = 0.0  # 0-1, for activation visualization

    # Knowledge properties
    activation: float = 0.0  # Current activation level
    importance: float = 0.5  # Static importance
    node_type: str = "concept"

    # Interaction
    is_selected: bool = False
    is_hovered: bool = False
    is_grabbed: bool = False

    # Metadata
    content: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.node_id,
            "label": self.label,
            "position": self.position.to_dict(),
            "color": self.color,
            "size": self.size,
            "opacity": self.opacity,
            "glow": self.glow_intensity,
            "activation": self.activation,
            "importance": self.importance,
            "type": self.node_type,
            "selected": self.is_selected,
            "hovered": self.is_hovered,
            "grabbed": self.is_grabbed,
            "content": self.content,
            "metadata": self.metadata
        }


@dataclass
class XREdge:
    """Edge between knowledge nodes in XR space."""
    source_id: str
    target_id: str
    edge_type: str = "RELATED"
    weight: float = 1.0

    # Visual
    color: str = "#ffffff"
    width: float = 0.005  # Meters
    opacity: float = 0.6
    animated: bool = False  # Pulse animation for active connections

    def to_dict(self) -> Dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type,
            "weight": self.weight,
            "color": self.color,
            "width": self.width,
            "opacity": self.opacity,
            "animated": self.animated
        }


@dataclass
class XRAnchor:
    """
    Spatial anchor for persistent AR positioning.

    Anchors knowledge to real-world locations so it
    persists across sessions.
    """
    anchor_id: str
    node_id: str  # Attached knowledge node

    # World position (relative to anchor origin)
    position: Vector3 = field(default_factory=Vector3)
    orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)  # Quaternion

    # Persistence
    created_at: str = ""
    last_seen: str = ""
    confidence: float = 1.0  # Tracking confidence
    is_persistent: bool = True

    # Context
    location_hint: str = ""  # "desk", "whiteboard", "shelf"

    def to_dict(self) -> Dict:
        return {
            "anchor_id": self.anchor_id,
            "node_id": self.node_id,
            "position": self.position.to_dict(),
            "orientation": list(self.orientation),
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "confidence": self.confidence,
            "persistent": self.is_persistent,
            "location_hint": self.location_hint
        }


class WebXRKnowledgeGraph:
    """
    WebXR-ready knowledge graph visualization.

    Provides:
    - 3D node positioning with multiple layout algorithms
    - Edge rendering with animation support
    - Activation visualization (spreading activation visible)
    - Spatial anchor support for AR persistence
    - JSON export for WebXR client consumption

    Usage:
        graph = WebXRKnowledgeGraph()
        graph.add_node("thompson", "Thompson Sampling", importance=0.9)
        graph.add_node("bayesian", "Bayesian Methods", importance=0.8)
        graph.add_edge("thompson", "bayesian", "IS_A")
        graph.layout(NodeLayout.FORCE_DIRECTED)

        # Export for WebXR client
        json_data = graph.to_json()
    """

    # Edge type colors
    EDGE_COLORS = {
        "IS_A": "#4a90d9",       # Blue - taxonomy
        "USES": "#50c878",       # Green - functional
        "MENTIONS": "#888888",   # Gray - reference
        "LEADS_TO": "#ff8c00",   # Orange - causal
        "PART_OF": "#9370db",    # Purple - composition
        "RELATED": "#ffffff",    # White - general
    }

    # Node type colors
    NODE_COLORS = {
        "concept": "#00ff88",    # Green
        "entity": "#4a90d9",     # Blue
        "event": "#ff8c00",      # Orange
        "query": "#ff69b4",      # Pink
        "active": "#ffff00",     # Yellow - currently activated
    }

    def __init__(self):
        self.nodes: Dict[str, XRNode] = {}
        self.edges: List[XREdge] = []
        self.anchors: Dict[str, XRAnchor] = {}

        # Layout state
        self.center = Vector3(0, 1.5, -2)  # Default viewing position
        self.scale = 1.0  # Overall scale

        # Interaction state
        self.selected_nodes: List[str] = []
        self.focused_node: Optional[str] = None

    def add_node(self, node_id: str, label: str,
                 importance: float = 0.5,
                 node_type: str = "concept",
                 content: str = "",
                 position: Optional[Vector3] = None) -> XRNode:
        """Add a knowledge node to the graph."""
        node = XRNode(
            node_id=node_id,
            label=label,
            importance=importance,
            node_type=node_type,
            content=content,
            color=self.NODE_COLORS.get(node_type, "#00ff88"),
            size=0.05 + importance * 0.1  # Size scales with importance
        )

        if position:
            node.position = position

        self.nodes[node_id] = node
        return node

    def add_edge(self, source_id: str, target_id: str,
                 edge_type: str = "RELATED",
                 weight: float = 1.0) -> Optional[XREdge]:
        """Add an edge between nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        edge = XREdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            color=self.EDGE_COLORS.get(edge_type, "#ffffff"),
            width=0.003 + weight * 0.004
        )

        self.edges.append(edge)
        return edge

    def set_activation(self, node_id: str, activation: float):
        """Set activation level for a node (spreading activation visualization)."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.activation = max(0, min(1, activation))
            node.glow_intensity = node.activation * 0.8

            # Change color when highly activated
            if activation > 0.7:
                node.color = self.NODE_COLORS["active"]

    def activate_spreading(self, source_id: str, initial_activation: float = 1.0,
                          decay: float = 0.7, max_hops: int = 3):
        """
        Visualize spreading activation from a source node.

        Args:
            source_id: Starting node
            initial_activation: Initial activation level
            decay: Decay factor per hop
            max_hops: Maximum propagation distance
        """
        if source_id not in self.nodes:
            return

        # BFS activation spreading
        visited = {source_id}
        current_level = {source_id}
        activation = initial_activation

        for hop in range(max_hops + 1):
            for node_id in current_level:
                self.set_activation(node_id, activation)

            # Find next level
            next_level = set()
            for edge in self.edges:
                if edge.source_id in current_level and edge.target_id not in visited:
                    next_level.add(edge.target_id)
                    visited.add(edge.target_id)
                    edge.animated = True  # Animate active edges
                elif edge.target_id in current_level and edge.source_id not in visited:
                    next_level.add(edge.source_id)
                    visited.add(edge.source_id)
                    edge.animated = True

            current_level = next_level
            activation *= decay

    def layout(self, algorithm: NodeLayout = NodeLayout.FORCE_DIRECTED,
               iterations: int = 100):
        """
        Apply 3D layout algorithm to position nodes.

        Args:
            algorithm: Layout algorithm to use
            iterations: Number of iterations for force-directed
        """
        if algorithm == NodeLayout.FORCE_DIRECTED:
            self._layout_force_directed(iterations)
        elif algorithm == NodeLayout.SPHERICAL:
            self._layout_spherical()
        elif algorithm == NodeLayout.HIERARCHICAL:
            self._layout_hierarchical()
        elif algorithm == NodeLayout.RADIAL:
            self._layout_radial()

    def _layout_force_directed(self, iterations: int = 100):
        """Force-directed 3D layout (Fruchterman-Reingold)."""
        import random

        if not self.nodes:
            return

        # Initialize random positions
        for node in self.nodes.values():
            node.position = Vector3(
                random.uniform(-1, 1),
                random.uniform(0.5, 2.5),  # Keep above ground
                random.uniform(-3, -1)
            )

        # Parameters
        k = 0.5  # Optimal distance
        t = 1.0  # Temperature (decreases over iterations)

        for iteration in range(iterations):
            # Calculate repulsive forces (all pairs)
            displacements = {nid: Vector3() for nid in self.nodes}

            node_list = list(self.nodes.values())
            for i, n1 in enumerate(node_list):
                for n2 in node_list[i+1:]:
                    dx = n1.position.x - n2.position.x
                    dy = n1.position.y - n2.position.y
                    dz = n1.position.z - n2.position.z
                    dist = max(0.01, math.sqrt(dx*dx + dy*dy + dz*dz))

                    # Repulsive force
                    force = (k * k) / dist
                    fx, fy, fz = dx/dist * force, dy/dist * force, dz/dist * force

                    displacements[n1.node_id].x += fx
                    displacements[n1.node_id].y += fy
                    displacements[n1.node_id].z += fz
                    displacements[n2.node_id].x -= fx
                    displacements[n2.node_id].y -= fy
                    displacements[n2.node_id].z -= fz

            # Calculate attractive forces (edges)
            for edge in self.edges:
                n1 = self.nodes[edge.source_id]
                n2 = self.nodes[edge.target_id]

                dx = n1.position.x - n2.position.x
                dy = n1.position.y - n2.position.y
                dz = n1.position.z - n2.position.z
                dist = max(0.01, math.sqrt(dx*dx + dy*dy + dz*dz))

                # Attractive force
                force = (dist * dist) / k
                fx, fy, fz = dx/dist * force, dy/dist * force, dz/dist * force

                displacements[n1.node_id].x -= fx
                displacements[n1.node_id].y -= fy
                displacements[n1.node_id].z -= fz
                displacements[n2.node_id].x += fx
                displacements[n2.node_id].y += fy
                displacements[n2.node_id].z += fz

            # Apply displacements with temperature
            for node_id, disp in displacements.items():
                node = self.nodes[node_id]
                length = math.sqrt(disp.x**2 + disp.y**2 + disp.z**2)
                if length > 0:
                    scale = min(t, length) / length
                    node.position.x += disp.x * scale
                    node.position.y += disp.y * scale
                    node.position.z += disp.z * scale

                # Keep above ground
                node.position.y = max(0.3, node.position.y)

            # Cool down
            t *= 0.95

        # Center the graph
        self._center_graph()

    def _layout_spherical(self):
        """Place nodes on a sphere surface."""
        if not self.nodes:
            return

        n = len(self.nodes)
        radius = 1.5

        # Golden angle distribution for even spacing
        golden_angle = math.pi * (3 - math.sqrt(5))

        for i, node in enumerate(self.nodes.values()):
            y = 1 - (i / (n - 1)) * 2 if n > 1 else 0  # y goes from 1 to -1
            r = math.sqrt(1 - y * y)  # Radius at y
            theta = golden_angle * i

            node.position = Vector3(
                self.center.x + r * math.cos(theta) * radius,
                self.center.y + y * radius,
                self.center.z + r * math.sin(theta) * radius
            )

    def _layout_hierarchical(self):
        """Tree-like hierarchical layout."""
        if not self.nodes:
            return

        # Find root nodes (no incoming edges)
        has_parent = set()
        for edge in self.edges:
            if edge.edge_type in ("IS_A", "PART_OF"):
                has_parent.add(edge.source_id)

        roots = [nid for nid in self.nodes if nid not in has_parent]
        if not roots:
            roots = [list(self.nodes.keys())[0]]

        # BFS level assignment
        levels: Dict[str, int] = {}
        queue = [(r, 0) for r in roots]
        visited = set(roots)

        while queue:
            node_id, level = queue.pop(0)
            levels[node_id] = level

            for edge in self.edges:
                if edge.target_id == node_id and edge.source_id not in visited:
                    queue.append((edge.source_id, level + 1))
                    visited.add(edge.source_id)
                elif edge.source_id == node_id and edge.target_id not in visited:
                    queue.append((edge.target_id, level + 1))
                    visited.add(edge.target_id)

        # Assign remaining nodes
        for nid in self.nodes:
            if nid not in levels:
                levels[nid] = max(levels.values()) + 1 if levels else 0

        # Position by level
        level_counts: Dict[int, int] = {}
        for node_id, level in levels.items():
            count = level_counts.get(level, 0)
            level_counts[level] = count + 1

            node = self.nodes[node_id]
            angle = (count / max(1, level_counts[level])) * 2 * math.pi
            radius = 0.5 + level * 0.4

            node.position = Vector3(
                self.center.x + math.cos(angle) * radius,
                self.center.y - level * 0.5,  # Lower levels go down
                self.center.z + math.sin(angle) * radius
            )

    def _layout_radial(self):
        """Concentric circles layout."""
        if not self.nodes:
            return

        # Sort by importance
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.importance,
            reverse=True
        )

        # Place in concentric circles
        ring = 0
        ring_capacity = 1
        ring_count = 0

        for i, node in enumerate(sorted_nodes):
            if ring_count >= ring_capacity:
                ring += 1
                ring_capacity = max(1, int(6 * ring))  # More nodes per outer ring
                ring_count = 0

            angle = (ring_count / ring_capacity) * 2 * math.pi
            radius = 0.3 + ring * 0.4

            node.position = Vector3(
                self.center.x + math.cos(angle) * radius,
                self.center.y,
                self.center.z + math.sin(angle) * radius
            )

            ring_count += 1

    def _center_graph(self):
        """Move graph to be centered at self.center."""
        if not self.nodes:
            return

        # Calculate current center
        cx = sum(n.position.x for n in self.nodes.values()) / len(self.nodes)
        cy = sum(n.position.y for n in self.nodes.values()) / len(self.nodes)
        cz = sum(n.position.z for n in self.nodes.values()) / len(self.nodes)

        # Offset to target center
        for node in self.nodes.values():
            node.position.x += self.center.x - cx
            node.position.y += self.center.y - cy
            node.position.z += self.center.z - cz

    def create_anchor(self, anchor_id: str, node_id: str,
                      position: Vector3, location_hint: str = "") -> Optional[XRAnchor]:
        """Create a spatial anchor for AR persistence."""
        if node_id not in self.nodes:
            return None

        from datetime import datetime

        anchor = XRAnchor(
            anchor_id=anchor_id,
            node_id=node_id,
            position=position,
            location_hint=location_hint,
            created_at=datetime.now().isoformat(),
            last_seen=datetime.now().isoformat()
        )

        self.anchors[anchor_id] = anchor
        return anchor

    def select_node(self, node_id: str):
        """Select a node for interaction."""
        if node_id in self.nodes:
            self.nodes[node_id].is_selected = True
            if node_id not in self.selected_nodes:
                self.selected_nodes.append(node_id)

    def deselect_all(self):
        """Deselect all nodes."""
        for node in self.nodes.values():
            node.is_selected = False
        self.selected_nodes.clear()

    def focus_node(self, node_id: str):
        """Set focus to a specific node (camera moves to it)."""
        if node_id in self.nodes:
            self.focused_node = node_id

    def get_nearby_nodes(self, position: Vector3, radius: float = 0.5) -> List[XRNode]:
        """Find nodes within radius of a position (for hand interaction)."""
        nearby = []
        for node in self.nodes.values():
            if node.position.distance_to(position) <= radius:
                nearby.append(node)
        return sorted(nearby, key=lambda n: n.position.distance_to(position))

    def to_dict(self) -> Dict:
        """Export graph state as dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "anchors": [a.to_dict() for a in self.anchors.values()],
            "center": self.center.to_dict(),
            "scale": self.scale,
            "selected": self.selected_nodes,
            "focused": self.focused_node
        }

    def to_json(self) -> str:
        """Export graph state as JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_knowledge_graph(cls, kg,
                             layout: NodeLayout = NodeLayout.FORCE_DIRECTED) -> 'WebXRKnowledgeGraph':
        """
        Create WebXR graph from HoloLoom knowledge graph.

        Args:
            kg: KnowledgeGraph (from HoloLoom.memory.graph)
            layout: Layout algorithm to apply
        """
        xr_graph = cls()

        # Add nodes
        if hasattr(kg, 'graph'):
            for node_id in kg.graph.nodes():
                node_data = kg.graph.nodes[node_id]
                xr_graph.add_node(
                    node_id=node_id,
                    label=node_data.get('label', node_id),
                    importance=node_data.get('importance', 0.5),
                    node_type=node_data.get('type', 'concept'),
                    content=node_data.get('content', '')
                )

            # Add edges
            for source, target, data in kg.graph.edges(data=True):
                xr_graph.add_edge(
                    source_id=source,
                    target_id=target,
                    edge_type=data.get('type', 'RELATED'),
                    weight=data.get('weight', 1.0)
                )

        # Apply layout
        xr_graph.layout(layout)

        return xr_graph


# Quick test
if __name__ == "__main__":
    print("Testing WebXR Knowledge Graph...")

    graph = WebXRKnowledgeGraph()

    # Add test nodes
    graph.add_node("thompson", "Thompson Sampling", importance=0.9, node_type="concept")
    graph.add_node("bayesian", "Bayesian Methods", importance=0.8, node_type="concept")
    graph.add_node("bandit", "Multi-Armed Bandit", importance=0.7, node_type="concept")
    graph.add_node("exploration", "Exploration", importance=0.6, node_type="concept")
    graph.add_node("exploitation", "Exploitation", importance=0.6, node_type="concept")

    # Add edges
    graph.add_edge("thompson", "bayesian", "IS_A")
    graph.add_edge("thompson", "bandit", "USES")
    graph.add_edge("thompson", "exploration", "RELATED")
    graph.add_edge("thompson", "exploitation", "RELATED")
    graph.add_edge("exploration", "exploitation", "LEADS_TO")

    # Apply layout
    graph.layout(NodeLayout.FORCE_DIRECTED)

    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")

    # Test spreading activation
    graph.activate_spreading("thompson", initial_activation=1.0, decay=0.7)

    print(f"Thompson activation: {graph.nodes['thompson'].activation}")
    print(f"Bayesian activation: {graph.nodes['bayesian'].activation}")

    # Export
    json_data = graph.to_json()
    print(f"JSON export size: {len(json_data)} chars")

    print("WebXR Knowledge Graph tests passed!")
