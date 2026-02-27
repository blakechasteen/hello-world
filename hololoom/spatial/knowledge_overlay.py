# HoloLoom Phase 4: AR Knowledge Overlay System
# November 2025
#
# Displays HoloLoom knowledge graph as AR overlays in spatial environments
# - Knowledge nodes as floating 3D cards
# - Relationship edges as visual connections
# - Context-aware knowledge surfacing
# - Interactive query and exploration
# - Memory palace visualization

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from datetime import datetime
import asyncio
import logging
import math
import uuid

from HoloLoom.spatial.math_types import Vector3, Quaternion, Color, Transform, BoundingBox

logger = logging.getLogger(__name__)


class OverlayStyle(Enum):
    """Visual style for knowledge overlays."""
    CARD = auto()           # Flat card floating in space
    SPHERE = auto()         # 3D sphere with content
    HOLOGRAM = auto()       # Holographic projection
    MINIMAL = auto()        # Text-only minimal
    RICH = auto()           # Full rich media
    GLOW = auto()           # Glowing orb with details on focus
    CONSTELLATION = auto()  # Star-like with connections
    MEMORY_PALACE = auto()  # Architectural element


class EdgeStyle(Enum):
    """Visual style for relationship edges."""
    LINE = auto()           # Simple line
    ARROW = auto()          # Directional arrow
    BEAM = auto()           # Energy beam
    PARTICLES = auto()      # Flowing particles
    DASHED = auto()         # Dashed line
    GRADIENT = auto()       # Color gradient
    PULSING = auto()        # Pulsing animation


class LayoutAlgorithm(Enum):
    """Layout algorithms for knowledge graph visualization."""
    FORCE_DIRECTED = auto()   # Physics-based spring layout
    HIERARCHICAL = auto()     # Tree-like hierarchy
    RADIAL = auto()           # Radial from center
    CIRCULAR = auto()         # Arranged in circle
    GRID = auto()             # Grid layout
    CLUSTER = auto()          # Grouped by clusters
    SPATIAL = auto()          # Based on spatial context
    MEMORY_PALACE = auto()    # Room-based layout


class VisibilityMode(Enum):
    """How overlays become visible."""
    ALWAYS = auto()         # Always visible
    PROXIMITY = auto()      # Visible when nearby
    GAZE = auto()           # Visible when looked at
    GESTURE = auto()        # Visible on gesture
    CONTEXT = auto()        # Context-aware visibility
    QUERY = auto()          # Visible from query results


@dataclass
class KnowledgeNodeOverlay:
    """
    A single knowledge node displayed as an AR overlay.
    """
    overlay_id: str
    node_id: str  # HoloLoom memory/knowledge node ID

    # Content
    title: str
    summary: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0

    # Spatial
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    scale: float = 1.0

    # Visual
    style: OverlayStyle = OverlayStyle.CARD
    primary_color: Color = field(default_factory=lambda: Color(0.2, 0.4, 0.8, 0.9))
    secondary_color: Color = field(default_factory=lambda: Color(0.1, 0.2, 0.4, 0.8))
    text_color: Color = field(default_factory=Color.white)

    # Size
    width: float = 0.4  # meters
    height: float = 0.3

    # Visibility
    visibility_mode: VisibilityMode = VisibilityMode.PROXIMITY
    visibility_distance: float = 5.0  # meters
    is_visible: bool = True
    opacity: float = 1.0

    # Interaction
    is_interactive: bool = True
    is_selected: bool = False
    is_expanded: bool = False
    is_pinned: bool = False

    # Animation
    hover_offset: float = 0.0  # Current hover animation offset
    pulse_phase: float = 0.0   # Current pulse animation phase

    # Metadata
    importance: float = 0.5    # 0-1 importance score
    recency: float = 0.5       # 0-1 recency score
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overlay_id": self.overlay_id,
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "tags": self.tags,
            "confidence": self.confidence,
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
            "scale": self.scale,
            "style": self.style.name,
            "primary_color": self.primary_color.to_dict(),
            "secondary_color": self.secondary_color.to_dict(),
            "text_color": self.text_color.to_dict(),
            "width": self.width,
            "height": self.height,
            "visibility_mode": self.visibility_mode.name,
            "visibility_distance": self.visibility_distance,
            "is_visible": self.is_visible,
            "opacity": self.opacity,
            "is_interactive": self.is_interactive,
            "is_selected": self.is_selected,
            "is_expanded": self.is_expanded,
            "is_pinned": self.is_pinned,
            "importance": self.importance,
            "recency": self.recency,
            "access_count": self.access_count
        }


@dataclass
class KnowledgeEdgeOverlay:
    """
    A relationship edge displayed between knowledge nodes.
    """
    edge_id: str
    source_overlay_id: str
    target_overlay_id: str
    relationship_type: str  # IS_A, USES, MENTIONS, LEADS_TO, etc.

    # Visual
    style: EdgeStyle = EdgeStyle.LINE
    color: Color = field(default_factory=lambda: Color(0.5, 0.5, 0.8, 0.6))
    width: float = 0.01  # meters
    show_label: bool = True

    # Animation
    flow_speed: float = 1.0
    pulse_enabled: bool = False

    # Visibility
    is_visible: bool = True
    opacity: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_overlay_id": self.source_overlay_id,
            "target_overlay_id": self.target_overlay_id,
            "relationship_type": self.relationship_type,
            "style": self.style.name,
            "color": self.color.to_dict(),
            "width": self.width,
            "show_label": self.show_label,
            "flow_speed": self.flow_speed,
            "pulse_enabled": self.pulse_enabled,
            "is_visible": self.is_visible,
            "opacity": self.opacity
        }


@dataclass
class OverlayCluster:
    """
    A cluster of related knowledge overlays.
    """
    cluster_id: str
    name: str
    overlays: List[str] = field(default_factory=list)  # Overlay IDs

    # Spatial
    center: Vector3 = field(default_factory=Vector3)
    radius: float = 2.0

    # Visual
    color: Color = field(default_factory=lambda: Color(0.3, 0.5, 0.7, 0.2))
    show_boundary: bool = True
    label_visible: bool = True

    # State
    is_expanded: bool = True
    is_focused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "overlays": self.overlays,
            "center": self.center.to_dict(),
            "radius": self.radius,
            "color": self.color.to_dict(),
            "show_boundary": self.show_boundary,
            "label_visible": self.label_visible,
            "is_expanded": self.is_expanded,
            "is_focused": self.is_focused
        }


@dataclass
class MemoryPalaceRoom:
    """
    A room in the memory palace containing knowledge overlays.
    """
    room_id: str
    name: str
    theme: str  # Visual theme (library, garden, lab, etc.)

    # Spatial
    center: Vector3 = field(default_factory=Vector3)
    size: Vector3 = field(default_factory=lambda: Vector3(10, 4, 10))
    entrance_position: Vector3 = field(default_factory=Vector3)

    # Content
    overlays: List[str] = field(default_factory=list)
    connected_rooms: List[str] = field(default_factory=list)

    # Visual
    ambient_color: Color = field(default_factory=lambda: Color(0.2, 0.25, 0.3))
    lighting_intensity: float = 1.0

    # State
    is_discovered: bool = False
    is_current: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "name": self.name,
            "theme": self.theme,
            "center": self.center.to_dict(),
            "size": self.size.to_dict(),
            "entrance_position": self.entrance_position.to_dict(),
            "overlays": self.overlays,
            "connected_rooms": self.connected_rooms,
            "ambient_color": self.ambient_color.to_dict(),
            "lighting_intensity": self.lighting_intensity,
            "is_discovered": self.is_discovered,
            "is_current": self.is_current
        }


class LayoutEngine:
    """
    Computes spatial layouts for knowledge overlays.
    """

    def __init__(self):
        self.algorithm = LayoutAlgorithm.FORCE_DIRECTED
        self.center = Vector3(0, 1.5, 0)  # Default center at eye level
        self.spread = 3.0  # Spread radius

        # Force-directed parameters
        self.repulsion_strength = 1.0
        self.attraction_strength = 0.1
        self.damping = 0.9
        self.iterations = 100

    def compute_layout(
        self,
        overlays: List[KnowledgeNodeOverlay],
        edges: List[KnowledgeEdgeOverlay],
        algorithm: Optional[LayoutAlgorithm] = None
    ) -> Dict[str, Vector3]:
        """
        Compute positions for overlays using specified algorithm.

        Returns: Dict mapping overlay_id to position
        """
        alg = algorithm or self.algorithm

        if alg == LayoutAlgorithm.FORCE_DIRECTED:
            return self._force_directed_layout(overlays, edges)
        elif alg == LayoutAlgorithm.HIERARCHICAL:
            return self._hierarchical_layout(overlays, edges)
        elif alg == LayoutAlgorithm.RADIAL:
            return self._radial_layout(overlays, edges)
        elif alg == LayoutAlgorithm.CIRCULAR:
            return self._circular_layout(overlays)
        elif alg == LayoutAlgorithm.GRID:
            return self._grid_layout(overlays)
        elif alg == LayoutAlgorithm.CLUSTER:
            return self._cluster_layout(overlays, edges)
        else:
            return self._radial_layout(overlays, edges)

    def _force_directed_layout(
        self,
        overlays: List[KnowledgeNodeOverlay],
        edges: List[KnowledgeEdgeOverlay]
    ) -> Dict[str, Vector3]:
        """Force-directed spring layout."""
        if not overlays:
            return {}

        # Initialize positions randomly
        positions: Dict[str, Vector3] = {}
        velocities: Dict[str, Vector3] = {}

        for i, overlay in enumerate(overlays):
            angle = (i / len(overlays)) * 2 * math.pi
            positions[overlay.overlay_id] = Vector3(
                self.center.x + math.cos(angle) * self.spread * 0.5,
                self.center.y + (hash(overlay.overlay_id) % 100) / 100 * 0.5,
                self.center.z + math.sin(angle) * self.spread * 0.5
            )
            velocities[overlay.overlay_id] = Vector3(0, 0, 0)

        # Build edge lookup
        edge_pairs = set()
        for edge in edges:
            edge_pairs.add((edge.source_overlay_id, edge.target_overlay_id))

        # Iterate
        for _ in range(self.iterations):
            forces: Dict[str, Vector3] = {oid: Vector3(0, 0, 0) for oid in positions}

            # Repulsion between all pairs
            overlay_ids = list(positions.keys())
            for i, id1 in enumerate(overlay_ids):
                for id2 in overlay_ids[i+1:]:
                    p1 = positions[id1]
                    p2 = positions[id2]

                    dx = p2.x - p1.x
                    dy = p2.y - p1.y
                    dz = p2.z - p1.z
                    dist = max(0.1, math.sqrt(dx*dx + dy*dy + dz*dz))

                    # Repulsion force (inverse square)
                    force = self.repulsion_strength / (dist * dist)
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    fz = (dz / dist) * force

                    forces[id1] = Vector3(
                        forces[id1].x - fx,
                        forces[id1].y - fy,
                        forces[id1].z - fz
                    )
                    forces[id2] = Vector3(
                        forces[id2].x + fx,
                        forces[id2].y + fy,
                        forces[id2].z + fz
                    )

            # Attraction for connected pairs
            for source_id, target_id in edge_pairs:
                if source_id not in positions or target_id not in positions:
                    continue

                p1 = positions[source_id]
                p2 = positions[target_id]

                dx = p2.x - p1.x
                dy = p2.y - p1.y
                dz = p2.z - p1.z
                dist = max(0.1, math.sqrt(dx*dx + dy*dy + dz*dz))

                # Attraction force (linear)
                force = self.attraction_strength * dist
                fx = (dx / dist) * force
                fy = (dy / dist) * force
                fz = (dz / dist) * force

                forces[source_id] = Vector3(
                    forces[source_id].x + fx,
                    forces[source_id].y + fy,
                    forces[source_id].z + fz
                )
                forces[target_id] = Vector3(
                    forces[target_id].x - fx,
                    forces[target_id].y - fy,
                    forces[target_id].z - fz
                )

            # Update velocities and positions
            for oid in positions:
                v = velocities[oid]
                f = forces[oid]

                velocities[oid] = Vector3(
                    (v.x + f.x) * self.damping,
                    (v.y + f.y) * self.damping,
                    (v.z + f.z) * self.damping
                )

                p = positions[oid]
                positions[oid] = Vector3(
                    p.x + velocities[oid].x * 0.1,
                    p.y + velocities[oid].y * 0.1,
                    p.z + velocities[oid].z * 0.1
                )

        return positions

    def _radial_layout(
        self,
        overlays: List[KnowledgeNodeOverlay],
        edges: List[KnowledgeEdgeOverlay]
    ) -> Dict[str, Vector3]:
        """Radial layout from center, with more important nodes closer."""
        positions = {}

        # Sort by importance
        sorted_overlays = sorted(overlays, key=lambda o: o.importance, reverse=True)

        for i, overlay in enumerate(sorted_overlays):
            # Higher importance = closer to center
            ring = int(i / 8)  # 8 items per ring
            angle = (i % 8) / 8 * 2 * math.pi

            radius = self.spread * 0.5 + ring * 0.8
            height_offset = ring * 0.2

            positions[overlay.overlay_id] = Vector3(
                self.center.x + math.cos(angle) * radius,
                self.center.y + height_offset,
                self.center.z + math.sin(angle) * radius
            )

        return positions

    def _hierarchical_layout(
        self,
        overlays: List[KnowledgeNodeOverlay],
        edges: List[KnowledgeEdgeOverlay]
    ) -> Dict[str, Vector3]:
        """Tree-like hierarchical layout."""
        positions = {}

        # Find roots (no incoming edges)
        incoming: Dict[str, int] = {o.overlay_id: 0 for o in overlays}
        children: Dict[str, List[str]] = {o.overlay_id: [] for o in overlays}

        for edge in edges:
            if edge.target_overlay_id in incoming:
                incoming[edge.target_overlay_id] += 1
            if edge.source_overlay_id in children:
                children[edge.source_overlay_id].append(edge.target_overlay_id)

        roots = [oid for oid, count in incoming.items() if count == 0]
        if not roots:
            roots = [overlays[0].overlay_id] if overlays else []

        # BFS to assign levels
        levels: Dict[str, int] = {}
        queue = [(rid, 0) for rid in roots]
        visited = set()

        while queue:
            oid, level = queue.pop(0)
            if oid in visited:
                continue
            visited.add(oid)
            levels[oid] = level

            for child_id in children.get(oid, []):
                if child_id not in visited:
                    queue.append((child_id, level + 1))

        # Position by level
        level_counts: Dict[int, int] = {}
        level_indices: Dict[str, int] = {}

        for oid, level in levels.items():
            if level not in level_counts:
                level_counts[level] = 0
            level_indices[oid] = level_counts[level]
            level_counts[level] += 1

        for overlay in overlays:
            oid = overlay.overlay_id
            level = levels.get(oid, 0)
            index = level_indices.get(oid, 0)
            count = level_counts.get(level, 1)

            x_offset = (index - count / 2) * 1.2
            y_offset = -level * 0.8

            positions[oid] = Vector3(
                self.center.x + x_offset,
                self.center.y + y_offset,
                self.center.z + level * 0.5
            )

        return positions

    def _circular_layout(self, overlays: List[KnowledgeNodeOverlay]) -> Dict[str, Vector3]:
        """Simple circular layout."""
        positions = {}

        for i, overlay in enumerate(overlays):
            angle = (i / max(1, len(overlays))) * 2 * math.pi
            positions[overlay.overlay_id] = Vector3(
                self.center.x + math.cos(angle) * self.spread,
                self.center.y,
                self.center.z + math.sin(angle) * self.spread
            )

        return positions

    def _grid_layout(self, overlays: List[KnowledgeNodeOverlay]) -> Dict[str, Vector3]:
        """Grid layout."""
        positions = {}

        cols = max(1, int(math.sqrt(len(overlays))))
        spacing = 0.8

        for i, overlay in enumerate(overlays):
            row = i // cols
            col = i % cols

            positions[overlay.overlay_id] = Vector3(
                self.center.x + (col - cols / 2) * spacing,
                self.center.y - row * spacing * 0.6,
                self.center.z
            )

        return positions

    def _cluster_layout(
        self,
        overlays: List[KnowledgeNodeOverlay],
        edges: List[KnowledgeEdgeOverlay]
    ) -> Dict[str, Vector3]:
        """Cluster-based layout grouping connected nodes."""
        # Simple clustering by tag similarity
        clusters: Dict[str, List[str]] = {}

        for overlay in overlays:
            # Use first tag as cluster key
            cluster_key = overlay.tags[0] if overlay.tags else "default"
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(overlay.overlay_id)

        positions = {}
        cluster_angle = 0
        angle_step = 2 * math.pi / max(1, len(clusters))

        for cluster_name, overlay_ids in clusters.items():
            # Position cluster center
            cluster_center = Vector3(
                self.center.x + math.cos(cluster_angle) * self.spread,
                self.center.y,
                self.center.z + math.sin(cluster_angle) * self.spread
            )

            # Position items within cluster
            for i, oid in enumerate(overlay_ids):
                inner_angle = (i / max(1, len(overlay_ids))) * 2 * math.pi
                inner_radius = 0.5

                positions[oid] = Vector3(
                    cluster_center.x + math.cos(inner_angle) * inner_radius,
                    cluster_center.y + (i % 3) * 0.3,
                    cluster_center.z + math.sin(inner_angle) * inner_radius
                )

            cluster_angle += angle_step

        return positions


class KnowledgeOverlayManager:
    """
    Manages AR overlays for HoloLoom knowledge visualization.
    """

    def __init__(self):
        self.overlays: Dict[str, KnowledgeNodeOverlay] = {}
        self.edges: Dict[str, KnowledgeEdgeOverlay] = {}
        self.clusters: Dict[str, OverlayCluster] = {}
        self.rooms: Dict[str, MemoryPalaceRoom] = {}

        self.layout_engine = LayoutEngine()
        self.current_layout = LayoutAlgorithm.FORCE_DIRECTED

        # Style defaults
        self.default_style = OverlayStyle.CARD
        self.default_visibility = VisibilityMode.PROXIMITY

        # Animation
        self._animation_time = 0.0
        self._running = False

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Color palette for tags
        self._tag_colors: Dict[str, Color] = {}
        self._color_palette = [
            Color(0.2, 0.4, 0.8),   # Blue
            Color(0.2, 0.7, 0.4),   # Green
            Color(0.8, 0.4, 0.2),   # Orange
            Color(0.7, 0.2, 0.6),   # Purple
            Color(0.2, 0.7, 0.7),   # Cyan
            Color(0.8, 0.7, 0.2),   # Yellow
            Color(0.6, 0.3, 0.3),   # Brown
            Color(0.5, 0.5, 0.7),   # Lavender
        ]
        self._color_index = 0

    def _get_tag_color(self, tag: str) -> Color:
        """Get consistent color for a tag."""
        if tag not in self._tag_colors:
            self._tag_colors[tag] = self._color_palette[self._color_index % len(self._color_palette)]
            self._color_index += 1
        return self._tag_colors[tag]

    # === Overlay Management ===

    def create_overlay(
        self,
        node_id: str,
        title: str,
        summary: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        confidence: float = 1.0,
        importance: float = 0.5,
        position: Optional[Vector3] = None,
        style: Optional[OverlayStyle] = None
    ) -> KnowledgeNodeOverlay:
        """Create a new knowledge overlay."""
        overlay_id = str(uuid.uuid4())[:8]

        # Determine color from tags
        primary_color = self._color_palette[0]
        if tags:
            primary_color = self._get_tag_color(tags[0])

        overlay = KnowledgeNodeOverlay(
            overlay_id=overlay_id,
            node_id=node_id,
            title=title,
            summary=summary,
            content=content,
            tags=tags or [],
            confidence=confidence,
            importance=importance,
            position=position or Vector3(0, 1.5, -2),
            style=style or self.default_style,
            primary_color=primary_color,
            visibility_mode=self.default_visibility
        )

        self.overlays[overlay_id] = overlay
        self._emit_event("overlay_created", {"overlay": overlay.to_dict()})

        return overlay

    def create_overlays_from_knowledge(
        self,
        knowledge_nodes: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Tuple[List[KnowledgeNodeOverlay], List[KnowledgeEdgeOverlay]]:
        """
        Create overlays from HoloLoom knowledge graph data.

        Args:
            knowledge_nodes: List of knowledge node dictionaries with:
                - id, title, summary, content, tags, confidence, importance
            relationships: List of relationship dictionaries with:
                - source_id, target_id, relationship_type

        Returns:
            (overlays, edges)
        """
        # Create node-to-overlay mapping
        node_to_overlay: Dict[str, str] = {}
        created_overlays = []

        for node in knowledge_nodes:
            overlay = self.create_overlay(
                node_id=node.get("id", str(uuid.uuid4())),
                title=node.get("title", "Untitled"),
                summary=node.get("summary"),
                content=node.get("content"),
                tags=node.get("tags", []),
                confidence=node.get("confidence", 1.0),
                importance=node.get("importance", 0.5)
            )
            node_to_overlay[node.get("id")] = overlay.overlay_id
            created_overlays.append(overlay)

        # Create edges
        created_edges = []
        for rel in relationships:
            source_id = node_to_overlay.get(rel.get("source_id"))
            target_id = node_to_overlay.get(rel.get("target_id"))

            if source_id and target_id:
                edge = self.create_edge(
                    source_overlay_id=source_id,
                    target_overlay_id=target_id,
                    relationship_type=rel.get("relationship_type", "RELATED")
                )
                created_edges.append(edge)

        # Compute layout
        self.apply_layout()

        return created_overlays, created_edges

    def remove_overlay(self, overlay_id: str) -> bool:
        """Remove an overlay."""
        if overlay_id not in self.overlays:
            return False

        overlay = self.overlays.pop(overlay_id)

        # Remove associated edges
        edges_to_remove = [
            eid for eid, edge in self.edges.items()
            if edge.source_overlay_id == overlay_id or edge.target_overlay_id == overlay_id
        ]
        for eid in edges_to_remove:
            del self.edges[eid]

        self._emit_event("overlay_removed", {"overlay_id": overlay_id})
        return True

    def get_overlay(self, overlay_id: str) -> Optional[KnowledgeNodeOverlay]:
        """Get overlay by ID."""
        return self.overlays.get(overlay_id)

    def get_overlay_by_node(self, node_id: str) -> Optional[KnowledgeNodeOverlay]:
        """Get overlay by knowledge node ID."""
        for overlay in self.overlays.values():
            if overlay.node_id == node_id:
                return overlay
        return None

    # === Edge Management ===

    def create_edge(
        self,
        source_overlay_id: str,
        target_overlay_id: str,
        relationship_type: str,
        style: EdgeStyle = EdgeStyle.LINE
    ) -> KnowledgeEdgeOverlay:
        """Create an edge between overlays."""
        edge_id = str(uuid.uuid4())[:8]

        # Determine color based on relationship type
        edge_colors = {
            "IS_A": Color(0.2, 0.6, 0.9),
            "USES": Color(0.2, 0.8, 0.4),
            "MENTIONS": Color(0.5, 0.5, 0.5),
            "LEADS_TO": Color(0.9, 0.6, 0.2),
            "PART_OF": Color(0.7, 0.3, 0.7),
            "RELATED": Color(0.5, 0.5, 0.8)
        }

        edge = KnowledgeEdgeOverlay(
            edge_id=edge_id,
            source_overlay_id=source_overlay_id,
            target_overlay_id=target_overlay_id,
            relationship_type=relationship_type,
            style=style,
            color=edge_colors.get(relationship_type, Color(0.5, 0.5, 0.5))
        )

        self.edges[edge_id] = edge
        return edge

    # === Layout ===

    def apply_layout(
        self,
        algorithm: Optional[LayoutAlgorithm] = None,
        center: Optional[Vector3] = None,
        spread: Optional[float] = None
    ):
        """Apply layout algorithm to all overlays."""
        if center:
            self.layout_engine.center = center
        if spread:
            self.layout_engine.spread = spread

        positions = self.layout_engine.compute_layout(
            list(self.overlays.values()),
            list(self.edges.values()),
            algorithm
        )

        for overlay_id, position in positions.items():
            if overlay_id in self.overlays:
                self.overlays[overlay_id].position = position

        self._emit_event("layout_applied", {"algorithm": (algorithm or self.current_layout).name})

    def set_center(self, position: Vector3):
        """Set the center point for layouts."""
        self.layout_engine.center = position

    # === Visibility ===

    def update_visibility(
        self,
        viewer_position: Vector3,
        gaze_target: Optional[str] = None
    ):
        """Update overlay visibility based on viewer position and gaze."""
        for overlay in self.overlays.values():
            if overlay.is_pinned:
                overlay.is_visible = True
                overlay.opacity = 1.0
                continue

            if overlay.visibility_mode == VisibilityMode.ALWAYS:
                overlay.is_visible = True
                overlay.opacity = 1.0

            elif overlay.visibility_mode == VisibilityMode.PROXIMITY:
                dx = overlay.position.x - viewer_position.x
                dy = overlay.position.y - viewer_position.y
                dz = overlay.position.z - viewer_position.z
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)

                if distance < overlay.visibility_distance:
                    overlay.is_visible = True
                    # Fade based on distance
                    fade_start = overlay.visibility_distance * 0.7
                    if distance > fade_start:
                        overlay.opacity = 1.0 - (distance - fade_start) / (overlay.visibility_distance - fade_start)
                    else:
                        overlay.opacity = 1.0
                else:
                    overlay.is_visible = False
                    overlay.opacity = 0.0

            elif overlay.visibility_mode == VisibilityMode.GAZE:
                overlay.is_visible = gaze_target == overlay.overlay_id
                overlay.opacity = 1.0 if overlay.is_visible else 0.0

    # === Interaction ===

    def select_overlay(self, overlay_id: str) -> bool:
        """Select an overlay."""
        # Deselect all
        for overlay in self.overlays.values():
            overlay.is_selected = False

        if overlay_id in self.overlays:
            self.overlays[overlay_id].is_selected = True
            self.overlays[overlay_id].access_count += 1
            self.overlays[overlay_id].last_accessed = datetime.now()
            self._emit_event("overlay_selected", {"overlay_id": overlay_id})
            return True
        return False

    def expand_overlay(self, overlay_id: str) -> bool:
        """Expand an overlay to show full content."""
        if overlay_id in self.overlays:
            self.overlays[overlay_id].is_expanded = True
            self._emit_event("overlay_expanded", {"overlay_id": overlay_id})
            return True
        return False

    def collapse_overlay(self, overlay_id: str) -> bool:
        """Collapse an expanded overlay."""
        if overlay_id in self.overlays:
            self.overlays[overlay_id].is_expanded = False
            self._emit_event("overlay_collapsed", {"overlay_id": overlay_id})
            return True
        return False

    def pin_overlay(self, overlay_id: str) -> bool:
        """Pin an overlay to keep it visible."""
        if overlay_id in self.overlays:
            self.overlays[overlay_id].is_pinned = True
            return True
        return False

    def unpin_overlay(self, overlay_id: str) -> bool:
        """Unpin an overlay."""
        if overlay_id in self.overlays:
            self.overlays[overlay_id].is_pinned = False
            return True
        return False

    # === Clusters ===

    def create_cluster(
        self,
        name: str,
        overlay_ids: List[str],
        center: Optional[Vector3] = None
    ) -> OverlayCluster:
        """Create a cluster of overlays."""
        cluster_id = str(uuid.uuid4())[:8]

        # Calculate center if not provided
        if not center and overlay_ids:
            sum_x = sum_y = sum_z = 0
            count = 0
            for oid in overlay_ids:
                if oid in self.overlays:
                    p = self.overlays[oid].position
                    sum_x += p.x
                    sum_y += p.y
                    sum_z += p.z
                    count += 1
            if count > 0:
                center = Vector3(sum_x / count, sum_y / count, sum_z / count)

        cluster = OverlayCluster(
            cluster_id=cluster_id,
            name=name,
            overlays=overlay_ids,
            center=center or Vector3(0, 1.5, 0)
        )

        self.clusters[cluster_id] = cluster
        return cluster

    def auto_cluster(self, min_cluster_size: int = 2) -> List[OverlayCluster]:
        """Automatically cluster overlays by tags."""
        tag_groups: Dict[str, List[str]] = {}

        for overlay in self.overlays.values():
            for tag in overlay.tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(overlay.overlay_id)

        clusters = []
        for tag, overlay_ids in tag_groups.items():
            if len(overlay_ids) >= min_cluster_size:
                cluster = self.create_cluster(tag.title(), overlay_ids)
                clusters.append(cluster)

        return clusters

    # === Memory Palace ===

    def create_memory_palace_room(
        self,
        name: str,
        theme: str,
        center: Vector3,
        size: Optional[Vector3] = None
    ) -> MemoryPalaceRoom:
        """Create a room in the memory palace."""
        room_id = str(uuid.uuid4())[:8]

        room = MemoryPalaceRoom(
            room_id=room_id,
            name=name,
            theme=theme,
            center=center,
            size=size or Vector3(10, 4, 10),
            entrance_position=Vector3(center.x, center.y, center.z + size.z / 2 if size else center.z + 5)
        )

        self.rooms[room_id] = room
        return room

    def assign_to_room(self, overlay_id: str, room_id: str) -> bool:
        """Assign an overlay to a memory palace room."""
        if overlay_id not in self.overlays or room_id not in self.rooms:
            return False

        room = self.rooms[room_id]
        if overlay_id not in room.overlays:
            room.overlays.append(overlay_id)
        return True

    def build_memory_palace(
        self,
        room_configs: List[Dict[str, Any]]
    ) -> Dict[str, MemoryPalaceRoom]:
        """
        Build a complete memory palace from configuration.

        Args:
            room_configs: List of room configurations with:
                - name, theme, position (x, y, z), size (optional)
                - overlay_tags: List of tags to include in room
                - connected_rooms: List of room names to connect to
        """
        # Create rooms
        room_by_name: Dict[str, str] = {}  # name -> room_id

        for i, config in enumerate(room_configs):
            pos = config.get("position", {"x": i * 15, "y": 0, "z": 0})
            room = self.create_memory_palace_room(
                name=config.get("name", f"Room {i+1}"),
                theme=config.get("theme", "default"),
                center=Vector3(pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)),
                size=Vector3(
                    config.get("size", {}).get("x", 10),
                    config.get("size", {}).get("y", 4),
                    config.get("size", {}).get("z", 10)
                ) if "size" in config else None
            )
            room_by_name[config.get("name")] = room.room_id

            # Assign overlays by tag
            overlay_tags = config.get("overlay_tags", [])
            for overlay in self.overlays.values():
                if any(tag in overlay.tags for tag in overlay_tags):
                    self.assign_to_room(overlay.overlay_id, room.room_id)

        # Connect rooms
        for config in room_configs:
            room_id = room_by_name.get(config.get("name"))
            if room_id:
                for connected_name in config.get("connected_rooms", []):
                    connected_id = room_by_name.get(connected_name)
                    if connected_id:
                        self.rooms[room_id].connected_rooms.append(connected_id)

        # Layout overlays within each room
        for room in self.rooms.values():
            room_overlays = [self.overlays[oid] for oid in room.overlays if oid in self.overlays]
            if room_overlays:
                positions = self.layout_engine.compute_layout(
                    room_overlays,
                    [e for e in self.edges.values()
                     if e.source_overlay_id in room.overlays and e.target_overlay_id in room.overlays],
                    LayoutAlgorithm.CIRCULAR
                )
                # Offset to room center
                for oid, pos in positions.items():
                    if oid in self.overlays:
                        self.overlays[oid].position = Vector3(
                            room.center.x + pos.x,
                            room.center.y + pos.y,
                            room.center.z + pos.z
                        )

        return self.rooms

    # === Query Integration ===

    async def show_query_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        viewer_position: Vector3
    ):
        """Display query results as overlays arranged around the viewer."""
        # Clear existing non-pinned overlays
        to_remove = [oid for oid, o in self.overlays.items() if not o.is_pinned]
        for oid in to_remove:
            self.remove_overlay(oid)

        # Create overlays for results
        for i, result in enumerate(results[:20]):  # Limit to 20
            self.create_overlay(
                node_id=result.get("id", str(uuid.uuid4())),
                title=result.get("title", "Result"),
                summary=result.get("summary"),
                content=result.get("content"),
                tags=result.get("tags", []),
                confidence=result.get("confidence", 1.0),
                importance=result.get("importance", 0.5)
            )

        # Layout around viewer
        self.layout_engine.center = viewer_position
        self.apply_layout(LayoutAlgorithm.RADIAL)

        self._emit_event("query_results_shown", {"query": query, "count": len(results)})

    # === Animation ===

    def update(self, delta_time: float):
        """Update animations."""
        self._animation_time += delta_time

        for overlay in self.overlays.values():
            # Hover animation
            overlay.hover_offset = math.sin(self._animation_time * 2 + hash(overlay.overlay_id) % 100) * 0.02

            # Pulse animation for important/selected
            if overlay.is_selected or overlay.importance > 0.8:
                overlay.pulse_phase = (self._animation_time * 3) % (2 * math.pi)

    # === Events ===

    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def _emit_event(self, event: str, data: Dict[str, Any]):
        """Emit event to handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                handler(event, data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    # === State Export ===

    def to_state(self) -> Dict[str, Any]:
        """Export state for WebXR client."""
        return {
            "overlays": {oid: o.to_dict() for oid, o in self.overlays.items()},
            "edges": {eid: e.to_dict() for eid, e in self.edges.items()},
            "clusters": {cid: c.to_dict() for cid, c in self.clusters.items()},
            "rooms": {rid: r.to_dict() for rid, r in self.rooms.items()},
            "current_layout": self.current_layout.name,
            "overlay_count": len(self.overlays),
            "edge_count": len(self.edges)
        }


# Factory function

def create_knowledge_overlay_manager() -> KnowledgeOverlayManager:
    """Create a new knowledge overlay manager."""
    return KnowledgeOverlayManager()
