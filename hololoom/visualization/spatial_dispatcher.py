"""
Spatial Scene Dispatcher — ConversationGraph → Spatial Overlays
================================================================
Bridges the conversation graph (Stage 2) to the spatial overlay system
(Stage 3). Converts topics/entities/relationships into positioned 3D
overlays using the KnowledgeOverlayManager's layout engine.

This is the only genuinely new logic for Stage 3. Everything else is
wiring between existing systems.

Design: clear-and-rebuild on each significant change. The graph is small
(max 50 nodes) and the layout engine is fast. Incremental diffing would
add complexity for no measurable gain.

Author: HoloLoom Team
Date: 2026-03-11
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

from hololoom.visualization.conversation_graph import Trajectory

_RELATION_MAP = {
    "co_occurs": "RELATED",
    "compared_with": "COMPARED",
    "leads_to": "LEADS_TO",
}

# Populated on first use (avoids heavy import at module level)
_TRAJECTORY_LAYOUT: dict[Trajectory, Any] | None = None


def _map_relation(relation: str) -> str:
    return _RELATION_MAP.get(relation, "RELATED")


def _trajectory_to_layout(trajectory: Trajectory):
    """Map conversation trajectory to spatial layout algorithm."""
    global _TRAJECTORY_LAYOUT
    if _TRAJECTORY_LAYOUT is None:
        from hololoom.spatial.knowledge_overlay import LayoutAlgorithm
        _TRAJECTORY_LAYOUT = {
            Trajectory.EXPLORING: LayoutAlgorithm.FORCE_DIRECTED,
            Trajectory.COMPARING: LayoutAlgorithm.GRID,
            Trajectory.DEEP_DIVE: LayoutAlgorithm.RADIAL,
            Trajectory.DECIDING: LayoutAlgorithm.HIERARCHICAL,
            Trajectory.WRAPPING_UP: LayoutAlgorithm.FORCE_DIRECTED,
        }
    return _TRAJECTORY_LAYOUT.get(trajectory, _TRAJECTORY_LAYOUT[Trajectory.EXPLORING])


class SpatialSceneDispatcher:
    """
    Converts ConversationGraph state into a positioned spatial scene.

    On each call to update_from_graph(), it:
    1. Checks fingerprint — skips if graph hasn't changed
    2. Extracts overlay-compatible dicts from the graph
    3. Remaps fields for KnowledgeOverlayManager
    4. Clears and rebuilds overlays with trajectory-aware layout
    5. Returns the full scene state dict for WebSocket broadcast

    Not thread-safe — called from a single async context per room.
    """

    def __init__(self, room_id: str):
        self.room_id = room_id
        self._last_fingerprint: frozenset | None = None
        self._overlay_manager = None  # Lazy init

    @property
    def overlay_manager(self):
        """Lazy-init the overlay manager (avoids import at module level)."""
        if self._overlay_manager is None:
            from hololoom.spatial.knowledge_overlay import KnowledgeOverlayManager
            self._overlay_manager = KnowledgeOverlayManager()
        return self._overlay_manager

    def update_from_graph(
        self,
        graph: 'ConversationGraph',
        trajectory: Trajectory,
    ) -> dict[str, Any] | None:
        """
        Convert conversation graph to spatial scene.

        Returns scene state dict if changed, None if unchanged.
        """
        fp = graph.fingerprint()
        if fp == self._last_fingerprint:
            return None

        try:
            return self._rebuild_scene(graph, trajectory, fp)
        except Exception as e:
            logger.warning("Spatial scene rebuild failed: %s", e)
            return None

    def get_current_state(self) -> dict[str, Any] | None:
        """Return the current scene state without rebuilding."""
        if self._overlay_manager is None:
            return None
        return self.overlay_manager.to_state()

    def _rebuild_scene(
        self,
        graph: 'ConversationGraph',
        trajectory: Trajectory,
        fingerprint: frozenset,
    ) -> dict[str, Any]:
        """Clear and rebuild the spatial scene from the conversation graph."""
        mgr = self.overlay_manager

        overlay_dicts = graph.to_spatial_overlay()
        knowledge_nodes = [
            {
                "id": d["node_id"],
                "title": d["title"],
                "summary": d.get("summary", ""),
                "tags": d.get("tags", []),
                "confidence": d.get("confidence", 1.0),
                "importance": d.get("importance", 0.5),
            }
            for d in overlay_dicts
        ]
        relationships = [
            {
                "source_id": e.source,
                "target_id": e.target,
                "relationship_type": _map_relation(e.relation),
            }
            for e in graph.edges
        ]

        mgr.overlays.clear()
        mgr.edges.clear()
        mgr.clusters.clear()
        mgr.create_overlays_from_knowledge(knowledge_nodes, relationships)

        target_layout = _trajectory_to_layout(trajectory)
        if target_layout != mgr.current_layout:
            mgr.current_layout = target_layout
            mgr.apply_layout(target_layout)

        self._last_fingerprint = fingerprint
        state = mgr.to_state()
        state["room_id"] = self.room_id
        state["trajectory"] = trajectory
        state["turn_count"] = graph.turn_count
        state["timestamp"] = time.time()

        logger.debug(
            "Spatial scene rebuilt: room=%s trajectory=%s overlays=%d edges=%d",
            self.room_id, trajectory, len(mgr.overlays), len(mgr.edges),
        )
        return state
