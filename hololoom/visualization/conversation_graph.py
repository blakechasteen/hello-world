"""
Conversation Graph — Renderer-Agnostic Topic/Entity Graph
===========================================================
Tracks topics, entities, and relationships as a conversation evolves.

This is transient visualization state — not durable conversation history.
Lives in-memory per room, garbage-collected after idle timeout. The same
graph feeds three renderers:

  Stage 2: to_table_spec()    → Matrix HTML (topic summary table)
  Stage 2: to_graph_spec()    → Jenny GRAPH panel (nodes + edges)
  Stage 3: to_spatial_overlay() → AR knowledge overlay (KnowledgeNodeOverlay format)

Design: immutable nodes/edges are replaced on update (append-only graph).
The graph itself is mutable (turns accumulate), but individual data points
are plain dataclasses for easy serialization.

Author: HoloLoom Team
Date: 2026-03-11
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class Trajectory(str, Enum):
    """Conversation trajectory — derived from graph shape, consumed by renderers."""
    EXPLORING = "exploring"      # Many topics, similar weights (fan-out)
    COMPARING = "comparing"      # Two topics dominate with near-equal weight
    DEEP_DIVE = "deep_dive"      # One topic dominates, weight still growing
    DECIDING = "deciding"        # Comparison + decision language detected
    WRAPPING_UP = "wrapping_up"  # Topic weights declining, conversation winding down


@dataclass
class ConversationNode:
    """A topic or entity that appeared in the conversation."""
    name: str
    node_type: str  # "topic" | "entity"
    weight: float = 1.0  # number of turns where this appeared
    first_seen_turn: int = 0
    last_seen_turn: int = 0


@dataclass
class ConversationEdge:
    """A relationship between two conversation nodes."""
    source: str  # node name
    target: str  # node name
    relation: str  # "co_occurs" | "compared_with" | "leads_to"
    weight: float = 1.0


# Maximum nodes before merging low-weight ones
MAX_NODES = 50
# Minimum turn gap between renders to avoid flooding
MIN_RENDER_GAP = 2


class ConversationGraph:
    """
    Accumulates topics/entities/relationships across conversation turns.

    Not thread-safe — called from a single async context per room.
    """

    def __init__(self, room_id: str):
        self.room_id = room_id
        self.nodes: dict[str, ConversationNode] = {}
        self.edges: list[ConversationEdge] = []
        self.turn_count: int = 0
        self.session_start: datetime = datetime.now()
        self.last_updated: datetime = datetime.now()

        # Rendering state
        self._last_fingerprint: frozenset | None = None
        self._last_rendered_turn: int = 0

    def add_turn(
        self,
        topics: list[str],
        entities: list[str],
        turn_number: int,
    ) -> None:
        """
        Add extracted topics/entities from a conversation turn.

        Topics that appeared in the same turn get co_occurs edges.
        """
        self.turn_count = turn_number
        self.last_updated = datetime.now()

        self._upsert_nodes(topics, "topic", turn_number)
        self._upsert_nodes(entities, "entity", turn_number)

        # Add co-occurrence edges between topics in the same turn
        all_items = [t.lower() for t in topics] + [e.lower() for e in entities]
        for i, a in enumerate(all_items):
            for b in all_items[i + 1:]:
                if a != b:
                    self._add_or_increment_edge(a, b, "co_occurs")

        # Enforce node cap
        if len(self.nodes) > MAX_NODES:
            self._merge_low_weight_nodes()

    def add_comparison_edge(self, topic_a: str, topic_b: str) -> None:
        """Mark two topics as being compared."""
        a, b = topic_a.lower(), topic_b.lower()
        if a in self.nodes and b in self.nodes:
            self._add_or_increment_edge(a, b, "compared_with")

    def fingerprint(self) -> frozenset:
        """
        Content fingerprint for change detection.

        Returns a frozenset of (name, weight) tuples + edge tuples.
        Fast to compare — if equal, no visual change.
        """
        node_fp = frozenset((k, n.weight) for k, n in self.nodes.items())
        edge_fp = frozenset((e.source, e.target, e.relation) for e in self.edges)
        return node_fp | edge_fp

    def has_significant_change(self) -> bool:
        """
        True if the graph changed meaningfully since last render.

        Checks both content fingerprint AND minimum turn gap.
        """
        if self._last_fingerprint is None:
            return self.turn_count >= 3  # Accumulate before first render

        if self.turn_count - self._last_rendered_turn < MIN_RENDER_GAP:
            return False

        return self.fingerprint() != self._last_fingerprint

    def mark_rendered(self) -> None:
        """Record that a visualization was generated from the current state."""
        self._last_fingerprint = self.fingerprint()
        self._last_rendered_turn = self.turn_count

    def reset(self) -> None:
        """Reset for a new conversation session."""
        self.nodes.clear()
        self.edges.clear()
        self.turn_count = 0
        self.session_start = datetime.now()
        self.last_updated = datetime.now()
        self._last_fingerprint = None
        self._last_rendered_turn = 0

    # ========================================================================
    # Output methods (renderer-agnostic → specific formats)
    # ========================================================================

    def to_table_spec(self) -> dict[str, Any]:
        """
        Produce a TABLE panel content dict.

        Columns: Topic, Mentions, Related Topics
        Sorted by weight (most discussed first).
        """
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.weight,
            reverse=True,
        )

        # Build related-topics lookup from edges
        related: dict[str, set[str]] = {}
        for edge in self.edges:
            related.setdefault(edge.source, set()).add(self.nodes[edge.target].name)
            related.setdefault(edge.target, set()).add(self.nodes[edge.source].name)

        headers = ["Topic", "Mentions", "Related"]
        rows = []
        for node in sorted_nodes[:15]:  # Top 15
            key = node.name.lower()
            rel = related.get(key, set())
            rel_str = ", ".join(sorted(rel)[:4])
            rows.append([node.name, str(int(node.weight)), rel_str])

        return {
            "conversation_graph": True,
            "headers": headers,
            "rows": rows,
        }

    def to_comparison_spec(self, topic_a: str, topic_b: str) -> dict[str, Any]:
        """
        Produce a COMPARISON panel content dict for two topics.
        """
        node_a = self.nodes.get(topic_a.lower())
        node_b = self.nodes.get(topic_b.lower())

        if not node_a or not node_b:
            return self.to_table_spec()

        return {
            "conversation_graph": True,
            "headers": [node_a.name, node_b.name],
            "items": [
                {"aspect": "First mentioned", "values": [
                    f"Turn {node_a.first_seen_turn}",
                    f"Turn {node_b.first_seen_turn}",
                ]},
                {"aspect": "Mentions", "values": [
                    str(int(node_a.weight)),
                    str(int(node_b.weight)),
                ]},
                {"aspect": "Last discussed", "values": [
                    f"Turn {node_a.last_seen_turn}",
                    f"Turn {node_b.last_seen_turn}",
                ]},
            ],
        }

    def to_graph_spec(self) -> dict[str, Any]:
        """
        Produce a GRAPH panel content dict (nodes + edges).

        For Jenny GRAPH panel type and React interactive rendering.
        """
        nodes = []
        for node in sorted(self.nodes.values(), key=lambda n: n.weight, reverse=True):
            nodes.append({
                "id": node.name.lower(),
                "label": node.name,
                "type": node.node_type,
                "weight": node.weight,
            })

        edges = []
        for edge in self.edges:
            edges.append({
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "weight": edge.weight,
            })

        return {
            "conversation_graph": True,
            "nodes": nodes[:30],
            "edges": edges,
        }

    def to_spatial_overlay(self) -> list[dict[str, Any]]:
        """
        Produce KnowledgeNodeOverlay-compatible dicts for Stage 3 AR.

        Matches the schema in spatial/knowledge_overlay.py:
        overlay_id, node_id, title, summary, tags, confidence,
        importance, recency.

        Position calculation is deferred to the spatial layout engine.
        """
        overlays = []
        max_weight = max((n.weight for n in self.nodes.values()), default=1)

        # Precompute adjacency: node_key → set of neighbor names
        adjacency: dict[str, list[str]] = {}
        for edge in self.edges:
            adjacency.setdefault(edge.source, []).append(self.nodes[edge.target].name)
            adjacency.setdefault(edge.target, []).append(self.nodes[edge.source].name)

        for node in self.nodes.values():
            key = node.name.lower()
            importance = node.weight / max_weight
            recency = 1.0 - (self.turn_count - node.last_seen_turn) / max(self.turn_count, 1)
            tags = adjacency.get(key, [])

            overlays.append({
                "overlay_id": f"conv-{self.room_id}-{node.name.lower()}",
                "node_id": node.name.lower(),
                "title": node.name,
                "summary": f"Discussed {int(node.weight)} times (turns {node.first_seen_turn}-{node.last_seen_turn})",
                "tags": tags[:5],
                "confidence": 1.0,
                "importance": round(importance, 2),
                "recency": round(max(0.0, recency), 2),
            })

        return overlays

    # ========================================================================
    # Internal
    # ========================================================================

    def _upsert_nodes(self, names: list[str], node_type: str, turn: int) -> None:
        """Add new nodes or increment weight of existing ones."""
        for name in names:
            key = name.lower()
            if key in self.nodes:
                self.nodes[key].weight += 1
                self.nodes[key].last_seen_turn = turn
            else:
                self.nodes[key] = ConversationNode(
                    name=name,
                    node_type=node_type,
                    weight=1.0,
                    first_seen_turn=turn,
                    last_seen_turn=turn,
                )

    def _add_or_increment_edge(self, source: str, target: str, relation: str) -> None:
        """Add a new edge or increment weight of an existing one."""
        # Canonical order to avoid duplicates
        a, b = min(source, target), max(source, target)

        for edge in self.edges:
            if edge.source == a and edge.target == b and edge.relation == relation:
                edge.weight += 1
                return

        self.edges.append(ConversationEdge(
            source=a, target=b, relation=relation, weight=1.0,
        ))

    def _merge_low_weight_nodes(self) -> None:
        """Remove lowest-weight nodes to stay under MAX_NODES."""
        if len(self.nodes) <= MAX_NODES:
            return

        sorted_keys = sorted(self.nodes, key=lambda k: self.nodes[k].weight)
        to_remove = set(sorted_keys[:len(self.nodes) - MAX_NODES])

        for key in to_remove:
            del self.nodes[key]

        # Remove edges referencing removed nodes
        self.edges = [
            e for e in self.edges
            if e.source not in to_remove and e.target not in to_remove
        ]
