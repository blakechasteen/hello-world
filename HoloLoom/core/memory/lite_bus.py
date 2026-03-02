"""
LiteMemoryBus — In-memory implementation of the Memory Bus cascade.

Same API as bus.py but requires zero external infrastructure (no Neo4j,
no Postgres). Designed as:
  1. A drop-in development/testing backend
  2. A stepping stone to the full database-backed bus
  3. A standalone option for single-session use

4-Level Cascade:
  Level 1 EXACT:      dict[entity_id] → item
  Level 2 STRUCTURED: predicate scan (type, time, memory_type)
  Level 3 GRAPH:      adjacency list + keyword search
  Level 4 SEMANTIC:   pluggable — delegates to shard embeddings or any fn

Also implements:
  - Token budgeting (same algorithm as full bus)
  - Resolution path tracking
  - Forgetting via TTL + access-count decay
  - Entity aliases (in-memory dict)
"""

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .bus_config import (
    AuditAction,
    MemoryBusConfig,
    MemoryType,
    PressureTier,
    ResolutionPath,
)
from .bus import MemoryQuery, MemoryItem, MemoryResult

logger = logging.getLogger(__name__)


# =============================================================================
# In-Memory Stores
# =============================================================================

@dataclass
class StoredItem:
    """An item in the lite memory store with metadata."""
    id: str
    content: str
    memory_type: str
    entity_type: Optional[str] = None
    entity_ids: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    timestamp: str = ""
    access_count: int = 0
    last_accessed: Optional[str] = None
    ttl_seconds: Optional[float] = None  # None = never expires

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        created = datetime.fromisoformat(self.timestamp)
        return (datetime.utcnow() - created).total_seconds() > self.ttl_seconds

    def touch(self):
        self.access_count += 1
        self.last_accessed = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "entity_type": self.entity_type,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            **self.properties,
        }

    def to_snapshot(self) -> Dict[str, Any]:
        """Full serialization for persistence (includes entity_ids, last_accessed)."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "entity_type": self.entity_type,
            "entity_ids": self.entity_ids,
            "properties": self.properties,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_snapshot(cls, d: Dict[str, Any]) -> "StoredItem":
        """Reconstruct StoredItem from snapshot dict."""
        return cls(
            id=d["id"],
            content=d["content"],
            memory_type=d["memory_type"],
            entity_type=d.get("entity_type"),
            entity_ids=d.get("entity_ids", []),
            properties=d.get("properties", {}),
            importance=d.get("importance", 0.5),
            timestamp=d.get("timestamp", ""),
            access_count=d.get("access_count", 0),
            last_accessed=d.get("last_accessed"),
            ttl_seconds=d.get("ttl_seconds"),
        )


# =============================================================================
# LiteMemoryBus
# =============================================================================

class LiteMemoryBus:
    """In-memory 4-level cascade memory bus.

    API-compatible with MemoryBus but requires no external databases.
    Plug in a semantic_fn for Level 4 to connect to shard embeddings.
    """

    def __init__(
        self,
        config: Optional[MemoryBusConfig] = None,
        semantic_fn: Optional[Callable] = None,
        default_ttl: Optional[float] = None,
    ):
        """
        Args:
            config: Bus configuration (uses defaults if None)
            semantic_fn: Optional async fn(query_text, max_results) → List[Dict]
                        for Level 4 semantic search. If None, Level 4 returns [].
            default_ttl: Default TTL in seconds for stored items (None = no expiry)
        """
        self.config = config or MemoryBusConfig()
        self.semantic_fn = semantic_fn
        self.default_ttl = default_ttl

        # Core stores
        self._items: Dict[str, StoredItem] = {}              # id → StoredItem
        self._by_type: Dict[str, List[str]] = defaultdict(list)  # memory_type → [ids]
        self._by_entity_type: Dict[str, List[str]] = defaultdict(list)

        # Graph: adjacency list
        self._edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # id → [(target_id, rel_type)]

        # Entity aliases
        self._aliases: Dict[str, str] = {}  # alias_lower → canonical_id

        # Audit log (in-memory ring buffer)
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = 1000

        # Stats
        self._query_count = 0
        self._resolution_counts: Dict[str, int] = {
            ResolutionPath.EXACT.value: 0,
            ResolutionPath.STRUCTURED.value: 0,
            ResolutionPath.GRAPH.value: 0,
            ResolutionPath.SEMANTIC.value: 0,
        }
        self._total_tokens_used = 0
        self._initialized = False

    # =========================================================================
    # Lifecycle (matches MemoryBus API)
    # =========================================================================

    async def initialize(self) -> Dict[str, Any]:
        self._initialized = True
        return {
            "neo4j": "lite (in-memory)",
            "postgres": "lite (in-memory)",
            "schema": {"mode": "lite_bus"},
        }

    async def close(self):
        self._initialized = False

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # =========================================================================
    # Snapshot Persistence
    # =========================================================================

    SNAPSHOT_VERSION = 1

    def to_snapshot_dict(self) -> Dict[str, Any]:
        """Serialize graph state to a dict (items, edges, aliases)."""
        return {
            "_version": self.SNAPSHOT_VERSION,
            "items": {
                item_id: item.to_snapshot()
                for item_id, item in self._items.items()
            },
            "edges": {
                node_id: [[target, rel] for target, rel in edge_list]
                for node_id, edge_list in self._edges.items()
            },
            "aliases": dict(self._aliases),
        }

    def from_snapshot_dict(self, snapshot: Dict[str, Any]) -> None:
        """Restore graph state from a dict. Rebuilds secondary indices."""
        version = snapshot.get("_version", 0)
        if version > self.SNAPSHOT_VERSION:
            raise ValueError(
                f"Snapshot version {version} is newer than supported "
                f"version {self.SNAPSHOT_VERSION}. Upgrade HoloLoom."
            )

        self._items.clear()
        for item_id, item_dict in snapshot.get("items", {}).items():
            self._items[item_id] = StoredItem.from_snapshot(item_dict)

        self._edges.clear()
        for node_id, edge_list in snapshot.get("edges", {}).items():
            self._edges[node_id] = [(target, rel) for target, rel in edge_list]

        self._aliases.clear()
        self._aliases.update(snapshot.get("aliases", {}))

        self._by_type.clear()
        self._by_entity_type.clear()
        for item_id, item in self._items.items():
            self._by_type[item.memory_type].append(item_id)
            if item.entity_type:
                self._by_entity_type[item.entity_type].append(item_id)

    def save_snapshot(self, path: str) -> None:
        """Save full graph state to a JSON file (atomic write)."""
        snapshot = self.to_snapshot_dict()
        snapshot["_saved_at"] = datetime.utcnow().isoformat()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(snapshot, f, indent=2)
        tmp.replace(p)
        logger.info(
            "Saved graph snapshot: %d items, %d edges, %d aliases -> %s",
            len(self._items),
            sum(len(e) for e in self._edges.values()) // 2,
            len(self._aliases), path,
        )

    def load_snapshot(self, path: str) -> bool:
        """Load graph state from a JSON snapshot.

        Returns True if loaded, False if file not found.
        Raises ValueError for unsupported versions, json.JSONDecodeError for corrupt files.
        """
        p = Path(path)
        if not p.exists():
            return False

        with open(p, "r") as f:
            snapshot = json.load(f)

        self.from_snapshot_dict(snapshot)

        logger.info(
            "Loaded graph snapshot (v%d): %d items, %d edges, %d aliases from %s",
            snapshot.get("_version", 0), len(self._items),
            sum(len(e) for e in self._edges.values()) // 2,
            len(self._aliases), path,
        )
        return True

    # =========================================================================
    # Query — 4-Level Cascade
    # =========================================================================

    async def query(self, q: MemoryQuery) -> MemoryResult:
        """Query through the 4-level resolution cascade."""
        self._gc_expired()
        loop_id = str(uuid.uuid4())[:8]

        # Level 1: EXACT — lookup by entity ID(s)
        if q.entity_ids:
            items = self._query_exact(q.entity_ids)
            if items:
                result = self._build_result(items, q, ResolutionPath.EXACT)
                self._audit_log(loop_id, q, result)
                return result

        # Level 2: STRUCTURED — predicate filters
        if q.entity_type or q.memory_type or q.time_after or q.time_before:
            items = self._query_structured(q)
            if items:
                result = self._build_result(items, q, ResolutionPath.STRUCTURED)
                self._audit_log(loop_id, q, result)
                return result

        # Level 3: GRAPH — keyword search + adjacency traversal
        if q.intent:
            items = self._query_graph(q)
            if items:
                result = self._build_result(items, q, ResolutionPath.GRAPH)
                self._audit_log(loop_id, q, result)
                return result

        # Level 4: SEMANTIC — delegate to pluggable function
        if q.semantic_text or q.intent:
            items = await self._query_semantic(q)
            if items:
                result = self._build_result(items, q, ResolutionPath.SEMANTIC)
                self._audit_log(loop_id, q, result)
                return result

        # Nothing matched
        result = MemoryResult(items=[], query=q, resolution_path="none", token_estimate=0)
        self._audit_log(loop_id, q, result)
        return result

    def _query_exact(self, entity_ids: List[str]) -> List[Dict]:
        """Level 1: Direct ID lookup."""
        items = []
        for eid in entity_ids:
            if eid in self._items and not self._items[eid].is_expired():
                stored = self._items[eid]
                stored.touch()
                items.append(stored.to_dict())
        return items

    def _query_structured(self, q: MemoryQuery) -> List[Dict]:
        """Level 2: Predicate scan over stored items."""
        candidates = list(self._items.values())
        results = []

        for item in candidates:
            if item.is_expired():
                continue
            if q.entity_type and item.entity_type != q.entity_type:
                continue
            if q.memory_type and item.memory_type != q.memory_type:
                continue
            if q.time_after and item.timestamp < q.time_after:
                continue
            if q.time_before and item.timestamp > q.time_before:
                continue
            if item.importance < q.min_importance:
                continue
            item.touch()
            results.append(item.to_dict())

        # Sort by timestamp desc, limit
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:q.max_results]

    def _query_graph(self, q: MemoryQuery) -> List[Dict]:
        """Level 3: Keyword search + graph traversal."""
        intent_lower = q.intent.lower()
        keywords = intent_lower.split()
        scored: List[Tuple[float, Dict]] = []

        for item in self._items.values():
            if item.is_expired():
                continue
            content_lower = item.content.lower()
            # Score by keyword overlap
            hits = sum(1 for kw in keywords if kw in content_lower)
            if hits > 0:
                score = hits / len(keywords)
                item.touch()
                d = item.to_dict()
                d["_search_score"] = score
                scored.append((score, d))

        # Also traverse edges from matched items (1-hop)
        matched_ids = {s[1]["id"] for s in scored}
        for mid in list(matched_ids):
            for target_id, rel_type in self._edges.get(mid, []):
                if target_id in self._items and target_id not in matched_ids:
                    neighbor = self._items[target_id]
                    if not neighbor.is_expired():
                        neighbor.touch()
                        d = neighbor.to_dict()
                        d["_search_score"] = 0.3  # traversal bonus
                        d["_traversal_from"] = mid
                        d["_relationship"] = rel_type
                        scored.append((0.3, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:q.max_results]]

    async def _query_semantic(self, q: MemoryQuery) -> List[Dict]:
        """Level 4: Delegate to pluggable semantic function."""
        if not self.semantic_fn:
            return []
        try:
            text = q.semantic_text or q.intent
            return await self.semantic_fn(text, q.max_results)
        except Exception as e:
            logger.warning("Semantic search failed: %s", e)
            return []

    # =========================================================================
    # Store
    # =========================================================================

    async def store(self, item: MemoryItem) -> str:
        """Store a memory item. Returns the item ID."""
        node_id = str(uuid.uuid4())[:12]
        now = datetime.utcnow().isoformat()

        stored = StoredItem(
            id=node_id,
            content=item.content,
            memory_type=item.memory_type,
            entity_type=(item.properties or {}).get("type"),
            entity_ids=item.entity_ids or [],
            properties=item.properties or {},
            importance=item.importance,
            timestamp=now,
            ttl_seconds=self.default_ttl,
        )
        self._items[node_id] = stored
        self._by_type[item.memory_type].append(node_id)
        if stored.entity_type:
            self._by_entity_type[stored.entity_type].append(node_id)

        # Connect to referenced entities
        if item.entity_ids:
            for eid in item.entity_ids:
                self._edges[node_id].append((eid, "ABOUT"))
                self._edges[eid].append((node_id, "ABOUT"))

        # Auto-alias: register content as alias for entity types
        if item.memory_type == "entity":
            self._aliases[item.content.lower()] = node_id

        self._audit_log(
            str(uuid.uuid4())[:8], None,
            detail={"action": "store", "node_id": node_id, "type": item.memory_type}
        )
        return node_id

    async def store_edge(self, source_id: str, target_id: str, rel_type: str = "RELATED_TO"):
        """Add a relationship between two items."""
        self._edges[source_id].append((target_id, rel_type))
        self._edges[target_id].append((source_id, rel_type))

    # =========================================================================
    # Entity Resolution
    # =========================================================================

    async def resolve_entity(
        self,
        name_or_alias: str,
        loom_scope: Optional[str] = None,
        max_candidates: int = 3,
    ) -> List[Dict]:
        """Resolve a name/alias to entity IDs."""
        candidates = []
        key = name_or_alias.lower()

        # Check alias table
        if key in self._aliases:
            eid = self._aliases[key]
            if eid in self._items:
                candidates.append({
                    "entity_id": eid,
                    "matched_alias": name_or_alias,
                    "confidence": 0.95,
                    "source": "alias_table",
                })

        # Fuzzy: check if alias is substring of any registered alias
        if not candidates:
            for alias, eid in self._aliases.items():
                if key in alias or alias in key:
                    if eid in self._items:
                        candidates.append({
                            "entity_id": eid,
                            "matched_alias": alias,
                            "confidence": 0.6,
                            "source": "alias_fuzzy",
                        })
                        if len(candidates) >= max_candidates:
                            break

        return candidates

    async def add_alias(self, alias: str, entity_id: str):
        """Register an alias for an entity."""
        self._aliases[alias.lower()] = entity_id

    # =========================================================================
    # Forgetting (Invariant 7)
    # =========================================================================

    def _gc_expired(self):
        """Remove expired items (called on every query)."""
        expired = [k for k, v in self._items.items() if v.is_expired()]
        for k in expired:
            self._forget(k)
        if expired:
            logger.info("GC: forgot %d expired items", len(expired))

    def _forget(self, item_id: str):
        """Remove a single item and its edges."""
        if item_id in self._items:
            item = self._items.pop(item_id)
            # Clean up type indices
            if item_id in self._by_type.get(item.memory_type, []):
                self._by_type[item.memory_type].remove(item_id)
            if item.entity_type and item_id in self._by_entity_type.get(item.entity_type, []):
                self._by_entity_type[item.entity_type].remove(item_id)
            # Clean up edges
            self._edges.pop(item_id, None)
            for edges in self._edges.values():
                edges[:] = [(t, r) for t, r in edges if t != item_id]
            # Clean up aliases
            self._aliases = {a: e for a, e in self._aliases.items() if e != item_id}

    async def decay(self, max_age_seconds: float = 86400, min_importance: float = 0.1):
        """Decay importance of items that haven't been accessed recently.

        Items below min_importance after decay get forgotten.
        Returns count of forgotten items.
        """
        now = datetime.utcnow()
        forgotten = 0
        for item_id in list(self._items.keys()):
            item = self._items[item_id]
            created = datetime.fromisoformat(item.timestamp)
            age = (now - created).total_seconds()

            if age > max_age_seconds:
                # Decay based on access frequency
                # More accessed items decay slower
                access_shield = min(1.0, item.access_count / 10)
                decay_rate = 0.1 * (1.0 - access_shield)
                item.importance = max(0.0, item.importance - decay_rate)

                if item.importance < min_importance:
                    self._forget(item_id)
                    forgotten += 1

        if forgotten:
            logger.info("Decay: forgot %d low-importance items", forgotten)
        return forgotten

    async def consolidate(self, memory_type: str = "episodic", max_items: int = 10) -> Optional[str]:
        """Consolidate multiple items of the same type into a summary.

        Returns the new summary item ID, or None if nothing to consolidate.
        """
        candidates = []
        for item_id in self._by_type.get(memory_type, []):
            if item_id in self._items:
                candidates.append(self._items[item_id])

        if len(candidates) <= max_items:
            return None

        # Sort by importance (ascending) — consolidate least important
        candidates.sort(key=lambda x: x.importance)
        to_merge = candidates[:len(candidates) - max_items]

        # Create summary
        merged_content = " | ".join(item.content[:100] for item in to_merge)
        avg_importance = sum(item.importance for item in to_merge) / len(to_merge)

        summary = MemoryItem(
            content=f"[consolidated {len(to_merge)} {memory_type}] {merged_content}",
            memory_type=memory_type,
            importance=avg_importance * 1.2,  # Slight boost for consolidated items
        )
        summary_id = await self.store(summary)

        # Remove originals
        for item in to_merge:
            self._forget(item.id)

        logger.info("Consolidated %d %s items → %s", len(to_merge), memory_type, summary_id)
        return summary_id

    # =========================================================================
    # Token Budgeting (Invariant 3)
    # =========================================================================

    def _build_result(
        self, items: List[Dict], q: MemoryQuery, path: ResolutionPath
    ) -> MemoryResult:
        """Build result with token budget enforcement."""
        token_estimate = sum(self._estimate_tokens(item) for item in items)
        truncated = False

        if token_estimate > self.config.token_budget_absolute:
            fitted = []
            running = 0
            for item in items:
                item_tokens = self._estimate_tokens(item)
                if running + item_tokens > self.config.token_budget_absolute:
                    truncated = True
                    break
                fitted.append(item)
                running += item_tokens
            items = fitted
            token_estimate = running

        self._query_count += 1
        self._resolution_counts[path.value] = self._resolution_counts.get(path.value, 0) + 1
        self._total_tokens_used += token_estimate

        return MemoryResult(
            items=items,
            query=q,
            resolution_path=path.value,
            token_estimate=token_estimate,
            truncated=truncated,
        )

    @staticmethod
    def _estimate_tokens(item: Dict) -> int:
        return max(1, len(str(item)) // 4)

    # =========================================================================
    # Status & Pressure
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        total_queries = max(1, self._query_count)
        semantic_pct = self._resolution_counts.get(ResolutionPath.SEMANTIC.value, 0) / total_queries
        return {
            "initialized": self._initialized,
            "mode": "lite (in-memory)",
            "item_count": len(self._items),
            "edge_count": sum(len(e) for e in self._edges.values()) // 2,
            "alias_count": len(self._aliases),
            "query_count": self._query_count,
            "resolution_distribution": dict(self._resolution_counts),
            "total_tokens_used": self._total_tokens_used,
            "semantic_fallback_pct": round(semantic_pct, 3),
        }

    def pressure(self) -> Dict[str, Any]:
        utilization = self._total_tokens_used / max(1, self.config.token_budget_absolute)
        tier = self.config.pressure_tier_for(min(1.0, utilization))
        return {
            "tier": tier.value,
            "tier_name": tier.name,
            "context_tokens_used": self._total_tokens_used,
            "token_budget": self.config.token_budget_absolute,
            "utilization_pct": round(utilization, 3),
            "item_count": len(self._items),
        }

    # =========================================================================
    # Explain
    # =========================================================================

    async def explain(self, node_id: str) -> Dict[str, Any]:
        """Provenance chain for an item."""
        if node_id not in self._items:
            return {"node_id": node_id, "error": "not found"}

        item = self._items[node_id]
        edges = self._edges.get(node_id, [])

        return {
            "node_id": node_id,
            "content": item.content,
            "memory_type": item.memory_type,
            "importance": item.importance,
            "access_count": item.access_count,
            "timestamp": item.timestamp,
            "connected": [
                {"target": tid, "relationship": rel} for tid, rel in edges
            ],
            "aliases": [a for a, eid in self._aliases.items() if eid == node_id],
        }

    # =========================================================================
    # Graph Visualization Export
    # =========================================================================

    @staticmethod
    def _dot_escape(s: str) -> str:
        """Escape special characters for DOT label strings."""
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def to_dot(
        self,
        max_label_len: int = 40,
        highlight_ids: Optional[Set[str]] = None,
    ) -> str:
        """Export the graph as a Graphviz DOT string.

        Args:
            max_label_len: Truncate node labels to this length.
            highlight_ids: Set of node IDs to highlight with bold red border.

        Returns:
            Valid Graphviz DOT string.
        """
        highlight_ids = highlight_ids or set()

        color_map = {
            "factual": "lightblue",
            "entity": "gold",
            "chunk": "lightgreen",
            "case": "lightsalmon",
        }
        shape_map = {
            "entity": "diamond",
            "chunk": "hexagon",
            "case": "trapezium",
        }

        lines = [
            'digraph LiteMemoryBus {',
            '  rankdir=LR;',
            '  fontname="Helvetica";',
            '  node [fontname="Helvetica"];',
            '  edge [fontname="Helvetica"];',
        ]

        for item_id, item in self._items.items():
            label = item.content
            if len(label) > max_label_len:
                label = label[:max_label_len] + "..."
            label = self._dot_escape(label)

            fill = color_map.get(item.memory_type, "white")
            shape = shape_map.get(item.memory_type, "ellipse")

            attrs = [
                f'label="{label}"',
                f'shape={shape}',
                f'style=filled',
                f'fillcolor="{fill}"',
            ]
            if item_id in highlight_ids:
                attrs.append('penwidth=3')
                attrs.append('color=red')

            lines.append(f'  "{self._dot_escape(item_id)}" [{", ".join(attrs)}];')

        seen_edges: Set[Tuple[str, str, str]] = set()
        for source_id, edge_list in self._edges.items():
            for target_id, rel_type in edge_list:
                pair = tuple(sorted([source_id, target_id])) + (rel_type,)
                if pair in seen_edges:
                    continue
                seen_edges.add(pair)
                rel_label = self._dot_escape(rel_type)
                lines.append(
                    f'  "{self._dot_escape(source_id)}" -> '
                    f'"{self._dot_escape(target_id)}" '
                    f'[label="{rel_label}"];'
                )

        lines.append("}")
        return "\n".join(lines)

    def to_graph_dict(self, include_content: bool = True) -> Dict[str, Any]:
        """Export the graph as a plain dict (for networkx or JSON visualization).

        Args:
            include_content: Whether to include item content in nodes.

        Returns:
            Dict with "nodes", "edges", "aliases" keys.
        """
        nodes = []
        for item_id, item in self._items.items():
            node: Dict[str, Any] = {
                "id": item_id,
                "memory_type": item.memory_type,
                "importance": item.importance,
                "entity_type": item.entity_type,
                "access_count": item.access_count,
            }
            if include_content:
                node["content"] = item.content
            nodes.append(node)

        seen_edges: Set[Tuple[str, str, str]] = set()
        edges = []
        for source_id, edge_list in self._edges.items():
            for target_id, rel_type in edge_list:
                pair = tuple(sorted([source_id, target_id])) + (rel_type,)
                if pair in seen_edges:
                    continue
                seen_edges.add(pair)
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "type": rel_type,
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "aliases": dict(self._aliases),
        }

    def subgraph(
        self,
        center_id: str,
        depth: int = 2,
        max_nodes: int = 50,
    ) -> "LiteMemoryBus":
        """Extract a neighborhood subgraph via BFS.

        Args:
            center_id: Starting node for BFS.
            depth: Maximum hops from center.
            max_nodes: Stop BFS after collecting this many nodes.

        Returns:
            A new LiteMemoryBus containing only the neighborhood.
        """
        visited: Set[str] = set()
        queue: List[Tuple[str, int]] = [(center_id, 0)]

        while queue:
            node_id, d = queue.pop(0)
            if node_id in visited:
                continue
            if node_id not in self._items:
                continue
            if len(visited) >= max_nodes:
                break
            visited.add(node_id)
            if d < depth:
                for target_id, _rel in self._edges.get(node_id, []):
                    if target_id not in visited:
                        queue.append((target_id, d + 1))

        sub = LiteMemoryBus(config=self.config)
        for nid in visited:
            if nid in self._items:
                item = self._items[nid]
                sub._items[nid] = StoredItem(
                    id=item.id,
                    content=item.content,
                    memory_type=item.memory_type,
                    entity_type=item.entity_type,
                    entity_ids=list(item.entity_ids),
                    properties=dict(item.properties),
                    importance=item.importance,
                    timestamp=item.timestamp,
                    access_count=item.access_count,
                    last_accessed=item.last_accessed,
                    ttl_seconds=item.ttl_seconds,
                )
                sub._by_type[item.memory_type].append(nid)
                if item.entity_type:
                    sub._by_entity_type[item.entity_type].append(nid)

        for nid in visited:
            for target_id, rel_type in self._edges.get(nid, []):
                if target_id in visited:
                    sub._edges[nid].append((target_id, rel_type))

        for alias, eid in self._aliases.items():
            if eid in visited:
                sub._aliases[alias] = eid

        return sub

    # =========================================================================
    # Bulk Load (for shard migration)
    # =========================================================================

    async def load_shards(self, shards: List[Any], memory_type: str = "factual") -> int:
        """Bulk-load MemoryShard objects into the bus.

        Bridges the existing shard-based system into the bus architecture.
        Returns count of items loaded.
        """
        count = 0
        for shard in shards:
            text = shard.text if hasattr(shard, 'text') else str(shard)
            shard_id = shard.id if hasattr(shard, 'id') else str(uuid.uuid4())[:12]
            metadata = shard.metadata if hasattr(shard, 'metadata') else {}

            item = MemoryItem(
                content=text,
                memory_type=memory_type,
                properties=metadata,
                importance=metadata.get("importance", 0.5),
            )
            # Use shard's original ID if possible
            stored = StoredItem(
                id=shard_id,
                content=text,
                memory_type=memory_type,
                properties=metadata,
                importance=metadata.get("importance", 0.5),
                timestamp=datetime.utcnow().isoformat(),
                ttl_seconds=self.default_ttl,
            )
            self._items[shard_id] = stored
            self._by_type[memory_type].append(shard_id)
            count += 1

        logger.info("Loaded %d shards into LiteMemoryBus", count)
        return count

    # =========================================================================
    # Audit (in-memory ring buffer)
    # =========================================================================

    def _audit_log(self, loop_id: str, q: Optional[MemoryQuery], result: Optional[MemoryResult] = None, detail: Optional[Dict] = None):
        entry = {
            "loop_id": loop_id,
            "timestamp": datetime.utcnow().isoformat(),
            "intent": q.intent if q else None,
            "resolution_path": result.resolution_path if result else None,
            "result_count": len(result.items) if result else 0,
            "tokens": result.token_estimate if result else 0,
        }
        if detail:
            entry.update(detail)
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit = self._audit[-self._max_audit:]

    def get_audit(self, limit: int = 20) -> List[Dict]:
        return list(reversed(self._audit[-limit:]))
