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
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .bus import MemoryItem, MemoryQuery, MemoryResult
from .bus_config import (
    MemoryBusConfig,
    ResolutionPath,
)
from .security_screen import SecurityScreenError
from .version_log import MemoryDelta

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
    entity_type: str | None = None
    entity_ids: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    timestamp: str = ""
    access_count: int = 0
    last_accessed: str | None = None
    ttl_seconds: float | None = None  # None = never expires
    token_estimate: int = 0  # Pre-computed token cost (content + properties)

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        created = datetime.fromisoformat(self.timestamp)
        return (datetime.utcnow() - created).total_seconds() > self.ttl_seconds

    def touch(self):
        self.access_count += 1
        self.last_accessed = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "entity_type": self.entity_type,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "token_estimate": self.token_estimate,
            **self.properties,
        }

    def to_snapshot(self) -> dict[str, Any]:
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
            "token_estimate": self.token_estimate,
        }

    @classmethod
    def from_snapshot(cls, d: dict[str, Any]) -> "StoredItem":
        """Reconstruct StoredItem from snapshot dict."""
        item = cls(
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
            token_estimate=d.get("token_estimate", 0),
        )
        # Backfill for v1 snapshots that lack token_estimate
        if item.token_estimate == 0:
            item.token_estimate = max(1, len(item.content) // 4) + sum(
                len(str(v)) // 4 for v in item.properties.values()
            )
        return item


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
        config: MemoryBusConfig | None = None,
        semantic_fn: Callable | None = None,
        default_ttl: float | None = None,
        retrieval_planner: Optional["RetrievalPlanner"] = None,
        version_log: Optional["VersionLog"] = None,
        synthesis_fn: Callable | None = None,
        security_screen: Callable | None = None,
        query_cache: Optional["QueryCache"] = None,
        multi_resolution: Optional["MultiResolutionRenderer"] = None,
        repo_cache: Optional["RepoCache"] = None,
    ):
        """
        Args:
            config: Bus configuration (uses defaults if None)
            semantic_fn: Optional async fn(query_text, max_results) → List[Dict]
                        for Level 4 semantic search. If None, Level 4 returns [].
            default_ttl: Default TTL in seconds for stored items (None = no expiry)
            retrieval_planner: Optional planner to short-circuit cascade levels
                              based on query complexity (SimpleMem idea).
            version_log: Optional append-only delta log for versioned memory
                        history with rollback/diff support (Letta/GCC idea).
            synthesis_fn: Optional async fn(content, candidates) → SynthesisResult
                         for write-time dedup/merge (SimpleMem idea).
            security_screen: Optional fn(content) → bool. Returns True if safe,
                            False to block storage (Repomix idea).
        """
        self.config = config or MemoryBusConfig()
        self.semantic_fn = semantic_fn
        self.default_ttl = default_ttl
        self._retrieval_planner = retrieval_planner
        self._version_log = version_log
        self._synthesis_fn = synthesis_fn
        self._security_screen = security_screen
        self._query_cache = query_cache
        self._multi_resolution = multi_resolution
        self._repo_cache = repo_cache

        # Core stores
        self._items: dict[str, StoredItem] = {}              # id → StoredItem
        self._by_type: dict[str, list[str]] = defaultdict(list)  # memory_type → [ids]
        self._by_entity_type: dict[str, list[str]] = defaultdict(list)

        # Graph: adjacency list
        self._edges: dict[str, list[tuple[str, str]]] = defaultdict(list)  # id → [(target_id, rel_type)]

        # Entity aliases
        self._aliases: dict[str, str] = {}  # alias_lower → canonical_id

        # Audit log (in-memory ring buffer)
        self._audit: list[dict[str, Any]] = []
        self._max_audit = 1000

        # Post-store hooks: list of fn(item_id, StoredItem) callables
        self._post_store_hooks: list[Callable] = []

        # Stats
        self._query_count = 0
        self._resolution_counts: dict[str, int] = {
            ResolutionPath.EXACT.value: 0,
            ResolutionPath.STRUCTURED.value: 0,
            ResolutionPath.GRAPH.value: 0,
            ResolutionPath.SEMANTIC.value: 0,
        }
        self._total_tokens_used = 0
        self._initialized = False

        # Generation counter: monotonically increasing on every mutation.
        # Used by QueryCache/RenderCache to detect staleness.
        self._generation: int = 0

    @property
    def generation(self) -> int:
        """Monotonic counter incremented on every mutation. Cache staleness signal."""
        return self._generation

    # =========================================================================
    # Lifecycle (matches MemoryBus API)
    # =========================================================================

    async def initialize(self) -> dict[str, Any]:
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

    SNAPSHOT_VERSION = 2  # v2: added StoredItem.token_estimate

    def to_snapshot_dict(self) -> dict[str, Any]:
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

    def from_snapshot_dict(self, snapshot: dict[str, Any]) -> None:
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
        self._generation += 1

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
        # Also persist version log alongside snapshot
        if self._version_log:
            self._version_log.save(p.with_suffix(".deltas.json"))
        # Also persist repo cache alongside snapshot
        if self._repo_cache:
            self._repo_cache.save(str(p) + ".repo_cache.json")

    def load_snapshot(self, path: str) -> bool:
        """Load graph state from a JSON snapshot.

        Returns True if loaded, False if file not found.
        Raises ValueError for unsupported versions, json.JSONDecodeError for corrupt files.
        """
        p = Path(path)
        if not p.exists():
            return False

        with open(p) as f:
            snapshot = json.load(f)

        self.from_snapshot_dict(snapshot)

        logger.info(
            "Loaded graph snapshot (v%d): %d items, %d edges, %d aliases from %s",
            snapshot.get("_version", 0), len(self._items),
            sum(len(e) for e in self._edges.values()) // 2,
            len(self._aliases), path,
        )
        # Also load version log if present alongside snapshot
        if self._version_log:
            deltas_path = p.with_suffix(".deltas.json")
            self._version_log.load(deltas_path)
        # Also load repo cache if present alongside snapshot
        if self._repo_cache:
            self._repo_cache.load(str(p) + ".repo_cache.json")
        return True

    # =========================================================================
    # Query — 4-Level Cascade
    # =========================================================================

    async def query(self, q: MemoryQuery) -> MemoryResult:
        """Query through the 4-level resolution cascade."""
        self._gc_expired()

        # Query cache: return cached result if bus state unchanged
        if self._query_cache:
            cached = self._query_cache.get(q, self._generation)
            if cached is not None:
                self._query_count += 1
                return cached

        loop_id = str(uuid.uuid4())[:8]

        # Retrieval planning: determine how deep to cascade (SimpleMem idea)
        plan = self._retrieval_planner.plan(q) if self._retrieval_planner else None
        max_level = plan.max_cascade_level if plan else 4

        def _return(result):
            self._audit_log(loop_id, q, result)
            if self._query_cache:
                self._query_cache.put(q, result, self._generation)
            return result

        # Level 1: EXACT — lookup by entity ID(s)
        if q.entity_ids:
            items = self._query_exact(q.entity_ids)
            if items:
                return _return(self._build_result(items, q, ResolutionPath.EXACT))

        # Level 2: STRUCTURED — predicate filters
        if q.entity_type or q.memory_type or q.time_after or q.time_before or q.properties_match:
            items = self._query_structured(q)
            if items:
                return _return(self._build_result(items, q, ResolutionPath.STRUCTURED))

        # Level 3: GRAPH — keyword search + adjacency traversal
        if max_level >= 3 and q.intent:
            items = self._query_graph(q)
            if items:
                return _return(self._build_result(items, q, ResolutionPath.GRAPH))

        # Level 4: SEMANTIC — delegate to pluggable function
        if max_level >= 4 and (q.semantic_text or q.intent):
            items = await self._query_semantic(q)
            if items:
                return _return(self._build_result(items, q, ResolutionPath.SEMANTIC))

        # Nothing matched
        return _return(MemoryResult(items=[], query=q, resolution_path="none", token_estimate=0))

    def get_item(self, item_id: str) -> StoredItem | None:
        """Public accessor for a stored item by ID. Returns None if not found or expired."""
        item = self._items.get(item_id)
        if item is None or item.is_expired():
            return None
        return item

    def _query_exact(self, entity_ids: list[str]) -> list[dict]:
        """Level 1: Direct ID lookup."""
        items = []
        for eid in entity_ids:
            if eid in self._items and not self._items[eid].is_expired():
                stored = self._items[eid]
                stored.touch()
                items.append(stored.to_dict())
        return items

    def _query_structured(self, q: MemoryQuery) -> list[dict]:
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
            # Properties match: all specified keys must equal
            if q.properties_match:
                props = item.properties or {}
                skip = False
                for k, v in q.properties_match.items():
                    if props.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            item.touch()
            results.append(item.to_dict())

        # Sort by timestamp desc, limit
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:q.max_results]

    def _query_graph(self, q: MemoryQuery) -> list[dict]:
        """Level 3: Keyword search + graph traversal."""
        intent_lower = q.intent.lower()
        keywords = intent_lower.split()
        scored: list[tuple[float, dict]] = []

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

    async def _query_semantic(self, q: MemoryQuery) -> list[dict]:
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
        # Security screen: block sensitive content before any processing (Repomix idea)
        if self._security_screen and not self._security_screen(item.content):
            raise SecurityScreenError("content", "Memory content failed security screening")

        node_id = str(uuid.uuid4())[:12]
        now = datetime.utcnow().isoformat()

        props = item.properties or {}

        # Write-time synthesis: check for merge/link before storing (SimpleMem idea)
        _synthesis_result = None
        if self._synthesis_fn and self.semantic_fn:
            try:
                similar = await self.semantic_fn(item.content, 3)
                if similar:
                    candidates = [
                        self._items[s["id"]] for s in similar
                        if s.get("id") in self._items
                    ]
                    if candidates:
                        _synthesis_result = await self._synthesis_fn(item.content, candidates)
                        if _synthesis_result.action == "merge_into" and _synthesis_result.target_id:
                            target = self._items[_synthesis_result.target_id]
                            old_snapshot = target.to_snapshot()
                            target.content = _synthesis_result.merged_content or item.content
                            target.importance = max(target.importance, item.importance)
                            target.touch()
                            target.token_estimate = max(1, len(target.content) // 4) + sum(
                                len(str(v)) // 4 for v in target.properties.values()
                            )
                            if self._version_log:
                                self._version_log.record(MemoryDelta(
                                    delta_id=str(uuid.uuid4())[:12],
                                    timestamp=now,
                                    operation="merge",
                                    item_id=_synthesis_result.target_id,
                                    before=old_snapshot,
                                    after=target.to_snapshot(),
                                    metadata={"merged_content": item.content[:100]},
                                ))
                            return _synthesis_result.target_id
                        elif _synthesis_result.action == "supersedes" and _synthesis_result.target_id:
                            self._forget(_synthesis_result.target_id)
                            # Fall through to normal store
            except Exception as e:
                logger.warning("Synthesis failed, storing normally: %s", e)
                _synthesis_result = None

        stored = StoredItem(
            id=node_id,
            content=item.content,
            memory_type=item.memory_type,
            entity_type=props.get("type"),
            entity_ids=item.entity_ids or [],
            properties=props,
            importance=item.importance,
            timestamp=now,
            ttl_seconds=self.default_ttl,
        )
        # Pre-compute token estimate (Repomix idea: know your context cost)
        stored.token_estimate = max(1, len(item.content) // 4) + sum(
            len(str(v)) // 4 for v in props.values()
        )
        self._items[node_id] = stored
        self._generation += 1
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

        # Version log: record store delta (Letta/GCC idea)
        if self._version_log:
            self._version_log.record(MemoryDelta(
                delta_id=str(uuid.uuid4())[:12],
                timestamp=now,
                operation="store",
                item_id=node_id,
                before=None,
                after=stored.to_snapshot(),
                metadata={"memory_type": item.memory_type},
            ))

        # Post-store synthesis edges: create SIMILAR_TO links (reuses pre-store result)
        if _synthesis_result and _synthesis_result.action == "link_to" and _synthesis_result.edges:
            for src, tgt, rel in _synthesis_result.edges:
                actual_src = node_id if src == "__NEW__" else src
                await self.store_edge(actual_src, tgt, rel)

        self._audit_log(
            str(uuid.uuid4())[:8], None,
            detail={"action": "store", "node_id": node_id, "type": item.memory_type}
        )

        # Fire post-store hooks (e.g. LiveSemanticIndex.hook_store)
        for hook in self._post_store_hooks:
            try:
                hook(node_id, stored)
            except Exception as e:
                logger.warning("Post-store hook failed: %s", e)

        return node_id

    async def store_edge(self, source_id: str, target_id: str, rel_type: str = "RELATED_TO"):
        """Add a relationship between two items."""
        self._edges[source_id].append((target_id, rel_type))
        self._edges[target_id].append((source_id, rel_type))
        self._generation += 1

    # =========================================================================
    # Entity Resolution
    # =========================================================================

    async def resolve_entity(
        self,
        name_or_alias: str,
        loom_scope: str | None = None,
        max_candidates: int = 3,
    ) -> list[dict]:
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
            # Version log: record forget delta before removal
            if self._version_log:
                self._version_log.record(MemoryDelta(
                    delta_id=str(uuid.uuid4())[:12],
                    timestamp=datetime.utcnow().isoformat(),
                    operation="forget",
                    item_id=item_id,
                    before=self._items[item_id].to_snapshot(),
                    after=None,
                    metadata={},
                ))
            item = self._items.pop(item_id)
            self._generation += 1
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
            self._generation += 1
            logger.info("Decay: forgot %d low-importance items", forgotten)
        return forgotten

    async def consolidate(self, memory_type: str = "episodic", max_items: int = 10) -> str | None:
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
        self, items: list[dict], q: MemoryQuery, path: ResolutionPath
    ) -> MemoryResult:
        """Build result with token budget enforcement and multi-resolution packing."""
        budget = self.config.token_budget_absolute
        token_estimate = sum(self._item_tokens(item) for item in items)
        truncated = False

        if token_estimate > budget:
            fitted = []
            overflow = []
            running = 0
            for item in items:
                item_tokens = self._item_tokens(item)
                if running + item_tokens > budget:
                    truncated = True
                    overflow.append(item)
                else:
                    fitted.append(item)
                    running += item_tokens

            # Multi-resolution: pack overflow items at COMPACT/STUB resolution
            if self._multi_resolution and overflow:
                from .multi_resolution import RenderResolution
                keywords = (q.intent or "").lower().split()
                for item_dict in overflow:
                    item_id = item_dict.get("id", "")
                    importance = item_dict.get("importance", 0.0)
                    # Score to decide resolution
                    if keywords:
                        content = str(item_dict.get("content", "")).lower()
                        kw_score = sum(1 for kw in keywords if kw in content) / len(keywords)
                    else:
                        kw_score = 0.0
                    effective = 0.6 * importance + 0.4 * kw_score
                    if effective >= self._multi_resolution.policy.compact_threshold:
                        cost = self._multi_resolution.policy.compact_max_tokens
                        res = RenderResolution.COMPACT
                    else:
                        cost = self._multi_resolution.policy.stub_max_tokens
                        res = RenderResolution.STUB
                    if running + cost <= budget:
                        item_dict["_render_resolution"] = res.value
                        fitted.append(item_dict)
                        running += cost
                    if running >= budget:
                        break

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
    def _item_tokens(item: dict) -> int:
        """Use pre-computed token_estimate if available, fall back to heuristic."""
        cached = item.get("token_estimate", 0)
        if cached > 0:
            return cached
        return max(1, len(str(item)) // 4)

    @staticmethod
    def _estimate_tokens(item: dict) -> int:
        """Legacy heuristic token estimation. Prefer _item_tokens()."""
        return max(1, len(str(item)) // 4)

    # =========================================================================
    # Status & Pressure
    # =========================================================================

    def status(self) -> dict[str, Any]:
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

    def pressure(self) -> dict[str, Any]:
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

    async def explain(self, node_id: str) -> dict[str, Any]:
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
        highlight_ids: set[str] | None = None,
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
                'style=filled',
                f'fillcolor="{fill}"',
            ]
            if item_id in highlight_ids:
                attrs.append('penwidth=3')
                attrs.append('color=red')

            lines.append(f'  "{self._dot_escape(item_id)}" [{", ".join(attrs)}];')

        seen_edges: set[tuple[str, str, str]] = set()
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

    def to_graph_dict(self, include_content: bool = True) -> dict[str, Any]:
        """Export the graph as a plain dict (for networkx or JSON visualization).

        Args:
            include_content: Whether to include item content in nodes.

        Returns:
            Dict with "nodes", "edges", "aliases" keys.
        """
        nodes = []
        for item_id, item in self._items.items():
            node: dict[str, Any] = {
                "id": item_id,
                "memory_type": item.memory_type,
                "importance": item.importance,
                "entity_type": item.entity_type,
                "access_count": item.access_count,
            }
            if include_content:
                node["content"] = item.content
            nodes.append(node)

        seen_edges: set[tuple[str, str, str]] = set()
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
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(center_id, 0)]

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

    async def load_shards(self, shards: list[Any], memory_type: str = "factual") -> int:
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

    def _audit_log(self, loop_id: str, q: MemoryQuery | None, result: MemoryResult | None = None, detail: dict | None = None):
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

    def get_audit(self, limit: int = 20) -> list[dict]:
        return list(reversed(self._audit[-limit:]))
