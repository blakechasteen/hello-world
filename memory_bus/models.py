"""
Data models for the Memory Bus MVP — 2026-02-09

All types shared across adapters, router, formatter, and MCP tools.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ResolutionPath(str, Enum):
    """Query routing cascade (structured-first)."""
    EXACT = "exact"
    STRUCTURED = "structured"
    GRAPH = "graph"
    SEMANTIC = "semantic"  # STUB in MVP


class DetailLevel(str, Enum):
    """Progressive detail for token-budgeted formatting."""
    TITLE = "title"
    SUMMARY = "summary"
    COMPACT = "compact"
    FULL = "full"


class PressureTier(str, Enum):
    """Ephemeral memory pressure tiers."""
    TIER0 = "tier0"  # <50 %
    TIER1 = "tier1"  # 50-75 %
    TIER2 = "tier2"  # 75-90 %
    TIER3 = "tier3"  # >90 %


class EntityType(str, Enum):
    PERSON = "person"
    PLACE = "place"
    OBJECT = "object"
    CONCEPT = "concept"
    ORGANISATION = "organisation"
    EVENT = "event"
    OTHER = "other"


class EpisodeType(str, Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    DECISION = "decision"
    PLAN = "plan"
    INFERENCE = "inference"
    OTHER = "other"


class PlanState(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class AuditAction(str, Enum):
    QUERY = "query"
    STORE = "store"
    RESOLVE = "resolve"
    EXPLAIN = "explain"
    PROMOTE = "promote"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Neo4j node models
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """Corresponds to Neo4j :Entity label."""
    id: str = Field(default_factory=lambda: f"ent_{uuid.uuid4().hex[:12]}")
    type: EntityType = EntityType.OTHER
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    loom_origin: Optional[str] = None
    properties: dict[str, Any] = Field(default_factory=dict)


class Episode(BaseModel):
    """Corresponds to Neo4j :Episode label."""
    id: str = Field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:12]}")
    type: EpisodeType = EpisodeType.OBSERVATION
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    summary: str
    content: Optional[str] = None
    significance: float = Field(default=0.5, ge=0.0, le=1.0)
    loom_origin: Optional[str] = None
    entity_ids: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Postgres table models
# ---------------------------------------------------------------------------

class AuditRow(BaseModel):
    """Single row in memory_audit table."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    loop_id: Optional[str] = None
    loom_id: Optional[str] = None
    action: AuditAction
    resolution_path: Optional[ResolutionPath] = None
    tokens_used: int = 0
    pressure_tier: PressureTier = PressureTier.TIER0
    details: dict[str, Any] = Field(default_factory=dict)


class PlanRow(BaseModel):
    """Row in plans table."""
    id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:12]}")
    neo4j_node_id: Optional[str] = None
    name: str
    state: PlanState = PlanState.DRAFT
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    loom_id: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    details: dict[str, Any] = Field(default_factory=dict)


class EntityAlias(BaseModel):
    """Row in entity_aliases table."""
    alias: str
    entity_id: str
    loom_id: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ConfigRow(BaseModel):
    """Row in configs table."""
    key: str
    value: dict[str, Any] = Field(default_factory=dict)
    loom_id: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Query / Result models
# ---------------------------------------------------------------------------

class MemoryQuery(BaseModel):
    """Inbound query to the memory bus."""
    text: str
    entity_ids: list[str] = Field(default_factory=list)
    entity_names: list[str] = Field(default_factory=list)
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    episode_types: list[EpisodeType] = Field(default_factory=list)
    max_results: int = Field(default=20, ge=1, le=200)
    detail_level: DetailLevel = DetailLevel.COMPACT
    token_budget: Optional[int] = None  # per-query cap; None = use default
    loom_id: Optional[str] = None
    loop_id: Optional[str] = None


class MemoryItem(BaseModel):
    """Single result item from memory retrieval."""
    id: str
    kind: str  # "entity" | "episode"
    title: str
    summary: Optional[str] = None
    content: Optional[str] = None
    timestamp: Optional[datetime] = None
    importance: float = 0.5
    resolution_path: ResolutionPath = ResolutionPath.STRUCTURED
    properties: dict[str, Any] = Field(default_factory=dict)
    related_entity_ids: list[str] = Field(default_factory=list)


class MemoryResult(BaseModel):
    """Result bundle returned to the caller."""
    items: list[MemoryItem] = Field(default_factory=list)
    total_tokens_used: int = 0
    resolution_path: ResolutionPath = ResolutionPath.STRUCTURED
    detail_level: DetailLevel = DetailLevel.COMPACT
    pressure_tier: PressureTier = PressureTier.TIER0
    query_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    formatted_context: str = ""
    warnings: list[str] = Field(default_factory=list)
    # Distinguishes "stub can't search" from "real search found nothing"
    semantic_stub_reached: bool = False
    no_results_reason: Optional[str] = None


class ResolveCandidate(BaseModel):
    """Single candidate from entity resolution."""
    entity_id: str
    name: str
    type: EntityType = EntityType.OTHER
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "alias"  # "alias" | "neo4j_name_search"


class ResolveResult(BaseModel):
    """Result of entity resolution."""
    query_name: str
    candidates: list[ResolveCandidate] = Field(default_factory=list)
    best_match: Optional[ResolveCandidate] = None


class ExplainResult(BaseModel):
    """Provenance chain for a memory item."""
    item_id: str
    derived_from: list[str] = Field(default_factory=list)
    loop_id: Optional[str] = None
    confidence: float = 0.0
    status: str = "confirmed"  # "confirmed" | "inferred"
    contradictions: list[str] = Field(default_factory=list)
    audit_trail: list[AuditRow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pressure signals
# ---------------------------------------------------------------------------

class PressureSignals(BaseModel):
    """Tracked signals for pressure tier computation."""
    context_tokens_used: int = 0
    context_window_size: int = 128_000  # default model context
    context_utilization_pct: float = 0.0
    retrieval_result_count: int = 0
    retrieval_latency_ms: float = 0.0
    promotion_queue_depth: int = 0
    decay_floor_population: int = 0  # stub

    @property
    def tier(self) -> PressureTier:
        pct = self.context_utilization_pct
        if pct >= 90.0:
            return PressureTier.TIER3
        if pct >= 75.0:
            return PressureTier.TIER2
        if pct >= 50.0:
            return PressureTier.TIER1
        return PressureTier.TIER0


class PressureStatus(BaseModel):
    """Published via memory://pressure resource."""
    tier: PressureTier
    signals: PressureSignals
    policy: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Store request
# ---------------------------------------------------------------------------

class MemoryStoreRequest(BaseModel):
    """Inbound store/promotion request."""
    entities: list[Entity] = Field(default_factory=list)
    episodes: list[Episode] = Field(default_factory=list)
    aliases: list[EntityAlias] = Field(default_factory=list)
    plans: list[PlanRow] = Field(default_factory=list)
    loom_id: Optional[str] = None
    loop_id: Optional[str] = None
    immediate: bool = True  # write-through by default
