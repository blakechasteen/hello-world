"""
Pydantic Models for HoloLoom API
==================================

Request and response models with validation and OpenAPI schema generation.

Author: HoloLoom Team
Date: 2025-11-18
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class MemoryScope(str, Enum):
    """Memory scope (matches lifecycle_manager)."""
    SESSION = "SESSION"
    AGENT = "AGENT"
    USER = "USER"
    GLOBAL = "GLOBAL"


class ConsolidationStrategy(str, Enum):
    """Consolidation strategy."""
    ENTITY_EXTRACTION = "entity_extraction"
    FACT_EXTRACTION = "fact_extraction"
    SUMMARIZATION = "summarization"
    DEDUPLICATION = "deduplication"


class UnderstandingState(str, Enum):
    """Understanding states for temporal evolution."""
    UNKNOWN = "unknown"
    INTRODUCED = "introduced"
    LEARNING = "learning"
    FAMILIAR = "familiar"
    MASTERY = "mastery"
    DORMANT = "dormant"
    FORGOTTEN = "forgotten"


class ExplorationSuggestionType(str, Enum):
    """Types of exploration suggestions."""
    GAP = "gap"
    CONTRADICTION = "contradiction"
    RELATED_CONCEPT = "related_concept"
    TRENDING = "trending"
    DEEP_DIVE = "deep_dive"


# ============================================================================
# Base Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    systems: Dict[str, str] = Field(
        default_factory=lambda: {
            "consolidation": "ok",
            "semantic_transition": "ok",
            "temporal_evolution": "ok",
            "curiosity": "ok",
            "graph_reasoning": "ok"
        }
    )
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Memory Operations
# ============================================================================

class ExperienceRequest(BaseModel):
    """Request to store new episodic memory."""
    content: str = Field(..., description="Memory content", min_length=1)
    scope: MemoryScope = Field(MemoryScope.SESSION, description="Memory scope")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "content": "Thompson Sampling balances exploration and exploitation",
                "scope": "SESSION",
                "metadata": {"source": "user_query"}
            }
        }


class ExperienceResponse(BaseModel):
    """Response from experience operation."""
    status: str = "success"
    memory_id: str
    scope: MemoryScope
    timestamp: datetime = Field(default_factory=datetime.now)


class RecallRequest(BaseModel):
    """Request to retrieve memories."""
    query: str = Field(..., description="Query text", min_length=1)
    max_results: int = Field(10, description="Maximum results to return", ge=1, le=100)
    scope: Optional[MemoryScope] = Field(None, description="Limit to specific scope")
    include_metadata: bool = Field(False, description="Include full metadata")

    class Config:
        schema_extra = {
            "example": {
                "query": "What is Thompson Sampling?",
                "max_results": 10,
                "scope": None,
                "include_metadata": False
            }
        }


class MemoryResult(BaseModel):
    """Single memory result."""
    memory_id: str
    content: str
    scope: MemoryScope
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class RecallResponse(BaseModel):
    """Response from recall operation."""
    query: str
    memories: List[MemoryResult]
    count: int
    total_available: int
    latency_ms: float


# ============================================================================
# Consolidation
# ============================================================================

class ConsolidationTriggerRequest(BaseModel):
    """Request to trigger consolidation."""
    strategy: Optional[ConsolidationStrategy] = Field(
        None,
        description="Consolidation strategy (defaults to all)"
    )
    max_episodes: int = Field(100, description="Max episodes to consolidate", ge=1)
    prune_episodes: bool = Field(False, description="Delete consolidated episodes")

    class Config:
        schema_extra = {
            "example": {
                "strategy": "fact_extraction",
                "max_episodes": 100,
                "prune_episodes": False
            }
        }


class ConsolidationResult(BaseModel):
    """Result from consolidation operation."""
    strategy: ConsolidationStrategy
    input_episodes: int
    output_facts: int
    facts_stored: List[str]
    episodes_pruned: int
    consolidation_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConsolidationResponse(BaseModel):
    """Response from consolidation trigger."""
    status: str = "queued"
    job_id: str
    estimated_duration_ms: float


class ConsolidationStatsResponse(BaseModel):
    """Consolidation statistics."""
    total_consolidations: int
    total_episodes_processed: int
    total_facts_created: int
    average_consolidation_time_ms: float
    last_consolidation: Optional[datetime] = None


# ============================================================================
# Semantic Transition
# ============================================================================

class PatternDetectionRequest(BaseModel):
    """Request to detect episodic patterns."""
    threshold: int = Field(3, description="Min pattern frequency", ge=1)
    similarity_threshold: float = Field(0.75, description="Min similarity", ge=0.0, le=1.0)
    window_days: int = Field(7, description="Look-back window in days", ge=1)
    max_patterns: int = Field(50, description="Max patterns to return", ge=1, le=1000)

    class Config:
        schema_extra = {
            "example": {
                "threshold": 3,
                "similarity_threshold": 0.75,
                "window_days": 7,
                "max_patterns": 50
            }
        }


class EpisodicPattern(BaseModel):
    """Detected episodic pattern."""
    pattern_id: str
    pattern_type: str
    frequency: int
    similarity_score: float
    representative_query: str
    common_entities: List[str]
    common_motifs: List[str]
    episodic_memory_ids: List[str]
    first_seen: datetime
    last_seen: datetime


class PatternDetectionResponse(BaseModel):
    """Response from pattern detection."""
    patterns: List[EpisodicPattern]
    count: int
    detection_time_ms: float


class PromotePatternRequest(BaseModel):
    """Request to promote pattern to semantic concept."""
    pattern_id: str = Field(..., description="Pattern ID to promote")
    concept_name: Optional[str] = Field(None, description="Override concept name")

    class Config:
        schema_extra = {
            "example": {
                "pattern_id": "pattern_abc123",
                "concept_name": "thompson_sampling_basics"
            }
        }


class PromotePatternResponse(BaseModel):
    """Response from pattern promotion."""
    status: str = "queued"
    pattern_id: str
    job_id: str


class SemanticStatsResponse(BaseModel):
    """Semantic transition statistics."""
    total_patterns_detected: int
    total_concepts_created: int
    average_pattern_frequency: float
    episodic_to_semantic_ratio: float
    last_transition: Optional[datetime] = None


# ============================================================================
# Temporal Evolution
# ============================================================================

class TemporalQueryRequest(BaseModel):
    """Request to query understanding at specific time."""
    concept: str = Field(..., description="Concept to query", min_length=1)
    timestamp: datetime = Field(..., description="Point-in-time to query")

    class Config:
        schema_extra = {
            "example": {
                "concept": "thompson_sampling",
                "timestamp": "2025-11-01T12:00:00Z"
            }
        }

    @validator('timestamp')
    def timestamp_not_future(cls, v):
        """Ensure timestamp is not in the future."""
        if v > datetime.now():
            raise ValueError("timestamp cannot be in the future")
        return v


class UnderstandingSnapshot(BaseModel):
    """Snapshot of understanding at a point in time."""
    concept: str
    state: UnderstandingState
    memory_count: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    notable_memories: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TemporalQueryResponse(BaseModel):
    """Response from temporal query."""
    concept: str
    timestamp: datetime
    snapshot: Optional[UnderstandingSnapshot]
    found: bool


class ConceptHistoryResponse(BaseModel):
    """Complete evolution history for a concept."""
    concept: str
    first_learned: Optional[datetime]
    current_state: UnderstandingState
    state_transitions: List[Dict[str, Any]]
    memory_timeline: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]
    total_memories: int


class EvolutionSummaryRequest(BaseModel):
    """Request for evolution summary."""
    days: int = Field(30, description="Look-back period in days", ge=1, le=365)

    class Config:
        schema_extra = {
            "example": {
                "days": 30
            }
        }


class EvolutionSummaryResponse(BaseModel):
    """Summary of understanding evolution."""
    period_days: int
    total_concepts: int
    concepts_by_state: Dict[UnderstandingState, int]
    new_concepts: int
    mastered_concepts: int
    forgotten_concepts: int
    top_learning_areas: List[str]
    learning_velocity: float


# ============================================================================
# Curiosity Engine
# ============================================================================

class CuriositySuggestionRequest(BaseModel):
    """Request for exploration suggestions."""
    limit: int = Field(5, description="Max suggestions", ge=1, le=20)
    importance_threshold: float = Field(0.5, description="Min importance", ge=0.0, le=1.0)
    include_serendipity: bool = Field(True, description="Include random suggestions")

    class Config:
        schema_extra = {
            "example": {
                "limit": 5,
                "importance_threshold": 0.5,
                "include_serendipity": True
            }
        }


class ExplorationSuggestion(BaseModel):
    """Proactive exploration suggestion."""
    type: ExplorationSuggestionType
    concept: str
    reason: str
    importance: float = Field(..., ge=0.0, le=1.0)
    suggested_query: str
    expected_benefit: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class CuriositySuggestionResponse(BaseModel):
    """Response from curiosity suggestion."""
    suggestions: List[ExplorationSuggestion]
    count: int
    generation_time_ms: float


class CuriosityStatsResponse(BaseModel):
    """Curiosity engine statistics."""
    total_suggestions_generated: int
    suggestions_by_type: Dict[ExplorationSuggestionType, int]
    average_importance: float
    suggestions_followed: int
    follow_rate: float


# ============================================================================
# Graph Reasoning
# ============================================================================

class MultiHopQueryRequest(BaseModel):
    """Request for multi-hop graph query."""
    query: str = Field(..., description="Query text", min_length=1)
    max_hops: int = Field(3, description="Max hops to traverse", ge=1, le=5)
    max_results: int = Field(10, description="Max results", ge=1, le=100)
    edge_types: Optional[List[str]] = Field(
        None,
        description="Specific edge types to traverse"
    )

    class Config:
        schema_extra = {
            "example": {
                "query": "What papers cite Transformers?",
                "max_hops": 2,
                "max_results": 10,
                "edge_types": ["CITES", "RELATED"]
            }
        }


class GraphPath(BaseModel):
    """Path through knowledge graph."""
    start_entity: str
    end_entity: str
    path: List[str]
    edge_types: List[str]
    total_weight: float
    hop_count: int


class MultiHopResultItem(BaseModel):
    """Single result from multi-hop query."""
    memory_id: str
    content: str
    relevance_score: float
    hop_distance: int
    reasoning_path: Optional[GraphPath] = None


class MultiHopQueryResponse(BaseModel):
    """Response from multi-hop query."""
    query: str
    max_hops: int
    results: List[MultiHopResultItem]
    total_paths_explored: int
    query_entities: List[str]
    latency_ms: float


# ============================================================================
# Statistics
# ============================================================================

class SystemStatsResponse(BaseModel):
    """Complete system statistics."""
    consolidation: ConsolidationStatsResponse
    semantic: SemanticStatsResponse
    curiosity: CuriosityStatsResponse
    graph: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
