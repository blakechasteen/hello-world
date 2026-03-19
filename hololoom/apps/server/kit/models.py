"""
Shared Pydantic request/response models for chat agents.

Agent-specific fields (intent, refinement_id, jenny_id) are Optional
so simpler agents like Elle can ignore them.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000, description="Message text")
    room_id: str = Field(default="default", description="Room/conversation ID for memory")


class RoutingInfo(BaseModel):
    """Transparent routing decision — why this model was chosen."""
    intent: str | None = None
    intent_confidence: float | None = None
    model_id: str | None = None
    reason: str | None = None
    fallback_model: str | None = None
    speed_weight: float | None = None
    health_status: dict[str, bool] = Field(default_factory=dict)


class RefinementInfo(BaseModel):
    """Refinement trigger context — why (or why not) multi-pass was fired."""
    triggered: bool = False
    complexity_score: float | None = None
    threshold: float = 0.25
    max_passes: int | None = None


class MemoryHit(BaseModel):
    """A single recalled memory with provenance."""
    text: str
    relevance: float | None = None
    source: str | None = None  # Episode or entity origin


class MemoryContext(BaseModel):
    """Structured memory recall metadata."""
    hits: list[MemoryHit] = Field(default_factory=list)
    recall_ms: float | None = None
    backend: str = "none"  # "hololoom_lite" | "none" | "degraded"


class ChatResponse(BaseModel):
    response: str
    room_id: str
    model: str
    tokens: int
    duration_ms: float
    turns_in_memory: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    intent: str | None = None
    refinement_id: str | None = None
    jenny_id: str | None = None
    routing: RoutingInfo | None = None
    refinement_info: RefinementInfo | None = None
    memory_context: MemoryContext | None = None
