"""
Pydantic request/response models for the Promptly API.

Shared models (ChatRequest, ChatResponse) come from kit.
Promptly-specific models are defined here.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# Re-export shared models so existing imports still work
from ..kit.models import (  # noqa: F401
    ChatRequest,
    ChatResponse,
    MemoryContext,
    MemoryHit,
    RefinementInfo,
    RoutingInfo,
)


class RefinementResponse(BaseModel):
    status: str  # "pending", "complete", "skipped", "converged", "error"
    refinement: str | None = None
    reason: str | None = None
    pass_number: int | None = None
    max_passes: int | None = None
    next_refinement_id: str | None = None
    model: str | None = None
    strategy: str | None = None  # MRF strategy used: VERIFY, CRITIQUE, ELEGANCE, etc.


class ProposeRequest(BaseModel):
    title: str = Field(..., description="Claim-style title for the note")
    content: str = Field(..., description="Markdown body with wiki links")
    links: list[str] | None = Field(None, description="Wiki link targets")
    department: str | None = Field(None, description="Target department (None = vault inbox)")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: str | None = Field(None, description="Voice ID (default: server config)")
