"""
Agentic API Request/Response Schemas
=====================================
Pydantic models matching TypeScript interfaces in squad/src/HoloLoomBridge.ts.

Extracted from agentic_api.py (March 2026 Refactor).
"""

from typing import Any

from pydantic import BaseModel, Field, validator


class CodeContext(BaseModel):
    """Code context from VS Code editor (matches TypeScript interface)."""
    currentFile: str | None = None
    fileName: str | None = None
    languageId: str | None = None
    selection: str | None = None
    workspace: str | None = None
    diagnostics: list[dict] | None = None


class QueryRequest(BaseModel):
    """Query request from VS Code extension."""
    text: str = Field(..., description="Query text")
    context: CodeContext | None = Field(None, description="Code context from editor")
    mode: str = Field("verify", description="Reasoning mode: direct, verify, research, plan_execute")
    max_steps: int = Field(5, description="Maximum reasoning steps")

    @validator('text')
    def validate_text_size(cls, v):
        """Validate query text size (max 100KB)."""
        if len(v) > 100_000:
            raise ValueError(f"Query text too large: {len(v)} bytes (max 100KB)")
        return v

    @validator('max_steps')
    def validate_max_steps(cls, v):
        """Validate max_steps is reasonable."""
        if v < 1 or v > 20:
            raise ValueError(f"max_steps must be between 1 and 20 (got {v})")
        return v


class VerificationResponse(BaseModel):
    """Verification result (matches TypeScript interface)."""
    verified: bool
    confidence: float
    contradictions: list[str]
    supporting_evidence: list[str]
    suggested_refinements: list[str]


class ReasoningStepResponse(BaseModel):
    """Single reasoning step (matches TypeScript interface)."""
    type: str
    query: str | None = None
    confidence: float | None = None
    finding: str | None = None
    completed: bool | None = None
    tool: str | None = None


class AgenticResponse(BaseModel):
    """Agentic reasoning result (matches TypeScript AgenticResult interface)."""
    response: str
    confidence: float
    reasoning_mode: str
    steps_taken: list[ReasoningStepResponse]
    total_queries: int
    total_duration_ms: float
    verification: VerificationResponse | None = None
    timestamp: str
    query_id: str


class MemoryAddRequest(BaseModel):
    """
    Request model for adding memories.
    SECURITY: Pydantic validation prevents resource exhaustion attacks.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="Memory text content (max 100KB)"
    )
    episode: str = Field(
        default="default",
        max_length=1000,
        description="Episode/category for the memory (max 1KB)"
    )
    entities: list[str] = Field(
        default_factory=list,
        max_items=100,
        description="List of entities mentioned (max 100 items)"
    )
    motifs: list[str] = Field(
        default_factory=list,
        max_items=100,
        description="List of motifs/patterns (max 100 items)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @validator('entities', 'motifs', each_item=True)
    def validate_list_items(cls, v):
        """SECURITY: Limit individual list item lengths."""
        if len(v) > 1000:
            raise ValueError("List items must be under 1KB")
        return v

    @validator('metadata')
    def validate_metadata_size(cls, v):
        """SECURITY: Limit metadata size to prevent abuse."""
        import json
        if len(json.dumps(v)) > 10000:
            raise ValueError("Metadata must be under 10KB when serialized")
        return v
