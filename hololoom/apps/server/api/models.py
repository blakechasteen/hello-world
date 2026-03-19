"""
API Request/Response Models
===========================

Pydantic models for API request validation and response serialization.
These models match the TypeScript interfaces in squad/src/HoloLoomBridge.ts

Classes:
    CodeContext: Code context from VS Code editor
    QueryRequest: Query request from extension
    VerificationResponse: Verification result
    ReasoningStepResponse: Single reasoning step
    AgenticResponse: Main response for agentic queries
"""


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
        if len(v) > 100_000:  # 100KB limit
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
    """
    Agentic reasoning result (matches TypeScript AgenticResult interface).

    This is the main response returned to the VS Code extension.
    """
    response: str
    confidence: float
    reasoning_mode: str
    steps_taken: list[ReasoningStepResponse]
    total_queries: int
    total_duration_ms: float
    verification: VerificationResponse | None = None

    # Additional metadata
    timestamp: str
    query_id: str
