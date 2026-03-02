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

from typing import Optional, List, Dict
from pydantic import BaseModel, Field, validator


class CodeContext(BaseModel):
    """Code context from VS Code editor (matches TypeScript interface)."""
    currentFile: Optional[str] = None
    fileName: Optional[str] = None
    languageId: Optional[str] = None
    selection: Optional[str] = None
    workspace: Optional[str] = None
    diagnostics: Optional[List[Dict]] = None


class QueryRequest(BaseModel):
    """Query request from VS Code extension."""
    text: str = Field(..., description="Query text")
    context: Optional[CodeContext] = Field(None, description="Code context from editor")
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
    contradictions: List[str]
    supporting_evidence: List[str]
    suggested_refinements: List[str]


class ReasoningStepResponse(BaseModel):
    """Single reasoning step (matches TypeScript interface)."""
    type: str
    query: Optional[str] = None
    confidence: Optional[float] = None
    finding: Optional[str] = None
    completed: Optional[bool] = None
    tool: Optional[str] = None


class AgenticResponse(BaseModel):
    """
    Agentic reasoning result (matches TypeScript AgenticResult interface).

    This is the main response returned to the VS Code extension.
    """
    response: str
    confidence: float
    reasoning_mode: str
    steps_taken: List[ReasoningStepResponse]
    total_queries: int
    total_duration_ms: float
    verification: Optional[VerificationResponse] = None

    # Additional metadata
    timestamp: str
    query_id: str
