"""
Chain API Models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ChainStepStatus(str, Enum):
    """Chain step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ChainCreate(BaseModel):
    """Create chain request"""
    name: str = Field(..., min_length=1, max_length=255, description="Chain name")
    steps: List[str] = Field(..., min_items=1, description="Prompt names in order")
    description: Optional[str] = Field(default=None, description="Chain description")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "research-pipeline",
                "steps": ["query-expander", "searcher", "summarizer"],
                "description": "Research query pipeline"
            }
        }


class ChainResponse(BaseModel):
    """Chain response model"""
    name: str
    steps: List[str]
    description: Optional[str]
    created_at: str


class ChainExecuteRequest(BaseModel):
    """Execute chain request"""
    chain_name: str = Field(..., description="Chain to execute")
    initial_input: Dict[str, Any] = Field(..., description="Initial input variables")
    model_endpoint: Optional[str] = Field(default=None, description="Optional model endpoint")
    stream: bool = Field(default=False, description="Stream results via WebSocket")


class ChainStepResult(BaseModel):
    """Chain step result"""
    step_name: str
    step_index: int
    status: ChainStepStatus
    formatted_prompt: str
    output: Optional[str]
    error: Optional[str] = None
    execution_time_ms: float


class ChainExecutionResponse(BaseModel):
    """Chain execution response"""
    execution_id: str
    chain_name: str
    status: str
    steps: List[ChainStepResult]
    total_execution_time_ms: float
    started_at: datetime
    completed_at: Optional[datetime] = None


class ChainExecutionStatusRequest(BaseModel):
    """Get chain execution status"""
    execution_id: str


class ChainListResponse(BaseModel):
    """List of chains"""
    chains: List[ChainResponse]
    total: int
