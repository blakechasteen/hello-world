"""
Prompt API Models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class PromptCreate(BaseModel):
    """Create/update prompt request"""
    name: str = Field(..., min_length=1, max_length=255, description="Prompt name")
    content: str = Field(..., min_length=1, description="Prompt content/template")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "summarizer",
                "content": "Summarize the following text:\n\n{text}",
                "metadata": {"tags": ["nlp", "summarization"], "version": "1.0"}
            }
        }


class PromptResponse(BaseModel):
    """Prompt response model"""
    name: str
    content: str
    branch: str
    version: int
    commit_hash: str
    created_at: str
    metadata: Dict[str, Any] = {}


class PromptListItem(BaseModel):
    """Prompt list item"""
    name: str
    version: int
    commit_hash: str
    created_at: str


class PromptSearchRequest(BaseModel):
    """Prompt search request"""
    query: Optional[str] = Field(default=None, description="Search query")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    branch: Optional[str] = Field(default=None, description="Filter by branch")
    limit: int = Field(default=20, ge=1, le=100)


class PromptVersionRequest(BaseModel):
    """Get specific prompt version"""
    version: Optional[int] = Field(default=None, description="Specific version number")
    commit_hash: Optional[str] = Field(default=None, description="Specific commit hash")


class PromptDiffResponse(BaseModel):
    """Prompt diff response"""
    prompt_name: str
    version_old: int
    version_new: int
    commit_hash_old: str
    commit_hash_new: str
    diff: str
    changes_summary: Dict[str, Any]
