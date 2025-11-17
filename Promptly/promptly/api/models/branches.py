"""
Branch API Models
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class BranchCreate(BaseModel):
    """Create branch request"""
    name: str = Field(..., min_length=1, max_length=255, description="Branch name")
    from_branch: Optional[str] = Field(default=None, description="Source branch")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "feature/new-prompts",
                "from_branch": "main"
            }
        }


class BranchResponse(BaseModel):
    """Branch response model"""
    name: str
    head_commit: str
    created_at: str
    is_current: bool = False


class BranchCheckoutRequest(BaseModel):
    """Checkout branch request"""
    branch_name: str = Field(..., description="Branch to checkout")


class BranchListResponse(BaseModel):
    """List of branches"""
    branches: list[BranchResponse]
    current_branch: str


class BranchDeleteRequest(BaseModel):
    """Delete branch request"""
    force: bool = Field(default=False, description="Force delete even if not merged")
