"""
History API Routes
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from promptly import Promptly, PromptlyError
from ..middleware.auth import api_key_auth
from ..config import get_settings
from pydantic import BaseModel

settings = get_settings()
router = APIRouter()


def get_promptly() -> Promptly:
    """Get Promptly instance"""
    return Promptly(root_dir=settings.promptly_root_dir)


class CommitLogItem(BaseModel):
    """Commit log item"""
    commit_hash: str
    name: str
    version: int
    branch: str
    created_at: str


class BlameLineItem(BaseModel):
    """Blame line item"""
    line_number: int
    content: str
    commit_hash: str
    version: int
    created_at: str


@router.get("/log", response_model=List[CommitLogItem])
async def get_commit_log(
    name: Optional[str] = Query(default=None, description="Filter by prompt name"),
    limit: int = Query(default=10, ge=1, le=100, description="Number of commits to show"),
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Get commit history

    Returns commit history for all prompts or a specific prompt.
    """
    try:
        commits = promptly.log(name=name, limit=limit)
        return [CommitLogItem(**commit) for commit in commits]

    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/blame/{name}", response_model=List[BlameLineItem])
async def get_prompt_blame(
    name: str,
    version: Optional[int] = Query(default=None, description="Specific version"),
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Get blame information for a prompt

    Shows which version/commit introduced each line of the prompt.
    """
    try:
        # Get the prompt
        prompt = promptly.get(name=name, version=version)

        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt '{name}' not found"
            )

        # For simplicity, attribute all lines to the current version
        # A full implementation would track line-by-line changes across versions
        lines = prompt['content'].splitlines()

        blame_info = [
            BlameLineItem(
                line_number=i + 1,
                content=line,
                commit_hash=prompt['commit_hash'],
                version=prompt['version'],
                created_at=prompt['created_at']
            )
            for i, line in enumerate(lines)
        ]

        return blame_info

    except HTTPException:
        raise
    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
