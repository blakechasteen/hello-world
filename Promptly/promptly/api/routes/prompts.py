"""
Prompts API Routes
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Optional
import sys
from pathlib import Path

# Add parent directory to path to import Promptly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from promptly import Promptly, PromptNotFoundError, PromptlyError
from ..models.prompts import (
    PromptCreate,
    PromptResponse,
    PromptListItem,
    PromptSearchRequest,
    PromptDiffResponse,
)
from ..models.common import SuccessResponse, PaginatedResponse
from ..middleware.auth import api_key_auth
from ..config import get_settings

settings = get_settings()
router = APIRouter()


def get_promptly() -> Promptly:
    """Get Promptly instance"""
    return Promptly(root_dir=settings.promptly_root_dir)


@router.post("", response_model=SuccessResponse, status_code=status.HTTP_201_CREATED)
async def create_prompt(
    prompt: PromptCreate,
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Create or update a prompt

    Creates a new prompt or updates an existing one with a new version.
    """
    try:
        message = promptly.add(
            name=prompt.name,
            content=prompt.content,
            metadata=prompt.metadata
        )

        return SuccessResponse(
            message=message,
            data={"name": prompt.name}
        )

    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{name}", response_model=PromptResponse)
async def get_prompt(
    name: str,
    version: Optional[int] = Query(default=None, description="Specific version number"),
    commit_hash: Optional[str] = Query(default=None, description="Specific commit hash"),
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Get a prompt by name

    Retrieves the latest version by default, or a specific version/commit if specified.
    """
    try:
        prompt = promptly.get(name=name, version=version, commit_hash=commit_hash)

        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt '{name}' not found"
            )

        return PromptResponse(**prompt)

    except PromptNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("", response_model=List[PromptListItem])
async def list_prompts(
    branch: Optional[str] = Query(default=None, description="Filter by branch"),
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    List all prompts

    Returns all prompts on the current branch or specified branch.
    """
    try:
        prompts = promptly.list_prompts(branch=branch)
        return [PromptListItem(**p) for p in prompts]

    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/{name}", response_model=SuccessResponse)
async def delete_prompt(
    name: str,
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Delete a prompt

    Removes a prompt and all its versions from the current branch.
    """
    try:
        # Note: This would require adding delete functionality to Promptly core
        # For now, return not implemented
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Delete functionality not yet implemented in Promptly core"
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/search", response_model=List[PromptListItem])
async def search_prompts(
    search: PromptSearchRequest,
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Search prompts

    Search prompts by query, tags, or branch.
    """
    try:
        # Get all prompts
        prompts = promptly.list_prompts(branch=search.branch)

        # Filter by query (search in name and content)
        if search.query:
            filtered = []
            for prompt_item in prompts:
                # Need to get full prompt to search content
                full_prompt = promptly.get(prompt_item['name'], version=prompt_item['version'])
                if full_prompt:
                    if (search.query.lower() in full_prompt['name'].lower() or
                        search.query.lower() in full_prompt['content'].lower()):
                        filtered.append(prompt_item)
            prompts = filtered

        # Filter by tags
        if search.tags:
            filtered = []
            for prompt_item in prompts:
                full_prompt = promptly.get(prompt_item['name'], version=prompt_item['version'])
                if full_prompt and full_prompt.get('metadata', {}).get('tags'):
                    prompt_tags = full_prompt['metadata']['tags']
                    if any(tag in prompt_tags for tag in search.tags):
                        filtered.append(prompt_item)
            prompts = filtered

        # Apply limit
        prompts = prompts[:search.limit]

        return [PromptListItem(**p) for p in prompts]

    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{name}/diff", response_model=PromptDiffResponse)
async def get_prompt_diff(
    name: str,
    version_old: int = Query(..., description="Old version number"),
    version_new: int = Query(..., description="New version number"),
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Get diff between two prompt versions

    Returns the differences between two versions of a prompt.
    """
    try:
        import difflib

        # Get both versions
        prompt_old = promptly.get(name=name, version=version_old)
        prompt_new = promptly.get(name=name, version=version_new)

        if not prompt_old or not prompt_new:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"One or both versions not found"
            )

        # Generate diff
        diff = list(difflib.unified_diff(
            prompt_old['content'].splitlines(keepends=True),
            prompt_new['content'].splitlines(keepends=True),
            fromfile=f"{name} v{version_old}",
            tofile=f"{name} v{version_new}",
            lineterm=''
        ))

        # Count changes
        additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))

        return PromptDiffResponse(
            prompt_name=name,
            version_old=version_old,
            version_new=version_new,
            commit_hash_old=prompt_old['commit_hash'],
            commit_hash_new=prompt_new['commit_hash'],
            diff=''.join(diff),
            changes_summary={
                "additions": additions,
                "deletions": deletions,
                "total_changes": additions + deletions
            }
        )

    except PromptNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
