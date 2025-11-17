"""
Branches API Routes
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from promptly import Promptly, BranchNotFoundError, BranchExistsError, PromptlyError
from ..models.branches import (
    BranchCreate,
    BranchResponse,
    BranchCheckoutRequest,
    BranchListResponse,
)
from ..models.common import SuccessResponse
from ..middleware.auth import api_key_auth
from ..config import get_settings

settings = get_settings()
router = APIRouter()


def get_promptly() -> Promptly:
    """Get Promptly instance"""
    return Promptly(root_dir=settings.promptly_root_dir)


@router.post("", response_model=SuccessResponse, status_code=status.HTTP_201_CREATED)
async def create_branch(
    branch: BranchCreate,
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Create a new branch

    Creates a new branch from the current branch or specified source branch.
    """
    try:
        message = promptly.branch(
            branch_name=branch.name,
            from_branch=branch.from_branch
        )

        return SuccessResponse(
            message=message,
            data={"branch_name": branch.name}
        )

    except BranchExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except BranchNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/checkout", response_model=SuccessResponse)
async def checkout_branch(
    checkout: BranchCheckoutRequest,
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Checkout a branch

    Switch to a different branch.
    """
    try:
        message = promptly.checkout(checkout.branch_name)

        return SuccessResponse(
            message=message,
            data={"current_branch": checkout.branch_name}
        )

    except BranchNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("", response_model=BranchListResponse)
async def list_branches(
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    List all branches

    Returns all branches with their head commits.
    """
    try:
        current_branch = promptly._get_current_branch()

        # Get branches from database
        with promptly._get_db() as db:
            cursor = db.conn.cursor()
            cursor.execute("SELECT name, head_commit, created_at FROM branches ORDER BY created_at DESC")
            branches = cursor.fetchall()

        branch_list = [
            BranchResponse(
                name=branch[0],
                head_commit=branch[1],
                created_at=branch[2],
                is_current=(branch[0] == current_branch)
            )
            for branch in branches
        ]

        return BranchListResponse(
            branches=branch_list,
            current_branch=current_branch
        )

    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/{branch_name}", response_model=SuccessResponse)
async def delete_branch(
    branch_name: str,
    force: bool = False,
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Delete a branch

    Deletes a branch. Cannot delete the current branch or main branch.
    """
    try:
        current_branch = promptly._get_current_branch()

        # Prevent deletion of current branch
        if branch_name == current_branch:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete current branch. Checkout another branch first."
            )

        # Prevent deletion of main branch
        if branch_name == "main" and not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete main branch without force flag"
            )

        # Delete branch
        with promptly._get_db() as db:
            cursor = db.conn.cursor()

            # Check if branch exists
            cursor.execute("SELECT name FROM branches WHERE name = ?", (branch_name,))
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Branch '{branch_name}' not found"
                )

            # Delete branch
            cursor.execute("DELETE FROM branches WHERE name = ?", (branch_name,))
            db.conn.commit()

        return SuccessResponse(
            message=f"Deleted branch '{branch_name}'",
            data={"branch_name": branch_name}
        )

    except HTTPException:
        raise
    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{branch_name}", response_model=BranchResponse)
async def get_branch(
    branch_name: str,
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Get branch details

    Returns information about a specific branch.
    """
    try:
        current_branch = promptly._get_current_branch()

        with promptly._get_db() as db:
            cursor = db.conn.cursor()
            cursor.execute(
                "SELECT name, head_commit, created_at FROM branches WHERE name = ?",
                (branch_name,)
            )
            branch = cursor.fetchone()

        if not branch:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Branch '{branch_name}' not found"
            )

        return BranchResponse(
            name=branch[0],
            head_commit=branch[1],
            created_at=branch[2],
            is_current=(branch[0] == current_branch)
        )

    except HTTPException:
        raise
    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
