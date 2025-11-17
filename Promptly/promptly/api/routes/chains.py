"""
Chains API Routes
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
import sys
from pathlib import Path
import uuid
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from promptly import Promptly, ChainNotFoundError, PromptlyError
from ..models.chains import (
    ChainCreate,
    ChainResponse,
    ChainExecuteRequest,
    ChainStepResult,
    ChainExecutionResponse,
    ChainListResponse,
    ChainStepStatus,
)
from ..models.common import SuccessResponse
from ..middleware.auth import api_key_auth
from ..config import get_settings

settings = get_settings()
router = APIRouter()


def get_promptly() -> Promptly:
    """Get Promptly instance"""
    return Promptly(root_dir=settings.promptly_root_dir)


# In-memory storage for chain executions (use database in production)
EXECUTIONS = {}


@router.post("", response_model=SuccessResponse, status_code=status.HTTP_201_CREATED)
async def create_chain(
    chain: ChainCreate,
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Create a new chain

    Creates a chain of prompts to be executed in sequence.
    """
    try:
        message = promptly.create_chain(
            name=chain.name,
            steps=chain.steps,
            description=chain.description
        )

        return SuccessResponse(
            message=message,
            data={"chain_name": chain.name, "steps": chain.steps}
        )

    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("", response_model=ChainListResponse)
async def list_chains(
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    List all chains

    Returns all defined chains.
    """
    try:
        # Get chains from database
        with promptly._get_db() as db:
            cursor = db.conn.cursor()
            cursor.execute("SELECT name, steps, description, created_at FROM chains ORDER BY created_at DESC")
            chains = cursor.fetchall()

        chain_list = [
            ChainResponse(
                name=chain[0],
                steps=eval(chain[1]) if isinstance(chain[1], str) else chain[1],  # JSON decode
                description=chain[2],
                created_at=chain[3]
            )
            for chain in chains
        ]

        return ChainListResponse(
            chains=chain_list,
            total=len(chain_list)
        )

    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{chain_name}", response_model=ChainResponse)
async def get_chain(
    chain_name: str,
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Get chain details

    Returns information about a specific chain.
    """
    try:
        import json

        with promptly._get_db() as db:
            cursor = db.conn.cursor()
            cursor.execute(
                "SELECT name, steps, description, created_at FROM chains WHERE name = ?",
                (chain_name,)
            )
            chain = cursor.fetchone()

        if not chain:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chain '{chain_name}' not found"
            )

        return ChainResponse(
            name=chain[0],
            steps=json.loads(chain[1]),
            description=chain[2],
            created_at=chain[3]
        )

    except HTTPException:
        raise
    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/execute", response_model=ChainExecutionResponse)
async def execute_chain(
    execution: ChainExecuteRequest,
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Execute a chain

    Executes a chain with the provided initial input.
    """
    try:
        execution_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        start_time = time.time()

        # Execute chain
        results = promptly.execute_chain(
            name=execution.chain_name,
            initial_input=execution.initial_input
        )

        # Convert results
        step_results = []
        for i, result in enumerate(results):
            step_results.append(
                ChainStepResult(
                    step_name=result['step'],
                    step_index=i,
                    status=ChainStepStatus.COMPLETED,
                    formatted_prompt=result['prompt'],
                    output=result.get('output'),
                    error=None,
                    execution_time_ms=0.0  # Would need to track per-step timing
                )
            )

        total_time = (time.time() - start_time) * 1000

        response = ChainExecutionResponse(
            execution_id=execution_id,
            chain_name=execution.chain_name,
            status="completed",
            steps=step_results,
            total_execution_time_ms=total_time,
            started_at=started_at,
            completed_at=datetime.utcnow()
        )

        # Store execution
        EXECUTIONS[execution_id] = response

        return response

    except ChainNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/executions/{execution_id}", response_model=ChainExecutionResponse)
async def get_execution_status(
    execution_id: str,
    authenticated: bool = Depends(api_key_auth)
):
    """
    Get chain execution status

    Retrieves the status and results of a chain execution.
    """
    execution = EXECUTIONS.get(execution_id)

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution '{execution_id}' not found"
        )

    return execution


@router.delete("/{chain_name}", response_model=SuccessResponse)
async def delete_chain(
    chain_name: str,
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Delete a chain

    Removes a chain definition.
    """
    try:
        with promptly._get_db() as db:
            cursor = db.conn.cursor()

            # Check if chain exists
            cursor.execute("SELECT name FROM chains WHERE name = ?", (chain_name,))
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chain '{chain_name}' not found"
                )

            # Delete chain
            cursor.execute("DELETE FROM chains WHERE name = ?", (chain_name,))
            db.conn.commit()

        return SuccessResponse(
            message=f"Deleted chain '{chain_name}'",
            data={"chain_name": chain_name}
        )

    except HTTPException:
        raise
    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
