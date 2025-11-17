"""
Evaluations API Routes
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Optional
import sys
from pathlib import Path
import uuid
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from promptly import Promptly, PromptNotFoundError, PromptlyError
from ..models.evaluations import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationResponse,
    EvaluationListItem,
    EvaluationCompareRequest,
    EvaluationCompareResponse,
)
from ..middleware.auth import api_key_auth
from ..config import get_settings

settings = get_settings()
router = APIRouter()


def get_promptly() -> Promptly:
    """Get Promptly instance"""
    return Promptly(root_dir=settings.promptly_root_dir)


# In-memory storage for evaluation results (use database in production)
EVALUATIONS = {}


@router.post("", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
async def run_evaluation(
    evaluation: EvaluationRequest,
    authenticated: bool = Depends(api_key_auth),
    promptly: Promptly = Depends(get_promptly)
):
    """
    Run evaluation on a prompt

    Evaluates a prompt against test cases using specified evaluator.
    """
    try:
        # Convert test cases to Promptly format
        test_cases = [
            {
                "id": tc.id,
                "inputs": tc.inputs,
                "expected": tc.expected,
                "context": tc.context or {}
            }
            for tc in evaluation.test_cases
        ]

        # Run evaluation
        results = promptly.eval_prompt(
            name=evaluation.prompt_name,
            test_cases=test_cases,
            evaluator_name=evaluation.evaluator
        )

        # Get prompt info
        prompt = promptly.get(evaluation.prompt_name)

        # Convert results
        eval_results = [
            EvaluationResult(
                test_case_id=r['test_case'].get('id'),
                formatted_prompt=r['formatted_prompt'],
                actual=r.get('actual'),
                expected=r['test_case'].get('expected'),
                score=r.get('score'),
                metrics=r.get('metrics', {})
            )
            for r in results
        ]

        # Calculate statistics
        scores = [r.score for r in eval_results if r.score is not None]
        avg_score = sum(scores) / len(scores) if scores else None

        # Count passed/failed (using 0.7 threshold)
        passed = sum(1 for s in scores if s >= 0.7)
        failed = len(scores) - passed

        # Create response
        evaluation_id = str(uuid.uuid4())

        response = EvaluationResponse(
            evaluation_id=evaluation_id,
            prompt_name=evaluation.prompt_name,
            commit_hash=prompt['commit_hash'],
            results=eval_results,
            average_score=avg_score,
            total_tests=len(eval_results),
            passed_tests=passed,
            failed_tests=failed,
            created_at=datetime.utcnow()
        )

        # Store evaluation
        EVALUATIONS[evaluation_id] = response

        return response

    except PromptNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PromptlyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: str,
    authenticated: bool = Depends(api_key_auth)
):
    """
    Get evaluation results

    Retrieves results of a previously run evaluation.
    """
    evaluation = EVALUATIONS.get(evaluation_id)

    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation '{evaluation_id}' not found"
        )

    return evaluation


@router.get("", response_model=List[EvaluationListItem])
async def list_evaluations(
    prompt_name: Optional[str] = Query(default=None, description="Filter by prompt name"),
    limit: int = Query(default=20, ge=1, le=100),
    authenticated: bool = Depends(api_key_auth)
):
    """
    List evaluations

    Returns list of all evaluations, optionally filtered by prompt name.
    """
    evaluations = list(EVALUATIONS.values())

    # Filter by prompt name
    if prompt_name:
        evaluations = [e for e in evaluations if e.prompt_name == prompt_name]

    # Sort by created_at descending
    evaluations.sort(key=lambda x: x.created_at, reverse=True)

    # Apply limit
    evaluations = evaluations[:limit]

    # Convert to list items
    items = [
        EvaluationListItem(
            evaluation_id=e.evaluation_id,
            prompt_name=e.prompt_name,
            commit_hash=e.commit_hash,
            average_score=e.average_score,
            total_tests=e.total_tests,
            created_at=e.created_at.isoformat()
        )
        for e in evaluations
    ]

    return items


@router.post("/compare", response_model=EvaluationCompareResponse)
async def compare_evaluations(
    compare: EvaluationCompareRequest,
    authenticated: bool = Depends(api_key_auth)
):
    """
    Compare evaluations

    Compares multiple evaluations and determines the best performing one.
    """
    # Get all evaluations
    evaluations_to_compare = []

    for eval_id in compare.evaluation_ids:
        evaluation = EVALUATIONS.get(eval_id)
        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation '{eval_id}' not found"
            )
        evaluations_to_compare.append(evaluation)

    # Convert to list items
    eval_items = [
        EvaluationListItem(
            evaluation_id=e.evaluation_id,
            prompt_name=e.prompt_name,
            commit_hash=e.commit_hash,
            average_score=e.average_score,
            total_tests=e.total_tests,
            created_at=e.created_at.isoformat()
        )
        for e in evaluations_to_compare
    ]

    # Calculate comparison metrics
    scores = [e.average_score for e in evaluations_to_compare if e.average_score is not None]

    if scores:
        winner_idx = scores.index(max(scores))
        winner_id = evaluations_to_compare[winner_idx].evaluation_id
    else:
        winner_id = None

    comparison = {
        "average_scores": {
            e.evaluation_id: e.average_score
            for e in evaluations_to_compare
        },
        "total_tests": {
            e.evaluation_id: e.total_tests
            for e in evaluations_to_compare
        },
        "passed_tests": {
            e.evaluation_id: e.passed_tests
            for e in evaluations_to_compare
        }
    }

    return EvaluationCompareResponse(
        evaluations=eval_items,
        comparison=comparison,
        winner=winner_id
    )
