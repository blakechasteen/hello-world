"""Submission routes"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from datetime import datetime

from backend.database import get_db
from backend.models import Submission, User
from backend.auth import get_current_user

router = APIRouter()


class SubmissionCreate(BaseModel):
    assessment_id: str
    content: str
    attachments: List[str] = []


class SubmissionResponse(BaseModel):
    submission_id: str
    assessment_id: str
    student_id: str
    submitted_at: str
    score: float | None
    feedback: str | None

    class Config:
        from_attributes = True


@router.post("/", response_model=SubmissionResponse)
async def create_submission(
    submission: SubmissionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Submit work for an assessment"""
    new_submission = Submission(
        assessment_id=submission.assessment_id,
        student_id=current_user.user_id,
        content=submission.content,
        attachments=submission.attachments,
        word_count=len(submission.content.split()) if submission.content else 0
    )

    db.add(new_submission)
    await db.commit()
    await db.refresh(new_submission)

    return SubmissionResponse.model_validate(new_submission)


@router.get("/student/me", response_model=List[SubmissionResponse])
async def my_submissions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all submissions for current user"""
    result = await db.execute(
        select(Submission)
        .where(Submission.student_id == current_user.user_id)
        .order_by(Submission.submitted_at.desc())
    )
    submissions = result.scalars().all()
    return [SubmissionResponse.model_validate(s) for s in submissions]
