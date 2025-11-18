"""Assessment routes"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from backend.database import get_db
from backend.models import Assessment, User
from backend.auth import get_current_user

router = APIRouter()


class AssessmentResponse(BaseModel):
    assessment_id: str
    title: str
    description: str | None
    assessment_type: str
    max_score: float
    due_date: str | None
    status: str

    class Config:
        from_attributes = True


@router.get("/course/{course_id}", response_model=List[AssessmentResponse])
async def list_assessments(
    course_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all published assessments for a course"""
    result = await db.execute(
        select(Assessment)
        .where(Assessment.course_id == course_id)
        .where(Assessment.status == "published")
    )
    assessments = result.scalars().all()
    return [AssessmentResponse.model_validate(a) for a in assessments]
