"""Lesson routes"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from backend.database import get_db
from backend.models import Lesson, User
from backend.auth import get_current_user

router = APIRouter()


class LessonResponse(BaseModel):
    lesson_id: str
    title: str
    description: str | None
    theme: str
    concept: str | None
    status: str

    class Config:
        from_attributes = True


@router.get("/course/{course_id}", response_model=List[LessonResponse])
async def list_lessons(
    course_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all published lessons for a course"""
    result = await db.execute(
        select(Lesson)
        .where(Lesson.course_id == course_id)
        .where(Lesson.status == "published")
        .order_by(Lesson.order_index)
    )
    lessons = result.scalars().all()
    return [LessonResponse.model_validate(l) for l in lessons]
