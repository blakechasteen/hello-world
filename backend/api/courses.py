"""Course routes"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from backend.database import get_db
from backend.models import Course, User
from backend.auth import get_current_user

router = APIRouter()


class CourseResponse(BaseModel):
    course_id: str
    code: str
    name: str
    description: str | None
    semester: str | None
    year: int | None
    status: str

    class Config:
        from_attributes = True


@router.get("/", response_model=List[CourseResponse])
async def list_courses(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all active courses for current user's institution"""
    result = await db.execute(
        select(Course)
        .where(Course.institution_id == current_user.institution_id)
        .where(Course.status == "active")
    )
    courses = result.scalars().all()
    return [CourseResponse.model_validate(c) for c in courses]


@router.get("/{course_id}", response_model=CourseResponse)
async def get_course(
    course_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get course by ID"""
    result = await db.execute(select(Course).where(Course.course_id == course_id))
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    return CourseResponse.model_validate(course)
