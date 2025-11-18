"""API routes"""
from fastapi import APIRouter

from backend.api import auth, courses, lessons, assessments, submissions

api_router = APIRouter(prefix="/api")

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(courses.router, prefix="/courses", tags=["courses"])
api_router.include_router(lessons.router, prefix="/lessons", tags=["lessons"])
api_router.include_router(assessments.router, prefix="/assessments", tags=["assessments"])
api_router.include_router(submissions.router, prefix="/submissions", tags=["submissions"])

__all__ = ["api_router"]
