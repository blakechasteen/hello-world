"""
EdWIN Mobile API - Optimized endpoints for mobile devices

Provides lightweight, efficient API endpoints designed for mobile constraints:
- Reduced payload sizes
- Batch operations
- Offline sync support
- Progressive loading
- Lower latency responses

Features:
- Lightweight student summaries
- Batch question answering
- Offline queue support
- Progressive objective loading
- Mobile-optimized image handling

Author: EdWIN Team
Date: 2025-11-15
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import asyncio
import logging

from .core import EdWIN
from .curriculum_graph import EdWINKnowledgeGraph
from .multimodal_tutor import EdWINMultimodalTutor
from .analytics import LearningAnalytics
from .student_model import EdWINStudentModel

logger = logging.getLogger(__name__)


# ================== Mobile-Optimized Request Models ==================

class LightweightStudentCreate(BaseModel):
    """Minimal student creation for mobile"""
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    grade_level: int = Field(..., ge=1, le=12)


class QuickQuestionRequest(BaseModel):
    """Lightweight question request"""
    student_id: str
    question: str = Field(..., min_length=1, max_length=500)
    include_sources: bool = False  # Reduce payload if not needed


class BatchQuestionRequest(BaseModel):
    """Batch multiple questions for efficiency"""
    student_id: str
    questions: List[str] = Field(..., max_items=10)
    include_sources: bool = False


class OfflineSyncRequest(BaseModel):
    """Sync offline interactions when back online"""
    student_id: str
    interactions: List[Dict[str, Any]] = Field(..., max_items=100)


class ProgressiveObjectivesRequest(BaseModel):
    """Request objectives in chunks for progressive loading"""
    student_id: str
    page: int = Field(1, ge=1)
    page_size: int = Field(10, ge=1, le=50)
    subject_filter: Optional[str] = None


# ================== Mobile-Optimized Response Models ==================

class LightweightStudentResponse(BaseModel):
    """Minimal student data for mobile"""
    student_id: str
    name: str
    grade_level: int
    mastered_count: int
    engagement_score: float
    last_synced: float


class QuickAnswerResponse(BaseModel):
    """Lightweight answer response"""
    answer: str
    confidence: float
    timestamp: float
    # Sources omitted by default for mobile


class BatchAnswerResponse(BaseModel):
    """Batch answer response"""
    answers: List[QuickAnswerResponse]
    total: int
    processing_time_ms: float


class ProgressivePage(BaseModel):
    """Paginated objectives for progressive loading"""
    items: List[Dict[str, Any]]
    page: int
    page_size: int
    total_pages: int
    has_more: bool


class SyncResult(BaseModel):
    """Offline sync result"""
    synced_count: int
    failed_count: int
    errors: List[str]
    timestamp: float


# ================== Mobile API Application ==================

class MobileAppState:
    """Global state for mobile API"""
    def __init__(self):
        self.curriculum_kg: Optional[EdWINKnowledgeGraph] = None
        self.multimodal_tutor: Optional[EdWINMultimodalTutor] = None
        self.analytics: Optional[LearningAnalytics] = None
        self.students: Dict[str, EdWIN] = {}


def create_mobile_app() -> FastAPI:
    """Create FastAPI app with mobile-optimized endpoints"""

    app = FastAPI(
        title="EdWIN Mobile API",
        version="3.0.0",
        description="Lightweight API for mobile devices"
    )

    # CORS for mobile apps
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    state = MobileAppState()

    # ================== Startup/Shutdown ==================

    @app.on_event("startup")
    async def startup():
        """Initialize components"""
        state.curriculum_kg = EdWINKnowledgeGraph()
        state.multimodal_tutor = EdWINMultimodalTutor()
        state.analytics = LearningAnalytics()

        # Initialize curriculum
        try:
            await state.curriculum_kg.ingest_curriculum()
            await state.multimodal_tutor.initialize()
            logger.info("Mobile API initialized")
        except Exception as e:
            logger.error(f"Initialization error: {e}")

    # ================== Health & Status ==================

    @app.get("/mobile/health")
    async def health_check():
        """Minimal health check for mobile"""
        return {
            "status": "ok",
            "timestamp": datetime.now().timestamp(),
            "students": len(state.students)
        }

    @app.get("/mobile/status")
    async def status():
        """Lightweight status for mobile dashboard"""
        return {
            "online": True,
            "timestamp": datetime.now().timestamp(),
            "student_count": len(state.students),
            "curriculum_loaded": state.curriculum_kg is not None
        }

    # ================== Student Management ==================

    @app.post("/mobile/students", response_model=LightweightStudentResponse)
    async def create_student(request: LightweightStudentCreate):
        """Create student with minimal payload"""
        try:
            # Create EdWIN instance
            edwin = EdWIN(
                first_name=request.first_name,
                last_name=request.last_name,
                grade_level=request.grade_level
            )

            # Initialize
            await edwin.initialize(
                curriculum_kg=state.curriculum_kg,
                multimodal_tutor=state.multimodal_tutor
            )

            # Generate lightweight student ID
            student_id = f"{request.first_name.lower()}.{request.last_name.lower()}.{len(state.students)}"
            state.students[student_id] = edwin

            # Calculate engagement
            engagement = state.analytics.calculate_engagement(edwin, window_days=7)

            return LightweightStudentResponse(
                student_id=student_id,
                name=f"{request.first_name} {request.last_name}",
                grade_level=request.grade_level,
                mastered_count=0,
                engagement_score=engagement.engagement_score,
                last_synced=datetime.now().timestamp()
            )

        except Exception as e:
            logger.error(f"Error creating student: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/mobile/students/{student_id}", response_model=LightweightStudentResponse)
    async def get_student(student_id: str):
        """Get lightweight student summary"""
        if student_id not in state.students:
            raise HTTPException(status_code=404, detail="Student not found")

        edwin = state.students[student_id]
        engagement = state.analytics.calculate_engagement(edwin, window_days=7)

        return LightweightStudentResponse(
            student_id=student_id,
            name=f"{edwin.student_model.first_name} {edwin.student_model.last_name}",
            grade_level=edwin.student_model.grade_level,
            mastered_count=len(edwin.student_model.mastered_objectives),
            engagement_score=engagement.engagement_score,
            last_synced=datetime.now().timestamp()
        )

    # ================== Quick Q&A ==================

    @app.post("/mobile/ask", response_model=QuickAnswerResponse)
    async def quick_question(request: QuickQuestionRequest):
        """Lightweight question answering (no sources)"""
        if request.student_id not in state.students:
            raise HTTPException(status_code=404, detail="Student not found")

        edwin = state.students[request.student_id]

        try:
            # Get answer
            response = await edwin.teach(request.question)

            # Log activity
            await state.analytics.track_interaction(
                student_id=request.student_id,
                interaction_type="question",
                objective_id=None,
                success=response.confidence > 0.75,
                confidence=response.confidence,
                mastered=False,
                xp_gained=10
            )

            return QuickAnswerResponse(
                answer=response.answer,
                confidence=response.confidence,
                timestamp=datetime.now().timestamp()
            )

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/mobile/ask-batch", response_model=BatchAnswerResponse)
    async def batch_questions(request: BatchQuestionRequest):
        """Batch multiple questions for efficiency"""
        if request.student_id not in state.students:
            raise HTTPException(status_code=404, detail="Student not found")

        edwin = state.students[request.student_id]
        start_time = datetime.now()

        try:
            # Process questions concurrently (up to 5 at a time)
            answers = []
            semaphore = asyncio.Semaphore(5)

            async def process_question(question: str) -> QuickAnswerResponse:
                async with semaphore:
                    response = await edwin.teach(question)
                    return QuickAnswerResponse(
                        answer=response.answer,
                        confidence=response.confidence,
                        timestamp=datetime.now().timestamp()
                    )

            # Process all questions
            tasks = [process_question(q) for q in request.questions]
            answers = await asyncio.gather(*tasks)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return BatchAnswerResponse(
                answers=answers,
                total=len(answers),
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"Error in batch questions: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ================== Progressive Loading ==================

    @app.post("/mobile/objectives", response_model=ProgressivePage)
    async def get_objectives_paginated(request: ProgressiveObjectivesRequest):
        """Get objectives in pages for progressive loading"""
        if request.student_id not in state.students:
            raise HTTPException(status_code=404, detail="Student not found")

        edwin = state.students[request.student_id]

        try:
            # Get recommended objectives
            all_objectives = await edwin.get_recommended_next(count=100)

            # Filter by subject if requested
            if request.subject_filter:
                all_objectives = [
                    obj for obj in all_objectives
                    if obj['objective_id'].startswith(request.subject_filter)
                ]

            # Paginate
            start_idx = (request.page - 1) * request.page_size
            end_idx = start_idx + request.page_size
            page_items = all_objectives[start_idx:end_idx]

            total_pages = (len(all_objectives) + request.page_size - 1) // request.page_size

            # Lightweight format
            lightweight_items = [
                {
                    "id": obj['objective_id'],
                    "title": obj['content'][:100],  # Truncate for mobile
                    "subject": obj['objective_id'].split('.')[0],
                    "grade": int(obj['objective_id'].split('.')[1])
                }
                for obj in page_items
            ]

            return ProgressivePage(
                items=lightweight_items,
                page=request.page,
                page_size=request.page_size,
                total_pages=total_pages,
                has_more=request.page < total_pages
            )

        except Exception as e:
            logger.error(f"Error getting objectives: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ================== Offline Sync ==================

    @app.post("/mobile/sync", response_model=SyncResult)
    async def offline_sync(request: OfflineSyncRequest):
        """Sync offline interactions when back online"""
        if request.student_id not in state.students:
            raise HTTPException(status_code=404, detail="Student not found")

        edwin = state.students[request.student_id]

        synced = 0
        failed = 0
        errors = []

        for interaction in request.interactions:
            try:
                interaction_type = interaction.get('type')

                if interaction_type == 'mastery_update':
                    # Update mastery
                    objective_id = interaction.get('objective_id')
                    success = interaction.get('success', True)
                    confidence = interaction.get('confidence', 0.8)

                    edwin.student_model.update_mastery(
                        objective_id=objective_id,
                        success=success,
                        confidence=confidence
                    )

                    # Track
                    await state.analytics.track_interaction(
                        student_id=request.student_id,
                        interaction_type="mastery_update",
                        objective_id=objective_id,
                        success=success,
                        confidence=confidence,
                        mastered=edwin.student_model.mastery_scores.get(objective_id, 0.0) > 0.8,
                        xp_gained=interaction.get('xp_gained', 0)
                    )

                    synced += 1

                elif interaction_type == 'question':
                    # Log question
                    await state.analytics.track_interaction(
                        student_id=request.student_id,
                        interaction_type="question",
                        objective_id=None,
                        success=True,
                        confidence=0.8,
                        mastered=False,
                        xp_gained=10
                    )
                    synced += 1

                else:
                    errors.append(f"Unknown interaction type: {interaction_type}")
                    failed += 1

            except Exception as e:
                logger.error(f"Error syncing interaction: {e}")
                errors.append(str(e))
                failed += 1

        # Save progress
        try:
            edwin.save_progress(f"./data/students/{request.student_id}_progress.json")
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")

        return SyncResult(
            synced_count=synced,
            failed_count=failed,
            errors=errors[:10],  # Limit error messages
            timestamp=datetime.now().timestamp()
        )

    # ================== Mobile Image Upload ==================

    @app.post("/mobile/upload-image")
    async def upload_image(
        student_id: str,
        question: str,
        file: UploadFile = File(...)
    ):
        """Mobile-optimized image upload with compression"""
        if student_id not in state.students:
            raise HTTPException(status_code=404, detail="Student not found")

        try:
            # Read image
            image_bytes = await file.read()

            # TODO: Compress image for mobile bandwidth
            # For now, save as-is
            image_path = f"./data/uploads/{student_id}_{datetime.now().timestamp()}.jpg"

            import os
            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            with open(image_path, 'wb') as f:
                f.write(image_bytes)

            # Process with multimodal tutor
            edwin = state.students[student_id]
            response = await state.multimodal_tutor.answer_with_image(
                question=question,
                image_path=image_path,
                student_model=edwin.student_model,
                mode="verify"
            )

            return QuickAnswerResponse(
                answer=response.answer,
                confidence=response.confidence,
                timestamp=datetime.now().timestamp()
            )

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ================== Development Server ==================

if __name__ == "__main__":
    import uvicorn

    app = create_mobile_app()

    # Run on different port (8002) from main API (8000) and dashboard (8001)
    uvicorn.run(app, host="0.0.0.0", port=8002)
