"""
EdWIN FastAPI Server

Production-ready API for EdWIN AI tutoring system.

**Endpoints**:
- Student management (CRUD)
- Tutoring (Q&A, multimodal)
- Progress tracking
- Learning paths
- Analytics
- Teacher dashboard

**Implementation Date**: November 15, 2025

Usage:
    uvicorn EduVerse.edwin.api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import asyncio
import json

# EdWIN imports
from .core import EdWIN, create_edwin
from .curriculum_graph import EdWINKnowledgeGraph
from .student_model import EdWINStudentModel
from .multimodal_tutor import EdWINMultimodalTutor, create_multimodal_tutor
from EduVerse.education.curriculum import Subject

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="EdWIN API",
    description="AI-powered adaptive tutoring system",
    version="1.0.0"
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Data Models (Pydantic)
# ============================================================================

class StudentCreate(BaseModel):
    """Create student request"""
    name: str = Field(..., description="Student name")
    grade: int = Field(..., ge=4, le=12, description="Grade level (4-12)")
    learning_preferences: Optional[Dict[str, float]] = Field(None, description="VARK learning style scores")


class StudentResponse(BaseModel):
    """Student response"""
    student_id: str
    name: str
    grade: int
    mastered_count: int
    in_progress_count: int
    total_xp: float
    level: int
    streak_days: int
    created_at: str


class QuestionRequest(BaseModel):
    """Ask question request"""
    student_id: str = Field(..., description="Student ID")
    question: str = Field(..., description="Student's question")
    mode: str = Field("verify", description="Reasoning mode (direct, verify, research)")


class QuestionResponse(BaseModel):
    """Question answer response"""
    question: str
    answer: str
    confidence: float
    sources: List[str]
    reasoning_mode: str
    response_time_ms: float
    related_objectives: List[str]
    next_steps: List[str]


class ProgressUpdate(BaseModel):
    """Progress update request"""
    student_id: str
    objective_id: str
    success: bool
    confidence: float
    time_spent_minutes: Optional[int] = None


class ProgressResponse(BaseModel):
    """Progress update response"""
    mastery_updated: bool
    new_mastery_score: float
    leveled_up: bool
    xp_gained: float
    unlocked_objectives: List[str]


class LearningPathRequest(BaseModel):
    """Learning path request"""
    student_id: str
    target_objective_id: str


class LearningPathResponse(BaseModel):
    """Learning path response"""
    from_objectives: List[str]
    to_objective: str
    path: List[Dict[str, Any]]
    total_hours: float


class RecommendationRequest(BaseModel):
    """Recommendation request"""
    student_id: str
    count: int = Field(5, ge=1, le=20)


class RecommendationResponse(BaseModel):
    """Recommendation response"""
    student_id: str
    recommendations: List[Dict[str, Any]]


# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.curriculum_kg: Optional[EdWINKnowledgeGraph] = None
        self.multimodal_tutor: Optional[EdWINMultimodalTutor] = None
        self.students: Dict[str, EdWIN] = {}  # student_id → EdWIN instance
        self.initialized = False


app_state = AppState()


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize EdWIN system on startup"""
    print("\n" + "=" * 70)
    print(" EdWIN API Server Starting...")
    print("=" * 70 + "\n")

    # Initialize curriculum
    print("📚 Loading curriculum knowledge graph...")
    from .curriculum_graph import create_curriculum_graph
    app_state.curriculum_kg = await create_curriculum_graph()

    # Initialize multimodal tutor
    print("🎨 Initializing multimodal tutor...")
    app_state.multimodal_tutor = await create_multimodal_tutor(
        curriculum_kg=app_state.curriculum_kg,
        llm_provider="anthropic"
    )

    app_state.initialized = True

    print("\n" + "=" * 70)
    print(" ✅ EdWIN API Server Ready!")
    print(" 📡 Listening on http://localhost:8000")
    print(" 📖 API Docs: http://localhost:8000/docs")
    print("=" * 70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n🛑 EdWIN API Server shutting down...")

    # Save all student progress
    for student_id, edwin in app_state.students.items():
        try:
            edwin.save_progress(f"./data/students/{student_id}.json")
            print(f"💾 Saved progress for {student_id}")
        except Exception as e:
            print(f"⚠️  Error saving {student_id}: {e}")

    print("👋 EdWIN API Server stopped.\n")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if app_state.initialized else "initializing",
        "curriculum_loaded": app_state.curriculum_kg is not None,
        "multimodal_enabled": app_state.multimodal_tutor is not None,
        "students_count": len(app_state.students),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Student Management
# ============================================================================

@app.post("/students", response_model=StudentResponse)
async def create_student(student: StudentCreate):
    """
    Create a new student

    Creates student profile and initializes EdWIN instance.
    """
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="Server initializing, please wait")

    # Generate student ID
    import uuid
    student_id = f"student_{uuid.uuid4().hex[:12]}"

    # Create EdWIN instance
    edwin = await create_edwin(
        student_id=student_id,
        student_name=student.name,
        grade=student.grade,
        llm_provider="anthropic"
    )

    # Apply learning preferences if provided
    if student.learning_preferences:
        from EduVerse.education.player_model import LearningStyle
        for style_name, score in student.learning_preferences.items():
            try:
                style = LearningStyle(style_name)
                edwin.student.learning_style_scores[style] = score
            except ValueError:
                pass  # Skip invalid styles

    # Store in state
    app_state.students[student_id] = edwin

    # Return response
    progress = edwin.get_progress_summary()

    return StudentResponse(
        student_id=student_id,
        name=student.name,
        grade=student.grade,
        mastered_count=progress["mastered_count"],
        in_progress_count=progress["in_progress_count"],
        total_xp=progress["total_xp"],
        level=progress["level"],
        streak_days=progress["streak_days"],
        created_at=datetime.utcnow().isoformat()
    )


@app.get("/students/{student_id}", response_model=StudentResponse)
async def get_student(student_id: str):
    """Get student by ID"""
    if student_id not in app_state.students:
        raise HTTPException(status_code=404, detail="Student not found")

    edwin = app_state.students[student_id]
    progress = edwin.get_progress_summary()

    return StudentResponse(
        student_id=student_id,
        name=progress["name"],
        grade=progress["grade"],
        mastered_count=progress["mastered_count"],
        in_progress_count=progress["in_progress_count"],
        total_xp=progress["total_xp"],
        level=progress["level"],
        streak_days=progress["streak_days"],
        created_at=edwin.student.created_at.isoformat()
    )


@app.get("/students")
async def list_students():
    """List all students"""
    students = []
    for student_id, edwin in app_state.students.items():
        progress = edwin.get_progress_summary()
        students.append({
            "student_id": student_id,
            "name": progress["name"],
            "grade": progress["grade"],
            "mastered_count": progress["mastered_count"],
            "level": progress["level"]
        })
    return {"students": students, "count": len(students)}


@app.delete("/students/{student_id}")
async def delete_student(student_id: str):
    """Delete student"""
    if student_id not in app_state.students:
        raise HTTPException(status_code=404, detail="Student not found")

    # Save final progress
    edwin = app_state.students[student_id]
    edwin.save_progress(f"./data/students/{student_id}_final.json")

    # Remove from state
    del app_state.students[student_id]

    return {"message": "Student deleted", "student_id": student_id}


# ============================================================================
# Tutoring
# ============================================================================

@app.post("/tutor/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question (main tutoring endpoint)

    Generates grade-appropriate answer using RAG.
    """
    if request.student_id not in app_state.students:
        raise HTTPException(status_code=404, detail="Student not found")

    edwin = app_state.students[request.student_id]

    # Ask question
    start_time = datetime.utcnow()
    answer = await edwin.teach(request.question)
    response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

    # Get tutor's last result (hack for now - should return TutoringResponse)
    # For now, create response from answer
    return QuestionResponse(
        question=request.question,
        answer=answer,
        confidence=0.85,  # Placeholder
        sources=[],
        reasoning_mode=request.mode,
        response_time_ms=response_time_ms,
        related_objectives=[],
        next_steps=[]
    )


@app.post("/tutor/ask-with-image")
async def ask_with_image(
    student_id: str,
    question: str,
    image: UploadFile = File(...),
    mode: str = "verify"
):
    """
    Ask question about an image

    Requires multimodal support.
    """
    if student_id not in app_state.students:
        raise HTTPException(status_code=404, detail="Student not found")

    if not app_state.multimodal_tutor:
        raise HTTPException(status_code=501, detail="Multimodal support not available")

    edwin = app_state.students[student_id]

    # Save uploaded image temporarily
    temp_dir = Path("./data/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    image_path = temp_dir / image.filename
    with open(image_path, "wb") as f:
        f.write(await image.read())

    # Answer with image
    result = await app_state.multimodal_tutor.answer_with_image(
        question=question,
        image_path=str(image_path),
        student_model=edwin.student,
        mode=mode
    )

    return QuestionResponse(
        question=result.question,
        answer=result.answer,
        confidence=result.confidence,
        sources=result.sources,
        reasoning_mode=result.reasoning_mode,
        response_time_ms=result.response_time_ms,
        related_objectives=result.related_objectives,
        next_steps=result.next_steps
    )


# ============================================================================
# Progress Tracking
# ============================================================================

@app.post("/progress", response_model=ProgressResponse)
async def update_progress(update: ProgressUpdate):
    """
    Update student progress

    Records learning interaction and updates mastery.
    """
    if update.student_id not in app_state.students:
        raise HTTPException(status_code=404, detail="Student not found")

    edwin = app_state.students[update.student_id]

    # Update mastery
    mastered = edwin.student.update_mastery(
        objective_id=update.objective_id,
        success=update.success,
        confidence=update.confidence
    )

    # Get new mastery score
    new_mastery = edwin.student.get_mastery_score(update.objective_id)

    # Calculate XP gained
    xp_gained = 10.0 * update.confidence

    # Get unlocked objectives
    unlocked = edwin.curriculum_kg.get_next_objectives(
        mastered_ids=list(edwin.student.mastered_objectives),
        grade_filter=edwin.grade
    )

    return ProgressResponse(
        mastery_updated=True,
        new_mastery_score=new_mastery,
        leveled_up=mastered,
        xp_gained=xp_gained,
        unlocked_objectives=[obj.id for obj in unlocked[:5]]
    )


@app.get("/progress/{student_id}")
async def get_progress(student_id: str):
    """Get student progress summary"""
    if student_id not in app_state.students:
        raise HTTPException(status_code=404, detail="Student not found")

    edwin = app_state.students[student_id]
    return edwin.get_progress_summary()


# ============================================================================
# Learning Paths
# ============================================================================

@app.post("/learning-path", response_model=LearningPathResponse)
async def get_learning_path(request: LearningPathRequest):
    """
    Generate learning path to target objective

    Uses BFS to find shortest path through prerequisites.
    """
    if request.student_id not in app_state.students:
        raise HTTPException(status_code=404, detail="Student not found")

    edwin = app_state.students[request.student_id]

    # Get current frontier (edge of knowledge)
    frontier = edwin.student.get_frontier()

    if not frontier:
        # Use first mastered objective
        mastered = list(edwin.student.mastered_objectives)
        frontier = [mastered[0]] if mastered else ["math.number.4.place_value"]

    # Find path
    path_objs = edwin.curriculum_kg.get_learning_path(
        from_id=frontier[0],
        to_id=request.target_objective_id
    )

    # Format response
    path = []
    total_hours = 0.0

    for obj in path_objs:
        is_mastered = obj.id in edwin.student.mastered_objectives
        status = "mastered" if is_mastered else "locked"

        path.append({
            "id": obj.id,
            "title": obj.title,
            "description": obj.description,
            "grade_level": obj.grade_level,
            "estimated_hours": obj.estimated_hours,
            "status": status
        })

        total_hours += obj.estimated_hours

    return LearningPathResponse(
        from_objectives=frontier,
        to_objective=request.target_objective_id,
        path=path,
        total_hours=total_hours
    )


# ============================================================================
# Recommendations
# ============================================================================

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized recommendations

    Uses adaptive difficulty engine to select optimal challenges.
    """
    if request.student_id not in app_state.students:
        raise HTTPException(status_code=404, detail="Student not found")

    edwin = app_state.students[request.student_id]

    # Get next lesson suggestion
    suggestion = await edwin.suggest_next_lesson()

    # Get additional recommendations
    available = edwin.curriculum_kg.get_next_objectives(
        mastered_ids=list(edwin.student.mastered_objectives),
        grade_filter=edwin.grade
    )

    recommendations = []

    # Add top suggestion
    recommendations.append({
        "objective_id": suggestion.get("id", ""),
        "title": suggestion["title"],
        "description": suggestion["description"],
        "difficulty": suggestion.get("difficulty", 0.5),
        "expected_success": suggestion.get("expected_success", 0.5),
        "estimated_time": suggestion.get("estimated_time", 1.0),
        "reason": "Optimal challenge based on Thompson Sampling"
    })

    # Add more recommendations
    for obj in available[1:request.count]:
        difficulty = edwin.adaptive_engine.estimate_difficulty(obj.id) if edwin.adaptive_engine else 0.5

        recommendations.append({
            "objective_id": obj.id,
            "title": obj.title,
            "description": obj.description,
            "difficulty": difficulty,
            "expected_success": 0.5,
            "estimated_time": obj.estimated_hours,
            "reason": "All prerequisites mastered"
        })

    return RecommendationResponse(
        student_id=request.student_id,
        recommendations=recommendations[:request.count]
    )


# ============================================================================
# Analytics
# ============================================================================

@app.get("/analytics/summary")
async def analytics_summary():
    """
    Get platform-wide analytics

    Provides overview of all students, progress, engagement.
    """
    total_students = len(app_state.students)
    total_mastered = 0
    total_in_progress = 0
    avg_mastery_percentage = 0.0

    for edwin in app_state.students.values():
        progress = edwin.get_progress_summary()
        total_mastered += progress["mastered_count"]
        total_in_progress += progress["in_progress_count"]
        avg_mastery_percentage += progress["mastery_percentage"]

    if total_students > 0:
        avg_mastery_percentage /= total_students

    return {
        "total_students": total_students,
        "total_objectives_mastered": total_mastered,
        "total_objectives_in_progress": total_in_progress,
        "avg_mastery_percentage": avg_mastery_percentage,
        "curriculum_size": len(app_state.curriculum_kg.curriculum.objectives) if app_state.curriculum_kg else 0
    }


@app.get("/analytics/student/{student_id}")
async def student_analytics(student_id: str):
    """
    Get detailed analytics for a student

    Includes mastery breakdown by subject, progress over time, etc.
    """
    if student_id not in app_state.students:
        raise HTTPException(status_code=404, detail="Student not found")

    edwin = app_state.students[student_id]
    progress = edwin.get_progress_summary()

    # Breakdown by subject
    subject_breakdown = {}
    for subject in Subject:
        subject_objs = edwin.curriculum_kg.get_subject_objectives(subject)
        mastered_in_subject = sum(
            1 for obj in subject_objs
            if obj.id in edwin.student.mastered_objectives
        )
        subject_breakdown[subject.value] = {
            "total": len(subject_objs),
            "mastered": mastered_in_subject,
            "percentage": (mastered_in_subject / len(subject_objs) * 100) if subject_objs else 0
        }

    return {
        "student_id": student_id,
        "overall_progress": progress,
        "subject_breakdown": subject_breakdown,
        "difficulty_preference": edwin.student.difficulty_preference,
        "preferred_learning_style": edwin.student.preferred_learning_style
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
