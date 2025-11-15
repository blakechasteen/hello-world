"""
EdWIN Teacher Dashboard - Real-time classroom oversight

Provides real-time monitoring of student progress, interventions, and analytics.
Uses WebSocket for live updates to teacher dashboard UI.

Features:
- Real-time student activity monitoring
- Live intervention alerts
- Classroom analytics dashboard
- Individual student deep-dives
- Knowledge gap heatmaps
- Engagement tracking

Author: EdWIN Team
Date: 2025-11-15
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import json
import logging
from pathlib import Path

from .analytics import (
    LearningAnalytics,
    LearningVelocity,
    KnowledgeGapAnalysis,
    EngagementMetrics,
    InterventionAlert
)
from .core import EdWIN
from .curriculum_graph import EdWINKnowledgeGraph
from .multimodal_tutor import EdWINMultimodalTutor

logger = logging.getLogger(__name__)


@dataclass
class StudentActivityEvent:
    """Real-time student activity event"""
    student_id: str
    student_name: str
    timestamp: float
    event_type: str  # "question", "mastery_update", "achievement"
    details: Dict[str, Any]


@dataclass
class ClassroomSnapshot:
    """Real-time classroom state snapshot"""
    timestamp: float
    total_students: int
    active_students: int  # Active in last 24h
    avg_engagement_score: float
    pending_interventions: int
    total_objectives_mastered: int
    avg_mastery_percentage: float
    subject_breakdown: Dict[str, int]  # Subject → mastered count


@dataclass
class StudentOverview:
    """Student overview for dashboard"""
    student_id: str
    name: str
    grade_level: int
    mastered_count: int
    engagement_score: float
    risk_level: str
    last_active: Optional[float]
    current_streak: int


class DashboardConnectionManager:
    """Manages WebSocket connections for real-time dashboard updates"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Dashboard connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"Dashboard disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected dashboards"""
        if not self.active_connections:
            return

        json_message = json.dumps(message)
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_text(json_message)
            except Exception as e:
                logger.warning(f"Failed to send to connection: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


class TeacherDashboard:
    """Real-time teacher dashboard for classroom oversight"""

    def __init__(
        self,
        curriculum_kg: EdWINKnowledgeGraph,
        multimodal_tutor: EdWINMultimodalTutor,
        analytics: LearningAnalytics
    ):
        self.curriculum_kg = curriculum_kg
        self.multimodal_tutor = multimodal_tutor
        self.analytics = analytics
        self.students: Dict[str, EdWIN] = {}  # student_id → EdWIN instance
        self.connection_manager = DashboardConnectionManager()
        self.activity_log: List[StudentActivityEvent] = []
        self.max_activity_log = 1000  # Keep last 1000 events

    # ================== Student Management ==================

    def add_student(self, student_id: str, edwin: EdWIN):
        """Register a student with the dashboard"""
        self.students[student_id] = edwin
        logger.info(f"Student {student_id} registered with dashboard")

    def remove_student(self, student_id: str):
        """Remove student from dashboard"""
        if student_id in self.students:
            del self.students[student_id]
            logger.info(f"Student {student_id} removed from dashboard")

    # ================== Real-time Events ==================

    async def log_activity(
        self,
        student_id: str,
        event_type: str,
        details: Dict[str, Any]
    ):
        """Log student activity and broadcast to dashboards"""
        if student_id not in self.students:
            return

        student = self.students[student_id]
        event = StudentActivityEvent(
            student_id=student_id,
            student_name=f"{student.student_model.first_name} {student.student_model.last_name}",
            timestamp=datetime.now().timestamp(),
            event_type=event_type,
            details=details
        )

        # Add to activity log
        self.activity_log.append(event)
        if len(self.activity_log) > self.max_activity_log:
            self.activity_log.pop(0)

        # Broadcast to connected dashboards
        await self.connection_manager.broadcast({
            "type": "student_activity",
            "data": asdict(event)
        })

    async def check_interventions(self):
        """Check for students needing intervention and broadcast alerts"""
        alerts = self.analytics.detect_interventions(
            list(self.students.values()),
            threshold_days=7
        )

        if alerts:
            await self.connection_manager.broadcast({
                "type": "interventions",
                "data": [asdict(alert) for alert in alerts]
            })

        return alerts

    # ================== Snapshot Generation ==================

    def get_classroom_snapshot(self) -> ClassroomSnapshot:
        """Get current classroom state snapshot"""
        if not self.students:
            return ClassroomSnapshot(
                timestamp=datetime.now().timestamp(),
                total_students=0,
                active_students=0,
                avg_engagement_score=0.0,
                pending_interventions=0,
                total_objectives_mastered=0,
                avg_mastery_percentage=0.0,
                subject_breakdown={}
            )

        # Calculate metrics
        now = datetime.now().timestamp()
        day_ago = now - 86400  # 24 hours

        active_count = 0
        total_engagement = 0.0
        total_mastered = 0
        subject_counts: Dict[str, int] = {}

        for student in self.students.values():
            # Check activity
            last_active = self._get_last_activity(student.student_id)
            if last_active and last_active > day_ago:
                active_count += 1

            # Engagement
            engagement = self.analytics.calculate_engagement(student, window_days=7)
            total_engagement += engagement.engagement_score

            # Mastery
            mastered = len(student.student_model.mastered_objectives)
            total_mastered += mastered

            # Subject breakdown
            for obj_id in student.student_model.mastered_objectives:
                # Extract subject from objective ID (format: SUBJECT.GRADE.NUMBER)
                subject = obj_id.split('.')[0]
                subject_counts[subject] = subject_counts.get(subject, 0) + 1

        # Interventions
        interventions = self.analytics.detect_interventions(
            list(self.students.values()),
            threshold_days=7
        )

        # Average mastery percentage
        total_objectives = len(self.curriculum_kg.kg.get_all_nodes())
        avg_mastery_pct = (total_mastered / (len(self.students) * total_objectives)) * 100 if total_objectives > 0 else 0.0

        return ClassroomSnapshot(
            timestamp=now,
            total_students=len(self.students),
            active_students=active_count,
            avg_engagement_score=total_engagement / len(self.students),
            pending_interventions=len(interventions),
            total_objectives_mastered=total_mastered,
            avg_mastery_percentage=avg_mastery_pct,
            subject_breakdown=subject_counts
        )

    def get_student_overviews(self) -> List[StudentOverview]:
        """Get overview of all students for dashboard"""
        overviews = []

        for student_id, student in self.students.items():
            engagement = self.analytics.calculate_engagement(student, window_days=7)
            velocity = self.analytics.calculate_learning_velocity(student, window_days=7)

            overviews.append(StudentOverview(
                student_id=student_id,
                name=f"{student.student_model.first_name} {student.student_model.last_name}",
                grade_level=student.student_model.grade_level,
                mastered_count=len(student.student_model.mastered_objectives),
                engagement_score=engagement.engagement_score,
                risk_level=engagement.risk_level,
                last_active=self._get_last_activity(student_id),
                current_streak=velocity.current_streak_days
            ))

        # Sort by risk level (high risk first) then engagement score
        risk_order = {"high": 0, "medium": 1, "low": 2}
        overviews.sort(key=lambda x: (risk_order[x.risk_level], -x.engagement_score))

        return overviews

    def get_student_detail(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed view of individual student"""
        if student_id not in self.students:
            return None

        student = self.students[student_id]

        # Calculate all metrics
        velocity = self.analytics.calculate_learning_velocity(student, window_days=7)
        gaps = self.analytics.analyze_knowledge_gaps(student)
        engagement = self.analytics.calculate_engagement(student, window_days=7)
        progress_report = self.analytics.generate_progress_report(student)

        return {
            "student_id": student_id,
            "name": f"{student.student_model.first_name} {student.student_model.last_name}",
            "grade_level": student.student_model.grade_level,
            "velocity": asdict(velocity),
            "gaps": asdict(gaps),
            "engagement": asdict(engagement),
            "progress": progress_report,
            "recent_activity": self._get_recent_activity(student_id, limit=10)
        }

    # ================== Helper Methods ==================

    def _get_last_activity(self, student_id: str) -> Optional[float]:
        """Get timestamp of student's last activity"""
        for event in reversed(self.activity_log):
            if event.student_id == student_id:
                return event.timestamp
        return None

    def _get_recent_activity(self, student_id: str, limit: int = 10) -> List[Dict]:
        """Get recent activity for student"""
        student_events = [
            asdict(event) for event in reversed(self.activity_log)
            if event.student_id == student_id
        ]
        return student_events[:limit]

    # ================== Background Tasks ==================

    async def start_monitoring(self, update_interval: float = 30.0):
        """Start background monitoring task"""
        logger.info(f"Starting dashboard monitoring (interval: {update_interval}s)")

        while True:
            try:
                # Generate snapshot
                snapshot = self.get_classroom_snapshot()

                # Broadcast snapshot
                await self.connection_manager.broadcast({
                    "type": "classroom_snapshot",
                    "data": asdict(snapshot)
                })

                # Check for interventions
                await self.check_interventions()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            await asyncio.sleep(update_interval)


# ================== FastAPI Integration ==================

def create_dashboard_app(
    curriculum_kg: EdWINKnowledgeGraph,
    multimodal_tutor: EdWINMultimodalTutor,
    analytics: LearningAnalytics
) -> tuple[FastAPI, TeacherDashboard]:
    """Create FastAPI app with teacher dashboard"""

    app = FastAPI(title="EdWIN Teacher Dashboard", version="3.0.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create dashboard
    dashboard = TeacherDashboard(curriculum_kg, multimodal_tutor, analytics)

    # Serve static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ================== HTTP Endpoints ==================

    @app.get("/", response_class=HTMLResponse)
    async def serve_dashboard():
        """Serve dashboard HTML"""
        html_path = Path(__file__).parent / "static" / "dashboard.html"
        if html_path.exists():
            return FileResponse(html_path)
        else:
            return HTMLResponse("<h1>EdWIN Teacher Dashboard</h1><p>Dashboard UI not found. See /docs for API.</p>")

    @app.get("/api/classroom/snapshot")
    async def get_snapshot():
        """Get current classroom snapshot"""
        snapshot = dashboard.get_classroom_snapshot()
        return asdict(snapshot)

    @app.get("/api/classroom/students")
    async def get_students():
        """Get overview of all students"""
        overviews = dashboard.get_student_overviews()
        return [asdict(o) for o in overviews]

    @app.get("/api/student/{student_id}")
    async def get_student_detail(student_id: str):
        """Get detailed view of student"""
        detail = dashboard.get_student_detail(student_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Student not found")
        return detail

    @app.get("/api/interventions")
    async def get_interventions():
        """Get current intervention alerts"""
        alerts = dashboard.analytics.detect_interventions(
            list(dashboard.students.values()),
            threshold_days=7
        )
        return [asdict(alert) for alert in alerts]

    @app.get("/api/activity")
    async def get_recent_activity(limit: int = 50):
        """Get recent classroom activity"""
        return [asdict(event) for event in dashboard.activity_log[-limit:]]

    # ================== WebSocket Endpoint ==================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time dashboard updates"""
        await dashboard.connection_manager.connect(websocket)

        try:
            # Send initial snapshot
            snapshot = dashboard.get_classroom_snapshot()
            await websocket.send_json({
                "type": "classroom_snapshot",
                "data": asdict(snapshot)
            })

            # Keep connection alive
            while True:
                # Wait for ping from client
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")

        except WebSocketDisconnect:
            dashboard.connection_manager.disconnect(websocket)

    # ================== Startup/Shutdown ==================

    @app.on_event("startup")
    async def startup():
        """Start background monitoring"""
        # Start monitoring task
        asyncio.create_task(dashboard.start_monitoring(update_interval=30.0))
        logger.info("Teacher dashboard started")

    return app, dashboard


if __name__ == "__main__":
    # Example usage
    import uvicorn
    from .curriculum_graph import EdWINKnowledgeGraph
    from .multimodal_tutor import EdWINMultimodalTutor
    from .analytics import LearningAnalytics

    # Initialize components
    curriculum_kg = EdWINKnowledgeGraph()
    multimodal_tutor = EdWINMultimodalTutor()
    analytics = LearningAnalytics()

    # Create app
    app, dashboard = create_dashboard_app(curriculum_kg, multimodal_tutor, analytics)

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8001)
