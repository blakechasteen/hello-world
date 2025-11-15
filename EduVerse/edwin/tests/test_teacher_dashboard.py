"""
Tests for EdWIN Teacher Dashboard (Phase 3)

Tests:
- Dashboard connection management
- Classroom snapshot generation
- Student overview creation
- Intervention detection
- Real-time event broadcasting

Author: EdWIN Team
Date: 2025-11-15
"""

import pytest
import asyncio
from datetime import datetime

from EduVerse.edwin.teacher_dashboard import (
    TeacherDashboard,
    DashboardConnectionManager,
    StudentActivityEvent,
    ClassroomSnapshot,
    StudentOverview
)
from EduVerse.edwin.curriculum_graph import EdWINKnowledgeGraph
from EduVerse.edwin.multimodal_tutor import EdWINMultimodalTutor
from EduVerse.edwin.analytics import LearningAnalytics
from EduVerse.edwin.core import EdWIN


class TestDashboardConnectionManager:
    """Tests for WebSocket connection management"""

    @pytest.fixture
    def manager(self):
        """Create connection manager"""
        return DashboardConnectionManager()

    def test_connection_manager_initialization(self, manager):
        """Test manager initializes correctly"""
        assert manager is not None
        assert len(manager.active_connections) == 0

    def test_disconnect(self, manager):
        """Test connection disconnection"""
        # Mock WebSocket
        class MockWebSocket:
            pass

        ws = MockWebSocket()
        manager.active_connections.add(ws)
        assert len(manager.active_connections) == 1

        manager.disconnect(ws)
        assert len(manager.active_connections) == 0


class TestTeacherDashboard:
    """Tests for teacher dashboard functionality"""

    @pytest.fixture
    async def dashboard(self):
        """Create dashboard instance"""
        curriculum_kg = EdWINKnowledgeGraph()
        await curriculum_kg.ingest_curriculum()

        multimodal_tutor = EdWINMultimodalTutor()
        await multimodal_tutor.initialize()

        analytics = LearningAnalytics()

        return TeacherDashboard(curriculum_kg, multimodal_tutor, analytics)

    @pytest.mark.asyncio
    async def test_dashboard_initialization(self, dashboard):
        """Test dashboard initializes correctly"""
        assert dashboard is not None
        assert dashboard.curriculum_kg is not None
        assert dashboard.multimodal_tutor is not None
        assert dashboard.analytics is not None
        assert len(dashboard.students) == 0
        assert len(dashboard.activity_log) == 0

    @pytest.mark.asyncio
    async def test_add_student(self, dashboard):
        """Test adding student to dashboard"""
        student = EdWIN("Test", "Student", 8)
        await student.initialize(
            curriculum_kg=dashboard.curriculum_kg,
            multimodal_tutor=dashboard.multimodal_tutor
        )

        dashboard.add_student("test.student.1", student)

        assert "test.student.1" in dashboard.students
        assert dashboard.students["test.student.1"] == student

    @pytest.mark.asyncio
    async def test_remove_student(self, dashboard):
        """Test removing student from dashboard"""
        student = EdWIN("Test", "Student", 8)
        await student.initialize(
            curriculum_kg=dashboard.curriculum_kg,
            multimodal_tutor=dashboard.multimodal_tutor
        )

        dashboard.add_student("test.student.1", student)
        dashboard.remove_student("test.student.1")

        assert "test.student.1" not in dashboard.students

    @pytest.mark.asyncio
    async def test_log_activity(self, dashboard):
        """Test activity logging"""
        student = EdWIN("Test", "Student", 8)
        await student.initialize(
            curriculum_kg=dashboard.curriculum_kg,
            multimodal_tutor=dashboard.multimodal_tutor
        )

        dashboard.add_student("test.student.1", student)

        await dashboard.log_activity(
            student_id="test.student.1",
            event_type="question",
            details={"question": "What is 2 + 2?"}
        )

        assert len(dashboard.activity_log) == 1
        event = dashboard.activity_log[0]
        assert event.student_id == "test.student.1"
        assert event.event_type == "question"
        assert event.details["question"] == "What is 2 + 2?"

    @pytest.mark.asyncio
    async def test_classroom_snapshot(self, dashboard):
        """Test classroom snapshot generation"""
        # Add some students
        for i in range(3):
            student = EdWIN(f"Student{i}", "Test", 8)
            await student.initialize(
                curriculum_kg=dashboard.curriculum_kg,
                multimodal_tutor=dashboard.multimodal_tutor
            )
            dashboard.add_student(f"student{i}", student)

        snapshot = dashboard.get_classroom_snapshot()

        assert isinstance(snapshot, ClassroomSnapshot)
        assert snapshot.total_students == 3
        assert snapshot.active_students >= 0
        assert 0.0 <= snapshot.avg_engagement_score <= 1.0
        assert snapshot.pending_interventions >= 0
        assert snapshot.total_objectives_mastered >= 0

    @pytest.mark.asyncio
    async def test_student_overviews(self, dashboard):
        """Test student overview generation"""
        # Add student
        student = EdWIN("Test", "Student", 8)
        await student.initialize(
            curriculum_kg=dashboard.curriculum_kg,
            multimodal_tutor=dashboard.multimodal_tutor
        )

        # Simulate some mastery
        student.student_model.mastered_objectives = {"MATH.8.1", "MATH.8.2"}

        dashboard.add_student("test.student.1", student)

        overviews = dashboard.get_student_overviews()

        assert len(overviews) == 1
        overview = overviews[0]
        assert isinstance(overview, StudentOverview)
        assert overview.student_id == "test.student.1"
        assert overview.name == "Test Student"
        assert overview.grade_level == 8
        assert overview.mastered_count == 2

    @pytest.mark.asyncio
    async def test_student_detail(self, dashboard):
        """Test detailed student view"""
        student = EdWIN("Test", "Student", 8)
        await student.initialize(
            curriculum_kg=dashboard.curriculum_kg,
            multimodal_tutor=dashboard.multimodal_tutor
        )

        dashboard.add_student("test.student.1", student)

        detail = dashboard.get_student_detail("test.student.1")

        assert detail is not None
        assert detail["student_id"] == "test.student.1"
        assert detail["name"] == "Test Student"
        assert "velocity" in detail
        assert "gaps" in detail
        assert "engagement" in detail
        assert "progress" in detail


class TestStudentActivityEvent:
    """Tests for student activity events"""

    def test_activity_event_creation(self):
        """Test activity event dataclass"""
        event = StudentActivityEvent(
            student_id="test.student.1",
            student_name="Test Student",
            timestamp=datetime.now().timestamp(),
            event_type="question",
            details={"question": "Test question"}
        )

        assert event.student_id == "test.student.1"
        assert event.student_name == "Test Student"
        assert event.event_type == "question"
        assert "question" in event.details


class TestClassroomSnapshot:
    """Tests for classroom snapshot"""

    def test_snapshot_creation(self):
        """Test snapshot dataclass"""
        snapshot = ClassroomSnapshot(
            timestamp=datetime.now().timestamp(),
            total_students=25,
            active_students=20,
            avg_engagement_score=0.75,
            pending_interventions=2,
            total_objectives_mastered=150,
            avg_mastery_percentage=45.5,
            subject_breakdown={"MATH": 50, "SCIENCE": 40, "READING": 60}
        )

        assert snapshot.total_students == 25
        assert snapshot.active_students == 20
        assert snapshot.avg_engagement_score == 0.75
        assert snapshot.pending_interventions == 2
        assert "MATH" in snapshot.subject_breakdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
