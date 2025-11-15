"""
Tests for EdWIN FastAPI Server (Phase 2)

Tests:
- Student management endpoints
- Tutoring endpoints
- Progress tracking
- Learning paths
- Recommendations
- Analytics endpoints

Requires API server running for integration tests.

Author: EdWIN Team
Date: 2025-11-15
"""

import pytest
from fastapi.testclient import TestClient

# Unit tests (without running server)


class TestAPIModels:
    """Tests for API request/response models"""

    def test_student_create_model(self):
        """Test student creation model validation"""
        from EduVerse.edwin.api import StudentCreate

        # Valid student
        student = StudentCreate(
            first_name="John",
            last_name="Doe",
            grade_level=8
        )

        assert student.first_name == "John"
        assert student.last_name == "Doe"
        assert student.grade_level == 8

    def test_question_request_model(self):
        """Test question request model"""
        from EduVerse.edwin.api import QuestionRequest

        request = QuestionRequest(
            student_id="test.student.1",
            question="What is 2 + 2?",
            mode="verify"
        )

        assert request.student_id == "test.student.1"
        assert request.question == "What is 2 + 2?"
        assert request.mode == "verify"

    def test_progress_update_model(self):
        """Test progress update model"""
        from EduVerse.edwin.api import ProgressUpdate

        update = ProgressUpdate(
            student_id="test.student.1",
            objective_id="MATH.8.1",
            success=True,
            confidence=0.95
        )

        assert update.success is True
        assert update.confidence == 0.95

    def test_recommendation_request_model(self):
        """Test recommendation request model"""
        from EduVerse.edwin.api import RecommendationRequest

        request = RecommendationRequest(
            student_id="test.student.1",
            count=5
        )

        assert request.count == 5


class TestAPIValidation:
    """Tests for API input validation"""

    def test_invalid_grade_level(self):
        """Test grade level validation"""
        from EduVerse.edwin.api import StudentCreate
        from pydantic import ValidationError

        # Grade level too low
        with pytest.raises(ValidationError):
            StudentCreate(
                first_name="John",
                last_name="Doe",
                grade_level=0  # Invalid
            )

        # Grade level too high
        with pytest.raises(ValidationError):
            StudentCreate(
                first_name="John",
                last_name="Doe",
                grade_level=13  # Invalid
            )

    def test_empty_question(self):
        """Test empty question validation"""
        from EduVerse.edwin.api import QuestionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QuestionRequest(
                student_id="test.student.1",
                question="",  # Empty not allowed
                mode="direct"
            )

    def test_recommendation_count_limits(self):
        """Test recommendation count validation"""
        from EduVerse.edwin.api import RecommendationRequest
        from pydantic import ValidationError

        # Count too low
        with pytest.raises(ValidationError):
            RecommendationRequest(
                student_id="test.student.1",
                count=0  # Invalid
            )

        # Count too high
        with pytest.raises(ValidationError):
            RecommendationRequest(
                student_id="test.student.1",
                count=21  # Invalid (max 20)
            )


# Integration tests (require running server)


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from EduVerse.edwin.api import create_app

        app = create_app()
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_create_student(self, client):
        """Test student creation"""
        response = client.post(
            "/students",
            json={
                "first_name": "Test",
                "last_name": "Student",
                "grade_level": 8
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "student_id" in data
        assert data["name"] == "Test Student"
        assert data["grade_level"] == 8

    def test_ask_question(self, client):
        """Test question answering"""
        # First create student
        create_response = client.post(
            "/students",
            json={"first_name": "Test", "last_name": "Student", "grade_level": 8}
        )
        student_id = create_response.json()["student_id"]

        # Ask question
        response = client.post(
            "/tutor/ask",
            json={
                "student_id": student_id,
                "question": "What is 2 + 2?",
                "mode": "direct"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_get_nonexistent_student(self, client):
        """Test getting nonexistent student"""
        response = client.get("/students/nonexistent")

        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
