"""
Tests for EdWIN Multimodal Tutor (Phase 2)

Tests:
- Multimodal tutor initialization
- Visual Q&A functionality
- Video lesson ingestion
- Photo retrieval
- Graceful degradation

Author: EdWIN Team
Date: 2025-11-15
"""

import pytest
import asyncio
from pathlib import Path

from EduVerse.edwin.multimodal_tutor import (
    EdWINMultimodalTutor,
    MultimodalLearningResource,
    ResourceType
)
from EduVerse.edwin.student_model import EdWINStudentModel


class TestMultimodalTutor:
    """Tests for multimodal tutor"""

    @pytest.fixture
    async def tutor(self):
        """Create multimodal tutor"""
        tutor = EdWINMultimodalTutor()
        await tutor.initialize()
        return tutor

    @pytest.fixture
    def student_model(self):
        """Create student model"""
        return EdWINStudentModel(
            first_name="Test",
            last_name="Student",
            grade_level=8
        )

    @pytest.mark.asyncio
    async def test_initialization(self, tutor):
        """Test tutor initializes correctly"""
        assert tutor is not None
        assert tutor.config is not None

    @pytest.mark.asyncio
    async def test_visual_qa_fallback(self, tutor, student_model):
        """Test visual Q&A with graceful fallback"""
        # Should not crash even without image
        response = await tutor.answer_with_image(
            question="What is this?",
            image_path="nonexistent.jpg",
            student_model=student_model,
            mode="direct"
        )

        assert response is not None
        assert isinstance(response.answer, str)
        assert 0.0 <= response.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_ingest_video_lesson(self, tutor):
        """Test video lesson ingestion"""
        # Test with invalid URL (should handle gracefully)
        try:
            await tutor.ingest_video_lesson(
                video_url="invalid_url",
                objective_ids=["MATH.8.1"],
                title="Test Video",
                grade_level=8
            )
        except Exception as e:
            # Should handle error gracefully
            assert "video" in str(e).lower() or "url" in str(e).lower()

    @pytest.mark.asyncio
    async def test_ingest_diagram(self, tutor):
        """Test diagram ingestion"""
        # Should handle missing image gracefully
        try:
            await tutor.ingest_diagram(
                image_path="missing.jpg",
                objective_ids=["SCIENCE.9.1"],
                title="Test Diagram",
                caption="Test caption",
                grade_level=9
            )
        except Exception as e:
            # Should provide meaningful error
            assert "image" in str(e).lower() or "file" in str(e).lower()

    @pytest.mark.asyncio
    async def test_get_visual_resources(self, tutor):
        """Test visual resource retrieval"""
        resources = await tutor.get_visual_resources(
            objective_id="MATH.8.1",
            resource_type=ResourceType.DIAGRAM
        )

        assert isinstance(resources, list)
        # May be empty if no resources ingested
        for resource in resources:
            assert isinstance(resource, MultimodalLearningResource)
            assert resource.resource_type == ResourceType.DIAGRAM

    def test_multimodal_resource_creation(self):
        """Test multimodal resource dataclass"""
        resource = MultimodalLearningResource(
            resource_id="test_001",
            resource_type=ResourceType.DIAGRAM,
            title="Test Diagram",
            description="Test description",
            url="https://example.com/diagram.png",
            related_objectives=["MATH.8.1"],
            grade_level=8
        )

        assert resource.resource_id == "test_001"
        assert resource.resource_type == ResourceType.DIAGRAM
        assert resource.title == "Test Diagram"
        assert "MATH.8.1" in resource.related_objectives

    def test_resource_type_enum(self):
        """Test resource type enumeration"""
        assert ResourceType.DIAGRAM.value == "diagram"
        assert ResourceType.VIDEO.value == "video"
        assert ResourceType.INTERACTIVE.value == "interactive"
        assert ResourceType.PHOTO.value == "photo"


class TestGracefulDegradation:
    """Tests for graceful degradation without multimodal dependencies"""

    @pytest.mark.asyncio
    async def test_no_multimodal_dependencies(self):
        """Test tutor works without multimodal dependencies"""
        # Create tutor
        tutor = EdWINMultimodalTutor()

        # Should initialize even without dependencies
        await tutor.initialize()

        assert tutor is not None

    @pytest.mark.asyncio
    async def test_fallback_to_text_only(self):
        """Test fallback to text-only Q&A"""
        tutor = EdWINMultimodalTutor()
        await tutor.initialize()

        student = EdWINStudentModel("Test", "Student", 8)

        # Should fall back to text-only answer
        response = await tutor.answer_with_image(
            question="What is 2 + 2?",
            image_path="missing.jpg",
            student_model=student,
            mode="direct"
        )

        assert response.answer
        # Should still provide reasonable answer
        assert "4" in response.answer or "four" in response.answer.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
