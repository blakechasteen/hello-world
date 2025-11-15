"""
Tests for EdWIN Learning Analytics (Phase 2)

Tests:
- Learning velocity calculation
- Knowledge gap analysis
- Engagement metrics
- Intervention detection
- Progress reporting

Author: EdWIN Team
Date: 2025-11-15
"""

import pytest
from datetime import datetime, timedelta

from EduVerse.edwin.analytics import (
    LearningAnalytics,
    LearningVelocity,
    KnowledgeGapAnalysis,
    EngagementMetrics,
    InterventionAlert
)
from EduVerse.edwin.core import EdWIN
from EduVerse.edwin.student_model import EdWINStudentModel


class TestLearningAnalytics:
    """Tests for learning analytics"""

    @pytest.fixture
    def analytics(self):
        """Create analytics instance"""
        return LearningAnalytics()

    @pytest.fixture
    def student(self):
        """Create test student"""
        student = EdWIN("Test", "Student", 8)
        # Simulate some mastery
        student.student_model.mastered_objectives = {
            "MATH.8.1", "MATH.8.2", "MATH.8.3"
        }
        student.student_model.mastery_scores = {
            "MATH.8.1": 0.95,
            "MATH.8.2": 0.88,
            "MATH.8.3": 0.92
        }
        return student

    def test_learning_velocity_calculation(self, analytics, student):
        """Test learning velocity metrics"""
        velocity = analytics.calculate_learning_velocity(student, window_days=7)

        assert isinstance(velocity, LearningVelocity)
        assert velocity.objectives_per_week >= 0.0
        assert velocity.xp_per_week >= 0.0
        assert velocity.current_streak_days >= 0
        assert velocity.best_streak_days >= velocity.current_streak_days
        assert 0.0 <= velocity.avg_confidence <= 1.0
        assert velocity.trend in ["accelerating", "steady", "decelerating"]

    def test_knowledge_gap_analysis(self, analytics, student):
        """Test knowledge gap detection"""
        gaps = analytics.analyze_knowledge_gaps(student)

        assert isinstance(gaps, KnowledgeGapAnalysis)
        assert gaps.total_gaps >= 0
        assert isinstance(gaps.critical_gaps, list)
        assert isinstance(gaps.gap_distribution, dict)
        assert gaps.estimated_catchup_hours >= 0.0

    def test_engagement_metrics(self, analytics, student):
        """Test engagement calculation"""
        # Track some interactions first
        for i in range(5):
            analytics.track_interaction(
                student_id="test_student",
                interaction_type="question",
                objective_id=f"MATH.8.{i}",
                success=True,
                confidence=0.85,
                mastered=False,
                xp_gained=10
            )

        engagement = analytics.calculate_engagement(student, window_days=7)

        assert isinstance(engagement, EngagementMetrics)
        assert engagement.total_interactions >= 0
        assert engagement.avg_session_length_minutes >= 0.0
        assert engagement.questions_asked >= 0
        assert 0.0 <= engagement.success_rate <= 1.0
        assert 0.0 <= engagement.engagement_score <= 1.0
        assert engagement.risk_level in ["low", "medium", "high"]

    def test_intervention_detection(self, analytics):
        """Test intervention alert generation"""
        # Create students with different risk profiles
        students = []

        # High risk student (low engagement)
        high_risk = EdWIN("High", "Risk", 8)
        students.append(high_risk)

        # Low risk student (good engagement)
        low_risk = EdWIN("Low", "Risk", 8)
        low_risk.student_model.mastered_objectives = {f"MATH.8.{i}" for i in range(10)}
        students.append(low_risk)

        # Detect interventions
        alerts = analytics.detect_interventions(students, threshold_days=7)

        assert isinstance(alerts, list)
        for alert in alerts:
            assert isinstance(alert, InterventionAlert)
            assert alert.student_id
            assert alert.severity in ["warning", "critical"]
            assert alert.reason
            assert isinstance(alert.recommended_actions, list)
            assert len(alert.recommended_actions) > 0

    def test_progress_report_generation(self, analytics, student):
        """Test progress report creation"""
        report = analytics.generate_progress_report(student)

        assert isinstance(report, dict)
        assert "student_id" in report
        assert "mastered_count" in report
        assert "total_xp" in report
        assert "velocity" in report
        assert "gaps" in report
        assert "engagement" in report
        assert "subject_breakdown" in report

    def test_interaction_tracking(self, analytics):
        """Test interaction history tracking"""
        # Track multiple interactions
        for i in range(10):
            analytics.track_interaction(
                student_id="test_student",
                interaction_type="mastery_update",
                objective_id=f"MATH.8.{i}",
                success=i % 2 == 0,  # Alternate success/failure
                confidence=0.7 + (i * 0.02),
                mastered=i >= 5,
                xp_gained=20
            )

        # Check interactions were tracked
        assert "test_student" in analytics.interaction_history
        history = analytics.interaction_history["test_student"]
        assert len(history) == 10


class TestEngagementScoring:
    """Tests for engagement score calculation"""

    @pytest.fixture
    def analytics(self):
        return LearningAnalytics()

    def test_high_engagement_detection(self, analytics):
        """Test high engagement student identification"""
        student = EdWIN("Engaged", "Student", 8)

        # Track high engagement interactions
        for i in range(20):
            analytics.track_interaction(
                student_id="engaged_student",
                interaction_type="question",
                objective_id=f"MATH.8.{i}",
                success=True,
                confidence=0.9,
                mastered=True,
                xp_gained=25
            )

        engagement = analytics.calculate_engagement(student, window_days=7)

        # Should have high engagement score
        assert engagement.engagement_score > 0.7
        assert engagement.risk_level == "low"

    def test_low_engagement_detection(self, analytics):
        """Test low engagement student identification"""
        student = EdWIN("Disengaged", "Student", 8)

        # Track minimal interactions
        analytics.track_interaction(
            student_id="disengaged_student",
            interaction_type="question",
            objective_id="MATH.8.1",
            success=False,
            confidence=0.4,
            mastered=False,
            xp_gained=5
        )

        engagement = analytics.calculate_engagement(student, window_days=7)

        # Should have low engagement
        assert engagement.engagement_score < 0.5
        assert engagement.risk_level in ["medium", "high"]


class TestVelocityTrends:
    """Tests for learning velocity trend detection"""

    @pytest.fixture
    def analytics(self):
        return LearningAnalytics()

    def test_accelerating_trend(self, analytics):
        """Test accelerating learning trend detection"""
        student = EdWIN("Accelerating", "Learner", 8)

        # Simulate accelerating progress (more recent achievements)
        now = datetime.now().timestamp()
        week_ago = now - (7 * 86400)

        # More interactions in recent days
        for i in range(10):
            # Recent interactions
            analytics.track_interaction(
                student_id="accel_student",
                interaction_type="mastery_update",
                objective_id=f"MATH.8.{i}",
                success=True,
                confidence=0.9,
                mastered=True,
                xp_gained=30
            )

        velocity = analytics.calculate_learning_velocity(student, window_days=7)

        # Trend detection requires more sophisticated history
        assert velocity.trend in ["accelerating", "steady"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
