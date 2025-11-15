"""
EdWIN Learning Analytics

Comprehensive analytics and insights for students, teachers, and administrators.

**Features**:
- Student progress tracking
- Learning velocity (concepts/week)
- Knowledge gap analysis
- Difficulty calibration
- Engagement metrics
- Intervention detection (struggling students)

**Implementation Date**: November 15, 2025
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from .student_model import EdWINStudentModel
from .curriculum_graph import EdWINKnowledgeGraph
from EduVerse.education.curriculum import Subject, BloomLevel


@dataclass
class LearningVelocity:
    """Learning velocity metrics"""
    objectives_per_week: float
    xp_per_week: float
    current_streak_days: int
    best_streak_days: int
    avg_confidence: float
    trend: str  # "accelerating", "steady", "decelerating"


@dataclass
class KnowledgeGapAnalysis:
    """Knowledge gap analysis"""
    total_gaps: int
    critical_gaps: List[str]  # High-priority missing prerequisites
    gap_distribution: Dict[Subject, int]
    estimated_catchup_hours: float


@dataclass
class EngagementMetrics:
    """Student engagement metrics"""
    total_interactions: int
    avg_session_length_minutes: float
    questions_asked: int
    success_rate: float
    engagement_score: float  # 0.0-1.0
    risk_level: str  # "low", "medium", "high"


@dataclass
class InterventionAlert:
    """Alert for struggling student"""
    student_id: str
    severity: str  # "warning", "critical"
    reason: str
    affected_objectives: List[str]
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class LearningAnalytics:
    """
    Learning Analytics Engine

    Analyzes student progress and generates insights for teachers.

    Example:
        >>> analytics = LearningAnalytics(curriculum_kg)
        >>>
        >>> # Analyze single student
        >>> velocity = analytics.calculate_learning_velocity(student)
        >>> gaps = analytics.analyze_knowledge_gaps(student)
        >>>
        >>> # Detect struggling students
        >>> alerts = analytics.detect_interventions(students)
    """

    def __init__(self, curriculum_kg: EdWINKnowledgeGraph):
        """
        Initialize analytics engine

        Args:
            curriculum_kg: Curriculum knowledge graph
        """
        self.curriculum_kg = curriculum_kg

        # History tracking (in-memory for now)
        self.interaction_history: Dict[str, List[Dict]] = defaultdict(list)

    def calculate_learning_velocity(
        self,
        student: EdWINStudentModel,
        window_days: int = 7
    ) -> LearningVelocity:
        """
        Calculate learning velocity (speed of progress)

        Args:
            student: Student model
            window_days: Time window for velocity calculation

        Returns:
            LearningVelocity metrics
        """
        # Get recent interactions
        recent = self._get_recent_interactions(student.student_id, window_days)

        if not recent:
            return LearningVelocity(
                objectives_per_week=0.0,
                xp_per_week=0.0,
                current_streak_days=student.player_model.stats.streak_days,
                best_streak_days=student.player_model.stats.streak_days,
                avg_confidence=0.0,
                trend="steady"
            )

        # Calculate metrics
        objectives_mastered = sum(1 for interaction in recent if interaction.get("mastered", False))
        total_xp = sum(interaction.get("xp_gained", 0) for interaction in recent)
        confidences = [interaction.get("confidence", 0) for interaction in recent]

        # Normalize to per-week
        days_covered = min(window_days, len(set(i.get("date") for i in recent)))
        weeks = max(days_covered / 7.0, 0.1)

        objectives_per_week = objectives_mastered / weeks
        xp_per_week = total_xp / weeks
        avg_confidence = statistics.mean(confidences) if confidences else 0.0

        # Detect trend
        if len(confidences) > 3:
            first_half = statistics.mean(confidences[:len(confidences)//2])
            second_half = statistics.mean(confidences[len(confidences)//2:])

            if second_half > first_half * 1.1:
                trend = "accelerating"
            elif second_half < first_half * 0.9:
                trend = "decelerating"
            else:
                trend = "steady"
        else:
            trend = "steady"

        return LearningVelocity(
            objectives_per_week=objectives_per_week,
            xp_per_week=xp_per_week,
            current_streak_days=student.player_model.stats.streak_days,
            best_streak_days=student.player_model.stats.streak_days,  # TODO: track separately
            avg_confidence=avg_confidence,
            trend=trend
        )

    def analyze_knowledge_gaps(
        self,
        student: EdWINStudentModel
    ) -> KnowledgeGapAnalysis:
        """
        Analyze knowledge gaps

        Identifies missing prerequisites for in-progress objectives.

        Args:
            student: Student model

        Returns:
            KnowledgeGapAnalysis
        """
        # Get all in-progress objectives
        in_progress = list(student.in_progress_objectives)

        all_gaps = set()
        critical_gaps = []

        for obj_id in in_progress:
            gaps = self.curriculum_kg.get_knowledge_gaps(
                obj_id,
                student.mastered_objectives
            )
            all_gaps.update(gaps)

            # Critical gaps: prerequisites with Bloom level >= APPLY
            for gap_id in gaps:
                gap_obj = self.curriculum_kg.get_objective(gap_id)
                if gap_obj and gap_obj.bloom_level.value >= BloomLevel.APPLY.value:
                    if gap_id not in critical_gaps:
                        critical_gaps.append(gap_id)

        # Distribution by subject
        gap_distribution = {}
        for subject in Subject:
            count = sum(
                1 for gap_id in all_gaps
                if self.curriculum_kg.get_objective(gap_id).subject == subject
            )
            if count > 0:
                gap_distribution[subject] = count

        # Estimate catchup time
        estimated_hours = sum(
            self.curriculum_kg.get_objective(gap_id).estimated_hours
            for gap_id in all_gaps
            if self.curriculum_kg.get_objective(gap_id)
        )

        return KnowledgeGapAnalysis(
            total_gaps=len(all_gaps),
            critical_gaps=critical_gaps[:5],  # Top 5
            gap_distribution=gap_distribution,
            estimated_catchup_hours=estimated_hours
        )

    def calculate_engagement(
        self,
        student: EdWINStudentModel,
        window_days: int = 7
    ) -> EngagementMetrics:
        """
        Calculate engagement metrics

        Args:
            student: Student model
            window_days: Time window

        Returns:
            EngagementMetrics
        """
        recent = self._get_recent_interactions(student.student_id, window_days)

        if not recent:
            return EngagementMetrics(
                total_interactions=0,
                avg_session_length_minutes=0.0,
                questions_asked=0,
                success_rate=0.0,
                engagement_score=0.0,
                risk_level="high"
            )

        # Calculate metrics
        total_interactions = len(recent)
        questions_asked = sum(1 for i in recent if i.get("type") == "question")
        successes = sum(1 for i in recent if i.get("success", False))
        success_rate = successes / total_interactions if total_interactions > 0 else 0.0

        # Session length (placeholder - needs session tracking)
        avg_session_length = 15.0  # Default

        # Engagement score
        # Combines: interaction frequency, success rate, confidence
        frequency_score = min(total_interactions / (window_days * 5), 1.0)  # Target: 5/day
        success_score = success_rate
        avg_confidence = statistics.mean([i.get("confidence", 0) for i in recent])

        engagement_score = (
            0.4 * frequency_score +
            0.3 * success_score +
            0.3 * avg_confidence
        )

        # Risk level
        if engagement_score < 0.3:
            risk_level = "high"
        elif engagement_score < 0.6:
            risk_level = "medium"
        else:
            risk_level = "low"

        return EngagementMetrics(
            total_interactions=total_interactions,
            avg_session_length_minutes=avg_session_length,
            questions_asked=questions_asked,
            success_rate=success_rate,
            engagement_score=engagement_score,
            risk_level=risk_level
        )

    def detect_interventions(
        self,
        students: List[EdWINStudentModel],
        threshold_days: int = 7
    ) -> List[InterventionAlert]:
        """
        Detect students needing intervention

        Identifies students who are:
        - Disengaged (no activity)
        - Struggling (low success rate)
        - Stuck (same objective for too long)

        Args:
            students: List of students
            threshold_days: Days for intervention detection

        Returns:
            List of intervention alerts
        """
        alerts = []

        for student in students:
            # Check engagement
            engagement = self.calculate_engagement(student, threshold_days)

            if engagement.risk_level == "high":
                alerts.append(InterventionAlert(
                    student_id=student.student_id,
                    severity="critical",
                    reason=f"Low engagement ({engagement.engagement_score:.1%})",
                    affected_objectives=list(student.in_progress_objectives)[:3],
                    recommended_actions=[
                        "Check in with student",
                        "Adjust difficulty level",
                        "Recommend easier objectives"
                    ]
                ))

            # Check success rate
            if engagement.success_rate < 0.5 and engagement.total_interactions > 5:
                alerts.append(InterventionAlert(
                    student_id=student.student_id,
                    severity="warning",
                    reason=f"Low success rate ({engagement.success_rate:.1%})",
                    affected_objectives=list(student.in_progress_objectives)[:3],
                    recommended_actions=[
                        "Review prerequisites",
                        "Provide additional support",
                        "Offer one-on-one tutoring"
                    ]
                ))

            # Check knowledge gaps
            gaps = self.analyze_knowledge_gaps(student)

            if gaps.total_gaps > 5:
                alerts.append(InterventionAlert(
                    student_id=student.student_id,
                    severity="warning",
                    reason=f"Large knowledge gaps ({gaps.total_gaps} missing prerequisites)",
                    affected_objectives=gaps.critical_gaps,
                    recommended_actions=[
                        f"Address critical gaps: {', '.join(gaps.critical_gaps[:3])}",
                        f"Estimated catchup time: {gaps.estimated_catchup_hours:.1f} hours"
                    ]
                ))

        return alerts

    def generate_progress_report(
        self,
        student: EdWINStudentModel
    ) -> Dict:
        """
        Generate comprehensive progress report

        Args:
            student: Student model

        Returns:
            Complete progress report dict
        """
        progress = student.get_progress_summary(self.curriculum_kg)
        velocity = self.calculate_learning_velocity(student)
        gaps = self.analyze_knowledge_gaps(student)
        engagement = self.calculate_engagement(student)

        # Subject-specific progress
        subject_progress = {}
        for subject in Subject:
            subject_objs = self.curriculum_kg.get_subject_objectives(subject)
            mastered_in_subject = sum(
                1 for obj in subject_objs
                if obj.id in student.mastered_objectives
            )
            subject_progress[subject.value] = {
                "total": len(subject_objs),
                "mastered": mastered_in_subject,
                "percentage": (mastered_in_subject / len(subject_objs) * 100) if subject_objs else 0
            }

        return {
            "student": {
                "id": student.student_id,
                "name": student.name,
                "grade": student.grade
            },
            "overall_progress": {
                "mastered_count": progress["mastered_count"],
                "in_progress_count": progress["in_progress_count"],
                "mastery_percentage": progress["mastery_percentage"],
                "level": progress["level"],
                "total_xp": progress["total_xp"],
                "streak_days": progress["streak_days"]
            },
            "subject_progress": subject_progress,
            "learning_velocity": {
                "objectives_per_week": velocity.objectives_per_week,
                "xp_per_week": velocity.xp_per_week,
                "avg_confidence": velocity.avg_confidence,
                "trend": velocity.trend
            },
            "knowledge_gaps": {
                "total_gaps": gaps.total_gaps,
                "critical_gaps": gaps.critical_gaps,
                "distribution": {k.value: v for k, v in gaps.gap_distribution.items()},
                "estimated_catchup_hours": gaps.estimated_catchup_hours
            },
            "engagement": {
                "total_interactions": engagement.total_interactions,
                "success_rate": engagement.success_rate,
                "engagement_score": engagement.engagement_score,
                "risk_level": engagement.risk_level
            },
            "recommendations": [obj.title for obj in progress["recommended"]],
            "generated_at": datetime.utcnow().isoformat()
        }

    def track_interaction(
        self,
        student_id: str,
        interaction_type: str,
        objective_id: Optional[str] = None,
        success: bool = False,
        confidence: float = 0.0,
        mastered: bool = False,
        xp_gained: float = 0.0
    ):
        """
        Track student interaction for analytics

        Args:
            student_id: Student ID
            interaction_type: Type ("question", "practice", etc.)
            objective_id: Related objective
            success: Whether successful
            confidence: Confidence score
            mastered: Whether mastery achieved
            xp_gained: XP awarded
        """
        self.interaction_history[student_id].append({
            "type": interaction_type,
            "objective_id": objective_id,
            "success": success,
            "confidence": confidence,
            "mastered": mastered,
            "xp_gained": xp_gained,
            "date": datetime.utcnow().date().isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        })

    def _get_recent_interactions(
        self,
        student_id: str,
        window_days: int
    ) -> List[Dict]:
        """Get recent interactions within time window"""
        cutoff = datetime.utcnow() - timedelta(days=window_days)

        return [
            interaction for interaction in self.interaction_history.get(student_id, [])
            if datetime.fromisoformat(interaction["timestamp"]) > cutoff
        ]


# Helper functions

def create_analytics(curriculum_kg: EdWINKnowledgeGraph) -> LearningAnalytics:
    """
    Create learning analytics engine

    Args:
        curriculum_kg: Curriculum knowledge graph

    Returns:
        LearningAnalytics instance
    """
    return LearningAnalytics(curriculum_kg)
