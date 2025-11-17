# LMS Plugin Interfaces
**Date**: 2025-11-17
**Purpose**: Protocol-based plugin interface specifications

## Table of Contents

1. [Overview](#overview)
2. [Base Plugin Protocol](#base-plugin-protocol)
3. [Assessment Plugin Interface](#assessment-plugin-interface)
4. [Content Plugin Interface](#content-plugin-interface)
5. [Analytics Plugin Interface](#analytics-plugin-interface)
6. [Communication Plugin Interface](#communication-plugin-interface)
7. [Integration Plugin Interface](#integration-plugin-interface)
8. [Complete Examples](#complete-examples)

---

## Overview

All LMS plugins implement protocol-based interfaces inspired by HoloLoom's architecture. This enables:

- **Type safety**: Compile-time interface checking
- **Swappability**: Easy plugin replacement
- **Extensibility**: Add new methods without breaking existing plugins
- **Testing**: Mock implementations for unit tests

### Design Philosophy

> **"Protocols, not classes. Interfaces, not implementations."**

Following HoloLoom's design, plugins define **what** they do (protocols), not **how** they do it (implementation details).

---

## Base Plugin Protocol

All plugins must implement the base `PluginProtocol`:

```python
from typing import Protocol, Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class PluginMetadata:
    """Plugin metadata"""
    plugin_id: str
    name: str
    version: str
    category: str
    author: str
    description: str
    permissions: List[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class HealthStatus:
    """Plugin health status"""
    healthy: bool
    message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    last_check: Optional[datetime] = None

class PluginProtocol(Protocol):
    """Base protocol that all plugins must implement"""

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        ...

    async def initialize(self, context: 'PluginContext') -> None:
        """
        Initialize plugin with context.
        Called once when plugin is activated.
        """
        ...

    async def health_check(self) -> HealthStatus:
        """
        Check plugin health.
        Called periodically (every 60 seconds).
        """
        ...

    async def cleanup(self) -> None:
        """
        Cleanup resources before plugin deactivation.
        Called when plugin is deactivated or system shuts down.
        """
        ...
```

### Plugin Context

Every plugin receives a `PluginContext` with access to core services:

```python
@dataclass
class PluginContext:
    """Context provided to plugins"""

    # Plugin identity
    plugin_id: str
    plugin_config: Dict[str, Any]
    institution_id: str

    # Core services
    database: 'DatabaseClient'
    knowledge_graph: 'KnowledgeGraphClient'
    event_bus: 'EventBusClient'
    shared_data: 'SharedDataClient'
    file_storage: 'FileStorageClient'
    logger: 'LoggerClient'

    # Optional services (if permissions granted)
    llm: Optional['LLMClient'] = None
    http_client: Optional['HTTPClient'] = None

    # User context (set per request)
    student: Optional['Student'] = None
    instructor: Optional['Instructor'] = None
    admin: Optional['Admin'] = None

    # Request context
    request_id: str
    timestamp: datetime
```

---

## Assessment Plugin Interface

Plugins in the **Assessment** category evaluate student learning.

### AssessmentPluginProtocol

```python
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

class AssessmentType(Enum):
    """Types of assessments"""
    QUIZ = "quiz"
    EXAM = "exam"
    ASSIGNMENT = "assignment"
    PEER_REVIEW = "peer_review"
    PORTFOLIO = "portfolio"
    DISCUSSION = "discussion"
    PROJECT = "project"

class GradingStrategy(Enum):
    """Grading strategies"""
    AUTO = "auto"              # Fully automated
    MANUAL = "manual"          # Fully manual
    HYBRID = "hybrid"          # Mix of auto and manual
    PEER = "peer"              # Peer grading
    RUBRIC = "rubric"          # Rubric-based

@dataclass
class AssessmentSpec:
    """Assessment specification"""
    assessment_id: str
    type: AssessmentType
    title: str
    description: str
    concept: str                    # Knowledge graph concept
    grading_strategy: GradingStrategy
    max_score: float
    passing_score: float
    time_limit: Optional[timedelta]
    attempts_allowed: int
    due_date: Optional[datetime]
    metadata: Dict[str, Any]

@dataclass
class Submission:
    """Student submission"""
    submission_id: str
    assessment_id: str
    student_id: str
    submitted_at: datetime
    attempt_number: int
    answers: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class GradingResult:
    """Result of grading"""
    submission_id: str
    score: float                    # 0.0 to 1.0 (percentage)
    points: float                   # Actual points earned
    max_points: float               # Maximum possible points
    feedback: str
    detailed_feedback: Dict[str, Any]
    graded_at: datetime
    graded_by: Optional[str]        # Instructor ID (if manual)
    confidence: float               # Grading confidence (0.0-1.0)

class AssessmentPluginProtocol(PluginProtocol, Protocol):
    """Protocol for assessment plugins"""

    async def create_assessment(
        self,
        spec: AssessmentSpec,
        context: PluginContext
    ) -> str:
        """
        Create a new assessment.

        Args:
            spec: Assessment specification
            context: Plugin context

        Returns:
            Assessment ID
        """
        ...

    async def render_assessment(
        self,
        assessment_id: str,
        student_id: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """
        Render assessment for student.
        Personalizes content based on student's knowledge graph.

        Args:
            assessment_id: Assessment ID
            student_id: Student ID
            context: Plugin context

        Returns:
            Rendered assessment data (questions, instructions, etc.)
        """
        ...

    async def submit_assessment(
        self,
        assessment_id: str,
        student_id: str,
        answers: Dict[str, Any],
        context: PluginContext
    ) -> Submission:
        """
        Submit assessment answers.

        Args:
            assessment_id: Assessment ID
            student_id: Student ID
            answers: Student's answers
            context: Plugin context

        Returns:
            Submission object
        """
        ...

    async def grade_submission(
        self,
        submission: Submission,
        context: PluginContext
    ) -> GradingResult:
        """
        Grade a submission.
        Implementation depends on grading_strategy.

        Args:
            submission: Submission to grade
            context: Plugin context

        Returns:
            Grading result
        """
        ...

    async def provide_feedback(
        self,
        submission: Submission,
        grading_result: GradingResult,
        context: PluginContext
    ) -> str:
        """
        Generate detailed feedback for student.

        Args:
            submission: Student submission
            grading_result: Grading result
            context: Plugin context

        Returns:
            Feedback text (HTML/Markdown)
        """
        ...

    async def update_knowledge_graph(
        self,
        submission: Submission,
        grading_result: GradingResult,
        context: PluginContext
    ) -> None:
        """
        Update student's knowledge graph based on assessment results.

        Args:
            submission: Student submission
            grading_result: Grading result
            context: Plugin context
        """
        ...

    async def get_analytics(
        self,
        assessment_id: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """
        Get assessment analytics.

        Args:
            assessment_id: Assessment ID
            context: Plugin context

        Returns:
            Analytics data (scores, completion rate, etc.)
        """
        ...
```

### Example: Quiz Plugin Implementation

```python
class QuizPlugin(AssessmentPluginProtocol):
    """Quiz plugin implementation"""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            plugin_id="quiz-builder",
            name="Quiz Builder",
            version="1.0.0",
            category="assessment",
            author="LMS Team",
            description="Create and grade multiple-choice quizzes",
            permissions=[
                "knowledge_graph:write",
                "database:read",
                "database:write",
                "events:publish"
            ],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    async def initialize(self, context: PluginContext) -> None:
        """Initialize quiz plugin"""
        self.context = context
        self.logger = context.logger

        # Create database tables if needed
        await self._ensure_tables()

        self.logger.info("Quiz plugin initialized")

    async def health_check(self) -> HealthStatus:
        """Check plugin health"""
        try:
            # Check database connectivity
            await self.context.database.execute("SELECT 1")
            return HealthStatus(healthy=True, message="All systems operational")
        except Exception as e:
            return HealthStatus(
                healthy=False,
                message=f"Database connection failed: {e}"
            )

    async def cleanup(self) -> None:
        """Cleanup resources"""
        self.logger.info("Quiz plugin cleanup complete")

    async def create_assessment(
        self,
        spec: AssessmentSpec,
        context: PluginContext
    ) -> str:
        """Create a new quiz"""
        quiz_id = f"quiz_{uuid.uuid4().hex}"

        await context.database.quizzes.create({
            "quiz_id": quiz_id,
            "title": spec.title,
            "description": spec.description,
            "concept": spec.concept,
            "time_limit": spec.time_limit.total_seconds() if spec.time_limit else None,
            "max_score": spec.max_score,
            "passing_score": spec.passing_score,
            "created_by": context.instructor.id if context.instructor else None,
            "institution_id": context.institution_id,
            "created_at": datetime.now()
        })

        self.logger.info(f"Created quiz: {quiz_id}")
        return quiz_id

    async def render_assessment(
        self,
        assessment_id: str,
        student_id: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """Render quiz for student"""

        # Load quiz
        quiz = await context.database.quizzes.get(assessment_id)
        if not quiz:
            raise ValueError(f"Quiz {assessment_id} not found")

        # Check if student has already taken quiz
        previous_attempts = await context.database.quiz_submissions.count({
            "quiz_id": assessment_id,
            "student_id": student_id
        })

        if previous_attempts >= quiz.attempts_allowed:
            raise ValueError("Maximum attempts exceeded")

        # Personalize questions based on knowledge graph
        mastered = await context.knowledge_graph.get_mastered_concepts(
            student_id=student_id,
            min_confidence=0.8
        )

        questions = quiz.questions

        # Filter out questions on already-mastered concepts
        if quiz.adaptive:
            questions = [
                q for q in questions
                if q.concept not in mastered
            ]

        return {
            "quiz_id": quiz.quiz_id,
            "title": quiz.title,
            "description": quiz.description,
            "time_limit": quiz.time_limit,
            "questions": [self._render_question(q) for q in questions],
            "attempt_number": previous_attempts + 1,
            "attempts_remaining": quiz.attempts_allowed - previous_attempts - 1
        }

    async def submit_assessment(
        self,
        assessment_id: str,
        student_id: str,
        answers: Dict[str, Any],
        context: PluginContext
    ) -> Submission:
        """Submit quiz answers"""

        submission_id = f"sub_{uuid.uuid4().hex}"
        attempt_number = await self._get_attempt_number(assessment_id, student_id)

        submission = Submission(
            submission_id=submission_id,
            assessment_id=assessment_id,
            student_id=student_id,
            submitted_at=datetime.now(),
            attempt_number=attempt_number,
            answers=answers,
            metadata={}
        )

        # Store submission
        await context.database.quiz_submissions.create(
            submission.__dict__
        )

        # Trigger grading (async)
        await context.event_bus.publish("assessment_submitted", {
            "submission_id": submission_id,
            "plugin_id": self.metadata.plugin_id
        })

        return submission

    async def grade_submission(
        self,
        submission: Submission,
        context: PluginContext
    ) -> GradingResult:
        """Grade quiz submission"""

        quiz = await context.database.quizzes.get(submission.assessment_id)
        questions = quiz.questions

        correct = 0
        total = len(questions)
        detailed_feedback = {}

        for question in questions:
            student_answer = submission.answers.get(question.id)
            correct_answer = question.correct_answer

            is_correct = student_answer == correct_answer

            if is_correct:
                correct += 1

            detailed_feedback[question.id] = {
                "correct": is_correct,
                "student_answer": student_answer,
                "correct_answer": correct_answer,
                "explanation": question.explanation
            }

        score = correct / total if total > 0 else 0.0
        points = score * quiz.max_score

        feedback = self._generate_overall_feedback(score, quiz.passing_score)

        result = GradingResult(
            submission_id=submission.submission_id,
            score=score,
            points=points,
            max_points=quiz.max_score,
            feedback=feedback,
            detailed_feedback=detailed_feedback,
            graded_at=datetime.now(),
            graded_by=None,  # Auto-graded
            confidence=1.0   # High confidence for multiple-choice
        )

        # Store grading result
        await context.database.grading_results.create(result.__dict__)

        return result

    async def provide_feedback(
        self,
        submission: Submission,
        grading_result: GradingResult,
        context: PluginContext
    ) -> str:
        """Generate detailed feedback"""

        feedback_parts = [
            f"# Quiz Results",
            f"",
            f"**Score**: {grading_result.score:.1%} ({grading_result.points}/{grading_result.max_points} points)",
            f"",
            grading_result.feedback,
            f"",
            f"## Question-by-Question Breakdown",
            f""
        ]

        for question_id, details in grading_result.detailed_feedback.items():
            status = "✓ Correct" if details["correct"] else "✗ Incorrect"
            feedback_parts.extend([
                f"### Question {question_id}",
                f"{status}",
                f"",
                f"**Your answer**: {details['student_answer']}",
                f"**Correct answer**: {details['correct_answer']}",
                f"",
                f"{details['explanation']}",
                f""
            ])

        return "\n".join(feedback_parts)

    async def update_knowledge_graph(
        self,
        submission: Submission,
        grading_result: GradingResult,
        context: PluginContext
    ) -> None:
        """Update knowledge graph"""

        quiz = await context.database.quizzes.get(submission.assessment_id)

        # Record overall mastery
        await context.knowledge_graph.record_learning_event(
            student_id=submission.student_id,
            concept=quiz.concept,
            mastery_level=grading_result.score,
            evidence=f"Quiz: {quiz.title}",
            timestamp=grading_result.graded_at
        )

        # Record per-question mastery
        for question_id, details in grading_result.detailed_feedback.items():
            question = next(q for q in quiz.questions if q.id == question_id)

            if details["correct"]:
                # Mastered this sub-concept
                await context.knowledge_graph.record_learning_event(
                    student_id=submission.student_id,
                    concept=question.sub_concept,
                    mastery_level=1.0,
                    evidence=f"Quiz question: {question.text}",
                    timestamp=grading_result.graded_at
                )
            else:
                # Struggling with this sub-concept
                await context.knowledge_graph.record_learning_event(
                    student_id=submission.student_id,
                    concept=question.sub_concept,
                    mastery_level=0.0,
                    evidence=f"Quiz question: {question.text} (incorrect)",
                    timestamp=grading_result.graded_at
                )

    async def get_analytics(
        self,
        assessment_id: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """Get quiz analytics"""

        submissions = await context.database.quiz_submissions.find({
            "assessment_id": assessment_id
        })

        grading_results = await context.database.grading_results.find({
            "submission_id": {"$in": [s.submission_id for s in submissions]}
        })

        scores = [r.score for r in grading_results]

        return {
            "total_attempts": len(submissions),
            "unique_students": len(set(s.student_id for s in submissions)),
            "completion_rate": len(grading_results) / len(submissions) if submissions else 0,
            "average_score": sum(scores) / len(scores) if scores else 0,
            "median_score": sorted(scores)[len(scores) // 2] if scores else 0,
            "passing_rate": sum(1 for s in scores if s >= 0.7) / len(scores) if scores else 0,
            "score_distribution": self._calculate_distribution(scores)
        }
```

---

## Content Plugin Interface

Plugins in the **Content** category deliver learning materials.

### ContentPluginProtocol

```python
from enum import Enum
from typing import List, Dict, Any, Optional

class ContentType(Enum):
    """Types of content"""
    VIDEO = "video"
    TEXT = "text"
    SLIDES = "slides"
    INTERACTIVE = "interactive"
    SIMULATION = "simulation"
    CODE = "code"
    DOCUMENT = "document"

@dataclass
class ContentSpec:
    """Content specification"""
    content_id: str
    type: ContentType
    title: str
    description: str
    concept: str                    # Knowledge graph concept
    difficulty: str                 # "beginner", "intermediate", "advanced"
    duration_minutes: int
    prerequisites: List[str]        # Required concepts
    metadata: Dict[str, Any]

@dataclass
class ContentBlock:
    """Block of content"""
    block_id: str
    type: str                       # "text", "video", "image", "quiz", etc.
    content: Any
    metadata: Dict[str, Any]

@dataclass
class EngagementEvent:
    """Content engagement event"""
    event_id: str
    student_id: str
    content_id: str
    event_type: str                 # "view", "progress", "complete", etc.
    event_data: Dict[str, Any]
    timestamp: datetime

class ContentPluginProtocol(PluginProtocol, Protocol):
    """Protocol for content plugins"""

    async def create_content(
        self,
        spec: ContentSpec,
        blocks: List[ContentBlock],
        context: PluginContext
    ) -> str:
        """
        Create new content.

        Args:
            spec: Content specification
            blocks: Content blocks
            context: Plugin context

        Returns:
            Content ID
        """
        ...

    async def render_content(
        self,
        content_id: str,
        student_id: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """
        Render content for student.
        Personalizes based on knowledge graph.

        Args:
            content_id: Content ID
            student_id: Student ID
            context: Plugin context

        Returns:
            Rendered content data
        """
        ...

    async def track_engagement(
        self,
        event: EngagementEvent,
        context: PluginContext
    ) -> None:
        """
        Track student engagement with content.

        Args:
            event: Engagement event
            context: Plugin context
        """
        ...

    async def get_progress(
        self,
        content_id: str,
        student_id: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """
        Get student's progress on content.

        Args:
            content_id: Content ID
            student_id: Student ID
            context: Plugin context

        Returns:
            Progress data (percentage, time spent, etc.)
        """
        ...

    async def recommend_next(
        self,
        student_id: str,
        current_content_id: str,
        context: PluginContext
    ) -> List[str]:
        """
        Recommend next content based on knowledge graph.

        Args:
            student_id: Student ID
            current_content_id: Current content ID
            context: Plugin context

        Returns:
            List of recommended content IDs
        """
        ...

    async def generate_transcript(
        self,
        content_id: str,
        context: PluginContext
    ) -> str:
        """
        Generate transcript/summary of content.
        For accessibility and search indexing.

        Args:
            content_id: Content ID
            context: Plugin context

        Returns:
            Transcript text
        """
        ...
```

### Example: Video Player Plugin Implementation

```python
class VideoPlayerPlugin(ContentPluginProtocol):
    """Interactive video player plugin"""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            plugin_id="video-player",
            name="Interactive Video Player",
            version="1.0.0",
            category="content",
            author="LMS Team",
            description="Video player with annotations, quizzes, and analytics",
            permissions=[
                "knowledge_graph:write",
                "database:read",
                "database:write",
                "files:read",
                "events:publish"
            ],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    async def render_content(
        self,
        content_id: str,
        student_id: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """Render video player"""

        video = await context.database.videos.get(content_id)
        if not video:
            raise ValueError(f"Video {content_id} not found")

        # Check prerequisites
        mastered = await context.knowledge_graph.get_mastered_concepts(
            student_id=student_id
        )

        unmet_prereqs = [
            prereq for prereq in video.prerequisites
            if prereq not in mastered
        ]

        if unmet_prereqs:
            return {
                "error": "Prerequisites not met",
                "prerequisites": unmet_prereqs,
                "suggested_content": await self._suggest_prerequisite_content(unmet_prereqs)
            }

        # Get student's previous progress
        progress = await self.get_progress(content_id, student_id, context)

        # Get video URL (signed URL for security)
        video_url = await context.file_storage.get_signed_url(
            video.file_path,
            expires_in=3600  # 1 hour
        )

        return {
            "video_id": video.video_id,
            "title": video.title,
            "description": video.description,
            "duration_seconds": video.duration_seconds,
            "video_url": video_url,
            "thumbnail_url": video.thumbnail_url,
            "captions": video.captions,
            "annotations": video.annotations,
            "quiz_points": video.quiz_points,
            "progress": progress,
            "resume_time": progress.get("last_position", 0)
        }

    async def track_engagement(
        self,
        event: EngagementEvent,
        context: PluginContext
    ) -> None:
        """Track video engagement"""

        # Store event
        await context.database.engagement_events.create(event.__dict__)

        # Update progress
        if event.event_type == "video_progress":
            await self._update_progress(event, context)

        # Update knowledge graph on completion
        if event.event_type == "video_complete":
            await self._update_knowledge_graph(event, context)

        # Publish event for other plugins
        await context.event_bus.publish("video_engagement", {
            "student_id": event.student_id,
            "video_id": event.content_id,
            "event_type": event.event_type,
            "event_data": event.event_data
        })

    async def recommend_next(
        self,
        student_id: str,
        current_content_id: str,
        context: PluginContext
    ) -> List[str]:
        """Recommend next videos"""

        current_video = await context.database.videos.get(current_content_id)

        # Get concepts covered in current video
        concepts = current_video.concepts

        # Get next concepts in learning path
        next_concepts = await context.knowledge_graph.get_next_concepts(
            student_id=student_id,
            current_concepts=concepts
        )

        # Find videos covering next concepts
        recommended_videos = await context.database.videos.find({
            "concepts": {"$in": next_concepts},
            "difficulty": self._get_appropriate_difficulty(student_id, context)
        })

        return [v.video_id for v in recommended_videos[:5]]  # Top 5
```

---

## Analytics Plugin Interface

Plugins in the **Analytics** category track and analyze learning data.

### AnalyticsPluginProtocol

```python
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

class MetricType(Enum):
    """Types of metrics"""
    ENGAGEMENT = "engagement"
    PERFORMANCE = "performance"
    PROGRESS = "progress"
    PREDICTIVE = "predictive"
    BEHAVIORAL = "behavioral"

@dataclass
class MetricSpec:
    """Metric specification"""
    metric_id: str
    name: str
    type: MetricType
    description: str
    aggregation: str                # "sum", "avg", "count", "max", "min"
    time_window: timedelta
    metadata: Dict[str, Any]

@dataclass
class DataPoint:
    """Single data point"""
    timestamp: datetime
    value: float
    dimensions: Dict[str, str]      # e.g., {"student_id": "123", "course_id": "456"}
    metadata: Dict[str, Any]

@dataclass
class AnalyticsResult:
    """Analytics query result"""
    metric_id: str
    data_points: List[DataPoint]
    summary: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    insights: List[str]

class AnalyticsPluginProtocol(PluginProtocol, Protocol):
    """Protocol for analytics plugins"""

    async def register_metric(
        self,
        spec: MetricSpec,
        context: PluginContext
    ) -> str:
        """
        Register a new metric.

        Args:
            spec: Metric specification
            context: Plugin context

        Returns:
            Metric ID
        """
        ...

    async def record_metric(
        self,
        metric_id: str,
        value: float,
        dimensions: Dict[str, str],
        context: PluginContext
    ) -> None:
        """
        Record metric value.

        Args:
            metric_id: Metric ID
            value: Metric value
            dimensions: Dimension values
            context: Plugin context
        """
        ...

    async def query_metrics(
        self,
        metric_id: str,
        filters: Dict[str, Any],
        time_range: tuple[datetime, datetime],
        context: PluginContext
    ) -> AnalyticsResult:
        """
        Query metric data.

        Args:
            metric_id: Metric ID
            filters: Filter criteria
            time_range: (start, end) time range
            context: Plugin context

        Returns:
            Analytics result with data and visualizations
        """
        ...

    async def generate_dashboard(
        self,
        user_id: str,
        role: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """
        Generate personalized dashboard.

        Args:
            user_id: User ID (student/instructor)
            role: User role
            context: Plugin context

        Returns:
            Dashboard configuration
        """
        ...

    async def detect_anomalies(
        self,
        metric_id: str,
        context: PluginContext
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metric data.

        Args:
            metric_id: Metric ID
            context: Plugin context

        Returns:
            List of detected anomalies
        """
        ...

    async def predict_outcome(
        self,
        student_id: str,
        prediction_type: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """
        Predict student outcome.

        Args:
            student_id: Student ID
            prediction_type: "final_grade", "at_risk", "completion", etc.
            context: Plugin context

        Returns:
            Prediction with confidence
        """
        ...

    async def generate_report(
        self,
        report_type: str,
        filters: Dict[str, Any],
        context: PluginContext
    ) -> bytes:
        """
        Generate report (PDF/CSV/Excel).

        Args:
            report_type: Type of report
            filters: Report filters
            context: Plugin context

        Returns:
            Report file bytes
        """
        ...
```

### Example: Engagement Analytics Plugin

```python
class EngagementAnalyticsPlugin(AnalyticsPluginProtocol):
    """Student engagement analytics plugin"""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            plugin_id="engagement-analytics",
            name="Engagement Analytics",
            version="1.0.0",
            category="analytics",
            author="LMS Team",
            description="Track and analyze student engagement patterns",
            permissions=[
                "database:read",
                "knowledge_graph:read",
                "events:subscribe"
            ],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    async def generate_dashboard(
        self,
        user_id: str,
        role: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """Generate engagement dashboard"""

        if role == "student":
            return await self._student_dashboard(user_id, context)
        elif role == "instructor":
            return await self._instructor_dashboard(user_id, context)
        else:
            raise ValueError(f"Unknown role: {role}")

    async def _student_dashboard(
        self,
        student_id: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """Generate dashboard for student"""

        # Get engagement metrics
        now = datetime.now()
        week_ago = now - timedelta(days=7)

        engagement_events = await context.database.engagement_events.find({
            "student_id": student_id,
            "timestamp": {"$gte": week_ago}
        })

        # Calculate metrics
        total_time_minutes = sum(
            e.event_data.get("duration_seconds", 0) / 60
            for e in engagement_events
        )

        videos_watched = len([
            e for e in engagement_events
            if e.event_type == "video_complete"
        ])

        quizzes_completed = len([
            e for e in engagement_events
            if e.event_type == "quiz_complete"
        ])

        # Get knowledge graph progress
        concepts = await context.knowledge_graph.get_mastered_concepts(
            student_id=student_id
        )

        # Generate visualizations
        time_series = self._generate_time_series(engagement_events)
        concept_map = self._generate_concept_map(concepts, context)

        return {
            "summary": {
                "total_time_minutes": total_time_minutes,
                "videos_watched": videos_watched,
                "quizzes_completed": quizzes_completed,
                "concepts_mastered": len(concepts)
            },
            "visualizations": [
                {
                    "type": "line_chart",
                    "title": "Daily Engagement",
                    "data": time_series
                },
                {
                    "type": "network_graph",
                    "title": "Knowledge Map",
                    "data": concept_map
                }
            ],
            "insights": [
                f"You've spent {total_time_minutes:.0f} minutes learning this week.",
                f"You've mastered {len(concepts)} concepts so far.",
                self._generate_insight(engagement_events, concepts)
            ],
            "recommendations": await self._generate_recommendations(
                student_id, concepts, context
            )
        }

    async def predict_outcome(
        self,
        student_id: str,
        prediction_type: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """Predict student outcome"""

        if prediction_type == "at_risk":
            return await self._predict_at_risk(student_id, context)
        elif prediction_type == "final_grade":
            return await self._predict_final_grade(student_id, context)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

    async def _predict_at_risk(
        self,
        student_id: str,
        context: PluginContext
    ) -> Dict[str, Any]:
        """Predict if student is at-risk"""

        # Gather features
        features = await self._extract_features(student_id, context)

        # Simple rule-based model (in production, use ML model)
        risk_factors = []
        risk_score = 0.0

        # Low engagement
        if features["engagement_rate"] < 0.3:
            risk_factors.append("Low engagement")
            risk_score += 0.3

        # Poor quiz performance
        if features["avg_quiz_score"] < 0.6:
            risk_factors.append("Low assessment scores")
            risk_score += 0.4

        # Falling behind
        if features["lessons_behind"] > 2:
            risk_factors.append("Falling behind schedule")
            risk_score += 0.3

        at_risk = risk_score > 0.5

        return {
            "at_risk": at_risk,
            "risk_score": risk_score,
            "confidence": 0.85,
            "risk_factors": risk_factors,
            "recommendations": self._generate_interventions(risk_factors)
        }
```

---

## Communication Plugin Interface

Plugins in the **Communication** category enable interaction.

### CommunicationPluginProtocol

```python
from enum import Enum
from typing import List, Dict, Any, Optional

class MessageType(Enum):
    """Types of messages"""
    TEXT = "text"
    REPLY = "reply"
    ANNOUNCEMENT = "announcement"
    QUESTION = "question"
    ANSWER = "answer"

@dataclass
class Message:
    """A message"""
    message_id: str
    thread_id: Optional[str]        # For replies
    author_id: str
    author_role: str                # "student", "instructor", "ta"
    content: str
    message_type: MessageType
    created_at: datetime
    edited_at: Optional[datetime]
    metadata: Dict[str, Any]

class CommunicationPluginProtocol(PluginProtocol, Protocol):
    """Protocol for communication plugins"""

    async def send_message(
        self,
        message: Message,
        context: PluginContext
    ) -> str:
        """Send a message"""
        ...

    async def get_messages(
        self,
        filters: Dict[str, Any],
        context: PluginContext
    ) -> List[Message]:
        """Get messages"""
        ...

    async def moderate_message(
        self,
        message_id: str,
        action: str,
        context: PluginContext
    ) -> None:
        """Moderate message (approve/flag/delete)"""
        ...

    async def notify_participants(
        self,
        message: Message,
        context: PluginContext
    ) -> None:
        """Notify participants of new message"""
        ...
```

---

## Integration Plugin Interface

Plugins in the **Integration** category connect external services.

### IntegrationPluginProtocol

```python
class IntegrationPluginProtocol(PluginProtocol, Protocol):
    """Protocol for integration plugins"""

    async def authenticate(
        self,
        user_id: str,
        credentials: Dict[str, Any],
        context: PluginContext
    ) -> Dict[str, Any]:
        """Authenticate with external service"""
        ...

    async def sync_data(
        self,
        direction: str,             # "import" or "export"
        data_type: str,             # "students", "grades", "assignments"
        context: PluginContext
    ) -> Dict[str, Any]:
        """Sync data with external service"""
        ...

    async def trigger_action(
        self,
        action: str,
        params: Dict[str, Any],
        context: PluginContext
    ) -> Dict[str, Any]:
        """Trigger action in external service"""
        ...
```

---

## Complete Examples

See complete plugin implementations in the examples repository:

- **Quiz Plugin**: Assessment plugin with auto-grading
- **Video Player Plugin**: Content plugin with engagement tracking
- **Dashboard Plugin**: Analytics plugin with predictive models
- **Forum Plugin**: Communication plugin with moderation
- **GitHub Plugin**: Integration plugin with repo sync

Repository: https://github.com/lms/plugin-examples

---

**Author**: Claude Code
**Date**: 2025-11-17
**Version**: 1.0
