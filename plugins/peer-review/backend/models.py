"""
Data models for peer review system
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass


class ReviewType(str, Enum):
    """Type of peer review"""
    SINGLE_BLIND = "single_blind"      # Reviewer knows author
    DOUBLE_BLIND = "double_blind"      # Neither knows the other
    OPEN = "open"                      # Both know each other


class RubricType(str, Enum):
    """Type of rubric"""
    ANALYTICAL = "analytical"          # Multi-criteria with scales
    HOLISTIC = "holistic"             # Single overall score
    CHECKLIST = "checklist"           # Binary yes/no items
    SINGLE_POINT = "single_point"     # Feedback-focused


class ReviewStatus(str, Enum):
    """Status of a review"""
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    DISPUTED = "disputed"
    RESOLVED = "resolved"


class DisputeStatus(str, Enum):
    """Status of a dispute"""
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


# ============================================================================
# Rubric Models
# ============================================================================

class RubricCriterion(BaseModel):
    """A single criterion in a rubric"""
    criterion_id: str
    name: str
    description: str
    max_score: float
    levels: List[Dict[str, Any]]  # List of score levels with descriptions


class Rubric(BaseModel):
    """Rubric for peer review"""
    rubric_id: str
    name: str
    rubric_type: RubricType
    criteria: List[RubricCriterion]
    total_points: float
    created_at: datetime


# ============================================================================
# Assignment Models
# ============================================================================

class PeerReviewAssignment(BaseModel):
    """Peer review assignment configuration"""
    assignment_id: str
    title: str
    description: str
    instructions: str

    # Review configuration
    review_type: ReviewType
    rubric: Rubric
    reviews_per_submission: int = 3
    enable_calibration: bool = True
    calibration_threshold: float = 0.8

    # Deadlines
    submission_deadline: datetime
    review_deadline: datetime
    dispute_deadline: Optional[datetime] = None

    # Weighting
    quality_weight: float = 0.7        # Weight of review quality in grade
    completion_weight: float = 0.3     # Weight of completing reviews

    # Metadata
    created_by: str
    institution_id: str
    created_at: datetime
    metadata: Dict[str, Any] = {}


# ============================================================================
# Submission Models
# ============================================================================

class Submission(BaseModel):
    """Student submission for peer review"""
    submission_id: str
    assignment_id: str
    student_id: str

    # Content
    content: str                       # Text content or URL to file
    attachments: List[str] = []       # File URLs

    # Metadata
    submitted_at: datetime
    word_count: Optional[int] = None
    metadata: Dict[str, Any] = {}


# ============================================================================
# Review Models
# ============================================================================

class ReviewScore(BaseModel):
    """Score for a single rubric criterion"""
    criterion_id: str
    score: float
    feedback: str


class Review(BaseModel):
    """Peer review of a submission"""
    review_id: str
    submission_id: str
    reviewer_id: str

    # Review content
    scores: List[ReviewScore]
    total_score: float
    overall_feedback: str

    # Quality metrics
    quality_score: Optional[float] = None    # How good is this review?
    helpfulness_score: Optional[float] = None
    timeliness_score: Optional[float] = None

    # Status
    status: ReviewStatus = ReviewStatus.ASSIGNED
    submitted_at: Optional[datetime] = None
    time_spent_minutes: Optional[int] = None

    # Metadata
    metadata: Dict[str, Any] = {}


# ============================================================================
# Calibration Models
# ============================================================================

class CalibrationExercise(BaseModel):
    """Calibration exercise to train reviewers"""
    exercise_id: str
    assignment_id: str

    # Reference submission (expert-reviewed)
    submission: Submission
    expert_review: Review
    expert_rationale: str              # Why scores were given

    # Metadata
    difficulty: str  # "easy", "medium", "hard"
    created_at: datetime


class CalibrationAttempt(BaseModel):
    """Student's calibration attempt"""
    attempt_id: str
    exercise_id: str
    student_id: str

    # Student's review
    student_review: Review

    # Comparison to expert
    accuracy_score: float              # 0.0-1.0
    alignment_details: Dict[str, Any]  # Per-criterion alignment

    # Metadata
    submitted_at: datetime
    time_spent_minutes: int


class CalibrationScore(BaseModel):
    """Student's overall calibration score"""
    student_id: str
    assignment_id: str

    # Overall score
    calibration_score: float           # 0.0-1.0 (average of attempts)
    attempts_completed: int
    passed: bool                       # Above threshold?

    # Details
    attempts: List[CalibrationAttempt]
    last_attempt_at: datetime


# ============================================================================
# Dispute Models
# ============================================================================

class DisputeRequest(BaseModel):
    """Student dispute of a peer review"""
    dispute_id: str
    review_id: str
    student_id: str                    # Author disputing

    # Dispute details
    reasons: List[str]                 # Selected reasons
    explanation: str                   # Detailed explanation
    requested_action: str              # "reevaluate", "remove", "adjust_score"

    # Status
    status: DisputeStatus = DisputeStatus.PENDING
    resolution: Optional[str] = None
    resolved_by: Optional[str] = None  # Instructor ID
    resolved_at: Optional[datetime] = None

    # Metadata
    submitted_at: datetime
    metadata: Dict[str, Any] = {}


# ============================================================================
# Analytics Models
# ============================================================================

class ReviewQualityMetrics(BaseModel):
    """Metrics for review quality"""
    review_id: str
    reviewer_id: str

    # Quality dimensions
    thoroughness: float                # 0.0-1.0 (length, detail)
    specificity: float                 # 0.0-1.0 (concrete vs vague)
    constructiveness: float            # 0.0-1.0 (actionable feedback)
    alignment: float                   # 0.0-1.0 (vs other reviews)

    # Overall
    quality_score: float               # Weighted average

    # Metadata
    calculated_at: datetime


class ReviewerProfile(BaseModel):
    """Profile of a reviewer's performance"""
    student_id: str
    assignment_id: str

    # Review statistics
    reviews_completed: int
    reviews_assigned: int
    completion_rate: float

    # Quality
    avg_quality_score: float
    avg_helpfulness_score: float
    avg_time_per_review_minutes: float

    # Calibration
    calibration_score: float
    calibrated: bool

    # Reliability
    consistency_score: float           # 0.0-1.0 (alignment with others)
    timeliness_score: float           # 0.0-1.0 (on-time submissions)

    # Metadata
    last_updated: datetime


# ============================================================================
# Assignment Results Models
# ============================================================================

class StudentReviewResults(BaseModel):
    """Results of peer review for a student"""
    student_id: str
    assignment_id: str
    submission_id: str

    # Reviews received
    reviews_received: List[Review]
    avg_score: float
    median_score: float
    score_variance: float

    # Reviews given
    reviews_given: List[Review]
    reviewing_quality: float

    # Final grade
    final_score: float                 # Weighted combination
    grade_breakdown: Dict[str, float]  # Components

    # Feedback
    consolidated_feedback: str         # AI-generated summary
    key_strengths: List[str]
    key_improvements: List[str]

    # Metadata
    calculated_at: datetime


class AssignmentAnalytics(BaseModel):
    """Analytics for an assignment"""
    assignment_id: str

    # Participation
    submissions_count: int
    reviews_completed: int
    reviews_pending: int
    completion_rate: float

    # Score distribution
    avg_score: float
    median_score: float
    score_distribution: Dict[str, int] # Histogram

    # Quality
    avg_review_quality: float
    calibration_pass_rate: float

    # Timeliness
    avg_review_time_hours: float
    late_reviews_count: int

    # Disputes
    disputes_count: int
    dispute_rate: float

    # Top reviewers
    top_reviewers: List[Dict[str, Any]]

    # Metadata
    calculated_at: datetime
