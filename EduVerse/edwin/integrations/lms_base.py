"""
LMS Integration Base Protocol

Defines the abstract interface for all LMS integrations.
This enables adding new LMS platforms (Schoology, Blackboard, Moodle) easily.

Author: Agent D
Date: 2025-11-15
"""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class StudentRole(Enum):
    """Student role in course"""
    STUDENT = "student"
    TA = "ta"
    TEACHER = "teacher"


class SubmissionStatus(Enum):
    """Assignment submission status"""
    NOT_SUBMITTED = "not_submitted"
    SUBMITTED = "submitted"
    GRADED = "graded"
    LATE = "late"


@dataclass
class Student:
    """Student information from LMS"""
    id: str
    name: str
    email: str
    role: StudentRole
    external_id: Optional[str] = None  # LMS-specific ID


@dataclass
class RosterData:
    """Course roster from LMS"""
    course_id: str
    course_name: str
    students: List[Student]
    teachers: List[Student]
    sync_timestamp: datetime


@dataclass
class Assignment:
    """Assignment from LMS"""
    id: str
    name: str
    description: str
    due_date: Optional[datetime]
    points_possible: float
    course_id: str
    external_id: Optional[str] = None  # LMS-specific ID


@dataclass
class Submission:
    """Student submission"""
    id: str
    assignment_id: str
    student_id: str
    status: SubmissionStatus
    score: Optional[float]
    submitted_at: Optional[datetime]
    graded_at: Optional[datetime]


class LMSIntegration(Protocol):
    """
    Abstract protocol for LMS integrations.

    All LMS integrations (Canvas, Google Classroom, etc.) must implement this interface.
    """

    async def authenticate(self) -> bool:
        """
        Authenticate with the LMS.

        Returns:
            True if authentication successful, False otherwise
        """
        ...

    async def sync_roster(self, course_id: str) -> RosterData:
        """
        Sync course roster from LMS.

        Args:
            course_id: LMS course identifier

        Returns:
            RosterData with students and teachers
        """
        ...

    async def create_assignment(self, assignment: Assignment) -> str:
        """
        Create assignment in LMS.

        Args:
            assignment: Assignment to create

        Returns:
            LMS assignment ID
        """
        ...

    async def submit_grade(
        self,
        student_id: str,
        assignment_id: str,
        grade: float,
        comment: Optional[str] = None
    ) -> bool:
        """
        Submit grade to LMS gradebook.

        Args:
            student_id: Student identifier
            assignment_id: Assignment identifier
            grade: Grade value (0.0-1.0 or points)
            comment: Optional feedback comment

        Returns:
            True if grade submitted successfully
        """
        ...

    async def get_assignments(self, course_id: str) -> List[Assignment]:
        """
        Get all assignments for a course.

        Args:
            course_id: Course identifier

        Returns:
            List of assignments
        """
        ...

    async def get_student_submissions(
        self,
        assignment_id: str
    ) -> List[Submission]:
        """
        Get all student submissions for an assignment.

        Args:
            assignment_id: Assignment identifier

        Returns:
            List of submissions
        """
        ...

    async def get_courses(self) -> List[Dict[str, Any]]:
        """
        Get list of courses accessible to authenticated user.

        Returns:
            List of course dictionaries
        """
        ...

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to LMS and return status.

        Returns:
            Dictionary with connection status and details
        """
        ...


class LMSConfig:
    """Base configuration for LMS integrations"""

    def __init__(self, **kwargs):
        self.base_url: str = kwargs.get("base_url", "")
        self.client_id: str = kwargs.get("client_id", "")
        self.client_secret: str = kwargs.get("client_secret", "")
        self.redirect_uri: str = kwargs.get("redirect_uri", "")
        self.scopes: List[str] = kwargs.get("scopes", [])
        self.extra_params: Dict[str, Any] = kwargs.get("extra_params", {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "base_url": self.base_url,
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scopes": self.scopes,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LMSConfig":
        """Create config from dictionary"""
        return cls(**data)
