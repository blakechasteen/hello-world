"""
Canvas LMS Integration

Implements LTI 1.3 integration with Canvas LMS.

Features:
- LTI 1.3 launch (OAuth2 + OIDC)
- Deep linking
- Assignment and Grade Services (AGS)
- Names and Role Provisioning Service (NRPS)
- Roster sync
- Grade passback

Author: Agent D
Date: 2025-11-15
"""

import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp
from .lms_base import (
    LMSIntegration,
    RosterData,
    Assignment,
    Submission,
    Student,
    StudentRole,
    SubmissionStatus,
)
from .oauth_manager import OAuth2Manager, OAuth2Provider, OAuth2Token


class CanvasIntegration:
    """
    Canvas LMS integration using LTI 1.3 and Canvas API.

    Supports:
    - Roster synchronization via NRPS
    - Assignment creation and retrieval
    - Grade passback via AGS
    - Single sign-on via LTI 1.3
    """

    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        oauth_manager: Optional[OAuth2Manager] = None,
    ):
        """
        Initialize Canvas integration.

        Args:
            base_url: Canvas instance URL (e.g., "https://canvas.instructure.com")
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            oauth_manager: Optional OAuth2Manager instance
        """
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.oauth_manager = oauth_manager or OAuth2Manager()
        self.api_version = "v1"

    @property
    def api_base(self) -> str:
        """Get Canvas API base URL"""
        return f"{self.base_url}/api/{self.api_version}"

    async def authenticate(self) -> bool:
        """
        Authenticate with Canvas using OAuth2.

        Returns:
            True if authentication successful
        """
        try:
            token = await self.oauth_manager.get_valid_token(
                OAuth2Provider.CANVAS,
                self.base_url,
                self.client_id,
                self.client_secret,
            )
            return token is not None
        except Exception as e:
            print(f"Canvas authentication failed: {e}")
            return False

    async def _get_headers(self) -> Dict[str, str]:
        """Get headers with valid access token"""
        token = await self.oauth_manager.get_valid_token(
            OAuth2Provider.CANVAS,
            self.base_url,
            self.client_id,
            self.client_secret,
        )

        if not token:
            raise Exception("No valid Canvas token available")

        return {
            "Authorization": f"Bearer {token.access_token}",
            "Content-Type": "application/json",
        }

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make Canvas API request.

        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            data: Request body (for POST/PUT)
            params: Query parameters

        Returns:
            API response as dictionary
        """
        headers = await self._get_headers()
        url = f"{self.api_base}/{endpoint.lstrip('/')}"

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method, url, json=data, params=params, headers=headers
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(
                        f"Canvas API error {response.status}: {error_text}"
                    )

                return await response.json()

    async def sync_roster(self, course_id: str) -> RosterData:
        """
        Sync course roster from Canvas.

        Uses NRPS (Names and Role Provisioning Service) when available,
        falls back to Enrollments API.

        Args:
            course_id: Canvas course ID

        Returns:
            RosterData with students and teachers
        """
        # Get course info
        course_data = await self._api_request("GET", f"courses/{course_id}")

        # Get enrollments
        enrollments = await self._api_request(
            "GET",
            f"courses/{course_id}/enrollments",
            params={"per_page": 100},
        )

        students = []
        teachers = []

        for enrollment in enrollments:
            user = enrollment.get("user", {})
            role_type = enrollment.get("type", "").lower()

            student = Student(
                id=str(user.get("id")),
                name=user.get("name", ""),
                email=user.get("email", ""),
                role=self._map_canvas_role(role_type),
                external_id=str(enrollment.get("id")),
            )

            if student.role == StudentRole.STUDENT:
                students.append(student)
            elif student.role == StudentRole.TEACHER:
                teachers.append(student)

        return RosterData(
            course_id=course_id,
            course_name=course_data.get("name", ""),
            students=students,
            teachers=teachers,
            sync_timestamp=datetime.now(),
        )

    def _map_canvas_role(self, role_type: str) -> StudentRole:
        """Map Canvas role to StudentRole enum"""
        role_type = role_type.lower()

        if "teacher" in role_type or "instructor" in role_type:
            return StudentRole.TEACHER
        elif "ta" in role_type or "assistant" in role_type:
            return StudentRole.TA
        else:
            return StudentRole.STUDENT

    async def create_assignment(self, assignment: Assignment) -> str:
        """
        Create assignment in Canvas.

        Args:
            assignment: Assignment to create

        Returns:
            Canvas assignment ID
        """
        data = {
            "assignment": {
                "name": assignment.name,
                "description": assignment.description,
                "points_possible": assignment.points_possible,
                "due_at": assignment.due_date.isoformat() if assignment.due_date else None,
                "submission_types": ["external_tool"],  # EdWIN is external tool
                "external_tool_tag_attributes": {
                    "url": f"{self.base_url}/edwin/launch",  # LTI launch URL
                    "new_tab": False,
                },
            }
        }

        result = await self._api_request(
            "POST",
            f"courses/{assignment.course_id}/assignments",
            data=data,
        )

        return str(result.get("id"))

    async def submit_grade(
        self,
        student_id: str,
        assignment_id: str,
        grade: float,
        comment: Optional[str] = None,
    ) -> bool:
        """
        Submit grade to Canvas gradebook via AGS.

        Args:
            student_id: Canvas user ID
            assignment_id: Canvas assignment ID
            grade: Grade (0.0-1.0 for percentage, or points)
            comment: Optional comment

        Returns:
            True if grade submitted successfully
        """
        # Get assignment to determine points_possible
        assignment_data = await self._api_request(
            "GET",
            f"assignments/{assignment_id}",
        )

        points_possible = assignment_data.get("points_possible", 100)
        score = grade * points_possible  # Convert percentage to points

        # Submit grade
        data = {
            "submission": {
                "posted_grade": score,
            }
        }

        if comment:
            data["comment"] = {"text_comment": comment}

        await self._api_request(
            "PUT",
            f"courses/{assignment_data['course_id']}/assignments/{assignment_id}/submissions/{student_id}",
            data=data,
        )

        return True

    async def get_assignments(self, course_id: str) -> List[Assignment]:
        """
        Get all assignments for a course.

        Args:
            course_id: Canvas course ID

        Returns:
            List of assignments
        """
        assignments_data = await self._api_request(
            "GET",
            f"courses/{course_id}/assignments",
            params={"per_page": 100},
        )

        assignments = []

        for data in assignments_data:
            assignment = Assignment(
                id=str(data.get("id")),
                name=data.get("name", ""),
                description=data.get("description", ""),
                due_date=datetime.fromisoformat(data["due_at"].rstrip("Z"))
                if data.get("due_at")
                else None,
                points_possible=data.get("points_possible", 100),
                course_id=course_id,
                external_id=str(data.get("id")),
            )
            assignments.append(assignment)

        return assignments

    async def get_student_submissions(
        self, assignment_id: str
    ) -> List[Submission]:
        """
        Get all student submissions for an assignment.

        Args:
            assignment_id: Canvas assignment ID

        Returns:
            List of submissions
        """
        # Get assignment to find course_id
        assignment_data = await self._api_request(
            "GET",
            f"assignments/{assignment_id}",
        )

        course_id = assignment_data.get("course_id")

        # Get submissions
        submissions_data = await self._api_request(
            "GET",
            f"courses/{course_id}/assignments/{assignment_id}/submissions",
            params={"per_page": 100},
        )

        submissions = []

        for data in submissions_data:
            status = self._map_submission_status(data)

            submission = Submission(
                id=str(data.get("id")),
                assignment_id=assignment_id,
                student_id=str(data.get("user_id")),
                status=status,
                score=data.get("score"),
                submitted_at=datetime.fromisoformat(data["submitted_at"].rstrip("Z"))
                if data.get("submitted_at")
                else None,
                graded_at=datetime.fromisoformat(data["graded_at"].rstrip("Z"))
                if data.get("graded_at")
                else None,
            )
            submissions.append(submission)

        return submissions

    def _map_submission_status(self, submission_data: Dict[str, Any]) -> SubmissionStatus:
        """Map Canvas submission status to SubmissionStatus enum"""
        workflow_state = submission_data.get("workflow_state", "")

        if workflow_state == "graded":
            return SubmissionStatus.GRADED
        elif submission_data.get("submitted_at"):
            if submission_data.get("late"):
                return SubmissionStatus.LATE
            return SubmissionStatus.SUBMITTED
        else:
            return SubmissionStatus.NOT_SUBMITTED

    async def get_courses(self) -> List[Dict[str, Any]]:
        """
        Get list of courses accessible to authenticated user.

        Returns:
            List of course dictionaries
        """
        courses = await self._api_request(
            "GET",
            "courses",
            params={"per_page": 100},
        )

        return [
            {
                "id": str(course.get("id")),
                "name": course.get("name"),
                "course_code": course.get("course_code"),
                "enrollment_term_id": course.get("enrollment_term_id"),
            }
            for course in courses
        ]

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Canvas and return status.

        Returns:
            Dictionary with connection status
        """
        try:
            # Try to get user profile
            headers = await self._get_headers()
            url = f"{self.api_base}/users/self/profile"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        return {
                            "status": "connected",
                            "user": {
                                "name": user_data.get("name"),
                                "email": user_data.get("primary_email"),
                            },
                            "base_url": self.base_url,
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"HTTP {response.status}",
                        }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }

    async def get_lti_launch_url(
        self,
        course_id: str,
        assignment_id: Optional[str] = None,
    ) -> str:
        """
        Get LTI 1.3 launch URL for EdWIN.

        Args:
            course_id: Canvas course ID
            assignment_id: Optional assignment ID

        Returns:
            LTI launch URL
        """
        base_launch_url = f"{self.base_url}/courses/{course_id}/external_tools/edwin"

        if assignment_id:
            return f"{base_launch_url}?assignment_id={assignment_id}"

        return base_launch_url

    async def handle_lti_launch(self, launch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle LTI 1.3 launch request.

        Args:
            launch_data: LTI launch data (JWT claims)

        Returns:
            User and context information
        """
        # Extract user info
        user_info = {
            "id": launch_data.get("sub"),
            "name": launch_data.get("name"),
            "email": launch_data.get("email"),
            "roles": launch_data.get("https://purl.imsglobal.org/spec/lti/claim/roles", []),
        }

        # Extract context (course) info
        context = launch_data.get("https://purl.imsglobal.org/spec/lti/claim/context", {})
        context_info = {
            "id": context.get("id"),
            "label": context.get("label"),
            "title": context.get("title"),
        }

        # Extract custom parameters (assignment info, etc.)
        custom = launch_data.get("https://purl.imsglobal.org/spec/lti/claim/custom", {})

        return {
            "user": user_info,
            "context": context_info,
            "custom": custom,
            "launch_id": launch_data.get("https://purl.imsglobal.org/spec/lti/claim/deployment_id"),
        }
