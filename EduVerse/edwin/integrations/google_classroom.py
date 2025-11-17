"""
Google Classroom Integration

Implements integration with Google Classroom API.

Features:
- OAuth2 authentication
- Roster synchronization
- Assignment creation and retrieval
- Grade passback
- Materials integration

Author: Agent D
Date: 2025-11-15
"""

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
from .oauth_manager import OAuth2Manager, OAuth2Provider


class GoogleClassroomIntegration:
    """
    Google Classroom integration using Google Classroom API.

    Supports:
    - Roster synchronization
    - Assignment creation and retrieval
    - Grade passback
    - Single sign-on via Google OAuth2
    """

    # Google Classroom API scopes
    REQUIRED_SCOPES = [
        "https://www.googleapis.com/auth/classroom.courses.readonly",
        "https://www.googleapis.com/auth/classroom.rosters.readonly",
        "https://www.googleapis.com/auth/classroom.coursework.students",
        "https://www.googleapis.com/auth/classroom.student-submissions.students.readonly",
        "https://www.googleapis.com/auth/classroom.profile.emails",
    ]

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        oauth_manager: Optional[OAuth2Manager] = None,
    ):
        """
        Initialize Google Classroom integration.

        Args:
            client_id: Google OAuth2 client ID
            client_secret: Google OAuth2 client secret
            oauth_manager: Optional OAuth2Manager instance
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.oauth_manager = oauth_manager or OAuth2Manager()
        self.api_base = "https://classroom.googleapis.com/v1"

    async def authenticate(self) -> bool:
        """
        Authenticate with Google Classroom using OAuth2.

        Returns:
            True if authentication successful
        """
        try:
            token = await self.oauth_manager.get_valid_token(
                OAuth2Provider.GOOGLE,
                "",  # Not used for Google
                self.client_id,
                self.client_secret,
            )
            return token is not None
        except Exception as e:
            print(f"Google Classroom authentication failed: {e}")
            return False

    async def _get_headers(self) -> Dict[str, str]:
        """Get headers with valid access token"""
        token = await self.oauth_manager.get_valid_token(
            OAuth2Provider.GOOGLE,
            "",  # Not used for Google
            self.client_id,
            self.client_secret,
        )

        if not token:
            raise Exception("No valid Google token available")

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
        Make Google Classroom API request.

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
                        f"Google Classroom API error {response.status}: {error_text}"
                    )

                if response.status == 204:  # No content
                    return {}

                return await response.json()

    async def sync_roster(self, course_id: str) -> RosterData:
        """
        Sync course roster from Google Classroom.

        Args:
            course_id: Google Classroom course ID

        Returns:
            RosterData with students and teachers
        """
        # Get course info
        course_data = await self._api_request("GET", f"courses/{course_id}")

        # Get students
        students_data = await self._api_request(
            "GET",
            f"courses/{course_id}/students",
            params={"pageSize": 100},
        )

        # Get teachers
        teachers_data = await self._api_request(
            "GET",
            f"courses/{course_id}/teachers",
            params={"pageSize": 100},
        )

        students = []
        for student_data in students_data.get("students", []):
            profile = student_data.get("profile", {})
            student = Student(
                id=student_data.get("userId", ""),
                name=profile.get("name", {}).get("fullName", ""),
                email=profile.get("emailAddress", ""),
                role=StudentRole.STUDENT,
                external_id=student_data.get("userId"),
            )
            students.append(student)

        teachers = []
        for teacher_data in teachers_data.get("teachers", []):
            profile = teacher_data.get("profile", {})
            teacher = Student(
                id=teacher_data.get("userId", ""),
                name=profile.get("name", {}).get("fullName", ""),
                email=profile.get("emailAddress", ""),
                role=StudentRole.TEACHER,
                external_id=teacher_data.get("userId"),
            )
            teachers.append(teacher)

        return RosterData(
            course_id=course_id,
            course_name=course_data.get("name", ""),
            students=students,
            teachers=teachers,
            sync_timestamp=datetime.now(),
        )

    async def create_assignment(self, assignment: Assignment) -> str:
        """
        Create assignment in Google Classroom.

        Args:
            assignment: Assignment to create

        Returns:
            Google Classroom coursework ID
        """
        # Create coursework
        data = {
            "title": assignment.name,
            "description": assignment.description,
            "maxPoints": assignment.points_possible,
            "workType": "ASSIGNMENT",
            "state": "PUBLISHED",
        }

        if assignment.due_date:
            # Google Classroom expects separate date and time
            due_date = assignment.due_date
            data["dueDate"] = {
                "year": due_date.year,
                "month": due_date.month,
                "day": due_date.day,
            }
            data["dueTime"] = {
                "hours": due_date.hour,
                "minutes": due_date.minute,
            }

        result = await self._api_request(
            "POST",
            f"courses/{assignment.course_id}/courseWork",
            data=data,
        )

        return result.get("id", "")

    async def submit_grade(
        self,
        student_id: str,
        assignment_id: str,
        grade: float,
        comment: Optional[str] = None,
    ) -> bool:
        """
        Submit grade to Google Classroom.

        Args:
            student_id: Google user ID
            assignment_id: Google Classroom coursework ID
            grade: Grade (0.0-1.0 for percentage)
            comment: Optional comment

        Returns:
            True if grade submitted successfully
        """
        # Get coursework to find course_id and max_points
        coursework = await self._api_request(
            "GET",
            f"courses/-/courseWork/{assignment_id}",
        )

        course_id = coursework.get("courseId")
        max_points = coursework.get("maxPoints", 100)
        points = grade * max_points

        # Get submission ID
        submissions = await self._api_request(
            "GET",
            f"courses/{course_id}/courseWork/{assignment_id}/studentSubmissions",
            params={"userId": student_id},
        )

        if not submissions.get("studentSubmissions"):
            raise Exception(f"No submission found for student {student_id}")

        submission_id = submissions["studentSubmissions"][0]["id"]

        # Submit grade
        data = {
            "assignedGrade": points,
        }

        await self._api_request(
            "PATCH",
            f"courses/{course_id}/courseWork/{assignment_id}/studentSubmissions/{submission_id}",
            data=data,
            params={"updateMask": "assignedGrade"},
        )

        # Add comment if provided
        if comment:
            comment_data = {
                "text": comment,
            }
            await self._api_request(
                "POST",
                f"courses/{course_id}/courseWork/{assignment_id}/studentSubmissions/{submission_id}/addOnAttachments/-/studentSubmissions/{submission_id}/comments",
                data=comment_data,
            )

        return True

    async def get_assignments(self, course_id: str) -> List[Assignment]:
        """
        Get all assignments for a course.

        Args:
            course_id: Google Classroom course ID

        Returns:
            List of assignments
        """
        coursework_data = await self._api_request(
            "GET",
            f"courses/{course_id}/courseWork",
            params={"pageSize": 100},
        )

        assignments = []

        for data in coursework_data.get("courseWork", []):
            # Parse due date
            due_date = None
            if "dueDate" in data and "dueTime" in data:
                date_dict = data["dueDate"]
                time_dict = data.get("dueTime", {"hours": 23, "minutes": 59})
                due_date = datetime(
                    year=date_dict["year"],
                    month=date_dict["month"],
                    day=date_dict["day"],
                    hour=time_dict.get("hours", 23),
                    minute=time_dict.get("minutes", 59),
                )

            assignment = Assignment(
                id=data.get("id", ""),
                name=data.get("title", ""),
                description=data.get("description", ""),
                due_date=due_date,
                points_possible=data.get("maxPoints", 100),
                course_id=course_id,
                external_id=data.get("id"),
            )
            assignments.append(assignment)

        return assignments

    async def get_student_submissions(
        self, assignment_id: str
    ) -> List[Submission]:
        """
        Get all student submissions for an assignment.

        Args:
            assignment_id: Google Classroom coursework ID

        Returns:
            List of submissions
        """
        # Get coursework to find course_id
        coursework = await self._api_request(
            "GET",
            f"courses/-/courseWork/{assignment_id}",
        )

        course_id = coursework.get("courseId")

        # Get submissions
        submissions_data = await self._api_request(
            "GET",
            f"courses/{course_id}/courseWork/{assignment_id}/studentSubmissions",
            params={"pageSize": 100},
        )

        submissions = []

        for data in submissions_data.get("studentSubmissions", []):
            status = self._map_submission_status(data)

            # Parse timestamps
            submitted_at = None
            if "updateTime" in data:
                submitted_at = datetime.fromisoformat(
                    data["updateTime"].rstrip("Z")
                )

            graded_at = None
            if data.get("assignedGrade") is not None:
                graded_at = submitted_at  # Google doesn't track separate graded time

            submission = Submission(
                id=data.get("id", ""),
                assignment_id=assignment_id,
                student_id=data.get("userId", ""),
                status=status,
                score=data.get("assignedGrade"),
                submitted_at=submitted_at,
                graded_at=graded_at,
            )
            submissions.append(submission)

        return submissions

    def _map_submission_status(
        self, submission_data: Dict[str, Any]
    ) -> SubmissionStatus:
        """Map Google Classroom submission status to SubmissionStatus enum"""
        state = submission_data.get("state", "")

        if state == "TURNED_IN" or state == "RETURNED":
            if submission_data.get("assignedGrade") is not None:
                return SubmissionStatus.GRADED
            elif submission_data.get("late"):
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
        courses_data = await self._api_request(
            "GET",
            "courses",
            params={"pageSize": 100},
        )

        return [
            {
                "id": course.get("id"),
                "name": course.get("name"),
                "section": course.get("section"),
                "descriptionHeading": course.get("descriptionHeading"),
            }
            for course in courses_data.get("courses", [])
        ]

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Google Classroom and return status.

        Returns:
            Dictionary with connection status
        """
        try:
            # Try to get user profile
            headers = await self._get_headers()
            url = "https://www.googleapis.com/oauth2/v2/userinfo"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        user_data = await response.json()
                        return {
                            "status": "connected",
                            "user": {
                                "name": user_data.get("name"),
                                "email": user_data.get("email"),
                            },
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

    async def add_material_to_course(
        self,
        course_id: str,
        title: str,
        description: str,
        link_url: str,
    ) -> str:
        """
        Add EdWIN material to Google Classroom stream.

        Args:
            course_id: Google Classroom course ID
            title: Material title
            description: Material description
            link_url: URL to EdWIN content

        Returns:
            Material ID
        """
        data = {
            "title": title,
            "description": description,
            "materials": [
                {
                    "link": {
                        "url": link_url,
                        "title": title,
                    }
                }
            ],
        }

        result = await self._api_request(
            "POST",
            f"courses/{course_id}/announcements",
            data=data,
        )

        return result.get("id", "")

    async def share_to_google_drive(
        self,
        file_name: str,
        file_content: bytes,
        mime_type: str = "application/pdf",
    ) -> str:
        """
        Share EdWIN report to Google Drive.

        Args:
            file_name: File name
            file_content: File content as bytes
            mime_type: MIME type

        Returns:
            Google Drive file ID
        """
        # Note: This requires Google Drive API scope
        # https://www.googleapis.com/auth/drive.file

        headers = await self._get_headers()
        headers["Content-Type"] = "multipart/related; boundary=foo_bar_baz"

        # Multipart upload
        metadata = {
            "name": file_name,
            "mimeType": mime_type,
        }

        body = (
            "--foo_bar_baz\r\n"
            "Content-Type: application/json; charset=UTF-8\r\n\r\n"
            f"{metadata}\r\n"
            "--foo_bar_baz\r\n"
            f"Content-Type: {mime_type}\r\n\r\n"
            f"{file_content.decode('latin1')}\r\n"
            "--foo_bar_baz--"
        )

        url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=body, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Google Drive upload failed: {error_text}")

                result = await response.json()
                return result.get("id", "")
