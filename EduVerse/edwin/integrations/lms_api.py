"""
LMS Integration API Endpoints

FastAPI routes for LMS integration management.

Author: Agent D
Date: 2025-11-15
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime

from .oauth_manager import OAuth2Manager, OAuth2Provider
from .canvas_integration import CanvasIntegration
from .google_classroom import GoogleClassroomIntegration
from .assignment_mapper import AssignmentMapper, GradingScheme
from .gradebook_sync import GradebookSync, SyncStrategy, SyncDirection, ConflictResolution
from .roster_manager import RosterManager
from .lms_webhooks import LMSWebhooksHandler, register_default_handlers
from .migration import LMSMigration


# Pydantic models for API

class ConnectCanvasRequest(BaseModel):
    base_url: str
    client_id: str
    client_secret: str


class ConnectGoogleRequest(BaseModel):
    client_id: str
    client_secret: str


class MapAssignmentRequest(BaseModel):
    lms_assignment_id: str
    lms_assignment_name: str
    edwin_objectives: List[str]
    grading_scheme: str = "mastery_percentage"
    grading_weights: Optional[Dict[str, float]] = None
    max_points: float = 100.0


class SyncGradesRequest(BaseModel):
    assignment_id: str
    student_scores: Dict[str, Dict[str, float]]
    sync_strategy: str = "immediate"
    conflict_resolution: str = "manual"


class SyncRosterRequest(BaseModel):
    course_id: str


class WebhookRequest(BaseModel):
    """Generic webhook request"""
    pass


def create_lms_api_routes() -> APIRouter:
    """
    Create LMS integration API routes.

    Returns:
        FastAPI APIRouter
    """
    router = APIRouter(prefix="/integrations", tags=["lms"])

    # Global instances (in production, these would be in app state)
    oauth_manager = OAuth2Manager()
    canvas_integration: Optional[CanvasIntegration] = None
    google_integration: Optional[GoogleClassroomIntegration] = None
    assignment_mapper = AssignmentMapper()
    webhooks_handler = LMSWebhooksHandler()

    # Register default webhook handlers
    register_default_handlers(webhooks_handler)

    @router.post("/canvas/connect")
    async def connect_canvas(request: ConnectCanvasRequest) -> Dict[str, Any]:
        """
        Connect to Canvas LMS.

        Initiates OAuth2 flow.
        """
        nonlocal canvas_integration

        try:
            # Create Canvas integration
            canvas_integration = CanvasIntegration(
                base_url=request.base_url,
                client_id=request.client_id,
                client_secret=request.client_secret,
                oauth_manager=oauth_manager,
            )

            # Generate OAuth URL
            scopes = [
                "url:GET|/api/v1/courses",
                "url:GET|/api/v1/courses/:course_id/enrollments",
                "url:GET|/api/v1/courses/:course_id/assignments",
                "url:PUT|/api/v1/courses/:course_id/assignments/:id/submissions/:user_id",
            ]

            auth_url = oauth_manager.generate_auth_url(
                provider=OAuth2Provider.CANVAS,
                base_url=request.base_url,
                client_id=request.client_id,
                redirect_uri="http://localhost:8000/integrations/canvas/callback",
                scopes=scopes,
            )

            return {
                "status": "pending",
                "auth_url": auth_url,
                "message": "Visit auth_url to complete OAuth flow",
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/canvas/callback")
    async def canvas_callback(code: str, state: str) -> Dict[str, Any]:
        """Handle Canvas OAuth callback"""
        if not canvas_integration:
            raise HTTPException(status_code=400, detail="Canvas not initialized")

        try:
            # Exchange code for token
            token = await oauth_manager.exchange_code_for_token(
                provider=OAuth2Provider.CANVAS,
                base_url=canvas_integration.base_url,
                client_id=canvas_integration.client_id,
                client_secret=canvas_integration.client_secret,
                redirect_uri="http://localhost:8000/integrations/canvas/callback",
                code=code,
            )

            return {
                "status": "connected",
                "expires_at": token.expires_at.isoformat(),
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/google/connect")
    async def connect_google(request: ConnectGoogleRequest) -> Dict[str, Any]:
        """
        Connect to Google Classroom.

        Initiates OAuth2 flow.
        """
        nonlocal google_integration

        try:
            # Create Google integration
            google_integration = GoogleClassroomIntegration(
                client_id=request.client_id,
                client_secret=request.client_secret,
                oauth_manager=oauth_manager,
            )

            # Generate OAuth URL
            auth_url = oauth_manager.generate_auth_url(
                provider=OAuth2Provider.GOOGLE,
                base_url="",  # Not used for Google
                client_id=request.client_id,
                redirect_uri="http://localhost:8000/integrations/google/callback",
                scopes=GoogleClassroomIntegration.REQUIRED_SCOPES,
            )

            return {
                "status": "pending",
                "auth_url": auth_url,
                "message": "Visit auth_url to complete OAuth flow",
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/google/callback")
    async def google_callback(code: str, state: str) -> Dict[str, Any]:
        """Handle Google OAuth callback"""
        if not google_integration:
            raise HTTPException(status_code=400, detail="Google not initialized")

        try:
            # Exchange code for token
            token = await oauth_manager.exchange_code_for_token(
                provider=OAuth2Provider.GOOGLE,
                base_url="",  # Not used for Google
                client_id=google_integration.client_id,
                client_secret=google_integration.client_secret,
                redirect_uri="http://localhost:8000/integrations/google/callback",
                code=code,
            )

            return {
                "status": "connected",
                "expires_at": token.expires_at.isoformat(),
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/courses")
    async def get_courses(provider: str = "canvas") -> List[Dict[str, Any]]:
        """Get list of courses from LMS"""
        try:
            if provider == "canvas" and canvas_integration:
                courses = await canvas_integration.get_courses()
            elif provider == "google" and google_integration:
                courses = await google_integration.get_courses()
            else:
                raise HTTPException(status_code=400, detail="LMS not connected")

            return courses

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/roster/sync")
    async def sync_roster(
        request: SyncRosterRequest,
        provider: str = "canvas",
    ) -> Dict[str, Any]:
        """Sync course roster from LMS"""
        try:
            if provider == "canvas" and canvas_integration:
                lms = canvas_integration
            elif provider == "google" and google_integration:
                lms = google_integration
            else:
                raise HTTPException(status_code=400, detail="LMS not connected")

            # Get EdWIN students (mock for now)
            edwin_students = []

            roster_manager = RosterManager(lms)
            result = await roster_manager.sync_roster(
                request.course_id, edwin_students
            )

            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/assignments/map")
    async def map_assignment(request: MapAssignmentRequest) -> Dict[str, Any]:
        """Map LMS assignment to EdWIN objectives"""
        try:
            mapping = assignment_mapper.create_mapping(
                lms_assignment_id=request.lms_assignment_id,
                lms_assignment_name=request.lms_assignment_name,
                edwin_objectives=request.edwin_objectives,
                grading_scheme=GradingScheme(request.grading_scheme),
                grading_weights=request.grading_weights,
                max_points=request.max_points,
            )

            return {
                "status": "success",
                "mapping": {
                    "assignment_id": mapping.lms_assignment_id,
                    "assignment_name": mapping.lms_assignment_name,
                    "objectives": mapping.edwin_objectives,
                    "grading_scheme": mapping.grading_scheme.value,
                },
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/grades/sync")
    async def sync_grades(
        request: SyncGradesRequest,
        provider: str = "canvas",
    ) -> Dict[str, Any]:
        """Sync grades to LMS"""
        try:
            if provider == "canvas" and canvas_integration:
                lms = canvas_integration
            elif provider == "google" and google_integration:
                lms = google_integration
            else:
                raise HTTPException(status_code=400, detail="LMS not connected")

            gradebook = GradebookSync(lms, assignment_mapper)

            logs = await gradebook.batch_sync(
                assignment_id=request.assignment_id,
                student_scores=request.student_scores,
                direction=SyncDirection.EDWIN_TO_LMS,
                conflict_resolution=ConflictResolution(request.conflict_resolution),
            )

            return {
                "status": "success",
                "synced": len([log for log in logs if log.success]),
                "failed": len([log for log in logs if not log.success]),
                "logs": [
                    {
                        "student_id": log.student_id,
                        "success": log.success,
                        "error": log.error,
                    }
                    for log in logs
                ],
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/sync-logs")
    async def get_sync_logs(limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent sync logs"""
        # This would retrieve from gradebook sync instance
        # Simplified for now
        return []

    @router.post("/test-connection")
    async def test_connection(provider: str = "canvas") -> Dict[str, Any]:
        """Test connection to LMS"""
        try:
            if provider == "canvas" and canvas_integration:
                result = await canvas_integration.test_connection()
            elif provider == "google" and google_integration:
                result = await google_integration.test_connection()
            else:
                raise HTTPException(status_code=400, detail="LMS not connected")

            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/webhooks/canvas")
    async def canvas_webhook(request: Request):
        """Handle Canvas webhook"""
        headers = dict(request.headers)
        body = await request.body()

        result = await webhooks_handler.handle_canvas_webhook(headers, body)
        return result

    @router.post("/webhooks/google")
    async def google_webhook(request: Request):
        """Handle Google Classroom push notification"""
        headers = dict(request.headers)
        body = await request.body()

        result = await webhooks_handler.handle_google_webhook(headers, body)
        return result

    return router
