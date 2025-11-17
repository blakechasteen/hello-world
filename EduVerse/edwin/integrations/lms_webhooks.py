"""
LMS Webhooks Handler

Real-time event handling for Canvas and Google Classroom.

Author: Agent D
Date: 2025-11-15
"""

from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import hashlib
import hmac


class WebhookEvent(Enum):
    """Webhook event types"""
    # Canvas events
    ENROLLMENT_CREATED = "enrollment_created"
    ENROLLMENT_UPDATED = "enrollment_updated"
    ENROLLMENT_DELETED = "enrollment_deleted"
    ASSIGNMENT_CREATED = "assignment_created"
    ASSIGNMENT_UPDATED = "assignment_updated"
    GRADE_CHANGED = "grade_changed"
    SUBMISSION_CREATED = "submission_created"

    # Google Classroom events (push notifications)
    COURSE_WORK_CREATED = "course_work_created"
    COURSE_WORK_CHANGED = "course_work_changed"
    STUDENT_SUBMISSION_CREATED = "student_submission_created"
    STUDENT_SUBMISSION_CHANGED = "student_submission_changed"


@dataclass
class WebhookPayload:
    """Webhook payload"""
    event_type: WebhookEvent
    timestamp: datetime
    data: Dict[str, Any]
    source: str  # "canvas" or "google"
    raw_payload: Dict[str, Any]


class LMSWebhooksHandler:
    """
    Real-time event handling for LMS webhooks.

    Features:
    - Canvas webhook verification
    - Google Classroom push notification handling
    - Event handlers registration
    - Automatic EdWIN updates
    """

    def __init__(self, webhook_secret: Optional[str] = None):
        """
        Initialize webhooks handler.

        Args:
            webhook_secret: Secret for webhook signature verification
        """
        self.webhook_secret = webhook_secret
        self.handlers: Dict[WebhookEvent, List[Callable]] = {}

    def register_handler(
        self,
        event_type: WebhookEvent,
        handler: Callable[[WebhookPayload], None],
    ):
        """
        Register event handler.

        Args:
            event_type: Event type to handle
            handler: Handler function
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []

        self.handlers[event_type].append(handler)

    async def handle_canvas_webhook(
        self,
        headers: Dict[str, str],
        body: bytes,
    ) -> Dict[str, Any]:
        """
        Handle Canvas webhook.

        Args:
            headers: Request headers
            body: Request body

        Returns:
            Processing result
        """
        # Verify signature
        if not self._verify_canvas_signature(headers, body):
            return {"status": "error", "message": "Invalid signature"}

        # Parse payload
        try:
            payload_data = json.loads(body)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON"}

        # Determine event type
        event_type = self._map_canvas_event(payload_data)
        if not event_type:
            return {"status": "ignored", "message": "Unknown event type"}

        # Create webhook payload
        payload = WebhookPayload(
            event_type=event_type,
            timestamp=datetime.now(),
            data=self._extract_canvas_data(payload_data),
            source="canvas",
            raw_payload=payload_data,
        )

        # Trigger handlers
        await self._trigger_handlers(payload)

        return {"status": "success", "event": event_type.value}

    async def handle_google_webhook(
        self,
        headers: Dict[str, str],
        body: bytes,
    ) -> Dict[str, Any]:
        """
        Handle Google Classroom push notification.

        Args:
            headers: Request headers
            body: Request body

        Returns:
            Processing result
        """
        # Parse payload
        try:
            payload_data = json.loads(body)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON"}

        # Determine event type
        event_type = self._map_google_event(payload_data)
        if not event_type:
            return {"status": "ignored", "message": "Unknown event type"}

        # Create webhook payload
        payload = WebhookPayload(
            event_type=event_type,
            timestamp=datetime.now(),
            data=self._extract_google_data(payload_data),
            source="google",
            raw_payload=payload_data,
        )

        # Trigger handlers
        await self._trigger_handlers(payload)

        return {"status": "success", "event": event_type.value}

    def _verify_canvas_signature(
        self,
        headers: Dict[str, str],
        body: bytes,
    ) -> bool:
        """Verify Canvas webhook signature"""
        if not self.webhook_secret:
            return True  # Skip verification if no secret configured

        signature_header = headers.get("X-Canvas-Signature")
        if not signature_header:
            return False

        # Calculate expected signature
        expected_signature = hmac.new(
            self.webhook_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(signature_header, expected_signature)

    def _map_canvas_event(
        self,
        payload: Dict[str, Any],
    ) -> Optional[WebhookEvent]:
        """Map Canvas event to WebhookEvent enum"""
        event_type = payload.get("metadata", {}).get("event_name", "")

        mapping = {
            "enrollment_created": WebhookEvent.ENROLLMENT_CREATED,
            "enrollment_updated": WebhookEvent.ENROLLMENT_UPDATED,
            "enrollment_state_updated": WebhookEvent.ENROLLMENT_UPDATED,
            "enrollment_deleted": WebhookEvent.ENROLLMENT_DELETED,
            "assignment_created": WebhookEvent.ASSIGNMENT_CREATED,
            "assignment_updated": WebhookEvent.ASSIGNMENT_UPDATED,
            "grade_change": WebhookEvent.GRADE_CHANGED,
            "submission_created": WebhookEvent.SUBMISSION_CREATED,
        }

        return mapping.get(event_type)

    def _extract_canvas_data(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract relevant data from Canvas webhook"""
        body = payload.get("body", {})

        return {
            "course_id": body.get("course_id"),
            "user_id": body.get("user_id"),
            "assignment_id": body.get("assignment_id"),
            "enrollment_id": body.get("enrollment_id"),
            "grade": body.get("grade"),
            "score": body.get("score"),
            "workflow_state": body.get("workflow_state"),
        }

    def _map_google_event(
        self,
        payload: Dict[str, Any],
    ) -> Optional[WebhookEvent]:
        """Map Google event to WebhookEvent enum"""
        # Google Classroom uses Cloud Pub/Sub notifications
        message = payload.get("message", {})
        attributes = message.get("attributes", {})

        event_type = attributes.get("eventType", "")

        mapping = {
            "COURSE_WORK_CREATED": WebhookEvent.COURSE_WORK_CREATED,
            "COURSE_WORK_CHANGED": WebhookEvent.COURSE_WORK_CHANGED,
            "COURSE_WORK_DELETED": WebhookEvent.COURSE_WORK_CHANGED,
            "STUDENT_SUBMISSION_CREATED": WebhookEvent.STUDENT_SUBMISSION_CREATED,
            "STUDENT_SUBMISSION_CHANGED": WebhookEvent.STUDENT_SUBMISSION_CHANGED,
        }

        return mapping.get(event_type)

    def _extract_google_data(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract relevant data from Google webhook"""
        message = payload.get("message", {})
        attributes = message.get("attributes", {})

        return {
            "course_id": attributes.get("courseId"),
            "course_work_id": attributes.get("courseWorkId"),
            "student_id": attributes.get("studentId"),
            "submission_id": attributes.get("submissionId"),
        }

    async def _trigger_handlers(self, payload: WebhookPayload):
        """Trigger registered handlers for event"""
        handlers = self.handlers.get(payload.event_type, [])

        for handler in handlers:
            try:
                await handler(payload)
            except Exception as e:
                print(f"Handler error for {payload.event_type}: {e}")


# Example handlers

async def handle_enrollment_created(payload: WebhookPayload):
    """
    Handle student enrollment.

    Auto-add student to EdWIN classroom.
    """
    print(f"Student enrolled: {payload.data}")

    # Implementation would:
    # 1. Get student info from LMS
    # 2. Create or update student in EdWIN
    # 3. Add to classroom
    # 4. Send welcome notification


async def handle_enrollment_deleted(payload: WebhookPayload):
    """
    Handle student drop.

    Archive student in EdWIN (preserve progress).
    """
    print(f"Student dropped: {payload.data}")

    # Implementation would:
    # 1. Mark student as inactive in EdWIN
    # 2. Archive progress data
    # 3. Remove from active classroom roster
    # 4. Send notification to teacher


async def handle_assignment_created(payload: WebhookPayload):
    """
    Handle new assignment in LMS.

    Notify teacher to map to EdWIN objectives.
    """
    print(f"Assignment created: {payload.data}")

    # Implementation would:
    # 1. Get assignment details from LMS
    # 2. Check if already mapped
    # 3. Send notification to teacher if unmapped
    # 4. Suggest mapping based on description


async def handle_grade_changed(payload: WebhookPayload):
    """
    Handle grade change in LMS.

    Update EdWIN if teacher manually adjusted grade.
    """
    print(f"Grade changed: {payload.data}")

    # Implementation would:
    # 1. Get new grade from LMS
    # 2. Compare with EdWIN grade
    # 3. If different, update EdWIN (LMS wins)
    # 4. Log the override


async def handle_submission_created(payload: WebhookPayload):
    """
    Handle student submission.

    Track submission in EdWIN.
    """
    print(f"Submission created: {payload.data}")

    # Implementation would:
    # 1. Get submission details
    # 2. Update EdWIN student model
    # 3. Trigger grading if applicable


# Default handler registration

def register_default_handlers(handler: LMSWebhooksHandler):
    """Register default event handlers"""
    handler.register_handler(
        WebhookEvent.ENROLLMENT_CREATED, handle_enrollment_created
    )
    handler.register_handler(
        WebhookEvent.ENROLLMENT_DELETED, handle_enrollment_deleted
    )
    handler.register_handler(
        WebhookEvent.ASSIGNMENT_CREATED, handle_assignment_created
    )
    handler.register_handler(WebhookEvent.GRADE_CHANGED, handle_grade_changed)
    handler.register_handler(
        WebhookEvent.SUBMISSION_CREATED, handle_submission_created
    )
