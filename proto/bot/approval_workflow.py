#!/usr/bin/env python3
"""
Approval Workflow System for Promptly Matrix Bot

Enables team collaboration through reaction-based voting on actions:
- Deploy prompts to production
- Execute high-risk workflows
- Modify shared configurations
- Delete saved prompts

Features:
- Reaction-based voting (✅/❌)
- Multi-user approval requirements
- Configurable thresholds
- Timeout handling
- Audit trail
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Approval request status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ActionRisk(Enum):
    """Risk level for actions"""
    LOW = "low"          # No approval needed
    MEDIUM = "medium"    # 1 approval required
    HIGH = "high"        # 2+ approvals required
    CRITICAL = "critical"  # All team members must approve


@dataclass
class ApprovalRequest:
    """Approval request data"""
    request_id: str
    room_id: str
    initiator: str
    action: str
    context: Dict[str, Any]
    risk_level: ActionRisk
    required_approvals: int
    approvals: List[str] = field(default_factory=list)
    rejections: List[str] = field(default_factory=list)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created: datetime = field(default_factory=datetime.now)
    expires: Optional[datetime] = None
    message_id: Optional[str] = None  # Event ID of approval message

    def is_expired(self) -> bool:
        """Check if request has expired"""
        if self.expires and datetime.now() > self.expires:
            self.status = ApprovalStatus.EXPIRED
            return True
        return False

    def add_approval(self, user_id: str) -> bool:
        """Add approval vote"""
        if user_id in self.approvals or user_id in self.rejections:
            return False  # Already voted

        self.approvals.append(user_id)

        # Check if approved
        if len(self.approvals) >= self.required_approvals:
            self.status = ApprovalStatus.APPROVED

        return True

    def add_rejection(self, user_id: str) -> bool:
        """Add rejection vote"""
        if user_id in self.approvals or user_id in self.rejections:
            return False  # Already voted

        self.rejections.append(user_id)

        # Any rejection blocks approval
        self.status = ApprovalStatus.REJECTED
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "request_id": self.request_id,
            "room_id": self.room_id,
            "initiator": self.initiator,
            "action": self.action,
            "context": self.context,
            "risk_level": self.risk_level.value,
            "required_approvals": self.required_approvals,
            "approvals": self.approvals,
            "rejections": self.rejections,
            "status": self.status.value,
            "created": self.created.isoformat(),
            "expires": self.expires.isoformat() if self.expires else None,
            "message_id": self.message_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApprovalRequest':
        """Create from dictionary"""
        return cls(
            request_id=data["request_id"],
            room_id=data["room_id"],
            initiator=data["initiator"],
            action=data["action"],
            context=data["context"],
            risk_level=ActionRisk(data["risk_level"]),
            required_approvals=data["required_approvals"],
            approvals=data.get("approvals", []),
            rejections=data.get("rejections", []),
            status=ApprovalStatus(data["status"]),
            created=datetime.fromisoformat(data["created"]),
            expires=datetime.fromisoformat(data["expires"]) if data.get("expires") else None,
            message_id=data.get("message_id")
        )


class ApprovalWorkflowManager:
    """
    Manages approval workflows for team actions.

    Usage:
        manager = ApprovalWorkflowManager(state_manager, bot_client)

        # Request approval
        request = await manager.request_approval(
            room_id="!room:matrix.org",
            initiator="@user:matrix.org",
            action="deploy_prompt",
            context={"prompt_name": "customer_support_qa"},
            risk_level=ActionRisk.HIGH
        )

        # Handle reaction
        await manager.handle_reaction(
            room_id=room_id,
            event_id=event_id,
            user_id=user_id,
            reaction="✅"
        )
    """

    def __init__(self, state_manager, bot_client=None):
        """
        Initialize approval workflow manager.

        Args:
            state_manager: StateManager instance
            bot_client: Matrix client (for sending messages/reactions)
        """
        self.state = state_manager
        self.client = bot_client
        self.pending_requests: Dict[str, ApprovalRequest] = {}

        # Approval thresholds by risk level
        self.thresholds = {
            ActionRisk.LOW: 0,       # No approval needed
            ActionRisk.MEDIUM: 1,    # 1 approval
            ActionRisk.HIGH: 2,      # 2 approvals
            ActionRisk.CRITICAL: 3   # 3+ approvals
        }

        # Timeout durations (in hours)
        self.timeout_hours = {
            ActionRisk.LOW: None,
            ActionRisk.MEDIUM: 24,
            ActionRisk.HIGH: 48,
            ActionRisk.CRITICAL: 72
        }

    async def request_approval(
        self,
        room_id: str,
        initiator: str,
        action: str,
        context: Dict[str, Any],
        risk_level: ActionRisk = ActionRisk.MEDIUM,
        required_approvals: Optional[int] = None
    ) -> ApprovalRequest:
        """
        Request approval for an action.

        Args:
            room_id: Matrix room ID
            initiator: User requesting approval
            action: Action being requested
            context: Action context/details
            risk_level: Risk level of action
            required_approvals: Override default threshold

        Returns:
            ApprovalRequest object
        """
        # Generate request ID
        request_id = f"approval_{room_id}_{action}_{int(datetime.now().timestamp())}"

        # Determine required approvals
        if required_approvals is None:
            required_approvals = self.thresholds.get(risk_level, 1)

        # Calculate expiry
        expires = None
        timeout_hours = self.timeout_hours.get(risk_level)
        if timeout_hours:
            expires = datetime.now() + timedelta(hours=timeout_hours)

        # Create request
        request = ApprovalRequest(
            request_id=request_id,
            room_id=room_id,
            initiator=initiator,
            action=action,
            context=context,
            risk_level=risk_level,
            required_approvals=required_approvals,
            expires=expires
        )

        # Store in state manager
        self.state.create_approval_request(
            room_id=room_id,
            request_id=request_id,
            data=request.to_dict(),
            required_approvals=required_approvals
        )

        # Track locally
        self.pending_requests[request_id] = request

        # Send approval message to room
        if self.client:
            message_id = await self._send_approval_message(request)
            request.message_id = message_id

        logger.info(f"Created approval request {request_id} for action '{action}' in {room_id}")

        return request

    async def handle_reaction(
        self,
        room_id: str,
        event_id: str,
        user_id: str,
        reaction: str
    ) -> Optional[ApprovalRequest]:
        """
        Handle reaction to approval message.

        Args:
            room_id: Room where reaction occurred
            event_id: Message event ID
            user_id: User who reacted
            reaction: Reaction emoji (✅ or ❌)

        Returns:
            Updated ApprovalRequest if reaction was on approval message
        """
        # Find request by message ID
        request = None
        for req in self.pending_requests.values():
            if req.message_id == event_id and req.room_id == room_id:
                request = req
                break

        if not request:
            return None

        # Check if already completed
        if request.status != ApprovalStatus.PENDING:
            return request

        # Check if expired
        if request.is_expired():
            await self._notify_expired(request)
            return request

        # Don't allow initiator to vote
        if user_id == request.initiator:
            logger.info(f"Ignoring vote from initiator {user_id}")
            return request

        # Handle approval/rejection
        if reaction == "✅":
            added = request.add_approval(user_id)
            if added:
                logger.info(f"User {user_id} approved request {request.request_id}")

                # Update state
                self.state.add_approval(room_id, request.request_id, user_id, True)

                # Check if approved
                if request.status == ApprovalStatus.APPROVED:
                    await self._notify_approved(request)

        elif reaction == "❌":
            added = request.add_rejection(user_id)
            if added:
                logger.info(f"User {user_id} rejected request {request.request_id}")

                # Update state
                self.state.add_approval(room_id, request.request_id, user_id, False)

                # Notify rejection
                await self._notify_rejected(request)

        return request

    async def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get approval request by ID"""
        # Check local cache
        if request_id in self.pending_requests:
            return self.pending_requests[request_id]

        # Check state manager
        # Note: state_manager stores in different format, need to adapt
        return None

    async def cancel_request(self, request_id: str, canceller: str) -> bool:
        """
        Cancel approval request.

        Args:
            request_id: Request ID
            canceller: User cancelling request

        Returns:
            True if cancelled, False if not found
        """
        request = await self.get_request(request_id)
        if not request:
            return False

        # Only initiator can cancel
        if canceller != request.initiator:
            logger.warning(f"User {canceller} tried to cancel request initiated by {request.initiator}")
            return False

        request.status = ApprovalStatus.REJECTED
        del self.pending_requests[request_id]

        if self.client:
            await self._notify_cancelled(request)

        logger.info(f"Request {request_id} cancelled by {canceller}")
        return True

    async def _send_approval_message(self, request: ApprovalRequest) -> str:
        """Send approval request message to room"""
        if not self.client:
            return ""

        # Format message
        action_desc = self._format_action(request.action, request.context)
        risk_emoji = self._risk_emoji(request.risk_level)

        body = f"""🔔 **Approval Required** {risk_emoji}

**Action:** {action_desc}
**Requested by:** {request.initiator}
**Risk Level:** {request.risk_level.value.upper()}
**Approvals needed:** {request.required_approvals}

React with:
✅ to approve
❌ to reject

"""

        if request.expires:
            expires_in = request.expires - datetime.now()
            hours = int(expires_in.total_seconds() / 3600)
            body += f"**Expires in:** {hours} hours\n"

        # Send message
        response = await self.client.room_send(
            room_id=request.room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.notice",
                "body": body
            }
        )

        return response.event_id

    async def _notify_approved(self, request: ApprovalRequest):
        """Notify that request was approved"""
        if not self.client:
            return

        body = f"""✅ **Approved!**

Action '{request.action}' has been approved.

**Approvers:** {', '.join(request.approvals)}

Executing now...
"""

        await self.client.room_send(
            room_id=request.room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.notice",
                "body": body,
                "m.relates_to": {
                    "rel_type": "m.thread",
                    "event_id": request.message_id
                }
            }
        )

    async def _notify_rejected(self, request: ApprovalRequest):
        """Notify that request was rejected"""
        if not self.client:
            return

        body = f"""❌ **Rejected**

Action '{request.action}' has been rejected.

**Rejector:** {request.rejections[-1]}

Request cancelled.
"""

        await self.client.room_send(
            room_id=request.room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.notice",
                "body": body,
                "m.relates_to": {
                    "rel_type": "m.thread",
                    "event_id": request.message_id
                }
            }
        )

    async def _notify_expired(self, request: ApprovalRequest):
        """Notify that request expired"""
        if not self.client:
            return

        body = f"""⏰ **Expired**

Approval request for '{request.action}' has expired.

**Status:** {len(request.approvals)}/{request.required_approvals} approvals received

Request cancelled.
"""

        await self.client.room_send(
            room_id=request.room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.notice",
                "body": body,
                "m.relates_to": {
                    "rel_type": "m.thread",
                    "event_id": request.message_id
                }
            }
        )

    async def _notify_cancelled(self, request: ApprovalRequest):
        """Notify that request was cancelled"""
        if not self.client:
            return

        body = f"🚫 Approval request cancelled by {request.initiator}"

        await self.client.room_send(
            room_id=request.room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.notice",
                "body": body,
                "m.relates_to": {
                    "rel_type": "m.thread",
                    "event_id": request.message_id
                }
            }
        )

    def _format_action(self, action: str, context: Dict[str, Any]) -> str:
        """Format action description"""
        if action == "deploy_prompt":
            return f"Deploy prompt '{context.get('prompt_name', 'unknown')}' to production"
        elif action == "delete_prompt":
            return f"Delete prompt '{context.get('prompt_name', 'unknown')}'"
        elif action == "modify_config":
            return f"Modify configuration: {context.get('setting', 'unknown')}"
        elif action == "execute_workflow":
            return f"Execute workflow '{context.get('workflow', 'unknown')}'"
        else:
            return action

    def _risk_emoji(self, risk_level: ActionRisk) -> str:
        """Get emoji for risk level"""
        return {
            ActionRisk.LOW: "🟢",
            ActionRisk.MEDIUM: "🟡",
            ActionRisk.HIGH: "🟠",
            ActionRisk.CRITICAL: "🔴"
        }.get(risk_level, "⚪")


# Singleton
_approval_manager = None

def get_approval_manager(state_manager, bot_client=None) -> ApprovalWorkflowManager:
    """Get singleton approval manager"""
    global _approval_manager
    if _approval_manager is None:
        _approval_manager = ApprovalWorkflowManager(state_manager, bot_client)
    return _approval_manager


# Test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from bot.state_manager import StateManager

    # Test approval workflow
    async def test_approval():
        state = StateManager()
        manager = ApprovalWorkflowManager(state)

        # Request approval
        request = await manager.request_approval(
            room_id="!room123:matrix.org",
            initiator="@alice:matrix.org",
            action="deploy_prompt",
            context={"prompt_name": "customer_support_qa"},
            risk_level=ActionRisk.HIGH
        )

        print(f"✅ Created request: {request.request_id}")
        print(f"   Status: {request.status.value}")
        print(f"   Required approvals: {request.required_approvals}")

        # Simulate approvals
        request.add_approval("@bob:matrix.org")
        print(f"✅ Bob approved: {request.status.value}")

        request.add_approval("@charlie:matrix.org")
        print(f"✅ Charlie approved: {request.status.value}")

        assert request.status == ApprovalStatus.APPROVED

        # Test rejection
        request2 = await manager.request_approval(
            room_id="!room123:matrix.org",
            initiator="@alice:matrix.org",
            action="delete_prompt",
            context={"prompt_name": "old_prompt"},
            risk_level=ActionRisk.MEDIUM
        )

        request2.add_rejection("@bob:matrix.org")
        assert request2.status == ApprovalStatus.REJECTED
        print(f"✅ Rejection test passed")

        # Test expiry
        request3 = await manager.request_approval(
            room_id="!room123:matrix.org",
            initiator="@alice:matrix.org",
            action="test_action",
            context={},
            risk_level=ActionRisk.HIGH
        )

        # Manually expire
        request3.expires = datetime.now() - timedelta(hours=1)
        assert request3.is_expired()
        print(f"✅ Expiry test passed")

        print("\n✅ All approval workflow tests passed!")

    asyncio.run(test_approval())
