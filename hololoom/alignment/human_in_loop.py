"""
Human-in-the-Loop System for HoloLoom
=====================================
Real-time feedback, override channels, and intervention logging.

Philosophy:
- Human oversight for critical decisions
- Real-time feedback integration
- Override capability for incorrect decisions
- Complete intervention logging
- Learning from human corrections

Components:
1. Feedback Collector: Gathers user feedback on responses
2. Override System: Allows humans to override system decisions
3. Intervention Logger: Records all human interventions
4. Learning Integration: Updates system based on feedback

Feedback Types:
- THUMBS_UP/THUMBS_DOWN: Simple satisfaction rating
- CORRECTION: User provides corrected response
- EXPLANATION_REQUEST: User asks for clarification
- SAFETY_CONCERN: User flags unsafe/inappropriate response
- FEATURE_REQUEST: User suggests improvement
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any

logger = logging.getLogger("hololoom.alignment.human_in_loop")


class FeedbackType(Enum):
    """Type of user feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"
    EXPLANATION_REQUEST = "explanation_request"
    SAFETY_CONCERN = "safety_concern"
    FEATURE_REQUEST = "feature_request"
    QUALITY_RATING = "quality_rating"


class InterventionType(Enum):
    """Type of human intervention."""
    OVERRIDE_DECISION = "override_decision"
    PROVIDE_CORRECTION = "provide_correction"
    BLOCK_ACTION = "block_action"
    APPROVE_ACTION = "approve_action"
    MODIFY_RESPONSE = "modify_response"


@dataclass
class Feedback:
    """User feedback on a response."""
    query_id: str
    feedback_type: FeedbackType
    feedback_data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query_id": self.query_id,
            "feedback_type": self.feedback_type.value,
            "feedback_data": self.feedback_data,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
        }


@dataclass
class Intervention:
    """Human intervention record."""
    query_id: str
    intervention_type: InterventionType
    original_decision: dict[str, Any]
    override_decision: dict[str, Any]
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    operator_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query_id": self.query_id,
            "intervention_type": self.intervention_type.value,
            "original_decision": self.original_decision,
            "override_decision": self.override_decision,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "operator_id": self.operator_id,
        }


class FeedbackCollector:
    """
    Collects and processes user feedback.

    Features:
    - Multiple feedback types
    - Structured feedback data
    - Feedback aggregation
    - Learning signal extraction
    """

    def __init__(self):
        """Initialize feedback collector."""
        self.feedback_history: list[Feedback] = []
        self.query_feedback: dict[str, list[Feedback]] = {}
        self._lock = Lock()  # Thread-safe access to shared state

        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def collect_feedback(
        self,
        query_id: str,
        feedback_type: FeedbackType,
        feedback_data: dict[str, Any] | None = None,
        user_id: str | None = None,
    ):
        """
        Collect user feedback.

        Args:
            query_id: Query identifier
            feedback_type: Type of feedback
            feedback_data: Additional feedback data
            user_id: User providing feedback
        """
        feedback = Feedback(
            query_id=query_id,
            feedback_type=feedback_type,
            feedback_data=feedback_data or {},
            user_id=user_id,
        )

        with self._lock:
            self.feedback_history.append(feedback)

            # Index by query
            if query_id not in self.query_feedback:
                self.query_feedback[query_id] = []
            self.query_feedback[query_id].append(feedback)

        logger.info(f"Feedback collected: {query_id} - {feedback_type.value}")

    def get_feedback_for_query(self, query_id: str) -> list[Feedback]:
        """Get all feedback for a specific query."""
        return self.query_feedback.get(query_id, [])

    def get_recent_feedback(self, limit: int = 100) -> list[Feedback]:
        """Get recent feedback."""
        return self.feedback_history[-limit:]

    def get_satisfaction_rate(self, window: int = 100) -> float:
        """
        Get satisfaction rate (thumbs up / total ratings).

        Args:
            window: Number of recent feedback items to consider

        Returns:
            Satisfaction rate (0.0 to 1.0)
        """
        recent = self.get_recent_feedback(limit=window)

        thumbs = [f for f in recent if f.feedback_type in [FeedbackType.THUMBS_UP, FeedbackType.THUMBS_DOWN]]

        if not thumbs:
            return 0.5  # Neutral if no ratings

        thumbs_up = sum(1 for f in thumbs if f.feedback_type == FeedbackType.THUMBS_UP)
        return thumbs_up / len(thumbs)

    def get_statistics(self) -> dict[str, Any]:
        """Get feedback statistics."""
        total = len(self.feedback_history)

        by_type = {}
        for feedback_type in FeedbackType:
            count = sum(1 for f in self.feedback_history if f.feedback_type == feedback_type)
            by_type[feedback_type.value] = count

        return {
            "total_feedback": total,
            "by_type": by_type,
            "satisfaction_rate": self.get_satisfaction_rate(),
            "unique_queries": len(self.query_feedback),
        }


class OverrideSystem:
    """
    Allows humans to override system decisions.

    Features:
    - Real-time override capability
    - Intervention logging
    - Learning from overrides
    - Approval workflows
    """

    def __init__(self):
        """Initialize override system."""
        self.interventions: list[Intervention] = []
        self.pending_approvals: dict[str, dict[str, Any]] = {}
        self._lock = Lock()  # Thread-safe access to shared state

        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def request_approval(
        self,
        query_id: str,
        decision: dict[str, Any],
        reason: str,
    ) -> str:
        """
        Request human approval for a decision.

        Args:
            query_id: Query identifier
            decision: Decision requiring approval
            reason: Why approval is needed

        Returns:
            approval_id: Unique approval request ID
        """
        with self._lock:
            approval_id = f"approval_{query_id}_{len(self.pending_approvals)}"

            self.pending_approvals[approval_id] = {
                "query_id": query_id,
                "decision": decision,
                "reason": reason,
                "requested_at": datetime.now().isoformat(),
            }

        logger.warning(f"Approval requested: {approval_id} - {reason}")

        return approval_id

    def approve(
        self,
        approval_id: str,
        operator_id: str | None = None,
        modified_decision: dict[str, Any] | None = None,
    ):
        """
        Approve (and optionally modify) a pending decision.

        Args:
            approval_id: Approval request ID
            operator_id: ID of approving operator
            modified_decision: Modified decision (if changed)
        """
        with self._lock:
            if approval_id not in self.pending_approvals:
                raise ValueError(f"Approval ID not found: {approval_id}")

            request = self.pending_approvals.pop(approval_id)

            # Record intervention
            intervention = Intervention(
                query_id=request["query_id"],
                intervention_type=InterventionType.APPROVE_ACTION,
                original_decision=request["decision"],
                override_decision=modified_decision or request["decision"],
                reason=f"Approved: {request['reason']}",
                operator_id=operator_id,
            )

            self.interventions.append(intervention)

        logger.info(f"Approval granted: {approval_id}")

    def override_decision(
        self,
        query_id: str,
        original_decision: dict[str, Any],
        override_decision: dict[str, Any],
        reason: str,
        operator_id: str | None = None,
    ):
        """
        Override a system decision.

        Args:
            query_id: Query identifier
            original_decision: Original system decision
            override_decision: Human override decision
            reason: Why override was needed
            operator_id: ID of operator performing override
        """
        intervention = Intervention(
            query_id=query_id,
            intervention_type=InterventionType.OVERRIDE_DECISION,
            original_decision=original_decision,
            override_decision=override_decision,
            reason=reason,
            operator_id=operator_id,
        )

        with self._lock:
            self.interventions.append(intervention)

        logger.warning(f"Decision overridden: {query_id} - {reason}")

    def get_pending_approvals(self) -> list[dict[str, Any]]:
        """Get all pending approval requests."""
        return list(self.pending_approvals.values())

    def get_interventions_for_query(self, query_id: str) -> list[Intervention]:
        """Get all interventions for a query."""
        return [i for i in self.interventions if i.query_id == query_id]

    def get_recent_interventions(self, limit: int = 100) -> list[Intervention]:
        """Get recent interventions."""
        return self.interventions[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        """Get intervention statistics."""
        total = len(self.interventions)

        by_type = {}
        for intervention_type in InterventionType:
            count = sum(1 for i in self.interventions if i.intervention_type == intervention_type)
            by_type[intervention_type.value] = count

        return {
            "total_interventions": total,
            "by_type": by_type,
            "pending_approvals": len(self.pending_approvals),
        }


class HumanInLoopSystem:
    """
    Complete human-in-the-loop system.

    Integrates:
    - Feedback collection
    - Override capability
    - Intervention logging
    - Learning from human input

    Usage:
        hitl = HumanInLoopSystem()

        # Collect feedback
        hitl.collect_feedback(
            query_id="q123",
            feedback_type=FeedbackType.THUMBS_UP,
        )

        # Request approval
        approval_id = hitl.request_approval(
            query_id="q123",
            decision={"tool": "delete", "target": "all_data"},
            reason="High-risk operation",
        )

        # Approve with modification
        hitl.approve(
            approval_id=approval_id,
            operator_id="operator_001",
            modified_decision={"tool": "delete", "target": "specific_data"},
        )

        # Override incorrect decision
        hitl.override_decision(
            query_id="q124",
            original_decision={"response": "incorrect answer"},
            override_decision={"response": "corrected answer"},
            reason="Factual error",
        )
    """

    def __init__(self):
        """Initialize human-in-the-loop system."""
        self.feedback_collector = FeedbackCollector()
        self.override_system = OverrideSystem()

        logger.info("Human-in-the-loop system initialized")

    # Feedback methods
    def collect_feedback(
        self,
        query_id: str,
        feedback_type: FeedbackType,
        feedback_data: dict[str, Any] | None = None,
        user_id: str | None = None,
    ):
        """Collect user feedback."""
        self.feedback_collector.collect_feedback(
            query_id, feedback_type, feedback_data, user_id
        )

    def get_satisfaction_rate(self) -> float:
        """Get overall satisfaction rate."""
        return self.feedback_collector.get_satisfaction_rate()

    # Override/approval methods
    def request_approval(
        self,
        query_id: str,
        decision: dict[str, Any],
        reason: str,
    ) -> str:
        """Request human approval."""
        return self.override_system.request_approval(query_id, decision, reason)

    def approve(
        self,
        approval_id: str,
        operator_id: str | None = None,
        modified_decision: dict[str, Any] | None = None,
    ):
        """Approve pending decision."""
        self.override_system.approve(approval_id, operator_id, modified_decision)

    def override_decision(
        self,
        query_id: str,
        original_decision: dict[str, Any],
        override_decision: dict[str, Any],
        reason: str,
        operator_id: str | None = None,
    ):
        """Override system decision."""
        self.override_system.override_decision(
            query_id, original_decision, override_decision, reason, operator_id
        )

    # Query methods
    def get_pending_approvals(self) -> list[dict[str, Any]]:
        """Get pending approval requests."""
        return self.override_system.get_pending_approvals()

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "feedback": self.feedback_collector.get_statistics(),
            "interventions": self.override_system.get_statistics(),
            "overall_satisfaction": self.get_satisfaction_rate(),
        }
