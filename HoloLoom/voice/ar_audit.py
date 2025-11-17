"""
AR Audit Trail
==============
Complete provenance tracking for AR decisions and interactions.

Extends HoloLoom's alignment framework AuditTrail with AR-specific logging:
- Gesture events and commands
- AR object interactions
- Visual overlays and modifications
- Spatial context and user position
- Complete decision history with temporal queries

Features:
- AR-specific decision logging
- Gesture and spatial context capture
- Temporal queries (time-based filtering)
- Export for compliance and debugging
- Integration with base AuditTrail

Author: HoloLoom Team
Date: November 17, 2025
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import json

from HoloLoom.alignment.audit_trail import (
    AuditTrail,
    DecisionLog,
    DecisionType,
    OutcomeType,
    ProvenanceTracer,
)
from HoloLoom.voice.ar_context import ARContext, ARObject, ARObjectType, GestureType
from HoloLoom.voice.ar_safety_gate import ARActionCategory

logger = logging.getLogger("HoloLoom.voice.ar_audit")


# ============================================================================
# AR Decision Types
# ============================================================================

class ARDecisionType(Enum):
    """Types of AR decisions that can be logged."""

    GESTURE_COMMAND = "gesture_command"          # Gesture-triggered command
    VOICE_COMMAND = "voice_command"              # Voice-triggered command
    OBJECT_INTERACTION = "object_interaction"    # AR object interaction
    VISUAL_OVERLAY = "visual_overlay"            # Visual overlay change
    SPATIAL_MODIFICATION = "spatial_modification"  # Spatial state change
    SAFETY_GATE = "safety_gate"                  # Safety gate decision
    DECEPTION_CHECK = "deception_check"          # Deception detection
    CONTEXT_RETRIEVAL = "context_retrieval"      # KG context retrieval


# ============================================================================
# AR Decision Log
# ============================================================================

@dataclass
class ARDecisionLog(DecisionLog):
    """
    Extended decision log with AR-specific context.

    Captures all AR-specific information needed for complete provenance:
    - Gesture type and confidence
    - AR object details (ID, type, position)
    - User position and gaze
    - Visual context (scene, visible objects)
    """

    ar_decision_type: Optional[ARDecisionType] = None
    gesture_type: Optional[str] = None
    gesture_confidence: float = 0.0

    # AR object details
    target_object_id: Optional[str] = None
    target_object_type: Optional[str] = None
    target_object_position: Optional[List[float]] = None

    # User context
    user_position: Optional[List[float]] = None
    user_gaze_target: Optional[str] = None
    active_scene: Optional[str] = None
    visible_objects_count: int = 0

    # AR action details
    ar_action_category: Optional[str] = None
    overlay_type: Optional[str] = None
    spatial_change_description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            "ar_decision_type": self.ar_decision_type.value if self.ar_decision_type else None,
            "gesture_type": self.gesture_type,
            "gesture_confidence": self.gesture_confidence,
            "target_object_id": self.target_object_id,
            "target_object_type": self.target_object_type,
            "target_object_position": self.target_object_position,
            "user_position": self.user_position,
            "user_gaze_target": self.user_gaze_target,
            "active_scene": self.active_scene,
            "visible_objects_count": self.visible_objects_count,
            "ar_action_category": self.ar_action_category,
            "overlay_type": self.overlay_type,
            "spatial_change_description": self.spatial_change_description,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ARDecisionLog":
        """Create from dictionary."""
        # Convert base fields
        base_log = DecisionLog.from_dict(data)

        # Add AR-specific fields
        return cls(
            decision_id=base_log.decision_id,
            decision_type=base_log.decision_type,
            outcome=base_log.outcome,
            reason=base_log.reason,
            timestamp=base_log.timestamp,
            query_text=base_log.query_text,
            action_description=base_log.action_description,
            risk_level=base_log.risk_level,
            confidence=base_log.confidence,
            metadata=base_log.metadata,
            reasoning_chain=base_log.reasoning_chain,
            data_sources=base_log.data_sources,
            ar_decision_type=ARDecisionType(data["ar_decision_type"]) if data.get("ar_decision_type") else None,
            gesture_type=data.get("gesture_type"),
            gesture_confidence=data.get("gesture_confidence", 0.0),
            target_object_id=data.get("target_object_id"),
            target_object_type=data.get("target_object_type"),
            target_object_position=data.get("target_object_position"),
            user_position=data.get("user_position"),
            user_gaze_target=data.get("user_gaze_target"),
            active_scene=data.get("active_scene"),
            visible_objects_count=data.get("visible_objects_count", 0),
            ar_action_category=data.get("ar_action_category"),
            overlay_type=data.get("overlay_type"),
            spatial_change_description=data.get("spatial_change_description"),
        )


# ============================================================================
# AR Audit Trail
# ============================================================================

class ARAuditTrail:
    """
    Main AR audit trail system.

    Extends base AuditTrail with AR-specific logging:
    - Gesture events
    - AR object interactions
    - Spatial context
    - Visual overlays

    Usage:
        audit = ARAuditTrail(persist_path="./ar_audit_logs")

        # Log gesture command
        log = audit.log_gesture_command(
            gesture_type="tap",
            gesture_confidence=0.95,
            action="highlight_object",
            target_object_id="hive_001",
            ar_context=ar_context,
            outcome=OutcomeType.APPROVED,
            reason="Safe display action"
        )

        # Log voice command with spatial context
        log = audit.log_voice_command(
            query="Show me the health of this hive",
            action="display_health_info",
            ar_context=ar_context,
            outcome=OutcomeType.APPROVED
        )

        # Query history
        recent_gestures = audit.query_by_gesture_type("tap", limit=10)
        failed_commands = audit.query_by_outcome(OutcomeType.REJECTED)
    """

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        auto_flush: bool = True,
        flush_interval: int = 10,
    ):
        """
        Initialize AR audit trail.

        Args:
            persist_path: Directory for persistent storage (None = memory only)
            auto_flush: Automatically flush to disk after each log
            flush_interval: If auto_flush=False, flush every N logs
        """
        # Create base audit trail
        self.base_audit = AuditTrail(
            persist_path=persist_path,
            auto_flush=auto_flush,
            flush_interval=flush_interval,
        )

        # AR-specific logs
        self.ar_logs: List[ARDecisionLog] = []

        self.persist_path = Path(persist_path) if persist_path else None

        if self.persist_path:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._load_ar_logs()

        logger.info(
            f"ARAuditTrail initialized (persist_path={persist_path}, auto_flush={auto_flush})"
        )

    def log_gesture_command(
        self,
        gesture_type: str,
        gesture_confidence: float,
        action: str,
        ar_context: ARContext,
        outcome: OutcomeType,
        reason: str,
        target_object_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ARDecisionLog:
        """
        Log a gesture-triggered command.

        Args:
            gesture_type: Type of gesture (e.g., "tap", "swipe_left")
            gesture_confidence: Gesture recognition confidence
            action: Action performed
            ar_context: Current AR context
            outcome: Decision outcome
            reason: Reason for outcome
            target_object_id: Target AR object (if any)
            risk_level: Risk level assessment
            metadata: Additional metadata

        Returns:
            Created ARDecisionLog
        """
        # Find target object
        target_object = None
        if target_object_id:
            target_object = ar_context.find_object_by_id(target_object_id)

        # Create AR decision log
        decision_id = f"gesture_{datetime.now().timestamp()}"

        log = ARDecisionLog(
            decision_id=decision_id,
            decision_type=DecisionType.TOOL_SELECTION,  # Map to base type
            outcome=outcome,
            reason=reason,
            query_text=f"Gesture: {gesture_type}",
            action_description=action,
            risk_level=risk_level,
            confidence=gesture_confidence,
            metadata=metadata or {},
            ar_decision_type=ARDecisionType.GESTURE_COMMAND,
            gesture_type=gesture_type,
            gesture_confidence=gesture_confidence,
            target_object_id=target_object_id,
            target_object_type=target_object.type.value if target_object else None,
            target_object_position=[
                target_object.position.x,
                target_object.position.y,
                target_object.position.z
            ] if target_object else None,
            user_position=[
                ar_context.user_position.x,
                ar_context.user_position.y,
                ar_context.user_position.z
            ],
            user_gaze_target=ar_context.gaze_target,
            active_scene=ar_context.active_scene,
            visible_objects_count=len(ar_context.visible_objects),
        )

        self.ar_logs.append(log)

        logger.info(
            f"Logged gesture command: {gesture_type} → {action} ({outcome.value})"
        )

        # Flush to disk if enabled
        if self.base_audit.auto_flush and self.persist_path:
            self._flush_ar_log(log)

        return log

    def log_voice_command(
        self,
        query: str,
        action: str,
        ar_context: ARContext,
        outcome: OutcomeType,
        reason: str = "",
        confidence: float = 0.0,
        risk_level: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ARDecisionLog:
        """
        Log a voice-triggered command.

        Args:
            query: Voice query text
            action: Action performed
            ar_context: Current AR context
            outcome: Decision outcome
            reason: Reason for outcome
            confidence: Voice recognition confidence
            risk_level: Risk level assessment
            metadata: Additional metadata

        Returns:
            Created ARDecisionLog
        """
        decision_id = f"voice_{datetime.now().timestamp()}"

        log = ARDecisionLog(
            decision_id=decision_id,
            decision_type=DecisionType.TOOL_SELECTION,
            outcome=outcome,
            reason=reason,
            query_text=query,
            action_description=action,
            risk_level=risk_level,
            confidence=confidence,
            metadata=metadata or {},
            ar_decision_type=ARDecisionType.VOICE_COMMAND,
            user_position=[
                ar_context.user_position.x,
                ar_context.user_position.y,
                ar_context.user_position.z
            ],
            user_gaze_target=ar_context.gaze_target,
            active_scene=ar_context.active_scene,
            visible_objects_count=len(ar_context.visible_objects),
        )

        self.ar_logs.append(log)

        logger.info(f"Logged voice command: {query[:30]}... → {action} ({outcome.value})")

        if self.base_audit.auto_flush and self.persist_path:
            self._flush_ar_log(log)

        return log

    def log_ar_decision(
        self,
        ar_decision_type: ARDecisionType,
        outcome: OutcomeType,
        reason: str,
        ar_context: ARContext,
        query_text: Optional[str] = None,
        action_description: Optional[str] = None,
        risk_level: Optional[str] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ARDecisionLog:
        """
        Log a general AR decision.

        Args:
            ar_decision_type: Type of AR decision
            outcome: Decision outcome
            reason: Reason for outcome
            ar_context: Current AR context
            query_text: Query text (if any)
            action_description: Action description
            risk_level: Risk level assessment
            confidence: Decision confidence
            metadata: Additional metadata

        Returns:
            Created ARDecisionLog
        """
        decision_id = f"{ar_decision_type.value}_{datetime.now().timestamp()}"

        log = ARDecisionLog(
            decision_id=decision_id,
            decision_type=DecisionType.TOOL_SELECTION,
            outcome=outcome,
            reason=reason,
            query_text=query_text,
            action_description=action_description,
            risk_level=risk_level,
            confidence=confidence,
            metadata=metadata or {},
            ar_decision_type=ar_decision_type,
            user_position=[
                ar_context.user_position.x,
                ar_context.user_position.y,
                ar_context.user_position.z
            ],
            user_gaze_target=ar_context.gaze_target,
            active_scene=ar_context.active_scene,
            visible_objects_count=len(ar_context.visible_objects),
        )

        self.ar_logs.append(log)

        logger.info(
            f"Logged AR decision: {ar_decision_type.value} → {outcome.value}"
        )

        if self.base_audit.auto_flush and self.persist_path:
            self._flush_ar_log(log)

        return log

    # Query methods
    def query_by_ar_decision_type(
        self, ar_decision_type: ARDecisionType, limit: Optional[int] = None
    ) -> List[ARDecisionLog]:
        """Query logs by AR decision type."""
        results = [
            log for log in self.ar_logs
            if log.ar_decision_type == ar_decision_type
        ]
        return results[:limit] if limit else results

    def query_by_gesture_type(
        self, gesture_type: str, limit: Optional[int] = None
    ) -> List[ARDecisionLog]:
        """Query logs by gesture type."""
        results = [
            log for log in self.ar_logs
            if log.gesture_type == gesture_type
        ]
        return results[:limit] if limit else results

    def query_by_target_object(
        self, object_id: str, limit: Optional[int] = None
    ) -> List[ARDecisionLog]:
        """Query logs by target object ID."""
        results = [
            log for log in self.ar_logs
            if log.target_object_id == object_id
        ]
        return results[:limit] if limit else results

    def query_by_scene(
        self, scene_name: str, limit: Optional[int] = None
    ) -> List[ARDecisionLog]:
        """Query logs by AR scene."""
        results = [
            log for log in self.ar_logs
            if log.active_scene == scene_name
        ]
        return results[:limit] if limit else results

    def query_by_outcome(
        self, outcome: OutcomeType, limit: Optional[int] = None
    ) -> List[ARDecisionLog]:
        """Query logs by outcome."""
        results = [log for log in self.ar_logs if log.outcome == outcome]
        return results[:limit] if limit else results

    def query_by_time_range(
        self, start: datetime, end: datetime, limit: Optional[int] = None
    ) -> List[ARDecisionLog]:
        """Query logs by time range."""
        results = [
            log for log in self.ar_logs
            if start <= log.timestamp <= end
        ]
        return results[:limit] if limit else results

    # Persistence methods
    def flush(self):
        """Flush all logs to disk."""
        if not self.persist_path:
            return

        for log in self.ar_logs:
            self._flush_ar_log(log)

        logger.info(f"Flushed {len(self.ar_logs)} AR logs to disk")

    def _flush_ar_log(self, log: ARDecisionLog):
        """Flush a single AR log to disk (JSON Lines format)."""
        if not self.persist_path:
            return

        log_file = self.persist_path / "ar_decisions.jsonl"

        with log_file.open("a") as f:
            f.write(json.dumps(log.to_dict()) + "\n")

    def _load_ar_logs(self):
        """Load AR logs from disk on initialization."""
        if not self.persist_path:
            return

        log_file = self.persist_path / "ar_decisions.jsonl"
        if not log_file.exists():
            return

        with log_file.open() as f:
            for line in f:
                data = json.loads(line.strip())
                log = ARDecisionLog.from_dict(data)
                self.ar_logs.append(log)

        logger.info(f"Loaded {len(self.ar_logs)} AR logs from disk")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get AR audit statistics.

        Returns:
            Dictionary of statistics including AR-specific metrics
        """
        total_logs = len(self.ar_logs)
        approved = sum(1 for log in self.ar_logs if log.outcome == OutcomeType.APPROVED)
        rejected = sum(1 for log in self.ar_logs if log.outcome == OutcomeType.REJECTED)

        by_ar_type = {}
        for ar_type in ARDecisionType:
            type_logs = [log for log in self.ar_logs if log.ar_decision_type == ar_type]
            by_ar_type[ar_type.value] = {
                "total": len(type_logs),
                "approved": sum(1 for log in type_logs if log.outcome == OutcomeType.APPROVED),
                "rejected": sum(1 for log in type_logs if log.outcome == OutcomeType.REJECTED),
            }

        by_gesture = {}
        for log in self.ar_logs:
            if log.gesture_type:
                if log.gesture_type not in by_gesture:
                    by_gesture[log.gesture_type] = {"total": 0, "approved": 0, "rejected": 0}
                by_gesture[log.gesture_type]["total"] += 1
                if log.outcome == OutcomeType.APPROVED:
                    by_gesture[log.gesture_type]["approved"] += 1
                elif log.outcome == OutcomeType.REJECTED:
                    by_gesture[log.gesture_type]["rejected"] += 1

        by_scene = {}
        for log in self.ar_logs:
            if log.active_scene:
                if log.active_scene not in by_scene:
                    by_scene[log.active_scene] = {"total": 0, "approved": 0, "rejected": 0}
                by_scene[log.active_scene]["total"] += 1
                if log.outcome == OutcomeType.APPROVED:
                    by_scene[log.active_scene]["approved"] += 1
                elif log.outcome == OutcomeType.REJECTED:
                    by_scene[log.active_scene]["rejected"] += 1

        return {
            "total_ar_logs": total_logs,
            "approved": approved,
            "rejected": rejected,
            "by_ar_decision_type": by_ar_type,
            "by_gesture_type": by_gesture,
            "by_scene": by_scene,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_ar_audit_trail(
    persist_path: Optional[Path] = None,
    auto_flush: bool = True,
) -> ARAuditTrail:
    """
    Create AR audit trail instance.

    Args:
        persist_path: Directory for persistent storage (None = memory only)
        auto_flush: Automatically flush to disk after each log

    Returns:
        Configured ARAuditTrail instance
    """
    return ARAuditTrail(persist_path=persist_path, auto_flush=auto_flush)
