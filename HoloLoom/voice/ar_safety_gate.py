"""
AR Safety Gate
===============
Safety guardrails for augmented reality actions.

Wraps HoloLoom's alignment framework SafetyGuardrails for AR-specific context,
providing risk-based gating for AR actions like gesture commands, visual overlays,
and spatial modifications.

Features:
- AR-specific risk assessment (gesture, visual, spatial, system)
- 4 risk levels: LOW, MEDIUM, HIGH, CRITICAL
- Human-in-the-loop escalation for HIGH/CRITICAL
- Adversarial gesture pattern detection
- Contextual safety based on AR environment

Integration:
- Uses SafetyGuardrails from HoloLoom.alignment
- Integrates with ARContext from HoloLoom.voice.ar_context
- Logs to AuditTrail for complete provenance

Author: HoloLoom Team
Date: November 17, 2025
"""

import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from HoloLoom.alignment.safety_guardrails import (
    SafetyGuardrails,
    ActionRequest,
    ActionCategory,
    SafetyDecision,
    RiskLevel,
)
from HoloLoom.voice.ar_context import ARContext, ARObject, ARObjectType, GestureType

logger = logging.getLogger("HoloLoom.voice.ar_safety_gate")


# ============================================================================
# AR Action Categories
# ============================================================================

class ARActionCategory(Enum):
    """Categories of AR actions with associated risk levels."""

    # Display actions (LOW risk)
    DISPLAY_INFO = "display_info"              # Show information overlay
    HIGHLIGHT_OBJECT = "highlight_object"      # Highlight an AR object
    SHOW_LABEL = "show_label"                  # Display label/annotation

    # Visual overlay actions (MEDIUM risk)
    ADD_OVERLAY = "add_overlay"                # Add visual overlay to scene
    MODIFY_OVERLAY = "modify_overlay"          # Modify existing overlay
    REMOVE_OVERLAY = "remove_overlay"          # Remove overlay
    CHANGE_VISUALIZATION = "change_visualization"  # Change visualization mode

    # Spatial modification actions (HIGH risk)
    MODIFY_OBJECT_STATE = "modify_object_state"  # Change AR object state
    MOVE_OBJECT = "move_object"                # Move AR object in space
    CREATE_OBJECT = "create_object"            # Create new AR object
    DELETE_OBJECT = "delete_object"            # Delete AR object

    # System actions (CRITICAL risk)
    EXECUTE_COMMAND = "execute_command"        # Execute system command
    MODIFY_SYSTEM_STATE = "modify_system_state"  # Change system configuration
    ACCESS_EXTERNAL_API = "access_external_api"  # Call external API
    MODIFY_PERMISSIONS = "modify_permissions"  # Change user permissions


# ============================================================================
# AR Safety Decision
# ============================================================================

@dataclass
class ARSafetyDecision(SafetyDecision):
    """
    Extended safety decision with AR-specific context.

    Adds AR-specific metadata like gesture type, target object, and
    spatial context to the base SafetyDecision.
    """

    ar_action_category: Optional[ARActionCategory] = None
    gesture_type: Optional[str] = None
    target_object_id: Optional[str] = None
    target_object_type: Optional[str] = None
    spatial_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "ar_action_category": self.ar_action_category.value if self.ar_action_category else None,
            "gesture_type": self.gesture_type,
            "target_object_id": self.target_object_id,
            "target_object_type": self.target_object_type,
            "spatial_context": self.spatial_context,
        })
        return base_dict


# ============================================================================
# Adversarial Gesture Detector
# ============================================================================

class AdversarialGestureDetector:
    """
    Detects potentially malicious or unsafe gesture patterns.

    Identifies:
    - Rapid gesture sequences (potential DOS)
    - Contradictory gestures (confusion attacks)
    - Unsafe spatial targeting (targeting critical objects)
    - Gesture spoofing patterns
    """

    # Patterns indicating potential adversarial gestures
    UNSAFE_GESTURE_SEQUENCES = [
        # Rapid repeated gestures (potential DOS)
        ("rapid_sequence", ["tap", "tap", "tap", "tap", "tap"]),
        ("rapid_swipes", ["swipe_left", "swipe_right", "swipe_left", "swipe_right"]),

        # Contradictory gestures (confusion)
        ("contradictory", ["pinch", "spread", "pinch", "spread"]),

        # Dangerous targeting patterns
        ("system_targeting", ["point", "circle", "tap"]),  # Point at system object, circle, tap
    ]

    # Critical objects that should not be targeted
    CRITICAL_OBJECT_TYPES = {
        ARObjectType.MARKER,  # System markers
        ARObjectType.ANNOTATION,  # System annotations
    }

    def __init__(self, gesture_history_size: int = 10):
        """
        Initialize adversarial gesture detector.

        Args:
            gesture_history_size: Number of recent gestures to track
        """
        self.gesture_history: List[str] = []
        self.gesture_history_size = gesture_history_size
        self.gesture_timestamps: List[float] = []

    def record_gesture(self, gesture_type: str, timestamp: Optional[float] = None):
        """
        Record a gesture for pattern analysis.

        Args:
            gesture_type: Type of gesture (e.g., "tap", "swipe_left")
            timestamp: Timestamp of gesture (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        self.gesture_history.append(gesture_type)
        self.gesture_timestamps.append(timestamp)

        # Trim to history size
        if len(self.gesture_history) > self.gesture_history_size:
            self.gesture_history.pop(0)
            self.gesture_timestamps.pop(0)

    def detect(
        self,
        gesture_type: str,
        target_object: Optional[ARObject] = None,
        ar_context: Optional[ARContext] = None
    ) -> tuple[bool, str]:
        """
        Detect adversarial patterns in gesture.

        Args:
            gesture_type: Current gesture type
            target_object: Target AR object (if any)
            ar_context: Current AR context

        Returns:
            Tuple of (is_adversarial, reason)
        """
        # Record this gesture
        self.record_gesture(gesture_type)

        # Check for rapid gesture sequences (DOS attack)
        if len(self.gesture_timestamps) >= 5:
            recent_window = self.gesture_timestamps[-5:]
            time_span = recent_window[-1] - recent_window[0]

            if time_span < 1.0:  # 5 gestures in 1 second
                return True, "Rapid gesture sequence detected (potential DOS attack)"

        # Check for unsafe sequence patterns
        recent_sequence = self.gesture_history[-5:] if len(self.gesture_history) >= 5 else self.gesture_history

        for pattern_name, pattern in self.UNSAFE_GESTURE_SEQUENCES:
            if len(recent_sequence) >= len(pattern):
                if recent_sequence[-len(pattern):] == pattern:
                    return True, f"Unsafe gesture pattern detected: {pattern_name}"

        # Check for targeting critical objects
        if target_object and target_object.type in self.CRITICAL_OBJECT_TYPES:
            return True, f"Attempt to target critical system object: {target_object.type.value}"

        # Check for out-of-bounds spatial targeting
        if ar_context and target_object:
            distance = target_object.distance_from(ar_context.user_position)
            if distance > 50.0:  # 50 meters (unrealistic reach)
                return True, f"Attempt to target object outside safe range: {distance:.1f}m"

        return False, ""

    def clear_history(self):
        """Clear gesture history (useful for testing)."""
        self.gesture_history.clear()
        self.gesture_timestamps.clear()


# ============================================================================
# AR Safety Gate
# ============================================================================

class ARSafetyGate:
    """
    Main safety gate for AR actions.

    Wraps SafetyGuardrails with AR-specific logic:
    - Maps AR actions to risk levels
    - Detects adversarial gesture patterns
    - Validates spatial context
    - Provides human-in-the-loop escalation for high-risk actions

    Usage:
        gate = ARSafetyGate()

        decision = await gate.gate_action(
            action="add_overlay",
            ar_context=ar_context,
            gesture_type="tap",
            target_object_id="hive_001"
        )

        if decision.allowed:
            # Proceed with action
            await perform_action()
        elif decision.requires_approval:
            # Escalate for human approval
            approved = await request_approval(decision)
            if approved:
                await perform_action()
        else:
            # Action blocked
            logger.warning(f"Action blocked: {decision.reason}")
    """

    # Default risk levels for AR actions
    AR_ACTION_RISK_LEVELS = {
        # Display actions (LOW risk)
        ARActionCategory.DISPLAY_INFO: RiskLevel.LOW,
        ARActionCategory.HIGHLIGHT_OBJECT: RiskLevel.LOW,
        ARActionCategory.SHOW_LABEL: RiskLevel.LOW,

        # Visual overlay actions (MEDIUM risk)
        ARActionCategory.ADD_OVERLAY: RiskLevel.MEDIUM,
        ARActionCategory.MODIFY_OVERLAY: RiskLevel.MEDIUM,
        ARActionCategory.REMOVE_OVERLAY: RiskLevel.MEDIUM,
        ARActionCategory.CHANGE_VISUALIZATION: RiskLevel.MEDIUM,

        # Spatial modification actions (HIGH risk)
        ARActionCategory.MODIFY_OBJECT_STATE: RiskLevel.HIGH,
        ARActionCategory.MOVE_OBJECT: RiskLevel.HIGH,
        ARActionCategory.CREATE_OBJECT: RiskLevel.HIGH,
        ARActionCategory.DELETE_OBJECT: RiskLevel.HIGH,

        # System actions (CRITICAL risk)
        ARActionCategory.EXECUTE_COMMAND: RiskLevel.CRITICAL,
        ARActionCategory.MODIFY_SYSTEM_STATE: RiskLevel.CRITICAL,
        ARActionCategory.ACCESS_EXTERNAL_API: RiskLevel.CRITICAL,
        ARActionCategory.MODIFY_PERMISSIONS: RiskLevel.CRITICAL,
    }

    def __init__(
        self,
        safety_guardrails: Optional[SafetyGuardrails] = None,
        enable_adversarial_detection: bool = True,
        testing_mode: bool = False,
    ):
        """
        Initialize AR safety gate.

        Args:
            safety_guardrails: Base safety guardrails (creates default if None)
            enable_adversarial_detection: Whether to detect adversarial gestures
            testing_mode: If True, bypass approval requirements (for development)
        """
        self.guardrails = safety_guardrails or SafetyGuardrails(testing_mode=testing_mode)
        self.adversarial_detector = (
            AdversarialGestureDetector() if enable_adversarial_detection else None
        )
        self.testing_mode = testing_mode

        # Track decisions for statistics
        self.ar_decisions: List[ARSafetyDecision] = []

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

    async def gate_action(
        self,
        action: str,
        ar_context: ARContext,
        gesture_type: Optional[str] = None,
        target_object_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ARSafetyDecision:
        """
        Gate an AR action through safety checks.

        Args:
            action: Action to perform (e.g., "add_overlay", "move_object")
            ar_context: Current AR context
            gesture_type: Gesture that triggered action (e.g., "tap", "swipe_left")
            target_object_id: ID of target AR object (if any)
            metadata: Additional action metadata
            user_id: User performing action
            session_id: Current session ID

        Returns:
            ARSafetyDecision indicating whether action is allowed
        """
        # Determine AR action category
        ar_action_category = self._categorize_action(action)

        # Get base risk level
        base_risk_level = self.AR_ACTION_RISK_LEVELS.get(
            ar_action_category, RiskLevel.MEDIUM
        )

        # Find target object
        target_object = None
        if target_object_id:
            target_object = ar_context.find_object_by_id(target_object_id)

        # Check for adversarial gestures
        if self.adversarial_detector and gesture_type:
            is_adversarial, reason = self.adversarial_detector.detect(
                gesture_type, target_object, ar_context
            )

            if is_adversarial:
                logger.warning(f"Adversarial gesture detected: {reason}")
                return ARSafetyDecision(
                    allowed=False,
                    risk_level=RiskLevel.CRITICAL,
                    reason=f"Blocked: {reason}",
                    metadata={"adversarial_detected": True},
                    ar_action_category=ar_action_category,
                    gesture_type=gesture_type,
                    target_object_id=target_object_id,
                    target_object_type=target_object.type.value if target_object else None,
                )

        # Adjust risk based on spatial context
        adjusted_risk = self._adjust_risk_for_context(
            base_risk_level, ar_context, target_object
        )

        # Build spatial context for decision
        spatial_context = {
            "user_position": [
                ar_context.user_position.x,
                ar_context.user_position.y,
                ar_context.user_position.z
            ],
            "gaze_target": ar_context.gaze_target,
            "visible_objects_count": len(ar_context.visible_objects),
            "active_scene": ar_context.active_scene,
        }

        if target_object:
            spatial_context["target_position"] = [
                target_object.position.x,
                target_object.position.y,
                target_object.position.z
            ]
            spatial_context["target_distance"] = target_object.distance_from(
                ar_context.user_position
            )

        # Map to ActionCategory for base guardrails
        base_category = self._map_to_base_category(ar_action_category)

        # Create request for base guardrails
        request = ActionRequest(
            action=action,
            category=base_category,
            context={
                **(metadata or {}),
                "ar_action_category": ar_action_category.value,
                "gesture_type": gesture_type,
                "target_object_id": target_object_id,
                "spatial_context": spatial_context,
            },
            user_id=user_id,
            session_id=session_id,
        )

        # Evaluate with base guardrails
        base_decision = self.guardrails.evaluate(request)

        # Create AR-specific decision
        ar_decision = ARSafetyDecision(
            allowed=base_decision.allowed,
            risk_level=adjusted_risk,
            reason=base_decision.reason,
            requires_approval=base_decision.requires_approval,
            metadata=base_decision.metadata,
            timestamp=base_decision.timestamp,
            context=base_decision.context,
            alternative_action=base_decision.alternative_action,
            ar_action_category=ar_action_category,
            gesture_type=gesture_type,
            target_object_id=target_object_id,
            target_object_type=target_object.type.value if target_object else None,
            spatial_context=spatial_context,
        )

        # Log decision
        self._log_decision(ar_decision)

        # Store decision
        self.ar_decisions.append(ar_decision)

        return ar_decision

    def _categorize_action(self, action: str) -> ARActionCategory:
        """
        Categorize an action string into ARActionCategory.

        Args:
            action: Action string (e.g., "add_overlay", "display_info")

        Returns:
            ARActionCategory
        """
        action_lower = action.lower().replace(" ", "_")

        # Try exact match
        for category in ARActionCategory:
            if category.value == action_lower:
                return category

        # Try fuzzy match
        if "display" in action_lower or "show" in action_lower or "info" in action_lower:
            return ARActionCategory.DISPLAY_INFO
        elif "highlight" in action_lower:
            return ARActionCategory.HIGHLIGHT_OBJECT
        elif "label" in action_lower:
            return ARActionCategory.SHOW_LABEL
        elif "overlay" in action_lower:
            if "add" in action_lower or "create" in action_lower:
                return ARActionCategory.ADD_OVERLAY
            elif "remove" in action_lower or "delete" in action_lower:
                return ARActionCategory.REMOVE_OVERLAY
            else:
                return ARActionCategory.MODIFY_OVERLAY
        elif "move" in action_lower:
            return ARActionCategory.MOVE_OBJECT
        elif "delete" in action_lower or "remove" in action_lower:
            return ARActionCategory.DELETE_OBJECT
        elif "create" in action_lower:
            return ARActionCategory.CREATE_OBJECT
        elif "command" in action_lower or "execute" in action_lower:
            return ARActionCategory.EXECUTE_COMMAND
        elif "system" in action_lower:
            return ARActionCategory.MODIFY_SYSTEM_STATE
        elif "api" in action_lower or "external" in action_lower:
            return ARActionCategory.ACCESS_EXTERNAL_API
        elif "permission" in action_lower:
            return ARActionCategory.MODIFY_PERMISSIONS

        # Default to medium-risk overlay modification
        return ARActionCategory.MODIFY_OVERLAY

    def _map_to_base_category(self, ar_category: ARActionCategory) -> ActionCategory:
        """
        Map AR action category to base ActionCategory.

        Args:
            ar_category: AR-specific action category

        Returns:
            Base ActionCategory for guardrails
        """
        mapping = {
            # Display actions → QUERY (read-only)
            ARActionCategory.DISPLAY_INFO: ActionCategory.QUERY,
            ARActionCategory.HIGHLIGHT_OBJECT: ActionCategory.QUERY,
            ARActionCategory.SHOW_LABEL: ActionCategory.QUERY,

            # Visual overlay actions → MODIFICATION
            ARActionCategory.ADD_OVERLAY: ActionCategory.MODIFICATION,
            ARActionCategory.MODIFY_OVERLAY: ActionCategory.MODIFICATION,
            ARActionCategory.REMOVE_OVERLAY: ActionCategory.MODIFICATION,
            ARActionCategory.CHANGE_VISUALIZATION: ActionCategory.MODIFICATION,

            # Spatial modification actions → MODIFICATION/DELETION
            ARActionCategory.MODIFY_OBJECT_STATE: ActionCategory.MODIFICATION,
            ARActionCategory.MOVE_OBJECT: ActionCategory.MODIFICATION,
            ARActionCategory.CREATE_OBJECT: ActionCategory.MODIFICATION,
            ARActionCategory.DELETE_OBJECT: ActionCategory.DELETION,

            # System actions → SYSTEM/EXTERNAL
            ARActionCategory.EXECUTE_COMMAND: ActionCategory.EXECUTION,
            ARActionCategory.MODIFY_SYSTEM_STATE: ActionCategory.SYSTEM,
            ARActionCategory.ACCESS_EXTERNAL_API: ActionCategory.EXTERNAL,
            ARActionCategory.MODIFY_PERMISSIONS: ActionCategory.SYSTEM,
        }

        return mapping.get(ar_category, ActionCategory.MODIFICATION)

    def _adjust_risk_for_context(
        self,
        base_risk: RiskLevel,
        ar_context: ARContext,
        target_object: Optional[ARObject]
    ) -> RiskLevel:
        """
        Adjust risk level based on AR context.

        Increases risk for:
        - Actions on critical objects
        - Actions in sensitive scenes
        - Actions at extreme distances

        Args:
            base_risk: Base risk level
            ar_context: Current AR context
            target_object: Target object (if any)

        Returns:
            Adjusted risk level
        """
        # Start with base risk
        risk_score = self._risk_to_score(base_risk)

        # Increase risk for critical object types
        if target_object:
            if target_object.type in AdversarialGestureDetector.CRITICAL_OBJECT_TYPES:
                risk_score += 2  # Increase by 2 levels

            # Increase risk for extreme distances (potential spatial confusion)
            distance = target_object.distance_from(ar_context.user_position)
            if distance > 20.0:  # > 20 meters
                risk_score += 1

        # Increase risk in sensitive scenes
        sensitive_scenes = {"system_settings", "admin_mode", "calibration"}
        if ar_context.active_scene in sensitive_scenes:
            risk_score += 1

        # Cap risk score
        risk_score = min(risk_score, 4)  # Max CRITICAL (4)

        return self._score_to_risk(risk_score)

    def _risk_to_score(self, risk: RiskLevel) -> int:
        """Convert RiskLevel to numeric score."""
        mapping = {
            RiskLevel.SAFE: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }
        return mapping.get(risk, 2)

    def _score_to_risk(self, score: int) -> RiskLevel:
        """Convert numeric score to RiskLevel."""
        mapping = {
            0: RiskLevel.SAFE,
            1: RiskLevel.LOW,
            2: RiskLevel.MEDIUM,
            3: RiskLevel.HIGH,
            4: RiskLevel.CRITICAL,
        }
        return mapping.get(score, RiskLevel.MEDIUM)

    def _log_decision(self, decision: ARSafetyDecision):
        """Log AR safety decision."""
        log_msg = (
            f"AR Safety decision: action={decision.ar_action_category.value if decision.ar_action_category else 'unknown'}, "
            f"gesture={decision.gesture_type}, "
            f"risk={decision.risk_level.value}, "
            f"allowed={decision.allowed}, "
            f"requires_approval={decision.requires_approval}"
        )

        if decision.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get AR safety statistics.

        Returns:
            Dictionary of statistics including AR-specific metrics
        """
        base_stats = self.guardrails.get_statistics()

        total_ar_decisions = len(self.ar_decisions)
        allowed = sum(1 for d in self.ar_decisions if d.allowed)
        blocked = total_ar_decisions - allowed

        by_ar_category = {}
        for category in ARActionCategory:
            category_decisions = [
                d for d in self.ar_decisions
                if d.ar_action_category == category
            ]
            by_ar_category[category.value] = {
                "total": len(category_decisions),
                "allowed": sum(1 for d in category_decisions if d.allowed),
                "blocked": sum(1 for d in category_decisions if not d.allowed),
            }

        by_gesture_type = {}
        for decision in self.ar_decisions:
            if decision.gesture_type:
                if decision.gesture_type not in by_gesture_type:
                    by_gesture_type[decision.gesture_type] = {"total": 0, "allowed": 0, "blocked": 0}
                by_gesture_type[decision.gesture_type]["total"] += 1
                if decision.allowed:
                    by_gesture_type[decision.gesture_type]["allowed"] += 1
                else:
                    by_gesture_type[decision.gesture_type]["blocked"] += 1

        return {
            "base_statistics": base_stats,
            "ar_total_decisions": total_ar_decisions,
            "ar_allowed": allowed,
            "ar_blocked": blocked,
            "by_ar_category": by_ar_category,
            "by_gesture_type": by_gesture_type,
        }

    def clear_history(self):
        """Clear decision history (useful for testing)."""
        self.ar_decisions.clear()
        if self.adversarial_detector:
            self.adversarial_detector.clear_history()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_ar_safety_gate(
    enable_adversarial_detection: bool = True,
    testing_mode: bool = False,
) -> ARSafetyGate:
    """
    Create AR safety gate with optional configuration.

    Args:
        enable_adversarial_detection: Whether to detect adversarial gestures
        testing_mode: If True, bypass approval requirements (for development)

    Returns:
        Configured ARSafetyGate instance
    """
    return ARSafetyGate(
        enable_adversarial_detection=enable_adversarial_detection,
        testing_mode=testing_mode,
    )
