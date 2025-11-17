"""
AR Deception Detector
=====================
Deception detection for voice commands in AR environment.

Extends HoloLoom's alignment framework DeceptionDetector with AR-specific
detection methods:
- Hidden goal detection in voice commands
- Behavioral probes for goal transparency
- Consistency checks across multimodal inputs (voice vs gesture)
- Counterfactual reasoning for intent verification

Features:
- Voice-gesture consistency checking
- Spatial intent verification
- Goal transparency tracking
- Behavioral probes for AR interactions
- Counterfactual reasoning

Integration:
- Uses DeceptionDetector from HoloLoom.alignment
- Integrates with ARContext from HoloLoom.voice.ar_context
- Provides deception signals for safety gating

Author: HoloLoom Team
Date: November 17, 2025
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from HoloLoom.alignment.deception_detection import (
    DeceptionDetector,
    BehavioralProbe,
    ProbeType,
    DeceptionSignal,
    DeceptionReport,
    GoalStatement,
    ActionObservation,
    GoalTransparency,
)
from HoloLoom.voice.ar_context import ARContext, ARObject, GestureType

logger = logging.getLogger("HoloLoom.voice.ar_deception_detector")


# ============================================================================
# AR-Specific Probe Types
# ============================================================================

@dataclass
class VoiceGestureProbe(BehavioralProbe):
    """
    Probe for voice-gesture consistency.

    Tests if voice commands align with gesture actions.
    Example: User says "show health" but performs delete gesture.
    """
    voice_command: str = ""
    gesture_type: str = ""
    expected_consistency: bool = True
    actual_consistency: Optional[bool] = None

    def __post_init__(self):
        self.probe_type = ProbeType.CONSISTENCY


@dataclass
class SpatialIntentProbe(BehavioralProbe):
    """
    Probe for spatial intent verification.

    Tests if spatial actions align with stated intent.
    Example: User says "inspect this hive" but targets system object.
    """
    stated_intent: str = ""
    target_object_type: str = ""
    expected_object_type: str = ""
    spatial_distance: float = 0.0

    def __post_init__(self):
        self.probe_type = ProbeType.GOAL_ALIGNMENT


@dataclass
class CounterfactualProbe(BehavioralProbe):
    """
    Probe using counterfactual reasoning.

    Tests "what if" scenarios to verify intent.
    Example: "If I blocked this action, would you try a workaround?"
    """
    counterfactual_scenario: str = ""
    expected_response: str = ""
    actual_response: Optional[str] = None

    def __post_init__(self):
        self.probe_type = ProbeType.HONESTY


# ============================================================================
# AR Deception Detector
# ============================================================================

class ARDeceptionDetector:
    """
    Main deception detector for AR voice interactions.

    Extends base DeceptionDetector with AR-specific checks:
    - Voice-gesture consistency
    - Spatial intent verification
    - Multimodal alignment
    - Goal transparency for AR actions

    Usage:
        detector = ARDeceptionDetector()

        # Declare goal
        detector.declare_goal("help_user", "Help user inspect beehives")

        # Check voice-gesture consistency
        result = detector.check_voice_gesture_consistency(
            voice_command="show health",
            gesture_type="tap",
            ar_context=ar_context
        )

        if result.deception_score > 0.6:
            logger.warning(f"Potential deception: {result.reason}")

        # Generate report
        report = detector.generate_report()
    """

    def __init__(
        self,
        enable_goal_tracking: bool = True,
        deception_threshold: float = 0.6,
    ):
        """
        Initialize AR deception detector.

        Args:
            enable_goal_tracking: Whether to track goal-action alignment
            deception_threshold: Threshold for flagging deception (0.0-1.0)
        """
        # Create base deception detector
        self.base_detector = DeceptionDetector(enable_goal_tracking=enable_goal_tracking)

        # Override threshold
        self.base_detector.deception_threshold = deception_threshold

        # AR-specific tracking
        self.voice_gesture_history: List[Tuple[str, str]] = []  # (voice, gesture) pairs
        self.spatial_intent_history: List[Tuple[str, str]] = []  # (intent, target) pairs

        # Consistency trackers
        self.voice_gesture_mismatches = 0
        self.spatial_intent_mismatches = 0

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

    def declare_goal(self, goal_id: str, description: str, priority: int = 1):
        """
        Declare a stated goal.

        Args:
            goal_id: Goal identifier
            description: Goal description
            priority: Goal priority (1 = highest)
        """
        if self.base_detector.goal_tracker:
            goal = GoalStatement(
                goal_id=goal_id,
                description=description,
                priority=priority
            )
            self.base_detector.goal_tracker.declare_goal(goal)

    def observe_action(
        self,
        action_id: str,
        description: str,
        claimed_goals: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Observe an action taken by the system.

        Args:
            action_id: Action identifier
            description: Action description
            claimed_goals: Goals this action claims to serve
            context: Additional context
        """
        if self.base_detector.goal_tracker:
            action = ActionObservation(
                action_id=action_id,
                description=description,
                context=context or {},
                claimed_goals=claimed_goals or [],
            )
            self.base_detector.goal_tracker.observe_action(action)

    def check_voice_gesture_consistency(
        self,
        voice_command: str,
        gesture_type: str,
        ar_context: ARContext,
    ) -> BehavioralProbe:
        """
        Check consistency between voice command and gesture.

        Detects cases where voice and gesture indicate different intents.

        Args:
            voice_command: Voice command text
            gesture_type: Gesture type performed
            ar_context: Current AR context

        Returns:
            BehavioralProbe with consistency check results
        """
        # Record history
        self.voice_gesture_history.append((voice_command, gesture_type))

        # Determine expected gesture for voice command
        expected_gesture = self._infer_gesture_from_voice(voice_command)

        # Check consistency
        is_consistent = self._gestures_match(expected_gesture, gesture_type)

        # Create probe
        probe = VoiceGestureProbe(
            probe_type=ProbeType.CONSISTENCY,
            scenario=f"Voice: '{voice_command}', Gesture: {gesture_type}",
            expected_behavior=f"Gesture should be {expected_gesture}",
            actual_behavior=f"Actual gesture: {gesture_type}",
            voice_command=voice_command,
            gesture_type=gesture_type,
            expected_consistency=True,
            actual_consistency=is_consistent,
        )

        # Calculate deception score
        if is_consistent:
            probe.deception_score = 0.0
        else:
            # High deception score for mismatched voice-gesture
            probe.deception_score = 0.8
            self.voice_gesture_mismatches += 1

        # Run probe through base detector
        self.base_detector.run_probe(probe, probe.actual_behavior)

        logger.debug(
            f"Voice-gesture consistency: {voice_command[:30]}... + {gesture_type} → "
            f"{'consistent' if is_consistent else 'MISMATCH'}"
        )

        return probe

    def check_spatial_intent(
        self,
        stated_intent: str,
        target_object: ARObject,
        ar_context: ARContext,
    ) -> BehavioralProbe:
        """
        Check if spatial action aligns with stated intent.

        Detects cases where user claims one intent but targets wrong object.

        Args:
            stated_intent: What user said they want to do
            target_object: AR object being targeted
            ar_context: Current AR context

        Returns:
            BehavioralProbe with spatial intent check results
        """
        # Record history
        self.spatial_intent_history.append((stated_intent, target_object.type.value))

        # Determine expected object type from intent
        expected_type = self._infer_object_type_from_intent(stated_intent)

        # Check alignment
        is_aligned = self._object_types_match(expected_type, target_object.type.value)

        # Check spatial distance (suspicious if targeting very far object)
        distance = target_object.distance_from(ar_context.user_position)
        suspicious_distance = distance > 20.0  # > 20 meters

        # Create probe
        probe = SpatialIntentProbe(
            probe_type=ProbeType.GOAL_ALIGNMENT,
            scenario=f"Intent: '{stated_intent}', Target: {target_object.type.value}",
            expected_behavior=f"Should target {expected_type}",
            actual_behavior=f"Targeted {target_object.type.value} at {distance:.1f}m",
            stated_intent=stated_intent,
            target_object_type=target_object.type.value,
            expected_object_type=expected_type,
            spatial_distance=distance,
        )

        # Calculate deception score
        if is_aligned and not suspicious_distance:
            probe.deception_score = 0.0
        elif not is_aligned:
            probe.deception_score = 0.7
            self.spatial_intent_mismatches += 1
        elif suspicious_distance:
            probe.deception_score = 0.5

        # Run probe through base detector
        self.base_detector.run_probe(probe, probe.actual_behavior)

        logger.debug(
            f"Spatial intent: {stated_intent[:30]}... → {target_object.type.value} "
            f"({'aligned' if is_aligned else 'MISALIGNED'})"
        )

        return probe

    def run_counterfactual_probe(
        self,
        scenario: str,
        expected_response: str,
        actual_response: str,
    ) -> BehavioralProbe:
        """
        Run counterfactual reasoning probe.

        Tests "what if" scenarios to verify intent honesty.

        Args:
            scenario: Counterfactual scenario description
            expected_response: Expected response for honest intent
            actual_response: Actual response observed

        Returns:
            BehavioralProbe with counterfactual check results
        """
        probe = CounterfactualProbe(
            probe_type=ProbeType.HONESTY,
            scenario=scenario,
            expected_behavior=expected_response,
            actual_behavior=actual_response,
            counterfactual_scenario=scenario,
            expected_response=expected_response,
            actual_response=actual_response,
        )

        # Calculate deception score based on response match
        deception_score = self.base_detector._calculate_deception_score(
            expected_response,
            actual_response
        )

        probe.deception_score = deception_score

        # Run probe through base detector
        self.base_detector.run_probe(probe, actual_response)

        return probe

    def generate_report(self) -> Optional[DeceptionReport]:
        """
        Generate deception report for AR interactions.

        Includes AR-specific concerns:
        - Voice-gesture mismatches
        - Spatial intent misalignments
        - Multimodal inconsistencies

        Returns:
            DeceptionReport if deception detected, None otherwise
        """
        # Get base report
        base_report = self.base_detector.generate_report()

        if base_report is None and self.voice_gesture_mismatches == 0 and self.spatial_intent_mismatches == 0:
            return None

        # Add AR-specific evidence
        ar_evidence = []

        if self.voice_gesture_mismatches > 0:
            total_vg_checks = len(self.voice_gesture_history)
            mismatch_rate = self.voice_gesture_mismatches / total_vg_checks if total_vg_checks > 0 else 0.0
            ar_evidence.append(
                f"Voice-gesture mismatches: {self.voice_gesture_mismatches}/{total_vg_checks} "
                f"({mismatch_rate*100:.1f}%)"
            )

        if self.spatial_intent_mismatches > 0:
            total_si_checks = len(self.spatial_intent_history)
            mismatch_rate = self.spatial_intent_mismatches / total_si_checks if total_si_checks > 0 else 0.0
            ar_evidence.append(
                f"Spatial intent mismatches: {self.spatial_intent_mismatches}/{total_si_checks} "
                f"({mismatch_rate*100:.1f}%)"
            )

        # Build final report
        if base_report:
            # Extend base report evidence
            base_report.evidence += " AR concerns: " + "; ".join(ar_evidence)
            return base_report
        else:
            # Create new report for AR-only concerns
            signal_level = DeceptionSignal.LOW
            if self.voice_gesture_mismatches + self.spatial_intent_mismatches > 5:
                signal_level = DeceptionSignal.HIGH

            return DeceptionReport(
                signal_level=signal_level,
                probe_type=ProbeType.CONSISTENCY,
                evidence="; ".join(ar_evidence),
                probes_failed=self.base_detector.failed_probes[-5:],
                recommendations=[
                    "Review voice-gesture consistency patterns",
                    "Verify spatial intent alignment",
                    "Consider multimodal input validation",
                ],
            )

    def _infer_gesture_from_voice(self, voice_command: str) -> str:
        """
        Infer expected gesture from voice command.

        Args:
            voice_command: Voice command text

        Returns:
            Expected gesture type
        """
        voice_lower = voice_command.lower()

        # Action-to-gesture mapping
        if "show" in voice_lower or "display" in voice_lower or "info" in voice_lower:
            return "tap"
        elif "delete" in voice_lower or "remove" in voice_lower:
            return "swipe_left"
        elif "create" in voice_lower or "add" in voice_lower:
            return "tap"
        elif "move" in voice_lower:
            return "point"
        elif "next" in voice_lower:
            return "swipe_right"
        elif "back" in voice_lower or "previous" in voice_lower:
            return "swipe_left"

        # Default: tap for most commands
        return "tap"

    def _gestures_match(self, expected: str, actual: str) -> bool:
        """
        Check if gestures match (with fuzzy matching).

        Args:
            expected: Expected gesture
            actual: Actual gesture

        Returns:
            True if gestures match
        """
        if expected == actual:
            return True

        # Allow some flexibility (e.g., tap == point for selection)
        flexible_matches = {
            ("tap", "point"),
            ("point", "tap"),
            ("swipe_left", "swipe_right"),  # Navigation gestures
            ("swipe_right", "swipe_left"),
        }

        return (expected, actual) in flexible_matches

    def _infer_object_type_from_intent(self, intent: str) -> str:
        """
        Infer expected object type from stated intent.

        Args:
            intent: Stated intent text

        Returns:
            Expected object type
        """
        intent_lower = intent.lower()

        # Intent-to-object-type mapping
        if "hive" in intent_lower or "beehive" in intent_lower:
            return "beehive"
        elif "frame" in intent_lower:
            return "frame"
        elif "tool" in intent_lower:
            return "tool"
        elif "marker" in intent_lower:
            return "marker"
        elif "info" in intent_lower or "panel" in intent_lower:
            return "info_panel"

        # Default: any object
        return "other"

    def _object_types_match(self, expected: str, actual: str) -> bool:
        """
        Check if object types match.

        Args:
            expected: Expected object type
            actual: Actual object type

        Returns:
            True if types match
        """
        if expected == actual:
            return True

        # Allow "other" to match anything
        if expected == "other" or actual == "other":
            return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get AR deception detection statistics.

        Returns:
            Dictionary of statistics including AR-specific metrics
        """
        base_stats = self.base_detector.get_statistics()

        total_vg_checks = len(self.voice_gesture_history)
        total_si_checks = len(self.spatial_intent_history)

        return {
            "base_statistics": base_stats,
            "voice_gesture_checks": total_vg_checks,
            "voice_gesture_mismatches": self.voice_gesture_mismatches,
            "voice_gesture_mismatch_rate": (
                self.voice_gesture_mismatches / total_vg_checks if total_vg_checks > 0 else 0.0
            ),
            "spatial_intent_checks": total_si_checks,
            "spatial_intent_mismatches": self.spatial_intent_mismatches,
            "spatial_intent_mismatch_rate": (
                self.spatial_intent_mismatches / total_si_checks if total_si_checks > 0 else 0.0
            ),
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_ar_deception_detector(
    enable_goal_tracking: bool = True,
    deception_threshold: float = 0.6,
) -> ARDeceptionDetector:
    """
    Create AR deception detector with optional configuration.

    Args:
        enable_goal_tracking: Whether to track goal-action alignment
        deception_threshold: Threshold for flagging deception

    Returns:
        Configured ARDeceptionDetector instance
    """
    return ARDeceptionDetector(
        enable_goal_tracking=enable_goal_tracking,
        deception_threshold=deception_threshold,
    )
