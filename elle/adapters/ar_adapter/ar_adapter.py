"""
AR Adapter - Main adapter for AR platform integration

Translates between AR platform events and Elle requests/responses.

Flow:
    AR Event → ar_event_to_elle_request() → ElleRequest
    ElleRequest → Elle Engine → ElleAction
    ElleAction → elle_action_to_visualizations() → ARVisualization[]

Created: 2025-11-22
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

# Elle core imports
from elle.domain.scene import SceneSnapshot, ObjectInScene, Intent
from elle.domain.action import ElleAction, Symbol

# AR adapter imports
from elle.adapters.ar_adapter.ar_events import (
    AREvent,
    AREventType,
    ARContext,
    ARObject,
    Vector3,
    GazeEvent,
    ScanEvent,
    SelectionEvent,
    GestureEvent,
    VoiceEvent,
)
from elle.adapters.ar_adapter.ar_renderer import (
    ARVisualization,
    ARVisualizationSet,
    AROverlay,
    ARHighlight,
    ARPath,
    ARPanel,
    VisualStyle,
)


logger = logging.getLogger(__name__)


# ============================================================================
# AR → Elle Request Conversion
# ============================================================================

class ARAdapter:
    """
    Main AR adapter coordinating AR ↔ Elle translation.

    Follows adapter pattern from elle/adapters/cli_adapter/cli.py
    """

    def __init__(self):
        self.session_id: Optional[str] = None
        self.active_visualizations = ARVisualizationSet()

    # ========================================================================
    # AR Event → Elle Request
    # ========================================================================

    def ar_event_to_elle_request(self, event: AREvent) -> Dict[str, Any]:
        """
        Convert AR platform event to Elle request format.

        Returns dict matching ElleRequest structure:
            {
                "scene": SceneSnapshot,
                "intent": Intent,
                "user_context": {...},
                "metadata": {...}
            }
        """
        # Convert AR context to Elle scene
        scene = self._ar_context_to_scene(event.context)

        # Infer intent from event type
        intent = self._infer_intent(event)

        # Build user context
        user_context = self._build_user_context(event)

        return {
            "scene": scene,
            "intent": intent,
            "user_context": user_context,
            "metadata": {
                "event_type": event.event_type.value,
                "session_id": event.context.session_id,
                "platform": event.context.platform,
                "timestamp": event.timestamp.isoformat(),
            },
        }

    def _ar_context_to_scene(self, context: ARContext) -> SceneSnapshot:
        """Convert AR spatial context to Elle scene snapshot"""
        objects = []

        for ar_obj in context.visible_objects:
            obj = ObjectInScene(
                id=ar_obj.id,
                label=ar_obj.label,
                object_type=ar_obj.object_type,
                position={
                    "x": ar_obj.position.x,
                    "y": ar_obj.position.y,
                    "z": ar_obj.position.z,
                },
                metadata={
                    "confidence": ar_obj.confidence,
                    **ar_obj.metadata,
                },
            )
            objects.append(obj)

        # Add gaze target if any
        gaze_target = None
        if context.gaze_target:
            gaze_obj = context.get_object_by_id(context.gaze_target)
            if gaze_obj:
                gaze_target = gaze_obj.label

        return SceneSnapshot(
            objects=objects,
            user_position={
                "x": context.user_position.x,
                "y": context.user_position.y,
                "z": context.user_position.z,
            },
            gaze_target=gaze_target,
            timestamp=context.timestamp,
            metadata={
                "visible_object_count": len(objects),
                "platform": context.platform,
            },
        )

    def _infer_intent(self, event: AREvent) -> Intent:
        """Infer user intent from AR event type"""
        if isinstance(event, GazeEvent):
            return Intent.SEEKING_GUIDANCE

        elif isinstance(event, ScanEvent):
            if event.scan_type == "slow_scan":
                return Intent.SEEKING_GUIDANCE
            return Intent.EXPLORING

        elif isinstance(event, SelectionEvent):
            return Intent.REQUESTING_ACTION

        elif isinstance(event, GestureEvent):
            if event.gesture_type in ["point", "grab"]:
                return Intent.REQUESTING_ACTION
            return Intent.EXPLORING

        elif isinstance(event, VoiceEvent):
            # Could classify intent from voice transcript
            # For now, default to seeking guidance
            return Intent.SEEKING_GUIDANCE

        return Intent.EXPLORING

    def _build_user_context(self, event: AREvent) -> Dict[str, Any]:
        """Build user context from AR event"""
        context = {
            "mode": "ar",
            "timestamp": event.timestamp.isoformat(),
        }

        # Add event-specific context
        if isinstance(event, VoiceEvent):
            context["voice"] = {
                "transcript": event.transcript,
                "intent": event.intent,
                "entities": event.entities,
                "confidence": event.confidence,
                "language": event.language,
            }

        elif isinstance(event, GazeEvent):
            context["gaze"] = {
                "target_id": event.target_object_id,
                "dwell_ms": event.dwell_duration_ms,
            }

        elif isinstance(event, SelectionEvent):
            context["selection"] = {
                "target_id": event.target_object_id,
                "method": event.selection_method,
            }

        elif isinstance(event, GestureEvent):
            context["gesture"] = {
                "type": event.gesture_type,
                "hand": event.hand,
                "params": event.gesture_params,
            }

        return context

    # ========================================================================
    # Elle Action → AR Visualizations
    # ========================================================================

    def elle_action_to_visualizations(
        self,
        action: ElleAction,
        context: ARContext,
    ) -> ARVisualizationSet:
        """
        Convert Elle action to AR visualizations.

        Args:
            action: Elle's decision
            context: Current AR context (for target lookups)

        Returns:
            Set of visualizations to render
        """
        viz_set = ARVisualizationSet(
            metadata={
                "symbol": action.symbol.value if action.symbol else None,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Symbol influences visual style
        style = self._get_style_for_symbol(action.symbol)

        # Add visual guidance if present
        if action.visual_guidance:
            viz_set = self._add_visual_guidance(
                viz_set,
                action.visual_guidance,
                context,
                style,
            )

        # Add suggested actions as interactive elements
        if action.suggested_actions:
            viz_set = self._add_action_suggestions(
                viz_set,
                action.suggested_actions,
                context,
                style,
            )

        return viz_set

    def _get_style_for_symbol(self, symbol: Optional[Symbol]) -> VisualStyle:
        """Get visual style based on mythic symbol"""
        if symbol == Symbol.CHIMBORAZO:
            # Focus - bright, attention-grabbing
            return VisualStyle(
                color="#FFD700",  # Gold
                opacity=1.0,
                animation="pulse",
            )

        elif symbol == Symbol.PLATO:
            # Clarity - calm, informative
            return VisualStyle(
                color="#87CEEB",  # Sky blue
                opacity=0.9,
            )

        elif symbol == Symbol.PENELOPE:
            # Patience - soft, subtle
            return VisualStyle(
                color="#E6E6FA",  # Lavender
                opacity=0.7,
                animation="fade",
            )

        # Default
        return VisualStyle()

    def _add_visual_guidance(
        self,
        viz_set: ARVisualizationSet,
        guidance: str,
        context: ARContext,
        style: VisualStyle,
    ) -> ARVisualizationSet:
        """
        Add visual guidance visualization.

        Converts text guidance to appropriate AR visualization type.
        """
        # If guidance references an object, highlight it
        target_obj = self._find_referenced_object(guidance, context)

        if target_obj:
            # Highlight the referenced object
            viz_set.add(
                ARHighlight(
                    id=f"highlight_{target_obj.id}",
                    target_object_id=target_obj.id,
                    style=style,
                    auto_dismiss_sec=5.0,
                )
            )

            # Add overlay with guidance text
            viz_set.add(
                AROverlay(
                    id=f"overlay_{target_obj.id}",
                    target_object_id=target_obj.id,
                    content=guidance,
                    style=style,
                    auto_dismiss_sec=5.0,
                )
            )
        else:
            # No specific object - show as head-locked panel
            viz_set.add(
                ARPanel(
                    id="guidance_panel",
                    content={"text": guidance},
                    lock_mode="head",
                    style=style,
                    auto_dismiss_sec=3.0,
                )
            )

        return viz_set

    def _add_action_suggestions(
        self,
        viz_set: ARVisualizationSet,
        actions: List[str],
        context: ARContext,
        style: VisualStyle,
    ) -> ARVisualizationSet:
        """Add suggested actions as interactive elements"""
        # Create panel with action buttons
        panel_content = {
            "title": "Suggestions",
            "actions": actions,
        }

        viz_set.add(
            ARPanel(
                id="actions_panel",
                content=panel_content,
                lock_mode="head",
                position=Vector3(0.3, 1.3, -0.8),  # Right side, closer
                size=(0.3, 0.2),
                style=style,
                auto_dismiss_sec=10.0,
            )
        )

        return viz_set

    def _find_referenced_object(
        self,
        text: str,
        context: ARContext,
    ) -> Optional[ARObject]:
        """
        Find object referenced in text.

        Simple keyword matching - could be enhanced with NLP.
        """
        text_lower = text.lower()

        # Check if any object label is mentioned
        for obj in context.visible_objects:
            if obj.label.lower() in text_lower:
                return obj

        # Check gaze target as fallback
        if context.gaze_target:
            return context.get_object_by_id(context.gaze_target)

        return None

    # ========================================================================
    # Session Management
    # ========================================================================

    def start_session(self, session_id: str) -> None:
        """Initialize AR session"""
        self.session_id = session_id
        self.active_visualizations = ARVisualizationSet()
        logger.info(f"AR session started: {session_id}")

    def end_session(self) -> None:
        """Clean up AR session"""
        logger.info(f"AR session ended: {self.session_id}")
        self.session_id = None
        self.active_visualizations.clear_non_persistent()

    def update_visualizations(self, viz_set: ARVisualizationSet) -> None:
        """Update active visualizations"""
        # Clear non-persistent old visualizations
        self.active_visualizations.clear_non_persistent()

        # Add new visualizations
        for viz in viz_set.visualizations:
            self.active_visualizations.add(viz)
