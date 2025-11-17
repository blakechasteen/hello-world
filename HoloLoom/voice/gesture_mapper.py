"""
Context-Aware Gesture Mapper
==============================
Maps hand gestures to commands based on AR context.

This module provides intelligent gesture-to-command mapping that adapts
based on the current AR scene, selected objects, and user context.

Author: HoloLoom Team (Agent M)
Date: November 17, 2025
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import time
import math

from HoloLoom.voice.gesture_recognition import Gesture, GestureType
from HoloLoom.voice.ar_context import ARContext, ARObject, ARObjectType, Vector3


# ============================================================================
# Gesture Command
# ============================================================================

@dataclass
class GestureCommand:
    """Mapped gesture command."""
    gesture_type: GestureType
    command: str  # e.g., "select_hive", "navigate_forward", "zoom_in"
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "gesture_type": self.gesture_type.value,
            "command": self.command,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }

    def __str__(self) -> str:
        return (
            f"GestureCommand({self.command}, "
            f"gesture={self.gesture_type.value}, "
            f"conf={self.confidence:.2f})"
        )


# ============================================================================
# Context Types
# ============================================================================

class ContextType(Enum):
    """AR context types for gesture mapping."""
    DEFAULT = "default"                  # No special context
    HIVE_SELECTED = "hive_selected"      # Beehive is selected
    TOOL_SELECTED = "tool_selected"      # Tool is selected
    NAVIGATION_ACTIVE = "navigation_active"  # Navigation in progress
    INSPECTION_MODE = "inspection_mode"  # Inspection mode active
    DATA_VIEW = "data_view"              # Viewing data panel
    MENU_OPEN = "menu_open"              # Menu is open


# ============================================================================
# Gesture Mapping Rules
# ============================================================================

@dataclass
class GestureMappingRule:
    """Rule for mapping gesture to command."""
    gesture_type: GestureType
    context_type: ContextType
    command: str
    confidence_multiplier: float = 1.0
    requires_target: bool = False
    description: str = ""

    def matches(self, gesture: Gesture, context_type: ContextType) -> bool:
        """Check if rule matches gesture and context."""
        if self.gesture_type != gesture.type:
            return False

        if self.context_type != context_type and self.context_type != ContextType.DEFAULT:
            return False

        if self.requires_target and gesture.target_position is None:
            return False

        return True


# ============================================================================
# Context-Aware Gesture Mapper
# ============================================================================

class ContextAwareGestureMapper:
    """Maps gestures to commands based on AR context."""

    def __init__(self):
        """Initialize gesture mapper with default mappings."""
        self.gesture_history: List[Gesture] = []
        self.command_history: List[GestureCommand] = []
        self.mapping_rules = self._load_default_mappings()

    def _load_default_mappings(self) -> List[GestureMappingRule]:
        """Load default gesture mapping rules."""
        rules = []

        # ============ POINT Gesture ============
        rules.append(GestureMappingRule(
            gesture_type=GestureType.POINT,
            context_type=ContextType.DEFAULT,
            command="select_object",
            requires_target=True,
            confidence_multiplier=1.0,
            description="Point at object to select it"
        ))

        rules.append(GestureMappingRule(
            gesture_type=GestureType.POINT,
            context_type=ContextType.NAVIGATION_ACTIVE,
            command="set_waypoint",
            requires_target=True,
            confidence_multiplier=0.9,
            description="Point to set navigation waypoint"
        ))

        # ============ OPEN_PALM Gesture ============
        rules.append(GestureMappingRule(
            gesture_type=GestureType.OPEN_PALM,
            context_type=ContextType.HIVE_SELECTED,
            command="show_details",
            confidence_multiplier=0.95,
            description="Show details panel for selected hive"
        ))

        rules.append(GestureMappingRule(
            gesture_type=GestureType.OPEN_PALM,
            context_type=ContextType.MENU_OPEN,
            command="close_menu",
            confidence_multiplier=0.9,
            description="Close open menu"
        ))

        rules.append(GestureMappingRule(
            gesture_type=GestureType.OPEN_PALM,
            context_type=ContextType.DEFAULT,
            command="open_menu",
            confidence_multiplier=0.85,
            description="Open main menu"
        ))

        # ============ GRAB Gesture ============
        rules.append(GestureMappingRule(
            gesture_type=GestureType.GRAB,
            context_type=ContextType.HIVE_SELECTED,
            command="deselect_object",
            confidence_multiplier=0.9,
            description="Deselect current object"
        ))

        rules.append(GestureMappingRule(
            gesture_type=GestureType.GRAB,
            context_type=ContextType.DATA_VIEW,
            command="hide_data",
            confidence_multiplier=0.9,
            description="Hide data panel"
        ))

        # ============ SWIPE Gestures ============
        rules.append(GestureMappingRule(
            gesture_type=GestureType.SWIPE_LEFT,
            context_type=ContextType.HIVE_SELECTED,
            command="navigate_previous",
            confidence_multiplier=0.9,
            description="Navigate to previous hive"
        ))

        rules.append(GestureMappingRule(
            gesture_type=GestureType.SWIPE_RIGHT,
            context_type=ContextType.HIVE_SELECTED,
            command="navigate_next",
            confidence_multiplier=0.9,
            description="Navigate to next hive"
        ))

        rules.append(GestureMappingRule(
            gesture_type=GestureType.SWIPE_UP,
            context_type=ContextType.DATA_VIEW,
            command="scroll_up",
            confidence_multiplier=0.85,
            description="Scroll data panel up"
        ))

        rules.append(GestureMappingRule(
            gesture_type=GestureType.SWIPE_DOWN,
            context_type=ContextType.DATA_VIEW,
            command="scroll_down",
            confidence_multiplier=0.85,
            description="Scroll data panel down"
        ))

        rules.append(GestureMappingRule(
            gesture_type=GestureType.SWIPE_UP,
            context_type=ContextType.DEFAULT,
            command="show_navigation",
            confidence_multiplier=0.8,
            description="Show navigation overlay"
        ))

        # ============ PINCH Gesture ============
        rules.append(GestureMappingRule(
            gesture_type=GestureType.PINCH,
            context_type=ContextType.DATA_VIEW,
            command="toggle_zoom",
            confidence_multiplier=0.9,
            description="Toggle zoom on data"
        ))

        rules.append(GestureMappingRule(
            gesture_type=GestureType.PINCH,
            context_type=ContextType.HIVE_SELECTED,
            command="start_inspection",
            confidence_multiplier=0.85,
            description="Start detailed inspection"
        ))

        # ============ CIRCLE Gesture ============
        rules.append(GestureMappingRule(
            gesture_type=GestureType.CIRCLE,
            context_type=ContextType.DEFAULT,
            command="refresh_view",
            confidence_multiplier=0.8,
            description="Refresh current view"
        ))

        # ============ WAVE Gesture ============
        rules.append(GestureMappingRule(
            gesture_type=GestureType.WAVE,
            context_type=ContextType.DEFAULT,
            command="call_assistant",
            confidence_multiplier=0.85,
            description="Call Elle assistant"
        ))

        return rules

    async def map_gesture(
        self,
        gesture: Gesture,
        ar_context: ARContext
    ) -> Optional[GestureCommand]:
        """
        Map gesture to command based on context.

        Args:
            gesture: Detected gesture
            ar_context: Current AR context

        Returns:
            Mapped command, or None if no mapping found

        Context-aware examples:
        - Point at hive → select_hive
        - Point at sky → show_navigation
        - Open palm while looking at hive → show_details
        - Swipe left while in hive view → previous_hive
        - Pinch while looking at data → zoom_in
        """
        # Update gesture history
        self.gesture_history.append(gesture)
        self.gesture_history = self.gesture_history[-10:]  # Keep last 10

        # Determine context type
        context_type = self._determine_context(ar_context)

        # Find matching rules
        matching_rules = [
            rule for rule in self.mapping_rules
            if rule.matches(gesture, context_type)
        ]

        if not matching_rules:
            # Try DEFAULT context as fallback
            matching_rules = [
                rule for rule in self.mapping_rules
                if rule.gesture_type == gesture.type and rule.context_type == ContextType.DEFAULT
            ]

        if not matching_rules:
            return None

        # Use first matching rule (rules are ordered by priority)
        rule = matching_rules[0]

        # Build command parameters
        parameters = await self._build_parameters(gesture, rule, ar_context)

        # Calculate confidence
        confidence = gesture.confidence * rule.confidence_multiplier

        command = GestureCommand(
            gesture_type=gesture.type,
            command=rule.command,
            parameters=parameters,
            confidence=confidence
        )

        # Update command history
        self.command_history.append(command)
        self.command_history = self.command_history[-10:]

        return command

    def _determine_context(self, ar_context: ARContext) -> ContextType:
        """Determine current context type."""
        # Check if hive is selected
        if ar_context.selected_object:
            if ar_context.selected_object.type == ARObjectType.BEEHIVE:
                return ContextType.HIVE_SELECTED
            elif ar_context.selected_object.type == ARObjectType.TOOL:
                return ContextType.TOOL_SELECTED

        # Check conversation context for other states
        if "navigation" in ar_context.conversation_context.lower():
            return ContextType.NAVIGATION_ACTIVE

        if "inspection" in ar_context.conversation_context.lower():
            return ContextType.INSPECTION_MODE

        if "data" in ar_context.conversation_context.lower():
            return ContextType.DATA_VIEW

        if "menu" in ar_context.conversation_context.lower():
            return ContextType.MENU_OPEN

        return ContextType.DEFAULT

    async def _build_parameters(
        self,
        gesture: Gesture,
        rule: GestureMappingRule,
        ar_context: ARContext
    ) -> Dict[str, Any]:
        """Build command parameters from gesture and context."""
        parameters = {}

        # For POINT gesture with target
        if gesture.type == GestureType.POINT and gesture.target_position:
            # Find object at target position
            target_obj = self._find_object_at_position(
                gesture.target_position,
                ar_context
            )

            if target_obj:
                parameters["object_id"] = target_obj.id
                parameters["object_type"] = target_obj.type.value
                parameters["object_name"] = target_obj.name
            else:
                # Use target position directly
                parameters["target_position"] = gesture.target_position

        # For SWIPE gestures
        if gesture.type in (
            GestureType.SWIPE_LEFT,
            GestureType.SWIPE_RIGHT,
            GestureType.SWIPE_UP,
            GestureType.SWIPE_DOWN
        ):
            if gesture.direction:
                parameters["direction"] = gesture.direction

        # Add current selected object if relevant
        if ar_context.selected_object:
            parameters["current_selection"] = ar_context.selected_object.id

        # Add hand information
        parameters["hand"] = gesture.hand.handedness

        return parameters

    def _find_object_at_position(
        self,
        target_position: Tuple[float, float, float],
        ar_context: ARContext
    ) -> Optional[ARObject]:
        """
        Find AR object at target position (raycast simulation).

        Args:
            target_position: Normalized position (-1 to 1, -1 to 1, depth)
            ar_context: Current AR context

        Returns:
            Closest object to target position, or None
        """
        # Convert target position to Vector3
        target = Vector3(target_position[0], target_position[1], target_position[2])

        # Find closest visible object
        closest_obj = None
        closest_dist = float('inf')

        for obj in ar_context.visible_objects:
            if not obj.selectable:
                continue

            # Project object position to screen space (simplified)
            # In real implementation, this would use camera projection matrix
            screen_pos = self._project_to_screen(obj.position, ar_context)

            # Calculate distance in screen space
            dist = math.sqrt(
                (screen_pos[0] - target.x) ** 2 +
                (screen_pos[1] - target.y) ** 2
            )

            if dist < closest_dist and dist < 0.2:  # Within 0.2 normalized units
                closest_dist = dist
                closest_obj = obj

        return closest_obj

    def _project_to_screen(
        self,
        world_pos: Vector3,
        ar_context: ARContext
    ) -> Tuple[float, float]:
        """
        Project world position to screen space (simplified).

        In real implementation, this would use proper camera projection.
        For now, we use a simple perspective projection.

        Args:
            world_pos: Position in world space
            ar_context: Current AR context

        Returns:
            (x, y) in normalized screen space (-1 to 1)
        """
        # Get vector from user to object
        dx = world_pos.x - ar_context.user_position.x
        dy = world_pos.y - ar_context.user_position.y
        dz = world_pos.z - ar_context.user_position.z

        # Simple perspective projection
        if dz <= 0:
            return (0, 0)  # Behind camera

        # Project to screen
        fov = 60.0  # Field of view in degrees
        aspect = 16.0 / 9.0  # Aspect ratio

        screen_x = (dx / dz) / math.tan(math.radians(fov / 2))
        screen_y = (dy / dz) / math.tan(math.radians(fov / 2)) * aspect

        return (screen_x, screen_y)

    def get_gesture_history(self, limit: int = 10) -> List[Gesture]:
        """Get recent gesture history."""
        return self.gesture_history[-limit:]

    def get_command_history(self, limit: int = 10) -> List[GestureCommand]:
        """Get recent command history."""
        return self.command_history[-limit:]

    def add_custom_rule(self, rule: GestureMappingRule) -> None:
        """Add a custom mapping rule."""
        self.mapping_rules.append(rule)

    def clear_history(self) -> None:
        """Clear gesture and command history."""
        self.gesture_history.clear()
        self.command_history.clear()


# ============================================================================
# Helper Functions
# ============================================================================

def create_default_mapper() -> ContextAwareGestureMapper:
    """Create a gesture mapper with default mappings."""
    return ContextAwareGestureMapper()


if __name__ == "__main__":
    # Example usage
    print("Context-Aware Gesture Mapper Demo")
    print("=" * 50)

    from HoloLoom.voice.ar_context import create_test_context
    from HoloLoom.voice.gesture_recognition import create_test_hand

    # Create test context
    context = create_test_context()
    print(f"\nAR Context: {context}")
    print(f"  Selected: {context.selected_object}")
    print(f"  Visible objects: {len(context.visible_objects)}")

    # Create mapper
    mapper = ContextAwareGestureMapper()

    # Test gesture: POINT
    print("\n" + "=" * 50)
    print("Test 1: POINT gesture at hive")
    hand = create_test_hand(GestureType.POINT)
    gesture = Gesture(
        type=GestureType.POINT,
        hand=hand,
        confidence=0.9,
        target_position=(0.1, 0.0, 5.0)  # Pointing at hive_003
    )

    import asyncio
    command = asyncio.run(mapper.map_gesture(gesture, context))
    if command:
        print(f"  Command: {command}")
        print(f"  Parameters: {command.parameters}")

    # Test gesture: OPEN_PALM with selected hive
    print("\n" + "=" * 50)
    print("Test 2: OPEN_PALM gesture with hive selected")
    context.selected_object = context.visible_objects[0]  # Select hive_001
    hand = create_test_hand(GestureType.OPEN_PALM)
    gesture = Gesture(
        type=GestureType.OPEN_PALM,
        hand=hand,
        confidence=0.95
    )

    command = asyncio.run(mapper.map_gesture(gesture, context))
    if command:
        print(f"  Command: {command}")
        print(f"  Parameters: {command.parameters}")

    # Test gesture: SWIPE_LEFT with hive selected
    print("\n" + "=" * 50)
    print("Test 3: SWIPE_LEFT gesture with hive selected")
    hand = create_test_hand(GestureType.GRAB)  # Reuse grab hand
    gesture = Gesture(
        type=GestureType.SWIPE_LEFT,
        hand=hand,
        confidence=0.85,
        direction=(-1.0, 0.0, 0.0)
    )

    command = asyncio.run(mapper.map_gesture(gesture, context))
    if command:
        print(f"  Command: {command}")
        print(f"  Parameters: {command.parameters}")

    # Show mapping rules
    print("\n" + "=" * 50)
    print(f"Total mapping rules: {len(mapper.mapping_rules)}")
    print("\nSample rules:")
    for rule in mapper.mapping_rules[:5]:
        print(f"  {rule.gesture_type.value} + {rule.context_type.value} → {rule.command}")
