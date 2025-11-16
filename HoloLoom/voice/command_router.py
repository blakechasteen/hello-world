"""
Command Router Module
======================
Intent parsing and action routing for voice commands.

This module parses natural language voice inputs into structured intents
and routes them to appropriate Elle AR actions.

Author: HoloLoom Team
Date: November 15, 2025
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import re

from HoloLoom.voice.ar_context import ARContext, ARObjectType


# ============================================================================
# Intent Types
# ============================================================================

class IntentType(Enum):
    """Types of voice intents."""
    QUERY = "query"             # "What is...", "Show me...", "Tell me about..."
    NAVIGATE = "navigate"       # "Go to...", "Take me to...", "Next/previous"
    SELECT = "select"           # "This one", "That hive"
    COMMAND = "command"         # "Start inspection", "Record observation"
    EXPLAIN = "explain"         # "Why is...", "Explain...", "How does..."
    UNKNOWN = "unknown"         # Could not classify


# ============================================================================
# Intent Data Structure
# ============================================================================

@dataclass
class Intent:
    """
    Structured representation of a voice command intent.

    Attributes:
        type: Intent classification
        action: Specific action to perform
        target: Target object/entity (ID or type)
        parameters: Additional parameters for the action
        confidence: Confidence score (0-1)
        raw_text: Original voice input
    """
    type: IntentType
    action: str
    target: Optional[str] = None
    parameters: Dict[str, Any] = None
    confidence: float = 0.0
    raw_text: str = ""

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "action": self.action,
            "target": self.target,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "raw_text": self.raw_text
        }

    def __str__(self) -> str:
        return (
            f"Intent({self.type.value}, action={self.action}, "
            f"target={self.target}, conf={self.confidence:.2f})"
        )


# ============================================================================
# Elle Actions
# ============================================================================

class ElleActionType(Enum):
    """Types of Elle AR actions."""
    DISPLAY_INFO = "display_info"
    HIGHLIGHT_OBJECT = "highlight_object"
    NAVIGATE_TO = "navigate_to"
    SELECT_OBJECT = "select_object"
    START_TASK = "start_task"
    RECORD_DATA = "record_data"
    SHOW_OVERLAY = "show_overlay"
    PLAY_ANIMATION = "play_animation"
    SPEAK = "speak"
    UNKNOWN = "unknown"


@dataclass
class ElleAction:
    """
    Action to be performed by Elle AR system.

    Attributes:
        type: Type of action
        target_object: ID of target AR object
        parameters: Action-specific parameters
        voice_response: Text response to speak
        visual_data: Data for AR visualization
    """
    type: ElleActionType
    target_object: Optional[str] = None
    parameters: Dict[str, Any] = None
    voice_response: str = ""
    visual_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.visual_data is None:
            self.visual_data = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "target_object": self.target_object,
            "parameters": self.parameters,
            "voice_response": self.voice_response,
            "visual_data": self.visual_data
        }

    def __str__(self) -> str:
        return f"ElleAction({self.type.value}, target={self.target_object})"


# ============================================================================
# Command Router
# ============================================================================

class CommandRouter:
    """
    Routes voice commands to Elle AR actions.

    This class parses natural language voice inputs, classifies intents,
    resolves spatial references using AR context, and generates appropriate
    Elle actions.
    """

    def __init__(self):
        """Initialize command router with intent patterns."""
        # Intent classification patterns (regex)
        self.intent_patterns = {
            IntentType.QUERY: [
                r"^(what|show|tell|display|describe)\s",
                r"\s(what|info|information|details|status)\s",
                r"^how (many|much)\s"
            ],
            IntentType.NAVIGATE: [
                r"^(go|take|bring|move|navigate)\s",
                r"^(next|previous|first|last)\s",
                r"\s(the way to|directions to)\s"
            ],
            IntentType.SELECT: [
                r"^(this|that|select|choose|pick)\s",
                r"the one (i'm|im)\s",
                r"on (my |the )(left|right|front)"
            ],
            IntentType.COMMAND: [
                r"^(start|begin|initiate|launch)\s",
                r"^(record|save|store|log)\s",
                r"^(mark|flag|tag)\s"
            ],
            IntentType.EXPLAIN: [
                r"^(why|explain|how|teach)\s",
                r"^what (does|is|are)\s.*\s(mean|do)\s",
                r"\s(explain|clarify|elaborate)\s"
            ]
        }

        # Action mapping from intents
        self.action_map = {
            # Query intents
            ("query", "show"): ElleActionType.DISPLAY_INFO,
            ("query", "what"): ElleActionType.DISPLAY_INFO,
            ("query", "tell"): ElleActionType.SPEAK,
            ("query", "display"): ElleActionType.SHOW_OVERLAY,
            ("query", "describe"): ElleActionType.SPEAK,

            # Navigate intents
            ("navigate", "go"): ElleActionType.NAVIGATE_TO,
            ("navigate", "take"): ElleActionType.NAVIGATE_TO,
            ("navigate", "next"): ElleActionType.NAVIGATE_TO,
            ("navigate", "previous"): ElleActionType.NAVIGATE_TO,

            # Select intents
            ("select", "this"): ElleActionType.SELECT_OBJECT,
            ("select", "that"): ElleActionType.SELECT_OBJECT,
            ("select", "select"): ElleActionType.SELECT_OBJECT,

            # Command intents
            ("command", "start"): ElleActionType.START_TASK,
            ("command", "record"): ElleActionType.RECORD_DATA,
            ("command", "mark"): ElleActionType.HIGHLIGHT_OBJECT,

            # Explain intents
            ("explain", "why"): ElleActionType.SPEAK,
            ("explain", "explain"): ElleActionType.SPEAK,
            ("explain", "how"): ElleActionType.SPEAK,
        }

    async def parse_intent(
        self,
        transcript: str,
        ar_context: ARContext
    ) -> Intent:
        """
        Parse voice input into structured intent.

        Args:
            transcript: Voice input text
            ar_context: Current AR context for disambiguation

        Returns:
            Intent with type, action, target, and confidence
        """
        text = transcript.lower().strip()

        # Classify intent type
        intent_type = self._classify_intent(text)

        # Extract action verb
        action_verb = self._extract_action_verb(text, intent_type)

        # Extract and resolve target
        target = self._resolve_target(text, ar_context)

        # Extract parameters
        parameters = self._extract_parameters(text, intent_type)

        # Calculate confidence
        confidence = self._calculate_confidence(text, intent_type, target)

        return Intent(
            type=intent_type,
            action=action_verb,
            target=target,
            parameters=parameters,
            confidence=confidence,
            raw_text=transcript
        )

    async def route_to_action(
        self,
        intent: Intent,
        ar_context: ARContext
    ) -> ElleAction:
        """
        Map intent to Elle action.

        Args:
            intent: Parsed intent
            ar_context: Current AR context

        Returns:
            ElleAction to execute
        """
        # Look up action type from intent
        key = (intent.type.value, intent.action)
        action_type = self.action_map.get(key, ElleActionType.UNKNOWN)

        # Generate voice response
        voice_response = self._generate_voice_response(intent, ar_context)

        # Generate visual data
        visual_data = self._generate_visual_data(intent, ar_context)

        return ElleAction(
            type=action_type,
            target_object=intent.target,
            parameters=intent.parameters,
            voice_response=voice_response,
            visual_data=visual_data
        )

    def _classify_intent(self, text: str) -> IntentType:
        """Classify intent type from text."""
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return intent_type
        return IntentType.UNKNOWN

    def _extract_action_verb(self, text: str, intent_type: IntentType) -> str:
        """Extract primary action verb from text."""
        # Common verbs by intent type
        verbs = {
            IntentType.QUERY: ["show", "tell", "what", "display", "describe"],
            IntentType.NAVIGATE: ["go", "take", "next", "previous", "navigate"],
            IntentType.SELECT: ["this", "that", "select", "choose", "pick"],
            IntentType.COMMAND: ["start", "record", "mark", "create", "save"],
            IntentType.EXPLAIN: ["why", "explain", "how", "teach"]
        }

        words = text.split()
        for verb in verbs.get(intent_type, []):
            if verb in words:
                return verb

        # Fallback to first word
        return words[0] if words else "unknown"

    def _resolve_target(self, text: str, ar_context: ARContext) -> Optional[str]:
        """
        Resolve target object from text using AR context.

        Handles spatial references like:
        - "this hive" → gaze target
        - "that one" → second closest
        - "the hive on my left" → directional search
        """
        text_lower = text.lower()

        # Update spatial references
        ar_context.update_spatial_references()

        # Check for "this" reference
        if "this" in text_lower:
            obj = ar_context.nearby_objects.get("this")
            return obj.id if obj else None

        # Check for "that" reference
        if "that" in text_lower:
            obj = ar_context.nearby_objects.get("that")
            return obj.id if obj else None

        # Check for "gaze" reference
        if "looking at" in text_lower or "seeing" in text_lower:
            return ar_context.gaze_target

        # Check for directional reference
        for direction in ["left", "right", "front", "back", "above", "below"]:
            if direction in text_lower:
                # Extract object type if mentioned
                object_type = self._extract_object_type(text_lower)
                obj = ar_context.find_object_in_direction(direction, object_type)
                return obj.id if obj else None

        # Check for "closest" or "nearest"
        if "closest" in text_lower or "nearest" in text_lower:
            object_type = self._extract_object_type(text_lower)
            obj = ar_context.find_closest_object(object_type)
            return obj.id if obj else None

        # Check for explicit hive number (e.g., "hive 003", "hive three")
        hive_match = re.search(r"hive\s+(\d+|one|two|three|four|five)", text_lower)
        if hive_match:
            hive_num = hive_match.group(1)
            # Convert word to number if needed
            number_map = {"one": "001", "two": "002", "three": "003", "four": "004", "five": "005"}
            if hive_num in number_map:
                hive_num = number_map[hive_num]
            elif len(hive_num) < 3:
                hive_num = hive_num.zfill(3)
            return f"hive_{hive_num}"

        # No target resolved
        return None

    def _extract_object_type(self, text: str) -> Optional[ARObjectType]:
        """Extract object type from text."""
        if "hive" in text or "beehive" in text:
            return ARObjectType.BEEHIVE
        if "frame" in text:
            return ARObjectType.FRAME
        if "tool" in text:
            return ARObjectType.TOOL
        if "marker" in text:
            return ARObjectType.MARKER
        return None

    def _extract_parameters(self, text: str, intent_type: IntentType) -> Dict[str, Any]:
        """Extract additional parameters from text."""
        params = {}

        # Extract data type for query intents
        if intent_type == IntentType.QUERY:
            if "health" in text:
                params["data_type"] = "health"
            elif "population" in text:
                params["data_type"] = "population"
            elif "temperature" in text:
                params["data_type"] = "temperature"
            elif "history" in text or "records" in text:
                params["data_type"] = "history"

        # Extract task name for command intents
        if intent_type == IntentType.COMMAND:
            if "inspection" in text:
                params["task_name"] = "inspection"
            elif "observation" in text or "note" in text:
                params["task_name"] = "observation"

        return params

    def _calculate_confidence(
        self,
        text: str,
        intent_type: IntentType,
        target: Optional[str]
    ) -> float:
        """
        Calculate confidence score for intent classification.

        Factors:
        - Clear intent pattern match: +0.4
        - Target resolved: +0.3
        - Complete sentence: +0.2
        - Length appropriate: +0.1
        """
        confidence = 0.5  # Base confidence

        # Bonus for clear pattern match
        if intent_type != IntentType.UNKNOWN:
            confidence += 0.3

        # Bonus for resolved target
        if target:
            confidence += 0.2

        # Bonus for complete sentence (has verb and object)
        words = text.split()
        if len(words) >= 3:
            confidence += 0.1

        # Penalty for very short or very long text
        if len(words) < 2:
            confidence -= 0.2
        elif len(words) > 20:
            confidence -= 0.1

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def _generate_voice_response(self, intent: Intent, ar_context: ARContext) -> str:
        """Generate voice response based on intent."""
        # Get target object details
        target_obj = None
        if intent.target:
            target_obj = ar_context.find_object_by_id(intent.target)

        # Generate response based on intent type
        if intent.type == IntentType.QUERY:
            if target_obj:
                return f"Showing {intent.parameters.get('data_type', 'information')} for {target_obj.name}."
            else:
                return "Let me find that information for you."

        elif intent.type == IntentType.NAVIGATE:
            if target_obj:
                distance = target_obj.distance_from(ar_context.user_position)
                return f"Navigating to {target_obj.name}, {distance:.1f} meters away."
            else:
                return "I'm not sure where you'd like to go."

        elif intent.type == IntentType.SELECT:
            if target_obj:
                return f"You've selected {target_obj.name}."
            else:
                return "I'm not sure which one you mean."

        elif intent.type == IntentType.COMMAND:
            task = intent.parameters.get("task_name", "task")
            return f"Starting {task}."

        elif intent.type == IntentType.EXPLAIN:
            return "Let me explain that for you."

        return "I'm processing your request."

    def _generate_visual_data(self, intent: Intent, ar_context: ARContext) -> Dict[str, Any]:
        """Generate AR visualization data based on intent."""
        visual_data = {}

        if intent.type == IntentType.QUERY:
            visual_data["overlay_type"] = "info_panel"
            visual_data["data_type"] = intent.parameters.get("data_type", "general")

        elif intent.type == IntentType.NAVIGATE:
            visual_data["show_path"] = True
            visual_data["show_distance"] = True

        elif intent.type == IntentType.SELECT:
            visual_data["highlight_color"] = "blue"
            visual_data["highlight_duration"] = 2.0

        elif intent.type == IntentType.COMMAND:
            visual_data["show_task_panel"] = True

        return visual_data


# ============================================================================
# Helper Functions
# ============================================================================

async def test_command_router():
    """Test command router with sample commands."""
    from HoloLoom.voice.ar_context import create_test_context

    router = CommandRouter()
    context = create_test_context()

    # Test commands
    test_commands = [
        "Show me the health status of this hive",
        "What is that hive?",
        "Go to the hive on my left",
        "Start inspection",
        "This one",
        "Explain why the population is low",
        "Tell me about hive 003"
    ]

    print("Testing Command Router")
    print("=" * 60)

    for cmd in test_commands:
        print(f"\nCommand: '{cmd}'")

        # Parse intent
        intent = await router.parse_intent(cmd, context)
        print(f"Intent: {intent}")

        # Route to action
        action = await router.route_to_action(intent, context)
        print(f"Action: {action}")
        print(f"Voice: \"{action.voice_response}\"")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_command_router())
