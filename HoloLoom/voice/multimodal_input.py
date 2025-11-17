"""
Multimodal Input Processor
============================
Combines gesture and voice inputs for natural AR interaction.

This module fuses voice and gesture modalities to create seamless,
context-aware commands for Elle AR assistant.

Author: HoloLoom Team (Agent M)
Date: November 17, 2025
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import time
import numpy as np

from HoloLoom.voice.gesture_recognition import GestureRecognizer, Gesture, GestureType
from HoloLoom.voice.gesture_mapper import ContextAwareGestureMapper, GestureCommand
from HoloLoom.voice.ar_context import ARContext
from HoloLoom.voice.command_router import CommandRouter, Intent, IntentType


# ============================================================================
# Multimodal Input Types
# ============================================================================

@dataclass
class MultimodalInput:
    """Combined gesture + voice input."""
    voice_transcript: Optional[str] = None
    gestures: List[Gesture] = field(default_factory=list)
    ar_context: Optional[ARContext] = None
    timestamp: float = field(default_factory=time.time)
    camera_frame: Optional[np.ndarray] = None

    def has_voice(self) -> bool:
        """Check if voice input is present."""
        return self.voice_transcript is not None and len(self.voice_transcript) > 0

    def has_gesture(self) -> bool:
        """Check if gesture input is present."""
        return len(self.gestures) > 0

    def is_multimodal(self) -> bool:
        """Check if input contains both voice and gesture."""
        return self.has_voice() and self.has_gesture()

    def __str__(self) -> str:
        modalities = []
        if self.has_voice():
            modalities.append(f"voice='{self.voice_transcript[:20]}...'")
        if self.has_gesture():
            modalities.append(f"gestures={len(self.gestures)}")

        return f"MultimodalInput({', '.join(modalities)})"


@dataclass
class MultimodalCommand:
    """Resolved multimodal command."""
    command: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    modalities: List[str] = field(default_factory=list)  # ["voice", "gesture"]
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    fusion_strategy: str = "none"  # "voice_only", "gesture_only", "complementary", "disambiguated"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "command": self.command,
            "parameters": self.parameters,
            "modalities": self.modalities,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "fusion_strategy": self.fusion_strategy
        }

    def __str__(self) -> str:
        return (
            f"MultimodalCommand({self.command}, "
            f"modalities={self.modalities}, "
            f"conf={self.confidence:.2f}, "
            f"fusion={self.fusion_strategy})"
        )


# ============================================================================
# Fusion Strategies
# ============================================================================

class FusionStrategy(Enum):
    """Strategies for fusing multimodal inputs."""
    VOICE_ONLY = "voice_only"              # Use voice only
    GESTURE_ONLY = "gesture_only"          # Use gesture only
    COMPLEMENTARY = "complementary"        # Voice provides action, gesture provides target
    DISAMBIGUATED = "disambiguated"        # Gesture disambiguates ambiguous voice
    REINFORCED = "reinforced"              # Both modalities agree, higher confidence
    CONFLICTING = "conflicting"            # Modalities conflict, need resolution


# ============================================================================
# Multimodal Input Processor
# ============================================================================

class MultimodalInputProcessor:
    """Process combined gesture and voice inputs."""

    def __init__(
        self,
        gesture_recognizer: Optional[GestureRecognizer] = None,
        gesture_mapper: Optional[ContextAwareGestureMapper] = None,
        command_router: Optional[CommandRouter] = None,
        enable_gesture: bool = True,
        enable_voice: bool = True
    ):
        """
        Initialize multimodal processor.

        Args:
            gesture_recognizer: Gesture recognition engine
            gesture_mapper: Context-aware gesture mapper
            command_router: Voice command router
            enable_gesture: Enable gesture processing
            enable_voice: Enable voice processing
        """
        self.gesture_recognizer = gesture_recognizer or GestureRecognizer(use_mediapipe=True)
        self.gesture_mapper = gesture_mapper or ContextAwareGestureMapper()
        self.command_router = command_router or CommandRouter()
        self.enable_gesture = enable_gesture
        self.enable_voice = enable_voice

        # Processing history
        self.input_history: List[MultimodalInput] = []
        self.command_history: List[MultimodalCommand] = []

        # Ambiguity patterns (voice phrases that need gesture disambiguation)
        self.ambiguous_patterns = [
            "this one",
            "that one",
            "this",
            "that",
            "here",
            "there",
            "it",
            "the one"
        ]

    async def process(
        self,
        voice_transcript: Optional[str] = None,
        camera_frame: Optional[np.ndarray] = None,
        ar_context: Optional[ARContext] = None
    ) -> MultimodalCommand:
        """
        Process multimodal input.

        Args:
            voice_transcript: Transcribed voice input
            camera_frame: Camera frame for gesture recognition (H x W x 3)
            ar_context: Current AR context

        Returns:
            Resolved multimodal command

        Examples:
        - "This one" + pointing gesture → select pointed hive
        - "Show me details" + open palm → show details panel
        - "Next" + swipe right → navigate to next hive
        """
        # Recognize gestures from camera frame
        gestures = []
        if self.enable_gesture and camera_frame is not None:
            gestures = await self.gesture_recognizer.recognize(camera_frame)

        # Create multimodal input
        multimodal_input = MultimodalInput(
            voice_transcript=voice_transcript,
            gestures=gestures,
            ar_context=ar_context,
            camera_frame=camera_frame
        )

        # Store in history
        self.input_history.append(multimodal_input)
        self.input_history = self.input_history[-20:]  # Keep last 20

        # Parse voice intent
        voice_intent = None
        if self.enable_voice and voice_transcript:
            voice_intent = await self.command_router.parse_intent(
                voice_transcript,
                ar_context
            )

        # Map gestures to commands
        gesture_commands = []
        if ar_context:
            for gesture in gestures:
                gesture_command = await self.gesture_mapper.map_gesture(
                    gesture,
                    ar_context
                )
                if gesture_command:
                    gesture_commands.append(gesture_command)

        # Fuse inputs
        command = await self._fuse_inputs(
            voice_intent,
            gesture_commands,
            ar_context
        )

        # Store in history
        self.command_history.append(command)
        self.command_history = self.command_history[-20:]

        return command

    async def _fuse_inputs(
        self,
        voice_intent: Optional[Intent],
        gesture_commands: List[GestureCommand],
        ar_context: Optional[ARContext]
    ) -> MultimodalCommand:
        """
        Fuse voice and gesture inputs.

        Fusion strategies:
        1. Voice only: Use voice intent
        2. Gesture only: Use gesture command
        3. Complementary: Voice provides action, gesture provides target
        4. Disambiguated: Gesture resolves ambiguous voice
        5. Reinforced: Both agree, boost confidence
        6. Conflicting: Resolve conflict
        """
        # Case 1: Voice only (no gestures)
        if voice_intent and not gesture_commands:
            return MultimodalCommand(
                command=voice_intent.action,
                parameters={"target": voice_intent.target, **voice_intent.parameters},
                modalities=["voice"],
                confidence=voice_intent.confidence,
                fusion_strategy="voice_only"
            )

        # Case 2: Gesture only (no voice)
        if gesture_commands and not voice_intent:
            gesture_cmd = gesture_commands[0]  # Use first gesture
            return MultimodalCommand(
                command=gesture_cmd.command,
                parameters=gesture_cmd.parameters,
                modalities=["gesture"],
                confidence=gesture_cmd.confidence,
                fusion_strategy="gesture_only"
            )

        # Case 3: Both voice and gesture present (multimodal fusion)
        if voice_intent and gesture_commands:
            gesture_cmd = gesture_commands[0]  # Use primary gesture

            # Sub-case 3a: Ambiguous voice + disambiguating gesture
            if self._is_ambiguous(voice_intent):
                return await self._disambiguate_with_gesture(
                    voice_intent,
                    gesture_cmd,
                    ar_context
                )

            # Sub-case 3b: Complementary (voice action + gesture target)
            if self._is_complementary(voice_intent, gesture_cmd):
                return await self._complementary_fusion(
                    voice_intent,
                    gesture_cmd
                )

            # Sub-case 3c: Reinforced (both agree)
            if self._is_reinforced(voice_intent, gesture_cmd):
                return await self._reinforced_fusion(
                    voice_intent,
                    gesture_cmd
                )

            # Sub-case 3d: Conflicting (disagree)
            # Prefer voice for now (more explicit intent)
            return MultimodalCommand(
                command=voice_intent.action,
                parameters={"target": voice_intent.target, **voice_intent.parameters},
                modalities=["voice", "gesture"],
                confidence=voice_intent.confidence * 0.7,  # Reduce confidence due to conflict
                fusion_strategy="conflicting"
            )

        # Fallback: No input
        return MultimodalCommand(
            command="unknown",
            parameters={},
            modalities=[],
            confidence=0.0,
            fusion_strategy="none"
        )

    def _is_ambiguous(self, voice_intent: Intent) -> bool:
        """Check if voice intent is ambiguous."""
        if voice_intent.raw_text:
            text_lower = voice_intent.raw_text.lower()
            return any(pattern in text_lower for pattern in self.ambiguous_patterns)
        return False

    async def _disambiguate_with_gesture(
        self,
        voice_intent: Intent,
        gesture_cmd: GestureCommand,
        ar_context: Optional[ARContext]
    ) -> MultimodalCommand:
        """
        Disambiguate ambiguous voice with gesture.

        Example:
        - Voice: "this one" (ambiguous)
        - Gesture: pointing at hive_002 (specific)
        - Result: select_hive(hive_002)
        """
        # Use voice action with gesture target
        parameters = {**gesture_cmd.parameters}

        # Determine command from voice intent type
        command = voice_intent.action

        # If voice says "this one" and gesture points, use "select_object"
        if voice_intent.type == IntentType.SELECT and gesture_cmd.gesture_type == GestureType.POINT:
            command = "select_object"

        return MultimodalCommand(
            command=command,
            parameters=parameters,
            modalities=["voice", "gesture"],
            confidence=min(voice_intent.confidence, gesture_cmd.confidence),
            fusion_strategy="disambiguated"
        )

    def _is_complementary(self, voice_intent: Intent, gesture_cmd: GestureCommand) -> bool:
        """Check if voice and gesture are complementary."""
        # Voice provides action, gesture provides target
        # Example: "show details" (voice) + pointing (gesture)

        # Check if voice has action but no clear target
        voice_has_action = voice_intent.action and voice_intent.action != "unknown"
        voice_lacks_target = not voice_intent.target or voice_intent.target in self.ambiguous_patterns

        # Check if gesture provides target
        gesture_has_target = "object_id" in gesture_cmd.parameters or "target_position" in gesture_cmd.parameters

        return voice_has_action and voice_lacks_target and gesture_has_target

    async def _complementary_fusion(
        self,
        voice_intent: Intent,
        gesture_cmd: GestureCommand
    ) -> MultimodalCommand:
        """
        Fuse complementary voice and gesture.

        Example:
        - Voice: "show details" (action)
        - Gesture: pointing at hive_002 (target)
        - Result: show_details(hive_002)
        """
        # Use voice action with gesture parameters
        parameters = {**gesture_cmd.parameters, **voice_intent.parameters}

        return MultimodalCommand(
            command=voice_intent.action,
            parameters=parameters,
            modalities=["voice", "gesture"],
            confidence=min(voice_intent.confidence, gesture_cmd.confidence) * 1.1,  # Boost for complementary
            fusion_strategy="complementary"
        )

    def _is_reinforced(self, voice_intent: Intent, gesture_cmd: GestureCommand) -> bool:
        """Check if voice and gesture reinforce each other."""
        # Both modalities agree on the same command
        # Example: "next hive" (voice) + swipe right (gesture)

        # Mapping of voice actions to gesture commands
        reinforcement_map = {
            "navigate_next": ["navigate_next"],
            "navigate_previous": ["navigate_previous"],
            "select_object": ["select_object"],
            "show_details": ["show_details"],
            "start_inspection": ["start_inspection"]
        }

        voice_action = voice_intent.action
        gesture_command = gesture_cmd.command

        if voice_action in reinforcement_map:
            return gesture_command in reinforcement_map[voice_action]

        return False

    async def _reinforced_fusion(
        self,
        voice_intent: Intent,
        gesture_cmd: GestureCommand
    ) -> MultimodalCommand:
        """
        Fuse reinforced voice and gesture.

        Example:
        - Voice: "next hive"
        - Gesture: swipe right
        - Result: navigate_next (high confidence)
        """
        # Combine parameters
        parameters = {**voice_intent.parameters, **gesture_cmd.parameters}

        # Boost confidence (both modalities agree)
        confidence = min(voice_intent.confidence, gesture_cmd.confidence) * 1.2
        confidence = min(confidence, 1.0)  # Cap at 1.0

        return MultimodalCommand(
            command=voice_intent.action,
            parameters=parameters,
            modalities=["voice", "gesture"],
            confidence=confidence,
            fusion_strategy="reinforced"
        )

    def get_input_history(self, limit: int = 10) -> List[MultimodalInput]:
        """Get recent input history."""
        return self.input_history[-limit:]

    def get_command_history(self, limit: int = 10) -> List[MultimodalCommand]:
        """Get recent command history."""
        return self.command_history[-limit:]

    def clear_history(self) -> None:
        """Clear input and command history."""
        self.input_history.clear()
        self.command_history.clear()

    async def close(self) -> None:
        """Release resources."""
        if self.gesture_recognizer:
            self.gesture_recognizer.close()


# ============================================================================
# Helper Functions
# ============================================================================

async def create_multimodal_processor(
    enable_gesture: bool = True,
    enable_voice: bool = True,
    use_mediapipe: bool = True
) -> MultimodalInputProcessor:
    """
    Create a multimodal input processor with default configuration.

    Args:
        enable_gesture: Enable gesture processing
        enable_voice: Enable voice processing
        use_mediapipe: Use MediaPipe for gesture recognition

    Returns:
        Configured multimodal processor
    """
    gesture_recognizer = None
    if enable_gesture:
        gesture_recognizer = GestureRecognizer(use_mediapipe=use_mediapipe)

    gesture_mapper = ContextAwareGestureMapper()
    command_router = CommandRouter()

    return MultimodalInputProcessor(
        gesture_recognizer=gesture_recognizer,
        gesture_mapper=gesture_mapper,
        command_router=command_router,
        enable_gesture=enable_gesture,
        enable_voice=enable_voice
    )


if __name__ == "__main__":
    # Example usage
    print("Multimodal Input Processor Demo")
    print("=" * 50)

    import asyncio
    from HoloLoom.voice.ar_context import create_test_context

    async def demo():
        # Create test context
        context = create_test_context()
        print(f"\nAR Context: {context}")

        # Create processor
        processor = await create_multimodal_processor(
            enable_gesture=True,
            enable_voice=True,
            use_mediapipe=False  # Use simplified mode for demo
        )

        # Test 1: Voice only
        print("\n" + "=" * 50)
        print("Test 1: Voice only - 'show me the hive details'")
        command = await processor.process(
            voice_transcript="show me the hive details",
            ar_context=context
        )
        print(f"  Result: {command}")
        print(f"  Fusion: {command.fusion_strategy}")

        # Test 2: Ambiguous voice + gesture (simulated)
        print("\n" + "=" * 50)
        print("Test 2: Ambiguous voice - 'this one'")
        command = await processor.process(
            voice_transcript="this one",
            ar_context=context
        )
        print(f"  Result: {command}")
        print(f"  Fusion: {command.fusion_strategy}")

        # Test 3: Complementary
        print("\n" + "=" * 50)
        print("Test 3: Complementary - 'show details' + pointing")
        command = await processor.process(
            voice_transcript="show details",
            ar_context=context
        )
        print(f"  Result: {command}")
        print(f"  Fusion: {command.fusion_strategy}")

        # Show history
        print("\n" + "=" * 50)
        print(f"Command history ({len(processor.command_history)} commands):")
        for i, cmd in enumerate(processor.get_command_history()):
            print(f"  {i+1}. {cmd.command} ({cmd.fusion_strategy}, conf={cmd.confidence:.2f})")

        await processor.close()

    asyncio.run(demo())
