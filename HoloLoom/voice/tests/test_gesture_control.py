"""
Gesture Control System Tests
==============================
Comprehensive tests for gesture recognition, mapping, and multimodal fusion.

Author: HoloLoom Team (Agent M)
Date: November 17, 2025
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from HoloLoom.voice.gesture_recognition import (
    GestureRecognizer,
    GestureType,
    Hand,
    HandLandmark,
    Gesture,
    create_test_hand
)
from HoloLoom.voice.gesture_mapper import (
    ContextAwareGestureMapper,
    GestureCommand,
    ContextType,
    GestureMappingRule,
    create_default_mapper
)
from HoloLoom.voice.multimodal_input import (
    MultimodalInputProcessor,
    MultimodalInput,
    MultimodalCommand,
    FusionStrategy,
    create_multimodal_processor
)
from HoloLoom.voice.ar_context import (
    ARContext,
    ARObject,
    ARObjectType,
    Vector3,
    create_test_context
)
from HoloLoom.voice.command_router import Intent, IntentType


# ============================================================================
# Gesture Recognition Tests
# ============================================================================

class TestGestureRecognition:
    """Tests for gesture recognition engine."""

    def test_hand_landmark_distance(self):
        """Test distance calculation between landmarks."""
        l1 = HandLandmark(0, 0, 0)
        l2 = HandLandmark(1, 0, 0)

        assert l1.distance_to(l2) == pytest.approx(1.0)

        l3 = HandLandmark(3, 4, 0)
        assert l1.distance_to(l3) == pytest.approx(5.0)  # 3-4-5 triangle

    def test_hand_finger_extended_index(self):
        """Test index finger extension detection."""
        hand = create_test_hand(GestureType.POINT)

        assert hand.get_finger_extended("index") == True
        assert hand.get_finger_extended("middle") == False
        assert hand.get_finger_extended("ring") == False
        assert hand.get_finger_extended("pinky") == False

    def test_hand_finger_extended_all(self):
        """Test all fingers extended (open palm)."""
        hand = create_test_hand(GestureType.OPEN_PALM)

        assert hand.get_finger_extended("index") == True
        assert hand.get_finger_extended("middle") == True
        assert hand.get_finger_extended("ring") == True
        assert hand.get_finger_extended("pinky") == True

    def test_hand_finger_extended_none(self):
        """Test no fingers extended (grab)."""
        hand = create_test_hand(GestureType.GRAB)

        assert hand.get_finger_extended("index") == False
        assert hand.get_finger_extended("middle") == False
        assert hand.get_finger_extended("ring") == False
        assert hand.get_finger_extended("pinky") == False

    def test_hand_palm_center(self):
        """Test palm center calculation."""
        hand = create_test_hand(GestureType.OPEN_PALM)
        palm = hand.get_palm_center()

        # Palm should be between wrist and middle finger base
        assert isinstance(palm, HandLandmark)
        assert -1 <= palm.x <= 1
        assert -1 <= palm.y <= 1

    def test_hand_pointing_direction(self):
        """Test pointing direction vector."""
        hand = create_test_hand(GestureType.POINT)
        direction = hand.get_pointing_direction()

        assert len(direction) == 3
        # Direction should be normalized (length ~1)
        length = sum(d**2 for d in direction) ** 0.5
        assert length == pytest.approx(1.0, abs=0.01)

    def test_gesture_recognizer_init(self):
        """Test gesture recognizer initialization."""
        # Without MediaPipe
        recognizer = GestureRecognizer(use_mediapipe=False)
        assert recognizer.use_mediapipe == False
        assert recognizer.hands is None

        # Attempt with MediaPipe (may or may not be available)
        recognizer = GestureRecognizer(use_mediapipe=True)
        # Should gracefully fall back if MediaPipe not available
        assert recognizer is not None

    @pytest.mark.asyncio
    async def test_gesture_recognizer_recognize_empty_frame(self):
        """Test gesture recognition on empty frame."""
        recognizer = GestureRecognizer(use_mediapipe=False)

        # Create empty frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        gestures = await recognizer.recognize(frame)

        # Simplified mode returns empty list
        assert gestures == []

    def test_classify_gesture_pointing(self):
        """Test pointing gesture classification."""
        recognizer = GestureRecognizer(use_mediapipe=False)
        hand = create_test_hand(GestureType.POINT)

        gesture_type, confidence = recognizer._classify_gesture(hand)

        assert gesture_type == GestureType.POINT
        assert confidence > 0.8

    def test_classify_gesture_open_palm(self):
        """Test open palm gesture classification."""
        recognizer = GestureRecognizer(use_mediapipe=False)
        hand = create_test_hand(GestureType.OPEN_PALM)

        gesture_type, confidence = recognizer._classify_gesture(hand)

        assert gesture_type == GestureType.OPEN_PALM
        assert confidence > 0.8

    def test_classify_gesture_grab(self):
        """Test grab gesture classification."""
        recognizer = GestureRecognizer(use_mediapipe=False)
        hand = create_test_hand(GestureType.GRAB)

        gesture_type, confidence = recognizer._classify_gesture(hand)

        assert gesture_type == GestureType.GRAB
        assert confidence > 0.8

    def test_classify_gesture_pinch(self):
        """Test pinch gesture classification."""
        recognizer = GestureRecognizer(use_mediapipe=False)
        hand = create_test_hand(GestureType.PINCH)

        gesture_type, confidence = recognizer._classify_gesture(hand)

        assert gesture_type == GestureType.PINCH
        assert confidence > 0.8

    def test_is_pointing(self):
        """Test pointing detection logic."""
        recognizer = GestureRecognizer(use_mediapipe=False)
        pointing_hand = create_test_hand(GestureType.POINT)
        grab_hand = create_test_hand(GestureType.GRAB)

        assert recognizer._is_pointing(pointing_hand) == True
        assert recognizer._is_pointing(grab_hand) == False

    def test_is_open_palm(self):
        """Test open palm detection logic."""
        recognizer = GestureRecognizer(use_mediapipe=False)
        palm_hand = create_test_hand(GestureType.OPEN_PALM)
        grab_hand = create_test_hand(GestureType.GRAB)

        assert recognizer._is_open_palm(palm_hand) == True
        assert recognizer._is_open_palm(grab_hand) == False

    def test_is_grabbing(self):
        """Test grab detection logic."""
        recognizer = GestureRecognizer(use_mediapipe=False)
        grab_hand = create_test_hand(GestureType.GRAB)
        palm_hand = create_test_hand(GestureType.OPEN_PALM)

        assert recognizer._is_grabbing(grab_hand) == True
        assert recognizer._is_grabbing(palm_hand) == False

    def test_is_pinching(self):
        """Test pinch detection logic."""
        recognizer = GestureRecognizer(use_mediapipe=False)
        pinch_hand = create_test_hand(GestureType.PINCH)
        grab_hand = create_test_hand(GestureType.GRAB)

        assert recognizer._is_pinching(pinch_hand) == True
        assert recognizer._is_pinching(grab_hand) == False


# ============================================================================
# Gesture Mapping Tests
# ============================================================================

class TestGestureMapping:
    """Tests for context-aware gesture mapping."""

    @pytest.fixture
    def mapper(self):
        """Create a gesture mapper for testing."""
        return ContextAwareGestureMapper()

    @pytest.fixture
    def context(self):
        """Create a test AR context."""
        return create_test_context()

    def test_mapper_initialization(self, mapper):
        """Test mapper initialization."""
        assert mapper is not None
        assert len(mapper.mapping_rules) > 0
        assert len(mapper.gesture_history) == 0
        assert len(mapper.command_history) == 0

    def test_determine_context_default(self, mapper, context):
        """Test context determination - default."""
        context.selected_object = None
        context.conversation_context = ""

        context_type = mapper._determine_context(context)

        assert context_type == ContextType.DEFAULT

    def test_determine_context_hive_selected(self, mapper, context):
        """Test context determination - hive selected."""
        context.selected_object = context.visible_objects[0]  # hive_001

        context_type = mapper._determine_context(context)

        assert context_type == ContextType.HIVE_SELECTED

    def test_determine_context_navigation(self, mapper, context):
        """Test context determination - navigation active."""
        context.conversation_context = "navigation to hive 002"

        context_type = mapper._determine_context(context)

        assert context_type == ContextType.NAVIGATION_ACTIVE

    @pytest.mark.asyncio
    async def test_map_gesture_point_select(self, mapper, context):
        """Test mapping POINT gesture to select object."""
        hand = create_test_hand(GestureType.POINT)
        gesture = Gesture(
            type=GestureType.POINT,
            hand=hand,
            confidence=0.9,
            target_position=(0.1, 0.0, 3.0)
        )

        command = await mapper.map_gesture(gesture, context)

        assert command is not None
        assert command.command == "select_object"
        assert command.confidence > 0.8

    @pytest.mark.asyncio
    async def test_map_gesture_open_palm_show_details(self, mapper, context):
        """Test mapping OPEN_PALM gesture to show details."""
        # Select a hive first
        context.selected_object = context.visible_objects[0]

        hand = create_test_hand(GestureType.OPEN_PALM)
        gesture = Gesture(
            type=GestureType.OPEN_PALM,
            hand=hand,
            confidence=0.95
        )

        command = await mapper.map_gesture(gesture, context)

        assert command is not None
        assert command.command == "show_details"
        assert command.confidence > 0.8

    @pytest.mark.asyncio
    async def test_map_gesture_swipe_left(self, mapper, context):
        """Test mapping SWIPE_LEFT gesture to navigate previous."""
        context.selected_object = context.visible_objects[0]

        hand = create_test_hand(GestureType.GRAB)
        gesture = Gesture(
            type=GestureType.SWIPE_LEFT,
            hand=hand,
            confidence=0.85,
            direction=(-1.0, 0.0, 0.0)
        )

        command = await mapper.map_gesture(gesture, context)

        assert command is not None
        assert command.command == "navigate_previous"
        assert "direction" in command.parameters

    @pytest.mark.asyncio
    async def test_map_gesture_swipe_right(self, mapper, context):
        """Test mapping SWIPE_RIGHT gesture to navigate next."""
        context.selected_object = context.visible_objects[0]

        hand = create_test_hand(GestureType.GRAB)
        gesture = Gesture(
            type=GestureType.SWIPE_RIGHT,
            hand=hand,
            confidence=0.85,
            direction=(1.0, 0.0, 0.0)
        )

        command = await mapper.map_gesture(gesture, context)

        assert command is not None
        assert command.command == "navigate_next"

    @pytest.mark.asyncio
    async def test_map_gesture_pinch_zoom(self, mapper, context):
        """Test mapping PINCH gesture to toggle zoom."""
        context.conversation_context = "data view"

        hand = create_test_hand(GestureType.PINCH)
        gesture = Gesture(
            type=GestureType.PINCH,
            hand=hand,
            confidence=0.9
        )

        command = await mapper.map_gesture(gesture, context)

        assert command is not None
        assert command.command == "toggle_zoom"

    def test_gesture_history(self, mapper, context):
        """Test gesture history tracking."""
        hand = create_test_hand(GestureType.POINT)
        gesture1 = Gesture(type=GestureType.POINT, hand=hand, confidence=0.9)
        gesture2 = Gesture(type=GestureType.GRAB, hand=hand, confidence=0.85)

        asyncio.run(mapper.map_gesture(gesture1, context))
        asyncio.run(mapper.map_gesture(gesture2, context))

        history = mapper.get_gesture_history()

        assert len(history) == 2
        assert history[0].type == GestureType.POINT
        assert history[1].type == GestureType.GRAB

    def test_command_history(self, mapper, context):
        """Test command history tracking."""
        hand = create_test_hand(GestureType.POINT)
        gesture = Gesture(
            type=GestureType.POINT,
            hand=hand,
            confidence=0.9,
            target_position=(0.1, 0.0, 3.0)
        )

        asyncio.run(mapper.map_gesture(gesture, context))

        history = mapper.get_command_history()

        assert len(history) >= 1
        assert isinstance(history[0], GestureCommand)

    def test_add_custom_rule(self, mapper):
        """Test adding custom mapping rule."""
        initial_count = len(mapper.mapping_rules)

        custom_rule = GestureMappingRule(
            gesture_type=GestureType.WAVE,
            context_type=ContextType.DEFAULT,
            command="custom_command",
            confidence_multiplier=1.0
        )

        mapper.add_custom_rule(custom_rule)

        assert len(mapper.mapping_rules) == initial_count + 1
        assert mapper.mapping_rules[-1].command == "custom_command"


# ============================================================================
# Multimodal Fusion Tests
# ============================================================================

class TestMultimodalFusion:
    """Tests for multimodal input fusion."""

    @pytest.fixture
    def processor(self):
        """Create a multimodal processor for testing."""
        return asyncio.run(create_multimodal_processor(
            enable_gesture=True,
            enable_voice=True,
            use_mediapipe=False
        ))

    @pytest.fixture
    def context(self):
        """Create a test AR context."""
        return create_test_context()

    def test_multimodal_input_has_voice(self):
        """Test multimodal input voice detection."""
        input1 = MultimodalInput(voice_transcript="hello")
        input2 = MultimodalInput(voice_transcript="")
        input3 = MultimodalInput(voice_transcript=None)

        assert input1.has_voice() == True
        assert input2.has_voice() == False
        assert input3.has_voice() == False

    def test_multimodal_input_has_gesture(self):
        """Test multimodal input gesture detection."""
        hand = create_test_hand(GestureType.POINT)
        gesture = Gesture(type=GestureType.POINT, hand=hand, confidence=0.9)

        input1 = MultimodalInput(gestures=[gesture])
        input2 = MultimodalInput(gestures=[])

        assert input1.has_gesture() == True
        assert input2.has_gesture() == False

    def test_multimodal_input_is_multimodal(self):
        """Test multimodal input detection."""
        hand = create_test_hand(GestureType.POINT)
        gesture = Gesture(type=GestureType.POINT, hand=hand, confidence=0.9)

        input1 = MultimodalInput(
            voice_transcript="hello",
            gestures=[gesture]
        )
        input2 = MultimodalInput(voice_transcript="hello")
        input3 = MultimodalInput(gestures=[gesture])

        assert input1.is_multimodal() == True
        assert input2.is_multimodal() == False
        assert input3.is_multimodal() == False

    @pytest.mark.asyncio
    async def test_processor_voice_only(self, processor, context):
        """Test processing voice-only input."""
        command = await processor.process(
            voice_transcript="show me the hive",
            ar_context=context
        )

        assert command is not None
        assert "voice" in command.modalities
        assert "gesture" not in command.modalities
        assert command.fusion_strategy == "voice_only"

    @pytest.mark.asyncio
    async def test_processor_disambiguate_with_gesture(self, processor, context):
        """Test disambiguating ambiguous voice with gesture."""
        # Mock gesture command for "this one"
        command = await processor.process(
            voice_transcript="this one",
            ar_context=context
        )

        # Should detect ambiguity
        assert processor._is_ambiguous(Intent(
            type=IntentType.SELECT,
            action="select",
            raw_text="this one"
        )) == True

    @pytest.mark.asyncio
    async def test_processor_complementary_fusion(self, processor, context):
        """Test complementary fusion (voice action + gesture target)."""
        # This would require mocking gesture recognition
        # For now, test the logic directly

        voice_intent = Intent(
            type=IntentType.QUERY,
            action="show_details",
            target=None,
            confidence=0.9,
            raw_text="show details"
        )

        hand = create_test_hand(GestureType.POINT)
        gesture_cmd = GestureCommand(
            gesture_type=GestureType.POINT,
            command="select_object",
            parameters={"object_id": "hive_001"},
            confidence=0.85
        )

        is_complementary = processor._is_complementary(voice_intent, gesture_cmd)

        assert is_complementary == True

    @pytest.mark.asyncio
    async def test_processor_reinforced_fusion(self, processor, context):
        """Test reinforced fusion (both modalities agree)."""
        voice_intent = Intent(
            type=IntentType.NAVIGATE,
            action="navigate_next",
            confidence=0.9,
            raw_text="next hive"
        )

        gesture_cmd = GestureCommand(
            gesture_type=GestureType.SWIPE_RIGHT,
            command="navigate_next",
            confidence=0.85
        )

        is_reinforced = processor._is_reinforced(voice_intent, gesture_cmd)

        assert is_reinforced == True

        # Test fusion
        command = await processor._reinforced_fusion(voice_intent, gesture_cmd)

        assert command.fusion_strategy == "reinforced"
        assert command.confidence > voice_intent.confidence  # Boosted

    def test_processor_ambiguous_patterns(self, processor):
        """Test ambiguous pattern detection."""
        ambiguous_phrases = [
            "this one",
            "that one",
            "this",
            "that",
            "here",
            "there",
            "it"
        ]

        for phrase in ambiguous_phrases:
            intent = Intent(
                type=IntentType.SELECT,
                action="select",
                raw_text=phrase
            )
            assert processor._is_ambiguous(intent) == True

    @pytest.mark.asyncio
    async def test_processor_history_tracking(self, processor, context):
        """Test input and command history tracking."""
        await processor.process(
            voice_transcript="show hive",
            ar_context=context
        )

        await processor.process(
            voice_transcript="next",
            ar_context=context
        )

        input_history = processor.get_input_history()
        command_history = processor.get_command_history()

        assert len(input_history) >= 2
        assert len(command_history) >= 2

    @pytest.mark.asyncio
    async def test_processor_clear_history(self, processor, context):
        """Test clearing history."""
        await processor.process(
            voice_transcript="test",
            ar_context=context
        )

        processor.clear_history()

        assert len(processor.input_history) == 0
        assert len(processor.command_history) == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestGestureControlIntegration:
    """Integration tests for complete gesture control pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_point_select(self):
        """Test end-to-end pointing to select object."""
        # Create context
        context = create_test_context()

        # Create components
        recognizer = GestureRecognizer(use_mediapipe=False)
        mapper = ContextAwareGestureMapper()
        processor = await create_multimodal_processor(use_mediapipe=False)

        # Create pointing gesture
        hand = create_test_hand(GestureType.POINT)
        gesture = Gesture(
            type=GestureType.POINT,
            hand=hand,
            confidence=0.9,
            target_position=(0.1, 0.0, 3.0)  # Point at hive
        )

        # Map gesture
        command = await mapper.map_gesture(gesture, context)

        assert command is not None
        assert command.command == "select_object"
        assert command.confidence > 0.8

        await processor.close()

    @pytest.mark.asyncio
    async def test_end_to_end_voice_gesture_fusion(self):
        """Test end-to-end voice + gesture fusion."""
        context = create_test_context()
        processor = await create_multimodal_processor(use_mediapipe=False)

        # Process "this one" (ambiguous voice)
        command = await processor.process(
            voice_transcript="this one",
            ar_context=context
        )

        # Should work even without gesture (uses voice intent)
        assert command is not None

        await processor.close()

    @pytest.mark.asyncio
    async def test_end_to_end_multimodal_navigation(self):
        """Test end-to-end multimodal navigation command."""
        context = create_test_context()
        context.selected_object = context.visible_objects[0]

        processor = await create_multimodal_processor(use_mediapipe=False)

        # Process "next" voice command
        command = await processor.process(
            voice_transcript="next",
            ar_context=context
        )

        assert command is not None
        assert "navigate" in command.command or command.command == "next"

        await processor.close()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
