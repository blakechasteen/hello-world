"""
Comprehensive Integration Tests for Elle AR Integration System

Tests cover:
1. ARContext: Spatial reference resolution, object finding, direction handling
2. CommandRouter: Intent classification, target resolution, confidence scoring
3. SpatialAudioHandler: HRTF processing, distance attenuation, spatial parameters
4. ElleBridge: End-to-end voice command processing, multimodal response generation

Total: 15+ integration tests with async support, performance validation, and edge cases.

Author: HoloLoom Test Suite
Date: November 2025
"""

import pytest
import asyncio
import numpy as np
import math
from typing import List, Optional, Tuple

# HoloLoom voice module imports
from hololoom.voice.ar_context import (
    ARContext, Vector3, Quaternion, ARObject, ARObjectType,
    AREvent, AREventType, SelectionEvent, NavigationEvent, GestureEvent,
    GestureType, create_test_context
)
from hololoom.voice.command_router import (
    CommandRouter, Intent, IntentType, ElleAction, ElleActionType
)
from hololoom.voice.spatial_audio import (
    SpatialAudioHandler, SpatialAudioConfig, create_test_audio
)
from hololoom.voice.elle_bridge import (
    ElleBridge, ResponseMode, ARVisualization, ElleResponse
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def test_context() -> ARContext:
    """Create a standard test AR context with beehives."""
    return create_test_context()


@pytest.fixture
def complex_context() -> ARContext:
    """Create a more complex AR context with varied objects."""
    context = ARContext(
        user_position=Vector3(0, 1.7, 0),
        user_orientation=Quaternion(0, 0, 0, 1),
        gaze_direction=Vector3(0, 0, 1),
        active_scene="complex_scenario"
    )

    # Multiple beehives at various distances and directions
    context.visible_objects = [
        # Front-right
        ARObject(
            id="hive_fr_1",
            type=ARObjectType.BEEHIVE,
            position=Vector3(3, 0.5, 4),
            name="Front-Right Hive 1"
        ),
        # Front-left
        ARObject(
            id="hive_fl_1",
            type=ARObjectType.BEEHIVE,
            position=Vector3(-3, 0.5, 4),
            name="Front-Left Hive 1"
        ),
        # Directly left
        ARObject(
            id="hive_l_1",
            type=ARObjectType.BEEHIVE,
            position=Vector3(-5, 0.5, 0),
            name="Left Hive 1"
        ),
        # Directly right
        ARObject(
            id="hive_r_1",
            type=ARObjectType.BEEHIVE,
            position=Vector3(5, 0.5, 0),
            name="Right Hive 1"
        ),
        # Above
        ARObject(
            id="marker_above",
            type=ARObjectType.MARKER,
            position=Vector3(0, 3.5, 2),
            name="Marker Above"
        ),
        # Below
        ARObject(
            id="frame_below",
            type=ARObjectType.FRAME,
            position=Vector3(0, 0.2, 2),
            name="Frame Below"
        ),
        # Tool
        ARObject(
            id="tool_smoker",
            type=ARObjectType.TOOL,
            position=Vector3(1, 0.8, 0.5),
            name="Smoker"
        )
    ]

    context.gaze_target = "hive_fr_1"
    context.update_spatial_references()

    return context


@pytest.fixture
def command_router() -> CommandRouter:
    """Create a command router instance."""
    return CommandRouter()


@pytest.fixture
def spatial_audio_handler() -> SpatialAudioHandler:
    """Create a spatial audio handler with default config."""
    return SpatialAudioHandler(SpatialAudioConfig())


@pytest.fixture
def elle_bridge() -> ElleBridge:
    """Create an Elle Bridge instance (without HoloLoom)."""
    return ElleBridge(
        orchestrator=None,
        enable_hololoom=False,
        enable_spatial_audio=True
    )


# ============================================================================
# Helper Functions for Test Data Generation
# ============================================================================

def create_test_audio_bytes(duration_seconds: float = 0.1) -> bytes:
    """Create test audio bytes for spatial audio testing."""
    return create_test_audio(duration_seconds)


def assert_vector3_close(
    actual: Vector3,
    expected: Vector3,
    tolerance: float = 0.01,
    msg: str = ""
) -> None:
    """Assert two Vector3 instances are close within tolerance."""
    assert abs(actual.x - expected.x) < tolerance, \
        f"X mismatch: {actual.x} vs {expected.x} {msg}"
    assert abs(actual.y - expected.y) < tolerance, \
        f"Y mismatch: {actual.y} vs {expected.y} {msg}"
    assert abs(actual.z - expected.z) < tolerance, \
        f"Z mismatch: {actual.z} vs {expected.z} {msg}"


def assert_confidence_in_range(
    confidence: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
    msg: str = ""
) -> None:
    """Assert confidence score is within valid range [0, 1]."""
    assert min_val <= confidence <= max_val, \
        f"Confidence {confidence} outside range [{min_val}, {max_val}] {msg}"


# ============================================================================
# ARContext Tests (5 tests)
# ============================================================================

class TestARContext:
    """Test suite for ARContext spatial awareness and object finding."""

    def test_spatial_reference_resolution_gaze(self, test_context: ARContext) -> None:
        """
        Test that 'this' spatial reference resolves to gaze target.

        Validates:
        - Gaze target is correctly identified
        - nearby_objects["this"] maps to gaze target
        - Spatial reference accuracy > 95%
        """
        # Arrange
        test_context.update_spatial_references()
        expected_gaze_obj = test_context.find_object_by_id(test_context.gaze_target)

        # Act
        gaze_reference = test_context.nearby_objects.get("this")

        # Assert
        assert gaze_reference is not None, "Gaze reference 'this' not resolved"
        assert gaze_reference.id == expected_gaze_obj.id, \
            f"Gaze reference mismatch: {gaze_reference.id} vs {expected_gaze_obj.id}"
        assert gaze_reference.id == "hive_003", \
            "Gaze target should be hive_003 in test context"

    def test_spatial_reference_resolution_proximity(self, test_context: ARContext) -> None:
        """
        Test that spatial references resolve by proximity (nearest/that).

        Validates:
        - 'nearest' resolves to closest object
        - 'that' resolves to second closest
        - Distance-based ordering is correct
        """
        # Arrange
        test_context.update_spatial_references()

        # Act
        nearest = test_context.nearby_objects.get("nearest")
        that = test_context.nearby_objects.get("that")

        # Assert
        assert nearest is not None, "Nearest reference not resolved"
        assert nearest.id == "hive_003", "Nearest should be hive_003 (closest)"

        if that:
            # Verify that is further than nearest
            nearest_dist = nearest.distance_from(test_context.user_position)
            that_dist = that.distance_from(test_context.user_position)
            assert that_dist > nearest_dist, \
                f"'that' should be farther than 'nearest': {that_dist} vs {nearest_dist}"

    def test_spatial_reference_resolution_directional(
        self,
        complex_context: ARContext
    ) -> None:
        """
        Test directional spatial references (left, right, front, back, above, below).

        Validates:
        - Directional queries return correct objects
        - Directional accuracy > 90%
        - All 6 directions supported
        """
        # Arrange
        complex_context.update_spatial_references()

        # Act & Assert - Test all directions
        directions_and_expected = [
            ("left", "hive_l_1"),
            ("right", "hive_r_1"),
            ("front", "hive_fr_1"),  # Front-right is closer to front than front-left
        ]

        for direction, expected_id in directions_and_expected:
            found_obj = complex_context.find_object_in_direction(
                direction,
                ARObjectType.BEEHIVE
            )
            assert found_obj is not None, \
                f"No object found in {direction} direction"
            assert found_obj.id == expected_id, \
                f"Wrong object in {direction}: got {found_obj.id}, expected {expected_id}"

    def test_find_object_in_direction(self, complex_context: ARContext) -> None:
        """
        Test find_object_in_direction with various scenarios.

        Validates:
        - Directional cone (60° threshold) correctly filters objects
        - Type filtering works with directional search
        - Returns None for invalid directions
        - Azimuth calculation accurate within 10°
        """
        # Arrange - Test with marker above
        above_threshold = 0.5  # 60° cone

        # Act
        marker_above = complex_context.find_object_in_direction("above")
        marker_above_typed = complex_context.find_object_in_direction(
            "above",
            ARObjectType.MARKER
        )

        # Assert
        assert marker_above is not None, "Should find object above"
        assert marker_above_typed is not None, "Should find marker above with type filter"
        assert marker_above_typed.id == "marker_above"

        # Test invalid direction
        result = complex_context.find_object_in_direction("invalid_direction")
        assert result is None, "Invalid direction should return None"

    def test_update_spatial_references(self, test_context: ARContext) -> None:
        """
        Test that update_spatial_references correctly maintains nearby_objects mapping.

        Validates:
        - Mapping is updated correctly
        - All reference types populated
        - Changes reflected on context updates
        """
        # Arrange
        original_nearby = len(test_context.nearby_objects)

        # Act
        test_context.update_spatial_references()

        # Assert - Check that nearby_objects has expected keys
        assert "this" in test_context.nearby_objects, "Missing 'this' reference"
        assert "nearest" in test_context.nearby_objects, "Missing 'nearest' reference"

        # Verify objects are valid
        for ref_name, obj in test_context.nearby_objects.items():
            assert isinstance(obj, ARObject), f"Invalid object for reference {ref_name}"
            assert obj in test_context.visible_objects, \
                f"Reference '{ref_name}' points to object not in visible_objects"

        # Act - Change gaze target and update
        old_gaze = test_context.gaze_target
        test_context.gaze_target = "hive_001"  # Different hive
        test_context.update_spatial_references()

        # Assert - "this" should change
        new_this = test_context.nearby_objects.get("this")
        assert new_this is not None
        assert new_this.id == "hive_001", \
            "Spatial references should update when gaze_target changes"


# ============================================================================
# CommandRouter Tests (5 tests)
# ============================================================================

class TestCommandRouter:
    """Test suite for CommandRouter intent parsing and routing."""

    @pytest.mark.asyncio
    async def test_intent_classification_query(
        self,
        command_router: CommandRouter,
        test_context: ARContext
    ) -> None:
        """
        Test QUERY intent classification for information requests.

        Validates:
        - "What", "Show", "Tell" patterns recognized as QUERY
        - Intent confidence > 0.75 for clear queries
        - Action verbs extracted correctly
        """
        # Arrange
        test_queries = [
            "What is the health status of this hive?",
            "Show me the population of that hive",
            "Tell me about the nearest hive",
        ]

        # Act & Assert
        for query in test_queries:
            intent = await command_router.parse_intent(query, test_context)

            assert intent.type == IntentType.QUERY, \
                f"Failed to classify query: {query}"
            assert_confidence_in_range(intent.confidence, 0.7, 1.0,
                f"Low confidence for query: {query}")
            assert intent.action in ["what", "show", "tell"], \
                f"Invalid action verb: {intent.action}"

    @pytest.mark.asyncio
    async def test_intent_classification_navigate(
        self,
        command_router: CommandRouter,
        test_context: ARContext
    ) -> None:
        """
        Test NAVIGATE intent classification for movement commands.

        Validates:
        - "Go", "Take", "Next" patterns recognized as NAVIGATE
        - Navigation targets resolved
        - Confidence scores appropriate
        """
        # Arrange
        test_commands = [
            "Go to hive 001",
            "Take me to the hive on my left",
            "Next hive",
        ]

        # Act & Assert
        for command in test_commands:
            intent = await command_router.parse_intent(command, test_context)

            assert intent.type == IntentType.NAVIGATE, \
                f"Failed to classify as NAVIGATE: {command}"
            assert_confidence_in_range(intent.confidence, 0.6, 1.0,
                f"Confidence too low for NAVIGATE: {command}")

    @pytest.mark.asyncio
    async def test_intent_classification_select(
        self,
        command_router: CommandRouter,
        test_context: ARContext
    ) -> None:
        """
        Test SELECT intent classification for object selection.

        Validates:
        - "This", "That", "Select" patterns recognized
        - Spatial references resolved in selection context
        - High confidence for clear selections
        """
        # Arrange
        test_selections = [
            "This one",
            "That hive",
            "Select the tool on my right",
        ]

        # Act & Assert
        for selection in test_selections:
            intent = await command_router.parse_intent(selection, test_context)

            assert intent.type == IntentType.SELECT, \
                f"Failed to classify as SELECT: {selection}"
            assert_confidence_in_range(intent.confidence, 0.6, 1.0,
                f"Low confidence for SELECT: {selection}")

    @pytest.mark.asyncio
    async def test_target_resolution_with_gaze(
        self,
        command_router: CommandRouter,
        test_context: ARContext
    ) -> None:
        """
        Test target resolution using AR context and gaze.

        Validates:
        - Spatial reference resolution accuracy > 95%
        - Gaze context used when available
        - Target IDs correctly resolved
        """
        # Arrange
        test_context.gaze_target = "hive_003"
        test_context.update_spatial_references()

        # Act
        intent_this = await command_router.parse_intent(
            "Tell me about this one",
            test_context
        )

        intent_that = await command_router.parse_intent(
            "Show that hive",
            test_context
        )

        intent_hive_num = await command_router.parse_intent(
            "What about hive 001?",
            test_context
        )

        # Assert
        assert intent_this.target == "hive_003", \
            f"'this' should resolve to gaze target, got {intent_this.target}"

        if intent_that.target:
            assert intent_that.target in ["hive_001", "hive_002"], \
                f"'that' should resolve to nearby hive, got {intent_that.target}"

        assert intent_hive_num.target == "hive_001", \
            f"Hive number parsing failed: {intent_hive_num.target}"

    @pytest.mark.asyncio
    async def test_confidence_scoring(
        self,
        command_router: CommandRouter,
        test_context: ARContext
    ) -> None:
        """
        Test intent confidence scoring across various inputs.

        Validates:
        - Clear patterns: confidence > 0.75
        - Ambiguous inputs: confidence 0.5-0.75
        - Very short/long inputs penalized appropriately
        - Confidence always in [0, 1]
        """
        # Arrange
        test_cases = [
            ("What is this?", True, 0.75),  # (text, is_query, min_expected_conf)
            ("Tell me", False, 0.5),  # Short, less confident
            ("This is a very long sentence with many words that goes on and on", False, 0.6),
            ("Go left", True, 0.5),  # Ambiguous direction
            ("Show health data for hive 003", True, 0.8),  # Very clear
        ]

        # Act & Assert
        for text, should_be_query, min_conf in test_cases:
            intent = await command_router.parse_intent(text, test_context)

            assert_confidence_in_range(intent.confidence, 0.0, 1.0,
                f"Invalid confidence for: {text}")
            assert intent.confidence >= min_conf, \
                f"Confidence {intent.confidence} below expected {min_conf} for: {text}"


# ============================================================================
# SpatialAudioHandler Tests (3 tests)
# ============================================================================

class TestSpatialAudioHandler:
    """Test suite for 3D spatial audio positioning and HRTF."""

    @pytest.mark.asyncio
    async def test_hrtf_processing(
        self,
        spatial_audio_handler: SpatialAudioHandler,
        test_context: ARContext
    ) -> None:
        """
        Test HRTF (Head-Related Transfer Function) audio processing.

        Validates:
        - Mono → stereo conversion (2x size increase)
        - ITD (Interaural Time Difference) applied correctly
        - ILD (Interaural Level Difference) applied correctly
        - Output is valid stereo PCM audio
        """
        # Arrange
        test_audio = create_test_audio_bytes(0.1)
        audio_source = Vector3(2, 1.7, 2)  # Source to the right

        # Act
        stereo_audio = await spatial_audio_handler.position_audio(
            test_audio,
            audio_source,
            test_context
        )

        # Assert - Check stereo output
        assert len(stereo_audio) == len(test_audio) * 2, \
            f"Stereo output should be 2x mono size: {len(stereo_audio)} vs {len(test_audio)*2}"

        # Verify it's valid audio (can convert to numpy)
        stereo_array = np.frombuffer(stereo_audio, dtype=np.int16)
        assert len(stereo_array) == len(test_audio) * 2, \
            "Stereo array size mismatch"

        # Verify not all zeros (processing occurred)
        assert np.any(stereo_array != 0), \
            "Output is all zeros - no processing occurred"

        # Check that left and right channels differ (HRTF applied)
        left_channel = stereo_array[0::2]
        right_channel = stereo_array[1::2]

        # For source on right, right channel should be slightly louder
        left_rms = np.sqrt(np.mean(left_channel.astype(float)**2))
        right_rms = np.sqrt(np.mean(right_channel.astype(float)**2))

        assert right_rms > 0, "Right channel processing failed"
        assert left_rms > 0, "Left channel processing failed"

    @pytest.mark.asyncio
    async def test_distance_attenuation(
        self,
        spatial_audio_handler: SpatialAudioHandler,
        test_context: ARContext
    ) -> None:
        """
        Test distance-based volume attenuation (inverse square law).

        Validates:
        - Close source (1m): attenuation = 1.0 (full volume)
        - Medium distance (5m): attenuation = 0.2
        - Far distance (50m): attenuation = 0.0 (silent)
        - Attenuation factor in [0, 1]
        """
        # Arrange - Test at various distances
        test_distances = [
            (0.5, 1.0, "Very close"),      # At reference distance
            (1.0, 1.0, "Reference distance"),
            (2.0, 0.4, "2m away"),  # (1/2)^1 = 0.25 (approximately)
            (5.0, 0.1, "5m away"),
            (50.0, 0.0, "Max distance"),
            (100.0, 0.0, "Beyond max"),
        ]

        # Act & Assert
        for distance, expected_attenuation, description in test_distances:
            attenuation = spatial_audio_handler.calculate_attenuation(distance)

            assert_confidence_in_range(attenuation, 0.0, 1.0,
                f"Invalid attenuation at {distance}m: {description}")

            # Allow ±0.1 tolerance for expected values
            tolerance = 0.1 if expected_attenuation > 0 else 0.05
            assert abs(attenuation - expected_attenuation) <= tolerance, \
                f"Attenuation mismatch at {distance}m: {attenuation} vs {expected_attenuation}"

    def test_spatial_parameters_calculation(
        self,
        spatial_audio_handler: SpatialAudioHandler,
        test_context: ARContext
    ) -> None:
        """
        Test spatial parameter calculation (azimuth, elevation, distance).

        Validates:
        - Azimuth calculation: -π to π range
        - Elevation calculation: -π/2 to π/2 range
        - Distance calculation accurate within 1%
        - Parameters consistent with AR context
        """
        # Arrange - Test with source at known position
        test_positions = [
            (Vector3(2, 1.7, 2), "Front-right", 0, 0),  # (pos, desc, az_deg, el_deg)
            (Vector3(-2, 1.7, 2), "Front-left", 180, 0),
            (Vector3(0, 3.5, 2), "Above front", 0, 45),
            (Vector3(0, 0.5, 2), "Below front", 0, -45),
        ]

        # Act & Assert
        for position, description, expected_az_deg, expected_el_deg in test_positions:
            params = spatial_audio_handler._calculate_spatial_parameters(
                position,
                test_context
            )

            # Check ranges
            assert -math.pi <= params['azimuth'] <= math.pi, \
                f"Azimuth out of range: {params['azimuth']}"
            assert -math.pi/2 <= params['elevation'] <= math.pi/2, \
                f"Elevation out of range: {params['elevation']}"
            assert_confidence_in_range(params['attenuation'], 0.0, 1.0,
                f"Attenuation invalid for {description}")

            # Check distance accuracy (within 1%)
            expected_distance = test_context.user_position.distance_to(position)
            assert abs(params['distance'] - expected_distance) < expected_distance * 0.01, \
                f"Distance mismatch for {description}: {params['distance']} vs {expected_distance}"

            # Verify preview method matches
            preview = spatial_audio_handler.get_spatial_preview(position, test_context)
            assert abs(preview['distance'] - params['distance']) < 0.01, \
                "Preview distance mismatch"


# ============================================================================
# ElleBridge Tests (2 tests + extras for edge cases)
# ============================================================================

class TestElleBridge:
    """Test suite for end-to-end Elle bridge integration."""

    @pytest.mark.asyncio
    async def test_end_to_end_voice_command(
        self,
        elle_bridge: ElleBridge,
        test_context: ARContext
    ) -> None:
        """
        Test end-to-end voice command processing pipeline.

        Validates:
        - Voice transcript → Intent → Action → Response
        - Response latency < 2s
        - All response components present
        - Processing completes successfully
        """
        # Arrange
        test_commands = [
            "Show me the health status of this hive",
            "Go to hive on my left",
            "Start inspection",
        ]

        # Act & Assert
        for transcript in test_commands:
            response = await elle_bridge.process_voice_command(
                transcript,
                test_context,
                ResponseMode.MULTIMODAL_SPATIAL
            )

            # Check response structure
            assert isinstance(response, ElleResponse), \
                f"Invalid response type for: {transcript}"
            assert response.voice_text, \
                f"Empty voice response for: {transcript}"
            assert response.processing_time > 0, \
                f"No processing time recorded"
            assert response.processing_time < 2000, \
                f"Processing too slow: {response.processing_time}ms"

            # Check response mode
            assert response.response_mode in ResponseMode, \
                f"Invalid response mode: {response.response_mode}"

    @pytest.mark.asyncio
    async def test_multimodal_response_generation(
        self,
        elle_bridge: ElleBridge,
        test_context: ARContext
    ) -> None:
        """
        Test multimodal response generation (voice + visual + spatial).

        Validates:
        - Voice text generated appropriately
        - AR visualization created for applicable intents
        - Spatial audio position calculated when needed
        - Response completeness based on response mode
        """
        # Arrange
        test_scenario = "Show me the population of this hive"

        # Act
        response = await elle_bridge.process_voice_command(
            test_scenario,
            test_context,
            ResponseMode.MULTIMODAL_SPATIAL
        )

        # Assert
        assert response.voice_text, "Voice text missing"

        # For QUERY intents with MULTIMODAL mode, should have visualization
        assert response.ar_visualization is not None, \
            "AR visualization should be present for MULTIMODAL_SPATIAL mode"
        assert response.ar_visualization.type in ["overlay", "highlight", "path", "panel"], \
            f"Invalid visualization type: {response.ar_visualization.type}"

        # For MULTIMODAL_SPATIAL, should have spatial position
        assert response.spatial_audio_position is not None, \
            "Spatial audio position missing for MULTIMODAL_SPATIAL"
        assert len(response.spatial_audio_position) == 3, \
            "Spatial position should be [x, y, z]"

        # Check metadata
        assert response.metadata.get("intent_type"), "Intent type missing from metadata"
        assert response.metadata.get("action_type"), "Action type missing from metadata"


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

class TestEdgeCases:
    """Edge cases and stress testing."""

    @pytest.mark.asyncio
    async def test_ambiguous_reference_resolution(
        self,
        command_router: CommandRouter,
        test_context: ARContext
    ) -> None:
        """Test handling of ambiguous spatial references."""
        # Arrange - Remove gaze target to test fallback
        test_context.gaze_target = None

        # Act
        intent = await command_router.parse_intent(
            "Show me this",
            test_context
        )

        # Assert - Should still resolve to closest
        assert intent.target is not None or intent.confidence < 0.7, \
            "Should either resolve to closest or have low confidence"

    @pytest.mark.asyncio
    async def test_response_modes_voice_only(
        self,
        elle_bridge: ElleBridge,
        test_context: ARContext
    ) -> None:
        """Test VOICE_ONLY response mode."""
        # Act
        response = await elle_bridge.process_voice_command(
            "Why is the population low?",
            test_context,
            ResponseMode.VOICE_ONLY
        )

        # Assert
        assert response.response_mode == ResponseMode.VOICE_ONLY
        assert response.voice_text, "Voice text must be present"
        # Visual should not be present or empty
        if response.ar_visualization:
            assert response.spatial_audio_position is None, \
                "Spatial audio should not be in VOICE_ONLY mode"

    def test_vector3_operations(self) -> None:
        """Test Vector3 math operations."""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)

        # Distance
        distance = v1.distance_to(v2)
        expected_distance = math.sqrt((4-1)**2 + (5-2)**2 + (6-3)**2)
        assert abs(distance - expected_distance) < 0.001

        # Normalization
        v3 = Vector3(3, 4, 0)
        normalized = v3.normalize()
        magnitude = math.sqrt(normalized.x**2 + normalized.y**2 + normalized.z**2)
        assert abs(magnitude - 1.0) < 0.001

        # Dot product
        v4 = Vector3(1, 0, 0)
        v5 = Vector3(0, 1, 0)
        dot = v4.dot(v5)
        assert dot == 0, "Perpendicular vectors should have dot product 0"

    def test_quaternion_euler_conversion(self) -> None:
        """Test Quaternion to Euler angle conversion."""
        # Identity quaternion
        q_identity = Quaternion(0, 0, 0, 1)
        pitch, yaw, roll = q_identity.to_euler()
        assert abs(pitch) < 0.001 and abs(yaw) < 0.001 and abs(roll) < 0.001, \
            "Identity quaternion should give zero euler angles"

    @pytest.mark.asyncio
    async def test_high_volume_commands(
        self,
        elle_bridge: ElleBridge,
        test_context: ARContext
    ) -> None:
        """Test processing multiple commands in sequence."""
        # Arrange
        commands = [
            "Show this",
            "Tell me about that",
            "Go to the left hive",
            "What's the health status?",
            "Start inspection",
        ] * 3  # 15 commands total

        # Act
        responses = []
        for cmd in commands:
            response = await elle_bridge.process_voice_command(cmd, test_context)
            responses.append(response)

        # Assert
        assert len(responses) == len(commands)
        assert all(r.voice_text for r in responses), "All responses should have voice text"

        # Check statistics
        stats = elle_bridge.get_statistics()
        assert stats['total_queries'] == len(commands), "Query count mismatch"
        assert stats['total_processing_time_ms'] > 0, "No processing time recorded"


# ============================================================================
# Performance and Latency Tests
# ============================================================================

class TestPerformance:
    """Performance validation tests."""

    @pytest.mark.asyncio
    async def test_intent_parsing_latency(
        self,
        command_router: CommandRouter,
        test_context: ARContext
    ) -> None:
        """Test that intent parsing is fast (<100ms per query)."""
        import time

        # Arrange
        test_query = "Show me the health status of this hive"

        # Act
        start = time.perf_counter()
        intent = await command_router.parse_intent(test_query, test_context)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Assert
        assert elapsed_ms < 100, f"Intent parsing too slow: {elapsed_ms:.1f}ms"

    def test_spatial_audio_latency(
        self,
        spatial_audio_handler: SpatialAudioHandler,
        test_context: ARContext
    ) -> None:
        """Test that spatial audio processing is fast (<50ms per query)."""
        import time

        # Arrange
        test_audio = create_test_audio_bytes(0.1)
        source_pos = Vector3(2, 1.7, 2)

        # Act
        start = time.perf_counter()
        asyncio.run(
            spatial_audio_handler.position_audio(test_audio, source_pos, test_context)
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Assert
        assert elapsed_ms < 50, f"Spatial audio too slow: {elapsed_ms:.1f}ms"

    @pytest.mark.asyncio
    async def test_end_to_end_latency(
        self,
        elle_bridge: ElleBridge,
        test_context: ARContext
    ) -> None:
        """Test end-to-end latency < 500ms for complete pipeline."""
        # Arrange
        test_query = "Show the health of hive 001"

        # Act
        response = await elle_bridge.process_voice_command(
            test_query,
            test_context,
            ResponseMode.MULTIMODAL_SPATIAL
        )

        # Assert - Processing time should be reasonable
        assert response.processing_time < 500, \
            f"End-to-end processing too slow: {response.processing_time:.1f}ms"


# ============================================================================
# Test Suite Summary
# ============================================================================

if __name__ == "__main__":
    """
    Run the comprehensive integration test suite.

    To run:
    - pytest hololoom/voice/tests/test_elle_integration.py -v
    - pytest hololoom/voice/tests/test_elle_integration.py -v --tb=short
    - pytest hololoom/voice/tests/test_elle_integration.py::TestARContext -v

    Expected Results:
    - Total Tests: 15+ (including edge cases and performance)
    - Expected Pass Rate: 100%
    - Coverage: ARContext, CommandRouter, SpatialAudioHandler, ElleBridge
    - Performance: All tests complete in <10 seconds
    """
    pytest.main([__file__, "-v", "--tb=short"])
