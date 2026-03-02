"""
Elle Bridge Module
===================
Bidirectional bridge between VoiceAgent and Elle AR systems.

This module integrates HoloLoom VoiceAgent with the Elle AR assistant,
enabling voice-controlled augmented reality interactions with spatial
awareness and multimodal responses.

Author: HoloLoom Team
Date: November 15, 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio
import time

# HoloLoom imports
try:
    from hololoom.protocols.types import Query, ModalityType
    from hololoom.weaving_orchestrator import WeavingOrchestrator
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False

# Voice module imports
from hololoom.voice.ar_context import ARContext, AREvent, AREventType
from hololoom.voice.command_router import CommandRouter, Intent, ElleAction
from hololoom.voice.spatial_audio import SpatialAudioHandler, SpatialAudioConfig


# ============================================================================
# Response Types
# ============================================================================

class ResponseMode(Enum):
    """Modes for multimodal responses."""
    VOICE_ONLY = "voice_only"                # Voice response only
    VISUAL_ONLY = "visual_only"              # AR visualization only
    MULTIMODAL = "multimodal"                # Voice + AR visualization
    SPATIAL_AUDIO = "spatial_audio"          # 3D positioned audio
    MULTIMODAL_SPATIAL = "multimodal_spatial"  # Voice + visual + 3D audio


@dataclass
class ARVisualization:
    """AR visualization data."""
    type: str = "overlay"          # "overlay", "highlight", "path", "panel"
    target_object: Optional[str] = None
    content: Dict[str, Any] = field(default_factory=dict)
    duration: float = 5.0          # Display duration in seconds
    position: Optional[List[float]] = None  # [x, y, z] position
    style: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "target_object": self.target_object,
            "content": self.content,
            "duration": self.duration,
            "position": self.position,
            "style": self.style
        }


@dataclass
class ElleResponse:
    """
    Complete response from Elle integration.

    Combines voice, visual, and spatial audio components for
    multimodal AR interactions.

    Attributes:
        voice_text: Text response to speak
        voice_audio: TTS audio bytes (if generated)
        ar_visualization: AR visualization data
        spatial_audio_position: 3D position for audio (if spatial)
        response_mode: How to present the response
        metadata: Additional response metadata
        processing_time: Time taken to generate response (ms)
    """
    voice_text: str = ""
    voice_audio: Optional[bytes] = None
    ar_visualization: Optional[ARVisualization] = None
    spatial_audio_position: Optional[List[float]] = None
    response_mode: ResponseMode = ResponseMode.MULTIMODAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "voice_text": self.voice_text,
            "has_audio": self.voice_audio is not None,
            "ar_visualization": self.ar_visualization.to_dict() if self.ar_visualization else None,
            "spatial_audio_position": self.spatial_audio_position,
            "response_mode": self.response_mode.value,
            "metadata": self.metadata,
            "processing_time": self.processing_time
        }


# ============================================================================
# Elle Bridge
# ============================================================================

class ElleBridge:
    """
    Bidirectional bridge between VoiceAgent and Elle AR systems.

    This is the main integration point that:
    - Processes voice commands with AR context
    - Routes commands through HoloLoom orchestrator
    - Generates multimodal responses (voice + visual)
    - Positions audio in 3D space
    - Handles AR events (selections, navigation, gestures)
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        enable_hololoom: bool = True,
        enable_spatial_audio: bool = True,
        default_response_mode: ResponseMode = ResponseMode.MULTIMODAL_SPATIAL
    ):
        """
        Initialize Elle Bridge.

        Args:
            orchestrator: HoloLoom WeavingOrchestrator instance
            enable_hololoom: Whether to use HoloLoom for processing
            enable_spatial_audio: Whether to use 3D spatial audio
            default_response_mode: Default mode for responses
        """
        self.orchestrator = orchestrator
        self.enable_hololoom = enable_hololoom and HOLOLOOM_AVAILABLE
        self.enable_spatial_audio = enable_spatial_audio
        self.default_response_mode = default_response_mode

        # Initialize sub-components
        self.command_router = CommandRouter()
        self.spatial_audio_handler = SpatialAudioHandler() if enable_spatial_audio else None

        # Session state
        self.active_context: Optional[ARContext] = None
        self.recent_intents: List[Intent] = []
        self.recent_responses: List[ElleResponse] = []

        # Performance tracking
        self.total_queries = 0
        self.total_processing_time = 0.0

    async def process_voice_command(
        self,
        transcript: str,
        ar_context: ARContext,
        response_mode: Optional[ResponseMode] = None
    ) -> ElleResponse:
        """
        Process voice command with AR context.

        This is the main entry point for voice interactions in AR.

        Args:
            transcript: Voice input text from transcription
            ar_context: Current AR state (user position, gaze, objects)
            response_mode: How to present the response (optional)

        Returns:
            ElleResponse with voice, visual, and spatial audio components

        Example:
            >>> bridge = ElleBridge(orchestrator)
            >>> context = create_test_context()
            >>> response = await bridge.process_voice_command(
            ...     "Show me the health status of this hive",
            ...     context
            ... )
            >>> print(response.voice_text)
            >>> # Display response.ar_visualization in AR
        """
        start_time = time.time()

        # Update active context
        self.active_context = ar_context
        ar_context.update_spatial_references()

        # Step 1: Parse intent
        intent = await self.command_router.parse_intent(transcript, ar_context)
        self.recent_intents.append(intent)
        if len(self.recent_intents) > 10:
            self.recent_intents.pop(0)

        # Step 2: Process with HoloLoom if available
        hololoom_response = None
        if self.enable_hololoom and self.orchestrator:
            hololoom_response = await self._process_with_hololoom(
                transcript,
                intent,
                ar_context
            )

        # Step 3: Route to Elle action
        elle_action = await self.command_router.route_to_action(intent, ar_context)

        # Step 4: Generate multimodal response
        response = await self._generate_response(
            intent,
            elle_action,
            ar_context,
            hololoom_response,
            response_mode or self.default_response_mode
        )

        # Update metrics
        processing_time = (time.time() - start_time) * 1000  # ms
        response.processing_time = processing_time
        self.total_queries += 1
        self.total_processing_time += processing_time

        # Store response
        self.recent_responses.append(response)
        if len(self.recent_responses) > 10:
            self.recent_responses.pop(0)

        return response

    async def handle_ar_event(
        self,
        event: AREvent,
        ar_context: ARContext
    ) -> Optional[ElleResponse]:
        """
        Handle AR events (selections, navigation, gestures).

        Args:
            event: AR event to handle
            ar_context: Current AR context

        Returns:
            Optional voice response to event
        """
        # Update active context
        self.active_context = ar_context

        # Generate response based on event type
        if event.type == AREventType.OBJECT_SELECTED:
            object_id = event.data.get("object_id")
            obj = ar_context.find_object_by_id(object_id)

            if obj:
                voice_text = f"You've selected {obj.name}. Would you like to see more information?"

                return ElleResponse(
                    voice_text=voice_text,
                    ar_visualization=ARVisualization(
                        type="highlight",
                        target_object=object_id,
                        duration=3.0,
                        style={"color": "blue", "opacity": 0.5}
                    ),
                    response_mode=ResponseMode.MULTIMODAL
                )

        elif event.type == AREventType.NAVIGATION_STARTED:
            destination = event.data.get("destination")
            distance = event.data.get("distance_remaining", 0)

            voice_text = f"Navigating to {destination}, {distance:.1f} meters away."

            return ElleResponse(
                voice_text=voice_text,
                ar_visualization=ARVisualization(
                    type="path",
                    target_object=destination,
                    content={"show_distance": True, "show_waypoints": True}
                ),
                response_mode=ResponseMode.MULTIMODAL
            )

        elif event.type == AREventType.GESTURE_DETECTED:
            gesture_type = event.data.get("gesture_type")
            confidence = event.data.get("confidence", 0)

            if confidence > 0.8:
                voice_text = f"I detected a {gesture_type} gesture. How can I help?"
                return ElleResponse(
                    voice_text=voice_text,
                    response_mode=ResponseMode.VOICE_ONLY
                )

        return None

    async def update_ar_display(
        self,
        visualization: ARVisualization
    ) -> None:
        """
        Update AR display with new content.

        This would be implemented by the Elle AR system to actually
        render the visualization.

        Args:
            visualization: AR content to display
        """
        # This is a hook for the Elle AR system
        # In production, this would call Elle's rendering engine
        pass

    async def _process_with_hololoom(
        self,
        transcript: str,
        intent: Intent,
        ar_context: ARContext
    ) -> Optional[Dict]:
        """
        Process query through HoloLoom orchestrator.

        Args:
            transcript: Voice input text
            intent: Parsed intent
            ar_context: AR context

        Returns:
            HoloLoom Spacetime response (if available)
        """
        if not self.orchestrator:
            return None

        try:
            # Create HoloLoom query with AR context
            query = Query(
                text=transcript,
                modality=ModalityType.AUDIO,
                metadata={
                    "ar_context": ar_context.to_dict(),
                    "intent": intent.to_dict(),
                    "source": "elle_ar"
                }
            )

            # Process through orchestrator
            spacetime = await self.orchestrator.weave(query)

            # Extract response
            return {
                "response_text": spacetime.response.get("text", ""),
                "confidence": spacetime.confidence,
                "tools_used": spacetime.metadata.get("tools_used", []),
                "reasoning_trace": spacetime.metadata.get("reasoning_trace", [])
            }

        except Exception as e:
            print(f"HoloLoom processing error: {e}")
            return None

    async def _generate_response(
        self,
        intent: Intent,
        elle_action: ElleAction,
        ar_context: ARContext,
        hololoom_response: Optional[Dict],
        response_mode: ResponseMode
    ) -> ElleResponse:
        """
        Generate multimodal response combining voice and visual.

        Args:
            intent: Parsed intent
            elle_action: Elle action to perform
            ar_context: AR context
            hololoom_response: HoloLoom response (if available)
            response_mode: How to present response

        Returns:
            Complete ElleResponse
        """
        # Get voice text (prefer HoloLoom if available)
        if hololoom_response and hololoom_response.get("response_text"):
            voice_text = hololoom_response["response_text"]
        else:
            voice_text = elle_action.voice_response

        # Create AR visualization
        ar_viz = None
        if response_mode in [ResponseMode.VISUAL_ONLY, ResponseMode.MULTIMODAL, ResponseMode.MULTIMODAL_SPATIAL]:
            ar_viz = self._create_ar_visualization(intent, elle_action, ar_context)

        # Determine spatial audio position
        spatial_position = None
        if response_mode in [ResponseMode.SPATIAL_AUDIO, ResponseMode.MULTIMODAL_SPATIAL]:
            spatial_position = self._get_spatial_audio_position(intent, ar_context)

        # Create response
        response = ElleResponse(
            voice_text=voice_text,
            ar_visualization=ar_viz,
            spatial_audio_position=spatial_position,
            response_mode=response_mode,
            metadata={
                "intent_type": intent.type.value,
                "intent_confidence": intent.confidence,
                "action_type": elle_action.type.value,
                "hololoom_used": hololoom_response is not None
            }
        )

        return response

    def _create_ar_visualization(
        self,
        intent: Intent,
        elle_action: ElleAction,
        ar_context: ARContext
    ) -> Optional[ARVisualization]:
        """Create AR visualization based on intent and action."""
        target_obj = None
        if intent.target:
            target_obj = ar_context.find_object_by_id(intent.target)

        # Query intents → info panel overlay
        if intent.type.value == "query":
            return ARVisualization(
                type="overlay",
                target_object=intent.target,
                content={
                    "title": f"Information: {target_obj.name if target_obj else 'Unknown'}",
                    "data_type": intent.parameters.get("data_type", "general"),
                    **elle_action.visual_data
                },
                duration=5.0,
                style={"background": "rgba(0, 0, 0, 0.8)", "text_color": "white"}
            )

        # Navigate intents → path visualization
        elif intent.type.value == "navigate":
            return ARVisualization(
                type="path",
                target_object=intent.target,
                content={
                    "show_distance": True,
                    "show_waypoints": True,
                    **elle_action.visual_data
                },
                duration=30.0,  # Path shown until arrival
                style={"color": "blue", "line_width": 0.1}
            )

        # Select intents → object highlight
        elif intent.type.value == "select":
            return ARVisualization(
                type="highlight",
                target_object=intent.target,
                content=elle_action.visual_data,
                duration=3.0,
                style={"color": "yellow", "opacity": 0.6, "pulse": True}
            )

        # Command intents → task panel
        elif intent.type.value == "command":
            return ARVisualization(
                type="panel",
                target_object=intent.target,
                content={
                    "task_name": intent.parameters.get("task_name", "task"),
                    "status": "started",
                    **elle_action.visual_data
                },
                duration=10.0,
                style={"position": "top_right", "size": "medium"}
            )

        return None

    def _get_spatial_audio_position(
        self,
        intent: Intent,
        ar_context: ARContext
    ) -> Optional[List[float]]:
        """Get 3D position for spatial audio."""
        # If there's a target object, position audio there
        if intent.target:
            target_obj = ar_context.find_object_by_id(intent.target)
            if target_obj:
                return [
                    target_obj.position.x,
                    target_obj.position.y,
                    target_obj.position.z
                ]

        # If user is looking at something, position audio there
        if ar_context.gaze_target:
            gaze_obj = ar_context.find_object_by_id(ar_context.gaze_target)
            if gaze_obj:
                return [
                    gaze_obj.position.x,
                    gaze_obj.position.y,
                    gaze_obj.position.z
                ]

        # Default: position slightly in front of user
        return [
            ar_context.user_position.x,
            ar_context.user_position.y,
            ar_context.user_position.z + 2.0  # 2 meters in front
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_queries": self.total_queries,
            "total_processing_time_ms": self.total_processing_time,
            "avg_processing_time_ms": (
                self.total_processing_time / self.total_queries
                if self.total_queries > 0 else 0
            ),
            "recent_intents": len(self.recent_intents),
            "recent_responses": len(self.recent_responses),
            "hololoom_enabled": self.enable_hololoom,
            "spatial_audio_enabled": self.enable_spatial_audio
        }


# ============================================================================
# Helper Functions
# ============================================================================

async def test_elle_bridge():
    """Test Elle Bridge with sample scenarios."""
    from hololoom.voice.ar_context import create_test_context

    print("Testing Elle Bridge")
    print("=" * 60)

    # Create bridge (without HoloLoom for testing)
    bridge = ElleBridge(
        orchestrator=None,
        enable_hololoom=False,
        enable_spatial_audio=True
    )

    # Create test context
    context = create_test_context()

    # Test scenarios
    test_scenarios = [
        ("Show me the health status of this hive", ResponseMode.MULTIMODAL_SPATIAL),
        ("Go to the hive on my left", ResponseMode.MULTIMODAL),
        ("This one", ResponseMode.MULTIMODAL),
        ("Why is the population low?", ResponseMode.VOICE_ONLY),
    ]

    for transcript, mode in test_scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: \"{transcript}\"")
        print(f"Mode: {mode.value}")
        print()

        # Process command
        response = await bridge.process_voice_command(transcript, context, mode)

        print(f"Voice: \"{response.voice_text}\"")
        if response.ar_visualization:
            print(f"AR Viz: {response.ar_visualization.type} on {response.ar_visualization.target_object}")
        if response.spatial_audio_position:
            print(f"Spatial Audio: {response.spatial_audio_position}")
        print(f"Processing Time: {response.processing_time:.1f}ms")

    # Print statistics
    print(f"\n{'='*60}")
    print("Statistics:")
    stats = bridge.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_elle_bridge())
