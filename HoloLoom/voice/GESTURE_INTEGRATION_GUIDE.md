# Gesture Control Integration Guide

**Date**: November 17, 2025
**Author**: Agent M
**For**: Elle AR + VoiceAgent Integration

## Quick Integration

### 1. Add Gesture Control to VoiceAgent

```python
# In HoloLoom/voice/voice_agent.py

from HoloLoom.voice.multimodal_input import create_multimodal_processor
from HoloLoom.voice.ar_context import ARContext

class VoiceAgent:
    def __init__(self):
        # Existing initialization
        self.tts_engine = ...
        self.command_router = ...

        # NEW: Add multimodal processor
        self.multimodal_processor = None

    async def start(self):
        """Start VoiceAgent with gesture control."""
        # Existing startup
        await self.tts_engine.start()

        # NEW: Initialize multimodal processor
        self.multimodal_processor = await create_multimodal_processor(
            enable_gesture=True,
            enable_voice=True,
            use_mediapipe=True  # Production mode
        )

    async def process_input(
        self,
        voice_transcript: str = None,
        camera_frame: np.ndarray = None,
        ar_context: ARContext = None
    ):
        """
        Process multimodal input (voice + gesture).

        Args:
            voice_transcript: Transcribed voice input
            camera_frame: Camera frame for gesture recognition
            ar_context: Current AR context from Elle

        Returns:
            MultimodalCommand with fused input
        """
        # Process through multimodal processor
        command = await self.multimodal_processor.process(
            voice_transcript=voice_transcript,
            camera_frame=camera_frame,
            ar_context=ar_context
        )

        # Execute command
        await self.execute_command(command)

        return command

    async def execute_command(self, command):
        """Execute multimodal command."""
        # Route to appropriate handler
        if command.command == "select_object":
            await self.handle_select_object(command.parameters)
        elif command.command == "show_details":
            await self.handle_show_details(command.parameters)
        elif command.command == "navigate_next":
            await self.handle_navigate_next(command.parameters)
        # ... etc

    async def close(self):
        """Cleanup resources."""
        if self.multimodal_processor:
            await self.multimodal_processor.close()
```

### 2. Update Elle AR Bridge

```python
# In HoloLoom/voice/elle_bridge.py

from HoloLoom.voice.ar_context import ARContext, ARObject, ARObjectType, Vector3, Quaternion

class ElleBridge:
    def get_ar_context(self) -> ARContext:
        """
        Get current AR context from Elle AR.

        Returns:
            ARContext with current state
        """
        # Get state from Elle AR
        user_pos = self.get_user_position()  # From Elle AR headset
        user_orient = self.get_user_orientation()
        gaze_target = self.get_gaze_target()
        visible_objects = self.get_visible_objects()
        selected_object = self.get_selected_object()

        # Create AR context
        context = ARContext(
            user_position=Vector3(*user_pos),
            user_orientation=Quaternion(*user_orient),
            gaze_direction=self.get_gaze_direction(),
            gaze_target=gaze_target,
            visible_objects=self.convert_to_ar_objects(visible_objects),
            selected_object=selected_object,
            active_scene=self.current_scene,
            conversation_context=self.conversation_state
        )

        # Update spatial references
        context.update_spatial_references()

        return context

    def convert_to_ar_objects(self, elle_objects) -> List[ARObject]:
        """Convert Elle AR objects to ARObject format."""
        ar_objects = []

        for obj in elle_objects:
            ar_obj = ARObject(
                id=obj.id,
                type=self.map_object_type(obj.type),
                position=Vector3(*obj.position),
                name=obj.name,
                metadata=obj.metadata,
                visible=obj.is_visible,
                selectable=obj.is_selectable
            )
            ar_objects.append(ar_obj)

        return ar_objects

    def map_object_type(self, elle_type: str) -> ARObjectType:
        """Map Elle object types to ARObjectType."""
        type_map = {
            "beehive": ARObjectType.BEEHIVE,
            "frame": ARObjectType.FRAME,
            "tool": ARObjectType.TOOL,
            "marker": ARObjectType.MARKER,
            "annotation": ARObjectType.ANNOTATION,
            "panel": ARObjectType.INFO_PANEL,
            "arrow": ARObjectType.NAVIGATION_ARROW
        }
        return type_map.get(elle_type, ARObjectType.OTHER)
```

### 3. Main AR Loop

```python
# Main AR application loop

import cv2
import asyncio
from HoloLoom.voice.voice_agent import VoiceAgent
from HoloLoom.voice.elle_bridge import ElleBridge

async def main():
    # Initialize components
    voice_agent = VoiceAgent()
    elle_bridge = ElleBridge()

    # Start VoiceAgent
    await voice_agent.start()

    # Open camera
    cap = cv2.VideoCapture(0)

    try:
        while True:
            # Get camera frame
            ret, frame = cap.read()
            if not ret:
                break

            # Get AR context from Elle
            ar_context = elle_bridge.get_ar_context()

            # Get voice transcript (if available)
            voice_transcript = await voice_agent.get_voice_transcript()

            # Process multimodal input
            command = await voice_agent.process_input(
                voice_transcript=voice_transcript,
                camera_frame=frame,
                ar_context=ar_context
            )

            # Log command
            print(f"Command: {command.command}")
            print(f"Modalities: {command.modalities}")
            print(f"Fusion: {command.fusion_strategy}")
            print(f"Confidence: {command.confidence:.2%}")

            # Small delay
            await asyncio.sleep(0.033)  # ~30 FPS

    finally:
        cap.release()
        await voice_agent.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Example Interactions

### Scenario 1: Selecting a Beehive

```python
# User points at hive and says "this one"
voice_transcript = "this one"
camera_frame = <frame with pointing gesture>
ar_context = <context with visible hives>

command = await voice_agent.process_input(
    voice_transcript=voice_transcript,
    camera_frame=camera_frame,
    ar_context=ar_context
)

# Result:
# command.command = "select_object"
# command.parameters = {"object_id": "hive_002"}
# command.fusion_strategy = "disambiguated"
# command.modalities = ["voice", "gesture"]
```

### Scenario 2: Showing Details

```python
# User shows open palm while looking at selected hive
# (no voice)
camera_frame = <frame with open palm gesture>
ar_context = <context with hive_001 selected>

command = await voice_agent.process_input(
    voice_transcript=None,
    camera_frame=camera_frame,
    ar_context=ar_context
)

# Result:
# command.command = "show_details"
# command.parameters = {"current_selection": "hive_001"}
# command.fusion_strategy = "gesture_only"
# command.modalities = ["gesture"]
```

### Scenario 3: Navigation

```python
# User says "next" and swipes right
voice_transcript = "next"
camera_frame = <frame with swipe right gesture>
ar_context = <context with hive selected>

command = await voice_agent.process_input(
    voice_transcript=voice_transcript,
    camera_frame=camera_frame,
    ar_context=ar_context
)

# Result:
# command.command = "navigate_next"
# command.fusion_strategy = "reinforced"
# command.confidence = 0.92  # Boosted!
# command.modalities = ["voice", "gesture"]
```

## Configuration

### Production Configuration

```python
# config/gesture_config.py

GESTURE_CONFIG = {
    # MediaPipe settings
    "use_mediapipe": True,
    "max_num_hands": 2,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,

    # Performance settings
    "frame_skip": 1,  # Process every frame
    "motion_history_length": 30,

    # Fusion settings
    "enable_voice": True,
    "enable_gesture": True,
    "confidence_boost_reinforced": 1.2,
    "confidence_reduce_conflicting": 0.7,

    # Context settings
    "enable_context_aware_mapping": True,
    "custom_rules": []
}
```

### Development Configuration

```python
# For development/testing without MediaPipe
DEV_GESTURE_CONFIG = {
    "use_mediapipe": False,  # Simplified mode
    "max_num_hands": 1,
    "enable_voice": True,
    "enable_gesture": True
}
```

## Troubleshooting

### Issue: Gestures not detected

**Check**:
1. MediaPipe installed: `pip install mediapipe`
2. Camera working: `cv2.VideoCapture(0).read()`
3. Lighting conditions adequate
4. Hands in camera frame

### Issue: Low confidence scores

**Solutions**:
1. Lower confidence thresholds in config
2. Improve lighting
3. Check hand positioning (not too close/far)

### Issue: Wrong context detected

**Check**:
1. AR context properly set
2. `selected_object` set when needed
3. `conversation_context` updated

## Performance Optimization

### For High Frame Rates (60+ FPS)

```python
# Skip gesture recognition on some frames
frame_count = 0
FRAME_SKIP = 2  # Process every 2nd frame

while True:
    ret, frame = cap.read()
    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        # Process gestures
        command = await voice_agent.process_input(
            voice_transcript=voice_transcript,
            camera_frame=frame,
            ar_context=ar_context
        )
```

### For Lower Latency

```python
# Single-hand mode
gesture_recognizer = GestureRecognizer(
    max_num_hands=1,  # Faster than 2 hands
    min_detection_confidence=0.8  # Higher threshold
)
```

## Next Steps

1. **Integration Testing**: Test with real Elle AR environment
2. **User Testing**: Gather feedback on gesture intuitiveness
3. **Performance Tuning**: Optimize for production frame rates
4. **Custom Rules**: Add domain-specific gesture mappings
5. **Analytics**: Track gesture usage patterns

## Support

For issues or questions:
- See `GESTURE_CONTROL_README.md` for full documentation
- Check demos in `demos/demo_gesture_*.py`
- Review tests in `HoloLoom/voice/tests/test_gesture_control.py`
