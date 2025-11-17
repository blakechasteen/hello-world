"""
Gesture Recognition Demo
=========================
Basic demonstration of hand gesture recognition.

This demo shows how to use the GestureRecognizer to detect and classify
hand gestures in real-time.

Author: HoloLoom Team (Agent M)
Date: November 17, 2025
"""

import asyncio
import numpy as np
import time
from HoloLoom.voice.gesture_recognition import (
    GestureRecognizer,
    GestureType,
    create_test_hand,
    Gesture
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


async def demo_basic_recognition():
    """Demo: Basic gesture recognition."""
    print_header("Demo 1: Basic Gesture Recognition")

    # Create gesture recognizer (simplified mode for demo)
    recognizer = GestureRecognizer(use_mediapipe=False)

    print("\n✓ GestureRecognizer initialized (simplified mode)")
    print(f"  MediaPipe: {recognizer.use_mediapipe}")

    # Test all gesture types
    gesture_types = [
        GestureType.POINT,
        GestureType.OPEN_PALM,
        GestureType.GRAB,
        GestureType.PINCH
    ]

    print("\nTesting gesture classification:")
    print("-" * 70)

    for gesture_type in gesture_types:
        # Create test hand
        hand = create_test_hand(gesture_type)

        # Classify gesture
        classified_type, confidence = recognizer._classify_gesture(hand)

        # Print results
        status = "✓" if classified_type == gesture_type else "✗"
        print(f"\n{status} {gesture_type.value.upper()}:")
        print(f"  Expected: {gesture_type.value}")
        print(f"  Detected: {classified_type.value}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Hand: {hand.handedness}")

        # Show finger states
        print("  Finger states:")
        for finger in ["index", "middle", "ring", "pinky"]:
            extended = hand.get_finger_extended(finger)
            state = "extended" if extended else "curled"
            print(f"    {finger.capitalize()}: {state}")


async def demo_gesture_properties():
    """Demo: Gesture properties and metadata."""
    print_header("Demo 2: Gesture Properties")

    # Create test gestures
    hand_point = create_test_hand(GestureType.POINT)
    hand_palm = create_test_hand(GestureType.OPEN_PALM)

    print("\nPOINTING Gesture:")
    print("-" * 70)
    print(f"  Palm center: {hand_point.get_palm_center()}")
    direction = hand_point.get_pointing_direction()
    print(f"  Pointing direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})")
    print(f"  Direction magnitude: {sum(d**2 for d in direction)**0.5:.3f} (should be ~1.0)")

    print("\nOPEN PALM Gesture:")
    print("-" * 70)
    print(f"  Palm center: {hand_palm.get_palm_center()}")
    all_extended = all([
        hand_palm.get_finger_extended('index'),
        hand_palm.get_finger_extended('middle'),
        hand_palm.get_finger_extended('ring'),
        hand_palm.get_finger_extended('pinky')
    ])
    print(f"  All fingers extended: {all_extended}")


async def demo_recognition_pipeline():
    """Demo: Full recognition pipeline with frame."""
    print_header("Demo 3: Recognition Pipeline")

    recognizer = GestureRecognizer(use_mediapipe=False)

    # Create dummy camera frame (would be from camera in real use)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print("\nProcessing camera frame:")
    print(f"  Frame shape: {frame.shape}")
    print(f"  Frame dtype: {frame.dtype}")

    start_time = time.time()
    gestures = await recognizer.recognize(frame)
    elapsed = time.time() - start_time

    print(f"\n✓ Recognition complete in {elapsed*1000:.2f}ms")
    print(f"  Gestures detected: {len(gestures)}")

    if not gestures:
        print("  (No gestures detected - simplified mode with empty frame)")


async def demo_gesture_creation():
    """Demo: Creating and using Gesture objects."""
    print_header("Demo 4: Gesture Objects")

    hand = create_test_hand(GestureType.POINT)

    # Create a gesture
    gesture = Gesture(
        type=GestureType.POINT,
        hand=hand,
        confidence=0.92,
        direction=(0.1, -0.2, 0.97),
        target_position=(0.15, -0.05, 5.0),
        timestamp=time.time()
    )

    print("\nGesture Object:")
    print("-" * 70)
    print(f"  Type: {gesture.type.value}")
    print(f"  Hand: {gesture.hand.handedness}")
    print(f"  Confidence: {gesture.confidence:.2%}")
    print(f"  Direction: {gesture.direction}")
    print(f"  Target position: {gesture.target_position}")
    print(f"  Timestamp: {gesture.timestamp:.2f}")
    print(f"\n  String representation: {gesture}")


async def demo_motion_tracking():
    """Demo: Motion history for swipe detection."""
    print_header("Demo 5: Motion History (Swipe Detection)")

    recognizer = GestureRecognizer(use_mediapipe=False)

    print("\nMotion history tracking:")
    print(f"  History length: {recognizer.motion_history_length} frames")
    print(f"  Current tracked hands: {len(recognizer.motion_history)}")

    # Simulate adding motion history
    hand = create_test_hand(GestureType.POINT)
    recognizer._update_motion_history(hand)

    print(f"\n✓ Updated motion history for {hand.handedness} hand")
    print(f"  History entries: {len(recognizer.motion_history.get(hand.handedness, []))}")


async def demo_performance():
    """Demo: Performance characteristics."""
    print_header("Demo 6: Performance Characteristics")

    recognizer = GestureRecognizer(use_mediapipe=False)

    # Test classification speed
    hand = create_test_hand(GestureType.POINT)

    iterations = 1000
    start_time = time.time()

    for _ in range(iterations):
        recognizer._classify_gesture(hand)

    elapsed = time.time() - start_time
    avg_time = (elapsed / iterations) * 1000  # ms

    print(f"\nPerformance Test:")
    print("-" * 70)
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Average per classification: {avg_time:.3f}ms")
    print(f"  Throughput: {iterations / elapsed:.0f} classifications/second")

    # Test with multiple gesture types
    print("\nClassification speed by gesture type:")
    for gesture_type in [GestureType.POINT, GestureType.OPEN_PALM, GestureType.GRAB, GestureType.PINCH]:
        hand = create_test_hand(gesture_type)
        iterations = 100

        start_time = time.time()
        for _ in range(iterations):
            recognizer._classify_gesture(hand)
        elapsed = time.time() - start_time

        print(f"  {gesture_type.value.upper()}: {(elapsed / iterations) * 1000:.3f}ms")


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  GESTURE RECOGNITION DEMO".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  HoloLoom VoiceAgent + Elle AR Integration".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run demos
    await demo_basic_recognition()
    await demo_gesture_properties()
    await demo_recognition_pipeline()
    await demo_gesture_creation()
    await demo_motion_tracking()
    await demo_performance()

    # Summary
    print_header("Demo Complete")
    print("\nKey Takeaways:")
    print("  ✓ 10 gesture types supported (Point, Grab, Open Palm, Pinch, Swipes, etc.)")
    print("  ✓ MediaPipe Hands integration with graceful fallback")
    print("  ✓ Real-time gesture classification (<1ms per gesture)")
    print("  ✓ Motion history for swipe/wave detection")
    print("  ✓ Comprehensive gesture metadata (direction, target, confidence)")
    print("\nNext steps:")
    print("  → Run demo_gesture_mapping.py to see context-aware mapping")
    print("  → Run demo_multimodal_fusion.py to see voice + gesture fusion")
    print()


if __name__ == "__main__":
    asyncio.run(main())
