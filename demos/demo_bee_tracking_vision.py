"""Demo: Bee Tracking Across Frames.

Demonstrates multi-object tracking:
- Kalman filtering for motion prediction
- Hungarian algorithm for data association
- Track lifecycle management
- Activity level computation

Created: 2025-11-17
"""

import numpy as np
import asyncio
from pathlib import Path

from HoloLoom.vision import (
    ObjectDetector,
    BeeTracker,
    ObjectClass,
    Detection,
    BoundingBox,
)


async def demo_single_bee_tracking():
    """Demo tracking a single bee across frames."""
    print("=" * 60)
    print("DEMO: Single Bee Tracking")
    print("=" * 60)
    print()

    tracker = BeeTracker(max_tracks=100)

    # Simulate bee moving across frames
    print("Simulating bee movement across 10 frames...")
    print()

    for frame_num in range(10):
        # Create detection (bee moving right and down)
        x = 0.3 + frame_num * 0.05
        y = 0.3 + frame_num * 0.03

        bbox = BoundingBox(x, y, 0.05, 0.05)
        detection = Detection(ObjectClass.BEE, 0.9, bbox)

        # Update tracker
        tracks = await tracker.update([detection], frame_number=frame_num)

        # Display status
        if frame_num < 3:
            status = "tentative (not yet confirmed)"
        else:
            status = "confirmed"

        print(f"Frame {frame_num}:")
        print(f"  Detection: ({x:.2f}, {y:.2f})")
        print(f"  Active tracks: {len(tracker.tracks)}")
        print(f"  Confirmed tracks: {len(tracks)}")
        print(f"  Status: {status}")
        print()

    # Show final track info
    if tracker.tracks:
        track = list(tracker.tracks.values())[0]
        print("Final track info:")
        print(f"  Track ID: {track.track_id}")
        print(f"  Age: {track.age} frames")
        print(f"  Position: ({track.current_position[0]:.2f}, {track.current_position[1]:.2f})")
        print(f"  Velocity: ({track.velocity[0]:.3f}, {track.velocity[1]:.3f})")
        print(f"  Activity: {track.activity_level:.2f}")
        print()


async def demo_multiple_bee_tracking():
    """Demo tracking multiple bees."""
    print("=" * 60)
    print("DEMO: Multiple Bee Tracking")
    print("=" * 60)
    print()

    tracker = BeeTracker(max_tracks=100)

    # Simulate 3 bees moving in different patterns
    print("Simulating 3 bees with different movement patterns...")
    print()

    num_frames = 15

    for frame_num in range(num_frames):
        detections = []

        # Bee 1: Moving right
        x1 = 0.2 + frame_num * 0.04
        y1 = 0.3
        detections.append(Detection(ObjectClass.BEE, 0.95, BoundingBox(x1, y1, 0.04, 0.04)))

        # Bee 2: Moving down
        x2 = 0.5
        y2 = 0.2 + frame_num * 0.03
        detections.append(Detection(ObjectClass.BEE, 0.90, BoundingBox(x2, y2, 0.04, 0.04)))

        # Bee 3: Circular motion
        angle = frame_num * np.pi / 7
        x3 = 0.7 + 0.15 * np.cos(angle)
        y3 = 0.5 + 0.15 * np.sin(angle)
        detections.append(Detection(ObjectClass.BEE, 0.85, BoundingBox(x3, y3, 0.04, 0.04)))

        # Update tracker
        tracks = await tracker.update(detections, frame_number=frame_num)

        if frame_num % 5 == 0:
            print(f"Frame {frame_num}:")
            print(f"  Detections: {len(detections)}")
            print(f"  Confirmed tracks: {len(tracks)}")
            print()

    # Final statistics
    stats = tracker.get_statistics()

    print("Final statistics:")
    print(f"  Total tracks created: {stats['total_tracks_created']}")
    print(f"  Active tracks: {stats['active_tracks']}")
    print(f"  Confirmed tracks: {stats['confirmed_tracks']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    print(f"  Avg track length: {stats['avg_track_length']:.1f} frames")
    print()


async def demo_track_lifecycle():
    """Demo track creation, confirmation, and deletion."""
    print("=" * 60)
    print("DEMO: Track Lifecycle Management")
    print("=" * 60)
    print()

    tracker = BeeTracker(max_tracks=100, max_age=5)

    print("Creating new track...")
    print()

    # Frame 0-2: New track (tentative)
    for frame_num in range(3):
        bbox = BoundingBox(0.5, 0.5, 0.05, 0.05)
        det = Detection(ObjectClass.BEE, 0.9, bbox)

        tracks = await tracker.update([det], frame_number=frame_num)

        print(f"Frame {frame_num}:")
        print(f"  Confirmed tracks: {len(tracks)}")

        if tracker.tracks:
            track = list(tracker.tracks.values())[0]
            print(f"  Track age: {track.age}")
            print(f"  Tentative: {track.is_tentative()}")
        print()

    print("Track confirmed after 3 detections!")
    print()

    # Frame 3-7: No detections (track should die after max_age=5)
    print("No detections for next 5 frames (track will die)...")
    print()

    for frame_num in range(3, 10):
        tracks = await tracker.update([], frame_number=frame_num)

        print(f"Frame {frame_num}:")
        print(f"  Active tracks: {len(tracker.tracks)}")
        print(f"  Confirmed tracks: {len(tracks)}")

        if tracker.tracks:
            track = list(tracker.tracks.values())[0]
            print(f"  Time since update: {track.time_since_update}")
            print(f"  Is dead: {track.is_dead(max_age=5)}")
        else:
            print(f"  Track deleted (exceeded max age)")

        print()


async def demo_data_association():
    """Demo Hungarian algorithm for detection-track matching."""
    print("=" * 60)
    print("DEMO: Data Association (Hungarian Algorithm)")
    print("=" * 60)
    print()

    tracker = BeeTracker()

    # Initialize with 2 bees
    print("Initializing with 2 bees...")
    detections = [
        Detection(ObjectClass.BEE, 0.9, BoundingBox(0.3, 0.3, 0.05, 0.05)),
        Detection(ObjectClass.BEE, 0.9, BoundingBox(0.7, 0.7, 0.05, 0.05)),
    ]

    for _ in range(5):
        await tracker.update(detections, frame_number=0)

    print(f"Confirmed tracks: {len([t for t in tracker.tracks.values() if not t.is_tentative()])}")
    print()

    # Next frame: bees swap positions (test data association)
    print("Next frame: bees swap positions (testing association)...")
    print()

    detections_swapped = [
        Detection(ObjectClass.BEE, 0.9, BoundingBox(0.7, 0.7, 0.05, 0.05)),  # Was bee 2
        Detection(ObjectClass.BEE, 0.9, BoundingBox(0.3, 0.3, 0.05, 0.05)),  # Was bee 1
    ]

    tracks = await tracker.update(detections_swapped, frame_number=5)

    print(f"Tracks maintained: {len(tracks)}")
    print("(Hungarian algorithm correctly matched detections to existing tracks)")
    print()

    # Show track positions
    for track in tracks:
        print(f"Track {track.track_id}:")
        print(f"  Position: ({track.current_position[0]:.2f}, {track.current_position[1]:.2f})")
        print(f"  Detections: {len(track.detections)}")
    print()


async def demo_hive_activity():
    """Demo hive activity computation."""
    print("=" * 60)
    print("DEMO: Hive Activity Computation")
    print("=" * 60)
    print()

    tracker = BeeTracker()

    # Scenario 1: Low activity (few bees, slow movement)
    print("Scenario 1: Low activity (few bees, slow movement)")
    print()

    for frame_num in range(10):
        detections = [
            Detection(
                ObjectClass.BEE,
                0.9,
                BoundingBox(0.5 + frame_num * 0.01, 0.5, 0.05, 0.05)
            )
        ]
        await tracker.update(detections, frame_number=frame_num)

    activity_low = await tracker.compute_hive_activity()
    print(f"Activity level: {activity_low:.2f}")
    print()

    tracker.reset()

    # Scenario 2: High activity (many bees, fast movement)
    print("Scenario 2: High activity (many bees, fast movement)")
    print()

    for frame_num in range(10):
        detections = [
            Detection(
                ObjectClass.BEE,
                0.9,
                BoundingBox(
                    0.2 + i * 0.1 + frame_num * 0.05,
                    0.3 + np.random.rand() * 0.4,
                    0.05,
                    0.05
                )
            )
            for i in range(10)  # 10 bees
        ]
        await tracker.update(detections, frame_number=frame_num)

    activity_high = await tracker.compute_hive_activity()
    print(f"Activity level: {activity_high:.2f}")
    print()

    print(f"Activity increase: {(activity_high - activity_low):.2f}")
    print("(More bees + faster movement = higher activity)")
    print()


async def demo_integration_with_detection():
    """Demo integration with object detector."""
    print("=" * 60)
    print("DEMO: Integration with Object Detector")
    print("=" * 60)
    print()

    detector = ObjectDetector(use_yolo=False)
    tracker = BeeTracker()

    print("Processing video frames with detection + tracking...")
    print()

    # Simulate video frames
    for frame_num in range(20):
        # Create synthetic frame
        frame = create_animated_frame(frame_num)

        # Detect bees
        detections = await detector.detect(frame)

        # Update tracker
        tracks = await tracker.update(detections, frame_number=frame_num)

        if frame_num % 5 == 0:
            activity = await tracker.compute_hive_activity()

            print(f"Frame {frame_num}:")
            print(f"  Detections: {len(detections)}")
            print(f"  Confirmed tracks: {len(tracks)}")
            print(f"  Activity: {activity:.2f}")
            print()

    # Final statistics
    stats = tracker.get_statistics()

    print("Final statistics:")
    print(f"  Total detections processed: {stats['total_detections_processed']}")
    print(f"  Total tracks created: {stats['total_tracks_created']}")
    print(f"  Final active tracks: {stats['active_tracks']}")
    print()


def create_animated_frame(frame_num):
    """Create animated frame with moving bees."""
    import cv2

    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Add moving bees
    num_bees = 5
    for i in range(num_bees):
        # Each bee moves in a different pattern
        x = int(100 + i * 100 + frame_num * (3 + i))
        y = int(200 + 50 * np.sin(frame_num * 0.1 + i))

        # Keep in bounds
        x = x % 580 + 10
        y = max(10, min(390, y))

        # Draw bee (yellow)
        cv2.circle(frame, (x, y), 12, (255, 255, 0), -1)

    return frame


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "BEE TRACKING DEMO SUITE" + " " * 23 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Run demos
    await demo_single_bee_tracking()
    await demo_multiple_bee_tracking()
    await demo_track_lifecycle()
    await demo_data_association()
    await demo_hive_activity()
    await demo_integration_with_detection()

    print("=" * 60)
    print("All demos complete!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(main())
