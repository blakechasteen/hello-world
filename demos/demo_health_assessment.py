"""Demo: Hive Health Assessment.

Demonstrates visual health assessment:
- Population and activity metrics
- Brood pattern analysis
- Pest detection (varroa mites)
- Resource store estimation
- Health status classification
- Trend analysis

Created: 2025-11-17
"""

import numpy as np
import asyncio
from pathlib import Path

from HoloLoom.vision import (
    ObjectDetector,
    BeeTracker,
    HealthAssessor,
    HealthStatus,
    ObjectClass,
    Detection,
    BoundingBox,
)


async def demo_basic_health_assessment():
    """Demo basic health assessment."""
    print("=" * 60)
    print("DEMO: Basic Health Assessment")
    print("=" * 60)
    print()

    detector = ObjectDetector(use_yolo=False)
    tracker = BeeTracker()
    assessor = HealthAssessor(detector, tracker)

    # Create test frame
    frame = create_healthy_hive_frame()

    print("Assessing hive health from frame...")
    print()

    # Assess
    metrics = await assessor.assess(frame, frame_number=0)

    print("Health Metrics:")
    print(f"  Bee population: {metrics.bee_population}")
    print(f"  Queen present: {metrics.queen_present}")
    print(f"  Activity level: {metrics.activity_level:.2f}")
    print(f"  Brood pattern: {metrics.brood_pattern_score:.2f}")
    print(f"  Varroa detected: {metrics.varroa_detected}")
    print(f"  Honey stores: {metrics.honey_stores:.1%}")
    print(f"  Pollen stores: {metrics.pollen_stores:.1%}")
    print()

    print(f"Overall Health: {metrics.overall_health:.2f}")
    print(f"Health Status: {metrics.health_status.value.upper()}")
    print()


async def demo_health_summary():
    """Demo health summary generation."""
    print("=" * 60)
    print("DEMO: Health Summary with Recommendations")
    print("=" * 60)
    print()

    detector = ObjectDetector(use_yolo=False)
    tracker = BeeTracker()
    assessor = HealthAssessor(detector, tracker)

    # Assess
    frame = create_problematic_hive_frame()
    metrics = await assessor.assess(frame, frame_number=0)

    # Print summary
    print(metrics.get_summary())
    print()


async def demo_health_scenarios():
    """Demo different health scenarios."""
    print("=" * 60)
    print("DEMO: Health Scenarios")
    print("=" * 60)
    print()

    detector = ObjectDetector(use_yolo=False)
    tracker = BeeTracker()

    scenarios = [
        ("Healthy Hive", create_healthy_hive_frame()),
        ("Low Population", create_low_population_frame()),
        ("Varroa Infestation", create_varroa_frame()),
        ("Low Resources", create_low_resources_frame()),
    ]

    for scenario_name, frame in scenarios:
        print(f"\nScenario: {scenario_name}")
        print("-" * 40)

        assessor = HealthAssessor(detector, tracker)
        metrics = await assessor.assess(frame, frame_number=0)

        print(f"  Population: {metrics.bee_population}")
        print(f"  Overall Health: {metrics.overall_health:.2f}")
        print(f"  Status: {metrics.health_status.value}")

        # Show top recommendation
        recommendations = metrics._get_recommendations()
        if recommendations:
            print(f"  Top Issue: {recommendations[0]}")

        print()


async def demo_trend_analysis():
    """Demo health trend analysis over time."""
    print("=" * 60)
    print("DEMO: Health Trend Analysis")
    print("=" * 60)
    print()

    detector = ObjectDetector(use_yolo=False)
    tracker = BeeTracker()
    assessor = HealthAssessor(detector, tracker)

    print("Simulating hive health over 30 frames...")
    print()

    # Simulate improving health
    for frame_num in range(30):
        # Gradually improve conditions
        bee_count = 30 + frame_num * 2
        activity = 0.5 + frame_num * 0.015

        frame = create_custom_frame(
            bee_count=bee_count,
            activity_level=activity,
            has_varroa=frame_num < 10  # Varroa cleared after frame 10
        )

        metrics = await assessor.assess(frame, frame_number=frame_num)

        if frame_num % 10 == 0:
            print(f"Frame {frame_num}:")
            print(f"  Health: {metrics.overall_health:.2f}")
            print(f"  Population: {metrics.bee_population}")
            print(f"  Varroa: {metrics.varroa_detected}")
            print()

    # Get trend
    trend = assessor.get_trend(window=20)
    stats = assessor.get_statistics()

    print("Health Trend Analysis:")
    print(f"  Trend: {trend}")
    print(f"  Avg health: {stats['avg_health']:.2f}")
    print(f"  Min health: {stats['min_health']:.2f}")
    print(f"  Max health: {stats['max_health']:.2f}")
    print(f"  Varroa detection rate: {stats['varroa_detection_rate']:.1%}")
    print()


async def demo_brood_pattern_analysis():
    """Demo brood pattern analysis."""
    print("=" * 60)
    print("DEMO: Brood Pattern Analysis")
    print("=" * 60)
    print()

    detector = ObjectDetector(use_yolo=False)
    tracker = BeeTracker()
    assessor = HealthAssessor(detector, tracker)

    # Scenario 1: Compact healthy brood
    print("Scenario 1: Compact Healthy Brood Pattern")
    print("-" * 40)

    brood_detections_compact = [
        Detection(
            ObjectClass.BROOD,
            0.95,
            BoundingBox(0.4 + i * 0.05, 0.4 + j * 0.05, 0.04, 0.04)
        )
        for i in range(4)
        for j in range(3)
    ]

    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    score_compact = await assessor._analyze_brood_pattern(brood_detections_compact, frame)

    print(f"Brood pattern score: {score_compact:.2f}")
    print(f"Interpretation: {'Healthy' if score_compact > 0.7 else 'Needs attention'}")
    print()

    # Scenario 2: Spotty brood (disease indicator)
    print("Scenario 2: Spotty Brood Pattern")
    print("-" * 40)

    brood_detections_spotty = [
        Detection(
            ObjectClass.BROOD,
            0.85,
            BoundingBox(0.2 + i * 0.15, 0.3 + j * 0.2, 0.04, 0.04)
        )
        for i in range(3)
        for j in range(2)
    ]

    score_spotty = await assessor._analyze_brood_pattern(brood_detections_spotty, frame)

    print(f"Brood pattern score: {score_spotty:.2f}")
    print(f"Interpretation: {'Healthy' if score_spotty > 0.7 else 'Possible disease'}")
    print()

    print(f"Difference: {(score_compact - score_spotty):.2f}")
    print("(Lower score indicates scattered brood cells - potential health issue)")
    print()


async def demo_resource_estimation():
    """Demo honey and pollen store estimation."""
    print("=" * 60)
    print("DEMO: Resource Store Estimation")
    print("=" * 60)
    print()

    detector = ObjectDetector(use_yolo=False)
    tracker = BeeTracker()
    assessor = HealthAssessor(detector, tracker)

    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Test honey stores
    print("Honey Store Levels:")
    print("-" * 40)

    honey_scenarios = [
        ("Low", [BoundingBox(0.5, 0.5, 0.1, 0.1)]),
        ("Medium", [BoundingBox(0.3, 0.3, 0.15, 0.15), BoundingBox(0.7, 0.7, 0.15, 0.15)]),
        ("High", [BoundingBox(0.3 + i * 0.15, 0.3, 0.2, 0.2) for i in range(4)]),
    ]

    for scenario_name, bboxes in honey_scenarios:
        detections = [
            Detection(ObjectClass.HONEY, 0.9, bbox)
            for bbox in bboxes
        ]

        stores = await assessor._estimate_stores(detections, ObjectClass.HONEY, frame)

        print(f"  {scenario_name}: {stores:.1%}")

    print()

    # Test pollen stores
    print("Pollen Store Levels:")
    print("-" * 40)

    pollen_scenarios = [
        ("Low", [BoundingBox(0.5, 0.5, 0.08, 0.08)]),
        ("Medium", [BoundingBox(0.4 + i * 0.2, 0.4, 0.12, 0.12) for i in range(2)]),
        ("High", [BoundingBox(0.3 + i * 0.12, 0.3, 0.15, 0.15) for i in range(3)]),
    ]

    for scenario_name, bboxes in pollen_scenarios:
        detections = [
            Detection(ObjectClass.POLLEN, 0.9, bbox)
            for bbox in bboxes
        ]

        stores = await assessor._estimate_stores(detections, ObjectClass.POLLEN, frame)

        print(f"  {scenario_name}: {stores:.1%}")

    print()


async def demo_health_classification():
    """Demo health status classification."""
    print("=" * 60)
    print("DEMO: Health Status Classification")
    print("=" * 60)
    print()

    detector = ObjectDetector(use_yolo=False)
    tracker = BeeTracker()
    assessor = HealthAssessor(detector, tracker)

    print("Health Score → Status Mapping:")
    print("-" * 40)

    test_scores = [0.95, 0.75, 0.55, 0.35, 0.15]

    for score in test_scores:
        status = assessor._classify_health_status(score)

        print(f"  Score {score:.2f} → {status.value.upper()}")

    print()


async def demo_integration_pipeline():
    """Demo complete pipeline with all components."""
    print("=" * 60)
    print("DEMO: Complete Health Assessment Pipeline")
    print("=" * 60)
    print()

    detector = ObjectDetector(use_yolo=False)
    tracker = BeeTracker()
    assessor = HealthAssessor(detector, tracker)

    print("Processing video sequence...")
    print()

    # Process 10 frames
    for frame_num in range(10):
        frame = create_dynamic_frame(frame_num)

        metrics = await assessor.assess(frame, frame_number=frame_num)

        if frame_num % 3 == 0:
            print(f"Frame {frame_num}:")
            print(f"  Population: {metrics.bee_population}")
            print(f"  Activity: {metrics.activity_level:.2f}")
            print(f"  Health: {metrics.overall_health:.2f} ({metrics.health_status.value})")
            print()

    # Final statistics
    stats = assessor.get_statistics()

    print("Final Statistics:")
    print(f"  Assessments: {stats['assessments_count']}")
    print(f"  Avg health: {stats['avg_health']:.2f}")
    print(f"  Trend: {stats['trend']}")
    print(f"  Queen detection rate: {stats['queen_detection_rate']:.1%}")
    print()


# ============================================================================
# Helper Functions for Creating Test Frames
# ============================================================================


def create_healthy_hive_frame():
    """Create frame representing a healthy hive."""
    import cv2

    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Many bees
    for _ in range(50):
        x = np.random.randint(50, 550)
        y = np.random.randint(50, 350)
        cv2.circle(frame, (x, y), 10, (255, 255, 0), -1)

    # Compact brood
    for i in range(5):
        for j in range(4):
            x = 100 + i * 40
            y = 100 + j * 30
            cv2.circle(frame, (x, y), 15, (255, 192, 203), -1)

    # Honey stores
    for i in range(6):
        x = 350 + i * 40
        y = 200
        cv2.circle(frame, (x, y), 12, (255, 215, 0), -1)

    return frame


def create_problematic_hive_frame():
    """Create frame with health issues."""
    import cv2

    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Few bees
    for _ in range(15):
        x = np.random.randint(50, 550)
        y = np.random.randint(50, 350)
        cv2.circle(frame, (x, y), 10, (255, 255, 0), -1)

    # Spotty brood
    for i in range(3):
        x = np.random.randint(100, 300)
        y = np.random.randint(100, 250)
        cv2.circle(frame, (x, y), 15, (255, 192, 203), -1)

    # Varroa mites (small red dots)
    for _ in range(5):
        x = np.random.randint(50, 550)
        y = np.random.randint(50, 350)
        cv2.circle(frame, (x, y), 3, (139, 0, 0), -1)

    return frame


def create_low_population_frame():
    """Create frame with low bee population."""
    import cv2

    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Very few bees
    for _ in range(8):
        x = np.random.randint(50, 550)
        y = np.random.randint(50, 350)
        cv2.circle(frame, (x, y), 10, (255, 255, 0), -1)

    return frame


def create_varroa_frame():
    """Create frame with varroa mite infestation."""
    import cv2

    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Normal bees
    for _ in range(30):
        x = np.random.randint(50, 550)
        y = np.random.randint(50, 350)
        cv2.circle(frame, (x, y), 10, (255, 255, 0), -1)

    # Many varroa mites
    for _ in range(15):
        x = np.random.randint(50, 550)
        y = np.random.randint(50, 350)
        cv2.circle(frame, (x, y), 3, (139, 0, 0), -1)

    return frame


def create_low_resources_frame():
    """Create frame with low honey/pollen stores."""
    import cv2

    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Bees
    for _ in range(25):
        x = np.random.randint(50, 550)
        y = np.random.randint(50, 350)
        cv2.circle(frame, (x, y), 10, (255, 255, 0), -1)

    # Very little honey (just 1-2 cells)
    for i in range(2):
        x = 300 + i * 50
        y = 200
        cv2.circle(frame, (x, y), 12, (255, 215, 0), -1)

    return frame


def create_custom_frame(bee_count=30, activity_level=0.5, has_varroa=False):
    """Create customized test frame."""
    import cv2

    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Bees
    for _ in range(bee_count):
        x = np.random.randint(50, 550)
        y = np.random.randint(50, 350)
        cv2.circle(frame, (x, y), 10, (255, 255, 0), -1)

    # Varroa (if needed)
    if has_varroa:
        for _ in range(10):
            x = np.random.randint(50, 550)
            y = np.random.randint(50, 350)
            cv2.circle(frame, (x, y), 3, (139, 0, 0), -1)

    return frame


def create_dynamic_frame(frame_num):
    """Create frame that changes over time."""
    import cv2

    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Population grows
    bee_count = 20 + frame_num * 3

    for i in range(bee_count):
        x = int(100 + (i % 10) * 50 + frame_num * 2)
        y = int(100 + (i // 10) * 50 + 10 * np.sin(frame_num * 0.5))

        x = x % 580 + 10
        y = max(10, min(390, y))

        cv2.circle(frame, (x, y), 10, (255, 255, 0), -1)

    return frame


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "HEALTH ASSESSMENT DEMO SUITE" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Run demos
    await demo_basic_health_assessment()
    await demo_health_summary()
    await demo_health_scenarios()
    await demo_trend_analysis()
    await demo_brood_pattern_analysis()
    await demo_resource_estimation()
    await demo_health_classification()
    await demo_integration_pipeline()

    print("=" * 60)
    print("All demos complete!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(main())
