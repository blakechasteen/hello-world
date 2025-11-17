"""Demo: Object Detection for Beekeeping.

Demonstrates object detection capabilities:
- YOLOv8 integration with fallback
- Bounding box visualization
- Multi-class detection
- Non-maximum suppression

Created: 2025-11-17
"""

import numpy as np
import asyncio
from pathlib import Path

from HoloLoom.vision import ObjectDetector, ObjectClass


async def demo_basic_detection():
    """Demo basic object detection."""
    print("=" * 60)
    print("DEMO: Basic Object Detection")
    print("=" * 60)
    print()

    # Create detector (uses simplified detection if YOLOv8 unavailable)
    detector = ObjectDetector(use_yolo=True, confidence_threshold=0.5)

    print(f"Detector initialized: {type(detector).__name__}")
    print(f"Using YOLO: {detector.use_yolo}")
    print()

    # Create synthetic test frame (yellow blobs = bees)
    print("Creating test frame with synthetic bees...")
    frame = create_test_frame_with_bees()

    print(f"Frame shape: {frame.shape}")
    print()

    # Detect objects
    print("Running detection...")
    detections = await detector.detect(frame)

    print(f"Found {len(detections)} detections")
    print()

    # Display results
    for i, det in enumerate(detections, 1):
        print(f"Detection {i}:")
        print(f"  Class: {det.class_name.value}")
        print(f"  Confidence: {det.confidence:.2f}")
        print(f"  BBox: ({det.bounding_box.x:.3f}, {det.bounding_box.y:.3f}, "
              f"{det.bounding_box.width:.3f}, {det.bounding_box.height:.3f})")
        print(f"  Features: {det.features.shape if det.features is not None else 'None'}")
        print()


async def demo_visualization():
    """Demo detection visualization."""
    print("=" * 60)
    print("DEMO: Detection Visualization")
    print("=" * 60)
    print()

    detector = ObjectDetector(use_yolo=False)

    # Create frame with bees
    frame = create_test_frame_with_bees()

    # Detect
    detections = await detector.detect(frame)

    print(f"Detected {len(detections)} objects")
    print()

    # Visualize
    print("Visualizing detections...")
    annotated = detector.visualize_detections(
        frame,
        detections,
        show_labels=True,
        show_confidence=True
    )

    print(f"Annotated frame shape: {annotated.shape}")
    print()

    # Save (if possible)
    try:
        import cv2
        output_dir = Path("demos/output")
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / "detection_visualization.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        print(f"Saved visualization to: {output_path}")
        print()

    except ImportError:
        print("OpenCV not available - skipping save")
        print()


async def demo_multi_class_detection():
    """Demo detection of multiple object classes."""
    print("=" * 60)
    print("DEMO: Multi-Class Detection")
    print("=" * 60)
    print()

    detector = ObjectDetector(use_yolo=False)

    # Create frame with various objects
    frame = create_complex_test_frame()

    # Detect
    detections = await detector.detect(frame)

    print(f"Total detections: {len(detections)}")
    print()

    # Group by class
    by_class = {}
    for det in detections:
        class_name = det.class_name.value
        if class_name not in by_class:
            by_class[class_name] = []
        by_class[class_name].append(det)

    print("Detections by class:")
    for class_name, dets in sorted(by_class.items()):
        print(f"  {class_name}: {len(dets)} detections")
        avg_conf = np.mean([d.confidence for d in dets])
        print(f"    Avg confidence: {avg_conf:.2f}")
    print()


async def demo_nms():
    """Demo Non-Maximum Suppression."""
    print("=" * 60)
    print("DEMO: Non-Maximum Suppression (NMS)")
    print("=" * 60)
    print()

    from HoloLoom.vision import Detection, BoundingBox

    detector = ObjectDetector(use_yolo=False, nms_threshold=0.4)

    # Create overlapping detections manually
    detections = [
        Detection(ObjectClass.BEE, 0.95, BoundingBox(0.5, 0.5, 0.1, 0.1)),
        Detection(ObjectClass.BEE, 0.85, BoundingBox(0.51, 0.51, 0.1, 0.1)),  # Overlap
        Detection(ObjectClass.BEE, 0.90, BoundingBox(0.52, 0.52, 0.1, 0.1)),  # Overlap
        Detection(ObjectClass.BEE, 0.88, BoundingBox(0.8, 0.8, 0.1, 0.1)),    # Separate
    ]

    print(f"Before NMS: {len(detections)} detections")
    print()

    # Apply NMS
    filtered = detector._non_max_suppression(detections)

    print(f"After NMS: {len(filtered)} detections")
    print()

    print("Kept detections:")
    for i, det in enumerate(filtered, 1):
        print(f"  {i}. Confidence: {det.confidence:.2f}, "
              f"Position: ({det.bounding_box.x:.2f}, {det.bounding_box.y:.2f})")
    print()

    print("(Overlapping detections were removed, keeping highest confidence)")
    print()


async def demo_feature_extraction():
    """Demo feature extraction for tracking."""
    print("=" * 60)
    print("DEMO: Feature Extraction")
    print("=" * 60)
    print()

    from HoloLoom.vision import BoundingBox

    detector = ObjectDetector(use_yolo=False)

    # Create test frame
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    # Extract features from different regions
    regions = [
        BoundingBox(0.25, 0.25, 0.1, 0.1),
        BoundingBox(0.75, 0.75, 0.1, 0.1),
    ]

    print("Extracting features from regions...")
    print()

    for i, bbox in enumerate(regions, 1):
        features = detector._extract_features(frame, bbox)

        print(f"Region {i}:")
        print(f"  BBox: ({bbox.x:.2f}, {bbox.y:.2f})")
        print(f"  Feature shape: {features.shape}")
        print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  Feature mean: {features.mean():.3f}")
        print()

    # Compute similarity
    feat1 = detector._extract_features(frame, regions[0])
    feat2 = detector._extract_features(frame, regions[1])

    # Cosine similarity
    feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-6)
    feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-6)
    similarity = np.dot(feat1_norm, feat2_norm)

    print(f"Feature similarity: {similarity:.3f}")
    print("(Used for tracking bee identity across frames)")
    print()


def create_test_frame_with_bees():
    """Create test frame with synthetic bees (yellow blobs)."""
    import cv2

    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Draw some yellow circles (bees)
    bee_positions = [
        (150, 150),
        (300, 200),
        (450, 250),
        (200, 350),
    ]

    for pos in bee_positions:
        # Yellow color (RGB)
        cv2.circle(frame, pos, 15, (255, 255, 0), -1)

        # Add some brown shading
        cv2.circle(frame, pos, 10, (139, 90, 43), -1)

    return frame


def create_complex_test_frame():
    """Create frame with various beekeeping objects."""
    import cv2

    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Bees (yellow)
    for _ in range(10):
        x = np.random.randint(50, 550)
        y = np.random.randint(50, 350)
        cv2.circle(frame, (x, y), 12, (255, 255, 0), -1)

    # Brood cells (pink)
    for _ in range(5):
        x = np.random.randint(50, 550)
        y = np.random.randint(50, 350)
        cv2.circle(frame, (x, y), 20, (255, 192, 203), -1)

    # Honey cells (gold)
    for _ in range(7):
        x = np.random.randint(50, 550)
        y = np.random.randint(50, 350)
        cv2.circle(frame, (x, y), 18, (255, 215, 0), -1)

    return frame


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "OBJECT DETECTION DEMO SUITE" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Run demos
    await demo_basic_detection()
    await demo_visualization()
    await demo_multi_class_detection()
    await demo_nms()
    await demo_feature_extraction()

    print("=" * 60)
    print("All demos complete!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(main())
