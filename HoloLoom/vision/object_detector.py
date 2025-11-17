"""Object Detection for Beekeeping.

Detects beekeeping objects in camera feed using YOLOv8 with graceful fallback
to simplified detection when YOLOv8 is unavailable.

Classes:
- ObjectClass: Enum of detectable object types
- BoundingBox: 2D bounding box representation
- Detection: Single object detection result
- ObjectDetector: Main detection engine

Created: 2025-11-17
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np
import asyncio
from pathlib import Path


class ObjectClass(Enum):
    """Detectable object classes in beekeeping environment."""

    BEEHIVE = "beehive"
    BEE = "bee"
    FRAME = "frame"  # Hive frame
    BROOD = "brood"  # Brood cells
    HONEY = "honey"  # Honey cells
    POLLEN = "pollen"  # Pollen cells
    QUEEN = "queen"  # Queen bee
    VARROA_MITE = "varroa_mite"
    SMOKER = "smoker"
    HIVE_TOOL = "hive_tool"


@dataclass
class BoundingBox:
    """2D bounding box in normalized coordinates (0-1)."""

    x: float  # Center x (0-1)
    y: float  # Center y (0-1)
    width: float  # Width (0-1)
    height: float  # Height (0-1)

    def to_pixels(
        self,
        image_width: int,
        image_height: int
    ) -> Tuple[int, int, int, int]:
        """Convert to pixel coordinates (x1, y1, x2, y2).

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Tuple of (x1, y1, x2, y2) in pixels
        """
        cx = int(self.x * image_width)
        cy = int(self.y * image_height)
        w = int(self.width * image_width)
        h = int(self.height * image_height)

        x1 = max(0, cx - w // 2)
        y1 = max(0, cy - h // 2)
        x2 = min(image_width, cx + w // 2)
        y2 = min(image_height, cy + h // 2)

        return (x1, y1, x2, y2)

    def from_pixels(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_width: int,
        image_height: int
    ) -> "BoundingBox":
        """Create from pixel coordinates.

        Args:
            x1, y1, x2, y2: Corner coordinates in pixels
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            BoundingBox in normalized coordinates
        """
        cx = (x1 + x2) / (2 * image_width)
        cy = (y1 + y2) / (2 * image_height)
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        return BoundingBox(cx, cy, width, height)

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union with another box.

        Args:
            other: Another bounding box

        Returns:
            IoU score (0-1)
        """
        # Convert to corner coordinates
        x1_self = self.x - self.width / 2
        y1_self = self.y - self.height / 2
        x2_self = self.x + self.width / 2
        y2_self = self.y + self.height / 2

        x1_other = other.x - other.width / 2
        y1_other = other.y - other.height / 2
        x2_other = other.x + other.width / 2
        y2_other = other.y + other.height / 2

        # Intersection
        x1_inter = max(x1_self, x1_other)
        y1_inter = max(y1_self, y1_other)
        x2_inter = min(x2_self, x2_other)
        y2_inter = min(y2_self, y2_other)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Union
        self_area = self.width * self.height
        other_area = other.width * other.height
        union_area = self_area + other_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def area(self) -> float:
        """Compute box area (normalized)."""
        return self.width * self.height


@dataclass
class Detection:
    """Single object detection result."""

    class_name: ObjectClass
    confidence: float
    bounding_box: BoundingBox
    features: Optional[np.ndarray] = None  # Feature vector for tracking
    depth: Optional[float] = None  # Estimated depth (meters)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "class": self.class_name.value,
            "confidence": float(self.confidence),
            "bounding_box": {
                "x": float(self.bounding_box.x),
                "y": float(self.bounding_box.y),
                "width": float(self.bounding_box.width),
                "height": float(self.bounding_box.height),
            },
            "depth": float(self.depth) if self.depth is not None else None,
            "metadata": self.metadata,
        }


class ObjectDetector:
    """Detect beekeeping objects in images.

    Uses YOLOv8 when available, with graceful fallback to simplified
    color/shape-based detection.

    Example:
        ```python
        detector = ObjectDetector(model_path="beekeeping_yolo.pt")

        # Detect objects in frame
        detections = await detector.detect(frame)

        for det in detections:
            print(f"{det.class_name.value}: {det.confidence:.2f}")
        ```
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_yolo: bool = True,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        """Initialize object detector.

        Args:
            model_path: Path to YOLO model weights (or None for default)
            use_yolo: Whether to use YOLO (if available)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.use_yolo = use_yolo
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = None

        # Class mapping (YOLO class ID -> ObjectClass)
        self.class_mapping = {
            0: ObjectClass.BEEHIVE,
            1: ObjectClass.BEE,
            2: ObjectClass.FRAME,
            3: ObjectClass.BROOD,
            4: ObjectClass.HONEY,
            5: ObjectClass.POLLEN,
            6: ObjectClass.QUEEN,
            7: ObjectClass.VARROA_MITE,
            8: ObjectClass.SMOKER,
            9: ObjectClass.HIVE_TOOL,
        }

        if use_yolo:
            try:
                from ultralytics import YOLO

                # Load model
                if model_path and Path(model_path).exists():
                    self.model = YOLO(model_path)
                else:
                    # Use default YOLOv8 nano model
                    # (would need fine-tuning for beekeeping objects)
                    self.model = YOLO("yolov8n.pt")

                print(f"YOLOv8 loaded: {model_path or 'yolov8n.pt'}")

            except ImportError:
                print("YOLOv8 not available (pip install ultralytics)")
                print("Using simplified color/shape-based detection")
                self.use_yolo = False

            except Exception as e:
                print(f"Error loading YOLOv8: {e}")
                print("Using simplified detection")
                self.use_yolo = False

    async def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in frame.

        Args:
            frame: RGB image (H x W x 3), uint8

        Returns:
            List of detections sorted by confidence (high to low)
        """
        if self.use_yolo and self.model is not None:
            detections = await self._detect_yolo(frame)
        else:
            detections = await self._detect_simplified(frame)

        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return detections

    async def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Detect using YOLOv8.

        Args:
            frame: RGB image

        Returns:
            List of detections
        """
        # Run inference (async to avoid blocking)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.model(frame, verbose=False, conf=self.confidence_threshold)
        )

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                # Extract box info
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Map to ObjectClass
                class_name = self.class_mapping.get(
                    cls_id,
                    ObjectClass.BEEHIVE  # Default fallback
                )

                # Convert to normalized coordinates
                h, w = frame.shape[:2]
                bbox = BoundingBox.from_pixels(
                    int(x1), int(y1), int(x2), int(y2), w, h
                )

                # Extract features (use box region for tracking)
                features = self._extract_features(frame, bbox)

                detection = Detection(
                    class_name=class_name,
                    confidence=conf,
                    bounding_box=bbox,
                    features=features,
                    depth=None,  # TODO: Estimate from depth camera
                    metadata={"method": "yolo"}
                )

                detections.append(detection)

        # Apply NMS to remove duplicates
        detections = self._non_max_suppression(detections)

        return detections

    async def _detect_simplified(self, frame: np.ndarray) -> List[Detection]:
        """Simplified color/shape-based detection.

        Fallback when YOLOv8 is unavailable. Uses color segmentation
        to detect bees (brown/yellow) and basic shape detection.

        Args:
            frame: RGB image

        Returns:
            List of detections
        """
        import cv2

        detections = []
        h, w = frame.shape[:2]

        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Detect bees (yellow/brown color range)
        # HSV ranges for bee colors
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])

        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([20, 255, 200])

        # Create masks
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        mask = cv2.bitwise_or(mask_yellow, mask_brown)

        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Process contours
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by size (bee-sized objects)
            if area < 50 or area > 5000:
                continue

            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)

            # Convert to normalized coordinates
            bbox = BoundingBox.from_pixels(x, y, x + bw, y + bh, w, h)

            # Simple confidence based on area and shape
            aspect_ratio = bw / max(bh, 1)

            # Bees are roughly circular (aspect ratio ~1)
            shape_score = 1.0 - abs(aspect_ratio - 1.0)
            shape_score = max(0, min(1, shape_score))

            confidence = 0.5 + 0.3 * shape_score

            # Extract features
            features = self._extract_features(frame, bbox)

            detection = Detection(
                class_name=ObjectClass.BEE,
                confidence=confidence,
                bounding_box=bbox,
                features=features,
                depth=None,
                metadata={"method": "color", "area": int(area)}
            )

            detections.append(detection)

        return detections

    def _extract_features(
        self,
        frame: np.ndarray,
        bbox: BoundingBox
    ) -> np.ndarray:
        """Extract feature vector from bounding box region.

        Uses color histogram as simple feature descriptor for tracking.

        Args:
            frame: RGB image
            bbox: Bounding box

        Returns:
            Feature vector (64-dimensional)
        """
        import cv2

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.to_pixels(w, h)

        # Extract region
        region = frame[y1:y2, x1:x2]

        if region.size == 0:
            return np.zeros(64)

        # Compute color histogram (RGB, 8 bins each = 24 features)
        hist_r = cv2.calcHist([region], [0], None, [8], [0, 256])
        hist_g = cv2.calcHist([region], [1], None, [8], [0, 256])
        hist_b = cv2.calcHist([region], [2], None, [8], [0, 256])

        # Normalize
        hist_r = hist_r.flatten() / (hist_r.sum() + 1e-6)
        hist_g = hist_g.flatten() / (hist_g.sum() + 1e-6)
        hist_b = hist_b.flatten() / (hist_b.sum() + 1e-6)

        # Concatenate
        features = np.concatenate([hist_r, hist_g, hist_b])

        # Add simple shape features (center, size)
        shape_features = np.array([
            bbox.x,
            bbox.y,
            bbox.width,
            bbox.height,
            bbox.area(),
            bbox.width / max(bbox.height, 1e-6),  # Aspect ratio
        ])

        # Pad to 64 dimensions
        features = np.concatenate([features, shape_features])
        features = np.pad(features, (0, 64 - len(features)), mode='constant')

        return features[:64]

    def _non_max_suppression(
        self,
        detections: List[Detection],
        iou_threshold: Optional[float] = None
    ) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove duplicate detections.

        Args:
            detections: List of detections
            iou_threshold: IoU threshold for suppression (uses self.nms_threshold if None)

        Returns:
            Filtered list of detections
        """
        if not detections:
            return []

        threshold = iou_threshold or self.nms_threshold

        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        # Keep track of which detections to keep
        keep = []

        while detections:
            # Take highest confidence detection
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections
            detections = [
                d for d in detections
                if (
                    d.class_name != best.class_name or
                    d.bounding_box.iou(best.bounding_box) < threshold
                )
            ]

        return keep

    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_labels: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """Draw detections on frame for visualization.

        Args:
            frame: RGB image
            detections: List of detections to draw
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores

        Returns:
            Annotated frame
        """
        import cv2

        # Create copy to avoid modifying original
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Color map for different classes
        color_map = {
            ObjectClass.BEEHIVE: (255, 165, 0),  # Orange
            ObjectClass.BEE: (255, 255, 0),  # Yellow
            ObjectClass.FRAME: (165, 42, 42),  # Brown
            ObjectClass.BROOD: (255, 192, 203),  # Pink
            ObjectClass.HONEY: (255, 215, 0),  # Gold
            ObjectClass.POLLEN: (255, 140, 0),  # Dark orange
            ObjectClass.QUEEN: (255, 0, 0),  # Red
            ObjectClass.VARROA_MITE: (128, 0, 0),  # Dark red
            ObjectClass.SMOKER: (128, 128, 128),  # Gray
            ObjectClass.HIVE_TOOL: (0, 128, 128),  # Teal
        }

        for det in detections:
            # Get bounding box
            x1, y1, x2, y2 = det.bounding_box.to_pixels(w, h)

            # Get color
            color = color_map.get(det.class_name, (0, 255, 0))

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if show_labels or show_confidence:
                label_parts = []

                if show_labels:
                    label_parts.append(det.class_name.value)

                if show_confidence:
                    label_parts.append(f"{det.confidence:.2f}")

                label = " ".join(label_parts)

                # Background rectangle
                (label_w, label_h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1
                )

                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1
                )

                # Text
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        return annotated
