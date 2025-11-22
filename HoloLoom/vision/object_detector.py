"""
Object Detector - Real-time object detection for AR

Supports multiple backends:
    - TensorFlow.js COCO-SSD (web, 80 classes)
    - YOLOv8 (Python backend, 80 classes)
    - Custom models

Created: 2025-11-22
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import numpy as np
from datetime import datetime

from HoloLoom.vision.protocol import (
    DetectedObject,
    BoundingBox,
    ObjectDetectorProtocol,
)


logger = logging.getLogger(__name__)


# ============================================================================
# COCO Class Labels (80 classes)
# ============================================================================

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# ============================================================================
# Object Detector Base
# ============================================================================

class ObjectDetector(ObjectDetectorProtocol):
    """
    Object detector with pluggable backends.

    Graceful degradation: Falls back to mock detector if backend unavailable.
    """

    def __init__(
        self,
        backend: str = "mock",
        confidence_threshold: float = 0.5,
        max_detections: int = 20,
        model_path: Optional[str] = None,
    ):
        """
        Initialize object detector.

        Args:
            backend: "tfjs", "yolo", "mock"
            confidence_threshold: Minimum confidence (0.0-1.0)
            max_detections: Maximum objects to return
            model_path: Path to custom model (optional)
        """
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.model_path = model_path
        self.model = None
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize detector backend"""
        try:
            if self.backend == "yolo":
                return await self._initialize_yolo()
            elif self.backend == "tfjs":
                # TensorFlow.js runs in browser, not here
                logger.info("TensorFlow.js detector runs in browser")
                self.initialized = True
                return True
            elif self.backend == "mock":
                logger.info("Using mock object detector")
                self.initialized = True
                return True
            else:
                logger.warning(f"Unknown backend: {self.backend}, using mock")
                self.backend = "mock"
                self.initialized = True
                return True

        except Exception as e:
            logger.error(f"Failed to initialize {self.backend}: {e}")
            logger.warning("Falling back to mock detector")
            self.backend = "mock"
            self.initialized = True
            return True  # Always succeed via fallback

    async def _initialize_yolo(self) -> bool:
        """Initialize YOLO backend (requires ultralytics)"""
        try:
            from ultralytics import YOLO

            # Load model
            model_path = self.model_path or "yolov8n.pt"  # Nano model (6.3MB)
            self.model = YOLO(model_path)
            self.initialized = True
            logger.info(f"YOLO model loaded: {model_path}")
            return True

        except ImportError:
            logger.warning("ultralytics not installed, install with: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    async def process_frame(self, frame: np.ndarray) -> List[DetectedObject]:
        """Process frame (delegates to detect_objects)"""
        return await self.detect_objects(frame, self.confidence_threshold)

    async def detect_objects(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[DetectedObject]:
        """
        Detect objects in frame.

        Args:
            frame: RGB image (HxWx3)
            confidence_threshold: Minimum confidence

        Returns:
            List of detected objects
        """
        if not self.initialized:
            await self.initialize()

        if self.backend == "yolo":
            return await self._detect_yolo(frame, confidence_threshold)
        elif self.backend == "mock":
            return self._detect_mock(frame, confidence_threshold)
        else:
            return []

    async def _detect_yolo(
        self,
        frame: np.ndarray,
        confidence_threshold: float
    ) -> List[DetectedObject]:
        """YOLO detection"""
        if not self.model:
            return []

        # Run inference
        results = self.model(frame, conf=confidence_threshold, verbose=False)

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes

            for i, box in enumerate(boxes):
                # Get bbox (xyxy format)
                xyxy = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = xyxy

                # Normalize to 0-1
                h, w = frame.shape[:2]
                bbox = BoundingBox(
                    x_min=x_min / w,
                    y_min=y_min / h,
                    x_max=x_max / w,
                    y_max=y_max / h,
                )

                # Get class and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

                detection = DetectedObject(
                    id=f"obj_{i}_{datetime.now().timestamp()}",
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    class_id=class_id,
                )

                detections.append(detection)

                # Limit detections
                if len(detections) >= self.max_detections:
                    break

        return detections

    def _detect_mock(
        self,
        frame: np.ndarray,
        confidence_threshold: float
    ) -> List[DetectedObject]:
        """Mock detection for testing"""
        # Return some dummy objects for demonstration
        return [
            DetectedObject(
                id="mock_obj_1",
                label="toolbox",
                confidence=0.85,
                bbox=BoundingBox(0.3, 0.4, 0.5, 0.7),
                class_id=0,
                metadata={"mock": True},
            ),
            DetectedObject(
                id="mock_obj_2",
                label="drill",
                confidence=0.92,
                bbox=BoundingBox(0.6, 0.3, 0.8, 0.6),
                class_id=1,
                metadata={"mock": True},
            ),
        ]

    async def cleanup(self) -> None:
        """Clean up resources"""
        self.model = None
        self.initialized = False

    def get_capabilities(self) -> Dict[str, bool]:
        """Get detector capabilities"""
        return {
            "object_detection": True,
            "real_time": self.backend in ["yolo", "tfjs"],
            "80_classes": self.backend in ["yolo", "tfjs"],
            "custom_models": self.backend == "yolo",
        }


# ============================================================================
# Factory
# ============================================================================

def create_object_detector(
    backend: str = "mock",
    **kwargs
) -> ObjectDetector:
    """
    Factory for creating object detectors.

    Args:
        backend: "yolo", "tfjs", "mock"
        **kwargs: Backend-specific options

    Returns:
        ObjectDetector instance
    """
    return ObjectDetector(backend=backend, **kwargs)


# ============================================================================
# Helper Functions
# ============================================================================

def filter_objects_by_type(
    objects: List[DetectedObject],
    labels: List[str]
) -> List[DetectedObject]:
    """Filter objects by label"""
    return [obj for obj in objects if obj.label in labels]


def get_largest_object(
    objects: List[DetectedObject]
) -> Optional[DetectedObject]:
    """Get object with largest bounding box"""
    if not objects:
        return None

    return max(objects, key=lambda obj: obj.bbox.area())


def get_closest_to_center(
    objects: List[DetectedObject],
    frame_center: Tuple[float, float] = (0.5, 0.5)
) -> Optional[DetectedObject]:
    """Get object closest to frame center"""
    if not objects:
        return None

    def distance_to_center(obj: DetectedObject) -> float:
        cx, cy = obj.bbox.center()
        return ((cx - frame_center[0])**2 + (cy - frame_center[1])**2)**0.5

    return min(objects, key=distance_to_center)


def non_max_suppression(
    objects: List[DetectedObject],
    iou_threshold: float = 0.5
) -> List[DetectedObject]:
    """
    Apply non-maximum suppression to remove overlapping boxes.

    Args:
        objects: List of detected objects
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered list of objects
    """
    if not objects:
        return []

    # Sort by confidence (descending)
    sorted_objs = sorted(objects, key=lambda x: x.confidence, reverse=True)

    keep = []
    while sorted_objs:
        # Keep highest confidence object
        current = sorted_objs.pop(0)
        keep.append(current)

        # Remove overlapping objects
        sorted_objs = [
            obj for obj in sorted_objs
            if current.bbox.iou(obj.bbox) < iou_threshold
        ]

    return keep
