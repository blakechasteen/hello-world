"""Computer Vision for Beekeeping AR.

This module provides computer vision capabilities for detecting hive components,
tracking bees, and assessing hive health through visual analysis.

Components:
- ObjectDetector: Detect beekeeping objects (hive, bees, frames, etc.)
- BeeTracker: Track individual bees across frames
- HealthAssessor: Assess hive health from visual cues

Created: 2025-11-17
"""

from .object_detector import (
    ObjectDetector,
    ObjectClass,
    Detection,
    BoundingBox,
)
from .bee_tracker import (
    BeeTracker,
    BeeTrack,
)
from .health_assessor import (
    HealthAssessor,
    HealthMetrics,
)

__all__ = [
    # Object Detection
    "ObjectDetector",
    "ObjectClass",
    "Detection",
    "BoundingBox",
    # Bee Tracking
    "BeeTracker",
    "BeeTrack",
    # Health Assessment
    "HealthAssessor",
    "HealthMetrics",
]
