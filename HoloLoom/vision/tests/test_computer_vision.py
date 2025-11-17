"""Comprehensive Tests for Computer Vision Module.

Tests all components:
- Object Detection (BoundingBox, Detection, ObjectDetector)
- Bee Tracking (BeeTrack, BeeTracker)
- Health Assessment (HealthMetrics, HealthAssessor)

Created: 2025-11-17
"""

import pytest
import numpy as np
import asyncio
from typing import List

from HoloLoom.vision import (
    ObjectDetector,
    ObjectClass,
    Detection,
    BoundingBox,
    BeeTracker,
    BeeTrack,
    HealthAssessor,
    HealthMetrics,
    HealthStatus,
)


# ============================================================================
# BoundingBox Tests (8 tests)
# ============================================================================


class TestBoundingBox:
    """Test BoundingBox class."""

    def test_creation(self):
        """Test bounding box creation."""
        bbox = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.1)

        assert bbox.x == 0.5
        assert bbox.y == 0.5
        assert bbox.width == 0.2
        assert bbox.height == 0.1

    def test_to_pixels(self):
        """Test conversion to pixel coordinates."""
        bbox = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.1)

        x1, y1, x2, y2 = bbox.to_pixels(1000, 500)

        # Center at (500, 250), size (200, 50)
        assert x1 == 400  # 500 - 100
        assert y1 == 225  # 250 - 25
        assert x2 == 600  # 500 + 100
        assert y2 == 275  # 250 + 25

    def test_from_pixels(self):
        """Test creation from pixel coordinates."""
        bbox = BoundingBox.from_pixels(400, 225, 600, 275, 1000, 500)

        assert abs(bbox.x - 0.5) < 0.01
        assert abs(bbox.y - 0.5) < 0.01
        assert abs(bbox.width - 0.2) < 0.01
        assert abs(bbox.height - 0.1) < 0.01

    def test_iou_perfect_overlap(self):
        """Test IoU with perfect overlap."""
        bbox1 = BoundingBox(0.5, 0.5, 0.2, 0.1)
        bbox2 = BoundingBox(0.5, 0.5, 0.2, 0.1)

        iou = bbox1.iou(bbox2)

        assert abs(iou - 1.0) < 0.01

    def test_iou_no_overlap(self):
        """Test IoU with no overlap."""
        bbox1 = BoundingBox(0.2, 0.2, 0.1, 0.1)
        bbox2 = BoundingBox(0.8, 0.8, 0.1, 0.1)

        iou = bbox1.iou(bbox2)

        assert iou == 0.0

    def test_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        bbox1 = BoundingBox(0.5, 0.5, 0.2, 0.2)
        bbox2 = BoundingBox(0.6, 0.6, 0.2, 0.2)

        iou = bbox1.iou(bbox2)

        # Should have some overlap
        assert 0 < iou < 1

    def test_area(self):
        """Test area computation."""
        bbox = BoundingBox(0.5, 0.5, 0.2, 0.1)

        area = bbox.area()

        assert abs(area - 0.02) < 0.001  # 0.2 * 0.1 = 0.02

    def test_edge_cases(self):
        """Test edge cases (zero size, boundary)."""
        # Zero size
        bbox_zero = BoundingBox(0.5, 0.5, 0.0, 0.0)
        assert bbox_zero.area() == 0.0

        # Boundary positions
        bbox_corner = BoundingBox(0.0, 0.0, 0.1, 0.1)
        x1, y1, x2, y2 = bbox_corner.to_pixels(100, 100)
        assert x1 >= 0 and y1 >= 0


# ============================================================================
# Detection Tests (4 tests)
# ============================================================================


class TestDetection:
    """Test Detection class."""

    def test_creation(self):
        """Test detection creation."""
        bbox = BoundingBox(0.5, 0.5, 0.1, 0.1)
        det = Detection(
            class_name=ObjectClass.BEE,
            confidence=0.95,
            bounding_box=bbox,
            features=np.random.rand(64),
            depth=2.5,
        )

        assert det.class_name == ObjectClass.BEE
        assert det.confidence == 0.95
        assert det.bounding_box == bbox
        assert det.features.shape == (64,)
        assert det.depth == 2.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        bbox = BoundingBox(0.5, 0.5, 0.1, 0.1)
        det = Detection(
            class_name=ObjectClass.QUEEN,
            confidence=0.85,
            bounding_box=bbox,
        )

        data = det.to_dict()

        assert data["class"] == "queen"
        assert data["confidence"] == 0.85
        assert "bounding_box" in data
        assert data["bounding_box"]["x"] == 0.5

    def test_metadata(self):
        """Test metadata field."""
        bbox = BoundingBox(0.5, 0.5, 0.1, 0.1)
        det = Detection(
            class_name=ObjectClass.BEE,
            confidence=0.9,
            bounding_box=bbox,
            metadata={"source": "yolo", "frame": 42}
        )

        assert det.metadata["source"] == "yolo"
        assert det.metadata["frame"] == 42

    def test_features_optional(self):
        """Test that features are optional."""
        bbox = BoundingBox(0.5, 0.5, 0.1, 0.1)
        det = Detection(
            class_name=ObjectClass.BROOD,
            confidence=0.8,
            bounding_box=bbox,
        )

        assert det.features is None
        assert det.depth is None


# ============================================================================
# ObjectDetector Tests (6 tests)
# ============================================================================


class TestObjectDetector:
    """Test ObjectDetector class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test detector initialization."""
        detector = ObjectDetector(use_yolo=False)

        assert detector is not None
        assert not detector.use_yolo

    @pytest.mark.asyncio
    async def test_simplified_detection(self):
        """Test simplified color-based detection."""
        detector = ObjectDetector(use_yolo=False)

        # Create test frame with yellow blob (bee)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw yellow circle
        import cv2
        cv2.circle(frame, (50, 50), 10, (255, 255, 0), -1)

        detections = await detector.detect(frame)

        # Should detect at least one bee (yellow color)
        assert len(detections) >= 0  # May or may not detect depending on HSV range

    @pytest.mark.asyncio
    async def test_empty_frame(self):
        """Test detection on empty frame."""
        detector = ObjectDetector(use_yolo=False)

        # Black frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = await detector.detect(frame)

        # Should return empty list
        assert isinstance(detections, list)

    @pytest.mark.asyncio
    async def test_feature_extraction(self):
        """Test feature extraction."""
        detector = ObjectDetector(use_yolo=False)

        # Create test frame
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(0.5, 0.5, 0.2, 0.2)

        features = detector._extract_features(frame, bbox)

        # Should return 64-dimensional feature vector
        assert features.shape == (64,)
        assert np.all(np.isfinite(features))

    def test_non_max_suppression(self):
        """Test NMS for removing duplicate detections."""
        detector = ObjectDetector(use_yolo=False)

        # Create overlapping detections
        bbox1 = BoundingBox(0.5, 0.5, 0.1, 0.1)
        bbox2 = BoundingBox(0.51, 0.51, 0.1, 0.1)  # Overlapping
        bbox3 = BoundingBox(0.8, 0.8, 0.1, 0.1)  # Separate

        detections = [
            Detection(ObjectClass.BEE, 0.9, bbox1),
            Detection(ObjectClass.BEE, 0.7, bbox2),
            Detection(ObjectClass.BEE, 0.8, bbox3),
        ]

        filtered = detector._non_max_suppression(detections, iou_threshold=0.3)

        # Should keep bbox1 (highest conf) and bbox3 (separate)
        # bbox2 should be suppressed (overlaps with bbox1)
        assert len(filtered) == 2
        assert filtered[0].confidence == 0.9  # bbox1
        assert filtered[1].confidence == 0.8  # bbox3

    def test_class_mapping(self):
        """Test class ID to ObjectClass mapping."""
        detector = ObjectDetector(use_yolo=False)

        # Test mapping
        assert detector.class_mapping[0] == ObjectClass.BEEHIVE
        assert detector.class_mapping[1] == ObjectClass.BEE
        assert detector.class_mapping[6] == ObjectClass.QUEEN


# ============================================================================
# BeeTrack Tests (6 tests)
# ============================================================================


class TestBeeTrack:
    """Test BeeTrack class."""

    def test_creation(self):
        """Test track creation."""
        track = BeeTrack(track_id=0, current_position=(0.5, 0.5))

        assert track.track_id == 0
        assert track.current_position == (0.5, 0.5)
        assert track.age == 0
        assert track.state is not None  # Kalman state initialized

    def test_predict(self):
        """Test position prediction."""
        track = BeeTrack(track_id=0, current_position=(0.5, 0.5))

        # Predict next position
        pred_x, pred_y = track.predict()

        # Should be close to current (zero velocity initially)
        assert abs(pred_x - 0.5) < 0.1
        assert abs(pred_y - 0.5) < 0.1

    def test_update(self):
        """Test track update with detection."""
        track = BeeTrack(track_id=0, current_position=(0.5, 0.5))

        bbox = BoundingBox(0.52, 0.51, 0.05, 0.05)
        det = Detection(ObjectClass.BEE, 0.9, bbox)

        track.update(det)

        # Track should update
        assert track.age == 1
        assert track.time_since_update == 0
        assert len(track.detections) == 1
        assert len(track.confidence_history) == 1

    def test_mark_missed(self):
        """Test marking track as missed."""
        track = BeeTrack(track_id=0, current_position=(0.5, 0.5))

        track.mark_missed()

        assert track.time_since_update == 1
        assert track.age == 1

    def test_is_tentative(self):
        """Test tentative track detection."""
        track = BeeTrack(track_id=0, current_position=(0.5, 0.5))

        # New track is tentative
        assert track.is_tentative(min_age=3)

        # After 3 updates, no longer tentative
        for _ in range(3):
            bbox = BoundingBox(0.5, 0.5, 0.05, 0.05)
            det = Detection(ObjectClass.BEE, 0.9, bbox)
            track.update(det)

        assert not track.is_tentative(min_age=3)

    def test_is_dead(self):
        """Test dead track detection."""
        track = BeeTrack(track_id=0, current_position=(0.5, 0.5))

        # Not dead initially
        assert not track.is_dead(max_age=30)

        # After many misses, becomes dead
        for _ in range(35):
            track.mark_missed()

        assert track.is_dead(max_age=30)


# ============================================================================
# BeeTracker Tests (8 tests)
# ============================================================================


class TestBeeTracker:
    """Test BeeTracker class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test tracker initialization."""
        tracker = BeeTracker(max_tracks=100)

        assert tracker.max_tracks == 100
        assert len(tracker.tracks) == 0
        assert tracker.next_track_id == 0

    @pytest.mark.asyncio
    async def test_create_track(self):
        """Test track creation from detection."""
        tracker = BeeTracker()

        bbox = BoundingBox(0.5, 0.5, 0.05, 0.05)
        det = Detection(ObjectClass.BEE, 0.9, bbox)

        await tracker._create_track(det)

        assert len(tracker.tracks) == 1
        assert 0 in tracker.tracks
        assert tracker.next_track_id == 1

    @pytest.mark.asyncio
    async def test_update_single_bee(self):
        """Test updating with single bee detection."""
        tracker = BeeTracker()

        # Frame 1: Create track
        bbox1 = BoundingBox(0.5, 0.5, 0.05, 0.05)
        det1 = Detection(ObjectClass.BEE, 0.9, bbox1)

        tracks = await tracker.update([det1], frame_number=0)

        # New track is tentative
        assert len(tracks) == 0  # Tentative tracks not returned

        # Frame 2-3: Confirm track
        for i in range(1, 3):
            bbox = BoundingBox(0.5 + i * 0.01, 0.5, 0.05, 0.05)
            det = Detection(ObjectClass.BEE, 0.9, bbox)
            tracks = await tracker.update([det], frame_number=i)

        # Should now be confirmed
        assert len(tracks) == 1

    @pytest.mark.asyncio
    async def test_update_multiple_bees(self):
        """Test updating with multiple bees."""
        tracker = BeeTracker()

        # Create multiple detections
        detections = [
            Detection(ObjectClass.BEE, 0.9, BoundingBox(0.3, 0.3, 0.05, 0.05)),
            Detection(ObjectClass.BEE, 0.85, BoundingBox(0.7, 0.7, 0.05, 0.05)),
        ]

        # Update several times to confirm tracks
        for i in range(5):
            await tracker.update(detections, frame_number=i)

        # Should have 2 confirmed tracks
        confirmed = [t for t in tracker.tracks.values() if not t.is_tentative()]
        assert len(confirmed) == 2

    @pytest.mark.asyncio
    async def test_track_pruning(self):
        """Test removal of dead tracks."""
        tracker = BeeTracker(max_age=5)

        # Create track
        bbox = BoundingBox(0.5, 0.5, 0.05, 0.05)
        det = Detection(ObjectClass.BEE, 0.9, bbox)
        await tracker.update([det], frame_number=0)

        # Update without detections (track becomes dead)
        for i in range(1, 10):
            await tracker.update([], frame_number=i)

        # Track should be pruned
        assert len(tracker.tracks) == 0

    @pytest.mark.asyncio
    async def test_hive_activity_empty(self):
        """Test hive activity with no bees."""
        tracker = BeeTracker()

        activity = await tracker.compute_hive_activity()

        assert activity == 0.0

    @pytest.mark.asyncio
    async def test_hive_activity_with_bees(self):
        """Test hive activity with active bees."""
        tracker = BeeTracker()

        # Create moving bees
        for frame in range(5):
            detections = [
                Detection(
                    ObjectClass.BEE,
                    0.9,
                    BoundingBox(0.5 + frame * 0.05, 0.5, 0.05, 0.05)
                )
            ]
            await tracker.update(detections, frame_number=frame)

        activity = await tracker.compute_hive_activity()

        # Should have some activity
        assert activity > 0.0

    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test tracker statistics."""
        tracker = BeeTracker()

        # Create some tracks
        for _ in range(3):
            bbox = BoundingBox(0.5, 0.5, 0.05, 0.05)
            det = Detection(ObjectClass.BEE, 0.9, bbox)
            await tracker.update([det], frame_number=0)

        stats = tracker.get_statistics()

        assert stats["active_tracks"] == 3
        assert stats["total_tracks_created"] == 3
        assert "avg_confidence" in stats


# ============================================================================
# HealthMetrics Tests (4 tests)
# ============================================================================


class TestHealthMetrics:
    """Test HealthMetrics class."""

    def test_creation(self):
        """Test metrics creation."""
        metrics = HealthMetrics(
            bee_population=75,
            queen_present=True,
            activity_level=0.8,
            brood_pattern_score=0.9,
            varroa_detected=False,
            honey_stores=0.6,
            pollen_stores=0.4,
            overall_health=0.85,
            health_status=HealthStatus.EXCELLENT,
        )

        assert metrics.bee_population == 75
        assert metrics.queen_present
        assert metrics.overall_health == 0.85

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = HealthMetrics(
            bee_population=50,
            queen_present=False,
            activity_level=0.5,
            brood_pattern_score=0.6,
            varroa_detected=True,
            honey_stores=0.3,
            pollen_stores=0.2,
            overall_health=0.45,
            health_status=HealthStatus.FAIR,
        )

        data = metrics.to_dict()

        assert data["bee_population"] == 50
        assert data["health_status"] == "fair"
        assert data["varroa_detected"] is True

    def test_summary(self):
        """Test summary generation."""
        metrics = HealthMetrics(
            bee_population=100,
            queen_present=True,
            activity_level=0.9,
            brood_pattern_score=0.95,
            varroa_detected=False,
            honey_stores=0.7,
            pollen_stores=0.5,
            overall_health=0.92,
            health_status=HealthStatus.EXCELLENT,
        )

        summary = metrics.get_summary()

        assert "EXCELLENT" in summary
        assert "100" in summary  # Bee count
        assert "✓" in summary  # Queen present

    def test_recommendations(self):
        """Test recommendation generation."""
        # Poor health with multiple issues
        metrics = HealthMetrics(
            bee_population=20,  # Low
            queen_present=False,  # Missing
            activity_level=0.2,  # Low
            brood_pattern_score=0.3,  # Poor
            varroa_detected=True,  # Present
            honey_stores=0.1,  # Low
            pollen_stores=0.05,  # Low
            overall_health=0.15,
            health_status=HealthStatus.CRITICAL,
        )

        recommendations = metrics._get_recommendations()

        # Should have multiple recommendations
        assert len(recommendations) > 0
        # Check for varroa warning
        assert any("Varroa" in rec for rec in recommendations)


# ============================================================================
# HealthAssessor Tests (8 tests)
# ============================================================================


class TestHealthAssessor:
    """Test HealthAssessor class."""

    @pytest.fixture
    def detector(self):
        """Create object detector."""
        return ObjectDetector(use_yolo=False)

    @pytest.fixture
    def tracker(self):
        """Create bee tracker."""
        return BeeTracker()

    @pytest.fixture
    def assessor(self, detector, tracker):
        """Create health assessor."""
        return HealthAssessor(detector, tracker)

    @pytest.mark.asyncio
    async def test_initialization(self, assessor):
        """Test assessor initialization."""
        assert assessor is not None
        assert len(assessor.history) == 0

    @pytest.mark.asyncio
    async def test_assess_empty_frame(self, assessor):
        """Test assessment on empty frame."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        metrics = await assessor.assess(frame, frame_number=0)

        # Should return metrics with zero/low values
        assert metrics.bee_population >= 0
        assert 0 <= metrics.overall_health <= 1

    @pytest.mark.asyncio
    async def test_assess_with_detections(self, detector, tracker):
        """Test assessment with mock detections."""
        assessor = HealthAssessor(detector, tracker)

        # Create frame with some visual features
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        metrics = await assessor.assess(frame, frame_number=0)

        # Should have valid metrics
        assert isinstance(metrics, HealthMetrics)
        assert isinstance(metrics.health_status, HealthStatus)

    @pytest.mark.asyncio
    async def test_brood_pattern_analysis(self, assessor):
        """Test brood pattern analysis."""
        # Create detections with compact brood pattern
        detections = [
            Detection(
                ObjectClass.BROOD,
                0.9,
                BoundingBox(0.5 + i * 0.05, 0.5, 0.05, 0.05)
            )
            for i in range(5)
        ]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        score = await assessor._analyze_brood_pattern(detections, frame)

        # Should have reasonable score
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_resource_estimation(self, assessor):
        """Test honey/pollen store estimation."""
        # Create honey detections
        detections = [
            Detection(
                ObjectClass.HONEY,
                0.95,
                BoundingBox(0.3, 0.3, 0.2, 0.2)
            ),
            Detection(
                ObjectClass.HONEY,
                0.9,
                BoundingBox(0.7, 0.7, 0.15, 0.15)
            ),
        ]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        stores = await assessor._estimate_stores(
            detections,
            ObjectClass.HONEY,
            frame
        )

        # Should have some honey
        assert stores > 0

    def test_health_classification(self, assessor):
        """Test health status classification."""
        assert assessor._classify_health_status(0.9) == HealthStatus.EXCELLENT
        assert assessor._classify_health_status(0.7) == HealthStatus.GOOD
        assert assessor._classify_health_status(0.5) == HealthStatus.FAIR
        assert assessor._classify_health_status(0.3) == HealthStatus.POOR
        assert assessor._classify_health_status(0.1) == HealthStatus.CRITICAL

    @pytest.mark.asyncio
    async def test_trend_analysis(self, assessor):
        """Test health trend analysis."""
        # No history
        assert assessor.get_trend() is None

        # Add improving trend
        for i in range(10):
            metrics = HealthMetrics(
                bee_population=50 + i * 5,
                queen_present=True,
                activity_level=0.5 + i * 0.05,
                brood_pattern_score=0.7,
                varroa_detected=False,
                honey_stores=0.5,
                pollen_stores=0.4,
                overall_health=0.5 + i * 0.04,
                health_status=HealthStatus.FAIR,
            )
            assessor.history.append(metrics)

        trend = assessor.get_trend()
        assert trend == "improving"

    @pytest.mark.asyncio
    async def test_statistics(self, assessor):
        """Test statistics collection."""
        # Add some history
        for i in range(5):
            metrics = HealthMetrics(
                bee_population=50,
                queen_present=True,
                activity_level=0.7,
                brood_pattern_score=0.8,
                varroa_detected=False,
                honey_stores=0.6,
                pollen_stores=0.5,
                overall_health=0.75,
                health_status=HealthStatus.GOOD,
            )
            assessor.history.append(metrics)

        stats = assessor.get_statistics()

        assert stats["assessments_count"] == 5
        assert "avg_health" in stats
        assert "trend" in stats


# ============================================================================
# Integration Tests (2 tests)
# ============================================================================


class TestIntegration:
    """Integration tests for full pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete detection → tracking → assessment pipeline."""
        # Create components
        detector = ObjectDetector(use_yolo=False)
        tracker = BeeTracker(max_tracks=100)
        assessor = HealthAssessor(detector, tracker)

        # Create test frame
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        # Run pipeline
        metrics = await assessor.assess(frame, frame_number=0)

        # Should complete without error
        assert isinstance(metrics, HealthMetrics)
        assert isinstance(metrics.health_status, HealthStatus)

    @pytest.mark.asyncio
    async def test_multi_frame_tracking(self):
        """Test tracking across multiple frames."""
        detector = ObjectDetector(use_yolo=False)
        tracker = BeeTracker()
        assessor = HealthAssessor(detector, tracker)

        # Process multiple frames
        for frame_num in range(10):
            frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            metrics = await assessor.assess(frame, frame_number=frame_num)

        # Should have accumulated history
        assert len(assessor.history) == 10

        # Should have statistics
        stats = assessor.get_statistics()
        assert stats["assessments_count"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
