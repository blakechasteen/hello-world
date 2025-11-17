"""Bee Tracking Across Video Frames.

Tracks individual bees across video frames using Hungarian algorithm for
data association and Kalman filtering for motion prediction.

Classes:
- BeeTrack: Single bee track with history
- BeeTracker: Multi-object tracker for bees

Created: 2025-11-17
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
import numpy as np
import asyncio
from collections import deque

from .object_detector import Detection, ObjectClass, BoundingBox


@dataclass
class BeeTrack:
    """Single bee track with historical information."""

    track_id: int
    detections: List[Detection] = field(default_factory=list)
    current_position: Tuple[float, float] = (0.0, 0.0)  # (x, y) normalized
    velocity: Tuple[float, float] = (0.0, 0.0)  # Pixels per frame
    age: int = 0  # Frames since first detection
    time_since_update: int = 0  # Frames since last detection
    activity_level: float = 0.0  # Movement metric (0-1)

    # Kalman filter state (position and velocity)
    state: Optional[np.ndarray] = None  # [x, y, vx, vy]
    covariance: Optional[np.ndarray] = None  # 4x4 covariance matrix

    # Track quality
    confidence_history: List[float] = field(default_factory=list)
    avg_confidence: float = 0.0

    def __post_init__(self):
        """Initialize Kalman filter state."""
        if self.state is None:
            self.state = np.array([
                self.current_position[0],
                self.current_position[1],
                0.0,  # vx
                0.0   # vy
            ])

        if self.covariance is None:
            # Initial uncertainty
            self.covariance = np.eye(4) * 0.1

    def predict(self) -> Tuple[float, float]:
        """Predict next position using Kalman filter.

        Returns:
            Predicted (x, y) position
        """
        # State transition matrix (constant velocity model)
        F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])

        # Process noise
        Q = np.eye(4) * 0.01

        # Predict state
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q

        return (self.state[0], self.state[1])

    def update(self, detection: Detection):
        """Update track with new detection.

        Args:
            detection: New detection to incorporate
        """
        # Measurement matrix (we observe position only)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Measurement noise
        R = np.eye(2) * 0.05

        # Measurement
        z = np.array([
            detection.bounding_box.x,
            detection.bounding_box.y
        ])

        # Kalman update
        y = z - H @ self.state  # Innovation
        S = H @ self.covariance @ H.T + R  # Innovation covariance
        K = self.covariance @ H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ H) @ self.covariance

        # Update track info
        self.detections.append(detection)
        self.current_position = (self.state[0], self.state[1])
        self.velocity = (self.state[2], self.state[3])
        self.age += 1
        self.time_since_update = 0

        # Update confidence history
        self.confidence_history.append(detection.confidence)
        self.avg_confidence = np.mean(self.confidence_history[-10:])  # Last 10

        # Update activity level (based on velocity magnitude)
        velocity_mag = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        # Normalize to 0-1 range (assume max velocity is 0.1 per frame)
        self.activity_level = min(velocity_mag / 0.1, 1.0)

    def mark_missed(self):
        """Mark track as missed (no detection this frame)."""
        self.time_since_update += 1
        self.age += 1

    def is_tentative(self, min_age: int = 3) -> bool:
        """Check if track is still tentative (not confirmed).

        Args:
            min_age: Minimum age for confirmation

        Returns:
            True if tentative
        """
        return self.age < min_age

    def is_dead(self, max_age: int = 30) -> bool:
        """Check if track should be deleted.

        Args:
            max_age: Maximum time since update

        Returns:
            True if track is dead
        """
        return self.time_since_update > max_age

    def get_predicted_bbox(self) -> BoundingBox:
        """Get predicted bounding box.

        Uses last detection's size with predicted position.

        Returns:
            Predicted bounding box
        """
        if not self.detections:
            return BoundingBox(
                self.current_position[0],
                self.current_position[1],
                0.05,  # Default size
                0.05
            )

        last_bbox = self.detections[-1].bounding_box
        pred_pos = self.predict()

        return BoundingBox(
            pred_pos[0],
            pred_pos[1],
            last_bbox.width,
            last_bbox.height
        )


class BeeTracker:
    """Track bees across video frames.

    Uses Hungarian algorithm for data association and Kalman filtering
    for motion prediction.

    Example:
        ```python
        tracker = BeeTracker(max_tracks=1000)

        for frame_number, frame in enumerate(video):
            # Detect bees
            detections = await detector.detect(frame)

            # Update tracks
            tracks = await tracker.update(detections, frame_number)

            # Compute hive activity
            activity = await tracker.compute_hive_activity()
            print(f"Hive activity: {activity:.2f}")
        ```
    """

    def __init__(
        self,
        max_tracks: int = 1000,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        feature_weight: float = 0.3,
    ):
        """Initialize bee tracker.

        Args:
            max_tracks: Maximum number of simultaneous tracks
            max_age: Maximum frames to keep track without update
            min_hits: Minimum detections to confirm track
            iou_threshold: IoU threshold for matching
            feature_weight: Weight for feature similarity (0-1)
        """
        self.max_tracks = max_tracks
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_weight = feature_weight

        self.tracks: Dict[int, BeeTrack] = {}
        self.next_track_id = 0

        # Statistics
        self.total_tracks_created = 0
        self.total_detections_processed = 0

    async def update(
        self,
        detections: List[Detection],
        frame_number: int
    ) -> List[BeeTrack]:
        """Update tracks with new detections.

        Uses Hungarian algorithm for optimal detection-to-track assignment.

        Args:
            detections: List of detections from current frame
            frame_number: Current frame number

        Returns:
            List of active tracks (confirmed only)
        """
        # Filter for bee detections only
        bee_detections = [
            d for d in detections
            if d.class_name == ObjectClass.BEE
        ]

        self.total_detections_processed += len(bee_detections)

        # Predict positions for existing tracks
        for track in self.tracks.values():
            track.predict()

        # Match detections to tracks
        matches, unmatched_detections, unmatched_tracks = await self._associate(
            bee_detections
        )

        # Update matched tracks
        for det_idx, track_id in matches:
            self.tracks[track_id].update(bee_detections[det_idx])

        # Mark unmatched tracks as missed
        for track_id in unmatched_tracks:
            self.tracks[track_id].mark_missed()

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            if len(self.tracks) < self.max_tracks:
                await self._create_track(bee_detections[det_idx])

        # Remove dead tracks
        await self._prune_tracks()

        # Return confirmed tracks only
        return [
            track for track in self.tracks.values()
            if not track.is_tentative(self.min_hits)
        ]

    async def _associate(
        self,
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections with tracks using Hungarian algorithm.

        Args:
            detections: List of detections

        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks)
            - matches: List of (detection_idx, track_id) pairs
            - unmatched_detections: List of detection indices
            - unmatched_tracks: List of track IDs
        """
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(self.tracks.keys())

        # Build cost matrix
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(detections), len(track_ids)))

        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                cost = await self._compute_cost(det, track)
                cost_matrix[i, j] = cost

        # Solve assignment problem using Hungarian algorithm
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ImportError:
            # Fallback to greedy matching
            row_ind, col_ind = self._greedy_matching(cost_matrix)

        # Filter by threshold
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = track_ids.copy()

        for i, j in zip(row_ind, col_ind):
            cost = cost_matrix[i, j]

            # Accept match if cost is below threshold
            # (lower cost = better match)
            if cost < (1.0 - self.iou_threshold):
                track_id = track_ids[j]
                matches.append((i, track_id))

                if i in unmatched_detections:
                    unmatched_detections.remove(i)
                if track_id in unmatched_tracks:
                    unmatched_tracks.remove(track_id)

        return matches, unmatched_detections, unmatched_tracks

    async def _compute_cost(
        self,
        detection: Detection,
        track: BeeTrack
    ) -> float:
        """Compute assignment cost between detection and track.

        Combines IoU (spatial overlap) and feature similarity.

        Args:
            detection: Detection
            track: Track

        Returns:
            Cost (0 = perfect match, 1 = no match)
        """
        # Spatial cost (1 - IoU)
        predicted_bbox = track.get_predicted_bbox()
        iou = detection.bounding_box.iou(predicted_bbox)
        spatial_cost = 1.0 - iou

        # Feature cost (if available)
        if (
            detection.features is not None and
            track.detections and
            track.detections[-1].features is not None
        ):
            # Cosine similarity
            det_features = detection.features
            track_features = track.detections[-1].features

            # Normalize
            det_norm = det_features / (np.linalg.norm(det_features) + 1e-6)
            track_norm = track_features / (np.linalg.norm(track_features) + 1e-6)

            similarity = np.dot(det_norm, track_norm)
            feature_cost = 1.0 - similarity
        else:
            feature_cost = 0.5  # Neutral if features unavailable

        # Weighted combination
        total_cost = (
            (1.0 - self.feature_weight) * spatial_cost +
            self.feature_weight * feature_cost
        )

        return total_cost

    def _greedy_matching(
        self,
        cost_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Greedy matching fallback (when scipy unavailable).

        Args:
            cost_matrix: Cost matrix (detections x tracks)

        Returns:
            Tuple of (row_indices, col_indices)
        """
        n_det, n_track = cost_matrix.shape

        matched_rows = []
        matched_cols = []

        # Greedy: assign lowest cost matches first
        costs = cost_matrix.copy()

        while True:
            # Find minimum cost
            min_idx = np.argmin(costs)
            i, j = np.unravel_index(min_idx, costs.shape)

            if costs[i, j] == np.inf:
                break

            matched_rows.append(i)
            matched_cols.append(j)

            # Mark row and column as used
            costs[i, :] = np.inf
            costs[:, j] = np.inf

        return np.array(matched_rows), np.array(matched_cols)

    async def _create_track(self, detection: Detection):
        """Create new track from detection.

        Args:
            detection: Initial detection
        """
        track_id = self.next_track_id
        self.next_track_id += 1

        track = BeeTrack(
            track_id=track_id,
            current_position=(
                detection.bounding_box.x,
                detection.bounding_box.y
            )
        )

        track.update(detection)

        self.tracks[track_id] = track
        self.total_tracks_created += 1

    async def _prune_tracks(self):
        """Remove dead tracks."""
        dead_tracks = [
            track_id
            for track_id, track in self.tracks.items()
            if track.is_dead(self.max_age)
        ]

        for track_id in dead_tracks:
            del self.tracks[track_id]

    async def compute_hive_activity(self) -> float:
        """Compute overall hive activity level.

        Combines bee count and individual bee activity.

        Returns:
            Activity level (0-1)
        """
        # Get confirmed tracks only
        confirmed_tracks = [
            track for track in self.tracks.values()
            if not track.is_tentative(self.min_hits)
        ]

        if not confirmed_tracks:
            return 0.0

        # Average individual bee activity
        avg_activity = np.mean([
            track.activity_level
            for track in confirmed_tracks
        ])

        # Bee count factor (normalized by typical hive population)
        # Assume 50,000 bees max visible, normalize to 0-1
        bee_count = len(confirmed_tracks)
        bee_count_factor = min(bee_count / 100, 1.0)  # Max 100 tracked

        # Combine (weighted average)
        activity = 0.7 * avg_activity + 0.3 * bee_count_factor

        return activity

    def get_statistics(self) -> Dict[str, any]:
        """Get tracker statistics.

        Returns:
            Dictionary of statistics
        """
        confirmed_tracks = [
            track for track in self.tracks.values()
            if not track.is_tentative(self.min_hits)
        ]

        return {
            "active_tracks": len(self.tracks),
            "confirmed_tracks": len(confirmed_tracks),
            "tentative_tracks": len(self.tracks) - len(confirmed_tracks),
            "total_tracks_created": self.total_tracks_created,
            "total_detections_processed": self.total_detections_processed,
            "avg_confidence": np.mean([
                track.avg_confidence
                for track in confirmed_tracks
            ]) if confirmed_tracks else 0.0,
            "avg_track_length": np.mean([
                track.age
                for track in confirmed_tracks
            ]) if confirmed_tracks else 0.0,
        }

    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_track_id = 0
        self.total_tracks_created = 0
        self.total_detections_processed = 0
