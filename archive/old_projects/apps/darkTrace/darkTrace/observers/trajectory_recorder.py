"""
Trajectory Recorder
===================
Persistent storage and retrieval of semantic trajectories.

The TrajectoryRecorder saves observed trajectories to disk for:
- Later analysis
- Fingerprint generation
- Training trajectory predictors
- LLM behavior comparison
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import logging

from darkTrace.observers.semantic_observer import StateSnapshot


logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """
    A complete semantic trajectory with metadata.

    Represents a full sequence of semantic states from a single
    LLM interaction or text generation session.
    """

    # Metadata
    trajectory_id: str
    model_name: Optional[str] = None
    prompt: Optional[str] = None
    full_text: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Trajectory data
    snapshots: List[StateSnapshot] = field(default_factory=list)

    # Statistics
    total_tokens: int = 0
    avg_velocity: float = 0.0
    max_velocity: float = 0.0
    avg_curvature: float = 0.0
    max_curvature: float = 0.0

    # Tags for organization
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_statistics(self):
        """Compute trajectory statistics from snapshots."""
        if not self.snapshots:
            return

        self.total_tokens = len(self.snapshots)
        self.full_text = "".join(s.text for s in self.snapshots)
        self.start_time = self.snapshots[0].timestamp
        self.end_time = self.snapshots[-1].timestamp

        velocities = [
            s.velocity_magnitude for s in self.snapshots
            if s.velocity_magnitude > 0
        ]
        curvatures = [
            s.curvature for s in self.snapshots
            if s.curvature is not None
        ]

        self.avg_velocity = sum(velocities) / len(velocities) if velocities else 0.0
        self.max_velocity = max(velocities) if velocities else 0.0
        self.avg_curvature = sum(curvatures) / len(curvatures) if curvatures else 0.0
        self.max_curvature = max(curvatures) if curvatures else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trajectory_id": self.trajectory_id,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "full_text": self.full_text,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "total_tokens": self.total_tokens,
            "avg_velocity": self.avg_velocity,
            "max_velocity": self.max_velocity,
            "avg_curvature": self.avg_curvature,
            "max_curvature": self.max_curvature,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class TrajectoryRecorder:
    """
    Persistent storage for semantic trajectories.

    Saves trajectories to disk in JSON or pickle format for later
    analysis and comparison.

    Usage:
        recorder = TrajectoryRecorder(storage_dir="./trajectories")

        # Record trajectory
        trajectory = Trajectory(
            trajectory_id="gpt4_response_001",
            model_name="gpt-4",
            snapshots=observer.get_trajectory()
        )
        recorder.save(trajectory)

        # Load later
        loaded = recorder.load("gpt4_response_001")
    """

    def __init__(
        self,
        storage_dir: str = "./darkTrace_trajectories",
        format: str = "json",  # "json" or "pickle"
    ):
        """
        Initialize trajectory recorder.

        Args:
            storage_dir: Directory to store trajectories
            format: Storage format ("json" or "pickle")
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        if format not in ("json", "pickle"):
            raise ValueError(f"Invalid format: {format}. Use 'json' or 'pickle'")
        self.format = format

        # Index of all trajectories
        self.index_file = self.storage_dir / "index.json"
        self.index = self._load_index()

        logger.info(f"TrajectoryRecorder initialized: {self.storage_dir} ({format} format)")

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load trajectory index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load index: {e}")
        return {}

    def _save_index(self):
        """Save trajectory index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save index: {e}")

    def save(
        self,
        trajectory: Trajectory,
        overwrite: bool = False,
    ) -> Path:
        """
        Save trajectory to disk.

        Args:
            trajectory: Trajectory to save
            overwrite: Overwrite if already exists

        Returns:
            Path to saved file

        Raises:
            FileExistsError: If trajectory exists and overwrite=False
        """
        # Compute statistics
        trajectory.compute_statistics()

        # Check if exists
        if trajectory.trajectory_id in self.index and not overwrite:
            raise FileExistsError(
                f"Trajectory '{trajectory.trajectory_id}' already exists. "
                f"Use overwrite=True to replace."
            )

        # Determine filename
        ext = ".json" if self.format == "json" else ".pkl"
        filename = f"{trajectory.trajectory_id}{ext}"
        filepath = self.storage_dir / filename

        # Save trajectory
        try:
            if self.format == "json":
                with open(filepath, 'w') as f:
                    json.dump(trajectory.to_dict(), f, indent=2)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(trajectory, f)

            # Update index
            self.index[trajectory.trajectory_id] = {
                "filename": filename,
                "model_name": trajectory.model_name,
                "total_tokens": trajectory.total_tokens,
                "start_time": trajectory.start_time.isoformat() if trajectory.start_time else None,
                "end_time": trajectory.end_time.isoformat() if trajectory.end_time else None,
                "tags": trajectory.tags,
            }
            self._save_index()

            logger.info(f"Saved trajectory '{trajectory.trajectory_id}' to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save trajectory: {e}")
            raise

    def load(self, trajectory_id: str) -> Trajectory:
        """
        Load trajectory from disk.

        Args:
            trajectory_id: ID of trajectory to load

        Returns:
            Trajectory object

        Raises:
            FileNotFoundError: If trajectory doesn't exist
        """
        if trajectory_id not in self.index:
            raise FileNotFoundError(f"Trajectory '{trajectory_id}' not found")

        filename = self.index[trajectory_id]["filename"]
        filepath = self.storage_dir / filename

        try:
            if self.format == "json":
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # Reconstruct StateSnapshot objects
                    snapshots = []
                    for snap_data in data.get("snapshots", []):
                        # Convert timestamp string back to datetime
                        snap_data["timestamp"] = datetime.fromisoformat(snap_data["timestamp"])
                        snapshots.append(StateSnapshot(**snap_data))
                    data["snapshots"] = snapshots

                    # Convert time strings back to datetime
                    if data.get("start_time"):
                        data["start_time"] = datetime.fromisoformat(data["start_time"])
                    if data.get("end_time"):
                        data["end_time"] = datetime.fromisoformat(data["end_time"])

                    return Trajectory(**data)
            else:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)

        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}")
            raise

    def list_trajectories(
        self,
        model_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List available trajectories with optional filtering.

        Args:
            model_name: Filter by model name
            tags: Filter by tags (any match)

        Returns:
            List of trajectory metadata dicts
        """
        results = []

        for traj_id, info in self.index.items():
            # Filter by model
            if model_name and info.get("model_name") != model_name:
                continue

            # Filter by tags
            if tags and not any(tag in info.get("tags", []) for tag in tags):
                continue

            results.append({
                "trajectory_id": traj_id,
                **info
            })

        return results

    def delete(self, trajectory_id: str):
        """
        Delete trajectory from storage.

        Args:
            trajectory_id: ID of trajectory to delete
        """
        if trajectory_id not in self.index:
            raise FileNotFoundError(f"Trajectory '{trajectory_id}' not found")

        filename = self.index[trajectory_id]["filename"]
        filepath = self.storage_dir / filename

        try:
            filepath.unlink()
            del self.index[trajectory_id]
            self._save_index()
            logger.info(f"Deleted trajectory '{trajectory_id}'")
        except Exception as e:
            logger.error(f"Failed to delete trajectory: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics across all trajectories.

        Returns:
            Dictionary with aggregate statistics
        """
        total_trajectories = len(self.index)
        models = set()
        total_tokens = 0

        for info in self.index.values():
            if info.get("model_name"):
                models.add(info["model_name"])
            total_tokens += info.get("total_tokens", 0)

        return {
            "total_trajectories": total_trajectories,
            "unique_models": len(models),
            "models": list(models),
            "total_tokens": total_tokens,
            "storage_dir": str(self.storage_dir),
            "format": self.format,
        }
