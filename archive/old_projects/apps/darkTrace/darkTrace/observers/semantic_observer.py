"""
Semantic Observer
=================
Real-time semantic state monitoring using HoloLoom semantic calculus.

The SemanticObserver tracks LLM outputs in 244D semantic space,
performing smart selection to 36D for analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

try:
    from HoloLoom.semantic_calculus.analyzer import SemanticCalculusAnalyzer
    from HoloLoom.semantic_calculus.config import SemanticCalculusConfig
    from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS
except ImportError:
    SemanticCalculusAnalyzer = None
    SemanticCalculusConfig = None
    EXTENDED_244_DIMENSIONS = None
    logging.warning(
        "HoloLoom semantic calculus not available. "
        "Install with: pip install -e ../../HoloLoom"
    )

from darkTrace.config import DarkTraceConfig, ObserverConfig


logger = logging.getLogger(__name__)


@dataclass
class StateSnapshot:
    """
    Snapshot of semantic state at a specific point in time.

    Captures the semantic position, velocity, and other metrics
    for a single token or chunk of text.
    """

    # Metadata
    timestamp: datetime
    token_index: int
    text: str

    # Position in semantic space (36D)
    position: List[float]  # 36D position vector

    # Velocity and acceleration
    velocity: Optional[List[float]] = None  # 36D velocity vector
    velocity_magnitude: float = 0.0
    acceleration: Optional[List[float]] = None  # 36D acceleration vector
    acceleration_magnitude: float = 0.0

    # Curvature (trajectory bending)
    curvature: Optional[float] = None

    # Dominant dimensions
    dominant_dimensions: List[str] = field(default_factory=list)
    dimension_scores: Dict[str, float] = field(default_factory=dict)

    # Ethical valence (if computed)
    ethical_valence: Optional[float] = None
    ethical_dimensions: Optional[Dict[str, float]] = None

    # Flow metrics (attractor strength, divergence)
    flow_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "token_index": self.token_index,
            "text": self.text,
            "position": self.position,
            "velocity": self.velocity,
            "velocity_magnitude": self.velocity_magnitude,
            "acceleration": self.acceleration,
            "acceleration_magnitude": self.acceleration_magnitude,
            "curvature": self.curvature,
            "dominant_dimensions": self.dominant_dimensions,
            "dimension_scores": self.dimension_scores,
            "ethical_valence": self.ethical_valence,
            "ethical_dimensions": self.ethical_dimensions,
            "flow_metrics": self.flow_metrics,
        }


class SemanticObserver:
    """
    Real-time semantic state observer.

    Uses HoloLoom semantic calculus to track LLM outputs in 244D→36D
    semantic space, computing velocity, acceleration, curvature, and
    other geometric properties.

    Usage:
        config = DarkTraceConfig.narrative()
        observer = SemanticObserver(config)

        # Observe tokens incrementally
        for token in tokens:
            state = observer.observe(token)
            print(f"Position: {state.position[:3]}...")
            print(f"Velocity: {state.velocity_magnitude:.3f}")
    """

    def __init__(
        self,
        config: DarkTraceConfig,
    ):
        """
        Initialize semantic observer.

        Args:
            config: DarkTraceConfig with observer settings
        """
        self.config = config
        self.observer_config = config.observer

        # Check if HoloLoom is available
        if SemanticCalculusAnalyzer is None:
            raise ImportError(
                "HoloLoom semantic calculus not available. "
                "Install with: pip install -e ../../HoloLoom"
            )

        # Create semantic calculus config
        self.sem_config = self._create_semantic_config()

        # Initialize analyzer
        self.analyzer = SemanticCalculusAnalyzer(self.sem_config)

        # Trajectory state
        self.token_count = 0
        self.full_text = ""
        self.snapshots: List[StateSnapshot] = []

        # Previous state for computing derivatives
        self.prev_position: Optional[List[float]] = None
        self.prev_velocity: Optional[List[float]] = None

        logger.info(
            f"SemanticObserver initialized with {self.observer_config.dimensions}D "
            f"({self.observer_config.selection_strategy} strategy, "
            f"{self.observer_config.domain} domain)"
        )

    def _create_semantic_config(self) -> 'SemanticCalculusConfig':
        """Create SemanticCalculusConfig from observer settings."""
        # Map domain to factory method
        domain = self.observer_config.domain
        strategy = self.observer_config.selection_strategy

        if domain == "narrative":
            config = SemanticCalculusConfig.fused_narrative()
        elif domain == "dialogue":
            config = SemanticCalculusConfig.fused_dialogue()
        elif domain == "technical":
            config = SemanticCalculusConfig.fused_technical()
        else:
            config = SemanticCalculusConfig.fused_general()

        # Override with observer settings
        config.dimensions = self.observer_config.dimensions
        config.selection_strategy = strategy
        config.domain = domain
        config.compute_flow = self.observer_config.compute_flow
        config.compute_curvature = self.observer_config.compute_curvature
        config.compute_ethics = self.observer_config.compute_ethics

        return config

    def observe(self, text: str) -> StateSnapshot:
        """
        Observe a token/chunk and return semantic state.

        Args:
            text: Token or text chunk to observe

        Returns:
            StateSnapshot with semantic state at this point
        """
        # Update full text
        self.full_text += text
        self.token_count += 1

        # Analyze semantic state
        result = self.analyzer.analyze(text)

        # Extract position (embedding + spectral features)
        position = result.get("embedding", [0.0] * self.observer_config.dimensions)
        if len(position) < self.observer_config.dimensions:
            position = position + [0.0] * (self.observer_config.dimensions - len(position))
        position = position[:self.observer_config.dimensions]

        # Compute velocity (change in position)
        velocity = None
        velocity_magnitude = 0.0
        if self.prev_position is not None:
            velocity = [
                position[i] - self.prev_position[i]
                for i in range(len(position))
            ]
            velocity_magnitude = sum(v**2 for v in velocity) ** 0.5

        # Compute acceleration (change in velocity)
        acceleration = None
        acceleration_magnitude = 0.0
        if self.prev_velocity is not None and velocity is not None:
            acceleration = [
                velocity[i] - self.prev_velocity[i]
                for i in range(len(velocity))
            ]
            acceleration_magnitude = sum(a**2 for a in acceleration) ** 0.5

        # Compute curvature if enabled
        curvature = None
        if self.observer_config.compute_curvature and velocity_magnitude > 0:
            # Curvature = |acceleration × velocity| / |velocity|^3
            if acceleration is not None:
                # Simplified curvature in high-D space
                curvature = acceleration_magnitude / (velocity_magnitude ** 2 + 1e-8)

        # Extract dominant dimensions
        dimension_scores = result.get("dimension_scores", {})
        sorted_dims = sorted(
            dimension_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        dominant_dimensions = [dim for dim, _ in sorted_dims[:5]]

        # Extract ethics if computed
        ethical_valence = None
        ethical_dimensions = None
        if self.observer_config.compute_ethics:
            ethical_valence = result.get("ethical_valence", 0.0)
            ethical_dimensions = result.get("ethical_dimensions", {})

        # Extract flow metrics if computed
        flow_metrics = None
        if self.observer_config.compute_flow:
            flow_metrics = {
                "divergence": result.get("divergence", 0.0),
                "curl": result.get("curl", 0.0),
                "attractor_strength": result.get("attractor_strength", 0.0),
            }

        # Create snapshot
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            token_index=self.token_count,
            text=text,
            position=position,
            velocity=velocity,
            velocity_magnitude=velocity_magnitude,
            acceleration=acceleration,
            acceleration_magnitude=acceleration_magnitude,
            curvature=curvature,
            dominant_dimensions=dominant_dimensions,
            dimension_scores=dimension_scores,
            ethical_valence=ethical_valence,
            ethical_dimensions=ethical_dimensions,
            flow_metrics=flow_metrics,
        )

        # Store snapshot if recording enabled
        if self.observer_config.enable_recording:
            self.snapshots.append(snapshot)

            # Trim if exceeding max length
            if len(self.snapshots) > self.observer_config.max_trajectory_length:
                self.snapshots = self.snapshots[-self.observer_config.max_trajectory_length:]

        # Update previous state
        self.prev_position = position
        self.prev_velocity = velocity

        return snapshot

    def get_trajectory(self) -> List[StateSnapshot]:
        """
        Get full recorded trajectory.

        Returns:
            List of StateSnapshot objects in chronological order
        """
        return self.snapshots.copy()

    def get_current_state(self) -> Optional[StateSnapshot]:
        """
        Get most recent state snapshot.

        Returns:
            Latest StateSnapshot or None if no observations yet
        """
        return self.snapshots[-1] if self.snapshots else None

    def reset(self):
        """Reset observer state."""
        self.token_count = 0
        self.full_text = ""
        self.snapshots = []
        self.prev_position = None
        self.prev_velocity = None
        logger.info("Observer state reset")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get trajectory statistics.

        Returns:
            Dictionary with trajectory metrics
        """
        if not self.snapshots:
            return {
                "token_count": 0,
                "trajectory_length": 0,
                "avg_velocity": 0.0,
                "max_velocity": 0.0,
                "avg_curvature": 0.0,
                "max_curvature": 0.0,
            }

        velocities = [
            s.velocity_magnitude for s in self.snapshots
            if s.velocity_magnitude > 0
        ]
        curvatures = [
            s.curvature for s in self.snapshots
            if s.curvature is not None
        ]

        return {
            "token_count": self.token_count,
            "trajectory_length": len(self.snapshots),
            "avg_velocity": sum(velocities) / len(velocities) if velocities else 0.0,
            "max_velocity": max(velocities) if velocities else 0.0,
            "avg_curvature": sum(curvatures) / len(curvatures) if curvatures else 0.0,
            "max_curvature": max(curvatures) if curvatures else 0.0,
            "text_length": len(self.full_text),
        }
