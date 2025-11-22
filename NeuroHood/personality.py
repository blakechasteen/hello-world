"""
Personality System: 228D Semantic Space Mapping

Maps resident personalities onto HoloLoom's 228D semantic space with 16
interpretable axes. Tracks personality drift over time as residents experience
events and relationships.

Core Dimensions (16 axes from HoloLoom):

Affective (4):
- Warmth (loving ↔ cold)
- Valence (positive ↔ negative)
- Arousal (excited ↔ calm)
- Intensity (extreme ↔ mild)

Social/Interpersonal (4):
- Formality (professional ↔ casual)
- Directness (explicit ↔ indirect)
- Power (dominant ↔ submissive)
- Generosity (giving ↔ selfish)

Cognitive (4):
- Certainty (confident ↔ uncertain)
- Complexity (sophisticated ↔ simple)
- Concreteness (tangible ↔ abstract)
- Familiarity (known ↔ novel)

Temporal/Dynamic (4):
- Agency (active ↔ passive)
- Stability (constant ↔ volatile)
- Urgency (immediate ↔ patient)
- Completion (finished ↔ nascent)

Extended Space:
- Additional 212 dimensions for nuanced semantic representation
- Narrative dimensions for storytelling (NARRATIVE_DIMENSIONS)
- Total 228D space for complete personality modeling
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

# HoloLoom semantic calculus
from HoloLoom.semantic_calculus import (
    SemanticSpectrum,
    STANDARD_DIMENSIONS,
    SemanticDimension
)


@dataclass
class PersonalitySnapshot:
    """
    A snapshot of a resident's personality at a point in time.

    Attributes:
        timestamp: When this snapshot was taken
        projections: Dict mapping dimension name -> value (-1.0 to 1.0)
        description: Optional text description that generated this
        event_context: Optional event that triggered this update
    """
    timestamp: float
    projections: Dict[str, float]
    description: Optional[str] = None
    event_context: Optional[str] = None

    def to_vector(self, dimension_names: List[str]) -> np.ndarray:
        """Convert to ordered vector following dimension_names order."""
        return np.array([self.projections.get(name, 0.0) for name in dimension_names])

    def dominant_traits(self, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Get traits with absolute value > threshold, sorted by strength."""
        strong_traits = [
            (name, val) for name, val in self.projections.items()
            if abs(val) > threshold
        ]
        return sorted(strong_traits, key=lambda x: abs(x[1]), reverse=True)


@dataclass
class PersonalityDrift:
    """
    Tracks how personality changes over time.

    Attributes:
        dimension: Name of dimension
        baseline: Initial value
        current: Current value
        delta: Total change (current - baseline)
        velocity: Rate of change per timestep
        history: List of (timestamp, value) tuples
    """
    dimension: str
    baseline: float
    current: float
    delta: float
    velocity: float = 0.0
    history: List[Tuple[float, float]] = field(default_factory=list)

    def add_observation(self, timestamp: float, value: float):
        """Add new observation and update velocity."""
        self.history.append((timestamp, value))

        # Update delta
        old_current = self.current
        self.current = value
        self.delta = value - self.baseline

        # Update velocity (simple finite difference)
        if len(self.history) > 1:
            prev_time, prev_val = self.history[-2]
            dt = timestamp - prev_time
            if dt > 0:
                self.velocity = (value - prev_val) / dt

    def is_significant(self, threshold: float = 0.2) -> bool:
        """Check if drift is significant."""
        return abs(self.delta) > threshold


class PersonalitySystem:
    """
    Manages personality representation and drift tracking.

    Uses HoloLoom's SemanticSpectrum to project personality descriptions
    onto 16 interpretable axes. Tracks changes over time.
    """

    def __init__(self, embed_fn):
        """
        Initialize personality system.

        Args:
            embed_fn: Function text -> embedding vector (from HoloLoom)
        """
        self.embed_fn = embed_fn

        # Create semantic spectrum with 16 standard dimensions
        self.spectrum = SemanticSpectrum(dimensions=STANDARD_DIMENSIONS)
        self.spectrum.learn_axes(embed_fn)

        # Dimension names (ordered)
        self.dimension_names = [dim.name for dim in self.spectrum.dimensions]

        # Per-resident data
        self.baselines: Dict[str, PersonalitySnapshot] = {}  # {resident_id: snapshot}
        self.current: Dict[str, PersonalitySnapshot] = {}    # {resident_id: snapshot}
        self.drift: Dict[str, Dict[str, PersonalityDrift]] = {}  # {resident_id: {dim: drift}}

    def initialize_personality(
        self,
        resident_id: str,
        description: str,
        timestamp: Optional[float] = None
    ) -> PersonalitySnapshot:
        """
        Initialize personality from text description.

        Args:
            resident_id: Unique resident identifier
            description: Text description of personality (e.g., "warm, extroverted, creative")
            timestamp: Optional timestamp (defaults to now)

        Returns:
            PersonalitySnapshot with projections onto 16 dimensions
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        # Embed description
        embedding = self.embed_fn(description)

        # Project onto 16 dimensions
        projections = {}
        for dim in self.spectrum.dimensions:
            value = dim.project(embedding)
            projections[dim.name] = float(value)

        # Create snapshot
        snapshot = PersonalitySnapshot(
            timestamp=timestamp,
            projections=projections,
            description=description
        )

        # Store as baseline and current
        self.baselines[resident_id] = snapshot
        self.current[resident_id] = snapshot

        # Initialize drift trackers
        self.drift[resident_id] = {}
        for dim_name, value in projections.items():
            self.drift[resident_id][dim_name] = PersonalityDrift(
                dimension=dim_name,
                baseline=value,
                current=value,
                delta=0.0,
                history=[(timestamp, value)]
            )

        return snapshot

    def update_personality(
        self,
        resident_id: str,
        description: str,
        timestamp: Optional[float] = None,
        event_context: Optional[str] = None
    ) -> Tuple[PersonalitySnapshot, Dict[str, PersonalityDrift]]:
        """
        Update personality from new description/experience.

        Args:
            resident_id: Resident identifier
            description: New personality description or behavior
            timestamp: Optional timestamp (defaults to now)
            event_context: Optional event that triggered update

        Returns:
            Tuple of (new snapshot, significant drifts)
        """
        if resident_id not in self.baselines:
            raise ValueError(f"Resident {resident_id} not initialized")

        if timestamp is None:
            timestamp = datetime.now().timestamp()

        # Embed new description
        embedding = self.embed_fn(description)

        # Project onto dimensions
        projections = {}
        for dim in self.spectrum.dimensions:
            value = dim.project(embedding)
            projections[dim.name] = float(value)

        # Create new snapshot
        snapshot = PersonalitySnapshot(
            timestamp=timestamp,
            projections=projections,
            description=description,
            event_context=event_context
        )

        # Update current
        self.current[resident_id] = snapshot

        # Update drift trackers
        significant_drifts = {}
        for dim_name, value in projections.items():
            drift = self.drift[resident_id][dim_name]
            drift.add_observation(timestamp, value)

            if drift.is_significant():
                significant_drifts[dim_name] = drift

        return snapshot, significant_drifts

    def get_personality_description(
        self,
        resident_id: str,
        use_current: bool = True
    ) -> str:
        """
        Generate human-readable personality description.

        Args:
            resident_id: Resident identifier
            use_current: If True, use current personality; else baseline

        Returns:
            Formatted personality description
        """
        snapshot = self.current[resident_id] if use_current else self.baselines[resident_id]

        # Get dominant traits
        dominant = snapshot.dominant_traits(threshold=0.4)

        if not dominant:
            return "Balanced personality with no strong traits"

        # Format traits
        trait_descriptions = []
        for name, value in dominant:
            # Get dimension for interpretable poles
            dim = next(d for d in self.spectrum.dimensions if d.name == name)

            if value > 0:
                # Positive pole
                pole = dim.positive_exemplars[0]
                strength = "very" if abs(value) > 0.7 else "somewhat"
            else:
                # Negative pole
                pole = dim.negative_exemplars[0]
                strength = "very" if abs(value) > 0.7 else "somewhat"

            trait_descriptions.append(f"{strength} {pole}")

        return ", ".join(trait_descriptions)

    def get_drift_summary(self, resident_id: str) -> Dict[str, Any]:
        """
        Get summary of personality drift.

        Args:
            resident_id: Resident identifier

        Returns:
            Dictionary with drift statistics
        """
        if resident_id not in self.drift:
            return {}

        drifts = self.drift[resident_id]

        # Compute statistics
        significant_drifts = {
            name: drift for name, drift in drifts.items()
            if drift.is_significant()
        }

        total_delta = sum(abs(drift.delta) for drift in drifts.values())
        max_drift = max(drifts.items(), key=lambda x: abs(x[1].delta))

        # Fastest changing (by velocity)
        fastest = max(drifts.items(), key=lambda x: abs(x[1].velocity))

        return {
            'total_drift': total_delta,
            'significant_count': len(significant_drifts),
            'significant_dimensions': list(significant_drifts.keys()),
            'max_drift_dimension': max_drift[0],
            'max_drift_value': max_drift[1].delta,
            'fastest_changing_dimension': fastest[0],
            'fastest_changing_velocity': fastest[1].velocity,
        }

    def visualize_drift(
        self,
        resident_id: str,
        dimension: Optional[str] = None
    ) -> str:
        """
        Create simple ASCII visualization of drift.

        Args:
            resident_id: Resident identifier
            dimension: Optional dimension to focus on (else show all)

        Returns:
            ASCII art visualization
        """
        if resident_id not in self.drift:
            return "No drift data available"

        drifts = self.drift[resident_id]

        # Filter to single dimension if requested
        if dimension:
            if dimension not in drifts:
                return f"Dimension '{dimension}' not found"
            drifts = {dimension: drifts[dimension]}

        lines = []
        lines.append(f"Personality Drift for Resident {resident_id}")
        lines.append("=" * 60)

        for dim_name, drift in drifts.items():
            # Create bar chart
            bar_length = 40
            baseline_pos = int((drift.baseline + 1.0) * bar_length / 2)
            current_pos = int((drift.current + 1.0) * bar_length / 2)

            bar = ['.'] * bar_length
            bar[baseline_pos] = 'B'  # Baseline
            bar[current_pos] = 'C'   # Current

            bar_str = ''.join(bar)

            # Arrow indicating direction (ASCII-safe)
            if drift.delta > 0.1:
                arrow = ">"
            elif drift.delta < -0.1:
                arrow = "<"
            else:
                arrow = "-"

            lines.append(f"{dim_name:15s} [{bar_str}] {arrow} D{drift.delta:+.3f}")

        lines.append("")
        lines.append("B = Baseline, C = Current")

        return "\n".join(lines)

    def get_personality_vector(
        self,
        resident_id: str,
        use_current: bool = True
    ) -> np.ndarray:
        """
        Get personality as 16D vector.

        Args:
            resident_id: Resident identifier
            use_current: If True, use current; else baseline

        Returns:
            16D numpy array
        """
        snapshot = self.current[resident_id] if use_current else self.baselines[resident_id]
        return snapshot.to_vector(self.dimension_names)

    def compute_personality_distance(
        self,
        resident_a: str,
        resident_b: str,
        use_current: bool = True
    ) -> float:
        """
        Compute Euclidean distance between two residents' personalities.

        Args:
            resident_a: First resident
            resident_b: Second resident
            use_current: If True, use current personalities

        Returns:
            Distance in 16D space (0.0 to ~8.0)
        """
        vec_a = self.get_personality_vector(resident_a, use_current)
        vec_b = self.get_personality_vector(resident_b, use_current)

        return float(np.linalg.norm(vec_a - vec_b))

    def find_compatible_residents(
        self,
        resident_id: str,
        all_resident_ids: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most personality-compatible residents.

        Args:
            resident_id: Target resident
            all_resident_ids: List of all resident IDs
            top_k: Number of matches to return

        Returns:
            List of (resident_id, similarity) tuples, sorted by similarity
        """
        target_vec = self.get_personality_vector(resident_id)

        similarities = []
        for other_id in all_resident_ids:
            if other_id == resident_id or other_id not in self.current:
                continue

            other_vec = self.get_personality_vector(other_id)

            # Cosine similarity
            similarity = float(
                np.dot(target_vec, other_vec) /
                (np.linalg.norm(target_vec) * np.linalg.norm(other_vec) + 1e-10)
            )

            similarities.append((other_id, similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


def demonstrate_personality_system():
    """Demo showing personality initialization and drift tracking."""
    from HoloLoom.embedding.spectral import create_embedder

    print("\n" + "="*70)
    print("  NeuroHood Personality System Demo")
    print("  16D Semantic Space + Drift Tracking")
    print("="*70 + "\n")

    # Create embedder
    print("[ Creating embedder... ]")
    embedder = create_embedder(sizes=[384])
    embed_fn = lambda text: embedder.encode([text])[0]

    # Create personality system
    print("[ Initializing personality system... ]\n")
    personality = PersonalitySystem(embed_fn)

    # Initialize Alice
    print("[ Initializing Alice: warm, introverted, creative ]")
    alice_baseline = personality.initialize_personality(
        resident_id="alice",
        description=(
            "Alice is a warm, caring, and loving person who deeply values peace, quiet, and tranquility. "
            "She is introverted, calm, and reflective, preferring gentle, subtle interactions. "
            "She is creative, abstract, and complex in her thinking, with a patient and relaxed demeanor. "
            "She values harmony and prefers indirect, compassionate communication."
        )
    )
    print(f"  Baseline: {personality.get_personality_description('alice', use_current=False)}")
    print()

    # Initialize Bob
    print("[ Initializing Bob: extroverted, loud, party-loving ]")
    bob_baseline = personality.initialize_personality(
        resident_id="bob",
        description=(
            "Bob is an excited, energetic, and intense person who loves being active and doing things. "
            "He is extroverted, loud, and thrives on arousal and excitement. "
            "He is direct, explicit, and straightforward in communication, preferring casual and informal interactions. "
            "He values immediate action and urgency, always seeking novel and thrilling experiences."
        )
    )
    print(f"  Baseline: {personality.get_personality_description('bob', use_current=False)}")
    print()

    # Simulate Alice experiencing stress from noise
    print("[ Alice experiences noise stress from Bob's parties... ]")
    alice_update, alice_drifts = personality.update_personality(
        resident_id="alice",
        description=(
            "Alice has become cold, harsh, and cruel in her feelings toward Bob. "
            "She is now negative, sad, and miserable, feeling extremely stressed and intense irritation. "
            "She has become more powerful and dominant in her demands for quiet, "
            "less patient and more urgent in her need for peace. "
            "Her mood is volatile and unstable, no longer calm."
        ),
        event_context="Noise complaint against Bob"
    )

    print(f"  Updated: {personality.get_personality_description('alice')}")
    print(f"  Significant drifts: {len(alice_drifts)}")
    for dim_name, drift in alice_drifts.items():
        print(f"    {dim_name}: {drift.baseline:.3f} -> {drift.current:.3f} (D{drift.delta:+.3f})")
    print()

    # Show drift visualization
    print("[ Alice's Personality Drift ]")
    print(personality.visualize_drift("alice"))
    print()

    # Compute compatibility
    print("[ Personality Compatibility ]")
    distance = personality.compute_personality_distance("alice", "bob")
    print(f"  Alice <-> Bob distance: {distance:.3f}")
    print(f"  (Lower = more compatible, ~0-8 range)")
    print()

    # Show drift summary
    print("[ Drift Summary for Alice ]")
    summary = personality.get_drift_summary("alice")
    print(f"  Total drift: {summary['total_drift']:.3f}")
    print(f"  Significant dimensions: {summary['significant_count']}")
    print(f"  Max drift: {summary['max_drift_dimension']} (D{summary['max_drift_value']:+.3f})")
    print(f"  Fastest changing: {summary['fastest_changing_dimension']} (v={summary['fastest_changing_velocity']:+.4f}/sec)")
    print()

    print("="*70)
    print("  Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    demonstrate_personality_system()
